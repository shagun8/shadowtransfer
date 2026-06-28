"""
MRFP Modules for Domain-Invariant Feature Perturbation
=======================================================
Reference: Udupa et al., "MRFP: Learning Generalizable Semantic Segmentation
from Sim-2-Real with Multi-Resolution Feature Perturbation", CVPR 2024.

Two modules:
    HRFP  — High-Resolution Feature Perturbation (overcomplete autoencoder,
             randomly re-initialized every forward pass, frozen).
    NP+   — Normalized Perturbation (channel-statistic style randomization,
             no learnable parameters).

Both are training-time only; at eval they are identity / not called.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# HRFP — High-Resolution Feature Perturbation
# ======================================================================

class HRFPModule(nn.Module):
    """
    Overcomplete autoencoder with *decreasing* receptive field.

    Architecture (paper §3.3, Fig. 2):
        Encoder — 4 × [Conv3×3 → BN → ReLU] with ~1.2× bilinear up each step
                  (max spatial resolution = 2× input)
        Decoder — 4 × [Conv3×3 → BN → ReLU] with progressive downsample
                  back to original spatial size

    Weight policy (following RandConv / ProRandConv convention):
        • All weights are **re-initialized** every forward call so
          perturbations are truly random at each training step.
        • Conv weights: He (Kaiming-normal) init.
        • BN γ, β: sampled from N(0, bn_std²).
        • BN running stats disabled (track_running_stats=False).
        • All parameters frozen (requires_grad = False).

    Returns:
        perturbation        — [B, C, H, W]  same size as input  (O₁ branch)
        max_resolution_feat — [B, C, ~2H, ~2W]                  (O₂ branch for HRFP+)
    """

    def __init__(self, in_channels: int = 64, bn_std: float = 0.5,
                 scale_factor: float = 1.2, num_layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.bn_std = bn_std
        self.scale_factor = scale_factor
        self.num_layers = num_layers

        # ---- encoder (overcomplete: increasing spatial resolution) ----
        self.enc_convs = nn.ModuleList()
        self.enc_bns = nn.ModuleList()
        for _ in range(num_layers):
            self.enc_convs.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            self.enc_bns.append(
                nn.BatchNorm2d(in_channels, track_running_stats=False))

        # ---- decoder (reduce back to original resolution) ----
        self.dec_convs = nn.ModuleList()
        self.dec_bns = nn.ModuleList()
        for _ in range(num_layers):
            self.dec_convs.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            self.dec_bns.append(
                nn.BatchNorm2d(in_channels, track_running_stats=False))

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------

    @torch.no_grad()
    def _reinit_weights(self):
        """Re-sample all random weights (called once per forward pass)."""
        for conv in self.enc_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out',
                                    nonlinearity='relu')
        for conv in self.dec_convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out',
                                    nonlinearity='relu')
        for bn in list(self.enc_bns) + list(self.dec_bns):
            nn.init.normal_(bn.weight, mean=0.0, std=self.bn_std)
            nn.init.normal_(bn.bias, mean=0.0, std=self.bn_std)

    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] — stage-0 encoder features

        Returns:
            dict  {'perturbation': Tensor, 'max_resolution_feat': Tensor}
        """
        # Re-randomize every call (RandConv convention)
        self._reinit_weights()

        H, W = x.shape[2], x.shape[3]
        max_H, max_W = H * 2, W * 2

        # ---- Encoder: progressively upsample ----
        feat = x
        for i in range(self.num_layers):
            new_H = min(int(H * (self.scale_factor ** (i + 1))), max_H)
            new_W = min(int(W * (self.scale_factor ** (i + 1))), max_W)
            feat = F.interpolate(feat, size=(new_H, new_W),
                                 mode='bilinear', align_corners=False)
            feat = F.relu(self.enc_bns[i](self.enc_convs[i](feat)),
                          inplace=False)

        max_res_feat = feat              # ~2H×2W  →  O₂

        # ---- Decoder: progressively downsample ----
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                t = (i + 1) / self.num_layers
                new_H = int(max_res_feat.shape[2] * (1 - t) + H * t)
                new_W = int(max_res_feat.shape[3] * (1 - t) + W * t)
            else:
                new_H, new_W = H, W      # guarantee exact match
            feat = F.interpolate(feat, size=(new_H, new_W),
                                 mode='bilinear', align_corners=False)
            feat = F.relu(self.dec_bns[i](self.dec_convs[i](feat)),
                          inplace=False)

        return {
            'perturbation':        feat,          # O₁  [B, C, H, W]
            'max_resolution_feat': max_res_feat,  # O₂  [B, C, ~2H, ~2W]
        }


# ======================================================================
# NP+ — Normalized Perturbation
# ======================================================================

class NormalizedPerturbation(nn.Module):
    """
    Normalized Perturbation (NP+) — channel-statistic style randomization.

    Paper Eq. 4-6:
        y  = α·x + δ·(β − α)·μ_c
    where
        μ_c   = per-sample channel mean  [B, C, 1, 1]
        Δ     = cross-batch variance of μ_c  [1, C, 1, 1]
        δ     = Δ / max(Δ)   (normalized ∈ [0,1])
        α, β  ~ N(0, 1)      [B, C, 1, 1]

    No learnable parameters.  Training-time only.
    """

    def forward(self, x):
        if not self.training:
            return x

        B, C, _, _ = x.shape

        # Per-sample channel mean
        mu_c = x.mean(dim=(2, 3), keepdim=True)            # [B, C, 1, 1]

        # Cross-batch variance of channel means
        mu_batch_mean = mu_c.mean(dim=0, keepdim=True)     # [1, C, 1, 1]
        delta = ((mu_c - mu_batch_mean) ** 2).mean(dim=0, keepdim=True)
        delta_norm = delta / (delta.max() + 1e-10)          # [1, C, 1, 1]

        # Random coefficients
        alpha = torch.randn(B, C, 1, 1, device=x.device, dtype=x.dtype)
        beta  = torch.randn(B, C, 1, 1, device=x.device, dtype=x.dtype)

        # Eq. 6
        y = alpha * x + delta_norm * (beta - alpha) * mu_c
        return y


# ======================================================================
# Quick self-test
# ======================================================================
if __name__ == "__main__":
    print("Testing HRFP …")
    hrfp = HRFPModule(in_channels=64, bn_std=0.5)
    hrfp.train()
    x = torch.randn(2, 64, 384, 384)
    out = hrfp(x)
    print(f"  perturbation       : {out['perturbation'].shape}")
    print(f"  max_resolution_feat: {out['max_resolution_feat'].shape}")

    print("\nTesting NP+ …")
    np_plus = NormalizedPerturbation()
    np_plus.train()
    feat = torch.randn(4, 512, 48, 48)
    perturbed = np_plus(feat)
    print(f"  output: {perturbed.shape}")

    # Verify no learnable params
    hrfp_params = sum(p.numel() for p in hrfp.parameters() if p.requires_grad)
    np_params   = sum(p.numel() for p in np_plus.parameters() if p.requires_grad)
    print(f"\nTrainable params — HRFP: {hrfp_params}  NP+: {np_params}")