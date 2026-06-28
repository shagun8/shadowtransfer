"""
Illumination-Invariant Module (IIM) from YOLA (NeurIPS 2024).

Extracts illumination-invariant features using learnable zero-mean kernels
applied to cross-color-channel log-domain differences.  The features are
fused with the original image via a lightweight FuseConv block, producing a
3-channel output that replaces the raw RGB input to any downstream backbone.

Reference
---------
Hong et al., "You Only Look Around: Learning Illumination Invariant Feature
for Low-light Object Detection", NeurIPS 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IIM(nn.Module):
    """
    Illumination-Invariant Module.

    Pipeline (per Eq. 8 & Fig. 2):
        1. Denormalize input → clamp → log(R), log(G), log(B)
        2. For each learnable kernel W_i (zero-mean enforced):
              feat_rb = W_i ⊛ log(R) + (−W_i) ⊛ log(B)
              feat_rg = W_i ⊛ log(R) + (−W_i) ⊛ log(G)
              feat_gb = W_i ⊛ log(G) + (−W_i) ⊛ log(B)
           → 3 × num_kernels channels
        3. FuseConv: merge IIM features with original image → 3 channels

    Parameters
    ----------
    num_kernels : int   Number of learnable kernels (default 8 → 24-d features).
    kernel_size : int   Spatial size of each kernel (default 5).
    mean, std   : tuple ImageNet normalisation constants used by the dataloader.
    """

    def __init__(self, num_kernels=8, kernel_size=5,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super().__init__()

        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.num_pairs = 3            # (R,B), (R,G), (G,B)
        self.feat_dim = num_kernels * self.num_pairs  # 24 for 8 kernels

        # Normalisation buffers (moved with .to(device) automatically)
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

        # Learnable kernels — shape [num_kernels, 1, k, k]
        # Kaiming initialization gives proper gradient scale, then we hard-
        # project to zero-mean so features are illumination-invariant from
        # the very first forward pass.
        _k = torch.empty(num_kernels, 1, kernel_size, kernel_size)
        nn.init.kaiming_normal_(_k, mode='fan_in', nonlinearity='linear')
        _k -= _k.mean(dim=(-2, -1), keepdim=True)          # zero-mean
        self.kernels = nn.Parameter(_k)

        # ----- FuseConv (Fig. 2) -----
        # Path 1: 3×3 Conv on IIM features
        self.iim_conv = nn.Sequential(
            nn.Conv2d(self.feat_dim, 24, 3, padding=1, bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        # Path 2: 3×3 Conv on original (normalised) image
        self.img_conv = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1, bias=True),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        # Merge: concat(24+24) → 32 → 3
        self.fuse = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1, bias=True),
        )

    # ------------------------------------------------------------------
    # Zero-mean helpers
    # ------------------------------------------------------------------

    def _get_zero_mean_kernels(self):
        """Return kernels with zero-mean constraint applied on-the-fly."""
        return self.kernels - self.kernels.mean(dim=(-2, -1), keepdim=True)

    @torch.no_grad()
    def enforce_zero_mean(self):
        """Hard project kernels to zero-mean (call after optimizer.step())."""
        self.kernels.data -= self.kernels.data.mean(dim=(-2, -1), keepdim=True)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _denormalize(self, x):
        """ImageNet-normalised → [0, 1]."""
        return x * self.std + self.mean

    def extract_features(self, x):
        """
        Extract raw illumination-invariant features (before FuseConv).

        Args:
            x : [B, 3, H, W]  ImageNet-normalised RGB tensor.
        Returns:
            features : [B, feat_dim, H, W]
        """
        rgb = self._denormalize(x)
        rgb = torch.clamp(rgb, min=1e-6)
        log_r = torch.log(rgb[:, 0:1])
        log_g = torch.log(rgb[:, 1:2])
        log_b = torch.log(rgb[:, 2:3])

        W = self._get_zero_mean_kernels()       # [K, 1, k, k]
        neg_W = -W
        pad = self.kernel_size // 2

        # Cross-colour-channel differences in log domain (Eq. 8)
        feat_rb = F.conv2d(log_r, W, padding=pad) + F.conv2d(log_b, neg_W, padding=pad)
        feat_rg = F.conv2d(log_r, W, padding=pad) + F.conv2d(log_g, neg_W, padding=pad)
        feat_gb = F.conv2d(log_g, W, padding=pad) + F.conv2d(log_b, neg_W, padding=pad)

        return torch.cat([feat_rb, feat_rg, feat_gb], dim=1)  # [B, K*3, H, W]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x : [B, 3, H, W]  ImageNet-normalised RGB.
        Returns:
            fused    : [B, 3, H, W]  illumination-invariant representation.
            features : [B, feat_dim, H, W]  raw IIM features (for II-loss).
        """
        features = self.extract_features(x)

        p1 = self.iim_conv(features)              # [B, 24, H, W]
        p2 = self.img_conv(x)                     # [B, 24, H, W]
        fused = self.fuse(torch.cat([p1, p2], dim=1))  # [B, 3, H, W]

        return fused, features


# ----------------------------------------------------------------------
# II Loss helper
# ----------------------------------------------------------------------

def compute_ii_loss(features_orig, iim_module, rgb_norm,
                    gamma_range=(0.5, 2.0), beta=1.0):
    """
    Illumination-Invariant consistency loss (Eq. 13).

    Compares IIM features from the original image with those from a
    gamma-transformed version.  Uses Smooth-L1 (Huber) loss.

    Args:
        features_orig : [B, D, H, W]  IIM features already computed.
        iim_module    : IIM            the module (for extract_features & buffers).
        rgb_norm      : [B, 3, H, W]  original normalised RGB image.
        gamma_range   : tuple          uniform range for random gamma.
        beta          : float          Smooth-L1 threshold (default 1.0).

    Returns:
        loss : scalar tensor.
    """
    # Gamma transform in pixel space
    rgb_raw = iim_module._denormalize(rgb_norm)
    rgb_raw = torch.clamp(rgb_raw, min=1e-6)

    gamma = torch.empty(1, device=rgb_norm.device).uniform_(*gamma_range).item()
    rgb_gamma = rgb_raw.pow(gamma)

    # Re-normalise
    rgb_gamma_norm = (rgb_gamma - iim_module.mean) / iim_module.std

    features_gamma = iim_module.extract_features(rgb_gamma_norm)

    return F.smooth_l1_loss(features_orig, features_gamma, beta=beta)


# ----------------------------------------------------------------------
# Quick test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    iim = IIM(num_kernels=8, kernel_size=5)

    x = torch.randn(2, 3, 256, 256)
    fused, feats = iim(x)

    print(f"Input:    {x.shape}")
    print(f"Fused:    {fused.shape}")
    print(f"Features: {feats.shape}")

    total = sum(p.numel() for p in iim.parameters())
    print(f"\nIIM parameters: {total:,}  ({total/1e6:.3f} M)")

    # Test zero-mean enforcement
    iim.enforce_zero_mean()
    km = iim.kernels.mean(dim=(-2, -1))
    print(f"Kernel means after enforcement (should be ~0): {km.abs().max().item():.1e}")

    # Test II loss
    loss = compute_ii_loss(feats, iim, x)
    print(f"II loss: {loss.item():.6f}")