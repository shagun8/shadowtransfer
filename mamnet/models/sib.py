"""
Spectral Information Bottleneck (SIB) module for MAMNet.

Components (all independently toggleable):
  - Haar wavelet decomposition: splits features into LL (content/domain),
    LH/HL (edges/task), HH (noise)
  - Differential VIB: aggressive compression on LL (β_content), gentle on
    LH/HL (β_edge)
  - Intensity-adaptive β: stronger compression on bright surfaces
  - Content augmentation: stochastic perturbation of LL only during training
  - Skip Attention Gates (SAG): lightweight channel-wise sigmoid gating on
    skip connections
  - Multi-scale SIB: optional lightweight SIB at each encoder stage
  - Passthrough Gate: learned scalar gate that blends VIB output with
    original features, allowing VIB to auto-disable when compression
    is not beneficial (e.g. when domain gap is small)
  - Module Bypass Gate: learned scalar gate that wraps the ENTIRE SIB
    pipeline in a residual, allowing the module to be completely bypassed
    when intervention is not needed

NEW — Diagnostic-motivated additions (§4.3 orphan coverage):
  - CACR support: SIB can return pre-augmentation bottleneck features
    for class-asymmetric confidence regularization
  - TENT utilities: helper functions for test-time entropy minimization
    on BN/LN affine parameters

Ablation flags (all default False — enable for specific ablation experiments):
  - symmetric_vib:      A3 — Apply β_content (instead of β_edge) to LH/HL
  - aug_all_subbands:   A5 — Apply content augmentation to LH/HL/HH too
  - no_edge_vib:        A9 — Skip VIB on LH/HL (pass through unchanged)
  - vib_wrong_subband:  A10 — Apply content VIB to HL only (LL passes through)

Placement in MAMNet:
  After MSCAF bottleneck [B, 512, H/16, W/16], before CCA decoder.

Reference:
  Adapted from dinov3/sib.py for MAMNet architecture (512-ch bottleneck,
  4 skip connection levels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================================================================
# Haar Wavelet Transform (2D, per-channel)
# ======================================================================

class HaarWavelet2D(nn.Module):
    """
    2D Haar wavelet decomposition / reconstruction.

    Forward:  [B, C, H, W] → LL, LH, HL, HH each [B, C, H/2, W/2]
    Inverse:  LL, LH, HL, HH → [B, C, H, W]

    No learnable parameters — purely analytical.
    """

    def forward(self, x):
        """Decompose x into 4 sub-bands."""
        # x: [B, C, H, W]  (H, W must be even)
        x_even_h = x[:, :, 0::2, :]   # even rows
        x_odd_h  = x[:, :, 1::2, :]   # odd rows

        # Column-wise
        ll = (x_even_h[:, :, :, 0::2] + x_even_h[:, :, :, 1::2] +
              x_odd_h[:, :, :, 0::2]  + x_odd_h[:, :, :, 1::2]) * 0.25
        lh = (x_even_h[:, :, :, 0::2] + x_even_h[:, :, :, 1::2] -
              x_odd_h[:, :, :, 0::2]  - x_odd_h[:, :, :, 1::2]) * 0.25
        hl = (x_even_h[:, :, :, 0::2] - x_even_h[:, :, :, 1::2] +
              x_odd_h[:, :, :, 0::2]  - x_odd_h[:, :, :, 1::2]) * 0.25
        hh = (x_even_h[:, :, :, 0::2] - x_even_h[:, :, :, 1::2] -
              x_odd_h[:, :, :, 0::2]  + x_odd_h[:, :, :, 1::2]) * 0.25

        return ll, lh, hl, hh

    def inverse(self, ll, lh, hl, hh):
        """Reconstruct from 4 sub-bands."""
        B, C, H2, W2 = ll.shape
        H, W = H2 * 2, W2 * 2

        out = torch.zeros(B, C, H, W, device=ll.device, dtype=ll.dtype)

        out[:, :, 0::2, 0::2] = ll + lh + hl + hh
        out[:, :, 0::2, 1::2] = ll + lh - hl - hh
        out[:, :, 1::2, 0::2] = ll - lh + hl - hh
        out[:, :, 1::2, 1::2] = ll - lh - hl + hh

        return out


# ======================================================================
# Variational Information Bottleneck (VIB)
# ======================================================================

class VIB(nn.Module):
    """
    Variational Information Bottleneck.

    Learns μ and log(σ2) per channel, samples z = μ + σ·ε during training.
    KL divergence = 0.5 * Σ (μ2 + σ2 - log(σ2) - 1).

    Args:
        channels: Number of input/output channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.mu_layer = nn.Conv2d(channels, channels, 1)
        self.logvar_layer = nn.Conv2d(channels, channels, 1)
        # Initialize logvar to small negative → σ ≈ 1 initially
        nn.init.constant_(self.logvar_layer.bias, -2.0)

    def forward(self, x):
        """
        Returns:
            z: Sampled (training) or mean (eval) features.
            kl: KL divergence scalar.
        """
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu

        # KL divergence: 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        kl = kl.sum(dim=(1, 2, 3)).mean()  # average over batch

        return z, kl

    def forward_mean(self, x):
        """
        Deterministic forward (mean only, no sampling).
        Used for CACR reference path.

        Returns:
            mu: Mean features (no stochastic sampling).
            kl: KL divergence scalar (same formula, just not used for backprop).
        """
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
        kl = kl.sum(dim=(1, 2, 3)).mean()
        return mu, kl


# ======================================================================
# Passthrough Gate (VIB-level — gates VIB output only)
# ======================================================================

class PassthroughGate(nn.Module):
    """
    Learned scalar gate that blends VIB output with original features.

        output = g * vib_output + (1 - g) * original

    where g = sigmoid(linear(GAP(original))).

    Initialized with bias = -2.0 → sigmoid(-2) ≈ 0.12, so the module
    starts in near-passthrough mode.  The model must learn evidence that
    VIB compression is beneficial before the gate opens.

    Args:
        channels: Number of feature channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, -2.0)

    def forward(self, vib_out, original):
        """
        Args:
            vib_out:  VIB-compressed features  [B, C, H, W].
            original: Pre-VIB features         [B, C, H, W].

        Returns:
            Blended features [B, C, H, W].
        """
        gap = original.mean(dim=[2, 3])              # [B, C]
        g = torch.sigmoid(self.fc(gap))               # [B, 1]
        g = g.unsqueeze(-1).unsqueeze(-1)             # [B, 1, 1, 1]
        return g * vib_out + (1.0 - g) * original


# ======================================================================
# Module Bypass Gate (wraps ENTIRE SIB pipeline)
# ======================================================================

class ModuleBypassGate(nn.Module):
    """
    Learned scalar gate that wraps the entire SIB module in a residual.

        F_out = α · F_sib + (1 − α) · F_encoder

    where α = sigmoid(linear(GAP(F_encoder))).

    When α → 0, the SIB module is completely bypassed (decoder sees raw
    encoder features, equivalent to vanilla).  When α → 1, full SIB is
    applied.  The gate learns from the segmentation loss which regime
    is appropriate for each input.

    Initialized with bias = +2.0 → sigmoid(2) ≈ 0.88, meaning SIB is
    "on" by default and the gate must actively learn to turn off.

    Args:
        channels: Number of feature channels (512 for MAMNet bottleneck).
    """

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 2.0)

    def forward(self, sib_out, original):
        """
        Args:
            sib_out:  Features after full SIB processing  [B, C, H, W].
            original: Features before SIB (encoder output) [B, C, H, W].

        Returns:
            blended: Gated features [B, C, H, W].
            alpha:   Per-sample gate values [B] for diagnostic logging.
        """
        gap = original.mean(dim=[2, 3])                # [B, C]
        alpha = torch.sigmoid(self.fc(gap))            # [B, 1]
        alpha_spatial = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
        blended = alpha_spatial * sib_out + (1.0 - alpha_spatial) * original
        return blended, alpha.squeeze(-1)              # [B, C, H, W], [B]


# ======================================================================
# Content Augmentation
# ======================================================================

class ContentAugmentation(nn.Module):
    """
    Stochastic perturbation of LL (content/domain) sub-band during training.
    Adds Gaussian noise scaled by a learnable factor, only to LL.
    Encourages invariance to domain-specific low-frequency information.

    Args:
        channels: Number of channels in LL sub-band.
        noise_scale: Initial noise standard deviation.
    """

    def __init__(self, channels, noise_scale=0.1):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.tensor(noise_scale))

    def forward(self, ll):
        if self.training:
            noise = torch.randn_like(ll) * self.noise_scale.abs()
            return ll + noise
        return ll


# ======================================================================
# Intensity-Adaptive Beta
# ======================================================================

class IntensityAdaptiveBeta(nn.Module):
    """
    Computes per-pixel β scaling based on image intensity.
    Brighter surfaces (where Probe 1 showed largest transfer gap)
    get stronger compression.

    Args:
        base_beta: Base β value.
        max_multiplier: Maximum β multiplier at full brightness.
    """

    def __init__(self, base_beta=1e-3, max_multiplier=3.0):
        super().__init__()
        self.base_beta = base_beta
        self.max_multiplier = max_multiplier

    def forward(self, intensity_map, spatial_size):
        """
        Args:
            intensity_map: [B, 1, H_img, W_img] normalized intensity (0-1).
            spatial_size: (H_feat, W_feat) target spatial size.

        Returns:
            beta_map: [B, 1, H_feat, W_feat] spatially varying β.
        """
        # Downsample intensity map to feature spatial size
        beta_map = F.interpolate(intensity_map, size=spatial_size,
                                 mode='bilinear', align_corners=False)
        # Scale: β = base_β * (1 + (max_mult - 1) * intensity)
        beta_map = self.base_beta * (1.0 + (self.max_multiplier - 1.0) * beta_map)
        return beta_map


# ======================================================================
# Skip Attention Gate (SAG)
# ======================================================================

class SkipAttentionGate(nn.Module):
    """
    Lightweight channel-wise sigmoid gate on skip connections.
    Initialized to near-identity (bias=+2.0 → sigmoid ≈ 0.88).

    Args:
        channels: Number of channels in the skip connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
        # Initialize last Linear (index -2, before Sigmoid) to near-identity
        # weight=0 + bias=2.0 → output≈2.0 → sigmoid(2.0)≈0.88
        nn.init.constant_(self.gate[-2].bias, 2.0)
        nn.init.zeros_(self.gate[-2].weight)

    def forward(self, x):
        """
        Args:
            x: Skip connection features [B, C, H, W].
        Returns:
            Gated features [B, C, H, W].
        """
        g = self.gate(x)  # [B, C]
        return x * g.unsqueeze(-1).unsqueeze(-1)


# ======================================================================
# Lightweight Multi-Scale SIB (for encoder stages)
# ======================================================================

class LightweightSIB(nn.Module):
    """
    Simplified SIB for skip connections: Haar → VIB on LL only → inverse Haar.
    No content augmentation or intensity adaptation — just spectral compression.

    Args:
        channels: Number of channels at this encoder stage.
        beta: β weight for VIB KL loss at this stage.
    """

    def __init__(self, channels, beta=1e-4):
        super().__init__()
        self.haar = HaarWavelet2D()
        self.vib_ll = VIB(channels)
        self.beta = beta

    def forward(self, x):
        """
        Returns:
            out: Reconstructed features after spectral filtering.
            kl_loss: Weighted KL divergence.
        """
        # Ensure even spatial dims
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        ll, lh, hl, hh = self.haar(x)
        ll_z, kl = self.vib_ll(ll)
        out = self.haar.inverse(ll_z, lh, hl, hh)

        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return out, kl * self.beta


# ======================================================================
# Main SIB Module (bottleneck placement)
# ======================================================================

class SpectralInformationBottleneck(nn.Module):
    """
    Full SIB module for MAMNet bottleneck.

    Placed after MSCAF [B, 512, H/16, W/16], before CCA decoder.

    All components are independently toggleable for ablation.

    Args:
        in_channels: Input channels (512 for MAMNet MSCAF output).
        embed_dim: Projection dimension (512 for MAMNet, identity-shaped).
        use_haar: Enable Haar wavelet decomposition.
        use_vib: Enable VIB (differential when Haar on, uniform when off).
        use_content_aug: Enable content augmentation on LL.
        adaptive_beta: Enable intensity-adaptive β.
        use_passthrough_gate: Enable learned passthrough gate on VIB output.
        beta_content: β for LL (content) sub-band VIB.
        beta_edge: β for LH/HL (edge) sub-bands VIB.
        noise_scale: Initial noise scale for content augmentation.
        beta_max_multiplier: Max β multiplier for intensity adaptation.
        return_pre_aug: When True and training, also return pre-augmentation
                        bottleneck features for CACR loss computation.

    Ablation-only args (all default False):
        symmetric_vib: A3 — Use β_content for edge VIB (instead of β_edge).
        aug_all_subbands: A5 — Also augment LH/HL/HH (MRFP+ analog).
        no_edge_vib: A9 — Skip VIB on LH/HL (pass through unchanged).
        vib_wrong_subband: A10 — Apply content VIB to HL only (LL passes through).
    """

    def __init__(self, in_channels=512, embed_dim=512,
                 use_haar=True, use_vib=True, use_content_aug=True,
                 adaptive_beta=True, use_passthrough_gate=False,
                 beta_content=1e-3, beta_edge=1e-5,
                 noise_scale=0.1, beta_max_multiplier=3.0,
                 return_pre_aug=False,
                 # Ablation flags
                 symmetric_vib=False,
                 aug_all_subbands=False,
                 no_edge_vib=False,
                 vib_wrong_subband=False):
        super().__init__()

        self.use_haar = use_haar
        self.use_vib = use_vib
        self.use_content_aug = use_content_aug
        self.adaptive_beta = adaptive_beta
        self.use_passthrough_gate = use_passthrough_gate
        self.beta_content = beta_content
        self.beta_edge = beta_edge
        self.return_pre_aug = return_pre_aug

        # Ablation flags
        self.symmetric_vib = symmetric_vib
        self.aug_all_subbands = aug_all_subbands
        self.no_edge_vib = no_edge_vib
        self.vib_wrong_subband = vib_wrong_subband

        # Projection (identity-shaped for MAMNet: 512→512)
        if in_channels != embed_dim:
            self.proj = nn.Conv2d(in_channels, embed_dim, 1)
        else:
            self.proj = nn.Identity()

        self.out_proj = (nn.Conv2d(embed_dim, in_channels, 1)
                         if in_channels != embed_dim else nn.Identity())

        # Haar wavelet
        if use_haar:
            self.haar = HaarWavelet2D()

        # VIB components
        if use_vib:
            if use_haar:
                # Differential VIB: separate for content vs edges
                self.vib_content = VIB(embed_dim)   # LL (or HL when wrong subband)
                self.vib_edge = VIB(embed_dim)       # LH, HL (normal path)
            else:
                # Uniform VIB (no Haar → single VIB on full features)
                self.vib_uniform = VIB(embed_dim)

        # Passthrough gate on content VIB (Haar mode) or uniform VIB
        if use_passthrough_gate and use_vib:
            self.passthrough_gate = PassthroughGate(embed_dim)

        # Content augmentation
        if use_content_aug and use_haar:
            self.content_aug = ContentAugmentation(embed_dim, noise_scale)

        # Intensity-adaptive beta
        if adaptive_beta:
            self.intensity_beta = IntensityAdaptiveBeta(
                base_beta=beta_content,
                max_multiplier=beta_max_multiplier)

        # Log ablation config if any are active
        active_ablations = []
        if symmetric_vib:
            active_ablations.append('symmetric_vib(A3)')
        if aug_all_subbands:
            active_ablations.append('aug_all_subbands(A5)')
        if no_edge_vib:
            active_ablations.append('no_edge_vib(A9)')
        if vib_wrong_subband:
            active_ablations.append('vib_wrong_subband(A10)')
        if active_ablations:
            print(f'  SIB ablation flags active: {", ".join(active_ablations)}')
        if return_pre_aug:
            print(f'  SIB: return_pre_aug=True (CACR reference path enabled)')

    def forward(self, x, intensity_map=None):
        """
        Args:
            x: Bottleneck features [B, 512, H/16, W/16].
            intensity_map: [B, 1, H_img, W_img] for adaptive β (optional).

        Returns:
            out: Processed features [B, 512, H/16, W/16].
            losses: Dict with 'kl_content', 'kl_edge', 'kl_total'.
                    If return_pre_aug=True and training, also contains
                    'pre_aug_bottleneck' [B, 512, H/16, W/16].
        """
        losses = {}
        x = self.proj(x)
        B, C, H, W = x.shape

        if self.use_haar:
            # Ensure even spatial dims
            pad_h = H % 2
            pad_w = W % 2
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
                _, _, H_p, W_p = x.shape
            else:
                H_p, W_p = H, W

            ll, lh, hl, hh = self.haar(x)

            # ── Save pre-aug LL for CACR reference path ───────────────
            ll_pre_aug = None
            if self.return_pre_aug and self.training:
                ll_pre_aug = ll.clone().detach()

            # ── Content augmentation ──────────────────────────────────
            if self.use_content_aug and hasattr(self, 'content_aug'):
                ll = self.content_aug(ll)
                # A5: also augment edge/noise subbands (MRFP+ analog)
                if self.aug_all_subbands:
                    lh = self.content_aug(lh)
                    hl = self.content_aug(hl)
                    hh = self.content_aug(hh)

            # ── Differential VIB ──────────────────────────────────────
            if self.use_vib:

                if self.vib_wrong_subband:
                    # ═══ A10: VIB on HL only (wrong subband) ═══
                    # LL passes through → domain info leaks
                    # HL gets compressed → task-relevant edges lost
                    # This should DEGRADE performance (inverse evidence)
                    ll_z = ll
                    lh_z = lh
                    hl_z, kl_wrong = self.vib_content(hl)

                    # Apply same beta schedule as normal content path
                    if self.adaptive_beta and intensity_map is not None:
                        beta_map = self.intensity_beta(
                            intensity_map, (hl.shape[2], hl.shape[3]))
                        losses['kl_content'] = kl_wrong * (
                            beta_map.mean() / self.beta_content
                        ) * self.beta_content
                    else:
                        losses['kl_content'] = kl_wrong * self.beta_content

                    losses['kl_edge'] = torch.tensor(0.0, device=x.device)

                else:
                    # ═══ Standard path: VIB on LL (content) ═══
                    ll_z, kl_content = self.vib_content(ll)

                    # Passthrough gate: blend VIB output with original LL
                    if (self.use_passthrough_gate
                            and hasattr(self, 'passthrough_gate')):
                        ll_z = self.passthrough_gate(ll_z, ll)

                    # Intensity-adaptive β for content
                    if self.adaptive_beta and intensity_map is not None:
                        beta_map = self.intensity_beta(
                            intensity_map, (ll.shape[2], ll.shape[3]))
                        kl_content_weighted = (
                            kl_content * beta_map.mean() / self.beta_content
                        )
                        losses['kl_content'] = (
                            kl_content_weighted * self.beta_content
                        )
                    else:
                        losses['kl_content'] = kl_content * self.beta_content

                    # ── Edge VIB ──────────────────────────────────
                    if self.no_edge_vib:
                        # ═══ A9: No edge VIB — pass through unchanged ═══
                        lh_z = lh
                        hl_z = hl
                        losses['kl_edge'] = torch.tensor(
                            0.0, device=x.device)
                    else:
                        lh_z, kl_edge_lh = self.vib_edge(lh)
                        hl_z, kl_edge_hl = self.vib_edge(hl)

                        if self.symmetric_vib:
                            # ═══ A3: Same high β for all subbands ═══
                            if (self.adaptive_beta
                                    and intensity_map is not None):
                                # Use same intensity-adaptive beta as
                                # content (beta_map already computed above)
                                losses['kl_edge'] = (
                                    (kl_edge_lh + kl_edge_hl)
                                    * beta_map.mean()
                                    / self.beta_content
                                    * self.beta_content
                                )
                            else:
                                losses['kl_edge'] = (
                                    (kl_edge_lh + kl_edge_hl)
                                    * self.beta_content
                                )
                        else:
                            # Normal: low β for edges
                            losses['kl_edge'] = (
                                (kl_edge_lh + kl_edge_hl) * self.beta_edge
                            )

                losses['kl_total'] = (
                    losses['kl_content']
                    + losses.get('kl_edge', torch.tensor(0.0, device=x.device))
                )

                out = self.haar.inverse(ll_z, lh_z, hl_z, hh)

                # ── CACR: build pre-aug reference bottleneck ──────────
                if (self.return_pre_aug and self.training
                        and ll_pre_aug is not None
                        and not self.vib_wrong_subband):
                    # Run VIB mean (deterministic) on pre-aug LL
                    # Uses same VIB weights but no stochastic sampling
                    with torch.no_grad():
                        ll_z_ref, _ = self.vib_content.forward_mean(ll_pre_aug)
                    # Reconstruct pre-aug path with same edge VIBs
                    out_ref = self.haar.inverse(ll_z_ref, lh_z.detach(),
                                                hl_z.detach(), hh)
                    if pad_h or pad_w:
                        out_ref = out_ref[:, :, :H, :W]
                    losses['pre_aug_bottleneck'] = self.out_proj(out_ref)
            else:
                # No VIB: just reconstruct
                out = self.haar.inverse(ll, lh, hl, hh)
                losses['kl_total'] = torch.tensor(0.0, device=x.device)

            # Remove padding
            if pad_h or pad_w:
                out = out[:, :, :H, :W]
        else:
            # ── No Haar: uniform VIB on full features ─────────────────
            if self.use_vib:
                out, kl = self.vib_uniform(x)

                # Passthrough gate in uniform mode
                if (self.use_passthrough_gate
                        and hasattr(self, 'passthrough_gate')):
                    out = self.passthrough_gate(out, x)

                losses['kl_content'] = kl * self.beta_content
                losses['kl_total'] = losses['kl_content']
            else:
                out = x
                losses['kl_total'] = torch.tensor(0.0, device=x.device)

        out = self.out_proj(out)
        return out, losses


# ======================================================================
# Multi-Scale SIB Manager (for encoder stages)
# ======================================================================

class MultiScaleSIB(nn.Module):
    """
    Manages lightweight SIB modules at each encoder skip connection level.

    β scales by stage depth: β_stage_k = β_base × k/K
    where k is stage index (1-based), K is total stages.

    Args:
        stage_channels: List of channels per stage [64, 128, 256, 512].
        beta_base: Base β for deepest stage.
    """

    def __init__(self, stage_channels=(64, 128, 256, 512), beta_base=1e-4):
        super().__init__()
        K = len(stage_channels)
        self.sibs = nn.ModuleList([
            LightweightSIB(ch, beta=beta_base * (k + 1) / K)
            for k, ch in enumerate(stage_channels)
        ])

    def forward(self, features):
        """
        Args:
            features: List of [feat1, feat2, feat3, feat4] from encoder.

        Returns:
            filtered: List of filtered features.
            total_kl: Sum of all stage KL losses.
        """
        filtered = []
        total_kl = torch.tensor(0.0, device=features[0].device)
        for feat, sib in zip(features, self.sibs):
            out, kl = sib(feat)
            filtered.append(out)
            total_kl = total_kl + kl
        return filtered, total_kl


# ======================================================================
# TENT: Test-Time Entropy Minimization Utilities
# ======================================================================

def collect_bn_params(model):
    """
    Collect BatchNorm affine parameters (weight, bias) from a model.
    Used by TENT for test-time adaptation.

    Returns:
        params: List of (weight, bias) Parameter tensors from BN layers.
        bn_layers: List of BN module references (for mode toggling).
    """
    params = []
    bn_layers = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d,
                               nn.SyncBatchNorm)):
            params.append(module.weight)
            params.append(module.bias)
            bn_layers.append(module)
    return params, bn_layers


def collect_ln_params(model):
    """
    Collect LayerNorm affine parameters from a model.
    Used by TENT for test-time adaptation on ViT-based models (e.g. DINOv3).

    Returns:
        params: List of (weight, bias) Parameter tensors from LN layers.
        ln_layers: List of LN module references.
    """
    params = []
    ln_layers = []
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.elementwise_affine:
                params.append(module.weight)
                params.append(module.bias)
                ln_layers.append(module)
    return params, ln_layers


def configure_tent(model, use_bn=True, use_ln=False):
    """
    Configure a model for TENT adaptation.

    1. Set model to eval mode.
    2. Freeze all parameters.
    3. Enable grad + train mode only for BN/LN affine params.

    Args:
        model: The model to configure.
        use_bn: Adapt BatchNorm layers (for CNNs: MAMNet, OGLANet).
        use_ln: Adapt LayerNorm layers (for ViTs: DINOv3).

    Returns:
        tent_params: Parameters to optimize during TENT.
        norm_layers: Norm layer references for mode toggling.
    """
    # Freeze everything
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    tent_params = []
    norm_layers = []

    if use_bn:
        bn_params, bn_layers = collect_bn_params(model)
        for p in bn_params:
            p.requires_grad_(True)
        tent_params.extend(bn_params)
        norm_layers.extend(bn_layers)

    if use_ln:
        ln_params, ln_layers = collect_ln_params(model)
        for p in ln_params:
            p.requires_grad_(True)
        tent_params.extend(ln_params)
        norm_layers.extend(ln_layers)

    # Set norm layers to train mode (updates running stats)
    for layer in norm_layers:
        layer.train()

    return tent_params, norm_layers


def tent_adapt_step(model, images, intensity_map, tent_optimizer,
                    norm_layers, pred_pos_only=True):
    """
    Run one TENT adaptation step: minimize entropy on (optionally
    predicted-positive) pixels.

    NOTE: Skips adaptation when the batch is too small to satisfy
    BatchNorm's train-mode requirement (≥2 values per channel).
    Architectures with post-GAP BN layers (MAMNet's MSCAF, OGLANet's
    decoder GAP block) reduce spatial dims to 1×1, so batch size must
    be ≥ 2 for those BN layers to compute variance. The trailing test
    batch is size 1 with batch_size=8 and a non-divisible test set;
    we skip adaptation for that batch and rely on the affine params
    already adapted by previous batches.

    Args:
        model: Model configured for TENT.
        images: Input images [B, C, H, W].
        intensity_map: [B, 1, H, W] for SIB.
        tent_optimizer: Optimizer over norm affine params.
        norm_layers: Norm layers in train mode.
        pred_pos_only: If True, minimize entropy only on predicted-positive
                       pixels (motivated by §4.3 Orphan 2: TP_pred logit
                       drop is the dominant failure mode).

    Returns:
        entropy_mean: Mean entropy value (for logging), or 0.0 if skipped.
    """
    # Guard: BN in train mode requires ≥2 spatial values per channel.
    # Post-GAP BN layers see [B, C, 1, 1], so we need B ≥ 2.
    if images.size(0) < 2:
        return 0.0

    # Ensure norm layers are in train mode
    for layer in norm_layers:
        layer.train()

    outputs, _ = model(images, intensity_map=intensity_map)
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs['main']
    probs = F.softmax(logits, dim=1)  # [B, 2, H, W]

    # Entropy: -Σ p log p
    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)  # [B, H, W]

    if pred_pos_only:
        # Only minimize entropy for predicted-positive (shadow) pixels
        pred_pos_mask = probs[:, 1, :, :] > 0.5
        if pred_pos_mask.sum() > 0:
            loss = entropy[pred_pos_mask].mean()
        else:
            # Fallback: all pixels
            loss = entropy.mean()
    else:
        loss = entropy.mean()

    tent_optimizer.zero_grad()
    loss.backward()
    tent_optimizer.step()

    return loss.item()