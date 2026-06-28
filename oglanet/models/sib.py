"""
Spectral Information Bottleneck (SIB) for OGLANet.

Components:
  - HaarWaveletDecomposition / InverseHaar: lossless 2D Haar via depthwise conv
  - SpectralVIB: differential VIB with per-subband beta (content/edge/noise)
  - ContentAugmentation: style noise + cross-sample mixing on LL subband
  - PassthroughGate: learned scalar gate that blends VIB output with
    original features, allowing VIB to auto-disable when compression
    is not beneficial (e.g. when domain gap is small)
  - ModuleBypassGate: learned per-sample scalar that blends the ENTIRE
    SIB pipeline output with the original encoder features, allowing the
    module to be completely bypassed when it is not beneficial
  - SkipAttentionGate (SAG): channel-wise domain filter for skip connections
  - SIB: main module composing Haar → VIB → Aug → InvHaar
  - MultiScaleSIB: lightweight SIB at multiple encoder stages

Primary placement: feat4 [B, 512, 12, 12]
  → Haar produces 4 subbands of [B, 512, 6, 6]
  → 36 spatial positions per subband (adequate for VIB)

Ablation flags (all default to off / preserve C4 behavior):
  - skip_ll_vib:      A1 — skip VIB on LL, keep VIB on LH/HL/HH
  - symmetric_beta:   A3 — force all subbands to use beta_content
  - aug_all_subbands: A6 — apply ContentAugmentation to all subbands
  - vib_only_band:    A10 — apply VIB only to the named band

NEW — Diagnostic-motivated additions (§4.3 orphan coverage):
  - SpectralVIB.forward_mean(): deterministic forward (mu only, no
    reparameterization sampling) used by the CACR reference path.
  - SIB.return_pre_aug flag: when True and training, the module also
    returns the post-VIB pre-augmentation reconstruction in the kl_losses
    dict under key 'pre_aug_bottleneck'. This is the reference feature
    for CACR's asymmetric penalty.
  - TENT utilities (configure_tent, tent_adapt_step + helpers): set BN
    layers to train mode, freeze everything else, then minimize entropy
    on predicted-positive pixels. Includes a guard for batch sizes < 2
    so trailing test batches don't trip post-GAP BN's variance check.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════════════════════
# Haar Wavelet Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

class HaarWaveletDecomposition(nn.Module):
    """
    2D Haar wavelet decomposition via fixed depthwise convolution (stride=2).
    Produces 4 subbands: LL (content), LH (horizontal edges),
    HL (vertical edges), HH (diagonal/noise).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        filters = torch.tensor([
            [[ 0.5,  0.5], [ 0.5,  0.5]],  # LL
            [[ 0.5, -0.5], [ 0.5, -0.5]],  # LH
            [[ 0.5,  0.5], [-0.5, -0.5]],  # HL
            [[ 0.5, -0.5], [-0.5,  0.5]],  # HH
        ], dtype=torch.float32)

        weight = filters.unsqueeze(1).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', weight)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Spatial dims must be even, got {H}×{W}"

        out = F.conv2d(x, self.weight, stride=2, groups=C)
        out = out.view(B, C, 4, H // 2, W // 2)
        return {
            'LL': out[:, :, 0],
            'LH': out[:, :, 1],
            'HL': out[:, :, 2],
            'HH': out[:, :, 3],
        }


class InverseHaarWavelet(nn.Module):
    """Inverse 2D Haar wavelet: reconstruct from 4 subbands."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        filters = torch.tensor([
            [[ 0.5,  0.5], [ 0.5,  0.5]],
            [[ 0.5, -0.5], [ 0.5, -0.5]],
            [[ 0.5,  0.5], [-0.5, -0.5]],
            [[ 0.5, -0.5], [-0.5,  0.5]],
        ], dtype=torch.float32)

        weight = filters.unsqueeze(1).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', weight)

    def forward(self, subbands: dict) -> torch.Tensor:
        B, C, Hh, Wh = subbands['LL'].shape
        stacked = torch.stack(
            [subbands['LL'], subbands['LH'], subbands['HL'], subbands['HH']],
            dim=2
        )
        stacked = stacked.view(B, C * 4, Hh, Wh)
        out = F.conv_transpose2d(stacked, self.weight, stride=2, groups=C)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Spectral Variational Information Bottleneck
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralVIB(nn.Module):
    """
    Differential VIB applied per subband with different beta strengths:
      - LL (content):  beta_content  = 0.01  (moderate compression)
      - LH/HL (edges): beta_edge     = 0.001 (gentle, preserve edges)
      - HH (noise):    beta_noise    = 0.05  (aggressive, remove noise)

    Optionally uses intensity-adaptive beta scaling via a small MLP.

    Each subband gets its own mu/logvar projection (1×1 conv).

    Ablation flags:
      skip_ll_vib:    A1 — pass LL through unchanged, VIB only on LH/HL/HH
      symmetric_beta: A3 — use beta_content for ALL bands (no asymmetry)
      vib_only_band:  A10 — apply VIB only to the named band, pass others through
    """

    def __init__(self, channels: int,
                 beta_content: float = 0.01,
                 beta_edge: float = 0.001,
                 beta_noise: float = 0.05,
                 adaptive_beta: bool = True,
                 skip_ll_vib: bool = False,
                 symmetric_beta: bool = False,
                 vib_only_band: str = None):
        super().__init__()
        self.base_betas = {
            'LL': beta_content,
            'LH': beta_edge,
            'HL': beta_edge,
            'HH': beta_noise,
        }
        self.adaptive_beta = adaptive_beta
        self.skip_ll_vib = skip_ll_vib
        self.symmetric_beta = symmetric_beta
        self.vib_only_band = vib_only_band

        if vib_only_band is not None:
            assert vib_only_band in ('LL', 'LH', 'HL', 'HH'), \
                f"vib_only_band must be one of LL/LH/HL/HH, got '{vib_only_band}'"

        self.mu_projs = nn.ModuleDict()
        self.logvar_projs = nn.ModuleDict()
        for band in ['LL', 'LH', 'HL', 'HH']:
            self.mu_projs[band] = nn.Conv2d(channels, channels, 1)
            self.logvar_projs[band] = nn.Conv2d(channels, channels, 1)
            nn.init.constant_(self.logvar_projs[band].bias, -5.0)
            nn.init.zeros_(self.logvar_projs[band].weight)

        if adaptive_beta:
            self.beta_mlp = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 4),
                nn.Softplus(),
            )
            nn.init.zeros_(self.beta_mlp[0].weight)
            nn.init.zeros_(self.beta_mlp[0].bias)
            nn.init.zeros_(self.beta_mlp[2].weight)
            nn.init.constant_(self.beta_mlp[2].bias, 0.5)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def _should_skip_band(self, band: str) -> bool:
        if self.skip_ll_vib and band == 'LL':
            return True
        if self.vib_only_band is not None and band != self.vib_only_band:
            return True
        return False

    def _compute_scale_map(self, intensity_map, device):
        """Helper: compute adaptive beta scales (shared by forward + forward_mean)."""
        if self.adaptive_beta and intensity_map is not None:
            mean_intensity = intensity_map.mean(dim=[1, 2, 3], keepdim=False)
            scales = self.beta_mlp(mean_intensity.unsqueeze(1))
            return {
                'LL': scales[:, 0], 'LH': scales[:, 1],
                'HL': scales[:, 2], 'HH': scales[:, 3],
            }
        return {k: torch.ones(1, device=device)
                for k in ['LL', 'LH', 'HL', 'HH']}

    def forward(self, subbands: dict, intensity_map: torch.Tensor = None):
        """
        Args:
            subbands: dict 'LL','LH','HL','HH' each (B, C, H/2, W/2)
            intensity_map: (B, 1, H_orig, W_orig) or None

        Returns:
            filtered_subbands: dict with same structure
            kl_losses: dict mapping band name → weighted KL scalar
        """
        device = next(iter(subbands.values())).device
        scale_map = self._compute_scale_map(intensity_map, device)

        filtered = {}
        kl_losses = {}

        for band in ['LL', 'LH', 'HL', 'HH']:
            x = subbands[band]

            if self._should_skip_band(band):
                filtered[band] = x
                kl_losses[band] = torch.tensor(0.0, device=device)
                continue

            mu = self.mu_projs[band](x)
            logvar = self.logvar_projs[band](x)
            z = self._reparameterize(mu, logvar)
            filtered[band] = z

            kl_raw = self._kl_divergence(mu, logvar)
            beta = self.base_betas['LL'] if self.symmetric_beta else self.base_betas[band]
            avg_scale = scale_map[band].mean()
            kl_losses[band] = beta * avg_scale * kl_raw

        return filtered, kl_losses

    # ── NEW: deterministic forward for CACR reference path ────────────
    def forward_mean(self, subbands: dict, intensity_map: torch.Tensor = None):
        """
        Deterministic forward — returns mu directly (no reparameterization).

        Used by the CACR reference path so that the only difference between
        the main and reference paths is content augmentation, not VIB
        sampling stochasticity.

        Iterates only over the bands present in `subbands`, so callers can
        pass {'LL': ll} alone if only LL is needed.

        Args:
            subbands: dict with one or more of 'LL','LH','HL','HH'
            intensity_map: (B, 1, H, W) or None

        Returns:
            filtered: dict with deterministic mu outputs (same keys)
            kl_losses: dict of weighted KL scalars (same keys; not used
                       for backprop, but returned for symmetry)
        """
        device = next(iter(subbands.values())).device
        scale_map = self._compute_scale_map(intensity_map, device)

        filtered = {}
        kl_losses = {}

        for band in subbands.keys():
            x = subbands[band]

            if self._should_skip_band(band):
                filtered[band] = x
                kl_losses[band] = torch.tensor(0.0, device=device)
                continue

            mu = self.mu_projs[band](x)
            logvar = self.logvar_projs[band](x)
            filtered[band] = mu

            kl_raw = self._kl_divergence(mu, logvar)
            beta = self.base_betas['LL'] if self.symmetric_beta else self.base_betas[band]
            avg_scale = scale_map.get(band, torch.ones(1, device=device)).mean()
            kl_losses[band] = beta * avg_scale * kl_raw

        return filtered, kl_losses


# ═══════════════════════════════════════════════════════════════════════════════
# Content Augmentation (applied to LL subband only, unless aug_all_subbands)
# ═══════════════════════════════════════════════════════════════════════════════

class ContentAugmentation(nn.Module):
    """
    Style perturbation on LL (content) subband:
      1. LogNormal multiplicative noise (style variation)
      2. Normal additive shift
      3. Cross-sample mixing (within batch)

    Only active during training.
    """

    def __init__(self, sigma_style: float = 0.1, sigma_shift: float = 0.05,
                 p_aug: float = 0.5, p_mix: float = 0.3):
        super().__init__()
        self.sigma_style = sigma_style
        self.sigma_shift = sigma_shift
        self.p_aug = p_aug
        self.p_mix = p_mix

    def forward(self, ll: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return ll

        B = ll.shape[0]
        device = ll.device

        aug_mask = torch.rand(B, device=device) < self.p_aug
        if not aug_mask.any():
            return ll

        out = ll.clone()

        style_noise = torch.exp(
            torch.randn(B, ll.shape[1], 1, 1, device=device) * self.sigma_style
        )
        shift = torch.randn(B, ll.shape[1], 1, 1, device=device) * self.sigma_shift

        augmented = ll * style_noise + shift

        aug_mask_expanded = aug_mask.view(B, 1, 1, 1).float()
        out = out * (1 - aug_mask_expanded) + augmented * aug_mask_expanded

        mix_mask = (torch.rand(B, device=device) < self.p_mix) & aug_mask
        if mix_mask.any() and B > 1:
            perm = torch.randperm(B, device=device)
            alpha = torch.rand(B, 1, 1, 1, device=device) * 0.3
            mixed = out * (1 - alpha) + out[perm] * alpha
            mix_expanded = mix_mask.view(B, 1, 1, 1).float()
            out = out * (1 - mix_expanded) + mixed * mix_expanded

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Passthrough Gate
# ═══════════════════════════════════════════════════════════════════════════════

class PassthroughGate(nn.Module):
    """
    Learned scalar gate that blends VIB output with original features.

        output = g * vib_output + (1 - g) * original

    where g = sigmoid(linear(GAP(original))).

    Initialized with bias = -2.0 → sigmoid(-2) ≈ 0.12, so the module
    starts in near-passthrough mode.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, -2.0)

    def forward(self, vib_out: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        gap = original.mean(dim=[2, 3])
        g = torch.sigmoid(self.fc(gap))
        g = g.unsqueeze(-1).unsqueeze(-1)
        return g * vib_out + (1.0 - g) * original


# ═══════════════════════════════════════════════════════════════════════════════
# Module Bypass Gate
# ═══════════════════════════════════════════════════════════════════════════════

class ModuleBypassGate(nn.Module):
    """
    Module-level residual bypass that wraps the ENTIRE SIB pipeline:

        F_out = α · F_sib + (1 − α) · F_encoder

    Initialized with bias = +2.0 → sigmoid(2) ≈ 0.88, so SIB is "on" by
    default. The gate must learn to turn off via segmentation loss
    gradient for inputs where SIB hurts.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Linear(channels, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 2.0)

    def forward(self, sib_out: torch.Tensor, original: torch.Tensor):
        gap = original.mean(dim=[2, 3])
        alpha = torch.sigmoid(self.fc(gap)).squeeze(-1)
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        blended = alpha_expanded * sib_out + (1.0 - alpha_expanded) * original
        return blended, alpha


# ═══════════════════════════════════════════════════════════════════════════════
# Skip Attention Gate (SAG)
# ═══════════════════════════════════════════════════════════════════════════════

class SkipAttentionGate(nn.Module):
    """Channel-wise sigmoid gate for skip connections (SE-block style)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.gate[4].weight)
        nn.init.constant_(self.gate[4].bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate(x)
        return x * g.unsqueeze(-1).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Main SIB Module
# ═══════════════════════════════════════════════════════════════════════════════

class SIB(nn.Module):
    """
    Spectral Information Bottleneck: Haar → VIB → ContentAug → InvHaar.

    Primary application: feat4 of GLAMEncoder [B, 512, 12, 12].
    All components are independently toggleable for ablation.

    Ablation-specific flags (all default to off / C4 behavior):
      skip_ll_vib:      A1 — skip VIB on LL subband
      symmetric_beta:   A3 — same beta_content for all subbands
      aug_all_subbands: A6 — apply ContentAugmentation to LH/HL/HH too
      vib_only_band:    A10 — apply VIB only to the named band

    NEW — return_pre_aug: when True and training, the module also returns
    the post-VIB pre-augmentation reconstruction in the kl_losses dict
    under key 'pre_aug_bottleneck'. This is the reference feature for
    CACR's asymmetric penalty. The reference reconstruction is built
    inside torch.no_grad() — no gradient flows through the ref path.
    """

    def __init__(self, channels: int = 512,
                 beta_content: float = 0.01,
                 beta_edge: float = 0.001,
                 beta_noise: float = 0.05,
                 adaptive_beta: bool = True,
                 sigma_style: float = 0.1,
                 sigma_shift: float = 0.05,
                 aug_p_aug: float = 0.5,
                 aug_p_mix: float = 0.3,
                 use_haar: bool = True,
                 use_vib: bool = True,
                 use_aug: bool = True,
                 use_passthrough_gate: bool = False,
                 # Ablation flags
                 skip_ll_vib: bool = False,
                 symmetric_beta: bool = False,
                 aug_all_subbands: bool = False,
                 vib_only_band: str = None,
                 # NEW: CACR support
                 return_pre_aug: bool = False):
        super().__init__()
        self.use_haar = use_haar
        self.use_vib = use_vib
        self.use_aug = use_aug
        self.use_passthrough_gate = use_passthrough_gate
        self.aug_all_subbands = aug_all_subbands
        self.return_pre_aug = return_pre_aug

        if use_haar:
            self.haar = HaarWaveletDecomposition(channels)
            self.inv_haar = InverseHaarWavelet(channels)

        if use_vib:
            self.vib = SpectralVIB(
                channels, beta_content, beta_edge, beta_noise, adaptive_beta,
                skip_ll_vib=skip_ll_vib,
                symmetric_beta=symmetric_beta,
                vib_only_band=vib_only_band,
            )

        if use_aug:
            self.aug = ContentAugmentation(sigma_style, sigma_shift, aug_p_aug, aug_p_mix)

        if use_passthrough_gate and use_vib:
            self.passthrough_gate = PassthroughGate(channels)

    def forward(self, x: torch.Tensor, intensity_map: torch.Tensor = None,
                city_ids: torch.Tensor = None):
        """
        Args:
            x:             (B, C, H, W) encoder feature (e.g. feat4)
            intensity_map: (B, 1, H_img, W_img) raw intensity
            city_ids:      (B,) long tensor (unused here, reserved for future)

        Returns:
            out:       (B, C, H, W) filtered feature
            kl_losses: dict of band → scalar KL losses (empty if VIB disabled).
                       When return_pre_aug=True and training, also contains
                       'pre_aug_bottleneck' [B, C, H, W] for CACR.
        """
        kl_losses = {}

        if not self.use_haar:
            # No Haar: apply uniform VIB directly if enabled.
            # CACR reference path is not meaningful without Haar (no aug),
            # so we don't populate pre_aug_bottleneck here.
            if self.use_vib:
                x_original = x
                subbands = {'LL': x, 'LH': torch.zeros_like(x),
                            'HL': torch.zeros_like(x), 'HH': torch.zeros_like(x)}
                filtered, kl_losses = self.vib(subbands, intensity_map)
                out = filtered['LL']
                if self.use_passthrough_gate and hasattr(self, 'passthrough_gate'):
                    out = self.passthrough_gate(out, x_original)
            else:
                out = x
            return out, kl_losses

        # ── Haar decomposition ──
        subbands = self.haar(x)

        # ── Spectral VIB ──
        if self.use_vib:
            ll_original = subbands['LL']
            subbands, kl_losses = self.vib(subbands, intensity_map)
            if self.use_passthrough_gate and hasattr(self, 'passthrough_gate'):
                subbands['LL'] = self.passthrough_gate(subbands['LL'], ll_original)

        # ── CACR: snapshot post-VIB pre-aug subbands for reference path ──
        # The ref path reconstructs from these (no augmentation applied),
        # so the difference between main and ref is purely the augmentation
        # effect. Wrapped in no_grad — gradients flow only through main path.
        ref_features = None
        if self.return_pre_aug and self.training and self.use_aug:
            with torch.no_grad():
                pre_aug_subbands = {k: v.detach() for k, v in subbands.items()}
                ref_features = self.inv_haar(pre_aug_subbands)

        # ── Content augmentation (main path) ──
        if self.use_aug:
            subbands['LL'] = self.aug(subbands['LL'])
            if self.aug_all_subbands:
                subbands['LH'] = self.aug(subbands['LH'])
                subbands['HL'] = self.aug(subbands['HL'])
                subbands['HH'] = self.aug(subbands['HH'])

        # ── Inverse Haar ──
        out = self.inv_haar(subbands)

        if ref_features is not None:
            kl_losses['pre_aug_bottleneck'] = ref_features

        return out, kl_losses


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Scale SIB (optional lightweight SIB at early encoder stages)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleSIB(nn.Module):
    """Apply lightweight SIB at multiple encoder stages with channel reduction."""

    def __init__(self, channel_list: list, bottleneck_ratio: float = 0.25, **sib_kwargs):
        super().__init__()
        self.sibs = nn.ModuleList()
        self.reduce_convs = nn.ModuleList()
        self.restore_convs = nn.ModuleList()

        # MultiScaleSIB never produces CACR ref features — strip the flag if present
        sib_kwargs.pop('return_pre_aug', None)

        for ch in channel_list:
            bottleneck_ch = max(int(ch * bottleneck_ratio), 16)
            self.reduce_convs.append(
                nn.Conv2d(ch, bottleneck_ch, 1, bias=False)
            )
            self.sibs.append(
                SIB(channels=bottleneck_ch, **sib_kwargs)
            )
            self.restore_convs.append(
                nn.Conv2d(bottleneck_ch, ch, 1, bias=False)
            )

    def forward(self, features: list, intensity_map=None, city_ids=None):
        filtered = []
        all_kl = {}

        for i, (feat, reduce, sib, restore) in enumerate(
            zip(features, self.reduce_convs, self.sibs, self.restore_convs)
        ):
            _, _, h, w = feat.shape
            if h % 2 != 0 or w % 2 != 0:
                filtered.append(feat)
                continue

            reduced = reduce(feat)
            sib_out, kl = sib(reduced, intensity_map, city_ids)
            restored = restore(sib_out)
            filtered.append(feat + restored)

            for band, val in kl.items():
                # Skip non-loss tensors (pre_aug_bottleneck shouldn't appear
                # here since we stripped return_pre_aug, but guard anyway)
                if band == 'pre_aug_bottleneck':
                    continue
                all_kl[f"scale{i}_{band}"] = val

        return filtered, all_kl


# ═══════════════════════════════════════════════════════════════════════════════
# TENT: Test-Time Entropy Minimization Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def collect_bn_params(model):
    """
    Collect BatchNorm affine parameters (weight, bias) from a model.
    Used by TENT for test-time adaptation on CNNs.

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
    Used by TENT for test-time adaptation on ViT-based models.
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
        use_bn: Adapt BatchNorm layers (for CNNs: OGLANet, MAMNet).
        use_ln: Adapt LayerNorm layers (for ViTs: DINOv3).

    Returns:
        tent_params: Parameters to optimize during TENT.
        norm_layers: Norm layer references for mode toggling.
    """
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

    for layer in norm_layers:
        layer.train()

    return tent_params, norm_layers


def tent_adapt_step(model, images, intensity_map, tent_optimizer,
                    norm_layers, pred_pos_only=True, city_ids=None):
    """
    Run one TENT adaptation step: minimize entropy on (optionally
    predicted-positive) pixels of the main p6 prediction.

    NOTE: Skips adaptation when batch is too small to satisfy BatchNorm's
    train-mode requirement (≥2 values per channel). Architectures with
    post-GAP BN layers reduce spatial dims to 1×1, so batch size must be
    ≥ 2 for those BN layers to compute variance. The trailing test batch
    is size 1 with batch_size=8 and a non-divisible test set; we skip
    adaptation for that batch and rely on affine params already adapted
    by previous batches.

    Args:
        model: Model configured for TENT (output dict with 'predictions').
        images: Input images [B, C, H, W].
        intensity_map: [B, 1, H, W] for SIB.
        tent_optimizer: Optimizer over norm affine params.
        norm_layers: Norm layers in train mode.
        pred_pos_only: If True, minimize entropy only on predicted-positive
                       (shadow) pixels (motivated by §4.3 Orphan 2).
        city_ids: Optional [B] long tensor (forwarded to model).

    Returns:
        entropy_mean: Mean entropy value (for logging), or 0.0 if skipped.
    """
    # Guard: BN in train mode requires ≥2 spatial values per channel.
    # Post-GAP BN layers see [B, C, 1, 1], so we need B ≥ 2.
    if images.size(0) < 2:
        return 0.0

    for layer in norm_layers:
        layer.train()

    out = model(images, intensity_map=intensity_map, city_ids=city_ids)
    predictions = out['predictions']
    # OGLANet returns a dict — adapt on p6 (full-resolution main output).
    logits = predictions['p6']
    probs = F.softmax(logits, dim=1)

    entropy = -(probs * (probs + 1e-10).log()).sum(dim=1)

    if pred_pos_only:
        pred_pos_mask = probs[:, 1, :, :] > 0.5
        if pred_pos_mask.sum() > 0:
            loss = entropy[pred_pos_mask].mean()
        else:
            loss = entropy.mean()
    else:
        loss = entropy.mean()

    tent_optimizer.zero_grad()
    loss.backward()
    tent_optimizer.step()

    return loss.item()