"""
Spectral Information Bottleneck (SIB)

A plug-and-play module inserted at the encoder-decoder interface of any
dense prediction architecture.  Applies differential information-theoretic
compression across frequency subbands of learned features.

Design rationale (grounded in diagnostics):
  - Thread 1b showed transfer gap widens monotonically with surface
    intensity → intensity-adaptive compression on domain-carrying content.
  - Thread 1d showed CNN encoders carry city identity → compress domain
    features while preserving task-relevant boundaries.
  - Experiment C showed pixel-level standardisation is catastrophic →
    operate in feature space, not input space.
  - FADA (NeurIPS 2024) validated Haar wavelet separates content from
    style in feature space → use fixed Haar decomposition instead of
    learned disentanglement (C1/HSIC failed experimentally).

Components:
  HAAR  — Haar wavelet decomposition of features into 4 subbands
           (LL=content, LH/HL=edges, HH=diagonal texture).
           Zero parameters, perfectly invertible.
  VIB   — Variational Information Bottleneck.
           When HAAR is on: differential compression (aggressive on LL,
           gentle on LH/HL, pass-through on HH).
           When HAAR is off: uniform compression on all features.
           Intensity-adaptive β (optional) applies stronger compression
           at bright surfaces where domain entanglement is worst.
  AUG   — Stochastic content augmentation (training only).
           When HAAR is on: perturbs only LL content subband.
           When HAAR is off: perturbs all features.
           Includes random style perturbation and cross-domain mixing.
  GATE  — Passthrough gate (optional).
           Learned sigmoid gate that blends VIB output with original
           features.  Allows VIB to auto-disable when compression is
           not beneficial (e.g. when domain gap is small).
  MODULE_BYPASS — Module-level residual bypass gate (optional).
           Wraps the entire SIB pipeline (Haar → VIB → Aug → InverseHaar)
           in a learned residual.  Allows the module to completely bypass
           itself when the domain gap is small.

Ablation-specific flags (for §5.3 component ablation study):
  disable_content_vib — A1: skip content VIB on F_LL, keep edge VIB
  symmetric_vib       — A3: apply content-level VIB (high β) to LL, LH, HL
  aug_all_subbands    — A6: apply content augmentation to all subbands
  vib_on_hl_only      — A10: apply content VIB to F_HL (wrong subband)

NEW (diagnostic-motivated additions, parity with MAMNet/OGLANet ports):
  return_pre_aug — when True and training, snapshot post-VIB pre-aug
                   subbands (or post-VIB pre-aug features if no Haar),
                   reconstruct under no_grad, and return the result in
                   sib_losses['pre_aug_bottleneck'].  Used by the model
                   wrapper to drive a second decoder pass for CACR.
                   When module bypass is active, ref path also passes
                   through the same gate using the same x_pre_sib anchor
                   so main vs ref differ only in augmentation.

  forward_mean(x, ...) — deterministic VIB forward (uses mu, no
                         reparameterization sampling).  Available on
                         ContentVIB / EdgeVIB / UniformVIB for parity
                         with MAMNet's CACR ref-path pattern.

  TENT utilities:
    collect_bn_params, collect_ln_params, configure_tent, tent_adapt_step
    — per the original TENT paper (Wang et al., ICLR 2021), test-time
    adaptation by entropy minimization on normalization affine params.
    For DINOv3 (ViT backbone), the typical use is use_ln=True; the BN
    helper is included for parity with MAMNet/OGLANet.

Usage:
    sib = SIB(
        in_channels=1536,       # 4 × 384 concatenated encoder features
        embed_dim=384,          # output dim matching decoder expectation
        use_haar=True,
        use_vib=True,
        use_content_aug=True,
        adaptive_beta=True,
        use_passthrough_gate=False,
        use_module_bypass=False,
        num_domains=2,
    )

    task_features, sib_losses = sib(
        features_concat, intensity_map, city_ids,
        vib_warmup_factor=1.0,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Haar wavelet transform (zero parameters, perfectly invertible)
# ---------------------------------------------------------------------------

def haar_forward(x):
    """
    Single-level 2D Haar wavelet transform in feature space.

    Args:
        x: [B, C, H, W]  (H and W must be even)
    Returns:
        x_LL, x_LH, x_HL, x_HH — each [B, C, H/2, W/2]

    LL = low-low  (content / structure — domain-specific)
    LH = low-high (horizontal edges — task-relevant)
    HL = high-low (vertical edges — task-relevant)
    HH = high-high (diagonal texture / noise)
    """
    # Even/odd indexed rows and columns
    ee = x[:, :, 0::2, 0::2]   # even row, even col
    eo = x[:, :, 0::2, 1::2]   # even row, odd col
    oe = x[:, :, 1::2, 0::2]   # odd row, even col
    oo = x[:, :, 1::2, 1::2]   # odd row, odd col

    x_LL = (ee + eo + oe + oo) / 2.0
    x_LH = (ee + eo - oe - oo) / 2.0
    x_HL = (ee - eo + oe - oo) / 2.0
    x_HH = (ee - eo - oe + oo) / 2.0

    return x_LL, x_LH, x_HL, x_HH


def haar_inverse(x_LL, x_LH, x_HL, x_HH):
    """
    Inverse single-level 2D Haar wavelet transform.

    Args:
        x_LL, x_LH, x_HL, x_HH — each [B, C, H/2, W/2]
    Returns:
        x: [B, C, H, W]  (perfect reconstruction)
    """
    B, C, H2, W2 = x_LL.shape
    x = torch.zeros(B, C, H2 * 2, W2 * 2,
                     device=x_LL.device, dtype=x_LL.dtype)

    x[:, :, 0::2, 0::2] = (x_LL + x_LH + x_HL + x_HH) / 2.0
    x[:, :, 0::2, 1::2] = (x_LL + x_LH - x_HL - x_HH) / 2.0
    x[:, :, 1::2, 0::2] = (x_LL - x_LH + x_HL - x_HH) / 2.0
    x[:, :, 1::2, 1::2] = (x_LL - x_LH - x_HL + x_HH) / 2.0

    return x


# ---------------------------------------------------------------------------
# Passthrough Gate — learned VIB bypass
# ---------------------------------------------------------------------------

class PassthroughGate(nn.Module):
    """
    Learned scalar gate that blends VIB output with original features.

        output = g * vib_output + (1 - g) * original

    where g = sigmoid(linear(GAP(original))).

    Initialized with bias = -2.0 → sigmoid(-2) ≈ 0.12, so the module
    starts in near-passthrough mode.  The model must learn evidence that
    VIB compression is beneficial before the gate opens.

    This prevents VIB from destroying features when the domain gap is
    small (e.g. DINOv3 on Phoenix where the encoder is already
    domain-invariant).

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
        gap = original.mean(dim=[2, 3])               # [B, C]
        g = torch.sigmoid(self.fc(gap))                # [B, 1]
        g = g.unsqueeze(-1).unsqueeze(-1)              # [B, 1, 1, 1]
        return g * vib_out + (1.0 - g) * original


# ---------------------------------------------------------------------------
# Module Bypass Gate — learned residual around entire SIB pipeline
# ---------------------------------------------------------------------------

class ModuleBypassGate(nn.Module):
    """
    Module-level residual bypass gate wrapping the entire SIB pipeline
    (Haar → VIB → Aug → InverseHaar).

        F_out = α · F_sib + (1 − α) · F_pre_sib

    where α = sigmoid(linear(GAP(F_pre_sib))).

    Initialized with bias = +2.0 → sigmoid(2) ≈ 0.88, so SIB is "on"
    by default.  The gate must actively learn to turn off for inputs
    where SIB hurts (e.g. DINOv3 on Phoenix where the domain gap is
    small).

    This is separate from PassthroughGate which only gates the VIB
    output within the module.  ModuleBypassGate wraps everything after
    projection.

    Args:
        channels: Number of feature channels (post-projection dim).
    """

    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 2.0)

    def forward(self, sib_out, original):
        """
        Args:
            sib_out:  SIB-processed features [B, C, H, W].
            original: Pre-SIB features       [B, C, H, W] (post-projection).

        Returns:
            blended: [B, C, H, W]
            alpha:   [B] per-sample gate values for diagnostics.
        """
        gap = original.mean(dim=[2, 3])                # [B, C]
        alpha = torch.sigmoid(self.fc(gap))             # [B, 1]
        alpha_diag = alpha.squeeze(-1)                  # [B]
        alpha_4d = alpha.unsqueeze(-1).unsqueeze(-1)    # [B, 1, 1, 1]
        blended = alpha_4d * sib_out + (1.0 - alpha_4d) * original
        return blended, alpha_diag


# ---------------------------------------------------------------------------
# Content VIB — aggressive compression on low-frequency content
# ---------------------------------------------------------------------------

class ContentVIB(nn.Module):
    """
    Per-pixel VIB for the LL (content) subband.

    Intensity-adaptive β applies stronger compression at bright surfaces
    where domain-specific albedo information concentrates.

    Args:
        channels:      feature channels.
        beta_base:     minimum compression (used everywhere).
        beta_scale:    additional compression range for bright surfaces.
        adaptive_beta: if False, uses fixed beta = beta_base everywhere.
    """

    def __init__(self, channels, beta_base=0.01, beta_scale=0.02,
                 adaptive_beta=True):
        super().__init__()
        self.beta_base = beta_base
        self.beta_scale = beta_scale
        self.adaptive_beta = adaptive_beta

        mid = channels // 2
        self.mu_net = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )
        self.logvar_net = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )

        if adaptive_beta:
            self.intensity_net = nn.Sequential(
                nn.Conv2d(1, 16, 1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )

    def forward(self, f_ll, intensity_map=None, training=True):
        """
        Args:
            f_ll:          [B, C, H_f, W_f]
            intensity_map: [B, 1, H_img, W_img] in [0, 1] (can be None)
            training:      bool
        Returns:
            z:       [B, C, H_f, W_f]
            kl_loss: scalar
        """
        mu = self.mu_net(f_ll)
        logvar = self.logvar_net(f_ll)

        if training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        # KL(q(z|x) || N(0,I)) — per pixel, summed over channels
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)   # [B,C,H,W]
        kl = kl.sum(dim=1, keepdim=True)                       # [B,1,H,W]

        # Compute beta map
        if self.adaptive_beta and intensity_map is not None:
            _, _, H_f, W_f = f_ll.shape
            inten = F.adaptive_avg_pool2d(intensity_map, (H_f, W_f))
            beta_map = (self.beta_base
                        + self.beta_scale
                        * torch.sigmoid(self.intensity_net(inten)))
        else:
            beta_map = self.beta_base

        kl_loss = (beta_map * kl).mean()
        return z, kl_loss

    def forward_mean(self, f_ll, intensity_map=None):
        """
        Deterministic forward — returns mu only, no reparameterization
        sampling.  No KL is computed; this is for the CACR reference
        path so the only main-vs-ref difference is augmentation.

        Args:
            f_ll:          [B, C, H_f, W_f]
            intensity_map: ignored, kept for signature parity.
        Returns:
            z: [B, C, H_f, W_f]
        """
        return self.mu_net(f_ll)


# ---------------------------------------------------------------------------
# Edge VIB — gentle regularisation on high-frequency boundary subbands
# ---------------------------------------------------------------------------

class EdgeVIB(nn.Module):
    """
    Lightweight per-pixel VIB shared across LH and HL subbands.
    Fixed small β — just enough to regularise, not destroy boundary detail.

    Args:
        channels: feature channels.
        beta:     fixed compression strength (default very small).
    """

    def __init__(self, channels, beta=0.0001):
        super().__init__()
        self.beta = beta

        mid = channels // 4   # lighter than content VIB
        self.mu_net = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )
        self.logvar_net = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )

    def forward(self, f_edge, training=True):
        """
        Args:
            f_edge:   [B, C, H_f, W_f]
            training: bool
        Returns:
            z:       [B, C, H_f, W_f]
            kl_loss: scalar
        """
        mu = self.mu_net(f_edge)
        logvar = self.logvar_net(f_edge)

        if training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl = kl.sum(dim=1, keepdim=True)
        kl_loss = (self.beta * kl).mean()
        return z, kl_loss

    def forward_mean(self, f_edge):
        """Deterministic forward (mu only). See ContentVIB.forward_mean."""
        return self.mu_net(f_edge)


# ---------------------------------------------------------------------------
# Uniform VIB — used when Haar is off (operates on all features)
# ---------------------------------------------------------------------------

class UniformVIB(nn.Module):
    """
    Per-pixel VIB applied uniformly to all features (no frequency
    decomposition).  Used when Haar wavelet is disabled.

    Supports intensity-adaptive β when enabled.
    """

    def __init__(self, channels, beta_base=0.01, beta_scale=0.02,
                 adaptive_beta=True):
        super().__init__()
        self.beta_base = beta_base
        self.beta_scale = beta_scale
        self.adaptive_beta = adaptive_beta

        mid = channels // 2
        self.mu_net = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )
        self.logvar_net = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
        )

        if adaptive_beta:
            self.intensity_net = nn.Sequential(
                nn.Conv2d(1, 16, 1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
            )

    def forward(self, x, intensity_map=None, training=True):
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)

        if training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu

        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar)
        kl = kl.sum(dim=1, keepdim=True)

        if self.adaptive_beta and intensity_map is not None:
            _, _, H_f, W_f = x.shape
            inten = F.adaptive_avg_pool2d(intensity_map, (H_f, W_f))
            beta_map = (self.beta_base
                        + self.beta_scale
                        * torch.sigmoid(self.intensity_net(inten)))
        else:
            beta_map = self.beta_base

        kl_loss = (beta_map * kl).mean()
        return z, kl_loss

    def forward_mean(self, x, intensity_map=None):
        """Deterministic forward (mu only). See ContentVIB.forward_mean."""
        return self.mu_net(x)


# ---------------------------------------------------------------------------
# Content augmentation — stochastic perturbation (training only)
# ---------------------------------------------------------------------------

class ContentAugmentation(nn.Module):
    """
    Stochastic feature augmentation applied to content features only.

    Two modes (applied stochastically):
      1. Random style: re-normalise channel statistics with random γ, β.
      2. Cross-domain mixing: swap channel stats between samples from
         different source cities within the same batch.

    Zero learnable parameters.  Completely off at inference.

    Args:
        sigma_style: log-normal σ for style perturbation.
        sigma_shift: normal σ for shift perturbation.
        p_aug:       probability of random style perturbation.
        p_mix:       probability of cross-domain content mixing.
    """

    def __init__(self, sigma_style=0.25, sigma_shift=0.15,
                 p_aug=0.5, p_mix=0.3):
        super().__init__()
        self.sigma_style = sigma_style
        self.sigma_shift = sigma_shift
        self.p_aug = p_aug
        self.p_mix = p_mix

    def forward(self, z, city_ids=None, training=True):
        """
        Args:
            z:        [B, C, H, W]
            city_ids: [B] (optional; needed for cross-domain mixing)
            training: bool — no-op at inference.
        Returns:
            z (potentially augmented): [B, C, H, W]
        """
        if not training:
            return z

        B, C, H, W = z.shape
        eps = 1e-5

        # --- Random style perturbation ---
        if torch.rand(1).item() < self.p_aug:
            mu = z.mean(dim=(2, 3), keepdim=True)
            sigma = z.std(dim=(2, 3), keepdim=True).clamp(min=eps)

            gamma = torch.empty(B, C, 1, 1, device=z.device).log_normal_(
                0, self.sigma_style)
            beta = torch.empty(B, C, 1, 1, device=z.device).normal_(
                0, self.sigma_shift)

            z = gamma * (z - mu) / sigma + mu + beta

        # --- Cross-domain content mixing ---
        if (city_ids is not None and B > 1
                and torch.rand(1).item() < self.p_mix):
            unique = city_ids.unique()
            if len(unique) > 1:
                mu = z.mean(dim=(2, 3), keepdim=True)
                sigma = z.std(dim=(2, 3), keepdim=True).clamp(min=eps)
                z_out = z.clone()
                for i in range(B):
                    others = torch.where(city_ids != city_ids[i])[0]
                    if others.numel() > 0:
                        j = others[torch.randint(others.numel(), (1,)).item()]
                        z_out[i] = (sigma[j] * (z[i] - mu[i]) / sigma[i]
                                    + mu[j])
                z = z_out

        return z


# ---------------------------------------------------------------------------
# Full SIB wrapper
# ---------------------------------------------------------------------------

class SIB(nn.Module):
    """
    Spectral Information Bottleneck.

    Sits between encoder output (concatenated multi-block features) and
    decoder upsampling stages.

    Component interactions:
      use_haar=T, use_vib=T  → Differential VIB (content + edge)
      use_haar=F, use_vib=T  → Uniform VIB on all features
      use_haar=T, use_vib=F  → Haar decompose/reconstruct (identity)
      use_haar=F, use_vib=F  → Pure pass-through

      use_content_aug:       augments LL (if haar) or all features (if no haar)
      adaptive_beta:         intensity-adaptive β in VIB (otherwise fixed)
      use_passthrough_gate:  learned gate blending VIB output with original;
                             allows VIB to auto-disable when not beneficial
      use_module_bypass:     learned residual gate wrapping the entire SIB
                             pipeline (post-projection); allows module to
                             completely bypass itself per-sample

    Ablation flags (§5.3, mutually exclusive VIB modes):
      disable_content_vib:   A1 — no content VIB on F_LL, edge VIB only
      symmetric_vib:         A3 — content-level VIB (high β) on LL, LH, HL
      vib_on_hl_only:        A10 — content VIB on F_HL (wrong subband),
                             edge VIB on F_LH, nothing on F_LL

    Ablation flag (augmentation variant):
      aug_all_subbands:      A6 — augment all subbands, not just F_LL

    NEW (CACR support):
      return_pre_aug:        when True and training, snapshot post-VIB
                             pre-aug subbands and reconstruct under
                             no_grad.  Result placed in
                             sib_losses['pre_aug_bottleneck'] for the
                             model wrapper to consume.

    Args:
        in_channels:          concatenated encoder feature channels (e.g. 1536).
        embed_dim:            output channels the decoder expects (e.g. 384).
        use_haar:             enable Haar wavelet decomposition.
        use_vib:              enable variational information bottleneck.
        use_content_aug:      enable content augmentation (training only).
        adaptive_beta:        enable intensity-adaptive β in VIB.
        use_passthrough_gate: enable learned passthrough gate on VIB output.
        use_module_bypass:    enable module-level residual bypass gate.
        disable_content_vib:  A1 ablation — skip content VIB on F_LL.
        symmetric_vib:        A3 ablation — high-β VIB on LL, LH, HL.
        aug_all_subbands:     A6 ablation — augment all subbands.
        vib_on_hl_only:       A10 ablation — content VIB on F_HL instead of F_LL.
        num_domains:          number of source domains during training.
        vib_beta_content:     β for content VIB (or uniform VIB when no haar).
        vib_beta_edge:        β for edge VIB (only when haar is on).
        vib_beta_scale:       adaptive β range (only when adaptive_beta is on).
        aug_sigma_style:      content aug random-style log-normal σ.
        aug_sigma_shift:      content aug random-shift normal σ.
        aug_p_aug:            probability of random style perturbation.
        aug_p_mix:            probability of cross-domain content mixing.
    """

    def __init__(self, in_channels=1536, embed_dim=384,
                 use_haar=True, use_vib=True,
                 use_content_aug=True, adaptive_beta=True,
                 use_passthrough_gate=False,
                 use_module_bypass=False,
                 # ---- Ablation flags ----
                 disable_content_vib=False,
                 symmetric_vib=False,
                 aug_all_subbands=False,
                 vib_on_hl_only=False,
                 # ---- Hyperparameters ----
                 num_domains=2,
                 vib_beta_content=0.01, vib_beta_edge=0.0001,
                 vib_beta_scale=0.02,
                 aug_sigma_style=0.25, aug_sigma_shift=0.15,
                 aug_p_aug=0.5, aug_p_mix=0.3):
        super().__init__()

        self.use_haar = use_haar
        self.use_vib = use_vib
        self.use_content_aug = use_content_aug
        self.adaptive_beta = adaptive_beta
        self.use_passthrough_gate = use_passthrough_gate
        self.use_module_bypass = use_module_bypass

        # Ablation flags
        self.disable_content_vib = disable_content_vib
        self.symmetric_vib = symmetric_vib
        self.aug_all_subbands = aug_all_subbands
        self.vib_on_hl_only = vib_on_hl_only

        # NEW: CACR ref-path snapshot flag (set externally by model wrapper)
        self.return_pre_aug = False

        # --- Projection from concatenated encoder features to embed_dim ---
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # --- VIB modules ---
        if use_vib:
            if use_haar:
                # Differential VIB: aggressive on content, gentle on edges
                self.content_vib = ContentVIB(
                    channels=embed_dim,
                    beta_base=vib_beta_content,
                    beta_scale=vib_beta_scale,
                    adaptive_beta=adaptive_beta,
                )
                self.edge_vib = EdgeVIB(
                    channels=embed_dim,
                    beta=vib_beta_edge,
                )
            else:
                # Uniform VIB on all features
                self.uniform_vib = UniformVIB(
                    channels=embed_dim,
                    beta_base=vib_beta_content,
                    beta_scale=vib_beta_scale,
                    adaptive_beta=adaptive_beta,
                )

        # --- Passthrough gate (optional, requires VIB) ---
        if use_passthrough_gate and use_vib:
            self.passthrough_gate = PassthroughGate(embed_dim)

        # --- Module bypass gate (optional) ---
        if use_module_bypass:
            self.module_bypass_gate = ModuleBypassGate(embed_dim)

        # --- Content augmentation ---
        if use_content_aug:
            self.content_aug = ContentAugmentation(
                sigma_style=aug_sigma_style,
                sigma_shift=aug_sigma_shift,
                p_aug=aug_p_aug,
                p_mix=aug_p_mix,
            )

        # --- Log configuration ---
        active = []
        if use_haar:              active.append('HAAR')
        if use_vib:               active.append('VIB' + ('-diff' if use_haar
                                                          else '-uniform'))
        if adaptive_beta:         active.append('AdaptiveBeta')
        if use_content_aug:       active.append('ContentAug')
        if use_passthrough_gate:  active.append('PassthroughGate')
        if use_module_bypass:     active.append('ModuleBypass')
        # Ablation flags
        if disable_content_vib:   active.append('ABLATION:NoContentVIB')
        if symmetric_vib:         active.append('ABLATION:SymmetricVIB')
        if aug_all_subbands:      active.append('ABLATION:AugAllSubbands')
        if vib_on_hl_only:        active.append('ABLATION:VIBonHLonly')
        print(f'SIB initialised  |  in={in_channels} → out={embed_dim}  |'
              f'  active: {", ".join(active) if active else "none (pass-through)"}')

    # ------------------------------------------------------------------
    def forward(self, features_concat, intensity_map=None, city_ids=None,
                vib_warmup_factor=1.0):
        """
        Args:
            features_concat:   [B, in_channels, H, W]
            intensity_map:     [B, 1, H_img, W_img] in [0,1] (for VIB; None OK)
            city_ids:          [B] int64 domain labels (for aug; None OK)
            vib_warmup_factor: float in [0, 1]. Scales KL loss during warmup.
                               0 = no compression, 1 = full compression.
        Returns:
            x:          [B, embed_dim, H, W]
            sib_losses: dict  {'kl_content': …, 'kl_edge': …} or
                        {'kl_uniform': …} or {} when VIB is off.
                        If module bypass is on, also includes
                        'bypass_alpha': [B] per-sample gate values.
                        If return_pre_aug is on (training), also includes
                        'pre_aug_bottleneck': [B, embed_dim, H, W] features
                        post-VIB pre-augmentation, post-module-bypass.
        """
        training = self.training
        sib_losses = {}

        # 1. Projection
        x = self.projection(features_concat)   # [B, embed_dim, H, W]

        # Save post-projection features for module bypass gate
        if self.use_module_bypass:
            x_pre_sib = x

        # NEW: CACR-only path returns snapshot of post-VIB pre-aug
        # subbands (or features if no Haar) via no_grad reconstruction
        # before the augmentation step.  The augmented (main) path
        # continues normally.
        capture_pre_aug = self.return_pre_aug and training

        if self.use_haar:
            # ----------------------------------------------------------
            # HAAR pathway: decompose → per-band VIB → augment → reconstruct
            # ----------------------------------------------------------
            f_ll, f_lh, f_hl, f_hh = haar_forward(x)

            if self.use_vib:
                # ----- A10: VIB on wrong subband (HL instead of LL) -----
                if self.vib_on_hl_only:
                    # Content VIB on F_HL (wrong subband for inverse evidence)
                    f_hl_orig = f_hl
                    f_hl, kl_content = self.content_vib(
                        f_hl, intensity_map, training=training)
                    if (self.use_passthrough_gate
                            and hasattr(self, 'passthrough_gate')):
                        f_hl = self.passthrough_gate(f_hl, f_hl_orig)

                    # Edge VIB on F_LH only (F_LL untouched)
                    f_lh, kl_edge_lh = self.edge_vib(f_lh, training=training)
                    kl_edge = kl_edge_lh

                    sib_losses['kl_content'] = kl_content * vib_warmup_factor
                    sib_losses['kl_edge'] = kl_edge * vib_warmup_factor
                    sib_losses['kl_edge_lh'] = kl_edge_lh * vib_warmup_factor

                # ----- A3: Symmetric VIB (high β on LL, LH, HL) -----
                elif self.symmetric_vib:
                    # Apply content VIB (high β) to all three subbands
                    f_ll_orig = f_ll
                    f_ll, kl_ll = self.content_vib(
                        f_ll, intensity_map, training=training)
                    if (self.use_passthrough_gate
                            and hasattr(self, 'passthrough_gate')):
                        f_ll = self.passthrough_gate(f_ll, f_ll_orig)

                    # LH and HL also get content-level compression
                    f_lh, kl_lh = self.content_vib(
                        f_lh, intensity_map, training=training)
                    f_hl, kl_hl = self.content_vib(
                        f_hl, intensity_map, training=training)

                    kl_content = kl_ll + kl_lh + kl_hl

                    sib_losses['kl_content'] = kl_content * vib_warmup_factor
                    sib_losses['kl_edge'] = torch.tensor(
                        0.0, device=x.device)
                    # Per-subband breakdown for diagnostics
                    sib_losses['kl_ll'] = kl_ll * vib_warmup_factor
                    sib_losses['kl_lh'] = kl_lh * vib_warmup_factor
                    sib_losses['kl_hl'] = kl_hl * vib_warmup_factor

                # ----- A1: No content VIB on LL (edge VIB only) -----
                elif self.disable_content_vib:
                    # F_LL passes through uncompressed
                    f_lh, kl_edge_lh = self.edge_vib(
                        f_lh, training=training)
                    f_hl, kl_edge_hl = self.edge_vib(
                        f_hl, training=training)
                    kl_edge = kl_edge_lh + kl_edge_hl

                    sib_losses['kl_content'] = torch.tensor(
                        0.0, device=x.device)
                    sib_losses['kl_edge'] = kl_edge * vib_warmup_factor
                    sib_losses['kl_edge_lh'] = kl_edge_lh * vib_warmup_factor
                    sib_losses['kl_edge_hl'] = kl_edge_hl * vib_warmup_factor

                # ----- Default C4: Content VIB on LL, Edge VIB on LH/HL -----
                else:
                    f_ll_orig = f_ll
                    f_ll, kl_content = self.content_vib(
                        f_ll, intensity_map, training=training)

                    # Passthrough gate: blend VIB output with original LL
                    if (self.use_passthrough_gate
                            and hasattr(self, 'passthrough_gate')):
                        f_ll = self.passthrough_gate(f_ll, f_ll_orig)

                    f_lh, kl_edge_lh = self.edge_vib(
                        f_lh, training=training)
                    f_hl, kl_edge_hl = self.edge_vib(
                        f_hl, training=training)
                    kl_edge = kl_edge_lh + kl_edge_hl

                    sib_losses['kl_content'] = kl_content * vib_warmup_factor
                    sib_losses['kl_edge'] = kl_edge * vib_warmup_factor
                    # Per-subband breakdown for diagnostics
                    sib_losses['kl_edge_lh'] = kl_edge_lh * vib_warmup_factor
                    sib_losses['kl_edge_hl'] = kl_edge_hl * vib_warmup_factor

            # ── CACR snapshot: post-VIB, pre-aug subbands ──
            # Reconstruct the pre-augmentation features under no_grad.
            # If module bypass is active, also pass through the gate
            # using x_pre_sib as the residual anchor (same as main).
            if capture_pre_aug:
                with torch.no_grad():
                    x_pre_aug = haar_inverse(
                        f_ll.detach(), f_lh.detach(),
                        f_hl.detach(), f_hh.detach())
                    if self.use_module_bypass:
                        x_pre_aug, _ = self.module_bypass_gate(
                            x_pre_aug, x_pre_sib.detach())
                    sib_losses['pre_aug_bottleneck'] = x_pre_aug

            # --- Content augmentation ---
            if self.use_content_aug:
                f_ll = self.content_aug(f_ll, city_ids, training=training)
                # A6: Also augment edge and noise subbands
                if self.aug_all_subbands:
                    f_lh = self.content_aug(f_lh, city_ids, training=training)
                    f_hl = self.content_aug(f_hl, city_ids, training=training)
                    f_hh = self.content_aug(f_hh, city_ids, training=training)

            # Inverse Haar (f_hh passes through uncompressed)
            x = haar_inverse(f_ll, f_lh, f_hl, f_hh)

        else:
            # ----------------------------------------------------------
            # NON-HAAR pathway: uniform VIB → augment
            # ----------------------------------------------------------
            if self.use_vib:
                # Save original for passthrough gate
                x_orig = x

                x, kl_uniform = self.uniform_vib(
                    x, intensity_map, training=training)

                # Passthrough gate: blend VIB output with original
                if (self.use_passthrough_gate
                        and hasattr(self, 'passthrough_gate')):
                    x = self.passthrough_gate(x, x_orig)

                sib_losses['kl_uniform'] = kl_uniform * vib_warmup_factor

            # ── CACR snapshot: post-VIB, pre-aug features ──
            if capture_pre_aug:
                with torch.no_grad():
                    x_pre_aug = x.detach().clone()
                    if self.use_module_bypass:
                        x_pre_aug, _ = self.module_bypass_gate(
                            x_pre_aug, x_pre_sib.detach())
                    sib_losses['pre_aug_bottleneck'] = x_pre_aug

            if self.use_content_aug:
                x = self.content_aug(x, city_ids, training=training)

        # Module bypass gate: blend SIB output with pre-SIB features
        if self.use_module_bypass:
            x, bypass_alpha = self.module_bypass_gate(x, x_pre_sib)
            sib_losses['bypass_alpha'] = bypass_alpha   # [B] for diagnostics

        return x, sib_losses


# ===========================================================================
# TENT — Test-time Entropy Minimization
# (Wang et al., "Tent: Fully Test-time Adaptation by Entropy Minimization",
#  ICLR 2021)
#
# At test time, optimize a small number of normalization affine parameters
# (γ, β) to minimize prediction entropy on the target stream.  No source
# data, no target labels.  For DINOv3 (ViT backbone) the relevant norm is
# LayerNorm; the BN helper is included for parity with MAMNet/OGLANet.
# ===========================================================================

def collect_bn_params(model):
    """
    Collect BatchNorm affine parameters and put their layers in train mode.

    Returns:
        params:  list of nn.Parameter (BN γ and β across the model).
        layers:  list of (name, module) for the BN layers being adapted.
    """
    params = []
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()
            # Disable running stat tracking — use the current batch's stats
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            for p_name, p in m.named_parameters(recurse=False):
                if p_name in ('weight', 'bias'):
                    p.requires_grad_(True)
                    params.append(p)
            layers.append((name, m))
    return params, layers


def collect_ln_params(model):
    """
    Collect LayerNorm affine parameters.  LN does not track running stats,
    so train mode is not required, but we still ensure affine params have
    requires_grad=True.

    Returns:
        params:  list of nn.Parameter (LN γ and β across the model).
        layers:  list of (name, module).
    """
    params = []
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            for p_name, p in m.named_parameters(recurse=False):
                if p_name in ('weight', 'bias') and p is not None:
                    p.requires_grad_(True)
                    params.append(p)
            layers.append((name, m))
    return params, layers


def configure_tent(model, use_bn=False, use_ln=True):
    """
    Configure model for TENT test-time adaptation.

    Defaults set for ViT (DINOv3): adapt LayerNorm only.  For CNN models
    (MAMNet/OGLANet) call with use_bn=True, use_ln=False.

    Steps:
      1. Set model to eval (so dropout etc are off).
      2. Freeze all parameters.
      3. Re-enable grad on requested norm-layer affine params.
      4. For BN, additionally put those layers in train mode and disable
         running-stat tracking so they use batch stats.

    Returns:
        adapt_params: list of nn.Parameter that the TENT optimizer will update.
        layers:       list of (name, module) being adapted.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    adapt_params = []
    layers = []

    if use_bn:
        bn_params, bn_layers = collect_bn_params(model)
        adapt_params.extend(bn_params)
        layers.extend(bn_layers)

    if use_ln:
        ln_params, ln_layers = collect_ln_params(model)
        adapt_params.extend(ln_params)
        layers.extend(ln_layers)

    return adapt_params, layers


def tent_adapt_step(model, images, intensity_map, optimizer, layers,
                    pred_pos_only=True, city_ids=None,
                    vib_warmup_factor=1.0):
    """
    One TENT adaptation step on a target batch.

    Minimizes prediction entropy: H(p) = -sum_c p_c log p_c

    DINOv3 model returns (logits, sib_losses) — we unpack and use logits
    directly.  The model's forward signature is:
        model(x, intensity_map, city_ids=None, vib_warmup_factor=1.0)
              -> (logits[B,C,H,W], sib_losses_dict)

    For BatchNorm-based models (used elsewhere), batch size 1 fails because
    BN train mode requires ≥2 samples per channel.  For LayerNorm-based
    models (DINOv3) batch size 1 is fine.  We keep the guard anyway since
    it costs nothing — caller is responsible for the choice.

    Args:
        model:           model to adapt (already in eval mode + norm affines
                         trainable via configure_tent).
        images:          [B, C, H, W] target batch.
        intensity_map:   [B, 1, H, W] (DINOv3 forward expects this).
        optimizer:       SGD/Adam over the adapt_params from configure_tent.
        layers:          list of (name, module) being adapted (unused; kept
                         for caller convenience / future logging).
        pred_pos_only:   if True, restrict entropy to predicted-positive
                         (shadow) pixels — focus adaptation on the class
                         the model is uncertain about.
        city_ids:        [B] optional, passed through to model.
        vib_warmup_factor: passed through to model (typically 1.0 at test).

    Returns:
        entropy: scalar tensor (the loss that was minimized).
    """
    # Forward
    out = model(images, intensity_map=intensity_map, city_ids=city_ids,
                vib_warmup_factor=vib_warmup_factor)

    # Unpack DINOv3's (logits, sib_losses) tuple
    if isinstance(out, tuple) and len(out) == 2:
        logits = out[0]
    elif isinstance(out, dict) and 'predictions' in out:
        # Fallback for OGLANet-style dict output (not used by DINOv3)
        preds = out['predictions']
        if isinstance(preds, dict) and 'p6' in preds:
            logits = preds['p6']
        else:
            logits = preds
    else:
        logits = out

    # Per-pixel entropy: -sum_c p_c log p_c
    log_probs = F.log_softmax(logits, dim=1)
    probs     = log_probs.exp()
    pixel_ent = -(probs * log_probs).sum(dim=1)         # [B, H, W]

    if pred_pos_only:
        with torch.no_grad():
            pred_pos = logits.argmax(dim=1) == 1         # [B, H, W]
        if pred_pos.sum() > 0:
            entropy = pixel_ent[pred_pos].mean()
        else:
            entropy = pixel_ent.mean()
    else:
        entropy = pixel_ent.mean()

    optimizer.zero_grad()
    entropy.backward()
    optimizer.step()
    return entropy.detach()