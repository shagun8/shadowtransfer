"""Arm F / CrossEarth — SIB on a ViT: the Haar + VIB path only.

Per paper §5.1, a ViT encoder receives NEITHER Skip Attention Gates NOR FDA
input preprocessing; both are CNN-only. Neither is present in this file.

Components, Appendix H and Table 37:
  - single-level 2-D Haar decomposition of encoder features -> LL/LH/HL/HH
  - content VIB on F_LL, hidden C/2, per-pixel mean and log-variance,
    sampled at train time and mean-only at inference
  - KL on the content branch weighted per pixel by
        beta_content(x) = beta_base + beta_scale * sigmoid(MLP(I_local(x)))
    beta_base 0.01, beta_scale 0.02, I_local the local mean RGB intensity
    downsampled to bottleneck resolution through a 1 -> 16 -> 1 ReLU MLP
  - edge VIB shared across F_LH and F_HL, hidden C/4, beta_edge 1e-4
  - F_HH unmodified
  - inverse Haar before the decoder
  - KL ramps linearly from 0 to target over the first 10% of iterations

INSERTION POINT
---------------
Our DINOv3 decoder consumes one fused feature map, so "between encoder and
decoder" is a single tensor there. Here it is not. `ReinsDinoVisionTransformer`
returns `(feats, querys)`, where `feats` is a 4-element list taken from blocks
[7, 11, 15, 23]. `ReinMask2FormerHead.forward` immediately does
    mask_features, multi_scale_memorys = self.pixel_decoder(x)
so that 4-element list IS the whole encoder->decoder interface.

The four maps leave forward_features at one resolution, (B, 1024, H/16, W/16),
but DinoVisionTransformer.forward (dino_v2.py:328) then rescales three of them
to the strides the head declares -- element 0 by 4x, element 1 by 2x, element 3
by 0.5x, element 2 untouched. MEASURED at crop 384 by the gate job (20601326):

    [4, 1024, 96, 96]  [4, 1024, 48, 48]  [4, 1024, 24, 24]  [4, 1024, 12, 12]

so the interface is genuinely multi-scale, at strides 4 / 8 / 16 / 32. An
earlier draft of this docstring claimed all four were 24x24; that was wrong,
and measuring rather than asserting it is what the gate job existed for.

Proposed insertion point, one proposal, reported not iterated:

    module path : model.decode_head.sib
    operates on : the 4-element backbone feature list, in
                  ReinMask2FormerHead.forward, after `x, query_embed = x` and
                  BEFORE `self.pixel_decoder(x)`
    scales      : all four, with ONE shared-weight SIB block, not four blocks
                  and not one privileged depth
    shapes      : in and out as measured above; Haar subbands are half each
                  side, [1024, 48/24/12/6 squared]
    KL          : one scalar, the mean of the four maps' KL terms

Sharing one block across the four maps is what keeps this a single bottleneck
rather than four independent ones, which is the closest available analogue of
the single fused map the CNN and DINOv3 designs bottleneck. It is well defined
across the scales because all four carry the same 1024 channels and the block
is fully convolutional in 1x1, so only the spatial extent differs. Applying it
to only one map would leave three unbottlenecked paths into the pixel decoder,
which defeats the mechanism.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BETA_BASE = 0.01
BETA_SCALE = 0.02
BETA_EDGE = 1e-4
KL_WARMUP_FRAC = 0.10
TAU = 0.5


def haar_fwd(x):
    """Single-level orthonormal 2-D Haar. (B,C,H,W) -> 4 x (B,C,H/2,W/2)."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    return ((a + b + c + d) / 2.0, (a + b - c - d) / 2.0,
            (a - b + c - d) / 2.0, (a - b - c + d) / 2.0)


def haar_inv(ll, lh, hl, hh):
    """Exact inverse of haar_fwd."""
    a = (ll + lh + hl + hh) / 2.0
    b = (ll + lh - hl - hh) / 2.0
    c = (ll - lh + hl - hh) / 2.0
    d = (ll - lh - hl + hh) / 2.0
    bsz, ch, h, w = a.shape
    out = a.new_empty((bsz, ch, h * 2, w * 2))
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out


class VIB(nn.Module):
    """Per-pixel Gaussian bottleneck: two 1x1-conv MLP heads off one hidden."""

    def __init__(self, channels, hidden):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                 nn.ReLU(inplace=True))
        self.mu = nn.Conv2d(hidden, channels, 1)
        self.logvar = nn.Conv2d(hidden, channels, 1)

    def forward(self, x, sample):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10.0, 10.0)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        # per-pixel KL against N(0, I), summed over channels
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1,
                                                                 keepdim=True)
        return z, kl


class SIBBlock(nn.Module):
    """Haar + content VIB + shared edge VIB + inverse Haar."""

    def __init__(self, channels, beta_base=BETA_BASE, beta_scale=BETA_SCALE,
                 beta_edge=BETA_EDGE):
        super().__init__()
        self.channels = channels
        self.beta_base = beta_base
        self.beta_scale = beta_scale
        self.beta_edge = beta_edge
        self.content_vib = VIB(channels, max(channels // 2, 1))
        self.edge_vib = VIB(channels, max(channels // 4, 1))
        # I_local -> beta_content, a 1 -> 16 -> 1 ReLU MLP as 1x1 convs
        self.beta_mlp = nn.Sequential(nn.Conv2d(1, 16, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 1, 1))
        self.last_kl = None

    def beta_content(self, intensity, hw):
        """beta_base + beta_scale * sigmoid(MLP(I_local)) at bottleneck res."""
        i_local = F.adaptive_avg_pool2d(intensity, hw)
        return self.beta_base + self.beta_scale * torch.sigmoid(
            self.beta_mlp(i_local))

    def forward(self, feats, intensity, sample):
        """feats: list of (B, C, H, W). intensity: (B, 1, H0, W0) in [0, 1]."""
        outs, kls = [], []
        for f in feats:
            ll, lh, hl, hh = haar_fwd(f)
            z_ll, kl_c = self.content_vib(ll, sample)
            z_lh, kl_e1 = self.edge_vib(lh, sample)
            z_hl, kl_e2 = self.edge_vib(hl, sample)
            beta_c = self.beta_content(intensity, ll.shape[-2:])
            kl = ((beta_c * kl_c).mean()
                  + self.beta_edge * (kl_e1.mean() + kl_e2.mean()))
            outs.append(haar_inv(z_ll, z_lh, z_hl, hh))
            kls.append(kl)
        self.last_kl = torch.stack(kls).mean()
        return outs


def attach_sib(model, mean, std, crop=384):
    """Insert one shared SIB block at the encoder->decoder interface.

    `mean`/`std` are the data preprocessor's, used to recover raw [0, 1] RGB
    intensity from the normalised tensor the backbone is handed, so I_local is
    image intensity and not a normalisation artefact.

    Returns (sib, n_sib_params, report_dict). The block is registered as
    `model.decode_head.sib` so it is saved, moved and optimised with the head.
    """
    head = model.decode_head
    channels = int(model.backbone.embed_dim)
    sib = SIBBlock(channels).to(next(model.parameters()).device)
    head.sib = sib

    dev = next(model.parameters()).device
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    state = {"intensity": None}

    def _pre_hook(_mod, args):
        x = args[0]
        raw = (x.detach().float() * std_t + mean_t) / 255.0
        state["intensity"] = raw.mean(dim=1, keepdim=True).clamp(0.0, 1.0)
        return None

    model.backbone.register_forward_pre_hook(_pre_hook)

    original_forward = head.forward

    def forward(x, batch_data_samples):
        feats, query_embed = x
        feats = sib(feats, state["intensity"], sib.training)
        return original_forward((feats, query_embed), batch_data_samples)

    head.forward = forward

    n = sum(p.numel() for p in sib.parameters())
    report = {
        "insertion_point": "model.decode_head.sib, applied to the 4-element "
                           "ReinsDinoVisionTransformer feature list inside "
                           "ReinMask2FormerHead.forward, before "
                           "self.pixel_decoder(x)",
        "applied_to": "all four scales, one shared-weight SIB block",
        "n_scales": 4,
        # The backbone rescales its four maps to strides 4/8/16/32 in
        # DinoVisionTransformer.forward, so the interface is multi-scale.
        # Measured by the gate job at crop 384; the head's own `strides` key
        # declares the same ladder.
        "feature_shapes_chw": [[channels, crop // s, crop // s]
                               for s in (4, 8, 16, 32)],
        "subband_shapes_chw": [[channels, crop // (2 * s), crop // (2 * s)]
                               for s in (4, 8, 16, 32)],
        "strides": [4, 8, 16, 32],
        "hidden_widths": {"content": channels // 2, "edge": channels // 4},
        "beta_base": BETA_BASE, "beta_scale": BETA_SCALE,
        "beta_edge": BETA_EDGE, "kl_warmup_frac": KL_WARMUP_FRAC,
        "n_sib_params": int(n),
    }
    return sib, int(n), report


def kl_ramp(it, total_iters, frac=KL_WARMUP_FRAC):
    """Linear 0 -> 1 over the first `frac` of training."""
    w = max(int(total_iters * frac), 1)
    return float(min(1.0, (it + 1) / w))


# =============================================================================
# LIFTED-FROM verify_c_star_val.py :: find_c_star_val (lines 67-76)
# Body verbatim, imports only. This is the deployment-time coverage selection
# rule, not a metric; the risk-coverage curve it consumes comes from
# shadowtransfer.metrics.per_image_rc, which is the canonical implementation.
# =============================================================================
def find_c_star_val(val_rc_aggregate, coverage_grid, beta, fallback=1.0):
    """Largest c such that aggregate val sel-err(c) <= sel-err(c=1.0)*(1-beta)."""
    val_full = float(val_rc_aggregate[-1])
    if val_full <= 0:
        return fallback, val_full, val_full * (1 - beta)
    target = val_full * (1.0 - beta)
    valid = val_rc_aggregate <= target
    if not np.any(valid):
        return fallback, val_full, target
    return float(coverage_grid[np.where(valid)[0].max()]), val_full, target


def selective_shadow_error(gts, probs):
    """Macro-averaged selective shadow error curve over a split.

    Per-image risk-coverage on the GT-shadow pixel population from
    metrics.per_image_rc (mode 'gt_shadow'), averaged across images that have
    enough shadow pixels to define one. COVERAGE_GRID is
    np.linspace(0.10, 1.00, 19), i.e. step 0.05, which is the grid the spec
    asks for. Returns (aggregate curve, grid, n_images_used) or (None, ...).
    """
    from shadowtransfer.metrics import COVERAGE_GRID, per_image_rc
    curves = [rc for rc in (per_image_rc(np.asarray(g), np.asarray(p), "gt_shadow")
                            for g, p in zip(gts, probs)) if rc is not None]
    if not curves:
        return None, COVERAGE_GRID, 0
    return (np.mean(np.asarray(curves, dtype=np.float64), axis=0),
            COVERAGE_GRID, len(curves))
