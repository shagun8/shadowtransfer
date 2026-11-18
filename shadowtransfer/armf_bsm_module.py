"""Arm F / BSM — batch style mixing, the AdaIN mixing module.

Reimplementation of Luo, Ji, Wei, "A Diverse Large-Scale Building Dataset and a
Novel Plug-and-Play Domain Generalization Method for Building Extraction," IEEE
JSTARS 16:4122-4138, 2023. No official code exists; this file and its caller
are our declared reimplementation from the paper.

The frozen style network is the Huang & Belongie (2017) AdaIN encoder-decoder,
weights from the unofficial-but-canonical naoto0804/pytorch-AdaIN release
(github.com/naoto0804/pytorch-AdaIN, tag v0.0.0 — the original repo's own
release links; the repo is archived but the GitHub Release asset URLs
resolve directly, no Google Drive). Architecture indices below were verified
against the downloaded state_dicts' own keys/shapes (torch.load + inspect),
not reconstructed from memory: every Conv2d in and out channel count below is
required to exactly match the checkpoint or `load_state_dict(strict=True)`
fails loudly at attach time.

INPUT-SPACE NOTE. This project's three published architectures (MAMNet,
OGLANet, DINOv3) and CrossEarth all hand their model's forward() an
already-normalised tensor (ImageNet mean/std for the three published archs;
mmcv 0-255-scale mean/std for CrossEarth) — never raw [0,1] pixels. The AdaIN
encoder's first layer (`vgg[0]`, a fixed 1x1 conv baked into
vgg_normalised.pth) performs its own channel mean-centering and expects [0,1]
RGB, per the original implementation's convention. `attach_bsm_hook` therefore
denormalises to [0,1] immediately before the encoder and renormalises
immediately after the decoder, using the caller-supplied mean/std/scale, so
BSM operates purely as a `forward_pre_hook` on the already-constructed model
with no dataset-code changes anywhere. This denormalise/renormalise round
trip is a design choice this project makes because the paper is silent on how
BSM composes with a co-existing input normalisation step.
"""

import hashlib
from pathlib import Path

import torch
import torch.nn as nn

# Huang & Belongie encoder: truncated VGG19 (+ fixed 1x1 RGB-normalising
# conv at index 0) up to and including relu4_1. Full vgg_normalised.pth
# stores through relu5_4 (52 modules); AdaIN only ever uses the first 31.
_VGG_FULL = [
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),  # 29,30 = conv4_1, relu4_1
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),
]
ENCODER_TRUNCATE_AT = 31  # vgg[:31] ends right after relu4_1 (index 30)

_DECODER = [
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
]


def sha256_and_size(path):
    h = hashlib.sha256()
    n = 0
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def load_adain(weights_dir, device='cpu'):
    """Build the frozen encoder (vgg[:31], relu4_1) and decoder, strict-load.

    Returns (encoder, decoder, report). Both modules are eval()'d and every
    parameter has requires_grad_(False); report carries path/sha256/bytes for
    each weight file plus the trainable-parameter count (must be 0).
    """
    weights_dir = Path(weights_dir)
    dec_path = weights_dir / 'decoder.pth'
    vgg_path = weights_dir / 'vgg_normalised.pth'

    vgg_full = nn.Sequential(*_VGG_FULL)
    vgg_full.load_state_dict(torch.load(vgg_path, map_location='cpu'), strict=True)
    encoder = nn.Sequential(*list(vgg_full.children())[:ENCODER_TRUNCATE_AT])

    decoder = nn.Sequential(*_DECODER)
    decoder.load_state_dict(torch.load(dec_path, map_location='cpu'), strict=True)

    encoder = encoder.to(device).eval()
    decoder = decoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    n_trainable = (sum(p.numel() for p in encoder.parameters() if p.requires_grad)
                   + sum(p.numel() for p in decoder.parameters() if p.requires_grad))
    dec_sha, dec_bytes = sha256_and_size(dec_path)
    vgg_sha, vgg_bytes = sha256_and_size(vgg_path)
    report = {
        "arch": "Huang & Belongie (2017) AdaIN encoder (VGG19 truncated at "
                "relu4_1) + mirror decoder",
        "weights_url": "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/",
        "decoder": {"path": str(dec_path), "sha256": dec_sha, "bytes": dec_bytes},
        "vgg_normalised": {"path": str(vgg_path), "sha256": vgg_sha, "bytes": vgg_bytes},
        "trainable_params": int(n_trainable),
    }
    return encoder, decoder, report


def calc_mean_std(feat, eps=1e-5):
    b, c = feat.shape[:2]
    var = feat.view(b, c, -1).var(dim=2, unbiased=False) + eps
    std = var.sqrt().view(b, c, 1, 1)
    mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return mean, std


def adain(content_feat, style_feat, eps=1e-5):
    c_mean, c_std = calc_mean_std(content_feat, eps)
    s_mean, s_std = calc_mean_std(style_feat, eps)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


def batch_shuffle_perm(batch_size, generator=None):
    """A random permutation of the batch dim, resampled only if it is the
    identity (i.e. zero shuffling happened at all). Occasional fixed points
    (an image paired with itself) are left as-is: CLAUDE.md's stage-0 gate
    asks that self-pairing occur no more than chance would give, not that it
    never occurs — forcing a derangement would be a different, stronger
    guarantee than the spec asks for."""
    identity = torch.arange(batch_size)
    perm = identity
    for _ in range(50):
        perm = torch.randperm(batch_size, generator=generator)
        if not torch.equal(perm, identity):
            break
    self_pairs = int((perm == identity).sum().item())
    return perm, self_pairs


def bsm_mix(content01, encoder, decoder, alpha, generator=None):
    """content01: (B,3,H,W) in [0,1]. Returns (mixed01, perm, self_pairs).

    alpha=0 must return content unchanged (up to encode/decode round-trip
    numerical error); alpha=1 returns the fully-stylised AdaIN output.
    """
    b = content01.shape[0]
    perm, self_pairs = batch_shuffle_perm(b, generator)
    style01 = content01[perm]
    with torch.no_grad():
        c_feat = encoder(content01)
        s_feat = encoder(style01)
        t = adain(c_feat, s_feat)
        t = alpha * t + (1.0 - alpha) * c_feat
        out = decoder(t)
    return out.clamp(0.0, 1.0), perm, self_pairs


def attach_bsm_hook(module, mean, std, scale, encoder, decoder, alpha,
                     generator=None, channels_to_mix=3, state=None,
                     gate=None):
    """Register a forward_pre_hook on `module` that BSM-mixes the first
    `channels_to_mix` input channels when `module.training` is True, and is a
    strict no-op (returns None, i.e. unmodified args) otherwise.

    `mean`/`std`/`scale` describe how to invert the caller's own input
    normalisation back to [0,1] RGB: raw01 = (x*std + mean) / scale, and the
    inverse renormalisation after mixing. Any channels beyond
    `channels_to_mix` (e.g. MAMNet/OGLANet's contrast channel) pass through
    unmodified. `state`, if given a dict, is updated in place with the last
    permutation and self-pair count for logging.

    Returns the hook handle (`.remove()` to detach).
    """
    mean_t = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

    def _pre_hook(mod, args):
        # D-B3-1 (operator, 2026-10-10). `mod.training` is a PROXY for "we are
        # inside a training step." On a FROZEN backbone the proxy is simply
        # wrong: CrossEarth's ReinsDinoVisionTransformer.train(True)
        # (reins_dinov2.py:41) never calls super().train(True) -- it calls
        # set_train(self, ["reins"]), whose first action is model.train(False)
        # (backbones/utils.py:48). The backbone therefore reports
        # training=False INSIDE a real train step, the hook returns None, and a
        # cell trains vanilla under a bsm label.
        #
        # `gate`, when supplied, is an explicit callable the installer sets
        # True around train steps and False around validation/test/dump. The
        # SEMANTICS ARE UNCHANGED -- mix during training, never during eval --
        # only the signal is now the truth rather than a proxy for it.
        #
        # gate=None reproduces the published behaviour BIT-FOR-BIT, so every
        # existing BSM cell (27 CNN cells, a3/a4) is untouched.
        if not (mod.training if gate is None else bool(gate())):
            return None
        x = args[0]
        dev, dtype = x.device, x.dtype
        mt, st = mean_t.to(dev), std_t.to(dev)
        rgb_norm = x[:, :channels_to_mix].float()
        rest = x[:, channels_to_mix:] if x.shape[1] > channels_to_mix else None

        raw01 = ((rgb_norm * st + mt) / scale).clamp(0.0, 1.0)
        mixed01, perm, self_pairs = bsm_mix(raw01, encoder, decoder, alpha, generator)
        mixed_norm = ((mixed01 * scale - mt) / st).to(dtype)

        x2 = torch.cat([mixed_norm, rest], dim=1) if rest is not None else mixed_norm
        if state is not None:
            state["last_perm"] = perm.tolist()
            state["last_self_pairs"] = self_pairs
            state["n_calls"] = state.get("n_calls", 0) + 1
        return (x2,) + tuple(args[1:])

    return module.register_forward_pre_hook(_pre_hook)
