"""
Arm E gate E-1 — encoder drift measurement. Weight-space and grouping helpers.

READ-ONLY with respect to every published asset. Nothing here trains, writes a
checkpoint, or writes a dump. The only artifact this gate produces is one
results JSON (arme_e1_drift.py).

WHAT "drift" MEANS HERE
------------------------
For a fine-tuned encoder W_ft and its initialisation W_init, relative L2 drift
is ||W_ft - W_init||_F / ||W_init||_F, computed by flattening every tensor in a
named group into one vector (so a per-block or per-stage number is the drift of
that whole group treated as one parameter, not an average of per-tensor
ratios).

DINOv3's W_init is "the pretrained weights AS LOADED by the published
dinov3_backbone.py construction" (per the gate spec), which per Gate E-0 has
NO LayerScale gammas (ls1/ls2 are nn.Identity()) -- so the DINOv3 group lists
below never include a `.gamma` key, and that omission is the E-0 finding, not
a bug here. DINOv3's LayerNorm modules carry no buffers, so its
`state_dict()` is already parameters-only; no filtering is needed there.

MAMNet/OGLANet's W_init is the actual published construction too: both archs'
`models/encoder_4ch.py` build a 4-channel ResNet-34 stem by copying the
ImageNet-pretrained 3-channel conv1 weights into channels 0-2 and filling
channel 3 (contrast) with their mean, then reuse the pretrained bn1/layer1-4
modules unchanged. That fill is deterministic (mean of already-pretrained
weights, no randomness), so W_init is reproducible exactly, every call.

BUFFERS MUST BE EXCLUDED, and were not in the first pass of this gate.
`nn.Module.state_dict()` returns parameters AND buffers together. ResNet-34's
BatchNorm layers register three buffers each (`running_mean`, `running_var`,
`num_batches_tracked`), none of which is a weight: `num_batches_tracked` is an
int64 STEP COUNTER (0 at init, in the thousands after training) with no
parameter-scale meaning at all, and including it made every ResNet
relative-L2 number here come out above 1 (some above 100) against DINOv3's
0.01-0.02, on the first run of this gate (2026-07-29, job 20563356). That
mismatch is what caught the bug, not a real cross-architecture finding: fixed
by intersecting every key list against `fresh.named_parameters()` (a real
`nn.Module`'s own parameter names), not against its full `state_dict()`. See
`resnet_stage_keys`/`resnet_whole_encoder_report`'s `param_keys` argument.

Comparator scope, per the gate spec: conv1 (bundled with bn1, the stem) and
layer1-4. The mamnet-only `downsample` module (a new 3x3 conv the decoder
wiring adds, not part of ImageNet ResNet-34) is out of scope: it has no
"ImageNet-pretrained init" to drift from.
"""

import importlib.util
from pathlib import Path

import torch

DINOV3_MODULES = ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.fc1", "mlp.fc2")
N_DINOV3_BLOCKS = 12

RESNET_STAGES = ("conv1", "layer1", "layer2", "layer3", "layer4")
RESNET_ENCODER_PREFIX = {
    "mamnet": "encoder.",
    "oglanet": "encoder.resnet_encoder.",
}


# ----------------------------------------------------------------------------
# generic relative-L2 grouping, architecture-agnostic
# ----------------------------------------------------------------------------

def group_relative_l2(sd_init, sd_ft, keys):
    """Relative L2 drift of one named group, keys flattened into one vector.

    Returns nan (not raise) when init_sq == 0 or a key is entirely absent from
    both dicts, so a caller can report an empty group rather than crash on it.
    """
    diff_sq, init_sq, n_params = 0.0, 0.0, 0
    missing = []
    for k in keys:
        if k not in sd_init or k not in sd_ft:
            missing.append(k)
            continue
        a = sd_init[k].detach().to(torch.float64)
        b = sd_ft[k].detach().to(torch.float64)
        diff_sq += float(((b - a) ** 2).sum())
        init_sq += float((a ** 2).sum())
        n_params += a.numel()
    out = {
        "relative_l2": (diff_sq ** 0.5) / (init_sq ** 0.5) if init_sq > 0 else float("nan"),
        "l2_diff": diff_sq ** 0.5,
        "l2_init_norm": init_sq ** 0.5,
        "n_params": int(n_params),
        "n_keys": len(keys) - len(missing),
    }
    if missing:
        out["missing_keys"] = missing
    return out


def keys_with_prefixes(sd, prefixes):
    return [k for k in sd if any(k.startswith(p) for p in prefixes)]


def strip_prefix(full_sd, prefix):
    """{k: v} for keys under `prefix`, prefix removed -- puts a full-model
    checkpoint's encoder sub-tree into the same local key space a freshly
    constructed encoder's own .state_dict() uses."""
    return {k[len(prefix):]: v for k, v in full_sd.items() if k.startswith(prefix)}


# ----------------------------------------------------------------------------
# DINOv3 ViT-S/16 block/module grouping
# ----------------------------------------------------------------------------

def dinov3_module_keys(sd, block_idx, module):
    prefix = f"blocks.{block_idx}.{module}."
    return [k for k in sd if k.startswith(prefix)]


def dinov3_block_report(sd_init, sd_ft, block_idx):
    modules = {}
    for m in DINOV3_MODULES:
        keys = dinov3_module_keys(sd_init, block_idx, m)
        modules[m] = group_relative_l2(sd_init, sd_ft, keys)
    whole_block_keys = [k for k in sd_init if k.startswith(f"blocks.{block_idx}.")]
    return {
        "block": block_idx,
        "modules": modules,
        "whole_block": group_relative_l2(sd_init, sd_ft, whole_block_keys),
    }


def dinov3_whole_encoder_report(sd_init, sd_ft):
    return group_relative_l2(sd_init, sd_ft, list(sd_init.keys()))


# ----------------------------------------------------------------------------
# ResNet-34 (4-channel stem) stage grouping
# ----------------------------------------------------------------------------

def resnet_stage_keys(sd, stage, param_keys):
    """`param_keys` MUST be a real nn.Module's named_parameters() key set, so
    BatchNorm buffers (running_mean/running_var/num_batches_tracked) never
    enter a weight-drift comparison. See the module docstring."""
    prefixes = ("conv1.", "bn1.") if stage == "conv1" else (f"{stage}.",)
    return [k for k in keys_with_prefixes(sd, prefixes) if k in param_keys]


def resnet_stage_report(sd_init, sd_ft, stage, param_keys):
    keys = resnet_stage_keys(sd_init, stage, param_keys)
    return {"stage": stage, **group_relative_l2(sd_init, sd_ft, keys)}


def resnet_whole_encoder_report(sd_init, sd_ft, param_keys):
    keys = []
    for stage in RESNET_STAGES:
        keys.extend(resnet_stage_keys(sd_init, stage, param_keys))
    return group_relative_l2(sd_init, sd_ft, keys)


def _load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_fresh_resnet_encoder(base, arch):
    """A freshly constructed ResNet34Encoder4Ch(pretrained=True), loaded by
    explicit file path (not sys.path) so mamnet's and oglanet's same-named
    `encoder_4ch` modules never collide in sys.modules. This IS W_init: no
    checkpoint is touched, only the published construction against
    torchvision's ImageNet-1k ResNet-34 weights."""
    path = Path(base) / "python" / arch / "models" / "encoder_4ch.py"
    mod = _load_module_from_path(f"_arme_e1_{arch}_encoder_4ch", path)
    return mod.ResNet34Encoder4Ch(pretrained=True)


def load_full_checkpoint_state(ckpt_path):
    """mamnet/oglanet train.py both save {'epoch':.., 'model_state_dict':..,
    ...}; no DataParallel wrapping (single-GPU jobs), so keys are unprefixed."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck.get("state_dict", ck))
    return sd, ck.get("epoch")
