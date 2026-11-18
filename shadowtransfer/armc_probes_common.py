"""
Arm C — the §4.2-4.4 diagnostic probes. Shared runtime.

Answers AC1 and zNBu-W3 through the ladder constraint pre-committed in
REVIEWS.md: "Run the §4.2-4.4 diagnostics on the 900 models. If the SP gap is
still foreground-concentrated and the locus is unchanged at 900, the framework
is not a small-data artifact."

What this module owns
---------------------
Everything the four probes share: resolving a ladder cell's dump and
checkpoint, rebuilding that cell's model, the encoder/decoder split, the
inference loop that produces a CONTRACT.md §3 row, and the recovery ratio R.

Lifted, not reimplemented
-------------------------
`classify_parameters`, `freeze_encoder`, `freeze_decoder`, `reinit_decoder`
and `extract_logits` are copied verbatim from the published §4.4 code
(`python/final_loco/experiment_utils.py`, `experiment_b2_encoder_retrain.py`),
with only imports changed. Those functions produced Table 36; a probe that
localises the failure differently from the published run is not comparing
against it. The published files are NOT edited and NOT imported — importing
them would drag in their sys.path surgery (`_clean_model_paths`), which fights
the arch-tree import that armc_train.py already establishes.

Metrics are NOT lifted here. Every number this arm reports comes from
`shadowtransfer/metrics.py` through `shadowtransfer/armc_dump.py`, per
CONTRACT.md rule zero.

Ladder dumps live in two places
-------------------------------
`dumps/st/` normally, and `${ST_SCRATCH_ROOT}/dumps_armC_alt/st/` for the
one cell whose real dump path is quarantined
(`mamnet-vanilla-loco-r225-phoenix-hr-s0`; CLAUDE.md, "Quarantined paths").
`ladder_dump_dir` prefers whichever root carries a `.DONE`, so the quarantined
cell resolves without special-casing at the call sites.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_dump                             # noqa: E402

LADDER_DUMP_ROOTS = (
    BASE / "dumps" / "st",
    Path(os.environ.get("ST_SCRATCH_ROOT", "")) / "dumps_armC_alt/st",
)

ARCH_DIR = {"mamnet": "mamnet", "oglanet": "oglanet", "dinov3": "dinov3"}

# Encoder name prefixes, verbatim from experiment_utils.ENCODER_PREFIXES.
# `any(startswith)` makes these effectively 'encoder.' / 'backbone.': for
# MAMNet the MSCAF block and the aux head count as decoder, for OGLANet the
# DFFM/decoder/OAM stack does. That is the published §4.4 split.
ENCODER_PREFIXES = {
    "mamnet":  ["encoder.resnet_encoder.", "encoder.resnet.", "encoder."],
    "oglanet": ["encoder.resnet_encoder.", "encoder.resnet.", "encoder."],
    "dinov3":  ["backbone.dinov3.", "backbone."],
}

# The encoder's final feature map, keyed by what each encoder's forward returns.
ENCODER_FEATURE_KEY = {"mamnet": "feat5", "oglanet": "feat5",
                       "dinov3": "feat_block11"}

# The upper-bound comparator for each rung. 900 has no upper bound: every city
# holds exactly 450 training images, so LOCO-900 is scored against Upper-450
# and the run is flagged is_budget_asymmetric (REVIEWS.md, arm C).
COMPARATOR_RUNG = {225: 225, 450: 450, 900: 450}

# App G.9 claims a 100x encoder-LR reduction for the §4.4 Experiment B. The
# published runs actually used these ratios. Probe 3 applies the literal 100x
# and records both (spec, 2026-07-30).
PUBLISHED_ENCODER_LR_RATIO = {"mamnet": 1.0, "oglanet": 5.0, "dinov3": 5.0}

# Images whose headroom is below this are dropped from R: the ratio's
# denominator is (IoU_upper - IoU_LOCO) and a near-zero one manufactures
# arbitrarily large |R|.
R_EXCLUDE_MIN = 0.02

N_TEST = 150


def sha256_file(p):
    return armc_dump.sha256_file(p)


# ---------------------------------------------------------------------------
# Ladder cells
# ---------------------------------------------------------------------------

def ladder_run_id(arch, mode, rung, city, seed=0):
    method = "vanilla" if mode == "loco" else "none"
    return f"{arch}-{method}-{mode}-r{rung}-{city}-hr-s{seed}"


def ladder_dump_dir(run_id):
    """The completed dump for a ladder cell, from either root."""
    tried = []
    for root in LADDER_DUMP_ROOTS:
        d = root / run_id
        tried.append(str(d))
        if (d / ".DONE").exists():
            return d
    raise SystemExit(f"FATAL no completed ladder dump for {run_id}; tried {tried}")


def ladder_run_json(run_id):
    return json.loads((ladder_dump_dir(run_id) / "run.json").read_text())


def ladder_perimage(run_id):
    import pandas as pd
    return pd.read_parquet(ladder_dump_dir(run_id) / "perimage.parquet")


def ladder_checkpoint(run_id):
    """The cell's checkpoint_best.pth, taken from its own run.json."""
    p = Path(ladder_run_json(run_id)["ckpt_path"])
    if not p.exists():
        raise SystemExit(f"FATAL checkpoint recorded by {run_id} is missing: {p}")
    return p


# ---------------------------------------------------------------------------
# The architecture's own tree
# ---------------------------------------------------------------------------

def enter_arch_tree(arch):
    """Put the architecture's own package on sys.path and cd into it.

    Identical to armc_train.main()'s setup, including dropping any `utils`
    module already bound: shadowtransfer.metrics imports MAMNet's
    utils.contrast_utils, and leaving that binding would make the OGLANet and
    DINOv3 trees silently import MAMNet's losses and metrics.
    """
    arch_root = BASE / ARCH_DIR[arch]
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    for m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[m]
    return arch_root


def build_model(arch, cfg, device):
    """Rebuild a ladder cell's model from its own resolved config.

    `cfg` is the `config_verbatim` block of the cell's run.json — what the
    ladder actually resolved, defaults already filled. Construction mirrors
    each architecture's own Trainer.__init__ line for line.

    `pretrained=True` is forced, so the freshly built model carries the
    pretrained initialisation (ImageNet for the CNNs, the DINOv3 weights file
    for the ViT) before any checkpoint is loaded. Probe 3's Experiment B
    snapshots the encoder at that point rather than rebuilding a second model.
    """
    import torch
    if arch == "mamnet":
        from models.mamnet import MAMNet
        model = MAMNet(num_classes=cfg["num_classes"], pretrained=True,
                       use_aux=True, use_contrast=cfg["use_contrast"])
    elif arch == "oglanet":
        from models.oglanet import OGLANet
        model = OGLANet(num_classes=cfg["num_classes"], pretrained=True,
                        img_size=cfg["img_size"], use_contrast=cfg["use_contrast"])
    elif arch == "dinov3":
        from dinov3_model import DINOv3ShadowDetector
        model = DINOv3ShadowDetector(
            num_classes=cfg["num_classes"], model_name=cfg["model_name"],
            weights_path=cfg["weights_path"], pretrained=True,
            frozen_stages=cfg["frozen_stages"],
            freeze_backbone=cfg.get("freeze_backbone", False))
    else:
        raise SystemExit(f"FATAL unknown arch {arch}")
    return model.to(torch.device(device))


def load_ladder_weights(model, ckpt_path, device):
    """Load a ladder checkpoint's model weights. Nothing else from it is used:
    the optimizer, scheduler and best-metric state belong to the cell that is
    being probed, not to the probe."""
    import torch
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=True)
    epoch = int(ck.get("epoch", -1))
    del ck
    return {"ckpt_epoch": epoch,
            "missing_keys": list(missing), "unexpected_keys": list(unexpected)}


# ---------------------------------------------------------------------------
# Encoder / decoder split — lifted from the published §4.4 code
# ---------------------------------------------------------------------------

def classify_parameters(model, arch):
    """Verbatim from experiment_utils.classify_parameters."""
    prefixes = ENCODER_PREFIXES.get(arch, ["encoder.", "backbone."])
    encoder_params, decoder_params = [], []
    for name, param in model.named_parameters():
        is_enc = any(name.startswith(p) for p in prefixes)
        (encoder_params if is_enc else decoder_params).append((name, param))
    print(f"  Parameters — encoder: {len(encoder_params)}, "
          f"decoder: {len(decoder_params)}", flush=True)
    return encoder_params, decoder_params


def freeze_encoder(model, arch):
    """Verbatim from experiment_utils.freeze_encoder."""
    encoder_params, decoder_params = classify_parameters(model, arch)
    for _, param in encoder_params:
        param.requires_grad = False
    trainable = sum(1 for _, p in decoder_params if p.requires_grad)
    print(f"  Frozen: {len(encoder_params)} encoder params, "
          f"trainable: {trainable} decoder params", flush=True)
    return len(encoder_params)


def freeze_decoder(model, arch):
    """Verbatim from experiment_b2_encoder_retrain.freeze_decoder."""
    encoder_params, decoder_params = classify_parameters(model, arch)
    for _, param in decoder_params:
        param.requires_grad = False
    trainable = sum(1 for _, p in encoder_params if p.requires_grad)
    print(f"  Frozen: {len(decoder_params)} decoder params, "
          f"trainable: {trainable} encoder params", flush=True)
    return len(decoder_params)


def reinit_decoder(model, arch):
    """Verbatim from experiment_utils.reinit_decoder."""
    import torch.nn as nn
    _, decoder_params = classify_parameters(model, arch)
    decoder_names = {name for name, _ in decoder_params}

    reinit_count = 0
    for name, module in model.named_modules():
        module_params = {n for n, _ in module.named_parameters(prefix=name)}
        if not module_params.intersection(decoder_names):
            continue
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            reinit_count += 1
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            reinit_count += 1
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            reinit_count += 1

    print(f"  Re-initialized {reinit_count} decoder modules", flush=True)
    return reinit_count


def extract_logits(outputs):
    """Verbatim from experiment_utils.extract_logits."""
    import torch
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, dict):
        return outputs.get('main',
               outputs.get('p6',
               outputs.get('pred_fused',
               list(outputs.values())[0])))
    return outputs


def encoder_state(model, arch):
    """Encoder parameter AND buffer names, in state_dict order.

    Buffers matter: BatchNorm running statistics are part of what "the
    encoder" is, and Experiment B resets them along with the weights, exactly
    as experiment_b2_encoder_retrain.reinit_encoder_to_pretrained does.
    """
    prefixes = ENCODER_PREFIXES.get(arch, ["encoder.", "backbone."])
    names = set()
    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if any(name.startswith(p) for p in prefixes):
            names.add(name)
    return names


def snapshot_encoder(model, arch):
    sd = model.state_dict()
    return {k: sd[k].detach().clone() for k in encoder_state(model, arch)}


def restore_encoder(model, snap):
    sd = model.state_dict()
    n_p = 0
    for k, v in snap.items():
        if sd[k].shape != v.shape:
            raise SystemExit(f"FATAL encoder restore shape mismatch at {k}")
        sd[k] = v.clone()
        n_p += 1
    model.load_state_dict(sd)
    print(f"  Reset encoder to its pretrained initialisation: {n_p} entries",
          flush=True)
    return n_p


def encoder_forward(model, arch, x):
    """The encoder's final feature map, [B, C, h, w].

    Called as a submodule rather than through a forward hook so the tensor is
    unambiguous: both CNN encoders return a dict of stage outputs and the
    DINOv3 backbone returns a dict of intermediate blocks.
    """
    enc = model.backbone if arch == "dinov3" else model.encoder
    out = enc(x)
    if not isinstance(out, dict):
        return out
    key = ENCODER_FEATURE_KEY[arch]
    if key not in out:
        key = sorted(out)[-1]
    return out[key]


# ---------------------------------------------------------------------------
# Inference — the ladder's own recipe, so a probe dump is comparable to it
# ---------------------------------------------------------------------------

PREDICTION_RECIPE = (
    "argmax over the two class logits, then "
    "filter_small_predictions(min_pixels=10); identical to the "
    "architecture's own test() loop")


def infer_rows(model, loader, run_id, city, device, geo, res="hr"):
    """Run the ladder's inference recipe and return (rows, probs).

    Byte-for-byte the loop in armc_train.py: softmax channel 1 for the stored
    probability, argmax after the <10 px filter for the prediction, and
    armc_dump.perimage_row for every metric. Nothing is recomputed inline.
    """
    import torch
    from utils.postprocessing import filter_small_predictions

    model.eval()
    rows, probs = [], {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            fname = batch["filename"][0]
            stem = Path(fname).stem

            logits = extract_logits(model(images))
            prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
            filtered = filter_small_predictions(logits, min_pixels=10)
            pred = torch.argmax(filtered, dim=1)[0].cpu().numpy().astype(np.uint8)
            gt = masks[0].numpy().astype(np.uint8)

            rows.append(armc_dump.perimage_row(
                run_id, stem, city, res, gt, prob, pred, geo.get(fname)))
            probs[stem] = prob.astype(np.float16)
    return rows, probs


# ---------------------------------------------------------------------------
# Recovery ratio R
# ---------------------------------------------------------------------------

def recovery_ratio(iou_probe, iou_loco, iou_upper, exclude_min=R_EXCLUDE_MIN):
    """Per-image R = (probe - LOCO) / (upper - LOCO), NaN where headroom < 0.02.

    Arrays are aligned by the caller on image_id. The exclusion is on the
    DENOMINATOR only, per the arm spec: an image the LOCO model already scores
    as well as the within-city upper bound carries no recoverable gap, and its
    ratio is undefined rather than zero.
    """
    iou_probe = np.asarray(iou_probe, dtype=np.float64)
    iou_loco = np.asarray(iou_loco, dtype=np.float64)
    iou_upper = np.asarray(iou_upper, dtype=np.float64)
    head = iou_upper - iou_loco
    ok = head >= exclude_min
    r = np.full(len(head), np.nan)
    r[ok] = (iou_probe[ok] - iou_loco[ok]) / head[ok]
    return r, ok, head


def align_on_image_id(df_probe, df_other, col):
    """Values of `col` from df_other, ordered to match df_probe's image_id.

    Fails loudly on any id the two dumps do not share. A silent reindex here
    would compare an image against a different image and nothing downstream
    would notice.
    """
    m = dict(zip(df_other["image_id"], df_other[col]))
    missing = [i for i in df_probe["image_id"] if i not in m]
    if missing:
        raise SystemExit(f"FATAL {len(missing)} image_ids absent from the "
                         f"comparison dump, first: {missing[:3]}")
    return np.array([m[i] for i in df_probe["image_id"]], dtype=np.float64)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()
