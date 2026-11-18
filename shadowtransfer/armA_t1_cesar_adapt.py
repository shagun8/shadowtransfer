#!/usr/bin/env python3
"""ARM A TAIL SESSION / BLOCK 1 -- the CrossEarth adapter for the PUBLISHED arm F SAR.

`shadowtransfer/armf_sar_cells.py` IS NOT EDITED and `shadowtransfer/sar.py` and
`shadowtransfer/armf_sar_pass.py` are IMPORTED UNCHANGED. Everything CrossEarth
needs that those three cannot supply lives here, installed by `install()` as a
module-attribute rebinding -- the pattern `armA_wave2_probes.py` already uses.

D-B3-3 -- WHAT SAR CONSUMES AS "CROSSEARTH'S LOGITS"
----------------------------------------------------
CrossEarth exposes no logits. `ReinMask2FormerHead` inherits
`mmseg.models.decode_heads.mask2former_head.Mask2FormerHead.predict`, which
returns, verbatim from mmseg 2.1 `mask2former_head.py:160-162`:

    cls_score  = F.softmax(mask_cls_results, dim=-1)[..., :-1]
    mask_pred  = mask_pred_results.sigmoid()
    seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)

Those channels are non-negative and DO NOT SUM TO 1, which is why the project's
own probability definition (`armf_crossearth_runtime.foreground_prob`) is the
NORMALISED foreground channel and not a softmax of the two. `sar.adapt_image`,
by contrast, assumes `model(x)` returns RAW LOGITS: it minimises
`softmax_entropy(logits) = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)`
and reads `prob_fg = logits.softmax(1)[0, 1]` (`sar.py:103`, `sar.py:274`).

OPERATOR RULING (D-B3-3): SAR consumes `log(normalised score)`. The two-class
score map is normalised to sum to 1 per pixel, the log is taken, and that is
what `model(x)` returns. Then `softmax(1)` recovers the normalised score
EXACTLY, so SAR's softmax assumption is true rather than approximately true,
`softmax_entropy` is the model's real predictive entropy, `sar.e0_margin` and
`sar.reset_constant` are on that same entropy scale, and `prob_fg` equals
`foreground_prob` -- the probability definition every other CrossEarth cell in
this project uses, including this arm's own scoring. Declared as a deviation in
every run.json with this justification and the einsum quoted.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import sar                                     # noqa: E402
from shadowtransfer import armf_sar_pass                           # noqa: E402

ARCH = "crossearth"
NUM_CLASSES = 2
CROP = 384
SCORE_EPS = 1e-12
LOG_EPS = 1e-12

EINSUM_VERBATIM = (
    "cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]; "
    "mask_pred = mask_pred_results.sigmoid(); "
    "seg_logits = torch.einsum('bqc, bqhw->bchw', cls_score, mask_pred)   "
    "[mmseg/models/decode_heads/mask2former_head.py:160-162]")

LOGIT_DEVIATION = {
    "deviation": "SAR consumes log(normalised two-class score), not raw logits",
    "authority": "operator ruling D-B3-3 (2026-10-12)",
    "why": (
        "CrossEarth exposes no logits. ReinMask2FormerHead.predict returns "
        "non-negative scores that do not sum to 1, so the project's own "
        "probability definition (armf_crossearth_runtime.foreground_prob) is "
        "the NORMALISED foreground channel. sar.adapt_image softmaxes its "
        "input as raw logits and reads prob_fg = logits.softmax(1)[0,1]. "
        "Passing log(p) where p is the normalised score makes softmax(1) "
        "recover p exactly: SAR's softmax assumption becomes true rather than "
        "approximately true, softmax_entropy becomes the model's real "
        "predictive entropy (so e0_margin and reset_constant are on the right "
        "scale), and prob_fg equals foreground_prob. The project's "
        "probability definition is kept intact."),
    "einsum_verbatim": EINSUM_VERBATIM,
    "transform": "p = s / s.sum(1, keepdim=True).clamp_min(1e-12); return p.clamp_min(1e-12).log()",
    "score_eps": SCORE_EPS,
    "log_eps": LOG_EPS,
    "files_unedited": ["shadowtransfer/sar.py",
                       "shadowtransfer/armf_sar_pass.py",
                       "shadowtransfer/armf_sar_cells.py"],
}


# ---------------------------------------------------------------------------
# BRANCH 1 -- armf_sar_cells.build_model raises for crossearth
# ---------------------------------------------------------------------------
class CrossEarthSARModel(nn.Module):
    """`model(x) -> log(normalised two-class score)`, per D-B3-3.

    `x` is a float tensor [1, 3, H, W] holding RAW 0..255 RGB, because
    CrossEarth's `SegDataPreProcessor` does its own normalisation from
    mmcv `pixel_mean`/`pixel_std`. That is BRANCH 3's contract and this class
    is the only place that depends on it.

    The forward goes through `EncoderDecoder.encode_decode`, the one
    differentiable score path in the released checkout: `extract_feat` then
    `decode_head.predict`, no argmax and no no_grad. `data_samples` carries
    METAINFO ONLY and never a `gt_sem_seg`, which is a second, structural
    guarantee beside `assert_no_label_tensor_reaches_adaptation` that no label
    can enter the adaptation step.
    """

    def __init__(self, net, crop=CROP):
        super().__init__()
        self.net = net
        self.crop = int(crop)

    def _data_samples(self, h, w):
        from mmseg.structures import SegDataSample
        s = SegDataSample()
        s.set_metainfo(dict(img_shape=(h, w), ori_shape=(h, w),
                            pad_shape=(h, w), scale_factor=(1.0, 1.0),
                            reduce_zero_label=False))
        return [s]

    def forward(self, x):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        samples = self._data_samples(h, w)
        data = self.net.data_preprocessor(
            {"inputs": [x[i] for i in range(x.shape[0])],
             "data_samples": samples * int(x.shape[0])}, training=False)
        metas = [s.metainfo for s in data["data_samples"]]
        scores = self.net.encode_decode(data["inputs"], metas)
        # D-B3-3: normalise to a probability, then log. softmax(1) of this is
        # the normalised score exactly, which is foreground_prob's definition.
        p = scores / scores.sum(dim=1, keepdim=True).clamp_min(SCORE_EPS)
        return p.clamp_min(LOG_EPS).log()


def build_model_crossearth(arch, cfg, device):
    """CrossEarth's OWN constructor, through `armf_crossearth_runtime`.

    Resolved from CrossEarth's object graph, not carried from another arch:
    `armf_sar_cells.build_model` has a three-way `if` on mamnet/oglanet/dinov3
    and `raise SystemExit` otherwise, and none of those constructors exists
    here -- the model is built by `mmseg.registry.MODELS.build(cfg.model)`
    after `import CrossEarth` registers the four released classes.

    The three CrossEarth-only inputs travel in `cfg` under `_ce_*` keys, which
    the driver sets from the parent cell's own run.json and from the submit
    script's verified backbone path.
    """
    if arch != ARCH:
        raise SystemExit(f"FATAL build_model_crossearth called for arch {arch!r}")
    from shadowtransfer import armf_crossearth_runtime as cert

    repo_root = cfg.get("_ce_repo_root")
    backbone = cfg.get("_ce_backbone_ckpt")
    crop = int(cfg.get("_ce_crop", CROP))
    for k, v in (("_ce_repo_root", repo_root), ("_ce_backbone_ckpt", backbone)):
        if not v:
            raise SystemExit(f"FATAL cfg carries no {k}; the CrossEarth model "
                             f"cannot be built without it and will not guess")
    cert.add_repo_to_path(str(repo_root))
    ce_cfg = cert.build_cfg(str(repo_root), str(backbone),
                            num_classes=NUM_CLASSES, crop=crop)
    cert.patch_msda_fp32()
    net = cert.build_model(ce_cfg, device=str(device))
    nc = int(net.decode_head.num_classes)
    if nc != NUM_CLASSES:
        raise SystemExit(f"FATAL built decode_head.num_classes={nc} != {NUM_CLASSES}")
    model = CrossEarthSARModel(net, crop=crop).to(device)
    return model


# ---------------------------------------------------------------------------
# BRANCH 2 -- load_checkpoint_strict requires `model_state_dict`
# ---------------------------------------------------------------------------
def load_checkpoint_crossearth(model, ckpt_path, device):
    """CrossEarth's own loader, `armf_crossearth_engine.load_trainable_state_dict`.

    MEASURED on disk (2026-10-12), not assumed: every one of the nine
    crossearth vanilla LOCO r3500 checkpoints is 94,366,828 bytes with
    top-level keys ['best_decision_miou', 'epoch', 'state_dict', 'val'] and a
    303-key `state_dict` whose only prefixes are `backbone.reins.` and
    `decode_head.`. So `armf_sar_cells.load_checkpoint_strict` fails twice
    over: it looks for `model_state_dict`, which is absent, and it then
    requires a STRICT load, which cannot succeed because the frozen DINOv2
    trunk is deliberately never saved (`armf_crossearth_engine.
    trainable_state_dict`, which is what WROTE these files).

    The trunk is not missing data: `cert.build_model` -> `model.init_weights()`
    has already loaded it from the verified `dinov2_converted.pth`. So the
    faithful route is the engine's own `load_trainable_state_dict`, which loads
    non-strictly and then REFUSES any unexpected key and any missing
    rein/decode_head key -- i.e. it is strict about exactly the tensors the
    checkpoint claims to carry. Nothing about that contract is weakened here.
    """
    from shadowtransfer import armf_crossearth_engine as eng

    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "state_dict" not in ck:
        raise SystemExit(
            f"FATAL {ckpt_path} has no 'state_dict'; keys {sorted(ck)}. "
            f"CrossEarth checkpoints are written by "
            f"armf_crossearth_engine.trainable_state_dict and carry "
            f"'state_dict', never 'model_state_dict'.")
    inner = model.net if isinstance(model, CrossEarthSARModel) else model
    rep = eng.load_trainable_state_dict(inner, ck["state_dict"])
    epoch = ck.get("epoch")
    val = ck.get("val")
    n_keys = len(ck["state_dict"])
    del ck
    LOAD_REPORT.clear()
    LOAD_REPORT.update({
        "route": ("armf_crossearth_engine.load_trainable_state_dict on the "
                  "wrapped segmentor, after cert.build_model's init_weights() "
                  "loaded the frozen DINOv2 trunk"),
        "top_level_key": "state_dict",
        "n_checkpoint_keys": int(n_keys),
        "n_loaded": rep["n_loaded"],
        "n_missing_frozen": rep["n_missing_frozen"],
        "missing_are_frozen_trunk_only": True,
        "unexpected_keys": 0,
        "checkpoint_epoch": epoch,
        "checkpoint_val": val,
        "why_not_load_checkpoint_strict": (
            "armf_sar_cells.load_checkpoint_strict demands a "
            "'model_state_dict' key and a strict load; a CrossEarth "
            "checkpoint carries 'state_dict' and holds reins + decode_head "
            "only, because the frozen trunk is never re-saved."),
    })
    return epoch


#: Filled by `load_checkpoint_crossearth`, read by the driver for run.json.
LOAD_REPORT = {}


# ---------------------------------------------------------------------------
# BRANCH 3 -- _images_from_patch_ids builds CNN-normalised images
# ---------------------------------------------------------------------------
def images_from_patch_ids_crossearth(cfg, patch_ids, split, label, limit,
                                     img_root, mask_root, patch_index,
                                     splits_dir):
    """Raw uint8 patches, through the SAME path every CrossEarth S-EO cell uses.

    `armA_a4_sar_seo._images_from_patch_ids` builds ImageNet-normalised float
    tensors via `armA_phase3_train.build_split`, which is right for the three
    CNN/ViT architectures and wrong here: CrossEarth's `SegDataPreProcessor`
    normalises from mmcv `pixel_mean`/`pixel_std` on a 0..255 input.

    Resolved from CrossEarth's own graph rather than by rescaling: this is
    `armA_pilot_train.build_split(..., use_contrast=False)` wrapped in
    `seo_crossearth.SEOCrossEarthEval`, which is literally what
    `armA_wave0_crossearth_train.py:666,697` builds its test set from. The
    recovery is exact and self-asserting (`recover_uint8` checks the inverse
    normalisation's residual on every item), and `seo_dataset.py` is not
    edited, so `loader_module_sha` still matches the parent cells.

    THE GROUND TRUTH IS `mask`, PLAIN {0,1}, WHICH IS WHAT THE OTHER 27 SAR
    CELLS SCORE ON. `SEODataset` returns `mask` plain and `mask_ignore` with
    255; `armA_a4_sar_seo` reads `s["mask"]`, so its `valid = gt != 255` line
    is inert and SAR's fg-only IoU K=2 counts DSM-unseen pixels as background
    on all four architectures. Reproduced here deliberately: changing it for
    crossearth alone would make this cell's number incomparable with the
    mamnet/oglanet/dinov3 SAR cells. Recorded in run.json as
    `unseen_handling`, not corrected.
    """
    import armA_pilot_train as pilot
    from shadowtransfer import seo_crossearth as adapt
    from shadowtransfer.armf_sar_cells import coverage_band

    idx = (json.loads(Path(patch_index).read_text())
           if not isinstance(patch_index, dict) else patch_index)
    base, rows, _bases, _parts = pilot.build_split(
        img_root, mask_root, idx, splits_dir, split, list(patch_ids),
        False, label, False)
    ds = adapt.SEOCrossEarthEval(base, limit=(limit or None))

    out = []
    for i in range(len(ds)):
        s = ds[i]
        gt = np.asarray(s["gt"], dtype=np.uint8)
        tf = float(gt.mean())
        r = rows[i]
        out.append({
            "image_id": f"{r['image_id']}_y{r['y']}_x{r['x']}",
            # float32 0..255, [1,3,384,384]: the preprocessor's own contract.
            "x": torch.from_numpy(np.ascontiguousarray(s["img"])).float().unsqueeze(0),
            "gt": gt,
            "true_fg_fraction": tf,
            "coverage_band": coverage_band(tf),
        })
    if not out:
        raise SystemExit(f"FATAL {label}: resolved to 0 images")
    IMAGE_REPORT.clear()
    IMAGE_REPORT.update({
        "route": ("armA_pilot_train.build_split(use_contrast=False) -> "
                  "seo_crossearth.SEOCrossEarthEval, the path "
                  "armA_wave0_crossearth_train.py builds its test set from"),
        "image_dtype": "float32 holding raw 0..255 RGB",
        "image_recovery": adapt.IMAGE_RECOVERY,
        "image_recovery_max_residual": float(ds.max_residual),
        "recovery_tol": adapt.RECOVERY_TOL,
        "n_materialised": len(out),
        "gt_source": "SEODataset['mask'], plain {0,1}",
        "unseen_handling": (
            "DSM-unseen pixels are NOT excluded. SEODataset returns `mask` "
            "plain and `mask_ignore` with 255; armA_a4_sar_seo reads "
            "s['mask'], so its `valid = gt != 255` test is inert and all 27 "
            "existing SAR cells score with unseen pixels counted as "
            "background. Reproduced here so the crossearth SAR cell is "
            "comparable with them. Reported, not corrected."),
    })
    return out


#: Filled by `images_from_patch_ids_crossearth`, read by the driver.
IMAGE_REPORT = {}


# ---------------------------------------------------------------------------
# BRANCH 4 -- THE ONE THE 2026-10-10 LOG DID NOT NAME, AND THE DANGEROUS ONE
# ---------------------------------------------------------------------------
#: `ReinsDinoVisionTransformer.state_dict` (reins_dinov2.py:47-54) POPS every
#: key without "rein" in it. `sar.snapshot_state` is
#: `{k: v.clone() for k, v in model.state_dict().items()}`, so on CrossEarth a
#: snapshot silently omits the entire frozen DINOv2 trunk. Two consequences,
#: both silent:
#:
#:   1. `sar.restore_state` is `load_state_dict(snap, strict=True)` and would
#:      RAISE on the first image, because every trunk key is missing.
#:   2. Worse, if it did not raise: `sar.collect_params` collects LayerNorm
#:      affines across the WHOLE model minus upstream's skip list
#:      ("layer4", "blocks.9", "blocks.10", "blocks.11"), so
#:      `backbone.blocks.N.norm1/norm2` for every N outside {9,10,11} IS
#:      collected and IS adapted -- and none of those tensors is in the
#:      snapshot, so they would never be restored. Adaptation would leak
#:      across images while `sar.state_matches`, which also reads
#:      `state_dict()`, reported episodic isolation intact. That is the
#:      shape of a plausible wrong number.
#:
#: FIX: snapshot and restore over `named_parameters()` + `named_buffers()`,
#: which are separate traversals the override does not touch. This does not
#: change what SAR adapts, it makes arm F's own episodic-isolation contract
#: TRUE on this architecture -- over a strictly LARGER tensor set than the
#: three CNN/ViT architectures get. Byte-identity is still arm F's
#: `sar._bytes_equal`, imported, not reimplemented.
SNAPSHOT_DEVIATION = {
    "deviation": ("episodic snapshot/restore/compare go over "
                  "named_parameters()+named_buffers(), not state_dict()"),
    "why": (
        "ReinsDinoVisionTransformer.state_dict pops every non-rein key, so "
        "sar.snapshot_state would omit the frozen DINOv2 trunk entirely. "
        "sar.restore_state's strict load would raise on image 0; and the "
        "trunk LayerNorms that sar.collect_params DOES collect and adapt "
        "would never be restored, while sar.state_matches -- which reads the "
        "same truncated state_dict -- reported isolation intact."),
    "effect": ("makes arm F's episodic-isolation assertion true on CrossEarth "
               "over a LARGER tensor set than the CNNs get; does not change "
               "which parameters SAR adapts"),
    "byte_compare": "sar._bytes_equal, imported unchanged",
    "installed_as": ["sar.snapshot_state", "sar.restore_state",
                     "sar.state_matches"],
}


def _full_state(model):
    snap = {}
    for n, p in model.named_parameters():
        snap[f"P:{n}"] = p.detach().clone()
    for n, b in model.named_buffers():
        if b is not None:
            snap[f"B:{n}"] = b.detach().clone()
    return snap


def full_snapshot_state(model):
    return _full_state(model)


def full_restore_state(model, snap):
    params = dict(model.named_parameters())
    bufs = dict(model.named_buffers())
    with torch.no_grad():
        for k, v in snap.items():
            kind, name = k.split(":", 1)
            t = params.get(name) if kind == "P" else bufs.get(name)
            if t is None:
                raise SystemExit(
                    f"FATAL episodic restore: snapshot key {k} has no live "
                    f"tensor on the model; the snapshot and the model "
                    f"disagree and the pass would not be episodic.")
            t.copy_(v)


def full_state_matches(model, snap):
    cur = _full_state(model)
    if set(cur) != set(snap):
        return False, sorted(set(cur) ^ set(snap))
    bad = [k for k in cur if not sar._bytes_equal(cur[k], snap[k])]
    return (not bad), bad


def assert_snapshot_covers_collector(model):
    """Every parameter SAR will adapt must be IN the snapshot. This is the
    machine check for BRANCH 4 and it runs in the smoke before any pass."""
    _p, names, skipped = sar.collect_params(model)
    snap = full_snapshot_state(model)
    published = set(model.state_dict().keys())
    missing_published = sorted(n for n in names if n not in published)
    missing_full = sorted(n for n in names if f"P:{n}" not in snap)
    rep = {
        "n_collected": len(names),
        "n_skipped_norm_modules": len(skipped),
        "n_keys_published_state_dict": len(published),
        "n_keys_full_snapshot": len(snap),
        "n_params": sum(1 for _ in model.named_parameters()),
        "n_buffers": sum(1 for _ in model.named_buffers()),
        "n_collected_absent_from_state_dict": len(missing_published),
        "collected_absent_from_state_dict_sample": missing_published[:8],
        "n_collected_absent_from_full_snapshot": len(missing_full),
        "collected_absent_from_full_snapshot": missing_full,
        "verdict": "PASS" if not missing_full else "FAIL",
    }
    if missing_full:
        raise SystemExit(
            f"FATAL BRANCH-4 assertion: {len(missing_full)} collected "
            f"parameters are absent from the full snapshot "
            f"({missing_full[:5]}); the pass would not be episodic.")
    return rep


def collector_report(model):
    """What SAR's collector is exposed to on CrossEarth. REPORTED, NOT JUDGED.

    The dinov3 comparability flag applies and is restated per cell: upstream's
    skip list ("layer4", "blocks.9", "blocks.10", "blocks.11") was written for
    a 12-block ViT-Base, where it removes the last three blocks. CrossEarth's
    trunk is DINOv2 ViT-L with 24 blocks, so the same three substrings land on
    blocks 9/10/11 -- the middle of the trunk -- and the last three blocks
    (21/22/23) are collected. Transferred as written, exactly as it was for
    DINOv3 ViT-S/16. No judgement, no change.
    """
    _p, names, skipped = sar.collect_params(model)
    n_total = sum(p.numel() for p in model.parameters())
    coll = {n for n in names}
    n_coll_el = sum(p.numel() for n, p in model.named_parameters() if n in coll)
    blocks = sorted({int(n.split("blocks.")[1].split(".")[0])
                     for n in names if "blocks." in n})
    top = sorted({n.split(".")[1] if n.startswith("net.") else n.split(".")[0]
                  for n in names})
    return {
        "n_collected_params": len(names),
        "n_collected_elements": int(n_coll_el),
        "n_model_elements": int(n_total),
        "collected_param_names": names,
        "skipped_norm_modules": skipped,
        "skip_substrings": list(sar.SKIP_SUBSTRINGS),
        "collected_top_level": top,
        "trunk_blocks_collected": blocks,
        "comparability_flag": (
            "upstream's skip list is ViT-Base shaped (12 blocks, removes the "
            "last three). CrossEarth's trunk is DINOv2 ViT-L with 24 blocks, "
            "so blocks.9/10/11 are mid-trunk and blocks 21/22/23 ARE "
            "collected. The same transfer-as-written applies as for DINOv3 "
            "ViT-S/16. Reported, not judged."),
    }


# ---------------------------------------------------------------------------
# cell resolution and arch-tree entry -- neither has a crossearth path
# ---------------------------------------------------------------------------
def vanilla_run_id(fold, seed):
    return f"{ARCH}-vanilla-loco-r3500-{fold.lower()}-hr-s{seed}"


def sar_run_id(fold, seed):
    return f"{ARCH}-sar-loco-r3500-{fold.lower()}-hr-s{seed}"


def resolve_vanilla_cell(fold, seed, vanilla_dumps, repo_root, backbone_ckpt,
                         crop=CROP):
    """`armA_a4_sar_seo.resolve_vanilla_cell` refuses every crossearth cell.

    MEASURED, not assumed: its final guard is
    `if "num_classes" not in cfg: raise SystemExit(... refusing to guess the
    architecture)`, and the nine crossearth vanilla cells' `config_verbatim`
    holds {batch_size, crop, deform_attn_fp32, epochs_max, frozen_backbone,
    iters_max, iters_per_epoch, lr, numpy_trapezoid_shim, optimizer,
    patience_epochs, polylr_end, polylr_eta_min, polylr_power, precision,
    xformers_disabled} -- no num_classes, because a CrossEarth cell's class
    count lives in the released mmseg config, not in its args.

    So num_classes is NOT guessed: it is taken from the arm's own CrossEarth
    constructor, `armf_crossearth_runtime.build_cfg(num_classes=2)`, which is
    the call that built the parent cell, and the built model's
    `decode_head.num_classes` is asserted against it in build_model_crossearth.
    """
    rid = vanilla_run_id(fold, seed)
    rj = Path(vanilla_dumps) / rid / "run.json"
    if not rj.exists():
        raise SystemExit(f"FATAL SAR needs the vanilla LOCO cell {rid}; its "
                         f"run.json is absent at {rj}. SAR adapts an existing "
                         f"checkpoint and cannot create one.")
    run = json.loads(rj.read_text())
    ckpt = run.get("ckpt_path")
    if not ckpt or not Path(ckpt).exists():
        raise SystemExit(f"FATAL {rid}: ckpt_path {ckpt!r} from its run.json "
                         f"does not exist on disk.")
    cfg = dict(run.get("config_verbatim") or {})
    cfg["_ce_repo_root"] = str(repo_root)
    cfg["_ce_backbone_ckpt"] = str(backbone_ckpt)
    cfg["_ce_crop"] = int(cfg.get("crop", crop))
    prov = {
        "num_classes_source": (
            "armf_crossearth_runtime.build_cfg(num_classes=2) -- the arm's own "
            "CrossEarth constructor, the same call that built the parent cell. "
            "The parent's config_verbatim carries no num_classes key, which is "
            "why armA_a4_sar_seo.resolve_vanilla_cell refuses crossearth."),
        "num_classes_asserted_on_built_model": True,
        "config_verbatim_keys": sorted(k for k in cfg if not k.startswith("_ce_")),
        "ce_keys_added": ["_ce_repo_root", "_ce_backbone_ckpt", "_ce_crop"],
    }
    return Path(ckpt), cfg, run, prov


def enter_arch_tree(repo_root):
    """`armf_sar_cells.enter_arch_tree` is `armc_train.ARCH_DIR[arch]`, and
    ARCH_DIR is exactly {"mamnet","oglanet","dinov3"} (armc_train.py:109), so
    crossearth KeyErrors. CrossEarth has no arch tree under python/ at all:
    its code is the external checkout, entered by
    `armf_crossearth_runtime.add_repo_to_path`. No chdir and no utils/data/
    models purge, because nothing is imported from an arch tree here.
    """
    from shadowtransfer import armf_crossearth_runtime as cert
    cert.add_repo_to_path(str(repo_root))
    return {"route": "armf_crossearth_runtime.add_repo_to_path",
            "repo_root": str(repo_root),
            "why_not_armf_sar_cells_enter_arch_tree": (
                "it indexes armc_train.ARCH_DIR, which holds mamnet/oglanet/"
                "dinov3 only; CrossEarth has no python/<arch> tree")}


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------
def install():
    """Rebind the four module attributes `armf_sar_pass.run_pass` reaches.

    `armf_sar_pass` did `from ... import build_model, load_checkpoint_strict`,
    so the names live in ITS namespace and are rebound there. `sar.adapt_image`
    resolves `restore_state` as a module global at call time, so rebinding on
    `sar` reaches the model-recovery path too. No file is edited.
    """
    rep = {
        "armf_sar_pass.build_model": "build_model_crossearth",
        "armf_sar_pass.load_checkpoint_strict": "load_checkpoint_crossearth",
        "sar.snapshot_state": "full_snapshot_state",
        "sar.restore_state": "full_restore_state",
        "sar.state_matches": "full_state_matches",
    }
    armf_sar_pass.build_model = build_model_crossearth
    armf_sar_pass.load_checkpoint_strict = load_checkpoint_crossearth
    sar.snapshot_state = full_snapshot_state
    sar.restore_state = full_restore_state
    sar.state_matches = full_state_matches
    return {"rebindings": rep,
            "files_unedited": LOGIT_DEVIATION["files_unedited"],
            "logit_deviation": LOGIT_DEVIATION,
            "snapshot_deviation": SNAPSHOT_DEVIATION}
