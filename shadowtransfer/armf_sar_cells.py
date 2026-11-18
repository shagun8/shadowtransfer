"""
Arm F SAR close-out — cell resolution, model construction and image loading.

Shared by `python/armf_sar_lr_pilot.py` (stage 1, source-city validation) and
`python/armf_sar_grid.py` (stage 2, held-out-city test).

Lifted from `python/armf_sar_pilot.py`, which stays frozen as the record of the
2026-07-30 gate session (its reproduction gate was a hard stop; the operator has
since removed that gate, so the file is superseded rather than edited). Three
additions here: OGLANet, a source-city validation image set, and arch-tree entry
as a callable so one driver serves any architecture.

THE SOURCE-CITY VALIDATION SPLIT IS RUNG-INDEPENDENT
----------------------------------------------------
`armc_train.resolve_selection` draws the LOCO validation set as
`split_rung(N_VAL=150, 2)` = 75 + 75 images from the fold's two training
cities, by a seeded permutation that does not depend on the rung. `source_val_pairs`
reproduces exactly that draw without touching the train selection, so no rung
has to be named to get the same 150 images the published cells validated on.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import armc_train                                          # noqa: E402
from shadowtransfer.metrics import per_image_counts        # noqa: E402

ARM = "F_sar"
WEAKNESS_IDS = ["AC3", "v1UJ-W3"]
RES_DIR = {"hr": "highres", "mr": "midres"}
NUM_CLASSES = 2
BANDS = ("lt05", "05to15", "15to30", "ge30")

# The divergence guard of the arm F spec: a candidate whose final-step predicted
# foreground fraction falls below this on more than 20% of validation images is
# marked diverged and excluded from selection.
FG_COLLAPSE_FRACTION = 0.01
FG_COLLAPSE_IMAGE_SHARE = 0.20

# Config A, the only configuration run in this arm.
NORM_MODE = "eval_source_stats_affine_adapted"


def coverage_band(frac):
    """True-foreground coverage band, cut at 0.05 / 0.15 / 0.30."""
    if frac < 0.05:
        return "lt05"
    if frac < 0.15:
        return "05to15"
    if frac < 0.30:
        return "15to30"
    return "ge30"


def legacy_run_dir(arch, city, res_dir):
    """The published VANILLA LOCO cell. `vanilla` carries no token in the legacy
    directory name; it is the bare form. This is a run_dir, never a run_id."""
    return f"{arch}_loco_holdout_{city}_{res_dir}_1"


def resolve_cell(arch, city, res_dir):
    """(cell_dir, checkpoint, args.json, test_results.json). Every path is
    reported absolute; none is guessed at read time."""
    d = BASE / "data" / arch / "outputs" / legacy_run_dir(arch, city, res_dir)
    ck, cfgp, tr = d / "checkpoint_best.pth", d / "args.json", d / "test_results.json"
    for p in (ck, cfgp):
        if not p.exists():
            # A path archived to Fortress is not data loss. Raises
            # ArchivedToFortress with the recall instruction when an
            # .ARCHIVED_FORTRESS marker covers it (2026-09-22 archive);
            # returns a plain message otherwise, so the original SystemExit
            # below still fires for a genuinely missing path.
            from shadowtransfer import fortress_recall
            fortress_recall.check(p, what="published vanilla LOCO checkpoint")
            raise SystemExit(f"FATAL missing {p}")
    return d, ck, cfgp, tr


def stored_fg_iou_k2(tr_path):
    """The cell's own stored foreground-only IoU at K=2,
    `detailed.boundary_tolerant.tolerant_2px.iou` (CONTRACT.md section 8).
    Tables 30-32 are NOT this value (erratum E-7) and are not read.

    Returns (value_or_None, reason). NEVER raises and never gates: the operator
    removed the reproduction stop on 2026-07-30, so a missing or disagreeing
    stored value is recorded and warned about, not acted on.
    """
    p = Path(tr_path)
    if not p.exists():
        return None, f"no stored test_results.json at {p}"
    try:
        t = json.loads(p.read_text())
        return float(t["detailed"]["boundary_tolerant"]["tolerant_2px"]["iou"]), None
    except (KeyError, TypeError, ValueError) as e:
        return None, f"{p} carries no readable detailed.boundary_tolerant.tolerant_2px.iou ({e})"


def build_model(arch, cfg, device):
    """The architecture's own class, then a strict load of the published
    checkpoint. `pretrained=False` on the CNNs avoids a torchvision fetch from a
    compute node; every weight is replaced by the checkpoint immediately
    afterwards and the strict load is what proves the architecture matches.
    DINOv3 has no non-pretrained construction path, so it builds from the local
    weights file — a disk read, not a network one."""
    if arch == "mamnet":
        from models.mamnet import MAMNet
        model = MAMNet(num_classes=cfg.get("num_classes", NUM_CLASSES),
                       pretrained=False, use_aux=True,
                       use_contrast=cfg.get("use_contrast", False))
    elif arch == "oglanet":
        from models.oglanet import OGLANet
        model = OGLANet(num_classes=cfg.get("num_classes", NUM_CLASSES),
                        pretrained=False,
                        img_size=cfg.get("img_size", 384),
                        use_contrast=cfg.get("use_contrast", True))
    elif arch == "dinov3":
        from dinov3_model import DINOv3ShadowDetector
        model = DINOv3ShadowDetector(
            num_classes=cfg.get("num_classes", NUM_CLASSES),
            model_name=cfg.get("model_name", "dinov3_vits16"),
            weights_path=str(armc_train.DINOV3_WEIGHTS),
            pretrained=True,
            frozen_stages=cfg.get("frozen_stages", -1),
            freeze_backbone=cfg.get("freeze_backbone", False))
    else:
        raise SystemExit(f"FATAL unsupported arch {arch}")
    return model.to(device)


def load_checkpoint_strict(model, ckpt_path, device):
    import torch
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "model_state_dict" not in ck:
        raise SystemExit(f"FATAL {ckpt_path} has no model_state_dict; "
                         f"keys {sorted(ck)}")
    out = model.load_state_dict(ck["model_state_dict"], strict=True)
    missing = getattr(out, "missing_keys", [])
    unexpected = getattr(out, "unexpected_keys", [])
    if missing or unexpected:
        raise SystemExit(f"FATAL non-strict load: missing={missing} "
                         f"unexpected={unexpected}")
    epoch = ck.get("epoch")
    del ck
    return epoch


def band_counts(pred, gt):
    """Raw TP/FP/FN/TN at K=0, K=2 and K=5, from the one sanctioned metric
    module. `per_image_counts(K)` returns the strict (K=0) population alongside
    the K population, so two calls cover all three band widths. Counts are free
    and let the band be revisited without a rerun."""
    c2 = per_image_counts(pred, gt, boundary_tolerance=2)
    c5 = per_image_counts(pred, gt, boundary_tolerance=5)
    return {
        "tp_k0": int(c2["tp_str"]), "fp_k0": int(c2["fp_str"]),
        "fn_k0": int(c2["fn_str"]), "tn_k0": int(c2["tn_str"]),
        "tp_k2": int(c2["tp_tol"]), "fp_k2": int(c2["fp_tol"]),
        "fn_k2": int(c2["fn_tol"]), "tn_k2": int(c2["tn_tol"]),
        "tp_k5": int(c5["tp_tol"]), "fp_k5": int(c5["fp_tol"]),
        "fn_k5": int(c5["fn_tol"]), "tn_k5": int(c5["tn_tol"]),
        "n_valid_px_k2": int(c2["n_valid_px"]),
    }


def enter_arch_tree(arch):
    """Put the architecture's own tree on sys.path and chdir into it, dropping
    any cached `utils`/`data`/`models` package so a second architecture in the
    same process cannot import the first one's modules. Every driver here runs
    exactly ONE architecture per job, so this is called once; the purge is kept
    because a silent cross-arch import would produce a plausible wrong number.
    """
    root = BASE / armc_train.ARCH_DIR[arch]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)
    for m in [k for k in sys.modules
              if k in ("utils", "data", "models")
              or k.startswith(("utils.", "data.", "models."))]:
        del sys.modules[m]
    return root


def _images_from_pairs(arch, pairs, split, res_dir, limit):
    """Materialise one image set once, so every pass is compared on
    byte-identical inputs. Augmentation off, batch size 1, the architecture's
    own dataset class and transforms."""
    ds = armc_train.build_dataset(arch, pairs, split, res_dir, 384, False)
    if limit:
        for attr in ("img_files", "img_paths", "mask_paths"):
            setattr(ds, attr, getattr(ds, attr)[:limit])
    out = []
    for i in range(len(ds)):
        s = ds[i]
        gt = np.squeeze(s["mask"].numpy())
        gt = (gt > 0).astype(np.uint8)
        tf = float(gt.mean())
        out.append({
            "image_id": Path(str(s["filename"])).stem,
            "x": s["image"].unsqueeze(0),
            "gt": gt,
            "true_fg_fraction": tf,
            "coverage_band": coverage_band(tf),
        })
    if not out:
        raise SystemExit(f"FATAL {split} set resolved to 0 images")
    return out


def test_images(arch, city, res_dir, limit=0):
    """The held-out city's test split in the fixed released test-list order."""
    pairs = [(city, f) for f in armc_train.city_split_files(city, res_dir, "test")]
    return _images_from_pairs(arch, pairs, "test", res_dir, limit)


def source_val_pairs(holdout, res_dir, seed=0):
    """The fold's SOURCE-CITY validation draw: 75 + 75 images from the two
    training cities, identical to what `armc_train.resolve_selection` returns
    for `val`, and independent of the rung. No held-out-city image is touched.
    Returns (pairs, source_cities, n_per_city)."""
    fold = armc_train.FOLD_OF_HOLDOUT[holdout]
    src = armc_train.LOCO_FOLDS[fold]["train"]
    if armc_train.LOCO_FOLDS[fold]["test"] != holdout:
        raise SystemExit("FATAL fold table disagrees with holdout")
    n_each = armc_train.split_rung(armc_train.N_VAL, len(src))
    pairs = [(c, f) for c, n in zip(src, n_each)
             for f in armc_train.take_prefix(
                 armc_train.city_split_files(c, res_dir, "val"), n, c, "val", seed)]
    if any(c == holdout for c, _ in pairs):
        raise SystemExit(f"FATAL source-city validation draw contains the "
                         f"held-out city {holdout}")
    return pairs, list(src), dict(zip(src, n_each))


def source_val_images(arch, holdout, res_dir, seed=0, limit=0):
    pairs, src, n_per_city = source_val_pairs(holdout, res_dir, seed)
    images = _images_from_pairs(arch, pairs, "val", res_dir, limit)
    return images, src, n_per_city
