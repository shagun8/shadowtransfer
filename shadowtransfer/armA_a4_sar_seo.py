"""
ARM A SESSION A4 / BLOCK 2 -- S-EO adapter for the PUBLISHED arm F SAR.

WHAT IS NEW HERE AND WHAT IS NOT
--------------------------------
NOTHING about SAR itself is new. `shadowtransfer/sar.py` (E_0 = 0.4*ln(C) =
0.2773 at C=2, reset_constant 0.020069, EMA momentum 0.9, SAM rho 0.05,
lr_base 2.5e-4, BatchNorm running buffers set to None in eval) and
`shadowtransfer/armf_sar_pass.py` (`run_pass`, the episodic byte-compare, the
divergence guard) are IMPORTED UNCHANGED and are not copied into this file.
The operator's Block 2 spec names exactly those constants, and they are read
from that module rather than restated here so the two cannot drift.

What this module supplies is the ONE thing arm F's SAR could not know: where
S-EO's images and checkpoints are. `armf_sar_cells.py` resolves the published
ShadowTransfer LOCO cells over chicago/miami/phoenix from `armc_train`. S-EO
is a different dataset -- folds OMA/JAX/UCSD, 384 px patches cut at native
GSD, a budget manifest instead of a released test list -- so cell resolution
and image materialisation are re-implemented against `seo_budget`/
`seo_dataset` and `armA_phase3_train.build_split`, which is what every arm A
S-EO cell already trains and evaluates through.

`build_model`, `load_checkpoint_strict`, `band_counts` and `enter_arch_tree`
ARE reused from `armf_sar_cells` verbatim: they are keyed on the architecture
and a config dict, not on the dataset, and the three architectures are the
same three. Reusing them is what keeps a SAR number on S-EO comparable to a
SAR number on ShadowTransfer.

TARGET ACCESS -- deployment_time, AND WHY NO LABEL CAN REACH THE UPDATE
-----------------------------------------------------------------------
The grid adapts on the HELD-OUT CITY'S TEST STREAM. That is `deployment_time`
access, and it is recorded as such. It is NOT `target_labels`: the adaptation
objective is entropy of the model's own prediction and nothing else.

`assert_no_label_tensor_reaches_adaptation()` is the machine check, and it is
run in every job before the first image. It reads the SIGNATURE of
`sar.adapt_image` and refuses to continue if any parameter could carry a
label. The signature is quoted verbatim into run.json so a reader does not
have to take this docstring's word for it.
"""

import inspect
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import sar                                    # noqa: E402
from shadowtransfer import seo_budget                             # noqa: E402
from shadowtransfer.armf_sar_cells import (                       # noqa: E402
    band_counts, build_model, coverage_band, enter_arch_tree,
    load_checkpoint_strict)

ARM = "A_a4_sar"
DATASET = "seo"
NUM_CLASSES = 2
CITIES = ("OMA", "JAX", "UCSD")
RUNG = 3500
RES = "hr"
SELECTION_SPLIT = "source_city_val"

# The pilot ladder, exactly the published stage-1 shape.
LR_MULTIPLIERS = (1.0, 10.0, 100.0)
STEP_LADDER = (1, 10)

# Default roots. Every one is overridable from the CLI so a smoke can be
# pointed somewhere disposable.
DEF_VANILLA_DUMPS = os.path.join(os.environ["ST_SCRATCH_ROOT"], "dumps/seo")
DEF_BUDGET_DIR = os.path.join(os.environ["ST_SCRATCH_ROOT"], "data/seo_budget_acq3")
DEF_SPLITS_DIR = os.path.join(os.environ["ST_SCRATCH_ROOT"], "data/seo_splits_acq3")
DEF_IMG_ROOT = os.path.join(os.environ["ST_SCRATCH_ROOT"], "data/seo_raw/pansharpened")
DEF_MASK_ROOT = os.path.join(os.environ["ST_SCRATCH_ROOT"], "data/seo_raw/shadows_max")
DEF_PATCH_INDEX = (os.path.join(os.environ["PROJECT_ROOT"], "logs") + "/"
                   "armA_seo_patch_index_acq3.json")


# --------------------------------------------------------------------------
# the label assertion
# --------------------------------------------------------------------------
_LABELISH = ("target", "label", "labels", "y", "gt", "mask", "masks",
             "ground_truth", "annotation")


def assert_no_label_tensor_reaches_adaptation():
    """Refuse to adapt if `sar.adapt_image` could take a label.

    This is a SIGNATURE check, not a comment: the adaptation entry point is
    the only place a gradient is taken, and if no parameter of it can carry a
    label then no label can enter the update no matter what the caller holds.
    The caller additionally passes `im["x"]` alone; `im["gt"]` is used after
    the fact, for scoring only.
    """
    sig = inspect.signature(sar.adapt_image)
    quoted = f"sar.adapt_image{sig}"
    bad = [p for p in sig.parameters if p.lower() in _LABELISH]
    if bad:
        raise SystemExit(
            f"FATAL target-access assertion: {quoted} has label-carrying "
            f"parameter(s) {bad}. The grid claims deployment_time access and "
            f"reads NO labels during adaptation; that claim is not "
            f"defensible against this signature, so the job stops.")
    return {
        "assertion": "no label tensor reaches the adaptation step",
        "checked": "signature of shadowtransfer.sar.adapt_image",
        "signature_verbatim": quoted,
        "label_like_names_rejected": list(_LABELISH),
        "result": "PASS -- no parameter of the adaptation entry point can "
                  "carry a label; the caller passes im['x'] only, and im['gt'] "
                  "is consumed after the fact for scoring",
    }


# --------------------------------------------------------------------------
# cell resolution
# --------------------------------------------------------------------------
def vanilla_run_id(arch, fold, seed):
    return f"{arch}-vanilla-loco-r{RUNG}-{fold.lower()}-hr-s{seed}"


def sar_run_id(arch, fold, seed):
    return f"{arch}-sar-loco-r{RUNG}-{fold.lower()}-hr-s{seed}"


def resolve_vanilla_cell(arch, fold, seed, vanilla_dumps=DEF_VANILLA_DUMPS):
    """The vanilla LOCO cell SAR adapts. Returns (ckpt_path, cfg, run_json).

    SAR trains nothing. It starts from an EXISTING arm A S-EO vanilla LOCO
    checkpoint and the recipe config that produced it, both taken from that
    cell's own run.json rather than re-derived, so the adapted model is
    provably the published one plus SAR's affine updates.
    """
    rid = vanilla_run_id(arch, fold, seed)
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
    cfg = run.get("config_verbatim") or {}
    if "num_classes" not in cfg:
        raise SystemExit(f"FATAL {rid}: run.json config_verbatim carries no "
                         f"num_classes; refusing to guess the architecture.")
    return Path(ckpt), cfg, run


def load_cell_manifest(fold, budget_dir=DEF_BUDGET_DIR):
    """This fold's S-EO budget manifest -- the same patch-id lists every arm A
    S-EO cell trains and tests on. Read, never rebuilt: rebuilding it here
    would be a second copy of the split rule."""
    for name in (f"{fold}_loco.json", f"{fold.lower()}_loco.json",
                 f"{fold}.json", f"{fold.lower()}.json"):
        p = Path(budget_dir) / name
        if p.exists():
            return json.loads(p.read_text()), str(p)
    have = sorted(q.name for q in Path(budget_dir).glob("*.json"))
    raise SystemExit(f"FATAL no budget manifest for fold {fold} in "
                     f"{budget_dir}; present: {have}")


def source_cities_of(fold):
    src = [c for c in CITIES if c != fold]
    if len(src) != 2:
        raise SystemExit(f"FATAL fold {fold} is not one of {CITIES}")
    return src


# --------------------------------------------------------------------------
# image materialisation -- the `run_pass` contract
# --------------------------------------------------------------------------
def _images_from_patch_ids(arch, cfg, patch_ids, split, label, limit,
                           img_root, mask_root, patch_index, splits_dir):
    """Materialise one image set ONCE, so every pass in a job is compared on
    byte-identical inputs -- the arm F requirement, carried over.

    Built through `armA_phase3_train.build_split`, which is the same function
    the arm A S-EO trainer uses, so the transforms, the contrast channel and
    the valid-region logic are the cell's own and not a second implementation.
    """
    import numpy as np
    import armA_phase3_train as p3

    ds, rows, _bases, _parts = p3.build_split(
        img_root=img_root, mask_root=mask_root, patch_index=patch_index,
        splits_dir=splits_dir, split=split, patch_ids=list(patch_ids),
        augment=False, label=label,
        use_contrast=bool(cfg.get("use_contrast", False)), fda=None)

    n = len(ds) if limit in (0, None) else min(limit, len(ds))
    out = []
    for i in range(n):
        s = ds[i]
        gt = np.squeeze(s["mask"].numpy())
        valid = gt != 255
        fg = ((gt == 1) & valid).astype(np.uint8)
        tf = float(fg[valid].mean()) if valid.any() else 0.0
        r = rows[i]
        out.append({
            "image_id": f"{r['image_id']}_y{r['y']}_x{r['x']}",
            "x": s["image"].unsqueeze(0),
            "gt": gt.astype(np.uint8),
            "true_fg_fraction": tf,
            "coverage_band": coverage_band(tf),
        })
    if not out:
        raise SystemExit(f"FATAL {label}: resolved to 0 images")
    return out


def source_val_images(arch, cfg, fold, limit=0, budget_dir=DEF_BUDGET_DIR,
                      **roots):
    """The fold's SOURCE-CITY validation split -- 450 + 450 from the two
    training cities, which is `seo_budget.N_VAL` = 900 split evenly.

    THE SELECTION SPLIT, AND THE LEAK CHECK THAT GUARDS IT. Every patch id is
    asserted to come from a source city. No held-out-city image is loaded, no
    held-out-city number is computed, and none can enter the selection.
    """
    cell, cell_path = load_cell_manifest(fold, budget_dir)
    src = source_cities_of(fold)
    ids = list(cell["val"])
    import armA_pilot_train as P
    bad = [p for p in ids if P.city_of(p) == fold]
    if bad:
        raise SystemExit(
            f"FATAL LEAK: {len(bad)} of {len(ids)} source-city validation "
            f"patch ids belong to the HELD-OUT city {fold} (first: {bad[:3]}). "
            f"The pilot selects on this split; it stops rather than select "
            f"against the target.")
    per_city = {}
    for p in ids:
        per_city[P.city_of(p)] = per_city.get(P.city_of(p), 0) + 1
    imgs = _images_from_patch_ids(arch, cfg, ids, "val",
                                  f"{fold}/source_city_val", limit, **roots)
    meta = {"split": SELECTION_SPLIT, "fold": fold, "source_cities": src,
            "n_per_city": per_city, "n_ids": len(ids),
            "n_materialised": len(imgs), "budget_manifest": cell_path,
            "leak_check": f"PASS -- 0 of {len(ids)} ids belong to {fold}"}
    return imgs, meta


def target_test_images(arch, cfg, fold, limit=0, budget_dir=DEF_BUDGET_DIR,
                       **roots):
    """The HELD-OUT city's FULL test split, in budget-manifest order.

    This is the stream the grid adapts on, and adapting on it is exactly what
    makes the grid `deployment_time`. The images carry labels in `gt` because
    the pass has to be SCORED; `assert_no_label_tensor_reaches_adaptation`
    is what separates scoring from training on them.
    """
    cell, cell_path = load_cell_manifest(fold, budget_dir)
    ids = list(cell["test"])
    import armA_pilot_train as P
    wrong = [p for p in ids if P.city_of(p) != fold]
    if wrong:
        raise SystemExit(f"FATAL {fold} test split contains {len(wrong)} ids "
                         f"from other cities (first: {wrong[:3]})")
    imgs = _images_from_patch_ids(arch, cfg, ids, "test", f"{fold}/test",
                                  limit, **roots)
    meta = {"split": "target_city_test", "fold": fold, "n_ids": len(ids),
            "n_materialised": len(imgs), "budget_manifest": cell_path,
            "access": "deployment_time"}
    return imgs, meta


def roots_from_args(args):
    return dict(img_root=args.img_root, mask_root=args.mask_root,
                patch_index=args.patch_index, splits_dir=args.splits_dir)


def sar_constants_block():
    """Every constant the operator's Block 2 spec names, READ FROM sar.py."""
    return {
        "source_module": "shadowtransfer/sar.py",
        "batch_size": 1,
        "ema_momentum": sar.EMA_MOMENTUM,
        "sam_rho": sar.SAM_RHO,
        "e0_margin": sar.e0_margin(NUM_CLASSES),
        "reset_constant": sar.reset_constant(NUM_CLASSES),
        "reset_constant_derivation": sar.reset_constant_derivation(NUM_CLASSES),
        "lr_base": sar.SAR_LR_BASE,
        "optimizer": "SAM wrapping SGD, momentum 0.9",
        "bn_buffers": "None (batch statistics in eval mode)",
        "num_classes": NUM_CLASSES,
    }
