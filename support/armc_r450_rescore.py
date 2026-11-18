#!/usr/bin/env python3
"""ARM C r450 published recompute -- RE-SCORE the remaining 61 cells.

CPU only. No GPU, no checkpoint load, no inference. Reads STORED probability
maps plus the GT masks and writes dumps/st_r450_published_recompute/{run_id}/
in the SAME schema and with the SAME scoring code as the 17 cells already in
that directory: shadowtransfer.armc_dump.perimage_row for every row and
shadowtransfer.metrics for every quantity. Nothing here computes a metric.

WHAT THIS ADDS OVER armc_r450.py (which handled only vanilla/base)
------------------------------------------------------------------
1. A METHOD axis. armc_r450.py hardcoded method="vanilla"/variant="vanilla"
   for loco and "none"/"base" for upper. The 54 adaptation-baseline cells and
   the 6 published-SIB (C4clean) cells need the method carried explicitly.
2. A SOURCE axis. The 55 Test_img_probs cells read
   data/Test_img_probs/{mode}/{city}/{res}/{arch}/{variant}/. The 6 C4clean
   cells read data/{arch}/sp_gap_results/c4clean_probs_{arch}_{city}_{res}/,
   which is a different tree with a different writer.
3. A RUN-DIR RESOLVER THAT CHECKS THE FILES IT NEEDS.  This is the fix for the
   July crash. armc_r450.py's resolve_run_dir returned the first root carrying
   args.json, and /projects' copy of oglanet_miami_highres_1 is a PARTIAL copy:
   args.json and checkpoint_best.pth are there, test_results.json is not. So
   the resolver handed back /projects and repro_target_from_test_results then
   exited 1 -- job 20553148, the only failure of the 18. The complete copy was
   on /scratch the whole time. resolve_run_dir below scores each root on the
   full required-file set and records which root won and why.
4. TWO PREDICTION RECIPES, both reported.  The published classic cells were
   scored after filter_small_predictions(min_pixels=10) (run_inference.py), and
   that is what the 17 finished cells use, so it stays the primary recipe and
   the dump columns are directly comparable to them. The published SIB test
   path takes a plain argmax with NO size filter
   (train_oglanet_sib.py:test_and_save_predictions), and sp_gap_analysis.py's
   per-image `miou` is strict K=0 two-class on (p > 0.5) with no filter. Both
   recipes are therefore computed for every cell and both are written; each
   gate names the recipe it used. Neither is adjusted to match the other.

THE CHECKS, per the operator's spec
-----------------------------------
* Table 38 is the per-image mean of two-class boundary-tolerant mIoU at K=2.
  Verified, not assumed: on the 12 finished mamnet+dinov3 cells
  info_miou_k2_twoclass reproduces the printed Table 38 value to within
  0.0077, and all 5 finished oglanet cells sit inside 0.311. Tolerance is
  0.01, widened to 1.12 for oglanet -- erratum E-9, OGLANet inference is
  nondeterministic and every published OGLANet number is one draw.
* The cell's own stored test_results.json, foreground-only at K=2, within
  0.01 (CONTRACT.md section 8). Reported, never adjusted. Note this gate
  already FAILS on 11 of the 17 finished cells, so a failure here is a
  continuation of a recorded finding, not a new one.
* Usable-image count under every definition in play, because 112 and 113 are
  both in circulation for miami and they come from different floors.

n_train / n_val
---------------
NOT guessed and NOT inherited. The 2026-07-28 operator authorization that set
a uniform n_train=450 / n_val=150 was scoped to "the arm C r450 recompute (18
cells)" and these 61 are not those 18. Each cell reports what its own run
recorded: the oglanet C4clean cells' train.log says 900/300 explicitly, so
those three carry rung=null plus a rung_label_anomaly block rather than a
rung-450 label the evidence contradicts. Where nothing recorded it, the field
is null with n_train_val_source saying so -- a null is recoverable downstream,
a wrong value is not (CONTRACT.md section 3). --force_n_train/--force_n_val
exist for an explicit operator override and are recorded as cli_override.
"""

import argparse
import ast
import json
import os
import re
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_dump                        # noqa: E402
from shadowtransfer.armc_dump import sha256_file            # noqa: E402
from shadowtransfer.metrics import (                        # noqa: E402
    per_image_counts,
    threshold_and_filter,
)

SCRATCH_BASE = Path(os.environ["ST_SCRATCH_ROOT"])
FOLD_OF_HOLDOUT = {"phoenix": 0, "miami": 1, "chicago": 2}
RES_DIR = {"hr": "highres", "mr": "midres"}
CITIES = ("chicago", "miami", "phoenix")

# Printed Table 38 (final_paper.md, appendix I), per-cell tolerant mIoU,
# transcribed CHI / MIA / PHX. This is the per-image mean of the two-class
# K=2 value; see the module docstring for how that was established.
TABLE38 = {
    "upper":     {"mamnet": (86.31, 79.11, 73.75),
                  "oglanet": (79.12, 73.89, 66.15),
                  "dinov3": (80.27, 76.00, 67.82)},
    "vanilla":   {"mamnet": (79.72, 74.64, 62.84),
                  "oglanet": (68.47, 66.18, 61.18),
                  "dinov3": (67.26, 72.61, 65.24)},
    "fda":       {"mamnet": (81.88, 74.21, 67.77),
                  "oglanet": (71.07, 67.30, 63.27),
                  "dinov3": (72.26, 70.81, 58.72)},
    "segdesic":  {"mamnet": (82.44, 73.14, 58.80),
                  "oglanet": (66.47, 59.90, 61.17),
                  "dinov3": (75.52, 71.18, 68.04)},
    "iim":       {"mamnet": (81.06, 70.05, 55.43),
                  "oglanet": (70.01, 66.36, 55.25),
                  "dinov3": (77.43, 72.23, 56.51)},
    "isw":       {"mamnet": (75.23, 67.74, 51.08),
                  "oglanet": (64.19, 67.78, 59.69),
                  "dinov3": (57.60, 59.86, 55.68)},
    "mrfp_plus": {"mamnet": (75.76, 72.80, 53.35),
                  "oglanet": (63.82, 57.30, 58.03),
                  "dinov3": (77.46, 73.20, 60.78)},
    "fada":      {"mamnet": (79.59, 60.90, 62.30),
                  "oglanet": (66.17, 65.20, 60.26),
                  "dinov3": (70.15, 71.43, 62.81)},
    "sibc4":     {"mamnet": (81.65, 74.54, 68.64),
                  "oglanet": (73.05, 66.71, 63.97),
                  "dinov3": (76.57, 71.71, 61.45)},
}

# Erratum E-9: OGLANet inference is nondeterministic, cell-mean SD up to
# 0.4247 mIoU over repeated passes of the same checkpoint. 1.12 is the spread
# the operator specified for this family of checks.
T38_TOL_DEFAULT = 0.01
T38_TOL_OGLANET = 1.12

REPRO_GATE_TOL = 0.01

# From results/target_access_audit.json (arm I, repo_commit 054b2493), which
# is the authoritative per-method audit. Not re-derived here.
TARGET_ACCESS = {
    ("vanilla", "loco"): ("source_only",
        "arm I audit: train split built from the two source cities only"),
    ("fda", "loco"): ("unlabeled_target",
        "arm I audit: FDA amplitude donors sampled from the held-out city's "
        "train/images; target images reach preprocessing and training"),
    ("segdesic", "loco"): ("unlabeled_target",
        "arm I audit: held-out-city images reach training"),
    ("iim", "loco"): ("source_only", "arm I audit: no held-out-city data"),
    ("isw", "loco"): ("source_only", "arm I audit: no held-out-city data"),
    ("mrfp_plus", "loco"): ("source_only", "arm I audit: no held-out-city data"),
    ("fada", "loco"): ("source_only", "arm I audit: no held-out-city data"),
    ("none", "upper"): ("target_labels",
        "within-city upper bound: trained on the held-out city's own labelled "
        "train split; arm I audit lists train, val and model_selection"),
}

# SIB/C4clean splits by architecture -- erratum I-1. The CNN SIB configs set
# fda_target_root to the held-out city's directory ROOT, so the FDA donor walk
# reaches that city's images and its ground-truth PNGs; DINOv3's SIB config has
# use_fda=false and fda_target_root=null, so it touches nothing.
TARGET_ACCESS_SIBC4 = {
    # Operator ruling 2026-10-26 (n450 decision 3): source_only, per the
    # 2026-08-03 OGLANet-SIB access verdict. Was "unlabeled_target" (erratum
    # I-1 reading of the declared fda_target_root).
    "oglanet": ("source_only",
        "2026-08-03 access verdict, applied by operator ruling 2026-10-26: "
        "train_oglanet_sib.py declares fda_target_root and never dereferences "
        "it; get_dataloaders_sib cross-mixes the two SOURCE cities' train sets "
        "for FDA; 0/111 OGLANet SIB logs show a target-domain load. No "
        "held-out-city data is read in training."),
    "dinov3": ("source_only",
        "erratum I-1 does not apply to dinov3: this cell's args.json has "
        "use_fda=false and fda_target_root=null, so no held-out-city data is "
        "read anywhere in the pipeline."),
    "mamnet": ("unlabeled_target",
        "erratum I-1, CNN branch: FDA donor pool is the held-out city root."),
}


# ---------------------------------------------------------------------------
# cell geometry
# ---------------------------------------------------------------------------

def legacy_run_dir_for(arch, method, mode, city, res_dir):
    """The published output directory name for a cell. Not a run_id."""
    if mode == "upper":
        return f"{arch}_{city}_{res_dir}_1"
    if method == "vanilla":
        return f"{arch}_loco_holdout_{city}_{res_dir}_1"
    if method == "sibc4":
        if arch == "oglanet":
            return (f"oglanet_sib_C4clean_haar_vib_sag_fda_ctr__"
                    f"loco_holdout_{city}__{res_dir}")
        if arch == "dinov3":
            return f"dinov3_sib_C4clean_haar_vib_loco_holdout_{city}_{res_dir}_1"
        return f"mamnet_sib_C4clean_haar_vib_loco_holdout_{city}_{res_dir}_1"
    return f"{arch}_{method}_loco_holdout_{city}_{res_dir}_1"


def prob_dir_for(arch, method, mode, city, res_dir):
    """Where the stored probability maps live, and the variant token used."""
    if method == "sibc4":
        return (BASE / "data" / arch / "sp_gap_results"
                / f"c4clean_probs_{arch}_{city}_{res_dir}"), "c4clean"
    variant = "base" if mode == "upper" else method
    return (BASE / "data/Test_img_probs" / mode / city / res_dir / arch
            / variant), variant


def mask_dir_for(mode, city, res_dir):
    """GT masks. Verified 2026-10-25: for all three cities the Final_data_loco
    test/masks tree and the Final_data_test one are byte-identical (md5 over
    the concatenated 150 PNGs matches), so this choice cannot move a number.
    armc_r450.py's convention is kept so the 61 new cells and the 17 finished
    ones name the same source.
    """
    if mode == "upper":
        return BASE / "data/Final_data_test" / city / res_dir / "test/masks"
    fold = f"fold_{FOLD_OF_HOLDOUT[city]}_holdout_{city}"
    return BASE / "data/Final_data_loco" / fold / res_dir / "test/masks"


def resolve_run_dir(arch, legacy_run_dir, required):
    """Pick the root whose copy of the run directory actually has what we need.

    THE FIX FOR JOB 20553148. /projects holds partial copies: its
    oglanet_miami_highres_1 has args.json and the checkpoint but no
    test_results.json, and oglanet_sib_C4clean_...chicago__highres has no
    /projects entry at all. Scoring roots on the full required set, instead of
    on args.json alone, is what stops a partial copy from winning.

    Returns (dir, root_name, root_path, found, missing).
    """
    attempts = []
    for name, root in (("projects", BASE), ("scratch", SCRATCH_BASE)):
        d = root / "data" / arch / "outputs" / legacy_run_dir
        if not d.is_dir():
            attempts.append((name, root, d, [], list(required)))
            continue
        found = [f for f in required if (d / f).is_file()]
        missing = [f for f in required if f not in found]
        attempts.append((name, root, d, found, missing))
        if not missing:
            return d, name, str(root), found, missing
    # nothing complete: hand back the richest attempt so the error names it
    attempts.sort(key=lambda a: -len(a[3]))
    name, root, d, found, missing = attempts[0]
    raise SystemExit(
        "FATAL no root has a complete copy of "
        f"{legacy_run_dir}\n  required: {list(required)}\n" +
        "".join(f"  {n:9s} exists={d2.is_dir()} found={f2} missing={m2}\n"
                for n, _, d2, f2, m2 in attempts))


def read_config(run_dir):
    """(cfg, source, path). args.json / config.json where they exist; the
    oglanet C4clean cells have neither, and their train.log's `Args: {...}`
    line is the run's own record of its configuration."""
    for fname in ("args.json", "config.json"):
        p = run_dir / fname
        if p.is_file():
            return json.loads(p.read_text()), fname, p
    p = run_dir / "train.log"
    if p.is_file():
        for line in p.read_text(errors="replace").splitlines()[:200]:
            m = re.search(r"Args:\s*(\{.*\})\s*$", line)
            if m:
                return ast.literal_eval(m.group(1)), "train.log:Args", p
    return None, "unavailable", None


def repro_target(run_dir, family):
    """The cell's own stored foreground-only K=2 IoU, CONTRACT.md section 8.

    Two schemas. The classic trainers nest it under
    detailed.boundary_tolerant.tolerant_2px.iou; the SIB trainer writes a flat
    tolerant_{K}px block whose Shadow_IOU is the same quantity (per-image mean
    foreground IoU at K=2, _average_metrics over _compute_tolerant_metrics).
    """
    p = run_dir / "test_results.json"
    if not p.is_file():
        raise SystemExit(f"FATAL no test_results.json at {p}")
    t = json.loads(p.read_text())
    if family == "sib":
        try:
            v = t["tolerant_2px"]["Shadow_IOU"]
            field = "tolerant_2px.Shadow_IOU"
        except (KeyError, TypeError):
            raise SystemExit(f"FATAL {p}: no tolerant_2px.Shadow_IOU; "
                             f"keys {sorted(t)}")
    else:
        try:
            v = t["detailed"]["boundary_tolerant"]["tolerant_2px"]["iou"]
            field = "detailed.boundary_tolerant.tolerant_2px.iou"
        except (KeyError, TypeError):
            raise SystemExit(f"FATAL {p}: no detailed.boundary_tolerant."
                             f"tolerant_2px.iou; keys {sorted(t)}")
    return float(v), field, str(p), sha256_file(p)


def published_n_train_val(cfg, cfg_source, run_dir):
    """What this cell's own run recorded. No inference, no inheritance."""
    if cfg_source == "train.log:Args" or (run_dir / "train.log").is_file():
        p = run_dir / "train.log"
        if p.is_file():
            for line in p.read_text(errors="replace").splitlines()[:400]:
                m = re.search(r"Train:\s*(\d+)\s*samples\s+Val:\s*(\d+)", line)
                if m:
                    return (int(m.group(1)), int(m.group(2)),
                            "train.log: 'Train: N samples  Val: M'")
    if cfg:
        for k in ("n_train", "num_train", "train_size", "n_train_images"):
            if k in cfg:
                for kv in ("n_val", "num_val", "val_size", "n_val_images"):
                    if kv in cfg:
                        return cfg[k], cfg[kv], f"{cfg_source}:{k}/{kv}"
    return None, None, "unrecorded"


# ===========================================================================
# MEASURED TRAINING-SET SIZE  (operator condition 2, 2026-10-26: do not
# override; measure from each cell's own training log or split files)
# ===========================================================================
#
# Two sources, in this order of authority:
#
#   1. The cell's own train.log line `Train: N samples  Val: M`. Only the three
#      oglanet C4clean cells still have a train.log. Direct measurement.
#   2. The split files on disk, combined with the dataloader path the cell's own
#      trainer takes. A split-file count ALONE cannot separate 450 from 900: the
#      pool is 450 train / 150 val PER CITY, a LOCO cell draws on two cities,
#      and whether it subsamples is decided by `samples_per_dir` in code. So the
#      count is measured from disk and the multiplier is read from the source.
#
# The audit, verified by reading each trainer (2026-10-26):
#
#   mamnet, every method, use_contrast=true (all 18 cells here):
#       train.py / train_isw.py / train_segdesic.py / train_iim.py /
#       train_mrfp.py / train_fada.py build ShadowDatasetEnhanced DIRECTLY over
#       both source-city roots. mamnet/data/dataset_enhanced.py's
#       ShadowDatasetEnhanced has NO samples_per_dir parameter at all (0
#       occurrences in the file), so no subsampling is possible.
#       -> 2 x 450 = 900 train, 2 x 150 = 300 val.
#   oglanet, fda/iim/isw/mrfp_plus/fada, use_contrast=true:
#       routed through data/dataset.py get_dataloaders, which computes
#       samples_per_dir = single_city_size // len(train_cities) = 450 // 2 = 225.
#       -> 2 x 225 = 450 train. Val reuses the SAME samples_per_dir against a
#       150-image val dir, and the dataset takes min(225, 150) = 150 per city.
#       -> 300 val.
#   oglanet, segdesic: train_segdesic.py's use_contrast branch builds
#       ShadowDatasetEnhanced DIRECTLY, like mamnet -> 900 / 300. This is the
#       one OGLANet method that does not match its siblings.
#   oglanet, sibc4: data/dataset_sib.py get_dataloaders_sib ConcatDatasets both
#       cities' full datasets and contains NO samples_per_dir (0 occurrences)
#       -> 900 / 300. Confirmed directly by train.log.
#   dinov3, every method: use_contrast is absent from all 18 configs (DINOv3 has
#       no contrast channel), so every cell goes through get_dataloaders with
#       samples_per_dir = 225 -> 450 / 300.
#   dinov3, sibc4: data/dataset_sib.py get_dataloaders_sib DOES subsample
#       (samples_per_dir = n // len(train_cities)) -> 450 / 300.
#   mode=upper (the oglanet miami cell): mode='single', one city root,
#       samples_per_dir=None -> 450 train / 150 val.
#
# Cross-check, independent of the audit: at batch_size 8 with drop_last,
# 900 images is 112 batches/epoch and 450 is 56. armc_r450.py's docstring
# records exactly 112 for mamnet LOCO and 56 for oglanet/dinov3 LOCO and for
# every upper cell, read off the published runs' own logs; the oglanet C4clean
# train.log still on disk prints `Epoch 1 [0/112]`. The audit reproduces that
# evidence rather than contradicting it.
#
# NOTE ON A STANDING INSTRUCTION. CLAUDE.md records a 2026-07-28 operator
# instruction that on-disk directory contents are not authoritative for
# n_train/n_val "now or later", because some split directories were
# repopulated by later non-ShadowTransfer work. The 2026-10-26 instruction
# asks for split files to be used. The newer instruction is followed; the
# conflict is recorded here and in run.json's n_train_val_caveat so a future
# reader sees both. Nothing downstream of this field gates on it.

SPLIT_COUNT_CACHE = {}


def count_split(path):
    """Number of image files in one split directory. Measured, not assumed."""
    key = str(path)
    if key not in SPLIT_COUNT_CACHE:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        d = Path(path)
        SPLIT_COUNT_CACHE[key] = (
            sum(1 for f in d.iterdir() if f.suffix.lower() in exts)
            if d.is_dir() else None)
    return SPLIT_COUNT_CACHE[key]


# (arch, method) -> "pool" (load both cities whole) or "subsample" (225 each)
LOADER_PATH = {}
for _a, _ms, _p in (
    ("mamnet", ("fda", "segdesic", "iim", "isw", "mrfp_plus", "fada"), "pool"),
    ("oglanet", ("fda", "iim", "isw", "mrfp_plus", "fada"), "subsample"),
    ("oglanet", ("segdesic", "sibc4"), "pool"),
    ("dinov3", ("fda", "segdesic", "iim", "isw", "mrfp_plus", "fada", "sibc4"),
     "subsample"),
):
    for _m in _ms:
        LOADER_PATH[(_a, _m)] = _p

LOADER_EVIDENCE = {
    "pool": ("the trainer builds its dataset over both source-city roots with "
             "no samples_per_dir, so both full train pools are loaded"),
    "subsample": ("the trainer routes through get_dataloaders, which sets "
                  "samples_per_dir = single_city_size // n_train_cities = "
                  "450 // 2 = 225 per city"),
    "single": ("mode='single': one city root, samples_per_dir=None"),
}


def measure_n_train_val(arch, method, mode, city, res_dir, run_dir):
    """(n_train, n_val, source, detail). Measured; never inherited."""
    # 1. the cell's own training log, where one survives
    p = run_dir / "train.log"
    if p.is_file():
        for line in p.read_text(errors="replace").splitlines()[:400]:
            m = re.search(r"Train:\s*(\d+)\s*samples\s+Val:\s*(\d+)", line)
            if m:
                return (int(m.group(1)), int(m.group(2)),
                        "train.log (this cell's own run record)",
                        f"line: {line.strip()[:160]}")

    # 2. split files on disk x the audited loader path
    if mode == "upper":
        root = BASE / "data/Final_data_test" / city / res_dir
        tr = count_split(root / "train/images")
        va = count_split(root / "val/images")
        return (tr, va, "split_files x code_path_audit",
                f"mode=single, one city root {root.relative_to(BASE)}: "
                f"train={tr} val={va}; {LOADER_EVIDENCE['single']}")

    fold = FOLD_OF_HOLDOUT[city]
    src = [c for c in CITIES if c != city]
    per_tr = [count_split(BASE / "data/Final_data_test" / c / res_dir / "train/images")
              for c in src]
    per_va = [count_split(BASE / "data/Final_data_test" / c / res_dir / "val/images")
              for c in src]
    if any(v is None for v in per_tr + per_va):
        return (None, None, "unmeasurable",
                f"a source-city split directory is missing under "
                f"data/Final_data_test/{{{','.join(src)}}}/{res_dir}")
    path = LOADER_PATH.get((arch, method))
    if path == "pool":
        tr = sum(per_tr)
        va = sum(per_va)
    elif path == "subsample":
        spd = per_tr[0] // len(src)
        tr = spd * len(src)
        va = sum(min(spd, v) for v in per_va)
    else:
        return (None, None, "unmeasurable",
                f"no audited loader path for ({arch}, {method})")
    return (tr, va, "split_files x code_path_audit",
            f"fold_{fold}, source cities {src}: per-city train {per_tr}, "
            f"per-city val {per_va}; loader path '{path}' -- "
            f"{LOADER_EVIDENCE[path]}")


def two_class(tp, fp, tn, fn):
    eps = 1e-10
    fg = tp / (tp + fp + fn + eps)
    bg = tn / (tn + fp + fn + eps)
    return fg * 100.0, bg * 100.0, (fg + bg) / 2.0 * 100.0


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["mamnet", "oglanet", "dinov3"])
    ap.add_argument("--method", required=True,
                    choices=["none", "vanilla", "fda", "segdesic", "iim",
                             "isw", "mrfp_plus", "fada", "sibc4"])
    ap.add_argument("--mode", required=True, choices=["loco", "upper"])
    ap.add_argument("--city", required=True, choices=list(CITIES))
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--out_root", default=None,
                    help="dump root; default dumps/st_r450_published_recompute")
    ap.add_argument("--force_n_train", type=int, default=None,
                    help="operator override only; recorded as cli_override")
    ap.add_argument("--force_n_val", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="stage-1 smoke: first N")
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    t0 = time.time()
    res_dir = RES_DIR[args.res]
    arch, method, mode, city = args.arch, args.method, args.mode, args.city

    expect = f"{arch}-{method}-{mode}-r450-{city}-{args.res}-s0"
    if args.run_id != expect:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {expect}")
    if (mode == "upper") != (method == "none"):
        raise SystemExit(f"FATAL mode={mode} with method={method}: upper takes "
                         f"method 'none', loco takes a named method")

    family = "sib" if method == "sibc4" else "classic"
    t38_key = "upper" if mode == "upper" else method
    t38 = TABLE38[t38_key][arch][CITIES.index(city)]
    t38_tol = T38_TOL_OGLANET if arch == "oglanet" else T38_TOL_DEFAULT

    prob_dir, variant = prob_dir_for(arch, method, mode, city, res_dir)
    mask_dir = mask_dir_for(mode, city, res_dir)
    legacy_run_dir = legacy_run_dir_for(arch, method, mode, city, res_dir)
    required = ["test_results.json"]
    run_dir, config_root, config_root_path, found, _ = resolve_run_dir(
        arch, legacy_run_dir, required)
    cfg, cfg_source, cfg_path = read_config(run_dir)

    if args.force_n_train is not None and args.force_n_val is not None:
        n_train, n_val = args.force_n_train, args.force_n_val
        n_src, n_detail = "cli_override", "operator override on the command line"
    else:
        n_train, n_val, n_src, n_detail = measure_n_train_val(
            arch, method, mode, city, res_dir, run_dir)

    declared_rung = 450
    # Three states, not two. A MEASURED n_train that contradicts 450 is
    # evidence against the rung label and writes rung=null plus an anomaly
    # block. An UNRECORDED n_train is absence of evidence, not contradiction:
    # these are the published-protocol cells, paper section 3.3 states 450, and
    # the 17 finished cells in this same directory carry rung=450, so writing
    # null here would make 54 of 61 cells non-comparable with them over a
    # provenance gap rather than a disagreement. The rung is written 450 with
    # rung_provenance saying exactly what backs it.
    if n_train is None:
        rung_value = declared_rung
        rung_provenance = ("declared; this cell's own config records no "
                           "n_train and nothing was inferred. Paper section "
                           "3.3 states the 450-image protocol and the 17 "
                           "previously finished cells in this dump root carry "
                           "rung=450.")
    elif n_train == declared_rung:
        rung_value = declared_rung
        rung_provenance = f"measured and agrees: n_train={n_train} via {n_src}"
    else:
        rung_value = None
        rung_provenance = (f"CONTRADICTED: measured n_train={n_train} via "
                           f"{n_src}; rung written null")
    rung_ok = (rung_value is not None)
    anomaly = None
    if n_train is not None and n_train != declared_rung:
        anomaly = {
            "declared_rung": declared_rung,
            "measured_n_train": n_train,
            "measured_n_val": n_val,
            "evidence_source": n_src,
            "consequence": ("this cell is not a rung-450 cell; rung is written "
                            "null so the ladder cannot silently absorb it. The "
                            "run_id keeps the r450 token because the operator "
                            "named the output directory, and the token is not "
                            "evidence."),
            "scope_note": ("the 2026-07-28 uniform n_train=450/n_val=150 "
                           "operator authorization was scoped to the 18-cell "
                           "arm C r450 recompute and is NOT applied here"),
        }

    ta_key = (method, mode)
    if family == "sib":
        target_access, ta_detail = TARGET_ACCESS_SIBC4[arch]
    else:
        target_access, ta_detail = TARGET_ACCESS[ta_key]

    repro_val, repro_field, repro_path, repro_sha = repro_target(run_dir, family)

    out_root = (Path(args.out_root) if args.out_root
                else BASE / "dumps/st_r450_published_recompute")
    out = out_root / args.run_id

    print("=" * 78)
    print(f"ARM C r450 RE-SCORE  run_id={args.run_id}")
    print(f"  probs          {prob_dir}")
    print(f"  masks          {mask_dir}")
    print(f"  run_dir        {run_dir}   [config_root={config_root}]")
    print(f"  config         {cfg_source}  ({cfg_path})")
    print(f"  repro target   {repro_val:.4f}  (fg-only K=2, {repro_field})")
    print(f"  Table 38       {t38:.2f}   tol +-{t38_tol} ({arch})")
    print(f"  n_train/n_val  {n_train} / {n_val}   source={n_src}")
    print(f"  rung           {rung_value if rung_value is not None else 'null'}"
          f"   ({rung_provenance.split(';')[0]})")
    print(f"  target_access  {target_access}")
    print(f"  out            {out}")
    print("=" * 78, flush=True)

    if (out / ".DONE").exists():
        print(f"SKIP: {out}/.DONE exists (grid is resumable)")
        return

    prob_stems = {p.stem for p in prob_dir.glob("*.npy")}
    mask_stems = {p.stem for p in mask_dir.glob("*.png")}
    if prob_stems != mask_stems:
        op, om = sorted(prob_stems - mask_stems), sorted(mask_stems - prob_stems)
        raise SystemExit(
            f"FATAL stem-set mismatch for {args.run_id}\n"
            f"  n_prob={len(prob_stems)} n_mask={len(mask_stems)}\n"
            f"  probs-only ({len(op)}): {op[:10]}\n"
            f"  masks-only ({len(om)}): {om[:10]}")
    if len(prob_stems) != 150:
        raise SystemExit(f"FATAL expected 150 images for {args.run_id}, "
                         f"found {len(prob_stems)}")
    stems = sorted(prob_stems)
    print(f"stem sets identical, n = {len(stems)}", flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    from shadowtransfer import armc_runtime               # noqa: E402
    armc_runtime.preflight_write_targets(
        BASE,
        [out / "perimage.parquet", out / "probs.npz",
         out / "run.json", out / ".DONE"],
        label=args.run_id)

    if args.limit:
        if args.out_root is None:
            raise SystemExit("FATAL --limit requires --out_root (never the "
                             "real dump root)")
        stems = stems[:args.limit]
        print(f"SMOKE: truncated to {len(stems)} images", flush=True)

    # sp_gap per-image comparison, for the C4clean cells only.
    spgap = None
    if family == "sib":
        sp = (BASE / "data" / arch / "sp_gap_results"
              / f"sp_gap_{arch}_c4clean_{city}_{res_dir}.json")
        if sp.is_file():
            sj = json.loads(sp.read_text())
            spgap = {
                "path": str(sp.relative_to(BASE)),
                "sha256": sha256_file(sp),
                "summary": {k: sj.get(k) for k in
                            ("n_images_total", "n_valid_aurc",
                             "min_shadow_pixels", "miou_mean",
                             "aurc_shadow_mean", "ece_pred_pos_mean")},
                "per_image": {Path(e["filename"]).stem: e
                              for e in sj.get("per_image", [])},
            }

    geo = armc_dump.load_geo_meta(BASE)
    rows, probs = [], {}
    nf_fg, nf_bg, nf_mi = [], [], []            # no-filter, K=2
    nf_fg_s, nf_bg_s, nf_mi_s = [], [], []      # no-filter, K=0 (sp_gap recipe)
    gt_fg_raw = []
    sp_mi_mine, sp_mi_theirs, sp_keys = [], [], []

    for stem in stems:
        prob = np.load(prob_dir / f"{stem}.npy").astype(np.float32)
        raw = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise SystemExit(f"FATAL unreadable mask {stem}")
        if raw.shape != prob.shape:
            raise SystemExit(f"FATAL shape mismatch {stem}: mask {raw.shape} "
                             f"prob {prob.shape}")
        gt = (raw > 127).astype(np.uint8)
        gt_fg_raw.append(int(gt.sum()))

        pred = threshold_and_filter(prob, min_pixels=10)
        rows.append(armc_dump.perimage_row(
            args.run_id, stem, city, args.res, gt, prob, pred,
            geo.get(f"{stem}.png")))
        probs[stem] = np.load(prob_dir / f"{stem}.npy")   # float16, as stored

        # second recipe: plain p > 0.5, no size filter -- the published SIB
        # test path and sp_gap_analysis.compute_miou_per_image
        pred_nf = (prob > 0.5).astype(np.uint8)
        c = per_image_counts(pred_nf, gt, boundary_tolerance=2)
        f_, b_, m_ = two_class(c["tp_tol"], c["fp_tol"], c["tn_tol"], c["fn_tol"])
        nf_fg.append(f_); nf_bg.append(b_); nf_mi.append(m_)
        fs, bs, ms = two_class(c["tp_str"], c["fp_str"], c["tn_str"], c["fn_str"])
        nf_fg_s.append(fs); nf_bg_s.append(bs); nf_mi_s.append(ms)

        if spgap is not None and stem in spgap["per_image"]:
            theirs = spgap["per_image"][stem].get("miou")
            if theirs is not None:
                sp_mi_mine.append(ms / 100.0)
                sp_mi_theirs.append(float(theirs))
                sp_keys.append(stem)

    import pandas as pd                                    # noqa: E402
    df = pd.DataFrame(rows)
    wall = time.time() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
    miou_k2 = float(df["miou_tol_pct"].mean())
    miou_k0 = float(df["miou_str_pct"].mean())
    miou_k5 = float(
        (df["tp_k5"] / (df["tp_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100
         + df["tn_k5"] / (df["tn_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100)
        .mean() / 2.0)

    a = np.array(gt_fg_raw)
    usable = {
        "n_images": int(len(df)),
        "n_gt_fg_gt0": int((a > 0).sum()),
        "n_gt_fg_ge5": int((a >= 5).sum()),
        "n_gt_fg_ge50": int((a >= 50).sum()),
        "n_empty_fg_valid_k2": int(df["empty_fg_valid"].sum()),
        "n_aurc_fg_finite": int(np.isfinite(df["aurc_fg"]).sum()),
        "definitions": {
            "n_gt_fg_ge5": "floor used by sp_gap_analysis.py "
                           "(min_shadow_pixels=5); this is its n_valid_aurc",
            "n_gt_fg_ge50": "floor used by the canonical AURC population "
                            "builders (MIN_POP=50 in armc_dump, '< 50' in "
                            "metrics.py); this is what aurc_* can be computed on",
            "n_empty_fg_valid_k2": "images with zero GT foreground INSIDE the "
                                   "K=2 valid region; CONTRACT.md "
                                   "empty_fg_valid",
        },
        "miami_112_vs_113": (
            "miami is the only city where the two floors disagree. 113 images "
            "carry >=1 GT shadow pixel and 112 carry >=50; the single image "
            "between them is miami_session23_highres_paired_001 with 27 GT "
            "shadow pixels. sp_gap_*_c4clean_miami's n_valid_aurc=113 uses the "
            "5-pixel floor, test4/test4v2_per_image's 112 uses the 50-pixel "
            "one. chicago (139) and phoenix (129) agree under both floors "
            "because their smallest non-empty mask has 112 and 142 pixels."
            if city == "miami" else
            "not applicable to this city; both floors agree")
    }

    t38_delta = miou_k2 - t38
    t38_pass = abs(t38_delta) <= t38_tol
    repro_delta = fg_iou_tol - repro_val
    repro_pass = abs(repro_delta) <= REPRO_GATE_TOL

    nofilter = {
        "recipe": "p > 0.5, no filter_small_predictions",
        "why": ("sp_gap_analysis.compute_miou_per_image is strict K=0 "
                "two-class on (p > 0.5) with no size filter, and "
                "oglanet/train_oglanet_sib.py's test path has no "
                "filter_small_predictions call at all. Every other trainer in "
                "this arm, dinov3/train_dinov3_sib.py included, does filter. "
                "The primary dump columns keep the filtered recipe so they "
                "stay comparable with the 17 finished cells; see "
                "recipe_verdict for which recipe this cell's own published "
                "row used."),
        "fg_iou_k2_pct": float(np.mean(nf_fg)),
        "miou_k2_twoclass": float(np.mean(nf_mi)),
        "miou_k0_twoclass": float(np.mean(nf_mi_s)),
        "fg_iou_k0_pct": float(np.mean(nf_fg_s)),
        "t38_delta_using_nofilter_k2": float(np.mean(nf_mi)) - t38,
    }

    # ---- operator condition 3: WHICH recipe did the printed row use? -------
    # Decided per cell by evidence, not by inference: the published value is
    # this cell's own stored test_results.json foreground-only K=2 number, and
    # whichever of the two recipes reproduces it is the one that row was
    # computed with. Both deltas are recorded either way.
    d_filt = fg_iou_tol - repro_val
    d_nofilt = nofilter["fg_iou_k2_pct"] - repro_val
    if abs(d_filt) <= REPRO_GATE_TOL and abs(d_nofilt) <= REPRO_GATE_TOL:
        which = "indistinguishable (both inside the 0.01 gate)"
    elif abs(d_filt) <= REPRO_GATE_TOL:
        which = "filtered (p>0.5 then filter_small_predictions(min_pixels=10))"
    elif abs(d_nofilt) <= REPRO_GATE_TOL:
        which = "nofilter (p>0.5, no size filter)"
    else:
        which = ("neither reproduces the published value inside 0.01; the "
                 "closer of the two is " +
                 ("filtered" if abs(d_filt) <= abs(d_nofilt) else "nofilter"))
    recipe_verdict = {
        "published_value_fg_iou_k2": repro_val,
        "published_value_field": repro_field,
        "filtered_recompute": fg_iou_tol,
        "filtered_delta": d_filt,
        "nofilter_recompute": nofilter["fg_iou_k2_pct"],
        "nofilter_delta": d_nofilt,
        "gate": REPRO_GATE_TOL,
        "recipe_matching_published": which,
        "code_expectation": (
            # The two SIB trainers DISAGREE, verified 2026-10-26 by locating
            # every filter_small_predictions call relative to each trainer's
            # test_results.json write:
            #   oglanet/train_oglanet_sib.py  0 calls in the whole file ->
            #       test_and_save_predictions takes a plain argmax on p6
            #   dinov3/train_dinov3_sib.py    calls at lines 1109/1215/1401,
            #       test_results.json written at 1475 -> the filter IS applied
            # So the SIB rows of Table 38 were not all computed the same way.
            ("oglanet/train_oglanet_sib.py contains NO "
             "filter_small_predictions call anywhere, so this cell's published "
             "row is a plain argmax on p6 with no size filter"
             if arch == "oglanet" else
             "dinov3/train_dinov3_sib.py DOES call filter_small_predictions "
             "(lines 1109/1215/1401) ahead of its test_results.json write at "
             "line 1475, so this cell's published row is filtered -- unlike "
             "the oglanet SIB cells, whose trainer has no such call")
            if family == "sib" else
            "filter_small_predictions appears in some classic trainers' test "
            "paths and not others (present in oglanet/train.py, train_iim.py, "
            "train_segdesic.py, mamnet/train_iim.py, train_mrfp.py, "
            "train_fada.py and run_inference.py; absent from mamnet/train.py "
            "and mamnet/train_isw.py), so it is decided per cell by the deltas "
            "above rather than assumed per architecture"),
    }

    spgap_cmp = None
    if spgap is not None:
        mine = np.array(sp_mi_mine)
        theirs = np.array(sp_mi_theirs)
        d = mine - theirs
        spgap_cmp = {
            "comparator": spgap["path"],
            "comparator_sha256": spgap["sha256"],
            "comparator_summary": spgap["summary"],
            "comparable_quantity": (
                "per-image two-class mIoU, STRICT K=0, prediction (p > 0.5) "
                "with no size filter -- exactly "
                "sp_gap_analysis.compute_miou_per_image"),
            "n_compared": int(len(mine)),
            "mine_mean": float(mine.mean() * 100) if len(mine) else None,
            "theirs_mean": float(theirs.mean() * 100) if len(mine) else None,
            "mean_delta_pct": float(d.mean() * 100) if len(mine) else None,
            "max_abs_delta_pct": float(np.abs(d).max() * 100) if len(mine) else None,
            "n_images_differing_gt_1e4": int((np.abs(d) > 1e-4).sum()),
            "worst_images": [
                {"image_id": sp_keys[i], "mine_pct": float(mine[i] * 100),
                 "theirs_pct": float(theirs[i] * 100),
                 "delta_pct": float(d[i] * 100)}
                for i in np.argsort(-np.abs(d))[:5]] if len(mine) else [],
            "their_miou_mean_pct": (spgap["summary"]["miou_mean"] * 100
                                    if spgap["summary"]["miou_mean"] is not None
                                    else None),
            "my_miou_k0_nofilter_mean_pct": nofilter["miou_k0_twoclass"],
            "my_aurc_fg_trapz19_mean": float(np.nanmean(df["aurc_fg"])),
            "their_aurc_shadow_mean": spgap["summary"]["aurc_shadow_mean"],
            "aurc_note": (
                "NOT the same definition. aurc_fg here is aurc_trapz19 (19-point "
                "trapezoid over coverage, CONTRACT.md section 3 leaves the "
                "canonical choice blank); sp_gap's aurc_shadow is its own "
                "20-point construction. The two are reported side by side and "
                "neither is declared canonical."),
            "my_ece_pred_pos_mean": float(np.nanmean(df["ece_pred_pos"])),
            "their_ece_pred_pos_mean": spgap["summary"]["ece_pred_pos_mean"],
        }

        # ---- operator condition 4: is the ranking error a few images or all?
        # aurc_shadow is the published per-image ranking-error score. A high
        # cell mean can come from a heavy tail or from a shifted bulk, and the
        # two mean different things, so both are measured here rather than
        # inferred from the mean.
        ar = [(k, v.get("aurc_shadow"), v.get("miou"), v.get("ece_pred_pos"))
              for k, v in spgap["per_image"].items()
              if v.get("aurc_shadow") is not None]
        ar.sort(key=lambda t: -t[1])
        av = np.array([t[1] for t in ar], dtype=float)
        if av.size:
            tot = float(av.sum())
            trimmed = {f"drop_worst_{k}": (float(av[k:].mean())
                                           if av.size > k else None)
                       for k in (0, 1, 3, 5, 10, 20, 30)}
            share = {f"worst_{k}_share_of_total_pct":
                     (float(av[:k].sum() / tot * 100) if tot > 0 and av.size >= k
                      else None)
                     for k in (5, 10, 20, 30)}
            n_ceil = int((av >= 0.999999).sum())
            spgap_cmp["ranking_error_profile"] = {
                "quantity": ("per-image aurc_shadow from the cell's own "
                             "sp_gap json: area under the risk-coverage curve "
                             "on GT-shadow pixels, so 1.0 means confidence "
                             "never ranks a correct shadow pixel above an "
                             "incorrect one at any coverage"),
                "n_images_with_a_score": int(av.size),
                "mean": float(av.mean()), "median": float(np.median(av)),
                "p25": float(np.percentile(av, 25)),
                "p75": float(np.percentile(av, 75)),
                "p90": float(np.percentile(av, 90)),
                "min": float(av.min()), "max": float(av.max()),
                "n_at_ceiling_1p0": n_ceil,
                "mean_after_trimming_worst": trimmed,
                "mass_held_by_worst": share,
                "reading": (
                    "a FEW images: the mean falls to near the comparator cells "
                    "once a handful are trimmed"
                    if (av.size > 10 and float(av[10:].mean()) < 0.5 * float(av.mean()))
                    else "ALL of them: trimming the worst 10 leaves the mean "
                         "essentially unchanged, so the whole distribution is "
                         "shifted rather than a tail being long"),
                "worst_10_images": [
                    {"image_id": t[0], "aurc_shadow": float(t[1]),
                     "miou_pct": (None if t[2] is None else float(t[2]) * 100),
                     "ece_pred_pos": (None if t[3] is None else float(t[3])),
                     "at_ceiling": bool(t[1] >= 0.999999)}
                    for t in ar[:10]],
            }

    run = {
        "run_id": args.run_id, "dataset": "st", "arm": "C",
        "arch": arch, "method": method, "mode": mode,
        "method_canonical": "sib" if family == "sib" else method,
        "method_variant": "c4clean" if family == "sib" else None,
        "method_token_note": (
            "CONTRACT.md section 1's method vocabulary has 'sib'; the operator "
            "named these cells 'sibc4' to separate the published C4clean "
            "configuration from any other SIB variant. The run_id uses the "
            "operator's token; method_canonical carries the contract one."
            if family == "sib" else None),
        "rung": rung_value,
        "rung_declared": declared_rung,
        "rung_provenance": rung_provenance,
        "rung_label_anomaly": anomaly,
        "n_train": n_train, "n_train_per_source_city": None, "n_val": n_val,
        "n_train_val_source": n_src,
        "n_train_val_detail": n_detail,
        "n_train_val_measured": True,
        "n_train_val_caveat": (
            "Measured per operator instruction 2026-10-26 ('do not override; "
            "measure from each cell's own training log or split files'). Where "
            "no train.log survives, the per-city split directories are counted "
            "on disk and combined with the dataloader path read from the "
            "cell's own trainer, because a split count alone cannot separate "
            "450 from 900. CLAUDE.md carries an earlier 2026-07-28 operator "
            "instruction that directory contents are not authoritative for "
            "this field; the newer instruction is followed and the conflict is "
            "recorded. Nothing downstream gates on this field."),
        "n_test": int(len(df)),
        "holdout": city, "res": args.res, "seed": 0,
        "seed_source": "nominal; the legacy config records no seed",
        "n_seeds": 1,
        "target_access": target_access,
        "target_access_detail": ta_detail,
        "epochs_max": (cfg or {}).get("epochs"),
        "epochs_run": None, "steps_to_best": None, "best_val_metric": None,
        "early_stop_patience": (cfg or {}).get("early_stopping_patience"),
        "lr": (cfg or {}).get("lr"),
        "batch_size": (cfg or {}).get("batch_size"),
        "metric_module_sha": sha256_file(BASE / "python/shadowtransfer/metrics.py"),
        "metric_module_path": "python/shadowtransfer/metrics.py",
        "dump_module_sha": sha256_file(BASE / "python/shadowtransfer/armc_dump.py"),
        "recompute_script_sha": sha256_file(Path(__file__).resolve()),
        "git_hash": None,
        "git_hash_reason": "project root is not a git repository (CLAUDE.md)",
        "config_sha256": sha256_file(cfg_path) if cfg_path else None,
        "config_verbatim": cfg,
        "config_source": cfg_source,
        "config_path": str(cfg_path) if cfg_path else None,
        "config_root": config_root, "config_root_path": config_root_path,
        "config_root_note": (
            "resolve_run_dir scores each root on the required-file set, not on "
            "args.json alone. /projects carries partial copies of some "
            "published run directories; that is what failed job 20553148."),
        "cluster": "delta",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "gpu": None,
        "compute": ("cpu-only pass over stored probability maps; no checkpoint "
                    "loaded, no inference run"),
        "compute_note": ("CPU-only workload submitted on gpuA100x4; no CPU "
                         "account exists on this allocation. GPU allocated and "
                         "unused."),
        "node": os.environ.get("SLURMD_NODENAME"),
        "wall_clock_s": round(wall, 2), "peak_rss_mb": round(peak_rss_mb, 1),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "ckpt_path": None,
        "ckpt_path_note": ("not loaded by this job; most of these cells' "
                           "checkpoints were archived to Purdue Fortress on "
                           "2026-09-22 and deleted locally"),
        "parent_run_id": None,
        "legacy_run_dir": legacy_run_dir,
        "structure_cols": "deferred",
        "source_probs_dir": str(prob_dir.relative_to(BASE)),
        "source_probs_variant": variant,
        "source_mask_dir": str(mask_dir.relative_to(BASE)),
        "source_mask_note": ("Final_data_loco and Final_data_test test/masks "
                             "verified byte-identical for all three cities on "
                             "2026-10-25 (md5 over the concatenated 150 PNGs)"),
        "probs_are_prefilter": True,
        "prediction_recipe": "p > 0.5 then filter_small_predictions(min_pixels=10)",
        "band_width_headline": 2, "band_widths_written": [0, 2, 5],
        "headline_metric": "foreground-only IoU at K=2 (column iou_tol)",
        "headline_fg_iou_tol_pct": fg_iou_tol,
        "miou_k2_twoclass": miou_k2,
        "miou_k0_twoclass": miou_k0,
        "miou_k5_twoclass": miou_k5,
        "table38_printed": t38,
        "table38_source": "final_paper.md appendix I, Table 38",
        "table38_quantity": ("per-image mean of two-class boundary-tolerant "
                             "mIoU at K=2; established empirically against the "
                             "12 finished mamnet+dinov3 cells"),
        "table38_recompute": miou_k2,
        "table38_delta": t38_delta,
        "table38_gate_tol": t38_tol,
        "table38_gate_tol_reason": (
            "erratum E-9: OGLANet inference is nondeterministic, so every "
            "published OGLANet number is one draw; 1.12 is the operator's "
            "stated run-to-run spread" if arch == "oglanet" else
            "0.01, the operator's stated tolerance"),
        "table38_gate_pass": bool(t38_pass),
        "repro_target_source": "test_results.json",
        "repro_target_field": repro_field,
        "repro_target_path": repro_path,
        "repro_target_sha256": repro_sha,
        "repro_target_fg_iou_k2": repro_val,
        "repro_fg_iou_k2": fg_iou_tol,
        "repro_delta_fg_iou_k2": repro_delta,
        "repro_gate_tol": REPRO_GATE_TOL,
        "repro_gate_pass": bool(repro_pass),
        "repro_metric_definition": (
            "foreground-only IoU with the +-2 px don't-care band, CONTRACT.md "
            "section 8. Mean over the per-image iou_fg_tol_pct; target is the "
            "cell's own stored test_results.json. Reported, never adjusted. "
            "This gate already fails on 11 of the 17 previously finished cells, "
            "so a failure here continues a recorded finding."),
        "nofilter_recipe": nofilter,
        "recipe_verdict": recipe_verdict,
        "usable_images": usable,
        "spgap_comparison": spgap_cmp,
        "aurc_definition": "aurc_trapz19",
        "aurc_definition_note": (
            "CONTRACT.md section 3 leaves the canonical AURC blank. "
            "aurc_fg/bg/all use aurc_trapz19; aurc_*_mean20 carry the "
            "SIB-trainer plain mean over 20 points. Neither is canonical here."),
        "smoke": bool(args.limit),
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    flags = []
    if anomaly:
        flags.append(f"rung label disputed: n_train={n_train} from {n_src}, "
                     f"declared 450; see rung_label_anomaly")
    if n_train is None:
        flags.append("n_train/n_val unrecorded by this cell's own config; "
                     "written null rather than guessed")
    if usable["n_empty_fg_valid_k2"]:
        flags.append(f"empty_fg_valid on {usable['n_empty_fg_valid_k2']} images")
    if not t38_pass:
        flags.append(f"TABLE 38 GATE FAILED: 2-class K=2 {miou_k2:.4f} vs "
                     f"printed {t38:.2f}, delta {t38_delta:+.4f} > {t38_tol}")
    if not repro_pass:
        flags.append(f"REPRODUCTION GATE FAILED: fg IoU K=2 {fg_iou_tol:.4f} vs "
                     f"test_results.json {repro_val:.4f}, delta "
                     f"{repro_delta:+.4f} > {REPRO_GATE_TOL}")
    if cfg is None:
        flags.append("no config file and no train.log Args line; "
                     "config_verbatim is null")
    if which.startswith("neither"):
        flags.append(f"NEITHER prediction recipe reproduces the published "
                     f"fg IoU K=2 {repro_val:.4f} inside {REPRO_GATE_TOL} "
                     f"(filtered {fg_iou_tol:.4f}, nofilter "
                     f"{nofilter['fg_iou_k2_pct']:.4f})")
    if n_train is not None and n_train != 450:
        side = "above" if n_train > 450 else "below"
        flags.append(f"measured n_train={n_train} ({side} 450), "
                     f"n_val={n_val}; source {n_src}")
    run["sanity_flags"] = flags

    armc_dump.write_dump(out, rows, probs, run)

    print(f"\n[{args.run_id}]")
    print(f"  images                  {len(df)}")
    print(f"  fg IoU K=2 (headline)   {fg_iou_tol:.4f}")
    print(f"  2-class K=2 (Table 38)  {miou_k2:.4f}  printed {t38:.2f}  "
          f"delta {t38_delta:+.4f}  tol {t38_tol} -> "
          f"{'PASS' if t38_pass else 'FAIL'}")
    print(f"  repro target (K=2 fg)   {repro_val:.4f}  delta {repro_delta:+.4f}"
          f"  tol {REPRO_GATE_TOL} -> {'PASS' if repro_pass else 'FAIL'}")
    print(f"  no-filter 2-class K=2   {nofilter['miou_k2_twoclass']:.4f}   "
          f"K=0 {nofilter['miou_k0_twoclass']:.4f}")
    print(f"  recipe vs published     filtered {d_filt:+.4f}   "
          f"nofilter {d_nofilt:+.4f}  ->  {which}")
    print(f"  usable images           gt_fg>0 {usable['n_gt_fg_gt0']}  "
          f">=5 {usable['n_gt_fg_ge5']}  >=50 {usable['n_gt_fg_ge50']}  "
          f"empty_fg_valid {usable['n_empty_fg_valid_k2']}")
    if spgap_cmp:
        print(f"  sp_gap per-image        n={spgap_cmp['n_compared']}  "
              f"mine {spgap_cmp['mine_mean']:.4f} theirs "
              f"{spgap_cmp['theirs_mean']:.4f}  mean delta "
              f"{spgap_cmp['mean_delta_pct']:+.6f}  max |delta| "
              f"{spgap_cmp['max_abs_delta_pct']:.6f}")
    print(f"  n_train/n_val           {n_train} / {n_val}  ({n_src})")
    print(f"  config_root             {config_root}  config={cfg_source}")
    print(f"  sanity flags            {flags or 'none'}")
    print(f"  wall {wall:.1f}s ({wall/len(df):.3f}s/img)  "
          f"peak RSS {peak_rss_mb:.0f} MB")
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
