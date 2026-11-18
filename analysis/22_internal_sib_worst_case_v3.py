#!/usr/bin/env python
"""
Arm F / E-12 / E-2 -- internal SIB worst-case adjudication, v3.

Answers BxRx-W3, AC3, v1UJ-W3.

v1 (results/11_internal_sib_worst_case.json) is GATE_FAILED: the six published
adaptation baselines and the published SIB cells existed on the analysis host only as Table
38 printed values, so no worst-case gap could be computed. The per-image scores
now exist. This job adjudicates the pre-committed readings R1-R4.

v2 (results/12_internal_sib_worst_case_v2.json) adjudicated R1 on the declared
PNG tree, but the PNG tree then carried no Vanilla cell at all, so all nine
cross-city Vanilla rows, and the nine SAR rows whose unadapted level is a
Vanilla cell, fell back to the probability-map tree. For OGLANet the two trees
are different predictions rather than two scorings of one prediction, and the
upper bounds are PNG-scored, so those eighteen rows were on a different
population from their own upper bound. The nine PNG-scored Vanilla cells have
since arrived and each reproduces final_comparison.json to 0.000000. v3 is v2
with Vanilla, and the SAR unadapted level, taken from the PNG tree. Nothing else
in the computation changes.

DECLARED PATH
  metric      two-class average boundary-tolerant mIoU at K=2, in percent (the
              Table 38 / revised Table 42 family), computed per image from raw
              tp/fp/fn/tn on the valid region and macro-averaged over the cell's
              150 test images
  degenerate  App E rule: an image with tp+fp+fn == 0 at K=2 scores foreground
              IoU 0 and background IoU 100, so its two-class value is exactly 50
  resolution  high resolution only
  rung        450
  source      dumps/st_r450_published_png/ -- the saved prediction PNG masks,
              which are the tree Table 38 and final_comparison.json were scored
              from. 85 cells, the nine high-resolution cross-city Vanilla
              cells included. The eighteen high-resolution standard cells
              reproduce final_comparison.json to 0.000000 and the 59 method
              cells reproduce printed Table 38 to a maximum |diff| of 0.0072.
  fallback    dumps/st_r450_published_recompute/ (published cells re-passed over
              their own stored float16 probability maps), used ONLY where the PNG
              tree does not carry a cell, and recorded per row. No row falls back
              in v3.

GAP SIGN. CONVENTIONS.md fixes `gap` = LOCO minus the matched within-city upper
bound, so the paper's gap is negative. The operator's brief asks for `upper minus
method`, which is the same quantity with the sign flipped and is the one the
worst-case claim is phrased in. Both are emitted on every row:
  gap                       = method - upper   (CONVENTIONS.md sign, negative)
  gap_upper_minus_method    = upper - method   (the brief's sign, positive)
"worst case" always means the largest `gap_upper_minus_method` over the cells of
a scope, i.e. the deployment-relevant shortfall.

Read-only on every dump tree. No training, no GPU inference, no checkpoint
loaded. Nothing under dumps/ or results/ is modified. Writes exactly two files:
results/22_internal_sib_worst_case_v3.json and the PNG tree's index at
analysis/_inv/INDEX_st_r450_published_png.parquet (CLAUDE.md requires selection
from an index; the operator's brief forbids writing inside dumps/, so the index
is built here instead of beside the tree).

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/22_internal_sib_worst_case_v3.py [--n450]
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
N450 = "--n450" in sys.argv          # swap in the matched-450 retrains
B = 10000                            # paired bootstrap replicates, everywhere
BOOT_SEED = 0

PNG = os.path.join(BASE, "dumps", "st_r450_published_png")
RECOMPUTE = os.path.join(BASE, "dumps", "st_r450_published_recompute")
ST = os.path.join(BASE, "dumps", "st")
E12 = os.path.join(BASE, "delta_dumps", "dumps", "e12")
INV = os.path.join(BASE, "analysis", "_inv")

# Trees searched for an n450 counterpart when --n450 is given, in order. The
# matched-450 retrain is still in flight on the SLURM box; when it lands, its
# dumps arrive under one of these roots and nothing else here has to change.
N450_TREES = [("st_r450_published_png", PNG),
              ("st_r450_published_recompute", RECOMPUTE),
              ("st", ST),
              ("st_n450", os.path.join(BASE, "dumps", "st_n450"))]

ARCHS = ("mamnet", "oglanet", "dinov3")
CITIES = ("chicago", "miami", "phoenix")
CE = "crossearth"

E9_MAX_SD5 = 0.4247                  # erratum E-9's measured OGLANet band
E9_MAX_SPAN9 = 1.1158

# Printed Table 38 values, used ONLY in agreement checks and in the explicit
# missing-cell bound. Nothing is rounded to match them.
TABLE38 = {
    ("upper", "mamnet"): (86.31, 79.11, 73.75),
    ("upper", "oglanet"): (79.12, 73.89, 66.15),
    ("upper", "dinov3"): (80.27, 76.00, 67.82),
    ("vanilla", "mamnet"): (79.72, 74.64, 62.84),
    ("vanilla", "oglanet"): (68.47, 66.18, 61.18),
    ("vanilla", "dinov3"): (67.26, 72.61, 65.24),
    ("fda", "mamnet"): (81.88, 74.21, 67.77),
    ("fda", "oglanet"): (71.07, 67.30, 63.27),
    ("fda", "dinov3"): (72.26, 70.81, 58.72),
    ("segdesic", "mamnet"): (82.44, 73.14, 58.80),
    ("segdesic", "oglanet"): (66.47, 59.90, 61.17),
    ("segdesic", "dinov3"): (75.52, 71.18, 68.04),
    ("iim", "mamnet"): (81.06, 70.05, 55.43),
    ("iim", "oglanet"): (70.01, 66.36, 55.25),
    ("iim", "dinov3"): (77.43, 72.23, 56.51),
    ("isw", "mamnet"): (75.23, 67.74, 51.08),
    ("isw", "oglanet"): (64.19, 67.78, 59.69),
    ("isw", "dinov3"): (57.60, 59.86, 55.68),
    ("mrfp_plus", "mamnet"): (75.76, 72.80, 53.35),
    ("mrfp_plus", "oglanet"): (63.82, 57.30, 58.03),
    ("mrfp_plus", "dinov3"): (77.46, 73.20, 60.78),
    ("fada", "mamnet"): (79.59, 60.90, 62.30),
    ("fada", "oglanet"): (66.17, 65.20, 60.26),
    ("fada", "dinov3"): (70.15, 71.43, 62.81),
    ("sib", "mamnet"): (81.65, 74.54, 68.64),
    ("sib", "oglanet"): (73.05, 66.71, 63.97),
    ("sib", "dinov3"): (76.57, 71.71, 61.45),
}

# Access classes: the I-1 countersigned taxonomy as the operator's brief states
# it. These OVERRIDE each dump's own target_access field, which on the PNG tree
# describes the scoring pass ("source_only; scoring pass over stored artifacts")
# and on arm F's CrossEarth upper cells reads source_only where arm C/D uppers
# read target_labels. Every disagreement is recorded per row.
ACCESS = {
    "upper": "target_labels",
    "vanilla": "source_only",
    "fda": "unlabeled_target",
    "segdesic": "unlabeled_target",
    "iim": "source_only",
    "isw": "source_only",
    "mrfp_plus": "source_only",
    "fada": "source_only",
    "bsm": "source_only",
    "sar": "deployment_time",
    "sib_mamnet": "unlabeled_target",
    "sib_oglanet": "source_only",
    "sib_dinov3": "source_only",
    "sib_crossearth": "source_only",
}

# Pre-committed readings, verbatim from results/11_internal_sib_worst_case.json.
PRECOMMITTED = {
    "R1": ("SIB has the smallest pooled worst-case gap on the 9 original cells "
           "AND on the 12 cells incl. CrossEarth -> claim stands as written; "
           "print with the MAMNet seed band."),
    "R2": ("SIB smallest on 9, not on 12 -> claim scoped to the three "
           "architectures SIB was developed on (the posted AC wording). "
           "CrossEarth reported beside it."),
    "R3": ("Any other method's worst-case gap <= SIB's on the 9 cells -> "
           "\"smallest worst-case of any method\" is withdrawn. HLT call on the "
           "replacement sentence."),
    "R4": ("SIB's worst-case gap >= vanilla's on either scope -> \"smaller than "
           "doing nothing\" (contribution 4) is withdrawn."),
}

CONTRACT_COLS = {
    "run_id": "object", "image_id": "object", "city": "object", "res": "object",
    "lon": "float", "lat": "float", "source_tile_id": "object",
    "pair_id": "object",
    "tp_tol": "int", "fp_tol": "int", "fn_tol": "int", "tn_tol": "int",
    "tp_str": "int", "fp_str": "int", "fn_str": "int", "tn_str": "int",
    "n_valid_px": "int", "n_gt_fg_valid": "int", "empty_fg_valid": "bool",
    "iou_tol": "float", "iou_str": "float",
    "pbar_gt_shadow": "float", "pbar_gt_bg": "float",
    "logit_mean_tp": "float", "logit_mean_fp": "float",
    "logit_mean_fn": "float", "logit_mean_tn": "float",
    "aurc_fg": "float", "aurc_bg": "float", "aurc_all": "float",
    "aurc_ce_fg": "float", "aurc_ce_bg": "float",
    "ece_pred_pos": "float", "ece_pred_neg": "float", "n_pred_pos": "int",
    "n_gt_blobs": "int", "n_pred_blobs": "int",
    "gt_blob_area_median": "float", "small_object_frac": "float",
    "fn_px_by_cat": "object", "tp_px_by_cat": "object",
    "gt_px_by_cat": "object",
    "fp_px_near_gt": "int", "fp_px_far": "int",
    "gt_contrast_median": "float",
}

LOAD_BEARING = ("run_id", "image_id", "city", "res",
                "tp_tol", "fp_tol", "fn_tol", "tn_tol",
                "n_valid_px", "n_gt_fg_valid", "empty_fg_valid",
                "iou_tol", "iou_fg_tol_pct", "iou_bg_tol_pct",
                "miou_tol_pct")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fail(msg):
    raise SystemExit("ASSERTION FAILED: " + msg)


# ----------------------------------------------------------------------------
# the one metric implementation on this box
# ----------------------------------------------------------------------------

def two_class_tol_k2(df):
    """Per-image two-class boundary-tolerant mIoU at K=2, percent.

    CONTRACT.md rule zero puts the canonical implementation in
    shadowtransfer/metrics.py on the SLURM box. No copy of that file is
    reachable from the analysis host (searched the project base, analysis/, python/ and the
    Shade Maps checkout), so the metric is recomputed here from the raw
    tp/fp/fn/tn counts that CONTRACT.md section 3 calls the source of truth, and
    is then asserted against the stored iou_fg_tol_pct / iou_bg_tol_pct /
    miou_tol_pct columns on every cell read. Any disagreement is a loud failure.
    """
    tp = df.tp_tol.to_numpy(dtype=np.float64)
    fp = df.fp_tol.to_numpy(dtype=np.float64)
    fn = df.fn_tol.to_numpy(dtype=np.float64)
    tn = df.tn_tol.to_numpy(dtype=np.float64)
    den_fg = tp + fp + fn
    den_bg = tn + fp + fn
    if (den_bg <= 0).any():
        fail("background IoU denominator is zero on some image")
    fg = np.where(den_fg > 0, tp / np.where(den_fg > 0, den_fg, 1.0), 0.0) * 100.0
    bg = tn / den_bg * 100.0
    return (fg + bg) / 2.0, fg, bg


def two_class_from_counts(tp, fp, fn, tn):
    """Same formula over bare count arrays, for the SAR perstep parquet."""
    tp, fp, fn, tn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn, tn))
    den_fg, den_bg = tp + fp + fn, tn + fp + fn
    if (den_bg <= 0).any():
        fail("background IoU denominator is zero on some SAR image")
    fg = np.where(den_fg > 0, tp / np.where(den_fg > 0, den_fg, 1.0), 0.0) * 100.0
    bg = tn / den_bg * 100.0
    return (fg + bg) / 2.0


def read_cell(root, run_id):
    d = os.path.join(root, run_id)
    pq = os.path.join(d, "perimage.parquet")
    sj = os.path.join(d, "run.json")
    if not os.path.isdir(d):
        return None, None, "no dump directory"
    if not os.path.isfile(pq):
        extra = (" (.FAILED marker present)"
                 if os.path.isfile(os.path.join(d, ".FAILED")) else "")
        return None, None, "no perimage.parquet" + extra
    if not os.path.isfile(sj):
        fail(f"{run_id}: perimage.parquet with no run.json sidecar "
             "(CONTRACT.md section 7)")
    df = pd.read_parquet(pq)
    meta = json.load(open(sj))
    missing = [c for c in CONTRACT_COLS if c not in df.columns]
    if missing:
        fail(f"{run_id}: CONTRACT.md section 3 columns absent: {missing}")
    deferred = []
    for col, kind in CONTRACT_COLS.items():
        dt = df[col].dtype
        if df[col].isna().all():
            deferred.append(col)
            continue
        ok = {"int": pd.api.types.is_integer_dtype,
              "float": pd.api.types.is_float_dtype,
              "bool": pd.api.types.is_bool_dtype,
              "object": lambda t: t == object}[kind](dt)
        if not ok:
            fail(f"{run_id}: column {col} dtype {dt} is not {kind}; never "
                 "coerce a mismatched dtype (CLAUDE.md)")
    for col in LOAD_BEARING:
        if col not in df.columns:
            fail(f"{run_id}: load-bearing column {col} absent")
        if df[col].isna().any():
            fail(f"{run_id}: load-bearing column {col} carries "
                 f"{int(df[col].isna().sum())} nulls")
    for col in ("tp_tol", "fp_tol", "fn_tol", "tn_tol"):
        if not pd.api.types.is_integer_dtype(df[col].dtype):
            fail(f"{run_id}: count column {col} dtype {df[col].dtype} is not "
                 "integer; never coerce a mismatched dtype")
    if len(df) != 150:
        fail(f"{run_id}: n_images {len(df)} != the expected test-pool size 150")
    m2, fg, bg = two_class_tol_k2(df)
    for name, got, stored in (("fg", fg, df.iou_fg_tol_pct.to_numpy()),
                              ("bg", bg, df.iou_bg_tol_pct.to_numpy()),
                              ("2class", m2, df.miou_tol_pct.to_numpy())):
        mx = float(np.abs(got - stored).max())
        if mx > 1e-9:
            fail(f"{run_id}: recomputed {name} IoU disagrees with the stored "
                 f"column by {mx:.3e}")
    meta["_deferred_cols"] = deferred
    return df, meta, None


# ----------------------------------------------------------------------------
# index
# ----------------------------------------------------------------------------

SCALAR = (str, int, float, bool, type(None))


def build_index(label, root, write_to=None):
    """CONTRACT.md section 5: one row per run_id, every scalar run.json field
    plus n_images, mean_iou_tol, parquet_mtime, dump_sha256.

    Every index this job selects from is built here and written outside the dump
    tree. The PNG tree arrived from Delta with no INDEX.parquet at all, and the
    recompute tree's on-disk index is stale (it predates the three trial cells
    copied after it was written). CLAUDE.md requires a current index and the
    operator's brief forbids writing inside dumps/, so the two instructions are
    satisfied by building the index here. The on-disk index of each tree that has
    one is still read, hashed and reported, including its staleness.
    """
    rows, unindexable = [], []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        pq, sj = os.path.join(d, "perimage.parquet"), os.path.join(d, "run.json")
        if not (os.path.isfile(pq) and os.path.getsize(pq) > 0
                and os.path.isfile(sj)):
            unindexable.append(dict(
                run_id=name, has_parquet=os.path.isfile(pq),
                parquet_bytes=os.path.getsize(pq) if os.path.isfile(pq) else 0,
                has_run_json=os.path.isfile(sj),
                failed_marker=os.path.isfile(os.path.join(d, ".FAILED"))))
            continue
        meta = json.load(open(sj))
        df = pd.read_parquet(pq, columns=["iou_tol"])
        r = {k: v for k, v in meta.items() if isinstance(v, SCALAR)}
        r.update(run_id=name, n_images=int(len(df)),
                 mean_iou_tol=float(df.iou_tol.mean()),
                 parquet_mtime=float(os.path.getmtime(pq)),
                 dump_sha256=sha256_file(pq))
        rows.append(r)
    idx = pd.DataFrame(rows)
    # run.json keys are not type-stable across arms (`probe` is an int on the
    # arm C probes and the string "noise" elsewhere), and parquet needs one type
    # per column. Columns carrying more than one python type are written as
    # strings, and which ones were cast is recorded rather than hidden.
    mixed = []
    for col in idx.columns:
        if idx[col].dtype != object:
            continue
        kinds = {type(v).__name__ for v in idx[col] if v is not None}
        if len(kinds) > 1:
            mixed.append(dict(column=col, types=sorted(kinds)))
            idx[col] = idx[col].map(lambda v: None if v is None else str(v))
    for rid, g in idx.groupby("run_id"):
        if "config_sha256" in g.columns and g.config_sha256.nunique() > 1:
            fail(f"{label}: run_id {rid} carries "
                 f"{g.config_sha256.nunique()} config_sha256 values")
    info = dict(label=label, root=os.path.relpath(root, BASE),
                n_rows=int(len(idx)), n_unindexable=len(unindexable),
                unindexable=unindexable, built_here=True,
                columns_cast_to_str=mixed)
    if write_to:
        os.makedirs(os.path.dirname(write_to), exist_ok=True)
        tmp = write_to + ".new"
        idx.to_parquet(tmp, index=False)
        os.replace(tmp, write_to)
        info.update(path=os.path.relpath(write_to, BASE),
                    sha256=sha256_file(write_to))
    return idx, info


def on_disk_index_state(label, root):
    """Read, hash and age-check a tree's own INDEX.parquet. Reported, never
    selected from: this job selects from the indexes it builds itself, because
    it may not write inside dumps/ and a stale index may not be selected from.
    """
    p = os.path.join(root, "INDEX.parquet")
    if not os.path.isfile(p):
        return dict(label=label, path=os.path.relpath(p, BASE), present=False,
                    sha256=None, n_rows=None, stale=None,
                    note="no INDEX.parquet in this tree")
    idx = pd.read_parquet(p)
    mt = os.path.getmtime(p)
    newer = [n for n in sorted(os.listdir(root))
             if os.path.isdir(os.path.join(root, n))
             and os.path.isfile(os.path.join(root, n, "perimage.parquet"))
             and os.path.getmtime(os.path.join(root, n, "perimage.parquet")) > mt]
    absent = [n for n in sorted(os.listdir(root))
              if os.path.isdir(os.path.join(root, n))
              and os.path.isfile(os.path.join(root, n, "perimage.parquet"))
              and n not in set(idx.run_id.astype(str))]
    return dict(label=label, path=os.path.relpath(p, BASE), present=True,
                sha256=sha256_file(p), n_rows=int(len(idx)),
                stale=bool(newer or absent),
                dumps_newer_than_the_index=newer,
                dumps_absent_from_the_index=absent,
                note=("current" if not (newer or absent) else
                      "STALE: this job does not select from it and does not "
                      "rewrite it (the brief forbids writing inside dumps/)"))


# ----------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------

def paired_bootstrap(delta, b=B, seed=BOOT_SEED):
    """Paired image-level bootstrap on per-image deltas, two-tailed.

    Returns the point estimate, its standard error, the percentile 95% CI and a
    two-tailed p-value floored at 1/B (the arm F convention: p at 1e-04 reads as
    p <= 1/B).
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    reps = delta[idx].mean(axis=1)
    point = float(delta.mean())
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    p = 2.0 * min(float((reps <= 0).mean()), float((reps >= 0).mean()))
    return dict(mean=point,
                stderr=float(delta.std(ddof=1) / np.sqrt(n)),
                n=int(n), ci95=[lo, hi], ci_method=f"paired_bootstrap_B{b}",
                p=float(min(1.0, max(1.0 / b, p))), B=b, two_tailed=True,
                bootstrap_sd=float(reps.std(ddof=1)))


def cluster_bootstrap_mean_over_cells(per_cell_deltas, b=B, seed=BOOT_SEED):
    """Mean over cells of a per-image paired delta, with an image-cluster
    bootstrap.

    per_cell_deltas maps a cluster key (the held-out city, whose 150 test images
    are shared by every cell of that city in identical order) to a list of
    per-image delta arrays. Images are resampled once per city and the same
    resampled index vector is applied to every cell of that city, so the
    clustering matches Appendix G.9's image-cluster bootstrap rather than
    treating two cells of one city as independent.
    """
    cells = [(k, a) for k, arrs in per_cell_deltas.items() for a in arrs]
    if not cells:
        return None
    point = float(np.mean([a.mean() for _, a in cells]))
    rng = np.random.default_rng(seed)
    keys = sorted(per_cell_deltas)
    n_by_key = {k: per_cell_deltas[k][0].size for k in keys}
    reps = np.empty(b, dtype=np.float64)
    for r in range(b):
        draw = {k: rng.integers(0, n_by_key[k], size=n_by_key[k]) for k in keys}
        reps[r] = np.mean([a[draw[k]].mean() for k, a in cells])
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    p = 2.0 * min(float((reps <= 0).mean()), float((reps >= 0).mean()))
    return dict(mean=point, stderr=float(reps.std(ddof=1)), n_cells=len(cells),
                n_images_total=int(sum(a.size for _, a in cells)),
                ci95=[lo, hi],
                ci_method=f"paired_image_cluster_bootstrap_B{b}",
                p=float(min(1.0, max(1.0 / b, p))), B=b, two_tailed=True)


def holm(pvals):
    """Holm-Bonferroni, returning adjusted p-values in input order.
    python/armL_results.py::holm's rule, reimplemented (that file is on Delta).
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


# ----------------------------------------------------------------------------
# row resolution
# ----------------------------------------------------------------------------
# One row per (method, arch, city). Each row records which tree supplied it and
# whether that is the declared PNG tree or the recorded fallback.

METHOD_ORDER = ["upper", "vanilla", "fda", "segdesic", "iim", "isw",
                "mrfp_plus", "fada", "sib", "bsm", "sar"]
METHOD_LABEL = {"upper": "Upper bound", "vanilla": "Vanilla", "fda": "FDA",
                "segdesic": "SegDeSiC", "iim": "IIM", "isw": "ISW",
                "mrfp_plus": "MRFP+", "fada": "FADA", "sib": "SIB",
                "bsm": "BSM", "sar": "SAR"}
BASELINES = ["fda", "segdesic", "iim", "isw", "mrfp_plus", "fada", "bsm", "sar"]


def declared_run_id(method, arch, city):
    """The run_id the declared path looks for, before any n450 substitution."""
    if arch == CE:
        tok = {"upper": "vanilla-upper", "vanilla": "vanilla-loco",
               "fda": "fda-loco", "sib": "sib-loco", "bsm": "bsm-loco"}[method]
        return f"{CE}-{tok}-r450-{city}-hr-s0"
    if method == "upper":
        return f"{arch}-none-upper-r450-{city}-hr-s0"
    if method == "vanilla":
        return f"{arch}-vanilla-loco-r450-{city}-hr-s0"
    if method == "sib":
        return f"{arch}-sibc4-loco-r450-{city}-hr-s0"
    return f"{arch}-{method}-loco-r450-{city}-hr-s0"


def resolve_n450(method, arch, city):
    """Find a matched-450 retrain for this cell, if one has arrived.

    The retrain's run_id differs from the published cell's only in its seed token
    and a trailing suffix: the two cells already finished on Delta are
    `mamnet-vanilla-loco-r450-phoenix-hr-s42-n450` and
    `oglanet-sibc4-loco-r450-phoenix-hr-s42-n450`. For OGLANet a plain `-n450`
    dump is NOT accepted: every dump-based OGLANet family is written at inference
    batch size 1, where DFFM's 20 unregistered batch-statistic layers behave as
    InstanceNorms and the cell scores 6-7 foreground-IoU points below the batch-4
    published trees (delta_runs/dffm_defect/provenance.json). Only an OGLANet
    retrain whose suffix also carries `bs4` is comparable with a published
    OGLANet row, so a batch-1 OGLANet n450 dump is reported as present and
    refused rather than silently substituted.
    """
    base = declared_run_id(method, arch, city)
    stem = re.sub(r"-s\d+$", "", base)
    pat = re.compile(re.escape(stem) + r"-s\d+-n450(?P<suffix>[-\w]*)$")
    found, refused = [], []
    for label, root in N450_TREES:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            m = pat.match(name)
            if not m:
                continue
            if not os.path.isfile(os.path.join(root, name, "perimage.parquet")):
                continue
            if arch == "oglanet" and "bs4" not in name:
                refused.append(dict(tree=label, run_id=name,
                                    reason="OGLANet n450 dump without a bs4 "
                                           "suffix is inference-batch-1 and is "
                                           "not comparable with the batch-4 "
                                           "published tree (DFFM)"))
                continue
            found.append((label, root, name))
    return found, refused


def resolve_row(method, arch, city):
    """Return the row's per-image two-class K=2 vector plus its provenance."""
    row = dict(method=method, method_label=METHOD_LABEL[method], arch=arch,
               city=city,
               access_class=ACCESS.get(f"{method}_{arch}", ACCESS.get(method)),
               table38_printed=(TABLE38[(method, arch)][CITIES.index(city)]
                                if (method, arch) in TABLE38 else None))
    n450_found, n450_refused = resolve_n450(method, arch, city)
    row["n450_candidates"] = [f"{t}/{n}" for t, _, n in n450_found]
    row["n450_refused"] = n450_refused
    row["n450_substituted"] = False

    if N450 and n450_found:
        label, root, rid = n450_found[0]
        df, meta, why = read_cell(root, rid)
        if df is not None:
            m2, fg, _ = two_class_tol_k2(df)
            row.update(run_id=rid, tree=label, source_kind="matched-450 retrain",
                       on_declared_path=False, fell_back=False,
                       n450_substituted=True, n=int(len(df)),
                       value=float(m2.mean()), value_fg=float(fg.mean()),
                       per_image=m2, image_id=list(df.image_id),
                       sidecar_target_access=meta.get("target_access"),
                       metric_module_sha=meta.get("metric_module_sha"),
                       empty_fg_images=int(df.empty_fg_valid.sum()),
                       blocker=None)
            return row

    # --- MAMNet SIB: the E-12 clean-pool retrains, three seeds -------------
    if method == "sib" and arch == "mamnet":
        seeds, per_seed = {}, []
        for tag, seed in (("s42-e12clean", 42), ("s43-e12clean", 43),
                          ("s44-e12clean", 44)):
            rid = f"mamnet-sibc4-loco-r450-{city}-hr-{tag}"
            df, meta, why = read_cell(E12, rid)
            if df is None:
                fail(f"{rid}: {why}")
            m2, fg, _ = two_class_tol_k2(df)
            stated = meta.get("two_class_tol_k2_pct")
            if stated is not None and abs(float(m2.mean()) - stated) > 1e-4:
                fail(f"{rid}: declared path {m2.mean():.6f} disagrees with the "
                     f"sidecar's two_class_tol_k2_pct {stated:.6f}")
            seeds[seed] = m2
            per_seed.append(dict(seed=seed, run_id=rid,
                                 value=float(m2.mean()), value_fg=float(fg.mean()),
                                 target_access=meta.get("target_access")))
            ids = list(df.image_id)
        pub = f"mamnet-sibc4-loco-r450-{city}-hr-sPUB-e12orig"
        dfp, metap, _ = read_cell(E12, pub)
        m2p, _, _ = two_class_tol_k2(dfp)
        stack = np.vstack([seeds[s] for s in (42, 43, 44)])
        row.update(run_id=[r["run_id"] for r in per_seed],
                   tree="delta_dumps/dumps/e12",
                   source_kind="E-12 clean-pool retrain, seed mean over "
                               "{42, 43, 44}",
                   on_declared_path=True, fell_back=False,
                   n=int(stack.shape[1]),
                   value=float(stack.mean(axis=0).mean()),
                   value_fg=float(np.mean([r["value_fg"] for r in per_seed])),
                   per_image=stack.mean(axis=0), image_id=ids,
                   per_seed=per_seed,
                   seed_values=[r["value"] for r in per_seed],
                   seed_sd=float(np.std([r["value"] for r in per_seed], ddof=1)),
                   published_contaminated_value=float(m2p.mean()),
                   published_contaminated_repro_delta=metap.get(
                       "repro_delta_miou"),
                   sidecar_target_access=per_seed[0]["target_access"],
                   metric_module_sha=metap.get("metric_module_sha"),
                   empty_fg_images=int(dfp.empty_fg_valid.sum()),
                   blocker=None)
        return row

    # --- SAR: declared-path unadapted published cell + SAR's own change ----
    if method == "sar":
        p = os.path.join(BASE, "runs", f"armF_sar_{arch}_{city}_highres")
        rj, pp = os.path.join(p, "results.json"), os.path.join(p, "perstep.parquet")
        if not (os.path.isfile(rj) and os.path.isfile(pp)):
            row.update(run_id=None, tree=None, source_kind=None,
                       on_declared_path=False, fell_back=False, value=None,
                       per_image=None, blocker="no arm F SAR run on this box")
            return row
        d = json.load(open(rj))
        ps = pd.read_parquet(pp)
        lab0 = ps[ps.pass_id == "unadapted"]
        if lab0.empty:
            fail(f"armF_sar_{arch}_{city}: no unadapted pass in perstep.parquet")
        sarp = ps[ps.pass_id != "unadapted"]
        if sarp.empty:
            fail(f"armF_sar_{arch}_{city}: no adapted pass in perstep.parquet")
        # step 0 and the final step of the SAR pass itself: the within-pass delta
        step0 = sarp[sarp.step == sarp.step.min()].sort_values("image_id")
        stepN = sarp[sarp.step == sarp.step.max()].sort_values("image_id")
        if list(step0.image_id) != list(stepN.image_id):
            fail(f"armF_sar_{arch}_{city}: SAR step-0 and final-step image sets "
                 "differ; a paired delta is not defined")
        m0 = two_class_from_counts(step0.tp_k2, step0.fp_k2, step0.fn_k2,
                                   step0.tn_k2)
        mN = two_class_from_counts(stepN.tp_k2, stepN.fp_k2, stepN.fn_k2,
                                   stepN.tn_k2)
        delta_per_image = mN - m0
        stated = d["sar"]["delta_adaptation_only_miou_tol2"]
        if abs(float(delta_per_image.mean()) - stated) > 1e-6:
            fail(f"armF_sar_{arch}_{city}: recomputed within-pass two-class "
                 f"delta {delta_per_image.mean():.8f} disagrees with the run's "
                 f"own delta_adaptation_only_miou_tol2 {stated:.8f}")
        # The published unadapted level. v3 change: the declared PNG tree now
        # carries all nine cross-city Vanilla cells, so the SAR unadapted level
        # is read from the PNG tree and the probability-map tree is only a
        # recorded fallback, as it is for every other row.
        base_rid = declared_run_id("vanilla", arch, city)
        bdf = bmeta = None
        base_tree, base_fell_back, bwhy = None, None, None
        for blabel, broot in (("st_r450_published_png", PNG),
                              ("st_r450_published_recompute", RECOMPUTE)):
            bdf, bmeta, bwhy = read_cell(broot, base_rid)
            if bdf is not None:
                base_tree = blabel
                base_fell_back = (blabel == "st_r450_published_recompute")
                break
        if bdf is None:
            row.update(run_id=None, tree=None, source_kind=None,
                       on_declared_path=False, fell_back=True, value=None,
                       per_image=None,
                       blocker=f"no unadapted published cell: {bwhy}")
            return row
        bm2, bfg, _ = two_class_tol_k2(bdf)
        # The SAR change and the published unadapted level come from two
        # artifacts with their own row orders, and the row below adds them per
        # image, so both sides are put in image_id order and the alignment is
        # asserted rather than assumed.
        base_sorted = bdf.sort_values("image_id").reset_index(drop=True)
        if list(base_sorted.image_id) != list(step0.image_id):
            fail(f"armF_sar_{arch}_{city}: the SAR pass and the published "
                 "unadapted cell do not carry the same 150 image_ids in "
                 "image_id order, so SAR's change cannot be added per image")
        bm2_sorted, _, _ = two_class_tol_k2(base_sorted)
        per_image = bm2_sorted + delta_per_image
        row.update(run_id=f"armF_sar_{arch}_{city}_highres",
                   tree="runs/armF_sar_*",
                   source_kind=f"published unadapted cell ({base_tree}) plus "
                               "SAR's own within-pass change; ASSUMPTION, "
                               "caveat 4",
                   on_declared_path=False, fell_back=bool(base_fell_back),
                   unadapted_base_tree=base_tree,
                   n=int(per_image.size), value=float(per_image.mean()),
                   value_fg=None, per_image=per_image,
                   image_id=list(base_sorted.image_id),
                   unadapted_base_run_id=base_rid,
                   unadapted_base_value=float(bm2.mean()),
                   sar_own_unadapted_value=d["unadapted"][
                       "miou_tol2_derived_from_counts"]["value"],
                   sar_own_unadapted_minus_declared_base=float(
                       d["unadapted"]["miou_tol2_derived_from_counts"]["value"]
                       - bm2.mean()),
                   sar_delta_per_image=delta_per_image,
                   sar_delta_cell_mean=float(delta_per_image.mean()),
                   sar_diverged=bool(d["sar"]["diverged"]),
                   sar_batch_size=d["provenance"].get("batch_size"),
                   sidecar_target_access=d.get("target_access"),
                   metric_module_sha=d["provenance"].get("metric_module_sha256"),
                   empty_fg_images=int(bdf.empty_fg_valid.sum()),
                   blocker=None)
        return row

    # --- everything else: PNG tree first, recompute as the fallback --------
    rid = declared_run_id(method, arch, city)
    trees = ([("st_r450_published_png", PNG),
              ("st_r450_published_recompute", RECOMPUTE)]
             if arch != CE else [("st", ST)])
    if method == "bsm" and arch != CE:
        trees = [("st", ST)]
    for i, (label, root) in enumerate(trees):
        df, meta, why = read_cell(root, rid)
        if df is None:
            continue
        m2, fg, _ = two_class_tol_k2(df)
        row.update(run_id=rid, tree=label,
                   source_kind=(meta.get("source")
                                or ("arm F / arm C-D training run"
                                    if label == "st" else None)),
                   on_declared_path=(label == "st_r450_published_png"),
                   fell_back=(label == "st_r450_published_recompute"),
                   n=int(len(df)), value=float(m2.mean()),
                   value_fg=float(fg.mean()), per_image=m2,
                   image_id=list(df.image_id),
                   png_table38_delta=meta.get("table38_delta"),
                   png_table38_gate_pass=meta.get("table38_gate_pass"),
                   png_final_comparison_delta=meta.get(
                       "final_comparison_delta"),
                   png_vs_recompute_delta=(
                       (meta.get("recompute_counterpart") or {}).get(
                           "delta_png_minus_recompute_miou_k2")),
                   sidecar_target_access=meta.get("target_access"),
                   metric_module_sha=meta.get("metric_module_sha"),
                   empty_fg_images=int(df.empty_fg_valid.sum()),
                   blocker=None)
        return row
    row.update(run_id=rid, tree=None, source_kind=None,
               on_declared_path=False, fell_back=False, value=None,
               per_image=None,
               blocker=("no dump in the declared PNG tree and none in the "
                        "recompute fallback"))
    return row


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

PNG_VANILLA_CELLS = [f"{a}-vanilla-loco-r450-{c}-hr-s0"
                     for a in ARCHS for c in CITIES]


def assert_png_vanilla_present():
    """v3's entry condition: the nine cross-city Vanilla cells must be in the
    declared PNG tree, each complete and each marked .DONE. Any absence stops
    the job; v3 exists only to move those nine rows, and the nine SAR rows
    built on them, off the probability-map fallback.
    """
    state, missing = [], []
    for rid in PNG_VANILLA_CELLS:
        d = os.path.join(PNG, rid)
        rec = dict(run_id=rid,
                   dir_present=os.path.isdir(d),
                   done_marker=os.path.isfile(os.path.join(d, ".DONE")),
                   failed_marker=os.path.isfile(os.path.join(d, ".FAILED")),
                   has_perimage=os.path.isfile(
                       os.path.join(d, "perimage.parquet")),
                   has_run_json=os.path.isfile(os.path.join(d, "run.json")))
        if rec["has_run_json"]:
            m = json.load(open(os.path.join(d, "run.json")))
            rec.update(
                final_comparison_delta=m.get("final_comparison_delta"),
                final_comparison_gate_pass=m.get("final_comparison_gate_pass"),
                png_miou_k2_twoclass=m.get("headline_miou_k2_twoclass"),
                delta_png_minus_recompute_miou_k2=(
                    (m.get("recompute_counterpart") or {}).get(
                        "delta_png_minus_recompute_miou_k2")))
        bad = [k for k in ("dir_present", "done_marker", "has_perimage",
                           "has_run_json") if not rec[k]]
        if bad or rec["failed_marker"]:
            missing.append(f"{rid}: " + ", ".join(bad + (
                [".FAILED marker present"] if rec["failed_marker"] else [])))
        state.append(rec)
    if missing:
        fail("the nine PNG-scored cross-city Vanilla cells are the entry "
             "condition for v3 and are not all present:\n  "
             + "\n  ".join(missing))
    return state


def main():
    hr = "=" * 100
    print(hr)
    print("Arm F / E-12 / E-2 -- internal SIB worst-case adjudication, v3")
    print(f"DRY_RUN = {DRY_RUN}   N450 = {N450}   B = {B}")
    print(hr)

    # ---- v3 entry condition ---------------------------------------------
    png_vanilla_state = assert_png_vanilla_present()
    print("\n--- v3 entry condition: the nine PNG-scored Vanilla cells ---")
    for r in png_vanilla_state:
        print(f"  {r['run_id']:42s} .DONE  final_comparison_delta "
              f"{r['final_comparison_delta']}  PNG minus recompute "
              f"{r['delta_png_minus_recompute_miou_k2']:+.4f}")

    # ---- indexes ---------------------------------------------------------
    trees = [("st_r450_published_png", PNG),
             ("st_r450_published_recompute", RECOMPUTE),
             ("st", ST),
             ("e12", E12)]
    indexes, built = {}, {}
    for label, root in trees:
        # v3 writes its own index files. results/12 cites
        # analysis/_inv/INDEX_{label}.parquet by sha256, and the PNG tree has
        # changed since those were written (nine Vanilla cells arrived), so
        # rewriting them in place would invalidate a published provenance
        # citation. v3's indexes carry a _v3 suffix and v2's are left untouched.
        idx, info = build_index(
            label, root,
            write_to=os.path.join(INV, f"INDEX_{label}_v3.parquet"))
        built[label] = idx
        indexes[label] = info
    on_disk = {label: on_disk_index_state(label, root)
               for label, root in trees}
    png_idx = built["st_r450_published_png"]
    png_info = indexes["st_r450_published_png"]
    print("\n--- indexes (CLAUDE.md: start from the index, never from a glob) ---")
    for lbl, i in indexes.items():
        o = on_disk[lbl]
        print(f"  {lbl:32s} built rows {i['n_rows']:4d}  sha256 "
              f"{i['sha256'][:16]}...  on-disk index "
              f"{'absent' if not o['present'] else ('STALE' if o['stale'] else 'current')}")
    print(f"  PNG tree unindexable dirs: {png_info['n_unindexable']}")

    png_names = set(png_idx.run_id.astype(str))
    print(f"  PNG tree cells: {len(png_names)} "
          f"(hr {sum('-hr-' in n for n in png_names)}, "
          f"mr {sum('-mr-' in n for n in png_names)})")

    # ---- declared-path reproduction check on the PNG tree ----------------
    print("\n--- declared-path reproduction, as recorded by each PNG dump ---")
    t38d, fcd = [], []
    for n in sorted(png_names):
        meta = json.load(open(os.path.join(PNG, n, "run.json")))
        if meta.get("table38_delta") is not None:
            t38d.append((n, abs(meta["table38_delta"])))
        if meta.get("final_comparison_delta") is not None:
            fcd.append((n, abs(meta["final_comparison_delta"])))
    repro = dict(
        n_cells_with_table38=len(t38d),
        max_abs_delta_vs_table38=max(v for _, v in t38d),
        worst_cell_vs_table38=max(t38d, key=lambda x: x[1])[0],
        n_cells_with_final_comparison=len(fcd),
        max_abs_delta_vs_final_comparison=max(v for _, v in fcd),
        n_final_comparison_exact=sum(1 for _, v in fcd if v == 0.0))
    print(f"  Table 38: {repro['n_cells_with_table38']} cells, max |delta| "
          f"{repro['max_abs_delta_vs_table38']:.4f} "
          f"({repro['worst_cell_vs_table38']})")
    print(f"  final_comparison.json: {repro['n_cells_with_final_comparison']} "
          f"cells, max |delta| {repro['max_abs_delta_vs_final_comparison']:.6f}, "
          f"exact on {repro['n_final_comparison_exact']}")

    # ---- resolve every row -----------------------------------------------
    print("\n--- resolving rows ---")
    rows = {}
    for method in METHOD_ORDER:
        for arch in ARCHS:
            for city in CITIES:
                rows[(method, arch, city)] = resolve_row(method, arch, city)
    for method in ("upper", "vanilla", "fda", "sib", "bsm"):
        for city in CITIES:
            rows[(method, CE, city)] = resolve_row(method, CE, city)

    # image-id identity across trees, asserted not assumed
    by_city = {}
    for k, r in rows.items():
        if r.get("image_id"):
            by_city.setdefault(r["city"], []).append((k, r["image_id"]))
    # Every paired delta below is a positional subtraction of two per-image
    # vectors, so the ORDER has to match, not only the set. Both are checked.
    id_check = {}
    for city, lst in by_city.items():
        ref = lst[0][1]
        bad_set = [f"{m}/{a}" for (m, a, _), ids in lst
                   if sorted(ids) != sorted(ref)]
        if bad_set:
            fail(f"{city}: image_id set differs on {bad_set}; a paired "
                 "bootstrap across trees is not defined")
        bad_order = [f"{m}/{a}" for (m, a, _), ids in lst if ids != ref]
        if bad_order:
            fail(f"{city}: image_id ORDER differs on {bad_order}; the paired "
                 "deltas here are positional subtractions, so a reordered "
                 "cell would pair the wrong images")
        id_check[city] = dict(n_cells=len(lst), n_images=len(ref),
                              same_set=True, same_order=True,
                              reference_cell=f"{lst[0][0][0]}/{lst[0][0][1]}",
                              order_is_lexicographic=(ref == sorted(ref)))
    print("  image_id sets AND order identical within every city: "
          + ", ".join(f"{c} ({v['n_cells']} cells, n={v['n_images']})"
                      for c, v in sorted(id_check.items())))

    # ---- per-cell gaps ---------------------------------------------------
    per_cell, cell_pvals = [], {}
    for (method, arch, city), r in sorted(rows.items()):
        if method == "upper":
            continue
        u = rows[("upper", arch, city)]
        rec = dict(method=method, method_label=METHOD_LABEL[method], arch=arch,
                   city=city, access_class=r["access_class"],
                   scope=("9_cell" if arch != CE else "crossearth"),
                   run_id=r.get("run_id"), tree=r.get("tree"),
                   source_kind=r.get("source_kind"),
                   unadapted_base_tree=r.get("unadapted_base_tree"),
                   on_declared_path=r.get("on_declared_path"),
                   fell_back=r.get("fell_back"),
                   n450_substituted=r.get("n450_substituted"),
                   n450_candidates=r.get("n450_candidates"),
                   n450_refused=r.get("n450_refused"),
                   upper_run_id=u.get("run_id"), upper_tree=u.get("tree"),
                   method_value=r.get("value"), upper_value=u.get("value"),
                   table38_printed=r.get("table38_printed"),
                   sidecar_target_access=r.get("sidecar_target_access"),
                   sidecar_access_disagrees=(
                       r.get("sidecar_target_access") is not None
                       and r.get("sidecar_target_access") != r["access_class"]),
                   blocker=r.get("blocker"))
        if r.get("per_image") is None or u.get("per_image") is None:
            rec.update(gap=None, gap_upper_minus_method=None, test=None)
            per_cell.append(rec)
            continue
        d = np.asarray(r["per_image"]) - np.asarray(u["per_image"])
        bs = paired_bootstrap(d)
        rec.update(gap=float(d.mean()),
                   gap_upper_minus_method=float(-d.mean()),
                   n=int(d.size),
                   stderr=bs["stderr"], ci95_gap=bs["ci95"],
                   ci95_gap_upper_minus_method=[-bs["ci95"][1], -bs["ci95"][0]],
                   ci_method=bs["ci_method"], p=bs["p"],
                   test="paired image-level bootstrap on per-image two-class "
                        "K=2 deltas, method minus its within-city upper bound, "
                        "two-tailed",
                   paired_across_trees=(r.get("tree") != u.get("tree")),
                   d_vs_table38=(None if r.get("table38_printed") is None
                                 else float(r["value"] - r["table38_printed"])))
        cell_pvals.setdefault(method, []).append((len(per_cell), bs["p"]))
        per_cell.append(rec)

    # Holm-Bonferroni within each method's cell family
    for method, lst in cell_pvals.items():
        adj = holm([p for _, p in lst])
        for (i, _), a in zip(lst, adj):
            per_cell[i]["p_holm_within_method_family"] = a
            per_cell[i]["holm_family"] = (
                f"{METHOD_LABEL[method]} cells vs their own upper bounds, "
                f"m = {len(lst)}")

    pc = pd.DataFrame(per_cell)
    pd.set_option("display.width", 240)
    print("\n--- per-cell gap, upper minus method, two-class K=2 (percent) ---")
    show = pc[pc.scope == "9_cell"][
        ["method", "arch", "city", "method_value", "upper_value",
         "gap_upper_minus_method", "p", "on_declared_path", "fell_back"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n--- CrossEarth cells (its own upper bound) ---")
    print(pc[pc.scope == "crossearth"][
        ["method", "city", "method_value", "upper_value",
         "gap_upper_minus_method", "p"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---- the missing-cell bound ------------------------------------------
    # OGLANet-SIB miami and phoenix have no dump in either tree. Their only
    # machine-readable evidence is Table 38's printed 2-dp value. Scoring the
    # declared tree reproduces Table 38 to <= 0.0072 on all 59 method cells, so
    # the printed value bounds those two cells to about +-0.01. The bound is
    # reported so the reading can be adjudicated; it is never a row value.
    bound = []
    for (method, arch, city), r in sorted(rows.items()):
        if method == "upper" or r.get("per_image") is not None:
            continue
        if r.get("table38_printed") is None:
            continue
        u = rows[("upper", arch, city)]
        if u.get("value") is None:
            continue
        bound.append(dict(
            method=method, arch=arch, city=city,
            table38_printed_method=r["table38_printed"],
            declared_path_upper=u["value"],
            bound_gap_upper_minus_method=float(u["value"]
                                               - r["table38_printed"]),
            basis="Table 38's printed 2-dp value for the method cell against "
                  "the declared-path PNG upper bound; the PNG tree reproduces "
                  "Table 38 to at most 0.0072 on 59 of 59 method cells, so the "
                  "bound is tight to about +-0.01",
            is_a_row_value=False))
    if bound:
        print("\n--- missing-cell bound from Table 38 (not a row value) ---")
        print(pd.DataFrame(bound)[
            ["method", "arch", "city", "table38_printed_method",
             "declared_path_upper", "bound_gap_upper_minus_method"]].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---- per-method, per-scope summaries ---------------------------------
    def cells_of(method, scope):
        if scope == "9_cell":
            return [r for r in per_cell
                    if r["method"] == method and r["arch"] != CE]
        return [r for r in per_cell if r["method"] == method]

    def cluster_input(recs):
        out = {}
        for r in recs:
            rr = rows[(r["method"], r["arch"], r["city"])]
            u = rows[("upper", r["arch"], r["city"])]
            if rr.get("per_image") is None or u.get("per_image") is None:
                continue
            out.setdefault(r["city"], []).append(
                np.asarray(u["per_image"]) - np.asarray(rr["per_image"]))
        return out

    per_method = []
    for scope in ("9_cell", "12_cell"):
        n_expected = 9 if scope == "9_cell" else 12
        for method in METHOD_ORDER:
            if method == "upper":
                continue
            recs = cells_of(method, scope)
            have = [r for r in recs if r["gap_upper_minus_method"] is not None]
            miss = [f"{r['arch']}/{r['city']}" for r in recs
                    if r["gap_upper_minus_method"] is None]
            no_ce = (scope == "12_cell"
                     and not any(r["arch"] == CE for r in recs))
            entry = dict(
                method=method, method_label=METHOD_LABEL[method], scope=scope,
                access_class=sorted({r["access_class"] for r in recs}),
                n_cells_expected=(9 if no_ce else n_expected),
                n_cells_available=len(have),
                cells_missing=miss,
                has_crossearth_counterpart=not no_ce,
                archs=sorted({r["arch"] for r in have}))
            if not have:
                entry.update(worst_case_gap=None, worst_cell=None,
                             mean_gap=None, mean_gap_ci95=None,
                             mean_gap_stderr=None, mean_gap_p=None)
                per_method.append(entry)
                continue
            w = max(have, key=lambda r: r["gap_upper_minus_method"])
            cb = cluster_bootstrap_mean_over_cells(cluster_input(have))
            entry.update(
                worst_case_gap=w["gap_upper_minus_method"],
                worst_cell=f"{w['arch']}/{w['city']}",
                worst_cell_run_id=w["run_id"],
                worst_cell_ci95=w["ci95_gap_upper_minus_method"],
                worst_cell_p=w["p"],
                best_case_gap=min(r["gap_upper_minus_method"] for r in have),
                mean_gap=cb["mean"], mean_gap_stderr=cb["stderr"],
                mean_gap_ci95=cb["ci95"], mean_gap_ci_method=cb["ci_method"],
                mean_gap_p=cb["p"],
                per_cell_gaps={f"{r['arch']}/{r['city']}":
                               r["gap_upper_minus_method"] for r in have},
                n_cells_on_declared_path=sum(1 for r in have
                                             if r["on_declared_path"]),
                n_cells_fell_back=sum(1 for r in have if r["fell_back"]))
            per_method.append(entry)

    pm = pd.DataFrame(per_method)
    print("\n--- worst case and mean gap per method, per scope "
          "(upper minus method) ---")
    print(pm[["scope", "method", "n_cells_available", "n_cells_expected",
              "worst_case_gap", "worst_cell", "mean_gap", "mean_gap_p",
              "access_class"]].to_string(index=False,
                                         float_format=lambda v: f"{v:+.4f}"))

    # ---- matched-cell comparison -----------------------------------------
    # SIB is a 7-of-9 row, and the two absent cells are OGLANet/Miami and
    # OGLANet/Phoenix, the architecture with the largest gaps. Comparing a
    # 7-cell worst case against a 9-cell worst case favours SIB by construction,
    # so the primary comparison restricts every method to the cells where SIB
    # exists, and the as-available comparison is reported beside it.
    sib_cells_9 = {(r["arch"], r["city"]) for r in cells_of("sib", "9_cell")
                   if r["gap_upper_minus_method"] is not None}
    sib_cells_12 = {(r["arch"], r["city"]) for r in cells_of("sib", "12_cell")
                    if r["gap_upper_minus_method"] is not None}
    matched = []
    for scope, keep in (("9_cell", sib_cells_9), ("12_cell", sib_cells_12)):
        for method in METHOD_ORDER:
            if method == "upper":
                continue
            have = [r for r in cells_of(method, scope)
                    if (r["arch"], r["city"]) in keep
                    and r["gap_upper_minus_method"] is not None]
            if not have:
                continue
            w = max(have, key=lambda r: r["gap_upper_minus_method"])
            cb = cluster_bootstrap_mean_over_cells(cluster_input(have))
            matched.append(dict(
                scope=scope, method=method, method_label=METHOD_LABEL[method],
                n_cells=len(have), n_cells_in_matched_set=len(keep),
                cells=[f"{r['arch']}/{r['city']}" for r in have],
                worst_case_gap=w["gap_upper_minus_method"],
                worst_cell=f"{w['arch']}/{w['city']}",
                mean_gap=cb["mean"], mean_gap_ci95=cb["ci95"],
                mean_gap_stderr=cb["stderr"], mean_gap_p=cb["p"],
                access_class=sorted({r["access_class"] for r in have})))
    mt = pd.DataFrame(matched)
    print("\n--- matched-cell comparison: every method restricted to the "
          "cells where SIB exists ---")
    print(mt[["scope", "method", "n_cells", "worst_case_gap", "worst_cell",
              "mean_gap"]].to_string(index=False,
                                     float_format=lambda v: f"{v:+.4f}"))

    # ---- SIB's margin over the strongest baseline ------------------------
    margins = {}
    for scope in ("9_cell", "12_cell"):
        sib = next((e for e in per_method
                    if e["method"] == "sib" and e["scope"] == scope), None)
        cand = [e for e in per_method
                if e["scope"] == scope and e["method"] in BASELINES
                and e["mean_gap"] is not None]
        van = next((e for e in per_method
                    if e["method"] == "vanilla" and e["scope"] == scope), None)
        if sib is None or sib["mean_gap"] is None or not cand:
            margins[scope] = dict(strongest_baseline=None,
                                  why="no comparable baseline")
            continue
        strongest = min(cand, key=lambda e: e["mean_gap"])
        # per-image paired margin on the cells both rows cover
        keep = (set(strongest["per_cell_gaps"]) & set(sib["per_cell_gaps"]))
        per_city = {}
        for key in sorted(keep):
            arch, city = key.split("/")
            s = rows[("sib", arch, city)]
            b = rows[(strongest["method"], arch, city)]
            if s.get("per_image") is None or b.get("per_image") is None:
                continue
            per_city.setdefault(city, []).append(
                np.asarray(s["per_image"]) - np.asarray(b["per_image"]))
        cb = cluster_bootstrap_mean_over_cells(per_city) if per_city else None
        margins[scope] = dict(
            sib_mean_gap=sib["mean_gap"], sib_mean_gap_ci95=sib["mean_gap_ci95"],
            sib_n_cells=sib["n_cells_available"],
            strongest_baseline=strongest["method"],
            strongest_baseline_label=strongest["method_label"],
            strongest_baseline_mean_gap=strongest["mean_gap"],
            strongest_baseline_n_cells=strongest["n_cells_available"],
            strongest_baseline_access_class=strongest["access_class"],
            margin_mean_gap_difference=float(strongest["mean_gap"]
                                             - sib["mean_gap"]),
            margin_definition="strongest baseline's mean gap minus SIB's mean "
                              "gap; positive means SIB sits closer to the "
                              "within-city upper bound",
            paired_margin_cells=sorted(keep),
            paired_margin=cb,
            paired_margin_definition="per image, SIB minus the strongest "
                                     "baseline on the cells both rows cover, "
                                     "averaged over cells, image-cluster "
                                     "bootstrap by held-out city",
            vanilla_mean_gap=(van or {}).get("mean_gap"),
            baseline_mean_gaps={e["method"]: e["mean_gap"] for e in cand},
            crosses_access_class=(sorted({a for e in (sib, strongest)
                                          for a in e["access_class"]})
                                  if len({tuple(sib["access_class"]),
                                          tuple(strongest["access_class"])}) > 1
                                  else None))
    print("\n--- SIB mean gap and margin over the strongest baseline ---")
    for scope, m in margins.items():
        if not m.get("strongest_baseline"):
            print(f"  {scope}: {m['why']}")
            continue
        pm_ = m["paired_margin"]
        print(f"  {scope}: SIB mean gap {m['sib_mean_gap']:+.4f} "
              f"(n_cells {m['sib_n_cells']}); strongest baseline "
              f"{m['strongest_baseline_label']} {m['strongest_baseline_mean_gap']:+.4f}; "
              f"margin {m['margin_mean_gap_difference']:+.4f}")
        if pm_:
            print(f"      paired margin {pm_['mean']:+.4f} "
                  f"CI95 [{pm_['ci95'][0]:+.4f}, {pm_['ci95'][1]:+.4f}] "
                  f"p {pm_['p']:.4f} over {pm_['n_cells']} cells")

    # ---- within-access-class rankings -----------------------------------
    rankings = {}
    for scope in ("9_cell", "12_cell"):
        for cls in ("source_only", "unlabeled_target", "deployment_time"):
            ent = [e for e in per_method
                   if e["scope"] == scope and e["mean_gap"] is not None
                   and e["access_class"] == [cls]]
            if not ent:
                continue
            rankings[f"{scope}/{cls}"] = dict(
                access_class=cls, scope=scope, n_methods=len(ent),
                by_worst_case=[dict(method=e["method"],
                                    worst_case_gap=e["worst_case_gap"],
                                    worst_cell=e["worst_cell"],
                                    n_cells=e["n_cells_available"])
                               for e in sorted(ent,
                                               key=lambda x: x["worst_case_gap"])],
                by_mean_gap=[dict(method=e["method"], mean_gap=e["mean_gap"],
                                  ci95=e["mean_gap_ci95"],
                                  n_cells=e["n_cells_available"])
                             for e in sorted(ent, key=lambda x: x["mean_gap"])])
    # SIB's cells do not share one access class: MAMNet-SIB is
    # unlabeled_target through FDA while OGLANet-SIB and DINOv3-SIB are
    # source_only, so the whole-method ranking above cannot place SIB at all.
    # The second view splits every method by its CELLS' access class, which is
    # the only way a same-class comparison can include SIB.
    rankings_by_cell = {}
    for scope in ("9_cell", "12_cell"):
        for cls in ("source_only", "unlabeled_target", "deployment_time",
                    "target_labels"):
            ent = []
            for method in METHOD_ORDER:
                if method == "upper":
                    continue
                have = [r for r in cells_of(method, scope)
                        if r["access_class"] == cls
                        and r["gap_upper_minus_method"] is not None]
                if not have:
                    continue
                w = max(have, key=lambda r: r["gap_upper_minus_method"])
                cb = cluster_bootstrap_mean_over_cells(cluster_input(have))
                ent.append(dict(
                    method=method, method_label=METHOD_LABEL[method],
                    n_cells=len(have),
                    cells=[f"{r['arch']}/{r['city']}" for r in have],
                    worst_case_gap=w["gap_upper_minus_method"],
                    worst_cell=f"{w['arch']}/{w['city']}",
                    mean_gap=cb["mean"], mean_gap_ci95=cb["ci95"],
                    mean_gap_stderr=cb["stderr"], mean_gap_p=cb["p"]))
            if not ent:
                continue
            rankings_by_cell[f"{scope}/{cls}"] = dict(
                access_class=cls, scope=scope, n_methods=len(ent),
                by_worst_case=sorted(ent, key=lambda e: e["worst_case_gap"]),
                by_mean_gap=sorted(ent, key=lambda e: e["mean_gap"]),
                note="each method is restricted to the cells whose own access "
                     "class is this one, so a method whose cells span classes "
                     "(SIB) appears in more than one ranking with different "
                     "cells, and the n_cells column is not constant down a "
                     "ranking")
    print("\n--- within-access-class ranking, whole methods, by worst case "
          "(smallest first) ---")
    for k, v in sorted(rankings.items()):
        print(f"  {k}: " + " < ".join(
            f"{e['method']} {e['worst_case_gap']:+.3f}"
            for e in v["by_worst_case"]))
    print("\n--- within-access-class ranking, by CELL access class "
          "(places SIB) ---")
    for k, v in sorted(rankings_by_cell.items()):
        print(f"  {k}: " + " < ".join(
            f"{e['method']}[{e['n_cells']}] {e['worst_case_gap']:+.3f}"
            for e in v["by_worst_case"]))

    # ---- vanilla's worst case -------------------------------------------
    vanilla_worst = {}
    for scope in ("9_cell", "12_cell"):
        e = next((x for x in per_method
                  if x["method"] == "vanilla" and x["scope"] == scope), None)
        vanilla_worst[scope] = dict(
            worst_case_gap=e["worst_case_gap"], worst_cell=e["worst_cell"],
            worst_cell_ci95=e.get("worst_cell_ci95"),
            worst_cell_p=e.get("worst_cell_p"),
            mean_gap=e["mean_gap"], mean_gap_ci95=e["mean_gap_ci95"],
            n_cells=e["n_cells_available"],
            all_cells_fell_back=bool(e["n_cells_fell_back"]
                                     and not e["n_cells_on_declared_path"]))
    print("\n--- Vanilla worst case ---")
    for scope, v in vanilla_worst.items():
        print(f"  {scope}: {v['worst_case_gap']:+.4f} at {v['worst_cell']} "
              f"(n_cells {v['n_cells']})")

    # ---- the pre-committed readings --------------------------------------
    def adjudicate(basis_rows, scope):
        sib = next((e for e in basis_rows
                    if e["method"] == "sib" and e["scope"] == scope), None)
        others = [e for e in basis_rows
                  if e["scope"] == scope and e["method"] != "sib"
                  and e.get("worst_case_gap") is not None]
        if sib is None or sib.get("worst_case_gap") is None:
            return dict(sib_worst_case=None,
                        verdict="not adjudicable; SIB has no worst case")
        ties = [dict(method=e["method"], worst_case_gap=e["worst_case_gap"],
                     worst_cell=e["worst_cell"])
                for e in others
                if e["worst_case_gap"] <= sib["worst_case_gap"]]
        van = next((e for e in others if e["method"] == "vanilla"), None)
        return dict(
            sib_worst_case=sib["worst_case_gap"],
            sib_worst_cell=sib["worst_cell"],
            sib_n_cells=sib.get("n_cells", sib.get("n_cells_available")),
            methods_with_worst_case_at_or_below_sib=ties,
            sib_is_strictly_smallest=(len(ties) == 0),
            vanilla_worst_case=(van or {}).get("worst_case_gap"),
            vanilla_worst_cell=(van or {}).get("worst_cell"),
            sib_at_or_above_vanilla=(
                None if van is None
                else bool(sib["worst_case_gap"] >= van["worst_case_gap"])),
            ranking=[dict(method=e["method"],
                          worst_case_gap=e["worst_case_gap"],
                          worst_cell=e["worst_cell"])
                     for e in sorted(others + [sib],
                                     key=lambda x: x["worst_case_gap"])])

    adj = dict(
        matched_cell={s: adjudicate(matched, s)
                      for s in ("9_cell", "12_cell")},
        as_available={s: adjudicate(per_method, s)
                      for s in ("9_cell", "12_cell")})

    def fire(a):
        """R1-R4 from one basis. R1/R2 are joint over the two scopes; R3 is the
        9-cell scope; R4 is either scope."""
        s9, s12 = a["9_cell"], a["12_cell"]
        out = {}
        out["R1"] = bool(s9.get("sib_is_strictly_smallest")
                         and s12.get("sib_is_strictly_smallest"))
        out["R2"] = bool(s9.get("sib_is_strictly_smallest")
                         and not s12.get("sib_is_strictly_smallest"))
        out["R3"] = bool(s9.get("methods_with_worst_case_at_or_below_sib"))
        out["R4"] = bool(s9.get("sib_at_or_above_vanilla")
                         or s12.get("sib_at_or_above_vanilla"))
        return out

    fired = {b: fire(adj[b]) for b in adj}
    primary = "matched_cell"

    def one_reading(basis, scope):
        f = fired[basis]
        a = adj[basis][scope]
        if a.get("sib_worst_case") is None:
            return None
        if scope == "9_cell":
            if f["R3"]:
                return "R3"
            return "R1" if f["R1"] else ("R2" if f["R2"] else "R3")
        return "R1" if f["R1"] else "R2"

    reading = dict(
        basis=primary,
        basis_why=("SIB is a 7-of-9 row and the two absent cells are OGLANet/"
                   "Miami and OGLANet/Phoenix. Comparing a 7-cell worst case "
                   "against a 9-cell worst case favours SIB by construction, so "
                   "the reading rests on the matched-cell comparison, in which "
                   "every method is restricted to the cells where SIB exists. "
                   "The as-available comparison is reported beside it and the "
                   "two are checked for agreement."),
        scope_9_cell=one_reading(primary, "9_cell"),
        scope_12_cell=one_reading(primary, "12_cell"),
        scope_12_cell_note=("Only Upper, Vanilla, FDA, SIB and BSM have a "
                            "CrossEarth counterpart, so on the 12-cell scope "
                            "SIB's worst case is taken over 10 cells while "
                            "SegDeSiC, IIM, ISW, MRFP+, FADA and SAR are still "
                            "9-cell rows. Adding cells can only raise a worst "
                            "case, so SIB being smallest over a superset of "
                            "those methods' cells is the conservative "
                            "direction of the asymmetry."),
        scope_9_cell_as_available=one_reading("as_available", "9_cell"),
        scope_12_cell_as_available=one_reading("as_available", "12_cell"),
        bases_agree=(one_reading(primary, "9_cell")
                     == one_reading("as_available", "9_cell")
                     and one_reading(primary, "12_cell")
                     == one_reading("as_available", "12_cell")),
        r4_also_fires={b: fired[b]["R4"] for b in fired},
        boolean_evaluation=fired,
        adjudication=adj,
        missing_cell_bound=bound,
        missing_cell_bound_changes_the_worst_case=None)

    # does the Table 38 bound on the missing cells change SIB's worst case?
    sib_m = adj[primary]["9_cell"]
    sib_bound = [b for b in bound if b["method"] == "sib"]
    if sib_m.get("sib_worst_case") is not None and sib_bound:
        worst_bound = max(b["bound_gap_upper_minus_method"] for b in sib_bound)
        reading["missing_cell_bound_changes_the_worst_case"] = dict(
            sib_worst_case_observed=sib_m["sib_worst_case"],
            sib_worst_case_observed_cell=sib_m["sib_worst_cell"],
            largest_bounded_missing_cell_gap=worst_bound,
            missing_cells=[f"{b['arch']}/{b['city']}" for b in sib_bound],
            changes_it=bool(worst_bound > sib_m["sib_worst_case"]),
            statement=("The two absent OGLANet-SIB cells are bounded by Table "
                       "38's printed values at "
                       + ", ".join(f"{b['arch']}/{b['city']} "
                                   f"{b['bound_gap_upper_minus_method']:+.2f}"
                                   for b in sib_bound)
                       + f". The largest of those, {worst_bound:+.2f}, is "
                       + ("ABOVE" if worst_bound > sib_m["sib_worst_case"]
                          else "below")
                       + f" the observed SIB worst case of "
                         f"{sib_m['sib_worst_case']:+.2f} at "
                         f"{sib_m['sib_worst_cell']}, so filling those two "
                         "cells "
                       + ("WOULD" if worst_bound > sib_m["sib_worst_case"]
                          else "would not")
                       + " move SIB's worst case."))

    print("\n" + hr)
    print("PRE-COMMITTED READING")
    print(f"  basis               : {reading['basis']}")
    print(f"  9-cell scope        : {reading['scope_9_cell']}")
    print(f"  12-cell scope       : {reading['scope_12_cell']}")
    print(f"  as-available 9 / 12 : {reading['scope_9_cell_as_available']} / "
          f"{reading['scope_12_cell_as_available']}")
    print(f"  bases agree         : {reading['bases_agree']}")
    print(f"  R4 also fires       : {reading['r4_also_fires']}")
    for b in ("matched_cell", "as_available"):
        a = adj[b]["9_cell"]
        print(f"  [{b}] SIB worst case {a['sib_worst_case']:+.4f} at "
              f"{a['sib_worst_cell']}; methods at or below SIB: "
              + (", ".join(f"{t['method']} {t['worst_case_gap']:+.3f}"
                           for t in a["methods_with_worst_case_at_or_below_sib"])
                 or "none"))
    print(hr)

    # ---- CONTRACT section 6 conditions and tests -------------------------
    conditions, tests = [], []
    for e in per_method:
        if e["mean_gap"] is None:
            continue
        conditions.append(dict(
            label=f"{e['method_label']} / {e['scope']} / mean gap to the "
                  f"within-city upper bound",
            metric="gap_upper_minus_method_mIoU_tol_K2_pct",
            mean=e["mean_gap"], stderr=e["mean_gap_stderr"],
            n=e["n_cells_available"],
            n_images=sum(150 for _ in range(e["n_cells_available"])),
            ci95=e["mean_gap_ci95"], ci_method=e["mean_gap_ci_method"],
            baseline="within-city upper bound, same architecture and city",
            access_class=e["access_class"],
            worst_case_gap=e["worst_case_gap"], worst_cell=e["worst_cell"]))
        tests.append(dict(
            name="paired image-cluster bootstrap",
            compares=[f"{e['method']}/{e['scope']}", "upper"],
            statistic=e["mean_gap"], p=e["mean_gap_p"], B=B,
            two_tailed=True,
            clustering="images resampled within held-out city, shared across "
                       "the cells of that city"))
    for r in per_cell:
        if r.get("p") is None:
            continue
        tests.append(dict(
            name="paired image-level bootstrap",
            compares=[r["run_id"] if isinstance(r["run_id"], str)
                      else ",".join(r["run_id"]), r["upper_run_id"]],
            statistic=r["gap_upper_minus_method"], p=r["p"],
            p_holm=r.get("p_holm_within_method_family"),
            holm_family=r.get("holm_family"), B=B, two_tailed=True,
            correction="Holm-Bonferroni within the method's cell family"))
    for scope, m in margins.items():
        if m.get("paired_margin"):
            tests.append(dict(
                name="paired image-cluster bootstrap, SIB minus the strongest "
                     "baseline",
                compares=["sib", m["strongest_baseline"]],
                statistic=m["paired_margin"]["mean"],
                p=m["paired_margin"]["p"], B=B, two_tailed=True,
                scope=scope,
                note="no multiplicity correction: one pre-specified comparison "
                     "per scope"))
    for r in per_cell:
        if r["method"] != "sar":
            continue
        rr = rows[("sar", r["arch"], r["city"])]
        if rr.get("sar_delta_per_image") is None:
            continue
        bs = paired_bootstrap(rr["sar_delta_per_image"])
        tests.append(dict(
            name="paired image-level bootstrap on SAR's within-pass change",
            compares=[f"{rr['run_id']} final step",
                      f"{rr['run_id']} step 0"],
            statistic=bs["mean"], p=bs["p"], ci95=bs["ci95"], B=B,
            two_tailed=True,
            note="the only part of a SAR row that is measured on one forward "
                 "path; the row's level is the published unadapted cell plus "
                 "this change (caveat 4)"))

    # ---- n450 switch state ----------------------------------------------
    n450_block = dict(
        enabled=N450,
        rule=("--n450 swaps in the matched-450 retrain for any cell where one "
              "exists. The retrain's run_id is the published cell's with its "
              "seed token replaced and a trailing -n450 suffix. For OGLANet a "
              "plain -n450 dump is refused: every dump-based OGLANet family is "
              "written at inference batch size 1, where DFFM's 20 unregistered "
              "batch-statistic layers behave as InstanceNorms, and only a "
              "retrain whose suffix also carries bs4 is comparable with the "
              "batch-4 published trees."),
        trees_searched=[t for t, r in N450_TREES if os.path.isdir(r)],
        trees_absent=[t for t, r in N450_TREES if not os.path.isdir(r)],
        n_cells_substituted=sum(1 for r in rows.values()
                                if r.get("n450_substituted")),
        n_cells_with_a_candidate=sum(1 for r in rows.values()
                                     if r.get("n450_candidates")),
        candidates={f"{m}/{a}/{c}": r["n450_candidates"]
                    for (m, a, c), r in sorted(rows.items())
                    if r.get("n450_candidates")},
        refused={f"{m}/{a}/{c}": r["n450_refused"]
                 for (m, a, c), r in sorted(rows.items())
                 if r.get("n450_refused")},
        retrain_status=("in progress on the SLURM box as of 2026-10-28: 2 of 27 "
                        "cells scored (oglanet-sibc4 phoenix, oglanet-segdesic "
                        "phoenix), 1 running, 19 pending, 5 with no live job "
                        "(3 MAMNet-ISW cells cancelled behind a timed-out smoke "
                        "and 2 OGLANet-SegDeSiC fans cancelled by their own "
                        "gate). No n450 dump has reached the analysis host."))
    print("\n--- --n450 switch ---")
    print(f"  enabled {N450}; candidates {n450_block['n_cells_with_a_candidate']}; "
          f"substituted {n450_block['n_cells_substituted']}; "
          f"trees absent {n450_block['trees_absent']}")

    # ---- fallbacks and exclusions ---------------------------------------
    fell_back = [dict(method=r["method"], arch=r["arch"], city=r["city"],
                      run_id=r["run_id"], tree=r["tree"],
                      unadapted_base_tree=r.get("unadapted_base_tree"),
                      reason="the declared PNG tree carries no dump for this "
                             "cell, so it is taken from the probability-map "
                             "fallback tree and recorded here")
                 for r in per_cell if r.get("fell_back")]
    excluded = [dict(method=r["method"], arch=r["arch"], city=r["city"],
                     run_id=r["run_id"], reason=r["blocker"])
                for r in per_cell if r.get("blocker")]
    print(f"\n  rows that fell back from the PNG tree: {len(fell_back)}")
    for f in fell_back:
        print(f"    {f['method']}/{f['arch']}/{f['city']} -> {f['tree']}")
    print(f"  rows with no value: {len(excluded)}")
    for e in excluded:
        print(f"    {e['method']}/{e['arch']}/{e['city']}: {e['reason']}")

    # ---- sanity flags ----------------------------------------------------
    sanity = []
    exact_t38 = [r for r in per_cell
                 if r.get("d_vs_table38") is not None
                 and abs(r["d_vs_table38"]) < 1e-9]
    sanity.append(
        "Every OGLANet row is one draw of a nondeterministic inference "
        "pipeline whose cause is now named: DFFM._apply_downsample_n_times "
        "rebuilds 20 freshly random, unregistered, train-mode Conv2d+BatchNorm2d "
        "layers inside forward() on every call (dffm.py:33-37 factory, 78-82 "
        "call site; 0 of 20 carry the same weights across two forwards, 20 of "
        "20 normalise with batch statistics, 0 of their parameters appear in "
        "state_dict). Erratum E-9 names the phenomenon and attributes it to "
        "cudnn; the attribution is wrong, and "
        "torch.use_deterministic_algorithms(True) neither raises nor fixes it.")
    sanity.append(
        "The OGLANet PNG masks and the OGLANet probability maps are different "
        "predictions, not two scorings of one prediction: 20 of 20 OGLANet "
        "cells disagree, 0.76 to 3.05 percent of pixels, up to 2.03 two-class "
        "mIoU, with the 10-pixel small-component filter explaining 0.1 percent "
        "of it and the disagreeing pixels sitting far from p = 0.5 (median "
        "|p - 0.5| 0.074 to 0.088). MAMNet and DINOv3 disagree by at most "
        "0.0147. In v2 the Vanilla and SAR rows came from the probability-map "
        "tree while every other row came from the PNG tree, which put the three "
        "OGLANet Vanilla rows and the three OGLANet SAR rows on a different "
        "population from their own upper bounds. In v3 every Vanilla row, and "
        "every SAR row's unadapted level, is PNG-scored, so no row here is "
        "paired across the two prediction populations. The measured PNG minus "
        "probability-map difference on the nine Vanilla cells is "
        + ", ".join(f"{r['run_id'].split('-')[0]}/{r['run_id'].split('-')[4]} "
                    f"{r['delta_png_minus_recompute_miou_k2']:+.4f}"
                    for r in png_vanilla_state)
        + ", so the three OGLANet Vanilla rows move by that much between v2 "
          "and v3 and the six MAMNet and DINOv3 rows do not.")
    sanity.append(
        f"The declared tree's own reproduction record: Table 38 to "
        f"{repro['max_abs_delta_vs_table38']:.4f} on "
        f"{repro['n_cells_with_table38']} cells and final_comparison.json "
        f"exactly (0.000000) on {repro['n_final_comparison_exact']} of "
        f"{repro['n_cells_with_final_comparison']}. An exact match on "
        f"{repro['n_final_comparison_exact']} cells "
        "is expected rather than suspicious -- it is the same metric code over "
        "the same files -- but it means the PNG tree and final_comparison.json "
        "are not independent evidence about each other.")
    if exact_t38:
        sanity.append(
            f"{len(exact_t38)} row(s) land exactly on their printed Table 38 "
            "value to 1e-9: " + ", ".join(f"{r['method']}/{r['arch']}/{r['city']}"
                                          for r in exact_t38)
            + ". Flagged per CONVENTIONS.md; the printed values are 2-dp so an "
              "exact tie is arithmetic coincidence, not a recovered number.")
    else:
        sanity.append(
            "No row lands exactly on a printed Table 38 value; the printed "
            "values are 2-dp and every declared-path value carries more digits.")
    sib9 = next(e for e in matched if e["method"] == "sib"
                and e["scope"] == "9_cell")
    sanity.append(
        f"SIB's worst cell on the 9-cell scope is {sib9['worst_cell']} at "
        f"{sib9['worst_case_gap']:+.4f}. The published claim's 7.18 is OGLANet/"
        "Miami (Table 38: 73.89 - 66.71), a cell that has no dump on this box. "
        "The G1 flip recorded in summaries/HLT_COUNTERSIGNS_armA_e12.md "
        "(MAMNet/Phoenix ~7.48 under the clean pool, computed as 73.75 - "
        "66.2665) reproduces here on the declared path, which is a check of "
        "this job rather than a new result.")
    sanity.append(
        "Two OGLANet n450 cells finished on Delta with foreground IoU 25.6396 "
        "(sibc4) and 25.6375 (segdesic) -- two different methods, 100 against "
        "37 epochs, 0.00205 apart. The Delta session flagged it as too clean "
        "and offered DFFM at batch 1 as an untested candidate explanation. No "
        "n450 dump is on this box and none enters any number here, but it is "
        "the reason the --n450 switch refuses a batch-1 OGLANet substitution.")
    above_upper = [r for r in per_cell
                   if r.get("gap_upper_minus_method") is not None
                   and r["gap_upper_minus_method"] < 0]
    if above_upper:
        sanity.append(
            "Cell(s) scoring ABOVE their own within-city upper bound: "
            + ", ".join(f"{r['method']}/{r['arch']}/{r['city']} "
                        f"{r['gap_upper_minus_method']:+.4f} (p {r['p']:.4f})"
                        for r in above_upper)
            + ". The published Table 38 already prints this pair the same way "
              "(SegDeSiC/DINOv3/Phoenix 68.04 against the DINOv3/Phoenix upper "
              "bound 67.82), so it is a property of the published grid rather "
              "than of this recomputation, and the paired bootstrap does not "
              "separate it from zero.")
    sar_v = [(r, next(x for x in per_cell if x["method"] == "vanilla"
                      and x["arch"] == r["arch"] and x["city"] == r["city"]))
             for r in per_cell if r["method"] == "sar"
             and r.get("gap_upper_minus_method") is not None]
    if sar_v:
        mx = max(abs(a["gap_upper_minus_method"] - b["gap_upper_minus_method"])
                 for a, b in sar_v)
        sanity.append(
            f"The SAR rows and the Vanilla rows differ by at most {mx:.4f} "
            "two-class mIoU. That is the size of SAR's measured adaptation "
            "effect on this metric, and it is far below the OGLANet "
            "nondeterminism band (E-9 cell-mean SD up to "
            f"{E9_MAX_SD5}, nine-pass span up to {E9_MAX_SPAN9}). SAR is "
            "therefore not separable from doing nothing on the three OGLANet "
            "cells, and its rows are reported as a separate access class "
            "rather than as a distinct level.")
    sanity.append(
        "The operator's brief states the 59 method cells reproduce printed "
        "Table 38 to a maximum |diff| of 0.0072. The maximum recorded in the "
        f"dumps themselves is {repro['max_abs_delta_vs_table38']:.4f} at "
        f"{repro['worst_cell_vs_table38']}, i.e. tighter than stated, and the "
        "Table 38 gate passes on 59 of 59. The premise checks out; the stated "
        "figure is not reproduced and the measured one is used.")
    cross_tree = [r for r in per_cell if r.get("paired_across_trees")]
    sanity.append(
        f"{len(cross_tree)} of {sum(1 for r in per_cell if r.get('p') is not None)} "
        "per-cell bootstraps pair a method cell against an upper-bound cell "
        "from a different dump tree. The pairing itself is sound -- the "
        "image_id sets are identical and in identical order within every city, "
        "asserted here, not assumed -- but a cross-tree delta carries both "
        "trees' provenance, which for OGLANet means two different predictions.")

    # ---- caveats ---------------------------------------------------------
    caveats = [
        # 1
        "Training budgets as measured (delta_runs/train_count_audit, and the "
        "2026-10-25/26 Delta audits behind it): MAMNet standard and its six "
        "adaptation methods (FDA, SegDeSiC, IIM, ISW, MRFP+, FADA) trained on "
        "900 images, 450 per source city; OGLANet SIB and OGLANet SegDeSiC on "
        "900; every other cross-city cell on 450; every within-city upper "
        "bound on 450; every cross-city cell validated on 300, not the 150 that "
        "section 3.3 and CONTRACT.md state. A matched-450 retrain of the 27 "
        "affected cells is in progress on the SLURM box and 2 of 27 have been "
        "scored; until it lands these rows are as published, so the MAMNet rows "
        "and the two OGLANet method rows named above are compared against "
        "450-image upper bounds at twice the training budget.",
        # 2
        "BSM and internal CrossEarth used a different 450-image draw from the "
        "published standard cells, with 150 validation images rather than 300. "
        "Measured on Delta: BSM draws its own 450 (225 per source city) through "
        "its wrapper; CrossEarth's 450 comes from the Final_data_loco split "
        "directory and overlaps the shared list on 222, 234 and 225 images of "
        "450. No BSM or CrossEarth cell is above 450.",
        # 3
        "OGLANet inference is nondeterministic and inference-batch-size "
        "dependent because of a defect in our own DFFM reimplementation: "
        "python/oglanet/models/dffm.py builds 20 freshly random-initialised, "
        "unregistered Conv2d+BatchNorm2d layers inside forward() on every call, "
        "so model.eval() never reaches them and they normalise with batch "
        "statistics (delta_runs/dffm_defect/, delta_runs/oglanet_nondeterminism/"
        "dffm_confirm.json). Measured spread on one OGLANet cell: foreground "
        "IoU at K=2 of 33.05 at batch 1, 37.33 at batch 2, 39.62 at batch 4, "
        "40.55 at batch 8. The published OGLANet tables, and therefore every "
        "OGLANet row on the declared path here, were produced at batch size 4. "
        "Every OGLANet value is one draw.",
        # 4
        "The SAR rows are an ASSUMPTION. Each is the published unadapted cell's "
        "level plus the change SAR measured inside its own pass, and that "
        "change was measured at inference batch size 1 while the published "
        "unadapted value is batch 4. For OGLANet that difference is not small: "
        "arm F's unadapted pass sits 3.02, 8.14 and 0.21 foreground-IoU points "
        "below the same cells' PNG-scored published values on chicago, miami "
        "and phoenix, against 0.0009 for MAMNet and 0.0057 for DINOv3. SAR is "
        "episodic and per-image, so its baseline is the batch-1 regime by "
        "construction and no harness change makes the OGLANet comparison sound. "
        "The within-pass change itself is measured on one forward path and is "
        "the only part of a SAR row that carries a paired test.",
        # 5
        "Six published cells were retrained after their March 2026 runs "
        "collapsed (delta_runs/retrained_cells_record/): OGLANet MRFP+ chicago "
        "hr and mr, OGLANet MRFP+ miami hr, DINOv3 MRFP+ chicago mr and miami "
        "hr, and DINOv3 vanilla miami hr. Table 38 uses the later runs, "
        "confirmed from the dates of the prediction trees it scores. Two of "
        "those cells' published numbers moved by +26.68 and +39.36 mIoU between "
        "the March and April artifacts. One cell, OGLANet MRFP+ miami, changed "
        "its learning rate from 1e-4 to 3e-5; the other five were retrained at "
        "the same learning rate and simply did not collapse again. Who made "
        "that change is unrecorded: no ledger line, no commit and no DECISIONS "
        "entry covers 2026-04, and the only actor evidence is the SLURM job "
        "owner.",
        # 6
        "Seeds: every published cell on this grid is a single draw. MAMNet-SIB "
        "is the exception and carries three clean-pool seeds {42, 43, 44}; its "
        "row is the seed mean and the per-seed values are in the side table. "
        "The published MAMNet-SIB cell is contaminated (E-12) and is never used "
        "as a value, only reported beside the retrains.",
        # the rest are this job's own limitations
        "SIB is a 7-of-9 row on the three original architectures. "
        "OGLANet-SIB miami and OGLANet-SIB phoenix have no dump in the declared "
        "PNG tree or in the recompute fallback: the 56-cell re-score of "
        "2026-10-25 deliberately held those two cells pending the OGLANet SIB "
        "AMP question, and the PNG scoring pass covered only the cells that "
        "re-score produced. Their Table 38 printed values are reported as an "
        "explicit bound, never as a row value.",
        "The three OGLANet SIB cells' published Table 38 rows are not "
        "reproducible from the stored probability maps at any scoring setting: "
        "Table 38's SIB row was written under fp16 autocast by "
        "train_oglanet_sib.py while the stored maps were written in fp32 by "
        "sp_gap_analysis.py. The PNG tree sidesteps this -- the one available "
        "OGLANet SIB cell reproduces Table 38 from its PNG masks -- but it is "
        "why the other two cells were held.",
        "RESOLVED in v3. v2's tree-mixing caveat read: Vanilla rows and SAR "
        "rows come from the recompute fallback tree (stored probability maps), "
        "every other row from the declared PNG tree; the PNG tree carries no "
        "vanilla cell at all; for OGLANet the two trees are different "
        "predictions and OGLANet phoenix vanilla had no measured PNG score "
        "anywhere on this box. It is resolved by the nine PNG-scored "
        "cross-city Vanilla dumps at "
        "dumps/st_r450_published_png/{mamnet,oglanet,dinov3}-vanilla-loco-r450-"
        "{chicago,miami,phoenix}-hr-s0, written 2026-10-28 by the same scoring "
        "pass that produced the rest of the tree, each carrying .DONE and each "
        "reproducing final_comparison.json to 0.000000. Every Vanilla row here, "
        "and every SAR row's unadapted level, is now PNG-scored, so no row is "
        "paired against an upper bound from a different prediction population "
        "and no row in this JSON falls back to the probability-map tree. The "
        "nine cells move by " + ", ".join(
            f"{r['run_id'].split('-')[0]}/{r['run_id'].split('-')[4]} "
            f"{r['delta_png_minus_recompute_miou_k2']:+.4f}"
            for r in png_vanilla_state)
        + " two-class mIoU against their v2 probability-map values. The SAR "
          "rows remain an ASSUMPTION for the separate reason in caveat 4: "
          "their within-pass change is measured at inference batch size 1 "
          "while the PNG-scored unadapted level is batch 4.",
        "BSM is an 8-of-9 row on the three original architectures and 11 of 12 "
        "with CrossEarth. DINOv3/chicago collapsed in training at best "
        "validation mIoU 5.51 against the codebase floor of 10.0 and its dump "
        "directory is empty. It is never imputed.",
        "CrossEarth exists for Upper, Vanilla, FDA, SIB and BSM only. SegDeSiC, "
        "IIM, ISW, MRFP+, FADA and SAR have no CrossEarth counterpart, so on "
        "the 12-cell scope those six methods are still 9-cell rows and their "
        "n_cells_expected is recorded as 9. CrossEarth is measured against its "
        "own within-city upper bound in every case.",
        "Access classes follow erratum I-1's countersigned taxonomy as the "
        "operator's brief states it, which OVERRIDES each dump's own "
        "target_access field. Two systematic disagreements are recorded per "
        "row: every PNG-tree dump records source_only because the PNG pass is a "
        "scoring pass over stored artifacts, and arm F's CrossEarth "
        "within-city upper cells record source_only where arm C and arm D "
        "uppers record target_labels. "
        "summaries/target_access_audit.md v3 also lists SIB as "
        "unlabeled_target on OGLANet; I-1 is later and makes OGLANet-SIB "
        "source_only, which is what is used here.",
        "Comparing SIB against a baseline crosses access classes on two of "
        "three architectures: MAMNet-SIB is unlabeled_target while OGLANet-SIB "
        "and DINOv3-SIB are source_only, and the strongest baseline may sit in "
        "either class. The within-access-class rankings are reported so a "
        "same-class comparison is available; no pooled number here is corrected "
        "for the class mix, and the 2026-08-03 pooling ruling applies to the "
        "external criterion, not internally.",
        "CONTRACT.md rule zero puts the one metric implementation in "
        "shadowtransfer/metrics.py on the SLURM box. No copy of that file is "
        "reachable from the analysis host, so the metric is recomputed from the raw "
        "tp/fp/fn/tn counts and asserted on every cell read against the stored "
        "iou_fg_tol_pct / iou_bg_tol_pct / miou_tol_pct columns (max |delta| "
        "under 1e-9 everywhere), against the E-12 sidecars' own "
        "two_class_tol_k2_pct, and against each arm F SAR run's own "
        "delta_adaptation_only_miou_tol2 (max |delta| under 1e-6).",
        "The declared PNG tree arrived from Delta without an INDEX.parquet and "
        "the brief forbids writing inside dumps/, so the index was rebuilt here "
        "and written to analysis/_inv/INDEX_st_r450_published_png_v3.parquet. "
        "Its sha256 is the index_sha256 of this JSON; the other three tree "
        "indexes are carried in index_sha256_all. The _v3 suffix is "
        "deliberate: results/12 cites analysis/_inv/INDEX_"
        "st_r450_published_png.parquet by sha256 and that file describes a "
        "76-cell tree, so it is left on disk unchanged rather than rewritten "
        "to describe the 85-cell tree v3 reads.",
        "High resolution only, rung 450 only, two-class average "
        "boundary-tolerant mIoU at K=2 only. The nine mid-resolution standard "
        "cells in the PNG tree are scored but are not used here and have no "
        "LOCO counterpart, so Table 2's pooled upper bounds still cannot be "
        "recomputed and erratum E-2's Table 2 side is still unchecked.",
        "No value anywhere in this JSON is rounded to match a printed value. "
        "Table 38 appears only inside agreement checks and inside the explicitly "
        "labelled missing-cell bound.",
    ]

    headline = (
        f"On the declared path of two-class boundary-tolerant mIoU at K=2, high "
        f"resolution, rung 450, scored from the published PNG masks, SIB's "
        f"worst-case gap to the within-city upper bound is "
        f"{sib9['worst_case_gap']:.2f} mIoU at {sib9['worst_cell']} over "
        f"{sib9['n_cells']} of 9 cells, its mean gap is "
        f"{margins['9_cell']['sib_mean_gap']:.2f} mIoU, its mean margin over "
        f"{margins['9_cell']['strongest_baseline_label']} is "
        f"{margins['9_cell']['margin_mean_gap_difference']:+.2f} mIoU, and "
        f"reading {reading['scope_9_cell']} fires on the 9-cell scope and "
        f"{reading['scope_12_cell']} on the 12-cell scope.")

    # ---- assemble --------------------------------------------------------
    def strip(r):
        return {k: v for k, v in r.items()
                if k not in ("per_image", "image_id", "sar_delta_per_image")}

    out = dict(
        id="22_internal_sib_worst_case_v3",
        arm="F",
        answers=["BxRx-W3", "AC3", "v1UJ-W3"],
        question=("Does SIB hold the smallest largest-single-cell gap to the "
                  "within-city upper bound of any evaluated method on the "
                  "internal grid, and what is its mean-mIoU margin over the "
                  "strongest baseline, on one declared computation path?"),
        status="ADJUDICATED",
        supersedes=["results/12_internal_sib_worst_case_v2.json"],
        supersedes_why=("v2 fell back to the probability-map tree for all nine "
                        "cross-city Vanilla cells and for the nine SAR rows "
                        "built on them, because the declared PNG tree carried "
                        "no Vanilla cell. Those nine PNG-scored Vanilla cells "
                        "now exist and reproduce final_comparison.json to "
                        "0.000000, so v3 recomputes every number on one tree. "
                        "v2 in turn superseded results/11_internal_sib_worst_"
                        "case.json (GATE_FAILED; its blockers B1, B2 and B3 "
                        "were cleared by the PNG tree and B4 was superseded by "
                        "the DFFM finding)."),
        png_vanilla_entry_condition=png_vanilla_state,
        declared_path=dict(
            metric="two_class_mIoU_boundary_tolerant_K2_pct",
            definition=("per image, the mean of foreground IoU and background "
                        "IoU computed from raw tp/fp/fn/tn on the valid region "
                        "with a +-2 px don't-care band, then macro-averaged "
                        "over the cell's 150 test images"),
            degenerate_rule=("App E: an image with tp+fp+fn == 0 at K=2 scores "
                             "foreground 0 and background 100, so its "
                             "two-class value is exactly 50"),
            gap_sign=dict(
                gap="method minus its within-city upper bound (CONVENTIONS.md "
                    "sign; negative when the method is below the upper bound)",
                gap_upper_minus_method="upper bound minus method (the brief's "
                                       "sign; positive is a shortfall). "
                                       "\"Worst case\" always means the largest "
                                       "gap_upper_minus_method over a scope."),
            table_family="Table 38 / revised Table 42",
            resolution="hr", rung=450,
            source=("dumps/st_r450_published_png (the saved prediction PNG "
                    "masks, the tree Table 38 and final_comparison.json were "
                    f"scored from), {png_info['n_rows']} cells"),
            fallback=("dumps/st_r450_published_recompute (published cells "
                      "re-passed over their own stored float16 probability "
                      "maps), used only where the PNG tree carries no dump and "
                      f"recorded per row; {len(fell_back)} rows fall back in "
                      "v3"),
            other_trees=dict(
                bsm="dumps/st (arm F BSM training runs)",
                crossearth="dumps/st (arm F / arm C-D CrossEarth runs)",
                mamnet_sib="delta_dumps/dumps/e12 (E-12 clean-pool retrains)",
                sar="runs/armF_sar_*/perstep.parquet"),
            reproduction=repro,
            validated_against=[
                "stored per-image iou_fg_tol_pct / iou_bg_tol_pct / "
                "miou_tol_pct columns, max |delta| < 1e-9 on every cell read",
                "the PNG dumps' own recorded Table 38 and "
                "final_comparison.json deltas",
                "the E-12 sidecars' two_class_tol_k2_pct",
                "each arm F SAR run's own delta_adaptation_only_miou_tol2, "
                "max |delta| < 1e-6"],
            image_id_identity=id_check),
        scopes=dict(
            scope_9_cell="the three original architectures (MAMNet, OGLANet, "
                         "DINOv3) x three held-out cities",
            scope_12_cell="the same plus CrossEarth, for the five methods "
                          "CrossEarth was run with (Upper, Vanilla, FDA, SIB, "
                          "BSM); the other six methods have no CrossEarth "
                          "counterpart and stay 9-cell rows"),
        precommitted_readings=PRECOMMITTED,
        reading_fired=reading,
        per_method=per_method,
        matched_cell_comparison=matched,
        per_cell=[strip(r) for r in per_cell],
        vanilla_worst_case=vanilla_worst,
        sib_margin=margins,
        within_access_class_rankings=dict(
            whole_method=rankings,
            whole_method_note="methods whose cells all share one access class. "
                              "SIB is absent from this view because its cells "
                              "do not: MAMNet-SIB is unlabeled_target through "
                              "FDA while OGLANet-SIB and DINOv3-SIB are "
                              "source_only.",
            by_cell_access_class=rankings_by_cell,
            by_cell_access_class_note="every method restricted to the cells "
                                      "whose own access class is the one named, "
                                      "which is the only view a same-class "
                                      "comparison can place SIB in. Cell counts "
                                      "differ within a ranking, so these are "
                                      "comparisons of different cell sets and "
                                      "the matched-cell comparison is the one "
                                      "the reading rests on."),
        mamnet_sib_seed_table=[
            dict(city=c,
                 per_seed=rows[("sib", "mamnet", c)]["per_seed"],
                 seed_mean=rows[("sib", "mamnet", c)]["value"],
                 seed_sd=rows[("sib", "mamnet", c)]["seed_sd"],
                 published_contaminated=rows[("sib", "mamnet", c)][
                     "published_contaminated_value"],
                 published_contaminated_repro_delta=rows[
                     ("sib", "mamnet", c)]["published_contaminated_repro_delta"])
            for c in CITIES],
        conditions=conditions,
        tests=tests,
        n450_switch=n450_block,
        rows_fell_back_from_png_tree=fell_back,
        rows_without_a_value=excluded,
        runs=sorted({r["run_id"] for r in rows.values()
                     if isinstance(r.get("run_id"), str)}
                    | {x for r in rows.values()
                       if isinstance(r.get("run_id"), list)
                       for x in r["run_id"]}),
        runs_excluded=[dict(run_id=e["run_id"], reason=e["reason"])
                       for e in excluded]
        + [dict(run_id=f"{a}-{t}-r450-{c}-mr-s0",
                reason="mid-resolution; the declared path is high resolution "
                       "only")
           for a in ARCHS for t in ("none-upper",) for c in CITIES],
        index_sha256=png_info["sha256"],
        index_sha256_all={k: dict(path=v.get("path"), tree=v.get("root"),
                                  sha256=v["sha256"], n_rows=v["n_rows"],
                                  built_here=v.get("built_here"),
                                  n_unindexable=v.get("n_unindexable"),
                                  unindexable=v.get("unindexable"))
                          for k, v in indexes.items()},
        index_on_disk_state=on_disk,
        source_artifacts=[
            dict(path="delta_runs/dffm_defect/provenance.json",
                 role="the DFFM defect and the inference batch size of every "
                      "published OGLANet artifact family (caveats 3 and 4)"),
            dict(path="delta_runs/retrained_cells_record/results.json",
                 role="the six cells retrained after 2026-03-19 (caveat 5)"),
            dict(path="delta_runs/train_count_audit_projects/"
                      "retrain_at_450_list.json",
                 role="the measured training counts and the 27-cell matched-450 "
                      "retrain list (caveat 1)"),
            dict(path="delta_runs/oglanet_saved_prediction_source/"
                      "vanilla_reference_scores.json",
                 role="PNG-scored OGLANet vanilla values for chicago and "
                      "miami, used only to quantify the fallback mixing"),
            dict(path="summaries/HLT_COUNTERSIGNS_armA_e12.md",
                 role="the countersigned E-12 seed table and the G1 flip"),
            dict(path="final_paper.md",
                 role="Table 38 printed values, used only in agreement checks "
                      "and in the labelled missing-cell bound"),
        ],
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/22_internal_sib_worst_case_v3.py",
        env=dict(python=sys.version.split()[0],
                 conda_env="myenv "
                           "(conda env)",
                 pythonnousersite=os.environ.get("PYTHONNOUSERSITE"),
                 numpy=np.__version__, pandas=pd.__version__),
        seed=BOOT_SEED,
        seed_note=f"the only stochastic step is the bootstrap, B = {B}, "
                  f"numpy default_rng seeded at {BOOT_SEED}",
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; "
                        "only final_paper/ is git-connected and this job does "
                        "not touch it",
        compute_note="read-only CPU analysis on the analysis host. No training, no GPU "
                     "inference, no checkpoint loaded, nothing under dumps/ or "
                     "results/ modified.",
        headline=headline,
        sanity_flags=sanity,
        caveats=caveats,
        figure=None,
        caption=None,
    )

    print("\nheadline: " + headline)

    if DRY_RUN:
        print("\nDRY_RUN: nothing written.")
        return

    outpath = os.path.join(BASE, "results",
                           "22_internal_sib_worst_case_v3.json")
    if os.path.exists(outpath):
        fail(f"{outpath} already exists; this job writes exactly one new file")

    def enc(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")

    def clean(o):
        """NaN is pandas' way of saying 'no value'; JSON says null. A
        non-finite value that is NOT a missing value still raises."""
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.floating):
            return clean(float(o))
        if isinstance(o, float) and not np.isfinite(o):
            if np.isnan(o):
                return None
            fail(f"non-finite, non-missing value in the output: {o}")
        return o

    tmp = outpath + ".new"
    with open(tmp, "w") as fh:
        json.dump(clean(out), fh, indent=1, default=enc, allow_nan=False)
    os.replace(tmp, outpath)
    print(f"\nwrote {outpath}")
    print(f"  sha256 {sha256_file(outpath)}")


if __name__ == "__main__":
    main()
