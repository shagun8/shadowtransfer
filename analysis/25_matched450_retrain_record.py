#!/usr/bin/env python
"""
Matched-450 retrain record -- the 27 published cells that trained on 900 images,
retrained at 450, against their published counterparts.

Answers AC1. The budget asymmetry it measures is results/21_disclosure_record.json
finding 1: 27 published cells trained on 900 source images (450 per source city)
while every within-city upper bound trained on 450, so those 27 method and
standard cells were compared against upper bounds at half their training budget.
results/22_internal_sib_worst_case_v3.json carries that asymmetry as its caveat 1
and names this retrain as the thing that would discharge it. The 27 matched-450
retrains have now landed; this job is their record.

DECLARED PATH
  metric      two-class average boundary-tolerant mIoU at K=2, in percent (the
              Table 38 / revised Table 42 family), computed per image from raw
              tp/fp/fn/tn on the valid region and macro-averaged over the cell's
              150 test images. Foreground-only IoU at K=2 is reported beside it
              on every row, as CONTRACT.md section 3's iou_tol.
  degenerate  App E rule: an image with tp+fp+fn == 0 at K=2 scores foreground
              IoU 0 and background IoU 100, so its two-class value is exactly 50
  resolution  high resolution only
  rung        450 train / 150 test, both sides
  published   dumps/st_r450_published_png/ -- the saved prediction PNG masks,
              the tree Table 38 and final_comparison.json were scored from.
  n450        dumps/st_r450_published_recompute/*-s42-n450[-bs4] -- the matched-450
              retrains, scored in-job by n450_score.py from each cell's own
              probability maps. One scorer sha on all 27.

SIGNS. CONVENTIONS.md fixes `gap` = LOCO minus the matched within-city upper
bound, so the paper's gap is negative. Both signs are emitted on every row, as in
analysis/22:
  gap                     = method - upper   (CONVENTIONS.md sign, negative)
  gap_upper_minus_method  = upper - method   (the brief's sign, positive)
The retrain contrast is always  delta = n450 - published, so a negative delta
means the matched-450 retrain scores below the published 900-image cell.

Read-only on every dump tree. No training, no GPU inference, no checkpoint
loaded. Nothing under dumps/ or under an existing results/ file is modified.
Writes results/25_matched450_retrain_record.json and three indexes under
analysis/_inv/ with a _25 suffix (CLAUDE.md requires selection from an index; the
brief forbids writing inside dumps/, and the _v3 indexes results/22 cites by
sha256 must not be rewritten).

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/25_matched450_retrain_record.py [--dry-run]
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
B = 10000                            # paired bootstrap replicates, everywhere
BOOT_SEED = 0

PNG = os.path.join(BASE, "dumps", "st_r450_published_png")
RECOMPUTE = os.path.join(BASE, "dumps", "st_r450_published_recompute")
INV = os.path.join(BASE, "analysis", "_inv")
OUT = os.path.join(BASE, "results", "25_matched450_retrain_record.json")

ARCHS = ("mamnet", "oglanet", "dinov3")
CITIES = ("chicago", "miami", "phoenix")

# The 27 matched-450 cells, as results/21 finding 1 lists them. Written out
# rather than globbed: CLAUDE.md forbids selecting cells from a glob, and the
# retrain tree also holds a `.failed_23272684` directory and a `-bs4` re-score
# that a glob would pick up.
N450_GRID = ([("mamnet", m, c)
              for m in ("vanilla", "fda", "segdesic", "iim", "isw",
                        "mrfp_plus", "fada") for c in CITIES]
             + [("oglanet", "segdesic", c) for c in CITIES]
             + [("oglanet", "sibc4", c) for c in CITIES])

# The inference batch size each n450 cell's published comparator was produced at,
# and therefore the batch size its own scored pass must match. OGLANet scores are
# batch-size dependent (results/21 finding 3: DFFM rebuilds 20 unregistered
# batch-statistic BatchNorms inside forward()), so this is a comparability
# requirement for OGLANet and a record for MAMNet.
REQUIRED_INFER_BS = {
    ("mamnet", "vanilla"): 4, ("mamnet", "fda"): 4, ("mamnet", "segdesic"): 4,
    ("mamnet", "iim"): 4, ("mamnet", "isw"): 4, ("mamnet", "mrfp_plus"): 4,
    ("mamnet", "fada"): 4,
    # the published OGLANet classic tables are batch-4 fp32
    # (run_inference.sh:73, run_inference_probs.sh:77)
    ("oglanet", "segdesic"): 4,
    # the published Table 38 OGLANet-SIB row came from train_oglanet_sib.py's own
    # test pass at args.batch_size 8 under fp16 autocast, NOT from
    # run_inference_probs.py at batch 4, so batch 8 is the matched comparator
    # here. run_inference_probs.py cannot load a SIB checkpoint at all
    # (prob_variant is None), so no batch-4 OGLANet-SIB pass exists.
    ("oglanet", "sibc4"): 8,
}

# Cells whose `inference_batch_size` field is null in run.json, with the session
# log line that supplies the value instead. Both are single-cell or interactive
# resubmissions that predate the 1c change that added the field.
BS_FROM_LOG = {
    "mamnet-vanilla-loco-r450-phoenix-hr-s42-n450": dict(
        inference_batch_size=4,
        source=("delta_logs/session_logs/2026-10-28-n450-bs4-isw.md, STEP 5 "
                "table: \"| mamnet-vanilla-loco-r450-phoenix | **30.8823** | "
                "28.4405 | **+2.4418** | 25 | 15 | early_stopping | **4** "
                "(`run_inference_probs.py`) |\""),
        corroboration=("the same log's STEP 1 1a finding, \"n450_train.py was "
                       "already scoring the classic family from a batch-4 pass. "
                       "It calls run_inference_probs.py with no --batch_size, "
                       "and that default is 4\"")),
    "oglanet-segdesic-loco-r450-phoenix-hr-s42-n450": dict(
        inference_batch_size=4,
        source=("delta_logs/session_logs/2026-10-28-n450-bs4-isw.md, STEP 1 1a "
                "table: \"oglanet-segdesic-loco-r450-phoenix-hr-s42-n450 (g3) | "
                "**25.637529** | batch **4**, fp32, `run_inference_probs.py`\""),
        corroboration=("not used for a value: this cell is superseded by its "
                       "-bs4 re-score, which records inference_batch_size 4 "
                       "directly")),
}

# Printed Table 38, used ONLY in agreement checks and in the two explicit
# missing-cell bounds, exactly as results/22's missing_cell_bound does.
TABLE38 = {
    ("upper", "mamnet"): (86.31, 79.11, 73.75),
    ("upper", "oglanet"): (79.12, 73.89, 66.15),
    ("upper", "dinov3"): (80.27, 76.00, 67.82),
    ("vanilla", "mamnet"): (79.72, 74.64, 62.84),
    ("fda", "mamnet"): (81.88, 74.21, 67.77),
    ("segdesic", "mamnet"): (82.44, 73.14, 58.80),
    ("segdesic", "oglanet"): (66.47, 59.90, 61.17),
    ("iim", "mamnet"): (81.06, 70.05, 55.43),
    ("isw", "mamnet"): (75.23, 67.74, 51.08),
    ("mrfp_plus", "mamnet"): (75.76, 72.80, 53.35),
    ("fada", "mamnet"): (79.59, 60.90, 62.30),
    ("sibc4", "oglanet"): (73.05, 66.71, 63.97),
}

METHOD_LABEL = {"vanilla": "Vanilla", "fda": "FDA", "segdesic": "SegDeSiC",
                "iim": "IIM", "isw": "ISW", "mrfp_plus": "MRFP+",
                "fada": "FADA", "sibc4": "SIB", "none": "Upper bound"}

# results/01_scaling_ladder.json's collapse rule, applied exactly.
COLLAPSE_RULE = dict(
    definition=("n_pred_pos / n_valid_px, pooled over the cell's images, "
                "greater than the threshold"),
    threshold=0.5,
    source="results/01_scaling_ladder.json -> collapse_rule",
    threshold_provenance=(
        "Carried verbatim from results/01_scaling_ladder.json: \"Not "
        "operator-authorised; derived from the ground truth. GT shadow occupies "
        "at most 0.086 of the valid region on any cell ... so any cell at "
        "pred-positive >= 0.50 has fg IoU <= 0.086 / 0.50 = 17.2 by "
        "construction, meaning the rule cannot catch a working model. ... any "
        "threshold in (0.15, 0.65) gives the same exclusion set.\""),
    known_collapsed_in_results_01=[
        "dinov3-none-upper-r225-phoenix-hr-s0",
        "dinov3-vanilla-loco-r225-miami-hr-s0",
        "dinov3-none-upper-r225-miami-hr-s0",
        "oglanet-none-upper-r225-phoenix-hr-s2"],
)

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
                "miou_tol_pct", "n_pred_pos")


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
    """Per-image two-class boundary-tolerant mIoU at K=2, percent, and its two
    halves.

    CONTRACT.md rule zero puts the canonical implementation in
    shadowtransfer/metrics.py on the SLURM box. No copy of that file is reachable
    from the analysis host, so the metric is recomputed here from the raw tp/fp/fn/tn counts
    that CONTRACT.md section 3 calls the source of truth, then asserted against
    the stored iou_fg_tol_pct / iou_bg_tol_pct / miou_tol_pct columns on every
    cell read. Identical to analysis/22_internal_sib_worst_case_v3.py's
    implementation, deliberately: the two jobs must not differ on the metric.
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


def read_cell(root, run_id):
    """CONTRACT.md section 3 + 7 validation on read. Fails loudly, never coerces."""
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
    for col in ("tp_tol", "fp_tol", "fn_tol", "tn_tol", "n_pred_pos",
                "n_valid_px"):
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


def pooled_pred_pos_fraction(df):
    """results/01's collapse statistic: n_pred_pos / n_valid_px, POOLED over the
    cell's images (summed numerator over summed denominator), not the mean of the
    per-image ratios."""
    num = float(df.n_pred_pos.to_numpy(dtype=np.float64).sum())
    den = float(df.n_valid_px.to_numpy(dtype=np.float64).sum())
    if den <= 0:
        fail("n_valid_px sums to zero on a cell")
    return num / den


# ----------------------------------------------------------------------------
# index
# ----------------------------------------------------------------------------

SCALAR = (str, int, float, bool, type(None))


def build_index(label, root, write_to=None):
    """CONTRACT.md section 5: one row per run_id, every scalar run.json field
    plus n_images, mean_iou_tol, parquet_mtime, dump_sha256. Written outside the
    dump tree, with a _25 suffix so neither the trees' own INDEX.parquet nor the
    _v3 indexes results/22 cites by sha256 are touched."""
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
                failed_marker=os.path.isfile(os.path.join(d, ".FAILED")),
                done_marker=os.path.isfile(os.path.join(d, ".DONE"))))
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
    if write_to and not DRY_RUN:
        os.makedirs(os.path.dirname(write_to), exist_ok=True)
        tmp = write_to + ".new"
        idx.to_parquet(tmp, index=False)
        os.replace(tmp, write_to)
        info.update(path=os.path.relpath(write_to, BASE),
                    sha256=sha256_file(write_to))
    return idx, info


def on_disk_index_state(label, root):
    p = os.path.join(root, "INDEX.parquet")
    if not os.path.isfile(p):
        return dict(label=label, path=os.path.relpath(p, BASE), present=False,
                    note="no INDEX.parquet in this tree")
    idx = pd.read_parquet(p)
    mt = os.path.getmtime(p)
    names = [n for n in sorted(os.listdir(root))
             if os.path.isdir(os.path.join(root, n))
             and os.path.isfile(os.path.join(root, n, "perimage.parquet"))]
    newer = [n for n in names
             if os.path.getmtime(os.path.join(root, n,
                                              "perimage.parquet")) > mt]
    absent = [n for n in names if n not in set(idx.run_id.astype(str))]
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
    """Paired image-level bootstrap on per-image deltas, two-tailed, p floored at
    1/B (the arm F convention: p at 1e-04 reads as p <= 1/B)."""
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    reps = delta[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    p = 2.0 * min(float((reps <= 0).mean()), float((reps >= 0).mean()))
    return dict(mean=float(delta.mean()),
                stderr=float(delta.std(ddof=1) / np.sqrt(n)),
                n=int(n), ci95=[lo, hi], ci_method=f"paired_bootstrap_B{b}",
                p=float(min(1.0, max(1.0 / b, p))), B=b, two_tailed=True,
                bootstrap_sd=float(reps.std(ddof=1)))


def cluster_bootstrap_mean_over_cells(per_cell_deltas, b=B, seed=BOOT_SEED):
    """Mean over cells of a per-image paired delta, image-cluster bootstrap.

    per_cell_deltas maps the held-out city (whose 150 test images are shared by
    every cell of that city in identical order) to a list of per-image delta
    arrays. Images are resampled once per city and the same index vector is
    applied to every cell of that city, matching Appendix G.9's image-cluster
    bootstrap rather than treating two cells of one city as independent.
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
    """Holm-Bonferroni, adjusted p-values in input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


# ----------------------------------------------------------------------------
# cell resolution
# ----------------------------------------------------------------------------

def published_run_id(arch, method, city):
    if method == "none":
        return f"{arch}-none-upper-r450-{city}-hr-s0"
    return f"{arch}-{method}-loco-r450-{city}-hr-s0"


def resolve_n450_dir(arch, method, city):
    """Pick the n450 dump directory for one cell, and say why.

    Rule, stated rather than inferred:
      * the candidate set is every directory matching
        `{arch}-{method}-loco-r450-{city}-hr-s<seed>-n450<suffix>` that carries a
        perimage.parquet AND a .DONE marker;
      * a `-bs4` re-score is preferred over its parent, because it is a fresh
        pass at the batch size the published OGLANet tables were produced at and
        it is what the operator's brief names;
      * the chosen directory's recorded inference batch size must equal the batch
        size its published comparator was produced at (REQUIRED_INFER_BS), read
        from run.json's `inference_batch_size` or, where that field is null, from
        the session log line in BS_FROM_LOG. A mismatch is a hard stop, not a
        silent substitution.
    """
    stem = f"{arch}-{method}-loco-r450-{city}-hr"
    pat = re.compile(re.escape(stem) + r"-s(?P<seed>\d+)-n450(?P<suffix>[-\w]*)$")
    cands = []
    for name in sorted(os.listdir(RECOMPUTE)):
        m = pat.match(name)
        if not m:
            continue
        d = os.path.join(RECOMPUTE, name)
        rec = dict(run_id=name, seed=int(m.group("seed")),
                   suffix=m.group("suffix") or "",
                   has_perimage=os.path.isfile(os.path.join(d,
                                                            "perimage.parquet")),
                   done_marker=os.path.isfile(os.path.join(d, ".DONE")),
                   failed_marker=os.path.isfile(os.path.join(d, ".FAILED")))
        cands.append(rec)
    usable = [c for c in cands if c["has_perimage"] and c["done_marker"]
              and not c["failed_marker"]]
    if not usable:
        fail(f"{stem}: no .DONE n450 dump with a perimage.parquet; candidates "
             f"{cands}")
    bs4 = [c for c in usable if "bs4" in c["suffix"]]
    chosen = (bs4 or usable)[0]
    if len(bs4 or usable) > 1:
        fail(f"{stem}: {len(bs4 or usable)} equally preferred n450 dumps "
             f"{[c['run_id'] for c in (bs4 or usable)]}; the choice is not "
             "determined")
    return chosen, cands


TRAINING_FIELDS = ("seed", "n_train", "n_train_per_source_city", "n_val",
                   "n_test", "epochs_max", "epochs_run", "best_epoch",
                   "best_val_metric", "stopped_on", "early_stop_patience",
                   "target_access", "config_sha256", "train_list_sha256",
                   "val_list_sha256")


def training_meta(run_id, meta):
    """The cell's training record.

    A `-bs4` directory is a re-inference of its parent's checkpoint and its
    run.json carries only the inference and scoring record, so the training
    fields are read from the parent named by `parent_run_id` and the borrowing is
    recorded. Nothing is invented: a field absent on both is reported absent.
    """
    out = {k: meta.get(k) for k in TRAINING_FIELDS}
    src = {k: "own run.json" for k in TRAINING_FIELDS if meta.get(k) is not None}
    parent = meta.get("parent_run_id")
    need = [k for k in TRAINING_FIELDS if out[k] is None]
    if need and parent:
        pj = os.path.join(RECOMPUTE, parent, "run.json")
        if not os.path.isfile(pj):
            fail(f"{run_id}: parent_run_id {parent} has no run.json, so its "
                 f"training record cannot be read for {need}")
        pm = json.load(open(pj))
        for k in need:
            if pm.get(k) is not None:
                out[k] = pm[k]
                src[k] = f"parent_run_id {parent}"
    out["training_metadata_sources"] = src
    out["training_metadata_parent_run_id"] = parent
    return out


def infer_bs_of(run_id, meta):
    """The cell's scored-pass inference batch size, with its provenance."""
    v = meta.get("inference_batch_size")
    if v is not None:
        return dict(inference_batch_size=int(v),
                    provenance="run.json field `inference_batch_size`",
                    run_json_source=meta.get("inference_batch_size_source"),
                    inference_precision=meta.get("inference_precision"),
                    from_session_log=False)
    if run_id in BS_FROM_LOG:
        rec = dict(BS_FROM_LOG[run_id])
        rec.update(provenance=("run.json field `inference_batch_size` is NULL; "
                               "value read from the session log"),
                   from_session_log=True,
                   inference_precision=meta.get("inference_precision"))
        return rec
    fail(f"{run_id}: inference_batch_size is null in run.json and no session "
         "log line is registered for it in BS_FROM_LOG")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    hr = "=" * 100
    print(hr)
    print("Matched-450 retrain record -- 27 cells, published vs n450")
    print(f"DRY_RUN = {DRY_RUN}   B = {B}   seed = {BOOT_SEED}")
    print(hr)

    # ---- indexes ---------------------------------------------------------
    indexes = {}
    for label, root in (("st_r450_published_png", PNG),
                        ("st_r450_published_recompute", RECOMPUTE)):
        _, info = build_index(label, root,
                              write_to=os.path.join(INV,
                                                    f"INDEX_{label}_25.parquet"))
        indexes[label] = info
    on_disk = {label: on_disk_index_state(label, root)
               for label, root in (("st_r450_published_png", PNG),
                                   ("st_r450_published_recompute", RECOMPUTE))}
    print("\n--- indexes (CLAUDE.md: start from the index, never from a glob) ---")
    for lbl, i in indexes.items():
        o = on_disk[lbl]
        print(f"  {lbl:32s} built rows {i['n_rows']:4d}  "
              f"unindexable {i['n_unindexable']}  on-disk index "
              + ("absent" if not o.get("present")
                 else ("STALE" if o.get("stale") else "current")))
        for u in i["unindexable"]:
            print(f"      unindexable: {u['run_id']} "
                  f"(parquet {u['has_parquet']}, run.json {u['has_run_json']}, "
                  f".FAILED {u['failed_marker']})")

    # ---- the 27 n450 directories and their inference batch size ---------
    print("\n--- the 27 n450 dump directories, in selection order ---")
    selected, bs_table, candidate_log = {}, [], {}
    for arch, method, city in N450_GRID:
        chosen, cands = resolve_n450_dir(arch, method, city)
        selected[(arch, method, city)] = chosen["run_id"]
        candidate_log[f"{arch}/{method}/{city}"] = cands
        df, meta, why = read_cell(RECOMPUTE, chosen["run_id"])
        if df is None:
            fail(f"{chosen['run_id']}: {why}")
        bs = infer_bs_of(chosen["run_id"], meta)
        need = REQUIRED_INFER_BS[(arch, method)]
        if bs["inference_batch_size"] != need:
            fail(f"{chosen['run_id']}: scored at inference batch size "
                 f"{bs['inference_batch_size']} but its published comparator is "
                 f"batch {need}; OGLANet scores are batch-size dependent "
                 "(results/21 finding 3) and a mismatch is not substituted "
                 "silently")
        tm = training_meta(chosen["run_id"], meta)
        rec = dict(n=len(bs_table) + 1, arch=arch, method=method, city=city,
                   run_id=chosen["run_id"],
                   dump="dumps/st_r450_published_recompute/" + chosen["run_id"],
                   n_images=int(len(df)),
                   **tm,
                   scorer_sha=meta.get("scorer_sha"),
                   metric_module_sha=meta.get("metric_module_sha"),
                   self_check_delta=(meta.get("scoring", {})
                                     .get("self_check", {}).get("delta")),
                   self_check_pass_0p01=(meta.get("scoring", {})
                                         .get("self_check", {})
                                         .get("pass_0p01")),
                   superseded_sibling=[c["run_id"] for c in cands
                                       if c["run_id"] != chosen["run_id"]],
                   **bs)
        # On a -bs4 cell the self-check compares two batch-4 passes and is 0 by
        # construction; the delta the Delta sessions flagged belongs to the
        # parent, whose side B is the trainer's batch-1 test(). Both are carried.
        if tm["training_metadata_parent_run_id"]:
            pm = json.load(open(os.path.join(
                RECOMPUTE, tm["training_metadata_parent_run_id"], "run.json")))
            rec["parent_self_check_delta"] = (pm.get("scoring", {})
                                              .get("self_check", {}).get("delta"))
            rec["parent_self_check_pass_0p01"] = (pm.get("scoring", {})
                                                  .get("self_check", {})
                                                  .get("pass_0p01"))
            rec["parent_headline_fg_iou_tol_pct"] = pm.get(
                "headline_fg_iou_tol_pct")
        else:
            rec["parent_self_check_delta"] = None
        bs_table.append(rec)
        print(f"  {rec['n']:2d}. {rec['run_id']:62s} bs="
              f"{rec['inference_batch_size']} "
              f"({'run.json' if not rec['from_session_log'] else 'SESSION LOG'})"
              f"  n_val={rec['n_val']}  epochs={rec['epochs_run']}/"
              f"{rec['epochs_max']} best={rec['best_epoch']} "
              f"{rec['stopped_on']}")
    if len(bs_table) != 27:
        fail(f"{len(bs_table)} n450 cells selected, expected 27")

    shas = sorted({r["scorer_sha"] for r in bs_table})
    if len(shas) != 1:
        fail(f"the 27 n450 cells carry {len(shas)} scorer_sha values: {shas}")
    print(f"\n  one scorer_sha on all 27: {shas[0]}")
    mm = sorted({r["metric_module_sha"] for r in bs_table})
    print(f"  metric_module_sha on the 27: {mm}")

    # ---- dry run stops here ---------------------------------------------
    if DRY_RUN:
        d0, m0, _ = read_cell(RECOMPUTE, bs_table[0]["run_id"])
        print("\n--- resolved dump schema (CONTRACT.md section 3) ---")
        print(f"  {bs_table[0]['run_id']}: {len(d0.columns)} columns, "
              f"{len(d0)} rows")
        for c in sorted(d0.columns):
            print(f"    {c:24s} {str(d0[c].dtype)}")
        print("\n--- published counterparts ---")
        for arch, method, city in N450_GRID:
            rid = published_run_id(arch, method, city)
            ok = os.path.isfile(os.path.join(PNG, rid, "perimage.parquet"))
            print(f"  {rid:46s} {'PNG dump' if ok else 'ABSENT -> Table 38 bound'}")
        print("\nDRY_RUN: nothing written.")
        return

    # ---- read every cell -------------------------------------------------
    def load(tree_root, tree_label, rid):
        df, meta, why = read_cell(tree_root, rid)
        if df is None:
            return None
        m2, fg, _ = two_class_tol_k2(df)
        return dict(run_id=rid, tree=tree_label, df=df, meta=meta,
                    per_image_2c=m2, per_image_fg=fg,
                    value_2c=float(m2.mean()), value_fg=float(fg.mean()),
                    image_id=list(df.image_id),
                    pred_pos_frac=pooled_pred_pos_fraction(df),
                    empty_fg_images=int(df.empty_fg_valid.sum()))

    cells = {}            # ("n450"|"published"|"upper", arch, method, city)
    for arch, method, city in N450_GRID:
        c = load(RECOMPUTE, "st_r450_published_recompute",
                 selected[(arch, method, city)])
        if c is None:
            fail(f"{selected[(arch, method, city)]}: unreadable after selection")
        cells[("n450", arch, method, city)] = c
        p = load(PNG, "st_r450_published_png",
                 published_run_id(arch, method, city))
        if p is not None:
            cells[("published", arch, method, city)] = p
    for arch in ARCHS:
        for city in CITIES:
            u = load(PNG, "st_r450_published_png",
                     published_run_id(arch, "none", city))
            if u is None:
                fail(f"{published_run_id(arch, 'none', city)}: the within-city "
                     "upper bound must be on the PNG tree")
            cells[("upper", arch, "none", city)] = u

    # ---- image_id identity, asserted not assumed -------------------------
    by_city = {}
    for k, c in cells.items():
        by_city.setdefault(k[3], []).append((k, c["image_id"]))
    id_check = {}
    for city, lst in sorted(by_city.items()):
        ref = lst[0][1]
        bad_set = [f"{k[0]}/{k[1]}/{k[2]}" for k, ids in lst
                   if sorted(ids) != sorted(ref)]
        if bad_set:
            fail(f"{city}: image_id SET differs on {bad_set}")
        bad_order = [f"{k[0]}/{k[1]}/{k[2]}" for k, ids in lst if ids != ref]
        if bad_order:
            fail(f"{city}: image_id ORDER differs on {bad_order}; every delta "
                 "here is a positional subtraction")
        id_check[city] = dict(n_cells=len(lst), n_images=len(ref),
                              same_set=True, same_order=True,
                              reference_cell=f"{lst[0][0][0]}/{lst[0][0][1]}/"
                                             f"{lst[0][0][2]}",
                              order_is_lexicographic=(ref == sorted(ref)))
    print("\n  image_id sets AND order identical within every city: "
          + ", ".join(f"{c} ({v['n_cells']} cells, n={v['n_images']})"
                      for c, v in sorted(id_check.items())))

    # ---- collapse flag ---------------------------------------------------
    collapse = []
    for (kind, arch, method, city), c in sorted(cells.items()):
        f = c["pred_pos_frac"]
        collapse.append(dict(
            side=kind, arch=arch, method=method, city=city,
            run_id=c["run_id"], tree=c["tree"],
            pooled_pred_pos_fraction=f,
            flagged=bool(f > COLLAPSE_RULE["threshold"]),
            value_2c=c["value_2c"], value_fg=c["value_fg"]))
    flagged = [r for r in collapse if r["flagged"]]
    print(f"\n--- collapse flag, results/01's rule at "
          f"{COLLAPSE_RULE['threshold']} ---")
    print(f"  cells checked {len(collapse)} "
          f"(n450 27, published 25, upper 9); flagged {len(flagged)}")
    cmax = max(collapse, key=lambda r: r["pooled_pred_pos_fraction"])
    print(f"  largest pooled predicted-positive fraction: "
          f"{cmax['pooled_pred_pos_fraction']:.6f} at {cmax['run_id']} "
          f"({cmax['side']})")
    for r in flagged:
        print(f"  FLAGGED {r['run_id']} {r['pooled_pred_pos_fraction']:.6f}")
    flagged_keys = {(r["arch"], r["method"], r["city"]) for r in flagged}

    # ---- the 27 rows -----------------------------------------------------
    rows, test_idx = [], []
    for arch, method, city in N450_GRID:
        n = cells[("n450", arch, method, city)]
        p = cells.get(("published", arch, method, city))
        u = cells[("upper", arch, "none", city)]
        t38 = (TABLE38[(method, arch)][CITIES.index(city)]
               if (method, arch) in TABLE38 else None)
        meta = n["meta"]
        bsrec = next(r for r in bs_table if r["run_id"] == n["run_id"])
        row = dict(
            arch=arch, method=method, method_label=METHOD_LABEL[method],
            city=city,
            n450_run_id=n["run_id"],
            n450_tree=n["tree"],
            published_run_id=published_run_id(arch, method, city),
            published_tree=(p["tree"] if p else None),
            upper_run_id=u["run_id"], upper_tree=u["tree"],
            n=150,
            inference_batch_size=bsrec["inference_batch_size"],
            n_val=bsrec["n_val"], n_train=bsrec["n_train"],
            epochs_run=bsrec["epochs_run"],
            epochs_max=bsrec["epochs_max"],
            best_epoch=bsrec["best_epoch"],
            stopped_on=bsrec["stopped_on"],
            target_access=bsrec["target_access"],
            collapse_flagged_n450=bool(n["pred_pos_frac"]
                                       > COLLAPSE_RULE["threshold"]),
            collapse_flagged_published=(None if p is None else
                                        bool(p["pred_pos_frac"]
                                             > COLLAPSE_RULE["threshold"])),
            pooled_pred_pos_fraction_n450=n["pred_pos_frac"],
            pooled_pred_pos_fraction_published=(p["pred_pos_frac"] if p
                                                else None),
            empty_fg_images=n["empty_fg_images"],
            # --- the declared path: two-class K=2 ---
            n450_mIoU_tol_K2=n["value_2c"],
            published_mIoU_tol_K2=(p["value_2c"] if p else t38),
            published_mIoU_tol_K2_is_bound=(p is None),
            # --- foreground IoU K=2 ---
            n450_fg_IoU_K2=n["value_fg"],
            published_fg_IoU_K2=(p["value_fg"] if p else None),
            # --- the upper bound ---
            upper_mIoU_tol_K2=u["value_2c"], upper_fg_IoU_K2=u["value_fg"],
            table38_printed=t38)
        if p is not None:
            row.update(
                delta_mIoU_tol_K2=n["value_2c"] - p["value_2c"],
                delta_fg_IoU_K2=n["value_fg"] - p["value_fg"],
                published_table38_delta=p["meta"].get("table38_delta"),
                published_table38_gate_pass=p["meta"].get("table38_gate_pass"),
                published_final_comparison_delta=p["meta"].get(
                    "final_comparison_delta"),
                published_png_minus_probmap_mIoU_K2=(
                    (p["meta"].get("recompute_counterpart") or {}).get(
                        "delta_png_minus_recompute_miou_k2")),
                published_probmap_mIoU_tol_K2=(
                    (p["meta"].get("recompute_counterpart") or {}).get(
                        "miou_k2_twoclass")),
                published_probmap_fg_IoU_K2=(
                    (p["meta"].get("recompute_counterpart") or {}).get(
                        "fg_iou_k2")))
            d2 = n["per_image_2c"] - p["per_image_2c"]
            dfgv = n["per_image_fg"] - p["per_image_fg"]
            bs2, bsf = paired_bootstrap(d2), paired_bootstrap(dfgv)
            row.update(
                delta_mIoU_test=dict(
                    name="paired image-level bootstrap on per-image two-class "
                         "K=2 deltas, n450 minus published, two-tailed",
                    **bs2),
                delta_fg_test=dict(
                    name="paired image-level bootstrap on per-image foreground "
                         "IoU K=2 deltas, n450 minus published, two-tailed",
                    **bsf),
                paired_across_trees=True,
                paired_across_prediction_populations=(arch == "oglanet"))
            test_idx.append((len(rows), bs2["p"]))
        else:
            row.update(
                delta_mIoU_tol_K2=(n["value_2c"] - t38
                                   if t38 is not None else None),
                delta_mIoU_tol_K2_is_bound=True,
                delta_fg_IoU_K2=None,
                delta_mIoU_test=None, delta_fg_test=None,
                no_paired_test_reason=(
                    "the published cell has no per-image dump in either tree, "
                    "so the delta is against Table 38's printed 2-dp value and "
                    "is a bound, not a paired measurement; no paired test is "
                    "run on it"),
                published_fg_IoU_K2_absent_reason=(
                    "Table 38 is the two-class family (erratum E-7), so it "
                    "prints no foreground-only value for this cell and no "
                    "foreground bound exists"),
                published_own_test_results_fg_k2=(
                    (meta.get("scoring", {}).get("original") or {})
                    .get("orig_test_results_fg_k2")),
                published_own_test_results_note=(
                    "the published cell's own trainer test_results.json field "
                    "detailed.boundary_tolerant.tolerant_2px.iou, recorded in "
                    "the n450 cell's run.json. Reported as a recorded artifact, "
                    "never as a row value: results/21 finding 5 found 43 of 44 "
                    "audited test_results.json match no scoring definition in a "
                    "96-combination sweep"))
        # --- gaps, both signs, published and n450 ---
        for side, src in (("published", p), ("n450", n)):
            if src is None:
                row[f"gap_{side}"] = (t38 - u["value_2c"]
                                      if t38 is not None else None)
                row[f"gap_upper_minus_method_{side}"] = (
                    u["value_2c"] - t38 if t38 is not None else None)
                row[f"gap_{side}_is_bound"] = True
                continue
            row[f"gap_{side}"] = src["value_2c"] - u["value_2c"]
            row[f"gap_upper_minus_method_{side}"] = (u["value_2c"]
                                                     - src["value_2c"])
            row[f"gap_{side}_is_bound"] = False
            g = src["per_image_2c"] - u["per_image_2c"]
            gb = paired_bootstrap(g)
            row[f"gap_{side}_test"] = dict(
                name="paired image-level bootstrap on per-image two-class K=2 "
                     "deltas, method minus its within-city upper bound, "
                     "two-tailed", **gb)
            row[f"gap_fg_{side}"] = src["value_fg"] - u["value_fg"]
        if row.get("gap_published") is not None:
            row["gap_change_n450_minus_published"] = (row["gap_n450"]
                                                      - row["gap_published"])
        rows.append(row)

    # Holm-Bonferroni over the published-vs-n450 family (the testable cells)
    adj = holm([p for _, p in test_idx])
    for (i, _), a in zip(test_idx, adj):
        rows[i]["delta_mIoU_test"]["p_holm"] = a
        rows[i]["delta_mIoU_test"]["holm_family"] = (
            f"the {len(test_idx)} published-vs-n450 two-class deltas with both "
            "per-image vectors, m = " + str(len(test_idx)))

    # ---- MAMNet margins over MAMNet vanilla ------------------------------
    margins = []
    for method in ("fda", "segdesic", "iim", "isw", "mrfp_plus", "fada"):
        for city in CITIES:
            rec = dict(arch="mamnet", method=method,
                       method_label=METHOD_LABEL[method], city=city,
                       definition="method minus MAMNet Vanilla on the same "
                                  "side and city; positive means the method "
                                  "beats doing nothing")
            for side in ("published", "n450"):
                m = cells.get((side, "mamnet", method, city))
                v = cells.get((side, "mamnet", "vanilla", city))
                if m is None or v is None:
                    rec[f"margin_{side}"] = None
                    continue
                rec[f"margin_{side}"] = m["value_2c"] - v["value_2c"]
                rec[f"margin_fg_{side}"] = m["value_fg"] - v["value_fg"]
                rec[f"{side}_vanilla_run_id"] = v["run_id"]
                d = m["per_image_2c"] - v["per_image_2c"]
                rec[f"margin_{side}_test"] = dict(
                    name="paired image-level bootstrap, method minus MAMNet "
                         "Vanilla, same side, two-tailed",
                    **paired_bootstrap(d))
            if (rec.get("margin_published") is not None
                    and rec.get("margin_n450") is not None):
                rec["margin_change_n450_minus_published"] = (
                    rec["margin_n450"] - rec["margin_published"])
                mn = cells[("n450", "mamnet", method, city)]["per_image_2c"]
                vn = cells[("n450", "mamnet", "vanilla", city)]["per_image_2c"]
                mp = cells[("published", "mamnet", method, city)]["per_image_2c"]
                vp = cells[("published", "mamnet", "vanilla",
                            city)]["per_image_2c"]
                rec["margin_change_test"] = dict(
                    name="paired image-level bootstrap on the delta of deltas, "
                         "(method - vanilla) at n450 minus (method - vanilla) "
                         "published, two-tailed",
                    **paired_bootstrap((mn - vn) - (mp - vp)))
            margins.append(rec)

    # ---- summaries, twice: all cells and flagged cells excluded ----------
    def summarise(keep_keys, label):
        """Every summary over one cell set. keep_keys is a set of
        (arch, method, city)."""
        sel = [r for r in rows if (r["arch"], r["method"],
                                   r["city"]) in keep_keys]
        testable = [r for r in sel if r.get("delta_mIoU_test")]
        bound = [r for r in sel if r["published_mIoU_tol_K2_is_bound"]]

        def clus(getter):
            out = {}
            for r in testable:
                out.setdefault(r["city"], []).append(getter(r))
            return cluster_bootstrap_mean_over_cells(out) if out else None

        d2 = clus(lambda r: (cells[("n450", r["arch"], r["method"],
                                    r["city"])]["per_image_2c"]
                             - cells[("published", r["arch"], r["method"],
                                      r["city"])]["per_image_2c"]))
        dfg = clus(lambda r: (cells[("n450", r["arch"], r["method"],
                                     r["city"])]["per_image_fg"]
                              - cells[("published", r["arch"], r["method"],
                                       r["city"])]["per_image_fg"]))
        gp = clus(lambda r: (cells[("upper", r["arch"], "none",
                                    r["city"])]["per_image_2c"]
                             - cells[("published", r["arch"], r["method"],
                                      r["city"])]["per_image_2c"]))
        gn = clus(lambda r: (cells[("upper", r["arch"], "none",
                                    r["city"])]["per_image_2c"]
                             - cells[("n450", r["arch"], r["method"],
                                      r["city"])]["per_image_2c"]))

        def worst(key, recs=None):
            rr = [r for r in (recs if recs is not None else sel)
                  if r.get(key) is not None]
            if not rr:
                return None
            w = max(rr, key=lambda r: r[key])
            return dict(cell=f"{w['arch']}/{w['method']}/{w['city']}",
                        value=w[key], run_id=w.get("n450_run_id"),
                        is_bound=w.get(f"{key.rsplit('_', 1)[0]}_is_bound"))

        def best(key, recs=None):
            rr = [r for r in (recs if recs is not None else sel)
                  if r.get(key) is not None]
            if not rr:
                return None
            w = min(rr, key=lambda r: r[key])
            return dict(cell=f"{w['arch']}/{w['method']}/{w['city']}",
                        value=w[key])

        dd = [r["delta_mIoU_tol_K2"] for r in testable]
        ddf = [r["delta_fg_IoU_K2"] for r in testable]
        return dict(
            label=label,
            n_cells_in_set=len(sel),
            n_cells_with_a_paired_test=len(testable),
            n_cells_bounded_by_table38=len(bound),
            bounded_cells=[f"{r['arch']}/{r['method']}/{r['city']}"
                           for r in bound],
            cells_excluded_from_this_set=[
                f"{r['arch']}/{r['method']}/{r['city']}" for r in rows
                if (r["arch"], r["method"], r["city"]) not in keep_keys],
            # the retrain contrast
            mean_delta_mIoU_tol_K2=(float(np.mean(dd)) if dd else None),
            mean_delta_mIoU_tol_K2_cluster_bootstrap=d2,
            mean_delta_fg_IoU_K2=(float(np.mean(ddf)) if ddf else None),
            mean_delta_fg_IoU_K2_cluster_bootstrap=dfg,
            n_cells_delta_negative=sum(1 for v in dd if v < 0),
            n_cells_delta_positive=sum(1 for v in dd if v > 0),
            largest_drop_mIoU=best("delta_mIoU_tol_K2", testable),
            largest_rise_mIoU=worst("delta_mIoU_tol_K2", testable),
            largest_drop_fg=best("delta_fg_IoU_K2", testable),
            largest_rise_fg=worst("delta_fg_IoU_K2", testable),
            # the gap, both sides
            worst_case_gap_published=worst("gap_upper_minus_method_published"),
            worst_case_gap_n450=worst("gap_upper_minus_method_n450"),
            best_case_gap_published=best("gap_upper_minus_method_published"),
            best_case_gap_n450=best("gap_upper_minus_method_n450"),
            mean_gap_published_cluster_bootstrap=gp,
            mean_gap_n450_cluster_bootstrap=gn,
            mean_gap_published_over_cells=float(np.mean(
                [r["gap_upper_minus_method_published"] for r in sel
                 if r.get("gap_upper_minus_method_published") is not None])),
            mean_gap_n450_over_cells=float(np.mean(
                [r["gap_upper_minus_method_n450"] for r in sel
                 if r.get("gap_upper_minus_method_n450") is not None])),
            n_cells_scoring_above_their_upper_bound_published=sum(
                1 for r in sel
                if (r.get("gap_upper_minus_method_published") or 0) < 0),
            n_cells_scoring_above_their_upper_bound_n450=sum(
                1 for r in sel
                if (r.get("gap_upper_minus_method_n450") or 0) < 0),
            per_arch=(lambda: {
                a: dict(
                    n_cells=sum(1 for r in sel if r["arch"] == a),
                    mean_delta_mIoU=(float(np.mean(
                        [r["delta_mIoU_tol_K2"] for r in testable
                         if r["arch"] == a]))
                        if any(r["arch"] == a for r in testable) else None),
                    worst_case_gap_published=worst(
                        "gap_upper_minus_method_published",
                        [r for r in sel if r["arch"] == a]),
                    worst_case_gap_n450=worst(
                        "gap_upper_minus_method_n450",
                        [r for r in sel if r["arch"] == a]))
                for a in sorted({r["arch"] for r in sel})})(),
            per_city=(lambda: {
                c: dict(
                    n_cells=sum(1 for r in sel if r["city"] == c),
                    mean_delta_mIoU=(float(np.mean(
                        [r["delta_mIoU_tol_K2"] for r in testable
                         if r["city"] == c]))
                        if any(r["city"] == c for r in testable) else None),
                    worst_case_gap_n450=worst(
                        "gap_upper_minus_method_n450",
                        [r for r in sel if r["city"] == c]))
                for c in CITIES})(),
        )

    all_keys = {(r["arch"], r["method"], r["city"]) for r in rows}
    kept_keys = {k for k in all_keys
                 if k not in {(a, m, c) for (a, m, c) in flagged_keys}}
    summaries = dict(
        all_cells=summarise(all_keys, "all 27 cells"),
        flagged_excluded=summarise(
            kept_keys,
            "collapse-flagged cells excluded on either side "
            f"({len(all_keys) - len(kept_keys)} excluded)"))

    # ---- the 27-row table, printed ---------------------------------------
    tab = pd.DataFrame([{k: r.get(k) for k in (
        "arch", "method", "city", "published_mIoU_tol_K2",
        "n450_mIoU_tol_K2", "delta_mIoU_tol_K2", "published_fg_IoU_K2",
        "n450_fg_IoU_K2", "delta_fg_IoU_K2", "upper_mIoU_tol_K2",
        "gap_upper_minus_method_published", "gap_upper_minus_method_n450",
        "published_mIoU_tol_K2_is_bound", "collapse_flagged_n450",
        "inference_batch_size")} for r in rows])
    pd.set_option("display.width", 260)
    print("\n--- the 27 rows: declared path (two-class K=2) and foreground "
          "IoU K=2 ---")
    print(tab.to_string(index=False,
                        float_format=lambda v: f"{v:+.4f}"))

    print("\n--- MAMNet: margin over MAMNet Vanilla, published vs n450 ---")
    print(pd.DataFrame([{k: m.get(k) for k in (
        "method", "city", "margin_published", "margin_n450",
        "margin_change_n450_minus_published")} for m in margins]).to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    for name, s in summaries.items():
        print(f"\n--- summary: {s['label']} ---")
        print(f"  cells {s['n_cells_in_set']}, paired tests "
              f"{s['n_cells_with_a_paired_test']}, Table-38 bounds "
              f"{s['n_cells_bounded_by_table38']}")
        print(f"  mean delta (n450 - published) two-class "
              f"{s['mean_delta_mIoU_tol_K2']:+.4f}  "
              f"CI95 [{s['mean_delta_mIoU_tol_K2_cluster_bootstrap']['ci95'][0]:+.4f}, "
              f"{s['mean_delta_mIoU_tol_K2_cluster_bootstrap']['ci95'][1]:+.4f}]  "
              f"p {s['mean_delta_mIoU_tol_K2_cluster_bootstrap']['p']:.4f}")
        print(f"  mean delta foreground IoU "
              f"{s['mean_delta_fg_IoU_K2']:+.4f}  "
              f"CI95 [{s['mean_delta_fg_IoU_K2_cluster_bootstrap']['ci95'][0]:+.4f}, "
              f"{s['mean_delta_fg_IoU_K2_cluster_bootstrap']['ci95'][1]:+.4f}]")
        print(f"  cells down {s['n_cells_delta_negative']} / up "
              f"{s['n_cells_delta_positive']}")
        print(f"  worst-case gap published {s['worst_case_gap_published']['value']:+.4f} "
              f"at {s['worst_case_gap_published']['cell']}")
        print(f"  worst-case gap n450      {s['worst_case_gap_n450']['value']:+.4f} "
              f"at {s['worst_case_gap_n450']['cell']}")
        print(f"  mean gap published {s['mean_gap_published_over_cells']:+.4f} -> "
              f"n450 {s['mean_gap_n450_over_cells']:+.4f}")

    # ---- Task C: the OGLANet Phoenix near-tie ----------------------------
    near_tie = near_tie_block(cells, selected)

    # ---- sanity flags ----------------------------------------------------
    sanity = sanity_flags(rows, bs_table, collapse, cells, summaries, near_tie)
    print("\n--- sanity flags ---")
    for s in sanity:
        print("  * " + s[:300].replace("\n", " "))

    # ---- caveats ---------------------------------------------------------
    caveats = build_caveats(bs_table, rows, near_tie)

    headline = (
        "Across the 27 matched-450 retrains the two-class boundary-tolerant "
        "mIoU at K=2 changes by "
        f"{summaries['all_cells']['mean_delta_mIoU_tol_K2']:+.2f} on average "
        f"against the published cells "
        f"(95% CI "
        f"[{summaries['all_cells']['mean_delta_mIoU_tol_K2_cluster_bootstrap']['ci95'][0]:+.2f}, "
        f"{summaries['all_cells']['mean_delta_mIoU_tol_K2_cluster_bootstrap']['ci95'][1]:+.2f}], "
        f"{summaries['all_cells']['n_cells_with_a_paired_test']} of 27 cells "
        "paired), and the worst-case gap to the within-city upper bound over "
        "these cells moves from "
        f"{summaries['all_cells']['worst_case_gap_published']['value']:.2f} at "
        f"{summaries['all_cells']['worst_case_gap_published']['cell']} to "
        f"{summaries['all_cells']['worst_case_gap_n450']['value']:.2f} at "
        f"{summaries['all_cells']['worst_case_gap_n450']['cell']}.")
    print("\nheadline: " + headline)

    out = dict(
        id="25_matched450_retrain_record",
        arm="H2",
        answers=["AC1"],
        answers_detail=(
            "AC1 asks us not to overstate and to acknowledge confounders. The "
            "confounder here is the training-budget asymmetry that "
            "results/21_disclosure_record.json finding 1 measured: 27 published "
            "cells trained on 900 source images against 450-image within-city "
            "upper bounds. This job is the record of those 27 cells retrained "
            "at 450. It also supplies the basis on which "
            "results/26_internal_sib_worst_case_n450.json re-adjudicates arm "
            "F's pre-committed readings (BxRx-W3, AC3, v1UJ-W3); this JSON "
            "claims no reading of its own."),
        question=("What do the 27 published cells that trained on 900 source "
                  "images score when retrained at 450, on the declared path and "
                  "on foreground IoU, and how do their gaps to the within-city "
                  "upper bound and their margins over MAMNet Vanilla move?"),
        status="COMPLETE",
        declared_path=dict(
            metric="two_class_mIoU_boundary_tolerant_K2_pct",
            definition=("per image, the mean of foreground IoU and background "
                        "IoU computed from raw tp/fp/fn/tn on the valid region "
                        "with a +-2 px don't-care band, then macro-averaged "
                        "over the cell's 150 test images"),
            secondary_metric="fg_IoU_boundary_tolerant_K2_pct (CONTRACT.md "
                             "section 3 iou_tol; Appendix E's stated "
                             "foreground-only definition)",
            degenerate_rule=("App E: an image with tp+fp+fn == 0 at K=2 scores "
                             "foreground 0 and background 100, so its "
                             "two-class value is exactly 50"),
            sign_conventions=dict(
                delta="n450 minus published; negative means the matched-450 "
                      "retrain scores below the published 900-image cell",
                gap="method minus its within-city upper bound (CONVENTIONS.md "
                    "sign, negative)",
                gap_upper_minus_method="upper bound minus method (positive is "
                                       "a shortfall); \"worst case\" always "
                                       "means the largest of these over a set",
                margin_over_vanilla="method minus MAMNet Vanilla on the same "
                                    "side and city; positive beats doing "
                                    "nothing"),
            resolution="hr", rung=450,
            published_source=("dumps/st_r450_published_png (the saved "
                              "prediction PNG masks, the tree Table 38 and "
                              "final_comparison.json were scored from)"),
            n450_source=("dumps/st_r450_published_recompute/*-s42-n450[-bs4], "
                         "scored in-job by n450_score.py from each cell's own "
                         "probability maps"),
            upper_bound_source="dumps/st_r450_published_png, the same tree as "
                               "the published method cells",
            image_id_identity=id_check,
            validated_against=[
                "stored per-image iou_fg_tol_pct / iou_bg_tol_pct / "
                "miou_tol_pct columns, max |delta| < 1e-9 on every cell read",
                "every CONTRACT.md section 3 column present with its declared "
                "dtype, never coerced",
                "150 rows on every cell",
                "one scorer_sha across all 27 n450 cells",
                "the published dumps' own recorded Table 38 and "
                "final_comparison.json deltas"]),
        n450_cells=bs_table,
        n450_cell_selection_rule=(
            "candidates are directories matching "
            "{arch}-{method}-loco-r450-{city}-hr-s<seed>-n450<suffix> under "
            "dumps/st_r450_published_recompute that carry both a "
            "perimage.parquet and a .DONE marker; a -bs4 re-score is preferred "
            "over its parent; the chosen cell's recorded inference batch size "
            "must equal the batch size its published comparator was produced "
            "at, which is asserted, not assumed."),
        n450_cell_candidates=candidate_log,
        inference_batch_size_required=(
            {f"{a}/{m}": v for (a, m), v in REQUIRED_INFER_BS.items()}),
        collapse_rule=COLLAPSE_RULE,
        collapse_check=collapse,
        collapse_flagged=flagged,
        per_cell=[{k: v for k, v in r.items()} for r in rows],
        mamnet_margin_over_vanilla=margins,
        summaries=summaries,
        oglanet_phoenix_near_tie=near_tie,
        conditions=conditions_block(rows, summaries),
        tests=tests_block(rows, margins, summaries),
        runs=sorted({r["n450_run_id"] for r in rows}
                    | {r["published_run_id"] for r in rows
                       if not r["published_mIoU_tol_K2_is_bound"]}
                    | {r["upper_run_id"] for r in rows}),
        runs_excluded=[
            dict(run_id=published_run_id(r["arch"], r["method"], r["city"]),
                 reason="no per-image dump in the PNG tree or the recompute "
                        "tree; its published value enters only as an explicit "
                        "Table 38 bound")
            for r in rows if r["published_mIoU_tol_K2_is_bound"]]
        + [dict(run_id="mamnet-vanilla-loco-r450-phoenix-hr-s42-n450"
                       ".failed_23272684",
                reason=".FAILED marker, no perimage.parquet; the preserved "
                       "timed-out first attempt of that cell"),
           dict(run_id="oglanet-segdesic-loco-r450-phoenix-hr-s42-n450",
                reason="superseded by its -bs4 re-score, which is the pass the "
                       "brief names; its value is reported in the sanity flags "
                       "and in caveat 3, never as a row value")],
        index_sha256=indexes["st_r450_published_recompute"]["sha256"],
        index_sha256_note=("the matched-450 retrains' tree index, built by this "
                           "job. The published PNG tree's index is in "
                           "index_sha256_all."),
        index_sha256_all={k: dict(path=v.get("path"), tree=v.get("root"),
                                  sha256=v.get("sha256"), n_rows=v["n_rows"],
                                  built_here=True,
                                  n_unindexable=v["n_unindexable"],
                                  unindexable=v["unindexable"])
                          for k, v in indexes.items()},
        index_on_disk_state=on_disk,
        source_artifacts=[
            dict(path="results/21_disclosure_record.json",
                 role="finding 1 (the 27 cells' measured training counts and "
                      "the matched-450 retrain list) and finding 3 (the DFFM "
                      "defect and the inference batch size of every published "
                      "OGLANet artifact family)"),
            dict(path="results/01_scaling_ladder.json",
                 role="the collapse rule, its 0.5 threshold and that "
                      "threshold's provenance, applied here verbatim"),
            dict(path="results/22_internal_sib_worst_case_v3.json",
                 role="the declared path, the gap signs, the metric "
                      "implementation and caveat 1, which this job's cells "
                      "address"),
            dict(path="delta_logs/session_logs/2026-10-28-n450-bs4-isw.md",
                 role="the inference batch size of the two cells whose "
                      "run.json field is null, the batch-4 re-score of "
                      "oglanet-segdesic phoenix, and the g3 self-check "
                      "explanation"),
            dict(path="delta_logs/session_logs/2026-10-29-n450-final-sar.md",
                 role="the four failing self-checks, the two null batch-size "
                      "fields, the two missing published comparators, the "
                      "near-collapse Phoenix reading, and the arm F SAR n450 "
                      "build"),
            dict(path="final_paper.md",
                 role="Table 38 printed values, used only in agreement checks "
                      "and in the two labelled missing-cell bounds")],
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/25_matched450_retrain_record.py",
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
                     "inference, no checkpoint loaded, nothing under dumps/ "
                     "and no existing file under results/ modified.",
        headline=headline,
        sanity_flags=sanity,
        caveats=caveats,
        figure=None,
        caption=None,
    )

    write_json(out)


# ----------------------------------------------------------------------------
# Task C -- the OGLANet Phoenix near-tie
# ----------------------------------------------------------------------------

def probs_of(root, run_id, image_ids):
    """Per-image float16 probability maps from a dump's probs.npz, in the
    parquet's image_id order."""
    p = os.path.join(root, run_id, "probs.npz")
    if not os.path.isfile(p):
        return None, f"no probs.npz in {run_id}"
    z = np.load(p, allow_pickle=False)
    keys = set(z.files)
    missing = [i for i in image_ids if i not in keys]
    if missing:
        return None, (f"{run_id}: probs.npz is missing "
                      f"{len(missing)} of {len(image_ids)} image_ids")
    return [np.asarray(z[i], dtype=np.float32) for i in image_ids], None


def mask_of(prob, min_pixels=10):
    """The originals' primary prediction recipe,
    threshold_and_filter(prob, min_pixels=10), and the plain p > 0.5 mask."""
    raw = prob > 0.5
    try:
        from scipy import ndimage
    except Exception as e:                       # loud, never silent
        fail(f"scipy.ndimage is required for the 10-px component filter: {e}")
    lab, n = ndimage.label(raw)
    if n == 0:
        return raw, raw
    sizes = ndimage.sum(raw, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_pixels
    return raw, keep[lab]


def pair_stats(a, b, label, same_images):
    """fg-IoU correlation, thresholded-mask pixel agreement, mean |dp|."""
    fga, fgb = a["per_image_fg"], b["per_image_fg"]
    n = min(fga.size, fgb.size)
    pear = float(np.corrcoef(fga[:n], fgb[:n])[0, 1])
    order = np.argsort
    ra = np.argsort(order(fga[:n])).astype(np.float64)
    rb = np.argsort(order(fgb[:n])).astype(np.float64)
    spear = float(np.corrcoef(ra, rb)[0, 1])
    pa, whya = probs_of(RECOMPUTE, a["run_id"], a["image_id"])
    pb, whyb = probs_of(RECOMPUTE, b["run_id"], b["image_id"])
    rec = dict(label=label, pair=[a["run_id"], b["run_id"]],
               same_test_images=bool(same_images),
               n_images=int(n),
               fg_iou_mean=[a["value_fg"], b["value_fg"]],
               fg_iou_mean_difference=float(a["value_fg"] - b["value_fg"]),
               two_class_mean=[a["value_2c"], b["value_2c"]],
               per_image_fg_iou_pearson_r=pear,
               per_image_fg_iou_spearman_rho=spear,
               per_image_fg_iou_mean_abs_difference=float(
                   np.abs(fga[:n] - fgb[:n]).mean()),
               per_image_fg_iou_max_abs_difference=float(
                   np.abs(fga[:n] - fgb[:n]).max()),
               n_images_identical_fg_iou=int((fga[:n] == fgb[:n]).sum()))
    if pa is None or pb is None:
        rec.update(probs_available=False,
                   probs_blocker=[x for x in (whya, whyb) if x])
        return rec
    agree_f, agree_r, absd, tot = 0, 0, 0.0, 0
    for x, y in zip(pa, pb):
        if x.shape != y.shape:
            fail(f"{label}: probability maps differ in shape "
                 f"{x.shape} vs {y.shape}")
        rx, fx = mask_of(x)
        ry, fy = mask_of(y)
        agree_f += int((fx == fy).sum())
        agree_r += int((rx == ry).sum())
        absd += float(np.abs(x - y).sum())
        tot += x.size
    rec.update(
        probs_available=True,
        n_pixels_compared=int(tot),
        pixel_agreement_filtered_recipe=agree_f / tot,
        pixel_agreement_plain_p_gt_0p5=agree_r / tot,
        mean_absolute_probability_difference=absd / tot,
        recipe_note=("filtered = threshold_and_filter(prob, min_pixels=10), the "
                     "originals' primary recipe and the one n450_score.py "
                     "scored; plain = p > 0.5 with no size filter. Agreement is "
                     "over all 384x384 pixels of all 150 images, not over the "
                     "valid region: the dumps carry n_valid_px but not the "
                     "don't-care mask itself."))
    return rec


def near_tie_block(cells, selected):
    out = dict(
        what=("oglanet-sibc4 and oglanet-segdesic Phoenix at n450. Three "
              "measurements on the pair -- per-image foreground-IoU "
              "correlation, pixel agreement of the thresholded masks, mean "
              "absolute probability difference -- with reference pairs beside "
              "them. Numbers only, no verdict."),
        which_segdesic_pass=(
            "The near-tie the brief names, 25.639580 against 25.637529, is "
            "between SIB and SegDeSiC's SUPERSEDED batch-4 pass. The pass Task "
            "A's rows use is the -bs4 re-score at 25.751900, a fresh batch-4 "
            "pass of the same checkpoint that sits +0.114371 away from it. Both "
            "pairs are measured here: the brief's pair is the near-tie, and the "
            "-bs4 pair is the one Task A's numbers are built on."),
        n450_pair=None, n450_pair_bs4=None, published_pair=None,
        reference_pairs=[])
    bs_note = (
        "the two sides are not one inference regime: SIB is batch 8 under fp16 "
        "autocast from the SIB trainer's own test loader, SegDeSiC is batch 4 "
        "fp32 from run_inference_probs.py. OGLANet scores are batch-size "
        "dependent (results/21 finding 3), so part of any disagreement here is "
        "the regime and not the model.")
    a = cells[("n450", "oglanet", "sibc4", "phoenix")]
    # the brief's pair: SegDeSiC's superseded, non-bs4 batch-4 pass
    sup = "oglanet-segdesic-loco-r450-phoenix-hr-s42-n450"
    sdf, smeta, swhy = read_cell(RECOMPUTE, sup)
    if sdf is None:
        fail(f"{sup}: {swhy}; it is the pass the brief's near-tie names")
    sm2, sfg, _ = two_class_tol_k2(sdf)
    sup_cell = dict(run_id=sup, tree="st_r450_published_recompute",
                    per_image_2c=sm2, per_image_fg=sfg,
                    value_2c=float(sm2.mean()), value_fg=float(sfg.mean()),
                    image_id=list(sdf.image_id))
    out["n450_pair"] = pair_stats(
        a, sup_cell,
        "n450 OGLANet Phoenix: SIB vs SegDeSiC, the brief's pair (SegDeSiC's "
        "superseded batch-4 pass)", True)
    out["n450_pair"]["batch_size_note"] = bs_note
    out["n450_pair_bs4"] = pair_stats(
        a, cells[("n450", "oglanet", "segdesic", "phoenix")],
        "n450 OGLANet Phoenix: SIB vs SegDeSiC -bs4, the pass Task A's rows "
        "use", True)
    out["n450_pair_bs4"]["batch_size_note"] = bs_note
    # the two SegDeSiC batch-4 passes against each other: a same-checkpoint,
    # same-batch-size reference for how far DFFM alone moves the prediction
    out["segdesic_pass_to_pass"] = pair_stats(
        sup_cell, cells[("n450", "oglanet", "segdesic", "phoenix")],
        "DFFM pass-to-pass reference: the two batch-4 passes of the SAME "
        "oglanet-segdesic phoenix n450 checkpoint", True)
    out["segdesic_pass_to_pass"]["reading_note"] = (
        "one checkpoint, one batch size, two inference passes. Any "
        "disagreement here is DFFM nondeterminism alone, with no model "
        "difference, so it is the floor against which the SIB-vs-SegDeSiC "
        "numbers should be read.")
    # the published Phoenix pair
    pub_sib = os.path.join(PNG, "oglanet-sibc4-loco-r450-phoenix-hr-s0")
    out["published_pair"] = dict(
        label="published OGLANet Phoenix: SIB vs SegDeSiC",
        computable=False,
        blocker=(
            "the published OGLANet-SIB phoenix cell has no per-image dump in "
            "the PNG tree or the recompute tree (the 2026-10-25 56-cell "
            "re-score deliberately held OGLANet-SIB miami and phoenix pending "
            "the AMP question), and the published OGLANet-SegDeSiC cells have "
            "no probs.npz in either tree, so neither the per-image foreground "
            "IoU correlation nor a pixel-level comparison is defined for the "
            "published Phoenix pair."),
        png_sib_phoenix_dir_present=os.path.isdir(pub_sib),
        published_segdesic_phoenix_has_probs=os.path.isfile(os.path.join(
            RECOMPUTE, "oglanet-segdesic-loco-r450-phoenix-hr-s0",
            "probs.npz")),
        nearest_available_published_pair=(
            "chicago is the only city where a published OGLANet-SIB dump "
            "exists, and the published OGLANet-SegDeSiC chicago cell still has "
            "no probs.npz, so no published OGLANet SIB-vs-SegDeSiC pair is "
            "comparable at pixel level anywhere on this box."))
    # reference pairs
    out["reference_pairs"].append(pair_stats(
        cells[("n450", "oglanet", "sibc4", "chicago")],
        cells[("n450", "oglanet", "segdesic", "miami")],
        "cross-city reference: n450 OGLANet SIB chicago vs SegDeSiC miami",
        False))
    out["reference_pairs"][-1]["reading_note"] = (
        "the two sides score DIFFERENT 150-image test sets, so this pair is a "
        "reference level for unrelated OGLANet predictions, not a comparison. "
        "The per-image statistics pair image k of chicago with image k of "
        "miami, which is arbitrary by construction; the pixel statistics are "
        "well defined because both arrays are 150 x 384 x 384.")
    for city in ("chicago", "miami"):
        out["reference_pairs"].append(pair_stats(
            cells[("n450", "oglanet", "sibc4", city)],
            cells[("n450", "oglanet", "segdesic", city)],
            f"same-city reference: n450 OGLANet SIB vs SegDeSiC {city}",
            True))
        out["reference_pairs"][-1]["reading_note"] = (
            "the same method pair in a city where the two do not nearly tie, "
            "on one test set. Reported beside the cross-city reference because "
            "it is the only pairing that holds the images fixed.")
    return out


# ----------------------------------------------------------------------------
# CONTRACT section 6 conditions and tests
# ----------------------------------------------------------------------------

def conditions_block(rows, summaries):
    conds = []
    for r in rows:
        for side, key in (("published", "published_mIoU_tol_K2"),
                          ("n450", "n450_mIoU_tol_K2")):
            t = r.get(f"gap_{side}_test")
            conds.append(dict(
                label=f"{r['method_label']} / {r['arch']} / {r['city']} / "
                      f"{side} / gap to the within-city upper bound",
                metric="gap_upper_minus_method_mIoU_tol_K2_pct",
                mean=r.get(f"gap_upper_minus_method_{side}"),
                stderr=(t["stderr"] if t else None),
                n=(t["n"] if t else None),
                ci95=([-t["ci95"][1], -t["ci95"][0]] if t else None),
                ci_method=(t["ci_method"] if t else None),
                baseline="within-city upper bound, same architecture and city "
                         f"({r['upper_run_id']})",
                is_bound=bool(r.get(f"gap_{side}_is_bound")),
                value_mIoU=r.get(key),
                value_fg_IoU=r.get(f"{side}_fg_IoU_K2")))
    for name, s in summaries.items():
        cb = s["mean_delta_mIoU_tol_K2_cluster_bootstrap"]
        conds.append(dict(
            label=f"matched-450 retrain contrast / {s['label']} / mean "
                  "two-class K=2 change",
            metric="delta_mIoU_tol_K2_pct_n450_minus_published",
            mean=cb["mean"], stderr=cb["stderr"], n=cb["n_cells"],
            n_images=cb["n_images_total"], ci95=cb["ci95"],
            ci_method=cb["ci_method"],
            baseline="the same cell as published, 900 source images",
            is_bound=False))
        cf = s["mean_delta_fg_IoU_K2_cluster_bootstrap"]
        conds.append(dict(
            label=f"matched-450 retrain contrast / {s['label']} / mean "
                  "foreground IoU K=2 change",
            metric="delta_fg_IoU_K2_pct_n450_minus_published",
            mean=cf["mean"], stderr=cf["stderr"], n=cf["n_cells"],
            n_images=cf["n_images_total"], ci95=cf["ci95"],
            ci_method=cf["ci_method"],
            baseline="the same cell as published, 900 source images",
            is_bound=False))
        for side in ("published", "n450"):
            cg = s[f"mean_gap_{side}_cluster_bootstrap"]
            conds.append(dict(
                label=f"{s['label']} / {side} / mean gap to the within-city "
                      "upper bound",
                metric="gap_upper_minus_method_mIoU_tol_K2_pct",
                mean=cg["mean"], stderr=cg["stderr"], n=cg["n_cells"],
                n_images=cg["n_images_total"], ci95=cg["ci95"],
                ci_method=cg["ci_method"],
                baseline="within-city upper bound, same architecture and city",
                is_bound=False))
    return conds


def tests_block(rows, margins, summaries):
    tests = []
    for r in rows:
        t = r.get("delta_mIoU_test")
        if t:
            tests.append(dict(
                name=t["name"],
                compares=[r["n450_run_id"], r["published_run_id"]],
                statistic=t["mean"], p=t["p"], p_holm=t.get("p_holm"),
                holm_family=t.get("holm_family"),
                ci95=t["ci95"], B=t["B"], two_tailed=True,
                metric="mIoU_tol_K2_pct"))
        t = r.get("delta_fg_test")
        if t:
            tests.append(dict(
                name=t["name"],
                compares=[r["n450_run_id"], r["published_run_id"]],
                statistic=t["mean"], p=t["p"], ci95=t["ci95"], B=t["B"],
                two_tailed=True, metric="fg_IoU_tol_K2_pct",
                correction="none; the Holm family is defined on the two-class "
                           "deltas, which are the declared path"))
        for side in ("published", "n450"):
            t = r.get(f"gap_{side}_test")
            if t:
                tests.append(dict(
                    name=t["name"],
                    compares=[(r["n450_run_id"] if side == "n450"
                               else r["published_run_id"]), r["upper_run_id"]],
                    statistic=-t["mean"], p=t["p"], ci95=[-t["ci95"][1],
                                                          -t["ci95"][0]],
                    B=t["B"], two_tailed=True, side=side,
                    metric="gap_upper_minus_method_mIoU_tol_K2_pct"))
    for m in margins:
        for side in ("published", "n450"):
            t = m.get(f"margin_{side}_test")
            if t:
                tests.append(dict(
                    name=t["name"], compares=[f"mamnet/{m['method']}/"
                                              f"{m['city']}/{side}",
                                              f"mamnet/vanilla/{m['city']}/"
                                              f"{side}"],
                    statistic=t["mean"], p=t["p"], ci95=t["ci95"], B=t["B"],
                    two_tailed=True, metric="mIoU_tol_K2_pct"))
        t = m.get("margin_change_test")
        if t:
            tests.append(dict(
                name=t["name"],
                compares=[f"mamnet/{m['method']}/{m['city']} margin at n450",
                          f"mamnet/{m['method']}/{m['city']} margin published"],
                statistic=t["mean"], p=t["p"], ci95=t["ci95"], B=t["B"],
                two_tailed=True, metric="mIoU_tol_K2_pct"))
    for name, s in summaries.items():
        cb = s["mean_delta_mIoU_tol_K2_cluster_bootstrap"]
        tests.append(dict(
            name="paired image-cluster bootstrap on the matched-450 retrain "
                 "contrast",
            compares=[f"n450 / {s['label']}", f"published / {s['label']}"],
            statistic=cb["mean"], p=cb["p"], ci95=cb["ci95"], B=cb["B"],
            two_tailed=True,
            clustering="images resampled within held-out city, shared across "
                       "the cells of that city (App G.9)"))
    return tests


# ----------------------------------------------------------------------------
# sanity flags and caveats
# ----------------------------------------------------------------------------

def sanity_flags(rows, bs_table, collapse, cells, summaries, near_tie):
    s = []
    nt = near_tie["n450_pair"]
    b4 = near_tie["n450_pair_bs4"]
    pp = near_tie["segdesic_pass_to_pass"]
    s.append(
        "TOO CLEAN, flagged before being asked. oglanet-sibc4 and "
        "oglanet-segdesic Phoenix at n450 differ by "
        f"{abs(nt['fg_iou_mean_difference']):.6f} foreground-IoU points "
        f"({nt['fg_iou_mean'][0]:.6f} against {nt['fg_iou_mean'][1]:.6f}) after "
        "two different methods trained for 100 and 37 epochs. The pair is NOT "
        "two scorings of one prediction: their per-image foreground-IoU "
        f"correlation is {nt['per_image_fg_iou_pearson_r']:+.4f} (Spearman "
        f"{nt['per_image_fg_iou_spearman_rho']:+.4f}) with a mean absolute "
        f"per-image difference of "
        f"{nt['per_image_fg_iou_mean_abs_difference']:.4f} IoU points and "
        f"{nt['n_images_identical_fg_iou']} of {nt['n_images']} images "
        "identical, their thresholded masks agree on "
        f"{nt.get('pixel_agreement_filtered_recipe', float('nan')):.6f} of "
        "pixels and their mean absolute probability difference is "
        f"{nt.get('mean_absolute_probability_difference', float('nan')):.6f}. "
        "For scale on the same statistics: two batch-4 passes of the SAME "
        "SegDeSiC checkpoint agree on "
        f"{pp.get('pixel_agreement_filtered_recipe', float('nan')):.6f} of "
        "pixels at a mean absolute probability difference of "
        f"{pp.get('mean_absolute_probability_difference', float('nan')):.6f}, "
        "and SIB against SegDeSiC's -bs4 pass is "
        f"{abs(b4['fg_iou_mean_difference']):.6f} foreground-IoU points apart "
        f"with correlation {b4['per_image_fg_iou_pearson_r']:+.4f}. Task C "
        "reports the numbers; this job states no verdict on them.")
    chi = next((r for r in near_tie["reference_pairs"]
                if r["same_test_images"] and "chicago" in r["label"]), None)
    if chi is not None:
        s.append(
            "The Phoenix near-tie is not the only one in the pair. The same two "
            "n450 OGLANet methods land "
            f"{abs(chi['fg_iou_mean_difference']):.4f} foreground-IoU points "
            f"apart on chicago as well ({chi['fg_iou_mean'][0]:.4f} against "
            f"{chi['fg_iou_mean'][1]:.4f}), at a per-image correlation of "
            f"{chi['per_image_fg_iou_pearson_r']:+.4f}. So two of the three "
            "cities show a cell-mean near-tie between OGLANet SIB and OGLANet "
            "SegDeSiC while the per-image predictions disagree substantially. "
            "Reported as measured; nothing here identifies a cause.")
    bad_self = [r for r in bs_table
                if r["self_check_pass_0p01"] is False
                or r.get("parent_self_check_pass_0p01") is False]
    if bad_self:
        s.append(
            f"{len(bad_self)} of 27 n450 cells fail the trainer-versus-scorer "
            "self-check at the 0.01 tolerance, counting the superseded parent "
            "pass where the selected cell is a -bs4 re-score: "
            + ", ".join(
                f"{r['run_id'].split('-loco-')[0]}/{r['city']} "
                f"{(r['self_check_delta'] if r.get('parent_self_check_delta') is None else r['parent_self_check_delta']):+.4f}"
                + ("(parent)" if r.get("parent_self_check_pass_0p01") is False
                   else "")
                for r in bad_self)
            + ". The 2026-10-28 Delta session showed the three "
              "oglanet-segdesic deltas are a batch-size artefact of the "
              "comparison (the trainer's test() runs at batch 1, "
              "train_segdesic.py:296; the scorer reads a batch-4 pass) and "
              "collapse to 0.000000 when both sides are computed at batch 4. "
              "The deltas are a scoring-path difference, not a training one.")
    excess = [r for r in rows
              if r.get("delta_mIoU_test")
              and abs(r["delta_mIoU_tol_K2"]) > 5.0]
    if excess:
        s.append(
            f"{len(excess)} cells move by more than 5 two-class mIoU between "
            "the published 900-image cell and the matched-450 retrain: "
            + ", ".join(f"{r['arch']}/{r['method']}/{r['city']} "
                        f"{r['delta_mIoU_tol_K2']:+.2f}" for r in excess)
            + ". Halving the training budget is one explanation; a second is "
              "that training is not reproducible across jobs at a fixed seed "
              "(caveat 1), and a third is that the Phoenix cells sit near the "
              "floor where small absolute moves are large relative ones "
              "(caveat 2). Nothing here separates them.")
    flagged = [r for r in collapse if r["flagged"]]
    if flagged:
        s.append(f"{len(flagged)} of {len(collapse)} cells are collapse-flagged "
                 "by results/01's rule: "
                 + ", ".join(f"{r['run_id']} "
                             f"{r['pooled_pred_pos_fraction']:.4f}"
                             for r in flagged) + ".")
    else:
        cmax = max(collapse, key=lambda r: r["pooled_pred_pos_fraction"])
        s.append(
            f"No cell is collapse-flagged. The largest pooled predicted-positive "
            f"fraction over all {len(collapse)} cells read is "
            f"{cmax['pooled_pred_pos_fraction']:.6f} at {cmax['run_id']}, far "
            "below the 0.5 threshold, so the all-cells and flagged-excluded "
            "summaries are identical by construction and both are still "
            "reported. Low Phoenix scores here are NOT collapse in this rule's "
            "sense: a cell that predicts almost nothing has a near-zero "
            "predicted-positive fraction, and the rule only catches a cell that "
            "predicts almost everything.")
    floor = [r for r in rows if r["n450_fg_IoU_K2"] < 15.0]
    if floor:
        s.append(
            "Near-floor foreground IoU at n450 on "
            + ", ".join(f"{r['arch']}/{r['method']}/{r['city']} "
                        f"{r['n450_fg_IoU_K2']:.2f}" for r in floor)
            + ". Their published counterparts were "
            + ", ".join(f"{r['published_fg_IoU_K2']:.2f}"
                        if r["published_fg_IoU_K2"] is not None else "absent"
                        for r in floor)
            + ", so the condition is not new at n450 on every one of them. Kept "
              "as measured; no cell is reseeded, dropped or imputed.")
    above = [r for r in rows if (r.get("gap_upper_minus_method_n450") or 0) < 0]
    if above:
        s.append(
            "Cell(s) scoring ABOVE their own within-city upper bound at n450: "
            + ", ".join(f"{r['arch']}/{r['method']}/{r['city']} "
                        f"{r['gap_upper_minus_method_n450']:+.4f}"
                        for r in above)
            + ". The published grid already prints this pattern (Table 38's "
              "SegDeSiC/DINOv3/Phoenix 68.04 against the DINOv3/Phoenix upper "
              "bound 67.82), so it is a property of the grid rather than of "
              "this retrain.")
    og = [r for r in rows if r["arch"] == "oglanet"
          and r.get("published_png_minus_probmap_mIoU_K2") is not None]
    if og:
        s.append(
            "Every OGLANet row pairs two prediction populations. The published "
            "side is PNG-scored and the n450 side is scored from probability "
            "maps, and for OGLANet those are different predictions rather than "
            "two scorings of one prediction. Measured on the published cells "
            "themselves, PNG minus probability map is "
            + ", ".join(f"{r['method']}/{r['city']} "
                        f"{r['published_png_minus_probmap_mIoU_K2']:+.4f}"
                        for r in og)
            + " two-class mIoU, so that much of each OGLANet delta is the "
              "scoring population. The published probability-map value is "
              "carried on every row as published_probmap_mIoU_tol_K2 so the "
              "population-matched contrast can be read off directly.")
    man = [r for r in rows if r["arch"] == "mamnet"
           and r.get("published_png_minus_probmap_mIoU_K2") is not None]
    if man:
        mx = max(abs(r["published_png_minus_probmap_mIoU_K2"]) for r in man)
        s.append(
            f"On the 21 MAMNet rows the same PNG-minus-probability-map "
            f"difference is at most {mx:.4f} two-class mIoU, so the "
            "tree-mixing above is an OGLANet problem and not a MAMNet one.")
    seg = next((r for r in rows if r["arch"] == "oglanet"
                and r["method"] == "segdesic" and r["city"] == "phoenix"), None)
    if seg:
        s.append(
            "oglanet-segdesic phoenix has two batch-4 passes of the same "
            f"checkpoint: the original scored pass at 25.637529 foreground IoU "
            "and the -bs4 re-score this job uses at "
            f"{seg['n450_fg_IoU_K2']:.6f}, a difference of "
            f"{seg['n450_fg_IoU_K2'] - 25.637529:+.6f}. Both are batch 4, so "
            "that is DFFM pass-to-pass nondeterminism at a fixed batch size, "
            "inside erratum E-9's measured nine-pass span of 1.1158. The "
            "operator's brief names the -bs4 pass, which is what is used; the "
            "brief's own quoted 25.637529 is the superseded pass.")
    s.append(
        "One seed per n450 cell and one draw per published cell, so no cell "
        "here carries a seed band. The three n450 Phoenix cells nearest the "
        "floor are the ones where that matters most, and caveat 1 records the "
        "measured evidence that a rerun of a fixed-seed cell does not reproduce "
        "its own epoch count.")
    return s


def build_caveats(bs_table, rows, near_tie):
    by_id = {r["run_id"]: r for r in bs_table}
    seg_cells = [r for r in bs_table
                 if r["arch"] == "oglanet" and r["method"] == "segdesic"]
    sib_cells = [r for r in bs_table
                 if r["arch"] == "oglanet" and r["method"] == "sibc4"]
    nval = sorted({r["n_val"] for r in bs_table})
    return [
        # --- the five the brief requires, in its order ---
        "One seed (s42) per n450 cell against one published draw, so every "
        "difference in this JSON is a single-draw contrast and carries no seed "
        "band. Training is not reproducible across jobs at a fixed seed: the "
        "rerun of mamnet-segdesic-phoenix stopped at epoch "
        f"{by_id['mamnet-segdesic-loco-r450-phoenix-hr-s42-n450']['epochs_run']} "
        "(best "
        f"{by_id['mamnet-segdesic-loco-r450-phoenix-hr-s42-n450']['best_epoch']}) "
        "where the timed-out first attempt reached 52 (best 42) at the same "
        "seed, the same 450-image list and the same recipe "
        "(delta_logs/session_logs/2026-10-29-n450-final-sar.md, step 1). A "
        "difference of this kind cannot be separated from a training-budget "
        "effect on one seed.",

        "MAMNet Phoenix IIM, ISW and MRFP+ score "
        + ", ".join(
            f"{next(r['n450_fg_IoU_K2'] for r in rows if r['arch'] == 'mamnet' and r['method'] == m and r['city'] == 'phoenix'):.2f}"
            for m in ("iim", "isw", "mrfp_plus"))
        + " foreground IoU at n450 with normal validation (n_val 300, as every "
        "published pooled-LOCO cell). The published ISW and MRFP+ Phoenix cells "
        "were also low, at 5.51 and 9.59 foreground IoU, so the condition is "
        "not introduced by the retrain on those two; published IIM Phoenix was "
        "13.98. Kept as measured by operator decision; no cell was reseeded "
        "after test scores were seen, and none is dropped or imputed.",

        "OGLANet values remain subject to the DFFM defect "
        "(results/21_disclosure_record.json finding 3): "
        "python/oglanet/models/dffm.py builds 20 freshly random-initialised, "
        "unregistered Conv2d+BatchNorm2d layers inside forward() on every call, "
        "so model.eval() never reaches them, they normalise with batch "
        "statistics, and every OGLANet number is one draw at one inference "
        "batch size. The OGLANet SegDeSiC self-check deltas "
        + ", ".join(
            f"{r['city']} "
            f"{(r['self_check_delta'] if r.get('parent_self_check_delta') is None else r['parent_self_check_delta']):+.4f}"
            + (" (read from the -bs4 cell's parent, whose side B is the "
               "batch-1 pass; the -bs4 cell's own self-check compares two "
               "batch-4 passes and is 0.0000 by construction)"
               if r.get("parent_self_check_delta") is not None else "")
            for r in sorted(seg_cells, key=lambda r: r["city"]))
        + " compare the trainer's batch-1 test() with the batch-4 scorer "
        "(delta_logs/session_logs/2026-10-28-n450-bs4-isw.md, step 2: the same "
        "delta is 0.000000 when both sides are computed at batch 4); the "
        "batch-4 value is the scored one and is what this JSON reports. "
        "mamnet-mrfp_plus-phoenix's self-check delta is "
        f"{by_id['mamnet-mrfp_plus-loco-r450-phoenix-hr-s42-n450']['self_check_delta']:+.4f} "
        "against a 0.01 tolerance, which is a different and unexplained "
        "disagreement on an architecture with no DFFM.",

        "All three OGLANet SIB n450 cells stopped at the 100-epoch maximum, not "
        "on patience: "
        + ", ".join(f"{r['city']} epochs_run {r['epochs_run']} of "
                    f"{r['epochs_max']}, best {r['best_epoch']}, stopped_on "
                    f"{r['stopped_on']}"
                    for r in sorted(sib_cells, key=lambda r: r["city"]))
        + ". Their validation metric was still improving when the budget ran "
        "out, so these three are a lower bound on what the recipe reaches at "
        "450 images rather than a converged result.",

        "Validation-set size per cell as recorded in run.json: n_val = "
        + ", ".join(str(v) for v in nval)
        + " on all 27 cells. That matches every published pooled-LOCO cell, "
        "which also validated on 300, and it does NOT match paper section 3.3, "
        "CONTRACT.md's own example, or the 150 the prepared retrain list asked "
        "for (results/21 finding 1: \"The requested n_val=150 is a CHANGE from "
        "every published LOCO cell, not a restoration\"). Holding validation at "
        "300 is the choice that keeps the retrain a one-variable change in the "
        "training budget.",

        # --- this job's own limitations ---
        "Two of the 27 published comparators do not exist as per-image dumps: "
        "OGLANet-SIB miami and OGLANet-SIB phoenix. The 2026-10-25 56-cell "
        "re-score deliberately held those two pending the OGLANet SIB AMP "
        "question and the PNG scoring pass covered only what that re-score "
        "produced. Their published values enter only as Table 38's printed 2-dp "
        "value, marked as bounds, and no paired test is run on them. Table 38 "
        "is the two-class family (erratum E-7), so those two rows have no "
        "published foreground-IoU value at all; each cell's own "
        "test_results.json foreground value is carried beside the row as a "
        "recorded artifact, never as a value, because results/21 finding 5 "
        "found 43 of 44 audited test_results.json match no scoring definition.",

        "Every OGLANet row pairs a PNG-scored published value against a "
        "probability-map-scored n450 value, and for OGLANet those are different "
        "predictions rather than two scorings of one prediction (20 of 20 "
        "OGLANet cells disagree, 0.76 to 3.05 percent of pixels, up to 2.03 "
        "two-class mIoU). The published probability-map value is carried on "
        "every row as published_probmap_mIoU_tol_K2 and "
        "published_probmap_fg_IoU_K2, read from the PNG dump's own "
        "recompute_counterpart record, so the population-matched contrast can "
        "be read off directly. On the 21 MAMNet rows the two populations differ "
        "by at most 0.0147 two-class mIoU and the question does not arise.",

        "The three n450 OGLANet SIB cells were scored at inference batch size 8 "
        "under fp16 autocast, by the SIB trainer's own test loader. That is the "
        "regime the published Table 38 OGLANet-SIB row was produced in "
        "(train_oglanet_sib.py:1554 builds the test loader at args.batch_size "
        "8, use_amp defaults True, and the three published C4clean train.log "
        "Args lines all carry batch_size=8, use_amp=True), so the comparison is "
        "matched. It is NOT matched to the batch-4 fp32 regime of every other "
        "published OGLANet table, and no batch-4 OGLANet-SIB pass can be "
        "produced with the existing code: run_inference_probs.py has no SIB "
        "model_variant and prob_variant is None for every SIB cell, which is "
        "why job 23273276 failed.",

        "oglanet-segdesic phoenix is reported from its -bs4 re-score, as the "
        "brief directs. Its superseded original pass, also batch 4, scored "
        "25.637529 foreground IoU against the -bs4 pass's "
        f"{next(r['n450_fg_IoU_K2'] for r in rows if r['arch'] == 'oglanet' and r['method'] == 'segdesic' and r['city'] == 'phoenix'):.6f}; "
        "both are in the dump tree and only the -bs4 one enters a number here. "
        "The other two OGLANet SegDeSiC cells have no -bs4 directory because "
        "they were already scored from a batch-4 pass, which their run.json "
        "records directly.",

        "The inference batch size is read from run.json on 25 of the 27 cells. "
        "On mamnet-vanilla-phoenix the field is null and the value 4 comes from "
        "the session log line cited in this JSON's n450_cells block; that cell "
        "and the superseded oglanet-segdesic-phoenix pass are the only two with "
        "a null field, both single-cell or interactive resubmissions that "
        "predate the change which added it.",

        "The retrain changed the training budget and nothing else that is "
        "recorded, but it did not and could not restore the published cells' "
        "image draw. results/21 finding 1 states the problem: MAMNet's LOCO "
        "standard trained on all 900 source images and so never drew a "
        "450-image subset, so there is no \"standard models' exact image "
        "lists\" at 450 for MAMNet to return to. All 27 cells share one "
        "450-image list, recorded as train_list_sha256 with its source file, "
        "which is a new decision rather than a restoration.",

        "Thirteen of the 27 cells carry a published Table 38 printed value and "
        "the PNG dumps reproduce it on every one; the 21 MAMNet standard cells' "
        "rows are not printed in Table 38 and are checked against "
        "final_comparison.json instead, which they reproduce exactly. No value "
        "anywhere in this JSON is rounded to match a printed value, and Table 38 "
        "appears only inside agreement checks and inside the two labelled "
        "missing-cell bounds.",

        "CONTRACT.md rule zero puts the one metric implementation in "
        "shadowtransfer/metrics.py on the SLURM box. No copy of that file is "
        "reachable from the analysis host, so the metric is recomputed from the raw "
        "tp/fp/fn/tn counts and asserted on every cell read against the stored "
        "iou_fg_tol_pct / iou_bg_tol_pct / miou_tol_pct columns, max |delta| "
        "under 1e-9 everywhere. The implementation is character-identical to "
        "analysis/22_internal_sib_worst_case_v3.py's, deliberately, so the two "
        "jobs cannot differ on the metric.",

        "High resolution only, rung 450 only, two-class average "
        "boundary-tolerant mIoU at K=2 and foreground IoU at K=2 only. No "
        "mid-resolution cell was retrained, so erratum E-2's Table 2 side is "
        "still unchecked by this job.",

        "This JSON is a record and adjudicates nothing. It reports no reading "
        "of any pre-commitment and makes no claim about whether the budget "
        "asymmetry changed a conclusion; "
        "results/26_internal_sib_worst_case_n450.json re-runs arm F's "
        "adjudication on these cells and is where that question is answered.",

        "The collapse rule is results/01_scaling_ladder.json's, applied exactly: "
        "pooled n_pred_pos / n_valid_px greater than 0.5. Its threshold is "
        "recorded in results/01 as \"Not operator-authorised; derived from the "
        "ground truth\", and that provenance is carried here unchanged rather "
        "than re-derived.",
    ]


def write_json(out):
    if os.path.exists(OUT):
        fail(f"{OUT} already exists; this job writes exactly one new results "
             "file and never overwrites one")

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
            return {k: clean(v) for k, v in o.items()
                    if not isinstance(v, (pd.DataFrame, pd.Series))
                    and k != "df"}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return [clean(v) for v in o.tolist()]
        if isinstance(o, np.floating):
            return clean(float(o))
        if isinstance(o, float) and not np.isfinite(o):
            if np.isnan(o):
                return None
            fail(f"non-finite, non-missing value in the output: {o}")
        return o

    tmp = OUT + ".new"
    with open(tmp, "w") as fh:
        json.dump(clean(out), fh, indent=1, default=enc, allow_nan=False)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}")
    print(f"  sha256 {sha256_file(OUT)}")


if __name__ == "__main__":
    main()
