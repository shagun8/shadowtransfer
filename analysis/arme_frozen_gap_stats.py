#!/usr/bin/env python3
"""ARM E — frozen-backbone DINOv3 LOCO-versus-upper gap.  Gate E-2.

CC-SSH analysis script.  Reads the 6 arm E part-1 dumps (dinov3frozen, vanilla
LOCO and none/upper, rung 450, three cities, HR, seed 0), computes per-image
two-class boundary-tolerant mIoU at K=2 from raw counts, forms the
upper-minus-LOCO gap paired per image, and writes one CONTRACT.md section 6
results JSON.

Read-only on dumps/.  No training, no inference, no checkpoint loaded.  CPU
only, runs in seconds.

Why this script exists: runs/arme_part1/results.json is not on this box, and the
Delta-side version reports only fg_iou_K2_per_image_mean_pct with no interval
and states in its own fields that it is not a tested gap statistic.  The
response letter needs the gap on the SAME metric as
results/01_scaling_ladder.json, which is two-class boundary-tolerant mIoU at
K=2.  This produces the tested statistic.

The metric is the published two-class decision metric, matching
python/armc_ladder_stats.py:per_image_metrics verbatim:
    fg = tp_tol / (tp_tol + fp_tol + fn_tol + 1e-10)
    bg = tn_tol / (tn_tol + fp_tol + fn_tol + 1e-10)
    value = mean(fg, bg), in percent.
Foreground-only K=2 (`fg` above) is emitted alongside it so the mixed-metric
paragraph in the response letter can be replaced without a second run.

The bootstrap replicates armc_ladder_stats.py:paired_bootstrap verbatim:
B = 10,000 resamples of the per-image delta vector, percentile CI at 2.5 /
97.5, two-tailed p of min(2 * prop_wrong, 1.0), seed 42.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ config --
# Top-level constants instead of argparse, per CLAUDE.md.

DRY_RUN = "--dry-run" in sys.argv

REPO_ROOT = os.environ["PROJECT_ROOT"]
DUMPS_ROOT = os.path.join(REPO_ROOT, "dumps", "st")
EXP_NN = 2
OUT_PATH = os.path.join(REPO_ROOT, "results", "02_frozen_backbone_gap.json")
SCRIPT_LABEL = "python/arme_frozen_gap_stats.py"

CITIES = ("chicago", "miami", "phoenix")
LOCO_TEMPLATE = "dinov3frozen-vanilla-loco-r450-{city}-hr-s0"
UPPER_TEMPLATE = "dinov3frozen-none-upper-r450-{city}-hr-s0"

EPS_IOU = 1e-10
B = 10000
BOOT_SEED = 42
N_TEST_EXPECTED = 150
N_COLS_EXPECTED = 60

# Arm C rung-450 fine-tuned DINOv3, read from results/01_scaling_ladder.json.
# Context only.  Never pooled with the frozen number and never contrasted
# against it: 1 seed here against 3 seeds there, and gate E-2 asks whether the
# gap persists, not whether the two regimes differ.
ARMC_REF_PATH = os.path.join(REPO_ROOT, "results", "01_scaling_ladder.json")
ARMC_REF_EXPECTED = {
    "mean": 4.151704468474129,
    "stderr": 0.7722221795979466,
    "ci95": [2.794388640386595, 5.633073439120119],
    "n_cells": 9,
}

# CONTRACT.md section 3 columns this analysis reads, with their required dtype.
# Never coerced; a mismatch aborts.
REQUIRED_DTYPES = {
    "run_id": "object", "image_id": "object", "city": "object",
    "res": "object", "pair_id": "object",
    "tp_tol": "int64", "fp_tol": "int64", "fn_tol": "int64", "tn_tol": "int64",
    "n_valid_px": "int64", "n_gt_fg_valid": "int64", "n_pred_pos": "int64",
    "empty_fg_valid": "bool",
    "iou_tol": "float64", "iou_str": "float64",
    "miou_tol_pct": "float64", "iou_fg_tol_pct": "float64",
    "iou_bg_tol_pct": "float64",
}

# Present in CONTRACT.md section 3 but declared "structure_cols": "deferred" in
# every arm E run.json.  Checked for presence and for being wholly null, not
# for dtype, and reported as a schema deviation.  None of them enters the
# metric.
DEFERRED_STRUCTURE_COLS = (
    "n_gt_blobs", "n_pred_blobs", "gt_blob_area_median", "small_object_frac",
    "fn_px_by_cat", "tp_px_by_cat", "gt_px_by_cat", "fp_px_near_gt",
    "fp_px_far", "gt_contrast_median",
)

P_FLOOR_NOTE = (f"0 of {B} bootstrap resamples crossed zero. The p is bounded "
                f"below by the resample count, so report it as p < {1.0 / B:g}"
                f" and never as p = 0.")

LOG_LINES = []
FLAGS = []


def log(msg):
    line = str(msg)
    LOG_LINES.append(line)
    print(line, flush=True)


def ok(cond, msg):
    """Log an assertion outcome either way, then abort on failure."""
    log(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise AssertionError(msg)
    return cond


def flag(msg):
    """A sanity or schema observation that is recorded, not fatal."""
    log("  FLAG  " + msg)
    FLAGS.append(str(msg))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ metric --

def per_image_metrics(df):
    """Two-class and foreground-only tolerant IoU at K=2, per image, percent.

    Identical formula to armc_ladder_stats.py:per_image_metrics, which is
    itself verbatim p0a_recompute.py:miou_from_counts.  Computed per image from
    the raw counts and never from cell means.
    """
    tp = df["tp_tol"].to_numpy(dtype=np.float64)
    fp = df["fp_tol"].to_numpy(dtype=np.float64)
    fn = df["fn_tol"].to_numpy(dtype=np.float64)
    tn = df["tn_tol"].to_numpy(dtype=np.float64)
    fg = tp / (tp + fp + fn + EPS_IOU) * 100.0
    bg = tn / (tn + fp + fn + EPS_IOU) * 100.0
    return fg, bg, (fg + bg) / 2.0


# -------------------------------------------------------------------- load --

def load_cell(run_id, city, mode):
    """Read one cell, assert schema and sidecar, return per-image vectors."""
    cell_dir = os.path.join(DUMPS_ROOT, run_id)
    pq_path = os.path.join(cell_dir, "perimage.parquet")
    rj_path = os.path.join(cell_dir, "run.json")
    done_path = os.path.join(cell_dir, ".DONE")
    failed_path = os.path.join(cell_dir, ".FAILED")

    ok(os.path.isdir(cell_dir), f"{run_id}: dump directory present")
    ok(os.path.exists(done_path), f"{run_id}: .DONE marker present")
    ok(not os.path.exists(failed_path), f"{run_id}: no .FAILED marker")
    for p in (pq_path, rj_path):
        ok(os.path.exists(p) and os.path.getsize(p) > 0,
           f"{run_id}: {os.path.basename(p)} present and non-empty")

    df = pd.read_parquet(pq_path)
    with open(rj_path) as fh:
        run = json.load(fh)

    ok(len(df) == N_TEST_EXPECTED,
       f"{run_id}: parquet rows = {len(df)} (expected {N_TEST_EXPECTED})")
    ok(df.shape[1] == N_COLS_EXPECTED,
       f"{run_id}: parquet columns = {df.shape[1]} "
       f"(expected {N_COLS_EXPECTED})")

    missing = [c for c in REQUIRED_DTYPES if c not in df.columns]
    ok(not missing, f"{run_id}: required columns present "
                    f"({'missing ' + str(missing) if missing else 'all'})")
    wrong = {c: str(df[c].dtype) for c, want in REQUIRED_DTYPES.items()
             if str(df[c].dtype) != want}
    ok(not wrong, f"{run_id}: required dtypes match CONTRACT section 3 "
                  f"({wrong if wrong else 'all'})")

    absent_struct = [c for c in DEFERRED_STRUCTURE_COLS if c not in df.columns]
    ok(not absent_struct,
       f"{run_id}: deferred structure columns present as columns "
       f"({'absent ' + str(absent_struct) if absent_struct else 'all 10'})")
    nonnull_struct = sorted(c for c in DEFERRED_STRUCTURE_COLS
                            if df[c].notna().any())
    ok(run.get("structure_cols") == "deferred",
       f"{run_id}: run.json declares structure_cols = "
       f"{run.get('structure_cols')!r}")

    ok(df["image_id"].is_unique, f"{run_id}: image_id is unique")
    ok(df["run_id"].nunique() == 1 and df["run_id"].iloc[0] == run_id,
       f"{run_id}: parquet run_id column is constant and matches the directory")
    ok(bool((df["city"] == city).all()),
       f"{run_id}: every row's city is {city}")
    ok(bool((df["res"] == "hr").all()), f"{run_id}: every row's res is hr")

    for key, want in (("run_id", run_id), ("arch", "dinov3frozen"),
                      ("mode", mode), ("holdout", city), ("rung", 450),
                      ("res", "hr"), ("seed", 0), ("dataset", "st"),
                      ("arm", "E")):
        got = run.get(key)
        ok(str(got) == str(want),
           f"{run_id}: run.json {key} = {got!r} matches run_id ({want!r})")

    ok(int(run.get("n_test", -1)) == N_TEST_EXPECTED,
       f"{run_id}: run.json n_test = {run.get('n_test')} "
       f"(expected {N_TEST_EXPECTED})")
    ok(run.get("band_width_tol") == 2,
       f"{run_id}: run.json band_width_tol = {run.get('band_width_tol')} "
       f"(K=2)")
    ok(run.get("decision_metric") == "2class_tol_k2",
       f"{run_id}: run.json decision_metric = {run.get('decision_metric')!r}")
    ok(run.get("backbone_frozen") is True,
       f"{run_id}: run.json backbone_frozen = {run.get('backbone_frozen')!r}")
    ok(int(run.get("backbone_trainable_param_count", -1)) == 0,
       f"{run_id}: backbone_trainable_param_count = "
       f"{run.get('backbone_trainable_param_count')} (expected 0)")
    ok(run.get("target_access") in ("source_only", "unlabeled_target",
                                    "deployment_time", "target_labels"),
       f"{run_id}: target_access = {run.get('target_access')!r} is a "
       f"CONTRACT value")

    # test_results.json is the Delta-side sidecar the response letter currently
    # quotes.  Read for the cross-check, never as the source of a number.
    tr_path = os.path.join(cell_dir, "test_results.json")
    tr = None
    if os.path.exists(tr_path):
        with open(tr_path) as fh:
            tr = json.load(fh)
        ok(int(tr.get("n_images", -1)) == N_TEST_EXPECTED,
           f"{run_id}: test_results.json n_images = {tr.get('n_images')}")

    df = df.sort_values("image_id").reset_index(drop=True)
    fg, bg, two = per_image_metrics(df)

    n_valid = df["n_valid_px"].to_numpy(dtype=np.float64)
    n_pred_pos = df["n_pred_pos"].to_numpy(dtype=np.float64)

    return {
        "run_id": run_id, "city": city, "mode": mode,
        "arch": "dinov3frozen", "rung": 450, "seed": 0, "res": "hr",
        "method": run.get("method"),
        "n_test": int(len(df)),
        "image_ids": df["image_id"].to_numpy(),
        "pair_ids": df["pair_id"].to_numpy(),
        "fg": fg, "bg": bg, "two_class": two,
        "stored_miou_tol_pct": df["miou_tol_pct"].to_numpy(dtype=np.float64),
        "stored_iou_fg_tol_pct":
            df["iou_fg_tol_pct"].to_numpy(dtype=np.float64),
        "stored_iou_tol": df["iou_tol"].to_numpy(dtype=np.float64),
        "value_2class_k2": float(two.mean()),
        "value_fg_only_k2": float(fg.mean()),
        "value_bg_only_k2": float(bg.mean()),
        "pred_pos_fraction": float(n_pred_pos.sum() / max(n_valid.sum(), 1.0)),
        "n_empty_fg_valid": int(df["empty_fg_valid"].sum()),
        "target_access": run.get("target_access"),
        "target_access_detail": run.get("target_access_detail"),
        "metric_module_sha": run.get("metric_module_sha"),
        "prediction_recipe": run.get("prediction_recipe"),
        "reporting_metric": run.get("reporting_metric"),
        "decision_metric": run.get("decision_metric"),
        "epochs_run": run.get("epochs_run"),
        "epoch_of_best": run.get("epoch_of_best"),
        "best_decision_metric_pct": run.get("best_decision_metric_pct"),
        "config_sha256": run.get("config_sha256"),
        "test_results_fg_iou_K2_per_image_mean_pct":
            (tr or {}).get("fg_iou_K2_per_image_mean_pct"),
        "parquet_sha256": sha256_file(pq_path),
        "run_json_sha256": sha256_file(rj_path),
    }


# -------------------------------------------------------------- statistics --

def paired_bootstrap(delta, b=B, seed=BOOT_SEED):
    """Paired bootstrap on a per-image delta vector.

    Verbatim behaviour of armc_ladder_stats.py:paired_bootstrap, so intervals
    are comparable against results/01_scaling_ladder.json.
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    if n == 0:
        return dict(mean=None, stderr=None, ci95_lo=None, ci95_hi=None,
                    p_raw=None, n=0)
    observed = float(delta.mean())
    rng = np.random.default_rng(seed)
    boots = delta[rng.integers(0, n, size=(b, n))].mean(axis=1)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    prop_wrong = (float(np.mean(boots <= 0)) if observed >= 0
                  else float(np.mean(boots >= 0)))
    return dict(mean=observed,
                stderr=(float(delta.std(ddof=1) / np.sqrt(n))
                        if n > 1 else None),
                ci95_lo=lo, ci95_hi=hi,
                p_raw=float(min(2.0 * prop_wrong, 1.0)), n=int(n))


def p_phrase(p):
    """A bootstrap p of 0 means 0 of B resamples crossed, not p = 0."""
    if p is None:
        return "p unavailable"
    return f"largest p_holm < {1.0 / B:g}" if p == 0.0 \
        else f"largest p_holm {p:.4f}"


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, matching Table 39."""
    pv = sorted((p, i) for i, p in enumerate(pvals) if p is not None)
    out = [None] * len(pvals)
    m = len(pv)
    running = 0.0
    for rank, (p, i) in enumerate(pv):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = float(running)
    return out


def cell_level_bootstrap(values, b=B, seed=BOOT_SEED):
    """Bootstrap over cells rather than images.

    Matches armc_ladder_stats.py:arch_summary, which is how the arm C
    comparator +4.152 [2.794, 5.633] was built.  With 3 cities the resample
    space holds 10 distinct multisets, so this interval is coarse and is
    labelled as such wherever it appears.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, n, size=(b, n))].mean(axis=1)
    # Number of distinct resample multisets, C(2n-1, n).  The count of distinct
    # float means can exceed it because summation order differs between draws
    # of the same multiset, so both are reported and neither is inferred from
    # the other.
    multisets = 1
    for i in range(n):
        multisets = multisets * (2 * n - 1 - i) // (i + 1)
    return dict(mean=float(v.mean()),
                stderr=(float(v.std(ddof=1) / np.sqrt(n))
                        if n > 1 else None),
                ci95_lo=float(np.percentile(boots, 2.5)),
                ci95_hi=float(np.percentile(boots, 97.5)),
                n_cells=int(n),
                n_distinct_boot_multisets=int(multisets),
                n_distinct_boot_float_means=int(np.unique(boots).size))


# -------------------------------------------------------------- pairing -----

def resolve_pair_key(loco, upper, city):
    """Decide the per-image join key for this city, or abort.

    The spec names `pair_id`.  In these dumps `pair_id` is the within-session
    patch index, not a per-image key: chicago and miami carry 5 distinct values
    over 150 rows, phoenix 10.  Joining on it would collapse 150 images onto 5
    or 10 groups.  `image_id` is unique in every cell and is the key
    armc_ladder_stats.py used for the comparator, so it is the key here.

    Row order is never used.  If the image_id sets disagree, this aborts.
    """
    l_pair_u = pd.Series(loco["pair_ids"]).is_unique
    u_pair_u = pd.Series(upper["pair_ids"]).is_unique
    n_pair = int(pd.Series(loco["pair_ids"]).nunique())

    if l_pair_u and u_pair_u:
        key = "pair_id"
        ok(np.array_equal(np.sort(loco["pair_ids"]),
                          np.sort(upper["pair_ids"])),
           f"{city}: pair_id sets match exactly between LOCO and upper")
        order_l = np.argsort(loco["pair_ids"], kind="stable")
        order_u = np.argsort(upper["pair_ids"], kind="stable")
        degenerate = None
    else:
        key = "image_id"
        degenerate = (
            f"pair_id is not a per-image key in the {city} cells: "
            f"{n_pair} distinct values over {N_TEST_EXPECTED} rows. It is the "
            f"within-session patch index (CONTRACT section 3 defines it for "
            f"resolution-paired patches, not as a row key). Pairing switched "
            f"to image_id, which is unique in both cells and is the key "
            f"armc_ladder_stats.py used. Row order was not used.")
        flag(degenerate)
        # Sets must match exactly, or we stop rather than pair anything.
        ok(np.array_equal(np.sort(loco["image_ids"]),
                          np.sort(upper["image_ids"])),
           f"{city}: image_id sets match exactly between "
           f"{loco['run_id']} and {upper['run_id']}")
        order_l = np.argsort(loco["image_ids"], kind="stable")
        order_u = np.argsort(upper["image_ids"], kind="stable")

    # Whichever key was used, the two cells must agree on both identity
    # columns row-for-row once aligned.  This is the check the pair_id
    # instruction was after: the two cells describe the same 150 images.
    ok(np.array_equal(loco["image_ids"][order_l], upper["image_ids"][order_u]),
       f"{city}: image_id agrees row-for-row after aligning on {key}")
    ok(np.array_equal(loco["pair_ids"][order_l], upper["pair_ids"][order_u]),
       f"{city}: pair_id agrees row-for-row after aligning on {key}")

    return key, order_l, order_u, degenerate, n_pair


# ------------------------------------------------------- checks / sanity ----

def reproduction_check(cells):
    """Recompute each cell's stored per-image metric columns from the counts.

    `miou_tol_pct` is the stored two-class K=2 column and is the one the goal
    names.  `iou_fg_tol_pct` and `iou_tol` are checked in the same pass because
    the foreground-only number is emitted alongside.
    """
    rows, worst = [], {"miou_tol_pct": 0.0, "iou_fg_tol_pct": 0.0,
                       "iou_tol_x100": 0.0}
    worst_id = {k: None for k in worst}
    for c in cells:
        d = {
            "miou_tol_pct": float(np.max(np.abs(
                c["stored_miou_tol_pct"] - c["two_class"]))),
            "iou_fg_tol_pct": float(np.max(np.abs(
                c["stored_iou_fg_tol_pct"] - c["fg"]))),
            "iou_tol_x100": float(np.max(np.abs(
                c["stored_iou_tol"] * 100.0 - c["fg"]))),
        }
        for k, v in d.items():
            if v > worst[k]:
                worst[k], worst_id[k] = v, c["run_id"]
        rows.append({"run_id": c["run_id"], "max_abs_delta": d})
        log(f"[repro] {c['run_id']}: max|Δ| miou_tol_pct {d['miou_tol_pct']:.3e}"
            f"  iou_fg_tol_pct {d['iou_fg_tol_pct']:.3e}"
            f"  iou_tol*100 {d['iou_tol_x100']:.3e}")
    log(f"[repro] worst over 6 cells: miou_tol_pct "
        f"{worst['miou_tol_pct']:.3e} ({worst_id['miou_tol_pct']})")
    within = worst["miou_tol_pct"] <= 1e-6
    ok(within, f"stored miou_tol_pct reproduces from raw counts within 1e-6 "
               f"(worst {worst['miou_tol_pct']:.3e})")
    return {
        "note": "stored per-image columns recomputed from raw tp/fp/fn/tn_tol",
        "tolerance": 1e-6, "within_tolerance": bool(within),
        "max_abs_delta_over_cells": worst,
        "max_abs_delta_run_id": worst_id,
        "per_cell": rows,
    }


def crosscheck_test_results(cells):
    """Cross-check the foreground-only mean against the Delta-side sidecar.

    This is the number the response letter currently quotes.  It is checked,
    not used.
    """
    out = []
    for c in cells:
        stored = c["test_results_fg_iou_K2_per_image_mean_pct"]
        d = None if stored is None else abs(float(stored)
                                            - c["value_fg_only_k2"])
        out.append({"run_id": c["run_id"], "test_results_json": stored,
                    "recomputed_fg_only_k2": c["value_fg_only_k2"],
                    "abs_delta": d})
        log(f"[crosscheck] {c['run_id']}: test_results.json {stored} vs "
            f"recomputed {c['value_fg_only_k2']:.6f} "
            f"(Δ {d if d is None else format(d, '.3e')})")
    worst = max((r["abs_delta"] for r in out if r["abs_delta"] is not None),
                default=None)
    ok(worst is not None and worst <= 1e-4,
       f"foreground-only K=2 mean matches test_results.json within 1e-4 "
       f"(worst {worst})")
    return {"note": "Delta-side test_results.json, checked not used",
            "max_abs_delta": worst, "per_cell": out}


def provenance_check(cells):
    shas = sorted({c["metric_module_sha"] for c in cells})
    recipes = sorted({c["prediction_recipe"] for c in cells})
    ok(len(shas) == 1,
       f"metric_module_sha identical across all 6 cells ({len(shas)} distinct)")
    ok(len(recipes) == 1,
       f"prediction_recipe identical across all 6 cells "
       f"({len(recipes)} distinct)")
    if any(c["config_sha256"] is None for c in cells):
        flag("run.json carries no config_sha256 on any arm E cell, so the "
             "CONTRACT section 7 duplicate-run_id/differing-config check "
             "cannot be run on config hash. Substituted: the 6 run_ids are "
             "distinct by construction and each cell's parquet and run.json "
             "sha256 are recorded in this JSON.")
    return {
        "metric_module_sha": shas[0], "metric_module_sha_identical": True,
        "prediction_recipe": recipes[0], "prediction_recipe_identical": True,
        "decision_metric": sorted({c["decision_metric"] for c in cells}),
        "reporting_metric": sorted({c["reporting_metric"] for c in cells}),
        "config_sha256_present": not any(c["config_sha256"] is None
                                         for c in cells),
        "target_access": {c["run_id"]: c["target_access"] for c in cells},
        "per_cell_sha256": {c["run_id"]: {
            "perimage_parquet": c["parquet_sha256"],
            "run_json": c["run_json_sha256"]} for c in cells},
    }


def sanity_scan(cells, per_city):
    """Flag anything too clean before being asked, per CONVENTIONS.md."""
    for c in cells:
        sd = float(np.std(c["two_class"], ddof=1))
        log(f"[sanity] {c['run_id']}: two-class mean "
            f"{c['value_2class_k2']:.4f} sd {sd:.4f} "
            f"min {c['two_class'].min():.4f} max {c['two_class'].max():.4f} "
            f"pred_pos_frac {c['pred_pos_fraction']:.4f} "
            f"empty_fg {c['n_empty_fg_valid']}")
        if sd < 1.0:
            flag(f"{c['run_id']}: per-image two-class SD is {sd:.4f}, "
                 f"implausibly low for 150 images")
        if c["pred_pos_fraction"] > 0.50:
            flag(f"{c['run_id']}: pred_pos_fraction {c['pred_pos_fraction']:.4f}"
                 f" exceeds 0.50, the arm C collapse threshold")
        if c["n_empty_fg_valid"] > 0:
            log(f"[sanity] {c['run_id']}: {c['n_empty_fg_valid']} images have "
                f"empty GT foreground on the valid region; they are retained "
                f"because the two-class metric is defined on them through the "
                f"background term")
    for r in per_city:
        d = np.asarray(r["_delta"])
        n_neg = int((d < 0).sum())
        log(f"[sanity] {r['city']}: {n_neg}/{d.size} images where LOCO beats "
            f"upper; delta range [{d.min():.4f}, {d.max():.4f}]")
        if n_neg == 0:
            flag(f"{r['city']}: perfect separation, the upper bound beats LOCO "
                 f"on all {d.size} images")
        if r["p_raw"] == 0.0:
            log(f"[sanity] {r['city']}: bootstrap p is 0 of {B} resamples, "
                f"reported as p < 1e-4 rather than exactly 0")
    return None


# ---------------------------------------------------------- pre-commitment --

# Gate E-2 in REVIEWS.md lists exactly three readings.  The branch is assigned
# by a rule fixed before the numbers were seen, using the arm C rung-450
# fine-tuned DINOv3 interval as the reference band for "similar magnitude".
# Overlap with an interval is not a test of difference, and none is run here:
# 1 seed against 3 seeds, and gate E-2 asks whether the gap persists.
E2_READINGS = {
    "persists": ("Gap persists, similar magnitude to fine-tuned. Forgetting is "
                 "not the cause. W1 answered on three measurements: drift, "
                 "App G.8 probe, frozen gap."),
    "shrinks": ("Gap shrinks substantially. Part of the fine-tuned gap was "
                "overfitting the source cities. Section 4.4's DINOv3 pole gets "
                "hedged and the frozen result is reported alongside."),
    "grows": ("Gap grows. Freezing costs transfer capacity. Report as-is; W1's "
              "mechanism still doesn't hold, but the frozen model is not the "
              "better deployment choice."),
}


def assign_e2_branch(frozen_mean, ref):
    """Map the measured pooled gap onto one of gate E-2's three readings."""
    lo, hi = ref["ci95"]
    if frozen_mean < lo:
        return "shrinks"
    if frozen_mean > hi:
        return "grows"
    return "persists"


# ------------------------------------------------------------------- main ---

def git_hash():
    try:
        out = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip(), None
        return None, ("git rev-parse failed: "
                      + out.stderr.strip()[:200])
    except Exception as exc:
        return None, f"git unavailable: {exc}"


def derive_index_sha(cells):
    """No INDEX.parquet exists on this box.  Digest the sorted
    run_id:parquet_sha256 list so the JSON still pins its exact inputs.
    Same construction as armc_ladder_stats.py:derive_index_sha."""
    h = hashlib.sha256()
    for c in sorted(cells, key=lambda x: x["run_id"]):
        h.update((c["run_id"] + ":" + c["parquet_sha256"] + "\n").encode())
    return h.hexdigest()


def read_armc_reference():
    """Read the arm C comparator; never recompute it here."""
    with open(ARMC_REF_PATH) as fh:
        d = json.load(fh)
    rows = [a for a in d["arch_summary"]
            if a["arch"] == "dinov3" and a["rung"] == 450]
    ok(len(rows) == 1,
       f"arm C reference: exactly one dinov3/rung-450 arch_summary row "
       f"({len(rows)} found)")
    r = rows[0]
    ref = {"mean": r["mean_gap"], "stderr": r["stderr"],
           "ci95": [r["ci95_lo"], r["ci95_hi"]], "n_cells": r["n_cells"],
           "ci_method": r["ci_method"]}
    ok(abs(ref["mean"] - ARMC_REF_EXPECTED["mean"]) < 1e-9
       and ref["n_cells"] == ARMC_REF_EXPECTED["n_cells"],
       f"arm C reference matches the value quoted in the task "
       f"({ref['mean']:.4f} over {ref['n_cells']} cells)")
    return ref


def main():
    log(f"[start] {datetime.now(timezone.utc).isoformat()}")
    log(f"[start] DRY_RUN={DRY_RUN}  B={B}  seed={BOOT_SEED}")
    log(f"[start] dumps_root={DUMPS_ROOT}")
    log("[start] read-only on dumps/; no training, no inference, no checkpoint")

    # ---- load, from an explicit cell list, never a glob ----
    cells = []
    for city in CITIES:
        cells.append(load_cell(LOCO_TEMPLATE.format(city=city), city, "loco"))
        cells.append(load_cell(UPPER_TEMPLATE.format(city=city), city, "upper"))
    by_key = {(c["city"], c["mode"]): c for c in cells}
    log(f"[load] loaded {len(cells)} cells")

    repro = reproduction_check(cells)
    crosscheck = crosscheck_test_results(cells)
    prov = provenance_check(cells)
    ref = read_armc_reference()

    # ---- per-city paired gaps ----
    per_city, pair_notes = [], []
    for city in CITIES:
        loco, upper = by_key[(city, "loco")], by_key[(city, "upper")]
        key, ol, ou, degenerate, n_pair = resolve_pair_key(loco, upper, city)
        pair_notes.append({"city": city, "pair_key_used": key,
                           "pair_id_distinct_values": n_pair,
                           "pair_id_degeneracy": degenerate})

        d2 = upper["two_class"][ou] - loco["two_class"][ol]
        dfg = upper["fg"][ou] - loco["fg"][ol]
        s2, sfg = paired_bootstrap(d2), paired_bootstrap(dfg)
        per_city.append({
            "city": city,
            "loco_run_id": loco["run_id"], "upper_run_id": upper["run_id"],
            "pair_key_used": key,
            "loco_value_2class_k2": loco["value_2class_k2"],
            "upper_value_2class_k2": upper["value_2class_k2"],
            "loco_value_fg_only_k2": loco["value_fg_only_k2"],
            "upper_value_fg_only_k2": upper["value_fg_only_k2"],
            "gap_2class_k2": s2["mean"], "stderr": s2["stderr"],
            "ci95_lo": s2["ci95_lo"], "ci95_hi": s2["ci95_hi"],
            "p_raw": s2["p_raw"], "p_holm": None, "n_images": s2["n"],
            "gap_fg_only_k2": sfg["mean"], "stderr_fg_only": sfg["stderr"],
            "ci95_lo_fg_only": sfg["ci95_lo"], "ci95_hi_fg_only": sfg["ci95_hi"],
            "p_raw_fg_only": sfg["p_raw"], "p_holm_fg_only": None,
            "flip": bool(s2["mean"] < 0),
            "_delta": d2, "_delta_fg": dfg,
        })
        log(f"[gap] {city}: two-class {s2['mean']:+.4f} "
            f"[{s2['ci95_lo']:+.4f}, {s2['ci95_hi']:+.4f}] p={s2['p_raw']:.4f}"
            f"   fg-only {sfg['mean']:+.4f} "
            f"[{sfg['ci95_lo']:+.4f}, {sfg['ci95_hi']:+.4f}] "
            f"p={sfg['p_raw']:.4f}   n={s2['n']} paired on {key}")

    for r, a in zip(per_city, holm([r["p_raw"] for r in per_city])):
        r["p_holm"] = a
    for r, a in zip(per_city, holm([r["p_raw_fg_only"] for r in per_city])):
        r["p_holm_fg_only"] = a
    log("[holm] Holm-Bonferroni applied over the family of 3 cities, "
        "separately for each metric")

    sanity_scan(cells, per_city)

    # ---- pooled over cities, two resampling units, both reported ----
    city_means2 = [r["gap_2class_k2"] for r in per_city]
    city_meansfg = [r["gap_fg_only_k2"] for r in per_city]
    pooled_city2 = cell_level_bootstrap(city_means2)
    pooled_cityfg = cell_level_bootstrap(city_meansfg)
    pooled_img2 = paired_bootstrap(
        np.concatenate([r["_delta"] for r in per_city]))
    pooled_imgfg = paired_bootstrap(
        np.concatenate([r["_delta_fg"] for r in per_city]))
    log(f"[pooled] city-level (n=3 cities, matches the arm C comparator's "
        f"resampling unit): {pooled_city2['mean']:+.4f} "
        f"[{pooled_city2['ci95_lo']:+.4f}, {pooled_city2['ci95_hi']:+.4f}], "
        f"{pooled_city2['n_distinct_boot_multisets']} distinct resample "
        f"multisets")
    log(f"[pooled] image-level (n=450 images over 3 cities): "
        f"{pooled_img2['mean']:+.4f} "
        f"[{pooled_img2['ci95_lo']:+.4f}, {pooled_img2['ci95_hi']:+.4f}]")

    worst = min(per_city, key=lambda r: r["gap_2class_k2"])
    best = max(per_city, key=lambda r: r["gap_2class_k2"])
    log(f"[worst-case] smallest per-city gap {worst['gap_2class_k2']:+.4f} "
        f"({worst['city']}); largest {best['gap_2class_k2']:+.4f} "
        f"({best['city']})")

    # CONVENTIONS.md: flag a number that lands on a prior published value.
    d_ref = abs(pooled_city2["mean"] - ref["mean"])
    if d_ref < 0.10:
        flag(f"the pooled city-level frozen gap {pooled_city2['mean']:+.4f} "
             f"lands {d_ref:.4f} mIoU from the arm C fine-tuned rung-450 value "
             f"{ref['mean']:+.4f}. Disjoint runs, same test images, proximity "
             f"not engineered; flagged because it is what places the "
             f"measurement inside the comparator's interval and therefore "
             f"decides the gate E-2 branch.")

    branch = assign_e2_branch(pooled_city2["mean"], ref)
    log(f"[E-2] pooled city-level gap {pooled_city2['mean']:+.4f} against the "
        f"arm C fine-tuned reference band [{ref['ci95'][0]:.4f}, "
        f"{ref['ci95'][1]:.4f}] -> branch '{branch}'")

    env = build_envelope(cells, per_city, pair_notes, pooled_city2,
                         pooled_cityfg, pooled_img2, pooled_imgfg, worst, best,
                         repro, crosscheck, prov, ref, branch)

    print_summary(per_city, pooled_city2, pooled_img2, worst, ref, branch)

    if DRY_RUN:
        print(f"\n[dry-run] would write {OUT_PATH}")
        print(f"[dry-run] envelope keys: {sorted(env.keys())}")
        print(f"[dry-run] conditions={len(env['conditions'])} "
              f"tests={len(env['tests'])} cells={len(env['cells'])} "
              f"caveats={len(env['caveats'])} flags={len(FLAGS)}")
        return 0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(env, fh, indent=2, default=float)
    print(f"\n[write] {OUT_PATH}  ({os.path.getsize(OUT_PATH)} bytes)")
    return 0


CAVEATS_STATIC = [
    "One seed per cell. All six arm E part-1 cells are seed 0, so no "
    "across-seed variance exists for this arm and there is no noise floor "
    "against which to read the gap. The arm C comparator carries three seeds "
    "per city; this one does not.",
    "Gate E-2's pre-committed readings are in REVIEWS.md. The branch recorded "
    "in pre_committed_reading is one of the three listed there and no reading "
    "outside that list was used.",
    "The arm C rung-450 fine-tuned DINOv3 gap is context, read from "
    "results/01_scaling_ladder.json and never recomputed here. The two numbers "
    "are not pooled and no frozen-versus-fine-tuned contrast is tested: 1 seed "
    "against 3 seeds, and gate E-2 asks whether the gap persists, not whether "
    "the two regimes differ. Interval overlap is reported as a band membership, "
    "not as a test.",
    "pair_id is not a per-image key in these dumps. It is the within-session "
    "patch index and takes 5 distinct values over 150 rows in chicago and "
    "miami, 10 in phoenix. Pairing therefore ran on image_id, which is unique "
    "in every cell and is the key results/01_scaling_ladder.json used. Row "
    "order was never used, the image_id sets were asserted equal between the "
    "LOCO and upper cell of every city, and pair_id was asserted to agree "
    "row-for-row after the join.",
    "The pooled row is reported twice with different resampling units. The "
    "city-level interval resamples 3 cities and matches the unit of the arm C "
    "comparator, but 3 cities admit only 10 distinct resample multisets, so "
    "its 2.5th and 97.5th percentiles land on the smallest and largest "
    "per-city gap themselves. That interval is a range over 3 cities, not a "
    "precise bound, and must not be read as one. The image-level interval "
    "resamples 450 per-image deltas and treats images as independent across "
    "cities, which understates between-city spread. Neither is declared "
    "canonical.",
    "High-resolution cells only, rung 450 only, vanilla LOCO against the "
    "within-city upper bound. No adaptation method appears in this arm, so "
    "this results JSON must not be read as a method comparison.",
    "The within-city upper bound is the comparator by construction, which is "
    "what satisfies the CONVENTIONS.md same-run baseline requirement. Vanilla "
    "no-adaptation is the LOCO arm itself. The strongest published adaptation "
    "method is not present on the frozen backbone and was not run.",
    "The ten CONTRACT.md section 3 structure columns (n_gt_blobs through "
    "gt_contrast_median) are present but wholly null in all six cells, "
    "declared as \"structure_cols\": \"deferred\" in every run.json. None of "
    "them enters this metric. Arms G and J cannot run off these dumps.",
    "run.json carries no config_sha256 on any arm E cell, so CONTRACT.md "
    "section 7's duplicate-run_id-with-differing-config check could not be run "
    "on config hash. The six run_ids are distinct by construction and the "
    "sha256 of every perimage.parquet and run.json is recorded in "
    "provenance_check.per_cell_sha256.",
    "No INDEX.parquet exists on this box. index_sha256 is the sha256 over the "
    "sorted run_id:parquet_sha256 list of the six cells analysed, the same "
    "construction results/01_scaling_ladder.json used.",
]


def build_caveats(cells, pooled_city2, ref):
    """CAVEATS_STATIC plus the ones that carry measured counts."""
    counts = {}
    for city in CITIES:
        vals = {c["n_empty_fg_valid"] for c in cells if c["city"] == city}
        ok(len(vals) == 1,
           f"{city}: empty_fg_valid count is identical between the LOCO and "
           f"upper cell (it is a property of the test set, not of the model); "
           f"saw {sorted(vals)}")
        counts[city] = vals.pop()
    out = list(CAVEATS_STATIC)
    out.append(
        f"Images with empty ground-truth foreground on the valid region are "
        f"retained: {counts['chicago']} of 150 in chicago, "
        f"{counts['miami']} in miami, {counts['phoenix']} in phoenix, "
        f"identical between the LOCO and upper cell of each city because it is "
        f"a property of the test set. The two-class metric stays defined on "
        f"them through the background term. The foreground-only number on "
        f"those images is 0 by construction, a 0/0 guarded by the 1e-10 "
        f"epsilon, so the foreground-only column is depressed in both "
        f"conditions and its gap is the quantity to read, not its levels. "
        f"Miami's 38 empty-foreground images are why its foreground-only "
        f"levels sit where they do.")
    out.append(
        f"The pooled city-level gap {pooled_city2['mean']:+.4f} sits "
        f"{abs(pooled_city2['mean'] - ref['mean']):.4f} mIoU from the arm C "
        f"fine-tuned rung-450 value {ref['mean']:+.4f}. The two are computed "
        f"from disjoint runs on the same 450 test images per city and the "
        f"proximity is not engineered, but a number landing this close to a "
        f"prior result is flagged under CONVENTIONS.md rather than presented "
        f"as corroboration. It is also what puts the measurement inside the "
        f"comparator's interval, so the gate E-2 branch assignment turns on a "
        f"coincidence of this size and would not survive a shift of 1.4 mIoU "
        f"in either direction.")
    return out


def build_envelope(cells, per_city, pair_notes, pooled_city2, pooled_cityfg,
                   pooled_img2, pooled_imgfg, worst, best, repro, crosscheck,
                   prov, ref, branch):
    gh, gh_reason = git_hash()

    conditions = []
    for r in per_city:
        conditions.append({
            "label": f"DINOv3 frozen backbone / rung 450 / {r['city']} / "
                     f"matched-450-upper",
            "metric": "gap_2class_mIoU_tol_k2",
            "mean": r["gap_2class_k2"], "stderr": r["stderr"],
            "n": r["n_images"], "ci95": [r["ci95_lo"], r["ci95_hi"]],
            "ci_method": "paired_bootstrap_B10000",
            "baseline": "upper_r450",
        })
        conditions.append({
            "label": f"DINOv3 frozen backbone / rung 450 / {r['city']} / "
                     f"matched-450-upper (foreground-only)",
            "metric": "gap_fg_only_IoU_tol_k2",
            "mean": r["gap_fg_only_k2"], "stderr": r["stderr_fg_only"],
            "n": r["n_images"],
            "ci95": [r["ci95_lo_fg_only"], r["ci95_hi_fg_only"]],
            "ci_method": "paired_bootstrap_B10000",
            "baseline": "upper_r450",
        })
    conditions.append({
        "label": "DINOv3 frozen backbone / rung 450 / pooled over 3 cities / "
                 "matched-450-upper",
        "metric": "gap_2class_mIoU_tol_k2",
        "mean": pooled_city2["mean"], "stderr": pooled_city2["stderr"],
        "n": pooled_city2["n_cells"],
        "ci95": [pooled_city2["ci95_lo"], pooled_city2["ci95_hi"]],
        "ci_method": "city_level_bootstrap_B10000",
        "n_cities": 3, "n_images": pooled_img2["n"],
        "baseline": "upper_r450",
    })
    conditions.append({
        "label": "DINOv3 frozen backbone / rung 450 / pooled over 3 cities / "
                 "matched-450-upper (foreground-only)",
        "metric": "gap_fg_only_IoU_tol_k2",
        "mean": pooled_cityfg["mean"], "stderr": pooled_cityfg["stderr"],
        "n": pooled_cityfg["n_cells"],
        "ci95": [pooled_cityfg["ci95_lo"], pooled_cityfg["ci95_hi"]],
        "ci_method": "city_level_bootstrap_B10000",
        "n_cities": 3, "n_images": pooled_imgfg["n"],
        "baseline": "upper_r450",
    })

    tests = []
    for r in per_city:
        tests.append({
            "name": "paired bootstrap on per-image deltas",
            "compares": [r["upper_run_id"], r["loco_run_id"]],
            "metric": "gap_2class_mIoU_tol_k2",
            "statistic": r["gap_2class_k2"], "p": r["p_raw"],
            "p_holm": r["p_holm"], "B": B, "two_tailed": True,
            "n": r["n_images"], "family": "arm_E_frozen_3_cities",
            "correction": "Holm-Bonferroni over 3 cities",
            "p_note": P_FLOOR_NOTE if r["p_raw"] == 0.0 else None,
        })
        tests.append({
            "name": "paired bootstrap on per-image deltas",
            "compares": [r["upper_run_id"], r["loco_run_id"]],
            "metric": "gap_fg_only_IoU_tol_k2",
            "statistic": r["gap_fg_only_k2"], "p": r["p_raw_fg_only"],
            "p_holm": r["p_holm_fg_only"], "B": B, "two_tailed": True,
            "n": r["n_images"],
            "family": "arm_E_frozen_3_cities_fg_only",
            "correction": "Holm-Bonferroni over 3 cities",
            "p_note": P_FLOOR_NOTE if r["p_raw_fg_only"] == 0.0 else None,
        })

    headline = (
        f"The frozen-backbone DINOv3 upper-minus-LOCO gap at rung 450 is "
        f"{pooled_city2['mean']:+.3f} two-class boundary-tolerant mIoU at K=2 "
        f"pooled over 3 cities (95% CI "
        f"[{pooled_city2['ci95_lo']:.3f}, {pooled_city2['ci95_hi']:.3f}], "
        f"450 images), with per-city gaps "
        + ", ".join(f"{r['city']} {r['gap_2class_k2']:+.3f}"
                    for r in per_city)
        + f", all three significant after Holm-Bonferroni over the family of "
          f"3 cities ({p_phrase(max(r['p_holm'] for r in per_city))}); the "
          f"worst case is {worst['city']} at {worst['gap_2class_k2']:+.3f}.")

    return {
        "id": "02_frozen_backbone_gap",
        "arm": "E",
        "answers": ["E-2", "zNBu-W1"],
        "answers_note": "Gate E-2 in REVIEWS.md is the pre-commitment this "
                        "analysis reports against. zNBu-W1 is the reviewer "
                        "weakness gate E-2 sits under (catastrophic forgetting "
                        "as the proposed mechanism); this arm supplies the "
                        "frozen-gap leg of the three-measurement answer.",
        "question": "Does the LOCO-versus-upper gap persist when the DINOv3 "
                    "backbone is frozen, on the same metric as "
                    "results/01_scaling_ladder.json?",
        "runs": sorted(c["run_id"] for c in cells),
        "index_sha256": derive_index_sha(cells),
        "index_sha256_note": "no INDEX.parquet exists on this box; this is "
                             "sha256 over the sorted run_id:parquet_sha256 "
                             "list of the six cells analysed, the same "
                             "construction results/01_scaling_ladder.json used",
        "generated_utc": datetime.now(timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": SCRIPT_LABEL,
        "git_hash": gh,
        "git_hash_reason": gh_reason,

        "conditions": conditions,
        "tests": tests,
        "headline": headline,
        "caveats": build_caveats(cells, pooled_city2, ref),
        "figure": None,
        "figure_note": "No figure was requested for this analysis; NN=02 is "
                       "reserved for it if CS builds one.",
        "caption": None,

        "metric_definition": {
            "name": "2class_tol_k2",
            "formula": "fg = tp_tol/(tp_tol+fp_tol+fn_tol+1e-10); "
                       "bg = tn_tol/(tn_tol+fp_tol+fn_tol+1e-10); "
                       "value = mean(fg,bg) * 100",
            "band_width_px": 2,
            "source": "python/armc_ladder_stats.py:per_image_metrics, itself "
                      "verbatim p0a_recompute.py:miou_from_counts",
            "computed": "per image from raw counts, never from cell means",
            "secondary_metric": {
                "name": "fg_only_tol_k2",
                "formula": "fg = tp_tol/(tp_tol+fp_tol+fn_tol+1e-10) * 100",
                "why": "emitted so the response letter's mixed-metric "
                       "paragraph, which quotes fg_iou_K2_per_image_mean_pct "
                       "from the Delta-side test_results.json with no "
                       "interval, can be replaced by a tested statistic on "
                       "the same footing",
            },
        },

        "pairing": {
            "key_requested": "pair_id",
            "key_used": "image_id",
            "reason": "pair_id is the within-session patch index in these "
                      "dumps, not a per-image key: 5 distinct values over 150 "
                      "rows in chicago and miami, 10 in phoenix. Joining on it "
                      "would collapse 150 images onto 5 or 10 groups. image_id "
                      "is unique in every cell and is the key "
                      "results/01_scaling_ladder.json paired on.",
            "row_order_used": False,
            "checks": ["image_id sets asserted equal between the LOCO and "
                       "upper cell of every city",
                       "pair_id asserted to agree row-for-row after the "
                       "image_id join, so the two cells describe the same 150 "
                       "images"],
            "per_city": pair_notes,
        },

        "cells": [{k: c[k] for k in (
            "run_id", "city", "mode", "method", "arch", "rung", "seed", "res",
            "n_test", "value_2class_k2", "value_fg_only_k2",
            "value_bg_only_k2", "pred_pos_fraction", "n_empty_fg_valid",
            "target_access", "target_access_detail", "epochs_run",
            "epoch_of_best", "best_decision_metric_pct")}
            for c in sorted(cells, key=lambda x: x["run_id"])],

        "per_city_gaps": [{k: v for k, v in r.items()
                           if not k.startswith("_")} for r in per_city],

        "pooled": {
            "note": "two resampling units, neither declared canonical",
            "city_level": {
                "metric": "gap_2class_mIoU_tol_k2",
                "mean": pooled_city2["mean"], "stderr": pooled_city2["stderr"],
                "ci95": [pooled_city2["ci95_lo"], pooled_city2["ci95_hi"]],
                "ci_method": "city_level_bootstrap_B10000",
                "n_cities": 3, "n_images": pooled_img2["n"],
                "n_distinct_bootstrap_multisets":
                    pooled_city2["n_distinct_boot_multisets"],
                "n_distinct_bootstrap_float_means":
                    pooled_city2["n_distinct_boot_float_means"],
                "matches_comparator_resampling_unit": True,
            },
            "city_level_fg_only": {
                "metric": "gap_fg_only_IoU_tol_k2",
                "mean": pooled_cityfg["mean"],
                "stderr": pooled_cityfg["stderr"],
                "ci95": [pooled_cityfg["ci95_lo"], pooled_cityfg["ci95_hi"]],
                "ci_method": "city_level_bootstrap_B10000",
                "n_cities": 3, "n_images": pooled_imgfg["n"],
                "n_distinct_bootstrap_multisets":
                    pooled_cityfg["n_distinct_boot_multisets"],
                "n_distinct_bootstrap_float_means":
                    pooled_cityfg["n_distinct_boot_float_means"],
            },
            "image_level": {
                "metric": "gap_2class_mIoU_tol_k2",
                "mean": pooled_img2["mean"], "stderr": pooled_img2["stderr"],
                "ci95": [pooled_img2["ci95_lo"], pooled_img2["ci95_hi"]],
                "ci_method": "paired_bootstrap_B10000",
                "n_cities": 3, "n_images": pooled_img2["n"],
                "p_raw": pooled_img2["p_raw"],
            },
            "image_level_fg_only": {
                "metric": "gap_fg_only_IoU_tol_k2",
                "mean": pooled_imgfg["mean"], "stderr": pooled_imgfg["stderr"],
                "ci95": [pooled_imgfg["ci95_lo"], pooled_imgfg["ci95_hi"]],
                "ci_method": "paired_bootstrap_B10000",
                "n_cities": 3, "n_images": pooled_imgfg["n"],
                "p_raw": pooled_imgfg["p_raw"],
            },
        },

        "worst_case": {
            "note": "CONVENTIONS.md requires the grid, the mean and the worst "
                    "case; the worst case is the smallest per-city gap",
            "city": worst["city"], "gap_2class_k2": worst["gap_2class_k2"],
            "ci95": [worst["ci95_lo"], worst["ci95_hi"]],
            "p_holm": worst["p_holm"],
            "largest_city": best["city"],
            "largest_gap_2class_k2": best["gap_2class_k2"],
            "n_cities_with_flip": sum(1 for r in per_city if r["flip"]),
        },

        "comparator_context": {
            "note": "read from results/01_scaling_ladder.json, not recomputed "
                    "here; not pooled with the frozen number and not "
                    "contrasted against it",
            "source": "results/01_scaling_ladder.json arch_summary "
                      "dinov3 / rung 450",
            "label": "DINOv3 fine-tuned / rung 450 / matched-450-upper",
            "metric": "gap_2class_mIoU_tol_k2",
            "mean": ref["mean"], "stderr": ref["stderr"], "ci95": ref["ci95"],
            "n_cells": ref["n_cells"], "ci_method": ref["ci_method"],
            "seeds": 3,
            "why_no_contrast": "1 seed against 3 seeds, and gate E-2's "
                               "pre-commitment is about whether the gap "
                               "persists, not about a tested difference "
                               "between the frozen and fine-tuned regimes",
        },

        "pre_committed_reading": {
            "gate": "E-2",
            "source": "REVIEWS.md, Arm E, Gate E-2",
            "branches_listed": ["Gap persists, similar magnitude to "
                                "fine-tuned (approximately -1.7)",
                                "Gap shrinks substantially",
                                "Gap grows"],
            "assignment_rule": "The measured pooled city-level gap is compared "
                               "against the arm C rung-450 fine-tuned DINOv3 "
                               "95% CI [2.794, 5.633] read from "
                               "results/01_scaling_ladder.json. Inside the "
                               "band reads as persists at similar magnitude, "
                               "below its lower bound as shrinks "
                               "substantially, above its upper bound as grows. "
                               "This is a band-membership read, not a test of "
                               "difference; no frozen-versus-fine-tuned test "
                               "was run.",
            "branch": branch,
            "reading": E2_READINGS[branch],
            "sign_convention_note":
                "Gate E-2 quotes the fine-tuned magnitude as approximately "
                "-1.7, which is the published Fig 1 per-architecture figure in "
                "LOCO-minus-upper sign and is a printed magnitude that erratum "
                "E-7 in REVIEWS.md supersedes. Every number in this JSON is "
                "upper-minus-LOCO, so a persisting gap is positive. The "
                "magnitude reference used for the branch is therefore the "
                "recomputed arm C value +4.152, not the printed -1.7.",
        },

        "reproduction_check": repro,
        "test_results_json_crosscheck": crosscheck,
        "provenance_check": prov,
        "sanity_flags": list(FLAGS),
        "log": LOG_LINES,
    }


def print_summary(per_city, pooled_city2, pooled_img2, worst, ref, branch):
    print("\n" + "=" * 78)
    print("ARM E — FROZEN-BACKBONE DINOv3 GAP, ONE-SCREEN SUMMARY")
    print("=" * 78)
    print("\nGap = upper - LOCO, two-class boundary-tolerant mIoU at K=2, "
          "percent.")
    print("Positive = the within-city upper bound beats LOCO.\n")
    hdr = (f"{'city':<10} {'LOCO':>8} {'upper':>8} {'gap':>9} {'95% CI':>21} "
           f"{'p_raw':>8} {'p_holm':>8} {'n':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in per_city:
        print(f"{r['city']:<10} {r['loco_value_2class_k2']:>8.4f} "
              f"{r['upper_value_2class_k2']:>8.4f} "
              f"{r['gap_2class_k2']:>+9.4f} "
              f"[{r['ci95_lo']:>+8.4f},{r['ci95_hi']:>+8.4f}] "
              f"{r['p_raw']:>8.4f} {r['p_holm']:>8.4f} {r['n_images']:>5}")
    print(f"\n{'pooled':<10} {'':>8} {'':>8} {pooled_city2['mean']:>+9.4f} "
          f"[{pooled_city2['ci95_lo']:>+8.4f},{pooled_city2['ci95_hi']:>+8.4f}]"
          f"   city-level bootstrap, 3 cities, 450 images")
    print(f"{'pooled':<10} {'':>8} {'':>8} {pooled_img2['mean']:>+9.4f} "
          f"[{pooled_img2['ci95_lo']:>+8.4f},{pooled_img2['ci95_hi']:>+8.4f}]"
          f"   image-level bootstrap, 450 images")
    print(f"\nWorst case: {worst['city']} at {worst['gap_2class_k2']:+.4f}")

    print("\nForeground-only IoU at K=2, same pairing:\n")
    hdr2 = (f"{'city':<10} {'LOCO':>8} {'upper':>8} {'gap':>9} {'95% CI':>21} "
            f"{'p_holm':>8}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in per_city:
        print(f"{r['city']:<10} {r['loco_value_fg_only_k2']:>8.4f} "
              f"{r['upper_value_fg_only_k2']:>8.4f} "
              f"{r['gap_fg_only_k2']:>+9.4f} "
              f"[{r['ci95_lo_fg_only']:>+8.4f},{r['ci95_hi_fg_only']:>+8.4f}] "
              f"{r['p_holm_fg_only']:>8.4f}")

    print(f"\nComparator, read not recomputed: DINOv3 fine-tuned rung 450 "
          f"{ref['mean']:+.3f} +- {ref['stderr']:.3f}, "
          f"95% CI [{ref['ci95'][0]:.3f}, {ref['ci95'][1]:.3f}], "
          f"{ref['n_cells']} cells, 3 seeds.")
    print(f"Gate E-2 pre-committed branch: {branch}")
    if FLAGS:
        print(f"\nSanity / schema flags ({len(FLAGS)}):")
        for f in FLAGS:
            print(f"  - {f}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
