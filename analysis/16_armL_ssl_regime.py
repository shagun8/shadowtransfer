#!/usr/bin/env python3
"""ARM L — SSL CNN under DINOv3's fine-tuning regime.  Answers zNBu-W5 (primary),
zNBu-W1, AC1, AC2.

CC-SSH reanalysis.  Reads the arm L dumps already on this box (72 training cells
+ 216 probe dumps), recomputes every number from per-image values, and writes
results/16_armL_ssl_regime.json plus results/16_armL_summary.md.

No GPU, no network, no retraining, no writes under dumps/.

Metric.  There is no metrics.py on this box, so the per-image metric is
recomputed from the raw tp/fp/fn/tn counts using the formula of
python/armc_ladder_stats.py:per_image_metrics (which
python/arme_frozen_gap_stats.py:138 documents as identical, and which is itself
verbatim p0a_recompute.py:miou_from_counts):

    fg = tp_tol / (tp_tol + fp_tol + fn_tol + 1e-10)
    bg = tn_tol / (tn_tol + fp_tol + fn_tol + 1e-10)
    two_class = (fg + bg) / 2

Both K=2 definitions are carried everywhere.  PRIMARY is two-class K=2 (the
Table 38 / arm C family).  Foreground-only K=2 is reported alongside; it is the
definition the dumps' own stored recovery ratios were computed on, which makes
the stored values an exact cross-check on this reimplementation.

Stages, set by STAGE below:
    'assert'  enumerate and validate the analysis set, write nothing
    'dry'     'assert' + the full analysis restricted to DRY_CORNERS, print,
              write nothing
    'full'    the whole 2x2, write both output files
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ config --

BASE = os.environ["PROJECT_ROOT"]
DUMPS_ROOT = os.path.join(BASE, "dumps", "st")
OUT_JSON = os.path.join(BASE, "results", "16_armL_ssl_regime.json")
OUT_MD = os.path.join(BASE, "results", "16_armL_summary.md")
SCRIPT_LABEL = "analysis/16_armL_ssl_regime.py"

STAGE = "dry" if "--dry-run" in sys.argv else "full"          # 'assert' | 'dry' | 'full'
DRY_CORNERS = ("sup_hi",)  # used by STAGE == 'dry'

B = 10000
BOOT_SEED = 42
EPS_IOU = 1e-10

ARCH = "mamnetr50"
RUNG = 450
RES = "hr"
CORNERS = ("sup_hi", "sup_lo", "ssl_hi", "ssl_lo")
CITIES = ("chicago", "miami", "phoenix")
SEEDS = (0, 1, 2)
MODES = ("loco", "upper")
PROBES = ("p1histmatch", "p2cityprobe", "p3expA", "p3expB", "p3selpred",
          "p5drift")

N_TEST_EXPECTED = 150
N_TRAIN_CELLS_EXPECTED = 72
N_PROBES_PER_FAMILY_EXPECTED = 36

# Both paths carry .FAILED and no data.  They are parent-directory markers left
# by probe job 20714436, which crashed on a deterministic n_perm=0 bug in
# armc_probes_ops.py, was fixed (commit e46888d) and resubmitted as 20715046.
# That cell's six sub-probes are all present and .DONE.  Excluded by exact path;
# never deleted, because they are accurate history.
EXCLUDED_PATHS = (
    "dumps/st/mamnetr50-sup_hi-loco-r450-chicago-hr-s0-probes",
    "dumps/st/mamnetr50-sup_hi-probes-r450-chicago-hr-s0",
)
EXCLUDED_REASON = (
    "empty .FAILED parent-directory marker from crashed probe job 20714436 "
    "(deterministic n_perm=0 bug in armc_probes_ops.py, fixed in commit "
    "e46888d, resubmitted as 20715046); the cell's six sub-probes are all "
    "present and .DONE"
)

# R's exclusion rule, on the IoU fraction scale (0-1), matching the rule the
# dumps recorded: 'headroom (upper - LOCO) < 0.02'.
HEADROOM_MIN = 0.02

# G1: the pre-committed CNN-like threshold on Experiment B median R.
G1_THRESHOLD = 0.70
PUBLISHED_EXPB = {
    "mamnet_r34": {"median_R": 0.88, "ci95": [0.83, 0.90]},
    "oglanet": {"median_R": 0.86, "ci95": [0.81, 0.89]},
    "dinov3": {"median_R": -0.13, "ci95": [-0.27, -0.03]},
}
# G3: 'a HIGH R under a binding budget is VALID; a LOW R under a binding budget
# is AMBIGUOUS'.  HIGH is the same CNN-like threshold the arm's signature table
# uses, so the readable rule and the outcome table share one number.
G3_HIGH_R = G1_THRESHOLD
G3_HIGH_R_STRICT = 0.30   # sensitivity: the DINOv3-like boundary

# Table 38, published MAMNet-ResNet-34 vanilla high-res, two-class tolerant
# K=2 (the family E-7 verified reproduces at K=2).  Carried verbatim from the
# arm spec; this box does not recompute the published R34 cells.
PUBLISHED_R34_TABLE38 = {
    "chicago": {"loco": 79.72, "upper": 86.31},
    "miami": {"loco": 74.64, "upper": 79.11},
    "phoenix": {"loco": 62.84, "upper": 73.75},
}

# The one number CC-SLURM flagged, carried so the JSON states what was expected.
FLAGGED_CELL = "mamnetr50-ssl_lo-loco-r450-miami-hr-s2"

# Reported per city as a fraction of the 150 test images; identical across all
# 24 cells of each city, which this script asserts rather than assumes.
EXPECTED_EMPTY_FG = {"chicago": 11, "miami": 38, "phoenix": 21}

METRICS = ("2class_tol_k2", "fg_only_tol_k2")
PRIMARY_METRIC = "2class_tol_k2"

REQUIRED_COLS = ("run_id", "image_id", "city", "res", "tp_tol", "fp_tol",
                 "fn_tol", "tn_tol", "n_valid_px", "n_gt_fg_valid",
                 "empty_fg_valid", "iou_tol", "n_pred_pos",
                 "aurc_fg", "aurc_bg", "aurc_fg_mean20", "aurc_bg_mean20")
INT_COLS = ("tp_tol", "fp_tol", "fn_tol", "tn_tol", "n_valid_px",
            "n_gt_fg_valid", "n_pred_pos")

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
    """A sanity observation that is recorded, not fatal."""
    log("  FLAG  " + msg)
    FLAGS.append(str(msg))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(path):
    return sha256_file(path) if os.path.exists(path) else None


# ------------------------------------------------------------------ metric --

def per_image_metrics(df):
    """Two-class and foreground-only tolerant IoU at K=2, per image, FRACTION.

    Formula verbatim from armc_ladder_stats.py:per_image_metrics (documented as
    identical at arme_frozen_gap_stats.py:138), except that that script scales
    by 100 at the end.  Here the fraction is returned and the scaling to percent
    happens only where a level or a gap is reported, so that R's headroom rule
    keeps the 0.02 threshold the dumps recorded.
    """
    tp = df["tp_tol"].to_numpy(dtype=np.float64)
    fp = df["fp_tol"].to_numpy(dtype=np.float64)
    fn = df["fn_tol"].to_numpy(dtype=np.float64)
    tn = df["tn_tol"].to_numpy(dtype=np.float64)
    fg = tp / (tp + fp + fn + EPS_IOU)
    bg = tn / (tn + fp + fn + EPS_IOU)
    return {"fg_only_tol_k2": fg, "2class_tol_k2": (fg + bg) / 2.0}


# --------------------------------------------------------------- discovery --

def run_id_of(corner, mode, city, seed):
    return f"{ARCH}-{corner}-{mode}-r{RUNG}-{city}-{RES}-s{seed}"


def enumerate_analysis_set():
    """Build the expected analysis set by name and validate it against disk.

    Enumerating the expected 288 names and checking each one is what makes a
    half-copied or stray directory visible: an extra mamnetr50 directory on disk
    that is not in the expected set is reported, not silently ignored.
    """
    training, probes = [], []
    for corner in CORNERS:
        for city in CITIES:
            for seed in SEEDS:
                for mode in MODES:
                    training.append({
                        "run_id": run_id_of(corner, mode, city, seed),
                        "corner": corner, "mode": mode, "city": city,
                        "seed": seed})
                loco = run_id_of(corner, "loco", city, seed)
                for probe in PROBES:
                    probes.append({"dir": f"{loco}-{probe}", "probe": probe,
                                   "parent": loco, "corner": corner,
                                   "city": city, "seed": seed})
    return training, probes


def assert_analysis_set():
    log("=" * 78)
    log("STAGE 1 — analysis-set assertion")
    log("=" * 78)
    log(f"[set] dumps root = {DUMPS_ROOT}")

    on_disk = sorted(d for d in os.listdir(DUMPS_ROOT)
                     if os.path.isdir(os.path.join(DUMPS_ROOT, d))
                     and d.startswith(ARCH + "-"))
    log(f"[set] directories on disk starting '{ARCH}-': {len(on_disk)}")

    excluded_names = {p.split("/")[-1] for p in EXCLUDED_PATHS}
    for name in sorted(excluded_names):
        path = os.path.join(DUMPS_ROOT, name)
        present = os.path.isdir(path)
        ok(present, f"excluded path present on disk (history preserved): {name}")
        contents = sorted(os.listdir(path)) if present else []
        has_failed = ".FAILED" in contents
        data = [c for c in contents if c not in (".FAILED",)]
        ok(has_failed, f"excluded {name}: carries .FAILED")
        ok(not data, f"excluded {name}: carries no data (contents {contents})")

    training, probes = enumerate_analysis_set()
    expected = {t["run_id"] for t in training} | {p["dir"] for p in probes}
    ok(len(expected) == N_TRAIN_CELLS_EXPECTED + 6 * N_PROBES_PER_FAMILY_EXPECTED,
       f"expected analysis set enumerates to "
       f"{N_TRAIN_CELLS_EXPECTED + 216} names (got {len(expected)})")

    stray = sorted(set(on_disk) - expected - excluded_names)
    ok(not stray, f"no unexpected {ARCH} directory on disk beyond the analysis "
                  f"set and the two excluded paths (stray: {stray})")
    missing = sorted(expected - set(on_disk))
    ok(not missing, f"every expected directory is on disk "
                    f"(missing: {len(missing)}{'' if not missing else ': ' + str(missing[:6])})")

    # 72 training run.json, and their perimage.parquet alongside.
    n_train_ok = 0
    for t in training:
        d = os.path.join(DUMPS_ROOT, t["run_id"])
        for fn in ("run.json", "perimage.parquet"):
            p = os.path.join(d, fn)
            if not (os.path.exists(p) and os.path.getsize(p) > 0):
                ok(False, f"{t['run_id']}: {fn} present and non-empty")
        n_train_ok += 1
    ok(n_train_ok == N_TRAIN_CELLS_EXPECTED,
       f"training cells with run.json + perimage.parquet: {n_train_ok} "
       f"(expected {N_TRAIN_CELLS_EXPECTED})")

    # 36 of each probe family, each with run.json.
    by_family = defaultdict(int)
    for p in probes:
        d = os.path.join(DUMPS_ROOT, p["dir"])
        rj = os.path.join(d, "run.json")
        if not (os.path.exists(rj) and os.path.getsize(rj) > 0):
            ok(False, f"{p['dir']}: run.json present and non-empty")
        by_family[p["probe"]] += 1
    for probe in PROBES:
        ok(by_family[probe] == N_PROBES_PER_FAMILY_EXPECTED,
           f"probe family {probe}: {by_family[probe]} dumps "
           f"(expected {N_PROBES_PER_FAMILY_EXPECTED})")

    # Zero .FAILED anywhere inside the analysis set; .DONE on every member.
    failed, not_done = [], []
    for name in sorted(expected):
        d = os.path.join(DUMPS_ROOT, name)
        if os.path.exists(os.path.join(d, ".FAILED")):
            failed.append(name)
        if not os.path.exists(os.path.join(d, ".DONE")):
            not_done.append(name)
    ok(not failed, f".FAILED markers inside the analysis set after the two "
                   f"exclusions: {len(failed)} {failed[:6]}")
    ok(not not_done, f"every analysis-set member carries .DONE "
                     f"(missing: {len(not_done)} {not_done[:6]})")

    log(f"[set] analysis set validated: {N_TRAIN_CELLS_EXPECTED} training "
        f"cells + {6 * N_PROBES_PER_FAMILY_EXPECTED} probe dumps, "
        f"{len(EXCLUDED_PATHS)} paths excluded by name")
    return training, probes


# -------------------------------------------------------------------- load --

def load_training_cell(meta):
    """Read one training cell's parquet and sidecar; assert schema and identity."""
    run_id = meta["run_id"]
    d = os.path.join(DUMPS_ROOT, run_id)
    pq_path, rj_path = (os.path.join(d, "perimage.parquet"),
                        os.path.join(d, "run.json"))
    df = pd.read_parquet(pq_path)
    with open(rj_path) as fh:
        run = json.load(fh)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    ok(not missing, f"{run_id}: required columns present "
                    f"({'missing ' + str(missing) if missing else 'all'})")
    bad = [c for c in INT_COLS if not pd.api.types.is_integer_dtype(df[c])]
    ok(not bad, f"{run_id}: count columns are integer dtype "
                f"({'non-integer ' + str(bad) if bad else 'all'})")
    ok(pd.api.types.is_bool_dtype(df["empty_fg_valid"]),
       f"{run_id}: empty_fg_valid is boolean dtype")
    ok(len(df) == N_TEST_EXPECTED,
       f"{run_id}: parquet rows = {len(df)} (expected {N_TEST_EXPECTED})")
    ok(int(run.get("n_test", -1)) == N_TEST_EXPECTED,
       f"{run_id}: run.json n_test = {run.get('n_test')}")
    ok(df["image_id"].is_unique, f"{run_id}: image_id is unique")
    ok(set(df["city"].unique()) == {meta["city"]},
       f"{run_id}: every test row is from the held-out city {meta['city']}")

    for key, want in (("arch", ARCH), ("mode", meta["mode"]),
                      ("holdout", meta["city"]), ("seed", meta["seed"]),
                      ("rung", RUNG), ("res", RES), ("method", meta["corner"]),
                      ("cell_2x2", meta["corner"])):
        ok(str(run.get(key)) == str(want),
           f"{run_id}: run.json {key} = {run.get(key)!r} matches run_id "
           f"({want!r})")
    ok(run.get("target_access") in ("source_only", "unlabeled_target",
                                    "deployment_time", "target_labels"),
       f"{run_id}: target_access = {run.get('target_access')!r} is a CONTRACT "
       f"value")

    df = df.sort_values("image_id").reset_index(drop=True)
    iou = per_image_metrics(df)
    n_pred_pos = df["n_pred_pos"].to_numpy(dtype=np.float64)
    n_valid = df["n_valid_px"].to_numpy(dtype=np.float64)

    return {
        "run_id": run_id, "dir": d,
        "corner": meta["corner"], "mode": meta["mode"], "city": meta["city"],
        "seed": meta["seed"],
        "image_ids": df["image_id"].to_numpy(),
        "iou": iou,
        "level_pct": {m: float(v.mean() * 100.0) for m, v in iou.items()},
        "n_train": run.get("n_train"), "n_val": run.get("n_val"),
        "n_test": int(len(df)),
        "n_rows_loaded": int(len(df)),
        "pretraining": run.get("pretraining"),
        "lr": run.get("lr"), "lr_label": run.get("lr_label"),
        "target_access": run.get("target_access"),
        "target_access_detail": run.get("target_access_detail"),
        "steps_to_best": run.get("steps_to_best"),
        "epochs_run": run.get("epochs_run"), "best_epoch": run.get("best_epoch"),
        "best_val_metric": run.get("best_val_metric"),
        # Two conventions, both carried. The dumps' own pred_pos_fraction is
        # the mean of per-image fractions, which is what CC-SLURM's flag on
        # ssl_lo/miami/s2 quotes, so that is the primary here and the flag
        # comparison is like-for-like. armc_ladder_stats.py pools over pixels
        # instead; that value is carried alongside so the two arms' columns are
        # never silently read as the same quantity.
        "pred_pos_fraction": float(np.mean(n_pred_pos / n_valid)),
        "pred_pos_fraction_pooled_px": float(n_pred_pos.sum()
                                             / max(n_valid.sum(), 1.0)),
        "pred_pos_fraction_stored": run.get("pred_pos_fraction"),
        "determinism_max_delta": run.get("determinism_max_abs_delta_per_image"),
        "determinism_passes": run.get("determinism_passes"),
        "empty_fg_valid": int(df["empty_fg_valid"].sum()),
        "headline_2class_stored": run.get("headline_miou_2class_k2"),
        "headline_fg_stored": run.get("headline_fg_iou_k2_pct"),
        "aurc": {
            "trapz19_fg": df["aurc_fg"].to_numpy(dtype=np.float64),
            "trapz19_bg": df["aurc_bg"].to_numpy(dtype=np.float64),
            "mean20_fg": df["aurc_fg_mean20"].to_numpy(dtype=np.float64),
            "mean20_bg": df["aurc_bg_mean20"].to_numpy(dtype=np.float64),
        },
        "metric_module_sha": run.get("metric_module_sha"),
        "metric_module_sha_prev": run.get("metric_module_sha_prev"),
        "model_module_sha": run.get("model_module_sha"),
        "dump_module_sha": run.get("dump_module_sha"),
        "runtime_module_sha": run.get("runtime_module_sha"),
        "train_script_sha": run.get("train_script_sha"),
        "git_hash": run.get("git_hash"),
        "git_hash_reason": run.get("git_hash_reason"),
        "prediction_recipe": run.get("prediction_recipe"),
        "decision_metric": run.get("decision_metric"),
        "optimizer": run.get("optimizer"),
        "sanity_flags": run.get("sanity_flags"),
        "parquet_sha256": sha256_file(pq_path),
    }


def load_probe(parent_run_id, probe):
    d = os.path.join(DUMPS_ROOT, f"{parent_run_id}-{probe}")
    with open(os.path.join(d, "run.json")) as fh:
        run = json.load(fh)
    ok(run.get("parent_run_id") == parent_run_id,
       f"{parent_run_id}-{probe}: parent_run_id matches its cell")
    out = {"run": run, "dir": d, "df": None}
    pq = os.path.join(d, "perimage.parquet")
    if os.path.exists(pq):
        df = pd.read_parquet(pq).sort_values("image_id").reset_index(drop=True)
        ok(len(df) == N_TEST_EXPECTED,
           f"{parent_run_id}-{probe}: parquet rows = {len(df)}")
        out["df"] = df
        out["iou"] = per_image_metrics(df)
        out["image_ids"] = df["image_id"].to_numpy()
        out["parquet_sha256"] = sha256_file(pq)
    return out


# -------------------------------------------------------------- statistics --

def _rng(seed=BOOT_SEED):
    return np.random.default_rng(seed)


def _pad_clusters(values, cluster_ids):
    """Ragged clusters as a (n_clusters, max_size) array padded with NaN.

    The padding is what lets a bootstrap over clusters be vectorised: a
    replicate is a row-index into this matrix, and the statistic is a
    NaN-skipping reduction over the flattened result.  Cluster sizes here are 1
    (one seed) or 3 (three seeds of one image), so the padding waste is small.
    """
    keys, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    inv_s, val_s = inv[order], np.asarray(values, dtype=np.float64)[order]
    sizes = np.bincount(inv_s, minlength=keys.size)
    width = int(sizes.max())
    pad = np.full((keys.size, width), np.nan, dtype=np.float64)
    slot = np.concatenate([np.arange(s) for s in sizes]) if keys.size else \
        np.array([], dtype=int)
    pad[inv_s, slot] = val_s
    return pad, int(keys.size)


def cluster_bootstrap(values, cluster_ids, B=B, seed=BOOT_SEED, chunk=500):
    """Percentile CI for the MEDIAN under resampling of image clusters.

    The resampling unit is the image, not the (image, seed) row: the same 150
    held-out images recur under every seed, so seed replicates of one image are
    one cluster.  Matches App G.9's image-cluster bootstrap.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return dict(point=None, ci95_lo=None, ci95_hi=None, n=0, n_clusters=0,
                    B=B)
    pad, n_clusters = _pad_clusters(values, cluster_ids)
    rng = _rng(seed)
    boots = np.empty(B, dtype=np.float64)
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        idx = rng.integers(0, n_clusters, size=(hi - lo, n_clusters))
        boots[lo:hi] = np.nanmedian(
            pad[idx].reshape(hi - lo, -1), axis=1)
    return dict(point=float(np.median(values)),
                ci95_lo=float(np.percentile(boots, 2.5)),
                ci95_hi=float(np.percentile(boots, 97.5)),
                n=int(values.size), n_clusters=n_clusters, B=B)


def paired_bootstrap(delta, cluster_ids=None, B=B, seed=BOOT_SEED):
    """Paired bootstrap on a per-image delta vector.

    Behaviour matches armc_ladder_stats.py:paired_bootstrap — B resamples,
    percentile CI at 2.5 / 97.5, two-tailed p of min(2 * prop_wrong, 1.0) —
    with the resampling unit set to the image cluster when cluster_ids is given.
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    if n == 0:
        return dict(mean=None, median=None, stderr=None, ci95_lo=None,
                    ci95_hi=None, p_raw=None, n=0, n_clusters=0, B=B)
    observed = float(delta.mean())
    if cluster_ids is None:
        rng = _rng(seed)
        boots = delta[rng.integers(0, n, size=(B, n))].mean(axis=1)
        n_clusters = n
    else:
        pad, n_clusters = _pad_clusters(delta, cluster_ids)
        rng = _rng(seed)
        boots = np.empty(B, dtype=np.float64)
        chunk = 500
        for lo in range(0, B, chunk):
            hi = min(lo + chunk, B)
            idx = rng.integers(0, n_clusters, size=(hi - lo, n_clusters))
            flat = pad[idx].reshape(hi - lo, -1)
            boots[lo:hi] = np.nanmean(flat, axis=1)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    prop_wrong = (float(np.mean(boots <= 0)) if observed >= 0
                  else float(np.mean(boots >= 0)))
    return dict(mean=observed, median=float(np.median(delta)),
                stderr=(float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else None),
                ci95_lo=lo, ci95_hi=hi, p_raw=float(min(2.0 * prop_wrong, 1.0)),
                n=int(n), n_clusters=n_clusters, B=B)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, matching Table 39."""
    pv = [(p, i) for i, p in enumerate(pvals) if p is not None]
    out = [None] * len(pvals)
    if not pv:
        return out
    pv.sort()
    m, running = len(pv), 0.0
    for rank, (p, i) in enumerate(pv):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = float(running)
    return out


# ------------------------------------------------------- recovery ratio R ----

def recovery_ratio(loco_iou, upper_iou, retrained_iou):
    """R = (retrained - LOCO) / (Upper - LOCO), per image, on the IoU fraction
    scale, with images whose headroom is below 0.02 set to NaN.

    The exclusion is the dumps' own recorded rule, 'headroom (upper - LOCO) <
    0.02'.  It exists because R's denominator vanishes when the LOCO model is
    already at its upper bound on that image, and a ratio through a vanishing
    denominator is noise, not recovery.
    """
    headroom = upper_iou - loco_iou
    usable = headroom >= HEADROOM_MIN
    R = np.full(loco_iou.shape, np.nan, dtype=np.float64)
    R[usable] = ((retrained_iou[usable] - loco_iou[usable])
                 / headroom[usable])
    return R, headroom, usable


# ------------------------------------------------------------------- build --

def build_cells(training, corners):
    cells = {}
    for meta in training:
        if meta["corner"] not in corners:
            continue
        c = load_training_cell(meta)
        cells[(c["corner"], c["city"], c["seed"], c["mode"])] = c
    log(f"[load] loaded {len(cells)} training cells "
        f"({len(corners)} corner(s))")
    return cells


def build_per_cell(cells, corners):
    """One record per LOCO cell, carrying every per-image vector downstream
    stages need.  Every quantity here is recomputed from per-image values."""
    out = []
    for corner in corners:
        for city in CITIES:
            for seed in SEEDS:
                loco = cells[(corner, city, seed, "loco")]
                upper = cells[(corner, city, seed, "upper")]
                ok(np.array_equal(loco["image_ids"], upper["image_ids"]),
                   f"{loco['run_id']}: image-ID set matches its upper bound "
                   f"{upper['run_id']}")

                rec = {
                    "arm": "L",
                    "weakness_ids": ["zNBu-W5", "zNBu-W1", "AC1", "AC2"],
                    "run_id": loco["run_id"],
                    "upper_run_id": upper["run_id"],
                    "corner": corner, "pretraining": loco["pretraining"],
                    "lr": loco["lr"], "lr_label": loco["lr_label"],
                    "arch": ARCH, "city": city, "seed": seed,
                    "target_access": loco["target_access"],
                    "target_access_detail": loco["target_access_detail"],
                    "n_train": loco["n_train"], "n_val": loco["n_val"],
                    "n_test": loco["n_test"],
                    "n_train_upper": upper["n_train"],
                    "n_rows_loaded_loco": loco["n_rows_loaded"],
                    "n_rows_loaded_upper": upper["n_rows_loaded"],
                    "steps_to_best_train": loco["steps_to_best"],
                    "epochs_run_train": loco["epochs_run"],
                    "pred_pos_fraction": loco["pred_pos_fraction"],
                    "pred_pos_fraction_definition":
                        "mean over images of n_pred_pos / n_valid_px, the "
                        "convention the dumps' own pred_pos_fraction field "
                        "uses",
                    "pred_pos_fraction_pooled_px":
                        loco["pred_pos_fraction_pooled_px"],
                    "pred_pos_fraction_upper": upper["pred_pos_fraction"],
                    "determinism_max_delta": loco["determinism_max_delta"],
                    "determinism_passes": loco["determinism_passes"],
                    "empty_fg_valid": loco["empty_fg_valid"],
                    "empty_fg_valid_upper": upper["empty_fg_valid"],
                    "git_hash": loco["git_hash"],
                    "metric_module_sha": loco["metric_module_sha"],
                    "metric_module_sha_prev": loco["metric_module_sha_prev"],
                    "train_script_sha": loco["train_script_sha"],
                    "levels_pct": {},
                    "gaps": {},
                    "expA": {}, "expB": {}, "histmatch": {},
                    "probe": {}, "drift": {}, "selpred": {},
                    "caveats": [],
                    "_image_ids": loco["image_ids"],
                    "_R": {}, "_gap_delta": {},
                }

                # -------- levels and the LOCO-vs-upper gap, both definitions
                for m in METRICS:
                    lv, uv = loco["iou"][m], upper["iou"][m]
                    rec["levels_pct"][m] = {
                        "loco": float(lv.mean() * 100.0),
                        "upper": float(uv.mean() * 100.0),
                    }
                    delta_pct = (uv - lv) * 100.0
                    st = paired_bootstrap(delta_pct)
                    rec["gaps"][m] = {
                        "metric": f"gap_upper_minus_loco_{m}_pct",
                        "mean": st["mean"], "stderr": st["stderr"],
                        "ci95": [st["ci95_lo"], st["ci95_hi"]],
                        "ci_method": "paired_bootstrap_B10000_per_image",
                        "p_raw": st["p_raw"], "p_holm": None, "n": st["n"],
                        "test": "paired bootstrap on per-image deltas",
                    }
                    rec["_gap_delta"][m] = delta_pct

                # -------- Experiments A and B
                for probe, key in (("p3expA", "expA"), ("p3expB", "expB")):
                    pr = load_probe(loco["run_id"], probe)
                    prun = pr["run"][key]
                    ok(np.array_equal(pr["image_ids"], loco["image_ids"]),
                       f"{loco['run_id']}-{probe}: image-ID set matches its "
                       f"parent cell")
                    blk = {
                        "probe_dir": os.path.basename(pr["dir"]),
                        "target_access": pr["run"].get("target_access"),
                        "steps_to_best": prun.get("steps_to_best"),
                        "steps_ceiling": pr["run"].get("steps_ceiling"),
                        "epochs_run": prun.get("epochs_run"),
                        "budget_bound": prun.get("budget_bound"),
                        "budget_bound_note": prun.get("budget_bound_note"),
                        "n_trainable_params": prun.get("n_trainable_params"),
                        "encoder_lr_multiplier":
                            prun.get("exp_b_encoder_lr_multiplier"),
                        "stored_median_R_fg_only": prun.get("median_R"),
                        "stored_n_used": prun.get("n_used"),
                        "stored_n_excluded": prun.get("n_excluded"),
                        "R": {},
                    }
                    for m in METRICS:
                        R, headroom, usable = recovery_ratio(
                            loco["iou"][m], upper["iou"][m], pr["iou"][m])
                        bs = cluster_bootstrap(R[usable],
                                               loco["image_ids"][usable])
                        blk["R"][m] = {
                            "median_R": bs["point"],
                            "ci95": [bs["ci95_lo"], bs["ci95_hi"]],
                            "ci_method": "image_cluster_bootstrap_B10000",
                            "mean_R": (float(np.mean(R[usable]))
                                       if usable.any() else None),
                            "n_used": int(usable.sum()),
                            "n_excluded": int((~usable).sum()),
                            "exclusion_rule":
                                f"headroom (upper - LOCO) < {HEADROOM_MIN} on "
                                f"the IoU fraction scale",
                            "retrained_level_pct":
                                float(pr["iou"][m].mean() * 100.0),
                        }
                        rec["_R"].setdefault(key, {})[m] = (R, usable)
                    rec[key] = blk

                # -------- Section 4.2 histogram matching
                pr = load_probe(loco["run_id"], "p1histmatch")
                ok(np.array_equal(pr["image_ids"], loco["image_ids"]),
                   f"{loco['run_id']}-p1histmatch: image-ID set matches its "
                   f"parent cell")
                hm = {"probe_dir": os.path.basename(pr["dir"]),
                      "rule": pr["run"]["matching_rules"]["headline"],
                      "stored_gray_median_R_fg_only":
                          pr["run"]["gray"]["median_R"],
                      "stored_gray_improved": pr["run"]["gray"]["improved"],
                      "stored_per_channel_median_R_fg_only":
                          pr["run"]["per_channel"]["median_R"],
                      "R": {}}
                for m in METRICS:
                    R, headroom, usable = recovery_ratio(
                        loco["iou"][m], upper["iou"][m], pr["iou"][m])
                    bs = cluster_bootstrap(R[usable], loco["image_ids"][usable])
                    hm["R"][m] = {
                        "median_R": bs["point"],
                        "ci95": [bs["ci95_lo"], bs["ci95_hi"]],
                        "ci_method": "image_cluster_bootstrap_B10000",
                        "mean_R": (float(np.mean(R[usable]))
                                   if usable.any() else None),
                        "n_used": int(usable.sum()),
                        "n_excluded": int((~usable).sum()),
                        "matched_level_pct": float(pr["iou"][m].mean() * 100.0),
                        "mean_delta_vs_loco_pct":
                            float((pr["iou"][m] - loco["iou"][m]).mean() * 100.0),
                        "improved": bool(pr["iou"][m].mean()
                                         > loco["iou"][m].mean()),
                    }
                    rec["_R"].setdefault("histmatch", {})[m] = (R, usable)
                rec["histmatch"] = hm

                # -------- P2 city-identity probe (E-8 scoped)
                pr = load_probe(loco["run_id"], "p2cityprobe")
                p = pr["run"]["probe"]
                ctrl = pr["run"]["raw_pixel_control"]
                per_img = np.asarray(pr["run"]["per_image_accuracy"],
                                     dtype=np.float64)
                rec["probe"] = {
                    "probe_dir": os.path.basename(pr["dir"]),
                    "tap": pr["run"].get("tap"),
                    "feature_probe_accuracy": p["probe_accuracy"],
                    "feature_probe_accuracy_from_per_image":
                        float(per_img.mean()),
                    "null_mean": p["null_mean"], "null_p99": p["null_p99"],
                    "p_value": p["p_value"],
                    "n_permutations": p["n_permutations"],
                    "chance": p["chance"],
                    "raw_pixel_control_accuracy": ctrl["probe_accuracy"],
                    "raw_pixel_control_features": ctrl["features"],
                    "feature_minus_raw_pixel_control":
                        float(p["probe_accuracy"] - ctrl["probe_accuracy"]),
                    "cv": pr["run"].get("cv"),
                }

                # -------- P5 encoder drift
                pr = load_probe(loco["run_id"], "p5drift")
                stages = pr["run"]["drift_rel_l2_per_stage"]
                rec["drift"] = {
                    "probe_dir": os.path.basename(pr["dir"]),
                    "relative_l2": {k: v["relative_l2"]
                                    for k, v in stages.items()},
                    "per_module_relative_l2": {
                        k: {mk: mv["relative_l2"]
                            for mk, mv in v.get("per_module", {}).items()}
                        for k, v in stages.items()},
                    "cka": dict(pr["run"]["cka_per_stage"]),
                    "cka_definition": pr["run"].get("cka_definition"),
                    "cka_images": pr["run"].get("cka_images"),
                    "drift_note": pr["run"].get("drift_note"),
                }

                # -------- P3 selective prediction, both AURC definitions
                pr = load_probe(loco["run_id"], "p3selpred")
                sp = {"probe_dir": os.path.basename(pr["dir"]),
                      "coverage_grid_n": len(pr["run"]["coverage_grid"]),
                      "definition_note": pr["run"]["definition_note"],
                      "aurc": {}, "stored_aurc": {}}
                # aurc_fg is undefined on an image with no ground-truth
                # foreground: there is no foreground population to build a
                # risk-coverage curve over. Those images arrive as NaN, exactly
                # the empty_fg_valid set, and the cell mean skips them. The
                # count that contributed is carried so the denominator is never
                # implicit.
                for defn, cols in (("aurc_trapz19",
                                    ("trapz19_fg", "trapz19_bg")),
                                   ("aurc_mean20",
                                    ("mean20_fg", "mean20_bg"))):
                    sp["aurc"][defn] = {
                        "fg": float(np.nanmean(loco["aurc"][cols[0]])),
                        "bg": float(np.nanmean(loco["aurc"][cols[1]])),
                        "n_images_fg": int(np.isfinite(
                            loco["aurc"][cols[0]]).sum()),
                        "n_images_bg": int(np.isfinite(
                            loco["aurc"][cols[1]]).sum()),
                    }
                    for pop, col in (("fg", cols[0]), ("bg", cols[1])):
                        n_nan = int((~np.isfinite(loco["aurc"][col])).sum())
                        if pop == "fg":
                            ok(n_nan == loco["empty_fg_valid"],
                               f"{loco['run_id']}: aurc_fg is undefined on "
                               f"exactly the {loco['empty_fg_valid']} "
                               f"empty-foreground images ({n_nan} NaN)")
                        else:
                            ok(n_nan == 0,
                               f"{loco['run_id']}: aurc_bg is defined on every "
                               f"image ({n_nan} NaN)")
                    sp["stored_aurc"][defn] = {
                        "fg": pr["run"][defn]["fg"],
                        "bg": pr["run"][defn]["bg"],
                    }
                sp["_per_image"] = loco["aurc"]
                rec["selpred"] = sp

                out.append(rec)
    log(f"[build] built {len(out)} LOCO-cell records")
    return out


# ------------------------------------------------------------------ gates ----

def gate_G1(corner_R, per_cell):
    """sup_hi's Experiment B median R against the CNN-like threshold."""
    res = {"gate": "G1", "name": "control: sup_hi Experiment B is CNN-like",
           "threshold": G1_THRESHOLD,
           "published_comparators": PUBLISHED_EXPB,
           "corner": "sup_hi", "by_metric": {}}
    if "sup_hi" not in corner_R.get("expB", {}):
        res["status"] = "NOT_EVALUATED"
        res["note"] = "sup_hi is not in the corners analysed in this run"
        return res
    for m in METRICS:
        row = corner_R["expB"]["sup_hi"][m]
        res["by_metric"][m] = {
            "median_R": row["median_R"], "ci95": row["ci95"],
            "n_used": row["n_used"], "n_excluded": row["n_excluded"],
            "passes": bool(row["median_R"] is not None
                           and row["median_R"] >= G1_THRESHOLD),
        }
    primary = res["by_metric"][PRIMARY_METRIC]
    res["status"] = "PASS" if primary["passes"] else "FAIL"
    res["status_metric"] = PRIMARY_METRIC
    res["agrees_across_metrics"] = (
        res["by_metric"]["2class_tol_k2"]["passes"]
        == res["by_metric"]["fg_only_tol_k2"]["passes"])
    res["per_city_median_R"] = {
        m: {c["city"]: c["expB"]["R"][m]["median_R"]
            for c in per_cell if c["corner"] == "sup_hi"}
        for m in METRICS}
    return res


def gate_G2(per_cell, corners):
    """Does the _lo label correspond to measurably smaller encoder drift?"""
    stages = ("conv1", "layer1", "layer2", "layer3", "layer4", "whole_encoder")
    table = {}
    for corner in corners:
        rows = [c for c in per_cell if c["corner"] == corner]
        table[corner] = {
            "n_cells": len(rows),
            "relative_l2": {s: {
                "mean": float(np.mean([r["drift"]["relative_l2"][s]
                                       for r in rows])),
                "min": float(np.min([r["drift"]["relative_l2"][s]
                                     for r in rows])),
                "max": float(np.max([r["drift"]["relative_l2"][s]
                                     for r in rows])),
            } for s in stages},
            "cka": {s: {
                "mean": float(np.mean([r["drift"]["cka"][s] for r in rows])),
                "min": float(np.min([r["drift"]["cka"][s] for r in rows])),
                "max": float(np.max([r["drift"]["cka"][s] for r in rows])),
            } for s in stages if s in rows[0]["drift"]["cka"]},
        }

    separates = {}
    for axis, lo, hi in (("ssl", "ssl_lo", "ssl_hi"),
                         ("sup", "sup_lo", "sup_hi")):
        if lo not in table or hi not in table:
            continue
        separates[axis] = {
            "relative_l2_lo_below_hi_no_overlap": {
                s: bool(table[lo]["relative_l2"][s]["max"]
                        < table[hi]["relative_l2"][s]["min"])
                for s in stages},
            "cka_lo_above_hi_no_overlap": {
                s: bool(table[lo]["cka"][s]["min"] > table[hi]["cka"][s]["max"])
                for s in table[lo]["cka"]},
            "relative_l2_lo_below_hi_in_mean": {
                s: bool(table[lo]["relative_l2"][s]["mean"]
                        < table[hi]["relative_l2"][s]["mean"])
                for s in stages},
            "cka_lo_above_hi_in_mean": {
                s: bool(table[lo]["cka"][s]["mean"] > table[hi]["cka"][s]["mean"])
                for s in table[lo]["cka"]},
            "relative_l2_ratio_hi_over_lo_in_mean": {
                s: float(table[hi]["relative_l2"][s]["mean"]
                         / table[lo]["relative_l2"][s]["mean"])
                for s in stages},
        }

    def every(field):
        vals = [v for a in separates.values() for v in a[field].values()]
        return bool(vals) and all(vals)

    l2_sep = every("relative_l2_lo_below_hi_no_overlap")
    cka_sep = every("cka_lo_above_hi_no_overlap")
    l2_mean = every("relative_l2_lo_below_hi_in_mean")
    cka_mean = every("cka_lo_above_hi_in_mean")
    status = ("PASS" if separates and l2_sep and cka_sep
              else ("PARTIAL" if separates and (l2_sep or l2_mean)
                    else ("FAIL" if separates else "NOT_EVALUATED")))
    finding = None
    if status == "PARTIAL":
        finding = (
            "Relative L2 separates the _lo corners from the _hi corners with "
            f"no overlap at every stage on both axes ({l2_sep}); linear CKA "
            f"orders them the same way in the mean at every stage "
            f"({cka_mean}) but the corners' nine-cell ranges overlap, so CKA "
            "does not separate them under the no-overlap rule. The gate asked "
            "for both, and the two measures disagree on strength, not on "
            "direction.")
    return {
        "gate": "G2", "name": "'low LR' is a regime, not a label",
        "status": status,
        "separation_rule": "the _lo corner's worst cell is strictly below the "
                           "_hi corner's best cell on relative L2 (and strictly "
                           "above on CKA) at that stage, i.e. the corners do "
                           "not overlap across their 9 cells",
        "components": {
            "relative_l2_separates_no_overlap": l2_sep,
            "relative_l2_orders_correctly_in_mean": l2_mean,
            "cka_separates_no_overlap": cka_sep,
            "cka_orders_correctly_in_mean": cka_mean,
        },
        "finding": finding,
        "per_corner": table,
        "separation": separates,
        "arm_E_dinov3_drift_comparison": None,
        "arm_E_dinov3_drift_comparison_note":
            "NOT EVALUATED. The gate also asks whether ssl_lo's drift sits "
            "nearer arm E's measured DINOv3 encoder drift than sup_hi's does. "
            "Arm E's relative-L2 / CKA drift numbers are not on this box: "
            "results/02_frozen_backbone_gap.json carries the frozen-backbone "
            "gap only, and no p5drift dump exists for any dinov3 or "
            "dinov3frozen cell. This half of G2 is left blank rather than "
            "guessed.",
    }


def gate_G3(per_cell):
    """Budget asymmetry: a high R under a binding budget is valid, a low one
    is unreadable."""
    cells, counts = [], defaultdict(int)
    for c in per_cell:
        row = {"run_id": c["run_id"], "corner": c["corner"],
               "city": c["city"], "seed": c["seed"]}
        for key in ("expA", "expB"):
            med = c[key]["R"][PRIMARY_METRIC]["median_R"]
            bound = bool(c[key]["budget_bound"])
            readable = (not bound) or (med is not None and med >= G3_HIGH_R)
            readable30 = (not bound) or (med is not None
                                         and med >= G3_HIGH_R_STRICT)
            row[key] = {
                "steps_to_best": c[key]["steps_to_best"],
                "steps_ceiling": c[key]["steps_ceiling"],
                "budget_bound": bound,
                "median_R_primary": med,
                "readable": bool(readable),
                "readable_strict30": bool(readable30),
            }
            c[key]["readable"] = bool(readable)
            c[key]["readable_strict30"] = bool(readable30)
            c[key]["budget_bound_flag"] = bound
            counts[f"{key}_budget_bound"] += int(bound)
            counts[f"{key}_readable"] += int(readable)
            counts[f"{key}_unreadable"] += int(not readable)
            counts[f"{key}_readable_strict30"] += int(readable30)
        cells.append(row)
    n = len(per_cell)
    by_corner = {}
    for corner in sorted({c["corner"] for c in per_cell}):
        rows = [r for r in cells if r["corner"] == corner]
        by_corner[corner] = {
            key: {"n_cells": len(rows),
                  "n_budget_bound": sum(r[key]["budget_bound"] for r in rows),
                  "n_readable": sum(r[key]["readable"] for r in rows),
                  "n_unreadable": sum(not r[key]["readable"] for r in rows)}
            for key in ("expA", "expB")}
    return {
        "gate": "G3", "name": "budget asymmetry",
        "rule": ("a cell is READABLE when its 30-epoch budget does not bind, or "
                 f"when it binds and the cell's median R is >= {G3_HIGH_R} "
                 "(undertraining can only depress R, so a high R under a "
                 "binding budget is valid); it is UNREADABLE when the budget "
                 f"binds and median R < {G3_HIGH_R}, which is ambiguous "
                 "between locus and undertraining"),
        "high_R_threshold": G3_HIGH_R,
        "high_R_threshold_provenance":
            "the arm's own CNN-like threshold, so the readable rule and the "
            "signature table share one number. The pre-committed reading does "
            "not state a HIGH-R threshold separately, and 0.70 is calibrated "
            "on Experiment B; it is applied unchanged to Experiment A, whose "
            "published CNN-like band is 0.12-0.67, so the Experiment A "
            "unreadable count is the conservative reading. The direction of "
            "the rule is experiment-agnostic — undertraining can only depress "
            "R — and only the threshold is borrowed.",
        "sensitivity_threshold": G3_HIGH_R_STRICT,
        "readable_metric": PRIMARY_METRIC,
        "n_cells": n,
        "counts": dict(counts),
        "counts_by_corner": by_corner,
        "expected_binding_from_cc_slurm": {"expA": 29, "expB": 26,
                                           "of_n_cells": 36},
        "readable_axis_note":
            "Experiment B is the arm's decision variable and is also the more "
            "readable axis here. Experiment A's unreadable count is the larger "
            "one, and the asymmetry matters in one direction: the DINOv3-like "
            "signature needs Experiment A HIGH, and a binding budget can only "
            "depress R, so a low Experiment A under a binding budget is not "
            "evidence against that signature. The Experiment B numbers do not "
            "depend on Experiment A being readable.",
        "per_cell": cells,
    }


# ------------------------------------------------------------- aggregation ----

def pool_R(per_cell, corners):
    """Median R per (corner, city) and per corner, image-cluster bootstrapped."""
    out = {"expA": {}, "expB": {}, "histmatch": {},
           "expA_by_city": {}, "expB_by_city": {}, "histmatch_by_city": {}}
    for key in ("expA", "expB", "histmatch"):
        for corner in corners:
            rows = [c for c in per_cell if c["corner"] == corner]
            out[key][corner] = {}
            for m in METRICS:
                vals, clusters = [], []
                for c in rows:
                    R, usable = c["_R"][key][m]
                    vals.append(R[usable])
                    clusters.append(np.char.add(
                        np.asarray([c["city"] + "|"] * int(usable.sum())),
                        c["_image_ids"][usable].astype(str)))
                v = np.concatenate(vals) if vals else np.array([])
                k = np.concatenate(clusters) if clusters else np.array([])
                bs = cluster_bootstrap(v, k)
                out[key][corner][m] = {
                    "median_R": bs["point"], "ci95": [bs["ci95_lo"],
                                                      bs["ci95_hi"]],
                    "ci_method": "image_cluster_bootstrap_B10000",
                    "cluster_unit": "(city, image_id), pooled over 3 seeds",
                    "mean_R": float(v.mean()) if v.size else None,
                    "n_used": int(v.size), "n_clusters": bs["n_clusters"],
                    "n_excluded": int(sum(
                        (~c["_R"][key][m][1]).sum() for c in rows)),
                    "n_cells": len(rows),
                }
            for city in CITIES:
                crows = [c for c in rows if c["city"] == city]
                if not crows:
                    continue
                out[key + "_by_city"].setdefault(corner, {})[city] = {}
                for m in METRICS:
                    vals, clusters = [], []
                    for c in crows:
                        R, usable = c["_R"][key][m]
                        vals.append(R[usable])
                        clusters.append(c["_image_ids"][usable].astype(str))
                    v = np.concatenate(vals) if vals else np.array([])
                    k = np.concatenate(clusters) if clusters else np.array([])
                    bs = cluster_bootstrap(v, k)
                    out[key + "_by_city"][corner][city][m] = {
                        "median_R": bs["point"],
                        "ci95": [bs["ci95_lo"], bs["ci95_hi"]],
                        "ci_method": "image_cluster_bootstrap_B10000",
                        "cluster_unit": "image_id, pooled over 3 seeds",
                        "mean_R": float(v.mean()) if v.size else None,
                        "n_used": int(v.size),
                        "n_clusters": bs["n_clusters"],
                        "n_excluded": int(sum(
                            (~c["_R"][key][m][1]).sum() for c in crows)),
                        "n_cells": len(crows),
                    }
    return out


CONTRASTS = (("ssl_lo_vs_sup_hi", "ssl_lo", "sup_hi",
              "full regime swap; zNBu-W5's literal ask"),
             ("ssl_lo_vs_ssl_hi", "ssl_lo", "ssl_hi",
              "isolates fine-tuning LR at fixed pretraining"),
             ("ssl_lo_vs_sup_lo", "ssl_lo", "sup_lo",
              "isolates pretraining objective at fixed LR"))


def per_cell_caveats(per_cell):
    """Fill each cell's own caveats list.  Runs after G3, because a cell's
    readability is one of them."""
    for c in per_cell:
        cav = c["caveats"]
        for key, label in (("expA", "Experiment A"), ("expB", "Experiment B")):
            if c[key]["budget_bound"]:
                cav.append(
                    f"{label}'s 30-epoch budget binds "
                    f"({c[key]['steps_to_best']} of "
                    f"{c[key]['steps_ceiling']} steps), so its R is a lower "
                    f"bound. G3 marks this cell "
                    f"{'READABLE' if c[key]['readable'] else 'UNREADABLE'}: "
                    + ("a high R under a binding budget is still valid."
                       if c[key]["readable"] else
                       "a low R under a binding budget is ambiguous between "
                       "locus and undertraining."))
        for key, label in (("expA", "Experiment A"), ("expB", "Experiment B")):
            for mm in METRICS:
                n_ex = c[key]["R"][mm]["n_excluded"]
                if n_ex:
                    cav.append(
                        f"{label} [{mm}]: {n_ex} of {c['n_test']} images "
                        f"excluded from R by the headroom rule "
                        f"(upper - LOCO < {HEADROOM_MIN}); the median R "
                        f"describes the remaining {c[key]['R'][mm]['n_used']}.")
        if c["empty_fg_valid"]:
            cav.append(
                f"{c['empty_fg_valid']} of {c['n_test']} test images have no "
                f"ground-truth foreground on the valid region. Foreground IoU "
                f"is 0 by construction on those images and aurc_fg is "
                f"undefined, so the AURC foreground mean is over "
                f"{c['n_test'] - c['empty_fg_valid']} images.")
        if c["run_id"] == FLAGGED_CELL:
            cav.append(
                "CC-SLURM flagged this cell: LOCO 56.77 against siblings 66.71 "
                "and 67.56, pred_pos_fraction 2-3x its siblings, training "
                "otherwise healthy. It is reported as measured and is included "
                "in every pooled number; not dropped, not winsorised.")
        if c["expB"].get("encoder_lr_multiplier") is not None and \
                float(c["expB"]["encoder_lr_multiplier"]) != 1.0:
            cav.append(
                f"Experiment B ran with an encoder LR multiplier of "
                f"{c['expB']['encoder_lr_multiplier']}, not the 1x the "
                f"published MAMNet runs used (erratum E-10).")
        if c["target_access"] != "source_only":
            cav.append(f"The LOCO cell's target_access is "
                       f"{c['target_access']}, not source_only.")


def contrasts(per_cell, corners, key="expB"):
    """Paired bootstrap on per-image R deltas between corners.

    Pairing is on (city, seed, image_id): the same held-out images and the same
    seeds appear in every corner.  An image enters a contrast only when R is
    defined on BOTH sides, because the headroom exclusion is evaluated per
    corner against that corner's own LOCO and upper bound.
    """
    idx = {}
    for c in per_cell:
        for m in METRICS:
            R, usable = c["_R"][key][m]
            idx[(c["corner"], c["city"], c["seed"], m)] = (
                c["_image_ids"], R, usable)
    rows = []
    for name, a, b, note in CONTRASTS:
        if a not in corners or b not in corners:
            continue
        row = {"contrast": name, "compares": [a, b], "note": note,
               "experiment": key, "by_metric": {}}
        for m in METRICS:
            deltas, clusters, n_a_only, n_b_only = [], [], 0, 0
            for city in CITIES:
                for seed in SEEDS:
                    ia, Ra, ua = idx[(a, city, seed, m)]
                    ib, Rb, ub = idx[(b, city, seed, m)]
                    ok(np.array_equal(ia, ib),
                       f"contrast {name} {city}/s{seed}: image-ID sets match "
                       f"across corners {a} and {b}")
                    both = ua & ub
                    n_a_only += int((ua & ~ub).sum())
                    n_b_only += int((ub & ~ua).sum())
                    deltas.append(Ra[both] - Rb[both])
                    clusters.append(np.char.add(
                        np.asarray([city + "|"] * int(both.sum())),
                        ia[both].astype(str)))
            d = np.concatenate(deltas) if deltas else np.array([])
            k = np.concatenate(clusters) if clusters else np.array([])
            st = paired_bootstrap(d, k)
            st_plain = paired_bootstrap(d, None)
            row["by_metric"][m] = {
                "test": "paired bootstrap on per-image R deltas, "
                        "image-cluster resampling",
                "mean": st["mean"], "median": st["median"],
                "stderr": st["stderr"],
                "ci95": [st["ci95_lo"], st["ci95_hi"]],
                "ci_method": "paired_image_cluster_bootstrap_B10000",
                "cluster_unit": "(city, image_id), pooled over 3 seeds",
                "p_raw": st["p_raw"], "p_holm": None,
                "n": st["n"], "n_clusters": st["n_clusters"],
                "B": B, "two_tailed": True,
                "n_usable_left_only": n_a_only,
                "n_usable_right_only": n_b_only,
                "unclustered_sensitivity": {
                    "mean": st_plain["mean"],
                    "ci95": [st_plain["ci95_lo"], st_plain["ci95_hi"]],
                    "p_raw": st_plain["p_raw"],
                    "ci_method": "paired_bootstrap_B10000_per_image",
                },
            }
        rows.append(row)
    for m in METRICS:
        adj = holm([r["by_metric"][m]["p_raw"] for r in rows])
        for r, a in zip(rows, adj):
            r["by_metric"][m]["p_holm"] = a
            r["by_metric"][m]["holm_family"] = (
                f"arm_L_{key}_three_contrasts_vs_sup_hi_axis")
    return rows


def corner_gaps(per_cell, corners):
    """Pooled per-corner LOCO-vs-upper gap, per-image, image-cluster CI."""
    out = {}
    for corner in corners:
        rows = [c for c in per_cell if c["corner"] == corner]
        out[corner] = {}
        for m in METRICS:
            d, k = [], []
            for c in rows:
                d.append(c["_gap_delta"][m])
                k.append(np.char.add(
                    np.asarray([c["city"] + "|"] * len(c["_image_ids"])),
                    c["_image_ids"].astype(str)))
            dd, kk = np.concatenate(d), np.concatenate(k)
            st = paired_bootstrap(dd, kk)
            out[corner][m] = {
                "metric": f"gap_upper_minus_loco_{m}_pct",
                "mean": st["mean"], "stderr": st["stderr"],
                "ci95": [st["ci95_lo"], st["ci95_hi"]],
                "ci_method": "paired_image_cluster_bootstrap_B10000",
                "p_raw": st["p_raw"], "n": st["n"],
                "n_clusters": st["n_clusters"], "n_cells": len(rows),
                "worst_case_cell": max(
                    rows, key=lambda c: c["gaps"][m]["mean"])["run_id"],
                "worst_case_gap": max(c["gaps"][m]["mean"] for c in rows),
                "best_case_gap": min(c["gaps"][m]["mean"] for c in rows),
            }
    return out


# --------------------------------------------------------- cross-checks -----

def reproduction_check(per_cell):
    """The dumps stored their own per-image R on the foreground-only K=2
    definition.  Recomputing it here from raw counts must land on the stored
    value exactly; that is the check that this reimplementation of the metric
    is the same metric."""
    worst, worst_id, rows, nonfinite = 0.0, None, [], []
    for c in per_cell:
        for key in ("expA", "expB"):
            stored = c[key]["stored_median_R_fg_only"]
            got = c[key]["R"]["fg_only_tol_k2"]["median_R"]
            d = abs(float(stored) - float(got))
            if not np.isfinite(d):
                nonfinite.append(f"{c['run_id']}:{key}")
            rows.append({"run_id": c["run_id"], "experiment": key,
                         "stored_median_R": float(stored),
                         "recomputed_median_R": float(got), "abs_delta": d,
                         "stored_n_used": c[key]["stored_n_used"],
                         "recomputed_n_used":
                             c[key]["R"]["fg_only_tol_k2"]["n_used"]})
            if d > worst:
                worst, worst_id = d, f"{c['run_id']}:{key}"
        s = c["histmatch"]["stored_gray_median_R_fg_only"]
        g = c["histmatch"]["R"]["fg_only_tol_k2"]["median_R"]
        d = abs(float(s) - float(g))
        if not np.isfinite(d):
            nonfinite.append(f"{c['run_id']}:histmatch")
        rows.append({"run_id": c["run_id"], "experiment": "histmatch_gray",
                     "stored_median_R": float(s), "recomputed_median_R":
                         float(g), "abs_delta": d})
        if d > worst:
            worst, worst_id = d, f"{c['run_id']}:histmatch"
    ok(not nonfinite,
       f"every recomputed-vs-stored median R comparison is finite "
       f"({len(nonfinite)} non-finite: {nonfinite[:3]})")
    log(f"[repro] max |delta| recomputed-vs-stored median R (foreground-only "
        f"K=2) = {worst:.3e} ({worst_id})")
    n_mismatch = sum(1 for r in rows
                     if r.get("recomputed_n_used") is not None
                     and r["recomputed_n_used"] != r["stored_n_used"])
    ok(worst < 1e-9,
       f"recomputed per-cell median R reproduces the stored value to <1e-9 "
       f"(worst {worst:.3e})")
    ok(n_mismatch == 0,
       f"recomputed headroom-exclusion counts match the stored n_used on every "
       f"cell ({n_mismatch} mismatches)")

    # The p1histmatch parquet is asserted to be the headline (grayscale) run,
    # not the per-channel one that rides along in the sidecar.
    pc_worst = max(abs(float(c["histmatch"]["stored_per_channel_median_R_fg_only"])
                       - float(c["histmatch"]["R"]["fg_only_tol_k2"]["median_R"]))
                   for c in per_cell)
    log(f"[repro] the same recomputation is {pc_worst:.4f} away from the "
        f"sidecar's per-channel median R, which identifies the parquet as the "
        f"grayscale (published-rule) matched run")

    # NaN-aware: a NaN delta must fail this check, not slip through it, which
    # is what a bare `d > worst` comparison would do.
    sp_worst, sp_id, sp_nan = 0.0, None, []
    for c in per_cell:
        for defn in ("aurc_trapz19", "aurc_mean20"):
            for pop in ("fg", "bg"):
                got = c["selpred"]["aurc"][defn][pop]
                stored = c["selpred"]["stored_aurc"][defn][pop]
                d = abs(float(got) - float(stored))
                if not np.isfinite(d):
                    sp_nan.append(f"{c['run_id']}:{defn}:{pop} "
                                  f"(recomputed {got}, stored {stored})")
                elif d > sp_worst:
                    sp_worst, sp_id = d, f"{c['run_id']}:{defn}:{pop}"
    ok(not sp_nan,
       f"every recomputed-vs-stored AURC comparison is finite "
       f"({len(sp_nan)} non-finite: {sp_nan[:3]})")
    ok(sp_worst < 1e-9,
       f"per-cell mean AURC recomputed from the per-image columns reproduces "
       f"the p3selpred sidecar scalars under both definitions "
       f"(worst {sp_worst:.3e}, {sp_id})")

    return {"max_abs_delta_median_R": worst,
            "max_abs_delta_run_id": worst_id,
            "n_checked": len(rows),
            "n_used_mismatches": n_mismatch,
            "distance_to_per_channel_variant": pc_worst,
            "aurc_max_abs_delta": sp_worst,
            "aurc_max_abs_delta_id": sp_id,
            "note": "the dumps' stored recovery ratios are on the "
                    "foreground-only K=2 definition; the two-class K=2 numbers "
                    "reported here have no stored counterpart and are checked "
                    "only through this shared code path",
            "per_check": rows}


def provenance_check(cells, per_cell):
    def uniq(vals):
        return sorted({v for v in vals if v is not None})
    tracked = ("metric_module_sha", "metric_module_sha_prev",
               "model_module_sha", "dump_module_sha", "runtime_module_sha",
               "train_script_sha", "git_hash", "prediction_recipe",
               "decision_metric", "optimizer")
    shas = {k: uniq(c.get(k) for c in cells.values()) for k in tracked}
    # A field that takes more than one value across the analysis set is
    # recorded with its exact membership, not just its count, because a
    # partition that falls on a corner boundary is a different thing from one
    # that does not.
    shas["multi_valued_membership"] = {
        k: {v: sorted(c["run_id"] for c in cells.values() if c.get(k) == v)
            for v in shas[k]}
        for k in tracked if len(shas[k]) > 1}
    # target_access is expected to differ by mode — a within-city upper bound
    # trains on held-out-city labels by construction — so it is broken out by
    # mode rather than flagged as a provenance anomaly.
    shas["target_access_training_by_mode"] = {
        mode: uniq(c["target_access"] for c in cells.values()
                   if c["mode"] == mode)
        for mode in MODES}
    # The optimizer string differs by construction: it carries the LR, which is
    # one of the two axes of the 2x2. It is recorded, not flagged.
    expected_multi = ("optimizer",)
    for k, v in shas.items():
        if not isinstance(v, list):
            continue
        log(f"[provenance] {k}: {len(v)} distinct value(s)")
        if len(v) > 1 and k not in expected_multi:
            memb = shas["multi_valued_membership"][k]
            sizes = {vv[:12]: len(mm) for vv, mm in memb.items()}
            flag(f"provenance: {k} takes {len(v)} distinct values across the "
                 f"{len(cells)} training cells, split {sizes}. The metric, "
                 f"model, dump and train-script modules are byte-identical "
                 f"across the whole set, which bounds what changed; the exact "
                 f"membership is in provenance_check.multi_valued_membership.")
    log(f"[provenance] target_access by mode: "
        f"{shas['target_access_training_by_mode']}")
    ok(len(shas["metric_module_sha"]) == 1,
       f"metric_module_sha is identical across every cell "
       f"({len(shas['metric_module_sha'])} distinct)")
    shas["local_script_sha256"] = {
        SCRIPT_LABEL: sha256_text(os.path.join(BASE, SCRIPT_LABEL)),
        "python/armc_ladder_stats.py":
            sha256_text(os.path.join(os.path.dirname(os.path.abspath(__file__)), "armc_ladder_stats.py")),
        "python/arme_frozen_gap_stats.py":
            sha256_text(os.path.join(os.path.dirname(os.path.abspath(__file__)), "arme_frozen_gap_stats.py")),
    }
    shas["metric_provenance_note"] = (
        "No metrics.py exists on this box, so the per-image metric is "
        "recomputed here with the formula of "
        "python/armc_ladder_stats.py:per_image_metrics, documented as identical "
        "at python/arme_frozen_gap_stats.py:138. metric_module_sha / "
        "metric_module_sha_prev are the SLURM-side shas carried in the dumps' "
        "sidecars and are recorded, not used.")
    shas["probe_target_access"] = {
        p: sorted({c[k].get("target_access") for c in per_cell})
        for p, k in (("p3expA", "expA"), ("p3expB", "expB"))}
    return shas


def sanity_checks(per_cell, cells):
    """Flag anything too clean or internally inconsistent before it is asked
    for."""
    det = [c["determinism_max_delta"] for c in per_cell
           if c["determinism_max_delta"] is not None]
    if det and max(det) == 0.0:
        log(f"[sanity] determinism_max_abs_delta_per_image is exactly 0.0 on "
            f"all {len(det)} cells checked. This is the expected MAMNet "
            f"behaviour (erratum E-9 records OGLANet, not MAMNet, as "
            f"nondeterministic), so it is recorded as a control that passed, "
            f"not as an implausibly clean result.")

    ctrl = sorted({c["probe"]["raw_pixel_control_accuracy"] for c in per_cell})
    if len(ctrl) == 1:
        log(f"[sanity] the raw-pixel control is bit-identical at "
            f"{ctrl[0]:.6f} on all {len(per_cell)} cells. Expected: the "
            f"control reads six statistics of the input image and touches no "
            f"model, so it cannot vary across cells evaluated on one fixed "
            f"image set. E-8 reports 0.871 for the same control.")
    else:
        flag(f"the raw-pixel control varies across cells ({len(ctrl)} distinct "
             f"values); E-8's model-independence argument predicts one value")

    for city in CITIES:
        vals = sorted({c["empty_fg_valid"] for c in per_cell
                       if c["city"] == city})
        uvals = sorted({c["empty_fg_valid_upper"] for c in per_cell
                        if c["city"] == city})
        if len(set(vals) | set(uvals)) != 1:
            flag(f"empty_fg_valid is not constant within {city}: LOCO {vals}, "
                 f"upper {uvals}")
        elif vals and vals[0] != EXPECTED_EMPTY_FG[city]:
            flag(f"empty_fg_valid on {city} is {vals[0]}, not the "
                 f"{EXPECTED_EMPTY_FG[city]} the arm spec carried")

    for c in per_cell:
        for m in METRICS:
            g = c["gaps"][m]
            if g["ci95"][0] is not None and not (
                    g["ci95"][0] <= g["mean"] <= g["ci95"][1]):
                flag(f"{c['run_id']} [{m}]: bootstrap CI "
                     f"[{g['ci95'][0]:.4f}, {g['ci95'][1]:.4f}] excludes its "
                     f"own point estimate {g['mean']:.4f}")
    ppf_worst, ppf_id = 0.0, None
    for meta in cells.values():
        if meta["pred_pos_fraction_stored"] is None:
            continue
        d = abs(float(meta["pred_pos_fraction_stored"])
                - meta["pred_pos_fraction"])
        if d > ppf_worst:
            ppf_worst, ppf_id = d, meta["run_id"]
    ok(ppf_worst < 1e-9,
       f"recomputed pred_pos_fraction (mean of per-image fractions) "
       f"reproduces the sidecar's stored value on every cell "
       f"(worst {ppf_worst:.3e}, {ppf_id})")

    # Levels stored in the sidecar vs recomputed from counts.
    worst, worst_id = 0.0, None
    for key, meta in cells.items():
        if meta["headline_2class_stored"] is None:
            continue
        d = abs(float(meta["headline_2class_stored"])
                - meta["level_pct"]["2class_tol_k2"])
        if d > worst:
            worst, worst_id = d, meta["run_id"]
    log(f"[sanity] max |delta| recomputed cell-mean two-class K=2 vs the "
        f"sidecar's headline_miou_2class_k2 = {worst:.3e} ({worst_id})")
    if worst > 1e-6:
        flag(f"recomputed cell-mean two-class K=2 differs from the sidecar's "
             f"headline by up to {worst:.6f} ({worst_id})")
    return {"determinism_max_delta_over_cells": (max(det) if det else None),
            "raw_pixel_control_distinct_values": ctrl,
            "pred_pos_fraction_vs_sidecar_max_abs_delta": ppf_worst,
            "pred_pos_fraction_vs_sidecar_max_abs_delta_run_id": ppf_id,
            "level_vs_sidecar_max_abs_delta": worst,
            "level_vs_sidecar_max_abs_delta_run_id": worst_id}


def flagged_cell_report(per_cell):
    """ssl_lo/miami/seed 2, reported next to its two siblings. Not dropped,
    not winsorised."""
    sibs = [c for c in per_cell
            if c["corner"] == "ssl_lo" and c["city"] == "miami"]
    if not sibs:
        return {"status": "NOT_IN_SCOPE",
                "note": "ssl_lo is not among the corners analysed in this run"}
    rows = []
    for c in sorted(sibs, key=lambda x: x["seed"]):
        rows.append({
            "run_id": c["run_id"], "seed": c["seed"],
            "is_flagged_cell": c["run_id"] == FLAGGED_CELL,
            "loco_level_pct": {m: c["levels_pct"][m]["loco"] for m in METRICS},
            "upper_level_pct": {m: c["levels_pct"][m]["upper"] for m in METRICS},
            "gap_pct": {m: c["gaps"][m]["mean"] for m in METRICS},
            "expA_median_R": {m: c["expA"]["R"][m]["median_R"] for m in METRICS},
            "expB_median_R": {m: c["expB"]["R"][m]["median_R"] for m in METRICS},
            "pred_pos_fraction": c["pred_pos_fraction"],
            "expA_readable": c["expA"].get("readable"),
            "expB_readable": c["expB"].get("readable"),
        })
    return {
        "status": "REPORTED_NOT_DROPPED",
        "cell": FLAGGED_CELL,
        "cc_slurm_flag": "56.77 against siblings 66.71 and 67.56, "
                         "pred_pos_fraction 0.2164 (2-3x its siblings), "
                         "training otherwise healthy",
        "treatment": "reported alongside its two siblings; not dropped, not "
                     "winsorised, not trimmed, and included in every pooled "
                     "number in this JSON",
        "siblings": rows,
    }


def empty_fg_report(per_cell):
    out = {}
    for city in CITIES:
        rows = [c for c in per_cell if c["city"] == city]
        if not rows:
            continue
        vals = sorted({c["empty_fg_valid"] for c in rows})
        out[city] = {
            "n_empty_fg_valid": vals[0] if len(vals) == 1 else vals,
            "n_test": N_TEST_EXPECTED,
            "fraction_of_150": (vals[0] / N_TEST_EXPECTED
                                if len(vals) == 1 else None),
            "percent_of_150": (round(100.0 * vals[0] / N_TEST_EXPECTED, 2)
                               if len(vals) == 1 else None),
            "identical_across_cells": len(vals) == 1,
            "n_cells_checked": len(rows) * 2,
        }
    out["scope_note"] = (
        "HIGH-RESOLUTION ONLY, and only the arm L held-out test pools "
        "(150 images per city). This is the quantity behind App E's unfilled "
        "'X.X%' (zNBu-M4, BxRx-W5), but the paper's placeholder covers all of "
        "ShadowTransfer, not the high-resolution arm L test pools alone. It "
        "must not be pasted into App E without that scope stated.")
    return out


def published_r34_control(per_cell, corners):
    """sup_hi's gap next to the published MAMNet-R34 vanilla HR gap, as the
    R34 -> R50 control reading."""
    if "sup_hi" not in corners:
        return {"status": "NOT_EVALUATED"}
    rows = []
    for city in CITIES:
        cells = [c for c in per_cell
                 if c["corner"] == "sup_hi" and c["city"] == city]
        pub = PUBLISHED_R34_TABLE38[city]
        pub_gap = pub["upper"] - pub["loco"]
        m = PRIMARY_METRIC
        loco = float(np.mean([c["levels_pct"][m]["loco"] for c in cells]))
        upper = float(np.mean([c["levels_pct"][m]["upper"] for c in cells]))
        gap = float(np.mean([c["gaps"][m]["mean"] for c in cells]))
        rows.append({
            "city": city,
            "published_r34_loco": pub["loco"], "published_r34_upper":
                pub["upper"], "published_r34_gap": round(pub_gap, 2),
            "armL_sup_hi_r50_loco": loco, "armL_sup_hi_r50_upper": upper,
            "armL_sup_hi_r50_gap": gap,
            "gap_divergence_r50_minus_r34": gap - pub_gap,
            "loco_divergence": loco - pub["loco"],
            "upper_divergence": upper - pub["upper"],
            "n_cells_averaged": len(cells),
        })
    return {
        "metric": PRIMARY_METRIC,
        "published_source": "Table 38, MAMNet-ResNet-34 vanilla high-res, "
                            "carried verbatim from the arm spec; this box does "
                            "not recompute the published R34 cells",
        "per_city": rows,
        "note": "divergence is reported as measured and is not smoothed. The "
                "published values are single printed numbers with no interval, "
                "so no test is run against them.",
    }


def probe_summary(per_cell, corners):
    out = {}
    for corner in corners:
        rows = [c for c in per_cell if c["corner"] == corner]
        acc = np.array([c["probe"]["feature_probe_accuracy"] for c in rows])
        ctrl = np.array([c["probe"]["raw_pixel_control_accuracy"]
                         for c in rows])
        d = acc - ctrl
        nm = np.array([c["probe"]["null_mean"] for c in rows])
        n99 = np.array([c["probe"]["null_p99"] for c in rows])
        out[corner] = {
            "feature_minus_raw_pixel_control": {
                "mean": float(d.mean()), "min": float(d.min()),
                "max": float(d.max()), "n_cells": len(rows)},
            "feature_probe_accuracy_context": {
                "mean": float(acc.mean()), "min": float(acc.min()),
                "max": float(acc.max())},
            "raw_pixel_control_context": {
                "mean": float(ctrl.mean()), "min": float(ctrl.min()),
                "max": float(ctrl.max())},
            "permutation_null_mean": float(nm.mean()),
            "permutation_null_p99_max": float(n99.max()),
            "n_cells_above_null_p99": int((acc > n99).sum()),
            "n_cells_above_raw_pixel_control": int((d > 0).sum()),
        }
    out["_reading_rule"] = (
        "E-8: the published city-identity probe taps the wrong tensor on all "
        "three architectures, and a raw-pixel control built from six input "
        "statistics reaches 0.871. No accuracy above chance carries a claim "
        "about the encoder. The only quantity worth reading here is "
        "feature_minus_raw_pixel_control; raw accuracies are context. The "
        "control is bit-identical across all cells because it is "
        "model-independent on a fixed image set, which matches E-8's own "
        "reported value.")
    return out


def histmatch_summary(per_cell, corners, pooled):
    rows = []
    for corner in corners:
        cells = [c for c in per_cell if c["corner"] == corner]
        for m in METRICS:
            rows.append({
                "corner": corner, "metric": m,
                "n_cells_improving": int(sum(
                    c["histmatch"]["R"][m]["improved"] for c in cells)),
                "n_cells": len(cells),
                "mean_R_over_cells": float(np.mean(
                    [c["histmatch"]["R"][m]["mean_R"] for c in cells])),
                "median_R_pooled": pooled["histmatch"][corner][m]["median_R"],
                "median_R_pooled_ci95":
                    pooled["histmatch"][corner][m]["ci95"],
                "n_used_pooled": pooled["histmatch"][corner][m]["n_used"],
                "n_excluded_pooled":
                    pooled["histmatch"][corner][m]["n_excluded"],
                "mean_delta_vs_loco_pct": float(np.mean(
                    [c["histmatch"]["R"][m]["mean_delta_vs_loco_pct"]
                     for c in cells])),
            })
    total = {m: int(sum(c["histmatch"]["R"][m]["improved"] for c in per_cell))
             for m in METRICS}
    return {"per_corner": rows,
            "n_cells_improving_total": total,
            "n_cells_total": len(per_cell),
            "rule": "the published Section 4.2 grayscale CDF match with the "
                    "per-pixel channel ratio preserved; 'improving' is the "
                    "cell's mean matched IoU above its mean LOCO IoU on the "
                    "same 150 images",
            "arm_C_comparator": "arm C found 0 of 26 usable cells improving at "
                                "every rung under the same rule"}


def selpred_summary(per_cell, corners):
    out = {}
    for corner in corners:
        cells = [c for c in per_cell if c["corner"] == corner]
        out[corner] = {}
        for defn in ("aurc_trapz19", "aurc_mean20"):
            out[corner][defn] = {
                pop: {"mean_over_cells": float(np.mean(
                          [c["selpred"]["aurc"][defn][pop] for c in cells])),
                      "min": float(np.min([c["selpred"]["aurc"][defn][pop]
                                           for c in cells])),
                      "max": float(np.max([c["selpred"]["aurc"][defn][pop]
                                           for c in cells])),
                      "n_cells": len(cells)}
                for pop in ("fg", "bg")}
    out["_definition_note"] = (
        "CONTRACT.md section 3 leaves the canonical AURC definition an OPEN "
        "BLANK. Both candidates are reported side by side and neither is "
        "chosen here: aurc_trapz19 is the trapezoid over the 19 coverage "
        "intervals, aurc_mean20 is the plain mean over the 20 grid points. "
        "They are not averaged together. Lower is better.")
    return out


def drift_table(per_cell, corners):
    """The per-stage drift table, four corners side by side."""
    stages = ("conv1", "layer1", "layer2", "layer3", "layer4", "whole_encoder")
    rows = []
    for s in stages:
        row = {"stage": s}
        for corner in corners:
            cells = [c for c in per_cell if c["corner"] == corner]
            v = np.array([c["drift"]["relative_l2"][s] for c in cells])
            row[corner] = {"relative_l2_mean": float(v.mean()),
                           "relative_l2_min": float(v.min()),
                           "relative_l2_max": float(v.max()),
                           "n_cells": len(cells)}
            if s in cells[0]["drift"]["cka"]:
                k = np.array([c["drift"]["cka"][s] for c in cells])
                row[corner].update({"cka_mean": float(k.mean()),
                                    "cka_min": float(k.min()),
                                    "cka_max": float(k.max())})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- envelope ---

def git_hash():
    for root in (os.path.join(BASE, "final_paper"), BASE):
        try:
            r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip(), f"git -C {root} rev-parse HEAD"
        except Exception:
            continue
    return None, ("no git repository at the project base; the project base is "
                  "not a git repo on this box and final_paper/ was not "
                  "reachable")


def derive_index_sha(cells, per_cell):
    """No INDEX.parquet exists under dumps/; digest the sorted
    run_id:parquet_sha256 list instead, following the arm C precedent."""
    h = hashlib.sha256()
    for c in sorted(cells.values(), key=lambda x: x["run_id"]):
        h.update((c["run_id"] + ":" + c["parquet_sha256"] + "\n").encode())
    return h.hexdigest()


METRIC_NOTE = (
    "Metric family: boundary-tolerant IoU at a +-2 px don't-care band (K=2), "
    "recomputed per image from the raw tp/fp/fn/tn counts. Two definitions are "
    "carried throughout. PRIMARY is the two-class average "
    "(2class_tol_k2 = mean of foreground and background IoU), which is the "
    "Table 38 / arm C family. Foreground-only K=2 (fg_only_tol_k2) is reported "
    "alongside and is the definition the dumps' own stored recovery ratios "
    "were computed on. These magnitudes are NOT comparable to Table 30, which "
    "erratum E-7 shows reproduces at K=5 on the two-class average, and the "
    "two-class numbers are NOT comparable to foreground-only numbers anywhere "
    "in the paper. Recovery ratios computed on the two definitions are also "
    "not comparable to each other: R is a ratio of IoU differences and its "
    "denominator changes with the definition, so the headroom exclusion "
    "selects a different image subset under each. Published Experiment B "
    "comparators (MAMNet-R34 0.88, OGLANet 0.86, DINOv3 -0.13) carry no stated "
    "metric definition in the ledger; the gate is evaluated under both and "
    "whether the two agree is recorded.")

CAVEATS = [
    "Arm L is high-resolution only and vanilla only. No adaptation method "
    "appears here and this results JSON must not be read as a method "
    "comparison.",
    "Arm L is one architecture (MAMNet with a ResNet-50 encoder) at rung 450, "
    "three held-out cities, three seeds. It is a 2x2 within that architecture, "
    "not a survey.",
    "Standing arm-L caveat, true under every outcome: DINO-R50 is ImageNet-1k "
    "(1.28M images) and DINOv3 is LVD-1689M. No public SSL CNN closes that gap "
    "at fixed capacity. The 2x2 separates pretraining objective and "
    "fine-tuning LR; pretraining SCALE remains confounded.",
    "Experiments A and B run under a 30-epoch, 420-step ceiling. Where that "
    "budget binds, R is a lower bound. Gate G3 applies the asymmetry "
    "explicitly and flags each cell readable or unreadable; unreadable cells "
    "are still reported and still enter the pooled numbers, so any pooled R "
    "that includes them inherits the ambiguity.",
    "Experiments A and B use held-out-city labels (target_access = "
    "target_labels). They are the Section 4.4 component-retraining "
    "experiments, not a deployable adaptation setting.",
    "The recovery ratio excludes images whose headroom (upper - LOCO) is below "
    "0.02, because R's denominator vanishes there. The exclusion count is "
    "reported per cell and per pooled group; it removes roughly a quarter to a "
    "third of the images on most cells, so a pooled median R describes the "
    "images that had recoverable headroom, not the whole test pool.",
    "The image-cluster bootstrap resamples images, so the same held-out image "
    "under three seeds is one unit. Intervals therefore do not treat seeds as "
    "independent replicates of an image.",
    "Per erratum E-8, no city-identity probe accuracy above chance carries a "
    "claim about the encoder. Only feature-probe minus raw-pixel control is "
    "read; raw accuracies are context.",
    "Per erratum E-10, Experiment B's encoder LR multiplier is recorded per "
    "cell rather than implied. App G.9's '100x smaller' text does not describe "
    "the published runs.",
    "CONTRACT.md section 3 leaves the canonical AURC definition an open blank. "
    "Both candidates are reported and neither is chosen; they are never "
    "averaged.",
    "The G3 readable rule needs a HIGH-R threshold, which the arm's "
    "pre-committed reading does not state separately. It is set to the same "
    "0.70 the signature table uses for CNN-like, and the count under the "
    "alternative 0.30 DINOv3-like boundary is reported alongside.",
    "Gate G2's second half — whether ssl_lo's drift sits nearer arm E's "
    "measured DINOv3 drift than sup_hi's does — is NOT evaluated. Arm E's "
    "drift numbers are not on this box. It is left blank rather than guessed.",
    "No INDEX.parquet exists under dumps/. index_sha256 is a sha256 over the "
    "sorted run_id:parquet_sha256 list of the cells analysed, following the "
    "arm C precedent.",
    "Two paths carrying .FAILED and no data are excluded by exact path and are "
    "listed in excluded_paths. Nothing else is excluded, winsorised or trimmed. "
    "The cell CC-SLURM flagged (" + FLAGGED_CELL + ") is reported next to its "
    "siblings and is included in every pooled number.",
    "empty_fg_valid is reported for the arm L high-resolution test pools only. "
    "App E's 'X.X%' placeholder covers all of ShadowTransfer, so this number "
    "cannot fill it without that scope stated.",
]


def build_envelope(per_cell, cells, pooled, contr_B, contr_A, gaps_by_corner,
                   g1, g2, g3, repro, prov, sanity, corners):
    gh, gh_reason = git_hash()
    m = PRIMARY_METRIC
    parts = [f"{corner} {pooled['expB'][corner][m]['median_R']:+.4f} "
             f"[{pooled['expB'][corner][m]['ci95'][0]:+.4f}, "
             f"{pooled['expB'][corner][m]['ci95'][1]:+.4f}]"
             for corner in corners]
    headline = (
        "Experiment B median recovery ratio R on MAMNet-ResNet-50 at rung 450, "
        "two-class boundary-tolerant K=2, pooled over 3 held-out cities x 3 "
        "seeds with an image-cluster bootstrap: " + "; ".join(parts) + ".")

    conditions = []
    for key in ("expA", "expB"):
        for corner in corners:
            for mm in METRICS:
                row = pooled[key][corner][mm]
                conditions.append({
                    "label": f"{corner} / Experiment {key[-1]} / median R / "
                             f"{mm}",
                    "metric": f"recovery_ratio_R_median_{mm}",
                    "mean": row["mean_R"], "median": row["median_R"],
                    "stderr": None, "n": row["n_used"],
                    "ci95": row["ci95"], "ci_method": row["ci_method"],
                    "baseline": "within-city upper bound (R = 1) and the LOCO "
                                "model itself (R = 0), both per image",
                })
    for corner in corners:
        for mm in METRICS:
            row = gaps_by_corner[corner][mm]
            conditions.append({
                "label": f"{corner} / LOCO-vs-upper gap / {mm}",
                "metric": row["metric"], "mean": row["mean"],
                "stderr": row["stderr"], "n": row["n"], "ci95": row["ci95"],
                "ci_method": row["ci_method"],
                "baseline": "within-city upper bound at the same corner, "
                            "city and seed",
            })

    tests = []
    for c in contr_B + contr_A:
        for mm in METRICS:
            r = c["by_metric"][mm]
            tests.append({
                "name": r["test"], "compares": c["compares"],
                "experiment": c["experiment"], "metric": mm,
                "statistic": r["mean"], "p": r["p_raw"],
                "p_holm": r["p_holm"], "holm_family": r["holm_family"],
                "B": r["B"], "two_tailed": True, "n": r["n"],
                "ci95": r["ci95"],
            })
    for c in per_cell:
        for mm in METRICS:
            g = c["gaps"][mm]
            tests.append({
                "name": g["test"], "compares": [c["run_id"],
                                                c["upper_run_id"]],
                "metric": mm, "statistic": g["mean"], "p": g["p_raw"],
                "p_holm": g["p_holm"], "holm_family": "per_cell_gaps",
                "B": B, "two_tailed": True, "n": g["n"], "ci95": g["ci95"],
            })

    clean = []
    for c in per_cell:
        d = {k: v for k, v in c.items() if not k.startswith("_")}
        d["selpred"] = {k: v for k, v in c["selpred"].items()
                        if not k.startswith("_")}
        clean.append(d)

    return {
        "id": "16_armL_ssl_regime",
        "arm": "L",
        "answers": ["zNBu-W5", "zNBu-W1", "AC1", "AC2"],
        "question": "Does a CNN put under DINOv3's regime — self-supervised "
                    "pretraining and low-LR fine-tuning — move the recoverable "
                    "failure component out of its encoder?",
        "runs": sorted({c["run_id"] for c in cells.values()}
                       | {f"{c['run_id']}-{p}" for c in per_cell
                          for p in PROBES}),
        "index_sha256": derive_index_sha(cells, per_cell),
        "index_sha256_note": "no INDEX.parquet exists under dumps/; this is "
                             "sha256 over the sorted "
                             "run_id:parquet_sha256 list of the 72 training "
                             "cells analysed",
        "generated_utc": datetime.now(timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": SCRIPT_LABEL,
        "git_hash": gh, "git_hash_reason": gh_reason,
        "conda_env": "myenv (conda env), "
                     "python 3.9.21, pandas 2.2.3, numpy 2.0.2, pyarrow 20.0.0, "
                     "PYTHONNOUSERSITE=1",

        "metric_note": METRIC_NOTE,
        "metric_definition": {
            "primary": PRIMARY_METRIC,
            "also_reported": "fg_only_tol_k2",
            "formula": "fg = tp_tol/(tp_tol+fp_tol+fn_tol+1e-10); "
                       "bg = tn_tol/(tn_tol+fp_tol+fn_tol+1e-10); "
                       "2class_tol_k2 = mean(fg,bg)",
            "band_width_px": 2,
            "source": "python/armc_ladder_stats.py:per_image_metrics, "
                      "documented as identical at "
                      "python/arme_frozen_gap_stats.py:138",
            "computed": "per image from raw counts, never from cell means",
            "recovery_ratio": "R = (retrained - LOCO) / (upper - LOCO) per "
                              "image on the IoU fraction scale, excluding "
                              f"images with headroom < {HEADROOM_MIN}",
        },

        "gates": {"G1": g1, "G2": g2, "G3": g3},
        "corner_medians": {
            "expB": {c: pooled["expB"][c] for c in corners},
            "expA": {c: pooled["expA"][c] for c in corners},
            "expB_by_city": pooled["expB_by_city"],
            "expA_by_city": pooled["expA_by_city"],
            "signature_table": {
                "cnn_like": {"expB_median_R": ">= 0.70",
                             "expA_median_R": "0.12 - 0.67"},
                "dinov3_like": {"expB_median_R": "< 0.30",
                                "expA_median_R": ">= 0.50"},
                "between": "anything else",
                "source": "REVIEWS.md, arm L pre-committed reading",
            },
            "published_comparators": PUBLISHED_EXPB,
        },
        "contrasts": {"expB_primary": contr_B, "expA_corroborating": contr_A},
        "gaps_by_corner": gaps_by_corner,
        "published_r34_control": published_r34_control(per_cell, corners),
        "city_identity_probe": probe_summary(per_cell, corners),
        "histogram_matching": histmatch_summary(per_cell, corners, pooled),
        "encoder_drift_table": drift_table(per_cell, corners),
        "selective_prediction": selpred_summary(per_cell, corners),
        "flagged_cell": flagged_cell_report(per_cell),
        "empty_fg_valid": empty_fg_report(per_cell),

        "cells": clean,
        "conditions": conditions,
        "tests": tests,

        "excluded_paths": [{"path": p, "reason": EXCLUDED_REASON}
                           for p in EXCLUDED_PATHS],
        "exclusions_beyond_named_paths": "none. No cell, seed or image is "
                                         "excluded, winsorised or trimmed "
                                         "beyond the per-image headroom rule "
                                         "on R, whose counts are reported.",
        "reproduction_check": repro,
        "provenance_check": prov,
        "sanity_checks": sanity,
        "sanity_flags": FLAGS,

        "headline": headline,
        "caveats": CAVEATS,
        "pre_committed_reading": {
            "source": "REVIEWS.md, section 'Arm L — SSL CNN under DINOv3's "
                      "fine-tuning regime', written 2026-07-31 before the "
                      "72-cell grid reported",
            "decision_variable": "Experiment B median recovery ratio R "
                                 "(encoder retraining)",
            "outcome_declared": None,
            "outcome_declared_note":
                "This box emits the numbers the gates and outcomes are defined "
                "on and states which outcomes they are consistent with. It "
                "does not declare which outcome landed. That call is the "
                "operator's.",
        },
        "figure": None,
        "figure_note": "no figure is produced by this analysis; CS builds "
                       "figures from this JSON",
        "caption": None,
        "log": LOG_LINES,
    }


# ---------------------------------------------------------------- outcomes ---

def classify(median_R_B, median_R_A):
    """Which signature row a corner's pair of medians sits in. Descriptive
    only; it applies the pre-committed table and declares no outcome."""
    if median_R_B is None:
        return "unclassified"
    cnn = median_R_B >= 0.70
    dino = median_R_B < 0.30
    if cnn:
        a_ok = (median_R_A is not None and 0.12 <= median_R_A <= 0.67)
        return "CNN-like" if a_ok else "CNN-like on Exp B, Exp A outside the " \
                                       "published 0.12-0.67 band"
    if dino:
        a_ok = (median_R_A is not None and median_R_A >= 0.50)
        return "DINOv3-like" if a_ok else "DINOv3-like on Exp B, Exp A below " \
                                          "the 0.50 the signature requires"
    return "between"


def outcome_consistency(pooled, corners, g1, g3, contr_B, alpha=0.05):
    """Which pre-committed outcomes the numbers are consistent with, and which
    they cannot distinguish.  No outcome is declared."""
    m = PRIMARY_METRIC

    # Outcome 4's second clause is 'no cell separates from sup_hi'.  Of the
    # three pre-committed contrasts only ssl_lo-vs-sup_hi is against sup_hi, so
    # that clause is evaluated on it, after Holm.
    sep_detail = [{
        "contrast": c["contrast"], "compares": c["compares"],
        "mean_delta_R": c["by_metric"][m]["mean"],
        "ci95": c["by_metric"][m]["ci95"],
        "p_holm": c["by_metric"][m]["p_holm"],
        "separates_at_alpha_0.05": bool(c["by_metric"][m]["p_holm"] is not None
                                        and c["by_metric"][m]["p_holm"] < alpha),
    } for c in contr_B if "sup_hi" in c["compares"]]
    sep_none = not any(s["separates_at_alpha_0.05"] for s in sep_detail)
    sig = {c: {
        "expB_median_R": pooled["expB"][c][m]["median_R"],
        "expB_ci95": pooled["expB"][c][m]["ci95"],
        "expA_median_R": pooled["expA"][c][m]["median_R"],
        "expA_ci95": pooled["expA"][c][m]["ci95"],
        "signature": classify(pooled["expB"][c][m]["median_R"],
                              pooled["expA"][c][m]["median_R"]),
    } for c in corners}

    if g1.get("status") != "PASS":
        return {
            "gate_G1_status": g1.get("status"),
            "signatures": sig,
            "consistent_with": [],
            "cannot_distinguish": ["1", "2", "3", "4"],
            "statement":
                "G1 failed: sup_hi's Experiment B median R is below the 0.70 "
                "the pre-committed reading requires for CNN-like. The ledger "
                "fixes the consequence in advance — 'the R34 -> R50 swap broke "
                "the comparison, nothing in L is interpretable, report the "
                "swap as a finding, do not read the 2x2'. The four outcomes "
                "are therefore all undistinguished, and the corner medians "
                "below are emitted as measurements, not as a reading of the "
                "2x2.",
        }

    is_dino = {c: sig[c]["signature"].startswith("DINOv3-like") for c in sig}
    is_cnn = {c: sig[c]["signature"].startswith("CNN-like") for c in sig}

    why = {"1": "ssl_lo DINOv3-like; ssl_hi also DINOv3-like; sup_lo CNN-like "
                "-> pretraining drives the locus",
           "2": "ssl_lo DINOv3-like; sup_lo also DINOv3-like; ssl_hi CNN-like "
                "-> fine-tuning LR drives the locus",
           "3": "ssl_lo DINOv3-like; ssl_hi and sup_lo both CNN-like -> "
                "interaction, both ingredients needed",
           "4": "ssl_lo CNN-like, no cell separates from sup_hi -> "
                "architecture stands"}

    # Every outcome is decomposed into the clauses the pre-committed table
    # states, and each clause is evaluated on its own.  An outcome whose
    # clauses split is reported as split rather than forced into a verdict.
    clauses = {
        "1": {"ssl_lo is DINOv3-like": is_dino.get("ssl_lo"),
              "ssl_hi is DINOv3-like": is_dino.get("ssl_hi"),
              "sup_lo is CNN-like": is_cnn.get("sup_lo")},
        "2": {"ssl_lo is DINOv3-like": is_dino.get("ssl_lo"),
              "sup_lo is DINOv3-like": is_dino.get("sup_lo"),
              "ssl_hi is CNN-like": is_cnn.get("ssl_hi")},
        "3": {"ssl_lo is DINOv3-like": is_dino.get("ssl_lo"),
              "ssl_hi is CNN-like": is_cnn.get("ssl_hi"),
              "sup_lo is CNN-like": is_cnn.get("sup_lo")},
        "4": {"ssl_lo is CNN-like": is_cnn.get("ssl_lo"),
              "no corner separates from sup_hi on the Exp B contrasts":
                  bool(sep_none)},
    }
    # ssl_lo is the arm's decision cell: every outcome is defined first by what
    # ssl_lo does, and the other clauses only say which ingredient explains it.
    # A clause split that fails on ssl_lo is therefore not the same kind of
    # partial match as one that fails on a supporting corner.
    load_bearing = {"1": "ssl_lo is DINOv3-like", "2": "ssl_lo is DINOv3-like",
                    "3": "ssl_lo is DINOv3-like", "4": "ssl_lo is CNN-like"}
    lb_holds = {k: bool(clauses[k][load_bearing[k]]) for k in clauses}

    cons = [k for k, v in clauses.items() if all(v.values())]
    inc = [k for k, v in clauses.items() if not any(v.values())]
    split = [k for k, v in clauses.items()
             if k not in cons and any(v.values())]

    unread = [c["run_id"] for c in g3["per_cell"] if not c["expB"]["readable"]]
    return {
        "gate_G1_status": g1.get("status"),
        "signatures": sig,
        "outcome_definitions": why,
        "outcome_clauses": clauses,
        "load_bearing_clause": load_bearing,
        "load_bearing_clause_holds": lb_holds,
        "load_bearing_note":
            "ssl_lo is the arm's decision cell. Outcomes 1, 2 and 3 each "
            "require ssl_lo to be DINOv3-like on Experiment B (median R < "
            "0.30); outcome 4 requires it to be CNN-like (>= 0.70). The "
            "supporting clauses only name which ingredient explains ssl_lo, so "
            "a split that fails on ssl_lo is a different thing from one that "
            "fails on a supporting corner. Experiment B is also the readable "
            "axis: the decision does not rest on Experiment A, which G3 marks "
            "unreadable on most cells.",
        "consistent_with": cons,
        "inconsistent_with": inc,
        "clauses_split": split,
        "cannot_distinguish": [o for o in ("1", "2", "3", "4")
                               if o not in cons],
        "separating_contrasts_vs_sup_hi": sep_detail,
        "unreadable_expB_cells": unread,
        "n_unreadable_expB_cells": len(unread),
        "statement": "The signature assignment above applies the "
                     "pre-committed table to the pooled corner medians, clause "
                     "by clause. Outcomes listed under inconsistent_with have "
                     "no clause satisfied; outcomes under clauses_split have "
                     "some but not all. Which outcome landed is the operator's "
                     "call and is not declared here.",
    }


# ---------------------------------------------------------------- summary ----

def write_summary_md(env, pooled, contr_B, corners, outcomes, path):
    m = PRIMARY_METRIC
    L = []
    A = L.append
    A("# Arm L — SSL CNN under DINOv3's fine-tuning regime")
    A("")
    A(f"Results JSON: `results/16_armL_ssl_regime.json`  ")
    A(f"Generated {env['generated_utc']} by `{SCRIPT_LABEL}`  ")
    A(f"Answers zNBu-W5 (primary), zNBu-W1, AC1, AC2.")
    A("")
    A("MAMNet-ResNet-50, rung 450, 3 held-out cities x 3 seeds, "
      "high-resolution only, vanilla only. 36 LOCO cells and their 36 matched "
      "within-city upper bounds. Every number is recomputed from per-image "
      "values.")
    A("")
    A("Metric: two-class boundary-tolerant mIoU at K=2 is primary; "
      "foreground-only K=2 is reported alongside in the JSON. These magnitudes "
      "are **not** comparable to Table 30 (K=5) or to foreground-only numbers.")
    A("")

    A("## Validity gates")
    A("")
    A("| Gate | Test | Number | Status |")
    A("|---|---|---|---|")
    g1 = env["gates"]["G1"]
    if g1.get("status") in ("PASS", "FAIL"):
        b = g1["by_metric"][m]
        fgb = g1["by_metric"]["fg_only_tol_k2"]
        A(f"| G1 | sup_hi Exp B median R >= {G1_THRESHOLD} (CNN-like) | "
          f"two-class **{b['median_R']:+.4f}** [{b['ci95'][0]:+.4f}, "
          f"{b['ci95'][1]:+.4f}], n={b['n_used']}; foreground-only "
          f"{fgb['median_R']:+.4f} [{fgb['ci95'][0]:+.4f}, "
          f"{fgb['ci95'][1]:+.4f}] | **{g1['status']}** |")
    else:
        A(f"| G1 | sup_hi Exp B median R >= {G1_THRESHOLD} | — | "
          f"{g1.get('status')} |")
    g2 = env["gates"]["G2"]
    comp = g2.get("components", {})
    A(f"| G2 | _lo corners show smaller encoder drift than _hi | relative L2 "
      f"separates with no overlap at every stage: "
      f"{comp.get('relative_l2_separates_no_overlap')}; CKA orders them the "
      f"same way in the mean at every stage: "
      f"{comp.get('cka_orders_correctly_in_mean')}, but separates with no "
      f"overlap: {comp.get('cka_separates_no_overlap')} | **{g2['status']}** "
      f"(arm E comparison NOT EVALUATED — arm E drift is not on this box) |")
    g3 = env["gates"]["G3"]
    cts = g3["counts"]
    A(f"| G3 | budget binds; high R still valid, low R unreadable | Exp A "
      f"binding {cts.get('expA_budget_bound', 0)}/{g3['n_cells']}, Exp B "
      f"binding {cts.get('expB_budget_bound', 0)}/{g3['n_cells']}; Exp B "
      f"unreadable {cts.get('expB_unreadable', 0)}/{g3['n_cells']}, Exp A "
      f"unreadable {cts.get('expA_unreadable', 0)}/{g3['n_cells']} | "
      f"applied |")
    A("")
    A(f"G3 per corner (unreadable cells of 9): " + ", ".join(
        f"{c} Exp B {g3['counts_by_corner'][c]['expB']['n_unreadable']}, "
        f"Exp A {g3['counts_by_corner'][c]['expA']['n_unreadable']}"
        for c in corners if c in g3["counts_by_corner"]) + ". "
      + g3["readable_axis_note"])
    A("")
    if g1.get("status") == "FAIL":
        A("> **G1 failed.** The pre-committed reading fixes the consequence in "
          "advance: the R34 -> R50 swap broke the comparison, nothing in the "
          "2x2 is interpretable, and the swap itself is the finding. The "
          "corner medians below are emitted as measurements, not as a reading "
          "of the 2x2.")
        A("")

    A("### G2 — encoder drift per stage, four corners side by side")
    A("")
    hdr = "| Stage | " + " | ".join(
        f"{c} rel L2 | {c} CKA" for c in corners) + " |"
    A(hdr)
    A("|" + "---|" * (1 + 2 * len(corners)))
    for row in env["encoder_drift_table"]:
        cells = []
        for c in corners:
            r = row.get(c, {})
            cells.append(f"{r.get('relative_l2_mean', float('nan')):.4f}")
            cells.append(f"{r['cka_mean']:.4f}" if "cka_mean" in r else "—")
        A(f"| {row['stage']} | " + " | ".join(cells) + " |")
    A("")
    A("Mean over the 9 cells of each corner. Relative L2 is "
      "‖W_ft − W_init‖ / ‖W_init‖ against that cell's own pretrained "
      "initialisation; CKA is linear CKA between the initial and fine-tuned "
      "encoder on the held-out city's 150 test images. Lower relative L2 and "
      "higher CKA mean the encoder moved less.")
    A("")

    A("## Four corner medians — Experiment B (decision variable)")
    A("")
    A("Median per-image recovery ratio R, pooled over 3 cities x 3 seeds, "
      "image-cluster bootstrap 95% CI, B = 10,000.")
    A("")
    A("| Corner | pretraining | LR | Exp B median R (two-class) | 95% CI | "
      "Exp A median R | 95% CI | n images used | signature |")
    A("|---|---|---|---|---|---|---|---|---|")
    lab = {"sup_hi": ("supervised", "high"), "sup_lo": ("supervised", "low"),
           "ssl_hi": ("SSL", "high"), "ssl_lo": ("SSL", "low")}
    for c in corners:
        b, a = pooled["expB"][c][m], pooled["expA"][c][m]
        s = outcomes["signatures"][c]["signature"]
        A(f"| {c} | {lab[c][0]} | {lab[c][1]} | **{b['median_R']:+.4f}** | "
          f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] | "
          f"{a['median_R']:+.4f} | [{a['ci95'][0]:+.4f}, "
          f"{a['ci95'][1]:+.4f}] | {b['n_used']} | {s} |")
    A("")
    A("Published comparators, Experiment B: MAMNet-R34 0.88 [0.83, 0.90], "
      "OGLANet 0.86 [0.81, 0.89], DINOv3 −0.13 [−0.27, −0.03]. Signature "
      "bands: CNN-like is Exp B ≥ 0.70 with Exp A in 0.12–0.67; DINOv3-like is "
      "Exp B < 0.30 with Exp A ≥ 0.50.")
    A("")

    A("## Three contrasts — Experiment B, per-image R deltas")
    A("")
    A("Paired bootstrap on per-image R deltas, image-cluster resampling, "
      "B = 10,000, Holm over the three.")
    A("")
    A("| Contrast | What it isolates | Mean ΔR | 95% CI | test | p_raw | "
      "p_holm | n |")
    A("|---|---|---|---|---|---|---|---|")
    for c in contr_B:
        r = c["by_metric"][m]
        A(f"| {c['compares'][0]} vs {c['compares'][1]} | {c['note']} | "
          f"{r['mean']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
          f"paired bootstrap | {r['p_raw']:.4f} | {r['p_holm']:.4f} | "
          f"{r['n']} |")
    A("")

    A("## Pre-committed outcomes")
    A("")
    A("Each outcome is decomposed into the clauses the pre-committed table "
      "states, and each clause is evaluated on its own. The load-bearing "
      "clause is the one about ssl_lo, the arm's decision cell; the others "
      "only name which ingredient explains it.")
    A("")
    A("| # | Pattern | Load-bearing clause (ssl_lo) | Supporting clauses | "
      "Status against these numbers |")
    A("|---|---|---|---|---|")
    defs = outcomes.get("outcome_definitions") or {}
    clauses = outcomes.get("outcome_clauses", {})
    lb = outcomes.get("load_bearing_clause", {})
    for k in ("1", "2", "3", "4"):
        cl = clauses.get(k, {})
        lbname = lb.get(k)
        lbtxt = (f"{'**HOLDS**' if cl.get(lbname) else '**FAILS**'} — {lbname}"
                 if lbname else "—")
        sup = "; ".join(f"{'holds' if v else 'fails'} — {name}"
                        for name, v in cl.items() if name != lbname) or "—"
        if k in outcomes.get("consistent_with", []):
            status = "**consistent — every clause holds**"
        elif k in outcomes.get("inconsistent_with", []):
            status = "inconsistent — no clause holds"
        elif not cl.get(lbname):
            status = "**not consistent — fails on the decision cell**"
        else:
            status = "clauses split — decision cell holds, a supporting " \
                     "clause does not"
        A(f"| {k} | {defs.get(k, '')} | {lbtxt} | {sup} | {status} |")
    A("")
    for s in outcomes.get("separating_contrasts_vs_sup_hi", []):
        A(f"Outcome 4's second clause, on `{s['contrast']}`: mean ΔR "
          f"{s['mean_delta_R']:+.4f} [{s['ci95'][0]:+.4f}, "
          f"{s['ci95'][1]:+.4f}], p_holm {s['p_holm']:.4f} — "
          f"{'separates' if s['separates_at_alpha_0.05'] else 'does not separate'} "
          f"at α = 0.05.")
    A("")
    A(outcomes["statement"])
    A("")
    A("**No outcome is declared here.** The call is the operator's.")
    A("")

    A("## Also emitted")
    A("")
    hm = env["histogram_matching"]
    A(f"- Section 4.2 histogram matching: "
      f"{hm['n_cells_improving_total'][m]} of {hm['n_cells_total']} cells "
      f"improve under the published grayscale rule at two-class K=2 "
      f"({hm['n_cells_improving_total']['fg_only_tol_k2']} of "
      f"{hm['n_cells_total']} foreground-only). Arm C's comparator: 0 of 26 "
      f"usable cells improving at every rung.")
    pr = env["city_identity_probe"]
    A("- City-identity probe (E-8 scoped): feature-probe minus raw-pixel "
      "control by corner — " + ", ".join(
          f"{c} {pr[c]['feature_minus_raw_pixel_control']['mean']:+.4f}"
          for c in corners) + ". Raw accuracies are context only.")
    A("- Selective prediction: AURC reported under BOTH candidate definitions "
      "(trapezoid over 19 intervals, mean over 20 grid points), foreground and "
      "background, per corner. Neither is chosen; CONTRACT.md §3's AURC "
      "definition is an open blank.")
    ef = env["empty_fg_valid"]
    A("- empty_fg_valid, high-resolution arm L test pools only: " + ", ".join(
        f"{c} {ef[c]['n_empty_fg_valid']}/150 ({ef[c]['percent_of_150']}%)"
        for c in CITIES if c in ef) + ".")
    fc = env["flagged_cell"]
    if fc.get("status") == "REPORTED_NOT_DROPPED":
        A(f"- {FLAGGED_CELL} (CC-SLURM flagged) is reported next to its two "
          f"siblings and is included in every pooled number; not dropped, not "
          f"winsorised.")
    A("")
    A("## Excluded paths")
    A("")
    for p in EXCLUDED_PATHS:
        A(f"- `{p}` — {EXCLUDED_REASON}")
    A("")
    A("Nothing else is excluded. The per-image headroom rule on R "
      f"(upper − LOCO < {HEADROOM_MIN}) is a definitional exclusion, and its "
      "counts are reported per cell and per pooled group in the JSON.")
    A("")
    if FLAGS:
        A("## Sanity flags")
        A("")
        for f in FLAGS:
            A(f"- {f}")
        A("")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[write] {path}  ({os.path.getsize(path)} bytes)")


def print_screen_summary(pooled, contr_B, corners, g1, g3, gaps_by_corner):
    m = PRIMARY_METRIC
    print("\n" + "=" * 78)
    print("ARM L — ONE-SCREEN SUMMARY  (metric: two-class tolerant K=2)")
    print("=" * 78)
    print("\nGate G1 — sup_hi Experiment B median R vs the 0.70 CNN-like "
          "threshold")
    if g1.get("status") in ("PASS", "FAIL"):
        for mm in METRICS:
            b = g1["by_metric"][mm]
            print(f"  {mm:<16} {b['median_R']:+.4f} "
                  f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  n={b['n_used']}"
                  f"  -> {'PASS' if b['passes'] else 'FAIL'}")
        print(f"  status ({PRIMARY_METRIC}): {g1['status']}   "
              f"metrics agree: {g1['agrees_across_metrics']}")
    else:
        print(f"  {g1.get('status')}")

    print("\nCorner medians (pooled 3 cities x 3 seeds, image-cluster "
          "bootstrap)")
    print(f"  {'corner':<8} {'ExpB med R':>11} {'95% CI':>22} "
          f"{'ExpA med R':>11} {'95% CI':>22} {'n':>6}")
    print("  " + "-" * 84)
    for c in corners:
        b, a = pooled["expB"][c][m], pooled["expA"][c][m]
        print(f"  {c:<8} {b['median_R']:>+11.4f} "
              f"[{b['ci95'][0]:>+9.4f},{b['ci95'][1]:>+9.4f}] "
              f"{a['median_R']:>+11.4f} "
              f"[{a['ci95'][0]:>+9.4f},{a['ci95'][1]:>+9.4f}] "
              f"{b['n_used']:>6}")

    if contr_B:
        print("\nContrasts (Experiment B, per-image R deltas, Holm over 3)")
        print(f"  {'contrast':<20} {'mean dR':>9} {'95% CI':>22} "
              f"{'p_raw':>8} {'p_holm':>8} {'n':>6}")
        print("  " + "-" * 78)
        for c in contr_B:
            r = c["by_metric"][m]
            print(f"  {c['contrast']:<20} {r['mean']:>+9.4f} "
                  f"[{r['ci95'][0]:>+9.4f},{r['ci95'][1]:>+9.4f}] "
                  f"{r['p_raw']:>8.4f} {r['p_holm']:>8.4f} {r['n']:>6}")

    print("\nGate G3 — budget")
    cts = g3["counts"]
    print(f"  Exp A binding {cts.get('expA_budget_bound', 0)}/{g3['n_cells']} "
          f"(CC-SLURM reported 29/36); unreadable "
          f"{cts.get('expA_unreadable', 0)}")
    print(f"  Exp B binding {cts.get('expB_budget_bound', 0)}/{g3['n_cells']} "
          f"(CC-SLURM reported 26/36); unreadable "
          f"{cts.get('expB_unreadable', 0)}")

    print("\nLOCO-vs-upper gap (upper minus LOCO, percentage points)")
    for c in corners:
        r = gaps_by_corner[c][m]
        print(f"  {c:<8} mean {r['mean']:+7.4f} "
              f"[{r['ci95'][0]:+7.4f},{r['ci95'][1]:+7.4f}]  "
              f"worst cell {r['worst_case_gap']:+7.4f} "
              f"({r['worst_case_cell']})")
    print("=" * 78 + "\n")


# ------------------------------------------------------------------- main ----

def main():
    log(f"[start] {datetime.now(timezone.utc).isoformat()}  STAGE={STAGE}")
    log(f"[start] B={B} boot_seed={BOOT_SEED} headroom_min={HEADROOM_MIN}")
    training, probes = assert_analysis_set()
    if STAGE == "assert":
        log("[stage] STAGE='assert': analysis set validated, stopping before "
            "any statistic, as the run order requires.")
        return 0

    corners = tuple(DRY_CORNERS) if STAGE == "dry" else CORNERS
    log("")
    log("=" * 78)
    log(f"STAGE 2/3 — analysis over corners {corners}")
    log("=" * 78)

    cells = build_cells(training, corners)
    per_cell = build_per_cell(cells, corners)
    repro = reproduction_check(per_cell)
    prov = provenance_check(cells, per_cell)
    sanity = sanity_checks(per_cell, cells)

    pooled = pool_R(per_cell, corners)
    g1 = gate_G1(pooled, per_cell)
    g2 = gate_G2(per_cell, corners)
    g3 = gate_G3(per_cell)
    per_cell_caveats(per_cell)
    contr_B = contrasts(per_cell, corners, "expB")
    contr_A = contrasts(per_cell, corners, "expA")
    gaps_by_corner = corner_gaps(per_cell, corners)

    # Holm over the 36 per-cell gaps, per metric definition.
    for mm in METRICS:
        adj = holm([c["gaps"][mm]["p_raw"] for c in per_cell])
        for c, a in zip(per_cell, adj):
            c["gaps"][mm]["p_holm"] = a
            c["gaps"][mm]["holm_family"] = f"arm_L_per_cell_gaps_{len(per_cell)}"

    outcomes = outcome_consistency(pooled, corners, g1, g3, contr_B)
    env = build_envelope(per_cell, cells, pooled, contr_B, contr_A,
                         gaps_by_corner, g1, g2, g3, repro, prov, sanity,
                         corners)
    env["outcome_consistency"] = outcomes
    env["pre_committed_reading"]["consistent_with"] = \
        outcomes["consistent_with"]
    env["pre_committed_reading"]["cannot_distinguish"] = \
        outcomes["cannot_distinguish"]

    print_screen_summary(pooled, contr_B, corners, g1, g3, gaps_by_corner)

    if STAGE == "dry":
        print(f"[dry-run] corners analysed: {corners}")
        print(f"[dry-run] would write {OUT_JSON} and {OUT_MD}")
        print(f"[dry-run] envelope keys: {sorted(env.keys())}")
        print(f"[dry-run] cells={len(env['cells'])} "
              f"conditions={len(env['conditions'])} tests={len(env['tests'])}")
        print("[dry-run] nothing written")
        return 0

    ok(set(corners) == set(CORNERS),
       f"the full pass covers all four corners ({sorted(corners)})")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(env, fh, indent=2, default=float)
    print(f"[write] {OUT_JSON}  ({os.path.getsize(OUT_JSON)} bytes)")
    write_summary_md(env, pooled, contr_B, corners, outcomes, OUT_MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
