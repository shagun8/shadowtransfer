#!/usr/bin/env python3
"""ARM C — scaling ladder statistics.  Answers AC1 and zNBu-W3.

CC-SSH analysis script.  Reads the 135 arm C ladder dumps (perimage.parquet +
run.json), computes per-image two-class boundary-tolerant mIoU at K=2 from raw
counts, forms LOCO-versus-upper gaps paired on image_id, and writes one
CONTRACT.md section 6 results JSON.

Runs on CPU in minutes.  No GPU, no network, no writes outside --repo-root.

The metric is the published two-class decision metric, matching
p0a_recompute.py:miou_from_counts verbatim:
    fg = tp / (tp + fp + fn + EPS_IOU)
    bg = tn / (tn + fp + fn + EPS_IOU)
    value = mean(fg, bg), in percent.

The bootstrap replicates the published protocol in
python/statistical_analysis.py:bootstrap_comparison verbatim: B = 10,000
resamples of the per-image delta vector, percentile CI at 2.5 / 97.5, and a
two-tailed p of min(2 * prop_wrong, 1.0).  It is reimplemented here rather than
imported because that module is frozen legacy.
"""

import argparse
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

EPS_IOU = 1e-10
B_DEFAULT = 10000
BOOT_SEED = 42
# 0.50 is justified by the GT foreground fraction, not by authorisation.
# GT shadow occupies at most 0.086 of the valid region on any cell (chicago,
# mean over images; pooled 0.0839; miami 0.080, phoenix 0.035). Foreground
# IoU is bounded above by gt_fg / pred_pos, so a cell at pred-positive >= 0.50
# has fg IoU <= 0.086 / 0.50 = 17.2 by construction. The rule therefore cannot
# catch a working model at any threshold >= 0.50, which is what makes it safe.
# The spec's literal 0.99 detects only 1 of the 4 known collapsed cells.
# Measured pooled pred-positive fractions: the 4 known sit at 0.6483-0.9986
# with fg IoU 3.51-8.02; the highest non-collapsed cell is 0.1490 at 50.93 fg
# IoU, and the remaining 130 cells fall below it. Any threshold in
# (0.15, 0.65) yields the identical exclusion set; 0.50 sits mid-band.
# Override with --collapse-threshold to reproduce a different exclusion.
COLLAPSE_THRESHOLD = 0.50
N_TEST_EXPECTED = 150
N_CELLS_EXPECTED = 135

ARCHS = ("mamnet", "oglanet", "dinov3")
CITIES = ("chicago", "miami", "phoenix")
RUNGS = (225, 450, 900)
SEEDS = (0, 1, 2)

# Rung 900 has no matched upper bound: the held-out city holds exactly 450
# training images at high resolution, so upper-900 is unbuildable.  Its
# comparator is upper-450 and the quantity is the `budget-asymmetric gap`.
COMPARATOR_RUNG = {225: 225, 450: 450, 900: 450}
BUDGET_ASYMMETRIC = {225: False, 450: False, 900: True}

KNOWN_COLLAPSED = (
    "dinov3-none-upper-r225-phoenix-hr-s0",
    "dinov3-vanilla-loco-r225-miami-hr-s0",
    "dinov3-none-upper-r225-miami-hr-s0",
    "oglanet-none-upper-r225-phoenix-hr-s2",
)

REQUIRED_COUNT_COLS = ("tp_tol", "fp_tol", "fn_tol", "tn_tol",
                       "n_valid_px", "n_pred_pos", "image_id")

RUN_ID_RE = re.compile(
    r"^(?P<arch>[a-z0-9]+)-(?P<method>[a-z_]+)-(?P<mode>loco|upper)"
    r"-r(?P<rung>\d+)-(?P<city>[a-z]+)-(?P<res>hr|mr)-s(?P<seed>\d+)$")

LOG_LINES = []
INCOMPLETE = []


def log(msg):
    """Every assertion outcome goes through here, pass or fail."""
    line = str(msg)
    LOG_LINES.append(line)
    print(line, flush=True)


def ok(cond, msg):
    """Log an assertion outcome either way, then abort on failure."""
    log(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise AssertionError(msg)
    return cond


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- metric ----

def per_image_metrics(df):
    """Two-class and foreground-only tolerant IoU at K=2, per image, percent.

    Verbatim denominators from p0a_recompute.py:miou_from_counts.  Computed per
    image and never from cell means.
    """
    tp = df["tp_tol"].to_numpy(dtype=np.float64)
    fp = df["fp_tol"].to_numpy(dtype=np.float64)
    fn = df["fn_tol"].to_numpy(dtype=np.float64)
    tn = df["tn_tol"].to_numpy(dtype=np.float64)
    fg = tp / (tp + fp + fn + EPS_IOU) * 100.0
    bg = tn / (tn + fp + fn + EPS_IOU) * 100.0
    return fg, bg, (fg + bg) / 2.0


# ------------------------------------------------------------- discovery ----

def parse_run_id(run_id):
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None
    d = m.groupdict()
    d["rung"] = int(d["rung"])
    d["seed"] = int(d["seed"])
    return d


def discover_cells(dumps_root, archs, expect):
    """Enumerate arm C ladder cells under dumps_root.

    dinov3frozen is arm E/F and is excluded by name, as are smoke cells.
    """
    if not os.path.isdir(dumps_root):
        raise SystemExit(f"dumps root does not exist: {dumps_root}")
    names = sorted(os.listdir(dumps_root))
    kept, excluded_frozen, excluded_other = [], [], []
    for name in names:
        if not os.path.isdir(os.path.join(dumps_root, name)):
            continue
        if name.startswith("dinov3frozen"):
            excluded_frozen.append(name)
            continue
        if name.endswith("-smoke"):
            excluded_other.append(name)
            continue
        meta = parse_run_id(name)
        if meta is None or meta["res"] != "hr" or meta["rung"] not in RUNGS:
            excluded_other.append(name)
            continue
        if meta["arch"] not in ARCHS or meta["city"] not in CITIES:
            excluded_other.append(name)
            continue
        meta["run_id"] = name
        meta["dir"] = os.path.join(dumps_root, name)
        kept.append(meta)

    log(f"[discover] dumps_root = {dumps_root}")
    log(f"[discover] arm C ladder cells found: {len(kept)}")
    log(f"[discover] excluded dinov3frozen (arm F): {len(excluded_frozen)}")
    if excluded_frozen:
        log(f"[discover]   {sorted(excluded_frozen)}")
    log(f"[discover] excluded other (smoke / off-grid): {len(excluded_other)}")
    if excluded_other:
        log(f"[discover]   {sorted(excluded_other)}")

    ok(len(kept) == expect,
       f"arm C ladder cell count is {len(kept)}, expected {expect}")

    if archs != list(ARCHS):
        kept = [c for c in kept if c["arch"] in archs]
        log(f"[discover] architecture filter {archs} -> {len(kept)} cells")
    return kept


# ----------------------------------------------------------------- load -----

def load_cell(meta, skip_incomplete=False):
    """Read one cell's parquet and sidecar; assert schema and n_test.

    Returns None when skip_incomplete is set and an artefact is missing, so one
    unwritten cell does not cost the other 134.
    """
    run_id = meta["run_id"]
    pq_path = os.path.join(meta["dir"], "perimage.parquet")
    rj_path = os.path.join(meta["dir"], "run.json")
    for p in (pq_path, rj_path):
        present = os.path.exists(p) and os.path.getsize(p) > 0
        if skip_incomplete and not present:
            log(f"  SKIP  {run_id}: {os.path.basename(p)} missing or empty; "
                f"cell dropped (--skip-incomplete)")
            INCOMPLETE.append({"run_id": run_id,
                               "missing": os.path.basename(p)})
            return None
        ok(present, f"{run_id}: {os.path.basename(p)} present and non-empty")

    df = pd.read_parquet(pq_path)
    missing = [c for c in REQUIRED_COUNT_COLS if c not in df.columns]
    ok(not missing, f"{run_id}: required columns present "
                    f"({'missing ' + str(missing) if missing else 'all'})")

    with open(rj_path) as fh:
        run = json.load(fh)

    ok(len(df) == N_TEST_EXPECTED,
       f"{run_id}: parquet rows = {len(df)} (expected {N_TEST_EXPECTED})")
    ok(int(run.get("n_test", -1)) == N_TEST_EXPECTED,
       f"{run_id}: run.json n_test = {run.get('n_test')} "
       f"(expected {N_TEST_EXPECTED})")
    ok(df["image_id"].is_unique, f"{run_id}: image_id is unique")

    for key, want in (("arch", meta["arch"]), ("mode", meta["mode"]),
                      ("holdout", meta["city"]), ("seed", meta["seed"]),
                      ("rung", meta["rung"])):
        got = run.get(key)
        ok(str(got) == str(want),
           f"{run_id}: run.json {key} = {got!r} matches run_id ({want!r})")

    df = df.sort_values("image_id").reset_index(drop=True)
    fg, bg, two = per_image_metrics(df)

    n_pred_pos = df["n_pred_pos"].to_numpy(dtype=np.float64)
    n_valid = df["n_valid_px"].to_numpy(dtype=np.float64)
    # Cell-level predicted-positive fraction, pooled over pixels.
    pred_pos_fraction = float(n_pred_pos.sum() / max(n_valid.sum(), 1.0))

    # target_access is stored per cell; carried through, never derived.
    target_access = run.get("target_access")
    ok(target_access in ("source_only", "unlabeled_target",
                         "deployment_time", "target_labels"),
       f"{run_id}: target_access = {target_access!r} is a CONTRACT value")

    return {
        "run_id": run_id,
        "arch": meta["arch"], "city": meta["city"], "rung": meta["rung"],
        "mode": meta["mode"], "seed": meta["seed"],
        "target_access": target_access,
        "target_access_detail": run.get("target_access_detail"),
        "n_test": int(len(df)),
        "image_ids": df["image_id"].to_numpy(),
        "fg": fg, "bg": bg, "two_class": two,
        "value_2class_k2": float(two.mean()),
        "value_fg_only_k2": float(fg.mean()),
        "pred_pos_fraction": pred_pos_fraction,
        "collapsed": bool(pred_pos_fraction > COLLAPSE_THRESHOLD),
        "best_epoch": run.get("best_epoch"),
        "epochs_run": run.get("epochs_run"),
        "steps_to_best": run.get("steps_to_best"),
        "metric_module_sha": run.get("metric_module_sha"),
        "prediction_recipe": run.get("prediction_recipe"),
        "reporting_metric": run.get("reporting_metric"),
        "reporting_metric_value": run.get("reporting_metric_value"),
        "decision_metric": run.get("decision_metric"),
        "parquet_sha256": sha256_file(pq_path),
    }


# ------------------------------------------------------------ statistics ----

def paired_bootstrap(delta, B=B_DEFAULT, seed=BOOT_SEED):
    """Paired bootstrap on a per-image delta vector.

    Replicates statistical_analysis.py:bootstrap_comparison verbatim in
    behaviour: B resamples of the delta vector with replacement, percentile CI
    at 2.5 / 97.5, two-tailed p of min(2 * prop_wrong, 1.0).
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    if n == 0:
        return dict(mean=None, stderr=None, ci95_lo=None, ci95_hi=None,
                    p_raw=None, n=0)
    observed = float(delta.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    boots = delta[idx].mean(axis=1)
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    if observed >= 0:
        prop_wrong = float(np.mean(boots <= 0))
    else:
        prop_wrong = float(np.mean(boots >= 0))
    p = min(2.0 * prop_wrong, 1.0)
    stderr = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else None
    return dict(mean=observed, stderr=stderr, ci95_lo=lo, ci95_hi=hi,
                p_raw=float(p), n=int(n))


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, matching Table 39."""
    pv = [(p, i) for i, p in enumerate(pvals) if p is not None]
    out = [None] * len(pvals)
    if not pv:
        return out
    pv.sort()
    m = len(pv)
    running = 0.0
    for rank, (p, i) in enumerate(pv):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[i] = float(running)
    return out


# ------------------------------------------------------------------ gaps ----

def build_gaps(cells_by_key, exclude_collapsed, label):
    """gap = upper - LOCO, per image, paired on image_id.

    Rungs 225 and 450 pair against the same-rung upper bound.  Rung 900 pairs
    against upper-450 and is the budget-asymmetric gap.
    """
    gaps, excluded, per_image = [], [], {}
    log(f"[gaps:{label}] building (exclude_collapsed={exclude_collapsed})")
    for arch in ARCHS:
        for city in CITIES:
            for rung in RUNGS:
                for seed in SEEDS:
                    loco = cells_by_key.get((arch, city, rung, "loco", seed))
                    up_r = COMPARATOR_RUNG[rung]
                    upper = cells_by_key.get((arch, city, up_r, "upper", seed))
                    if loco is None or upper is None:
                        continue
                    tag = f"{arch}/{city}/r{rung}/s{seed}"
                    if exclude_collapsed and (loco["collapsed"]
                                              or upper["collapsed"]):
                        side = [c["run_id"] for c in (loco, upper)
                                if c["collapsed"]]
                        excluded.append({"arch": arch, "city": city,
                                         "rung": rung, "seed": seed,
                                         "collapsed_side": side})
                        log(f"[gaps:{label}]   EXCLUDED {tag} "
                            f"collapsed: {side}")
                        continue

                    same = np.array_equal(loco["image_ids"],
                                          upper["image_ids"])
                    ok(same, f"{tag}: image-ID sets match exactly between "
                             f"{loco['run_id']} and {upper['run_id']}")
                    delta = upper["two_class"] - loco["two_class"]
                    st = paired_bootstrap(delta)
                    gaps.append({
                        "arch": arch, "city": city, "rung": rung, "seed": seed,
                        "loco_run_id": loco["run_id"],
                        "upper_run_id": upper["run_id"],
                        "loco_value_2class_k2": loco["value_2class_k2"],
                        "upper_value_2class_k2": upper["value_2class_k2"],
                        "gap_mean": st["mean"], "stderr": st["stderr"],
                        "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
                        "p_raw": st["p_raw"], "p_holm": None,
                        "n_test": st["n"],
                        "comparator": f"upper_r{up_r}",
                        "is_budget_asymmetric": BUDGET_ASYMMETRIC[rung],
                        "flip": bool(st["mean"] < 0),
                    })
                    per_image[(arch, city, rung, seed)] = (
                        loco["image_ids"], delta)

    # Holm within each rung's family of 9 cells.
    for rung in RUNGS:
        fam = [g for g in gaps if g["rung"] == rung]
        adj = holm([g["p_raw"] for g in fam])
        for g, a in zip(fam, adj):
            g["p_holm"] = a
        log(f"[gaps:{label}] rung {rung}: Holm over family of {len(fam)} cells")
    return gaps, excluded, per_image


def rung_contrasts(per_image, label):
    """Per-image delta-of-delta across rungs; the same 150 test images recur at
    every rung for a given (arch, city, seed), so the contrast is paired."""
    out = []
    pairs = [("gap225_minus_gap450", 225, 450),
             ("gap450_minus_gap900", 450, 900),
             ("gap225_minus_gap900", 225, 900)]
    for name, ra, rb in pairs:
        pooled, clusters = [], []
        for arch in ARCHS:
            for city in CITIES:
                for seed in SEEDS:
                    a = per_image.get((arch, city, ra, seed))
                    b = per_image.get((arch, city, rb, seed))
                    if a is None or b is None:
                        continue
                    ok(np.array_equal(a[0], b[0]),
                       f"contrast {name} {arch}/{city}/s{seed}: image-ID sets "
                       f"match across rungs r{ra} and r{rb}")
                    d = a[1] - b[1]
                    pooled.append(d)
                    clusters.append(float(d.mean()))
        if not pooled:
            log(f"[contrast:{label}] {name}: no complete pairs, skipped")
            continue
        vec = np.concatenate(pooled)
        st = paired_bootstrap(vec)
        out.append({
            "contrast_name": name, "mean": st["mean"], "stderr": st["stderr"],
            "ci95_lo": st["ci95_lo"], "ci95_hi": st["ci95_hi"],
            "p_raw": st["p_raw"], "p_holm": None,
            "n_images": st["n"], "n_cell_triples": len(clusters),
            "cluster_mean_sd": (float(np.std(clusters, ddof=1))
                                if len(clusters) > 1 else None),
        })
        log(f"[contrast:{label}] {name}: n={st['n']} over "
            f"{len(clusters)} (arch,city,seed) triples")
    for c, a in zip(out, holm([c["p_raw"] for c in out])):
        c["p_holm"] = a
    return out


def seed_variance(gaps):
    """SD of cell-level gap across seeds, per (arch, city, rung)."""
    grp = defaultdict(list)
    for g in gaps:
        grp[(g["arch"], g["city"], g["rung"])].append(g["gap_mean"])
    out = []
    for (arch, city, rung), vals in sorted(grp.items()):
        out.append({
            "arch": arch, "city": city, "rung": rung,
            "sd_across_seeds": (float(np.std(vals, ddof=1))
                                if len(vals) > 1 else None),
            "mean_across_seeds": float(np.mean(vals)),
            "n_seeds": len(vals),
        })
    return out


def rung_summary(gaps, label):
    """Per-rung mean gap over its 9 cells, with a cell-level bootstrap CI.

    The resampling unit is the cell, not the image, because the 9 cells are the
    independent replicates; an image-level CI here would understate the spread.
    """
    out = []
    for rung in RUNGS:
        fam = [g for g in gaps if g["rung"] == rung]
        if not fam:
            continue
        vals = np.array([g["gap_mean"] for g in fam], dtype=np.float64)
        rng = np.random.default_rng(BOOT_SEED)
        boots = vals[rng.integers(0, vals.size,
                                  size=(B_DEFAULT, vals.size))].mean(axis=1)
        flips = [g for g in fam if g["flip"]]
        out.append({
            "rung": rung,
            "mean_gap_9cells": float(vals.mean()),
            "stderr": (float(vals.std(ddof=1) / np.sqrt(vals.size))
                       if vals.size > 1 else None),
            "ci95_lo": float(np.percentile(boots, 2.5)),
            "ci95_hi": float(np.percentile(boots, 97.5)),
            "ci_method": "cell_level_bootstrap_B10000",
            "n_cells": int(vals.size),
            "is_budget_asymmetric": BUDGET_ASYMMETRIC[rung],
            "n_flips": len(flips),
            "flip_run_ids": [g["loco_run_id"] for g in flips],
        })
        log(f"[rung:{label}] r{rung}: mean gap {vals.mean():+.4f} over "
            f"{vals.size} cells, {len(flips)} flips")
    return out


def arch_summary(gaps):
    """Per-architecture mean gap per rung, over its 3 cities x 3 seeds."""
    grp = defaultdict(list)
    for g in gaps:
        grp[(g["arch"], g["rung"])].append(g["gap_mean"])
    out = []
    for (arch, rung), vals in sorted(grp.items()):
        v = np.array(vals, dtype=np.float64)
        rng = np.random.default_rng(BOOT_SEED)
        boots = v[rng.integers(0, v.size, size=(B_DEFAULT, v.size))].mean(1)
        out.append({
            "arch": arch, "rung": rung, "mean_gap": float(v.mean()),
            "stderr": (float(v.std(ddof=1) / np.sqrt(v.size))
                       if v.size > 1 else None),
            "ci95_lo": float(np.percentile(boots, 2.5)),
            "ci95_hi": float(np.percentile(boots, 97.5)),
            "ci_method": "cell_level_bootstrap_B10000", "n_cells": int(v.size),
        })
    return out


# -------------------------------------------------------- sanity / provenance

def reproduction_check(cells):
    """Recompute each cell's stored reporting_metric_value (foreground-only
    K=2) from the parquet counts.  Expect a max |delta| of 0.0000."""
    rows, worst, worst_id = [], 0.0, None
    for c in cells:
        stored = c["reporting_metric_value"]
        ok(c["reporting_metric"] == "fg_only_tol_k2",
           f"{c['run_id']}: reporting_metric = {c['reporting_metric']!r}")
        if stored is None:
            rows.append({"run_id": c["run_id"], "stored": None,
                         "recomputed": c["value_fg_only_k2"], "abs_delta": None})
            continue
        d = abs(float(stored) - c["value_fg_only_k2"])
        if d > worst:
            worst, worst_id = d, c["run_id"]
        rows.append({"run_id": c["run_id"], "stored": float(stored),
                     "recomputed": c["value_fg_only_k2"], "abs_delta": d})
    log(f"[repro] max |delta| vs stored reporting_metric_value = "
        f"{worst:.6f} ({worst_id})")
    log("[repro] reporting_metric_value is stored rounded to 4 decimals, so a "
        "residual up to 5e-5 is the rounding itself, not a metric "
        "disagreement")
    return {"max_abs_delta": worst, "max_abs_delta_run_id": worst_id,
            "n_cells_checked": len(rows), "tolerance_expected": 0.0001,
            "within_tolerance": bool(worst <= 0.0001), "per_cell": rows}


def provenance_check(cells):
    recipes = sorted({c["prediction_recipe"] for c in cells})
    shas = sorted({c["metric_module_sha"] for c in cells})
    ok(len(recipes) == 1,
       f"prediction_recipe identical across all {len(cells)} cells "
       f"({len(recipes)} distinct)")
    log(f"[provenance] metric_module_sha distinct values: {len(shas)}")
    if len(shas) > 1:
        log("[provenance] ANOMALY: cells were dumped under more than one "
            "metrics.py. Confirm the difference does not touch the tolerant "
            "count path before comparing across the split.")
    # Record which cells sit on which metrics.py, not just how many shas exist.
    sha_breakdown = []
    for s in shas:
        members = sorted(c["run_id"] for c in cells if c["metric_module_sha"] == s)
        by_rung = defaultdict(int)
        for c in cells:
            if c["metric_module_sha"] == s:
                by_rung[c["rung"]] += 1
        sha_breakdown.append({
            "metric_module_sha": s, "n_cells": len(members),
            "cells_by_rung": dict(sorted(by_rung.items())),
            "run_ids": members,
        })
        log(f"[provenance]   {s}: {len(members)} cells, "
            f"by rung {dict(sorted(by_rung.items()))}")
    return {"prediction_recipe": recipes[0] if len(recipes) == 1 else recipes,
            "metric_module_sha_breakdown": sha_breakdown,
            "prediction_recipe_identical": len(recipes) == 1,
            "metric_module_sha": shas[0] if len(shas) == 1 else shas,
            "metric_module_sha_identical": len(shas) == 1,
            "metric_module_sha_anomaly": None if len(shas) == 1 else (
                "Cells were dumped under more than one metrics.py. Recorded, "
                "not suppressed. See metric_module_sha_breakdown for the exact "
                "per-cell assignment; the split does not fall cleanly on the "
                "rung boundary."),
            "decision_metric": sorted({c["decision_metric"] for c in cells}),
            "target_access_counts": dict(sorted(
                pd.Series([c["target_access"] for c in cells])
                .value_counts().items()))}


def collapse_report(cells):
    detected = sorted(c["run_id"] for c in cells if c["collapsed"])
    known = [r for r in KNOWN_COLLAPSED
             if r in {c["run_id"] for c in cells}]
    additional = sorted(set(detected) - set(KNOWN_COLLAPSED))
    missed = sorted(set(known) - set(detected))
    log(f"[collapse] rule: pred_pos_fraction = n_pred_pos / n_valid_px > "
        f"{COLLAPSE_THRESHOLD}")
    log(f"[collapse] detected {len(detected)}: {detected}")
    log(f"[collapse] known-in-scope {len(known)}, additional "
        f"{len(additional)}: {additional}")
    if missed:
        ranked = sorted(((c["pred_pos_fraction"], c["run_id"]) for c in cells),
                        reverse=True)[:8]
        log("[collapse] pred_pos_fraction, highest cells:")
        for frac, rid in ranked:
            log(f"[collapse]   {frac:7.4f}  {rid}")
        log("[collapse] the threshold does not reproduce the known collapsed "
            "list; set --collapse-threshold to a value that does")
    ok(not missed, f"rule catches every known collapsed cell in scope "
                   f"(missed: {missed})")
    return detected, additional


def derive_index_sha(cells):
    """No INDEX.parquet is assumed; digest the sorted (run_id, parquet sha)
    list so the JSON still pins the exact inputs it was computed from."""
    h = hashlib.sha256()
    for c in sorted(cells, key=lambda x: x["run_id"]):
        h.update((c["run_id"] + ":" + c["parquet_sha256"] + "\n").encode())
    return h.hexdigest()


def git_hash(repo_root):
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip(), None
        return None, f"git rev-parse failed: {out.stderr.strip()[:200]}"
    except Exception as exc:
        return None, f"git unavailable: {exc}"


def next_exp_nn(repo_root):
    """Next number after the last EXP in summaries/EXPERIMENTS.md, also
    respecting any NN_ prefix already used in results/."""
    seen, unnumbered = [], 0
    exp_md = os.path.join(repo_root, "summaries", "EXPERIMENTS.md")
    if os.path.exists(exp_md):
        with open(exp_md) as fh:
            for line in fh:
                m = re.match(r"^##\s*EXP-(\d+)", line)
                if m:
                    seen.append(int(m.group(1)))
                elif re.match(r"^##\s*EXP\b", line):
                    unnumbered += 1
    res_dir = os.path.join(repo_root, "results")
    if os.path.isdir(res_dir):
        for f in os.listdir(res_dir):
            m = re.match(r"^(\d+)_", f)
            if m:
                seen.append(int(m.group(1)))
    nn = (max(seen) if seen else 0) + 1
    log(f"[numbering] numbered EXP/results maxima -> {sorted(set(seen))}; "
        f"unnumbered EXP blocks in EXPERIMENTS.md: {unnumbered}")
    log(f"[numbering] next NN = {nn:02d} "
        f"(override with --exp-nn if the operator numbers differently)")
    return nn


# --------------------------------------------------------------- summary ----

def print_summary(rung_sum, arch_sum, contrasts, gaps, excluded, extra):
    print("\n" + "=" * 78)
    print("ARM C — SCALING LADDER, ONE-SCREEN SUMMARY")
    print("=" * 78)
    print("\nMean gap (upper - LOCO, two-class tolerant mIoU K=2, percent).")
    print("Negative = LOCO beats its upper bound.\n")
    hdr = f"{'rung':>6} {'arch':<9} {'mean gap':>10} {'95% CI':>22} {'n':>4}"
    print(hdr)
    print("-" * len(hdr))
    for r in rung_sum:
        tag = "  [budget-asym]" if r["is_budget_asymmetric"] else ""
        print(f"{r['rung']:>6} {'ALL':<9} {r['mean_gap_9cells']:>+10.4f} "
              f"[{r['ci95_lo']:>+8.4f},{r['ci95_hi']:>+8.4f}] "
              f"{r['n_cells']:>4}{tag}")
        for a in [x for x in arch_sum if x["rung"] == r["rung"]]:
            print(f"{'':>6} {a['arch']:<9} {a['mean_gap']:>+10.4f} "
                  f"[{a['ci95_lo']:>+8.4f},{a['ci95_hi']:>+8.4f}] "
                  f"{a['n_cells']:>4}")
        print(f"{'':>6} {'flips':<9} {r['n_flips']:>10d}"
              + (f"   {r['flip_run_ids']}" if r["flip_run_ids"] else ""))
    print("\nRung contrasts (per-image delta-of-delta, paired bootstrap "
          "B=10,000):\n")
    hdr2 = (f"{'contrast':<22} {'mean':>9} {'95% CI':>22} {'p_raw':>9} "
            f"{'p_holm':>9} {'n':>6}")
    print(hdr2)
    print("-" * len(hdr2))
    for c in contrasts:
        print(f"{c['contrast_name']:<22} {c['mean']:>+9.4f} "
              f"[{c['ci95_lo']:>+8.4f},{c['ci95_hi']:>+8.4f}] "
              f"{c['p_raw']:>9.4f} {c['p_holm']:>9.4f} {c['n_images']:>6}")
    sig = sum(1 for g in gaps if g["p_holm"] is not None and g["p_holm"] < 0.05)
    print(f"\nCells: {len(gaps)} gaps, {sig} significant after Holm at 0.05.")
    print(f"Excluded gap-seeds (collapsed either side): {len(excluded)}")
    for e in excluded:
        print(f"  {e['arch']}/{e['city']}/r{e['rung']}/s{e['seed']} "
              f"-> {e['collapsed_side']}")
    print(f"Additional collapsed cells beyond the four known: {extra}")
    print("=" * 78 + "\n")


# ------------------------------------------------------------------ main ----

CAVEATS = [
    "Rung 900 is budget-asymmetric: LOCO trains on 900 images, the upper bound "
    "on 450, because a held-out city holds exactly 450 high-resolution "
    "training images. It is reported in its own row as the budget-asymmetric "
    "gap and is never merged with the gap.",
    "High-resolution cells only. Mid-resolution was not run for this arm.",
    "Arm C is vanilla-only by design. No adaptation method appears here; the "
    "adaptation-method baselines are arm F, and this results JSON must not be "
    "read as a method comparison.",
    "The within-city upper bound is the comparator by construction, which is "
    "what satisfies the CONVENTIONS.md same-run baseline requirement. Vanilla "
    "no-adaptation is the LOCO arm itself.",
    "Gaps whose LOCO or upper side collapsed to a near-all-positive prediction "
    "are excluded from the primary analysis and reported in "
    "excluded_gap_seeds. The r225 row depends on that exclusion; "
    "sensitivity_including_collapsed repeats the whole analysis with them in.",
    "The rung contrasts pool per-image delta-of-delta across (architecture, "
    "city, seed) and bootstrap over images, so the interval treats images as "
    "independent within a rung triple. cluster_mean_sd on each contrast gives "
    "the spread of the 27 triple-level means for comparison.",
    "Per-rung and per-architecture intervals resample cells, not images, "
    "because the cells are the independent replicates at that level.",
    "Validation is held at 150 images at every rung, so model-selection "
    "quality does not move with the rung.",
]


def build_envelope(nn, cells, gaps, contrasts, seedvar, rungs, archs_sum,
                   excluded, additional, repro, prov, sens, args, index_sha):
    gh, gh_reason = git_hash(args.repo_root)
    by_rung = {r["rung"]: r for r in rungs}
    parts = []
    for r in RUNGS:
        if r in by_rung:
            row = by_rung[r]
            name = "budget-asymmetric gap" if row["is_budget_asymmetric"] \
                else "gap"
            parts.append(f"{row['mean_gap_9cells']:+.2f} at rung {r} "
                         f"({name}, {row['n_cells']} cells)")
    headline = ("The mean upper-minus-LOCO gap in two-class boundary-tolerant "
                "mIoU at K=2 is " + ", ".join(parts) + ".") if parts else \
        "No complete rung was available."

    conditions = []
    for r in rungs:
        conditions.append({
            "label": f"all architectures / rung {r['rung']} / "
                     f"matched-{COMPARATOR_RUNG[r['rung']]}-upper",
            "metric": ("gap_budget_asymmetric_2class_mIoU_tol_k2"
                       if r["is_budget_asymmetric"]
                       else "gap_2class_mIoU_tol_k2"),
            "mean": r["mean_gap_9cells"], "stderr": r["stderr"],
            "n": r["n_cells"], "ci95": [r["ci95_lo"], r["ci95_hi"]],
            "ci_method": r["ci_method"],
            "baseline": f"upper_r{COMPARATOR_RUNG[r['rung']]}",
        })
    for a in archs_sum:
        conditions.append({
            "label": f"{a['arch']} / rung {a['rung']} / "
                     f"matched-{COMPARATOR_RUNG[a['rung']]}-upper",
            "metric": ("gap_budget_asymmetric_2class_mIoU_tol_k2"
                       if BUDGET_ASYMMETRIC[a["rung"]]
                       else "gap_2class_mIoU_tol_k2"),
            "mean": a["mean_gap"], "stderr": a["stderr"], "n": a["n_cells"],
            "ci95": [a["ci95_lo"], a["ci95_hi"]], "ci_method": a["ci_method"],
            "baseline": f"upper_r{COMPARATOR_RUNG[a['rung']]}",
        })

    tests = []
    for g in gaps:
        tests.append({
            "name": "paired bootstrap",
            "compares": [g["loco_run_id"], g["upper_run_id"]],
            "statistic": g["gap_mean"], "p": g["p_raw"],
            "p_holm": g["p_holm"], "B": B_DEFAULT, "two_tailed": True,
            "family": f"rung_{g['rung']}_9_cells",
        })
    for c in contrasts:
        tests.append({
            "name": "paired bootstrap, per-image delta-of-delta",
            "compares": [c["contrast_name"]],
            "statistic": c["mean"], "p": c["p_raw"], "p_holm": c["p_holm"],
            "B": B_DEFAULT, "two_tailed": True,
            "family": "rung_contrasts_3",
        })

    return {
        "id": f"{nn:02d}_scaling_ladder",
        "arm": "C",
        "answers": ["zNBu-W3", "AC1"],
        "question": "Does the LOCO gap persist as source training data "
                    "scales from 225 to 450 to 900 images?",
        "runs": sorted(c["run_id"] for c in cells),
        "index_sha256": index_sha,
        "index_sha256_note": "no INDEX.parquet was read; this is sha256 over "
                             "the sorted run_id:parquet_sha256 list of the "
                             "cells analysed",
        "generated_utc": datetime.now(timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": args.script_label,
        "git_hash": gh,
        "git_hash_reason": gh_reason,
        "conditions": conditions,
        "tests": tests,
        "headline": headline,
        "caveats": CAVEATS,
        "figure": f"plots/{nn:02d}_scaling_ladder.png",
        "caption": (
            f"Figure {nn:02d}. Leave-one-city-out generalisation gap across the "
            "225 / 450 / 900 scaling ladder, for MAMNet, OGLANet and DINOv3 on "
            "high-resolution imagery. The gap is the within-city upper bound "
            "minus the LOCO model, in two-class boundary-tolerant mIoU at a "
            "+-2 px don't-care band, computed per test image and averaged. "
            "Bars show the mean over the nine cells at each rung (three cities "
            "x three seeds); error bars are 95% percentile bootstrap intervals "
            "over cells, B = 10,000. The dashed grey line marks a zero gap, "
            "where LOCO matches its within-city upper bound. Rung 900 is "
            "budget-asymmetric: no 900-image within-city upper bound exists "
            "because a held-out city holds 450 high-resolution training "
            "images, so its comparator is the 450-image upper bound. n = 150 "
            "test images per cell."),
        "metric_definition": {
            "name": "2class_tol_k2",
            "formula": "fg = tp/(tp+fp+fn+1e-10); bg = tn/(tn+fp+fn+1e-10); "
                       "value = mean(fg,bg) * 100",
            "band_width_px": 2,
            "source": "p0a_recompute.py:miou_from_counts",
            "computed": "per image from raw counts, never from cell means",
        },
        "cells": [{k: c[k] for k in (
            "run_id", "arch", "city", "rung", "mode", "seed", "target_access",
            "n_test", "value_2class_k2", "value_fg_only_k2",
            "pred_pos_fraction", "collapsed", "best_epoch", "epochs_run",
            "steps_to_best")} for c in sorted(cells,
                                              key=lambda x: x["run_id"])],
        "gaps": gaps,
        "rung_contrasts": contrasts,
        "seed_variance": seedvar,
        "rung_summary": rungs,
        "arch_summary": archs_sum,
        "excluded_gap_seeds": excluded,
        "additional_collapsed_detected": additional,
        "reproduction_check": repro,
        "provenance_check": prov,
        "sensitivity_including_collapsed": sens,
        "incomplete_cells_dropped": list(INCOMPLETE),
        "collapse_rule": {
            "definition": "n_pred_pos / n_valid_px, pooled over the cell's "
                          "images, greater than the threshold",
            "threshold": COLLAPSE_THRESHOLD,
            "known_collapsed": list(KNOWN_COLLAPSED),
            "threshold_provenance":
                "Not operator-authorised; derived from the ground truth. GT "
                "shadow occupies at most 0.086 of the valid region on any cell "
                "(chicago, mean over images; pooled 0.0839; miami 0.080, "
                "phoenix 0.035). Foreground IoU is bounded above by "
                "gt_fg / pred_pos, so any cell at pred-positive >= 0.50 has fg "
                "IoU <= 0.086 / 0.50 = 17.2 by construction, meaning the rule "
                "cannot catch a working model. The spec named 0.99, which "
                "detects only 1 of the 4 known collapsed cells "
                "(dinov3-vanilla-loco-r225-miami-hr-s0, at 0.9986). Measured "
                "pooled pred-positive fractions: the 4 known sit at "
                "0.6483-0.9986 with fg IoU 3.51-8.02, the highest "
                "non-collapsed cell is 0.1490 at 50.93 fg IoU "
                "(mamnet-vanilla-loco-r225-miami-hr-s1), and the remaining 130 "
                "cells fall below it, so any threshold in (0.15, 0.65) gives "
                "the same exclusion set.",
        },
        "log": LOG_LINES,
    }


def analyse(cells, label, exclude_collapsed):
    by_key = {(c["arch"], c["city"], c["rung"], c["mode"], c["seed"]): c
              for c in cells}
    gaps, excluded, per_image = build_gaps(by_key, exclude_collapsed, label)
    contrasts = rung_contrasts(per_image, label)
    return (gaps, excluded, contrasts, seed_variance(gaps),
            rung_summary(gaps, label), arch_summary(gaps))


def main():
    global COLLAPSE_THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dumps-root",
                    default=os.path.join(os.environ["PROJECT_ROOT"], "dumps/armC/st"))
    ap.add_argument("--repo-root",
                    default=os.environ["PROJECT_ROOT"])
    ap.add_argument("--archs", default=",".join(ARCHS),
                    help="comma-separated subset; the dry run uses mamnet")
    ap.add_argument("--expect-cells", type=int, default=N_CELLS_EXPECTED)
    ap.add_argument("--exp-nn", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve schema, run the full analysis, print it, "
                         "write nothing")
    ap.add_argument("--collapse-threshold", type=float,
                    default=COLLAPSE_THRESHOLD,
                    help="pred_pos_fraction above which a cell is collapsed. "
                         "Default 0.50: GT foreground peaks at 0.086 of the "
                         "valid region, so a cell at pred-positive >= 0.50 has "
                         "fg IoU <= 17.2 by construction and the rule cannot "
                         "catch a working model. The spec's 0.99 detects only "
                         "1 of the 4 known collapsed cells.")
    ap.add_argument("--skip-incomplete", action="store_true",
                    help="drop cells missing an artefact instead of aborting")
    ap.add_argument("--validate-only", action="store_true",
                    help="load and assert only; compute no statistics")
    ap.add_argument("--script-label",
                    default="python/armc_ladder_stats.py")
    args = ap.parse_args()
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    COLLAPSE_THRESHOLD = args.collapse_threshold

    log(f"[start] {datetime.now(timezone.utc).isoformat()}")
    log(f"[start] archs={archs} dry_run={args.dry_run} "
        f"validate_only={args.validate_only}")
    log(f"[start] collapse_threshold={COLLAPSE_THRESHOLD} "
        f"skip_incomplete={args.skip_incomplete}")

    metas = discover_cells(args.dumps_root, archs, args.expect_cells)
    cells = [load_cell(m, args.skip_incomplete) for m in metas]
    cells = [c for c in cells if c is not None]
    log(f"[load] loaded {len(cells)} cells; "
        f"{len(INCOMPLETE)} incomplete and dropped")
    for rec in INCOMPLETE:
        log(f"[load]   incomplete: {rec['run_id']} ({rec['missing']})")

    repro = reproduction_check(cells)
    prov = provenance_check(cells)
    detected, additional = collapse_report(cells)

    if args.validate_only:
        log("[validate-only] stopping before statistics, as requested")
        return 0

    gaps, excluded, contrasts, seedvar, rungs, archs_sum = analyse(
        cells, "primary", exclude_collapsed=True)
    s_gaps, _, s_contrasts, s_seedvar, s_rungs, s_archs = analyse(
        cells, "sensitivity", exclude_collapsed=False)
    sens = {"note": "identical analysis with collapsed cells included",
            "gaps": s_gaps, "rung_contrasts": s_contrasts,
            "seed_variance": s_seedvar, "rung_summary": s_rungs,
            "arch_summary": s_archs}

    nn = args.exp_nn if args.exp_nn is not None else next_exp_nn(args.repo_root)
    env = build_envelope(nn, cells, gaps, contrasts, seedvar, rungs, archs_sum,
                         excluded, additional, repro, prov, sens, args,
                         derive_index_sha(cells))

    print_summary(rungs, archs_sum, contrasts, gaps, excluded, additional)

    out = args.out or os.path.join(args.repo_root, "results",
                                   f"{nn:02d}_scaling_ladder.json")
    if args.dry_run:
        print(f"[dry-run] would write {out}")
        print(f"[dry-run] envelope keys: {sorted(env.keys())}")
        print(f"[dry-run] cells={len(env['cells'])} gaps={len(env['gaps'])} "
              f"contrasts={len(env['rung_contrasts'])} "
              f"tests={len(env['tests'])}")
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(env, fh, indent=2, default=float)
    print(f"[write] {out}  ({os.path.getsize(out)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
