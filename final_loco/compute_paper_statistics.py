"""
compute_paper_statistics.py  (v2 — robustness-hardened, no mIoU floor)

Changes from previous version:
  - Removed all mIoU floor / threshold logic on models being considered
  - All cells are now included regardless of LOCO baseline mIoU
  - D3: Per-image R with cluster bootstrap, Kruskal-Wallis, Cliff's delta
  - D2 permutation control flagged for GroupKFold fix (pending extract_features.py)

Requirements:
    - evaluate_experiments.py MUST be patched to save per_image_iou
      and re-run BEFORE running this script
    - Saved experiment evaluation results
    - Extracted features in FEATURE_BASE (for permutation control only)

Usage:
    python compute_paper_statistics.py
    python compute_paper_statistics.py --skip_permutation
    python compute_paper_statistics.py --permutation_n 100
"""

import os, sys, json, argparse, time, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CITIES, RESOLUTIONS, MODELS, LOCO_VARIANTS, OUTPUT_BASE, SHADOW_TYPE_MAP

FEATURE_BASE = os.path.join(os.environ["PROJECT_ROOT"], "data", "extracted_features")

CHANCE_ACCURACY = 1.0 / len(CITIES)
MAX_SAMPLES_PER_CITY = 30000

# Minimum upper-loco gap per image to compute meaningful R
MIN_GAP_FOR_R = 0.02


def load_json_safe(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found"); return None
    with open(path) as f: return json.load(f)

def safe_float(v):
    if v is None: return float('nan')
    try: return float(v)
    except: return float('nan')

def make_serializable(obj):
    if isinstance(obj, dict): return {str(k): make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, np.bool_): return bool(obj)
    return obj


# ============================================================
# STATISTICAL UTILITIES
# ============================================================

def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42, statistic='mean'):
    """Bootstrap CI for mean or median."""
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) == 0:
        return float('nan'), (float('nan'), float('nan'))
    if len(values) == 1:
        return float(values[0]), (float(values[0]), float(values[0]))
    rng = np.random.RandomState(seed)
    func = np.median if statistic == 'median' else np.mean
    boot = np.array([func(rng.choice(values, len(values), True)) for _ in range(n_boot)])
    a = (1 - ci) / 2
    return float(func(values)), (float(np.percentile(boot, a*100)),
                                  float(np.percentile(boot, (1-a)*100)))


def cluster_bootstrap_ci(values, n_boot=10000, ci=0.95, seed=42):
    """
    Bootstrap CI resampling at the image level.
    Each value = one image's R. Returns (median, mean, (ci_low, ci_high)).

    Since each image contributes exactly one R value,
    cluster bootstrap = standard bootstrap on the R array.
    """
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) == 0:
        return float('nan'), float('nan'), (float('nan'), float('nan'))
    if len(values) == 1:
        v = float(values[0])
        return v, v, (v, v)
    rng = np.random.RandomState(seed)
    boot_medians = np.array([np.median(rng.choice(values, len(values), True))
                             for _ in range(n_boot)])
    a = (1 - ci) / 2
    return (float(np.median(values)),
            float(np.mean(values)),
            (float(np.percentile(boot_medians, a*100)),
             float(np.percentile(boot_medians, (1-a)*100))))


def cliffs_delta(x, y):
    """
    Cliff's delta: non-parametric effect size.
    delta = P(X > Y) - P(X < Y).
    Range: [-1, 1].

    Thresholds (Romano et al. 2006):
        |delta| < 0.147 → negligible
        |delta| < 0.330 → small
        |delta| < 0.474 → medium
        |delta| >= 0.474 → large
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float('nan'), "undefined"

    # Vectorized pairwise comparison
    diff = x[:, None] - y[None, :]
    more = (diff > 0).sum()
    less = (diff < 0).sum()
    n = len(x) * len(y)
    delta = float((more - less) / n)

    # Interpret
    ad = abs(delta)
    if ad < 0.147:
        interp = "negligible"
    elif ad < 0.330:
        interp = "small"
    elif ad < 0.474:
        interp = "medium"
    else:
        interp = "large"
    return delta, interp


def cliffs_delta_bootstrap_ci(x, y, n_boot=5000, ci=0.95, seed=42):
    """Bootstrap CI for Cliff's delta."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float('nan'), (float('nan'), float('nan'))

    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        xb = rng.choice(x, len(x), True)
        yb = rng.choice(y, len(y), True)
        diff = xb[:, None] - yb[None, :]
        d = (diff > 0).sum() - (diff < 0).sum()
        deltas.append(d / (len(xb) * len(yb)))
    deltas = np.array(deltas)
    a = (1 - ci) / 2
    return float(np.median(deltas)), (float(np.percentile(deltas, a*100)),
                                       float(np.percentile(deltas, (1-a)*100)))


# ============================================================
# PER-IMAGE RECOVERY RATIO
# ============================================================

def compute_per_image_R(exp_ious, loco_ious, upper_ious, min_gap=MIN_GAP_FOR_R):
    """
    Compute per-image recovery ratio: R_i = (exp_i - loco_i) / (upper_i - loco_i).

    Args:
        exp_ious: list of per-image IoU for experiment condition
        loco_ious: list of per-image IoU for LOCO vanilla baseline
        upper_ious: list of per-image IoU for upper bound
        min_gap: minimum |upper - loco| gap per image to include
                 (avoids division instability on images where both are ~equal)

    Returns:
        R_valid: numpy array of per-image R for valid images
        n_total: total images
        n_excluded_nan: excluded because at least one condition was NaN
        n_excluded_gap: excluded because upper-loco gap < min_gap
    """
    exp = np.array([safe_float(v) for v in exp_ious])
    loco = np.array([safe_float(v) for v in loco_ious])
    upper = np.array([safe_float(v) for v in upper_ious])

    n_total = len(exp)
    assert len(loco) == n_total and len(upper) == n_total, \
        f"Length mismatch: exp={len(exp)}, loco={len(loco)}, upper={len(upper)}"

    # Mask 1: all three non-NaN
    valid_values = ~(np.isnan(exp) | np.isnan(loco) | np.isnan(upper))
    n_excluded_nan = int((~valid_values).sum())

    # Mask 2: gap large enough
    gap = upper - loco
    valid_gap = np.abs(gap) >= min_gap
    n_excluded_gap = int(valid_values.sum() - (valid_values & valid_gap).sum())

    valid = valid_values & valid_gap
    R = (exp[valid] - loco[valid]) / gap[valid]

    return R, n_total, n_excluded_nan, n_excluded_gap


# ============================================================
# ADAPTIVE-BIN SPEARMAN (unchanged from v1)
# ============================================================

def _adaptive_bin_spearman(intensities, values, min_per_bin=10, max_bins=100):
    intensities, values = np.array(intensities), np.array(values)
    valid = ~(np.isnan(intensities) | np.isnan(values))
    intensities, values = intensities[valid], values[valid]
    n = len(intensities)
    if n < 20: return float('nan'), float('nan'), 0
    n_bins = max(10, min(max_bins, n // min_per_bin))
    edges = np.unique(np.percentile(intensities, np.linspace(0, 100, n_bins+1)))
    actual = len(edges) - 1
    if actual < 5: return float('nan'), float('nan'), 0
    bin_idx = np.clip(np.digitize(intensities, edges) - 1, 0, actual - 1)
    centers, means = [], []
    for b in range(actual):
        m = bin_idx == b
        if m.sum() < 3: continue
        centers.append(float((edges[b]+edges[b+1])/2))
        means.append(float(np.mean(values[m])))
    if len(centers) < 5: return float('nan'), float('nan'), 0
    rho, p = sp_stats.spearmanr(centers, means)
    return float(rho), float(p), len(centers)


def _gap_spearman(u_int, u_iou, l_int, l_iou, min_per_bin=10, max_bins=100):
    u_int, u_iou = np.array(u_int), np.array(u_iou)
    l_int, l_iou = np.array(l_int), np.array(l_iou)
    if len(u_int) == len(l_int) and len(u_int) > 0 and np.allclose(u_int, l_int, atol=0.5):
        gaps = u_iou - l_iou
        valid = ~np.isnan(gaps)
        return _adaptive_bin_spearman(u_int[valid], gaps[valid], min_per_bin, max_bins)
    n = min(len(u_int), len(l_int))
    if n < 20: return float('nan'), float('nan'), 0
    n_bins = max(10, min(max_bins, n // min_per_bin))
    all_int = np.concatenate([u_int, l_int])
    edges = np.unique(np.percentile(all_int, np.linspace(0, 100, n_bins+1)))
    actual = len(edges) - 1
    if actual < 5: return float('nan'), float('nan'), 0
    centers, gaps = [], []
    for b in range(actual):
        um = np.clip(np.digitize(u_int, edges)-1, 0, actual-1) == b
        lm = np.clip(np.digitize(l_int, edges)-1, 0, actual-1) == b
        if um.sum() < 3 or lm.sum() < 3: continue
        u_m, l_m = np.nanmean(u_iou[um]), np.nanmean(l_iou[lm])
        if np.isnan(u_m) or np.isnan(l_m): continue
        centers.append(float((edges[b]+edges[b+1])/2))
        gaps.append(float(u_m - l_m))
    if len(centers) < 5: return float('nan'), float('nan'), 0
    rho, p = sp_stats.spearmanr(centers, gaps)
    return float(rho), float(p), len(centers)


# ============================================================
# DIAGNOSTIC 1 (Section 4.1)
# ============================================================

def compute_d1_statistics(res="highres"):
    print("\n" + "="*70 + "\nDIAGNOSTIC 1: Intensity-Conditioned IoU\n" + "="*70)
    d1b = load_json_safe(os.path.join(OUTPUT_BASE, "thread1", "1b_intensity_curves", "intensity_curve_results.json"))
    d1a = load_json_safe(os.path.join(OUTPUT_BASE, "thread1", "1a_fp_clustering", "fp_clustering_results.json"))
    results = {}

    if d1b:
        sample_key = next((k for k in d1b if "upper" in k), None)
        has_raw = sample_key and "raw_ious" in d1b.get(sample_key, {})
        if not has_raw:
            print("  WARNING: raw_ious not in 1b results. Apply patch & re-run diagnostics.")
            print("  Falling back to 8-bin Spearman (low power).")

        gap_cells, darkest_gaps, brightest_gaps = [], [], []

        for model in MODELS:
            for city in CITIES:
                ukey = f"upper_{model}_{city}_{res}"
                u_data = d1b.get(ukey, {})
                if not u_data.get("bins"): continue

                for variant in LOCO_VARIANTS:
                    lkey = f"loco_{model}_{variant}_{city}_{res}"
                    l_data = d1b.get(lkey, {})
                    if not l_data.get("bins"): continue

                    if has_raw:
                        rho, p, nb = _gap_spearman(
                            u_data["raw_intensities"], u_data["raw_ious"],
                            l_data["raw_intensities"], l_data["raw_ious"])
                    else:
                        u_bins, l_bins = u_data["bins"], l_data["bins"]
                        cens, gps = [], []
                        for ub in u_bins:
                            ui = safe_float(ub.get("iou_mean")); uc = safe_float(ub.get("bin_center"))
                            if np.isnan(ui) or ub.get("count",0)==0: continue
                            lb = min((b for b in l_bins if b.get("bin_center") is not None),
                                     key=lambda b: abs(b["bin_center"]-uc), default=None)
                            if lb is None or abs(lb["bin_center"]-uc) > 30: continue
                            li = safe_float(lb.get("iou_mean"))
                            if np.isnan(li) or lb.get("count",0)==0: continue
                            cens.append(uc); gps.append(ui - li)
                        if len(cens) >= 5:
                            rho, p = sp_stats.spearmanr(cens, gps); nb = len(cens)
                        else:
                            rho, p, nb = float('nan'), float('nan'), 0

                    if not np.isnan(rho):
                        gap_cells.append({"model": model, "city": city, "variant": variant,
                                          "rho": rho, "p": p, "n_bins": nb, "sig_05": bool(p<0.05)})
                        print(f"  Gap ρ {model}/{variant}/{city}: ρ={rho:.3f}, p={p:.4f}, n={nb}")

                    if variant == "vanilla":
                        u_bins, l_bins = u_data.get("bins",[]), l_data.get("bins",[])
                        vg = []
                        for ub in u_bins:
                            ui = safe_float(ub.get("iou_mean")); uc = safe_float(ub.get("bin_center"))
                            if np.isnan(ui) or ub.get("count",0)==0: continue
                            lb = min((b for b in l_bins if b.get("bin_center") is not None and b.get("count",0)>0),
                                     key=lambda b: abs(b["bin_center"]-uc), default=None)
                            if lb and abs(lb["bin_center"]-uc)<30:
                                li = safe_float(lb.get("iou_mean"))
                                if not np.isnan(li): vg.append(ui-li)
                        if len(vg) >= 2:
                            darkest_gaps.append(vg[0]); brightest_gaps.append(vg[-1])

        if gap_cells:
            rhos = [c["rho"] for c in gap_cells]
            n_sig = sum(1 for c in gap_cells if c["sig_05"])
            per_model = {}
            for model in MODELS:
                mc = [c for c in gap_cells if c["model"]==model]
                if mc: per_model[model] = {"mean_rho": float(np.mean([c["rho"] for c in mc])),
                                            "n_sig": sum(1 for c in mc if c["sig_05"]),
                                            "n_total": len(mc)}
            results["spearman_monotonicity"] = {
                "cells": gap_cells, "mean_rho": float(np.mean(rhos)),
                "n_sig_05": n_sig, "n_total": len(gap_cells),
                "per_model": per_model, "using_raw": has_raw,
                "summary": f"Mean ρ={np.mean(rhos):.2f}, {n_sig}/{len(gap_cells)} p<0.05"}
            print(f"\n  Summary: mean ρ={np.mean(rhos):.3f}, {n_sig}/{len(gap_cells)} sig (raw={has_raw})")

        if darkest_gaps and brightest_gaps:
            dk, dk_ci = bootstrap_ci(darkest_gaps); br, br_ci = bootstrap_ci(brightest_gaps)
            w = br/dk if abs(dk)>0.001 else float('nan')
            results["gap_magnitude"] = {"dark_mean": dk, "dark_ci": list(dk_ci),
                                         "bright_mean": br, "bright_ci": list(br_ci), "widening": w,
                                         "summary": f"Gap: {dk:.3f}→{br:.3f}, {w:.1f}×"}
            print(f"  Gap: dark={dk:.3f}, bright={br:.3f}, {w:.1f}×")

    # FP concentration
    if d1a:
        loco_f, upper_f = [], []
        for model in MODELS:
            for city in CITIES:
                ukey = f"upper_{model}_{city}_{res}"
                if ukey in d1a and d1a[ukey].get("total_fp_pixels",0)>0:
                    p = d1a[ukey].get("cluster_proportions",[])
                    if len(p)>=2: upper_f.append(p[0]+p[1])
                for variant in LOCO_VARIANTS:
                    lkey = f"loco_{model}_{variant}_{city}_{res}"
                    if lkey in d1a and d1a[lkey].get("total_fp_pixels",0)>0:
                        p = d1a[lkey].get("cluster_proportions",[])
                        if len(p)>=2: loco_f.append(p[0]+p[1])
        if loco_f and upper_f:
            lm, lci = bootstrap_ci(loco_f); um, uci = bootstrap_ci(upper_f)
            U, mp = sp_stats.mannwhitneyu(loco_f, upper_f, alternative='two-sided') if len(loco_f)>=2 and len(upper_f)>=2 else (float('nan'), float('nan'))
            results["fp_concentration"] = {"loco_mean": lm, "loco_n": len(loco_f),
                                            "upper_mean": um, "upper_n": len(upper_f),
                                            "mw_U": float(U), "mw_p": float(mp),
                                            "summary": f"LOCO {lm*100:.1f}% vs Upper {um*100:.1f}%, MW p={mp:.4f}"}
            print(f"\n  FP: LOCO {lm*100:.1f}% (n={len(loco_f)}) vs Upper {um*100:.1f}% (n={len(upper_f)}), p={mp:.4f}")

    # Architecture directionality
    if d1b:
        ad = {}
        for model in MODELS:
            p_drops, r_drops = [], []
            for variant in LOCO_VARIANTS:
                for city in CITIES:
                    u = d1b.get(f"upper_{model}_{city}_{res}", {}).get("bins", [])
                    l = d1b.get(f"loco_{model}_{variant}_{city}_{res}", {}).get("bins", [])
                    if not u or not l: continue
                    up=[safe_float(b.get("precision_mean")) for b in u if b.get("count",0)>0]
                    lp=[safe_float(b.get("precision_mean")) for b in l if b.get("count",0)>0]
                    ur=[safe_float(b.get("recall_mean")) for b in u if b.get("count",0)>0]
                    lr=[safe_float(b.get("recall_mean")) for b in l if b.get("count",0)>0]
                    up=[v for v in up if not np.isnan(v)]; lp=[v for v in lp if not np.isnan(v)]
                    ur=[v for v in ur if not np.isnan(v)]; lr=[v for v in lr if not np.isnan(v)]
                    if up and lp and ur and lr:
                        dp = np.mean(up)-np.mean(lp); dr = np.mean(ur)-np.mean(lr)
                        p_drops.append(dp); r_drops.append(dr)
            ad[model] = {"prec_drop": float(np.mean(p_drops)) if p_drops else float('nan'),
                         "recall_drop": float(np.mean(r_drops)) if r_drops else float('nan'), "n": len(p_drops)}
            if p_drops:
                print(f"  Direction {model}: Δprec={np.mean(p_drops):.3f}, Δrecall={np.mean(r_drops):.3f} (n={len(p_drops)})")

        ct2 = {}
        for model in MODELS:
            p_drops, r_drops = [], []
            for variant in LOCO_VARIANTS:
                for city in CITIES:
                    u = d1b.get(f"upper_{model}_{city}_{res}", {}).get("bins", [])
                    l = d1b.get(f"loco_{model}_{variant}_{city}_{res}", {}).get("bins", [])
                    if not u or not l: continue
                    up=[safe_float(b.get("precision_mean")) for b in u if b.get("count",0)>0]
                    lp=[safe_float(b.get("precision_mean")) for b in l if b.get("count",0)>0]
                    ur=[safe_float(b.get("recall_mean")) for b in u if b.get("count",0)>0]
                    lr=[safe_float(b.get("recall_mean")) for b in l if b.get("count",0)>0]
                    up=[v for v in up if not np.isnan(v)]; lp=[v for v in lp if not np.isnan(v)]
                    ur=[v for v in ur if not np.isnan(v)]; lr=[v for v in lr if not np.isnan(v)]
                    if up and lp and ur and lr:
                        dp = np.mean(up)-np.mean(lp); dr = np.mean(ur)-np.mean(lr)
                        p_drops.append(dp); r_drops.append(dr)
            ct2[model] = [sum(1 for p,r in zip(p_drops,r_drops) if p>r),
                          sum(1 for p,r in zip(p_drops,r_drops) if r>=p)]
        table = np.array([ct2[m] for m in MODELS if m in ct2])
        if table.shape[0]>=2 and table.sum()>0:
            try: chi2, chi2_p, dof, _ = sp_stats.chi2_contingency(table)
            except: chi2, chi2_p, dof = float('nan'), float('nan'), 0
        else: chi2, chi2_p, dof = float('nan'), float('nan'), 0
        results["directionality"] = {"per_model": ad, "chi2": float(chi2), "chi2_p": float(chi2_p)}

    # Histogram matching
    exp_eval = load_json_safe(os.path.join(OUTPUT_BASE, "experiment_evaluation", "experiment_results.json"))
    if exp_eval:
        hm = []
        for r in exp_eval.values():
            if r.get("experiment") != "c": continue
            exp_iou = safe_float(r.get("global",{}).get("experiment",{}).get("iou"))
            loco_iou = safe_float(r.get("global",{}).get("loco",{}).get("iou"))
            if np.isnan(exp_iou) or np.isnan(loco_iou): continue
            hm.append(exp_iou - loco_iou)
        if hm:
            hm_m, hm_ci = bootstrap_ci(hm)
            t, p = sp_stats.ttest_1samp(hm, 0) if len(hm)>=2 else (float('nan'), float('nan'))
            ni = sum(1 for d in hm if d>0)
            results["histogram_matching"] = {"mean": hm_m, "ci": list(hm_ci), "n": len(hm),
                                              "n_improved": ni, "t": float(t), "p": float(p),
                                              "summary": f"Δ={hm_m:.3f} [{hm_ci[0]:.3f},{hm_ci[1]:.3f}], p={p:.4f}, {ni}/{len(hm)} improved"}
            print(f"\n  Histogram match: Δ={hm_m:.3f}, p={p:.4f}, {ni}/{len(hm)} improved")
    return results


# ============================================================
# DIAGNOSTIC 2 (Section 4.2)
# ============================================================

def compute_d2_statistics(res="highres", permutation_n=100, skip_permutation=False):
    print("\n" + "="*70 + "\nDIAGNOSTIC 2: Linear Probes\n" + "="*70)
    probe_results = load_json_safe(os.path.join(OUTPUT_BASE, "thread1", "1d_linear_probe", "probe_results.json"))
    results = {}

    # NOTE: If you have replaced _run_permutation with the GroupKFold version
    # (from permutation_replacement.py), the probe results from thread1_1d_v2.py
    # will also use GroupKFold. Both should be updated together.
    #
    # If thread1_1d_v2.py has been run, the probe results will contain
    # cv_method='GroupKFold_image' and the permutation null should be ~0.333.
    #
    # If still using the old thread1_1d.py, the StratifiedKFold leak persists.
    probe_cv_method = "unknown"
    sample_probe = next(iter(probe_results.values()), {}).get("probe", {})
    probe_cv_method = sample_probe.get("cv_method", "StratifiedKFold_legacy")
    results["cv_method_detected"] = probe_cv_method
    if probe_cv_method != "GroupKFold_image":
        results["methodological_note"] = (
            "Probe CV uses StratifiedKFold (legacy). Replace thread1_1d.py with "
            "thread1_1d_v2.py and re-run for GroupKFold fix."
        )

    if probe_results:
        probe_summary = {}
        for key, r in probe_results.items():
            acc, std = r["probe"]["cv_acc"], r["probe"]["cv_std"]
            nf = r["probe"].get("n_folds", 5); ch = r["chance_accuracy"]
            se = std / np.sqrt(nf); tc = sp_stats.t.ppf(0.975, df=nf-1)
            ci = [float(acc-tc*se), float(acc+tc*se)]
            ts = (acc-ch)/se if se>0 else float('inf')
            pv = float(2*(1-sp_stats.t.cdf(abs(ts), df=nf-1)))
            probe_summary[key] = {"model": r["model_type"], "ckpt": r["checkpoint_id"],
                                   "acc": acc, "std": std, "ci95": ci, "chance": ch,
                                   "t": float(ts), "p": pv, "sig": pv<0.05, "n": r["n_samples"]}
            s = "***" if pv<0.001 else ("n.s." if pv>=0.05 else "*")
            print(f"  {key}: {acc:.3f} [{ci[0]:.3f},{ci[1]:.3f}] p={pv:.4f} {s}")
        results["probe_accuracy"] = probe_summary

        # Intensity-stratified Pearson r
        ip = {}
        for key, r in probe_results.items():
            bins = r.get("intensity_conditioned",{}).get("bins",[])
            valid = [(b["bin_center"], b["accuracy"]) for b in bins
                     if b.get("accuracy") is not None and b.get("count",0)>=20]
            if len(valid)<3: continue
            c, a = zip(*valid)
            pr, pp = sp_stats.pearsonr(c, a)
            ip[key] = {"model": r["model_type"], "r": float(pr), "p": float(pp), "n": len(valid),
                        "min_acc": float(min(a)), "max_acc": float(max(a))}
            print(f"  Intensity {key}: r={pr:.3f}, p={pp:.4f}")
        results["intensity_probe"] = ip

        # LOCO vs Upper comparison
        by_model = defaultdict(list)
        for key, r in probe_results.items(): by_model[r["model_type"]].append((key, r))
        comparisons = {}
        for mt, entries in by_model.items():
            ua = [r["probe"]["cv_acc"] for k,r in entries if "upper" in r["checkpoint_id"]]
            la = [r["probe"]["cv_acc"] for k,r in entries if "loco" in r["checkpoint_id"]]
            if ua and la:
                um, lm = float(np.mean(ua)), float(np.mean(la))
                t, p = sp_stats.ttest_ind(la, ua, equal_var=False) if len(ua)>=2 and len(la)>=2 else (float('nan'), float('nan'))
                comparisons[mt] = {"upper": um, "loco": lm, "delta": lm-um,
                                    "amplified": lm>um, "t": float(t), "p": float(p)}
                print(f"  LOCO vs Upper {mt}: upper={um:.3f}, loco={lm:.3f}, Δ={lm-um:+.3f}")
        results["loco_vs_upper"] = comparisons

    # Permutation control
    if not skip_permutation:
        print("\n  --- Permutation control ---")
        results["permutation"] = _run_permutation(permutation_n)
    return results


def _run_permutation(n_perm):
    """
    Image-level permutation null for linear probe.

    For each iteration:
      1. Assign each unique image a random city label (shuffled from true labels)
      2. All pixels from that image get the shuffled label
      3. Train logistic regression with GroupKFold CV (image as group)
      4. Record mean CV accuracy

    The null distribution represents probe accuracy when city labels are
    uninformative at the image level — the correct null for GroupKFold probes.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score

    combos = []
    if os.path.isdir(FEATURE_BASE):
        for mt in os.listdir(FEATURE_BASE):
            mdir = os.path.join(FEATURE_BASE, mt)
            if not os.path.isdir(mdir): continue
            for ck in os.listdir(mdir):
                if "upper" not in ck: continue
                cdir = os.path.join(mdir, ck)
                if not os.path.isdir(cdir): continue
                nc = sum(1 for e in os.listdir(cdir)
                         if os.path.exists(os.path.join(cdir, e, "features.npz"))
                         and e.split("_")[0] in CITIES)
                if nc >= 2: combos.append((mt, ck))

    results = {}
    done = set()
    for mt, ck in combos:
        if mt in done: continue
        done.add(mt)
        key = f"{mt}_{ck}"
        print(f"    Permutation (GroupKFold): {key} ({n_perm} shuffles)...")

        base = os.path.join(FEATURE_BASE, mt, ck)
        city_idx = {c: i for i, c in enumerate(CITIES)}
        aX, ay, a_img = [], [], []
        global_offset = 0
        has_image_ids = True

        for e in sorted(os.listdir(base)):
            fp = os.path.join(base, e, "features.npz")
            if not os.path.exists(fp): continue
            cn = e.split("_")[0]
            if cn not in city_idx: continue

            d = np.load(fp)
            f = d["features"].astype(np.float32)
            n = f.shape[0]

            # Load image_indices
            if 'image_indices' in d:
                img_ids = d['image_indices'].astype(np.int32)
            else:
                print(f"      WARNING: {fp} missing image_indices. "
                      f"Falling back to pixel-level permutation for {cn}.")
                img_ids = np.arange(n, dtype=np.int32)  # each pixel = own group
                has_image_ids = False

            if n > MAX_SAMPLES_PER_CITY:
                idx = np.linspace(0, n - 1, MAX_SAMPLES_PER_CITY, dtype=int)
                f = f[idx]
                img_ids = img_ids[idx]
                n = MAX_SAMPLES_PER_CITY

            # Offset for global uniqueness
            img_ids = img_ids + global_offset
            global_offset = int(img_ids.max()) + 1

            aX.append(f)
            ay.append(np.full(n, city_idx[cn], dtype=int))
            a_img.append(img_ids)

        if len(aX) < 2: continue
        X = np.concatenate(aX)
        y = np.concatenate(ay)
        image_ids = np.concatenate(a_img)

        unique_images = np.unique(image_ids)
        n_unique = len(unique_images)
        print(f"      {len(X)} samples, {X.shape[1]}D, {n_unique} unique images")

        # Build image-level label mapping (true labels)
        # Each image belongs to exactly one city
        img_to_city = {}
        for img_id in unique_images:
            mask = image_ids == img_id
            img_to_city[img_id] = y[mask][0]
        true_img_cities = np.array([img_to_city[img] for img in unique_images])

        sc = StandardScaler()
        Xs = sc.fit_transform(X)

        # Determine n_folds for GroupKFold
        n_folds = min(5, n_unique)
        if n_folds < 2:
            print(f"      SKIP: only {n_unique} unique images, cannot do CV")
            continue

        gkf = GroupKFold(n_splits=n_folds)
        rng = np.random.RandomState(42)

        # Also compute the ACTUAL probe CV accuracy under true labels
        # (matches thread1_1d_v2.py run_probe)
        actual_fold_accs = []
        for train_idx, test_idx in gkf.split(Xs, y, groups=image_ids):
            clf = LogisticRegression(max_iter=1000, solver='lbfgs',
                                     random_state=42, C=1.0)
            clf.fit(Xs[train_idx], y[train_idx])
            actual_fold_accs.append(accuracy_score(y[test_idx],
                                                    clf.predict(Xs[test_idx])))
        actual_cv_acc = float(np.mean(actual_fold_accs))
        print(f"      Actual CV accuracy (GroupKFold): {actual_cv_acc:.3f}")

        # Permutation iterations
        perm_accs = []
        for i in range(n_perm):
            # Shuffle city labels at IMAGE level
            shuffled_img_cities = rng.permutation(true_img_cities)
            ys = np.zeros_like(y)
            for img_id, new_city in zip(unique_images, shuffled_img_cities):
                ys[image_ids == img_id] = new_city

            # GroupKFold CV with shuffled labels
            fold_accs = []
            for train_idx, test_idx in gkf.split(Xs, ys, groups=image_ids):
                clf = LogisticRegression(max_iter=500, solver='lbfgs',
                                         random_state=42, C=1.0)
                clf.fit(Xs[train_idx], ys[train_idx])
                fold_accs.append(accuracy_score(ys[test_idx],
                                                clf.predict(Xs[test_idx])))
            perm_accs.append(float(np.mean(fold_accs)))

            if (i + 1) % 10 == 0:
                print(f"        {i+1}/{n_perm} "
                      f"(perm mean={np.mean(perm_accs):.3f})")

        perm_accs = np.array(perm_accs)
        m = float(np.mean(perm_accs))
        ci = (float(np.percentile(perm_accs, 2.5)),
              float(np.percentile(perm_accs, 97.5)))

        # Percentile of actual accuracy within null distribution
        # p-value = fraction of permutations >= actual accuracy
        p_value = float(np.mean(perm_accs >= actual_cv_acc))
        percentile = float(np.mean(perm_accs <= actual_cv_acc) * 100)

        results[key] = {
            "model": mt, "ckpt": ck,
            "actual_cv_acc": actual_cv_acc,
            "perm_mean": m, "perm_ci95": list(ci),
            "perm_std": float(np.std(perm_accs)),
            "n_perm": n_perm,
            "p_value": p_value,
            "actual_percentile": percentile,
            "chance": CHANCE_ACCURACY,
            "n_samples": len(X),
            "n_unique_images": n_unique,
            "n_folds": n_folds,
            "has_image_ids": has_image_ids,
            "cv_method": "GroupKFold_image",
        }
        print(f"      {key}: actual={actual_cv_acc:.3f}, "
              f"perm={m:.3f} [{ci[0]:.3f},{ci[1]:.3f}], "
              f"p={p_value:.4f}, percentile={percentile:.1f}%")

    return results


# ============================================================
# DIAGNOSTIC 3 (Section 4.3) — per-image R
# ============================================================

def compute_d3_statistics(res="highres"):
    """
    Diagnostic 3: Controlled Interventions with per-image recovery ratio.

    - Computes R per image (~150 per cell), not per cell
    - Reports median R with 95% cluster-bootstrap CI
    - Kruskal-Wallis across architectures
    - Cliff's delta for pairwise comparisons
    """
    print("\n" + "="*70 + "\nDIAGNOSTIC 3: Controlled Interventions (per-image R)\n" + "="*70)
    exp_eval = load_json_safe(os.path.join(OUTPUT_BASE, "experiment_evaluation", "experiment_results.json"))
    if not exp_eval:
        print("  No experiment results found."); return {}

    results = {"data_availability": {}}
    by_exp = defaultdict(list)
    for r in exp_eval.values():
        by_exp[r.get("experiment","")].append(r)

    # Check if per-image data is available
    sample_r = next(iter(exp_eval.values()), {})
    has_per_image = "per_image_iou" in sample_r.get("global", {})
    results["data_availability"]["has_per_image_iou"] = has_per_image

    if not has_per_image:
        print("  " + "!"*60)
        print("  WARNING: per_image_iou not found in experiment_results.json")
        print("  You MUST apply the evaluate_experiments.py patch and re-run")
        print("  before per-image R analysis can proceed.")
        print("  Falling back to cell-level R (same as v1).")
        print("  " + "!"*60)
        return _compute_d3_fallback(by_exp, res)

    # === EXPERIMENT A: Decoder retraining ===
    if by_exp.get("a"):
        results["decoder_retraining"] = _analyze_experiment_per_image(
            by_exp["a"], "a", "Decoder retraining", res)

    # === EXPERIMENT B: BN swap ===
    if by_exp.get("b"):
        results["bn_swap"] = _analyze_experiment_per_image(
            by_exp["b"], "b", "BN swap", res)

    # Layer-wise BN
    for g in ["b_lw_early","b_lw_mid","b_lw_late"]:
        if by_exp.get(g):
            results[f"bn_{g}"] = _analyze_experiment_per_image(
                by_exp[g], g, f"BN {g}", res)

    # === EXPERIMENT C: Histogram matching ===
    if by_exp.get("c"):
        results["histogram_matching"] = _analyze_experiment_per_image(
            by_exp["c"], "c", "Histogram matching", res)

    # === DATA EFFICIENCY VARIANTS ===
    de_fracs = {"a_de5pct":0.05, "a_de10pct":0.10, "a_de15pct":0.15,
                "a_de20pct":0.20, "a":0.25}
    de_results = defaultdict(dict)
    for en, fr in de_fracs.items():
        for r in by_exp.get(en, []):
            pi = r.get("global",{}).get("per_image_iou",{})
            if not pi: continue
            R_arr, nt, nn, ng = compute_per_image_R(
                pi["experiment"], pi["loco"], pi["upper"])
            if len(R_arr) >= 5:
                med, mean, ci95 = cluster_bootstrap_ci(R_arr)
                de_results[(r["model"], r["holdout_city"])][fr] = {
                    "median_R": med, "mean_R": mean, "ci95": list(ci95), "n": len(R_arr)}
    if de_results:
        results["data_efficiency"] = {f"{m}_{c}": fd for (m,c), fd in de_results.items()}

    return results


def _analyze_experiment_per_image(exp_records, exp_name, exp_label, res):
    """
    Core per-image R analysis for one experiment type.
    Returns dict with per-model results, cross-architecture tests, and interaction matrix.
    """
    print(f"\n  --- Experiment {exp_name.upper()}: {exp_label} ---")

    per_model_R = defaultdict(list)  # model -> list of per-image R values
    per_cell = {}                     # model_city -> per-image R array + metadata

    for r in exp_records:
        model = r["model"]
        city = r["holdout_city"]
        cell_key = f"{model}_{city}"

        loco_iou = safe_float(r.get("global",{}).get("loco",{}).get("iou"))

        pi = r.get("global",{}).get("per_image_iou")
        if not pi:
            print(f"    SKIP {cell_key}: no per_image_iou data")
            continue

        R_arr, n_total, n_nan, n_gap = compute_per_image_R(
            pi["experiment"], pi["loco"], pi["upper"])

        if len(R_arr) < 5:
            print(f"    SKIP {cell_key}: only {len(R_arr)} valid images (need ≥5)")
            continue

        # Per-image statistics for this cell
        med, mean, ci95 = cluster_bootstrap_ci(R_arr)

        per_cell[cell_key] = {
            "model": model, "city": city,
            "median_R": med, "mean_R": mean, "ci95": list(ci95),
            "n_valid": len(R_arr), "n_total": n_total,
            "n_excluded_nan": n_nan, "n_excluded_gap": n_gap,
            "R_values": R_arr.tolist(),  # store for cross-arch tests
            "loco_iou": float(loco_iou),
            "upper_iou": safe_float(r.get("global",{}).get("upper",{}).get("iou")),
            "exp_iou": safe_float(r.get("global",{}).get("experiment",{}).get("iou")),
        }
        per_model_R[model].extend(R_arr.tolist())

        print(f"    {cell_key}: median R = {med:.3f} [{ci95[0]:.3f}, {ci95[1]:.3f}] "
              f"(n={len(R_arr)}/{n_total}, excl: {n_nan} NaN, {n_gap} gap)")

    # --- Per-model aggregate ---
    per_model_stats = {}
    for model in MODELS:
        R_all = np.array(per_model_R.get(model, []))
        if len(R_all) < 5:
            per_model_stats[model] = {"n": len(R_all), "note": "insufficient data"}
            continue
        med, mean, ci95 = cluster_bootstrap_ci(R_all)

        # Also compute fraction R >= 0 (net positive) and R >= 0.7 (strong recovery)
        frac_positive = float(np.mean(R_all >= 0))
        frac_strong = float(np.mean(R_all >= 0.7))

        per_model_stats[model] = {
            "median_R": med, "mean_R": mean, "ci95": list(ci95),
            "n": len(R_all),
            "R_min": float(np.min(R_all)), "R_max": float(np.max(R_all)),
            "R_q25": float(np.percentile(R_all, 25)),
            "R_q75": float(np.percentile(R_all, 75)),
            "frac_positive": frac_positive,
            "frac_strong_recovery": frac_strong,
        }
        print(f"  {model} pooled: median R = {med:.3f} [{ci95[0]:.3f}, {ci95[1]:.3f}] "
              f"(n={len(R_all)}, R∈[{np.min(R_all):.3f}, {np.max(R_all):.3f}])")

    # --- Cross-architecture tests ---
    cross_arch = {}

    # Kruskal-Wallis (non-parametric ANOVA, appropriate since R distributions
    # may be non-normal, especially for OGLANet which can have R < 0)
    model_groups = {m: np.array(per_model_R.get(m, []))
                    for m in MODELS if len(per_model_R.get(m, [])) >= 5}

    if len(model_groups) >= 2:
        groups = list(model_groups.values())
        H, kw_p = sp_stats.kruskal(*groups)
        cross_arch["kruskal_wallis"] = {
            "H": float(H), "p": float(kw_p),
            "n_groups": len(groups),
            "group_ns": {m: len(v) for m, v in model_groups.items()},
            "significant": bool(kw_p < 0.05),
        }
        print(f"\n  Kruskal-Wallis: H={H:.3f}, p={kw_p:.6f} "
              f"({'***' if kw_p<0.001 else ('*' if kw_p<0.05 else 'n.s.')})")

        # Pairwise Cliff's delta with bootstrap CI
        pairwise = {}
        model_list = sorted(model_groups.keys())
        for i in range(len(model_list)):
            for j in range(i+1, len(model_list)):
                m1, m2 = model_list[i], model_list[j]
                delta, interp = cliffs_delta(model_groups[m1], model_groups[m2])
                delta_med, delta_ci = cliffs_delta_bootstrap_ci(
                    model_groups[m1], model_groups[m2])

                # Check CI non-overlap (between the per-model bootstrap CIs)
                ci1 = per_model_stats.get(m1, {}).get("ci95", [float('nan'), float('nan')])
                ci2 = per_model_stats.get(m2, {}).get("ci95", [float('nan'), float('nan')])
                non_overlap = (ci1[0] > ci2[1]) or (ci2[0] > ci1[1])

                # Also run Mann-Whitney U as a complementary test
                U, mw_p = sp_stats.mannwhitneyu(
                    model_groups[m1], model_groups[m2], alternative='two-sided')

                pair_key = f"{m1}_vs_{m2}"
                pairwise[pair_key] = {
                    "delta": delta, "interpretation": interp,
                    "delta_ci95": list(delta_ci),
                    "mann_whitney_U": float(U), "mann_whitney_p": float(mw_p),
                    "ci_non_overlapping": non_overlap,
                    "n1": len(model_groups[m1]), "n2": len(model_groups[m2]),
                }
                print(f"  {pair_key}: Cliff's δ = {delta:.3f} ({interp}), "
                      f"CI [{delta_ci[0]:.3f}, {delta_ci[1]:.3f}], "
                      f"MW p={mw_p:.6f}, CIs {'non-overlapping' if non_overlap else 'overlapping'}")

        cross_arch["pairwise"] = pairwise

    result = {
        "per_cell": per_cell,
        "per_model": per_model_stats,
        "cross_architecture": cross_arch,
    }
    return result


def _compute_d3_fallback(by_exp, res):
    """
    Fallback: cell-level R only (same as v1) when per_image_iou is not available.
    Used before the evaluate_experiments.py patch is applied.
    """
    print("  [FALLBACK MODE — cell-level R only]")
    results = {"WARNING": "per_image_iou not available; using cell-level R (underpowered)"}

    for exp_name, records in by_exp.items():
        if exp_name not in ["a", "b", "c"]: continue
        am = defaultdict(list)
        for r in records:
            R = safe_float(r.get("global",{}).get("recovery",{}).get("iou"))
            am[r["model"]].append({"city": r["holdout_city"], "R": R})

        dr = {}
        for model in MODELS:
            Rv = [c["R"] for c in am.get(model,[]) if not np.isnan(c["R"])]
            if not Rv: continue
            m, ci = bootstrap_ci(Rv)
            dr[model] = {"cells": am[model], "R_mean": m, "R_ci": list(ci),
                         "R_min": float(min(Rv)), "R_max": float(max(Rv)), "n": len(Rv),
                         "WARNING": "n=3 per architecture — underpowered"}
            print(f"  Exp {exp_name.upper()} {model}: R={m:.3f} (n={len(Rv)}, FALLBACK)")
        results[f"experiment_{exp_name}"] = dr

    return results


# ============================================================
# SUMMARY
# ============================================================

def print_paper_summary(R, out_path=None):
    L = ["="*80, "PAPER-READY STATISTICS (v2 — robustness-hardened, no mIoU floor)", "="*80]
    d1, d2, d3 = R.get("diagnostic_1",{}), R.get("diagnostic_2",{}), R.get("diagnostic_3",{})

    L.append("\n--- §4.1 Intensity-Conditioned IoU ---")
    for k in ["spearman_monotonicity","gap_magnitude","fp_concentration","histogram_matching"]:
        s = d1.get(k,{}).get("summary")
        if s: L.append(f"  {k}: {s}")
    ad = d1.get("directionality",{})
    if ad:
        for m,v in ad.get("per_model",{}).items():
            L.append(f"  {m}: Δprec={v.get('prec_drop','?'):.3f}, Δrecall={v.get('recall_drop','?'):.3f}")
        L.append(f"  χ²={ad.get('chi2','?')}, p={ad.get('chi2_p','?')}")

    L.append("\n--- §4.2 Linear Probes ---")
    note = d2.get("methodological_note")
    if note: L.append(f"  NOTE: {note}")
    for k,v in d2.get("probe_accuracy",{}).items():
        s = "***" if v["p"]<0.001 else ("n.s." if v["p"]>=0.05 else "*")
        L.append(f"  {k}: {v['acc']:.3f} [{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] p={v['p']:.4f} {s}")
    for k,v in d2.get("permutation",{}).items():
        # Handle both old ('mean'/'ci95') and new ('perm_mean'/'perm_ci95') key names
        pm = v.get('perm_mean', v.get('mean', float('nan')))
        pci = v.get('perm_ci95', v.get('ci95', [float('nan'), float('nan')]))
        actual = v.get('actual_cv_acc')
        pval = v.get('p_value')
        line = f"  PERM {k}: null={pm:.3f} [{pci[0]:.3f},{pci[1]:.3f}]"
        if actual is not None:
            line += f", actual={actual:.3f}"
        if pval is not None:
            line += f", p={pval:.4f}"
        L.append(line)
    for m,v in d2.get("loco_vs_upper",{}).items():
        L.append(f"  {m}: upper={v['upper']:.3f}, loco={v['loco']:.3f}, Δ={v['delta']:+.3f}")

    L.append("\n--- §4.3 Controlled Interventions (PER-IMAGE R) ---")
    if d3.get("WARNING"):
        L.append(f"  WARNING: {d3['WARNING']}")

    for exp_key in ["decoder_retraining", "bn_swap", "histogram_matching"]:
        exp_data = d3.get(exp_key)
        if not exp_data: continue
        L.append(f"\n  {exp_key}:")

        # Per-model
        pm = exp_data.get("per_model", {})
        for m, v in pm.items():
            if "median_R" in v:
                L.append(f"    {m}: median R={v['median_R']:.3f} "
                         f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] "
                         f"(n={v['n']}, range=[{v['R_min']:.3f}, {v['R_max']:.3f}])")
            elif "note" in v:
                L.append(f"    {m}: {v['note']}")

        # Cross-architecture
        ca = exp_data.get("cross_architecture", {})
        kw = ca.get("kruskal_wallis")
        if kw:
            L.append(f"    Kruskal-Wallis: H={kw['H']:.3f}, p={kw['p']:.6f} "
                     f"({'SIG' if kw['significant'] else 'n.s.'})")
        for pk, pv in ca.get("pairwise", {}).items():
            L.append(f"    {pk}: Cliff's δ={pv['delta']:.3f} ({pv['interpretation']}), "
                     f"MW p={pv['mann_whitney_p']:.6f}, "
                     f"CIs {'NON-OVERLAP' if pv['ci_non_overlapping'] else 'overlap'}")

    text = "\n".join(L); print(text)
    if out_path:
        with open(out_path, "w") as f: f.write(text)
        print(f"\n  Saved to {out_path}")
    return text


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res", default="highres")
    p.add_argument("--permutation_n", type=int, default=100)
    p.add_argument("--skip_permutation", action="store_true")
    args = p.parse_args()

    t0 = time.time()

    R = {}
    R["diagnostic_1"] = compute_d1_statistics(args.res)
    R["diagnostic_2"] = compute_d2_statistics(args.res, args.permutation_n,
                                               args.skip_permutation)
    R["diagnostic_3"] = compute_d3_statistics(args.res)

    # Metadata
    R["metadata"] = {
        "version": "v2_robustness_hardened_no_floor",
        "min_gap_for_R": MIN_GAP_FOR_R,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "notes": [
            "D3 uses per-image R with cluster bootstrap (requires patched evaluate_experiments.py)",
            "D2 GroupKFold fix available: replace thread1_1d.py with v2 and _run_permutation with GroupKFold version",
            "All cells included — no mIoU floor / threshold on models considered",
        ]
    }

    out = os.path.join(OUTPUT_BASE, "paper_statistics"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "paper_statistics.json"), "w") as f:
        json.dump(make_serializable(R), f, indent=2)
    print(f"\n  JSON → {out}/paper_statistics.json")
    print_paper_summary(R, os.path.join(out, "paper_statistics.txt"))
    print(f"\n  Time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()