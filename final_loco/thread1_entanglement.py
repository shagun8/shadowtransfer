"""
Thread 1: Feature-Level Entanglement of Causal vs. Spurious Factors

Diagnostics:
  1a — FP composition clustering (k-means on RGB of false-positive boundary pixels)
  1b — Intensity-conditioned performance (boundary IoU vs. interior median intensity)
  1c — Per-class recall drop (per shadow type, same-city vs cross-city)
"""
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

from config import (
    CITIES, RESOLUTIONS, MODELS, LOCO_VARIANTS,
    SHADOW_TYPE_MAP, SHADOW_TYPE_SHORT,
    upper_pred_dir, loco_pred_dir, output_dir,
)
from utils import (
    load_city_data, load_predictions,
    compute_boundary_band, compute_eroded_mask,
    extract_shadow_instances, compute_instance_boundary_band,
    compute_boundary_metrics, compute_instance_boundary_metrics,
    compute_instance_eval_mask, fast_instance_metrics,
    safe_mean, safe_std, bin_values, BOUNDARY_WIDTH,
)
try:
    from thread1_1d import diagnostic_1d
    HAS_1D = True
except ImportError:
    HAS_1D = False


# ============================================================
# 1a: FP COMPOSITION CLUSTERING
# ============================================================

def run_1a_fp_clustering(city_data, preds, label,
                          n_clusters=5, max_fp_pixels=50000):
    """
    Cluster false-positive boundary-band pixels by their RGB values.

    Args:
        city_data:     dict from load_city_data
        preds:         list of prediction masks (0/1)
        label:         string label for this (model, variant) combo
        n_clusters:    number of k-means clusters
        max_fp_pixels: cap on FP pixels to cluster (for speed)

    Returns:
        dict with cluster_centers, cluster_proportions,
        cluster_mean_intensity, total_fp_pixels
    """
    fp_rgb_list = []
    gt_cache    = city_data["gt_cache"]

    for i, (gt_bin, pred, rgb_img) in enumerate(
        zip(city_data["gt_binary"], preds, city_data["images"])
    ):
        if pred is None:
            continue

        band = gt_cache[i]["band"]

        # FP: predicted shadow in definite non-shadow region (outside ±band)
        fp_mask = (~band) & (pred == 1) & (gt_bin == 0)

        if fp_mask.sum() > 0:
            fp_rgb_list.append(rgb_img[fp_mask])

    if not fp_rgb_list:
        return {"total_fp_pixels": 0, "label": label}

    all_fp_rgb = np.concatenate(fp_rgb_list, axis=0)
    total_fp   = len(all_fp_rgb)

    # Deterministic subsampling
    if total_fp > max_fp_pixels:
        indices    = np.linspace(0, total_fp - 1, max_fp_pixels, dtype=int)
        all_fp_rgb = all_fp_rgb[indices]

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_fp_rgb)

    counts       = np.bincount(cluster_labels, minlength=n_clusters)
    proportions  = counts / counts.sum()
    centers      = kmeans.cluster_centers_
    mean_intens  = centers.mean(axis=1)

    # Sort by mean intensity for consistent cross-city comparison
    sort_idx = np.argsort(mean_intens)

    return {
        "label":                label,
        "total_fp_pixels":      total_fp,
        "n_clusters":           n_clusters,
        "cluster_centers_rgb":  centers[sort_idx].tolist(),
        "cluster_proportions":  proportions[sort_idx].tolist(),
        "cluster_mean_intensity": mean_intens[sort_idx].tolist(),
    }


def diagnostic_1a(city_data_cache):
    """Run 1a for all (city, res, model, variant) combos."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1a: FP Composition Clustering")
    print("=" * 70)

    results = {}

    for res in RESOLUTIONS:
        for model in MODELS:
            # Upper bound (same-city)
            for city in CITIES:
                key  = f"upper_{model}_{city}_{res}"
                data = city_data_cache.get((city, res))
                if data is None:
                    continue

                pred_d = upper_pred_dir(city, res, model)
                preds  = load_predictions(pred_d, data["filenames"])
                if preds is None:
                    print(f"  SKIP {key}: no predictions")
                    continue

                result = run_1a_fp_clustering(data, preds, key)
                results[key] = result
                print(f"  {key}: {result['total_fp_pixels']} FP pixels")

            # LOCO variants
            # Variants: vanilla, fda, segdesic, iim, isw, mrfp_plus, fada
            for variant in LOCO_VARIANTS:
                for city in CITIES:
                    key  = f"loco_{model}_{variant}_{city}_{res}"
                    data = city_data_cache.get((city, res))
                    if data is None:
                        continue

                    pred_d = loco_pred_dir(city, res, model, variant)
                    preds  = load_predictions(pred_d, data["filenames"])
                    if preds is None:
                        print(f"  SKIP {key}: no predictions")
                        continue

                    result = run_1a_fp_clustering(data, preds, key)
                    results[key] = result
                    print(f"  {key}: {result['total_fp_pixels']} FP pixels")

    out = output_dir("thread1", "1a_fp_clustering")
    with open(os.path.join(out, "fp_clustering_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {out}")
    return results


# ============================================================
# 1b: INTENSITY-CONDITIONED PERFORMANCE
# ============================================================

def run_1b_intensity_curves(city_data, preds, label, n_bins=8):
    """
    Compute boundary-band performance as a function of shadow interior
    median intensity.

    Uses QUANTILE-BASED bins so each bin has roughly equal instance count.
    Aggregation: per-image mean (not dataset-level pixel pooling).
    """
    instance_records = []
    gt_cache = city_data["gt_cache"]

    for i, pred in enumerate(preds):
        if pred is None:
            continue

        for inst in gt_cache[i]["instances"]:
            median_intensity = inst.get("median_intensity", np.nan)
            if np.isnan(median_intensity):
                continue

            metrics = fast_instance_metrics(pred, inst)
            if metrics["eval_pixels"] < 5:
                continue

            instance_records.append({
                "median_intensity": median_intensity,
                "precision":    metrics["precision"],
                "recall":       metrics["recall"],
                "iou":          metrics["iou"],
                "area":         inst["area"],
                "shadow_type":  inst["shadow_type"],
            })

    if not instance_records:
        return {"label": label, "n_instances": 0}

    # Quantile-based binning
    intensities    = np.array([r["median_intensity"] for r in instance_records])
    quantile_points = np.linspace(0, 100, n_bins + 1)
    bin_edges      = np.unique(np.percentile(intensities, quantile_points))
    actual_n_bins  = len(bin_edges) - 1

    if actual_n_bins < 2:
        return {"label": label, "n_instances": len(instance_records),
                "n_bins": 0, "bins": []}

    bin_indices = np.digitize(intensities, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, actual_n_bins - 1)

    bin_results = []
    for b in range(actual_n_bins):
        in_bin = [instance_records[j]
                  for j in range(len(instance_records)) if bin_indices[j] == b]
        center = float((bin_edges[b] + bin_edges[b + 1]) / 2)

        if len(in_bin) == 0:
            bin_results.append({
                "bin_center": center,
                "bin_low":    float(bin_edges[b]),
                "bin_high":   float(bin_edges[b + 1]),
                "count": 0,
                "precision_mean": None,
                "recall_mean":    None,
                "iou_mean":       None,
            })
        else:
            bin_results.append({
                "bin_center":   center,
                "bin_low":      float(bin_edges[b]),
                "bin_high":     float(bin_edges[b + 1]),
                "count":        len(in_bin),
                "precision_mean": safe_mean([r["precision"] for r in in_bin]),
                "recall_mean":    safe_mean([r["recall"]    for r in in_bin]),
                "iou_mean":       safe_mean([r["iou"]       for r in in_bin]),
                "precision_std":  safe_std( [r["precision"] for r in in_bin]),
                "recall_std":     safe_std( [r["recall"]    for r in in_bin]),
                "iou_std":        safe_std( [r["iou"]       for r in in_bin]),
            })

    return {
        "label":           label,
        "n_instances":     len(instance_records),
        "n_bins":          actual_n_bins,
        "bin_edges":       bin_edges.tolist(),
        "bins":            bin_results,
        "raw_intensities": intensities.tolist(),
        "raw_ious":        [r["iou"] for r in instance_records],
        "raw_precisions":  [r["precision"] for r in instance_records],
        "raw_recalls":     [r["recall"] for r in instance_records],
    }


def diagnostic_1b(city_data_cache):
    """Run 1b for all combos."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1b: Intensity-Conditioned Performance Curves")
    print("=" * 70)

    results = {}

    for res in RESOLUTIONS:
        for model in MODELS:
            # Upper bound
            for city in CITIES:
                key  = f"upper_{model}_{city}_{res}"
                data = city_data_cache.get((city, res))
                if data is None:
                    continue

                pred_d = upper_pred_dir(city, res, model)
                preds  = load_predictions(pred_d, data["filenames"])
                if preds is None:
                    continue

                result = run_1b_intensity_curves(data, preds, key)
                results[key] = result
                print(f"  {key}: {result['n_instances']} instances")

            # LOCO
            for variant in LOCO_VARIANTS:
                for city in CITIES:
                    key  = f"loco_{model}_{variant}_{city}_{res}"
                    data = city_data_cache.get((city, res))
                    if data is None:
                        continue

                    pred_d = loco_pred_dir(city, res, model, variant)
                    preds  = load_predictions(pred_d, data["filenames"])
                    if preds is None:
                        continue

                    result = run_1b_intensity_curves(data, preds, key)
                    results[key] = result
                    print(f"  {key}: {result['n_instances']} instances")

    out = output_dir("thread1", "1b_intensity_curves")
    with open(os.path.join(out, "intensity_curve_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {out}")
    return results


# ============================================================
# 1c: PER-CLASS RECALL DROP
# ============================================================

def run_1c_per_class_metrics(city_data, preds, label):
    """
    Compute boundary-band recall per shadow type.
    Aggregation: per-image mean (not dataset-level pixel pooling).

    Returns:
        dict mapping shadow_type -> {recall, precision, f1, iou, instance_count}
    """
    type_tp    = defaultdict(int)
    type_fn    = defaultdict(int)
    type_fp    = defaultdict(int)
    type_count = defaultdict(int)
    gt_cache   = city_data["gt_cache"]

    for i, pred in enumerate(preds):
        if pred is None:
            continue

        for inst in gt_cache[i]["instances"]:
            stype = inst["shadow_type"]
            if stype == 0:
                continue

            metrics = fast_instance_metrics(pred, inst)
            if metrics["eval_pixels"] == 0:
                continue

            type_tp[stype]    += metrics["tp"]
            type_fn[stype]    += metrics["fn"]
            type_fp[stype]    += metrics["fp"]
            type_count[stype] += 1

    per_class = {}
    for stype in sorted(type_count.keys()):
        tp = type_tp[stype]
        fn = type_fn[stype]
        fp = type_fp[stype]

        recall    = tp / (tp + fn)          if (tp + fn)          > 0 else float('nan')
        precision = tp / (tp + fp)          if (tp + fp)          > 0 else float('nan')
        f1        = 2*tp/(2*tp + fp + fn)   if (2*tp + fp + fn)   > 0 else float('nan')
        iou       = tp / (tp + fp + fn)     if (tp + fp + fn)     > 0 else float('nan')

        per_class[stype] = {
            "shadow_type_name": SHADOW_TYPE_MAP.get(stype, "Unknown"),
            "recall":           float(recall),
            "precision":        float(precision),
            "f1":               float(f1),
            "iou":              float(iou),
            "instance_count":   type_count[stype],
            "tp": tp, "fn": fn, "fp": fp,
        }

    return {"label": label, "per_class": per_class}


def diagnostic_1c(city_data_cache):
    """Run 1c: per-class recall for upper vs LOCO, then compute recall drops."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1c: Per-Class Recall Drop (Same-City vs Cross-City)")
    print("=" * 70)

    raw_results = {}

    for res in RESOLUTIONS:
        for model in MODELS:
            # Upper bound
            for city in CITIES:
                key  = f"upper_{model}_{city}_{res}"
                data = city_data_cache.get((city, res))
                if data is None:
                    continue

                pred_d = upper_pred_dir(city, res, model)
                preds  = load_predictions(pred_d, data["filenames"])
                if preds is None:
                    continue

                result = run_1c_per_class_metrics(data, preds, key)
                raw_results[key] = result
                print(f"  {key}: {len(result['per_class'])} shadow types")

            # LOCO
            for variant in LOCO_VARIANTS:
                for city in CITIES:
                    key  = f"loco_{model}_{variant}_{city}_{res}"
                    data = city_data_cache.get((city, res))
                    if data is None:
                        continue

                    pred_d = loco_pred_dir(city, res, model, variant)
                    preds  = load_predictions(pred_d, data["filenames"])
                    if preds is None:
                        continue

                    result = run_1c_per_class_metrics(data, preds, key)
                    raw_results[key] = result

    # Compute recall drops: Δrecall = upper_recall - loco_recall per class
    drop_results = {}
    for res in RESOLUTIONS:
        for model in MODELS:
            for variant in LOCO_VARIANTS:
                for city in CITIES:
                    upper_key = f"upper_{model}_{city}_{res}"
                    loco_key  = f"loco_{model}_{variant}_{city}_{res}"

                    if upper_key not in raw_results or loco_key not in raw_results:
                        continue

                    upper_cls = raw_results[upper_key]["per_class"]
                    loco_cls  = raw_results[loco_key]["per_class"]

                    drops = {}
                    for stype in set(list(upper_cls.keys()) + list(loco_cls.keys())):
                        u_recall = upper_cls.get(stype, {}).get("recall", float('nan'))
                        l_recall = loco_cls.get(stype,  {}).get("recall", float('nan'))

                        if np.isnan(u_recall) or np.isnan(l_recall):
                            delta      = float('nan')
                            norm_delta = float('nan')
                        else:
                            delta      = u_recall - l_recall
                            norm_delta = delta / u_recall if u_recall > 0 else float('nan')

                        drops[int(stype)] = {
                            "shadow_type_name": SHADOW_TYPE_MAP.get(stype, "Unknown"),
                            "upper_recall":     float(u_recall),
                            "loco_recall":      float(l_recall),
                            "delta_recall":     float(delta),
                            "normalized_delta": float(norm_delta),
                            "upper_instances":  upper_cls.get(stype, {}).get("instance_count", 0),
                            "loco_instances":   loco_cls.get(stype,  {}).get("instance_count", 0),
                        }

                    drop_results[f"drop_{model}_{variant}_{city}_{res}"] = drops

    # Save
    out = output_dir("thread1", "1c_class_recall_drop")
    with open(os.path.join(out, "per_class_raw.json"),   "w") as f:
        json.dump(raw_results,  f, indent=2, default=str)
    with open(os.path.join(out, "per_class_drops.json"), "w") as f:
        json.dump(drop_results, f, indent=2, default=str)

    print(f"  Saved to {out}")
    return raw_results, drop_results


# ============================================================
# ENTRY POINT
# ============================================================

def run_all_thread1(city_data_cache):
    """Run all Thread 1 diagnostics."""
    results_1a               = diagnostic_1a(city_data_cache)
    results_1b               = diagnostic_1b(city_data_cache)
    results_1c_raw, results_1c_drops = diagnostic_1c(city_data_cache)

    results_1d = {}
    if HAS_1D:
        try:
            results_1d = diagnostic_1d()
        except Exception as e:
            print(f"  1d skipped: {e}")
    else:
        print("  1d skipped: thread1_1d.py not available or "
              "sklearn not installed")

    return {
        "1a":       results_1a,
        "1b":       results_1b,
        "1c_raw":   results_1c_raw,
        "1c_drops": results_1c_drops,
        "1d":       results_1d,
    }