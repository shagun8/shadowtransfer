"""
Thread 3: Content-Based Attention vs. Geometry-Directed Attention

Diagnostics:
  3a — Boundary IoU vs. shadow elongation, conditioned on shadow category
  3b — Boundary IoU vs. shadow orientation (8 directional bins)
  3e — Per-city shadow geometry distributions (length, orientation, elongation, area)
"""
import os
import json
import numpy as np
from collections import defaultdict

from config import (
    CITIES, RESOLUTIONS, MODELS, LOCO_VARIANTS,
    SHADOW_TYPE_MAP, SHADOW_TYPE_SHORT,
    upper_pred_dir, loco_pred_dir, output_dir,
)
from utils import (
    load_city_data, load_predictions,
    extract_shadow_instances, compute_instance_geometry,
    compute_instance_boundary_metrics, fast_instance_metrics,
    safe_mean, safe_std,
    DIRECTION_BINS, angle_to_direction_bin,
)


# ============================================================
# HELPER: Build per-instance geometry + metrics table
# ============================================================

def build_instance_table(city_data, preds=None):
    """
    Build a table of per-instance geometry and (optionally) performance metrics.
    Uses pre-computed GT cache — no redundant instance extraction or geometry.
    
    If preds is None, only geometry is returned (for 3e distribution analysis).
    
    Returns:
        list of dicts, each with geometry fields and optionally metric fields
    """
    gt_cache = city_data["gt_cache"]
    records = []
    
    for i, img_cache in enumerate(gt_cache):
        pred = preds[i] if preds is not None else None
        
        for inst in img_cache["instances"]:
            if inst["shadow_type"] == 0:
                continue
            
            # Skip if geometry wasn't computed (too small)
            if "major_axis" not in inst:
                continue
            
            record = {
                "image_idx": i,
                "label_id": inst["label_id"],
                "shadow_type": inst["shadow_type"],
                "shadow_type_name": inst["shadow_type_name"],
                "area": inst["area"],
                "centroid_row": inst["centroid"][0],
                "centroid_col": inst["centroid"][1],
                "major_axis": inst["major_axis"],
                "minor_axis": inst["minor_axis"],
                "elongation": inst["elongation"],
                "orientation_deg": inst["orientation_deg"],
                "skeleton_length": inst["skeleton_length"],
            }
            
            # Fast metrics if predictions available
            if pred is not None:
                metrics = fast_instance_metrics(pred, inst)
                record.update({
                    "boundary_precision": metrics["precision"],
                    "boundary_recall": metrics["recall"],
                    "boundary_iou": metrics["iou"],
                    "boundary_f1": metrics["f1"],
                    "eval_pixels": metrics["eval_pixels"],
                })
            
            records.append(record)
    
    return records


# ============================================================
# 3a: BOUNDARY IoU vs. ELONGATION (conditioned on category)
# ============================================================

def run_3a_elongation_performance(records, label, n_bins=6):
    """
    Bin shadow instances by elongation ratio within each category.
    Compute boundary IoU per bin.
    
    We use area + elongation jointly:
    - Report separate curves per shadow type
    - Also report a 2D binning (area x elongation) for the combined view
    """
    if not records or "boundary_iou" not in records[0]:
        return {"label": label, "n_instances": 0}
    
    # Filter to instances with valid boundary metrics
    valid = [r for r in records if not np.isnan(r.get("boundary_iou", float('nan')))
             and r["eval_pixels"] >= 5]
    
    if not valid:
        return {"label": label, "n_instances": 0}
    
    # Determine elongation bin edges (across all instances)
    all_elong = np.array([r["elongation"] for r in valid])
    # Cap extreme elongations for binning
    cap = np.percentile(all_elong, 95)
    elong_edges = np.linspace(1.0, max(cap, 2.0), n_bins + 1)
    
    # Per-category elongation curves
    per_category = {}
    for stype in sorted(set(r["shadow_type"] for r in valid)):
        cat_records = [r for r in valid if r["shadow_type"] == stype]
        if len(cat_records) < 5:
            continue
        
        elongations = np.array([r["elongation"] for r in cat_records])
        ious = np.array([r["boundary_iou"] for r in cat_records])
        areas = np.array([r["area"] for r in cat_records])
        
        bin_idx = np.digitize(elongations, elong_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        bins = []
        for b in range(n_bins):
            in_bin = bin_idx == b
            count = int(in_bin.sum())
            if count == 0:
                bins.append({
                    "bin_low": float(elong_edges[b]),
                    "bin_high": float(elong_edges[b + 1]),
                    "count": 0,
                    "iou_mean": None, "iou_std": None,
                    "area_mean": None,
                })
            else:
                bins.append({
                    "bin_low": float(elong_edges[b]),
                    "bin_high": float(elong_edges[b + 1]),
                    "count": count,
                    "iou_mean": float(np.nanmean(ious[in_bin])),
                    "iou_std": float(np.nanstd(ious[in_bin])),
                    "area_mean": float(np.mean(areas[in_bin])),
                })
        
        per_category[int(stype)] = {
            "shadow_type_name": SHADOW_TYPE_MAP.get(stype, "Unknown"),
            "instance_count": len(cat_records),
            "bins": bins,
        }
    
    # Also: 2D binning (area x elongation) — combined across all types
    # Area bins: small / medium / large based on percentiles
    all_areas = np.array([r["area"] for r in valid])
    area_terciles = np.percentile(all_areas, [33, 67])
    area_labels = ["small", "medium", "large"]
    
    area_x_elong = {}
    for ai, alab in enumerate(area_labels):
        if ai == 0:
            area_mask = all_areas <= area_terciles[0]
        elif ai == 1:
            area_mask = (all_areas > area_terciles[0]) & (all_areas <= area_terciles[1])
        else:
            area_mask = all_areas > area_terciles[1]
        
        area_records = [valid[j] for j in range(len(valid)) if area_mask[j]]
        if len(area_records) < 5:
            continue
        
        elongations = np.array([r["elongation"] for r in area_records])
        ious = np.array([r["boundary_iou"] for r in area_records])
        
        bin_idx = np.digitize(elongations, elong_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        
        bins = []
        for b in range(n_bins):
            in_bin = bin_idx == b
            count = int(in_bin.sum())
            bins.append({
                "bin_low": float(elong_edges[b]),
                "bin_high": float(elong_edges[b + 1]),
                "count": count,
                "iou_mean": float(np.nanmean(ious[in_bin])) if count > 0 else None,
                "iou_std": float(np.nanstd(ious[in_bin])) if count > 0 else None,
            })
        
        area_x_elong[alab] = {
            "instance_count": len(area_records),
            "bins": bins,
        }
    
    return {
        "label": label,
        "n_instances": len(valid),
        "elongation_bin_edges": elong_edges.tolist(),
        "per_category": per_category,
        "area_x_elongation": area_x_elong,
        "area_tercile_thresholds": area_terciles.tolist(),
    }


def diagnostic_3a(city_data_cache):
    """Run 3a for all combos."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3a: Boundary IoU vs. Shadow Elongation")
    print("=" * 70)
    
    results = {}
    
    for res in RESOLUTIONS:
        for model in MODELS:
            for city in CITIES:
                data = city_data_cache.get((city, res))
                if data is None:
                    continue
                
                # Upper bound
                key = f"upper_{model}_{city}_{res}"
                pred_d = upper_pred_dir(city, res, model)
                preds = load_predictions(pred_d, data["filenames"])
                if preds is not None:
                    records = build_instance_table(data, preds)
                    results[key] = run_3a_elongation_performance(records, key)
                    print(f"  {key}: {results[key]['n_instances']} instances")
                
                # LOCO
                for variant in LOCO_VARIANTS:
                    key = f"loco_{model}_{variant}_{city}_{res}"
                    pred_d = loco_pred_dir(city, res, model, variant)
                    preds = load_predictions(pred_d, data["filenames"])
                    if preds is not None:
                        records = build_instance_table(data, preds)
                        results[key] = run_3a_elongation_performance(records, key)
                        print(f"  {key}: {results[key]['n_instances']} instances")
    
    out = output_dir("thread3", "3a_elongation_performance")
    with open(os.path.join(out, "elongation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved to {out}")
    return results


# ============================================================
# 3b: BOUNDARY IoU vs. ORIENTATION (8 directional bins)
# ============================================================

def run_3b_orientation_performance(records, label):
    """
    Bin shadow instances by orientation (8 compass directions).
    Compute boundary IoU per direction.
    
    Separate results per model to detect axis-alignment bias (MAMNet criss-cross).
    """
    valid = [r for r in records if not np.isnan(r.get("boundary_iou", float('nan')))
             and r["eval_pixels"] >= 5]
    
    if not valid:
        return {"label": label, "n_instances": 0}
    
    # Assign direction bins
    dir_bins = defaultdict(list)
    for r in valid:
        d_bin = angle_to_direction_bin(r["orientation_deg"])
        dir_bins[d_bin].append(r)
    
    direction_results = {}
    for d_idx in range(8):
        d_name = DIRECTION_BINS[d_idx]
        recs = dir_bins.get(d_idx, [])
        
        if len(recs) == 0:
            direction_results[d_name] = {
                "count": 0, "iou_mean": None, "iou_std": None,
                "recall_mean": None, "precision_mean": None,
            }
        else:
            direction_results[d_name] = {
                "count": len(recs),
                "iou_mean": float(np.nanmean([r["boundary_iou"] for r in recs])),
                "iou_std": float(np.nanstd([r["boundary_iou"] for r in recs])),
                "recall_mean": float(np.nanmean([r["boundary_recall"] for r in recs])),
                "precision_mean": float(np.nanmean([r["boundary_precision"] for r in recs])),
            }
    
    # Also per-category x direction (for detecting category-specific directional bias)
    per_cat_direction = {}
    for stype in sorted(set(r["shadow_type"] for r in valid)):
        cat_recs = [r for r in valid if r["shadow_type"] == stype]
        if len(cat_recs) < 10:
            continue
        
        cat_dir = {}
        for r in cat_recs:
            d_bin = angle_to_direction_bin(r["orientation_deg"])
            d_name = DIRECTION_BINS[d_bin]
            if d_name not in cat_dir:
                cat_dir[d_name] = []
            cat_dir[d_name].append(r["boundary_iou"])
        
        cat_dir_summary = {}
        for d_name in DIRECTION_BINS.values():
            vals = cat_dir.get(d_name, [])
            cat_dir_summary[d_name] = {
                "count": len(vals),
                "iou_mean": float(np.nanmean(vals)) if vals else None,
            }
        
        per_cat_direction[int(stype)] = {
            "shadow_type_name": SHADOW_TYPE_MAP.get(stype, "Unknown"),
            "directions": cat_dir_summary,
        }
    
    # Axis vs diagonal comparison (for detecting criss-cross bias)
    axis_dirs = ["N", "E", "S", "W"]
    diag_dirs = ["NE", "SE", "SW", "NW"]
    
    axis_ious = [r["boundary_iou"] for d in axis_dirs
                 for r in dir_bins.get(list(DIRECTION_BINS.values()).index(d), [])]
    diag_ious = [r["boundary_iou"] for d in diag_dirs
                 for r in dir_bins.get(list(DIRECTION_BINS.values()).index(d), [])]
    
    # Proper indexing
    axis_bin_indices = [i for i, name in DIRECTION_BINS.items() if name in axis_dirs]
    diag_bin_indices = [i for i, name in DIRECTION_BINS.items() if name in diag_dirs]
    
    axis_ious = [r["boundary_iou"] for idx in axis_bin_indices for r in dir_bins.get(idx, [])]
    diag_ious = [r["boundary_iou"] for idx in diag_bin_indices for r in dir_bins.get(idx, [])]
    
    axis_vs_diag = {
        "axis_iou_mean": float(np.nanmean(axis_ious)) if axis_ious else None,
        "diag_iou_mean": float(np.nanmean(diag_ious)) if diag_ious else None,
        "axis_count": len(axis_ious),
        "diag_count": len(diag_ious),
        "axis_minus_diag": (float(np.nanmean(axis_ious)) - float(np.nanmean(diag_ious)))
            if (axis_ious and diag_ious) else None,
    }
    
    return {
        "label": label,
        "n_instances": len(valid),
        "direction_results": direction_results,
        "per_category_direction": per_cat_direction,
        "axis_vs_diagonal": axis_vs_diag,
    }


def diagnostic_3b(city_data_cache):
    """Run 3b for all combos."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3b: Boundary IoU vs. Shadow Orientation")
    print("=" * 70)
    
    results = {}
    
    for res in RESOLUTIONS:
        for model in MODELS:
            for city in CITIES:
                data = city_data_cache.get((city, res))
                if data is None:
                    continue
                
                # Upper bound
                key = f"upper_{model}_{city}_{res}"
                pred_d = upper_pred_dir(city, res, model)
                preds = load_predictions(pred_d, data["filenames"])
                if preds is not None:
                    records = build_instance_table(data, preds)
                    results[key] = run_3b_orientation_performance(records, key)
                    avd = results[key].get('axis_vs_diagonal', {}).get('axis_minus_diag', 'N/A')
                    print(f"  {key}: {results[key]['n_instances']} instances, axis-diag={avd}")
                
                # LOCO
                for variant in LOCO_VARIANTS:
                    key = f"loco_{model}_{variant}_{city}_{res}"
                    pred_d = loco_pred_dir(city, res, model, variant)
                    preds = load_predictions(pred_d, data["filenames"])
                    if preds is not None:
                        records = build_instance_table(data, preds)
                        results[key] = run_3b_orientation_performance(records, key)
    
    out = output_dir("thread3", "3b_orientation_performance")
    with open(os.path.join(out, "orientation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved to {out}")
    return results


# ============================================================
# 3e: PER-CITY SHADOW GEOMETRY DISTRIBUTIONS
# ============================================================

def diagnostic_3e(city_data_cache):
    """
    Compute shadow geometry distributions per (city, res).
    This is GT-only — no predictions needed.
    
    Outputs: histograms of major_axis, elongation, orientation, area per city per category.
    Also: empirical sun azimuth estimate from building shadow orientations.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3e: Per-City Shadow Geometry Distributions")
    print("=" * 70)
    
    results = {}
    
    for res in RESOLUTIONS:
        for city in CITIES:
            data = city_data_cache.get((city, res))
            if data is None:
                continue
            
            key = f"{city}_{res}"
            records = build_instance_table(data, preds=None)
            
            if not records:
                results[key] = {"city": city, "res": res, "n_instances": 0}
                continue
            
            # Overall distributions
            all_major = [r["major_axis"] for r in records]
            all_elong = [r["elongation"] for r in records]
            all_orient = [r["orientation_deg"] for r in records]
            all_area = [r["area"] for r in records]
            all_skel = [r["skeleton_length"] for r in records]
            
            city_result = {
                "city": city,
                "res": res,
                "n_instances": len(records),
                "overall": {
                    "major_axis": {
                        "mean": float(np.mean(all_major)),
                        "median": float(np.median(all_major)),
                        "std": float(np.std(all_major)),
                        "percentiles": {
                            "25": float(np.percentile(all_major, 25)),
                            "75": float(np.percentile(all_major, 75)),
                            "95": float(np.percentile(all_major, 95)),
                        },
                    },
                    "elongation": {
                        "mean": float(np.mean(all_elong)),
                        "median": float(np.median(all_elong)),
                        "std": float(np.std(all_elong)),
                    },
                    "orientation_histogram": np.histogram(
                        all_orient, bins=np.linspace(0, 360, 9)  # 8 bins
                    )[0].tolist(),
                    "area": {
                        "mean": float(np.mean(all_area)),
                        "median": float(np.median(all_area)),
                        "std": float(np.std(all_area)),
                    },
                },
                "per_category": {},
            }
            
            # Per-category distributions
            for stype in sorted(set(r["shadow_type"] for r in records)):
                if stype == 0:
                    continue
                cat_recs = [r for r in records if r["shadow_type"] == stype]
                if len(cat_recs) < 3:
                    continue
                
                cat_major = [r["major_axis"] for r in cat_recs]
                cat_elong = [r["elongation"] for r in cat_recs]
                cat_orient = [r["orientation_deg"] for r in cat_recs]
                cat_area = [r["area"] for r in cat_recs]
                
                city_result["per_category"][int(stype)] = {
                    "shadow_type_name": SHADOW_TYPE_MAP.get(stype, "Unknown"),
                    "count": len(cat_recs),
                    "major_axis": {
                        "mean": float(np.mean(cat_major)),
                        "median": float(np.median(cat_major)),
                        "std": float(np.std(cat_major)),
                    },
                    "elongation": {
                        "mean": float(np.mean(cat_elong)),
                        "median": float(np.median(cat_elong)),
                        "std": float(np.std(cat_elong)),
                    },
                    "orientation_histogram": np.histogram(
                        cat_orient, bins=np.linspace(0, 360, 9)
                    )[0].tolist(),
                    "orientation_circular_mean": float(
                        np.degrees(np.arctan2(
                            np.mean(np.sin(np.radians(cat_orient))),
                            np.mean(np.cos(np.radians(cat_orient)))
                        )) % 360
                    ),
                    "area": {
                        "mean": float(np.mean(cat_area)),
                        "median": float(np.median(cat_area)),
                    },
                }
            
            # Empirical sun azimuth estimate from building shadows (type 1)
            building_recs = [r for r in records if r["shadow_type"] == 1]
            if len(building_recs) >= 10:
                b_orient = [r["orientation_deg"] for r in building_recs]
                sun_azimuth_est = float(
                    np.degrees(np.arctan2(
                        np.mean(np.sin(np.radians(b_orient))),
                        np.mean(np.cos(np.radians(b_orient)))
                    )) % 360
                )
                city_result["empirical_sun_azimuth_from_buildings"] = sun_azimuth_est
                print(f"  {key}: {len(records)} instances, "
                      f"est. sun azimuth={sun_azimuth_est:.1f}°")
            else:
                print(f"  {key}: {len(records)} instances")
            
            results[key] = city_result
    
    out = output_dir("thread3", "3e_geometry_distributions")
    with open(os.path.join(out, "geometry_distributions.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved to {out}")
    return results


# ============================================================
# ENTRY POINT
# ============================================================

def run_all_thread3(city_data_cache):
    """Run all Thread 3 diagnostics."""
    results_3e = diagnostic_3e(city_data_cache)  # GT-only, run first
    results_3a = diagnostic_3a(city_data_cache)
    results_3b = diagnostic_3b(city_data_cache)
    
    return {
        "3a": results_3a,
        "3b": results_3b,
        "3e": results_3e,
    }