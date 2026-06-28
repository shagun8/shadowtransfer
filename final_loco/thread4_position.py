"""
Thread 4: Positional Encoding Fragility Under Domain Shift

Diagnostics:
  4a — Spatial error heatmaps (ΔIoU per spatial region, same-city vs cross-city)
  4b — Shadow spatial density divergence (JSD between cities)
"""
import os
import json
import numpy as np
from scipy.spatial.distance import jensenshannon

from config import (
    CITIES, RESOLUTIONS, MODELS, LOCO_VARIANTS,
    IMG_SIZE, upper_pred_dir, loco_pred_dir, output_dir,
)
from utils import (
    load_city_data, load_predictions,
    compute_boundary_band,
    compute_boundary_metrics,
)


# ============================================================
# 4a: SPATIAL ERROR HEATMAPS
# ============================================================

def compute_spatial_grid_metrics(city_data, preds, label, grid_size=2, min_images=3):
    """
    Divide images into a grid_size x grid_size grid.
    Compute boundary-band IoU per grid cell.
    
    Only include a cell if at least `min_images` contribute boundary-band pixels to it.
    
    Also compute radial bin metrics (center vs periphery).
    
    Returns:
        dict with grid metrics and radial metrics
    """
    n_cells = grid_size * grid_size
    cell_h = IMG_SIZE // grid_size
    cell_w = IMG_SIZE // grid_size
    
    # Accumulators per cell: tp, fp, fn, and image contribution count
    cell_tp = np.zeros(n_cells, dtype=np.int64)
    cell_fp = np.zeros(n_cells, dtype=np.int64)
    cell_fn = np.zeros(n_cells, dtype=np.int64)
    cell_image_count = np.zeros(n_cells, dtype=np.int64)  # how many images have band pixels in this cell
    
    # Radial bins: 3 concentric rings
    # Center of image
    cy, cx = IMG_SIZE / 2, IMG_SIZE / 2
    max_r = np.sqrt(cy**2 + cx**2)
    radial_edges = np.linspace(0, max_r, 4)  # 3 bins
    n_radial = 3
    radial_tp = np.zeros(n_radial, dtype=np.int64)
    radial_fp = np.zeros(n_radial, dtype=np.int64)
    radial_fn = np.zeros(n_radial, dtype=np.int64)
    radial_image_count = np.zeros(n_radial, dtype=np.int64)
    
    # Pre-compute pixel-to-cell and pixel-to-radial mappings
    rows, cols = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cell_row_idx = np.minimum(rows // cell_h, grid_size - 1)
    cell_col_idx = np.minimum(cols // cell_w, grid_size - 1)
    cell_idx_map = cell_row_idx * grid_size + cell_col_idx  # (H, W) -> cell index
    
    dist_from_center = np.sqrt((rows - cy)**2 + (cols - cx)**2)
    radial_idx_map = np.digitize(dist_from_center, radial_edges) - 1
    radial_idx_map = np.clip(radial_idx_map, 0, n_radial - 1)
    
    for i, (gt_bin, pred) in enumerate(zip(city_data["gt_binary"], preds)):
        if pred is None:
            continue
        
        band = city_data["gt_cache"][i]["band"]
        eval_mask = ~band
        if eval_mask.sum() == 0:
            continue
        
        pred_eval_pixels = pred[eval_mask]
        gt_eval_pixels = gt_bin[eval_mask]
        eval_cell_indices = cell_idx_map[eval_mask]
        eval_radial_indices = radial_idx_map[eval_mask]
        
        tp_mask = (pred_eval_pixels == 1) & (gt_eval_pixels == 1)
        fp_mask = (pred_eval_pixels == 1) & (gt_eval_pixels == 0)
        fn_mask = (pred_eval_pixels == 0) & (gt_eval_pixels == 1)
        
        # Per-cell accumulation
        cells_with_eval = set()
        for c in range(n_cells):
            c_mask = eval_cell_indices == c
            if c_mask.sum() == 0:
                continue
            cells_with_eval.add(c)
            cell_tp[c] += tp_mask[c_mask].sum()
            cell_fp[c] += fp_mask[c_mask].sum()
            cell_fn[c] += fn_mask[c_mask].sum()
        
        for c in cells_with_eval:
            cell_image_count[c] += 1
        
        # Per-radial accumulation
        radials_with_eval = set()
        for r in range(n_radial):
            r_mask = eval_radial_indices == r
            if r_mask.sum() == 0:
                continue
            radials_with_eval.add(r)
            radial_tp[r] += tp_mask[r_mask].sum()
            radial_fp[r] += fp_mask[r_mask].sum()
            radial_fn[r] += fn_mask[r_mask].sum()
        
        for r in radials_with_eval:
            radial_image_count[r] += 1
    
    # Compute IoU per cell
    grid_results = []
    for c in range(n_cells):
        row_idx = c // grid_size
        col_idx = c % grid_size
        
        if cell_image_count[c] < min_images:
            grid_results.append({
                "cell": c, "row": row_idx, "col": col_idx,
                "sufficient_data": False,
                "image_count": int(cell_image_count[c]),
            })
        else:
            tp, fp, fn = int(cell_tp[c]), int(cell_fp[c]), int(cell_fn[c])
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
            precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
            recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            
            grid_results.append({
                "cell": c, "row": row_idx, "col": col_idx,
                "sufficient_data": True,
                "image_count": int(cell_image_count[c]),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "tp": tp, "fp": fp, "fn": fn,
            })
    
    # Compute IoU per radial bin
    radial_results = []
    radial_labels = ["center", "mid-ring", "periphery"]
    for r in range(n_radial):
        if radial_image_count[r] < min_images:
            radial_results.append({
                "ring": radial_labels[r],
                "sufficient_data": False,
                "image_count": int(radial_image_count[r]),
            })
        else:
            tp, fp, fn = int(radial_tp[r]), int(radial_fp[r]), int(radial_fn[r])
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
            
            radial_results.append({
                "ring": radial_labels[r],
                "sufficient_data": True,
                "image_count": int(radial_image_count[r]),
                "iou": float(iou),
                "tp": tp, "fp": fp, "fn": fn,
            })
    
    return {
        "label": label,
        "grid_size": grid_size,
        "grid": grid_results,
        "radial": radial_results,
    }


def diagnostic_4a(city_data_cache):
    """
    Run 4a: spatial error heatmaps for upper vs LOCO.
    Compute ΔIoU = upper_IoU - loco_IoU per spatial cell.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4a: Spatial Error Heatmaps")
    print("=" * 70)
    
    raw_results = {}
    
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
                    raw_results[key] = compute_spatial_grid_metrics(data, preds, key)
                    print(f"  {key}: computed")
                
                # LOCO
                for variant in LOCO_VARIANTS:
                    key = f"loco_{model}_{variant}_{city}_{res}"
                    pred_d = loco_pred_dir(city, res, model, variant)
                    preds = load_predictions(pred_d, data["filenames"])
                    if preds is not None:
                        raw_results[key] = compute_spatial_grid_metrics(data, preds, key)
    
    # Compute ΔIoU maps
    delta_results = {}
    for res in RESOLUTIONS:
        for model in MODELS:
            for variant in LOCO_VARIANTS:
                for city in CITIES:
                    upper_key = f"upper_{model}_{city}_{res}"
                    loco_key = f"loco_{model}_{variant}_{city}_{res}"
                    
                    if upper_key not in raw_results or loco_key not in raw_results:
                        continue
                    
                    upper_grid = raw_results[upper_key]["grid"]
                    loco_grid = raw_results[loco_key]["grid"]
                    
                    delta_grid = []
                    for ug, lg in zip(upper_grid, loco_grid):
                        if ug.get("sufficient_data") and lg.get("sufficient_data"):
                            u_iou = ug.get("iou", float('nan'))
                            l_iou = lg.get("iou", float('nan'))
                            
                            if not np.isnan(u_iou) and not np.isnan(l_iou):
                                delta = u_iou - l_iou
                            else:
                                delta = float('nan')
                            
                            delta_grid.append({
                                "cell": ug["cell"],
                                "row": ug["row"],
                                "col": ug["col"],
                                "upper_iou": u_iou,
                                "loco_iou": l_iou,
                                "delta_iou": float(delta),
                            })
                        else:
                            delta_grid.append({
                                "cell": ug["cell"],
                                "row": ug["row"],
                                "col": ug["col"],
                                "sufficient_data": False,
                            })
                    
                    # Radial delta
                    upper_radial = raw_results[upper_key]["radial"]
                    loco_radial = raw_results[loco_key]["radial"]
                    
                    delta_radial = []
                    for ur, lr in zip(upper_radial, loco_radial):
                        if ur.get("sufficient_data") and lr.get("sufficient_data"):
                            u_iou = ur.get("iou", float('nan'))
                            l_iou = lr.get("iou", float('nan'))
                            delta = u_iou - l_iou if not (np.isnan(u_iou) or np.isnan(l_iou)) else float('nan')
                            delta_radial.append({
                                "ring": ur["ring"],
                                "upper_iou": u_iou,
                                "loco_iou": l_iou,
                                "delta_iou": float(delta),
                            })
                        else:
                            delta_radial.append({
                                "ring": ur["ring"],
                                "sufficient_data": False,
                            })
                    
                    delta_key = f"delta_{model}_{variant}_{city}_{res}"
                    delta_results[delta_key] = {
                        "grid": delta_grid,
                        "radial": delta_radial,
                    }
    
    # Save
    out = output_dir("thread4", "4a_spatial_error_maps")
    with open(os.path.join(out, "spatial_raw.json"), "w") as f:
        json.dump(raw_results, f, indent=2)
    with open(os.path.join(out, "spatial_deltas.json"), "w") as f:
        json.dump(delta_results, f, indent=2)
    
    print(f"  Saved to {out}")
    return raw_results, delta_results


# ============================================================
# 4b: SHADOW SPATIAL DENSITY DIVERGENCE
# ============================================================

def compute_shadow_density_map(city_data, grid_size=8):
    """
    Compute a spatial density map of shadow pixels across all images.
    Returns a grid_size x grid_size array of shadow pixel fractions.
    """
    cell_h = IMG_SIZE // grid_size
    cell_w = IMG_SIZE // grid_size
    
    shadow_counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    total_counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    
    for gt_bin in city_data["gt_binary"]:
        for r in range(grid_size):
            for c in range(grid_size):
                r_start, r_end = r * cell_h, min((r + 1) * cell_h, IMG_SIZE)
                c_start, c_end = c * cell_w, min((c + 1) * cell_w, IMG_SIZE)
                
                cell = gt_bin[r_start:r_end, c_start:c_end]
                shadow_counts[r, c] += cell.sum()
                total_counts[r, c] += cell.size
    
    # Normalize to get density (fraction of shadow pixels)
    density = shadow_counts / np.maximum(total_counts, 1)
    return density


def diagnostic_4b(city_data_cache):
    """
    Compute shadow spatial density per city and JSD between city pairs.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4b: Shadow Spatial Density Divergence")
    print("=" * 70)
    
    grid_size = 8
    densities = {}
    
    for res in RESOLUTIONS:
        for city in CITIES:
            data = city_data_cache.get((city, res))
            if data is None:
                continue
            
            key = f"{city}_{res}"
            density = compute_shadow_density_map(data, grid_size)
            densities[key] = density
            
            print(f"  {key}: shadow density range [{density.min():.3f}, {density.max():.3f}]")
    
    # Compute JSD between all city pairs (same resolution)
    jsd_results = {}
    for res in RESOLUTIONS:
        for i, city_a in enumerate(CITIES):
            for city_b in CITIES[i + 1:]:
                key_a = f"{city_a}_{res}"
                key_b = f"{city_b}_{res}"
                
                if key_a not in densities or key_b not in densities:
                    continue
                
                # Flatten density maps to distributions
                dist_a = densities[key_a].flatten()
                dist_b = densities[key_b].flatten()
                
                # Normalize to sum to 1 (probability distributions)
                dist_a = dist_a / dist_a.sum() if dist_a.sum() > 0 else dist_a
                dist_b = dist_b / dist_b.sum() if dist_b.sum() > 0 else dist_b
                
                jsd = float(jensenshannon(dist_a, dist_b))
                
                pair_key = f"{city_a}_vs_{city_b}_{res}"
                jsd_results[pair_key] = {
                    "city_a": city_a,
                    "city_b": city_b,
                    "res": res,
                    "jsd": jsd,
                }
                print(f"  JSD({city_a}, {city_b}) @ {res}: {jsd:.4f}")
    
    # Save
    out = output_dir("thread4", "4b_spatial_density")
    
    # Save density maps as nested lists for JSON serialization
    density_json = {k: v.tolist() for k, v in densities.items()}
    with open(os.path.join(out, "density_maps.json"), "w") as f:
        json.dump(density_json, f, indent=2)
    with open(os.path.join(out, "jsd_results.json"), "w") as f:
        json.dump(jsd_results, f, indent=2)
    
    # Also save density maps as numpy arrays for easy plotting
    for k, v in densities.items():
        np.save(os.path.join(out, f"density_{k}.npy"), v)
    
    print(f"  Saved to {out}")
    return densities, jsd_results


# ============================================================
# ENTRY POINT
# ============================================================

def run_all_thread4(city_data_cache):
    """Run all Thread 4 diagnostics."""
    results_4a_raw, results_4a_delta = diagnostic_4a(city_data_cache)
    densities, results_4b = diagnostic_4b(city_data_cache)
    
    return {
        "4a_raw": results_4a_raw,
        "4a_delta": results_4a_delta,
        "4b_densities": densities,
        "4b_jsd": results_4b,
    }