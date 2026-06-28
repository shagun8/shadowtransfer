"""
Core utilities for diagnostic analysis.
Handles data loading, boundary band computation, instance extraction, and metrics.

Metric convention (matches BDRAR / MTMT / DSDNet / SCOTCH):
  - Primary metric: tolerant boundary IoU — pixels within ±BOUNDARY_WIDTH of
    the GT shadow edge are excluded from evaluation (don't-care band).
  - Aggregation: per-image mean, NOT dataset-level pixel pooling.
    i.e.  final_IoU = mean_over_images( per_image_IoU )
"""
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, label as sk_label
from skimage.morphology import skeletonize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from config import (
    IMG_SIZE, BOUNDARY_WIDTH, MIN_INSTANCE_AREA,
    SHADOW_TYPE_MAP, SHADOW_TYPE_SHORT,
    gt_mask_dir, gt_multiclass_dir, image_dir,
    upper_pred_dir, loco_pred_dir,
)


# ============================================================
# DATA LOADING
# ============================================================

def load_mask(path, binarize=False):
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    if binarize:
        arr = (arr > 127).astype(np.uint8)
    return arr


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def get_filenames(directory):
    if not os.path.isdir(directory):
        return []
    return sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])


def load_city_data(city, res):
    mask_dir = gt_mask_dir(city, res)
    mc_dir   = gt_multiclass_dir(city, res)
    img_dir  = image_dir(city, res)

    fnames = get_filenames(mask_dir)
    if not fnames:
        print(f"  WARNING: No GT masks found in {mask_dir}")
        return None

    data = {"filenames": fnames, "gt_binary": [], "gt_multiclass": [], "images": []}

    for fn in fnames:
        gt_bin = load_mask(os.path.join(mask_dir, fn), binarize=True)
        data["gt_binary"].append(gt_bin)

        mc_path = os.path.join(mc_dir, fn)
        gt_mc = load_mask(mc_path, binarize=False) if os.path.exists(mc_path) \
                else np.zeros_like(gt_bin, dtype=np.uint8)
        data["gt_multiclass"].append(gt_mc)

        img_loaded = False
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            base = os.path.splitext(fn)[0]
            ip = os.path.join(img_dir, base + ext)
            if os.path.exists(ip):
                data["images"].append(load_rgb(ip))
                img_loaded = True
                break
        if not img_loaded:
            ip = os.path.join(img_dir, fn)
            if os.path.exists(ip):
                data["images"].append(load_rgb(ip))
            else:
                data["images"].append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    return data


def load_predictions(pred_directory, filenames):
    if not os.path.isdir(pred_directory):
        return None
    preds = []
    for fn in filenames:
        pred_path = os.path.join(pred_directory, fn)
        if os.path.exists(pred_path):
            preds.append(load_mask(pred_path, binarize=True))
        else:
            base = os.path.splitext(fn)[0]
            found = False
            for ext in ['.png', '.jpg', '.jpeg']:
                alt = os.path.join(pred_directory, base + ext)
                if os.path.exists(alt):
                    preds.append(load_mask(alt, binarize=True))
                    found = True
                    break
            if not found:
                preds.append(None)
    return preds


# ============================================================
# BOUNDARY BAND COMPUTATION
# ============================================================

def compute_boundary_band(gt_binary, width=BOUNDARY_WIDTH):
    if gt_binary.sum() == 0 or gt_binary.sum() == gt_binary.size:
        empty = np.zeros_like(gt_binary, dtype=bool)
        return empty, empty, empty

    dist_inside  = distance_transform_edt(gt_binary)
    dist_outside = distance_transform_edt(1 - gt_binary)

    band_inside  = (dist_inside  > 0) & (dist_inside  <= width)
    band_outside = (dist_outside > 0) & (dist_outside <= width)
    band = band_inside | band_outside
    return band, band_inside, band_outside


def compute_eroded_mask(gt_binary, erosion=BOUNDARY_WIDTH):
    if gt_binary.sum() == 0:
        return np.zeros_like(gt_binary, dtype=bool)
    dist_inside = distance_transform_edt(gt_binary)
    return dist_inside > erosion


def compute_instance_boundary_band(instance_mask, width=BOUNDARY_WIDTH):
    if instance_mask.sum() == 0:
        empty = np.zeros_like(instance_mask, dtype=bool)
        return empty, empty, empty
    inst_u8 = instance_mask.astype(np.uint8)
    dist_in  = distance_transform_edt(inst_u8)
    dist_out = distance_transform_edt(1 - inst_u8)
    band_in  = (dist_in  > 0) & (dist_in  <= width)
    band_out = (dist_out > 0) & (dist_out <= width)
    return (band_in | band_out), band_in, band_out


# ============================================================
# INSTANCE EXTRACTION
# ============================================================

def extract_shadow_instances(gt_binary, gt_multiclass, min_area=MIN_INSTANCE_AREA):
    labeled, num_features = ndimage.label(gt_binary)
    instances = []

    for i in range(1, num_features + 1):
        component_mask = (labeled == i)
        area = int(component_mask.sum())
        if area < min_area:
            continue

        rows = np.where(component_mask.any(axis=1))[0]
        cols = np.where(component_mask.any(axis=0))[0]
        bbox     = (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))
        centroid = (float(rows.mean()), float(cols.mean()))

        types_in     = gt_multiclass[component_mask]
        types_nonzero = types_in[types_in > 0]
        dominant_type = int(np.bincount(types_nonzero).argmax()) \
                        if len(types_nonzero) > 0 else 0

        instances.append({
            "label_id": i,
            "mask": component_mask,
            "bbox": bbox,
            "area": area,
            "shadow_type": dominant_type,
            "shadow_type_name": SHADOW_TYPE_MAP.get(dominant_type, "Unknown"),
            "centroid": centroid,
        })
    return instances


# ============================================================
# GEOMETRY COMPUTATION
# ============================================================

def compute_instance_geometry(instance_mask):
    labeled = instance_mask.astype(int)
    props   = regionprops(labeled)
    if not props:
        return None
    p = props[0]
    major = p.major_axis_length
    minor = p.minor_axis_length
    elongation     = major / minor if minor > 0 else float('inf')
    orientation_deg = np.degrees(p.orientation) % 360
    try:
        skel = skeletonize(instance_mask)
        skeleton_length = int(skel.sum())
    except Exception:
        skeleton_length = int(major)
    return {
        "major_axis": float(major),
        "minor_axis": float(minor),
        "elongation": float(elongation),
        "orientation_deg": float(orientation_deg),
        "skeleton_length": skeleton_length,
        "area": int(p.area),
        "perimeter": float(p.perimeter),
    }


# ============================================================
# PER-IMAGE BOUNDARY METRIC COMPUTATION
# (tolerant: ±BOUNDARY_WIDTH don't-care band is excluded)
# ============================================================

def compute_boundary_metrics(pred_binary, gt_binary, band):
    """
    Compute precision / recall / F1 / IoU for a SINGLE IMAGE,
    excluding the ±BOUNDARY_WIDTH don't-care band.

    This is the tolerant metric.  Aggregate across images with
    aggregate_per_image_metrics(), NOT by pooling TP/FP/FN.
    """
    eval_mask = ~band
    if eval_mask.sum() == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "iou": np.nan,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "eval_pixels": 0}

    pred_eval = pred_binary[eval_mask]
    gt_eval   = gt_binary[eval_mask]

    tp = int(((pred_eval == 1) & (gt_eval == 1)).sum())
    fp = int(((pred_eval == 1) & (gt_eval == 0)).sum())
    fn = int(((pred_eval == 0) & (gt_eval == 1)).sum())
    tn = int(((pred_eval == 0) & (gt_eval == 0)).sum())

    precision = tp / (tp + fp)       if (tp + fp)       > 0 else np.nan
    recall    = tp / (tp + fn)       if (tp + fn)       > 0 else np.nan
    f1        = 2*tp / (2*tp+fp+fn)  if (2*tp+fp+fn)   > 0 else np.nan
    iou       = tp / (tp + fp + fn)  if (tp + fp + fn)  > 0 else np.nan

    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "iou": float(iou),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "eval_pixels": int(eval_mask.sum())}


# ============================================================
# PER-IMAGE MEAN AGGREGATION  (PRIMARY AGGREGATION METHOD)
# ============================================================

def compute_per_image_metrics(city_data, preds):
    """
    Compute tolerant boundary metrics for each image independently.

    Returns a list of per-image metric dicts (one per valid image).
    Callers should aggregate with aggregate_per_image_metrics().

    Why per-image mean?  Every major shadow detection benchmark
    (BDRAR, MTMT, DSDNet, SCOTCH) reports per-image averaged metrics,
    not dataset-level pixel pooling.  Pooled metrics are dominated by
    large images / high-shadow-density images and are not comparable
    across datasets with different image sizes.
    """
    per_image = []
    gt_cache  = city_data["gt_cache"]

    for i, (gt_bin, pred) in enumerate(zip(city_data["gt_binary"], preds)):
        if pred is None:
            continue
        band = gt_cache[i]["band"]
        m = compute_boundary_metrics(pred, gt_bin, band)
        if m["eval_pixels"] > 0 and not np.isnan(m["iou"]):
            per_image.append(m)

    return per_image


def aggregate_per_image_metrics(per_image_list):
    """
    Average per-image metrics.  Returns dict with mean ± std for each metric.
    """
    if not per_image_list:
        nan = float('nan')
        return {"iou": nan, "iou_std": nan, "precision": nan,
                "recall": nan, "f1": nan, "n_images": 0}

    def _agg(key):
        vals = [m[key] for m in per_image_list if not np.isnan(m.get(key, float('nan')))]
        return (float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else float('nan'),
                len(vals))

    iou_mean, iou_std, n = _agg("iou")
    prec_mean, _, _      = _agg("precision")
    rec_mean,  _, _      = _agg("recall")
    f1_mean,   _, _      = _agg("f1")

    return {"iou": iou_mean, "iou_std": iou_std,
            "precision": prec_mean, "recall": rec_mean,
            "f1": f1_mean, "n_images": n}


# ============================================================
# INSTANCE-LEVEL METRICS  (used by thread1 / thread3)
# ============================================================

def compute_instance_boundary_metrics(pred_binary, instance_mask,
                                       width=BOUNDARY_WIDTH, neighborhood_margin=10):
    pad  = width + neighborhood_margin + 2
    rows = np.where(instance_mask.any(axis=1))[0]
    cols = np.where(instance_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "iou": np.nan,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "eval_pixels": 0}

    r0 = max(0, rows[0] - pad);  r1 = min(instance_mask.shape[0], rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad);  c1 = min(instance_mask.shape[1], cols[-1] + pad + 1)

    inst_crop = instance_mask[r0:r1, c0:c1]
    pred_crop = pred_binary[r0:r1, c0:c1]
    inst_u8   = inst_crop.astype(np.uint8)

    dist_in  = distance_transform_edt(inst_u8)
    dist_out = distance_transform_edt(1 - inst_u8)

    band = ((dist_in > 0) & (dist_in <= width)) | ((dist_out > 0) & (dist_out <= width))
    neighborhood = inst_crop | (dist_out <= width + neighborhood_margin)
    eval_mask = neighborhood & ~band

    if eval_mask.sum() == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "iou": np.nan,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "eval_pixels": 0}

    pred_e = pred_crop[eval_mask];  gt_e = inst_u8[eval_mask]
    tp = int(((pred_e == 1) & (gt_e == 1)).sum())
    fp = int(((pred_e == 1) & (gt_e == 0)).sum())
    fn = int(((pred_e == 0) & (gt_e == 1)).sum())
    tn = int(((pred_e == 0) & (gt_e == 0)).sum())

    precision = tp / (tp + fp)      if (tp + fp)      > 0 else np.nan
    recall    = tp / (tp + fn)      if (tp + fn)      > 0 else np.nan
    f1        = 2*tp/(2*tp+fp+fn)   if (2*tp+fp+fn)  > 0 else np.nan
    iou       = tp / (tp+fp+fn)     if (tp+fp+fn)    > 0 else np.nan

    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "iou": float(iou),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "eval_pixels": int(eval_mask.sum())}


def compute_instance_eval_mask(instance_mask, width=BOUNDARY_WIDTH, neighborhood_margin=10):
    pad  = width + neighborhood_margin + 2
    rows = np.where(instance_mask.any(axis=1))[0]
    cols = np.where(instance_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros_like(instance_mask, dtype=bool)

    H, W = instance_mask.shape
    r0 = max(0, rows[0] - pad);  r1 = min(H, rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad);  c1 = min(W, cols[-1] + pad + 1)

    inst_crop = instance_mask[r0:r1, c0:c1].astype(np.uint8)
    dist_in  = distance_transform_edt(inst_crop)
    dist_out = distance_transform_edt(1 - inst_crop)

    band      = ((dist_in > 0) & (dist_in <= width)) | ((dist_out > 0) & (dist_out <= width))
    neighborhood = inst_crop.astype(bool) | (dist_out <= width + neighborhood_margin)
    eval_crop = neighborhood & ~band

    eval_full = np.zeros_like(instance_mask, dtype=bool)
    eval_full[r0:r1, c0:c1] = eval_crop
    return eval_full


# ============================================================
# PRE-COMPUTATION CACHE
# ============================================================

def precompute_gt_cache(city_data):
    cache = []

    for i, (gt_bin, gt_mc, rgb_img) in enumerate(
        zip(city_data["gt_binary"], city_data["gt_multiclass"], city_data["images"])
    ):
        band, _, _ = compute_boundary_band(gt_bin)
        gray       = np.mean(rgb_img.astype(float), axis=2)
        instances  = extract_shadow_instances(gt_bin, gt_mc)

        processed_instances = []
        for inst in instances:
            mask = inst["mask"]
            H, W = mask.shape

            eroded = compute_eroded_mask(mask.astype(np.uint8))
            if eroded.sum() < 10:
                continue

            geom = compute_instance_geometry(mask)
            if geom is not None:
                inst.update(geom)

            inst["median_intensity"] = float(np.median(gray[eroded]))

            pad  = BOUNDARY_WIDTH + 10 + 2
            rows_inst = np.where(mask.any(axis=1))[0]
            cols_inst = np.where(mask.any(axis=0))[0]

            if len(rows_inst) == 0 or len(cols_inst) == 0:
                inst["eval_crop"] = None
            else:
                r0 = max(0, rows_inst[0] - pad);  r1 = min(H, rows_inst[-1] + pad + 1)
                c0 = max(0, cols_inst[0] - pad);  c1 = min(W, cols_inst[-1] + pad + 1)

                inst_crop = mask[r0:r1, c0:c1].astype(np.uint8)
                dist_in  = distance_transform_edt(inst_crop)
                dist_out = distance_transform_edt(1 - inst_crop)

                band_crop = (
                    ((dist_in  > 0) & (dist_in  <= BOUNDARY_WIDTH)) |
                    ((dist_out > 0) & (dist_out <= BOUNDARY_WIDTH))
                )
                neighborhood = inst_crop.astype(bool) | (dist_out <= BOUNDARY_WIDTH + 10)
                eval_crop    = neighborhood & ~band_crop

                inst["eval_crop"] = {
                    "r0": r0, "r1": r1, "c0": c0, "c1": c1,
                    "eval_mask": eval_crop,
                    "gt_crop":   inst_crop,
                }

            del inst["mask"]
            processed_instances.append(inst)

        cache.append({"band": band, "instances": processed_instances})

    return cache


def fast_instance_metrics(pred_binary, inst):
    ec = inst.get("eval_crop")
    if ec is None or ec["eval_mask"].sum() == 0:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "iou": np.nan,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "eval_pixels": 0}

    pred_crop = pred_binary[ec["r0"]:ec["r1"], ec["c0"]:ec["c1"]]
    eval_mask = ec["eval_mask"]
    gt_crop   = ec["gt_crop"]

    pred_e = pred_crop[eval_mask];  gt_e = gt_crop[eval_mask]

    tp = int(((pred_e == 1) & (gt_e == 1)).sum())
    fp = int(((pred_e == 1) & (gt_e == 0)).sum())
    fn = int(((pred_e == 0) & (gt_e == 1)).sum())
    tn = int(((pred_e == 0) & (gt_e == 0)).sum())

    precision = tp / (tp + fp)      if (tp + fp)      > 0 else np.nan
    recall    = tp / (tp + fn)      if (tp + fn)      > 0 else np.nan
    f1        = 2*tp/(2*tp+fp+fn)   if (2*tp+fp+fn)  > 0 else np.nan
    iou       = tp / (tp+fp+fn)     if (tp+fp+fn)    > 0 else np.nan

    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "iou": float(iou),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "eval_pixels": int(eval_mask.sum())}


# ============================================================
# AGGREGATION HELPERS
# ============================================================

def safe_mean(values):
    vals = [v for v in values if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def safe_std(values):
    vals = [v for v in values if not np.isnan(v)]
    return float(np.std(vals)) if len(vals) > 1 else np.nan


def bin_values(values, bin_edges):
    indices = np.digitize(values, bin_edges) - 1
    return np.clip(indices, 0, len(bin_edges) - 2)


# ============================================================
# DIRECTION BINNING  (Thread 3b)
# ============================================================

DIRECTION_BINS = {
    0: "N", 1: "NE", 2: "E",  3: "SE",
    4: "S", 5: "SW", 6: "W",  7: "NW",
}


def angle_to_direction_bin(angle_deg):
    angle_deg = angle_deg % 360
    shifted   = (angle_deg + 22.5) % 360
    return int(shifted // 45)