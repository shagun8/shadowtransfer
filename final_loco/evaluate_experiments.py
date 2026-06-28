"""
Evaluate Experiments A, B, C

Runs the Thread 1 diagnostic suite (1a, 1b, 1c) on experiment predictions
and computes recovery ratios against upper-bound and LOCO-vanilla baselines.

Recovery ratio: R = (experiment_IoU - LOCO_IoU) / (upper_IoU - LOCO_IoU)
  R ≈ 0  → experiment did nothing
  R ≈ 1  → experiment fully closes the transfer gap
  R > 1  → experiment overcorrects

For each experiment, evaluates:
  - Global metrics (IoU, precision, recall, F1) outside ±5px band
  - 1a: FP composition clustering
  - 1b: Intensity-conditioned performance curves
  - 1c: Per-class recall drop
  - Recovery ratios per metric, per intensity bin, per class

Usage:
    python evaluate_experiments.py --experiments a b c
    python evaluate_experiments.py --experiments a --models mamnet oglanet dinov3
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict

# Add parent directory for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    CITIES, RESOLUTIONS, MODELS,
    SHADOW_TYPE_MAP, SHADOW_TYPE_SHORT,
    GT_BASE, PRED_BASE, OUTPUT_BASE, IMG_SIZE,
    upper_pred_dir, loco_pred_dir, output_dir,
)
from utils import (
    load_city_data, load_predictions, precompute_gt_cache,
    fast_instance_metrics, safe_mean, safe_std,
    BOUNDARY_WIDTH,
)
from thread1_entanglement import (
    run_1a_fp_clustering,
    run_1b_intensity_curves,
    run_1c_per_class_metrics,
)


# ============================================================
# EXPERIMENT PREDICTION DIRECTORIES
# ============================================================

EXP_PRED_BASE = PRED_BASE

EXPERIMENT_DIRS = {
    'a': os.path.join(EXP_PRED_BASE, 'experiment_a'),
    'b': os.path.join(EXP_PRED_BASE, 'experiment_b'),
    'b2': os.path.join(EXP_PRED_BASE, 'experiment_b2'),
    'c': os.path.join(EXP_PRED_BASE, 'experiment_c'),
    # Layer-wise BN swap variants
    'b_lw_early': os.path.join(EXP_PRED_BASE, 'experiment_b_lw_early'),
    'b_lw_mid': os.path.join(EXP_PRED_BASE, 'experiment_b_lw_mid'),
    'b_lw_late': os.path.join(EXP_PRED_BASE, 'experiment_b_lw_late'),
    # Data efficiency variants
    'a_de5pct': os.path.join(EXP_PRED_BASE, 'experiment_a_de5pct'),
    'a_de10pct': os.path.join(EXP_PRED_BASE, 'experiment_a_de10pct'),
    'a_de15pct': os.path.join(EXP_PRED_BASE, 'experiment_a_de15pct'),
    'a_de20pct': os.path.join(EXP_PRED_BASE, 'experiment_a_de20pct'),
}


def exp_pred_dir(experiment, city, res, model, variant='vanilla'):
    """Path to experiment prediction masks."""
    base = EXPERIMENT_DIRS.get(experiment, '')
    return os.path.join(base, city, res, model, variant)


# ============================================================
# GLOBAL METRICS (outside ±5px band)
# ============================================================

def compute_global_metrics(city_data, preds, label):
    """
    Compute aggregate IoU, precision, recall, F1 over all images,
    using only pixels outside the ±5px don't-care band.
    """
    total_tp, total_fp, total_fn = 0, 0, 0
    gt_cache = city_data["gt_cache"]

    for i, pred in enumerate(preds):
        if pred is None:
            continue

        gt_bin = city_data["gt_binary"][i]
        band = gt_cache[i]["band"]
        eval_mask = ~band

        if eval_mask.sum() == 0:
            continue

        pred_eval = pred[eval_mask]
        gt_eval = gt_bin[eval_mask]

        total_tp += int(((pred_eval == 1) & (gt_eval == 1)).sum())
        total_fp += int(((pred_eval == 1) & (gt_eval == 0)).sum())
        total_fn += int(((pred_eval == 0) & (gt_eval == 1)).sum())

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else float('nan')
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else float('nan')
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else float('nan')
    f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else float('nan')

    return {
        "label": label,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
    }


def compute_per_image_iou(city_data, preds, label):
    '''
    Compute per-image IoU outside ±5px don't-care band.
    Returns list of float IoU values (one per image, NaN if no evaluable pixels).
    '''
    gt_cache = city_data["gt_cache"]
    per_image = []
 
    for i, pred in enumerate(preds):
        if pred is None:
            per_image.append(float('nan'))
            continue
 
        gt_bin = city_data["gt_binary"][i]
        band = gt_cache[i]["band"]
        eval_mask = ~band
 
        if eval_mask.sum() == 0:
            per_image.append(float('nan'))
            continue
 
        pred_eval = pred[eval_mask]
        gt_eval = gt_bin[eval_mask]
 
        tp = int(((pred_eval == 1) & (gt_eval == 1)).sum())
        fp = int(((pred_eval == 1) & (gt_eval == 0)).sum())
        fn = int(((pred_eval == 0) & (gt_eval == 1)).sum())
 
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float('nan')
        per_image.append(float(iou))
 
    return per_image


# ============================================================
# RECOVERY RATIO COMPUTATION
# ============================================================

def compute_recovery(exp_val, loco_val, upper_val):
    """
    Compute recovery ratio: R = (exp - loco) / (upper - loco).
    Returns NaN if denominator is zero or any input is NaN.
    """
    if any(np.isnan(v) for v in [exp_val, loco_val, upper_val]):
        return float('nan')
    gap = upper_val - loco_val
    if abs(gap) < 1e-8:
        return float('nan')
    return float((exp_val - loco_val) / gap)


def compute_recovery_dict(exp_metrics, loco_metrics, upper_metrics, metric_keys=None):
    """Compute recovery ratio for all metric keys."""
    if metric_keys is None:
        metric_keys = ['iou', 'precision', 'recall', 'f1']
    recovery = {}
    for k in metric_keys:
        recovery[k] = compute_recovery(
            exp_metrics.get(k, float('nan')),
            loco_metrics.get(k, float('nan')),
            upper_metrics.get(k, float('nan')),
        )
    return recovery


# ============================================================
# INTENSITY-CONDITIONED RECOVERY
# ============================================================

def compute_1b_recovery(exp_1b, loco_1b, upper_1b):
    """
    Compute recovery ratio per intensity bin from 1b results.
    Returns list of per-bin recovery dicts.
    """
    if not exp_1b.get('bins') or not loco_1b.get('bins') or not upper_1b.get('bins'):
        return []

    # Match bins by center (they should be the same if computed on same GT)
    # Use the experiment's bins as reference
    recovery_bins = []
    for exp_bin in exp_1b['bins']:
        center = exp_bin.get('bin_center')
        if center is None or exp_bin.get('iou_mean') is None:
            continue

        # Find matching bins in loco and upper
        loco_bin = _find_closest_bin(loco_1b['bins'], center)
        upper_bin = _find_closest_bin(upper_1b['bins'], center)

        if loco_bin is None or upper_bin is None:
            continue

        r_iou = compute_recovery(
            exp_bin.get('iou_mean', float('nan')),
            loco_bin.get('iou_mean', float('nan')),
            upper_bin.get('iou_mean', float('nan')),
        )
        r_recall = compute_recovery(
            exp_bin.get('recall_mean', float('nan')),
            loco_bin.get('recall_mean', float('nan')),
            upper_bin.get('recall_mean', float('nan')),
        )

        recovery_bins.append({
            'bin_center': center,
            'bin_low': exp_bin.get('bin_low'),
            'bin_high': exp_bin.get('bin_high'),
            'recovery_iou': r_iou,
            'recovery_recall': r_recall,
            'exp_iou': exp_bin.get('iou_mean'),
            'loco_iou': loco_bin.get('iou_mean'),
            'upper_iou': upper_bin.get('iou_mean'),
            'count': exp_bin.get('count', 0),
        })

    return recovery_bins


def _find_closest_bin(bins, target_center, tol=20):
    """Find bin with closest center to target."""
    best = None
    best_dist = float('inf')
    for b in bins:
        c = b.get('bin_center')
        if c is None or b.get('iou_mean') is None:
            continue
        dist = abs(c - target_center)
        if dist < best_dist and dist < tol:
            best = b
            best_dist = dist
    return best


# ============================================================
# PER-CLASS RECOVERY
# ============================================================

def compute_1c_recovery(exp_1c, loco_1c, upper_1c):
    """Compute recovery ratio per shadow class from 1c results."""
    recovery = {}

    all_types = set()
    for r in [exp_1c, loco_1c, upper_1c]:
        if r and 'per_class' in r:
            all_types.update(r['per_class'].keys())

    for stype in sorted(all_types):
        exp_cls = exp_1c.get('per_class', {}).get(stype, {})
        loco_cls = loco_1c.get('per_class', {}).get(stype, {})
        upper_cls = upper_1c.get('per_class', {}).get(stype, {})

        r_recall = compute_recovery(
            exp_cls.get('recall', float('nan')),
            loco_cls.get('recall', float('nan')),
            upper_cls.get('recall', float('nan')),
        )
        r_iou = compute_recovery(
            exp_cls.get('iou', float('nan')),
            loco_cls.get('iou', float('nan')),
            upper_cls.get('iou', float('nan')),
        )

        recovery[stype] = {
            'shadow_type_name': SHADOW_TYPE_MAP.get(int(stype), "Unknown"),
            'recovery_recall': r_recall,
            'recovery_iou': r_iou,
            'exp_recall': exp_cls.get('recall'),
            'loco_recall': loco_cls.get('recall'),
            'upper_recall': upper_cls.get('recall'),
            'instance_count': exp_cls.get('instance_count', 0),
        }

    return recovery


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def evaluate_single_experiment(experiment_name, city_data_cache, args):
    """
    Evaluate one experiment across all (model × holdout_city × res) combos.
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING EXPERIMENT {experiment_name.upper()}")
    print(f"{'='*70}")

    results = {}
    models_to_eval = args.models if args.models else MODELS
    resolutions = args.resolutions if args.resolutions else ['highres']

    for res in resolutions:
        for model in models_to_eval:
            for holdout_city in CITIES:
                data = city_data_cache.get((holdout_city, res))
                if data is None:
                    continue

                key = f"{experiment_name}_{model}_{holdout_city}_{res}"

                # Load predictions
                exp_dir = exp_pred_dir(experiment_name, holdout_city, res, model)
                exp_preds = load_predictions(exp_dir, data["filenames"])

                upper_dir = upper_pred_dir(holdout_city, res, model)
                upper_preds = load_predictions(upper_dir, data["filenames"])

                loco_dir = loco_pred_dir(holdout_city, res, model, 'vanilla')
                loco_preds = load_predictions(loco_dir, data["filenames"])

                if exp_preds is None:
                    print(f"  SKIP {key}: no experiment predictions at {exp_dir}")
                    continue
                if upper_preds is None:
                    print(f"  SKIP {key}: no upper predictions")
                    continue
                if loco_preds is None:
                    print(f"  SKIP {key}: no LOCO predictions")
                    continue

                print(f"\n  --- {key} ---")

                # A1/B1/C1: Global metrics
                exp_global = compute_global_metrics(data, exp_preds, f"exp_{key}")
                upper_global = compute_global_metrics(data, upper_preds, f"upper_{key}")
                loco_global = compute_global_metrics(data, loco_preds, f"loco_{key}")

                global_recovery = compute_recovery_dict(exp_global, loco_global, upper_global)
                print(f"    Global: IoU exp={exp_global['iou']:.3f} "
                      f"loco={loco_global['iou']:.3f} upper={upper_global['iou']:.3f} "
                      f"R={global_recovery['iou']:.3f}")
                
                exp_per_image_iou = compute_per_image_iou(data, exp_preds, f"exp_{key}")
                upper_per_image_iou = compute_per_image_iou(data, upper_preds, f"upper_{key}")
                loco_per_image_iou = compute_per_image_iou(data, loco_preds, f"loco_{key}")
 
                n_valid = sum(1 for v in exp_per_image_iou if not np.isnan(v))
                print(f"    Per-image IoU: {n_valid}/{len(exp_per_image_iou)} valid images")

                # A2/B2/C2: 1b Intensity-conditioned
                exp_1b = run_1b_intensity_curves(data, exp_preds, f"exp_{key}")
                upper_1b = run_1b_intensity_curves(data, upper_preds, f"upper_{key}")
                loco_1b = run_1b_intensity_curves(data, loco_preds, f"loco_{key}")
                intensity_recovery = compute_1b_recovery(exp_1b, loco_1b, upper_1b)

                if intensity_recovery:
                    low_r = [b['recovery_iou'] for b in intensity_recovery[:3]
                             if not np.isnan(b['recovery_iou'])]
                    high_r = [b['recovery_iou'] for b in intensity_recovery[-3:]
                              if not np.isnan(b['recovery_iou'])]
                    print(f"    Intensity recovery: low bins R≈{safe_mean(low_r):.3f}, "
                          f"high bins R≈{safe_mean(high_r):.3f}")

                # A3/B3/C3: 1c Per-class
                exp_1c = run_1c_per_class_metrics(data, exp_preds, f"exp_{key}")
                upper_1c = run_1c_per_class_metrics(data, upper_preds, f"upper_{key}")
                loco_1c = run_1c_per_class_metrics(data, loco_preds, f"loco_{key}")
                class_recovery = compute_1c_recovery(exp_1c, loco_1c, upper_1c)

                for stype, cr in class_recovery.items():
                    if cr.get('recovery_recall') is not None and not np.isnan(cr['recovery_recall']):
                        print(f"    Class {cr['shadow_type_name']}: "
                              f"R_recall={cr['recovery_recall']:.3f} "
                              f"(n={cr['instance_count']})")

                # A4/B4/C4: 1a FP composition
                exp_1a = run_1a_fp_clustering(data, exp_preds, f"exp_{key}")
                upper_1a = run_1a_fp_clustering(data, upper_preds, f"upper_{key}")
                loco_1a = run_1a_fp_clustering(data, loco_preds, f"loco_{key}")

                fp_change = "N/A"
                if (exp_1a.get('total_fp_pixels', 0) > 0 and
                    loco_1a.get('total_fp_pixels', 0) > 0):
                    fp_ratio = exp_1a['total_fp_pixels'] / loco_1a['total_fp_pixels']
                    fp_change = f"{fp_ratio:.2f}x"
                print(f"    FP count: exp={exp_1a.get('total_fp_pixels', 0)} "
                      f"loco={loco_1a.get('total_fp_pixels', 0)} ({fp_change})")

                results[key] = {
                    'experiment': experiment_name,
                    'model': model,
                    'holdout_city': holdout_city,
                    'res': res,
                    'global': {
                        'experiment': exp_global,
                        'upper': upper_global,
                        'loco': loco_global,
                        'recovery': global_recovery,
                        'per_image_iou': {
                            'experiment': exp_per_image_iou,
                            'upper': upper_per_image_iou,
                            'loco': loco_per_image_iou,
                        },
                    },
                    'intensity_conditioned': {
                        'experiment_1b': exp_1b,
                        'upper_1b': upper_1b,
                        'loco_1b': loco_1b,
                        'recovery_bins': intensity_recovery,
                    },
                    'per_class': {
                        'experiment_1c': exp_1c,
                        'upper_1c': upper_1c,
                        'loco_1c': loco_1c,
                        'recovery': class_recovery,
                    },
                    'fp_composition': {
                        'experiment_1a': exp_1a,
                        'upper_1a': upper_1a,
                        'loco_1a': loco_1a,
                    },
                }

    return results


def build_summary_table(all_results):
    """
    Build the 3×3 (models × experiments) recovery ratio matrix.
    """
    summary = {}

    for key, r in all_results.items():
        exp = r['experiment']
        model = r['model']
        city = r['holdout_city']

        table_key = f"{model}_{city}"
        if table_key not in summary:
            summary[table_key] = {'model': model, 'city': city}

        summary[table_key][f'{exp}_R_iou'] = r['global']['recovery'].get('iou', float('nan'))
        summary[table_key][f'{exp}_R_recall'] = r['global']['recovery'].get('recall', float('nan'))
        summary[table_key][f'{exp}_exp_iou'] = r['global']['experiment'].get('iou', float('nan'))
        summary[table_key][f'{exp}_loco_iou'] = r['global']['loco'].get('iou', float('nan'))
        summary[table_key][f'{exp}_upper_iou'] = r['global']['upper'].get('iou', float('nan'))

    return summary


def print_summary_table(summary):
    """Print formatted summary table."""
    print("\n" + "=" * 90)
    print("RECOVERY RATIO SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Model':<10} {'City':<10} ", end="")
    for exp in ['a', 'b', 'c']:
        print(f"{'Exp'+exp.upper()+' R_IoU':<14} ", end="")
    print(f"{'LOCO IoU':<10} {'Upper IoU':<10}")
    print("-" * 90)

    for key in sorted(summary.keys()):
        s = summary[key]
        print(f"{s['model']:<10} {s['city']:<10} ", end="")
        for exp in ['a', 'b', 'c']:
            r = s.get(f'{exp}_R_iou', float('nan'))
            if np.isnan(r):
                print(f"{'N/A':<14} ", end="")
            else:
                print(f"{r:<14.3f} ", end="")
        loco = s.get('a_loco_iou', s.get('b_loco_iou', s.get('c_loco_iou', float('nan'))))
        upper = s.get('a_upper_iou', s.get('b_upper_iou', s.get('c_upper_iou', float('nan'))))
        print(f"{loco:<10.3f} {upper:<10.3f}" if not np.isnan(loco) else "")

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser(description="Evaluate Experiments A, B, C")
    p.add_argument('--experiments', nargs='+', default=['a', 'b', 'c'],
                   help='Which experiments to evaluate')
    p.add_argument('--models', nargs='+', default=None,
                   help='Models to evaluate (default: all)')
    p.add_argument('--resolutions', nargs='+', default=['highres'],
                   choices=['highres', 'midres'])
    p.add_argument('--cities', nargs='+', default=None,
                   help='Holdout cities to evaluate (default: all)')
    p.add_argument('--output_dir', default=os.path.join(OUTPUT_BASE, 'experiment_evaluation'))
    args = p.parse_args()

    # If specific cities requested, override
    if args.cities:
        global CITIES
        CITIES = args.cities

    # Load GT data
    print("Loading GT data and pre-computing caches...")
    city_data_cache = {}
    for res in args.resolutions:
        for city in CITIES:
            data = load_city_data(city, res)
            if data is not None:
                data["gt_cache"] = precompute_gt_cache(data)
                city_data_cache[(city, res)] = data
                n_inst = sum(len(c["instances"]) for c in data["gt_cache"])
                print(f"  {city}/{res}: {len(data['filenames'])} images, {n_inst} instances")

    # Run evaluation for each experiment
    all_results = {}
    for exp_name in args.experiments:
        # Also check layer-wise BN variants
        exp_variants = [exp_name]
        if exp_name == 'b':
            for lw in ['b_lw_early', 'b_lw_mid', 'b_lw_late']:
                if os.path.isdir(EXPERIMENT_DIRS.get(lw, '')):
                    exp_variants.append(lw)
        if exp_name == 'a':
            for de in ['a_de5pct', 'a_de10pct', 'a_de15pct', 'a_de20pct']:
                if os.path.isdir(EXPERIMENT_DIRS.get(de, '')):
                    exp_variants.append(de)

        for ev in exp_variants:
            results = evaluate_single_experiment(ev, city_data_cache, args)
            all_results.update(results)

    # Build and print summary
    summary = build_summary_table(all_results)
    print_summary_table(summary)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert non-serializable values
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(os.path.join(args.output_dir, 'experiment_results.json'), 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)

    with open(os.path.join(args.output_dir, 'summary_table.json'), 'w') as f:
        json.dump(make_serializable(summary), f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()