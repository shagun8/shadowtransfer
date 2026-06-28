"""
Statistical analysis of shadow detection inference results.
Performs bootstrap hypothesis testing to compare LOCO and cross-resolution
methods against upper bounds.

Boundary-tolerant evaluation uses a ±K pixel don't-care zone around GT shadow
boundaries. Pixels in this band are excluded from evaluation entirely, since GT
labeling is inherently imprecise at boundaries.

LOCO methods (updated):
    removed : mcl
    added   : iim, isw, mrfp_plus, fada
    retained: vanilla, fda, segdesic

Usage:
    python statistical_analysis.py \
        --inference_dir ./Test_img_results \
        --data_root /path/to/Final_data_test \
        --output_csv ./statistical_results.csv \
        --n_bootstrap 10000
"""

import os
import sys
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), 'mamnet'))


def get_args():
    parser = argparse.ArgumentParser(
        description='Statistical analysis of inference results')

    parser.add_argument('--inference_dir', type=str,
                        default='./Test_img_results',
                        help='Directory with inference results')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory with ground truth data')
    parser.add_argument('--output_csv', type=str,
                        default='./statistical_results.csv',
                        help='Output CSV file with all statistical results')
    parser.add_argument('--n_bootstrap', type=int, default=10000,
                        help='Number of bootstrap samples')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level (default: 0.05 for 95% CI)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_prediction(pred_path):
    """Load prediction mask (0 or 255) → binary {0, 1}."""
    pred = np.array(Image.open(pred_path).convert('L'))
    return (pred > 127).astype(np.uint8)


def load_ground_truth(gt_path):
    """Load GT mask (0 or 255) → binary {0, 1}."""
    gt = np.array(Image.open(gt_path).convert('L'))
    return (gt > 127).astype(np.uint8)


# ---------------------------------------------------------------------------
# Morphological kernel cache
# ---------------------------------------------------------------------------

_TOLERANCE_KERNEL_CACHE = {}


def _get_tolerance_kernel(tolerance):
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (tolerance * 2 + 1, tolerance * 2 + 1))
    return _TOLERANCE_KERNEL_CACHE[tolerance]


# ---------------------------------------------------------------------------
# Per-image metric computation
# ---------------------------------------------------------------------------

def compute_metrics(pred, gt, tolerance=None):
    """
    Compute segmentation metrics for a single image.

    tolerance=None : strict evaluation — all pixels included
    tolerance=K    : ±K px band around GT boundaries excluded
    """
    if tolerance is not None:
        kernel   = _get_tolerance_kernel(tolerance)
        gt_u8    = gt.astype(np.uint8)
        eroded   = cv2.erode(gt_u8, kernel)
        dilated  = cv2.dilate(gt_u8, kernel)
        band     = (dilated - eroded) > 0
        valid    = ~band
        pred_eval = pred[valid]
        gt_eval   = gt[valid]
    else:
        pred_eval = pred
        gt_eval   = gt

    tp = np.logical_and(pred_eval == 1, gt_eval == 1).sum()
    fp = np.logical_and(pred_eval == 1, gt_eval == 0).sum()
    tn = np.logical_and(pred_eval == 0, gt_eval == 0).sum()
    fn = np.logical_and(pred_eval == 0, gt_eval == 1).sum()

    precision   = tp / (tp + fp + 1e-10)
    recall      = tp / (tp + fn + 1e-10)
    f1          = 2 * precision * recall / (precision + recall + 1e-10)
    shadow_iou  = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou        = (shadow_iou + nonshadow_iou) / 2
    total       = tp + tn + fp + fn
    oa          = (tp + tn) / (total + 1e-10)
    shadow_err  = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber         = (shadow_err + nonshadow_err) / 2

    return {
        'OA':        float(oa        * 100),
        'Precision': float(precision * 100),
        'Recall':    float(recall    * 100),
        'F1':        float(f1        * 100),
        'BER':       float(ber       * 100),
        'mIOU':      float(miou      * 100),
        'Shadow_IOU':float(shadow_iou* 100),
    }


# ---------------------------------------------------------------------------
# Load per-image metrics for a model directory
# ---------------------------------------------------------------------------

def load_all_predictions(pred_dir, gt_dir, tolerance=None):
    """
    Compute per-image metrics for every PNG in pred_dir.
    Returns a list of metric dicts, or None if the directory is absent/empty.
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir) / 'test' / 'masks'

    if not pred_dir.exists():
        return None

    pred_files = sorted(pred_dir.glob('*.png'))
    if not pred_files:
        return None

    metrics_list = []
    for pred_file in pred_files:
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            continue
        pred    = load_prediction(pred_file)
        gt      = load_ground_truth(gt_file)
        metrics = compute_metrics(pred, gt, tolerance=tolerance)
        metrics_list.append(metrics)

    return metrics_list if metrics_list else None


# ---------------------------------------------------------------------------
# Bootstrap comparison
# ---------------------------------------------------------------------------

def bootstrap_comparison(baseline_metrics, method_metrics,
                          n_bootstrap=10000, random_seed=42):
    """
    Paired bootstrap test: method vs baseline.

    Returns a dict keyed by metric name, each value containing:
        baseline_mean, method_mean, delta_mean,
        ci_lower, ci_upper, p_value, significant
    """
    np.random.seed(random_seed)
    n_samples = len(baseline_metrics)
    results   = {}

    for metric_name in baseline_metrics[0].keys():
        baseline_values = np.array(
            [m[metric_name] for m in baseline_metrics])
        method_values   = np.array(
            [m[metric_name] for m in method_metrics])

        observed_diff      = method_values - baseline_values
        observed_mean_diff = np.mean(observed_diff)

        # Non-parametric bootstrap resampling
        bootstrap_diffs = np.array([
            np.mean(observed_diff[
                np.random.choice(n_samples, size=n_samples, replace=True)])
            for _ in range(n_bootstrap)
        ])

        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)

        # Two-tailed p-value
        if observed_mean_diff >= 0:
            prop_wrong = np.mean(bootstrap_diffs <= 0)
        else:
            prop_wrong = np.mean(bootstrap_diffs >= 0)
        p_value = min(2 * prop_wrong, 1.0)

        results[metric_name] = {
            'baseline_mean': float(np.mean(baseline_values)),
            'method_mean':   float(np.mean(method_values)),
            'delta_mean':    float(observed_mean_diff),
            'ci_lower':      float(ci_lower),
            'ci_upper':      float(ci_upper),
            'p_value':       float(p_value),
            'significant':   p_value < 0.05,
        }

    return results


# ---------------------------------------------------------------------------
# LOCO scenario analysis
# ---------------------------------------------------------------------------

def analyze_loco_scenario(inference_dir, data_root, args):
    """
    For every (model, city, resolution, method) LOCO combination, run a
    paired bootstrap comparison against the within-city upper bound.

    LOCO methods:
        vanilla  — base model, no augmentation
        fda      — trained with FDA (same base arch as vanilla)
        segdesic — geographic domain adaptation module
        iim      — Illumination-Invariant Module
        isw      — Instance Selective Whitening (training-only; base arch)
        mrfp_plus— Multi-Resolution Feature Perturbation+
        fada     — Frequency-Adapted Domain Adaptation
    """
    inference_dir = Path(inference_dir)
    data_root     = Path(data_root)

    results = []

    cities      = ['chicago', 'miami', 'phoenix']
    resolutions = ['highres', 'midres']
    models      = ['mamnet', 'oglanet', 'dinov3']
    methods     = ['vanilla', 'fda', 'segdesic', 'iim', 'isw',
                   'mrfp_plus', 'fada']

    print("\n" + "=" * 80)
    print("ANALYZING LOCO SCENARIOS")
    print("=" * 80)
    print(f"Methods: {methods}")

    for model_type in models:
        for city in cities:
            for res in resolutions:
                print(f"\n{model_type.upper()} | {city} | {res}")
                print("-" * 60)

                # Upper bound: model trained on this city at this resolution
                upper_dir = (inference_dir / 'upper' / city / res
                             / model_type / 'base')
                gt_dir    = data_root / city / res

                upper_strict   = load_all_predictions(
                    upper_dir, gt_dir, tolerance=None)
                upper_tolerant = load_all_predictions(
                    upper_dir, gt_dir, tolerance=5)

                if upper_strict is None:
                    print(f"  ⚠ Upper bound not found — skipping")
                    continue
                print(f"  ✓ Upper bound loaded ({len(upper_strict)} images)")

                loco_dir = (inference_dir / 'loco' / city / res / model_type)

                for method in methods:
                    method_dir = loco_dir / method

                    method_strict   = load_all_predictions(
                        method_dir, gt_dir, tolerance=None)
                    method_tolerant = load_all_predictions(
                        method_dir, gt_dir, tolerance=5)

                    if method_strict is None:
                        print(f"    ⚠ {method}: not found")
                        continue

                    print(f"    • {method}: running bootstrap "
                          f"({args.n_bootstrap} samples)…")

                    bs_strict   = bootstrap_comparison(
                        upper_strict, method_strict,
                        n_bootstrap=args.n_bootstrap,
                        random_seed=args.random_seed)
                    bs_tolerant = bootstrap_comparison(
                        upper_tolerant, method_tolerant,
                        n_bootstrap=args.n_bootstrap,
                        random_seed=args.random_seed)

                    # Pack into CSV rows (one row per metric × evaluation type)
                    for metric_name in bs_strict.keys():
                        base_row = {
                            'test_type':  'loco',
                            'city':       city,
                            'resolution': res,
                            'model':      model_type,
                            'method':     method,
                            'metric':     metric_name,
                        }
                        for eval_type, bs in [('strict',   bs_strict),
                                              ('tolerant', bs_tolerant)]:
                            results.append({
                                **base_row,
                                'evaluation':      eval_type,
                                'upper_bound_mean':bs[metric_name]['baseline_mean'],
                                'method_mean':     bs[metric_name]['method_mean'],
                                'delta_mean':      bs[metric_name]['delta_mean'],
                                'ci_lower':        bs[metric_name]['ci_lower'],
                                'ci_upper':        bs[metric_name]['ci_upper'],
                                'p_value':         bs[metric_name]['p_value'],
                                'significant':     bs[metric_name]['significant'],
                            })

                    # Console summary for F1
                    f1 = bs_strict['F1']
                    sig = '***' if f1['significant'] else ''
                    print(f"      F1: {f1['method_mean']:.2f}%  "
                          f"(Δ={f1['delta_mean']:+.2f}, "
                          f"CI=[{f1['ci_lower']:.2f}, {f1['ci_upper']:.2f}], "
                          f"p={f1['p_value']:.4f}{sig})")

    return results


# ---------------------------------------------------------------------------
# Cross-resolution scenario analysis
# ---------------------------------------------------------------------------

def analyze_crossres_scenario(inference_dir, data_root, args):
    """
    For every cross-resolution combination, compare each cross-res method
    against the within-city upper bound at the *test* resolution.

    Cross-res methods are not updated here — only LOCO changed.
    """
    inference_dir = Path(inference_dir)
    data_root     = Path(data_root)

    results = []

    cities     = ['chicago', 'miami', 'phoenix']
    models     = ['mamnet', 'oglanet', 'dinov3']
    directions = [('midres', 'highres'), ('highres', 'midres')]

    print("\n" + "=" * 80)
    print("ANALYZING CROSS-RESOLUTION SCENARIOS")
    print("=" * 80)

    for model_type in models:
        for city in cities:
            for train_res, test_res in directions:
                print(f"\n{model_type.upper()} | {city} "
                      f"| {train_res}→{test_res}")
                print("-" * 60)

                # Upper bound: model trained on *test* resolution
                upper_dir = (inference_dir / 'upper' / city / test_res
                             / model_type / 'base')
                gt_dir    = data_root / city / test_res

                upper_strict   = load_all_predictions(
                    upper_dir, gt_dir, tolerance=None)
                upper_tolerant = load_all_predictions(
                    upper_dir, gt_dir, tolerance=5)

                if upper_strict is None:
                    print(f"  ⚠ Upper bound not found — skipping")
                    continue
                print(f"  ✓ Upper bound loaded ({len(upper_strict)} images)")

                # Which cross-res methods to test per model
                if model_type == 'oglanet':
                    crossres_methods = ['base']
                elif model_type == 'dinov3':
                    if train_res == 'highres' and test_res == 'midres':
                        crossres_methods = ['base', 'hrda', 'gsdpe']
                    else:
                        crossres_methods = ['base', 'gsdpe']
                else:  # mamnet
                    crossres_methods = ['base', 'hrda', 'gsdpe']

                res_str     = f"{train_res}_to_{test_res}"
                crossres_dir = (inference_dir / 'cross-res' / city
                                / res_str / model_type)

                for method in crossres_methods:
                    method_dir = crossres_dir / method

                    method_strict   = load_all_predictions(
                        method_dir, gt_dir, tolerance=None)
                    method_tolerant = load_all_predictions(
                        method_dir, gt_dir, tolerance=5)

                    if method_strict is None:
                        print(f"    ⚠ {method}: not found")
                        continue

                    print(f"    • {method}: running bootstrap "
                          f"({args.n_bootstrap} samples)…")

                    bs_strict   = bootstrap_comparison(
                        upper_strict, method_strict,
                        n_bootstrap=args.n_bootstrap,
                        random_seed=args.random_seed)
                    bs_tolerant = bootstrap_comparison(
                        upper_tolerant, method_tolerant,
                        n_bootstrap=args.n_bootstrap,
                        random_seed=args.random_seed)

                    for metric_name in bs_strict.keys():
                        base_row = {
                            'test_type':  'cross-res',
                            'city':       city,
                            'train_res':  train_res,
                            'test_res':   test_res,
                            'resolution': res_str,
                            'model':      model_type,
                            'method':     method,
                            'metric':     metric_name,
                        }
                        for eval_type, bs in [('strict',   bs_strict),
                                              ('tolerant', bs_tolerant)]:
                            results.append({
                                **base_row,
                                'evaluation':      eval_type,
                                'upper_bound_mean':bs[metric_name]['baseline_mean'],
                                'method_mean':     bs[metric_name]['method_mean'],
                                'delta_mean':      bs[metric_name]['delta_mean'],
                                'ci_lower':        bs[metric_name]['ci_lower'],
                                'ci_upper':        bs[metric_name]['ci_upper'],
                                'p_value':         bs[metric_name]['p_value'],
                                'significant':     bs[metric_name]['significant'],
                            })

                    f1  = bs_strict['F1']
                    sig = '***' if f1['significant'] else ''
                    print(f"      F1: {f1['method_mean']:.2f}%  "
                          f"(Δ={f1['delta_mean']:+.2f}, "
                          f"CI=[{f1['ci_lower']:.2f}, {f1['ci_upper']:.2f}], "
                          f"p={f1['p_value']:.4f}{sig})")

    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_to_csv(results, output_path):
    if not results:
        print("No results to save!")
        return

    # Collect all field names across all rows (cross-res adds train_res/test_res)
    fieldnames = sorted({k for row in results for k in row.keys()})

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction='ignore')
        writer.writeheader()
        # Fill missing fields with empty string so DictWriter doesn't raise
        for row in results:
            padded = {fn: row.get(fn, '') for fn in fieldnames}
            writer.writerow(padded)

    print(f"\n{'=' * 80}")
    print(f"✓ Results saved to: {output_path}")
    print(f"  Total rows: {len(results)}")
    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()

    print("=" * 80)
    print("SHADOW DETECTION — STATISTICAL ANALYSIS")
    print("=" * 80)
    print(f"Bootstrap samples : {args.n_bootstrap}")
    print(f"Significance level: {args.alpha}")
    print(f"Random seed       : {args.random_seed}")

    all_results = []

    loco_results     = analyze_loco_scenario(
        args.inference_dir, args.data_root, args)
    crossres_results = analyze_crossres_scenario(
        args.inference_dir, args.data_root, args)

    all_results.extend(loco_results)
    all_results.extend(crossres_results)

    save_to_csv(all_results, args.output_csv)

    # Summary counts
    n_loco     = len(loco_results)
    n_crossres = len(crossres_results)
    n_total    = len(all_results)
    n_sig      = sum(1 for r in all_results if r['significant'])

    print("\nSUMMARY:")
    print(f"  LOCO comparisons     : {n_loco}")
    print(f"  Cross-res comparisons: {n_crossres}")
    print(f"  Total comparisons    : {n_total}")
    if n_total > 0:
        print(f"  Significant (p<0.05) : {n_sig}/{n_total} "
              f"({100 * n_sig / n_total:.1f}%)")


if __name__ == '__main__':
    main()