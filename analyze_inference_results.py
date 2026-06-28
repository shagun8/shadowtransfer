"""
Analyze inference results with boundary-tolerant evaluation.
Computes metrics both strictly and with ±5px boundary don't-care zone.

The don't-care zone excludes pixels within ±K px of GT shadow boundaries
from evaluation entirely, since GT labeling is inherently imprecise at
boundaries.

Usage:
    python analyze_inference_results.py \
        --inference_dir ./Test_img_results \
        --data_root /path/to/Final_data_test \
        --output_dir ./analysis_results
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from collections import defaultdict

# Add utility path
sys.path.append(os.path.join(os.path.dirname(__file__), 'mamnet'))
from mamnet.utils.evaluation_detailed import DetailedEvaluator


def get_args():
    parser = argparse.ArgumentParser(description='Analyze inference results')
    
    parser.add_argument('--inference_dir', type=str, default='./Test_img_results',
                       help='Directory with inference results')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory with ground truth data')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Output directory for analysis results')
    
    return parser.parse_args()


def load_prediction(pred_path):
    """Load prediction mask (0 or 255) → convert to (0 or 1)"""
    pred = np.array(Image.open(pred_path).convert('L'))
    return (pred > 127).astype(np.uint8)


def load_ground_truth(gt_path):
    """Load ground truth mask (0 or 255) → convert to (0 or 1)"""
    gt = np.array(Image.open(gt_path).convert('L'))
    return (gt > 127).astype(np.uint8)


def compute_strict_metrics(pred, gt):
    """Compute strict per-pixel metrics"""
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    shadow_iou = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou = (shadow_iou + nonshadow_iou) / 2
    
    oa = (tp + tn) / (tp + tn + fp + fn)
    
    shadow_error = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_error = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber = (shadow_error + nonshadow_error) / 2
    
    return {
        'OA': float(oa * 100),
        'Precision': float(precision * 100),
        'Recall': float(recall * 100),
        'F1': float(f1 * 100),
        'BER': float(ber * 100),
        'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100)
    }


# Pre-create kernel once (reused across all images)
_TOLERANCE_KERNEL_CACHE = {}

def _get_tolerance_kernel(tolerance):
    """Get or create cached morphological kernel for given tolerance"""
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (tolerance * 2 + 1, tolerance * 2 + 1)
        )
    return _TOLERANCE_KERNEL_CACHE[tolerance]


def compute_tolerant_metrics(pred, gt, tolerance=5):
    """
    Compute boundary-tolerant metrics using a ±K pixel don't-care zone.
    
    Pixels within ±K px of GT shadow boundaries are excluded from evaluation
    entirely (not counted as TP, FP, TN, or FN), since GT labeling is
    inherently imprecise at boundaries.
    
    Args:
        pred: Binary prediction [H, W] with values {0, 1}
        gt: Binary ground truth [H, W] with values {0, 1}
        tolerance: Pixels within this distance of GT boundary are excluded
        
    Returns:
        dict of metrics computed only on valid (non-band) pixels
    """
    kernel = _get_tolerance_kernel(tolerance)
    gt_uint8 = gt.astype(np.uint8)
    
    # Compute don't-care band via morphological ops (fast, no distance_transform_edt)
    eroded = cv2.erode(gt_uint8, kernel)
    dilated = cv2.dilate(gt_uint8, kernel)
    band = (dilated - eroded) > 0
    
    # Valid mask: everything NOT in the band
    valid = ~band
    
    # Evaluate only on valid pixels
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    
    tp = np.logical_and(pred_valid == 1, gt_valid == 1).sum()
    fp = np.logical_and(pred_valid == 1, gt_valid == 0).sum()
    tn = np.logical_and(pred_valid == 0, gt_valid == 0).sum()
    fn = np.logical_and(pred_valid == 0, gt_valid == 1).sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    shadow_iou = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou = (shadow_iou + nonshadow_iou) / 2
    
    total_valid = tp + tn + fp + fn
    oa = (tp + tn) / (total_valid + 1e-10)
    
    shadow_error = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_error = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber = (shadow_error + nonshadow_error) / 2
    
    total_pixels = pred.size
    num_excluded = band.sum()
    
    return {
        'OA': float(oa * 100),
        'Precision': float(precision * 100),
        'Recall': float(recall * 100),
        'F1': float(f1 * 100),
        'BER': float(ber * 100),
        'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
        'pixels_excluded': int(num_excluded),
        'pct_excluded': float(num_excluded / total_pixels * 100)
    }


def analyze_single_model(pred_dir, gt_dir, output_path):
    """
    Analyze a single model's predictions.
    Saves both strict and tolerant metrics.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir) / 'test' / 'masks'
    
    # Check if predictions exist
    if not pred_dir.exists():
        print(f"  ⚠ Predictions not found: {pred_dir}")
        return None
    
    # Get prediction files FIRST (prioritize actual predictions)
    pred_files = sorted(list(pred_dir.glob('*.png')))
    
    # If we have predictions, use them (ignore marker files)
    if len(pred_files) > 0:
        # Clean up any marker files if predictions exist
        marker_files = list(pred_dir.glob('*_MODEL*.txt'))
        if marker_files:
            print(f"  ℹ Found {len(marker_files)} marker file(s) but predictions exist - removing markers")
            for marker in marker_files:
                marker.unlink()
    else:
        # Only check marker files if there are NO predictions
        if (pred_dir / 'MISSING_MODEL.txt').exists():
            print(f"  ⚠ Missing model marker found (no predictions)")
            return -1
        
        if (pred_dir / 'MODEL_LOAD_FAILED.txt').exists():
            print(f"  ⚠ Model load failed marker found (no predictions)")
            return -1
        
        print(f"  ⚠ No prediction files found")
        return -1
    
    print(f"  Found {len(pred_files)} predictions")
    
    # Accumulate metrics
    strict_metrics_list = []
    tolerant_metrics_list = []
    
    for pred_file in tqdm(pred_files, desc="  Computing metrics", ncols=80):
        # Load prediction
        pred = load_prediction(pred_file)
        
        # Load corresponding GT
        gt_file = gt_dir / pred_file.name
        if not gt_file.exists():
            print(f"  ⚠ GT not found: {gt_file}")
            continue
        
        gt = load_ground_truth(gt_file)
        
        # Compute metrics
        strict = compute_strict_metrics(pred, gt)
        tolerant = compute_tolerant_metrics(pred, gt, tolerance=5)
        
        strict_metrics_list.append(strict)
        tolerant_metrics_list.append(tolerant)
    
    if len(strict_metrics_list) == 0:
        print(f"  ⚠ No valid predictions processed")
        return -1
    
    # Average metrics (exclude non-metric keys from tolerant averaging)
    strict_avg = {}
    tolerant_avg = {}
    
    metric_keys = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
    
    for key in metric_keys:
        strict_avg[key] = float(np.mean([m[key] for m in strict_metrics_list]))
        tolerant_avg[key] = float(np.mean([m[key] for m in tolerant_metrics_list]))
    
    # Also average exclusion stats for tolerant
    tolerant_avg['avg_pixels_excluded'] = float(np.mean([m['pixels_excluded'] for m in tolerant_metrics_list]))
    tolerant_avg['avg_pct_excluded'] = float(np.mean([m['pct_excluded'] for m in tolerant_metrics_list]))
    
    # Save results
    results = {
        'num_images': len(strict_metrics_list),
        'strict': strict_avg,
        'tolerant_5px': tolerant_avg
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save strict
    with open(output_path.parent / 'results_strict.json', 'w') as f:
        json.dump({'num_images': results['num_images'], **results['strict']}, f, indent=4)
    
    # Save tolerant
    with open(output_path.parent / 'results_tolerant.json', 'w') as f:
        json.dump({'num_images': results['num_images'], **results['tolerant_5px']}, f, indent=4)
    
    # Save combined
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"  ✓ Results saved ({len(strict_metrics_list)} images)")
    print(f"    • Strict:   F1={strict_avg['F1']:.2f}% | mIOU={strict_avg['mIOU']:.2f}% | Shadow_IoU={strict_avg['Shadow_IOU']:.2f}%")
    print(f"    • Tolerant: F1={tolerant_avg['F1']:.2f}% | mIOU={tolerant_avg['mIOU']:.2f}% | Shadow_IoU={tolerant_avg['Shadow_IOU']:.2f}%")
    print(f"    • Avg excluded: {tolerant_avg['avg_pct_excluded']:.1f}% of pixels")
    print("")  # Blank line for readability
    
    return results


def analyze_all_results(inference_dir, data_root, output_dir):
    """Analyze all inference results"""
    inference_dir = Path(inference_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ANALYZING INFERENCE RESULTS")
    print("="*80)
    
    # Traverse all test scenarios
    test_types = ['upper', 'loco', 'cross-res']
    
    all_results = {}
    
    for test_type in test_types:
        test_type_dir = inference_dir / test_type
        if not test_type_dir.exists():
            continue
        
        print(f"\n{'='*80}")
        print(f"TEST TYPE: {test_type.upper()}")
        print(f"{'='*80}")
        
        all_results[test_type] = {}
        
        # Iterate over cities
        for city_dir in sorted(test_type_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            
            city = city_dir.name
            print(f"\nCity: {city}")
            
            all_results[test_type][city] = {}
            
            # Iterate over resolutions
            for res_dir in sorted(city_dir.iterdir()):
                if not res_dir.is_dir():
                    continue
                
                res = res_dir.name
                print(f"  Resolution: {res}")
                
                all_results[test_type][city][res] = {}
                
                # Determine GT path
                if test_type == 'cross-res':
                    # Extract test resolution from "midres_to_highres"
                    test_res = res.split('_to_')[-1]
                    gt_path = Path(data_root) / city / test_res
                else:
                    gt_path = Path(data_root) / city / res
                
                # Iterate over models
                for model_dir in sorted(res_dir.iterdir()):
                    if not model_dir.is_dir():
                        continue
                    
                    model = model_dir.name
                    print(f"    Model: {model}")
                    
                    all_results[test_type][city][res][model] = {}
                    
                    # Iterate over variants
                    for variant_dir in sorted(model_dir.iterdir()):
                        if not variant_dir.is_dir():
                            continue
                        
                        variant = variant_dir.name
                        print(f"      Variant: {variant}")
                        
                        # Analyze this model
                        output_path = output_dir / test_type / city / res / model / variant / 'results.json'
                        
                        result = analyze_single_model(
                            pred_dir=variant_dir,
                            gt_dir=gt_path,
                            output_path=output_path
                        )
                        
                        all_results[test_type][city][res][model][variant] = result
    
    # Save comprehensive results
    summary_path = output_dir / 'all_results_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"  Summary saved to: {summary_path}")
    print(f"{'='*80}\n")


def main():
    args = get_args()
    
    analyze_all_results(
        inference_dir=args.inference_dir,
        data_root=args.data_root,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()