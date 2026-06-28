"""
complete_eval.py — Resume post-training evaluation from a saved checkpoint.

Picks up exactly where train_oglanet_sib.py was interrupted:
  1. Loads best_model.pth (which contains the original training args)
  2. Reconstructs the model and test dataloader using those args
  3. Runs test_and_save_predictions  → saves predictions/ + test_results.json
  4. Runs compare_with_baselines     → saves comparison_results.json

Usage (minimal — everything else comes from the checkpoint):
    python complete_eval.py \
        --checkpoint_path /path/to/checkpoints/best_model.pth

Optional overrides (only needed if paths moved since training):
    --base_data_root   /new/path/to/Final_data_test/
    --output_dir       /new/output/dir/          (default: checkpoint's parent)
    --comparison_inference_dir  /path/to/baseline/experiment/
    --comparison_data_root      /path/to/Final_data_test/phoenix/midres/
    --num_workers 2
    --batch_size  4
"""

import os
import sys
import argparse
import json
import logging
import types

import numpy as np
import cv2
from PIL import Image

import torch
from torch.cuda.amp import autocast

# ── Project imports (same as train_oglanet_sib.py) ──────────────────────────
from data.dataset_sib import get_dataloaders_sib
from models.oglanet_sib import OGLANetSIB
from utils.evaluation_detailed import DetailedEvaluator


# ════════════════════════════════════════════════════════════════════════════
# Logging
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str, log_name: str = 'complete_eval.log'):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_name)),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Resume post-training eval from a saved OGLANet+SIB checkpoint.')

    # Required
    p.add_argument('--checkpoint_path', type=str, required=True,
                   help='Path to best_model.pth (contains saved training args)')

    # Optional path overrides
    p.add_argument('--base_data_root', type=str, default=None,
                   help='Override data root (if data moved since training)')
    p.add_argument('--output_dir', type=str, default=None,
                   help='Override output dir (default: checkpoint grandparent dir)')
    p.add_argument('--comparison_inference_dir', type=str, default=None,
                   help='Override baseline inference dir')
    p.add_argument('--comparison_data_root', type=str, default=None,
                   help='Override baseline data root (GT masks location)')

    # Perf overrides
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--batch_size',  type=int, default=None)

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Metric helpers (identical to train_oglanet_sib.py)
# ════════════════════════════════════════════════════════════════════════════

_TOLERANCE_KERNEL_CACHE = {}


def _get_tolerance_kernel(tolerance):
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (tolerance * 2 + 1, tolerance * 2 + 1))
    return _TOLERANCE_KERNEL_CACHE[tolerance]


def _compute_strict_metrics(pred, gt):
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    tn = np.logical_and(pred == 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()

    precision     = tp / (tp + fp + 1e-10)
    recall        = tp / (tp + fn + 1e-10)
    f1            = 2 * precision * recall / (precision + recall + 1e-10)
    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou          = (shadow_iou + nonshadow_iou) / 2
    oa            = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber           = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _compute_tolerant_metrics(pred, gt, tolerance=5):
    kernel   = _get_tolerance_kernel(tolerance)
    gt_uint8 = gt.astype(np.uint8)
    eroded   = cv2.erode(gt_uint8, kernel)
    dilated  = cv2.dilate(gt_uint8, kernel)
    band     = (dilated - eroded) > 0
    valid    = ~band

    p  = pred[valid]
    g  = gt[valid]
    tp = np.logical_and(p == 1, g == 1).sum()
    fp = np.logical_and(p == 1, g == 0).sum()
    tn = np.logical_and(p == 0, g == 0).sum()
    fn = np.logical_and(p == 0, g == 1).sum()

    precision     = tp / (tp + fp + 1e-10)
    recall        = tp / (tp + fn + 1e-10)
    f1            = 2 * precision * recall / (precision + recall + 1e-10)
    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou          = (shadow_iou + nonshadow_iou) / 2
    oa            = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber           = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _average_metrics(metrics_list):
    if not metrics_list:
        return {k: 0.0 for k in
                ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
    keys = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


# ════════════════════════════════════════════════════════════════════════════
# Test + save predictions
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_and_save_predictions(model, loader, device, args, logger):
    model.eval()

    pred_save_dir = os.path.join(args.output_dir, 'predictions')
    os.makedirs(pred_save_dir, exist_ok=True)

    sib_strict_list   = []
    sib_tolerant_list = []
    all_filenames     = []

    for batch_idx, batch in enumerate(loader):
        images        = batch['image'].to(device)
        masks         = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        with autocast(enabled=args.use_amp):
            out         = model(images, intensity_map=intensity_map)
            predictions = out['predictions']

        pred_p6      = predictions['p6']           # (B, C, H, W)
        preds        = pred_p6.argmax(dim=1)       # (B, H, W)

        for i, fname in enumerate(batch['filename']):
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            gt_np   = masks[i].cpu().numpy().astype(np.uint8)

            Image.fromarray(pred_np * 255).save(
                os.path.join(pred_save_dir, fname))

            sib_strict_list.append(_compute_strict_metrics(pred_np, gt_np))
            sib_tolerant_list.append(
                _compute_tolerant_metrics(pred_np, gt_np, tolerance=5))
            all_filenames.append(fname)

        if (batch_idx + 1) % 20 == 0:
            logger.info(f'  Processed {len(all_filenames)} images...')

    sib_strict   = _average_metrics(sib_strict_list)
    sib_tolerant = _average_metrics(sib_tolerant_list)

    logger.info(f'\nSIB Results ({len(all_filenames)} images):')
    logger.info(
        f'  Strict  : OA={sib_strict["OA"]:.2f}  '
        f'P={sib_strict["Precision"]:.2f}  R={sib_strict["Recall"]:.2f}  '
        f'F1={sib_strict["F1"]:.2f}  BER={sib_strict["BER"]:.2f}  '
        f'mIOU={sib_strict["mIOU"]:.2f}  ShIOU={sib_strict["Shadow_IOU"]:.2f}')
    logger.info(
        f'  Tolerant: OA={sib_tolerant["OA"]:.2f}  '
        f'P={sib_tolerant["Precision"]:.2f}  R={sib_tolerant["Recall"]:.2f}  '
        f'F1={sib_tolerant["F1"]:.2f}  BER={sib_tolerant["BER"]:.2f}  '
        f'mIOU={sib_tolerant["mIOU"]:.2f}  ShIOU={sib_tolerant["Shadow_IOU"]:.2f}')

    test_results = {
        'num_images':   len(all_filenames),
        'strict':       sib_strict,
        'tolerant_5px': sib_tolerant,
    }
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
    logger.info(f"  Saved → {os.path.join(args.output_dir, 'test_results.json')}")

    return (sib_strict, sib_tolerant,
            sib_strict_list, sib_tolerant_list,
            all_filenames)


# ════════════════════════════════════════════════════════════════════════════
# Baseline discovery  (identical logic to train_oglanet_sib.py)
# ════════════════════════════════════════════════════════════════════════════

def _find_baseline_experiments(args, logger):
    if not args.comparison_inference_dir:
        return []

    parent_dir = os.path.dirname(args.comparison_inference_dir.rstrip('/'))
    if not os.path.isdir(parent_dir):
        logger.info(f'Baseline parent dir not found: {parent_dir}')
        return []

    test_city = getattr(args, 'test_city', None)
    res       = args.resolution
    if test_city is None:
        return []

    baseline_patterns = [
        ('Upper Bound', [
            f'oglanet_upper_{test_city}_{res}_1',
            f'oglanet_all_{res}_1',
            f'oglanet_base_{test_city}_{res}_1',
        ]),
        ('LOCO Vanilla', [
            f'oglanet_loco_holdout_{test_city}_{res}_1',
            f'oglanet_vanilla_loco_holdout_{test_city}_{res}_1',
        ]),
        ('LOCO FDA', [
            f'oglanet_fda_loco_holdout_{test_city}_{res}_1',
            f'oglanet_loco_fda_holdout_{test_city}_{res}_1',
        ]),
        ('LOCO SegDesic', [
            f'oglanet_segdesic_loco_holdout_{test_city}_{res}_1',
            f'oglanet_loco_segdesic_holdout_{test_city}_{res}_1',
        ]),
        ('LOCO mCL-LC', [
            f'oglanet_mcl_loco_holdout_{test_city}_{res}_1',
            f'oglanet_loco_mcl_holdout_{test_city}_{res}_1',
            f'oglanet_mclc_loco_holdout_{test_city}_{res}_1',
        ]),
    ]

    found = []
    for label, patterns in baseline_patterns:
        for pat in patterns:
            candidate = os.path.join(parent_dir, pat)
            if not os.path.isdir(candidate):
                continue
            pred_dir  = os.path.join(candidate, 'predictions')
            test_json = os.path.join(candidate, 'test_results.json')
            comp_json = os.path.join(candidate, 'comparison_results.json')
            if os.path.isdir(pred_dir):
                found.append((label, pred_dir, 'predictions'))
                logger.info(f'  Found {label}: {pred_dir} (prediction images)')
                break
            elif os.path.isfile(test_json):
                found.append((label, test_json, 'test_results'))
                logger.info(f'  Found {label}: {test_json} (pre-computed)')
                break
            elif os.path.isfile(comp_json):
                found.append((label, comp_json, 'comparison_self'))
                logger.info(f'  Found {label}: {comp_json} (self metrics)')
                break

    # Donor scan
    if len(found) < 3:
        logger.info('  Scanning sibling experiments for donor baseline data...')
        for entry in sorted(os.listdir(parent_dir)):
            if 'sib' in entry.lower() and 'ddib' not in entry.lower():
                continue
            if test_city not in entry.lower() or res not in entry.lower():
                continue
            comp_path = os.path.join(parent_dir, entry, 'comparison_results.json')
            if not os.path.isfile(comp_path):
                continue
            try:
                with open(comp_path) as f:
                    data = json.load(f)
                if len(data.get('baselines', {})) > 0:
                    found.append(('_donor_' + entry, comp_path, 'donor'))
                    logger.info(f'  Found donor baselines in: {entry}')
                    break
            except (json.JSONDecodeError, OSError):
                continue

    return found


def _compute_baseline_metrics_from_predictions(pred_dir, gt_dir, filenames,
                                                img_size, logger):
    strict_list   = []
    tolerant_list = []
    n_matched     = 0

    for fname in filenames:
        pred_path = os.path.join(pred_dir, fname)
        gt_path   = os.path.join(gt_dir,   fname)
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue

        pred_np  = np.array(Image.open(pred_path).convert('L').resize(
            (img_size, img_size), Image.NEAREST))
        pred_bin = (pred_np > 127).astype(np.uint8)

        gt_np  = np.array(Image.open(gt_path).convert('L').resize(
            (img_size, img_size), Image.NEAREST))
        gt_bin = (gt_np > 127).astype(np.uint8)

        strict_list.append(_compute_strict_metrics(pred_bin, gt_bin))
        tolerant_list.append(
            _compute_tolerant_metrics(pred_bin, gt_bin, tolerance=5))
        n_matched += 1

    if n_matched == 0:
        logger.info(f'    Warning: no matching images in {pred_dir}')
        return None

    return {
        'strict':       _average_metrics(strict_list),
        'tolerant_5px': _average_metrics(tolerant_list),
        'strict_list':  strict_list,
        'tolerant_list': tolerant_list,
        'n_images':     n_matched,
    }


# ════════════════════════════════════════════════════════════════════════════
# Comparison  (identical to train_oglanet_sib.py)
# ════════════════════════════════════════════════════════════════════════════

def compare_with_baselines(sib_strict, sib_tolerant,
                           sib_strict_list, sib_tolerant_list,
                           filenames, args, logger):
    test_city = getattr(args, 'test_city', 'unknown')
    res       = args.resolution
    img_size  = args.img_size

    # Locate GT mask directory
    gt_dir = None
    if args.comparison_data_root:
        for candidate in [
            os.path.join(args.comparison_data_root, 'test', 'masks'),
            os.path.join(args.comparison_data_root, 'masks'),
        ]:
            if os.path.isdir(candidate):
                gt_dir = candidate
                break
        if gt_dir is None:
            logger.info(f'Warning: GT mask dir not found under {args.comparison_data_root}')

    logger.info(f'\n{"="*70}')
    logger.info(f'BASELINE COMPARISON')
    logger.info(f'  Test city:  {test_city}  |  Resolution: {res}')
    logger.info(f'  GT masks:   {gt_dir}')
    logger.info(f'{"="*70}')

    found_baselines = _find_baseline_experiments(args, logger)

    baseline_results = {}
    donor_baselines  = {}

    for label, path, source_type in found_baselines:

        if source_type == 'predictions' and gt_dir is not None:
            bl_metrics = _compute_baseline_metrics_from_predictions(
                path, gt_dir, filenames, img_size, logger)
            if bl_metrics:
                baseline_results[label] = bl_metrics
                logger.info(f'  {label}: computed from {bl_metrics["n_images"]} images')

        elif source_type == 'test_results':
            try:
                with open(path) as f:
                    data = json.load(f)
                bl_entry = {}
                if 'strict' in data:
                    bl_entry['strict'] = data['strict']
                if 'tolerant_5px' in data:
                    bl_entry['tolerant_5px'] = data['tolerant_5px']
                if bl_entry:
                    baseline_results[label] = bl_entry
                    logger.info(f'  {label}: loaded from test_results.json')
            except (json.JSONDecodeError, OSError) as e:
                logger.info(f'  {label}: failed to load: {e}')

        elif source_type == 'comparison_self':
            try:
                with open(path) as f:
                    data = json.load(f)
                for key in ['sib', 'ddib']:
                    if key in data and 'strict' in data[key]:
                        baseline_results[label] = {
                            'strict':       data[key]['strict'],
                            'tolerant_5px': data[key].get('tolerant_5px',
                                            data[key].get('tolerant', {})),
                        }
                        logger.info(f'  {label}: loaded self metrics')
                        break
            except (json.JSONDecodeError, OSError) as e:
                logger.info(f'  {label}: failed to load: {e}')

        elif source_type == 'donor':
            try:
                with open(path) as f:
                    data = json.load(f)
                donor_baselines = data.get('baselines', {})
                logger.info(f'  Donor: loaded {len(donor_baselines)} baselines')
            except (json.JSONDecodeError, OSError) as e:
                logger.info(f'  Donor: failed to load: {e}')

    for bl_label, bl_data in donor_baselines.items():
        if bl_label not in baseline_results:
            baseline_results[bl_label] = bl_data
            logger.info(f'  {bl_label}: from donor experiment')

    if baseline_results:
        _print_comparison_table(
            'STRICT METRICS (all pixels)',
            baseline_results, sib_strict, 'strict', logger)
        _print_comparison_table(
            'TOLERANT METRICS (±5 px dont-care zone)',
            baseline_results, sib_tolerant, 'tolerant_5px', logger)
        _print_recovery_ratios(baseline_results, sib_strict, sib_tolerant, logger)

        for bl_label in ['LOCO Vanilla', 'LOCO FDA',
                         'LOCO SegDesic', 'LOCO mCL-LC']:
            if (bl_label in baseline_results
                    and 'strict_list' in baseline_results[bl_label]):
                _print_bootstrap_comparison(
                    baseline_results[bl_label],
                    sib_strict_list, sib_tolerant_list,
                    baseline_label=bl_label, logger=logger)
    else:
        logger.info('\n  No baselines found for comparison.')

    # Save comparison_results.json
    comp = {
        'test_city':  test_city,
        'resolution': res,
        'eval_size':  img_size,
        'sib':  {'strict': sib_strict, 'tolerant_5px': sib_tolerant},
        'ddib': {'strict': sib_strict, 'tolerant_5px': sib_tolerant},
        'baselines': {},
    }
    for label, br in baseline_results.items():
        comp['baselines'][label] = {
            'strict':       br.get('strict', {}),
            'tolerant_5px': br.get('tolerant_5px', br.get('tolerant', {})),
        }
        if 'n_images' in br:
            comp['baselines'][label]['n_images'] = br['n_images']

    comp_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(comp_path, 'w') as f:
        json.dump(comp, f, indent=4)
    logger.info(f'\nComparison saved to {comp_path}')
    return comp


# ════════════════════════════════════════════════════════════════════════════
# Printing helpers  (identical to train_oglanet_sib.py)
# ════════════════════════════════════════════════════════════════════════════

def _print_comparison_table(title, baseline_results, sib_metrics,
                            metric_type, logger):
    logger.info(f'\n{"-"*70}')
    logger.info(f'{title:^70}')
    logger.info(f'{"-"*70}')
    logger.info(
        f'  {"Method":<20} {"OA":>6} {"Prec":>6} {"Rec":>6} '
        f'{"F1":>6} {"BER":>6} {"mIOU":>6} {"ShIOU":>6}')
    logger.info('  ' + '-' * 62)

    for label in ['Upper Bound', 'LOCO Vanilla', 'LOCO FDA',
                  'LOCO SegDesic', 'LOCO mCL-LC']:
        if label not in baseline_results:
            continue
        m = baseline_results[label].get(metric_type, {})
        if not m:
            continue
        logger.info(
            f'  {label:<20} {m.get("OA", 0):6.2f} '
            f'{m.get("Precision", 0):6.2f} {m.get("Recall", 0):6.2f} '
            f'{m.get("F1", 0):6.2f} {m.get("BER", 0):6.2f} '
            f'{m.get("mIOU", 0):6.2f} {m.get("Shadow_IOU", 0):6.2f}')

    d = sib_metrics
    logger.info(
        f'  {"SIB (ours)":<20} {d["OA"]:6.2f} '
        f'{d["Precision"]:6.2f} {d["Recall"]:6.2f} '
        f'{d["F1"]:6.2f} {d["BER"]:6.2f} '
        f'{d["mIOU"]:6.2f} {d["Shadow_IOU"]:6.2f}')


def _print_recovery_ratios(baseline_results, sib_strict, sib_tolerant, logger):
    if ('Upper Bound'  not in baseline_results
            or 'LOCO Vanilla' not in baseline_results):
        return

    logger.info(f'\n{"-"*70}')
    logger.info(f'{"RECOVERY RATIOS":^70}')
    logger.info(f'  R = (SIB − LOCO_Vanilla) / (Upper − LOCO_Vanilla)')
    logger.info(f'  0 = no help  |  1 = gap fully closed')
    logger.info(f'{"-"*70}')

    for eval_type, sib_m, label in [
            ('strict',       sib_strict,   'Strict'),
            ('tolerant_5px', sib_tolerant, 'Tolerant')]:
        ub = baseline_results['Upper Bound'].get(eval_type, {})
        lv = baseline_results['LOCO Vanilla'].get(eval_type, {})
        if not ub or not lv:
            continue
        parts = []
        for k in ['F1', 'mIOU', 'Shadow_IOU', 'BER']:
            if k not in ub or k not in lv:
                continue
            gap = ub[k] - lv[k]
            rec = sib_m[k] - lv[k]
            if k == 'BER':
                gap, rec = -gap, -rec
            R = rec / gap if abs(gap) > 0.01 else float('nan')
            parts.append(f'{k}={R:.3f}')
        logger.info(f'  {label:<10}  ' + '  '.join(parts))


def _print_bootstrap_comparison(loco_baseline, sib_strict_list,
                                 sib_tolerant_list, baseline_label,
                                 logger, n_bootstrap=5000):
    logger.info(f'\n{"-"*70}')
    logger.info(f'{"BOOTSTRAP: SIB vs " + baseline_label + " (n=5000)":^70}')
    logger.info(f'{"-"*70}')

    np.random.seed(42)
    for eval_type, sib_list, label in [
            ('strict_list',   sib_strict_list,   'Strict'),
            ('tolerant_list', sib_tolerant_list, 'Tolerant')]:
        loco_list = loco_baseline.get(eval_type, [])
        n = min(len(loco_list), len(sib_list))
        if n == 0:
            continue
        logger.info(f'\n  {label}:')
        for k in ['F1', 'mIOU', 'Shadow_IOU']:
            loco_vals  = np.array([m[k] for m in loco_list[:n]])
            sib_vals   = np.array([m[k] for m in sib_list[:n]])
            diff       = sib_vals - loco_vals
            obs_mean   = np.mean(diff)
            boot_means = np.array([
                np.mean(diff[np.random.choice(n, n, replace=True)])
                for _ in range(n_bootstrap)])
            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
            if obs_mean >= 0:
                p_val = 2 * max(np.mean(boot_means <= 0), 1.0 / n_bootstrap)
            else:
                p_val = 2 * max(np.mean(boot_means >= 0), 1.0 / n_bootstrap)
            p_val = min(p_val, 1.0)
            sig = (' ***' if p_val < 0.001 else
                   ' **'  if p_val < 0.01  else
                   ' *'   if p_val < 0.05  else '')
            logger.info(
                f'    {k:<12} delta={obs_mean:+.2f}  '
                f'95%CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  p={p_val:.4f}{sig}')
    logger.info('')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    cli = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load checkpoint (contains saved training args) ──────────────────────
    print(f'Loading checkpoint: {cli.checkpoint_path}')
    checkpoint = torch.load(cli.checkpoint_path, map_location=device,
                            weights_only=False)

    saved_args_dict = checkpoint.get('args', {})
    if not saved_args_dict:
        raise RuntimeError(
            'Checkpoint does not contain saved args. '
            'Cannot reconstruct model configuration.')

    # Build an args namespace from saved dict
    args = types.SimpleNamespace(**saved_args_dict)

    # ── Apply CLI overrides ──────────────────────────────────────────────────
    if cli.base_data_root is not None:
        args.base_data_root = cli.base_data_root
    if cli.output_dir is not None:
        args.output_dir = cli.output_dir
    else:
        # Default: grandparent of checkpoint (i.e., the experiment dir)
        args.output_dir = os.path.dirname(os.path.dirname(cli.checkpoint_path))
    if cli.comparison_inference_dir is not None:
        args.comparison_inference_dir = cli.comparison_inference_dir
    if cli.comparison_data_root is not None:
        args.comparison_data_root = cli.comparison_data_root
    if cli.num_workers is not None:
        args.num_workers = cli.num_workers
    if cli.batch_size is not None:
        args.batch_size = cli.batch_size

    # Guard: ensure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Skip if already completed (idempotency guard)
    comp_path = os.path.join(args.output_dir, 'comparison_results.json')
    pred_dir  = os.path.join(args.output_dir, 'predictions')
    if os.path.exists(comp_path) and os.path.isdir(pred_dir):
        n_preds = len([f for f in os.listdir(pred_dir)
                       if f.lower().endswith(('.png', '.jpg', '.tif'))])
        if n_preds > 0:
            print(f'Evaluation already complete ({n_preds} predictions found). Exiting.')
            print(f'  comparison_results.json: {comp_path}')
            print(f'  predictions/            : {pred_dir}')
            return

    logger = setup_logging(args.output_dir)
    logger.info('=' * 70)
    logger.info('OGLANet+SIB  —  Completing interrupted evaluation')
    logger.info('=' * 70)
    logger.info(f'  Checkpoint  : {cli.checkpoint_path}')
    logger.info(f'  Trained at epoch {checkpoint.get("epoch", "?")}  '
                f'best_mIOU={checkpoint.get("best_miou", 0):.4f}')
    logger.info(f'  Output dir  : {args.output_dir}')
    logger.info(f'  Device      : {device}')
    logger.info(f'  Mode        : {args.mode}  |  city: '
                f'{getattr(args, "test_city", "?")}  |  res: {args.resolution}')

    # ── Ensure derived attributes are present (older checkpoints may lack them) ──
    if not hasattr(args, 'use_sib'):
        args.use_sib = getattr(args, 'use_haar', False) or getattr(args, 'use_vib', False)
    if not hasattr(args, 'in_channels'):
        args.in_channels = 4 if getattr(args, 'use_contrast', False) else 3
    if not hasattr(args, 'use_amp'):
        args.use_amp = True
    if not hasattr(args, 'fold_names'):
        args.fold_names = ['phoenix', 'miami', 'chicago']
    if not hasattr(args, 'test_city') and args.mode == 'loco':
        args.test_city = args.fold_names[args.fold_id]

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info('\nBuilding test dataloader...')
    data = get_dataloaders_sib(
        data_root=args.base_data_root,
        test_city=args.test_city,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_contrast=args.use_contrast,
        use_fda=args.use_fda,
        fda_L=args.fda_L,
    )
    test_loader = data['test_loader']
    logger.info(f'  Test set: {len(data["test_dataset"])} samples')

    # ── Model ───────────────────────────────────────────────────────────────
    logger.info('\nReconstructing model...')
    model = OGLANetSIB(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        pretrained_encoder=False,           # weights come from checkpoint
        use_sib=args.use_sib,
        sib_channels=512,
        beta_content=args.beta_content,
        beta_edge=args.beta_edge,
        beta_noise=args.noise_scale,
        adaptive_beta=args.adaptive_beta,
        use_haar=args.use_haar,
        use_vib=args.use_vib,
        use_aug=args.use_content_aug,
        sigma_style=args.sigma_style,
        sigma_shift=args.sigma_shift,
        aug_p_aug=args.aug_p_aug,
        aug_p_mix=args.aug_p_mix,
        use_sag=args.use_sag,
        use_multiscale_sib=args.use_multiscale_sib,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Model loaded  ({n_params:,} parameters)')

    # ── Test: predictions + per-image metrics ───────────────────────────────
    logger.info('\nRunning inference on test set...')
    (sib_strict, sib_tolerant,
     sib_strict_list, sib_tolerant_list,
     all_filenames) = test_and_save_predictions(
        model, test_loader, device, args, logger)

    # ── Baseline comparison ─────────────────────────────────────────────────
    if args.mode == 'loco':
        logger.info('\nRunning baseline comparison...')
        compare_with_baselines(
            sib_strict, sib_tolerant,
            sib_strict_list, sib_tolerant_list,
            all_filenames, args, logger)
    else:
        comp = {
            'sib':      {'strict': sib_strict, 'tolerant_5px': sib_tolerant},
            'ddib':     {'strict': sib_strict, 'tolerant_5px': sib_tolerant},
            'baselines': {},
        }
        with open(comp_path, 'w') as f:
            json.dump(comp, f, indent=4)

    logger.info(f'\n{"="*70}')
    logger.info('Evaluation complete.')
    logger.info(f'  Predictions : {pred_dir}')
    logger.info(f'  Results     : {os.path.join(args.output_dir, "test_results.json")}')
    logger.info(f'  Comparison  : {comp_path}')
    logger.info(f'{"="*70}')


if __name__ == '__main__':
    main()