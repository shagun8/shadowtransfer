"""
Post-hoc class-conditional temperature scaling on a trained MAMNet+SIB checkpoint.

Reads:  <output_dir>/best_model.pth + the LOCO val/test loaders.
Writes: <output_dir>/tempscale_results.json
        <output_dir>/predictions_tempscale/*.png

Reproduces the training-time model construction from args saved in the
checkpoint, loads weights, runs val to fit (T_pos, T_neg), applies
tempscale to test logits, computes SP-gap metrics + strict/tolerant.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_sib import build_mamnet_sib
from data.dataset_sib import get_dataloaders_sib
from utils.postprocessing import filter_small_predictions

# Reuse helpers already defined in train_mamnet_sib.py
from train_mamnet_sib import (
    _compute_strict_metrics, _compute_tolerant_metrics, _average_metrics,
    collect_logits_and_labels, fit_class_conditional_temperature,
    apply_tempscale, compute_sp_metrics,
)

CITY_FOLDS = {0: 'phoenix', 1: 'miami', 2: 'chicago'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True,
                   help='Path to a trained C4-clean output dir containing best_model.pth')
    p.add_argument('--base_data_root', type=str, required=True)
    p.add_argument('--resolution', type=str, default='highres')
    p.add_argument('--fold_id', type=int, required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--boundary_tolerance', type=int, default=2)
    p.add_argument('--tempscale_max_iter', type=int, default=200)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No best_model.pth in {args.checkpoint_dir}')

    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = argparse.Namespace(**ckpt['args'])

    # Resolve LOCO test city + FDA target from the training args
    test_city = CITY_FOLDS[args.fold_id]
    fda_target_root = (os.path.join(args.base_data_root, test_city, args.resolution)
                       if getattr(train_args, 'use_fda', False) else None)

    print(f'Building dataloaders: holdout={test_city}, res={args.resolution}')
    dataloaders = get_dataloaders_sib(
        data_root=None,
        base_data_root=args.base_data_root,
        mode='loco',
        resolution=args.resolution,
        fold_id=args.fold_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_fda=getattr(train_args, 'use_fda', False),
        fda_target_root=fda_target_root,
        fda_L=getattr(train_args, 'fda_L', 0.005),
        use_contrast=getattr(train_args, 'use_contrast', False),
    )
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    print('Rebuilding model from saved args...')
    model = build_mamnet_sib(train_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded best model from epoch {ckpt["epoch"]}')

    # Step 1: collect val logits
    print('\nCollecting source-city validation logits...')
    val_logits, val_labels, _ = collect_logits_and_labels(model, val_loader, device)
    val_logits = val_logits.to(device)
    val_labels = val_labels.to(device)
    print(f'  Val: {val_logits.size(0)} images')

    # Step 2: fit T_pos, T_neg
    print('Fitting (T_pos, T_neg) via LBFGS...')
    T_pos, T_neg = fit_class_conditional_temperature(
        val_logits, val_labels, max_iter=args.tempscale_max_iter)
    print(f'  -> T_pos = {T_pos:.4f}, T_neg = {T_neg:.4f}')

    # Step 3: collect test logits + compute SP-gap metrics
    print('Collecting test logits and computing SP-gap metrics...')
    test_logits, test_labels, test_fnames = collect_logits_and_labels(
        model, test_loader, device)

    sp_baseline = compute_sp_metrics(test_logits, test_labels)

    scaled_test_logits = apply_tempscale(test_logits, T_pos, T_neg)
    sp_tempscale = compute_sp_metrics(scaled_test_logits, test_labels)

    # Re-compute strict/tolerant on baseline AND rescaled predictions
    print('Computing strict/tolerant metrics for baseline and rescaled...')
    base_filtered = filter_small_predictions(test_logits, min_pixels=10)
    base_preds = base_filtered.argmax(dim=1).numpy().astype(np.uint8)

    ts_filtered = filter_small_predictions(scaled_test_logits, min_pixels=10)
    ts_preds = ts_filtered.argmax(dim=1).numpy().astype(np.uint8)
    gt_np = test_labels.numpy().astype(np.uint8)

    ts_pred_dir = os.path.join(args.checkpoint_dir, 'predictions_tempscale')
    os.makedirs(ts_pred_dir, exist_ok=True)

    base_strict_list, base_tolerant_list = [], []
    ts_strict_list, ts_tolerant_list = [], []
    for i, fn in enumerate(test_fnames):
        Image.fromarray(ts_preds[i] * 255).save(os.path.join(ts_pred_dir, fn))
        base_strict_list.append(_compute_strict_metrics(base_preds[i], gt_np[i]))
        base_tolerant_list.append(_compute_tolerant_metrics(
            base_preds[i], gt_np[i], tolerance=args.boundary_tolerance))
        ts_strict_list.append(_compute_strict_metrics(ts_preds[i], gt_np[i]))
        ts_tolerant_list.append(_compute_tolerant_metrics(
            ts_preds[i], gt_np[i], tolerance=args.boundary_tolerance))

    base_strict = _average_metrics(base_strict_list)
    base_tolerant = _average_metrics(base_tolerant_list)
    ts_strict = _average_metrics(ts_strict_list)
    ts_tolerant = _average_metrics(ts_tolerant_list)
    tol_key = f'tolerant_{args.boundary_tolerance}px'

    summary = {
        'T_pos': T_pos, 'T_neg': T_neg,
        'baseline_T1': {
            'strict': base_strict, tol_key: base_tolerant,
            'sp_metrics': sp_baseline,
        },
        'tempscale': {
            'strict': ts_strict, tol_key: ts_tolerant,
            'sp_metrics': sp_tempscale,
        },
        'sp_gap_reduction': {
            'aurc_shadow': sp_baseline['aurc_shadow'] - sp_tempscale['aurc_shadow'],
            'ece_pred_pos': sp_baseline['ece_pred_pos'] - sp_tempscale['ece_pred_pos'],
        },
    }
    out_path = os.path.join(args.checkpoint_dir, 'tempscale_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f'\nBaseline (T=1.0):')
    print(f'  mIoU={base_strict["mIOU"]:.2f}  '
          f'AURC_shadow={sp_baseline["aurc_shadow"]:.4f}  '
          f'ECE_pos={sp_baseline["ece_pred_pos"]:.4f}')
    print(f'Tempscale (T_pos={T_pos:.3f}, T_neg={T_neg:.3f}):')
    print(f'  mIoU={ts_strict["mIOU"]:.2f}  '
          f'AURC_shadow={sp_tempscale["aurc_shadow"]:.4f}  '
          f'ECE_pos={sp_tempscale["ece_pred_pos"]:.4f}')
    print(f'dAURC_shadow = {summary["sp_gap_reduction"]["aurc_shadow"]:+.4f}')
    print(f'dECE_pos     = {summary["sp_gap_reduction"]["ece_pred_pos"]:+.4f}')
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()