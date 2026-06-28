"""
Post-hoc class-conditional temperature scaling on a trained OGLANet+SIB
checkpoint. Reads <ckpt_dir>/checkpoints/best_model.pth, writes
<ckpt_dir>/tempscale_results.json.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_sib import OGLANetSIB
from data.dataset_sib import get_dataloaders_sib

from train_oglanet_sib import (
    _compute_strict_metrics, _compute_tolerant_metrics, _average_metrics,
    collect_logits_and_labels_oglanet, fit_class_conditional_temperature,
    apply_tempscale, compute_sp_metrics,
)

CITY_FOLDS = {0: 'phoenix', 1: 'miami', 2: 'chicago'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint_dir', type=str, required=True)
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

    ckpt_path = os.path.join(args.checkpoint_dir, 'checkpoints', 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No checkpoints/best_model.pth in {args.checkpoint_dir}')

    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_args = argparse.Namespace(**ckpt['args'])

    test_city = CITY_FOLDS[args.fold_id]
    print(f'Building dataloaders: holdout={test_city}, res={args.resolution}')
    data = get_dataloaders_sib(
        data_root=args.base_data_root,
        test_city=test_city,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_contrast=getattr(train_args, 'use_contrast', False),
        use_fda=getattr(train_args, 'use_fda', False),
        fda_L=getattr(train_args, 'fda_L', 0.005),
    )
    val_loader = data['val_loader']
    test_loader = data['test_loader']

    print('Rebuilding model from saved args...')
    in_channels = 4 if getattr(train_args, 'use_contrast', False) else 3
    model = OGLANetSIB(
        num_classes=train_args.num_classes,
        in_channels=in_channels,
        pretrained_encoder=train_args.pretrained_encoder,
        use_sib=(train_args.use_haar or train_args.use_vib),
        sib_channels=512,
        beta_content=train_args.beta_content,
        beta_edge=train_args.beta_edge,
        beta_noise=train_args.noise_scale,
        adaptive_beta=train_args.adaptive_beta,
        use_haar=train_args.use_haar,
        use_vib=train_args.use_vib,
        use_aug=train_args.use_content_aug,
        sigma_style=train_args.sigma_style,
        sigma_shift=train_args.sigma_shift,
        aug_p_aug=train_args.aug_p_aug,
        aug_p_mix=train_args.aug_p_mix,
        use_passthrough_gate=getattr(train_args, 'use_passthrough_gate', False),
        use_module_bypass=getattr(train_args, 'use_module_bypass', False),
        use_sag=train_args.use_sag,
        use_multiscale_sib=train_args.use_multiscale_sib,
        skip_ll_vib=getattr(train_args, 'skip_ll_vib', False),
        symmetric_beta=getattr(train_args, 'symmetric_beta', False),
        aug_all_subbands=getattr(train_args, 'aug_all_subbands', False),
        vib_only_band=getattr(train_args, 'vib_only_band', None),
        use_cacr=False, use_ce_aurc=False, use_tent=False,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Loaded best model from epoch {ckpt["epoch"]}')

    # Mimic train_args.use_amp for consistent inference
    train_args.use_amp = getattr(train_args, 'use_amp', True)

    print('\nCollecting source-city validation logits...')
    val_logits, val_labels, _ = collect_logits_and_labels_oglanet(
        model, val_loader, device, train_args)
    val_logits = val_logits.to(device)
    val_labels = val_labels.to(device)
    print(f'  Val: {val_logits.size(0)} images')

    print('Fitting (T_pos, T_neg) via LBFGS...')
    T_pos, T_neg = fit_class_conditional_temperature(
        val_logits, val_labels, max_iter=args.tempscale_max_iter)
    print(f'  -> T_pos = {T_pos:.4f}, T_neg = {T_neg:.4f}')

    print('Collecting test logits and computing SP-gap metrics...')
    test_logits, test_labels, test_fnames = collect_logits_and_labels_oglanet(
        model, test_loader, device, train_args)

    sp_baseline = compute_sp_metrics(test_logits, test_labels)
    scaled_test_logits = apply_tempscale(test_logits, T_pos, T_neg)
    sp_tempscale = compute_sp_metrics(scaled_test_logits, test_labels)

    base_preds = test_logits.argmax(dim=1).numpy().astype(np.uint8)
    ts_preds = scaled_test_logits.argmax(dim=1).numpy().astype(np.uint8)
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

    print(f'\nBaseline (T=1.0):  mIoU={base_strict["mIOU"]:.2f}  '
          f'AURC_shadow={sp_baseline["aurc_shadow"]:.4f}  '
          f'ECE_pos={sp_baseline["ece_pred_pos"]:.4f}')
    print(f'Tempscale:         mIoU={ts_strict["mIOU"]:.2f}  '
          f'AURC_shadow={sp_tempscale["aurc_shadow"]:.4f}  '
          f'ECE_pos={sp_tempscale["ece_pred_pos"]:.4f}')
    print(f'dAURC_shadow = {summary["sp_gap_reduction"]["aurc_shadow"]:+.4f}')
    print(f'dECE_pos     = {summary["sp_gap_reduction"]["ece_pred_pos"]:+.4f}')
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()