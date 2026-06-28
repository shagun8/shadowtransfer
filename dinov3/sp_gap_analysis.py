"""
sp_gap_analysis.py  —  Phase 1 SP-gap measurement for DINOv3 C4-clean.

Same methodology as mamnet/sp_gap_analysis.py, adapted to DINOv3:
  - DINOv3ShadowDetectorSIB.forward returns (logits, sib_losses).
  - Vanilla DINOv3 AURC will be computed in the aggregator from saved .npy
    files in Test_img_probs/, NOT here (this script only handles C4-clean).
  - DINOv3's C4-clean has only Haar + VIB (no SAG, no FDA — see §4.4).

Per-image AURC_shadow on ground-truth shadow pixels:
  Sort gt-shadow pixels by predicted P(shadow) descending.
  At coverage c in linspace(0.10, 1.00, 20), error = fraction of top-c
  predicted as background. AURC = mean over coverage grid.

Bootstrap unit: IMAGE (n=150 per cell). B=10,000 resamples. Two-sided p.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_model_sib import DINOv3ShadowDetectorSIB
from data.dataset_sib import get_dataloaders_sib

CITY_FOLDS = {0: 'phoenix', 1: 'miami', 2: 'chicago'}


# =====================================================================
# Argument parsing
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1 SP-gap analysis for DINOv3 C4-clean')

    p.add_argument('--c4clean_checkpoint', type=str, required=True)
    p.add_argument('--base_data_root', type=str, required=True)
    p.add_argument('--fold_id', type=int, required=True, choices=[0, 1, 2])
    p.add_argument('--resolution', type=str, default='highres',
                   choices=['highres', 'midres'])
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--weights_path', type=str, default=None,
                   help='Path to DINOv3 backbone pretrained weights '
                        '(needed to construct the model architecture)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--bootstrap_B', type=int, default=10000)
    p.add_argument('--min_shadow_pixels', type=int, default=5)
    p.add_argument('--n_coverage', type=int, default=20)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# =====================================================================
# Model loading
# =====================================================================

def load_c4clean_dinov3_sib(checkpoint_path, weights_path, device):
    """
    Load a DINOv3ShadowDetectorSIB from a C4-clean checkpoint.
    Checkpoints saved by train_dinov3_sib.py contain 'args' dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'C4-clean checkpoint not found: {checkpoint_path}')

    print(f'Loading C4-clean DINOv3-SIB: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'args' not in ckpt:
        raise KeyError(
            "C4-clean checkpoint missing 'args' key. "
            "Expected a checkpoint saved by train_dinov3_sib.py.")

    train_args = argparse.Namespace(**ckpt['args'])
    # Override weights_path if provided (for portability)
    backbone_weights = weights_path if weights_path is not None \
        else getattr(train_args, 'weights_path', None)

    model = DINOv3ShadowDetectorSIB(
        num_classes=getattr(train_args, 'num_classes', 2),
        model_name=getattr(train_args, 'model_name', 'dinov3_vits16'),
        weights_path=backbone_weights,
        pretrained=True,
        frozen_stages=getattr(train_args, 'frozen_stages', -1),
        use_haar=train_args.use_haar,
        use_vib=train_args.use_vib,
        use_content_aug=train_args.use_content_aug,
        adaptive_beta=train_args.adaptive_beta,
        use_passthrough_gate=getattr(train_args, 'use_passthrough_gate', False),
        use_module_bypass=getattr(train_args, 'use_module_bypass', False),
        disable_content_vib=getattr(train_args, 'disable_content_vib', False),
        symmetric_vib=getattr(train_args, 'symmetric_vib', False),
        aug_all_subbands=getattr(train_args, 'aug_all_subbands', False),
        vib_on_hl_only=getattr(train_args, 'vib_on_hl_only', False),
        num_domains=2,
        vib_beta_content=getattr(train_args, 'vib_beta_content', 0.01),
        vib_beta_edge=getattr(train_args, 'vib_beta_edge', 0.0001),
        vib_beta_scale=getattr(train_args, 'vib_beta_scale', 0.02),
        aug_sigma_style=getattr(train_args, 'aug_sigma_style', 0.25),
        aug_sigma_shift=getattr(train_args, 'aug_sigma_shift', 0.15),
        aug_p_aug=getattr(train_args, 'aug_p_aug', 0.5),
        aug_p_mix=getattr(train_args, 'aug_p_mix', 0.3),
        use_cacr=getattr(train_args, 'use_cacr', False),
        use_ce_aurc=getattr(train_args, 'use_ce_aurc', False),
        use_tent=getattr(train_args, 'use_tent', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'  Epoch {ckpt.get("epoch", "?")}  '
          f'(best: {ckpt.get("best_miou", "?")})')
    print(f'  SIB config: '
          f'haar={train_args.use_haar}  vib={train_args.use_vib}  '
          f'aug={train_args.use_content_aug}  ab={train_args.adaptive_beta}')
    return model, train_args


# =====================================================================
# Inference
# =====================================================================

@torch.no_grad()
def run_c4clean_inference(model, loader, device):
    """
    Run C4-clean DINOv3-SIB on the test loader.
    DINOv3ShadowDetectorSIB.forward returns (logits, sib_losses).

    Returns:
        c4_logits_list: list of [2, H, W] float32 CPU tensors
        labels_list:    list of [H, W] long CPU tensors
        filenames:      list of strings
    """
    c4_logits_list, labels_list, filenames = [], [], []

    for batch in loader:
        images        = batch['image'].to(device)
        labels        = batch['mask']
        intensity_map = batch['intensity_map'].to(device)

        logits, _ = model(images, intensity_map=intensity_map,
                          vib_warmup_factor=1.0)
        logits = logits.float().cpu()

        B = images.shape[0]
        for i in range(B):
            c4_logits_list.append(logits[i])
            labels_list.append(labels[i])
            filenames.append(batch['filename'][i])

    print(f'  Collected {len(filenames)} C4-clean predictions.')
    return c4_logits_list, labels_list, filenames


# =====================================================================
# Per-image metric functions (identical to MAMNet/OGLANet)
# =====================================================================

def per_image_rc_curve_shadow(shadow_prob_hw, gt_label_hw,
                               n_coverage, min_pixels):
    """Per-image selective shadow error at each coverage level. Returns list of length n_coverage."""
    coverage_grid = np.linspace(0.10, 1.0, n_coverage)
    shadow_mask = (gt_label_hw.ravel() == 1)
    n_shadow = int(shadow_mask.sum())
    if n_shadow < min_pixels:
        return [None] * n_coverage
    confs = shadow_prob_hw.ravel()[shadow_mask].astype(np.float32)
    correct = (confs > 0.5).astype(np.float32)
    sort_idx = np.argsort(-confs)
    sorted_correct = correct[sort_idx]
    errs = []
    for c in coverage_grid:
        k = max(1, int(round(c * n_shadow)))
        errs.append(float(1.0 - sorted_correct[:k].mean()))
    return errs

def compute_aurc_shadow_per_image(shadow_prob_hw, gt_label_hw,
                                   n_coverage, min_pixels):
    """AURC_shadow on gt-shadow pixels. Lower = better."""
    shadow_mask = (gt_label_hw.ravel() == 1)
    n_shadow = int(shadow_mask.sum())
    if n_shadow < min_pixels:
        return float('nan')

    confs = shadow_prob_hw.ravel()[shadow_mask].astype(np.float32)
    correct = (confs > 0.5).astype(np.float32)

    sort_idx = np.argsort(-confs)
    sorted_correct = correct[sort_idx]

    coverage_grid = np.linspace(0.10, 1.0, n_coverage)
    aurc = 0.0
    for c in coverage_grid:
        k = max(1, int(round(c * n_shadow)))
        aurc += 1.0 - float(sorted_correct[:k].mean())
    return aurc / n_coverage


def compute_ece_pred_pos_per_image(shadow_prob_hw, gt_label_hw,
                                    n_bins=15, min_pixels=5):
    """ECE on predicted-positive pixels. Lower = better."""
    pred_pos = (shadow_prob_hw.ravel() > 0.5)
    if int(pred_pos.sum()) < min_pixels:
        return float('nan')

    confs   = shadow_prob_hw.ravel()[pred_pos].astype(np.float32)
    correct = (gt_label_hw.ravel()[pred_pos] == 1).astype(np.float32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confs)
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum() > 0:
            ece += float(m.sum() / n) * abs(
                float(confs[m].mean()) - float(correct[m].mean()))
    return ece


def compute_miou_per_image(shadow_prob_hw, gt_label_hw):
    """Standard mIoU at threshold 0.5 (sanity check)."""
    pred = (shadow_prob_hw > 0.5).astype(np.int32)
    gt = gt_label_hw.astype(np.int32)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    shadow_iou = tp / (tp + fp + fn + 1e-10)
    bg_iou = tn / (tn + fp + fn + 1e-10)
    return float((shadow_iou + bg_iou) / 2)


# =====================================================================
# Main
# =====================================================================

def main():
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    test_city = CITY_FOLDS[args.fold_id]
    print(f'\n{"="*65}')
    print(f'DINOv3 SP-Gap Phase 1 (C4-clean only)')
    print(f'  Architecture : DINOv3')
    print(f'  Holdout city : {test_city}')
    print(f'  Resolution   : {args.resolution}')
    print(f'  Bootstrap B  : {args.bootstrap_B}')
    print(f'{"="*65}')

    print('\nLoading test data...')
    loaders = get_dataloaders_sib(
        data_root=None,
        base_data_root=args.base_data_root,
        mode='loco',
        resolution=args.resolution,
        fold_id=args.fold_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_fda=False,
    )
    test_loader = loaders['test']
    print(f'  Test images: {len(test_loader.dataset)}')

    model, c4_args = load_c4clean_dinov3_sib(
        args.c4clean_checkpoint, args.weights_path, device)

    print('\nRunning C4-clean inference...')
    c4_logits_list, labels_list, filenames = run_c4clean_inference(
        model, test_loader, device)

    n_images = len(filenames)
    print(f'\nComputing per-image metrics ({n_images} images)...')

    c4_aurc_arr = np.empty(n_images)
    c4_ece_arr = np.empty(n_images)
    c4_miou_arr = np.empty(n_images)

    probs_save_dir = os.path.join(args.output_dir, f'c4clean_probs_dinov3_{test_city}_{args.resolution}')
    os.makedirs(probs_save_dir, exist_ok=True)

    test_rc_records = []

    for i in range(n_images):
        gt_np = labels_list[i].numpy().astype(np.int32)
        c4_prob = F.softmax(c4_logits_list[i], dim=0)[1].numpy().astype(np.float16)

        stem = os.path.splitext(filenames[i])[0]
        np.save(os.path.join(probs_save_dir, f'{stem}.npy'), c4_prob)

        c4_prob_f32 = c4_prob.astype(np.float32)
        c4_aurc_arr[i] = compute_aurc_shadow_per_image(
            c4_prob_f32, gt_np, args.n_coverage, args.min_shadow_pixels)
        c4_ece_arr[i] = compute_ece_pred_pos_per_image(
            c4_prob_f32, gt_np, min_pixels=args.min_shadow_pixels)
        c4_miou_arr[i] = compute_miou_per_image(c4_prob_f32, gt_np)

        test_rc_records.append({
            'filename': filenames[i],
            'rc_curve': per_image_rc_curve_shadow(
                c4_prob_f32, gt_np, args.n_coverage, args.min_shadow_pixels),
        })

    n_valid_aurc = int(np.sum(~np.isnan(c4_aurc_arr)))

    print(f'\n  AURC_shadow  : mean={np.nanmean(c4_aurc_arr):.4f}  '
          f'median={np.nanmedian(c4_aurc_arr):.4f}  '
          f'(valid={n_valid_aurc}/{n_images})')
    print(f'  ECE_pred_pos : mean={np.nanmean(c4_ece_arr):.4f}')
    print(f'  mIoU         : mean={np.nanmean(c4_miou_arr):.4f}')

    per_image_records = []
    for i in range(n_images):
        per_image_records.append({
            'filename':    filenames[i],
            'aurc_shadow': float(c4_aurc_arr[i]) if not np.isnan(c4_aurc_arr[i]) else None,
            'ece_pred_pos': float(c4_ece_arr[i]) if not np.isnan(c4_ece_arr[i]) else None,
            'miou':        float(c4_miou_arr[i]),
        })

    val_loader = loaders['val']
    print(f'\nRunning C4-clean inference on source-val ({len(val_loader.dataset)} images)...')
    val_rc_records = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['mask']
            intensity_map = batch['intensity_map'].to(device)
            logits, _ = model(images, intensity_map=intensity_map,
                            vib_warmup_factor=1.0)
            c4_logits = logits.float().cpu()
            for i in range(images.shape[0]):
                gt = labels[i].numpy().astype(np.int32)
                prob = F.softmax(c4_logits[i], dim=0)[1].numpy()
                rc = per_image_rc_curve_shadow(prob, gt, args.n_coverage,
                                            args.min_shadow_pixels)
                val_rc_records.append({'filename': batch['filename'][i],
                                    'rc_curve': rc})

    results = {
        'architecture':       'dinov3',
        'method':             'c4clean',
        'test_city':          test_city,
        'resolution':         args.resolution,
        'fold_id':            args.fold_id,
        'n_images_total':     n_images,
        'n_valid_aurc':       n_valid_aurc,
        'min_shadow_pixels':  args.min_shadow_pixels,
        'n_coverage_levels':  args.n_coverage,
        'c4clean_checkpoint': args.c4clean_checkpoint,
        'c4clean_probs_dir':  probs_save_dir,
        'aurc_shadow_mean':   float(np.nanmean(c4_aurc_arr)),
        'aurc_shadow_median': float(np.nanmedian(c4_aurc_arr)),
        'ece_pred_pos_mean':  float(np.nanmean(c4_ece_arr)),
        'miou_mean':          float(np.nanmean(c4_miou_arr)),
        'per_image':          per_image_records,
    }

    results['val_rc_records']  = val_rc_records
    results['test_rc_records'] = test_rc_records
    results['coverage_grid']   = list(np.linspace(0.10, 1.0, args.n_coverage).astype(float))

    out_path = os.path.join(
        args.output_dir,
        f'sp_gap_dinov3_c4clean_{test_city}_{args.resolution}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {out_path}')
    print(f'C4-clean .npy maps saved to: {probs_save_dir}')


if __name__ == '__main__':
    main()