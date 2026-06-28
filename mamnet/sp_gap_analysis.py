"""
sp_gap_analysis.py  —  Phase 1 SP-gap measurement (§4.3 primary metric).

PATCHED: Now writes the same JSON shape as oglanet/dinov3 versions:
  - includes 'per_image' array with per-image AURC/ECE/mIoU records
  - saves C4-clean .npy probability maps to a side directory
  - the existing Vanilla-vs-C4 internal comparison + bootstrap CI is
    preserved at the top level for backwards compatibility

This script is idempotent on the existing checkpoint set — re-running it
gives the same numbers (modulo bootstrap RNG seed = 42) but adds the
per-image records the aggregator needs.

Loads a Vanilla MAMNet checkpoint and a C4-clean MAMNetSIB checkpoint,
runs both on the held-out LOCO test city in a single joint loop
(guaranteeing identical image order), and computes per-image AURC_shadow.

Statistical design (pre-registered):
    PRIMARY endpoint: AURC_shadow delta.
    Bootstrap unit: IMAGE (n=150 per cell). Pixel correlation within an
    image is absorbed — each image contributes exactly one observation.
    B=10,000 bootstrap resamples.

Interpretation:
    delta_i = AURC_shadow(C4clean)_i - AURC_shadow(Vanilla)_i
    Negative delta -> C4clean BETTER (reduced shadow-class selective error).
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet import MAMNet
from models.mamnet_sib import build_mamnet_sib
from data.dataset_sib import get_dataloaders_sib

CITY_FOLDS = {0: 'phoenix', 1: 'miami', 2: 'chicago'}


# =====================================================================
# Argument parsing
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 1 SP-gap analysis: AURC_shadow for Vanilla vs C4-clean')

    p.add_argument('--vanilla_checkpoint', type=str, required=True,
                   help='Path to Vanilla MAMNet checkpoint_best.pth')
    p.add_argument('--c4clean_checkpoint', type=str, required=True,
                   help='Path to C4-clean MAMNetSIB best_model.pth')
    p.add_argument('--base_data_root', type=str, required=True,
                   help='Root of Final_data_test')
    p.add_argument('--fold_id', type=int, required=True, choices=[0, 1, 2])
    p.add_argument('--resolution', type=str, default='highres',
                   choices=['highres', 'midres'])
    p.add_argument('--output_dir', type=str, required=True)
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

def load_vanilla_mamnet(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Vanilla checkpoint not found: {checkpoint_path}')

    print(f'Loading Vanilla MAMNet: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = MAMNet(num_classes=2, pretrained=False, use_aux=True,
                   use_contrast=True)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print(f'  Epoch {ckpt.get("epoch", "?")}  '
          f'(best val metric: {ckpt.get("best_metric", "?")})')
    return model


def load_c4clean_mamnet_sib(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'C4-clean checkpoint not found: {checkpoint_path}')

    print(f'Loading C4-clean MAMNetSIB: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'args' not in ckpt:
        raise KeyError(
            "C4-clean checkpoint missing 'args' key. "
            "Expected a checkpoint saved by train_mamnet_sib.py.")

    train_args = argparse.Namespace(**ckpt['args'])
    model = build_mamnet_sib(train_args)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f'  Epoch {ckpt.get("epoch", "?")}  '
          f'(best metric: {ckpt.get("best_metric", "?")})')
    print(f'  SIB config: '
          f'haar={train_args.use_haar}  vib={train_args.use_vib}  '
          f'sag={train_args.use_sag}  fda={train_args.use_fda}  '
          f'aug={train_args.use_content_aug}  ab={train_args.adaptive_beta}')
    return model, train_args


# =====================================================================
# Joint inference
# =====================================================================

@torch.no_grad()
def run_joint_inference(vanilla_model, c4clean_model, loader, device):
    van_logits_list, c4_logits_list = [], []
    labels_list, filenames = [], []

    for batch in loader:
        images        = batch['image'].to(device)
        labels        = batch['mask']
        intensity_map = batch['intensity_map'].to(device)

        # Vanilla MAMNet forward
        van_out = vanilla_model(images)
        if isinstance(van_out, tuple):
            van_out = van_out[0]
        if isinstance(van_out, dict):
            van_logits = van_out.get('main', list(van_out.values())[0])
        else:
            van_logits = van_out
        van_logits = van_logits.float().cpu()

        # C4-clean MAMNetSIB forward
        c4_out, _ = c4clean_model(images, intensity_map=intensity_map)
        if isinstance(c4_out, dict):
            c4_logits = c4_out.get('main', list(c4_out.values())[0])
        else:
            c4_logits = c4_out
        c4_logits = c4_logits.float().cpu()

        B = images.shape[0]
        for i in range(B):
            van_logits_list.append(van_logits[i])
            c4_logits_list.append(c4_logits[i])
            labels_list.append(labels[i])
            filenames.append(batch['filename'][i])

    print(f'  Collected {len(filenames)} paired predictions.')
    return van_logits_list, c4_logits_list, labels_list, filenames


# =====================================================================
# Per-image metric functions
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
    pred_pos = (shadow_prob_hw.ravel() > 0.5)
    if int(pred_pos.sum()) < min_pixels:
        return float('nan')

    confs   = shadow_prob_hw.ravel()[pred_pos].astype(np.float32)
    correct = (gt_label_hw.ravel()[pred_pos] == 1).astype(np.float32)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    n     = len(confs)
    for i in range(n_bins):
        m = (confs > edges[i]) & (confs <= edges[i + 1])
        if m.sum() > 0:
            ece += float(m.sum() / n) * abs(
                float(confs[m].mean()) - float(correct[m].mean()))
    return ece


def compute_miou_per_image(shadow_prob_hw, gt_label_hw):
    pred = (shadow_prob_hw > 0.5).astype(np.int32)
    gt   = gt_label_hw.astype(np.int32)
    tp   = int(((pred == 1) & (gt == 1)).sum())
    fp   = int(((pred == 1) & (gt == 0)).sum())
    tn   = int(((pred == 0) & (gt == 0)).sum())
    fn   = int(((pred == 0) & (gt == 1)).sum())
    shadow_iou = tp / (tp + fp + fn + 1e-10)
    bg_iou     = tn / (tn + fp + fn + 1e-10)
    return float((shadow_iou + bg_iou) / 2)


# =====================================================================
# Bootstrap
# =====================================================================

def bootstrap_delta_ci(deltas, B, seed=42):
    valid = deltas[~np.isnan(deltas)]
    n = len(valid)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    obs_mean = float(np.mean(valid))
    rng = np.random.RandomState(seed)

    boot_means = np.array([
        float(np.mean(valid[rng.choice(n, n, replace=True)]))
        for _ in range(B)
    ])

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    if obs_mean < 0:
        p = 2.0 * max(float(np.mean(boot_means >= 0)), 1.0 / B)
    else:
        p = 2.0 * max(float(np.mean(boot_means <= 0)), 1.0 / B)
    p = min(p, 1.0)

    return obs_mean, ci_lo, ci_hi, p


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
    print(f'SP-Gap Phase 1 Analysis  —  MAMNet')
    print(f'  Holdout city : {test_city}')
    print(f'  Resolution   : {args.resolution}')
    print(f'  Bootstrap B  : {args.bootstrap_B}')
    print(f'  Min shadow px: {args.min_shadow_pixels}')
    print(f'  Coverage grid: {args.n_coverage} levels')
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
        fda_target_root=None,
        fda_L=0.005,
        use_contrast=True,
    )
    test_loader = loaders['test']
    print(f'  Test images  : {len(test_loader.dataset)}')

    vanilla_model           = load_vanilla_mamnet(
        args.vanilla_checkpoint, device)
    c4clean_model, c4_args  = load_c4clean_mamnet_sib(
        args.c4clean_checkpoint, device)

    print('\nRunning joint inference...')
    van_logits_list, c4_logits_list, labels_list, filenames = \
        run_joint_inference(vanilla_model, c4clean_model, test_loader, device)

    n_images = len(filenames)
    print(f'\nComputing per-image metrics ({n_images} images)...')

    van_aurc_arr = np.empty(n_images)
    c4_aurc_arr  = np.empty(n_images)
    van_ece_arr  = np.empty(n_images)
    c4_ece_arr   = np.empty(n_images)
    van_miou_arr = np.empty(n_images)
    c4_miou_arr  = np.empty(n_images)

    # Save C4-clean .npy probability maps for downstream tooling
    probs_save_dir = os.path.join(
        args.output_dir,
        f'c4clean_probs_mamnet_{test_city}_{args.resolution}')
    os.makedirs(probs_save_dir, exist_ok=True)

    for i in range(n_images):
        gt_np = labels_list[i].numpy().astype(np.int32)

        van_prob = F.softmax(van_logits_list[i], dim=0)[1].numpy()
        c4_prob  = F.softmax(c4_logits_list[i],  dim=0)[1].numpy()

        # Save C4-clean prob map (for reproducibility / aggregator)
        stem = os.path.splitext(filenames[i])[0]
        np.save(os.path.join(probs_save_dir, f'{stem}.npy'),
                c4_prob.astype(np.float16))

        van_aurc_arr[i] = compute_aurc_shadow_per_image(
            van_prob, gt_np, args.n_coverage, args.min_shadow_pixels)
        c4_aurc_arr[i]  = compute_aurc_shadow_per_image(
            c4_prob,  gt_np, args.n_coverage, args.min_shadow_pixels)

        van_ece_arr[i] = compute_ece_pred_pos_per_image(
            van_prob, gt_np, min_pixels=args.min_shadow_pixels)
        c4_ece_arr[i]  = compute_ece_pred_pos_per_image(
            c4_prob,  gt_np, min_pixels=args.min_shadow_pixels)

        van_miou_arr[i] = compute_miou_per_image(van_prob, gt_np)
        c4_miou_arr[i]  = compute_miou_per_image(c4_prob,  gt_np)

    # Deltas
    delta_aurc = c4_aurc_arr  - van_aurc_arr
    delta_ece  = c4_ece_arr   - van_ece_arr
    delta_miou = c4_miou_arr  - van_miou_arr

    n_valid_aurc   = int(np.sum(~np.isnan(delta_aurc)))
    n_improve_aurc = int(np.sum(delta_aurc < 0))

    print(f'Running bootstrap (B={args.bootstrap_B})...')
    m_aurc,  ci_lo_aurc,  ci_hi_aurc,  p_aurc  = \
        bootstrap_delta_ci(delta_aurc, args.bootstrap_B)
    m_ece,   ci_lo_ece,   ci_hi_ece,   p_ece   = \
        bootstrap_delta_ci(delta_ece,  args.bootstrap_B)
    m_miou,  ci_lo_miou,  ci_hi_miou,  p_miou  = \
        bootstrap_delta_ci(delta_miou, args.bootstrap_B)

    def _dir(v):
        if np.isnan(v): return '—'
        return 'C4-clean BETTER ↓' if v < 0 else 'Vanilla BETTER ↑'

    print(f'\n{"="*65}')
    print(f'RESULTS  |  MAMNet  {test_city}  {args.resolution}')
    print(f'{"="*65}')

    print(f'\n  PRIMARY ENDPOINT — AURC_shadow  (lower = better)')
    print(f'  Vanilla  : {np.nanmean(van_aurc_arr):.4f}')
    print(f'  C4-clean : {np.nanmean(c4_aurc_arr):.4f}')
    print(f'  Δ(C4−Van): {m_aurc:+.4f}  95%CI [{ci_lo_aurc:+.4f}, {ci_hi_aurc:+.4f}]')
    print(f'  p (two-sided bootstrap): {p_aurc:.4f}')
    print(f'  Direction: {n_improve_aurc}/{n_valid_aurc} images improve  →  {_dir(m_aurc)}')

    print(f'\n  SECONDARY — ECE_pred_pos')
    print(f'  Δ(C4−Van): {m_ece:+.4f}  95%CI [{ci_lo_ece:+.4f}, {ci_hi_ece:+.4f}]  p={p_ece:.4f}')

    print(f'\n  SANITY — mIoU')
    print(f'  Vanilla  : {np.nanmean(van_miou_arr):.2f}%')
    print(f'  C4-clean : {np.nanmean(c4_miou_arr):.2f}%')
    print(f'  Δ(C4−Van): {m_miou:+.4f}  95%CI [{ci_lo_miou:+.4f}, {ci_hi_miou:+.4f}]  p={p_miou:.4f}')

    # ── Build per-image records (NEW — what the aggregator needs) ────
    per_image_records = []
    for i in range(n_images):
        per_image_records.append({
            'filename':    filenames[i],
            # C4-clean: this is what the aggregator reads
            'aurc_shadow': float(c4_aurc_arr[i]) if not np.isnan(c4_aurc_arr[i]) else None,
            'ece_pred_pos': float(c4_ece_arr[i]) if not np.isnan(c4_ece_arr[i]) else None,
            'miou':        float(c4_miou_arr[i]),
            # Also include Vanilla per-image (useful for downstream auditing)
            'vanilla_aurc_shadow':  float(van_aurc_arr[i]) if not np.isnan(van_aurc_arr[i]) else None,
            'vanilla_ece_pred_pos': float(van_ece_arr[i]) if not np.isnan(van_ece_arr[i]) else None,
            'vanilla_miou':         float(van_miou_arr[i]),
        })

    # ---- Source-val inference for c*_val fitting (label-free on held-out) ----
    val_loader = loaders['val']
    print(f'\nRunning C4-clean inference on source-val ({len(val_loader.dataset)} images)...')
    val_rc_records = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['mask']
            intensity_map = batch['intensity_map'].to(device)
            c4_out, _ = c4clean_model(images, intensity_map=intensity_map)
            if isinstance(c4_out, dict):
                c4_logits = c4_out.get('main', list(c4_out.values())[0])
            else:
                c4_logits = c4_out
            c4_logits = c4_logits.float().cpu()
            for i in range(images.shape[0]):
                gt = labels[i].numpy().astype(np.int32)
                prob = F.softmax(c4_logits[i], dim=0)[1].numpy()
                rc = per_image_rc_curve_shadow(prob, gt, args.n_coverage,
                                            args.min_shadow_pixels)
                val_rc_records.append({'filename': batch['filename'][i],
                                    'rc_curve': rc})

    # Per-image test RC curves (parallel to existing per-image AURC scalars)
    test_rc_records = []
    for i in range(n_images):
        gt_np = labels_list[i].numpy().astype(np.int32)
        c4_prob = F.softmax(c4_logits_list[i], dim=0)[1].numpy()
        rc = per_image_rc_curve_shadow(c4_prob, gt_np, args.n_coverage,
                                    args.min_shadow_pixels)
        test_rc_records.append({'filename': filenames[i], 'rc_curve': rc})

    # ── Build full results JSON (preserves old shape + adds per_image) ──
    results = {
        # ---- Top-level metadata (compatible with aggregator) ------------
        'architecture':       'mamnet',
        'method':             'c4clean',
        'test_city':          test_city,
        'resolution':         args.resolution,
        'fold_id':            args.fold_id,
        'n_images_total':     n_images,
        'n_valid_aurc':       n_valid_aurc,
        'n_improve_aurc':     n_improve_aurc,
        'bootstrap_B':        args.bootstrap_B,
        'min_shadow_pixels':  args.min_shadow_pixels,
        'n_coverage_levels':  args.n_coverage,
        'vanilla_checkpoint': args.vanilla_checkpoint,
        'c4clean_checkpoint': args.c4clean_checkpoint,
        'c4clean_probs_dir':  probs_save_dir,
        # ---- C4-clean cell summary (matches OGLANet/DINOv3 shape) ------
        'aurc_shadow_mean':   float(np.nanmean(c4_aurc_arr)),
        'aurc_shadow_median': float(np.nanmedian(c4_aurc_arr)),
        'ece_pred_pos_mean':  float(np.nanmean(c4_ece_arr)),
        'miou_mean':          float(np.nanmean(c4_miou_arr)),
        # ---- Per-image records (what aggregator reads) ------------------
        'per_image':          per_image_records,
        # ---- Legacy nested comparison (preserved for auditing) ---------
        'aurc_shadow': {
            'vanilla_mean':  float(np.nanmean(van_aurc_arr)),
            'c4clean_mean':  float(np.nanmean(c4_aurc_arr)),
            'mean_delta':    m_aurc,
            'ci_lo':         ci_lo_aurc,
            'ci_hi':         ci_hi_aurc,
            'p_two_sided':   p_aurc,
            'n_improve':     n_improve_aurc,
            'n_valid':       n_valid_aurc,
        },
        'ece_pred_pos': {
            'vanilla_mean':  float(np.nanmean(van_ece_arr)),
            'c4clean_mean':  float(np.nanmean(c4_ece_arr)),
            'mean_delta':    m_ece,
            'ci_lo':         ci_lo_ece,
            'ci_hi':         ci_hi_ece,
            'p_two_sided':   p_ece,
        },
        'miou': {
            'vanilla_mean':  float(np.nanmean(van_miou_arr)),
            'c4clean_mean':  float(np.nanmean(c4_miou_arr)),
            'mean_delta':    m_miou,
            'ci_lo':         ci_lo_miou,
            'ci_hi':         ci_hi_miou,
            'p_two_sided':   p_miou,
        },
    }

    # Add to results dict (do NOT remove existing fields)
    results['val_rc_records'] = val_rc_records
    results['test_rc_records'] = test_rc_records
    results['coverage_grid'] = list(np.linspace(0.10, 1.0, args.n_coverage).astype(float))

    out_path = os.path.join(
        args.output_dir,
        f'sp_gap_mamnet_{test_city}_{args.resolution}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {out_path}')
    print(f'C4-clean .npy maps saved to: {probs_save_dir}')


if __name__ == '__main__':
    main()