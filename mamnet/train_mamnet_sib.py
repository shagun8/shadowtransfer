"""
Training script for MAMNet + SIB (Spectral Information Bottleneck).

Supports three modes:
  loco:   Leave-One-City-Out (train on 2 cities, test on holdout)
  all:    Train on all cities of a given resolution
  single: Train on a single data_root

Features:
  - All SIB components independently toggleable for ablation (M1-M14)
  - Module bypass gate: learned residual wrapping entire SIB pipeline
  - VIB warmup: linear ramp from 0→1 over first 10% of epochs
  - Boundary-tolerant evaluation (±Kpx don't-care zone)
  - Per-band KL loss tracking (LL / LH / HL / HH from Haar decomp)
  - Bypass gate α diagnostic logging (per-image gate values)
  - Comprehensive loss curves: total + task + per-band KL
  - Prediction image saving + per-image strict/tolerant metrics
  - Baseline comparison → comparison_results.json
  - Best/Worst prediction visualizations
  - Early stopping with patience

NEW — Diagnostic-motivated additions (§4.3 orphan coverage):
  - CACR: Class-Asymmetric Confidence Regularizer (training loss)
  - CE-AURC: Cross-entropy AURC auxiliary loss (training loss)
  - TENT: Test-time entropy minimization on BN affine params

Usage:
    python train_mamnet_sib.py \\
        --mode loco \\
        --base_data_root /path/to/data \\
        --resolution highres \\
        --fold_id 0 \\
        --use_haar --use_vib --use_content_aug --adaptive_beta \\
        --use_fda --fda_L 0.005 \\
        --use_contrast --use_sag \\
        --use_module_bypass \\
        --use_cacr --cacr_weight 0.1 \\
        --use_ce_aurc --ce_aurc_weight 0.01 \\
        --use_tent --tent_steps 1 \\
        --output_dir /path/to/output \\
        --eval_boundary_tolerant
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # headless — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_sib import build_mamnet_sib
from data.dataset_sib import ShadowDatasetSIB, get_dataloaders_sib
from data.dataset import LOCO_FOLDS
from utils.losses import MAMNetLoss, CACRLoss, CEAURCLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions

# TENT utilities (for test-time adaptation)
from models.sib import configure_tent, tent_adapt_step


# ════════════════════════════════════════════════════════════════════════════
# GPU diagnostics
# ════════════════════════════════════════════════════════════════════════════

print("=" * 50)
print("GPU DIAGNOSTICS")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("=" * 50)


# ════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Train MAMNet + SIB')

    # Mode
    p.add_argument('--mode', type=str, default='loco',
                   choices=['loco', 'all', 'single'])
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default='highres',
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=0)

    # Training
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--num_workers', type=int, default=1)
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--early_stopping_patience', type=int, default=15)
    p.add_argument('--device', type=str, default='cuda')

    # SIB components
    p.add_argument('--use_haar', action='store_true')
    p.add_argument('--use_vib', action='store_true')
    p.add_argument('--use_content_aug', action='store_true')
    p.add_argument('--adaptive_beta', action='store_true')
    p.add_argument('--use_sag', action='store_true')
    p.add_argument('--use_multiscale_sib', action='store_true')
    p.add_argument('--use_passthrough_gate', action='store_true')
    p.add_argument('--use_module_bypass', action='store_true')

    # SIB ablation flags (A3, A5, A9, A10)
    p.add_argument('--symmetric_vib', action='store_true',
                   help='A3: Use beta_content for edge VIB (instead of beta_edge)')
    p.add_argument('--aug_all_subbands', action='store_true',
                   help='A5: Also augment LH/HL/HH subbands (MRFP+ analog)')
    p.add_argument('--no_edge_vib', action='store_true',
                   help='A9: Skip VIB on LH/HL (pass through unchanged)')
    p.add_argument('--vib_wrong_subband', action='store_true',
                   help='A10: Apply content VIB to HL only (LL passes through)')

    # SIB hyperparameters
    p.add_argument('--beta_content', type=float, default=1e-3)
    p.add_argument('--beta_edge', type=float, default=1e-5)
    p.add_argument('--noise_scale', type=float, default=0.1)
    p.add_argument('--beta_max_multiplier', type=float, default=3.0)
    p.add_argument('--multiscale_beta_base', type=float, default=1e-4)
    p.add_argument('--vib_warmup_fraction', type=float, default=0.1)

    # ── NEW: Diagnostic-motivated modules ─────────────────────────────
    p.add_argument('--use_cacr', action='store_true',
                   help='Enable CACR (Class-Asymmetric Confidence Regularizer)')
    p.add_argument('--cacr_weight', type=float, default=0.1,
                   help='CACR loss weight (default: 0.1)')
    p.add_argument('--cacr_neg_weight', type=float, default=0.0,
                   help='CACR weight for pred-negative pixels (default: 0, unconstrained)')

    p.add_argument('--use_ce_aurc', action='store_true',
                   help='Enable CE-AURC auxiliary loss on gt_shadow pixels')
    p.add_argument('--ce_aurc_weight', type=float, default=0.01,
                   help='CE-AURC loss weight (default: 0.01)')
    p.add_argument('--ce_aurc_floor', type=float, default=0.5,
                   help='CE-AURC minimum weight for shadow pixels (default: 0.5)')

    p.add_argument('--use_tent', action='store_true',
                   help='Enable TENT (test-time entropy minimization on BN affine)')
    p.add_argument('--tent_steps', type=int, default=1,
                   help='TENT adaptation steps per batch (default: 1)')
    p.add_argument('--tent_lr', type=float, default=0.001,
                   help='TENT optimizer learning rate (default: 0.001)')
    p.add_argument('--tent_pred_pos_only', action='store_true', default=True,
                   help='TENT: minimize entropy only on pred-positive pixels (default: True)')

    # Data options
    p.add_argument('--use_contrast', action='store_true')
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_L', type=float, default=0.01)
    p.add_argument('--fda_target_root', type=str, default=None)

    # Comparison baselines
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--comparison_inference_dir', type=str, default=None,
                   help='Root of Test_img_results: .../loco/{city}/{res}/mamnet/vanilla/')
    p.add_argument('--comparison_data_root', type=str, default=None,
                   help='Base data root for GT masks (Final_data_test)')
    p.add_argument('--mamnet_output_dir', type=str, default=None,
                   help='MAMNet outputs root to scan for a completed DDIB donor '
                        'experiment (e.g. .../data/mamnet/outputs/)')
    p.add_argument('--boundary_tolerance', type=int, default=2)

    # §4.3 module: Class-conditional temperature scaling
    p.add_argument('--use_class_cond_tempscale', action='store_true',
                   help='Fit T_pos/T_neg on source-city val, apply at test')
    p.add_argument('--tempscale_max_iter', type=int, default=200)

    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Per-image metric functions
# ════════════════════════════════════════════════════════════════════════════

_KERNEL_CACHE = {}


def _tolerance_kernel(tol):
    if tol not in _KERNEL_CACHE:
        _KERNEL_CACHE[tol] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    return _KERNEL_CACHE[tol]


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


def _compute_tolerant_metrics(pred, gt, tolerance=2):
    kernel   = _tolerance_kernel(tolerance)
    gt_u8    = gt.astype(np.uint8)
    eroded   = cv2.erode(gt_u8, kernel)
    dilated  = cv2.dilate(gt_u8, kernel)
    valid    = ~((dilated - eroded) > 0)

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


def _average_metrics(lst):
    if not lst:
        return {k: 0.0 for k in
                ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
    keys = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
    return {k: float(np.mean([m[k] for m in lst])) for k in keys}


# ════════════════════════════════════════════════════════════════════════════
# §4.3 MODULE: Class-Conditional Temperature Scaling
#   Fit T_pos and T_neg on source-city validation logits (no target labels),
#   apply at inference on held-out city. Reduces SP-gap without target data.
#   Reference: Guo et al. (ICML 2017), Tian et al. (CVPR 2023).
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    """Run inference, return concatenated CPU logits + labels + filenames."""
    model.eval()
    all_logits, all_labels, all_filenames = [], [], []
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)
        outputs, _ = model(images, intensity_map=intensity_map)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs['main']
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_filenames.extend(batch['filename'])
    return (torch.cat(all_logits, dim=0),
            torch.cat(all_labels, dim=0),
            all_filenames)


def fit_class_conditional_temperature(logits, labels, max_iter=200):
    """
    Fit (T_pos, T_neg) on validation logits via LBFGS minimizing NLL.
    Per-channel scaling: channel 0 / T_neg, channel 1 / T_pos.
    Both temperatures parameterized as exp(log_T) to keep them positive.

    Args:
        logits: [N, 2, H, W] tensor on any device.
        labels: [N, H, W] long tensor on the same device.

    Returns: (T_pos, T_neg) as Python floats.
    """
    device = logits.device

    log_T_pos = torch.nn.Parameter(torch.zeros(1, device=device))
    log_T_neg = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_T_pos, log_T_neg],
                                   lr=0.1, max_iter=max_iter,
                                   line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        T_pos = log_T_pos.exp()
        T_neg = log_T_neg.exp()
        scaled = torch.stack([
            logits[:, 0, :, :] / T_neg,
            logits[:, 1, :, :] / T_pos,
        ], dim=1)
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T_pos.exp().item()), float(log_T_neg.exp().item())


def apply_tempscale(logits, T_pos, T_neg):
    """
    Per-channel temperature scaling for binary segmentation.
    Channel 0 (background) is divided by T_neg.
    Channel 1 (foreground) is divided by T_pos.

    Args:
        logits: [B, 2, H, W] tensor.
        T_pos, T_neg: Python floats.

    Returns: [B, 2, H, W] rescaled logits.
    """
    return torch.stack([
        logits[:, 0, :, :] / T_neg,
        logits[:, 1, :, :] / T_pos,
    ], dim=1)


def compute_aurc(confidences, correct, n_coverage=20):
    """
    Area Under Risk-Coverage curve.
    Sort by confidence (desc), error rate at coverage c uses top c-fraction.
    """
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    if len(confidences) == 0:
        return float('nan')
    sort_idx = np.argsort(-confidences)
    sorted_correct = correct[sort_idx]
    coverage_grid = np.linspace(0.10, 1.0, n_coverage)
    aurc = 0.0
    for c in coverage_grid:
        k = max(1, int(c * len(confidences)))
        aurc += 1.0 - sorted_correct[:k].mean()
    return aurc / n_coverage


def compute_ece(confidences, correct, n_bins=15):
    """Expected Calibration Error."""
    confidences = np.asarray(confidences)
    correct = np.asarray(correct).astype(float)
    if len(confidences) == 0:
        return float('nan')
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(confidences)
    for i in range(n_bins):
        m = (confidences > edges[i]) & (confidences <= edges[i+1])
        if m.sum() > 0:
            ece += (m.sum() / n) * abs(confidences[m].mean() - correct[m].mean())
    return ece


def compute_sp_metrics(logits, labels):
    """
    SP-gap (foreground class), background AURC, and class-stratified ECE.
    All values are computed on the full pixel population.
    """
    probs = F.softmax(logits, dim=1)
    shadow_prob = probs[:, 1, :, :]                         # [N, H, W]
    preds = logits.argmax(dim=1)                            # [N, H, W]

    gt_shadow = (labels == 1)
    gt_bg = (labels == 0)
    pred_pos = (preds == 1)
    pred_neg = (preds == 0)

    s_conf = shadow_prob[gt_shadow].numpy()
    s_correct = (preds[gt_shadow] == 1).numpy()
    b_conf = (1.0 - shadow_prob)[gt_bg].numpy()
    b_correct = (preds[gt_bg] == 0).numpy()

    pp_conf = shadow_prob[pred_pos].numpy() if pred_pos.any() else np.array([])
    pp_correct = (labels[pred_pos] == 1).numpy() if pred_pos.any() else np.array([])
    pn_conf = (1.0 - shadow_prob)[pred_neg].numpy() if pred_neg.any() else np.array([])
    pn_correct = (labels[pred_neg] == 0).numpy() if pred_neg.any() else np.array([])

    return {
        'aurc_shadow':   float(compute_aurc(s_conf, s_correct)),
        'aurc_bg':       float(compute_aurc(b_conf, b_correct)),
        'ece_pred_pos':  float(compute_ece(pp_conf, pp_correct)),
        'ece_pred_neg':  float(compute_ece(pn_conf, pn_correct)),
        'n_gt_shadow':   int(gt_shadow.sum()),
        'n_gt_bg':       int(gt_bg.sum()),
    }


# ════════════════════════════════════════════════════════════════════════════
# VIB warmup
# ════════════════════════════════════════════════════════════════════════════

def vib_warmup_weight(epoch, total_epochs, warmup_fraction=0.1):
    warmup_epochs = max(1, int(total_epochs * warmup_fraction))
    if epoch < warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    return 1.0


# ════════════════════════════════════════════════════════════════════════════
# Training loop  — tracks per-band KL losses + bypass alpha
#                  + NEW: CACR and CE-AURC losses
# ════════════════════════════════════════════════════════════════════════════

# Keys that are aggregate sums (not leaf-level per-band losses)
_AGGREGATE_KEYS = {'kl_total', 'kl_multiscale'}
# Keys that are diagnostic tensors, not scalar losses
_NON_LOSS_KEYS = {'bypass_alpha', 'ref_logits', 'pre_aug_bottleneck'}


def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    epoch, total_epochs, vib_warmup_frac=0.1,
                    cacr_criterion=None, cacr_weight=0.0,
                    ce_aurc_criterion=None, ce_aurc_weight=0.0):
    """
    Train one epoch with optional CACR and CE-AURC losses.

    Args:
        model, dataloader, optimizer, criterion, device: standard.
        epoch, total_epochs, vib_warmup_frac: VIB warmup schedule.
        cacr_criterion: CACRLoss instance (None to disable).
        cacr_weight: Weight for CACR loss term.
        ce_aurc_criterion: CEAURCLoss instance (None to disable).
        ce_aurc_weight: Weight for CE-AURC loss term.
    """
    model.train()

    total_loss      = 0.0
    total_task_loss = 0.0
    total_kl_loss   = 0.0
    total_cacr_loss = 0.0
    total_ce_aurc_loss = 0.0
    band_kl_accum   = defaultdict(float)   # per-band KL accumulator
    cacr_diag_accum = defaultdict(float)   # CACR diagnostic accumulator
    ce_aurc_diag_accum = defaultdict(float)
    n_batches       = 0
    metrics         = ShadowMetrics()

    vib_w = vib_warmup_weight(epoch, total_epochs, vib_warmup_frac)

    for batch in dataloader:
        images        = batch['image'].to(device)
        labels        = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        optimizer.zero_grad()

        outputs, sib_losses = model(images, intensity_map=intensity_map)

        # Task loss
        task_losses = criterion(outputs, labels)
        task_loss   = task_losses['total']

        # Accumulate per-band KL for history tracking (leaf terms only)
        for band_key, band_val in sib_losses.items():
            if isinstance(band_val, torch.Tensor) \
                    and band_key not in _AGGREGATE_KEYS \
                    and band_key not in _NON_LOSS_KEYS:
                if band_val.dim() == 0:
                    band_kl_accum[band_key] += band_val.item()

        # Track bypass gate alpha mean (diagnostic, not a loss)
        if 'bypass_alpha' in sib_losses:
            band_kl_accum['bypass_alpha_mean'] += \
                sib_losses['bypass_alpha'].detach().mean().item()

        # KL for backprop: use pre-summed kl_total + optional multiscale
        kl_loss = sib_losses.get('kl_total', torch.tensor(0.0, device=device))
        if 'kl_multiscale' in sib_losses:
            kl_loss = kl_loss + sib_losses['kl_multiscale']
            band_kl_accum['kl_multiscale'] += sib_losses['kl_multiscale'].item()

        # ── NEW: CACR loss ─────────────────────────────────────────────
        cacr_loss_val = torch.tensor(0.0, device=device)
        if cacr_criterion is not None and 'ref_logits' in sib_losses:
            main_logits = outputs['main'] if isinstance(outputs, dict) else outputs
            ref_logits = sib_losses['ref_logits'].detach()
            cacr_loss_val, cacr_diag = cacr_criterion(
                main_logits, ref_logits, targets=labels)
            for k, v in cacr_diag.items():
                cacr_diag_accum[k] += v

        # ── NEW: CE-AURC auxiliary loss ────────────────────────────────
        ce_aurc_loss_val = torch.tensor(0.0, device=device)
        if ce_aurc_criterion is not None:
            main_logits = outputs['main'] if isinstance(outputs, dict) else outputs
            ce_aurc_loss_val, aurc_diag = ce_aurc_criterion(
                main_logits, labels)
            for k, v in aurc_diag.items():
                ce_aurc_diag_accum[k] += v

        # ── Total loss ─────────────────────────────────────────────────
        loss = (task_loss
                + vib_w * kl_loss
                + cacr_weight * cacr_loss_val
                + ce_aurc_weight * ce_aurc_loss_val)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            filtered = filter_small_predictions(outputs['main'], min_pixels=10)
            metrics.update(filtered, labels)

        total_loss         += loss.item()
        total_task_loss    += task_loss.item()
        total_kl_loss      += kl_loss.item()
        total_cacr_loss    += cacr_loss_val.item()
        total_ce_aurc_loss += ce_aurc_loss_val.item()
        n_batches          += 1

    m   = metrics.compute()
    nb  = max(n_batches, 1)
    avg_band_kl = {k: v / nb for k, v in band_kl_accum.items()}

    result = {
        'total':      total_loss / nb,
        'task':       total_task_loss / nb,
        'kl':         total_kl_loss / nb,
        'band_kl':    avg_band_kl,
        'vib_weight': vib_w,
        'metrics':    m,
    }

    # CACR diagnostics
    if cacr_criterion is not None:
        result['cacr'] = total_cacr_loss / nb
        result['cacr_diag'] = {k: v / nb for k, v in cacr_diag_accum.items()}

    # CE-AURC diagnostics
    if ce_aurc_criterion is not None:
        result['ce_aurc'] = total_ce_aurc_loss / nb
        result['ce_aurc_diag'] = {k: v / nb for k, v in ce_aurc_diag_accum.items()}

    return result


# ════════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, dataloader, criterion, device, boundary_tolerant=False, tolerance=5):
    model.eval()
    metrics         = ShadowMetrics()
    all_preds_np    = []
    all_labels_np   = []
    total_loss      = 0.0
    n_batches       = 0

    for batch in dataloader:
        images        = batch['image'].to(device)
        labels        = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        outputs, _ = model(images, intensity_map=intensity_map)
        logits     = outputs if isinstance(outputs, torch.Tensor) else outputs['main']

        total_loss += criterion.criterion(logits, labels).item()

        filtered = filter_small_predictions(logits, min_pixels=10)
        metrics.update(filtered, labels)

        preds_np = filtered.argmax(dim=1).cpu().numpy()
        for i in range(preds_np.shape[0]):
            all_preds_np.append(preds_np[i])
            all_labels_np.append(labels[i].cpu().numpy())

        n_batches += 1

    val_loss = total_loss / max(n_batches, 1)
    m        = metrics.compute()

    strict = _average_metrics([
        _compute_strict_metrics(p, g)
        for p, g in zip(all_preds_np, all_labels_np)
    ])

    result = {
        'loss':           val_loss,
        'shadow_metrics': m,
        'strict':         strict,
    }

    if boundary_tolerant:
        tol_list = [
            _compute_tolerant_metrics(p, g, tolerance=tolerance)
            for p, g in zip(all_preds_np, all_labels_np)
        ]
        result[f'tolerant_{tolerance}px'] = _average_metrics(tol_list)

    return result


# ════════════════════════════════════════════════════════════════════════════
# Test + save predictions + bypass alpha diagnostics
#   + NEW: TENT (test-time entropy minimization)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test_and_save_predictions(model, loader, device, args, output_dir):
    """
    Run test inference, save prediction PNGs, compute per-image metrics.
    If module bypass gate is active, also logs per-image α values to
    bypass_gate_alpha.json.

    NEW: If --use_tent is enabled, applies test-time entropy minimization
    on BN affine parameters before making predictions.

    Returns averaged strict + tolerant metrics plus per-image lists.
    """
    model.eval()
    pred_save_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_save_dir, exist_ok=True)

    strict_list   = []
    tolerant_list = []
    all_filenames = []
    all_alphas    = []   # per-image bypass gate α (if module bypass active)

    # ── TENT setup (optional) ─────────────────────────────────────────
    tent_active = getattr(args, 'use_tent', False)
    tent_params = None
    tent_optimizer = None
    norm_layers = None

    if tent_active:
        print(f'  TENT enabled: steps={args.tent_steps}, lr={args.tent_lr}, '
              f'pred_pos_only={args.tent_pred_pos_only}')
        tent_params, norm_layers = configure_tent(
            model, use_bn=True, use_ln=False)
        if tent_params:
            tent_optimizer = torch.optim.SGD(
                tent_params, lr=args.tent_lr, momentum=0.9)
            print(f'  TENT: {len(tent_params)} params to adapt '
                  f'from {len(norm_layers)} BN layers')
        else:
            print(f'  TENT: no BN layers found — disabling.')
            tent_active = False

    for batch_idx, batch in enumerate(loader):
        images        = batch['image'].to(device)
        masks         = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        # ── TENT adaptation steps ─────────────────────────────────
        if tent_active and tent_optimizer is not None:
            if images.size(0) < 2:
                # Trailing partial batch — TENT can't adapt because
                # post-GAP BN layers in MSCAF need batch_size ≥ 2.
                # Predictions happen in eval mode below using already-adapted
                # affine params from previous batches.
                pass
            else:
                for _ in range(args.tent_steps):
                    tent_adapt_step(
                        model, images, intensity_map,
                        tent_optimizer, norm_layers,
                        pred_pos_only=args.tent_pred_pos_only)
            # After adaptation (or skip), set norm layers back to eval
            model.eval()

        with torch.no_grad():
            outputs, sib_losses = model(images, intensity_map=intensity_map)
            logits     = outputs if isinstance(outputs, torch.Tensor) else outputs['main']
            filtered   = filter_small_predictions(logits, min_pixels=10)
            preds      = filtered.argmax(dim=1)   # (B, H, W)

        # Collect bypass alpha if available
        if 'bypass_alpha' in sib_losses:
            alpha_batch = sib_losses['bypass_alpha'].cpu().numpy()
            for j in range(alpha_batch.shape[0]):
                all_alphas.append(float(alpha_batch[j]))

        for i, fname in enumerate(batch['filename']):
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            gt_np   = masks[i].cpu().numpy().astype(np.uint8)

            Image.fromarray(pred_np * 255).save(
                os.path.join(pred_save_dir, fname))

            strict_list.append(_compute_strict_metrics(pred_np, gt_np))
            tolerant_list.append(
                _compute_tolerant_metrics(pred_np, gt_np, tolerance=args.boundary_tolerance))
            all_filenames.append(fname)

        if (batch_idx + 1) % 20 == 0:
            print(f'  [test] {len(all_filenames)} images done...')

    strict   = _average_metrics(strict_list)
    tolerant = _average_metrics(tolerant_list)

    print(f'\nTest Results ({len(all_filenames)} images):')
    print(f'  Strict  : OA={strict["OA"]:.2f}  P={strict["Precision"]:.2f}  '
          f'R={strict["Recall"]:.2f}  F1={strict["F1"]:.2f}  '
          f'BER={strict["BER"]:.2f}  mIOU={strict["mIOU"]:.2f}  '
          f'ShIOU={strict["Shadow_IOU"]:.2f}')
    print(f'  Tolerant: OA={tolerant["OA"]:.2f}  P={tolerant["Precision"]:.2f}  '
          f'R={tolerant["Recall"]:.2f}  F1={tolerant["F1"]:.2f}  '
          f'BER={tolerant["BER"]:.2f}  mIOU={tolerant["mIOU"]:.2f}  '
          f'ShIOU={tolerant["Shadow_IOU"]:.2f}')
    if tent_active:
        print(f'  TENT: active ({args.tent_steps} steps/batch)')

    # Save bypass gate alpha diagnostics
    if all_alphas:
        alpha_data = {
            'description': 'Per-image module bypass gate alpha values. '
                           'alpha=1 means full SIB applied, alpha=0 means SIB bypassed.',
            'num_images': len(all_alphas),
            'mean_alpha': float(np.mean(all_alphas)),
            'std_alpha': float(np.std(all_alphas)),
            'min_alpha': float(np.min(all_alphas)),
            'max_alpha': float(np.max(all_alphas)),
            'median_alpha': float(np.median(all_alphas)),
            'per_image': {fn: alpha for fn, alpha in zip(all_filenames, all_alphas)},
        }
        alpha_path = os.path.join(output_dir, 'bypass_gate_alpha.json')
        with open(alpha_path, 'w') as f:
            json.dump(alpha_data, f, indent=4)
        print(f'  Bypass gate α: mean={alpha_data["mean_alpha"]:.4f}  '
              f'std={alpha_data["std_alpha"]:.4f}  '
              f'range=[{alpha_data["min_alpha"]:.4f}, {alpha_data["max_alpha"]:.4f}]')
        print(f'  Saved → {alpha_path}')

    test_results = {
        'num_images':   len(all_filenames),
        'strict':       strict,
        f'tolerant_{args.boundary_tolerance}px': tolerant,
        'tent_active':  tent_active,
    }
    # Include alpha summary in test_results if available
    if all_alphas:
        test_results['bypass_alpha_summary'] = {
            'mean': float(np.mean(all_alphas)),
            'std': float(np.std(all_alphas)),
            'min': float(np.min(all_alphas)),
            'max': float(np.max(all_alphas)),
        }
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

    return (strict, tolerant, strict_list, tolerant_list, all_filenames)


# ════════════════════════════════════════════════════════════════════════════
# Baseline discovery + comparison
#
# Two strategies, tried in the order below.  Both can succeed simultaneously.
#
#   Strategy 1 — DDIB donor  (fast path, optional)
#     Look for a completed DDIB comparison_results.json in
#     --mamnet_output_dir.  If found, its pre-computed baseline metrics
#     are used directly — no image re-scanning needed.
#     The DDIB model's own metrics also become an extra comparison row.
#
#   Strategy 2 — Raw prediction dirs  (always attempted)
#     Use --comparison_inference_dir as the Test_img_results root:
#       {root}/loco/{city}/{res}/mamnet/vanilla/
#       {root}/loco/{city}/{res}/mamnet/fda/
#       {root}/loco/{city}/{res}/mamnet/segdesic/
#       {root}/loco/{city}/{res}/mamnet/iim/
#       {root}/loco/{city}/{res}/mamnet/isw/
#       {root}/loco/{city}/{res}/mamnet/mrfp_plus/
#       {root}/loco/{city}/{res}/mamnet/fada/
#       {root}/upper/{city}/{res}/mamnet/base/
#     Per-image metrics computed from scratch against GT masks.
#
# Every path checked is printed so failures are immediately obvious.
# ════════════════════════════════════════════════════════════════════════════

def _dir_has_images(path):
    """Return count of prediction images directly inside path (0 if absent)."""
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path)
               if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff', '.jpeg')))


# ── Strategy 1: DDIB donor ────────────────────────────────────────────────

def _find_ddib_donor(args):
    """
    Look for a completed DDIB experiment in args.mamnet_output_dir.

    Returns (baselines_dict, ddib_self_metrics) if a valid donor is found,
    otherwise (None, None).

    A valid donor must have comparison_results.json with at least one
    baseline entry whose strict F1 > 0.
    """
    output_root = getattr(args, 'mamnet_output_dir', None)
    test_city   = getattr(args, 'test_city', '')
    res         = args.resolution

    if not output_root:
        print('  [S1] --mamnet_output_dir not set; skipping DDIB donor search.')
        return None, None
    if not os.path.isdir(output_root):
        print(f'  [S1] mamnet_output_dir not found: {output_root}')
        return None, None

    print(f'\n  [S1] Scanning for DDIB donor in: {output_root}')
    candidates = []
    for entry in os.listdir(output_root):
        el = entry.lower()
        if 'ddib' not in el:
            continue
        if test_city not in el or res not in el:
            continue
        comp_path = os.path.join(output_root, entry, 'comparison_results.json')
        if os.path.isfile(comp_path):
            candidates.append((entry, comp_path))

    if not candidates:
        print(f'       No DDIB experiments found for city={test_city} res={res}.')
        return None, None

    # Most-recently modified first
    candidates.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)

    for entry, comp_path in candidates:
        print(f'       Trying: {entry}')
        try:
            with open(comp_path) as f:
                data = json.load(f)
            baselines = data.get('baselines', {})
            valid = {k: v for k, v in baselines.items()
                     if isinstance(v, dict) and v.get('strict', {}).get('F1', 0) > 0}
            if valid:
                ddib_self = data.get('ddib', data.get('sib', {}))
                print(f'       ✓  Donor {entry}: {len(valid)} baseline(s) + DDIB self metrics')
                return valid, ddib_self
            else:
                print(f'       ~  {entry}: comparison_results.json has empty/zero baselines')
        except (json.JSONDecodeError, OSError) as e:
            print(f'       ~  {entry}: could not load — {e}')

    print(f'       No valid DDIB donor found.')
    return None, None


# ── Strategy 2: Raw prediction dirs ──────────────────────────────────────

def _find_raw_prediction_dirs(args):
    """
    Locate raw prediction dirs under the Test_img_results root
    (args.comparison_inference_dir) using the canonical path structure:

        {root}/loco/{city}/{res}/mamnet/{vanilla,fda,segdesic,iim,isw,mrfp_plus,fada}/
        {root}/upper/{city}/{res}/mamnet/base/

    Returns a dict  label → pred_dir  for every dir that exists and
    contains at least one image.
    """
    inf_root  = getattr(args, 'comparison_inference_dir', None)
    test_city = getattr(args, 'test_city', '')
    res       = args.resolution

    if not inf_root:
        print('  [S2] --comparison_inference_dir not set; skipping raw prediction lookup.')
        return {}

    raw_dirs = {
        'Upper Bound':   os.path.join(inf_root, 'upper', test_city, res, 'mamnet', 'base'),
        'LOCO Vanilla':  os.path.join(inf_root, 'loco',  test_city, res, 'mamnet', 'vanilla'),
        'LOCO FDA':      os.path.join(inf_root, 'loco',  test_city, res, 'mamnet', 'fda'),
        'LOCO SegDesic': os.path.join(inf_root, 'loco',  test_city, res, 'mamnet', 'segdesic'),
        'LOCO IIM':      os.path.join(inf_root, 'loco',  test_city, res, 'mamnet', 'iim'),
        'LOCO ISW':      os.path.join(inf_root, 'loco',  test_city, res, 'mamnet', 'isw'),
        'LOCO MRFP+':    os.path.join(inf_root, 'loco',  test_city, res, 'mamnet', 'mrfp_plus'),
        'LOCO FADA':     os.path.join(inf_root, 'loco',  test_city, res, 'mamnet', 'fada'),
    }

    print(f'\n  [S2] Checking raw prediction dirs under: {inf_root}')
    found = {}
    for label, pred_dir in raw_dirs.items():
        n = _dir_has_images(pred_dir)
        if n > 0:
            found[label] = pred_dir
            print(f'       ✓  {label:<16} {pred_dir}  ({n} images)')
        else:
            reason = 'dir missing' if not os.path.isdir(pred_dir) else 'no images'
            print(f'       ✗  {label:<16} {pred_dir}  ({reason})')

    return found


# ── Compute metrics from a raw prediction directory ───────────────────────

def _baseline_metrics_from_predictions(pred_dir, gt_dir, filenames, img_size, tolerance=5):
    """
    Compute strict + tolerant per-image metrics for a baseline prediction dir.

    Matching strategy (priority order):
      1. Exact filename match  (pred_dir/foo.png  ↔  gt_dir/foo.png)
      2. Stem match ignoring extension

    Args:
        pred_dir  : directory containing prediction images (0/255 binary PNGs)
        gt_dir    : directory containing GT mask images   (0/255 binary PNGs)
        filenames : list of test-set filenames used to restrict scoring to the
                    current test split; stems are used for matching.
                    Pass [] to accept all files in pred_dir.
        img_size  : resize side — predictions and GT both resized to img_size²

    Returns:
        dict with keys strict, tolerant_Kpx, n_images, strict_list,
        tolerant_list  — or None if no matching pairs were found.
    """
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def _stem_map(directory):
        """Return {stem: full_path} for all images in directory."""
        m = {}
        if not os.path.isdir(directory):
            return m
        for fn in os.listdir(directory):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                stem = os.path.splitext(fn)[0]
                m[stem] = os.path.join(directory, fn)
        return m

    pred_map = _stem_map(pred_dir)
    gt_map   = _stem_map(gt_dir)

    if not pred_map or not gt_map:
        return None

    # Build allowed-stem set from the test-set filenames (empty = accept all)
    allowed_stems = set()
    if filenames:
        for fn in filenames:
            allowed_stems.add(os.path.splitext(fn)[0])

    # Find matching (pred, gt) pairs restricted to the test split
    pairs = []
    for stem, pred_path in pred_map.items():
        if allowed_stems and stem not in allowed_stems:
            continue
        if stem in gt_map:
            pairs.append((pred_path, gt_map[stem]))

    if not pairs:
        return None

    strict_list   = []
    tolerant_list = []
    sz = (img_size, img_size)

    for pred_path, gt_path in pairs:
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
        if pred_img is None or gt_img is None:
            continue

        # Resize to evaluation size if needed
        if pred_img.shape != (img_size, img_size):
            pred_img = cv2.resize(pred_img, sz, interpolation=cv2.INTER_NEAREST)
        if gt_img.shape != (img_size, img_size):
            gt_img   = cv2.resize(gt_img,   sz, interpolation=cv2.INTER_NEAREST)

        # Binarise: >127 → shadow (1), else background (0)
        pred_bin = (pred_img > 127).astype(np.uint8)
        gt_bin   = (gt_img   > 127).astype(np.uint8)

        strict_list.append(_compute_strict_metrics(pred_bin, gt_bin))
        tolerant_list.append(
            _compute_tolerant_metrics(pred_bin, gt_bin, tolerance=tolerance))

    if not strict_list:
        return None

    tol_key_local = f'tolerant_{tolerance}px'
    return {
        'strict':        _average_metrics(strict_list),
        tol_key_local:   _average_metrics(tolerant_list) if tolerant_list else
                         _average_metrics([]),
        'n_images':      len(strict_list),
        'strict_list':   strict_list,
        'tolerant_list': tolerant_list,
    }


# ── Main comparison entry point ───────────────────────────────────────────

def compare_with_baselines(strict, tolerant, strict_list, tolerant_list,
                           filenames, args, output_dir):
    """
    Run both strategies, build the full baseline_results dict, print
    comparison tables, and save comparison_results.json.

    Strategy 1 (DDIB donor): load pre-computed baseline metrics from a
    completed DDIB experiment in --mamnet_output_dir.  Fast, no images needed.

    Strategy 2 (raw pred dirs): compute metrics from scratch from
    --comparison_inference_dir prediction images vs GT masks.
    Strategy 2 overwrites Strategy 1 values when both succeed for the
    same label (fresh computation is preferred).

    The saved JSON always has:
      sib       → SIB (this run) metrics
      ddib      → DDIB metrics (from donor, if found; else same as sib)
      baselines → {Upper Bound, LOCO Vanilla, LOCO FDA, LOCO SegDesic,
                   LOCO IIM, LOCO ISW, LOCO MRFP+, LOCO FADA}
    """
    test_city = getattr(args, 'test_city', 'unknown')
    res       = args.resolution
    img_size  = args.img_size
    tol_key   = f'tolerant_{args.boundary_tolerance}px'

    # ── Resolve GT mask directory ─────────────────────────────────────────
    gt_dir = None
    gt_candidates = []
    if args.comparison_data_root:
        gt_candidates = [
            os.path.join(args.comparison_data_root, test_city, res, 'test', 'masks'),
            os.path.join(args.comparison_data_root, test_city, res, 'masks'),
            os.path.join(args.comparison_data_root, 'test', 'masks'),
            os.path.join(args.comparison_data_root, 'masks'),
        ]
        for c in gt_candidates:
            if os.path.isdir(c):
                gt_dir = c
                break

    print(f'\n{"="*70}')
    print(f'BASELINE COMPARISON')
    print(f'  Test city : {test_city}  |  Resolution: {res}')
    print(f'  GT masks  : {gt_dir}')
    if gt_dir is None and gt_candidates:
        print(f'  ⚠  GT mask dir not found. Tried:')
        for c in gt_candidates:
            print(f'       {c}')
    print(f'{"="*70}')

    baseline_results  = {}
    ddib_self_metrics = None

    # ────────────────────────────────────────────────────────────────────
    # Strategy 1 — DDIB donor
    # ────────────────────────────────────────────────────────────────────
    donor_baselines, ddib_self = _find_ddib_donor(args)

    if donor_baselines:
        for label, bl_data in donor_baselines.items():
            if isinstance(bl_data, dict) and 'strict' in bl_data:
                baseline_results[label] = bl_data
        if ddib_self and ddib_self.get('strict', {}).get('F1', 0) > 0:
            ddib_self_metrics = ddib_self
        print(f'\n  S1: {len(baseline_results)} baseline(s) loaded from DDIB donor.')
        print(f'  S1 baselines: {list(baseline_results.keys())}')
    else:
        print('\n  S1: No DDIB donor found — proceeding with S2 only.')

    # ────────────────────────────────────────────────────────────────────
    # Strategy 2 — Raw prediction dirs
    # ────────────────────────────────────────────────────────────────────
    raw_pred_dirs = _find_raw_prediction_dirs(args)

    if raw_pred_dirs and gt_dir is None:
        print('\n  S2: raw prediction dirs found but GT dir is missing — '
              'cannot compute metrics from images.')
        print('      Pass --comparison_data_root pointing at Final_data_test '
              'to enable per-image metric computation.')
    elif raw_pred_dirs:
        print(f'\n  S2: computing metrics for {len(raw_pred_dirs)} baseline(s)...')
        for label, pred_dir in raw_pred_dirs.items():
            bl = _baseline_metrics_from_predictions(
                pred_dir, gt_dir, filenames, img_size,
                tolerance=args.boundary_tolerance)
            if bl:
                baseline_results[label] = bl
                # Defensive lookup: never crash the print if a tol-key
                # mismatch slips through (e.g. donor data with different tol).
                tol_metrics = bl.get(tol_key, {})
                tol_f1 = tol_metrics.get('F1', 0.0)
                strict_f1 = bl.get('strict', {}).get('F1', 0.0)
                print(f'  S2: ✓ {label} — {bl["n_images"]} images '
                      f'(strict F1={strict_f1:.2f}  '
                      f'tol F1={tol_f1:.2f})')
            else:
                print(f'  S2: ✗ {label} — no matching image pairs found '
                      f'(pred={pred_dir}  gt={gt_dir})')

    # ────────────────────────────────────────────────────────────────────
    # Print tables
    # ────────────────────────────────────────────────────────────────────
    if baseline_results:
        _print_comparison_table(
            'STRICT METRICS (all pixels)',
            baseline_results, strict, tolerant,
            ddib_self_metrics)
        _print_comparison_table(
            f'TOLERANT METRICS (±{args.boundary_tolerance} px dont-care zone)',
            baseline_results, strict, tolerant,
            ddib_self_metrics, metric_type=tol_key)
        _print_recovery_ratios(baseline_results, strict, tolerant, tol_key=tol_key)
        for bl_label in ['LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
                         'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA']:
            if (bl_label in baseline_results
                    and 'strict_list' in baseline_results[bl_label]):
                _print_bootstrap_comparison(
                    baseline_results[bl_label],
                    strict_list, tolerant_list,
                    baseline_label=bl_label)
    else:
        print('\n  ⚠  No baselines available.  Check:')
        print('       --comparison_inference_dir  (Test_img_results root)')
        print('       --mamnet_output_dir         (for DDIB donor)')
        print('       --comparison_data_root      (GT masks root)')

    # ────────────────────────────────────────────────────────────────────
    # Save comparison_results.json
    # ────────────────────────────────────────────────────────────────────
    comp = {
        'test_city':  test_city,
        'resolution': res,
        'eval_size':  img_size,
        'sib':  {'strict': strict, tol_key: tolerant},
        'ddib': ddib_self_metrics if ddib_self_metrics else
                {'strict': strict, tol_key: tolerant},
        'baselines': {},
    }
    for label, br in baseline_results.items():
        comp['baselines'][label] = {
            'strict':  br.get('strict', {}),
            tol_key:   br.get(tol_key, br.get('tolerant', {})),
        }
        if 'n_images' in br:
            comp['baselines'][label]['n_images'] = br['n_images']

    comp_path = os.path.join(output_dir, 'comparison_results.json')
    with open(comp_path, 'w') as f:
        json.dump(comp, f, indent=4)
    print(f'\nComparison results saved → {comp_path}')
    return comp


def _print_comparison_table(title, baseline_results, sib_strict, sib_tolerant,
                             ddib_self_metrics=None, metric_type='strict'):
    """
    Print a comparison table.

    Rows (in order):
      Upper Bound
      LOCO Vanilla, LOCO FDA, LOCO SegDesic
      LOCO IIM, LOCO ISW, LOCO MRFP+, LOCO FADA
      DDIB (only if ddib_self_metrics is provided)
      SIB (ours)
    """
    sib_m = sib_strict if metric_type == 'strict' else sib_tolerant

    print(f'\n{"-"*70}')
    print(f'{title:^70}')
    print(f'{"-"*70}')
    print(f'  {"Method":<22} {"OA":>6} {"Prec":>6} {"Rec":>6} '
          f'{"F1":>6} {"BER":>6} {"mIOU":>6} {"ShIOU":>6}')
    print('  ' + '-' * 64)

    def _row(label, m):
        if not m:
            return
        print(f'  {label:<22} {m.get("OA",0):6.2f} {m.get("Precision",0):6.2f} '
              f'{m.get("Recall",0):6.2f} {m.get("F1",0):6.2f} '
              f'{m.get("BER",0):6.2f} {m.get("mIOU",0):6.2f} '
              f'{m.get("Shadow_IOU",0):6.2f}')

    for label in ['Upper Bound', 'LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
                  'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA']:
        if label not in baseline_results:
            continue
        _row(label, baseline_results[label].get(metric_type, {}))

    if ddib_self_metrics:
        ddib_m = ddib_self_metrics.get(metric_type,
                 ddib_self_metrics.get(metric_type if metric_type == 'strict'
                                       else 'strict', {}))
        if isinstance(ddib_m, dict) and 'strict' in ddib_m:
            ddib_m = ddib_m.get(metric_type, {})
        _row('DDIB', ddib_m)

    print('  ' + '-' * 64)
    _row('SIB (ours)', sib_m)


def _print_recovery_ratios(baseline_results, sib_strict, sib_tolerant, tol_key='tolerant_2px'):
    if ('Upper Bound' not in baseline_results
            or 'LOCO Vanilla' not in baseline_results):
        return
    print(f'\n{"-"*70}')
    print(f'{"RECOVERY RATIOS":^70}')
    print(f'  R = (SIB − LOCO_Vanilla) / (Upper − LOCO_Vanilla)')
    print(f'{"-"*70}')
    for eval_type, sib_m, label in [
            ('strict', sib_strict, 'Strict'),
            (tol_key, sib_tolerant, 'Tolerant')]:
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
        print(f'  {label:<10}  ' + '  '.join(parts))


def _print_bootstrap_comparison(loco_bl, sib_sl, sib_tl,
                                 baseline_label, n_bootstrap=5000):
    print(f'\n{"-"*70}')
    print(f'{"BOOTSTRAP: SIB vs " + baseline_label + " (n=5000)":^70}')
    print(f'{"-"*70}')
    np.random.seed(42)
    for eval_type, sib_list, label in [
            ('strict_list', sib_sl, 'Strict'),
            ('tolerant_list', sib_tl, 'Tolerant')]:
        loco_list = loco_bl.get(eval_type, [])
        n = min(len(loco_list), len(sib_list))
        if n == 0:
            continue
        print(f'\n  {label}:')
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
            sig = (' ***' if p_val < 0.001 else ' **' if p_val < 0.01
                   else ' *' if p_val < 0.05 else '')
            print(f'    {k:<12} delta={obs_mean:+.2f}  '
                  f'95%CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  p={p_val:.4f}{sig}')
    print('')


# ════════════════════════════════════════════════════════════════════════════
# Comprehensive loss plotting
# All losses on one figure: total, task, kl_total, + per-band KL from Haar
# + NEW: CACR and CE-AURC loss curves
# ════════════════════════════════════════════════════════════════════════════

matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman'] + matplotlib.rcParams['font.serif'],
    'font.size':          10,
    'axes.titlesize':     11,
    'axes.labelsize':     10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'figure.titlesize':   13,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})

_BAND_DISPLAY = {
    'kl_ll':              'KL — LL (content)',
    'kl_lh':              'KL — LH (h-edge)',
    'kl_hl':              'KL — HL (v-edge)',
    'kl_hh':              'KL — HH (noise)',
    'kl_content':         'KL — Content (LL)',
    'kl_edge_lh':         'KL — Edge LH',
    'kl_edge_hl':         'KL — Edge HL',
    'kl_noise':           'KL — Noise (HH)',
    'kl_multiscale':      'KL — MultiScale',
    'kl_total':           'KL — total',
    'bypass_alpha_mean':  'Bypass Gate α (mean)',
}


def _band_label(key):
    return _BAND_DISPLAY.get(key, key.replace('_', ' ').title())


def plot_all_losses(history, output_dir):
    """
    Saves two PNG files:

    loss_curves.png
      Row 0 (full-width): Overview — every loss on one axes.
      Row 1+: One subplot per loss component, each on its own y-axis scale.

    loss_kl_detail.png  (paper-ready, bands only)
    """
    if not history:
        print('  No history to plot.')
        return

    epochs      = [h['epoch']           for h in history]
    train_total = [h['train_loss']      for h in history]
    train_task  = [h['train_task_loss'] for h in history]
    train_kl    = [h['train_kl_loss']   for h in history]
    val_total   = [h['val_loss']        for h in history]
    val_miou    = [h.get('val_mIOU', 0) for h in history]

    # NEW: CACR and CE-AURC loss histories
    train_cacr    = [h.get('train_cacr_loss', 0) for h in history]
    train_ce_aurc = [h.get('train_ce_aurc_loss', 0) for h in history]

    all_band_keys = sorted({
        k for h in history for k in h.get('band_kl', {}).keys()
    })
    band_series = {
        k: [h.get('band_kl', {}).get(k, 0.0) for h in history]
        for k in all_band_keys
    }

    C_TRAIN  = '#2E86AB'
    C_VAL    = '#A23B72'
    C_TASK   = '#F18F01'
    C_TASK_V = '#C73E1D'
    C_KL     = '#555555'
    C_MIOU   = '#6A994E'
    C_CACR   = '#E63946'
    C_AURC   = '#457B9D'
    BAND_COLS = ['#BC4749', '#FF6B35', '#8338EC', '#3A86FF',
                 '#FFBE0B', '#FB5607', '#06D6A0', '#E63946']

    def _style_ax(ax, title, ylabel, xlabel='Epoch'):
        ax.set_title(title, fontweight='bold', pad=6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.22, ls='--', lw=0.6)
        ax.legend(fontsize=8, framealpha=0.85)

    panels = []
    panels.append((
        'Total Loss', 'Loss',
        [(train_total, 'Train total', dict(color=C_TRAIN, ls='-',  marker='o')),
         (val_total,   'Val total',   dict(color=C_VAL,   ls='--', marker='s'))],
    ))
    panels.append((
        'Task (CE) Loss', 'Loss',
        [(train_task, 'Train task',  dict(color=C_TASK,   ls='-',  marker='o')),
         (val_total,  'Val (proxy)', dict(color=C_TASK_V, ls='--', marker='s'))],
    ))
    panels.append((
        'KL Loss (total)', 'KL',
        [(train_kl, 'KL total', dict(color=C_KL, ls='-', marker='^'))],
    ))
    for idx, bk in enumerate(all_band_keys):
        col = BAND_COLS[idx % len(BAND_COLS)]
        panels.append((
            _band_label(bk), 'KL' if 'alpha' not in bk else 'α',
            [(band_series[bk], _band_label(bk), dict(color=col, ls='-', marker='o'))],
        ))
    # NEW: CACR loss panel
    if any(v > 0 for v in train_cacr):
        panels.append((
            'CACR Loss', 'Loss',
            [(train_cacr, 'CACR', dict(color=C_CACR, ls='-', marker='D'))],
        ))
    # NEW: CE-AURC loss panel
    if any(v > 0 for v in train_ce_aurc):
        panels.append((
            'CE-AURC Loss', 'Loss',
            [(train_ce_aurc, 'CE-AURC', dict(color=C_AURC, ls='-', marker='v'))],
        ))
    panels.append((
        'Val mIOU', 'mIOU (%)',
        [(val_miou, 'Val mIOU', dict(color=C_MIOU, ls='-', marker='D'))],
    ))

    n_individual  = len(panels)
    MAX_COLS      = 4
    n_rows_ind    = (n_individual + MAX_COLS - 1) // MAX_COLS
    n_rows_total  = 1 + n_rows_ind
    fig_w = min(5.5 * MAX_COLS, 22)
    fig_h = 4.5 + 4.0 * n_rows_ind

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(
        n_rows_total, 1, figure=fig,
        hspace=0.55, height_ratios=[4.0] + [4.0] * n_rows_ind)

    ax_ov = fig.add_subplot(outer[0])
    ax_ov.plot(epochs, train_total, '-',  lw=2,   color=C_TRAIN,  label='Train total',   alpha=0.9)
    ax_ov.plot(epochs, val_total,   '--', lw=1.8, color=C_VAL,    label='Val total',     alpha=0.9)
    ax_ov.plot(epochs, train_task,  '-',  lw=1.5, color=C_TASK,   label='Train task',    alpha=0.75)
    ax_ov.plot(epochs, train_kl,    lw=1.5, color=C_KL,     label='KL total',      alpha=0.65,
               ls=(0, (3, 2)))
    for idx, bk in enumerate(all_band_keys):
        if 'alpha' in bk:
            continue  # Don't plot alpha on the shared loss overview y-axis
        col = BAND_COLS[idx % len(BAND_COLS)]
        ax_ov.plot(epochs, band_series[bk], '-', lw=1.2, color=col,
                   label=_band_label(bk), alpha=0.7)
    # NEW: CACR and CE-AURC on overview
    if any(v > 0 for v in train_cacr):
        ax_ov.plot(epochs, train_cacr, '-', lw=1.3, color=C_CACR, label='CACR', alpha=0.8)
    if any(v > 0 for v in train_ce_aurc):
        ax_ov.plot(epochs, train_ce_aurc, '-', lw=1.3, color=C_AURC, label='CE-AURC', alpha=0.8)
    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold', pad=7)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss / KL')
    ax_ov.legend(fontsize=8, ncol=min(4, 2 + len(all_band_keys)),
                 framealpha=0.88, loc='upper right')
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows_ind, MAX_COLS, subplot_spec=outer[1:],
        hspace=0.55, wspace=0.38)

    for pidx, (title, ylabel, series_list) in enumerate(panels):
        row = pidx // MAX_COLS
        col = pidx %  MAX_COLS
        ax  = fig.add_subplot(inner[row, col])
        for vals, lbl, kwargs in series_list:
            kw = dict(lw=1.8, ms=3.5, mfc='white', mew=1.2, alpha=0.9)
            kw.update(kwargs)
            ax.plot(epochs, vals, label=lbl, **kw)
        if title == 'Val mIOU' and val_miou and max(val_miou) > 0:
            best_ep = epochs[int(np.argmax(val_miou))]
            best_v  = max(val_miou)
            ax.axvline(x=best_ep, color='red', ls=':', lw=1.2, alpha=0.7)
            ax.scatter([best_ep], [best_v], color='red', zorder=5, s=55,
                       label=f'Best ep={best_ep} ({best_v:.2f}%)')
        _style_ax(ax, title, ylabel)

    for pidx in range(n_individual, n_rows_ind * MAX_COLS):
        row = pidx // MAX_COLS
        col = pidx %  MAX_COLS
        fig.add_subplot(inner[row, col]).set_visible(False)

    fig.suptitle('MAMNet+SIB — Training Loss Curves', fontweight='bold',
                 fontsize=13, y=1.005)
    path_main = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(path_main, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved loss_curves.png  ({n_individual + 1} panels)  → {path_main}')

    if all_band_keys:
        n_kl = len(all_band_keys)
        n_kl_cols  = min(n_kl, 4)
        n_kl_rows  = (n_kl + n_kl_cols - 1) // n_kl_cols
        fig_kl, axes_kl = plt.subplots(
            n_kl_rows, n_kl_cols,
            figsize=(5.0 * n_kl_cols, 3.8 * n_kl_rows),
            squeeze=False)
        for idx, bk in enumerate(all_band_keys):
            r, c  = divmod(idx, n_kl_cols)
            ax_kl = axes_kl[r][c]
            col   = BAND_COLS[idx % len(BAND_COLS)]
            ax_kl.plot(epochs, band_series[bk], '-o', lw=2, ms=3.5,
                       color=col, mfc='white', mew=1.2,
                       label=_band_label(bk))
            ax_kl.set_title(_band_label(bk), fontweight='bold', pad=5)
            ax_kl.set_xlabel('Epoch')
            ax_kl.set_ylabel('KL Loss' if 'alpha' not in bk else 'α value')
            ax_kl.grid(True, alpha=0.22, ls='--', lw=0.6)
            ax_kl.legend(fontsize=8, framealpha=0.85)
        for idx in range(n_kl, n_kl_rows * n_kl_cols):
            r, c = divmod(idx, n_kl_cols)
            axes_kl[r][c].set_visible(False)
        fig_kl.suptitle('Per-Band KL Losses — Haar Wavelet Decomposition',
                         fontweight='bold', fontsize=12, y=1.01)
        plt.tight_layout()
        path_kl = os.path.join(output_dir, 'loss_kl_detail.png')
        fig_kl.savefig(path_kl, dpi=250, bbox_inches='tight')
        plt.close(fig_kl)
        print(f'  Saved loss_kl_detail.png ({n_kl} band panels)  → {path_kl}')


# ════════════════════════════════════════════════════════════════════════════
# Best/Worst prediction visualizations
# ════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(img_tensor):
    """CHW tensor → HW3 float [0,1] numpy (handles 3-ch or 4-ch)."""
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img[:, :, :3] * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img, 0, 1)


def save_best_worst_predictions(model, test_loader, device, output_dir,
                                 n_display=10):
    """
    Save best/worst PNG grids and an iou_statistics.json.
    Grid columns: Input | GT overlay | Pred overlay | Probability map
    """
    model.eval()
    all_images     = []
    all_masks_gt   = []
    all_masks_pred = []
    all_probs      = []
    all_ious       = []
    all_filenames  = []

    print('\nGenerating best/worst visualizations...')

    with torch.no_grad():
        for batch in test_loader:
            images        = batch['image'].to(device)
            masks         = batch['mask'].to(device)
            intensity_map = batch['intensity_map'].to(device)
            filenames     = batch['filename']

            outputs, _ = model(images, intensity_map=intensity_map)
            logits     = outputs if isinstance(outputs, torch.Tensor) else outputs['main']

            probs_batch  = F.softmax(logits, dim=1)[:, 1, :]
            filtered     = filter_small_predictions(logits, min_pixels=10)
            preds_batch  = filtered.argmax(dim=1)

            for i in range(images.size(0)):
                p  = preds_batch[i]
                g  = masks[i]
                tp = ((p == 1) & (g == 1)).sum().float()
                fp = ((p == 1) & (g == 0)).sum().float()
                fn = ((p == 0) & (g == 1)).sum().float()
                iou = (tp / (tp + fp + fn + 1e-10) * 100).item()

                all_images.append(images[i].cpu())
                all_masks_gt.append(g.cpu())
                all_masks_pred.append(p.cpu())
                all_probs.append(probs_batch[i].cpu())
                all_ious.append(iou)
                all_filenames.append(filenames[i])

    shadow_idx = [i for i in range(len(all_masks_gt))
                  if all_masks_gt[i].sum() > 0]
    if not shadow_idx:
        print('  Warning: no images with shadow in test set; skipping viz.')
        return

    shadow_ious = [all_ious[i] for i in shadow_idx]
    sorted_pos  = np.argsort(shadow_ious)

    worst_idx = [shadow_idx[p] for p in sorted_pos[:n_display]]
    best_idx  = [shadow_idx[p] for p in sorted_pos[-n_display:][::-1]]

    def _grid(indices, title, fname):
        n   = len(indices)
        fig = plt.figure(figsize=(16, 4 * n))
        gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.3, wspace=0.08)
        for row, idx in enumerate(indices):
            img_np    = _denorm(all_images[idx])
            gt_np     = all_masks_gt[idx].numpy()
            pred_np   = all_masks_pred[idx].numpy()
            prob_np   = all_probs[idx].numpy()
            iou_val   = all_ious[idx]

            ax0 = fig.add_subplot(gs[row, 0])
            ax0.imshow(img_np)
            ax0.set_title(f'Image {row+1} | ShIOU={iou_val:.1f}%',
                          fontsize=9, fontweight='bold')
            ax0.axis('off')

            ax1 = fig.add_subplot(gs[row, 1])
            ax1.imshow(img_np)
            ov = np.zeros((*gt_np.shape, 4))
            ov[gt_np == 1] = [0, 1, 0, 0.42]
            ax1.imshow(ov)
            ax1.set_title('GT overlay', fontsize=9, fontweight='bold')
            ax1.axis('off')

            ax2 = fig.add_subplot(gs[row, 2])
            ax2.imshow(img_np)
            ov2 = np.zeros((*pred_np.shape, 4))
            ov2[pred_np == 1] = [1, 0, 0, 0.42]
            ax2.imshow(ov2)
            ax2.set_title('Pred overlay', fontsize=9, fontweight='bold')
            ax2.axis('off')

            ax3 = fig.add_subplot(gs[row, 3])
            im  = ax3.imshow(prob_np, cmap='jet', vmin=0, vmax=1)
            ax3.set_title('Shadow prob', fontsize=9, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        plt.suptitle(title, fontsize=13, fontweight='bold', y=1.002)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {fname}')

    _grid(best_idx,  f'Top {n_display} Best  Predictions (Shadow IoU)',
          'best_predictions.png')
    _grid(worst_idx, f'Top {n_display} Worst Predictions (Shadow IoU)',
          'worst_predictions.png')

    stats = {
        'mean_iou':    float(np.mean(all_ious)),
        'std_iou':     float(np.std(all_ious)),
        'min_iou':     float(np.min(all_ious)),
        'max_iou':     float(np.max(all_ious)),
        'median_iou':  float(np.median(all_ious)),
        'best_files':  [all_filenames[i] for i in best_idx],
        'best_ious':   [all_ious[i]      for i in best_idx],
        'worst_files': [all_filenames[i] for i in worst_idx],
        'worst_ious':  [all_ious[i]      for i in worst_idx],
    }
    with open(os.path.join(output_dir, 'iou_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    print(f'  IoU stats  → iou_statistics.json  '
          f'(mean={stats["mean_iou"]:.2f}%  median={stats["median_iou"]:.2f}%)')


# ════════════════════════════════════════════════════════════════════════════
# LOCO fold mapping
# ════════════════════════════════════════════════════════════════════════════

CITY_FOLDS = {
    0: {'holdout': 'phoenix'},
    1: {'holdout': 'miami'},
    2: {'holdout': 'chicago'},
}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def run_training(args):
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    print(f'SIB config: haar={args.use_haar}, vib={args.use_vib}, '
          f'aug={args.use_content_aug}, adaptive_beta={args.adaptive_beta}, '
          f'sag={args.use_sag}, multiscale={args.use_multiscale_sib}, '
          f'gate={args.use_passthrough_gate}, mod_bypass={args.use_module_bypass}')
    print(f'Data: contrast={args.use_contrast}, fda={args.use_fda} '
          f'(L={args.fda_L})')
    print(f'New modules: CACR={args.use_cacr} (w={args.cacr_weight}), '
          f'CE-AURC={args.use_ce_aurc} (w={args.ce_aurc_weight}), '
          f'TENT={args.use_tent} (steps={args.tent_steps})')

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.mode == 'loco':
        args.test_city = CITY_FOLDS[args.fold_id]['holdout']
    else:
        args.test_city = None

    fda_target_root = getattr(args, 'fda_target_root', None)
    if args.mode == 'loco' and args.use_fda and not fda_target_root:
        fda_target_root = os.path.join(
            args.base_data_root, args.test_city, args.resolution)
        print(f'FDA target auto-resolved: {fda_target_root}')

    dataloaders = get_dataloaders_sib(
        data_root=args.data_root,
        base_data_root=args.base_data_root,
        mode=args.mode,
        resolution=args.resolution,
        fold_id=args.fold_id if args.mode == 'loco' else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_fda=args.use_fda,
        fda_target_root=fda_target_root if args.use_fda else None,
        fda_L=args.fda_L,
        use_contrast=args.use_contrast,
    )

    train_loader = dataloaders['train']
    val_loader   = dataloaders['val']
    test_loader  = dataloaders['test']

    print(f'Train: {len(train_loader)} batches | '
          f'Val: {len(val_loader)} | Test: {len(test_loader)}')

    model     = build_mamnet_sib(args).to(device)
    criterion = MAMNetLoss(aux_weight=0.4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── NEW: Instantiate CACR and CE-AURC losses ─────────────────────
    cacr_criterion = None
    if args.use_cacr:
        cacr_criterion = CACRLoss(
            pos_weight=1.0,
            neg_weight=args.cacr_neg_weight)
        print(f'CACR loss: weight={args.cacr_weight}, '
              f'neg_weight={args.cacr_neg_weight}')

    ce_aurc_criterion = None
    if args.use_ce_aurc:
        ce_aurc_criterion = CEAURCLoss(
            floor_weight=args.ce_aurc_floor)
        print(f'CE-AURC loss: weight={args.ce_aurc_weight}, '
              f'floor={args.ce_aurc_floor}')

    best_metric      = -float('inf')
    patience_counter = 0
    history          = []

    for epoch in range(args.epochs):
        t0 = time.time()

        train_stats = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs, args.vib_warmup_fraction,
            cacr_criterion=cacr_criterion,
            cacr_weight=args.cacr_weight if args.use_cacr else 0.0,
            ce_aurc_criterion=ce_aurc_criterion,
            ce_aurc_weight=args.ce_aurc_weight if args.use_ce_aurc else 0.0)

        val_stats = validate(model, val_loader, criterion, device,
                      boundary_tolerant=args.eval_boundary_tolerant,
                      tolerance=args.boundary_tolerance)

        scheduler.step()
        elapsed = time.time() - t0

        if args.eval_boundary_tolerant and f'tolerant_{args.boundary_tolerance}px' in val_stats:
            current_metric = val_stats[f'tolerant_{args.boundary_tolerance}px'].get('mIOU', 0.0)
        else:
            current_metric = val_stats['strict'].get('mIOU', 0.0)

        tm = train_stats['metrics']
        bk = train_stats['band_kl']
        print(f'\nEpoch {epoch+1}/{args.epochs} ({elapsed:.1f}s)')
        print(f'  Loss  total={train_stats["total"]:.4f}  '
              f'task={train_stats["task"]:.4f}  '
              f'kl={train_stats["kl"]:.6f}  '
              f'vib_w={train_stats["vib_weight"]:.3f}')
        # NEW: CACR and CE-AURC loss logging
        if 'cacr' in train_stats:
            cacr_d = train_stats.get('cacr_diag', {})
            print(f'  CACR={train_stats["cacr"]:.6f}  '
                  f'pos_shift={cacr_d.get("cacr_pos_shift", 0):.4f}  '
                  f'n_pos={cacr_d.get("cacr_n_pos", 0):.0f}')
        if 'ce_aurc' in train_stats:
            aurc_d = train_stats.get('ce_aurc_diag', {})
            print(f'  CE-AURC={train_stats["ce_aurc"]:.6f}  '
                  f'mean_conf={aurc_d.get("ce_aurc_mean_shadow_conf", 0):.4f}  '
                  f'mean_ce={aurc_d.get("ce_aurc_mean_shadow_ce", 0):.4f}')
        if bk:
            band_str = '  '.join(
                f'{_band_label(k)}={v:.6f}' for k, v in sorted(bk.items()))
            print(f'  KL bands: {band_str}')
        vs = val_stats['strict']
        print(f'  Val strict:   F1={vs["F1"]:.2f}  '
              f'mIOU={vs["mIOU"]:.2f}  ShIOU={vs["Shadow_IOU"]:.2f}  '
              f'BER={vs["BER"]:.2f}')
        if f'tolerant_{args.boundary_tolerance}px' in val_stats:
            vt = val_stats[f'tolerant_{args.boundary_tolerance}px']
            print(f'  Val tolerant: F1={vt["F1"]:.2f}  '
                  f'mIOU={vt["mIOU"]:.2f}  ShIOU={vt["Shadow_IOU"]:.2f}')
        print(f'  Tracking mIOU: {current_metric:.4f}')

        history_entry = {
            'epoch':               epoch + 1,
            'train_loss':          train_stats['total'],
            'train_task_loss':     train_stats['task'],
            'train_kl_loss':       train_stats['kl'],
            'band_kl':             {k: float(v) for k, v in bk.items()},
            'vib_warmup_weight':   train_stats['vib_weight'],
            'val_loss':            val_stats['loss'],
            'val_metrics_strict':  val_stats['strict'],
            'val_metrics_tolerant': val_stats.get(f'tolerant_{args.boundary_tolerance}px', {}),
            'val_mIOU':            current_metric,
            'lr':                  optimizer.param_groups[0]['lr'],
        }
        # NEW: save CACR/CE-AURC to history
        if 'cacr' in train_stats:
            history_entry['train_cacr_loss'] = train_stats['cacr']
            history_entry['cacr_diag'] = train_stats.get('cacr_diag', {})
        if 'ce_aurc' in train_stats:
            history_entry['train_ce_aurc_loss'] = train_stats['ce_aurc']
            history_entry['ce_aurc_diag'] = train_stats.get('ce_aurc_diag', {})

        history.append(history_entry)

        if current_metric > best_metric:
            best_metric      = current_metric
            patience_counter = 0
            ckpt = {
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric':          best_metric,
                'args':                 vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pth'))
            print(f'  ★ New best: mIOU={best_metric:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print('\nGenerating loss plots...')
    plot_all_losses(history, args.output_dir)

    print(f'\n{"="*70}')
    if args.mode == 'loco':
        print(f'Final Test on {args.test_city} (0-shot LOCO)')
    else:
        print(f'Final Test (mode={args.mode})')
    if args.use_tent:
        print(f'  TENT enabled: steps={args.tent_steps} lr={args.tent_lr}')
    print(f'{"="*70}')

    ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f'Loaded best model from epoch {ckpt["epoch"]}')

    (strict, tolerant,
     strict_list, tolerant_list,
     all_filenames) = test_and_save_predictions(
        model, test_loader, device, args, args.output_dir)
    
    # ── §4.3: Class-Conditional Temperature Scaling (post-hoc) ────────
    if args.use_class_cond_tempscale:
        print(f'\n{"="*70}')
        print(f'§4.3 MODULE: Class-Conditional Temperature Scaling')
        print(f'{"="*70}')

        # Step 1: collect val logits (source-city, labeled)
        print('  Collecting source-city validation logits...')
        val_logits, val_labels, _ = collect_logits_and_labels(
            model, val_loader, device)
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        print(f'  Val: {val_logits.size(0)} images')

        # Step 2: fit T_pos, T_neg
        print('  Fitting (T_pos, T_neg) via LBFGS...')
        T_pos, T_neg = fit_class_conditional_temperature(
            val_logits, val_labels, max_iter=args.tempscale_max_iter)
        print(f'  → T_pos = {T_pos:.4f}, T_neg = {T_neg:.4f}')

        # Step 3: collect test logits, evaluate with and without tempscale
        print('  Collecting test logits and computing SP-gap metrics...')
        test_logits, test_labels, test_fnames = collect_logits_and_labels(
            model, test_loader, device)

        # Baseline (T=1.0): SP-gap reference
        sp_baseline = compute_sp_metrics(test_logits, test_labels)

        # Tempscale applied
        scaled_test_logits = apply_tempscale(test_logits, T_pos, T_neg)
        sp_tempscale = compute_sp_metrics(scaled_test_logits, test_labels)

        # Re-compute strict / tolerant on rescaled predictions and save PNGs
        ts_pred_dir = os.path.join(args.output_dir, 'predictions_tempscale')
        os.makedirs(ts_pred_dir, exist_ok=True)
        ts_filtered = filter_small_predictions(scaled_test_logits, min_pixels=10)
        ts_preds = ts_filtered.argmax(dim=1).numpy().astype(np.uint8)
        gt_np = test_labels.numpy().astype(np.uint8)

        ts_strict_list, ts_tolerant_list = [], []
        for i, fn in enumerate(test_fnames):
            Image.fromarray(ts_preds[i] * 255).save(
                os.path.join(ts_pred_dir, fn))
            ts_strict_list.append(_compute_strict_metrics(ts_preds[i], gt_np[i]))
            ts_tolerant_list.append(_compute_tolerant_metrics(
                ts_preds[i], gt_np[i], tolerance=args.boundary_tolerance))
        ts_strict = _average_metrics(ts_strict_list)
        ts_tolerant = _average_metrics(ts_tolerant_list)
        tol_key = f'tolerant_{args.boundary_tolerance}px'

        ts_summary = {
            'T_pos': T_pos, 'T_neg': T_neg,
            'baseline_T1': {
                'strict': strict, tol_key: tolerant, 'sp_metrics': sp_baseline,
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
        with open(os.path.join(args.output_dir, 'tempscale_results.json'), 'w') as f:
            json.dump(ts_summary, f, indent=4)

        print(f'\n  Baseline (T=1.0):')
        print(f'    mIoU={strict["mIOU"]:.2f}  '
              f'AURC_shadow={sp_baseline["aurc_shadow"]:.4f}  '
              f'ECE_pos={sp_baseline["ece_pred_pos"]:.4f}')
        print(f'  Tempscale (T_pos={T_pos:.3f}, T_neg={T_neg:.3f}):')
        print(f'    mIoU={ts_strict["mIOU"]:.2f}  '
              f'AURC_shadow={sp_tempscale["aurc_shadow"]:.4f}  '
              f'ECE_pos={sp_tempscale["ece_pred_pos"]:.4f}')
        print(f'  ΔAURC_shadow = {ts_summary["sp_gap_reduction"]["aurc_shadow"]:+.4f}')
        print(f'  ΔECE_pos     = {ts_summary["sp_gap_reduction"]["ece_pred_pos"]:+.4f}')
        print(f'  Saved → {os.path.join(args.output_dir, "tempscale_results.json")}')

    if args.mode == 'loco':
        compare_with_baselines(
            strict, tolerant,
            strict_list, tolerant_list,
            all_filenames, args, args.output_dir)
    else:
        comp = {
            'sib':       {'strict': strict, f'tolerant_{args.boundary_tolerance}px': tolerant},
            'ddib':      {'strict': strict, f'tolerant_{args.boundary_tolerance}px': tolerant},
            'baselines': {},
        }
        with open(os.path.join(args.output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(comp, f, indent=4)

    save_best_worst_predictions(model, test_loader, device,
                                 args.output_dir, n_display=10)

    print(f'\nDone! Output: {args.output_dir}')
    return {'strict': strict, f'tolerant_{args.boundary_tolerance}px': tolerant}


if __name__ == '__main__':
    args = parse_args()
    run_training(args)