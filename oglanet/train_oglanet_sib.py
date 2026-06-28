"""
Training script for OGLANet + SIB in 0-shot LOCO cross-location setting.

CLI interface matches oglanet_sib.sh / oglanet_sib_newmod.sh env-var-to-flag mapping:
  --mode loco|all|single
  --base_data_root / --data_root
  --fold_id 0|1|2
  --use_haar, --use_vib, --use_content_aug, --adaptive_beta
  --use_sag, --use_multiscale_sib, --use_passthrough_gate
  --use_module_bypass
  --use_contrast, --use_fda, --fda_L, --fda_target_root
  --beta_content, --beta_edge, --noise_scale
  --beta_max_multiplier, --multiscale_beta_base, --vib_warmup_fraction
  --comparison_inference_dir, --comparison_data_root
  --oglanet_output_dir   (root scanned for a completed donor experiment)

  Ablation flags (A1–A10):
  --skip_ll_vib          A1: skip VIB on LL subband
  --symmetric_beta       A3: force all subbands to use beta_content
  --aug_all_subbands     A6: apply ContentAugmentation to all subbands
  --vib_only_band BAND   A10: apply VIB only to the named band (LL/LH/HL/HH)

  NEW — Diagnostic-motivated module flags:
  --use_cacr             Enable Class-Asymmetric Confidence Regularizer
  --cacr_weight          CACR loss weight (default 0.1)
  --cacr_neg_weight      CACR weight for pred-negative pixels (default 0)
  --use_ce_aurc          Enable CE-AURC auxiliary loss on gt_shadow pixels
  --ce_aurc_weight       CE-AURC loss weight (default 0.01)
  --ce_aurc_floor        CE-AURC minimum weight for shadow pixels (default 0.5)
  --use_tent             Enable test-time entropy minimization on BN affine
  --tent_steps           TENT adaptation steps per batch (default 1)
  --tent_lr              TENT optimizer learning rate (default 0.001)

Baseline comparison — two strategies (tried in order, both may succeed):

  Strategy 1 — OGLANet donor  (fast path, optional)
    Scan --oglanet_output_dir for a completed OGLANet experiment that
    already has a comparison_results.json with populated baselines.

  Strategy 2 — Raw prediction dirs  (always attempted)
    Scan --comparison_inference_dir for per-city prediction images.

Loss plots saved after training:
  loss_curves.png      — overview panel + one subplot per loss component
                         (now includes CACR and CE-AURC panels when active)
  loss_kl_detail.png   — standalone per-band KL figure (paper-ready)
  val_miou.png         — validation decision mIOU with best-epoch marker
"""

import os
import sys
import argparse
import time
import json
import logging
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use('Agg')   # headless — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# ── Project imports ──
from data.dataset_sib import get_dataloaders_sib
from models.oglanet_sib import OGLANetSIB, OGLANetSIBLoss
from utils.evaluation_detailed import DetailedEvaluator
from utils.losses import CACRLoss, CEAURCLoss

# TENT utilities (test-time adaptation)
from models.sib import configure_tent, tent_adapt_step


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='OGLANet+SIB Training')

    # ── Mode & data paths ──
    p.add_argument('--mode', type=str, required=True,
                   choices=['loco', 'all', 'single'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--data_root',      type=str, default=None)
    p.add_argument('--resolution', type=str, default='highres',
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=0, choices=[0, 1, 2])
    p.add_argument('--img_size',    type=int, default=384)
    p.add_argument('--num_workers', type=int, default=1)

    # ── SIB component flags ──
    p.add_argument('--use_haar',           action='store_true', default=False)
    p.add_argument('--use_vib',            action='store_true', default=False)
    p.add_argument('--use_content_aug',    action='store_true', default=False)
    p.add_argument('--adaptive_beta',      action='store_true', default=False)
    p.add_argument('--use_sag',            action='store_true', default=False)
    p.add_argument('--use_multiscale_sib', action='store_true', default=False)
    p.add_argument('--use_passthrough_gate', action='store_true', default=False)
    p.add_argument('--use_module_bypass',  action='store_true', default=False)

    # ── SIB hyperparameters ──
    p.add_argument('--beta_content',         type=float, default=0.001)
    p.add_argument('--beta_edge',            type=float, default=0.0001)
    p.add_argument('--noise_scale',          type=float, default=0.05)
    p.add_argument('--beta_max_multiplier',  type=float, default=3.0)
    p.add_argument('--multiscale_beta_base', type=float, default=0.0005)
    p.add_argument('--vib_warmup_fraction',  type=float, default=0.1)

    # ── Content augmentation hyperparameters ──
    p.add_argument('--sigma_style', type=float, default=0.1)
    p.add_argument('--sigma_shift', type=float, default=0.05)
    p.add_argument('--aug_p_aug',   type=float, default=0.5)
    p.add_argument('--aug_p_mix',   type=float, default=0.3)

    # ── Data flags ──
    p.add_argument('--use_contrast', action='store_true', default=False)
    p.add_argument('--use_fda',      action='store_true', default=False)
    p.add_argument('--fda_L',           type=float, default=0.01)
    p.add_argument('--fda_target_root', type=str,   default=None)

    # ── Model ──
    p.add_argument('--num_classes',        type=int,  default=2)
    p.add_argument('--pretrained_encoder', action='store_true', default=True)

    # ── Training ──
    p.add_argument('--batch_size',              type=int,   default=8)
    p.add_argument('--epochs',                  type=int,   default=100)
    p.add_argument('--lr',                      type=float, default=0.0003)
    p.add_argument('--encoder_lr_mult',         type=float, default=0.1)
    p.add_argument('--weight_decay',            type=float, default=1e-4)
    p.add_argument('--grad_clip',               type=float, default=1.0)
    p.add_argument('--early_stopping_patience', type=int,   default=15)
    p.add_argument('--use_amp',     action='store_true', default=True)
    p.add_argument('--lambda_kl',   type=float, default=0.001)

    # ── Evaluation ──
    p.add_argument('--eval_boundary_tolerant', action='store_true', default=False)

    # ── Comparison baselines ──
    p.add_argument('--comparison_inference_dir', type=str, default=None,
                   help='Root of Test_img_results: .../loco/{city}/{res}/oglanet/vanilla/')
    p.add_argument('--comparison_data_root', type=str, default=None,
                   help='Base data root for GT masks (Final_data_test)')
    p.add_argument('--oglanet_output_dir', type=str, default=None,
                   help='OGLANet outputs root scanned for a completed donor '
                        'experiment (e.g. .../data/oglanet/outputs/)')

    # ── Output ──
    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--boundary_tolerance', type=int, default=2)

    # ══════════════════════════════════════════════════════════════════════
    # Ablation flags (A1–A10) — all default off to preserve C4 behavior
    # ══════════════════════════════════════════════════════════════════════
    p.add_argument('--skip_ll_vib', action='store_true', default=False,
                   help='A1: skip VIB on LL subband (pass through unchanged)')
    p.add_argument('--symmetric_beta', action='store_true', default=False,
                   help='A3: use beta_content for ALL subbands (no asymmetry)')
    p.add_argument('--aug_all_subbands', action='store_true', default=False,
                   help='A6: apply ContentAugmentation to LH/HL/HH too')
    p.add_argument('--vib_only_band', type=str, default=None,
                   choices=['LL', 'LH', 'HL', 'HH'],
                   help='A10: apply VIB only to this band, skip all others')

    # ══════════════════════════════════════════════════════════════════════
    # NEW: Diagnostic-motivated modules (CACR, CE-AURC, TENT)
    # ══════════════════════════════════════════════════════════════════════
    p.add_argument('--use_cacr', action='store_true', default=False,
                   help='Enable CACR (Class-Asymmetric Confidence Regularizer)')
    p.add_argument('--cacr_weight', type=float, default=0.1,
                   help='CACR loss weight (default: 0.1)')
    p.add_argument('--cacr_neg_weight', type=float, default=0.0,
                   help='CACR weight for pred-negative pixels (default 0; '
                        'set >0 to also penalize background logit shift)')

    p.add_argument('--use_ce_aurc', action='store_true', default=False,
                   help='Enable CE-AURC auxiliary loss on gt_shadow pixels')
    p.add_argument('--ce_aurc_weight', type=float, default=0.01,
                   help='CE-AURC loss weight (default: 0.01)')
    p.add_argument('--ce_aurc_floor', type=float, default=0.5,
                   help='CE-AURC minimum weight for shadow pixels (default: 0.5)')

    p.add_argument('--use_tent', action='store_true', default=False,
                   help='Enable TENT (test-time entropy minimization on BN affine)')
    p.add_argument('--tent_steps', type=int, default=1,
                   help='TENT adaptation steps per batch (default: 1)')
    p.add_argument('--tent_lr', type=float, default=0.001,
                   help='TENT optimizer learning rate (default: 0.001)')
    p.add_argument('--tent_pred_pos_only', action='store_true', default=True,
                   help='TENT: minimize entropy only on pred-positive pixels')
    
    # §4.3 module: Class-conditional temperature scaling
    p.add_argument('--use_class_cond_tempscale', action='store_true',
                   help='Fit T_pos/T_neg on source-city val, apply at test')
    p.add_argument('--tempscale_max_iter', type=int, default=200)

    args = p.parse_args()

    args.use_sib         = args.use_haar or args.use_vib
    args.in_channels     = 4 if args.use_contrast else 3
    args.kl_warmup_epochs = max(1, int(args.epochs * args.vib_warmup_fraction))
    args.fold_names      = ['phoenix', 'miami', 'chicago']
    if args.mode == 'loco':
        args.test_city = args.fold_names[args.fold_id]

    return args


# ═══════════════════════════════════════════════════════════════════════════════
# Per-image metric functions
# ═══════════════════════════════════════════════════════════════════════════════

_TOLERANCE_KERNEL_CACHE = {}


def _get_tolerance_kernel(tolerance):
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1))
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


def _compute_tolerant_metrics(pred, gt, tolerance=2):
    kernel   = _get_tolerance_kernel(tolerance)
    gt_uint8 = gt.astype(np.uint8)
    eroded   = cv2.erode(gt_uint8, kernel)
    dilated  = cv2.dilate(gt_uint8, kernel)
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


def _average_metrics(metrics_list):
    if not metrics_list:
        return {k: 0.0 for k in
                ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
    keys = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

# ════════════════════════════════════════════════════════════════════════════
# §4.3 MODULE: Class-Conditional Temperature Scaling
#   Fit T_pos and T_neg on source-city validation logits (no target labels),
#   apply at inference on held-out city. Reduces SP-gap without target data.
#   Reference: Guo et al. (ICML 2017), Tian et al. (CVPR 2023).
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_logits_and_labels_oglanet(model, loader, device, args):
    """Run inference on `loader`, return concatenated CPU p6 logits + labels."""
    model.eval()
    all_logits, all_labels, all_filenames = [], [], []
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)
        city_ids = batch.get('city_id', None)
        if city_ids is not None:
            city_ids = city_ids.to(device)

        with autocast(enabled=args.use_amp):
            out = model(images, intensity_map=intensity_map, city_ids=city_ids)
        logits = out['predictions']['p6'].float()  # [B, 2, H, W]
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
    """Area Under Risk-Coverage curve."""
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
    """SP-gap (foreground class), background AURC, and class-stratified ECE."""
    probs = F.softmax(logits, dim=1)
    shadow_prob = probs[:, 1, :, :]
    preds = logits.argmax(dim=1)

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


# ═══════════════════════════════════════════════════════════════════════════════
# Training  — tracks per-band KL losses + bypass alpha + CACR + CE-AURC
# ═══════════════════════════════════════════════════════════════════════════════

# Keys in kl_losses that are NOT actual losses (should not be summed into KL)
_NON_LOSS_KEYS = {'bypass_alpha', 'ref_predictions', 'pre_aug_bottleneck'}
_AGGREGATE_KEYS = {'kl_total', 'kl_multiscale'}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, args,
                    epoch, logger,
                    cacr_criterion=None, cacr_weight=0.0,
                    ce_aurc_criterion=None, ce_aurc_weight=0.0):
    """
    Train one epoch with optional CACR and CE-AURC losses.

    Args:
        model, loader, criterion, optimizer, scaler, device, args, epoch, logger:
            standard.
        cacr_criterion: CACRLoss instance (None to disable).
        cacr_weight: Weight for CACR loss term.
        ce_aurc_criterion: CEAURCLoss instance (None to disable).
        ce_aurc_weight: Weight for CE-AURC loss term.
    """
    model.train()
    criterion.set_epoch(epoch)

    total_loss         = 0.0
    total_seg_loss     = 0.0
    total_kl_loss      = 0.0
    total_cacr_loss    = 0.0
    total_ce_aurc_loss = 0.0
    band_kl_accum      = defaultdict(float)
    cacr_diag_accum    = defaultdict(float)
    ce_aurc_diag_accum = defaultdict(float)
    n_batches          = 0

    for batch_idx, batch in enumerate(loader):
        images        = batch['image'].to(device)
        masks         = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)
        city_ids      = batch['city_id'].to(device)

        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):
            out         = model(images, intensity_map=intensity_map,
                                city_ids=city_ids)
            predictions = out['predictions']
            kl_losses   = out['kl_losses']

            loss_dict = criterion(predictions, masks, kl_losses)
            loss      = loss_dict['total']

            # ── CACR loss ─────────────────────────────────────────────
            cacr_loss_val = torch.tensor(0.0, device=device)
            if (cacr_criterion is not None
                    and 'ref_predictions' in kl_losses):
                main_p6 = predictions['p6']
                # ref predictions were generated under no_grad in the model,
                # but detach again for safety
                ref_p6 = kl_losses['ref_predictions']['p6'].detach()
                cacr_loss_val, cacr_diag = cacr_criterion(
                    main_p6, ref_p6, targets=masks)
                for k, v in cacr_diag.items():
                    cacr_diag_accum[k] += v
                loss = loss + cacr_weight * cacr_loss_val

            # ── CE-AURC auxiliary loss ────────────────────────────────
            ce_aurc_loss_val = torch.tensor(0.0, device=device)
            if ce_aurc_criterion is not None:
                main_p6 = predictions['p6']
                ce_aurc_loss_val, aurc_diag = ce_aurc_criterion(main_p6, masks)
                for k, v in aurc_diag.items():
                    ce_aurc_diag_accum[k] += v
                loss = loss + ce_aurc_weight * ce_aurc_loss_val

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate per-band KL for history tracking (leaf terms only)
        for band_key, band_val in kl_losses.items():
            if band_key in _NON_LOSS_KEYS:
                continue
            if isinstance(band_val, torch.Tensor) and band_key not in _AGGREGATE_KEYS:
                if band_val.dim() == 0:
                    band_kl_accum[band_key] += band_val.item()

        # Track bypass gate alpha mean (diagnostic, not a loss)
        if 'bypass_alpha' in kl_losses:
            band_kl_accum['bypass_alpha_mean'] += \
                kl_losses['bypass_alpha'].detach().mean().item()

        total_loss         += loss.item()
        total_seg_loss     += loss_dict['seg'].item()
        total_kl_loss      += loss_dict['kl'].item()
        total_cacr_loss    += cacr_loss_val.item()
        total_ce_aurc_loss += ce_aurc_loss_val.item()
        n_batches          += 1

        if batch_idx % 50 == 0:
            log_msg = (
                f'  Epoch {epoch} [{batch_idx}/{len(loader)}] '
                f'loss={loss.item():.4f} seg={loss_dict["seg"].item():.4f} '
                f'kl={loss_dict["kl"].item():.6f} '
                f'kl_w={loss_dict["kl_weighted"].item():.6f}')
            if cacr_criterion is not None:
                log_msg += f' cacr={cacr_loss_val.item():.6f}'
            if ce_aurc_criterion is not None:
                log_msg += f' ceaurc={ce_aurc_loss_val.item():.6f}'
            if 'bypass_alpha' in kl_losses:
                log_msg += f' α={kl_losses["bypass_alpha"].mean().item():.3f}'
            logger.info(log_msg)

    nb = max(n_batches, 1)
    avg_band_kl = {k: v / nb for k, v in band_kl_accum.items()}

    result = {
        'total':   total_loss / nb,
        'task':    total_seg_loss / nb,
        'kl':      total_kl_loss / nb,
        'band_kl': avg_band_kl,
    }

    if cacr_criterion is not None:
        result['cacr'] = total_cacr_loss / nb
        result['cacr_diag'] = {k: v / nb for k, v in cacr_diag_accum.items()}

    if ce_aurc_criterion is not None:
        result['ce_aurc'] = total_ce_aurc_loss / nb
        result['ce_aurc_diag'] = {k: v / nb for k, v in ce_aurc_diag_accum.items()}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, criterion, evaluator, device, args, epoch):
    model.eval()
    criterion.set_epoch(epoch)
    evaluator.reset()

    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        images        = batch['image'].to(device)
        masks         = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)
        city_ids      = batch['city_id'].to(device)

        with autocast(enabled=args.use_amp):
            out         = model(images, intensity_map=intensity_map,
                                city_ids=city_ids)
            predictions = out['predictions']
            kl_losses   = out['kl_losses']

            loss_dict   = criterion(predictions, masks, kl_losses)
            total_loss += loss_dict['total'].item()

        pred_p6      = predictions['p6']
        pred_classes = pred_p6.argmax(dim=1)
        evaluator.update(pred_classes, masks, images)
        n_batches += 1

    results = evaluator.compute_metrics()

    if args.eval_boundary_tolerant:
        tolerant = results['boundary_tolerant'][f'tolerant_{args.boundary_tolerance}px']
        decision_miou = tolerant['iou']
    else:
        strict        = results['overall']
        decision_miou = strict['iou']

    return {
        'loss':          total_loss / max(n_batches, 1),
        'metrics':       results,
        'decision_miou': decision_miou,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Test — saves predictions + computes per-image strict/tolerant metrics
#         + saves per-image bypass gate alpha diagnostics
#         + NEW: optionally applies TENT (test-time entropy minimization)
# ═══════════════════════════════════════════════════════════════════════════════

def test_and_save_predictions(model, loader, device, args, logger):
    """
    Run test inference, save prediction PNGs, compute per-image metrics.

    NEW: If --use_tent is enabled, applies test-time entropy minimization
    on BN affine parameters before making predictions. Trailing batches
    of size 1 skip TENT (BN train mode requires ≥2 samples per channel),
    but predictions still happen in eval mode.
    """
    model.eval()

    pred_save_dir = os.path.join(args.output_dir, 'predictions')
    os.makedirs(pred_save_dir, exist_ok=True)

    sib_strict_list   = []
    sib_tolerant_list = []
    all_filenames     = []
    all_alphas        = []

    # ── TENT setup (optional) ────────────────────────────────────────
    tent_active    = getattr(args, 'use_tent', False)
    tent_params    = None
    tent_optimizer = None
    norm_layers    = None

    if tent_active:
        logger.info(f'\n  TENT enabled: steps={args.tent_steps}  '
                    f'lr={args.tent_lr}  '
                    f'pred_pos_only={args.tent_pred_pos_only}')
        tent_params, norm_layers = configure_tent(
            model, use_bn=True, use_ln=False)
        if tent_params:
            tent_optimizer = torch.optim.SGD(
                tent_params, lr=args.tent_lr, momentum=0.9)
            logger.info(f'  TENT: {len(tent_params)} params to adapt '
                        f'from {len(norm_layers)} BN layers')
        else:
            logger.info('  TENT: no BN layers found — disabling.')
            tent_active = False

    for batch in loader:
        images        = batch['image'].to(device)
        masks         = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)

        # Keep city_ids consistent with the dataloader (may not exist on test
        # batches in some configurations — guard with .get)
        city_ids = batch.get('city_id', None)
        if city_ids is not None:
            city_ids = city_ids.to(device)

        # ── TENT adaptation (skip if batch < 2; BN-in-train needs ≥2) ──
        if tent_active and tent_optimizer is not None:
            if images.size(0) < 2:
                # Trailing partial batch — adaptation skipped. Predictions
                # below run in eval mode using already-adapted affine params
                # from previous batches.
                pass
            else:
                for _ in range(args.tent_steps):
                    tent_adapt_step(
                        model, images, intensity_map,
                        tent_optimizer, norm_layers,
                        pred_pos_only=args.tent_pred_pos_only,
                        city_ids=city_ids)
            # Always set norm layers back to eval before prediction
            model.eval()

        # ── Prediction (eval mode) ────────────────────────────────────
        with torch.no_grad():
            with autocast(enabled=args.use_amp):
                out         = model(images, intensity_map=intensity_map,
                                    city_ids=city_ids)
                predictions = out['predictions']
                kl_losses   = out['kl_losses']

        pred_p6 = predictions['p6']
        preds   = pred_p6.argmax(dim=1)

        # Collect bypass gate alpha values (per-image)
        if 'bypass_alpha' in kl_losses:
            alpha_batch = kl_losses['bypass_alpha'].cpu().numpy()
            for j in range(alpha_batch.shape[0]):
                all_alphas.append(float(alpha_batch[j]))

        for i, fname in enumerate(batch['filename']):
            pred_np = preds[i].cpu().numpy().astype(np.uint8)
            gt_np   = masks[i].cpu().numpy().astype(np.uint8)

            Image.fromarray(pred_np * 255).save(
                os.path.join(pred_save_dir, fname))

            sib_strict_list.append(_compute_strict_metrics(pred_np, gt_np))
            sib_tolerant_list.append(
                _compute_tolerant_metrics(pred_np, gt_np, tolerance=args.boundary_tolerance))
            all_filenames.append(fname)

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
    if tent_active:
        logger.info(f'  TENT: active ({args.tent_steps} steps/batch)')

    test_results = {
        'num_images':   len(all_filenames),
        'strict':       sib_strict,
        f'tolerant_{args.boundary_tolerance}px': sib_tolerant,
        'tent_active':  tent_active,
    }
    if all_alphas:
        test_results['bypass_alpha_summary'] = {
            'mean': float(np.mean(all_alphas)),
            'std':  float(np.std(all_alphas)),
            'min':  float(np.min(all_alphas)),
            'max':  float(np.max(all_alphas)),
        }
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)

    # Save bypass gate alpha diagnostics (if module bypass was active)
    if all_alphas:
        alpha_data = {
            'mean_alpha': float(np.mean(all_alphas)),
            'std_alpha':  float(np.std(all_alphas)),
            'min_alpha':  float(np.min(all_alphas)),
            'max_alpha':  float(np.max(all_alphas)),
            'per_image': {fn: a for fn, a in zip(all_filenames, all_alphas)},
        }
        alpha_path = os.path.join(args.output_dir, 'bypass_gate_alpha.json')
        with open(alpha_path, 'w') as f:
            json.dump(alpha_data, f, indent=4)
        logger.info(f'  Bypass gate α: mean={alpha_data["mean_alpha"]:.4f}  '
                    f'std={alpha_data["std_alpha"]:.4f}  '
                    f'min={alpha_data["min_alpha"]:.4f}  '
                    f'max={alpha_data["max_alpha"]:.4f}')
        logger.info(f'  Saved bypass_gate_alpha.json → {alpha_path}')

    return (sib_strict, sib_tolerant,
            sib_strict_list, sib_tolerant_list,
            all_filenames)


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _dir_has_images(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path)
               if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff', '.jpeg')))


def _find_oglanet_donor(args, logger):
    output_root = getattr(args, 'oglanet_output_dir', None)
    test_city   = getattr(args, 'test_city', '')
    res         = args.resolution

    if not output_root:
        logger.info('  [S1] --oglanet_output_dir not set; skipping donor search.')
        return None, None
    if not os.path.isdir(output_root):
        logger.info(f'  [S1] oglanet_output_dir not found: {output_root}')
        return None, None

    logger.info(f'\n  [S1] Scanning for OGLANet donor in: {output_root}')
    candidates = []
    for entry in os.listdir(output_root):
        el = entry.lower()
        if test_city not in el or res not in el:
            continue
        comp_path = os.path.join(output_root, entry, 'comparison_results.json')
        if os.path.isfile(comp_path):
            candidates.append((entry, comp_path))

    if not candidates:
        logger.info(f'       No completed experiments found for '
                    f'city={test_city} res={res}.')
        return None, None

    candidates.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)

    for entry, comp_path in candidates:
        logger.info(f'       Trying: {entry}')
        try:
            with open(comp_path) as f:
                data = json.load(f)
            baselines = data.get('baselines', {})
            valid = {k: v for k, v in baselines.items()
                     if isinstance(v, dict) and v.get('strict', {}).get('F1', 0) > 0}
            if valid:
                donor_self = data.get('sib', data.get('ddib', {}))
                logger.info(f'       ✓  Donor {entry}: '
                            f'{len(valid)} baseline(s) loaded')
                return valid, donor_self
            else:
                logger.info(f'       ~  {entry}: baselines empty/zero, skipping')
        except (json.JSONDecodeError, OSError) as e:
            logger.info(f'       ~  {entry}: could not load — {e}')

    logger.info('       No valid donor found.')
    return None, None


def _find_raw_prediction_dirs(args, logger):
    inf_root  = getattr(args, 'comparison_inference_dir', None)
    test_city = getattr(args, 'test_city', '')
    res       = args.resolution

    if not inf_root:
        logger.info('  [S2] --comparison_inference_dir not set; '
                    'skipping raw prediction lookup.')
        return {}

    raw_dirs = {
        'Upper Bound':    os.path.join(inf_root, 'upper', test_city, res, 'oglanet', 'base'),
        'LOCO Vanilla':   os.path.join(inf_root, 'loco',  test_city, res, 'oglanet', 'vanilla'),
        'LOCO FDA':       os.path.join(inf_root, 'loco',  test_city, res, 'oglanet', 'fda'),
        'LOCO SegDesic':  os.path.join(inf_root, 'loco',  test_city, res, 'oglanet', 'segdesic'),
        'LOCO IIM':       os.path.join(inf_root, 'loco',  test_city, res, 'oglanet', 'iim'),
        'LOCO ISW':       os.path.join(inf_root, 'loco',  test_city, res, 'oglanet', 'isw'),
        'LOCO MRFP+':     os.path.join(inf_root, 'loco',  test_city, res, 'oglanet', 'mrfp_plus'),
        'LOCO FADA':      os.path.join(inf_root, 'loco',  test_city, res, 'oglanet', 'fada'),
    }

    logger.info(f'\n  [S2] Checking raw prediction dirs under: {inf_root}')
    found = {}
    for label, pred_dir in raw_dirs.items():
        n = _dir_has_images(pred_dir)
        if n > 0:
            found[label] = pred_dir
            logger.info(f'       ✓  {label:<16} {pred_dir}  ({n} images)')
        else:
            reason = 'dir missing' if not os.path.isdir(pred_dir) else 'no images'
            logger.info(f'       ✗  {label:<16} {pred_dir}  ({reason})')

    return found


def _baseline_metrics_from_predictions(pred_dir, gt_dir, filenames,
                                        img_size, logger, tol_key,
                                        tolerance=2):
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def _stem_map(directory):
        m = {}
        if not os.path.isdir(directory):
            return m
        for fn in os.listdir(directory):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                m[os.path.splitext(fn)[0]] = os.path.join(directory, fn)
        return m

    pred_map = _stem_map(pred_dir)
    gt_map   = _stem_map(gt_dir)

    if not pred_map or not gt_map:
        return None

    allowed_stems = set()
    if filenames:
        for fn in filenames:
            allowed_stems.add(os.path.splitext(fn)[0])

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

        if pred_img.shape != (img_size, img_size):
            pred_img = cv2.resize(pred_img, sz, interpolation=cv2.INTER_NEAREST)
        if gt_img.shape != (img_size, img_size):
            gt_img   = cv2.resize(gt_img,   sz, interpolation=cv2.INTER_NEAREST)

        pred_bin = (pred_img > 127).astype(np.uint8)
        gt_bin   = (gt_img   > 127).astype(np.uint8)

        strict_list.append(_compute_strict_metrics(pred_bin, gt_bin))
        tolerant_list.append(
            _compute_tolerant_metrics(pred_bin, gt_bin, tolerance=tolerance))

    if not strict_list:
        return None

    return {
        'strict':        _average_metrics(strict_list),
        tol_key:         _average_metrics(tolerant_list) if tolerant_list else
                         _average_metrics([]),
        'n_images':      len(strict_list),
        'strict_list':   strict_list,
        'tolerant_list': tolerant_list,
    }


def compare_with_baselines(sib_strict, sib_tolerant,
                           sib_strict_list, sib_tolerant_list,
                           filenames, args, logger):
    test_city = getattr(args, 'test_city', 'unknown')
    res       = args.resolution
    img_size  = args.img_size
    tol_key = f'tolerant_{args.boundary_tolerance}px'

    gt_dir = None
    gt_candidates = []
    if args.comparison_data_root:
        gt_candidates = [
            os.path.join(args.comparison_data_root, test_city, res,
                         'test', 'masks'),
            os.path.join(args.comparison_data_root, test_city, res, 'masks'),
            os.path.join(args.comparison_data_root, 'test', 'masks'),
            os.path.join(args.comparison_data_root, 'masks'),
        ]
        for c in gt_candidates:
            if os.path.isdir(c):
                gt_dir = c
                break

    logger.info(f'\n{"="*70}')
    logger.info('BASELINE COMPARISON')
    logger.info(f'  Test city : {test_city}  |  Resolution: {res}')
    logger.info(f'  GT masks  : {gt_dir}')
    if gt_dir is None and gt_candidates:
        logger.info('  ⚠  GT mask dir not found. Tried:')
        for c in gt_candidates:
            logger.info(f'       {c}')
    logger.info(f'{"="*70}')

    baseline_results  = {}
    donor_self_metrics = None

    donor_baselines, donor_self = _find_oglanet_donor(args, logger)

    if donor_baselines:
        for label, bl_data in donor_baselines.items():
            if isinstance(bl_data, dict) and 'strict' in bl_data:
                baseline_results[label] = bl_data
        if donor_self and donor_self.get('strict', {}).get('F1', 0) > 0:
            donor_self_metrics = donor_self
        logger.info(f'\n  S1: {len(baseline_results)} baseline(s) loaded '
                    f'from OGLANet donor.')
        logger.info(f'  S1 baselines: {list(baseline_results.keys())}')
    else:
        logger.info('\n  S1: No OGLANet donor found — proceeding with S2 only.')

    raw_pred_dirs = _find_raw_prediction_dirs(args, logger)

    if raw_pred_dirs and gt_dir is None:
        logger.info('\n  S2: raw prediction dirs found but GT dir is missing — '
                    'cannot compute metrics from images.')
        logger.info('      Pass --comparison_data_root pointing at '
                    'Final_data_test to enable per-image metric computation.')
    elif raw_pred_dirs:
        logger.info(f'\n  S2: computing metrics for '
                    f'{len(raw_pred_dirs)} baseline(s)...')
        for label, pred_dir in raw_pred_dirs.items():
            bl = _baseline_metrics_from_predictions(
                pred_dir, gt_dir, filenames, img_size, logger, tol_key,
                tolerance=args.boundary_tolerance)
            if bl:
                baseline_results[label] = bl
                # Defensive lookup — never crash the print on tol-key mismatch
                tol_metrics = bl.get(tol_key, {})
                tol_f1      = tol_metrics.get('F1', 0.0)
                strict_f1   = bl.get('strict', {}).get('F1', 0.0)
                logger.info(f'  S2: ✓ {label} — {bl["n_images"]} images  '
                             f'(strict F1={strict_f1:.2f}  '
                             f'tol F1={tol_f1:.2f})')
            else:
                logger.info(f'  S2: ✗ {label} — no matching image pairs '
                             f'(pred={pred_dir}  gt={gt_dir})')

    if baseline_results:
        _print_comparison_table(
            'STRICT METRICS (all pixels)',
            baseline_results, sib_strict, sib_tolerant,
            donor_self_metrics, metric_type='strict', logger=logger)
        _print_comparison_table(
            f'TOLERANT METRICS (±{args.boundary_tolerance} px dont-care zone)',
            baseline_results, sib_strict, sib_tolerant,
            donor_self_metrics, metric_type=tol_key, logger=logger)
        _print_recovery_ratios(baseline_results, sib_strict, sib_tolerant,
                               logger, tol_key=tol_key)
        for bl_label in ['LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
                 'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA']:
            if (bl_label in baseline_results
                    and 'strict_list' in baseline_results[bl_label]):
                _print_bootstrap_comparison(
                    baseline_results[bl_label],
                    sib_strict_list, sib_tolerant_list,
                    baseline_label=bl_label, logger=logger)
    else:
        logger.info('\n  ⚠  No baselines available.  Check:')
        logger.info('       --comparison_inference_dir  (Test_img_results root)')
        logger.info('       --oglanet_output_dir        (for donor experiment)')
        logger.info('       --comparison_data_root      (GT masks root)')

    comp = {
        'test_city':  test_city,
        'resolution': res,
        'eval_size':  img_size,
        'sib':  {'strict': sib_strict, tol_key: sib_tolerant},
        'ddib': donor_self_metrics if donor_self_metrics else
                {'strict': sib_strict, tol_key: sib_tolerant},
        'baselines': {},
    }
    for label, br in baseline_results.items():
        comp['baselines'][label] = {
            'strict':       br.get('strict', {}),
            tol_key: br.get(tol_key, br.get('tolerant', {})),
        }
        if 'n_images' in br:
            comp['baselines'][label]['n_images'] = br['n_images']

    comp_path = os.path.join(args.output_dir, 'comparison_results.json')
    with open(comp_path, 'w') as f:
        json.dump(comp, f, indent=4)
    logger.info(f'\nComparison results saved → {comp_path}')
    return comp


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison table printing
# ═══════════════════════════════════════════════════════════════════════════════

def _print_comparison_table(title, baseline_results, sib_strict, sib_tolerant,
                             donor_self_metrics=None, metric_type='strict',
                             logger=None):
    sib_m = sib_strict if metric_type == 'strict' else sib_tolerant
    log   = logger.info if logger else print

    log(f'\n{"-"*70}')
    log(f'{title:^70}')
    log(f'{"-"*70}')
    log(f'  {"Method":<22} {"OA":>6} {"Prec":>6} {"Rec":>6} '
        f'{"F1":>6} {"BER":>6} {"mIOU":>6} {"ShIOU":>6}')
    log('  ' + '-' * 64)

    def _row(label, m):
        if not m:
            return
        log(f'  {label:<22} {m.get("OA", 0):6.2f} '
            f'{m.get("Precision", 0):6.2f} {m.get("Recall", 0):6.2f} '
            f'{m.get("F1", 0):6.2f} {m.get("BER", 0):6.2f} '
            f'{m.get("mIOU", 0):6.2f} {m.get("Shadow_IOU", 0):6.2f}')

    for label in ['Upper Bound', 'LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
              'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA']:
        if label not in baseline_results:
            continue
        _row(label, baseline_results[label].get(metric_type, {}))

    if donor_self_metrics:
        donor_m = donor_self_metrics.get(metric_type, {})
        if isinstance(donor_m, dict) and 'strict' in donor_m:
            donor_m = donor_m.get(metric_type, {})
        _row('Donor', donor_m)

    log('  ' + '-' * 64)
    _row('SIB (ours)', sib_m)


def _print_recovery_ratios(baseline_results, sib_strict, sib_tolerant, logger, tol_key='tolerant_2px'):
    if ('Upper Bound'  not in baseline_results
            or 'LOCO Vanilla' not in baseline_results):
        return
    log = logger.info
    log(f'\n{"-"*70}')
    log(f'{"RECOVERY RATIOS":^70}')
    log(f'  R = (SIB − LOCO_Vanilla) / (Upper − LOCO_Vanilla)')
    log(f'  0 = no help, 1 = gap fully closed')
    log(f'{"-"*70}')
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
        log(f'  {label:<10}  ' + '  '.join(parts))


def _print_bootstrap_comparison(loco_baseline, sib_strict_list,
                                 sib_tolerant_list, baseline_label,
                                 logger, n_bootstrap=5000):
    log = logger.info
    log(f'\n{"-"*70}')
    log(f'{"BOOTSTRAP: SIB vs " + baseline_label + " (n=5000)":^70}')
    log(f'{"-"*70}')
    np.random.seed(42)
    for eval_type, sib_list, label in [
            ('strict_list',   sib_strict_list,   'Strict'),
            ('tolerant_list', sib_tolerant_list, 'Tolerant')]:
        loco_list = loco_baseline.get(eval_type, [])
        n = min(len(loco_list), len(sib_list))
        if n == 0:
            continue
        log(f'\n  {label}:')
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
            log(f'    {k:<12} delta={obs_mean:+.2f}  '
                f'95%CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  p={p_val:.4f}{sig}')
    log('')


# ═══════════════════════════════════════════════════════════════════════════════
# Loss plotting — now includes CACR and CE-AURC panels
# ═══════════════════════════════════════════════════════════════════════════════

matplotlib.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman'] + matplotlib.rcParams['font.serif'],
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
    'figure.titlesize':  13,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

_BAND_DISPLAY = {
    'kl_ll':          'KL — LL (content)',
    'kl_lh':          'KL — LH (h-edge)',
    'kl_hl':          'KL — HL (v-edge)',
    'kl_hh':          'KL — HH (noise)',
    'kl_content':     'KL — Content (LL)',
    'kl_edge_lh':     'KL — Edge LH',
    'kl_edge_hl':     'KL — Edge HL',
    'kl_noise':       'KL — Noise (HH)',
    'kl_multiscale':  'KL — MultiScale',
    'kl_total':       'KL — total',
    'LL':             'KL — LL (content)',
    'LH':             'KL — LH (h-edge)',
    'HL':             'KL — HL (v-edge)',
    'HH':             'KL — HH (noise)',
    'bypass_alpha_mean': 'Bypass Gate α (mean)',
}

_BAND_COLS = [
    '#BC4749', '#FF6B35', '#8338EC', '#3A86FF',
    '#FFBE0B', '#FB5607', '#06D6A0', '#E63946',
]


def _band_label(key):
    return _BAND_DISPLAY.get(key, key.replace('_', ' ').title())


def plot_all_losses(history, output_dir, logger):
    if not history:
        logger.info('  No history to plot.')
        return

    epochs      = [h['epoch']           for h in history]
    train_total = [h['train_loss']      for h in history]
    train_task  = [h['train_task_loss'] for h in history]
    train_kl    = [h['train_kl_loss']   for h in history]
    val_total   = [h['val_loss']        for h in history]
    val_miou    = [h.get('val_mIOU', 0) for h in history]

    # NEW: CACR and CE-AURC histories (zero if not active)
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

    def _style_ax(ax, title, ylabel):
        ax.set_title(title, fontweight='bold', pad=6)
        ax.set_xlabel('Epoch')
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
        [(train_task, 'Train seg (CE)', dict(color=C_TASK,   ls='-',  marker='o')),
         (val_total,  'Val (proxy)',    dict(color=C_TASK_V, ls='--', marker='s'))],
    ))
    panels.append((
        'KL Loss (total)', 'KL',
        [(train_kl, 'KL total', dict(color=C_KL, ls='-', marker='^'))],
    ))
    for idx, bk in enumerate(all_band_keys):
        col = _BAND_COLS[idx % len(_BAND_COLS)]
        ylabel = 'α' if 'alpha' in bk else 'KL'
        panels.append((
            _band_label(bk), ylabel,
            [(band_series[bk], _band_label(bk), dict(color=col, ls='-', marker='o'))],
        ))
    # NEW: CACR loss panel (only if active)
    if any(v > 0 for v in train_cacr):
        panels.append((
            'CACR Loss', 'Loss',
            [(train_cacr, 'CACR', dict(color=C_CACR, ls='-', marker='D'))],
        ))
    # NEW: CE-AURC loss panel (only if active)
    if any(v > 0 for v in train_ce_aurc):
        panels.append((
            'CE-AURC Loss', 'Loss',
            [(train_ce_aurc, 'CE-AURC', dict(color=C_AURC, ls='-', marker='v'))],
        ))
    panels.append((
        'Val Decision mIOU', 'mIOU (%)',
        [(val_miou, 'Val mIOU', dict(color=C_MIOU, ls='-', marker='D'))],
    ))

    n_individual = len(panels)
    MAX_COLS     = 4
    n_rows_ind   = (n_individual + MAX_COLS - 1) // MAX_COLS
    fig_w = min(5.5 * MAX_COLS, 22)
    fig_h = 4.5 + 4.0 * n_rows_ind

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(
        1 + n_rows_ind, 1, figure=fig,
        hspace=0.55,
        height_ratios=[4.0] + [4.0] * n_rows_ind)

    ax_ov = fig.add_subplot(outer[0])
    ax_ov.plot(epochs, train_total, '-',  lw=2,   color=C_TRAIN, alpha=0.9,
               label='Train total')
    ax_ov.plot(epochs, val_total,   '--', lw=1.8, color=C_VAL,   alpha=0.9,
               label='Val total')
    ax_ov.plot(epochs, train_task,  '-',  lw=1.5, color=C_TASK,  alpha=0.75,
               label='Train task (CE)')
    ax_ov.plot(epochs, train_kl,    ls=(0, (3, 2)), lw=1.5, color=C_KL,
               alpha=0.65, label='KL total')
    for idx, bk in enumerate(all_band_keys):
        if 'alpha' in bk:
            continue  # different scale — skip from shared y-axis overview
        col = _BAND_COLS[idx % len(_BAND_COLS)]
        ax_ov.plot(epochs, band_series[bk], '-', lw=1.2, color=col,
                   alpha=0.7, label=_band_label(bk))
    # NEW: CACR / CE-AURC on overview (only if active)
    if any(v > 0 for v in train_cacr):
        ax_ov.plot(epochs, train_cacr, '-', lw=1.3, color=C_CACR,
                   alpha=0.8, label='CACR')
    if any(v > 0 for v in train_ce_aurc):
        ax_ov.plot(epochs, train_ce_aurc, '-', lw=1.3, color=C_AURC,
                   alpha=0.8, label='CE-AURC')
    ax_ov.set_title('Overview — All Losses (shared y-axis)',
                    fontweight='bold', pad=7)
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

        if title == 'Val Decision mIOU' and val_miou and max(val_miou) > 0:
            best_ep = epochs[int(np.argmax(val_miou))]
            best_v  = max(val_miou)
            ax.axvline(x=best_ep, color='red', ls=':', lw=1.2, alpha=0.7)
            ax.scatter([best_ep], [best_v], color='red', zorder=5, s=55,
                       label=f'Best ep={best_ep} ({best_v:.2f}%)')
            ax.legend(fontsize=8, framealpha=0.85)

        _style_ax(ax, title, ylabel)

    for pidx in range(n_individual, n_rows_ind * MAX_COLS):
        row = pidx // MAX_COLS
        col = pidx %  MAX_COLS
        fig.add_subplot(inner[row, col]).set_visible(False)

    fig.suptitle('OGLANet+SIB — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)

    path_main = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(path_main, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'  Saved loss_curves.png  ({n_individual + 1} panels)'
                f'  → {path_main}')

    # Per-band KL detail plot (exclude bypass alpha — it's not a KL loss)
    kl_band_keys = [bk for bk in all_band_keys if 'alpha' not in bk]
    if kl_band_keys:
        n_kl = len(kl_band_keys)
        n_kl_cols = min(n_kl, 4)
        n_kl_rows = (n_kl + n_kl_cols - 1) // n_kl_cols

        fig_kl, axes_kl = plt.subplots(
            n_kl_rows, n_kl_cols,
            figsize=(5.0 * n_kl_cols, 3.8 * n_kl_rows),
            squeeze=False)

        for idx, bk in enumerate(kl_band_keys):
            r, c  = divmod(idx, n_kl_cols)
            ax_kl = axes_kl[r][c]
            col   = _BAND_COLS[idx % len(_BAND_COLS)]
            # Add label so legend() doesn't warn about empty legend
            ax_kl.plot(epochs, band_series[bk], '-o', lw=2, ms=3.5,
                       color=col, mfc='white', mew=1.2,
                       label=_band_label(bk))
            ax_kl.set_title(_band_label(bk), fontweight='bold', pad=5)
            ax_kl.set_xlabel('Epoch')
            ax_kl.set_ylabel('KL Loss')
            ax_kl.grid(True, alpha=0.22, ls='--', lw=0.6)
            ax_kl.legend(fontsize=8, framealpha=0.85)

        for idx in range(n_kl, n_kl_rows * n_kl_cols):
            r, c = divmod(idx, n_kl_cols)
            axes_kl[r][c].set_visible(False)

        fig_kl.suptitle(
            'Per-Band KL Losses — Haar Wavelet Decomposition (Train)',
            fontweight='bold', fontsize=12, y=1.01)
        plt.tight_layout()
        path_kl = os.path.join(output_dir, 'loss_kl_detail.png')
        fig_kl.savefig(path_kl, dpi=250, bbox_inches='tight')
        plt.close(fig_kl)
        logger.info(f'  Saved loss_kl_detail.png ({n_kl} band panels)'
                    f'  → {path_kl}')


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_metrics(metrics):
    if isinstance(metrics, dict):
        return {k: _serialize_metrics(v) for k, v in metrics.items()}
    elif isinstance(metrics, torch.Tensor):
        return metrics.item()
    elif isinstance(metrics, (list, tuple)):
        return [_serialize_metrics(v) for v in metrics]
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    writer = SummaryWriter(args.output_dir)

    logger.info(f'Mode: {args.mode}')
    if args.mode == 'loco':
        logger.info(f'LOCO fold_id={args.fold_id} → holdout: {args.test_city}')
    logger.info(f'Device: {device}')
    logger.info(f'Args: {vars(args)}')

    # Log ablation flags explicitly
    logger.info(f'Ablation flags: skip_ll_vib={args.skip_ll_vib}  '
                f'symmetric_beta={args.symmetric_beta}  '
                f'aug_all_subbands={args.aug_all_subbands}  '
                f'vib_only_band={args.vib_only_band}')
    # NEW: log diagnostic-module flags
    logger.info(f'New modules: CACR={args.use_cacr} (w={args.cacr_weight}, '
                f'neg_w={args.cacr_neg_weight})  '
                f'CE-AURC={args.use_ce_aurc} (w={args.ce_aurc_weight})  '
                f'TENT={args.use_tent} (steps={args.tent_steps}, '
                f'lr={args.tent_lr})')

    # ── Data ──────────────────────────────────────────────────────────────
    if args.mode == 'loco':
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
    elif args.mode == 'all':
        data = get_dataloaders_sib(
            data_root=args.base_data_root,
            test_city=args.fold_names[0],
            resolution=args.resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            img_size=args.img_size,
            use_contrast=args.use_contrast,
            use_fda=args.use_fda,
            fda_L=args.fda_L,
        )
    elif args.mode == 'single':
        from data.dataset_sib import ShadowDatasetSIB
        from torch.utils.data import DataLoader
        train_ds = ShadowDatasetSIB(args.data_root, split='train',
                                     img_size=args.img_size,
                                     use_contrast=args.use_contrast)
        val_ds   = ShadowDatasetSIB(args.data_root, split='val',
                                     img_size=args.img_size,
                                     use_contrast=args.use_contrast)
        test_ds  = ShadowDatasetSIB(args.data_root, split='test',
                                     img_size=args.img_size,
                                     use_contrast=args.use_contrast)
        data = {
            'train_loader': DataLoader(train_ds, batch_size=args.batch_size,
                                       shuffle=True,  num_workers=args.num_workers,
                                       pin_memory=True, drop_last=True),
            'val_loader':   DataLoader(val_ds,   batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True),
            'test_loader':  DataLoader(test_ds,  batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True),
            'train_dataset': train_ds,
            'val_dataset':   val_ds,
            'test_dataset':  test_ds,
        }

    train_loader = data['train_loader']
    val_loader   = data['val_loader']
    test_loader  = data['test_loader']

    logger.info(f'Train: {len(data["train_dataset"])} samples  '
                f'Val: {len(data["val_dataset"])}  '
                f'Test: {len(data["test_dataset"])}')

    # ── Model ──────────────────────────────────────────────────────────────
    model = OGLANetSIB(
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        pretrained_encoder=args.pretrained_encoder,
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
        use_passthrough_gate=getattr(args, 'use_passthrough_gate', False),
        use_module_bypass=getattr(args, 'use_module_bypass', False),
        use_sag=args.use_sag,
        use_multiscale_sib=args.use_multiscale_sib,
        # ── Ablation flags ──
        skip_ll_vib=args.skip_ll_vib,
        symmetric_beta=args.symmetric_beta,
        aug_all_subbands=args.aug_all_subbands,
        vib_only_band=args.vib_only_band,
        # ── NEW: Diagnostic-motivated modules ──
        use_cacr=args.use_cacr,
        use_ce_aurc=args.use_ce_aurc,
        use_tent=args.use_tent,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {param_count:,}')
    logger.info(f'Module bypass gate: {getattr(args, "use_module_bypass", False)}')

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────
    criterion = OGLANetSIBLoss(
        num_classes=args.num_classes,
        lambda_kl=args.lambda_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
    ).to(device)

    # ── NEW: instantiate CACR and CE-AURC criteria if enabled ──────────
    cacr_criterion = None
    if args.use_cacr:
        cacr_criterion = CACRLoss(
            pos_weight=1.0,
            neg_weight=args.cacr_neg_weight)
        logger.info(f'CACR loss: weight={args.cacr_weight}, '
                    f'neg_weight={args.cacr_neg_weight}')

    ce_aurc_criterion = None
    if args.use_ce_aurc:
        ce_aurc_criterion = CEAURCLoss(
            floor_weight=args.ce_aurc_floor)
        logger.info(f'CE-AURC loss: weight={args.ce_aurc_weight}, '
                    f'floor={args.ce_aurc_floor}')

    param_groups = model.get_trainable_params(
        lr=args.lr, encoder_lr_mult=args.encoder_lr_mult)
    optimizer = Adamax(param_groups, lr=args.lr,
                       weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7,
        min_lr=1e-6)

    scaler    = GradScaler(enabled=args.use_amp)
    evaluator = DetailedEvaluator(boundary_tolerance=args.boundary_tolerance)

    # ── Training loop ──────────────────────────────────────────────────────
    best_miou        = 0.0
    patience_counter = 0
    history          = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f'Epoch {epoch}/{args.epochs}  '
                    f'KL weight: {criterion.kl_weight:.6f}')
        logger.info(f"{'='*60}")

        t0            = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, args, epoch, logger,
            cacr_criterion=cacr_criterion,
            cacr_weight=args.cacr_weight if args.use_cacr else 0.0,
            ce_aurc_criterion=ce_aurc_criterion,
            ce_aurc_weight=args.ce_aurc_weight if args.use_ce_aurc else 0.0)
        train_time    = time.time() - t0

        logger.info(f'Train: loss={train_metrics["total"]:.4f}  '
                    f'seg={train_metrics["task"]:.4f}  '
                    f'kl={train_metrics["kl"]:.6f}  '
                    f'time={train_time:.1f}s')
        # NEW: log CACR and CE-AURC averages
        if 'cacr' in train_metrics:
            cacr_d = train_metrics.get('cacr_diag', {})
            logger.info(f'  CACR={train_metrics["cacr"]:.6f}  '
                        f'pos_shift={cacr_d.get("cacr_pos_shift", 0):.4f}  '
                        f'n_pos={cacr_d.get("cacr_n_pos", 0):.0f}')
        if 'ce_aurc' in train_metrics:
            aurc_d = train_metrics.get('ce_aurc_diag', {})
            logger.info(f'  CE-AURC={train_metrics["ce_aurc"]:.6f}  '
                        f'mean_conf={aurc_d.get("ce_aurc_mean_shadow_conf", 0):.4f}  '
                        f'mean_ce={aurc_d.get("ce_aurc_mean_shadow_ce", 0):.4f}')

        bk = train_metrics['band_kl']
        if bk:
            band_str = '  '.join(
                f'{_band_label(k)}={v:.6f}' for k, v in sorted(bk.items()))
            logger.info(f'  KL bands: {band_str}')

        val_results = evaluate(model, val_loader, criterion, evaluator,
                               device, args, epoch)
        val_miou    = val_results['decision_miou']
        logger.info(f'Val: loss={val_results["loss"]:.4f}  '
                    f'decision_mIOU={val_miou:.4f}')

        val_m = val_results['metrics']
        if 'overall' in val_m:
            s = val_m['overall']
            logger.info(f'  Strict:   iou={s["iou"]:.2f}%  f1={s["f1"]:.2f}%')
        if ('boundary_tolerant' in val_m
                and f'tolerant_{args.boundary_tolerance}px' in val_m['boundary_tolerant']):
            t5 = val_m['boundary_tolerant'][f'tolerant_{args.boundary_tolerance}px']
            logger.info(f'  Tolerant: iou={t5["iou"]:.2f}%  f1={t5["f1"]:.2f}%')

        writer.add_scalar('train/loss',     train_metrics['total'], epoch)
        writer.add_scalar('train/seg_loss', train_metrics['task'],  epoch)
        writer.add_scalar('train/kl_loss',  train_metrics['kl'],    epoch)
        if 'cacr' in train_metrics:
            writer.add_scalar('train/cacr_loss', train_metrics['cacr'], epoch)
        if 'ce_aurc' in train_metrics:
            writer.add_scalar('train/ce_aurc_loss', train_metrics['ce_aurc'], epoch)
        for bk_name, bk_val in bk.items():
            writer.add_scalar(f'train/kl_{bk_name}', bk_val, epoch)
        writer.add_scalar('val/loss',          val_results['loss'], epoch)
        writer.add_scalar('val/decision_miou', val_miou,            epoch)

        scheduler.step(val_miou)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr}")

        is_best = val_miou > best_miou
        if is_best:
            best_miou        = val_miou
            patience_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou':            best_miou,
                'args':                 vars(args),
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            logger.info(f'  ★ New best mIOU: {best_miou:.4f}')
        else:
            patience_counter += 1
            logger.info(f'  No improvement '
                        f'({patience_counter}/{args.early_stopping_patience})')

        history_entry = {
            'epoch':           epoch,
            'train_loss':      train_metrics['total'],
            'train_task_loss': train_metrics['task'],
            'train_kl_loss':   train_metrics['kl'],
            'band_kl':         {k: float(v) for k, v in bk.items()},
            'val_loss':        val_results['loss'],
            'val_mIOU':        val_miou,
            'lr':              optimizer.param_groups[0]['lr'],
        }
        # NEW: save CACR/CE-AURC to history
        if 'cacr' in train_metrics:
            history_entry['train_cacr_loss'] = train_metrics['cacr']
            history_entry['cacr_diag'] = train_metrics.get('cacr_diag', {})
        if 'ce_aurc' in train_metrics:
            history_entry['train_ce_aurc_loss'] = train_metrics['ce_aurc']
            history_entry['ce_aurc_diag'] = train_metrics.get('ce_aurc_diag', {})
        history.append(history_entry)

        if patience_counter >= args.early_stopping_patience:
            logger.info(f'Early stopping at epoch {epoch}')
            break

    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    logger.info('\nGenerating loss plots...')
    plot_all_losses(history, args.output_dir, logger)

    logger.info(f"\n{'='*70}")
    if args.mode == 'loco':
        logger.info(f'Final Test on {args.test_city} (0-shot LOCO)')
    else:
        logger.info(f'Final Test (mode={args.mode})')
    if args.use_tent:
        logger.info(f'  TENT enabled: steps={args.tent_steps} lr={args.tent_lr}')
    logger.info(f"{'='*70}")

    best_ckpt = os.path.join(ckpt_dir, 'best_model.pth')
    if os.path.exists(best_ckpt):
        checkpoint = torch.load(best_ckpt, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f'Loaded best model from epoch {checkpoint["epoch"]}')

    (sib_strict, sib_tolerant,
     sib_strict_list, sib_tolerant_list,
     all_filenames) = test_and_save_predictions(
        model, test_loader, device, args, logger)
    
    # ── §4.3: Class-Conditional Temperature Scaling (post-hoc) ────────
    if args.use_class_cond_tempscale:
        logger.info(f'\n{"="*70}')
        logger.info(f'§4.3 MODULE: Class-Conditional Temperature Scaling')
        logger.info(f'{"="*70}')

        # Step 1: collect val logits (source-city, labeled)
        logger.info('  Collecting source-city validation logits...')
        val_logits, val_labels, _ = collect_logits_and_labels_oglanet(
            model, val_loader, device, args)
        val_logits = val_logits.to(device)
        val_labels = val_labels.to(device)
        logger.info(f'  Val: {val_logits.size(0)} images')

        # Step 2: fit T_pos, T_neg
        logger.info('  Fitting (T_pos, T_neg) via LBFGS...')
        T_pos, T_neg = fit_class_conditional_temperature(
            val_logits, val_labels, max_iter=args.tempscale_max_iter)
        logger.info(f'  → T_pos = {T_pos:.4f}, T_neg = {T_neg:.4f}')

        # Step 3: collect test logits, evaluate with and without tempscale
        logger.info('  Collecting test logits and computing SP-gap metrics...')
        test_logits, test_labels, test_fnames = collect_logits_and_labels_oglanet(
            model, test_loader, device, args)

        # Baseline (T=1.0): SP-gap reference
        sp_baseline = compute_sp_metrics(test_logits, test_labels)

        # Tempscale applied
        scaled_test_logits = apply_tempscale(test_logits, T_pos, T_neg)
        sp_tempscale = compute_sp_metrics(scaled_test_logits, test_labels)

        # Re-compute strict/tolerant on rescaled predictions and save PNGs
        ts_pred_dir = os.path.join(args.output_dir, 'predictions_tempscale')
        os.makedirs(ts_pred_dir, exist_ok=True)
        ts_preds = scaled_test_logits.argmax(dim=1).numpy().astype(np.uint8)
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
                'strict': sib_strict, tol_key: sib_tolerant,
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
        with open(os.path.join(args.output_dir, 'tempscale_results.json'), 'w') as f:
            json.dump(ts_summary, f, indent=4)

        logger.info(f'\n  Baseline (T=1.0):')
        logger.info(f'    mIoU={sib_strict["mIOU"]:.2f}  '
                    f'AURC_shadow={sp_baseline["aurc_shadow"]:.4f}  '
                    f'ECE_pos={sp_baseline["ece_pred_pos"]:.4f}')
        logger.info(f'  Tempscale (T_pos={T_pos:.3f}, T_neg={T_neg:.3f}):')
        logger.info(f'    mIoU={ts_strict["mIOU"]:.2f}  '
                    f'AURC_shadow={sp_tempscale["aurc_shadow"]:.4f}  '
                    f'ECE_pos={sp_tempscale["ece_pred_pos"]:.4f}')
        logger.info(f'  ΔAURC_shadow = {ts_summary["sp_gap_reduction"]["aurc_shadow"]:+.4f}')
        logger.info(f'  ΔECE_pos     = {ts_summary["sp_gap_reduction"]["ece_pred_pos"]:+.4f}')
        logger.info(f'  Saved → {os.path.join(args.output_dir, "tempscale_results.json")}')

    if args.mode == 'loco':
        compare_with_baselines(
            sib_strict, sib_tolerant,
            sib_strict_list, sib_tolerant_list,
            all_filenames, args, logger)
    else:
        comp = {
            'sib':      {'strict': sib_strict, f'tolerant_{args.boundary_tolerance}px': sib_tolerant},
            'ddib':     {'strict': sib_strict, f'tolerant_{args.boundary_tolerance}px': sib_tolerant},
            'baselines': {},
        }
        with open(os.path.join(args.output_dir,
                               'comparison_results.json'), 'w') as f:
            json.dump(comp, f, indent=4)

    logger.info(f'\nDone! Output: {args.output_dir}')
    writer.close()


if __name__ == '__main__':
    main()