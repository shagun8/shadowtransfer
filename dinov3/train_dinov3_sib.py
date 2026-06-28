"""
Training script for DINOv3 + SIB Shadow Detection.

Decision metrics (best checkpoint, early stopping) are based on
Tolerant mIOU when --eval_boundary_tolerant is enabled.

NEW (diagnostic-motivated additions, parity with MAMNet/OGLANet ports):
  --use_cacr               Enable Class-Asymmetric Confidence Regularizer
  --cacr_weight            CACR loss weight (default 0.1)
  --cacr_neg_weight        CACR weight for pred-negative pixels (default 0)
  --use_ce_aurc            Enable CE-AURC auxiliary loss on gt_shadow pixels
  --ce_aurc_weight         CE-AURC loss weight (default 0.01)
  --ce_aurc_floor          CE-AURC floor weight (default 0.5)
  --use_tent               Enable test-time entropy minimization on LN affine
  --tent_steps             TENT adaptation steps per batch (default 1)
  --tent_lr                TENT optimizer learning rate (default 0.001)
  --tent_pred_pos_only     TENT: focus entropy on pred-positive pixels
  --tent_use_ln            TENT: adapt LayerNorm affine (default True for ViT)
  --tent_use_bn            TENT: adapt BatchNorm affine (default False for ViT)

Usage examples:

  # D1: SIB-full (Haar + VIB + Aug + AdaptiveBeta)
  python train_dinov3_sib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --weights_path /path/to/dinov3_vits16.pth \
      --use_haar --use_vib --use_content_aug --adaptive_beta

  # N1: D1 + CACR (NEW)
  python train_dinov3_sib.py \
      --mode loco --fold_id 0 \
      --base_data_root /path/to/data --resolution highres \
      --weights_path /path/to/dinov3_vits16.pth \
      --use_haar --use_vib --use_content_aug --adaptive_beta \
      --use_cacr --cacr_weight 0.1
"""

import os
import argparse
import time
import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
import cv2

# Headless matplotlib (must come before pyplot)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_model_sib import DINOv3ShadowDetectorSIB
from data.dataset_sib import get_dataloaders_sib
from utils.losses import CrossEntropyLoss, CACRLoss, CEAURCLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.evaluation_detailed import DetailedEvaluator

# TENT helpers (test-time adaptation)
from sib import configure_tent, tent_adapt_step


# ======================================================================
# Constants
# ======================================================================

_AGGREGATE_KEYS = {'kl_total', 'kl_multiscale'}

# Keys in sib_losses that are NOT scalar losses (should not be summed
# into KL accumulators directly).  Extended for the new module flags so
# 'ref_logits' (CACR ref-path output) and 'pre_aug_bottleneck' (defensive
# — should normally be popped in the model wrapper) are skipped.
_NON_LOSS_KEYS = {'bypass_alpha', 'ref_logits', 'pre_aug_bottleneck'}

# ======================================================================
# Publication-quality matplotlib defaults
# ======================================================================

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
    'kl_ll':         'KL — LL (content)',
    'kl_lh':         'KL — LH (h-edge)',
    'kl_hl':         'KL — HL (v-edge)',
    'kl_hh':         'KL — HH (noise)',
    'kl_content':    'KL — Content',
    'kl_uniform':    'KL — Uniform',
    'kl_edge':       'KL — Edge',
    'kl_edge_lh':    'KL — Edge LH',
    'kl_edge_hl':    'KL — Edge HL',
    'kl_noise':      'KL — Noise',
    'kl_multiscale': 'KL — MultiScale',
    'kl_total':      'KL — total',
    'bypass_alpha_mean': 'Bypass Gate α (mean)',
}

_BAND_COLS = [
    '#BC4749', '#FF6B35', '#8338EC', '#3A86FF',
    '#FFBE0B', '#FB5607', '#06D6A0', '#E63946',
]


def _band_label(key):
    return _BAND_DISPLAY.get(key, key.replace('_', ' ').title())


# ======================================================================
# Arguments
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(description='Train DINOv3 + SIB')

    # -- Data --
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # -- Mode --
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities', type=str, nargs='+', default=None)

    # -- Backbone --
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--model_name', type=str, default='dinov3_vits16',
                   choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--weights_path', type=str, default=None)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--frozen_stages', type=int, default=-1)

    # -- SIB component toggles --
    p.add_argument('--use_haar', action='store_true', default=False,
                   help='Enable Haar wavelet decomposition')
    p.add_argument('--use_vib', action='store_true', default=False,
                   help='Enable VIB (differential when Haar on, '
                        'uniform when Haar off)')
    p.add_argument('--use_content_aug', action='store_true', default=False,
                   help='Enable content augmentation (training only)')
    p.add_argument('--adaptive_beta', action='store_true', default=False,
                   help='Enable intensity-adaptive beta in VIB')
    p.add_argument('--use_passthrough_gate', action='store_true', default=False,
                   help='Enable learned passthrough gate on VIB output '
                        '(allows VIB to auto-disable)')
    p.add_argument('--use_module_bypass', action='store_true', default=False,
                   help='Enable module-level residual bypass gate wrapping '
                        'entire SIB pipeline (allows per-sample bypass)')
    p.add_argument('--exp_tag', type=str, default='',
                   help='Experiment tag (e.g. D7, D10) prepended to output '
                        'folder name for disambiguation.')

    # -- SIB ablation flags (§5.3) --
    p.add_argument('--disable_content_vib', action='store_true', default=False,
                   help='A1 ablation: skip content VIB on F_LL, keep edge VIB')
    p.add_argument('--symmetric_vib', action='store_true', default=False,
                   help='A3 ablation: apply content-level VIB (high beta) '
                        'to LL, LH, HL subbands')
    p.add_argument('--aug_all_subbands', action='store_true', default=False,
                   help='A6 ablation: apply content augmentation to all '
                        'subbands, not just F_LL')
    p.add_argument('--vib_on_hl_only', action='store_true', default=False,
                   help='A10 ablation: apply content VIB to F_HL (wrong '
                        'subband) instead of F_LL')

    # -- SIB hyper-parameters --
    p.add_argument('--vib_beta_content', type=float, default=0.01,
                   help='Beta for content VIB (or uniform VIB when no Haar)')
    p.add_argument('--vib_beta_edge', type=float, default=0.0001,
                   help='Beta for edge VIB (only when Haar is on)')
    p.add_argument('--vib_beta_scale', type=float, default=0.02,
                   help='Adaptive beta range (only when adaptive_beta on)')
    p.add_argument('--lambda_content', type=float, default=1.0,
                   help='Weight for content/uniform VIB KL loss')
    p.add_argument('--lambda_edge', type=float, default=0.1,
                   help='Weight for edge VIB KL loss')
    p.add_argument('--aug_sigma_style', type=float, default=0.25)
    p.add_argument('--aug_sigma_shift', type=float, default=0.15)
    p.add_argument('--aug_p_aug', type=float, default=0.5)
    p.add_argument('--aug_p_mix', type=float, default=0.3)

    # -- VIB warmup --
    p.add_argument('--vib_warmup_fraction', type=float, default=0.1,
                   help='Fraction of training epochs for VIB warmup '
                        '(0 → target linearly). Default 0.1 = 10%%.')

    # -- Training --
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--min_lr', type=float, default=1e-6)

    # -- FDA --
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L', type=float, default=0.01)

    # -- Checkpoint / logging --
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq', type=int, default=5)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--early_stopping_patience', type=int, default=5)

    p.add_argument('--boundary_tolerance', type=int, default=2,
                   help='±K px don\'t-care zone for boundary-tolerant '
                        'evaluation. Passed from bash via BOUNDARY_TOLERANCE.')

    # -- Comparison baselines --
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root', type=str, default=None)
    p.add_argument('--dinov3_output_dir', type=str, default=None,
                   help='DINOv3 outputs root scanned for a completed donor '
                        'experiment whose comparison_results.json can supply '
                        'pre-computed baseline metrics (Strategy 1).')

    # ══════════════════════════════════════════════════════════════════
    # NEW: Diagnostic-motivated module flags (CACR, CE-AURC, TENT)
    # ══════════════════════════════════════════════════════════════════
    p.add_argument('--use_cacr', action='store_true', default=False,
                   help='Enable CACR (Class-Asymmetric Confidence Regularizer)')
    p.add_argument('--cacr_weight', type=float, default=0.1,
                   help='CACR loss weight (default 0.1)')
    p.add_argument('--cacr_neg_weight', type=float, default=0.0,
                   help='CACR weight for pred-negative pixels (default 0)')

    p.add_argument('--use_ce_aurc', action='store_true', default=False,
                   help='Enable CE-AURC auxiliary loss on gt_shadow pixels')
    p.add_argument('--ce_aurc_weight', type=float, default=0.01,
                   help='CE-AURC loss weight (default 0.01)')
    p.add_argument('--ce_aurc_floor', type=float, default=0.5,
                   help='CE-AURC floor weight (default 0.5)')

    p.add_argument('--use_tent', action='store_true', default=False,
                   help='Enable TENT (test-time entropy minimization)')
    p.add_argument('--tent_steps', type=int, default=1,
                   help='TENT adaptation steps per batch (default 1)')
    p.add_argument('--tent_lr', type=float, default=0.001,
                   help='TENT optimizer learning rate (default 0.001)')
    p.add_argument('--tent_pred_pos_only', action='store_true', default=True,
                   help='TENT: focus entropy on pred-positive pixels')
    p.add_argument('--tent_use_ln', action='store_true', default=True,
                   help='TENT: adapt LayerNorm affine (default True for ViT)')
    p.add_argument('--tent_use_bn', action='store_true', default=False,
                   help='TENT: adapt BatchNorm affine (default False for ViT; '
                        'set True for CNN backbones)')
    
    # §4.3 module: Class-conditional temperature scaling
    p.add_argument('--use_class_cond_tempscale', action='store_true',
                   default=False,
                   help='Fit T_pos/T_neg on source-city val, apply at test')
    p.add_argument('--tempscale_max_iter', type=int, default=200)

    return p.parse_args()


# ======================================================================
# LR schedule
# ======================================================================

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ======================================================================
# Per-image metric functions
# ======================================================================

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

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou = (shadow_iou + nonshadow_iou) / 2
    oa = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _compute_tolerant_metrics(pred, gt, tolerance=2):
    kernel = _get_tolerance_kernel(tolerance)
    gt_uint8 = gt.astype(np.uint8)
    eroded  = cv2.erode(gt_uint8, kernel)
    dilated = cv2.dilate(gt_uint8, kernel)
    band = (dilated - eroded) > 0
    valid = ~band

    p = pred[valid]
    g = gt[valid]

    tp = np.logical_and(p == 1, g == 1).sum()
    fp = np.logical_and(p == 1, g == 0).sum()
    tn = np.logical_and(p == 0, g == 0).sum()
    fn = np.logical_and(p == 0, g == 1).sum()

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou = (shadow_iou + nonshadow_iou) / 2
    total = tp + tn + fp + fn
    oa = (tp + tn) / (total + 1e-10)

    shadow_err    = fn / (tp + fn + 1e-10) if (tp + fn) > 0 else 0
    nonshadow_err = fp / (tn + fp + 1e-10) if (tn + fp) > 0 else 0
    ber = (shadow_err + nonshadow_err) / 2

    return {
        'OA': float(oa * 100), 'Precision': float(precision * 100),
        'Recall': float(recall * 100), 'F1': float(f1 * 100),
        'BER': float(ber * 100), 'mIOU': float(miou * 100),
        'Shadow_IOU': float(shadow_iou * 100),
    }


def _average_metrics(metrics_list):
    if not metrics_list:
        return {k: 0.0 for k in
                ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU',
                 'Shadow_IOU']}
    keys = ['OA', 'Precision', 'Recall', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


# ════════════════════════════════════════════════════════════════════════════
# §4.3 MODULE: Class-Conditional Temperature Scaling
#   Fit T_pos and T_neg on source-city validation logits (no target labels),
#   apply at inference on held-out city. Reduces SP-gap without target data.
#   Reference: Guo et al. (ICML 2017), Tian et al. (CVPR 2023).
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_logits_and_labels_dinov3(model, loader, device):
    """Run inference on `loader`, return concatenated CPU logits + labels."""
    model.eval()
    all_logits, all_labels, all_filenames = [], [], []
    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['mask'].to(device)
        intensity_map = batch['intensity_map'].to(device)
        city_ids = batch.get('city_id', None)
        if city_ids is not None:
            city_ids = city_ids.to(device)

        logits, _ = model(images, intensity_map, city_ids=city_ids,
                          vib_warmup_factor=1.0)  # [B, 2, H, W]
        all_logits.append(logits.float().cpu())
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


# ======================================================================
# Baseline comparison helpers
# ======================================================================

def _dir_has_images(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path)
               if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff', '.jpeg')))


def _find_dinov3_donor(output_root, test_city, res):
    if not output_root or not os.path.isdir(output_root):
        print(f'  [S1] dinov3_output_dir not set or missing: {output_root}')
        return None, None

    print(f'\n  [S1] Scanning for DINOv3 donor in: {output_root}')
    candidates = []
    for entry in os.listdir(output_root):
        el = entry.lower()
        if test_city not in el or res not in el:
            continue
        comp_path = os.path.join(output_root, entry, 'comparison_results.json')
        if os.path.isfile(comp_path):
            candidates.append((entry, comp_path))

    if not candidates:
        print(f'       No completed experiments found for '
              f'city={test_city} res={res}.')
        return None, None

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
                donor_self = data.get('sib', data.get('ddib', {}))
                print(f'       ✓  Donor {entry}: {len(valid)} baseline(s) loaded')
                return valid, donor_self
            else:
                print(f'       ~  {entry}: baselines empty/zero, skipping')
        except (json.JSONDecodeError, OSError) as e:
            print(f'       ~  {entry}: could not load — {e}')

    print('       No valid donor found.')
    return None, None


def _find_raw_prediction_dirs(inf_root, test_city, res):
    if not inf_root:
        print('  [S2] --comparison_inference_dir not set; skipping.')
        return {}

    raw_dirs = {
        'Upper Bound':    os.path.join(inf_root, 'upper', test_city, res, 'dinov3', 'base'),
        'LOCO Vanilla':   os.path.join(inf_root, 'loco',  test_city, res, 'dinov3', 'vanilla'),
        'LOCO FDA':       os.path.join(inf_root, 'loco',  test_city, res, 'dinov3', 'fda'),
        'LOCO SegDesic':  os.path.join(inf_root, 'loco',  test_city, res, 'dinov3', 'segdesic'),
        'LOCO IIM':       os.path.join(inf_root, 'loco',  test_city, res, 'dinov3', 'iim'),
        'LOCO ISW':       os.path.join(inf_root, 'loco',  test_city, res, 'dinov3', 'isw'),
        'LOCO MRFP+':     os.path.join(inf_root, 'loco',  test_city, res, 'dinov3', 'mrfp_plus'),
        'LOCO FADA':      os.path.join(inf_root, 'loco',  test_city, res, 'dinov3', 'fada'),
    }

    print(f'\n  [S2] Checking raw prediction dirs under: {inf_root}')
    found = {}
    for label, pred_dir in raw_dirs.items():
        n = _dir_has_images(pred_dir)
        if n > 0:
            found[label] = pred_dir
            print(f'       ✓  {label:<18} {pred_dir}  ({n} images)')
        else:
            reason = 'dir missing' if not os.path.isdir(pred_dir) else 'no images'
            print(f'       ✗  {label:<18} {pred_dir}  ({reason})')

    return found


def _baseline_metrics_from_predictions(pred_dir, gt_dir, filenames,
                                        img_size, tol_key, boundary_tolerance):
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

    allowed_stems = {os.path.splitext(fn)[0] for fn in filenames} if filenames else set()

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
            gt_img   = cv2.resize(gt_img, sz, interpolation=cv2.INTER_NEAREST)

        pred_bin = (pred_img > 127).astype(np.uint8)
        gt_bin   = (gt_img   > 127).astype(np.uint8)

        strict_list.append(_compute_strict_metrics(pred_bin, gt_bin))
        tolerant_list.append(
            _compute_tolerant_metrics(pred_bin, gt_bin,
                                      tolerance=boundary_tolerance))

    if not strict_list:
        return None

    return {
        'strict':        _average_metrics(strict_list),
        tol_key:         _average_metrics(tolerant_list),
        'n_images':      len(strict_list),
        'strict_list':   strict_list,
        'tolerant_list': tolerant_list,
    }


# ======================================================================
# Comprehensive loss visualisation — extended with CACR & CE-AURC panels
# ======================================================================

def plot_all_losses(history, output_dir):
    if not history:
        print('  No history to plot.')
        return

    epochs      = [h['epoch']           for h in history]
    train_total = [h['train_loss']      for h in history]
    train_task  = [h['train_task_loss'] for h in history]
    train_kl    = [h['train_kl_loss']   for h in history]
    val_total   = [h['val_loss']        for h in history]
    val_miou    = [h.get('val_mIOU', 0) for h in history]

    # NEW: CACR / CE-AURC histories (zero if not active)
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
        # Skip bypass_alpha_mean from the individual KL panels — it gets
        # its own dedicated panel below and is on a different scale.
        if 'alpha' in bk:
            continue
        col = _BAND_COLS[idx % len(_BAND_COLS)]
        panels.append((
            _band_label(bk), 'KL',
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

    # Dedicated bypass alpha panel (if present)
    if 'bypass_alpha_mean' in all_band_keys:
        panels.append((
            'Bypass Gate α (mean)', 'α',
            [(band_series['bypass_alpha_mean'], 'α',
              dict(color='#E63946', ls='-', marker='s'))],
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
        # Skip bypass_alpha from shared-axis overview (different scale)
        if 'alpha' in bk:
            continue
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

    fig.suptitle('DINOv3+SIB — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)

    path_main = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(path_main, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved loss_curves.png  ({n_individual + 1} panels)  → {path_main}')

    # Detailed KL-only plot (exclude bypass alpha — it's not a KL loss)
    kl_band_keys = [bk for bk in all_band_keys if 'alpha' not in bk]
    if kl_band_keys:
        n_kl      = len(kl_band_keys)
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
            'Per-Band KL Losses — SIB (Train)',
            fontweight='bold', fontsize=12, y=1.01)
        plt.tight_layout()
        path_kl = os.path.join(output_dir, 'loss_kl_detail.png')
        fig_kl.savefig(path_kl, dpi=250, bbox_inches='tight')
        plt.close(fig_kl)
        print(f'  Saved loss_kl_detail.png ({n_kl} band panels)  → {path_kl}')


# ======================================================================
# Trainer
# ======================================================================

_LOCO_BASELINE_LABELS = [
    'LOCO Vanilla',
    'LOCO FDA',
    'LOCO SegDesic',
    'LOCO IIM',
    'LOCO ISW',
    'LOCO MRFP+',
    'LOCO FADA',
]

_TABLE_ORDER = ['Upper Bound'] + _LOCO_BASELINE_LABELS


class TrainerSIB:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')

        # ---- Output dir ----
        sib_tag = ''
        if args.use_haar:              sib_tag += '_haar'
        if args.use_vib:               sib_tag += '_vib'
        if args.use_content_aug:       sib_tag += '_aug'
        if args.adaptive_beta:         sib_tag += '_ab'
        if args.use_passthrough_gate:  sib_tag += '_gate'
        if args.use_module_bypass:     sib_tag += '_bypass'
        if not sib_tag:                sib_tag = '_noSIB'
        if args.disable_content_vib:  sib_tag += '_noConVIB'
        if args.symmetric_vib:        sib_tag += '_symVIB'
        if args.aug_all_subbands:     sib_tag += '_augAll'
        if args.vib_on_hl_only:       sib_tag += '_vibHL'
        if args.exp_tag:
            sib_tag = f'_{args.exp_tag}{sib_tag}'

        # Ablation flags
        disable_content_vib=args.disable_content_vib,
        symmetric_vib=args.symmetric_vib,
        aug_all_subbands=args.aug_all_subbands,
        vib_on_hl_only=args.vib_on_hl_only,


        fda_suffix = '_fda' if args.use_fda else ''

        if args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = (f'dinov3_sib{sib_tag}{fda_suffix}'
                        f'_loco_holdout_{test_city}_{args.resolution}_1')
        elif args.mode == 'all':
            exp_name = (f'dinov3_sib{sib_tag}{fda_suffix}'
                        f'_all_{args.resolution}_1')
        else:
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = f'dinov3_sib{sib_tag}{fda_suffix}_{city}_{res}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ---- Data ----
        print('\nLoading datasets …')
        loaders = get_dataloaders_sib(
            data_root=args.data_root,
            base_data_root=args.base_data_root,
            mode=args.mode, cities=args.cities,
            resolution=args.resolution, fold_id=args.fold_id,
            batch_size=args.batch_size, num_workers=args.num_workers,
            img_size=args.img_size,
            use_fda=args.use_fda,
            fda_target_root=args.fda_target_root, fda_L=args.fda_L)
        self.dataloaders = loaders
        num_domains = loaders['num_domains']
        print(f'Train: {len(loaders["train"].dataset)}  |  '
              f'Val: {len(loaders["val"].dataset)}  |  '
              f'Test: {len(loaders["test"].dataset)}')

        # ---- Model ----
        # Bug fix: actually pass the ablation flags + new diagnostic
        # module flags into the model constructor.  The orphaned
        # `disable_content_vib=args.disable_content_vib,` etc. tuple-
        # creating lines above are preserved as-is for traceability,
        # but the model now receives them properly via this call.
        print('\nBuilding model …')
        self.model = DINOv3ShadowDetectorSIB(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            use_haar=args.use_haar,
            use_vib=args.use_vib,
            use_content_aug=args.use_content_aug,
            adaptive_beta=args.adaptive_beta,
            use_passthrough_gate=args.use_passthrough_gate,
            use_module_bypass=args.use_module_bypass,
            # Ablation flags (now properly forwarded)
            disable_content_vib=args.disable_content_vib,
            symmetric_vib=args.symmetric_vib,
            aug_all_subbands=args.aug_all_subbands,
            vib_on_hl_only=args.vib_on_hl_only,
            num_domains=num_domains,
            vib_beta_content=args.vib_beta_content,
            vib_beta_edge=args.vib_beta_edge,
            vib_beta_scale=args.vib_beta_scale,
            aug_sigma_style=args.aug_sigma_style,
            aug_sigma_shift=args.aug_sigma_shift,
            aug_p_aug=args.aug_p_aug,
            aug_p_mix=args.aug_p_mix,
            # NEW: diagnostic-motivated module flags
            use_cacr=args.use_cacr,
            use_ce_aurc=args.use_ce_aurc,
            use_tent=args.use_tent,
        ).to(self.device)

        # ---- Loss, optimiser, scheduler ----
        self.criterion = CrossEntropyLoss()

        # NEW: diagnostic-motivated auxiliary losses
        self.cacr_criterion = None
        if args.use_cacr:
            self.cacr_criterion = CACRLoss(
                pos_weight=1.0,
                neg_weight=args.cacr_neg_weight)
            print(f'CACR loss: weight={args.cacr_weight}, '
                  f'neg_weight={args.cacr_neg_weight}')

        self.ce_aurc_criterion = None
        if args.use_ce_aurc:
            self.ce_aurc_criterion = CEAURCLoss(
                floor_weight=args.ce_aurc_floor)
            print(f'CE-AURC loss: weight={args.ce_aurc_weight}, '
                  f'floor={args.ce_aurc_floor}')

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999))
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, args.warmup_epochs, args.epochs,
            args.lr, args.min_lr)

        # ---- VIB warmup ----
        self.vib_warmup_epochs = int(args.vib_warmup_fraction * args.epochs)
        if args.use_vib:
            print(f'VIB warmup: {self.vib_warmup_epochs} epochs '
                  f'({args.vib_warmup_fraction * 100:.0f}%)')

        # ---- Decision metric ----
        self.use_tolerant_for_decisions = args.eval_boundary_tolerant
        if self.use_tolerant_for_decisions:
            print(f'>>> Decision metric: Tolerant mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print('>>> Decision metric: Strict mIOU')

        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        if args.eval_boundary_tolerant:
            self.detailed_eval_train = DetailedEvaluator(
                boundary_tolerance=args.boundary_tolerance)
            self.detailed_eval_val   = DetailedEvaluator(
                boundary_tolerance=args.boundary_tolerance)

        # ---- Tracking ----
        self.start_epoch = 0
        self.best_miou = 0.0
        self.best_strict_miou = 0.0
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        self.epochs_without_improvement = 0

        self.train_losses = []
        self.val_losses = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [],
            'mIOU': [], 'Shadow_IOU': []}
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [],
            'mIOU': [], 'Shadow_IOU': []}

        self.history = []

        if args.resume:
            self._load_checkpoint(args.resume)

    # ------------------------------------------------------------------
    def _get_vib_warmup_factor(self, epoch):
        if self.vib_warmup_epochs <= 0 or not self.args.use_vib:
            return 1.0
        if epoch <= self.vib_warmup_epochs:
            return float(epoch) / float(self.vib_warmup_epochs)
        return 1.0

    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        epoch_seg = 0.0
        epoch_kl_content = 0.0
        epoch_kl_edge = 0.0
        epoch_cacr = 0.0
        epoch_ce_aurc = 0.0
        cacr_diag_accum = defaultdict(float)
        ce_aurc_diag_accum = defaultdict(float)

        metrics = ShadowMetrics()
        n_batches = len(self.dataloaders['train'])
        t0 = time.time()

        vib_warmup_factor = self._get_vib_warmup_factor(epoch)

        band_kl_accum = defaultdict(float)

        for i, batch in enumerate(self.dataloaders['train']):
            images   = batch['image'].to(self.device)
            masks    = batch['mask'].to(self.device)
            int_map  = batch['intensity_map'].to(self.device)
            city_ids = batch['city_id'].to(self.device)

            outputs, sib_losses = self.model(
                images, int_map, city_ids,
                vib_warmup_factor=vib_warmup_factor)

            seg_loss = self.criterion(outputs, masks)
            total = seg_loss

            kl_content_val = sib_losses.get(
                'kl_content', sib_losses.get('kl_uniform',
                                             torch.tensor(0.0)))
            kl_edge_val = sib_losses.get('kl_edge', torch.tensor(0.0))

            if self.args.use_vib:
                total = total + self.args.lambda_content * kl_content_val
                total = total + self.args.lambda_edge * kl_edge_val

            # ── NEW: CACR loss ───────────────────────────────────────
            cacr_loss_val = torch.tensor(0.0, device=self.device)
            if (self.cacr_criterion is not None
                    and 'ref_logits' in sib_losses):
                # outputs = main path logits, sib_losses['ref_logits'] = ref
                # ref_logits already detached and produced under no_grad
                # in the model wrapper.
                cacr_loss_val, cacr_diag = self.cacr_criterion(
                    outputs, sib_losses['ref_logits'].detach(),
                    targets=masks)
                for k, v in cacr_diag.items():
                    cacr_diag_accum[k] += v
                total = total + self.args.cacr_weight * cacr_loss_val

            # ── NEW: CE-AURC auxiliary loss ──────────────────────────
            ce_aurc_loss_val = torch.tensor(0.0, device=self.device)
            if self.ce_aurc_criterion is not None:
                ce_aurc_loss_val, aurc_diag = self.ce_aurc_criterion(
                    outputs, masks)
                for k, v in aurc_diag.items():
                    ce_aurc_diag_accum[k] += v
                total = total + self.args.ce_aurc_weight * ce_aurc_loss_val

            for band_key, band_val in sib_losses.items():
                if band_key in _NON_LOSS_KEYS:
                    continue
                if (isinstance(band_val, torch.Tensor)
                        and band_key not in _AGGREGATE_KEYS):
                    band_kl_accum[band_key] += band_val.item()

            # Track bypass alpha mean separately (not a loss)
            if 'bypass_alpha' in sib_losses:
                band_kl_accum['bypass_alpha_mean'] += \
                    sib_losses['bypass_alpha'].detach().mean().item()

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()

            filtered = filter_small_predictions(outputs.detach(),
                                                min_pixels=10)
            metrics.update(filtered, masks)

            if self.args.eval_boundary_tolerant:
                preds = torch.argmax(outputs.detach(), dim=1)
                self.detailed_eval_train.update(preds, masks, images)

            epoch_loss += total.item()
            epoch_seg += seg_loss.item()
            epoch_kl_content += kl_content_val.item()
            epoch_kl_edge += kl_edge_val.item()
            epoch_cacr += cacr_loss_val.item()
            epoch_ce_aurc += ce_aurc_loss_val.item()

            if (i + 1) % 10 == 0 or (i + 1) == n_batches:
                alpha_str = ''
                if 'bypass_alpha' in sib_losses:
                    alpha_str = (
                        f'  α={sib_losses["bypass_alpha"].detach().mean().item():.3f}'
                    )
                extra_str = ''
                if self.cacr_criterion is not None:
                    extra_str += f'  cacr={cacr_loss_val.item():.6f}'
                if self.ce_aurc_criterion is not None:
                    extra_str += f'  ceaurc={ce_aurc_loss_val.item():.6f}'
                print(f'  [{i+1}/{n_batches}]  loss={total.item():.4f}  '
                      f'seg={seg_loss.item():.4f}  '
                      f'kl_c={kl_content_val.item():.6f}  '
                      f'kl_e={kl_edge_val.item():.6f}  '
                      f'vib_wu={vib_warmup_factor:.2f}{alpha_str}{extra_str}')

        epoch_loss       /= n_batches
        epoch_seg        /= n_batches
        epoch_kl_content /= n_batches
        epoch_kl_edge    /= n_batches
        epoch_cacr       /= n_batches
        epoch_ce_aurc    /= n_batches
        avg_band_kl       = {k: v / n_batches for k, v in band_kl_accum.items()}
        avg_cacr_diag     = {k: v / n_batches for k, v in cacr_diag_accum.items()}
        avg_ce_aurc_diag  = {k: v / n_batches for k, v in ce_aurc_diag_accum.items()}

        m = metrics.compute()
        print(f'\nEpoch {epoch} train  ({time.time()-t0:.1f}s)')
        print(f'  loss={epoch_loss:.4f}  seg={epoch_seg:.4f}  '
              f'kl_c={epoch_kl_content:.6f}  kl_e={epoch_kl_edge:.6f}')
        print(f'  OA={m["OA"]:.2f}  P={m["Precision"]:.2f}  '
              f'F1={m["F1"]:.2f}  BER={m["BER"]:.2f}  '
              f'mIOU={m["mIOU"]:.2f}  ShIOU={m["Shadow_IOU"]:.2f}')
        if 'bypass_alpha_mean' in avg_band_kl:
            print(f'  Bypass α (mean): {avg_band_kl["bypass_alpha_mean"]:.4f}')
        if self.cacr_criterion is not None:
            print(f'  CACR={epoch_cacr:.6f}  '
                  f'pos_shift={avg_cacr_diag.get("cacr_pos_shift", 0):.4f}  '
                  f'n_pos={avg_cacr_diag.get("cacr_n_pos", 0):.0f}')
        if self.ce_aurc_criterion is not None:
            print(f'  CE-AURC={epoch_ce_aurc:.6f}  '
                  f'mean_conf={avg_ce_aurc_diag.get("ce_aurc_mean_shadow_conf", 0):.4f}  '
                  f'mean_ce={avg_ce_aurc_diag.get("ce_aurc_mean_shadow_ce", 0):.4f}')

        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Train/Seg_Loss', epoch_seg, epoch)
        self.writer.add_scalar('Train/KL_Content', epoch_kl_content, epoch)
        self.writer.add_scalar('Train/KL_Edge', epoch_kl_edge, epoch)
        if self.cacr_criterion is not None:
            self.writer.add_scalar('Train/CACR', epoch_cacr, epoch)
        if self.ce_aurc_criterion is not None:
            self.writer.add_scalar('Train/CE_AURC', epoch_ce_aurc, epoch)
        self.writer.add_scalar('Train/VIB_Warmup', vib_warmup_factor, epoch)
        for k in m:
            self.writer.add_scalar(f'Train/{k}', m[k], epoch)
        for bk, bv in avg_band_kl.items():
            self.writer.add_scalar(f'Train/KL_band_{bk}', bv, epoch)

        self.train_losses.append(epoch_loss)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(m[k])

        if self.args.eval_boundary_tolerant:
            dr = self.detailed_eval_train.compute_metrics()
            t_tol = dr['boundary_tolerant'][self.tol_key]
            self.writer.add_scalar('Train/F1_Tolerant',   t_tol['f1'],  epoch)
            self.writer.add_scalar('Train/mIOU_Tolerant', t_tol['iou'], epoch)
            self.detailed_eval_train.reset()
            print(f'  Boundary-Tolerant (±{self.args.boundary_tolerance}px): '
                  f'F1={t_tol["f1"]:.2f}  mIOU={t_tol["iou"]:.2f}')

        return (epoch_loss, m, epoch_seg, epoch_kl_content, epoch_kl_edge,
                avg_band_kl, epoch_cacr, epoch_ce_aurc,
                avg_cacr_diag, avg_ce_aurc_diag)

    # ------------------------------------------------------------------
    def validate(self, epoch):
        print('\nValidating …')
        self.model.eval()
        val_loss = 0.0
        metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images  = batch['image'].to(self.device)
                masks   = batch['mask'].to(self.device)
                int_map = batch['intensity_map'].to(self.device)

                outputs, _ = self.model(images, int_map)
                val_loss += self.criterion(outputs, masks).item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                metrics.update(filtered, masks)

                if self.args.eval_boundary_tolerant:
                    preds = torch.argmax(outputs, dim=1)
                    self.detailed_eval_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        m = metrics.compute()
        print(f'Val  loss={val_loss:.4f}')
        print(f'  OA={m["OA"]:.2f}  P={m["Precision"]:.2f}  '
              f'F1={m["F1"]:.2f}  BER={m["BER"]:.2f}  '
              f'mIOU={m["mIOU"]:.2f}  ShIOU={m["Shadow_IOU"]:.2f}')

        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for k in m:
            self.writer.add_scalar(f'Val/{k}', m[k], epoch)

        self.val_losses.append(val_loss)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(m[k])

        decision_miou = m['mIOU']

        if self.args.eval_boundary_tolerant:
            dr = self.detailed_eval_val.compute_metrics()
            t_tol = dr['boundary_tolerant'][self.tol_key]
            self.writer.add_scalar('Val/F1_Tolerant',   t_tol['f1'],  epoch)
            self.writer.add_scalar('Val/mIOU_Tolerant', t_tol['iou'], epoch)
            self.detailed_eval_val.reset()
            print(f'  Boundary-Tolerant (±{self.args.boundary_tolerance}px): '
                  f'F1={t_tol["f1"]:.2f}  mIOU={t_tol["iou"]:.2f}')
            decision_miou = t_tol['iou']

        return val_loss, m, decision_miou

    # ------------------------------------------------------------------
    def train(self):
        print('\n' + '=' * 60)
        print('Starting SIB training')
        print('=' * 60)

        metric_label = (f'Tolerant mIOU (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict mIOU')

        for epoch in range(self.start_epoch, self.args.epochs):
            lr = self.scheduler.step(epoch)
            print(f'\n{"="*60}\nEpoch {epoch+1}/{self.args.epochs}  '
                  f'lr={lr:.2e}')

            (train_loss, _, train_seg, train_kl_c, train_kl_e, band_kl,
             train_cacr, train_ce_aurc, cacr_diag, ce_aurc_diag
             ) = self.train_epoch(epoch + 1)

            val_loss, val_m, decision_miou = self.validate(epoch + 1)

            history_entry = {
                'epoch':           epoch + 1,
                'train_loss':      train_loss,
                'train_task_loss': train_seg,
                'train_kl_loss':   train_kl_c + train_kl_e,
                'band_kl':         {k: float(v) for k, v in band_kl.items()},
                'val_loss':        val_loss,
                'val_mIOU':        float(decision_miou),
                'lr':              lr,
            }
            if self.cacr_criterion is not None:
                history_entry['train_cacr_loss'] = float(train_cacr)
                history_entry['cacr_diag'] = cacr_diag
            if self.ce_aurc_criterion is not None:
                history_entry['train_ce_aurc_loss'] = float(train_ce_aurc)
                history_entry['ce_aurc_diag'] = ce_aurc_diag
            self.history.append(history_entry)

            is_best = False
            if decision_miou > self.best_miou:
                self.best_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'  ★ New best {metric_label}: '
                      f'{self.best_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            if val_m['mIOU'] > self.best_strict_miou:
                self.best_strict_miou = val_m['mIOU']
                print(f'  New best Strict mIOU: '
                      f'{self.best_strict_miou:.2f}%')
            if val_m['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_m['Shadow_IOU']
            if val_m['F1'] > self.best_f1:
                self.best_f1 = val_m['F1']

            self._save_checkpoint(epoch + 1, is_best)
            self.writer.add_scalar('Train/LR', lr, epoch + 1)

            if (self.args.early_stopping_patience > 0
                    and self.epochs_without_improvement
                    >= self.args.early_stopping_patience):
                print(f'\nEarly stopping — no {metric_label} improvement '
                      f'for {self.args.early_stopping_patience} epochs.')
                break

        print(f'\nTraining finished.')
        print(f'  Best {metric_label}: {self.best_miou:.2f}%')
        print(f'  Best Strict mIOU: {self.best_strict_miou:.2f}%')
        print(f'  Best Shadow_IOU:  {self.best_shadow_iou:.2f}%')
        print(f'  Best F1:          {self.best_f1:.2f}%')

        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)

        print('\nGenerating loss plots...')
        plot_all_losses(self.history, self.output_dir)

        self.writer.close()

    # ------------------------------------------------------------------
    def test(self):
        print('\n' + '=' * 70)
        print('TESTING')
        if self.args.use_tent:
            print(f'  TENT enabled: steps={self.args.tent_steps}  '
                  f'lr={self.args.tent_lr}  '
                  f'use_ln={self.args.tent_use_ln}  '
                  f'use_bn={self.args.tent_use_bn}')
        print('=' * 70)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self._load_checkpoint(best_ckpt)
        else:
            print('Warning: best checkpoint not found, using current weights')

        self.model.eval()

        # ── NEW: TENT setup ──────────────────────────────────────────
        tent_active    = self.args.use_tent
        tent_optimizer = None
        norm_layers    = None
        tent_params    = None
        if tent_active:
            tent_params, norm_layers = configure_tent(
                self.model,
                use_bn=self.args.tent_use_bn,
                use_ln=self.args.tent_use_ln)
            if tent_params:
                tent_optimizer = torch.optim.SGD(
                    tent_params, lr=self.args.tent_lr, momentum=0.9)
                print(f'  TENT: {len(tent_params)} params to adapt '
                      f'from {len(norm_layers)} norm layers')
            else:
                print('  TENT: no eligible norm layers found — disabling.')
                tent_active = False

        pred_save_dir = os.path.join(self.output_dir, 'predictions')
        os.makedirs(pred_save_dir, exist_ok=True)

        sib_strict_list   = []
        sib_tolerant_list = []
        all_filenames     = []
        all_alphas        = []

        for batch in self.dataloaders['test']:
            images  = batch['image'].to(self.device)
            masks   = batch['mask'].to(self.device)
            int_map = batch['intensity_map'].to(self.device)
            city_ids = batch.get('city_id', None)
            if city_ids is not None:
                city_ids = city_ids.to(self.device)

            # ── TENT adaptation step (skip when batch < 2) ──────────
            if tent_active and tent_optimizer is not None:
                if images.size(0) < 2:
                    # LayerNorm doesn't strictly need ≥2, but the
                    # guard preserves parity with the BN code path
                    # and is harmless for ViT.
                    pass
                else:
                    for _ in range(self.args.tent_steps):
                        tent_adapt_step(
                            self.model, images, int_map,
                            tent_optimizer, norm_layers,
                            pred_pos_only=self.args.tent_pred_pos_only,
                            city_ids=city_ids,
                            vib_warmup_factor=1.0)
                # Always set to eval before the prediction forward
                self.model.eval()

            # ── Prediction (eval mode) ──────────────────────────────
            with torch.no_grad():
                outputs, sib_losses = self.model(
                    images, int_map, city_ids=city_ids,
                    vib_warmup_factor=1.0)

            filtered = filter_small_predictions(outputs.detach(),
                                                min_pixels=10)
            preds = torch.argmax(filtered, dim=1)

            # Collect per-image bypass alpha
            if 'bypass_alpha' in sib_losses:
                alpha_batch = sib_losses['bypass_alpha'].cpu().numpy()
                for j in range(alpha_batch.shape[0]):
                    all_alphas.append(float(alpha_batch[j]))

            for i, fname in enumerate(batch['filename']):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                gt_np   = masks[i].cpu().numpy().astype(np.uint8)

                Image.fromarray(pred_np * 255).save(
                    os.path.join(pred_save_dir, fname))

                sib_strict_list.append(
                    _compute_strict_metrics(pred_np, gt_np))
                sib_tolerant_list.append(
                    _compute_tolerant_metrics(
                        pred_np, gt_np,
                        tolerance=self.args.boundary_tolerance))
                all_filenames.append(fname)

        sib_strict   = _average_metrics(sib_strict_list)
        sib_tolerant = _average_metrics(sib_tolerant_list)

        print(f'\nSIB Results ({len(all_filenames)} images):')
        print(f'  Strict:   OA={sib_strict["OA"]:.2f}  '
              f'P={sib_strict["Precision"]:.2f}  '
              f'R={sib_strict["Recall"]:.2f}  '
              f'F1={sib_strict["F1"]:.2f}  '
              f'BER={sib_strict["BER"]:.2f}  '
              f'mIOU={sib_strict["mIOU"]:.2f}  '
              f'ShIOU={sib_strict["Shadow_IOU"]:.2f}')
        print(f'  Tolerant (±{self.args.boundary_tolerance}px): '
              f'OA={sib_tolerant["OA"]:.2f}  '
              f'P={sib_tolerant["Precision"]:.2f}  '
              f'R={sib_tolerant["Recall"]:.2f}  '
              f'F1={sib_tolerant["F1"]:.2f}  '
              f'BER={sib_tolerant["BER"]:.2f}  '
              f'mIOU={sib_tolerant["mIOU"]:.2f}  '
              f'ShIOU={sib_tolerant["Shadow_IOU"]:.2f}')
        if tent_active:
            print(f'  TENT: active ({self.args.tent_steps} steps/batch)')

        # Save bypass gate alpha diagnostics
        if all_alphas:
            alpha_data = {
                'mean_alpha': float(np.mean(all_alphas)),
                'std_alpha': float(np.std(all_alphas)),
                'min_alpha': float(np.min(all_alphas)),
                'max_alpha': float(np.max(all_alphas)),
                'n_images': len(all_alphas),
                'per_image': {fn: a for fn, a in
                              zip(all_filenames, all_alphas)},
            }
            alpha_path = os.path.join(self.output_dir,
                                       'bypass_gate_alpha.json')
            with open(alpha_path, 'w') as f:
                json.dump(alpha_data, f, indent=4)
            print(f'\n  Bypass Gate α diagnostics:')
            print(f'    mean={alpha_data["mean_alpha"]:.4f}  '
                  f'std={alpha_data["std_alpha"]:.4f}  '
                  f'min={alpha_data["min_alpha"]:.4f}  '
                  f'max={alpha_data["max_alpha"]:.4f}')
            print(f'    Saved to {alpha_path}')

        results = {
            'num_images': len(all_filenames),
            'strict':     sib_strict,
            self.tol_key: sib_tolerant,
            'tent_active': tent_active,
        }

        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # ── §4.3: Class-Conditional Temperature Scaling (post-hoc) ────────
        if self.args.use_class_cond_tempscale:
            print('\n' + '=' * 70)
            print('§4.3 MODULE: Class-Conditional Temperature Scaling')
            print('=' * 70)

            # Step 1: collect val logits (source-city, labeled)
            print('  Collecting source-city validation logits...')
            val_logits, val_labels, _ = collect_logits_and_labels_dinov3(
                self.model, self.dataloaders['val'], self.device)
            val_logits = val_logits.to(self.device)
            val_labels = val_labels.to(self.device)
            print(f'  Val: {val_logits.size(0)} images')

            # Step 2: fit T_pos, T_neg
            print('  Fitting (T_pos, T_neg) via LBFGS...')
            T_pos, T_neg = fit_class_conditional_temperature(
                val_logits, val_labels,
                max_iter=self.args.tempscale_max_iter)
            print(f'  → T_pos = {T_pos:.4f}, T_neg = {T_neg:.4f}')

            # Step 3: collect test logits, evaluate with and without tempscale
            print('  Collecting test logits and computing SP-gap metrics...')
            test_logits, test_labels, test_fnames = (
                collect_logits_and_labels_dinov3(
                    self.model, self.dataloaders['test'], self.device))

            # Baseline (T=1.0): SP-gap reference
            sp_baseline = compute_sp_metrics(test_logits, test_labels)

            # Tempscale applied
            scaled_test_logits = apply_tempscale(test_logits, T_pos, T_neg)
            sp_tempscale = compute_sp_metrics(scaled_test_logits, test_labels)

            # Re-compute strict/tolerant on rescaled predictions and save PNGs
            ts_pred_dir = os.path.join(self.output_dir,
                                        'predictions_tempscale')
            os.makedirs(ts_pred_dir, exist_ok=True)
            ts_filtered = filter_small_predictions(scaled_test_logits,
                                                    min_pixels=10)
            ts_preds = ts_filtered.argmax(dim=1).numpy().astype(np.uint8)
            gt_np = test_labels.numpy().astype(np.uint8)

            ts_strict_list, ts_tolerant_list = [], []
            for i, fn in enumerate(test_fnames):
                Image.fromarray(ts_preds[i] * 255).save(
                    os.path.join(ts_pred_dir, fn))
                ts_strict_list.append(_compute_strict_metrics(
                    ts_preds[i], gt_np[i]))
                ts_tolerant_list.append(_compute_tolerant_metrics(
                    ts_preds[i], gt_np[i],
                    tolerance=self.args.boundary_tolerance))
            ts_strict = _average_metrics(ts_strict_list)
            ts_tolerant = _average_metrics(ts_tolerant_list)
            tol_key = self.tol_key

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
                    'aurc_shadow': sp_baseline['aurc_shadow']
                                   - sp_tempscale['aurc_shadow'],
                    'ece_pred_pos': sp_baseline['ece_pred_pos']
                                    - sp_tempscale['ece_pred_pos'],
                },
            }
            with open(os.path.join(self.output_dir,
                                    'tempscale_results.json'), 'w') as f:
                json.dump(ts_summary, f, indent=4)

            print(f'\n  Baseline (T=1.0):')
            print(f'    mIoU={sib_strict["mIOU"]:.2f}  '
                  f'AURC_shadow={sp_baseline["aurc_shadow"]:.4f}  '
                  f'ECE_pos={sp_baseline["ece_pred_pos"]:.4f}')
            print(f'  Tempscale (T_pos={T_pos:.3f}, T_neg={T_neg:.3f}):')
            print(f'    mIoU={ts_strict["mIOU"]:.2f}  '
                  f'AURC_shadow={sp_tempscale["aurc_shadow"]:.4f}  '
                  f'ECE_pos={sp_tempscale["ece_pred_pos"]:.4f}')
            print(f'  ΔAURC_shadow = '
                  f'{ts_summary["sp_gap_reduction"]["aurc_shadow"]:+.4f}')
            print(f'  ΔECE_pos     = '
                  f'{ts_summary["sp_gap_reduction"]["ece_pred_pos"]:+.4f}')
            print(f'  Saved → {os.path.join(self.output_dir, "tempscale_results.json")}')

        if self.args.mode == 'loco' and (
                self.args.comparison_inference_dir
                or self.args.dinov3_output_dir):
            self._compare_with_baselines(
                sib_strict, sib_tolerant,
                sib_strict_list, sib_tolerant_list,
                all_filenames)

        return sib_strict

    # ------------------------------------------------------------------
    def _compare_with_baselines(self, sib_strict, sib_tolerant,
                                sib_strict_list, sib_tolerant_list,
                                filenames):
        from data.dataset import LOCO_FOLDS

        fold_id   = self.args.fold_id
        test_city = LOCO_FOLDS[fold_id]['test']
        res       = self.args.resolution
        img_size  = self.args.img_size
        tol_key   = self.tol_key

        gt_dir = None
        if self.args.comparison_data_root:
            for candidate in [
                os.path.join(self.args.comparison_data_root,
                             test_city, res, 'test', 'masks'),
                os.path.join(self.args.comparison_data_root,
                             test_city, res, 'masks'),
            ]:
                if os.path.isdir(candidate):
                    gt_dir = candidate
                    break

        print('\n' + '=' * 70)
        print('BASELINE COMPARISON')
        print(f'  Test city: {test_city}  |  Resolution: {res}')
        print(f'  GT masks:  {gt_dir}')
        print(f'  Tolerance: ±{self.args.boundary_tolerance}px')
        print('=' * 70)

        baseline_results   = {}
        donor_self_metrics = None

        donor_baselines, donor_self = _find_dinov3_donor(
            self.args.dinov3_output_dir, test_city, res)

        if donor_baselines:
            for label, bl_data in donor_baselines.items():
                if isinstance(bl_data, dict) and 'strict' in bl_data:
                    baseline_results[label] = bl_data
            if donor_self and donor_self.get('strict', {}).get('F1', 0) > 0:
                donor_self_metrics = donor_self
            print(f'\n  S1: {len(baseline_results)} baseline(s) loaded '
                  f'from DINOv3 donor.')
        else:
            print('\n  S1: No DINOv3 donor found — proceeding with S2 only.')

        raw_pred_dirs = _find_raw_prediction_dirs(
            self.args.comparison_inference_dir, test_city, res)

        if raw_pred_dirs and gt_dir is None:
            print('\n  S2: raw dirs found but GT dir missing — '
                  'cannot compute metrics.  '
                  'Pass --comparison_data_root to enable.')
        elif raw_pred_dirs:
            print(f'\n  S2: computing metrics for '
                  f'{len(raw_pred_dirs)} baseline(s)...')
            for label, pred_dir in raw_pred_dirs.items():
                bl = _baseline_metrics_from_predictions(
                    pred_dir, gt_dir, filenames,
                    img_size, tol_key,
                    self.args.boundary_tolerance)
                if bl:
                    baseline_results[label] = bl
                    # Defensive — never crash on tol-key mismatch
                    tol_metrics = bl.get(tol_key, {})
                    tol_f1 = tol_metrics.get('F1', 0.0)
                    print(f'  S2: ✓ {label} — {bl["n_images"]} images  '
                          f'(strict F1={bl["strict"]["F1"]:.2f}  '
                          f'tol F1={tol_f1:.2f})')
                else:
                    print(f'  S2: ✗ {label} — no matching image pairs')

        if not baseline_results:
            print('\n  ⚠  No baselines available.  Check:')
            print('       --comparison_inference_dir  (Test_img_results root)')
            print('       --dinov3_output_dir        (for donor experiment)')
            print('       --comparison_data_root     (GT masks root)')
        else:
            self._print_comparison_table(
                'STRICT METRICS (all pixels)',
                baseline_results, sib_strict, sib_tolerant,
                donor_self_metrics, metric_type='strict')
            self._print_comparison_table(
                f'TOLERANT METRICS (±{self.args.boundary_tolerance}px '
                f'dont-care zone)',
                baseline_results, sib_strict, sib_tolerant,
                donor_self_metrics, metric_type=tol_key)
            self._print_recovery_ratios(
                baseline_results, sib_strict, sib_tolerant)
            for bl_label in _LOCO_BASELINE_LABELS:
                if (bl_label in baseline_results
                        and 'strict_list' in baseline_results[bl_label]):
                    self._print_bootstrap_comparison(
                        baseline_results[bl_label],
                        sib_strict_list, sib_tolerant_list,
                        baseline_label=bl_label)

        comp = {
            'test_city':  test_city,
            'resolution': res,
            'eval_size':  img_size,
            'sib':  {'strict': sib_strict, tol_key: sib_tolerant},
            'ddib': donor_self_metrics if donor_self_metrics
                    else {'strict': sib_strict, tol_key: sib_tolerant},
            'baselines': {},
        }
        for label, br in baseline_results.items():
            comp['baselines'][label] = {
                'strict':  br.get('strict', {}),
                tol_key:   br.get(tol_key, br.get('tolerant', {})),
            }
            if 'n_images' in br:
                comp['baselines'][label]['n_images'] = br['n_images']

        comp_path = os.path.join(self.output_dir, 'comparison_results.json')
        with open(comp_path, 'w') as f:
            json.dump(comp, f, indent=4)
        print(f'\nComparison saved to {comp_path}')

    # ------------------------------------------------------------------
    def _print_comparison_table(self, title, baseline_results,
                                sib_strict, sib_tolerant,
                                donor_self_metrics=None,
                                metric_type='strict'):
        sib_m = sib_strict if metric_type == 'strict' else sib_tolerant

        print('\n' + '-' * 70)
        print(f'{title:^70}')
        print('-' * 70)
        header = (f'  {"Method":<22} {"OA":>6} {"Prec":>6} {"Rec":>6} '
                  f'{"F1":>6} {"BER":>6} {"mIOU":>6} {"ShIOU":>6}')
        print(header)
        print('  ' + '-' * 64)

        def _row(label, m):
            if not m:
                return
            print(f'  {label:<22} {m.get("OA", 0):6.2f} '
                  f'{m.get("Precision", 0):6.2f} {m.get("Recall", 0):6.2f} '
                  f'{m.get("F1", 0):6.2f} {m.get("BER", 0):6.2f} '
                  f'{m.get("mIOU", 0):6.2f} {m.get("Shadow_IOU", 0):6.2f}')

        for label in _TABLE_ORDER:
            if label not in baseline_results:
                continue
            _row(label, baseline_results[label].get(metric_type, {}))

        if donor_self_metrics:
            donor_m = donor_self_metrics.get(metric_type, {})
            _row('Donor', donor_m)

        print('  ' + '-' * 64)
        _row('SIB (ours)', sib_m)

    # ------------------------------------------------------------------
    def _print_recovery_ratios(self, baseline_results,
                               sib_strict, sib_tolerant):
        if ('Upper Bound' not in baseline_results
                or 'LOCO Vanilla' not in baseline_results):
            return

        tol_key = self.tol_key

        print('\n' + '-' * 70)
        print(f'{"RECOVERY RATIOS":^70}')
        print(f'  R = (SIB - LOCO_Vanilla) / (Upper - LOCO_Vanilla)')
        print(f'  0 = no help, 1 = gap fully closed')
        print('-' * 70)

        key_metrics = ['F1', 'mIOU', 'Shadow_IOU', 'BER']

        for eval_type, sib_m, label in [
                ('strict', sib_strict, 'Strict'),
                (tol_key, sib_tolerant, 'Tolerant')]:
            ub = baseline_results['Upper Bound'].get(eval_type, {})
            lv = baseline_results['LOCO Vanilla'].get(eval_type, {})
            if not ub or not lv:
                continue
            parts = []
            for k in key_metrics:
                if k not in ub or k not in lv:
                    continue
                gap = ub[k] - lv[k]
                rec = sib_m[k] - lv[k]
                if k == 'BER':
                    gap, rec = -gap, -rec
                R = rec / gap if abs(gap) > 0.01 else float('nan')
                parts.append(f'{k}={R:.3f}')
            print(f'  {label:<10}  ' + '  '.join(parts))

        adapt_methods = [
            'LOCO FDA', 'LOCO SegDesic',
            'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA',
        ]
        has_any = any(m in baseline_results for m in adapt_methods)
        if not has_any:
            return

        print('\n' + '-' * 70)
        print(f'{"SIB IMPROVEMENT OVER ADAPTATION METHODS (delta)":^70}')
        print(f'  Positive = SIB better.  For BER, negative = SIB better.')
        print('-' * 70)

        for eval_type, sib_m, label in [
                ('strict', sib_strict, 'Strict'),
                (tol_key, sib_tolerant, 'Tolerant')]:
            print(f'\n  {label}:')
            print(f'    {"Method":<16} {"dF1":>7} {"dmIOU":>7} '
                  f'{"dShIOU":>7} {"dBER":>7}')
            print(f'    ' + '-' * 40)
            for bl in adapt_methods:
                if bl not in baseline_results:
                    continue
                bm = baseline_results[bl].get(eval_type, {})
                if not bm:
                    continue
                parts = [f'{sib_m[k] - bm[k]:+7.2f}' for k in key_metrics
                         if k in sib_m and k in bm]
                print(f'    {bl:<16} ' + ' '.join(parts))

    # ------------------------------------------------------------------
    def _print_bootstrap_comparison(self, loco_baseline,
                                    sib_strict_list, sib_tolerant_list,
                                    baseline_label='LOCO Vanilla',
                                    n_bootstrap=5000):
        tol_key = self.tol_key

        print('\n' + '-' * 70)
        print(f'{"BOOTSTRAP: SIB vs " + baseline_label + " (n=5000)":^70}')
        print('-' * 70)

        np.random.seed(42)
        key_metrics = ['F1', 'mIOU', 'Shadow_IOU']

        for eval_type, sib_list, label in [
                ('strict_list',   sib_strict_list,   'Strict'),
                ('tolerant_list', sib_tolerant_list, 'Tolerant')]:
            loco_list = loco_baseline.get(eval_type, [])
            n = min(len(loco_list), len(sib_list))
            if n == 0:
                continue
            print(f'\n  {label}:')
            for k in key_metrics:
                loco_vals = np.array([m[k] for m in loco_list[:n]])
                sib_vals  = np.array([m[k] for m in sib_list[:n]])
                diff = sib_vals - loco_vals
                obs_mean = np.mean(diff)
                boot_means = np.array([
                    np.mean(diff[np.random.choice(n, n, replace=True)])
                    for _ in range(n_bootstrap)])
                ci_lo = np.percentile(boot_means, 2.5)
                ci_hi = np.percentile(boot_means, 97.5)
                if obs_mean >= 0:
                    p_val = 2 * max(np.mean(boot_means <= 0),
                                    1.0 / n_bootstrap)
                else:
                    p_val = 2 * max(np.mean(boot_means >= 0),
                                    1.0 / n_bootstrap)
                p_val = min(p_val, 1.0)
                sig = ''
                if p_val < 0.001:  sig = ' ***'
                elif p_val < 0.01: sig = ' **'
                elif p_val < 0.05: sig = ' *'
                print(f'    {k:<12} delta={obs_mean:+.2f}  '
                      f'95%CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  '
                      f'p={p_val:.4f}{sig}')
        print()

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_miou': self.best_miou,
            'best_strict_miou': self.best_strict_miou,
            'best_shadow_iou': self.best_shadow_iou,
            'best_f1': self.best_f1,
            'use_tolerant_for_decisions': self.use_tolerant_for_decisions,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history': self.val_metrics_history,
            'history': self.history,
            'args': vars(self.args),
        }
        torch.save(ckpt, os.path.join(self.output_dir,
                                       'checkpoint_latest.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(self.output_dir,
                                           'checkpoint_best.pth'))
        if epoch % self.args.save_freq == 0:
            torch.save(ckpt, os.path.join(
                self.output_dir, f'checkpoint_epoch_{epoch}.pth'))

    def _load_checkpoint(self, path):
        print(f'Loading checkpoint: {path}')
        ckpt = torch.load(path, map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_miou = ckpt.get('best_miou', 0.0)
        self.best_strict_miou = ckpt.get('best_strict_miou', 0.0)
        self.best_shadow_iou = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1 = ckpt.get('best_f1', 0.0)
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses = ckpt.get('val_losses', [])
        self.train_metrics_history = ckpt.get(
            'train_metrics_history',
            {k: [] for k in self.train_metrics_history})
        self.val_metrics_history = ckpt.get(
            'val_metrics_history',
            {k: [] for k in self.val_metrics_history})
        self.history = ckpt.get('history', [])

        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict')
        print(f'  Resumed from epoch {ckpt["epoch"]}  '
              f'(best {metric_label} mIOU={self.best_miou:.2f}%, '
              f'best Strict mIOU={self.best_strict_miou:.2f}%)')


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()

    active = []
    if args.use_haar:              active.append('HAAR')
    if args.use_vib:               active.append('VIB')
    if args.use_content_aug:       active.append('AUG')
    if args.adaptive_beta:         active.append('AdaptiveBeta')
    if args.use_passthrough_gate:  active.append('PassthroughGate')
    if args.use_module_bypass:     active.append('ModuleBypass')
    print(f'\nSIB components: {", ".join(active) if active else "NONE"}')
    if args.use_vib:
        print(f'  vib_beta_content={args.vib_beta_content}  '
              f'vib_beta_edge={args.vib_beta_edge}  '
              f'vib_beta_scale={args.vib_beta_scale}')
        print(f'  lambda_content={args.lambda_content}  '
              f'lambda_edge={args.lambda_edge}')
        print(f'  vib_warmup_fraction={args.vib_warmup_fraction}')
    print(f'  boundary_tolerance=±{args.boundary_tolerance}px')
    print(f'  passthrough_gate={"ON" if args.use_passthrough_gate else "OFF"}')
    print(f'  module_bypass={"ON" if args.use_module_bypass else "OFF"}')
    if args.disable_content_vib:  print('  ABLATION A1: disable_content_vib=ON')
    if args.symmetric_vib:        print('  ABLATION A3: symmetric_vib=ON')
    if args.aug_all_subbands:     print('  ABLATION A6: aug_all_subbands=ON')
    if args.vib_on_hl_only:       print('  ABLATION A10: vib_on_hl_only=ON')
    # NEW: log diagnostic-module flags
    print(f'  CACR={"ON" if args.use_cacr else "OFF"}'
          f' (w={args.cacr_weight}, neg_w={args.cacr_neg_weight})')
    print(f'  CE-AURC={"ON" if args.use_ce_aurc else "OFF"}'
          f' (w={args.ce_aurc_weight}, floor={args.ce_aurc_floor})')
    print(f'  TENT={"ON" if args.use_tent else "OFF"}'
          f' (steps={args.tent_steps}, lr={args.tent_lr}, '
          f'use_ln={args.tent_use_ln}, use_bn={args.tent_use_bn})')

    trainer = TrainerSIB(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()