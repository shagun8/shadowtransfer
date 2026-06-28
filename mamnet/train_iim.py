"""
Training script for MAMNet + IIM (Illumination-Invariant Module)

Key differences from base train.py
-----------------------------------
1. Model    : MAMNetIIM (IIM front-end → encoder → MSCAF → decoder)
2. Loss     : Per-image mean Cross-Entropy + weighted II Loss (Eq. 13)
3. Optim    : Zero-mean constraint enforced on IIM kernels after every step
4. Tracking : Separate histories for main-CE, aux, and II-loss (raw + weighted)
5. Viz      : Overview subplot (all losses) + per-component subplots (own y-scale)

Decision metrics (LR scheduler, best checkpoint, early stopping) use
**per-image** mIOU from DetailedEvaluator — never pooled ShadowMetrics.
"""

import os
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_iim import MAMNetIIM
from models.iim import compute_ii_loss
from data.dataset import get_dataloaders
from data.dataset_enhanced import ShadowDatasetEnhanced
from utils.evaluation_detailed import DetailedEvaluator
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ======================================================================
# GPU diagnostics
# ======================================================================
print("=" * 50)
print("GPU DIAGNOSTICS")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("=" * 50)


# ======================================================================
# Per-image cross-entropy
# ======================================================================

class PerImageCrossEntropy(nn.Module):
    """CE loss computed per image then averaged across the batch."""

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred   : [B, 2, H, W] logits
            target : [B, H, W]    integer labels
        Returns:
            scalar — mean of per-image CE losses
        """
        loss_map = F.cross_entropy(
            pred, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )  # [B, H, W]
        return loss_map.mean(dim=(1, 2)).mean()   # per-image mean → batch mean


class MAMNetIIMLoss(nn.Module):
    """
    Combined loss: per-image CE (main + aux) + II consistency loss.

    II loss weighting modes
    -----------------------
    adaptive (default):
        Dynamically scales II loss so its effective contribution equals
        ``ii_target_ratio`` × task_loss each step.  Robust across datasets
        and training stages — no manual weight tuning required.

        Mechanism:
            task_loss      = main_CE + aux              (detached for weight calc)
            eff_weight     = ratio × task_loss / (ii_raw + ε)
            ii_effective   = eff_weight × ii_raw

        Gradient flows only through ii_raw, not through task_loss.

    fixed:
        Simple fixed scalar weight (``ii_loss_weight``).  Useful when you
        know the loss scales and want full manual control.

    Returns a dict of component losses for logging.
    """

    def __init__(self, aux_weight=0.4, ii_loss_weight=0.01,
                 ii_target_ratio=0.01, ii_loss_mode='adaptive',
                 weight=None):
        super().__init__()
        self.aux_weight = aux_weight
        self.ii_loss_weight = ii_loss_weight
        self.ii_target_ratio = ii_target_ratio
        self.ii_loss_mode = ii_loss_mode
        self.criterion = PerImageCrossEntropy(weight=weight)

    def forward(self, outputs, target, ii_loss=None):
        """
        Args:
            outputs  : dict with 'main', optionally 'aux1'-'aux3'
            target   : [B, H, W]
            ii_loss  : scalar tensor (raw, unweighted) or None

        Returns:
            dict with keys:
                total, main, aux (weighted), ii (weighted), ii_raw,
                ii_eff_weight (the multiplier actually applied to ii_raw)
        """
        main_loss = self.criterion(outputs['main'], target)
        total = main_loss
        result = {'main': main_loss}

        if 'aux1' in outputs:
            a1 = self.criterion(outputs['aux1'], target)
            a2 = self.criterion(outputs['aux2'], target)
            a3 = self.criterion(outputs['aux3'], target)
            aux_raw = (a1 + a2 + a3) / 3.0
            aux_weighted = self.aux_weight * aux_raw
            total = total + aux_weighted
            result['aux'] = aux_weighted
        else:
            result['aux'] = torch.tensor(0.0, device=main_loss.device)

        if ii_loss is not None and ii_loss.item() > 0:
            if self.ii_loss_mode == 'adaptive':
                # Scale II loss to contribute ii_target_ratio of task loss
                task_loss_detached = total.detach()
                ii_detached = ii_loss.detach() + 1e-8
                eff_weight = self.ii_target_ratio * task_loss_detached / ii_detached
                ii_weighted = eff_weight * ii_loss
            else:
                eff_weight = torch.tensor(self.ii_loss_weight,
                                          device=main_loss.device)
                ii_weighted = self.ii_loss_weight * ii_loss

            total = total + ii_weighted
            result['ii'] = ii_weighted
            result['ii_raw'] = ii_loss
            result['ii_eff_weight'] = eff_weight
        else:
            zero = torch.tensor(0.0, device=main_loss.device)
            result['ii'] = zero
            result['ii_raw'] = zero
            result['ii_eff_weight'] = zero

        result['total'] = total
        return result


# ======================================================================
# Visualisation
# ======================================================================

# Colour palette
_C = {
    'total_train': '#2E86AB', 'total_val': '#A23B72',
    'main_train':  '#F18F01', 'main_val':  '#C73E1D',
    'aux_train':   '#6A994E',
    'ii_train':    '#8338EC', 'ii_val':    '#3A86FF',
}

_METRIC_COLORS = {
    'OA': '#2E86AB', 'Precision': '#F18F01', 'F1': '#A23B72',
    'BER': '#E63946', 'mIOU': '#6A994E', 'Shadow_IOU': '#8338EC',
}

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


def plot_loss_curves_iim(
    train_total, val_total,
    train_main, val_main,
    train_aux,
    train_ii, val_ii,
    save_path,
):
    """
    Layout
    ------
    Row 0  : Overview — all losses on shared y-axis.
    Row 1+ : Individual panels (each with own y-scale, no total):
               Main CE (train + val)
               Auxiliary (train only)
               II Loss  (train + val)
    """
    epochs = list(range(1, len(train_total) + 1))

    # --- Decide which individual panels to show ---
    panels = []
    if train_main and len(train_main) == len(epochs):
        panels.append({
            'title': 'Main CE Loss', 'ylabel': 'Loss',
            'series': [
                (train_main, 'Train Main', _C['main_train'], '-', 'o'),
                (val_main,   'Val Main',   _C['main_val'],   '--', 's'),
            ],
        })
    has_aux = train_aux and any(v > 1e-9 for v in train_aux)
    if has_aux:
        panels.append({
            'title': 'Auxiliary Loss (weighted)', 'ylabel': 'Loss',
            'series': [
                (train_aux, 'Train Aux', _C['aux_train'], '-', '^'),
            ],
        })
    has_ii = train_ii and any(v > 1e-9 for v in train_ii)
    if has_ii:
        panels.append({
            'title': 'II Loss (raw, unweighted)', 'ylabel': 'Loss',
            'series': [
                (train_ii, 'Train II', _C['ii_train'], '-', 'D'),
                (val_ii,   'Val II',   _C['ii_val'],   '--', 'v'),
            ],
        })

    MAX_COLS = 4
    n_ind = len(panels)
    n_rows_ind = max(1, (n_ind + MAX_COLS - 1) // MAX_COLS) if n_ind else 0

    fig_w = max(9, min(5.5 * max(n_ind, 1), 22))
    fig_h = 4.2 + 3.8 * n_rows_ind

    if n_ind:
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55,
                                 height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ---- Overview ----
    ax_ov.plot(epochs, train_total, '-', lw=2, color=_C['total_train'],
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_total, '--', lw=1.8, color=_C['total_val'],
               label='Val total', marker='s', ms=3, mfc='white', mew=1.2)
    if train_main and len(train_main) == len(epochs):
        ax_ov.plot(epochs, train_main, '-', lw=1.3, color=_C['main_train'],
                   alpha=0.7, label='Train main CE')
        ax_ov.plot(epochs, val_main, '--', lw=1.3, color=_C['main_val'],
                   alpha=0.7, label='Val main CE')
    if has_aux:
        ax_ov.plot(epochs, train_aux, '-', lw=1.3, color=_C['aux_train'],
                   alpha=0.7, label='Train aux')
    if has_ii:
        ax_ov.plot(epochs, train_ii, '-', lw=1.3, color=_C['ii_train'],
                   alpha=0.7, label='Train II (raw)')
        ax_ov.plot(epochs, val_ii, '--', lw=1.3, color=_C['ii_val'],
                   alpha=0.7, label='Val II (raw)')

    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold')
    ax_ov.set_xlabel('Epoch'); ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=7, framealpha=0.85, ncol=4)
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ---- Per-component panels ----
    if n_ind:
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_ind, MAX_COLS, subplot_spec=outer[1],
            hspace=0.55, wspace=0.42)

        for pidx, panel in enumerate(panels):
            r, c = divmod(pidx, MAX_COLS)
            ax = fig.add_subplot(inner[r, c])
            for vals, lbl, col, ls, mk in panel['series']:
                ep = list(range(1, len(vals) + 1))
                ax.plot(ep, vals, ls=ls, lw=1.8, color=col, label=lbl,
                        marker=mk, ms=3.5, mfc='white', mew=1.2, alpha=0.9)
            _style_ax(ax, panel['title'], panel['ylabel'])

        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('MAMNet-IIM — Training Loss Curves', fontweight='bold',
                 fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


def plot_metrics_curves(train_hist, val_hist, save_path):
    """Per-metric train / val curves (pooled ShadowMetrics, for reference)."""
    metrics = list(train_hist.keys())
    n = len(metrics)
    if n == 0:
        return
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows),
                             squeeze=False)
    for idx, m in enumerate(metrics):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        color = _METRIC_COLORS.get(m, '#333')
        ep_tr = list(range(1, len(train_hist[m]) + 1))
        ep_vl = list(range(1, len(val_hist[m]) + 1))
        ax.plot(ep_tr, train_hist[m], '-', lw=1.8, color=color,
                label='Train', marker='o', ms=3.5, mfc='white', mew=1.2)
        ax.plot(ep_vl, val_hist[m], '--', lw=1.8, color=color,
                label='Val', marker='s', ms=3.5, mfc='white', mew=1.2, alpha=0.75)
        _style_ax(ax, m, '%')
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)
    fig.suptitle('MAMNet-IIM — Metric Curves (pooled, reference)',
                 fontweight='bold', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Metric curves saved → {save_path}')


# ======================================================================
# Best / worst visualisations (imported from base)
# ======================================================================
from utils.visualization import save_best_worst_visualizations  # noqa: E402


# ======================================================================
# CLI
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(description='Train MAMNet-IIM')

    # Data
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # Multi-city / LOCO
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities', type=str, nargs='+', default=None)

    # Model
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--aux_weight', type=float, default=0.4)
    p.add_argument('--use_contrast', action='store_true')

    # IIM-specific
    p.add_argument('--num_kernels', type=int, default=8,
                   help='Number of IIM learnable kernels (default 8)')
    p.add_argument('--kernel_size', type=int, default=5,
                   help='IIM kernel spatial size (default 5)')
    p.add_argument('--ii_loss_mode', type=str, default='adaptive',
                   choices=['adaptive', 'fixed'],
                   help='How to weight II loss. "adaptive" scales dynamically '
                        'to maintain ii_target_ratio of task loss. '
                        '"fixed" uses ii_loss_weight as a static multiplier.')
    p.add_argument('--ii_target_ratio', type=float, default=0.01,
                   help='Target fraction of task loss for II loss '
                        '(adaptive mode, default 0.01 = 1%%)')
    p.add_argument('--ii_loss_weight', type=float, default=0.01,
                   help='Fixed weight for II loss (fixed mode only)')
    p.add_argument('--gamma_range_lo', type=float, default=0.5)
    p.add_argument('--gamma_range_hi', type=float, default=2.0)

    # Training
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # Checkpoints
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')

    # Device
    p.add_argument('--device', type=str, default='cuda')

    # FDA (kept for compatibility)
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L', type=float, default=0.01)

    # Boundary-tolerant evaluation
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--boundary_tolerance', type=int, default=2)

    # Early stopping
    p.add_argument('--early_stopping_patience', type=int, default=0)

    # Comparison
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root', type=str, default=None)

    return p.parse_args()


# ======================================================================
# Trainer
# ======================================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.tol_key = f'tolerant_{args.boundary_tolerance}px'
        self.gamma_range = (args.gamma_range_lo, args.gamma_range_hi)

        # ---- Output dir ----
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        modifiers = ['iim']
        if args.use_fda:
            modifiers.append('fda')
        mod_str = '_'.join(modifiers)

        if args.mode == 'single':
            city = args.data_root.rstrip('/').split("/")[-2]
            res  = args.data_root.rstrip('/').split("/")[-1]
            exp_name = f'mamnet_{mod_str}_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'mamnet_{mod_str}_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'mamnet_{mod_str}_loco_holdout_{test_city}_{args.resolution}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # ---- Model ----
        print('Initializing MAMNet-IIM...')
        self.model = MAMNetIIM(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            use_aux=True,
            use_contrast=args.use_contrast,
            num_kernels=args.num_kernels,
            kernel_size=args.kernel_size,
        ).to(self.device)

        total_p = sum(p.numel() for p in self.model.parameters())
        train_p = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        iim_p   = sum(p.numel() for p in self.model.iim.parameters())
        print(f'Total parameters:     {total_p:,}')
        print(f'Trainable parameters: {train_p:,}')
        print(f'IIM parameters:       {iim_p:,}  ({iim_p/1e6:.4f} M)')

        # Kernel diagnostic
        with torch.no_grad():
            k = self.model.iim.kernels
            print(f'IIM kernel stats — '
                  f'std: {k.std().item():.4f}  '
                  f'abs_mean: {k.abs().mean().item():.4f}  '
                  f'zero-mean check: {k.mean(dim=(-2,-1)).abs().max().item():.1e}')

        # ---- Loss ----
        self.criterion = MAMNetIIMLoss(
            aux_weight=args.aux_weight,
            ii_loss_weight=args.ii_loss_weight,
            ii_target_ratio=args.ii_target_ratio,
            ii_loss_mode=args.ii_loss_mode,
        )
        print(f'>> II loss mode: {args.ii_loss_mode}  '
              f'{"target_ratio=" + str(args.ii_target_ratio) if args.ii_loss_mode == "adaptive" else "weight=" + str(args.ii_loss_weight)}')

        # ---- Optimiser & scheduler ----
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3)

        # ---- Decision metric ----
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU')

        self.detailed_eval_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_eval_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ---- Tracking ----
        self.start_epoch = 0
        self.best_miou = 0.0
        self.best_shadow_iou = 0.0
        self.best_f1 = 0.0
        self.best_decision_miou = 0.0
        self.epochs_without_improvement = 0

        # Loss histories
        self.train_total   = []
        self.train_main    = []
        self.train_aux     = []
        self.train_ii_raw  = []   # unweighted II loss
        self.val_total     = []
        self.val_main      = []
        self.val_ii_raw    = []   # unweighted II loss (validation)

        # Metric histories (pooled, reference only)
        self.train_metrics_history = {
            k: [] for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
        self.val_metrics_history = {
            k: [] for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}

        if args.resume:
            self.load_checkpoint(args.resume)

        # ---- Dataloaders ----
        self._init_dataloaders(args)

    # ------------------------------------------------------------------
    def _init_dataloaders(self, args):
        if args.use_contrast:
            if args.mode == 'single':
                if args.data_root is None:
                    raise ValueError("data_root required for single mode")
                train_paths = val_paths = test_paths = [args.data_root]
            elif args.mode == 'all':
                if args.base_data_root is None or args.resolution is None:
                    raise ValueError("base_data_root and resolution required")
                cities = args.cities or ['chicago', 'miami', 'phoenix']
                train_paths = [os.path.join(args.base_data_root, c, args.resolution)
                               for c in cities]
                val_paths = test_paths = train_paths
            elif args.mode == 'loco':
                if args.base_data_root is None or args.resolution is None or args.fold_id is None:
                    raise ValueError("base_data_root, resolution, fold_id required")
                from data.dataset import LOCO_FOLDS
                fold = LOCO_FOLDS[args.fold_id]
                train_paths = [os.path.join(args.base_data_root, c, args.resolution)
                               for c in fold['train']]
                val_paths = train_paths
                test_paths = [os.path.join(args.base_data_root, fold['test'],
                                           args.resolution)]

            from torch.utils.data import DataLoader
            ds_train = ShadowDatasetEnhanced(
                root_dir=train_paths, split='train', img_size=args.img_size,
                task_id=2, augment=True,
                use_fda=args.use_fda, fda_target_root=args.fda_target_root,
                fda_L=args.fda_L)
            ds_val = ShadowDatasetEnhanced(
                root_dir=val_paths, split='val', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)
            ds_test = ShadowDatasetEnhanced(
                root_dir=test_paths, split='test', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)

            self.dataloaders = {
                'train': DataLoader(ds_train, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers,
                                    pin_memory=True, drop_last=True),
                'val':   DataLoader(ds_val, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True),
                'test':  DataLoader(ds_test, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True),
            }
            print(f'Train: {len(ds_train)}  Val: {len(ds_val)}  Test: {len(ds_test)}')
        else:
            self.dataloaders = get_dataloaders(
                data_root=args.data_root,
                base_data_root=args.base_data_root,
                mode=args.mode,
                cities=args.cities,
                resolution=args.resolution,
                fold_id=args.fold_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_size=args.img_size,
                use_fda=getattr(args, 'use_fda', False),
                fda_target_root=getattr(args, 'fda_target_root', None),
                fda_L=getattr(args, 'fda_L', 0.01),
            )
            print(f'Train: {len(self.dataloaders["train"].dataset)}  '
                  f'Val: {len(self.dataloaders["val"].dataset)}  '
                  f'Test: {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    def _get_decision_miou(self, detailed_results):
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_decision:
            return bt[self.tol_key]['iou']
        else:
            return bt['strict']['iou']

    # ------------------------------------------------------------------
    def _extract_rgb(self, images):
        """Return the RGB portion of the input (first 3 channels)."""
        return images[:, :3]

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def train_epoch(self, epoch):
        self.model.train()

        ep_total = ep_main = ep_aux = ep_ii_raw = ep_ii_eff = ep_ii_w = 0.0
        train_metrics = ShadowMetrics()
        num_batches = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start = time.time()

        for bi, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            # ---------- Forward ----------
            outputs = self.model(images)  # dict with 'main','aux*','iim_features'

            # ---------- II Loss ----------
            rgb = self._extract_rgb(images)
            ii_loss = compute_ii_loss(
                outputs['iim_features'], self.model.iim, rgb,
                gamma_range=self.gamma_range, beta=1.0)

            # ---------- Combined loss ----------
            losses = self.criterion(outputs, masks, ii_loss=ii_loss)
            loss = losses['total']

            # ---------- Backward ----------
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---------- Zero-mean constraint ----------
            self.model.iim.enforce_zero_mean()

            # ---------- Metrics ----------
            filtered = filter_small_predictions(outputs['main'], min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_eval_train.update(preds, masks, images)

            ep_total  += losses['total'].item()
            ep_main   += losses['main'].item()
            ep_aux    += losses['aux'].item()
            ep_ii_raw += losses['ii_raw'].item()
            ep_ii_eff += losses['ii'].item()
            ep_ii_w   += losses['ii_eff_weight'].item() if torch.is_tensor(losses['ii_eff_weight']) else losses['ii_eff_weight']

            if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                ew = losses['ii_eff_weight']
                ew_val = ew.item() if torch.is_tensor(ew) else ew
                ii_pct = 100 * losses['ii'].item() / (losses['total'].item() + 1e-12)
                print(f'Batch [{bi+1}/{num_batches}] | '
                      f'Total: {losses["total"].item():.4f} | '
                      f'Main: {losses["main"].item():.4f} | '
                      f'Aux: {losses["aux"].item():.4f} | '
                      f'II(raw): {losses["ii_raw"].item():.2e}  '
                      f'w={ew_val:.1f}  '
                      f'II(eff): {losses["ii"].item():.4f} [{ii_pct:.1f}%]')

        ep_total  /= num_batches
        ep_main   /= num_batches
        ep_aux    /= num_batches
        ep_ii_raw /= num_batches
        ep_ii_eff /= num_batches
        ep_ii_w   /= num_batches

        metrics = train_metrics.compute()
        elapsed = time.time() - start

        ii_pct = 100 * ep_ii_eff / (ep_total + 1e-12)
        print(f'\nTraining Results:')
        print(f'Time: {elapsed:.1f}s | Total: {ep_total:.4f} | '
              f'Main: {ep_main:.4f} | Aux: {ep_aux:.4f}')
        print(f'II(raw): {ep_ii_raw:.2e}  '
              f'II(eff): {ep_ii_eff:.4f} [{ii_pct:.1f}% of total]  '
              f'avg_w: {ep_ii_w:.1f}')
        print(f'OA: {metrics["OA"]:.2f}%  Prec: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'ShadIOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Train/Loss',           ep_total,  epoch)
        self.writer.add_scalar('Train/MainLoss',       ep_main,   epoch)
        self.writer.add_scalar('Train/AuxLoss',        ep_aux,    epoch)
        self.writer.add_scalar('Train/IILoss_raw',     ep_ii_raw, epoch)
        self.writer.add_scalar('Train/IILoss_eff',     ep_ii_eff, epoch)
        self.writer.add_scalar('Train/II_eff_weight',  ep_ii_w,   epoch)
        for k in self.train_metrics_history:
            self.writer.add_scalar(f'Train/{k}', metrics[k], epoch)

        # Store histories
        self.train_total.append(ep_total)
        self.train_main.append(ep_main)
        self.train_aux.append(ep_aux)
        self.train_ii_raw.append(ep_ii_raw)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(metrics[k])

        # DetailedEvaluator — per-image
        dr = self.detailed_eval_train.compute_metrics()
        self.detailed_eval_train.reset()
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_pi',   s['iou'], epoch)
        self.writer.add_scalar('Train/F1_strict_pi',     s['f1'],  epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_pi', t['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_pi',   t['f1'],  epoch)
        print(f'Per-image Strict:   F1={s["f1"]:.2f}%  mIOU={s["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t["f1"]:.2f}%  mIOU={t["iou"]:.2f}%')

        return ep_total, ep_main, ep_aux, ep_ii_raw, metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    def validate(self, epoch):
        print('\nValidating...')
        self.model.eval()

        val_main_sum = 0.0
        val_ii_sum   = 0.0
        val_metrics  = ShadowMetrics()

        per_img_ce = PerImageCrossEntropy()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs = self.model(images)  # eval → logits tensor

                # Per-image main CE
                val_main_sum += per_img_ce(outputs, masks).item()

                # II loss tracking
                rgb = self._extract_rgb(images)
                feats_orig = self.model.iim.extract_features(rgb)
                ii_loss = compute_ii_loss(
                    feats_orig, self.model.iim, rgb,
                    gamma_range=self.gamma_range, beta=1.0)
                val_ii_sum += ii_loss.item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_eval_val.update(preds, masks, images)

        n = len(self.dataloaders['val'])
        val_main_avg = val_main_sum / n
        val_ii_avg   = val_ii_sum / n
        # Total val: main + adaptively-weighted II  (no aux in eval mode)
        if self.args.ii_loss_mode == 'adaptive' and val_ii_avg > 0:
            ii_w = self.args.ii_target_ratio * val_main_avg / (val_ii_avg + 1e-8)
        else:
            ii_w = self.args.ii_loss_weight
        val_ii_eff = ii_w * val_ii_avg
        val_total_avg = val_main_avg + val_ii_eff

        metrics = val_metrics.compute()

        ii_pct = 100 * val_ii_eff / (val_total_avg + 1e-12)
        print(f'Validation Results:')
        print(f'Total: {val_total_avg:.4f} | Main: {val_main_avg:.4f} | '
              f'II(raw): {val_ii_avg:.2e}  II(eff): {val_ii_eff:.4f} [{ii_pct:.1f}%]  '
              f'w={ii_w:.1f}')
        print(f'OA: {metrics["OA"]:.2f}%  Prec: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'ShadIOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Val/Loss',       val_total_avg, epoch)
        self.writer.add_scalar('Val/MainLoss',   val_main_avg,  epoch)
        self.writer.add_scalar('Val/IILoss_raw', val_ii_avg,    epoch)
        for k in self.val_metrics_history:
            self.writer.add_scalar(f'Val/{k}', metrics[k], epoch)

        self.val_total.append(val_total_avg)
        self.val_main.append(val_main_avg)
        self.val_ii_raw.append(val_ii_avg)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(metrics[k])

        # DetailedEvaluator
        dr = self.detailed_eval_val.compute_metrics()
        self.detailed_eval_val.reset()
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Val/mIOU_strict_pi',   s['iou'], epoch)
        self.writer.add_scalar('Val/F1_strict_pi',     s['f1'],  epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_pi', t['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_pi',   t['f1'],  epoch)
        print(f'Per-image Strict:   F1={s["f1"]:.2f}%  mIOU={s["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t["f1"]:.2f}%  mIOU={t["iou"]:.2f}%')

        return val_total_avg, metrics, dr

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_miou':            self.best_miou,
            'best_shadow_iou':      self.best_shadow_iou,
            'best_f1':              self.best_f1,
            'best_decision_miou':   self.best_decision_miou,
            'epochs_without_improvement': self.epochs_without_improvement,
            # Losses
            'train_total':   self.train_total,
            'train_main':    self.train_main,
            'train_aux':     self.train_aux,
            'train_ii_raw':  self.train_ii_raw,
            'val_total':     self.val_total,
            'val_main':      self.val_main,
            'val_ii_raw':    self.val_ii_raw,
            # Metrics
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history':   self.val_metrics_history,
            'args': vars(self.args),
        }
        if is_best:
            p = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(ckpt, p)
            print(f'Best checkpoint saved to {p}')
        if epoch % self.args.save_freq == 0:
            p = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(ckpt, p)

    def load_checkpoint(self, path):
        print(f'Loading checkpoint from {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        try:
            self.model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print("WARNING: size mismatch — attempting partial load...")
                sd = ckpt['model_state_dict']
                md = self.model.state_dict()
                compat = {k: v for k, v in sd.items()
                          if k in md and v.size() == md[k].size()}
                md.update(compat)
                self.model.load_state_dict(md)
                print(f"Loaded {len(compat)}/{len(sd)} layers")
            else:
                raise

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_miou          = ckpt.get('best_miou', 0.0)
        self.best_shadow_iou    = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1            = ckpt.get('best_f1', 0.0)
        self.best_decision_miou = ckpt.get('best_decision_miou', 0.0)
        self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)

        self.train_total  = ckpt.get('train_total', [])
        self.train_main   = ckpt.get('train_main', [])
        self.train_aux    = ckpt.get('train_aux', [])
        self.train_ii_raw = ckpt.get('train_ii_raw', [])
        self.val_total    = ckpt.get('val_total', [])
        self.val_main     = ckpt.get('val_main', [])
        self.val_ii_raw   = ckpt.get('val_ii_raw', [])

        self.train_metrics_history = ckpt.get('train_metrics_history', {
            k: [] for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']})
        self.val_metrics_history = ckpt.get('val_metrics_history', {
            k: [] for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']})

        print(f'Resumed from epoch {ckpt["epoch"]}')

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self):
        print('\n' + '=' * 50)
        print('Starting MAMNet-IIM training...')
        print(f'IIM kernels: {self.args.num_kernels} × '
              f'{self.args.kernel_size}×{self.args.kernel_size}')
        if self.args.ii_loss_mode == 'adaptive':
            print(f'II loss: ADAPTIVE  target_ratio={self.args.ii_target_ratio} '
                  f'(~{100*self.args.ii_target_ratio:.0f}% of task loss)')
        else:
            print(f'II loss: FIXED  weight={self.args.ii_loss_weight}')
        print('=' * 50)

        patience = self.args.early_stopping_patience
        if patience > 0:
            label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                     if self.use_tolerant_decision else 'Strict per-image mIOU')
            print(f'Early stopping: patience={patience}  metric={label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            ep = epoch + 1
            self.train_epoch(ep)
            _, val_metrics, dr = self.validate(ep)

            decision_miou = self._get_decision_miou(dr)
            metric_label = (f'Tolerant ({self.tol_key}) mIOU'
                            if self.use_tolerant_decision
                            else 'Strict per-image mIOU')

            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, ep)

            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'>> New best {metric_label}: {self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']

            self.save_checkpoint(ep, is_best=is_best)
            self.writer.add_scalar('Train/LearningRate',
                                   self.optimizer.param_groups[0]['lr'], ep)

            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping after {patience} epochs '
                      f'without improvement in {metric_label}.')
                break
            print('=' * 50)

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (ref): {self.best_miou:.2f}%')

        print('\nGenerating plots...')
        plot_loss_curves_iim(
            self.train_total, self.val_total,
            self.train_main, self.val_main,
            self.train_aux,
            self.train_ii_raw, self.val_ii_raw,
            os.path.join(self.output_dir, 'loss_curves.png'),
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png'),
        )
        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test(self):
        print('\n' + '=' * 50)
        print('Testing model...')
        print('=' * 50)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self.load_checkpoint(best_ckpt)
        else:
            print('WARNING: best checkpoint not found, using current weights')

        self.model.eval()
        test_metrics = ShadowMetrics()
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs = self.model(images)
                filtered = filter_small_predictions(outputs, min_pixels=10)
                test_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics = test_metrics.compute()
        dr = detailed_eval.compute_metrics()

        print('\n' + '=' * 50)
        print('Pooled Test Results (reference):')
        print('=' * 50)
        for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']:
            print(f'{k:12s}: {metrics[k]:.2f}%')

        print('\n' + '=' * 50)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('=' * 50)
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        print(f"Strict   — F1: {s['f1']:.2f}%   mIOU: {s['iou']:.2f}%")
        print(f"Tolerant (±{self.args.boundary_tolerance}px) — "
              f"F1: {t['f1']:.2f}%   mIOU: {t['iou']:.2f}%")
        print(f"Pixels excluded by band: {t['pixels_excluded']} "
              f"({t['pct_excluded']:.1f}%)")

        if 'size_stratified' in dr:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in dr['size_stratified']:
                    m = dr['size_stratified'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'size_stratified_tolerant' in dr:
            print(f'\nSize-Stratified (Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in dr['size_stratified_tolerant']:
                    m = dr['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'fp_fn_analysis' in dr and 'fp' in dr['fp_fn_analysis']:
            fp = dr['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        results = {'standard': metrics, 'detailed': dr}
        rp = os.path.join(self.output_dir, 'test_results.json')
        with open(rp, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\nResults saved to {rp}')

        try:
            print('\nGenerating best/worst visualizations...')
            save_best_worst_visualizations(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except Exception as e:
            print(f'Visualization skipped: {e}')

        return metrics


# ======================================================================
# Entry point
# ======================================================================

def main():
    args = get_args()
    trainer = Trainer(args)
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()