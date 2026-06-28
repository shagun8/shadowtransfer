"""
Training script for DINOv3 + IIM (Illumination-Invariant Module).

Key differences from base train_dinov3.py
------------------------------------------
1. Model      : DINOv3ShadowDetectorIIM  (IIM front-end → backbone → decoder)
2. Loss       : Per-image mean Cross-Entropy  +  adaptively-weighted II consistency loss
3. Constraint : Zero-mean on IIM kernels enforced after every optimizer step
4. Tracking   : train_total / val_total,  train_main / val_main,  train_ii_raw / val_ii_raw
5. LR sched   : Epoch-based cosine warmup (NOT ReduceLROnPlateau — DINOv3 standard for ViT)
6. Viz        : Overview subplot (all losses, shared y-axis)  +
                individual subplot per component  (main CE train/val; II-raw train/val)
                — total loss is shown ONLY in the overview, never in individual panels

Decision metrics (best checkpoint, early stopping) are based on
per-image mIOU from DetailedEvaluator — never pooled ShadowMetrics.

When --eval_boundary_tolerant is set, decisions use per-image TOLERANT mIOU.
Otherwise decisions use per-image STRICT mIOU.

DetailedEvaluator ALWAYS runs regardless of --eval_boundary_tolerant.
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

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_iim_model import DINOv3ShadowDetectorIIM
from iim import compute_ii_loss
from data.dataset import get_dataloaders
from utils.losses import CrossEntropyLoss          # kept for reference / compat
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import save_best_worst_visualizations
from utils.evaluation_detailed import DetailedEvaluator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ======================================================================
# GPU diagnostics
# ======================================================================
print('=' * 50)
print('GPU DIAGNOSTICS')
print('=' * 50)
print(f'CUDA available:      {torch.cuda.is_available()}')
print(f'CUDA device count:   {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current CUDA device: {torch.cuda.current_device()}')
    print(f'CUDA device name:    {torch.cuda.get_device_name(0)}')
    print(f'CUDA capability:     {torch.cuda.get_device_capability(0)}')
print(f'CUDA_VISIBLE_DEVICES:{os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")}')
print('=' * 50)


# ======================================================================
# Per-image Cross-Entropy
# ======================================================================

class PerImageCrossEntropy(nn.Module):
    """
    Cross-Entropy loss computed per image then averaged across the batch.

    This matches how shadow detection papers report metrics (BDRAR, MTMT,
    DSDNet, SCOTCH) — per-image error rates, not globally pooled pixels.
    """

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        pred   : [B, 2, H, W]  logits
        target : [B, H, W]     integer labels {0, 1}
        Returns: scalar — mean of per-image CE losses
        """
        loss_map = F.cross_entropy(
            pred, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )  # [B, H, W]
        return loss_map.mean(dim=(1, 2)).mean()   # per-image mean → batch mean


# ======================================================================
# Combined loss: per-image CE  +  adaptive II consistency loss
# ======================================================================

class DINOv3IIMLoss(nn.Module):
    """
    total = main_CE  +  eff_weight × ii_raw

    DINOv3 has no auxiliary branches, so there is no aux term.

    II-loss weighting modes
    -----------------------
    adaptive (default)
        eff_weight  = ii_target_ratio × task_loss.detach() / (ii_raw.detach() + ε)
        → II contribution ≈ ii_target_ratio × task_loss at every step.
        Gradients flow only through ii_raw; task_loss is detached.

    fixed
        eff_weight  = ii_loss_weight  (static scalar)
    """

    def __init__(self, ii_loss_weight=0.01, ii_target_ratio=0.01,
                 ii_loss_mode='adaptive', weight=None):
        super().__init__()
        self.ii_loss_weight = ii_loss_weight
        self.ii_target_ratio = ii_target_ratio
        self.ii_loss_mode = ii_loss_mode
        self.criterion = PerImageCrossEntropy(weight=weight)

    def forward(self, outputs, target, ii_loss=None):
        """
        outputs  : dict with key 'main'  [B, 2, H, W]
        target   : [B, H, W]
        ii_loss  : scalar tensor (raw, unweighted) or None

        Returns dict:
            total, main, ii (weighted), ii_raw, ii_eff_weight
        """
        main_loss = self.criterion(outputs['main'], target)
        total = main_loss
        result = {'main': main_loss}

        if ii_loss is not None and ii_loss.item() > 0:
            if self.ii_loss_mode == 'adaptive':
                task_det  = total.detach()
                ii_det    = ii_loss.detach() + 1e-8
                eff_weight = self.ii_target_ratio * task_det / ii_det
                ii_weighted = eff_weight * ii_loss
            else:
                eff_weight  = torch.tensor(self.ii_loss_weight,
                                           device=main_loss.device)
                ii_weighted = self.ii_loss_weight * ii_loss

            total = total + ii_weighted
            result['ii']          = ii_weighted
            result['ii_raw']      = ii_loss
            result['ii_eff_weight'] = eff_weight
        else:
            z = torch.tensor(0.0, device=main_loss.device)
            result['ii']          = z
            result['ii_raw']      = z
            result['ii_eff_weight'] = z

        result['total'] = total
        return result


# ======================================================================
# LR schedule — identical to base DINOv3 (cosine warmup, epoch-based)
# ======================================================================

class CosineWarmupScheduler:
    """
    Cosine LR schedule with linear warmup.
    Standard for ViT fine-tuning — NOT metric-gated.
    """

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.base_lr       = base_lr
        self.min_lr        = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ======================================================================
# Visualisation
# ======================================================================

_C = {
    'total_train': '#2E86AB', 'total_val':  '#A23B72',
    'main_train':  '#F18F01', 'main_val':   '#C73E1D',
    'ii_train':    '#8338EC', 'ii_val':     '#3A86FF',
}
_METRIC_COLORS = {
    'OA': '#2E86AB', 'Precision': '#F18F01', 'F1': '#A23B72',
    'BER': '#E63946', 'mIOU': '#6A994E', 'Shadow_IOU': '#8338EC',
}

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11,  'axes.labelsize': 10,
    'xtick.labelsize': 9,  'ytick.labelsize': 9,
    'legend.fontsize': 9,  'figure.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


def plot_loss_curves_dinov3_iim(
    train_total, val_total,
    train_main,  val_main,
    train_ii_raw, val_ii_raw,
    save_path,
):
    """
    Layout
    ------
    Row 0  (full-width overview):
        ALL losses on a shared y-axis so relative magnitudes are visible.
        Includes: train/val total, train/val main CE, train/val II raw.

    Row 1  (individual panels — own y-scale each, total NOT shown):
        Panel 0: Main CE Loss  — train (solid) + val (dashed)
        Panel 1: II Loss raw   — train (solid) + val (dashed)

    Keeping total out of individual panels lets you see whether each
    component is actually decreasing on its own scale.
    """
    epochs = list(range(1, len(train_total) + 1))

    # ---- Decide which individual panels exist ----
    panels = []

    if train_main and len(train_main) == len(epochs):
        panels.append({
            'title': 'Main CE Loss  (per-image mean)',
            'ylabel': 'Loss',
            'series': [
                (train_main, 'Train Main CE', _C['main_train'], '-',  'o'),
                (val_main,   'Val Main CE',   _C['main_val'],   '--', 's'),
            ],
        })

    has_ii = train_ii_raw and any(v > 1e-10 for v in train_ii_raw)
    if has_ii:
        panels.append({
            'title': 'II Loss  (raw, unweighted)',
            'ylabel': 'Loss',
            'series': [
                (train_ii_raw, 'Train II raw', _C['ii_train'], '-',  'D'),
                (val_ii_raw,   'Val II raw',   _C['ii_val'],   '--', 'v'),
            ],
        })

    MAX_COLS  = 4
    n_ind     = len(panels)
    n_rows_ind = max(1, (n_ind + MAX_COLS - 1) // MAX_COLS) if n_ind else 0

    fig_w = max(9, min(5.5 * max(n_ind, 1), 22))
    fig_h = 4.2 + 3.8 * n_rows_ind

    if n_ind:
        fig   = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55,
                                  height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ---- Row 0: Overview (shared y-axis, all losses) ----
    ax_ov.plot(epochs, train_total, '-',  lw=2,   color=_C['total_train'],
               label='Train total',    marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_total,   '--', lw=1.8, color=_C['total_val'],
               label='Val total',      marker='s', ms=3, mfc='white', mew=1.2)
    if train_main and len(train_main) == len(epochs):
        ax_ov.plot(epochs, train_main, '-',  lw=1.4, color=_C['main_train'],
                   alpha=0.75, label='Train main CE')
        ax_ov.plot(epochs, val_main,   '--', lw=1.4, color=_C['main_val'],
                   alpha=0.75, label='Val main CE')
    if has_ii:
        ax_ov.plot(epochs, train_ii_raw, '-',  lw=1.2, color=_C['ii_train'],
                   alpha=0.70, label='Train II raw')
        ax_ov.plot(epochs, val_ii_raw,   '--', lw=1.2, color=_C['ii_val'],
                   alpha=0.70, label='Val II raw')

    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=7, framealpha=0.88, ncol=min(4, 2 + n_ind))
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ---- Row 1+: Individual panels (own y-scale, no total) ----
    if n_ind:
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows_ind, MAX_COLS, subplot_spec=outer[1],
            hspace=0.55, wspace=0.42)

        for pidx, panel in enumerate(panels):
            r, c = divmod(pidx, MAX_COLS)
            ax   = fig.add_subplot(inner[r, c])
            for vals, lbl, col, ls, mk in panel['series']:
                ep = list(range(1, len(vals) + 1))
                ax.plot(ep, vals, ls=ls, lw=1.8, color=col, label=lbl,
                        marker=mk, ms=3.5, mfc='white', mew=1.2, alpha=0.9)
            _style_ax(ax, panel['title'], panel['ylabel'])

        # Hide unused subplot slots
        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('DINOv3-IIM — Training Loss Curves', fontweight='bold',
                 fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


def plot_metrics_curves(train_hist, val_hist, save_path):
    """Per-metric train/val curves (pooled ShadowMetrics, reference only)."""
    metrics = list(train_hist.keys())
    n = len(metrics)
    if n == 0:
        return
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 4.0 * n_rows),
                              squeeze=False)
    for idx, m in enumerate(metrics):
        r, c  = divmod(idx, n_cols)
        ax    = axes[r][c]
        color = _METRIC_COLORS.get(m, '#333333')
        ax.plot(range(1, len(train_hist[m]) + 1), train_hist[m],
                '-', lw=1.8, color=color, label='Train',
                marker='o', ms=3.5, mfc='white', mew=1.2)
        ax.plot(range(1, len(val_hist[m]) + 1),   val_hist[m],
                '--', lw=1.8, color=color, label='Val',
                marker='s', ms=3.5, mfc='white', mew=1.2, alpha=0.75)
        _style_ax(ax, m, '%')
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)
    fig.suptitle('DINOv3-IIM — Metric Curves (pooled ShadowMetrics, reference)',
                 fontweight='bold', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Metric curves saved → {save_path}')


# ======================================================================
# CLI
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(description='Train DINOv3-IIM for Shadow Detection')

    # Data
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--img_size',  type=int, default=384)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # Multi-city / LOCO
    p.add_argument('--mode', type=str, default='loco',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities', type=str, nargs='+', default=None)

    # Model
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--model_name', type=str, default='dinov3_vits16',
                   choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--weights_path', type=str, default=None)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--frozen_stages', type=int, default=-1)

    # IIM-specific
    p.add_argument('--num_kernels', type=int, default=8,
                   help='Number of IIM learnable kernels (default 8)')
    p.add_argument('--kernel_size', type=int, default=5,
                   help='IIM kernel spatial size (default 5)')
    p.add_argument('--ii_loss_mode', type=str, default='adaptive',
                   choices=['adaptive', 'fixed'],
                   help='"adaptive" keeps II loss at ii_target_ratio of task loss. '
                        '"fixed" applies ii_loss_weight as a static multiplier.')
    p.add_argument('--ii_target_ratio', type=float, default=0.01,
                   help='Target fraction of task loss for II loss (adaptive, default 0.01)')
    p.add_argument('--ii_loss_weight', type=float, default=0.01,
                   help='Fixed weight for II loss (fixed mode only)')
    p.add_argument('--gamma_range_lo', type=float, default=0.5)
    p.add_argument('--gamma_range_hi', type=float, default=2.0)

    # Training (DINOv3 paper recommendations — cosine warmup for ViT)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-5,
                   help='Base LR for ViT fine-tuning (default 5e-5)')
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--min_lr', type=float, default=1e-6)

    # FDA (kept for compatibility, not used in IIM runs)
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L', type=float, default=0.01)

    # Checkpoints / logging
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq', type=int, default=5)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--device', type=str, default='cuda')

    # Boundary-tolerant evaluation
    p.add_argument('--eval_boundary_tolerant', action='store_true',
                   help='Use tolerant mIOU (instead of strict) for all decisions. '
                        'DetailedEvaluator always runs; this only selects the metric.')
    p.add_argument('--boundary_tolerance', type=int, default=2,
                   help="Don't-care band half-width in pixels (default 2).")

    # Early stopping
    p.add_argument('--early_stopping_patience', type=int, default=15,
                   help='Epochs without improvement before stopping. 0 = disabled.')

    # Comparison (passed from shell, optional)
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root', type=str, default=None)

    return p.parse_args()


# ======================================================================
# Trainer
# ======================================================================

class Trainer:
    def __init__(self, args):
        self.args   = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.tol_key      = f'tolerant_{args.boundary_tolerance}px'
        self.gamma_range  = (args.gamma_range_lo, args.gamma_range_hi)

        # ---- Output directory ----
        if args.mode == 'single':
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = f'dinov3_iim_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'dinov3_iim_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name  = f'dinov3_iim_loco_holdout_{test_city}_{args.resolution}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # ---- Model ----
        print('Initializing DINOv3-IIM model...')
        self.model = DINOv3ShadowDetectorIIM(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            num_kernels=args.num_kernels,
            kernel_size=args.kernel_size,
        ).to(self.device)

        # IIM kernel diagnostics
        with torch.no_grad():
            k = self.model.iim.kernels
            print(f'IIM kernel stats — '
                  f'std: {k.std().item():.4f}  '
                  f'abs_mean: {k.abs().mean().item():.4f}  '
                  f'zero-mean check: {k.mean(dim=(-2, -1)).abs().max().item():.1e}')

        # ---- Loss ----
        self.criterion = DINOv3IIMLoss(
            ii_loss_weight=args.ii_loss_weight,
            ii_target_ratio=args.ii_target_ratio,
            ii_loss_mode=args.ii_loss_mode,
        )
        print(f'>> II loss mode: {args.ii_loss_mode}  '
              f'{"target_ratio=" + str(args.ii_target_ratio) if args.ii_loss_mode == "adaptive" else "weight=" + str(args.ii_loss_weight)}')

        # ---- Optimizer (AdamW, DINOv3 paper) ----
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

        # ---- LR scheduler — epoch-based cosine warmup (NOT metric-gated) ----
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
        )

        # ---- Decision metric ----
        self.use_tolerant_for_decisions = args.eval_boundary_tolerant
        if self.use_tolerant_for_decisions:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU')

        # DetailedEvaluator ALWAYS runs (flag only controls decision path)
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ---- Tracking ----
        self.start_epoch = 0
        self.best_decision_miou          = 0.0
        self.best_strict_pooled_miou     = 0.0   # pooled, reference only
        self.best_shadow_iou             = 0.0
        self.best_f1                     = 0.0
        self.epochs_without_improvement  = 0

        # Loss histories
        self.train_total   = []
        self.train_main    = []
        self.train_ii_raw  = []
        self.val_total     = []
        self.val_main      = []
        self.val_ii_raw    = []

        # Pooled metric histories (reference only)
        self.train_metrics_history = {
            k: [] for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
        self.val_metrics_history = {
            k: [] for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}

        if args.resume:
            self.load_checkpoint(args.resume)

        # ---- Dataloaders ----
        print('\nLoading datasets...')
        if args.use_fda:
            print(f'FDA enabled: L={args.fda_L}, target_root={args.fda_target_root}')

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
            use_fda=args.use_fda,
            fda_target_root=args.fda_target_root,
            fda_L=args.fda_L,
        )
        print(f'Training samples:   {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples:       {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    # Decision metric helper
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """
        Return the mIOU that drives checkpoint selection and early stopping.
        Both options are per-image means from DetailedEvaluator.
        """
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_for_decisions:
            return bt[self.tol_key]['iou']
        else:
            return bt['strict']['iou']

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        """
        Returns: (ep_total, ep_main, ep_ii_raw, metrics_dict)
        """
        self.model.train()

        ep_total = ep_main = ep_ii_raw = ep_ii_eff = ep_ii_w = 0.0
        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start = time.time()

        for bi, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            # ---- Forward ----
            outputs = self.model(images)  # dict: 'main', 'iim_features'

            # ---- II consistency loss ----
            ii_loss = compute_ii_loss(
                outputs['iim_features'],
                self.model.iim,
                images,
                gamma_range=self.gamma_range,
                beta=1.0,
            )

            # ---- Combined loss ----
            losses = self.criterion(outputs, masks, ii_loss=ii_loss)
            loss   = losses['total']

            # ---- Backward ----
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- Enforce zero-mean constraint on IIM kernels ----
            self.model.iim.enforce_zero_mean()

            # ---- Metrics ----
            filtered = filter_small_predictions(outputs['main'], min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            ep_total  += losses['total'].item()
            ep_main   += losses['main'].item()
            ep_ii_raw += losses['ii_raw'].item()
            ep_ii_eff += losses['ii'].item()
            ew = losses['ii_eff_weight']
            ep_ii_w   += ew.item() if torch.is_tensor(ew) else float(ew)

            if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                ew_v  = ew.item() if torch.is_tensor(ew) else float(ew)
                ii_pct = 100.0 * losses['ii'].item() / (losses['total'].item() + 1e-12)
                print(f'  Batch [{bi+1}/{num_batches}] '
                      f'total={loss.item():.4f}  '
                      f'main={losses["main"].item():.4f}  '
                      f'II_raw={losses["ii_raw"].item():.2e}  '
                      f'w={ew_v:.1f}  II_eff={losses["ii"].item():.4f} [{ii_pct:.1f}%]')

        n             = num_batches
        ep_total     /= n
        ep_main      /= n
        ep_ii_raw    /= n
        ep_ii_eff    /= n
        ep_ii_w      /= n

        metrics     = train_metrics.compute()
        elapsed     = time.time() - start
        ii_pct_avg  = 100.0 * ep_ii_eff / (ep_total + 1e-12)

        print(f'\nTraining Results:')
        print(f'  Time={elapsed:.1f}s  total={ep_total:.4f}  '
              f'main={ep_main:.4f}  '
              f'II_raw={ep_ii_raw:.2e}  II_eff={ep_ii_eff:.4f} [{ii_pct_avg:.1f}%]  '
              f'avg_w={ep_ii_w:.1f}')
        print(f'  OA={metrics["OA"]:.2f}%  Prec={metrics["Precision"]:.2f}%  '
              f'F1={metrics["F1"]:.2f}%  BER={metrics["BER"]:.2f}%  '
              f'mIOU(pooled)={metrics["mIOU"]:.2f}%  ShadIOU={metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Train/Loss',           ep_total,  epoch)
        self.writer.add_scalar('Train/MainLoss',       ep_main,   epoch)
        self.writer.add_scalar('Train/IILoss_raw',     ep_ii_raw, epoch)
        self.writer.add_scalar('Train/IILoss_eff',     ep_ii_eff, epoch)
        self.writer.add_scalar('Train/II_eff_weight',  ep_ii_w,   epoch)
        for k in self.train_metrics_history:
            self.writer.add_scalar(f'Train/{k}', metrics[k], epoch)

        # Store histories
        self.train_total.append(ep_total)
        self.train_main.append(ep_main)
        self.train_ii_raw.append(ep_ii_raw)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(metrics[k])

        # DetailedEvaluator — per-image strict + tolerant
        dr = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_perimage',   s['iou'], epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     s['f1'],  epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', t['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   t['f1'],  epoch)
        print(f'  Per-image Strict:   F1={s["f1"]:.2f}%  mIOU={s["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t["f1"]:.2f}%  mIOU={t["iou"]:.2f}%')

        return ep_total, ep_main, ep_ii_raw, metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        """
        Returns: (val_total_avg, metrics_dict, detailed_results)
        """
        print('\nValidating...')
        self.model.eval()

        val_main_sum   = 0.0
        val_ii_raw_sum = 0.0
        val_metrics    = ShadowMetrics()
        per_img_ce     = PerImageCrossEntropy()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs = self.model(images)   # eval mode → logits [B, 2, H, W]

                # Per-image main CE
                val_main_sum += per_img_ce(outputs, masks).item()

                # II loss tracking (no gradient needed; extract_features is pure conv)
                feats_orig = self.model.iim.extract_features(images)
                ii_loss    = compute_ii_loss(
                    feats_orig, self.model.iim, images,
                    gamma_range=self.gamma_range, beta=1.0)
                val_ii_raw_sum += ii_loss.item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        n              = len(self.dataloaders['val'])
        val_main_avg   = val_main_sum   / n
        val_ii_raw_avg = val_ii_raw_sum / n

        # Reconstruct effective total with the same adaptive logic used in training
        if self.args.ii_loss_mode == 'adaptive' and val_ii_raw_avg > 0:
            ii_w = self.args.ii_target_ratio * val_main_avg / (val_ii_raw_avg + 1e-8)
        else:
            ii_w = self.args.ii_loss_weight
        val_ii_eff_avg  = ii_w * val_ii_raw_avg
        val_total_avg   = val_main_avg + val_ii_eff_avg

        metrics = val_metrics.compute()
        ii_pct  = 100.0 * val_ii_eff_avg / (val_total_avg + 1e-12)

        print(f'Validation Results:')
        print(f'  total={val_total_avg:.4f}  main={val_main_avg:.4f}  '
              f'II_raw={val_ii_raw_avg:.2e}  II_eff={val_ii_eff_avg:.4f} [{ii_pct:.1f}%]  '
              f'w={ii_w:.1f}')
        print(f'  OA={metrics["OA"]:.2f}%  Prec={metrics["Precision"]:.2f}%  '
              f'F1={metrics["F1"]:.2f}%  BER={metrics["BER"]:.2f}%  '
              f'mIOU(pooled)={metrics["mIOU"]:.2f}%  ShadIOU={metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Val/Loss',       val_total_avg,  epoch)
        self.writer.add_scalar('Val/MainLoss',   val_main_avg,   epoch)
        self.writer.add_scalar('Val/IILoss_raw', val_ii_raw_avg, epoch)
        for k in self.val_metrics_history:
            self.writer.add_scalar(f'Val/{k}', metrics[k], epoch)

        # Store histories
        self.val_total.append(val_total_avg)
        self.val_main.append(val_main_avg)
        self.val_ii_raw.append(val_ii_raw_avg)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(metrics[k])

        # DetailedEvaluator
        dr = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Val/mIOU_strict_perimage',   s['iou'], epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     s['f1'],  epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', t['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   t['f1'],  epoch)
        print(f'  Per-image Strict:   F1={s["f1"]:.2f}%  mIOU={s["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t["f1"]:.2f}%  mIOU={t["iou"]:.2f}%')

        return val_total_avg, metrics, dr

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch':                     epoch,
            'model_state_dict':          self.model.state_dict(),
            'optimizer_state_dict':      self.optimizer.state_dict(),
            'best_decision_miou':        self.best_decision_miou,
            'best_strict_pooled_miou':   self.best_strict_pooled_miou,
            'best_shadow_iou':           self.best_shadow_iou,
            'best_f1':                   self.best_f1,
            'epochs_without_improvement': self.epochs_without_improvement,
            'use_tolerant_for_decisions': self.use_tolerant_for_decisions,
            'train_total':    self.train_total,
            'train_main':     self.train_main,
            'train_ii_raw':   self.train_ii_raw,
            'val_total':      self.val_total,
            'val_main':       self.val_main,
            'val_ii_raw':     self.val_ii_raw,
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history':   self.val_metrics_history,
            'args': vars(self.args),
        }
        latest = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(ckpt, latest)
        print(f'Checkpoint saved → {latest}')

        if is_best:
            best = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(ckpt, best)
            print(f'Best checkpoint saved → {best}')

        if epoch % self.args.save_freq == 0:
            ep_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(ckpt, ep_path)

    def load_checkpoint(self, path):
        print(f'Loading checkpoint from {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1

        self.best_decision_miou       = ckpt.get('best_decision_miou', 0.0)
        self.best_strict_pooled_miou  = ckpt.get('best_strict_pooled_miou', 0.0)
        self.best_shadow_iou          = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1                  = ckpt.get('best_f1', 0.0)
        self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)

        self.train_total   = ckpt.get('train_total',  [])
        self.train_main    = ckpt.get('train_main',   [])
        self.train_ii_raw  = ckpt.get('train_ii_raw', [])
        self.val_total     = ckpt.get('val_total',    [])
        self.val_main      = ckpt.get('val_main',     [])
        self.val_ii_raw    = ckpt.get('val_ii_raw',   [])

        empty = {k: [] for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']}
        self.train_metrics_history = ckpt.get('train_metrics_history', empty)
        self.val_metrics_history   = ckpt.get('val_metrics_history',   empty)

        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict per-image')
        print(f'Resumed from epoch {ckpt["epoch"]}')
        print(f'Best decision mIOU ({metric_label}): {self.best_decision_miou:.2f}%  '
              f'Epochs w/o improvement: {self.epochs_without_improvement}')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print('\n' + '=' * 50)
        print('Starting DINOv3-IIM training...')
        print(f'  IIM kernels: {self.args.num_kernels} × '
              f'{self.args.kernel_size}×{self.args.kernel_size}')
        if self.args.ii_loss_mode == 'adaptive':
            print(f'  II loss: ADAPTIVE  target_ratio={self.args.ii_target_ratio}  '
                  f'(≈{100*self.args.ii_target_ratio:.0f}% of task loss)')
        else:
            print(f'  II loss: FIXED  weight={self.args.ii_loss_weight}')
        print('=' * 50)

        patience     = self.args.early_stopping_patience
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_for_decisions
                        else 'Strict per-image mIOU')
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            # Update LR (cosine warmup — epoch-based, NOT metric-gated)
            current_lr = self.scheduler.step(epoch)
            print(f'\nLearning rate: {current_lr:.2e}')

            # Train
            _, _, _, _ = self.train_epoch(epoch + 1)

            # Validate
            _, val_metrics, dr = self.validate(epoch + 1)

            # Decision metric
            decision_miou = self._get_decision_miou(dr)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            # Best checkpoint
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'*** New best {metric_label}: {self.best_decision_miou:.2f}% ***')
            else:
                self.epochs_without_improvement += 1

            # Track pooled bests (reference logging only)
            if val_metrics['mIOU'] > self.best_strict_pooled_miou:
                self.best_strict_pooled_miou = val_metrics['mIOU']
                print(f'New best strict pooled mIOU (ref): {self.best_strict_pooled_miou:.2f}%')
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
                print(f'New best Shadow IoU: {self.best_shadow_iou:.2f}%')
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                print(f'New best F1: {self.best_f1:.2f}%')

            self.save_checkpoint(epoch + 1, is_best=is_best)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)
            print('=' * 50)

            # Early stopping
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping: no {metric_label} improvement for '
                      f'{patience} epochs.')
                break

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best strict pooled mIOU (ref): {self.best_strict_pooled_miou:.2f}%')
        print(f'Best Shadow IoU:               {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                       {self.best_f1:.2f}%')

        print('\nGenerating plots...')
        plot_loss_curves_dinov3_iim(
            self.train_total, self.val_total,
            self.train_main,  self.val_main,
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
        print('Testing model (best checkpoint)...')
        print('=' * 50)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self.load_checkpoint(best_ckpt)
        else:
            print('WARNING: best checkpoint not found, using current model weights.')

        self.model.eval()
        test_metrics  = ShadowMetrics()
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs  = self.model(images)    # eval → logits
                filtered = filter_small_predictions(outputs, min_pixels=10)
                test_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics = test_metrics.compute()
        dr      = detailed_eval.compute_metrics()

        print('\n' + '=' * 50)
        print('Pooled Test Results (reference):')
        print('=' * 50)
        for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']:
            print(f'  {k:12s}: {metrics[k]:.2f}%')

        print('\n' + '=' * 50)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('=' * 50)
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        print(f"  Strict   — F1: {s['f1']:.2f}%   mIOU: {s['iou']:.2f}%")
        print(f"  Tolerant (±{self.args.boundary_tolerance}px) — "
              f"F1: {t['f1']:.2f}%   mIOU: {t['iou']:.2f}%")
        print(f"  Pixels excluded by band: {t['pixels_excluded']} "
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

        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({'standard': metrics, 'detailed': dr}, f, indent=4)
        print(f'\nResults saved → {results_path}')

        try:
            print('\nGenerating best/worst prediction visualizations...')
            save_best_worst_visualizations(
                self.model,
                self.dataloaders['test'],
                self.device,
                self.output_dir,
                num_images=10,
            )
        except Exception as e:
            print(f'Visualization skipped: {e}')

        return metrics


# ======================================================================
# Entry point
# ======================================================================

def main():
    args    = get_args()
    trainer = Trainer(args)
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()