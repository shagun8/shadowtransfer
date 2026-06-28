"""
Training script for OGLANet + IIM (Illumination-Invariant Module).

Key differences from base train.py
------------------------------------
1. Model     : OGLANetIIM  (IIM → GLAMEncoder → DFFM → Decoder → OAM)
2. Loss      : Per-image mean CE across all 6 OAM outputs + adaptive II loss
               (II loss target ratio = 1 % of the 6-head task loss by default)
3. Optimizer : Adam at lr=0.0001 — IIM kernels (0.008 M) and the pretrained
               ResNet-34 backbone (~21 M) have very different gradient scales;
               Adam's per-parameter adaptivity is more robust than Adamax here.
               (Adamax was chosen for base OGLANet which had no such scale gap.)
4. Constraint: Zero-mean projection on IIM kernels after every optimizer step.
5. Tracking  : Separate histories for loss1-6 (train), val P6-CE, II-raw
               (train + val).  Val total = val_p6_ce + adaptive-weighted II.
6. Viz       : One overview panel (all losses, shared y-axis) +
               one panel per component on its own y-axis (no total in panels).
               Panels: val-P6-CE | loss1 | loss2 | loss3 | loss4 | loss5 |
                       loss6 | II-raw (train+val together).

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

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_iim import OGLANetIIM
from models.iim import compute_ii_loss
from data.dataset import get_dataloaders
from utils.evaluation_detailed import DetailedEvaluator
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import save_best_worst_visualizations

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
print(f"CUDA available:         {torch.cuda.is_available()}")
print(f"CUDA device count:      {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device:    {torch.cuda.current_device()}")
    print(f"CUDA device name:       {torch.cuda.get_device_name(0)}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA_VISIBLE_DEVICES:   {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("=" * 50)


# ======================================================================
# Loss components
# ======================================================================

class PerImageCrossEntropy(nn.Module):
    """
    Cross-entropy loss computed per image then averaged across the batch.

    Per-image averaging prevents images with large shadow regions from
    dominating the gradient — consistent with how shadow detection papers
    (BDRAR, MTMT, DSDNet, SCOTCH) compute their evaluation metrics.
    """

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args
        ----
        pred   : [B, 2, H, W]  logits
        target : [B, H, W]     integer class labels {0, 1}

        Returns
        -------
        scalar — mean over images of the per-image spatial mean CE
        """
        loss_map = F.cross_entropy(
            pred, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )                                          # [B, H, W]
        return loss_map.mean(dim=(1, 2)).mean()    # per-image mean → batch mean


class OGLANetIIMLoss(nn.Module):
    """
    Combined loss: per-image CE for all 6 OAM prediction heads + II loss.

    Task loss  = Σ per-image-CE(P1…P6)   (6 terms, all equal weight)
    II loss is weighted adaptively so its contribution stays at
    ``ii_target_ratio`` × task_loss throughout training (adaptive mode),
    or with a fixed scalar (fixed mode).

    Adaptive rationale
    ------------------
    The raw II loss (~1e-3) is orders of magnitude smaller than the CE
    terms (~0.5 each × 6).  A fixed weight either vanishes or requires
    per-dataset tuning.  Adaptive weighting keeps the II signal at a
    constant fraction of the task signal regardless of training stage
    or dataset statistics.  Gradient flows only through ii_raw; the
    detached task_loss is used solely to compute the dynamic weight.
    """

    def __init__(self, ii_loss_weight=0.01, ii_target_ratio=0.01,
                 ii_loss_mode='adaptive', weight=None):
        super().__init__()
        self.ii_loss_weight = ii_loss_weight
        self.ii_target_ratio = ii_target_ratio
        self.ii_loss_mode = ii_loss_mode
        # Shared per-image CE criterion for all 6 heads
        self.criterion = PerImageCrossEntropy(weight=weight)

    def forward(self, predictions, target, ii_loss=None):
        """
        Args
        ----
        predictions : dict  keys 'p1'…'p6' (and 'iim_features', ignored here)
        target      : [B, H, W]  ground-truth labels
        ii_loss     : scalar tensor (raw, unweighted) or None

        Returns
        -------
        dict with keys:
            total, loss1…loss6,
            ii (weighted), ii_raw, ii_eff_weight
        """
        l1 = self.criterion(predictions['p1'], target)
        l2 = self.criterion(predictions['p2'], target)
        l3 = self.criterion(predictions['p3'], target)
        l4 = self.criterion(predictions['p4'], target)
        l5 = self.criterion(predictions['p5'], target)
        l6 = self.criterion(predictions['p6'], target)

        task_loss = l1 + l2 + l3 + l4 + l5 + l6
        total = task_loss

        result = {
            'loss1': l1, 'loss2': l2, 'loss3': l3,
            'loss4': l4, 'loss5': l5, 'loss6': l6,
        }

        if ii_loss is not None and ii_loss.item() > 0:
            if self.ii_loss_mode == 'adaptive':
                # Adaptive weight: II contribution ≈ target_ratio × task_loss
                task_detached = task_loss.detach()
                ii_detached   = ii_loss.detach() + 1e-8
                eff_weight    = self.ii_target_ratio * task_detached / ii_detached
                ii_weighted   = eff_weight * ii_loss
            else:
                eff_weight  = torch.tensor(self.ii_loss_weight,
                                           device=l1.device)
                ii_weighted = self.ii_loss_weight * ii_loss

            total = total + ii_weighted
            result['ii']          = ii_weighted
            result['ii_raw']      = ii_loss
            result['ii_eff_weight'] = eff_weight
        else:
            zero = torch.tensor(0.0, device=l1.device)
            result['ii']          = zero
            result['ii_raw']      = zero
            result['ii_eff_weight'] = zero

        result['total'] = total
        return result


# ======================================================================
# Visualisation
# ======================================================================

matplotlib.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 11,   'axes.labelsize': 10,
    'xtick.labelsize': 9,   'ytick.labelsize': 9,
    'legend.fontsize': 9,   'figure.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Colour palette
_C = {
    'train_total': '#2E86AB',
    'val_total':   '#A23B72',
    'loss1':       '#F18F01',
    'loss2':       '#C73E1D',
    'loss3':       '#6A994E',
    'loss4':       '#8338EC',
    'loss5':       '#FF6B35',
    'loss6':       '#0D6E75',
    'ii_train':    '#E63946',
    'ii_val':      '#3A86FF',
    'val_p6':      '#A23B72',
}

_METRIC_COLORS = {
    'OA':         '#2E86AB', 'Precision':  '#F18F01',
    'F1':         '#A23B72', 'BER':        '#E63946',
    'mIOU':       '#6A994E', 'Shadow_IOU': '#8338EC',
}

MAX_COLS = 4


def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


def plot_loss_curves_iim(
    train_total, val_total,
    train_loss1, train_loss2, train_loss3,
    train_loss4, train_loss5, train_loss6,
    train_ii_raw, val_p6_ce, val_ii_raw,
    save_path,
):
    """
    Two-section loss figure.

    Section 0 — Overview (full-width, shared y-axis)
        All tracked series overlaid so relative magnitudes are visible.
        Includes: train/val totals, all 6 component CEs, train/val II-raw.

    Section 1+ — Individual panels (one per component, own y-axis)
        Each panel shows exactly one loss type so fine-grained trends
        are legible.  Total loss is NOT shown here.
        Panels (left-to-right, top-to-bottom):
          1. Val P6-CE          (val only)
          2. Train loss1        (train only)
          3. Train loss2        (train only)
          4. Train loss3        (train only)
          5. Train loss4        (train only)
          6. Train loss5        (train only)
          7. Train loss6        (train only)
          8. II-raw             (train + val on same panel)

    Parameters
    ----------
    train_total, val_total : list[float]  total loss per epoch
    train_loss1…6          : list[float]  per-head CE (train)
    train_ii_raw           : list[float]  raw (unweighted) II loss (train)
    val_p6_ce              : list[float]  per-image CE on P6 (val)
    val_ii_raw             : list[float]  raw II loss (val)
    save_path              : str          output PNG path
    """
    epochs = list(range(1, len(train_total) + 1))

    # ---- Build individual panel specs ----
    component_pairs = [
        ('loss1', train_loss1), ('loss2', train_loss2),
        ('loss3', train_loss3), ('loss4', train_loss4),
        ('loss5', train_loss5), ('loss6', train_loss6),
    ]

    panels = []

    # Panel 1: val P6-CE
    if val_p6_ce and len(val_p6_ce) == len(epochs):
        panels.append({
            'title': 'Val P6-CE (per-image)',
            'ylabel': 'Loss',
            'series': [
                (val_p6_ce, 'Val P6-CE', _C['val_p6'], '--', 's'),
            ],
        })

    # Panels 2-7: individual CE heads (train only)
    for name, vals in component_pairs:
        if vals and len(vals) == len(epochs):
            panels.append({
                'title': f'Train {name}',
                'ylabel': 'Loss',
                'series': [
                    (vals, f'Train {name}', _C[name], '-', 'o'),
                ],
            })

    # Panel 8: II-raw (train + val together — same metric, both splits)
    has_ii = (train_ii_raw and any(v > 1e-12 for v in train_ii_raw))
    if has_ii:
        series_ii = [(train_ii_raw, 'Train II-raw', _C['ii_train'], '-', 'D')]
        if val_ii_raw and len(val_ii_raw) == len(epochs):
            series_ii.append((val_ii_raw, 'Val II-raw', _C['ii_val'], '--', 'v'))
        panels.append({
            'title': 'II Loss — raw unweighted (train + val)',
            'ylabel': 'Loss',
            'series': series_ii,
        })

    n_ind      = len(panels)
    n_rows_ind = max(1, (n_ind + MAX_COLS - 1) // MAX_COLS) if n_ind else 0

    fig_w = max(10, min(5.5 * max(n_ind, 1), 24))
    fig_h = 4.2 + 3.8 * n_rows_ind

    if n_ind:
        fig   = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.55,
            height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ---- Section 0: Overview (shared y-axis) -------------------------
    ax_ov.plot(epochs, train_total, '-', lw=2.2, color=_C['train_total'],
               label='Train total', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_total, '--', lw=2.0, color=_C['val_total'],
               label='Val total (P6-CE + II)', marker='s', ms=3,
               mfc='white', mew=1.2)

    for name, vals in component_pairs:
        if vals and len(vals) == len(epochs):
            ax_ov.plot(epochs, vals, '-', lw=1.2, color=_C[name],
                       alpha=0.60, label=f'Train {name}')

    if has_ii:
        ax_ov.plot(epochs, train_ii_raw, '-', lw=1.2, color=_C['ii_train'],
                   alpha=0.60, label='Train II-raw')
        if val_ii_raw and len(val_ii_raw) == len(epochs):
            ax_ov.plot(epochs, val_ii_raw, '--', lw=1.2, color=_C['ii_val'],
                       alpha=0.60, label='Val II-raw')

    if val_p6_ce and len(val_p6_ce) == len(epochs):
        ax_ov.plot(epochs, val_p6_ce, ':', lw=1.4, color=_C['val_p6'],
                   alpha=0.75, label='Val P6-CE')

    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=7, framealpha=0.88, ncol=5)
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ---- Section 1+: Individual panels (own y-axis, no total) --------
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

        for pidx in range(n_ind, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('OGLANet-IIM — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)
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
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    for idx, m in enumerate(metrics):
        r, c  = divmod(idx, n_cols)
        ax    = axes[r][c]
        color = _METRIC_COLORS.get(m, '#333333')
        ep_tr = list(range(1, len(train_hist[m]) + 1))
        ep_vl = list(range(1, len(val_hist[m]) + 1))
        ax.plot(ep_tr, train_hist[m], '-', lw=1.8, color=color,
                label='Train', marker='o', ms=3.5, mfc='white', mew=1.2)
        ax.plot(ep_vl, val_hist[m],   '--', lw=1.8, color=color,
                label='Val',   marker='s', ms=3.5, mfc='white', mew=1.2,
                alpha=0.75)
        _style_ax(ax, m, '%')
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)
    fig.suptitle('OGLANet-IIM — Metric Curves (pooled ShadowMetrics, reference)',
                 fontweight='bold', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Metric curves saved → {save_path}')


# ======================================================================
# CLI
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(description='Train OGLANet-IIM')

    # Data
    p.add_argument('--data_root',    type=str, default=None)
    p.add_argument('--img_size',     type=int, default=384)
    p.add_argument('--batch_size',   type=int, default=8)
    p.add_argument('--num_workers',  type=int, default=1)

    # Multi-city / LOCO
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution',   type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id',      type=int, default=None,
                   choices=[0, 1, 2])
    p.add_argument('--cities',       type=str, nargs='+', default=None)

    # Model
    p.add_argument('--num_classes',  type=int,  default=2)
    p.add_argument('--pretrained',   action='store_true', default=True)
    p.add_argument('--use_contrast', action='store_true')

    # IIM-specific
    p.add_argument('--num_kernels', type=int, default=8,
                   help='Number of IIM learnable kernels (default 8)')
    p.add_argument('--kernel_size', type=int, default=5,
                   help='IIM kernel spatial size (default 5)')
    p.add_argument('--ii_loss_mode', type=str, default='adaptive',
                   choices=['adaptive', 'fixed'],
                   help='"adaptive" scales II loss to ii_target_ratio of task loss; '
                        '"fixed" uses ii_loss_weight as a static multiplier.')
    p.add_argument('--ii_target_ratio', type=float, default=0.01,
                   help='Target fraction of task loss for II loss (adaptive, default 0.01)')
    p.add_argument('--ii_loss_weight',  type=float, default=0.01,
                   help='Static weight for II loss (fixed mode only)')
    p.add_argument('--gamma_range_lo', type=float, default=0.5)
    p.add_argument('--gamma_range_hi', type=float, default=2.0)

    # Training
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--lr',           type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # FDA (kept for compatibility with shared data pipeline)
    p.add_argument('--use_fda',          action='store_true')
    p.add_argument('--fda_target_root',  type=str, default=None)
    p.add_argument('--fda_L',           type=float, default=0.01)

    # Checkpoints
    p.add_argument('--output_dir',  type=str, default='./outputs')
    p.add_argument('--save_freq',   type=int, default=10)
    p.add_argument('--resume',      type=str, default=None)
    p.add_argument('--eval_only',   action='store_true')
    p.add_argument('--device',      type=str, default='cuda')

    # Boundary-tolerant evaluation
    p.add_argument('--eval_boundary_tolerant', action='store_true',
                   help='Use tolerant mIOU for all decisions. '
                        'DetailedEvaluator always runs regardless of this flag.')
    p.add_argument('--boundary_tolerance', type=int, default=2,
                   help="Don't-care band half-width in pixels (default 2).")

    # Early stopping
    p.add_argument('--early_stopping_patience', type=int, default=0,
                   help='Epochs without improvement before stopping. 0 = disabled.')

    return p.parse_args()


# ======================================================================
# Trainer
# ======================================================================

class Trainer:

    def __init__(self, args):
        self.args   = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        self.tol_key     = f'tolerant_{args.boundary_tolerance}px'
        self.gamma_range = (args.gamma_range_lo, args.gamma_range_hi)

        # ---- Output directory ----
        modifiers = ['iim']
        if args.use_fda:
            modifiers.append('fda')
        mod_str = '_'.join(modifiers)

        if args.mode == 'single':
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = f'oglanet_{mod_str}_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'oglanet_{mod_str}_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name  = (f'oglanet_{mod_str}_loco_holdout_'
                         f'{test_city}_{args.resolution}_1')

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ---- Model ----
        print('Initializing OGLANet-IIM...')
        self.model = OGLANetIIM(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            use_contrast=args.use_contrast,
            num_kernels=args.num_kernels,
            kernel_size=args.kernel_size,
        ).to(self.device)

        total_p = sum(p.numel() for p in self.model.parameters())
        train_p = sum(p.numel() for p in self.model.parameters()
                      if p.requires_grad)
        iim_p   = sum(p.numel() for p in self.model.iim.parameters())
        print(f'Total parameters:     {total_p:,}')
        print(f'Trainable parameters: {train_p:,}')
        print(f'IIM parameters:       {iim_p:,}  ({iim_p / 1e6:.4f} M)')

        with torch.no_grad():
            k = self.model.iim.kernels
            print(f'IIM kernel stats — '
                  f'std: {k.std().item():.4f}  '
                  f'abs_mean: {k.abs().mean().item():.4f}  '
                  f'zero-mean check: '
                  f'{k.mean(dim=(-2,-1)).abs().max().item():.1e}')

        # ---- Loss ----
        self.criterion = OGLANetIIMLoss(
            ii_loss_weight=args.ii_loss_weight,
            ii_target_ratio=args.ii_target_ratio,
            ii_loss_mode=args.ii_loss_mode,
        )
        # Separate criterion for validation (per-image CE on P6 only)
        self.val_criterion = PerImageCrossEntropy()

        mode_desc = (
            f'ADAPTIVE  target_ratio={args.ii_target_ratio}'
            if args.ii_loss_mode == 'adaptive'
            else f'FIXED  weight={args.ii_loss_weight}'
        )
        print(f'>> II loss: {mode_desc}')

        # ---- Optimiser — Adam handles IIM/backbone scale gap ----
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5)

        # ---- Decision metric ----
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU')

        # Always-on DetailedEvaluators
        self.detailed_eval_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_eval_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ---- Tracking ----
        self.start_epoch               = 0
        self.best_miou                 = 0.0
        self.best_shadow_iou           = 0.0
        self.best_f1                   = 0.0
        self.best_decision_miou        = 0.0
        self.epochs_without_improvement = 0

        # Loss histories (train)
        self.train_total  = []
        self.train_loss1  = []
        self.train_loss2  = []
        self.train_loss3  = []
        self.train_loss4  = []
        self.train_loss5  = []
        self.train_loss6  = []
        self.train_ii_raw = []   # raw, unweighted II loss

        # Loss histories (val)
        self.val_total   = []    # val_p6_ce + adaptive-weighted II
        self.val_p6_ce   = []    # per-image CE on P6
        self.val_ii_raw  = []    # raw II loss (val)

        # Pooled metric histories (reference only)
        _metric_keys = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = {k: [] for k in _metric_keys}
        self.val_metrics_history   = {k: [] for k in _metric_keys}

        if args.resume:
            self.load_checkpoint(args.resume)

        # ---- Dataloaders ----
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
            use_contrast=args.use_contrast,
        )
        print(f'Train: {len(self.dataloaders["train"].dataset)}  '
              f'Val:   {len(self.dataloaders["val"].dataset)}  '
              f'Test:  {len(self.dataloaders["test"].dataset)}')

        if args.early_stopping_patience and args.early_stopping_patience > 0:
            print(f'>> Early stopping patience: {args.early_stopping_patience}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """Per-image mIOU from DetailedEvaluator that drives all decisions."""
        bt = detailed_results['boundary_tolerant']
        return (bt[self.tol_key]['iou'] if self.use_tolerant_decision
                else bt['strict']['iou'])

    def _metric_label(self):
        return (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                if self.use_tolerant_decision else 'Strict per-image mIOU')

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()

        ep = {k: 0.0 for k in
              ['total', 'loss1', 'loss2', 'loss3', 'loss4', 'loss5',
               'loss6', 'ii_raw', 'ii_eff', 'ii_w']}
        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 55)
        start = time.time()

        for bi, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            # ---------- Forward ----------
            outputs = self.model(images)
            # training → dict: p1-p6 + iim_features

            # ---------- II loss ----------
            rgb = images[:, :3]
            ii_loss = compute_ii_loss(
                outputs['iim_features'], self.model.iim, rgb,
                gamma_range=self.gamma_range, beta=1.0)

            # ---------- Combined loss ----------
            losses = self.criterion(outputs, masks, ii_loss=ii_loss)
            loss   = losses['total']

            # ---------- Backward ----------
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---------- Enforce zero-mean on IIM kernels ----------
            self.model.iim.enforce_zero_mean()

            # ---------- Metrics ----------
            # Use P6 (final prediction) for metric tracking, same as val/test
            filtered = filter_small_predictions(outputs['p6'], min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_eval_train.update(preds, masks, images)

            # ---------- Accumulate component losses ----------
            ep['total']  += losses['total'].item()
            ep['loss1']  += losses['loss1'].item()
            ep['loss2']  += losses['loss2'].item()
            ep['loss3']  += losses['loss3'].item()
            ep['loss4']  += losses['loss4'].item()
            ep['loss5']  += losses['loss5'].item()
            ep['loss6']  += losses['loss6'].item()
            ep['ii_raw'] += losses['ii_raw'].item()
            ep['ii_eff'] += losses['ii'].item()
            ew = losses['ii_eff_weight']
            ep['ii_w']   += (ew.item() if torch.is_tensor(ew) else ew)

            if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                ew_val = (losses['ii_eff_weight'].item()
                          if torch.is_tensor(losses['ii_eff_weight'])
                          else losses['ii_eff_weight'])
                ii_pct = 100 * losses['ii'].item() / (losses['total'].item() + 1e-12)
                print(f'  Batch [{bi+1:4d}/{num_batches}] | '
                      f'Total: {losses["total"].item():.4f} | '
                      f'l1-6: '
                      f'{losses["loss1"].item():.3f}/'
                      f'{losses["loss2"].item():.3f}/'
                      f'{losses["loss3"].item():.3f}/'
                      f'{losses["loss4"].item():.3f}/'
                      f'{losses["loss5"].item():.3f}/'
                      f'{losses["loss6"].item():.3f} | '
                      f'II(raw): {losses["ii_raw"].item():.2e} '
                      f'w={ew_val:.1f} [{ii_pct:.1f}%]')

        # Per-epoch averages
        for k in ep:
            ep[k] /= num_batches

        metrics = train_metrics.compute()
        elapsed = time.time() - start

        ii_pct = 100 * ep['ii_eff'] / (ep['total'] + 1e-12)
        print(f'\nTraining Results:')
        print(f'  Time: {elapsed:.1f}s | Total: {ep["total"]:.4f}')
        print(f'  l1={ep["loss1"]:.4f} l2={ep["loss2"]:.4f} '
              f'l3={ep["loss3"]:.4f} l4={ep["loss4"]:.4f} '
              f'l5={ep["loss5"]:.4f} l6={ep["loss6"]:.4f}')
        print(f'  II(raw): {ep["ii_raw"]:.2e}  '
              f'II(eff): {ep["ii_eff"]:.4f} [{ii_pct:.1f}% of total]  '
              f'avg_w: {ep["ii_w"]:.1f}')
        print(f'  OA: {metrics["OA"]:.2f}%  Prec: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'ShadIOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Train/Loss',         ep['total'],  epoch)
        for k in ['loss1','loss2','loss3','loss4','loss5','loss6']:
            self.writer.add_scalar(f'Train/{k}',     ep[k],        epoch)
        self.writer.add_scalar('Train/IILoss_raw',   ep['ii_raw'], epoch)
        self.writer.add_scalar('Train/IILoss_eff',   ep['ii_eff'], epoch)
        self.writer.add_scalar('Train/II_eff_weight',ep['ii_w'],   epoch)
        for k in self.train_metrics_history:
            self.writer.add_scalar(f'Train/{k}',    metrics[k],   epoch)

        # DetailedEvaluator — per-image
        dr = self.detailed_eval_train.compute_metrics()
        self.detailed_eval_train.reset()
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_pi',   s['iou'], epoch)
        self.writer.add_scalar('Train/F1_strict_pi',     s['f1'],  epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_pi', t['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_pi',   t['f1'],  epoch)
        print(f'  Per-image Strict:   F1={s["f1"]:.2f}%  mIOU={s["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t["f1"]:.2f}%  mIOU={t["iou"]:.2f}%')

        # Store histories
        self.train_total.append(ep['total'])
        self.train_loss1.append(ep['loss1'])
        self.train_loss2.append(ep['loss2'])
        self.train_loss3.append(ep['loss3'])
        self.train_loss4.append(ep['loss4'])
        self.train_loss5.append(ep['loss5'])
        self.train_loss6.append(ep['loss6'])
        self.train_ii_raw.append(ep['ii_raw'])
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(metrics[k])

        return ep['total'], metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        print('\nValidating...')
        self.model.eval()

        val_main_sum = 0.0
        val_ii_sum   = 0.0
        val_metrics  = ShadowMetrics()
        n_batches    = len(self.dataloaders['val'])

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Eval mode → P6 tensor only
                p6 = self.model(images)

                # Per-image CE on P6
                val_main_sum += self.val_criterion(p6, masks).item()

                # II loss (for tracking; no gradient needed)
                rgb        = images[:, :3]
                feats_orig = self.model.iim.extract_features(rgb)
                ii_loss    = compute_ii_loss(
                    feats_orig, self.model.iim, rgb,
                    gamma_range=self.gamma_range, beta=1.0)
                val_ii_sum += ii_loss.item()

                filtered = filter_small_predictions(p6, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_eval_val.update(preds, masks, images)

        val_main_avg = val_main_sum / n_batches
        val_ii_avg   = val_ii_sum   / n_batches

        # Compute adaptive weight for val (mirrors training logic)
        if self.args.ii_loss_mode == 'adaptive' and val_ii_avg > 0:
            ii_w = self.args.ii_target_ratio * val_main_avg / (val_ii_avg + 1e-8)
        else:
            ii_w = self.args.ii_loss_weight
        val_ii_eff   = ii_w * val_ii_avg
        val_total    = val_main_avg + val_ii_eff

        metrics = val_metrics.compute()

        ii_pct = 100 * val_ii_eff / (val_total + 1e-12)
        print(f'Validation Results:')
        print(f'  Total: {val_total:.4f} | P6-CE: {val_main_avg:.4f} | '
              f'II(raw): {val_ii_avg:.2e}  '
              f'II(eff): {val_ii_eff:.4f} [{ii_pct:.1f}%]  w={ii_w:.1f}')
        print(f'  OA: {metrics["OA"]:.2f}%  Prec: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'ShadIOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Val/Loss',       val_total,    epoch)
        self.writer.add_scalar('Val/P6_CE',      val_main_avg, epoch)
        self.writer.add_scalar('Val/IILoss_raw', val_ii_avg,   epoch)
        for k in self.val_metrics_history:
            self.writer.add_scalar(f'Val/{k}', metrics[k], epoch)

        # DetailedEvaluator
        dr = self.detailed_eval_val.compute_metrics()
        self.detailed_eval_val.reset()
        s = dr['boundary_tolerant']['strict']
        t = dr['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Val/mIOU_strict_pi',   s['iou'], epoch)
        self.writer.add_scalar('Val/F1_strict_pi',     s['f1'],  epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_pi', t['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_pi',   t['f1'],  epoch)
        print(f'  Per-image Strict:   F1={s["f1"]:.2f}%  mIOU={s["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t["f1"]:.2f}%  mIOU={t["iou"]:.2f}%')

        # Store histories
        self.val_total.append(val_total)
        self.val_p6_ce.append(val_main_avg)
        self.val_ii_raw.append(val_ii_avg)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(metrics[k])

        return val_total, metrics, dr

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch':                        epoch,
            'model_state_dict':             self.model.state_dict(),
            'optimizer_state_dict':         self.optimizer.state_dict(),
            'scheduler_state_dict':         self.scheduler.state_dict(),
            'best_miou':                    self.best_miou,
            'best_shadow_iou':              self.best_shadow_iou,
            'best_f1':                      self.best_f1,
            'best_decision_miou':           self.best_decision_miou,
            'epochs_without_improvement':   self.epochs_without_improvement,
            # Loss histories
            'train_total':  self.train_total,
            'train_loss1':  self.train_loss1,
            'train_loss2':  self.train_loss2,
            'train_loss3':  self.train_loss3,
            'train_loss4':  self.train_loss4,
            'train_loss5':  self.train_loss5,
            'train_loss6':  self.train_loss6,
            'train_ii_raw': self.train_ii_raw,
            'val_total':    self.val_total,
            'val_p6_ce':    self.val_p6_ce,
            'val_ii_raw':   self.val_ii_raw,
            # Metric histories
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history':   self.val_metrics_history,
            'args': vars(self.args),
        }

        # Always overwrite latest
        torch.save(ckpt, os.path.join(self.output_dir, 'checkpoint_latest.pth'))

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(ckpt, best_path)
            print(f'Best checkpoint saved → {best_path}')

        if epoch % self.args.save_freq == 0:
            ep_path = os.path.join(self.output_dir,
                                   f'checkpoint_epoch_{epoch}.pth')
            torch.save(ckpt, ep_path)

    def load_checkpoint(self, path):
        print(f'Loading checkpoint from {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        try:
            self.model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print('WARNING: size mismatch — attempting partial load…')
                sd    = ckpt['model_state_dict']
                md    = self.model.state_dict()
                compat = {k: v for k, v in sd.items()
                          if k in md and v.size() == md[k].size()}
                md.update(compat)
                self.model.load_state_dict(md)
                print(f'Loaded {len(compat)}/{len(sd)} layers')
            else:
                raise

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch               = ckpt['epoch'] + 1
        self.best_miou                 = ckpt.get('best_miou',        0.0)
        self.best_shadow_iou           = ckpt.get('best_shadow_iou',  0.0)
        self.best_f1                   = ckpt.get('best_f1',          0.0)
        self.best_decision_miou        = ckpt.get('best_decision_miou', 0.0)
        self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)

        self.train_total  = ckpt.get('train_total',  [])
        self.train_loss1  = ckpt.get('train_loss1',  [])
        self.train_loss2  = ckpt.get('train_loss2',  [])
        self.train_loss3  = ckpt.get('train_loss3',  [])
        self.train_loss4  = ckpt.get('train_loss4',  [])
        self.train_loss5  = ckpt.get('train_loss5',  [])
        self.train_loss6  = ckpt.get('train_loss6',  [])
        self.train_ii_raw = ckpt.get('train_ii_raw', [])
        self.val_total    = ckpt.get('val_total',    [])
        self.val_p6_ce    = ckpt.get('val_p6_ce',    [])
        self.val_ii_raw   = ckpt.get('val_ii_raw',   [])

        _mk = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = ckpt.get('train_metrics_history',
                                              {k: [] for k in _mk})
        self.val_metrics_history   = ckpt.get('val_metrics_history',
                                              {k: [] for k in _mk})
        print(f'Resumed from epoch {ckpt["epoch"]}  '
              f'best_decision_miou={self.best_decision_miou:.2f}%')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print('\n' + '=' * 55)
        print('Starting OGLANet-IIM training…')
        print(f'IIM: {self.args.num_kernels} kernels × '
              f'{self.args.kernel_size}×{self.args.kernel_size}')
        if self.args.ii_loss_mode == 'adaptive':
            print(f'II loss: ADAPTIVE  target_ratio={self.args.ii_target_ratio} '
                  f'(~{100*self.args.ii_target_ratio:.0f}% of task loss)')
        else:
            print(f'II loss: FIXED  weight={self.args.ii_loss_weight}')
        print(f'Decision metric: {self._metric_label()}')
        print('=' * 55)

        patience = self.args.early_stopping_patience or 0

        for epoch in range(self.start_epoch, self.args.epochs):
            ep = epoch + 1
            self.train_epoch(ep)
            _, val_metrics, dr = self.validate(ep)

            decision_miou = self._get_decision_miou(dr)
            self.scheduler.step(decision_miou)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, ep)
            self.writer.add_scalar('Train/LearningRate',
                                   self.optimizer.param_groups[0]['lr'], ep)

            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou       = decision_miou
                is_best                       = True
                self.epochs_without_improvement = 0
                print(f'>> New best {self._metric_label()}: '
                      f'{self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            # Track pooled bests for reference only
            if val_metrics['mIOU']       > self.best_miou:
                self.best_miou       = val_metrics['mIOU']
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
            if val_metrics['F1']         > self.best_f1:
                self.best_f1         = val_metrics['F1']

            self.save_checkpoint(ep, is_best=is_best)

            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered after '
                      f'{self.epochs_without_improvement} epochs without '
                      f'improvement in {self._metric_label()}.')
                break

            print('=' * 55)

        # ---- Plots ----
        print('\nGenerating plots…')
        plot_loss_curves_iim(
            train_total=self.train_total,
            val_total=self.val_total,
            train_loss1=self.train_loss1,
            train_loss2=self.train_loss2,
            train_loss3=self.train_loss3,
            train_loss4=self.train_loss4,
            train_loss5=self.train_loss5,
            train_loss6=self.train_loss6,
            train_ii_raw=self.train_ii_raw,
            val_p6_ce=self.val_p6_ce,
            val_ii_raw=self.val_ii_raw,
            save_path=os.path.join(self.output_dir, 'loss_curves.png'),
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png'),
        )

        print('\nTraining completed!')
        print(f'Best {self._metric_label()}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (ref):       {self.best_miou:.2f}%')
        print(f'Best Shadow IoU (ref):         {self.best_shadow_iou:.2f}%')
        print(f'Best F1 (ref):                 {self.best_f1:.2f}%')
        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        print('\n' + '=' * 55)
        print('Testing OGLANet-IIM…')
        print('=' * 55)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self.load_checkpoint(best_ckpt)
        else:
            print('WARNING: best checkpoint not found — using current weights')

        self.model.eval()
        test_metrics  = ShadowMetrics()
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                p6       = self.model(images)
                filtered = filter_small_predictions(p6, min_pixels=10)
                test_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics = test_metrics.compute()
        dr      = detailed_eval.compute_metrics()

        print('\n' + '=' * 55)
        print('Pooled Test Results (reference):')
        print('=' * 55)
        for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']:
            print(f'  {k:12s}: {metrics[k]:.2f}%')

        print('\n' + '=' * 55)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('=' * 55)
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
            print(f'\nSize-Stratified '
                  f'(Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in dr['size_stratified_tolerant']:
                    m = dr['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'fp_fn_analysis' in dr and 'fp' in dr['fp_fn_analysis']:
            fp = dr['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within  1px: {fp['pct_within_1px']:.1f}%")
            print(f"  Within  5px: {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        results = {'standard': metrics, 'detailed': dr}
        rp = os.path.join(self.output_dir, 'test_results.json')
        with open(rp, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\nResults saved → {rp}')

        try:
            print('\nGenerating best/worst prediction visualizations…')
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
    args    = get_args()
    trainer = Trainer(args)
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()