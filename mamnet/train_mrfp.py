"""
Training script for MAMNet + MRFP / MRFP+
==========================================
Adds Multi-Resolution Feature Perturbation (CVPR 2024) to MAMNet for
cross-city shadow-detection generalization.

Key differences from base train.py:
    • Model : MAMNetMRFP (HRFP + NP+ + optional HRFP+)
    • Loss  : Per-image mean CE (not global pixel mean)
    • Eval  : Tolerant per-image mIOU drives all decisions (unchanged)
    • Viz   : Overview subplot + one subplot per loss component

Perturbation modules are training-only; inference is identical to base MAMNet.
"""

import os
import argparse
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_mrfp import MAMNetMRFP
from data.dataset import get_dataloaders
from data.dataset_enhanced import ShadowDatasetEnhanced
from utils.evaluation_detailed import DetailedEvaluator
from utils.losses_mrfp import MAMNetMRFPLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions

# ---- Visualization ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    'font.family':       'serif',
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

# Colour palette
_C = {
    'train_total': '#2E86AB', 'val_total':  '#A23B72',
    'main_train':  '#F18F01', 'main_val':   '#C73E1D',
    'aux_train':   '#6A994E',
}
_METRIC_COLORS = {
    'OA': '#2E86AB', 'Precision': '#F18F01', 'F1': '#A23B72',
    'BER': '#E63946', 'mIOU': '#6A994E', 'Shadow_IOU': '#8338EC',
}

# ======================================================================
# GPU diagnostics
# ======================================================================
print("=" * 50)
print("GPU DIAGNOSTICS")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name:    {torch.cuda.get_device_name(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("=" * 50)


# ======================================================================
# Args
# ======================================================================
def get_args():
    p = argparse.ArgumentParser(description='Train MAMNet + MRFP')

    # Data
    p.add_argument('--data_root',   type=str, default=None)
    p.add_argument('--img_size',    type=int, default=384)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # Multi-city / LOCO
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution',     type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id',  type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities',   type=str, nargs='+', default=None)

    # Model
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--pretrained',  action='store_true', default=True)
    p.add_argument('--aux_weight',  type=float, default=0.4)

    # MRFP
    p.add_argument('--mrfp_plus',       action='store_true', default=True,
                   help='Use MRFP+ (HRFP+HRFP++NP+). Default True (best variant).')
    p.add_argument('--no_mrfp_plus',    action='store_true', default=False,
                   help='Disable HRFP+; use plain MRFP (HRFP+NP+).')
    p.add_argument('--hrfp_prob',       type=float, default=0.5)
    p.add_argument('--np_prob',         type=float, default=0.5)
    p.add_argument('--hrfp_plus_prob',  type=float, default=0.5)
    p.add_argument('--hrfp_bn_std',     type=float, default=0.5)

    # Training
    p.add_argument('--epochs',       type=int,   default=15)
    p.add_argument('--lr',           type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # Checkpoint / logging
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq',  type=int, default=10)
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--eval_only',  action='store_true')

    # Device
    p.add_argument('--device', type=str, default='cuda')

    # Contrast channel
    p.add_argument('--use_contrast', action='store_true')

    # FDA (if needed)
    p.add_argument('--use_fda',         action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L',           type=float, default=0.01)

    # Boundary-tolerant eval
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--boundary_tolerance',     type=int, default=2)

    # Early stopping
    p.add_argument('--early_stopping_patience', type=int, default=0)

    # Comparison (optional)
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root',     type=str, default=None)

    args = p.parse_args()

    # Resolve MRFP+ flag
    if args.no_mrfp_plus:
        args.mrfp_plus = False

    return args


# ======================================================================
# Visualization helpers
# ======================================================================

def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


def plot_loss_curves_mrfp(train_losses, val_losses, save_path,
                          train_main_losses=None, train_aux_losses=None):
    """
    Row 0 — Overview: all available losses on a shared y-axis.
    Row 1+ — Individual panels: one loss per subplot, own y-axis,
             NO total loss (so fine-grained trends are visible).
    """
    epochs = list(range(1, len(train_losses) + 1))

    # Build per-component panels
    panels = []
    if train_main_losses and len(train_main_losses) == len(epochs):
        panels.append({
            'title': 'Main (CE) Loss',
            'series': [
                (train_main_losses, 'Train Main CE', _C['main_train'], '-',  'o'),
                (val_losses,        'Val (Main CE)', _C['main_val'],   '--', 's'),
            ],
        })
    if (train_aux_losses
            and len(train_aux_losses) == len(epochs)
            and any(v > 1e-9 for v in train_aux_losses)):
        panels.append({
            'title': 'Auxiliary Loss (weighted)',
            'series': [
                (train_aux_losses, 'Train Aux', _C['aux_train'], '-', '^'),
            ],
        })

    n_panels   = len(panels)
    n_rows_ind = max(1, (n_panels + 3) // 4) if n_panels > 0 else 0
    fig_w = max(9, min(5.5 * max(n_panels, 1), 22))
    fig_h = 4.2 + 3.8 * n_rows_ind

    MAX_COLS = 4

    if n_panels > 0:
        fig   = plt.figure(figsize=(fig_w, fig_h))
        outer = gridspec.GridSpec(
            2, 1, figure=fig, hspace=0.55,
            height_ratios=[3.8, 3.8 * n_rows_ind])
        ax_ov = fig.add_subplot(outer[0])
    else:
        fig, ax_ov = plt.subplots(figsize=(10, 4.2))

    # ---- Overview ----
    ax_ov.plot(epochs, train_losses, '-', lw=2,
               color=_C['train_total'], label='Train total',
               marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses, '--', lw=1.8,
               color=_C['val_total'], label='Val total',
               marker='s', ms=3, mfc='white', mew=1.2)
    if train_main_losses and len(train_main_losses) == len(epochs):
        ax_ov.plot(epochs, train_main_losses, '-', lw=1.5,
                   color=_C['main_train'], alpha=0.75, label='Train main CE')
    if (train_aux_losses
            and len(train_aux_losses) == len(epochs)
            and any(v > 1e-9 for v in train_aux_losses)):
        ax_ov.plot(epochs, train_aux_losses, '-', lw=1.5,
                   color=_C['aux_train'], alpha=0.75, label='Train aux')
    ax_ov.set_title('Overview — All Losses (shared y-axis)', fontweight='bold')
    ax_ov.set_xlabel('Epoch'); ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=8, framealpha=0.88, ncol=4)
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ---- Individual panels (no total!) ----
    if n_panels > 0:
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
            _style_ax(ax, panel['title'], 'Loss')
        for pidx in range(n_panels, n_rows_ind * MAX_COLS):
            r, c = divmod(pidx, MAX_COLS)
            fig.add_subplot(inner[r, c]).set_visible(False)

    fig.suptitle('MAMNet + MRFP — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


def plot_metrics_curves_mrfp(train_hist, val_hist, save_path):
    """One subplot per metric, train + val across epochs."""
    metrics = list(train_hist.keys())
    n = len(metrics)
    if n == 0:
        return
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5.5 * n_cols, 4.0 * n_rows),
                              squeeze=False)
    for idx, metric in enumerate(metrics):
        r, c  = divmod(idx, n_cols)
        ax    = axes[r][c]
        color = _METRIC_COLORS.get(metric, '#333333')
        ep_tr = list(range(1, len(train_hist[metric]) + 1))
        ep_vl = list(range(1, len(val_hist[metric]) + 1))
        ax.plot(ep_tr, train_hist[metric], '-', lw=1.8, color=color,
                label='Train', marker='o', ms=3.5, mfc='white', mew=1.2)
        ax.plot(ep_vl, val_hist[metric], '--', lw=1.8, color=color,
                label='Val', marker='s', ms=3.5, mfc='white', mew=1.2,
                alpha=0.75)
        _style_ax(ax, metric, '%')
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)
    fig.suptitle('MAMNet + MRFP — Metric Curves (pooled, reference)',
                 fontweight='bold', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Metric curves saved → {save_path}')


# Import best/worst visualizations from base utils
from utils.visualization import save_best_worst_visualizations


# ======================================================================
# Trainer
# ======================================================================

class Trainer:

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Tolerant key for DetailedEvaluator
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # ---- Output directory ----
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        variant   = 'mrfp_plus' if args.mrfp_plus else 'mrfp'

        if args.mode == 'single':
            city = args.data_root.rstrip('/').split("/")[-2]
            res  = args.data_root.rstrip('/').split("/")[-1]
            exp_name = f'mamnet_{variant}_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'mamnet_{variant}_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'mamnet_{variant}_loco_holdout_{test_city}_{args.resolution}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ---- Model ----
        print('Initializing MAMNetMRFP …')
        self.model = MAMNetMRFP(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            use_aux=True,
            use_contrast=args.use_contrast,
            use_mrfp_plus=args.mrfp_plus,
            hrfp_prob=args.hrfp_prob,
            np_prob=args.np_prob,
            hrfp_plus_prob=args.hrfp_plus_prob,
            bn_std=args.hrfp_bn_std,
        ).to(self.device)

        total_p = sum(p.numel() for p in self.model.parameters())
        train_p = sum(p.numel() for p in self.model.parameters()
                      if p.requires_grad)
        print(f'Total parameters:     {total_p:,}')
        print(f'Trainable parameters: {train_p:,}')
        print(f'Frozen (HRFP):        {total_p - train_p:,}')

        # ---- Loss ----
        self.criterion = MAMNetMRFPLoss(aux_weight=args.aux_weight)

        # ---- Optimizer & scheduler ----
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3)

        # ---- Decision metric ----
        self.use_tolerant_decision = args.eval_boundary_tolerant
        label = (f'TOLERANT mIOU (±{args.boundary_tolerance}px)'
                 if self.use_tolerant_decision
                 else 'STRICT per-image mIOU')
        print(f'>> Decision metric: {label}')

        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ---- Tracking ----
        self.start_epoch              = 0
        self.best_miou                = 0.0
        self.best_shadow_iou          = 0.0
        self.best_f1                  = 0.0
        self.best_decision_miou       = 0.0
        self.epochs_without_improvement = 0

        # Loss histories
        self.train_losses      = []
        self.train_main_losses = []
        self.train_aux_losses  = []
        self.val_losses        = []

        # Metric histories (ShadowMetrics pooled — reference only)
        _keys = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = {k: [] for k in _keys}
        self.val_metrics_history   = {k: [] for k in _keys}

        if args.resume:
            self.load_checkpoint(args.resume)

        # ---- Data loaders ----
        self._build_dataloaders(args)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _build_dataloaders(self, args):
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
                if None in (args.base_data_root, args.resolution, args.fold_id):
                    raise ValueError("base_data_root, resolution, fold_id required")
                from data.dataset import LOCO_FOLDS
                fold = LOCO_FOLDS[args.fold_id]
                train_paths = [os.path.join(args.base_data_root, c, args.resolution)
                               for c in fold['train']]
                val_paths   = train_paths
                test_paths  = [os.path.join(args.base_data_root,
                                            fold['test'], args.resolution)]
            else:
                raise ValueError(f"Invalid mode: {args.mode}")

            from torch.utils.data import DataLoader
            train_ds = ShadowDatasetEnhanced(
                root_dir=train_paths, split='train', img_size=args.img_size,
                task_id=2, augment=True,
                use_fda=args.use_fda, fda_target_root=args.fda_target_root,
                fda_L=args.fda_L)
            val_ds = ShadowDatasetEnhanced(
                root_dir=val_paths, split='val', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)
            test_ds = ShadowDatasetEnhanced(
                root_dir=test_paths, split='test', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)

            self.dataloaders = {
                'train': DataLoader(train_ds, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers,
                                    pin_memory=True, drop_last=True),
                'val':   DataLoader(val_ds,   batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True),
                'test':  DataLoader(test_ds,  batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True),
            }
            print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  '
                  f'Test: {len(test_ds)}')
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
                fda_L=getattr(args, 'fda_L', 0.01))
            print(f'Train: {len(self.dataloaders["train"].dataset)}  '
                  f'Val: {len(self.dataloaders["val"].dataset)}  '
                  f'Test: {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    # Decision metric helper
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_decision:
            return bt[self.tol_key]['iou']
        else:
            return bt['strict']['iou']

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()

        epoch_loss      = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss  = 0.0

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        t0 = time.time()

        for bi, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            outputs = self.model(images)
            losses  = self.criterion(outputs, masks)
            loss    = losses['total']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics on filtered predictions
            filtered = filter_small_predictions(outputs['main'], min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss      += loss.item()
            epoch_main_loss += losses['main'].item()
            epoch_aux_loss  += losses.get('aux', torch.tensor(0.0)).item()

            if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                print(f'  Batch [{bi+1}/{num_batches}] | '
                      f'Loss {loss.item():.4f} | '
                      f'Main {losses["main"].item():.4f} | '
                      f'Aux {losses.get("aux", torch.tensor(0.0)).item():.4f}')

        epoch_loss      /= num_batches
        epoch_main_loss /= num_batches
        epoch_aux_loss  /= num_batches

        metrics = train_metrics.compute()
        dt = time.time() - t0

        print(f'\nTrain Results ({dt:.1f}s):')
        print(f'  Total={epoch_loss:.4f}  Main={epoch_main_loss:.4f}  '
              f'Aux={epoch_aux_loss:.4f}')
        print(f'  OA={metrics["OA"]:.2f}%  P={metrics["Precision"]:.2f}%  '
              f'F1={metrics["F1"]:.2f}%  BER={metrics["BER"]:.2f}%  '
              f'mIOU(pooled)={metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU={metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Train/Loss',     epoch_loss,      epoch)
        self.writer.add_scalar('Train/MainLoss', epoch_main_loss, epoch)
        self.writer.add_scalar('Train/AuxLoss',  epoch_aux_loss,  epoch)
        for k in self.train_metrics_history:
            self.writer.add_scalar(f'Train/{k}', metrics[k], epoch)

        # Histories
        self.train_losses.append(epoch_loss)
        self.train_main_losses.append(epoch_main_loss)
        self.train_aux_losses.append(epoch_aux_loss)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(metrics[k])

        # DetailedEvaluator per-image metrics
        det = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict   = det['boundary_tolerant']['strict']
        tolerant = det['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_perimage',   strict['iou'],   epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     strict['f1'],    epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', tolerant['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   tolerant['f1'],  epoch)
        print(f'  Per-image Strict:   F1={strict["f1"]:.2f}%  '
              f'mIOU={strict["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        return epoch_loss, epoch_main_loss, epoch_aux_loss, metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        print('\nValidating …')
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs = self.model(images)  # tensor in eval mode

                loss      = self.criterion({'main': outputs}, masks)['main']
                val_loss += loss.item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics   = val_metrics.compute()

        print(f'Val Results:')
        print(f'  Loss={val_loss:.4f}')
        print(f'  OA={metrics["OA"]:.2f}%  P={metrics["Precision"]:.2f}%  '
              f'F1={metrics["F1"]:.2f}%  BER={metrics["BER"]:.2f}%  '
              f'mIOU(pooled)={metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU={metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for k in self.val_metrics_history:
            self.writer.add_scalar(f'Val/{k}', metrics[k], epoch)

        self.val_losses.append(val_loss)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(metrics[k])

        # DetailedEvaluator
        det = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict   = det['boundary_tolerant']['strict']
        tolerant = det['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Val/mIOU_strict_perimage',   strict['iou'],   epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     strict['f1'],    epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', tolerant['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   tolerant['f1'],  epoch)
        print(f'  Per-image Strict:   F1={strict["f1"]:.2f}%  '
              f'mIOU={strict["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        return val_loss, metrics, det

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch':                      epoch,
            'model_state_dict':           self.model.state_dict(),
            'optimizer_state_dict':       self.optimizer.state_dict(),
            'scheduler_state_dict':       self.scheduler.state_dict(),
            'best_miou':                  self.best_miou,
            'best_shadow_iou':            self.best_shadow_iou,
            'best_f1':                    self.best_f1,
            'best_decision_miou':         self.best_decision_miou,
            'epochs_without_improvement': self.epochs_without_improvement,
            'train_losses':               self.train_losses,
            'train_main_losses':          self.train_main_losses,
            'train_aux_losses':           self.train_aux_losses,
            'val_losses':                 self.val_losses,
            'train_metrics_history':      self.train_metrics_history,
            'val_metrics_history':        self.val_metrics_history,
            'args':                       vars(self.args),
        }
        if is_best:
            path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(ckpt, path)
            print(f'Best checkpoint → {path}')
        if epoch % self.args.save_freq == 0:
            path = os.path.join(self.output_dir,
                                f'checkpoint_epoch_{epoch}.pth')
            torch.save(ckpt, path)

    def load_checkpoint(self, path):
        print(f'Loading checkpoint: {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        try:
            self.model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print("WARNING: size mismatch — attempting partial load")
                sd_ckpt  = ckpt['model_state_dict']
                sd_model = self.model.state_dict()
                matched  = {k: v for k, v in sd_ckpt.items()
                            if k in sd_model and v.size() == sd_model[k].size()}
                sd_model.update(matched)
                self.model.load_state_dict(sd_model)
                print(f"Loaded {len(matched)}/{len(sd_ckpt)} layers")
            else:
                raise

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch              = ckpt['epoch'] + 1
        self.best_miou                = ckpt.get('best_miou', 0.0)
        self.best_shadow_iou          = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1                  = ckpt.get('best_f1', 0.0)
        self.best_decision_miou       = ckpt.get('best_decision_miou', 0.0)
        self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        self.train_losses      = ckpt.get('train_losses', [])
        self.train_main_losses = ckpt.get('train_main_losses', [])
        self.train_aux_losses  = ckpt.get('train_aux_losses', [])
        self.val_losses        = ckpt.get('val_losses', [])
        _keys = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = ckpt.get(
            'train_metrics_history', {k: [] for k in _keys})
        self.val_metrics_history   = ckpt.get(
            'val_metrics_history', {k: [] for k in _keys})
        print(f'Resumed from epoch {ckpt["epoch"]}  '
              f'best_decision_mIOU={self.best_decision_miou:.2f}%')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print('\n' + '=' * 50)
        print('Starting MRFP training …')
        print('=' * 50)

        patience = self.args.early_stopping_patience
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_decision
                        else 'Strict per-image mIOU')
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            ep = epoch + 1

            train_loss, train_main, train_aux, train_met = self.train_epoch(ep)
            val_loss, val_met, det = self.validate(ep)

            # Decision metric
            decision_miou = self._get_decision_miou(det)
            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, ep)

            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou       = decision_miou
                is_best                       = True
                self.epochs_without_improvement = 0
                print(f'>> New best {metric_label}: '
                      f'{self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            if val_met['mIOU'] > self.best_miou:
                self.best_miou = val_met['mIOU']
            if val_met['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_met['Shadow_IOU']
            if val_met['F1'] > self.best_f1:
                self.best_f1 = val_met['F1']

            self.save_checkpoint(ep, is_best=is_best)

            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', lr, ep)

            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping after {patience} epochs '
                      f'w/o improvement in {metric_label}.')
                break
            print('=' * 50)

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU:   {self.best_miou:.2f}%')
        print(f'Best Shadow IoU:    {self.best_shadow_iou:.2f}%')
        print(f'Best F1:            {self.best_f1:.2f}%')

        # Plots
        print('\nGenerating plots …')
        plot_loss_curves_mrfp(
            self.train_losses, self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            train_main_losses=self.train_main_losses,
            train_aux_losses=self.train_aux_losses)
        plot_metrics_curves_mrfp(
            self.train_metrics_history, self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png'))

        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        print('\n' + '=' * 50)
        print('Testing model …')
        print('=' * 50)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self.load_checkpoint(best_ckpt)
        else:
            print('WARNING: best checkpoint not found, using current weights')

        self.model.eval()

        test_metrics  = ShadowMetrics()
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs  = self.model(images)
                filtered = filter_small_predictions(outputs, min_pixels=10)
                test_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics = test_metrics.compute()
        det     = detailed_eval.compute_metrics()

        print('\nPooled Test Results (reference):')
        for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']:
            print(f'  {k:12s}: {metrics[k]:.2f}%')

        print('\nPer-Image Test Results (DetailedEvaluator):')
        strict   = det['boundary_tolerant']['strict']
        tolerant = det['boundary_tolerant'][self.tol_key]
        print(f'  Strict   — F1: {strict["f1"]:.2f}%  '
              f'mIOU: {strict["iou"]:.2f}%')
        print(f'  Tolerant (±{self.args.boundary_tolerance}px) — '
              f'F1: {tolerant["f1"]:.2f}%  mIOU: {tolerant["iou"]:.2f}%')
        print(f'  Pixels excluded: {tolerant["pixels_excluded"]} '
              f'({tolerant["pct_excluded"]:.1f}%)')

        # Size-stratified
        for label_key, result_key in [
                ('Strict', 'size_stratified'),
                (f'Tolerant ±{self.args.boundary_tolerance}px',
                 'size_stratified_tolerant')]:
            if result_key in det:
                print(f'\n  Size-Stratified ({label_key}):')
                for cat in ['tiny', 'small', 'medium', 'large']:
                    if cat in det[result_key]:
                        m = det[result_key][cat]
                        print(f'    {cat:8s}: Miss={m["miss_rate"]:5.1f}%  '
                              f'IoU={m["avg_iou"]:5.1f}%  ({m["total"]} shadows)')

        # FP analysis
        if 'fp_fn_analysis' in det and 'fp' in det['fp_fn_analysis']:
            fp = det['fp_fn_analysis']['fp']
            print('\n  FP Spatial Distribution:')
            print(f'    Within 1px:  {fp["pct_within_1px"]:.1f}%')
            print(f'    Within 5px:  {fp["pct_within_5px"]:.1f}%')
            print(f'    Within 10px: {fp["pct_within_10px"]:.1f}%')

        # Save results
        results = {'standard': metrics, 'detailed': det}
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\nResults → {results_path}')

        try:
            print('\nGenerating best/worst visualizations …')
            save_best_worst_visualizations(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except Exception as e:
            print(f'Visualization skipped: {e}')

        return metrics


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