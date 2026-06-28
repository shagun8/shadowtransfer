"""
Training script for DINOv3 + MRFP/MRFP+ Shadow Detection
==========================================================
Adds Multi-Resolution Feature Perturbation (CVPR 2024) to DINOv3 for
cross-city shadow-detection generalisation.

Key differences from train_dinov3.py:
    • Model   : DINOv3ShadowDetectorMRFP  (HRFP + NP+ + optional HRFP+)
    • Loss    : PerImageCrossEntropyLoss  (per-image mean, not pooled pixel mean)
    • Viz     : Overview subplot (all losses, shared y-axis) +
                individual subplot per loss component (own y-axis, NO total loss)
    • Eval    : Tolerant per-image mIOU drives all decisions (unchanged)

MRFP perturbation modules are training-only; inference is identical to base DINOv3.

Decision metrics — same convention as train_dinov3.py:
    eval_boundary_tolerant=True  → per-image TOLERANT mIOU (drives decisions)
    eval_boundary_tolerant=False → per-image STRICT  mIOU (drives decisions)
    DetailedEvaluator always runs; flag only selects the decision metric.
    ShadowMetrics (pooled) is logged to TensorBoard for reference only.
"""

import os
import argparse
import time
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_model_mrfp import DINOv3ShadowDetectorMRFP
from data.dataset import get_dataloaders
from utils.losses_mrfp import PerImageCrossEntropyLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.evaluation_detailed import DetailedEvaluator
from utils.visualization import save_best_worst_visualizations

# ---- Matplotlib (headless) ----
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

_C_TRAIN  = '#2E86AB'
_C_VAL    = '#A23B72'
_METRIC_COLORS = {
    'OA':         '#2E86AB',
    'Precision':  '#F18F01',
    'F1':         '#A23B72',
    'BER':        '#E63946',
    'mIOU':       '#6A994E',
    'Shadow_IOU': '#8338EC',
}

# ======================================================================
# GPU diagnostics
# ======================================================================
print('=' * 50)
print('GPU DIAGNOSTICS')
print('=' * 50)
print(f'CUDA available:       {torch.cuda.is_available()}')
print(f'CUDA device count:    {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current CUDA device:  {torch.cuda.current_device()}')
    print(f'CUDA device name:     {torch.cuda.get_device_name(0)}')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")}')
print('=' * 50)


# ======================================================================
# Arguments
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Train DINOv3 + MRFP/MRFP+ for Shadow Detection')

    # ---- Data ----
    p.add_argument('--data_root',   type=str, default=None)
    p.add_argument('--img_size',    type=int, default=384)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # ---- Multi-city / LOCO ----
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution',     type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id',  type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities',   type=str, nargs='+', default=None)

    # ---- Model ----
    p.add_argument('--num_classes',  type=int,  default=2)
    p.add_argument('--model_name',   type=str,  default='dinov3_vits16',
                   choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--weights_path', type=str,  default=None)
    p.add_argument('--pretrained',   action='store_true', default=True)
    p.add_argument('--frozen_stages', type=int, default=-1)

    # ---- Training (DINOv3 cosine-warmup schedule) ----
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--lr',           type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=int,  default=5)
    p.add_argument('--min_lr',       type=float, default=1e-6)

    # ---- MRFP ----
    p.add_argument('--mrfp_plus', action='store_true', default=True,
                   help='Use MRFP+ (HRFP+HRFP++NP+). Default: True (best variant).')
    p.add_argument('--no_mrfp_plus', action='store_true', default=False,
                   help='Disable HRFP+; use plain MRFP (HRFP+NP+).')
    p.add_argument('--hrfp_prob',       type=float, default=0.5)
    p.add_argument('--np_prob',         type=float, default=0.5)
    p.add_argument('--hrfp_plus_prob',  type=float, default=0.5)
    p.add_argument('--hrfp_bn_std',     type=float, default=0.5)

    # ---- FDA ----
    p.add_argument('--use_fda',         action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L',           type=float, default=0.01)

    # ---- Checkpoint / logging ----
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq',  type=int, default=5)
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--eval_only',  action='store_true')

    # ---- Device ----
    p.add_argument('--device', type=str, default='cuda')

    # ---- Boundary-tolerant evaluation ----
    p.add_argument('--eval_boundary_tolerant', action='store_true',
                   help='Use tolerant mIOU (instead of strict) for all decisions.')
    p.add_argument('--boundary_tolerance', type=int, default=2,
                   help="Don't-care band half-width in pixels (default 2).")

    # ---- Early stopping ----
    p.add_argument('--early_stopping_patience', type=int, default=15)

    # ---- Comparison (optional, passed from shell) ----
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root',     type=str, default=None)

    args = p.parse_args()

    # Resolve MRFP+ flag
    if args.no_mrfp_plus:
        args.mrfp_plus = False

    return args


# ======================================================================
# LR Scheduler (same cosine-warmup as base DINOv3)
# ======================================================================

class CosineWarmupScheduler:
    """Cosine LR schedule with linear warmup.  Epoch-based; not metric-gated."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer      = optimizer
        self.warmup_epochs  = warmup_epochs
        self.total_epochs   = total_epochs
        self.base_lr        = base_lr
        self.min_lr         = min_lr
        self.current_epoch  = 0

    def step(self, epoch):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = ((epoch - self.warmup_epochs)
                        / (self.total_epochs - self.warmup_epochs))
            lr = (self.min_lr
                  + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress)))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ======================================================================
# Visualisation
# ======================================================================

def _style_ax(ax, title, ylabel):
    ax.set_title(title, fontweight='bold', pad=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax.legend(fontsize=8, framealpha=0.85)


def plot_loss_curves(train_losses, val_losses, save_path):
    """
    DINOv3 MRFP loss visualisation.

    Layout
    ------
    Row 0  — Overview (1 subplot):
        train CE + val CE plotted on a *shared* y-axis.

    Row 1  — Individual component panel (1 subplot):
        CE Loss with train and val on its *own* y-axis.
        Total loss is intentionally NOT included here so the per-component
        scale is clearly visible.

    For DINOv3+MRFP there is only a single CE loss (no aux branches), so
    there is exactly one individual panel.
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig = plt.figure(figsize=(12, 8.0))
    outer = gridspec.GridSpec(
        2, 1, figure=fig, hspace=0.55,
        height_ratios=[3.8, 3.8])

    # ------ Row 0: Overview (shared y-axis) ------
    ax_ov = fig.add_subplot(outer[0])
    ax_ov.plot(epochs, train_losses, '-', lw=2, color=_C_TRAIN,
               label='Train CE', marker='o', ms=3, mfc='white', mew=1.2)
    ax_ov.plot(epochs, val_losses,   '--', lw=1.8, color=_C_VAL,
               label='Val CE',   marker='s', ms=3, mfc='white', mew=1.2)
    ax_ov.set_title('Overview — CE Loss (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=8, framealpha=0.88, ncol=2)
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)

    # ------ Row 1: Individual CE panel (own y-axis) ------
    # One subplot only: CE Loss (train + val), own y-axis.
    # NO total loss — lets the CE scale be uncompressed.
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 1, subplot_spec=outer[1], hspace=0.55, wspace=0.42)
    ax_ce = fig.add_subplot(inner[0, 0])
    ax_ce.plot(epochs, train_losses, '-', lw=1.8, color=_C_TRAIN,
               label='Train CE', marker='o', ms=3.5, mfc='white', mew=1.2, alpha=0.9)
    ax_ce.plot(epochs, val_losses,   '--', lw=1.8, color=_C_VAL,
               label='Val CE',   marker='s', ms=3.5, mfc='white', mew=1.2, alpha=0.9)
    _style_ax(ax_ce, 'CE Loss (per-image mean)', 'Loss')

    variant = 'DINOv3 + MRFP/MRFP+'
    fig.suptitle(f'{variant} — Training Loss Curves',
                 fontweight='bold', fontsize=13, y=1.005)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


def plot_metrics_curves(train_hist, val_hist, save_path):
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
    fig.suptitle('DINOv3 + MRFP/MRFP+ — Metric Curves (pooled ShadowMetrics, reference)',
                 fontweight='bold', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Metric curves saved → {save_path}')


# ======================================================================
# Trainer
# ======================================================================

class Trainer:

    def __init__(self, args):
        self.args = args

        # ---- Device ----
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # ---- Tolerant key ----
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # ---- Output directory ----
        variant = 'mrfp_plus' if args.mrfp_plus else 'mrfp'
        if args.mode == 'single':
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = f'dinov3_{variant}_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'dinov3_{variant}_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'dinov3_{variant}_loco_holdout_{test_city}_{args.resolution}_1'
        else:
            exp_name = f'dinov3_{variant}_custom_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # ---- Model ----
        print('\nInitialising DINOv3 + MRFP model...')
        self.model = DINOv3ShadowDetectorMRFP(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            use_mrfp_plus=args.mrfp_plus,
            hrfp_prob=args.hrfp_prob,
            np_prob=args.np_prob,
            hrfp_plus_prob=args.hrfp_plus_prob,
            bn_std=args.hrfp_bn_std,
        ).to(self.device)

        # ---- Loss — per-image mean CE (no aux branches in DINOv3) ----
        self.criterion = PerImageCrossEntropyLoss()

        # ---- Optimizer — AdamW, skip frozen HRFP params ----
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

        # ---- LR scheduler — epoch-based cosine warmup ----
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
            print(f'>> Decision metric: STRICT per-image mIOU '
                  f'(DetailedEvaluator, not pooled ShadowMetrics)')

        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ---- Tracking ----
        self.start_epoch                = 0
        self.best_decision_miou         = 0.0
        self.best_strict_miou           = 0.0
        self.best_shadow_iou            = 0.0
        self.best_f1                    = 0.0
        self.epochs_without_improvement = 0

        # Loss history (single CE loss — no aux branches)
        self.train_losses = []
        self.val_losses   = []

        # Metric history (ShadowMetrics pooled — reference only)
        _keys = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = {k: [] for k in _keys}
        self.val_metrics_history   = {k: [] for k in _keys}

        if args.resume:
            self.load_checkpoint(args.resume)

        # ---- Datasets ----
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
        Return the mIOU that drives all decisions (best checkpoint, early stopping).
        Both options are per-image means from DetailedEvaluator, never pooled.
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
        Returns
        -------
        epoch_loss : float  — per-image mean CE loss
        metrics    : dict   — ShadowMetrics pooled (reference only)
        """
        self.model.train()
        epoch_loss    = 0.0
        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        t0 = time.time()

        for bi, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            outputs = self.model(images)                 # [B, 2, H, W]
            loss    = self.criterion(outputs, masks)     # per-image mean CE

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            filtered = filter_small_predictions(outputs, min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss += loss.item()

            if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                print(f'  Batch [{bi+1}/{num_batches}] | Loss: {loss.item():.4f}')

        epoch_loss /= num_batches
        metrics     = train_metrics.compute()
        dt          = time.time() - t0

        print(f'\nTrain Results ({dt:.1f}s):')
        print(f'  Loss (per-img mean CE): {epoch_loss:.4f}')
        print(f'  OA={metrics["OA"]:.2f}%  P={metrics["Precision"]:.2f}%  '
              f'F1={metrics["F1"]:.2f}%  BER={metrics["BER"]:.2f}%  '
              f'mIOU(pooled)={metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU={metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Train/Loss',        epoch_loss,            epoch)
        self.writer.add_scalar('Train/OA',          metrics['OA'],         epoch)
        self.writer.add_scalar('Train/Precision',   metrics['Precision'],  epoch)
        self.writer.add_scalar('Train/F1',          metrics['F1'],         epoch)
        self.writer.add_scalar('Train/BER',         metrics['BER'],        epoch)
        self.writer.add_scalar('Train/mIOU_pooled', metrics['mIOU'],       epoch)
        self.writer.add_scalar('Train/Shadow_IOU',  metrics['Shadow_IOU'], epoch)

        self.train_losses.append(epoch_loss)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(metrics[k])

        # DetailedEvaluator — always computed
        det           = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()
        strict_tr     = det['boundary_tolerant']['strict']
        tolerant_tr   = det['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Train/mIOU_strict_perimage',   strict_tr['iou'],   epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     strict_tr['f1'],    epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', tolerant_tr['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   tolerant_tr['f1'],  epoch)

        print(f'  Per-image Strict:   F1={strict_tr["f1"]:.2f}%  '
              f'mIOU={strict_tr["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant_tr["f1"]:.2f}%  mIOU={tolerant_tr["iou"]:.2f}%')

        return epoch_loss, metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        """
        Returns
        -------
        val_loss         : float
        metrics          : dict  — ShadowMetrics pooled (reference only)
        detailed_results : dict  — DetailedEvaluator per-image metrics
        """
        print('\nValidating...')
        self.model.eval()
        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs   = self.model(images)
                val_loss += self.criterion(outputs, masks).item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics   = val_metrics.compute()

        print('Val Results:')
        print(f'  Loss (per-img mean CE): {val_loss:.4f}')
        print(f'  OA={metrics["OA"]:.2f}%  P={metrics["Precision"]:.2f}%  '
              f'F1={metrics["F1"]:.2f}%  BER={metrics["BER"]:.2f}%  '
              f'mIOU(pooled)={metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU={metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Val/Loss',          val_loss,              epoch)
        self.writer.add_scalar('Val/OA',            metrics['OA'],         epoch)
        self.writer.add_scalar('Val/Precision',     metrics['Precision'],  epoch)
        self.writer.add_scalar('Val/F1',            metrics['F1'],         epoch)
        self.writer.add_scalar('Val/BER',           metrics['BER'],        epoch)
        self.writer.add_scalar('Val/mIOU_pooled',   metrics['mIOU'],       epoch)
        self.writer.add_scalar('Val/Shadow_IOU',    metrics['Shadow_IOU'], epoch)

        self.val_losses.append(val_loss)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(metrics[k])

        # DetailedEvaluator — always computed
        det         = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()
        strict_val  = det['boundary_tolerant']['strict']
        tolerant_val = det['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Val/mIOU_strict_perimage',   strict_val['iou'],   epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     strict_val['f1'],    epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', tolerant_val['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   tolerant_val['f1'],  epoch)

        print(f'  Per-image Strict:   F1={strict_val["f1"]:.2f}%  '
              f'mIOU={strict_val["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant_val["f1"]:.2f}%  mIOU={tolerant_val["iou"]:.2f}%')

        return val_loss, metrics, det

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch':                        epoch,
            'model_state_dict':             self.model.state_dict(),
            'optimizer_state_dict':         self.optimizer.state_dict(),
            'best_decision_miou':           self.best_decision_miou,
            'best_strict_miou':             self.best_strict_miou,
            'best_shadow_iou':              self.best_shadow_iou,
            'best_f1':                      self.best_f1,
            'epochs_without_improvement':   self.epochs_without_improvement,
            'use_tolerant_for_decisions':   self.use_tolerant_for_decisions,
            'train_losses':                 self.train_losses,
            'val_losses':                   self.val_losses,
            'train_metrics_history':        self.train_metrics_history,
            'val_metrics_history':          self.val_metrics_history,
            'args':                         vars(self.args),
        }
        torch.save(ckpt, os.path.join(self.output_dir, 'checkpoint_latest.pth'))
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(ckpt, best_path)
            print(f'Best checkpoint saved → {best_path}')
        if epoch % self.args.save_freq == 0:
            torch.save(ckpt,
                       os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth'))

    def load_checkpoint(self, path):
        print(f'Loading checkpoint: {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch                = ckpt['epoch'] + 1
        self.best_decision_miou         = ckpt.get('best_decision_miou',
                                                    ckpt.get('best_miou', 0.0))
        self.best_strict_miou           = ckpt.get('best_strict_miou',   0.0)
        self.best_shadow_iou            = ckpt.get('best_shadow_iou',    0.0)
        self.best_f1                    = ckpt.get('best_f1',             0.0)
        self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        self.train_losses               = ckpt.get('train_losses', [])
        self.val_losses                 = ckpt.get('val_losses',   [])
        _keys = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = ckpt.get(
            'train_metrics_history', {k: [] for k in _keys})
        self.val_metrics_history   = ckpt.get(
            'val_metrics_history',   {k: [] for k in _keys})
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict per-image')
        print(f'Resumed from epoch {ckpt["epoch"]}  '
              f'best {metric_label} mIOU: {self.best_decision_miou:.2f}%  '
              f'epochs w/o improvement: {self.epochs_without_improvement}')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print('\n' + '=' * 50)
        print('Starting DINOv3 + MRFP training...')
        print('=' * 50)

        patience     = self.args.early_stopping_patience
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_for_decisions
                        else 'Strict per-image mIOU')
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            # Epoch-based LR schedule (not metric-gated)
            current_lr = self.scheduler.step(epoch)
            print(f'\nLearning rate: {current_lr:.2e}')

            train_loss, train_metrics = self.train_epoch(epoch + 1)
            val_loss, val_metrics, det = self.validate(epoch + 1)

            # Decision metric (per-image, from DetailedEvaluator)
            decision_miou = self._get_decision_miou(det)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou         = decision_miou
                is_best                         = True
                self.epochs_without_improvement = 0
                print(f'*** New best {metric_label}: '
                      f'{self.best_decision_miou:.2f}% ***')
            else:
                self.epochs_without_improvement += 1

            # Track pooled bests for reference logging only
            if val_metrics['mIOU'] > self.best_strict_miou:
                self.best_strict_miou = val_metrics['mIOU']
                print(f'New best strict pooled mIOU (reference): '
                      f'{self.best_strict_miou:.2f}%')
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
                print(f'\nEarly stopping: no {metric_label} improvement '
                      f'for {patience} epochs.')
                break

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best strict pooled mIOU (reference): {self.best_strict_miou:.2f}%')
        print(f'Best Shadow IoU:                      {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                              {self.best_f1:.2f}%')

        # Plots
        print('\nGenerating plots...')
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'))
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png'))

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
            print('WARNING: best checkpoint not found, using current model weights')

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

        print('\n' + '=' * 50)
        print('Pooled Test Results (reference):')
        print('=' * 50)
        for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']:
            print(f'  {k:12s}: {metrics[k]:.2f}%')

        print('\n' + '=' * 50)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('=' * 50)
        strict   = det['boundary_tolerant']['strict']
        tolerant = det['boundary_tolerant'][self.tol_key]
        print(f"  Strict   — F1: {strict['f1']:.2f}%   mIOU: {strict['iou']:.2f}%")
        print(f"  Tolerant (±{self.args.boundary_tolerance}px) — "
              f"F1: {tolerant['f1']:.2f}%   mIOU: {tolerant['iou']:.2f}%")
        print(f"  Pixels excluded: {tolerant['pixels_excluded']} "
              f"({tolerant['pct_excluded']:.1f}%)")

        if 'size_stratified' in det:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in det['size_stratified']:
                    m = det['size_stratified'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'size_stratified_tolerant' in det:
            print(f'\nSize-Stratified (Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in det['size_stratified_tolerant']:
                    m = det['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if ('fp_fn_analysis' in det
                and 'fp' in det['fp_fn_analysis']):
            fp = det['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({'standard': metrics, 'detailed': det}, f, indent=4)
        print(f'\nResults saved → {results_path}')

        print('\nGenerating best/worst prediction visualisations...')
        try:
            save_best_worst_visualizations(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except Exception as e:
            print(f'Visualisation skipped: {e}')

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