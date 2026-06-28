"""
Training script for DINOv3-FADA Shadow Detection
Implements frequency-adapted domain generalisation (Bi et al., NeurIPS 2024)
on a frozen DINOv3 ViT-S/16 backbone for cross-city shadow detection.

Key differences from train_dinov3.py
─────────────────────────────────────
MODEL     : DINOv3FADAShadowDetector (frozen ViT + trainable FADA + decoder)
OPTIMIZER : Adam  lr=1e-4  wd=1e-4   (paper default for frozen-backbone FADA)
SCHEDULER : ReduceLROnPlateau(mode='max', factor=0.5, patience=3)
            stepped on per-image decision mIOU after every val epoch
LOSS      : single CE (no auxiliary branches — same as base DINOv3)

Decision metrics (best checkpoint, early stopping) are per-image mIOU from
DetailedEvaluator — never pooled ShadowMetrics.
When --eval_boundary_tolerant is set, tolerant mIOU (±K px band excluded)
drives all decisions; otherwise strict per-image mIOU is used.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_model_fada import DINOv3FADAShadowDetector
from data.dataset import get_dataloaders
from utils.losses import CrossEntropyLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import plot_metrics_curves, save_best_worst_visualizations
from utils.evaluation_detailed import DetailedEvaluator

print("=" * 50)
print("GPU DIAGNOSTICS")
print("=" * 50)
print(f"CUDA available   : {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device   : {torch.cuda.current_device()}")
    print(f"Device name      : {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("=" * 50)


# ──────────────────────────────────────────────────────────────────────────
# Loss visualisation (two panels: overview + per-component at own scale)
# ──────────────────────────────────────────────────────────────────────────

def _plot_fada_loss_curves(
    train_losses: list,
    val_losses: list,
    save_path: str,
    title: str = 'DINOv3-FADA — Training Loss Curves',
) -> None:
    """
    Two-panel loss figure.

    Panel 1 — Overview:   Train CE + Val CE on a SHARED y-axis so
                          relative magnitude is visible at a glance.
    Panel 2 — Component:  CE loss alone (train + val) at its OWN y-axis
                          scale so small decreases are clearly visible.
                          No 'total loss' line is added here.

    For DINOv3-FADA there is only one loss component (cross-entropy),
    so the two panels carry the same data at different scales.  If the
    model is extended to include auxiliary losses, add extra component
    panels following the same pattern.
    """
    epochs = list(range(1, len(train_losses) + 1))
    if not epochs:
        return

    fig = plt.figure(figsize=(10, 9))
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55)

    _C_OV_TR = '#2E86AB'
    _C_OV_VL = '#A23B72'
    _C_CE_TR  = '#F18F01'
    _C_CE_VL  = '#C73E1D'
    _MKWARGS  = dict(ms=3.5, mfc='white', mew=1.2)

    # ---- Panel 1: Overview (shared y-axis) --------------------------------
    ax_ov = fig.add_subplot(gs[0])
    ax_ov.plot(epochs, train_losses, '-',  lw=2.0, color=_C_OV_TR,
               label='Train CE', marker='o', **_MKWARGS)
    ax_ov.plot(epochs, val_losses,   '--', lw=1.8, color=_C_OV_VL,
               label='Val CE',   marker='s', **_MKWARGS)
    ax_ov.set_title('Overview — all losses (shared y-axis)', fontweight='bold', pad=6)
    ax_ov.set_xlabel('Epoch')
    ax_ov.set_ylabel('Loss')
    ax_ov.legend(fontsize=9, framealpha=0.88)
    ax_ov.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax_ov.spines['top'].set_visible(False)
    ax_ov.spines['right'].set_visible(False)

    # ---- Panel 2: CE component at own scale (no total loss line) ----------
    ax_ce = fig.add_subplot(gs[1])
    ax_ce.plot(epochs, train_losses, '-',  lw=1.8, color=_C_CE_TR,
               label='Train', marker='o', **_MKWARGS)
    ax_ce.plot(epochs, val_losses,   '--', lw=1.8, color=_C_CE_VL,
               label='Val',   marker='s', **_MKWARGS)
    ax_ce.set_title('Cross-Entropy Loss  (own y-axis scale)', fontweight='bold', pad=6)
    ax_ce.set_xlabel('Epoch')
    ax_ce.set_ylabel('CE Loss')
    ax_ce.legend(fontsize=9, framealpha=0.88)
    ax_ce.grid(True, alpha=0.22, ls='--', lw=0.6)
    ax_ce.spines['top'].set_visible(False)
    ax_ce.spines['right'].set_visible(False)

    fig.suptitle(title, fontweight='bold', fontsize=13, y=1.01)
    fig.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Loss curves saved → {save_path}')


# ──────────────────────────────────────────────────────────────────────────
# CLI arguments
# ──────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Train DINOv3-FADA for domain-generalised shadow detection'
    )

    # ---- Data ---------------------------------------------------------------
    p.add_argument('--data_root', type=str, default=None,
                   help='Dataset root (required for single mode)')
    p.add_argument('--img_size',  type=int, default=384)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # ---- LOCO / multi-city --------------------------------------------------
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities', type=str, nargs='+', default=None)

    # ---- Model --------------------------------------------------------------
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--model_name', type=str, default='dinov3_vits16',
                   choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--weights_path', type=str, default=None,
                   help='Path to DINOv3 pretrained weights .pth file')
    p.add_argument('--pretrained', action='store_true', default=True)

    # ---- FADA-specific ------------------------------------------------------
    p.add_argument('--fada_rank', type=int, default=16,
                   help='LoRA rank r (paper best: 16-32, Table 3)')
    p.add_argument('--fada_token_length', type=int, default=100,
                   help='Token length m (paper default 100, stable 75-125, Fig 8)')
    p.add_argument('--fada_stages', type=int, nargs='+', default=[3, 6, 9, 11],
                   help='ViT block indices to attach FADA (must include 3 6 9 11)')

    # ---- Training -----------------------------------------------------------
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-4,
                   help='Learning rate (paper default for Adam + frozen backbone)')
    p.add_argument('--lr_fada', type=float, default=None,
                   help='Separate LR for FADA adapters (default: same as --lr)')
    p.add_argument('--lr_decoder', type=float, default=None,
                   help='Separate LR for decoder (default: same as --lr)')
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # ---- ReduceLROnPlateau --------------------------------------------------
    p.add_argument('--sched_factor',  type=float, default=0.5)
    p.add_argument('--sched_patience', type=int,  default=3)

    # ---- Checkpoint / logging -----------------------------------------------
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq',  type=int, default=10)
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--eval_only',  action='store_true')

    # ---- Device -------------------------------------------------------------
    p.add_argument('--device', type=str, default='cuda')

    # ---- Boundary-tolerant evaluation ---------------------------------------
    p.add_argument('--eval_boundary_tolerant', action='store_true',
                   help='Use tolerant mIOU for all decisions (best ckpt, early stop)')
    p.add_argument('--boundary_tolerance', type=int, default=2,
                   help="Don't-care band half-width in pixels (default: 2)")

    # ---- Early stopping -----------------------------------------------------
    p.add_argument('--early_stopping_patience', type=int, default=15,
                   help='Patience in epochs without improvement (0 = disabled)')

    # ---- FDA input augmentation (source-domain only, optional) --------------
    p.add_argument('--use_fda', action='store_true',
                   help='Apply FDA pixel-level augmentation to training images')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L', type=float, default=0.01)

    # ---- Comparison (passed from shell, optional) ---------------------------
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root',     type=str, default=None)

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────

class TrainerFADA:
    """Trainer for DINOv3-FADA shadow detection."""

    def __init__(self, args):
        self.args = args

        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Tolerant metric key
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # ------------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------------
        if args.mode == 'single':
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = f'dinov3_fada_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'dinov3_fada_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'dinov3_fada_loco_holdout_{test_city}_{args.resolution}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        print('\nInitialising DINOv3-FADA …')
        self.model = DINOv3FADAShadowDetector(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            fada_stages=tuple(args.fada_stages),
            fada_token_length=args.fada_token_length,
            fada_rank=args.fada_rank,
        ).to(self.device)
        self.model.count_parameters()

        # ------------------------------------------------------------------
        # Loss — single CE (DINOv3 has no auxiliary branches)
        # ------------------------------------------------------------------
        self.criterion = CrossEntropyLoss()

        # ------------------------------------------------------------------
        # Optimiser — Adam, only trainable params (FADA + decoder)
        # Paper default: Adam lr=1e-4, weight_decay=1e-4
        # ------------------------------------------------------------------
        lr_fada    = args.lr_fada    if args.lr_fada    is not None else args.lr
        lr_decoder = args.lr_decoder if args.lr_decoder is not None else args.lr

        param_groups = self.model.get_param_groups(
            lr_fada=lr_fada, lr_decoder=lr_decoder)
        self.optimizer = optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        for pg in self.optimizer.param_groups:
            n = pg.get('name', 'unnamed')
            print(f"  Optimiser group '{n}': "
                  f"{sum(p.numel() for p in pg['params']):,} params  lr={pg['lr']}")

        # ------------------------------------------------------------------
        # LR Scheduler — ReduceLROnPlateau on decision mIOU
        # (metric-gated, not epoch-based)
        # ------------------------------------------------------------------
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=args.sched_factor,
            patience=args.sched_patience,
        )

        # ------------------------------------------------------------------
        # Decision metric
        # ------------------------------------------------------------------
        self.use_tolerant_for_decisions = args.eval_boundary_tolerant
        if self.use_tolerant_for_decisions:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU (DetailedEvaluator)')

        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ------------------------------------------------------------------
        # Tracking state
        # ------------------------------------------------------------------
        self.start_epoch = 0
        self.best_decision_miou = 0.0
        self.best_strict_miou   = 0.0
        self.best_shadow_iou    = 0.0
        self.best_f1            = 0.0
        self.epochs_without_improvement = 0

        self.train_losses = []
        self.val_losses   = []
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': [],
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': [],
        }

        if args.resume:
            self._load_checkpoint(args.resume)

        # ------------------------------------------------------------------
        # Datasets
        # ------------------------------------------------------------------
        print('\nLoading datasets …')
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
        print(f'Train: {len(self.dataloaders["train"].dataset)}'
              f'  Val: {len(self.dataloaders["val"].dataset)}'
              f'  Test: {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    # Decision metric helper
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results: dict) -> float:
        """Per-image mIOU that drives all decisions (never pooled)."""
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_for_decisions:
            return bt[self.tol_key]['iou']
        return bt['strict']['iou']

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int):
        self.model.train()   # backbone stays eval via override

        epoch_loss   = 0.0
        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        t0 = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            outputs = self.model(images)               # [B, 2, H, W]
            loss    = self.criterion(outputs, masks)   # per-image mean CE

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            filtered = filter_small_predictions(outputs, min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'  Batch [{batch_idx+1}/{num_batches}] | '
                      f'CE: {loss.item():.4f}')

        epoch_loss /= num_batches
        metrics     = train_metrics.compute()
        elapsed     = time.time() - t0

        print(f'\nTrain results ({elapsed:.1f}s):')
        print(f'  Loss: {epoch_loss:.4f}')
        print(f'  OA: {metrics["OA"]:.2f}%  P: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'ShadowIOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Train/Loss',        epoch_loss,        epoch)
        self.writer.add_scalar('Train/mIOU_pooled', metrics['mIOU'],   epoch)
        self.writer.add_scalar('Train/Shadow_IOU',  metrics['Shadow_IOU'], epoch)
        self.writer.add_scalar('Train/F1',          metrics['F1'],     epoch)
        self.writer.add_scalar('Train/BER',         metrics['BER'],    epoch)

        self.train_losses.append(epoch_loss)
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(metrics[k])

        # DetailedEvaluator per-image metrics
        det = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()
        s_tr = det['boundary_tolerant']['strict']
        t_tr = det['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_perimage',   s_tr['iou'], epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     s_tr['f1'],  epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', t_tr['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   t_tr['f1'],  epoch)
        print(f'  Per-image Strict   : F1={s_tr["f1"]:.2f}%  mIOU={s_tr["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t_tr["f1"]:.2f}%  mIOU={t_tr["iou"]:.2f}%')

        return epoch_loss, metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def _validate(self, epoch: int):
        print('\nValidating …')
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs  = self.model(images)
                val_loss += self.criterion(outputs, masks).item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics   = val_metrics.compute()

        print(f'Val results:')
        print(f'  Loss: {val_loss:.4f}')
        print(f'  OA: {metrics["OA"]:.2f}%  P: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'ShadowIOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Val/Loss',        val_loss,         epoch)
        self.writer.add_scalar('Val/mIOU_pooled', metrics['mIOU'],  epoch)
        self.writer.add_scalar('Val/Shadow_IOU',  metrics['Shadow_IOU'], epoch)
        self.writer.add_scalar('Val/F1',          metrics['F1'],    epoch)
        self.writer.add_scalar('Val/BER',         metrics['BER'],   epoch)

        self.val_losses.append(val_loss)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(metrics[k])

        # DetailedEvaluator per-image metrics
        det = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()
        s_vl = det['boundary_tolerant']['strict']
        t_vl = det['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Val/mIOU_strict_perimage',   s_vl['iou'], epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     s_vl['f1'],  epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', t_vl['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   t_vl['f1'],  epoch)
        print(f'  Per-image Strict   : F1={s_vl["f1"]:.2f}%  mIOU={s_vl["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={t_vl["f1"]:.2f}%  mIOU={t_vl["iou"]:.2f}%')

        return val_loss, metrics, det

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        ckpt = {
            'epoch':                      epoch,
            'model_state_dict':           self.model.state_dict(),
            'optimizer_state_dict':       self.optimizer.state_dict(),
            'scheduler_state_dict':       self.scheduler.state_dict(),
            'best_decision_miou':         self.best_decision_miou,
            'best_strict_miou':           self.best_strict_miou,
            'best_shadow_iou':            self.best_shadow_iou,
            'best_f1':                    self.best_f1,
            'epochs_without_improvement': self.epochs_without_improvement,
            'use_tolerant_for_decisions': self.use_tolerant_for_decisions,
            'train_losses':               self.train_losses,
            'val_losses':                 self.val_losses,
            'train_metrics_history':      self.train_metrics_history,
            'val_metrics_history':        self.val_metrics_history,
            'args':                       vars(self.args),
        }
        latest = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(ckpt, latest)
        print(f'Checkpoint saved → {latest}')

        if is_best:
            best = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(ckpt, best)
            print(f'Best checkpoint  → {best}')

        if epoch % self.args.save_freq == 0:
            ep_path = os.path.join(self.output_dir,
                                   f'checkpoint_epoch_{epoch}.pth')
            torch.save(ckpt, ep_path)

    def _load_checkpoint(self, path: str) -> None:
        print(f'Loading checkpoint from {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_decision_miou = ckpt.get('best_decision_miou', 0.0)
        self.best_strict_miou   = ckpt.get('best_strict_miou',   0.0)
        self.best_shadow_iou    = ckpt.get('best_shadow_iou',    0.0)
        self.best_f1            = ckpt.get('best_f1',            0.0)
        self.epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses   = ckpt.get('val_losses',   [])
        self.train_metrics_history = ckpt.get('train_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []})
        self.val_metrics_history = ckpt.get('val_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []})
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict per-image')
        print(f'Resumed from epoch {ckpt["epoch"]}')
        print(f'Best decision mIOU ({metric_label}): {self.best_decision_miou:.2f}%')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        print('\n' + '=' * 50)
        print('Starting DINOv3-FADA training …')
        print(f'  FADA: rank={self.args.fada_rank}'
              f'  m={self.args.fada_token_length}'
              f'  stages={self.args.fada_stages}')
        print('=' * 50)

        patience = self.args.early_stopping_patience
        metric_label = (
            f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
            if self.use_tolerant_for_decisions else 'Strict per-image mIOU')
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            # ---- Train -------------------------------------------------------
            _train_loss, train_metrics = self._train_epoch(epoch + 1)

            # ---- Validate ----------------------------------------------------
            val_loss, val_metrics, det = self._validate(epoch + 1)

            # ---- Decision metric (per-image, from DetailedEvaluator) ---------
            decision_miou = self._get_decision_miou(det)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            # ---- LR scheduler (metric-gated) ---------------------------------
            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f'Learning rate: {current_lr:.2e}')
            for i, pg in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(
                    f'Train/LR_{pg.get("name", i)}', pg['lr'], epoch + 1)

            # ---- Best checkpoint --------------------------------------------
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'*** New best {metric_label}: '
                      f'{self.best_decision_miou:.2f}% ***')
            else:
                self.epochs_without_improvement += 1

            # Track pooled bests for reference logging
            if val_metrics['mIOU'] > self.best_strict_miou:
                self.best_strict_miou = val_metrics['mIOU']
                print(f'New best strict pooled mIOU (ref): {self.best_strict_miou:.2f}%')
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
                print(f'New best Shadow IoU: {self.best_shadow_iou:.2f}%')
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                print(f'New best F1: {self.best_f1:.2f}%')

            self._save_checkpoint(epoch + 1, is_best=is_best)
            print('=' * 50)

            # ---- Early stopping ---------------------------------------------
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping: no {metric_label} improvement '
                      f'for {patience} epochs.')
                break

        print('\nTraining complete.')
        print(f'Best {metric_label}         : {self.best_decision_miou:.2f}%')
        print(f'Best strict pooled mIOU (ref): {self.best_strict_miou:.2f}%')
        print(f'Best Shadow IoU              : {self.best_shadow_iou:.2f}%')
        print(f'Best F1                      : {self.best_f1:.2f}%')

        print('\nGenerating plots …')
        _plot_fada_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            title='DINOv3-FADA — Training Loss Curves',
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

    def test(self) -> dict:
        print('\n' + '=' * 50)
        print('Testing DINOv3-FADA …')
        print('=' * 50)

        best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_path):
            self._load_checkpoint(best_path)
        else:
            print('Warning: best checkpoint not found, using current weights.')

        self.model.eval()
        test_metrics = ShadowMetrics()
        det_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs  = self.model(images)
                filtered = filter_small_predictions(outputs, min_pixels=10)

                test_metrics.update(filtered, masks)
                preds = torch.argmax(filtered, dim=1)
                det_eval.update(preds, masks, images)

        metrics = test_metrics.compute()
        det_res = det_eval.compute_metrics()

        print('\n' + '=' * 50)
        print('Pooled Test Results (reference):')
        print('=' * 50)
        print(f'OA        : {metrics["OA"]:.2f}%')
        print(f'Precision : {metrics["Precision"]:.2f}%')
        print(f'F1        : {metrics["F1"]:.2f}%')
        print(f'BER       : {metrics["BER"]:.2f}%')
        print(f'mIOU      : {metrics["mIOU"]:.2f}%')
        print(f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        print('\n' + '=' * 50)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('=' * 50)
        strict   = det_res['boundary_tolerant']['strict']
        tolerant = det_res['boundary_tolerant'][self.tol_key]
        print(f'Strict   — F1: {strict["f1"]:.2f}%   mIOU: {strict["iou"]:.2f}%')
        print(f'Tolerant (±{self.args.boundary_tolerance}px) — '
              f'F1: {tolerant["f1"]:.2f}%   mIOU: {tolerant["iou"]:.2f}%')
        print(f'Pixels excluded by band: {tolerant["pixels_excluded"]} '
              f'({tolerant["pct_excluded"]:.1f}%)')

        if 'size_stratified' in det_res:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in det_res['size_stratified']:
                    m = det_res['size_stratified'][cat]
                    print(f'  {cat:8s}: Miss={m["miss_rate"]:5.1f}%  '
                          f'IoU={m["avg_iou"]:5.1f}%  ({m["total"]} shadows)')

        if 'size_stratified_tolerant' in det_res:
            print(f'\nSize-Stratified (Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in det_res['size_stratified_tolerant']:
                    m = det_res['size_stratified_tolerant'][cat]
                    print(f'  {cat:8s}: Miss={m["miss_rate"]:5.1f}%  '
                          f'IoU={m["avg_iou"]:5.1f}%  ({m["total"]} shadows)')

        if ('fp_fn_analysis' in det_res
                and 'fp' in det_res['fp_fn_analysis']):
            fp = det_res['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f'  Within  1px: {fp["pct_within_1px"]:.1f}%')
            print(f'  Within  5px: {fp["pct_within_5px"]:.1f}%')
            print(f'  Within 10px: {fp["pct_within_10px"]:.1f}%')

        results_to_save = {
            'standard': metrics,
            'detailed': det_res,
            'fada_config': {
                'rank':         self.args.fada_rank,
                'token_length': self.args.fada_token_length,
                'stages':       self.args.fada_stages,
            },
        }
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f'\nResults saved → {results_path}')

        try:
            print('\nGenerating best/worst visualisations …')
            save_best_worst_visualizations(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except Exception as e:
            print(f'Visualisation skipped: {e}')

        return metrics


# ──────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────

def main():
    args    = get_args()
    trainer = TrainerFADA(args)
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()