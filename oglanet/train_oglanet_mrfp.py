"""
Training script for OGLANet + MRFP / MRFP+
===========================================
Adds Multi-Resolution Feature Perturbation (CVPR 2024) to OGLANet for
cross-city shadow-detection generalization.

Key differences from base train.py:
    • Model  : OGLANetMRFP  (HRFP + NP+ + optional HRFP+)
    • Loss   : Per-image mean CE summed over 6 deep-supervision outputs
    • Eval   : Tolerant per-image mIOU drives all decisions (unchanged)
    • Viz    : Reuses utils/visualization.py:
               – Overview subplot (all losses, shared y-axis)
               – One subplot per component loss (loss1-6, own scale, NO total)

Decision metrics (LR scheduler, best checkpoint, early stopping) use
per-image mIOU from DetailedEvaluator — never pooled ShadowMetrics.

When --eval_boundary_tolerant is set, decisions use tolerant per-image mIOU.
Otherwise strict per-image mIOU is used.

Perturbation modules are training-only; inference is identical to base OGLANet.
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

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_mrfp import OGLANetMRFP
from data.dataset import get_dataloaders
from utils.evaluation_detailed import DetailedEvaluator
from utils.losses_oglanet_mrfp import OGLANetMRFPLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    save_best_worst_visualizations,
)

# ======================================================================
# GPU diagnostics
# ======================================================================
print('=' * 50)
print('GPU DIAGNOSTICS')
print('=' * 50)
print(f'CUDA available:       {torch.cuda.is_available()}')
print(f'CUDA device count:    {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current device:       {torch.cuda.current_device()}')
    print(f'Device name:          {torch.cuda.get_device_name(0)}')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")}')
print('=' * 50)


# ======================================================================
# Args
# ======================================================================

def get_args():
    p = argparse.ArgumentParser(
        description='Train OGLANet + MRFP for Shadow Detection')

    # Data
    p.add_argument('--data_root',   type=str, default=None,
                   help='Root directory of dataset (single mode)')
    p.add_argument('--img_size',    type=int, default=384)
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # Multi-city / LOCO
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution',     type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id',  type=int, default=None,
                   choices=[0, 1, 2])
    p.add_argument('--cities',   type=str, nargs='+', default=None)

    # Model
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--pretrained',  action='store_true', default=True)

    # MRFP
    p.add_argument('--mrfp_plus',      action='store_true', default=True,
                   help='Use MRFP+ (HRFP+HRFP++NP+). Default True.')
    p.add_argument('--no_mrfp_plus',   action='store_true', default=False,
                   help='Disable HRFP+; use plain MRFP (HRFP+NP+).')
    p.add_argument('--hrfp_prob',      type=float, default=0.5)
    p.add_argument('--np_prob',        type=float, default=0.5)
    p.add_argument('--hrfp_plus_prob', type=float, default=0.5)
    p.add_argument('--hrfp_bn_std',    type=float, default=0.5)

    # Training
    # lr=0.0001 matches user-confirmed preference for MRFP-based training.
    # epochs=100 matches base OGLANet (full training from scratch, not
    # fine-tuning — unlike MAMNet+MRFP which is a 7-epoch fine-tune).
    # adamax matches base OGLANet paper spec.
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--lr',           type=float, default=0.0001)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--optimizer',    type=str,   default='adamax',
                   choices=['adam', 'adamax'])

    # Contrast channel
    p.add_argument('--use_contrast', action='store_true')

    # FDA (kept for completeness; not expected in LOCO runs)
    p.add_argument('--use_fda',         action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L',           type=float, default=0.01)

    # Checkpoint / logging
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq',  type=int, default=10)
    p.add_argument('--resume',     type=str, default=None)
    p.add_argument('--eval_only',  action='store_true')

    # Device
    p.add_argument('--device', type=str, default='cuda')

    # Boundary-tolerant evaluation
    # DetailedEvaluator ALWAYS runs.  This flag controls which per-image
    # metric drives LR scheduler / checkpointing / early stopping.
    p.add_argument('--eval_boundary_tolerant', action='store_true',
                   help='Use tolerant mIOU (±boundary_tolerance px) for all '
                        'decisions.  Strict mIOU is always logged too.')
    p.add_argument('--boundary_tolerance', type=int, default=2,
                   help="Don't-care band half-width in pixels (default: 2).")

    # Early stopping
    p.add_argument('--early_stopping_patience', type=int, default=None,
                   help='Epochs without improvement before stopping. '
                        '0 or None = disabled.')

    args = p.parse_args()

    # Resolve MRFP+ flag
    if args.no_mrfp_plus:
        args.mrfp_plus = False

    return args


# ======================================================================
# Trainer
# ======================================================================

class Trainer:

    def __init__(self, args):
        self.args   = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Tolerant key for DetailedEvaluator
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # ---- Output directory ----
        variant = 'mrfp_plus' if args.mrfp_plus else 'mrfp'

        if args.mode == 'single':
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = f'oglanet_{variant}_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'oglanet_{variant}_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = (f'oglanet_{variant}_loco_holdout_'
                        f'{test_city}_{args.resolution}_1')

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ---- Model ----
        print('Initializing OGLANetMRFP …')
        self.model = OGLANetMRFP(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
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
        self.criterion = OGLANetMRFPLoss()

        # ---- Optimizer & scheduler ----
        # adamax matches base OGLANet paper spec; lr=0.0001 per user preference
        # for MRFP-based training; weight_decay adds mild regularization.
        if args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5)

        # ---- Decision metric ----
        # DetailedEvaluator ALWAYS runs.
        # eval_boundary_tolerant controls WHICH per-image metric drives
        # decisions: tolerant mIOU (True) or strict mIOU (False).
        # ShadowMetrics (pooled) is logged for reference only.
        self.use_tolerant_decision = args.eval_boundary_tolerant
        label = (f'TOLERANT mIOU (±{args.boundary_tolerance}px)'
                 if self.use_tolerant_decision else 'STRICT per-image mIOU')
        print(f'>> Decision metric: {label}')

        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ---- Tracking ----
        self.start_epoch                = 0
        self.best_miou                  = 0.0   # pooled, reference only
        self.best_shadow_iou            = 0.0   # pooled, reference only
        self.best_f1                    = 0.0   # pooled, reference only
        self.best_decision_miou         = 0.0   # drives checkpoint/early-stop
        self.epochs_without_improvement = 0

        # Loss histories — total + per-component
        self.train_losses        = []
        self.train_loss1_history = []
        self.train_loss2_history = []
        self.train_loss3_history = []
        self.train_loss4_history = []
        self.train_loss5_history = []
        self.train_loss6_history = []
        self.val_losses          = []

        # Metric histories (ShadowMetrics pooled — reference only)
        _keys = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = {k: [] for k in _keys}
        self.val_metrics_history   = {k: [] for k in _keys}

        if args.resume:
            self.load_checkpoint(args.resume)

        # ---- Data loaders ----
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
              f'Val: {len(self.dataloaders["val"].dataset)}  '
              f'Test: {len(self.dataloaders["test"].dataset)}')

        if (args.early_stopping_patience is not None
                and args.early_stopping_patience > 0):
            print(f'>> Early stopping patience: '
                  f'{args.early_stopping_patience} epochs')

    # ------------------------------------------------------------------
    # Decision metric helper
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """
        Return the mIOU driving all decisions.
        Both options are per-image means from DetailedEvaluator.
        """
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_decision:
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
        epoch_loss : float  — average total loss
        metrics    : dict   — ShadowMetrics pooled (reference only)
        """
        self.model.train()

        epoch_loss   = 0.0
        epoch_losses = {
            'loss1': 0.0, 'loss2': 0.0, 'loss3': 0.0,
            'loss4': 0.0, 'loss5': 0.0, 'loss6': 0.0,
        }

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        t0 = time.time()

        for bi, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            predictions = self.model(images)       # dict of 6 tensors
            losses      = self.criterion(predictions, masks)
            loss        = losses['total']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics on filtered P6 (consistent with val / test)
            filtered = filter_small_predictions(
                predictions['p6'], min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss += loss.item()
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            if (bi + 1) % 10 == 0 or (bi + 1) == num_batches:
                print(f'  Batch [{bi+1}/{num_batches}]  '
                      f'Total {loss.item():.4f}  '
                      f'L1 {losses["loss1"].item():.4f}  '
                      f'L6 {losses["loss6"].item():.4f}')

        epoch_loss /= num_batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        metrics = train_metrics.compute()
        dt      = time.time() - t0

        print(f'\nTrain Results ({dt:.1f}s):')
        print(f'  Total={epoch_loss:.4f}  '
              + '  '.join(f'{k}={v:.4f}'
                           for k, v in epoch_losses.items()))
        print(f'  OA={metrics["OA"]:.2f}%  P={metrics["Precision"]:.2f}%  '
              f'F1={metrics["F1"]:.2f}%  BER={metrics["BER"]:.2f}%  '
              f'mIOU(pooled)={metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU={metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Train/TotalLoss', epoch_loss, epoch)
        for k, v in epoch_losses.items():
            self.writer.add_scalar(f'Train/{k}', v, epoch)

        # TensorBoard — pooled metrics (reference)
        for k, v in metrics.items():
            self.writer.add_scalar(f'Train/{k}', v, epoch)

        # DetailedEvaluator — per-image metrics (ALWAYS computed)
        det   = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict   = det['boundary_tolerant']['strict']
        tolerant = det['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar(
            'Train/mIOU_strict_perimage',   strict['iou'],   epoch)
        self.writer.add_scalar(
            'Train/F1_strict_perimage',     strict['f1'],    epoch)
        self.writer.add_scalar(
            'Train/mIOU_tolerant_perimage', tolerant['iou'], epoch)
        self.writer.add_scalar(
            'Train/F1_tolerant_perimage',   tolerant['f1'],  epoch)

        print(f'  Per-image Strict:   '
              f'F1={strict["f1"]:.2f}%  mIOU={strict["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        # Histories
        self.train_losses.append(epoch_loss)
        self.train_loss1_history.append(epoch_losses['loss1'])
        self.train_loss2_history.append(epoch_losses['loss2'])
        self.train_loss3_history.append(epoch_losses['loss3'])
        self.train_loss4_history.append(epoch_losses['loss4'])
        self.train_loss5_history.append(epoch_losses['loss5'])
        self.train_loss6_history.append(epoch_losses['loss6'])
        for k in self.train_metrics_history:
            self.train_metrics_history[k].append(metrics[k])

        return epoch_loss, metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        """
        Returns
        -------
        val_loss         : float
        metrics          : dict   — ShadowMetrics pooled (reference)
        detailed_results : dict   — per-image metrics (drive all decisions)
        """
        print('\nValidating …')
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Eval mode: model returns P6 tensor directly
                predictions = self.model(images)

                # Val loss — per-image mean CE on raw P6
                loss      = self.criterion.criterion(predictions, masks)
                val_loss += loss.item()

                # Metrics on filtered predictions (consistent with train/test)
                filtered = filter_small_predictions(
                    predictions, min_pixels=10)
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
        self.writer.add_scalar('Val/Loss',        val_loss,        epoch)
        self.writer.add_scalar('Val/mIOU_pooled', metrics['mIOU'], epoch)
        for k, v in metrics.items():
            self.writer.add_scalar(f'Val/{k}', v, epoch)

        # DetailedEvaluator — per-image metrics (drive all decisions)
        det  = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict   = det['boundary_tolerant']['strict']
        tolerant = det['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar(
            'Val/mIOU_strict_perimage',   strict['iou'],   epoch)
        self.writer.add_scalar(
            'Val/F1_strict_perimage',     strict['f1'],    epoch)
        self.writer.add_scalar(
            'Val/mIOU_tolerant_perimage', tolerant['iou'], epoch)
        self.writer.add_scalar(
            'Val/F1_tolerant_perimage',   tolerant['f1'],  epoch)

        print(f'  Per-image Strict:   '
              f'F1={strict["f1"]:.2f}%  mIOU={strict["iou"]:.2f}%')
        print(f'  Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        # Histories
        self.val_losses.append(val_loss)
        for k in self.val_metrics_history:
            self.val_metrics_history[k].append(metrics[k])

        return val_loss, metrics, det

    # ------------------------------------------------------------------
    # Checkpoint I/O
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
            'train_losses':                 self.train_losses,
            'train_loss1_history':          self.train_loss1_history,
            'train_loss2_history':          self.train_loss2_history,
            'train_loss3_history':          self.train_loss3_history,
            'train_loss4_history':          self.train_loss4_history,
            'train_loss5_history':          self.train_loss5_history,
            'train_loss6_history':          self.train_loss6_history,
            'val_losses':                   self.val_losses,
            'train_metrics_history':        self.train_metrics_history,
            'val_metrics_history':          self.val_metrics_history,
            'args':                         vars(self.args),
        }

        # Always save latest
        path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(ckpt, path)
        print(f'Checkpoint saved → {path}')

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(ckpt, best_path)
            print(f'Best checkpoint   → {best_path}')

        if epoch % self.args.save_freq == 0:
            ep_path = os.path.join(
                self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(ckpt, ep_path)

    def load_checkpoint(self, path):
        print(f'Loading checkpoint: {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        try:
            self.model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e):
                print('WARNING: size mismatch — attempting partial load')
                sd_ckpt  = ckpt['model_state_dict']
                sd_model = self.model.state_dict()
                matched  = {k: v for k, v in sd_ckpt.items()
                            if k in sd_model
                            and v.size() == sd_model[k].size()}
                sd_model.update(matched)
                self.model.load_state_dict(sd_model)
                print(f'Loaded {len(matched)}/{len(sd_ckpt)} layers')
            else:
                raise

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch              = ckpt['epoch'] + 1
        self.best_miou                = ckpt.get('best_miou', 0.0)
        self.best_shadow_iou          = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1                  = ckpt.get('best_f1', 0.0)
        self.best_decision_miou       = ckpt.get('best_decision_miou', 0.0)
        self.epochs_without_improvement = ckpt.get(
            'epochs_without_improvement', 0)

        self.train_losses        = ckpt.get('train_losses', [])
        self.train_loss1_history = ckpt.get('train_loss1_history', [])
        self.train_loss2_history = ckpt.get('train_loss2_history', [])
        self.train_loss3_history = ckpt.get('train_loss3_history', [])
        self.train_loss4_history = ckpt.get('train_loss4_history', [])
        self.train_loss5_history = ckpt.get('train_loss5_history', [])
        self.train_loss6_history = ckpt.get('train_loss6_history', [])
        self.val_losses          = ckpt.get('val_losses', [])

        _keys = ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']
        self.train_metrics_history = ckpt.get(
            'train_metrics_history', {k: [] for k in _keys})
        self.val_metrics_history   = ckpt.get(
            'val_metrics_history', {k: [] for k in _keys})

        metric_label = (
            f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
            if self.use_tolerant_decision else 'Strict per-image mIOU')
        print(f'Resumed from epoch {ckpt["epoch"]}  '
              f'best {metric_label}: {self.best_decision_miou:.2f}%  '
              f'epochs w/o improvement: {self.epochs_without_improvement}')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print('\n' + '=' * 50)
        print('Starting OGLANet + MRFP training …')
        metric_label = (
            f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
            if self.use_tolerant_decision else 'Strict per-image mIOU')
        print(f'Decision metric: {metric_label}')
        print('=' * 50)

        patience = (self.args.early_stopping_patience
                    if self.args.early_stopping_patience is not None else 0)
        if patience > 0:
            print(f'Early stopping patience: {patience}')

        for epoch in range(self.start_epoch, self.args.epochs):
            ep = epoch + 1

            # Train
            train_loss, train_met = self.train_epoch(ep)

            # Validate (always returns detailed_results)
            val_loss, val_met, det = self.validate(ep)

            # Decision metric (per-image, from DetailedEvaluator)
            decision_miou = self._get_decision_miou(det)
            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, ep)

            # Best checkpoint
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou         = decision_miou
                is_best                         = True
                self.epochs_without_improvement = 0
                print(f'>> New best {metric_label}: '
                      f'{self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            # Track pooled bests for reference logging only
            if val_met['mIOU'] > self.best_miou:
                self.best_miou = val_met['mIOU']
            if val_met['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_met['Shadow_IOU']
            if val_met['F1'] > self.best_f1:
                self.best_f1 = val_met['F1']

            self.save_checkpoint(ep, is_best=is_best)

            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', lr, ep)

            # Early stopping
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping after {patience} epochs '
                      f'without improvement in {metric_label}.')
                break

            print('=' * 50)

        # ---- Plots ----
        print('\nGenerating plots …')
        # plot_loss_curves from utils/visualization.py:
        #   – Row 0 overview: all losses on shared y-axis
        #   – Individual panels: Val Total + Train loss1..loss6 at own scale
        #   – Total loss NOT shown in individual panels (per user request)
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            component_losses={
                'loss1': self.train_loss1_history,
                'loss2': self.train_loss2_history,
                'loss3': self.train_loss3_history,
                'loss4': self.train_loss4_history,
                'loss5': self.train_loss5_history,
                'loss6': self.train_loss6_history,
            },
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png'),
        )

        print('\nTraining completed!')
        print(f'Best {metric_label}:          {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (reference): {self.best_miou:.2f}%')
        print(f'Best Shadow IoU:              {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                      {self.best_f1:.2f}%')

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

                # Eval mode: model returns P6
                predictions = self.model(images)
                filtered    = filter_small_predictions(
                    predictions, min_pixels=10)
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
                        print(f'    {cat:8s}: '
                              f'Miss={m["miss_rate"]:5.1f}%  '
                              f'IoU={m["avg_iou"]:5.1f}%  '
                              f'({m["total"]} shadows)')

        # FP analysis
        if ('fp_fn_analysis' in det
                and 'fp' in det['fp_fn_analysis']):
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