"""
Training script for DINOv3 + ISW (Instance Selective Whitening).

Adds RobustNet-style ISW regularisation on top of DINOv3 shadow detection.

Total loss = CE_loss + λ_ISW × ISW_loss

where ISW_loss is the per-image mean instance selective whitening loss
computed from DINOv3 ViT encoder features at blocks [3, 6, 9].

Decision metrics (best checkpoint, early stopping) use per-image mIOU from
DetailedEvaluator — never the pooled ShadowMetrics value.

When --eval_boundary_tolerant is set, decisions use per-image TOLERANT mIOU
(boundary band width = --boundary_tolerance px, default 2).
Otherwise decisions use per-image STRICT mIOU.

Val ISW loss is computed and logged for monitoring only (does not affect
the scheduler or any decisions).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dinov3_model import DINOv3ShadowDetector
from data.dataset import get_dataloaders
from utils.losses import CrossEntropyLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.evaluation_detailed import DetailedEvaluator
from utils.isw_loss_dinov3 import ISWLoss, DINOv3FeatureHooks
from utils.visualization_dinov3_isw import (
    plot_loss_curves_dinov3_isw,
    plot_metrics_curves,
    save_best_worst_visualizations,
)

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


# ──────────────────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Train DINOv3 + ISW for Shadow Detection')

    # Data
    p.add_argument('--data_root',       type=str, default=None)
    p.add_argument('--img_size',        type=int, default=384)
    p.add_argument('--batch_size',      type=int, default=8)
    p.add_argument('--num_workers',     type=int, default=1)

    # LOCO / multi-city
    p.add_argument('--mode',            type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root',  type=str, default=None)
    p.add_argument('--resolution',      type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id',         type=int, default=None,
                   choices=[0, 1, 2])
    p.add_argument('--cities',          type=str, nargs='+', default=None)

    # Model
    p.add_argument('--num_classes',     type=int, default=2)
    p.add_argument('--model_name',      type=str, default='dinov3_vits16',
                   choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--weights_path',    type=str, default=None)
    p.add_argument('--pretrained',      action='store_true', default=True)
    p.add_argument('--frozen_stages',   type=int, default=-1)

    # Training  (cosine warmup — same as base DINOv3)
    p.add_argument('--epochs',          type=int, default=50)
    p.add_argument('--lr',              type=float, default=5e-5)
    p.add_argument('--weight_decay',    type=float, default=0.05)
    p.add_argument('--warmup_epochs',   type=int, default=5)
    p.add_argument('--min_lr',          type=float, default=1e-6)

    # FDA
    p.add_argument('--use_fda',         action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L',           type=float, default=0.01)

    # Checkpoint & logging
    p.add_argument('--output_dir',      type=str, default='./outputs')
    p.add_argument('--save_freq',       type=int, default=5)
    p.add_argument('--resume',          type=str, default=None)
    p.add_argument('--eval_only',       action='store_true')

    # Device
    p.add_argument('--device',          type=str, default='cuda')

    # Boundary-tolerant evaluation
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--boundary_tolerance',     type=int, default=2)

    # Early stopping
    p.add_argument('--early_stopping_patience', type=int, default=15)

    # Comparison / inference dirs (passed from shell but optional)
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root',     type=str, default=None)

    # ── ISW-specific ──────────────────────────────────────────────────────────
    p.add_argument('--isw_mask_dir', type=str, required=True,
                   help='Directory with precomputed ISW mask .npy files '
                        '(from compute_isw_masks_dinov3.py)')
    p.add_argument('--isw_lambda',   type=float, default=0.6,
                   help='Weight λ for ISW loss (default: 0.6)')
    p.add_argument('--isw_layers',   type=str, default='block3,block6,block9',
                   help='Comma-separated block names for ISW hooks '
                        '(must match precomputed masks)')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Cosine warmup LR scheduler  (same as base DINOv3 — epoch-based, not metric)
# ──────────────────────────────────────────────────────────────────────────────

class CosineWarmupScheduler:
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
            lr = (self.min_lr + (self.base_lr - self.min_lr)
                  * 0.5 * (1 + np.cos(np.pi * progress)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class TrainerDINOv3ISW:
    """Trainer for DINOv3 + ISW shadow detection."""

    def __init__(self, args):
        self.args   = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Dynamic tolerant key
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # ── Output directory ──────────────────────────────────────────────────
        fda_suffix = '_fda' if args.use_fda else ''
        base_suffix = f'dinov3_isw{fda_suffix}'

        if args.mode == 'single':
            city = args.data_root.rstrip('/').split('/')[-2]
            res  = args.data_root.rstrip('/').split('/')[-1]
            exp_name = f'{base_suffix}_{city}_{res}_1'
        elif args.mode == 'all':
            exp_name = f'{base_suffix}_all_{args.resolution}_1'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name  = f'{base_suffix}_loco_holdout_{test_city}_{args.resolution}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ── Model ─────────────────────────────────────────────────────────────
        print('Initialising DINOv3 model...')
        self.model = DINOv3ShadowDetector(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
        ).to(self.device)

        total_p    = sum(p.numel() for p in self.model.parameters())
        trainable_p = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
        print(f'Total parameters:     {total_p:,}')
        print(f'Trainable parameters: {trainable_p:,}')

        # ── ISW loss + ViT block hooks ─────────────────────────────────────────
        isw_layers = args.isw_layers.split(',')
        print(f'\nISW layers: {isw_layers}  λ={args.isw_lambda}')
        print(f'ISW mask dir: {args.isw_mask_dir}')

        self.isw_loss_module = ISWLoss(
            mask_dir=args.isw_mask_dir,
            layer_names=isw_layers,
        ).to(self.device)

        self.feature_hooks = DINOv3FeatureHooks(
            self.model,
            layer_names=isw_layers,
            img_size=args.img_size,
            patch_size=16,
        )

        # ── Losses & optimiser ─────────────────────────────────────────────────
        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )

        # Epoch-based cosine warmup (same as base DINOv3 — not metric-gated)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr,
        )

        # ── Decision metric ────────────────────────────────────────────────────
        self.use_tolerant_for_decisions = args.eval_boundary_tolerant
        if self.use_tolerant_for_decisions:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU')

        # DetailedEvaluator always instantiated regardless of flag
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val   = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ── Tracking variables ─────────────────────────────────────────────────
        self.start_epoch                = 0
        self.best_decision_miou         = 0.0
        self.best_strict_miou           = 0.0
        self.best_shadow_iou            = 0.0
        self.best_f1                    = 0.0
        self.epochs_without_improvement = 0

        # Loss history
        self.train_losses     = []   # total (CE + λ×ISW) per epoch
        self.train_ce_losses  = []   # CE component
        self.train_isw_losses = []   # λ×ISW component
        self.val_losses       = []   # val CE per epoch
        self.val_isw_losses   = []   # val λ×ISW (monitoring)

        # Pooled metric history (ShadowMetrics, reference only)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # ── Dataloaders ────────────────────────────────────────────────────────
        print('\nLoading datasets...')
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

    # ──────────────────────────────────────────────────────────────────────────
    # Decision metric helper
    # ──────────────────────────────────────────────────────────────────────────

    def _get_decision_miou(self, detailed_results):
        """
        Per-image mIOU from DetailedEvaluator that drives all decisions.
        Never uses pooled ShadowMetrics value.
        """
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_for_decisions:
            return bt[self.tol_key]['iou']
        else:
            return bt['strict']['iou']

    # ──────────────────────────────────────────────────────────────────────────
    # Train one epoch
    # ──────────────────────────────────────────────────────────────────────────

    def train_epoch(self, epoch):
        self.model.train()

        epoch_loss     = 0.0
        epoch_ce_loss  = 0.0
        epoch_isw_loss = 0.0

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        t0 = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            # Forward — feature hooks fire automatically inside model.backbone
            outputs = self.model(images)

            # CE segmentation loss
            ce_loss = self.criterion(outputs, masks)

            # ISW loss from hooked ViT block features
            isw_raw      = self.isw_loss_module(self.feature_hooks.features)
            isw_weighted = self.args.isw_lambda * isw_raw

            # Total loss
            loss = ce_loss + isw_weighted

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics (filtered, consistent with val/test)
            filtered = filter_small_predictions(outputs, min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss     += loss.item()
            epoch_ce_loss  += ce_loss.item()
            epoch_isw_loss += isw_weighted.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'  Batch [{batch_idx + 1:4d}/{num_batches}] | '
                      f'Total: {loss.item():.4f} | '
                      f'CE: {ce_loss.item():.4f} | '
                      f'ISW: {isw_weighted.item():.4f}')

        epoch_loss     /= num_batches
        epoch_ce_loss  /= num_batches
        epoch_isw_loss /= num_batches

        metrics    = train_metrics.compute()
        epoch_time = time.time() - t0

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Total: {epoch_loss:.4f} | '
              f'CE: {epoch_ce_loss:.4f} | ISW: {epoch_isw_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Train/Loss',      epoch_loss,     epoch)
        self.writer.add_scalar('Train/CELoss',    epoch_ce_loss,  epoch)
        self.writer.add_scalar('Train/ISWLoss',   epoch_isw_loss, epoch)
        # TensorBoard — pooled metrics (reference)
        self.writer.add_scalar('Train/OA',          metrics['OA'],         epoch)
        self.writer.add_scalar('Train/Precision',   metrics['Precision'],  epoch)
        self.writer.add_scalar('Train/F1',          metrics['F1'],         epoch)
        self.writer.add_scalar('Train/BER',         metrics['BER'],        epoch)
        self.writer.add_scalar('Train/mIOU_pooled', metrics['mIOU'],       epoch)
        self.writer.add_scalar('Train/Shadow_IOU',  metrics['Shadow_IOU'], epoch)

        # Store history
        self.train_losses.append(epoch_loss)
        self.train_ce_losses.append(epoch_ce_loss)
        self.train_isw_losses.append(epoch_isw_loss)
        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

        # DetailedEvaluator per-image metrics
        detailed = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict   = detailed['boundary_tolerant']['strict']
        tolerant = detailed['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_perimage',   strict['iou'],   epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     strict['f1'],    epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', tolerant['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   tolerant['f1'],  epoch)

        print(f'Per-image Strict:   F1={strict["f1"]:.2f}%  '
              f'mIOU={strict["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        return epoch_loss, epoch_ce_loss, epoch_isw_loss, metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Validate
    # ──────────────────────────────────────────────────────────────────────────

    def validate(self, epoch):
        print('\nValidating...')
        self.model.eval()

        val_loss     = 0.0
        val_isw_loss = 0.0
        val_metrics  = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Forward — hooks fire
                outputs = self.model(images)

                # Val CE loss
                ce_loss   = self.criterion(outputs, masks)
                val_loss += ce_loss.item()

                # Val ISW (monitoring only — not used for any decision)
                isw_raw       = self.isw_loss_module(self.feature_hooks.features)
                val_isw_loss += (self.args.isw_lambda * isw_raw).item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        n_batches     = len(self.dataloaders['val'])
        val_loss     /= n_batches
        val_isw_loss /= n_batches
        metrics       = val_metrics.compute()

        print(f'Validation Results:')
        print(f'Val CE: {val_loss:.4f} | Val ISW (monitor): {val_isw_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Val/Loss',       val_loss,     epoch)
        self.writer.add_scalar('Val/ISWLoss',    val_isw_loss, epoch)
        # TensorBoard — pooled metrics (reference)
        self.writer.add_scalar('Val/OA',          metrics['OA'],         epoch)
        self.writer.add_scalar('Val/Precision',   metrics['Precision'],  epoch)
        self.writer.add_scalar('Val/F1',          metrics['F1'],         epoch)
        self.writer.add_scalar('Val/BER',         metrics['BER'],        epoch)
        self.writer.add_scalar('Val/mIOU_pooled', metrics['mIOU'],       epoch)
        self.writer.add_scalar('Val/Shadow_IOU',  metrics['Shadow_IOU'], epoch)

        self.val_losses.append(val_loss)
        self.val_isw_losses.append(val_isw_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

        # DetailedEvaluator — always computed; drives decisions
        detailed = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict   = detailed['boundary_tolerant']['strict']
        tolerant = detailed['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Val/mIOU_strict_perimage',   strict['iou'],   epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     strict['f1'],    epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', tolerant['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   tolerant['f1'],  epoch)

        print(f'Per-image Strict:   F1={strict["f1"]:.2f}%  '
              f'mIOU={strict["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        return val_loss, val_isw_loss, metrics, detailed

    # ──────────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch':                        epoch,
            'model_state_dict':             self.model.state_dict(),
            'optimizer_state_dict':         self.optimizer.state_dict(),
            'best_decision_miou':           self.best_decision_miou,
            'best_strict_miou':             self.best_strict_miou,
            'best_shadow_iou':              self.best_shadow_iou,
            'best_f1':                      self.best_f1,
            'epochs_without_improvement':   self.epochs_without_improvement,
            'use_tolerant_for_decisions':   self.use_tolerant_for_decisions,
            # Loss histories
            'train_losses':                 self.train_losses,
            'train_ce_losses':              self.train_ce_losses,
            'train_isw_losses':             self.train_isw_losses,
            'val_losses':                   self.val_losses,
            'val_isw_losses':               self.val_isw_losses,
            # Metric histories
            'train_metrics_history':        self.train_metrics_history,
            'val_metrics_history':          self.val_metrics_history,
            'args':                         vars(self.args),
        }

        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')

        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(
                self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1

        self.best_decision_miou         = ckpt.get('best_decision_miou', 0.0)
        self.best_strict_miou           = ckpt.get('best_strict_miou',   0.0)
        self.best_shadow_iou            = ckpt.get('best_shadow_iou',    0.0)
        self.best_f1                    = ckpt.get('best_f1',            0.0)
        self.epochs_without_improvement = ckpt.get(
            'epochs_without_improvement', 0)

        self.train_losses     = ckpt.get('train_losses',     [])
        self.train_ce_losses  = ckpt.get('train_ce_losses',  [])
        self.train_isw_losses = ckpt.get('train_isw_losses', [])
        self.val_losses       = ckpt.get('val_losses',       [])
        self.val_isw_losses   = ckpt.get('val_isw_losses',   [])

        self.train_metrics_history = ckpt.get('train_metrics_history', {
            k: [] for k in self.train_metrics_history})
        self.val_metrics_history = ckpt.get('val_metrics_history', {
            k: [] for k in self.val_metrics_history})

        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict per-image')
        print(f'Resumed from epoch {ckpt["epoch"]}')
        print(f'Best decision mIOU ({metric_label}): '
              f'{self.best_decision_miou:.2f}%  '
              f'Epochs w/o improvement: {self.epochs_without_improvement}')

    # ──────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────────

    def train(self):
        print('\n' + '=' * 50)
        print('Starting DINOv3 + ISW training...')
        print('=' * 50)

        patience     = self.args.early_stopping_patience
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_for_decisions
                        else 'Strict per-image mIOU')
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            ep = epoch + 1

            # Epoch-based cosine LR (not metric-gated)
            current_lr = self.scheduler.step(epoch)
            print(f'\nLearning rate: {current_lr:.2e}')

            # ---- Train ----
            train_loss, train_ce, train_isw, train_metrics = \
                self.train_epoch(ep)

            # ---- Validate ----
            val_loss, val_isw, val_metrics, detailed = self.validate(ep)

            # ---- Decision metric ----
            decision_miou = self._get_decision_miou(detailed)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, ep)

            # ---- Best checkpoint ----
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou = decision_miou
                is_best                 = True
                self.epochs_without_improvement = 0
                print(f'*** New best {metric_label}: '
                      f'{self.best_decision_miou:.2f}% ***')
            else:
                self.epochs_without_improvement += 1

            # Reference bests (pooled, for logging only)
            if val_metrics['mIOU'] > self.best_strict_miou:
                self.best_strict_miou = val_metrics['mIOU']
                print(f'New best strict pooled mIOU (ref): '
                      f'{self.best_strict_miou:.2f}%')
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
                print(f'New best Shadow IoU: {self.best_shadow_iou:.2f}%')
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                print(f'New best F1: {self.best_f1:.2f}%')

            self.save_checkpoint(ep, is_best=is_best)
            self.writer.add_scalar('Train/LearningRate', current_lr, ep)

            print('=' * 50)

            # ---- Early stopping ----
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered! No {metric_label} '
                      f'improvement for {patience} epochs.')
                break

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best strict pooled mIOU (ref): {self.best_strict_miou:.2f}%')
        print(f'Best Shadow IoU:               {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                       {self.best_f1:.2f}%')

        print('\nGenerating plots...')
        plot_loss_curves_dinov3_isw(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            train_ce_losses=self.train_ce_losses,
            train_isw_losses=self.train_isw_losses,
            val_isw_losses=self.val_isw_losses,
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png'),
        )

        self.writer.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Test
    # ──────────────────────────────────────────────────────────────────────────

    def test(self):
        print('\n' + '=' * 50)
        print('Testing model (best checkpoint)...')
        print('=' * 50)

        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_ckpt):
            self.load_checkpoint(best_ckpt)
        else:
            print('Warning: best checkpoint not found, using current weights')

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

        metrics  = test_metrics.compute()
        detailed = detailed_eval.compute_metrics()

        print('\n' + '=' * 50)
        print('Pooled Test Results (reference):')
        print('=' * 50)
        for k in ['OA', 'Precision', 'F1', 'BER', 'mIOU', 'Shadow_IOU']:
            print(f'{k:12s}: {metrics[k]:.2f}%')

        print('\n' + '=' * 50)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('=' * 50)
        strict   = detailed['boundary_tolerant']['strict']
        tolerant = detailed['boundary_tolerant'][self.tol_key]
        print(f"Strict   — F1: {strict['f1']:.2f}%   mIOU: {strict['iou']:.2f}%")
        print(f"Tolerant (±{self.args.boundary_tolerance}px) — "
              f"F1: {tolerant['f1']:.2f}%   mIOU: {tolerant['iou']:.2f}%")
        print(f"Pixels excluded by band: {tolerant['pixels_excluded']} "
              f"({tolerant['pct_excluded']:.1f}%)")

        if 'size_stratified' in detailed:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed['size_stratified']:
                    m = detailed['size_stratified'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'size_stratified_tolerant' in detailed:
            print(f'\nSize-Stratified (Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed['size_stratified_tolerant']:
                    m = detailed['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if ('fp_fn_analysis' in detailed
                and 'fp' in detailed['fp_fn_analysis']):
            fp = detailed['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({'standard': metrics, 'detailed': detailed}, f, indent=4)
        print(f'\nResults saved to {results_path}')

        print('\nGenerating best/worst prediction visualisations...')
        save_best_worst_visualizations(
            self.model, self.dataloaders['test'],
            self.device, self.output_dir, num_images=10)

        return metrics


# ──────────────────────────────────────────────────────────────────────────────

def main():
    args    = get_args()
    trainer = TrainerDINOv3ISW(args)
    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()