"""
Training script for DINOv3 Shadow Detection with FDA support
Implements training with hyperparameters from DINOv3 paper + FDA augmentation.

Decision metrics (best checkpoint, early stopping) are based on
per-image mIOU from DetailedEvaluator — never the pooled ShadowMetrics value.

When --eval_boundary_tolerant is set, decisions use per-image TOLERANT mIOU
(boundary band width = --boundary_tolerance px, default 2).
Otherwise decisions use per-image STRICT mIOU.

ShadowMetrics (pooled) is still computed and logged to TensorBoard for
reference, but is NOT used for any decisions.

DetailedEvaluator ALWAYS runs regardless of --eval_boundary_tolerant.
The flag only controls WHICH per-image metric drives decisions.
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

from dinov3_model import DINOv3ShadowDetector
from data.dataset import get_dataloaders
from utils.losses import CrossEntropyLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    save_best_worst_visualizations
)
from utils.evaluation_detailed import DetailedEvaluator

print("="*50)
print("GPU DIAGNOSTICS")
print("="*50)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print("="*50)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train DINOv3 for Shadow Detection with FDA')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                      help='Root directory of dataset (required for single mode)')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=1,
                      help='Number of data loading workers')

    # LOCO and multi-city parameters
    parser.add_argument('--mode', type=str, default='single',
                      choices=['single', 'all', 'loco'],
                      help='Training mode: single city, all cities, or LOCO')
    parser.add_argument('--base_data_root', type=str, default=None,
                      help='Base directory for all/loco modes')
    parser.add_argument('--resolution', type=str, default=None,
                      choices=['highres', 'midres'],
                      help='Resolution for all/loco modes')
    parser.add_argument('--fold_id', type=int, default=None,
                      choices=[0, 1, 2],
                      help='Fold ID for LOCO mode')
    parser.add_argument('--cities', type=str, nargs='+', default=None,
                      help='List of cities for all mode')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes (default: 2)')
    parser.add_argument('--model_name', type=str, default='dinov3_vits16',
                      choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                      help='DINOv3 model variant (default: vits16 for ~22M params)')
    parser.add_argument('--weights_path', type=str, default=None,
                      help='Path to DINOv3 pretrained weights .pth file (if not using torch.hub)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained DINOv3 weights')
    parser.add_argument('--frozen_stages', type=int, default=-1,
                      help='Number of backbone stages to freeze (-1 = train all)')

    # Training parameters (DINOv3 paper recommendations)
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-5,
                      help='Learning rate (default: 5e-5 for ViT fine-tuning)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay (default: 0.05 for ViT)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of warmup epochs (default: 5)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='Minimum learning rate for cosine schedule')

    # FDA parameters
    parser.add_argument('--use_fda', action='store_true',
                      help='Apply FDA (Fourier Domain Adaptation)')
    parser.add_argument('--fda_target_root', type=str, default=None,
                      help='Path to target domain images for FDA')
    parser.add_argument('--fda_L', type=float, default=0.01,
                      help='FDA low-frequency ratio (beta), typically 0.01-0.09')

    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=5,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate the model')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')

    # Boundary tolerant evaluation
    # CHANGED: added --boundary_tolerance (was missing entirely, K was hardcoded to 5)
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Use tolerant mIOU (instead of strict) for all decisions. '
                             'DetailedEvaluator always runs; this only selects the '
                             'decision metric.')
    parser.add_argument('--boundary_tolerance', type=int, default=2,
                        help="Don't-care band half-width in pixels (default: 2). "
                             'Controls DetailedEvaluator for both strict and tolerant '
                             'per-image metrics.')

    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement). 0 to disable.')

    # Comparison / inference dirs (passed from shell but optional)
    parser.add_argument('--comparison_inference_dir', type=str, default=None,
                        help='Directory with comparison method inference results')
    parser.add_argument('--comparison_data_root', type=str, default=None,
                        help='Data root used by comparison methods')

    return parser.parse_args()


class CosineWarmupScheduler:
    """
    Cosine learning rate schedule with warmup.
    Standard for ViT fine-tuning.
    NOTE: This scheduler is purely epoch-based (not metric-gated).
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, epoch):
        """Update learning rate"""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class Trainer:
    """Trainer class for DINOv3 Shadow Detection with FDA"""

    def __init__(self, args):
        self.args = args

        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # CHANGED: dynamic tolerant key — no more hardcoded 'tolerant_5px'
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # Create output directory
        if args.mode == 'single':
            city = args.data_root.rstrip('/').split("/")[-2]
            res = args.data_root.rstrip('/').split("/")[-1]
            fda_suffix = '_fda' if args.use_fda else ''
            exp_name = f'dinov3{fda_suffix}_{city}_{res}_{1}'
        elif args.mode == 'all':
            fda_suffix = '_fda' if args.use_fda else ''
            exp_name = f'dinov3{fda_suffix}_all_{args.resolution}_{1}'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            fda_suffix = '_fda' if args.use_fda else ''
            exp_name = f'dinov3{fda_suffix}_loco_holdout_{test_city}_{args.resolution}_{1}'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Initialize model
        print('Initializing DINOv3 model...')
        self.model = DINOv3ShadowDetector(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages
        ).to(self.device)

        # Loss function — single CrossEntropyLoss (no aux branches in DINOv3)
        self.criterion = CrossEntropyLoss()

        # Optimizer (AdamW as per DINOv3 paper)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

        # LR scheduler — epoch-based cosine warmup (not metric-gated)
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr
        )

        # ------------------------------------------------------------------
        # Decision metric:
        #   eval_boundary_tolerant=True  → per-image TOLERANT mIOU
        #   eval_boundary_tolerant=False → per-image STRICT  mIOU
        # Both come from DetailedEvaluator (per-image mean), NOT ShadowMetrics.
        # ------------------------------------------------------------------
        self.use_tolerant_for_decisions = args.eval_boundary_tolerant
        if self.use_tolerant_for_decisions:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU '
                  f'(DetailedEvaluator, not pooled ShadowMetrics)')

        # CHANGED: DetailedEvaluator ALWAYS instantiated — not guarded by flag.
        # boundary_tolerance is now configurable from CLI.
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # Tracking variables
        self.start_epoch = 0
        # CHANGED: renamed best_miou → best_decision_miou (drives checkpoint/early-stop)
        self.best_decision_miou = 0.0
        # Reference-only bests (pooled ShadowMetrics, for logging)
        self.best_strict_miou   = 0.0   # pooled strict mIOU
        self.best_shadow_iou    = 0.0
        self.best_f1            = 0.0
        self.epochs_without_improvement = 0

        # Loss history — DINOv3 has only one CE loss (no aux branches)
        self.train_losses = []   # CE loss per epoch
        self.val_losses   = []   # CE loss per epoch (eval mode)

        # Metric history (ShadowMetrics pooled, reference only)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # Load datasets
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
            fda_L=args.fda_L
        )

        print(f'Training samples:   {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples:       {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    # Decision metric helper
    # CHANGED: new method — mirrors MAMNet._get_decision_miou exactly.
    # Both paths return per-image mean from DetailedEvaluator, never pooled.
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """
        Return the mIOU that drives all decisions (best checkpoint, early stopping).

        Both options are per-image means from DetailedEvaluator, never the
        pooled ShadowMetrics value.

        Args:
            detailed_results: dict from DetailedEvaluator.compute_metrics()

        Returns:
            float mIOU (%)
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
        Train for one epoch.

        Returns
        -------
        epoch_loss : float   — average CE loss
        metrics    : dict    — ShadowMetrics pooled (reference only)
        """
        self.model.train()

        epoch_loss    = 0.0
        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            outputs = self.model(images)
            loss    = self.criterion(outputs, masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Filtered predictions — consistent with val/test
            filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
            train_metrics.update(filtered_outputs, masks)

            # CHANGED: DetailedEvaluator update is now ALWAYS done (no if-guard).
            preds = torch.argmax(filtered_outputs, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | Loss: {loss.item():.4f}')

        epoch_loss /= num_batches
        metrics     = train_metrics.compute()
        epoch_time  = time.time() - start_time

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — loss + pooled metrics (reference)
        self.writer.add_scalar('Train/Loss',        epoch_loss,        epoch)
        self.writer.add_scalar('Train/OA',          metrics['OA'],     epoch)
        self.writer.add_scalar('Train/Precision',   metrics['Precision'], epoch)
        self.writer.add_scalar('Train/F1',          metrics['F1'],     epoch)
        self.writer.add_scalar('Train/BER',         metrics['BER'],    epoch)
        self.writer.add_scalar('Train/mIOU_pooled', metrics['mIOU'],   epoch)
        self.writer.add_scalar('Train/Shadow_IOU',  metrics['Shadow_IOU'], epoch)

        self.train_losses.append(epoch_loss)
        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

        # CHANGED: DetailedEvaluator always computed; use self.tol_key not 'tolerant_5px'
        detailed_results = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict_tr   = detailed_results['boundary_tolerant']['strict']
        tolerant_tr = detailed_results['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Train/mIOU_strict_perimage',   strict_tr['iou'],   epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     strict_tr['f1'],    epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', tolerant_tr['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   tolerant_tr['f1'],  epoch)

        print(f'Per-image Strict:   F1={strict_tr["f1"]:.2f}%  '
              f'mIOU={strict_tr["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant_tr["f1"]:.2f}%  mIOU={tolerant_tr["iou"]:.2f}%')

        return epoch_loss, metrics

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        """
        Validate the model.

        Returns
        -------
        val_loss         : float
        metrics          : dict   — ShadowMetrics pooled (reference only)
        detailed_results : dict   — DetailedEvaluator per-image metrics
                                    (ALWAYS populated; used for decisions)
        """
        print('\nValidating...')
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs = self.model(images)
                loss    = self.criterion(outputs, masks)
                val_loss += loss.item()

                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                # CHANGED: always update (no if-guard), use filtered_outputs consistently
                preds = torch.argmax(filtered_outputs, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics   = val_metrics.compute()

        print(f'Validation Results:')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — loss + pooled metrics (reference)
        self.writer.add_scalar('Val/Loss',          val_loss,           epoch)
        self.writer.add_scalar('Val/OA',            metrics['OA'],      epoch)
        self.writer.add_scalar('Val/Precision',     metrics['Precision'], epoch)
        self.writer.add_scalar('Val/F1',            metrics['F1'],      epoch)
        self.writer.add_scalar('Val/BER',           metrics['BER'],     epoch)
        self.writer.add_scalar('Val/mIOU_pooled',   metrics['mIOU'],    epoch)
        self.writer.add_scalar('Val/Shadow_IOU',    metrics['Shadow_IOU'], epoch)

        self.val_losses.append(val_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

        # CHANGED: DetailedEvaluator always computed; return detailed_results
        # (caller calls _get_decision_miou on it — no longer returning a scalar here)
        detailed_results = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict_val   = detailed_results['boundary_tolerant']['strict']
        tolerant_val = detailed_results['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Val/mIOU_strict_perimage',   strict_val['iou'],   epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     strict_val['f1'],    epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', tolerant_val['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   tolerant_val['f1'],  epoch)

        print(f'Per-image Strict:   F1={strict_val["f1"]:.2f}%  '
              f'mIOU={strict_val["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant_val["f1"]:.2f}%  mIOU={tolerant_val["iou"]:.2f}%')

        return val_loss, metrics, detailed_results

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch':                        epoch,
            'model_state_dict':             self.model.state_dict(),
            'optimizer_state_dict':         self.optimizer.state_dict(),
            # CHANGED: best_decision_miou replaces best_miou as the primary field
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

        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')

        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1

        # CHANGED: load best_decision_miou; fall back to old 'best_miou' key for compat
        self.best_decision_miou = checkpoint.get(
            'best_decision_miou', checkpoint.get('best_miou', 0.0))
        self.best_strict_miou   = checkpoint.get('best_strict_miou', 0.0)
        self.best_shadow_iou    = checkpoint.get('best_shadow_iou',  0.0)
        self.best_f1            = checkpoint.get('best_f1',           0.0)
        self.epochs_without_improvement = checkpoint.get(
            'epochs_without_improvement', 0)

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses   = checkpoint.get('val_losses',   [])
        self.train_metrics_history = checkpoint.get('train_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        self.val_metrics_history = checkpoint.get('val_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })

        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict per-image')
        print(f'Resumed from epoch {checkpoint["epoch"]}')
        print(f'Best decision mIOU ({metric_label}): {self.best_decision_miou:.2f}%  '
              f'Best strict pooled: {self.best_strict_miou:.2f}%  '
              f'Epochs w/o improvement: {self.epochs_without_improvement}')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """Main training loop"""
        print('\n' + '='*50)
        print('Starting training...')
        print('='*50)

        patience = self.args.early_stopping_patience
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_for_decisions else 'Strict per-image mIOU')
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            # Update learning rate (epoch-based cosine schedule, not metric-gated)
            current_lr = self.scheduler.step(epoch)
            print(f'\nLearning rate: {current_lr:.2e}')

            # --- Train ---
            train_loss, train_metrics = self.train_epoch(epoch + 1)

            # --- Validate — returns detailed_results (not scalar decision_miou) ---
            val_loss, val_metrics, detailed_results = self.validate(epoch + 1)

            # --- Decision metric (per-image from DetailedEvaluator) ---
            # CHANGED: decision now comes from _get_decision_miou(detailed_results)
            # rather than being computed inside validate()
            decision_miou = self._get_decision_miou(detailed_results)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            # --- Best checkpoint ---
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'*** New best {metric_label}: {self.best_decision_miou:.2f}% ***')
            else:
                self.epochs_without_improvement += 1

            # Track pooled bests for reference logging only
            if val_metrics['mIOU'] > self.best_strict_miou:
                self.best_strict_miou = val_metrics['mIOU']
                print(f'New best strict pooled mIOU (reference): {self.best_strict_miou:.2f}%')
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
                print(f'New best Shadow IoU: {self.best_shadow_iou:.2f}%')
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']
                print(f'New best F1: {self.best_f1:.2f}%')

            self.save_checkpoint(epoch + 1, is_best=is_best)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)

            print('='*50)

            # --- Early stopping ---
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered! No {metric_label} improvement for '
                      f'{patience} epochs.')
                break

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best strict pooled mIOU (reference): {self.best_strict_miou:.2f}%')
        print(f'Best Shadow IoU:                      {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                              {self.best_f1:.2f}%')

        print('\nGenerating plots...')
        # DINOv3 has a single CE loss — no train_main_losses / train_aux_losses to pass.
        # plot_loss_curves will show a clean overview panel (train + val).
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            train_main_losses=self.train_losses,   # same as total; gives individual panel
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png')
        )

        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        """Test the model using the best checkpoint"""
        print('\n' + '='*50)
        print('Testing model...')
        print('='*50)

        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        else:
            print('Warning: Best checkpoint not found, using current model weights')

        self.model.eval()
        test_metrics = ShadowMetrics()

        # CHANGED: DetailedEvaluator always instantiated in test() — no if-guard.
        # Uses boundary_tolerance from args, consistent with training.
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs          = self.model(images)
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)

                # CHANGED: both evaluators use filtered_outputs — consistent with training
                test_metrics.update(filtered_outputs, masks)
                preds = torch.argmax(filtered_outputs, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics          = test_metrics.compute()
        detailed_results = detailed_eval.compute_metrics()

        print('\n' + '='*50)
        print('Pooled Test Results (reference):')
        print('='*50)
        print(f'OA:         {metrics["OA"]:.2f}%')
        print(f'Precision:  {metrics["Precision"]:.2f}%')
        print(f'F1:         {metrics["F1"]:.2f}%')
        print(f'BER:        {metrics["BER"]:.2f}%')
        print(f'mIOU:       {metrics["mIOU"]:.2f}%')
        print(f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # CHANGED: use self.tol_key instead of hardcoded 'tolerant_5px'
        print('\n' + '='*50)
        print('Per-Image Test Results (DetailedEvaluator):')
        print('='*50)
        strict   = detailed_results['boundary_tolerant']['strict']
        tolerant = detailed_results['boundary_tolerant'][self.tol_key]
        print(f"Strict   — F1: {strict['f1']:.2f}%   mIOU: {strict['iou']:.2f}%")
        print(f"Tolerant (±{self.args.boundary_tolerance}px) — "
              f"F1: {tolerant['f1']:.2f}%   mIOU: {tolerant['iou']:.2f}%")
        print(f"Pixels excluded by band: {tolerant['pixels_excluded']} "
              f"({tolerant['pct_excluded']:.1f}%)")

        if 'size_stratified' in detailed_results:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed_results['size_stratified']:
                    m = detailed_results['size_stratified'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'size_stratified_tolerant' in detailed_results:
            print(f'\nSize-Stratified (Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed_results['size_stratified_tolerant']:
                    m = detailed_results['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if ('fp_fn_analysis' in detailed_results
                and 'fp' in detailed_results['fp_fn_analysis']):
            fp = detailed_results['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        results_path = os.path.join(self.output_dir, 'test_results.json')
        results_to_save = {
            'standard': metrics,
            'detailed': detailed_results,
        }
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f'\nResults saved to {results_path}')

        print('\nGenerating best/worst predictions visualizations...')
        save_best_worst_visualizations(
            self.model,
            self.dataloaders['test'],
            self.device,
            self.output_dir,
            num_images=10
        )

        return metrics


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