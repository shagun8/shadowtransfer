"""
Training script for OGLANet
Implements the training procedure from the paper.
Follows paper's hyperparameter configuration in Section 3.2.

Decision metrics (LR scheduler, best checkpoint, early stopping) use
**per-image** mIOU from DetailedEvaluator — never pooled ShadowMetrics.

When --eval_boundary_tolerant is set, decisions use per-image tolerant mIOU
(boundary band width controlled by --boundary_tolerance, default 2px).
Otherwise decisions use per-image strict mIOU.

ShadowMetrics (pooled) is still computed and logged to TensorBoard for
reference, but is NOT used for any decisions.

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
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.evaluation_detailed import DetailedEvaluator
import sys; print(">>> Python started", flush=True)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.oglanet import OGLANet
from data.dataset import get_dataloaders
from utils.losses import OGLANetLoss
from utils.metrics import ShadowMetrics, evaluate_model
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    save_best_worst_visualizations
)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train OGLANet for Shadow Detection')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                      help='Root directory of dataset')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=12,
                      help='Batch size (default: 12, as per paper)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')

    # LOCO and multi-city parameters
    parser.add_argument('--mode', type=str, default='single',
                    choices=['single', 'all', 'loco'],
                    help='Training mode')
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
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained ResNet-34 encoder')

    # Training parameters (from paper Section 3.2)
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs (default: 100, as per paper)')
    parser.add_argument('--lr', type=float, default=0.0005,
                      help='Learning rate (default: 0.0005, as per paper)')
    parser.add_argument('--optimizer', type=str, default='adamax',
                      choices=['adam', 'adamax'],
                      help='Optimizer (default: adamax, as per paper)')

    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='Directory to save outputs')
    parser.add_argument('--save_freq', type=int, default=10,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate the model')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use (cuda/cpu)')

    # FDA Module parameters
    parser.add_argument('--use_fda', action='store_true',
                        help='Apply FDA (Fourier Domain Adaptation)')
    parser.add_argument('--fda_target_root', type=str, default=None,
                        help='Path to target domain images for FDA')
    parser.add_argument('--fda_L', type=float, default=0.01,
                        help='FDA low-frequency ratio (beta), typically 0.01-0.09')

    # Contrast channel
    parser.add_argument('--use_contrast', action='store_true',
                        help='Use contrast as 4th input channel')

    # Boundary-tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Use tolerant mIOU (instead of strict) for all decisions. '
                             'DetailedEvaluator ALWAYS runs; this flag only controls '
                             'which per-image metric drives LR scheduler / checkpointing / '
                             'early stopping.')
    parser.add_argument('--boundary_tolerance', type=int, default=2,
                        help="Don't-care band half-width in pixels (default: 2). "
                             'Controls DetailedEvaluator for both strict and tolerant '
                             'per-image metrics.')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping patience (epochs without improvement). '
                             'Uses tolerant mIOU when --eval_boundary_tolerant is set, '
                             'strict per-image mIOU otherwise. 0 or None = disabled.')

    return parser.parse_args()


class Trainer:
    """Trainer class for OGLANet"""

    def __init__(self, args):

        self.args = args
        print(">>> init: device setup", flush=True)
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f">>> init: device={self.device}", flush=True)

        # Tolerant key — used everywhere instead of hardcoded 'tolerant_5px'
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        modifiers = []
        if args.use_fda:
            modifiers.append('fda')
        modifier_str = '_'.join(modifiers)
        if modifier_str:
            modifier_str = '_' + modifier_str

        if args.mode == 'single':
            city = args.data_root.rstrip('/').split("/")[-2]
            res  = args.data_root.rstrip('/').split("/")[-1]
            exp_name = f'oglanet{modifier_str}_{city}_{res}_{1}'
        elif args.mode == 'all':
            exp_name = f'oglanet{modifier_str}_all_{args.resolution}_{1}'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'oglanet{modifier_str}_loco_holdout_{test_city}_{args.resolution}_{1}'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f">>> init: output_dir created: {self.output_dir}", flush=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        print(">>> init: creating SummaryWriter", flush=True)
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        print(">>> init: SummaryWriter done", flush=True)

        print(">>> init: creating model", flush=True)
        print('Initializing OGLANet model...')
        print('ASSUMPTION: Using ResNet-34 encoder instead of ResNet-101 (paper spec)')
        print('            for fair comparison with MAMNet.')
        self.model = OGLANet(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            use_contrast=args.use_contrast
        ).to(self.device)
        print(">>> init: model done", flush=True)

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters:     {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

        # Setup loss
        self.criterion = OGLANetLoss()

        # Setup optimizer (Adamax as per paper)
        if args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(self.model.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # LR scheduler — driven by decision metric
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # ---- Decision-metric configuration ----
        # DetailedEvaluator ALWAYS runs (not guarded by eval_boundary_tolerant).
        # eval_boundary_tolerant only controls WHICH per-image metric drives decisions:
        #   True  → tolerant mIOU  (±boundary_tolerance px band excluded)
        #   False → strict  mIOU   (all pixels, per-image mean)
        # ShadowMetrics (pooled) is logged for reference only.
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU '
                  f'(DetailedEvaluator, NOT pooled ShadowMetrics)')

        # Always-on DetailedEvaluators
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # Initialize tracking variables
        self.start_epoch                 = 0
        self.best_miou                   = 0.0   # pooled, reference only
        self.best_shadow_iou             = 0.0   # pooled, reference only
        self.best_f1                     = 0.0   # pooled, reference only
        self.best_decision_miou          = 0.0   # drives checkpoint/early-stop
        self.epochs_without_improvement  = 0

        # Loss histories — total + per-component for rich plotting
        self.train_losses           = []   # total
        self.train_loss1_history    = []
        self.train_loss2_history    = []
        self.train_loss3_history    = []
        self.train_loss4_history    = []
        self.train_loss5_history    = []
        self.train_loss6_history    = []
        self.val_losses             = []   # single CE loss on P6 (eval mode)

        # Metric history (ShadowMetrics pooled, for reference plots only)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        print(">>> init: loading data", flush=True)
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
            use_contrast=args.use_contrast
        )
        print(">>> init: data loaded", flush=True)

        print(f'Training samples:   {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples:       {len(self.dataloaders["test"].dataset)}')

        if args.early_stopping_patience is not None and args.early_stopping_patience > 0:
            print(f'>> Early stopping patience: {args.early_stopping_patience} epochs')

    # ------------------------------------------------------------------
    # Decision metric
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """
        Return the mIOU driving all decisions (LR scheduler, best checkpoint,
        early stopping).

        Both options are per-image means from DetailedEvaluator — never the
        pooled ShadowMetrics value.

        Args:
            detailed_results: dict from DetailedEvaluator.compute_metrics()

        Returns:
            float mIOU (%)
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
        Train for one epoch.

        Returns
        -------
        epoch_loss : float  — average total loss
        metrics    : dict   — ShadowMetrics pooled (reference only)
        """
        self.model.train()

        epoch_loss   = 0.0
        epoch_losses = {
            'loss1': 0.0, 'loss2': 0.0, 'loss3': 0.0,
            'loss4': 0.0, 'loss5': 0.0, 'loss6': 0.0
        }

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            predictions = self.model(images)
            losses      = self.criterion(predictions, masks)
            loss        = losses['total']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Filtered predictions — used CONSISTENTLY for both ShadowMetrics
            # and DetailedEvaluator so numbers are comparable
            filtered_predictions = filter_small_predictions(predictions['p6'], min_pixels=10)
            train_metrics.update(filtered_predictions, masks)

            # DetailedEvaluator — ALWAYS active, NOT guarded by eval_boundary_tolerant
            preds = torch.argmax(filtered_predictions, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss += loss.item()
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | Loss: {loss.item():.4f}')

        # Average over batches
        epoch_loss /= num_batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        metrics    = train_metrics.compute()
        epoch_time = time.time() - start_time

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Total Loss: {epoch_loss:.4f}')
        print(f'  loss1={epoch_losses["loss1"]:.4f}  loss2={epoch_losses["loss2"]:.4f}  '
              f'loss3={epoch_losses["loss3"]:.4f}  loss4={epoch_losses["loss4"]:.4f}  '
              f'loss5={epoch_losses["loss5"]:.4f}  loss6={epoch_losses["loss6"]:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Train/TotalLoss', epoch_loss, epoch)
        for key, val in epoch_losses.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)

        # TensorBoard — pooled metrics (reference only)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)

        # DetailedEvaluator — per-image metrics (always computed)
        detailed_results = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict_train   = detailed_results['boundary_tolerant']['strict']
        tolerant_train = detailed_results['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Train/mIOU_strict_perimage',   strict_train['iou'],   epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     strict_train['f1'],    epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', tolerant_train['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   tolerant_train['f1'],  epoch)

        print(f'Per-image Strict:   F1={strict_train["f1"]:.2f}%  '
              f'mIOU={strict_train["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant_train["f1"]:.2f}%  mIOU={tolerant_train["iou"]:.2f}%')

        # Store histories
        self.train_losses.append(epoch_loss)
        self.train_loss1_history.append(epoch_losses['loss1'])
        self.train_loss2_history.append(epoch_losses['loss2'])
        self.train_loss3_history.append(epoch_losses['loss3'])
        self.train_loss4_history.append(epoch_losses['loss4'])
        self.train_loss5_history.append(epoch_losses['loss5'])
        self.train_loss6_history.append(epoch_losses['loss6'])
        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

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
                                    (ALWAYS populated; used for all decisions)
        """
        print('\nValidating...')
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Inference mode: model returns P6 tensor directly
                predictions = self.model(images)

                # Val loss — single CE on P6
                # NOTE: assumes OGLANetLoss has a .criterion attribute (same pattern
                #       as MAMNetLoss). Provide losses.py if this needs changing.
                loss      = self.criterion.criterion(predictions, masks)
                val_loss += loss.item()

                # Filtered predictions — consistent with train and test
                filtered_predictions = filter_small_predictions(predictions, min_pixels=10)
                val_metrics.update(filtered_predictions, masks)

                # DetailedEvaluator — ALWAYS active, uses filtered preds for consistency
                preds = torch.argmax(filtered_predictions, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics   = val_metrics.compute()

        print(f'Validation Results:')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — pooled metrics (reference)
        self.writer.add_scalar('Val/Loss',        val_loss,          epoch)
        self.writer.add_scalar('Val/mIOU_pooled', metrics['mIOU'],   epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)

        # DetailedEvaluator — per-image metrics (drive all decisions)
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

        # Store for plotting
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

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
            'scheduler_state_dict':         self.scheduler.state_dict(),
            'best_miou':                    self.best_miou,
            'best_shadow_iou':              self.best_shadow_iou,
            'best_f1':                      self.best_f1,
            'best_decision_miou':           self.best_decision_miou,
            'epochs_without_improvement':   self.epochs_without_improvement,
            # Loss histories — total + per-component
            'train_losses':                 self.train_losses,
            'train_loss1_history':          self.train_loss1_history,
            'train_loss2_history':          self.train_loss2_history,
            'train_loss3_history':          self.train_loss3_history,
            'train_loss4_history':          self.train_loss4_history,
            'train_loss5_history':          self.train_loss5_history,
            'train_loss6_history':          self.train_loss6_history,
            'val_losses':                   self.val_losses,
            # Metric histories
            'train_metrics_history':        self.train_metrics_history,
            'val_metrics_history':          self.val_metrics_history,
            'args':                         vars(self.args),
        }

        # Always save latest
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
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch                = checkpoint['epoch'] + 1
        self.best_miou                  = checkpoint.get('best_miou', 0.0)
        self.best_shadow_iou            = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1                    = checkpoint.get('best_f1', 0.0)
        self.best_decision_miou         = checkpoint.get('best_decision_miou', 0.0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        # Loss histories — backward-compat defaults
        self.train_losses        = checkpoint.get('train_losses', [])
        self.train_loss1_history = checkpoint.get('train_loss1_history', [])
        self.train_loss2_history = checkpoint.get('train_loss2_history', [])
        self.train_loss3_history = checkpoint.get('train_loss3_history', [])
        self.train_loss4_history = checkpoint.get('train_loss4_history', [])
        self.train_loss5_history = checkpoint.get('train_loss5_history', [])
        self.train_loss6_history = checkpoint.get('train_loss6_history', [])
        self.val_losses          = checkpoint.get('val_losses', [])

        self.train_metrics_history = checkpoint.get('train_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        self.val_metrics_history = checkpoint.get('val_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })

        print(f'Resumed from epoch {checkpoint["epoch"]}')
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_decision else 'Strict per-image mIOU')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%  '
              f'Epochs w/o improvement: {self.epochs_without_improvement}')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Main training loop.

        Decision logic (LR scheduler, best checkpoint, early stopping) is
        driven by the per-image *decision metric* from DetailedEvaluator:
          - Tolerant mIOU  when --eval_boundary_tolerant is set
          - Strict  mIOU   otherwise
        Both strict and tolerant per-image metrics are always logged.
        Pooled ShadowMetrics is logged for reference only.
        """
        print('\n' + '='*50)
        print('Starting training...')
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_decision else 'Strict per-image mIOU')
        print(f'Decision metric: {metric_label}')
        print('='*50)

        patience = (self.args.early_stopping_patience
                    if self.args.early_stopping_patience is not None else 0)

        for epoch in range(self.start_epoch, self.args.epochs):
            # --- Train ---
            train_loss, train_metrics = self.train_epoch(epoch + 1)

            # --- Validate (always returns detailed_results) ---
            val_loss, val_metrics, detailed_results = self.validate(epoch + 1)

            # --- Decision metric (per-image from DetailedEvaluator) ---
            decision_miou = self._get_decision_miou(detailed_results)

            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            # --- Best checkpoint ---
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou      = decision_miou
                is_best                      = True
                self.epochs_without_improvement = 0
                print(f'>> New best {metric_label}: {self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            # Track pooled bests for reference logging only
            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']

            self.save_checkpoint(epoch + 1, is_best=is_best)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)

            # --- Early stopping ---
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered after {self.epochs_without_improvement} '
                      f'epochs without improvement in {metric_label}.')
                break

            print('='*50)

        print('\nGenerating plots...')
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
            }
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png')
        )

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (reference): {self.best_miou:.2f}%')
        print(f'Best Shadow IoU:              {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                      {self.best_f1:.2f}%')

        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        """Test the model using best checkpoint"""
        print('\n' + '='*50)
        print('Testing model...')
        print('='*50)

        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        else:
            print('Warning: Best checkpoint not found, using current model weights')

        self.model.eval()
        test_metrics  = ShadowMetrics()
        # DetailedEvaluator always instantiated
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                predictions      = self.model(images)
                filtered_predictions = filter_small_predictions(predictions, min_pixels=10)

                test_metrics.update(filtered_predictions, masks)

                # Consistent: filtered preds for DetailedEvaluator too
                preds = torch.argmax(filtered_predictions, dim=1)
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

        results_to_save = {
            'standard': metrics,
            'detailed': detailed_results,
        }
        results_path = os.path.join(self.output_dir, 'test_results.json')
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
    print(">>> before get_args", flush=True)
    args = get_args()
    print(f">>> args parsed: mode={args.mode}, fold_id={args.fold_id}", flush=True)
    trainer = Trainer(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()