"""
Training script for MAMNet
Implements the training procedure described in the paper.

Decision metrics (LR scheduler, best checkpoint, early stopping) use
**per-image** mIOU from DetailedEvaluator — never pooled ShadowMetrics.

When --eval_boundary_tolerant is set, decisions use per-image tolerant mIOU
(boundary band width controlled by --boundary_tolerance, default 2px).
Otherwise decisions use per-image strict mIOU.

ShadowMetrics (pooled) is still computed and logged to TensorBoard for
reference, but is NOT used for any decisions.
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
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet import MAMNet
from data.dataset import get_dataloaders
from data.dataset_enhanced import ShadowDatasetEnhanced
from utils.evaluation_detailed import DetailedEvaluator
from utils.losses import MAMNetLoss
from utils.metrics import ShadowMetrics, evaluate_model
from utils.postprocessing import filter_small_predictions

from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    save_best_worst_visualizations
)

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
    parser = argparse.ArgumentParser(description='Train MAMNet for Shadow Detection')

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
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--aux_weight', type=float, default=0.4)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Checkpoint and logging
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')

    # Device
    parser.add_argument('--device', type=str, default='cuda')

    # FDA
    parser.add_argument('--use_fda', action='store_true')
    parser.add_argument('--fda_target_root', type=str, default=None)
    parser.add_argument('--fda_L', type=float, default=0.01)

    # Contrast channel
    parser.add_argument('--use_contrast', action='store_true')

    # Boundary-tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Use tolerant mIOU (instead of strict) for all decisions')
    parser.add_argument('--boundary_tolerance', type=int, default=2,
                        help='Don\'t-care band half-width in pixels (default: 2). '
                             'Controls DetailedEvaluator for both strict and tolerant '
                             'per-image metrics.')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')

    # Comparison paths
    parser.add_argument('--comparison_inference_dir', type=str, default=None)
    parser.add_argument('--comparison_data_root', type=str, default=None)

    return parser.parse_args()


class Trainer:
    """Trainer class for MAMNet"""

    def __init__(self, args):
        self.args = args

        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Tolerant key — used everywhere instead of hardcoded 'tolerant_5px'
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        modifiers = []
        if args.use_fda:
            modifiers.append('fda')
        modifier_str = ('_' + '_'.join(modifiers)) if modifiers else ''

        if args.mode == 'single':
            city = args.data_root.rstrip('/').split("/")[-2]
            res  = args.data_root.rstrip('/').split("/")[-1]
            exp_name = f'mamnet{modifier_str}_{city}_{res}_{1}'
        elif args.mode == 'all':
            exp_name = f'mamnet{modifier_str}_all_{args.resolution}_{1}'
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'mamnet{modifier_str}_loco_holdout_{test_city}_{args.resolution}_{1}'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Initialize model
        print('Initializing model...')
        self.model = MAMNet(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            use_aux=True,
            use_contrast=args.use_contrast
        ).to(self.device)

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters:     {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

        self.criterion = MAMNetLoss(aux_weight=args.aux_weight)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

        # ---- Decision-metric tracking ----
        # DetailedEvaluator ALWAYS runs (not guarded by eval_boundary_tolerant).
        # eval_boundary_tolerant controls WHICH per-image metric drives decisions:
        #   True  → tolerant mIOU (±boundary_tolerance px band excluded)
        #   False → strict  mIOU  (all pixels, per-image mean)
        # ShadowMetrics (pooled) is logged for reference only.
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU '
                  f'(DetailedEvaluator, not pooled ShadowMetrics)')

        # Always-on DetailedEvaluators
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # Initialize tracking variables
        self.start_epoch             = 0
        self.best_miou               = 0.0   # best strict pooled mIOU (reference only)
        self.best_shadow_iou         = 0.0
        self.best_f1                 = 0.0
        self.best_decision_miou      = 0.0   # drives checkpoint/early-stop
        self.epochs_without_improvement = 0

        # Loss history — total + per-component for rich plotting
        self.train_losses      = []   # total
        self.train_main_losses = []   # main CE
        self.train_aux_losses  = []   # weighted aux (0.0 when no aux)
        self.val_losses        = []   # main CE (model is in eval mode → no aux)

        # Metric history (ShadowMetrics pooled, for reference plots only)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # Load datasets
        if args.use_contrast:
            if args.mode == 'single':
                if args.data_root is None:
                    raise ValueError("data_root must be provided for single city mode")
                train_paths = val_paths = test_paths = [args.data_root]
            elif args.mode == 'all':
                if args.base_data_root is None or args.resolution is None:
                    raise ValueError("base_data_root and resolution required for 'all' mode")
                cities      = args.cities if args.cities else ['chicago', 'miami', 'phoenix']
                train_paths = [os.path.join(args.base_data_root, c, args.resolution)
                               for c in cities]
                val_paths   = test_paths = train_paths
            elif args.mode == 'loco':
                if args.base_data_root is None or args.resolution is None or args.fold_id is None:
                    raise ValueError("base_data_root, resolution and fold_id required for LOCO")
                from data.dataset import LOCO_FOLDS
                fold_config  = LOCO_FOLDS[args.fold_id]
                train_paths  = [os.path.join(args.base_data_root, c, args.resolution)
                                for c in fold_config['train']]
                val_paths    = train_paths
                test_paths   = [os.path.join(args.base_data_root,
                                             fold_config['test'], args.resolution)]
            else:
                raise ValueError(f"Invalid mode: {args.mode}")

            from torch.utils.data import DataLoader

            train_dataset = ShadowDatasetEnhanced(
                root_dir=train_paths, split='train', img_size=args.img_size,
                task_id=2, augment=True,
                use_fda=args.use_fda, fda_target_root=args.fda_target_root,
                fda_L=args.fda_L)
            val_dataset = ShadowDatasetEnhanced(
                root_dir=val_paths, split='val', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)
            test_dataset = ShadowDatasetEnhanced(
                root_dir=test_paths, split='test', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)

            self.dataloaders = {
                'train': DataLoader(train_dataset, batch_size=args.batch_size,
                                    shuffle=True, num_workers=args.num_workers,
                                    pin_memory=True, drop_last=True),
                'val':   DataLoader(val_dataset,   batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True),
                'test':  DataLoader(test_dataset,  batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True),
            }
            print(f'Train: {len(train_dataset)}  Val: {len(val_dataset)}  '
                  f'Test: {len(test_dataset)}')
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
        epoch_loss      : float  — average total loss
        epoch_main_loss : float  — average main CE loss
        epoch_aux_loss  : float  — average weighted aux loss (0.0 if no aux)
        metrics         : dict   — ShadowMetrics pooled (reference only)
        """
        self.model.train()

        epoch_loss      = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss  = 0.0

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            outputs = self.model(images)
            losses  = self.criterion(outputs, masks)
            loss    = losses['total']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Filtered predictions for metrics (consistent with val/test)
            filtered_outputs = filter_small_predictions(outputs['main'], min_pixels=10)
            train_metrics.update(filtered_outputs, masks)

            # DetailedEvaluator — ALWAYS, not guarded by eval_boundary_tolerant
            preds = torch.argmax(filtered_outputs, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss      += loss.item()
            epoch_main_loss += losses['main'].item()
            epoch_aux_loss  += losses.get('aux', torch.tensor(0.0)).item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Main: {losses["main"].item():.4f} | '
                      f'Aux: {losses.get("aux", torch.tensor(0.0)).item():.4f}')

        epoch_loss      /= num_batches
        epoch_main_loss /= num_batches
        epoch_aux_loss  /= num_batches

        metrics    = train_metrics.compute()
        epoch_time = time.time() - start_time

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Total Loss: {epoch_loss:.4f} | '
              f'Main: {epoch_main_loss:.4f} | Aux: {epoch_aux_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Train/Loss',     epoch_loss,      epoch)
        self.writer.add_scalar('Train/MainLoss', epoch_main_loss, epoch)
        self.writer.add_scalar('Train/AuxLoss',  epoch_aux_loss,  epoch)
        # TensorBoard — pooled metrics (reference)
        self.writer.add_scalar('Train/OA',         metrics['OA'],         epoch)
        self.writer.add_scalar('Train/Precision',  metrics['Precision'],  epoch)
        self.writer.add_scalar('Train/F1',         metrics['F1'],         epoch)
        self.writer.add_scalar('Train/BER',        metrics['BER'],        epoch)
        self.writer.add_scalar('Train/mIOU_pooled',metrics['mIOU'],       epoch)
        self.writer.add_scalar('Train/Shadow_IOU', metrics['Shadow_IOU'], epoch)

        # Store pooled metrics for reference plots
        self.train_losses.append(epoch_loss)
        self.train_main_losses.append(epoch_main_loss)
        self.train_aux_losses.append(epoch_aux_loss)
        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

        # DetailedEvaluator — per-image metrics (logged, not used for decisions)
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

        return epoch_loss, epoch_main_loss, epoch_aux_loss, metrics

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

                # model returns raw logits tensor [B, 2, H, W] in eval mode
                outputs = self.model(images)

                # Val loss = main CE only (model eval mode, no aux branches)
                loss      = self.criterion.criterion(outputs, masks)
                val_loss += loss.item()

                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                # DetailedEvaluator — ALWAYS, consistent with filtered preds
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
        self.writer.add_scalar('Val/Loss',          val_loss,          epoch)
        self.writer.add_scalar('Val/OA',            metrics['OA'],     epoch)
        self.writer.add_scalar('Val/Precision',     metrics['Precision'], epoch)
        self.writer.add_scalar('Val/F1',            metrics['F1'],     epoch)
        self.writer.add_scalar('Val/BER',           metrics['BER'],    epoch)
        self.writer.add_scalar('Val/mIOU_pooled',   metrics['mIOU'],   epoch)
        self.writer.add_scalar('Val/Shadow_IOU',    metrics['Shadow_IOU'], epoch)

        self.val_losses.append(val_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

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

        return val_loss, metrics, detailed_results

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch':                    epoch,
            'model_state_dict':         self.model.state_dict(),
            'optimizer_state_dict':     self.optimizer.state_dict(),
            'scheduler_state_dict':     self.scheduler.state_dict(),
            'best_miou':                self.best_miou,
            'best_shadow_iou':          self.best_shadow_iou,
            'best_f1':                  self.best_f1,
            'best_decision_miou':       self.best_decision_miou,
            'epochs_without_improvement': self.epochs_without_improvement,
            # Loss histories (total + components)
            'train_losses':             self.train_losses,
            'train_main_losses':        self.train_main_losses,
            'train_aux_losses':         self.train_aux_losses,
            'val_losses':               self.val_losses,
            # Metric histories
            'train_metrics_history':    self.train_metrics_history,
            'val_metrics_history':      self.val_metrics_history,
            'args':                     vars(self.args),
        }

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

        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e) and 'conv1.weight' in str(e):
                print("WARNING: Checkpoint has different input channels. "
                      "Attempting partial load...")
                state_dict = checkpoint['model_state_dict']
                model_dict = self.model.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items()
                                   if k in model_dict and v.size() == model_dict[k].size()}
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                print(f"Loaded {len(pretrained_dict)}/{len(state_dict)} layers")
            else:
                raise e

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch                = checkpoint['epoch'] + 1
        self.best_miou                  = checkpoint.get('best_miou', 0.0)
        self.best_shadow_iou            = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1                    = checkpoint.get('best_f1', 0.0)
        self.best_decision_miou         = checkpoint.get('best_decision_miou', 0.0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        # Loss histories — backward-compat defaults
        self.train_losses      = checkpoint.get('train_losses', [])
        self.train_main_losses = checkpoint.get('train_main_losses', [])
        self.train_aux_losses  = checkpoint.get('train_aux_losses', [])
        self.val_losses        = checkpoint.get('val_losses', [])

        self.train_metrics_history = checkpoint.get('train_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        self.val_metrics_history = checkpoint.get('val_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })

        print(f'Resumed from epoch {checkpoint["epoch"]}')
        print(f'Best decision mIOU: {self.best_decision_miou:.2f}%  '
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
        if patience > 0:
            decision_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                              if self.use_tolerant_decision else 'Strict per-image mIOU')
            print(f'Early stopping: patience={patience}  metric={decision_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            # --- Train ---
            train_loss, train_main_loss, train_aux_loss, train_metrics = \
                self.train_epoch(epoch + 1)

            # --- Validate (always returns detailed_results) ---
            val_loss, val_metrics, detailed_results = self.validate(epoch + 1)

            # --- Decision metric (per-image from DetailedEvaluator) ---
            decision_miou  = self._get_decision_miou(detailed_results)
            metric_label   = (f'Tolerant ({self.tol_key}) mIOU'
                              if self.use_tolerant_decision else 'Strict per-image mIOU')

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
                print(f'\nEarly stopping after {patience} epochs '
                      f'without improvement in {metric_label}.')
                break

            print('='*50)

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (reference): {self.best_miou:.2f}%')
        print(f'Best Shadow IoU:              {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                      {self.best_f1:.2f}%')

        print('\nGenerating plots...')
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            train_main_losses=self.train_main_losses,
            train_aux_losses=self.train_aux_losses,
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
        # DetailedEvaluator always instantiated (not guarded by flag)
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs          = self.model(images)
                filtered_outputs = filter_small_predictions(outputs, min_pixels=10)

                test_metrics.update(filtered_outputs, masks)

                # Use filtered_outputs consistently for DetailedEvaluator
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

        # Size-stratified (strict)
        if 'size_stratified' in detailed_results:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed_results['size_stratified']:
                    m = detailed_results['size_stratified'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        # Size-stratified (tolerant)
        if 'size_stratified_tolerant' in detailed_results:
            print(f'\nSize-Stratified (Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed_results['size_stratified_tolerant']:
                    m = detailed_results['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        # FP analysis
        if ('fp_fn_analysis' in detailed_results
                and 'fp' in detailed_results['fp_fn_analysis']):
            fp = detailed_results['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        # Save results
        results_to_save = {
            'standard': metrics,
            'detailed': detailed_results,
        }
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f'\nResults saved to {results_path}')

        try:
            print('\nGenerating best/worst prediction visualizations...')
            save_best_worst_visualizations(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except (ImportError, Exception) as e:
            print(f'Visualization skipped: {e}')

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