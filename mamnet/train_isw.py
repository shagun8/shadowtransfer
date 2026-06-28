"""
Training script for MAMNet + ISW (Instance Selective Whitening).

Adds RobustNet-style ISW regularisation on top of the base MAMNet training.

Total loss = seg_loss + λ_ISW × ISW_loss

where ISW_loss is the per-image mean instance selective whitening loss
computed from encoder features at feat1, feat2, feat3.

Decision metrics (LR scheduler, best checkpoint, early stopping) use
per-image mIOU from DetailedEvaluator — never pooled ShadowMetrics.

When --eval_boundary_tolerant is set, decisions use per-image tolerant mIOU
(boundary band width controlled by --boundary_tolerance, default 2px).
Otherwise decisions use per-image strict mIOU.
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime

# ── Debug: write to a file directly, bypassing stdout buffering issues ──
_DEBUG_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'train_isw_debug.log')
def _dlog(msg):
    """Write debug message to file AND stdout."""
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(_DEBUG_LOG, 'a') as f:
        f.write(line + '\n')

_dlog('train_isw.py: SCRIPT STARTING')
_dlog(f'  __name__ = {__name__}')
_dlog(f'  sys.argv = {" ".join(sys.argv)}')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2

_dlog('train_isw.py: torch imported')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet import MAMNet
from data.dataset import get_dataloaders
from data.dataset_enhanced import ShadowDatasetEnhanced
from utils.evaluation_detailed import DetailedEvaluator
from utils.losses import MAMNetLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.isw_loss import ISWLoss, EncoderFeatureHooks

_dlog('train_isw.py: all model/data imports done')

from utils.visualization_isw import (
    plot_loss_curves_isw,
    plot_metrics_curves,
    save_best_worst_visualizations,
)

_dlog('train_isw.py: all imports complete')

_dlog("=" * 50)
_dlog("GPU DIAGNOSTICS")
_dlog("=" * 50)
_dlog(f"CUDA available: {torch.cuda.is_available()}")
_dlog(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    _dlog(f"Current CUDA device: {torch.cuda.current_device()}")
    _dlog(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    _dlog(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
_dlog(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
_dlog("=" * 50)


# ──────────────────────────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description='Train MAMNet + ISW')

    # Data
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=1)

    # LOCO / multi-city
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--resolution', type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--cities', type=str, nargs='+', default=None)

    # Model
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--aux_weight', type=float, default=0.4)

    # Training
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-4)

    # Checkpoint & logging
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--eval_only', action='store_true')

    # Device
    p.add_argument('--device', type=str, default='cuda')

    # FDA
    p.add_argument('--use_fda', action='store_true')
    p.add_argument('--fda_target_root', type=str, default=None)
    p.add_argument('--fda_L', type=float, default=0.01)

    # Contrast channel
    p.add_argument('--use_contrast', action='store_true')

    # Boundary-tolerant evaluation
    p.add_argument('--eval_boundary_tolerant', action='store_true')
    p.add_argument('--boundary_tolerance', type=int, default=2)

    # Early stopping
    p.add_argument('--early_stopping_patience', type=int, default=0)

    # Comparison paths
    p.add_argument('--comparison_inference_dir', type=str, default=None)
    p.add_argument('--comparison_data_root', type=str, default=None)

    # ── ISW-specific ──────────────────────────────────────────────
    p.add_argument('--isw_mask_dir', type=str, required=True,
                   help='Directory with precomputed ISW mask .npy files')
    p.add_argument('--isw_lambda', type=float, default=0.6,
                   help='Weight λ for ISW loss (default: 0.6)')
    p.add_argument('--isw_layers', type=str, default='feat1,feat2,feat3',
                   help='Comma-separated encoder layers for ISW')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────

class TrainerISW:
    """Trainer for MAMNet + ISW."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Tolerant key
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # ── Output directory ──────────────────────────────────────
        modifiers = ['isw']
        if args.use_fda:
            modifiers.append('fda')
        modifier_str = '_' + '_'.join(modifiers)

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

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, 'tensorboard'))

        # ── Model ─────────────────────────────────────────────────
        print('Initialising model...')
        self.model = MAMNet(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            use_aux=True,
            use_contrast=args.use_contrast,
        ).to(self.device)

        total_p = sum(p.numel() for p in self.model.parameters())
        train_p = sum(p.numel() for p in self.model.parameters()
                      if p.requires_grad)
        print(f'Total parameters:     {total_p:,}')
        print(f'Trainable parameters: {train_p:,}')

        # ── ISW loss + hooks ──────────────────────────────────────
        isw_layers = args.isw_layers.split(',')
        print(f'ISW layers: {isw_layers}  λ={args.isw_lambda}')

        self.isw_loss_module = ISWLoss(
            mask_dir=args.isw_mask_dir,
            layer_names=isw_layers,
        ).to(self.device)

        self.encoder_hooks = EncoderFeatureHooks(
            self.model, layer_names=isw_layers)

        # ── Losses & optimiser ────────────────────────────────────
        self.criterion = MAMNetLoss(aux_weight=args.aux_weight)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3)

        # ── Decision metric config ────────────────────────────────
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU')

        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # ── Tracking variables ────────────────────────────────────
        self.start_epoch               = 0
        self.best_miou                 = 0.0
        self.best_shadow_iou           = 0.0
        self.best_f1                   = 0.0
        self.best_decision_miou        = 0.0
        self.epochs_without_improvement = 0

        # Loss history
        self.train_losses      = []  # total (seg + isw)
        self.train_main_losses = []  # main CE
        self.train_aux_losses  = []  # weighted aux
        self.train_isw_losses  = []  # λ × isw
        self.val_losses        = []  # val main CE
        self.val_isw_losses    = []  # val ISW (monitoring)

        # Pooled metric history (reference only)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': [],
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [],
            'BER': [], 'mIOU': [], 'Shadow_IOU': [],
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # ── Dataloaders ───────────────────────────────────────────
        self._setup_dataloaders(args)

    # ──────────────────────────────────────────────────────────────
    def _setup_dataloaders(self, args):
        if args.use_contrast:
            if args.mode == 'single':
                if args.data_root is None:
                    raise ValueError("data_root required for single mode")
                train_paths = val_paths = test_paths = [args.data_root]
            elif args.mode == 'all':
                if args.base_data_root is None or args.resolution is None:
                    raise ValueError("base_data_root + resolution required")
                cities = args.cities or ['chicago', 'miami', 'phoenix']
                train_paths = [os.path.join(args.base_data_root, c,
                                            args.resolution)
                               for c in cities]
                val_paths = test_paths = train_paths
            elif args.mode == 'loco':
                if (args.base_data_root is None
                        or args.resolution is None
                        or args.fold_id is None):
                    raise ValueError(
                        "base_data_root, resolution and fold_id required")
                from data.dataset import LOCO_FOLDS
                fold_cfg    = LOCO_FOLDS[args.fold_id]
                train_paths = [os.path.join(args.base_data_root, c,
                                            args.resolution)
                               for c in fold_cfg['train']]
                val_paths   = train_paths
                test_paths  = [os.path.join(args.base_data_root,
                                            fold_cfg['test'],
                                            args.resolution)]
            else:
                raise ValueError(f"Invalid mode: {args.mode}")

            from torch.utils.data import DataLoader
            train_ds = ShadowDatasetEnhanced(
                root_dir=train_paths, split='train', img_size=args.img_size,
                task_id=2, augment=True,
                use_fda=args.use_fda,
                fda_target_root=args.fda_target_root,
                fda_L=args.fda_L)
            val_ds = ShadowDatasetEnhanced(
                root_dir=val_paths, split='val', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)
            test_ds = ShadowDatasetEnhanced(
                root_dir=test_paths, split='test', img_size=args.img_size,
                task_id=2, augment=False, use_fda=False)

            self.dataloaders = {
                'train': DataLoader(train_ds, batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True, drop_last=True),
                'val':   DataLoader(val_ds, batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True),
                'test':  DataLoader(test_ds, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True),
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
                fda_L=getattr(args, 'fda_L', 0.01),
            )
            print(f'Train: {len(self.dataloaders["train"].dataset)}  '
                  f'Val: {len(self.dataloaders["val"].dataset)}  '
                  f'Test: {len(self.dataloaders["test"].dataset)}')

    # ──────────────────────────────────────────────────────────────
    # Decision metric
    # ──────────────────────────────────────────────────────────────

    def _get_decision_miou(self, detailed_results):
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_decision:
            return bt[self.tol_key]['iou']
        else:
            return bt['strict']['iou']

    # ──────────────────────────────────────────────────────────────
    # Train one epoch
    # ──────────────────────────────────────────────────────────────

    def train_epoch(self, epoch):
        self.model.train()

        epoch_loss      = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss  = 0.0
        epoch_isw_loss  = 0.0

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        t0 = time.time()

        for batch_idx, batch in enumerate(self.dataloaders['train']):
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)

            # Forward — hooks capture encoder features automatically
            outputs = self.model(images)

            # Segmentation loss
            seg_losses = self.criterion(outputs, masks)

            # ISW loss from hooked encoder features
            isw_raw = self.isw_loss_module(self.encoder_hooks.features)
            isw_weighted = self.args.isw_lambda * isw_raw

            # Total loss
            loss = seg_losses['total'] + isw_weighted

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics (filtered, consistent with val/test)
            filtered = filter_small_predictions(outputs['main'], min_pixels=10)
            train_metrics.update(filtered, masks)

            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss      += loss.item()
            epoch_main_loss += seg_losses['main'].item()
            epoch_aux_loss  += seg_losses.get(
                'aux', torch.tensor(0.0)).item()
            epoch_isw_loss  += isw_weighted.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Main: {seg_losses["main"].item():.4f} | '
                      f'Aux: {seg_losses.get("aux", torch.tensor(0.0)).item():.4f} | '
                      f'ISW: {isw_weighted.item():.4f}')

        epoch_loss      /= num_batches
        epoch_main_loss /= num_batches
        epoch_aux_loss  /= num_batches
        epoch_isw_loss  /= num_batches

        metrics    = train_metrics.compute()
        epoch_time = time.time() - t0

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Total: {epoch_loss:.4f} | '
              f'Main: {epoch_main_loss:.4f} | Aux: {epoch_aux_loss:.4f} | '
              f'ISW: {epoch_isw_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Train/Loss',      epoch_loss,      epoch)
        self.writer.add_scalar('Train/MainLoss',  epoch_main_loss, epoch)
        self.writer.add_scalar('Train/AuxLoss',   epoch_aux_loss,  epoch)
        self.writer.add_scalar('Train/ISWLoss',   epoch_isw_loss,  epoch)
        # TensorBoard — pooled metrics
        for k in ['OA', 'Precision', 'F1', 'BER', 'Shadow_IOU']:
            self.writer.add_scalar(f'Train/{k}', metrics[k], epoch)
        self.writer.add_scalar('Train/mIOU_pooled', metrics['mIOU'], epoch)

        # Store history
        self.train_losses.append(epoch_loss)
        self.train_main_losses.append(epoch_main_loss)
        self.train_aux_losses.append(epoch_aux_loss)
        self.train_isw_losses.append(epoch_isw_loss)
        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

        # DetailedEvaluator per-image metrics
        detailed = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict   = detailed['boundary_tolerant']['strict']
        tolerant = detailed['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Train/mIOU_strict_perimage',
                               strict['iou'], epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',
                               strict['f1'], epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage',
                               tolerant['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',
                               tolerant['f1'], epoch)

        print(f'Per-image Strict:   F1={strict["f1"]:.2f}%  '
              f'mIOU={strict["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        return epoch_loss, epoch_main_loss, epoch_aux_loss, epoch_isw_loss, metrics

    # ──────────────────────────────────────────────────────────────
    # Validate
    # ──────────────────────────────────────────────────────────────

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

                # Forward — hooks capture features
                outputs = self.model(images)

                # Val CE loss
                loss = self.criterion.criterion(outputs, masks)
                val_loss += loss.item()

                # Val ISW (monitoring only)
                isw_raw       = self.isw_loss_module(
                    self.encoder_hooks.features)
                val_isw_loss += (self.args.isw_lambda * isw_raw).item()

                filtered = filter_small_predictions(outputs, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        n_batches    = len(self.dataloaders['val'])
        val_loss    /= n_batches
        val_isw_loss /= n_batches
        metrics      = val_metrics.compute()

        print(f'Validation Results:')
        print(f'Loss (CE): {val_loss:.4f} | ISW (monitor): {val_isw_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard
        self.writer.add_scalar('Val/Loss',        val_loss,     epoch)
        self.writer.add_scalar('Val/ISWLoss',     val_isw_loss, epoch)
        for k in ['OA', 'Precision', 'F1', 'BER', 'Shadow_IOU']:
            self.writer.add_scalar(f'Val/{k}', metrics[k], epoch)
        self.writer.add_scalar('Val/mIOU_pooled', metrics['mIOU'], epoch)

        self.val_losses.append(val_loss)
        self.val_isw_losses.append(val_isw_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

        # DetailedEvaluator
        detailed = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict   = detailed['boundary_tolerant']['strict']
        tolerant = detailed['boundary_tolerant'][self.tol_key]
        self.writer.add_scalar('Val/mIOU_strict_perimage',
                               strict['iou'], epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',
                               strict['f1'], epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage',
                               tolerant['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',
                               tolerant['f1'], epoch)

        print(f'Per-image Strict:   F1={strict["f1"]:.2f}%  '
              f'mIOU={strict["iou"]:.2f}%')
        print(f'Per-image Tolerant (±{self.args.boundary_tolerance}px): '
              f'F1={tolerant["f1"]:.2f}%  mIOU={tolerant["iou"]:.2f}%')

        return val_loss, val_isw_loss, metrics, detailed

    # ──────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch':                      epoch,
            'model_state_dict':           self.model.state_dict(),
            'optimizer_state_dict':       self.optimizer.state_dict(),
            'scheduler_state_dict':       self.scheduler.state_dict(),
            'best_miou':                  self.best_miou,
            'best_shadow_iou':            self.best_shadow_iou,
            'best_f1':                    self.best_f1,
            'best_decision_miou':         self.best_decision_miou,
            'epochs_without_improvement': self.epochs_without_improvement,
            # Losses
            'train_losses':       self.train_losses,
            'train_main_losses':  self.train_main_losses,
            'train_aux_losses':   self.train_aux_losses,
            'train_isw_losses':   self.train_isw_losses,
            'val_losses':         self.val_losses,
            'val_isw_losses':     self.val_isw_losses,
            # Metrics
            'train_metrics_history': self.train_metrics_history,
            'val_metrics_history':   self.val_metrics_history,
            'args':               vars(self.args),
        }

        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')

        if epoch % self.args.save_freq == 0:
            ep_path = os.path.join(self.output_dir,
                                   f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, ep_path)

    def load_checkpoint(self, path):
        print(f'Loading checkpoint from {path}')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        try:
            self.model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            if 'size mismatch' in str(e) and 'conv1.weight' in str(e):
                print("WARNING: conv1 size mismatch — partial load")
                sd = ckpt['model_state_dict']
                md = self.model.state_dict()
                compatible = {k: v for k, v in sd.items()
                              if k in md and v.shape == md[k].shape}
                md.update(compatible)
                self.model.load_state_dict(md)
                print(f"Loaded {len(compatible)}/{len(sd)} layers")
            else:
                raise

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch                = ckpt['epoch'] + 1
        self.best_miou                  = ckpt.get('best_miou', 0.0)
        self.best_shadow_iou            = ckpt.get('best_shadow_iou', 0.0)
        self.best_f1                    = ckpt.get('best_f1', 0.0)
        self.best_decision_miou         = ckpt.get('best_decision_miou', 0.0)
        self.epochs_without_improvement = ckpt.get(
            'epochs_without_improvement', 0)

        self.train_losses      = ckpt.get('train_losses', [])
        self.train_main_losses = ckpt.get('train_main_losses', [])
        self.train_aux_losses  = ckpt.get('train_aux_losses', [])
        self.train_isw_losses  = ckpt.get('train_isw_losses', [])
        self.val_losses        = ckpt.get('val_losses', [])
        self.val_isw_losses    = ckpt.get('val_isw_losses', [])

        self.train_metrics_history = ckpt.get('train_metrics_history', {
            k: [] for k in self.train_metrics_history})
        self.val_metrics_history = ckpt.get('val_metrics_history', {
            k: [] for k in self.val_metrics_history})

        print(f'Resumed from epoch {ckpt["epoch"]}')
        print(f'Best decision mIOU: {self.best_decision_miou:.2f}%  '
              f'Epochs w/o improvement: {self.epochs_without_improvement}')

    # ──────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────

    def train(self):
        print('\n' + '=' * 50)
        print('Starting training (MAMNet + ISW)...')
        print('=' * 50)

        patience = self.args.early_stopping_patience
        if patience > 0:
            label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                     if self.use_tolerant_decision
                     else 'Strict per-image mIOU')
            print(f'Early stopping: patience={patience}  metric={label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            ep = epoch + 1

            # Train
            (train_loss, train_main, train_aux,
             train_isw, train_metrics) = self.train_epoch(ep)

            # Validate
            (val_loss, val_isw, val_metrics,
             detailed) = self.validate(ep)

            # Decision metric
            decision_miou = self._get_decision_miou(detailed)
            metric_label  = (f'Tolerant ({self.tol_key}) mIOU'
                             if self.use_tolerant_decision
                             else 'Strict per-image mIOU')

            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, ep)

            # Best checkpoint
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou = decision_miou
                is_best = True
                self.epochs_without_improvement = 0
                print(f'>> New best {metric_label}: '
                      f'{self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1

            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']

            self.save_checkpoint(ep, is_best=is_best)

            lr_now = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', lr_now, ep)

            # Early stopping
            if (patience > 0
                    and self.epochs_without_improvement >= patience):
                print(f'\nEarly stopping after {patience} epochs '
                      f'without improvement in {metric_label}.')
                break

            print('=' * 50)

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (ref): {self.best_miou:.2f}%')
        print(f'Best Shadow IoU:        {self.best_shadow_iou:.2f}%')
        print(f'Best F1:                {self.best_f1:.2f}%')

        # ── Plots ─────────────────────────────────────────────────
        print('\nGenerating plots...')
        plot_loss_curves_isw(
            self.train_losses, self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            train_main_losses=self.train_main_losses,
            train_aux_losses=self.train_aux_losses,
            train_isw_losses=self.train_isw_losses,
            val_isw_losses=self.val_isw_losses,
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png'),
        )

        self.writer.close()

    # ──────────────────────────────────────────────────────────────
    # Test
    # ──────────────────────────────────────────────────────────────

    def test(self):
        print('\n' + '=' * 50)
        print('Testing model...')
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
        print(f"Strict   — F1: {strict['f1']:.2f}%   "
              f"mIOU: {strict['iou']:.2f}%")
        print(f"Tolerant (±{self.args.boundary_tolerance}px) — "
              f"F1: {tolerant['f1']:.2f}%   mIOU: {tolerant['iou']:.2f}%")
        print(f"Pixels excluded by band: {tolerant['pixels_excluded']} "
              f"({tolerant['pct_excluded']:.1f}%)")

        # Size-stratified
        if 'size_stratified' in detailed:
            print('\nSize-Stratified (Strict):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed['size_stratified']:
                    m = detailed['size_stratified'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        if 'size_stratified_tolerant' in detailed:
            print(f'\nSize-Stratified '
                  f'(Tolerant ±{self.args.boundary_tolerance}px):')
            for cat in ['tiny', 'small', 'medium', 'large']:
                if cat in detailed['size_stratified_tolerant']:
                    m = detailed['size_stratified_tolerant'][cat]
                    print(f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                          f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)")

        # FP analysis
        if ('fp_fn_analysis' in detailed
                and 'fp' in detailed['fp_fn_analysis']):
            fp = detailed['fp_fn_analysis']['fp']
            print('\nFP Spatial Distribution:')
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        # Save
        results = {'standard': metrics, 'detailed': detailed}
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f'\nResults saved to {results_path}')

        try:
            print('\nGenerating best/worst prediction visualisations...')
            save_best_worst_visualizations(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except Exception as e:
            print(f'Visualisation skipped: {e}')

        return metrics


# ──────────────────────────────────────────────────────────────────

def main():
    _dlog('main() called — parsing args...')
    args    = get_args()
    _dlog(f'Args parsed. mode={args.mode}, isw_mask_dir={args.isw_mask_dir}')
    _dlog('Creating TrainerISW...')
    trainer = TrainerISW(args)
    _dlog('TrainerISW created successfully')

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    _dlog('__main__ block reached — calling main()')
    try:
        main()
        _dlog('main() completed successfully')
    except Exception as e:
        import traceback
        _dlog(f'FATAL EXCEPTION: {e}')
        _dlog(traceback.format_exc())
        raise