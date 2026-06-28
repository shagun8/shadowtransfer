"""
Training script for MAMNet with SegDesic module
Implements geographic domain adaptation through coordinate embeddings.

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet_segdesic import MAMNetSegDesic
from data.dataset import get_dataloaders
from data.dataset_enhanced import ShadowDatasetEnhanced
from utils.evaluation_detailed import DetailedEvaluator
from utils.geo_losses import SegDesicLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization_segdesic import plot_loss_curves, plot_metrics_curves, save_best_worst_visualizations

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
    parser = argparse.ArgumentParser(description='Train MAMNet with SegDesic for Shadow Detection')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                        help='Root directory of dataset (required for single mode)')
    parser.add_argument('--img_size', type=int, default=384,
                        help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # LOCO and multi-city parameters
    parser.add_argument('--mode', type=str, default='loco',
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

    # SegDesic parameters
    parser.add_argument('--geo_metadata', type=str, required=True,
                        help='Path to geocoordinate metadata JSON (mapping.json)')
    parser.add_argument('--segdesic_hidden_dim', type=int, default=256,
                        help='Hidden dimension for SegDesic module')
    parser.add_argument('--segdesic_num_scales', type=int, default=10,
                        help='Number of scales for GRID encoding')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for geographic domain loss')

    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ResNet-34 encoder')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                        help='Weight for auxiliary loss')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement, 0=disabled)')

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

    # Contrast channel
    parser.add_argument('--use_contrast', action='store_true',
                        help='Use contrast as 4th input channel')

    # Boundary-tolerant evaluation
    parser.add_argument('--eval_boundary_tolerant', action='store_true',
                        help='Use tolerant mIOU (instead of strict) for all decisions')
    parser.add_argument('--boundary_tolerance', type=int, default=2,
                        help="Don't-care band half-width in pixels (default: 2). "
                             "Controls DetailedEvaluator for both strict and tolerant "
                             "per-image metrics. Pass K from bash to set ±Kpx zone.")

    # Comparison / external inference results (optional)
    parser.add_argument('--comparison_inference_dir', type=str, default=None,
                        help='Directory with comparison method inference results')
    parser.add_argument('--comparison_data_root', type=str, default=None,
                        help='Data root corresponding to comparison inferences')

    return parser.parse_args()


class SegDesicTrainer:
    """Trainer class for MAMNet with SegDesic"""

    def __init__(self, args):
        self.args = args

        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Dynamic tolerant key — used everywhere instead of any hardcoded string
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # Decision metric flag
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU '
                  f'(DetailedEvaluator, not pooled ShadowMetrics)')

        # Create output directory
        if args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'mamnet_segdesic_loco_holdout_{test_city}_{args.resolution}_1'
        else:
            exp_name = f'mamnet_segdesic_{args.mode}_1'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Initialize model
        print('Initializing MAMNet with SegDesic...')
        self.model = MAMNetSegDesic(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            use_aux=True,
            segdesic_hidden_dim=args.segdesic_hidden_dim,
            segdesic_num_scales=args.segdesic_num_scales,
            use_contrast=args.use_contrast
        ).to(self.device)

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters:     {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

        # Setup loss function
        self.criterion = SegDesicLoss(aux_weight=args.aux_weight, alpha=args.alpha)

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Setup learning rate scheduler (driven by per-image decision mIOU)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )

        # Always-on DetailedEvaluators (not guarded by eval_boundary_tolerant).
        # eval_boundary_tolerant only controls WHICH metric drives decisions.
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # Initialize tracking variables
        self.start_epoch              = 0
        self.best_decision_miou       = 0.0   # drives checkpoint/LR/early-stop
        self.best_miou                = 0.0   # pooled, reference only
        self.best_shadow_iou          = 0.0   # pooled, reference only
        self.best_f1                  = 0.0   # pooled, reference only
        self.epochs_without_improvement = 0

        # Loss history — total + all per-component for rich plotting
        self.train_losses            = []   # total
        self.train_main_losses       = []   # main CE
        self.train_aux_losses        = []   # weighted aux
        self.train_domain_src_losses = []   # source domain (raw, pre-alpha)
        self.train_domain_tgt_losses = []   # target domain (raw, pre-alpha)
        self.val_losses              = []   # main CE (eval mode, no aux/domain)

        # Pooled metric history (reference plots only — NOT used for decisions)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # Load datasets WITH geocoordinate metadata
        if args.use_contrast:
            from torch.utils.data import DataLoader

            # Resolve paths and target_paths for all modes
            target_paths = None   # only set for LOCO (UDA target domain)

            if args.mode == 'loco':
                from data.dataset import LOCO_FOLDS
                fold_config  = LOCO_FOLDS[args.fold_id]
                train_paths  = [os.path.join(args.base_data_root, c, args.resolution)
                                for c in fold_config['train']]
                val_paths    = train_paths
                test_paths   = [os.path.join(args.base_data_root,
                                             fold_config['test'], args.resolution)]
                # Target domain = held-out test city (unlabeled, for domain loss)
                target_paths = test_paths
            elif args.mode == 'all':
                cities      = args.cities if args.cities else ['chicago', 'miami', 'phoenix']
                train_paths = [os.path.join(args.base_data_root, c, args.resolution)
                               for c in cities]
                val_paths   = train_paths
                test_paths  = train_paths
            else:  # single
                train_paths = val_paths = test_paths = [args.data_root]

            train_dataset = ShadowDatasetEnhanced(
                root_dir=train_paths, split='train', img_size=args.img_size,
                task_id=2, augment=True, geo_metadata_path=args.geo_metadata)
            val_dataset = ShadowDatasetEnhanced(
                root_dir=val_paths, split='val', img_size=args.img_size,
                task_id=2, augment=False, geo_metadata_path=args.geo_metadata)
            test_dataset = ShadowDatasetEnhanced(
                root_dir=test_paths, split='test', img_size=args.img_size,
                task_id=2, augment=False, geo_metadata_path=args.geo_metadata)

            self.dataloaders = {
                'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True),
                'val':   DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True),
                'test':  DataLoader(test_dataset,  batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True),
            }

            # Add target domain loader for LOCO mode (was missing — caused DomTgt=0 always)
            # ShadowDatasetEnhanced returns lat/lon when geo_metadata_path is set,
            # so it works directly as an unlabeled target loader.
            if target_paths is not None:
                print('\nCreating unlabeled target domain dataset for SegDesic UDA...')
                target_dataset = ShadowDatasetEnhanced(
                    root_dir=target_paths, split='train', img_size=args.img_size,
                    task_id=2, augment=False, geo_metadata_path=args.geo_metadata)
                self.dataloaders['target'] = DataLoader(
                    target_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
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
                geo_metadata_path=args.geo_metadata
            )

        self.use_target_domain = 'target' in self.dataloaders
        if self.use_target_domain:
            print(f'Target domain (unlabeled) samples: {len(self.dataloaders["target"].dataset)}')

        print(f'Training samples:   {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples:       {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    # Decision metric
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """
        Return the mIOU driving all decisions (LR scheduler, best checkpoint,
        early stopping).

        Both options are per-image means from DetailedEvaluator — never the
        pooled ShadowMetrics value.
        """
        bt = detailed_results['boundary_tolerant']
        if self.use_tolerant_decision:
            return bt[self.tol_key]['iou']
        return bt['strict']['iou']

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        """
        Train for one epoch with optional target domain adaptation.

        Returns
        -------
        epoch_loss        : float  total loss average
        epoch_main_loss   : float  main CE average
        epoch_aux_loss    : float  weighted aux average (0.0 if no aux)
        epoch_domain_src  : float  raw source domain loss average (pre-alpha)
        epoch_domain_tgt  : float  raw target domain loss average (pre-alpha)
        metrics           : dict   ShadowMetrics pooled (reference only)
        """
        self.model.train()
        self.detailed_evaluator_train.reset()

        epoch_loss       = 0.0
        epoch_main_loss  = 0.0
        epoch_aux_loss   = 0.0
        epoch_domain_src = 0.0
        epoch_domain_tgt = 0.0

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start_time = time.time()

        # Create iterator for target domain if available
        target_iter = None
        if self.use_target_domain:
            target_iter = iter(self.dataloaders['target'])

        for batch_idx, batch_source in enumerate(self.dataloaders['train']):
            # ---- SOURCE DOMAIN (labeled) ----
            images_src = batch_source['image'].to(self.device)
            masks_src  = batch_source['mask'].to(self.device)
            lat_src    = batch_source.get('lat', None)
            lon_src    = batch_source.get('lon', None)
            if lat_src is not None:
                lat_src = lat_src.to(self.device)
            if lon_src is not None:
                lon_src = lon_src.to(self.device)

            # Forward pass on source
            outputs_src     = self.model(images_src, lat_src, lon_src)
            geo_outputs_src = outputs_src.get('geo', None)
            losses_src      = self.criterion(outputs_src, masks_src, geo_outputs_src)
            loss            = losses_src['total']

            # ---- TARGET DOMAIN (unlabeled) — domain loss only ----
            domain_loss_tgt_item = 0.0
            if self.use_target_domain:
                try:
                    batch_target = next(target_iter)
                except StopIteration:
                    target_iter  = iter(self.dataloaders['target'])
                    batch_target = next(target_iter)

                images_tgt = batch_target['image'].to(self.device)
                lat_tgt    = batch_target.get('lat', None)
                lon_tgt    = batch_target.get('lon', None)
                if lat_tgt is not None:
                    lat_tgt = lat_tgt.to(self.device)
                if lon_tgt is not None:
                    lon_tgt = lon_tgt.to(self.device)

                outputs_tgt     = self.model(images_tgt, lat_tgt, lon_tgt)
                geo_outputs_tgt = outputs_tgt.get('geo', None)

                if geo_outputs_tgt is not None:
                    cos_sim_tgt      = F.cosine_similarity(
                        geo_outputs_tgt['pred_encoding'],
                        geo_outputs_tgt['gt_encoding'],
                        dim=1
                    )
                    domain_loss_tgt      = (1 - cos_sim_tgt).mean()
                    loss                 = loss + self.args.alpha * domain_loss_tgt
                    domain_loss_tgt_item = domain_loss_tgt.item()   # raw, pre-alpha

            # ---- Backward ----
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- Metrics — CONSISTENT: filter THEN update both evaluators ----
            filtered_outputs = filter_small_predictions(outputs_src['main'], min_pixels=10)
            train_metrics.update(filtered_outputs, masks_src)

            preds = torch.argmax(filtered_outputs, dim=1)
            self.detailed_evaluator_train.update(preds, masks_src, images_src)

            # ---- Accumulate losses ----
            epoch_loss       += loss.item()
            epoch_main_loss  += losses_src['seg_main'].item()
            # weighted aux contribution = seg_total - seg_main
            epoch_aux_loss   += (losses_src['seg_total'].item()
                                 - losses_src['seg_main'].item())
            # raw domain losses (pre-alpha) for plotting
            epoch_domain_src += losses_src.get('domain_loss',
                                               torch.tensor(0.0)).item()
            epoch_domain_tgt += domain_loss_tgt_item

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                dom_src_str = (f"DomSrc: {losses_src['domain_loss'].item():.4f}"
                               if 'domain_loss' in losses_src else "DomSrc: 0.0000")
                dom_tgt_str = f"DomTgt: {domain_loss_tgt_item:.4f}"
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Main: {losses_src["seg_main"].item():.4f} | '
                      f'{dom_src_str} | {dom_tgt_str}')

        # Average over batches
        epoch_loss       /= num_batches
        epoch_main_loss  /= num_batches
        epoch_aux_loss   /= num_batches
        epoch_domain_src /= num_batches
        epoch_domain_tgt /= num_batches

        metrics    = train_metrics.compute()
        epoch_time = time.time() - start_time

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Total: {epoch_loss:.4f} | '
              f'Main: {epoch_main_loss:.4f} | Aux: {epoch_aux_loss:.4f} | '
              f'DomSrc: {epoch_domain_src:.4f} | DomTgt: {epoch_domain_tgt:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  F1: {metrics["F1"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # ---- TensorBoard — losses ----
        self.writer.add_scalar('Train/Loss',          epoch_loss,       epoch)
        self.writer.add_scalar('Train/MainLoss',      epoch_main_loss,  epoch)
        self.writer.add_scalar('Train/AuxLoss',       epoch_aux_loss,   epoch)
        self.writer.add_scalar('Train/DomainSrcLoss', epoch_domain_src, epoch)
        self.writer.add_scalar('Train/DomainTgtLoss', epoch_domain_tgt, epoch)

        # ---- TensorBoard — pooled metrics (reference) ----
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)

        # ---- Store for plotting ----
        self.train_losses.append(epoch_loss)
        self.train_main_losses.append(epoch_main_loss)
        self.train_aux_losses.append(epoch_aux_loss)
        self.train_domain_src_losses.append(epoch_domain_src)
        self.train_domain_tgt_losses.append(epoch_domain_tgt)
        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

        # ---- DetailedEvaluator — per-image metrics (logged, train reference) ----
        detailed_results   = self.detailed_evaluator_train.compute_metrics()
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

        return (epoch_loss, epoch_main_loss, epoch_aux_loss,
                epoch_domain_src, epoch_domain_tgt, metrics)

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, epoch):
        """
        Validate the model.

        Returns
        -------
        val_loss         : float
        metrics          : dict   ShadowMetrics pooled (reference only)
        detailed_results : dict   DetailedEvaluator per-image (drives all decisions)
        """
        print('\nValidating...')
        self.model.eval()
        self.detailed_evaluator_val.reset()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Forward pass — no lat/lon at validation time
                outputs = self.model(images)

                # Val loss = main CE only
                loss      = self.criterion.criterion(outputs['main'], masks)
                val_loss += loss.item()

                # CONSISTENT: filter THEN update both evaluators with same predictions
                filtered_outputs = filter_small_predictions(outputs['main'], min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                preds = torch.argmax(filtered_outputs, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics   = val_metrics.compute()

        print(f'Validation Results (pooled reference):')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  F1: {metrics["F1"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — loss + pooled metrics (reference)
        self.writer.add_scalar('Val/Loss',         val_loss,          epoch)
        self.writer.add_scalar('Val/mIOU_pooled',  metrics['mIOU'],   epoch)
        self.writer.add_scalar('Val/F1_pooled',    metrics['F1'],     epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)

        self.val_losses.append(val_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

        # ---- DetailedEvaluator — per-image metrics (drive all decisions) ----
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
            'epoch':                      epoch,
            'model_state_dict':           self.model.state_dict(),
            'optimizer_state_dict':       self.optimizer.state_dict(),
            'scheduler_state_dict':       self.scheduler.state_dict(),
            'best_decision_miou':         self.best_decision_miou,
            'best_miou':                  self.best_miou,
            'best_shadow_iou':            self.best_shadow_iou,
            'best_f1':                    self.best_f1,
            'epochs_without_improvement': self.epochs_without_improvement,
            # Loss histories (total + all components)
            'train_losses':               self.train_losses,
            'train_main_losses':          self.train_main_losses,
            'train_aux_losses':           self.train_aux_losses,
            'train_domain_src_losses':    self.train_domain_src_losses,
            'train_domain_tgt_losses':    self.train_domain_tgt_losses,
            'val_losses':                 self.val_losses,
            # Metric histories
            'train_metrics_history':      self.train_metrics_history,
            'val_metrics_history':        self.val_metrics_history,
            'args':                       vars(self.args),
        }

        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

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
        self.best_decision_miou         = checkpoint.get('best_decision_miou', 0.0)
        self.best_miou                  = checkpoint.get('best_miou', 0.0)
        self.best_shadow_iou            = checkpoint.get('best_shadow_iou', 0.0)
        self.best_f1                    = checkpoint.get('best_f1', 0.0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        # Loss histories — backward-compat defaults
        self.train_losses            = checkpoint.get('train_losses', [])
        self.train_main_losses       = checkpoint.get('train_main_losses', [])
        self.train_aux_losses        = checkpoint.get('train_aux_losses', [])
        self.train_domain_src_losses = checkpoint.get('train_domain_src_losses', [])
        self.train_domain_tgt_losses = checkpoint.get('train_domain_tgt_losses', [])
        self.val_losses              = checkpoint.get('val_losses', [])

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
        print('Starting training with SegDesic...')
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_decision else 'Strict per-image mIOU')
        print(f'Decision metric: {metric_label}')
        patience = self.args.early_stopping_patience
        if patience > 0:
            print(f'Early stopping patience: {patience} epochs')
        print('='*50)

        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            (train_loss, train_main, train_aux,
             domain_src, domain_tgt, train_metrics) = self.train_epoch(epoch + 1)

            # Validate — always returns detailed_results for decision making
            val_loss, val_metrics, detailed_results = self.validate(epoch + 1)

            # ---- Decision metric (per-image from DetailedEvaluator) ----
            decision_miou = self._get_decision_miou(detailed_results)
            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            # ---- Best checkpoint ----
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou         = decision_miou
                is_best                         = True
                self.epochs_without_improvement = 0
                print(f'>> New best {metric_label}: {self.best_decision_miou:.2f}%')
            else:
                self.epochs_without_improvement += 1
                print(f'No improvement for {self.epochs_without_improvement} epoch(s) '
                      f'({metric_label}: {decision_miou:.2f}%  '
                      f'best: {self.best_decision_miou:.2f}%)')

            # Track pooled bests for reference logging only (not for decisions)
            if val_metrics['mIOU'] > self.best_miou:
                self.best_miou = val_metrics['mIOU']
            if val_metrics['Shadow_IOU'] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics['Shadow_IOU']
            if val_metrics['F1'] > self.best_f1:
                self.best_f1 = val_metrics['F1']

            self.save_checkpoint(epoch + 1, is_best=is_best)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch + 1)

            # ---- Early stopping ----
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered after {patience} epochs '
                      f'without improvement in {metric_label}.')
                break

            print('='*50)

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best pooled mIOU (reference): {self.best_miou:.2f}%')
        print(f'Best Shadow IoU (reference):  {self.best_shadow_iou:.2f}%')
        print(f'Best F1 (reference):          {self.best_f1:.2f}%')

        # Generate plots — pass ALL loss components for rich subplots
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            train_main_losses=self.train_main_losses,
            train_aux_losses=self.train_aux_losses,
            train_domain_src_losses=self.train_domain_src_losses,
            train_domain_tgt_losses=self.train_domain_tgt_losses,
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
        detailed_eval = DetailedEvaluator(boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs = self.model(images)

                # CONSISTENT: filter THEN update both evaluators
                filtered_outputs = filter_small_predictions(outputs['main'], min_pixels=10)
                test_metrics.update(filtered_outputs, masks)

                preds = torch.argmax(filtered_outputs, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics          = test_metrics.compute()
        detailed_results = detailed_eval.compute_metrics()

        print('\n' + '='*50)
        print('Pooled Test Results (reference):')
        print('='*50)
        for key, val in metrics.items():
            print(f'{key}: {val:.2f}%')

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
        results_to_save = {'standard': metrics, 'detailed': detailed_results}
        results_path    = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)
        print(f'\nResults saved to {results_path}')

        try:
            print('\nGenerating best/worst prediction visualizations...')
            save_best_worst_visualizations(
                self.model, self.dataloaders['test'],
                self.device, self.output_dir, num_images=10)
        except Exception as e:
            print(f'Visualization skipped: {e}')

        return metrics


def main():
    args    = get_args()
    trainer = SegDesicTrainer(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()