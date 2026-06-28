"""
Training script for OGLANet with SegDesic module
Implements geographic domain adaptation through coordinate embeddings.

Based on:
- OGLANet training: Xie et al. (2022)
- SegDesic adaptation: Verma et al. (2025)

Decision metrics (LR scheduler, best checkpoint, early stopping) use
**per-image** mIOU from DetailedEvaluator — never pooled ShadowMetrics.

When --eval_boundary_tolerant is set, decisions use per-image tolerant mIOU
(boundary band width controlled by --boundary_tolerance, default 2px).
Otherwise decisions use per-image strict mIOU from DetailedEvaluator.

DetailedEvaluator ALWAYS runs regardless of --eval_boundary_tolerant.
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

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_segdesic import OGLANetSegDesic
from data.dataset import get_dataloaders
from utils.geo_losses import SegDesicLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization_segdesic import plot_loss_curves, plot_metrics_curves, save_best_worst_visualizations
from data.dataset_enhanced import ShadowDatasetEnhanced
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
    parser = argparse.ArgumentParser(description='Train OGLANet with SegDesic for Shadow Detection')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                      help='Root directory of dataset (required for single mode)')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size (default: 4, reduced for SegDesic)')
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

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                      help='Learning rate (default: 0.0003 for SegDesic)')
    parser.add_argument('--optimizer', type=str, default='adamax',
                      choices=['adam', 'adamax'],
                      help='Optimizer (default: adamax)')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                      help='Stop after N epochs without improvement (default: disabled)')

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
                    help='Use tolerant mIOU (instead of strict per-image mIOU) for all '
                         'decisions. DetailedEvaluator ALWAYS runs; this flag only '
                         'controls which per-image metric drives LR scheduler / '
                         'checkpointing / early stopping.')
    parser.add_argument('--boundary_tolerance', type=int, default=2,
                    help="Don't-care band half-width in pixels (default: 2). "
                         'Controls DetailedEvaluator for both strict and tolerant '
                         'per-image metrics. Passed as K in ±K px.')

    # Comparison paths (for detailed evaluation against other methods)
    parser.add_argument('--comparison_inference_dir', type=str, default=None,
                      help='Directory with inference results from other methods')
    parser.add_argument('--comparison_data_root', type=str, default=None,
                      help='Data root for comparison evaluation')

    return parser.parse_args()


class SegDesicTrainer:
    """Trainer class for OGLANet with SegDesic"""

    def __init__(self, args):
        self.args = args

        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Tolerant key — used everywhere instead of hardcoded 'tolerant_5px'
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # Create output directory
        if args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'oglanet_segdesic_loco_holdout_{test_city}_{args.resolution}_{1}'
        else:
            exp_name = f'oglanet_segdesic_{args.mode}_{1}'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Initialize model
        print('Initializing OGLANet with SegDesic...')
        self.model = OGLANetSegDesic(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            segdesic_hidden_dim=args.segdesic_hidden_dim,
            segdesic_num_scales=args.segdesic_num_scales,
            use_contrast=args.use_contrast
        ).to(self.device)

        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters:     {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

        # Setup loss function
        self.criterion = SegDesicLoss(aux_weight=0.0, alpha=args.alpha)

        # Setup optimizer (Adamax as per OGLANet paper)
        if args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(self.model.parameters(), lr=args.lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # LR scheduler — driven by decision metric (per-image, never pooled)
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

        # Always-on DetailedEvaluators — one per split
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val   = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # Initialize tracking variables
        self.start_epoch                = 0
        self.best_decision_miou         = 0.0   # drives checkpoint / early-stop
        self.best_miou                  = 0.0   # pooled, reference only
        self.best_shadow_iou            = 0.0   # pooled, reference only
        self.best_f1                    = 0.0   # pooled, reference only
        self.epochs_without_improvement = 0

        # Loss histories — total + per-component for rich plotting
        self.train_losses              = []   # total
        self.train_seg_loss_history    = []   # segmentation (6-output deep supervision avg)
        self.train_domain_src_history  = []   # geo domain loss, source
        self.train_domain_tgt_history  = []   # geo domain loss, target (0.0 if no target)
        self.val_losses                = []   # single CE on P6

        # Metric histories (pooled ShadowMetrics, reference only)
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

            if args.mode == 'loco':
                from data.dataset import LOCO_FOLDS
                fold_config  = LOCO_FOLDS[args.fold_id]
                train_cities = fold_config['train']
                test_city    = fold_config['test']
                train_paths  = [os.path.join(args.base_data_root, city, args.resolution) for city in train_cities]
                val_paths    = train_paths
                test_paths   = [os.path.join(args.base_data_root, test_city, args.resolution)]
            elif args.mode == 'single':
                train_paths = [args.data_root]
                val_paths   = [args.data_root]
                test_paths  = [args.data_root]
            else:  # all
                cities      = args.cities or ['chicago', 'miami', 'phoenix']
                train_paths = [os.path.join(args.base_data_root, city, args.resolution) for city in cities]
                val_paths   = train_paths
                test_paths  = train_paths

            train_dataset = ShadowDatasetEnhanced(
                root_dirs=train_paths, split='train', img_size=args.img_size,
                augment=True,  geo_metadata_path=args.geo_metadata, use_contrast=True)
            val_dataset   = ShadowDatasetEnhanced(
                root_dirs=val_paths,   split='val',   img_size=args.img_size,
                augment=False, geo_metadata_path=args.geo_metadata, use_contrast=True)
            test_dataset  = ShadowDatasetEnhanced(
                root_dirs=test_paths,  split='test',  img_size=args.img_size,
                augment=False, geo_metadata_path=args.geo_metadata, use_contrast=True)

            self.dataloaders = {
                'train': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True, drop_last=True),
                'val':   DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True),
                'test':  DataLoader(test_dataset,  batch_size=1,               shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True),
            }

            # Target domain loader — LOCO mode only.
            # The held-out test city's train split is used as an unlabeled
            # target domain for the geo domain loss (no segmentation labels
            # consumed).  ShadowDatasetEnhanced returns lat/lon when
            # geo_metadata_path is set, so it works directly as an unlabeled
            # loader.  This was missing from the use_contrast=True branch,
            # causing self.use_target_domain=False and DomTgt=0 every run.
            if args.mode == 'loco':
                print('\nCreating unlabeled target domain dataset for SegDesic UDA...')
                target_dataset = ShadowDatasetEnhanced(
                    root_dirs=test_paths, split='train', img_size=args.img_size,
                    augment=False, geo_metadata_path=args.geo_metadata, use_contrast=True)
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
    # Deep supervision loss (6 OGLANet outputs)
    # ------------------------------------------------------------------

    def compute_oglanet_loss(self, predictions, masks):
        """
        Compute OGLANet's deep supervision loss (6 outputs, equal weight).

        Args:
            predictions: Dict with 'p1' to 'p6'
            masks: Ground truth masks

        Returns:
            Total segmentation loss (scalar tensor)
        """
        criterion = nn.CrossEntropyLoss()
        losses    = [criterion(predictions[f'p{i}'], masks)
                     for i in range(1, 7) if f'p{i}' in predictions]
        return sum(losses) / len(losses)

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        """
        Train for one epoch with optional target-domain geo-adaptation.

        Returns
        -------
        epoch_loss : float  — average total loss
        metrics    : dict   — ShadowMetrics pooled (reference only)
        """
        self.model.train()

        epoch_loss              = 0.0
        epoch_seg_loss          = 0.0
        epoch_domain_loss_src   = 0.0
        epoch_domain_loss_tgt   = 0.0

        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders['train'])

        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start_time = time.time()

        # Create target-domain iterator if available
        target_iter = None
        if self.use_target_domain:
            target_iter = iter(self.dataloaders['target'])

        for batch_idx, batch_source in enumerate(self.dataloaders['train']):
            # ---- Source domain (labeled) ----
            images_src = batch_source['image'].to(self.device)
            masks_src  = batch_source['mask'].to(self.device)
            lat_src    = batch_source.get('lat', None)
            lon_src    = batch_source.get('lon', None)
            if lat_src is not None:
                lat_src = lat_src.to(self.device)
            if lon_src is not None:
                lon_src = lon_src.to(self.device)

            outputs_src = self.model(images_src, lat_src, lon_src)

            seg_loss = self.compute_oglanet_loss(outputs_src, masks_src)

            domain_loss_src = 0.0
            if 'geo' in outputs_src:
                geo = outputs_src['geo']
                cos_sim_src     = torch.nn.functional.cosine_similarity(
                    geo['pred_encoding'], geo['gt_encoding'], dim=1)
                domain_loss_src = (1 - cos_sim_src).mean()

            loss = seg_loss + self.args.alpha * domain_loss_src

            # ---- Target domain (unlabeled, geo loss only) ----
            domain_loss_tgt = 0.0
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

                outputs_tgt = self.model(images_tgt, lat_tgt, lon_tgt)

                if 'geo' in outputs_tgt:
                    geo_tgt         = outputs_tgt['geo']
                    cos_sim_tgt     = torch.nn.functional.cosine_similarity(
                        geo_tgt['pred_encoding'], geo_tgt['gt_encoding'], dim=1)
                    domain_loss_tgt = (1 - cos_sim_tgt).mean()
                    loss            = loss + self.args.alpha * domain_loss_tgt

            # ---- Backward ----
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- Metrics — filtered preds used consistently ----
            filtered_outputs = filter_small_predictions(outputs_src['p6'], min_pixels=10)
            train_metrics.update(filtered_outputs, masks_src)

            # DetailedEvaluator — ALWAYS active (not guarded by eval_boundary_tolerant)
            preds = torch.argmax(filtered_outputs, dim=1)
            self.detailed_evaluator_train.update(preds, masks_src, images_src)

            # ---- Accumulate scalar losses ----
            epoch_loss            += loss.item()
            epoch_seg_loss        += seg_loss.item()
            epoch_domain_loss_src += (domain_loss_src.item()
                                      if not isinstance(domain_loss_src, float)
                                      else domain_loss_src)
            epoch_domain_loss_tgt += (domain_loss_tgt.item()
                                      if not isinstance(domain_loss_tgt, float)
                                      else domain_loss_tgt)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                d_src = domain_loss_src if isinstance(domain_loss_src, float) else domain_loss_src.item()
                d_tgt = domain_loss_tgt if isinstance(domain_loss_tgt, float) else domain_loss_tgt.item()
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | Seg: {seg_loss.item():.4f} | '
                      f'Domain_Src: {d_src:.4f} | Domain_Tgt: {d_tgt:.4f}')

        # Average over batches
        epoch_loss            /= num_batches
        epoch_seg_loss        /= num_batches
        epoch_domain_loss_src /= num_batches
        epoch_domain_loss_tgt /= num_batches

        metrics    = train_metrics.compute()
        epoch_time = time.time() - start_time

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f} | '
              f'Seg: {epoch_seg_loss:.4f} | '
              f'Domain_Src: {epoch_domain_loss_src:.4f} | '
              f'Domain_Tgt: {epoch_domain_loss_tgt:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # TensorBoard — losses
        self.writer.add_scalar('Train/TotalLoss',        epoch_loss,            epoch)
        self.writer.add_scalar('Train/SegLoss',           epoch_seg_loss,        epoch)
        self.writer.add_scalar('Train/DomainLoss_Source', epoch_domain_loss_src, epoch)
        if self.use_target_domain:
            self.writer.add_scalar('Train/DomainLoss_Target', epoch_domain_loss_tgt, epoch)

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
        self.train_seg_loss_history.append(epoch_seg_loss)
        self.train_domain_src_history.append(epoch_domain_loss_src)
        self.train_domain_tgt_history.append(epoch_domain_loss_tgt)
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
        metrics          : dict  — ShadowMetrics pooled (reference only)
        detailed_results : dict  — DetailedEvaluator per-image metrics
                                   (ALWAYS populated; used for all decisions)
        """
        print('\nValidating...')
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        seg_criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Forward pass — no geo coordinates during validation
                outputs = self.model(images)
                p6_output = outputs['p6'] if isinstance(outputs, dict) else outputs

                # Val loss — CE on P6 only
                val_loss += seg_criterion(p6_output, masks).item()

                # Filtered preds — consistent with train and test
                filtered_outputs = filter_small_predictions(p6_output, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                # DetailedEvaluator — ALWAYS active
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

        # TensorBoard — pooled metrics (reference)
        self.writer.add_scalar('Val/Loss',        val_loss,        epoch)
        self.writer.add_scalar('Val/mIOU_pooled', metrics['mIOU'], epoch)
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
            'best_decision_miou':           self.best_decision_miou,
            'best_miou':                    self.best_miou,
            'best_shadow_iou':              self.best_shadow_iou,
            'best_f1':                      self.best_f1,
            'epochs_without_improvement':   self.epochs_without_improvement,
            # Loss histories — total + per-component
            'train_losses':                 self.train_losses,
            'train_seg_loss_history':       self.train_seg_loss_history,
            'train_domain_src_history':     self.train_domain_src_history,
            'train_domain_tgt_history':     self.train_domain_tgt_history,
            'val_losses':                   self.val_losses,
            # Metric histories
            'train_metrics_history':        self.train_metrics_history,
            'val_metrics_history':          self.val_metrics_history,
            'args':                         vars(self.args),
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
        self.train_losses             = checkpoint.get('train_losses', [])
        self.train_seg_loss_history   = checkpoint.get('train_seg_loss_history', [])
        self.train_domain_src_history = checkpoint.get('train_domain_src_history', [])
        self.train_domain_tgt_history = checkpoint.get('train_domain_tgt_history', [])
        self.val_losses               = checkpoint.get('val_losses', [])

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
        print('Starting training with SegDesic...')
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_decision else 'Strict per-image mIOU')
        print(f'Decision metric: {metric_label}')
        if self.args.early_stopping_patience:
            print(f'Early stopping patience: {self.args.early_stopping_patience} epochs')
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
                self.best_decision_miou         = decision_miou
                is_best                         = True
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
                print(f'\nEarly stopping triggered after '
                      f'{self.epochs_without_improvement} epochs without '
                      f'improvement in {metric_label}.')
                break

            print('='*50)

        print('\nGenerating plots...')

        # Build component_losses dict; omit domain_tgt if no target domain
        # (all-zero history is uninteresting and would clutter the plot)
        component_losses = {
            'seg_loss':   self.train_seg_loss_history,
            'domain_src': self.train_domain_src_history,
        }
        if self.use_target_domain:
            component_losses['domain_tgt'] = self.train_domain_tgt_history

        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            component_losses=component_losses
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, 'metrics_curves.png')
        )

        print('\nTraining completed!')
        print(f'Best {metric_label}:        {self.best_decision_miou:.2f}%')
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
        detailed_eval = DetailedEvaluator(boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                outputs   = self.model(images)
                p6_output = outputs['p6'] if isinstance(outputs, dict) else outputs

                # Filtered preds — consistent with train and validate
                filtered_outputs = filter_small_predictions(p6_output, min_pixels=10)
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

        save_best_worst_visualizations(
            self.model,
            self.dataloaders['test'],
            self.device,
            self.output_dir,
            num_images=10
        )

        return metrics


def main():
    args = get_args()
    trainer = SegDesicTrainer(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()