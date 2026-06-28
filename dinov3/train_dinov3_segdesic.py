"""
Training script for DINOv3 with SegDesic module.
Implements geographic domain adaptation through coordinate embeddings.
Supports LOCO cross-city evaluation with unsupervised domain adaptation.

Decision metrics (best checkpoint, early stopping) use per-image mIOU
from DetailedEvaluator — never the pooled ShadowMetrics value.

When --eval_boundary_tolerant is set, decisions use per-image TOLERANT mIOU
(boundary band width = --boundary_tolerance px, default 2).
Otherwise decisions use per-image STRICT mIOU.

ShadowMetrics (pooled) is still computed and logged to TensorBoard for
reference but is NOT used for any decisions.

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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_segdesic import DINOv3SegDesic
from data.dataset import get_dataloaders

# Import utilities from MAMNet (reusable)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mamnet'))
from utils.geo_losses import SegDesicLoss
from utils.metrics import ShadowMetrics
from utils.postprocessing import filter_small_predictions
from utils.visualization_segdesic import plot_loss_curves, plot_metrics_curves, save_best_worst_visualizations
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
    parser = argparse.ArgumentParser(description='Train DINOv3 with SegDesic for Shadow Detection')

    # Data parameters
    parser.add_argument('--data_root', type=str, required=False, default=None,
                      help='Root directory of dataset (required for single mode)')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size (default: 384)')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size (default: 4, smaller due to SegDesic overhead)')
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
    parser.add_argument('--model_name', type=str, default='dinov3_vits16',
                      choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                      help='DINOv3 model variant')
    parser.add_argument('--weights_path', type=str, default=None,
                      help='Path to DINOv3 pretrained weights .pth file')
    parser.add_argument('--pretrained', action='store_true', default=True,
                      help='Use pretrained DINOv3 encoder')
    parser.add_argument('--frozen_stages', type=int, default=-1,
                      help='Number of backbone stages to freeze (-1 = train all)')

    # Training parameters (DINOv3 specific)
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                      help='Learning rate (higher for SegDesic)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                      help='Weight decay (ViT default)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                      help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='Minimum learning rate for cosine schedule')

    # FDA mutual exclusivity check
    parser.add_argument('--use_fda', action='store_true',
                      help='FDA is mutually exclusive with SegDesic (will raise error)')
    parser.add_argument('--fda_target_root', type=str, default=None,
                      help='Not used with SegDesic')
    parser.add_argument('--fda_L', type=float, default=0.01,
                      help='Not used with SegDesic')

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
    # CHANGED: added --boundary_tolerance (was missing; K was hardcoded to 5 in lookups
    #          but the DetailedEvaluator defaulted to 2, causing a KeyError at runtime)
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

    # Comparison / inference directories (passed from shell scripts)
    parser.add_argument('--comparison_inference_dir', type=str, default=None,
                        help='Directory containing comparison inference results')
    parser.add_argument('--comparison_data_root', type=str, default=None,
                        help='Data root for comparison evaluation')

    return parser.parse_args()


class CosineWarmupScheduler:
    """Cosine learning rate schedule with warmup (standard for ViT fine-tuning)"""
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
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class SegDesicTrainer:
    """Trainer class for DINOv3 with SegDesic"""

    def __init__(self, args):
        self.args = args

        # Check FDA mutual exclusivity
        if args.use_fda:
            raise ValueError(
                "FDA and SegDesic are mutually exclusive. "
                "Use --use_fda for FDA-based training OR --geo_metadata for SegDesic training, not both."
            )

        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # CHANGED: dynamic tolerant key — eliminates the hardcoded 'tolerant_5px'
        # that caused a KeyError because DetailedEvaluator defaulted to boundary_tolerance=2
        # and generated key 'tolerant_2px', not 'tolerant_5px'.
        self.tol_key = f'tolerant_{args.boundary_tolerance}px'

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]['test']
            exp_name = f'dinov3_segdesic_loco_holdout_{test_city}_{args.resolution}_{1}'
        elif args.mode == 'all':
            exp_name = f'dinov3_segdesic_all_{args.resolution}_{1}'
        else:
            exp_name = f'dinov3_segdesic_{args.mode}_{1}'

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Save arguments
        with open(os.path.join(self.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))

        # Initialize model
        print('Initializing DINOv3 with SegDesic...')
        self.model = DINOv3SegDesic(
            num_classes=args.num_classes,
            model_name=args.model_name,
            weights_path=args.weights_path,
            pretrained=args.pretrained,
            frozen_stages=args.frozen_stages,
            segdesic_hidden_dim=args.segdesic_hidden_dim,
            segdesic_num_scales=args.segdesic_num_scales
        ).to(self.device)

        # Print model info
        total_params     = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total parameters:     {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')

        # Setup loss function
        self.criterion = SegDesicLoss(aux_weight=0.0, alpha=args.alpha)  # No aux branches in DINOv3

        # Setup optimizer (AdamW for ViT)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

        # Setup learning rate scheduler
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
        #   eval_boundary_tolerant=False → per-image STRICT mIOU
        # Both come from DetailedEvaluator (per-image mean), NOT ShadowMetrics.
        # ------------------------------------------------------------------
        self.use_tolerant_for_decisions = args.eval_boundary_tolerant
        if self.use_tolerant_for_decisions:
            print(f'>> Decision metric: TOLERANT mIOU '
                  f'(±{args.boundary_tolerance}px boundary excluded)')
        else:
            print(f'>> Decision metric: STRICT per-image mIOU '
                  f'(DetailedEvaluator, not pooled ShadowMetrics)')

        # CHANGED: DetailedEvaluator ALWAYS instantiated — not guarded by
        # eval_boundary_tolerant flag. boundary_tolerance is now configurable
        # from CLI (was missing entirely, defaulted to 2 inside DetailedEvaluator
        # while all key lookups used hardcoded 'tolerant_5px' → KeyError).
        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance)

        # CHANGED: renamed best_miou → best_decision_miou (drives checkpoint/early-stop)
        # Added best_strict_miou for reference logging only (pooled ShadowMetrics value)
        self.start_epoch              = 0
        self.best_decision_miou       = 0.0
        self.best_strict_miou         = 0.0   # pooled strict mIOU (reference only)
        self.best_shadow_iou          = 0.0
        self.best_f1                  = 0.0
        self.epochs_without_improvement = 0

        # Total loss histories (CE + domain combined)
        self.train_losses = []
        self.val_losses   = []

        # CHANGED: per-component loss histories for plot_loss_curves individual panels.
        # These were previously only local variables in train_epoch and were never
        # accumulated across epochs, so the individual loss panels were never rendered.
        self.train_seg_losses        = []   # seg_total (CE) loss per epoch
        self.train_domain_src_losses = []   # source domain cosine loss per epoch
        self.train_domain_tgt_losses = []   # target domain cosine loss per epoch (0 if no target)

        # Metric history (ShadowMetrics pooled, reference only)
        self.train_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }
        self.val_metrics_history = {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # Load datasets WITH geocoordinate metadata
        print('\nLoading datasets with geocoordinate metadata...')
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
            geo_metadata_path=args.geo_metadata,  # Required for SegDesic
            use_fda=False  # Never use FDA with SegDesic
        )

        self.use_target_domain = 'target' in self.dataloaders
        if self.use_target_domain:
            print(f'Target domain (unlabeled) samples: {len(self.dataloaders["target"].dataset)}')

        print(f'Training samples:   {len(self.dataloaders["train"].dataset)}')
        print(f'Validation samples: {len(self.dataloaders["val"].dataset)}')
        print(f'Test samples:       {len(self.dataloaders["test"].dataset)}')

    # ------------------------------------------------------------------
    # Decision metric helper
    # CHANGED: mirrors train_dinov3.py._get_decision_miou exactly.
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
        Train for one epoch with optional target domain adaptation.

        Returns
        -------
        epoch_loss : float  — total combined loss (seg + domain)
        metrics    : dict   — ShadowMetrics pooled (reference only)
        """
        self.model.train()

        epoch_loss               = 0.0
        epoch_seg_loss           = 0.0
        epoch_domain_loss_source = 0.0
        epoch_domain_loss_target = 0.0

        train_metrics = ShadowMetrics()

        num_batches = len(self.dataloaders['train'])
        print(f'\nEpoch {epoch}/{self.args.epochs}')
        print('-' * 50)
        start_time = time.time()

        # Create iterator for target domain if available
        target_iter = None
        if self.use_target_domain:
            target_iter = iter(self.dataloaders['target'])

        domain_loss_tgt = None   # keeps linter happy; set inside the block below

        for batch_idx, batch_source in enumerate(self.dataloaders['train']):
            # SOURCE DOMAIN (labeled)
            images_src = batch_source['image'].to(self.device)
            masks_src  = batch_source['mask'].to(self.device)
            lat_src    = batch_source.get('lat', None)
            lon_src    = batch_source.get('lon', None)

            if lat_src is not None:
                lat_src = lat_src.to(self.device)
            if lon_src is not None:
                lon_src = lon_src.to(self.device)

            # Forward pass on source
            outputs_src = self.model(images_src, lat_src, lon_src)

            # Compute segmentation + domain loss for source
            geo_outputs_src = outputs_src.get('geo', None)
            losses_src = self.criterion(outputs_src, masks_src, geo_outputs_src)
            loss = losses_src['total']

            # TARGET DOMAIN (unlabeled) — only compute domain loss
            domain_loss_tgt = None
            if self.use_target_domain:
                try:
                    batch_target = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.dataloaders['target'])
                    batch_target = next(target_iter)

                images_tgt = batch_target['image'].to(self.device)
                lat_tgt    = batch_target.get('lat', None)
                lon_tgt    = batch_target.get('lon', None)

                if lat_tgt is not None:
                    lat_tgt = lat_tgt.to(self.device)
                if lon_tgt is not None:
                    lon_tgt = lon_tgt.to(self.device)

                # Forward pass on target (no segmentation loss)
                outputs_tgt     = self.model(images_tgt, lat_tgt, lon_tgt)
                geo_outputs_tgt = outputs_tgt.get('geo', None)

                # Compute only domain loss for target
                if geo_outputs_tgt is not None:
                    cos_sim_tgt    = F.cosine_similarity(
                        geo_outputs_tgt['pred_encoding'],
                        geo_outputs_tgt['gt_encoding'],
                        dim=1
                    )
                    domain_loss_tgt = (1 - cos_sim_tgt).mean()
                    loss = loss + self.args.alpha * domain_loss_tgt
                    epoch_domain_loss_target += domain_loss_tgt.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics — source domain only (has labels)
            # CHANGED: use filtered_outputs consistently for both ShadowMetrics
            # and DetailedEvaluator (was using raw outputs_src['main'] for
            # DetailedEvaluator before, creating an inconsistency)
            filtered_outputs = filter_small_predictions(outputs_src['main'], min_pixels=10)
            train_metrics.update(filtered_outputs, masks_src)

            # CHANGED: DetailedEvaluator update is now ALWAYS done (removed
            # eval_boundary_tolerant guard). Uses filtered_outputs for consistency.
            preds = torch.argmax(filtered_outputs, dim=1)
            self.detailed_evaluator_train.update(preds, masks_src, images_src)

            # Track losses
            epoch_loss       += loss.item()
            epoch_seg_loss   += losses_src['seg_total'].item()
            if 'domain_loss' in losses_src:
                epoch_domain_loss_source += losses_src['domain_loss'].item()

            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                domain_src_str = (f"Domain_Src: {losses_src['domain_loss'].item():.4f}"
                                  if 'domain_loss' in losses_src else "")
                domain_tgt_str = (f"Domain_Tgt: {domain_loss_tgt.item():.4f}"
                                  if domain_loss_tgt is not None else "")
                print(f'Batch [{batch_idx + 1}/{num_batches}] | '
                      f'Loss: {loss.item():.4f} | '
                      f'Seg: {losses_src["seg_total"].item():.4f} | '
                      f'{domain_src_str} | {domain_tgt_str}')

        # Compute average losses
        epoch_loss               /= num_batches
        epoch_seg_loss           /= num_batches
        epoch_domain_loss_source /= num_batches
        epoch_domain_loss_target /= num_batches

        # CHANGED: append component losses to per-epoch lists so they can be
        # passed to plot_loss_curves for the individual subplot panels.
        # Previously these were only local variables and were discarded.
        self.train_losses.append(epoch_loss)
        self.train_seg_losses.append(epoch_seg_loss)
        self.train_domain_src_losses.append(epoch_domain_loss_source)
        self.train_domain_tgt_losses.append(epoch_domain_loss_target)

        # Compute metrics
        metrics    = train_metrics.compute()
        epoch_time = time.time() - start_time

        print(f'\nTraining Results:')
        print(f'Time: {epoch_time:.2f}s | Loss: {epoch_loss:.4f} | '
              f'Seg: {epoch_seg_loss:.4f} | '
              f'Domain_Src: {epoch_domain_loss_source:.4f} | '
              f'Domain_Tgt: {epoch_domain_loss_target:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # Log to TensorBoard — loss components + pooled metrics (reference)
        self.writer.add_scalar('Train/TotalLoss',         epoch_loss,               epoch)
        self.writer.add_scalar('Train/SegLoss',           epoch_seg_loss,           epoch)
        self.writer.add_scalar('Train/DomainLoss_Source', epoch_domain_loss_source, epoch)
        if self.use_target_domain:
            self.writer.add_scalar('Train/DomainLoss_Target', epoch_domain_loss_target, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Train/{key}', val, epoch)

        for key in self.train_metrics_history:
            self.train_metrics_history[key].append(metrics[key])

        # CHANGED: DetailedEvaluator always computed; use self.tol_key instead
        # of hardcoded 'tolerant_5px' (which caused KeyError since the evaluator
        # defaulted to boundary_tolerance=2 and generated key 'tolerant_2px').
        detailed_results = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict_tr   = detailed_results['boundary_tolerant']['strict']
        tolerant_tr = detailed_results['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Train/mIOU_strict_perimage',   strict_tr['iou'],   epoch)
        self.writer.add_scalar('Train/F1_strict_perimage',     strict_tr['f1'],    epoch)
        self.writer.add_scalar('Train/mIOU_tolerant_perimage', tolerant_tr['iou'], epoch)
        self.writer.add_scalar('Train/F1_tolerant_perimage',   tolerant_tr['f1'],  epoch)

        print(f'Per-image Strict:   F1={strict_tr["f1"]:.2f}%  mIOU={strict_tr["iou"]:.2f}%')
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
                                    (ALWAYS populated; caller calls
                                     _get_decision_miou on it)
        """
        print('\nValidating...')
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders['val']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Forward pass (no geographic loss during validation)
                outputs = self.model(images)

                # Handle both dict and tensor outputs
                main_output = outputs['main'] if isinstance(outputs, dict) else outputs

                # Compute segmentation loss only
                seg_criterion = nn.CrossEntropyLoss()
                loss = seg_criterion(main_output, masks)
                val_loss += loss.item()

                # CHANGED: use filtered_outputs consistently for both evaluators
                filtered_outputs = filter_small_predictions(main_output, min_pixels=10)
                val_metrics.update(filtered_outputs, masks)

                # CHANGED: DetailedEvaluator always updated (removed
                # eval_boundary_tolerant guard). Uses filtered_outputs for
                # consistency with train_epoch.
                preds = torch.argmax(filtered_outputs, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders['val'])
        metrics   = val_metrics.compute()

        # Track val loss for plotting
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

        print(f'Validation Results:')
        print(f'Loss: {val_loss:.4f}')
        print(f'OA: {metrics["OA"]:.2f}%  Precision: {metrics["Precision"]:.2f}%  '
              f'F1: {metrics["F1"]:.2f}%  BER: {metrics["BER"]:.2f}%  '
              f'mIOU(pooled): {metrics["mIOU"]:.2f}%  '
              f'Shadow_IOU: {metrics["Shadow_IOU"]:.2f}%')

        # Log to TensorBoard — pooled metrics (reference)
        self.writer.add_scalar('Val/Loss', val_loss, epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f'Val/{key}', val, epoch)

        # CHANGED: DetailedEvaluator always computed; return detailed_results
        # so train() can call _get_decision_miou on it — scalar decision_miou
        # is no longer returned from validate() (mirrors train_dinov3.py).
        detailed_results = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict_val   = detailed_results['boundary_tolerant']['strict']
        tolerant_val = detailed_results['boundary_tolerant'][self.tol_key]

        self.writer.add_scalar('Val/mIOU_strict_perimage',   strict_val['iou'],   epoch)
        self.writer.add_scalar('Val/F1_strict_perimage',     strict_val['f1'],    epoch)
        self.writer.add_scalar('Val/mIOU_tolerant_perimage', tolerant_val['iou'], epoch)
        self.writer.add_scalar('Val/F1_tolerant_perimage',   tolerant_val['f1'],  epoch)

        print(f'Per-image Strict:   F1={strict_val["f1"]:.2f}%  mIOU={strict_val["iou"]:.2f}%')
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
            'boundary_tolerance':           self.args.boundary_tolerance,
            'train_losses':                 self.train_losses,
            'val_losses':                   self.val_losses,
            'train_seg_losses':             self.train_seg_losses,
            'train_domain_src_losses':      self.train_domain_src_losses,
            'train_domain_tgt_losses':      self.train_domain_tgt_losses,
            'train_metrics_history':        self.train_metrics_history,
            'val_metrics_history':          self.val_metrics_history,
            'args':                         vars(self.args),
        }

        # Save latest
        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if is_best:
            best_path = os.path.join(self.output_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f'Best checkpoint saved to {best_path}')

        # Save epoch checkpoint
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
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)

        # Restore per-component loss histories if available
        self.train_losses            = checkpoint.get('train_losses', [])
        self.val_losses              = checkpoint.get('val_losses', [])
        self.train_seg_losses        = checkpoint.get('train_seg_losses', [])
        self.train_domain_src_losses = checkpoint.get('train_domain_src_losses', [])
        self.train_domain_tgt_losses = checkpoint.get('train_domain_tgt_losses', [])
        self.train_metrics_history   = checkpoint.get('train_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })
        self.val_metrics_history     = checkpoint.get('val_metrics_history', {
            'OA': [], 'Precision': [], 'F1': [], 'BER': [], 'mIOU': [], 'Shadow_IOU': []
        })

        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px)'
                        if self.use_tolerant_for_decisions else 'Strict per-image')
        print(f'Resumed from epoch {checkpoint["epoch"]}')
        print(f'Best decision mIOU ({metric_label}): {self.best_decision_miou:.2f}%  '
              f'Epochs w/o improvement: {self.epochs_without_improvement}')

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Main training loop.

        Decision logic (best checkpoint, early stopping) is driven by
        _get_decision_miou(detailed_results) which returns per-image mIOU
        from DetailedEvaluator:
          - With --eval_boundary_tolerant → Tolerant mIOU (±K px band excluded)
          - Without                       → Strict  mIOU (standard pixel-level)

        ShadowMetrics pooled values are logged for reference only.
        """
        print('\n' + '='*50)
        print('Starting training with SegDesic...')
        print('='*50)

        patience = self.args.early_stopping_patience
        metric_label = (f'Tolerant (±{self.args.boundary_tolerance}px) mIOU'
                        if self.use_tolerant_for_decisions else 'Strict per-image mIOU')
        if patience > 0:
            print(f'Early stopping: patience={patience}  metric={metric_label}')

        for epoch in range(self.start_epoch, self.args.epochs):
            # Update learning rate
            current_lr = self.scheduler.step(epoch)
            print(f'\nLearning rate: {current_lr:.2e}')

            # Train
            train_loss, train_metrics = self.train_epoch(epoch + 1)

            # CHANGED: validate() now returns detailed_results (not scalar decision_miou).
            # Decision metric is extracted here via _get_decision_miou().
            val_loss, val_metrics, detailed_results = self.validate(epoch + 1)

            # CHANGED: decision now comes from _get_decision_miou(detailed_results)
            # — per-image mean from DetailedEvaluator, never pooled ShadowMetrics.
            decision_miou = self._get_decision_miou(detailed_results)
            self.writer.add_scalar('Val/Decision_mIOU', decision_miou, epoch + 1)

            # Best checkpoint keyed on decision_miou
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

            # Early stopping keyed on decision_miou
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(f'\nEarly stopping triggered! No {metric_label} improvement for '
                      f'{patience} epochs.')
                break

        print('\nTraining completed!')
        print(f'Best {metric_label}: {self.best_decision_miou:.2f}%')
        print(f'Best strict pooled mIOU (reference): {self.best_strict_miou:.2f}%')
        print(f'Best Shadow IoU: {self.best_shadow_iou:.2f}%')
        print(f'Best F1:         {self.best_f1:.2f}%')

        # CHANGED: pass per-component loss histories to plot_loss_curves so
        # individual panels actually render. Previously only train_losses and
        # val_losses were passed, so named_train_components was None and the
        # individual panels were silently skipped every run.
        print('\nGenerating plots...')
        loss_components = [
            {
                'label': 'Seg Total',
                'train': self.train_seg_losses,
                'val':   self.val_losses,        # val only computes CE/seg loss
                'color': '#F18F01',
            },
            {
                'label': 'Domain Source',
                'train': self.train_domain_src_losses,
                'val':   None,
                'color': '#6A994E',
            },
        ]
        if self.use_target_domain and any(v > 1e-9 for v in self.train_domain_tgt_losses):
            loss_components.append({
                'label': 'Domain Target',
                'train': self.train_domain_tgt_losses,
                'val':   None,
                'color': '#8338EC',
            })

        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, 'loss_curves.png'),
            named_train_components=loss_components,
            title='DINOv3+SegDesic — Training Loss Curves',
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

        # Load best checkpoint
        best_checkpoint = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        else:
            print('Warning: Best checkpoint not found, using current model weights')

        self.model.eval()
        test_metrics = ShadowMetrics()

        # CHANGED: DetailedEvaluator always instantiated in test() — removed
        # eval_boundary_tolerant guard. Uses boundary_tolerance from args.
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance)

        with torch.no_grad():
            for batch in self.dataloaders['test']:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)

                # Forward pass
                outputs = self.model(images)

                # Handle both dict and tensor outputs
                main_output = outputs['main'] if isinstance(outputs, dict) else outputs

                # CHANGED: use filtered_outputs consistently for both evaluators
                filtered_outputs = filter_small_predictions(main_output, min_pixels=10)
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

        # Save results
        results_path = os.path.join(self.output_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump({'standard': metrics, 'detailed': detailed_results}, f, indent=4)
        print(f'\nResults saved to {results_path}')

        # Generate visualizations
        # save_best_worst_visualizations now handles dict model outputs internally
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

    # Create trainer
    trainer = SegDesicTrainer(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()