"""
Training script for OGLANetFADA
Implements Frequency-Adapted Domain Generalization for cross-city shadow detection.

===============================================================================
KEY DIFFERENCES FROM train.py (base OGLANet):
===============================================================================

1. MODEL:      OGLANetFADA — GLAMEncoder frozen, FADA blocks + DFFM + Decoder +
               OAM are trainable.
2. OPTIMIZER:  Adam (not Adamax); only trainable parameters are optimised.
               Encoder parameters are excluded entirely.
3. LR:         Default 1e-4 (FADA paper default for Adam with frozen backbone)
               instead of 5e-4 (base OGLANet with Adamax).
4. LR SCHED:   ReduceLROnPlateau, patience=3, factor=0.5 (tighter than base
               OGLANet's patience=5, matching FADA paper practice).
5. ARGS:       Additional FADA hyperparameters: --fada_rank, --fada_token_length,
               --fada_stages, --lr_fada, --lr_decoder, --weight_decay.

DECISION METRICS — identical to train.py:
  DetailedEvaluator per-image mIOU drives LR scheduler, best checkpoint, and
  early stopping.
    --eval_boundary_tolerant set → tolerant mIOU (±K px band excluded)
    otherwise                   → strict per-image mIOU
  ShadowMetrics (pooled) is logged for reference only, never drives decisions.

LOSS — 6-head deep supervision, identical to base OGLANet:
  Total = Loss1 + Loss2 + Loss3 + Loss4 + Loss5 + Loss6
  CrossEntropyLoss(reduction='mean') with fixed image size (384×384) is
  mathematically equivalent to per-image mean loss: mean over H×W pixels gives
  the same result as averaging per-image means when all images share the same
  spatial dimensions.

VISUALIZATION — identical to train.py:
  • Overview panel: all losses (train total, val total, loss1–loss6) on a
    shared y-axis so relative magnitudes are visible.
  • Individual panels: one per component (loss1–loss6) + one for val total,
    each at its own y-axis scale. Total training loss is NOT shown here, so
    small decreases in individual components are clearly visible.
===============================================================================
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
print(">>> Python started", flush=True)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet_fada import OGLANetFADA
from data.dataset import get_dataloaders
from utils.evaluation_detailed import DetailedEvaluator
from utils.losses import OGLANetLoss
from utils.metrics import ShadowMetrics, evaluate_model
from utils.postprocessing import filter_small_predictions
from utils.visualization import (
    plot_loss_curves,
    plot_metrics_curves,
    save_best_worst_visualizations,
)


# ============================================================================
# Arguments
# ============================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Train OGLANetFADA for Domain-Generalised Shadow Detection"
    )

    # ---- Data ----
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root directory of dataset (single-city mode)")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)

    # ---- LOCO / multi-city ----
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "all", "loco"])
    parser.add_argument("--base_data_root", type=str, default=None)
    parser.add_argument("--resolution", type=str, default=None,
                        choices=["highres", "midres"])
    parser.add_argument("--fold_id", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--cities", type=str, nargs="+", default=None)

    # ---- Model ----
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true", default=True)

    # ---- FADA hyperparameters ----
    parser.add_argument(
        "--fada_rank", type=int, default=16,
        help="LoRA rank r for FADA token decomposition "
             "(paper default: 16, Table 3 best at 16–32)",
    )
    parser.add_argument(
        "--fada_token_length", type=int, default=100,
        help="Base token length m for FADA "
             "(paper default: 100, stable in 75–125, Fig 8)",
    )
    parser.add_argument(
        "--fada_stages", type=int, nargs="+", default=[3, 4, 5],
        help="GLAMEncoder stages at which FADA is applied "
             "(default: 3 4 5 → feat3, feat4, feat5)",
    )

    # ---- Training ----
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Base learning rate (FADA paper default: 1e-4 for Adam "
             "with frozen backbone)",
    )
    parser.add_argument(
        "--lr_fada", type=float, default=None,
        help="Separate LR for FADA adapters (default: same as --lr)",
    )
    parser.add_argument(
        "--lr_decoder", type=float, default=None,
        help="Separate LR for DFFM / Decoder / OAM (default: same as --lr)",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # ---- Checkpoint / logging ----
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")

    # ---- Device ----
    parser.add_argument("--device", type=str, default="cuda")

    # ---- Contrast channel ----
    parser.add_argument("--use_contrast", action="store_true")

    # ---- Boundary-tolerant evaluation ----
    parser.add_argument(
        "--eval_boundary_tolerant", action="store_true",
        help="Use tolerant mIOU for all decisions (LR sched, best ckpt, "
             "early stopping). DetailedEvaluator always runs regardless.",
    )
    parser.add_argument(
        "--boundary_tolerance", type=int, default=2,
        help="Don't-care band half-width in pixels (default: 2). "
             "Controls DetailedEvaluator for both strict and tolerant metrics.",
    )

    # ---- Early stopping ----
    parser.add_argument(
        "--early_stopping_patience", type=int, default=None,
        help="Patience in epochs without decision-metric improvement. "
             "Uses tolerant mIOU when --eval_boundary_tolerant is set, "
             "strict per-image mIOU otherwise. 0 or None = disabled.",
    )

    return parser.parse_args()


# ============================================================================
# Trainer
# ============================================================================

class TrainerFADA:
    """Trainer for OGLANetFADA."""

    def __init__(self, args):
        self.args = args

        print(">>> init: device setup", flush=True)
        self.device = torch.device(
            args.device if torch.cuda.is_available() else "cpu"
        )
        print(f">>> init: device={self.device}", flush=True)

        # Tolerant metric key — used wherever we index DetailedEvaluator results
        self.tol_key = f"tolerant_{args.boundary_tolerance}px"

        # ------------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------------
        modifiers = ["fada"]
        modifier_str = "_".join(modifiers)

        if args.mode == "single":
            city = args.data_root.rstrip("/").split("/")[-2]
            res  = args.data_root.rstrip("/").split("/")[-1]
            exp_name = f"oglanet_{modifier_str}_{city}_{res}_1"
        elif args.mode == "all":
            exp_name = f"oglanet_{modifier_str}_all_{args.resolution}_1"
        elif args.mode == "loco":
            from data.dataset import LOCO_FOLDS
            test_city = LOCO_FOLDS[args.fold_id]["test"]
            exp_name = (
                f"oglanet_{modifier_str}_loco_holdout_{test_city}"
                f"_{args.resolution}_1"
            )

        self.output_dir = os.path.join(args.output_dir, exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f">>> init: output_dir={self.output_dir}", flush=True)

        with open(os.path.join(self.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        self.writer = SummaryWriter(
            os.path.join(self.output_dir, "tensorboard")
        )

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        print(">>> init: building OGLANetFADA model", flush=True)
        self.model = OGLANetFADA(
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            img_size=args.img_size,
            use_contrast=args.use_contrast,
            fada_rank=args.fada_rank,
            fada_token_length=args.fada_token_length,
            fada_stages=tuple(args.fada_stages),
        ).to(self.device)

        self.model.count_parameters()
        print(">>> init: model done", flush=True)

        # ------------------------------------------------------------------
        # Loss — identical to base OGLANet (6-head deep supervision)
        # CrossEntropyLoss(reduction='mean') with fixed image size 384×384 is
        # equivalent to per-image mean loss.
        # ------------------------------------------------------------------
        self.criterion = OGLANetLoss()

        # ------------------------------------------------------------------
        # Optimizer — Adam on trainable params only (encoder excluded)
        # ------------------------------------------------------------------
        lr_fada    = args.lr_fada    if args.lr_fada    is not None else args.lr
        lr_decoder = args.lr_decoder if args.lr_decoder is not None else args.lr

        param_groups = self.model.get_param_groups(
            lr_fada=lr_fada, lr_decoder=lr_decoder
        )
        self.optimizer = optim.Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        for pg in self.optimizer.param_groups:
            n_params = sum(p.numel() for p in pg["params"])
            print(
                f"  Optimizer group '{pg.get('name', '?')}': "
                f"{n_params:,} params, lr={pg['lr']}"
            )

        # ------------------------------------------------------------------
        # LR Scheduler
        # Tighter patience (3) vs base OGLANet (5) — consistent with FADA paper
        # ------------------------------------------------------------------
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3)

        # ------------------------------------------------------------------
        # Decision-metric configuration
        # DetailedEvaluator ALWAYS runs.
        # eval_boundary_tolerant controls which per-image metric drives decisions:
        #   True  → tolerant mIOU  (±boundary_tolerance px band excluded)
        #   False → strict mIOU    (all pixels, per-image mean)
        # ShadowMetrics (pooled) is logged for reference only.
        # ------------------------------------------------------------------
        self.use_tolerant_decision = args.eval_boundary_tolerant
        if self.use_tolerant_decision:
            print(
                f">> Decision metric: TOLERANT mIOU "
                f"(±{args.boundary_tolerance}px boundary excluded)"
            )
        else:
            print(">> Decision metric: STRICT per-image mIOU (DetailedEvaluator)")

        self.detailed_evaluator_train = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance
        )
        self.detailed_evaluator_val = DetailedEvaluator(
            boundary_tolerance=args.boundary_tolerance
        )

        # ------------------------------------------------------------------
        # Tracking variables
        # ------------------------------------------------------------------
        self.start_epoch                = 0
        self.best_miou                  = 0.0   # pooled, reference only
        self.best_shadow_iou            = 0.0   # pooled, reference only
        self.best_f1                    = 0.0   # pooled, reference only
        self.best_decision_miou         = 0.0   # drives checkpoint / early-stop
        self.epochs_without_improvement = 0

        # Loss histories — total + 6 components + val
        self.train_losses        = []
        self.train_loss1_history = []
        self.train_loss2_history = []
        self.train_loss3_history = []
        self.train_loss4_history = []
        self.train_loss5_history = []
        self.train_loss6_history = []
        self.val_losses          = []

        # Metric history (pooled ShadowMetrics, reference only)
        self.train_metrics_history = {
            "OA": [], "Precision": [], "F1": [], "BER": [],
            "mIOU": [], "Shadow_IOU": [],
        }
        self.val_metrics_history = {
            "OA": [], "Precision": [], "F1": [], "BER": [],
            "mIOU": [], "Shadow_IOU": [],
        }

        if args.resume:
            self.load_checkpoint(args.resume)

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
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
            use_fda=False,          # input-level FDA not used with FADA
            fda_target_root=None,
            fda_L=0.01,
            use_contrast=args.use_contrast,
        )
        print(">>> init: data loaded", flush=True)
        print(f"Training samples:   {len(self.dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(self.dataloaders['val'].dataset)}")
        print(f"Test samples:       {len(self.dataloaders['test'].dataset)}")

        patience = (
            args.early_stopping_patience
            if args.early_stopping_patience is not None else 0
        )
        if patience > 0:
            print(f">> Early stopping patience: {patience} epochs")

    # ------------------------------------------------------------------
    # Decision metric
    # ------------------------------------------------------------------

    def _get_decision_miou(self, detailed_results):
        """
        Return the mIOU that drives all decisions (LR scheduler, best
        checkpoint, early stopping).

        Both options are per-image means from DetailedEvaluator — never
        the pooled ShadowMetrics value.
        """
        bt = detailed_results["boundary_tolerant"]
        if self.use_tolerant_decision:
            return bt[self.tol_key]["iou"]
        else:
            return bt["strict"]["iou"]

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Returns
        -------
        epoch_loss : float  — average total loss over batches
        metrics    : dict   — pooled ShadowMetrics (reference only)
        """
        self.model.train()  # encoder stays in eval via train() override

        epoch_loss   = 0.0
        epoch_losses = {
            "loss1": 0.0, "loss2": 0.0, "loss3": 0.0,
            "loss4": 0.0, "loss5": 0.0, "loss6": 0.0,
        }
        train_metrics = ShadowMetrics()
        num_batches   = len(self.dataloaders["train"])

        print(f"\nEpoch {epoch}/{self.args.epochs}")
        print("-" * 50)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.dataloaders["train"]):
            images = batch["image"].to(self.device)
            masks  = batch["mask"].to(self.device)

            predictions = self.model(images)       # dict {p1 ... p6}
            losses      = self.criterion(predictions, masks)
            loss        = losses["total"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Filtered predictions — consistent for ShadowMetrics and DetailedEvaluator
            filtered = filter_small_predictions(predictions["p6"], min_pixels=10)
            train_metrics.update(filtered, masks)

            # DetailedEvaluator — ALWAYS active
            preds = torch.argmax(filtered, dim=1)
            self.detailed_evaluator_train.update(preds, masks, images)

            epoch_loss += loss.item()
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(
                    f"Batch [{batch_idx + 1}/{num_batches}] | "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss /= num_batches
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        metrics    = train_metrics.compute()
        epoch_time = time.time() - start_time

        print(f"\nTraining Results:")
        print(
            f"Time: {epoch_time:.2f}s | Total Loss: {epoch_loss:.4f} | "
            f"loss1={epoch_losses['loss1']:.4f}  loss2={epoch_losses['loss2']:.4f}  "
            f"loss3={epoch_losses['loss3']:.4f}  loss4={epoch_losses['loss4']:.4f}  "
            f"loss5={epoch_losses['loss5']:.4f}  loss6={epoch_losses['loss6']:.4f}"
        )
        print(
            f"OA: {metrics['OA']:.2f}%  Precision: {metrics['Precision']:.2f}%  "
            f"F1: {metrics['F1']:.2f}%  BER: {metrics['BER']:.2f}%  "
            f"mIOU(pooled): {metrics['mIOU']:.2f}%  "
            f"Shadow_IOU: {metrics['Shadow_IOU']:.2f}%"
        )

        # TensorBoard — losses
        self.writer.add_scalar("Train/TotalLoss", epoch_loss, epoch)
        for key, val in epoch_losses.items():
            self.writer.add_scalar(f"Train/{key}", val, epoch)

        # TensorBoard — pooled metrics (reference only)
        for key, val in metrics.items():
            self.writer.add_scalar(f"Train/{key}", val, epoch)

        # DetailedEvaluator — per-image metrics (always computed)
        detailed_results = self.detailed_evaluator_train.compute_metrics()
        self.detailed_evaluator_train.reset()

        strict   = detailed_results["boundary_tolerant"]["strict"]
        tolerant = detailed_results["boundary_tolerant"][self.tol_key]

        self.writer.add_scalar("Train/mIOU_strict_perimage",   strict["iou"],   epoch)
        self.writer.add_scalar("Train/F1_strict_perimage",     strict["f1"],    epoch)
        self.writer.add_scalar("Train/mIOU_tolerant_perimage", tolerant["iou"], epoch)
        self.writer.add_scalar("Train/F1_tolerant_perimage",   tolerant["f1"],  epoch)

        print(
            f"Per-image Strict:   F1={strict['f1']:.2f}%  "
            f"mIOU={strict['iou']:.2f}%"
        )
        print(
            f"Per-image Tolerant (±{self.args.boundary_tolerance}px): "
            f"F1={tolerant['f1']:.2f}%  mIOU={tolerant['iou']:.2f}%"
        )

        # Histories
        self.train_losses.append(epoch_loss)
        self.train_loss1_history.append(epoch_losses["loss1"])
        self.train_loss2_history.append(epoch_losses["loss2"])
        self.train_loss3_history.append(epoch_losses["loss3"])
        self.train_loss4_history.append(epoch_losses["loss4"])
        self.train_loss5_history.append(epoch_losses["loss5"])
        self.train_loss6_history.append(epoch_losses["loss6"])
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
        metrics          : dict   — pooled ShadowMetrics (reference only)
        detailed_results : dict   — DetailedEvaluator per-image metrics
                                    (always populated; used for all decisions)
        """
        print("\nValidating...")
        self.model.eval()

        val_loss    = 0.0
        val_metrics = ShadowMetrics()

        with torch.no_grad():
            for batch in self.dataloaders["val"]:
                images = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)

                # Inference mode: model returns P6 tensor directly
                predictions = self.model(images)

                # Val loss — CE on P6
                loss     = self.criterion.criterion(predictions, masks)
                val_loss += loss.item()

                filtered = filter_small_predictions(predictions, min_pixels=10)
                val_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                self.detailed_evaluator_val.update(preds, masks, images)

        val_loss /= len(self.dataloaders["val"])
        metrics   = val_metrics.compute()

        print(f"Validation Results:")
        print(f"Loss: {val_loss:.4f}")
        print(
            f"OA: {metrics['OA']:.2f}%  Precision: {metrics['Precision']:.2f}%  "
            f"F1: {metrics['F1']:.2f}%  BER: {metrics['BER']:.2f}%  "
            f"mIOU(pooled): {metrics['mIOU']:.2f}%  "
            f"Shadow_IOU: {metrics['Shadow_IOU']:.2f}%"
        )

        # TensorBoard — pooled metrics (reference)
        self.writer.add_scalar("Val/Loss",        val_loss,        epoch)
        self.writer.add_scalar("Val/mIOU_pooled", metrics["mIOU"], epoch)
        for key, val in metrics.items():
            self.writer.add_scalar(f"Val/{key}", val, epoch)

        # DetailedEvaluator — per-image metrics (drive all decisions)
        detailed_results = self.detailed_evaluator_val.compute_metrics()
        self.detailed_evaluator_val.reset()

        strict   = detailed_results["boundary_tolerant"]["strict"]
        tolerant = detailed_results["boundary_tolerant"][self.tol_key]

        self.writer.add_scalar("Val/mIOU_strict_perimage",   strict["iou"],   epoch)
        self.writer.add_scalar("Val/F1_strict_perimage",     strict["f1"],    epoch)
        self.writer.add_scalar("Val/mIOU_tolerant_perimage", tolerant["iou"], epoch)
        self.writer.add_scalar("Val/F1_tolerant_perimage",   tolerant["f1"],  epoch)

        print(
            f"Per-image Strict:   F1={strict['f1']:.2f}%  "
            f"mIOU={strict['iou']:.2f}%"
        )
        print(
            f"Per-image Tolerant (±{self.args.boundary_tolerance}px): "
            f"F1={tolerant['f1']:.2f}%  mIOU={tolerant['iou']:.2f}%"
        )

        # Histories
        self.val_losses.append(val_loss)
        for key in self.val_metrics_history:
            self.val_metrics_history[key].append(metrics[key])

        return val_loss, metrics, detailed_results

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch":                        epoch,
            "model_state_dict":             self.model.state_dict(),
            "optimizer_state_dict":         self.optimizer.state_dict(),
            "scheduler_state_dict":         self.scheduler.state_dict(),
            "best_miou":                    self.best_miou,
            "best_shadow_iou":              self.best_shadow_iou,
            "best_f1":                      self.best_f1,
            "best_decision_miou":           self.best_decision_miou,
            "epochs_without_improvement":   self.epochs_without_improvement,
            "train_losses":                 self.train_losses,
            "train_loss1_history":          self.train_loss1_history,
            "train_loss2_history":          self.train_loss2_history,
            "train_loss3_history":          self.train_loss3_history,
            "train_loss4_history":          self.train_loss4_history,
            "train_loss5_history":          self.train_loss5_history,
            "train_loss6_history":          self.train_loss6_history,
            "val_losses":                   self.val_losses,
            "train_metrics_history":        self.train_metrics_history,
            "val_metrics_history":          self.val_metrics_history,
            "args":                         vars(self.args),
            "fada_config": {
                "rank":         self.args.fada_rank,
                "token_length": self.args.fada_token_length,
                "stages":       self.args.fada_stages,
            },
        }

        # Always save latest
        latest_path = os.path.join(self.output_dir, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)
        print(f"Checkpoint saved to {latest_path}")

        if is_best:
            best_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved to {best_path}")

        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(
                self.output_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(checkpoint, epoch_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print("WARNING: Size mismatch — attempting partial load …")
                sd  = checkpoint["model_state_dict"]
                md  = self.model.state_dict()
                pre = {k: v for k, v in sd.items()
                       if k in md and v.size() == md[k].size()}
                md.update(pre)
                self.model.load_state_dict(md)
                print(f"Loaded {len(pre)}/{len(sd)} layers")
            else:
                raise

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch                = checkpoint["epoch"] + 1
        self.best_miou                  = checkpoint.get("best_miou", 0.0)
        self.best_shadow_iou            = checkpoint.get("best_shadow_iou", 0.0)
        self.best_f1                    = checkpoint.get("best_f1", 0.0)
        self.best_decision_miou         = checkpoint.get("best_decision_miou", 0.0)
        self.epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

        self.train_losses        = checkpoint.get("train_losses", [])
        self.train_loss1_history = checkpoint.get("train_loss1_history", [])
        self.train_loss2_history = checkpoint.get("train_loss2_history", [])
        self.train_loss3_history = checkpoint.get("train_loss3_history", [])
        self.train_loss4_history = checkpoint.get("train_loss4_history", [])
        self.train_loss5_history = checkpoint.get("train_loss5_history", [])
        self.train_loss6_history = checkpoint.get("train_loss6_history", [])
        self.val_losses          = checkpoint.get("val_losses", [])

        self.train_metrics_history = checkpoint.get(
            "train_metrics_history",
            {"OA": [], "Precision": [], "F1": [], "BER": [],
             "mIOU": [], "Shadow_IOU": []},
        )
        self.val_metrics_history = checkpoint.get(
            "val_metrics_history",
            {"OA": [], "Precision": [], "F1": [], "BER": [],
             "mIOU": [], "Shadow_IOU": []},
        )

        metric_label = (
            f"Tolerant (±{self.args.boundary_tolerance}px) mIOU"
            if self.use_tolerant_decision else "Strict per-image mIOU"
        )
        print(
            f"Resumed from epoch {checkpoint['epoch']} | "
            f"Best {metric_label}: {self.best_decision_miou:.2f}% | "
            f"Epochs w/o improvement: {self.epochs_without_improvement}"
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        """
        Main training loop.

        Decisions (LR scheduler step, best-checkpoint selection, early stopping)
        are all driven by the per-image *decision metric* from DetailedEvaluator:
          - Tolerant mIOU  when --eval_boundary_tolerant is set
          - Strict  mIOU   otherwise
        Pooled ShadowMetrics is logged for reference but never drives decisions.
        """
        print("\n" + "=" * 52)
        print("Starting OGLANetFADA training …")
        print(
            f"FADA: rank={self.args.fada_rank}, "
            f"m={self.args.fada_token_length}, "
            f"stages={self.args.fada_stages}"
        )
        metric_label = (
            f"Tolerant (±{self.args.boundary_tolerance}px) mIOU"
            if self.use_tolerant_decision else "Strict per-image mIOU"
        )
        print(f"Decision metric: {metric_label}")
        print("=" * 52)

        patience = (
            self.args.early_stopping_patience
            if self.args.early_stopping_patience is not None else 0
        )

        for epoch in range(self.start_epoch, self.args.epochs):
            # ---- Train ----
            train_loss, train_metrics = self.train_epoch(epoch + 1)

            # ---- Validate ----
            val_loss, val_metrics, detailed_results = self.validate(epoch + 1)

            # ---- Decision metric ----
            decision_miou = self._get_decision_miou(detailed_results)

            self.scheduler.step(decision_miou)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr}")
            self.writer.add_scalar("Val/Decision_mIOU", decision_miou, epoch + 1)

            # ---- Best checkpoint ----
            is_best = False
            if decision_miou > self.best_decision_miou:
                self.best_decision_miou      = decision_miou
                is_best                      = True
                self.epochs_without_improvement = 0
                print(
                    f">> New best {metric_label}: "
                    f"{self.best_decision_miou:.2f}%"
                )
            else:
                self.epochs_without_improvement += 1

            # Reference-only pooled bests
            if val_metrics["mIOU"] > self.best_miou:
                self.best_miou = val_metrics["mIOU"]
            if val_metrics["Shadow_IOU"] > self.best_shadow_iou:
                self.best_shadow_iou = val_metrics["Shadow_IOU"]
            if val_metrics["F1"] > self.best_f1:
                self.best_f1 = val_metrics["F1"]

            self.save_checkpoint(epoch + 1, is_best=is_best)

            # Log learning rates per param group
            for pg in self.optimizer.param_groups:
                name = pg.get("name", "group")
                self.writer.add_scalar(
                    f"Train/LR_{name}", pg["lr"], epoch + 1
                )

            # ---- Early stopping ----
            if patience > 0 and self.epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping after {self.epochs_without_improvement} "
                    f"epochs without improvement in {metric_label}."
                )
                break

            print("=" * 52)

        # ---- Final plots ----
        print("\nGenerating plots …")
        plot_loss_curves(
            self.train_losses,
            self.val_losses,
            os.path.join(self.output_dir, "loss_curves.png"),
            component_losses={
                "loss1": self.train_loss1_history,
                "loss2": self.train_loss2_history,
                "loss3": self.train_loss3_history,
                "loss4": self.train_loss4_history,
                "loss5": self.train_loss5_history,
                "loss6": self.train_loss6_history,
            },
        )
        plot_metrics_curves(
            self.train_metrics_history,
            self.val_metrics_history,
            os.path.join(self.output_dir, "metrics_curves.png"),
        )

        print("\nTraining completed!")
        print(f"Best {metric_label}:           {self.best_decision_miou:.2f}%")
        print(f"Best pooled mIOU (reference):  {self.best_miou:.2f}%")
        print(f"Best Shadow IoU:               {self.best_shadow_iou:.2f}%")
        print(f"Best F1:                       {self.best_f1:.2f}%")

        self.writer.close()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        """Test the model using the best checkpoint."""
        print("\n" + "=" * 52)
        print("Testing OGLANetFADA …")
        print("=" * 52)

        best_checkpoint = os.path.join(self.output_dir, "checkpoint_best.pth")
        if os.path.exists(best_checkpoint):
            self.load_checkpoint(best_checkpoint)
        else:
            print("Warning: Best checkpoint not found — using current weights")

        self.model.eval()
        test_metrics  = ShadowMetrics()
        detailed_eval = DetailedEvaluator(
            boundary_tolerance=self.args.boundary_tolerance
        )

        with torch.no_grad():
            for batch in self.dataloaders["test"]:
                images = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)

                predictions = self.model(images)
                filtered    = filter_small_predictions(predictions, min_pixels=10)

                test_metrics.update(filtered, masks)

                preds = torch.argmax(filtered, dim=1)
                detailed_eval.update(preds, masks, images)

        metrics          = test_metrics.compute()
        detailed_results = detailed_eval.compute_metrics()

        print("\n" + "=" * 52)
        print("Pooled Test Results (reference):")
        print("=" * 52)
        print(f"OA:         {metrics['OA']:.2f}%")
        print(f"Precision:  {metrics['Precision']:.2f}%")
        print(f"F1:         {metrics['F1']:.2f}%")
        print(f"BER:        {metrics['BER']:.2f}%")
        print(f"mIOU:       {metrics['mIOU']:.2f}%")
        print(f"Shadow_IOU: {metrics['Shadow_IOU']:.2f}%")

        print("\n" + "=" * 52)
        print("Per-Image Test Results (DetailedEvaluator):")
        print("=" * 52)
        strict   = detailed_results["boundary_tolerant"]["strict"]
        tolerant = detailed_results["boundary_tolerant"][self.tol_key]
        print(f"Strict   — F1: {strict['f1']:.2f}%   mIOU: {strict['iou']:.2f}%")
        print(
            f"Tolerant (±{self.args.boundary_tolerance}px) — "
            f"F1: {tolerant['f1']:.2f}%   mIOU: {tolerant['iou']:.2f}%"
        )
        print(
            f"Pixels excluded by band: {tolerant['pixels_excluded']} "
            f"({tolerant['pct_excluded']:.1f}%)"
        )

        if "size_stratified" in detailed_results:
            print("\nSize-Stratified (Strict):")
            for cat in ["tiny", "small", "medium", "large"]:
                if cat in detailed_results["size_stratified"]:
                    m = detailed_results["size_stratified"][cat]
                    print(
                        f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                        f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)"
                    )

        if "size_stratified_tolerant" in detailed_results:
            print(
                f"\nSize-Stratified "
                f"(Tolerant ±{self.args.boundary_tolerance}px):"
            )
            for cat in ["tiny", "small", "medium", "large"]:
                if cat in detailed_results["size_stratified_tolerant"]:
                    m = detailed_results["size_stratified_tolerant"][cat]
                    print(
                        f"  {cat:8s}: Miss={m['miss_rate']:5.1f}%  "
                        f"IoU={m['avg_iou']:5.1f}%  ({m['total']} shadows)"
                    )

        if (
            "fp_fn_analysis" in detailed_results
            and "fp" in detailed_results["fp_fn_analysis"]
        ):
            fp = detailed_results["fp_fn_analysis"]["fp"]
            print("\nFP Spatial Distribution:")
            print(f"  Within 1px:  {fp['pct_within_1px']:.1f}%")
            print(f"  Within 5px:  {fp['pct_within_5px']:.1f}%")
            print(f"  Within 10px: {fp['pct_within_10px']:.1f}%")

        results_to_save = {
            "standard":   metrics,
            "detailed":   detailed_results,
            "fada_config": {
                "rank":         self.args.fada_rank,
                "token_length": self.args.fada_token_length,
                "stages":       self.args.fada_stages,
            },
        }
        results_path = os.path.join(self.output_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(results_to_save, f, indent=4)
        print(f"\nResults saved to {results_path}")

        print("\nGenerating best/worst predictions visualisations …")
        save_best_worst_visualizations(
            self.model,
            self.dataloaders["test"],
            self.device,
            self.output_dir,
            num_images=10,
        )

        return metrics


# ============================================================================
# Entry point
# ============================================================================

def main():
    print(">>> before get_args", flush=True)
    args = get_args()
    print(
        f">>> args parsed: mode={args.mode}, fold_id={args.fold_id}, "
        f"fada_rank={args.fada_rank}, fada_token_length={args.fada_token_length}, "
        f"fada_stages={args.fada_stages}",
        flush=True,
    )
    trainer = TrainerFADA(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == "__main__":
    main()