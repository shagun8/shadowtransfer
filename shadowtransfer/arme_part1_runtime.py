"""
Arm E Part 1 — frozen-backbone DINOv3 training. Runtime: Trainer-class
patches and per-cell argument/naming helpers.

Nothing here reimplements training. `train_dinov3.py`'s Trainer is reused
unmodified except for three methods, rebound on the CLASS (armc_runtime's
established pattern: rebind module/class attributes after import, before any
Trainer is constructed) rather than by editing the file:

  _assert_backbone_frozen -- the published, already-extended train_dinov3.py
      calls this twice (construction; after the first optimizer step) and
      already hard-fails if any backbone tensor is trainable. This patch
      ADDS the grad-is-None check the task also asks for. A no-op wrapper
      would be pointless; this genuinely extends the check.

  save_checkpoint / load_checkpoint -- rewritten to persist and reload ONLY
      `model.decoder`'s state_dict, never the frozen backbone's ~21M
      parameters. /projects sits at 429.8G of 500G soft; six ~500MB
      full-model checkpoints would cost ~3GB for weights that are always
      reproducible, byte-for-byte, from the same published pretrained
      backbone construction -- nothing about the backbone ever changes, so
      nothing about it needs to be saved. Also records `epoch_of_best` (the
      published checkpoint dict has no such field) for run.json's
      steps_to_best.

`rebuild_loco_val_loader` is a SEPARATE fix, applied by the driver AFTER
Trainer construction, not a class patch: `get_dataloaders` (unmodified) sizes
its `samples_per_dir` cap from the TRAIN split (450 // 2 = 225 per source
city) and applies the SAME cap to val. Each source city's own val pool is
only 150, under that cap, so val is never actually subset and LOCO cells pool
150+150=300 -- confirmed by measurement on the first two smoke runs
(2026-07-29). The operator confirmed the intended protocol is 150 total, 75
per source city (paper sec 3.3). Rather than patching get_dataloaders (shared
by every arch/mode) or hardcoding 75, this rebuilds ONLY the val loader for
LOCO cells with `samples_per_dir` computed from val's OWN per-city size (150
// 2 = 75), the same formula get_dataloaders already uses for train, just
applied to the right split. Train and test are untouched -- both already
correct (train 225/city already matches rung 450; test is the held-out city's
own 150-image test folder, never subset at all).

CELLS is the 6-cell grid: 3 holdout cities x {loco, upper}, highres, rung
450, seed 0. Naming matches the PROJECT'S EXISTING convention, checked
against mamnet-vanilla-loco-r225-*/mamnet-none-upper-r225-* run.json (not
the task's own shorthand "vanilla only"): method is "vanilla" for loco,
"none" for upper -- CONTRACT.md's run_id is
{arch}-{method}-{mode}-r{rung}-{holdout}-{res}-s{seed}, and "arch" here is
`dinov3frozen`, the exact token CONTRACT.md section 1 already reserves for
this configuration.
"""

import argparse
import os
import sys
from pathlib import Path

import torch

from . import armc_runtime

CITIES = ("chicago", "miami", "phoenix")
# data/dataset.py LOCO_FOLDS: 0 tests phoenix, 1 tests miami, 2 tests chicago.
FOLD_ID = {"phoenix": 0, "miami": 1, "chicago": 2}
ARCH = "dinov3frozen"
RUNG = 450

CELLS = tuple(
    {"mode": mode, "city": city, "method": ("vanilla" if mode == "loco" else "none")}
    for city in CITIES
    for mode in ("loco", "upper")
)


def run_id(cell, seed=0, suffix=""):
    return (f"{ARCH}-{cell['method']}-{cell['mode']}-r{RUNG}-{cell['city']}-hr-s{seed}"
            + suffix)


def dinov3_weights_path(base):
    return str(Path(base) / "python" / "dinov3" / "weights" /
               "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")


def build_args(cell, base, epochs, decoder_lr, early_stopping_patience,
              warmup_epochs, output_dir_base, weight_decay=0.05, batch_size=8,
              save_freq=999999, num_workers=1):
    """argparse.Namespace matching every attribute train_dinov3.Trainer reads
    off `args` (it never re-parses argv). Table 23: AdamW, wd 0.05, batch 8,
    max 100 epochs, cosine schedule w/ 5-epoch linear warmup, H/V flip p=0.5
    and ImageNet normalisation (both already in data/dataset.py, untouched),
    early-stopping patience 20. `decoder_lr`/`epochs`/`warmup_epochs`/
    `early_stopping_patience` are parameterised so the smoke stage can pass
    smaller values; the full grid passes the Table 23 + task values."""
    base = Path(base)
    a = argparse.Namespace()
    a.img_size = 384
    a.batch_size = batch_size
    a.num_workers = num_workers
    a.num_classes = 2
    a.model_name = "dinov3_vits16"
    a.weights_path = dinov3_weights_path(base)
    a.pretrained = True
    a.frozen_stages = -1
    a.freeze_backbone = True
    a.epochs = epochs
    a.lr = decoder_lr
    a.weight_decay = weight_decay
    a.warmup_epochs = warmup_epochs
    a.min_lr = 1e-6
    a.use_fda = False
    a.fda_target_root = None
    a.fda_L = 0.01
    a.output_dir = output_dir_base
    a.save_freq = save_freq
    a.resume = None
    a.eval_only = False
    a.device = "cuda"
    a.eval_boundary_tolerant = True
    a.boundary_tolerance = 2
    a.early_stopping_patience = early_stopping_patience
    a.comparison_inference_dir = None
    a.comparison_data_root = None

    if cell["mode"] == "loco":
        a.mode = "loco"
        a.data_root = None
        a.base_data_root = str(base / "data" / "Final_data_test") + "/"
        a.resolution = "highres"
        a.fold_id = FOLD_ID[cell["city"]]
        a.cities = None
    else:
        a.mode = "single"
        a.data_root = str(base / "data" / "Final_data_test" / cell["city"] / "highres") + "/"
        a.base_data_root = None
        a.resolution = None
        a.fold_id = None
        a.cities = None
    return a


def rebuild_loco_val_loader(base, cell, batch_size=8, num_workers=1, random_seed=42):
    """Replace `trainer.dataloaders['val']` for a LOCO cell with one that
    actually pools to 150 (75/source city), per the operator-confirmed sec
    3.3 protocol. See module docstring for why get_dataloaders' own val
    loader does not already do this.

    Mirrors get_dataloaders' LOCO val-dataset construction (data/dataset.py
    lines ~594-601) exactly -- same Dataset class, same DataLoader kwargs --
    except `samples_per_dir` is computed from val's OWN per-city pool (150),
    not train's (450), so this scales correctly if fold sizes ever change.
    """
    if str(Path(base) / "python" / "dinov3") not in sys.path:
        sys.path.insert(0, str(Path(base) / "python" / "dinov3"))
    from data.dataset import Dataset, LOCO_FOLDS

    fold_id = FOLD_ID[cell["city"]]
    train_cities = LOCO_FOLDS[fold_id]["train"]
    val_paths = [str(Path(base) / "data" / "Final_data_test" / c / "highres")
                for c in train_cities]

    sample_val_dir = os.path.join(val_paths[0], "val", "images")
    single_city_val_size = len([
        f for f in os.listdir(sample_val_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
    samples_per_dir = single_city_val_size // len(train_cities)

    val_dataset = Dataset(val_paths, split="val", img_size=384, augment=False,
                          samples_per_dir=samples_per_dir, random_seed=random_seed)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return val_loader, samples_per_dir


def patch_trainer_class(train_dinov3_mod):
    """Rebind Trainer methods BEFORE Trainer(args) is constructed -- the
    first `_assert_backbone_frozen` call happens inside `__init__`."""
    armc_runtime.silence_worker_side_effects(train_dinov3_mod)
    Trainer = train_dinov3_mod.Trainer
    orig_assert = Trainer._assert_backbone_frozen

    def _assert_backbone_frozen(self, where):
        orig_assert(self, where)
        if not getattr(self.args, "freeze_backbone", False):
            return
        bad = [n for n, p in self.model.backbone.named_parameters()
              if p.grad is not None]
        if bad:
            raise RuntimeError(
                f"FATAL --freeze_backbone set but {len(bad)} backbone tensors "
                f"have a non-None .grad {where}: {[n for n in bad][:10]}")
        print(f"  grad check OK ({where}): every backbone param.grad is None",
              flush=True)

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            self._epoch_of_best = epoch
        checkpoint = {
            "epoch": epoch,
            "decoder_state_dict": self.model.decoder.state_dict(),
            "backbone_excluded": True,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_decision_miou": self.best_decision_miou,
            "best_strict_miou": self.best_strict_miou,
            "best_shadow_iou": self.best_shadow_iou,
            "best_f1": self.best_f1,
            "epoch_of_best": getattr(self, "_epoch_of_best", None),
            "epochs_without_improvement": self.epochs_without_improvement,
            "use_tolerant_for_decisions": self.use_tolerant_for_decisions,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_metrics_history": self.train_metrics_history,
            "val_metrics_history": self.val_metrics_history,
            "args": vars(self.args),
        }
        latest = f"{self.output_dir}/checkpoint_latest.pth"
        torch.save(checkpoint, latest)
        n_dec = sum(v.numel() for v in checkpoint["decoder_state_dict"].values())
        print(f"[arme_part1] decoder-only checkpoint -> {latest} "
              f"({n_dec:,} params, backbone excluded)", flush=True)
        if is_best:
            best = f"{self.output_dir}/checkpoint_best.pth"
            torch.save(checkpoint, best)
            print(f"[arme_part1] decoder-only BEST checkpoint -> {best}", flush=True)
        # No periodic checkpoint_epoch_N.pth: best/latest only (CLAUDE.md).

    def load_checkpoint(self, checkpoint_path):
        print(f"[arme_part1] loading decoder-only checkpoint {checkpoint_path}",
              flush=True)
        ck = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.decoder.load_state_dict(ck["decoder_state_dict"])
        if "optimizer_state_dict" in ck:
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.start_epoch = ck["epoch"] + 1
        self.best_decision_miou = ck.get("best_decision_miou", 0.0)
        self.best_strict_miou = ck.get("best_strict_miou", 0.0)
        self.best_shadow_iou = ck.get("best_shadow_iou", 0.0)
        self.best_f1 = ck.get("best_f1", 0.0)
        self._epoch_of_best = ck.get("epoch_of_best")
        self.epochs_without_improvement = ck.get("epochs_without_improvement", 0)
        self.train_losses = ck.get("train_losses", [])
        self.val_losses = ck.get("val_losses", [])
        self.train_metrics_history = ck.get("train_metrics_history",
                                            self.train_metrics_history)
        self.val_metrics_history = ck.get("val_metrics_history",
                                          self.val_metrics_history)
        print(f"[arme_part1] resumed at epoch {ck['epoch']}, "
              f"best_decision_miou={self.best_decision_miou:.2f}", flush=True)

    Trainer._assert_backbone_frozen = _assert_backbone_frozen
    Trainer.save_checkpoint = save_checkpoint
    Trainer.load_checkpoint = load_checkpoint
    return Trainer
