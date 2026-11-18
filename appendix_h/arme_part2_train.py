"""
Arm E Part 2 — Experiment A (decoder retraining), one cell per process.

Backbone fully frozen via dinov3_backbone.py's verified `freeze_backbone=True`
switch, the SAME mechanism arm E Part 1 uses (reused unchanged via
shadowtransfer.arme_part1_runtime.patch_trainer_class). Decoder is freshly
constructed (never loaded from Part 1's checkpoint) and trained on a ~25%
subset of the holdout city's OWN train split, per App G.9.

App G.9's protocol has NO validation-based model selection: fixed 30 epochs,
then evaluate whatever exists at the end. `early_stopping_patience=0`
guarantees the full epoch count always runs, and `trainer.test()` is
deliberately NOT called (it would reload checkpoint_best.pth, which is a
best-BY-VAL-METRIC checkpoint App G.9's protocol never selects on) --
`trainer.model`'s current, post-training state is evaluated directly, which
IS "the last epoch". See module docstring in arme_part2_runtime.py for the
subset-selection provenance (deterministic np.linspace, not a random sample,
despite the paper's prose).

--dump_only reloads checkpoint_latest.pth (the LAST epoch), never
checkpoint_best.pth, to preserve that same "no val-based selection"
semantics on a rerun.

No plotting, no tensorboard anywhere: silenced via patch_trainer_class,
same as Part 1.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(BASE / "dinov3"))

from shadowtransfer import arme_part1_runtime as rt1        # noqa: E402
from shadowtransfer import arme_part2_runtime as rt2         # noqa: E402
from shadowtransfer import arme_part1_dump as dumpmod         # noqa: E402
from shadowtransfer import arme_common as common              # noqa: E402
from shadowtransfer import armc_runtime                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=rt2.CITIES)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_epochs", type=int, default=3)
    ap.add_argument("--dump_only", action="store_true",
                    help="skip training; load checkpoint_latest.pth (LAST "
                         "epoch, not best) and dump only")
    args = ap.parse_args()

    cell = {"mode": rt2.MODE, "city": args.city, "method": rt2.METHOD}
    rid = rt2.run_id(args.city, seed=0, suffix="-smoke" if args.smoke else "")

    dump_dir = BASE / "dumps" / "st" / rid
    train_out_base = str(BASE / "data" / "dinov3" / "outputs" /
                         ("armE_part2_smoke" if args.smoke else "armE_part2"))

    armc_runtime.preflight_write_targets(
        BASE, [dump_dir / "perimage.parquet", dump_dir / "probs.npz",
              dump_dir / "run.json", dump_dir / "test_results.json"],
        label="arme part2 dump")

    common.seed_all(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} city={args.city} "
          f"smoke={args.smoke} dump_only={args.dump_only} run_id={rid}", flush=True)

    epochs = args.smoke_epochs if args.smoke else 30
    warmup = 1 if args.smoke else 5

    train_args = rt1.build_args(
        cell, BASE, epochs=epochs, decoder_lr=1e-4,
        early_stopping_patience=0,  # App G.9: fixed-epoch, no val-based stop
        warmup_epochs=warmup, output_dir_base=train_out_base)

    import train_dinov3
    rt1.patch_trainer_class(train_dinov3)

    t0 = time.time()
    trainer = train_dinov3.Trainer(train_args)  # fresh decoder + frozen backbone

    n_backbone = sum(p.numel() for p in trainer.model.backbone.parameters())
    n_backbone_trainable = sum(p.numel() for p in trainer.model.backbone.parameters()
                              if p.requires_grad)
    n_decoder_trainable = sum(p.numel() for p in trainer.model.decoder.parameters()
                              if p.requires_grad)
    print(f"[arme_part2] backbone params {n_backbone:,} "
          f"({n_backbone_trainable} trainable); decoder trainable params "
          f"{n_decoder_trainable:,}", flush=True)
    assert n_backbone_trainable == 0, (
        f"FATAL backbone_trainable_param_count={n_backbone_trainable}, expected 0")

    train_loader, n_total, n_use, indices = rt2.rebuild_expa_train_loader(
        BASE, args.city, batch_size=train_args.batch_size,
        num_workers=train_args.num_workers)
    old_n_train = len(trainer.dataloaders["train"].dataset)
    trainer.dataloaders["train"] = train_loader
    print(f"[arme_part2] rebuilt Experiment-A train loader: {n_use}/{n_total} "
          f"({n_use/n_total*100:.1f}%) -- was {old_n_train} (mode=single, full city)",
          flush=True)

    n_val = len(trainer.dataloaders["val"].dataset)
    n_test = len(trainer.dataloaders["test"].dataset)
    print(f"[arme_part2] measured dataset sizes: n_train_subset={n_use} "
          f"n_train_pool={n_total} n_val={n_val} n_test={n_test}", flush=True)

    if args.dump_only:
        latest_ckpt = Path(trainer.output_dir) / "checkpoint_latest.pth"
        if not latest_ckpt.exists():
            raise SystemExit(f"FATAL --dump_only set but no checkpoint at {latest_ckpt}")
        trainer.load_checkpoint(str(latest_ckpt))
        wall_clock_train_s = None
    else:
        trainer.train()  # patience=0 -> always runs all `epochs` epochs
        wall_clock_train_s = round(time.time() - t0, 1)

    # Diagnostics only -- neither drives which checkpoint gets evaluated.
    epoch_of_best_loss = (int(np.argmin(trainer.train_losses)) + 1
                          if trainer.train_losses else None)
    best_training_loss = (float(min(trainer.train_losses))
                          if trainer.train_losses else None)
    epoch_of_best_val_metric = getattr(trainer, "_epoch_of_best", None)
    epochs_run = len(trainer.train_losses)

    # trainer.test() is deliberately NOT called -- see module docstring.

    with armc_runtime.PhaseTimeout(600, str(dump_dir), label="post-training dump"):
        run_extra = {
            "rung": 450, "n_train": n_use, "n_train_pool": n_total,
            "n_train_fraction": n_use / n_total,
            "n_train_subset_indices": indices,
            "subset_selection_method": (
                "np.linspace(0, n_total-1, n_use, dtype=int), verbatim from "
                "final_loco/experiment_a_decoder_retrain.py:164-167 "
                "(published code), applied to dinov3's own data/dataset.py "
                "Dataset class over the holdout city's own train split, sorted "
                "filename order. DETERMINISTIC -- no RNG call anywhere. App "
                "G.9's prose calls this 'a randomly sampled 25%'; the actual "
                "code is not random, so no seed exists to recover. Flagged, "
                "not resolved either way."),
            "subset_seed": None,
            "subset_seed_note": ("no seed used or recoverable -- the published "
                                 "selection mechanism is deterministic"),
            "n_val": n_val, "n_test": n_test,
            "encoder_identical_to_pretrained_by_construction": True,
            "encoder_confound_note": (
                "under full freezing, this cell's encoder is byte-identical, "
                "by construction, to arm E Part 1's LOCO and upper cells' "
                "encoders for the same city -- none of the three ever runs a "
                "gradient through the backbone, and all three build it from "
                "the same published pretrained construction. Any IoU "
                "difference among {LOCO, upper, this cell} is therefore "
                "attributable to the decoder alone; the encoder is not a "
                "confound in this comparison."),
            "target_access": "deployment_time",
            "target_access_detail": (
                f"trains the decoder on a labelled 25% subset ({n_use}/{n_total} "
                "images) of the held-out city's OWN train split -- direct, "
                "labelled access to held-out-city data, unlike Part 1's LOCO "
                "cells (source_only). Recorded as deployment_time per "
                "CONTRACT.md's closest matching value; not the full §4.4 "
                "target_labels protocol, since only a 25% subset is used, "
                "matching App G.9's framing as a targeted-retraining probe."),
            "decision_metric": "2class_tol_k2", "reporting_metric": "fg_only_tol_k2",
            "backbone_frozen": True, "backbone_param_count": int(n_backbone),
            "backbone_trainable_param_count": int(n_backbone_trainable),
            "decoder_trainable_param_count": int(n_decoder_trainable),
            "decoder_reinitialized": True,
            "decoder_lr": train_args.lr, "weight_decay": train_args.weight_decay,
            "warmup_epochs": train_args.warmup_epochs, "batch_size": train_args.batch_size,
            "epochs_max": train_args.epochs, "epochs_run": epochs_run,
            "early_stopping_patience": 0,
            "early_stopping_note": (
                "disabled: App G.9 describes a fixed 30-epoch protocol with no "
                "validation-based early stopping or checkpoint selection; the "
                "published experiment_a_decoder_retrain.py has no validation "
                "loop at all. The model evaluated below is whatever state "
                "exists after the fixed epoch count, not a best-by-val "
                "checkpoint."),
            "steps_to_best": epoch_of_best_loss,
            "steps_to_best_note": (
                "epoch of the LOWEST training loss (diagnostic only, matching "
                "the published train_decoder()'s own best_loss tracking, which "
                "is likewise never used to select which epoch gets evaluated). "
                f"best_training_loss={best_training_loss}. A separate, "
                "unused-for-selection epoch_of_best_val_decision_metric is "
                "also recorded below for reference."),
            "epoch_of_best_val_decision_metric_diagnostic_only": epoch_of_best_val_metric,
            "ckpt_path": str(Path(trainer.output_dir) / "checkpoint_latest.pth"),
            "ckpt_is_last_epoch_not_best": True,
            "ckpt_excludes_backbone": True,
            "legacy_train_output_dir": str(trainer.output_dir),
            "parent_run_id": None,
            "cluster": "delta", "partition": "gpuA100x4",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "wall_clock_train_s": wall_clock_train_s,
            "smoke": args.smoke, "dump_only": args.dump_only,
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        df, rows = dumpmod.dump_cell(trainer, cell, rid, BASE, run_extra, dump_dir, t0)
        summary = dumpmod.build_test_results_summary(rows)
        (dump_dir / "test_results.json").write_text(json.dumps(summary, indent=2))
        print(f"[arme_part2] wrote {dump_dir / 'test_results.json'}", flush=True)

    print(f"WROTE dump {dump_dir}  rows={len(df)}  "
          f"wall_clock_total_s={round(time.time() - t0, 1)}", flush=True)


if __name__ == "__main__":
    main()
