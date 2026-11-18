"""
Arm E Part 1 — frozen-backbone DINOv3 training, one cell per process.

Backbone fully frozen via dinov3_backbone.py's verified `freeze_backbone=True`
switch; trainable set is the decoder only (fusion conv, 4 ConvTranspose
stages, classifier head). Freeze is asserted at construction AND after the
first optimizer step (0 trainable backbone params, every backbone .grad is
None) by shadowtransfer.arme_part1_runtime.patch_trainer_class, which fails
the run loudly rather than continuing on a violation.

--dump_only skips training entirely: constructs the model+dataloaders (so
every assertion still runs), loads an existing checkpoint_best.pth, and goes
straight to inference + dump (CLAUDE.md: "A failed dump must never require
retraining").

No plotting, no tensorboard anywhere in this process: silenced on the
train_dinov3 module before Trainer() is constructed.

LOCO cells' val loader is REBUILT immediately after Trainer construction via
arme_part1_runtime.rebuild_loco_val_loader: get_dataloaders' own val loader
pools 300 images (150 per source city, uncapped -- its samples_per_dir is
sized from the TRAIN split and never bites on val's smaller pool). Operator
confirmed the intended protocol is 150 total, 75/source city (sec 3.3); see
that function's docstring for the exact mechanism and why get_dataloaders
itself is not patched.

n_train/n_val/n_test in run.json are the MEASURED dataset sizes
(`len(dataloader.dataset)`), taken AFTER the val-loader rebuild for LOCO
cells, never hardcoded.

epochs_run is read from checkpoint_latest.pth's own 'epoch' field on disk,
never from trainer.train_losses in memory: trainer.test() internally calls
load_checkpoint(best_checkpoint), and the BEST checkpoint's saved history is
a snapshot from whenever it was last the best epoch, not the run's actual
final epoch -- reading train_losses after test() silently under-reports
epochs_run (caught on the smoke run: reported 1 instead of the 3 actually
executed). checkpoint_latest.pth is overwritten every epoch unconditionally,
so its 'epoch' field is always the true last-epoch-run count.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, str(BASE / "dinov3"))

from shadowtransfer import arme_part1_runtime as rt        # noqa: E402
from shadowtransfer import arme_part1_dump as dumpmod        # noqa: E402
from shadowtransfer import arme_common as common             # noqa: E402
from shadowtransfer import armc_runtime                      # noqa: E402


def read_epochs_run(output_dir, epoch_of_best):
    """True last-epoch-run count, from checkpoint_latest.pth's own 'epoch'
    field -- never from trainer.train_losses (see module docstring)."""
    latest = Path(output_dir) / "checkpoint_latest.pth"
    if latest.exists():
        ck = torch.load(str(latest), map_location="cpu", weights_only=False)
        epoch = ck.get("epoch")
        del ck
        if epoch is not None:
            return epoch, "checkpoint_latest.pth 'epoch' field"
    return epoch_of_best, "checkpoint_latest.pth missing; fell back to epoch_of_best"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=rt.CITIES)
    ap.add_argument("--mode", required=True, choices=("loco", "upper"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_epochs", type=int, default=3)
    ap.add_argument("--dump_only", action="store_true",
                    help="skip training; load checkpoint_best.pth and dump only")
    args = ap.parse_args()

    cell = {"mode": args.mode, "city": args.city,
            "method": "vanilla" if args.mode == "loco" else "none"}
    rid = rt.run_id(cell, seed=0, suffix="-smoke" if args.smoke else "")

    dump_dir = BASE / "dumps" / "st" / rid
    train_out_base = str(BASE / "data" / "dinov3" / "outputs" /
                         ("armE_part1_smoke" if args.smoke else "armE_part1"))

    armc_runtime.preflight_write_targets(
        BASE, [dump_dir / "perimage.parquet", dump_dir / "probs.npz",
              dump_dir / "run.json", dump_dir / "test_results.json"],
        label="arme part1 dump")

    common.seed_all(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__} cell={cell} "
          f"smoke={args.smoke} dump_only={args.dump_only} run_id={rid}", flush=True)

    epochs = args.smoke_epochs if args.smoke else 100
    warmup = 1 if args.smoke else 5
    patience = 2 if args.smoke else 20

    train_args = rt.build_args(
        cell, BASE, epochs=epochs, decoder_lr=1e-4,
        early_stopping_patience=patience, warmup_epochs=warmup,
        output_dir_base=train_out_base)

    import train_dinov3
    rt.patch_trainer_class(train_dinov3)

    t0 = time.time()
    trainer = train_dinov3.Trainer(train_args)

    n_backbone = sum(p.numel() for p in trainer.model.backbone.parameters())
    n_backbone_trainable = sum(p.numel() for p in trainer.model.backbone.parameters()
                              if p.requires_grad)
    n_decoder_trainable = sum(p.numel() for p in trainer.model.decoder.parameters()
                              if p.requires_grad)
    print(f"[arme_part1] backbone params {n_backbone:,} "
          f"({n_backbone_trainable} trainable); decoder trainable params "
          f"{n_decoder_trainable:,}", flush=True)
    assert n_backbone_trainable == 0, (
        f"FATAL backbone_trainable_param_count={n_backbone_trainable}, expected 0")

    if cell["mode"] == "loco":
        new_val_loader, val_samples_per_dir = rt.rebuild_loco_val_loader(
            BASE, cell, batch_size=train_args.batch_size,
            num_workers=train_args.num_workers)
        old_n_val = len(trainer.dataloaders["val"].dataset)
        trainer.dataloaders["val"] = new_val_loader
        print(f"[arme_part1] rebuilt LOCO val loader: {val_samples_per_dir}/source-city "
              f"-> {len(new_val_loader.dataset)} total (get_dataloaders' own pooled "
              f"{old_n_val} unmodified)", flush=True)

    n_train = len(trainer.dataloaders["train"].dataset)
    n_val = len(trainer.dataloaders["val"].dataset)
    n_test = len(trainer.dataloaders["test"].dataset)
    print(f"[arme_part1] measured dataset sizes: n_train={n_train} n_val={n_val} "
          f"n_test={n_test}", flush=True)

    if args.dump_only:
        best_ckpt = Path(trainer.output_dir) / "checkpoint_best.pth"
        if not best_ckpt.exists():
            raise SystemExit(f"FATAL --dump_only set but no checkpoint at {best_ckpt}")
        trainer.load_checkpoint(str(best_ckpt))
        wall_clock_train_s = None
    else:
        trainer.train()
        wall_clock_train_s = round(time.time() - t0, 1)

    epoch_of_best = getattr(trainer, "_epoch_of_best", None)
    epochs_run, epochs_run_source = read_epochs_run(trainer.output_dir, epoch_of_best)

    trainer.test()  # writes the legacy-style test_results.json into trainer.output_dir

    with armc_runtime.PhaseTimeout(1200, str(dump_dir), label="post-training dump"):
        run_extra = {
            "rung": 450, "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "n_val_note": (
                "loco: 75/source city (150 total), rebuilt after Trainer construction "
                "per sec 3.3 -- get_dataloaders' own val loader pools 300 unmodified "
                "(samples_per_dir is sized from the train split and never caps val's "
                "smaller pool); see arme_part1_runtime.rebuild_loco_val_loader."
                if cell["mode"] == "loco" else
                "upper: single city, no pooling -- get_dataloaders' own val loader "
                "already gives 150 with no rebuild needed."),
            "target_access": "source_only",
            "target_access_detail": (
                "training and validation draw only from the two source cities; no "
                "held-out-city image is touched before test-time inference"
                if cell["mode"] == "loco" else
                "within-city upper bound: trains and evaluates on the same city; no "
                "held-out city exists in this protocol, so no cross-domain access "
                "occurs -- source_only records the absence of a held-out-city touch, "
                "not an adaptation claim"),
            "decision_metric": "2class_tol_k2", "reporting_metric": "fg_only_tol_k2",
            "backbone_frozen": True, "backbone_param_count": int(n_backbone),
            "backbone_trainable_param_count": int(n_backbone_trainable),
            "decoder_trainable_param_count": int(n_decoder_trainable),
            "decoder_lr": train_args.lr, "weight_decay": train_args.weight_decay,
            "warmup_epochs": train_args.warmup_epochs, "batch_size": train_args.batch_size,
            "epochs_max": train_args.epochs, "epochs_run": epochs_run,
            "epochs_run_source": epochs_run_source,
            "early_stopping_patience": train_args.early_stopping_patience,
            "best_decision_metric_pct": trainer.best_decision_miou,
            "epoch_of_best": epoch_of_best, "steps_to_best": epoch_of_best,
            "steps_to_best_note": "epoch-granular; no per-batch step counter is kept",
            "ckpt_path": str(Path(trainer.output_dir) / "checkpoint_best.pth"),
            "ckpt_excludes_backbone": True,
            "legacy_train_output_dir": str(trainer.output_dir),
            "legacy_test_results_json": str(Path(trainer.output_dir) / "test_results.json"),
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
        print(f"[arme_part1] wrote {dump_dir / 'test_results.json'}", flush=True)

    print(f"WROTE dump {dump_dir}  rows={len(df)}  "
          f"wall_clock_total_s={round(time.time() - t0, 1)}", flush=True)


if __name__ == "__main__":
    main()
