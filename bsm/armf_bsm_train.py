#!/usr/bin/env python
"""Arm F / BSM -- one cell of the 12-cell grid: mamnet, oglanet or dinov3.

CrossEarth's BSM cell is a separate script, armf_bsm_crossearth_train.py:
CrossEarth's training loop is armf_crossearth_engine.train(), not the
Trainer class these three architectures share, so the two paths cannot
reuse one body.

Writes, in this order:
  dumps/st/{run_id}/perimage.parquet, probs.npz, run.json, .DONE  (CONTRACT.md)
  runs/armF_bsm_{arch}_{city}_highres/results.json

Reuses armc_train's data selection, model/optimizer/scheduler/early-stop and
dump-writing machinery (import armc_train as ac), exactly as armd_train.py
already does for its own new method. The one BSM-specific addition is
armf_bsm_module.attach_bsm_hook on trainer.model, installed right after
Trainer construction and removed again before test inference. No published
file is edited.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import armc_train as ac                                        # noqa: E402
from shadowtransfer import armc_dump, armc_runtime              # noqa: E402
from shadowtransfer import armf_bsm_arch_lib as lib             # noqa: E402
from shadowtransfer import armf_bsm_module as bsm               # noqa: E402

ARM = "F_bsm"
RUNG = 450
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--arch", required=True, choices=["mamnet", "oglanet", "dinov3"])
    ap.add_argument("--holdout", required=True, choices=sorted(ac.FOLD_OF_HOLDOUT))
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.5, help="BSM mixing coefficient")
    ap.add_argument("--adain_weights_dir",
                    default=str(BASE / "data" / "bsm" / "adain_weights"))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke_epochs", type=int, default=0)
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout_s", type=int, default=1200)
    ap.add_argument("--out_root", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    smoke = bool(args.smoke_epochs or args.smoke_test)
    if smoke and args.out_root is None:
        raise SystemExit("FATAL --smoke_* requires --out_root")

    res_dir = ac.RES_DIR[args.res]
    expect_id = lib.bsm_run_id(args.arch, args.holdout, args.res, RUNG, args.seed)
    if args.run_id != expect_id:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {expect_id}")

    dump_dir = (Path(args.out_root) if args.out_root else BASE / "dumps") / "st" / args.run_id
    if (dump_dir / ".DONE").exists() and not args.force:
        print(f"[{args.run_id}] .DONE exists at {dump_dir} -- nothing to do")
        return

    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet", dump_dir / "probs.npz",
        dump_dir / "run.json", dump_dir / ".DONE",
    ], label="dump")

    train_pairs, val_pairs, test_pairs, train_cities, n_per_city = \
        ac.resolve_selection("loco", args.holdout, RUNG, res_dir, args.seed)

    with armc_runtime.ReadGuard(60, "published config"):
        cfg, cfg_path = ac.published_config(args.arch, "loco", args.holdout, res_dir)

    out_dir_parent = (Path(args.out_root) / "train" / args.run_id if smoke
                      else BASE / "data" / args.arch / "outputs" / "armF_bsm" / args.run_id)
    comparators = lib.comparator_paths(BASE, args.arch, args.holdout)

    print("=" * 78)
    print(f"ARM F / BSM  run_id     {args.run_id}")
    print(f"  arch/holdout/res/seed {args.arch} / {args.holdout} / {args.res} / {args.seed}")
    print(f"  alpha (mixing coef)   {args.alpha}")
    print(f"  train/val/test        {len(train_pairs)}/{len(val_pairs)}/{len(test_pairs)}  {n_per_city}")
    print(f"  published config      {cfg_path}")
    print(f"  output_dir            {out_dir_parent}")
    print(f"  dump                  {dump_dir}")
    print(f"  comparators           {json.dumps(comparators)}")
    print(f"  slurm job             {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    if smoke:
        print(f"  SMOKE epochs<={args.smoke_epochs} test<={args.smoke_test}")
    print("=" * 78, flush=True)

    if len(train_pairs) != RUNG:
        raise SystemExit(f"FATAL train set is {len(train_pairs)}, expected {RUNG}")
    if len(val_pairs) != ac.N_VAL:
        raise SystemExit(f"FATAL val set is {len(val_pairs)}, expected {ac.N_VAL}")
    if len(test_pairs) != ac.N_TEST:
        raise SystemExit(f"FATAL test set is {len(test_pairs)}, expected {ac.N_TEST}")
    if set(train_pairs) & set(val_pairs):
        raise SystemExit("FATAL train/val overlap")
    if set(train_pairs) & set(test_pairs) or set(val_pairs) & set(test_pairs):
        raise SystemExit("FATAL test leakage into train or val")
    heldout_hits = sum(1 for c, _ in train_pairs + val_pairs if c == args.holdout)
    if heldout_hits:
        raise SystemExit(f"FATAL {heldout_hits} held-out-city rows in train/val")

    if args.dry_run:
        print("\n--dry_run: resolved, nothing written.")
        print(f"  first 3 train: {train_pairs[:3]}")
        return

    import torch                                                # noqa: E402
    seed_sites = ac.seed_everything(args.seed)
    gen = torch.Generator().manual_seed(args.seed + 424242)

    arch_root = BASE / ac.ARCH_DIR[args.arch]
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    for _m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_m]
    archtrain = __import__(ac.ARCH_TRAIN_MODULE[args.arch])
    armc_runtime.silence_worker_side_effects(archtrain)

    from argparse import Namespace
    _defaults = ac.arch_parser_defaults(archtrain, ac.ARCH_TRAIN_MODULE[args.arch])
    a = {**_defaults, **cfg}
    a["output_dir"] = str(out_dir_parent)
    a["device"] = "cuda"
    a["eval_only"] = False
    a["mode"] = "loco"
    a["base_data_root"] = str(BASE / "data" / "Final_data_test") + "/"
    a["resolution"] = res_dir
    a["fold_id"] = ac.FOLD_OF_HOLDOUT[args.holdout]
    a["data_root"] = None
    for k in ("comparison_inference_dir", "comparison_data_root"):
        if k in a and a[k]:
            a[k] = str(BASE / "data" / Path(a[k].rstrip("/")).name) + "/"
    if args.arch == "dinov3":
        a["weights_path"] = str(ac.DINOV3_WEIGHTS)
    if args.smoke_epochs:
        a["epochs"] = args.smoke_epochs
    armc_runtime.disable_periodic_checkpoints(a)

    patience_epochs, spe_expected = ac.armc_patience_epochs(RUNG, a["batch_size"])
    a["early_stopping_patience"] = patience_epochs
    if not args.smoke_epochs:
        a["epochs"] = ac.ARMC_EPOCHS_MAX
    a["seed"] = args.seed
    a["resume"] = None
    trainer_args = Namespace(**a)

    trainer = archtrain.Trainer(trainer_args)
    run_out = Path(trainer.output_dir)

    # ---- BSM: the one method-specific addition -------------------------------
    encoder, decoder, bsm_report = bsm.load_adain(args.adain_weights_dir, device="cuda")
    bsm_state = {}
    hook = bsm.attach_bsm_hook(trainer.model, IMAGENET_MEAN, IMAGENET_STD, 1.0,
                               encoder, decoder, args.alpha, gen,
                               channels_to_mix=3, state=bsm_state)
    print(f"[bsm] style_net attached: {json.dumps(bsm_report)}", flush=True)
    if bsm_report["trainable_params"] != 0:
        raise SystemExit("FATAL AdaIN style net has trainable parameters")

    from torch.utils.data import DataLoader
    img_size = trainer_args.img_size
    nw, bs = trainer_args.num_workers, trainer_args.batch_size

    ds_train = ac.build_dataset(args.arch, train_pairs, "train", res_dir, img_size, True)
    ds_val = ac.build_dataset(args.arch, val_pairs, "val", res_dir, img_size, False)
    ds_test = ac.build_dataset(args.arch, test_pairs, "test", res_dir, img_size, False)
    if args.smoke_test:
        for attr in ("img_files", "img_paths", "mask_paths"):
            setattr(ds_test, attr, getattr(ds_test, attr)[:args.smoke_test])

    g = torch.Generator()
    g.manual_seed(args.seed)
    trainer.dataloaders = {
        "train": DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=True, drop_last=True, generator=g,
                            worker_init_fn=ac.dl_worker_init),
        "val": DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw,
                          pin_memory=True, worker_init_fn=ac.dl_worker_init),
        "test": DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=nw,
                           pin_memory=True, worker_init_fn=ac.dl_worker_init),
    }
    steps_per_epoch = len(trainer.dataloaders["train"])
    print(f"\nDATALOADERS train={len(ds_train)} val={len(ds_val)} test={len(ds_test)} "
          f"steps/epoch={steps_per_epoch}", flush=True)
    if not smoke and steps_per_epoch != spe_expected:
        raise SystemExit(f"FATAL steps/epoch {steps_per_epoch} != {spe_expected}")

    t_train = time.time()
    trainer.train()
    train_wall = time.time() - t_train
    print(f"[bsm] last hook state: {bsm_state}", flush=True)

    best_ckpt = run_out / "checkpoint_best.pth"
    if not best_ckpt.exists():
        raise SystemExit(f"FATAL no best checkpoint at {best_ckpt}")
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    best_epoch = int(ck["epoch"])
    epochs_run = len(trainer.train_losses)
    steps_to_best = best_epoch * steps_per_epoch
    best_val_metric = float(ck.get("best_decision_miou", float("nan")))
    del ck

    from utils.postprocessing import filter_small_predictions
    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference")
        trainer.load_checkpoint(str(best_ckpt))
        model = trainer.model
        model.eval()
        hook.remove()  # belt-and-suspenders: BSM cannot fire during eval anyway
        device = trainer.device

        geo = armc_dump.load_geo_meta(BASE)
        rows, probs = [], {}
        with torch.no_grad():
            for batch in trainer.dataloaders["test"]:
                images = batch["image"].to(device)
                masks = batch["mask"]
                fname = batch["filename"][0]
                stem = Path(fname).stem
                logits = model(images)
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
                filtered = filter_small_predictions(logits, min_pixels=10)
                pred = torch.argmax(filtered, dim=1)[0].cpu().numpy().astype(np.uint8)
                gt = masks[0].numpy().astype(np.uint8)
                rows.append(armc_dump.perimage_row(
                    args.run_id, stem, args.holdout, args.res, gt, prob, pred,
                    geo.get(fname)))
                probs[stem] = prob.astype(np.float16)

        tbx.phase("metrics+dump")
        import pandas as pd                                     # noqa: E402
        df = pd.DataFrame(rows)
        fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
        miou_tol = float(df["miou_tol_pct"].mean()) if "miou_tol_pct" in df else None
        peak_gb = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        wall = time.time() - t0

        run_cfg = {
            "epochs_max": trainer_args.epochs, "batch_size": bs,
            "optimizer_lr": trainer_args.lr,
            "decision_metric": ("tolerant mIoU (+-2px)" if
                                getattr(trainer_args, "eval_boundary_tolerant", False)
                                else "strict per-image mIoU"),
            "bsm_hook": "forward_pre_hook on the arch's top-level model, gated "
                        "on model.training; mixes the first 3 (RGB) input "
                        "channels via the frozen AdaIN encoder-decoder after "
                        "inverting this arch's own ImageNet input "
                        "normalisation, and reapplying it after; any 4th "
                        "(contrast) channel passes through unmodified.",
        }
        method_cfg = lib.method_config(
            bsm_report, args.alpha, bs, bsm_state,
            "ours (this arch's own published train-time augmentation, "
            "H-flip/V-flip p=0.5, unchanged); paper silent on its own base "
            "augmentation set beyond the style-mixing step itself")

        caveats = [lib.BSM_CAVEAT]
        if smoke:
            caveats.append("SMOKE PASS: truncated epochs/test images. Not a result.")

        run = {
            "run_id": args.run_id, "dataset": "st", "arm": ARM,
            "weakness_ids": lib.WEAKNESS_IDS,
            "arch": args.arch, "architecture": args.arch,
            "method": "bsm", "mode": "loco", "cell_type": "loco", "rung": RUNG,
            "n_train": len(ds_train), "n_train_per_source_city": n_per_city,
            "n_val": len(ds_val), "n_test": len(df),
            "holdout": args.holdout, "city": args.holdout,
            "res": args.res, "seed": args.seed, "seeds": seed_sites,
            "target_access": "source_only",
            "target_access_detail": "no external or held-out-city imagery "
                                    "enters training; the style source is the "
                                    "training batch itself, shuffled.",
            "epochs_max": trainer_args.epochs, "epochs_run": epochs_run,
            "steps_to_best": steps_to_best, "best_val_metric": best_val_metric,
            "early_stop_patience": patience_epochs, "lr": trainer_args.lr,
            "batch_size": bs,
            "metric_module_sha": armc_dump.sha256_file(
                str(BASE / "shadowtransfer/metrics.py")),
            "git_hash": None,
            "git_hash_reason": "project root is not a git repository",
            "config_sha256": armc_dump.sha256_bytes(
                json.dumps(run_cfg, sort_keys=True).encode()),
            "config_verbatim": run_cfg, "method_config": method_cfg,
            "implementation": {
                "source": "our_reimplementation_from_paper",
                "paper": "Luo, Ji, Wei, JSTARS 16:4122-4138, 2023",
                "official_code_exists": False,
                "style_net": bsm_report,
            },
            "comparators": comparators,
            "cluster": "delta", "partition": "gpuA100x4",
            "gpu": torch.cuda.get_device_name(0),
            "wall_clock_s": round(wall, 1),
            "train_wall_clock_s": round(train_wall, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "ckpt_path": str(best_ckpt), "parent_run_id": None,
            "legacy_run_dir": None, "structure_cols": "deferred",
            "compute_note": "no CPU account on this allocation; the whole "
                            "cell including inference runs on gpuA100x4",
            "sanity_checks": {
                "no_heldout_city_in_train_or_val": heldout_hits == 0,
                "heldout_hits": heldout_hits,
                "bsm_disabled_at_eval": True,
                "style_net_no_grad": bsm_report["trainable_params"] == 0,
                "shuffle_pairs_distinct": bsm_state,
                "steps_to_best_not_truncated": epochs_run < trainer_args.epochs,
            },
            "caveats": caveats, "peak_gpu_mem_gb": peak_gb,
            "requested_time": os.environ.get("REQUESTED_TIME"),
            "used_time": round(wall, 1),
            "test": {"miou_tol2": miou_tol, "iou_fg_tol2": fg_iou_tol,
                     "n_images": len(df)},
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        armc_dump.write_dump(dump_dir, rows, probs, run)

        runs_dir = BASE / "runs" / f"armF_bsm_{args.arch}_{args.holdout}_highres"
        runs_dir.mkdir(parents=True, exist_ok=True)
        results = dict(run)
        results["per_image_dump_path"] = str(dump_dir)
        results["generated_utc"] = run["written_utc"]
        lib.write_results_json(runs_dir, results)
        (runs_dir / ".DONE").touch()

    print(f"[done] {args.run_id} iou_fg_tol2 {fg_iou_tol:.4f} "
          f"steps_to_best {steps_to_best} wall {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
