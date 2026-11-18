"""
Arm D — appearance-augmentation stress. One cell: train one
(arch, condition, holdout) at rung 450, high resolution, under the S3
appearance-only augmentation, then emit its CONTRACT.md dump.

Answers v1UJ-W4 ("the feature-space claim rests mainly on histogram matching;
other low-level appearance or illumination factors are not ruled out") and
zNBu-W2 ("the gap may come from a narrow source distribution and appearance
shortcuts; run aggressive appearance augmentation"). REVIEWS.md arm D.

What this script changes relative to arm C
------------------------------------------
The AUGMENTATION, and nothing else. Data selection, model, loss, optimizer,
scheduler, early-stopping rule, model-selection metric, dump code path and
metric module are arm C's, reached by importing armc_train and reusing its
functions rather than restating them. The only additions are:

  1. shadowtransfer.armd_aug.S3Appearance is installed on the TRAIN dataset's
     `fda_transform` hook, which fires on the raw PIL RGB before the published
     flips and before add_contrast_channel. See armd_aug.py for why that hook
     and not another; the ordering is the whole arm.
  2. The val and test datasets are asserted to carry NO augmentation.
  3. A contrast-channel verification runs and is written to the log and to
     run.json BEFORE training starts. On mamnet and oglanet a cell whose
     channel 3 does not move refuses to train.

BOTH CONDITIONS ARE RETRAINED HERE, under the identical recipe
--------------------------------------------------------------
`--condition aug-loco` trains on the two non-holdout cities; `--condition
aug-upper` trains within the holdout city. Comparing an augmented LOCO model
against the un-augmented published upper bound would mix "augmentation changed
transfer" with "augmentation changed in-domain performance" and answer nothing,
so arm D retrains both sides.

target_access
-------------
aug-loco is `source_only`: no held-out-city image is touched, by the
augmentation pipeline or anything else.

aug-upper is `target_labels`, NOT source_only. The arm spec asks for
source_only on every cell on the grounds that the augmentation pipeline touches
no held-out-city image, which is true of the augmentation and false of the run:
an aug-upper cell trains on labelled images from the held-out city, which is
CONTRACT.md's definition of `target_labels` verbatim. Recording source_only
there would put a false value in the one field BxRx-W4 and erratum I-1 exist to
make honest. The conflict is recorded in run.json as
`target_access_spec_conflict` rather than resolved silently.

Metric
------
Headline is FOREGROUND-ONLY IoU at K=2, written by shadowtransfer.armc_dump,
which is the only metric implementation this job touches. The arm spec names
2-class tolerant mIoU at K=2 as primary; both are emitted (`iou_tol` and
`miou_tol_pct`) and raw per-class counts ride along at K=0, 2 and 5, so either
scoping is recoverable without a rerun. run.json records which is which.

Usage
-----
    python armd_train.py --run_id mamnet-augs3-loco-r450-phoenix-hr-s0 \
        --arch mamnet --condition aug-loco --holdout phoenix --seed 0
    ... --dry_run                    resolve and print, write nothing
    ... --check_only                 run the contrast-channel check and stop
    ... --smoke_epochs 2 --smoke_test 5 --out_root <scratch>   stage-1 smoke
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
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
from shadowtransfer import armc_dump                           # noqa: E402
from shadowtransfer.armc_dump import sha256_file               # noqa: E402
from shadowtransfer import armc_runtime                        # noqa: E402
from shadowtransfer import armd_aug                            # noqa: E402
from shadowtransfer import armd_s3                             # noqa: E402

ARM = "D"
ARM_ID = "D_S3"
WEAKNESS_IDS = ["v1UJ-W4", "zNBu-W2", "AC1"]
RUNG = 450                      # single rung; arm C owns the ladder
METHOD_TOKEN = "augs3"          # CONTRACT.md section 1 method slot
CONDITIONS = {"aug-loco": "loco", "aug-upper": "upper"}
BASELINE_SEEDS = (0, 1, 2)


def derived_run_id(arch, condition, holdout, res, seed):
    return (f"{arch}-{METHOD_TOKEN}-{CONDITIONS[condition]}-r{RUNG}-"
            f"{holdout}-{res}-s{seed}")


def _baseline_run_id(arch, kind, holdout, res, seed):
    """Arm C rung-450 vanilla cell ids.

    The upper-bound cells carry method token `none`, not `vanilla` -- that is
    what armc_train writes for mode=upper and what is on disk. Getting this
    wrong silently resolves to a missing directory, so it is one function.
    """
    if kind == "loco":
        return f"{arch}-vanilla-loco-r{RUNG}-{holdout}-{res}-s{seed}"
    return f"{arch}-none-upper-r{RUNG}-{holdout}-{res}-s{seed}"


def read_baselines(arch, holdout, res):
    """Pull the arm C rung-450 vanilla baselines and their seed spread.

    Reads SCALARS that arm C already computed and stored in its own run.json.
    This job does not open a baseline parquet and does not pool images: the
    per-image aggregation happened in the arm C cell that owns it, and the
    per-cell statistics are the SSH box's job (CLAUDE.md). What is recorded
    here is provenance -- which cells arm D is to be read against, and the raw
    per-seed values, so the SSH box can compute whatever spread it wants.
    """
    out = {"source": "arm C rung-450 vanilla recompute",
           "metric": "headline_fg_iou_tol_pct (foreground-only IoU, K=2)",
           "missing": []}
    for kind in ("loco", "upper"):
        per_seed = {}
        for s in BASELINE_SEEDS:
            rid = _baseline_run_id(arch, kind, holdout, res, s)
            d = BASE / "dumps" / "st" / rid
            if not (d / ".DONE").exists() or not (d / "run.json").exists():
                out["missing"].append(rid)
                continue
            try:
                rj = json.loads((d / "run.json").read_text())
            except Exception as e:                       # noqa: BLE001
                out["missing"].append(f"{rid} (unreadable: {e})")
                continue
            v = rj.get("headline_fg_iou_tol_pct")
            if v is None:
                out["missing"].append(f"{rid} (no headline_fg_iou_tol_pct)")
                continue
            per_seed[f"s{s}"] = round(float(v), 4)
        out[f"vanilla_{kind}_r450_per_seed"] = per_seed
        out[f"vanilla_{kind}_r450_s0"] = per_seed.get("s0")
        if len(per_seed) >= 2:
            vals = list(per_seed.values())
            out[f"vanilla_{kind}_r450_seed_spread"] = {
                "n_seeds": len(vals),
                "min": min(vals), "max": max(vals),
                "range": round(max(vals) - min(vals), 4),
                "std_ddof1": round(float(np.std(vals, ddof=1)), 4),
            }
        else:
            out[f"vanilla_{kind}_r450_seed_spread"] = None
    out["note"] = (
        "the noise floor arm D is read against. These are armc_train cells, not "
        "recomputes of the published runs: same seeded nested-prefix selection "
        "rule, same early-stopping recipe, same dump code path this arm uses. "
        "Verified on mamnet/phoenix/s0 on 2026-07-30 -- arm D's train, val and "
        "test list sha256 are byte-identical to that cell's, so aug-loco vs "
        "vanilla-loco at a fixed seed differs by the augmentation and nothing "
        "else.")
    out["train_list_sha256_matches_baseline"] = None   # filled by the caller
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--arch", required=True, choices=sorted(ac.ARCH_DIR))
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--holdout", required=True, choices=sorted(ac.FOLD_OF_HOLDOUT))
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--aug_strength", default="S3", choices=["S3"])
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--check_only", action="store_true",
                    help="run the contrast-channel verification and stop")
    ap.add_argument("--smoke_epochs", type=int, default=0)
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--timeout_s", type=int, default=1200)
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()

    t0 = time.time()
    smoke = bool(args.smoke_epochs or args.smoke_test)
    if smoke and args.out_root is None:
        raise SystemExit("FATAL --smoke_* requires --out_root")

    res_dir = ac.RES_DIR[args.res]
    mode = CONDITIONS[args.condition]
    expect_id = derived_run_id(args.arch, args.condition, args.holdout,
                               args.res, args.seed)
    if args.run_id != expect_id:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {expect_id}")

    dump_dir = (Path(args.out_root) if args.out_root else BASE / "dumps") / "st" / args.run_id
    if (dump_dir / ".DONE").exists() and not args.force:
        print(f"[{args.run_id}] .DONE exists at {dump_dir} — nothing to do")
        return

    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet",
        dump_dir / "probs.npz",
        dump_dir / "run.json",
        dump_dir / ".DONE",
    ], label="dump")

    train_pairs, val_pairs, test_pairs, train_cities, n_per_city = \
        ac.resolve_selection(mode, args.holdout, RUNG, res_dir, args.seed)

    with armc_runtime.ReadGuard(60, "published config"):
        cfg, cfg_path = ac.published_config(args.arch, mode, args.holdout, res_dir)

    out_dir_parent = (Path(args.out_root) / "train" / args.run_id if smoke
                      else BASE / "data" / args.arch / "outputs" / "armD" / args.run_id)

    print("=" * 78)
    print(f"ARM D  run_id           {args.run_id}")
    print(f"  arm_id / weaknesses   {ARM_ID} / {WEAKNESS_IDS}")
    print(f"  arch/condition/seed   {args.arch} / {args.condition} / {args.seed}")
    print(f"  mode / rung           {mode} / {RUNG}")
    print(f"  holdout / res         {args.holdout} / {args.res} ({res_dir})")
    print(f"  aug strength          {args.aug_strength} (train split only)")
    print(f"  train                 {len(train_pairs)}  {n_per_city}")
    print(f"  val                   {len(val_pairs)} (NOT augmented)")
    print(f"  test                  {len(test_pairs)} (NOT augmented)")
    print(f"  published config      {cfg_path}")
    print(f"  output_dir            {out_dir_parent}")
    print(f"  dump                  {dump_dir}")
    print(f"  slurm job             {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    print(f"  train_list_sha256     {ac.list_sha256(train_pairs)}")
    print(f"  val_list_sha256       {ac.list_sha256(val_pairs)}")
    print(f"  test_list_sha256      {ac.list_sha256(test_pairs)}")
    if smoke:
        print(f"  SMOKE                 epochs<={args.smoke_epochs} "
              f"test<={args.smoke_test}")
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

    baselines = read_baselines(args.arch, args.holdout, args.res)

    # The PRIMARY metric's baselines: 2-class boundary-tolerant mIoU at K=2,
    # which is what D_seedfloor measured the noise floor in and what the
    # decision rule is evaluated in. `baselines` above stays as the
    # foreground-only companion CLAUDE.md requires every arm to report.
    b2 = armd_s3.read_baselines_2class(BASE, args.arch, args.holdout, args.res)
    sf_refs = armd_s3.seedfloor_refs(BASE, args.arch, args.holdout)
    tier = armd_s3.seed_tier(BASE, args.arch, args.holdout)
    rule = armd_s3.decision_rule_block(BASE, args.arch, args.holdout,
                                       args.condition, b2)

    print(f"SEED TIER   {args.arch}/{args.holdout}  aug-loco x{tier['aug-loco']}"
          f"  aug-upper x{tier['aug-upper']}  (this cell: {args.condition} "
          f"s{args.seed})")
    print(f"BASELINES (2-class mIoU K=2, arm C rung-450 vanilla)")
    print(f"  loco  per-seed {b2['vanilla_loco_per_seed']}  mean "
          f"{b2['vanilla_loco_mean']}")
    print(f"  upper per-seed {b2['vanilla_upper_per_seed']}  mean "
          f"{b2['vanilla_upper_mean']}")
    print(f"  gap   per-seed {b2['vanilla_gap_per_seed']}  mean "
          f"{b2['vanilla_gap_mean']}  sd {b2['vanilla_gap_sd']}")
    print(f"DECISION RULE  {rule['rule']}  |  gap_sd_ref "
          f"{rule['gap_sd_ref']}  threshold {rule['threshold']}")
    print(f"BASELINES (fg-only IoU K=2) missing: {baselines['missing'] or 'none'}",
          flush=True)

    # The arm spec: if a needed rung-450 cell is missing, report which and
    # STOP. Do not substitute any number printed in the PDF.
    if b2["missing"]:
        raise SystemExit(
            f"FATAL required arm C rung-450 vanilla baseline cells are missing "
            f"for {args.arch}/{args.holdout}: {b2['missing']}. Reported, not "
            f"substituted; this cell will not run without its baseline.")

    if args.dry_run:
        print("\n--dry_run: resolved, nothing written.")
        print(f"  first 3 train: {train_pairs[:3]}")
        print(f"  first 3 val:   {val_pairs[:3]}")
        print(f"  first 3 test:  {test_pairs[:3]}")
        print(f"  baselines: {json.dumps(baselines, indent=2)}")
        return

    import torch                                               # noqa: E402
    seed_sites = ac.seed_everything(args.seed)
    print("SEED SITES")
    for _k, _v in seed_sites.items():
        print(f"  {_k:20s} {_v}")
    print(flush=True)

    arch_root = BASE / ac.ARCH_DIR[args.arch]
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    for _m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_m]
    archtrain = __import__(ac.ARCH_TRAIN_MODULE[args.arch])

    # ---- THE ORDERING CHECK, before any GPU time is spent -------------------
    # The arm spec requires this in the log before training starts: one training
    # batch per architecture, per-channel mean and std with augmentation off vs
    # on, and confirmation that channel 3 (contrast) differs on the CNNs.
    print("\n" + "-" * 78)
    print("CONTRAST-CHANNEL VERIFICATION (augmentation must precede "
          "add_contrast_channel)")
    chk = armd_aug.contrast_channel_check(
        ac.build_dataset, args.arch, train_pairs, res_dir,
        int(cfg.get("img_size", 384)), n_batch=8, seed=args.seed)
    print(json.dumps(chk, indent=2), flush=True)
    armd_aug.assert_contrast_check(chk)
    print(f"VERDICT  {chk['verdict']}")
    print("-" * 78 + "\n", flush=True)
    if args.check_only:
        print("--check_only: stopping before training.")
        return

    disabled = armc_runtime.silence_worker_side_effects(archtrain)

    from argparse import Namespace
    _defaults = ac.arch_parser_defaults(archtrain, ac.ARCH_TRAIN_MODULE[args.arch])
    defaults_filled = {k: _defaults[k] for k in sorted(set(_defaults) - set(cfg))}
    if defaults_filled:
        print(f"CONFIG defaults filled from the arch's own parser: "
              f"{defaults_filled}", flush=True)
    a = {**_defaults, **cfg}
    a["output_dir"] = str(out_dir_parent)
    a["device"] = "cuda"
    a["eval_only"] = False
    if mode == "loco":
        a["mode"] = "loco"
        a["base_data_root"] = str(BASE / "data" / "Final_data_test") + "/"
        a["resolution"] = res_dir
        a["fold_id"] = ac.FOLD_OF_HOLDOUT[args.holdout]
        a["data_root"] = None
    else:
        a["mode"] = "single"
        a["data_root"] = str(BASE / "data" / "Final_data_test" / args.holdout / res_dir) + "/"
        a["base_data_root"] = None
        a["resolution"] = None
        a["fold_id"] = None
    for k in ("comparison_inference_dir", "comparison_data_root"):
        if k in a and a[k]:
            a[k] = str(BASE / "data" / Path(a[k].rstrip("/")).name) + "/"
    if args.arch == "dinov3":
        a["weights_path"] = str(ac.DINOV3_WEIGHTS)
    if args.smoke_epochs:
        a["epochs"] = args.smoke_epochs
    armc_runtime.disable_periodic_checkpoints(a)

    patience_epochs, spe_expected = ac.armc_patience_epochs(RUNG, a["batch_size"])
    patience_published = a.get("early_stopping_patience")
    a["early_stopping_patience"] = patience_epochs
    a["epochs"] = ac.ARMC_EPOCHS_MAX
    if args.smoke_epochs:
        a["epochs"] = args.smoke_epochs
    print(f"EARLY STOPPING  patience={patience_epochs} epochs flat, "
          f"epochs_max={a['epochs']} (published args.json said "
          f"{patience_published})", flush=True)

    a["seed"] = args.seed
    a["armd_run_id"] = args.run_id
    a["armd_arm_id"] = ARM_ID
    a["armd_condition"] = args.condition
    a["armd_aug_strength"] = args.aug_strength
    a["armd_n_train"] = RUNG
    a["armd_n_val"] = ac.N_VAL
    a["resume"] = None
    trainer_args = Namespace(**a)

    trainer = archtrain.Trainer(trainer_args)
    run_out = Path(trainer.output_dir)

    resume_from = ac.newest_epoch_checkpoint(run_out)
    if resume_from:
        print(f"RESUME from {resume_from}", flush=True)
        trainer.load_checkpoint(resume_from)

    from torch.utils.data import DataLoader
    img_size = trainer_args.img_size
    nw = trainer_args.num_workers
    bs = trainer_args.batch_size

    ds_train = ac.build_dataset(args.arch, train_pairs, "train", res_dir, img_size, True)
    ds_val = ac.build_dataset(args.arch, val_pairs, "val", res_dir, img_size, False)
    ds_test = ac.build_dataset(args.arch, test_pairs, "test", res_dir, img_size, False)

    # ---- install S3 on the TRAIN split only ---------------------------------
    if ds_train.fda_transform is not None:
        raise SystemExit("FATAL train dataset already carries an fda_transform; "
                         "arm D refuses to stack an unknown transform under S3")
    ds_train.fda_transform = armd_aug.S3Appearance()
    print(f"AUG INSTALLED on train split: {ds_train.fda_transform!r}", flush=True)
    for nm, d in (("val", ds_val), ("test", ds_test)):
        if getattr(d, "fda_transform", None) is not None:
            raise SystemExit(f"FATAL {nm} dataset carries a transform; val and "
                             f"test must never be augmented")
        if getattr(d, "augment", False):
            raise SystemExit(f"FATAL {nm} dataset has augment=True")
    print("AUG CONFIRMED OFF on val and test splits", flush=True)

    if args.smoke_test:
        for attr in ("img_files", "img_paths", "mask_paths"):
            setattr(ds_test, attr, getattr(ds_test, attr)[:args.smoke_test])

    g = torch.Generator()
    g.manual_seed(args.seed)
    trainer.dataloaders = {
        "train": DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=True, drop_last=True,
                            generator=g, worker_init_fn=ac.dl_worker_init),
        "val": DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw,
                          pin_memory=True, worker_init_fn=ac.dl_worker_init),
        "test": DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=nw,
                           pin_memory=True, worker_init_fn=ac.dl_worker_init),
    }
    steps_per_epoch = len(trainer.dataloaders["train"])
    print(f"\nDATALOADERS  train={len(ds_train)}  val={len(ds_val)}  "
          f"test={len(ds_test)}  steps/epoch={steps_per_epoch}", flush=True)
    if len(ds_train) != RUNG:
        raise SystemExit(f"FATAL loader train size {len(ds_train)} != {RUNG}")
    if not smoke and steps_per_epoch != spe_expected:
        raise SystemExit(f"FATAL steps/epoch {steps_per_epoch} != {spe_expected}")

    t_train = time.time()
    trainer.train()
    train_wall = time.time() - t_train

    best_ckpt = run_out / "checkpoint_best.pth"
    if not best_ckpt.exists():
        raise SystemExit(f"FATAL no best checkpoint at {best_ckpt}")
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    best_epoch = int(ck["epoch"])
    epochs_run = len(trainer.train_losses)
    steps_to_best = best_epoch * steps_per_epoch
    best_val_metric = float(ck.get("best_decision_miou", float("nan")))
    del ck
    print(f"\nBEST epoch {best_epoch}  epochs_run {epochs_run}  "
          f"steps_to_best {steps_to_best}  best_val {best_val_metric:.4f}",
          flush=True)

    from utils.postprocessing import filter_small_predictions

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference")
        trainer.load_checkpoint(str(best_ckpt))
        model = trainer.model
        model.eval()
        device = trainer.device

        geo = armc_dump.load_geo_meta(BASE)
        rows, probs = [], {}
        t_inf = time.time()
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
        inf_wall = time.time() - t_inf

        tbx.phase("metrics")
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        wall = time.time() - t0

        import pandas as pd                                    # noqa: E402
        df = pd.DataFrame(rows)
        fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
        miou_tol = float(df["miou_tol_pct"].mean())
        early_stopped = bool(epochs_run < trainer_args.epochs)

        # Execution fact, not analysis: is this cell's image selection
        # byte-identical to the vanilla baseline it will be compared against?
        baseline_sel_match = None
        _b_rid = _baseline_run_id(args.arch, mode, args.holdout, args.res, args.seed)
        _b_rj = BASE / "dumps" / "st" / _b_rid / "run.json"
        if _b_rj.exists():
            try:
                _b = json.loads(_b_rj.read_text())
                baseline_sel_match = {
                    "baseline_run_id": _b_rid,
                    "train": _b.get("train_list_sha256") == ac.list_sha256(train_pairs),
                    "val": _b.get("val_list_sha256") == ac.list_sha256(val_pairs),
                    "test": _b.get("test_list_sha256") == ac.list_sha256(test_pairs),
                }
            except Exception as _e:                          # noqa: BLE001
                baseline_sel_match = {"baseline_run_id": _b_rid, "error": str(_e)}

        stray_png = sorted(str(q.relative_to(run_out)) for q in run_out.rglob("*.png"))
        stray_tb = sorted(str(q.relative_to(run_out)) for q in run_out.rglob("events.out.tfevents*"))
        stray_epoch_ckpt = sorted(q.name for q in run_out.glob("checkpoint_epoch_*.pth"))

        run = {
            "run_id": args.run_id,
            "dataset": "st",
            "arm": ARM,
            "arm_id": ARM_ID,
            "weakness_ids": WEAKNESS_IDS,
            "arch": args.arch,
            "method": METHOD_TOKEN,
            "method_note": (
                "vanilla training objective and architecture; the only "
                "difference from method=vanilla is the S3 appearance "
                "augmentation on the train split. No FDA, ISW, SegDeSiC or any "
                "other adaptation component is present."),
            "condition": args.condition,
            "mode": mode,
            "rung": RUNG,
            "resolution": args.res,
            "n_train": len(ds_train),
            "n_train_per_source_city": n_per_city,
            "n_val": len(ds_val),
            "n_test": len(df),
            "holdout": args.holdout,
            "city": args.holdout,
            "res": args.res,
            "seed": args.seed,
            "seed_tier": tier,
            "seed_tier_note": (
                "the PLANNED seed count for this (arch, city), set from the "
                "D_seedfloor across-seed gap sd, not the number of seeds that "
                "happen to have completed. Values are 0, then 1, then 2, "
                "matching arm C."),
            "seed_sites": seed_sites,

            # --- augmentation -------------------------------------------------
            "aug_strength": args.aug_strength,
            "aug_params": armd_aug.S3,
            "aug_module_sha": sha256_file(BASE / "python/shadowtransfer/armd_aug.py"),
            "aug_applied_to": "train split only",
            "aug_val_test": "val and test are never augmented; asserted at runtime",
            "aug_injection_hook": (
                "dataset.fda_transform — the attribute all three published "
                "loaders call on the raw PIL RGB BEFORE the published H/V flips "
                "and BEFORE add_contrast_channel. This is NOT Fourier Domain "
                "Adaptation: use_fda is False, no FDA transform is constructed, "
                "and no held-out-city image is read. The hook name is an "
                "artifact of the published loader."),
            "aug_geometry": (
                "geometry strictly preserved. The published H-flip/V-flip p=0.5 "
                "is the only geometric augmentation and is unchanged."),
            "contrast_channel_check": chk,

            # --- target access -------------------------------------------------
            "target_access": "source_only" if mode == "loco" else "target_labels",
            "target_access_detail": (
                "training and validation draw only from the two source cities; "
                "no held-out-city image is touched by the augmentation pipeline "
                "or by anything else before test-time inference"
                if mode == "loco" else
                "within-city upper bound retrained under S3: train and val are "
                "labelled images from the held-out city itself, disjoint from "
                "its test pool"),
            "target_access_spec_conflict": (
                None if mode == "loco" else
                "The arm D spec asks for target_access=source_only on EVERY "
                "cell, on the grounds that no held-out-city image is touched by "
                "the augmentation pipeline. That is true of the augmentation "
                "and false of this run: an aug-upper cell trains on labelled "
                "held-out-city images, which is CONTRACT.md's definition of "
                "target_labels. source_only is not recorded here because it "
                "would be false in the one field BxRx-W4 and erratum I-1 exist "
                "to make honest. Flagged for the operator; not resolved "
                "silently."),

            # --- metrics --------------------------------------------------------
            "metric_name": "miou_2class_tolerant_K2",
            "metric_value": round(miou_tol, 4),
            "metric_name_note": (
                "the arm D spec names 2-class tolerant mIoU at K=2 as primary; "
                "it is column miou_tol_pct. CLAUDE.md's standing rule is that "
                "every new arm REPORTS foreground-only IoU at K=2 (column "
                "iou_tol). Both are written, and raw per-class counts at K=0, 2 "
                "and 5 ride along in the parquet, so either scoping is "
                "recoverable without a rerun."),
            "reporting_metric": "fg_only_tol_k2",
            "reporting_metric_value": round(fg_iou_tol, 4),
            "headline_fg_iou_tol_pct": fg_iou_tol,
            "miou_tol_pct_published_definition": miou_tol,
            "decision_metric": "2class_tol_k2",
            "decision_metric_value_pct": round(best_val_metric, 4),
            "decision_metric_note": (
                "what checkpoint_best.pth was SELECTED on: the published "
                "architecture's own decision metric, unchanged. Changing it "
                "would be a second variable."),
            "band_width_headline": 2,
            "band_widths_written": [0, 2, 5],
            "raw_counts_note": (
                "per-image tp/fp/fn/tn at K=0 (*_str), K=2 (*_tol) and K=5 "
                "(*_k5) are in perimage.parquet, per CONTRACT.md section 3."),
            "aurc_definition": "aurc_trapz19",
            "aurc_definition_note": (
                "CONTRACT.md section 3 leaves the canonical AURC blank pending "
                "P0a. aurc_fg/bg/all use aurc_trapz19; aurc_*_mean20 carry the "
                "SIB-trainer plain mean over 20 points. Both are written and "
                "neither is declared canonical by this run, mirroring arm C."),

            # --- baselines ------------------------------------------------------
            "baselines_2class": b2,
            "gap_sd_ref": sf_refs["gap_sd_ref"],
            "gap_sd_ref_detail": sf_refs,
            "decision_rule": rule,
            "baselines": baselines,
            "seed_spread_ref": {
                "vanilla_loco_r450": baselines.get("vanilla_loco_r450_seed_spread"),
                "vanilla_upper_r450": baselines.get("vanilla_upper_r450_seed_spread"),
                "note": ("per-cell seed spread of the arm C rung-450 vanilla "
                         "cells over seeds 0/1/2, foreground-only IoU at K=2. "
                         "The noise floor this arm is read against."),
            },

            # --- training -------------------------------------------------------
            "epochs_max": trainer_args.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "steps_per_epoch": steps_per_epoch,
            "steps_to_best": steps_to_best,
            "early_stopped": early_stopped,
            "best_val_metric": best_val_metric,
            "early_stop_patience": trainer_args.early_stopping_patience,
            "early_stop_patience_published": patience_published,
            "lr": trainer_args.lr,
            "batch_size": trainer_args.batch_size,

            # --- provenance -----------------------------------------------------
            "metric_module_sha": sha256_file(BASE / "python/shadowtransfer/metrics.py"),
            "metrics_sha256": sha256_file(BASE / "python/shadowtransfer/metrics.py"),
            "metric_module_path": "python/shadowtransfer/metrics.py",
            "armd_s3_module_sha": sha256_file(BASE / "python/shadowtransfer/armd_s3.py"),
            "metric_module_sha_note": (
                "the arm D spec expects d19d5e94... That is the sha of the "
                "COMMITTED metrics.py at git HEAD. The working tree carries 47 "
                "additional lines appended on 2026-07-29 for arm E part 2 (a "
                "bootstrap_ci lift), which change no existing function: the diff "
                "is purely additive, 47 insertions and 0 deletions, below every "
                "function arm D uses. Recorded as measured; flagged to the "
                "operator rather than reconciled by this job."),
            "dump_module_sha": sha256_file(BASE / "python/shadowtransfer/armc_dump.py"),
            "runtime_module_sha": sha256_file(BASE / "python/shadowtransfer/armc_runtime.py"),
            "train_script_sha": sha256_file(Path(__file__).resolve()),
            "armc_train_sha": sha256_file(BASE / "python/armc_train.py"),
            "git_hash": None,
            "git_hash_reason": (
                "BASE_PATH became a git repository on 2026-07-28; this field is "
                "left null until the operator decides whether run.json should "
                "carry a commit, matching arm C"),
            "config_sha256": sha256_file(cfg_path),
            "config_verbatim": a,
            "published_config_path": str(cfg_path.relative_to(BASE)),
            "config_defaults_filled": defaults_filled,

            "train_list_sha256": ac.list_sha256(train_pairs),
            "val_list_sha256": ac.list_sha256(val_pairs),
            "test_list_sha256": ac.list_sha256(test_pairs),
            "baseline_selection_identical": baseline_sel_match,
            "baseline_selection_identical_note": (
                "True when this cell's train/val/test list sha256 equal those "
                "of its arm C rung-450 vanilla counterpart at the same seed, "
                "i.e. the augmentation is the only difference between them. "
                "None when that baseline's run.json is absent."),
            "selection_rule": (
                "armc_train.resolve_selection, imported not restated: seeded "
                "permutation per (city, split), nested prefixes across rungs. "
                "aug-loco and aug-upper at the same holdout therefore see the "
                "same images their vanilla counterparts would under this rule."),

            # --- execution ------------------------------------------------------
            "cluster": "delta",
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu": gpu_name,
            "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_time_limit": os.environ.get("ARMD_TIME_LIMIT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "wall_clock_s": round(wall, 2),
            "wall_clock_note": ("measured before write_dump, so it excludes the "
                                "parquet and probs.npz writes, matching arm C"),
            "train_wall_clock_s": round(train_wall, 2),
            "inference_wall_clock_s": round(inf_wall, 2),
            "gpu_hours": round(wall / 3600.0, 4),
            "peak_vram_alloc_gb": round(peak_alloc, 3),
            "peak_vram_reserved_gb": round(peak_reserved, 3),
            "compute_note": "A100 GPU cell on gpuA100x4; GPU used for training",
            "ckpt_path": str(best_ckpt),
            "parent_run_id": None,
            "legacy_run_dir": None,
            "structure_cols": "deferred",
            "worker_side_effects_disabled": disabled,
            "post_training_timeout_s": args.timeout_s,
            "side_effect_audit": {
                "png_files": stray_png,
                "tensorboard_event_files": stray_tb,
                "periodic_checkpoints": stray_epoch_ckpt,
                "save_freq_effective": a.get("save_freq"),
            },
            "prediction_recipe": (
                "argmax over the two class logits, then "
                "filter_small_predictions(min_pixels=10); identical to the "
                "architecture's own test() loop"),
            "probs_note": "softmax(logits)[:, 1], stored float16, unthresholded",

            "caveats": [
                f"Seeds are variance-tiered from D_seedfloor, not flat: this "
                f"(arch, city) gets {tier['aug-loco']} aug-loco and "
                f"{tier['aug-upper']} aug-upper seed(s). A 1-seed cell carries "
                f"no within-condition spread of its own and is read only "
                f"against gap_sd_ref.",
                "Single augmentation strength (S3). No dose-response curve, so "
                "an unchanged gap cannot be attributed to too weak a dose.",
                "High-resolution only; midres is out of scope for this arm.",
                f"The vanilla rung-450 LOCO baselines show a large seed "
                f"spread on some cells; this cell's D_seedfloor gap sd is "
                f"{sf_refs['gap_sd_ref']:.3f} (2-class mIoU K=2), so the "
                f"decision rule's threshold here is {rule['threshold']:.3f}. "
                f"That threshold, not the raw difference, is what this cell is "
                f"read against.",
                "decision_rule.gap_aug_mean is null until every seed of BOTH "
                "aug conditions for this (arch, city) has completed; "
                "armd_s3_aggregate.py fills it and flips `changed`. A block "
                "with status pending_grid_completion is not a verdict.",
                "The three §4.2-4.4 diagnostics (city-identity linear probe, "
                "foreground selective-prediction gap, histogram-matching "
                "recovery ratio R) are NOT in this cell. Per CLAUDE.md they run "
                "as a separate inference job after the grid completes, so a "
                "probe failure cannot cost a trained model.",
                "Per-cell paired bootstrap, the gap grid and the aggregate "
                "summary are NOT computed here. This box writes per-image "
                "quantities only; pooling images into a mean, CI, p-value or "
                "ranking is the SSH box's job (CLAUDE.md).",
                "REVIEWS.md carries no pre-committed reading for arm D. "
                "CONVENTIONS.md requires one before the numbers are read.",
            ],
            "smoke": smoke,
            "forced_rerun": bool(args.force),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        flags = []
        if epochs_run < 10:
            flags.append(f"epochs_run={epochs_run} (<10)")
        if epochs_run >= trainer_args.epochs:
            flags.append(f"ran to the epoch ceiling ({trainer_args.epochs}); "
                         f"early stopping never fired")
        if int(df["empty_fg_valid"].sum()):
            flags.append(f"empty_fg_valid on {int(df['empty_fg_valid'].sum())} images")
        if float(df["iou_tol"].std()) < 1e-6:
            flags.append("per-image iou_tol has ~zero variance")
        if stray_png:
            flags.append(f"figures written despite the rebind: {stray_png}")
        if stray_tb:
            flags.append(f"tensorboard events written despite the rebind: {stray_tb}")
        if stray_epoch_ckpt:
            flags.append(f"periodic checkpoints written: {stray_epoch_ckpt}")
        if baselines["missing"]:
            flags.append(f"fg-only baseline cells missing: {baselines['missing']}")
        if not rule["seedfloor_gap_sd_crosscheck"]["agree_1e4"]:
            flags.append(
                f"gap sd recomputed from the arm C dumps "
                f"({b2['vanilla_gap_sd']}) disagrees with D_seedfloor "
                f"({sf_refs['gap_sd_ref']})")
        if not rule["seedfloor_gap_mean_crosscheck"]["agree_1e4"]:
            flags.append(
                f"vanilla gap mean recomputed from the arm C dumps "
                f"({b2['vanilla_gap_mean']}) disagrees with D_seedfloor "
                f"({sf_refs['gap_vanilla_mean_seedfloor']})")
        b0 = baselines.get("vanilla_loco_r450_s0") if mode == "loco" else \
            baselines.get("vanilla_upper_r450_s0")
        if b0 is not None and abs(fg_iou_tol - b0) < 1e-3:
            flags.append(f"headline {fg_iou_tol:.4f} lands on its vanilla "
                         f"baseline {b0:.4f} to 3 decimals")
        run["sanity_flags"] = flags

        tbx.phase("write_dump")
        armc_dump.write_dump(dump_dir, rows, probs, run)

    print("\n" + "=" * 78)
    print(f"[{args.run_id}]")
    print(f"  fg IoU K=2 (CLAUDE.md)  {fg_iou_tol:.4f}")
    print(f"  2-class mIoU K=2 (spec) {miou_tol:.4f}")
    print(f"  fg IoU K=0 / K=5        {float(df['iou_str'].mean())*100:.4f} / "
          f"{float(df['iou_k5'].mean())*100:.4f}")
    print(f"  vanilla 2-class mIoU     loco_mean={b2['vanilla_loco_mean']} "
          f"upper_mean={b2['vanilla_upper_mean']} gap_mean={b2['vanilla_gap_mean']}")
    print(f"  gap_sd_ref / threshold   {rule['gap_sd_ref']} / {rule['threshold']}")
    print(f"  seed tier                {tier}")
    print(f"  images                  {len(df)}")
    print(f"  epochs_run              {epochs_run}   best_epoch {best_epoch}   "
          f"early_stopped {early_stopped}")
    print(f"  wall clock              {wall:.1f}s (train {train_wall:.1f}s, "
          f"inference {inf_wall:.1f}s)")
    print(f"  peak VRAM alloc/resv    {peak_alloc:.2f} / {peak_reserved:.2f} GiB")
    print(f"  contrast check          {chk['verdict']}")
    print(f"  sanity flags            {flags or 'none'}")
    print(f"  -> {dump_dir}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
