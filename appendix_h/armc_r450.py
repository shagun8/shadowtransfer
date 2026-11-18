"""
Arm C — the rung-450 middle of the ladder, recomputed from STORED probability
maps. No GPU, no checkpoint load, no inference.

CONTRACT.md §8: the per-image probability maps persist for all 144 published
cells, so the ladder's middle rung costs a CPU pass over stored arrays rather
than a retraining run. This script reads
data/Test_img_probs/{mode}/{city}/{res}/{arch}/{variant}/*.npy plus the GT masks
and writes dumps/st/{run_id}/ in the same schema armc_train.py writes, via the
same shadowtransfer.armc_dump module. Foreground-only IoU at K=2 is the
headline; raw counts at K=0 and K=5 ride along.

One seed only. These are the published runs, which record no seed; run.json says
so in `seed_source` and the cells carry seed 0 nominally.

THE RUNG LABEL IS NOT UNIFORM ACROSS THESE 18 CELLS
---------------------------------------------------
Arm C assumes the published cells sit at rung 450. That holds for 15 of the 18,
and does NOT hold for MAMNet's three LOCO cells, which trained on 900 source
images. `mamnet/train.py` takes a `use_contrast` branch that builds
ShadowDatasetEnhanced directly and never passes `samples_per_dir`, so it loads
both source cities' full 450-image pools; OGLANet sets the same flag but routes
through `get_dataloaders`, which sets `samples_per_dir = 450 // 2 = 225`.

Evidence, from the published runs' own logs:
    mamnet  LOCO chicago/miami/phoenix HR : 112 batches/epoch at bs 8 -> 900
    oglanet LOCO chicago/miami/phoenix HR :  56 batches/epoch        -> 450
    dinov3  LOCO miami HR                 :  56 batches/epoch        -> 450
    every upper-bound cell with a log     :  56 batches/epoch        -> 450

A cell whose measured n_train contradicts its declared rung is written with
`rung: null`, the measured `n_train`, and a `rung_label_anomaly` block. A missing
value is recoverable downstream; a wrong one is not. The operator decides
whether MAMNet needs its own trained rung-450 LOCO cells.

REPRODUCTION TARGET
-------------------
CONTRACT.md §8: the target is the cell's OWN stored `test_results.json`,
foreground-only at K=2, within 0.01. That is
`detailed.boundary_tolerant.tolerant_2px.iou`. Tables 30-32 were computed at
K=5 on the two-class average (erratum E-7) and gate nothing here; the K=5 and
K=2 two-class numbers are still written, ungated, under the `info_*` keys.

n_train / n_val COME FROM args.json
-----------------------------------
Not from the training logs. If a cell's `args.json` does not record them the run
stops and prints the keys it does carry, rather than guessing a value that
downstream cannot tell apart from a measured one.
"""

import argparse
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_dump                        # noqa: E402
from shadowtransfer.armc_dump import sha256_file            # noqa: E402
from shadowtransfer.metrics import threshold_and_filter     # noqa: E402

FOLD_OF_HOLDOUT = {"phoenix": 0, "miami": 1, "chicago": 2}
RES_DIR = {"hr": "highres", "mr": "midres"}

# F3: some published run directories live only on /scratch. `oglanet_miami_
# highres_1` is absent under BASE_PATH and present there. Nothing is copied;
# the root actually used is recorded in run.json as `config_root`.
SCRATCH_BASE = Path(os.environ["ST_SCRATCH_ROOT"])

# F4: fields an args.json could plausibly use to record the training and
# validation set sizes. Checked in order; the first present wins.
N_TRAIN_FIELDS = ["n_train", "num_train", "train_size", "n_train_images",
                  "num_train_images", "train_samples", "n_samples_train",
                  "samples_per_dir_total", "train_set_size"]
N_VAL_FIELDS = ["n_val", "num_val", "val_size", "n_val_images",
                "num_val_images", "val_samples", "n_samples_val",
                "val_set_size"]

# F2: the P0a dumps' own `iou_fg_tol_pct` mean, for the two cells that already
# have a dump. A regenerated cell must land within 0.001 of its prior value.
# (The prior spec had these two the other way round; these are the corrected
# assignments read off the P0a dumps.)
P0A_GATE = {
    "mamnet-none-upper-r450-phoenix-hr-s0": 48.8840,
    "mamnet-vanilla-loco-r450-phoenix-hr-s0": 28.4405,
}
P0A_GATE_TOL = 0.001

# F1: CONTRACT.md §8 names the cell's own stored test_results.json as the
# reproduction target, foreground-only at K=2, within 0.01 mIoU. That is
# `detailed.boundary_tolerant.tolerant_2px.iou`. Tables 30-32 were computed at
# K=5 on the two-class average (erratum E-7) and are NOT the target; the K=5
# two-class number is still reported, ungated, for information.
REPRO_GATE_TOL = 0.01


def resolve_run_dir(arch, legacy_run_dir):
    """F3. Return (run_dir, config_root_name, config_root_path).

    Prefers BASE_PATH; falls back to /scratch when the directory is absent
    there. Read-only: nothing is copied.
    """
    for name, root in (("projects", BASE), ("scratch", SCRATCH_BASE)):
        d = root / "data" / arch / "outputs" / legacy_run_dir
        if (d / "args.json").exists():
            return d, name, str(root)
    raise SystemExit(
        f"FATAL run directory {legacy_run_dir} has no args.json under either "
        f"root:\n  {BASE / 'data' / arch / 'outputs' / legacy_run_dir}\n"
        f"  {SCRATCH_BASE / 'data' / arch / 'outputs' / legacy_run_dir}")


def resolve_n_train_val(cfg, cfg_path, cli_train, cli_val):
    """F4. Read the published training and validation set sizes from args.json.

    Returns (n_train, n_val, source, missing) where `missing` is a list of the
    quantities args.json does not carry. An explicit CLI value overrides and is
    recorded as such; the operator has to supply one only if args.json is
    silent.
    """
    if cli_train is not None and cli_val is not None:
        return cli_train, cli_val, "cli_override", []

    n_train = next((cfg[k] for k in N_TRAIN_FIELDS if k in cfg), None)
    n_val = next((cfg[k] for k in N_VAL_FIELDS if k in cfg), None)
    if cli_train is not None:
        n_train = cli_train
    if cli_val is not None:
        n_val = cli_val

    missing = []
    if n_train is None:
        missing.append("n_train")
    if n_val is None:
        missing.append("n_val")
    src = "args.json" if not missing else None
    if not missing and (cli_train is not None or cli_val is not None):
        src = "args.json + cli_override"
    return n_train, n_val, src, missing


def repro_target_from_test_results(run_dir):
    """F1. The cell's own foreground-only K=2 IoU, the CONTRACT §8 target."""
    p = run_dir / "test_results.json"
    if not p.exists():
        raise SystemExit(f"FATAL no test_results.json at {p}; CONTRACT.md §8 "
                         f"names it as the reproduction target")
    t = json.loads(p.read_text())
    try:
        v = t["detailed"]["boundary_tolerant"]["tolerant_2px"]["iou"]
    except (KeyError, TypeError):
        raise SystemExit(
            f"FATAL {p} has no detailed.boundary_tolerant.tolerant_2px.iou; "
            f"top-level keys {sorted(t)}")
    return float(v), str(p), sha256_file(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", required=True, choices=["loco", "upper"])
    ap.add_argument("--city", required=True)
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--published_miou", type=float, required=True,
                    help="the cell's Table 30-32 two-class mIoU")
    ap.add_argument("--n_train_published", type=int, default=None,
                    help="F4: normally read from the cell's args.json; pass "
                         "only to override, and only on operator instruction")
    ap.add_argument("--n_val_published", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    res_dir = RES_DIR[args.res]
    method = "vanilla" if args.mode == "loco" else "none"
    variant = "vanilla" if args.mode == "loco" else "base"
    declared_rung = 450

    expect_id = (f"{args.arch}-{method}-{args.mode}-r{declared_rung}-"
                 f"{args.city}-{args.res}-s0")
    if args.run_id != expect_id:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {expect_id}")

    prob_dir = (BASE / "data/Test_img_probs" / args.mode / args.city
                / res_dir / args.arch / variant)
    if args.mode == "loco":
        fold_dir = f"fold_{FOLD_OF_HOLDOUT[args.city]}_holdout_{args.city}"
        mask_dir = BASE / "data/Final_data_loco" / fold_dir / res_dir / "test/masks"
        legacy_run_dir = f"{args.arch}_loco_holdout_{args.city}_{res_dir}_1"
    else:
        fold_dir = None
        mask_dir = BASE / "data/Final_data_test" / args.city / res_dir / "test/masks"
        legacy_run_dir = f"{args.arch}_{args.city}_{res_dir}_1"
    # F3: BASE_PATH first, /scratch as fallback. Nothing is copied.
    ckpt_dir, config_root, config_root_path = resolve_run_dir(
        args.arch, legacy_run_dir)
    cfg = json.loads((ckpt_dir / "args.json").read_text())

    # F4: the published train/val set sizes come from the cell's own args.json.
    n_train_pub, n_val_pub, n_source, n_missing = resolve_n_train_val(
        cfg, ckpt_dir / "args.json", args.n_train_published,
        args.n_val_published)

    # F1: the reproduction target is the cell's own test_results.json.
    repro_target, repro_target_path, repro_target_sha = \
        repro_target_from_test_results(ckpt_dir)

    # ---- the rung label, decided from the measured training-set size ---------
    rung_ok = (n_train_pub == declared_rung)
    anomaly = None
    if not rung_ok and n_train_pub is not None:
        anomaly = {
            "declared_rung": declared_rung,
            "measured_n_train": n_train_pub,
            "measured_n_val": n_val_pub,
            "evidence": ("batches/epoch in the published run's own log, at "
                         "batch_size 8 with drop_last=True"),
            "mechanism": ("mamnet/train.py takes a use_contrast branch that "
                          "builds ShadowDatasetEnhanced directly and never "
                          "passes samples_per_dir, so it loads both source "
                          "cities' full 450-image train pools"),
            "consequence": ("this cell is not a rung-450 cell; rung is written "
                            "null so the ladder cannot silently absorb it"),
        }

    print("=" * 78)
    print(f"ARM C r450  run_id={args.run_id}")
    print(f"  probs        {prob_dir}")
    print(f"  masks        {mask_dir}")
    print(f"  config       {ckpt_dir/'args.json'}  [config_root={config_root}]")
    print(f"  repro target {repro_target:.4f}  (fg-only K=2, from "
          f"test_results.json detailed.boundary_tolerant.tolerant_2px.iou)")
    print(f"  n_train/val  {n_train_pub} / {n_val_pub}   source={n_source}")
    if n_missing:
        print(f"  F4 MISSING   args.json carries no {', '.join(n_missing)}; "
              f"its {len(cfg)} keys are: {sorted(cfg)}")
    print(f"  rung         {'450' if rung_ok else 'null  <-- ANOMALY'}")
    if args.run_id in P0A_GATE:
        print(f"  P0a gate     iou_fg_tol_pct must land within "
              f"{P0A_GATE_TOL} of {P0A_GATE[args.run_id]:.4f}")
    print("=" * 78, flush=True)

    # ---- stem sets: fail loudly on any mismatch, never intersect -------------
    prob_stems = {p.stem for p in prob_dir.glob("*.npy")}
    mask_stems = {p.stem for p in mask_dir.glob("*.png")}
    if prob_stems != mask_stems:
        only_p, only_m = sorted(prob_stems - mask_stems), sorted(mask_stems - prob_stems)
        raise SystemExit(
            f"FATAL stem-set mismatch for {args.run_id}\n"
            f"  n_prob={len(prob_stems)} n_mask={len(mask_stems)}\n"
            f"  probs-only ({len(only_p)}): {only_p[:10]}\n"
            f"  masks-only ({len(only_m)}): {only_m[:10]}")
    if len(prob_stems) != 150:
        raise SystemExit(
            f"FATAL expected 150 images for {args.run_id}, found {len(prob_stems)}")
    stems = sorted(prob_stems)
    print(f"stem sets identical, n = {len(stems)}", flush=True)

    if args.dry_run:
        print(f"--dry_run: resolved, nothing written. "
              f"F4 {'BLOCKED: ' + ', '.join(n_missing) if n_missing else 'ok'}")
        return

    # F4 is a hard stop in a real run: a guessed n_train is not recoverable
    # downstream, and CONTRACT.md §4 requires the field.
    if n_missing:
        raise SystemExit(
            f"FATAL F4 {args.run_id}: {ckpt_dir/'args.json'} records no "
            f"{', '.join(n_missing)}.\n"
            f"  args.json keys ({len(cfg)}): {sorted(cfg)}\n"
            f"  Not guessed. Supply --n_train_published / --n_val_published "
            f"on operator instruction, or point this at a source that records "
            f"them.")

    if args.limit:
        if args.out_root is None:
            raise SystemExit("FATAL --limit requires --out_root")
        stems = stems[:args.limit]
        print(f"SMOKE: truncated to {len(stems)} images", flush=True)

    geo = armc_dump.load_geo_meta(BASE)
    rows, probs = [], {}
    for stem in stems:
        prob = np.load(prob_dir / f"{stem}.npy").astype(np.float32)
        raw = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise SystemExit(f"FATAL unreadable mask {stem}")
        if raw.shape != prob.shape:
            raise SystemExit(f"FATAL shape mismatch {stem}: mask {raw.shape} "
                             f"prob {prob.shape}")
        # ShadowDatasetEnhanced: PIL 'L' -> ToTensor (/255) -> (> 0.5). Masks are
        # already 384x384 and strictly {0,255}, so this is exactly mask > 127.
        gt = (raw > 127).astype(np.uint8)
        pred = threshold_and_filter(prob, min_pixels=10)

        rows.append(armc_dump.perimage_row(
            args.run_id, stem, args.city, args.res, gt, prob, pred,
            geo.get(f"{stem}.png")))
        probs[stem] = np.load(prob_dir / f"{stem}.npy")      # float16, as stored

    import pandas as pd                                     # noqa: E402
    df = pd.DataFrame(rows)
    wall = time.time() - t0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
    miou_tol = float(df["miou_tol_pct"].mean())
    miou_k5 = float(
        (df["tp_k5"] / (df["tp_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100
         + df["tn_k5"] / (df["tn_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100)
        .mean() / 2.0)

    out = (Path(args.out_root) if args.out_root else BASE / "dumps") / "st" / args.run_id

    # F1: the gate. Foreground-only K=2 against the cell's own test_results.json.
    repro_delta = fg_iou_tol - repro_target
    repro_gate_pass = abs(repro_delta) <= REPRO_GATE_TOL

    # F2: the P0a regression gate, for the two cells that already have a dump.
    p0a_prior = P0A_GATE.get(args.run_id)
    p0a_delta = None if p0a_prior is None else fg_iou_tol - p0a_prior
    p0a_gate_pass = None if p0a_prior is None else abs(p0a_delta) <= P0A_GATE_TOL

    run = {
        "run_id": args.run_id,
        "dataset": "st",
        "arm": "C",
        "arch": args.arch,
        "method": method,
        "mode": args.mode,
        "rung": declared_rung if rung_ok else None,
        "rung_declared": declared_rung,
        "rung_label_anomaly": anomaly,
        "n_train": n_train_pub,
        "n_train_per_source_city": None,
        "n_val": n_val_pub,
        "n_train_val_source": n_source,
        "n_test": len(df),
        "holdout": args.city,
        "res": args.res,
        "seed": 0,
        "seed_source": "nominal; the legacy args.json records no seed",
        "n_seeds": 1,
        "n_seeds_note": ("this rung carries one seed only, from the published "
                         "run; the trained rungs 225 and 900 carry three"),

        "target_access": "source_only" if args.mode == "loco" else "target_labels",
        "target_access_detail": (
            "args.json has use_fda=false and fda_target_root=null; no held-out-city "
            "image pool is named"
            if args.mode == "loco" else
            "within-city upper bound: trained on the held-out city's own labelled "
            "train split"),

        "epochs_max": cfg.get("epochs"),
        "epochs_run": None,
        "steps_to_best": None,
        "best_val_metric": None,
        "early_stop_patience": cfg.get("early_stopping_patience"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),

        "metric_module_sha": sha256_file(BASE / "python/shadowtransfer/metrics.py"),
        "metric_module_path": "python/shadowtransfer/metrics.py",
        "dump_module_sha": sha256_file(BASE / "python/shadowtransfer/armc_dump.py"),
        "recompute_script_sha": sha256_file(Path(__file__).resolve()),
        "git_hash": None,
        "git_hash_reason": "project root is not a git repository (CLAUDE.md)",
        "config_sha256": sha256_file(ckpt_dir / "args.json"),
        "config_verbatim": cfg,
        "config_root": config_root,
        "config_root_path": config_root_path,
        "config_path": str(ckpt_dir / "args.json"),

        "cluster": "delta",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "gpu": None,
        "compute": ("cpu-only pass over stored probability maps; no checkpoint "
                    "loaded, no inference run. Delta has no CPU allocation on "
                    "this account, so the job holds a GPU it does not use."),
        "compute_note": ("CPU-only workload submitted on gpuA100x4; no CPU "
                         "account exists on this allocation. GPU allocated and "
                         "unused."),
        "node": os.environ.get("SLURMD_NODENAME"),
        "wall_clock_s": round(wall, 2),
        "peak_rss_mb": round(peak_rss_mb, 1),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),

        "ckpt_path": str(ckpt_dir / "checkpoint_best.pth"),
        "parent_run_id": None,
        "legacy_run_dir": legacy_run_dir,
        "structure_cols": "deferred",

        "source_probs_dir": str(prob_dir.relative_to(BASE)),
        "source_mask_dir": str(mask_dir.relative_to(BASE)),
        "probs_are_prefilter": True,
        "prediction_recipe": "p > 0.5 then filter_small_predictions(min_pixels=10)",
        "band_width_headline": 2,
        "band_widths_written": [0, 2, 5],
        "headline_metric": "foreground-only IoU at K=2 (column iou_tol)",
        "headline_fg_iou_tol_pct": fg_iou_tol,

        # ---- F1: THE reproduction gate, CONTRACT.md §8 -----------------------
        "repro_target_source": "test_results.json",
        "repro_target_field": "detailed.boundary_tolerant.tolerant_2px.iou",
        "repro_target_path": repro_target_path,
        "repro_target_sha256": repro_target_sha,
        "repro_target_fg_iou_k2": repro_target,
        "repro_fg_iou_k2": fg_iou_tol,
        "repro_delta_fg_iou_k2": repro_delta,
        "repro_gate_tol": REPRO_GATE_TOL,
        "repro_gate_pass": repro_gate_pass,
        "repro_metric_definition": (
            "foreground-only IoU with the +-2 px don't-care band, per "
            "CONTRACT.md §8. The recomputed value is the mean over the 150 "
            "per-image `iou_fg_tol_pct`; the target is the cell's own stored "
            "test_results.json value. Gate 0.01. Reported, never adjusted."),

        # ---- for information only, NOT a gate (erratum E-7) -----------------
        "published_miou_tol": args.published_miou,
        "published_miou_source": "Table 30-32 (two-class mean IoU at K=5)",
        "info_miou_k5_twoclass": miou_k5,
        "info_delta_miou_k5_vs_table3032": miou_k5 - args.published_miou,
        "info_miou_k2_twoclass": miou_tol,
        "info_delta_miou_k2_vs_table3032": miou_tol - args.published_miou,
        "info_note": (
            "Tables 30-32 were computed at K=5 on the two-class average, so "
            "they are not this arm's reproduction target and nothing gates on "
            "them. Carried so the two definitions stay comparable without "
            "another pass."),

        # ---- F2: regression against the P0a dumps ---------------------------
        "p0a_prior_fg_iou_k2": p0a_prior,
        "p0a_delta": p0a_delta,
        "p0a_gate_tol": P0A_GATE_TOL,
        "p0a_gate_pass": p0a_gate_pass,

        "aurc_definition": "aurc_trapz19",
        "aurc_definition_note": (
            "CONTRACT.md section 3 leaves the canonical AURC blank. "
            "aurc_fg/bg/all use aurc_trapz19; aurc_*_mean20 carry the SIB-trainer "
            "plain mean over 20 points. Neither is declared canonical here."),

        "smoke": bool(args.limit),
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    flags = []
    if anomaly:
        flags.append("rung label disputed; see rung_label_anomaly")
    if int(df["empty_fg_valid"].sum()):
        flags.append(f"empty_fg_valid on {int(df['empty_fg_valid'].sum())} images")
    if not repro_gate_pass:
        flags.append(f"REPRODUCTION GATE FAILED: fg IoU K=2 {fg_iou_tol:.4f} vs "
                     f"test_results.json {repro_target:.4f}, delta "
                     f"{repro_delta:+.4f} > {REPRO_GATE_TOL}")
    if p0a_gate_pass is False:
        flags.append(f"P0a REGRESSION: {fg_iou_tol:.4f} vs prior "
                     f"{p0a_prior:.4f}, delta {p0a_delta:+.4f} > {P0A_GATE_TOL}")
    run["sanity_flags"] = flags

    # F2 is a code-regression check: if it fails, this script no longer
    # reproduces its own prior output and the other 16 cells are not
    # trustworthy either. Fail the job; do not write a .DONE.
    if p0a_gate_pass is False:
        armc_dump.mark_failed(out, flags[-1])
        print(f"\n[{args.run_id}] {flags[-1]}", flush=True)
        raise SystemExit(f"FATAL P0a regression gate failed for {args.run_id}")

    armc_dump.write_dump(out, rows, probs, run)

    print(f"\n[{args.run_id}]")
    print(f"  images                  {len(df)}")
    print(f"  fg IoU K=2 (headline)   {fg_iou_tol:.4f}")
    print(f"  repro target (K=2 fg)   {repro_target:.4f}  delta "
          f"{repro_delta:+.4f}  gate {REPRO_GATE_TOL} -> "
          f"{'PASS' if repro_gate_pass else 'FAIL'}")
    if p0a_prior is not None:
        print(f"  P0a prior               {p0a_prior:.4f}  delta "
              f"{p0a_delta:+.4f}  gate {P0A_GATE_TOL} -> "
              f"{'PASS' if p0a_gate_pass else 'FAIL'}")
    print(f"  info: 2-class K=5       {miou_k5:.4f}  Table 30-32 "
          f"{args.published_miou}  delta {miou_k5 - args.published_miou:+.4f} "
          f"(ungated)")
    print(f"  info: 2-class K=2       {miou_tol:.4f}  Table 30-32 "
          f"{args.published_miou}  delta {miou_tol - args.published_miou:+.4f} "
          f"(ungated)")
    print(f"  config_root             {config_root}")
    print(f"  empty_fg_valid          {int(df['empty_fg_valid'].sum())}")
    print(f"  sanity flags            {flags or 'none'}")
    print(f"  wall {wall:.1f}s ({wall/len(df):.3f}s/img)  peak RSS {peak_rss_mb:.0f} MB")
    print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
