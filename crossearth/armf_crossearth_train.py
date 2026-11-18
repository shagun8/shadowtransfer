#!/usr/bin/env python
"""Arm F / CrossEarth — one cell of the 12-cell gap row.

  upper/vanilla x3   loco/vanilla x3   loco/fda x3   loco/sib x3

Weakness IDs v1UJ-W3 and AC1 for every cell, plus AC3 and v1UJ-W2 for the fda
and sib cells. CrossEarth is a new architecture row: it compares only against
itself, and no cell here is compared against Tables 30-32 or against any
stored test_results.json.

Writes, in this order:
  dumps/st/{run_id}/perimage.parquet, probs.npz, run.json, .DONE  (CONTRACT.md)
  runs/armF_crossearth_{cell_type}_{method}_{city}_highres/results.json

Every quantity comes from shadowtransfer.metrics through
shadowtransfer.armc_dump. Nothing in this file computes a metric.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# release layout: shadowtransfer/ sits at the repository root, one level up.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from shadowtransfer import armc_dump, armc_runtime  # noqa: E402
from shadowtransfer import armf_crossearth_cells as cells  # noqa: E402
from shadowtransfer import armf_crossearth_engine as eng  # noqa: E402
from shadowtransfer import armf_crossearth_runtime as rt  # noqa: E402
from shadowtransfer import armf_crossearth_sib as sibmod  # noqa: E402

PIXEL_MEAN = [123.675, 116.28, 103.53]
PIXEL_STD = [58.395, 57.12, 57.375]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", required=True)
    p.add_argument("--repo_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt_sha256", required=True)
    p.add_argument("--ckpt_bytes", type=int, required=True)
    p.add_argument("--cell_type", required=True, choices=["upper", "loco"])
    p.add_argument("--method", required=True, choices=["vanilla", "fda", "sib"])
    p.add_argument("--city", required=True, choices=list(cells.CITIES))
    p.add_argument("--res", default="hr")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--fda_L", type=float, default=0.01)
    p.add_argument("--fda_pool_size", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--post_timeout_s", type=int, default=2400)
    p.add_argument("--smoke", action="store_true",
                   help="truncated end-to-end pass: few iters, 8 test images")
    return p.parse_args()


def dump_preflight(dump_dir, run_id):
    """Prove the CONTRACT row, parquet, npz and json all write, in seconds.

    A dump-path failure discovered after 1.7 h of training costs the cell; the
    same failure discovered here costs nothing. Runs before the model is built.
    """
    probe = Path(dump_dir) / ".preflight"
    if probe.exists():
        shutil.rmtree(probe, ignore_errors=True)
    rng = np.random.default_rng(0)
    gt = (rng.random((384, 384)) > 0.7).astype(np.uint8)
    prob = rng.random((384, 384)).astype(np.float32)
    from shadowtransfer.metrics import threshold_and_filter
    row = armc_dump.perimage_row(run_id, "preflight", "chicago", "hr", gt,
                                 prob, threshold_and_filter(prob))
    armc_dump.write_dump(probe, [row], {"preflight": prob.astype(np.float16)},
                         {"preflight": True})
    shutil.rmtree(probe, ignore_errors=True)
    print(f"[preflight] dump path writes cleanly: {dump_dir}", flush=True)
    return sorted(row.keys())


def build_datasets(args, base):
    pipeline = cells.build_train_pipeline(args.crop)
    tr_i, tr_m, _ = cells.load_split(base, args.cell_type, args.city, "train",
                                     args.res)
    va_i, va_m, _ = cells.load_split(base, args.cell_type, args.city, "val",
                                     args.res)
    te_i, te_m, _ = cells.load_split(base, args.cell_type, args.city, "test",
                                     args.res)
    pool, pool_dir, pool_avail = None, None, None
    if args.method == "fda":
        pool, pool_dir, pool_avail = cells.build_fda_pool(
            base, args.city, args.fda_pool_size, args.seed, args.res)
        print(f"[fda] pool {len(pool)} of {pool_avail} unlabelled images from "
              f"{pool_dir}", flush=True)
    limit = 8 if args.smoke else None
    train_ds = cells.ShadowTrainDataset(tr_i, tr_m, pipeline, fda_pool=pool,
                                        fda_L=args.fda_L, seed=args.seed)
    val_ds = cells.ShadowEvalDataset(va_i, va_m, limit=limit)
    test_ds = cells.ShadowEvalDataset(te_i, te_m, limit=limit)
    meta = {"n_train": len(tr_i), "n_val": len(va_i), "n_test": len(te_i),
            "fda_pool": pool, "fda_pool_dir": pool_dir,
            "fda_pool_available": pool_avail}
    return train_ds, val_ds, test_ds, meta


def main():
    args = parse_args()
    base = args.base_path
    t_start = time.time()
    run_id = cells.run_id_of(args.cell_type, args.method, args.city, args.res)
    dump_dir = Path(base) / "dumps" / "st" / run_id
    runs_dir = Path(base) / "runs" / cells.cell_dirname(
        args.cell_type, args.method, args.city, args.res)
    ckpt_dir = Path(base) / "data" / "crossearth" / "outputs" / "armF" / run_id
    for d in (dump_dir, runs_dir, ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "checkpoint_best.pth"

    print(f"[cell] run_id {run_id}", flush=True)
    armc_runtime.preflight_write_targets(
        base, [dump_dir / "perimage.parquet", dump_dir / "probs.npz",
               dump_dir / "run.json", runs_dir / "results.json", ckpt_path],
        label="armF dump")

    seeds = rt.seed_everything(args.seed)
    rt.add_repo_to_path(args.repo_root)
    deform_patched = rt.patch_msda_fp32()

    # --- provenance: re-verify the backbone at job start -------------------
    sha = rt.sha256_file(args.ckpt)
    nbytes = os.path.getsize(args.ckpt)
    if sha != args.ckpt_sha256 or nbytes != args.ckpt_bytes:
        raise RuntimeError(f"backbone mismatch: sha {sha} bytes {nbytes}")
    print(f"[ckpt] sha256 verified {sha} ({nbytes} bytes)", flush=True)

    # --- splits and the no-held-out-city assertion --------------------------
    sanity = cells.assert_no_heldout(base, args.cell_type, args.city, args.res)
    print(f"[splits] {json.dumps(sanity)}", flush=True)
    root = cells.split_root(base, args.cell_type, args.city, args.res)
    print(f"[splits] resolved root {root}", flush=True)

    dump_cols = dump_preflight(dump_dir, run_id)
    train_ds, val_ds, test_ds, dmeta = build_datasets(args, base)

    # --- model --------------------------------------------------------------
    cfg = rt.build_cfg(args.repo_root, args.ckpt, num_classes=2, crop=args.crop)
    model = rt.build_model(cfg, device="cuda")
    model.train()
    params = rt.param_report(model)
    if not params["backbone_frozen_except_reins"]:
        raise RuntimeError(f"backbone freeze failed: {params}")
    print(f"[params] {json.dumps(params)}", flush=True)
    rt.gpu_banner("after load")

    sib, sib_report, n_sib = None, None, 0
    if args.method == "sib":
        sib, n_sib, sib_report = sibmod.attach_sib(model, PIXEL_MEAN,
                                                   PIXEL_STD, args.crop)
        model.train()
        print(f"[sib] {json.dumps(sib_report)}", flush=True)

    # --- train --------------------------------------------------------------
    val_curve, best, steps_to_best, stats = eng.train(
        model, train_ds, val_ds, cells.collate, runs_dir, args.iters,
        args.batch_size, args.val_interval, args.seed, ckpt_path, sib=sib,
        num_workers=args.num_workers)
    train_wall = time.time() - t_start
    peak_gb = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)

    # --- everything after training is time-boxed ----------------------------
    with eng.phase_timeout(args.post_timeout_s, "inference+metrics+dump"):
        ck = torch.load(ckpt_path, map_location="cpu")
        eng.load_trainable_state_dict(model, ck["state_dict"])
        model.eval()

        ids, gts, probs = eng.predict_probs(model, test_ds, cells.collate,
                                            batch_size=args.batch_size)
        test_scores = eng.score_split(gts, probs)
        print(f"[test] {json.dumps(test_scores)}", flush=True)

        sib_sel = None
        if args.method == "sib":
            _, vg, vp = eng.predict_probs(model, val_ds, cells.collate,
                                          batch_size=args.batch_size)
            agg, grid, n_used = sibmod.selective_shadow_error(vg, vp)
            c_star, val_full, target = sibmod.find_c_star_val(
                agg, grid, sibmod.TAU) if agg is not None else (1.0, None, None)
            t_agg, t_grid, t_n = sibmod.selective_shadow_error(gts, probs)
            j = int(np.argmin(np.abs(t_grid - c_star)))
            sib_sel = {
                "c_star_val": float(c_star),
                "val_err_at_c1": val_full,
                "val_target": target,
                "val_n_images_with_curve": n_used,
                "err_at_c1": float(t_agg[-1]) if t_agg is not None else None,
                "err_at_c_star": float(t_agg[j]) if t_agg is not None else None,
                "delta": (float(t_agg[j] - t_agg[-1])
                          if t_agg is not None else None),
                "test_n_images_with_curve": t_n,
                "coverage_grid_step": 0.05, "tau": sibmod.TAU,
            }
            print(f"[sib_selective] {json.dumps(sib_sel)}", flush=True)

        # --- CONTRACT dump --------------------------------------------------
        geo = armc_dump.load_geo_meta(base)
        rows, prob_store = [], {}
        from shadowtransfer.metrics import threshold_and_filter
        for image_id, gt, prob in zip(ids, gts, probs):
            pred = threshold_and_filter(prob)
            rows.append(armc_dump.perimage_row(
                run_id, image_id, args.city, args.res, gt, prob, pred,
                geo.get(f"{image_id}.png")))
            prob_store[image_id] = prob.astype(np.float16)

        truncated = (steps_to_best is not None
                     and steps_to_best > args.iters - args.val_interval)
        fg = test_scores["pred_fg_fraction_mean"]
        saturated = args.iters >= 2000 and (fg < 0.01 or fg > 0.99)
        caveats = [cells.RELEASE_CAVEAT,
                   "xformers is not installed; the DINOv2 layers run their "
                   "plain-attention fallback under XFORMERS_DISABLED=1.",
                   "The released train pipeline (RandomChoiceResize / "
                   "RandomCrop cat_max_ratio 0.75 / RandomFlip / "
                   "PhotoMetricDistortion) is ported from its 512 base to our "
                   "384 patch; transform types, order and all other arguments "
                   "are the released ones.",
                   "PolyLR is kept at the released end=40000 rather than "
                   "rescaled to the 10,000-iteration budget, so the LR "
                   "multiplier at the final iteration is 0.772."]
        if truncated:
            caveats.append(
                f"steps_to_best={steps_to_best} lands within one validation "
                f"interval of the {args.iters}-iteration cap: possibly "
                f"truncated.")
        if saturated:
            caveats.append(
                f"mean predicted foreground fraction {fg:.4f} is saturated at "
                f"iteration {args.iters}; this is the stage-0 watch item and "
                f"is a finding, not a warm-up artefact.")
        if args.smoke:
            caveats.append("SMOKE PASS: truncated iterations and 8 eval "
                           "images. Not a result.")

        target_access = ("unlabeled_target" if args.method == "fda"
                         else "source_only")
        detail = ("FDA input preprocessing samples up to "
                  f"{args.fda_pool_size} unlabelled images from the held-out "
                  f"city's train split; no held-out label and no held-out "
                  f"image enters the loss."
                  if args.method == "fda" else
                  "no held-out-city data of any kind enters training or "
                  "validation; asserted on filenames and on split metadata.")
        if args.method == "sib":
            detail += (" The coverage threshold c*_val is fitted on the "
                       "source-city validation split only.")

        run_cfg = {
            "crop": args.crop, "batch_size": args.batch_size,
            "precision": "bf16_except_deformable_attn_fp32",
            "iters_max": args.iters, "val_interval": args.val_interval,
            "lr": args.lr, "optimizer": "AdamW(wd=0.05, eps=1e-8, "
                                        "betas=(0.9,0.999)) + PolyLR(power=0.9, end=40000)",
            "frozen_backbone": True,
            "n_trainable_params": params["n_trainable_params"] + n_sib,
            "n_frozen_params": params["n_frozen_params"],
            "n_sib_params": n_sib,
            "deform_attn_fp32": deform_patched,
            "numpy_trapezoid_shim": eng.NUMPY_TRAPEZOID_SHIM,
            "deviations_from_published": [
                "crop 512 -> 384 (our native patch)",
                "batch 1 -> 4",
                "iterations 30000 -> 10000 (~89 epochs at 450 images, our "
                "100-epoch protocol; 30000 projects to 5.3 h and exceeds the "
                "3:59:59 partition ceiling)",
                "bf16 autocast with mmcv deformable attention forced to fp32 "
                "(mmcv 2.1.0 has no bf16 ms_deform_attn_forward_cuda kernel)",
                "validation is source-city only; the released DG configs put "
                "the target domain in the validation dataloader",
                "test_cfg slide -> whole (image size equals crop size)",
                "num_classes -> 2",
            ],
        }
        method_cfg = {}
        if args.method == "fda":
            method_cfg = {"L": args.fda_L,
                          "b": cells.fda_half_width(args.crop, args.fda_L),
                          "swapped_region_px": 2 * cells.fda_half_width(
                              args.crop, args.fda_L),
                          "pool_path": dmeta["fda_pool_dir"],
                          "pool_size": len(dmeta["fda_pool"] or []),
                          "pool_available": dmeta["fda_pool_available"],
                          "applied_at": "input stage, before resize and "
                                        "normalisation, training only"}
        elif args.method == "sib":
            method_cfg = dict(sib_report)
            method_cfg["tau"] = sibmod.TAU
            method_cfg["coverage_grid_step"] = 0.05
            method_cfg["c_star_val"] = sib_sel["c_star_val"] if sib_sel else None

        run = {
            "run_id": run_id, "dataset": "st", "arm": "F_crossearth",
            "arch": "crossearth", "architecture": "CrossEarth",
            "method": args.method, "mode": args.cell_type, "rung": 450,
            "n_train": dmeta["n_train"], "n_val": dmeta["n_val"],
            "n_test": dmeta["n_test"],
            "holdout": args.city if args.cell_type == "loco" else None,
            "city": args.city, "res": args.res, "seed": args.seed,
            "target_access": target_access, "target_access_detail": detail,
            "epochs_max": None, "epochs_run": None,
            "steps_to_best": steps_to_best,
            "best_val_metric": best["miou_tol2"] if best else None,
            "decision_metric": "miou_2class_boundary_tolerant_K2_source_val",
            "reporting_metric": "iou_fg_boundary_tolerant_K2 (Appendix E); "
                                "miou_2class_boundary_tolerant_K2 also written",
            "early_stop_patience": None, "lr": args.lr,
            "batch_size": args.batch_size,
            "metric_module_sha": armc_dump.sha256_file(
                os.path.join(base, "shadowtransfer/metrics.py")),
            "git_hash": None,
            "git_hash_reason": "project root is not a git repository",
            "config_sha256": armc_dump.sha256_bytes(
                json.dumps(run_cfg, sort_keys=True).encode()),
            "config_verbatim": run_cfg, "method_config": method_cfg,
            "backbone_init": {"path": args.ckpt, "sha256": sha,
                              "bytes": nbytes},
            "cluster": "delta", "partition": "gpuA100x4",
            "gpu": torch.cuda.get_device_name(0),
            "wall_clock_s": round(time.time() - t_start, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "ckpt_path": str(ckpt_path), "parent_run_id": None,
            "legacy_run_dir": None, "structure_cols": "deferred",
            "compute_note": "no CPU account on this allocation; the whole cell "
                            "including inference runs on gpuA100x4",
            "seeds": seeds, "sanity": sanity, "val_curve": val_curve,
            "test": test_scores, "sib_selective": sib_sel,
            "caveats": caveats, "s_per_iter": stats["s_per_iter"],
            "peak_gpu_mem_gb": peak_gb, "train_wall_clock_s": round(train_wall, 1),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        armc_dump.write_dump(dump_dir, rows, prob_store, run)
        print(f"[dump] {dump_dir} : {len(rows)} rows, {len(dump_cols)} columns",
              flush=True)

        results = {
            "arm": "F_crossearth",
            "weakness_ids": cells.WEAKNESS_IDS[args.method],
            "target_access": target_access,
            "target_access_detail": detail,
            "architecture": "CrossEarth", "method": args.method,
            "cell_type": args.cell_type, "city": args.city,
            "resolution": "highres", "run_id": run_id,
            "metric": {"name": "miou_boundary_tolerant", "K": 2,
                       "raw_counts_at_K": [0, 2, 5],
                       "aggregation": "per-image scalars macro-averaged over "
                                      "the test images; counts are never "
                                      "micro-pooled"},
            "run_config": run_cfg, "backbone_init": run["backbone_init"],
            "method_config": method_cfg,
            "val_curve": [{"iter": v["iter"],
                           "val_miou_tol2": v["val_miou_tol2"],
                           "val_iou_fg_tol2": v["iou_fg_tol2"],
                           "pred_fg_fraction_mean": v["pred_fg_fraction_mean"]}
                          for v in val_curve],
            "steps_to_best": steps_to_best,
            "test": {"miou_tol2": test_scores["miou_tol2"],
                     "iou_fg_tol2": test_scores["iou_fg_tol2"],
                     "n_images": test_scores["n_images"]},
            "sib_selective": sib_sel,
            "per_image_dump_path": str(dump_dir),
            "sanity_checks": {
                "no_heldout_city_in_train_or_val": True,
                "heldout_hits": sanity["heldout_hits"],
                "backbone_frozen": params["backbone_frozen_except_reins"],
                "output_shape_ok": all(p.shape == (args.crop, args.crop)
                                       for p in probs),
                "loss_finite": stats["loss_finite"],
                "steps_to_best_not_truncated": not truncated,
                "pred_fg_fraction_not_saturated": not saturated,
                "backbone_sha256_reverified": True,
            },
            "caveats": caveats,
            "compute_note": run["compute_note"], "seed": args.seed,
            "wall_clock_s": run["wall_clock_s"],
            "s_per_iter": stats["s_per_iter"], "peak_gpu_mem_gb": peak_gb,
            "requested_time": os.environ.get("REQUESTED_TIME"),
            "used_time": round(time.time() - t_start, 1),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "generated_utc": run["written_utc"],
        }
        (runs_dir / "results.json").write_text(json.dumps(results, indent=2))
        (runs_dir / "loss_trace.json").write_text(
            json.dumps({"losses": stats["losses"]}))
        (runs_dir / ".DONE").touch()
    print(f"[done] {run_id} miou_tol2 {test_scores['miou_tol2']:.4f} "
          f"iou_fg_tol2 {test_scores['iou_fg_tol2']:.4f} "
          f"steps_to_best {steps_to_best} wall {run['wall_clock_s']}s",
          flush=True)


if __name__ == "__main__":
    main()
