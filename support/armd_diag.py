#!/usr/bin/env python
"""
Arm D / D_S3_diag — the diagnostics ride-along. Inference only, no retraining.

Answers v1UJ-W4, zNBu-W2, AC1. REVIEWS.md arm D.

The grid says the gap shrank on Phoenix. It does not say why. Three probes:

  1  encoder city-identity linear probe    App G.8 protocol, on the aug model
                                           AND its vanilla LOCO comparator, in
                                           one job so the difference is
                                           computed inside one code path
  2  foreground selective prediction       AURC gap, aug-LOCO vs aug-upper and
                                           vanilla-LOCO vs vanilla-upper,
                                           image-level paired bootstrap B=1000
  3  histogram matching                    per-channel CDF match toward the
                                           source-city training aggregate,
                                           weights unchanged, per-image R

Every probe operation is imported from the arm C implementation and called
unchanged; see shadowtransfer/armd_diag_lib.py for the import table and for
the App G.8 erratum E-8 deviations, which are recorded on every block.

The job unit is a GROUP -- one (arch, city, seed) -- not one cell. A group's
vanilla LOCO comparator is then probed once and shared by its aug-loco and
aug-upper cells instead of once per aug cell, which is the whole reason the
grid is 17 jobs rather than 46.

No model is retrained. No checkpoint is written or modified.

Usage
-----
    python armd_diag.py --arch oglanet --city phoenix --seed 0
    ... --dry_run                      resolve and print, write nothing
    ... --smoke_test 5 --smoke_perm 3 --out_root <scratch>    stage-1 smoke
    python armd_diag.py --aggregate    -> runs/D_S3/diagnostics.json
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
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import armc_train                                                # noqa: E402
from shadowtransfer import armc_dump                             # noqa: E402
from shadowtransfer import armc_runtime                          # noqa: E402
from shadowtransfer import armc_probes_common as C               # noqa: E402
from shadowtransfer import armc_probes_ops as OPS                # noqa: E402
from shadowtransfer import armd_diag_lib as L                    # noqa: E402
from shadowtransfer.armc_dump import sha256_file                 # noqa: E402

RESULTS_DIR = BASE / "runs" / "D_S3" / "diag"


def group_id(arch, city, seed):
    return f"{arch}-{city}-s{seed}"


def test_loader(arch, cities, res_dir, cfg, batch_size, smoke_test):
    """Test split of one or more cities, in dataset order, via armc_train's own
    dataset builder. The smoke truncation is PER CITY: the city probe reads all
    three cities and a global head would hand its 3-way regression one class."""
    from torch.utils.data import DataLoader
    pairs = []
    for c in cities:
        names = armc_train.city_split_files(c, res_dir, "test")
        pairs += [(c, f) for f in (names[:smoke_test] if smoke_test else names)]
    ds = armc_train.build_dataset(arch, pairs, "test", res_dir,
                                  cfg["img_size"], False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True,
                        worker_init_fn=armc_train.dl_worker_init)
    return ds, loader


def load_model(arch, rid, device):
    cfg = L.run_json(rid)["config_verbatim"]
    ckpt = L.checkpoint(rid)
    model = C.build_model(arch, cfg, device)
    info = C.load_ladder_weights(model, ckpt, device)
    print(f"  loaded {rid}  ckpt_epoch {info['ckpt_epoch']}  {ckpt}", flush=True)
    return model, cfg, ckpt, info


# ---------------------------------------------------------------------------
# Probe 1 — city-identity linear probe
# ---------------------------------------------------------------------------

def probe_city(args, cells, device, cfg_ref):
    res_dir = armc_train.RES_DIR[L.RES]
    arch = cells["arch"]
    targets = [(k, cells[k]) for k in ("aug_loco", "aug_upper", "vanilla_loco")
               if cells.get(k)]
    ds_f, loader_f = test_loader(arch, L.CITIES, res_dir, cfg_ref,
                                 args.feat_batch, args.smoke_test)
    cities_per_image = [Path(p).resolve().parents[3].name for p in ds_f.img_paths]
    stems = [Path(p).stem for p in ds_f.img_paths]
    classes = sorted(set(cities_per_image))
    img_city = np.array([classes.index(c) for c in cities_per_image])
    n_perm = args.smoke_perm or args.n_permutations

    out = {"probe": "city_identity_linear",
           "protocol": "App G.8",
           "protocol_deviations": L.PROBE_PROTOCOL_DEVIATIONS,
           "n_probe_images": len(stems),
           "probe_cities": classes,
           "probe_classes": len(classes),
           "chance": round(1.0 / len(classes), 6),
           "n_folds": args.n_folds,
           "n_permutations": n_perm,
           "cells": {}}

    for key, rid in targets:
        print(f"\n--- city probe: {key} = {rid} ---", flush=True)
        model, cfg, ckpt, info = load_model(arch, rid, device)
        t = time.time()
        X, ids, feat_shape, n_cells_total = OPS.extract_features(
            model, arch, loader_f, device, args.cells_per_image, args.seed_probe)
        wall_feat = time.time() - t
        if ids != stems:
            raise SystemExit("FATAL feature extraction order does not match "
                             "the dataset's own file order")
        t = time.time()
        stats = OPS.city_identity_probe(
            X, img_city, min(args.cells_per_image, n_cells_total),
            n_folds=args.n_folds, n_perm=n_perm, seed=args.seed_probe)
        wall_probe = time.time() - t
        stats.pop("per_image_accuracy", None)

        acc = stats["probe_accuracy"]
        null_mean = stats["null_mean"]
        out["cells"][key] = {
            "run_id": rid,
            "model_variant": cells.get(f"{key}_variant") or "armC_r450",
            "probe_accuracy": acc,
            "null_mean_measured": null_mean,
            "null_p99_measured": stats["null_p99"],
            "accuracy_minus_null_mean": round(acc - null_mean, 6),
            "p_value": stats["p_value"],
            "null_min": stats["null_min"],
            "null_max": stats["null_max"],
            "null_above_nominal_chance": bool(null_mean > out["chance"]),
            "null_note": (
                "the null mean is MEASURED, never assumed to sit at 1/3. A "
                "previous run on this project measured 0.363-0.381 and that "
                "discrepancy is unresolved; the arm C App G.8 cells measured "
                "0.329-0.333. This value is reported as measured either way."),
            "n": {"n_images": len(stems),
                  "n_samples": int(X.shape[0]),
                  "feature_dim": int(X.shape[1]),
                  "n_spatial_cells_total": int(n_cells_total),
                  "n_spatial_cells_sampled": int(min(args.cells_per_image,
                                                     n_cells_total)),
                  "n_folds": args.n_folds,
                  "n_permutations": n_perm,
                  "fold_sizes": stats["fold_sizes"]},
            "encoder_feature_key": C.ENCODER_FEATURE_KEY[arch],
            "encoder_feature_shape": list(feat_shape),
            "ckpt_path": str(ckpt),
            "ckpt_sha256": sha256_file(ckpt),
            "checkpoint_load": info,
            "feature_wall_clock_s": round(wall_feat, 2),
            "probe_wall_clock_s": round(wall_probe, 2),
        }
        print(f"  {key:13s} acc {acc:.4f}  null_mean {null_mean:.4f}  "
              f"null_p99 {stats['null_p99']:.4f}  acc-null "
              f"{acc - null_mean:+.4f}  p {stats['p_value']:.4g}  "
              f"({wall_feat:.0f}s feat + {wall_probe:.0f}s probe)", flush=True)
        del model, X
        import torch
        torch.cuda.empty_cache()

    for side in ("aug_loco", "aug_upper"):
        a = out["cells"].get(side)
        v = out["cells"].get("vanilla_loco")
        if a and v:
            out.setdefault("aug_minus_vanilla", {})[side] = {
                "delta_probe_accuracy": round(a["probe_accuracy"]
                                              - v["probe_accuracy"], 6),
                "delta_accuracy_minus_null": round(
                    a["accuracy_minus_null_mean"]
                    - v["accuracy_minus_null_mean"], 6),
                "aug_run_id": a["run_id"], "vanilla_run_id": v["run_id"],
                "note": ("computed inside this job, from two probes run on "
                         "the same images with the same fold structure and "
                         "the same seeded spatial sample."),
            }
    return out


# ---------------------------------------------------------------------------
# Probe 3 — histogram matching
# ---------------------------------------------------------------------------

def reinferred_denominator_R(args, arch, city, upper_rid, loader_t, ds_t,
                            device, geo, df_base, df_match, df_gray,
                            loco_rid):
    """Probe 3's R with a FULLY re-inferred denominator.

    Why this exists
    ---------------
    R = (IoU_match - IoU_LOCO) / (IoU_upper - IoU_LOCO). In the default path
    IoU_match is re-inferred by this job while IoU_LOCO and IoU_upper are read
    from the stored dumps. That mixing is harmless on an architecture whose
    checkpoints reproduce their own stored per-image IoU, which mamnet and
    dinov3 do at exactly 0. OGLANet does not: the measured stored-vs-reinferred
    max |delta| runs up to 0.395 mIoU against CONTRACT.md's 0.01 gate, and it
    predates arm D.

    So both are recorded and neither is chosen: `recovery_ratio` keeps the
    stored denominator, and this block re-infers IoU_LOCO and IoU_upper with
    this job's own weights and images so numerator and denominator come from
    one pass. The lifted metric functions are untouched.

    One extra unmatched inference over the same 150 test images, with the upper
    cell's checkpoint. No checkpoint is written.
    """
    import pandas as pd
    import torch

    print(f"  re-inferring the denominator with {upper_rid}", flush=True)
    was = ds_t.fda_transform
    ds_t.fda_transform = None
    model_up, _cfg_up, ckpt_up, info_up = load_model(arch, upper_rid, device)
    t = time.time()
    rows_up, _ = C.infer_rows(model_up, loader_t, upper_rid,
                              city, device, geo)
    wall_up = time.time() - t
    del model_up
    torch.cuda.empty_cache()
    ds_t.fda_transform = was

    df_up = pd.DataFrame(rows_up)
    if list(df_up["image_id"]) != list(df_base["image_id"]):
        raise SystemExit("FATAL the re-inferred upper pass disagrees with the "
                         "LOCO passes on image order")

    stored_up = L.perimage(upper_rid)
    rep_up = np.abs(df_up["iou_tol"].to_numpy(dtype=np.float64)
                    - C.align_on_image_id(df_up, stored_up, "iou_tol"))

    iou_loco_re = df_base["iou_tol"].to_numpy(dtype=np.float64)
    iou_up_re = df_up["iou_tol"].to_numpy(dtype=np.float64)
    r, ok, head = C.recovery_ratio(df_match["iou_tol"].to_numpy(),
                                   iou_loco_re, iou_up_re)
    rg, _, _ = C.recovery_ratio(df_gray["iou_tol"].to_numpy(),
                                iou_loco_re, iou_up_re)

    out = {
        "upper_run_id": upper_rid,
        "loco_run_id": loco_rid,
        "definition": (
            "per image, R = (IoU_match - IoU_LOCO) / (IoU_upper - IoU_LOCO) "
            "on foreground-only IoU at K=2, with IoU_LOCO AND IoU_upper both "
            "re-inferred by this job from the two checkpoints rather than "
            "read from the stored dumps. The companion block "
            "`recovery_ratio` uses the stored denominator. Both are "
            "reported and neither is declared correct here."),
        "per_channel": L.r_summary(r, seed=args.seed_boot),
        "grayscale": L.r_summary(rg, seed=args.seed_boot),
        "per_image_R": [None if not np.isfinite(x) else round(float(x), 6)
                        for x in r],
        "per_image_headroom": [round(float(x), 6) for x in head],
        "n_included": int(ok.sum()), "n_excluded": int((~ok).sum()),
        "upper_repro_max_abs_delta_iou_tol": float(rep_up.max()),
        "upper_repro_mean_abs_delta_iou_tol": float(rep_up.mean()),
        "upper_ckpt_path": str(ckpt_up),
        "upper_ckpt_sha256": sha256_file(ckpt_up),
        "upper_checkpoint_load": info_up,
        "iou_tol_upper_reinferred_mean_pct": float(df_up["iou_tol"].mean()) * 100.0,
        "iou_tol_upper_stored_mean_pct": float(stored_up["iou_tol"].mean()) * 100.0,
        "inference_wall_clock_s": round(wall_up, 2),
    }
    s = out["per_channel"]
    print(f"  R (re-inferred denom) median {s['median']}  ci95 {s['ci95']}  "
          f"n {s['n']}  upper repro max|d| "
          f"{out['upper_repro_max_abs_delta_iou_tol']:.3e}", flush=True)
    return out


def probe_histmatch(args, cells, device, cfg_ref):
    res_dir = armc_train.RES_DIR[L.RES]
    arch, city, seed = cells["arch"], cells["city"], cells["seed"]
    geo = armc_dump.load_geo_meta(BASE)

    # The source pool is the LOCO training list at this rung and seed. The aug
    # cell and its vanilla comparator draw byte-identical lists under
    # armc_train.resolve_selection, so ONE histogram serves both; that
    # identity is asserted below rather than assumed.
    train_pairs, _, _, train_cities, n_per_city = armc_train.resolve_selection(
        "loco", city, L.RUNG, res_dir, seed)
    train_sha = armc_train.list_sha256(train_pairs)
    src_ch, src_gray, n_src = OPS.source_cdfs(train_pairs, res_dir, BASE)

    ds_t, loader_t = test_loader(arch, [city], res_dir, cfg_ref, 1,
                                 args.smoke_test)

    out = {"probe": "histogram_matching",
           "protocol": "s4.2 / App G.3",
           "intervention": ("test-time per-channel CDF matching toward the "
                            "source-city training aggregate, no parameter "
                            "update, weights unchanged"),
           "match_injection_point": (
               "dataset.fda_transform -- the raw uint8 RGB array immediately "
               "after Image.open, before resize, normalisation and the "
               "contrast channel"),
           "match_definition_headline": (
               "per-channel: for each RGB channel independently the source "
               "pool's aggregate channel CDF is the target and the mapping is "
               "argmin_s |cdf_src[s] - cdf_img[t]|"),
           "match_definition_published": (
               "grayscale-plus-ratio, verbatim from "
               "final_loco/experiment_utils.histogram_match_image. Carried "
               "alongside as *_gray because the published section 4.2 finding "
               "was measured under it."),
           "n_source_images": n_src,
           "source_cities": train_cities,
           "n_train_per_source_city": n_per_city,
           "train_list_sha256": train_sha,
           "r_exclusion_rule": (
               f"images with (IoU_upper - IoU_LOCO) < {C.R_EXCLUDE_MIN}; the "
               f"ratio's denominator is that headroom and a near-zero one "
               f"manufactures arbitrarily large |R|"),
           "cells": {}, "skipped": []}

    pairs = [("aug_loco", cells.get("aug_loco"), cells.get("aug_upper")),
             ("vanilla_loco", cells.get("vanilla_loco"), cells.get("vanilla_upper"))]
    for key, rid, upper_rid in pairs:
        if not rid:
            out["skipped"].append({"cell": key, "reason": "no completed cell"})
            continue
        print(f"\n--- histmatch: {key} = {rid} ---", flush=True)
        rj = L.run_json(rid)
        if rj.get("train_list_sha256") not in (None, train_sha):
            raise SystemExit(
                f"FATAL {rid} recorded train_list_sha256 "
                f"{rj.get('train_list_sha256')}, but this job's source "
                f"histogram was built from {train_sha}. The matching "
                f"statistic would not be that model's own source pool.")
        model, cfg, ckpt, info = load_model(arch, rid, device)

        import pandas as pd
        import torch
        ds_t.fda_transform = None
        t = time.time()
        rows_base, _ = C.infer_rows(model, loader_t, rid, city, device, geo)
        w_base = time.time() - t
        ds_t.fda_transform = OPS.PerChannelMatcher(src_ch)
        t = time.time()
        rows_match, _ = C.infer_rows(model, loader_t, rid, city, device, geo)
        w_match = time.time() - t
        ds_t.fda_transform = OPS.GrayMatcher(src_gray)
        t = time.time()
        rows_gray, _ = C.infer_rows(model, loader_t, rid, city, device, geo)
        w_gray = time.time() - t
        ds_t.fda_transform = None

        df, dfm, dfg = (pd.DataFrame(rows_base), pd.DataFrame(rows_match),
                        pd.DataFrame(rows_gray))
        if not (list(df["image_id"]) == list(dfm["image_id"])
                == list(dfg["image_id"])):
            raise SystemExit("FATAL the three inference passes disagree on "
                             "image order")

        # Execution fact the job checks on itself: the same weights on the
        # same images must reproduce the cell's own stored per-image IoU.
        stored = L.perimage(rid)
        rep = np.abs(df["iou_tol"].to_numpy(dtype=np.float64)
                     - C.align_on_image_id(df, stored, "iou_tol"))
        print(f"  REPRO unmatched vs {rid}: max|d| {rep.max():.3e}", flush=True)

        blk = {"run_id": rid,
               "model_variant": cells.get(f"{key}_variant") or "armC_r450",
               "ckpt_path": str(ckpt), "ckpt_sha256": sha256_file(ckpt),
               "checkpoint_load": info,
               "repro_max_abs_delta_iou_tol": float(rep.max()),
               "repro_mean_abs_delta_iou_tol": float(rep.mean()),
               "repro_n_images": int(len(rep)),
               "iou_tol_unmatched_mean_pct": float(df["iou_tol"].mean()) * 100.0,
               "iou_tol_matched_mean_pct": float(dfm["iou_tol"].mean()) * 100.0,
               "iou_tol_matched_gray_mean_pct": float(dfg["iou_tol"].mean()) * 100.0,
               "miou_tol_unmatched_mean_pct": float(df["miou_tol_pct"].mean()),
               "miou_tol_matched_mean_pct": float(dfm["miou_tol_pct"].mean()),
               "inference_wall_clock_s": {"unmatched": round(w_base, 2),
                                          "matched": round(w_match, 2),
                                          "matched_gray": round(w_gray, 2)},
               "recovery_ratio": {}, "n": {"n_images": int(len(df))}}

        if upper_rid:
            up = L.perimage(upper_rid)
            iou_loco = C.align_on_image_id(df, stored, "iou_tol")
            iou_upper = C.align_on_image_id(df, up, "iou_tol")
            r, ok, head = C.recovery_ratio(dfm["iou_tol"].to_numpy(),
                                           iou_loco, iou_upper)
            rg, _, _ = C.recovery_ratio(dfg["iou_tol"].to_numpy(),
                                        iou_loco, iou_upper)
            blk["recovery_ratio"] = {
                "upper_run_id": upper_rid,
                "definition": (
                    "per image, R = (IoU_match - IoU_LOCO) / (IoU_upper - "
                    "IoU_LOCO) on foreground-only IoU at K=2. IoU_LOCO and "
                    "IoU_upper are read from the stored dumps and are NOT "
                    "re-inferred. The upper bound is the SAME condition's "
                    "within-city cell, so the aug ratio is not scored against "
                    "a vanilla ceiling."),
                "per_channel": L.r_summary(r, seed=args.seed_boot),
                "grayscale": L.r_summary(rg, seed=args.seed_boot),
                "per_image_R": [None if not np.isfinite(x) else round(float(x), 6)
                                for x in r],
                "per_image_headroom": [round(float(x), 6) for x in head],
                "n_included": int(ok.sum()), "n_excluded": int((~ok).sum()),
            }
            if getattr(args, "reinfer_denominator", False):
                blk["recovery_ratio_reinferred"] = reinferred_denominator_R(
                    args, arch, city, upper_rid, loader_t, ds_t, device,
                    geo, df, dfm, dfg, rid)
            s = blk["recovery_ratio"]["per_channel"]
            print(f"  R median {s['median']}  ci95 {s['ci95']}  n {s['n']}",
                  flush=True)
        else:
            blk["recovery_ratio"] = {
                "upper_run_id": None,
                "reason": "no completed within-city upper bound for this cell",
            }
        out["cells"][key] = blk
        del model
        torch.cuda.empty_cache()

    a = out["cells"].get("aug_loco", {}).get("recovery_ratio", {}).get("per_channel")
    v = out["cells"].get("vanilla_loco", {}).get("recovery_ratio", {}).get("per_channel")
    if a and v and a.get("median") is not None and v.get("median") is not None:
        out["aug_minus_vanilla"] = {
            "delta_median_R": round(a["median"] - v["median"], 6),
            "aug_median_R": a["median"], "vanilla_median_R": v["median"],
            "note": ("difference of two medians each with its own bootstrap "
                     "interval; no joint interval is computed for it here."),
        }
    return out


# ---------------------------------------------------------------------------
# One group
# ---------------------------------------------------------------------------

def run_group(args):
    import torch
    t0 = time.time()
    arch, city, seed = args.arch, args.city, args.seed
    gid = group_id(arch, city, seed)
    cells = L.group_cells(arch, city, seed)

    out_dir = (Path(args.out_root) if args.out_root else RESULTS_DIR)
    dst = out_dir / f"{gid}.json"

    print("=" * 78)
    print(f"ARM D / {L.ARM_ID}  group {gid}")
    print(f"  weaknesses      {L.WEAKNESS_IDS}")
    for k in ("aug_loco", "aug_upper", "vanilla_loco", "vanilla_upper"):
        print(f"  {k:15s} {cells.get(k)}  "
              f"({cells.get(k + '_variant') or 'armC_r450'})")
    print(f"  missing         {cells['missing'] or 'none'}")
    print(f"  probes          city / sp / histmatch")
    print(f"  out             {dst}")
    print(f"  slurm job       {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    print("=" * 78, flush=True)

    if cells["missing"] and not args.allow_partial:
        raise SystemExit(f"FATAL missing cells {cells['missing']} and "
                         f"--allow_partial not set")
    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    if (dst.exists() and (out_dir / f"{gid}.DONE").exists() and not args.force):
        print(f"[{gid}] .DONE exists at {out_dir} — nothing to do")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    armc_runtime.preflight_write_targets(
        BASE, [dst, out_dir / f"{gid}.DONE"], label="diag")

    C.enter_arch_tree(arch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_rid = (cells.get("aug_loco") or cells.get("aug_upper")
               or cells.get("vanilla_loco"))
    cfg_ref = L.run_json(ref_rid)["config_verbatim"]

    blocks, walls = {}, {}
    with armc_runtime.PhaseTimeout(args.timeout_s, out_dir) as tbx:
        tbx.phase("probe_sp")
        t = time.time()
        blocks["sp"] = L.sp_block(cells, seed=args.seed_boot)
        walls["sp"] = round(time.time() - t, 2)

        tbx.phase("probe_city")
        t = time.time()
        blocks["city"] = probe_city(args, cells, device, cfg_ref)
        walls["city"] = round(time.time() - t, 2)

        tbx.phase("probe_histmatch")
        t = time.time()
        blocks["histmatch"] = probe_histmatch(args, cells, device, cfg_ref)
        walls["histmatch"] = round(time.time() - t, 2)

        tbx.phase("write")
        rec = {
            "group_id": gid,
            "arm": L.ARM, "arm_id": L.ARM_ID, "weakness_ids": L.WEAKNESS_IDS,
            "arch": arch, "city": city, "seed": seed,
            "rung": L.RUNG, "res": L.RES,
            "cells": cells,
            "model_variant": {k: cells.get(f"{k}_variant")
                              for k in ("aug_loco", "aug_upper")},
            "metric_name": L.METRIC_NAME,
            "secondary_metric_name": L.SECONDARY_METRIC_NAME,
            "reporting_metric": "fg_only_tol_k2",
            "band_width_headline": 2,
            "band_widths_written": [0, 2, 5],
            "probes": blocks,
            "probe_wall_clock_s": walls,
            "wall_clock_s": round(time.time() - t0, 2),
            "gpu_hours": round((time.time() - t0) / 3600.0, 4),
            "peak_vram_alloc_gb": round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "retraining": "none; every model is loaded from an existing "
                          "checkpoint_best.pth and no checkpoint is written",
            "target_access": "source_only",
            "target_access_detail": (
                "the city probe reads encoder activations on TEST images of "
                "all three cities and no label changes a model weight; the "
                "matching statistic is the aggregate histogram of the source "
                "cities' own TRAINING images. The aug-upper and vanilla-upper "
                "cells being read are themselves target_labels runs, but this "
                "job only reads their stored per-image numbers."),
            "cluster": "delta",
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_time_limit": os.environ.get("ARMD_TIME_LIMIT"),
            "smoke": bool(args.smoke_test or args.smoke_perm),
            "git_hash": None,
            "git_hash_reason": "left null for arm D artifacts, matching arm C",
            "diag_script_sha": sha256_file(Path(__file__).resolve()),
            "written_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        }
        rec.update(L.module_shas())
        dst.write_text(json.dumps(rec, indent=2, default=str))
        (out_dir / f"{gid}.DONE").touch()

    print("\n" + "=" * 78)
    print(f"[{gid}] probe wall clock  {walls}")
    print(f"  total {rec['wall_clock_s']:.1f}s  peak VRAM "
          f"{rec['peak_vram_alloc_gb']:.2f} GiB")
    print(f"  -> {dst}")
    print("=" * 78, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=sorted(armc_train.ARCH_DIR))
    ap.add_argument("--city", choices=sorted(armc_train.FOLD_OF_HOLDOUT))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--list_groups", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--ignore_block", action="store_true",
                    help="aggregate despite Part A's gate; recorded "
                         "in the output. Operator instruction only.")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--timeout_s", type=int, default=3000)
    ap.add_argument("--cells_per_image", type=int, default=32)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_permutations", type=int, default=100)
    ap.add_argument("--feat_batch", type=int, default=8)
    ap.add_argument("--seed_probe", type=int, default=0)
    ap.add_argument("--seed_boot", type=int, default=0)
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--smoke_perm", type=int, default=0)
    ap.add_argument("--reinfer_denominator", action="store_true",
                    help="probe 3 only: also compute R with a "
                         "FULLY re-inferred denominator. Off by "
                         "default, so the validated path is "
                         "behaviourally unchanged when it is unset.")
    args = ap.parse_args()

    if args.list_groups:
        g = L.discover_groups()
        print(json.dumps({f"{a}-{c}-s{s}": m for (a, c, s), m in g.items()},
                         indent=2))
        print(f"n_groups {len(g)}")
        return
    if args.aggregate:
        import armd_diag_aggregate as AGG
        AGG.aggregate(args)
        return
    for k in ("arch", "city", "seed"):
        if getattr(args, k) is None:
            raise SystemExit(f"FATAL --{k} is required unless --aggregate or "
                             f"--list_groups")
    run_group(args)


if __name__ == "__main__":
    main()
