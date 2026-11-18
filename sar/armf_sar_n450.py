#!/usr/bin/env python3
"""
Arm F SAR, re-run on the n450 MAMNet standard (vanilla) LOCO checkpoints.

IDENTICAL HARNESS, ONE THING CHANGED
------------------------------------
`python/armf_sar_grid.py` ran the published nine-cell SAR grid. This driver
runs the SAME two passes (unadapted, then SAR at the stage-1 selected
configuration) through the SAME pass module, `shadowtransfer/armf_sar_pass.py`,
on the SAME 150 held-out-city test images in the fixed released order, at batch
size 1 with augmentation off and seed 0. The only input that differs is the
checkpoint: the n450 MAMNet vanilla LOCO cell of the same city
(`mamnet-vanilla-loco-r450-{city}-hr-s42-n450`) in place of the published
`mamnet_loco_holdout_{city}_highres_1`.

armf_sar_grid.py is NOT edited. It produced published numbers and stays frozen
(CLAUDE.md). This file imports its `read_selected` so the learning-rate and
step count come from the same stage-1 `selected_config.json`, byte-for-byte.

Output goes to a NEW directory with the `-n450` suffix, under /scratch
(/projects is archive-only, 2026-09-01). The published
`runs/armF_sar_mamnet_*_highres/` directories are never opened for writing.

THREE THINGS THIS ADDS, none of which change a pass
---------------------------------------------------
1. `miou_k2_twoclass`, derived in-run from the stored per-image K=2 counts by
   `armf_sar_naming_patch.counts_agg` -- the same function that wrote the
   published cells' `miou_tol2_derived_from_counts` post hoc.
2. THE n450 SANITY GATE, and unlike the published cells' stored-value check
   this one IS a stop (operator instruction, 2026-10-29): the unadapted pass
   must reproduce the n450 cell's own two-class mIoU at K=2, as scored by
   `n450_score.py` into its `run.json`, to within 0.01. MAMNet has no DFFM, so
   the two should agree. A miss writes .FAILED and exits non-zero.
3. A weight-drift measurement: per image, episodically, the adapted state_dict
   is diffed against the pristine snapshot with
   `trainaudit_sar_drift.diff_state` (lifted, not reimplemented), giving the
   largest |adapted - start| over every parameter and every image. The
   published results.json carries only `n_images_state_changed_by_adaptation`,
   a byte-identity count, which answers whether anything moved but not by how
   much.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_T0 = time.time()


def _tick(msg):
    """Timestamped progress to stdout. Added 2026-10-30 after smoke 23330650
    hit its 20-minute wall with no output at all: every step before the config
    banner is a Lustre read, and a silent hang there is indistinguishable from
    a slow import. Diagnostic only; changes no setting the passes use."""
    print(f"[tick {time.time() - _T0:8.1f}s] {msg}", flush=True)


_tick("stdlib imports done")
SCRIPT_PATH = Path(__file__).resolve()
BASE = SCRIPT_PATH.parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_runtime, armf_sar_meta, armf_sar_pass, sar   # noqa: E402
from shadowtransfer.armc_dump import sha256_file                            # noqa: E402
from shadowtransfer.armf_sar_cells import (ARM, NUM_CLASSES, RES_DIR,       # noqa: E402
                                           WEAKNESS_IDS, build_model,
                                           enter_arch_tree,
                                           load_checkpoint_strict,
                                           stored_fg_iou_k2, test_images)
from armf_sar_grid import read_selected                                     # noqa: E402
from armf_sar_naming_patch import counts_agg                                # noqa: E402
_tick("project imports done (torch loaded)")

STAGE = "grid_n450"
ARCH = "mamnet"
SUFFIX = "-n450"
GATE_TOL_MIOU = 0.01

N450_CKPT_ROOT = Path(os.environ["ST_SCRATCH_ROOT"]) / "data/mamnet/outputs/n450"
N450_DUMP_ROOT = Path(os.environ["ST_SCRATCH_ROOT"]) / ("dumps/"
                      "st_r450_published_recompute")
DEFAULT_OUT_ROOT = Path(os.environ["ST_SCRATCH_ROOT"]) / "runs"
PUBLISHED_SELECTED = BASE / "runs/armF_sar_lr_pilot/selected_config.json"


def n450_run_id(city):
    return f"mamnet-vanilla-loco-r450-{city}-hr-s42-n450"


def resolve_n450_cell(city, res_dir):
    """(cell_dir, checkpoint, args.json, test_results.json, run.json). The n450
    cell keeps the published legacy run-dir name inside its own run-id
    directory; that is a run_dir, never a run_id."""
    rid = n450_run_id(city)
    d = N450_CKPT_ROOT / rid / f"{ARCH}_loco_holdout_{city}_{res_dir}_1"
    ck, cfgp, tr = d / "checkpoint_best.pth", d / "args.json", d / "test_results.json"
    rj = N450_DUMP_ROOT / rid / "run.json"
    for p in (ck, cfgp, rj):
        if not p.exists():
            raise SystemExit(f"FATAL missing {p}")
    return d, ck, cfgp, tr, rj


def stored_n450_miou_k2_twoclass(run_json):
    """The n450 cell's own two-class mIoU at K=2, as `n450_score.py` wrote it.
    This is the gate's reference value and it is read, never recomputed."""
    d = json.loads(Path(run_json).read_text())
    s = d["scoring"]
    return (float(s["miou_k2_twoclass"]), float(s["headline_fg_iou_tol_pct"]),
            d.get("scorer_sha"), d.get("inference_batch_size"))


def two_class_from_rows(rows, pass_id, step):
    """Two-class mIoU at K=2 and foreground-only IoU at K=2 from the stored
    per-image counts of one pass at one step, via the function that wrote the
    published cells' derived value. No metric is computed here."""
    import pandas as pd
    sub = pd.DataFrame([r for r in rows
                        if r["pass_id"] == pass_id and r["step"] == step])
    miou2, fg, n, n_degen, n_empty_gt = counts_agg(sub)
    return {"miou_k2_twoclass": miou2, "iou_fg_k2": fg, "n_images": int(n),
            "n_images_fg_denominator_zero": n_degen,
            "n_images_empty_ground_truth": n_empty_gt}


def drift_pass(cfg, ckpt, device, images, lr, steps, rho, momentum,
               progress_every=25):
    """Largest |adapted - start| over every parameter and every image.

    Same episodic contract as a SAR pass: restore the pristine state, adapt,
    then diff the FULL state_dict so a parameter SAR did not claim to adapt
    cannot move unnoticed. `diff_state` is imported from
    `trainaudit_sar_drift.py` rather than rewritten.
    """
    import torch
    sys.path.insert(0, str(Path(os.environ["PROJECT_ROOT"]) / "support"))
    from trainaudit_sar_drift import diff_state

    t0 = time.time()
    model = build_model(ARCH, cfg, device)
    ck_epoch = load_checkpoint_strict(model, ckpt, device)
    model, _ = sar.configure_model(model, sar.BN_EVAL_SOURCE_STATS)
    params, names, skipped = sar.collect_params(model)
    for p in params:
        p.requires_grad_(True)
    margin = sar.e0_margin(NUM_CLASSES)
    reset_const = sar.reset_constant(NUM_CLASSES)
    pristine = sar.snapshot_state(model)
    collected = set(names)

    def optimizer_factory():
        return sar.make_optimizer(params, lr=lr, momentum=momentum, rho=rho)

    best = {"max_abs_delta": 0.0, "tensor": None, "param_name": None,
            "image_id": None, "start_value_at_argmax": None,
            "rel_delta_at_argmax": None}
    per_image, moved, outside = [], set(), set()
    for idx, im in enumerate(images):
        sar.restore_state(model, pristine)
        x = im["x"].to(device, non_blocking=True)
        sar.adapt_image(model, x, steps, margin, reset_const,
                        optimizer_factory, pristine)
        rows = diff_state(model, pristine)
        im_max = 0.0
        for r in rows:
            moved.add(r["param_name"])
            if r["param_name"] not in collected:
                outside.add(r["name"])
            m = float(r.get("max_abs_delta", 0.0))
            im_max = max(im_max, m)
            if m > best["max_abs_delta"]:
                best = {"max_abs_delta": m, "tensor": r["name"],
                        "param_name": r["param_name"],
                        "image_id": im["image_id"],
                        "start_value_at_argmax": r.get("start_value_at_argmax"),
                        "rel_delta_at_argmax": r.get("rel_delta_at_argmax")}
        per_image.append({"image_id": im["image_id"], "max_abs_delta": im_max,
                          "n_tensors_moved": len(rows)})
        del x
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"    drift: {idx + 1}/{len(images)} images, "
                  f"{time.time() - t0:.0f}s, running max "
                  f"{best['max_abs_delta']:.3e}", flush=True)

    maxes = sorted((p["max_abs_delta"] for p in per_image), reverse=True)
    out = {
        "what": ("largest |adapted - start| over every tensor of the full "
                 "state_dict and every image, measured episodically with the "
                 "same restore/adapt contract as the SAR pass"),
        "steps": steps, "lr": lr, "rho": rho, "momentum": momentum,
        "checkpoint_epoch": ck_epoch,
        "n_images": len(images),
        "n_collected_params": len(names),
        "n_skipped_norm_modules": len(skipped),
        "weights_changed": bool(best["max_abs_delta"] > 0.0),
        "max_abs_delta": best["max_abs_delta"],
        "max_abs_delta_tensor": best["tensor"],
        "max_abs_delta_image_id": best["image_id"],
        "max_abs_delta_start_value": best["start_value_at_argmax"],
        "max_abs_delta_relative": best["rel_delta_at_argmax"],
        "n_params_moved": len(moved),
        "n_images_with_any_movement": int(sum(1 for p in per_image
                                              if p["max_abs_delta"] > 0.0)),
        "moved_outside_collected": sorted(outside),
        "per_image_max_abs_delta_top10": maxes[:10],
        "per_image_max_abs_delta_median": (maxes[len(maxes) // 2]
                                           if maxes else None),
        "wall_clock_s": round(time.time() - t0, 2),
        # release layout: the lifted drift module ships at <repo>/support/
        "drift_module_sha256": sha256_file(
            Path(os.environ["PROJECT_ROOT"]) / "support/trainaudit_sar_drift.py"),
    }
    print(f"  -> drift   max |adapted-start| {out['max_abs_delta']:.6e}  on "
          f"{out['max_abs_delta_tensor']} (image {out['max_abs_delta_image_id']})"
          f"   tensors ever moved {out['n_params_moved']}   images with movement "
          f"{out['n_images_with_any_movement']}/{len(images)}   "
          f"{out['wall_clock_s']:.0f}s", flush=True)
    del model, pristine, params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True,
                    choices=["chicago", "miami", "phoenix"])
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--selected_config", default=str(PUBLISHED_SELECTED))
    ap.add_argument("--fallback_lr_mult", type=float, default=1.0)
    ap.add_argument("--fallback_steps", type=int, default=1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--rho", type=float, default=sar.SAM_RHO)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--tag", default=SUFFIX)
    ap.add_argument("--no_drift", action="store_true")
    ap.add_argument("--timeout_s", type=int, default=900)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    _tick(f"argparse done: city={args.city} limit={args.limit} tag={args.tag}")
    res_dir = RES_DIR[args.res]
    cell_dir, ckpt, cfg_path, tr_path, run_json = resolve_n450_cell(args.city, res_dir)
    _tick(f"resolve_n450_cell done: {cell_dir}")
    cfg = json.loads(cfg_path.read_text())
    _tick("args.json read")
    lr, lr_mult, steps, verdict, sel_path, sel_split = read_selected(
        args.selected_config, args.limit, args.fallback_lr_mult,
        args.fallback_steps)
    _tick("selected_config read")
    gate_ref, gate_ref_fg, scorer_sha, n450_ibs = stored_n450_miou_k2_twoclass(run_json)
    _tick(f"n450 run.json read: gate_ref={gate_ref:.6f}")
    specs = [armf_sar_pass.unadapted_spec(),
             armf_sar_pass.sar_spec(lr, lr_mult, steps)]

    out_dir = Path(args.out_root) / \
        f"armF_sar_{ARCH}_{args.city}_{res_dir}{args.tag}"
    results_path, dump_path = out_dir / "results.json", out_dir / "perstep.parquet"

    print("=" * 78)
    print(f"ARM F SAR on the n450 MAMNet standard model   city={args.city}  "
          f"res={args.res}")
    print(f"  n450 run_id     {n450_run_id(args.city)}")
    print(f"  cell dir        {cell_dir}")
    print(f"  checkpoint      {ckpt}")
    print(f"  config          {cfg_path}")
    print(f"  n450 run.json   {run_json}")
    print(f"  gate reference  two-class mIoU K=2 {gate_ref:.6f}  (fg-only "
          f"{gate_ref_fg:.6f}; scorer {scorer_sha})")
    print(f"  selected_config {sel_path}   verdict {verdict}")
    print(f"  lr              {lr:g}  ({lr_mult:g}x lr_base {sar.SAR_LR_BASE})"
          f"   steps {steps}")
    print(f"  E_0             {sar.e0_margin(2):.6f} nats   reset constant "
          f"{sar.reset_constant(2):.6f} nats")
    print(f"  norm mode       {armf_sar_meta.NORM_MODE}")
    print(f"  passes          {[s['pass_id'] for s in specs]}"
          f"{'' if args.no_drift else ' + drift'}")
    print(f"  out             {out_dir}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return 0
    if (out_dir / ".DONE").exists():
        print(f"skip: {out_dir}/.DONE exists")
        return 0

    armc_runtime.preflight_write_targets(BASE, [results_path, dump_path],
                                         label="armF SAR n450 output")
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from shadowtransfer.arme_common import seed_all
    seed_sites = seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enter_arch_tree(ARCH)
    images = test_images(ARCH, args.city, res_dir, args.limit)
    print(f"\nTEST SET {len(images)} images, fixed released order, augmentation "
          f"off, batch size 1; first {images[0]['image_id']}, last "
          f"{images[-1]['image_id']}", flush=True)

    summaries, rows = [], []
    base_iou_by_image, base_fg0_by_image = {}, {}
    for spec in specs:
        s, r = armf_sar_pass.run_pass(spec, ARCH, cfg, ckpt, device, images,
                                      rho=args.rho, momentum=args.momentum)
        summaries.append(s)
        rows.extend(r)
        if spec["pass_id"] == "unadapted":
            fin = [x for x in r if x["step"] == 0]
            base_iou_by_image = {x["image_id"]: x["iou_fg_tol2"] for x in fin}
            base_fg0_by_image = {x["image_id"]: x["pred_fg_fraction"] for x in fin}
            s["delta_vs_unadapted"] = 0.0
        else:
            s["delta_vs_unadapted"] = s["miou_fg_tol2"] - summaries[0]["miou_fg_tol2"]

    armf_sar_pass.mark_divergence(summaries)
    s0 = armf_sar_pass.step0_check(rows, summaries, base_fg0_by_image)
    una, sarp = summaries[0], summaries[1]

    two_un = two_class_from_rows(rows, "unadapted", 0)
    two_s0 = two_class_from_rows(rows, sarp["pass_id"], 0)
    two_sf = two_class_from_rows(rows, sarp["pass_id"], steps)

    # ---- THE n450 GATE: a stop, not a warning -------------------------------
    gate_diff = two_un["miou_k2_twoclass"] - gate_ref
    gate_ok = abs(gate_diff) <= GATE_TOL_MIOU
    gate = {
        "what": ("the harness's unadapted pass must reproduce the n450 cell's "
                 "own two-class mIoU at K=2 to within "
                 f"{GATE_TOL_MIOU}; MAMNet has no DFFM so the two paths should "
                 "agree"),
        "harness_unadapted_miou_k2_twoclass": two_un["miou_k2_twoclass"],
        "n450_run_json_miou_k2_twoclass": gate_ref,
        "difference": gate_diff,
        "tolerance": GATE_TOL_MIOU,
        "pass": bool(gate_ok),
        "n450_run_json": str(run_json),
        "n450_scorer_sha": scorer_sha,
        "n450_inference_batch_size": n450_ibs,
        "harness_unadapted_iou_fg_k2": una["miou_fg_tol2"],
        "n450_run_json_iou_fg_k2": gate_ref_fg,
        "difference_iou_fg_k2": una["miou_fg_tol2"] - gate_ref_fg,
        "action": ("HARD STOP on failure (operator instruction 2026-10-29). "
                   "This differs from the published grid, where the comparison "
                   "against a stored value is WARN-only."),
    }
    print(f"\n[n450 gate] harness unadapted two-class mIoU K=2 "
          f"{two_un['miou_k2_twoclass']:.6f}   n450 run.json {gate_ref:.6f}   "
          f"difference {gate_diff:+.6f}   tolerance {GATE_TOL_MIOU}   "
          f"{'PASS' if gate_ok else 'FAIL'}", flush=True)
    if not gate_ok and not args.limit:
        armc_runtime.mark_failed(out_dir, "n450_gate",
                                 f"unadapted {two_un['miou_k2_twoclass']} vs "
                                 f"stored {gate_ref}, difference {gate_diff}")
        (out_dir / "gate_failed.json").write_text(
            json.dumps(gate, indent=2, default=str) + "\n")
        raise SystemExit(
            f"FATAL n450 gate: the harness's unadapted pass gives two-class "
            f"mIoU K=2 {two_un['miou_k2_twoclass']:.6f}, the n450 cell's own "
            f"run.json gives {gate_ref:.6f}, difference {gate_diff:+.6f} "
            f"exceeds {GATE_TOL_MIOU}. Stopping; both values are above and in "
            f"{out_dir / 'gate_failed.json'}.")
    if not gate_ok:
        print("WARN gate failed on a SMOKE run; not a stop (n<150 makes the "
              "comparison meaningless).", flush=True)

    episodic_all = all(s["episodic_state_isolation"] for s in summaries)
    if not episodic_all:
        armc_runtime.mark_failed(out_dir, "episodic_isolation",
                                 "model state was not byte-identical between images")
        raise SystemExit("FATAL episodic state isolation failed.")

    drift = None
    if not args.no_drift:
        drift = drift_pass(cfg, ckpt, device, images, lr, steps, args.rho,
                           args.momentum)

    stored, stored_reason = stored_fg_iou_k2(tr_path)
    diff = None if stored is None else una["miou_fg_tol2"] - stored

    caveats = armf_sar_meta.base_caveats(
        lr=lr, lr_multiplier=lr_mult, steps=steps, rho=args.rho,
        momentum=args.momentum, config_a_collect="norm_all")
    caveats.append(
        f"The (lr, steps) pair was selected on the fold's SOURCE-CITY "
        f"validation split pooled over MAMNet and DINOv3 (stage 1, verdict "
        f"'{verdict}'), never on held-out-city images, and on the PUBLISHED "
        f"checkpoints. It is reused here unchanged: re-selecting it on the "
        f"n450 models would be a second variable.")
    caveats.append(
        "CHECKPOINT: this cell runs the n450 MAMNet vanilla LOCO model "
        f"({n450_run_id(args.city)}, 450 training images), not the published "
        "cell. Every other input -- images, order, batch size, augmentation, "
        "seed, SAR configuration, pass module -- is the published grid's.")
    caveats.append(
        "The two-class mIoU at K=2 in this file is derived in-run from the "
        "stored per-image counts by armf_sar_naming_patch.counts_agg, the same "
        "function that wrote the published cells' "
        "miou_tol2_derived_from_counts. The Appendix E degenerate rule is NOT "
        "applied, matching those cells.")
    if args.limit:
        caveats.append(f"SMOKE RUN: {args.limit} images. Not a result.")

    results = {
        "arm": ARM,
        "stage": STAGE,
        "weakness_ids": WEAKNESS_IDS,
        "target_access": armf_sar_meta.TARGET_ACCESS,
        "target_access_detail": armf_sar_meta.TARGET_ACCESS_DETAIL,
        "arch": ARCH,
        "holdout_city": args.city,
        "resolution": args.res,
        "parent_run_id": n450_run_id(args.city),
        "parent_run_dir": cell_dir.name,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "published_counterpart_run_dir": f"{ARCH}_loco_holdout_{args.city}_{res_dir}_1",
        "published_counterpart_results": str(
            BASE / f"runs/armF_sar_{ARCH}_{args.city}_{res_dir}/results.json"),
        "metric": dict(armf_sar_meta.METRIC_BLOCK),
        "sar_config": armf_sar_meta.sar_config_block(
            lr=lr, lr_multiplier=lr_mult, steps=steps, rho=args.rho,
            momentum=args.momentum,
            collected_by_pass={s["pass_id"]: s["collected_params"] for s in summaries},
            names_by_pass={s["pass_id"]: s["n_collected_params"] for s in summaries},
            selection_split=sel_split or "source_city_val"),
        "selected_config_path": sel_path,
        "pilot_verdict": verdict,
        "unadapted": {
            "iou_fg_tol2": una["miou_fg_tol2"],
            "mean_pred_fg_fraction": una["mean_pred_fg_fraction_final"],
            "miou_k2_twoclass": two_un["miou_k2_twoclass"],
            "two_class_detail": two_un,
        },
        "sar": {
            "iou_fg_tol2": sarp["miou_fg_tol2"],
            "delta_vs_unadapted_iou_fg_tol2": sarp["delta_vs_unadapted"],
            "miou_k2_twoclass": two_sf["miou_k2_twoclass"],
            "miou_k2_twoclass_step0": two_s0["miou_k2_twoclass"],
            "delta_vs_unadapted_miou_k2_twoclass":
                two_sf["miou_k2_twoclass"] - two_un["miou_k2_twoclass"],
            "delta_adaptation_only_miou_k2_twoclass":
                two_sf["miou_k2_twoclass"] - two_s0["miou_k2_twoclass"],
            "two_class_detail_final": two_sf,
            "two_class_detail_step0": two_s0,
            "mean_pred_fg_fraction_step0": sarp["mean_pred_fg_fraction_step0"],
            "mean_pred_fg_fraction_final": sarp["mean_pred_fg_fraction_final"],
            "n_images_with_zero_foreground_prediction":
                sarp["n_images_with_zero_foreground_prediction"],
            "n_reset_fires": sarp["n_reset_fires"],
            "n_images_pred_fg_below_0p01": sarp["n_images_pred_fg_below_0p01"],
            "frac_images_pred_fg_below_0p01": sarp["frac_images_pred_fg_below_0p01"],
            "diverged": sarp["diverged"],
            "divergence_note": sarp["divergence_note"],
            "n_updates_skipped": sarp["n_updates_skipped"],
            "n_images_state_changed_by_adaptation":
                sarp["n_images_state_changed_by_adaptation"],
        },
        "weight_drift": drift,
        "per_band": armf_sar_pass.per_band_blocks(rows, summaries, base_iou_by_image),
        "per_image_dump_path": str(dump_path),
        "sanity_checks": {
            "n450_gate": gate,
            "episodic_state_isolation": bool(episodic_all),
            "episodic_state_isolation_detail": {
                s["pass_id"]: {"ok": s["episodic_state_isolation"],
                               "mismatch_keys": s["episodic_mismatch_keys"]}
                for s in summaries},
            "collected_params_nonempty": all(s["n_collected_params"] >= 4
                                             for s in summaries),
            "step0_matches_unadapted": s0,
            "unadapted_vs_stored_test_results": {
                "computed": una["miou_fg_tol2"],
                "stored": stored,
                "difference": diff,
                "stored_source": str(tr_path),
                "stored_field": "detailed.boundary_tolerant.tolerant_2px.iou",
                "stored_read_error": stored_reason,
                "action": ("WARN only on the foreground-only field; the n450 "
                           "two-class gate above is the stop."),
            },
        },
        "caveats": caveats,
        "compute_note": armf_sar_meta.COMPUTE_NOTE,
        "seed": args.seed,
        "wall_clock_s": None,
        "s_per_image": None,
        "peak_gpu_mem_gb": max([s["peak_gpu_mem_gb"] for s in summaries
                                if s["peak_gpu_mem_gb"] is not None] or [None]),
        "per_pass": [{k: s[k] for k in (
            "pass_id", "config_label", "steps", "lr", "lr_multiplier",
            "miou_fg_tol2", "delta_vs_unadapted", "mean_pred_fg_fraction_step0",
            "mean_pred_fg_fraction_final", "n_reset_fires", "n_updates_skipped",
            "n_collected_params", "norm_mode", "episodic_state_isolation",
            "wall_clock_s", "s_per_image", "peak_gpu_mem_gb",
            "checkpoint_epoch")} for s in summaries],
        "provenance": armf_sar_meta.provenance_block(
            SCRIPT_PATH, seed_sites, len(images),
            extra={"smoke": bool(args.limit),
                   "config_path": str(cfg_path),
                   "config_sha256": sha256_file(cfg_path),
                   "legacy_run_dir": cell_dir.name,
                   "n450_run_json": str(run_json),
                   "grid_module_sha256": sha256_file(BASE / "python/armf_sar_grid.py"),
                   "naming_patch_module_sha256": sha256_file(
                       BASE / "python/armf_sar_naming_patch.py"),
                   "test_list_order": ("armc_train.city_split_files(city, "
                                       "res_dir, 'test') -- the fixed released "
                                       "order")}),
    }
    results["wall_clock_s"] = round(time.time() - t_start, 2)
    results["s_per_image"] = round(results["wall_clock_s"] /
                                   max(1, len(images) * len(specs)), 4)

    with armc_runtime.PhaseTimeout(args.timeout_s, out_dir,
                                   label="armF SAR n450 write") as tbx:
        tbx.phase("write_parquet")
        import pandas as pd
        pd.DataFrame(rows).to_parquet(dump_path, index=False)
        tbx.phase("write_results_json")
        results_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out_dir / ".DONE").touch()

    print("\n" + "=" * 78)
    print(f"ARM F SAR n450 DONE  {ARCH}/{args.city}/{args.res}   {len(rows)} "
          f"rows -> {dump_path}")
    print(f"  results        {results_path}")
    print(f"  unadapted      two-class mIoU K=2 {two_un['miou_k2_twoclass']:.4f}"
          f"   fg IoU K=2 {una['miou_fg_tol2']:.4f}")
    print(f"  adapted        two-class mIoU K=2 {two_sf['miou_k2_twoclass']:.4f}"
          f"   fg IoU K=2 {sarp['miou_fg_tol2']:.4f}")
    print(f"  adapted-unadap two-class mIoU K=2 "
          f"{two_sf['miou_k2_twoclass'] - two_un['miou_k2_twoclass']:+.6f}"
          f"   fg IoU K=2 {sarp['delta_vs_unadapted']:+.6f}")
    if drift:
        print(f"  weights moved  {drift['weights_changed']}   max "
              f"|adapted-start| {drift['max_abs_delta']:.6e} on "
              f"{drift['max_abs_delta_tensor']}")
    print(f"  n450 gate      {'PASS' if gate['pass'] else 'FAIL'}  difference "
          f"{gate['difference']:+.6f}")
    print(f"  wall           {results['wall_clock_s']:.0f}s")
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
