"""
Arm F, stage 2 — the 9-cell SAR grid. ONE CELL PER JOB.

One cell is one (architecture, held-out city) at high resolution on the published
vanilla LOCO checkpoint: an unadapted pass and one SAR pass at the (lr, steps)
that stage 1 selected on source-city validation, over the held-out city's 150 test
images in the fixed released order, augmentation off, batch size 1.

Every SAR number is a delta against the unadapted pass on the same checkpoint and
the same 150 images, computed in this run. The comparison against the cell's own
stored test_results.json is recorded and WARNed, never a stop (operator decision,
2026-07-30): no published cell value enters any comparison, so a disagreement
cannot move a reported number.

Config A only; see shadowtransfer/armf_sar_pass.py for why config B is not run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

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
from shadowtransfer.armf_sar_cells import (ARM, RES_DIR, WEAKNESS_IDS,      # noqa: E402
                                           enter_arch_tree, resolve_cell,
                                           stored_fg_iou_k2, test_images)

STAGE = "grid"


def read_selected(path, limit, fallback_mult, fallback_steps):
    """The grid runs the configuration stage 1 chose. A SMOKE run may fall back
    to a stated default so the machinery can be exercised before stage 1 lands;
    a real cell may not."""
    p = Path(path) if path else None
    if p and p.exists():
        d = json.loads(p.read_text())
        for k in ("lr", "lr_multiplier", "steps"):
            if k not in d:
                raise SystemExit(f"FATAL {p} has no '{k}'")
        return (float(d["lr"]), float(d["lr_multiplier"]), int(d["steps"]),
                d.get("pilot_verdict"), str(p.resolve()), d.get("selection_split"))
    if not limit:
        raise SystemExit(f"FATAL selected_config.json not found at {p}; a real "
                         f"grid cell may not guess the learning rate.")
    return (sar.SAR_LR_BASE * fallback_mult, float(fallback_mult),
            int(fallback_steps), "smoke_fallback", None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["mamnet", "oglanet", "dinov3"])
    ap.add_argument("--city", required=True, choices=["chicago", "miami", "phoenix"])
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--selected_config", default=None)
    ap.add_argument("--fallback_lr_mult", type=float, default=1.0)
    ap.add_argument("--fallback_steps", type=int, default=1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--rho", type=float, default=sar.SAM_RHO)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--timeout_s", type=int, default=900)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    res_dir = RES_DIR[args.res]
    cell_dir, ckpt, cfg_path, tr_path = resolve_cell(args.arch, args.city, res_dir)
    cfg = json.loads(cfg_path.read_text())
    lr, lr_mult, steps, verdict, sel_path, sel_split = read_selected(
        args.selected_config, args.limit, args.fallback_lr_mult, args.fallback_steps)

    specs = [armf_sar_pass.unadapted_spec(),
             armf_sar_pass.sar_spec(lr, lr_mult, steps)]

    out_dir = (Path(args.out_root) if args.out_root else BASE / "runs") / \
        f"armF_sar_{args.arch}_{args.city}_{res_dir}{args.tag}"
    results_path = out_dir / "results.json"
    dump_path = out_dir / "perstep.parquet"

    print("=" * 78)
    print(f"ARM F STAGE 2 — SAR GRID CELL   arch={args.arch}  "
          f"holdout={args.city}  res={args.res}")
    print(f"  cell dir        {cell_dir}")
    print(f"  checkpoint      {ckpt}")
    print(f"  config          {cfg_path}")
    print(f"  selected_config {sel_path}   verdict {verdict}")
    print(f"  lr              {lr:g}  ({lr_mult:g}x lr_base "
          f"{sar.SAR_LR_BASE})   steps {steps}")
    print(f"  E_0             {sar.e0_margin(2):.6f} nats   reset constant "
          f"{sar.reset_constant(2):.6f} nats")
    print(f"  norm mode       {armf_sar_meta.NORM_MODE}")
    print(f"  passes          {[s['pass_id'] for s in specs]}")
    print(f"  out             {out_dir}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    if (out_dir / ".DONE").exists():
        print(f"skip: {out_dir}/.DONE exists")
        return

    armc_runtime.preflight_write_targets(BASE, [results_path, dump_path],
                                        label="armF stage-2 output")
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from shadowtransfer.arme_common import seed_all
    seed_sites = seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enter_arch_tree(args.arch)
    images = test_images(args.arch, args.city, res_dir, args.limit)
    print(f"\nTEST SET {len(images)} images, fixed released order, augmentation "
          f"off, batch size 1; first {images[0]['image_id']}, last "
          f"{images[-1]['image_id']}", flush=True)

    summaries, rows = [], []
    base_iou_by_image, base_fg0_by_image = {}, {}
    for spec in specs:
        s, r = armf_sar_pass.run_pass(spec, args.arch, cfg, ckpt, device, images,
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

    # ---- the reproduction check: WARN, never a stop -------------------------
    stored, stored_reason = stored_fg_iou_k2(tr_path)
    diff = None if stored is None else una["miou_fg_tol2"] - stored
    if stored is None:
        print(f"WARN unadapted_vs_stored_test_results: {stored_reason}", flush=True)
    else:
        print(f"WARN unadapted_vs_stored_test_results  computed "
              f"{una['miou_fg_tol2']:.4f}  stored {stored:.4f}  difference "
              f"{diff:+.4f}. Recorded, not a stop (operator decision "
              f"2026-07-30). Every SAR number in this cell is a delta against "
              f"the unadapted pass on the same checkpoint and the same "
              f"{len(images)} images, so no published cell value enters any "
              f"comparison. Not investigated; no published cell recomputed.",
              flush=True)

    episodic_all = all(s["episodic_state_isolation"] for s in summaries)
    if not episodic_all:
        armc_runtime.mark_failed(out_dir, "episodic_isolation",
                                 "model state was not byte-identical between images")
        raise SystemExit("FATAL episodic state isolation failed; the arm F spec "
                         "stops the run. Per-pass detail is in the log above.")

    caveats = armf_sar_meta.base_caveats(
        lr=lr, lr_multiplier=lr_mult, steps=steps, rho=args.rho,
        momentum=args.momentum, config_a_collect="norm_all")
    caveats.append(
        f"The (lr, steps) pair was selected on the fold's SOURCE-CITY validation "
        f"split pooled over MAMNet and DINOv3 (stage 1, verdict '{verdict}'), "
        f"never on held-out-city images. OGLANet was not in the pilot and runs "
        f"here at the pair selected on the other two architectures.")
    if args.limit:
        caveats.append(f"SMOKE RUN: {args.limit} images at a fallback "
                       f"configuration. Not a result.")

    results = {
        "arm": ARM,
        "stage": STAGE,
        "weakness_ids": WEAKNESS_IDS,
        "target_access": armf_sar_meta.TARGET_ACCESS,
        "target_access_detail": armf_sar_meta.TARGET_ACCESS_DETAIL,
        "arch": args.arch,
        "holdout_city": args.city,
        "resolution": args.res,
        "parent_run_id": None,
        "parent_run_dir": cell_dir.name,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
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
            "miou_tol2": una["miou_fg_tol2"],
            "mean_pred_fg_fraction": una["mean_pred_fg_fraction_final"],
        },
        "sar": {
            "miou_tol2": sarp["miou_fg_tol2"],
            "delta_vs_unadapted": sarp["delta_vs_unadapted"],
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
        "per_band": armf_sar_pass.per_band_blocks(rows, summaries, base_iou_by_image),
        "per_image_dump_path": str(dump_path),
        "sanity_checks": {
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
                "action": ("WARN only, never a stop (operator decision "
                           "2026-07-30). Not investigated; no published cell "
                           "recomputed."),
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
                   "test_list_order": ("armc_train.city_split_files(city, "
                                       "res_dir, 'test') — the fixed released "
                                       "order")}),
    }
    results["wall_clock_s"] = round(time.time() - t_start, 2)
    results["s_per_image"] = round(results["wall_clock_s"] /
                                   max(1, len(images) * len(specs)), 4)

    with armc_runtime.PhaseTimeout(args.timeout_s, out_dir, label="armF stage-2 write") as tbx:
        tbx.phase("write_parquet")
        import pandas as pd
        pd.DataFrame(rows).to_parquet(dump_path, index=False)
        tbx.phase("write_results_json")
        results_path.write_text(json.dumps(results, indent=2, default=str) + "\n")
    (out_dir / ".DONE").touch()

    print("\n" + "=" * 78)
    print(f"ARM F STAGE 2 DONE  {args.arch}/{args.city}/{args.res}   "
          f"{len(rows)} rows -> {dump_path}")
    print(f"  results        {results_path}")
    print(f"  unadapted      {una['miou_fg_tol2']:.4f}   pred_fg "
          f"{una['mean_pred_fg_fraction_final']:.5f}")
    print(f"  sar            {sarp['miou_fg_tol2']:.4f}   delta "
          f"{sarp['delta_vs_unadapted']:+.4f}   pred_fg "
          f"{sarp['mean_pred_fg_fraction_step0']:.5f} -> "
          f"{sarp['mean_pred_fg_fraction_final']:.5f}   zeroFG "
          f"{sarp['n_images_with_zero_foreground_prediction']}   resets "
          f"{sarp['n_reset_fires']}   diverged {sarp['diverged']}")
    print(f"  vs stored      computed {una['miou_fg_tol2']:.4f}  stored "
          f"{stored}  difference {diff}")
    print(f"  wall           {results['wall_clock_s']:.0f}s")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
