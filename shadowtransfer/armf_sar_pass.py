"""
Arm F SAR close-out — one pass over one image set, and the per-band roll-up.

CONFIG A ONLY
-------------
Every pass in this arm holds normalisation layers in eval mode on their SOURCE
running statistics and adapts only their affine parameters. `sar.BN_BATCH_STATS`
is never requested from here. That path is undefined for MAMNet at batch size 1:
MSCAF's branch 5 pools to 1x1, and `F.batch_norm` refuses batch statistics at
one value per channel (`python/mamnet/models/mscaf.py:116`, job 20592854). It is
architectural, not a wrapper bug, and it is not run.

Holding config A across all three architectures makes the normalisation mode one
variable rather than three: DINOv3's ViT LayerNorms and the five decoder
BatchNorm2d layers, and MAMNet's and OGLANet's BatchNorm2d layers, are all in
eval on source statistics with their affines collected. A consequence worth
having: no forward path changes before the first adaptation step, so
`step0_matches_unadapted` is a real check on every architecture instead of a
codepath tell.
"""

import time

import numpy as np

from shadowtransfer import sar
from shadowtransfer.armc_dump import iou_from_counts
from shadowtransfer.armf_sar_cells import (BANDS, FG_COLLAPSE_FRACTION,
                                           FG_COLLAPSE_IMAGE_SHARE, NORM_MODE,
                                           NUM_CLASSES, band_counts, build_model,
                                           load_checkpoint_strict)
from shadowtransfer.metrics import threshold_and_filter


def make_spec(pass_id, steps, lr, lr_multiplier, label):
    return {"pass_id": pass_id, "steps": int(steps), "lr": lr,
            "lr_multiplier": lr_multiplier, "config_label": label}


def unadapted_spec():
    return make_spec("unadapted", 0, None, None,
                     "unadapted (SAR off; published inference configuration)")


def sar_spec(lr, lr_multiplier, steps):
    return make_spec(
        f"sar_lr{lr_multiplier:g}x_s{steps}", steps, lr, lr_multiplier,
        f"SAR config A ({NORM_MODE}), lr={lr:g} ({lr_multiplier:g}x lr_base), "
        f"{steps} step(s)")


def run_pass(spec, arch, cfg, ckpt_path, device, images, rho=sar.SAM_RHO,
             momentum=0.9, progress_every=25):
    """One pass over `images`. Returns (summary, rows).

    Episodic and mandatory: model state and the entropy moving average are
    restored before every image, and the restored state is byte-compared against
    the pristine snapshot on every image. Nothing crosses images.
    """
    import torch

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = build_model(arch, cfg, device)
    ck_epoch = load_checkpoint_strict(model, ckpt_path, device)
    model, bn_modules = sar.configure_model(model, sar.BN_EVAL_SOURCE_STATS)

    params, names, skipped = sar.collect_params(model)
    if spec["steps"] > 0:
        for p in params:
            p.requires_grad_(True)

    margin = sar.e0_margin(NUM_CLASSES)
    reset_const = sar.reset_constant(NUM_CLASSES)
    pristine = sar.snapshot_state(model)

    def optimizer_factory():
        return sar.make_optimizer(params, lr=spec["lr"], momentum=momentum, rho=rho)

    print(f"\n{'=' * 78}\nPASS {spec['pass_id']}  ({spec['config_label']})\n"
          f"  steps            {spec['steps']}   lr {spec['lr']}\n"
          f"  norm_mode        {NORM_MODE}   ({len(bn_modules)} BatchNorm2d modules)\n"
          f"  collected params {len(names)}  (skipped {len(skipped)} norm modules "
          f"by upstream's skip list)\n"
          f"  checkpoint epoch {ck_epoch}\n{'=' * 78}", flush=True)
    for n in names:
        print(f"    collect  {n}", flush=True)
    for n in skipped:
        print(f"    skip     {n}", flush=True)

    if len(names) < 4:
        raise SystemExit(
            f"FATAL pass {spec['pass_id']} collected {len(names)} parameters "
            f"({names}); the arm F spec stops the run below 4.")

    rows = []
    episodic_ok, episodic_bad = True, []
    n_changed_by_adaptation, n_reset_fires = 0, 0

    for idx, im in enumerate(images):
        if idx > 0:
            # Before restoring: did the previous image's adaptation move any
            # parameter at all? A SAR pass where this stays 0 adapted nothing.
            unchanged, _ = sar.state_matches(model, pristine)
            if not unchanged:
                n_changed_by_adaptation += 1
        sar.restore_state(model, pristine)
        ok, bad = sar.state_matches(model, pristine)
        if not ok:
            episodic_ok = False
            episodic_bad.extend(bad[:5])

        x = im["x"].to(device, non_blocking=True)
        recs, probs = sar.adapt_image(model, x, spec["steps"], margin,
                                      reset_const, optimizer_factory, pristine)

        for r in recs:
            prob = probs[r["step"]]
            pred = threshold_and_filter(prob, min_pixels=10)
            c = band_counts(pred, im["gt"])
            iou2 = float(iou_from_counts(c["tp_k2"], c["fp_k2"], c["fn_k2"]))
            if r["reset_fired"]:
                n_reset_fires += 1
            row = {
                "pass_id": spec["pass_id"],
                "config_label": spec["config_label"],
                "steps": spec["steps"],
                "lr": spec["lr"],
                "lr_multiplier": spec["lr_multiplier"],
                "arch": arch,
                "image_id": im["image_id"],
                "step": r["step"],
                "pred_fg_fraction": float(pred.mean()),
                "true_fg_fraction": im["true_fg_fraction"],
                "coverage_band": im["coverage_band"],
                "mean_entropy": r["mean_entropy"],
                "n_pixels_passing_E0": r["n_pixels_passing_E0"],
                "frac_pixels_passing_E0": r["frac_pixels_passing_E0"],
                "ema_entropy": r["ema_entropy"],
                "reset_fired": bool(r["reset_fired"]),
                "update_skipped": r["update_skipped"],
                "iou_fg_tol2": iou2,
                "n_pixels": r["n_pixels"],
            }
            row.update(c)
            rows.append(row)

        del x
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"    {spec['pass_id']}: {idx + 1}/{len(images)} images, "
                  f"{time.time() - t0:.0f}s", flush=True)

    wall = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / (1024 ** 3)
               if torch.cuda.is_available() else None)

    final = [r for r in rows if r["step"] == spec["steps"]]
    step0 = [r for r in rows if r["step"] == 0]
    n_low = int(sum(1 for r in final
                    if r["pred_fg_fraction"] < FG_COLLAPSE_FRACTION))
    summary = {
        "pass_id": spec["pass_id"],
        "config_label": spec["config_label"],
        "steps": spec["steps"],
        "lr": spec["lr"],
        "lr_multiplier": spec["lr_multiplier"],
        "miou_fg_tol2": 100.0 * float(np.mean([r["iou_fg_tol2"] for r in final])),
        "delta_vs_unadapted": None,
        "mean_pred_fg_fraction_final": float(np.mean([r["pred_fg_fraction"] for r in final])),
        "mean_pred_fg_fraction_step0": float(np.mean([r["pred_fg_fraction"] for r in step0])),
        "n_images_with_zero_foreground_prediction":
            int(sum(1 for r in final if r["pred_fg_fraction"] == 0.0)),
        "n_images_pred_fg_below_0p01": n_low,
        "frac_images_pred_fg_below_0p01": n_low / float(len(final)),
        "n_reset_fires": int(n_reset_fires),
        "wall_clock_s": round(wall, 2),
        "s_per_image": round(wall / max(1, len(images)), 3),
        "peak_gpu_mem_gb": None if peak_gb is None else round(peak_gb, 3),
        "collected_params": names,
        "n_collected_params": len(names),
        "skipped_norm_modules": skipped,
        "norm_mode": NORM_MODE,
        "n_bn2d_modules": len(bn_modules),
        "checkpoint_epoch": ck_epoch,
        "episodic_state_isolation": bool(episodic_ok),
        "episodic_mismatch_keys": episodic_bad[:20],
        "n_images_state_changed_by_adaptation": int(n_changed_by_adaptation),
        "n_updates_skipped": int(sum(1 for r in rows if r["update_skipped"])),
    }

    print(f"  -> {spec['pass_id']}  fg IoU K=2 {summary['miou_fg_tol2']:.4f}   "
          f"pred_fg step0 {summary['mean_pred_fg_fraction_step0']:.5f} -> final "
          f"{summary['mean_pred_fg_fraction_final']:.5f}   zero-fg images "
          f"{summary['n_images_with_zero_foreground_prediction']}   "
          f"fg<{FG_COLLAPSE_FRACTION} images {n_low}   resets "
          f"{summary['n_reset_fires']}   {wall:.0f}s "
          f"({summary['s_per_image']:.2f}s/img)", flush=True)

    del model, pristine, params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, rows


def mark_divergence(summaries):
    """The arm F divergence guard, applied to the SAR passes only.

    A candidate whose final-step predicted foreground fraction falls below
    FG_COLLAPSE_FRACTION on more than FG_COLLAPSE_IMAGE_SHARE of the images is
    marked `diverged` and excluded from selection regardless of mIoU. That is
    the foreground-collapse signal and it is a result, not an error. The
    unadapted pass carries the same statistic so the reader can see how much of
    it the data already supplies, but is never marked diverged.
    """
    for s in summaries:
        if s["pass_id"] == "unadapted":
            s["diverged"] = False
            s["divergence_note"] = ("baseline; the guard is not applied to the "
                                    "unadapted pass")
            continue
        s["diverged"] = bool(s["frac_images_pred_fg_below_0p01"]
                             > FG_COLLAPSE_IMAGE_SHARE)
        s["divergence_note"] = (
            f"{s['n_images_pred_fg_below_0p01']} images "
            f"({100 * s['frac_images_pred_fg_below_0p01']:.1f}%) predict a "
            f"foreground fraction below {FG_COLLAPSE_FRACTION}; the guard fires "
            f"above {100 * FG_COLLAPSE_IMAGE_SHARE:.0f}%")
        if s["diverged"]:
            print(f"WARN divergence guard fired on {s['pass_id']}: "
                  f"{s['divergence_note']}", flush=True)
    return summaries


def per_band_blocks(rows, summaries, base_iou_by_image):
    """One block per (pass, coverage band). `pass_id` is added to the arm F field
    list: with more than one pass a band block without one cannot be read."""
    out = []
    for s in summaries:
        pid, S = s["pass_id"], s["steps"]
        sub = [r for r in rows if r["pass_id"] == pid]
        for band in BANDS:
            ids = sorted({r["image_id"] for r in sub if r["coverage_band"] == band})
            if not ids:
                continue
            by_step = []
            for st in range(S + 1):
                v = [r["pred_fg_fraction"] for r in sub
                     if r["step"] == st and r["coverage_band"] == band]
                by_step.append(float(np.mean(v)) if v else None)
            fin = {r["image_id"]: r for r in sub
                   if r["step"] == S and r["coverage_band"] == band}
            out.append({
                "pass_id": pid,
                "coverage_band": band,
                "n_images": len(ids),
                "mean_pred_fg_fraction_by_step": by_step,
                "mean_true_fg_fraction": float(np.mean(
                    [fin[i]["true_fg_fraction"] for i in ids])),
                "mean_delta_iou": float(np.mean(
                    [fin[i]["iou_fg_tol2"] - base_iou_by_image[i] for i in ids])),
            })
    return out


def step0_check(rows, summaries, base_fg0_by_image):
    """`step0_matches_unadapted`, per pass. Under config A no forward path
    changes before adaptation, so this is expected to hold everywhere; a
    mismatch is WARNed and recorded, never a stop."""
    out = {}
    for s in summaries:
        if s["pass_id"] == "unadapted":
            continue
        sub = [r for r in rows if r["pass_id"] == s["pass_id"] and r["step"] == 0]
        d = [abs(r["pred_fg_fraction"] - base_fg0_by_image[r["image_id"]])
             for r in sub]
        n_bad = int(sum(1 for x in d if x > 0.0))
        out[s["pass_id"]] = {"n_images_mismatched": n_bad,
                             "max_abs_delta": float(max(d)) if d else 0.0,
                             "ok": n_bad == 0}
        if n_bad:
            print(f"WARN step0_matches_unadapted false on {s['pass_id']}: "
                  f"{n_bad} images mismatched, max abs delta "
                  f"{max(d):.6f}. Under config A no forward path changes before "
                  f"adaptation, so this is unexpected. Recorded, not a stop.",
                  flush=True)
    return out
