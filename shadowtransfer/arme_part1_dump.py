"""
Arm E Part 1 — dump-phase helpers: forward the trainer's own test loader,
build the CONTRACT.md section 3 rows, and pool them into a small per-cell
test_results.json.

Every per-image quantity comes from `armc_dump.perimage_row`
(shadowtransfer.metrics underneath); nothing here computes a metric.
`collect_probs`/`build_rows`/`finish` are arme_common's -- they were written
already for this exact frozen-backbone pipeline (see arme_common's own
docstring) and are reused unchanged.
"""

import numpy as np

from . import arme_common as common


def dump_cell(trainer, cell, run_id_str, base, run_extra, out_dir, t0, limit=None):
    """Forward `trainer.dataloaders['test']` (the SAME loader training
    selected on, never re-built), write the CONTRACT dump, return (df, rows).
    """
    device = trainer.device
    per_image = common.collect_probs(
        trainer.model, trainer.dataloaders["test"], device,
        extract_logits=lambda out: out, img_size=384, limit=limit)
    geo_meta = common.load_geo_meta(base)
    rows, probs = common.build_rows(run_id_str, per_image, cell["city"], "hr", geo_meta)

    run = common.base_run_json(
        run_id_str, "dinov3frozen", cell["method"], cell["mode"],
        cell["city"], "hr", 0, arm="E")
    run.update(run_extra)

    df = common.finish(out_dir, rows, probs, run, t0)
    return df, rows


def build_test_results_summary(rows):
    """Pooled TP/FP/FN/TN at K=0/2/5 plus the per-image mean foreground IoU
    at K=2 -- the task's own metric spec ("foreground-only K=2, emitting raw
    TP/FP/FN counts at K=0/2/5"). Pure aggregation of counts `perimage_row`
    already computed; not a new statistical claim (CLAUDE.md: an execution-
    fact aggregation a job runs on itself is not analysis)."""

    def pool(suffix):
        return {
            "tp": int(sum(r[f"tp_{suffix}"] for r in rows)),
            "fp": int(sum(r[f"fp_{suffix}"] for r in rows)),
            "fn": int(sum(r[f"fn_{suffix}"] for r in rows)),
            "tn": int(sum(r[f"tn_{suffix}"] for r in rows)),
        }

    fg_iou_tol = np.asarray([r["iou_tol"] for r in rows], dtype=np.float64)
    return {
        "n_images": len(rows),
        "headline_metric": "foreground-only IoU, K=2 (per-image mean, percent)",
        "fg_iou_K2_per_image_mean_pct": float(fg_iou_tol.mean()) * 100.0,
        "K0_strict_counts": pool("str"),
        "K2_tolerant_counts": pool("tol"),
        "K5_extra_counts": pool("k5"),
        "n_images_empty_fg_valid": int(sum(1 for r in rows if r["empty_fg_valid"])),
    }
