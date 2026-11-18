"""
Arm F SAR close-out — the parts of results.json that must read identically in
every cell: the caveat list, the `sar_config` block, and provenance.

One place, so a caveat cannot drift between the pilot and the grid.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

from shadowtransfer import sar
from shadowtransfer.armc_dump import sha256_file
from shadowtransfer.armf_sar_cells import BASE, NORM_MODE, NUM_CLASSES

TARGET_ACCESS = "deployment_time"
TARGET_ACCESS_DETAIL = (
    "SAR fits normalisation-layer affine parameters at deployment on each test "
    "image's own prediction. No held-out-city labels and no source data are "
    "touched. Episodic: state is restored before every image, so nothing "
    "accumulates across the target set. The learning rate and step count were "
    "selected on the SOURCE-CITY validation split of the fold, never on "
    "held-out-city images.")

COMPUTE_NOTE = (
    "GPU job on gpuA100x4; inference and test-time adaptation only, no training. "
    "Delta carries no CPU account on this allocation, so any CPU-only phase "
    "still holds a GPU.")

METRIC_BLOCK = {
    "name": "iou_fg_only_tolerant",
    "spec_label": "miou_boundary_tolerant",
    "K": 2,
    "raw_counts_at_K": [0, 2, 5],
    "aggregation": "per-image scalars macro-averaged over the pass; counts are "
                   "never micro-pooled across images",
}


def base_caveats(lr, lr_multiplier, steps, rho, momentum, config_a_collect,
                 unadapted_vs_stored=None):
    """The caveats every arm F SAR results.json carries, in a fixed order."""
    c = [
        "SAR run with normalisation layers in eval mode on source running "
        "statistics and affine parameters adapted; the as-released "
        "batch-statistics path is undefined for MAMNet at batch size 1 and is "
        "not run.",

        "Every SAR number here is a delta against the unadapted pass on the same "
        "checkpoint and the same images, computed in the same run. No published "
        "cell value enters any comparison.",

        "The reproduction check against the cell's own stored test_results.json "
        "is recorded and warned about, never a stop (operator decision, "
        "2026-07-30). It is reported in "
        "sanity_checks.unadapted_vs_stored_test_results and no published cell was "
        "recomputed to investigate a difference.",

        "Config B (normalisation layers forced onto batch statistics) is dropped "
        "from this arm. Config A is therefore one variable across all three "
        "architectures, and because no forward path changes before the first "
        "adaptation step, step0_matches_unadapted is a real check rather than a "
        "codepath tell.",

        f"The SAR learning rate is not an upstream-given value at this batch "
        f"size. lr_base = {sar.SAR_LR_BASE} is upstream SAR's ViT default used "
        f"unscaled; upstream additionally scales lr by batch_size/64, which at "
        f"the episodic batch size of 1 gives 3.9e-6 and no adaptation at all. "
        f"The ladder {{1x, 10x, 100x}} * lr_base was selected on the SOURCE-CITY "
        f"validation split of the fold. This run uses lr={lr} "
        f"({lr_multiplier}x lr_base), {steps} step(s), inside SAM at rho={rho} "
        f"over SGD(momentum={momentum}).",

        "The model-recovery threshold is 0.0201 nats, the same fraction of "
        "ln(2) that upstream's 0.2 is of ln(1000). It is derived from the "
        "entropy bound alone and was never swept against target mIoU.",

        "The model is held in eval() rather than train(). MAMNet's forward "
        "returns an auxiliary dict when self.training is True and its "
        "AuxiliaryModule carries p=0.3 dropout, either of which would change "
        "what is being measured; OGLANet's forward likewise returns six "
        "deep-supervision heads in train mode and the final prediction in eval.",

        "Where the reliability filter admits no pixel, upstream takes a NaN mean "
        "and calls backward() on it. Here the SAM ascent step is rolled back, no "
        "base step is taken, and the step is recorded with update_skipped set. "
        "NaN parameters would end a pass silently.",

        f"Parameter collection keeps upstream's skip list verbatim "
        f"({', '.join(sar.SKIP_SUBSTRINGS)}, plus a module named exactly 'norm') "
        f"and collects the affines of every remaining BatchNorm2d, LayerNorm and "
        f"GroupNorm (config_a_collect='{config_a_collect}'). The full collected "
        f"name list is in sar_config.collected_params for every pass.",

        "CONTRACT.md defines no schema for this artifact. Section 6's results "
        "JSON is the CC-SSH analysis product and carries a different field list; "
        "section 3's perimage.parquet is one row per image for a trained cell, "
        "and this dump is one row per (pass, image, adaptation step). The schema "
        "used is the arm F spec's own field list, verbatim, with section 4's "
        "mandatory target_access carried. The same deviation was recorded in the "
        "2026-07-30 pilot's results.json and is carried forward unchanged.",

        "metric.name is the project's own name for the quantity, foreground-only "
        "IoU with a +/-2 px don't-care band, macro-averaged over images. The arm "
        "F spec's field list labels the same quantity miou_boundary_tolerant; "
        "both names are recorded and no second metric is computed.",

        "per_band blocks carry an added pass_id; with more than one pass the arm "
        "F field list is not readable without one.",
    ]
    if unadapted_vs_stored:
        c.append(unadapted_vs_stored)
    return c


def sar_config_block(lr, lr_multiplier, steps, rho, momentum, collected_by_pass,
                     names_by_pass, selection_split="source_city_val"):
    return {
        "E0": sar.e0_margin(NUM_CLASSES),
        "reset_constant_em": sar.reset_constant(NUM_CLASSES),
        "reset_constant_derivation": sar.reset_constant_derivation(NUM_CLASSES),
        "rho": rho,
        "episodic": True,
        "lr": lr,
        "lr_base": sar.SAR_LR_BASE,
        "lr_multiplier": lr_multiplier,
        "lr_selection_split": selection_split,
        "steps": steps,
        "norm_mode": NORM_MODE,
        "collected_params": collected_by_pass,
        "n_collected_params": names_by_pass,
        "num_classes": NUM_CLASSES,
        "optimizer": f"SAM(rho={rho}) over SGD(lr={lr}, momentum={momentum})",
        "ema_momentum": sar.EMA_MOMENTUM,
        "skip_list": list(sar.SKIP_SUBSTRINGS) + ["<module named exactly 'norm'>"],
        "train_mode": "eval (see caveats)",
    }


def provenance_block(script_path, seed_sites, n_images, extra=None):
    import torch
    p = {
        "seed_sites": seed_sites,
        "n_images": n_images,
        "metric_module_path": "shadowtransfer/metrics.py",
        "metric_module_sha256": sha256_file(BASE / "shadowtransfer/metrics.py"),
        "sar_module_sha256": sha256_file(BASE / "shadowtransfer/sar.py"),
        "pass_module_sha256": sha256_file(BASE / "shadowtransfer/armf_sar_pass.py"),
        "cells_module_sha256": sha256_file(BASE / "shadowtransfer/armf_sar_cells.py"),
        "script_path": str(script_path),
        "script_sha256": sha256_file(Path(script_path)),
        "prediction_recipe": "p > 0.5 then filter_small_predictions(min_pixels=10)",
        "batch_size": 1,
        "augmentation": False,
        "git_hash": None,
        "git_hash_reason": "project root is not a git repository (CLAUDE.md)",
        "cluster": os.environ.get("ARMF_CLUSTER", "delta"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_time_limit_requested": os.environ.get("ARMF_TIME_LIMIT"),
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        p.update(extra)
    return p
