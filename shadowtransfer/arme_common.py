"""
Arm E — shared helpers for the frozen-backbone DINOv3 runs (Parts 1-2) and the
matched-learning-rate Experiment B rerun (Part 3).

Nothing here computes a metric. Every per-image quantity comes from
shadowtransfer.metrics via shadowtransfer.armc_dump.perimage_row, so arm E's
parquet is byte-schema-identical to arm C's.

Two things in this file are new code rather than lifted code, and both exist
because an existing function does not do what arm E needs:

  hard_freeze_backbone
      dinov3_backbone._freeze_stages cannot produce a fully frozen backbone at
      any value of frozen_stages. Measured on the real pretrained ViT-S/16 at
      frozen_stages=12 and 999, four tensors survive with requires_grad=True:

          backbone.dinov3.norm.weight   384    grad 7.37e+03
          backbone.dinov3.norm.bias     384    grad 6.77e-03
          backbone.dinov3.cls_token     384    grad 2.04e+01
          backbone.dinov3.mask_token    384    grad 0.00e+00

      dinov3.norm is the live one: get_intermediate_layers defaults to
      norm=True, so that final LayerNorm is applied to all four extracted
      feature maps (blocks 3/6/9/11) and its affine gain/shift is a per-channel
      transform of exactly the features the decoder consumes. mask_token's
      gradient is structurally zero (it enters only as cls_token + 0*mask_token).

      _freeze_stages is NOT edited — it produced published numbers and is the
      audit trail. This function runs after it and zeroes requires_grad on every
      remaining backbone parameter, giving the operator-specified definition:
      every backbone parameter has requires_grad = False.

  collect_probs
      experiment_utils.run_inference saves thresholded binary masks (argmax ->
      0/255 PNG). CONTRACT.md section 3 needs UNTHRESHOLDED probabilities, and
      section 2 requires probs.npz. Same forward path and the same
      extract_logits helper; only the probabilities are retained instead of
      being discarded.

Prediction recipe matches arm C exactly: p > 0.5 then filter_small_predictions
(min_pixels=10), via metrics.threshold_and_filter. For a 2-class softmax that
is identical to argmax over the class logits.
"""

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .armc_dump import (
    BAND_EXTRA,
    BAND_TOL,
    load_geo_meta,
    mark_failed,
    perimage_row,
    sha256_file,
    write_dump,
)
from .metrics import compute_per_image_recovery, threshold_and_filter

METRIC_MODULE = Path(__file__).with_name("metrics.py")

# Same prefixes experiment_utils uses, so "encoder" means the same thing in
# arm E as it does in the published Experiment A / B runs.
ENCODER_PREFIXES = {
    "mamnet": ["encoder.resnet_encoder.", "encoder.resnet.", "encoder."],
    "oglanet": ["encoder.resnet_encoder.", "encoder.resnet.", "encoder."],
    "dinov3": ["backbone.dinov3.", "backbone."],
    "dinov3frozen": ["backbone.dinov3.", "backbone."],
}

CITY_TO_FOLD = {"phoenix": 0, "miami": 1, "chicago": 2}
FOLD_TO_SOURCES = {0: ["chicago", "miami"], 1: ["chicago", "phoenix"], 2: ["miami", "phoenix"]}


# ----------------------------------------------------------------------------
# seeding
# ----------------------------------------------------------------------------

def seed_all(seed):
    """Seed python, numpy, torch and cuda. CLAUDE.md 'What every run must do'."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return {
        "seed": seed,
        "seed_python_random": seed,
        "seed_numpy": seed,
        "seed_torch": seed,
        "seed_torch_cuda": seed,
        "torch_deterministic_algorithms": False,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


# ----------------------------------------------------------------------------
# freezing and the parameter audit
# ----------------------------------------------------------------------------

def _is_backbone(name, arch="dinov3"):
    return any(name.startswith(p) for p in ENCODER_PREFIXES.get(arch, ["backbone."]))


def hard_freeze_backbone(model, arch="dinov3"):
    """Set requires_grad=False on EVERY backbone parameter.

    Runs after DINOv3Backbone.__init__ has already called _freeze_stages, and
    catches the tensors _freeze_stages leaves behind. Returns the list of names
    it had to newly freeze, which is the evidence that _freeze_stages was
    incomplete.
    """
    newly_frozen = []
    for name, p in model.named_parameters():
        if _is_backbone(name, arch) and p.requires_grad:
            p.requires_grad = False
            newly_frozen.append(name)
    return newly_frozen


def param_audit(model, arch="dinov3"):
    """Trainable / frozen parameter counts, split backbone vs decoder.

    `backbone_trainable_names` MUST be empty in a Part 1 frozen run. The caller
    asserts on it; sanity check 4 in the arm E spec.
    """
    bb_train_n = bb_frozen_n = dec_train_n = dec_frozen_n = 0
    bb_train_names = []
    for name, p in model.named_parameters():
        k = p.numel()
        if _is_backbone(name, arch):
            if p.requires_grad:
                bb_train_n += k
                bb_train_names.append({"name": name, "numel": int(k)})
            else:
                bb_frozen_n += k
        else:
            if p.requires_grad:
                dec_train_n += k
            else:
                dec_frozen_n += k
    return {
        "total_param_count": int(bb_train_n + bb_frozen_n + dec_train_n + dec_frozen_n),
        "trainable_param_count": int(bb_train_n + dec_train_n),
        "frozen_param_count": int(bb_frozen_n + dec_frozen_n),
        "backbone_trainable_param_count": int(bb_train_n),
        "backbone_frozen_param_count": int(bb_frozen_n),
        "decoder_trainable_param_count": int(dec_train_n),
        "decoder_frozen_param_count": int(dec_frozen_n),
        "backbone_trainable_names": bb_train_names,
    }


def print_param_audit(audit, label=""):
    print(f"--- parameter audit {label} ---", flush=True)
    for k in ("total_param_count", "trainable_param_count", "frozen_param_count",
              "backbone_trainable_param_count", "backbone_frozen_param_count",
              "decoder_trainable_param_count", "decoder_frozen_param_count"):
        print(f"    {k:38s} {audit[k]:>12,}", flush=True)
    if audit["backbone_trainable_names"]:
        print("    backbone params STILL TRAINABLE:", flush=True)
        for e in audit["backbone_trainable_names"]:
            print(f"        {e['name']:55s} {e['numel']:>10,}", flush=True)
    else:
        print("    backbone params still trainable: NONE", flush=True)


# ----------------------------------------------------------------------------
# inference retaining unthresholded probabilities
# ----------------------------------------------------------------------------

def collect_probs(model, dataloader, device, extract_logits, img_size=384,
                  limit=None):
    """Forward pass retaining P(shadow), the GT mask, and the filtered prediction.

    Returns dict: image_id -> {"prob": float32 (H,W), "gt": uint8 (H,W),
                               "pred": uint8 (H,W)}
    `limit` truncates to the first N images; used only by the stage-1 smoke run.
    """
    import torch.nn.functional as F

    model.eval()
    out = {}
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            fnames = batch["filename"]

            logits = extract_logits(model(images))
            if logits.shape[-2:] != (img_size, img_size):
                logits = F.interpolate(logits, size=(img_size, img_size),
                                       mode="bilinear", align_corners=False)
            prob = torch.softmax(logits, dim=1)[:, 1]

            for i, fn in enumerate(fnames):
                p = prob[i].detach().cpu().numpy().astype(np.float32)
                g = masks[i].detach().cpu().numpy()
                g = (np.squeeze(g) > 0).astype(np.uint8)
                out[str(fn)] = {
                    "prob": p,
                    "gt": g,
                    "pred": threshold_and_filter(p, min_pixels=10),
                }
                if limit is not None and len(out) >= limit:
                    return out
    return out


# ----------------------------------------------------------------------------
# dump emission
# ----------------------------------------------------------------------------

def build_rows(run_id, per_image, city, res, geo_meta, extra_cols=None):
    """One CONTRACT section 3 row per image, via armc_dump.perimage_row."""
    rows, probs = [], {}
    for image_id in sorted(per_image):
        d = per_image[image_id]
        row = perimage_row(run_id, image_id, city, res,
                           d["gt"], d["prob"], d["pred"],
                           meta_entry=geo_meta.get(image_id))
        if extra_cols:
            row.update(extra_cols.get(image_id, {}))
        rows.append(row)
        probs[image_id] = d["prob"].astype(np.float16)
    return rows, probs


def attach_recovery(rows, loco_iou_by_id, upper_iou_by_id, min_gap=0.02):
    """Per-image recovery ratio R, via metrics.compute_per_image_recovery.

    Writes recovery_R (null where the image was excluded), plus the two
    baseline IoUs it was computed against so the SSH box can re-derive it.
    Image-cluster bootstrap is deliberately NOT run here; aggregation is the
    SSH box's job (CLAUDE.md 'This box does NOT').
    """
    ids = [r["image_id"] for r in rows]
    have = [i for i in ids if i in loco_iou_by_id and i in upper_iou_by_id]
    if not have:
        for r in rows:
            r["recovery_R"] = None
            r["recovery_loco_iou"] = None
            r["recovery_upper_iou"] = None
        return {"computed": False, "reason": "no overlapping baseline image ids"}

    idx = [ids.index(i) for i in have]
    exp_i = [rows[j]["iou_tol"] for j in idx]
    loco_i = [loco_iou_by_id[i] for i in have]
    upper_i = [upper_iou_by_id[i] for i in have]

    R, valid, info = compute_per_image_recovery(exp_i, loco_i, upper_i, min_gap=min_gap)
    r_by_id = {have[v]: float(R[k]) for k, v in enumerate(valid)}

    for r in rows:
        iid = r["image_id"]
        r["recovery_R"] = r_by_id.get(iid)
        r["recovery_loco_iou"] = loco_iou_by_id.get(iid)
        r["recovery_upper_iou"] = upper_iou_by_id.get(iid)
    info["computed"] = True
    info["n_matched_ids"] = len(have)
    return info


def load_baseline_ious(dumps_root, run_id):
    """image_id -> iou_tol from an existing dump. Returns None if absent."""
    p = Path(dumps_root) / run_id / "perimage.parquet"
    if not (Path(dumps_root) / run_id / ".DONE").exists() or not p.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(p, columns=["image_id", "iou_tol"])
    return dict(zip(df["image_id"].astype(str), df["iou_tol"].astype(float)))


# ----------------------------------------------------------------------------
# run.json
# ----------------------------------------------------------------------------

def slurm_facts():
    return {
        "cluster": os.environ.get("ARME_CLUSTER", "delta"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "slurm_time_limit": os.environ.get("ARME_TIME_LIMIT"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def vram_facts():
    if not torch.cuda.is_available():
        return {"peak_vram_mb": None, "vram_total_mb": None}
    return {
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 1e6, 1),
        "vram_total_mb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
    }


def sha256_json(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def base_run_json(run_id, arch, method, mode, holdout, res, seed, arm="E"):
    return {
        "run_id": run_id,
        "dataset": "st",
        "arm": arm,
        "arch": arch,
        "method": method,
        "mode": mode,
        "rung": 450,
        "holdout": holdout,
        "res": res,
        "seed": seed,
        "metric_module_sha": sha256_file(METRIC_MODULE),
        "metric_module_path": "python/shadowtransfer/metrics.py",
        "git_hash": None,
        "git_hash_reason": "project root is not a git repository (CLAUDE.md)",
        "structure_cols": "deferred",
        "band_width_tol": BAND_TOL,
        "band_width_str": 0,
        "band_width_extra": BAND_EXTRA,
        "headline_metric": (
            "foreground-only IoU on the K=2 valid region (column iou_tol), "
            "per Appendix E; NOT the two-class mIoU of published Tables 30-32"),
        "prediction_recipe": "p > 0.5 then filter_small_predictions(min_pixels=10)",
        "aurc_definition": "aurc_trapz19",
        "aurc_definition_note": (
            "CONTRACT.md section 3 leaves the canonical AURC blank. aurc_fg/bg/all "
            "use aurc_trapz19; aurc_*_mean20 carry the SIB-trainer variant. "
            "Neither is declared canonical by this run."),
    }


def finish(out_dir, rows, probs, run, t0):
    run["wall_clock_s"] = round(time.time() - t0, 2)
    run["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run.update(vram_facts())
    df = write_dump(out_dir, rows, probs, run)
    print(f"\nWROTE {out_dir}  rows={len(df)}", flush=True)
    return df
