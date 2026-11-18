"""
Arm C — the per-image dump builder, shared by every arm-C job.

One implementation of the CONTRACT.md §3 row so the trained rungs (225, 900)
and the CPU-recomputed rung 450 write byte-identical schemas. Nothing in here
computes a metric; every quantity comes from shadowtransfer.metrics.

Arm C's headline metric is FOREGROUND-ONLY IoU with a +-2 px don't-care band
(K = 2), per Appendix E. That is the `iou_tol` column (and `iou_fg_tol_pct`,
the same number in percent). The two-class mIoU columns that reproduce the
published Tables 30-32 are also written, so the two definitions can be compared
without another pass; they are NOT arm C's metric.

Band widths written per row:
    K = 0  -> *_str   (CONTRACT §3 strict)
    K = 2  -> *_tol   (CONTRACT §3 tolerant; arm C's band)
    K = 5  -> *_k5    (arm C addition: counts are free and let the band choice
                       be revisited without retraining)
"""
# =============================================================================
# RECONSTRUCTED 2026-07-28. Read this before trusting a comment below.
#
# This file was truncated to 0 bytes on 2026-07-28 when a patcher was killed
# mid-write, and no source copy existed on any filesystem. It was rebuilt from
# the surviving bytecode,
# python/shadowtransfer/__pycache__/armc_dump.cpython-310.pyc (CPython 3.10,
# compiled 13:24 that day, the last-imported version).
#
# What that guarantees, and what it does not:
#   - The CODE is verified. Every function was compared instruction-for-
#     instruction against the .pyc: same opcodes, same arguments, same
#     constants, same order. Docstrings survive in bytecode and are original.
#   - The COMMENTS below are NOT original. Comments are not stored in bytecode.
#     Everything from here down that is a `#` comment was re-authored on
#     2026-07-28 from CLAUDE.md's lifting table. Treat it as a 2026-07-28
#     claim about provenance, not as the original author's words.
#   - LINE NUMBERS shift by exactly +50 against the .pyc, because this
#     block was inserted. The body layout below is otherwise unchanged, which
#     is what makes the offset constant and the comparison checkable.
#
# PROVENANCE OF THE QUANTITIES (CLAUDE.md, "Import the metric module").
# Nothing in this file computes a metric. Each name imported from .metrics
# below was lifted verbatim into that module from:
#   per_image_counts, valid_mask, tp/fp/tn/fn_tol, n_valid_px, n_gt_fg_valid,
#   empty_fg_valid, logit_means_confusion
#       <- python/mamnet/utils/evaluation_detailed.py  (078bcc7e...,
#          byte-identical across all three arch trees)
#   the *_str strict counts
#       <- python/mamnet/utils/metrics.py  (04d778df..., byte-identical
#          across all three arch trees)
#   per_image_rc, aurc_trapz19, pbar_populations, n_pred_pos
#       <- python/final_loco/composition_analysis/
#          s1_risk_coverage_select_predict_gap.py  (32d53b1f...)
#   per_image_rc_ce, aurc_ce_trapz19
#       <- python/final_loco/composition_analysis/f5_cross_entropy_aurc.py
#          (cb9d7f84...)
#   stratified_ece
#       <- python/final_loco/composition_analysis/
#          s2_f5_pop_stratified_ece.py  (cef72707...)
#   aurc_mean20 population construction
#       <- p0a_recompute.py, as _mean20_populations' own surviving docstring
#          states. That docstring is original; this line only repeats it.
#
# The wall_clock_s `finalize` hook that the truncating write was adding is NOT
# present and is not to be re-added: the operator cancelled it on 2026-07-28.
# write_dump below is the pre-incident four-argument function.
# =============================================================================

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import (
    aurc_ce_trapz19,
    aurc_mean20,
    aurc_trapz19,
    empty_fg_valid,
    logit_means_confusion,
    n_gt_fg_valid,
    n_pred_pos,
    pbar_populations,
    per_image_counts,
    per_image_rc,
    per_image_rc_ce,
    stratified_ece,
)

EPS_IOU = 1e-10
MIN_POP = 50

# Structural columns are deferred, not dropped: run.json records
# structure_cols="deferred" and these ride along as nulls so the schema is
# stable across arms.
DEFERRED_COLS = [
    "n_gt_blobs", "n_pred_blobs", "gt_blob_area_median", "small_object_frac",
    "fn_px_by_cat", "tp_px_by_cat", "gt_px_by_cat",
    "fp_px_near_gt", "fp_px_far", "gt_contrast_median",
]

BAND_TOL = 2
BAND_EXTRA = 5


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def iou_from_counts(tp, fp, fn):
    """Foreground-only IoU as a fraction. Arm C's metric at K = 2."""
    return float(tp / (tp + fp + fn + EPS_IOU))


def miou_from_counts(tp, fp, tn, fn):
    """Per-image 2-class mean IoU: the published Table 30-32 decision metric.

    Foreground IoU and background IoU under the ShadowMetrics denominators
    (fg: tp/(tp+fp+fn), bg: tn/(tn+fp+fn)), averaged. Percent. Written for
    comparability with the published tables; NOT arm C's metric.
    """
    fg = tp / (tp + fp + fn + EPS_IOU)
    bg = tn / (tn + fp + fn + EPS_IOU)
    return float(fg * 100), float(bg * 100), float((fg + bg) / 2.0 * 100)


def _mean20_populations(gt, prob, mode):
    """Feed aurc_mean20 the SAME pixel population per_image_rc uses, so the two
    AURC definitions differ only in their integration rule, not in their data.
    Lifted verbatim from p0a_recompute.py, whose conventions are the published
    composition_analysis ones.
    """
    gt_flat = np.asarray(gt).ravel().astype(bool)
    p_flat = np.asarray(prob).ravel()
    if mode == "gt_shadow":
        if gt_flat.sum() < MIN_POP:
            return None
        p = p_flat[gt_flat]
        conf, correct = p, p > 0.5
    elif mode == "gt_bg":
        if (~gt_flat).sum() < MIN_POP:
            return None
        p = p_flat[~gt_flat]
        conf, correct = 1.0 - p, p <= 0.5
    elif mode == "all_pixels":
        conf = np.maximum(p_flat, 1.0 - p_flat)
        correct = (p_flat > 0.5) == gt_flat
    else:
        raise ValueError(mode)
    if len(conf) < MIN_POP:
        return None
    return conf, correct.astype(np.float64)


def _aurc_mean20_or_nan(gt, prob, mode):
    pop = _mean20_populations(gt, prob, mode)
    if pop is None:
        return float("nan")
    return float(aurc_mean20(*pop))


def _aurc_trapz19_or_nan(gt, prob, mode):
    return aurc_trapz19(per_image_rc(gt, prob, mode))


def _aurc_ce_or_nan(gt, prob, mode):
    rc = per_image_rc_ce(gt, prob, mode)
    if rc is None:
        return float("nan")
    return aurc_ce_trapz19(rc)


def perimage_row(run_id, image_id, city, res, gt, prob, pred, meta_entry=None):
    """One CONTRACT §3 row.

    Args:
        gt   : uint8 (H, W), {0, 1}
        prob : float (H, W), UNTHRESHOLDED foreground probability
        pred : uint8 (H, W), the final prediction (argmax + <10 px filter)
    """
    gt = np.asarray(gt)
    prob = np.asarray(prob, dtype=np.float32)
    pred = np.asarray(pred)
    m = meta_entry or {}

    c = per_image_counts(pred, gt, boundary_tolerance=BAND_TOL)
    c5 = per_image_counts(pred, gt, boundary_tolerance=BAND_EXTRA)
    vmask = c["valid_mask"]

    fg_t, bg_t, mi_t = miou_from_counts(
        c["tp_tol"], c["fp_tol"], c["tn_tol"], c["fn_tol"])
    fg_s, bg_s, mi_s = miou_from_counts(
        c["tp_str"], c["fp_str"], c["tn_str"], c["fn_str"])

    lm = logit_means_confusion(prob, gt, pred, vmask)
    pbar_s, pbar_b = pbar_populations(prob, gt)
    ece_pos, ece_neg, _, _ = stratified_ece([(image_id, gt, prob)])

    row = {
        # identity and geography
        "run_id": run_id,
        "image_id": image_id,
        "city": city,
        "res": res,
        "lon": m.get("center_lon"),
        "lat": m.get("center_lat"),
        "source_tile_id": m.get("tile_name"),
        "pair_id": m.get("pair_id"),
        # K = 2, arm C's band
        "tp_tol": c["tp_tol"], "fp_tol": c["fp_tol"],
        "fn_tol": c["fn_tol"], "tn_tol": c["tn_tol"],
        # K = 0, strict
        "tp_str": c["tp_str"], "fp_str": c["fp_str"],
        "fn_str": c["fn_str"], "tn_str": c["tn_str"],
        # K = 5, carried so the band can be revisited without retraining
        "tp_k5": c5["tp_tol"], "fp_k5": c5["fp_tol"],
        "fn_k5": c5["fn_tol"], "tn_k5": c5["tn_tol"],
        "n_valid_px_k5": c5["n_valid_px"],
        "n_valid_px": c["n_valid_px"],
        "n_gt_fg_valid": n_gt_fg_valid(gt, vmask),
        "empty_fg_valid": empty_fg_valid(gt, vmask),
        # foreground-only IoU: iou_tol is the headline (Appendix E)
        "iou_tol": iou_from_counts(c["tp_tol"], c["fp_tol"], c["fn_tol"]),
        "iou_str": iou_from_counts(c["tp_str"], c["fp_str"], c["fn_str"]),
        "iou_k5": iou_from_counts(c5["tp_tol"], c5["fp_tol"], c5["fn_tol"]),
        # calibration and selective prediction
        "pbar_gt_shadow": pbar_s,
        "pbar_gt_bg": pbar_b,
        "logit_mean_tp": lm["logit_mean_tp"],
        "logit_mean_fp": lm["logit_mean_fp"],
        "logit_mean_fn": lm["logit_mean_fn"],
        "logit_mean_tn": lm["logit_mean_tn"],
        "aurc_fg": _aurc_trapz19_or_nan(gt, prob, "gt_shadow"),
        "aurc_bg": _aurc_trapz19_or_nan(gt, prob, "gt_bg"),
        "aurc_all": _aurc_trapz19_or_nan(gt, prob, "all_pixels"),
        "aurc_ce_fg": _aurc_ce_or_nan(gt, prob, "gt_shadow"),
        "aurc_ce_bg": _aurc_ce_or_nan(gt, prob, "gt_bg"),
        "ece_pred_pos": ece_pos,
        "ece_pred_neg": ece_neg,
        "n_pred_pos": n_pred_pos(pred, vmask),
        # the SIB-trainer plain-mean AURC, alongside the trapezoid one
        "aurc_fg_mean20": _aurc_mean20_or_nan(gt, prob, "gt_shadow"),
        "aurc_bg_mean20": _aurc_mean20_or_nan(gt, prob, "gt_bg"),
        "aurc_all_mean20": _aurc_mean20_or_nan(gt, prob, "all_pixels"),
        # 2-class mIoU, percent: reproduces Tables 30-32, NOT arm C's metric
        "iou_fg_tol_pct": fg_t, "iou_bg_tol_pct": bg_t, "miou_tol_pct": mi_t,
        "iou_fg_str_pct": fg_s, "iou_bg_str_pct": bg_s, "miou_str_pct": mi_s,
    }
    for k in DEFERRED_COLS:
        row[k] = None
    return row


def load_geo_meta(base):
    """original_filename -> metadata entry, from the benchmark mapping."""
    p = Path(base) / "data/Final_data_test/metadata/mapping.json"
    return {m["original_filename"]: m for m in json.loads(p.read_text())}


def write_dump(out_dir, rows, probs, run):
    """Write perimage.parquet, probs.npz, run.json, .DONE — in that order.

    .DONE lands last so a partial dump is never marked complete.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "perimage.parquet", index=False)
    np.savez_compressed(out / "probs.npz", **probs)
    (out / "run.json").write_text(json.dumps(run, indent=2))
    (out / ".DONE").touch()
    return df


def mark_failed(out_dir, reason):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".FAILED").write_text(str(reason))
