#!/usr/bin/env python3
"""PNG-only CONTRACT section 3 row, for cells that have NO probability maps.

Operator decision 2 (2026-11-05): the SIB component-ablation cells A1-A10 have
saved prediction PNG masks but no stored probability maps -- those exist only
for the C4clean reference.  Two-class boundary-tolerant mIoU at K=2 needs only
the masks, so the ablations are scored from the PNGs and the CONTRACT section 3
probability block is written NULL with a declared deviation.

WHAT THIS IS NOT
----------------
It is not a second metric implementation.  Every count and every IoU comes from
the same canonical functions `armc_dump.perimage_row` calls:

    shadowtransfer.metrics.per_image_counts      (counts at K=0/2/5)
    shadowtransfer.metrics.n_gt_fg_valid
    shadowtransfer.metrics.empty_fg_valid
    shadowtransfer.metrics.n_pred_pos
    shadowtransfer.armc_dump.iou_from_counts     (foreground IoU)
    shadowtransfer.armc_dump.miou_from_counts    (two-class mIoU, percent)
    shadowtransfer.armc_dump.DEFERRED_COLS / BAND_TOL / BAND_EXTRA

`armc_dump.py` is NOT modified.  It has ~40 callers across every arm and was
once truncated by a killed patcher; the probability path stays byte-identical
for every other cell because no byte of it changes.

What this module is, is a second ROW ASSEMBLER, and the risk in a second
assembler is schema drift.  `selftest_against_perimage_row` is the guard: for a
cell that does have probabilities it asserts that EVERY non-probability column
is exactly equal to `armc_dump.perimage_row`'s, and that the key set and key
ORDER are identical.  Run it before trusting a dump from here.
"""
import numpy as np

from . import armc_dump
from .armc_dump import (
    BAND_EXTRA,
    BAND_TOL,
    DEFERRED_COLS,
    iou_from_counts,
    miou_from_counts,
)
from .metrics import empty_fg_valid, n_gt_fg_valid, n_pred_pos, per_image_counts

# The CONTRACT section 3 columns that are DEFINED ON THE PROBABILITY FIELD and
# therefore have no PNG analogue.  Written null by this assembler.
PROB_ONLY_COLS = [
    "pbar_gt_shadow", "pbar_gt_bg",
    "logit_mean_tp", "logit_mean_fp", "logit_mean_fn", "logit_mean_tn",
    "aurc_fg", "aurc_bg", "aurc_all",
    "aurc_ce_fg", "aurc_ce_bg",
    "ece_pred_pos", "ece_pred_neg",
    "aurc_fg_mean20", "aurc_bg_mean20", "aurc_all_mean20",
]

PROB_SOURCE = "none (PNG only)"


def perimage_row_png_only(run_id, image_id, city, res, gt, pred,
                          meta_entry=None):
    """One CONTRACT section 3 row from the PNG mask alone.

    Args:
        gt   : uint8 (H, W), {0, 1}
        pred : uint8 (H, W), {0, 1} -- the stored 0/255 mask as (png > 127)

    Key set and key order match `armc_dump.perimage_row` exactly; the
    PROB_ONLY_COLS are None.
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    m = meta_entry or {}

    c = per_image_counts(pred, gt, boundary_tolerance=BAND_TOL)
    c5 = per_image_counts(pred, gt, boundary_tolerance=BAND_EXTRA)
    vmask = c["valid_mask"]

    fg_t, bg_t, mi_t = miou_from_counts(
        c["tp_tol"], c["fp_tol"], c["tn_tol"], c["fn_tol"])
    fg_s, bg_s, mi_s = miou_from_counts(
        c["tp_str"], c["fp_str"], c["tn_str"], c["fn_str"])

    row = {
        "run_id": run_id,
        "image_id": image_id,
        "city": city,
        "res": res,
        "lon": m.get("center_lon"),
        "lat": m.get("center_lat"),
        "source_tile_id": m.get("tile_name"),
        "pair_id": m.get("pair_id"),
        "tp_tol": c["tp_tol"], "fp_tol": c["fp_tol"],
        "fn_tol": c["fn_tol"], "tn_tol": c["tn_tol"],
        "tp_str": c["tp_str"], "fp_str": c["fp_str"],
        "fn_str": c["fn_str"], "tn_str": c["tn_str"],
        "tp_k5": c5["tp_tol"], "fp_k5": c5["fp_tol"],
        "fn_k5": c5["fn_tol"], "tn_k5": c5["tn_tol"],
        "n_valid_px_k5": c5["n_valid_px"],
        "n_valid_px": c["n_valid_px"],
        "n_gt_fg_valid": n_gt_fg_valid(gt, vmask),
        "empty_fg_valid": empty_fg_valid(gt, vmask),
        "iou_tol": iou_from_counts(c["tp_tol"], c["fp_tol"], c["fn_tol"]),
        "iou_str": iou_from_counts(c["tp_str"], c["fp_str"], c["fn_str"]),
        "iou_k5": iou_from_counts(c5["tp_tol"], c5["fp_tol"], c5["fn_tol"]),
        "pbar_gt_shadow": None,
        "pbar_gt_bg": None,
        "logit_mean_tp": None,
        "logit_mean_fp": None,
        "logit_mean_fn": None,
        "logit_mean_tn": None,
        "aurc_fg": None,
        "aurc_bg": None,
        "aurc_all": None,
        "aurc_ce_fg": None,
        "aurc_ce_bg": None,
        "ece_pred_pos": None,
        "ece_pred_neg": None,
        "n_pred_pos": n_pred_pos(pred, vmask),
        "aurc_fg_mean20": None,
        "aurc_bg_mean20": None,
        "aurc_all_mean20": None,
        "iou_fg_tol_pct": fg_t, "iou_bg_tol_pct": bg_t, "miou_tol_pct": mi_t,
        "iou_fg_str_pct": fg_s, "iou_bg_str_pct": bg_s, "miou_str_pct": mi_s,
    }
    for k in DEFERRED_COLS:
        row[k] = None
    return row


def selftest_against_perimage_row(gt, prob, pred, verbose=True):
    """Prove this assembler agrees with armc_dump.perimage_row.

    Returns a dict.  Raises AssertionError on any disagreement.
    """
    ref = armc_dump.perimage_row("rid", "img", "chicago", "hr", gt, prob, pred)
    mine = perimage_row_png_only("rid", "img", "chicago", "hr", gt, pred)

    assert list(ref.keys()) == list(mine.keys()), (
        "KEY ORDER DIFFERS\n ref : %s\n mine: %s"
        % (list(ref.keys()), list(mine.keys())))

    checked, mismatched = [], []
    for k in ref:
        if k in PROB_ONLY_COLS:
            assert mine[k] is None, "%s must be None in the PNG path" % k
            continue
        a, b = ref[k], mine[k]
        same = (a is None and b is None) or (a == b) or (
            isinstance(a, float) and isinstance(b, float)
            and np.isnan(a) and np.isnan(b))
        if not same:
            mismatched.append((k, a, b))
        checked.append(k)
    assert not mismatched, "NON-PROBABILITY COLUMNS DIFFER: %s" % (mismatched,)

    out = {"n_keys": len(ref), "n_checked_equal": len(checked),
           "n_nulled": len(PROB_ONLY_COLS), "n_deferred": len(DEFERRED_COLS),
           "key_order_identical": True, "mismatched": []}
    if verbose:
        print("armJ_png_only selftest PASS: %d keys, %d non-prob columns "
              "exactly equal to armc_dump.perimage_row, %d nulled, "
              "%d deferred, key order identical"
              % (out["n_keys"], out["n_checked_equal"], out["n_nulled"],
                 out["n_deferred"]))
    return out
