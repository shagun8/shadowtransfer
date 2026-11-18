#!/usr/bin/env python3
"""
Arm A / S-EO — the CONTRACT.md 3 per-patch row, with DSM-unseen pixels excluded.

Relationship to armc_dump
-------------------------
Same schema, same metric module, same band widths (K = 0 / 2 / 5). Two
differences, both forced by the dataset:

  1. S-EO carries `_unseen_pixels_mask`: pixels the DSM could not see. They are
     neither foreground nor background and leave the metric's numerator AND
     denominator (arm A's committed rule; see seo_dataset.py's docstring for
     why -- DSM coverage differs several-fold between JAX, OMA and UCSD, so
     folding them into background would make the headline partly a measure of
     DSM coverage, unequally across the three folds).
  2. Extra identity columns the sun-elevation analysis needs, which cannot be
     added later without a second inference pass.

How the unseen exclusion is done WITHOUT reimplementing a metric
---------------------------------------------------------------
`shadowtransfer.metrics` is the only metric implementation any new arm touches
(CLAUDE.md, CONTRACT.md rule zero), and the lifted `DetailedEvaluator` has no
ignore-index. It does not need one. Every count it produces, strict and
don't-care alike, is one of

    pred == 1 & target == 1      pred == 1 & target == 0
    pred == 0 & target == 0      pred == 0 & target == 1

so a pixel whose PREDICTION is neither 0 nor 1 falls out of all four. Setting
the prediction to UNSEEN_SENTINEL = 2 on unseen pixels therefore removes them
from tp/fp/tn/fn at every band width, and `n_valid_px = tp+fp+tn+fn` follows
automatically. Nothing in metrics.py is edited, copied, or re-derived.

The GT is left alone on purpose. `_compute_valid_mask` builds the +-K don't-care
band by eroding and dilating the target, so a sentinel in the TARGET would
corrupt the band's morphology around every unseen region. The band is a property
of GT geometry; it is computed on the full GT and the unseen pixels are removed
afterwards.

The calibration and selective-prediction quantities (pbar, AURC, ECE) take no
valid-mask argument, but every one of them works off `.ravel()` and boolean
indexing, so they are handed the 1-D valid-region arrays directly. That
restricts the pixel population; it does not change the estimator.

Category maps: S-EO has no six-way shadow taxonomy, so they are written EMPTY,
per CONTRACT.md 3 ("External datasets write an empty map rather than inventing
categories"). The remaining structural columns are `structure_cols: deferred`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .armc_dump import (
    DEFERRED_COLS,
    EPS_IOU,
    iou_from_counts,
    miou_from_counts,
    sha256_bytes,
    sha256_file,
)
from .metrics import (
    aurc_ce_trapz19,
    aurc_trapz19,
    logit_means_confusion,
    n_pred_pos,
    pbar_populations,
    per_image_counts,
    per_image_rc,
    per_image_rc_ce,
    stratified_ece,
)

BAND_TOL = 2
BAND_EXTRA = 5
BAND_STRICT = 0

#: A prediction value in neither class. Drops the pixel from all four
#: confusion cells of the lifted evaluator at every band width.
UNSEEN_SENTINEL = 2

#: S-EO has no shadow taxonomy. CONTRACT.md 3 says external datasets write empty.
EMPTY_CATEGORY_COLS = ("fn_px_by_cat", "tp_px_by_cat", "gt_px_by_cat")


def _rc(gt1d, p1d, mode):
    return aurc_trapz19(per_image_rc(gt1d, p1d, mode))


def _rc_ce(gt1d, p1d, mode):
    rc = per_image_rc_ce(gt1d, p1d, mode)
    return float("nan") if rc is None else aurc_ce_trapz19(rc)


def perimage_row(run_id, meta, gt, prob, pred, seen):
    """One CONTRACT.md 3 row for one S-EO patch.

    Args:
        meta : dict with image_id, aoi_id, source_image_id, patch_index, city,
               split, acquisition_date, sun_elevation, sun_azimuth, sun_dist,
               lon, lat, patch_y, patch_x
        gt   : uint8 (H, W) in {0, 1}; 1 = shadow (polarity already resolved by
               the loader: the released `_shadow.png` stores 0 = shadow)
        prob : float (H, W), UNTHRESHOLDED foreground probability
        pred : uint8 (H, W) in {0, 1}, argmax + <10 px filter
        seen : bool (H, W), True where the DSM saw the pixel. False pixels leave
               the metric entirely.
    """
    gt = np.asarray(gt).astype(np.uint8)
    prob = np.asarray(prob, dtype=np.float32)
    pred = np.asarray(pred).astype(np.uint8)
    seen = np.asarray(seen, dtype=bool)

    # The sentinel goes in the PREDICTION, never the target. See the docstring.
    pred_eval = np.where(seen, pred, np.uint8(UNSEEN_SENTINEL))

    c = per_image_counts(pred_eval, gt, boundary_tolerance=BAND_TOL)
    c5 = per_image_counts(pred_eval, gt, boundary_tolerance=BAND_EXTRA)
    band = np.asarray(c["valid_mask"], dtype=bool)
    v = band & seen                       # the region every column is computed on
    v5 = np.asarray(c5["valid_mask"], dtype=bool) & seen

    if int(v.sum()) != int(c["n_valid_px"]):
        raise AssertionError(
            f"{meta['image_id']}: n_valid_px {c['n_valid_px']} from the evaluator "
            f"disagrees with |band & seen| = {int(v.sum())}; the sentinel did not "
            f"take and the unseen exclusion is NOT live")

    fg_t, bg_t, mi_t = miou_from_counts(c["tp_tol"], c["fp_tol"], c["tn_tol"], c["fn_tol"])
    fg_s, bg_s, mi_s = miou_from_counts(c["tp_str"], c["fp_str"], c["tn_str"], c["fn_str"])
    fg_5, bg_5, mi_5 = miou_from_counts(c5["tp_tol"], c5["fp_tol"], c5["tn_tol"], c5["fn_tol"])

    lm = logit_means_confusion(prob, gt, pred, v)

    # Calibration / selective prediction on the valid region only.
    g1, p1 = gt[v].astype(np.uint8), prob[v]
    pbar_s, pbar_b = (pbar_populations(p1, g1) if g1.size else (float("nan"),) * 2)
    ece_pos, ece_neg, _, _ = stratified_ece([(meta["image_id"], g1, p1)])

    n_valid = int(c["n_valid_px"])
    n_fg = int(g1.sum())
    n_seen = int(seen.sum())
    n_px = int(seen.size)

    row = {
        # ---- identity (CONTRACT.md 3) ----
        "run_id": run_id,
        "image_id": meta["image_id"],
        "city": meta["city"],
        "res": "hr",
        "lon": meta.get("lon"), "lat": meta.get("lat"),
        "source_tile_id": meta.get("aoi_id"),
        "pair_id": None,
        # ---- arm A additions: the sun-elevation analysis depends on these ----
        "aoi_id": meta["aoi_id"],
        "source_image_id": meta["source_image_id"],
        "patch_index": meta["patch_index"],
        "patch_y": meta.get("patch_y"), "patch_x": meta.get("patch_x"),
        "split": meta["split"],
        "acquisition_date": meta["acquisition_date"],
        "sun_elevation": meta["sun_elevation"],
        "sun_azimuth": meta["sun_azimuth"],
        "sun_dist": meta["sun_dist"],
        # ---- counts on the valid region ----
        "tp_tol": c["tp_tol"], "fp_tol": c["fp_tol"],
        "fn_tol": c["fn_tol"], "tn_tol": c["tn_tol"],
        "tp_str": c["tp_str"], "fp_str": c["fp_str"],
        "fn_str": c["fn_str"], "tn_str": c["tn_str"],
        "tp_k5": c5["tp_tol"], "fp_k5": c5["fp_tol"],
        "fn_k5": c5["fn_tol"], "tn_k5": c5["tn_tol"],
        "n_valid_px": n_valid,
        "n_valid_px_k5": int(c5["n_valid_px"]),
        "n_gt_fg_valid": n_fg,
        "empty_fg_valid": bool(n_fg == 0),
        "all_shadow_flag": bool(n_valid > 0 and n_fg == n_valid),
        "n_px": n_px,
        "n_seen_px": n_seen,
        "n_unseen_px": n_px - n_seen,
        "unseen_frac": float((n_px - n_seen) / n_px),
        # ---- foreground-only IoU: iou_tol is the reporting metric ----
        "iou_tol": iou_from_counts(c["tp_tol"], c["fp_tol"], c["fn_tol"]),
        "iou_str": iou_from_counts(c["tp_str"], c["fp_str"], c["fn_str"]),
        "iou_k5": iou_from_counts(c5["tp_tol"], c5["fp_tol"], c5["fn_tol"]),
        # ---- 2-class mIoU: THIS arm's declared headline (percent) ----
        "iou_fg_tol_pct": fg_t, "iou_bg_tol_pct": bg_t, "miou_tol_pct": mi_t,
        "iou_fg_str_pct": fg_s, "iou_bg_str_pct": bg_s, "miou_str_pct": mi_s,
        "iou_fg_k5_pct": fg_5, "iou_bg_k5_pct": bg_5, "miou_k5_pct": mi_5,
        # ---- confidence and calibration ----
        "pbar_gt_shadow": pbar_s, "pbar_gt_bg": pbar_b,
        "logit_mean_tp": lm["logit_mean_tp"], "logit_mean_fp": lm["logit_mean_fp"],
        "logit_mean_fn": lm["logit_mean_fn"], "logit_mean_tn": lm["logit_mean_tn"],
        "aurc_fg": _rc(g1, p1, "gt_shadow"),
        "aurc_bg": _rc(g1, p1, "gt_bg"),
        "aurc_all": _rc(g1, p1, "all_pixels"),
        "aurc_ce_fg": _rc_ce(g1, p1, "gt_shadow"),
        "aurc_ce_bg": _rc_ce(g1, p1, "gt_bg"),
        "ece_pred_pos": ece_pos, "ece_pred_neg": ece_neg,
        "n_pred_pos": n_pred_pos(pred, v),
        "n_pred_pos_all_px": int((pred > 0).sum()),
    }
    for k in DEFERRED_COLS:
        row[k] = None
    for k in EMPTY_CATEGORY_COLS:
        row[k] = {}
    del v5
    return row


def _write_parquet(df, path):
    """DataFrame -> parquet, with the category columns forced to MAP type.

    CONTRACT.md 3 types fn_px_by_cat / tp_px_by_cat / gt_px_by_cat as
    map<str,int64> and tells external datasets to write an EMPTY map rather
    than inventing categories. An empty map is not the same statement as null:
    null means "deferred, not computed" (which is what the other structural
    columns say), an empty map means "computed, and S-EO has no taxonomy".

    pandas -> pyarrow infers a python {} as struct<> -- a struct with no child
    fields -- and the Parquet writer rejects it outright:

        ArrowNotImplementedError: Cannot write struct type 'fn_px_by_cat' with
        no child field to Parquet.

    Caught by the stage-1 smoke, job 20729183, 2026-08-01, after 2.5 minutes
    rather than after a 4-hour cell. The fix is to state the type instead of
    letting it be inferred: take pyarrow's own inferred schema and override
    exactly those three fields. Every other column keeps its inferred type, so
    this cannot silently re-type anything else.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    inferred = pa.Schema.from_pandas(df, preserve_index=False)
    map_type = pa.map_(pa.string(), pa.int64())
    fields = [pa.field(f.name, map_type) if f.name in EMPTY_CATEGORY_COLS else f
              for f in inferred]
    table = pa.Table.from_pandas(df, schema=pa.schema(fields), preserve_index=False)
    pq.write_table(table, str(path))
    return table.schema


REQUIRED_NON_NULL = ("run_id", "image_id", "aoi_id", "source_image_id", "city",
                     "split", "acquisition_date", "sun_elevation", "sun_azimuth",
                     "sun_dist", "n_valid_px", "unseen_frac")


def write_dump(out_dir, rows, probs, run):
    """perimage.parquet, probs.npz, run.json, .DONE -- in that order, .DONE last.

    Every file is written to {name}.new and renamed, so a killed writer never
    leaves a half-written artifact under the real name.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    missing = [c for c in REQUIRED_NON_NULL if c not in df.columns]
    if missing:
        raise AssertionError(f"dump is missing required columns: {missing}")
    null_cols = [c for c in REQUIRED_NON_NULL if df[c].isna().any()]
    if null_cols:
        raise AssertionError(f"required columns carry nulls: {null_cols}")

    p = out / "perimage.parquet"
    _write_parquet(df, p.with_suffix(".parquet.new"))
    p.with_suffix(".parquet.new").replace(p)

    q = out / "probs.npz"
    np.savez_compressed(str(q) + ".new.npz", **probs)
    Path(str(q) + ".new.npz").replace(q)

    r = out / "run.json"
    Path(str(r) + ".new").write_text(json.dumps(run, indent=2, default=str))
    Path(str(r) + ".new").replace(r)

    (out / ".DONE").touch()
    return df


__all__ = ["perimage_row", "write_dump", "UNSEEN_SENTINEL", "REQUIRED_NON_NULL",
           "sha256_file", "sha256_bytes", "EPS_IOU"]
