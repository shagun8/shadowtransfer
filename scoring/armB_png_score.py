#!/usr/bin/env python3
"""TASK B -- score the SAVED PREDICTION PNG masks for the 59 re-scored cells.

Table 38 was computed from data/Test_img_results/ (0/255 PNG masks), not from
data/Test_img_probs/ (float16 probability maps).  The 2026-10-26 re-score read
the probability maps; this job reads the PNG tree instead, with the SAME metric
code, so the two paths are comparable cell by cell.

Nothing here reimplements a metric.  Every count comes from
shadowtransfer.metrics.per_image_counts via shadowtransfer.armc_dump.perimage_row
-- the same two functions the re-score used -- and the two-class arithmetic is
armc_dump.miou_from_counts, which is identical (eps 1e-10) to
armc_r450_rescore.two_class.  The ONLY change is which array is passed as
`pred`:

    re-score (probability path) : pred = threshold_and_filter(prob, 10)
    this job  (PNG path)        : pred = (png > 127)

`prob` is still passed through, because the calibration and selective-prediction
columns of CONTRACT section 3 (pbar_*, logit_mean_*, aurc_*, ece_*) are defined
on the probability field and have no PNG analogue.  run.json records that split
explicitly under `column_provenance`.

Read-only with respect to every published artifact.  Writes only under
dumps/st_r450_published_png/.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("BASE_PATH") or os.environ["PROJECT_ROOT"])
SCRATCH = Path(os.environ.get("ST_SCRATCH_ROOT", str(BASE)))
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

RECOMPUTE = BASE / "dumps/st_r450_published_recompute"
FINAL_CMP = BASE / "data/final_comparison_output/final_comparison.json"
OUT_ROOT = SCRATCH / "dumps/st_r450_published_png"

ARCH_LBL = {"mamnet": "MAMNet", "oglanet": "OGLANet", "dinov3": "DINOv3"}
METHOD_LBL = {"vanilla": "LOCO Vanilla", "fda": "LOCO FDA",
              "segdesic": "LOCO SegDesic", "iim": "LOCO IIM",
              "isw": "LOCO ISW", "mrfp_plus": "LOCO MRFP+",
              "fada": "LOCO FADA", "sibc4": "C4_clean"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_dir(p):
    """A pred_dir may exist only on /scratch (partial copies on /projects).

    Returns (path, which_root).  Lexical only -- no resolve(), per CLAUDE.md.
    """
    p = str(p)
    cands = [p]
    if p.startswith(str(BASE)):
        cands.append(p.replace(str(BASE),
                               str(SCRATCH), 1))
    for c in cands:
        if os.path.isdir(c) and glob.glob(os.path.join(c, "*.png")):
            return c, ("projects" if c.startswith("/projects") else "scratch")
    return None, None


def build_inventory():
    """The 59 cells: a recompute run.json with a non-null table38_printed."""
    fc = json.load(open(FINAL_CMP))
    cells = []
    for d in sorted(os.listdir(RECOMPUTE)):
        rj = RECOMPUTE / d / "run.json"
        if not rj.is_file():
            continue
        j = json.load(open(rj))
        if j.get("table38_printed") is None:
            continue
        lbl = ("Upper Bound" if j["mode"] == "upper"
               else METHOD_LBL[j["method"]])
        e = (fc["results"].get(ARCH_LBL[j["arch"]], {})
             .get(j["holdout"], {}).get(lbl))
        if e is None:
            raise SystemExit(f"FATAL no final_comparison entry for {j['run_id']}")
        cells.append({
            "run_id": j["run_id"], "arch": j["arch"], "method": j["method"],
            "mode": j["mode"], "city": j["holdout"], "res": j["res"],
            "fc_label": lbl, "fc_mean": e["mean"], "fc_n": e["n"],
            "pred_dir_declared": e["pred_dir"],
            "probs_dir": str(BASE / j["source_probs_dir"]),
            "mask_dir": str(BASE / j["source_mask_dir"]),
            "legacy_run_dir": j["legacy_run_dir"],
            "table38_printed": j["table38_printed"],
            "recompute_miou_k2": j["miou_k2_twoclass"],
            "recompute_fg_iou_k2": j["headline_fg_iou_tol_pct"],
            "recompute_run_json_sha": sha256_file(rj),
        })
    return cells


def score_cell(c):
    """One cell.  Returns a summary dict; writes the dump as a side effect."""
    import cv2
    import pandas as pd
    from shadowtransfer import armc_dump
    from shadowtransfer.metrics import threshold_and_filter

    t0 = time.time()
    out_dir = OUT_ROOT / c["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    done, failed = out_dir / ".DONE", out_dir / ".FAILED"
    for m in (done, failed):
        if m.exists():
            m.unlink()

    pred_dir, pred_root = resolve_dir(c["pred_dir_declared"])
    if pred_dir is None:
        failed.write_text("pred_dir unresolvable on either root\n")
        return {**c, "error": "pred_dir unresolvable"}

    geo = armc_dump.load_geo_meta(BASE)
    stems = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(pred_dir, "*.png")))

    rows = []
    # PNG vs probability-map recipes, per image
    d_raw_n, d_flt_n = 0, 0           # images differing
    d_raw_px, d_flt_px = [], []       # per-image differing-pixel counts
    png_pos, raw_pos, flt_pos = [], [], []
    shape_mismatch = []

    for stem in stems:
        png = cv2.imread(os.path.join(pred_dir, stem + ".png"),
                         cv2.IMREAD_GRAYSCALE)
        raw_gt = cv2.imread(os.path.join(c["mask_dir"], stem + ".png"),
                            cv2.IMREAD_GRAYSCALE)
        pp = os.path.join(c["probs_dir"], stem + ".npy")
        if png is None or raw_gt is None or not os.path.isfile(pp):
            failed.write_text(f"missing input for {stem}\n")
            return {**c, "error": f"missing input for {stem}"}
        prob = np.load(pp).astype(np.float32)
        if not (png.shape == raw_gt.shape == prob.shape):
            shape_mismatch.append(
                {"stem": stem, "png": list(png.shape),
                 "gt": list(raw_gt.shape), "prob": list(prob.shape)})
            failed.write_text(f"shape mismatch {stem}\n")
            return {**c, "error": f"shape mismatch {stem}",
                    "shape_mismatch": shape_mismatch}

        pred = (png > 127).astype(np.uint8)
        gt = (raw_gt > 127).astype(np.uint8)
        rows.append(armc_dump.perimage_row(
            c["run_id"], stem, c["city"], c["res"], gt, prob, pred,
            geo.get(stem + ".png")))

        p_raw = (prob > 0.5).astype(np.uint8)
        p_flt = threshold_and_filter(prob, min_pixels=10).astype(np.uint8)
        n_raw = int((pred != p_raw).sum())
        n_flt = int((pred != p_flt).sum())
        d_raw_px.append(n_raw); d_flt_px.append(n_flt)
        d_raw_n += int(n_raw > 0); d_flt_n += int(n_flt > 0)
        png_pos.append(int(pred.sum())); raw_pos.append(int(p_raw.sum()))
        flt_pos.append(int(p_flt.sum()))

    df = pd.DataFrame(rows)
    pq = out_dir / "perimage.parquet"
    df.to_parquet(pq, index=False)

    fg_iou_k2 = float(df["iou_tol"].mean()) * 100.0
    miou_k2 = float(df["miou_tol_pct"].mean())
    miou_k0 = float(df["miou_str_pct"].mean())
    miou_k5 = float(
        (df["tp_k5"] / (df["tp_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100
         + df["tn_k5"] / (df["tn_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100)
        .mean() / 2.0)
    fg_iou_k5 = float(df["iou_k5"].mean()) * 100.0
    fg_iou_k0 = float(df["iou_str"].mean()) * 100.0

    # 2026-10-28: the 9 midres standard cells have neither a Table 38 value nor
    # a final_comparison entry (that file is the high-resolution family only),
    # so these deltas are optional. No-op for the 59, all of which have both.
    _t38 = c.get("table38_printed")
    _fcm = c.get("fc_mean")
    _rec = c.get("recompute_miou_k2")
    d_t38 = None if _t38 is None else miou_k2 - _t38
    d_fc = None if _fcm is None else miou_k2 - _fcm
    ar = np.array(d_raw_px); af = np.array(d_flt_px)
    summary = {
        **c, "pred_dir_used": pred_dir, "pred_root": pred_root,
        "n_images": int(len(df)),
        "png_miou_k2_twoclass": miou_k2, "png_miou_k0_twoclass": miou_k0,
        "png_miou_k5_twoclass": miou_k5,
        "png_fg_iou_k2": fg_iou_k2, "png_fg_iou_k0": fg_iou_k0,
        "png_fg_iou_k5": fg_iou_k5,
        "delta_vs_table38": d_t38,
        "table38_gate_pass": (None if d_t38 is None else abs(d_t38) <= 0.01),
        "delta_vs_final_comparison": d_fc,
        "final_comparison_gate_pass": (None if d_fc is None else abs(d_fc) <= 0.01),
        "delta_png_minus_recompute_miou_k2": (None if _rec is None
                                              else miou_k2 - _rec),
        "png_vs_prob": {
            "n_images": int(len(stems)),
            "vs_raw_threshold": {
                "recipe": "p > 0.5, no size filter",
                "n_images_differing": d_raw_n,
                "n_images_identical": int(len(stems)) - d_raw_n,
                "px_diff_total": int(ar.sum()), "px_diff_max": int(ar.max()),
                "px_diff_mean": float(ar.mean()),
                "px_diff_median": float(np.median(ar))},
            "vs_threshold_and_filter": {
                "recipe": "p > 0.5 then filter_small_predictions(min_pixels=10)",
                "n_images_differing": d_flt_n,
                "n_images_identical": int(len(stems)) - d_flt_n,
                "px_diff_total": int(af.sum()), "px_diff_max": int(af.max()),
                "px_diff_mean": float(af.mean()),
                "px_diff_median": float(np.median(af))},
            "positive_pixels": {
                "png_total": int(sum(png_pos)),
                "raw_threshold_total": int(sum(raw_pos)),
                "filtered_total": int(sum(flt_pos))},
        },
        "wall_clock_s": round(time.time() - t0, 2),
    }

    rj = {
        "run_id": c["run_id"], "dataset": "st", "arm": "C",
        "arch": c["arch"], "method": c["method"], "mode": c["mode"],
        "rung": 450, "holdout": c["city"], "res": c["res"],
        "seed": 0, "seed_source": "nominal; the legacy config records no seed",
        "n_test": int(len(df)),
        "source": "SAVED PREDICTION PNG MASKS (the tree Table 38 scored)",
        "pred_dir": pred_dir, "pred_root": pred_root,
        "pred_dir_declared_by": str(FINAL_CMP),
        "pred_dir_declared": c["pred_dir_declared"],
        "prediction_recipe": "(png > 127); the stored 0/255 argmax mask, "
                             "scored as-is with no thresholding or filtering",
        "source_mask_dir": c["mask_dir"],
        "source_probs_dir": c["probs_dir"],
        "column_provenance": {
            "counts_and_iou": "from the PNG mask ((png > 127)) via "
                              "metrics.per_image_counts at K=0/2/5",
            "pbar_*/logit_mean_*/aurc_*/ece_*/n_pred_pos":
                "from the stored float16 probability maps -- these CONTRACT "
                "section 3 columns are defined on the probability field and "
                "have no PNG analogue; they are IDENTICAL in construction to "
                "the probability-path re-score and are NOT evidence about the "
                "PNG tree",
        },
        "metric_module_path": "python/shadowtransfer/metrics.py",
        "metric_module_sha": sha256_file(BASE / "python/shadowtransfer/metrics.py"),
        "dump_module_sha": sha256_file(BASE / "python/shadowtransfer/armc_dump.py"),
        "score_script_sha": sha256_file(BASE / "python/armB_png_score.py"),
        "headline_metric": "two-class boundary-tolerant mIoU at K=2, "
                           "per-image mean (Table 38's quantity)",
        "headline_miou_k2_twoclass": miou_k2,
        "miou_k0_twoclass": miou_k0, "miou_k5_twoclass": miou_k5,
        "fg_iou_k2": fg_iou_k2, "fg_iou_k0": fg_iou_k0, "fg_iou_k5": fg_iou_k5,
        "band_width_headline": 2, "band_widths_written": [0, 2, 5],
        "table38_printed": _t38, "table38_png_score": miou_k2,
        "table38_delta": d_t38, "table38_gate_tol": 0.01,
        "table38_gate_pass": (None if d_t38 is None else abs(d_t38) <= 0.01),
        "final_comparison_mean": _fcm,
        "final_comparison_delta": d_fc,
        "final_comparison_gate_pass": (None if d_fc is None
                                       else abs(d_fc) <= 0.01),
        "final_comparison_label": c["fc_label"],
        "recompute_counterpart": {
            "dump": f"dumps/st_r450_published_recompute/{c['run_id']}",
            "run_json_sha256": c["recompute_run_json_sha"],
            "miou_k2_twoclass": _rec,
            "fg_iou_k2": c.get("recompute_fg_iou_k2"),
            "delta_png_minus_recompute_miou_k2": (None if _rec is None
                                                  else miou_k2 - _rec)},
        "png_vs_prob": summary["png_vs_prob"],
        "legacy_run_dir": c["legacy_run_dir"],
        "structure_cols": "deferred",
        "compute": "CPU-only pass over stored PNG masks, GT masks and stored "
                   "probability maps; no checkpoint loaded, no inference run",
        "git_hash": None,
        "git_hash_reason": "project root is not a git repository (CLAUDE.md)",
        "parent_run_id": None,
        "target_access": "source_only",
        "target_access_detail": "scoring pass over stored artifacts; no "
                                "held-out-city data enters any fit",
        "wall_clock_s": summary["wall_clock_s"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cluster": "delta",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "run.json").write_text(json.dumps(rj, indent=2))
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    return summary


def _worker(c):
    try:
        return score_cell(c)
    except Exception:
        return {**c, "error": "EXCEPTION", "traceback": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on run_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cells = build_inventory()
    if a.only:
        cells = [c for c in cells if a.only in c["run_id"]]
    if a.limit:
        cells = cells[:a.limit]
    print(f"cells: {len(cells)}   workers: {a.workers}   out_root: {OUT_ROOT}")
    if a.dry_run:
        for c in cells:
            pd_, root = resolve_dir(c["pred_dir_declared"])
            t38 = c.get("table38_printed"); fcm = c.get("fc_mean")
            print(f"  {c['run_id']:46s} "
                  f"t38={'  n/a ' if t38 is None else f'{t38:7.2f}'} "
                  f"fc={'       n/a      ' if fcm is None else f'{fcm:18.12f}'} "
                  f"root={root} {pd_}")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            res = pool.map(_worker, cells)
    else:
        res = [_worker(c) for c in cells]

    ok = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    print(f"\nscored {len(ok)}   failed {len(bad)}   "
          f"wall {time.time() - t0:.1f}s")
    for r in bad:
        print(f"  FAIL {r['run_id']}: {r['error']}")
        if "traceback" in r:
            print(r["traceback"])

    hdr = (f"{'run_id':46s} {'t38':>7s} {'final_cmp':>11s} {'png':>11s} "
           f"{'d_t38':>8s} {'d_fc':>9s} {'gate':>5s}")
    print("\n" + hdr)
    for r in sorted(ok, key=lambda x: x["run_id"]):
        t38 = r.get("table38_printed"); fcm = r.get("fc_mean")
        dt = r.get("delta_vs_table38"); df = r.get("delta_vs_final_comparison")
        gp = r.get("table38_gate_pass")
        print(f"{r['run_id']:46s} "
              f"{'    n/a' if t38 is None else f'{t38:7.2f}'} "
              f"{'        n/a' if fcm is None else f'{fcm:11.5f}'} "
              f"{r['png_miou_k2_twoclass']:11.5f} "
              f"{'     n/a' if dt is None else f'{dt:+8.4f}'} "
              f"{'      n/a' if df is None else f'{df:+9.5f}'} "
              f"{('n/a' if gp is None else ('PASS' if gp else 'FAIL')):>5s}")
    nt = [r for r in ok if r.get("table38_gate_pass") is not None]
    nf = [r for r in ok if r.get("final_comparison_gate_pass") is not None]
    print(f"\ntable38 gate: {sum(1 for r in nt if r['table38_gate_pass'])}"
          f"/{len(nt)} pass ({len(ok) - len(nt)} cells have no Table 38 value)")
    print(f"final_comparison gate: "
          f"{sum(1 for r in nf if r['final_comparison_gate_pass'])}/{len(nf)} "
          f"pass ({len(ok) - len(nf)} cells have no final_comparison entry)")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
                   "n_cells": len(cells), "n_ok": len(ok), "n_failed": len(bad),
                   "out_root": str(OUT_ROOT), "cells": res},
                  open(a.out, "w"), indent=2, default=str)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
