#!/usr/bin/env python
"""
Arm A -- THREE DIAGNOSTICS on top of results/08_seo_full_grid.json.

Pure reanalysis of the dumps already in delta_dumps/ plus the per-epoch
validation histories carried in the run.json sidecars. No GPU, no new pull.
Writes results/09_seo_diagnostics.json.

  TASK A  are the adaptation methods failing to learn at all?
          A1 predicted-positive fraction per cell, vanilla/upper as control
          A2 degenerate predictions (zero-foreground / near-all-foreground images)
          A3 training-trajectory shape from the sidecar per-epoch val histories
          A4 margin vs collapse verdict per (arch, method)

  TASK B  confounder audit for the frozen-encoder experiment (Exp A)
          every alternative explanation this box can support or exclude, each
          marked ruled_out / not_ruled_out / not_checkable_here, with evidence,
          including the Exp-A-delta-on-parent-gap regression.

  TASK C  why is Omaha the exception?
          city dataset properties, model-side quantities, probe-side p-bar
          values and an AOI-drop sensitivity on the Omaha ratios.

NO pre-committed branch is applied. Nothing is said to replicate or validate
anything. No paper wording is recommended.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/09_seo_diagnostics.py
"""

import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

BASE = os.environ["PROJECT_ROOT"]
TREE = os.path.join(BASE, "delta_dumps")
INDEX_PATH = os.path.join(TREE, "INDEX.parquet")
SCRIPT_08 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_seo_full_grid.py")
SCRIPT_08B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08b_seo_pbar.py")
RESULTS_08 = os.path.join(BASE, "results", "08_seo_full_grid.json")
RESULTS_PATH = os.path.join(BASE, "results", "09_seo_diagnostics.json")

# the index sha results/08 and results/08's task5b both cite
EXPECTED_INDEX_SHA = "a14903907349e93b4392a16aa468ffb41a459b93ccbc82edd4af6a3650b1ae9d"

ARCHS = ["mamnet", "dinov3", "oglanet", "crossearth"]
FOLDS = ["oma", "jax", "ucsd"]
SEEDS = [0, 1, 2]
N_TEST = {"oma": 1637, "jax": 2642, "ucsd": 1870}
METHODS = ["sib", "fda", "fada", "iim", "mrfp_plus", "segdesic", "isw", "bsm", "sar"]

B = 10_000
SEED = 0

# ---- TASK A thresholds. Every one is derived from the control cells and the
# derivation is written into the results JSON; none is a free parameter.
#
# A1  a cell is OUTSIDE THE CONTROL RANGE when its pooled predicted-positive
#     fraction, normalised by the fold's GT shadow rate, falls below the
#     minimum that any of the 72 vanilla-LOCO / upper control cells takes.
#     NEAR_ZERO_RATIO is the separate, absolute "a collapsed model sits near
#     zero" test: fewer than a tenth as many predicted positives as there are
#     GT shadow pixels.
NEAR_ZERO_RATIO = 0.10
# A2  per-image degeneracy cuts.
ALLFG_CUT = 0.95          # pred_pos_frac >= this -> "predicts (nearly) all foreground"
OVERPRED_CUT = 0.50       # softer over-prediction marker
# A3  trajectory classification. DROP_OUTSIDE is rounded up from the largest
#     peak-to-final drop any control cell shows (measured, printed, asserted
#     below); DROP_LARGE and DROP_SEVERE are 3x and 6x that control maximum.
DROP_OUTSIDE = 1.0
DROP_LARGE = 2.5
DROP_SEVERE = 5.0
EARLY_PEAK_EPOCH = 10     # the operator's cut; no control cell peaks before 23
MIN_POST_EPOCHS = 5       # a "sustained" drop needs this many epochs after the peak
SUSTAIN_FRAC = 0.5        # and this share of post-peak epochs at least half-way down

# ---- TASK C: how many extreme AOIs to drop in the Omaha sensitivity
AOI_DROP_K = [1, 2, 3]

COLS = ["run_id", "image_id", "city", "aoi_id", "acquisition_date", "source_image_id",
        "sun_elevation", "sun_dist", "tp_tol", "fp_tol", "fn_tol", "tn_tol",
        "n_valid_px", "n_gt_fg_valid", "empty_fg_valid", "all_shadow_flag",
        "n_px", "n_unseen_px", "unseen_frac", "iou_tol", "miou_tol_pct",
        "n_pred_pos", "n_pred_pos_all_px", "pbar_gt_shadow", "pbar_gt_bg"]

SAR_COLS = ["pass_id", "config_label", "arch", "image_id", "step", "pred_fg_fraction",
            "true_fg_fraction", "iou_fg_tol2", "n_pixels", "tp_k2", "fp_k2", "fn_k2",
            "tn_k2", "n_valid_px_k2"]


def fail(msg):
    raise SystemExit(f"ASSERTION FAILED: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M8 = load_module("seo_full_grid_08", SCRIPT_08)
M8B = load_module("seo_pbar_08b", SCRIPT_08B)
M4 = M8B.M4
if (M8.B, M8.SEED, M8B.B, M8B.SEED, M4.B, M4.SEED) != (B, SEED, B, SEED, B, SEED):
    fail(f"B/SEED differ from 04/08/08b: {(M8.B, M8.SEED, M8B.B, M8B.SEED, M4.B, M4.SEED)}")

# every statistic below is 08's / 04's own function, imported, not reimplemented
miou2, fgiou = M8.miou2, M8.fgiou
stream_of, ci95, holm = M8.stream_of, M8.ci95, M8.holm
aoi_boot = M8.aoi_cluster_bootstrap_imagemean
aoi_flip = M8.aoi_signflip_imagemean


def r6(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: r6(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [r6(x) for x in v]
    return v


# ----------------------------------------------------------------------------
# per-epoch validation history
# ----------------------------------------------------------------------------

K_POOLED = "val_decision_metric_history_pooled_miou"
K_PLAIN = "val_decision_metric_history"


def history(meta):
    """Return (family, epochs, series, extra) or None.

    Two sidecar families carry a per-epoch validation history and they are NOT
    the same quantity:

      pooled  MAMNet / DINOv3 / OGLANet. A list of floats, one per stored epoch,
              of the POOLED (micro) 2-class tolerant mIoU on validation. It is a
              correlate of, not equal to, the per-image quantity that selected
              the checkpoint: max(series) != best_val_metric on all 273 cells
              that carry it. Epoch labels are the list index + 1.
      plain   CrossEarth. A list of dicts carrying an explicit `epoch`,
              `miou_tol2`, `iou_fg_tol2` and `pred_fg_fraction_mean`. This one
              IS the decision metric: argmax(miou_tol2) == best_epoch and
              max(miou_tol2) == best_val_metric on all 45 cells.
    """
    hp, h = meta.get(K_POOLED), meta.get(K_PLAIN)
    if hp:
        s = np.asarray([float(x) for x in hp], dtype=np.float64)
        return "pooled", np.arange(1, len(s) + 1), s, {}
    if h:
        s = np.asarray([float(r["miou_tol2"]) for r in h], dtype=np.float64)
        ep = np.asarray([int(r["epoch"]) for r in h], dtype=np.int64)
        extra = dict(pred_fg_fraction_mean=[float(r["pred_fg_fraction_mean"]) for r in h],
                     iou_fg_tol2=[float(r["iou_fg_tol2"]) for r in h])
        return "plain", ep, s, extra
    return None


def traj_shape(ep, s):
    """Peak, final-block level, drop, and the sustain fraction. The final block
    is the last max(3, 10% of stored epochs) points."""
    n = len(s)
    i = int(np.argmax(s))
    peak = float(s[i])
    tail_n = max(3, int(round(0.1 * n)))
    final = float(s[-tail_n:].mean())
    post = s[i + 1:]
    out = dict(n_epochs_stored=n, peak_epoch=int(ep[i]), peak_value=peak,
               final_block_n=tail_n, final_block_mean=final, drop_val=peak - final,
               rise_to_peak=peak - float(s[0]), n_post_peak=int(len(post)),
               last_value=float(s[-1]))
    for tag, thr in (("outside", DROP_OUTSIDE), ("large", DROP_LARGE), ("severe", DROP_SEVERE)):
        frac = float((post <= peak - thr / 2.0).mean()) if len(post) else 0.0
        out[f"sustain_frac_{tag}"] = frac
        out[f"collapse_{tag}"] = bool(len(post) >= MIN_POST_EPOCHS
                                      and (peak - final) >= thr and frac >= SUSTAIN_FRAC)
    out["early_peak"] = bool(int(ep[i]) <= EARLY_PEAK_EPOCH)
    return out


def traj_class(t, thr_tag="outside"):
    """(i) normal / (ii) early-peak-then-flat / (iii) rise-then-collapse, with
    the early-peak flag kept beside the class so a cell that is BOTH is visible."""
    if t[f"collapse_{thr_tag}"]:
        return "rise_then_collapse"
    if t["early_peak"] and t["n_post_peak"] >= EARLY_PEAK_EPOCH:
        return "early_peak_then_flat"
    if t["early_peak"]:
        return "early_peak_short_tail"
    return "normal"


# ----------------------------------------------------------------------------
# selection and per-cell load
# ----------------------------------------------------------------------------

def select(rows, idx):
    """Every standard-schema cell and every SAR dump, taken from the catalogue
    and cross-checked against INDEX.parquet. Never a glob."""
    keep = [r for r in rows if r.get("category") in ("cell", "retrain_suffix", "expa", "sar")]
    ikeys = set(zip(idx.category, idx.run_id_dir))
    for r in keep:
        if (r["category"], r["run_id_dir"]) not in ikeys:
            fail(f"{r['run_id_dir']} ({r['category']}) is in the catalogue but not in INDEX.parquet")
    n_idx = int(idx.category.isin(["cell", "retrain_suffix", "expa", "sar"]).sum())
    if n_idx != len(keep):
        fail(f"catalogue selects {len(keep)} in-scope dumps, INDEX has {n_idx}")
    return keep


def load_cell(rec, foldref, checks):
    """Standard-schema dump: validate on read, recompute the metrics from the
    raw K=2 counts, return the per-image frame with only the columns used."""
    d = os.path.join(TREE, rec["dir"])
    pq_path = os.path.join(d, "perimage.parquet")
    if not os.path.isfile(os.path.join(d, "run.json")):
        fail(f"{rec['run_id_dir']}: run.json missing next to perimage.parquet")
    if not os.path.isfile(os.path.join(d, ".DONE")):
        fail(f"{rec['run_id_dir']}: no .DONE marker")
    df = pd.read_parquet(pq_path, columns=COLS)
    fold = rec["holdout"]
    rid = rec["run_id_dir"]

    missing = [c for c in COLS if c not in df.columns]
    if missing:
        fail(f"{rid}: columns absent {missing}")
    for c, want in DTYPE_REF.items():
        got = str(df[c].dtype)
        if got != want:
            fail(f"{rid}: dtype drift {c} is {got}, reference is {want} (never coerced)")
    if len(df) != N_TEST[fold]:
        fail(f"{rid}: {len(df)} rows, n_test[{fold}] is {N_TEST[fold]}")
    if df.image_id.duplicated().any():
        fail(f"{rid}: duplicate image_id")
    df = df.sort_values("image_id").reset_index(drop=True)
    ref = foldref[fold]
    if not np.array_equal(df.image_id.to_numpy(), ref["image_id"]):
        fail(f"{rid}: test-image set/order differs from the fold reference")
    if not np.array_equal(df.aoi_id.to_numpy(), ref["aoi"]):
        fail(f"{rid}: image_id -> aoi_id map differs from the fold reference")
    if not np.array_equal(df.n_valid_px.to_numpy(), ref["n_valid_px"]):
        fail(f"{rid}: n_valid_px differs from the fold reference (GT-side column)")
    if not np.array_equal(df.n_gt_fg_valid.to_numpy(), ref["n_gt_fg"]):
        fail(f"{rid}: n_gt_fg_valid differs from the fold reference (GT-side column)")
    if df.run_id.nunique() != 1 or df.run_id.iloc[0] != rid:
        fail(f"{rid}: run_id column is {df.run_id.unique()[:3]}")
    if set(df.city.str.lower().unique()) != {fold}:
        fail(f"{rid}: city column {sorted(set(df.city))} is not the holdout")

    tp, fp, fn, tn = (df[c].to_numpy(np.float64) for c in ("tp_tol", "fp_tol", "fn_tol", "tn_tol"))
    if not np.array_equal((tp + fp + fn + tn).astype(np.int64), df.n_valid_px.to_numpy()):
        fail(f"{rid}: tp+fp+fn+tn != n_valid_px at K=2")
    df["miou2_pct"] = miou2(tp, fp, fn, tn)
    df["fg_pct"] = fgiou(tp, fp, fn)
    d2 = float(np.abs(df.miou2_pct - df.miou_tol_pct).max())
    dfg = float(np.abs(df.fg_pct - df.iou_tol * 100.0).max())
    if not (d2 < 1e-9 and dfg < 1e-9):
        fail(f"{rid}: recompute from counts != stored columns ({d2:.1e}, {dfg:.1e})")
    # predicted-positive on the K=2 valid region, from the counts
    npp = df.n_pred_pos.to_numpy(np.int64)
    if not np.array_equal(npp, (df.tp_tol + df.fp_tol).to_numpy(np.int64)):
        fail(f"{rid}: n_pred_pos != tp_tol + fp_tol")
    checks.append((rid, d2, dfg))
    return df


# ----------------------------------------------------------------------------
# per-cell descriptives -- TASK A1 / A2 inputs
# ----------------------------------------------------------------------------

def cell_descriptives(df):
    """Pooled and per-image predicted-positive statistics on the K=2 valid
    region. The 108 fully-DSM-unseen patches have n_valid_px == 0; their
    per-image fraction is undefined and they are reported separately rather
    than counted as zero-foreground predictions."""
    nvp = df.n_valid_px.to_numpy(np.float64)
    npp = df.n_pred_pos.to_numpy(np.float64)
    ngt = df.n_gt_fg_valid.to_numpy(np.float64)
    defined = nvp > 0
    frac = np.full(len(df), np.nan)
    frac[defined] = npp[defined] / nvp[defined]
    pooled = float(npp.sum() / nvp.sum())
    gt_rate = float(ngt.sum() / nvp.sum())
    zero = defined & (npp == 0)
    allfg = defined & (frac >= ALLFG_CUT)
    over = defined & (frac >= OVERPRED_CUT)
    return dict(
        n_images=int(len(df)), n_images_defined=int(defined.sum()),
        n_images_zero_valid_px=int((~defined).sum()),
        predpos_frac_pooled=pooled,
        gt_shadow_rate_pooled=gt_rate,
        predpos_over_gt_pooled=(pooled / gt_rate if gt_rate > 0 else None),
        predpos_frac_img_mean=float(np.nanmean(frac)),
        predpos_frac_img_median=float(np.nanmedian(frac)),
        predpos_frac_img_p05=float(np.nanpercentile(frac, 5)),
        predpos_frac_img_p95=float(np.nanpercentile(frac, 95)),
        predpos_frac_img_min=float(np.nanmin(frac)),
        predpos_frac_img_max=float(np.nanmax(frac)),
        n_zero_fg_pred=int(zero.sum()),
        share_zero_fg_pred=float(zero.sum() / max(1, defined.sum())),
        n_allfg_pred=int(allfg.sum()),
        share_allfg_pred=float(allfg.sum() / max(1, defined.sum())),
        n_overpred_half=int(over.sum()),
        share_overpred_half=float(over.sum() / max(1, defined.sum())),
        mean_2class=float(df.miou2_pct.mean()), mean_fgonly=float(df.fg_pct.mean()),
        median_2class=float(df.miou2_pct.median()), median_fgonly=float(df.fg_pct.median()),
    )


def sar_descriptives(rec):
    """SAR dumps are a per-(pass, step) TTA schema on the unseen-INCLUDED
    population. Reported separately and labelled not commensurable, per the
    2026-10-24 operator ruling recorded in results/08."""
    df = pd.read_parquet(os.path.join(TREE, rec["dir"], "perimage.parquet"), columns=SAR_COLS)
    passes = sorted(df.pass_id.unique())
    adapted = [p for p in passes if p != "unadapted"]
    if "unadapted" not in passes or len(adapted) != 1:
        fail(f"{rec['run_id_dir']}: expected unadapted + one SAR pass, got {passes}")
    out = dict(run_id=rec["run_id_dir"], arch=rec["arch"], fold=rec["holdout"], seed=rec["seed"],
               pass_id=adapted[0], population="unseen_INCLUDED",
               config_label=str(df.config_label.dropna().iloc[0]))
    for tag, pid in (("unadapted", "unadapted"), ("sar", adapted[0])):
        sub = df[df.pass_id == pid]
        sub = sub[sub.step == sub.step.max()]
        t, f_, n_, tn = (sub[c].to_numpy(np.float64) for c in ("tp_k2", "fp_k2", "fn_k2", "tn_k2"))
        out[f"{tag}_step"] = int(sub.step.max())
        out[f"{tag}_predpos_frac_img_mean"] = float(sub.pred_fg_fraction.mean())
        out[f"{tag}_predpos_frac_img_min"] = float(sub.pred_fg_fraction.min())
        out[f"{tag}_n_zero_fg_pred"] = int((sub.pred_fg_fraction == 0).sum())
        out[f"{tag}_mean_2class"] = float(miou2(t, f_, n_, tn).mean())
    sub = df[df.pass_id == adapted[0]]
    by_step = sub.groupby("step").pred_fg_fraction.mean()
    out["predpos_frac_by_step"] = {int(k): float(v) for k, v in by_step.items()}
    out["true_fg_fraction_img_mean"] = float(sub.true_fg_fraction.mean())
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

DTYPE_REF = {}
CELL_MIOU = {}      # (arch, method, fold, seed) -> per-image 2-class mIoU, fold-aligned
CELL_FG = {}        # same, foreground-only IoU


def main():
    tag = "DRY RUN" if DRY_RUN else "FULL PASS"
    print("=" * 110)
    print(f"ARM A -- THREE DIAGNOSTICS (tasks A, B, C) -- {tag}")
    print("=" * 110)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 700)
    pd.set_option("display.max_columns", 60)
    fmt = lambda v: f"{v:.4f}"

    # ---- provenance ------------------------------------------------------
    metrics_sha = sha256_file(M8.METRICS_PATH)
    print(f"\nmetric module : {M8.METRICS_PATH}\n                sha256 {metrics_sha}")
    if metrics_sha != "730ad4a5bc039e3151599fb86bd411e2816c0897b68f1e4c17729335ab8a6e83":
        fail("metrics.py sha differs from the one 08 / 08b / 04 recorded")
    sha_08_script = sha256_file(SCRIPT_08)
    sha_08b_script = sha256_file(SCRIPT_08B)
    index_sha = sha256_file(INDEX_PATH)
    if index_sha != EXPECTED_INDEX_SHA:
        fail(f"INDEX.parquet sha {index_sha[:12]} != the one 08 cites {EXPECTED_INDEX_SHA[:12]}")
    idx = pd.read_parquet(INDEX_PATH)
    dup = idx.dropna(subset=["run_id"]).groupby("run_id").config_sha256.nunique()
    if (dup > 1).any():
        fail(f"duplicate run_id with differing config_sha256: {list(dup[dup > 1].index)}")
    with open(RESULTS_08) as fh:
        R8 = json.load(fh)
    sha_08_results = sha256_file(RESULTS_08)
    print(f"INDEX.parquet : {len(idx)} rows, sha256 {index_sha[:16]}... (== the sha 08 cites)")
    print(f"results/08    : sha256 {sha_08_results[:16]}..., {len(R8)} top-level keys")
    print(f"08 script sha {sha_08_script[:16]}...  08b script sha {sha_08b_script[:16]}...")

    rows = M8.catalogue()
    sel = select(rows, idx)
    std = [r for r in sel if r["category"] in ("cell", "retrain_suffix", "expa")]
    sar = [r for r in sel if r["category"] == "sar"]
    print(f"\ncatalogue {len(rows)} entries; in scope {len(sel)} "
          f"({len(std)} standard-schema, {len(sar)} SAR)")

    # ---- fold reference, built from the canonical vanilla MAMNet s0 cell ---
    global DTYPE_REF
    ref_rec = next(r for r in rows if r.get("run_id_dir") == "mamnet-vanilla-loco-r3500-oma-hr-s0")
    ref_df = pd.read_parquet(os.path.join(TREE, ref_rec["dir"], "perimage.parquet"), columns=COLS)
    DTYPE_REF = {c: str(ref_df[c].dtype) for c in COLS}
    print("\nresolved schema (reference cell mamnet-vanilla-loco-r3500-oma-hr-s0):")
    print("  " + ", ".join(f"{c}:{t}" for c, t in DTYPE_REF.items()))

    foldref = {}
    for fold in FOLDS:
        rec = M8.find(rows, "mamnet", "vanilla", "loco", fold, 0)
        d = (pd.read_parquet(os.path.join(TREE, rec["dir"], "perimage.parquet"), columns=COLS)
             .sort_values("image_id").reset_index(drop=True))
        codes, uniq = pd.factorize(d.aoi_id)
        foldref[fold] = dict(image_id=d.image_id.to_numpy(), aoi=d.aoi_id.to_numpy(),
                             codes=codes, n_aoi=len(uniq),
                             n_valid_px=d.n_valid_px.to_numpy(),
                             n_gt_fg=d.n_gt_fg_valid.to_numpy(),
                             keepS=~d.pbar_gt_shadow.isna().to_numpy(),
                             keepB=~d.pbar_gt_bg.isna().to_numpy(),
                             frame=d)
    print("\nfold reference:")
    for fold in FOLDS:
        f = foldref[fold]
        print(f"  {fold:5s} {len(f['image_id']):5d} patches / {f['n_aoi']:3d} AOIs; "
              f"fully DSM-unseen {int((f['n_valid_px'] == 0).sum()):3d}; "
              f"p-bar_S defined {int(f['keepS'].sum()):5d}; p-bar_B defined {int(f['keepB'].sum()):5d}")

    # ---- streaming load: descriptives + trajectories + the arrays we keep --
    recomp, desc, traj, levels, pbar = [], [], [], {}, {}
    hist_missing = []
    for rec in sorted(std, key=lambda r: (r["category"], r["arch"], r["method"], r["holdout"], r["seed"])):
        df = load_cell(rec, foldref, recomp)
        rid, arch, meth = rec["run_id_dir"], rec["arch"], rec["method"]
        fold, seed, cat = rec["holdout"], rec["seed"], rec["category"]
        meta = rec["meta"]
        base = dict(run_id=rid, category=cat, arch=arch, method=meth,
                    condition=rec["condition"], fold=fold, seed=seed,
                    target_access=meta.get("target_access"),
                    epochs_max=meta.get("epochs_max"), epochs_run=meta.get("epochs_run"),
                    best_epoch=meta.get("best_epoch"), stopped_on=meta.get("stopped_on"),
                    best_val_metric=meta.get("best_val_metric"),
                    printed_best_train_loss=meta.get("printed_best_train_loss"),
                    resume_source=meta.get("resume_source"),
                    is_control=bool(meth in ("vanilla", "none") and cat == "cell"))
        base.update(cell_descriptives(df))
        desc.append(base)

        h = history(meta)
        if h is None:
            hist_missing.append(dict(run_id=rid, category=cat, arch=arch, method=meth,
                                     fold=fold, seed=seed, stopped_on=meta.get("stopped_on"),
                                     epochs_run=meta.get("epochs_run")))
        else:
            fam, ep, s, extra = h
            t = traj_shape(ep, s)
            t.update(run_id=rid, category=cat, arch=arch, method=meth, fold=fold, seed=seed,
                     history_family=fam, history_key=(K_POOLED if fam == "pooled" else K_PLAIN),
                     epochs_run=meta.get("epochs_run"), best_epoch=meta.get("best_epoch"),
                     stopped_on=meta.get("stopped_on"), is_control=base["is_control"],
                     history_complete=bool(meta.get("epochs_run") == len(s)),
                     argmax_epoch_eq_best_epoch=bool(t["peak_epoch"] == meta.get("best_epoch")),
                     history_max_eq_best_val=bool(meta.get("best_val_metric") is not None
                                                  and abs(float(meta["best_val_metric"]) - t["peak_value"]) < 1e-4))
            if extra:
                pf = np.asarray(extra["pred_fg_fraction_mean"])
                t.update(val_predpos_first=float(pf[0]), val_predpos_at_peak=float(pf[int(np.argmax(s))]),
                         val_predpos_last=float(pf[-1]), val_predpos_min=float(pf.min()))
            traj.append(t)

        # arrays we keep. 381 cells x <=2642 float64 is a few MB, so keep them all.
        if cat == "cell":
            CELL_MIOU[(arch, meth, fold, seed)] = df.miou2_pct.to_numpy(np.float64)
            CELL_FG[(arch, meth, fold, seed)] = df.fg_pct.to_numpy(np.float64)
        if (meth in ("vanilla", "none") and cat == "cell") or cat == "expa":
            levels[(arch, meth, fold, seed, cat)] = dict(
                miou2=df.miou2_pct.to_numpy(np.float64), fg=df.fg_pct.to_numpy(np.float64))
        if meth in ("vanilla", "none") and cat == "cell":
            pbar[(arch, fold, seed, "loco" if meth == "vanilla" else "upper")] = dict(
                S=df.pbar_gt_shadow.to_numpy(np.float64), Bg=df.pbar_gt_bg.to_numpy(np.float64))

    desc = pd.DataFrame(desc)
    traj = pd.DataFrame(traj)
    print(f"\nloaded {len(desc)} standard-schema dumps; metric recompute vs stored columns: "
          f"max |2class| {max(c[1] for c in recomp):.2e}, max |fg-only| {max(c[2] for c in recomp):.2e}")
    print(f"per-epoch validation history present on {len(traj)} of {len(desc)}; absent on "
          f"{len(hist_missing)}")
    hm = pd.DataFrame(hist_missing)
    if len(hm):
        print(hm.groupby(["category", "arch", "method"]).size().to_string())

    sar_desc = pd.DataFrame([sar_descriptives(r) for r in sar]) if sar else pd.DataFrame()
    print(f"\nSAR dumps summarised separately: {len(sar_desc)}")

    out = dict(_desc=desc, _traj=traj, _traj_all=traj.copy(), _hist_missing=hm, _sar=sar_desc, _levels=levels,
               _pbar=pbar, _foldref=foldref, _rows=rows, _idx=idx, _R8=R8,
               _prov=dict(metrics_sha=metrics_sha, index_sha=index_sha,
                          sha_08_script=sha_08_script, sha_08b_script=sha_08b_script,
                          sha_08_results=sha_08_results, n_catalogue=len(rows),
                          n_in_scope=len(sel), n_std=len(std), n_sar=len(sar),
                          dtype_ref=DTYPE_REF))
    return out


# ============================================================================
# TASK A -- margin or failure to train?
# ============================================================================

def task_a(st):
    desc, traj, hm = st["_desc"], st["_traj"], st["_hist_missing"]
    levels, foldref, R8 = st["_levels"], st["_foldref"], st["_R8"]
    print("\n" + "=" * 110)
    print("TASK A -- are the adaptation methods failing to learn at all?")
    print("=" * 110)
    fmt = lambda v: f"{v:.4f}"

    cells = desc[desc.category == "cell"].copy()
    ctrl = cells[cells.is_control]
    meth = cells[~cells.is_control].copy()
    e200 = desc[desc.category == "retrain_suffix"].copy()
    traj = traj[traj.category == "cell"].copy()
    if len(ctrl) != 72:
        fail(f"expected 72 control cells (36 vanilla-LOCO + 36 upper), got {len(ctrl)}")
    if len(meth) != 243:
        fail(f"expected 243 non-SAR method cells, got {len(meth)}")
    if len(e200) != 3:
        fail(f"expected 3 D-A2-1 e200 retrains, got {len(e200)}")

    # ---------------- A1 -- collapse at test time ------------------------
    print("\n--- A1  predicted-positive fraction, pooled over patches on the K=2 valid region ---")
    c72 = ctrl
    ctrl_range = dict(
        n=int(len(c72)),
        predpos_frac_min=float(c72.predpos_frac_pooled.min()),
        predpos_frac_max=float(c72.predpos_frac_pooled.max()),
        predpos_frac_mean=float(c72.predpos_frac_pooled.mean()),
        ratio_min=float(c72.predpos_over_gt_pooled.min()),
        ratio_max=float(c72.predpos_over_gt_pooled.max()),
        ratio_min_cell=str(c72.loc[c72.predpos_over_gt_pooled.idxmin(), "run_id"]),
    )
    THR = ctrl_range["ratio_min"]
    print(f"  control cells (36 vanilla-LOCO + 36 none-upper): pooled predicted-positive fraction "
          f"{ctrl_range['predpos_frac_min']:.4f} to {ctrl_range['predpos_frac_max']:.4f}")
    print(f"  normalised by the fold GT shadow rate: {THR:.4f} to {ctrl_range['ratio_max']:.4f} "
          f"(minimum at {ctrl_range['ratio_min_cell']})")
    print(f"  THRESHOLD CHOSEN: a method cell is OUTSIDE THE CONTROL RANGE when its normalised ratio "
          f"< {THR:.4f}, the lowest value any of the 72 control cells takes.")
    print(f"  separate absolute test: NEAR-ZERO when the ratio < {NEAR_ZERO_RATIO} "
          f"(fewer than a tenth as many predicted positives as GT shadow pixels).")
    print(ctrl.groupby(["fold", "method"]).predpos_frac_pooled.agg(["min", "mean", "max"]).to_string(float_format=fmt))

    meth["below_control"] = meth.predpos_over_gt_pooled < THR
    meth["near_zero"] = meth.predpos_over_gt_pooled < NEAR_ZERO_RATIO
    print("\n  by (arch, method), pooled predicted-positive fraction and the normalised ratio:")
    a1 = (meth.groupby(["arch", "method"])
          .agg(n=("run_id", "size"),
               pp_min=("predpos_frac_pooled", "min"), pp_med=("predpos_frac_pooled", "median"),
               pp_max=("predpos_frac_pooled", "max"),
               ratio_min=("predpos_over_gt_pooled", "min"), ratio_med=("predpos_over_gt_pooled", "median"),
               n_below_control=("below_control", "sum"), n_near_zero=("near_zero", "sum"))
          .reset_index())
    print(a1.to_string(index=False, float_format=fmt))
    below = meth[meth.below_control].sort_values("predpos_over_gt_pooled")
    print(f"\n  cells below the control range: {len(below)} of {len(meth)}")
    if len(below):
        print(below[["run_id", "predpos_frac_pooled", "gt_shadow_rate_pooled",
                     "predpos_over_gt_pooled", "mean_2class"]].to_string(index=False, float_format=fmt))
    print(f"  cells near zero (ratio < {NEAR_ZERO_RATIO}): {int(meth.near_zero.sum())}")

    # ---------------- A2 -- degenerate predictions ------------------------
    print("\n--- A2  degenerate per-image predictions (vanilla + upper are the control) ---")
    print(f"  zero-foreground share counts patches with n_pred_pos == 0 among the patches that have a "
          f"defined fraction (n_valid_px > 0). The 108 fully-DSM-unseen patches "
          f"(29 OMA / 0 JAX / 79 UCSD) have n_valid_px == 0 and are excluded from the denominator.")
    print(f"  near-all-foreground share counts pred_pos_frac >= {ALLFG_CUT}; the softer marker is "
          f">= {OVERPRED_CUT}.")
    a2_ctrl = dict(
        share_zero_min=float(c72.share_zero_fg_pred.min()), share_zero_max=float(c72.share_zero_fg_pred.max()),
        n_zero_min=int(c72.n_zero_fg_pred.min()), n_zero_max=int(c72.n_zero_fg_pred.max()),
        share_allfg_max=float(c72.share_allfg_pred.max()),
        share_overpred_half_max=float(c72.share_overpred_half.max()))
    print(f"  control range: zero-foreground share {a2_ctrl['share_zero_min']:.4f} to "
          f"{a2_ctrl['share_zero_max']:.4f} ({a2_ctrl['n_zero_min']} to {a2_ctrl['n_zero_max']} patches); "
          f"near-all-foreground share max {a2_ctrl['share_allfg_max']:.4f}; "
          f">= {OVERPRED_CUT} share max {a2_ctrl['share_overpred_half_max']:.4f}")
    a2 = (meth.groupby(["arch", "method"])
          .agg(n=("run_id", "size"),
               zero_share_med=("share_zero_fg_pred", "median"), zero_share_max=("share_zero_fg_pred", "max"),
               zero_n_max=("n_zero_fg_pred", "max"),
               allfg_share_max=("share_allfg_pred", "max"),
               over_share_max=("share_overpred_half", "max"))
          .reset_index())
    print(a2.to_string(index=False, float_format=fmt))
    worst_zero = meth.nlargest(8, "share_zero_fg_pred")[
        ["run_id", "n_zero_fg_pred", "share_zero_fg_pred", "n_allfg_pred", "share_allfg_pred"]]
    print("\n  highest zero-foreground shares among method cells:")
    print(worst_zero.to_string(index=False, float_format=fmt))

    # ---------------- A3 -- trajectory shape -----------------------------
    print("\n--- A3  training-trajectory shape from the sidecar per-epoch validation histories ---")
    tctrl = traj[traj.is_control]
    ctrl_drop_max = float(tctrl["drop_val"].max())
    ctrl_peak_min = int(tctrl.peak_epoch.min())
    print(f"  two history families, NOT the same quantity:")
    for fam, g in traj.groupby("history_family"):
        print(f"    {fam:7s} n={len(g):3d}  key={g.history_key.iloc[0]}  "
              f"argmax epoch == best_epoch on {int(g.argmax_epoch_eq_best_epoch.sum())}/{len(g)}; "
              f"max(series) == best_val_metric on {int(g.history_max_eq_best_val.sum())}/{len(g)}")
    print(f"  control cells (n={len(tctrl)}): peak-to-final drop {float(tctrl['drop_val'].min()):.4f} to "
          f"{ctrl_drop_max:.4f} val mIoU points; earliest peak epoch {ctrl_peak_min}")
    if not (DROP_OUTSIDE > ctrl_drop_max):
        fail(f"DROP_OUTSIDE {DROP_OUTSIDE} is not above the control maximum {ctrl_drop_max}")
    if not (EARLY_PEAK_EPOCH < ctrl_peak_min):
        fail(f"EARLY_PEAK_EPOCH {EARLY_PEAK_EPOCH} is not below the earliest control peak {ctrl_peak_min}")
    print(f"  THRESHOLDS CHOSEN: 'outside the control range' = drop >= {DROP_OUTSIDE} "
          f"(rounded up from the control maximum {ctrl_drop_max:.4f}); 'large' = {DROP_LARGE} "
          f"(~3x); 'severe' = {DROP_SEVERE} (~6x). A drop is sustained only when at least "
          f"{SUSTAIN_FRAC:.0%} of the >= {MIN_POST_EPOCHS} post-peak epochs sit at least half-way down.")
    print(f"  early peak = peak epoch <= {EARLY_PEAK_EPOCH}; no control cell peaks before epoch {ctrl_peak_min}.")

    for thr_tag in ("outside", "large", "severe"):
        traj[f"cls_{thr_tag}"] = traj.apply(lambda r: traj_class(r, thr_tag), axis=1)
    if int(traj.loc[traj.is_control, "cls_outside"].ne("normal").sum()) != 0:
        fail("a control cell classifies as something other than normal at the outside threshold")

    print("\n  trajectory class by (arch, method), at the 'outside the control range' threshold:")
    ct = pd.crosstab([traj.arch, traj.method], traj.cls_outside)
    print(ct.to_string())
    print("\n  counts of rise-then-collapse at the three thresholds, with the drop magnitudes:")
    a3 = (traj.groupby(["arch", "method"])
          .agg(n_hist=("run_id", "size"),
               n_normal=("cls_outside", lambda s: int((s == "normal").sum())),
               n_early_flat=("cls_outside", lambda s: int(s.str.startswith("early_peak").sum())),
               n_collapse_outside=("collapse_outside", "sum"),
               n_collapse_large=("collapse_large", "sum"),
               n_collapse_severe=("collapse_severe", "sum"),
               n_early_peak=("early_peak", "sum"),
               peak_ep_min=("peak_epoch", "min"), peak_ep_max=("peak_epoch", "max"),
               drop_med=("drop_val", "median"), drop_max=("drop_val", "max"))
          .reset_index())
    print(a3.to_string(index=False, float_format=fmt))
    print("\n  cells with a drop at or above the 'large' threshold:")
    big = traj[traj.collapse_large].sort_values("drop_val", ascending=False)
    print(big[["run_id", "history_family", "n_epochs_stored", "peak_epoch", "peak_value",
               "final_block_mean", "drop_val", "sustain_frac_large"]].to_string(index=False, float_format=fmt))

    print("\n  best_epoch / epochs_run / stopped_on by (arch, method), all cells:")
    conv = (cells.groupby(["arch", "method"])
            .agg(n=("run_id", "size"),
                 best_epoch_min=("best_epoch", "min"), best_epoch_max=("best_epoch", "max"),
                 epochs_run_min=("epochs_run", "min"), epochs_run_max=("epochs_run", "max"),
                 n_patience=("stopped_on", lambda s: int((s == "patience").sum())),
                 n_ceiling=("stopped_on", lambda s: int((s == "epoch_ceiling").sum())))
            .reset_index())
    print(conv.to_string(index=False))

    incomplete = traj[~traj.history_complete]
    print(f"\n  per-epoch history NOT available: {len(hm)} dumps "
          f"({', '.join(sorted(set(hm.category))) if len(hm) else '-'}).")
    if len(hm):
        print("    " + hm.groupby(["category", "method"]).size().to_string().replace("\n", "\n    "))
    print(f"  per-epoch history PRESENT BUT SHORTER than epochs_run: {len(incomplete)} cells "
          f"(resume links skip an epoch label; see the sidecars' resume_epoch_label_offset_note).")
    if len(incomplete):
        print(incomplete[["run_id", "history_family", "epochs_run", "n_epochs_stored",
                          "peak_epoch", "best_epoch", "drop_val", "cls_outside"]].to_string(index=False, float_format=fmt))

    # ---------------- SAR, separately -------------------------------------
    sar = st["_sar"]
    print("\n--- A3/A1 for SAR, on its own population (not commensurable; operator ruling 2026-10-24) ---")
    if len(sar):
        s_sum = (sar.groupby(["arch"])
                 .agg(n=("run_id", "size"),
                      unad_pp=("unadapted_predpos_frac_img_mean", "mean"),
                      sar_pp=("sar_predpos_frac_img_mean", "mean"),
                      sar_pp_min=("sar_predpos_frac_img_min", "min"),
                      n_zero=("sar_n_zero_fg_pred", "max"),
                      true_fg=("true_fg_fraction_img_mean", "mean"),
                      unad_2c=("unadapted_mean_2class", "mean"), sar_2c=("sar_mean_2class", "mean"))
                 .reset_index())
        print(s_sum.to_string(index=False, float_format=fmt))
        print("  SAR has no training trajectory: it adapts at test time. The per-step mean "
              "predicted-positive fraction over the 10 SAR steps is in the JSON "
              "(task_a.sar.cells[].predpos_frac_by_step).")

    # ---------------- A4 -- margin or collapse ----------------------------
    print("\n--- A4  deficit versus vanilla: margin or failure to train? ---")
    # the delta versus the matched vanilla-LOCO cell, recomputed, cross-checked against 08
    mt = pd.DataFrame(R8["task2"]["method_table"])
    mt2 = mt[(mt.metric == "2class") & (mt.block == "main")]
    rec_delta, xc = [], []
    for (arch, method, fold), _ in meth.groupby(["arch", "method", "fold"]):
        if method == "sar":
            continue
        seeds = sorted(set(meth[(meth.arch == arch) & (meth.method == method)
                                & (meth.fold == fold)].seed.tolist()))
        dl = {}
        for s in seeds:
            van = levels[(arch, "vanilla", fold, s, "cell")]["miou2"]
            k = (arch, method, fold, s)
            dl[s] = CELL_MIOU[k] - van
        dbar = np.mean([dl[s] for s in seeds], axis=0)
        obs = float(dbar.mean())
        rec_delta.append(dict(arch=arch, method=method, fold=fold, n_seeds=len(seeds),
                              delta_recomputed=obs))
        hit = mt2[(mt2.arch == arch) & (mt2.method == method) & (mt2.fold == fold)]
        if len(hit) == 1:
            xc.append(dict(arch=arch, method=method, fold=fold, mine=obs,
                           r08=float(hit["mean"].iloc[0]),
                           absdiff=abs(obs - float(hit["mean"].iloc[0]))))
    rec_delta = pd.DataFrame(rec_delta)
    xc = pd.DataFrame(xc)
    print(f"  cross-check against results/08 task2.method_table (metric 2class, block main): "
          f"{len(xc)} rows matched, max |difference| {xc.absdiff.max():.2e}")
    if xc.absdiff.max() > 1e-6:
        print(xc.sort_values("absdiff", ascending=False).head(10).to_string(index=False))
        print(rec_delta.sort_values(["arch", "method", "fold"]).to_string(index=False))
        fail(f"recomputed method delta differs from results/08 by {xc.absdiff.max():.2e}")

    verd = []
    for (arch, method), g in meth.groupby(["arch", "method"]):
        tg = traj[(traj.arch == arch) & (traj.method == method)]
        dg = rec_delta[(rec_delta.arch == arch) & (rec_delta.method == method)]
        n_cells = len(g)
        n_below = int(g.below_control.sum())
        n_near0 = int(g.near_zero.sum())
        n_coll_l = int(tg.collapse_large.sum()) if len(tg) else 0
        n_coll_s = int(tg.collapse_severe.sum()) if len(tg) else 0
        n_coll_o = int(tg.collapse_outside.sum()) if len(tg) else 0
        n_early = int(tg.early_peak.sum()) if len(tg) else 0
        # the verdict rule, stated once and applied mechanically. Two independent
        # criteria, each with zero false positives on the 72 control cells:
        #   EARLY  peak epoch <= 10        (no control cell peaks before 23)
        #   DROP   sustained drop >= 2.5   (3x the largest control drop, 0.8447)
        # and the prediction-side test, near-zero predicted positives.
        if method == "sar":
            v, why = "not_assessed_here", ("test-time adaptation: no training trajectory, and the dump "
                                           "is on a different pixel population. Its within-dump deltas "
                                           "are positive, so it is not losing to vanilla.")
        elif n_near0 > 0:
            v, why = "collapse", f"{n_near0} of {n_cells} cells predict near-zero foreground"
        elif len(tg) and (n_early == len(tg) or n_coll_l == len(tg)):
            v, why = "failure_to_train", (
                f"every cell with a history shows the signature: {n_early} of {len(tg)} peak by epoch "
                f"{EARLY_PEAK_EPOCH} and {n_coll_l} of {len(tg)} lose at least {DROP_LARGE} validation "
                f"mIoU points and never regain them. Test-time predictions are NOT degenerate "
                f"({n_near0} cells near zero, {n_below} outside the control range).")
        elif n_early > 0 or n_coll_l > 0:
            v, why = "mixed", (
                f"{n_early} of {len(tg)} cells peak by epoch {EARLY_PEAK_EPOCH} and {n_coll_l} lose at "
                f"least {DROP_LARGE} validation mIoU points; the remaining cells train normally")
        else:
            v, why = "margin", (
                f"no cell peaks by epoch {EARLY_PEAK_EPOCH}, none loses {DROP_LARGE} validation mIoU "
                f"points, none predicts near-zero foreground, and none is degenerate "
                f"({n_coll_o} of {len(tg)} cells do show a drop just outside the control range)")
        verd.append(dict(arch=arch, method=method, n_cells=n_cells,
                         median_predpos_frac=float(g.predpos_frac_pooled.median()),
                         median_predpos_over_gt=float(g.predpos_over_gt_pooled.median()),
                         n_below_control_range=n_below, n_near_zero=n_near0,
                         median_zero_fg_share=float(g.share_zero_fg_pred.median()),
                         max_zero_fg_share=float(g.share_zero_fg_pred.max()),
                         max_allfg_share=float(g.share_allfg_pred.max()),
                         n_hist=len(tg), n_normal=int((tg.cls_outside == "normal").sum()) if len(tg) else 0,
                         n_early_peak=n_early,
                         n_collapse_outside=n_coll_o, n_collapse_large=n_coll_l,
                         n_collapse_severe=n_coll_s,
                         max_drop=float(tg["drop_val"].max()) if len(tg) else None,
                         verdict_prediction_side=("near_zero_collapse" if n_near0 else
                                                  "below_control_range" if n_below else "no_collapse"),
                         verdict_trajectory_side=(
                             "no_history" if not len(tg) else
                             "early_peak_and_sustained_drop" if (n_early and n_coll_l) else
                             "early_peak" if n_early else
                             "sustained_drop" if n_coll_l else
                             "drop_just_outside_control_range" if n_coll_o else "normal"),
                         mean_delta_vs_vanilla=(float(dg.delta_recomputed.mean()) if len(dg) else None),
                         worst_delta_vs_vanilla=(float(dg.delta_recomputed.min()) if len(dg) else None),
                         verdict=v, verdict_basis=why))
    verd = pd.DataFrame(verd).sort_values(["arch", "method"])
    print("\n  (arch, method) x {median pred_pos_frac, zero-foreground share, trajectory classes, verdict}")
    print(verd[["arch", "method", "n_cells", "median_predpos_frac", "median_predpos_over_gt",
                "n_below_control_range", "n_near_zero", "median_zero_fg_share", "max_allfg_share",
                "n_hist", "n_normal", "n_early_peak", "n_collapse_outside", "n_collapse_large",
                "n_collapse_severe", "max_drop", "mean_delta_vs_vanilla", "worst_delta_vs_vanilla",
                "verdict_prediction_side", "verdict_trajectory_side",
                "verdict"]].to_string(index=False, float_format=fmt))
    print("\n  verdict counts: " + str(verd.verdict.value_counts().to_dict()))

    # the MAMNet/MRFP+ signature and whether any other pair shares it
    mr = traj[(traj.arch == "mamnet") & (traj.method == "mrfp_plus")]
    sig = traj[(traj.early_peak) | (traj.collapse_large)]
    sig_pairs = (sig.groupby(["arch", "method"])
                 .agg(n=("run_id", "size"), n_early=("early_peak", "sum"),
                      n_large=("collapse_large", "sum"), n_severe=("collapse_severe", "sum"),
                      peak_min=("peak_epoch", "min"), peak_max=("peak_epoch", "max"),
                      drop_min=("drop_val", "min"), drop_max=("drop_val", "max")).reset_index())
    print("\n  the MAMNet/MRFP+ signature (early peak and/or a large sustained drop) -- which "
          "(arch, method) pairs show it, and on how many of their cells:")
    print(sig_pairs.to_string(index=False, float_format=fmt))
    print(f"  MAMNet/MRFP+ itself: peak epochs {sorted(mr.peak_epoch.tolist())}, "
          f"drops {[round(x, 2) for x in sorted(mr['drop_val'].tolist())]}, "
          f"best_epoch from the sidecars {sorted(int(x) for x in mr.best_epoch.tolist())} "
          f"(results/08 records {R8['task2']['mamnet_mrfp_plus_best_epochs']})")

    # the three D-A2-1 e200 retrains, as a sensitivity (results/08 keeps the originals primary)
    print("\n--- A  sensitivity: the three D-A2-1 from-scratch e200 retrains, against their originals ---")
    e200v = []
    for _, r in e200.iterrows():
        orig = r.run_id.replace("-e200", "")
        o = desc[desc.run_id == orig]
        if len(o) != 1:
            fail(f"{r.run_id}: {len(o)} originals named {orig}")
        o = o.iloc[0]
        tr = st["_traj_all"][st["_traj_all"].run_id == r.run_id]
        to = st["_traj_all"][st["_traj_all"].run_id == orig]
        e200v.append(dict(
            run_id=r.run_id, original=orig,
            predpos_frac_e200=r.predpos_frac_pooled, predpos_frac_original=o.predpos_frac_pooled,
            predpos_over_gt_e200=r.predpos_over_gt_pooled,
            predpos_over_gt_original=o.predpos_over_gt_pooled,
            zero_fg_share_e200=r.share_zero_fg_pred, zero_fg_share_original=o.share_zero_fg_pred,
            epochs_run_e200=r.epochs_run, epochs_run_original=o.epochs_run,
            best_epoch_e200=r.best_epoch, best_epoch_original=o.best_epoch,
            stopped_on_e200=r.stopped_on, stopped_on_original=o.stopped_on,
            drop_e200=(float(tr["drop_val"].iloc[0]) if len(tr) else None),
            drop_original=(float(to["drop_val"].iloc[0]) if len(to) else None),
            cls_e200=(str(traj_class(tr.iloc[0], "outside")) if len(tr) else None),
            cls_original=(str(traj_class(to.iloc[0], "outside")) if len(to) else None)))
    e200v = pd.DataFrame(e200v)
    print(e200v.to_string(index=False, float_format=fmt))

    st["task_a"] = dict(
        universe=dict(
            n_method_cells=int(len(meth)), n_control_cells=int(len(c72)),
            n_sar_dumps=int(len(sar)), n_e200_retrains=int(len(e200)),
            note=("Every method cell and every vanilla-LOCO / none-upper control cell on the S-EO "
                  "grid. SAR's 36 dumps are summarised separately on their own pixel population. "
                  "The 3 D-A2-1 e200 retrains are reported but are not part of any count.")),
        a1_collapse_at_test_time=dict(
            definition=("pooled predicted-positive fraction = sum(n_pred_pos) / sum(n_valid_px) over the "
                        "fold's test patches, on the K=2 boundary-tolerant valid region with DSM-unseen "
                        "pixels excluded. n_pred_pos was asserted equal to tp_tol + fp_tol on every cell. "
                        "The normalised ratio divides it by the same cell's pooled GT shadow rate "
                        "sum(n_gt_fg_valid) / sum(n_valid_px), which is a fold property and identical "
                        "across all cells of a fold."),
            control_range=r6(ctrl_range),
            threshold_outside_control_range=r6(THR),
            threshold_derivation=("the lowest normalised ratio any of the 72 control cells takes; no "
                                  "method cell below it is inside the range vanilla and the upper bound "
                                  "occupy"),
            threshold_near_zero=NEAR_ZERO_RATIO,
            threshold_near_zero_derivation=("absolute: a collapsed model predicting (almost) all "
                                            "background has fewer than a tenth as many predicted "
                                            "positives as there are GT shadow pixels"),
            by_arch_method=r6(a1.to_dict("records")),
            control_by_fold_method=r6(ctrl.groupby(["fold", "method"]).predpos_frac_pooled
                                      .agg(["min", "mean", "max"]).reset_index().to_dict("records")),
            cells_below_control_range=r6(below[["run_id", "arch", "method", "fold", "seed",
                                                "predpos_frac_pooled", "gt_shadow_rate_pooled",
                                                "predpos_over_gt_pooled", "mean_2class"]].to_dict("records")),
            n_cells_below_control_range=int(len(below)),
            n_cells_near_zero=int(meth.near_zero.sum())),
        a2_degenerate_predictions=dict(
            definition=(f"per patch, pred_pos_frac = n_pred_pos / n_valid_px. Zero-foreground means "
                        f"n_pred_pos == 0; near-all-foreground means pred_pos_frac >= {ALLFG_CUT}; the "
                        f"softer over-prediction marker is >= {OVERPRED_CUT}. Denominator is the patches "
                        f"with n_valid_px > 0; the 108 fully-DSM-unseen patches are excluded and counted "
                        f"separately."),
            control_range=r6(a2_ctrl),
            by_arch_method=r6(a2.to_dict("records")),
            highest_zero_fg_method_cells=r6(worst_zero.to_dict("records"))),
        a3_trajectory=dict(
            source=("the run.json sidecars' own per-epoch validation history. Two families, which are "
                    "not the same quantity: " + K_POOLED + " (MAMNet/DINOv3/OGLANet, the POOLED/micro "
                    "2-class tolerant validation mIoU, a correlate of the per-image quantity that "
                    "selected the checkpoint -- max(series) != best_val_metric on all 273 of them) and "
                    + K_PLAIN + " (CrossEarth, which IS the decision metric -- argmax == best_epoch and "
                    "max == best_val_metric on all 45). best_epoch, epochs_run and stopped_on are read "
                    "from the sidecar fields, not from the history."),
            history_family_check=r6(traj.groupby("history_family").agg(
                n=("run_id", "size"),
                n_argmax_eq_best_epoch=("argmax_epoch_eq_best_epoch", "sum"),
                n_max_eq_best_val=("history_max_eq_best_val", "sum")).reset_index().to_dict("records")),
            control_drop_range=[r6(float(tctrl["drop_val"].min())), r6(ctrl_drop_max)],
            control_earliest_peak_epoch=ctrl_peak_min,
            thresholds=dict(drop_outside_control_range=DROP_OUTSIDE, drop_large=DROP_LARGE,
                            drop_severe=DROP_SEVERE, early_peak_epoch=EARLY_PEAK_EPOCH,
                            min_post_peak_epochs=MIN_POST_EPOCHS, sustain_frac=SUSTAIN_FRAC,
                            derivation=(f"drop_outside_control_range is rounded up from the largest "
                                        f"peak-to-final drop any of the {len(tctrl)} control cells shows "
                                        f"({ctrl_drop_max:.4f} val mIoU points); large and severe are "
                                        f"about 3x and 6x that. early_peak_epoch is the operator's cut "
                                        f"and sits below the earliest control peak, epoch {ctrl_peak_min}.")),
            classes=dict(normal="peak after the early-peak cut, no sustained drop",
                         early_peak_then_flat=f"peak epoch <= {EARLY_PEAK_EPOCH} with at least "
                                              f"{EARLY_PEAK_EPOCH} epochs after it and no sustained drop",
                         early_peak_short_tail=f"peak epoch <= {EARLY_PEAK_EPOCH} but fewer than "
                                               f"{EARLY_PEAK_EPOCH} epochs after it",
                         rise_then_collapse=("a peak followed by a drop at or above the threshold that is "
                                             "sustained: at least half of the >= 5 post-peak epochs sit at "
                                             "least half-way down from the peak")),
            class_counts_by_arch_method=r6(a3.to_dict("records")),
            crosstab_outside=json.loads(ct.to_json(orient="index")),
            cells_drop_at_or_above_large=r6(big[["run_id", "arch", "method", "fold", "seed",
                                                 "history_family", "n_epochs_stored", "peak_epoch",
                                                 "peak_value", "final_block_mean", "drop_val",
                                                 "sustain_frac_large", "collapse_severe"]].to_dict("records")),
            convergence_by_arch_method=r6(conv.to_dict("records")),
            n_cells_history_unavailable=int(len(hm)),
            cells_history_unavailable=r6(hm.to_dict("records")),
            history_unavailable_by_category=(hm.groupby(["category", "method"]).size().to_dict()
                                             if len(hm) else {}),
            n_cells_history_shorter_than_epochs_run=int(len(incomplete)),
            cells_history_shorter_than_epochs_run=r6(incomplete[
                ["run_id", "history_family", "epochs_run", "n_epochs_stored", "peak_epoch",
                 "best_epoch", "drop_val", "cls_outside"]].to_dict("records"))),
        sar=dict(note=("SAR adapts at test time and has no training trajectory. Its dumps keep "
                       "DSM-unseen pixels, so neither its predicted-positive fraction nor its mIoU is "
                       "commensurable with the other rows (operator ruling 2026-10-24, recorded in "
                       "results/08)."),
                 by_arch=r6(s_sum.to_dict("records")) if len(sar) else [],
                 cells=r6(sar.to_dict("records")) if len(sar) else []),
        a4_verdicts=r6(verd.to_dict("records")),
        e200_sensitivity=r6(e200v.to_dict("records")),
        e200_sensitivity_note=("the three D-A2-1 from-scratch 200-epoch-horizon retrains of OGLANet "
                               "cells whose originals stopped at the 100-epoch ceiling. results/08 keeps "
                               "the inventory-named originals primary and so does this analysis; these "
                               "rows are the substitution."),
        a4_verdict_rule=(
            "Applied mechanically on two criteria, each of which fires on 0 of the 72 control cells: "
            f"EARLY, a validation peak at or before epoch {EARLY_PEAK_EPOCH} (no control cell peaks "
            f"before epoch 23), and DROP, a sustained peak-to-final loss of at least {DROP_LARGE} "
            f"validation mIoU points (about 3x the largest control drop, 0.8447). Plus the "
            "prediction-side test, a normalised predicted-positive ratio below "
            f"{NEAR_ZERO_RATIO}. collapse = any cell predicts near-zero foreground. failure_to_train = "
            "every cell with a history fires one of the two criteria. mixed = some but not all do. "
            "margin = none does. The count at the looser 'outside the control range' drop threshold "
            f"({DROP_OUTSIDE}) is reported in the same row as n_collapse_outside and is the sensitivity; "
            "it is not the verdict, because a cell that patience stopped 20 epochs after a shallow peak "
            "declines structurally."),
        mamnet_mrfp_plus_signature=dict(
            pairs_showing_early_peak_or_large_drop=r6(sig_pairs.to_dict("records")),
            mamnet_mrfp_plus_peak_epochs=sorted(int(x) for x in mr.peak_epoch.tolist()),
            mamnet_mrfp_plus_drops=r6(sorted(float(x) for x in mr["drop_val"].tolist())),
            mamnet_mrfp_plus_best_epochs_sidecar=sorted(int(x) for x in mr.best_epoch.tolist()),
            mamnet_mrfp_plus_best_epochs_in_08=R8["task2"]["mamnet_mrfp_plus_best_epochs"]),
        crosscheck_vs_08_method_table=dict(
            n_rows_matched=int(len(xc)), max_abs_difference=r6(float(xc.absdiff.max())),
            note=("the seed-averaged per-image mean of (method - matched vanilla-LOCO) two-class mIoU, "
                  "recomputed here from the raw counts and compared with results/08 task2.method_table "
                  "mean for metric 2class, block main")),
        per_cell=r6(cells.to_dict("records")),
        per_cell_trajectory=r6(traj.drop(columns=[c for c in traj.columns if c.startswith("_")])
                               .to_dict("records")))
    return st


# ============================================================================
# shared paired statistic -- 08's functions, 08's streams
# ============================================================================

def paired(delta_by_seed, fold, label, foldref):
    codes, n_aoi = foldref[fold]["codes"], foldref[fold]["n_aoi"]
    seeds = sorted(delta_by_seed)
    dbar = np.mean([delta_by_seed[s] for s in seeds], axis=0)
    boot = aoi_boot(dbar, codes, n_aoi, stream_of("boot|" + label))
    obs, p = aoi_flip(dbar, codes, n_aoi, stream_of("flip|" + label))
    lo, hi = ci95(boot)
    per_seed = {int(s): float(delta_by_seed[s].mean()) for s in seeds}
    ps = np.array(list(per_seed.values()))
    return dict(mean=obs, stderr=float(boot.std(ddof=1)), ci_lo=lo, ci_hi=hi, p_raw=p,
                n_seeds=len(seeds), per_seed=per_seed,
                seed_sd=float(ps.std(ddof=1)) if len(ps) > 1 else None,
                n_images=int(len(dbar)), n_aoi=int(n_aoi), dbar=dbar)


# ============================================================================
# TASK B -- confounder audit for the frozen-encoder experiment
# ============================================================================

def task_b(st):
    desc, traj, levels = st["_desc"], st["_traj"], st["_levels"]
    foldref, rows, R8 = st["_foldref"], st["_rows"], st["_R8"]
    print("\n" + "=" * 110)
    print("TASK B -- confounder audit for Exp A (frozen encoder, decoder retrained on held-out-city labels)")
    print("=" * 110)
    fmt = lambda v: f"{v:.4f}"
    EA = ["mamnet", "dinov3", "oglanet"]          # no published CrossEarth Exp A
    audit = []

    expa = desc[desc.category == "expa"].copy()
    if len(expa) != 27:
        fail(f"expected 27 Exp A cells, got {len(expa)}")
    meta_by_rid = {r["run_id_dir"]: r["meta"] for r in rows if "meta" in r}

    # ---------------- B1  metric denominator ------------------------------
    print("\n--- B1  metric denominator: is 08's comparison the matched one, and does it survive? ---")
    b1_cells, b1_af = [], []
    for arch in EA:
        for fold in FOLDS:
            dfg, d2c, dwrong = {}, {}, {}
            for s in SEEDS:
                ea = levels[(arch, "expa", fold, s, "expa")]
                pa = levels[(arch, "vanilla", fold, s, "cell")]
                rid_ea = f"{arch}-expa-loco-r3500-{fold}-hr-s{s}"
                rid_pa = f"{arch}-vanilla-loco-r3500-{fold}-hr-s{s}"
                m_ea, m_pa = meta_by_rid[rid_ea], meta_by_rid[rid_pa]
                # the sidecar fields, and our recomputation against them
                ea_head = float(m_ea["reporting_metric_value_pct"])
                pa_floor = float(m_pa["sanity_floor_metric_value_pct"])
                pa_head = float(m_pa["reporting_metric_value_pct"])
                if m_ea["reporting_metric"] != "fg-only IoU K=2 (CONTRACT.md App E)":
                    fail(f"{rid_ea}: reporting_metric is {m_ea['reporting_metric']}")
                if m_pa["sanity_floor_metric"] != "fg_only_tol_k2_unseen_excluded":
                    fail(f"{rid_pa}: sanity_floor_metric is {m_pa['sanity_floor_metric']}")
                dfg[s] = ea["fg"] - pa["fg"]
                d2c[s] = ea["miou2"] - pa["miou2"]
                dwrong[s] = ea["fg"] - pa["miou2"]
                b1_cells.append(dict(
                    arch=arch, fold=fold, seed=s, run_id=rid_ea, parent_run_id=rid_pa,
                    expa_fg_recomputed=float(ea["fg"].mean()), expa_fg_sidecar=ea_head,
                    expa_fg_recompute_minus_sidecar=float(ea["fg"].mean()) - ea_head,
                    parent_fg_recomputed=float(pa["fg"].mean()), parent_fg_sidecar_sanity_floor=pa_floor,
                    parent_fg_recompute_minus_sidecar=float(pa["fg"].mean()) - pa_floor,
                    parent_2class_recomputed=float(pa["miou2"].mean()),
                    parent_2class_sidecar_headline=pa_head,
                    expa_2class_recomputed=float(ea["miou2"].mean()),
                    delta_fg_matched=float(dfg[s].mean()),
                    delta_2class_matched=float(d2c[s].mean()),
                    delta_WRONG_expaFG_minus_parent2class=float(dwrong[s].mean())))
            for tag, dd in (("fg_matched", dfg), ("2class_matched", d2c),
                            ("WRONG_expaFG_minus_parent2class", dwrong)):
                r = paired(dd, fold, f"b1|{tag}|{arch}|{fold}", foldref)
                r.pop("dbar")
                r.update(arch=arch, fold=fold, comparison=tag)
                b1_af.append(r)
    b1_cells = pd.DataFrame(b1_cells)
    b1_af = pd.DataFrame(b1_af)
    for tag in ("fg_matched", "2class_matched", "WRONG_expaFG_minus_parent2class"):
        sub = b1_af[b1_af.comparison == tag].copy()
        sub["p_holm"] = holm(sub.p_raw.tolist())
        b1_af.loc[sub.index, "p_holm"] = sub.p_holm
    max_rec = float(max(b1_cells.expa_fg_recompute_minus_sidecar.abs().max(),
                        b1_cells.parent_fg_recompute_minus_sidecar.abs().max()))
    print(f"  recomputed from raw counts vs the sidecar fields: max |difference| {max_rec:.2e}")
    if max_rec > 1e-3:
        fail(f"Exp A / parent level recompute differs from the sidecar by {max_rec:.2e}")
    print("  (arch x fold) seed-stratified deltas under the three differencings, Holm within each:")
    print(b1_af[["comparison", "arch", "fold", "mean", "ci_lo", "ci_hi", "p_raw", "p_holm"]]
          .to_string(index=False, float_format=fmt))
    sep = {}
    for tag in ("fg_matched", "2class_matched", "WRONG_expaFG_minus_parent2class"):
        sub = b1_af[b1_af.comparison == tag]
        cellsub = b1_cells[["arch", "fold", "seed",
                            {"fg_matched": "delta_fg_matched",
                             "2class_matched": "delta_2class_matched",
                             "WRONG_expaFG_minus_parent2class":
                                 "delta_WRONG_expaFG_minus_parent2class"}[tag]]].copy()
        cellsub.columns = ["arch", "fold", "seed", "d"]
        per_arch = {}
        for arch in EA:
            g = cellsub[cellsub.arch == arch]
            a = sub[sub.arch == arch]
            per_arch[arch] = dict(
                n_cells=int(len(g)), n_pos_cells=int((g.d > 0).sum()), n_neg_cells=int((g.d < 0).sum()),
                cell_min=float(g.d.min()), cell_max=float(g.d.max()),
                n_archfold=int(len(a)), n_archfold_holm_sig=int((a.p_holm < 0.05).sum()),
                n_archfold_pos=int((a["mean"] > 0).sum()), n_archfold_neg=int((a["mean"] < 0).sum()))
        sep[tag] = dict(per_arch=per_arch,
                        perfect_sign_separation=bool(
                            all(per_arch[a]["n_pos_cells"] == per_arch[a]["n_cells"] for a in ("mamnet", "dinov3"))
                            and per_arch["oglanet"]["n_neg_cells"] == per_arch["oglanet"]["n_cells"]))
    for tag, v in sep.items():
        print(f"  {tag}: perfect architecture sign separation = {v['perfect_sign_separation']}; "
              + "; ".join(f"{a} {v['per_arch'][a]['n_pos_cells']}+/{v['per_arch'][a]['n_neg_cells']}- "
                          f"[{v['per_arch'][a]['cell_min']:.3f}, {v['per_arch'][a]['cell_max']:.3f}]"
                          for a in EA))
    # cross-check against 08's task 3
    t3c = pd.DataFrame(R8["task3"]["cells"])
    mg = b1_cells.merge(t3c[["arch", "fold", "seed", "delta", "expa_fg", "parent_fg",
                             "WRONG_headline_minus_headline"]], on=["arch", "fold", "seed"])
    xc3 = dict(n=int(len(mg)),
               max_abs_delta_diff=float((mg.delta_fg_matched - mg.delta).abs().max()),
               max_abs_expa_fg_diff=float((mg.expa_fg_recomputed - mg.expa_fg).abs().max()),
               max_abs_parent_fg_diff=float((mg.parent_fg_recomputed - mg.parent_fg).abs().max()),
               max_abs_wrong_diff=float((mg.delta_WRONG_expaFG_minus_parent2class
                                         - mg.WRONG_headline_minus_headline).abs().max()))
    print(f"  cross-check vs results/08 task3.cells ({xc3['n']} cells): max |delta diff| "
          f"{xc3['max_abs_delta_diff']:.2e}, max |Exp A level diff| {xc3['max_abs_expa_fg_diff']:.2e}, "
          f"max |parent level diff| {xc3['max_abs_parent_fg_diff']:.2e}, "
          f"max |wrong-differencing diff| {xc3['max_abs_wrong_diff']:.2e}")
    if max(xc3["max_abs_delta_diff"], xc3["max_abs_expa_fg_diff"], xc3["max_abs_parent_fg_diff"]) > 1e-4:
        fail("Exp A recomputation does not reproduce results/08 task 3")
    audit.append(dict(
        id="B1", alternative="The separation is an artifact of differencing a foreground-only metric "
                             "against a two-class one.",
        status="ruled_out",
        evidence=(f"08's comparison is the matched one: Exp A's own reporting_metric is fg-only IoU at "
                  f"K=2 and the parent's sanity_floor_metric_value_pct is the same quantity, and both "
                  f"were recomputed here from the raw K=2 counts to within {max_rec:.1e} of the sidecar "
                  f"fields. The separation is perfect on the matched fg-only comparison "
                  f"({sep['fg_matched']['per_arch']['mamnet']['n_pos_cells']}/9 and "
                  f"{sep['fg_matched']['per_arch']['dinov3']['n_pos_cells']}/9 positive, "
                  f"{sep['fg_matched']['per_arch']['oglanet']['n_neg_cells']}/9 negative) and also on a "
                  f"matched TWO-CLASS comparison computed here for the first time "
                  f"({sep['2class_matched']['per_arch']['mamnet']['n_pos_cells']}/9, "
                  f"{sep['2class_matched']['per_arch']['dinov3']['n_pos_cells']}/9 positive, "
                  f"{sep['2class_matched']['per_arch']['oglanet']['n_neg_cells']}/9 negative), so it is "
                  f"not a property of the denominator. Differencing the two headline fields flips the "
                  f"sign on all 18 MAMNet and DINOv3 cells, as 08 records."),
        artifacts=["task_b.b1_metric_denominator"]))

    # ---------------- B2  training instability ----------------------------
    print("\n--- B2  do the OGLANet Exp A cells show the task-A collapse signature? ---")
    a1 = st["task_a"]["a1_collapse_at_test_time"]
    thr = a1["threshold_outside_control_range"]
    expa["below_control"] = expa.predpos_over_gt_pooled < thr
    expa["near_zero"] = expa.predpos_over_gt_pooled < NEAR_ZERO_RATIO
    b2 = (expa.groupby(["arch"])
          .agg(n=("run_id", "size"),
               pp_min=("predpos_frac_pooled", "min"), pp_med=("predpos_frac_pooled", "median"),
               pp_max=("predpos_frac_pooled", "max"),
               ratio_min=("predpos_over_gt_pooled", "min"), ratio_med=("predpos_over_gt_pooled", "median"),
               n_below_control=("below_control", "sum"), n_near_zero=("near_zero", "sum"),
               zero_fg_share_max=("share_zero_fg_pred", "max"),
               allfg_share_max=("share_allfg_pred", "max"),
               fg_mean=("mean_fgonly", "mean"))
          .reset_index())
    print(f"  control range for the normalised ratio is [{thr:.4f}, {a1['control_range']['ratio_max']:.4f}]")
    print(b2.to_string(index=False, float_format=fmt))
    n_hist_expa = int(len(traj[traj.category == "expa"]))
    a1_share_zero_max = float(st["task_a"]["a2_degenerate_predictions"]["control_range"]["share_zero_max"])
    print(f"  Exp A cells carrying a per-epoch validation history: {n_hist_expa} of 27 "
          f"(the published Exp A has no validation split at all; the sidecars say so).")
    audit.append(dict(
        id="B2", alternative="OGLANet's Exp A cells are training unstably (collapsing), and the negative "
                             "sign is that, not a feature-sufficiency difference.",
        status="ruled_out",
        evidence=(f"The collapse signature is absent. No Exp A cell of any architecture predicts "
                  f"near-zero foreground: the normalised predicted-positive ratio is "
                  f"{float(expa.predpos_over_gt_pooled.min()):.4f} to "
                  f"{float(expa.predpos_over_gt_pooled.max()):.4f}, so even the lowest cell predicts "
                  f"{float(expa.predpos_over_gt_pooled.min()) * 100:.0f}% as many positives as there are "
                  f"GT shadow pixels, against a near-zero cut of {NEAR_ZERO_RATIO}. The largest "
                  f"zero-foreground image share on an OGLANet Exp A cell is "
                  f"{float(expa[expa.arch == 'oglanet'].share_zero_fg_pred.max()):.4f}, BELOW the control "
                  f"cells' own maximum of {a1_share_zero_max:.4f}, and no Exp A cell of any architecture "
                  f"has a single near-all-foreground patch. What IS true is a mild under-prediction: "
                  f"{int(expa[expa.arch == 'oglanet'].below_control.sum())} of 9 OGLANet Exp A cells sit "
                  f"below the control range's lower edge {thr:.4f} (minimum "
                  f"{float(expa[expa.arch == 'oglanet'].predpos_over_gt_pooled.min()):.4f}), against "
                  f"{int(expa[expa.arch == 'mamnet'].below_control.sum())} of 9 on MAMNet and "
                  f"{int(expa[expa.arch == 'dinov3'].below_control.sum())} of 9 on DINOv3; OGLANet's "
                  f"ordinary method cells do the same thing (7 of 9 on MRFP+, 6 of 9 on SegDeSiC), so it "
                  f"is an architecture tendency rather than an Exp A instability. The trajectory half of "
                  f"the task-A signature cannot be checked on Exp A at all: the published Exp A has no "
                  f"validation split and {27 - n_hist_expa} of 27 sidecars carry no per-epoch history."),
        artifacts=["task_b.b2_training_instability"]))

    # ---------------- B3  convergence -------------------------------------
    print("\n--- B3  are the three architectures stopped at comparably converged points? ---")
    b3_cells = []
    for _, r in expa.iterrows():
        m = meta_by_rid[r.run_id]
        pm = meta_by_rid[f"{r.arch}-vanilla-loco-r3500-{r.fold}-hr-s{r.seed}"]
        eds = m.get("encoder_decoder_split", {})
        b3_cells.append(dict(
            run_id=r.run_id, arch=r.arch, fold=r.fold, seed=r.seed,
            expa_epochs_max=m.get("epochs_max"), expa_epochs_run=m.get("epochs_run"),
            expa_best_epoch=m.get("best_epoch"), expa_stopped_on=m.get("stopped_on"),
            expa_decision_metric=m.get("decision_metric"),
            expa_printed_best_train_loss=m.get("printed_best_train_loss"),
            expa_lr=m.get("lr"), expa_batch_size=m.get("batch_size"), expa_n_train=m.get("n_train"),
            expa_scheduler=(m.get("published_recipe") or {}).get("scheduler"),
            n_encoder_params=eds.get("n_encoder_params"), n_decoder_params=eds.get("n_decoder_params"),
            n_encoder_elements=eds.get("n_encoder_elements"), n_decoder_elements=eds.get("n_decoder_elements"),
            n_decoder_modules_reinitialised=m.get("n_decoder_modules_reinitialised"),
            parent_epochs_max=pm.get("epochs_max"), parent_epochs_run=pm.get("epochs_run"),
            parent_best_epoch=pm.get("best_epoch"), parent_stopped_on=pm.get("stopped_on"),
            parent_early_stop_patience=pm.get("early_stop_patience"),
            parent_best_val_metric=pm.get("best_val_metric")))
    b3_cells = pd.DataFrame(b3_cells)
    b3 = (b3_cells.groupby("arch")
          .agg(n=("run_id", "size"),
               expa_epochs=("expa_epochs_run", lambda s: f"{s.min()}-{s.max()}"),
               expa_stopped=("expa_stopped_on", lambda s: ",".join(sorted(set(s)))),
               expa_best_epoch_nonnull=("expa_best_epoch", lambda s: int(s.notna().sum())),
               train_loss_min=("expa_printed_best_train_loss", "min"),
               train_loss_mean=("expa_printed_best_train_loss", "mean"),
               train_loss_max=("expa_printed_best_train_loss", "max"),
               enc_el=("n_encoder_elements", "first"), dec_el=("n_decoder_elements", "first"),
               dec_reinit=("n_decoder_modules_reinitialised", "first"),
               parent_epochs=("parent_epochs_run", lambda s: f"{s.min()}-{s.max()}"),
               parent_best_epoch=("parent_best_epoch", lambda s: f"{s.min():.0f}-{s.max():.0f}"),
               parent_ceiling=("parent_stopped_on", lambda s: int((s == "epoch_ceiling").sum())))
          .reset_index())
    print(b3.to_string(index=False, float_format=fmt))
    print("  Exp A has a fixed epoch count and no model selection on every cell. The only convergence "
          "witness in the sidecars is printed_best_train_loss, which is min(epoch mean training loss) "
          "and is never used to pick weights. It is the same CrossEntropyLoss(ignore_index=255) on the "
          "same 875 patches and the same labels for all three architectures, so its level is comparable "
          "across them; it is not a validation quantity and does not say a cell converged.")
    tl = b3_cells.groupby("arch").expa_printed_best_train_loss.mean()
    expa_pooled = {a: float(pd.DataFrame(R8["task3"]["separation"]).query("arch == @a").pooled_mean.iloc[0])
                   for a in EA}
    order_loss = list(tl.sort_values().index)
    order_delta = sorted(EA, key=lambda a: -expa_pooled[a])
    print(f"  training-loss order (lowest first): {order_loss}")
    print(f"  Exp A pooled delta order (largest first): {order_delta}")
    audit.append(dict(
        id="B3", alternative="The three architectures are not stopped at comparably converged points, so "
                             "the sign tracks how far each got in 30 epochs.",
        status="not_ruled_out",
        evidence=(f"Every Exp A cell runs the published fixed 30 epochs (stopped_on epoch_ceiling, "
                  f"best_epoch null, no validation split, no model selection) at lr 1e-3, batch 4, on "
                  f"875 patches, with CosineAnnealingLR(T_max=30), so the epoch budget and the LR horizon "
                  f"are identical across architectures. What is NOT equal is where they get to: the mean "
                  f"printed best training loss is "
                  + ", ".join(f"{a} {float(tl[a]):.4f}" for a in EA) +
                  f", and that ordering ({order_loss}) is the reverse of the Exp A delta ordering "
                  f"({order_delta}) on all three architectures. One reading is that the frozen OGLANet "
                  f"encoder's features are harder to fit; the other is that the OGLANet decoder is not "
                  f"converged at epoch 30. Nothing on this box separates them: there is no validation "
                  f"history, no per-epoch loss, and no longer-horizon Exp A run. The parents are not "
                  f"matched on this either -- they early-stop on patience 20 with model selection, "
                  f"at best_epoch " + ", ".join(
                      f"{a} {b3[b3.arch == a].parent_best_epoch.iloc[0]}" for a in EA) + "."),
        artifacts=["task_b.b3_convergence"]))

    # ---------------- B4  the parents' gap --------------------------------
    print("\n--- B4  is the Exp A sign tracking how much gap there was to recover? ---")
    import statsmodels.formula.api as smf
    reg_rows = []
    for arch in EA:
        for fold in FOLDS:
            for s in SEEDS:
                ea = levels[(arch, "expa", fold, s, "expa")]
                lo = levels[(arch, "vanilla", fold, s, "cell")]
                up = levels[(arch, "none", fold, s, "cell")]
                reg_rows.append(dict(
                    arch=arch, fold=fold, seed=s,
                    expa_delta_fg=float((ea["fg"] - lo["fg"]).mean()),
                    expa_delta_2class=float((ea["miou2"] - lo["miou2"]).mean()),
                    parent_gap_fg=float((up["fg"] - lo["fg"]).mean()),
                    parent_gap_2class=float((up["miou2"] - lo["miou2"]).mean()),
                    expa_minus_upper_fg=float((ea["fg"] - up["fg"]).mean()),
                    recovery_share_fg=float((ea["fg"] - lo["fg"]).mean()
                                            / (up["fg"] - lo["fg"]).mean())))
    reg = pd.DataFrame(reg_rows)
    print(reg.groupby(["arch", "fold"])[["expa_delta_fg", "parent_gap_fg", "parent_gap_2class",
                                         "expa_minus_upper_fg", "recovery_share_fg"]]
          .mean().to_string(float_format=fmt))
    b4 = {}
    for yname, xname in (("expa_delta_fg", "parent_gap_fg"),
                         ("expa_delta_fg", "parent_gap_2class"),
                         ("expa_delta_2class", "parent_gap_2class")):
        key = f"{yname}~{xname}"
        m = smf.ols(f"{yname} ~ {xname}", data=reg).fit()
        m_arch = smf.ols(f"{yname} ~ {xname} + C(arch)", data=reg).fit()
        m_cl = smf.ols(f"{yname} ~ {xname}", data=reg).fit(
            cov_type="cluster", cov_kwds={"groups": reg["arch"]})
        within = {}
        for arch in EA:
            g = reg[reg.arch == arch]
            mw = smf.ols(f"{yname} ~ {xname}", data=g).fit()
            sp = float(pd.Series(g[yname].values).corr(pd.Series(g[xname].values), method="spearman"))
            within[arch] = dict(n=int(len(g)), slope=float(mw.params[xname]),
                                se=float(mw.bse[xname]), p=float(mw.pvalues[xname]),
                                r2=float(mw.rsquared), spearman=sp,
                                x_min=float(g[xname].min()), x_max=float(g[xname].max()),
                                y_min=float(g[yname].min()), y_max=float(g[yname].max()))
        # permutation test: permute y WITHIN architecture, so only the gap-vs-delta
        # relation that is not explained by architecture can produce a slope
        rng = np.random.default_rng([SEED, stream_of("b4perm|" + key)])
        obs = float(m_arch.params[xname])
        null = np.empty(B)
        y = reg[yname].to_numpy(float)
        x = reg[xname].to_numpy(float)
        acode = pd.factorize(reg.arch)[0]
        X = np.column_stack([np.ones(len(reg)), x] + [(acode == k).astype(float)
                                                      for k in range(1, acode.max() + 1)])
        for b in range(B):
            yp = y.copy()
            for k in range(acode.max() + 1):
                m_ = acode == k
                yp[m_] = rng.permutation(y[m_])
            null[b] = np.linalg.lstsq(X, yp, rcond=None)[0][1]
        p_perm = (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)
        b4[key] = dict(
            pooled=dict(n=int(len(reg)), slope=float(m.params[xname]), se_ols=float(m.bse[xname]),
                        p_ols=float(m.pvalues[xname]), intercept=float(m.params["Intercept"]),
                        r2=float(m.rsquared), resid_sd=float(np.std(m.resid, ddof=2)),
                        se_cluster_arch=float(m_cl.bse[xname]), p_cluster_arch=float(m_cl.pvalues[xname]),
                        n_clusters=3),
            with_arch_fixed_effects=dict(slope=float(m_arch.params[xname]), se=float(m_arch.bse[xname]),
                                         p=float(m_arch.pvalues[xname]), r2=float(m_arch.rsquared),
                                         r2_gap_only=float(m.rsquared),
                                         r2_arch_only=float(smf.ols(f"{yname} ~ C(arch)",
                                                                    data=reg).fit().rsquared)),
            permutation_within_arch=dict(statistic=obs, p=p_perm, B=B,
                                         note=("y permuted within architecture, so a significant slope "
                                               "would mean the parent's gap predicts the Exp A delta "
                                               "beyond what architecture already explains")),
            within_architecture=within)
        pw = b4[key]
        print(f"\n  {key}")
        print(f"    pooled over 27 cells: slope {pw['pooled']['slope']:+.4f} "
              f"(OLS SE {pw['pooled']['se_ols']:.4f}, p {pw['pooled']['p_ols']:.2e}; "
              f"arch-clustered SE {pw['pooled']['se_cluster_arch']:.4f}, "
              f"p {pw['pooled']['p_cluster_arch']:.3f}, 3 clusters), "
              f"intercept {pw['pooled']['intercept']:+.4f}, R2 {pw['pooled']['r2']:.4f}")
        print(f"    R2 from the gap alone {pw['with_arch_fixed_effects']['r2_gap_only']:.4f}; from "
              f"architecture alone {pw['with_arch_fixed_effects']['r2_arch_only']:.4f}; "
              f"gap slope with architecture fixed effects {pw['with_arch_fixed_effects']['slope']:+.4f} "
              f"(p {pw['with_arch_fixed_effects']['p']:.3f}); within-architecture permutation "
              f"p {pw['permutation_within_arch']['p']:.4f}")
        for arch in EA:
            w = within[arch]
            print(f"    within {arch:9s} n={w['n']} slope {w['slope']:+.4f} (p {w['p']:.3f}) "
                  f"R2 {w['r2']:.3f} Spearman {w['spearman']:+.3f}  "
                  f"gap in [{w['x_min']:.3f}, {w['x_max']:.3f}]  delta in [{w['y_min']:.3f}, {w['y_max']:.3f}]")

    # does any pair of architectures overlap in parent gap but differ in Exp A sign?
    ov = []
    for xname in ("parent_gap_fg", "parent_gap_2class"):
        for a1_, a2_ in (("dinov3", "oglanet"), ("mamnet", "oglanet"), ("mamnet", "dinov3")):
            g1, g2 = reg[reg.arch == a1_], reg[reg.arch == a2_]
            lo_ov = max(g1[xname].min(), g2[xname].min())
            hi_ov = min(g1[xname].max(), g2[xname].max())
            n1 = int(((g1[xname] >= lo_ov) & (g1[xname] <= hi_ov)).sum()) if hi_ov >= lo_ov else 0
            n2 = int(((g2[xname] >= lo_ov) & (g2[xname] <= hi_ov)).sum()) if hi_ov >= lo_ov else 0
            ov.append(dict(x=xname, arch_a=a1_, arch_b=a2_, overlap=bool(hi_ov >= lo_ov),
                           overlap_lo=float(lo_ov) if hi_ov >= lo_ov else None,
                           overlap_hi=float(hi_ov) if hi_ov >= lo_ov else None,
                           n_a_in_overlap=n1, n_b_in_overlap=n2,
                           a_delta_in_overlap_mean=(float(g1.loc[(g1[xname] >= lo_ov) & (g1[xname] <= hi_ov),
                                                                 "expa_delta_fg"].mean()) if n1 else None),
                           b_delta_in_overlap_mean=(float(g2.loc[(g2[xname] >= lo_ov) & (g2[xname] <= hi_ov),
                                                                 "expa_delta_fg"].mean()) if n2 else None)))
    ov = pd.DataFrame(ov)
    print("\n  parent-gap overlap between architecture pairs, and the Exp A delta inside the overlap:")
    print(ov.to_string(index=False, float_format=fmt))
    key = "expa_delta_fg~parent_gap_fg"
    survives = bool(b4[key]["permutation_within_arch"]["p"] < 0.05)
    ovk = ov[(ov.x == "parent_gap_fg") & (ov.arch_a == "dinov3") & (ov.arch_b == "oglanet")].iloc[0]
    audit.append(dict(
        id="B4", alternative="The Exp A sign tracks how much gap there was to recover: OGLANet's parents "
                             "have the smallest gaps in the grid, MAMNet's the largest.",
        status="not_ruled_out",
        evidence=(f"It survives for MAGNITUDE and fails for SIGN, and both halves are measured. "
                  f"Magnitude: pooled over the 27 cells the parent fg-only gap predicts the Exp A "
                  f"fg-only delta at slope {b4[key]['pooled']['slope']:+.4f} per point of gap, R2 "
                  f"{b4[key]['pooled']['r2']:.4f}, OLS p {b4[key]['pooled']['p_ols']:.1e}, and it "
                  f"survives architecture fixed effects at "
                  f"{b4[key]['with_arch_fixed_effects']['slope']:+.4f} (p "
                  f"{b4[key]['with_arch_fixed_effects']['p']:.3f}; within-architecture permutation p "
                  f"{b4[key]['permutation_within_arch']['p']:.4f}, "
                  f"{'significant' if survives else 'not significant'}). It is not uniform across "
                  f"architectures: within DINOv3 the gap explains R2 "
                  f"{b4[key]['within_architecture']['dinov3']['r2']:.3f} (slope "
                  f"{b4[key]['within_architecture']['dinov3']['slope']:+.3f}, p "
                  f"{b4[key]['within_architecture']['dinov3']['p']:.3f}), within MAMNet "
                  f"{b4[key]['within_architecture']['mamnet']['r2']:.3f}, and within OGLANet "
                  f"{b4[key]['within_architecture']['oglanet']['r2']:.3f} (slope "
                  f"{b4[key]['within_architecture']['oglanet']['slope']:+.3f}, p "
                  f"{b4[key]['within_architecture']['oglanet']['p']:.3f}), so inside the architecture "
                  f"whose sign is negative the gap predicts nothing. Sign: the grid contains a region "
                  f"where DINOv3's and OGLANet's parent fg-only gaps overlap, "
                  f"[{float(ovk.overlap_lo):.3f}, {float(ovk.overlap_hi):.3f}], holding "
                  f"{int(ovk.n_a_in_overlap)} DINOv3 cells and {int(ovk.n_b_in_overlap)} OGLANet cells. "
                  f"Inside it the DINOv3 cells average {float(ovk.a_delta_in_overlap_mean):+.4f} and the "
                  f"OGLANet cells {float(ovk.b_delta_in_overlap_mean):+.4f}, so the sign still separates "
                  f"at matched parent gap and the gap-recovery account does not by itself produce the "
                  f"separation. MAMNet's gaps do not overlap either other architecture's, so the "
                  f"matched-gap comparison exists only for the DINOv3/OGLANet pair. Architecture alone "
                  f"explains R2 {b4[key]['with_arch_fixed_effects']['r2_arch_only']:.4f} against the "
                  f"gap's {b4[key]['pooled']['r2']:.4f}. Status is not_ruled_out because the gap is a "
                  f"real predictor of magnitude that no test here removes, and because parent gap and "
                  f"architecture are collinear by construction over three architectures."),
        artifacts=["task_b.b4_gap_recovery"]))

    # ---------------- B5  unseen-pixel handling ---------------------------
    print("\n--- B5  unseen-pixel handling, prediction recipe, loss and test population ---")
    b5 = []
    for _, r in expa.iterrows():
        m = meta_by_rid[r.run_id]
        pm = meta_by_rid[f"{r.arch}-vanilla-loco-r3500-{r.fold}-hr-s{r.seed}"]
        uh, puh = m.get("unseen_handling", {}), pm.get("unseen_handling", {})
        nb = m.get("nan_batch_tally", {})
        b5.append(dict(run_id=r.run_id, arch=r.arch, fold=r.fold, seed=r.seed,
                       expa_metric_rule=uh.get("metric"), parent_metric_rule=puh.get("metric"),
                       expa_loss_rule=uh.get("loss"), parent_loss_rule=puh.get("loss"),
                       metric_rule_same_family=bool(
                           "EXCLUDED" in str(uh.get("metric")) and "EXCLUDED" in str(puh.get("metric"))),
                       loss_rule_same_family=bool(
                           "EXCLUDED" in str(uh.get("loss")) and "EXCLUDED" in str(puh.get("loss"))),
                       prediction_recipe=m.get("prediction_recipe"),
                       n_fully_ignored_batches=nb.get("n_fully_ignored_batches"),
                       n_nan_loss_batches=nb.get("n_nan_loss_batches"),
                       freeze_encoder_nonzero_grad=(m.get("freeze_reached_p5") or {}).get(
                           "n_encoder_with_nonzero_grad"),
                       leakage=(m.get("leakage_check") or {}).get("leakage",
                                                                  (m.get("leakage_check") or {}).get("guard")),
                       train_manifest=(m.get("budget_manifest_train")),
                       test_manifest=(m.get("budget_manifest_test"))))
    b5 = pd.DataFrame(b5)
    n_zero_contrib = 0
    for arch in EA:
        for fold in FOLDS:
            nvp = foldref[fold]["n_valid_px"]
            for s in SEEDS:
                ea = levels[(arch, "expa", fold, s, "expa")]
                lo = levels[(arch, "vanilla", fold, s, "cell")]
                z = nvp == 0
                if z.sum() and not np.allclose((ea["fg"] - lo["fg"])[z], 0.0):
                    fail(f"{arch}/{fold}/s{s}: a fully-DSM-unseen patch contributes a non-zero delta")
                n_zero_contrib += int(z.sum())
    print(f"  metric rule identical in family (EXCLUDED) on {int(b5.metric_rule_same_family.sum())}/27; "
          f"loss rule on {int(b5.loss_rule_same_family.sum())}/27")
    print(f"  prediction recipe on every Exp A cell: {sorted(set(b5.prediction_recipe))}")
    print(f"  batches entirely DSM-unseen: {sorted(set(b5.n_fully_ignored_batches))}; "
          f"NaN-loss batches: {sorted(set(b5.n_nan_loss_batches))}; "
          f"frozen-encoder params with non-zero grad: {sorted(set(b5.freeze_encoder_nonzero_grad))}")
    print(f"  fully-DSM-unseen patches contributing exactly 0 to the delta: {n_zero_contrib} "
          f"patch-cells checked, all 0")
    print(f"  Exp A trains from {sorted(set(os.path.basename(str(x)) for x in b5.train_manifest))} and "
          f"tests on {sorted(set(os.path.basename(str(x)) for x in b5.test_manifest))}")
    audit.append(dict(
        id="B5", alternative="Unseen-pixel handling differs between Exp A and its parent, or the "
                             "prediction recipe does.",
        status="ruled_out",
        evidence=(f"All 27 Exp A sidecars record DSM-unseen pixels EXCLUDED from the metric at K=0/2/5 "
                  f"through the same prediction-side sentinel as the parents, and EXCLUDED from the loss "
                  f"through the same CrossEntropyLoss(ignore_index=255); the prediction recipe is the "
                  f"parents' own filter_small_predictions(logits, min_pixels=10) then argmax on every "
                  f"cell. The test population is byte-identical to the parent's: image_id, aoi_id, "
                  f"n_valid_px and n_gt_fg_valid were asserted equal to the fold reference on all 27, and "
                  f"every fully-DSM-unseen patch contributes a delta of exactly 0 "
                  f"({n_zero_contrib} patch-cells checked). 0 training batches were entirely ignored and "
                  f"0 returned a NaN loss, and 0 frozen encoder parameters received a non-zero gradient."),
        artifacts=["task_b.b5_unseen_and_recipe"]))

    # ---------------- B6  inference nondeterminism ------------------------
    print("\n--- B6  inference nondeterminism (OGLANet E-9) ---")
    e9 = []
    for r in rows:
        if r.get("category") not in ("cell", "retrain_suffix", "expa"):
            continue
        band = M8.e9_band(r)
        if band:
            e9.append(dict(run_id=r["run_id_dir"], category=r["category"], arch=r["arch"],
                           method=r["method"], fold=r["holdout"], seed=r["seed"], **band))
    e9 = pd.DataFrame(e9)
    e9_expa = e9[e9.category == "expa"] if len(e9) else e9
    e9_ogl = e9[e9.arch == "oglanet"] if len(e9) else e9
    print(f"  E-9 repeat-inference bands on disk, among the in-scope dumps: {len(e9)} cells "
          f"({len(e9_expa)} of them Exp A cells); by architecture "
          f"{e9.arch.value_counts().to_dict() if len(e9) else {}}")
    if len(e9):
        print(e9.groupby(["arch", "n_passes"]).agg(
            n=("run_id", "size"), width_min=("width_miou_pct", "min"),
            width_max=("width_miou_pct", "max")).to_string(float_format=fmt))
    if len(e9_ogl):
        print(e9_ogl.groupby(["method", "n_passes"]).agg(
            n=("run_id", "size"), width_min=("width_miou_pct", "min"),
            width_med=("width_miou_pct", "median"), width_max=("width_miou_pct", "max"),
            sd_max=("sd_miou_pct", "max")).to_string(float_format=fmt))
    ogl_af = b1_af[(b1_af.arch == "oglanet") & (b1_af.comparison == "fg_matched")]
    ogl_min_abs = float(ogl_af["mean"].abs().min())
    w_max = float(e9_ogl.width_miou_pct.max()) if len(e9_ogl) else None
    print(f"  smallest |OGLANet Exp A (arch x fold) delta| {ogl_min_abs:.4f} fg-only IoU points; "
          f"widest measured OGLANet E-9 band on any cell {w_max}")
    audit.append(dict(
        id="B6", alternative="OGLANet's measured inference nondeterminism (E-9) produces the negative "
                             "Exp A sign.",
        status="not_ruled_out",
        evidence=(f"No Exp A cell has a repeat-inference band: {len(e9_expa)} of the 27 Exp A dumps carry "
                  f"an inference_repeat file, so every Exp A number of every architecture is ONE "
                  f"inference draw and the band is unmeasured on exactly the cells the claim rests on. "
                  f"What is measured is the band on {len(e9_ogl)} OGLANet PARENT and method cells (plus "
                  f"{int((e9.arch == 'crossearth').sum())} CrossEarth cells): widths "
                  f"{float(e9_ogl.width_miou_pct.min()):.4f} to {w_max:.4f} two-class mIoU points at "
                  f"pass counts " + str(sorted(set(int(x) for x in e9_ogl.n_passes))) +
                  f" (" + ", ".join(f"{int(k)} passes: {int(v)} cells"
                                    for k, v in e9_ogl.n_passes.value_counts().sort_index().items()) +
                  f"). The smallest OGLANet Exp A (arch x fold) delta is {ogl_min_abs:.4f} fg-only "
                  f"points, which is larger than that band, but the band is on a different metric "
                  f"(two-class), a different cell family, and -- for the 3-pass files -- undersampled "
                  f"as a gate per D-B4-3. Measuring it on the Exp A cells needs GPU inference, which is "
                  f"not this box."),
        artifacts=["task_b.b6_inference_nondeterminism"]))

    # ---------------- B7  seed effects ------------------------------------
    print("\n--- B7  seed effects ---")
    b7 = []
    for arch in EA:
        for fold in FOLDS:
            g = reg[(reg.arch == arch) & (reg.fold == fold)]
            ps = g.set_index("seed").expa_delta_fg
            b7.append(dict(arch=arch, fold=fold, per_seed={int(k): float(v) for k, v in ps.items()},
                           seed_sd=float(ps.std(ddof=1)), n_same_sign=int((np.sign(ps) == np.sign(ps.mean())).sum()),
                           mean=float(ps.mean())))
    b7 = pd.DataFrame(b7)
    print(b7.to_string(index=False, float_format=fmt))
    n_all_same = int((b7.n_same_sign == 3).sum())
    print(f"  (arch x fold) cells whose 3 seeds all share the sign of their mean: {n_all_same} of 9")
    audit.append(dict(
        id="B7", alternative="The separation is seed noise.",
        status="ruled_out",
        evidence=(f"All 27 seed-resolved cells carry the sign of their architecture: "
                  f"{n_all_same} of 9 (arch x fold) cells have all three seeds on the same side as their "
                  f"mean, and the across-seed SD is {float(b7.seed_sd.min()):.4f} to "
                  f"{float(b7.seed_sd.max()):.4f} fg-only points against |cell means| of "
                  f"{float(b7['mean'].abs().min()):.4f} to {float(b7['mean'].abs().max()):.4f}. Three "
                  f"seeds is still three seeds, so this bounds seed noise rather than excluding a "
                  f"seed-by-architecture interaction."),
        artifacts=["task_b.b7_seed_effects"]))

    # ---------------- B8  decoder reinitialisation and capacity -----------
    print("\n--- B8  what Exp A actually varies: the decoder is DISCARDED, not fine-tuned ---")
    cap = b3_cells.groupby("arch").agg(
        n_encoder_elements=("n_encoder_elements", "first"),
        n_decoder_elements=("n_decoder_elements", "first"),
        n_decoder_modules_reinitialised=("n_decoder_modules_reinitialised", "first")).reset_index()
    cap["decoder_share_of_model"] = cap.n_decoder_elements / (cap.n_encoder_elements + cap.n_decoder_elements)
    cap["decoder_elements_per_train_patch"] = cap.n_decoder_elements / 875.0
    print(cap.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("  reinit_decoder discards the parent checkpoint's decoder, so the experiment asks whether the "
          "frozen encoder's features are sufficient for a FROM-SCRATCH decoder trained 30 epochs on 875 "
          "held-out-city patches, not whether the parent decoder can be nudged.")
    audit.append(dict(
        id="B8", alternative="The quantity that differs across architectures is decoder capacity and the "
                             "fact that the decoder is reinitialised from scratch, not the frozen "
                             "encoder's features.",
        status="not_ruled_out",
        evidence=("The published Exp A calls reinit_decoder, which discards the parent checkpoint's "
                  "decoder; the sidecars say so. The three decoders are not comparable objects: "
                  + ", ".join(f"{r.arch} {int(r.n_decoder_elements):,} decoder elements "
                              f"({r.decoder_share_of_model:.1%} of the model), "
                              f"{int(r.n_decoder_modules_reinitialised)} modules reinitialised"
                              for _, r in cap.iterrows()) +
                  ". MAMNet's decoder is about 4x OGLANet's and 10x DINOv3's in parameters and is 44% of "
                  "its model, so a fixed 30-epoch, 875-patch budget is a different fraction of what each "
                  "decoder needs. This is arch-correlated by construction and nothing on this box "
                  "separates it from encoder feature quality."),
        artifacts=["task_b.b8_decoder_capacity"]))

    # ---------------- B9  what Exp A recovers against the upper bound -----
    print("\n--- B9  Exp A against the within-city upper bound, not only against its parent ---")
    b9 = []
    for arch in EA:
        for fold in FOLDS:
            d_up = {s: (levels[(arch, "expa", fold, s, "expa")]["fg"]
                        - levels[(arch, "none", fold, s, "cell")]["fg"]) for s in SEEDS}
            r = paired(d_up, fold, f"b9|{arch}|{fold}", foldref)
            r.pop("dbar")
            r.update(arch=arch, fold=fold,
                     recovery_share_fg=float(reg[(reg.arch == arch) & (reg.fold == fold)]
                                             .recovery_share_fg.mean()))
            b9.append(r)
    b9 = pd.DataFrame(b9)
    b9["p_holm"] = holm(b9.p_raw.tolist())
    print(b9[["arch", "fold", "mean", "ci_lo", "ci_hi", "p_raw", "p_holm", "recovery_share_fg"]]
          .to_string(index=False, float_format=fmt))
    audit.append(dict(
        id="B9", alternative="A positive Exp A delta means the frozen encoder recovers the gap.",
        status="not_an_alternative_but_a_scope_limit",
        evidence=(f"Exp A is measured against its source-only parent, not against the within-city upper "
                  f"bound. Against the upper bound every (arch x fold) cell of all three architectures is "
                  f"still below it: {float(b9['mean'].max()):+.4f} is the best and "
                  f"{float(b9['mean'].min()):+.4f} the worst fg-only delta, "
                  f"{int((b9['mean'] < 0).sum())} of 9 negative. The fraction of the parent's fg-only gap "
                  f"that Exp A closes is {float(b9.recovery_share_fg.min()):+.3f} to "
                  f"{float(b9.recovery_share_fg.max()):+.3f}. Exp A also uses held-out-city LABELS "
                  f"(target_labels), which neither its parent nor any method row does."),
        artifacts=["task_b.b9_vs_upper_bound"]))

    # ---------------- B10  the parents' own convergence -------------------
    print("\n--- B10  are the Exp A parents themselves comparably converged? ---")
    par = desc[(desc.category == "cell") & (desc.method == "vanilla") & (desc.arch.isin(EA))]
    b10 = (par.groupby("arch").agg(
        n=("run_id", "size"), epochs_max=("epochs_max", lambda s: sorted(set(s.dropna()))),
        epochs_run_min=("epochs_run", "min"), epochs_run_max=("epochs_run", "max"),
        best_epoch_min=("best_epoch", "min"), best_epoch_max=("best_epoch", "max"),
        n_ceiling=("stopped_on", lambda s: int((s == "epoch_ceiling").sum()))).reset_index())
    print(b10.to_string(index=False))
    ceil_par = par[par.stopped_on == "epoch_ceiling"].run_id.tolist()
    print(f"  Exp A parents that stopped at their epoch ceiling rather than on patience: {ceil_par}")
    audit.append(dict(
        id="B10", alternative="The Exp A parents are not comparably converged, so the frozen encoders "
                              "being compared are at different points of their own training.",
        status="not_ruled_out",
        evidence=(f"The parents early-stop on patience 20 with model selection, but not at comparable "
                  f"points: epochs_max is " + "; ".join(
                      f"{r.arch} {r.epochs_max}" for _, r in b10.iterrows()) +
                  ", epochs_run " + "; ".join(f"{r.arch} {int(r.epochs_run_min)}-{int(r.epochs_run_max)}"
                                              for _, r in b10.iterrows()) +
                  ", best_epoch " + "; ".join(f"{r.arch} {int(r.best_epoch_min)}-{int(r.best_epoch_max)}"
                                              for _, r in b10.iterrows()) +
                  f". {len(ceil_par)} of the 27 Exp A parents stopped at their epoch ceiling rather than "
                  f"on patience ({ceil_par}), so those frozen encoders are the ones training ran out of "
                  f"budget on. Only OGLANet has a from-scratch longer-horizon retrain (D-A2-1), and no "
                  f"Exp A cell was re-run from it."),
        artifacts=["task_b.b10_parent_convergence"]))

    # ---------------- B11  train-pool mismatch ----------------------------
    print("\n--- B11  Exp A's own training pool: sun elevation and patch budget ---")
    b11 = []
    for _, r in expa.iterrows():
        m = meta_by_rid[r.run_id]
        pm = meta_by_rid[f"{r.arch}-vanilla-loco-r3500-{r.fold}-hr-s{r.seed}"]
        b11.append(dict(run_id=r.run_id, arch=r.arch, fold=r.fold, seed=r.seed,
                        expa_train_sun_elev_mean=m.get("train_sun_elev_mean"),
                        expa_train_sun_elev_sd=m.get("train_sun_elev_sd"),
                        parent_train_sun_elev_mean=pm.get("train_sun_elev_mean"),
                        parent_train_sun_elev_sd=pm.get("train_sun_elev_sd"),
                        expa_n_train=m.get("n_train"), parent_n_train=pm.get("n_train"),
                        expa_train_aoi=m.get("n_train_aoi"), parent_train_aoi=pm.get("n_train_aoi"),
                        train_pool_n_with_unseen_pixels=m.get("train_pool_n_with_unseen_pixels"),
                        train_fraction=m.get("train_fraction")))
    b11 = pd.DataFrame(b11)
    print(b11.groupby(["fold"])[["expa_train_sun_elev_mean", "expa_train_sun_elev_sd",
                                 "parent_train_sun_elev_mean", "parent_train_sun_elev_sd",
                                 "expa_n_train", "parent_n_train"]].mean().to_string(float_format=fmt))
    audit.append(dict(
        id="B11", alternative="Exp A's own training pool explains the result: it trains on 875 held-out "
                              "city patches, a quarter of the parent's budget, from a different sun-"
                              "elevation distribution.",
        status="not_ruled_out_but_not_arch_correlated",
        evidence=(f"Exp A trains on {int(b11.expa_n_train.iloc[0])} patches (train_fraction "
                  f"{b11.train_fraction.iloc[0]}) of the held-out city's upper train list against the "
                  f"parent's {int(b11.parent_n_train.iloc[0])} source-city patches, and its training-pool "
                  f"sun-elevation mean is the held-out city's, not the source cities' (per fold: " +
                  "; ".join(f"{f} Exp A {float(b11[b11.fold == f].expa_train_sun_elev_mean.iloc[0]):.2f}"
                            f"+-{float(b11[b11.fold == f].expa_train_sun_elev_sd.iloc[0]):.2f} vs parent "
                            f"{float(b11[b11.fold == f].parent_train_sun_elev_mean.iloc[0]):.2f}"
                            f"+-{float(b11[b11.fold == f].parent_train_sun_elev_sd.iloc[0]):.2f}"
                            for f in FOLDS) + "). This bounds the magnitude of any Exp A delta but is "
                  f"identical across the three architectures, so it cannot produce the architecture "
                  f"separation. The sidecars' sun_dist column is explicitly not comparable between an "
                  f"Exp A cell and its parent and was not used."),
        artifacts=["task_b.b11_train_pool"]))

    # ---------------- B12  not checkable here -----------------------------
    for item in (
        dict(id="B12",
             alternative="The frozen OGLANet encoder's BatchNorm statistics behave differently from "
                         "DINOv3's LayerNorm under train_decoder's 'put the model in train() then force "
                         "encoder BatchNorm back to eval' step.",
             status="not_checkable_here",
             evidence=("The sidecars record that the step happens and that the assertions were run with "
                       "model.eval(), but no sidecar or dump carries a per-module normalisation inventory "
                       "or the running statistics, and nothing here records whether a given encoder has "
                       "BatchNorm at all. Separating this needs the checkpoints, which stay on Delta."),
             artifacts=[]),
        dict(id="B13",
             alternative="A different frozen-encoder recipe (longer horizon, parent decoder fine-tuned "
                         "instead of reinitialised, or a validation split and model selection) would not "
                         "separate by architecture.",
             status="not_checkable_here",
             evidence=("Only one Exp A recipe was run. There is no second horizon, no fine-tune variant "
                       "and no validation split anywhere in the 27 dumps, so the recipe's contribution "
                       "cannot be estimated on this box."),
             artifacts=[]),
        dict(id="B14",
             alternative="CrossEarth would break the architecture separation.",
             status="not_checkable_here",
             evidence=("There is no published CrossEarth Experiment A. The Exp A sidecars record the "
                       "grep that found no crossearth entry in ENCODER_PREFIXES or load_model, so the "
                       "separation is over 3 architectures, not 4."),
             artifacts=[]),
    ):
        audit.append(item)

    aud = pd.DataFrame(audit)
    print("\n--- B  audit summary ---")
    print(aud[["id", "status", "alternative"]].to_string(index=False, max_colwidth=78))
    print(f"\n  ruled out: {int((aud.status == 'ruled_out').sum())}; "
          f"not ruled out: {int(aud.status.str.startswith('not_ruled_out').sum())}; "
          f"not checkable here: {int((aud.status == 'not_checkable_here').sum())}; "
          f"other: {int((~aud.status.isin(['ruled_out', 'not_checkable_here']) & ~aud.status.str.startswith('not_ruled_out')).sum())}")

    st["task_b"] = dict(
        scope=("Exp A freezes the encoder of each trained source-only LOCO model, REINITIALISES the "
               "decoder, and trains it for a fixed 30 epochs on 875 patches of the held-out city's own "
               "upper train list with the held-out city's labels (target_access target_labels). 27 cells: "
               "3 architectures x 3 folds x 3 seeds. CrossEarth has no published Exp A."),
        audit=audit,
        audit_counts=dict(ruled_out=int((aud.status == "ruled_out").sum()),
                          not_ruled_out=int(aud.status.str.startswith("not_ruled_out").sum()),
                          not_checkable_here=int((aud.status == "not_checkable_here").sum()),
                          other=int(len(aud) - (aud.status == "ruled_out").sum()
                                    - aud.status.str.startswith("not_ruled_out").sum()
                                    - (aud.status == "not_checkable_here").sum())),
        b1_metric_denominator=dict(
            matched_comparison=("Exp A reporting_metric (fg-only IoU at K=2, DSM-unseen excluded) against "
                                "the parent's sanity_floor_metric_value_pct, which is the same quantity. "
                                "Both recomputed here from the raw tp/fp/fn counts."),
            max_abs_recompute_minus_sidecar=r6(max_rec),
            per_cell=r6(b1_cells.to_dict("records")),
            archfold=r6(b1_af.to_dict("records")),
            separation_by_differencing=r6(sep),
            crosscheck_vs_08_task3=r6(xc3)),
        b2_training_instability=dict(
            control_range_normalised_ratio=[r6(thr), r6(a1["control_range"]["ratio_max"])],
            by_arch=r6(b2.to_dict("records")),
            n_expa_cells_with_val_history=n_hist_expa,
            per_cell=r6(expa[["run_id", "arch", "fold", "seed", "predpos_frac_pooled",
                              "predpos_over_gt_pooled", "share_zero_fg_pred", "share_allfg_pred",
                              "mean_fgonly", "mean_2class", "below_control", "near_zero"]].to_dict("records"))),
        b3_convergence=dict(by_arch=r6(b3.to_dict("records")), per_cell=r6(b3_cells.to_dict("records")),
                            train_loss_order_lowest_first=order_loss,
                            expa_pooled_delta_order_largest_first=order_delta,
                            expa_pooled_delta_from_08=r6(expa_pooled)),
        b4_gap_recovery=dict(regressions=r6(b4), per_cell=r6(reg.to_dict("records")),
                             overlap=r6(ov.to_dict("records"))),
        b5_unseen_and_recipe=dict(per_cell=r6(b5.to_dict("records")),
                                  n_fully_unseen_patch_cells_checked=n_zero_contrib),
        b6_inference_nondeterminism=dict(
            n_e9_bands_total=int(len(e9)), n_e9_bands_on_expa=int(len(e9_expa)),
            n_e9_bands_by_arch=({str(k): int(v) for k, v in e9.arch.value_counts().items()}
                                if len(e9) else {}),
            all_bands=r6(e9.to_dict("records")) if len(e9) else [],
            oglanet_bands=r6(e9_ogl.to_dict("records")) if len(e9_ogl) else [],
            oglanet_band_pass_counts=({int(k): int(v) for k, v in
                                       e9_ogl.n_passes.value_counts().sort_index().items()}
                                      if len(e9_ogl) else {}),
            smallest_abs_oglanet_expa_archfold_delta=r6(ogl_min_abs)),
        b7_seed_effects=r6(b7.to_dict("records")),
        b8_decoder_capacity=r6(cap.to_dict("records")),
        b9_vs_upper_bound=r6(b9.to_dict("records")),
        b10_parent_convergence=dict(by_arch=r6(b10.to_dict("records")),
                                    parents_at_epoch_ceiling=ceil_par),
        b11_train_pool=r6(b11.to_dict("records")))
    return st


# ============================================================================
# TASK C -- why is Omaha the exception?
# ============================================================================

def task_c(st):
    desc, foldref, rows, R8 = st["_desc"], st["_foldref"], st["_rows"], st["_R8"]
    pbar = st["_pbar"]
    print("\n" + "=" * 110)
    print("TASK C -- why is Omaha the exception?")
    print("=" * 110)
    fmt = lambda v: f"{v:.4f}"
    meta_by_rid = {r["run_id_dir"]: r["meta"] for r in rows if "meta" in r}

    # ---------------- C1  dataset properties per city ---------------------
    print("\n--- C1  dataset properties, from the fold reference frames (identical across all cells "
          "of a fold; asserted on every dump) ---")
    city = []
    for fold in FOLDS:
        d = foldref[fold]["frame"]
        nvp = d.n_valid_px.to_numpy(np.float64)
        ngt = d.n_gt_fg_valid.to_numpy(np.float64)
        defined = nvp > 0
        fgfrac = np.full(len(d), np.nan)
        fgfrac[defined] = ngt[defined] / nvp[defined]
        bgfrac = 1.0 - fgfrac
        pat_per_aoi = d.groupby("aoi_id").size()
        se = d.sun_elevation.to_numpy(np.float64)
        row = dict(
            fold=fold, city=str(d.city.iloc[0]),
            n_test_patches=int(len(d)), n_aoi=int(d.aoi_id.nunique()),
            n_source_images=int(d.source_image_id.nunique()),
            patches_per_aoi_mean=float(pat_per_aoi.mean()), patches_per_aoi_min=int(pat_per_aoi.min()),
            patches_per_aoi_max=int(pat_per_aoi.max()),
            n_acquisitions=int(d.acquisition_date.nunique()),
            acquisition_dates=sorted(d.acquisition_date.unique().tolist()),
            n_distinct_sun_elevation=int(d.sun_elevation.nunique()),
            sun_elev_mean=float(se.mean()), sun_elev_sd=float(se.std(ddof=1)),
            sun_elev_min=float(se.min()), sun_elev_max=float(se.max()),
            sun_elev_iqr=float(np.percentile(se, 75) - np.percentile(se, 25)),
            shadow_frac_mean=float(np.nanmean(fgfrac)), shadow_frac_median=float(np.nanmedian(fgfrac)),
            shadow_frac_sd=float(np.nanstd(fgfrac, ddof=1)),
            shadow_frac_p10=float(np.nanpercentile(fgfrac, 10)),
            shadow_frac_p90=float(np.nanpercentile(fgfrac, 90)),
            shadow_frac_max=float(np.nanmax(fgfrac)),
            n_patch_shadow_frac_gt_50=int(np.nansum(fgfrac > 0.50)),
            n_patch_shadow_frac_gt_90=int(np.nansum(fgfrac > 0.90)),
            n_patch_no_foreground=int(d.empty_fg_valid.sum()),
            share_patch_no_foreground=float(d.empty_fg_valid.mean()),
            n_patch_all_shadow_flag=int(d.all_shadow_flag.sum()),
            n_patch_zero_background_px=int(np.nansum(bgfrac <= 0.0) + int((~defined).sum())),
            n_patch_background_lt_1pct=int(np.nansum(bgfrac < 0.01)),
            unseen_frac_mean=float(d.unseen_frac.mean()), unseen_frac_median=float(d.unseen_frac.median()),
            unseen_frac_p90=float(d.unseen_frac.quantile(0.90)),
            unseen_frac_max=float(d.unseen_frac.max()),
            n_patch_fully_unseen=int((nvp == 0).sum()),
            share_patch_fully_unseen=float((nvp == 0).mean()),
            n_pbarS_undefined=int(d.pbar_gt_shadow.isna().sum()),
            n_pbarB_undefined=int(d.pbar_gt_bg.isna().sum()),
            n_aoi_with_no_defined_pbarB_patch=int(
                (d.assign(z=d.pbar_gt_bg.isna()).groupby("aoi_id").z.all()).sum()),
            n_train=int(meta_by_rid[f"mamnet-vanilla-loco-r3500-{fold}-hr-s0"]["n_train"]),
            n_train_aoi=int(meta_by_rid[f"mamnet-vanilla-loco-r3500-{fold}-hr-s0"]["n_train_aoi"]),
            loco_train_sun_elev_mean=float(
                meta_by_rid[f"mamnet-vanilla-loco-r3500-{fold}-hr-s0"]["train_sun_elev_mean"]),
            loco_train_sun_elev_sd=float(
                meta_by_rid[f"mamnet-vanilla-loco-r3500-{fold}-hr-s0"]["train_sun_elev_sd"]),
            upper_train_sun_elev_mean=float(
                meta_by_rid[f"mamnet-none-upper-r3500-{fold}-hr-s0"]["train_sun_elev_mean"]),
            upper_train_sun_elev_sd=float(
                meta_by_rid[f"mamnet-none-upper-r3500-{fold}-hr-s0"]["train_sun_elev_sd"]),
            sun_dist_loco_mean=float(d.sun_dist.mean()))
        row["loco_train_test_sun_elev_gap"] = row["loco_train_sun_elev_mean"] - row["sun_elev_mean"]
        row["upper_train_test_sun_elev_gap"] = row["upper_train_sun_elev_mean"] - row["sun_elev_mean"]
        city.append(row)
    city = pd.DataFrame(city).set_index("fold")
    show = ["n_test_patches", "n_aoi", "n_acquisitions", "patches_per_aoi_mean",
            "n_distinct_sun_elevation", "sun_elev_mean", "sun_elev_sd", "sun_elev_iqr",
            "shadow_frac_mean", "shadow_frac_sd", "shadow_frac_max",
            "n_patch_shadow_frac_gt_50", "n_patch_shadow_frac_gt_90",
            "n_patch_no_foreground", "n_patch_all_shadow_flag", "n_patch_zero_background_px",
            "unseen_frac_mean", "unseen_frac_p90", "n_patch_fully_unseen",
            "n_pbarS_undefined", "n_pbarB_undefined", "n_aoi_with_no_defined_pbarB_patch",
            "loco_train_sun_elev_mean", "loco_train_sun_elev_sd",
            "loco_train_test_sun_elev_gap", "upper_train_test_sun_elev_gap", "sun_dist_loco_mean"]
    print(city[show].T.to_string(float_format=fmt))

    # ---------------- C2  model-side --------------------------------------
    print("\n--- C2  model-side quantities per city ---")
    t1 = pd.DataFrame(R8["task1"]["cells"])
    g2 = t1[t1.metric == "2class"][["arch", "fold", "mean", "loco", "upper", "ci_lo", "ci_hi", "p_holm"]]
    gf = t1[t1.metric == "fgonly"][["arch", "fold", "mean", "loco", "upper"]]
    gf.columns = ["arch", "fold", "gap_fgonly", "loco_fgonly", "upper_fgonly"]
    gaps = g2.merge(gf, on=["arch", "fold"]).rename(columns={"mean": "gap_2class"})
    print("  vanilla gap per (arch, fold), from results/08 task 1:")
    print(gaps.pivot(index="arch", columns="fold", values="gap_2class").to_string(float_format=fmt))
    print(f"  per fold: 2-class gap range " + "; ".join(
        f"{f} {gaps[gaps.fold == f].gap_2class.min():.4f}-{gaps[gaps.fold == f].gap_2class.max():.4f}"
        for f in FOLDS))
    cells = desc[(desc.category == "cell")]
    pp = (cells[cells.method.isin(["vanilla", "none"])]
          .groupby(["fold", "method"])
          .agg(predpos_frac=("predpos_frac_pooled", "mean"),
               gt_rate=("gt_shadow_rate_pooled", "first"),
               ratio=("predpos_over_gt_pooled", "mean"),
               zero_fg_share=("share_zero_fg_pred", "mean"),
               allfg_share=("share_allfg_pred", "mean")).reset_index())
    print("\n  predicted-positive fraction, vanilla-LOCO and upper, by fold:")
    print(pp.to_string(index=False, float_format=fmt))
    print("\n  all-shadow patches: the all_shadow_flag column is true on "
          + ", ".join(f"{f.upper()} {int(city.loc[f, 'n_patch_all_shadow_flag'])}" for f in FOLDS)
          + " patches, and the flag is a GT property of the fold, identical in every cell.")
    # how the 11 OMA all-shadow patches are scored
    ash = []
    d_oma = foldref["oma"]["frame"]
    mask_ash = d_oma.all_shadow_flag.to_numpy()
    for arch in ARCHS:
        for mode, m in (("loco", "vanilla"), ("upper", "none")):
            v = np.mean([CELL_FG[(arch, m, "oma", s)][mask_ash] for s in SEEDS], axis=0)
            v2 = np.mean([CELL_MIOU[(arch, m, "oma", s)][mask_ash] for s in SEEDS], axis=0)
            ash.append(dict(arch=arch, mode=mode, n=int(mask_ash.sum()),
                            mean_fgonly=float(v.mean()), mean_2class=float(v2.mean())))
    ash = pd.DataFrame(ash)
    print("  the 11 OMA all-shadow patches, seed-averaged:")
    print(ash.to_string(index=False, float_format=fmt))

    # ---------------- C3  probe-side p-bar --------------------------------
    print("\n--- C3  probe-side: p-bar_S and p-bar_B per pair, recomputed, and the Omaha AOI "
          "sensitivity ---")
    t5b = pd.DataFrame(R8["task5b_pbar_S"]["pairs"])
    pairs, aoi_tables = [], {}
    for arch in ARCHS:
        for fold in FOLDS:
            keepS, keepB = foldref[fold]["keepS"], foldref[fold]["keepB"]
            aoi = foldref[fold]["aoi"]
            row = dict(arch=arch, fold=fold)
            for tag, col, keep in (("S", "S", keepS), ("B", "Bg", keepB)):
                dl = {s: (pbar[(arch, fold, s, "loco")][col] - pbar[(arch, fold, s, "upper")][col])[keep]
                      for s in SEEDS}
                if any(np.isnan(v).any() for v in dl.values()):
                    fail(f"{arch}/{fold}/pbar_{tag}: NaN survives the keep mask")
                codes, uniq = pd.factorize(aoi[keep])
                dbar = np.mean([dl[s] for s in SEEDS], axis=0)
                boot = aoi_boot(dbar, codes, len(uniq), stream_of(f"c3boot|{tag}|{arch}|{fold}"))
                obs, p = aoi_flip(dbar, codes, len(uniq), stream_of(f"c3flip|{tag}|{arch}|{fold}"))
                lo, hi = ci95(boot)
                row[f"d_pbar{tag}"] = obs
                row[f"d_pbar{tag}_ci_lo"], row[f"d_pbar{tag}_ci_hi"] = lo, hi
                row[f"d_pbar{tag}_p_raw"] = p
                row[f"loco_pbar{tag}"] = float(np.mean(
                    [pbar[(arch, fold, s, "loco")][col][keep].mean() for s in SEEDS]))
                row[f"upper_pbar{tag}"] = float(np.mean(
                    [pbar[(arch, fold, s, "upper")][col][keep].mean() for s in SEEDS]))
                row[f"n_images_{tag}"] = int(keep.sum())
                row[f"n_aoi_{tag}"] = int(len(uniq))
                aoi_tables[(arch, fold, tag)] = pd.DataFrame(
                    {"aoi_id": aoi[keep], "d": dbar}).groupby("aoi_id").agg(
                    d=("d", "mean"), n=("d", "size")).reset_index()
            row["ratio_absS_over_absB"] = (abs(row["d_pbarS"]) / abs(row["d_pbarB"])
                                           if row["d_pbarB"] != 0 else None)
            pairs.append(row)
    pairs = pd.DataFrame(pairs)
    for tag in ("S", "B"):
        pairs[f"d_pbar{tag}_p_holm"] = holm(pairs[f"d_pbar{tag}_p_raw"].tolist())
    print(pairs[["arch", "fold", "loco_pbarS", "upper_pbarS", "d_pbarS", "d_pbarS_ci_lo",
                 "d_pbarS_ci_hi", "d_pbarS_p_holm", "d_pbarB", "d_pbarB_p_holm",
                 "ratio_absS_over_absB"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    # cross-check against results/08 task5b
    xc5 = []
    for _, r in pairs.iterrows():
        hit = t5b[(t5b.arch == r.arch) & (t5b.fold == r.fold)]
        if len(hit) != 1:
            fail(f"task5b has {len(hit)} rows for {r.arch}/{r.fold}")
        hS, hB = hit.iloc[0]["pbar_S"], hit.iloc[0]["pbar_B_control"]
        xc5.append(dict(
            arch=r.arch, fold=r.fold,
            dS_mine=r.d_pbarS, dS_08=float(hS["delta"]), dS_absdiff=abs(r.d_pbarS - float(hS["delta"])),
            dB_mine=r.d_pbarB, dB_08=float(hB["delta"]), dB_absdiff=abs(r.d_pbarB - float(hB["delta"])),
            locoS_mine=r.loco_pbarS, locoS_08=float(hS["loco_mean"]),
            upperS_mine=r.upper_pbarS, upperS_08=float(hS["upper_mean"]),
            nimgS_mine=int(r.n_images_S), nimgS_08=int(hS["n_images"]),
            naoiS_mine=int(r.n_aoi_S), naoiS_08=int(hS["n_aoi"]),
            nimgB_mine=int(r.n_images_B), nimgB_08=int(hB["n_images"]),
            naoiB_mine=int(r.n_aoi_B), naoiB_08=int(hB["n_aoi"]),
            ratio_mine=r.ratio_absS_over_absB,
            ratio_08=float(hit.iloc[0]["fg_over_bg_abs_delta_ratio"])))
    xc5 = pd.DataFrame(xc5)
    print(f"\n  cross-check vs results/08 task5b_pbar_S.pairs: max |Delta p-bar_S difference| "
          f"{xc5.dS_absdiff.max():.2e}, max |Delta p-bar_B difference| {xc5.dB_absdiff.max():.2e}")
    if max(xc5.dS_absdiff.max(), xc5.dB_absdiff.max()) > 1e-6:
        print(xc5.sort_values("dS_absdiff", ascending=False).head(6).to_string(index=False))
        fail("recomputed p-bar deltas do not reproduce results/08 task5b")
    for a, b_ in (("nimgS_mine", "nimgS_08"), ("naoiS_mine", "naoiS_08"),
                  ("nimgB_mine", "nimgB_08"), ("naoiB_mine", "naoiB_08")):
        if not (xc5[a] == xc5[b_]).all():
            fail(f"p-bar population size {a} differs from results/08 {b_}")
    print(f"  populations match results/08 on all 12 pairs for both statistics; max |ratio difference| "
          f"{float((xc5.ratio_mine - xc5.ratio_08).abs().max()):.2e}")

    # the reframing: is the Omaha ratio driven by the numerator or the denominator?
    byfold = pairs.groupby("fold").agg(
        mean_d_pbarS=("d_pbarS", "mean"), mean_abs_d_pbarS=("d_pbarS", lambda s: float(s.abs().mean())),
        mean_d_pbarB=("d_pbarB", "mean"), mean_abs_d_pbarB=("d_pbarB", lambda s: float(s.abs().mean())),
        mean_loco_pbarS=("loco_pbarS", "mean"), mean_upper_pbarS=("upper_pbarS", "mean"),
        mean_loco_pbarB=("loco_pbarB", "mean"), mean_upper_pbarB=("upper_pbarB", "mean"),
        ratio_min=("ratio_absS_over_absB", "min"), ratio_max=("ratio_absS_over_absB", "max")).reset_index()
    print("\n  per fold -- which side of the ratio moves:")
    print(byfold.to_string(index=False, float_format=fmt))

    # AOI sensitivity on the Omaha pairs
    print("\n--- C3b  Omaha AOI sensitivity: recompute each Omaha ratio with the most extreme AOIs "
          "dropped ---")
    sens, jack = [], []
    for arch in ARCHS:
        base = pairs[(pairs.arch == arch) & (pairs.fold == "oma")].iloc[0]
        tS, tB = aoi_tables[(arch, "oma", "S")], aoi_tables[(arch, "oma", "B")]

        def recompute(drop_aoi):
            out = {}
            for tag, t in (("S", tS), ("B", tB)):
                keep = t[~t.aoi_id.isin(drop_aoi)]
                out[tag] = float((keep.d * keep.n).sum() / keep.n.sum())
                out[f"n_aoi_{tag}"] = int(len(keep))
            out["ratio"] = abs(out["S"]) / abs(out["B"]) if out["B"] != 0 else None
            return out

        full = recompute(set())
        if abs(full["S"] - base.d_pbarS) > 1e-9 or abs(full["B"] - base.d_pbarB) > 1e-9:
            fail(f"{arch}/oma: AOI-weighted reconstruction != the pooled estimate")
        # leave-one-AOI-out jackknife on the ratio
        jr = []
        for a in sorted(set(tS.aoi_id) | set(tB.aoi_id)):
            r = recompute({a})
            jr.append(dict(arch=arch, dropped=a, ratio=r["ratio"], dS=r["S"], dB=r["B"]))
        jr = pd.DataFrame(jr)
        jack.append(jr)
        for crit, t in (("by |contribution to d_pbarB|", tB), ("by |contribution to d_pbarS|", tS)):
            contrib = t.assign(c=(t.d * t.n).abs()).sort_values("c", ascending=False)
            for k in AOI_DROP_K:
                drop = set(contrib.aoi_id.head(k))
                r = recompute(drop)
                sens.append(dict(arch=arch, criterion=crit, k=k, dropped=sorted(drop),
                                 d_pbarS=r["S"], d_pbarB=r["B"], ratio=r["ratio"],
                                 n_aoi_S=r["n_aoi_S"], n_aoi_B=r["n_aoi_B"],
                                 ratio_full=base.ratio_absS_over_absB,
                                 d_pbarS_full=base.d_pbarS, d_pbarB_full=base.d_pbarB))
        # the all-shadow AOIs: drop every AOI that holds an all-shadow patch
        ash_aoi = set(d_oma.loc[d_oma.all_shadow_flag, "aoi_id"].unique())
        r = recompute(ash_aoi)
        sens.append(dict(arch=arch, criterion="every AOI holding an all_shadow_flag patch",
                         k=len(ash_aoi), dropped=sorted(ash_aoi), d_pbarS=r["S"], d_pbarB=r["B"],
                         ratio=r["ratio"], n_aoi_S=r["n_aoi_S"], n_aoi_B=r["n_aoi_B"],
                         ratio_full=base.ratio_absS_over_absB,
                         d_pbarS_full=base.d_pbarS, d_pbarB_full=base.d_pbarB))
    sens = pd.DataFrame(sens)
    jack = pd.concat(jack, ignore_index=True)
    print(sens[["arch", "criterion", "k", "d_pbarS_full", "d_pbarS", "d_pbarB_full", "d_pbarB",
                "ratio_full", "ratio", "n_aoi_S", "n_aoi_B"]].to_string(index=False, float_format=fmt))
    print("\n  leave-one-AOI-out range of the Omaha ratio:")
    print(jack.groupby("arch").agg(n=("dropped", "size"), ratio_min=("ratio", "min"),
                                   ratio_max=("ratio", "max"), dS_min=("dS", "min"),
                                   dS_max=("dS", "max"), dB_min=("dB", "min"),
                                   dB_max=("dB", "max")).to_string(float_format=fmt))
    ash_aoi = sorted(set(d_oma.loc[d_oma.all_shadow_flag, "aoi_id"].unique()))
    print(f"  the 11 OMA all-shadow patches sit in {len(ash_aoi)} AOIs: {ash_aoi}")
    print(f"  the one OMA AOI with no defined p-bar_B patch: "
          f"{sorted((d_oma.assign(z=d_oma.pbar_gt_bg.isna()).groupby('aoi_id').z.all().loc[lambda s: s].index))}")

    # ---------------- C4  which differences are large enough? -------------
    print("\n--- C4  which measured differences could plausibly bear on the probe behaviour ---")
    verdicts = []

    def add(prop, oma, jax, ucsd, where, verdict, why):
        verdicts.append(dict(property=prop, oma=oma, jax=jax, ucsd=ucsd,
                             where=where, bears_on_probe=verdict, reasoning=why))

    c = city
    ashdrop = sens[sens.criterion == "every AOI holding an all_shadow_flag patch"]
    jk = jack.groupby("arch").agg(rmin=("ratio", "min"), rmax=("ratio", "max")).reset_index()

    add("Delta p-bar_S, mean absolute over the 4 architectures",
        r6(float(byfold[byfold.fold == "oma"].mean_abs_d_pbarS.iloc[0])),
        r6(float(byfold[byfold.fold == "jax"].mean_abs_d_pbarS.iloc[0])),
        r6(float(byfold[byfold.fold == "ucsd"].mean_abs_d_pbarS.iloc[0])),
        "task_c.probe_side.by_fold", "this_is_the_difference",
        "The Omaha ratio is small because its NUMERATOR is small, not because its denominator is large. "
        f"Omaha's mean |Delta p-bar_B| is "
        f"{float(byfold[byfold.fold == 'oma'].mean_abs_d_pbarB.iloc[0]):.4f} against UCSD's "
        f"{float(byfold[byfold.fold == 'ucsd'].mean_abs_d_pbarB.iloc[0]):.4f}, which is the same order; "
        f"its mean |Delta p-bar_S| is "
        f"{float(byfold[byfold.fold == 'oma'].mean_abs_d_pbarS.iloc[0]):.4f} against UCSD's "
        f"{float(byfold[byfold.fold == 'ucsd'].mean_abs_d_pbarS.iloc[0]):.4f} and JAX's "
        f"{float(byfold[byfold.fold == 'jax'].mean_abs_d_pbarS.iloc[0]):.4f}, 5x to 6x smaller.")
    add("p-bar_S level, LOCO side (mean over the 4 architectures)",
        r6(float(byfold[byfold.fold == "oma"].mean_loco_pbarS.iloc[0])),
        r6(float(byfold[byfold.fold == "jax"].mean_loco_pbarS.iloc[0])),
        r6(float(byfold[byfold.fold == "ucsd"].mean_loco_pbarS.iloc[0])),
        "task_c.probe_side.by_fold", "no",
        "The three cities' LOCO models are within 0.03 of each other in mean shadow confidence, so the "
        "LOCO side is not where Omaha differs.")
    add("p-bar_S level, UPPER-BOUND side (mean over the 4 architectures)",
        r6(float(byfold[byfold.fold == "oma"].mean_upper_pbarS.iloc[0])),
        r6(float(byfold[byfold.fold == "jax"].mean_upper_pbarS.iloc[0])),
        r6(float(byfold[byfold.fold == "ucsd"].mean_upper_pbarS.iloc[0])),
        "task_c.probe_side.by_fold", "yes",
        "This is where the asymmetry sits. Omaha's WITHIN-CITY model does not reach the shadow "
        "confidence the other two cities' within-city models reach, so the LOCO-minus-upper difference "
        "has less room on Omaha. The quantity that makes Omaha the exception is a property of its upper "
        "bound, not of its LOCO cells.")
    add("predicted-positive fraction, vanilla-LOCO relative to its own upper bound",
        "; ".join(f"{r.method} {r.predpos_frac:.4f}" for _, r in pp[pp.fold == "oma"].iterrows()),
        "; ".join(f"{r.method} {r.predpos_frac:.4f}" for _, r in pp[pp.fold == "jax"].iterrows()),
        "; ".join(f"{r.method} {r.predpos_frac:.4f}" for _, r in pp[pp.fold == "ucsd"].iterrows()),
        "task_c.model_side.predpos", "yes",
        "Omaha is the only fold where the vanilla-LOCO model predicts MORE positive pixels than its own "
        "upper bound (0.1081 against 0.0945); on JAX and UCSD the upper bound predicts more. That is the "
        "same direction as the positive Delta p-bar_B on all four Omaha pairs and is a second, "
        "independent measurement of the same thing: on Omaha the LOCO model over-predicts shadow "
        "relative to the within-city model rather than under-predicting it.")
    add("mean sun_dist of the LOCO cells (|patch sun elevation - LOCO training-pool mean|)",
        r6(c.loc["oma", "sun_dist_loco_mean"]), r6(c.loc["jax", "sun_dist_loco_mean"]),
        r6(c.loc["ucsd", "sun_dist_loco_mean"]),
        "task_c.city_properties", "possible",
        "Omaha is the fold whose LOCO training illumination sits closest to its own test illumination "
        f"({c.loc['oma', 'sun_dist_loco_mean']:.2f} degrees against "
        f"{c.loc['jax', 'sun_dist_loco_mean']:.2f} on JAX and "
        f"{c.loc['ucsd', 'sun_dist_loco_mean']:.2f} on UCSD), and it also has the narrowest test-side "
        f"spread (SD {c.loc['oma', 'sun_elev_sd']:.2f} against {c.loc['jax', 'sun_elev_sd']:.2f} and "
        f"{c.loc['ucsd', 'sun_elev_sd']:.2f}). The ordering matches the Delta p-bar_S ordering across "
        "the three folds. It is a correlate at n = 3 folds, not a tested relationship, and results/08 "
        "task 4 reports the within-city sun_dist slope as negative and not significant under "
        "fold-clustered SEs.")
    add("GT shadow-fraction distribution (SD across patches; maximum patch fraction)",
        f"{r6(c.loc['oma', 'shadow_frac_sd'])}; {r6(c.loc['oma', 'shadow_frac_max'])}",
        f"{r6(c.loc['jax', 'shadow_frac_sd'])}; {r6(c.loc['jax', 'shadow_frac_max'])}",
        f"{r6(c.loc['ucsd', 'shadow_frac_sd'])}; {r6(c.loc['ucsd', 'shadow_frac_max'])}",
        "task_c.city_properties", "no",
        "Omaha's shadow-fraction distribution is the widest and the only one with a fat upper tail, but "
        f"the tail is one AOI: all {int(c.loc['oma', 'n_patch_all_shadow_flag'])} all-shadow patches sit "
        f"in {ash_aoi}, and dropping that whole AOI moves the four Omaha ratios by at most "
        f"{float((ashdrop.ratio - ashdrop.ratio_full).abs().max()):.4f} "
        f"({', '.join(f'{r.arch} {r.ratio_full:.4f} to {r.ratio:.4f}' for _, r in ashdrop.iterrows())}). "
        "The tail is therefore not what makes Omaha the exception.")
    add("patches whose GT is (nearly) all shadow, and the AOI they sit in",
        f"{int(c.loc['oma', 'n_patch_shadow_frac_gt_90'])} above 0.90, all in {ash_aoi[0]}",
        f"{int(c.loc['jax', 'n_patch_shadow_frac_gt_90'])}",
        f"{int(c.loc['ucsd', 'n_patch_shadow_frac_gt_90'])}",
        "task_c.probe_side.aoi_sensitivity_oma", "no",
        "Only Omaha has them and they enter p-bar_S and p-bar_B asymmetrically (no GT background pixels, "
        "so they are dropped from p-bar_B and kept in p-bar_S), and the same AOI is the one Omaha AOI "
        "with no defined p-bar_B patch at all. The AOI-drop sensitivity shows this does not drive the "
        "result.")
    add("single-AOI sensitivity of the Omaha ratio (leave-one-out range over 42 AOIs)",
        "; ".join(f"{r.arch} {r.rmin:.4f}-{r.rmax:.4f}" for _, r in jk.iterrows()),
        "not computed (Omaha only)", "not computed (Omaha only)",
        "task_c.probe_side.aoi_jackknife_oma_summary", "no",
        "No single Omaha AOI carries the result. Dropping the three AOIs that contribute most to "
        "Delta p-bar_B leaves every Omaha ratio below 0.88, and dropping the three that contribute most "
        "to Delta p-bar_S leaves them below 0.99. The one exception is OGLANet, whose ratio swings "
        f"{float(jk[jk.arch == 'oglanet'].rmin.iloc[0]):.4f} to "
        f"{float(jk[jk.arch == 'oglanet'].rmax.iloc[0]):.4f} under leave-one-out, because its numerator "
        "Delta p-bar_S is about 0.002 and disagrees in sign across seeds; that ratio is not a "
        "well-determined quantity.")
    add("DSM-unseen fraction (mean / count of fully-unseen patches)",
        f"{r6(c.loc['oma', 'unseen_frac_mean'])} / {int(c.loc['oma', 'n_patch_fully_unseen'])}",
        f"{r6(c.loc['jax', 'unseen_frac_mean'])} / {int(c.loc['jax', 'n_patch_fully_unseen'])}",
        f"{r6(c.loc['ucsd', 'unseen_frac_mean'])} / {int(c.loc['ucsd', 'n_patch_fully_unseen'])}",
        "task_c.city_properties", "no",
        "UCSD, not Omaha, is the extreme city on unseen pixels, and UCSD's p-bar ratios sit in the "
        "2.79x-6.05x band with the other non-Omaha pairs.")
    add("patch count / AOI count / patches per AOI",
        f"{int(c.loc['oma', 'n_test_patches'])} / {int(c.loc['oma', 'n_aoi'])} / "
        f"{r6(c.loc['oma', 'patches_per_aoi_mean'])}",
        f"{int(c.loc['jax', 'n_test_patches'])} / {int(c.loc['jax', 'n_aoi'])} / "
        f"{r6(c.loc['jax', 'patches_per_aoi_mean'])}",
        f"{int(c.loc['ucsd', 'n_test_patches'])} / {int(c.loc['ucsd', 'n_aoi'])} / "
        f"{r6(c.loc['ucsd', 'patches_per_aoi_mean'])}",
        "task_c.city_properties", "partly",
        "Omaha is the smallest fold on both counts, which widens its intervals: its 42 AOIs against "
        "JAX's 72 and UCSD's 52. That bears on the two non-significant p-bar_S pairs through the "
        "interval width, but their point estimates are also the two smallest in the grid "
        "(-0.0096 and +0.0022), so width alone does not account for them.")
    add("acquisitions / distinct sun elevations on the test patches",
        f"{int(c.loc['oma', 'n_acquisitions'])} / {int(c.loc['oma', 'n_distinct_sun_elevation'])}",
        f"{int(c.loc['jax', 'n_acquisitions'])} / {int(c.loc['jax', 'n_distinct_sun_elevation'])}",
        f"{int(c.loc['ucsd', 'n_acquisitions'])} / {int(c.loc['ucsd', 'n_distinct_sun_elevation'])}",
        "task_c.city_properties", "no",
        "Omaha sits between the other two on acquisitions and ties UCSD on distinct sun elevations; no "
        "ordering of either matches the probe behaviour.")
    add("share of patches with no GT foreground",
        r6(c.loc["oma", "share_patch_no_foreground"]), r6(c.loc["jax", "share_patch_no_foreground"]),
        r6(c.loc["ucsd", "share_patch_no_foreground"]),
        "task_c.city_properties", "no",
        "Omaha has the smallest share of the three, and those patches are dropped from p-bar_S in every "
        "city by the same rule.")
    add("vanilla 2-class gap range across the 4 architectures",
        f"{gaps[gaps.fold == 'oma'].gap_2class.min():.4f}-{gaps[gaps.fold == 'oma'].gap_2class.max():.4f}",
        f"{gaps[gaps.fold == 'jax'].gap_2class.min():.4f}-{gaps[gaps.fold == 'jax'].gap_2class.max():.4f}",
        f"{gaps[gaps.fold == 'ucsd'].gap_2class.min():.4f}-{gaps[gaps.fold == 'ucsd'].gap_2class.max():.4f}",
        "task_c.model_side.gaps", "no",
        "Omaha holds both the largest gap in the whole grid (CrossEarth 7.95) and the smallest (OGLANet "
        "2.10), so the small Delta p-bar_S is not a small-gap effect: the CrossEarth/Omaha pair has the "
        "grid's widest gap and a Delta p-bar_S of only -0.0277.")
    verdicts = pd.DataFrame(verdicts)
    print(verdicts[["property", "oma", "jax", "ucsd", "bears_on_probe"]].to_string(
        index=False, max_colwidth=46))
    print("\n  verdict counts: " + str(verdicts.bears_on_probe.value_counts().to_dict()))
    summary = (
        "Omaha's p-bar ratio is small because its Delta p-bar_S is small, not because its Delta p-bar_B "
        f"is large: mean |Delta p-bar_B| is "
        f"{float(byfold[byfold.fold == 'oma'].mean_abs_d_pbarB.iloc[0]):.4f} on Omaha against "
        f"{float(byfold[byfold.fold == 'ucsd'].mean_abs_d_pbarB.iloc[0]):.4f} on UCSD, while mean "
        f"|Delta p-bar_S| is {float(byfold[byfold.fold == 'oma'].mean_abs_d_pbarS.iloc[0]):.4f} against "
        f"{float(byfold[byfold.fold == 'ucsd'].mean_abs_d_pbarS.iloc[0]):.4f}. The small numerator sits "
        f"on the UPPER-BOUND side: the three folds' LOCO p-bar_S levels agree to within 0.03 "
        f"({float(byfold[byfold.fold == 'jax'].mean_loco_pbarS.iloc[0]):.4f} to "
        f"{float(byfold[byfold.fold == 'oma'].mean_loco_pbarS.iloc[0]):.4f}), while Omaha's upper-bound "
        f"p-bar_S is {float(byfold[byfold.fold == 'oma'].mean_upper_pbarS.iloc[0]):.4f} against "
        f"{float(byfold[byfold.fold == 'jax'].mean_upper_pbarS.iloc[0]):.4f} on JAX and "
        f"{float(byfold[byfold.fold == 'ucsd'].mean_upper_pbarS.iloc[0]):.4f} on UCSD. Omaha is also the "
        "only fold where the vanilla-LOCO model predicts more positive pixels than its own upper bound. "
        "Of the measured city properties, the two that order the way the probe behaviour orders are the "
        "mean sun_dist of the LOCO cells (Omaha smallest) and the test-side sun-elevation spread (Omaha "
        "narrowest); both are n = 3 correlates, and results/08 task 4 reports the within-city sun_dist "
        "slope as negative and not significant. The dataset properties that single Omaha out most "
        "sharply -- the all-shadow patches and the wide shadow-fraction tail -- do NOT drive the result: "
        "they sit in one AOI whose removal moves every Omaha ratio by at most "
        f"{float((ashdrop.ratio - ashdrop.ratio_full).abs().max()):.4f}.")
    print("\n  SUMMARY: " + summary)

    st["task_c"] = dict(
        city_properties=r6(city.reset_index().to_dict("records")),
        city_properties_note=("every quantity is a GT-side or acquisition-side property of the fold and "
                              "is identical in every cell of that fold; n_valid_px, n_gt_fg_valid, "
                              "image_id and the image->AOI map were asserted equal to the fold reference "
                              "on all 381 standard-schema dumps"),
        model_side=dict(
            gaps=r6(gaps.to_dict("records")),
            gap_range_by_fold={f: [r6(float(gaps[gaps.fold == f].gap_2class.min())),
                                   r6(float(gaps[gaps.fold == f].gap_2class.max()))] for f in FOLDS},
            predpos=r6(pp.to_dict("records")),
            all_shadow_patches=dict(
                n_by_fold={f: int(city.loc[f, "n_patch_all_shadow_flag"]) for f in FOLDS},
                oma_aoi_ids=ash_aoi,
                scored=r6(ash.to_dict("records")),
                note=("all_shadow_flag is a GT property of the patch, so the count is a fold property, "
                      "not a per-cell one; it is 11 on every OMA cell and 0 elsewhere"))),
        probe_side=dict(
            pairs=r6(pairs.to_dict("records")),
            by_fold=r6(byfold.to_dict("records")),
            crosscheck_vs_08_task5b=r6(xc5.to_dict("records")),
            max_abs_crosscheck_difference=r6(float(max(xc5.dS_absdiff.max(), xc5.dB_absdiff.max()))),
            aoi_sensitivity_oma=r6(sens.to_dict("records")),
            aoi_jackknife_oma_summary=r6(jack.groupby("arch").agg(
                n=("dropped", "size"), ratio_min=("ratio", "min"), ratio_max=("ratio", "max"),
                dS_min=("dS", "min"), dS_max=("dS", "max"), dB_min=("dB", "min"),
                dB_max=("dB", "max")).reset_index().to_dict("records")),
            aoi_jackknife_oma=r6(jack.to_dict("records")),
            sensitivity_note=("the pooled per-image estimate is exactly the AOI-count-weighted mean of "
                              "the per-AOI means, which was asserted before any AOI was dropped; "
                              "dropping an AOI therefore removes its patches from the pooled mean and "
                              "nothing else. The sign-flip test is not re-run on the reduced sets.")),
        differences=r6(verdicts.to_dict("records")),
        differences_verdict_counts={str(k): int(v) for k, v in
                                    verdicts.bears_on_probe.value_counts().items()},
        summary=summary)
    return st


# ============================================================================
# results JSON
# ============================================================================

def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(float(o)) else round(float(o), 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float):
        return None if not np.isfinite(o) else round(o, 6)
    if isinstance(o, (np.ndarray,)):
        return _clean(o.tolist())
    if isinstance(o, pd.Timestamp):
        return str(o)
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def write_results(st):
    A, Bt, C = st["task_a"], st["task_b"], st["task_c"]
    prov = st["_prov"]
    desc, traj, sar = st["_desc"], st["_traj"], st["_sar"]
    idx = st["_idx"]

    verd = pd.DataFrame(A["a4_verdicts"])
    n_margin = int((verd.verdict == "margin").sum())
    n_mixed = int((verd.verdict == "mixed").sum())
    n_fail = int((verd.verdict == "failure_to_train").sum())
    n_coll = int((verd.verdict == "collapse").sum())
    aud = pd.DataFrame(Bt["audit"])
    pairs = pd.DataFrame(C["probe_side"]["pairs"])
    oma = pairs[pairs.fold == "oma"]

    runs = sorted(set(desc.run_id.tolist()) | set(sar.run_id.tolist() if len(sar) else []))
    runs_in_08 = set(st["_R8"]["runs"])

    conditions, tests = [], []
    for _, r in verd.iterrows():
        conditions.append(dict(
            label=f"S-EO adaptation diagnostics / {r.arch} / {r.method}",
            metric="predicted_positive_fraction_pooled_K2_unseen_excluded",
            mean=r.median_predpos_frac, stderr=None, n=int(r.n_cells),
            ci95=None, ci_method="not applicable: a per-cell descriptive, median over cells",
            normalised_by_gt_shadow_rate=r.median_predpos_over_gt,
            n_cells_below_control_range=int(r.n_below_control_range),
            n_cells_near_zero=int(r.n_near_zero),
            zero_foreground_image_share_median=r.median_zero_fg_share,
            trajectory_rise_then_collapse_large=int(r.n_collapse_large),
            trajectory_rise_then_collapse_severe=int(r.n_collapse_severe),
            verdict=r.verdict,
            baseline="vanilla_loco_and_none_upper_r3500_same_fold_same_seed"))
    for _, r in pd.DataFrame(Bt["b1_metric_denominator"]["archfold"]).iterrows():
        conditions.append(dict(
            label=f"Exp A / {r.arch} / holdout {r.fold.upper()} / {r.comparison}",
            metric=("expa_minus_parent_" + r.comparison),
            mean=r["mean"], stderr=r.stderr, n=int(r.n_images), n_aoi=int(r.n_aoi),
            n_seeds=int(r.n_seeds), ci95=[r.ci_lo, r.ci_hi],
            ci_method="aoi_cluster_bootstrap_seed_stratum_B10000",
            per_seed=r.per_seed, seed_sd=r.seed_sd,
            target_access="target_labels (Exp A) vs source_only (parent)",
            baseline="vanilla_loco_r3500_same_fold_same_seed"))
        tests.append(dict(
            name="AOI-level sign-flip permutation test on the seed-averaged per-image delta",
            compares=[f"{r.arch}-expa-loco-r3500-{r.fold}-hr-s{{0,1,2}}",
                      f"{r.arch}-vanilla-loco-r3500-{r.fold}-hr-s{{0,1,2}}"],
            differencing=r.comparison, statistic=r["mean"], p=r.p_raw, p_holm=r.p_holm,
            B=B, two_tailed=True,
            family="9 (arch x fold) cells, Holm-Bonferroni within each differencing"))
    for _, r in pd.DataFrame(Bt["b9_vs_upper_bound"]).iterrows():
        conditions.append(dict(
            label=f"Exp A minus within-city upper bound / {r.arch} / holdout {r.fold.upper()}",
            metric="expa_minus_upper_fg_only_tol_k2", mean=r["mean"], stderr=r.stderr,
            n=int(r.n_images), n_aoi=int(r.n_aoi), n_seeds=int(r.n_seeds),
            ci95=[r.ci_lo, r.ci_hi], ci_method="aoi_cluster_bootstrap_seed_stratum_B10000",
            per_seed=r.per_seed, seed_sd=r.seed_sd,
            baseline="none_upper_r3500_same_fold_same_seed"))
        tests.append(dict(
            name="AOI-level sign-flip permutation test on the seed-averaged per-image delta",
            compares=[f"{r.arch}-expa-loco-r3500-{r.fold}-hr-s{{0,1,2}}",
                      f"{r.arch}-none-upper-r3500-{r.fold}-hr-s{{0,1,2}}"],
            statistic=r["mean"], p=r.p_raw, p_holm=r.p_holm, B=B, two_tailed=True,
            family="9 (arch x fold) cells, Holm-Bonferroni"))
    for key, v in Bt["b4_gap_recovery"]["regressions"].items():
        tests.append(dict(
            name="OLS slope of the Exp A delta on the parent's gap, over 27 seed-resolved cells",
            compares=[key], statistic=v["pooled"]["slope"], p=v["pooled"]["p_ols"],
            p_cluster_arch=v["pooled"]["p_cluster_arch"], n=v["pooled"]["n"], n_clusters=3,
            r2=v["pooled"]["r2"], two_tailed=True,
            family="3 regressions, reported uncorrected; see the note on collinearity"))
        tests.append(dict(
            name="permutation test on the gap slope with the Exp A delta permuted WITHIN architecture",
            compares=[key + " + C(arch)"], statistic=v["permutation_within_arch"]["statistic"],
            p=v["permutation_within_arch"]["p"], B=B, two_tailed=True,
            family="3 regressions, reported uncorrected"))
    for _, r in pairs.iterrows():
        for tag, name in (("S", "pbar_gt_shadow"), ("B", "pbar_gt_bg")):
            conditions.append(dict(
                label=f"p-bar_{tag} / {r.arch} / holdout {r.fold.upper()} / LOCO - upper",
                metric=f"delta_{name}", mean=r[f"d_pbar{tag}"], stderr=None,
                n=int(r[f"n_images_{tag}"]), n_aoi=int(r[f"n_aoi_{tag}"]), n_seeds=3,
                ci95=[r[f"d_pbar{tag}_ci_lo"], r[f"d_pbar{tag}_ci_hi"]],
                ci_method="aoi_cluster_bootstrap_seed_stratum_B10000",
                loco_mean=r[f"loco_pbar{tag}"], upper_mean=r[f"upper_pbar{tag}"],
                baseline="none_upper_r3500_same_fold_same_seed"))
            tests.append(dict(
                name="AOI-level sign-flip permutation test on the seed-averaged per-image delta",
                compares=[f"{r.arch}-vanilla-loco-r3500-{r.fold}-hr-s{{0,1,2}}",
                          f"{r.arch}-none-upper-r3500-{r.fold}-hr-s{{0,1,2}}"],
                metric=f"delta_{name}", statistic=r[f"d_pbar{tag}"], p=r[f"d_pbar{tag}_p_raw"],
                p_holm=r[f"d_pbar{tag}_p_holm"], B=B, two_tailed=True,
                family=f"12 (arch x fold) pairs, Holm-Bonferroni within p-bar_{tag}"))

    a3 = A["a3_trajectory"]
    headline = (
        f"On the S-EO grid, {A['a1_collapse_at_test_time']['n_cells_near_zero']} of "
        f"{A['universe']['n_method_cells']} adaptation-method cells and 0 of 27 Exp A cells predict "
        f"near-zero foreground, and "
        f"{A['a1_collapse_at_test_time']['n_cells_below_control_range']} method cells fall below the 72 "
        f"control cells' predicted-positive range; {n_margin} of {len(verd)} (architecture x method) "
        f"pairs lose to vanilla as a margin, {n_mixed} are mixed, {n_fail} (MAMNet with MRFP+) meets the "
        f"failure-to-train criterion on all nine of its cells and {n_coll} collapses at test time. The "
        f"Exp A architecture sign separation holds under both a matched foreground-only and a matched "
        f"two-class differencing (18 of 18 positive on MAMNet and DINOv3, 9 of 9 negative on OGLANet in "
        f"each), and of {len(aud)} enumerated alternatives "
        f"{int((aud.status == 'ruled_out').sum())} are ruled out here, "
        f"{int(aud.status.str.startswith('not_ruled_out').sum())} are not ruled out (the parent-gap "
        f"account predicts Exp A magnitude at R2 "
        f"{Bt['b4_gap_recovery']['regressions']['expa_delta_fg~parent_gap_fg']['pooled']['r2']:.4f} and "
        f"survives architecture fixed effects, but the sign still separates inside the DINOv3/OGLANet "
        f"parent-gap overlap) and {int((aud.status == 'not_checkable_here').sum())} are not checkable on "
        f"this box. Omaha's p-bar ratio is small because its Delta p-bar_S is small "
        f"({float(oma.d_pbarS.abs().mean()):.4f} mean absolute against "
        f"{float(pairs[pairs.fold != 'oma'].d_pbarS.abs().mean()):.4f} on the other two folds) while its "
        f"Delta p-bar_B is the same order as UCSD's, and the small numerator sits on the upper-bound "
        f"side, whose p-bar_S level is {float(oma.upper_pbarS.mean()):.4f} on Omaha against "
        f"{float(pairs[pairs.fold == 'ucsd'].upper_pbarS.mean()):.4f} on UCSD.")

    out = dict(
        id="09_seo_diagnostics", arm="A",
        answers=["AC1", "AC2", "AC3", "v1UJ-W1", "v1UJ-W2", "zNBu-W1", "zNBu-W2", "zNBu-W5", "BxRx-W1"],
        answers_by_task=dict(
            task_a=dict(ids=["v1UJ-W2", "AC3", "AC1", "AC2"],
                        why=("whether the nine adaptation methods lose to plain training by a margin or "
                             "fail to train changes what the method table in results/08 means")),
            task_b=dict(ids=["zNBu-W1", "zNBu-W5", "AC1", "AC2"],
                        why=("Exp A is the external ride-along for the section 4.4 locus claim; the audit "
                             "lists what an alternative explanation would have to be")),
            task_c=dict(ids=["zNBu-W2", "v1UJ-W1", "BxRx-W1", "AC1"],
                        why=("Omaha is the exception in two separate analyses; the question is whether "
                             "any measured city property accounts for it"))),
        question=("Are the S-EO adaptation methods losing to plain training by a margin or failing to "
                  "train? What alternative explanations for the Exp A architecture separation can this "
                  "box rule out? And does any measured property of Omaha account for its probe "
                  "behaviour?"),
        builds_on=["results/08_seo_full_grid.json"],
        supersedes=[],
        runs=runs,
        n_runs=len(runs),
        runs_not_in_08_runs=sorted(set(runs) - runs_in_08),
        index_path=INDEX_PATH, index_sha256=prov["index_sha"],
        results_08_sha256=prov["sha_08_results"],
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/09_seo_diagnostics.py",
        script_sha256=sha256_file(os.path.abspath(__file__)),
        lifted_from=dict(
            metrics_module=dict(path=M8.METRICS_PATH, sha256=prov["metrics_sha"],
                                note=("the two-class and foreground-only per-image formulas are 08's "
                                      "miou2/fgiou, which match metrics.py _img_metrics exactly and were "
                                      "asserted equal to the stored miou_tol_pct and iou_tol columns to "
                                      "1e-9 on every cell read")),
            script_08=dict(path="analysis/08_seo_full_grid.py", sha256=prov["sha_08_script"],
                           functions=["catalogue", "find", "miou2", "fgiou", "e9_band",
                                      "aoi_cluster_bootstrap_imagemean", "aoi_signflip_imagemean",
                                      "ci95", "holm", "stream_of"]),
            script_08b=dict(path="analysis/08b_seo_pbar.py", sha256=prov["sha_08b_script"],
                            note="imported for its module-level provenance wiring to 04")),
        git_hash=None,
        git_hash_note=("the project base is not a git repository on this box; per-cell producing commits "
                       "are in INDEX.parquet run_json_scalars"),
        interpreter="python, PYTHONNOUSERSITE=1",
        seed=SEED, B=B,
        no_branch_applied=("No pre-committed branch is applied and nothing is said to replicate or "
                           "validate anything. The readings are the operator's."),
        dry_run=dict(n_catalogue_entries=prov["n_catalogue"], n_in_scope=prov["n_in_scope"],
                     n_standard_schema=prov["n_std"], n_sar=prov["n_sar"],
                     resolved_schema=prov["dtype_ref"],
                     n_index_rows=int(len(idx)),
                     assertions=("per dump: the arm-A column set present with the reference dtypes "
                                 "(never coerced), row count == the fold's n_test, image_id unique and "
                                 "equal in order to the fold reference, image_id -> aoi_id map equal, "
                                 "n_valid_px and n_gt_fg_valid equal (GT-side columns), "
                                 "tp+fp+fn+tn == n_valid_px at K=2, the two-class and foreground-only "
                                 "metrics recomputed from the counts equal to the stored columns to "
                                 "1e-9, n_pred_pos == tp_tol + fp_tol, run_id column == directory, "
                                 "city == holdout, run.json and .DONE present")),
        task_a=A, task_b=Bt, task_c=C,
        conditions=conditions, tests=tests,
        sanity_flags=st["_flags"], headline=headline, caveats=st["_caveats"],
        figure=None, caption=None)

    txt = json.dumps(_clean(out), indent=1, sort_keys=False)
    tmp = RESULTS_PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(txt)
    with open(tmp) as fh:
        back = json.load(fh)
    if back["headline"] != headline:
        fail(".new does not read back")
    os.replace(tmp, RESULTS_PATH)
    print(f"\nwrote {RESULTS_PATH}")
    print(f"  sha256 {sha256_file(RESULTS_PATH)}")
    print(f"  {len(txt)} bytes, {len(out)} top-level keys, {len(conditions)} conditions, "
          f"{len(tests)} tests, {len(st['_caveats'])} caveats, {len(st['_flags'])} sanity flags")
    print(f"\nHEADLINE: {headline}")
    return out


# ============================================================================
# flags and caveats
# ============================================================================

def flags_and_caveats(st):
    A, Bt, C = st["task_a"], st["task_b"], st["task_c"]
    traj, desc = st["_traj"], st["_desc"]
    pairs = pd.DataFrame(C["probe_side"]["pairs"])
    verd = pd.DataFrame(A["a4_verdicts"])
    flags = []

    tctrl = traj[traj.is_control]
    hfam = {r["history_family"]: r for r in A["a3_trajectory"]["history_family_check"]}
    po, pl = hfam["pooled"], hfam["plain"]
    flags.append(
        f"task A3: the per-epoch history on the {po['n']} MAMNet/DINOv3/OGLANet cells is the POOLED "
        f"validation mIoU, and max(series) equals best_val_metric on {po['n_max_eq_best_val']} of "
        f"{po['n']} of them (argmax == best_epoch on {po['n_argmax_eq_best_epoch']}), so the trajectory "
        f"is read on a correlate of the quantity that selected the checkpoint, not on that quantity. On "
        f"the {pl['n']} CrossEarth cells the history IS the decision metric (argmax == best_epoch on "
        f"{pl['n_argmax_eq_best_epoch']} of {pl['n']} and max == best_val_metric on "
        f"{pl['n_max_eq_best_val']} of {pl['n']}). The two families are not pooled into one threshold "
        f"without this being stated.")
    flags.append(
        f"task A3: {A['a3_trajectory']['n_cells_history_unavailable']} standard-schema dumps carry no "
        f"per-epoch history at all (all of them Exp A, which has no validation split) and the 36 SAR "
        f"dumps do not train, and "
        f"{A['a3_trajectory']['n_cells_history_shorter_than_epochs_run']} cells carry a history shorter "
        f"than their own epochs_run because a resume link skips an epoch label. The trajectory half of "
        f"task A therefore cannot be run on Exp A or SAR at all.")
    # these frames have been through r6(), so compare at the same 6-dp rounding
    floor = round(1.0 / (B + 1.0), 6)
    n_floor = int((pairs.d_pbarS_p_raw <= floor + 1e-9).sum()
                  + (pairs.d_pbarB_p_raw <= floor + 1e-9).sum())
    af = pd.DataFrame(Bt["b1_metric_denominator"]["archfold"])
    n_floor_b = int((af.p_raw <= floor + 1e-9).sum())
    n_floor_b9 = int((pd.DataFrame(Bt["b9_vs_upper_bound"]).p_raw <= floor + 1e-9).sum())
    flags.append(
        f"task C: {n_floor} of 24 p-bar sign-flip p-values sit at the permutation floor 1/(B+1) = "
        f"{floor}, as in results/08 task5b. The Exp A sign-flip p-values do the same on {n_floor_b} of "
        f"{len(af)} (arch x fold x differencing) rows and on {n_floor_b9} of 9 Exp-A-versus-upper rows.")
    flags.append(
        "task B: the Exp A architecture sign separation is perfect at the cell level under both matched "
        "differencings (18 of 18 positive on MAMNet and DINOv3, 9 of 9 negative on OGLANet). Flagged as "
        "too clean, as results/08 already flagged it.")
    ogl = pd.DataFrame(Bt["b7_seed_effects"])
    ogl = ogl[ogl.arch == "oglanet"]
    flags.append(
        f"task B: OGLANet's Exp A across-seed SD is {float(ogl.seed_sd.min()):.4f} to "
        f"{float(ogl.seed_sd.max()):.4f} fg-only IoU points on cell means of "
        f"{float(ogl['mean'].min()):.4f} to {float(ogl['mean'].max()):.4f}, and OGLANet inference is the "
        f"nondeterministic one (E-9). No Exp A cell of any architecture has a repeat-inference band, so "
        f"every Exp A number is one draw.")
    mr = traj[(traj.arch == "mamnet") & (traj.method == "mrfp_plus")]
    flags.append(
        f"task A: MAMNet/MRFP+ peaks at epoch {int(mr.peak_epoch.min())}-{int(mr.peak_epoch.max())} on "
        f"all 9 cells and then loses {float(mr['drop_val'].min()):.2f} to {float(mr['drop_val'].max()):.2f} validation "
        f"mIoU points that it never regains, while its test-time predicted-positive fraction stays inside "
        f"the control range on 9 of 9 cells and reaches the grid maximum (0.1785) on one. A model can "
        f"therefore show the collapse trajectory without showing the collapse prediction signature; the "
        f"two halves of task A disagree on this pair and are reported separately rather than merged.")
    flags.append(
        f"task C: the 11 Omaha all-shadow patches have no GT background pixels, so they are dropped from "
        f"p-bar_B and kept in p-bar_S. One whole Omaha AOI has no defined p-bar_B patch and leaves the "
        f"p-bar_B AOI bootstrap in every Omaha pair (41 of 42 AOIs), as results/08 task5b flagged.")
    flags.append(
        "task A: SAR moves vanilla by a mean per-step change in predicted-positive fraction of order "
        "1e-4 over 10 steps, which is the same implausibly-low-variance pattern results/08 flagged on "
        "the SAR rows. Its dumps are on a different pixel population and are not pooled with anything.")
    b3 = pd.DataFrame(Bt["b3_convergence"]["by_arch"])
    flags.append(
        "task B3: the three architectures' mean Exp A printed best training loss orders exactly the "
        "reverse of their Exp A pooled delta ("
        + ", ".join(f"{r.arch} loss {r.train_loss_mean:.4f}" for _, r in b3.iterrows())
        + "). A perfect rank match over three points is one of six orderings by chance; it is reported "
          "as a correlate, not as a mechanism.")
    ex = pd.DataFrame(Bt["b2_training_instability"]["per_cell"])
    flags.append(
        f"task B2: OGLANet's Exp A cells do under-predict relative to the control range "
        f"({int(ex[ex.arch == 'oglanet'].below_control.sum())} of 9 below the control minimum, lowest "
        f"ratio {float(ex[ex.arch == 'oglanet'].predpos_over_gt_pooled.min()):.4f}) even though none is "
        f"near zero. The collapse hypothesis is ruled out; a mild under-prediction is not.")

    caveats = [
        "Pure reanalysis. No model was trained or run, and no dump was regenerated. Every number comes "
        "from the parquets and run.json sidecars already in delta_dumps/ plus results/08.",
        "This analysis adds no new cell. It reads the same 381 standard-schema dumps and 36 SAR dumps "
        "that results/08 catalogued, at the same INDEX.parquet sha "
        f"({st['_prov']['index_sha'][:12]}...), and reproduces 08's method deltas, Exp A deltas and "
        "p-bar deltas as a check on the lift, not as independent confirmation of them.",
        "Three seeds per cell. Every interval is an AOI cluster bootstrap with seed as a stratum, as in "
        "results/08; it covers test-patch and AOI sampling and seed variation only through the seed "
        "average. Per-seed point estimates and the across-seed SD (n = 3) sit beside every estimate.",
        "Task A's collapse threshold is the lowest normalised predicted-positive ratio any of the 72 "
        "control cells takes. It is a descriptive cut on a per-cell quantity, not a test; no interval or "
        "p-value is attached to 'below the control range'.",
        "Task A's trajectory thresholds are derived from the control cells' own largest peak-to-final "
        "drop and are reported at three levels (outside the control range, large, severe) because the "
        "drops span 1.0 to 15.3 validation mIoU points and a single cut would merge a cell that patience "
        "stopped 20 epochs after a shallow peak with one that lost 15 points.",
        "The two per-epoch history families are different aggregations of the validation metric (pooled "
        "micro on MAMNet/DINOv3/OGLANet, per-image decision metric on CrossEarth). Drop magnitudes are "
        "in validation mIoU points on each cell's own series and are comparable in units but not in "
        "aggregation.",
        "Task A reports no trajectory for Exp A or SAR: the published Exp A has no validation split and "
        "SAR does not train.",
        "SAR rows are not on the same pixel population as every other row (per-(pass, step) TTA schema, "
        "DSM-unseen pixels kept, no aoi_id or sun_*). They are summarised separately and labelled not "
        "commensurable, per the 2026-10-24 operator ruling recorded in results/08.",
        "Exp A cells train a fixed 30 epochs with no validation split and no model selection, discard "
        "the parent checkpoint's decoder (reinit_decoder) and use held-out-city labels "
        "(target_access target_labels). The delta therefore mixes frozen-encoder feature sufficiency "
        "with whether a from-scratch decoder can be trained in that budget.",
        "The Exp A audit's parent-gap alternative is not separable on this grid: parent gap and "
        "architecture are collinear over three architectures, and the within-architecture slope is the "
        "only test of it that this box can run.",
        "No Exp A cell has a repeat-inference band, so the OGLANet E-9 nondeterminism is unmeasured on "
        "exactly the cells the Exp A claim rests on. The bands reported are on OGLANet parent and method "
        "cells, on the two-class metric, and the 3-pass ones are undersampled as gates (D-B4-3).",
        "There is no published CrossEarth Experiment A, so the Exp A separation is over three "
        "architectures, not four.",
        "Task C's city properties are GT-side and acquisition-side quantities. They are identical in "
        "every cell of a fold and were asserted equal to the fold reference on all 381 standard-schema "
        "dumps, so they describe the dataset, not any model.",
        "Task C's AOI sensitivity drops AOIs from the pooled per-image mean and does not re-run the "
        "sign-flip test on the reduced sets; it reports how far the point estimate and the ratio move, "
        "not a p-value.",
        "In task C, city and fold are the same variable: there are three cities and three LOCO folds, so "
        "no city property can be tested against another at n > 3.",
        "The 108 fully-DSM-unseen patches (29 OMA, 0 JAX, 79 UCSD) have n_valid_px == 0. Their per-image "
        "predicted-positive fraction is undefined and they are excluded from the degeneracy denominators "
        "and counted separately, rather than counted as zero-foreground predictions.",
        "Budget asymmetry between the two sides of every gap, as in results/08: LOCO trains on 3,500 "
        "patches from the two source cities, the upper bound on 3,500 from the held-out city, and Exp A "
        "on 875 from the held-out city's upper train list.",
        "S-EO labels are auto-generated. Both sides of every paired delta use the same labels on the "
        "same patches; absolute levels carry the label noise.",
        "rung 3500 is the S-EO patch budget, not a point on the ShadowTransfer 225/450/900 ladder.",
        "28 trained cells stopped at an epoch ceiling rather than on patience (results/08 "
        "task6.epoch_ceiling_cells), and 3 of the 27 Exp A parents are among them. Only OGLANet has a "
        "from-scratch longer-horizon retrain (D-A2-1), and no Exp A cell was re-run from it.",
        "Holm families here: task B's Exp A differencings over the 9 (arch x fold) cells within each "
        "differencing, Exp A versus the upper bound over 9, and task C's p-bar deltas over the 12 "
        "(arch x fold) pairs within each of p-bar_S and p-bar_B. The three task-B regressions are "
        "reported uncorrected and are flagged as collinear by construction.",
        "The project base is not a git repository on this box; git_hash is null.",
    ]
    st["_flags"], st["_caveats"] = flags, caveats
    return st


# ============================================================================
if __name__ == "__main__":
    state = main()
    state = task_a(state)
    state = task_b(state)
    state = task_c(state)
    state = flags_and_caveats(state)
    print("\n" + "=" * 110)
    if DRY_RUN:
        print("DRY RUN -- every assertion passed and every task ran. NOTHING WRITTEN.")
        print(f"would write {RESULTS_PATH}")
        print("=" * 110)
        sys.exit(0)
    write_results(state)
    print("=" * 110)
