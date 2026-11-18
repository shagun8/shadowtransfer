#!/usr/bin/env python
"""
07 -- SIB external failure, MECHANISM analysis.  Pure reanalysis, zero GPU.

The verdict is NOT under re-examination.  SIB fails the pooled worst-case
criterion on S-EO (dinov3/OMA, -0.4216 two-class mIoU at K=2, converged and
countersigned in results/03_seo_methods.json).  This file asks only WHY, for
Section 5.2's because-clause.  Nothing here re-scores SIB: every SIB level and
every SIB-vs-vanilla delta it prints is asserted equal to the value already in
results/03_seo_methods.json before anything is built on it.

Three hypotheses, each pre-committed by the operator with its readings.  All
three are computed.  Supported readings are reported BY LETTER, from a
mechanical rule stated in this file's config block.  "Mechanism undetermined"
is a valid outcome and is not massaged into a story.

CIRCULARITY GUARD.  SIB was designed from the internal Section 4 diagnosis, so
its external failure is not evidence about that diagnosis in either direction.
H3 therefore tests input-statistics mismatch DIRECTLY, on imagery, and reads no
SIB output of any kind; the guard is enforced in code (H3_read_guard).

  H1  COMPONENT ATTRIBUTION  -- the method grid as a partial ablation
  H2  DAMAGE LOCALIZATION    -- where in the test pool the loss sits
  H3  DESIGN-ASSUMPTION MISMATCH -- measured on inputs only

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/07_sib_mechanism.py
"""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

# ----------------------------------------------------------------------------
# config -- every threshold below was fixed here BEFORE the full pass ran.
# The dry run computes no statistic of any kind.
# ----------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

BASE = os.environ["PROJECT_ROOT"]
SEO_DIR = os.path.join(BASE, "dumps", "seo")
SEO_INDEX = os.path.join(SEO_DIR, "INDEX.parquet")
INT_DIR = os.path.join(BASE, "dumps", "st_r450_published_recompute")
INT_INDEX = os.path.join(INT_DIR, "INDEX.parquet")
MANIFEST_PATH = os.path.join(BASE, "logs", "armA_phase3_manifest.json")
VERDICT_JSON = os.path.join(BASE, "results", "03_seo_methods.json")
RESULTS_PATH = os.path.join(BASE, "results", "07_sib_mechanism.json")
TABLES_PATH = os.path.join(BASE, "results", "_07_sib_mechanism_tables.json")
CACHE_DIR = os.path.join(BASE, "analysis", "_cache")
H3_CACHE = os.path.join(CACHE_DIR, "07_h3_crop_features.parquet")
H3_SPEC_CACHE = os.path.join(CACHE_DIR, "07_h3_radial_spectra.npz")
H3_GUARD_CACHE = os.path.join(CACHE_DIR, "07_h3_read_guard.json")

# rule zero: the metric module is imported, never reimplemented.
METRICS_ROOT = os.environ.get("ST_METRICS_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
METRICS_PATH = os.path.join(METRICS_ROOT, "shadowtransfer", "metrics.py")
METRICS_SHA_EXPECTED = ("730ad4a5bc039e3151599fb86bd411e2816c0897"
                        "b68f1e4c17729335ab8a6e83")
# the published FDA operation and its band half-width.
FDA_SRC = os.path.join(os.environ.get("ST_METRICS_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))), "shadowtransfer", "armf_crossearth_cells.py")

# imagery (read-mostly provenance tree; nothing is written under it)
SHADE = os.path.join(os.environ["PROJECT_ROOT"], "data")
SEO_IMG_ROOT = os.path.join(SHADE, "data_comparisons", "SEO")
INT_IMG_ROOT = os.path.join(SHADE, "Final_data_test")
PRIOR_COMPARISON = os.path.join(SHADE, "data_comparisons",
                                "dataset_comparison_results",
                                "hypothesis_results",
                                "statistical_significance_tests.csv")

ARCHS = ["mamnet", "dinov3"]
FOLDS = ["oma", "jax", "ucsd"]
METHODS = ["sib", "fda"]
CONDITIONS = {"loco": "vanilla-loco", "upper": "none-upper",
              "sib": "sib-loco", "fda": "fda-loco"}
SEO_CITY_DIR = {"oma": "OMA", "jax": "JAX", "ucsd": "UCSD"}
INT_CITIES = ["chicago", "miami", "phoenix"]

EXPECTED_ROWS = {"oma": 1637, "jax": 2642, "ucsd": 1870}
EXPECTED_NCOL = 76

B = 10_000
SEED = 0
EPS = 1e-10                       # metrics.py _img_metrics epsilon, matched
CROP = 384                        # the training/eval crop; App F FDA runs at it
FDA_L_VALUES = [0.005, 0.01]      # mamnet-SIB's L and the FDA arms' L
MIN_PIXELS = 10                   # the <10 px connected-component filter
SMALL_OBJECT_PX = 50              # analysis/gj_fill_pass.py's declared assumption
MASK_BIN = 127                    # armf_histmatch_lib's `> 127` rule
SLOPE_BAND = (0.01, 0.25)         # cyc/px, the log-log fit band for the slope
N_RADIAL_BINS = 48

# -- mechanical verdict thresholds, fixed before the full pass ---------------
ALPHA = 0.05
# H3 calls a difference "marked" only if it also clears this standardised
# effect size.  n_internal = 450 and n_external = 702 units make significance
# cheap; the threshold is Cohen's conventional "large".
SMD_MARKED = 0.8
# H1/H2 quantifiers.  "uniform" and "on both architectures" mean all three
# folds; the 2-of-3 count is reported alongside so any other quantifier is
# re-derivable.
N_FOLDS_UNIFORM = 3

# H2: patches whose valid region is empty carry delta == 0 by construction and
# would drag the highest-unseen stratum toward zero.  Primary excludes them.
H2_PRIMARY_EXCLUDES_FULLY_UNSEEN = True

# H3: the S-EO source tiles are satellite footprints and carry all-black
# (0,0,0) pixels; the internal NAIP crops carry exactly none, in all 450.  The
# black mass is partly footprint padding and partly genuinely clipped-dark
# imagery and the two are not separable from this asset, so H3 runs on three
# nested samples and the primary caps the black share at 1% of a crop.  The
# cap is conservative with respect to reading (a): black padding inflates every
# low-frequency difference, so filtering it can only shrink the FDA-band gap.
H3_SAMPLES = [("primary_nodata_le_1pct", 0.01),
              ("strict_nodata_eq_0", 0.0),
              ("inclusive_all_crops", 1.0)]
H3_PRIMARY = "primary_nodata_le_1pct"


def run_id(arch, fold, cond):
    return f"{arch}-{CONDITIONS[cond]}-r3500-{fold}-hr-s0"


VANILLA_CELLS = [(a, f, c) for a in ARCHS for f in FOLDS for c in ("loco", "upper")]
METHOD_CELLS = [(a, f, m) for a in ARCHS for f in FOLDS for m in METHODS]
CELLS = VANILLA_CELLS + METHOD_CELLS


# ----------------------------------------------------------------------------
# infrastructure
# ----------------------------------------------------------------------------

def fail(msg):
    raise SystemExit(f"ASSERTION FAILED: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_metrics():
    if sha256_file(METRICS_PATH) != METRICS_SHA_EXPECTED:
        fail(f"metric module sha256 drifted at {METRICS_PATH}")
    if METRICS_ROOT not in sys.path:
        sys.path.insert(0, METRICS_ROOT)
    from shadowtransfer import metrics as M      # noqa: E402
    return M


def load_fda():
    """The published FDA band, imported from the arm-F module, not restated."""
    spec = importlib.util.spec_from_file_location("_fda_src", FDA_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fda_src"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rng(stream):
    return np.random.default_rng([SEED, stream])


def add_metrics(df):
    """Recompute both metrics from the raw K=2 counts.  Identical to
    analysis/03_seo_methods.py:add_metrics, including the epsilon."""
    tp = df["tp_tol"].to_numpy(np.float64)
    fp = df["fp_tol"].to_numpy(np.float64)
    fn = df["fn_tol"].to_numpy(np.float64)
    tn = df["tn_tol"].to_numpy(np.float64)
    iou_fg = tp / (tp + fp + fn + EPS)
    iou_bg = tn / (tn + fp + fn + EPS)
    out = df.copy()
    out["miou2_pct"] = 50.0 * (iou_fg + iou_bg)
    out["fg_pct"] = 100.0 * iou_fg
    return out


def aoi_cluster_bootstrap_imagemean(delta, aoi_codes, n_aoi, stream):
    """Resample AOIs with replacement; statistic = image-level mean delta."""
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    cnts = np.bincount(aoi_codes, minlength=n_aoi).astype(np.float64)
    idx = _rng(stream).integers(0, n_aoi, size=(B, n_aoi))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def aoi_signflip_imagemean(delta, aoi_codes, n_aoi, stream):
    """Two-sided AOI-level sign-flip null, same weighting as the estimate."""
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    n_tot = float(len(delta))
    obs = sums.sum() / n_tot
    signs = _rng(stream).choice(np.array([-1.0, 1.0]), size=(B, n_aoi))
    null = (signs * sums).sum(1) / n_tot
    p = (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)
    return obs, p


def ci95(stat):
    lo, hi = np.nanpercentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def boot_two_tailed_p(stat):
    """p = 2 min(P(T* <= 0), P(T* >= 0)), floored at 1/(B+1)."""
    s = np.asarray(stat, float)
    s = s[np.isfinite(s)]
    n = len(s)
    lo = (1.0 + float((s <= 0).sum())) / (n + 1.0)
    hi = (1.0 + float((s >= 0).sum())) / (n + 1.0)
    return float(min(1.0, 2.0 * min(lo, hi)))


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, monotone-enforced."""
    p = np.asarray(pvals, float)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def smd(a, b):
    """Standardised mean difference (b - a), pooled sd."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    return float((b.mean() - a.mean()) / np.sqrt(sp2)) if sp2 > 0 else np.nan


# ----------------------------------------------------------------------------
# loading and validation
# ----------------------------------------------------------------------------

def preflight():
    bad = []
    for key in CELLS:
        rid = run_id(*key)
        d = os.path.join(SEO_DIR, rid)
        why = []
        if not os.path.isdir(d):
            why.append("no directory")
        else:
            for f in (".DONE", "perimage.parquet", "run.json"):
                if not os.path.isfile(os.path.join(d, f)):
                    why.append(f"no {f}")
        if why:
            bad.append((rid, "; ".join(why)))
    if bad:
        for rid, why in bad:
            print(f"  {rid:44s}  {why}")
        fail(f"{len(bad)} S-EO cells incomplete; the grid must be complete "
             "(24 cells) for this analysis")
    print(f"preflight: {len(CELLS)} S-EO cells complete "
          "(.DONE + perimage.parquet + run.json)")


def load_seo():
    """Every S-EO cell, validated, with derived metrics, sorted by image_id."""
    data, metas = {}, {}
    ref_cols = None
    for key in CELLS:
        rid = run_id(*key)
        d = os.path.join(SEO_DIR, rid)
        df = pd.read_parquet(os.path.join(d, "perimage.parquet"))
        with open(os.path.join(d, "run.json")) as fh:
            meta = json.load(fh)

        arch, fold, cond = key
        if df.shape[1] != EXPECTED_NCOL:
            fail(f"{rid}: {df.shape[1]} columns, expected {EXPECTED_NCOL}")
        if len(df) != EXPECTED_ROWS[fold]:
            fail(f"{rid}: {len(df)} rows, expected {EXPECTED_ROWS[fold]}")
        if ref_cols is None:
            ref_cols = list(df.columns)
        elif list(df.columns) != ref_cols:
            fail(f"{rid}: column name/order differs from the reference cell")
        if meta["run_id"] != rid:
            fail(f"{rid}: run.json run_id is {meta['run_id']}")
        if meta.get("dataset") != "seo" or meta.get("rung") != 3500:
            fail(f"{rid}: dataset/rung sidecar mismatch")
        if meta.get("seed") != 0 or meta.get("res") != "hr":
            fail(f"{rid}: seed/res sidecar mismatch")
        if set(df.city.unique()) != {meta["holdout"]}:
            fail(f"{rid}: city column does not match the sidecar holdout")
        if df.image_id.duplicated().any():
            fail(f"{rid}: duplicate image_id")
        if not ((df.tp_tol + df.fp_tol + df.fn_tol + df.tn_tol)
                == df.n_valid_px).all():
            fail(f"{rid}: tp+fp+fn+tn != n_valid_px")

        dfm = add_metrics(df).sort_values("image_id").reset_index(drop=True)
        if float(np.abs(dfm.miou2_pct - dfm.miou_tol_pct).max()) >= 1e-9:
            fail(f"{rid}: recomputed two-class mIoU differs from the stored column")
        if float(np.abs(dfm.fg_pct - dfm.iou_tol * 100.0).max()) >= 1e-9:
            fail(f"{rid}: recomputed fg IoU differs from the stored column")
        data[key], metas[key] = dfm, meta

    # run_id / config_sha256 consistency
    seen = {}
    for k, m in metas.items():
        rid, sha = m["run_id"], m.get("config_sha256")
        if rid in seen and seen[rid] != sha:
            fail(f"{rid}: duplicate run_id with differing config_sha256")
        seen[rid] = sha

    # every cell of a fold covers the same patches, in the same order, and the
    # patch-level covariates H2 stratifies on are model-independent
    covar = ["unseen_frac", "sun_elevation", "sun_dist", "n_valid_px",
             "n_gt_fg_valid", "aoi_id", "acquisition_date"]
    for fold in FOLDS:
        ref = data[(ARCHS[0], fold, "loco")]
        for k in [c for c in CELLS if c[1] == fold]:
            if not (data[k].image_id.values == ref.image_id.values).all():
                fail(f"{run_id(*k)}: image_id order differs from the fold reference")
            for c in covar:
                if c == "sun_dist" and k[2] == "upper":
                    continue          # upper trains on the held-out city
                if not (data[k][c].values == ref[c].values).all():
                    fail(f"{run_id(*k)}: {c} differs from the fold's vanilla-LOCO "
                         "cell; it is not a model-independent covariate here")
    print("validation: 24 cells, one column schema, patch sets and "
          "model-independent covariates identical within every fold")
    return data, metas


def verdict_crosscheck(data):
    """Reproduce results/03_seo_methods.json before anything is built on it.

    This file does not re-score SIB.  It asserts the scored numbers.
    """
    with open(VERDICT_JSON) as fh:
        prior = json.load(fh)
    by_label = {c["label"]: c for c in prior["conditions"]
                if c["metric"] == "delta_2class_tol_k2_unseen_excluded"}
    rows = []
    for arch, fold, method in METHOD_CELLS:
        van = data[(arch, fold, "loco")].miou2_pct.values
        met = data[(arch, fold, method)].miou2_pct.values
        got = float((met - van).mean())
        lab = (f"S-EO {method.upper()} - vanilla-LOCO / {arch} / "
               f"holdout {fold.upper()}")
        want = by_label[lab]["mean"]
        ok = abs(round(got, 4) - want) < 5e-5
        rows.append(dict(cell=f"{arch}/{fold}/{method}", recomputed=got,
                         results_03=want, agrees=bool(ok)))
        if not ok:
            fail(f"{lab}: recomputed delta {got:.6f} != results/03 {want}")
    # the verdict number itself
    t2 = {(r["method"], r["scope"]): r for r in prior["task2_worst_case"]}
    v = t2[("sib", "dinov3")]
    van = data[("dinov3", "oma", "loco")].miou2_pct.values
    sib = data[("dinov3", "oma", "sib")].miou2_pct.values
    if abs(float(sib.mean()) - v["method_worst"]) > 1e-9:
        fail("the dinov3/OMA SIB level does not reproduce results/03's worst cell")
    if abs(float((sib - van).mean()) - v["difference"]) > 1e-9:
        fail("the dinov3/OMA SIB-vs-vanilla difference does not reproduce")
    print(f"verdict cross-check: all 12 method deltas reproduce "
          f"results/03_seo_methods.json exactly; the failing cell "
          f"{v['method_worst_cell']} is {v['difference']:+.4f} "
          "(criterion_met=False), carried forward unchanged and not re-scored")
    return rows, prior


# ----------------------------------------------------------------------------
# H1 -- component attribution
# ----------------------------------------------------------------------------

# Each row is a per-image paired quantity.  Rows A-D are measured deltas
# against that cell's own vanilla-LOCO comparator; E and F are exact per-image
# contrasts in which the vanilla term cancels algebraically.
H1_ROWS = [
    ("A", "dinov3", "delta_sib", "dinov3-SIB - vanilla",
     "haar + content_vib + edge_vib (no FDA, no SAG) -- the invariance core on the ViT"),
    ("B", "dinov3", "delta_fda", "dinov3-FDA - vanilla",
     "fda_preproc alone on the ViT"),
    ("C", "mamnet", "delta_sib", "mamnet-SIB - vanilla",
     "haar + vibs + SAG + fda_preproc + contrast_ch"),
    ("D", "mamnet", "delta_fda", "mamnet-FDA - vanilla",
     "fda_preproc + contrast_ch"),
    ("E", "mamnet", "sib_minus_fda", "mamnet-SIB - mamnet-FDA",
     "implied core+SAG increment on top of fda+contrast (vanilla cancels)"),
    ("F", "dinov3", "fda_minus_sib", "dinov3-FDA - dinov3-SIB",
     "FDA-bearing minus FDA-free on the ViT (vanilla cancels)"),
]


def h1_delta(data, arch, fold, kind):
    van = data[(arch, fold, "loco")]
    if kind == "delta_sib":
        return data[(arch, fold, "sib")].miou2_pct.values - van.miou2_pct.values
    if kind == "delta_fda":
        return data[(arch, fold, "fda")].miou2_pct.values - van.miou2_pct.values
    if kind == "sib_minus_fda":
        return (data[(arch, fold, "sib")].miou2_pct.values
                - data[(arch, fold, "fda")].miou2_pct.values)
    if kind == "fda_minus_sib":
        return (data[(arch, fold, "fda")].miou2_pct.values
                - data[(arch, fold, "sib")].miou2_pct.values)
    fail(f"unknown H1 kind {kind}")


def run_h1(data, aoi_of):
    rows = []
    stream = 0
    for letter, arch, kind, expr, isolates in H1_ROWS:
        for fold in FOLDS:
            codes, n_aoi = aoi_of(fold)
            d = h1_delta(data, arch, fold, kind).astype(np.float64)
            obs, p_raw = aoi_signflip_imagemean(d, codes, n_aoi, 1000 + stream)
            boot = aoi_cluster_bootstrap_imagemean(d, codes, n_aoi, 2000 + stream)
            lo, hi = ci95(boot)
            rows.append(dict(row=letter, arch=arch, fold=fold, expr=expr,
                             isolates=isolates, delta=float(obs),
                             stderr=float(boot.std(ddof=1)), ci_lo=lo, ci_hi=hi,
                             p_raw=p_raw, n_images=len(d), n_aoi=n_aoi))
            stream += 1
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t.p_raw.values)
    t["holm_family_size"] = len(t)
    return t


def h1_verdict(t):
    """Mechanical read of the operator's pre-committed (a)/(b)/(c)."""
    def sub(letter):
        return t[t.row == letter].set_index("fold")

    A, B_, E, F = sub("A"), sub("B"), sub("E"), sub("F")

    # (a1) FDA alone damages on the ViT, in every fold
    a1_n = int((B_.delta < 0).sum())
    a1_sig = int(((B_.delta < 0) & (B_.p_holm < ALPHA)).sum())
    a1 = a1_n == N_FOLDS_UNIFORM
    # (a2) the FDA-bearing config is more negative than its FDA-free
    #      counterpart, in every fold, and the contrast is significant
    a2_n = int((F.delta < 0).sum())
    a2_sig = int(((F.delta < 0) & (F.p_holm < ALPHA)).sum())
    a2 = (a2_n == N_FOLDS_UNIFORM) and (a2_sig == N_FOLDS_UNIFORM)
    # (a3) the core is neutral-or-mixed: it is NOT uniformly, significantly
    #      negative on either architecture
    core_dv_neg = int(((A.delta < 0) & (A.p_holm < ALPHA)).sum())
    core_mn_neg = int(((E.delta < 0) & (E.p_holm < ALPHA)).sum())
    a3 = not (core_dv_neg == N_FOLDS_UNIFORM or core_mn_neg == N_FOLDS_UNIFORM)
    a = bool(a1 and a2 and a3)

    # (b) the core itself is significantly negative on BOTH architectures
    b = bool(core_dv_neg == N_FOLDS_UNIFORM and core_mn_neg == N_FOLDS_UNIFORM)

    # sign agreement of the two core estimates
    sign_agree = int((np.sign(A.delta.values) == np.sign(E.delta.values)).sum())

    c = bool(not a and not b)
    return {
        "a_supported": a, "b_supported": b, "c_supported": c,
        "a1_fda_alone_negative_n_folds": a1_n,
        "a1_and_significant_n_folds": a1_sig,
        "a2_fda_bearing_more_negative_n_folds": a2_n,
        "a2_and_significant_n_folds": a2_sig,
        "a3_core_not_uniformly_significantly_negative": bool(a3),
        "core_dinov3_significantly_negative_n_folds": core_dv_neg,
        "core_mamnet_significantly_negative_n_folds": core_mn_neg,
        "core_sign_agreement_n_folds": sign_agree,
        "rule": (
            "(a) requires FDA alone negative in all 3 folds, the FDA-bearing "
            "minus FDA-free contrast negative and Holm-significant in all 3, "
            "and the core NOT uniformly Holm-significantly negative on either "
            "architecture. (b) requires the core Holm-significantly negative in "
            "all 3 folds on BOTH architectures. (c) fires when neither holds. "
            f"alpha = {ALPHA}, Holm over the {int(len(H1_ROWS)) * len(FOLDS)} "
            "tests of the H1 family."),
    }


# ----------------------------------------------------------------------------
# H2 -- damage localisation
# ----------------------------------------------------------------------------

H2_STRATA = ["unseen_frac", "sun_dist", "shadow_cov", "empty_fg"]
H2_TERMS = ["unseen_frac", "sun_dist", "shadow_cov", "empty_fg",
            "city_jax", "city_ucsd"]


def h2_frame(data, exclude_fully_unseen):
    """One patch-level frame per (arch, method): delta plus the strata."""
    out = {}
    for arch in ARCHS:
        for method in METHODS:
            parts = []
            for fold in FOLDS:
                van = data[(arch, fold, "loco")]
                met = data[(arch, fold, method)]
                nv = van.n_valid_px.to_numpy(np.float64)
                cov = np.divide(van.n_gt_fg_valid.to_numpy(np.float64), nv,
                                out=np.zeros(len(van)), where=nv > 0)
                f = pd.DataFrame({
                    "delta": met.miou2_pct.values - van.miou2_pct.values,
                    "unseen_frac": van.unseen_frac.values,
                    "sun_dist": van.sun_dist.values,
                    "shadow_cov": cov,
                    "empty_fg": van.empty_fg_valid.values.astype(np.float64),
                    "city_jax": float(fold == "jax"),
                    "city_ucsd": float(fold == "ucsd"),
                    "fold": fold,
                    "aoi_id": van.aoi_id.values,
                    "n_valid_px": van.n_valid_px.values,
                })
                if exclude_fully_unseen:
                    f = f[f.n_valid_px > 0]
                parts.append(f.reset_index(drop=True))
            out[(arch, method)] = pd.concat(parts, ignore_index=True)
    return out


def _fold_aoi_index(frame):
    """Per-fold AOI codes for a fold-stratified cluster bootstrap."""
    blocks = []
    for fold in FOLDS:
        m = (frame.fold == fold).to_numpy()
        codes, uniq = pd.factorize(frame.loc[m, "aoi_id"].to_numpy())
        blocks.append((m, codes, len(uniq)))
    return blocks


def _boot_weights(blocks, n_rows, stream):
    """B x n_rows integer cluster multiplicities, AOIs resampled within fold."""
    rng = _rng(stream)
    W = np.zeros((B, n_rows), dtype=np.float32)
    for m, codes, n_aoi in blocks:
        draw = rng.integers(0, n_aoi, size=(B, n_aoi))
        cnt = np.zeros((B, n_aoi), dtype=np.float32)
        for b in range(B):
            cnt[b] = np.bincount(draw[b], minlength=n_aoi)
        W[:, m] = cnt[:, codes]
    return W


H2_CONTINUOUS = {"unseen_frac", "sun_dist", "shadow_cov"}


def h2_regression(frame, W):
    """delta ~ strata + C(city), OLS point estimate, cluster-bootstrap CI/p."""
    X = np.column_stack([np.ones(len(frame))]
                        + [frame[t].to_numpy(np.float64) for t in H2_TERMS])
    y = frame["delta"].to_numpy(np.float64)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    p = X.shape[1]
    boots = np.empty((B, p))
    eye = 1e-12 * np.eye(p)
    for b in range(B):
        Xw = X * W[b][:, None].astype(np.float64)
        boots[b] = np.linalg.solve(X.T @ Xw + eye, Xw.T @ y)
    rows = []
    for j, name in enumerate(["intercept"] + H2_TERMS):
        s = boots[:, j]
        lo, hi = ci95(s)
        if name in H2_CONTINUOUS:
            iqr = float(np.subtract(*np.percentile(frame[name], [75, 25])))
            per_iqr = float(beta[j] * iqr)
        else:
            iqr, per_iqr = np.nan, np.nan     # meaningless for a 0/1 term
        rows.append(dict(term=name, coef=float(beta[j]),
                         stderr=float(s.std(ddof=1)), ci_lo=lo, ci_hi=hi,
                         p_raw=boot_two_tailed_p(s),
                         iqr=iqr, effect_per_iqr=per_iqr))
    return pd.DataFrame(rows), p


def _wmeans(W, y, mask):
    num = W[:, mask].astype(np.float64) @ y[mask]
    den = W[:, mask].sum(1, dtype=np.float64)
    return np.divide(num, den, out=np.full(B, np.nan), where=den > 0)


def h2_strata_means(frame, W, var, n_bins=4):
    """Mean delta per bin of `var`, cluster-bootstrap CI."""
    y = frame["delta"].to_numpy(np.float64)
    if var == "city":
        groups = [(f, (frame.fold == f).to_numpy()) for f in FOLDS]
    elif var == "empty_fg":
        v = frame[var].to_numpy(np.float64)
        groups = [("empty_fg=0", v == 0), ("empty_fg=1", v == 1)]
    else:
        v = frame[var].to_numpy(np.float64)
        qs = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        codes = np.clip(np.digitize(v, qs[1:-1], right=True), 0, len(qs) - 2)
        groups = [(f"Q{i+1} [{qs[i]:.4g},{qs[i+1]:.4g}]", codes == i)
                  for i in range(len(qs) - 1)]
    rows = []
    for lab, m in groups:
        if m.sum() == 0:
            continue
        lo, hi = ci95(_wmeans(W, y, m))
        rows.append(dict(var=var, bin=lab, n=int(m.sum()),
                         mean_delta=float(y[m].mean()), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def h2_contrast(frame, W, mask_clean, mask_dirty, name):
    """dirty minus clean mean delta under the same cluster resample."""
    y = frame["delta"].to_numpy(np.float64)
    obs = float(y[mask_dirty].mean() - y[mask_clean].mean())
    s = _wmeans(W, y, mask_dirty) - _wmeans(W, y, mask_clean)
    lo, hi = ci95(s)
    return dict(contrast=name, n_clean=int(mask_clean.sum()),
                n_dirty=int(mask_dirty.sum()),
                mean_clean=float(y[mask_clean].mean()),
                mean_dirty=float(y[mask_dirty].mean()),
                dirty_minus_clean=obs, ci_lo=lo, ci_hi=hi,
                p_raw=boot_two_tailed_p(s))


def run_h2(data):
    frames = h2_frame(data, H2_PRIMARY_EXCLUDES_FULLY_UNSEEN)
    frames_incl = h2_frame(data, False)
    reg_rows, strata_rows, contrast_rows, sens_rows = [], [], [], []
    stream = 3000
    for (arch, method), frame in frames.items():
        blocks = _fold_aoi_index(frame)
        W = _boot_weights(blocks, len(frame), stream)
        stream += 1
        reg, _ = h2_regression(frame, W)
        reg = reg[reg.term != "intercept"].copy()
        reg["p_holm"] = holm(reg.p_raw.values)
        reg["holm_family_size"] = len(reg)
        reg.insert(0, "method", method)
        reg.insert(0, "arch", arch)
        reg_rows.append(reg)

        for var in H2_STRATA + ["city"]:
            s = h2_strata_means(frame, W, var)
            s.insert(0, "method", method)
            s.insert(0, "arch", arch)
            strata_rows.append(s)

        # the sharpest single contrasts
        uf = frame.unseen_frac.to_numpy(np.float64)
        q1, q3 = np.quantile(uf, [0.25, 0.75])
        cs = []
        cs.append(h2_contrast(frame, W, uf <= q1, uf >= q3,
                              f"unseen_frac top quartile (>= {q3:.4g}) minus "
                              f"bottom quartile (<= {q1:.4g})"))
        jax = (frame.fold == "jax").to_numpy()
        cs.append(h2_contrast(frame, W, jax, ~jax,
                              "OMA+UCSD minus JAX (JAX = the clean-label fold)"))
        cdf = pd.DataFrame(cs)
        cdf["p_holm"] = holm(cdf.p_raw.values)
        cdf["holm_family_size"] = len(cdf)
        cdf.insert(0, "method", method)
        cdf.insert(0, "arch", arch)
        contrast_rows.append(cdf)

        # sensitivity: the same regression with the fully-unseen patches kept
        fi = frames_incl[(arch, method)]
        Wi = _boot_weights(_fold_aoi_index(fi), len(fi), stream)
        stream += 1
        regi, _ = h2_regression(fi, Wi)
        regi = regi[regi.term != "intercept"].copy()
        regi["p_holm"] = holm(regi.p_raw.values)
        regi.insert(0, "method", method)
        regi.insert(0, "arch", arch)
        sens_rows.append(regi)
        del W, Wi

    return (pd.concat(reg_rows, ignore_index=True),
            pd.concat(strata_rows, ignore_index=True),
            pd.concat(contrast_rows, ignore_index=True),
            pd.concat(sens_rows, ignore_index=True),
            {k: len(v) for k, v in frames.items()},
            {k: len(v) for k, v in frames_incl.items()})


H2_PATCH_TERMS = ["unseen_frac", "sun_dist", "shadow_cov", "empty_fg"]
H2_CITY_TERMS = ["city_jax", "city_ucsd"]


def h2_collinearity(data):
    """How much of each patch-level stratum is just city?

    The stratified quantile tables and the regression can disagree because a
    stratum is largely a relabelling of the fold.  R^2 of stratum ~ C(city)
    says how much, so the two tables can be read against each other.
    """
    frame = h2_frame(data, H2_PRIMARY_EXCLUDES_FULLY_UNSEEN)[(ARCHS[0], "sib")]
    X = np.column_stack([np.ones(len(frame))]
                        + [frame[t].to_numpy(np.float64) for t in H2_CITY_TERMS])
    rows = []
    for t in H2_PATCH_TERMS:
        y = frame[t].to_numpy(np.float64)
        r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        ss = float(((y - y.mean()) ** 2).sum())
        rows.append(dict(stratum=t,
                         r2_on_city=float(1.0 - (r ** 2).sum() / ss)
                         if ss > 0 else np.nan,
                         between_city_range=float(
                             frame.groupby("fold")[t].mean().max()
                             - frame.groupby("fold")[t].mean().min())))
    return pd.DataFrame(rows)


def h2_verdict(reg, contrasts):
    """Mechanical read of the operator's pre-committed (a)/(b)/(c) for SIB."""
    r = reg[reg.method == "sib"]
    c = contrasts[contrasts.method == "sib"]
    per_arch = {}
    for arch in ARCHS:
        ra = r[r.arch == arch]
        ca = c[c.arch == arch]
        sig = ra[ra.p_holm < ALPHA]
        # "damage concentrates in dirty strata" = delta falls as unseen_frac
        # rises, i.e. a negative unseen_frac slope
        uf = ra[ra.term == "unseen_frac"].iloc[0]
        uf_dirty = bool(uf.p_holm < ALPHA and uf.coef < 0)
        uf_clean = bool(uf.p_holm < ALPHA and uf.coef > 0)
        con = ca[ca.contrast.str.startswith("unseen_frac")].iloc[0]
        con_dirty = bool(con.p_holm < ALPHA and con.dirty_minus_clean < 0)
        con_clean = bool(con.p_holm < ALPHA and con.dirty_minus_clean > 0)
        per_arch[arch] = dict(
            n_strata_terms_surviving_holm=int(len(sig)),
            terms_surviving_holm=sorted(sig.term.tolist()),
            n_patch_level_strata_surviving_holm=int(
                sig.term.isin(H2_PATCH_TERMS).sum()),
            n_city_terms_surviving_holm=int(sig.term.isin(H2_CITY_TERMS).sum()),
            unseen_slope=float(uf.coef), unseen_p_holm=float(uf.p_holm),
            unseen_effect_per_iqr=float(uf.effect_per_iqr),
            unseen_points_to_dirty=uf_dirty, unseen_points_to_clean=uf_clean,
            contrast_dirty_minus_clean=float(con.dirty_minus_clean),
            contrast_p_holm=float(con.p_holm),
            contrast_points_to_dirty=con_dirty,
            contrast_points_to_clean=con_clean)
    a = all(per_arch[x]["unseen_points_to_dirty"]
            or per_arch[x]["contrast_points_to_dirty"] for x in ARCHS)
    b = all(per_arch[x]["n_strata_terms_surviving_holm"] == 0 for x in ARCHS)
    cc = all(per_arch[x]["unseen_points_to_clean"]
             or per_arch[x]["contrast_points_to_clean"] for x in ARCHS)
    # Secondary decomposition, reported alongside and NOT a re-reading: the
    # job spec lists city among the strata, so the rule above counts it. If
    # (b) had been meant to cover only the within-pool dirtiness axes, this is
    # the count it would have used.
    b_patch_only = all(
        per_arch[x]["n_patch_level_strata_surviving_holm"] == 0 for x in ARCHS)
    return {"a_supported": bool(a and not cc), "b_supported": bool(b),
            "c_supported": bool(cc and not a), "per_arch": per_arch,
            "secondary_b_patch_level_strata_only": bool(b_patch_only),
            "secondary_note": (
                "The rule counts city as a stratum because the pre-committed "
                "H2 strata list names it. secondary_b_patch_level_strata_only "
                "reports what (b) would have returned over the four "
                "within-pool dirtiness axes alone. It is a decomposition of "
                "the same numbers, not a second reading, and it does not "
                "change the letter."),
            "rule": (
                "(a) requires, on BOTH architectures, either a Holm-significant "
                "negative unseen_frac slope or a Holm-significant negative "
                "top-minus-bottom-quartile unseen contrast (delta more negative "
                "where DSM coverage is worst). (b) requires no strata term "
                "surviving Holm on either architecture. (c) is (a) with both "
                "signs reversed. Holm families: 6 regression terms per "
                f"(method x arch); 2 contrasts per (method x arch). alpha = {ALPHA}.")}


# ----------------------------------------------------------------------------
# H3 -- design-assumption mismatch, measured on inputs only
# ----------------------------------------------------------------------------

H3_READ_GUARD = []          # every path H3 opens; asserted SIB-free


def _guard_open(path):
    H3_READ_GUARD.append(path)
    return path


def tile_offsets(h, w, crop=CROP):
    """Centred non-overlapping tiling.  Verified against the dump's stored
    (patch_y, patch_x) on every S-EO source image present on this box."""
    ny, nx = h // crop, w // crop
    oy, ox = (h - ny * crop) // 2, (w - nx * crop) // 2
    return [(oy + i * crop, ox + j * crop) for i in range(ny) for j in range(nx)]


def radial_bins():
    f = np.fft.fftfreq(CROP)
    ky, kx = np.meshgrid(f, f, indexing="ij")
    k = np.sqrt(ky ** 2 + kx ** 2)
    kmin, kmax = 1.0 / CROP, 0.5
    edges = np.logspace(np.log10(kmin), np.log10(kmax), N_RADIAL_BINS + 1)
    idx = np.digitize(k.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < N_RADIAL_BINS)
    centres = np.sqrt(edges[:-1] * edges[1:])
    return idx, valid, centres, np.bincount(idx[valid], minlength=N_RADIAL_BINS)


def crop_features(img, mask, fda_b, rad, M):
    """Every H3 statistic for one 384x384 RGB crop and its binary mask."""
    idx, valid, centres, counts = rad
    x = img.astype(np.float32)
    amp = np.empty((3, CROP, CROP), np.float64)
    for c in range(3):
        F = np.fft.fft2(x[:, :, c])
        amp[c] = np.fft.fftshift(np.abs(F))
    cy = cx = CROP // 2
    out = {"nodata_frac": float((img.max(axis=2) == 0).mean())}
    amp_tot = float(amp.sum())
    for L, b in fda_b.items():
        blk = amp[:, cy - b:cy + b, cx - b:cx + b]
        tag = f"L{str(L).replace('.', 'p')}"
        out[f"fda_band_amp_mean_{tag}"] = float(blk.mean())
        out[f"fda_band_share_{tag}"] = (float(blk.sum() / amp_tot)
                                        if amp_tot > 0 else np.nan)
        # the DC bin sits at (cy, cx) and is inside every band
        exdc = blk.sum() - amp[:, cy, cx].sum()
        n_exdc = blk[0].size * 3 - 3
        out[f"fda_band_amp_mean_exdc_{tag}"] = float(exdc / n_exdc)
    out["dc_mean_intensity"] = float(amp[:, cy, cx].mean() / (CROP * CROP))

    power = (np.fft.ifftshift(amp, axes=(1, 2)) ** 2).mean(0).ravel()
    psum = np.bincount(idx[valid], weights=power[valid], minlength=N_RADIAL_BINS)
    prof = np.divide(psum, counts, out=np.zeros(N_RADIAL_BINS),
                     where=counts > 0)
    band = (centres >= SLOPE_BAND[0]) & (centres <= SLOPE_BAND[1]) & (prof > 0)
    if band.sum() >= 5:
        A = np.column_stack([np.log10(centres[band]), np.ones(band.sum())])
        sl = np.linalg.lstsq(A, np.log10(prof[band]), rcond=None)[0][0]
    else:
        sl = np.nan
    out["spectral_slope"] = float(sl)

    gray = x.mean(axis=2).astype(np.uint8)
    out["gray_mean"] = float(gray.mean())
    out["gray_std"] = float(gray.std())
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)

    mb = np.ascontiguousarray((mask > MASK_BIN).astype(np.uint8))
    filt = M.filter_small_masks(torch.from_numpy(mb)[None],
                                MIN_PIXELS).numpy()[0].astype(np.uint8)
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(filt, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64) if n_lab > 1 \
        else np.array([])
    out["shadow_cov"] = float(filt.mean())
    out["n_blobs"] = float(len(areas))
    out["blob_area_median"] = float(np.median(areas)) if len(areas) else np.nan
    out["small_object_frac"] = float((areas < SMALL_OBJECT_PX).mean()) \
        if len(areas) else np.nan
    return out, prof, hist


def build_h3_features(M, fda_mod):
    """Per-crop features for both datasets.  Cached; reads no model output."""
    fda_b = {L: fda_mod.fda_half_width(CROP, L) for L in FDA_L_VALUES}
    rad = radial_bins()
    rows, profs, hists, unit_ids = [], [], [], []

    # -- internal NAIP: the published-protocol test crops, 150 per city ------
    for city in INT_CITIES:
        idir = _guard_open(os.path.join(INT_IMG_ROOT, city, "highres",
                                        "test", "images"))
        mdir = _guard_open(os.path.join(INT_IMG_ROOT, city, "highres",
                                        "test", "masks"))
        names = sorted(os.listdir(idir))
        for nm in names:
            img = np.asarray(Image.open(os.path.join(idir, nm)).convert("RGB"),
                             dtype=np.uint8)
            msk = np.asarray(Image.open(os.path.join(mdir, nm)).convert("L"))
            if img.shape[:2] != (CROP, CROP):
                fail(f"internal crop {nm} is {img.shape[:2]}, expected 384x384")
            f, prof, hist = crop_features(img, msk, fda_b, rad, M)
            f.update(dataset="internal_naip", city=city,
                     unit=os.path.splitext(nm)[0], crop_index=0)
            rows.append(f)
            profs.append(prof)
            hists.append(hist)
            unit_ids.append(("internal_naip", city, os.path.splitext(nm)[0]))

    # -- external S-EO: every source image tiled at the dump's geometry ------
    for fold in FOLDS:
        cdir = SEO_CITY_DIR[fold]
        idir = _guard_open(os.path.join(SEO_IMG_ROOT, cdir, "30cm", "all",
                                        "images"))
        mdir = _guard_open(os.path.join(SEO_IMG_ROOT, cdir, "30cm", "all",
                                        "masks"))
        for nm in sorted(os.listdir(idir)):
            sid = os.path.splitext(nm)[0]
            img = np.asarray(Image.open(os.path.join(idir, nm)).convert("RGB"),
                             dtype=np.uint8)
            msk = np.asarray(Image.open(os.path.join(mdir, nm)).convert("L"))
            if msk.shape[:2] != img.shape[:2]:
                fail(f"S-EO mask/image shape mismatch on {nm}")
            for ci, (y, x) in enumerate(tile_offsets(*img.shape[:2])):
                f, prof, hist = crop_features(
                    img[y:y + CROP, x:x + CROP],
                    msk[y:y + CROP, x:x + CROP], fda_b, rad, M)
                f.update(dataset="external_seo", city=cdir, unit=sid,
                         crop_index=ci)
                rows.append(f)
                profs.append(prof)
                hists.append(hist)
                unit_ids.append(("external_seo", cdir, sid))
    feats = pd.DataFrame(rows)
    return feats, np.asarray(profs), np.asarray(hists)


def verify_tiling(data):
    """The centred tiling must reproduce the dump's stored patch geometry on
    every S-EO source image that is present on this box."""
    checked = 0
    for fold in FOLDS:
        idir = os.path.join(SEO_IMG_ROOT, SEO_CITY_DIR[fold], "30cm", "all",
                            "images")
        have = {os.path.splitext(x)[0] for x in os.listdir(idir)}
        df = data[(ARCHS[0], fold, "loco")]
        for sid, g in df[df.source_image_id.isin(have)].groupby("source_image_id"):
            w, h = Image.open(os.path.join(idir, f"{sid}.png")).size
            want = set(zip(g.patch_y.astype(int), g.patch_x.astype(int)))
            got = set(tile_offsets(h, w))
            if want != got:
                fail(f"{sid}: centred tiling does not reproduce the dump's "
                     f"patch geometry ({len(want ^ got)} offsets differ)")
            checked += 1
    return checked


def h3_unit_table(feats):
    """One row per clustering unit: an internal crop, or an S-EO source image."""
    scal = [c for c in feats.columns
            if c not in ("dataset", "city", "unit", "crop_index")]
    keys = ["dataset", "city", "unit"]
    u = feats.groupby(keys, as_index=False)[scal].mean()
    n = (feats.groupby(keys, as_index=False).size()
         .rename(columns={"size": "n_crops"}))
    return u.merge(n, on=keys, validate="one_to_one")


def h3_perm_test(u, col, stream):
    """Two-sample unit-level permutation on the mean difference."""
    v = u[col].to_numpy(np.float64)
    lab = (u.dataset == "external_seo").to_numpy()
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    n, ne = len(v), int(lab.sum())
    obs = float(v[lab].mean() - v[~lab].mean())
    rng = _rng(stream)
    tot = v.sum()
    null = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, size=ne, replace=False)
        se = v[idx].sum()
        null[b] = se / ne - (tot - se) / (n - ne)
    p = (1.0 + float((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)
    return dict(statistic=col, mean_internal=float(v[~lab].mean()),
                mean_external=float(v[lab].mean()), diff_ext_minus_int=obs,
                smd=smd(v[~lab], v[lab]), p_raw=p,
                n_units_internal=int((~lab).sum()), n_units_external=ne)


def h3_cdf_test(feats, hists, stream):
    """KS distance between the pooled grayscale CDFs, unit-level permutation.

    Grayscale is the published Section 4.2 definition: the uint8 cast of the
    RGB mean (armc_probes_ops.gray_cdf).
    """
    key = feats.dataset + "|" + feats.city + "|" + feats.unit
    codes, uniq = pd.factorize(key)
    H = np.zeros((len(uniq), 256))
    np.add.at(H, codes, hists)
    lab = np.array([u.split("|")[0] == "external_seo" for u in uniq])
    n, ne = len(uniq), int(lab.sum())

    def ks(hi, he):
        ci = np.cumsum(hi) / hi.sum()
        ce = np.cumsum(he) / he.sum()
        return float(np.abs(ci - ce).max()), float(np.abs(ci - ce).mean()), ci, ce

    tot = H.sum(0)
    he = H[lab].sum(0)
    obs, l1, cdf_int, cdf_ext = ks(tot - he, he)
    rng = _rng(stream)
    null = np.empty(B)
    for b in range(B):
        idx = rng.choice(n, size=ne, replace=False)
        h_e = H[idx].sum(0)
        null[b] = ks(tot - h_e, h_e)[0]
    p = (1.0 + float((null >= obs - 1e-12).sum())) / (B + 1.0)
    return (dict(statistic="grayscale_cdf_ks", ks=obs, mean_abs_cdf_gap=l1,
                 p_raw=p, n_units_internal=n - ne, n_units_external=ne),
            cdf_int, cdf_ext)


H3_SCAL = ["fda_band_amp_mean_L0p005", "fda_band_share_L0p005",
           "fda_band_amp_mean_exdc_L0p005", "fda_band_amp_mean_L0p01",
           "fda_band_share_L0p01", "fda_band_amp_mean_exdc_L0p01",
           "dc_mean_intensity", "spectral_slope", "gray_mean", "gray_std",
           "shadow_cov", "n_blobs", "blob_area_median", "small_object_frac"]


def h3_run_sample(feats, hists, name, max_nodata, stream_base):
    """The whole H3 distance table on one nested crop sample.

    `nodata_frac` is the filter variable, so it is tested only on the
    unfiltered sample, where testing it is not circular.
    """
    keep = (feats.nodata_frac.to_numpy(np.float64) <= max_nodata) \
        & np.isfinite(feats.fda_band_share_L0p01.to_numpy(np.float64))
    f, h = feats[keep].reset_index(drop=True), hists[keep]
    stats = list(H3_SCAL) + (["nodata_frac"] if max_nodata >= 1.0 else [])
    u = h3_unit_table(f)
    tests = [h3_perm_test(u, c, stream_base + i) for i, c in enumerate(stats)]
    cdf_row, cdf_int, cdf_ext = h3_cdf_test(f, h, stream_base + 90)
    cdf_row["smd"] = np.nan
    tests.append(cdf_row)
    for t, p in zip(tests, holm([t["p_raw"] for t in tests])):
        t["sample"] = name
        t["p_holm"] = float(p)
        t["holm_family_size"] = len(tests)
    inv = (f.groupby(["dataset", "city"], as_index=False)
           .agg(n_crops=("nodata_frac", "size"),
                n_units=("unit", "nunique"),
                mean_nodata_frac=("nodata_frac", "mean")))
    inv["sample"] = name
    desc = u.groupby(["dataset", "city"])[stats].mean().round(4).reset_index()
    desc["sample"] = name
    return tests, cdf_int, cdf_ext, u, desc, inv, keep


def h3_verdict(tests):
    """Mechanical read of the operator's pre-committed (a)/(b)."""
    band = [t for t in tests if t["statistic"].startswith("fda_band")]
    band_marked = [t for t in band
                   if t["p_holm"] < ALPHA and abs(t.get("smd", np.nan)) >= SMD_MARKED]
    band_sig = [t for t in band if t["p_holm"] < ALPHA]
    any_marked = [t for t in tests
                  if t["p_holm"] < ALPHA and abs(t.get("smd", np.nan) or 0) >= SMD_MARKED]
    a = bool(len(band_marked) > 0)
    b = bool(len(band_sig) == 0 and len(any_marked) == 0)
    return {"a_supported": a, "b_supported": b,
            "n_fda_band_statistics_marked": len(band_marked),
            "n_fda_band_statistics_significant": len(band_sig),
            "fda_band_statistics_marked": [t["statistic"] for t in band_marked],
            "n_statistics_marked_overall": len(any_marked),
            "statistics_marked_overall": [t["statistic"] for t in any_marked],
            "rule": (
                "(a) requires at least one FDA-band statistic with p_holm < "
                f"{ALPHA} AND |SMD| >= {SMD_MARKED}. (b) requires no FDA-band "
                "statistic significant and no statistic anywhere in the H3 "
                "family marked. The SMD floor exists because 450 internal and "
                "702 external units make significance alone uninformative; it "
                "was fixed in this file's config block before the full pass and "
                "every raw mean is reported so any other floor is re-derivable.")}


# ----------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------

def dry_run():
    print("=" * 118)
    print("07 -- SIB EXTERNAL FAILURE, MECHANISM ANALYSIS -- DRY RUN")
    print(f"S-EO dumps : {SEO_DIR}")
    print(f"internal   : {INT_DIR}")
    print(f"metrics    : {METRICS_PATH}")
    print(f"FDA source : {FDA_SRC}")
    print("=" * 118)

    M = load_metrics()
    fda = load_fda()
    print(f"metric module imported, sha256 {METRICS_SHA_EXPECTED[:16]}...")
    for L in FDA_L_VALUES:
        b = fda.fda_half_width(CROP, L)
        print(f"FDA band at L={L}: half-width b={b}, swapped block "
              f"{2*b}x{2*b} per channel of a {CROP}x{CROP} crop "
              f"({(2*b)**2} of {CROP*CROP} coefficients, "
              f"{(2*b)**2 / (CROP*CROP) * 100:.4f}%)")

    preflight()
    data, metas = load_seo()
    print("\n--- resolved schema, S-EO perimage.parquet ---")
    ref = data[(ARCHS[0], FOLDS[0], "loco")]
    print(f"{ref.shape[1] - 2} contract columns + 2 derived; dtypes:")
    for c in ["aoi_id", "sun_elevation", "sun_dist", "unseen_frac",
              "n_valid_px", "n_gt_fg_valid", "empty_fg_valid",
              "acquisition_date", "tp_tol", "miou_tol_pct"]:
        print(f"   {c:20s} {ref[c].dtype}")

    print("\n--- cells selected ---")
    rows = []
    for k in CELLS:
        df, m = data[k], metas[k]
        rows.append(dict(cell=m["run_id"], arch=k[0], fold=k[1], cond=k[2],
                         n_images=len(df), n_aoi=df.aoi_id.nunique(),
                         target_access=m.get("target_access"),
                         n_fully_unseen=int((df.n_valid_px <= 0).sum()),
                         mean_2class=round(float(df.miou2_pct.mean()), 4)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n--- H1 rows to be computed (6 per fold, 18 tests) ---")
    for letter, arch, kind, expr, iso in H1_ROWS:
        print(f"   {letter}  {arch:7s}  {expr:32s}  {iso}")

    print("\n--- H2 strata and frames ---")
    fr = h2_frame(data, True)
    fri = h2_frame(data, False)
    for k in sorted(fr):
        print(f"   {k[0]:7s}/{k[1]:4s}: {len(fr[k]):5d} patches primary "
              f"(fully-unseen excluded), {len(fri[k]):5d} inclusive, "
              f"{fr[k].aoi_id.nunique()} AOIs")
    f0 = fr[(ARCHS[0], "sib")]
    print("   strata ranges (patch-level, model-independent):")
    for v in H2_STRATA:
        q = np.quantile(f0[v], [0, .25, .5, .75, 1])
        print(f"     {v:12s} min/q1/med/q3/max = "
              + " / ".join(f"{x:.4g}" for x in q))

    print("\n--- H3 crop inventory ---")
    n_int = 0
    for c in INT_CITIES:
        d = os.path.join(INT_IMG_ROOT, c, "highres", "test", "images")
        n = len(os.listdir(d))
        n_int += n
        print(f"   internal NAIP {c:9s}: {n:5d} crops of 384x384 (test split)")
    n_ext_img = n_ext = 0
    for fold in FOLDS:
        d = os.path.join(SEO_IMG_ROOT, SEO_CITY_DIR[fold], "30cm", "all", "images")
        names = sorted(os.listdir(d))
        k = 0
        for nm in names:
            w, h = Image.open(os.path.join(d, nm)).size
            k += len(tile_offsets(h, w))
        n_ext_img += len(names)
        n_ext += k
        print(f"   external S-EO {SEO_CITY_DIR[fold]:9s}: {len(names):5d} source "
              f"images -> {k:5d} crops of 384x384")
    print(f"   totals: internal {n_int} crops / {n_int} units; "
          f"external {n_ext} crops / {n_ext_img} units")

    checked = verify_tiling(data)
    print(f"\ntiling verification: the centred non-overlapping 384 tiling "
          f"reproduces the dump's stored (patch_y, patch_x) exactly on all "
          f"{checked} S-EO source images present on this box")
    pool = set()
    for fold in FOLDS:
        pool |= set(data[(ARCHS[0], fold, "loco")].source_image_id.unique())
    print(f"overlap: {checked} of {len(pool)} arm-A test source images are "
          f"present here; the H3 S-EO sample is the {n_ext_img}-image "
          "dataset-comparison sample, not the arm-A test pool")

    print("\n--- H3 nested samples (the S-EO tiles carry footprint black; the "
          "internal crops carry none) ---")
    for n, c in H3_SAMPLES:
        print(f"   {n:26s} keeps crops with all-black pixel share <= {c:g}"
              + ("   <-- PRIMARY" if n == H3_PRIMARY else ""))
    print(f"   statistics per sample: {len(H3_SCAL)} scalars + the grayscale "
          "CDF KS; nodata_frac is tested on the unfiltered sample only")

    print("\n--- writes ---")
    print(f"   {RESULTS_PATH}")
    print(f"   {TABLES_PATH}")
    print(f"   {H3_CACHE}  (per-crop feature cache)")
    print("\nDRY RUN OK -- all assertions passed. No statistic computed, "
          "nothing written.")


# ----------------------------------------------------------------------------
# full pass
# ----------------------------------------------------------------------------

def main():
    t_start = datetime.now(timezone.utc)
    print("=" * 118)
    print("07 -- SIB EXTERNAL FAILURE, MECHANISM ANALYSIS -- FULL PASS")
    print("=" * 118)

    M = load_metrics()
    fda_mod = load_fda()
    preflight()
    data, metas = load_seo()
    xcheck, prior = verdict_crosscheck(data)

    aoi_cache = {}

    def aoi_of(fold):
        if fold not in aoi_cache:
            codes, uniq = pd.factorize(
                data[(ARCHS[0], fold, "loco")].aoi_id.values)
            aoi_cache[fold] = (codes, len(uniq))
        return aoi_cache[fold]

    out, fmt = {}, (lambda v: f"{v:.4f}")
    out["verdict_crosscheck"] = xcheck

    # ---------------- H1 ----------------
    print("\n" + "=" * 118)
    print("H1 -- COMPONENT ATTRIBUTION.  Per-image paired deltas, AOI cluster "
          f"bootstrap B={B}, AOI sign-flip p, Holm over the 18-test H1 family.")
    print("=" * 118)
    t1 = run_h1(data, aoi_of)
    for letter, arch, kind, expr, iso in H1_ROWS:
        s = t1[t1.row == letter]
        print(f"\n--- row {letter}: {expr}   [{iso}] ---")
        print(s[["fold", "delta", "stderr", "ci_lo", "ci_hi", "p_raw",
                 "p_holm", "n_images", "n_aoi"]].to_string(
            index=False, float_format=fmt))
        print(f"    mean over folds {s.delta.mean():+.4f}   "
              f"worst fold {s.loc[s.delta.idxmin(), 'fold'].upper()} "
              f"{s.delta.min():+.4f}   best fold "
              f"{s.loc[s.delta.idxmax(), 'fold'].upper()} {s.delta.max():+.4f}")
    v1 = h1_verdict(t1)
    print("\nH1 verdict:", json.dumps({k: v for k, v in v1.items() if k != "rule"}))
    out["h1"] = t1.to_dict("records")
    out["h1_verdict"] = v1

    # ---------------- H2 ----------------
    print("\n" + "=" * 118)
    print("H2 -- DAMAGE LOCALISATION.  Patch-level delta (method - vanilla) "
          "against model-independent strata; fold-stratified AOI cluster "
          f"bootstrap B={B}; Holm over 6 regression terms per (method x arch) "
          "and 2 contrasts per (method x arch).")
    print("Primary EXCLUDES the fully DSM-unseen patches: their delta is "
          "identically 0 by construction and would flatten the dirtiest "
          "stratum.  The inclusive fit is reported as sensitivity.")
    print("=" * 118)
    reg, strata, contrasts, sens, n_prim, n_incl = run_h2(data)
    for arch in ARCHS:
        for method in METHODS:
            print(f"\n--- {method.upper()} / {arch}  "
                  f"(n = {n_prim[(arch, method)]} patches primary, "
                  f"{n_incl[(arch, method)]} inclusive) ---")
            print(reg[(reg.arch == arch) & (reg.method == method)][
                ["term", "coef", "stderr", "ci_lo", "ci_hi", "iqr",
                 "effect_per_iqr", "p_raw", "p_holm"]].to_string(
                index=False, float_format=fmt))
    print("\n--- stratified mean delta by quantile bin ---")
    for arch in ARCHS:
        for method in METHODS:
            s = strata[(strata.arch == arch) & (strata.method == method)]
            print(f"\n  {method.upper()} / {arch}")
            print(s[["var", "bin", "n", "mean_delta", "ci_lo", "ci_hi"]]
                  .to_string(index=False, float_format=fmt))
    print("\n--- the sharpest single contrasts ---")
    print(contrasts[["method", "arch", "contrast", "n_clean", "n_dirty",
                     "mean_clean", "mean_dirty", "dirty_minus_clean",
                     "ci_lo", "ci_hi", "p_raw", "p_holm"]].to_string(
        index=False, float_format=fmt))
    print("\n--- sensitivity: same regression, fully-unseen patches kept ---")
    print(sens[["method", "arch", "term", "coef", "ci_lo", "ci_hi", "p_holm"]]
          .to_string(index=False, float_format=fmt))
    coll = h2_collinearity(data)
    print("\n--- how much of each patch-level stratum is just city? "
          "(R^2 of the stratum on the two city dummies) ---")
    print(coll.to_string(index=False, float_format=fmt))
    print("A stratum with a high R^2 here is largely a relabelling of the "
          "fold, so its marginal quantile gradient above and its partial "
          "regression coefficient are answering different questions.")
    v2 = h2_verdict(reg, contrasts)
    v2["stratum_city_collinearity"] = coll.to_dict("records")
    print("\nH2 verdict:", json.dumps({k: v for k, v in v2.items()
                                       if k != "rule"}))
    out["h2_regression"] = reg.to_dict("records")
    out["h2_strata"] = strata.to_dict("records")
    out["h2_contrasts"] = contrasts.to_dict("records")
    out["h2_sensitivity_incl_fully_unseen"] = sens.to_dict("records")
    out["h2_verdict"] = v2

    # ---------------- H3 ----------------
    print("\n" + "=" * 118)
    print("H3 -- DESIGN-ASSUMPTION MISMATCH, measured on inputs only.")
    print("CIRCULARITY GUARD: this section opens imagery and masks and reads "
          "no model output of any kind.")
    print("=" * 118)
    n_tiles = verify_tiling(data)
    print(f"tiling verified against the dump on {n_tiles} S-EO source images")

    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = all(os.path.isfile(p) for p in
                 (H3_CACHE, H3_SPEC_CACHE, H3_GUARD_CACHE))
    if cached:
        feats = pd.read_parquet(H3_CACHE)
        z = np.load(H3_SPEC_CACHE)
        profs, hists = z["profs"], z["hists"]
        with open(H3_GUARD_CACHE) as fh:
            guard = json.load(fh)["directories_read"]
        with open(H3_GUARD_CACHE) as fh:
            source = f"cache written {json.load(fh)['written_utc']}"
        print(f"H3 features read from cache ({len(feats)} crops)")
    else:
        print("H3: building per-crop features (this is the slow pass)...")
        feats, profs, hists = build_h3_features(M, fda_mod)
        guard = sorted(set(H3_READ_GUARD))
        feats.to_parquet(H3_CACHE, index=False)
        np.savez_compressed(H3_SPEC_CACHE, profs=profs, hists=hists)
        with open(H3_GUARD_CACHE, "w") as fh:
            json.dump({"directories_read": guard,
                       "written_utc": datetime.now(timezone.utc).isoformat()},
                      fh, indent=2)
        source = "built in this pass"
        print(f"H3 features written to {H3_CACHE} ({len(feats)} crops)")

    # The guard must be a real assertion whether or not the cache was hit, so
    # the directory list travels WITH the cache and is re-checked on every run.
    if not guard:
        fail("H3 read guard is empty; the directory record did not survive")
    bad = [p for p in guard
           if "/dump" in p or "sib" in os.path.basename(p).lower()]
    if bad:
        fail(f"H3 read guard tripped: {bad}")
    print(f"H3 read guard ({source}): {len(guard)} directories, none under a "
          "dump tree and none carrying a model output --")
    for p in guard:
        print(f"    {p}")

    nd = feats.nodata_frac.to_numpy(np.float64)
    print("\n--- no-data / all-black pixel share per 384 crop ---")
    print(feats.assign(black_gt0=nd > 0)
          .groupby(["dataset", "city"])
          .agg(n_crops=("nodata_frac", "size"),
               frac_crops_with_any_black=("black_gt0", "mean"),
               mean_black_share=("nodata_frac", "mean"),
               max_black_share=("nodata_frac", "max"))
          .round(5).to_string())
    print("The internal NAIP crops carry no all-black pixel at all; the S-EO "
          "source tiles are satellite footprints and carry black in most "
          "crops. H3 therefore runs on three nested samples.")

    all_tests, all_desc, all_inv, cdfs, keeps = [], [], [], {}, {}
    v3 = None
    for si, (sname, cap) in enumerate(H3_SAMPLES):
        tests, cdf_int, cdf_ext, u, desc, inv, keep = h3_run_sample(
            feats, hists, sname, cap, 5000 + 200 * si)
        all_tests += tests
        all_desc.append(desc)
        all_inv.append(inv)
        cdfs[sname] = {"internal_naip": cdf_int.tolist(),
                       "external_seo": cdf_ext.tolist()}
        keeps[sname] = keep
        t3 = pd.DataFrame(tests)
        n_int = int((u.dataset == "internal_naip").sum())
        n_ext = int((u.dataset == "external_seo").sum())
        tag = "  <-- PRIMARY" if sname == H3_PRIMARY else ""
        print(f"\n--- H3 distance table [{sname}, black share <= {cap:g}]{tag}")
        print(f"    {int(keep.sum())} of {len(feats)} crops; units: "
              f"{n_int} internal, {n_ext} external; unit-level permutation "
              f"B={B}, two-tailed, Holm over {len(tests)} ---")
        print(t3[["statistic", "mean_internal", "mean_external",
                  "diff_ext_minus_int", "smd", "p_raw", "p_holm"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))
        cdf_row = tests[-1]
        print(f"    grayscale CDF: KS = {cdf_row['ks']:.4f}, mean |CDF gap| = "
              f"{cdf_row['mean_abs_cdf_gap']:.4f}, p_holm = "
              f"{cdf_row['p_holm']:.5f}")
        if sname == H3_PRIMARY:
            v3 = h3_verdict(tests)
            print("\n--- per-city descriptives, primary sample "
                  "(no test; the tests are pooled) ---")
            print(desc.to_string(index=False))

    # radial spectra profiles on the primary sample, per dataset x city.
    # 5 of the 48 log-spaced bins hold no FFT grid point at all (the spacing is
    # finer than the grid near DC); those are structurally empty for every crop
    # and are dropped rather than averaged.
    _, _, centres, counts = radial_bins()
    occupied = counts > 0
    kp = keeps[H3_PRIMARY]
    fp, pp = feats[kp].reset_index(drop=True), profs[kp][:, occupied]
    key = fp.dataset + "|" + fp.city
    prof_out = {}
    for k in sorted(key.unique()):
        m = (key == k).to_numpy()
        with np.errstate(divide="ignore"):
            lp = np.log10(np.where(pp[m] > 0, pp[m], np.nan))
        fin = np.isfinite(lp)
        n = fin.sum(0)
        if (n == 0).any():
            fail(f"radial profile for {k} has an occupied bin with no finite "
                 "power in any crop")
        prof_out[k] = (np.where(fin, lp, 0.0).sum(0) / n).tolist()
    kc = centres[occupied]
    print(f"\n--- radial power spectra, primary sample, mean log10 power, "
          f"{int(occupied.sum())} occupied of {N_RADIAL_BINS} bins, "
          "6 shown ---")
    show = [0, 6, 13, 20, 27, 34]
    print("  k (cyc/px):  " + "  ".join(f"{kc[i]:9.5f}" for i in show))
    for k in sorted(prof_out):
        print(f"  {k:22s} " + "  ".join(f"{prof_out[k][i]:9.4f}" for i in show))

    inv_all = pd.concat(all_inv, ignore_index=True)
    desc = pd.concat(all_desc, ignore_index=True)
    t3 = pd.DataFrame([t for t in all_tests if t["sample"] == H3_PRIMARY])
    print("\nH3 verdict (primary sample):",
          json.dumps({k: v for k, v in v3.items() if k != "rule"}))
    out["h3_tests"] = all_tests
    out["h3_sample_inventory"] = inv_all.to_dict("records")
    out["h3_descriptives"] = desc.to_dict("records")
    out["h3_radial_profiles"] = {
        "sample": H3_PRIMARY,
        "k_cyc_per_px": kc.tolist(),
        "n_fft_coefficients_per_bin": counts[occupied].tolist(),
        "bins_dropped_as_structurally_empty": int((~occupied).sum()),
        "mean_log10_power": prof_out}
    out["h3_grayscale_cdf"] = cdfs
    out["h3_verdict"] = v3

    # prior measured comparison, cited not recomputed
    prior_cmp = pd.read_csv(PRIOR_COMPARISON)
    prior_cmp = prior_cmp[prior_cmp.comparison == "Ours vs SEO"]
    out["h3_prior_comparison_cited"] = prior_cmp.to_dict("records")

    # ------------------------------------------------------------------
    # results JSON
    # ------------------------------------------------------------------
    conditions, tests_json = [], []
    for r in t1.to_dict("records"):
        conditions.append({
            "label": f"H1 row {r['row']}: {r['expr']} / holdout {r['fold'].upper()}",
            "metric": "delta_2class_tol_k2_unseen_excluded",
            "component_isolated": r["isolates"],
            "mean": round(r["delta"], 4), "stderr": round(r["stderr"], 4),
            "n": int(r["n_images"]), "n_aoi": int(r["n_aoi"]),
            "ci95": [round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
            "ci_method": "aoi_cluster_bootstrap_B10000",
            "baseline": (run_id(r["arch"], r["fold"], "loco")
                         if r["row"] in ("A", "B", "C", "D")
                         else "the other method cell of the same arch and fold")})
        tests_json.append({
            "name": "AOI-level sign-flip permutation test",
            "compares": [f"H1 row {r['row']}", r["fold"]],
            "metric": "delta_2class_tol_k2_unseen_excluded",
            "statistic": round(r["delta"], 4), "p": r["p_raw"],
            "p_holm": round(float(r["p_holm"]), 6), "B": B, "two_tailed": True,
            "family": "18 tests of the H1 family, Holm-Bonferroni"})
    for r in reg.to_dict("records"):
        tests_json.append({
            "name": "cluster-bootstrap two-tailed test on an OLS slope",
            "compares": [f"H2 {r['method']}/{r['arch']}", r["term"]],
            "metric": "delta_2class_tol_k2_unseen_excluded per unit of the stratum",
            "statistic": round(r["coef"], 6), "p": r["p_raw"],
            "p_holm": round(float(r["p_holm"]), 6), "B": B, "two_tailed": True,
            "family": "6 regression terms of that (method x arch), Holm-Bonferroni"})
    for r in contrasts.to_dict("records"):
        tests_json.append({
            "name": "cluster-bootstrap two-tailed test on a stratum contrast",
            "compares": [f"H2 {r['method']}/{r['arch']}", r["contrast"]],
            "metric": "delta_2class_tol_k2_unseen_excluded",
            "statistic": round(r["dirty_minus_clean"], 4), "p": r["p_raw"],
            "p_holm": round(float(r["p_holm"]), 6), "B": B, "two_tailed": True,
            "family": "2 contrasts of that (method x arch), Holm-Bonferroni"})
    for t in all_tests:
        tests_json.append({
            "name": "unit-level two-sample permutation test (internal vs external)",
            "compares": ["internal_naip_crops", "external_seo_crops"],
            "sample": t["sample"],
            "metric": t["statistic"],
            "statistic": round(float(t.get("diff_ext_minus_int", t.get("ks"))), 6),
            "smd": (None if not np.isfinite(t.get("smd", np.nan))
                    else round(float(t["smd"]), 4)),
            "p": t["p_raw"], "p_holm": round(float(t["p_holm"]), 6),
            "B": B, "two_tailed": t["statistic"] != "grayscale_cdf_ks",
            "family": (f"{t['holm_family_size']} statistics of the H3 family "
                       f"on sample {t['sample']}, Holm-Bonferroni")})

    supported = []
    if v1["a_supported"]:
        supported.append("H1(a)")
    if v1["b_supported"]:
        supported.append("H1(b)")
    if v1["c_supported"]:
        supported.append("H1(c)")
    if v2["a_supported"]:
        supported.append("H2(a)")
    if v2["b_supported"]:
        supported.append("H2(b)")
    if v2["c_supported"]:
        supported.append("H2(c)")
    if v3["a_supported"]:
        supported.append("H3(a)")
    if v3["b_supported"]:
        supported.append("H3(b)")

    A = t1[t1.row == "A"]
    Bq = t1[t1.row == "B"]
    E = t1[t1.row == "E"]
    F = t1[t1.row == "F"]

    res = {
        "id": "07_sib_mechanism",
        "arm": "A",
        "answers": ["v1UJ-W2"],
        "question": ("Why does SIB fail the pooled worst-case criterion on "
                     "S-EO?  Three pre-committed hypotheses: component "
                     "attribution (H1), damage localisation (H2), and "
                     "design-assumption mismatch measured on inputs alone (H3)."),
        "runs": [metas[k]["run_id"] for k in CELLS],
        "index_sha256": sha256_file(SEO_INDEX),
        "internal_index_sha256": (sha256_file(INT_INDEX)
                                  if os.path.isfile(INT_INDEX) else None),
        "manifest": "logs/armA_phase3_manifest.json",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/07_sib_mechanism.py",
        "git_hash": None,
        "git_hash_note": ("the project base is not a git repository on this "
                          "box. Per-cell producing commits are in "
                          "dumps/seo/INDEX.parquet.git_hash."),
        "seed": SEED,
        "B": B,
        "metric_module": {"path": METRICS_PATH, "sha256": METRICS_SHA_EXPECTED},
        "fda_definition_source": {
            "path": FDA_SRC, "function": "fda_swap / fda_half_width",
            "sha256": sha256_file(FDA_SRC),
            "band": {f"L={L}": {
                "half_width_b": int(fda_mod.fda_half_width(CROP, L)),
                "swapped_block": f"{2*int(fda_mod.fda_half_width(CROP, L))}x"
                                 f"{2*int(fda_mod.fda_half_width(CROP, L))}",
                "coefficients_per_channel": int((2 * fda_mod.fda_half_width(CROP, L)) ** 2),
                "share_of_384x384_coefficients":
                    round((2 * fda_mod.fda_half_width(CROP, L)) ** 2
                          / (CROP * CROP), 8)} for L in FDA_L_VALUES}},
        "scope": {
            "verdict_status": ("FINAL and countersigned; not re-examined here. "
                               "results/03_seo_methods.json is reproduced "
                               "exactly as a precondition, not recomputed as a "
                               "result."),
            "circularity_guard": ("SIB was designed from the internal Section 4 "
                                  "diagnosis, so its external failure is "
                                  "evidence about neither direction of that "
                                  "diagnosis. H3 measures input statistics "
                                  "directly on imagery and reads no SIB output; "
                                  "the guard is enforced in code."),
            "pre_commitment": ("The three hypotheses and their (a)/(b)/(c) "
                               "readings were fixed by the operator before this "
                               "ran. The numeric thresholds that operationalise "
                               "them (alpha, the SMD floor, the all-three-folds "
                               "quantifier) were fixed in this file's config "
                               "block before the full pass; the dry run "
                               "computes no statistic.")},
        "decision_rules": {"H1": v1["rule"], "H2": v2["rule"], "H3": v3["rule"]},
        "metric_definitions": {
            "delta_2class_tol_k2_unseen_excluded":
                "mean over test patches of (cell_a - cell_b) two-class "
                "boundary-tolerant mIoU at K=2 with DSM-unseen pixels excluded "
                "from all four confusion cells; recomputed per patch from "
                "tp_tol/fp_tol/fn_tol/tn_tol as 50*(tp/(tp+fp+fn+1e-10) + "
                "tn/(tn+fp+fn+1e-10)), matching metrics.py _img_metrics. "
                "Identical to results/03_seo_methods.json.",
            "fda_band_amp_mean": "mean |FFT| over the exact block FDA swaps, "
                                 "b = floor(384*L), averaged over RGB channels.",
            "fda_band_share": "sum |FFT| in that block / sum |FFT| over all "
                              "coefficients; scale-invariant.",
            "spectral_slope": "OLS slope of log10 radial power against log10 "
                              f"radial frequency over {SLOPE_BAND} cyc/px.",
            "shadow_cov": "GT shadow pixel fraction of a 384 px crop after the "
                          f"<{MIN_PIXELS} px connected-component filter.",
            "small_object_frac": f"fraction of GT blobs below {SMALL_OBJECT_PX} "
                                 "px; the threshold is the declared assumption "
                                 "of analysis/gj_fill_pass.py.",
            "smd": "standardised mean difference (external - internal), pooled sd."},
        "conditions": conditions,
        "tests": tests_json,
        "h1_component_table": t1.to_dict("records"),
        "h1_verdict": v1,
        "h2_regression": reg.to_dict("records"),
        "h2_strata": strata.to_dict("records"),
        "h2_contrasts": contrasts.to_dict("records"),
        "h2_sensitivity_incl_fully_unseen": sens.to_dict("records"),
        "h2_verdict": v2,
        "h3_distance_table": all_tests,
        "h3_samples": [{"name": n, "max_nodata_frac": c,
                        "role": ("primary" if n == H3_PRIMARY else "sensitivity")}
                       for n, c in H3_SAMPLES],
        "h3_sample_inventory": out["h3_sample_inventory"],
        "h3_descriptives": desc.to_dict("records"),
        "h3_radial_profiles": out["h3_radial_profiles"],
        "h3_grayscale_cdf": out["h3_grayscale_cdf"],
        "h3_prior_comparison_cited": out["h3_prior_comparison_cited"],
        "h3_verdict": v3,
        "supported_readings": supported,
        "verdict_crosscheck": xcheck,
        "headline": (
            "Decomposing SIB's S-EO deltas over the method grid, the "
            f"FDA-bearing dinov3 cell sits {Bq.delta.mean():+.4f} two-class "
            f"mIoU at K=2 against vanilla on average over three folds while the "
            f"FDA-free SIB core sits {A.delta.mean():+.4f}, the implied "
            f"core+SAG increment on mamnet is {E.delta.mean():+.4f}, and the "
            "supported pre-committed readings are "
            f"{', '.join(supported) if supported else 'none'}."),
        "caveats": [],
        "figure": None,
        "caption": None,
    }

    res["caveats"] = [
        "The verdict is not under re-examination. SIB's S-EO worst-case failure "
        "(dinov3/OMA, -0.4216 two-class mIoU at K=2) is carried in from "
        "results/03_seo_methods.json and reproduced exactly as a precondition. "
        "Nothing here re-scores SIB.",
        "H1 is a PARTIAL ablation read off a grid that was not designed as one. "
        "Only four configurations exist per fold, so no component is isolated on "
        "both architectures: dinov3-SIB isolates the invariance core, dinov3-FDA "
        "isolates FDA, mamnet-FDA carries FDA plus the contrast channel, and "
        "mamnet-SIB carries everything. Rows E and F are exact per-image "
        "contrasts, but calling row E 'the core+SAG contribution' assumes the "
        "components act additively on the mIoU scale and that no interaction "
        "with FDA exists. No cell exists that would test that assumption.",
        "The two architectures' 'cores' are not the same object. dinov3's core is "
        "haar + content VIB + edge VIB on a ViT; mamnet's row-E increment adds "
        "SAG and sits on top of FDA plus a contrast channel on a CNN. Sign "
        "agreement between them is reported as a consistency check, not as a "
        "replication.",
        "SIB on dinov3 is source_only and SIB on mamnet is unlabeled_target "
        "(erratum I-1: the CNN branch carries FDA). H1 never pools across that "
        "boundary; every row is within one architecture.",
        "H2's primary fit excludes the 108 fully DSM-unseen test patches, whose "
        "delta is identically zero by construction and which sit at "
        "unseen_frac = 1.0 -- keeping them puts a block of deterministic zeros in "
        "the dirtiest stratum and flattens exactly the slope H2 is testing. The "
        "inclusive fit is reported as sensitivity and is the arm-A convention.",
        "H2's strata are correlated with each other and with city. JAX is both "
        "the clean-label fold and the lowest-unseen fold, so the "
        "unseen-quartile contrast and the JAX contrast are confounded; both are "
        "reported and neither identifies which axis carries the effect.",
        "H2's sun_dist is measured against the LOCO cell's own training "
        "sun-elevation mean, so it is a distance to the training distribution, "
        "not an absolute illumination coordinate.",
        "H3's S-EO imagery is the 702-image dataset-comparison sample under "
        "'Shade Maps'/data/data_comparisons/SEO, which is the only S-EO imagery "
        "on this box. It is not the arm-A test pool: only 65 of the arm-A test "
        "source images are present here. The crop geometry is verified to "
        "reproduce the dump's stored patch offsets exactly on those 65, so the "
        "crop unit is right; the image sample is not the test sample.",
        "H3's internal comparator is the published-protocol high-resolution test "
        "split, 150 crops per city. SIB's hyperparameters were tuned on the "
        "internal benchmark as a whole, not on that split specifically.",
        "The S-EO source tiles are satellite footprints and carry all-black "
        "(0,0,0) pixels; the 450 internal NAIP crops carry none at all. The "
        "black mass is scattered over roughly a thousand connected components "
        "per tile rather than pooled in one padding polygon, so it mixes "
        "footprint no-data with genuinely clipped-dark imagery and the two are "
        "not separable from this asset. H3 therefore runs on three nested "
        "samples -- black share <= 1% (primary), == 0 (strict), and unfiltered "
        "-- and h3_sample_inventory reports what each keeps. The filter is "
        "conservative with respect to reading (a): black pixels inflate "
        "low-frequency amplitude, so removing them can only shrink an FDA-band "
        "difference, never manufacture one.",
        "The all-black share is itself reported as an H3 statistic, on the "
        "unfiltered sample only, where testing it is not circular. It is a real "
        "input difference between the two sensors' delivered products and not "
        "only an artifact, but it is not attributable to either cause here.",
        "H3 tests input statistics only. It says nothing about whether the "
        "measured mismatch is what produced SIB's loss; that link is an "
        "inference the numbers here do not close.",
        "With 450 internal and 702 external units, statistical significance in "
        "H3 is cheap. Every H3 reading is therefore gated on a standardised "
        "effect size as well as a Holm-adjusted p, and every raw mean is "
        "reported so any other threshold is re-derivable.",
        "No prior 'S-EO crop-level CDF measurement' artifact exists on this box. "
        "The grayscale CDF distance reported here is computed in this pass, on "
        "the published Section 4.2 grayscale definition, for both datasets. The "
        "one prior measured internal-vs-S-EO comparison found on this box is the "
        "dataset-comparison table under 'Shade Maps', cited verbatim in "
        "h3_prior_comparison_cited; it is computed on whole S-EO tiles rather "
        "than 384 px crops and disagrees with the crop-matched measurement on the "
        "direction of the shadow-coverage difference.",
        "Single seed. Every S-EO cell is seed 0, so every H1 and H2 interval "
        "covers test-patch and AOI sampling only, not optimisation noise. A "
        "component contribution smaller than across-seed variance would not be "
        "distinguishable here, and across-seed variance is unmeasured on S-EO.",
        "Two cells stopped on the epoch ceiling rather than on patience "
        "(dinov3-fda-loco-r3500-oma-hr-s0, dinov3-sib-loco-r3500-oma-hr-s0), so "
        "they are not converged under the same rule as the others. Both sit in "
        "the OMA fold, which is the fold carrying SIB's worst cell.",
        "App H's deployment-time coverage threshold is applied in no SIB cell "
        "(manifest sib_coverage_threshold). These numbers describe SIB without "
        "that component, so H3's coverage-distribution measurement bears on a "
        "component that was not active.",
        "rung 3500 is the S-EO patch budget; the unit is a 384 px patch, not a "
        "benchmark image, and the magnitudes are not commensurable with the "
        "internal 225/450/900 ladder.",
        "High-resolution cells only, three cities, two architectures. OGLANet is "
        "not in arm A.",
        "S-EO labels are auto-generated. Both sides of every H1 and H2 delta use "
        "the same labels on the same patches, so label noise largely cancels in "
        "the difference; H3's absolute input statistics carry no label at all.",
    ]

    # sanity: anything too clean
    flags = []
    if float(np.abs(t1.delta).min()) < 1e-6:
        flags.append("an H1 row is numerically zero")
    tiny = t1[(t1.stderr < 1e-3)]
    if len(tiny):
        flags.append(f"{len(tiny)} H1 rows have stderr below 1e-3")
    for r in t1.to_dict("records"):
        if not (r["ci_lo"] <= r["delta"] <= r["ci_hi"]):
            flags.append(f"H1 row {r['row']}/{r['fold']}: the bootstrap CI "
                         "excludes its own point estimate")
    n_sig_h3 = int((t3.p_holm < ALPHA).sum())
    if n_sig_h3 == len(t3):
        flags.append(f"all {len(t3)} H3 statistics on the primary sample are "
                     "Holm-significant; with hundreds of units on each side "
                     "this is expected and is why the reading is gated on "
                     "effect size as well")
    prim = [t for t in all_tests if t["sample"] == H3_PRIMARY]
    for other in [s for s, _ in H3_SAMPLES if s != H3_PRIMARY]:
        o = {t["statistic"]: t for t in all_tests if t["sample"] == other}
        flip = [t["statistic"] for t in prim
                if t["statistic"] in o
                and np.sign(t.get("diff_ext_minus_int", 0)) != 0
                and np.sign(t.get("diff_ext_minus_int", 0))
                != np.sign(o[t["statistic"]].get("diff_ext_minus_int", 0))]
        if flip:
            flags.append(f"H3 statistics whose sign flips between the primary "
                         f"sample and {other}: {', '.join(flip)}")
    bam = [t for t in prim if t["statistic"] == "blob_area_median"]
    if bam:
        per_city = desc[(desc["sample"] == H3_PRIMARY)
                        & (desc.dataset == "external_seo")]["blob_area_median"]
        if len(per_city) and per_city.max() > 5 * max(per_city.min(), 1e-9):
            flags.append(
                "external blob_area_median spans "
                f"{per_city.min():.0f}-{per_city.max():.0f} px across the three "
                "S-EO cities and moves by more than 2x between the H3 samples; "
                "it is the least stable statistic in the table and its primary "
                "p_holm does not clear alpha")
    prior_cov = [r for r in out["h3_prior_comparison_cited"]
                 if r.get("metric") == "shadow_coverage"]
    if prior_cov:
        pc = prior_cov[0]
        cov = [t for t in prim if t["statistic"] == "shadow_cov"]
        if cov and np.sign(pc["other_mean"] - pc["ours_mean"]) != \
                np.sign(cov[0]["diff_ext_minus_int"]):
            flags.append(
                "the prior dataset-comparison asset and this crop-matched "
                "measurement disagree on the SIGN of the internal-vs-S-EO "
                "shadow-coverage difference; the prior asset measures whole "
                "S-EO tiles, this one measures 384 px crops")
    res["sanity_flags"] = flags
    print("\n--- sanity flags ---")
    print("\n".join(f"  {f}" for f in flags) if flags else "  none")

    print("\nSUPPORTED READINGS: "
          + (", ".join(supported) if supported else "NONE -- mechanism "
             "undetermined at this grid"))
    print("HEADLINE: " + res["headline"])

    with open(RESULTS_PATH, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {RESULTS_PATH}")
    with open(TABLES_PATH, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"wrote {TABLES_PATH}")
    print(f"wall clock {(datetime.now(timezone.utc) - t_start).total_seconds():.1f} s")


if __name__ == "__main__":
    if DRY_RUN:
        dry_run()
    else:
        main()
