"""Arms G and J — error taxonomy for reviewer zNBu-W6.

Where do the models fail, and does geographic transfer move where they fail?

Five analyses, each over the 9 (arch, city) cell pairs
{mamnet, oglanet, dinov3} x {chicago, miami, phoenix}, r450 / hr / s0:

  A  FN / FP decomposition of the gap
  B  FN by shadow category
  C  FN by GT structure bin (skeleton width, image contrast quartile)
  D  FP proxies (darkness, distance to GT, blob count, blob size)
  E  boundary versus interior share of FN and FP

Every delta is LOCO minus the within-city upper bound, paired per image: the two
cells of a pair are evaluated on the same 150 test images, joined on image_id.

DENOMINATOR RULE, applied everywhere and stated in every output: per-category
rates use CATEGORISED GT ONLY.  342 of 2700 image-rows carry GT shadow pixels
whose multiclass value is 0 (an unrecognised annotation Type maps to 0, the same
value as background), and those pixels are absent from gt_px_by_cat.  Analysis B
therefore reports rates over categorised GT, not over all GT.  Analyses A, C, D
and E use the full valid-region counts and are unaffected.

Statistics follow CONVENTIONS.md: paired bootstrap on per-image deltas,
B = 10,000, percentile CI at 2.5 / 97.5, two-tailed p of min(2 * prop_wrong, 1),
Holm-Bonferroni within each family.  The bootstrap helpers are copied verbatim
from analysis/16_armL_ssl_regime.py so the intervals are comparable across
tables.

Analysis only.  No figures.
"""
import hashlib
import json
import os
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
DRY_RUN = "--dry-run" in sys.argv
DRY_PAIRS = [("mamnet", "phoenix")]

BASE = os.environ["PROJECT_ROOT"]
DUMPS = os.path.join(BASE, "dumps", "st")
DATA = os.path.join(os.environ["PROJECT_ROOT"], "data/Final_data_test")
OUTDIR = os.path.join(BASE, "analysis", "gj_analysis")
WIDTH_BINS = os.path.join(OUTDIR, "width_bins.parquet")
WIDTH_META = os.path.join(OUTDIR, "width_bins_meta.json")
FILL_REPORT = os.path.join(BASE, "analysis", "fill_pass_report.json")
LEGEND_PATH = os.path.join(BASE, "analysis", "category_legend.json")
RESULTS_JSON = os.path.join(OUTDIR, "results.json")
FINDINGS_MD = os.path.join(OUTDIR, "FINDINGS.md")

B = 10000
BOOT_SEED = 42
N_IMAGES_EXPECTED = 150
SPOT_CHECK_N = 5          # images per cell for the fn identity spot check
TIE_DRIFT_BOUND_PX = 21   # analysis/fill_pass_report.json, one-sided
FP_NEAR_PX = 15           # analysis D image-level near / far split
FP_NEAR_GT_PX = 5         # the fp_px_near_gt column's own threshold

ARCHES = ["mamnet", "oglanet", "dinov3"]
CITIES = ["chicago", "miami", "phoenix"]
CATS = ["building_canyon", "under_structure", "tree_canopy",
        "topography", "vehicle", "thin_linear"]
WIDTH_KEYS = ["w_lt3", "w_3to8", "w_gt8"]
QUART_KEYS = ["wq1", "wq2", "wq3", "wq4"]
CQ_KEYS = ["cq1", "cq2", "cq3", "cq4"]

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


def loco_id(arch, city):
    return f"{arch}-vanilla-loco-r450-{city}-hr-s0"


def upper_id(arch, city):
    return f"{arch}-none-upper-r450-{city}-hr-s0"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.join(BASE, "final_paper"), "rev-parse",
             "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# ------------------------------------------------------------- statistics
def _rng(seed=BOOT_SEED):
    return np.random.default_rng(seed)


def paired_bootstrap(delta, B=B, seed=BOOT_SEED):
    """Paired bootstrap on a per-image delta vector.

    Verbatim behaviour of analysis/16_armL_ssl_regime.py:paired_bootstrap in its
    no-cluster branch: B resamples of the image index, percentile CI at
    2.5 / 97.5, two-tailed p of min(2 * prop_wrong, 1.0).  NaN entries are
    excluded first and the surviving n is reported.
    """
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    n = delta.size
    empty = dict(mean=None, median=None, stderr=None, ci95_lo=None,
                 ci95_hi=None, p_raw=None, n=int(n), B=B)
    if n == 0:
        return empty
    observed = float(delta.mean())
    rng = _rng(seed)
    boots = delta[rng.integers(0, n, size=(B, n))].mean(axis=1)
    prop_wrong = (float(np.mean(boots <= 0)) if observed >= 0
                  else float(np.mean(boots >= 0)))
    return dict(mean=observed, median=float(np.median(delta)),
                stderr=(float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else None),
                ci95_lo=float(np.percentile(boots, 2.5)),
                ci95_hi=float(np.percentile(boots, 97.5)),
                p_raw=float(min(2.0 * prop_wrong, 1.0)), n=int(n), B=B)


def pooled_bootstrap(deltas_by_city, B=B, seed=BOOT_SEED):
    """Mean over cells of the per-cell mean delta, with an image-clustered CI.

    `deltas_by_city` maps city -> list of per-image delta arrays, one per arch,
    all aligned on that city's 150 test images.  A replicate resamples each
    city's image index ONCE and reuses it for every arch in that city, because
    the archs are evaluated on the same images; the replicate statistic is the
    unweighted mean over all cells.  This is the image-cluster bootstrap of
    App G.9 with the cluster set to the image.
    """
    cell_means, boot_cols = [], []
    rng = _rng(seed)
    for city in sorted(deltas_by_city):
        arrs = [np.asarray(a, dtype=np.float64) for a in deltas_by_city[city]]
        n = arrs[0].size
        idx = rng.integers(0, n, size=(B, n))
        for a in arrs:
            if not np.isfinite(a).any():
                continue
            cell_means.append(float(np.nanmean(a)))
            boot_cols.append(np.nanmean(a[idx], axis=1))
    empty = dict(mean=None, stderr=None, ci95_lo=None, ci95_hi=None,
                 p_raw=None, n_cells=0, B=B)
    if not cell_means:
        return empty
    observed = float(np.mean(cell_means))
    boots = np.nanmean(np.vstack(boot_cols), axis=0)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return empty
    prop_wrong = (float(np.mean(boots <= 0)) if observed >= 0
                  else float(np.mean(boots >= 0)))
    return dict(mean=observed,
                stderr=float(np.std(boots, ddof=1)),
                ci95_lo=float(np.percentile(boots, 2.5)),
                ci95_hi=float(np.percentile(boots, 97.5)),
                p_raw=float(min(2.0 * prop_wrong, 1.0)),
                n_cells=len(cell_means), B=B,
                stderr_basis="bootstrap sd over replicates of the cell mean")


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, matching Table 39."""
    pv = [(p, i) for i, p in enumerate(pvals) if p is not None]
    out = [None] * len(pvals)
    if not pv:
        return out
    pv.sort()
    m, running = len(pv), 0.0
    for rank, (p, i) in enumerate(pv):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = float(running)
    return out


def ratio_bootstrap(num, den, B=B, seed=BOOT_SEED):
    """CI for mean(num) / (mean(num) + mean(den)) under image resampling.

    Used once, for the share of the LOCO error increase that is false negatives.
    Both inputs are per-image paired deltas in the same units (px per 1000 valid
    px), so the ratio is a share of the total extra error, not a per-image mean
    of ratios.
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    n = num.size
    if n == 0:
        return dict(point=None, ci95_lo=None, ci95_hi=None, n=0, B=B)
    tot = num.mean() + den.mean()
    point = float(num.mean() / tot) if tot != 0 else None
    rng = _rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    a, b = num[idx].mean(axis=1), den[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / (a + b)
    r = r[np.isfinite(r)]
    return dict(point=point,
                ci95_lo=float(np.percentile(r, 2.5)) if r.size else None,
                ci95_hi=float(np.percentile(r, 97.5)) if r.size else None,
                n=int(n), B=B)


# ------------------------------------------------------------------ input
def safe_div(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    out[ok] = a[ok] / b[ok]
    return out


def load_cell(run_id, city, wb):
    cdir = os.path.join(DUMPS, run_id)
    ppath = os.path.join(cdir, "perimage.parquet")
    epath = os.path.join(cdir, "perimage_ext.parquet")
    rpath = os.path.join(cdir, "run.json")
    for req in (ppath, epath, rpath):
        if not os.path.exists(req):
            sys.exit(f"STOP: {req} absent")
    with open(rpath) as f:
        rj = json.load(f)
    df = pd.read_parquet(ppath)
    ext = pd.read_parquet(epath)
    if len(df) != N_IMAGES_EXPECTED:
        sys.exit(f"STOP: {run_id} n_images={len(df)}, expected {N_IMAGES_EXPECTED}")
    if set(df.city) != {city}:
        sys.exit(f"STOP: {run_id} city column {set(df.city)} != {city}")
    if rj.get("holdout") != city or int(rj.get("rung", -1)) != 450 \
            or rj.get("res") != "hr" or int(rj.get("seed", -1)) != 0:
        sys.exit(f"STOP: {run_id} run.json disagrees with its id: "
                 f"holdout={rj.get('holdout')} rung={rj.get('rung')} "
                 f"res={rj.get('res')} seed={rj.get('seed')}")
    if list(df.image_id) != list(ext.image_id):
        sys.exit(f"STOP: {run_id} perimage / ext image_id order differs")
    w = wb[wb.run_id == run_id].set_index("image_id").loc[list(df.image_id)] \
        if wb is not None else None
    if w is not None and len(w) != len(df):
        sys.exit(f"STOP: {run_id} width_bins rows {len(w)} != {len(df)}")
    df = df.set_index("image_id")
    ext = ext.set_index("image_id")
    return df, ext, w, rj


def cat_map_col(df, field, cat):
    """A map<string,int64> column -> per-image int array for one category key."""
    def get(m):
        if m is None:
            return 0
        if isinstance(m, dict):
            return int(m.get(cat, 0))
        return int(dict(m).get(cat, 0))
    return df[field].map(get).values.astype(np.float64)


def per_image_metrics(df, ext, w):
    """Every per-image quantity the five analyses need, on one cell."""
    m = {}
    n_gt = df.n_gt_fg_valid.values.astype(np.float64)
    n_valid = df.n_valid_px.values.astype(np.float64)
    fn = df.fn_tol.values.astype(np.float64)
    fp = df.fp_tol.values.astype(np.float64)

    # ---- A
    m["fn_rate"] = safe_div(fn, n_gt)
    m["fp_rate_neg"] = safe_div(fp, n_valid - n_gt)
    m["fp_rate_all"] = safe_div(fp, n_valid)
    m["fn_px_per_1k_valid"] = safe_div(fn, n_valid) * 1000.0
    m["fp_px_per_1k_valid"] = safe_div(fp, n_valid) * 1000.0
    m["err_px_per_1k_valid"] = m["fn_px_per_1k_valid"] + m["fp_px_per_1k_valid"]
    m["iou_tol"] = df.iou_tol.values.astype(np.float64)

    # ---- B: categorised GT only
    for c in CATS:
        gt_c = ext[f"gt_px_cat_{c}"].values.astype(np.float64)
        fn_c = ext[f"fn_px_cat_{c}"].values.astype(np.float64)
        m[f"fnrate_cat_{c}"] = safe_div(fn_c, gt_c)
        m[f"_gtpx_cat_{c}"] = gt_c
        m[f"_fnpx_cat_{c}"] = fn_c

    # ---- C: GT structure bins
    if w is not None:
        for k in WIDTH_KEYS + QUART_KEYS:
            g = w[f"gt_px_{k}"].values.astype(np.float64)
            f_ = w[f"fn_px_{k}"].values.astype(np.float64)
            m[f"fnrate_{k}"] = safe_div(f_, g)
            m[f"_gtpx_{k}"] = g
            m[f"_fnpx_{k}"] = f_

    # ---- D: FP proxies
    m["median_fp_blob_darkness_pctile"] = \
        ext.median_fp_blob_darkness_pctile.values.astype(np.float64)
    m["median_fp_blob_dist_to_gt_px"] = \
        ext.median_fp_blob_dist_to_gt_px.values.astype(np.float64)
    m["n_fp_blobs"] = ext.n_fp_blobs.values.astype(np.float64)
    m["median_fp_blob_px"] = ext.median_fp_blob_px.values.astype(np.float64)
    m["fp_far_share_px"] = safe_div(df.fp_px_far.values.astype(np.float64), fp)
    m["fp_blobs_far_indicator"] = np.where(
        np.isfinite(m["median_fp_blob_dist_to_gt_px"]),
        (m["median_fp_blob_dist_to_gt_px"] > FP_NEAR_PX).astype(np.float64),
        np.nan)

    # ---- E: boundary versus interior, FULL FRAME (the ext population)
    fn_full = ext.fn_px.values.astype(np.float64)
    fp_full = ext.fp_px.values.astype(np.float64)
    m["fn_boundary_share"] = safe_div(
        ext.fn_px_within_3px_of_gt_edge.values.astype(np.float64), fn_full)
    m["fp_boundary_share"] = safe_div(
        ext.fp_px_within_3px_of_pred_edge.values.astype(np.float64), fp_full)

    # ---- raw counts kept for pooled rates and sanity
    m["_fn_tol"] = fn
    m["_fp_tol"] = fp
    m["_n_gt_fg_valid"] = n_gt
    m["_n_valid_px"] = n_valid
    m["_gt_contrast"] = ext.gt_blob_ring_contrast_median.values.astype(np.float64)
    m["_fn_px_valid_recomp"] = ext.fn_px_valid.values.astype(np.float64)
    m["_uncat_gt_px"] = ext.gt_px_valid_uncategorized.values.astype(np.float64)
    m["_sum_fn_by_cat"] = np.sum(
        [cat_map_col(df, "fn_px_by_cat", c) for c in CATS], axis=0)
    m["_fn_uncat"] = ext.fn_px_valid_uncategorized.values.astype(np.float64)
    return m


# --------------------------------------------------------------- assembly
def build(pairs):
    wb = None
    if os.path.exists(WIDTH_BINS):
        wb = pd.read_parquet(WIDTH_BINS)
    else:
        print(f"NOTE: {WIDTH_BINS} absent; analysis C width bins skipped")
    cells, meta = {}, {}
    for arch, city in pairs:
        lid, uid = loco_id(arch, city), upper_id(arch, city)
        dl, el, wl, rl = load_cell(lid, city, wb)
        du, eu, wu, ru = load_cell(uid, city, wb)
        if list(dl.index) != list(du.index):
            sys.exit(f"STOP: {lid} and {uid} are not evaluated on the same "
                     "image list in the same order")
        # the GT-only columns must be bit-identical across the pair; if they are
        # not, the two cells are not the same evaluation and nothing pairs.
        for col, a, b in (("n_gt_fg_valid", dl.n_gt_fg_valid.values,
                           du.n_gt_fg_valid.values),
                          ("n_valid_px", dl.n_valid_px.values,
                           du.n_valid_px.values),
                          ("gt_px_cat_tree_canopy", el.gt_px_cat_tree_canopy.values,
                           eu.gt_px_cat_tree_canopy.values)):
            if not np.array_equal(a, b):
                sys.exit(f"STOP: GT column {col} differs between {lid} and {uid}")
        if rl.get("mode") != "loco" or ru.get("mode") != "upper":
            sys.exit(f"STOP: mode mismatch {rl.get('mode')} / {ru.get('mode')}")
        cells[(arch, city)] = dict(
            loco=per_image_metrics(dl, el, wl),
            upper=per_image_metrics(du, eu, wu),
            image_ids=list(dl.index))
        meta[(arch, city)] = dict(
            loco_run_id=lid, upper_run_id=uid,
            loco_sha256=sha256_file(os.path.join(DUMPS, lid, "perimage.parquet")),
            upper_sha256=sha256_file(os.path.join(DUMPS, uid, "perimage.parquet")),
            loco_target_access=rl.get("target_access"),
            upper_target_access=ru.get("target_access"),
            metric_module_sha=rl.get("metric_module_sha"),
            n_images=len(dl))
    return cells, meta


# ---------------------------------------------------- contrast quartile bins
def contrast_bins(cells, pairs):
    """Image-level GT contrast quartiles, cut per city.

    gt_blob_ring_contrast_median is a GT-and-image quantity, identical in the
    LOCO cell and its partner, so the bin an image falls in is model-independent.
    """
    edges, assign = {}, {}
    for city in sorted({c for _, c in pairs}):
        arch = next(a for a, c in pairs if c == city)
        v = cells[(arch, city)]["upper"]["_gt_contrast"]
        vv = v[np.isfinite(v)]
        e = [float(np.percentile(vv, q)) for q in (25, 50, 75)] if vv.size else []
        edges[city] = e
        lab = np.full(v.shape, np.nan)
        if e:
            lab = np.where(np.isfinite(v), 1.0, np.nan)
            for x in e:
                lab = np.where(np.isfinite(v) & (v > x), lab + 1, lab)
        assign[city] = lab
    return edges, assign


# ------------------------------------------------------------- the analyses
def delta_arrays(cells, pairs, key, orient="loco_minus_upper"):
    out = {}
    for arch, city in pairs:
        c = cells[(arch, city)]
        if key not in c["loco"] or key not in c["upper"]:
            continue
        a, b = c["loco"][key], c["upper"][key]
        d = (a - b) if orient == "loco_minus_upper" else (b - a)
        out[(arch, city)] = d
    return out


def summarise(cells, pairs, key, orient="loco_minus_upper", mask_by=None,
              worst_by="max_delta"):
    """Per-cell paired bootstraps, per-city and overall pooled means, worst cell.

    `mask_by` optionally restricts to images passing a per-cell boolean mask
    (used for the contrast-quartile bins, which partition images rather than
    pixels).
    """
    d = delta_arrays(cells, pairs, key, orient)
    if not d:
        return None
    per_cell, by_city = [], {}
    for (arch, city), arr in d.items():
        a = np.array(arr, dtype=np.float64)
        if mask_by is not None:
            a = np.where(mask_by[city], a, np.nan)
        st = paired_bootstrap(a)
        lo = cells[(arch, city)]["loco"][key]
        up = cells[(arch, city)]["upper"][key]
        if mask_by is not None:
            lo = np.where(mask_by[city], lo, np.nan)
            up = np.where(mask_by[city], up, np.nan)
        both = np.isfinite(lo) & np.isfinite(up)
        st.update(arch=arch, city=city,
                  loco_mean=(float(np.mean(lo[both])) if both.any() else None),
                  upper_mean=(float(np.mean(up[both])) if both.any() else None),
                  n_excluded=int(np.sum(~both)))
        per_cell.append(st)
        by_city.setdefault(city, []).append(a)
    per_city = {city: pooled_bootstrap({city: arrs})
                for city, arrs in by_city.items()}
    overall = pooled_bootstrap(by_city)
    vals = [(p["mean"], p) for p in per_cell if p["mean"] is not None]
    if vals:
        worst = (max(vals, key=lambda t: t[0])[1] if worst_by == "max_delta"
                 else max(vals, key=lambda t: abs(t[0]))[1])
        worst = dict(arch=worst["arch"], city=worst["city"], mean=worst["mean"],
                     ci95=[worst["ci95_lo"], worst["ci95_hi"]],
                     p_raw=worst["p_raw"], n=worst["n"], worst_by=worst_by)
    else:
        worst = None
    return dict(metric=key, orientation=orient, per_cell=per_cell,
                per_city_mean=per_city, overall_mean=overall, worst_cell=worst)


def pooled_rate(cells, pairs, num_key, den_key, side):
    """Pixel-pooled rate over all images of a cell: sum(num) / sum(den)."""
    out = {}
    for arch, city in pairs:
        m = cells[(arch, city)][side]
        if num_key not in m:
            continue
        num, den = np.nansum(m[num_key]), np.nansum(m[den_key])
        out[(arch, city)] = float(num / den) if den > 0 else None
    return out


def apply_holm(family, scope_key="per_cell"):
    """Holm within a family, applied per cell and to each pooled mean.

    `family` is an ordered dict of member name -> summarise() output.  Members
    are the family; the correction runs across members, separately for every
    cell and for every pooled mean.
    """
    names = list(family)
    if not names:
        return
    cellkeys = [(p["arch"], p["city"]) for p in family[names[0]]["per_cell"]]
    for ck in cellkeys:
        ps, slots = [], []
        for nm in names:
            hit = next((p for p in family[nm]["per_cell"]
                        if (p["arch"], p["city"]) == ck), None)
            ps.append(hit["p_raw"] if hit else None)
            slots.append(hit)
        for slot, ph in zip(slots, holm(ps)):
            if slot is not None:
                slot["p_holm"] = ph
                slot["holm_family"] = names
    for scope in ["overall_mean"]:
        ps = [family[nm][scope]["p_raw"] for nm in names]
        for nm, ph in zip(names, holm(ps)):
            family[nm][scope]["p_holm"] = ph
            family[nm][scope]["holm_family"] = names
    cities = list(family[names[0]]["per_city_mean"])
    for city in cities:
        ps = [family[nm]["per_city_mean"][city]["p_raw"] for nm in names]
        for nm, ph in zip(names, holm(ps)):
            family[nm]["per_city_mean"][city]["p_holm"] = ph


# ------------------------------------------------------------------ sanity
def sanity(cells, meta, pairs):
    out = {}
    # 1. recovered gap: within minus LOCO on iou_tol, must be positive on all 9
    gap = summarise(cells, pairs, "iou_tol", orient="upper_minus_loco")
    out["recovered_gap_iou_tol"] = dict(
        rule="mean paired delta (within - LOCO) must be > 0 in every cell",
        per_cell=[dict(arch=p["arch"], city=p["city"], mean=p["mean"],
                       stderr=p["stderr"], n=p["n"],
                       ci95=[p["ci95_lo"], p["ci95_hi"]], p_raw=p["p_raw"])
                  for p in gap["per_cell"]],
        overall_mean=gap["overall_mean"],
        n_cells_positive=int(sum(1 for p in gap["per_cell"] if p["mean"] > 0)),
        n_cells=len(gap["per_cell"]),
        passed=bool(all(p["mean"] > 0 for p in gap["per_cell"])))

    # 2. fn identity spot check on SPOT_CHECK_N random images per cell
    rng = _rng(BOOT_SEED)
    rows = []
    for (arch, city), c in cells.items():
        n = len(c["image_ids"])
        pick = rng.choice(n, size=min(SPOT_CHECK_N, n), replace=False)
        for side in ("loco", "upper"):
            m = c[side]
            for i in pick:
                lhs = float(m["_sum_fn_by_cat"][i] + m["_fn_uncat"][i])
                rows.append(dict(
                    arch=arch, city=city, side=side,
                    image_id=c["image_ids"][i],
                    sum_fn_by_cat_plus_uncat=lhs,
                    fn_px_valid_recomputed=float(m["_fn_px_valid_recomp"][i]),
                    fn_tol_stored=float(m["_fn_tol"][i]),
                    d_vs_recomputed=lhs - float(m["_fn_px_valid_recomp"][i]),
                    d_vs_stored=lhs - float(m["_fn_tol"][i])))
    dvr = [abs(r["d_vs_recomputed"]) for r in rows]
    dvs = [abs(r["d_vs_stored"]) for r in rows]
    out["fn_identity_spot_check"] = dict(
        rule=("sum(fn_px_by_cat) + fn_px_valid_uncategorized == the RECOMPUTED "
              "fn count exactly; against the STORED fn_tol it may differ by up "
              f"to {TIE_DRIFT_BOUND_PX} px per image from float16 tie collapse"),
        n_checked=len(rows),
        max_abs_diff_vs_recomputed=max(dvr) if dvr else None,
        n_violations_vs_recomputed=int(sum(1 for x in dvr if x != 0)),
        max_abs_diff_vs_stored=max(dvs) if dvs else None,
        n_over_tie_bound_vs_stored=int(sum(1 for x in dvs
                                           if x > TIE_DRIFT_BOUND_PX)),
        rows=rows,
        passed=bool(all(x == 0 for x in dvr)
                    and all(x <= TIE_DRIFT_BOUND_PX for x in dvs)))

    # 3. the denominator rule, per city: how much GT is uncategorised
    unc = {}
    for city in sorted({c for _, c in pairs}):
        arch = next(a for a, c in pairs if c == city)
        m = cells[(arch, city)]["upper"]
        gt_tot = float(np.nansum(m["_n_gt_fg_valid"]))
        gt_unc = float(np.nansum(m["_uncat_gt_px"]))
        unc[city] = dict(
            gt_px_valid_total=gt_tot, gt_px_uncategorized=gt_unc,
            frac_gt_px_uncategorized=(gt_unc / gt_tot) if gt_tot else None,
            n_images_with_uncategorized_gt=int(np.sum(m["_uncat_gt_px"] > 0)),
            n_images=int(m["_uncat_gt_px"].size))
    out["uncategorized_gt_by_city"] = unc

    # 4. category GT mass: a category with no GT pixels in a city has no rate
    catmass = {}
    for city in sorted({c for _, c in pairs}):
        arch = next(a for a, c in pairs if c == city)
        m = cells[(arch, city)]["upper"]
        tot = float(np.nansum(m["_n_gt_fg_valid"]))
        catmass[city] = {
            c: dict(gt_px=float(np.nansum(m[f"_gtpx_cat_{c}"])),
                    frac_of_valid_gt=(float(np.nansum(m[f"_gtpx_cat_{c}"]) / tot)
                                      if tot else None),
                    n_images_with_category=int(np.sum(m[f"_gtpx_cat_{c}"] > 0)))
            for c in CATS}
    out["category_gt_mass_by_city"] = catmass
    out["categories_absent_by_city"] = {
        city: [c for c in CATS if catmass[city][c]["gt_px"] == 0]
        for city in catmass}

    # 4b. the same fact straight off the rasters, so an absent category cannot be
    # an artefact of how the fill pass read the multiclass maps
    import PIL.Image as PImage
    hist = {}
    for city in sorted({c for _, c in pairs}):
        arch = next(a for a, c in pairs if c == city)
        ids = cells[(arch, city)]["image_ids"]
        tot = np.zeros(8, dtype=np.int64)
        nim = np.zeros(8, dtype=np.int64)
        for i in ids:
            mc = np.array(PImage.open(os.path.join(
                DATA, city, "highres", "test", "masks_multiclass", i + ".png")))
            c8 = np.bincount(mc.ravel(), minlength=8)[:8]
            tot += c8
            nim += (c8 > 0)
        hist[city] = dict(px_by_value=[int(x) for x in tot],
                          n_images_by_value=[int(x) for x in nim],
                          n_images=len(ids))
    out["multiclass_raster_histogram"] = dict(
        note="full-frame pixel counts of the multiclass GT raster by value, "
             "0 = background or unrecognised Type, 1-6 = the six categories in "
             "legend order; read directly from the PNGs, independent of the "
             "fill pass",
        legend={str(i + 1): c for i, c in enumerate(CATS)},
        by_city=hist)

    # 5. mean iou_tol per cell, both sides: context for a near-collapsed LOCO cell
    lev = []
    for (arch, city), c in cells.items():
        lev.append(dict(arch=arch, city=city,
                        upper_mean_iou_tol=float(np.nanmean(c["upper"]["iou_tol"])),
                        loco_mean_iou_tol=float(np.nanmean(c["loco"]["iou_tol"]))))
    out["mean_iou_tol_by_cell"] = lev

    # 6. too-clean flags
    flags = []
    for (arch, city), c in cells.items():
        for side in ("loco", "upper"):
            iou = c[side]["iou_tol"]
            v = iou[np.isfinite(iou)]
            if v.size and float(np.std(v)) < 1e-6:
                flags.append(f"{arch}/{city}/{side}: iou_tol has zero variance")
            if v.size and float(np.mean(v)) == 0.0:
                flags.append(f"{arch}/{city}/{side}: mean iou_tol is exactly 0")
            if v.size and float(np.mean(v)) < 0.10:
                flags.append(f"{arch}/{city}/{side}: mean iou_tol "
                             f"{float(np.mean(v)):.4f} < 0.10, near-collapsed cell")
        d = c["loco"]["iou_tol"] - c["upper"]["iou_tol"]
        d = d[np.isfinite(d)]
        if d.size and float(np.max(d)) < 0:
            flags.append(f"{arch}/{city}: LOCO below within on EVERY image "
                         "(perfect separation)")
        fnr = c["loco"]["fn_rate"]
        fnr = fnr[np.isfinite(fnr)]
        if fnr.size and float(np.mean(fnr)) > 0.90:
            flags.append(f"{arch}/{city}/loco: mean fn_rate "
                         f"{float(np.mean(fnr)):.4f} > 0.90, LOCO predicts "
                         "almost no shadow")
    out["too_clean_flags"] = flags
    return out


# ------------------------------------------------------------------ report
def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:.{nd}f}"


def fmt_p(p):
    """A bootstrap p of 0 means no replicate crossed zero: report it as < 1/B."""
    if p is None:
        return "n/a"
    return f"<{1.0/B:.0e}" if p == 0.0 else f"{p:.4g}"


def star(p):
    if p is None:
        return " "
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else " "))


def grid_table(family, nd=4, pkey="p_holm"):
    """rows = 9 cells + per-city means + overall mean + worst cell;
    cols = family members.  Cell text = mean delta with a Holm significance mark."""
    names = list(family)
    head = "| cell | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines = [head, sep]
    order = [(a, c) for c in CITIES for a in ARCHES]
    for arch, city in order:
        cellsv = []
        any_hit = False
        for nm in names:
            hit = next((p for p in family[nm]["per_cell"]
                        if (p["arch"], p["city"]) == (arch, city)), None)
            if hit is None:
                cellsv.append("-")
                continue
            any_hit = True
            cellsv.append(f"{fmt(hit['mean'], nd)}{star(hit.get(pkey))}")
        if any_hit:
            lines.append(f"| {arch}/{city} | " + " | ".join(cellsv) + " |")
    for city in CITIES:
        vals = []
        for nm in names:
            pc = family[nm]["per_city_mean"].get(city)
            vals.append(f"{fmt(pc['mean'], nd)}{star(pc.get('p_holm'))}"
                        if pc else "-")
        if any(v != "-" for v in vals):
            lines.append(f"| **3-arch mean / {city}** | " + " | ".join(vals) + " |")
    vals = [f"{fmt(family[nm]['overall_mean']['mean'], nd)}"
            f"{star(family[nm]['overall_mean'].get('p_holm'))}" for nm in names]
    lines.append("| **mean, all 9 cells** | " + " | ".join(vals) + " |")
    wc = []
    for nm in names:
        w = family[nm]["worst_cell"]
        wc.append(f"{fmt(w['mean'], nd)} ({w['arch']}/{w['city']})" if w else "-")
    lines.append("| **worst cell** | " + " | ".join(wc) + " |")
    return "\n".join(lines)


def levels_table(family, nd=4):
    names = list(family)
    lines = ["| cell | " + " | ".join(f"{n} within / LOCO" for n in names) + " |",
             "|---" * (len(names) + 1) + "|"]
    for city in CITIES:
        for arch in ARCHES:
            vals, any_hit = [], False
            for nm in names:
                hit = next((p for p in family[nm]["per_cell"]
                            if (p["arch"], p["city"]) == (arch, city)), None)
                if hit is None:
                    vals.append("-")
                    continue
                any_hit = True
                vals.append(f"{fmt(hit['upper_mean'], nd)} / {fmt(hit['loco_mean'], nd)}")
            if any_hit:
                lines.append(f"| {arch}/{city} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


# -------------------------------------------------------------------- main
def analysis_A(cells, pairs):
    fam = {"fn_rate": summarise(cells, pairs, "fn_rate"),
           "fp_rate_neg": summarise(cells, pairs, "fp_rate_neg",
                                    worst_by="max_abs_delta")}
    apply_holm(fam)
    aux = {k: summarise(cells, pairs, k)
           for k in ("fp_rate_all", "fn_px_per_1k_valid", "fp_px_per_1k_valid",
                     "err_px_per_1k_valid")}
    # share of the LOCO error increase that is false negatives
    dfn = delta_arrays(cells, pairs, "fn_px_per_1k_valid")
    dfp = delta_arrays(cells, pairs, "fp_px_per_1k_valid")
    share = []
    for (arch, city) in dfn:
        r = ratio_bootstrap(dfn[(arch, city)], dfp[(arch, city)])
        r.update(arch=arch, city=city,
                 d_fn_px_per_1k=float(np.nanmean(dfn[(arch, city)])),
                 d_fp_px_per_1k=float(np.nanmean(dfp[(arch, city)])))
        share.append(r)
    return dict(family=fam, family_name="{fn_rate, fp_rate_neg}", aux=aux,
                fn_share_of_error_increase=share,
                definitions=dict(
                    fn_rate="fn_tol / n_gt_fg_valid, per image, valid region K=2",
                    fp_rate_neg="fp_tol / (n_valid_px - n_gt_fg_valid), the FP "
                                "rate over valid GT-background pixels",
                    fp_rate_all="fp_tol / n_valid_px, FP per valid pixel",
                    fn_share_of_error_increase=
                        "mean(delta fn px per 1k valid) / (mean(delta fn) + "
                        "mean(delta fp)); 1.0 = the LOCO error increase is "
                        "entirely false negatives, 0.0 entirely false positives. "
                        "A share above 1 means the FP delta is negative: LOCO "
                        "makes fewer false positives than the within-city cell, "
                        "so the extra false negatives exceed the total extra "
                        "error. Read it with the two raw deltas beside it.",
                    exclusions="images with n_gt_fg_valid == 0 drop out of "
                               "fn_rate; none had a zero FP denominator"))


def analysis_B(cells, pairs):
    fam = {c: summarise(cells, pairs, f"fnrate_cat_{c}") for c in CATS}
    fam = {c: v for c, v in fam.items() if v is not None}
    apply_holm(fam)
    pooled = {}
    for c in CATS:
        lo = pooled_rate(cells, pairs, f"_fnpx_cat_{c}", f"_gtpx_cat_{c}", "loco")
        up = pooled_rate(cells, pairs, f"_fnpx_cat_{c}", f"_gtpx_cat_{c}", "upper")
        rows = []
        for k in lo:
            rel = (None if not up[k] else (lo[k] - up[k]) / up[k])
            rows.append(dict(arch=k[0], city=k[1], pooled_within=up[k],
                             pooled_loco=lo[k],
                             abs_inflation=(None if lo[k] is None or up[k] is None
                                            else lo[k] - up[k]),
                             rel_inflation=rel))
        pooled[c] = rows
    gtmass = {}
    for c in CATS:
        tot = {}
        for arch, city in pairs:
            m = cells[(arch, city)]["upper"]
            tot.setdefault(city, float(np.nansum(m[f"_gtpx_cat_{c}"])))
        gtmass[c] = tot
    return dict(family=fam, family_name="the 6 shadow categories",
                pooled_rates=pooled, gt_px_by_city=gtmass,
                definitions=dict(
                    per_image="fn_px_cat_c / gt_px_cat_c, images with "
                              "gt_px_cat_c == 0 excluded from that category",
                    pooled="sum(fn_px_cat_c) / sum(gt_px_cat_c) over the cell's "
                           "150 images",
                    denominator="CATEGORISED GT ONLY; GT pixels whose multiclass "
                                "value is 0 are in no category and are excluded"))


def analysis_C(cells, pairs, cq_assign, cq_edges):
    famw = {k: summarise(cells, pairs, f"fnrate_{k}") for k in WIDTH_KEYS}
    famw = {k: v for k, v in famw.items() if v is not None}
    if famw:
        apply_holm(famw)
    famq = {k: summarise(cells, pairs, f"fnrate_{k}") for k in QUART_KEYS}
    famq = {k: v for k, v in famq.items() if v is not None}
    if famq:
        apply_holm(famq)
    famc = {}
    for i, key in enumerate(CQ_KEYS, start=1):
        mask = {city: (cq_assign[city] == i) for city in cq_assign}
        famc[key] = summarise(cells, pairs, "fn_rate", mask_by=mask)
    apply_holm(famc)
    gtmass = {}
    for scope, keys in (("fixed", WIDTH_KEYS), ("quartile", QUART_KEYS)):
        per_city = {}
        for arch, city in pairs:
            if city in per_city:
                continue
            m = cells[(arch, city)]["upper"]
            if f"_gtpx_{keys[0]}" not in m:
                continue
            tot = sum(float(np.nansum(m[f"_gtpx_{k}"])) for k in keys)
            per_city[city] = {k: (float(np.nansum(m[f"_gtpx_{k}"])) / tot
                                  if tot else None) for k in keys}
        gtmass[scope] = per_city
    return dict(gt_px_share_by_bin=gtmass,
                width_family=famw, width_family_name="the 3 GT skeleton-width bins",
                width_quartile_family=famq,
                width_quartile_family_name="the 4 GT width quartile bins",
                contrast_family=famc,
                contrast_family_name="the 4 GT contrast quartile bins",
                contrast_edges_px=cq_edges,
                definitions=dict(
                    width_bins="GT blob skeleton width, 2 x median(EDT) along "
                               "the skeleton, binned <3 / 3-8 / >8 px; "
                               "GT-only, so identical across the pair",
                    width_quartiles="per-city pixel-weighted quartiles of the "
                                    "same GT width; supplementary, because the "
                                    "<3 px bin holds almost no GT pixels",
                    width_rate="fn_px in the bin / gt_px in the bin, per image; "
                               "images with no GT pixels in the bin excluded",
                    contrast_bins="image-level gt_blob_ring_contrast_median, "
                                  "quartiles cut per city; the metric inside a "
                                  "contrast bin is the image fn_rate",
                    caveat="the width-bin FN counts decompose the RECOMPUTED "
                           "binarisation, not the stored fn_tol"))


def analysis_D(cells, pairs):
    keys = ["median_fp_blob_darkness_pctile", "median_fp_blob_dist_to_gt_px",
            "n_fp_blobs", "median_fp_blob_px"]
    fam = {k: summarise(cells, pairs, k, worst_by="max_abs_delta") for k in keys}
    apply_holm(fam)
    aux = {k: summarise(cells, pairs, k, worst_by="max_abs_delta")
           for k in ("fp_far_share_px", "fp_blobs_far_indicator")}
    nearfar = []
    for arch, city in pairs:
        c = cells[(arch, city)]
        row = dict(arch=arch, city=city)
        for side in ("upper", "loco"):
            d = c[side]["median_fp_blob_dist_to_gt_px"]
            ok = np.isfinite(d)
            row[f"{side}_n_images_with_fp_blobs"] = int(ok.sum())
            row[f"{side}_n_near"] = int(np.sum(ok & (d <= FP_NEAR_PX)))
            row[f"{side}_n_far"] = int(np.sum(ok & (d > FP_NEAR_PX)))
        nearfar.append(row)
    return dict(family=fam, family_name="the 4 FP proxies", aux=aux,
                near_far_image_counts=nearfar,
                definitions=dict(
                    population="perimage_ext.parquet FP blob columns are FULL "
                               "FRAME, not the valid region",
                    exclusions="images with no FP blob have NaN for the three "
                               "median_fp_* columns and drop out of those "
                               "deltas; n is reported per cell",
                    near_far=f"image-level split on median_fp_blob_dist_to_gt_px "
                             f"at {FP_NEAR_PX} px",
                    fp_far_share_px=f"fp_px_far / fp_tol, the share of FP pixels "
                                    f"more than {FP_NEAR_GT_PX} px from a GT "
                                    f"shadow blob, valid region"))


def analysis_E(cells, pairs):
    fam = {"fn_boundary_share": summarise(cells, pairs, "fn_boundary_share",
                                          worst_by="max_abs_delta"),
           "fp_boundary_share": summarise(cells, pairs, "fp_boundary_share",
                                          worst_by="max_abs_delta")}
    apply_holm(fam)
    return dict(family=fam, family_name="{fn_boundary_share, fp_boundary_share}",
                definitions=dict(
                    fn_boundary_share="fn_px_within_3px_of_gt_edge / fn_px, "
                                      "full frame",
                    fp_boundary_share="fp_px_within_3px_of_pred_edge / fp_px, "
                                      "full frame",
                    population="FULL FRAME, matching how the ext columns were "
                               "written; a 2 px band fragments error blobs",
                    exclusions="images with zero FN (or zero FP) pixels drop out "
                               "of that share"))


def main():
    pairs = [(a, c) for c in CITIES for a in ARCHES]
    if DRY_RUN:
        pairs = [p for p in pairs if p in DRY_PAIRS]
        print(f"DRY RUN: pairs={pairs}, nothing written\n")

    cells, meta = build(pairs)
    cq_edges, cq_assign = contrast_bins(cells, pairs)
    print("contrast quartile edges (GT ring contrast, per city):", cq_edges, "\n")

    A = analysis_A(cells, pairs)
    Bres = analysis_B(cells, pairs)
    C = analysis_C(cells, pairs, cq_assign, cq_edges)
    D = analysis_D(cells, pairs)
    E = analysis_E(cells, pairs)
    S = sanity(cells, meta, pairs)

    for title, fam in (("A  FN/FP decomposition (LOCO - within)", A["family"]),
                       ("B  FN rate by category (LOCO - within)", Bres["family"]),
                       ("C  FN rate by GT width bin", C["width_family"]),
                       ("C  FN rate by GT width quartile", C["width_quartile_family"]),
                       ("C  FN rate by GT contrast quartile", C["contrast_family"]),
                       ("D  FP proxies (LOCO - within)", D["family"]),
                       ("E  boundary share (LOCO - within)", E["family"])):
        if not fam:
            continue
        print(f"\n### {title}")
        print(grid_table(fam))
        print("\nlevels (within / LOCO):")
        print(levels_table(fam))

    print("\n### A  FN share of the error increase")
    for r in A["fn_share_of_error_increase"]:
        print(f"  {r['arch']}/{r['city']}: share_FN={fmt(r['point'],3)} "
              f"CI[{fmt(r['ci95_lo'],3)}, {fmt(r['ci95_hi'],3)}] "
              f"dFN={fmt(r['d_fn_px_per_1k'],2)} dFP={fmt(r['d_fp_px_per_1k'],2)} "
              f"px/1k valid")

    print("\n### sanity")
    print(f"  recovered gap positive in "
          f"{S['recovered_gap_iou_tol']['n_cells_positive']}/"
          f"{S['recovered_gap_iou_tol']['n_cells']} cells "
          f"(passed={S['recovered_gap_iou_tol']['passed']})")
    for p in S["recovered_gap_iou_tol"]["per_cell"]:
        print(f"    {p['arch']}/{p['city']}: {fmt(p['mean'])} "
              f"CI[{fmt(p['ci95'][0])}, {fmt(p['ci95'][1])}] p={p['p_raw']:.4g} n={p['n']}")
    sc = S["fn_identity_spot_check"]
    print(f"  fn identity: {sc['n_checked']} checks, "
          f"max|d| vs recomputed={sc['max_abs_diff_vs_recomputed']}, "
          f"violations={sc['n_violations_vs_recomputed']}, "
          f"max|d| vs stored={sc['max_abs_diff_vs_stored']} "
          f"(bound {TIE_DRIFT_BOUND_PX}), passed={sc['passed']}")
    for city, u in S["uncategorized_gt_by_city"].items():
        print(f"  {city}: uncategorised GT {u['frac_gt_px_uncategorized']*100:.2f}% "
              f"of valid GT px, {u['n_images_with_uncategorized_gt']}/"
              f"{u['n_images']} images")
    print(f"  too-clean flags: {S['too_clean_flags'] or 'none'}")

    if DRY_RUN:
        print("\nDRY RUN complete, nothing written.")
        return

    if not S["recovered_gap_iou_tol"]["passed"]:
        print("\nWARNING: the recovered-gap anchor check did not pass in every "
              "cell; results are still written and the failure is recorded.")

    write_outputs(A, Bres, C, D, E, S, meta, pairs, cq_edges)


def jsonable(o):
    if isinstance(o, dict):
        return {(f"{k[0]}/{k[1]}" if isinstance(k, tuple) else str(k)):
                jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def sign_counts(fam, name, alpha=0.05, pkey="p_holm"):
    """How many of the family's cells move up, down, and significantly."""
    pc = fam[name]["per_cell"]
    ok = [p for p in pc if p["mean"] is not None]
    sig = [p for p in ok if p.get(pkey) is not None and p[pkey] < alpha]
    return dict(n=len(pc),
                pos=sum(1 for p in ok if p["mean"] > 0),
                neg=sum(1 for p in ok if p["mean"] < 0),
                sig=len(sig),
                sig_pos=sum(1 for p in sig if p["mean"] > 0),
                sig_neg=sum(1 for p in sig if p["mean"] < 0))


def top_by(fam, key="overall_mean", n=1, absolute=False):
    """The n family members with the largest (or largest absolute) pooled mean."""
    items = [(nm, fam[nm][key]["mean"]) for nm in fam
             if fam[nm][key]["mean"] is not None]
    items.sort(key=lambda t: (abs(t[1]) if absolute else t[1]), reverse=True)
    return items[:n]


def write_outputs(A, Bres, C, D, E, S, meta, pairs, cq_edges):
    os.makedirs(OUTDIR, exist_ok=True)
    runs = sorted({m[k] for m in meta.values()
                   for k in ("loco_run_id", "upper_run_id")})
    unc = S["uncategorized_gt_by_city"]
    unc_line = "; ".join(
        f"{c}: {u['frac_gt_px_uncategorized']*100:.2f}% of valid GT px in "
        f"{u['n_images_with_uncategorized_gt']}/{u['n_images']} images"
        for c, u in unc.items())

    denom_rule = (
        "Per-category rates (analysis B) use CATEGORISED GT ONLY. GT shadow "
        "pixels whose multiclass value is 0 carry no recoverable category "
        "(get_shadow_type_value maps an unrecognised annotation Type to 0, the "
        "same value as background) and are absent from gt_px_by_cat, so they "
        "are excluded from every per-category numerator and denominator. "
        "Excluded fraction per city (valid region, K=2): " + unc_line + ". "
        "Analyses A, C, D and E use the full valid-region or full-frame counts "
        "and are not affected.")

    caveats = [
        denom_rule,
        f"Tie drift: the structure and category columns were filled from a "
        f"re-binarisation of float16 probabilities, so summing them against the "
        f"stored fn_tol can disagree by up to {TIE_DRIFT_BOUND_PX} px per image "
        f"out of ~120k valid px. The drift is one-sided (predictions only lose "
        f"foreground) and is documented in analysis/fill_pass_report.json. "
        f"Analysis C's width-bin counts and analysis B's category counts inherit "
        f"it; analyses A, D and E read stored columns and do not.",
        "Single seed (s0) and single rung (r450), high resolution only. No "
        "seed-to-seed variance is available for these 18 cells, so every "
        "interval here is over images, not over training runs.",
        "LOCO here is method=vanilla (no adaptation); the within-city upper "
        "bound is method=none. No adaptation method is in this comparison.",
        "The FP blob columns and the boundary/interior columns (analyses D and "
        "E) are FULL FRAME, not the K=2 valid region, because a 2 px don't-care "
        "band fragments error blobs and corrupts their geometry.",
        "Images with no FP blob have NaN in the three median_fp_* columns and "
        "drop out of those deltas; the surviving n is reported per cell.",
        "The <3 px GT skeleton-width bin is nearly empty on this data: a 1 px "
        "wide blob already has skeleton width 2.0 and sub-10 px annotations were "
        "dropped at mask-build time. The per-city GT width quartile bins are "
        "reported alongside it for that reason.",
        "The GT width quartile edges were cut after the GT width distribution "
        "was inspected but before any FN count was computed; they are "
        "model-independent and error-blind.",
        "Analysis B is only well supported for building_canyon and tree_canopy, "
        "which together hold about 96% of valid GT shadow pixels in every city. "
        "thin_linear has ZERO GT pixels in all three test pools (confirmed "
        "against the multiclass rasters, not only the dump columns) and has no "
        "rate at all; topography appears in 1 chicago image, 2 miami images and "
        "8-9 phoenix images; under_structure and vehicle hold under 2.5% of GT "
        "pixels. Deltas in those columns rest on very little GT mass.",
    ]

    out = dict(
        id="gj_error_taxonomy",
        arm="G+J",
        answers=["zNBu-W6", "zNBu-M1"],
        question=("Where do the models fail, by error type, shadow category, GT "
                  "structure and error location, and does leave-one-city-out "
                  "transfer move where they fail?"),
        runs=runs,
        index_sha256=None,
        index_note=("dumps/INDEX.parquet does not exist on this box; the 18 "
                    "cells were selected by explicit run_id, never by a glob, "
                    "and each cell's perimage.parquet sha256 is recorded below."),
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        script="analysis/gj_analysis/gj_analysis.py",
        support_script="analysis/gj_analysis/gj_width_bins.py",
        git_hash=git_hash(),
        cells=jsonable(meta),
        statistics=dict(
            test="paired bootstrap on per-image deltas", B=B, seed=BOOT_SEED,
            two_tailed=True, ci="percentile 2.5 / 97.5",
            p_rule="min(2 * prop_wrong, 1.0)",
            p_floor=f"a reported p of 0.0 means no bootstrap replicate crossed "
                    f"zero, i.e. p < 1/B = {1.0/B:g}",
            multiplicity="Holm-Bonferroni within each family, applied per cell, "
                         "per city mean, and to the 9-cell pooled mean",
            pooled_mean="unweighted mean over cells of the per-cell mean delta; "
                        "the CI resamples each city's 150 images once per "
                        "replicate and reuses that resample for all three archs",
            pairing="LOCO and within-city cells share the same 150 test images; "
                    "every delta is paired per image_id"),
        denominator_rule=denom_rule,
        analyses=dict(A=jsonable(A), B=jsonable(Bres), C=jsonable(C),
                      D=jsonable(D), E=jsonable(E)),
        sanity=jsonable(S),
        headline=(
            "Across the 9 (architecture, city) cells the LOCO false-negative "
            f"rate rises by {fmt(A['family']['fn_rate']['overall_mean']['mean'], 4)} "
            "and the false-positive rate over valid background by "
            f"{fmt(A['family']['fp_rate_neg']['overall_mean']['mean'], 4)}, "
            "with the false-negative share of the extra error at "
            f"{fmt(np.mean([r['point'] for r in A['fn_share_of_error_increase'] if r['point'] is not None]), 3)} "
            "on the 9-cell mean."),
        caveats=caveats,
        figure=None,
        caption=None)

    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {RESULTS_JSON}")

    # ------------------------------------------------------------ FINDINGS.md
    L = []
    L.append("# Arms G and J — error taxonomy (reviewer zNBu-W6)\n")
    L.append(f"Generated {out['generated_utc']} by `{out['script']}`. "
             f"18 cells: {{mamnet, oglanet, dinov3}} x {{chicago, miami, "
             f"phoenix}} x {{vanilla LOCO, within-city upper bound}}, "
             f"r450 / hr / s0, 150 test images per cell.\n")
    L.append("Every delta is **LOCO minus within-city**, paired per image on the "
             "same 150 images. Paired bootstrap, B = 10,000, percentile 95% CI, "
             "two-tailed p, Holm-Bonferroni within each family. Marks in the "
             "tables are Holm-corrected: `*` p<0.05, `**` p<0.01, `***` p<0.001. "
             "Full numbers, raw and corrected p, n, and CIs are in "
             "`results.json`.\n")
    L.append("## Denominator rule\n")
    L.append(denom_rule + "\n")
    L.append("## Tie-drift caveat\n")
    L.append(caveats[1] + "\n")
    L.append("## Populations\n")
    L.append("- Analyses A, B, C: K=2 valid region, matching "
             "`tp_tol`/`fp_tol`/`fn_tol`.\n"
             "- Analyses D and E: full frame, matching how the FP blob and "
             "boundary columns in `perimage_ext.parquet` were written.\n")

    def block(title, fam, note, levels=True, nd=4):
        L.append(f"\n## {title}\n")
        L.append(grid_table(fam, nd=nd) + "\n")
        ns = [p["n"] for nm in fam for p in fam[nm]["per_cell"]
              if p.get("n") is not None]
        if ns:
            L.append(f"\nImages contributing to each delta: n from {min(ns)} to "
                     f"{max(ns)} of 150 across the cells of this family; "
                     f"per-delta n, stderr, 95% CI, raw p and Holm p are in "
                     f"`results.json`.\n")
        if levels:
            L.append("\nLevels, within / LOCO, each the mean over the "
                     "contributing images of that image's own rate:\n")
            L.append(levels_table(fam, nd=nd) + "\n")
        L.append("\n" + note + "\n")

    # ---- A
    fnm = A["family"]["fn_rate"]["overall_mean"]["mean"]
    fpm = A["family"]["fp_rate_neg"]["overall_mean"]["mean"]
    shares = {(r["arch"], r["city"]): r["point"]
              for r in A["fn_share_of_error_increase"] if r["point"] is not None}
    sh_by_city = {c: np.mean([v for (a, cc), v in shares.items() if cc == c])
                  for c in {c for _, c in shares}}
    hi = max(sh_by_city, key=sh_by_city.get) if sh_by_city else None
    lo = min(sh_by_city, key=sh_by_city.get) if sh_by_city else None
    scn, scp = sign_counts(A["family"], "fn_rate"), \
        sign_counts(A["family"], "fp_rate_neg")
    sh_lo = min(shares.items(), key=lambda t: t[1])
    sh_hi = max(shares.items(), key=lambda t: t[1])
    within = {c: (max(v for (a, cc), v in shares.items() if cc == c)
                  - min(v for (a, cc), v in shares.items() if cc == c))
              for c in {c for _, c in shares}}
    max_within_city = max(within, key=within.get)
    max_within = within[max_within_city]
    between = max(sh_by_city.values()) - min(sh_by_city.values())
    block("A. FN/FP decomposition of the gap",
          A["family"],
          f"On the 9-cell mean the FN rate rises by {fmt(fnm)} (up in "
          f"{scn['pos']}/9 cells, Holm-significant in {scn['sig_pos']}) and the "
          f"FP rate over valid background by {fmt(fpm)} (up in {scp['pos']}/9, "
          f"Holm-significant in {scp['sig_pos']}); the worst FN cell is "
          f"{A['family']['fn_rate']['worst_cell']['arch']}/"
          f"{A['family']['fn_rate']['worst_cell']['city']} at "
          f"{fmt(A['family']['fn_rate']['worst_cell']['mean'])}. "
          f"The FN share of the extra error runs from {fmt(sh_lo[1], 3)} "
          f"({sh_lo[0][0]}/{sh_lo[0][1]}) to {fmt(sh_hi[1], 3)} "
          f"({sh_hi[0][0]}/{sh_hi[0][1]}); its spread across the three "
          f"architectures within one city reaches {fmt(max_within, 3)} "
          f"(in {max_within_city}), against {fmt(between, 3)} across the three "
          f"3-arch city means.")
    L.append("\nTotal error and its two halves, LOCO minus within, in pixels per "
             "1000 valid pixels, 9-cell mean:\n")
    L.append("| measure | mean delta | 95% CI | p (raw) | worst cell |")
    L.append("|---|---|---|---|---|")
    for k in ("fn_px_per_1k_valid", "fp_px_per_1k_valid", "err_px_per_1k_valid",
              "fp_rate_all"):
        v = A["aux"][k]["overall_mean"]
        w = A["aux"][k]["worst_cell"]
        L.append(f"| {k} | {fmt(v['mean'], 3)} | [{fmt(v['ci95_lo'], 3)}, "
                 f"{fmt(v['ci95_hi'], 3)}] | {fmt_p(v['p_raw'])} | "
                 f"{fmt(w['mean'], 3)} ({w['arch']}/{w['city']}) |")
    L.append("\nFN share of the LOCO error increase, per cell "
             "(1.0 = entirely false negatives, 0.0 = entirely false positives; "
             "above 1.0 means LOCO also makes fewer false positives):\n")
    L.append("| cell | d FN px/1k valid | d FP px/1k valid | FN share | 95% CI |")
    L.append("|---|---|---|---|---|")
    for r in A["fn_share_of_error_increase"]:
        L.append(f"| {r['arch']}/{r['city']} | {fmt(r['d_fn_px_per_1k'], 2)} | "
                 f"{fmt(r['d_fp_px_per_1k'], 2)} | {fmt(r['point'], 3)} | "
                 f"[{fmt(r['ci95_lo'], 3)}, {fmt(r['ci95_hi'], 3)}] |")
    L.append("")

    # ---- B
    cm = S["category_gt_mass_by_city"]
    well = [c for c in CATS
            if all(cm[city][c]["frac_of_valid_gt"] is not None
                   and cm[city][c]["frac_of_valid_gt"] >= 0.01
                   and cm[city][c]["n_images_with_category"] >= 10
                   for city in cm)]
    thin = [c for c in CATS if c not in well]
    topB = top_by({k: v for k, v in Bres["family"].items() if k in well}, n=2)
    scb = {k: sign_counts(Bres["family"], k) for k in well}
    obsB = ("No category has a pooled mean delta." if not topB else
            f"Among the two categories with support in every city "
            f"({', '.join(well)}), which together hold about 96% of valid GT "
            f"shadow pixels, the 9-cell mean FN-rate inflation is "
            + ", ".join(f"{nm} {fmt(v)} (Holm-significant in "
                        f"{scb[nm]['sig_pos']}/9 cells)" for nm, v in topB)
            + f". The other four columns rest on under 2.5% of valid GT pixels "
              f"or on 1-9 images in a city, and thin_linear has zero GT pixels "
              f"in all three test pools, so no rate exists for it.")
    block("B. FN rate by shadow category (categorised GT only)",
          Bres["family"], obsB)
    L.append("\nCategory support, share of that city's valid GT shadow pixels "
             "and number of images containing the category:\n")
    L.append("| city | " + " | ".join(CATS) + " |")
    L.append("|---" * (len(CATS) + 1) + "|")
    for city in CITIES:
        if city not in cm:
            continue
        L.append(f"| {city} | " + " | ".join(
            (f"{cm[city][c]['frac_of_valid_gt']*100:.2f}% / "
             f"{cm[city][c]['n_images_with_category']} img"
             if cm[city][c]["frac_of_valid_gt"] is not None else "n/a")
            for c in CATS) + " |")
    L.append("")

    # ---- C
    if C["width_family"]:
        wm = [C["width_family"][k]["overall_mean"]["mean"] for k in WIDTH_KEYS
              if k in C["width_family"]]
        shares_w = C["gt_px_share_by_bin"]["fixed"]
        lt3 = np.mean([shares_w[c]["w_lt3"] for c in shares_w]) * 100
        block("C1. FN rate by GT blob skeleton width (fixed bins)",
              C["width_family"],
              f"The three fixed bins move together: 9-cell mean inflations of "
              + ", ".join(f"{k} {fmt(C['width_family'][k]['overall_mean']['mean'])}"
                          for k in WIDTH_KEYS if k in C["width_family"])
              + f", a spread of {fmt(max(wm) - min(wm))}. The <3 px bin holds "
                f"{lt3:.2f}% of valid GT shadow pixels on average across the "
                f"three cities, so it is the weakest column here and the "
                f"quartile bins below are the usable width axis.")
    if C["width_quartile_family"]:
        q = {k: C["width_quartile_family"][k]["overall_mean"]["mean"]
             for k in QUART_KEYS if k in C["width_quartile_family"]}
        percity = {c: (C["width_quartile_family"]["wq4"]["per_city_mean"][c]["mean"]
                       - C["width_quartile_family"]["wq1"]["per_city_mean"][c]["mean"])
                   for c in C["width_quartile_family"]["wq1"]["per_city_mean"]}
        block("C2. FN rate by GT width quartile (per-city, pixel-weighted)",
              C["width_quartile_family"],
              f"wq1 is the thinnest quarter of GT shadow pixels in that city and "
              f"wq4 the thickest; the 9-cell means are "
              + ", ".join(f"{k} {fmt(v)}" for k, v in q.items())
              + f". The thick-minus-thin difference of the 3-arch city means is "
              + ", ".join(f"{c} {fmt(v)}" for c, v in percity.items())
              + ", so the width ordering is not the same in every city.")
    cq = {k: C["contrast_family"][k]["overall_mean"]["mean"] for k in CQ_KEYS}
    lowest = min(cq, key=cq.get)
    cq1_top_upper = cq1_top_loco = 0
    for arch, city in pairs:
        lv = {}
        for k in CQ_KEYS:
            hit = next(p for p in C["contrast_family"][k]["per_cell"]
                       if (p["arch"], p["city"]) == (arch, city))
            lv[k] = (hit["upper_mean"], hit["loco_mean"])
        if max(CQ_KEYS, key=lambda k: lv[k][0]) == "cq1":
            cq1_top_upper += 1
        if max(CQ_KEYS, key=lambda k: lv[k][1]) == "cq1":
            cq1_top_loco += 1
    block("C3. FN rate by GT contrast quartile (image-level)",
          C["contrast_family"],
          f"cq1 is the lowest-contrast quarter of images in that city and cq4 "
          f"the highest; the 9-cell means are "
          + ", ".join(f"{k} {fmt(v)}" for k, v in cq.items())
          + f", with the smallest inflation in {lowest}. The FN rate level is "
            f"highest in the lowest-contrast quartile in {cq1_top_upper}/9 cells "
            f"within-city and {cq1_top_loco}/9 under LOCO; the per-city quartile "
            f"edges are in `results.json` under "
            f"`analyses.C.contrast_edges_px`.")

    # ---- D
    scd = {k: sign_counts(D["family"], k) for k in D["family"]}
    nb = D["family"]["n_fp_blobs"]["overall_mean"]
    dk = D["family"]["median_fp_blob_darkness_pctile"]["overall_mean"]
    sz = D["family"]["median_fp_blob_px"]["overall_mean"]
    ds = D["family"]["median_fp_blob_dist_to_gt_px"]["overall_mean"]
    far = D["aux"]["fp_far_share_px"]["overall_mean"]
    block("D. FP proxies", D["family"],
          f"Under LOCO the FP blob count falls in "
          f"{scd['n_fp_blobs']['neg']}/9 cells (9-cell mean {fmt(nb['mean'], 2)} "
          f"blobs, Holm-significant in {scd['n_fp_blobs']['sig_neg']}) while the "
          f"median FP blob is larger in {scd['median_fp_blob_px']['pos']}/9 "
          f"(mean {fmt(sz['mean'], 2)} px), and the darkness percentile "
          f"({fmt(dk['mean'], 2)}) and distance to GT ({fmt(ds['mean'], 2)} px) "
          f"move in both directions with 9-cell means that are not significant. "
          f"The share of FP pixels more than {FP_NEAR_GT_PX} px from a GT blob "
          f"shifts by {fmt(far['mean'])} (95% CI [{fmt(far['ci95_lo'])}, "
          f"{fmt(far['ci95_hi'])}], p={fmt_p(far['p_raw'])}).",
          nd=3)
    L.append("\nAuxiliary FP location measures, LOCO minus within, 9-cell mean:\n")
    L.append("| measure | mean delta | 95% CI | p (raw) |")
    L.append("|---|---|---|---|")
    for k, v in D["aux"].items():
        om = v["overall_mean"]
        L.append(f"| {k} | {fmt(om['mean'], 4)} | [{fmt(om['ci95_lo'], 4)}, "
                 f"{fmt(om['ci95_hi'], 4)}] | {fmt_p(om['p_raw'])} |")
    L.append("")

    # ---- E
    sce = {k: sign_counts(E["family"], k) for k in E["family"]}
    efn = E["family"]["fn_boundary_share"]["overall_mean"]
    efp = E["family"]["fp_boundary_share"]["overall_mean"]
    lv_fn = [p["upper_mean"] for p in E["family"]["fn_boundary_share"]["per_cell"]
             if p["upper_mean"] is not None]
    lv_fp = [p["upper_mean"] for p in E["family"]["fp_boundary_share"]["per_cell"]
             if p["upper_mean"] is not None]
    block("E. Boundary versus interior", E["family"],
          f"Shares are of that image's own error pixels lying within 3 px of the "
          f"relevant edge, full frame. The FN boundary share falls in "
          f"{sce['fn_boundary_share']['neg']}/9 cells (9-cell mean "
          f"{fmt(efn['mean'])}, Holm-significant in "
          f"{sce['fn_boundary_share']['sig_neg']}) and the FP boundary share in "
          f"{sce['fp_boundary_share']['neg']}/9 (mean {fmt(efp['mean'])}, "
          f"Holm-significant in {sce['fp_boundary_share']['sig_neg']}), off "
          f"within-city levels of {fmt(min(lv_fn), 3)}-{fmt(max(lv_fn), 3)} and "
          f"{fmt(min(lv_fp), 3)}-{fmt(max(lv_fp), 3)}.")

    # ---- sanity
    L.append("\n## Sanity checks\n")
    rg = S["recovered_gap_iou_tol"]
    L.append(f"**Recovered gap (anchor).** Mean paired delta of `iou_tol`, "
             f"within minus LOCO, positive in {rg['n_cells_positive']}/"
             f"{rg['n_cells']} cells (required: all).\n")
    L.append("| cell | within - LOCO iou_tol | 95% CI | p | n |")
    L.append("|---|---|---|---|---|")
    for p in rg["per_cell"]:
        L.append(f"| {p['arch']}/{p['city']} | {fmt(p['mean'])} | "
                 f"[{fmt(p['ci95'][0])}, {fmt(p['ci95'][1])}] | "
                 f"{fmt_p(p['p_raw'])} | {p['n']} |")
    om = rg["overall_mean"]
    L.append(f"| **mean, all 9 cells** | {fmt(om['mean'])} | "
             f"[{fmt(om['ci95_lo'])}, {fmt(om['ci95_hi'])}] | "
             f"{fmt_p(om['p_raw'])} | 9 cells |")
    L.append("\nMean `iou_tol` per cell, within / LOCO:\n")
    L.append("| cell | within | LOCO |")
    L.append("|---|---|---|")
    for r in S["mean_iou_tol_by_cell"]:
        L.append(f"| {r['arch']}/{r['city']} | {fmt(r['upper_mean_iou_tol'])} | "
                 f"{fmt(r['loco_mean_iou_tol'])} |")
    L.append("\n**GT mass per category, per city** (share of valid GT shadow "
             "pixels; a category with 0 px has no rate in that city):\n")
    L.append("| city | " + " | ".join(CATS) + " |")
    L.append("|---" * (len(CATS) + 1) + "|")
    for city, cm in S["category_gt_mass_by_city"].items():
        L.append(f"| {city} | " + " | ".join(
            f"{cm[c]['frac_of_valid_gt']*100:.2f}%"
            if cm[c]["frac_of_valid_gt"] is not None else "n/a"
            for c in CATS) + " |")
    absent = {c: v for c, v in S["categories_absent_by_city"].items() if v}
    L.append(f"\nCategories with zero GT pixels: "
             f"{absent if absent else 'none'}.\n")
    mh = S["multiclass_raster_histogram"]
    L.append("\n**Multiclass GT raster histogram**, read straight from the PNGs "
             "(full frame), so an absent category cannot be an artefact of the "
             "fill pass. Values 1-6 are "
             + ", ".join(f"{i+1}={c}" for i, c in enumerate(CATS)) + ":\n")
    L.append("| city | " + " | ".join(f"v{i}" for i in range(7)) + " |")
    L.append("|---" * 8 + "|")
    for city, h in mh["by_city"].items():
        L.append(f"| {city} px | " + " | ".join(str(h["px_by_value"][i])
                                                for i in range(7)) + " |")
        L.append(f"| {city} images | " + " | ".join(str(h["n_images_by_value"][i])
                                                    for i in range(7)) + " |")
    L.append("")
    sc = S["fn_identity_spot_check"]
    L.append(f"\n**FN identity spot check.** {sc['n_checked']} image-sides "
             f"({SPOT_CHECK_N} random images per cell, both sides): "
             f"`sum(fn_px_by_cat) + fn_px_valid_uncategorized` matches the "
             f"recomputed FN count with {sc['n_violations_vs_recomputed']} "
             f"violations (max |diff| {sc['max_abs_diff_vs_recomputed']}); "
             f"against the stored `fn_tol` the max |diff| is "
             f"{sc['max_abs_diff_vs_stored']} px, bound "
             f"{TIE_DRIFT_BOUND_PX} px. Passed: {sc['passed']}.\n")
    L.append("\n**Uncategorised GT, per city** (the excluded fraction of the "
             "analysis B denominator):\n")
    L.append("| city | valid GT px | uncategorised px | fraction | images affected |")
    L.append("|---|---|---|---|---|")
    for c, u in unc.items():
        L.append(f"| {c} | {int(u['gt_px_valid_total'])} | "
                 f"{int(u['gt_px_uncategorized'])} | "
                 f"{u['frac_gt_px_uncategorized']*100:.2f}% | "
                 f"{u['n_images_with_uncategorized_gt']}/{u['n_images']} |")
    L.append(f"\n**Too-clean flags:** "
             f"{'; '.join(S['too_clean_flags']) if S['too_clean_flags'] else 'none'}.\n")
    L.append("\n## Caveats\n")
    for c in caveats:
        L.append(f"- {c}")
    L.append("")

    with open(FINDINGS_MD, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {FINDINGS_MD}")


if __name__ == "__main__":
    main()
