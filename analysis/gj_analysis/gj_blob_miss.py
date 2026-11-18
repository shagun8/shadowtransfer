"""Arms G/J extension — whole-blob miss rate, the direct test of "under LOCO the
models miss ENTIRE shadows".

Analysis E in the main G/J pass measured that the FN boundary share FALLS in 9/9
cells: a smaller fraction of the missed pixels sits within 3 px of a GT edge.
That is indirect. If errors were still edge trims the share would hold; that it
falls is consistent with whole shadows going missing, but it does not count
them. This pass counts them.

  whole-blob miss rate = (GT shadow blobs with zero overlapping predicted
                          foreground px) / (GT shadow blobs)

per image, per cell side, on the FULL FRAME.

Definitions fixed before any number was looked at:
  - GT blobs are 8-connected components of the binary GT mask (multiclass > 0)
    AFTER the <10 px component filter, exactly the population
    gj_fill_pass.py:process_image counts into `n_gt_blobs`. The connectivity, the
    filter and the order of the two are copied from that function, not chosen
    here. Sub-10 px annotations were already dropped at mask-build time, so the
    filter is close to a no-op and is kept only for exact agreement.
  - Predictions come from metrics.threshold_and_filter(prob, min_pixels=10),
    imported from the staged module after a sha256 check. The threshold is never
    hardcoded and the function is never reimplemented (CONTRACT.md rule zero).
  - PRIMARY: missed = zero overlapping predicted px.
    SENSITIVITY: missed_10 = overlapping predicted px < 10% of the blob's area.
    Both are reported everywhere; the primary is the headline.

Per blob we also record area, majority-vote category and skeleton width, so the
miss rate by size and by category falls out of the same pass rather than needing
a second one.

GT blob geometry does not depend on the model, so it is computed ONCE per
(city, image) and reused for all six cells of that city. That is a 6x saving and
it also guarantees the LOCO and within-city sides of a pair see a bit-identical
blob set, which is what makes the per-image delta paired.

Statistics follow CONVENTIONS.md and reuse the bootstrap helpers of
analysis/gj_analysis/gj_analysis.py verbatim (paired bootstrap on per-image
deltas, B = 10,000, percentile CI, two-tailed p, Holm within family,
image-cluster pooled bootstrap) so the intervals are comparable with the main
results JSON.

Outputs:
  results/17_gj_error_taxonomy_ext.json
  analysis/gj_analysis/FINDINGS-EXT.md

Analysis only. No figures. Nothing under dumps/ is written.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# The system dist-packages on this box ships numpy-1.x-compiled numexpr and
# bottleneck that abort under a numpy 2.x venv. Drop it; every package needed
# here resolves from the interpreter's own environment. Under the operator's
# myenv this filter removes a path that is not on sys.path and is a no-op.
sys.path = [p for p in sys.path if p != "/usr/lib/python3/dist-packages"]

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- config
BASE = os.environ["PROJECT_ROOT"]
DUMPS = os.path.join(BASE, "dumps", "st")
DATA = os.path.join(os.environ["PROJECT_ROOT"], "data/Final_data_test")
OUTDIR = os.path.join(BASE, "analysis", "gj_analysis")
LEGEND_PATH = os.path.join(BASE, "analysis", "category_legend.json")
MAIN_JSON = os.path.join(BASE, "results", "17_gj_error_taxonomy.json")
RESULTS_JSON = os.path.join(BASE, "results", "17_gj_error_taxonomy_ext.json")
FINDINGS_MD = os.path.join(OUTDIR, "FINDINGS-EXT.md")
BLOB_PARQUET = os.path.join(OUTDIR, "blob_miss_perimage.parquet")
BLOB_TABLE_PARQUET = os.path.join(OUTDIR, "blob_miss_perblob.parquet")

STAGING = os.environ.get("ST_METRICS_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
METRICS_PATH = os.path.join(STAGING, "shadowtransfer", "metrics.py")
EXPECTED_METRIC_SHA_PREFIX = "730ad4a5"

B = 10000
BOOT_SEED = 42
MIN_PIXELS = 10           # the <10 px connected-component filter
N_IMAGES_EXPECTED = 150
TIE_DRIFT_BOUND_PX = 21   # analysis/fill_pass_report.json, one-sided
MISS_10_FRAC = 0.10       # the sensitivity definition

# Fixed GT blob AREA bins, in px, cut before any overlap was computed. They are
# model-independent and error-blind. The edges follow the n_gt_blobs_lt20 /
# lt50 / lt200 columns the fill pass already wrote, extended with a top bin.
AREA_EDGES = (50.0, 200.0, 1000.0)
AREA_KEYS = ("a_lt50", "a_50to200", "a_200to1000", "a_gt1000")
AREA_LABELS = {"a_lt50": "10-50 px", "a_50to200": "50-200 px",
               "a_200to1000": "200-1000 px", "a_gt1000": ">1000 px"}

ARCHES = ["mamnet", "oglanet", "dinov3"]
CITIES = ["chicago", "miami", "phoenix"]
CATS = ["building_canyon", "under_structure", "tree_canopy",
        "topography", "vehicle", "thin_linear"]


def loco_id(arch, city):
    return f"{arch}-vanilla-loco-r450-{city}-hr-s0"


def upper_id(arch, city):
    return f"{arch}-none-upper-r450-{city}-hr-s0"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.join(BASE, "final_paper"), "rev-parse",
             "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


# ------------------------------------------- first action: verify the module
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        sys.exit(f"STOP: metric module absent at {METRICS_PATH}")
    got = sha256_file(METRICS_PATH)
    if not got.startswith(EXPECTED_METRIC_SHA_PREFIX):
        sys.exit(f"STOP: metric sha does not start {EXPECTED_METRIC_SHA_PREFIX}\n"
                 f"  got {got}\nDo not reimplement metrics.")
    if STAGING not in sys.path:
        sys.path.insert(0, STAGING)
    import shadowtransfer.metrics as M
    print(f"metric module sha256 OK: {got}")
    print(f"metric module imported: {M.__file__}")
    return M, got


M, METRIC_SHA = load_metrics()

import cv2  # noqa: E402  (metrics.py already requires it)
from scipy.ndimage import distance_transform_edt  # noqa: E402
from skimage.morphology import skeletonize  # noqa: E402
import PIL.Image as PImage  # noqa: E402

with open(LEGEND_PATH) as f:
    LEGEND = json.load(f)
CAT_INT_TO_KEY = {int(k): v for k, v in LEGEND["int_to_contract_key"].items()}


# ------------------------------------------------------- bootstrap helpers
# Copied verbatim from analysis/gj_analysis/gj_analysis.py so the intervals in
# this JSON are comparable with the ones in the main results JSON.
def _rng(seed=BOOT_SEED):
    return np.random.default_rng(seed)


def paired_bootstrap(delta, B=B, seed=BOOT_SEED):
    delta = np.asarray(delta, dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    n = delta.size
    if n == 0:
        return dict(mean=None, median=None, stderr=None, ci95_lo=None,
                    ci95_hi=None, p_raw=None, n=0, B=B)
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
    """Unweighted mean over cells of the per-cell mean delta, image-clustered CI.

    One resample of a city's image index per replicate, reused across that
    city's architectures, because they are evaluated on the same images.
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
    if not cell_means:
        return dict(mean=None, stderr=None, ci95_lo=None, ci95_hi=None,
                    p_raw=None, n_cells=0, B=B)
    observed = float(np.mean(cell_means))
    boots = np.nanmean(np.vstack(boot_cols), axis=0)
    boots = boots[np.isfinite(boots)]
    prop_wrong = (float(np.mean(boots <= 0)) if observed >= 0
                  else float(np.mean(boots >= 0)))
    return dict(mean=observed, stderr=float(np.std(boots, ddof=1)),
                ci95_lo=float(np.percentile(boots, 2.5)),
                ci95_hi=float(np.percentile(boots, 97.5)),
                p_raw=float(min(2.0 * prop_wrong, 1.0)),
                n_cells=len(cell_means), B=B,
                stderr_basis="bootstrap sd over replicates of the cell mean")


def holm(pvals):
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


def rate_ratio_bootstrap(miss_l, tot_l, miss_u, tot_u, B=B, seed=BOOT_SEED):
    """Blob-weighted pooled rates and their ratio, resampling IMAGES.

    Inputs are per-image blob counts, aligned on the same images for the two
    sides. The pooled rate is sum(missed) / sum(blobs) over the images, so a
    replicate resamples whole images and recomputes both sums. The ratio is
    LOCO's pooled rate over the within-city pooled rate.
    """
    miss_l = np.asarray(miss_l, float); tot_l = np.asarray(tot_l, float)
    miss_u = np.asarray(miss_u, float); tot_u = np.asarray(tot_u, float)
    ok = np.isfinite(miss_l) & np.isfinite(tot_l) & np.isfinite(miss_u) \
        & np.isfinite(tot_u) & (tot_l > 0)
    miss_l, tot_l, miss_u, tot_u = miss_l[ok], tot_l[ok], miss_u[ok], tot_u[ok]
    n = miss_l.size
    if n == 0:
        return dict(rate_loco=None, rate_upper=None, ratio=None, n=0, B=B)
    r_l = float(miss_l.sum() / tot_l.sum())
    r_u = float(miss_u.sum() / tot_u.sum())
    rng = _rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    bl = miss_l[idx].sum(axis=1) / tot_l[idx].sum(axis=1)
    bu = miss_u[idx].sum(axis=1) / tot_u[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = bl / bu
    br = br[np.isfinite(br)]
    return dict(
        rate_loco=r_l, rate_upper=r_u,
        ratio=(float(r_l / r_u) if r_u > 0 else None),
        rate_loco_ci95=[float(np.percentile(bl, 2.5)),
                        float(np.percentile(bl, 97.5))],
        rate_upper_ci95=[float(np.percentile(bu, 2.5)),
                         float(np.percentile(bu, 97.5))],
        ratio_ci95=([float(np.percentile(br, 2.5)),
                     float(np.percentile(br, 97.5))] if br.size else None),
        n_images=int(n), B=B,
        definition=("blob-weighted pooled rate = sum(missed blobs) / "
                    "sum(GT blobs) over the cell's images; the CI resamples "
                    "images, not blobs, because blobs inside one image are "
                    "not independent"))


# --------------------------------------------------------------- geometry
def _blobs(mask):
    """8-connected components, matching gj_fill_pass.py:_blobs."""
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8),
                                            connectivity=8)


def _filter_small(mask, min_pixels=MIN_PIXELS):
    """The <10 px filter, gj_fill_pass.py:_filter_small."""
    out = mask.astype(np.uint8).copy()
    n, lab, stats, _ = _blobs(out)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_pixels:
            out[lab == i] = 0
    return out


def _skeleton_width(blob, edt_full):
    """2 x median EDT along the skeleton, gj_fill_pass.py:_skeleton_width."""
    skel = skeletonize(blob)
    vals = edt_full[skel]
    if vals.size == 0:
        vals = edt_full[blob]
    return 2.0 * float(np.median(vals)) if vals.size else float("nan")


def area_key(a):
    if a < AREA_EDGES[0]:
        return "a_lt50"
    if a < AREA_EDGES[1]:
        return "a_50to200"
    if a < AREA_EDGES[2]:
        return "a_200to1000"
    return "a_gt1000"


def load_rasters(city, image_id):
    """Byte-for-byte the loader gj_fill_pass.py:load_rasters uses."""
    p = os.path.join(DATA, city, "highres", "test")
    gt = (np.array(PImage.open(os.path.join(p, "masks", image_id + ".png")))
          > 0).astype(np.uint8)
    mc = np.array(PImage.open(os.path.join(p, "masks_multiclass",
                                           image_id + ".png")))
    return gt, mc


def gt_blob_table(city, image_id):
    """GT blob geometry for one image. Model-independent, so computed once.

    Returns (label image, list of per-blob dicts). Blob i in the list is
    label i + 1 in the label image.
    """
    gt, mc = load_rasters(city, image_id)
    gtb = gt.astype(bool)
    if not gtb.any():
        return np.zeros(gt.shape, np.int32), []
    gt_f = _filter_small(gtb).astype(bool)
    if not gt_f.any():
        return np.zeros(gt.shape, np.int32), []
    n, lab, stats, _ = _blobs(gt_f)
    edt = distance_transform_edt(gt_f)
    blobs = []
    for i in range(1, n):
        blob = (lab == i)
        area = int(stats[i, cv2.CC_STAT_AREA])
        vals = mc[blob]
        vals = vals[vals > 0]
        if vals.size:
            cat = CAT_INT_TO_KEY.get(int(np.bincount(vals).argmax()), None)
        else:
            cat = "uncategorized"
        blobs.append(dict(label=i, area=area, cat=cat,
                          width=_skeleton_width(blob, edt),
                          area_bin=area_key(area)))
    return lab.astype(np.int32), blobs


def per_image_miss(lab, blobs, predb):
    """Overlap of every GT blob with the predicted foreground.

    np.bincount over the label image restricted to predicted-positive pixels
    gives every blob's overlap in one pass, which is what keeps 2700 images
    cheap.
    """
    if not blobs:
        return None
    n_lab = len(blobs) + 1
    ov = np.bincount(lab[predb].ravel(), minlength=n_lab)[1:n_lab]
    areas = np.array([b["area"] for b in blobs], dtype=np.int64)
    missed = (ov == 0)
    missed10 = (ov < MISS_10_FRAC * areas)
    return ov, areas, missed, missed10


# ------------------------------------------------------------------ driver
def run():
    meta_cells, rows, blob_rows = {}, [], []
    viol_fn, viol_count = [], []

    for city in CITIES:
        # One pass over the city's 150 images for GT geometry, reused by all six
        # cells of that city.
        img_ids = None
        geom = {}
        for arch in ARCHES:
            df = pd.read_parquet(os.path.join(DUMPS, loco_id(arch, city),
                                              "perimage.parquet"))
            ids = list(df["image_id"])
            if img_ids is None:
                img_ids = ids
            elif ids != img_ids:
                sys.exit(f"STOP: image_id order differs across archs in {city}")
        assert len(img_ids) == N_IMAGES_EXPECTED, (city, len(img_ids))
        print(f"[{city}] GT blob geometry over {len(img_ids)} images", flush=True)
        for k, iid in enumerate(img_ids):
            geom[iid] = gt_blob_table(city, iid)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(img_ids)}", flush=True)

        for arch in ARCHES:
            for side, rid in (("loco", loco_id(arch, city)),
                              ("upper", upper_id(arch, city))):
                cdir = os.path.join(DUMPS, rid)
                per = pd.read_parquet(os.path.join(cdir, "perimage.parquet")
                                      ).set_index("image_id")
                ext = pd.read_parquet(os.path.join(cdir, "perimage_ext.parquet")
                                      ).set_index("image_id")
                meta_cells[rid] = dict(
                    arch=arch, city=city, side=side,
                    perimage_sha256=sha256_file(
                        os.path.join(cdir, "perimage.parquet")),
                    probs_sha256=sha256_file(os.path.join(cdir, "probs.npz")))
                with np.load(os.path.join(cdir, "probs.npz")) as z:
                    for iid in img_ids:
                        lab, blobs = geom[iid]
                        pred = M.threshold_and_filter(z[iid],
                                                      min_pixels=MIN_PIXELS)
                        predb = pred.astype(bool)
                        r = dict(run_id=rid, arch=arch, city=city, side=side,
                                 image_id=iid)
                        res = per_image_miss(lab, blobs, predb)
                        if res is None:
                            r.update(n_blobs=0, n_missed=0, n_missed10=0,
                                     miss_rate=np.nan, miss_rate10=np.nan,
                                     missed_blob_px=0)
                            rows.append(r)
                            continue
                        ov, areas, missed, missed10 = res
                        r.update(
                            n_blobs=int(len(blobs)),
                            n_missed=int(missed.sum()),
                            n_missed10=int(missed10.sum()),
                            miss_rate=float(missed.mean()),
                            miss_rate10=float(missed10.mean()),
                            missed_blob_px=int(areas[missed].sum()))
                        rows.append(r)

                        # sanity (a): every pixel of a fully missed blob is a
                        # full-frame false negative, so the missed-blob pixel
                        # count can never exceed the stored fn_px column.
                        fn_px = float(ext.loc[iid, "fn_px"])
                        if r["missed_blob_px"] > fn_px + TIE_DRIFT_BOUND_PX:
                            viol_fn.append(dict(
                                run_id=rid, image_id=iid,
                                missed_blob_px=r["missed_blob_px"],
                                fn_px=fn_px,
                                excess=r["missed_blob_px"] - fn_px))
                        # sanity (b): the blob population must match the count
                        # the fill pass wrote.
                        n_stored = int(per.loc[iid, "n_gt_blobs"])
                        if n_stored != len(blobs):
                            viol_count.append(dict(
                                run_id=rid, image_id=iid,
                                n_gt_blobs_stored=n_stored,
                                n_blobs_here=len(blobs)))

                        # per-blob rows, once per (city, image), on the LOCO
                        # side and the upper side separately so miss rate by
                        # size and by category can be cut per side.
                        for j, b in enumerate(blobs):
                            blob_rows.append(dict(
                                run_id=rid, arch=arch, city=city, side=side,
                                image_id=iid, area=b["area"],
                                area_bin=b["area_bin"], cat=b["cat"],
                                width=b["width"], overlap_px=int(ov[j]),
                                missed=bool(missed[j]),
                                missed10=bool(missed10[j])))
                print(f"[{city}] {rid} done", flush=True)

    return (pd.DataFrame(rows), pd.DataFrame(blob_rows), meta_cells,
            viol_fn, viol_count)


# ------------------------------------------------------------- summarising
def summarise_family(per, key):
    """Per-cell / per-city-mean / 9-cell-mean paired deltas of one per-image key."""
    fam = dict(metric=key, orientation="loco_minus_upper", per_cell=[])
    deltas_by_city = {c: [] for c in CITIES}
    for city in CITIES:
        for arch in ARCHES:
            lo = per[(per.arch == arch) & (per.city == city)
                     & (per.side == "loco")].set_index("image_id")[key]
            up = per[(per.arch == arch) & (per.city == city)
                     & (per.side == "upper")].set_index("image_id")[key]
            common = lo.index.intersection(up.index)
            dl = (lo.loc[common] - up.loc[common]).to_numpy(float)
            st = paired_bootstrap(dl)
            st.update(arch=arch, city=city,
                      loco_mean=float(np.nanmean(lo.loc[common])),
                      upper_mean=float(np.nanmean(up.loc[common])),
                      n_excluded=int((~np.isfinite(dl)).sum()))
            fam["per_cell"].append(st)
            deltas_by_city[city].append(dl)
    fam["per_city_mean"] = {
        c: pooled_bootstrap({c: deltas_by_city[c]}) for c in CITIES}
    fam["overall_mean"] = pooled_bootstrap(deltas_by_city)

    # Mean LEVELS beside the deltas, so a figure can plot a city's within and
    # LOCO position without deriving either from the per-cell rows. Same
    # unweighted-mean-over-cells convention the deltas use.
    for c in CITIES:
        rows = [r for r in fam["per_cell"] if r["city"] == c]
        fam["per_city_mean"][c]["upper_mean"] = float(
            np.mean([r["upper_mean"] for r in rows]))
        fam["per_city_mean"][c]["loco_mean"] = float(
            np.mean([r["loco_mean"] for r in rows]))
        fam["per_city_mean"][c]["level_basis"] = (
            "unweighted mean over the city's 3 architecture cells of that "
            "cell's mean per-image rate")
    fam["overall_mean"]["upper_mean"] = float(
        np.mean([r["upper_mean"] for r in fam["per_cell"]]))
    fam["overall_mean"]["loco_mean"] = float(
        np.mean([r["loco_mean"] for r in fam["per_cell"]]))
    fam["overall_mean"]["level_basis"] = (
        "unweighted mean over all 9 cells of that cell's mean per-image rate")
    finite = [r for r in fam["per_cell"] if r["mean"] is not None]
    w = max(finite, key=lambda r: r["mean"])
    fam["worst_cell"] = dict(arch=w["arch"], city=w["city"], mean=w["mean"],
                             ci95=[w["ci95_lo"], w["ci95_hi"]],
                             worst_by="max_delta")
    return fam


def apply_holm_family(fams):
    """Holm within the family, applied per cell, per city mean and to the mean."""
    names = list(fams)
    for i in range(len(fams[names[0]]["per_cell"])):
        ps = [fams[k]["per_cell"][i]["p_raw"] for k in names]
        for k, ph in zip(names, holm(ps)):
            fams[k]["per_cell"][i]["p_holm"] = ph
            fams[k]["per_cell"][i]["holm_family"] = names
    for city in CITIES:
        ps = [fams[k]["per_city_mean"][city]["p_raw"] for k in names]
        for k, ph in zip(names, holm(ps)):
            fams[k]["per_city_mean"][city]["p_holm"] = ph
    ps = [fams[k]["overall_mean"]["p_raw"] for k in names]
    for k, ph in zip(names, holm(ps)):
        fams[k]["overall_mean"]["p_holm"] = ph
        fams[k]["overall_mean"]["holm_family"] = names
    return fams


def pooled_rates(per, key_missed="n_missed"):
    """Blob-weighted pooled rate per cell side, plus the LOCO / within ratio."""
    out = []
    for city in CITIES:
        for arch in ARCHES:
            lo = per[(per.arch == arch) & (per.city == city)
                     & (per.side == "loco")].set_index("image_id")
            up = per[(per.arch == arch) & (per.city == city)
                     & (per.side == "upper")].set_index("image_id")
            common = lo.index.intersection(up.index)
            r = rate_ratio_bootstrap(
                lo.loc[common, key_missed], lo.loc[common, "n_blobs"],
                up.loc[common, key_missed], up.loc[common, "n_blobs"])
            r.update(arch=arch, city=city)
            out.append(r)
    return out


def strata_rates(blobs, by, miss_col="missed"):
    """Blob-weighted miss rate per stratum per side, pooled over the 9 cells.

    A stratum's rate is over blobs, not images: it answers "of the GT blobs of
    this size / category, what fraction did the side miss entirely". The CI
    resamples IMAGES, because blobs inside one image are not independent, and
    the SAME image resample drives both sides so the ratio's interval is paired
    rather than a quotient of two independent intervals.
    """
    out = {}
    for lvl, g in blobs.groupby(by, observed=True):
        piv = g.pivot_table(index=["city", "image_id"], columns="side",
                            values=miss_col, aggfunc=["sum", "size"])
        if ("sum", "loco") not in piv or ("sum", "upper") not in piv:
            continue
        ml = piv[("sum", "loco")].to_numpy(float)
        tl = piv[("size", "loco")].to_numpy(float)
        mu = piv[("sum", "upper")].to_numpy(float)
        tu = piv[("size", "upper")].to_numpy(float)
        ok = np.isfinite(ml) & np.isfinite(mu) & (tl > 0) & (tu > 0)
        ml, tl, mu, tu = ml[ok], tl[ok], mu[ok], tu[ok]
        n = ml.size
        if n == 0:
            continue
        r_l = float(ml.sum() / tl.sum())
        r_u = float(mu.sum() / tu.sum())
        rng = _rng(BOOT_SEED)
        idx = rng.integers(0, n, size=(B, n))
        bl = ml[idx].sum(axis=1) / tl[idx].sum(axis=1)
        bu = mu[idx].sum(axis=1) / tu[idx].sum(axis=1)
        bd = bl - bu
        with np.errstate(divide="ignore", invalid="ignore"):
            br = bl / bu
        br = br[np.isfinite(br)]
        rec = {
            "loco": dict(rate=r_l,
                         ci95=[float(np.percentile(bl, 2.5)),
                               float(np.percentile(bl, 97.5))],
                         n_blobs=int(tl.sum())),
            "upper": dict(rate=r_u,
                          ci95=[float(np.percentile(bu, 2.5)),
                                float(np.percentile(bu, 97.5))],
                          n_blobs=int(tu.sum())),
            "difference": dict(
                mean=r_l - r_u,
                ci95=[float(np.percentile(bd, 2.5)),
                      float(np.percentile(bd, 97.5))],
                p_raw=float(min(2.0 * (np.mean(bd <= 0) if r_l - r_u >= 0
                                       else np.mean(bd >= 0)), 1.0))),
            "n_images": int(n), "B": B,
        }
        if r_u > 0:
            rec["ratio"] = r_l / r_u
            rec["ratio_ci95"] = ([float(np.percentile(br, 2.5)),
                                  float(np.percentile(br, 97.5))]
                                 if br.size else None)
        out[str(lvl)] = rec
    return out


def figure_support(per, main_json):
    """Panel selection for the qualitative appendix figure, by rule.

    The rule is fixed here, in the analysis, so the figure script cannot choose
    its own images and the annotated per-panel counts are results-JSON numbers
    rather than caption-only ones.

      rows 1-3, one per city: the architecture with the largest recovered gap in
        that city (from the main JSON's sanity block), and the image maximising
        RECOVERED BLOBS = blobs the within-city model caught that LOCO missed
        entirely. That is the largest visible difference between the two
        prediction columns.
      row 4, the severe case: MAMNet/Phoenix, the image maximising the
        within-to-LOCO increase in whole-blob miss RATE among images holding at
        least 10 GT blobs. Rows 1-3 maximise an absolute count and this row
        maximises a proportion, so the two rules cannot pick the same image for
        the same reason; the 10-blob floor stops a 2-blob image winning on a
        single miss.
    """
    gapc = {}
    for r in main_json["sanity"]["recovered_gap_iou_tol"]["per_cell"]:
        gapc.setdefault(r["city"], []).append((r["mean"], r["arch"]))
    pick = {c: max(v)[1] for c, v in gapc.items()}

    def panel(arch, city, key, min_blobs=0):
        lo = per[(per.arch == arch) & (per.city == city)
                 & (per.side == "loco")].set_index("image_id")
        up = per[(per.arch == arch) & (per.city == city)
                 & (per.side == "upper")].set_index("image_id")
        common = lo.index.intersection(up.index)
        lo, up = lo.loc[common], up.loc[common]
        ok = lo["n_blobs"] >= max(min_blobs, 1)
        lo, up = lo[ok], up[ok]
        if key == "recovered_count":
            score = lo["n_missed"] - up["n_missed"]
        elif key == "rate_increase":
            score = lo["miss_rate"] - up["miss_rate"]
        else:
            raise ValueError(key)
        iid = score.idxmax()
        return dict(
            arch=arch, city=city, image_id=str(iid), selection_key=key,
            min_gt_blobs_required=int(min_blobs),
            n_gt_blobs=int(lo.loc[iid, "n_blobs"]),
            n_missed_upper=int(up.loc[iid, "n_missed"]),
            n_missed_loco=int(lo.loc[iid, "n_missed"]),
            n_recovered=int(lo.loc[iid, "n_missed"] - up.loc[iid, "n_missed"]),
            miss_rate_upper=float(up.loc[iid, "miss_rate"]),
            miss_rate_loco=float(lo.loc[iid, "miss_rate"]),
            n_candidate_images=int(len(lo)),
            recovered_gap_iou_tol=float(
                [r["mean"] for r in
                 main_json["sanity"]["recovered_gap_iou_tol"]["per_cell"]
                 if r["arch"] == arch and r["city"] == city][0]))

    rows = [panel(pick[c], c, "recovered_count") for c in CITIES]
    rows.append(panel("mamnet", "phoenix", "rate_increase", min_blobs=10))
    return dict(
        architecture_by_city=pick,
        architecture_rule=("the architecture with the largest recovered gap in "
                           "that city, mean paired within-minus-LOCO iou_tol "
                           "from results/17_gj_error_taxonomy.json "
                           "sanity.recovered_gap_iou_tol"),
        image_rule_rows_1_3=("the image maximising blobs the within-city model "
                             "caught that LOCO missed entirely"),
        image_rule_row_4=("MAMNet/Phoenix, the image maximising the "
                          "within-to-LOCO increase in whole-blob miss rate "
                          "among images holding at least 10 GT blobs"),
        panels=rows)


def main():
    t_start = datetime.now(timezone.utc)
    per, blobs, meta_cells, viol_fn, viol_count = run()
    per.to_parquet(BLOB_PARQUET, index=False)
    blobs.to_parquet(BLOB_TABLE_PARQUET, index=False)
    print(f"wrote {BLOB_PARQUET}")
    print(f"wrote {BLOB_TABLE_PARQUET}")

    fams = apply_holm_family({
        "whole_blob_miss_rate": summarise_family(per, "miss_rate"),
        "whole_blob_miss_rate_10pct": summarise_family(per, "miss_rate10"),
    })

    main_json = json.load(open(MAIN_JSON))
    out = {
        "id": "17_gj_error_taxonomy_ext",
        "arm": "G+J",
        "answers": ["zNBu-W6"],
        "question": ("Under leave-one-city-out, do the models miss ENTIRE "
                     "ground-truth shadows rather than trimming their edges?"),
        "extends": "results/17_gj_error_taxonomy.json",
        "runs": sorted(meta_cells),
        "cells": meta_cells,
        "generated_utc": t_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "analysis/gj_analysis/gj_blob_miss.py",
        "git_hash": git_hash(),
        "metric_module_sha256": METRIC_SHA,
        "index_sha256": None,
        "index_note": main_json.get("index_note"),
        "statistics": {
            "test": "paired bootstrap on per-image deltas",
            "B": B, "seed": BOOT_SEED, "two_tailed": True,
            "ci": "percentile 2.5 / 97.5",
            "p_rule": "min(2 * prop_wrong, 1.0)",
            "p_floor": ("a reported p of 0.0 means no bootstrap replicate "
                        "crossed zero, i.e. p < 1/B = 0.0001"),
            "multiplicity": ("Holm-Bonferroni within the 2-metric family "
                             "{whole_blob_miss_rate, whole_blob_miss_rate_10pct}"
                             ", applied per cell, per city mean and to the "
                             "9-cell mean"),
            "pooled_mean": ("unweighted mean over cells of the per-cell mean "
                            "delta; the CI resamples each city's 150 images "
                            "once per replicate and reuses that resample for "
                            "all three archs"),
            "pairing": ("LOCO and within-city cells share the same 150 test "
                        "images AND the same GT blob set per image, since GT "
                        "blob geometry is model-independent and is computed "
                        "once per (city, image)"),
            "blob_weighted_rates": ("pooled rate = sum(missed blobs) / "
                                    "sum(GT blobs) over images; CI resamples "
                                    "images because blobs within an image are "
                                    "not independent"),
        },
        "definitions": {
            "gt_blob": ("8-connected component of the binary GT mask "
                        "(multiclass > 0) after the <10 px component filter, "
                        "full frame; the population gj_fill_pass.py counts "
                        "into n_gt_blobs"),
            "whole_blob_miss_rate": ("per image, GT blobs with ZERO "
                                     "overlapping predicted foreground px "
                                     "divided by GT blobs; the primary "
                                     "definition"),
            "whole_blob_miss_rate_10pct": ("same, with missed redefined as "
                                           "overlapping predicted px < 10% of "
                                           "the blob's area; the sensitivity "
                                           "check"),
            "population": ("FULL FRAME, matching the blob and geometry columns "
                           "of perimage_ext.parquet; a 2 px don't-care band "
                           "fragments blobs and corrupts their geometry"),
            "binarization": ("metrics.threshold_and_filter(prob, "
                             "min_pixels=10), imported from the staged module; "
                             "argmax over [1-p, p] so p > 0.5 is foreground and "
                             "p == 0.5 resolves to background, then the <10 px "
                             "8-connectivity filter"),
            "blob_category": ("majority vote of the multiclass raster values "
                              "> 0 inside the blob; a blob whose pixels are all "
                              "value 0 is labelled uncategorized"),
            "blob_width": ("2 x median of the Euclidean distance transform "
                           "along the blob's skeleton, the estimator "
                           "gj_fill_pass.py:_skeleton_width uses"),
            "area_bins": ("fixed at 10-50 / 50-200 / 200-1000 / >1000 px, cut "
                          "before any overlap was computed; model-independent "
                          "and error-blind"),
            "exclusions": ("images with zero GT blobs after the <10 px filter "
                           "have no miss rate and drop out of that cell's "
                           "delta"),
        },
        "family": fams,
        "pooled_rates": {
            "whole_blob_miss_rate": pooled_rates(per, "n_missed"),
            "whole_blob_miss_rate_10pct": pooled_rates(per, "n_missed10"),
        },
        "by_area_bin": strata_rates(blobs, "area_bin"),
        "by_category": strata_rates(blobs, "cat"),
        "by_area_bin_10pct": strata_rates(blobs, "area_bin",
                                          miss_col="missed10"),
        "by_category_10pct": strata_rates(blobs, "cat", miss_col="missed10"),
        "area_bin_order": list(AREA_KEYS),
        "area_bin_labels": AREA_LABELS,
    }
    out["figure_support"] = figure_support(per, main_json)

    # ------------------------------------------------------------- overall
    lo = per[per.side == "loco"]
    up = per[per.side == "upper"]
    out["overall_pooled"] = rate_ratio_bootstrap(
        lo.groupby(["city", "image_id"])["n_missed"].sum(),
        lo.groupby(["city", "image_id"])["n_blobs"].sum(),
        up.groupby(["city", "image_id"])["n_missed"].sum(),
        up.groupby(["city", "image_id"])["n_blobs"].sum())
    out["overall_pooled"]["note"] = (
        "pooled over all 9 cells at once: the numerator sums missed blobs over "
        "the three architectures of an image, so an image contributes three "
        "times. Read the per-cell pooled_rates for the cell-level picture.")

    # -------------------------------------------------------------- sanity
    n_img_rows = len(per)
    out["sanity"] = {
        "missed_blob_px_vs_stored_fn_px": {
            "rule": ("every pixel of a fully missed blob is a full-frame false "
                     "negative, so missed_blob_px <= fn_px + 21 px tie drift"),
            "n_checked": int(n_img_rows),
            "n_violations": len(viol_fn),
            "violations": viol_fn[:20],
            "tie_drift_bound_px": TIE_DRIFT_BOUND_PX,
        },
        "blob_count_vs_stored_n_gt_blobs": {
            "rule": ("the GT blob population here must equal the n_gt_blobs "
                     "column the fill pass wrote for the same image"),
            "n_checked": int(n_img_rows),
            "n_violations": len(viol_count),
            "violations": viol_count[:20],
            "note": ("n_gt_blobs lives in perimage.parquet, not "
                     "perimage_ext.parquet as the task text had it; the ext "
                     "parquet carries n_gt_blobs_lt20/lt50/lt200"),
        },
        "images_with_zero_gt_blobs": {
            c: int(((per.city == c) & (per.n_blobs == 0)
                    & (per.side == "loco") & (per.arch == "mamnet")).sum())
            for c in CITIES},
        "gt_blob_set_identical_across_sides": {
            "rule": ("GT geometry is computed once per (city, image) and "
                     "reused, so the two sides of a pair see the same blob "
                     "set by construction"),
            "holds": True},
        "n_blobs_total": int(blobs[(blobs.side == "loco")].shape[0]),
    }

    lo_m = fams["whole_blob_miss_rate"]["overall_mean"]
    op = out["overall_pooled"]
    out["headline"] = (
        f"Across the 9 (architecture, city) cells the per-image whole-blob miss "
        f"rate rises by {lo_m['mean']:.4f} under LOCO "
        f"(95% CI [{lo_m['ci95_lo']:.4f}, {lo_m['ci95_hi']:.4f}]); pooled over "
        f"all cells and weighted by blob, {op['rate_upper']:.4f} of GT shadow "
        f"blobs are missed entirely by the within-city upper bound against "
        f"{op['rate_loco']:.4f} under LOCO, a ratio of {op['ratio']:.2f}.")
    out["caveats"] = [
        ("Single seed (s0) and single rung (r450), high resolution only, so "
         "every interval here is over images and not over training runs."),
        ("LOCO is method=vanilla with no adaptation and the within-city upper "
         "bound is method=none; target access differs by condition, "
         "source_only against target_labels."),
        ("Full frame, not the K=2 valid region: a 2 px don't-care band "
         "fragments blobs and corrupts their geometry. The FN pixel counts in "
         "the main results JSON's analysis A are valid-region and are not "
         "directly comparable to a full-frame blob count."),
        ("A blob is counted as missed on a zero-overlap rule, so a prediction "
         "that catches a single pixel of a large shadow counts as a hit. The "
         "10% sensitivity definition is reported beside every primary number "
         "for that reason."),
        ("Blob category is a majority vote over the blob's multiclass pixels, "
         "so a blob spanning two shadow types is assigned wholly to the larger "
         "one. thin_linear has zero GT pixels in all three test pools and "
         "appears in no stratum."),
        ("Blobs under 10 px were dropped at mask-build time and the <10 px "
         "component filter is applied again here for exact agreement with "
         "n_gt_blobs, so the smallest area bin starts at 10 px and the miss "
         "rate says nothing about annotations below that size."),
        ("Blob-weighted rates are pooled over images with a bootstrap that "
         "resamples images; blobs inside one image are not independent and are "
         "not resampled separately."),
    ]

    with open(RESULTS_JSON, "w") as f:
        json.dump(jsonable(out), f, indent=1)
    print(f"wrote {RESULTS_JSON}")
    write_findings(out, fams, per, blobs)
    dt = (datetime.now(timezone.utc) - t_start).total_seconds()
    print(f"elapsed {dt:.0f} s")
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def fmt(x, nd=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def star(p):
    if p is None:
        return " "
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "


def write_findings(out, fams, per, blobs):
    L = []
    L.append("# Arms G and J, extension — whole-blob miss rate "
             "(reviewer zNBu-W6)\n")
    L.append(f"Generated {out['generated_utc']} by `{out['script']}`. "
             "Extends `analysis/gj_analysis/FINDINGS.md`; the cells, the "
             "pairing and the bootstrap protocol are the same 18 cells and the "
             "same helpers as that pass.\n")
    L.append("Observations only. No interpretation beyond what the numbers "
             "state.\n")
    L.append("## What this measures\n")
    L.append("Analysis E in the main pass reported that the FN boundary share "
             "falls in 9/9 cells. That is an indirect signature of whole "
             "shadows going missing. This pass counts them directly: a GT "
             "shadow blob is MISSED when the prediction places zero foreground "
             "pixels anywhere inside it.\n")
    L.append(f"- {out['definitions']['gt_blob']}\n")
    L.append(f"- {out['definitions']['binarization']}\n")
    L.append(f"- Primary: {out['definitions']['whole_blob_miss_rate']}\n")
    L.append(f"- Sensitivity: "
             f"{out['definitions']['whole_blob_miss_rate_10pct']}\n")

    L.append("\n## A. Whole-blob miss rate, per-image paired delta "
             "(LOCO minus within-city)\n")
    L.append("| cell | miss rate (zero overlap) | miss rate (<10% overlap) |")
    L.append("|---|---|---|")
    pc0 = {(r["arch"], r["city"]): r
           for r in fams["whole_blob_miss_rate"]["per_cell"]}
    pc1 = {(r["arch"], r["city"]): r
           for r in fams["whole_blob_miss_rate_10pct"]["per_cell"]}
    for city in CITIES:
        for arch in ARCHES:
            a, b_ = pc0[(arch, city)], pc1[(arch, city)]
            L.append(f"| {arch}/{city} | {fmt(a['mean'])}{star(a['p_holm'])} | "
                     f"{fmt(b_['mean'])}{star(b_['p_holm'])} |")
    for city in CITIES:
        a = fams["whole_blob_miss_rate"]["per_city_mean"][city]
        b_ = fams["whole_blob_miss_rate_10pct"]["per_city_mean"][city]
        L.append(f"| **3-arch mean / {city}** | {fmt(a['mean'])}"
                 f"{star(a.get('p_holm'))} | {fmt(b_['mean'])}"
                 f"{star(b_.get('p_holm'))} |")
    a = fams["whole_blob_miss_rate"]["overall_mean"]
    b_ = fams["whole_blob_miss_rate_10pct"]["overall_mean"]
    L.append(f"| **mean, all 9 cells** | {fmt(a['mean'])}{star(a['p_holm'])} | "
             f"{fmt(b_['mean'])}{star(b_['p_holm'])} |")
    w = fams["whole_blob_miss_rate"]["worst_cell"]
    L.append(f"| **worst cell** | {fmt(w['mean'])} ({w['arch']}/{w['city']}) "
             f"| - |")
    L.append("\nMarks are Holm-corrected within the 2-metric family: "
             "`*` p<0.05, `**` p<0.01, `***` p<0.001.\n")

    L.append("\nLevels, within / LOCO, each the mean over the contributing "
             "images of that image's own rate:\n")
    L.append("| cell | miss rate within / LOCO |")
    L.append("|---|---|")
    for city in CITIES:
        for arch in ARCHES:
            r = pc0[(arch, city)]
            L.append(f"| {arch}/{city} | {fmt(r['upper_mean'])} / "
                     f"{fmt(r['loco_mean'])} |")

    L.append("\n## B. Blob-weighted pooled rates and the LOCO / within ratio\n")
    L.append("| cell | within | LOCO | ratio | LOCO 95% CI | ratio 95% CI |")
    L.append("|---|---|---|---|---|---|")
    for r in out["pooled_rates"]["whole_blob_miss_rate"]:
        ci = r["rate_loco_ci95"]; rc = r["ratio_ci95"]
        L.append(f"| {r['arch']}/{r['city']} | {fmt(r['rate_upper'])} | "
                 f"{fmt(r['rate_loco'])} | {fmt(r['ratio'], 2)} | "
                 f"[{fmt(ci[0])}, {fmt(ci[1])}] | "
                 f"[{fmt(rc[0], 2)}, {fmt(rc[1], 2)}] |")
    op = out["overall_pooled"]
    L.append(f"| **all 9 cells pooled** | {fmt(op['rate_upper'])} | "
             f"{fmt(op['rate_loco'])} | {fmt(op['ratio'], 2)} | "
             f"[{fmt(op['rate_loco_ci95'][0])}, "
             f"{fmt(op['rate_loco_ci95'][1])}] | "
             f"[{fmt(op['ratio_ci95'][0], 2)}, "
             f"{fmt(op['ratio_ci95'][1], 2)}] |")
    L.append(f"\n{op['note']}\n")

    def strata_block(title, key, order, labels, note=""):
        L.append(f"\n## {title}\n")
        L.append("| stratum | blobs | within | LOCO | difference [95% CI] "
                 "| ratio [95% CI] |")
        L.append("|---|---|---|---|---|---|")
        for k in order:
            r = out[key].get(k)
            if not r or "loco" not in r:
                continue
            dd = r["difference"]; rc = r.get("ratio_ci95") or [None, None]
            L.append(f"| {labels.get(k, k)} | {r['loco']['n_blobs']} | "
                     f"{fmt(r['upper']['rate'])} | {fmt(r['loco']['rate'])} | "
                     f"{fmt(dd['mean'])} [{fmt(dd['ci95'][0])}, "
                     f"{fmt(dd['ci95'][1])}] | {fmt(r.get('ratio'), 2)} "
                     f"[{fmt(rc[0], 2)}, {fmt(rc[1], 2)}] |")
        if note:
            L.append(f"\n{note}")

    strata_block("C. Miss rate by GT blob area (blob-weighted, 9 cells pooled)",
                 "by_area_bin", AREA_KEYS, AREA_LABELS,
                 "The area bins are model-independent and were cut before any "
                 "overlap was computed. Difference and ratio intervals are "
                 "paired: one image resample drives both sides.")
    strata_block("D. Miss rate by blob category (blob-weighted, 9 cells "
                 "pooled)", "by_category", CATS + ["uncategorized"], {})

    L.append("\n## E. Panels selected for the qualitative figure\n")
    fs = out["figure_support"]
    L.append(f"- Architecture: {fs['architecture_rule']}. "
             f"Selected: {fs['architecture_by_city']}.")
    L.append(f"- Rows 1-3: {fs['image_rule_rows_1_3']}.")
    L.append(f"- Row 4: {fs['image_rule_row_4']}.\n")
    L.append("| row | cell | image | GT blobs | within misses | LOCO misses |")
    L.append("|---|---|---|---|---|---|")
    for i, p in enumerate(fs["panels"], 1):
        L.append(f"| {i} | {p['arch']}/{p['city']} | `{p['image_id']}` | "
                 f"{p['n_gt_blobs']} | {p['n_missed_upper']} | "
                 f"{p['n_missed_loco']} |")

    s = out["sanity"]
    L.append("\n## Sanity checks\n")
    L.append(f"- Missed-blob pixels against the stored full-frame `fn_px`: "
             f"{s['missed_blob_px_vs_stored_fn_px']['n_violations']} "
             f"violations in "
             f"{s['missed_blob_px_vs_stored_fn_px']['n_checked']} image-sides, "
             f"bound {TIE_DRIFT_BOUND_PX} px.")
    L.append(f"- Blob count against the stored `n_gt_blobs`: "
             f"{s['blob_count_vs_stored_n_gt_blobs']['n_violations']} "
             f"violations in "
             f"{s['blob_count_vs_stored_n_gt_blobs']['n_checked']} "
             f"image-sides.")
    L.append(f"- Images with zero GT blobs after the <10 px filter, per city: "
             f"{s['images_with_zero_gt_blobs']}.")
    L.append(f"- GT blobs in the pooled population: {s['n_blobs_total']}.")

    L.append("\n## Caveats\n")
    for c in out["caveats"]:
        L.append(f"- {c}")
    L.append("")
    with open(FINDINGS_MD, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {FINDINGS_MD}")


def dry_run():
    """--dry-run: verify the metric module (done at import) and resolve every input; write nothing."""
    for p in (DUMPS, DATA, LEGEND_PATH, MAIN_JSON):
        print(f"{'ok' if os.path.exists(p) else 'MISSING'}  {p}")
    for arch in ARCHES:
        for city in CITIES:
            for cid in (loco_id(arch, city), upper_id(arch, city)):
                pq = os.path.join(DUMPS, cid, "perimage.parquet")
                n = len(pd.read_parquet(pq, columns=["image_id"])) if os.path.exists(pq) else None
                print(f"  {cid:45s} rows={n}")
    print("[dry-run] nothing written")


if __name__ == "__main__":
    dry_run() if "--dry-run" in sys.argv else main()
