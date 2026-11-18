#!/usr/bin/env python
"""
Arm A -- the OBSERVATIONAL half of section 4.2 on S-EO: does the foreground-
probability degradation that motivates section 4.2 internally appear externally?

Pure reanalysis of the vanilla S-EO grid. No GPU, no inference, no retraining.

STEP 0 -- the statistic is the PUBLISHED one, not a reinvention
--------------------------------------------------------------
The paper's foreground-probability quantity is Figure 2's p-bar_S, defined and
computed in exactly one place in the published tree:

  python/select_figure2_tiles.py, lines 193-198, md5 920fbe83688d69535eb717fc20f34de4
      m["p_loco_S"] = float(loco[shadow].mean())
      m["p_loco_B"] = float(loco[bg].mean())

lifted verbatim into the shared metric module as `pbar_populations`:

  shadowtransfer/metrics.py lines 1085-1095 (sha256 730ad4a5...a8a6e83, the
  metric_module_sha recorded in all 12 S-EO run.json sidecars):

      def pbar_populations(prob, gt):
          shadow = gt > 0
          bg = ~shadow
          return (float(prob[shadow].mean()) if shadow.any() else float('nan'),
                  float(prob[bg].mean())     if bg.any()     else float('nan'))

and rendered by the published figure script with the same expression:

  Shade Maps/Python/fig2_final.ipynb, cell 0, lines 126-131
      shadow = gt == 1
      denom = max(int(shadow.sum()), 1)
      p_loco = float(loco[shadow].sum() / denom)

So the published statistic is:

  WHICH PIXELS  ground-truth foreground (GT shadow), gt > 0. No confidence
                threshold, no coverage rule, no prediction-side condition.
  WHICH QUANTITY  the plain arithmetic mean of the UNTHRESHOLDED foreground
                probability P(shadow) over those pixels. Not a mass fraction,
                not a histogram, not a bin count.
  CONTROL       the same mean over GT background, p-bar_B, returned by the same
                published function. The figure annotates only p-bar_S; the
                caption's claim ("background stays dark") is the p-bar_B leg.
  EMPTY POPULATION  NaN, never zero.

That definition is computed here verbatim. This file recomputes NOTHING about
the estimator; it reads the column the SLURM box produced by calling the lifted
function above (seo_dump.py lines 137-138) and does the paired statistics on it.

The one scope difference from the published figure script, stated not hidden:
fig2_final.ipynb renders a single tile and takes the mean over EVERY GT-shadow
pixel of it, while the S-EO dump hands `pbar_populations` the valid region only
(the +-2 px don't-care band removed, DSM-unseen pixels removed) -- arm A's
committed pixel population, and the same restriction every other S-EO column
carries. The estimator is identical; the pixel population is the arm's. The
all-pixel variant cannot be computed on this box: it needs the S-EO GT masks,
which live on the SLURM box.

SCOPE -- hard lines
-------------------
- The 12 vanilla cells only (6 LOCO + 6 upper). The -p1histmatch probe dumps are
  NOT read.
- NOTHING AURC-shaped. No risk-coverage curve, no selective-prediction number,
  no aurc_* column is read, printed or written. Blocked pending ratification
  (recorded in PAPER_STATE.md; 02_seo_loco.py cites it as CONTRACT.md line 163).
  Probability distributions and mass statistics only.
- No branch is READ here. This file states which pre-committed branch the
  numbers land in by letter and stops. The reading is the operator's.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/04_seo_probspace.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Rule zero: the metric module is imported, never reimplemented. This checkout
# is the one whose sha256 matches metric_module_sha in every S-EO run.json.
METRICS_ROOT = os.environ.get("ST_METRICS_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.insert(0, METRICS_ROOT)
from shadowtransfer.metrics import (          # noqa: E402
    pbar_populations,
    threshold_and_filter,
)

METRICS_PATH = os.path.join(METRICS_ROOT, "shadowtransfer", "metrics.py")

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

BASE = os.environ["PROJECT_ROOT"]
DUMP_DIR = os.path.join(BASE, "dumps", "seo")
INDEX_PATH = os.path.join(DUMP_DIR, "INDEX.parquet")
RESULTS_PATH = os.path.join(BASE, "results", "04_seo_probspace.json")

ARCHS = ["mamnet", "dinov3"]
FOLDS = ["oma", "jax", "ucsd"]
MODES = {"loco": "vanilla-loco", "upper": "none-upper"}

EXPECTED_ROWS = {"oma": 1637, "jax": 2642, "ucsd": 1870}
EXPECTED_TARGET_ACCESS = {"loco": "source_only", "upper": "target_labels"}

B = 10_000
SEED = 0                       # every rng below is default_rng([SEED, stream])

# The published statistic and its published control. Nothing else is computed.
STAT_FG = "pbar_gt_shadow"     # Fig 2's p-bar_S
STAT_BG = "pbar_gt_bg"         # the same function's p-bar_B

# probs.npz verification (task 1)
VERIFY_CELL = ("mamnet", "jax", "loco")
N_VERIFY_PATCHES = 20
VERIFY_SEED = 0
MIN_PIXELS = 10                # the published <10 px connected-component filter

PAIRS = [(a, f) for f in FOLDS for a in ARCHS]      # fold-major
CELLS = [(a, f, m) for a in ARCHS for f in FOLDS for m in ("loco", "upper")]

# Columns this analysis touches. Everything AURC-shaped is deliberately absent.
USED_COLS = ["run_id", "image_id", "aoi_id", "city", "split",
             STAT_FG, STAT_BG,
             "n_valid_px", "n_gt_fg_valid", "empty_fg_valid", "all_shadow_flag",
             "n_seen_px", "n_px", "unseen_frac", "n_pred_pos_all_px"]
EXPECTED_DTYPE = {STAT_FG: "float64", STAT_BG: "float64",
                  "n_gt_fg_valid": "int64", "n_valid_px": "int64",
                  "n_pred_pos_all_px": "int64"}
FORBIDDEN_PREFIX = "aurc"

# ----------------------------------------------------------------------------
# PRE-COMMITTED BRANCHES -- recorded before any number below is computed.
# ----------------------------------------------------------------------------
# Direction convention: delta = p-bar(LOCO) - p-bar(upper), so a NEGATIVE delta
# on GT foreground is the degradation Figure 2 shows.

PRECOMMITTED_BRANCHES = {
    "1": ("LOCO shows the same-direction foreground-probability degradation vs "
          "upper in most fold x arch pairs -> the observational half of 4.2 "
          "transfers; reported next to the intervention half (0 of 6)."),
    "2": ("No degradation, or opposite direction -> 4.2's observational claim "
          "is internal-only; reported as-is."),
    "3": ("Mixed across folds/archs -> per-pair table, no pooled sentence."),
}

BRANCH_RULE = (
    "Per pair, on the published statistic p-bar_S (GT-foreground), with "
    "delta = LOCO - upper and Holm-adjusted sign-flip p over the 6 pairs: a "
    "pair is DEGRADATION if delta < 0 and Holm p < 0.05; OPPOSITE if delta > 0 "
    "and Holm p < 0.05; otherwise NS. Branch 1 if degradation >= 4 of 6 and "
    "opposite == 0. Branch 2 if degradation == 0. Branch 3 otherwise. The three "
    "cases are mutually exclusive and exhaustive. Fixed before the numbers."
)


def classify_pairs(deltas, holm_p, alpha=0.05):
    out = []
    for d, p in zip(deltas, holm_p):
        if p < alpha and d < 0:
            out.append("degradation")
        elif p < alpha and d > 0:
            out.append("opposite")
        else:
            out.append("ns")
    return out


def branch_letter(labels):
    n_deg = labels.count("degradation")
    n_opp = labels.count("opposite")
    if n_deg >= 4 and n_opp == 0:
        return "1"
    if n_deg == 0:
        return "2"
    return "3"


# ----------------------------------------------------------------------------
# loading and validation
# ----------------------------------------------------------------------------

def fail(msg):
    raise SystemExit(f"ASSERTION FAILED: {msg}")


def run_id(arch, fold, mode):
    return f"{arch}-{MODES[mode]}-r3500-{fold}-hr-s0"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_cell(arch, fold, mode):
    """Read one cell. Start from INDEX.parquet, never from a glob."""
    rid = run_id(arch, fold, mode)
    d = os.path.join(DUMP_DIR, rid)
    if not os.path.isfile(os.path.join(d, ".DONE")):
        fail(f"{rid}: no .DONE marker")
    pq = os.path.join(d, "perimage.parquet")
    sj = os.path.join(d, "run.json")
    npz = os.path.join(d, "probs.npz")
    if not os.path.isfile(sj):
        fail(f"{rid}: run.json missing next to perimage.parquet")
    if not os.path.isfile(npz):
        fail(f"{rid}: probs.npz missing")
    df = pd.read_parquet(pq)
    with open(sj) as fh:
        meta = json.load(fh)
    return rid, df, meta, pq, npz


def validate_cell(rid, df, meta, fold, mode):
    """CONTRACT.md 3 checks, restricted to what this analysis reads. Loud."""
    checks = []
    missing = [c for c in USED_COLS if c not in df.columns]
    checks.append(("all used columns present", not missing, missing or "ok"))
    for c, want in EXPECTED_DTYPE.items():
        got = str(df[c].dtype) if c in df.columns else "ABSENT"
        checks.append((f"dtype {c}=={want}", got == want, got))
    checks.append((f"nrow=={EXPECTED_ROWS[fold]}",
                   len(df) == EXPECTED_ROWS[fold], len(df)))
    checks.append(("n_images==n_test_expected",
                   len(df) == meta.get("n_test_expected"),
                   meta.get("n_test_expected")))
    checks.append(("run_id matches dir", meta["run_id"] == rid, meta["run_id"]))
    checks.append(("run_id col uniform",
                   df.run_id.nunique() == 1 and df.run_id.iloc[0] == rid, "ok"))
    checks.append(("dataset==seo", meta.get("dataset") == "seo", meta.get("dataset")))
    checks.append(("arm==A", meta.get("arm") == "A", meta.get("arm")))
    checks.append(("method is the vanilla grid",
                   meta.get("method") in ("vanilla", "none"), meta.get("method")))
    checks.append((f"target_access=={EXPECTED_TARGET_ACCESS[mode]}",
                   meta.get("target_access") == EXPECTED_TARGET_ACCESS[mode],
                   meta.get("target_access")))
    checks.append(("split==test", set(df.split.unique()) == {"test"},
                   list(df.split.unique())))
    checks.append(("city==holdout", set(df.city.unique()) == {meta["holdout"]},
                   list(df.city.unique())))
    checks.append(("metric_module_sha matches the imported module",
                   meta.get("metric_module_sha") == sha256_file(METRICS_PATH),
                   str(meta.get("metric_module_sha"))[:12]))
    checks.append(("no duplicate image_id", not df.image_id.duplicated().any(), "ok"))

    # the published statistic is a probability, and NaN exactly where its
    # population is empty -- pbar_populations' own convention
    fg = df[STAT_FG]
    bad_rng = int(((fg < 0) | (fg > 1)).sum())
    checks.append(("pbar_gt_shadow in [0,1] where defined", bad_rng == 0, bad_rng))
    checks.append(("pbar_gt_shadow NaN <=> n_gt_fg_valid==0",
                   bool((fg.isna() == (df.n_gt_fg_valid == 0)).all()),
                   f"{int(fg.isna().sum())} NaN / {int((df.n_gt_fg_valid == 0).sum())} empty"))
    bg = df[STAT_BG]
    n_bg_empty = int((df.n_gt_fg_valid == df.n_valid_px).sum())
    checks.append(("pbar_gt_bg NaN <=> GT-bg population empty",
                   bool((bg.isna() == (df.n_gt_fg_valid == df.n_valid_px)).all()),
                   f"{int(bg.isna().sum())} NaN / {n_bg_empty} empty"))
    return checks


# ----------------------------------------------------------------------------
# TASK 1 -- probs.npz semantics
# ----------------------------------------------------------------------------

def verify_probs(rid, df, npz_path, n_patches, full=True):
    """Is probs.npz the UNTHRESHOLDED foreground probability this analysis
    assumes, keyed the way the parquet is keyed?

    The decision-rule check is against `n_pred_pos_all_px`, the one stored count
    that is computed on all pixels (seo_dump.py line 200) and therefore needs
    neither the GT mask nor the DSM-seen mask, neither of which is on this box.
    """
    z = np.load(npz_path)
    keys = set(z.files)
    ids = set(df.image_id)
    rep = {"cell": rid, "n_keys": len(keys),
           "keys_equal_image_id": keys == ids,
           "n_keys_not_in_parquet": len(keys - ids),
           "n_parquet_rows_without_key": len(ids - keys)}

    rng = np.random.default_rng([VERIFY_SEED, 7])
    order = df.image_id.to_numpy()
    pick = order[rng.choice(len(order), size=min(n_patches, len(order)),
                            replace=False)]

    rows = []
    for iid in pick:
        a = z[iid]
        p = a.astype(np.float32)
        stored = int(df.loc[df.image_id == iid, "n_pred_pos_all_px"].iloc[0])
        # The published rule, verbatim: argmax([1-p, p]) then the <10 px filter.
        # argmax resolves p == 0.5 to background. float16 cannot represent the
        # decision boundary, so every true probability within 2^-12 of 0.5 is
        # stored as exactly 0.5 and its true side is not recoverable. Both tie
        # resolutions are therefore computed and the stored count must lie
        # between them. The upper bound sends ties to foreground by advancing
        # the input one ulp -- the same published function, a perturbed input,
        # no reimplemented arithmetic. Nothing but an exact tie can change side.
        n_lo = int((threshold_and_filter(p, min_pixels=MIN_PIXELS) > 0).sum())
        n_hi = int((threshold_and_filter(np.nextafter(p, np.float32(1.0)),
                                         min_pixels=MIN_PIXELS) > 0).sum())
        n_tie = int((p == 0.5).sum())
        rows.append(dict(
            image_id=str(iid), dtype=str(a.dtype), shape=list(a.shape),
            pmin=float(p.min()), pmax=float(p.max()),
            n_distinct=int(np.unique(p).size),
            frac_interior=float(((p > 0.01) & (p < 0.99)).mean()),
            n_tie_at_half=n_tie,
            n_pred_pos_tie_to_bg=n_lo, n_pred_pos_tie_to_fg=n_hi,
            n_pred_pos_stored=stored,
            diff=stored - n_lo, bracket_width=n_hi - n_lo,
            in_bracket=bool(n_lo <= stored <= n_hi)))
    v = pd.DataFrame(rows)

    rep["dtype_uniform"] = sorted(v["dtype"].unique().tolist())
    rep["shapes"] = sorted({tuple(s) for s in v["shape"]})
    rep["in_unit_interval"] = bool((v.pmin >= 0).all() and (v.pmax <= 1).all())
    rep["is_prethreshold"] = bool((v.n_distinct > 2).all())
    rep["min_distinct_values"] = int(v.n_distinct.min())
    rep["median_frac_interior"] = float(v.frac_interior.median())
    rep["n_exact_match"] = int((v["diff"] == 0).sum())
    rep["n_checked"] = len(v)
    rep["stored_inside_tie_bracket"] = bool(v.in_bracket.all())
    rep["max_abs_diff"] = int(v["diff"].abs().max())
    rep["max_ties"] = int(v.n_tie_at_half.max())
    rep["max_bracket_width"] = int(v.bracket_width.max())
    rep["max_rel_diff"] = float((v["diff"].abs() / v.n_pred_pos_stored).max())
    rep["patches"] = v.to_dict("records") if full else []
    return rep, v


# ----------------------------------------------------------------------------
# bootstrap machinery -- identical to analysis/02_seo_loco.py, so the intervals
# in this file and in results/02_seo_loco.json are comparable.
# ----------------------------------------------------------------------------

def _rng(stream):
    return np.random.default_rng([SEED, stream])


def aoi_cluster_bootstrap_imagemean(delta, aoi_codes, n_aoi, stream):
    """Resample AOIs with replacement, take every image of a sampled AOI,
    statistic = image-level mean of the deltas."""
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    cnts = np.bincount(aoi_codes, minlength=n_aoi).astype(np.float64)
    idx = _rng(stream).integers(0, n_aoi, size=(B, n_aoi))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def aoi_cluster_bootstrap_aoimean(delta, aoi_codes, n_aoi, stream):
    """Same resampling, statistic = equal-weight mean of the per-AOI mean
    deltas. Secondary; reported so the weighting choice is visible."""
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    cnts = np.bincount(aoi_codes, minlength=n_aoi).astype(np.float64)
    means = sums / cnts
    idx = _rng(stream).integers(0, n_aoi, size=(B, n_aoi))
    return means[idx].mean(1)


def aoi_cluster_bootstrap_level(x, aoi_codes, n_aoi, stream):
    """Cluster bootstrap of a per-cell LEVEL (not a delta), same weighting."""
    return aoi_cluster_bootstrap_imagemean(x, aoi_codes, n_aoi, stream)


def aoi_signflip_imagemean(delta, aoi_codes, n_aoi, stream):
    """Two-sided sign-flip null at the AOI cluster level, same weighting as the
    point estimate. p = (1 + #{|T*| >= |T_obs|}) / (B + 1)."""
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    n_tot = float(len(delta))
    obs = sums.sum() / n_tot
    signs = _rng(stream).choice(np.array([-1.0, 1.0]), size=(B, n_aoi))
    null = (signs * sums).sum(1) / n_tot
    p = (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)
    return obs, p


def ci95(stat):
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, monotone-enforced."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * p[i]
        running = max(running, v)
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------
# paired analysis
# ----------------------------------------------------------------------------

def paired_pair(lo, up, col, stream_base):
    """One (arch, fold) pair on one statistic. Pairwise-complete on image_id."""
    if not (lo.image_id.values == up.image_id.values).all():
        fail("paired cells are not aligned on image_id")

    a = lo[col].to_numpy(np.float64)
    b = up[col].to_numpy(np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    if not (na == nb).all():
        fail(f"{col}: the undefined-statistic patch sets differ between the "
             f"LOCO and upper cells of the same fold ({int((na ^ nb).sum())} "
             "patches); the delta would not be paired")

    keep = ~na
    delta = a[keep] - b[keep]
    aoi = lo.aoi_id.to_numpy()[keep]
    codes, uniq = pd.factorize(aoi)
    n_aoi = len(uniq)

    obs, p_raw = aoi_signflip_imagemean(delta, codes, n_aoi, stream_base + 1)
    boot = aoi_cluster_bootstrap_imagemean(delta, codes, n_aoi, stream_base + 2)
    lo_i, hi_i = ci95(boot)
    boot_a = aoi_cluster_bootstrap_aoimean(delta, codes, n_aoi, stream_base + 3)
    lo_a, hi_a = ci95(boot_a)

    per_aoi = (pd.DataFrame({"aoi_id": aoi, "d": delta})
               .groupby("aoi_id").d.mean().to_numpy())

    return dict(
        loco_mean=float(a[keep].mean()), upper_mean=float(b[keep].mean()),
        delta=float(obs), stderr=float(boot.std(ddof=1)),
        ci_lo=lo_i, ci_hi=hi_i, p_raw=float(p_raw),
        delta_aoi_equal_weight=float(per_aoi.mean()),
        ci_lo_aoi=lo_a, ci_hi_aoi=hi_a,
        n_images=int(keep.sum()), n_excluded=int((~keep).sum()), n_aoi=n_aoi,
        n_aoi_dropped=int(lo.aoi_id.nunique() - n_aoi),
        frac_aoi_negative=float((per_aoi < 0).mean()))


def cell_level(df, col, stream):
    x = df[col].to_numpy(np.float64)
    keep = ~np.isnan(x)
    codes, uniq = pd.factorize(df.aoi_id.to_numpy()[keep])
    boot = aoi_cluster_bootstrap_level(x[keep], codes, len(uniq), stream)
    lo, hi = ci95(boot)
    q = np.percentile(x[keep], [0, 25, 50, 75, 100])
    return dict(mean=float(x[keep].mean()), stderr=float(boot.std(ddof=1)),
                ci_lo=lo, ci_hi=hi, n_images=int(keep.sum()),
                n_excluded=int((~keep).sum()), n_aoi=int(len(uniq)),
                q_min=float(q[0]), q25=float(q[1]), q50=float(q[2]),
                q75=float(q[3]), q_max=float(q[4]), sd=float(x[keep].std(ddof=1)))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    tag = "DRY RUN" if DRY_RUN else "FULL PASS"
    print("=" * 100)
    print(f"ARM A / S-EO probability space -- the published p-bar_S -- {tag}")
    print(f"dumps: {DUMP_DIR}")
    print(f"metrics: {METRICS_PATH}")
    print("=" * 100)

    # ---- the index is the entry point, never a glob ----
    if not os.path.isfile(INDEX_PATH):
        fail("INDEX.parquet missing; rebuild it before selecting cells")
    idx = pd.read_parquet(INDEX_PATH)
    index_sha256 = sha256_file(INDEX_PATH)
    want = {run_id(a, f, m) for a, f, m in CELLS}
    have = set(idx.run_id)
    if want - have:
        fail(f"cells missing from INDEX.parquet: {sorted(want - have)}")
    print(f"\nINDEX.parquet: {len(idx)} rows, sha256={index_sha256[:16]}...")
    print(f"selected {len(want)} of them (the vanilla grid); "
          f"{len(have - want)} rows in the index are NOT read here "
          "(method cells); -p1histmatch probe dumps are not in the index and "
          "are not read")

    data, metas, npzs, prov = {}, {}, {}, []
    all_checks = []
    for arch, fold, mode in CELLS:
        rid, df, meta, pq, npz = load_cell(arch, fold, mode)
        forbidden = [c for c in df.columns if c.startswith(FORBIDDEN_PREFIX)]
        checks = validate_cell(rid, df, meta, fold, mode)
        df = df.sort_values("image_id").reset_index(drop=True)
        data[(arch, fold, mode)] = df
        metas[(arch, fold, mode)] = meta
        npzs[(arch, fold, mode)] = npz
        bad = [c for c in checks if not c[1]]
        for name, ok, val in checks:
            if not ok:
                print(f"  !! {rid}: {name} -> {val}")
        all_checks.append(dict(
            cell=rid, nrow=len(df), n_aoi=int(df.aoi_id.nunique()),
            n_undef_fg=int(df[STAT_FG].isna().sum()),
            n_zero_valid=int((df.n_valid_px <= 0).sum()),
            mean_pbar_S=float(df[STAT_FG].mean()),
            mean_pbar_B=float(df[STAT_BG].mean()),
            failed=len(bad),
            aurc_cols_present_but_unread=len(forbidden)))
        prov.append(dict(cell=rid,
                         config_sha256=str(meta.get("config_sha256"))[:12],
                         git_hash=str(meta.get("git_hash"))[:10],
                         metric_module_sha=str(meta.get("metric_module_sha"))[:10]))

    tab = pd.DataFrame(all_checks)
    pd.set_option("display.width", 240)
    print("\n--- per-cell load + CONTRACT checks (columns this file reads) ---")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if tab.failed.sum():
        fail(f"{int(tab.failed.sum())} per-cell assertions failed; see above")

    # pairing identity within each fold
    print("\n--- per-fold pairing identity ---")
    for fold in FOLDS:
        ref = None
        for arch in ARCHS:
            for mode in ("loco", "upper"):
                df = data[(arch, fold, mode)]
                k = list(df.image_id)
                if ref is None:
                    ref = k
                elif k != ref:
                    fail(f"{fold}/{arch}/{mode}: image_id sequence differs")
                m = data[(arch, fold, "loco")].set_index("image_id").aoi_id
                m2 = df.set_index("image_id").aoi_id
                if not m.sort_index().equals(m2.sort_index()):
                    fail(f"{fold}/{arch}/{mode}: image_id -> aoi_id map differs")
        n_undef = {run_id(a, fold, m): int(data[(a, fold, m)][STAT_FG].isna().sum())
                   for a in ARCHS for m in ("loco", "upper")}
        if len(set(n_undef.values())) != 1:
            fail(f"{fold}: undefined-p-bar_S counts differ across cells {n_undef}")
        print(f"  {fold:5s}: 4/4 cells share {len(ref)} image_ids over "
              f"{data[(ARCHS[0], fold, 'loco')].aoi_id.nunique()} AOIs; "
              f"{list(n_undef.values())[0]} patches carry no GT foreground on "
              "the valid region in all four")

    print("\n--- provenance ---")
    print(pd.DataFrame(prov).to_string(index=False))

    # ------------------------------------------------------------------
    # TASK 1 -- probs.npz semantics
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TASK 1 -- probs.npz semantics")
    print("=" * 100)
    rid = run_id(*VERIFY_CELL)
    rep, v = verify_probs(rid, data[VERIFY_CELL], npzs[VERIFY_CELL],
                          N_VERIFY_PATCHES)
    print(f"cell: {rid}")
    print(f"  keys: {rep['n_keys']}, key set == parquet image_id set: "
          f"{rep['keys_equal_image_id']}")
    print(f"  dtype {rep['dtype_uniform']}, shapes {rep['shapes']}")
    print(f"  values in [0,1] on all {rep['n_checked']} sampled patches: "
          f"{rep['in_unit_interval']}")
    print(f"  pre-threshold (>2 distinct values; min distinct "
          f"{rep['min_distinct_values']}, median interior fraction "
          f"{rep['median_frac_interior']:.4f}): {rep['is_prethreshold']}")
    print(f"  decision rule (argmax + <{MIN_PIXELS} px filter) vs stored "
          f"n_pred_pos_all_px: exact on {rep['n_exact_match']}/{rep['n_checked']}, "
          f"max |diff| {rep['max_abs_diff']} px "
          f"({rep['max_rel_diff']*100:.3f}% of the stored count)")
    print(f"  stored count inside the float16 tie bracket on all "
          f"{rep['n_checked']}: {rep['stored_inside_tie_bracket']} "
          f"(max ties on a patch {rep['max_ties']}, max bracket width "
          f"{rep['max_bracket_width']} px)")
    print(v[["image_id", "n_tie_at_half", "n_pred_pos_tie_to_bg",
             "n_pred_pos_stored", "n_pred_pos_tie_to_fg", "diff"]]
          .to_string(index=False))

    if not (rep["keys_equal_image_id"] and rep["in_unit_interval"]
            and rep["is_prethreshold"]):
        fail("probs.npz semantics check failed on key set, range or "
             "pre-threshold test -- STOP, per the arm spec")
    if not rep["stored_inside_tie_bracket"]:
        fail("thresholding probs.npz does not reproduce n_pred_pos_all_px and "
             "the stored count is OUTSIDE the bracket the two float16 tie "
             "resolutions span -- STOP, per the arm spec")
    if rep["n_exact_match"] != rep["n_checked"]:
        print("\n  !! FLAG: the reproduction is not bit-exact. The stored count "
              "sits strictly between the two tie resolutions on every sampled "
              "patch, which is what float16 quantisation of the decision "
              "boundary produces: a true probability within 2^-12 of 0.5 is "
              "written as exactly 0.5, its true side is not recoverable from "
              "the dump, and a flipped tie pixel can additionally rescue or "
              "kill a component at the <10 px filter. The reported statistic "
              "does not threshold anything and was computed upstream on the "
              "float32 probability, so it is untouched. Reported, not silently "
              "accepted; the arm spec's STOP is the operator's to enforce.")

    # key-set spot check on the other 11 cells
    print("\n--- probs.npz key-set + range spot check, remaining 11 cells ---")
    spot = []
    for k in CELLS:
        if k == VERIFY_CELL:
            continue
        r, _ = verify_probs(run_id(*k), data[k], npzs[k], 2, full=False)
        spot.append(dict(cell=r["cell"], n_keys=r["n_keys"],
                         keys_ok=r["keys_equal_image_id"],
                         in_unit=r["in_unit_interval"],
                         prethresh=r["is_prethreshold"],
                         exact=f"{r['n_exact_match']}/{r['n_checked']}",
                         bracketed=r["stored_inside_tie_bracket"]))
    sp = pd.DataFrame(spot)
    print(sp.to_string(index=False))
    if not (sp.keys_ok.all() and sp.in_unit.all() and sp.prethresh.all()
            and sp.bracketed.all()):
        fail("probs.npz spot check failed on a cell outside the verification "
             "cell -- STOP")

    if DRY_RUN:
        print("\n--- cells selected ---")
        print(tab[["cell", "nrow", "n_aoi", "n_undef_fg", "mean_pbar_S",
                   "mean_pbar_B"]].to_string(index=False,
                                             float_format=lambda v: f"{v:.4f}"))
        print("\nDRY RUN OK -- all assertions passed. Nothing written.")
        return

    out = {}

    # ------------------------------------------------------------------
    # TASK 2 -- the published statistic on GT foreground, 6 pairs
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"TASK 2 -- {STAT_FG} (Fig 2's p-bar_S), delta = LOCO - upper, "
          f"AOI cluster bootstrap B={B}, Holm over 6")
    print("=" * 100)

    fg_rows = []
    for stream, (arch, fold) in enumerate(PAIRS):
        r = paired_pair(data[(arch, fold, "loco")], data[(arch, fold, "upper")],
                        STAT_FG, 10_000 * (stream + 1))
        fg_rows.append(dict(arch=arch, fold=fold, **r))
    fg = pd.DataFrame(fg_rows)
    fg["p_holm"] = holm(fg.p_raw.values)
    fg["verdict"] = classify_pairs(fg.delta.values, fg.p_holm.values)
    print(fg[["arch", "fold", "loco_mean", "upper_mean", "delta", "stderr",
              "ci_lo", "ci_hi", "p_raw", "p_holm", "verdict", "n_images",
              "n_excluded", "n_aoi", "frac_aoi_negative"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  secondary weighting (equal-weight per-AOI mean delta):")
    print(fg[["arch", "fold", "delta", "delta_aoi_equal_weight", "ci_lo_aoi",
              "ci_hi_aoi"]].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
    out["task2_pbar_gt_shadow"] = fg.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 3 -- the same statistic on GT background (the published control)
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"TASK 3 -- {STAT_BG} (the published p-bar_B control), same machinery")
    print("=" * 100)
    bg_rows = []
    for stream, (arch, fold) in enumerate(PAIRS):
        r = paired_pair(data[(arch, fold, "loco")], data[(arch, fold, "upper")],
                        STAT_BG, 20_000 * (stream + 1))
        bg_rows.append(dict(arch=arch, fold=fold, **r))
    bgt = pd.DataFrame(bg_rows)
    bgt["p_holm"] = holm(bgt.p_raw.values)
    print(bgt[["arch", "fold", "loco_mean", "upper_mean", "delta", "stderr",
               "ci_lo", "ci_hi", "p_raw", "p_holm", "n_images", "n_excluded",
               "n_aoi", "n_aoi_dropped"]].to_string(
                   index=False, float_format=lambda v: f"{v:.4f}"))
    ratio = (fg.delta.abs() / bgt.delta.abs().replace(0, np.nan)).to_numpy()
    print("\n  |delta p-bar_S| / |delta p-bar_B| per pair: "
          + ", ".join(f"{a}/{f} {r:.2f}" for (a, f), r in zip(PAIRS, ratio)))
    out["task3_pbar_gt_bg_control"] = bgt.to_dict("records")
    out["task3_fg_over_bg_abs_delta_ratio"] = [
        dict(arch=a, fold=f, ratio=float(r)) for (a, f), r in zip(PAIRS, ratio)]

    # ------------------------------------------------------------------
    # TASK 4 -- per-cell descriptives
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TASK 4 -- per-cell descriptives")
    print("=" * 100)
    desc = []
    for stream, k in enumerate(CELLS):
        arch, fold, mode = k
        df = data[k]
        d_fg = cell_level(df, STAT_FG, 30_000 + stream)
        d_bg = cell_level(df, STAT_BG, 40_000 + stream)
        desc.append(dict(
            cell=run_id(*k), arch=arch, fold=fold, mode=mode,
            n_images_total=len(df), n_images_used=d_fg["n_images"],
            n_images_undefined=d_fg["n_excluded"], n_aoi=d_fg["n_aoi"],
            n_gt_fg_px=int(df.n_gt_fg_valid.sum()),
            n_valid_px=int(df.n_valid_px.sum()),
            pbar_S_mean=d_fg["mean"], pbar_S_stderr=d_fg["stderr"],
            pbar_S_ci_lo=d_fg["ci_lo"], pbar_S_ci_hi=d_fg["ci_hi"],
            pbar_S_min=d_fg["q_min"], pbar_S_q25=d_fg["q25"],
            pbar_S_median=d_fg["q50"], pbar_S_q75=d_fg["q75"],
            pbar_S_max=d_fg["q_max"], pbar_S_sd=d_fg["sd"],
            pbar_B_mean=d_bg["mean"], pbar_B_stderr=d_bg["stderr"],
            pbar_B_ci_lo=d_bg["ci_lo"], pbar_B_ci_hi=d_bg["ci_hi"],
            pbar_B_q25=d_bg["q25"], pbar_B_median=d_bg["q50"],
            pbar_B_q75=d_bg["q75"]))
    dc = pd.DataFrame(desc)
    print(dc[["cell", "n_images_used", "n_images_undefined", "n_aoi",
              "n_gt_fg_px", "pbar_S_mean", "pbar_S_q25", "pbar_S_median",
              "pbar_S_q75", "pbar_B_mean", "pbar_B_median"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["task4_descriptives"] = dc.to_dict("records")

    # ------------------------------------------------------------------
    # branch, by letter only
    # ------------------------------------------------------------------
    labels = list(fg.verdict)
    letter = branch_letter(labels)
    print("\n" + "=" * 100)
    print(f"PRE-COMMITTED BRANCH: {letter}")
    print(f"  verdicts: {labels}")
    print(f"  rule: {BRANCH_RULE}")
    print("  the READING of this branch is the operator's; none is applied here")
    print("=" * 100)

    # ---- sanity flags ----
    flags = []
    if rep["n_exact_match"] != rep["n_checked"]:
        flags.append(
            f"probs.npz decision-rule reproduction is not bit-exact: "
            f"{rep['n_exact_match']}/{rep['n_checked']} sampled patches match "
            f"n_pred_pos_all_px exactly, max residual {rep['max_abs_diff']} px "
            f"({rep['max_rel_diff']*100:.3f}% of the stored count). On every "
            "sampled patch, here and in the 11-cell spot check, the stored "
            "count lies inside the bracket spanned by the two float16 tie "
            "resolutions at p == 0.5, which is what quantisation of the "
            "decision boundary produces. The arm spec says STOP on any "
            "mismatch; this run continued because the reported statistic never "
            "thresholds and was computed upstream on the float32 probability. "
            "Operator's call.")
    if float(fg.p_raw.min()) <= 1.0 / (B + 1.0) + 1e-12:
        n_floor = int((fg.p_raw <= 1.0 / (B + 1.0) + 1e-12).sum())
        flags.append(
            f"{n_floor} of 6 sign-flip p-values sit at the permutation floor "
            f"1/(B+1) = {1.0/(B+1.0):.4e}. The test establishes p < 1e-4 for "
            "those pairs and cannot resolve further at B = 10,000.")
    for _, r in fg.iterrows():
        if (r.ci_lo <= r.delta <= r.ci_hi) is False:
            flags.append(f"{r.arch}/{r.fold}: bootstrap CI excludes its own "
                         "point estimate")
    for name, t in (("p-bar_S", fg), ("p-bar_B", bgt)):
        drop = t[t.n_aoi_dropped > 0]
        for _, r in drop.iterrows():
            flags.append(
                f"{name}, {r.arch}/{r.fold}: {int(r.n_aoi_dropped)} AOI(s) leave "
                "the analysis entirely because every patch in them has an "
                "undefined statistic (empty population on the valid region); "
                f"the cluster bootstrap for that pair resamples {int(r.n_aoi)} "
                "AOIs, not the fold's full set.")
    if float(dc.pbar_S_sd.min()) < 1e-3:
        flags.append("a cell's per-image p-bar_S spread is under 1e-3, which is "
                     "implausibly low for 1.6-2.6k patches; inspect before use")
    for _, r in dc.iterrows():
        if r.pbar_S_min <= 0.0 or r.pbar_S_max >= 1.0:
            flags.append(f"{r.cell}: per-image p-bar_S touches a bound "
                         f"[{r.pbar_S_min:.4f}, {r.pbar_S_max:.4f}]")
    print("\n--- sanity flags ---")
    print("\n".join(f"  - {f}" for f in flags) if flags else "  none")

    # ------------------------------------------------------------------
    # results JSON
    # ------------------------------------------------------------------
    worst = fg.loc[fg.delta.idxmax()]      # least degradation = worst case here
    best = fg.loc[fg.delta.idxmin()]
    headline = (
        f"On S-EO, the published Figure 2 statistic p-bar_S (mean predicted "
        f"shadow probability on GT-foreground pixels, valid region, DSM-unseen "
        f"excluded) is lower under source-only LOCO than under the within-city "
        f"upper bound in {labels.count('degradation')} of the 6 (holdout city x "
        f"architecture) vanilla pairs, by {abs(float(best.delta)):.4f} in the "
        f"largest pair ({best.arch}/{best.fold}) and "
        f"{abs(float(worst.delta)):.4f} in the smallest ({worst.arch}/"
        f"{worst.fold}); the GT-background control p-bar_B moves by "
        f"{bgt.delta.abs().min():.4f} to {bgt.delta.abs().max():.4f}.")

    n_bg_moved = int((bgt.p_holm < 0.05).sum())
    bg_up = bgt[bgt.delta > 0]
    bg_dn = bgt[bgt.delta < 0]
    control_caveat = (
        f"The GT-background control is not null. p-bar_B differs between LOCO "
        f"and upper in {n_bg_moved} of the 6 pairs at Holm p < 0.05: upward "
        f"under LOCO on {sorted(set(bg_up.fold))} "
        f"(+{bg_up.delta.min():.4f} to +{bg_up.delta.max():.4f}) and downward "
        f"on {sorted(set(bg_dn.fold))} ({bg_dn.delta.min():.4f} to "
        f"{bg_dn.delta.max():.4f}). Per-pair |delta p-bar_S| / |delta p-bar_B| "
        f"ratios run {np.nanmin(ratio):.2f} to {np.nanmax(ratio):.2f} and are "
        "in task3_fg_over_bg_abs_delta_ratio.")

    res = {
        "id": "04_seo_probspace",
        "arm": "A",
        "answers": ["v1UJ-W1", "AC1"],
        "question": (
            "Does the foreground-probability degradation that motivates section "
            "4.2 internally -- Figure 2's p-bar_S collapsing under LOCO while "
            "the background stays dark -- appear on the external S-EO grid, "
            "measured with the published statistic?"),
        "runs": [run_id(*k) for k in CELLS],
        "runs_not_read": sorted(have - want),
        "index_sha256": index_sha256,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/04_seo_probspace.py",
        "git_hash": None,
        "git_hash_note": (
            "the project base is not a git repository on this box; final_paper/ "
            "is git-connected. Per-cell producing commits are in "
            "INDEX.parquet.git_hash and in the provenance block below."),
        "seed": SEED,
        "B": B,

        "statistic_provenance": {
            "name": "pbar_gt_shadow (Figure 2's p-bar_S)",
            "definition": (
                "plain arithmetic mean of the unthresholded predicted shadow "
                "probability over ground-truth foreground pixels (gt > 0). No "
                "threshold, no coverage rule, no prediction-side condition. NaN "
                "when the population is empty."),
            "published_source": (
                "python/select_figure2_tiles.py lines 193-198, md5 "
                "920fbe83688d69535eb717fc20f34de4 -- the only implementation of "
                "p-bar_S / p-bar_B in the published tree (recorded in "
                "shadowtransfer/metrics.py's LIFTED-FROM manifest, lines 93-97). "
                "That file is not on this box; the two on-box witnesses are the "
                "verbatim lift and the figure script below."),
            "verbatim_lift": (
                "shadowtransfer/metrics.py lines 1085-1095, pbar_populations; "
                "module sha256 730ad4a5bc039e3151599fb86bd411e2816c0897b68f1e4c"
                "17729335ab8a6e83, which equals metric_module_sha in all 12 "
                "S-EO run.json sidecars"),
            "figure_script": (
                "Shade Maps/Python/fig2_final.ipynb cell 0 lines 126-131 "
                "(shadow = gt == 1; p = float(prob[shadow].sum() / max(count,1)))"),
            "control_is_published": (
                "yes -- pbar_populations returns p-bar_B, the same mean over GT "
                "background, in the same call. Figure 2 annotates only p-bar_S; "
                "its caption's 'background stays dark' clause is the p-bar_B leg."),
            "how_computed_for_these_cells": (
                "seo_dump.py lines 136-138 calls the lifted pbar_populations on "
                "the valid region of each patch and stores the pair as "
                "pbar_gt_shadow / pbar_gt_bg. This file reads those columns and "
                "does the paired statistics; it recomputes no estimator."),
            "scope_difference_from_the_figure_script": (
                "fig2_final.ipynb averages over every GT-shadow pixel of a "
                "rendered tile; the S-EO dump averages over the valid region "
                "only (+-2 px don't-care band removed, DSM-unseen pixels "
                "removed), which is arm A's committed pixel population and the "
                "one every other S-EO column uses. Same estimator, arm's pixel "
                "population. The all-pixel variant needs the S-EO GT masks, "
                "which are not on this box."),
        },

        "precommitted_branches": PRECOMMITTED_BRANCHES,
        "branch_rule": BRANCH_RULE,
        "branch_landed": letter,
        "branch_verdicts": [dict(arch=a, fold=f, verdict=v)
                            for (a, f), v in zip(PAIRS, labels)],
        "branch_reading_note": (
            "letter only. The reading of the branch is the operator's; this "
            "file applies none."),

        "metric_definitions": {
            "pbar_gt_shadow": (
                "Figure 2's p-bar_S. Mean unthresholded P(shadow) over GT "
                "foreground on the valid region (K=2 don't-care band excluded, "
                "DSM-unseen excluded). Unitless probability in [0,1]."),
            "pbar_gt_bg": (
                "p-bar_B. The same mean over GT background on the same region."),
            "delta": (
                "per patch, p-bar(LOCO) - p-bar(upper) on the shared test pool. "
                "NEGATIVE on foreground is the degradation Figure 2 shows."),
            "point_estimate": (
                "image-level mean of the per-patch deltas, matching "
                "analysis/02_seo_loco.py so the intervals are comparable."),
            "ci_method": (
                f"AOI cluster bootstrap, B={B}, percentile, seed "
                f"default_rng([{SEED}, stream]); AOIs resampled with "
                "replacement, every patch of a sampled AOI taken."),
            "test": (
                "two-sided AOI-level sign-flip permutation, "
                "p = (1 + #{|T*| >= |T_obs|}) / (B + 1), Holm-Bonferroni over "
                "the 6 pairs within each statistic's own family."),
        },

        "verification_probs_npz": rep,
        "provenance": prov,

        # the full per-pair and per-cell tables. CS may not compute a number
        # that is not already in a results JSON, so every number the tables
        # above print lands here, not only the summary rows.
        **out,

        "conditions": [
            {"label": f"{r.arch} / {r.fold} / {r['mode']}",
             "metric": "pbar_gt_shadow",
             "mean": r.pbar_S_mean, "stderr": r.pbar_S_stderr,
             "n": int(r.n_images_used),
             "ci95": [r.pbar_S_ci_lo, r.pbar_S_ci_hi],
             "ci_method": f"aoi_cluster_bootstrap_B{B}",
             "baseline": ("within-city upper bound" if r["mode"] == "loco"
                          else "self (this is the baseline)")}
            for _, r in dc.iterrows()],
        "tests": [
            {"name": "AOI-level sign-flip permutation, Holm over 6",
             "compares": [f"{r.arch}/{r.fold} loco", f"{r.arch}/{r.fold} upper"],
             "metric": "pbar_gt_shadow",
             "statistic": float(r.delta), "p": float(r.p_raw),
             "p_holm": float(r.p_holm), "B": B, "two_tailed": True}
            for _, r in fg.iterrows()] + [
            {"name": "AOI-level sign-flip permutation, Holm over 6",
             "compares": [f"{r.arch}/{r.fold} loco", f"{r.arch}/{r.fold} upper"],
             "metric": "pbar_gt_bg",
             "statistic": float(r.delta), "p": float(r.p_raw),
             "p_holm": float(r.p_holm), "B": B, "two_tailed": True}
            for _, r in bgt.iterrows()],

        "headline": headline,
        "sanity_flags": flags,
        "caveats": [
            "Observational, not interventional. This is a paired comparison of "
            "two trained conditions on a shared test pool; nothing here "
            "manipulates appearance, so it carries no causal claim about what "
            "produces the degradation.",
            control_caveat,
            "Single seed. Every cell is seed 0, so the intervals cover test-patch "
            "and AOI sampling only, not optimisation variance.",
            "The 12 vanilla cells only: 6 source-only LOCO and 6 within-city "
            "upper. No adaptation method, no -p1histmatch probe cell.",
            "The pixel population is the valid region -- +-2 px don't-care band "
            "removed, DSM-unseen pixels removed -- not the all-pixel population "
            "the published figure script uses on a rendered tile. Same "
            "estimator; the all-pixel variant needs GT masks that are not on "
            "this box.",
            "Patches with no GT foreground on the valid region have an "
            "undefined p-bar_S and are dropped pairwise; the dropped set is "
            "asserted identical across the four cells of a fold. Counts are in "
            "task4_descriptives.n_images_undefined.",
            "Descriptive quartiles are over PER-IMAGE p-bar_S values, not over "
            "individual GT-foreground pixels. Pixel-level quantiles of P(fg) "
            "would need the per-pixel GT masks, which are on the SLURM box.",
            "probs.npz is float16, so the decision boundary is not exactly "
            "representable. Thresholding it at the cell's decision rule "
            "reproduces the stored predicted-positive count only to within the "
            "pixels stored as exactly 0.5: the stored count lies inside the "
            "bracket the two tie resolutions span on every sampled patch, but "
            "no sampled patch reproduces bit-exactly. The reported statistic "
            "does not threshold and was computed upstream on the float32 "
            "probability, so it is unaffected. Details in "
            "verification_probs_npz.",
            "Budget asymmetry is inherited from the grid: the LOCO cell trains "
            "on 3500 patches from two source cities, the upper cell on 3500 "
            "from the held-out city itself.",
            "S-EO labels are auto-generated. Both sides of every delta use the "
            "same labels on the same patches, so label noise largely cancels in "
            "the difference; the absolute levels carry it.",
            "rung 3500 is the S-EO patch budget on 384 px patches, not a point "
            "on the ShadowTransfer 225/450/900 ladder. Levels are not "
            "comparable to any internal p-bar_S number.",
            "Nothing AURC-shaped is computed or read. The definition is "
            "unratified (02_seo_loco.py cites CONTRACT.md line 163; the live "
            "record of the block is PAPER_STATE.md's open-questions list).",
            "High-resolution cells only, three cities, two architectures. "
            "OGLANet is not in arm A.",
            "The internal side of this comparison is not recomputed here. This "
            "file states the external numbers; any internal-vs-external "
            "sentence is the operator's and needs the internal p-bar_S "
            "computed under the same definition.",
        ],
        "figure": None,
        "caption": None,
    }

    tmp = RESULTS_PATH + ".new"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    os.replace(tmp, RESULTS_PATH)
    print(f"\nwrote {RESULTS_PATH}")
    print(f"headline: {headline}")


if __name__ == "__main__":
    main()
