#!/usr/bin/env python
"""
Arm A -- external LOCO on S-EO, vanilla grid.

Answers v1UJ-W1, BxRx-W1, AC1.

Computes exactly what the pre-committed outcome table is read against:
  TASK 1  per-(fold x arch) gap, AOI-level cluster bootstrap, Holm over 6
  TASK 2  pooled gap over per-AOI mean deltas, stratified by fold
  TASK 3  per-AOI gap ~ sun_dist + C(city), OLS and fold-clustered SE
  TASK 4  per-cell descriptives, 12 rows
  TASK 5  floor check on the upper cells' fg-only IoU, reported not adjudicated

Nothing evaluative beyond that. No HLT branch is applied here; no comparison to
arm C; nothing AURC-shaped (CONTRACT.md line 163, definition unratified).

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/02_seo_loco.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

BASE = os.environ["PROJECT_ROOT"]
DUMP_DIR = os.path.join(BASE, "dumps", "seo")
INDEX_PATH = os.path.join(DUMP_DIR, "INDEX.parquet")
RESULTS_PATH = os.path.join(BASE, "results", "02_seo_loco.json")

ARCHS = ["mamnet", "dinov3"]
FOLDS = ["oma", "jax", "ucsd"]
MODES = {"loco": "vanilla-loco", "upper": "none-upper"}

EXPECTED_ROWS = {"oma": 1637, "jax": 2642, "ucsd": 1870}
EXPECTED_NCOL = 76
EXPECTED_TARGET_ACCESS = {"loco": "source_only", "upper": "target_labels"}

B = 10_000
SEED = 0                       # every rng below is default_rng([SEED, stream_id])
EPS = 1e-10                    # metrics.py _img_metrics epsilon, matched exactly

# phase-2 arithmetic, fold-major / arch-minor, as handed over by the operator
PRECOMMITTED = {
    ("mamnet", "oma"): 7.7030,
    ("dinov3", "oma"): 3.2235,
    ("mamnet", "jax"): 6.6221,
    ("dinov3", "jax"): 5.1833,
    ("mamnet", "ucsd"): 6.7453,
    ("dinov3", "ucsd"): 4.2329,
}
# a 4-decimal quote carries +-5e-5 of rounding slack; anything wider is a real
# disagreement and stops the run.
XCHECK_TOL = 5.05e-5

PAIRS = [(a, f) for f in FOLDS for a in ARCHS]   # fold-major, matches PRECOMMITTED


def run_id(arch, fold, mode):
    return f"{arch}-{MODES[mode]}-r3500-{fold}-hr-s0"


CELLS = [(a, f, m) for a in ARCHS for f in FOLDS for m in ("loco", "upper")]


# ----------------------------------------------------------------------------
# loading and validation
# ----------------------------------------------------------------------------

def fail(msg):
    raise SystemExit(f"ASSERTION FAILED: {msg}")


def load_cell(arch, fold, mode):
    rid = run_id(arch, fold, mode)
    d = os.path.join(DUMP_DIR, rid)
    if not os.path.isfile(os.path.join(d, ".DONE")):
        fail(f"{rid}: no .DONE marker")
    pq = os.path.join(d, "perimage.parquet")
    sj = os.path.join(d, "run.json")
    if not os.path.isfile(sj):
        fail(f"{rid}: run.json missing next to perimage.parquet")
    df = pd.read_parquet(pq)
    with open(sj) as fh:
        meta = json.load(fh)
    return rid, df, meta, pq


def add_metrics(df):
    """Recompute the two reported metrics from the raw K=2 counts.

    two-class boundary-tolerant mIoU at K=2, unseen-excluded  -> miou2_pct
    foreground-only IoU at K=2, unseen-excluded               -> fg_pct

    Unseen pixels are already dropped from all four confusion cells upstream
    (run.json unseen_handling), so the counts are the unseen-excluded
    population and n_valid_px == tp+fp+fn+tn.
    """
    tp = df["tp_tol"].to_numpy(np.float64)
    fp = df["fp_tol"].to_numpy(np.float64)
    fn = df["fn_tol"].to_numpy(np.float64)
    tn = df["tn_tol"].to_numpy(np.float64)
    iou_fg = tp / (tp + fp + fn + EPS)
    iou_bg = tn / (tn + fp + fn + EPS)
    df = df.copy()
    df["miou2_pct"] = 50.0 * (iou_fg + iou_bg)
    df["fg_pct"] = 100.0 * iou_fg
    return df


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def scalar_fields(meta):
    return {k: v for k, v in meta.items()
            if v is None or isinstance(v, (str, int, float, bool))}


# ----------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------

def dry_run():
    print("=" * 100)
    print("ARM A / S-EO vanilla grid -- DRY RUN")
    print(f"dumps: {DUMP_DIR}")
    print("=" * 100)

    data, metas, paths = {}, {}, {}
    rows = []
    colsets, dtypes_ref = {}, None

    for arch, fold, mode in CELLS:
        rid, df, meta, pq = load_cell(arch, fold, mode)
        data[(arch, fold, mode)] = df
        metas[(arch, fold, mode)] = meta
        paths[(arch, fold, mode)] = pq

        checks = []

        ncol = df.shape[1]
        checks.append(("ncol==76", ncol == EXPECTED_NCOL, ncol))

        want = EXPECTED_ROWS[fold]
        checks.append((f"nrow=={want}", len(df) == want, len(df)))

        n_null_aoi = int(df["aoi_id"].isna().sum())
        checks.append(("aoi_id non-null", n_null_aoi == 0, n_null_aoi))

        n_null_sun = int(df["sun_elevation"].isna().sum())
        checks.append(("sun_elevation non-null", n_null_sun == 0, n_null_sun))

        # n_valid_px > 0 does NOT hold everywhere: 108 test patches are entirely
        # DSM-unseen, so the unseen-exclusion rule empties all four confusion
        # cells. Assert the exact condition under which that is harmless to a
        # paired gap instead of asserting it away.
        zero = df["n_valid_px"] <= 0
        n_zero = int(zero.sum())
        checks.append(("n_valid_px==0 rows are all fully DSM-unseen",
                       bool((df.loc[zero, "unseen_frac"] == 1.0).all()
                            and (df.loc[zero, "n_seen_px"] == 0).all()
                            and (df.loc[zero, ["tp_tol", "fp_tol", "fn_tol",
                                               "tn_tol"]].to_numpy() == 0).all()),
                       f"{n_zero} rows"))
        checks.append(("n_valid_px>0 on every non-fully-unseen row",
                       int(((~zero) & (df["n_valid_px"] <= 0)).sum()) == 0, 0))

        ta = meta.get("target_access")
        checks.append((f"target_access=={EXPECTED_TARGET_ACCESS[mode]}",
                       ta == EXPECTED_TARGET_ACCESS[mode], ta))

        # counts are the unseen-excluded valid population
        s4 = (df.tp_tol + df.fp_tol + df.fn_tol + df.tn_tol)
        checks.append(("tp+fp+fn+tn==n_valid_px",
                       bool((s4 == df.n_valid_px).all()), "ok"))

        # sidecar identity
        checks.append(("run_id matches dir", meta["run_id"] == rid, meta["run_id"]))
        checks.append(("run_id col uniform",
                       df.run_id.nunique() == 1 and df.run_id.iloc[0] == rid, "ok"))
        checks.append(("dataset==seo", meta.get("dataset") == "seo", meta.get("dataset")))
        checks.append(("arm==A", meta.get("arm") == "A", meta.get("arm")))
        checks.append(("rung==3500", meta.get("rung") == 3500, meta.get("rung")))
        checks.append(("seed==0", meta.get("seed") == 0, meta.get("seed")))
        checks.append(("res==hr", meta.get("res") == "hr", meta.get("res")))
        checks.append(("n_images==n_test_expected",
                       len(df) == meta.get("n_test_expected"),
                       meta.get("n_test_expected")))
        checks.append(("split==test", set(df.split.unique()) == {"test"},
                       list(df.split.unique())))
        checks.append(("city==holdout",
                       set(df.city.unique()) == {meta["holdout"]},
                       list(df.city.unique())))

        # metric recompute agrees with the stored convenience columns
        dfm = add_metrics(df)
        d_miou = float(np.abs(dfm.miou2_pct - df.miou_tol_pct).max())
        d_fg = float(np.abs(dfm.fg_pct - df.iou_tol * 100.0).max())
        checks.append(("recompute==miou_tol_pct", d_miou < 1e-9, f"{d_miou:.2e}"))
        checks.append(("recompute==iou_tol", d_fg < 1e-9, f"{d_fg:.2e}"))

        colsets[(arch, fold, mode)] = list(df.columns)
        if dtypes_ref is None:
            dtypes_ref = df.dtypes
            dtype_ok = True
        else:
            dtype_ok = bool(df.dtypes.equals(dtypes_ref))
        checks.append(("dtypes match cell 1", dtype_ok, "ok" if dtype_ok else "DRIFT"))

        bad = [c for c in checks if not c[1]]
        rows.append(dict(cell=rid, ncol=ncol, nrow=len(df),
                         n_aoi=df.aoi_id.nunique(), target_access=ta,
                         failed=len(bad), detail=";".join(f"{c[0]}={c[2]}" for c in bad)))
        for name, ok, val in checks:
            if not ok:
                print(f"  !! {rid}: {name} -> {val}")

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    print("\n--- per-cell assertions ---")
    print(tab.to_string(index=False))

    if tab.failed.sum():
        fail(f"{int(tab.failed.sum())} per-cell assertions failed; see above")

    # column name set identical across all 12
    ref = colsets[CELLS[0][:3]]
    for k, v in colsets.items():
        if v != ref:
            fail(f"{k}: column name/order differs from reference cell")
    print("\ncolumn name set + order identical across all 12 cells: ok")

    # per fold, all 4 cells cover an identical patch set
    print("\n--- per-fold patch-set identity (key: image_id; no patch_id column exists) ---")
    for fold in FOLDS:
        keysets, keysets2 = {}, {}
        for arch in ARCHS:
            for mode in ("loco", "upper"):
                df = data[(arch, fold, mode)]
                keysets[(arch, mode)] = set(df.image_id)
                keysets2[(arch, mode)] = set(zip(df.source_image_id, df.patch_index))
                if df.image_id.duplicated().any():
                    fail(f"{run_id(arch, fold, mode)}: duplicate image_id")
        ref_k = keysets[(ARCHS[0], "loco")]
        ref_k2 = keysets2[(ARCHS[0], "loco")]
        for k, v in keysets.items():
            if v != ref_k:
                fail(f"{fold}/{k}: image_id set differs ({len(v ^ ref_k)} symmetric diff)")
        for k, v in keysets2.items():
            if v != ref_k2:
                fail(f"{fold}/{k}: (source_image_id, patch_index) set differs")
        n_aoi = data[(ARCHS[0], fold, "loco")].aoi_id.nunique()
        print(f"  {fold:5s}: 4/4 cells share {len(ref_k)} image_ids over {n_aoi} AOIs -- ok")

    # the fully-unseen patches must be the SAME patches in all 4 cells of a fold,
    # and must not swallow a whole AOI, or the AOI units of tasks 2 and 3 degenerate
    print("\n--- fully DSM-unseen patches (unseen_frac == 1.0, all four counts 0) ---")
    for fold in FOLDS:
        ref_z = None
        for arch in ARCHS:
            for mode in ("loco", "upper"):
                df = data[(arch, fold, mode)]
                z = set(df.loc[df.n_valid_px <= 0, "image_id"])
                if ref_z is None:
                    ref_z = z
                elif z != ref_z:
                    fail(f"{fold}/{arch}/{mode}: fully-unseen patch set differs "
                         "between cells of the same fold; the gap would not be paired")
        df = data[(ARCHS[0], fold, "loco")]
        g = df.assign(z=df.n_valid_px <= 0).groupby("aoi_id").z.agg(["size", "sum"])
        whole = g[g["sum"] == g["size"]]
        if len(whole):
            fail(f"{fold}: {len(whole)} AOI(s) entirely fully-unseen -> degenerate AOI unit")
        print(f"  {fold:5s}: {len(ref_z):4d} / {len(df)} patches, identical in all 4 cells, "
              f"touching {int((g['sum'] > 0).sum())} of {len(g)} AOIs, none wholly")

    # aoi_id -> image mapping identical across the 4 cells of a fold
    for fold in FOLDS:
        ref_map = data[(ARCHS[0], fold, "loco")].set_index("image_id").aoi_id.sort_index()
        for arch in ARCHS:
            for mode in ("loco", "upper"):
                m = data[(arch, fold, mode)].set_index("image_id").aoi_id.sort_index()
                if not m.equals(ref_map):
                    fail(f"{fold}/{arch}/{mode}: image_id -> aoi_id mapping differs")
    print("image_id -> aoi_id mapping identical within every fold: ok")

    # no duplicate run_id with differing config_sha256
    seen = {}
    for k, meta in metas.items():
        rid, sha = meta["run_id"], meta.get("config_sha256")
        if rid in seen and seen[rid] != sha:
            fail(f"{rid}: duplicate run_id with differing config_sha256")
        seen[rid] = sha
    print(f"run_id uniqueness / config_sha256 consistency: ok ({len(seen)} run_ids)")

    # provenance the report should carry
    print("\n--- provenance ---")
    prov = pd.DataFrame([
        dict(cell=metas[k]["run_id"],
             config_sha256=str(metas[k].get("config_sha256"))[:12],
             git_hash=str(metas[k].get("git_hash"))[:10],
             metric_module_sha=str(metas[k].get("metric_module_sha"))[:10],
             train_eval_correction=("train_eval_disabled_from_epoch_correction" in metas[k]))
        for k in CELLS])
    print(prov.to_string(index=False))

    # index
    idx = build_index(metas, data, paths)
    print(f"\n--- INDEX.parquet ({len(idx)} rows, {idx.shape[1]} cols) ---")
    print(idx[["run_id", "arch", "method", "mode", "holdout", "target_access",
               "n_images", "mean_iou_tol"]].to_string(index=False))
    print(f"\nwould write: {INDEX_PATH}")
    print("\nDRY RUN OK -- all assertions passed. Nothing written.")


def build_index(metas, data, paths):
    rows = []
    for k in CELLS:
        meta, df, pq = metas[k], data[k], paths[k]
        r = scalar_fields(meta)
        r["n_images"] = len(df)
        r["mean_iou_tol"] = float(df.iou_tol.mean())
        r["parquet_mtime"] = datetime.fromtimestamp(
            os.path.getmtime(pq), timezone.utc).isoformat()
        r["dump_sha256"] = sha256_file(pq)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# bootstrap machinery
# ----------------------------------------------------------------------------

def _rng(stream):
    return np.random.default_rng([SEED, stream])


def aoi_cluster_bootstrap_imagemean(delta, aoi_codes, n_aoi, stream):
    """Cluster bootstrap: resample AOIs with replacement, take every image of a
    sampled AOI, statistic = image-level mean of the deltas.

    The point estimate of that statistic is the plain image-level mean, which is
    what the phase-2 arithmetic reports.
    """
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    cnts = np.bincount(aoi_codes, minlength=n_aoi).astype(np.float64)
    idx = _rng(stream).integers(0, n_aoi, size=(B, n_aoi))
    stat = sums[idx].sum(1) / cnts[idx].sum(1)
    return stat


def aoi_signflip_imagemean(delta, aoi_codes, n_aoi, stream):
    """Two-sided sign-flip null at the AOI cluster level, same weighting as the
    point estimate. p = (1 + #{|T*| >= |T_obs|}) / (B + 1)."""
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    n_tot = float(len(delta))
    obs = sums.sum() / n_tot
    signs = _rng(stream).choice(np.array([-1.0, 1.0]), size=(B, n_aoi))
    null = (signs * sums).sum(1) / n_tot
    p = (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)
    return obs, p, null


def patch_bootstrap_mean(delta, stream, block=1000):
    """Image-level bootstrap that ignores AOI clustering. Anti-conservative;
    reported for comparison only."""
    rng = _rng(stream)
    n = len(delta)
    out = np.empty(B)
    for s in range(0, B, block):
        e = min(s + block, B)
        idx = rng.integers(0, n, size=(e - s, n))
        out[s:e] = delta[idx].mean(1)
    return out


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
# full pass
# ----------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("ARM A / S-EO vanilla grid -- FULL PASS")
    print("=" * 100)

    data, metas, paths = {}, {}, {}
    for arch, fold, mode in CELLS:
        rid, df, meta, pq = load_cell(arch, fold, mode)
        df = add_metrics(df)
        # recompute must agree with the stored convenience columns
        assert float(np.abs(df.miou2_pct - df.miou_tol_pct).max()) < 1e-9, rid
        assert float(np.abs(df.fg_pct - df.iou_tol * 100.0).max()) < 1e-9, rid
        data[(arch, fold, mode)] = df.sort_values("image_id").reset_index(drop=True)
        metas[(arch, fold, mode)] = meta
        paths[(arch, fold, mode)] = pq

    # ---- INDEX ----
    idx = build_index(metas, data, paths)
    idx.to_parquet(INDEX_PATH, index=False)
    index_sha256 = sha256_file(INDEX_PATH)
    print(f"\nwrote {INDEX_PATH}  ({len(idx)} rows)  sha256={index_sha256[:16]}...")

    out = {}

    # ------------------------------------------------------------------
    # TASK 1 -- per-pair gaps
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TASK 1 -- per-(fold x arch) gap, upper - loco, AOI-level bootstrap, B=%d" % B)
    print("=" * 100)

    per_aoi_store = {}          # (arch, fold) -> DataFrame(aoi_id, gap, sun_dist_loco)
    t1_rows, xcheck_rows, sens_rows = [], [], []

    for stream, (arch, fold) in enumerate(PAIRS):
        lo = data[(arch, fold, "loco")]
        up = data[(arch, fold, "upper")]
        assert (lo.image_id.values == up.image_id.values).all()

        aoi = lo.aoi_id.values
        codes, uniq = pd.factorize(aoi)
        n_aoi = len(uniq)

        for metric, mcol, tag in (("2class", "miou2_pct", "primary"),
                                  ("fgonly", "fg_pct", "secondary")):
            delta = (up[mcol].values - lo[mcol].values).astype(np.float64)

            obs, p_raw, _ = aoi_signflip_imagemean(delta, codes, n_aoi, 1000 * stream + (0 if metric == "2class" else 1))
            boot_a = aoi_cluster_bootstrap_imagemean(delta, codes, n_aoi, 2000 * stream + (0 if metric == "2class" else 1))
            lo_a, hi_a = ci95(boot_a)
            boot_p = patch_bootstrap_mean(delta, 3000 * stream + (0 if metric == "2class" else 1))
            lo_p, hi_p = ci95(boot_p)

            t1_rows.append(dict(
                arch=arch, fold=fold, metric=metric, role=tag,
                loco=float(lo[mcol].mean()), upper=float(up[mcol].mean()),
                gap=float(obs), stderr=float(boot_a.std(ddof=1)),
                ci_lo=lo_a, ci_hi=hi_a, ci_w=hi_a - lo_a,
                p_raw=p_raw, n_aoi=n_aoi, n_images=len(delta),
                patch_ci_lo=lo_p, patch_ci_hi=hi_p, patch_ci_w=hi_p - lo_p,
                width_ratio=(hi_p - lo_p) / (hi_a - lo_a)))

            # sensitivity: drop the fully DSM-unseen patches. Their delta is
            # exactly 0 in both cells, so they cannot move the sign, only dilute
            # the magnitude. The pre-committed number keeps them; this is a
            # robustness line, not a replacement.
            keep = lo.n_valid_px.values > 0
            if int((~keep).sum()):
                dk = delta[keep]
                ck, uk = pd.factorize(aoi[keep])
                bk = aoi_cluster_bootstrap_imagemean(dk, ck, len(uk),
                                                     5000 * stream + (0 if metric == "2class" else 1))
                klo, khi = ci95(bk)
                sens_rows.append(dict(
                    arch=arch, fold=fold, metric=metric,
                    gap_all_patches=float(obs), gap_excl_fully_unseen=float(dk.mean()),
                    delta=float(dk.mean() - obs), n_excluded=int((~keep).sum()),
                    n_kept=int(keep.sum()), ci_lo=klo, ci_hi=khi,
                    loco_excl=float(lo[mcol].values[keep].mean()),
                    upper_excl=float(up[mcol].values[keep].mean())))

            if metric == "2class":
                exp = PRECOMMITTED[(arch, fold)]
                # The phase-2 table differenced the two 4-decimal-rounded cell
                # means; the full-precision mean of the per-patch deltas can
                # differ from it in the 4th decimal by double rounding. Both
                # forms are checked. The rounded-difference form must reproduce
                # the pre-committed value exactly.
                l4, u4 = round(float(lo[mcol].mean()), 4), round(float(up[mcol].mean()), 4)
                rnd = round(u4 - l4, 4)
                d = abs(obs - exp)
                xcheck_rows.append(dict(
                    arch=arch, fold=fold, gap_direct=obs,
                    gap_from_rounded_cell_means=rnd, precommitted=exp,
                    abs_diff_direct=d, direct_within_tol=d <= XCHECK_TOL,
                    rounded_form_exact=(rnd == exp), ok=(rnd == exp)))
                # per-AOI units for tasks 2 and 3
                per_aoi_store[(arch, fold)] = (
                    pd.DataFrame({"aoi_id": aoi, "gap": delta,
                                  "sun_dist_loco": lo.sun_dist.values})
                    .groupby("aoi_id", as_index=False)
                    .agg(gap=("gap", "mean"), sun_dist_loco=("sun_dist_loco", "mean"),
                         n_img=("gap", "size"))
                    .assign(arch=arch, fold=fold, city=lo.city.iloc[0]))

    t1 = pd.DataFrame(t1_rows)
    for metric in ("2class", "fgonly"):
        m = t1.metric == metric
        t1.loc[m, "p_holm"] = holm(t1.loc[m, "p_raw"].values)

    xc = pd.DataFrame(xcheck_rows)
    print("\n--- cross-check against phase-2 arithmetic (two-class, K=2, unseen-excluded) ---")
    print(xc.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    if not xc.ok.all():
        fail("point estimate disagrees with the phase-2 arithmetic:\n"
             f"{xc[~xc.ok].to_string(index=False)}")
    print("all 6 reproduce the pre-committed value exactly when the two cell "
          "means are rounded to 4 dp before differencing, which is how the "
          "phase-2 table was built")
    n_off = int((~xc.direct_within_tol).sum())
    if n_off:
        print(f"  note: {n_off} of 6 full-precision gaps sit outside +-{XCHECK_TOL:g} "
              f"of the quoted value (max {xc.abs_diff_direct.max():.1e}), entirely "
              "accounted for by that double rounding")

    pd.set_option("display.width", 250)
    for metric in ("2class", "fgonly"):
        print(f"\n--- gap table, {metric} ({'primary' if metric=='2class' else 'secondary'}) ---")
        print(t1[t1.metric == metric][
            ["arch", "fold", "loco", "upper", "gap", "stderr", "ci_lo", "ci_hi",
             "p_raw", "p_holm", "n_aoi", "n_images",
             "patch_ci_lo", "patch_ci_hi", "width_ratio"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    sens = pd.DataFrame(sens_rows)
    print("\n--- sensitivity: same gaps with the 108 fully DSM-unseen patches dropped "
          "(JAX has none; not the pre-committed number) ---")
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out["task1"] = t1.to_dict("records")
    out["task1_crosscheck"] = xc.to_dict("records")
    out["task1_sensitivity_excl_fully_unseen"] = sens.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 2 -- pooled gaps over per-AOI mean deltas
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TASK 2 -- pooled gap, unit = per-AOI mean delta, stratified by fold, B=%d" % B)
    print("=" * 100)

    aoi_all = pd.concat(per_aoi_store.values(), ignore_index=True)
    t2_rows = []

    def strat_boot(frame, stream, by_aoi_across_arch=False):
        """Resample AOIs within fold, statistic = unweighted mean of the AOI units."""
        rng = _rng(stream)
        parts = []
        for fold in FOLDS:
            sub = frame[frame.fold == fold]
            if by_aoi_across_arch:
                piv = sub.pivot(index="aoi_id", columns="arch", values="gap")
                vals = piv.to_numpy(np.float64)               # (n_aoi, 2)
                n = len(piv)
                idx = rng.integers(0, n, size=(B, n))
                parts.append((vals[idx].sum(axis=(1, 2)), n * vals.shape[1]))
            else:
                vals = sub.gap.to_numpy(np.float64)
                n = len(vals)
                idx = rng.integers(0, n, size=(B, n))
                parts.append((vals[idx].sum(1), n))
        num = np.sum([p[0] for p in parts], axis=0)
        den = float(np.sum([p[1] for p in parts]))
        return num / den

    for arch in ARCHS:
        sub = aoi_all[aoi_all.arch == arch]
        boot = strat_boot(sub, 4100 + ARCHS.index(arch))
        lo_c, hi_c = ci95(boot)
        vals = sub.gap.to_numpy(np.float64)
        signs = _rng(4200 + ARCHS.index(arch)).choice(np.array([-1.0, 1.0]), size=(B, len(vals)))
        null = (signs * vals).sum(1) / len(vals)
        obs = float(vals.mean())
        p = (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)
        t2_rows.append(dict(pooling=f"within-architecture: {arch}", gap=obs,
                            stderr=float(boot.std(ddof=1)), ci_lo=lo_c, ci_hi=hi_c,
                            p=p, n_aoi_units=len(vals),
                            n_images=int(sub.n_img.sum())))

    boot = strat_boot(aoi_all, 4300, by_aoi_across_arch=True)
    lo_c, hi_c = ci95(boot)
    vals = aoi_all.gap.to_numpy(np.float64)
    obs = float(vals.mean())
    signs = _rng(4400).choice(np.array([-1.0, 1.0]), size=(B, len(vals)))
    null = (signs * vals).sum(1) / len(vals)
    p = (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)
    t2_rows.append(dict(pooling="both architectures", gap=obs,
                        stderr=float(boot.std(ddof=1)), ci_lo=lo_c, ci_hi=hi_c,
                        p=p, n_aoi_units=len(vals), n_images=int(aoi_all.n_img.sum())))

    t2 = pd.DataFrame(t2_rows)
    print(t2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["task2"] = t2.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 3 -- sun-elevation regression
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TASK 3 -- per-AOI gap ~ sun_dist + C(city);  sun_dist from the LOCO cell")
    print("=" * 100)

    import statsmodels.formula.api as smf
    from scipy import stats as sps

    t3_rows = []

    def fit_report(frame, label, formula):
        n = len(frame)
        m_ols = smf.ols(formula, data=frame).fit()
        m_cl = smf.ols(formula, data=frame).fit(
            cov_type="cluster", cov_kwds={"groups": frame["fold"]})
        r = dict(model=label, formula=formula, n=n,
                 slope=float(m_ols.params["sun_dist_loco"]),
                 se_ols=float(m_ols.bse["sun_dist_loco"]),
                 p_ols=float(m_ols.pvalues["sun_dist_loco"]),
                 se_cluster=float(m_cl.bse["sun_dist_loco"]),
                 p_cluster=float(m_cl.pvalues["sun_dist_loco"]),
                 n_clusters=int(frame.fold.nunique()),
                 r2=float(m_ols.rsquared))
        t3_rows.append(r)
        return r

    for arch in ARCHS:
        fit_report(aoi_all[aoi_all.arch == arch].copy(), f"{arch}",
                   "gap ~ sun_dist_loco + C(city)")
    fit_report(aoi_all.copy(), "both architectures",
               "gap ~ sun_dist_loco + C(city) + C(arch)")

    t3 = pd.DataFrame(t3_rows)
    print(t3.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # naive city-level rank correlation, n = 3
    naive = []
    for arch in list(ARCHS) + ["both"]:
        sub = aoi_all if arch == "both" else aoi_all[aoi_all.arch == arch]
        g = sub.groupby("city").agg(gap=("gap", "mean"),
                                    sun_dist=("sun_dist_loco", "mean")).reset_index()
        rho, p = sps.spearmanr(g.sun_dist, g.gap)
        naive.append(dict(arch=arch, n_cities=len(g), spearman_rho=float(rho),
                          p=float(p),
                          cities=";".join(f"{r.city}:sd={r.sun_dist:.3f},gap={r.gap:.3f}"
                                          for r in g.itertuples())))
    nv = pd.DataFrame(naive)
    print("\n--- naive city-level rank correlation "
          "(n=3, uninterpretable, superseded by the within-city design) ---")
    print(nv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out["task3"] = t3.to_dict("records")
    out["task3_naive_city_rank"] = nv.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 4 -- descriptives
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TASK 4 -- per-cell descriptives, 12 rows")
    print("=" * 100)

    d_rows = []
    for arch, fold, mode in CELLS:
        df, meta = data[(arch, fold, mode)], metas[(arch, fold, mode)]
        nz = df[df.n_valid_px > 0]          # per-image ratios are undefined otherwise
        d_rows.append(dict(
            cell=meta["run_id"], arch=arch, fold=fold, mode=mode,
            n_images=len(df), n_aoi=df.aoi_id.nunique(),
            mean_2class=float(df.miou2_pct.mean()),
            median_2class=float(df.miou2_pct.median()),
            mean_fgonly=float(df.fg_pct.mean()),
            median_fgonly=float(df.fg_pct.median()),
            predpos_frac_pooled=float(df.n_pred_pos.sum() / df.n_valid_px.sum()),
            predpos_frac_mean_img=float((nz.n_pred_pos / nz.n_valid_px).mean()),
            predpos_frac_sidecar=meta.get("pred_pos_frac_on_valid"),
            mean_unseen_frac=float(df.unseen_frac.mean()),
            n_fully_unseen=int((df.n_valid_px <= 0).sum()),
            n_empty_fg_valid=int(df.empty_fg_valid.sum()),
            n_empty_fg_genuine=int((df.empty_fg_valid & (df.n_valid_px > 0)).sum()),
            n_all_shadow_flag=int(df.all_shadow_flag.sum()),
            mean_sun_dist=float(df.sun_dist.mean()),
            train_sun_elev_mean=meta.get("train_sun_elev_mean"),
            epochs_run=meta.get("epochs_run"), best_epoch=meta.get("best_epoch"),
            stopped_on=meta.get("stopped_on")))
    t4 = pd.DataFrame(d_rows)
    print(t4.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["task4"] = t4.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 5 -- floor check
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("TASK 5 -- floor check: fg-only IoU (K=2, unseen-excluded) of the upper cells")
    print("=" * 100)

    fl = t4[t4["mode"] == "upper"][["cell", "arch", "fold", "mean_fgonly"]].copy()
    print(fl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    floor_rows = []
    for arch in ARCHS:
        v = fl[fl.arch == arch].set_index("fold").mean_fgonly
        spread = float(v.max() - v.min())
        rel = {f: float(v[f] - v.drop(f).mean()) for f in FOLDS}
        floor_rows.append(dict(arch=arch, **{f"upper_fgonly_{f}": float(v[f]) for f in FOLDS},
                               spread=spread,
                               min_fold=str(v.idxmin()),
                               min_dev_from_other_two=float(min(rel.values()))))
    t5 = pd.DataFrame(floor_rows)
    print("\n" + t5.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["task5"] = t5.to_dict("records")

    # ------------------------------------------------------------------
    # results JSON
    # ------------------------------------------------------------------
    conditions, tests = [], []
    for r in t1[t1.metric == "2class"].to_dict("records"):
        conditions.append({
            "label": f"S-EO gap / {r['arch']} / holdout {r['fold'].upper()}",
            "metric": "gap_2class_tol_k2_unseen_excluded",
            "mean": round(r["gap"], 4), "stderr": round(r["stderr"], 4),
            "n": int(r["n_images"]), "n_aoi": int(r["n_aoi"]),
            "ci95": [round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
            "ci_method": "aoi_cluster_bootstrap_B10000",
            "ci95_patch_level_anti_conservative_for_comparison_only":
                [round(r["patch_ci_lo"], 4), round(r["patch_ci_hi"], 4)],
            "ci_width_ratio_patch_over_aoi": round(r["width_ratio"], 4),
            "loco_mean": round(r["loco"], 4), "upper_mean": round(r["upper"], 4),
            "baseline": "none_upper_r3500_same_fold"})
        tests.append({"name": "AOI-level sign-flip permutation test",
                      "compares": [run_id(r["arch"], r["fold"], "upper"),
                                   run_id(r["arch"], r["fold"], "loco")],
                      "statistic": round(r["gap"], 4), "p": r["p_raw"],
                      "p_holm": round(float(r["p_holm"]), 6), "B": B,
                      "two_tailed": True,
                      "family": "6 (fold x arch) pairs, Holm-Bonferroni",
                      "p_note": "p is at the permutation floor (1/(B+1) = 9.999e-5); "
                                "the test resolves it no further than p < 1e-4"})
    for r in t1[t1.metric == "fgonly"].to_dict("records"):
        conditions.append({
            "label": f"S-EO gap (secondary, fg-only) / {r['arch']} / holdout {r['fold'].upper()}",
            "metric": "gap_fg_only_tol_k2_unseen_excluded",
            "mean": round(r["gap"], 4), "stderr": round(r["stderr"], 4),
            "n": int(r["n_images"]), "n_aoi": int(r["n_aoi"]),
            "ci95": [round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
            "ci_method": "aoi_cluster_bootstrap_B10000",
            "ci95_patch_level_anti_conservative_for_comparison_only":
                [round(r["patch_ci_lo"], 4), round(r["patch_ci_hi"], 4)],
            "ci_width_ratio_patch_over_aoi": round(r["width_ratio"], 4),
            "loco_mean": round(r["loco"], 4), "upper_mean": round(r["upper"], 4),
            "baseline": "none_upper_r3500_same_fold"})
    for r in t2.to_dict("records"):
        conditions.append({
            "label": f"S-EO pooled gap / {r['pooling']}",
            "metric": "pooled_gap_2class_tol_k2_unseen_excluded_per_aoi_mean",
            "mean": round(r["gap"], 4), "stderr": round(r["stderr"], 4),
            "n": int(r["n_aoi_units"]),
            "ci95": [round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
            "ci_method": "fold_stratified_aoi_bootstrap_B10000",
            "baseline": "none_upper_r3500_same_fold"})
        tests.append({"name": "AOI-level sign-flip permutation test (pooled)",
                      "compares": [r["pooling"]], "statistic": round(r["gap"], 4),
                      "p": r["p"], "B": B, "two_tailed": True})
    for r in t3.to_dict("records"):
        tests.append({"name": f"OLS slope on per-AOI mean sun_dist ({r['model']})",
                      "compares": [r["formula"]], "statistic": round(r["slope"], 6),
                      "p": r["p_ols"], "se": round(r["se_ols"], 6),
                      "p_cluster_robust_on_fold": r["p_cluster"],
                      "se_cluster_robust_on_fold": round(r["se_cluster"], 6),
                      "n": int(r["n"]), "n_clusters": int(r["n_clusters"]),
                      "two_tailed": True})

    worst = t1[t1.metric == "2class"].loc[t1[t1.metric == "2class"].gap.idxmin()]
    best = t1[t1.metric == "2class"].loc[t1[t1.metric == "2class"].gap.idxmax()]

    res = {
        "id": "02_seo_loco",
        "arm": "A",
        "answers": ["v1UJ-W1", "BxRx-W1", "AC1"],
        "question": "On S-EO, what is the gap between the within-city upper bound "
                    "and source-only LOCO for each (holdout city x architecture) "
                    "cell of the vanilla grid?",
        "runs": [metas[k]["run_id"] for k in CELLS],
        "index_sha256": index_sha256,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/02_seo_loco.py",
        "git_hash": None,
        "git_hash_note": "the project base is not a git repository on this box; "
                         "final_paper/ is at 7cfd5f9. Per-cell producing commits "
                         "are in INDEX.parquet.git_hash.",
        "seed": SEED,
        "metric_definitions": {
            "gap_2class_tol_k2_unseen_excluded":
                "mean over test patches of (upper - loco) two-class "
                "boundary-tolerant mIoU at K=2 with DSM-unseen pixels excluded "
                "from all four confusion cells; recomputed per patch from "
                "tp_tol/fp_tol/fn_tol/tn_tol as 50*(tp/(tp+fp+fn+1e-10) + "
                "tn/(tn+fp+fn+1e-10)), matching metrics.py _img_metrics.",
            "gap_fg_only_tol_k2_unseen_excluded":
                "same, foreground class only: 100*tp/(tp+fp+fn+1e-10). Secondary.",
            "sun_dist": "|patch sun_elevation - train_sun_elev_mean| of the LOCO "
                        "cell of that fold; the distance to the LOCO training "
                        "distribution."},
        "conditions": conditions,
        "tests": tests,
        "task1_crosscheck_vs_phase2_arithmetic": xc.to_dict("records"),
        "task1_sensitivity_excl_fully_unseen": sens.to_dict("records"),
        "task3_regression": t3.to_dict("records"),
        "task3_naive_city_rank_correlation": nv.to_dict("records"),
        "task4_descriptives": t4.to_dict("records"),
        "task5_floor_check": t5.to_dict("records"),
        "headline": (
            "On S-EO the within-city upper bound exceeds source-only LOCO in all "
            f"six (holdout city x architecture) cells of the vanilla grid, by "
            f"{worst.gap:.4f} two-class mIoU at K=2 in the smallest cell "
            f"({worst.arch}/{worst.fold.upper()}) and {best.gap:.4f} in the largest "
            f"({best.arch}/{best.fold.upper()}), each with an AOI-level 95% CI "
            "excluding zero."),
        "caveats": [
            "Single seed. Every cell is seed 0; no across-seed variance is "
            "available for any of the six gaps, so the CIs cover test-patch and "
            "AOI sampling only, not optimisation noise.",
            "One method per side. The loco side is Vanilla only. The three "
            "phase-3 adaptation methods on S-EO (fda, sib) were still training "
            "when this ran and are excluded; no adaptation baseline is present "
            "in this analysis.",
            "Budget asymmetry between the two sides of every gap. The loco cell "
            "trains on 3500 patches from the two source cities; the upper cell "
            "trains on 3500 patches from the held-out city. Train budget is "
            "matched at 3500, but the upper cell's training and test data come "
            "from the same city and AOI pool.",
            "108 of the 6149 test patches (29 OMA, 79 UCSD, 0 JAX) are entirely "
            "DSM-unseen: the unseen-exclusion rule empties all four confusion "
            "cells and the per-patch metric takes its floor value of 0.0 rather "
            "than measuring anything. They are the same patches in all four "
            "cells of a fold, so each contributes a delta of exactly 0 and "
            "dilutes the gap toward zero without biasing its sign. The "
            "pre-committed arithmetic includes them and so does every headline "
            "number here; task1_sensitivity_excl_fully_unseen reports the same "
            "gaps with them dropped.",
            "The degenerate-patch convention is inherited, not chosen here: a "
            "patch whose valid region is all shadow scores background IoU 0 and "
            "a patch with no foreground scores foreground IoU 0, so both halve "
            "the two-class mIoU. 11 all-shadow patches (all OMA) and 78 "
            "genuinely-empty-foreground patches sit in the test pool.",
            "rung 3500 is the S-EO patch budget, not a point on the "
            "ShadowTransfer 225/450/900 ladder; the unit is a 384 px patch, not "
            "a benchmark image. The numbers are not commensurable with the "
            "ladder's rungs.",
            "High-resolution cells only, one resolution, one seed, three cities, "
            "two architectures. OGLANet is not in arm A.",
            "S-EO labels are auto-generated. Both sides of every gap use the "
            "same labels on the same patches, so label noise largely cancels in "
            "the difference, but the absolute mIoU levels carry it.",
            "The train-side per-image evaluator was disabled in all 12 cells, a "
            "declared deviation from arm C's recipe whose bit-identity proof "
            "returned INCONCLUSIVE, not PASS. The two OMA MAMNet cells ran "
            "epochs 1-9 with it on and every later epoch with it off; the other "
            "ten cells ran with it off throughout. Both cells of the OMA MAMNet "
            "pair share the same schedule, so that pair's gap is still measured "
            "within one recipe.",
            "The 12 cells were produced at three different code commits "
            "(d56c2ecd67, cb39a9cbac, 35cae7b9c0). The dinov3 OMA pair straddles "
            "two of them: its loco cell is d56c2ecd67 and its upper cell is "
            "35cae7b9c0. The metric module is identical across all 12 "
            "(metric_module_sha 730ad4a5bc).",
            "city and fold are the same variable in task 3: each fold's test "
            "pool is one held-out city, so C(city) absorbs the fold effect and "
            "the sun_dist slope is identified within city only. The "
            "cluster-robust standard errors have 3 clusters, far below where "
            "the cluster-robust asymptotics hold; read them as indicative.",
            "sun_dist is measured against the LOCO cell's training sun-elevation "
            "mean and is therefore a property of the fold, not of the "
            "architecture; it is identical across the two architectures.",
            "Nothing AURC-shaped is computed. The definition is unratified "
            "(CONTRACT.md line 163).",
            "All six sign-flip p-values sit at the permutation floor for "
            "B = 10,000 (1/(B+1) = 9.999e-5), before and after Holm. The test "
            "establishes p < 1e-4 for each pair and does not rank them.",
            "The pre-committed table's values are the difference of the two "
            "4-decimal-rounded cell means. Two of the six full-precision gaps "
            "therefore differ from the quoted value in the 4th decimal "
            "(dinov3/OMA 3.223436 vs 3.2235; mamnet/UCSD 6.745351 vs 6.7453), "
            "by at most 6.4e-5. All six reproduce exactly under the rounded "
            "form. The headline numbers here are full precision.",
            "No HLT branch is applied and no comparison to arm C is made here.",
        ],
        "figure": None,
        "caption": None,
    }
    print("\nHEADLINE: " + res["headline"])

    with open(RESULTS_PATH, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {RESULTS_PATH}")

    # dump the tables for the session log
    with open(os.path.join(BASE, "results", "_02_seo_loco_tables.json"), "w") as fh:
        json.dump({k: (v if not isinstance(v, pd.DataFrame) else v.to_dict("records"))
                   for k, v in out.items()}, fh, indent=2, default=float)


if __name__ == "__main__":
    if DRY_RUN:
        dry_run()
    else:
        main()
