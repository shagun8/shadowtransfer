#!/usr/bin/env python
"""
Arm A -- external LOCO on S-EO, method cells (SIB, FDA) against vanilla-LOCO.

Answers v1UJ-W2.

  TASK 1  method vs vanilla-LOCO per cell, AOI bootstrap, Holm within family
  TASK 2  the pre-committed worst-case criterion
  TASK 3  fraction of the vanilla gap closed
  TASK 4  per-cell descriptives

The pre-committed criterion, fixed before any of these numbers were seen:
SIB's claim is WORST-CASE safety, not best mean. SIB's worst cell across its
(fold x arch) cells is no worse than vanilla-LOCO's worst cell. Mean deltas
are reported and are NOT the criterion. FDA is context, not a claim.

Access boundary, per logs/armA_phase3_manifest.json: dinov3-sib is source_only,
mamnet-sib is unlabeled_target, both FDA arms are unlabeled_target. Nothing is
pooled across that boundary anywhere in this file.

AS RUN: the complete grid, 24 cells. mamnet-sib-loco-r3500-jax-hr-s0 landed on
2026-08-03 and both method families are 6 cells, so Holm is sized SIB-6, FDA-6
and no worst-case number is provisional. The machinery for a partial grid is
retained in KNOWN_MISSING and is inert while that set is empty.

Nothing AURC-shaped is computed. No comparison to internal SIB numbers. The
branch reading is the operator's; this file states the criterion outcome and
stops.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/03_seo_methods.py
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
MANIFEST_PATH = os.path.join(BASE, "logs", "armA_phase3_manifest.json")
RESULTS_PATH = os.path.join(BASE, "results", "03_seo_methods.json")
TABLES_PATH = os.path.join(BASE, "results", "_03_seo_methods_tables.json")

ARCHS = ["mamnet", "dinov3"]
FOLDS = ["oma", "jax", "ucsd"]
METHODS = ["sib", "fda"]
CONDITIONS = {
    "loco": "vanilla-loco",
    "upper": "none-upper",
    "sib": "sib-loco",
    "fda": "fda-loco",
}

# Cells the operator has declared absent. Anything else missing stops the run.
# 2026-08-03: mamnet-sib-loco-r3500-jax-hr-s0 landed; the grid is complete at 24
# cells and the SIB Holm family is back to 6.
KNOWN_MISSING = set()
KNOWN_MISSING_REASON = ("still training on Delta at analysis time; arrives later. "
                        "Reported as absent, never imputed.")

# Sanity flags that are known artifacts, per the operator. Recorded on every row
# they appear on, never used to exclude a cell.
KNOWN_ARTIFACT_FLAGS = {
    "SIB KL term was never added to the loss": (
        "known --dump_only artifact. sib_runtime.kl_applied_steps is a "
        "process-local counter: a dump-only re-run executes no training step, so "
        "it reads 0 regardless of training history. Positive controls are the "
        "three SIB cells that were not dump-only, kl_applied_steps 18,791 "
        "(mamnet/JAX), 22,724 (dinov3/JAX) and 25,346 (dinov3/UCSD). Verified "
        "false by the operator."),
}

EXPECTED_ROWS = {"oma": 1637, "jax": 2642, "ucsd": 1870}
EXPECTED_NCOL = 76
EXPECTED_VANILLA_TARGET_ACCESS = {"loco": "source_only", "upper": "target_labels"}
FDA_POOL_SPLIT_EXPECTED = "train"

B = 10_000
SEED = 0                       # every rng below is default_rng([SEED, stream_id])
EPS = 1e-10                    # metrics.py _img_metrics epsilon, matched exactly

VANILLA_CELLS = [(a, f, c) for a in ARCHS for f in FOLDS for c in ("loco", "upper")]
METHOD_CELLS = [(a, f, m) for a in ARCHS for f in FOLDS for m in METHODS
                if (a, f, m) not in KNOWN_MISSING]
CELLS = VANILLA_CELLS + METHOD_CELLS

# folds each (arch, method) actually covers, after the missing cell
COVERED = {(a, m): [f for f in FOLDS if (a, f, m) in METHOD_CELLS]
           for a in ARCHS for m in METHODS}


def run_id(arch, fold, cond):
    return f"{arch}-{CONDITIONS[cond]}-r3500-{fold}-hr-s0"


# ----------------------------------------------------------------------------
# manifest -- the target_access authority
# ----------------------------------------------------------------------------

def load_manifest():
    with open(MANIFEST_PATH) as fh:
        man = json.load(fh)
    by_id = {c["run_id"]: c for c in man["cells"]}
    if len(by_id) != len(man["cells"]):
        fail("manifest contains duplicate run_ids")
    return man, by_id


# ----------------------------------------------------------------------------
# loading and validation
# ----------------------------------------------------------------------------

def fail(msg):
    raise SystemExit(f"ASSERTION FAILED: {msg}")


def preflight():
    """Stop before reading anything if a cell that should be here is not.

    CONTRACT.md 7: a dump without its run.json sidecar describes an unknown
    configuration and is unusable. A cell without .DONE may be half-copied.
    """
    bad, absent = [], []
    for arch, fold, cond in VANILLA_CELLS + [(a, f, m) for a in ARCHS for f in FOLDS
                                             for m in METHODS]:
        rid = run_id(arch, fold, cond)
        d = os.path.join(DUMP_DIR, rid)
        why = []
        if not os.path.isdir(d):
            why.append("no directory")
        else:
            for f in (".DONE", "perimage.parquet", "run.json"):
                if not os.path.isfile(os.path.join(d, f)):
                    why.append(f"no {f}")
        if why:
            if (arch, fold, cond) in KNOWN_MISSING:
                absent.append((rid, "; ".join(why)))
            else:
                bad.append((rid, "; ".join(why)))
    if bad:
        print(f"\n{len(bad)} unexpected incomplete cells:")
        for rid, why in bad:
            print(f"  {rid:44s}  {why}")
        fail(f"{len(bad)} cells incomplete and not on the declared-absent list")
    print(f"preflight: {len(CELLS)} cells complete (.DONE + parquet + run.json)")
    for rid, why in absent:
        print(f"preflight: DECLARED ABSENT  {rid}  [{why}] -- {KNOWN_MISSING_REASON}")
    if len(absent) != len(KNOWN_MISSING):
        fail(f"declared-absent list has {len(KNOWN_MISSING)} entries but "
             f"{len(absent)} are actually absent; the list is stale")


def load_cell(arch, fold, cond):
    rid = run_id(arch, fold, cond)
    d = os.path.join(DUMP_DIR, rid)
    pq = os.path.join(d, "perimage.parquet")
    df = pd.read_parquet(pq)
    with open(os.path.join(d, "run.json")) as fh:
        meta = json.load(fh)
    return rid, df, meta, pq


def add_metrics(df):
    """Recompute both reported metrics from the raw K=2 counts.

    two-class boundary-tolerant mIoU at K=2, unseen-excluded  -> miou2_pct
    foreground-only IoU at K=2, unseen-excluded               -> fg_pct

    Unseen pixels are already dropped from all four confusion cells upstream
    (run.json unseen_handling), so tp+fp+fn+tn == n_valid_px. Matches
    metrics.py _img_metrics exactly, including the epsilon.
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


def mblock(meta):
    return meta.get("method_block") or {}


def classify_flags(meta):
    """Split sanity_flags into known artifacts and everything else."""
    flags = meta.get("sanity_flags") or []
    known = [f for f in flags if f in KNOWN_ARTIFACT_FLAGS]
    other = [f for f in flags if f not in KNOWN_ARTIFACT_FLAGS]
    return flags, known, other


# ----------------------------------------------------------------------------
# dry run -- assertions only, writes nothing
# ----------------------------------------------------------------------------

def check_cell(arch, fold, cond, df, meta, man_by_id):
    """Every per-cell assertion. Returns (checks, target_access, extras)."""
    rid = run_id(arch, fold, cond)
    mb = mblock(meta)
    checks = []

    ncol = df.shape[1]
    checks.append(("ncol==76", ncol == EXPECTED_NCOL, ncol))
    want = EXPECTED_ROWS[fold]
    checks.append((f"nrow=={want}", len(df) == want, len(df)))
    checks.append(("aoi_id non-null", int(df["aoi_id"].isna().sum()) == 0,
                   int(df["aoi_id"].isna().sum())))
    checks.append(("sun_elevation non-null", int(df["sun_elevation"].isna().sum()) == 0,
                   int(df["sun_elevation"].isna().sum())))

    zero = df["n_valid_px"] <= 0
    checks.append(("n_valid_px==0 rows are all fully DSM-unseen",
                   bool((df.loc[zero, "unseen_frac"] == 1.0).all()
                        and (df.loc[zero, "n_seen_px"] == 0).all()
                        and (df.loc[zero, ["tp_tol", "fp_tol", "fn_tol",
                                           "tn_tol"]].to_numpy() == 0).all()),
                   f"{int(zero.sum())} rows"))
    checks.append(("tp+fp+fn+tn==n_valid_px",
                   bool(((df.tp_tol + df.fp_tol + df.fn_tol + df.tn_tol)
                         == df.n_valid_px).all()), "ok"))

    ta = meta.get("target_access")
    if cond in ("loco", "upper"):
        exp_ta, src = EXPECTED_VANILLA_TARGET_ACCESS[cond], "phase-2 (EXP-02)"
    else:
        exp_ta, src = man_by_id[rid]["target_access"], "phase-3 manifest"
    checks.append((f"target_access=={exp_ta} [{src}]", ta == exp_ta, ta))

    if cond in METHODS:
        entry = man_by_id[rid]
        # the sidecar's own method_block must agree with the manifest
        checks.append(("method_block.target_access==manifest",
                       mb.get("target_access") == entry["target_access"],
                       mb.get("target_access")))
        checks.append(("method==manifest", meta.get("method") == entry["method"],
                       meta.get("method")))
        checks.append(("holdout==manifest",
                       str(meta.get("holdout")).lower() == entry["holdout"].lower(),
                       meta.get("holdout")))
        if cond == "sib":
            got = sorted(mb.get("sib_components_active") or [])
            checks.append(("sib_components_active==manifest",
                           got == sorted(entry["components_active"]), ",".join(got)))
        # fda_target_pool_split == "train" wherever the manifest declares a pool
        if entry.get("fda_target_pool"):
            v = mb.get("fda_target_pool_split")
            checks.append((f"fda_target_pool_split=={FDA_POOL_SPLIT_EXPECTED}",
                           v == FDA_POOL_SPLIT_EXPECTED, v))
            checks.append(("fda_target_pool_split==manifest",
                           v == entry["fda_target_pool"]["split"], v))
            checks.append(("fda_target_pool_city==manifest",
                           mb.get("fda_target_pool_city") == entry["fda_target_pool"]["city"],
                           mb.get("fda_target_pool_city")))
            checks.append(("fda_target_pool_labels_used==False",
                           mb.get("fda_target_pool_labels_used") is False,
                           mb.get("fda_target_pool_labels_used")))
            checks.append(("fda_target_pool disjoint from cell splits",
                           mb.get("fda_target_pool_disjoint_from_cell_splits") is True,
                           mb.get("fda_target_pool_disjoint_from_cell_splits")))
        else:
            checks.append(("no FDA pool declared -> fda_active False",
                           mb.get("fda_active") is False, mb.get("fda_active")))

    checks.append(("run_id matches dir", meta["run_id"] == rid, meta["run_id"]))
    checks.append(("run_id col uniform",
                   df.run_id.nunique() == 1 and df.run_id.iloc[0] == rid, "ok"))
    checks.append(("dataset==seo", meta.get("dataset") == "seo", meta.get("dataset")))
    checks.append(("arm==A", meta.get("arm") == "A", meta.get("arm")))
    checks.append(("rung==3500", meta.get("rung") == 3500, meta.get("rung")))
    checks.append(("seed==0", meta.get("seed") == 0, meta.get("seed")))
    checks.append(("res==hr", meta.get("res") == "hr", meta.get("res")))
    checks.append(("n_images==n_test_expected",
                   len(df) == meta.get("n_test_expected"), meta.get("n_test_expected")))
    checks.append(("split==test", set(df.split.unique()) == {"test"},
                   list(df.split.unique())))
    checks.append(("city==holdout", set(df.city.unique()) == {meta["holdout"]},
                   list(df.city.unique())))
    checks.append(("smoke==False", meta.get("smoke") is False, meta.get("smoke")))
    checks.append(("parent_run_id is null", meta.get("parent_run_id") is None,
                   meta.get("parent_run_id")))

    dfm = add_metrics(df)
    d_miou = float(np.abs(dfm.miou2_pct - df.miou_tol_pct).max())
    d_fg = float(np.abs(dfm.fg_pct - df.iou_tol * 100.0).max())
    checks.append(("recompute==miou_tol_pct", d_miou < 1e-9, f"{d_miou:.2e}"))
    checks.append(("recompute==iou_tol", d_fg < 1e-9, f"{d_fg:.2e}"))

    return checks, ta


def dry_run():
    print("=" * 118)
    print("ARM A / S-EO method cells (SIB, FDA) -- DRY RUN")
    print(f"dumps:    {DUMP_DIR}")
    print(f"manifest: {MANIFEST_PATH}  (target_access authority)")
    print("=" * 118)

    preflight()
    man, man_by_id = load_manifest()
    print(f"manifest: {len(man_by_id)} phase-3 cells, submitted {man['submitted_utc']}, "
          f"git {man['git_hash'][:10]}, sib_variant {man['sib_variant']!r}")

    data, metas, paths, rows = {}, {}, {}, []
    colsets, dtypes_ref = {}, None

    for key in CELLS:
        arch, fold, cond = key
        rid, df, meta, pq = load_cell(*key)
        data[key], metas[key], paths[key] = df, meta, pq

        checks, ta = check_cell(arch, fold, cond, df, meta, man_by_id)

        colsets[key] = list(df.columns)
        if dtypes_ref is None:
            dtypes_ref, dtype_ok = df.dtypes, True
        else:
            dtype_ok = bool(df.dtypes.equals(dtypes_ref))
        checks.append(("dtypes match cell 1", dtype_ok, "ok" if dtype_ok else "DRIFT"))

        _, known, other = classify_flags(meta)
        bad = [c for c in checks if not c[1]]
        rows.append(dict(cell=rid, ncol=df.shape[1], nrow=len(df),
                         n_aoi=df.aoi_id.nunique(), target_access=ta,
                         pool_split=mblock(meta).get("fda_target_pool_split"),
                         stopped_on=meta.get("stopped_on"),
                         dump_only=meta.get("dump_only"),
                         known_artifact_flags=len(known), other_flags=len(other),
                         failed=len(bad),
                         detail=";".join(f"{c[0]}={c[2]}" for c in bad)))
        for name, ok, val in checks:
            if not ok:
                print(f"  !! {rid}: {name} -> {val}")

    tab = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 70)
    print(f"\n--- per-cell assertions, {len(CELLS)} cells ---")
    print(tab.to_string(index=False))
    if tab.failed.sum():
        fail(f"{int(tab.failed.sum())} per-cell assertions failed; see above")

    ref = colsets[CELLS[0]]
    for k, v in colsets.items():
        if v != ref:
            fail(f"{k}: column name/order differs from reference cell")
    print(f"\ncolumn name set + order identical across all {len(CELLS)} cells: ok")

    # per fold, every present cell covers an identical patch set
    print("\n--- per-fold patch-set identity "
          "(key: image_id; no patch_id column exists in the schema) ---")
    for fold in FOLDS:
        present = [k for k in CELLS if k[1] == fold]
        ks, ks2 = {}, {}
        for k in present:
            df = data[k]
            if df.image_id.duplicated().any():
                fail(f"{run_id(*k)}: duplicate image_id")
            ks[k] = set(df.image_id)
            ks2[k] = set(zip(df.source_image_id, df.patch_index))
        rk, rk2 = ks[(ARCHS[0], fold, "loco")], ks2[(ARCHS[0], fold, "loco")]
        for k, v in ks.items():
            if v != rk:
                fail(f"{run_id(*k)}: image_id set differs ({len(v ^ rk)} symmetric diff)")
        for k, v in ks2.items():
            if v != rk2:
                fail(f"{run_id(*k)}: (source_image_id, patch_index) set differs")
        n_aoi = data[(ARCHS[0], fold, "loco")].aoi_id.nunique()
        miss = [run_id(*k) for k in KNOWN_MISSING if k[1] == fold]
        print(f"  {fold:5s}: {len(present)}/{len(present)} present cells share "
              f"{len(rk)} image_ids over {n_aoi} AOIs -- ok"
              + (f"   [absent: {', '.join(miss)}]" if miss else ""))

    # n_valid_px is a property of the DSM and the labels, not of the model
    print("\n--- n_valid_px consistency with the fold's vanilla cells ---")
    for fold in FOLDS:
        base = data[(ARCHS[0], fold, "loco")].sort_values("image_id")
        ref_nv = base.set_index("image_id").n_valid_px
        ref_z = set(base.loc[base.n_valid_px <= 0, "image_id"])
        for k in [c for c in CELLS if c[1] == fold]:
            m = data[k].sort_values("image_id").set_index("image_id").n_valid_px
            if not m.equals(ref_nv):
                fail(f"{run_id(*k)}: n_valid_px differs from the fold's vanilla cell "
                     f"on {int((m != ref_nv).sum())} patches; the delta would not be paired")
            if set(data[k].query("n_valid_px <= 0").image_id) != ref_z:
                fail(f"{run_id(*k)}: fully-unseen patch set differs from vanilla")
        g = base.assign(z=base.n_valid_px <= 0).groupby("aoi_id").z.agg(["size", "sum"])
        if len(g[g["sum"] == g["size"]]):
            fail(f"{fold}: an AOI is entirely fully-unseen -> degenerate AOI unit")
        print(f"  {fold:5s}: n_valid_px elementwise identical in every present cell; "
              f"{len(ref_z):4d} fully-unseen patches identical in all, "
              f"touching {int((g['sum'] > 0).sum())} of {len(g)} AOIs, none wholly")

    for fold in FOLDS:
        ref_map = data[(ARCHS[0], fold, "loco")].set_index("image_id").aoi_id.sort_index()
        for k in [c for c in CELLS if c[1] == fold]:
            if not data[k].set_index("image_id").aoi_id.sort_index().equals(ref_map):
                fail(f"{run_id(*k)}: image_id -> aoi_id mapping differs")
    print("image_id -> aoi_id mapping identical within every fold: ok")

    seen = {}
    for k, meta in metas.items():
        rid, sha = meta["run_id"], meta.get("config_sha256")
        if rid in seen and seen[rid] != sha:
            fail(f"{rid}: duplicate run_id with differing config_sha256")
        seen[rid] = sha
    print(f"run_id uniqueness / config_sha256 consistency: ok ({len(seen)} run_ids)")

    # ---- sanity flags: known artifacts vs anything new ----
    print("\n--- sanity_flags ---")
    fl_rows = []
    for k in CELLS:
        flags, known, other = classify_flags(metas[k])
        if flags:
            fl_rows.append(dict(cell=metas[k]["run_id"],
                                dump_only=metas[k].get("dump_only"),
                                kl_applied_steps=mblock(metas[k]).get(
                                    "sib_runtime", {}).get("kl_applied_steps"),
                                known_artifact="; ".join(known) or None,
                                other="; ".join(other) or None))
    print(pd.DataFrame(fl_rows).to_string(index=False))

    # the KL flag must coincide exactly with dump_only among the SIB cells, or
    # the "--dump_only artifact" account does not hold on this box
    print("\n--- KL-flag / dump_only correspondence over the present SIB cells ---")
    kl_rows = []
    for k in [c for c in CELLS if c[2] == "sib"]:
        meta = metas[k]
        _, known, _ = classify_flags(meta)
        kl_rows.append(dict(cell=meta["run_id"], dump_only=meta.get("dump_only"),
                            kl_flag=any("KL term" in f for f in known),
                            kl_applied_steps=mblock(meta).get(
                                "sib_runtime", {}).get("kl_applied_steps"),
                            last_vib_warmup=mblock(meta).get(
                                "sib_runtime", {}).get("last_vib_warmup")))
    kl = pd.DataFrame(kl_rows)
    print(kl.to_string(index=False))
    if not (kl.kl_flag == kl.dump_only).all():
        fail("the KL flag does not coincide with dump_only across the SIB cells; "
             "the declared --dump_only artifact account does not hold here")
    if not (kl.loc[~kl.dump_only, "kl_applied_steps"] > 0).all():
        fail("a non-dump-only SIB cell reports kl_applied_steps == 0")
    if not (kl.loc[kl.dump_only, "kl_applied_steps"] == 0).all():
        fail("a dump-only SIB cell reports kl_applied_steps != 0")
    print("flag <-> dump_only correspondence is exact; positive controls "
          f"{sorted(kl.loc[~kl.dump_only, 'kl_applied_steps'].tolist())}. "
          "Treated as a known artifact; no cell excluded.")

    print("\n--- provenance ---")
    print(pd.DataFrame([
        dict(cell=metas[k]["run_id"],
             config_sha256=str(metas[k].get("config_sha256"))[:12],
             git_hash=str(metas[k].get("git_hash"))[:10],
             metric_module_sha=str(metas[k].get("metric_module_sha"))[:10],
             stopped_on=metas[k].get("stopped_on"))
        for k in CELLS]).to_string(index=False))

    idx = build_index(metas, data, paths)
    print(f"\n--- INDEX.parquet would cover {len(idx)} rows, {idx.shape[1]} cols ---")
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
# bootstrap machinery -- identical to analysis/02_seo_loco.py
# ----------------------------------------------------------------------------

def _rng(stream):
    return np.random.default_rng([SEED, stream])


def aoi_cluster_bootstrap_imagemean(delta, aoi_codes, n_aoi, stream):
    """Cluster bootstrap: resample AOIs with replacement, take every image of a
    sampled AOI, statistic = image-level mean of the deltas. The point estimate
    of that statistic is the plain image-level mean."""
    sums = np.bincount(aoi_codes, weights=delta, minlength=n_aoi)
    cnts = np.bincount(aoi_codes, minlength=n_aoi).astype(np.float64)
    idx = _rng(stream).integers(0, n_aoi, size=(B, n_aoi))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def aoi_cluster_bootstrap_ratio(num, den, aoi_codes, n_aoi, stream):
    """Image-cluster bootstrap for a per-image ratio (App G.9): resample AOIs,
    form the ratio of the two resampled image-level means, never the mean of
    per-image ratios."""
    sn = np.bincount(aoi_codes, weights=num, minlength=n_aoi)
    sd = np.bincount(aoi_codes, weights=den, minlength=n_aoi)
    idx = _rng(stream).integers(0, n_aoi, size=(B, n_aoi))
    a, b = sn[idx].sum(1), sd[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / b, np.nan)


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
    lo, hi = np.nanpercentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


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


# ----------------------------------------------------------------------------
# full pass
# ----------------------------------------------------------------------------

def main():
    print("=" * 118)
    print("ARM A / S-EO method cells (SIB, FDA) -- FULL PASS")
    print("=" * 118)

    preflight()
    man, man_by_id = load_manifest()
    print(f"manifest: {len(man_by_id)} phase-3 cells, submitted {man['submitted_utc']}, "
          f"git {man['git_hash'][:10]}")

    data, metas, paths, raw = {}, {}, {}, {}
    for key in CELLS:
        rid, df, meta, pq = load_cell(*key)
        raw[key] = df                      # validated as written, before derived cols
        dfm = add_metrics(df)
        assert float(np.abs(dfm.miou2_pct - dfm.miou_tol_pct).max()) < 1e-9, rid
        assert float(np.abs(dfm.fg_pct - dfm.iou_tol * 100.0).max()) < 1e-9, rid
        data[key] = dfm.sort_values("image_id").reset_index(drop=True)
        metas[key], paths[key] = meta, pq

    # every per-cell assertion also runs in the full pass; the dry run is not
    # trusted to have been run against this copy of the dumps. Checked on the raw
    # frame: add_metrics appends miou2_pct and fg_pct, which the 76-column
    # schema assertion must not see.
    n_fail = 0
    for key in CELLS:
        checks, _ = check_cell(*key, raw[key], metas[key], man_by_id)
        for name, ok, val in checks:
            if not ok:
                print(f"  !! {run_id(*key)}: {name} -> {val}")
                n_fail += 1
    if n_fail:
        fail(f"{n_fail} per-cell assertions failed in the full pass")

    kl = pd.DataFrame([
        dict(cell=metas[k]["run_id"], dump_only=metas[k].get("dump_only"),
             kl_flag=any("KL term" in f for f in classify_flags(metas[k])[1]),
             kl_applied_steps=mblock(metas[k]).get("sib_runtime", {}).get("kl_applied_steps"))
        for k in CELLS if k[2] == "sib"])
    if not (kl.kl_flag == kl.dump_only).all():
        fail("the KL flag does not coincide with dump_only across the SIB cells")
    ACCESS = {(a, m): man_by_id[run_id(a, FOLDS[0], m)]["target_access"]
              for a in ARCHS for m in METHODS}

    idx = build_index(metas, data, paths)
    idx.to_parquet(INDEX_PATH, index=False)
    index_sha256 = sha256_file(INDEX_PATH)
    print(f"\nwrote {INDEX_PATH}  ({len(idx)} rows)  sha256={index_sha256[:16]}...")

    out = {}
    fmt = lambda v: f"{v:.4f}"

    aoi_cache = {}
    def aoi_codes_for(fold):
        if fold not in aoi_cache:
            codes, uniq = pd.factorize(data[(ARCHS[0], fold, "loco")].aoi_id.values)
            aoi_cache[fold] = (codes, len(uniq))
        return aoi_cache[fold]

    # ------------------------------------------------------------------
    # TASK 1 -- method vs vanilla-LOCO, per cell
    # ------------------------------------------------------------------
    print("\n" + "=" * 118)
    print(f"TASK 1 -- per-image delta (method - vanilla_loco), AOI bootstrap, B={B}")
    print("=" * 118)

    t1_rows, delta_store = [], {}
    for stream, (arch, fold, method) in enumerate(METHOD_CELLS):
        van, met, up = (data[(arch, fold, "loco")], data[(arch, fold, method)],
                        data[(arch, fold, "upper")])
        assert (van.image_id.values == met.image_id.values).all()
        codes, n_aoi = aoi_codes_for(fold)

        for mi, (metric, mcol, role) in enumerate(
                (("2class", "miou2_pct", "primary"), ("fgonly", "fg_pct", "secondary"))):
            s = 10 * stream + mi
            delta = (met[mcol].values - van[mcol].values).astype(np.float64)
            obs, p_raw = aoi_signflip_imagemean(delta, codes, n_aoi, 1000 + s)
            boot = aoi_cluster_bootstrap_imagemean(delta, codes, n_aoi, 2000 + s)
            lo, hi = ci95(boot)
            t1_rows.append(dict(
                method=method, arch=arch, fold=fold, metric=metric, role=role,
                target_access=ACCESS[(arch, method)],
                vanilla_loco=float(van[mcol].mean()), method_loco=float(met[mcol].mean()),
                upper=float(up[mcol].mean()),
                delta=float(obs), stderr=float(boot.std(ddof=1)),
                ci_lo=lo, ci_hi=hi, p_raw=p_raw, n_aoi=n_aoi, n_images=len(delta)))
            if metric == "2class":
                delta_store[(arch, fold, method)] = delta

    t1 = pd.DataFrame(t1_rows)
    # Holm WITHIN each method family, sized as run: SIB-5, FDA-6
    fam_sizes = {}
    for method in METHODS:
        for metric in ("2class", "fgonly"):
            m = (t1.method == method) & (t1.metric == metric)
            t1.loc[m, "p_holm"] = holm(t1.loc[m, "p_raw"].values)
            t1.loc[m, "holm_family_size"] = int(m.sum())
            fam_sizes[method] = int(((t1.method == method) & (t1.metric == "2class")).sum())

    for method in METHODS:
        for metric in ("2class", "fgonly"):
            print(f"\n--- {method.upper()} vs vanilla-LOCO, {metric} "
                  f"({'primary' if metric == '2class' else 'secondary'}), "
                  f"Holm over this family's {fam_sizes[method]} cells as run ---")
            print(t1[(t1.method == method) & (t1.metric == metric)][
                ["arch", "fold", "target_access", "vanilla_loco", "method_loco",
                 "upper", "delta", "stderr", "ci_lo", "ci_hi", "p_raw", "p_holm",
                 "n_aoi", "n_images"]].to_string(index=False, float_format=fmt))
    for k in sorted(KNOWN_MISSING):
        print(f"\n  ABSENT: {run_id(*k)} -- {KNOWN_MISSING_REASON}")
    out["task1"] = t1.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 2 -- the pre-committed worst-case criterion
    # ------------------------------------------------------------------
    print("\n" + "=" * 118)
    print("TASK 2 -- worst-case criterion. Worst cell = lowest absolute two-class "
          "K=2 level among that condition's cells.")
    print("Criterion: the method's worst cell is no worse than vanilla-LOCO's worst cell.")
    print("=" * 118)

    lev = t1[t1.metric == "2class"].copy()
    van_lev = (lev.drop_duplicates(["arch", "fold"])[["arch", "fold", "vanilla_loco"]]
               .rename(columns={"vanilla_loco": "level"}))

    def worst_of(frame, col):
        r = frame.loc[frame[col].idxmin()]
        return r["arch"], r["fold"], float(r[col])

    t2_rows = []
    for mi, method in enumerate(METHODS):
        sub = lev[lev.method == method]
        scopes = [(a, sub[sub.arch == a], van_lev[van_lev.arch == a]) for a in ARCHS]
        scopes.append(("overall", sub, van_lev))
        for si, (scope, ms, vs) in enumerate(scopes):
            covered = sorted(set(zip(ms.arch, ms.fold)))
            missing = sorted((a, f) for (a, f, m) in KNOWN_MISSING
                             if m == method and (scope == "overall" or a == scope))
            m_arch, m_fold, m_val = worst_of(ms, "method_loco")
            v_arch, v_fold, v_val = worst_of(vs, "level")
            # fold-matched vanilla worst: restricted to the cells the method covers,
            # so the comparison is not confounded by the absent cell
            vm = vs[[(a, f) in covered for a, f in zip(vs.arch, vs.fold)]]
            vm_arch, vm_fold, vm_val = worst_of(vm, "level")
            same = (m_arch, m_fold) == (v_arch, v_fold)
            codes, n_aoi = aoi_codes_for(m_fold)
            boot = aoi_cluster_bootstrap_imagemean(
                delta_store[(m_arch, m_fold, method)], codes, n_aoi,
                7000 + 100 * mi + si)
            lo, hi = ci95(boot)
            t2_rows.append(dict(
                method=method, scope=scope,
                role=("claim" if method == "sib" else "context"),
                n_cells=len(covered),
                folds_absent=";".join(f"{a}/{f.upper()}" for a, f in missing) or None,
                provisional=bool(missing),
                method_worst_cell=f"{m_arch}/{m_fold.upper()}", method_worst=m_val,
                method_worst_access=ACCESS[(m_arch, method)],
                vanilla_worst_cell=f"{v_arch}/{v_fold.upper()}", vanilla_worst=v_val,
                difference=m_val - v_val, criterion_met=bool(m_val >= v_val),
                vanilla_worst_cell_foldmatched=f"{vm_arch}/{vm_fold.upper()}",
                vanilla_worst_foldmatched=vm_val,
                difference_foldmatched=m_val - vm_val,
                criterion_met_foldmatched=bool(m_val >= vm_val),
                worst_cells_same=same,
                paired_delta_fold=f"{m_arch}/{m_fold.upper()}",
                paired_delta=float(delta_store[(m_arch, m_fold, method)].mean()),
                paired_ci_lo=lo, paired_ci_hi=hi,
                paired_ci_is_on_the_difference=same))
    t2 = pd.DataFrame(t2_rows)
    print(t2[["method", "scope", "role", "n_cells", "folds_absent", "provisional",
              "method_worst_cell", "method_worst", "method_worst_access",
              "vanilla_worst_cell", "vanilla_worst", "difference", "criterion_met"]
             ].to_string(index=False, float_format=fmt))
    print("\n--- fold-matched variant: vanilla's worst restricted to the folds the "
          "method actually covers ---")
    print(t2[["method", "scope", "n_cells", "method_worst_cell", "method_worst",
              "vanilla_worst_cell_foldmatched", "vanilla_worst_foldmatched",
              "difference_foldmatched", "criterion_met_foldmatched"]
             ].to_string(index=False, float_format=fmt))
    print("\n--- CI on the paired delta of the fold carrying the method's worst cell ---")
    print(t2[["method", "scope", "paired_delta_fold", "paired_delta", "paired_ci_lo",
              "paired_ci_hi", "worst_cells_same", "paired_ci_is_on_the_difference"]
             ].to_string(index=False, float_format=fmt))
    print("\nnote: 'overall' crosses the target_access boundary "
          "(dinov3-sib source_only vs mamnet-sib unlabeled_target) and is reported "
          "as an identity pair only; no statistic is pooled across it.")
    out["task2"] = t2.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 3 -- fraction of the vanilla gap closed
    # ------------------------------------------------------------------
    print("\n" + "=" * 118)
    print("TASK 3 -- gap closure. vanilla gap = upper - vanilla_loco; residual gap = "
          "upper - method_loco; fraction closed = (method_loco - vanilla_loco) / vanilla gap.")
    print("=" * 118)

    t3_rows = []
    for stream, (arch, fold, method) in enumerate(METHOD_CELLS):
        van, met, up = (data[(arch, fold, "loco")], data[(arch, fold, method)],
                        data[(arch, fold, "upper")])
        codes, n_aoi = aoi_codes_for(fold)
        num = (met.miou2_pct.values - van.miou2_pct.values).astype(np.float64)
        den = (up.miou2_pct.values - van.miou2_pct.values).astype(np.float64)
        resid = (up.miou2_pct.values - met.miou2_pct.values).astype(np.float64)

        b_frac = aoi_cluster_bootstrap_ratio(num, den, codes, n_aoi, 8000 + stream)
        f_lo, f_hi = ci95(b_frac)
        b_res = aoi_cluster_bootstrap_imagemean(resid, codes, n_aoi, 8500 + stream)
        r_lo, r_hi = ci95(b_res)
        t3_rows.append(dict(
            method=method, arch=arch, fold=fold,
            target_access=ACCESS[(arch, method)],
            vanilla_gap=float(den.mean()), residual_gap=float(resid.mean()),
            residual_ci_lo=r_lo, residual_ci_hi=r_hi,
            frac_closed=float(num.mean() / den.mean()),
            frac_ci_lo=f_lo, frac_ci_hi=f_hi,
            n_aoi=n_aoi, n_images=len(num)))
    t3 = pd.DataFrame(t3_rows)
    for method in METHODS:
        print(f"\n--- {method.upper()} ---")
        print(t3[t3.method == method].to_string(index=False, float_format=fmt))
    print("\nfraction closed is a ratio of two image-level means under one AOI "
          "resample (App G.9), never a mean of per-image ratios. No pooling across "
          "target_access classes.")
    out["task3"] = t3.to_dict("records")

    # ------------------------------------------------------------------
    # TASK 4 -- descriptives
    # ------------------------------------------------------------------
    print("\n" + "=" * 118)
    print(f"TASK 4 -- per-cell descriptives, {len(METHOD_CELLS)} method cells as run")
    print("=" * 118)

    d_rows = []
    for arch, fold, method in METHOD_CELLS:
        df, meta = data[(arch, fold, method)], metas[(arch, fold, method)]
        mb, entry = mblock(meta), man_by_id[run_id(arch, fold, method)]
        nz = df[df.n_valid_px > 0]
        flags, known, other = classify_flags(meta)
        d_rows.append(dict(
            cell=meta["run_id"], method=method, arch=arch, fold=fold,
            target_access=meta.get("target_access"),
            n_images=len(df), n_aoi=df.aoi_id.nunique(),
            mean_2class=float(df.miou2_pct.mean()),
            median_2class=float(df.miou2_pct.median()),
            mean_fgonly=float(df.fg_pct.mean()),
            median_fgonly=float(df.fg_pct.median()),
            predpos_frac_pooled=float(df.n_pred_pos.sum() / df.n_valid_px.sum()),
            predpos_frac_mean_img=float((nz.n_pred_pos / nz.n_valid_px).mean()),
            predpos_frac_sidecar=meta.get("pred_pos_frac_on_valid"),
            epochs_run=meta.get("epochs_run"), best_epoch=meta.get("best_epoch"),
            epochs_max=meta.get("epochs_max"), stopped_on=meta.get("stopped_on"),
            early_stop_patience=meta.get("early_stop_patience"),
            dump_only=meta.get("dump_only"),
            kl_applied_steps=mb.get("sib_runtime", {}).get("kl_applied_steps"),
            fda_target_pool_split=mb.get("fda_target_pool_split"),
            fda_L=mb.get("fda_L"),
            components_active=";".join(entry["components_active"]),
            components_off=";".join(entry["components_off"]) or None,
            sanity_flags="; ".join(flags) or None,
            sanity_flags_known_artifact="; ".join(known) or None,
            sanity_flags_other="; ".join(other) or None))
    t4 = pd.DataFrame(d_rows)
    print(t4[["cell", "target_access", "mean_2class", "median_2class", "mean_fgonly",
              "median_fgonly", "predpos_frac_pooled", "predpos_frac_mean_img",
              "predpos_frac_sidecar", "epochs_run", "best_epoch", "stopped_on",
              "dump_only", "kl_applied_steps"]].to_string(index=False, float_format=fmt))
    print("\n--- components and flags ---")
    for r in d_rows:
        print(f"  {r['cell']}")
        print(f"      active: {r['components_active']}")
        print(f"      flags:  {r['sanity_flags'] or 'none'}")
    ceil_cells = t4[t4.stopped_on == "epoch_ceiling"].cell.tolist()
    print(f"\nstopped on epoch_ceiling rather than patience: "
          f"{', '.join(ceil_cells) if ceil_cells else 'none'} -- reported, not excluded")
    out["task4"] = t4.to_dict("records")

    # ------------------------------------------------------------------
    # results JSON
    # ------------------------------------------------------------------
    conditions, tests = [], []
    for r in t1.to_dict("records"):
        conditions.append({
            "label": f"S-EO {r['method'].upper()} - vanilla-LOCO / {r['arch']} / "
                     f"holdout {r['fold'].upper()}"
                     + ("" if r["metric"] == "2class" else " (secondary, fg-only)"),
            "metric": ("delta_2class_tol_k2_unseen_excluded" if r["metric"] == "2class"
                       else "delta_fg_only_tol_k2_unseen_excluded"),
            "target_access": r["target_access"],
            "mean": round(r["delta"], 4), "stderr": round(r["stderr"], 4),
            "n": int(r["n_images"]), "n_aoi": int(r["n_aoi"]),
            "ci95": [round(r["ci_lo"], 4), round(r["ci_hi"], 4)],
            "ci_method": "aoi_cluster_bootstrap_B10000",
            "vanilla_loco_mean": round(r["vanilla_loco"], 4),
            "method_loco_mean": round(r["method_loco"], 4),
            "upper_mean": round(r["upper"], 4),
            "baseline": run_id(r["arch"], r["fold"], "loco")})
        tests.append({
            "name": "AOI-level sign-flip permutation test",
            "compares": [run_id(r["arch"], r["fold"], r["method"]),
                         run_id(r["arch"], r["fold"], "loco")],
            "metric": r["metric"],
            "statistic": round(r["delta"], 4), "p": r["p_raw"],
            "p_holm": round(float(r["p_holm"]), 6), "B": B, "two_tailed": True,
            "family": f"{int(r['holm_family_size'])} cells of the "
                      f"{r['method'].upper()} family as run, Holm-Bonferroni"})

    sib_ov = t2[(t2.method == "sib") & (t2.scope == "overall")].iloc[0]

    res = {
        "id": "03_seo_methods",
        "arm": "A",
        "answers": ["v1UJ-W2"],
        "question": "On S-EO, how do SIB and FDA compare against vanilla-LOCO in each "
                    "(holdout city x architecture) cell, and does SIB's worst cell "
                    "sit no lower than vanilla-LOCO's worst cell?",
        "runs": [metas[k]["run_id"] for k in CELLS],
        "runs_absent": [{"run_id": run_id(*k), "reason": KNOWN_MISSING_REASON}
                        for k in sorted(KNOWN_MISSING)],
        "index_sha256": index_sha256,
        "manifest": "logs/armA_phase3_manifest.json",
        "manifest_git_hash": man["git_hash"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/03_seo_methods.py",
        "git_hash": None,
        "git_hash_note": "the project base is not a git repository on this box. "
                         "Per-cell producing commits are in INDEX.parquet.git_hash.",
        "seed": SEED,
        "decision_rule": {
            "fixed_before_numbers_seen": True,
            "criterion": "SIB's worst cell across its (fold x arch) cells, measured "
                         "as the lowest absolute two-class K=2 level, is no worse than "
                         "vanilla-LOCO's worst cell.",
            "mean_deltas": "reported, not the criterion",
            "fda": "context (strongest published adaptation method), not a claim",
            "access_boundary": "dinov3-sib is source_only, mamnet-sib is "
                               "unlabeled_target, both FDA arms are unlabeled_target, "
                               "per logs/armA_phase3_manifest.json. Nothing is pooled "
                               "across that boundary. Every row carries target_access.",
            "holm_families_as_run": {"sib": fam_sizes["sib"], "fda": fam_sizes["fda"]}},
        "metric_definitions": {
            "delta_2class_tol_k2_unseen_excluded":
                "mean over test patches of (method_loco - vanilla_loco) two-class "
                "boundary-tolerant mIoU at K=2 with DSM-unseen pixels excluded from "
                "all four confusion cells; recomputed per patch from "
                "tp_tol/fp_tol/fn_tol/tn_tol as 50*(tp/(tp+fp+fn+1e-10) + "
                "tn/(tn+fp+fn+1e-10)), matching metrics.py _img_metrics.",
            "delta_fg_only_tol_k2_unseen_excluded":
                "same, foreground class only: 100*tp/(tp+fp+fn+1e-10). Secondary.",
            "frac_closed":
                "(method_loco - vanilla_loco) / (upper - vanilla_loco), formed as a "
                "ratio of two image-level means under one AOI resample (App G.9)."},
        "conditions": conditions,
        "tests": tests,
        "task2_worst_case": t2.to_dict("records"),
        "task3_gap_closure": t3.to_dict("records"),
        "task4_descriptives": t4.to_dict("records"),
        "headline": (
            f"On S-EO, over the {int(sib_ov.n_cells)} SIB cells that have run, SIB's "
            f"worst cell is {sib_ov.method_worst:.4f} two-class mIoU at K=2 "
            f"({sib_ov.method_worst_cell}, {sib_ov.method_worst_access}) against "
            f"vanilla-LOCO's worst cell {sib_ov.vanilla_worst:.4f} "
            f"({sib_ov.vanilla_worst_cell}), a difference of {sib_ov.difference:+.4f}."),
        "caveats": [
            (f"INCOMPLETE GRID. "
             f"{', '.join(run_id(*k) for k in sorted(KNOWN_MISSING))} absent. Every "
             "worst-case number for an affected family is an upper bound on its true "
             "worst case: an absent cell can only lower it, never raise it. Rows "
             "carrying this are marked provisional in task2_worst_case."
             if KNOWN_MISSING else
             "Complete grid: all 24 cells present, both method families at 6 cells. "
             "No worst-case number is provisional."),
            (f"Holm families are sized as run: SIB {fam_sizes['sib']} cells, FDA "
             f"{fam_sizes['fda']}."
             + ("" if not KNOWN_MISSING else
                " These adjustments will change when the absent cells land.")),
            "Single seed. Every cell is seed 0; the CIs cover test-patch and AOI "
            "sampling only, not optimisation noise.",
            "The worst-case criterion compares absolute levels of two cells that need "
            "not be the same cell. Where they are not, the difference spans two folds "
            "and is not a paired quantity; task2_worst_case carries worst_cells_same "
            "and paired_ci_is_on_the_difference to mark this, and a fold-matched "
            "variant restricting vanilla's worst to the folds the method covers.",
            "dinov3-sib is source_only and mamnet-sib is unlabeled_target. The "
            "'overall' scope in task2_worst_case crosses that boundary and is an "
            "identity pair only; no statistic is pooled across it.",
            "FDA is reported as context, not as a claim under test.",
            "Two cells stopped on epoch_ceiling rather than patience "
            "(dinov3-fda-loco-r3500-oma-hr-s0, dinov3-sib-loco-r3500-oma-hr-s0): early "
            "stopping never fired within the 100-epoch cap, so those two are not "
            "converged under the same rule as the other cells. Reported, not excluded.",
            "The 'SIB KL term was never added to the loss' sanity flag on "
            "dinov3-sib-oma, mamnet-sib-oma and mamnet-sib-ucsd is a known --dump_only "
            "artifact: sib_runtime.kl_applied_steps is process-local and reads 0 in a "
            "dump-only re-run regardless of training history. On this box the flag "
            "coincides exactly with dump_only across all 6 SIB cells, with positive "
            "controls kl_applied_steps 18,791 (mamnet/JAX), 22,724 (dinov3/JAX) and "
            "25,346 (dinov3/UCSD) on the three non-dump-only cells. That "
            "correspondence is structural evidence, not affirmative proof that KL was "
            "applied during those three cells' original training; that proof is in the "
            "Delta training logs, which are not on this box. No cell is excluded on "
            "this flag.",
            "App H's deployment-time coverage threshold is NOT applied in any SIB cell "
            "(manifest sib_coverage_threshold); probs.npz is unthresholded. These "
            "numbers describe SIB without that component.",
            "108 of the 6149 test patches per architecture (29 OMA, 79 UCSD, 0 JAX) are "
            "entirely DSM-unseen: the unseen-exclusion rule empties all four confusion "
            "cells and the per-patch metric takes its floor value of 0.0. They are the "
            "same patches in every cell of a fold, so each contributes a delta of "
            "exactly 0 and dilutes every delta toward zero.",
            "The degenerate-patch convention is inherited: an all-shadow patch scores "
            "background IoU 0 and an empty-foreground patch scores foreground IoU 0, so "
            "both halve the two-class mIoU.",
            "rung 3500 is the S-EO patch budget, not a point on the ShadowTransfer "
            "225/450/900 ladder; the unit is a 384 px patch, not a benchmark image.",
            "High-resolution cells only, one resolution, one seed, three cities, two "
            "architectures. OGLANet is not in arm A.",
            "S-EO labels are auto-generated. Both sides of every delta use the same "
            "labels on the same patches, so label noise largely cancels in the "
            "difference; the absolute levels carry it.",
            "No comparison to internal (ShadowTransfer) SIB numbers is made here.",
            "Nothing AURC-shaped is computed. The definition is unratified.",
            "The pre-committed branch reading is not adjudicated on this box.",
        ],
        "figure": None,
        "caption": None,
    }
    print("\nHEADLINE: " + res["headline"])

    with open(RESULTS_PATH, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {RESULTS_PATH}")
    with open(TABLES_PATH, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"wrote {TABLES_PATH}")


if __name__ == "__main__":
    if DRY_RUN:
        dry_run()
    else:
        main()
