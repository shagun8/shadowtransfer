#!/usr/bin/env python
"""
Arm A -- task 5b: Figure 2's p-bar_S over the EXTENDED vanilla S-EO grid.

Appends one section, `task5b_pbar_S`, to results/08_seo_full_grid.json. Every
key already in 08 is left exactly as written (asserted before the move).

The statistic and the paired analysis are LIFTED, not rewritten:

  ESTIMATOR   shadowtransfer/metrics.py `pbar_populations` (sha256 730ad4a5...),
              called upstream by seo_dump.py; this file reads the stored columns
              pbar_gt_shadow / pbar_gt_bg, exactly as results/04 did. The module
              sha is asserted equal to the sha 04 recorded and to
              metric_module_sha on all 72 sidecars.
  04 PAIRING  analysis/04_seo_probspace.py `paired_pair` + `holm` +
              `classify_pairs`, imported from the file 04 ran (its mtime
              precedes results/04 by 15 s). Used verbatim, with 04's own rng
              streams, for the s0 MAMNet/DINOv3 cross-check.
  SEED STRATUM analysis/08_seo_full_grid.py `aoi_cluster_bootstrap_imagemean`,
              `aoi_signflip_imagemean`, `stream_of`, `ci95`, `holm` -- task 1's
              functions, imported. Per seed, per-image delta = LOCO - upper;
              the per-image seed-averaged delta is the input to the AOI cluster
              bootstrap and the AOI sign-flip, which is task 1's scheme.

Population (04's, stated): per patch, the mean unthresholded P(shadow) over
GT-foreground pixels on the valid region (+-2 px don't-care band removed,
DSM-unseen pixels removed). Patches whose population is empty (NaN, including
the fully DSM-unseen patches) are dropped pairwise; the dropped set is asserted
identical across every cell of a fold, so the pairing is exact. AOIs with no
defined patch leave the bootstrap, as in 04. p-bar_B is the same over GT
background, with its own drop set.

Nothing AURC-shaped is read. No external magnitude is compared to an internal
one. No branch letter is applied: the pre-commitment text for the 12-pair
extended grid is not on this box.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/08b_seo_pbar.py
"""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

BASE = os.environ["PROJECT_ROOT"]
TREE = os.path.join(BASE, "delta_dumps")
OLD_DUMPS = os.path.join(BASE, "dumps", "seo")          # what 04 read
INDEX_PATH = os.path.join(TREE, "INDEX.parquet")
RESULTS_PATH = os.path.join(BASE, "results", "08_seo_full_grid.json")
RESULTS_04 = os.path.join(BASE, "results", "04_seo_probspace.json")
SCRIPT_04 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_seo_probspace.py")
SCRIPT_08 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_seo_full_grid.py")
BACKUP_DIR = os.path.join(BASE, "logs")
SECTION = "task5b_pbar_S"

# the state of 08 this section is appended to; a different file means someone
# else wrote 08 since, and the append stops
EXPECTED_08_SHA = "d9648c57e3dcd395380b466cc74247ff304902d0f62cd4a26c62adf60bcdefe9"
EXPECTED_INDEX_SHA = "a14903907349e93b4392a16aa468ffb41a459b93ccbc82edd4af6a3650b1ae9d"

ARCHS = ["mamnet", "dinov3", "oglanet", "crossearth"]
FOLDS = ["oma", "jax", "ucsd"]
SEEDS = [0, 1, 2]
N_TEST = {"oma": 1637, "jax": 2642, "ucsd": 1870}
B = 10_000
SEED = 0

STAT_FG = "pbar_gt_shadow"      # Fig 2's p-bar_S
STAT_BG = "pbar_gt_bg"          # p-bar_B, the published control
USED_COLS = {"run_id": "object", "image_id": "object", "aoi_id": "object",
             "city": "object", STAT_FG: "float64", STAT_BG: "float64",
             "n_valid_px": "int64", "n_gt_fg_valid": "int64"}
FORBIDDEN_PREFIX = "aurc"
EXPECTED_ACCESS = {"loco": "source_only", "upper": "target_labels"}


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
    spec.loader.exec_module(mod)
    return mod


M4 = load_module("seo_probspace_04", SCRIPT_04)
M8 = load_module("seo_full_grid_08", SCRIPT_08)
if (M4.B, M4.SEED, M8.B, M8.SEED) != (B, SEED, B, SEED):
    fail(f"B/SEED differ from 04/08: {(M4.B, M4.SEED, M8.B, M8.SEED)}")


# ----------------------------------------------------------------------------
# selection and validation
# ----------------------------------------------------------------------------

def select_cells(idx):
    """The 72 vanilla cells, from INDEX.parquet. Never a glob."""
    sel = {}
    for arch in ARCHS:
        for fold in FOLDS:
            for seed in SEEDS:
                for mode, method in (("loco", "vanilla"), ("upper", "none")):
                    hit = idx[(idx.category == "cell") & (idx.dataset == "seo") & (idx.arch == arch)
                              & (idx.method == method) & (idx.condition == mode)
                              & (idx.holdout == fold) & (idx.seed == seed)]
                    if len(hit) != 1:
                        fail(f"{arch}/{method}/{mode}/{fold}/s{seed}: {len(hit)} INDEX rows")
                    sel[(arch, fold, seed, mode)] = hit.iloc[0]
    return sel


def load_and_validate(key, rec, metrics_sha):
    arch, fold, seed, mode = key
    d = os.path.join(TREE, rec["dir"])
    pq_path, sj = os.path.join(d, "perimage.parquet"), os.path.join(d, "run.json")
    if not os.path.isfile(sj):
        fail(f"{rec['run_id']}: run.json missing next to perimage.parquet")
    if not os.path.isfile(os.path.join(d, ".DONE")):
        fail(f"{rec['run_id']}: no .DONE")
    with open(sj) as fh:
        meta = json.load(fh)
    df = pd.read_parquet(pq_path)
    checks = []
    for c, want in USED_COLS.items():
        got = str(df[c].dtype) if c in df.columns else "ABSENT"
        checks.append((f"dtype {c}=={want}", got == want, got))
    if any(not ok for _, ok, _ in checks):
        return None, meta, checks, pq_path
    checks += [
        (f"nrow == n_test[{fold}]", len(df) == N_TEST[fold], len(df)),
        ("run_id == dir == col", meta["run_id"] == rec["run_id_dir"] == df.run_id.iloc[0]
         and df.run_id.nunique() == 1, meta["run_id"]),
        ("dataset == seo", meta.get("dataset") == "seo", meta.get("dataset")),
        (f"target_access == {EXPECTED_ACCESS[mode]}",
         meta.get("target_access") == EXPECTED_ACCESS[mode], meta.get("target_access")),
        ("metric_module_sha == imported module", meta.get("metric_module_sha") == metrics_sha,
         str(meta.get("metric_module_sha"))[:10]),
        ("city == holdout", set(df.city.str.lower()) == {fold}, sorted(set(df.city))),
        ("image_id unique", not df.image_id.duplicated().any(), "ok"),
        ("aoi_id non-null", int(df.aoi_id.isna().sum()) == 0, int(df.aoi_id.isna().sum())),
    ]
    # 04's validity checks on the stored statistic (input validation, not the
    # probs.npz semantics check, which needs probs.npz)
    fg, bg = df[STAT_FG], df[STAT_BG]
    checks += [
        ("pbar_gt_shadow in [0,1] where defined", int(((fg < 0) | (fg > 1)).sum()) == 0, "ok"),
        ("pbar_gt_bg in [0,1] where defined", int(((bg < 0) | (bg > 1)).sum()) == 0, "ok"),
        ("pbar_gt_shadow NaN <=> n_gt_fg_valid == 0",
         bool((fg.isna() == (df.n_gt_fg_valid == 0)).all()),
         f"{int(fg.isna().sum())} NaN / {int((df.n_gt_fg_valid == 0).sum())} empty"),
        ("pbar_gt_bg NaN <=> GT-bg population empty",
         bool((bg.isna() == (df.n_gt_fg_valid == df.n_valid_px)).all()),
         f"{int(bg.isna().sum())} NaN"),
    ]
    df = df.sort_values("image_id").reset_index(drop=True)
    return df, meta, checks, pq_path


# ----------------------------------------------------------------------------
# the seed-stratified paired statistic (task 1's functions, on 04's population)
# ----------------------------------------------------------------------------

def paired_seed_stratified(lo, up, col, keep, aoi, label):
    """lo/up: {seed: frame aligned to the fold reference}. keep: the fold's
    defined-statistic mask (asserted identical in every cell). The input to the
    bootstrap is the per-image seed-averaged delta, as in task 1."""
    deltas = {s: (lo[s][col].to_numpy(np.float64) - up[s][col].to_numpy(np.float64))[keep]
              for s in SEEDS}
    if any(np.isnan(v).any() for v in deltas.values()):
        fail(f"{label}: NaN survives the keep mask")
    dbar = np.mean([deltas[s] for s in SEEDS], axis=0)
    a = aoi[keep]
    codes, uniq = pd.factorize(a)
    n_aoi = len(uniq)
    boot = M8.aoi_cluster_bootstrap_imagemean(dbar, codes, n_aoi, M8.stream_of("boot|" + label))
    obs, p = M8.aoi_signflip_imagemean(dbar, codes, n_aoi, M8.stream_of("flip|" + label))
    lo_i, hi_i = M8.ci95(boot)
    boot_a = M4.aoi_cluster_bootstrap_aoimean(dbar, codes, n_aoi, M8.stream_of("aoiw|" + label))
    lo_a, hi_a = M8.ci95(boot_a)
    per_aoi = pd.DataFrame({"a": a, "d": dbar}).groupby("a").d.mean().to_numpy()
    per_seed = {int(s): float(deltas[s].mean()) for s in SEEDS}
    lo_lvl = {int(s): float(lo[s][col].to_numpy(np.float64)[keep].mean()) for s in SEEDS}
    up_lvl = {int(s): float(up[s][col].to_numpy(np.float64)[keep].mean()) for s in SEEDS}
    ps = np.array(list(per_seed.values()))
    return dict(
        loco_mean=float(np.mean(list(lo_lvl.values()))), upper_mean=float(np.mean(list(up_lvl.values()))),
        delta=float(obs), stderr=float(boot.std(ddof=1)), ci_lo=lo_i, ci_hi=hi_i, p_raw=float(p),
        delta_aoi_equal_weight=float(per_aoi.mean()), ci_lo_aoi=lo_a, ci_hi_aoi=hi_a,
        per_seed_delta=per_seed, per_seed_loco=lo_lvl, per_seed_upper=up_lvl,
        seed_sd=float(ps.std(ddof=1)),
        n_seeds_same_sign_as_mean=int((np.sign(ps) == np.sign(obs)).sum()),
        n_images=int(keep.sum()), n_excluded=int((~keep).sum()), n_aoi=int(n_aoi),
        n_aoi_dropped=int(len(pd.unique(aoi)) - n_aoi),
        frac_aoi_negative=float((per_aoi < 0).mean()),
        ci_excludes_point=bool(not (lo_i <= obs <= hi_i)))


def r6(v):
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6)
    if isinstance(v, dict):
        return {k: r6(x) for k, x in v.items()}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    tag = "DRY RUN" if DRY_RUN else "FULL PASS"
    print("=" * 110)
    print(f"ARM A / TASK 5b -- p-bar_S over the extended vanilla S-EO grid -- {tag}")
    print("=" * 110)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    fmt = lambda v: f"{v:.4f}"

    # ---- provenance of the lifted pieces ----
    with open(RESULTS_04) as fh:
        r04 = json.load(fh)
    sha_04_recorded = r04["statistic_provenance"]["verbatim_lift"].split("sha256 ")[1].split(",")[0]
    sha_04_recorded = "".join(sha_04_recorded.split())
    metrics_sha = sha256_file(M8.METRICS_PATH)
    m4_metrics = os.path.join(M4.METRICS_ROOT, "shadowtransfer", "metrics.py")
    print(f"metrics.py (08 path)  sha256 {metrics_sha}")
    print(f"metrics.py (04 path)  sha256 {sha256_file(m4_metrics)}")
    print(f"sha recorded by 04           {sha_04_recorded}")
    if not (metrics_sha == sha256_file(m4_metrics) == sha_04_recorded):
        fail("metric module sha differs from the one results/04 recorded")
    import shadowtransfer.metrics as MM          # on sys.path via the 04 import
    if os.path.realpath(MM.__file__) != os.path.realpath(m4_metrics):
        fail(f"imported metrics module is {MM.__file__}, not 04's")
    if not callable(getattr(MM, "pbar_populations", None)):
        fail("pbar_populations not in the metric module")
    s4_mtime = os.path.getmtime(SCRIPT_04)
    r4_mtime = os.path.getmtime(RESULTS_04)
    print(f"04 script sha256 {sha256_file(SCRIPT_04)[:16]}..., mtime precedes results/04 by "
          f"{r4_mtime - s4_mtime:.1f} s")
    if not (0 < r4_mtime - s4_mtime < 600):
        fail("analysis/04_seo_probspace.py was modified after results/04 was written")
    sha_08_script = sha256_file(SCRIPT_08)

    # ---- index ----
    if not os.path.isfile(INDEX_PATH):
        fail("INDEX.parquet missing")
    index_sha = sha256_file(INDEX_PATH)
    if index_sha != EXPECTED_INDEX_SHA:
        fail(f"INDEX.parquet sha {index_sha[:12]} != the one 08 cites {EXPECTED_INDEX_SHA[:12]}")
    idx = pd.read_parquet(INDEX_PATH)
    dup = idx.dropna(subset=["run_id"]).groupby("run_id").config_sha256.nunique()
    if (dup > 1).any():
        fail(f"duplicate run_id with differing config_sha256: {list(dup[dup > 1].index)}")
    sel = select_cells(idx)
    print(f"\nINDEX.parquet {len(idx)} rows, sha256 {index_sha[:16]}... (== 08's); selected {len(sel)} cells")

    # ---- load + validate ----
    data, metas, paths, tab = {}, {}, {}, []
    resolved_schema = None
    for key, rec in sel.items():
        df, meta, checks, pqp = load_and_validate(key, rec, metrics_sha)
        bad = [c for c in checks if not c[1]]
        for name, ok, val in checks:
            if not ok:
                print(f"  !! {rec['run_id']}: {name} -> {val}")
        if df is not None:
            if resolved_schema is None:
                resolved_schema = {c: str(df[c].dtype) for c in USED_COLS}
            n_aurc = len([c for c in df.columns if c.startswith(FORBIDDEN_PREFIX)])
            data[key], metas[key], paths[key] = df, meta, pqp
        tab.append(dict(run_id=rec["run_id"], nrow=(len(df) if df is not None else None),
                        n_aoi=(df.aoi_id.nunique() if df is not None else None),
                        n_nan_S=(int(df[STAT_FG].isna().sum()) if df is not None else None),
                        n_nan_B=(int(df[STAT_BG].isna().sum()) if df is not None else None),
                        mean_S=(float(df[STAT_FG].mean()) if df is not None else None),
                        mean_B=(float(df[STAT_BG].mean()) if df is not None else None),
                        n_checks=len(checks), failed=len(bad)))
    tab = pd.DataFrame(tab)
    print("\n--- resolved schema (columns this file reads) ---")
    print(resolved_schema)
    print("aurc* columns present but never read")
    print("\n--- cells selected, row counts, per-cell checks ---")
    print(tab.to_string(index=False, float_format=fmt))
    if tab.failed.sum():
        fail(f"{int(tab.failed.sum())} per-cell assertions failed")

    # ---- shared test-image sets, aoi maps, and drop sets, per fold ----
    foldref = {}
    print("\n--- per-fold identity (asserted before any pairing) ---")
    for fold in FOLDS:
        keys = [k for k in data if k[1] == fold]
        ref = data[keys[0]]
        for k in keys:
            df = data[k]
            if not np.array_equal(df.image_id.to_numpy(), ref.image_id.to_numpy()):
                fail(f"{k}: test-image set/order differs within fold {fold}")
            if not np.array_equal(df.aoi_id.to_numpy(), ref.aoi_id.to_numpy()):
                fail(f"{k}: image_id -> aoi_id map differs within fold {fold}")
            for col in (STAT_FG, STAT_BG):
                if not np.array_equal(df[col].isna().to_numpy(), ref[col].isna().to_numpy()):
                    fail(f"{k}: {col} undefined-patch set differs within fold {fold}")
        keepS = ~ref[STAT_FG].isna().to_numpy()
        keepB = ~ref[STAT_BG].isna().to_numpy()
        n_unseen = int((ref.n_valid_px <= 0).sum())
        foldref[fold] = dict(aoi=ref.aoi_id.to_numpy(), keepS=keepS, keepB=keepB)
        print(f"  {fold:5s}: {len(keys)} cells share {len(ref)} image_ids / {ref.aoi_id.nunique()} AOIs; "
              f"p-bar_S defined on {int(keepS.sum())} ({int((~keepS).sum())} dropped, of which "
              f"{n_unseen} fully DSM-unseen); p-bar_B defined on {int(keepB.sum())}")

    # ---- the 12 s0 MAMNet/DINOv3 parquets are the bytes 04 read ----
    ident = []
    for arch in ("mamnet", "dinov3"):
        for fold in FOLDS:
            for mode in ("loco", "upper"):
                rid = metas[(arch, fold, 0, mode)]["run_id"]
                old = os.path.join(OLD_DUMPS, rid, "perimage.parquet")
                ident.append(dict(run_id=rid, identical=os.path.isfile(old)
                                  and sha256_file(old) == sha256_file(paths[(arch, fold, 0, mode)]),
                                  probs_npz_on_box=os.path.isfile(os.path.join(OLD_DUMPS, rid, "probs.npz"))))
    ident = pd.DataFrame(ident)
    print("\n--- s0 MAMNet/DINOv3 parquets vs the dumps/seo copies 04 read ---")
    print(ident.to_string(index=False))
    if not ident.identical.all():
        fail("an s0 MAMNet/DINOv3 parquet differs from the one 04 read")
    n_npz_tree = sum(1 for k in data for _ in [0]
                     if os.path.isfile(os.path.join(TREE, sel[k]["dir"], "probs.npz")))
    print(f"probs.npz beside the 72 analysed parquets in delta_dumps/: {n_npz_tree}; "
          f"in dumps/seo for the 12 s0 MAMNet/DINOv3 cells: {int(ident.probs_npz_on_box.sum())}")

    if DRY_RUN:
        print("\nDRY RUN OK -- all assertions passed. Nothing written.")
        return

    # ---- 08 must be the file this append was written against ----
    sha_08 = sha256_file(RESULTS_PATH)
    if sha_08 != EXPECTED_08_SHA:
        fail(f"results/08 sha {sha_08[:12]} != expected {EXPECTED_08_SHA[:12]}; re-read before appending")
    with open(RESULTS_PATH) as fh:
        raw_08 = fh.read()
    j08 = json.loads(raw_08)
    if SECTION in j08:
        fail(f"{SECTION} already present in results/08")
    if json.dumps(j08, indent=1) != raw_08:
        fail("results/08 does not round-trip byte-identically through json.dumps(indent=1); "
             "an append would reformat existing content")
    if j08["index_sha256"] != index_sha:
        fail("08's index_sha256 differs from the live INDEX.parquet")

    # =================================================================
    # CROSS-CHECK vs results/04: 04's own function, 04's own streams
    # =================================================================
    print("\n" + "=" * 110)
    print("CROSS-CHECK -- s0 MAMNet/DINOv3, 04's paired_pair with 04's rng streams, Holm over 6")
    print("=" * 110)
    xc = {}
    for col, base, key04 in ((STAT_FG, 10_000, "task2_pbar_gt_shadow"),
                             (STAT_BG, 20_000, "task3_pbar_gt_bg_control")):
        rows = []
        for stream, (arch, fold) in enumerate(M4.PAIRS):
            r = M4.paired_pair(data[(arch, fold, 0, "loco")], data[(arch, fold, 0, "upper")],
                               col, base * (stream + 1))
            rows.append(dict(arch=arch, fold=fold, **r))
        t = pd.DataFrame(rows)
        t["p_holm"] = M4.holm(t.p_raw.values)
        if col == STAT_FG:
            t["verdict"] = M4.classify_pairs(t.delta.values, t.p_holm.values)
        ref = pd.DataFrame(r04[key04])
        m = t.merge(ref, on=["arch", "fold"], suffixes=("", "_04"))
        if len(m) != 6:
            fail(f"{col}: cross-check merge has {len(m)} rows")
        comp = []
        for _, r in m.iterrows():
            row = dict(arch=r.arch, fold=r.fold)
            ok = True
            for f in ("loco_mean", "upper_mean", "delta", "ci_lo", "ci_hi", "p_raw", "p_holm"):
                row[f] = round(float(r[f]), 4)
                row[f + "_04"] = round(float(r[f + "_04"]), 4)
                ok &= row[f] == row[f + "_04"]
            row["n_images"], row["n_images_04"] = int(r.n_images), int(r.n_images_04)
            ok &= row["n_images"] == row["n_images_04"]
            if col == STAT_FG:
                row["verdict"], row["verdict_04"] = r.verdict, r.verdict_04
                ok &= r.verdict == r.verdict_04
            row["match_4dp"] = bool(ok)
            comp.append(row)
        comp = pd.DataFrame(comp)
        print(f"\n{col}:")
        print(comp.to_string(index=False))
        if not comp.match_4dp.all():
            fail(f"{col}: s0 cross-check against results/04 does not reproduce to 4 dp")
        xc[col] = comp
    v04 = xc[STAT_FG]
    n_deg_04 = int((v04.verdict == "degradation").sum())
    exc_04 = [f"{r.arch}/{r.fold}" for _, r in v04.iterrows() if r.verdict != "degradation"]
    print(f"\n04's signature (delta < 0, Holm-6 p < 0.05) reproduces in {n_deg_04} of 6; "
          f"exception(s): {exc_04}")

    # =================================================================
    # TASK 5b -- 12 (arch x fold) pairs, seed stratum, Holm over 12
    # =================================================================
    print("\n" + "=" * 110)
    print(f"TASK 5b -- delta = LOCO - upper, per-image seed-averaged, AOI bootstrap B={B}, "
          "AOI sign-flip, Holm over 12 per statistic")
    print("=" * 110)
    res = {}
    for col, keepkey, tagc in ((STAT_FG, "keepS", "pbar_S"), (STAT_BG, "keepB", "pbar_B")):
        rows = []
        for arch in ARCHS:
            for fold in FOLDS:
                lo = {s: data[(arch, fold, s, "loco")] for s in SEEDS}
                up = {s: data[(arch, fold, s, "upper")] for s in SEEDS}
                r = paired_seed_stratified(lo, up, col, foldref[fold][keepkey], foldref[fold]["aoi"],
                                           f"t5b|{arch}|{fold}|{tagc}")
                rows.append(dict(arch=arch, fold=fold, **r))
        t = pd.DataFrame(rows)
        t["p_holm"] = M8.holm(t.p_raw.values)
        res[col] = t

    S, Bg = res[STAT_FG], res[STAT_BG]
    S["lower_and_holm_sig"] = (S.delta < 0) & (S.p_holm < 0.05)
    S["higher_and_holm_sig"] = (S.delta > 0) & (S.p_holm < 0.05)
    tbl = S[["arch", "fold", "loco_mean", "upper_mean", "delta", "stderr", "ci_lo", "ci_hi",
             "p_raw", "p_holm"]].copy()
    tbl = tbl.merge(Bg[["arch", "fold", "loco_mean", "upper_mean", "delta", "ci_lo", "ci_hi", "p_holm"]]
                    .rename(columns=lambda c: c if c in ("arch", "fold") else "B_" + c), on=["arch", "fold"])
    tbl["ratio_abs"] = tbl.delta.abs() / tbl.B_delta.abs().replace(0, np.nan)
    tbl["ratio_signed"] = tbl.delta / tbl.B_delta.replace(0, np.nan)
    print("\n--- per pair: p-bar_S (LOCO, upper, delta, CI, Holm-12 p) | p-bar_B control | ratio ---")
    print(tbl.to_string(index=False, float_format=fmt))
    S["per_seed_str"] = S.per_seed_delta.map(lambda d: " ".join(f"{v:+.4f}" for v in d.values()))
    print("\n--- p-bar_S per-seed deltas, seed SD, drops ---")
    print(S[["arch", "fold", "per_seed_str", "seed_sd", "n_seeds_same_sign_as_mean", "n_images",
             "n_excluded", "n_aoi", "n_aoi_dropped", "frac_aoi_negative", "delta_aoi_equal_weight",
             "ci_lo_aoi", "ci_hi_aoi"]].to_string(index=False, float_format=fmt))

    n_lower = int(S.lower_and_holm_sig.sum())
    n_higher = int(S.higher_and_holm_sig.sum())
    n_ns = 12 - n_lower - n_higher
    lower_by_arch = S.groupby("arch").lower_and_holm_sig.sum().astype(int).to_dict()
    lower_by_fold = S.groupby("fold").lower_and_holm_sig.sum().astype(int).to_dict()
    worst = S.loc[S.delta.idxmax()]
    best = S.loc[S.delta.idxmin()]
    n_bg_sig = int((Bg.p_holm < 0.05).sum())
    print(f"\nLOCO p-bar_S lower and Holm-significant: {n_lower} of 12 "
          f"(higher and Holm-significant: {n_higher}; ns: {n_ns})")
    print(f"  by arch {lower_by_arch}; by fold {lower_by_fold}")
    print(f"  largest delta (worst case for the signature): {worst.arch}/{worst.fold} {worst.delta:+.4f}; "
          f"most negative: {best.arch}/{best.fold} {best.delta:+.4f}")
    print(f"  p-bar_B control Holm-significant in {n_bg_sig} of 12")

    # s0 slice of the new pipeline vs 04 (per-seed deltas must equal 04's point estimates)
    s0x = []
    for _, r in S[S.arch.isin(["mamnet", "dinov3"])].iterrows():
        ref = xc[STAT_FG].query("arch == @r.arch and fold == @r.fold").iloc[0]
        s0x.append(dict(arch=r.arch, fold=r.fold, s0_delta_new=round(r.per_seed_delta[0], 4),
                        delta_04=ref.delta_04, s0_loco_new=round(r.per_seed_loco[0], 4),
                        loco_04=ref.loco_mean_04, s0_upper_new=round(r.per_seed_upper[0], 4),
                        upper_04=ref.upper_mean_04))
    s0x = pd.DataFrame(s0x)
    s0x["match_4dp"] = ((s0x.s0_delta_new == s0x.delta_04) & (s0x.s0_loco_new == s0x.loco_04)
                        & (s0x.s0_upper_new == s0x.upper_04))
    print("\n--- seed-0 slice of the 12-pair pipeline vs results/04 ---")
    print(s0x.to_string(index=False))
    if not s0x.match_4dp.all():
        fail("the seed-0 slice of the seed-stratified pipeline does not reproduce 04 to 4 dp")
    dv = S[(S.arch == "dinov3") & (S.fold == "oma")].iloc[0]

    # ---- sanity flags ----
    flags = []
    floor = 1.0 / (B + 1.0) + 1e-12
    for name, t in (("p-bar_S", S), ("p-bar_B", Bg)):
        nf = int((t.p_raw <= floor).sum())
        if nf:
            flags.append(f"{name}: {nf} of 12 sign-flip p-values sit at the permutation floor "
                         f"1/(B+1) = {1/(B+1):.4e}; the test cannot resolve below that at B = {B}.")
        for _, r in t[t.ci_excludes_point].iterrows():
            flags.append(f"{name} {r.arch}/{r.fold}: bootstrap CI excludes its own point estimate")
        for _, r in t[t.n_aoi_dropped > 0].iterrows():
            flags.append(f"{name} {r.arch}/{r.fold}: {int(r.n_aoi_dropped)} AOI(s) have no defined patch "
                         f"and leave the bootstrap ({int(r.n_aoi)} AOIs resampled).")
    mixed = S[S.n_seeds_same_sign_as_mean < 3]
    for _, r in mixed.iterrows():
        flags.append(f"p-bar_S {r.arch}/{r.fold}: per-seed deltas do not share the sign of the "
                     f"seed-averaged delta ({r.per_seed_str}).")
    lowr = tbl[tbl.ratio_abs < 1]
    for _, r in lowr.iterrows():
        flags.append(f"{r.arch}/{r.fold}: |delta p-bar_B| ({abs(r.B_delta):.4f}) exceeds |delta p-bar_S| "
                     f"({abs(r.delta):.4f}); ratio {r.ratio_abs:.2f}.")
    sd_min = min(float(data[k][STAT_FG].std()) for k in data)
    if sd_min < 1e-3:
        flags.append(f"a cell's per-image p-bar_S SD is {sd_min:.2e}, implausibly low")
    for k, df in data.items():
        x = df[STAT_FG].dropna()
        if x.min() <= 0.0 or x.max() >= 1.0:
            flags.append(f"{metas[k]['run_id']}: per-image p-bar_S touches a bound "
                         f"[{x.min():.4f}, {x.max():.4f}]")
    flags.append("The 4-dp agreement with results/04 on the s0 MAMNet/DINOv3 pairs is exact by "
                 "construction (byte-identical parquets, 04's own function and rng streams); it "
                 "verifies the lift, not the statistic.")
    print("\n--- sanity flags ---")
    print("\n".join(f"  - {f}" for f in flags))

    # ---- assemble the section ----
    pairs_out = []
    for (_, r), (_, rb), (_, rt) in zip(S.iterrows(), Bg.iterrows(), tbl.iterrows()):
        assert (r.arch, r.fold) == (rb.arch, rb.fold) == (rt.arch, rt.fold)
        pairs_out.append(dict(
            arch=r.arch, fold=r.fold,
            pbar_S=r6({k: r[k] for k in (
                "loco_mean", "upper_mean", "delta", "stderr", "ci_lo", "ci_hi", "p_raw", "p_holm",
                "delta_aoi_equal_weight", "ci_lo_aoi", "ci_hi_aoi", "per_seed_delta", "per_seed_loco",
                "per_seed_upper", "seed_sd", "n_seeds_same_sign_as_mean", "n_images", "n_excluded",
                "n_aoi", "n_aoi_dropped", "frac_aoi_negative", "ci_excludes_point",
                "lower_and_holm_sig", "higher_and_holm_sig")}),
            pbar_B_control=r6({k: rb[k] for k in (
                "loco_mean", "upper_mean", "delta", "stderr", "ci_lo", "ci_hi", "p_raw", "p_holm",
                "per_seed_delta", "seed_sd", "n_images", "n_excluded", "n_aoi", "n_aoi_dropped",
                "ci_excludes_point")}),
            fg_over_bg_abs_delta_ratio=r6(float(rt.ratio_abs)),
            fg_over_bg_signed_delta_ratio=r6(float(rt.ratio_signed))))
    runs = sorted(metas[k]["run_id"] for k in data)
    missing_runs = sorted(set(runs) - set(j08["runs"]))
    headline = (
        f"On the S-EO vanilla grid of 4 architectures x 3 holdout cities x 3 seeds, Figure 2's p-bar_S "
        f"(mean predicted shadow probability on GT-foreground pixels, valid region, DSM-unseen excluded) "
        f"is lower under source-only LOCO than under the within-city upper bound at Holm-12 p < 0.05 in "
        f"{n_lower} of 12 (architecture x city) pairs and higher at Holm-12 p < 0.05 in {n_higher}, with "
        f"seed-averaged deltas from {best.delta:+.4f} ({best.arch}/{best.fold}) to {worst.delta:+.4f} "
        f"({worst.arch}/{worst.fold}); the GT-background control p-bar_B differs at Holm-12 p < 0.05 in "
        f"{n_bg_sig} of 12.")
    bg_up, bg_dn = Bg[(Bg.delta > 0) & (Bg.p_holm < 0.05)], Bg[(Bg.delta < 0) & (Bg.p_holm < 0.05)]
    caveats = [
        "Observational, not interventional: a paired comparison of two trained conditions on a shared "
        "test pool. It carries no causal claim about what lowers p-bar_S.",
        f"The GT-background control is not null: p-bar_B differs between LOCO and upper at Holm-12 "
        f"p < 0.05 in {n_bg_sig} of 12 pairs ({len(bg_up)} upward under LOCO, {len(bg_dn)} downward). "
        f"Per-pair |delta p-bar_S| / |delta p-bar_B| runs {np.nanmin(tbl.ratio_abs):.2f} to "
        f"{np.nanmax(tbl.ratio_abs):.2f}.",
        "Three seeds per cell. Seed is a stratum exactly as in task 1: the per-image delta is formed "
        "within seed (LOCO s_k minus upper s_k), averaged over the three seeds per image, and AOIs are "
        "resampled with all seeds kept. Intervals cover test-patch and AOI sampling, and seed variation "
        "only through the seed average; per-seed deltas and the across-seed SD (n = 3) are reported.",
        "Pixel population is 04's: GT foreground on the valid region (+-2 px don't-care band removed, "
        "DSM-unseen pixels removed), not the all-pixel population the published figure script uses on a "
        "rendered tile. Same estimator (metrics.pbar_populations).",
        "Patches with no GT foreground on the valid region (including every fully DSM-unseen patch) have "
        "an undefined p-bar_S and are dropped; the dropped set is asserted identical across all 24 cells "
        "of a fold. Counts per pair are in pbar_S.n_excluded; p-bar_B has its own drop set.",
        "probs.npz semantics check: NOT DONE for the 60 cells added since results/04. It needs probs.npz, "
        "and no probs.npz is on this box for those cells. For the 12 s0 MAMNet/DINOv3 cells the check was "
        "done in results/04; the parquets analysed here are byte-identical to the ones 04 read. No "
        "substitute check was run in its place; the in-range and NaN-iff-empty assertions here are input "
        "validation carried over from 04's loader.",
        "Budget asymmetry inherited from the grid: LOCO trains on 3,500 patches from the two source "
        "cities, the upper bound on 3,500 from the held-out city.",
        "S-EO labels are auto-generated. Both sides of each delta use the same labels on the same patches; "
        "absolute p-bar levels carry the label noise.",
        "rung 3500 is the S-EO patch budget on 384 px patches, not a point on the ShadowTransfer "
        "225/450/900 ladder. Levels are not comparable to any internal p-bar_S number, and no internal "
        "number is used here.",
        "Primary cells are the inventory-named originals, as in task 1. The D-A2-1 -e200 retrains "
        "(oglanet vanilla-loco OMA s0, none-upper UCSD s0) are not substituted; no p-bar_S sensitivity "
        "with them was run. Five of the 72 cells stopped on the epoch ceiling (08 caveats).",
        "OGLANet inference is nondeterministic (E-9); every OGLANet cell is one inference draw.",
        "Holm families: 12 (arch x fold) pairs for p-bar_S, and separately 12 for p-bar_B. The 04 "
        "cross-check uses 04's own family of 6.",
        "Nothing AURC-shaped is computed or read.",
        "The pre-commitment for the 12-pair extended grid is not on this box (DECISIONS_armA_wave2.md is "
        "absent). results/04's pre-committed rule is on this box but is defined over 6 pairs "
        "(degradation >= 4 of 6); it is not applied to 12. No branch is named.",
    ]
    section = dict(
        appended_utc=datetime.now(timezone.utc).isoformat(),
        appended_by="analysis/08b_seo_pbar.py (CC-SSH, analysis host)",
        append_note=(
            "Second pass, appended to an unchanged 08 (sha256 before append "
            f"{EXPECTED_08_SHA}). Every pre-existing key is unchanged. task5.pbar_S ('NOT COMPUTED') and "
            "the top-level caveat 'p-bar_S ... is not computed in this pass' describe the first pass and are "
            "superseded by this section. The top-level flag/caveat on 73 probe sidecars describes their "
            "state at the first pass; see the 2026-10-24 armA-pbar session log for the repair."),
        answers=["v1UJ-W1", "AC1"],
        question=("On the extended S-EO vanilla grid (4 architectures x 3 folds x 3 seeds), is Figure 2's "
                  "p-bar_S lower under source-only LOCO than under the within-city upper bound, measured "
                  "with the published statistic and 04's population, and how does the GT-background "
                  "control move?"),
        runs=runs,
        runs_not_in_08_runs=missing_runs,
        index_sha256=index_sha,
        script="analysis/08b_seo_pbar.py",
        script_sha256=sha256_file(os.path.abspath(__file__)),
        git_hash=None,
        seed=SEED, B=B,
        lifted_functions=dict(
            estimator=dict(name="shadowtransfer.metrics.pbar_populations (called upstream by seo_dump.py; "
                                "stored as pbar_gt_shadow / pbar_gt_bg)",
                           module_path=M8.METRICS_PATH, module_sha256=metrics_sha,
                           sha_recorded_by_04=sha_04_recorded, sha_matches_04=True,
                           sidecar_metric_module_sha_all_72_equal=True),
            paired_04=dict(names=["paired_pair", "holm", "classify_pairs", "aoi_cluster_bootstrap_aoimean"],
                           file="analysis/04_seo_probspace.py", file_sha256=sha256_file(SCRIPT_04),
                           note="file mtime precedes results/04 by %.1f s; unchanged since 04 ran"
                                % (r4_mtime - s4_mtime)),
            seed_stratum_08_task1=dict(names=["aoi_cluster_bootstrap_imagemean", "aoi_signflip_imagemean",
                                              "stream_of", "ci95", "holm"],
                                       file="analysis/08_seo_full_grid.py", file_sha256=sha_08_script)),
        population_definition=(
            "per patch, mean unthresholded P(shadow) over GT-foreground pixels on the valid region "
            "(+-2 px K=2 don't-care band removed, DSM-unseen pixels removed); NaN when that population is "
            "empty; NaN patches dropped pairwise (set asserted identical across every cell of a fold); "
            "point estimate is the image-level mean of the per-image seed-averaged delta; AOIs with no "
            "defined patch leave the bootstrap. p-bar_B: the same over GT background."),
        delta_convention="delta = p-bar(LOCO) - p-bar(upper); negative on p-bar_S is the Figure 2 direction",
        ci_method=f"AOI cluster bootstrap, B={B}, percentile, seed as a stratum (task 1's scheme); "
                  "rng default_rng([0, sha256(label)[:8]])",
        test="two-sided AOI-level sign-flip permutation on the seed-averaged per-image delta, "
             "p = (1 + #{|T*| >= |T_obs|}) / (B + 1); Holm over the 12 (arch x fold) pairs per statistic",
        probs_npz_semantics_check=dict(
            done_without_probs_npz=False,
            status="NOT DONE for the 60 cells added since results/04 (no probs.npz on this box for them). "
                   "Not substituted.",
            covered_by_04=[r for r in ident.run_id],
            covered_by_04_note="04 ran the check on these 12 s0 MAMNet/DINOv3 cells (1 in full, 2-patch spot "
                               "checks on 11); the parquets read here are byte-identical to 04's inputs.",
            probs_npz_in_delta_dumps_for_analysed_cells=int(n_npz_tree)),
        count=dict(n_pairs=12, n_lower_and_holm_sig=n_lower, n_higher_and_holm_sig=n_higher, n_ns=n_ns,
                   lower_and_holm_sig_by_arch=lower_by_arch, lower_and_holm_sig_by_fold=lower_by_fold,
                   n_pbar_B_holm_sig=n_bg_sig),
        grid_summary=dict(
            mean_of_12_pair_deltas_pbar_S=r6(float(S.delta.mean())),
            mean_note="unweighted mean of the 12 seed-averaged pair deltas; descriptive, no interval, no test",
            worst_case_pbar_S=dict(arch=worst.arch, fold=worst.fold, delta=r6(worst.delta),
                                   ci95=[r6(worst.ci_lo), r6(worst.ci_hi)], p_holm=r6(worst.p_holm),
                                   note="largest (least negative) delta, i.e. the pair least in the Figure 2 "
                                        "direction"),
            most_negative_pbar_S=dict(arch=best.arch, fold=best.fold, delta=r6(best.delta))),
        pairs=pairs_out,
        crosscheck_vs_04=dict(
            method="04's paired_pair, 04's rng streams (10000*(i+1) for p-bar_S, 20000*(i+1) for p-bar_B, "
                   "fold-major pair order), Holm over 04's 6, on the byte-identical s0 parquets",
            pbar_S=xc[STAT_FG].to_dict("records"),
            pbar_B=xc[STAT_BG].to_dict("records"),
            all_match_4dp=True,
            n_degradation_of_6=n_deg_04, exceptions=exc_04,
            reproduces_04_5_of_6_with_dinov3_oma_exception=bool(n_deg_04 == 5 and exc_04 == ["dinov3/oma"]),
            seed0_slice_of_12pair_pipeline=s0x.to_dict("records"),
            dinov3_oma_in_12pair_seed_averaged=dict(delta=r6(dv.delta), ci95=[r6(dv.ci_lo), r6(dv.ci_hi)],
                                                    p_holm_12=r6(dv.p_holm),
                                                    per_seed_delta=r6(dv.per_seed_delta))),
        precommitted_branch=dict(
            letter=None,
            note="Not named. The pre-commitment for the 12-pair extended grid is not on this box "
                 "(DECISIONS_armA_wave2.md absent). results/04's pre-committed rule (branches 1/2/3, "
                 "degradation >= 4 of 6) is on this box but is defined over 6 pairs and is not applied "
                 "to 12. The reading is the operator's and HLT's."),
        conditions=[
            dict(label=f"{r.arch} / {r.fold} / {mode} (3-seed mean)", metric=STAT_FG,
                 mean=r6(r[f"{mode}_mean"]), stderr=None, n=int(r.n_images),
                 ci95=None, ci_method=None,
                 baseline=("within-city upper bound" if mode == "loco" else "self (this is the baseline)"))
            for _, r in S.iterrows() for mode in ("loco", "upper")] + [
            dict(label=f"{r.arch} / {r.fold} / LOCO - upper", metric="delta_" + STAT_FG,
                 mean=r6(r.delta), stderr=r6(r.stderr), n=int(r.n_images), ci95=[r6(r.ci_lo), r6(r.ci_hi)],
                 ci_method=f"aoi_cluster_bootstrap_seed_stratum_B{B}", baseline="within-city upper bound")
            for _, r in S.iterrows()] + [
            dict(label=f"{r.arch} / {r.fold} / LOCO - upper", metric="delta_" + STAT_BG,
                 mean=r6(r.delta), stderr=r6(r.stderr), n=int(r.n_images), ci95=[r6(r.ci_lo), r6(r.ci_hi)],
                 ci_method=f"aoi_cluster_bootstrap_seed_stratum_B{B}", baseline="within-city upper bound")
            for _, r in Bg.iterrows()],
        conditions_note="per-cell level conditions carry no interval here (levels are 3-seed means over the "
                        "pair's defined patches); intervals are on the paired deltas.",
        tests=[
            dict(name="AOI-level sign-flip permutation, seed stratum, Holm over 12",
                 compares=[f"{r.arch}/{r.fold} loco", f"{r.arch}/{r.fold} upper"], metric=col,
                 statistic=r6(r.delta), p=r6(r.p_raw), p_holm=r6(r.p_holm), B=B, two_tailed=True)
            for col, t in ((STAT_FG, S), (STAT_BG, Bg)) for _, r in t.iterrows()],
        headline=headline,
        sanity_flags=flags,
        caveats=caveats,
        figure=None, caption=None,
    )

    # ---- append: .new -> verify -> mv ----
    new = copy.deepcopy(j08)
    new[SECTION] = section
    text = json.dumps(new, indent=1, default=str)
    back = json.loads(text)
    for k in j08:
        if back[k] != j08[k]:
            fail(f"pre-existing key {k} changed in the appended file")
    if set(back) - set(j08) != {SECTION}:
        fail(f"unexpected new keys {set(back) - set(j08)}")
    if not text.startswith(raw_08[:-2]):
        fail("appended file does not begin with the original bytes")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(BACKUP_DIR, f"08_seo_full_grid.pre_pbar_{ts}.json")
    shutil.copy2(RESULTS_PATH, backup)
    if sha256_file(backup) != EXPECTED_08_SHA:
        fail("backup copy sha mismatch")
    tmp = RESULTS_PATH + ".new"
    with open(tmp, "w") as fh:
        fh.write(text)
    with open(tmp) as fh:
        if json.load(fh) != back:
            fail(".new does not read back")
    if sha256_file(RESULTS_PATH) != EXPECTED_08_SHA:
        fail("results/08 changed during the run")
    os.replace(tmp, RESULTS_PATH)
    print(f"\nbackup of 08 before append: {backup} (sha256 {EXPECTED_08_SHA[:12]}...)")
    print(f"appended key '{SECTION}' to {RESULTS_PATH}; new sha256 {sha256_file(RESULTS_PATH)}")
    print(f"bytes before {len(raw_08)}, after {len(text)}; pre-existing {len(j08)} keys unchanged")
    print(f"HEADLINE: {headline}")


if __name__ == "__main__":
    main()
