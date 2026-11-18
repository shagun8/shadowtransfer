#!/usr/bin/env python
"""
Arm A -- results/34: results/08's S-EO SIB-versus-Vanilla family recomputed with
the OGLANet-SIB cells rerun on the HELD-OUT-CITY FDA pool.

Answers the same IDs as results/08's SIB family (results/08 `answers`).

WHY. SIB's FDA input preprocessing draws its style pool from UNLABELED images of
the held-out city's train split. The S-EO OGLANet-SIB cells results/08 used drew
it from the source cities (cross-paired, 200). They were rerun with the correct
pool at seed 0 only: dumps/seo/oglanet-sib-loco-r3500-{oma,jax,ucsd}-hr-s0-tgtfda
(100 donor patches selected from the held-out city's 3,500 train candidates;
delta_logs/session_logs/2026-11-02-oglanet-sib-tgtfda-results.md). No seed 1 or
2 exists, by operator ruling.

TWO TABLES.
  (i)  HEADLINE, seed 0 for all four architectures. SIB and Vanilla LOCO seed-0
       cells only. Per (architecture, held-out city): SIB minus Vanilla,
       two-class mIoU at K=2 with DSM-unseen pixels excluded, AOI cluster
       bootstrap 95% CI and AOI sign-flip permutation p (both results/08's own
       functions), Holm over the 12. Then SIB's worst cell against Vanilla's
       worst cell over the 12 seed-0 cells and their difference.
  (ii) CONTEXT, labelled MIXED-SEED and never the headline. results/08's
       seed-mean grid (3 seeds) with only the three OGLANet-SIB levels replaced
       by their seed-0 tgtfda values; Vanilla stays at its 3-seed mean.

REUSE. analysis/08_seo_full_grid.py is imported, not copied: catalogue(),
build_foldref(), cell(), add_metrics(), paired_stats() (AOI cluster bootstrap +
AOI sign-flip, B = 10,000, rng default_rng([0, sha-derived stream])), holm(),
N_TEST. Its main() and dry_run() are never called (dry_run opens a metrics.py
path outside the project). The nine unchanged seed-mean rows are recomputed with
08's own stream labels and asserted against results/08's method_table, and the
nine unchanged seed-0 deltas against results/08's per_seed values, so the import
is checked rather than trusted.

TREES. Vanilla and the non-OGLANet SIB cells come from delta_dumps/ (results/08's
tree, via its catalogue). The three new cells come from dumps/seo/. Both are
indexed into analysis/_inv/INDEX_seo_34.parquet and cells are selected from that
index. Nothing under dumps/ or delta_dumps/ is written.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/34_seo_sib_family_tgtfda.py [--dry-run]
"""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---- config ----------------------------------------------------------------
BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
B = 10_000
SEO = os.path.join(BASE, "dumps", "seo")
INV = os.path.join(BASE, "analysis", "_inv")
INDEX_OUT = os.path.join(INV, "INDEX_seo_34.parquet")
R08 = os.path.join(BASE, "results", "08_seo_full_grid.json")
OUT = os.path.join(BASE, "results", "34_seo_sib_family_tgtfda.json")
NEW_DIR = "oglanet-sib-loco-r3500-{fold}-hr-s0-tgtfda"
ARCHS = ["mamnet", "dinov3", "oglanet", "crossearth"]
FOLDS = ["oma", "jax", "ucsd"]
ALPHA = 0.05
# ------------------------------------------------------------------------------


def fail(msg):
    raise SystemExit(f"ASSERTION FAILED: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


spec = importlib.util.spec_from_file_location(
    "r08", os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_seo_full_grid.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
if E.B != B or E.SEED != 0 or list(E.FOLDS) != FOLDS or list(E.ARCHS) != [
        "mamnet", "dinov3", "oglanet", "crossearth"]:
    fail("analysis/08's B / SEED / FOLDS / ARCHS are not what this job assumes")


def r6(x):
    return None if x is None else round(float(x), 6)


# ---- the index ---------------------------------------------------------------

def seo_tree_rows():
    """One record per dumps/seo/ directory, in the shape of 08's catalogue."""
    out = []
    for name in sorted(os.listdir(SEO)):
        d = os.path.join(SEO, name)
        if not os.path.isdir(d):
            continue
        fn = set(os.listdir(d))
        r = dict(dir=os.path.relpath(d, BASE), run_id_dir=name, tree="dumps/seo",
                 has_parquet="perimage.parquet" in fn,
                 has_run_json="run.json" in fn, done=".DONE" in fn,
                 failed=".FAILED" in fn)
        meta = json.load(open(os.path.join(d, "run.json"))) if r["has_run_json"] else {}
        r.update(meta=meta, run_id=meta.get("run_id"), dataset=meta.get("dataset"),
                 arch=meta.get("arch"), method=meta.get("method"),
                 condition=meta.get("mode"),
                 holdout=str(meta.get("holdout", "")).lower() or None,
                 seed=meta.get("seed"), smoke=meta.get("smoke"),
                 target_access=meta.get("target_access"),
                 metric_module_sha=meta.get("metric_module_sha"),
                 config_sha256=meta.get("config_sha256"))
        if r["has_parquet"] and not r["has_run_json"]:
            fail(f"dumps/seo/{name}: perimage.parquet with no run.json sidecar")
        out.append(r)
    return out


def build_index(cat, seo_rows):
    rows = []
    for tree, recs, base in (("delta_dumps", cat, E.TREE),
                             ("dumps/seo", seo_rows, BASE)):
        for r in recs:
            x = {k: r.get(k) for k in ("run_id_dir", "run_id", "category",
                                        "excluded_reason", "dataset", "arch",
                                        "method", "condition", "holdout", "seed",
                                        "suffix", "smoke", "target_access",
                                        "metric_module_sha", "config_sha256",
                                        "has_parquet")}
            x["tree"] = tree
            x["dir"] = r["dir"]
            meta = r.get("meta") or {}
            x["dump_module_sha"] = meta.get("dump_module_sha")
            x["fda_donor"] = meta.get("fda_donor")
            x["run_json_scalars"] = json.dumps(
                {k: v for k, v in meta.items()
                 if v is None or isinstance(v, (str, int, float, bool))},
                sort_keys=True)
            if r.get("has_parquet"):
                p = os.path.join(base, r["dir"], "perimage.parquet")
                x["n_images"] = pq.read_metadata(p).num_rows
                x["parquet_mtime"] = datetime.fromtimestamp(
                    os.path.getmtime(p), timezone.utc).isoformat()
                x["dump_sha256"] = sha256_file(p)
            rows.append(x)
    idx = pd.DataFrame(rows)
    for c in ("seed", "n_images"):
        idx[c] = idx[c].astype("Int64")
    for c in idx.columns:
        if idx[c].dtype == object:
            idx[c] = idx[c].map(lambda v: None if v is None or (
                isinstance(v, float) and np.isnan(v)) else str(v))
    return idx


def select_one(idx, **kw):
    m = np.ones(len(idx), dtype=bool)
    for k, v in kw.items():
        m &= (idx[k].astype(object) == v).to_numpy()
    hit = idx[m]
    if len(hit) != 1:
        fail(f"index selection {kw}: {len(hit)} rows")
    return hit.iloc[0]


# ---- the three new cells -------------------------------------------------------

def load_new_cell(fold, rows, seo_by_dir):
    name = NEW_DIR.format(fold=fold)
    r = seo_by_dir.get(name)
    if r is None:
        fail(f"dumps/seo/{name} absent")
    meta = r["meta"]
    checks = []

    def need(label, ok, got):
        checks.append(dict(check=label, ok=bool(ok), got=got))
        if not ok:
            fail(f"{name}: {label} (got {got!r})")

    need(".DONE present", r["done"], r["done"])
    need("no .FAILED", not r["failed"], r["failed"])
    need("run.json present", r["has_run_json"], r["has_run_json"])
    need("run_id == directory name", meta.get("run_id") == name, meta.get("run_id"))
    need("dataset seo", meta.get("dataset") == "seo", meta.get("dataset"))
    need("arch oglanet / method sib / mode loco",
         (meta.get("arch"), meta.get("method"), meta.get("mode"))
         == ("oglanet", "sib", "loco"),
         (meta.get("arch"), meta.get("method"), meta.get("mode")))
    need("holdout == fold", str(meta.get("holdout")).lower() == fold, meta.get("holdout"))
    need("seed 0", meta.get("seed") == 0, meta.get("seed"))
    need("not a smoke", not meta.get("smoke"), meta.get("smoke"))
    need("target_access unlabeled_target",
         meta.get("target_access") == "unlabeled_target", meta.get("target_access"))
    need("fda_donor heldout_city_train_patches",
         meta.get("fda_donor") == "heldout_city_train_patches", meta.get("fda_donor"))
    need("fda_donor_city == holdout",
         str(meta.get("fda_donor_city")).lower() == fold, meta.get("fda_donor_city"))
    need("n_test == fold n_test", meta.get("n_test") == E.N_TEST[fold], meta.get("n_test"))
    # comparators in 08's tree: the superseded cross-source cell and Vanilla s0
    old = E.find(rows, "oglanet", "sib", "loco", fold, 0)
    van = E.find(rows, "oglanet", "vanilla", "loco", fold, 0)
    for fld in ("metric_module_sha", "dump_module_sha"):
        need(f"{fld} == superseded cell's and Vanilla s0's",
             meta.get(fld) == old["meta"].get(fld) == van["meta"].get(fld),
             (meta.get(fld), old["meta"].get(fld), van["meta"].get(fld)))
    p = os.path.join(SEO, name, "perimage.parquet")
    df = pd.read_parquet(p)
    ref = pd.read_parquet(os.path.join(E.TREE, old["dir"], "perimage.parquet"))
    need("column set == superseded cell's", set(df.columns) == set(ref.columns),
         sorted(set(df.columns) ^ set(ref.columns)))
    bad = [c for c in ref.columns if df[c].dtype != ref[c].dtype]
    need("every dtype == superseded cell's (no coercion)", not bad,
         [(c, str(df[c].dtype), str(ref[c].dtype)) for c in bad])
    missing = [c for c in E.REQUIRED_NAMED if c not in df.columns]
    need("results/08 REQUIRED_NAMED columns present", not missing, missing)
    need("row count == n_test", len(df) == E.N_TEST[fold], len(df))
    df = df.sort_values("image_id").reset_index(drop=True)
    need("image_id order == fold reference",
         np.array_equal(df.image_id.to_numpy(), E.FOLDREF[fold]["image_id"]), None)
    df = E.add_metrics(df)
    d1 = float(np.abs(df.miou2_pct - df.miou_tol_pct).max())
    need("recomputed two-class == stored miou_tol_pct (1e-9)", d1 < 1e-9, d1)
    d2 = float(np.abs(df.fg_pct - df.iou_tol * 100.0).max())
    need("recomputed fg == 100*iou_tol (1e-9)", d2 < 1e-9, d2)
    rv = meta.get("reporting_metric_value_pct")
    d3 = None if rv is None else abs(float(df.miou2_pct.mean()) - float(rv))
    # run.json stores the reporting metric at 4 dp on every S-EO cell, old and new
    need("cell mean == run.json reporting_metric_value_pct (stored at 4 dp; 5e-5)",
         d3 is not None and d3 <= 5e-5, d3)
    return df, meta, checks, old


# ---- main ----------------------------------------------------------------------

def main():
    print("=" * 100)
    print("results/34 -- S-EO SIB vs Vanilla with OGLANet-SIB on the held-out-city "
          f"FDA pool   DRY_RUN={DRY_RUN}  B={B}")
    print("=" * 100)
    r08 = json.load(open(R08))
    rows = E.catalogue()
    E.build_foldref(rows)
    seo_rows = seo_tree_rows()
    seo_by_dir = {r["run_id_dir"]: r for r in seo_rows}
    idx = build_index(rows, seo_rows)
    print(f"index: {len(idx)} rows ({(idx.tree == 'delta_dumps').sum()} delta_dumps, "
          f"{(idx.tree == 'dumps/seo').sum()} dumps/seo)")

    # run_ids present in both trees: does the dump agree?
    both = sorted(set(idx[idx.tree == "dumps/seo"].run_id.dropna())
                  & set(idx[(idx.tree == "delta_dumps")
                            & (idx.category == "cell")].run_id.dropna()))
    cross_tree = []
    for rid in both:
        a = select_one(idx, tree="delta_dumps", run_id=rid, category="cell")
        b = idx[(idx.tree == "dumps/seo") & (idx.run_id == rid)].iloc[0]
        cross_tree.append(dict(run_id=rid, same_dump_sha256=a.dump_sha256 == b.dump_sha256))
    print(f"run_ids in both trees: {len(both)}; identical parquet: "
          f"{sum(c['same_dump_sha256'] for c in cross_tree)}")

    # ---- the six directories, before computing anything ----
    print("\n--- the six new directories (target_access, fda_donor) ---")
    six = []
    for c in ("chicago", "miami", "phoenix"):
        d = os.path.join(BASE, "dumps", "st_r450_published_recompute",
                         f"oglanet-sibc4tgt-loco-r450-{c}-hr-s42-n450")
        m = json.load(open(os.path.join(d, "run.json")))
        six.append(dict(dir=os.path.relpath(d, BASE), target_access=m["target_access"],
                        fda_donor=m["fda_donor"], fda_donor_n=m.get("fda_donor_n"),
                        used_in="results/31-33"))
    for f in FOLDS:
        d = os.path.join(SEO, NEW_DIR.format(fold=f))
        m = json.load(open(os.path.join(d, "run.json")))
        six.append(dict(dir=os.path.relpath(d, BASE), target_access=m["target_access"],
                        fda_donor=m["fda_donor"],
                        fda_donor_n=m.get("fda_donor_n_selected"),
                        used_in="results/34"))
    for s in six:
        print(f"  {s['dir']:80s} {s['target_access']:17s} {s['fda_donor']} "
              f"(n {s['fda_donor_n']})")

    # ---- cells ----
    new, new_checks, old_recs = {}, {}, {}
    for f in FOLDS:
        df, meta, checks, old = load_new_cell(f, rows, seo_by_dir)
        new[f] = (df, meta)
        new_checks[f] = checks
        old_recs[f] = old
    used = set()

    def sib_s0(arch, fold):
        if arch == "oglanet":
            df, meta = new[fold]
            sel = select_one(idx, tree="dumps/seo", run_id=meta["run_id"])
            return df, meta["run_id"], meta["target_access"], sel.dir
        rec = E.find(rows, arch, "sib", "loco", fold, 0)
        select_one(idx, tree="delta_dumps", run_id=rec["run_id"], category="cell")
        return E.cell(rec), rec["run_id"], rec["target_access"], rec["dir"]

    def van(arch, fold, s):
        rec = E.find(rows, arch, "vanilla", "loco", fold, s)
        select_one(idx, tree="delta_dumps", run_id=rec["run_id"], category="cell")
        return E.cell(rec), rec["run_id"]

    print("\n--- cells selected (seed 0) ---")
    for a in ARCHS:
        for f in FOLDS:
            S, srid, sacc, sdir = sib_s0(a, f)
            V, vrid = van(a, f, 0)
            print(f"  {a:10s} {f:5s} SIB {srid:42s} [{sacc:16s}] n={len(S):5d}   "
                  f"Vanilla {vrid} n={len(V)}")
    if DRY_RUN:
        for f in FOLDS:
            print(f"  new-cell checks {f}: "
                  f"{sum(c['ok'] for c in new_checks[f])}/{len(new_checks[f])} pass")
        print("\nDRY_RUN: nothing written.")
        return

    r08_t2 = {(r["arch"], r["fold"]): r for r in r08["task2"]["method_table"]
              if r["method"] == "sib" and r["metric"] == "2class"}

    # ================= (i) HEADLINE: seed 0 ==================
    t_i, lev_s0 = [], {}
    for a in ARCHS:
        for f in FOLDS:
            S, srid, sacc, _ = sib_s0(a, f)
            V, vrid = van(a, f, 0)
            used.update([srid, vrid])
            d = (S.miou2_pct - V.miou2_pct).to_numpy(np.float64)
            st = E.paired_stats({0: d}, f, f"t34|s0|sib|{a}|{f}|2class")
            lev_s0[("sib", a, f)] = float(S.miou2_pct.mean())
            lev_s0[("vanilla", a, f)] = float(V.miou2_pct.mean())
            r08_s0 = r08_t2[(a, f)]["per_seed"]["0"]
            if a != "oglanet" and abs(st["mean"] - r08_s0) > 1e-6:
                fail(f"{a}/{f}: seed-0 delta {st['mean']} != results/08 per_seed "
                     f"{r08_s0}")
            t_i.append(dict(arch=a, fold=f, sib_run_id=srid, vanilla_run_id=vrid,
                            sib_target_access=sacc,
                            sib_level=lev_s0[("sib", a, f)],
                            vanilla_level=lev_s0[("vanilla", a, f)],
                            delta=st["mean"], stderr=st["stderr"], ci_lo=st["ci_lo"],
                            ci_hi=st["ci_hi"], p_raw=st["p_raw"],
                            n_images=st["n_images"], n_aoi=st["n_aoi"],
                            ci_excludes_point=st["ci_excludes_point"],
                            results_08_seed0_delta=r08_s0,
                            sib_cell_changed_from_results_08=(a == "oglanet")))
    padj = E.holm([r["p_raw"] for r in t_i])
    for r, p in zip(t_i, padj):
        r["p_holm_12"] = float(p)
        r["significant_holm"] = bool(p < ALPHA)
        r["direction"] = "sib_above_vanilla" if r["delta"] > 0 else "sib_below_vanilla"

    # the same seed-0 family with the SUPERSEDED cross-source OGLANet cells, same
    # code and stream labels, so the predecessor column is like-for-like
    t_i_old = []
    for r in t_i:
        a, f = r["arch"], r["fold"]
        if a == "oglanet":
            O = E.cell(old_recs[f])
            V, _ = van(a, f, 0)
            d = (O.miou2_pct - V.miou2_pct).to_numpy(np.float64)
            st = E.paired_stats({0: d}, f, f"t34|s0|sib|{a}|{f}|2class")
            if abs(st["mean"] - r["results_08_seed0_delta"]) > 1e-6:
                fail(f"oglanet/{f}: superseded seed-0 delta does not reproduce "
                     "results/08's per_seed value")
            t_i_old.append(dict(arch=a, fold=f, sib_run_id=old_recs[f]["run_id"],
                                sib_level=float(O.miou2_pct.mean()),
                                delta=st["mean"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
                                p_raw=st["p_raw"]))
            lev_s0[("sib_old", a, f)] = float(O.miou2_pct.mean())
        else:
            t_i_old.append(dict(arch=a, fold=f, sib_run_id=r["sib_run_id"],
                                sib_level=r["sib_level"], delta=r["delta"],
                                ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], p_raw=r["p_raw"]))
    padj_old = E.holm([r["p_raw"] for r in t_i_old])
    for r, p in zip(t_i_old, padj_old):
        r["p_holm_12"] = float(p)

    # worst cells, seed 0
    def worst_table(levels, sib_key):
        out = []
        scopes = [("pooled: 4 archs", ARCHS), ("pooled: mamnet+dinov3", ["mamnet", "dinov3"])] \
            + [(f"per-arch: {a}", [a]) for a in ARCHS]
        for scope, archs in scopes:
            sc = {(a, f): levels[(sib_key if a == "oglanet" else "sib", a, f)]
                  for a in archs for f in FOLDS}
            vc = {(a, f): levels[("vanilla", a, f)] for a in archs for f in FOLDS}
            sk, vk = min(sc, key=sc.get), min(vc, key=vc.get)
            rec = dict(scope=scope, n_cells=len(sc),
                       sib_worst_cell=f"{sk[0]}/{sk[1].upper()}/s0", sib_worst=sc[sk],
                       vanilla_worst_cell=f"{vk[0]}/{vk[1].upper()}/s0",
                       vanilla_worst=vc[vk], difference_sib_minus_vanilla=sc[sk] - vc[vk],
                       worst_cells_same=(sk == vk))
            out.append(rec)
        return out

    wc_i = worst_table(lev_s0, "sib")
    wc_i_old = worst_table(lev_s0, "sib_old")
    x08 = next(w for w in r08["task2"]["sib_pooled_worst_case"]
               if w["unit"] == "s0 only" and w["scope"] == "pooled: 4 archs")
    if abs(wc_i_old[0]["difference_sib_minus_vanilla"] - x08["difference"]) > 1e-6:
        fail("superseded seed-0 pooled worst case does not reproduce results/08")

    # supplementary: AOI-cluster bootstrap of the difference of the two minima
    # over the 12 seed-0 cells (AOIs resampled within each fold, one draw per fold
    # shared by the 24 cells of that fold). Not part of results/08's protocol.
    rng = np.random.default_rng([0, E.stream_of("t34|wcdiff|s0")])
    per_fold = {}
    for f in FOLDS:
        codes, n_aoi = E.FOLDREF[f]["codes"], E.FOLDREF[f]["n_aoi"]
        cnt = np.bincount(codes, minlength=n_aoi).astype(np.float64)
        sums_s, sums_v = [], []
        for a in ARCHS:
            S, *_ = sib_s0(a, f)
            V, _ = van(a, f, 0)
            sums_s.append(np.bincount(codes, weights=S.miou2_pct.to_numpy(np.float64),
                                      minlength=n_aoi))
            sums_v.append(np.bincount(codes, weights=V.miou2_pct.to_numpy(np.float64),
                                      minlength=n_aoi))
        per_fold[f] = (np.vstack(sums_s), np.vstack(sums_v), cnt, n_aoi)
    reps = np.empty(B)
    smaller = 0
    for b in range(B):
        ms, mv = [], []
        for f in FOLDS:
            ss, sv, cnt, n_aoi = per_fold[f]
            ix = rng.integers(0, n_aoi, size=n_aoi)
            den = cnt[ix].sum()
            ms.append((ss[:, ix].sum(1) / den).min())
            mv.append((sv[:, ix].sum(1) / den).min())
        reps[b] = min(ms) - min(mv)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    wc_boot = dict(
        statistic="min over the 12 seed-0 SIB cell means minus min over the 12 "
                  "seed-0 Vanilla cell means",
        point=wc_i[0]["difference_sib_minus_vanilla"], ci95=[float(lo), float(hi)],
        bootstrap_median=float(np.median(reps)),
        share_sib_worst_above_vanilla_worst=float((reps > 0).mean()),
        p_two_tailed=float(min(1.0, max(1.0 / B, 2 * min((reps <= 0).mean(),
                                                          (reps >= 0).mean())))),
        B=B, method=("AOI cluster bootstrap within fold, one draw per fold applied "
                     "to all 8 cells of that fold; both minima recomputed per "
                     "replicate; percentile CI; difference of maxima/minima is "
                     "skewed and carries no bias correction"),
        status="SUPPLEMENTARY: not part of results/08's protocol, which gives a "
               "paired CI only when the two worst cells coincide")

    # ================= (ii) CONTEXT: mixed-seed =================
    t_ii = []
    for a in ARCHS:
        for f in FOLDS:
            base = r08_t2[(a, f)]
            if a != "oglanet":
                dbs = {}
                for s in (0, 1, 2):
                    rec = E.find(rows, a, "sib", "loco", f, s)
                    M = E.cell(rec)
                    V, vrid = van(a, f, s)
                    used.update([rec["run_id"], vrid])
                    dbs[s] = (M.miou2_pct - V.miou2_pct).to_numpy(np.float64)
                st = E.paired_stats(dbs, f, f"t2|sib|{a}|{f}|2class")
                for k08, kme in (("mean", "mean"), ("ci_lo", "ci_lo"),
                                 ("ci_hi", "ci_hi"), ("p_raw", "p_raw")):
                    if abs(st[kme] - base[k08]) > 1e-6:
                        fail(f"(ii) {a}/{f}: {k08} {st[kme]} != results/08 {base[k08]}")
                t_ii.append(dict(arch=a, fold=f, sib_seeds=[0, 1, 2],
                                 vanilla_seeds=[0, 1, 2],
                                 sib_level=base["method_level"],
                                 vanilla_level=base["vanilla_level"],
                                 delta=st["mean"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
                                 p_raw=st["p_raw"], source="results/08, recomputed and "
                                 "asserted to 1e-6", mixed_seed=False))
            else:
                S, srid, sacc, _ = sib_s0(a, f)
                Vs = []
                for s in (0, 1, 2):
                    V, vrid = van(a, f, s)
                    used.add(vrid)
                    Vs.append(V.miou2_pct.to_numpy(np.float64))
                vbar = np.mean(Vs, axis=0)
                d = S.miou2_pct.to_numpy(np.float64) - vbar
                st = E.paired_stats({0: d}, f, f"t34|mixed|sib|{a}|{f}|2class")
                vl = float(np.mean([v.mean() for v in Vs]))
                if abs(vl - base["vanilla_level"]) > 1e-6:
                    fail(f"(ii) oglanet/{f}: vanilla seed-mean level does not "
                         "reproduce results/08")
                t_ii.append(dict(arch=a, fold=f, sib_seeds=[0], vanilla_seeds=[0, 1, 2],
                                 sib_run_id=srid, sib_level=float(S.miou2_pct.mean()),
                                 vanilla_level=vl, delta=st["mean"], ci_lo=st["ci_lo"],
                                 ci_hi=st["ci_hi"], p_raw=st["p_raw"],
                                 results_08_seed_mean_delta=base["mean"],
                                 results_08_sib_level=base["method_level"],
                                 source="MIXED-SEED: SIB seed-0 tgtfda cell minus the "
                                        "per-image 3-seed Vanilla mean",
                                 mixed_seed=True))
    padj2 = E.holm([r["p_raw"] for r in t_ii])
    for r, p in zip(t_ii, padj2):
        r["p_holm_12_mixed_family"] = float(p)
        r["significant_holm"] = bool(p < ALPHA)
        r["results_08_p_holm"] = r08_t2[(r["arch"], r["fold"])]["p_holm"]
    lev_ii = {}
    for r in t_ii:
        lev_ii[("sib", r["arch"], r["fold"])] = r["sib_level"]
        lev_ii[("vanilla", r["arch"], r["fold"])] = r["vanilla_level"]
    wc_ii = worst_table(lev_ii, "sib")
    for w in wc_ii:
        w["sib_worst_cell"] = w["sib_worst_cell"].replace(
            "/s0", "/s0" if w["sib_worst_cell"].startswith("oglanet") else "/mean")
        w["vanilla_worst_cell"] = w["vanilla_worst_cell"].replace("/s0", "/mean")
    x08m = next(w for w in r08["task2"]["sib_pooled_worst_case"]
                if w["unit"] == "arch x fold, seed-mean level"
                and w["scope"] == "pooled: 4 archs")

    # ---- report ----
    fmt = lambda v: f"{v:+.4f}"
    print("\n(i) HEADLINE, seed 0: SIB minus Vanilla, 2-class mIoU K=2 unseen-excl.")
    for r in t_i:
        o = next(x for x in t_i_old if x["arch"] == r["arch"] and x["fold"] == r["fold"])
        print(f"  {r['arch']:10s} {r['fold']:5s} SIB {r['sib_level']:.4f} Van "
              f"{r['vanilla_level']:.4f} delta {fmt(r['delta'])} "
              f"[{fmt(r['ci_lo'])},{fmt(r['ci_hi'])}] p {r['p_raw']:.4f} Holm "
              f"{r['p_holm_12']:.4f}   (r08 s0 {fmt(o['delta'])}, Holm-12 "
              f"{o['p_holm_12']:.4f})")
    for w, wo in zip(wc_i, wc_i_old):
        print(f"  WORST {w['scope']:22s} SIB {w['sib_worst_cell']} {w['sib_worst']:.4f}"
              f" vs Van {w['vanilla_worst_cell']} {w['vanilla_worst']:.4f} diff "
              f"{fmt(w['difference_sib_minus_vanilla'])}   (superseded: "
              f"{wo['sib_worst_cell']} {fmt(wo['difference_sib_minus_vanilla'])})")
    print(f"  supplementary worst-case diff CI {wc_boot['ci95']} p {wc_boot['p_two_tailed']}")
    print("\n(ii) CONTEXT, MIXED-SEED grid")
    for r in t_ii:
        print(f"  {r['arch']:10s} {r['fold']:5s} SIB {r['sib_level']:.4f} Van "
              f"{r['vanilla_level']:.4f} delta {fmt(r['delta'])} "
              f"[{fmt(r['ci_lo'])},{fmt(r['ci_hi'])}] p {r['p_raw']:.4f} Holm "
              f"{r['p_holm_12_mixed_family']:.4f} {'MIXED' if r['mixed_seed'] else ''}")
    for w in wc_ii:
        print(f"  WORST {w['scope']:22s} SIB {w['sib_worst_cell']} {w['sib_worst']:.4f}"
              f" vs Van {w['vanilla_worst_cell']} {w['vanilla_worst']:.4f} diff "
              f"{fmt(w['difference_sib_minus_vanilla'])}")

    # ---- index, caveats, JSON ----
    os.makedirs(INV, exist_ok=True)
    if os.path.exists(INDEX_OUT) or os.path.exists(OUT):
        fail("results/34 or its index already exists; this job writes new files only")
    idx.to_parquet(INDEX_OUT, index=False)
    index_sha = sha256_file(INDEX_OUT)

    n_neg = sum(1 for r in t_i if r["significant_holm"] and r["delta"] < 0)
    n_pos = sum(1 for r in t_i if r["significant_holm"] and r["delta"] > 0)
    w0 = wc_i[0]
    headline = (
        f"On S-EO LOCO at seed 0 for all four architectures, with OGLANet-SIB rerun on "
        f"the held-out-city FDA pool, SIB minus Vanilla in two-class mIoU at K=2 "
        f"(DSM-unseen excluded) ranges from {min(r['delta'] for r in t_i):+.2f} to "
        f"{max(r['delta'] for r in t_i):+.2f} over the 12 (architecture, city) cells, "
        f"{n_neg} of 12 Holm-significant below Vanilla and {n_pos} above, and SIB's "
        f"worst cell ({w0['sib_worst_cell']}, {w0['sib_worst']:.2f}) sits "
        f"{w0['difference_sib_minus_vanilla']:+.2f} mIoU from Vanilla's worst cell "
        f"({w0['vanilla_worst_cell']}, {w0['vanilla_worst']:.2f}).")
    carry_ids = [0, 6, 7, 8, 9, 10, 11, 14, 15, 20, 22]
    carried = [f"Carried from results/08, caveat {i}: {r08['caveats'][i]}"
               for i in carry_ids]
    seed_caveats = [f"Carried from results/08, caveat {i} (applies to table (ii) only): "
                    f"{r08['caveats'][i]}" for i in (3, 4)]
    spans = {f"{a}/{f}": (max(r08_t2[(a, f)]["per_seed"].values())
                          - min(r08_t2[(a, f)]["per_seed"].values()))
             for a in ARCHS for f in FOLDS}
    max_span_row = max(spans, key=spans.get)
    max_span = spans[max_span_row]
    med_ci_w = float(np.median([r["ci_hi"] - r["ci_lo"] for r in t_i]))
    epochs = {f: (new[f][1].get("epochs_run"), new[f][1].get("best_epoch"),
                  new[f][1].get("stopped_on")) for f in FOLDS}
    caveats = [
        "SINGLE SEED in table (i). Every cell in the headline table is one training "
        "run at seed 0, for SIB and for Vanilla, on all four architectures. The AOI "
        "cluster bootstrap and the sign-flip p cover test-patch and AOI sampling only; "
        "no seed variation enters any interval. On results/08's 3-seed grid the "
        "per-seed SIB-minus-Vanilla deltas of one (architecture, city) row span up "
        f"to {max_span:.2f} mIoU ({max_span_row}); the median seed-0 CI width in "
        f"table (i) is {med_ci_w:.2f}.",
        "MIXED SEEDS in table (ii). Its three OGLANet rows set a seed-0 SIB cell "
        "against a 3-seed Vanilla mean, while its nine other rows are 3-seed against "
        "3-seed. The two kinds of row are not the same estimand, their intervals "
        "cover different variance sources, and the Holm family over 12 mixes them. "
        "Table (ii) is context and is never the headline.",
        "The three new OGLANet-SIB cells draw their FDA style pool from 100 patches "
        "selected from the held-out city's 3,500 unlabeled train candidates "
        "(fda_donor heldout_city_train_patches; donor assertions all zero: no test, "
        "train or val overlap, RGB only, mask and auxiliary roots excluded). The "
        "superseded cells drew from the source cities, cross-paired, 200. OGLANet-SIB "
        "is therefore unlabeled_target here, not source_only.",
        "The new cells' run_id carries a -tgtfda suffix "
        "(oglanet-sib-loco-r3500-{fold}-hr-s0-tgtfda) and requested_run_id records "
        "the operator's oglanet-sibtgt-... name; the operator ruled no rename "
        "(session log 2026-11-02 section 13). They live in dumps/seo/, not in "
        "results/08's delta_dumps/ tree, and are read by path through this job's own "
        "index.",
        "The new cells and their superseded counterparts differ in training as well "
        f"as in donor pool: new epochs_run/best/stopped_on {epochs}; the superseded "
        "cells' wall clock in the session log is a dump-only re-score and not "
        "comparable. Each new cell is one OGLANet inference draw (E-9).",
        "OGLANet inference depends on batch size and RNG state "
        "(results/30_oglanet_rng_dependence.json, measured on internal checkpoints); "
        "no repeat-inference band exists for any of the three new S-EO cells.",
        "Holm families here: table (i) over its 12 seed-0 cells; table (ii) over its "
        "12 mixed rows; the predecessor column in table (i) over the 12 seed-0 cells "
        "with the superseded OGLANet cells. results/08's task 2 Holm family (12, seed-"
        "averaged) is reported beside table (ii) as results_08_p_holm.",
        "The pooled worst case pools across access classes: MAMNet-SIB and "
        "OGLANet-SIB unlabeled_target, DINOv3- and CrossEarth-SIB source_only. The "
        "per-architecture rows are given beside it.",
        "The supplementary bootstrap of the worst-case difference is not in results/08's "
        "protocol; it is a difference of two minima, skewed, with no bias correction.",
    ] + seed_caveats + carried
    stale = [dict(index=len(caveats) - len(carried) + carry_ids.index(14),
                  correction="results/34: OGLANet-SIB is unlabeled_target in this JSON "
                             "(held-out-city FDA pool); results/08's caveat lists it as "
                             "source_only, which described the superseded cells.")]
    out = dict(
        id="34_seo_sib_family_tgtfda", arm="A", answers=r08["answers"],
        question=("On S-EO, with OGLANet-SIB rerun on the held-out-city FDA pool, how "
                  "does SIB compare with Vanilla per (architecture, city) at seed 0, "
                  "and how does SIB's worst cell compare with Vanilla's?"),
        supersedes_in_part=dict(
            file="results/08_seo_full_grid.json",
            what="task2's SIB rows and sib_pooled_worst_case for OGLANet only; every "
                 "other part of results/08 stands and is not recomputed here"),
        new_directories=six,
        new_cell_checks=new_checks,
        value_field=("per-patch two-class mIoU at K=2, DSM-unseen excluded: "
                     "50*(tp/(tp+fp+fn+1e-10)+tn/(tn+fp+fn+1e-10)) from tp_tol/fp_tol/"
                     "fn_tol/tn_tol, asserted equal to miou_tol_pct (1e-9) and to run.json "
                     "reporting_metric_value_pct (stored at 4 dp, 5e-5); results/08's "
                     "gap_2class_tol_k2_unseen_excluded definition"),
        reuse=dict(module="analysis/08_seo_full_grid.py",
                   functions=["catalogue", "build_foldref", "cell", "add_metrics",
                              "paired_stats", "holm", "find", "stream_of"],
                   checks=["nine unchanged seed-mean rows reproduce results/08 "
                           "mean/ci/p to 1e-6", "nine unchanged seed-0 deltas reproduce "
                           "results/08 per_seed to 1e-6", "superseded OGLANet seed-0 "
                           "deltas and the superseded s0 pooled worst case reproduce "
                           "results/08 to 1e-6"]),
        table_i_headline_seed0=dict(
            rows=t_i, holm_family="12 seed-0 (arch, city) cells",
            worst_case=wc_i, worst_case_difference_bootstrap_supplementary=wc_boot,
            predecessor_seed0_with_superseded_oglanet_cells=dict(
                rows=t_i_old, worst_case=wc_i_old,
                results_08_s0_pooled_worst=dict(sib_worst_cell=x08["sib_worst_cell"],
                                                difference=x08["difference"]))),
        table_ii_context_mixed_seed=dict(
            label="MIXED-SEED CONTEXT. Never the headline.",
            rows=t_ii, worst_case=wc_ii,
            results_08_seed_mean_pooled_worst=dict(
                sib_worst_cell=x08m["sib_worst_cell"], sib_worst=x08m["sib_worst"],
                vanilla_worst_cell=x08m["vanilla_worst_cell"],
                difference=x08m["difference"])),
        cross_tree_duplicates=cross_tree,
        runs=sorted(used),
        index_path=os.path.relpath(INDEX_OUT, BASE), index_sha256=index_sha,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/34_seo_sib_family_tgtfda.py",
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host",
        seed=0, B=B,
        seed_note="every bootstrap uses results/08's rng: default_rng([0, stream]) with "
                  "the stream derived from a label; new labels are prefixed t34|",
        headline=headline,
        caveats=caveats,
        caveats_stale_for_oglanet_sib=stale,
        figure=None, caption=None)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            o = float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, float) and not np.isfinite(o):
            fail(f"non-finite value in output: {o}")
        return o

    with open(OUT + ".new", "w") as fh:
        json.dump(clean(out), fh, indent=1, allow_nan=False)
    os.replace(OUT + ".new", OUT)
    print(f"\nwrote {OUT}  sha256 {sha256_file(OUT)}")
    print(f"wrote {INDEX_OUT}  sha256 {index_sha}")
    print("HEADLINE:", headline)


if __name__ == "__main__":
    main()
