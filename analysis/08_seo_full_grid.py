#!/usr/bin/env python
"""
Arm A -- the EXTENDED analysis over the full S-EO grid.

Supersedes results/02_seo_loco.json and results/03_seo_methods.json, which were
computed on a 1-seed, 2-architecture grid. Same decision variables, same
pre-committed rows, larger n. Writes results/08_seo_full_grid.json.

Ground truth: the dumps pulled to delta_dumps/ (nine roots) and the Delta-side
inventory JSONs in delta_logs/. Where PAPER_STATE.md's arm A rows conflict with
the dumps, the dumps win.

  DRY RUN  every .DONE dump in the pulled tree is catalogued; the S-EO cells are
           asserted (schema, row count == fold n_test, image_id identity within
           fold, target_access vs the access manifest, metric_module_sha
           identity, count/metric recompute). Prints the assertion table and the
           (arch, method, condition, seed) census. Writes nothing.
  FULL     tasks 1-6 below; writes delta_dumps/INDEX.parquet and the results
           JSON.

  TASK 1  vanilla gap, 4 archs x 3 folds x 3 seeds = 36 pairs
  TASK 2  method table vs vanilla-LOCO; SIB pooled worst case; CrossEarth block
  TASK 3  Exp A external (fg-only, matched against the parent's sanity floor)
  TASK 4  per-AOI gap ~ sun_dist + C(city) + C(arch)
  TASK 5  section 4.2 histogram-matching count from the probe dumps
          (p-bar_S is NOT computed here; see the task 5 block)
  TASK 6  per-cell descriptives

No branch is applied in prose. Rows are reported by their recorded designation.
Nothing AURC-shaped is read or written. No external magnitude is compared to an
internal one.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/08_seo_full_grid.py
"""

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

BASE = os.environ["PROJECT_ROOT"]
TREE = os.path.join(BASE, "delta_dumps")
LOGS = os.path.join(BASE, "delta_logs")
INVENTORY = os.path.join(LOGS, "armA_b4_probe_inventory_final.json")
T2_ROSTER = os.path.join(LOGS, "armA_b5_t2_roster.json")
INDEX_PATH = os.path.join(TREE, "INDEX.parquet")
RESULTS_PATH = os.path.join(BASE, "results", "08_seo_full_grid.json")
PRIOR_02 = os.path.join(BASE, "results", "02_seo_loco.json")
PRIOR_03 = os.path.join(BASE, "results", "03_seo_methods.json")
METRICS_PATH = os.path.join(os.environ.get("ST_METRICS_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))), "shadowtransfer", "metrics.py")

ARCHS = ["mamnet", "dinov3", "oglanet", "crossearth"]
FOLDS = ["oma", "jax", "ucsd"]
SEEDS = [0, 1, 2]
N_TEST = {"oma": 1637, "jax": 2642, "ucsd": 1870}      # a FOLD property

METHODS = ["sib", "fda", "fada", "iim", "mrfp_plus", "segdesic", "isw", "bsm", "sar"]
CROSSEARTH_METHODS = ["vanilla", "sib", "fda", "bsm", "sar"]

B = 10_000
SEED = 0                       # every rng below is default_rng([SEED, stream])
EPS = 1e-10                    # metrics.py _img_metrics epsilon, matched exactly

# The access manifest every cell is checked against. Source:
# summaries/target_access_audit.md (arm I, v3) for every method, with SIB's row
# per architecture as the S-EO sidecars record it: MAMNet-SIB carries FDA with a
# held-out-city target pool (unlabeled_target); DINOv3- and CrossEarth-SIB carry
# neither FDA nor SAG (source_only); OGLANet-SIB's FDA donor is cross-SOURCE
# (summaries/session_logs/2026-08-03-oglanet-sib-access.md), so source_only.
# expa is the section 4.4 component-retraining experiment (target_labels).
EXPECTED_ACCESS = {
    "vanilla": "source_only", "none": "target_labels", "fda": "unlabeled_target",
    "segdesic": "unlabeled_target", "iim": "source_only", "isw": "source_only",
    "mrfp_plus": "source_only", "fada": "source_only", "bsm": "source_only",
    "sar": "deployment_time", "expa": "target_labels",
}
EXPECTED_ACCESS_SIB = {"mamnet": "unlabeled_target", "dinov3": "source_only",
                       "oglanet": "source_only", "crossearth": "source_only"}

# the arm-A per-image column set every standard cell must carry, with dtypes
REQUIRED_NAMED = ["aoi_id", "sun_elevation", "sun_dist", "n_valid_px",
                  "tp_tol", "fp_tol", "fn_tol", "tn_tol",        # K=2
                  "tp_str", "fp_str", "fn_str", "tn_str",        # K=0
                  "tp_k5", "fp_k5", "fn_k5", "tn_k5"]            # K=5

SUPERSEDED_SUFFIX = re.compile(r"\.(truncated|unconverged|pre_w2g)$")
RUNID_RE = re.compile(
    r"^(?P<arch>[a-z0-9]+)-(?P<method>[a-z_0-9]+)-(?P<cond>loco|upper)-r(?P<rung>\d+)-"
    r"(?P<fold>[a-z]+)-(?P<res>hr|mr)-s(?P<seed>\d+)(?P<suffix>(-[a-z0-9]+)*)$")


def fail(msg):
    raise SystemExit(f"ASSERTION FAILED: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def expected_access(arch, method):
    if method == "sib":
        return EXPECTED_ACCESS_SIB[arch]
    return EXPECTED_ACCESS.get(method)


# ----------------------------------------------------------------------------
# catalogue -- the walk that BUILDS the index. Cells are selected from the
# catalogue afterwards, never from a glob.
# ----------------------------------------------------------------------------

def catalogue():
    rows = []
    for dp, dn, fn in os.walk(TREE):
        dn.sort()
        rel = os.path.relpath(dp, TREE)
        name = os.path.basename(dp)
        has_pq = "perimage.parquet" in fn
        if SUPERSEDED_SUFFIX.search(name):
            rows.append(dict(dir=rel, run_id_dir=name, category="superseded_snapshot",
                             excluded_reason=f"superseded snapshot ({SUPERSEDED_SUFFIX.search(name).group(0)}); "
                                             f"{'archive stub, no parquet' if not has_pq else 'has parquet'}",
                             has_parquet=has_pq))
            continue
        if not has_pq:
            if ".FAILED" in fn:
                rows.append(dict(dir=rel, run_id_dir=name, category="failed_no_dump",
                                 excluded_reason=".FAILED marker, no perimage.parquet",
                                 has_parquet=False))
            continue
        r = dict(dir=rel, run_id_dir=name, has_parquet=True,
                 has_run_json="run.json" in fn, done=".DONE" in fn, failed=".FAILED" in fn,
                 root=rel.split(os.sep)[0])
        meta = {}
        if r["has_run_json"]:
            with open(os.path.join(dp, "run.json")) as fh:
                meta = json.load(fh)
        r["meta"] = meta
        r["run_id"] = meta.get("run_id")
        r["dataset"] = meta.get("dataset")
        r["arch"] = meta.get("arch")
        r["method"] = meta.get("method")
        r["condition"] = meta.get("mode", meta.get("condition"))
        r["holdout"] = str(meta.get("holdout", "")).lower() or None
        r["seed"] = meta.get("seed")
        r["smoke"] = meta.get("smoke")
        r["target_access"] = meta.get("target_access")
        r["metric_module_sha"] = meta.get("metric_module_sha")
        r["config_sha256"] = meta.get("config_sha256")
        m = RUNID_RE.match(name)
        r["suffix"] = m.group("suffix") if m else None
        # category
        if not r["done"] or r["failed"] or not r["has_run_json"]:
            cat, why = "failed_or_incomplete", (
                ".FAILED marker" if r["failed"] else "no .DONE" if not r["done"] else "no run.json")
        elif r["dataset"] != "seo":
            cat, why = "non_seo", f"dataset={r['dataset']} (not S-EO; {rel.split(os.sep)[1]})"
        elif meta.get("smoke") is True:
            cat, why = "smoke", "run.json smoke=true"
        elif name.endswith("-p1histmatch"):
            cat, why = "probe", "probe dump (task 5 material, not a cell)"
        elif "dumps_x200" in rel:
            cat, why = "x200_duplicate", "dumps_x200 duplicate; dumps/seo is the canonical parent"
        elif r["method"] == "sar":
            cat, why = "sar", None
        elif r["method"] == "expa":
            cat, why = "expa", None
        elif r["suffix"]:
            cat, why = "retrain_suffix", f"D-A2-1 from-scratch retrain ({r['suffix']}); primary uses the inventory-named cell"
        else:
            cat, why = "cell", None
        r["category"], r["excluded_reason"] = cat, why
        rows.append(r)
    return rows


# ----------------------------------------------------------------------------
# per-image metrics, recomputed from raw counts
# ----------------------------------------------------------------------------

def miou2(tp, fp, fn, tn):
    """two-class mIoU in percent, matching metrics.py _img_metrics exactly."""
    return 50.0 * (tp / (tp + fp + fn + EPS) + tn / (tn + fp + fn + EPS))


def fgiou(tp, fp, fn):
    return 100.0 * tp / (tp + fp + fn + EPS)


def add_metrics(df, sfx=""):
    c = lambda k: df[f"{k}{sfx}"].to_numpy(np.float64)
    tp, fp, fn, tn = c("tp_tol"), c("fp_tol"), c("fn_tol"), c("tn_tol")
    df = df.copy()
    df["miou2_pct" + sfx] = miou2(tp, fp, fn, tn)
    df["fg_pct" + sfx] = fgiou(tp, fp, fn)
    return df


# ----------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------

def load_parquet(rec):
    return pd.read_parquet(os.path.join(TREE, rec["dir"], "perimage.parquet"))


def check_standard(rec, df, ref_dtypes, checks):
    """Assertions for a cell in the arm-A per-image schema (cells, e200, expa,
    probes). Appends (name, ok, value)."""
    fold = rec["holdout"]
    meta = rec["meta"]
    missing = [k for k in ref_dtypes.index if k not in df.columns]
    checks.append(("schema: arm-A column set present", not missing,
                   f"missing {missing[:6]}" if missing else f"{len(ref_dtypes)} cols"))
    if not missing:
        drift = [k for k in ref_dtypes.index if df[k].dtype != ref_dtypes[k]]
        checks.append(("schema: dtypes match reference", not drift,
                       f"drift {[(k, str(df[k].dtype)) for k in drift[:4]]}" if drift else "ok"))
    named = [k for k in REQUIRED_NAMED if k not in df.columns]
    checks.append(("schema: aoi/sun/n_valid/K0-2-5 counts", not named, named or "ok"))
    want = N_TEST[fold]
    checks.append((f"nrow == n_test[{fold}]={want}", len(df) == want, len(df)))
    for key in ("n_test", "n_test_expected"):
        if key in meta:
            checks.append((f"run.json {key} == {want}", meta[key] == want, meta[key]))
    checks.append(("image_id unique", not df.image_id.duplicated().any(), "ok"))
    if missing or named:
        return
    checks.append(("aoi_id non-null", int(df.aoi_id.isna().sum()) == 0, int(df.aoi_id.isna().sum())))
    checks.append(("sun_elevation/sun_dist non-null",
                   int(df.sun_elevation.isna().sum() + df.sun_dist.isna().sum()) == 0, "ok"))
    s4 = df.tp_tol + df.fp_tol + df.fn_tol + df.tn_tol
    checks.append(("tp+fp+fn+tn == n_valid_px (K=2)", bool((s4 == df.n_valid_px).all()), "ok"))
    zero = df.n_valid_px <= 0
    checks.append(("n_valid_px==0 rows fully DSM-unseen",
                   bool((df.loc[zero, "unseen_frac"] == 1.0).all()), int(zero.sum())))
    d = add_metrics(df)
    dm = float(np.abs(d.miou2_pct - df.miou_tol_pct).max())
    df_ = float(np.abs(d.fg_pct - df.iou_tol * 100.0).max())
    checks.append(("recompute == miou_tol_pct", dm < 1e-9, f"{dm:.1e}"))
    checks.append(("recompute == iou_tol", df_ < 1e-9, f"{df_:.1e}"))
    checks.append(("run_id col == dir", df.run_id.nunique() == 1 and df.run_id.iloc[0] == rec["run_id_dir"],
                   df.run_id.iloc[0]))
    checks.append(("city == holdout", set(df.city.str.lower().unique()) == {fold},
                   list(df.city.unique())))


def check_sar(rec, df, checks):
    fold = rec["holdout"]
    want = N_TEST[fold]
    missing = [k for k in REQUIRED_NAMED if k not in df.columns]
    checks.append(("schema: aoi/sun/n_valid/K0-2-5 counts", not missing, f"missing {missing}"))
    g = df.groupby(["pass_id", "step"]).image_id.agg(["size", "nunique"])
    checks.append((f"rows per (pass, step) == n_test[{fold}]={want}",
                   bool((g["size"] == want).all() and (g["nunique"] == want).all()),
                   f"{len(g)} (pass,step) groups"))
    s4 = df.tp_k2 + df.fp_k2 + df.fn_k2 + df.tn_k2
    checks.append(("tp+fp+fn+tn == n_valid_px_k2", bool((s4 == df.n_valid_px_k2).all()), "ok"))
    checks.append(("metric_module_sha present", rec["metric_module_sha"] is not None,
                   rec["metric_module_sha"]))


def dry_run(rows):
    print("=" * 110)
    print("ARM A / S-EO FULL GRID -- DRY RUN")
    print(f"tree: {TREE}")
    print("=" * 110)

    cat = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"} for r in rows])
    n_pq = int(cat.has_parquet.sum())
    print(f"\ncatalogue: {len(cat)} entries, {n_pq} perimage.parquet")
    print(cat.groupby(["category"]).size().to_string())

    # reference schema: the canonical vanilla-LOCO MAMNet/OMA/s0 cell (the 02 grid)
    ref_rec = next(r for r in rows if r.get("run_id_dir") == "mamnet-vanilla-loco-r3500-oma-hr-s0")
    ref_df = load_parquet(ref_rec)
    ref_dtypes = ref_df.dtypes
    print(f"\nreference schema: {ref_rec['dir']} ({len(ref_dtypes)} columns)")

    in_scope = [r for r in rows if r.get("category") in
                ("cell", "retrain_suffix", "expa", "probe", "sar")]
    table, fold_ids, fold_aoi, fold_unseen = [], defaultdict(dict), defaultdict(dict), defaultdict(dict)
    shas, cfg = Counter(), {}
    for r in in_scope:
        checks = []
        df = load_parquet(r)
        if r["category"] == "sar":
            check_sar(r, df, checks)
            ids = set(df.image_id)
        else:
            check_standard(r, df, ref_dtypes, checks)
            ids = set(df.image_id)
            if "aoi_id" in df:
                fold_aoi[r["holdout"]][r["run_id_dir"]] = df.set_index("image_id").aoi_id.sort_index()
                fold_unseen[r["holdout"]][r["run_id_dir"]] = frozenset(df.loc[df.n_valid_px <= 0, "image_id"])
        fold_ids[r["holdout"]][r["run_id_dir"]] = frozenset(ids)
        ta, ex = r["target_access"], expected_access(r["arch"], r["method"])
        if r["category"] == "probe":
            # a probe trains nothing; its own sidecar access describes the
            # matching statistic. Checked against the PARENT's manifest class
            # separately below, never silently.
            checks.append(("target_access present", ta is not None, ta))
        else:
            checks.append((f"target_access == manifest ({ex})", ta == ex, ta))
        checks.append(("run_id == dir name", r["run_id"] == r["run_id_dir"], r["run_id"]))
        checks.append(("smoke is not true", r["smoke"] is not True, r["smoke"]))
        if r["metric_module_sha"]:
            shas[r["metric_module_sha"]] += 1
        if r["run_id"] in cfg and cfg[r["run_id"]] != r["config_sha256"]:
            checks.append(("no duplicate run_id w/ differing config_sha256", False, r["run_id"]))
        cfg[r["run_id"]] = r["config_sha256"]
        bad = [c for c in checks if not c[1]]
        table.append(dict(category=r["category"], run_id=r["run_id_dir"], root=r["root"],
                          nrow=len(df), ncol=df.shape[1], target_access=ta,
                          n_checks=len(checks), n_failed=len(bad),
                          failed="; ".join(f"{c[0]} -> {c[2]}" for c in bad)))
    tab = pd.DataFrame(table)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 140)
    pd.set_option("display.max_rows", 1000)
    print("\n--- per-cell assertions, summarised by category ---")
    print(tab.groupby("category").agg(cells=("run_id", "size"), checks=("n_checks", "sum"),
                                      failed_cells=("n_failed", lambda s: int((s > 0).sum()))).to_string())
    fails = tab[tab.n_failed > 0]
    if len(fails):
        print(f"\n--- {len(fails)} cells with failed assertions ---")
        print(fails.groupby(["category", "failed"]).run_id.agg(["size", lambda s: ", ".join(sorted(s)[:3])])
              .to_string())

    # cross-cell assertions
    print("\n--- cross-cell assertions ---")
    x = []
    for fold in FOLDS:
        sets = fold_ids[fold]
        ref = next(iter(sets.values()))
        n_diff = sum(1 for v in sets.values() if v != ref)
        x.append((f"{fold}: image_id set identical across {len(sets)} dumps", n_diff == 0,
                  f"{len(ref)} ids; {n_diff} differ"))
        maps = fold_aoi[fold]
        refm = next(iter(maps.values()))
        n_m = sum(1 for v in maps.values() if not v.equals(refm))
        x.append((f"{fold}: image_id->aoi_id identical across {len(maps)} dumps", n_m == 0,
                  f"{refm.nunique()} AOIs; {n_m} differ"))
        us = fold_unseen[fold]
        refu = next(iter(us.values()))
        n_u = sum(1 for v in us.values() if v != refu)
        x.append((f"{fold}: fully-unseen patch set identical", n_u == 0, f"{len(refu)} patches; {n_u} differ"))
    x.append(("metric_module_sha identical where present", len(shas) == 1,
              {k[:10]: v for k, v in shas.items()}))
    with open(METRICS_PATH, "rb") as fh:
        live = hashlib.sha256(fh.read()).hexdigest()
    x.append(("on-box metrics.py sha == sidecar sha", list(shas)[0] == live if shas else False, live[:10]))
    for name, ok, val in x:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {val}")

    # probe access vs parent manifest class (reported, not a probe assertion)
    pr = tab[tab.category == "probe"].copy()
    pr["parent_class"] = [expected_access(*_arch_method(rid)) for rid in pr.run_id]
    mism = pr[pr.target_access != pr.parent_class]
    print(f"\nprobe sidecar access != parent's manifest class: {len(mism)} of {len(pr)} probes")
    if len(mism):
        print(mism.groupby(["target_access", "parent_class"]).size().to_string())

    # census
    cells = cat[cat.category.isin(["cell", "retrain_suffix", "expa", "sar"])].copy()
    print("\n--- census of non-probe S-EO cells: (arch, method, condition) x seed ---")
    cen = cells.assign(cs=cells.category.map({"cell": "", "retrain_suffix": "+e200", "expa": "", "sar": ""})) \
               .groupby(["arch", "method", "condition", "category", "seed"]).size().unstack(fill_value=0)
    print(cen.to_string())
    print("\n--- probe census: (arch, method) x seed ---")
    pc = cat[cat.category == "probe"].groupby(["arch", "method", "seed"]).size().unstack(fill_value=0)
    print(pc.to_string())

    print("\n--- excluded / out of scope ---")
    exc = cat[~cat.category.isin(["cell", "expa", "sar"])]
    print(exc.groupby(["category", "excluded_reason"], dropna=False).size().to_string())

    n_fail = int((tab.n_failed > 0).sum()) + sum(1 for _, ok, _ in x if not ok)
    print(f"\n{'DRY RUN FAILED' if n_fail else 'DRY RUN OK'}: {n_fail} failing cells/cross-checks. Nothing written.")
    return tab, x


def _arch_method(rid):
    m = RUNID_RE.match(rid)
    return m.group("arch"), m.group("method")


# ----------------------------------------------------------------------------
# bootstrap machinery -- semantics identical to analysis/02_seo_loco.py
# ----------------------------------------------------------------------------

def stream_of(label):
    """Deterministic rng stream from a label, so every interval is reproducible
    and independent of the order the script visits cells in."""
    return int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def _rng(stream):
    return np.random.default_rng([SEED, stream])


def aoi_cluster_bootstrap_imagemean(delta, codes, n_aoi, stream):
    """Resample AOIs with replacement, keep every image of a sampled AOI;
    statistic = image-level mean. With seed as a stratum, `delta` is the
    per-image SEED-AVERAGED delta, which is the same as resampling AOIs and
    averaging over all seeds within each sampled AOI."""
    sums = np.bincount(codes, weights=delta, minlength=n_aoi)
    cnts = np.bincount(codes, minlength=n_aoi).astype(np.float64)
    idx = _rng(stream).integers(0, n_aoi, size=(B, n_aoi))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def aoi_signflip_imagemean(delta, codes, n_aoi, stream):
    """Two-sided sign-flip null at the AOI level, same weighting as the point
    estimate. p = (1 + #{|T*| >= |T_obs|}) / (B + 1)."""
    sums = np.bincount(codes, weights=delta, minlength=n_aoi)
    obs = sums.sum() / float(len(delta))
    signs = _rng(stream).choice(np.array([-1.0, 1.0]), size=(B, n_aoi))
    null = (signs * sums).sum(1) / float(len(delta))
    return float(obs), (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)


def unit_signflip(vals, stream):
    obs = float(vals.mean())
    signs = _rng(stream).choice(np.array([-1.0, 1.0]), size=(B, len(vals)))
    null = (signs * vals).sum(1) / len(vals)
    return obs, (1.0 + int((np.abs(null) >= abs(obs) - 1e-12).sum())) / (B + 1.0)


def ci95(stat):
    lo, hi = np.percentile(stat, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_stats(delta_by_seed, fold, label):
    """delta_by_seed: {seed: per-image delta aligned to the fold reference}.
    Returns the seed-stratified AOI-cluster summary of the seed-averaged delta."""
    codes, n_aoi = FOLDREF[fold]["codes"], FOLDREF[fold]["n_aoi"]
    seeds = sorted(delta_by_seed)
    dbar = np.mean([delta_by_seed[s] for s in seeds], axis=0)
    boot = aoi_cluster_bootstrap_imagemean(dbar, codes, n_aoi, stream_of("boot|" + label))
    obs, p = aoi_signflip_imagemean(dbar, codes, n_aoi, stream_of("flip|" + label))
    lo, hi = ci95(boot)
    per_seed = {int(s): float(delta_by_seed[s].mean()) for s in seeds}
    ps = np.array(list(per_seed.values()))
    return dict(mean=obs, stderr=float(boot.std(ddof=1)), ci_lo=lo, ci_hi=hi, p_raw=p,
                n_seeds=len(seeds), seeds=seeds, per_seed=per_seed,
                seed_sd=float(ps.std(ddof=1)) if len(ps) > 1 else None,
                n_images=int(len(dbar)), n_aoi=int(n_aoi),
                ci_excludes_point=bool(not (lo <= obs <= hi)), dbar=dbar)


def per_aoi_units(dbar, fold):
    ref = FOLDREF[fold]
    return (pd.DataFrame({"aoi_id": ref["aoi"], "gap": dbar, "sun_dist_loco": ref["sun_dist_loco"]})
            .groupby("aoi_id", as_index=False)
            .agg(gap=("gap", "mean"), sun_dist_loco=("sun_dist_loco", "mean"), n_img=("gap", "size")))


def strat_pooled(frame, label, across="arch"):
    """Pooled mean over AOI units, resampling AOIs within fold (02's scheme).
    With across='arch', an AOI is resampled with all its architecture units."""
    rng = _rng(stream_of("pool|" + label))
    num, den = np.zeros(B), 0.0
    for fold in FOLDS:
        sub = frame[frame.fold == fold]
        piv = sub.pivot(index="aoi_id", columns=across, values="gap")
        vals = piv.to_numpy(np.float64)
        n = len(piv)
        idx = rng.integers(0, n, size=(B, n))
        num += vals[idx].sum(axis=(1, 2))
        den += n * vals.shape[1]
    boot = num / den
    vals = frame.gap.to_numpy(np.float64)
    obs, p = unit_signflip(vals, stream_of("poolflip|" + label))
    lo, hi = ci95(boot)
    return dict(mean=obs, stderr=float(boot.std(ddof=1)), ci_lo=lo, ci_hi=hi, p=p,
                n_aoi_units=len(vals), n_images=int(frame.n_img.sum()))


# ----------------------------------------------------------------------------
# cell access
# ----------------------------------------------------------------------------

FOLDREF = {}
_CACHE = {}


def build_foldref(rows):
    for fold in FOLDS:
        rec = find(rows, "mamnet", "vanilla", "loco", fold, 0)
        df = load_parquet(rec).sort_values("image_id").reset_index(drop=True)
        codes, uniq = pd.factorize(df.aoi_id)
        FOLDREF[fold] = dict(image_id=df.image_id.to_numpy(), aoi=df.aoi_id.to_numpy(),
                             codes=codes, n_aoi=len(uniq), city=df.city.iloc[0],
                             sun_dist_loco=None)


def find(rows, arch, method, cond, fold, seed, category="cell"):
    hits = [r for r in rows if r.get("category") == category and r["arch"] == arch
            and r["method"] == method and r["condition"] == cond
            and r["holdout"] == fold and r["seed"] == seed]
    if len(hits) != 1:
        fail(f"{arch}/{method}/{cond}/{fold}/s{seed} [{category}]: {len(hits)} matches")
    return hits[0]


def find_opt(rows, *a, **k):
    try:
        return find(rows, *a, **k)
    except SystemExit:
        return None


def cell(rec):
    """Standard-schema cell, aligned to the fold reference, metrics recomputed."""
    key = rec["dir"]
    if key in _CACHE:
        return _CACHE[key]
    df = load_parquet(rec).sort_values("image_id").reset_index(drop=True)
    fold = rec["holdout"]
    if not np.array_equal(df.image_id.to_numpy(), FOLDREF[fold]["image_id"]):
        fail(f"{rec['run_id_dir']}: image_id order differs from fold reference")
    df = add_metrics(df)
    assert float(np.abs(df.miou2_pct - df.miou_tol_pct).max()) < 1e-9, rec["run_id_dir"]
    assert float(np.abs(df.fg_pct - df.iou_tol * 100.0).max()) < 1e-9, rec["run_id_dir"]
    _CACHE[key] = df
    return df


def sar_cell(rec):
    """SAR: within-dump pair. Returns (unadapted, adapted-final) aligned frames
    with miou2/fg recomputed from the K=2 counts on SAR's OWN population, which
    keeps DSM-unseen pixels (n_valid_px_k2)."""
    key = rec["dir"]
    if key in _CACHE:
        return _CACHE[key]
    df = load_parquet(rec)
    passes = sorted(df.pass_id.unique())
    adapted = [p for p in passes if p != "unadapted"]
    if "unadapted" not in passes or len(adapted) != 1:
        fail(f"{rec['run_id_dir']}: expected unadapted + one SAR pass, got {passes}")
    out = {}
    for tag, pid in (("unadapted", "unadapted"), ("sar", adapted[0])):
        sub = df[df.pass_id == pid]
        sub = sub[sub.step == sub.step.max()].sort_values("image_id").reset_index(drop=True)
        if not np.array_equal(sub.image_id.to_numpy(), FOLDREF[rec["holdout"]]["image_id"]):
            fail(f"{rec['run_id_dir']}/{pid}: image_id set differs from fold reference")
        t, f_, n, tn = (sub[c].to_numpy(np.float64) for c in ("tp_k2", "fp_k2", "fn_k2", "tn_k2"))
        sub = sub.assign(miou2_pct=miou2(t, f_, n, tn), fg_pct=fgiou(t, f_, n))
        d = float(np.abs(sub.fg_pct - 100 * sub.iou_fg_tol2).max())
        assert d < 1e-6, f"{rec['run_id_dir']}: fg recompute vs iou_fg_tol2 {d}"
        out[tag] = sub
    out["pass_id"] = adapted[0]
    out["final_step"] = int(df[df.pass_id == adapted[0]].step.max())
    _CACHE[key] = out
    return out


def e9_band(rec):
    """E-9 band beside a cell: the 10-pass file if present, else the 3-pass one."""
    d = os.path.join(TREE, rec["dir"])
    for fn in ("inference_repeat_n10.json", "inference_repeat.json", "e9_inference_repeat_x3.json"):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            with open(p) as fh:
                j = json.load(fh)
            a = j.get("across_pass_sd", {})
            rng_ = a.get("miou_tol_pct_range")
            return dict(file=fn, n_passes=a.get("n_passes"),
                        width_miou_pct=(float(rng_[1] - rng_[0]) if rng_ else None),
                        sd_miou_pct=a.get("miou_tol_pct_sd"),
                        undersampled_as_gate=(a.get("n_passes") == 3))
    return None


def r6(x):
    return None if x is None else round(float(x), 6)


# ----------------------------------------------------------------------------
# full pass
# ----------------------------------------------------------------------------

def main(rows):
    print("=" * 110)
    print("ARM A / S-EO FULL GRID -- FULL PASS")
    print("=" * 110)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 80)
    fmt = lambda v: f"{v:.4f}"
    out, runs_used, caveats, flags = {}, set(), [], []

    # the dry run must pass, except the SAR schema failure the operator ruled on
    tab, xchk = dry_run(rows)
    bad = tab[(tab.n_failed > 0) & (tab.category != "sar")]
    if len(bad) or not all(ok for _, ok, _ in xchk):
        fail("dry-run assertions failed outside the ruled SAR exception")

    build_foldref(rows)

    # ---------------- INDEX ----------------
    idx_rows = []
    for r in rows:
        x = {k: r.get(k) for k in ("dir", "run_id_dir", "run_id", "root", "category",
                                    "excluded_reason", "dataset", "arch", "method", "condition",
                                    "holdout", "seed", "suffix", "smoke", "target_access",
                                    "metric_module_sha", "config_sha256", "has_parquet")}
        meta = r.get("meta") or {}
        x["run_json_scalars"] = json.dumps({k: v for k, v in meta.items()
                                            if v is None or isinstance(v, (str, int, float, bool))},
                                           sort_keys=True)
        if r.get("has_parquet"):
            p = os.path.join(TREE, r["dir"], "perimage.parquet")
            md = pq.read_metadata(p)
            x["n_images"] = md.num_rows
            x["parquet_mtime"] = datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat()
            x["dump_sha256"] = sha256_file(p)
            if "iou_tol" in pq.read_schema(p).names:
                x["mean_iou_tol"] = float(pd.read_parquet(p, columns=["iou_tol"]).iou_tol.mean())
        idx_rows.append(x)
    idx = pd.DataFrame(idx_rows)
    for c in ("seed", "n_images"):
        idx[c] = idx[c].astype("Int64")
    for c in idx.columns:
        if idx[c].dtype == object:
            idx[c] = idx[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    idx.to_parquet(INDEX_PATH, index=False)
    index_sha256 = sha256_file(INDEX_PATH)
    print(f"\nwrote {INDEX_PATH}  ({len(idx)} rows)  sha256={index_sha256[:16]}...")

    # sun_dist of the LOCO training distribution: identical across archs/seeds?
    for fold in FOLDS:
        sd = []
        for a in ARCHS:
            for s in SEEDS:
                sd.append(cell(find(rows, a, "vanilla", "loco", fold, s)).sun_dist.to_numpy())
        spread = float(np.max(np.ptp(np.vstack(sd), axis=0)))
        FOLDREF[fold]["sun_dist_loco"] = np.mean(np.vstack(sd), axis=0)
        FOLDREF[fold]["sun_dist_spread"] = spread
        print(f"{fold}: sun_dist of the 12 vanilla-LOCO cells, max per-patch spread {spread:.3e}")

    # =================================================================
    # TASK 1 -- vanilla gap
    # =================================================================
    print("\n" + "=" * 110)
    print("TASK 1 -- vanilla gap (upper - loco), 2-class tol mIoU K=2 unseen-excl., AOI bootstrap, "
          "seed stratum, B=%d, Holm over 12" % B)
    print("=" * 110)
    t1, aoi_store, lev = [], {}, {}
    for arch in ARCHS:
        for fold in FOLDS:
            for metric, col in (("2class", "miou2_pct"), ("fgonly", "fg_pct")):
                dbs, lo_m, up_m = {}, [], []
                for s in SEEDS:
                    rl, ru = find(rows, arch, "vanilla", "loco", fold, s), find(rows, arch, "none", "upper", fold, s)
                    runs_used.update([rl["run_id"], ru["run_id"]])
                    L, U = cell(rl), cell(ru)
                    dbs[s] = (U[col] - L[col]).to_numpy(np.float64)
                    lo_m.append(float(L[col].mean()))
                    up_m.append(float(U[col].mean()))
                    if metric == "2class":
                        lev[("vanilla", arch, fold, s)] = float(L[col].mean())
                        lev[("upper", arch, fold, s)] = float(U[col].mean())
                st = paired_stats(dbs, fold, f"t1|{arch}|{fold}|{metric}")
                if metric == "2class":
                    aoi_store[(arch, fold)] = per_aoi_units(st["dbar"], fold).assign(
                        arch=arch, fold=fold, city=FOLDREF[fold]["city"])
                t1.append(dict(arch=arch, fold=fold, metric=metric, loco=np.mean(lo_m),
                               upper=np.mean(up_m), **{k: v for k, v in st.items() if k != "dbar"}))
    t1 = pd.DataFrame(t1)
    for metric in ("2class", "fgonly"):
        m = t1.metric == metric
        t1.loc[m, "p_holm"] = holm(t1.loc[m, "p_raw"].values)
    t1p = t1[t1.metric == "2class"].copy()
    t1p["per_seed_str"] = t1p.per_seed.map(lambda d: " ".join(f"{v:.3f}" for v in d.values()))
    print(t1p[["arch", "fold", "loco", "upper", "mean", "stderr", "ci_lo", "ci_hi", "p_raw", "p_holm",
               "seed_sd", "per_seed_str", "n_aoi", "n_images"]].to_string(index=False, float_format=fmt))
    n_pos_sig = int(((t1p["mean"] > 0) & (t1p.p_holm < 0.05)).sum())
    jax = t1p[t1p.fold == "jax"]
    jax_ok = int(((jax["mean"] > 0) & (jax.p_holm < 0.05)).sum())
    print(f"\npositive and Holm-significant: {n_pos_sig} of 12 cells; JAX {jax_ok} of 4")

    # pooled
    aoi_all = pd.concat(aoi_store.values(), ignore_index=True)
    pooled = []
    for arch in ARCHS:
        r = strat_pooled(aoi_all[aoi_all.arch == arch], f"t1pool|{arch}")
        pooled.append(dict(pooling=f"within-architecture: {arch}", **r))
    pooled.append(dict(pooling="all four architectures", **strat_pooled(aoi_all, "t1pool|all")))
    pooled = pd.DataFrame(pooled)
    print("\n--- pooled over per-AOI seed-averaged gaps, fold-stratified ---")
    print(pooled.to_string(index=False, float_format=fmt))

    # cross-check vs results/02 (s0, mamnet/dinov3)
    with open(PRIOR_02) as fh:
        p02 = json.load(fh)
    ref02 = {}
    for c in p02["conditions"]:
        if c["metric"] == "gap_2class_tol_k2_unseen_excluded":
            _, a, f = [x.strip() for x in c["label"].split("/")]
            ref02[(a, f.replace("holdout ", "").lower())] = c["mean"]
    xc = []
    for (a, f), v02 in ref02.items():
        rl, ru = find(rows, a, "vanilla", "loco", f, 0), find(rows, a, "none", "upper", f, 0)
        g = float((cell(ru).miou2_pct - cell(rl).miou2_pct).mean())
        old = os.path.join(BASE, "dumps", "seo")
        sl = sha256_file(os.path.join(old, rl["run_id"], "perimage.parquet")) == \
            sha256_file(os.path.join(TREE, rl["dir"], "perimage.parquet"))
        su = sha256_file(os.path.join(old, ru["run_id"], "perimage.parquet")) == \
            sha256_file(os.path.join(TREE, ru["dir"], "perimage.parquet"))
        xc.append(dict(arch=a, fold=f, gap_s0_now=g, gap_02=v02, now_4dp=round(g, 4),
                       match_4dp=round(g, 4) == v02, loco_parquet_identical_to_02=sl,
                       upper_parquet_identical_to_02=su))
    xc = pd.DataFrame(xc)
    print("\n--- cross-check: s0 MAMNet/DINOv3 vs results/02_seo_loco.json ---")
    print(xc.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    if not xc.match_4dp.all():
        fail("s0 cross-check against results/02 does not reproduce to 4 dp")
    s0_2arch = []
    for (a, f) in ref02:
        rl, ru = find(rows, a, "vanilla", "loco", f, 0), find(rows, a, "none", "upper", f, 0)
        s0_2arch.append(per_aoi_units((cell(ru).miou2_pct - cell(rl).miou2_pct).to_numpy(), f)
                        .assign(arch=a, fold=f))
    s0p = float(pd.concat(s0_2arch).gap.mean())
    p02_pool = next(t["statistic"] for t in p02["tests"] if t["compares"] == ["both architectures"])
    print(f"s0 two-architecture pooled per-AOI gap {s0p:.4f} vs 02's {p02_pool} -> "
          f"{'match' if round(s0p, 4) == p02_pool else 'MISMATCH'}")

    out["task1"] = dict(
        cells=[{k: (r6(v) if isinstance(v, float) else v) for k, v in r.items() if k != "per_seed_str"}
               for r in t1.to_dict("records")],
        pooled=pooled.to_dict("records"),
        n_positive_holm_sig_of_12=n_pos_sig, n_jax_positive_holm_sig_of_4=jax_ok,
        crosscheck_vs_02=xc.to_dict("records"),
        crosscheck_s0_2arch_pooled=dict(now=s0p, results_02=p02_pool, match_4dp=round(s0p, 4) == p02_pool),
        row1_criterion="REVIEWS.md arm A row 1 as signed: gap positive and significant in the large "
                       "majority of cells including JAX.",
    )

    # e200 sensitivity on task 1
    sens_e200 = []
    for arch, fold, swap in (("oglanet", "oma", "loco"), ("oglanet", "ucsd", "upper")):
        dbs = {}
        for s in SEEDS:
            rl = find(rows, arch, "vanilla", "loco", fold, s)
            ru = find(rows, arch, "none", "upper", fold, s)
            if s == 0 and swap == "loco":
                rl = find(rows, arch, "vanilla", "loco", fold, 0, category="retrain_suffix")
            if s == 0 and swap == "upper":
                ru = find(rows, arch, "none", "upper", fold, 0, category="retrain_suffix")
            dbs[s] = (cell(ru).miou2_pct - cell(rl).miou2_pct).to_numpy()
        st = paired_stats(dbs, fold, f"t1e200|{arch}|{fold}")
        base = t1p[(t1p.arch == arch) & (t1p.fold == fold)].iloc[0]
        sens_e200.append(dict(arch=arch, fold=fold, swapped=f"{swap} s0 -> -e200 retrain",
                              gap_primary=float(base["mean"]), gap_with_e200=st["mean"],
                              ci_lo=st["ci_lo"], ci_hi=st["ci_hi"], p_raw=st["p_raw"]))
    out["task1"]["sensitivity_e200_retrain"] = sens_e200
    print("\n--- sensitivity: D-A2-1 -e200 retrain substituted for the s0 cell ---")
    print(pd.DataFrame(sens_e200).to_string(index=False, float_format=fmt))

    # =================================================================
    # TASK 2 -- method table
    # =================================================================
    print("\n" + "=" * 110)
    print("TASK 2 -- method vs vanilla-LOCO, AOI bootstrap with seed stratum, Holm within method family")
    print("=" * 110)
    t2 = []
    mlev = {}
    for method in METHODS:
        for arch in ARCHS:
            for fold in FOLDS:
                recs = {}
                for s in SEEDS:
                    if method == "sar":
                        r = find_opt(rows, arch, "sar", "loco", fold, s, category="sar")
                    else:
                        r = find_opt(rows, arch, method, "loco", fold, s)
                    if r is not None:
                        recs[s] = r
                if not recs:
                    continue
                if len(recs) != 3:
                    fail(f"{method}/{arch}/{fold}: {len(recs)} seeds")
                acc = {r["target_access"] for r in recs.values()}
                if len(acc) != 1:
                    fail(f"{method}/{arch}/{fold}: mixed target_access {acc}")
                acc = acc.pop()
                if acc != expected_access(arch, method):
                    fail(f"{method}/{arch}/{fold}: target_access {acc} != manifest")
                for metric, col in (("2class", "miou2_pct"), ("fgonly", "fg_pct")):
                    dbs, mm, vm, be, er, so, te, rs = {}, [], [], [], [], [], [], []
                    for s, r in recs.items():
                        runs_used.add(r["run_id"])
                        meta = r["meta"]
                        if method == "sar":
                            sc = sar_cell(r)
                            M, V = sc["sar"], sc["unadapted"]
                        else:
                            M = cell(r)
                            V = cell(find(rows, arch, "vanilla", "loco", fold, s))
                        dbs[s] = (M[col] - V[col]).to_numpy(np.float64)
                        mm.append(float(M[col].mean()))
                        vm.append(float(V[col].mean()))
                        if metric == "2class":
                            mlev[(method, arch, fold, s)] = float(M[col].mean())
                        be.append(meta.get("best_epoch"))
                        er.append(meta.get("epochs_run"))
                        so.append(meta.get("stopped_on"))
                        te.append(meta.get("train_eval_disabled"))
                        rs.append(meta.get("resume_source"))
                    st = paired_stats(dbs, fold, f"t2|{method}|{arch}|{fold}|{metric}")
                    t2.append(dict(method=method, arch=arch, fold=fold, metric=metric,
                                   target_access=acc,
                                   block=("crossearth" if arch == "crossearth" else "main"),
                                   population=("unseen_INCLUDED (SAR own counts; comparator = in-dump "
                                               "unadapted pass)" if method == "sar" else "unseen_excluded"),
                                   method_level=np.mean(mm), vanilla_level=np.mean(vm),
                                   upper_level=(None if method == "sar" else
                                                np.mean([lev[("upper", arch, fold, s)] for s in SEEDS])
                                                if metric == "2class" else None),
                                   best_epoch=be, epochs_run=er, stopped_on=so,
                                   train_eval_disabled=te, resume_source=rs,
                                   **{k: v for k, v in st.items() if k != "dbar"}))
                    if metric == "2class":
                        t2[-1]["_dbar"] = st["dbar"]
    t2 = pd.DataFrame(t2)
    for method in METHODS:
        for metric in ("2class", "fgonly"):
            m = (t2.method == method) & (t2.metric == metric)
            t2.loc[m, "p_holm"] = holm(t2.loc[m, "p_raw"].values)
            t2.loc[m, "holm_family_size"] = int(m.sum())
    t2p = t2[t2.metric == "2class"].copy()
    t2p["seeds_str"] = t2p.per_seed.map(lambda d: " ".join(f"{v:+.2f}" for v in d.values()))
    t2p["best_ep"] = t2p.best_epoch.map(lambda v: ",".join(str(x) for x in v))
    print(t2p[["method", "arch", "fold", "target_access", "method_level", "vanilla_level", "mean",
               "seed_sd", "ci_lo", "ci_hi", "p_holm", "seeds_str", "best_ep"]]
          .to_string(index=False, float_format=fmt))

    # family summaries
    fam = []
    for method in METHODS:
        sub = t2p[t2p.method == method]
        for scope, s2 in [("all", sub)] + [(a, sub[sub.arch == a]) for a in ARCHS if (sub.arch == a).any()]:
            fam.append(dict(method=method, scope=scope, n_cells=len(s2),
                            target_access=";".join(sorted(set(s2.target_access))),
                            n_neg_holm_sig=int(((s2["mean"] < 0) & (s2.p_holm < 0.05)).sum()),
                            n_pos_holm_sig=int(((s2["mean"] > 0) & (s2.p_holm < 0.05)).sum()),
                            n_below_vanilla_point=int((s2["mean"] < 0).sum()),
                            mean_delta_unweighted=float(s2["mean"].mean()),
                            worst_delta=float(s2["mean"].min()),
                            worst_delta_cell=f"{s2.loc[s2['mean'].idxmin(), 'arch']}/"
                                             f"{s2.loc[s2['mean'].idxmin(), 'fold'].upper()}"))
    fam = pd.DataFrame(fam)
    print("\n--- per method family (access classes never pooled: 'all' rows carry the set) ---")
    print(fam.to_string(index=False, float_format=fmt))

    # strongest method per arch, same access class shown
    strongest = []
    for arch in ARCHS:
        sub = t2p[(t2p.arch == arch) & (t2p.method != "sar")]
        g = sub.groupby(["method", "target_access"])["mean"].mean().reset_index().sort_values("mean")
        b = g.iloc[-1]
        strongest.append(dict(arch=arch, strongest_by_mean_delta=b["method"],
                              target_access=b["target_access"], mean_delta_over_folds=float(b["mean"]),
                              note="unseen-excluded methods only; SAR is on a different pixel population"))
    print("\n--- strongest adaptation method per architecture by fold-averaged delta (descriptive) ---")
    print(pd.DataFrame(strongest).to_string(index=False, float_format=fmt))

    # mrfp_plus mamnet early-peak flag
    mr = t2p[(t2p.method == "mrfp_plus") & (t2p.arch == "mamnet")]
    mr_be = sorted(x for v in mr.best_epoch for x in v)
    print(f"\nmamnet/mrfp_plus best_epoch over its 9 cells: {mr_be}")

    # faf sensitivity: drop the two parents' seeds
    faf = []
    for method, arch, fold, seed in (("bsm", "oglanet", "jax", 1), ("isw", "oglanet", "oma", 2)):
        dbs = {}
        for s in SEEDS:
            if s == seed:
                continue
            dbs[s] = (cell(find(rows, arch, method, "loco", fold, s)).miou2_pct -
                      cell(find(rows, arch, "vanilla", "loco", fold, s)).miou2_pct).to_numpy()
        st = paired_stats(dbs, fold, f"t2faf|{method}|{arch}|{fold}")
        base = t2p[(t2p.method == method) & (t2p.arch == arch) & (t2p.fold == fold)].iloc[0]
        faf.append(dict(cell=f"{arch}-{method}-loco-r3500-{fold}-hr-s{seed}",
                        row=f"{method}/{arch}/{fold.upper()}", delta_3_seeds=float(base["mean"]),
                        delta_excluding_flagged_seed=st["mean"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"],
                        p_raw=st["p_raw"]))
    print("\n--- sensitivity: failed-as-finding parents dropped from their rows (primary keeps them) ---")
    print(pd.DataFrame(faf).to_string(index=False, float_format=fmt))

    # e200 sensitivity on oglanet OMA method rows: vanilla s0 -> e200 (and fda s0 -> e200)
    e2 = []
    van_e = cell(find(rows, "oglanet", "vanilla", "loco", "oma", 0, category="retrain_suffix"))
    for method in [m for m in METHODS if m != "sar"]:
        if find_opt(rows, "oglanet", method, "loco", "oma", 0) is None:
            continue
        dbs = {}
        for s in SEEDS:
            M = cell(find(rows, "oglanet", method, "loco", "oma", s))
            if s == 0 and method == "fda":
                M = cell(find(rows, "oglanet", "fda", "loco", "oma", 0, category="retrain_suffix"))
            V = van_e if s == 0 else cell(find(rows, "oglanet", "vanilla", "loco", "oma", s))
            dbs[s] = (M.miou2_pct - V.miou2_pct).to_numpy()
        st = paired_stats(dbs, "oma", f"t2e200|{method}")
        base = t2p[(t2p.method == method) & (t2p.arch == "oglanet") & (t2p.fold == "oma")].iloc[0]
        e2.append(dict(row=f"{method}/oglanet/OMA", delta_primary=float(base["mean"]),
                       delta_with_e200=st["mean"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"]))
    print("\n--- sensitivity: OGLANet/OMA rows with the -e200 retrains substituted at s0 ---")
    print(pd.DataFrame(e2).to_string(index=False, float_format=fmt))

    # ---- SIB pooled worst case ----
    print("\n--- SIB worst-case criterion, POOLED scope (min over all SIB cells vs min over all "
          "vanilla-LOCO cells) ---")

    def worst(levels, method, archs, seeds, seed_mean):
        cand = {}
        for a in archs:
            for f in FOLDS:
                if seed_mean:
                    vals = [levels[(method, a, f, s)] for s in seeds if (method, a, f, s) in levels]
                    if len(vals) == len(seeds):
                        cand[(a, f, "mean")] = float(np.mean(vals))
                else:
                    for s in seeds:
                        if (method, a, f, s) in levels:
                            cand[(a, f, s)] = levels[(method, a, f, s)]
        k = min(cand, key=cand.get)
        return k, cand[k], len(cand)

    allev = dict(mlev)
    for (cond, a, f, s), v in lev.items():
        if cond == "vanilla":
            allev[("vanilla", a, f, s)] = v
    access_of = {a: EXPECTED_ACCESS_SIB[a] for a in ARCHS}
    wc = []
    for unit, seeds, seed_mean in (("arch x fold, seed-mean level", SEEDS, True),
                                   ("arch x fold x seed (seed-resolved cell)", SEEDS, False),
                                   ("s0 only", [0], False)):
        for scope, archs in [("pooled: 4 archs", ARCHS), ("pooled: mamnet+dinov3 (03's grid)",
                                                         ["mamnet", "dinov3"])] + \
                            [(f"per-arch: {a}", [a]) for a in ARCHS]:
            if unit != "s0 only" and scope.startswith("pooled: mamnet+dinov3"):
                continue
            (ma, mf, ms), mval, nm = worst(allev, "sib", archs, seeds, seed_mean)
            (va, vf, vs), vval, nv = worst(allev, "vanilla", archs, seeds, seed_mean)
            same = (ma, mf, ms) == (va, vf, vs)
            ci = (None, None)
            if same:
                if ms == "mean":
                    dbs = {s: (cell(find(rows, ma, "sib", "loco", mf, s)).miou2_pct -
                               cell(find(rows, ma, "vanilla", "loco", mf, s)).miou2_pct).to_numpy()
                           for s in seeds}
                else:
                    dbs = {ms: (cell(find(rows, ma, "sib", "loco", mf, ms)).miou2_pct -
                                cell(find(rows, ma, "vanilla", "loco", mf, ms)).miou2_pct).to_numpy()}
                st = paired_stats(dbs, mf, f"wc|{unit}|{scope}")
                ci = (st["ci_lo"], st["ci_hi"])
            wc.append(dict(unit=unit, scope=scope, n_sib_cells=nm, n_vanilla_cells=nv,
                           sib_worst_cell=f"{ma}/{mf.upper()}/{ms if ms == 'mean' else 's' + str(ms)}",
                           sib_worst=mval, sib_worst_access=access_of[ma],
                           vanilla_worst_cell=f"{va}/{vf.upper()}/{vs if vs == 'mean' else 's' + str(vs)}",
                           vanilla_worst=vval, difference=mval - vval, criterion_met=bool(mval >= vval),
                           worst_cells_same=same, paired_ci_lo=ci[0], paired_ci_hi=ci[1],
                           crosses_access_classes=len({access_of[a] for a in archs}) > 1))
    wc = pd.DataFrame(wc)
    print(wc[["unit", "scope", "n_sib_cells", "sib_worst_cell", "sib_worst", "sib_worst_access",
              "vanilla_worst_cell", "vanilla_worst", "difference", "criterion_met", "paired_ci_lo",
              "paired_ci_hi"]].to_string(index=False, float_format=fmt))
    x03 = wc[(wc.unit == "s0 only") & (wc.scope == "pooled: mamnet+dinov3 (03's grid)")].iloc[0]
    print(f"\ncross-check vs 03: s0 mamnet+dinov3 difference {x03.difference:+.4f} (03: -0.4216) -> "
          f"{'match' if round(x03.difference, 4) == -0.4216 else 'MISMATCH'}")
    if round(x03.difference, 4) != -0.4216:
        fail("SIB worst-case s0 cross-check does not reproduce 03's -0.4216")

    def rowdict(r):
        d = {k: (r6(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()
             if k not in ("_dbar", "seeds_str", "best_ep")}
        d["per_seed"] = {str(k): r6(v) for k, v in r["per_seed"].items()}
        return d

    out["task2"] = dict(
        method_table=[rowdict(r) for r in t2.to_dict("records")],
        family_summary=fam.to_dict("records"),
        strongest_method_per_arch_descriptive=strongest,
        sib_pooled_worst_case=wc.to_dict("records"),
        sib_crosscheck_vs_03=dict(s0_mamnet_dinov3_difference=float(x03.difference), results_03=-0.4216,
                                  match_4dp=True),
        mamnet_mrfp_plus_best_epochs=mr_be,
        sensitivity_failed_as_finding_parents_dropped=faf,
        sensitivity_e200_oglanet_oma=e2,
        crossearth_block=dict(
            evaluated=CROSSEARTH_METHODS,
            not_available=["fada", "iim", "mrfp_plus", "segdesic", "isw"],
            not_available_reason=("no published CrossEarth implementation exists for these five "
                                  "methods; they could not be mapped onto CrossEarth's Mask2Former "
                                  "architecture without inventing a recipe. An availability fact, "
                                  "not a result."),
            exp_a="no published CrossEarth Experiment A implementation exists; table entry, not a gap.",
            rows=[rowdict(r) for r in t2[(t2.arch == "crossearth")].to_dict("records")],
            vanilla_gap_rows=[{k: (r6(v) if isinstance(v, float) else v) for k, v in r.items()
                               if k != "per_seed_str"}
                              for r in t1[(t1.arch == "crossearth")].to_dict("records")],
            sar_recipe_note="CrossEarth SAR recipe is recorded as 'a4 pilot selection, BORROWED' "
                            "(lr 2.5e-4, 10 steps), not selected on CrossEarth."),
    )

    # =================================================================
    # TASK 3 -- Exp A external
    # =================================================================
    print("\n" + "=" * 110)
    print("TASK 3 -- Exp A vs its vanilla-LOCO parent, fg-only IoU K=2 (matched to the parent's "
          "sanity_floor_metric_value_pct)")
    print("=" * 110)
    t3, t3cells = [], []
    for arch in ["mamnet", "dinov3", "oglanet"]:
        for fold in FOLDS:
            dbs = {}
            for s in SEEDS:
                re_ = find(rows, arch, "expa", "loco", fold, s, category="expa")
                meta = re_["meta"]
                pr = find(rows, arch, "vanilla", "loco", fold, s)
                if meta.get("parent_run_id") != pr["run_id"]:
                    fail(f"{re_['run_id']}: parent_run_id {meta.get('parent_run_id')} != {pr['run_id']}")
                runs_used.update([re_["run_id"], pr["run_id"]])
                E, P = cell(re_), cell(pr)
                if not np.array_equal(E.n_valid_px.to_numpy(), P.n_valid_px.to_numpy()):
                    fail(f"{re_['run_id']}: n_valid_px differs from parent -> different population")
                dbs[s] = (E.fg_pct - P.fg_pct).to_numpy(np.float64)
                e_head, p_floor = meta["reporting_metric_value_pct"], pr["meta"]["sanity_floor_metric_value_pct"]
                p_head = pr["meta"]["reporting_metric_value_pct"]
                e_now, p_now = float(E.fg_pct.mean()), float(P.fg_pct.mean())
                stc = paired_stats({s: dbs[s]}, fold, f"t3cell|{arch}|{fold}|{s}")
                t3cells.append(dict(arch=arch, fold=fold, seed=s, run_id=re_["run_id"],
                                    expa_fg=e_now, parent_fg=p_now, delta=stc["mean"],
                                    ci_lo=stc["ci_lo"], ci_hi=stc["ci_hi"], p_raw=stc["p_raw"],
                                    expa_headline_sidecar=e_head, parent_sanity_floor_sidecar=p_floor,
                                    sidecar_delta=e_head - p_floor,
                                    recompute_vs_sidecar_expa=e_now - e_head,
                                    recompute_vs_sidecar_parent=p_now - p_floor,
                                    WRONG_headline_minus_headline=e_head - p_head,
                                    best_epoch=meta.get("best_epoch"), epochs_run=meta.get("epochs_run"),
                                    stopped_on=meta.get("stopped_on"),
                                    target_access=re_["target_access"]))
            st = paired_stats(dbs, fold, f"t3|{arch}|{fold}")
            t3.append(dict(arch=arch, fold=fold, target_access="target_labels",
                           **{k: v for k, v in st.items() if k != "dbar"}))
            t3[-1]["_units"] = per_aoi_units(st["dbar"], fold).assign(arch=arch, fold=fold)
    t3 = pd.DataFrame(t3)
    t3["p_holm"] = holm(t3.p_raw.values)
    t3c = pd.DataFrame(t3cells)
    t3c["p_holm_27"] = holm(t3c.p_raw.values)
    print(t3[["arch", "fold", "mean", "stderr", "ci_lo", "ci_hi", "p_raw", "p_holm", "seed_sd"]]
          .to_string(index=False, float_format=fmt))
    print("\n--- 27 seed-resolved cells ---")
    print(t3c[["arch", "fold", "seed", "expa_fg", "parent_fg", "delta", "ci_lo", "ci_hi", "p_holm_27",
               "recompute_vs_sidecar_expa", "recompute_vs_sidecar_parent", "WRONG_headline_minus_headline"]]
          .to_string(index=False, float_format=fmt))
    mx_rc = float(max(t3c.recompute_vs_sidecar_expa.abs().max(), t3c.recompute_vs_sidecar_parent.abs().max()))
    print(f"max |recompute - sidecar| over 54 values: {mx_rc:.2e}")
    sep = []
    for arch in ["mamnet", "dinov3", "oglanet"]:
        c = t3c[t3c.arch == arch]
        r = t3[t3.arch == arch]
        pooled_u = pd.concat(list(r["_units"]), ignore_index=True).assign(k="x")
        pl = strat_pooled(pooled_u.rename(columns={"k": "kk"}), f"t3pool|{arch}", across="kk")
        sep.append(dict(arch=arch,
                        sign_count_cells=f"{int((c.delta > 0).sum())} positive / {int((c.delta < 0).sum())} "
                                         f"negative of {len(c)}",
                        cell_delta_min=float(c.delta.min()), cell_delta_max=float(c.delta.max()),
                        n_cells_ci_excl_zero_holm27=int((c.p_holm_27 < 0.05).sum()),
                        n_cells_ci_lo_gt0=int((c.ci_lo > 0).sum()), n_cells_ci_hi_lt0=int((c.ci_hi < 0).sum()),
                        n_archfold_holm9_sig=int((r.p_holm < 0.05).sum()),
                        archfold_ci=[(round(a, 3), round(b, 3)) for a, b in zip(r.ci_lo, r.ci_hi)],
                        pooled_mean=pl["mean"], pooled_ci_lo=pl["ci_lo"], pooled_ci_hi=pl["ci_hi"],
                        pooled_p=pl["p"],
                        wrong_differencing_sign_flips=int((np.sign(c.WRONG_headline_minus_headline)
                                                           != np.sign(c.delta)).sum())))
    sep = pd.DataFrame(sep)
    print("\n--- separation by architecture ---")
    print(sep.to_string(index=False, float_format=fmt))
    rng_m = (sep.set_index("arch").cell_delta_min, sep.set_index("arch").cell_delta_max)
    perfect = bool(rng_m[0]["mamnet"] > 0 and rng_m[0]["dinov3"] > 0 and rng_m[1]["oglanet"] < 0)
    print(f"perfect sign separation (MAMNet, DINOv3 all > 0; OGLANet all < 0): {perfect}")
    out["task3"] = dict(
        archfold=[{k: (r6(v) if isinstance(v, float) else v) for k, v in r.items() if k != "_units"}
                  for r in t3.to_dict("records")],
        cells=t3c.to_dict("records"), separation=sep.to_dict("records"),
        perfect_sign_separation_by_arch=perfect,
        max_abs_recompute_minus_sidecar=mx_rc,
        crossearth="no published CrossEarth Experiment A implementation exists; table entry, not a gap.",
        denominator_note=("Exp A cells report fg-only IoU at K=2; the parents' headline is two-class. "
                          "The matched comparison is Exp A fg-only vs the parent's fg-only "
                          "(sanity_floor_metric_value_pct), recomputed per image from raw counts. "
                          "WRONG_headline_minus_headline is carried only to show the trap."))

    # =================================================================
    # TASK 4 -- sun-elevation regression
    # =================================================================
    print("\n" + "=" * 110)
    print("TASK 4 -- per-AOI seed-averaged gap ~ sun_dist + C(city) [+ C(arch)]; OLS and fold-clustered SE")
    print("=" * 110)
    import statsmodels.formula.api as smf
    t4 = []

    def fit(frame, label, formula):
        m_ols = smf.ols(formula, data=frame).fit()
        m_cl = smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame["fold"]})
        t4.append(dict(model=label, formula=formula, n=len(frame),
                       slope=float(m_ols.params["sun_dist_loco"]),
                       se_ols=float(m_ols.bse["sun_dist_loco"]), p_ols=float(m_ols.pvalues["sun_dist_loco"]),
                       se_cluster=float(m_cl.bse["sun_dist_loco"]),
                       p_cluster=float(m_cl.pvalues["sun_dist_loco"]),
                       n_clusters=int(frame.fold.nunique()), r2=float(m_ols.rsquared)))

    for arch in ARCHS:
        fit(aoi_all[aoi_all.arch == arch].copy(), arch, "gap ~ sun_dist_loco + C(city)")
    fit(aoi_all.copy(), "all four architectures", "gap ~ sun_dist_loco + C(city) + C(arch)")
    fit(aoi_all[aoi_all.arch.isin(["mamnet", "dinov3", "oglanet"])].copy(),
        "three non-CrossEarth architectures", "gap ~ sun_dist_loco + C(city) + C(arch)")
    t4 = pd.DataFrame(t4)
    print(t4.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    pooled4 = t4[t4.model == "all four architectures"].iloc[0]
    row7 = bool(pooled4.slope < 0 and pooled4.p_cluster >= 0.05)
    out["task4"] = dict(
        models=t4.to_dict("records"),
        sun_dist_spread_across_loco_cells={f: FOLDREF[f]["sun_dist_spread"] for f in FOLDS},
        pooled_slope_negative=bool(pooled4.slope < 0),
        pooled_p_cluster_ge_005=bool(pooled4.p_cluster >= 0.05),
        n_models_slope_negative=int((t4.slope < 0).sum()), n_models=len(t4),
        n_models_p_cluster_lt_005=int((t4.p_cluster < 0.05).sum()),
        row7_conditions_met_on_pooled_model=row7,
        wording_constraint="HLT signed: no support, and the point estimate leans opposite; not "
                           "significant at the resolution S-EO permits.")

    # =================================================================
    # TASK 5 -- section 4.2 probes
    # =================================================================
    print("\n" + "=" * 110)
    print("TASK 5 -- histogram matching (published grayscale rule) from the -p1histmatch dumps")
    print("=" * 110)
    probes = [r for r in rows if r.get("category") == "probe"]
    inv = json.load(open(INVENTORY))
    faf_probes = {c["run_id_probe"] if "run_id_probe" in c else c["run_id"] + "-p1histmatch"
                  for c in inv["cells"] if c["final_outcome"] == "failed_as_finding"}
    t5 = []
    for r in probes:
        meta = r["meta"]
        pf_path = os.path.join(TREE, r["dir"], "probe_family.json")
        if os.path.isfile(pf_path):
            with open(pf_path) as fh:
                pfam = json.load(fh)
            fam_src = "probe_family.json"
        else:
            # the 6 s0 MAMNet/DINOv3 vanilla probes of 2026-08-03 predate D-A2-2 and carry no
            # probe_family.json; a vanilla parent with its own fold/seed upper is the vanilla family
            if meta.get("method") != "vanilla":
                fail(f"{r['run_id_dir']}: no probe_family.json on a non-vanilla probe")
            pfam = dict(probe_family="vanilla", joins_headline_count=True, parent_is_a_d_a2_1_retrain=False)
            fam_src = "derived: pre-D-A2-2 vanilla probe without probe_family.json"
        san = meta.get("sanity", {})
        fold, arch, method, seed = r["holdout"], r["arch"], r["method"], r["seed"]
        parent_id = meta["parent_run_id"]
        prec = next(x for x in rows if x.get("run_id") == parent_id and x.get("category") in
                    ("cell", "retrain_suffix"))
        df = load_parquet(r).sort_values("image_id").reset_index(drop=True)
        if not np.array_equal(df.image_id.to_numpy(), FOLDREF[fold]["image_id"]):
            fail(f"{r['run_id_dir']}: image_id order differs")
        un = add_metrics(df)
        mg = df.copy()
        for k in ("tp_tol", "fp_tol", "fn_tol", "tn_tol"):
            mg[k] = df[f"{k}_matched_gray"]
        mg = add_metrics(mg)
        assert float(np.abs(mg.miou2_pct - df.miou_tol_pct_matched_gray).max()) < 1e-9
        P = cell(prec)
        repro = float(un.miou2_pct.mean() - P.miou2_pct.mean())
        repro_img = float(np.abs(un.miou2_pct - P.miou2_pct).max())
        rule = san.get("repro_rule") or pfam.get("reproduction_rule") or "bit_exact"
        tol = san.get("repro_tolerance_miou_pct", pfam.get("reproduction_tolerance_miou_pct"))
        band_n = (san.get("repro_rule_detail") or {}).get("band_n_passes")
        if tol is None:
            fail(f"{r['run_id_dir']}: no reproduction tolerance recorded")
        e9 = e9_band(prec)
        row = dict(probe=r["run_id_dir"], arch=arch, method=method, fold=fold, seed=seed,
                   parent=parent_id, family=pfam.get("probe_family"),
                   joins_headline=pfam.get("joins_headline_count"),
                   parent_is_e200=bool(pfam.get("parent_is_a_d_a2_1_retrain")),
                   parent_target_access=prec["target_access"], probe_sidecar_access=r["target_access"],
                   mixed_denominator=(meta["mixed_denominator"] if meta.get("mixed_denominator") is not None
                                      else pfam.get("probe_family") == "method"),
                   mixed_denominator_disclosure=(meta.get("mixed_denominator_disclosure")
                                                 or pfam.get("interpretive_guard")),
                   disclosure_source=("run.json" if meta.get("mixed_denominator_disclosure")
                                      else "probe_family.json interpretive_guard"),
                   family_source=fam_src,
                   repro_rule=rule, repro_tol_pct=tol, repro_gate_width_pct=2 * tol,
                   repro_band_n_passes=band_n,
                   repro_delta_sidecar_pct=san.get("repro_delta_miou_pct"),
                   repro_delta_cell_pct=repro,
                   repro_max_abs_per_image_pct=repro_img,
                   repro_pass=bool(abs(repro) <= tol),
                   repro_exact_4dp=bool(round(un.miou2_pct.mean(), 4) == round(P.miou2_pct.mean(), 4)),
                   parent_e9_band_n=(e9 or {}).get("n_passes"),
                   parent_e9_band_width_pct=(e9 or {}).get("width_miou_pct"),
                   unmatched_miou=float(un.miou2_pct.mean()), matched_miou=float(mg.miou2_pct.mean()),
                   unmatched_fg=float(un.fg_pct.mean()), matched_fg=float(mg.fg_pct.mean()),
                   R_gray_mean=float(df.loc[df.r_included.astype(bool), "recovery_ratio_R_gray"].mean()),
                   R_gray_median=float(df.loc[df.r_included.astype(bool), "recovery_ratio_R_gray"].median()),
                   r_n_included=int(df.r_included.astype(bool).sum()),
                   failed_as_finding=r["run_id_dir"] in faf_probes)
        st = paired_stats({seed: (mg.miou2_pct - un.miou2_pct).to_numpy()}, fold, f"t5|{r['run_id_dir']}")
        stf = paired_stats({seed: (mg.fg_pct - un.fg_pct).to_numpy()}, fold, f"t5fg|{r['run_id_dir']}")
        row.update(delta_miou=st["mean"], ci_lo=st["ci_lo"], ci_hi=st["ci_hi"], p_raw=st["p_raw"],
                   delta_fg=stf["mean"], fg_ci_lo=stf["ci_lo"], fg_ci_hi=stf["ci_hi"])
        t5.append(row)
        runs_used.add(r["run_id_dir"])
    t5 = pd.DataFrame(t5)
    head = t5[(t5.family == "vanilla") & (~t5.parent_is_e200)].copy()
    if len(head) != 36:
        fail(f"headline vanilla probes: {len(head)} (expected 36)")
    head["p_holm_36"] = holm(head.p_raw.values)
    n_imp = int((head.delta_miou > 0).sum())
    n_imp_fg = int((head.delta_fg > 0).sum())
    n_imp_sig = int(((head.delta_miou > 0) & (head.p_holm_36 < 0.05)).sum())
    n_deg_sig = int(((head.delta_miou < 0) & (head.p_holm_36 < 0.05)).sum())
    print(head[["probe", "repro_rule", "repro_delta_cell_pct", "repro_tol_pct", "repro_gate_width_pct",
                "parent_e9_band_n", "parent_e9_band_width_pct", "unmatched_miou", "matched_miou", "delta_miou", "ci_lo",
                "ci_hi", "p_holm_36", "delta_fg", "R_gray_mean"]].to_string(index=False, float_format=fmt))
    print(f"\nHEADLINE (vanilla family, D-A2-2): cells improving on two-class mIoU (point estimate): "
          f"{n_imp} of 36; on fg-only IoU: {n_imp_fg} of 36; improving at Holm-36 p<.05: {n_imp_sig}; "
          f"degrading at Holm-36 p<.05: {n_deg_sig}")
    print(f"combined with arm C's internal 0 of 26 usable cells (PAPER_STATE.md arm C row, not "
          f"recomputed here): {n_imp} of {36 + 26}")
    repro_fail = head[~head.repro_pass]
    if len(repro_fail):
        flags.append(f"{len(repro_fail)} headline probes fail their reproduction gate: "
                     f"{list(repro_fail.probe)}")
    by_arch = head.groupby("arch").agg(n=("probe", "size"), improving=("delta_miou", lambda s: int((s > 0).sum())),
                                       mean_delta=("delta_miou", "mean"),
                                       min_delta=("delta_miou", "min"), max_delta=("delta_miou", "max"))
    print(by_arch.to_string(float_format=fmt))

    meth = t5[(t5.family != "vanilla") & (~t5.failed_as_finding) & (~t5.parent_is_e200)].copy()
    meth["p_holm_within_method"] = np.nan
    for mth in meth.method.unique():
        m = meth.method == mth
        meth.loc[m, "p_holm_within_method"] = holm(meth.loc[m, "p_raw"].values)
    mf = meth.groupby(["method", "arch", "parent_target_access"]).agg(
        n=("probe", "size"), improving=("delta_miou", lambda s: int((s > 0).sum())),
        mean_delta=("delta_miou", "mean"), repro_all_pass=("repro_pass", "all"),
        mixed_denominator=("mixed_denominator", "first")).reset_index()
    print("\n--- METHOD FAMILY (separate; never enters the headline count) ---")
    print(mf.to_string(index=False, float_format=fmt))
    disclosures = {m: meth[meth.method == m].mixed_denominator_disclosure.iloc[0] for m in meth.method.unique()}

    # deferred / failed probes, by name
    deferred = []
    for c in inv["cells"]:
        if c["final_outcome"] == "deferred_sar":
            deferred.append(dict(probe=c["run_id"] + "-p1histmatch", outcome="deferred_sar", reason=c["reason"]))
        if c["final_outcome"] == "failed_as_finding":
            deferred.append(dict(probe=c["run_id"] + "-p1histmatch", outcome="failed_as_finding",
                                 reason=c["final_reason"]))
    on_disk = set(t5.probe)
    roster = json.load(open(T2_ROSTER))
    for c in roster["cells"]:
        pid = c["run_id_probe"]
        if pid not in on_disk:
            fdir = os.path.join(TREE, "dumps", "seo", pid)
            why = "no dump on disk"
            if os.path.isfile(os.path.join(fdir, ".FAILED")):
                with open(os.path.join(fdir, ".FAILED")) as fh:
                    why = ".FAILED: " + " ".join(fh.read().split())[:200]
            deferred.append(dict(probe=pid, outcome="tier2_not_complete (D-B5-1)", reason=why))
    deferred = pd.DataFrame(deferred)
    print("\n--- deferred / failed probe cells ---")
    print(deferred.groupby("outcome").probe.agg(["size", lambda s: ", ".join(sorted(s))[:300]]).to_string())

    # p-bar_S: availability only. The operator deferred the statistic to a second pass.
    pb = []
    for arch in ARCHS:
        for fold in FOLDS:
            for s in SEEDS:
                for cond, mth in (("loco", "vanilla"), ("upper", "none")):
                    c = cell(find(rows, arch, mth, cond, fold, s))
                    pb.append(dict(run_id=f"{arch}-{mth}-{cond}-r3500-{fold}-hr-s{s}",
                                   pbar_gt_shadow_nonnull=int(c.pbar_gt_shadow.notna().sum()),
                                   pbar_gt_bg_nonnull=int(c.pbar_gt_bg.notna().sum()),
                                   n_gt_fg_valid_pos=int((c.n_gt_fg_valid > 0).sum())))
    pb = pd.DataFrame(pb)
    pb_ok = bool((pb.pbar_gt_shadow_nonnull == pb.n_gt_fg_valid_pos).all())
    print(f"\np-bar_S column present in all 72 vanilla cells; non-null == n_gt_fg_valid>0 on every "
          f"cell: {pb_ok}")

    out["task5"] = dict(
        rule=("published grayscale-plus-ratio matching (miou_tol_pct_matched_gray). A cell IMPROVES when its "
              "cell-mean delta (matched - in-run unmatched re-inference) on two-class tol mIoU K=2 "
              "unseen-excluded is > 0, the arm F convention ('cells improving under matching'); fg-only and "
              "Holm-36 significance counts reported alongside."),
        headline=dict(n_cells=36, n_improving_miou=n_imp, n_improving_fg=n_imp_fg,
                      n_improving_holm36_sig=n_imp_sig, n_degrading_holm36_sig=n_deg_sig,
                      combined_with_armC_internal=dict(armC_usable=26, armC_improving=0,
                                                       armC_source="PAPER_STATE.md arm C row (0 of 26)",
                                                       combined=f"{n_imp} of {36 + 26}"),
                      by_arch=by_arch.reset_index().to_dict("records")),
        headline_cells=head.to_dict("records"),
        method_family=dict(summary=mf.to_dict("records"), cells=meth.to_dict("records"),
                           mixed_denominator_disclosures=disclosures,
                           note="SEPARATE family; never pooled into the headline count (D-A2-2)."),
        e200_probes=t5[t5.parent_is_e200].to_dict("records"),
        deferred_or_failed=deferred.to_dict("records"),
        failed_as_finding_probes=t5[t5.failed_as_finding].probe.tolist(),
        pbar_S=dict(status="NOT COMPUTED in this pass, per operator instruction",
                    column_present=pb_ok,
                    note=("p-bar_S is the parquet column pbar_gt_shadow, written upstream by the lifted "
                          "metrics.pbar_populations; results/04 read it from the parquet. probs.npz was used "
                          "in 04 only for the semantics verification (1 cell in full, 2-patch spot checks on 11)."),
                    probs_npz_cells_for_04_protocol=list(pb.run_id)))

    # =================================================================
    # TASK 6 -- descriptives
    # =================================================================
    print("\n" + "=" * 110)
    print("TASK 6 -- per-cell descriptives")
    print("=" * 110)
    t6 = []
    for r in rows:
        if r.get("category") not in ("cell", "retrain_suffix", "expa", "sar"):
            continue
        meta = r["meta"]
        base = dict(run_id=r["run_id_dir"], category=r["category"], arch=r["arch"], method=r["method"],
                    condition=r["condition"], fold=r["holdout"], seed=r["seed"],
                    target_access=r["target_access"], epochs_run=meta.get("epochs_run"),
                    best_epoch=meta.get("best_epoch"), stopped_on=meta.get("stopped_on"),
                    train_eval_disabled=meta.get("train_eval_disabled"),
                    train_eval_verdict=meta.get("train_eval_equivalence_verdict"),
                    resume_source=meta.get("resume_source"))
        if r["category"] == "sar":
            sc = sar_cell(r)
            S, U = sc["sar"], sc["unadapted"]
            base.update(population="unseen_INCLUDED", mean_2class=float(S.miou2_pct.mean()),
                        median_2class=float(S.miou2_pct.median()), mean_fgonly=float(S.fg_pct.mean()),
                        median_fgonly=float(S.fg_pct.median()),
                        predpos_frac_mean_img=float(S.pred_fg_fraction.mean()),
                        unadapted_mean_2class=float(U.miou2_pct.mean()))
        else:
            c = cell(r)
            base.update(population="unseen_excluded", mean_2class=float(c.miou2_pct.mean()),
                        median_2class=float(c.miou2_pct.median()), mean_fgonly=float(c.fg_pct.mean()),
                        median_fgonly=float(c.fg_pct.median()),
                        predpos_frac_pooled=float(c.n_pred_pos.sum() / c.n_valid_px.sum()))
        e9 = e9_band(r)
        if e9:
            base.update(e9_n_passes=e9["n_passes"], e9_width_pct=e9["width_miou_pct"],
                        e9_sd_pct=e9["sd_miou_pct"], e9_file=e9["file"],
                        e9_label=f"E-9 band n={e9['n_passes']}" +
                                 (" (undersampled as a gate, D-B4-3)" if e9["n_passes"] == 3 else ""))
        t6.append(base)
    t6 = pd.DataFrame(t6)
    print(t6.groupby(["arch", "method", "condition"]).agg(
        n=("run_id", "size"), mean_2class=("mean_2class", "mean"), mean_fg=("mean_fgonly", "mean"),
        epochs_run=("epochs_run", lambda s: f"{s.min()}-{s.max()}" if s.notna().any() else "-"),
        best_epoch=("best_epoch", lambda s: f"{s.min()}-{s.max()}" if s.notna().any() else "-"),
        n_e9=("e9_n_passes", lambda s: int(s.notna().sum()))).to_string(float_format=fmt))
    stop = t6.groupby(["stopped_on"], dropna=False).size()
    print("\nstopped_on:", stop.to_dict())
    ceil = t6[t6.stopped_on == "epoch_ceiling"]
    print("epoch_ceiling cells:", list(ceil.run_id))
    e9c = t6[t6.e9_n_passes.notna()].groupby("e9_n_passes").size().to_dict()
    print("E-9 bands by pass count:", e9c)
    te_mixed = t6[(t6.train_eval_disabled == False)].run_id.tolist()  # noqa: E712
    print("cells with train_eval_disabled == False:", te_mixed)
    out["task6"] = dict(cells=t6.to_dict("records"), stopped_on_counts={str(k): int(v) for k, v in stop.items()},
                        epoch_ceiling_cells=list(ceil.run_id), e9_bands_by_pass_count=
                        {str(int(k)): int(v) for k, v in e9c.items()},
                        train_eval_enabled_cells=te_mixed)

    # =================================================================
    # sanity sweep
    # =================================================================
    for name, frame in (("task1", t1), ("task2", t2), ("task3", t3)):
        bad_ci = frame[frame.ci_excludes_point]
        if len(bad_ci):
            flags.append(f"{name}: {len(bad_ci)} bootstrap CIs exclude their own point estimate")
    zero_sd = t2p[t2p.seed_sd.notna() & (t2p.seed_sd < 1e-6)]
    if len(zero_sd):
        flags.append(f"task2: {len(zero_sd)} rows with seed SD < 1e-6: "
                     f"{[(r.method, r.arch, r.fold) for r in zero_sd.itertuples()]}")
    if perfect:
        flags.append("task3: Exp A per-cell deltas separate perfectly by architecture sign "
                     "(MAMNet, DINOv3 all > 0; OGLANet all < 0); flagged as too clean per CONVENTIONS.")
    n_floor = int((t1p.p_raw <= 1.0 / (B + 1) + 1e-12).sum())
    flags.append(f"task1: {n_floor} of 12 sign-flip p-values sit at the permutation floor 1/(B+1)")

    out["_t1p"], out["_t2p"], out["_fam"], out["_wc"], out["_sep"], out["_t4"] = t1p, t2p, fam, wc, sep, t4
    out["_head"], out["_mf"], out["_deferred"] = head, mf, deferred
    out["_flags"], out["_pooled"], out["_xc"] = flags, pooled, xc
    out["_n_imp"], out["_row7"], out["_perfect"] = n_imp, row7, perfect
    out["_runs"], out["_index_sha256"] = sorted(runs_used), index_sha256
    write_results(out, rows, tab)


STATE_PICKLE = os.path.join(BASE, "analysis", "_cache", "08_state.pkl")
WRITE_ONLY_FROM_STATE = False   # True: skip the computation, re-emit the JSON from STATE_PICKLE


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if o is pd.NA:
        return None
    return o


def write_results(out, rows, tab):
    import pickle
    with open(STATE_PICKLE, "wb") as fh:
        pickle.dump(dict(out=out, tab=tab), fh)
    t1p, t2p, fam, wc, sep, t4 = (out[k] for k in ("_t1p", "_t2p", "_fam", "_wc", "_sep", "_t4"))
    head, mf, deferred, pooled = out["_head"], out["_mf"], out["_deferred"], out["_pooled"]

    # ---------------- conditions / tests in the CONTRACT 6 shape ----------------
    conditions, tests = [], []
    for r in t1p.itertuples():
        conditions.append(dict(
            label=f"S-EO vanilla gap / {r.arch} / holdout {r.fold.upper()}",
            metric="gap_2class_tol_k2_unseen_excluded", mean=r.mean, stderr=r.stderr,
            n=int(r.n_images), n_aoi=int(r.n_aoi), n_seeds=int(r.n_seeds), ci95=[r.ci_lo, r.ci_hi],
            ci_method="aoi_cluster_bootstrap_seed_stratum_B10000",
            per_seed=r.per_seed, seed_sd=r.seed_sd, loco_mean=r.loco, upper_mean=r.upper,
            target_access="source_only (loco) vs target_labels (upper)",
            baseline="none_upper_r3500_same_fold_same_seed"))
        tests.append(dict(name="AOI-level sign-flip permutation test on the seed-averaged per-image delta",
                          compares=[f"{r.arch}-none-upper-r3500-{r.fold}-hr-s{{0,1,2}}",
                                    f"{r.arch}-vanilla-loco-r3500-{r.fold}-hr-s{{0,1,2}}"],
                          statistic=r.mean, p=r.p_raw, p_holm=r.p_holm, B=B, two_tailed=True,
                          family="12 (arch x fold) pairs, Holm-Bonferroni"))
    for r in pooled.itertuples():
        conditions.append(dict(label=f"S-EO vanilla gap pooled / {r.pooling}",
                               metric="gap_2class_tol_k2_unseen_excluded_per_AOI_seed_mean",
                               mean=r.mean, stderr=r.stderr, n=int(r.n_images), n_aoi_units=int(r.n_aoi_units),
                               ci95=[r.ci_lo, r.ci_hi], ci_method="fold_stratified_AOI_bootstrap_B10000",
                               baseline="none_upper_r3500_same_fold"))
        tests.append(dict(name="AOI-unit sign-flip permutation test (pooled)", compares=[r.pooling],
                          statistic=r.mean, p=r.p, B=B, two_tailed=True))
    for r in t2p.itertuples():
        conditions.append(dict(
            label=f"S-EO {r.method} vs vanilla-LOCO / {r.arch} / holdout {r.fold.upper()}",
            metric=("delta_2class_tol_k2_unseen_INCLUDED_vs_in_dump_unadapted" if r.method == "sar"
                    else "delta_2class_tol_k2_unseen_excluded"),
            mean=r.mean, stderr=r.stderr, n=int(r.n_images), n_aoi=int(r.n_aoi), n_seeds=int(r.n_seeds),
            ci95=[r.ci_lo, r.ci_hi], ci_method="aoi_cluster_bootstrap_seed_stratum_B10000",
            per_seed=r.per_seed, seed_sd=r.seed_sd, method_level=r.method_level,
            vanilla_level=r.vanilla_level, upper_level=r.upper_level, best_epoch=r.best_epoch,
            target_access=r.target_access, block=r.block,
            baseline=("sar_unadapted_pass_same_dump" if r.method == "sar" else "vanilla_loco_same_fold_same_seed")))
        tests.append(dict(name="AOI-level sign-flip permutation test on the seed-averaged per-image delta",
                          compares=[f"{r.arch}-{r.method}-loco-r3500-{r.fold}-hr-s{{0,1,2}}",
                                    f"{r.arch}-vanilla-loco-r3500-{r.fold}-hr-s{{0,1,2}}"],
                          statistic=r.mean, p=r.p_raw, p_holm=r.p_holm, B=B, two_tailed=True,
                          family=f"method family '{r.method}', {int(r.holm_family_size)} cells, Holm-Bonferroni"))
    for r in out["task3"]["archfold"]:
        conditions.append(dict(
            label=f"S-EO Exp A vs vanilla-LOCO parent / {r['arch']} / holdout {r['fold'].upper()}",
            metric="delta_fg_only_tol_k2_unseen_excluded", mean=r["mean"], stderr=r["stderr"],
            n=r["n_images"], n_aoi=r["n_aoi"], n_seeds=r["n_seeds"], ci95=[r["ci_lo"], r["ci_hi"]],
            ci_method="aoi_cluster_bootstrap_seed_stratum_B10000", per_seed=r["per_seed"],
            seed_sd=r["seed_sd"], target_access="target_labels",
            baseline="parent vanilla_loco sanity_floor (fg-only K=2)"))
        tests.append(dict(name="AOI-level sign-flip permutation test", compares=[f"{r['arch']}-expa", "parent"],
                          statistic=r["mean"], p=r["p_raw"], p_holm=r["p_holm"], B=B, two_tailed=True,
                          family="9 (arch x fold) Exp A cells, Holm-Bonferroni"))
    for m in out["task4"]["models"]:
        tests.append(dict(name="OLS slope on per-AOI seed-averaged gap, fold-clustered SE",
                          compares=[m["model"], m["formula"]], statistic=m["slope"], p=m["p_cluster"],
                          p_ols=m["p_ols"], n=m["n"], n_clusters=m["n_clusters"], two_tailed=True))

    # ---------------- runs / exclusions ----------------
    excl = Counter()
    excluded_named = []
    for r in rows:
        c = r.get("category")
        if c in ("cell", "expa", "sar", "probe"):
            continue
        excl[(c, r.get("excluded_reason"))] += 1
        excluded_named.append(dict(dir=r["dir"], category=c, reason=r.get("excluded_reason")))

    # ---------------- headline ----------------
    lo_r = t1p.loc[t1p["mean"].idxmin()]
    hi_r = t1p.loc[t1p["mean"].idxmax()]
    n_sig = out["task1"]["n_positive_holm_sig_of_12"]
    headline = (f"On the S-EO vanilla grid of 4 architectures x 3 holdout cities x 3 seeds, the within-city "
                f"upper bound exceeds source-only LOCO in {n_sig} of 12 (architecture x city) cells at Holm "
                f"p < 0.05, by {lo_r['mean']:.4f} two-class mIoU at K=2 in the smallest cell "
                f"({lo_r.arch}/{lo_r.fold.upper()}) and {hi_r['mean']:.4f} in the largest "
                f"({hi_r.arch}/{hi_r.fold.upper()}), seed-averaged with AOI-level 95% CIs.")

    sib_wc = wc[(wc.unit == "arch x fold, seed-mean level") & (wc.scope == "pooled: 4 archs")].iloc[0]
    sib_wc_s = wc[(wc.unit == "arch x fold x seed (seed-resolved cell)") & (wc.scope == "pooled: 4 archs")].iloc[0]

    ceil = [c for c in out["task6"]["epoch_ceiling_cells"] if "-expa-" not in c]
    ceil_van = [c for c in ceil if "-vanilla-" in c or "-none-" in c]
    og = head[head.arch == "oglanet"]
    flags = list(out["_flags"]) + [
        f"task5: OGLANet vanilla probes are gated at a fixed +-0.1 mIoU tolerance (gate width 0.2); the parents' "
        f"measured E-9 bands are n=3 passes, width {og.parent_e9_band_width_pct.min():.4f} to "
        f"{og.parent_e9_band_width_pct.max():.4f}; the 0.1 tolerance is "
        f"{0.1 / og.parent_e9_band_width_pct.max():.1f}x to {0.1 / og.parent_e9_band_width_pct.min():.0f}x the "
        f"measured band width, so the gate was loose (operator ruling: state the width).",
        "task2: SAR rows on DINOv3 and CrossEarth move vanilla by 0.004 to 0.035 mIoU with across-seed SD "
        "0.0025 to 0.016; CIs exclude zero on tiny effects. Recorded as implausibly low variance.",
        f"task6: {len(ceil)} trained cells (Exp A excluded) stopped at the epoch ceiling, {len(ceil_van)} of them "
        f"vanilla-LOCO or upper cells; only the three OGLANet cells have a D-A2-1 retrain."]
    out["_flags"] = flags
    caveats = [
        f"{len(ceil)} trained cells stopped on the epoch ceiling rather than patience (listed in "
        f"task6.epoch_ceiling_cells, Exp A's fixed 30-epoch cells excluded from that count), including "
        f"{len(ceil_van)} vanilla-LOCO or upper cells ({', '.join(ceil_van)}). Only the three OGLANet cells have a "
        "from-scratch longer-horizon retrain.",
        "SAR rows are not on the same pixel population as every other row. The 36 SAR dumps are a per-(pass, step) "
        "test-time-adaptation schema without aoi_id, sun_* or the unseen-excluded counts; their K=2 counts keep "
        "DSM-unseen pixels. Per operator ruling 2026-10-24, each SAR row is the step-10 SAR pass minus the same "
        "dump's 'unadapted' pass (the vanilla-LOCO parent checkpoint) on that unseen-INCLUDED population, with AOI "
        "joined from the parent by image_id. SAR deltas are not commensurable with the other rows' deltas, and no "
        "upper bound is given for SAR rows. The 27 arm-A_a4 SAR sidecars carry no metric_module_sha; the 9 "
        "CrossEarth SAR sidecars carry 730ad4a5bc.",
        "The CrossEarth SAR recipe (lr 2.5e-4, 10 steps) is recorded as 'a4 pilot selection, BORROWED': it was "
        "selected on the CNN/DINOv3 pilot, not on CrossEarth.",
        "Three seeds per cell. Every interval is an AOI cluster bootstrap with seed as a stratum (AOIs resampled, "
        "all three seeds kept within each sampled AOI); it covers test-patch and AOI sampling, and seed variation "
        "only through the seed average. Per-seed point estimates and the across-seed SD (n = 3) are reported beside "
        "every estimate.",
        "Seed pairing between a method cell and its vanilla-LOCO comparator is by seed index. The seed-averaged "
        "estimate is invariant to that pairing; the per-seed deltas are not.",
        "Budget asymmetry between the two sides of every gap, as in results/02: LOCO trains on 3,500 patches from "
        "the two source cities, the upper bound on 3,500 patches from the held-out city.",
        "108 test patches (29 OMA, 79 UCSD, 0 JAX) are entirely DSM-unseen; their per-patch metric is 0.0 in every "
        "cell of the fold, they contribute a delta of exactly 0, and every headline number includes them.",
        "rung 3500 is the S-EO patch budget, not a point on the ShadowTransfer 225/450/900 ladder.",
        "S-EO labels are auto-generated. Both sides of every paired delta use the same labels on the same patches; "
        "absolute levels carry the label noise.",
        "The train-side per-image evaluator is disabled (train_eval_disabled = true, equivalence verdict "
        "INCONCLUSIVE) on every MAMNet/DINOv3/OGLANet trained cell except those listed in "
        "task6.train_eval_enabled_cells; CrossEarth sidecars carry no train_eval field.",
        "OGLANet inference is nondeterministic (E-9). Every OGLANet cell is one inference draw; where an E-9 band "
        "exists it is reported with its pass count, and 3-pass bands are undersampled as gates (D-B4-3).",
        "Three D-A2-1 from-scratch retrains at a 200-epoch horizon exist for OGLANet "
        "(vanilla-loco OMA s0, fda-loco OMA s0, none-upper UCSD s0), each of whose originals stopped at the "
        "100-epoch ceiling. The primary analysis uses the inventory-named originals; task1.sensitivity_e200_retrain "
        "and task2.sensitivity_e200_oglanet_oma substitute the retrains.",
        "Two probes are failed-as-finding (D-B4-3): oglanet-bsm-loco-r3500-jax-hr-s1-p1histmatch and "
        "oglanet-isw-loco-r3500-oma-hr-s2-p1histmatch; each parent's stored score lies outside its own 10-pass "
        "E-9 range. Per operator ruling 2026-10-24 the probes are excluded from task 5 and the parents are kept "
        "in task 2; task2.sensitivity_failed_as_finding_parents_dropped gives the rows without them.",
        "mamnet/mrfp_plus peaks early (best_epoch listed in task2.mamnet_mrfp_plus_best_epochs); its seed spread "
        "reflects epoch-pick noise. Delta-side reading (a): the same shape appears on 5 of 6 internal cells.",
        "The SIB pooled worst-case criterion pools across access classes (MAMNet-SIB unlabeled_target; DINOv3-, "
        "OGLANet- and CrossEarth-SIB source_only), as ruled 2026-08-03; the per-architecture decomposition is "
        "disclosure. It is reported at two units, the (arch x fold) seed-mean level and the seed-resolved cell; "
        "neither is selected here.",
        "Holm families: task 1 over 12 (arch x fold) pairs; task 2 within each method family over its own cells "
        "(12 for sib/fda/bsm/sar, 9 otherwise); task 3 over 9 (arch x fold) cells and, separately, over 27 "
        "seed-resolved cells; task 5 over the 36 headline probes and within each method family.",
        "73 probe sidecars record target_access source_only although their parents are unlabeled_target "
        "(FDA, SegDeSiC, MAMNet-SIB). The probe's own access describes the matching statistic only; every task-5 "
        "row carries the parent's class.",
        "In task 4, city and fold are the same variable, so the sun_dist slope is identified within city only; "
        "the cluster-robust SEs have 3 clusters.",
        "p-bar_S (task 5 observational half) is not computed in this pass, per operator instruction.",
        "The combined internal + external histogram-matching count uses arm C's 0 of 26 usable cells from "
        "PAPER_STATE.md; that internal count is not recomputed here.",
        "CrossEarth is evaluated on vanilla, SIB, FDA, BSM and SAR only. FADA, IIM, MRFP+, SegDeSiC and ISW have "
        "no published CrossEarth implementation and could not be mapped onto its Mask2Former architecture "
        "without inventing a recipe; there is no published CrossEarth Experiment A.",
        "Exp A cells train a fixed 30 epochs (stopped_on epoch_ceiling, best_epoch null) and use held-out-city "
        "labels (target_labels).",
        "The project base is not a git repository on this box; git_hash is null. INDEX.parquet stores every scalar "
        "run.json field as a JSON string in run_json_scalars because field types differ across dump families.",
    ]

    res = dict(
        id="08_seo_full_grid", arm="A",
        answers=["v1UJ-W1", "BxRx-W1", "AC1", "v1UJ-W2", "v1UJ-W4", "zNBu-W2"],
        question=("On the full S-EO grid (4 architectures x 3 holdout cities x 3 seeds), what is the vanilla "
                  "LOCO gap, how does each adaptation method move vanilla-LOCO, what does Exp A recover, does "
                  "the gap track sun-elevation distance, and how many vanilla cells does histogram matching improve?"),
        supersedes=["results/02_seo_loco.json", "results/03_seo_methods.json"],
        runs=out["_runs"], index_path=INDEX_PATH, index_sha256=out["_index_sha256"],
        generated_utc=datetime.now(timezone.utc).isoformat(), script="analysis/08_seo_full_grid.py",
        git_hash=None, git_hash_note="the project base is not a git repository on this box; per-cell producing "
                                     "commits are in INDEX.parquet run_json_scalars.",
        seed=SEED, B=B,
        operator_rulings=[
            "2026-10-24: SAR enters as a within-dump delta (step-10 SAR pass minus in-dump unadapted pass), "
            "unseen-included population, labelled not commensurable.",
            "2026-10-24: failed-as-finding excludes the two PROBES from task 5; parents stay in task 2 with an "
            "exclusion sensitivity."],
        metric_definitions=dict(
            gap_2class_tol_k2_unseen_excluded="per patch, 50*(tp/(tp+fp+fn+1e-10) + tn/(tn+fp+fn+1e-10)) from "
                "tp_tol/fp_tol/fn_tol/tn_tol (K=2, DSM-unseen excluded), matching metrics.py _img_metrics; "
                "asserted equal to the stored miou_tol_pct to 1e-9 on every cell. Gap = upper - loco.",
            fg_only_tol_k2="100*tp/(tp+fp+fn+1e-10), asserted equal to 100*iou_tol.",
            seed_stratum="per-image delta averaged over the 3 seeds, then AOIs resampled; equivalent to resampling "
                         "AOIs and keeping every seed within each sampled AOI.",
            sun_dist="|patch sun_elevation - train_sun_elev_mean| of the LOCO cell; averaged over the 12 vanilla "
                     "LOCO cells of the fold (max per-patch spread in task4.sun_dist_spread_across_loco_cells)."),
        dry_run=dict(
            n_parquet=int(sum(1 for r in rows if r.get("has_parquet"))),
            assertion_summary=tab.groupby("category").agg(cells=("run_id", "size"), checks=("n_checks", "sum"),
                                                         failed_cells=("n_failed", lambda s: int((s > 0).sum())))
                                 .reset_index().to_dict("records"),
            sar_schema_failure="36 SAR dumps fail the arm-A column-set assertion (no aoi_id, sun_elevation, "
                               "sun_dist, n_valid_px, tp_tol..tn_str); 27 also lack metric_module_sha. "
                               "Run stopped; operator ruled 2026-10-24.",
            excluded_counts=[dict(category=k[0], reason=k[1], n=v) for k, v in sorted(excl.items(), key=str)],
            excluded_named=excluded_named),
        precommitted_rows=dict(
            row1=dict(text="REVIEWS.md arm A row 1 as signed: gap positive and significant in the large majority "
                           "of cells including clean-label JAX.",
                      n_positive_holm_sig_of_12=n_sig,
                      n_jax_positive_holm_sig_of_4=out["task1"]["n_jax_positive_holm_sig_of_4"]),
            row7=dict(text="REVIEWS.md arm A row 7: sun-elevation slope ns under fold-clustered SEs, point estimate "
                           "opposite the illumination account's required (positive) direction.",
                      conditions_met_on_pooled_model=out["_row7"],
                      n_models_slope_negative=out["task4"]["n_models_slope_negative"],
                      n_models_p_cluster_lt_005=out["task4"]["n_models_p_cluster_lt_005"]),
            note="Row designations as recorded on this box (REVIEWS.md uses numbered rows). The text of rows 2-6 "
                 "is not on this box. The reading of any row is the operator's and HLT's; none is applied here."),
        conditions=conditions, tests=tests,
        task1=out["task1"], task2=out["task2"], task3=out["task3"], task4=out["task4"], task5=out["task5"],
        task6=out["task6"],
        sanity_flags=flags,
        headline=headline, caveats=caveats, figure=None, caption=None)
    res = _clean(res)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nwrote {RESULTS_PATH}")
    print("HEADLINE:", headline)
    print("SIB pooled worst (seed-mean):", sib_wc.sib_worst_cell, round(sib_wc.difference, 4),
          "| seed-resolved:", sib_wc_s.sib_worst_cell, round(sib_wc_s.difference, 4))


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    if WRITE_ONLY_FROM_STATE:
        import pickle
        with open(STATE_PICKLE, "rb") as fh:
            st = pickle.load(fh)
        write_results(st["out"], catalogue(), st["tab"])
        sys.exit(0)
    rows = catalogue()
    if DRY_RUN:
        dry_run(rows)
        sys.exit(0)
    main(rows)
