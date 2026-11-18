#!/usr/bin/env python3
"""
results/39: results/36 (analysis/36_table2_vs_vanilla.py, copied here unchanged
except where marked "v2") with the CrossEarth SAR row added.

v2 CHANGES:
  * reads analysis/38_table2_final_v2.py (results/38) in place of analysis/35
    (results/35); every main-block cell, SAR/BSM row and CrossEarth row is the
    one results/38 uses, asserted as before.
  * CrossEarth SAR vs CrossEarth Vanilla is a fourth CrossEarth comparison; its
    row comes from analysis/38:resolve_ce_sar (sha256-pinned perstep.parquet).
  * Holm-Bonferroni over the CrossEarth comparisons is now over 4 (FDA, SIB,
    SAR, BSM), so the adjusted p of FDA, SIB and BSM change from results/36's
    3-test family; the keys are p_holm4 / sig_holm4.
  * every leaf of this JSON is compared against results/36 before writing
    (equality_vs_results_36).

ORIGINAL DOCSTRING (results/36) FOLLOWS.

results/36: every Table 2 adaptation method against Vanilla on the method's own
cell set, at K=2, with a paired image bootstrap per method.

INPUTS: exactly the cells results/35 uses, read through analysis/35's readers
(build_cells, load_vec, read_cell validation, the mid-res gate, the n450
resolution) and analysis/31's resolve_row for the SAR, BSM and CrossEarth rows.
Nothing is re-resolved here. Every pooled Delta is asserted equal to the value
results/35 wrote, and every dump read is asserted to carry the dump_sha256 that
results/35's cited index recorded.

QUANTITIES
  1. Vanilla pooled Delta over the 9 high-res cells (MAMNet, OGLANet, DINOv3 x
     3 cities) and over the 8 of them BSM has (DINOv3/Chicago excluded).
  2. Per adaptation method: pooled Delta on its own cell set and Vanilla's
     pooled Delta on the same set. FDA, SegDeSiC, IIM, ISW, MRFP+, FADA: the 18
     main-block cells; SAR: 9 high-res; BSM: 8 high-res.
  3. Paired test per method vs Vanilla. Per image, method LOCO minus Vanilla
     LOCO (= method Delta minus Vanilla Delta: the within-city upper cancels),
     paired on image_id inside each cell. Pooled statistic = mean over the
     cells of the cell-mean difference (= pooled Delta(method) - pooled
     Delta(Vanilla)).
     PRIMARY resampling: the test image. Within each (city, resolution) stratum
     the 150 test images are drawn with replacement ONCE per replicate and the
     draw is applied to every architecture's cell in that stratum, because the
     same test image enters the MAMNet, OGLANet and DINOv3 cells. B = 10,000,
     numpy default_rng(0) per test, strata drawn in sorted (city, res) order.
     Percentile CI 2.5/97.5; two-tailed p = min(2 * share of replicates on the
     far side of zero from the observed mean, 1): the published protocol's CI
     and p rules (analysis/35:boot_published).
     SECONDARY: analysis/35:boot_published applied verbatim to the flat vector
     of all per-image differences (every (cell, image) treated as independent).
     Holm-Bonferroni over the 8 methods (primary p) and, separately, over the 3
     CrossEarth comparisons.
     Per-cell: boot_published on each cell's 150 differences (grid + worst).
  4. CrossEarth: FDA, SIB, BSM vs CrossEarth Vanilla on its 3 high-res cells.

Run:  PYTHONNOUSERSITE=1 \\
      python \\
      analysis/39_table2_vs_vanilla_v2.py [--dry-run]
"""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
B = 10000
BOOT_SEED = 0
ALPHA = 0.05
NN, NAME = 39, "table2_vs_vanilla_v2"                                   # v2
RESULTS = os.path.join(BASE, "results", f"{NN}_{NAME}.json")
INV = os.path.join(BASE, "analysis", "_inv")
# v2: analysis/38 / results/38 stand where analysis/35 / results/35 stood. The
# names S35_SCRIPT, R35_JSON and S35 are kept so the code below reads as in
# analysis/36; they point at 38.
S35_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "38_table2_final_v2.py")
R35_JSON = os.path.join(BASE, "results", "38_table2_final_v2.json")
R36_JSON = os.path.join(BASE, "results", "36_table2_vs_vanilla.json")
EQ_TOL = 1e-9

MAIN_ADAPT = ("fda", "segdesic", "iim", "isw", "mrfp_plus", "fada")   # the request's order
EXTRA_ADAPT = ("sar", "bsm")
CE_ADAPT = ("fda", "sib", "sar", "bsm")                                 # v2: + SAR


def fail(msg):
    raise SystemExit(f"FATAL: {msg}")


def load_s35():
    spec = importlib.util.spec_from_file_location("s35", S35_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.PHASE != "B" or mod.B != B or mod.BOOT_SEED != BOOT_SEED:
        fail("analysis/35 config (PHASE/B/seed) is not the one results/35 was written with")
    return mod


S35 = load_s35()
R31 = S35.R31
CITIES, ARCHS = S35.CITIES, S35.ARCHS_APPF
MLABEL = S35.MLABEL


# ----------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------

def _p_ci(boots, obs):
    prop_wrong = float((boots <= 0).mean()) if obs >= 0 else float((boots >= 0).mean())
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return [lo, hi], min(2.0 * prop_wrong, 1.0)


def boot_image_cluster(strata, b=B, seed=BOOT_SEED):
    """strata: {(city, res): (n_cells_in_stratum, 150) array of per-image
    differences, all rows in one image_id order}. One index draw per stratum
    per replicate, shared by every cell of the stratum. Statistic: mean over
    all cells of the cell-mean difference."""
    rng = np.random.default_rng(seed)
    per_cell_boots, per_cell_obs = [], []
    for key in sorted(strata):
        D = strata[key]
        n = D.shape[1]
        idx = rng.integers(0, n, size=(b, n))
        per_cell_boots.append(D[:, idx].mean(axis=2))          # (n_cells, b)
        per_cell_obs.append(D.mean(axis=1))
    boots = np.concatenate(per_cell_boots, axis=0).mean(axis=0)
    obs = float(np.concatenate(per_cell_obs).mean())
    ci, p = _p_ci(boots, obs)
    return dict(mean=obs, boot_se=float(boots.std(ddof=1)), ci95=ci, p=p, B=b, seed=seed,
                two_tailed=True, n_cells=int(sum(D.shape[0] for D in strata.values())),
                n_images_per_cell=150, n_strata=len(strata),
                ci_method=f"paired_image_cluster_bootstrap_stratified_city_res_B{b}_percentile")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


# ----------------------------------------------------------------------------
# cell assembly
# ----------------------------------------------------------------------------

def series(v):
    return pd.Series(v["per_image"], index=v["image_id"]).sort_index()


def cell_record(label_key, m_loco, v_loco, up, ids_runs):
    """m_loco, v_loco, up: per-image Series on one image_id index."""
    for s in (v_loco, up):
        if list(s.index) != list(m_loco.index):
            fail(f"{label_key}: method, Vanilla and upper do not share the 150 image_ids")
    d = (m_loco - v_loco).to_numpy()
    bt = S35.boot_published(d)
    return dict(cell="/".join(label_key), **ids_runs,
                method_delta=float((m_loco - up).mean()),
                vanilla_delta=float((v_loco - up).mean()),
                diff_mean=bt["mean"], diff_stderr=bt["stderr"], n=bt["n"], ci95=bt["ci95"],
                p=bt["p"], B=bt["B"], seed=bt["seed"], two_tailed=True, ci_method=bt["ci_method"],
                sig=bool(bt["p"] < ALPHA)), d


def compare(method, cells_in):
    """cells_in: list of (arch, city, res, m_series, v_series, up_series, ids)."""
    recs, strata, flat = [], {}, []
    for arch, city, res, ms, vs, us, ids in cells_in:
        rec, d = cell_record((arch, method, city, res), ms, vs, us, ids)
        recs.append(rec)
        strata.setdefault((city, res), []).append(d)
        flat.append(d)
    # every cell of a stratum must be on one image order (already sorted by image_id)
    strata = {k: np.vstack(v) for k, v in strata.items()}
    prim = boot_image_cluster(strata)
    sec = S35.boot_published(np.concatenate(flat))
    m_pooled = float(np.mean([r["method_delta"] for r in recs]))
    v_pooled = float(np.mean([r["vanilla_delta"] for r in recs]))
    if abs((m_pooled - v_pooled) - prim["mean"]) > 1e-9 or abs(prim["mean"] - sec["mean"]) > 1e-9:
        fail(f"{method}: pooled difference disagrees between definitions")
    worst = min(recs, key=lambda r: r["diff_mean"])
    best = max(recs, key=lambda r: r["diff_mean"])
    return dict(
        n_cells=len(recs), cells=[r["cell"] for r in recs],
        method_pooled_delta=m_pooled, vanilla_pooled_delta=v_pooled,
        diff_pooled=prim["mean"],
        method_more_negative_than_vanilla=bool(m_pooled < v_pooled),
        test_primary=dict(name="paired image-cluster bootstrap, stratified by (city, resolution)",
                          compares=[method, "vanilla"], statistic=prim["mean"], **prim,
                          sig=bool(prim["p"] < ALPHA)),
        test_secondary_flat=dict(name="analysis/35:boot_published on the flat vector of all "
                                      "per-image differences", statistic=sec["mean"],
                                 stderr=sec["stderr"], n=sec["n"], ci95=sec["ci95"], p=sec["p"],
                                 B=sec["B"], seed=sec["seed"], two_tailed=True,
                                 sig=bool(sec["p"] < ALPHA)),
        per_cell=recs,
        per_cell_n_diff_negative=int(sum(r["diff_mean"] < 0 for r in recs)),
        per_cell_n_sig_worse=int(sum(r["sig"] and r["diff_mean"] < 0 for r in recs)),
        per_cell_n_sig_better=int(sum(r["sig"] and r["diff_mean"] > 0 for r in recs)),
        worst_cell=dict(cell=worst["cell"], diff=worst["diff_mean"], p=worst["p"]),
        best_cell=dict(cell=best["cell"], diff=best["diff_mean"], p=best["p"]))


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    if os.path.exists(RESULTS):
        fail(f"{os.path.relpath(RESULTS, BASE)} exists; never overwrite")
    r35 = json.load(open(R35_JSON))
    F, _ = S35.parse_appf()
    idx_png, info_png = R31.build_index("st_r450_published_png", S35.PNG)
    idx_rec, info_rec = R31.build_index("st_r450_published_recompute", S35.RECOMPUTE)
    print(f"index png {info_png['n_rows']} rows, recompute {info_rec['n_rows']} rows")

    # the dumps must be the ones results/35's cited indexes hashed
    old = {}
    for lab, v in r35["index_sha256_all"].items():
        p = os.path.join(BASE, v["path"])
        if sha256_file(p) != v["sha256"]:
            fail(f"{v['path']}: sha256 differs from results/35's record")
        old[lab] = pd.read_parquet(p).set_index("run_id").dump_sha256
    for lab, idx in (("st_r450_published_png", idx_png), ("st_r450_published_recompute", idx_rec)):
        cur = idx.set_index("run_id").dump_sha256
        common = sorted(set(cur.index) & set(old[lab].index))
        changed = [r for r in common if cur[r] != old[lab][r]]
        if changed:
            fail(f"{lab}: {len(changed)} dumps changed since results/35: {changed[:5]}")
        if set(cur.index) != set(old[lab].index):
            fail(f"{lab}: run_id set differs from results/35's index "
                 f"(+{sorted(set(cur.index) - set(old[lab].index))[:5]} "
                 f"-{sorted(set(old[lab].index) - set(cur.index))[:5]})")
    print("dump_sha256 of every indexed run equals results/35's cited indexes")

    if DRY_RUN:
        hr = idx_png[idx_png.run_id.str.contains("-hr-")]
        print(f"DRY RUN: PNG tree hr rows {len(hr)}; n_images {sorted(idx_png.n_images.unique())}; "
              f"recompute n_images {sorted(idx_rec.n_images.unique())}")
        sel = [S35.png_rid(a, m, c, r) for a in ARCHS for m in S35.METHODS for c in CITIES for r in S35.RESES]
        print(f"DRY RUN: main-block LOCO cells selected {len(sel)} (+24 n450 substitutions for hr), "
              f"in index {len(set(sel) & set(idx_png.run_id))}")
        df = pd.read_parquet(os.path.join(S35.PNG, sel[0], "perimage.parquet"))
        print(f"schema of {sel[0]} ({df.shape[0]} rows):")
        print(df.dtypes.to_string())
        print("DRY RUN: nothing written")
        return

    # ---- the main block, exactly as results/35 builds it --------------------
    cells, upper, excluded, gate_rows, _ = S35.build_cells(idx_png, idx_rec, F)
    if excluded:
        fail(f"results/35 excluded no main-block cell; this run excluded {excluded}")
    if len(cells) != 126 or len(upper) != 18:
        fail(f"cells {len(cells)} / upper {len(upper)} != 126 / 18")
    print(f"main block: {len(cells)} LOCO, {len(upper)} upper; mid-res gate "
          f"{sum(g['passed'] for g in gate_rows)}/{len(gate_rows)}")

    # ---- SAR / BSM / CrossEarth rows through analysis/31, as results/35 ------
    R = {}
    for method in ("upper", "vanilla", "sar", "bsm"):
        for arch in ARCHS:
            for city in CITIES:
                if method == "bsm" and (arch, city) in S35.BSM_EXCLUDED:
                    continue
                R[(method, arch, city)] = R31.resolve_row(method, arch, city)
    for method in ("upper", "vanilla") + tuple(m for m in CE_ADAPT if m != "sar"):
        for city in CITIES:
            R[(method, "crossearth", city)] = R31.resolve_row(method, "crossearth", city)
    for city in CITIES:                                                 # v2
        R[("sar", "crossearth", city)] = S35.resolve_ce_sar(city, R[("vanilla", "crossearth", city)])
    for k, r in R.items():
        if r.get("per_image") is None:
            fail(f"{k}: no per-image values ({r.get('blocker')})")

    # analysis/31's Vanilla/upper rows must be the very vectors of the main block
    for arch in ARCHS:
        for city in CITIES:
            for meth, mine in (("upper", upper[(arch, city, "hr")]),
                               ("vanilla", cells[(arch, "vanilla", city, "hr")])):
                r = R[(meth, arch, city)]
                if r["run_id"] != mine["run_id"]:
                    fail(f"{meth}/{arch}/{city}: analysis/31 resolves {r['run_id']} not {mine['run_id']}")
                dv = float(np.abs(series(r).to_numpy() - mine["k2"].to_numpy()).max())
                if list(series(r).index) != list(mine["k2"].index) or dv > EQ_TOL:
                    fail(f"{meth}/{arch}/{city}: analysis/31's per-image vector differs by {dv:.3e}")

    def up_hr(arch, city):
        return upper[(arch, city, "hr")]["k2"]

    out, runs = {}, set()

    # ---- 6 main-block methods on the 18 cells --------------------------------
    for m in MAIN_ADAPT:
        cin = []
        for arch in ARCHS:
            for city in CITIES:
                for res in S35.RESES:
                    mc, vc = cells[(arch, m, city, res)], cells[(arch, "vanilla", city, res)]
                    uc = upper[(arch, city, res)]
                    ids = dict(method_run_id=mc["run_id"], vanilla_run_id=vc["run_id"],
                               upper_run_id=uc["run_id"])
                    runs |= set(ids.values())
                    cin.append((arch, city, res, mc["k2"], vc["k2"], uc["k2"], ids))
        out[m] = compare(m, cin)

    # ---- SAR (9) and BSM (8), high-res ---------------------------------------
    for m in EXTRA_ADAPT:
        cin = []
        for arch in ARCHS:
            for city in CITIES:
                if (m, arch, city) not in R:
                    continue
                vc = cells[(arch, "vanilla", city, "hr")]
                ids = dict(method_run_id=R[(m, arch, city)]["run_id"], method_tree=R[(m, arch, city)]["tree"],
                           vanilla_run_id=vc["run_id"], upper_run_id=upper[(arch, city, "hr")]["run_id"])
                runs |= {ids["method_run_id"], ids["vanilla_run_id"], ids["upper_run_id"]}
                cin.append((arch, city, "hr", series(R[(m, arch, city)]), vc["k2"], up_hr(arch, city), ids))
        out[m] = compare(m, cin)
    if out["sar"]["n_cells"] != 9 or out["bsm"]["n_cells"] != 8:
        fail("SAR/BSM cell counts are not 9/8")

    # ---- item 1: Vanilla pooled Delta, 9 hr cells and BSM's 8 ----------------
    def v_hr(keep):
        ks = [(a, c) for a in ARCHS for c in CITIES if keep(a, c)]
        vals = [float((cells[(a, "vanilla", c, "hr")]["k2"] - up_hr(a, c)).mean()) for a, c in ks]
        return dict(pooled_delta=float(np.mean(vals)), n_cells=len(ks),
                    cells=[f"{a}/vanilla/{c}/hr" for a, c in ks],
                    worst_cell=dict(cell=f"{ks[int(np.argmin(vals))][0]}/vanilla/{ks[int(np.argmin(vals))][1]}/hr",
                                    delta=float(min(vals))))
    van9 = v_hr(lambda a, c: True)
    van8 = v_hr(lambda a, c: (a, c) not in S35.BSM_EXCLUDED)
    van18 = dict(pooled_delta=out["fda"]["vanilla_pooled_delta"], n_cells=18)
    for v, ref in ((van9["pooled_delta"], out["sar"]["vanilla_pooled_delta"]),
                   (van8["pooled_delta"], out["bsm"]["vanilla_pooled_delta"])):
        if abs(v - ref) > EQ_TOL:
            fail("Vanilla hr pooled Delta disagrees between item 1 and item 2")

    # ---- CrossEarth ----------------------------------------------------------
    ce = {}
    for m in CE_ADAPT:
        cin = []
        for city in CITIES:
            u, v, a = R[("upper", "crossearth", city)], R[("vanilla", "crossearth", city)], R[(m, "crossearth", city)]
            ids = dict(method_run_id=a["run_id"], vanilla_run_id=v["run_id"], upper_run_id=u["run_id"])
            runs |= set(ids.values())
            cin.append(("crossearth", city, "hr", series(a), series(v), series(u), ids))
        ce[m] = compare(m, cin)

    # ---- cross-check every pooled Delta against results/35 -------------------
    t2 = r35["table2"]
    chk = []
    for m in MAIN_ADAPT:
        chk.append((m, out[m]["method_pooled_delta"], t2["main_k2"]["rows"][m]["pooled"]["delta"]))
    chk.append(("vanilla18", van18["pooled_delta"], t2["main_k2"]["rows"]["vanilla"]["pooled"]["delta"]))
    chk.append(("sar", out["sar"]["method_pooled_delta"], t2["sar_hr"]["summary"]["pooled"]["delta"]))
    chk.append(("bsm", out["bsm"]["method_pooled_delta"], t2["bsm_hr"]["summary"]["pooled"]["delta"]))
    for m in ("vanilla",) + CE_ADAPT:
        mine = ce["fda"]["vanilla_pooled_delta"] if m == "vanilla" else ce[m]["method_pooled_delta"]
        chk.append((f"crossearth_{m}", mine, t2["crossearth_hr"]["rows"][m]["summary"]["pooled"]["delta"]))
    for name, mine, theirs in chk:
        if abs(mine - theirs) > EQ_TOL:
            fail(f"{name}: pooled Delta {mine} != results/35's {theirs}")
    print(f"pooled Delta cross-check against results/35: {len(chk)} of {len(chk)} equal to {EQ_TOL}")

    # ---- multiplicity --------------------------------------------------------
    order8 = MAIN_ADAPT + EXTRA_ADAPT
    h8 = holm({m: out[m]["test_primary"]["p"] for m in order8})
    h8s = holm({m: out[m]["test_secondary_flat"]["p"] for m in order8})
    h4 = holm({m: ce[m]["test_primary"]["p"] for m in CE_ADAPT})       # v2: 4-test family
    h4s = holm({m: ce[m]["test_secondary_flat"]["p"] for m in CE_ADAPT})
    for m in order8:
        out[m]["test_primary"].update(p_holm8=h8[m], sig_holm8=bool(h8[m] < ALPHA))
        out[m]["test_secondary_flat"].update(p_holm8=h8s[m], sig_holm8=bool(h8s[m] < ALPHA))
    for m in CE_ADAPT:
        ce[m]["test_primary"].update(p_holm4=h4[m], sig_holm4=bool(h4[m] < ALPHA))
        ce[m]["test_secondary_flat"].update(p_holm4=h4s[m], sig_holm4=bool(h4s[m] < ALPHA))

    more_neg = [m for m in order8 if out[m]["method_more_negative_than_vanilla"]]
    ce_more_neg = [m for m in CE_ADAPT if ce[m]["method_more_negative_than_vanilla"]]

    # ---- console table -------------------------------------------------------
    def fmtp(p):
        return "<.0002" if p == 0 else f"{p:.4f}"
    print(f"\n=== ITEM 1: Vanilla pooled Delta, K=2 ===")
    print(f"  9 hr cells: {van9['pooled_delta']:+.2f}   8 hr cells (BSM set): {van8['pooled_delta']:+.2f}   "
          f"18 cells: {van18['pooled_delta']:+.2f}")
    hdr = (f"{'method':10s} {'cells':>5s} {'method':>8s} {'vanilla':>8s} {'diff':>7s} {'95% CI (cluster)':>18s} "
           f"{'p':>7s} {'p_holm':>7s} | {'p_flat':>7s} {'CI flat':>17s} | cells<V  sig-/sig+  worst cell")
    print("\n=== ITEMS 2-3: method vs Vanilla on the method's own cell set ===")
    print(hdr)

    def row(m, o, hk):
        tp, ts = o["test_primary"], o["test_secondary_flat"]
        print(f"{MLABEL[m]:10s} {o['n_cells']:5d} {o['method_pooled_delta']:+8.2f} {o['vanilla_pooled_delta']:+8.2f} "
              f"{o['diff_pooled']:+7.2f} [{tp['ci95'][0]:+6.2f},{tp['ci95'][1]:+6.2f}]   {fmtp(tp['p']):>7s} "
              f"{fmtp(tp[hk]):>7s} | {fmtp(ts['p']):>7s} [{ts['ci95'][0]:+6.2f},{ts['ci95'][1]:+6.2f}] | "
              f"{o['per_cell_n_diff_negative']:2d}/{o['n_cells']:<2d}  {o['per_cell_n_sig_worse']:2d}/{o['per_cell_n_sig_better']:<2d}   "
              f"{o['worst_cell']['cell']} {o['worst_cell']['diff']:+.2f}")
    for m in order8:
        row(m, out[m], "p_holm8")
    print(f"n_adapt_more_negative_than_vanilla: {len(more_neg)} of 8: {more_neg}")
    print("\n=== ITEM 4: CrossEarth, method vs CrossEarth Vanilla (3 hr cells) ===")
    print(hdr)
    for m in CE_ADAPT:
        row(m, ce[m], "p_holm4")
    print(f"crossearth n_more_negative_than_vanilla: {len(ce_more_neg)} of {len(CE_ADAPT)}: {ce_more_neg}")

    # ---- sanity --------------------------------------------------------------
    sanity = []
    allc = {**{m: out[m] for m in order8}, **{f"crossearth_{m}": ce[m] for m in CE_ADAPT}}
    disagree = [k for k, o in allc.items() if o["test_primary"]["sig"] != o["test_secondary_flat"]["sig"]]
    sanity.append("Primary (image-cluster, stratified) and secondary (flat per-image vector) tests disagree "
                  f"on p<0.05 for: {disagree or 'none'}.")
    zero = [k for k, o in allc.items() if o["test_primary"]["p"] == 0]
    sanity.append(f"Comparisons with primary p = 0 (no replicate on the far side of zero; reported as <.0002): "
                  f"{zero or 'none'}.")
    sar_d = [c["diff_mean"] for c in out["sar"]["per_cell"]]
    sanity.append("SAR minus Vanilla per cell is SAR's within-pass change (analysis/31 builds the SAR row as "
                  "Vanilla's per-image values plus that change), so it is not a separate inference of Vanilla: "
                  f"per-cell differences {', '.join(f'{x:+.4f}' for x in sar_d)}.")
    same_as_r35 = r35["sec34"]["k2"]["n_adapt_more_negative_than_vanilla"]
    sanity.append(f"results/35 sec34 counts {same_as_r35} of 6 main-block methods below Vanilla's 18-cell pooled "
                  f"Delta; this job counts {sum(out[m]['method_more_negative_than_vanilla'] for m in MAIN_ADAPT)} "
                  "of the same 6.")
    near = [k for k, o in allc.items() if abs(o["diff_pooled"]) < 0.05]
    sanity.append(f"Pooled differences with |diff| < 0.05 mIoU: "
                  f"{[(k, round(allc[k]['diff_pooled'], 4)) for k in near] or 'none'}.")
    for s in sanity:
        print("SANITY:", s)

    # ---- write ---------------------------------------------------------------
    idx_paths = {}
    for label, idx in (("st_r450_published_png", idx_png), ("st_r450_published_recompute", idx_rec)):
        pth = os.path.join(INV, f"INDEX_{label}_{NN}.parquet")
        if os.path.exists(pth):
            fail(f"{os.path.relpath(pth, BASE)} exists; never overwrite a cited index")
        idx.to_parquet(pth + ".tmp.parquet", index=False)
        os.replace(pth + ".tmp.parquet", pth)
        idx_paths[label] = dict(path=os.path.relpath(pth, BASE), sha256=sha256_file(pth), n_rows=int(len(idx)))

    def cond(label, o, which):
        tp = o["test_primary"]
        return dict(label=label, metric="pooled_delta_diff_method_minus_vanilla_two_class_mIoU_tol_K2_pct",
                    mean=o["diff_pooled"], stderr=tp["boot_se"], stderr_kind="bootstrap SD of the pooled statistic",
                    n=o["n_cells"] * 150, n_cells=o["n_cells"], ci95=tp["ci95"],
                    ci_method=tp["ci_method"], baseline=which)

    conditions = [cond(f"{MLABEL[m]} minus Vanilla, {out[m]['n_cells']} cells", out[m], "vanilla")
                  for m in order8]
    conditions += [cond(f"CrossEarth {MLABEL[m]} minus CrossEarth Vanilla, 3 hr cells", ce[m], "crossearth vanilla")
                   for m in CE_ADAPT]
    tests = []
    for k, o in [(m, out[m]) for m in order8] + [(f"crossearth_{m}", ce[m]) for m in CE_ADAPT]:
        tp = o["test_primary"]
        tests.append(dict(name=tp["name"], compares=[k, "vanilla" if "crossearth" not in k else "crossearth_vanilla"],
                          statistic=tp["mean"], p=tp["p"],
                          p_holm=tp.get("p_holm8", tp.get("p_holm4")),
                          holm_family="8 adaptation methods" if "crossearth" not in k else "4 CrossEarth methods",
                          B=B, seed=BOOT_SEED, two_tailed=True, resampling_unit="test image (stratified by city x res; "
                          "one draw shared by every architecture's cell in a stratum)"))

    nsig = [m for m in order8 if out[m]["test_primary"]["sig_holm8"]]
    headline = (f"At K=2, {len(more_neg)} of 8 adaptation methods have a pooled Delta below Vanilla's on the same "
                f"cells, and {len(nsig)} of 8 method-minus-Vanilla pooled differences have Holm-adjusted p<0.05 "
                f"(paired image-cluster bootstrap, B=10,000); the differences range from "
                f"{min(out[m]['diff_pooled'] for m in order8):+.2f} ({MLABEL[min(order8, key=lambda m: out[m]['diff_pooled'])]}) "
                f"to {max(out[m]['diff_pooled'] for m in order8):+.2f} "
                f"({MLABEL[max(order8, key=lambda m: out[m]['diff_pooled'])]}) mIoU.")

    caveats = [
        "Metric: two-class boundary-tolerant mIoU at K=2, results/35's metric and cells exactly; every pooled Delta "
        "here equals results/35's to 1e-9 (asserted).",
        "Pooled Delta = mean over the cells of the cell-mean Delta (results/35's Table 2 definition). Every cell has "
        "150 test images, so it equals the mean over all images.",
        "The six main-block methods are compared on 18 cells (3 architectures x 3 cities x 2 resolutions); SAR on 9 "
        "high-res cells; BSM on 8 high-res cells. The three cell sets differ, so the Vanilla reference differs "
        f"by method: {van18['pooled_delta']:+.2f} (18), {van9['pooled_delta']:+.2f} (9), {van8['pooled_delta']:+.2f} "
        "(8). Pooled differences across these sets are not on one scale.",
        "BSM DINOv3/Chicago is excluded (training collapse at seed 0, best validation mIoU 5.51 against the codebase "
        "floor of 10.0; dump directory carries only .FAILED); Vanilla's 8-cell reference drops the same cell. Never "
        "imputed.",
        "Mid-resolution cells (in the 18-cell sets) are the published runs at their published training budget; the "
        "high-resolution MAMNet cells (all methods) and OGLANet-SegDeSiC are matched-450 retrains (results/35 "
        "caveat 2). Method and Vanilla always come from the same budget class within a cell for MAMNet; for OGLANet "
        "high-res, SegDeSiC is a matched-450 retrain and Vanilla is the published 450 cell.",
        "SAR minus Vanilla per image is SAR's within-pass change: analysis/31 builds each SAR row as Vanilla's per-"
        "image values plus the change measured inside SAR's own pass (step N minus step 0) at inference batch size "
        "1, while Vanilla is a batch-4 scored pass (results/31 caveat 4). The SAR test is a test of that change. "
        "OGLANet and DINOv3 SAR changes are measured on the published checkpoints (results/31 caveat 4b).",
        "Paired test: per image, method LOCO minus Vanilla LOCO on the same image_id; the within-city upper cancels. "
        "Primary resampling unit: the test image, drawn once per (city, resolution) stratum per replicate and shared "
        "by every architecture's cell in that stratum, because each test image enters the MAMNet, OGLANet and DINOv3 "
        "cells. High- and mid-resolution test images share no image_id, pair_id or source tile, so they are separate "
        "strata. Percentile CI; two-tailed p = min(2 x share of replicates on the far side of zero, 1); B = 10,000; "
        "default_rng(0) per test. The secondary test applies analysis/35's boot_published verbatim to the flat vector "
        "of all per-image differences, which treats the same image in three architectures as independent.",
        "Multiplicity: Holm-Bonferroni over the 8 adaptation-method comparisons, and separately over the 4 CrossEarth "
        "comparisons (FDA, SIB, SAR, BSM; results/36 used 3). Raw p is given beside every adjusted p. The per-cell tests are not multiplicity-corrected.",
        "p = 0 means no bootstrap replicate fell on the far side of zero; the resolution of p at B = 10,000 is "
        "0.0002, so p = 0 is reported as <.0002.",
        "The bootstrap resamples test images only. It does not include training-seed variation (one training seed per "
        "cell; no seed band exists for any cell) or OGLANet's pass-to-pass inference nondeterminism (REVIEWS.md E-9: "
        "cell-mean SD up to 0.4247 mIoU), which enters both the method and the Vanilla side of every OGLANet cell.",
        "CrossEarth: high-resolution only (3 cells); CrossEarth Vanilla is arm F's crossearth-vanilla-loco-r450-"
        "{city}-hr-s0, whose training draw differs from arm C's (results/35 caveat). The CrossEarth block is not "
        "pooled with the other architectures.",
        "No interpretation is attached to which differences are significant; the counts are of comparisons, not of "
        "independent methods' merit on unmatched cell sets.",
    ]

    J = dict(
        id=f"{NN}_{NAME}", arm="H2", answers=["H2", "E-4", "AC1"],
        answers_detail=("H2/E-4: sections/benchmark.tex:28 prints a count of adaptation methods with pooled Delta "
                        "below Vanilla's and the abstract says all adaptation methods degrade; this gives the count "
                        "over all 8 methods on matched cell sets with a paired test per method. AC1: per-method "
                        "significance and Holm adjustment for calibrated wording."),
        question=("At K=2, which adaptation methods have a pooled Delta below Vanilla's on the same cells, and which "
                  "method-minus-Vanilla differences are significant under a paired image bootstrap?"),
        status="COMPLETED",
        metric=r35["metric"],
        significance=dict(
            primary="paired image-cluster bootstrap, stratified by (city, resolution); statistic = mean over cells of "
                    "the cell-mean (method LOCO - Vanilla LOCO); B=10000; numpy default_rng(0) per test; strata drawn "
                    "in sorted (city, res) order; percentile CI 2.5/97.5; two-tailed p = min(2*prop_wrong, 1)",
            secondary="analysis/35_table2_final.py:boot_published on the concatenated per-image differences",
            per_cell="analysis/35_table2_final.py:boot_published on each cell's 150 per-image differences",
            multiplicity="Holm-Bonferroni over the 8 adaptation methods; separately over the 4 CrossEarth methods",
            B=B, seed=BOOT_SEED, alpha=ALPHA),
        vanilla_reference=dict(hr9=van9, hr8_bsm_set=van8, all18=van18),
        methods={m: out[m] for m in order8},
        n_adapt_more_negative_than_vanilla=len(more_neg),
        n_adapt_of=8,
        adapt_more_negative_than_vanilla=more_neg,
        n_sig_p05=sum(out[m]["test_primary"]["sig"] for m in order8),
        n_sig_holm8=len(nsig), sig_holm8=nsig,
        crossearth=dict(methods=ce, vanilla_pooled_delta=ce["fda"]["vanilla_pooled_delta"],
                        n_more_negative_than_vanilla=len(ce_more_neg), n_of=len(CE_ADAPT),
                        more_negative_than_vanilla=ce_more_neg,
                        sig_holm4=[m for m in CE_ADAPT if ce[m]["test_primary"]["sig_holm4"]]),
        crosscheck_results35=[dict(quantity=n, here=a, results35=b) for n, a, b in chk],
        conditions=conditions, tests=tests,
        runs=sorted(runs),
        runs_excluded=[dict(run_id="dinov3-bsm-loco-r450-chicago-hr-s0",
                            reason=S35.BSM_EXCLUDED[("dinov3", "chicago")])],
        index_sha256=idx_paths["st_r450_published_png"]["sha256"], index_sha256_all=idx_paths,
        index_note="rebuilt here; dump_sha256 of every run_id equals results/35's cited indexes (asserted)",
        inputs=dict(results38=dict(path="results/38_table2_final_v2.json", sha256=sha256_file(R35_JSON)),
                    script38=dict(path="analysis/38_table2_final_v2.py", sha256=sha256_file(S35_SCRIPT)),
                    results36=dict(path="results/36_table2_vs_vanilla.json", sha256=sha256_file(R36_JSON)),
                    script31=dict(path="analysis/31_internal_sib_worst_case_tgtfda.py",
                                  sha256=sha256_file(S35.R31_SCRIPT))),
        paper=dict(ref=S35.PAPER_REF, commit=S35.paper_commit()),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script=f"analysis/{NN}_{NAME}.py", script_sha256=sha256_file(os.path.abspath(__file__)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 conda_env="myenv (conda env)", pythonnousersite="1"),
        seed=BOOT_SEED, git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; only final_paper/ is git-connected "
                        "and this job does not touch it",
        compute_note="read-only CPU analysis on the analysis host; nothing under dumps/, final_paper/ or existing results/ "
                     "modified; writes this JSON and two new _inv index files",
        headline=headline, caveats=caveats, sanity_flags=sanity, figure=None, caption=None)
    # ---- v2 additions ----------------------------------------------------------
    r36 = json.load(open(R36_JSON))
    J["provenance"] = dict(crossearth_sar=r35["provenance"]["crossearth_sar"],
                           crossearth_sar_rows_from="analysis/38_table2_final_v2.py:resolve_ce_sar, re-run here; "
                                                    "pooled Delta asserted equal to results/38",
                           base=dict(script="analysis/36_table2_vs_vanilla.py",
                                     script_sha256=sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "36_table2_vs_vanilla.py")),
                                     results="results/36_table2_vs_vanilla.json",
                                     results_sha256=sha256_file(R36_JSON)))
    J["holm_crossearth_family_change"] = dict(
        rule="results/36 Holm-adjusted the CrossEarth comparisons over 3 (FDA, SIB, BSM); here over 4 (+ SAR)",
        rows={m: dict(p_raw=ce[m]["test_primary"]["p"],
                      p_holm3_results36=r36["crossearth"]["methods"][m]["test_primary"]["p_holm3"],
                      p_holm4_here=ce[m]["test_primary"]["p_holm4"],
                      sig_holm3_results36=r36["crossearth"]["methods"][m]["test_primary"]["sig_holm3"],
                      sig_holm4_here=ce[m]["test_primary"]["sig_holm4"])
              for m in CE_ADAPT if m != "sar"})
    rows_ce_sar = {c: R[("sar", "crossearth", c)] for c in CITIES}
    J["caveats"] = J["caveats"] + S35.ce_sar_caveats(rows_ce_sar) + [
        "CrossEarth Holm family is 4 here and 3 in results/36: the Holm-adjusted p of CrossEarth FDA, SIB and BSM "
        "change; both are in holm_crossearth_family_change.",
        f"Paper text read at {S35.PAPER_REF[:8]} (results/35's commit) through analysis/38; origin/main has since "
        "moved."]
    J = S35.jclean(J)
    J["equality_vs_results_36"] = S35.diff_record(r36, J, "results/36_table2_vs_vanilla.json", S35.SKIP_TOP)
    eq = J["equality_vs_results_36"]
    print(f"\nv2 EQUALITY vs results/36: {eq['n_equal']} of {eq['n_reference_leaves']} leaves equal; "
          f"changed {eq['n_changed']}, removed {eq['n_removed']}, added {eq['n_added']}")
    for c in eq["changed"]:
        print(f"  CHANGED {c['path']}: {str(c['old'])[:90]} -> {str(c['new'])[:90]}")
    for r_ in eq["removed"]:
        print(f"  REMOVED {r_}")
    print(f"  reference caveats absent verbatim: {eq['reference_caveats_absent_verbatim']}")
    S35.write_new(RESULTS, json.dumps(J, indent=1, allow_nan=False) + "\n")
    print(f"\nWROTE {os.path.relpath(RESULTS, BASE)}; {len(J['runs'])} runs; {len(caveats)} caveats")
    print(f"HEADLINE: {headline}")


if __name__ == "__main__":
    main()
