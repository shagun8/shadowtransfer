#!/usr/bin/env python3
"""
results/40: results/37 (analysis/37_loco_all_cells.py, copied here unchanged
except where marked "v2") with the CrossEarth SAR row added: the family is
158 cells (CrossEarth 15 = Vanilla, FDA, SIB, SAR, BSM x 3 cities).

v2 CHANGES:
  * reads analysis/38_table2_final_v2.py (results/38) in place of analysis/35
    (results/35); every main-block, SAR, BSM and CrossEarth cell is asserted
    equal to results/38's, as 37 asserted them against results/35.
  * the CrossEarth SAR cells come from analysis/38:resolve_ce_sar.
  * Bonferroni over the full family is over 158 (results/37: 155).
  * every leaf of this JSON is compared against results/37 before writing
    (equality_vs_results_37). The 155-literal strings and the
    sig_bonferroni155 key become the family size (sig_bonferroni_family).

ORIGINAL DOCSTRING (results/37) FOLLOWS.

results/37: significance counts over ALL LOCO cells in Table 2, at K=2.

FAMILY (155 cells, every one LOCO vs its own within-city upper bound):
  * the 126 main-block cells (3 architectures x 7 methods x 3 cities x 2
    resolutions), exactly results/35's cells via analysis/35:build_cells;
  * SAR, 9 high-res cells (MAMNet, OGLANet, DINOv3 x 3 cities);
  * BSM, 8 high-res cells (DINOv3/Chicago excluded as in results/35);
  * CrossEarth, 12 high-res cells (Vanilla, FDA, SIB, BSM x 3 cities) against
    CrossEarth's own upper bound.
  SAR, BSM and CrossEarth rows come from analysis/31:resolve_row through
  analysis/35:extra_row, as results/35 builds them.

PER CELL: analysis/35:cell_stats / extra_row, i.e. analysis/35:boot_published
on the cell's 150 paired per-image deltas (LOCO minus upper), B = 10,000,
numpy default_rng(0) re-seeded per cell, percentile CI, two-tailed
p = min(2 * prop_wrong, 1). Metric: analysis/35:two_class_k2.

REPORTED for the 155-cell family and separately for the 126-cell main block:
n_cells; n with p<0.05 by direction; n with p < 0.05/n_family (Bonferroni over
that family's size) by direction; mean Delta (mean over cells of the cell-mean
Delta); worst and best cell. The 126-cell block is asserted to equal results/35
sec34.k2 exactly (analysis/35:sec34 re-run on the recomputed stats, compared
after the same JSON cleaning), and every SAR/BSM/CrossEarth cell's mean, CI and
p are asserted equal to results/35's table2 rows.

Run:  PYTHONNOUSERSITE=1 \\
      python \\
      analysis/40_loco_all_cells_v2.py [--dry-run]
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
NN, NAME = 40, "loco_all_cells_v2"                                      # v2
RESULTS = os.path.join(BASE, "results", f"{NN}_{NAME}.json")
INV = os.path.join(BASE, "analysis", "_inv")
# v2: analysis/38 / results/38 stand where analysis/35 / results/35 stood (names kept).
S35_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "38_table2_final_v2.py")
R35_JSON = os.path.join(BASE, "results", "38_table2_final_v2.json")
R37_JSON = os.path.join(BASE, "results", "37_loco_all_cells.json")

EXTRA = ("sar", "bsm")
CE_METHODS = ("vanilla", "fda", "sib", "sar", "bsm")                     # v2: + SAR
N_EXPECTED = dict(main=126, sar=9, bsm=8, crossearth=15)                 # v2: 12 -> 15


def fail(msg):
    raise SystemExit(f"FATAL: {msg}")


def load_s35():
    spec = importlib.util.spec_from_file_location("s35", S35_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if mod.PHASE != "B" or mod.B != B or mod.BOOT_SEED != BOOT_SEED or mod.ALPHA != ALPHA:
        fail("analysis/35 config (PHASE/B/seed/alpha) is not the one results/35 was written with")
    return mod


S35 = load_s35()
R31 = S35.R31
CITIES, ARCHS = S35.CITIES, S35.ARCHS_APPF


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def summarise(rows):
    """rows: list of dicts with cell, mean, p. Bonferroni over len(rows)."""
    n = len(rows)
    bonf = ALPHA / n
    deltas = np.array([r["mean"] for r in rows])
    worst = min(rows, key=lambda r: r["mean"])
    best = max(rows, key=lambda r: r["mean"])
    sig = [r for r in rows if r["p"] < ALPHA]
    sigb = [r for r in rows if r["p"] < bonf]
    return dict(
        n_cells=n,
        n_sig_p05=len(sig),
        n_sig_p05_degrade=sum(r["mean"] < 0 for r in sig),
        n_sig_p05_improve=sum(r["mean"] > 0 for r in sig),
        not_sig_p05=[dict(cell=r["cell"], delta=r["mean"], p=r["p"]) for r in rows if r["p"] >= ALPHA],
        sig_p05_improve=[dict(cell=r["cell"], delta=r["mean"], p=r["p"]) for r in sig if r["mean"] > 0],
        bonferroni_family_size=n,
        bonferroni_threshold=bonf,
        n_sig_bonferroni=len(sigb),
        n_sig_bonferroni_degrade=sum(r["mean"] < 0 for r in sigb),
        n_sig_bonferroni_improve=sum(r["mean"] > 0 for r in sigb),
        sig_p05_not_bonferroni=[dict(cell=r["cell"], delta=r["mean"], p=r["p"])
                                for r in sig if r["p"] >= bonf],
        mean_delta=float(deltas.mean()),
        sd_delta_across_cells=float(deltas.std(ddof=1)),
        stderr_delta_across_cells=float(deltas.std(ddof=1) / np.sqrt(n)),
        n_delta_negative=int((deltas < 0).sum()),
        n_delta_positive=int((deltas > 0).sum()),
        n_p_zero=sum(r["p"] == 0.0 for r in rows),
        worst_cell=dict(cell=worst["cell"], delta=worst["mean"], ci95=worst["ci95"], p=worst["p"]),
        best_cell=dict(cell=best["cell"], delta=best["mean"], ci95=best["ci95"], p=best["p"]))


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
        if set(cur.index) != set(old[lab].index):
            fail(f"{lab}: run_id set differs from results/35's index")
        changed = [r for r in cur.index if cur[r] != old[lab][r]]
        if changed:
            fail(f"{lab}: {len(changed)} dumps changed since results/35: {changed[:5]}")
    print("dump_sha256 of every indexed run equals results/35's cited indexes")

    # ---- SAR / BSM / CrossEarth rows through analysis/31, as results/35 ------
    R = {}
    for method in ("upper", "vanilla") + EXTRA:
        for arch in ARCHS:
            for city in CITIES:
                if method == "bsm" and (arch, city) in S35.BSM_EXCLUDED:
                    R[(method, arch, city)] = dict(per_image=None, blocker=S35.BSM_EXCLUDED[(arch, city)])
                    continue
                R[(method, arch, city)] = R31.resolve_row(method, arch, city)
    for method in ("upper",) + tuple(m for m in CE_METHODS if m != "sar"):
        for city in CITIES:
            R[(method, "crossearth", city)] = R31.resolve_row(method, "crossearth", city)
    for city in CITIES:                                                 # v2
        R[("sar", "crossearth", city)] = S35.resolve_ce_sar(city, R[("vanilla", "crossearth", city)])

    if DRY_RUN:
        sel = [S35.png_rid(a, m, c, r) for a in ARCHS for m in S35.METHODS for c in CITIES for r in S35.RESES]
        print(f"DRY RUN: main-block LOCO cells selected {len(sel)} (24 hr cells take n450 retrains), "
              f"in PNG index {len(set(sel) & set(idx_png.run_id))}; n_images "
              f"{sorted(idx_png[idx_png.run_id.isin(sel)].n_images.unique())}")
        for method in EXTRA:
            ok = [(a, c, len(R[(method, a, c)]["per_image"])) for a in ARCHS for c in CITIES
                  if R[(method, a, c)].get("per_image") is not None]
            print(f"DRY RUN: {method}: {len(ok)} cells resolved {ok}")
        for method in CE_METHODS:
            ok = [(c, R[(method, 'crossearth', c)]["run_id"], len(R[(method, 'crossearth', c)]["per_image"]))
                  for c in CITIES if R[(method, "crossearth", c)].get("per_image") is not None]
            print(f"DRY RUN: crossearth/{method}: {len(ok)} cells {ok}")
        df = pd.read_parquet(os.path.join(S35.PNG, sel[0], "perimage.parquet"))
        print(f"schema of {sel[0]} ({df.shape[0]} rows):")
        print(df.dtypes.to_string())
        print("DRY RUN: nothing written")
        return

    # ---- the main block, exactly as results/35 builds it --------------------
    cells, upper, excluded, gate_rows, _ = S35.build_cells(idx_png, idx_rec, F)
    if excluded:
        fail(f"results/35 excluded no main-block cell; this run excluded {excluded}")
    if len(cells) != N_EXPECTED["main"] or len(upper) != 18:
        fail(f"cells {len(cells)} / upper {len(upper)} != 126 / 18")
    stats = {k: dict(S35.cell_stats(c, upper[(k[0], k[2], k[3])], "k2"),
                     run_id=c["run_id"], tree=c["tree"], source=c["source"],
                     upper_run_id=upper[(k[0], k[2], k[3])]["run_id"])
             for k, c in cells.items()}

    # exact check against results/35 sec34.k2: same function, same inputs
    upper_k2 = {k: float(v["k2"].mean()) for k, v in upper.items()}

    def keys_for(arch, method):
        return [(arch, method, c, r) for c in CITIES for r in S35.RESES if (arch, method, c, r) in stats]

    T = S35.aggregate(stats, upper_k2, keys_for)
    s34 = json.loads(json.dumps(S35.jclean(S35.sec34(stats, T)), allow_nan=False))
    sec34_equal = s34 == r35["sec34"]["k2"]
    if not sec34_equal:
        diff = {k: (s34.get(k), r35["sec34"]["k2"].get(k)) for k in set(s34) | set(r35["sec34"]["k2"])
                if s34.get(k) != r35["sec34"]["k2"].get(k)}
        fail(f"126-cell block does not equal results/35 sec34.k2 exactly: {diff}")
    print("126-cell sec34 recomputed here == results/35 sec34.k2 (exact, every field)")

    # analysis/31's Vanilla/upper hr rows must be the very cells of the main block
    for arch in ARCHS:
        for city in CITIES:
            if R[("upper", arch, city)]["run_id"] != upper[(arch, city, "hr")]["run_id"]:
                fail(f"upper/{arch}/{city}: analysis/31 resolves a different upper cell")
            if R[("vanilla", arch, city)]["run_id"] != cells[(arch, "vanilla", city, "hr")]["run_id"]:
                fail(f"vanilla/{arch}/{city}: analysis/31 resolves a different Vanilla cell")

    ups = {(a, c): R[("upper", a, c)] for a in ARCHS + ("crossearth",) for c in CITIES}
    extra = {m: S35.extra_row(m, ARCHS, R, ups) for m in EXTRA}
    ce = {m: S35.extra_row(m, ("crossearth",), R, ups) for m in CE_METHODS}

    # every SAR/BSM/CrossEarth cell must equal results/35's table2 rows exactly
    t35 = r35["table2"]
    n_extra_checked = 0
    for group, per, ref in ([(m, extra[m], t35[f"{m}_hr"]["per_cell"]) for m in EXTRA]
                            + [(f"crossearth/{m}", ce[m], t35["crossearth_hr"]["rows"][m]["per_cell"])
                               for m in CE_METHODS]):
        if set(per) != set(ref):
            fail(f"{group}: architectures {sorted(per)} != results/35 {sorted(ref)}")
        for arch, cs in per.items():
            if len(cs) != len(ref[arch]):
                fail(f"{group}/{arch}: {len(cs)} cells here, {len(ref[arch])} in results/35")
            for c, rc in zip(cs, ref[arch]):
                if c["city"] != rc["city"] or c.get("run_id") != rc.get("run_id"):
                    fail(f"{group}/{arch}/{c['city']}: cell identity differs from results/35")
                if c.get("loco") is None:
                    if rc.get("loco") is not None:
                        fail(f"{group}/{arch}/{c['city']}: missing here, present in results/35")
                    continue
                for f in ("mean", "p", "ci95", "loco", "upper"):
                    if json.loads(json.dumps(c[f])) != rc[f]:
                        fail(f"{group}/{arch}/{c['city']}: {f} {c[f]} != results/35 {rc[f]}")
                n_extra_checked += 1
    print(f"{n_extra_checked} SAR/BSM/CrossEarth cells equal results/35's table2 rows (mean, p, ci95, loco, upper)")

    # ---- per-cell rows --------------------------------------------------------
    rows, runs = [], set()
    for k, s in stats.items():
        rows.append(dict(cell="/".join(k), block="main", arch=k[0], method=k[1], city=k[2], res=k[3],
                         run_id=s["run_id"], upper_run_id=s["upper_run_id"], source=s["source"],
                         loco=s["loco"], upper=s["upper"], mean=s["mean"], stderr=s["stderr"],
                         n=s["n"], ci95=s["ci95"], p=s["p"]))
    for m, per in extra.items():
        for arch, cs in per.items():
            for c in cs:
                if c.get("loco") is None:
                    continue
                rows.append(dict(cell=f"{arch}/{m}/{c['city']}/hr", block=m, arch=arch, method=m,
                                 city=c["city"], res="hr", run_id=c["run_id"], upper_run_id=c["upper_run_id"],
                                 source=c["source_kind"], loco=c["loco"], upper=c["upper"], mean=c["mean"],
                                 stderr=c["stderr"], n=c["n"], ci95=c["ci95"], p=c["p"]))
    for m, per in ce.items():
        for c in per["crossearth"]:
            if c.get("loco") is None:
                fail(f"crossearth/{m}/{c['city']}: no per-image values ({c.get('blocker')})")
            rows.append(dict(cell=f"crossearth/{m}/{c['city']}/hr", block="crossearth", arch="crossearth",
                             method=m, city=c["city"], res="hr", run_id=c["run_id"],
                             upper_run_id=c["upper_run_id"], source=c["source_kind"], loco=c["loco"],
                             upper=c["upper"], mean=c["mean"], stderr=c["stderr"], n=c["n"],
                             ci95=c["ci95"], p=c["p"]))
    for r in rows:
        runs |= {r["run_id"], r["upper_run_id"]}
        if r["n"] != 150:
            fail(f"{r['cell']}: n = {r['n']}, expected 150")
    counts = {b: sum(r["block"] == b for r in rows) for b in N_EXPECTED}
    if counts != N_EXPECTED:
        fail(f"cell counts {counts} != {N_EXPECTED}")
    if len({r["cell"] for r in rows}) != len(rows):
        fail("duplicate cell labels")

    main_rows = [r for r in rows if r["block"] == "main"]
    fam = summarise(rows)
    mb = summarise(main_rows)
    mb_at_fam = dict(bonferroni_threshold=fam["bonferroni_threshold"],
                     n_sig=sum(r["p"] < fam["bonferroni_threshold"] for r in main_rows),
                     n_sig_degrade=sum(r["p"] < fam["bonferroni_threshold"] and r["mean"] < 0
                                       for r in main_rows))
    by_block = {b: summarise([r for r in rows if r["block"] == b]) for b in ("sar", "bsm", "crossearth")}

    # the 126 numbers requested, field by field against results/35 sec34.k2
    k2 = r35["sec34"]["k2"]
    mb_check = [
        ("n_cells", mb["n_cells"], k2["n_cells"]),
        ("n_sig_p05", mb["n_sig_p05"], k2["n_sig_p05"]),
        ("n_sig_p05_degrade", mb["n_sig_p05_degrade"], k2["n_sig_degrade_p05"]),
        ("bonferroni_threshold", mb["bonferroni_threshold"], k2["bonferroni_threshold"]),
        ("n_sig_bonferroni", mb["n_sig_bonferroni"], k2["n_sig_bonferroni"]),
        ("n_sig_bonferroni_degrade", mb["n_sig_bonferroni_degrade"], k2["n_sig_degrade_bonferroni"]),
        ("mean_delta", mb["mean_delta"], k2["mean_delta"]),
        ("worst_cell", mb["worst_cell"]["cell"], k2["worst_cell"]["cell"]),
        ("worst_delta", mb["worst_cell"]["delta"], k2["worst_cell"]["delta"]),
        ("best_cell", mb["best_cell"]["cell"], k2["best_cell"]["cell"]),
        ("best_delta", mb["best_cell"]["delta"], k2["best_cell"]["delta"]),
    ]
    bad = [c for c in mb_check if c[1] != c[2]]
    if bad:
        fail(f"126-cell numbers differ from results/35 sec34.k2: {bad}")
    print(f"126-cell requested numbers equal results/35 sec34.k2 exactly ({len(mb_check)} of {len(mb_check)})")

    for lab, s in ((f"FAMILY ({fam['n_cells']})", fam), ("MAIN BLOCK (126)", mb)) + tuple(
            (f"{b} only", by_block[b]) for b in by_block):
        print(f"\n=== {lab} ===")
        print(f"n_cells {s['n_cells']}; p<0.05: {s['n_sig_p05']} (degrade {s['n_sig_p05_degrade']}, "
              f"improve {s['n_sig_p05_improve']}); Bonferroni /{s['bonferroni_family_size']} "
              f"(p<{s['bonferroni_threshold']:.6f}): {s['n_sig_bonferroni']} (degrade "
              f"{s['n_sig_bonferroni_degrade']}, improve {s['n_sig_bonferroni_improve']})")
        print(f"mean Delta {s['mean_delta']:+.4f} (across-cell SD {s['sd_delta_across_cells']:.4f}, "
              f"SE {s['stderr_delta_across_cells']:.4f}); worst {s['worst_cell']['cell']} "
              f"{s['worst_cell']['delta']:+.4f}; best {s['best_cell']['cell']} {s['best_cell']['delta']:+.4f}")
        print(f"not sig at 0.05: {[(x['cell'], round(x['delta'], 3), x['p']) for x in s['not_sig_p05']]}")
        print(f"sig at 0.05 but not Bonferroni: "
              f"{[(x['cell'], round(x['delta'], 3), x['p']) for x in s['sig_p05_not_bonferroni']]}")
    print(f"\nmain block under the {fam['n_cells']}-family threshold: {mb_at_fam}")

    # ---- sanity --------------------------------------------------------------
    pvals = sorted({r["p"] for r in rows})
    sanity = [
        f"126-cell block equals results/35 sec34.k2 exactly: analysis/35:sec34 re-run on the recomputed stats "
        f"returns a dict equal field-for-field to the stored one, and the {len(mb_check)} requested quantities "
        f"match exactly; {n_extra_checked} SAR/BSM/CrossEarth cells equal results/35's table2 rows (mean, p, CI, "
        f"LOCO, upper).",
        f"{fam['n_p_zero']} of {fam['n_cells']} cells have p = 0 (no bootstrap replicate on the far side of zero). "
        f"At B = 10,000 the smallest non-zero p is 0.0002, so both Bonferroni thresholds (0.000397 at 126, "
        f"{fam['bonferroni_threshold']:.6f} at {fam['n_cells']}) admit exactly p in {{0, 0.0002}}; widening the family from 126 "
        f"to {fam['n_cells']} cannot change any main-block call at this B.",
        f"Distinct p values over the {fam['n_cells']} cells: {len(pvals)}; the smallest non-zero is "
        f"{min([p for p in pvals if p > 0], default=None)}.",
        f"Cells with Delta > 0: {[(r['cell'], round(r['mean'], 4), r['p']) for r in rows if r['mean'] > 0]}.",
    ]
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

    def cond(label, s, n_img):
        return dict(label=label, metric="mean_cell_delta_two_class_mIoU_tol_K2_pct",
                    mean=s["mean_delta"], stderr=s["stderr_delta_across_cells"],
                    stderr_kind="across-cell SD / sqrt(n_cells); descriptive, cells share test images",
                    n=n_img, n_cells=s["n_cells"], ci95=None,
                    ci_note="no family-level CI; per-cell paired-bootstrap CIs are in cells[]",
                    baseline="within-city upper bound (CrossEarth cells: CrossEarth's own upper)")

    conditions = [cond(f"All Table 2 LOCO cells ({fam['n_cells']})", fam, fam["n_cells"] * 150),
                  cond("Table 2 main block (126)", mb, mb["n_cells"] * 150)]
    conditions += [dict(label=r["cell"], metric="delta_two_class_mIoU_tol_K2_pct", mean=r["mean"],
                        stderr=r["stderr"], n=r["n"], ci95=r["ci95"],
                        ci_method=f"paired_image_bootstrap_B{B}", baseline=r["upper_run_id"])
                   for r in rows]
    tests = [dict(name="paired image bootstrap (analysis/35:boot_published)", compares=[r["run_id"], r["upper_run_id"]],
                  cell=r["cell"], statistic=r["mean"], p=r["p"], B=B, seed=BOOT_SEED, two_tailed=True,
                  sig_p05=r["p"] < ALPHA, sig_bonferroni_family=r["p"] < fam["bonferroni_threshold"],
                  sig_bonferroni126=(r["p"] < mb["bonferroni_threshold"]) if r["block"] == "main" else None)
             for r in rows]

    headline = (f"Over all {fam['n_cells']} Table 2 LOCO cells at K=2, {fam['n_sig_p05']} differ from their "
                f"within-city upper bound at p<0.05 ({fam['n_sig_p05_degrade']} below, {fam['n_sig_p05_improve']} "
                f"above; paired image bootstrap, B=10,000) and {fam['n_sig_bonferroni']} after Bonferroni over "
                f"{fam['n_cells']}, with mean Delta {fam['mean_delta']:.2f} mIoU, worst cell "
                f"{fam['worst_cell']['cell']} at {fam['worst_cell']['delta']:.2f} and best cell "
                f"{fam['best_cell']['cell']} at {fam['best_cell']['delta']:+.2f}.")

    caveats = [
        "Metric: two-class boundary-tolerant mIoU at K=2, results/35's metric and cells exactly (analysis/35 "
        "readers, two_class_k2 and boot_published imported, not reimplemented).",
        "The family mixes cell scopes: the 126 main-block cells cover 3 cities x 2 resolutions; SAR (9), BSM (8) and "
        f"CrossEarth ({counts['crossearth']}) are high-resolution only. {sum(r['res'] == 'hr' for r in rows)} of the "
        f"{fam['n_cells']} cells are high-resolution, {sum(r['res'] == 'mr' for r in rows)} mid-resolution.",
        "Mid-resolution cells (63) are the published runs at their published training budget; 24 high-resolution "
        "main-block cells (all MAMNet, plus OGLANet SegDeSiC) are matched-450 retrains (results/35 caveats).",
        "BSM DINOv3/Chicago is excluded (training collapse at seed 0, best validation mIoU 5.51 against the codebase "
        "floor of 10.0; dump directory carries only .FAILED), as in results/35. Never imputed.",
        "SAR cells are built as results/31 builds them: Vanilla's per-image values plus SAR's within-pass change at "
        "inference batch size 1 (results/31 caveat 4). The SAR cells therefore share Vanilla's per-image values up "
        "to that change and are not independent of the Vanilla cells.",
        "CrossEarth cells are compared against CrossEarth's own within-city upper bound "
        "(crossearth-vanilla-upper-r450-{city}-hr-s0); CrossEarth Vanilla is arm F's run, whose training draw "
        "differs from arm C's (results/35 caveat).",
        "Per-cell test: analysis/35:boot_published on the 150 paired per-image deltas (LOCO minus upper), B = "
        "10,000, numpy default_rng(0) re-seeded per cell, percentile CI, two-tailed p = min(2 x share of replicates "
        "on the far side of zero, 1). Significance counts are in either direction, split by sign.",
        f"Bonferroni: alpha 0.05 divided by the family size ({fam['n_cells']} for the full family, 126 for the main block). At "
        "B = 10,000 the p-value resolution is 0.0002, so both thresholds admit exactly p in {0, 0.0002}; the "
        "correction's outcome is bounded by bootstrap resolution, not by the family size.",
        "The cells are not independent tests: each test image enters every architecture's and method's cell for its "
        "city and resolution, and every LOCO cell for an (architecture, city, resolution) shares one upper-bound "
        "cell. Bonferroni is valid under dependence; it is conservative here.",
        "Mean Delta is the unweighted mean over cells of the cell-mean Delta (every cell has 150 images). Its "
        "across-cell SD/SE is descriptive; no family-level CI is given because the cells share test images and "
        "upper-bound cells.",
        "The bootstrap resamples test images only. It does not include training-seed variation (one training seed "
        "per cell) or OGLANet's pass-to-pass inference nondeterminism (REVIEWS.md E-9: cell-mean SD up to 0.4247 "
        "mIoU).",
        "p = 0 means no bootstrap replicate fell on the far side of zero; it is reported as <.0002.",
    ]

    J = dict(
        id=f"{NN}_{NAME}", arm="H2", answers=["H2", "E-4"],
        answers_detail=("H2: sections/benchmark.tex:28 prints '125 of 126 ... 123 after Bonferroni correction over "
                        "the 126-test family', mean -9.02 and worst -28.50, over the main block only, while the same "
                        "paragraph's 'every method ... on all four architectures' covers SAR, BSM and CrossEarth "
                        "cells. This gives the same counts over every LOCO cell Table 2 prints. E-4: per-cell sign "
                        "and significance over the full family."),
        question=("Over every LOCO cell in Table 2 (126 main block, SAR 9, BSM 8, CrossEarth 15), how many differ "
                  "from the within-city upper bound at p<0.05 and after Bonferroni, in which direction, and what "
                  "are the mean, worst and best Delta?"),
        status="COMPLETED",
        metric=r35["metric"],
        significance=dict(
            per_cell="analysis/35_table2_final.py:boot_published on each cell's 150 paired per-image deltas",
            B=B, seed=BOOT_SEED, seed_note="numpy default_rng(0), re-seeded for every cell", alpha=ALPHA,
            ci="percentile 2.5/97.5",
            p="two-tailed, min(2 * share of replicates on the far side of zero from the observed mean, 1)",
            multiplicity=f"Bonferroni over the family size: {fam['n_cells']} (full family), 126 (main block)"),
        family=fam,
        main_block=mb,
        main_block_under_family_threshold=mb_at_fam,
        by_extra_block=by_block,
        check_main_block_vs_results35_sec34_k2=dict(
            sec34_dict_equal=sec34_equal,
            fields=[dict(quantity=n, here=a, results35=b, equal=a == b) for n, a, b in mb_check],
            extra_cells_equal_results35=n_extra_checked),
        cells=rows,
        conditions=conditions, tests=tests,
        runs=sorted(runs),
        runs_excluded=[dict(run_id="dinov3-bsm-loco-r450-chicago-hr-s0",
                            reason=S35.BSM_EXCLUDED[("dinov3", "chicago")])],
        index_sha256=idx_paths["st_r450_published_png"]["sha256"], index_sha256_all=idx_paths,
        index_note="rebuilt here; dump_sha256 of every run_id equals results/35's cited indexes (asserted)",
        inputs=dict(results38=dict(path="results/38_table2_final_v2.json", sha256=sha256_file(R35_JSON)),
                    script38=dict(path="analysis/38_table2_final_v2.py", sha256=sha256_file(S35_SCRIPT)),
                    results37=dict(path="results/37_loco_all_cells.json", sha256=sha256_file(R37_JSON)),
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
    r37 = json.load(open(R37_JSON))
    J["provenance"] = dict(crossearth_sar=r35["provenance"]["crossearth_sar"],
                           crossearth_sar_rows_from="analysis/38_table2_final_v2.py:resolve_ce_sar, re-run here; "
                                                    "every CrossEarth SAR cell asserted equal to results/38",
                           base=dict(script="analysis/37_loco_all_cells.py",
                                     script_sha256=sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "37_loco_all_cells.py")),
                                     results="results/37_loco_all_cells.json", results_sha256=sha256_file(R37_JSON)))
    J["family_change_vs_results_37"] = dict(
        n_cells=dict(results37=r37["family"]["n_cells"], here=fam["n_cells"]),
        bonferroni_threshold=dict(results37=r37["family"]["bonferroni_threshold"], here=fam["bonferroni_threshold"]),
        cells_added=sorted({r["cell"] for r in rows} - {c["cell"] for c in r37["cells"]}),
        cells_removed=sorted({c["cell"] for c in r37["cells"]} - {r["cell"] for r in rows}),
        bonferroni_call_changed_on_existing_cells=sorted(
            c["cell"] for c in r37["cells"]
            if (c["p"] < r37["family"]["bonferroni_threshold"]) != (c["p"] < fam["bonferroni_threshold"])))
    rows_ce_sar = {c: R[("sar", "crossearth", c)] for c in CITIES}
    J["caveats"] = J["caveats"] + S35.ce_sar_caveats(rows_ce_sar) + [
        f"Family is {fam['n_cells']} cells here and {r37['family']['n_cells']} in results/37 (+ the 3 CrossEarth SAR "
        f"cells); Bonferroni threshold {fam['bonferroni_threshold']:.6g} here vs "
        f"{r37['family']['bonferroni_threshold']:.6g} in results/37.",
        f"Paper text read at {S35.PAPER_REF[:8]} (results/35's commit) through analysis/38; origin/main has since "
        "moved."]
    J = S35.jclean(J)
    J["equality_vs_results_37"] = S35.diff_record(r37, J, "results/37_loco_all_cells.json", S35.SKIP_TOP)
    eq = J["equality_vs_results_37"]
    print(f"\nv2 EQUALITY vs results/37: {eq['n_equal']} of {eq['n_reference_leaves']} leaves equal; "
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
