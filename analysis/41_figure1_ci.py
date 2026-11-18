#!/usr/bin/env python3
"""
results/41: 95% confidence intervals for Figure 1 (Vanilla, high resolution,
two-class boundary-tolerant mIoU at K=2), per architecture.

CELLS: exactly the cells behind results/38 figure1_data.per_arch.*.high_res_only
(equal to results/35's). For each of dinov3, mamnet, oglanet, crossearth and each
held-out city in (chicago, miami, phoenix): the within-city upper cell and the
Vanilla LOCO cell, as analysis/31's resolve_row resolves them (results/38 asserts
that resolver returns its own upper and Vanilla cells). MAMNet Vanilla LOCO is
the matched-450 retrain (st_r450_published_recompute, *-s42-n450); CrossEarth is
crossearth-vanilla-{upper,loco}-r450-{city}-hr-s0 in dumps/st (arm F).
Every cell is read through analysis/35's load_vec (CONTRACT section 3 columns
and dtypes, run.json sidecar, .DONE marker, n_images 150, K=2 recomputed from
tp/fp/fn/tn_tol through two_class_k2 and asserted against the stored columns).

STATISTIC per architecture: pooled upper mean, pooled LOCO mean, pooled
Delta = LOCO - upper; each the mean over the 3 cells of the cell mean (150
images per cell).

INTERVAL: paired image bootstrap, B = 10,000, numpy default_rng(0), percentile
2.5/97.5. Each replicate draws 150 positions with replacement within each city
(cities drawn in the order chicago, miami, phoenix, each as one (B, 150) block
of rng.integers(0, 150)), applied to the city's image_ids sorted ascending. The
same draw is used for upper, LOCO and Delta and for all four architectures; the
image_id lists are asserted identical across all 8 cells of a city.

Run:  PYTHONNOUSERSITE=1 \\
      python \\
      analysis/41_figure1_ci.py [--dry-run]
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
NN, NAME = 41, "figure1_ci"
ARCHS = ("dinov3", "mamnet", "oglanet", "crossearth")
CITIES = ("chicago", "miami", "phoenix")
N_PER_CELL = 150
POINT_TOL = 1e-6

R35_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "35_table2_final.py")
R38_JSON = os.path.join(BASE, "results", "38_table2_final_v2.json")
R35_JSON = os.path.join(BASE, "results", "35_table2_final.json")
INV = os.path.join(BASE, "analysis", "_inv")
RESULTS = os.path.join(BASE, "results", f"{NN}_{NAME}.json")

# The request's 2 dp values (results/38 figure1_data.high_res_only).
REQUEST_2DP = {"dinov3": (74.70, 68.37, -6.33), "mamnet": (79.72, 73.11, -6.61),
               "oglanet": (73.06, 65.27, -7.78), "crossearth": (78.79, 75.27, -3.52)}


def fail(msg):
    raise SystemExit(f"FATAL: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_r35():
    spec = importlib.util.spec_from_file_location("r35_table2", R35_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M35 = load_r35()
R31 = M35.R31
TREES = {"st_r450_published_png": R31.PNG,
         "st_r450_published_recompute": R31.RECOMPUTE,
         "st": R31.ST}


def expected_rid(arch, side, city):
    if arch == "crossearth":
        return f"crossearth-vanilla-{side}-r450-{city}-hr-s0"
    if side == "upper":
        return f"{arch}-none-upper-r450-{city}-hr-s0"
    if arch == "mamnet":
        return f"mamnet-vanilla-loco-r450-{city}-hr-s42-n450"
    return f"{arch}-vanilla-loco-r450-{city}-hr-s0"


def resolve_cells(indexes):
    """(arch, side, city) -> dict(run_id, tree, k2 Series indexed by image_id)."""
    cells = {}
    for arch in ARCHS:
        for city in CITIES:
            for side, method in (("upper", "upper"), ("loco", "vanilla")):
                row = R31.resolve_row(method, arch, city)
                if row.get("per_image") is None:
                    fail(f"{method}/{arch}/{city}: analysis/31 resolves no cell ({row.get('blocker')})")
                rid, tree = row["run_id"], row["tree"]
                if rid != expected_rid(arch, side, city):
                    fail(f"{method}/{arch}/{city}: resolved {rid}, expected {expected_rid(arch, side, city)}")
                v = M35.load_vec(TREES[tree], rid, indexes[tree], with_k5=False)
                k2 = v["k2"]
                if len(k2) != N_PER_CELL:
                    fail(f"{rid}: {len(k2)} image_ids, expected {N_PER_CELL}")
                r31 = pd.Series(np.asarray(row["per_image"], dtype=np.float64),
                                index=row["image_id"]).sort_index()
                if list(r31.index) != list(k2.index) or float(np.abs(r31.values - k2.values).max()) > 0:
                    fail(f"{rid}: analysis/31 resolve_row and analysis/35 load_vec disagree per image")
                ir = indexes[tree].set_index("run_id").loc[rid]
                if int(ir.n_images) != N_PER_CELL:
                    fail(f"{rid}: index n_images {ir.n_images}")
                cells[(arch, side, city)] = dict(run_id=rid, tree=tree, k2=k2,
                                                 dump_sha256=ir.dump_sha256,
                                                 target_access=v["meta"].get("target_access"),
                                                 stored_maxdev=v["stored_maxdev"])
    return cells


def city_ids(cells):
    ids = {}
    for city in CITIES:
        ref = list(cells[(ARCHS[0], "upper", city)]["k2"].index)
        for arch in ARCHS:
            for side in ("upper", "loco"):
                if list(cells[(arch, side, city)]["k2"].index) != ref:
                    fail(f"{arch}/{side}/{city}: image_ids differ from {ARCHS[0]}/upper/{city}")
        ids[city] = ref
    return ids


def draws():
    rng = np.random.default_rng(BOOT_SEED)
    return {city: rng.integers(0, N_PER_CELL, size=(B, N_PER_CELL)) for city in CITIES}


def pooled(cells, arch, side):
    """Point value and (B,) replicate vector of the 3-cell pooled mean."""
    return np.stack([cells[(arch, side, c)]["k2"].to_numpy(np.float64) for c in CITIES])


def summarise(x, reps):
    """x: (3, 150) per-image values (or per-image deltas); reps: (B,) pooled replicates."""
    obs = float(x.mean(axis=1).mean())
    lo, hi = (float(v) for v in np.percentile(reps, [2.5, 97.5]))
    se_strat = float(np.sqrt(sum(x[i].var(ddof=1) / x.shape[1] for i in range(3))) / 3.0)
    return dict(mean=obs, ci95=[lo, hi], stderr=se_strat,
                stderr_method="stratified analytic: sqrt(sum_city s_city^2 / 150) / 3",
                bootstrap_se=float(reps.std(ddof=1)), n=int(x.size), n_cells=3, n_per_cell=int(x.shape[1]),
                ci_method=f"paired_image_bootstrap_B{B}_stratified_by_city_percentile")


def main():
    print(f"DRY_RUN {DRY_RUN}  B {B}  seed {BOOT_SEED}")
    if os.path.exists(RESULTS) and not DRY_RUN:
        fail(f"{os.path.relpath(RESULTS, BASE)} exists; never overwrite a results JSON")

    indexes, idx_info = {}, {}
    for label, root in TREES.items():
        idx, info = R31.build_index(label, root)
        indexes[label] = idx
        idx_info[label] = {k: v for k, v in info.items() if k != "unindexable"}
        print(f"index {label}: {info['n_rows']} rows, {info['n_unindexable']} unindexable")

    cells = resolve_cells(indexes)
    ids = city_ids(cells)
    print(f"\n{'cell':52s} {'tree':28s} n    mean_k2   target_access")
    for (arch, side, city), c in cells.items():
        print(f"{c['run_id']:52s} {c['tree']:28s} {len(c['k2'])}  {c['k2'].mean():8.4f}  {c['target_access']}")
    print(f"image_ids identical across all 8 cells per city: "
          f"{', '.join(f'{c} {len(ids[c])}' for c in CITIES)}")

    # ---- point means vs results/38 (and results/35) ------------------------
    R38 = json.load(open(R38_JSON))["figure1_data"]["per_arch"]
    R35 = json.load(open(R35_JSON))["figure1_data"]["per_arch"]
    point_check = {}
    for arch in ARCHS:
        U, L = pooled(cells, arch, "upper"), pooled(cells, arch, "loco")
        got = (float(U.mean(axis=1).mean()), float(L.mean(axis=1).mean()),
               float((L - U).mean(axis=1).mean()))
        h38, h35 = R38[arch]["high_res_only"], R35[arch]["high_res_only"]
        ref = (h38["upper_mean"], h38["loco_mean"], h38["delta_mean"])
        ref35 = (h35["upper_mean"], h35["loco_mean"], h35["delta_mean"])
        diff = max(abs(a - b) for a, b in zip(got, ref))
        diff35 = max(abs(a - b) for a, b in zip(got, ref35))
        r2 = tuple(round(v, 2) for v in got)
        print(f"{arch:10s} upper {got[0]:.4f} loco {got[1]:.4f} delta {got[2]:+.4f} | "
              f"max|diff| vs 38 {diff:.2e}, vs 35 {diff35:.2e} | 2dp {r2} request {REQUEST_2DP[arch]}")
        if diff > POINT_TOL or diff35 > POINT_TOL:
            fail(f"{arch}: point means differ from results/38 or results/35 high_res_only by more than {POINT_TOL}")
        if any(abs(a - b) > 1e-9 for a, b in zip(r2, REQUEST_2DP[arch])):
            fail(f"{arch}: 2 dp point means {r2} differ from the request's {REQUEST_2DP[arch]}")
        point_check[arch] = dict(max_abs_diff_vs_results_38=diff, max_abs_diff_vs_results_35=diff35)

    if DRY_RUN:
        print("\nDRY RUN: schema of", cells[("crossearth", "loco", "chicago")]["run_id"])
        df = pd.read_parquet(os.path.join(R31.ST, cells[("crossearth", "loco", "chicago")]["run_id"],
                                          "perimage.parquet"))
        print(df.dtypes.to_string())
        print("DRY RUN: nothing written.")
        return

    # ---- bootstrap ----------------------------------------------------------
    D = draws()
    per_arch, conditions, tests, flags = {}, [], [], []
    for arch in ARCHS:
        U, L = pooled(cells, arch, "upper"), pooled(cells, arch, "loco")
        Dl = L - U
        rep = {}
        for name, X in (("upper", U), ("loco", L), ("delta", Dl)):
            rep[name] = np.mean([X[i][D[c]].mean(axis=1) for i, c in enumerate(CITIES)], axis=0)
        if float(np.abs(rep["delta"] - (rep["loco"] - rep["upper"])).max()) > 1e-9:
            fail(f"{arch}: Delta replicates are not LOCO minus upper replicates")
        out = {k: summarise(X, rep[k]) for k, X in (("upper", U), ("loco", L), ("delta", Dl))}
        obs = out["delta"]["mean"]
        pw = float((rep["delta"] <= 0).mean()) if obs >= 0 else float((rep["delta"] >= 0).mean())
        out["delta"]["p_two_tailed"] = min(2.0 * pw, 1.0)
        out["delta"]["p_note"] = ("published protocol: p = min(2 * share of replicates on the far side of "
                                  "zero, 1); p = 0 means no replicate crossed zero (resolution 2/B = 0.0002)")
        out["cells"] = {c: dict(upper=cells[(arch, "upper", c)]["run_id"],
                                loco=cells[(arch, "loco", c)]["run_id"],
                                upper_mean=float(cells[(arch, "upper", c)]["k2"].mean()),
                                loco_mean=float(cells[(arch, "loco", c)]["k2"].mean()))
                        for c in CITIES}
        per_arch[arch] = out
        for k in ("upper", "loco", "delta"):
            s = out[k]
            if not (s["ci95"][0] <= s["mean"] <= s["ci95"][1]):
                flags.append(f"{arch} {k}: CI {s['ci95']} excludes its point estimate {s['mean']:.4f}")
            conditions.append(dict(
                label=f"{M35.ARCH_LABEL[arch]} / Vanilla / high-res / {k}",
                metric=("two_class_tol_k2_mIoU" if k != "delta"
                        else "gap_two_class_tol_k2_mIoU (LOCO minus within-city upper)"),
                mean=s["mean"], stderr=s["stderr"], n=s["n"], ci95=s["ci95"], ci_method=s["ci_method"],
                baseline=("within-city upper bound, same architecture" if k != "upper" else None)))
        tests.append(dict(name="paired image bootstrap, stratified by city", arch=arch,
                          compares=["vanilla_loco_hr (3 cells)", "within_city_upper_hr (3 cells)"],
                          statistic=obs, p=out["delta"]["p_two_tailed"], B=B, seed=BOOT_SEED,
                          two_tailed=True))
        print(f"{arch:10s} upper {out['upper']['mean']:.2f} [{out['upper']['ci95'][0]:.2f}, {out['upper']['ci95'][1]:.2f}]  "
              f"loco {out['loco']['mean']:.2f} [{out['loco']['ci95'][0]:.2f}, {out['loco']['ci95'][1]:.2f}]  "
              f"delta {obs:+.2f} [{out['delta']['ci95'][0]:+.2f}, {out['delta']['ci95'][1]:+.2f}] "
              f"p {out['delta']['p_two_tailed']:.4f}  bootSE/analyticSE delta "
              f"{out['delta']['bootstrap_se']:.3f}/{out['delta']['stderr']:.3f}")

    # ---- indexes and run list ----------------------------------------------
    idx_paths = {}
    for label, idx in indexes.items():
        pth = os.path.join(INV, f"INDEX_{label}_{NN}.parquet")
        if os.path.exists(pth):
            fail(f"{os.path.relpath(pth, BASE)} exists; never overwrite a cited index")
        idx.to_parquet(pth + ".tmp.parquet", index=False)
        os.replace(pth + ".tmp.parquet", pth)
        idx_paths[label] = dict(path=os.path.relpath(pth, BASE), sha256=sha256_file(pth),
                                n_rows=int(len(idx)), tree=os.path.relpath(TREES[label], BASE))
    runs = sorted(c["run_id"] for c in cells.values())
    r38_runs = set(json.load(open(R38_JSON))["runs"])
    if not set(runs) <= r38_runs:
        fail(f"runs not in results/38's run list: {sorted(set(runs) - r38_runs)}")

    d = {a: per_arch[a]["delta"] for a in ARCHS}
    headline = ("Vanilla high-resolution Delta (LOCO minus within-city upper, K=2 two-class tolerant mIoU, "
                "3 cells x 150 images) is " + "; ".join(
                    f"{M35.ARCH_LABEL[a]} {d[a]['mean']:+.2f} [{d[a]['ci95'][0]:+.2f}, {d[a]['ci95'][1]:+.2f}]"
                    for a in ARCHS) + " (95% paired image bootstrap CI, B=10,000).")
    caveats = [
        "Metric: two-class boundary-tolerant mIoU at K=2 (per image, mean of foreground and background IoU "
        "from tp_tol/fp_tol/fn_tol/tn_tol; cell = mean over 150 images), as in results/35 and results/38.",
        "High-resolution cells only (3 held-out cities per architecture). These are results/38's "
        "figure1_data high_res_only values, not Table 2's 6-cell (high- plus mid-resolution) pooling.",
        "The interval resamples test images only (150 per city, stratified by city, same draw for every "
        "architecture). It does not include training-seed variation: one training seed per cell.",
        "One training seed per matched-450 cell; no seed band exists for the MAMNet Vanilla LOCO "
        "substitutions (mamnet-vanilla-loco-r450-{city}-hr-s42-n450).",
        "Every OGLANet cell is a single inference pass at the published batch size; OGLANet inference is "
        "nondeterministic (REVIEWS.md E-9: cell-mean SD up to 0.4247 mIoU over repeated passes), and that "
        "pass-to-pass variation is not in the interval.",
        "CrossEarth Vanilla comes from arm F (crossearth-vanilla-loco-r450-{city}-hr-s0), whose training "
        "draw differs from arm C's; CrossEarth is compared against its own upper bound "
        "(crossearth-vanilla-upper-r450-{city}-hr-s0).",
        "The MAMNet Vanilla LOCO cells are matched-450 retrains; the DINOv3 and OGLANet Vanilla LOCO cells "
        "and every non-CrossEarth upper cell are the published cells (st_r450_published_png).",
        "Intervals of different architectures are computed on the same bootstrap draws and are therefore "
        "correlated; overlap or non-overlap of two architectures' intervals is not a test of their difference.",
        "p for Delta follows the published protocol (min(2 x share of replicates past zero, 1)); p = 0 means "
        "no replicate crossed zero, printed as <.0002 at B = 10,000. No multiplicity correction is applied; "
        "the figure makes no claim that rests on the family of four.",
    ]
    for a in ARCHS:
        if d[a]["p_two_tailed"] > 0.05:
            caveats.append(f"{M35.ARCH_LABEL[a]} Delta interval includes zero (p {d[a]['p_two_tailed']:.4f}).")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    J = dict(
        id=f"{NN}_{NAME}", arm="H2", answers=["E-3"],
        answers_detail=("E-3 is the Figure 1 item in REVIEWS.md; this JSON supplies the 95% CIs the rebuilt Figure 1 "
                        "right panel and caption require (operator request). It does not by itself resolve E-3's "
                        "wording issue. Point values are results/38 figure1_data high_res_only; no new point "
                        "estimate is made."),
        question=("What are the 95% paired image bootstrap CIs of the Vanilla high-resolution upper, LOCO and "
                  "Delta per architecture behind Figure 1?"),
        per_arch=per_arch, point_check=point_check,
        bootstrap=dict(B=B, seed=BOOT_SEED, rng="numpy.random.default_rng(0)",
                       draw=("per replicate, 150 positions with replacement within each city; cities drawn in "
                             "order chicago, miami, phoenix, each one (B, 150) block of rng.integers(0, 150), "
                             "applied to the city's image_ids sorted ascending; one draw shared by upper, LOCO, "
                             "Delta and all four architectures"),
                       statistic="mean over the 3 cities of the resampled cell mean",
                       ci="percentile 2.5 / 97.5"),
        image_ids_per_city={c: len(ids[c]) for c in CITIES},
        conditions=conditions, tests=tests, runs=runs,
        cells=[dict(arch=a, side=s, city=c, run_id=v["run_id"], tree=v["tree"], dump_sha256=v["dump_sha256"],
                    target_access=v["target_access"], stored_maxdev=v["stored_maxdev"])
               for (a, s, c), v in cells.items()],
        index_sha256=idx_paths["st_r450_published_png"]["sha256"], index_sha256_all=idx_paths,
        index_info=idx_info, generated_utc=now, script=f"analysis/{NN}_{NAME}.py",
        script_sha256=sha256_file(os.path.abspath(__file__)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 conda_env="myenv (conda env)", pythonnousersite="1"),
        seed=BOOT_SEED, git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; only final_paper/ is git-connected "
                        "and this job does not touch it",
        provenance=dict(
            point_values_source=dict(path=os.path.relpath(R38_JSON, BASE), sha256=sha256_file(R38_JSON),
                                     field="figure1_data.per_arch.*.high_res_only"),
            also_checked=dict(path=os.path.relpath(R35_JSON, BASE), sha256=sha256_file(R35_JSON)),
            readers=dict(script=os.path.relpath(R35_SCRIPT, BASE), sha256=sha256_file(R35_SCRIPT),
                         functions=["load_vec", "two_class_k2"],
                         resolver="analysis/31_internal_sib_worst_case_tgtfda.py resolve_row",
                         resolver_sha256=sha256_file(R31.__file__ if hasattr(R31, "__file__") else M35.R31_SCRIPT))),
        headline=headline, caveats=caveats, sanity_flags=flags,
        figure=None, caption=None)
    txt = json.dumps(J, indent=1, allow_nan=False)
    with open(RESULTS + ".tmp", "w") as fh:
        fh.write(txt + "\n")
    os.replace(RESULTS + ".tmp", RESULTS)
    print(f"\nWROTE {os.path.relpath(RESULTS, BASE)} sha256 {sha256_file(RESULTS)[:8]}; {len(runs)} runs; "
          f"index {idx_paths['st_r450_published_png']['sha256'][:8]}; flags {len(flags)}")
    print(f"HEADLINE: {headline}")


if __name__ == "__main__":
    main()
