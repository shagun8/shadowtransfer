#!/usr/bin/env python
"""
Arm A -- SECTION 4.3 (SELECTIVE PREDICTION) ON THE EXTERNAL DATASET.

The §4.3 half of the paper's diagnosis has no external evidence. This script
supplies it from the S-EO dumps already in delta_dumps/, on the 72 vanilla
cells (36 source-only LOCO + 36 within-city upper; 4 architectures x 3 folds
x 3 seeds). Writes results/10_seo_selective_prediction.json.

  DEFINITION  the AURC definition is NOT re-derived. results/06_aurc_recovery
              ratified trapz19 (numpy trapezoid over linspace(0.10, 1.00, 19))
              and identified Fig 3(a)'s printed +0.31 as the TERMINAL POINT of
              the risk-coverage curve, not an integral. Both are asserted
              against results/06 before anything is computed, and 06's own
              provenance is cited by file and line in the output.

  DRY RUN     resolves the dump schema, selects the 72 cells from
              delta_dumps/INDEX.parquet (never a glob), prints their row
              counts, runs every schema / pairing / reconstruction gate, and
              writes nothing.
  FULL        tasks 1-6 below; writes only results/10_seo_selective_prediction
              .json. INDEX.parquet, results/08 and results/09 are read-only.

  TASK 1  per cell: the 19-point risk-coverage curve (fg, bg), trapz19 AURC,
          the full-coverage terminal point, and the CE-risk AURC
  TASK 2  per (arch, fold): fg, bg and fg-minus-bg SP gaps as paired per-image
          deltas, AOI bootstrap B=10,000 with seed as a stratum (08 task 1's
          own machinery), Holm over 12 within each family
  TASK 3  the same for the terminal point, labelled as the terminal point
  TASK 4  pooled per architecture and overall (08's AOI-unit scheme)
  TASK 5  internal-versus-external columns: four internal quantities from the
          countersigned citable set, beside their external analogues. Two
          columns, never merged; no cross-dataset test is computed.
  TASK 6  sectioned out: the adaptation-method cells, per-cell sign counts
          only, never pooled with the vanilla result.

No pre-committed branch is applied: no arm A row definitions exist in either
tree (REVIEWS.md provenance note, 2026-10-24). Nothing is said to replicate,
validate or confirm anything.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/10_seo_selective_prediction.py
"""

import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

DRY_RUN = "--dry-run" in sys.argv

BASE = os.environ["PROJECT_ROOT"]
TREE = os.path.join(BASE, "delta_dumps")
INDEX_PATH = os.path.join(TREE, "INDEX.parquet")
SCRIPT_08 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_seo_full_grid.py")
RESULTS_06 = os.path.join(BASE, "results", "06_aurc_recovery.json")
RESULTS_08 = os.path.join(BASE, "results", "08_seo_full_grid.json")
RESULTS_PATH = os.path.join(BASE, "results", "10_seo_selective_prediction.json")
COUNTERSIGNS = os.path.join(BASE, "summaries", "HLT_COUNTERSIGNS_armA_e12.md")
PAPER_STATE = os.path.join(BASE, "PAPER_STATE.md")
FER = os.path.join(BASE, "Final_Experiments_Results.md")
REVIEWS = os.path.join(BASE, "REVIEWS.md")

# the index sha results/08 and results/09 cite; this script must not rebuild it
EXPECTED_INDEX_SHA = "a14903907349e93b4392a16aa468ffb41a459b93ccbc82edd4af6a3650b1ae9d"
EXPECTED_METRICS_SHA = "730ad4a5bc039e3151599fb86bd411e2816c0897b68f1e4c17729335ab8a6e83"

ARCHS = ["mamnet", "dinov3", "oglanet", "crossearth"]
FOLDS = ["oma", "jax", "ucsd"]
SEEDS = [0, 1, 2]
N_TEST = {"oma": 1637, "jax": 2642, "ucsd": 1870}
UPPER_METHOD = "none"            # the upper-bound cells carry method=none
LOCO_METHOD = "vanilla"

MIN_POP = 50                     # metrics.per_image_rc returns None below this
RECOVER_TOL = 0.005              # internal F6's tolerance, mirrored
AURC_COLS = ["aurc_fg", "aurc_bg", "aurc_all", "aurc_ce_fg", "aurc_ce_bg"]

# ----------------------------------------------------------------------------
# loud failure, hashing, citation
# ----------------------------------------------------------------------------


def fail(msg):
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cite(path, needle, what):
    """First line of `path` containing `needle`, as 'path:line'. Fails loudly
    if the string is not there, so a citation cannot silently drift."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            if needle in line:
                return dict(cite=f"{os.path.relpath(path, BASE)}:{i}", quotes=needle, what=what)
    fail(f"citation gate: {os.path.relpath(path, BASE)} does not contain {needle!r} ({what})")


# ----------------------------------------------------------------------------
# 08's own machinery, imported rather than reimplemented
# ----------------------------------------------------------------------------

M8 = load_module("seo_full_grid_08", SCRIPT_08)
B, SEED = M8.B, M8.SEED
METRICS = load_module("shadowtransfer_metrics", M8.METRICS_PATH)
GRID = METRICS.COVERAGE_GRID

stream_of, ci95, holm = M8.stream_of, M8.ci95, M8.holm
aoi_boot = M8.aoi_cluster_bootstrap_imagemean
aoi_flip = M8.aoi_signflip_imagemean


def r6(x):
    return None if x is None else round(float(x), 6)


# ----------------------------------------------------------------------------
# the definition gate -- asserted against results/06, never re-derived
# ----------------------------------------------------------------------------


def definition_gate():
    d06 = json.load(open(RESULTS_06))
    checks = []

    def chk(name, ok, val):
        checks.append(dict(check=name, ok=bool(ok), value=str(val)))
        if not ok:
            fail(f"definition gate: {name} -> {val}")

    chk("results/06 verdict == RECOVERED", d06["verdict"] == "RECOVERED", d06["verdict"])
    pub = d06["published_definition_value"]
    chk("06 records the paper's stated AURC as A_trapz_0.10-1.00x19",
        pub["id"] == "A_trapz_0.10-1.00x19", pub["id"])
    var = {v["id"]: v for v in d06["variants"]}
    rec = var[pub["id"]]["recipe"]
    chk("06's trapz19 recipe: rule trapz", rec["rule"] == "trapz", rec["rule"])
    chk("06's trapz19 recipe: 19 points on 0.10-1.00",
        rec["n_points"] == 19 and rec["grid"].startswith("0.10-1.00 x19"), rec["grid"])
    chk("06's trapz19 recipe: 0/1 risk", rec["risk"] == "0/1", rec["risk"])
    chk("06's trapz19 recipe: confidence p on the GT-shadow population",
        rec["population"] == "gt_shadow_conf_p", rec["population"])
    chk("06's recovered published quantity is the terminal point",
        d06["recovered_recipe"]["id"] == "B_risk_at_1.00", d06["recovered_recipe"]["id"])

    # the module this script integrates with, against 06's own metric module
    msha = sha256_file(M8.METRICS_PATH)
    chk("metrics.py sha == the one 06/08/09 and every dump sidecar record",
        msha == EXPECTED_METRICS_SHA and d06["provenance"]["metric_module_sha256"] == msha, msha[:16])
    chk("metrics.COVERAGE_GRID == linspace(0.10, 1.00, 19)",
        GRID.shape == (19,) and np.allclose(GRID, np.linspace(0.10, 1.00, 19)), GRID.shape)
    rc_ones = np.ones(19, dtype=np.float32)
    chk("metrics.aurc_trapz19 is an unnormalised trapezoid over a span of 0.9",
        abs(METRICS.aurc_trapz19(rc_ones) - 0.9) < 1e-12, METRICS.aurc_trapz19(rc_ones))
    chk("metrics.per_image_rc drops a population below 50 px",
        "< 50: return None" in open(M8.METRICS_PATH).read().replace("<50", "< 50"), MIN_POP)

    prov = dict(
        definition="aurc_trapz19 = numpy trapezoid of the 19-point risk-coverage curve over "
                   "linspace(0.10, 1.00, 19), unnormalised (span 0.9)",
        ratified_by="results/06_aurc_recovery.json (HLT-ordered recovery, 2026-08-03); "
                    "PAPER_STATE.md records the ratification as conditional on that one attempt",
        citations=[
            cite(RESULTS_06, '"A_trapz_0.10-1.00x19"', "the ratified trapz19 definition id"),
            cite(RESULTS_06, '"B_risk_at_1.00"', "the terminal-point quantity Fig 3(a) printed"),
            cite(RESULTS_06, '"metric_module_sha256"', "06's metric module, same sha as this run's"),
            cite(RESULTS_06, "the terminal point of the risk-coverage curve",
                 "06's headline: the printed value is the terminal point, not the integral"),
            cite(RESULTS_06, "carries no ranking information",
                 "06's flagged property: the foreground 0/1 curve is a function of recall alone"),
            cite(PAPER_STATE, "trapz19 ratified (generator's own aurc()", "the ratification line"),
            cite(PAPER_STATE, "re-verify under trapz19 BEFORE any",
                 "the standing instruction that §4.3 claims re-verify under trapz19"),
            cite(COUNTERSIGNS, "Fig 3(a) prints +0.31 labeled", "the countersigned E-7 addendum"),
            cite(REVIEWS, "trapz19 family cannot reach it", "the E-7 addendum row"),
        ],
        upstream_generator=dict(
            file=d06["provenance"]["figure_code"],
            sha256=d06["provenance"]["figure_code_sha256"],
            printed_line=d06["recovered_recipe"]["source_line"],
        ),
        published_cell_values=dict(
            trapz19_gap=d06["published_definition_value"]["value"],
            terminal_point_gap=d06["recovered_recipe"]["value"],
            cell="mamnet / phoenix held out / foreground, internal (ShadowTransfer), n=129",
        ),
        ranking_quantity="per-pixel confidence: the predicted shadow probability p on the "
                         "GT-shadow (foreground) population, and 1 - p on the GT-background "
                         "population; pixels are ordered by decreasing confidence "
                         "(metrics.per_image_rc)",
        risk_quantity="0/1 error on the covered pixels: on the foreground population a pixel is "
                      "an error when p <= 0.5, on the background population when p > 0.5; "
                      "rc(c) = (errors among the top ceil(c n) by confidence) / ceil(c n)",
        terminal_point="rc(1.00), the full-coverage error rate on the population. This is the "
                       "quantity Fig 3(a) printed under a Delta-AURC label. It is never called "
                       "an AURC here.",
        ce_risk="metrics.per_image_rc_ce: the same confidence ranking with -log p (foreground) / "
                "-log(1-p) (background) as the risk; integrated by aurc_ce_trapz19. Stored "
                "per image as aurc_ce_fg / aurc_ce_bg; its curve is not reconstructible from "
                "the dump (see caveats).",
    )
    return d06, prov, checks


# ----------------------------------------------------------------------------
# curve reconstruction: exact, from the stored trapz19 and the GT population
# ----------------------------------------------------------------------------


def _k_of(n):
    return np.maximum(np.ceil(np.outer(np.asarray(n, np.float64), GRID)).astype(np.int64), 1)


def rc_of_m(n, m, dtype=np.float32):
    """The exact risk-coverage curve of a 0/1 risk whose confidence ranking puts
    every correct pixel ahead of every error -- results/06's flagged property of
    the foreground 0/1 curve, which holds on the background population too
    (conf = 1-p, error iff p > 0.5, so no correct pixel can outrank an error).
    n = population size, m = number of correct pixels. dtype float32 matches
    metrics.per_image_rc's own array dtype."""
    k = _k_of(n)
    return (np.maximum(0, k - np.asarray(m, np.int64)[:, None]) / k).astype(dtype)


def aurc_of_m(n, m, dtype=np.float32):
    return np.trapezoid(rc_of_m(n, m, dtype), GRID, axis=1)


def invert_aurc(n, stored, label):
    """Recover the integer correct-pixel count m from the stored trapz19 AURC.
    aurc_of_m is strictly decreasing in m on [0, n], so m is identified; the
    returned residual is checked for bit-equality by the caller."""
    n = np.asarray(n, np.int64)
    lo = np.zeros(len(n), np.int64)
    hi = n.copy()
    for _ in range(64):
        if np.all(lo >= hi):
            break
        mid = (lo + hi) // 2
        le = aurc_of_m(n, mid, np.float64) <= stored
        hi = np.where(le, mid, hi)
        lo = np.where(le, lo, mid + 1)
    m, res = lo, None
    for d in (-1, 0, 1):
        cand = np.clip(lo + d, 0, n)
        r = np.abs(aurc_of_m(n, cand) - stored)
        if res is None:
            m, res = cand, r
        else:
            take = r < res
            m, res = np.where(take, cand, m), np.where(take, r, res)
    # identification margin: how far the neighbouring integers sit from the fit.
    # Only in-range neighbours count; at m == 0 or m == n one neighbour does not
    # exist, and clipping it would report a spurious margin of zero.
    nb = np.full(len(n), np.inf)
    for d in (-1, 1):
        cand = m + d
        ok = (cand >= 0) & (cand <= n)
        a = np.abs(aurc_of_m(n, np.clip(cand, 0, n)) - stored)
        nb = np.where(ok, np.minimum(nb, a), nb)
    if (m < 0).any() or (m > n).any():
        fail(f"{label}: inverted correct-count outside [0, n]")
    return m, res, nb


# ----------------------------------------------------------------------------
# cell selection -- from INDEX.parquet and 08's catalogue, never a glob
# ----------------------------------------------------------------------------


def select(rows, idx):
    in_index = set(idx.run_id_dir)
    dup = (idx.groupby("run_id").config_sha256.nunique(dropna=False) > 1)
    if dup.any():
        fail(f"duplicate run_id with differing config_sha256: {list(dup[dup].index)}")
    sel, miss = {}, []
    for arch in ARCHS:
        for fold in FOLDS:
            for seed in SEEDS:
                for cond, meth in (("loco", LOCO_METHOD), ("upper", UPPER_METHOD)):
                    rec = M8.find_opt(rows, arch, meth, cond, fold, seed)
                    if rec is None:
                        miss.append((arch, meth, cond, fold, seed))
                        continue
                    if rec["run_id_dir"] not in in_index:
                        fail(f"{rec['run_id_dir']} is not a row of INDEX.parquet")
                    sel[(arch, fold, seed, cond)] = rec
    if miss:
        fail(f"incomplete vanilla grid, missing {miss}")
    if len(sel) != 72:
        fail(f"expected 72 vanilla cells, selected {len(sel)}")
    return sel


def access_gate(sel):
    out = []
    for key, rec in sorted(sel.items()):
        want = "source_only" if key[3] == "loco" else "target_labels"
        got = rec["target_access"]
        if got != want:
            fail(f"{rec['run_id_dir']}: target_access {got}, expected {want}")
        if rec["metric_module_sha"] != EXPECTED_METRICS_SHA:
            fail(f"{rec['run_id_dir']}: metric_module_sha {rec['metric_module_sha']}")
        out.append((rec["run_id_dir"], got))
    return out


# ----------------------------------------------------------------------------
# per-cell quantities
# ----------------------------------------------------------------------------

POP = {"fg": "aurc_fg", "bg": "aurc_bg"}
CE = {"fg": "aurc_ce_fg", "bg": "aurc_ce_bg"}


def cell_quantities(rec, fold, ref):
    """Validate the cell on read, reconstruct both 0/1 curves exactly, and
    return the per-image arrays aligned to the fold reference order."""
    df = M8.cell(rec)                      # 08's loader: alignment + recompute gates
    rid = rec["run_id_dir"]
    if len(df) != N_TEST[fold]:
        fail(f"{rid}: {len(df)} rows, n_test[{fold}] = {N_TEST[fold]}")
    for col in AURC_COLS:
        if col not in df.columns:
            fail(f"{rid}: missing {col}")
        if df[col].dtype != np.dtype("float64"):
            fail(f"{rid}: {col} dtype {df[col].dtype}, expected float64 (never coerced)")
    if not np.array_equal(df.n_gt_fg_valid.to_numpy(), ref["n_fg"]):
        fail(f"{rid}: GT foreground population differs from the fold reference")
    if not np.array_equal(df.n_valid_px.to_numpy(), ref["n_valid"]):
        fail(f"{rid}: valid-region size differs from the fold reference")

    out = dict(run_id=rid, n_rows=len(df))
    for pop in ("fg", "bg"):
        n = ref["n_" + pop]
        stored = df[POP[pop]].to_numpy(np.float64)
        defined = ~np.isnan(stored)
        if not np.array_equal(defined, ref["def_" + pop]):
            fail(f"{rid}: {POP[pop]} defined-set differs from the fold reference")
        if ((n >= MIN_POP) != defined).any():
            fail(f"{rid}: {POP[pop]} NaN pattern is not the <{MIN_POP}px rule")
        m, res, nb = invert_aurc(n[defined], stored[defined], f"{rid}/{pop}")
        exact = int((res == 0.0).sum())
        if res.max() > 0.0:
            fail(f"{rid}/{pop}: reconstruction not bit-exact, max residual {res.max():.3e} "
                 f"({len(res) - exact} of {len(res)} images). The terminal point would need the "
                 f"probability maps; stopping rather than substituting.")
        rc = rc_of_m(n[defined], m)                        # (n_img, 19), float32
        # tie the vectorised integral to the shared implementation, row by row
        probe = np.arange(0, len(rc), max(1, len(rc) // 64))
        for i in probe:
            if abs(METRICS.aurc_trapz19(rc[i]) - stored[defined][i]) > 0.0:
                fail(f"{rid}/{pop}: metrics.aurc_trapz19 disagrees with the reconstruction")
        full = np.full(len(df), np.nan)
        full[defined] = (n[defined] - m) / n[defined]       # rc(1.00), the terminal point
        rc_full = np.full((len(df), 19), np.nan)
        rc_full[defined] = rc
        out[pop] = dict(aurc=stored, terminal=full, rc=rc_full, correct=m,
                        n_defined=int(defined.sum()), residual_max=float(res.max()),
                        ident_margin_min=float(nb.min()), n_bit_exact=exact,
                        n_ident_collision=int((nb == 0.0).sum()),
                        n_at_ceiling=int((stored[defined] >= 0.9 - 1e-12).sum()),
                        ce=df[CE[pop]].to_numpy(np.float64))
    out["aurc_all"] = df.aurc_all.to_numpy(np.float64)
    M8._CACHE.clear()
    return out


def fold_reference(rows, fold):
    rec = M8.find(rows, LOCO_METHOD, "loco", fold, 0) if False else \
        M8.find(rows, "mamnet", LOCO_METHOD, "loco", fold, 0)
    df = M8.cell(rec)
    n_fg = df.n_gt_fg_valid.to_numpy(np.int64)
    n_valid = df.n_valid_px.to_numpy(np.int64)
    n_bg = n_valid - n_fg
    ref = dict(n_fg=n_fg, n_valid=n_valid, n_bg=n_bg,
               def_fg=n_fg >= MIN_POP, def_bg=n_bg >= MIN_POP,
               aoi=M8.FOLDREF[fold]["aoi"], image_id=df.image_id.to_numpy(),
               n_unseen=int((n_valid == 0).sum()), reference_cell=rec["run_id_dir"])
    ref["def_both"] = ref["def_fg"] & ref["def_bg"]
    M8._CACHE.clear()
    return ref


# ----------------------------------------------------------------------------
# paired statistics -- 08 task 1's scheme, restricted to a defined population
# ----------------------------------------------------------------------------


def paired(delta_by_seed, fold, label, mask, ref):
    """08.paired_stats with the population restricted to `mask`. The mask is a
    GT property, identical in both conditions and in every cell of the fold, so
    the per-image pairing is preserved exactly."""
    seeds = sorted(delta_by_seed)
    if seeds != SEEDS:
        fail(f"{label}: seeds {seeds}")
    dbar = np.mean([delta_by_seed[s][mask] for s in seeds], axis=0)
    if not np.isfinite(dbar).all():
        fail(f"{label}: non-finite delta inside the defined population")
    codes, uniq = pd.factorize(ref["aoi"][mask])
    n_aoi = len(uniq)
    boot = aoi_boot(dbar, codes, n_aoi, stream_of("boot|" + label))
    obs, p = aoi_flip(dbar, codes, n_aoi, stream_of("flip|" + label))
    lo, hi = ci95(boot)
    per_seed = {int(s): float(delta_by_seed[s][mask].mean()) for s in seeds}
    ps = np.array(list(per_seed.values()))
    return dict(mean=obs, stderr=float(boot.std(ddof=1)), ci_lo=lo, ci_hi=hi, p_raw=p,
                n_seeds=len(seeds), per_seed=per_seed, seed_sd=float(ps.std(ddof=1)),
                n_images=int(mask.sum()), n_aoi=n_aoi,
                ci_excludes_point=bool(not (lo <= obs <= hi)), _dbar=dbar)


def aoi_units(dbar, fold, mask, ref):
    return (pd.DataFrame({"aoi_id": ref["aoi"][mask], "gap": dbar})
            .groupby("aoi_id", as_index=False)
            .agg(gap=("gap", "mean"), n_img=("gap", "size"))
            .assign(fold=fold))


def tag(d, extra=None):
    out = {k: (r6(v) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("_")}
    out["per_seed"] = {k: r6(v) for k, v in d["per_seed"].items()}
    if extra:
        out.update(extra)
    return out


# ----------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------


def dry_run(rows, idx, sel, refs, prov_checks):
    nimg = dict(zip(idx.run_id_dir, idx.n_images))
    print("=" * 108)
    print("ARM A / S-EO SELECTIVE PREDICTION (section 4.3 external) -- DRY RUN")
    print(f"tree  : {TREE}")
    print(f"index : {INDEX_PATH}  ({len(idx)} rows)")
    print("=" * 108)

    print("\n-- definition gate (results/06) " + "-" * 74)
    for c in prov_checks:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']:<72} {c['value']}")

    print("\n-- resolved dump schema (reference cell) " + "-" * 66)
    ref_rec = M8.find(rows, "mamnet", LOCO_METHOD, "loco", "oma", 0)
    df = M8.load_parquet(ref_rec)
    print(f"  {ref_rec['dir']}  ({len(df.columns)} columns, {len(df)} rows)")
    for col in ["image_id", "aoi_id", "n_valid_px", "n_gt_fg_valid", "tp_tol", "fn_tol"] + AURC_COLS:
        print(f"    {col:<16} {str(df[col].dtype):<10} "
              f"nulls {int(df[col].isna().sum()):>5}")
    M8._CACHE.clear()

    print("\n-- cells selected (72 = 4 arch x 3 folds x 3 seeds x 2 conditions) " + "-" * 40)
    print(f"  {'run_id':<46} {'rows':>6} {'fg_def':>7} {'bg_def':>7} {'both':>6} {'access':<16} band")
    for key in sorted(sel):
        arch, fold, seed, cond = key
        rec = sel[key]
        ref = refs[fold]
        band = M8.e9_band(rec)
        print(f"  {rec['run_id_dir']:<46} {nimg[rec['run_id_dir']]:>6} {int(ref['def_fg'].sum()):>7} "
              f"{int(ref['def_bg'].sum()):>7} {int(ref['def_both'].sum()):>6} "
              f"{rec['target_access']:<16} "
              f"{'-' if band is None else str(band['n_passes']) + 'pass'}")

    print("\n-- fold reference populations " + "-" * 77)
    for fold in FOLDS:
        r = refs[fold]
        print(f"  {fold:<6} n_test {len(r['image_id']):>5}  AOIs {M8.FOLDREF[fold]['n_aoi']:>4}  "
              f"fg>=50px {int(r['def_fg'].sum()):>5}  bg>=50px {int(r['def_bg'].sum()):>5}  "
              f"both {int(r['def_both'].sum()):>5}  n_valid_px==0 {r['n_unseen']:>4}  "
              f"ref {r['reference_cell']}")

    print("\n-- reconstruction gate on every cell (bit-exact or stop) " + "-" * 50)
    worst = 0.0
    for key in sorted(sel):
        arch, fold, seed, cond = key
        q = cell_quantities(sel[key], fold, refs[fold])
        for pop in ("fg", "bg"):
            worst = max(worst, q[pop]["residual_max"])
        print(f"  {q['run_id']:<46} fg resid {q['fg']['residual_max']:.1e} "
              f"margin {q['fg']['ident_margin_min']:.2e} ceil {q['fg']['n_at_ceiling']:>4} | "
              f"bg resid {q['bg']['residual_max']:.1e} margin {q['bg']['ident_margin_min']:.2e}")
    print(f"\n  worst residual over all 72 cells and both populations: {worst:.3e}")
    print("\nDRY RUN complete. Nothing written.")


# ----------------------------------------------------------------------------
# full pass
# ----------------------------------------------------------------------------


def main():
    if sha256_file(INDEX_PATH) != EXPECTED_INDEX_SHA:
        fail(f"INDEX.parquet sha {sha256_file(INDEX_PATH)[:12]} != the sha 08/09 cite "
             f"{EXPECTED_INDEX_SHA[:12]}; this script must not rebuild it")
    idx = pd.read_parquet(INDEX_PATH)
    d06, prov, prov_checks = definition_gate()

    rows = M8.catalogue()
    M8.build_foldref(rows)
    sel = select(rows, idx)
    access_gate(sel)
    refs = {f: fold_reference(rows, f) for f in FOLDS}

    if DRY_RUN:
        dry_run(rows, idx, sel, refs, prov_checks)
        return

    print(f"index sha {EXPECTED_INDEX_SHA[:16]}... | metrics sha {EXPECTED_METRICS_SHA[:16]}... "
          f"| B = {B} | seed = {SEED}")

    # ---------------- task 1: per cell -------------------------------------
    Q, cells = {}, []
    for key in sorted(sel):
        arch, fold, seed, cond = key
        q = cell_quantities(sel[key], fold, refs[fold])
        Q[key] = q
        ref = refs[fold]
        row = dict(arch=arch, fold=fold, seed=seed, condition=cond, run_id=q["run_id"],
                   n_images=q["n_rows"])
        for pop in ("fg", "bg"):
            d = q[pop]
            ok = ~np.isnan(d["aurc"])
            row[f"aurc_trapz19_{pop}"] = r6(np.nanmean(d["aurc"]))
            row[f"terminal_point_{pop}"] = r6(np.nanmean(d["terminal"]))
            row[f"aurc_ce_trapz19_{pop}"] = r6(np.nanmean(d["ce"]))
            row[f"n_images_{pop}"] = d["n_defined"]
            row[f"rc_curve_{pop}"] = [r6(v) for v in np.nanmean(d["rc"], axis=0)]
            row[f"n_at_aurc_ceiling_{pop}"] = d["n_at_ceiling"]
            row[f"reconstruction_residual_max_{pop}"] = d["residual_max"]
            row[f"reconstruction_identification_margin_min_{pop}"] = float(
                f"{d['ident_margin_min']:.3e}")
            row[f"reconstruction_identification_collisions_{pop}"] = d["n_ident_collision"]
        row["aurc_trapz19_all_pixels"] = r6(np.nanmean(q["aurc_all"]))
        row["aurc_trapz19_fg_minus_bg"] = r6(row["aurc_trapz19_fg"] - row["aurc_trapz19_bg"])
        row["terminal_point_fg_minus_bg"] = r6(row["terminal_point_fg"] - row["terminal_point_bg"])
        band = M8.e9_band(sel[key])
        row["e9_band"] = None if band is None else dict(
            file=band["file"], n_passes=band["n_passes"],
            miou_pct_sd=r6(band["sd_miou_pct"]), scope="two-class mIoU only, not any AURC quantity")
        cells.append(row)
        print(f"  cell {q['run_id']:<46} fg AURC {row['aurc_trapz19_fg']:.4f} "
              f"bg {row['aurc_trapz19_bg']:.6f} terminal fg {row['terminal_point_fg']:.4f}")

    # ---------------- tasks 2 and 3: per (arch, fold) ----------------------
    FAMILIES = [
        ("sp_gap_fg", "aurc", "fg", "both"),
        ("sp_gap_bg", "aurc", "bg", "both"),
        ("sp_gap_asymmetry", "aurc", "asym", "both"),
        ("terminal_gap_fg", "terminal", "fg", "both"),
        ("terminal_gap_bg", "terminal", "bg", "both"),
        ("terminal_gap_asymmetry", "terminal", "asym", "both"),
        ("sp_gap_ce_fg", "ce", "fg", "both"),
        ("sp_gap_fg_own_mask", "aurc", "fg", "fg"),
        ("sp_gap_bg_own_mask", "aurc", "bg", "bg"),
    ]

    def delta(key_l, key_u, kind, which):
        """LOCO minus upper, per image. Positive = LOCO carries the higher risk,
        matching CONVENTIONS.md's SP gap direction."""
        def get(k, pop):
            d = Q[k][pop]
            return d["aurc"] if kind == "aurc" else d["terminal"] if kind == "terminal" else d["ce"]
        if which == "asym":
            return (get(key_l, "fg") - get(key_u, "fg")) - (get(key_l, "bg") - get(key_u, "bg"))
        return get(key_l, which) - get(key_u, which)

    pair_rows, pooled_rows = [], []
    for fam, kind, which, maskname in FAMILIES:
        stats, keys = [], []
        for arch in ARCHS:
            for fold in FOLDS:
                mask = refs[fold]["def_" + maskname]
                dbs = {s: delta((arch, fold, s, "loco"), (arch, fold, s, "upper"), kind, which)
                       for s in SEEDS}
                st = paired(dbs, fold, f"{fam}|{arch}|{fold}", mask, refs[fold])
                stats.append(st)
                keys.append((arch, fold))
        padj = holm([s["p_raw"] for s in stats])
        for (arch, fold), st, pa in zip(keys, stats, padj):
            pair_rows.append(dict(family=fam, quantity=kind, population=which, arch=arch, fold=fold,
                                  **tag(st, dict(p_holm=r6(pa),
                                                 holm_family=f"{fam}: 12 (arch x fold) pairs"))))
        # pooled: AOI units of the seed-averaged per-image delta
        frame = pd.concat([aoi_units(st["_dbar"], fold, refs[fold]["def_" + maskname], refs[fold])
                           .assign(arch=arch) for (arch, fold), st in zip(keys, stats)],
                          ignore_index=True)
        for arch in ARCHS:
            pl = M8.strat_pooled(frame[frame.arch == arch], f"{fam}|pool|{arch}")
            pooled_rows.append(dict(family=fam, pooling=f"within-architecture: {arch}",
                                    **{k: r6(v) if isinstance(v, float) else v
                                       for k, v in pl.items()}))
        pl = M8.strat_pooled(frame, f"{fam}|pool|all")
        pooled_rows.append(dict(family=fam, pooling="all four architectures",
                                **{k: r6(v) if isinstance(v, float) else v for k, v in pl.items()}))
        print(f"  family {fam:<24} pooled(all) {pl['mean']:+.6f} "
              f"[{pl['ci_lo']:+.6f}, {pl['ci_hi']:+.6f}]")

    # ---------------- cross-check: 08 task 1 reproduced through this lift ----
    d08 = json.load(open(RESULTS_08))
    x08 = {(c["arch"], c["fold"]): c for c in d08["task1"]["cells"] if c["metric"] == "2class"}
    xchk = []
    for arch in ARCHS:
        for fold in FOLDS:
            dbs = {}
            for s in SEEDS:
                L = M8.cell(sel[(arch, fold, s, "loco")])
                U = M8.cell(sel[(arch, fold, s, "upper")])
                dbs[s] = (U.miou2_pct - L.miou2_pct).to_numpy(np.float64)
                M8._CACHE.clear()
            st = paired(dbs, fold, f"t1|{arch}|{fold}|2class",
                        np.ones(N_TEST[fold], bool), refs[fold])
            ref08 = x08[(arch, fold)]
            dev = max(abs(st[k] - ref08[k]) for k in ("mean", "stderr", "ci_lo", "ci_hi", "p_raw"))
            xchk.append(dict(arch=arch, fold=fold, max_abs_dev=float(f"{dev:.3e}")))
    worst_x = max(r["max_abs_dev"] for r in xchk)
    if worst_x > 5e-7:
        fail(f"cross-check against results/08 task 1 deviates by {worst_x:.3e} (> 08's 6-dp "
             f"rounding); the bootstrap lift in this script is not 08's")
    print(f"  cross-check vs 08 task 1 (2-class mIoU gap, 12 pairs): max abs dev {worst_x:.3e}")
    crosscheck = dict(
        what="the two-class mIoU gap (upper - loco) of 08 task 1, recomputed through this "
             "script's masked wrapper with the mask set to every image and 08's own rng stream "
             "labels, compared against results/08_seo_full_grid.json task1.cells",
        fields=["mean", "stderr", "ci_lo", "ci_hi", "p_raw"], max_abs_dev=worst_x,
        tolerance="5e-7, which is 08's own 6-dp rounding",
        verifies="the lift, not the statistics: it shows this script's bootstrap, sign-flip and "
                 "seed stratum are 08 task 1's, applied to a different quantity",
        per_pair=xchk)

    def fam_rows(fam):
        return [r for r in pair_rows if r["family"] == fam]

    asym = fam_rows("sp_gap_asymmetry")
    counts = dict(
        n_pairs=len(asym),
        n_fg_penalty_larger_than_bg=int(sum(r["mean"] > 0 for r in asym)),
        n_holm_sig=int(sum(r["p_holm"] < 0.05 for r in asym)),
        n_positive_and_holm_sig=int(sum(r["mean"] > 0 and r["p_holm"] < 0.05 for r in asym)),
        n_ci_excludes_zero=int(sum(r["ci_lo"] > 0 or r["ci_hi"] < 0 for r in asym)),
        definition="count over the 12 (architecture x fold) pairs of "
                   "sp_gap_asymmetry = (fg AURC gap) - (bg AURC gap) > 0, where each gap is "
                   "LOCO minus the within-city upper bound on the same images",
        seed_resolved=dict(
            n_cells=36,
            n_positive=int(sum(1 for r in asym for v in r["per_seed"].values() if v > 0)),
            note="per-seed point estimates, no interval or test at the seed-resolved level"),
    )
    term = fam_rows("terminal_gap_asymmetry")
    counts["terminal_point_n_positive"] = int(sum(r["mean"] > 0 for r in term))
    counts["terminal_point_n_positive_and_holm_sig"] = int(
        sum(r["mean"] > 0 and r["p_holm"] < 0.05 for r in term))

    # ---------------- task 5: recoverable threshold, CE factor -------------
    def cell_mean_curve(arch, fold, cond, pop, seeds=SEEDS):
        mask = refs[fold]["def_" + pop]
        return np.mean([np.nanmean(Q[(arch, fold, s, cond)][pop]["rc"][mask], axis=0)
                        for s in seeds], axis=0)

    recover = []
    for arch in ARCHS:
        for fold in FOLDS:
            row = dict(arch=arch, fold=fold)
            for lab, seeds in [("seed_averaged", SEEDS)] + [(f"s{s}", [s]) for s in SEEDS]:
                lo = cell_mean_curve(arch, fold, "loco", "fg", seeds)
                up = cell_mean_curve(arch, fold, "upper", "fg", seeds)
                thr = up[-1] + RECOVER_TOL
                ok = np.where(lo <= thr)[0]
                cstar = r6(GRID[ok.max()]) if len(ok) else None
                row[lab] = dict(c_star=cstar, discard=None if cstar is None else r6(1.0 - cstar),
                                rc_loco_at_1=r6(lo[-1]), rc_upper_at_1=r6(up[-1]),
                                threshold=r6(thr),
                                verdict="recoverable" if cstar is not None else "NOT_RECOVERABLE")
            recover.append(row)
    rec_ok = [r for r in recover if r["seed_averaged"]["c_star"] is not None]
    disc = sorted(r["seed_averaged"]["discard"] for r in rec_ok)
    recoverable = dict(
        tolerance=RECOVER_TOL,
        definition="per cell, the largest c on the ratified 19-point grid whose cell-mean "
                   "foreground risk under LOCO is at most the same cell's upper-bound cell-mean "
                   "risk at c = 1.00 plus 0.005; mirrors the internal F6 rule, including its "
                   "absence of an mIoU sanity floor",
        n_recoverable_of_12=len(rec_ok),
        median_discard=r6(float(np.median(disc))) if disc else None,
        discard_by_fold={f: r6(float(np.median([r["seed_averaged"]["discard"] for r in rec_ok
                                                if r["fold"] == f])))
                         for f in FOLDS if any(r["fold"] == f for r in rec_ok)},
        n_recoverable_of_36_seed_resolved=int(sum(
            1 for r in recover for s in SEEDS if r[f"s{s}"]["c_star"] is not None)),
        cells=recover,
    )

    ce_pairs = fam_rows("sp_gap_ce_fg")
    zero_one = {(r["arch"], r["fold"]): r for r in fam_rows("sp_gap_fg")}
    ce_rows = []
    for r in ce_pairs:
        z = zero_one[(r["arch"], r["fold"])]
        ce_rows.append(dict(arch=r["arch"], fold=r["fold"], gap_01=z["mean"], gap_ce=r["mean"],
                            ratio=r6(r["mean"] / z["mean"]) if z["mean"] else None,
                            ce_exceeds_01=bool(r["mean"] > z["mean"])))
    pool01 = next(p for p in pooled_rows
                  if p["family"] == "sp_gap_fg" and p["pooling"] == "all four architectures")
    poolce = next(p for p in pooled_rows
                  if p["family"] == "sp_gap_ce_fg" and p["pooling"] == "all four architectures")
    ce_pos = [r for r in ce_rows if r["gap_01"] > 0 and r["gap_ce"] > 0]
    ce_factor = dict(
        definition="the CE-risk AURC gap divided by the 0/1-risk AURC gap on the foreground "
                   "population, both trapz19, both LOCO minus the within-city upper bound; the "
                   "pooled figure is the ratio of the two pooled means",
        pooled_gap_01=pool01["mean"], pooled_gap_ce=poolce["mean"],
        pooled_ratio=r6(poolce["mean"] / pool01["mean"]),
        n_pairs_ce_exceeds_01=int(sum(r["ce_exceeds_01"] for r in ce_rows)),
        n_pairs=len(ce_rows),
        n_pairs_gap_ce_negative=int(sum(r["gap_ce"] < 0 for r in ce_rows)),
        pairs_gap_ce_negative=[(r["arch"], r["fold"]) for r in ce_rows if r["gap_ce"] < 0],
        pooled_ratio_both_positive_pairs_only=r6(
            float(np.mean([r["gap_ce"] for r in ce_pos]) / np.mean([r["gap_01"] for r in ce_pos]))),
        n_pairs_both_positive=len(ce_pos),
        per_pair=ce_rows,
        caveat="the CE curve is not reconstructible from the dump; only the stored "
               "aurc_ce_fg scalar enters, so no CE terminal point is reported",
    )

    internal = dict(
        rule="internal and external values are two numbers, never the same quantity. No "
             "cross-dataset test is computed and nothing here is said to replicate anything.",
        internal_scope="ShadowTransfer, NAIP, 3 US cities, 63 LOCO cells (7 methods x 3 "
                       "architectures x 3 cities), single seed, rung 450, published §4.3 "
                       "re-verification under trapz19",
        external_scope="S-EO, WorldView-3, DSM-derived auto labels, 3 cities, 12 "
                       "(architecture x fold) vanilla pairs over 36 LOCO and 36 upper cells, "
                       "3 seeds, rung 3500",
        citations=[
            cite(COUNTERSIGNS, "G3 citable set", "the countersigned citable set"),
            cite(PAPER_STATE, "citable set: 55/63, 60/63, 59/63", "the citable set in PAPER_STATE"),
            cite(FER, "55/63 cells have gap_shadow > gap_bg", "internal F2 sign count"),
            cite(FER, "60/63 cells have gap_CE > gap_01", "internal F5 count"),
            cite(FER, "Pooled mean: gap_01 = +0.136, gap_CE = +0.315, ratio of means = 2.31",
                 "internal F5 pooled factor"),
            cite(FER, "Recovery rate: 59/63 cells", "internal F6 recoverable count"),
            cite(FER, "median c* = 0.75", "internal F6 median threshold -> 25% discard"),
            cite(REVIEWS, "55/63, 60/63, 59/63, 25%/40% all hold", "the E-7 addendum row"),
        ],
        struck_internal_values=["2.4x (cite 2.3x)", "the printed CE-exception sentence",
                                "the per-city 20%/22% discard medians"],
        columns=[
            dict(quantity="cells whose foreground SP gap exceeds their background SP gap",
                 internal="55 of 63 cells (sign p 4.9e-10)",
                 external=f"{counts['n_fg_penalty_larger_than_bg']} of 12 pairs "
                          f"({counts['n_positive_and_holm_sig']} Holm-significant); "
                          f"{counts['seed_resolved']['n_positive']} of 36 seed-resolved cells",
                 unit_note="internal counts cells of 7 methods at one seed; external counts "
                           "(architecture x fold) pairs of vanilla over 3 seeds"),
            dict(quantity="cells with a recoverable coverage threshold at tolerance 0.005",
                 internal="59 of 63 cells (all 4 exceptions Phoenix)",
                 external=f"{recoverable['n_recoverable_of_12']} of 12 pairs; "
                          f"{recoverable['n_recoverable_of_36_seed_resolved']} of 36 "
                          f"seed-resolved cells",
                 unit_note="same rule, same tolerance, no mIoU sanity floor on either side"),
            dict(quantity="median discarded coverage among recoverable cells",
                 internal="25% overall, 40% on Phoenix",
                 external=(f"{100 * recoverable['median_discard']:.1f}% overall, by fold "
                           + ", ".join(f"{f} {100 * v:.1f}%"
                                       for f, v in recoverable["discard_by_fold"].items())),
                 unit_note="discard = 1 - c* on the same 19-point grid"),
            dict(quantity="pooled CE-risk over 0/1-risk AURC gap factor, foreground",
                 internal="2.3x (gap_01 +0.136, gap_CE +0.315, 60 of 63 cells; the printed "
                          "2.4x is struck)",
                 external=f"{ce_factor['pooled_ratio']}x (gap_01 {ce_factor['pooled_gap_01']}, "
                          f"gap_CE {ce_factor['pooled_gap_ce']}, "
                          f"{ce_factor['n_pairs_ce_exceeds_01']} of 12 pairs); the CE gap is "
                          f"NEGATIVE on {ce_factor['n_pairs_gap_ce_negative']} pairs "
                          f"{ce_factor['pairs_gap_ce_negative']}, so the pooled ratio averages "
                          f"over sign-reversed cells; over the "
                          f"{ce_factor['n_pairs_both_positive']} pairs with both gaps positive it "
                          f"is {ce_factor['pooled_ratio_both_positive_pairs_only']}x",
                 unit_note="internal pools 63 cells; external pools AOI units over 12 pairs"),
        ],
    )

    # ---------------- task 6: method cells, sectioned out ------------------
    meth_idx = idx[(idx.category == "cell") & (idx.condition == "loco")
                   & (~idx.method.isin([LOCO_METHOD, UPPER_METHOD])) & (idx.method != "sar")]
    meth_rows, meth_skipped = [], []
    for _, r in meth_idx.sort_values("run_id_dir").iterrows():
        arch, fold, seed, meth = r["arch"], r["holdout"], int(r["seed"]), r["method"]
        if (arch, fold, seed, "upper") not in Q:
            meth_skipped.append(dict(run_id=r["run_id_dir"], reason="no matched upper cell"))
            continue
        rec = M8.find(rows, arch, meth, "loco", fold, seed)
        q = cell_quantities(rec, fold, refs[fold])
        u = Q[(arch, fold, seed, "upper")]
        mask = refs[fold]["def_both"]
        g_fg = float(np.mean(q["fg"]["aurc"][mask] - u["fg"]["aurc"][mask]))
        g_bg = float(np.mean(q["bg"]["aurc"][mask] - u["bg"]["aurc"][mask]))
        g_ce = float(np.mean(q["fg"]["ce"][mask] - u["fg"]["ce"][mask]))
        meth_rows.append(dict(arch=arch, method=meth, fold=fold, seed=seed, run_id=rec["run_id_dir"],
                              sp_gap_fg=r6(g_fg), sp_gap_bg=r6(g_bg),
                              sp_gap_asymmetry=r6(g_fg - g_bg), sp_gap_ce_fg=r6(g_ce),
                              fg_exceeds_bg=bool(g_fg - g_bg > 0),
                              ce_exceeds_01=bool(g_ce > g_fg),
                              target_access=r["target_access"]))
    method_section = dict(
        status="SECTIONED OUT of the headline, per the brief. Point estimates only: no interval, "
               "no test, no Holm, no pooling, and never pooled with the vanilla result.",
        n_cells=len(meth_rows),
        n_fg_exceeds_bg=int(sum(r["fg_exceeds_bg"] for r in meth_rows)),
        n_ce_exceeds_01=int(sum(r["ce_exceeds_01"] for r in meth_rows)),
        excluded=dict(sar="the 36 SAR dumps carry no aurc_* columns at all",
                      expa="category expa, not cell: no validation split, target_labels",
                      retrain_suffix="the two -e200 OGLANet retrains of vanilla/upper cells; the "
                                     "inventory-named originals are the primary universe, as in 08",
                      other=meth_skipped),
        cells=meth_rows,
    )

    # ---------------- sanity flags ----------------------------------------
    flags = []
    floor = round(1.0 / (B + 1.0), 6)      # rows carry 6-dp p-values; compare like with like
    at_floor = sum(1 for r in pair_rows if r["p_raw"] <= floor + 1e-12)
    flags.append(f"{at_floor} of {len(pair_rows)} sign-flip p-values sit at the permutation "
                 f"floor 1/(B+1) = {floor:.4f}.")
    if counts["n_fg_penalty_larger_than_bg"] in (0, 12):
        flags.append(f"sp_gap_asymmetry has the same sign in all 12 pairs "
                     f"({counts['n_fg_penalty_larger_than_bg']} of 12 positive); perfect "
                     f"separation, flagged per CONVENTIONS.")
    bg = fam_rows("sp_gap_bg")
    tiny = [r for r in bg if abs(r["mean"]) < 0.01 and (r["ci_lo"] > 0 or r["ci_hi"] < 0)]
    if tiny:
        flags.append(f"{len(tiny)} of 12 background SP gaps are under 0.01 in absolute value with "
                     f"a CI excluding zero; implausibly low variance on a tiny effect, the same "
                     f"pattern 08 flagged on its SAR rows.")
    fgm = np.array([r["mean"] for r in fam_rows("sp_gap_fg")])
    tgm = np.array([r["mean"] for r in fam_rows("terminal_gap_fg")])
    rho = float(np.corrcoef(fgm, tgm)[0, 1])
    flags.append(f"the foreground AURC gap and the foreground terminal-point gap correlate at "
                 f"r = {rho:.4f} over the 12 pairs. This is structural, not a finding: "
                 f"results/06 records that the foreground 0/1 risk-coverage curve carries no "
                 f"ranking information, so both readouts are functions of per-image recall alone.")
    ceil_cells = [r for r in cells if r["n_at_aurc_ceiling_fg"] > 0]
    if ceil_cells:
        flags.append(f"{len(ceil_cells)} of 72 cells contain images whose foreground AURC sits "
                     f"exactly at the 0.9 ceiling (rc == 1 at every coverage, no correctly "
                     f"classified shadow pixel); max {max(r['n_at_aurc_ceiling_fg'] for r in cells)} "
                     f"images in one cell. The foreground AURC distribution is censored above.")
    ns = [r for r in asym if r["p_holm"] >= 0.05]
    if ns:
        byfold = sorted({r["fold"] for r in ns})
        flags.append(f"the {len(ns)} of 12 pairs whose sp_gap_asymmetry is not Holm-significant "
                     f"sit on folds {byfold}" + (f" -- all {len(ns)} on {byfold[0]}, which is "
                     f"every {byfold[0]} pair in the grid" if len(byfold) == 1 else "") + ".")
    disagree = [(r["arch"], r["fold"]) for r, t in zip(asym, term)
                if (r["mean"] > 0) != (t["mean"] > 0)]
    if disagree:
        flags.append(f"the AURC and terminal-point readouts of the asymmetry disagree in sign on "
                     f"{len(disagree)} of 12 pairs: {disagree}. Both are reported; neither is "
                     f"selected.")
    if ce_factor["n_pairs_gap_ce_negative"]:
        flags.append(f"the cross-entropy-risk foreground gap is negative on "
                     f"{ce_factor['n_pairs_gap_ce_negative']} of 12 pairs "
                     f"({ce_factor['pairs_gap_ce_negative']}) while the 0/1 gap is positive on all "
                     f"12, so the two risk definitions disagree in sign on those pairs and the "
                     f"pooled CE factor averages over them.")
    cip = [r for r in pair_rows if r["ci_excludes_point"]]
    flags.append(f"{len(cip)} of {len(pair_rows)} bootstrap CIs exclude their own point estimate."
                 if cip else "no bootstrap CI excludes its own point estimate.")
    coincide = []
    if abs(ce_factor["pooled_ratio"] - 2.3) < 0.05:
        coincide.append(f"the external pooled CE factor {ce_factor['pooled_ratio']} lands within "
                        f"0.05 of the internal 2.3x")
    if recoverable["median_discard"] is not None and abs(recoverable["median_discard"] - 0.25) < 1e-9:
        coincide.append("the external median discard lands exactly on the internal 25%")
    flags.append("no external number lands on a published internal value within rounding."
                 if not coincide else "COINCIDENCE WITH A PUBLISHED VALUE: " + "; ".join(coincide))
    bands = [r for r in cells if r["e9_band"] is not None]
    flags.append(f"{len(cells) - len(bands)} of 72 cells carry no inference-noise band; the "
                 f"{len(bands)} that do are 3-pass and measure two-class mIoU only, so no AURC or "
                 f"terminal-point number in this file has a run-to-run band. OGLANet's own probe "
                 f"records a per-pixel probability movement of up to 0.63 between passes with "
                 f"about 5.8M pixels changing prediction, which is exactly what a "
                 f"probability-ranked metric reads; every OGLANet number here is one draw.")

    # ---------------- write ------------------------------------------------
    runs = [sel[k]["run_id_dir"] for k in sorted(sel)]
    pooled_head = {p["family"]: p for p in pooled_rows
                   if p["pooling"] == "all four architectures"}
    headline = (
        f"On the 72 vanilla S-EO cells, the foreground-minus-background trapz19 AURC gap "
        f"(LOCO minus the within-city upper bound) is positive in "
        f"{counts['n_fg_penalty_larger_than_bg']} of 12 (architecture x fold) pairs and "
        f"Holm-significant in {counts['n_positive_and_holm_sig']}, pooling to "
        f"{pooled_head['sp_gap_asymmetry']['mean']:+.6f} "
        f"[{pooled_head['sp_gap_asymmetry']['ci_lo']:+.6f}, "
        f"{pooled_head['sp_gap_asymmetry']['ci_hi']:+.6f}] over "
        f"{pooled_head['sp_gap_asymmetry']['n_aoi_units']} AOI units, with a full-coverage "
        f"terminal-point asymmetry of {pooled_head['terminal_gap_asymmetry']['mean']:+.6f} "
        f"[{pooled_head['terminal_gap_asymmetry']['ci_lo']:+.6f}, "
        f"{pooled_head['terminal_gap_asymmetry']['ci_hi']:+.6f}].")

    out = dict(
        id="10_seo_selective_prediction",
        arm="A",
        answers=["AC1", "v1UJ-W1", "BxRx-W1", "BxRx-W2"],
        answers_note="AC1 / v1UJ-W1 / BxRx-W1 are arm A's rows: this is the §4.3 half of the "
                     "external evidence, the last part of the paper's diagnosis that had none. "
                     "BxRx-W2 is a TEXT row with no arm whose Lands column names §4.3; this "
                     "artifact supplies external numbers for that section, not a verdict on the "
                     "row. E-7 and its addendum are the definition provenance, answered by "
                     "results/06, not here.",
        question="Does the section 4.3 selective-prediction asymmetry -- a foreground AURC "
                 "penalty under LOCO that the background population does not carry -- appear on "
                 "the external dataset, and at what magnitude?",
        builds_on=dict(
            definition="results/06_aurc_recovery.json",
            grid="results/08_seo_full_grid.json (cell universe, bootstrap scheme, task 1)",
            diagnostics="results/09_seo_diagnostics.json (not read for any number here)"),
        runs=runs,
        n_runs=len(runs),
        index_path=INDEX_PATH,
        index_sha256=EXPECTED_INDEX_SHA,
        results_06_sha256=sha256_file(RESULTS_06),
        results_08_sha256=sha256_file(RESULTS_08),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/10_seo_selective_prediction.py",
        script_sha256=sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_seo_selective_prediction.py")),
        metrics_module=dict(path=M8.METRICS_PATH, sha256=EXPECTED_METRICS_SHA,
                            used_for="COVERAGE_GRID, aurc_trapz19; the risk-coverage curve is "
                                     "reconstructed and checked bit-exact against the stored "
                                     "trapz19 column rather than recomputed from probability maps"),
        interpreter="python",
        git_hash=None,
        git_hash_note="the project base is not a git repository on this box (CLAUDE.md)",
        seed=SEED, B=B,
        definition=prov,
        definition_gate=prov_checks,
        no_branch_applied="No arm A row definitions exist in either tree (REVIEWS.md provenance "
                          "note, 2026-10-24), so no pre-committed branch is applied and no "
                          "verdict is offered. Numbers only.",
        probs_npz_needed=dict(
            answer="no",
            columns_used=["aurc_fg", "aurc_bg", "aurc_all", "aurc_ce_fg", "aurc_ce_bg",
                          "n_gt_fg_valid", "n_valid_px", "image_id", "aoi_id",
                          "tp_tol", "fp_tol", "fn_tol", "tn_tol"],
            why="the foreground and background 0/1 risk-coverage curves are deterministic "
                "functions of the population size and the number of correctly classified pixels, "
                "because the confidence ranking puts every correct pixel ahead of every error "
                "(results/06's flagged property). Inverting the stored trapz19 column recovers "
                "that integer count, and the reconstructed curve reproduces the stored column "
                "bit-exactly on every image of every cell, so the full curve and its terminal "
                "point are exact.",
            not_computed=["the all-pixels curve and terminal point: its confidence is "
                          "max(p, 1-p) and the ranking does carry information, so the curve is "
                          "not reconstructible from counts. Only the stored aurc_all scalar is "
                          "reported.",
                          "the cross-entropy curve and terminal point: the risk depends on the "
                          "probability values. Only the stored aurc_ce_* scalars are reported."],
        ),
        conditions=[
            dict(label=f"{r['arch']} / {r['fold']} held out / foreground minus background",
                 metric="sp_gap_asymmetry_aurc_trapz19", mean=r["mean"], stderr=r["stderr"],
                 n=r["n_images"], ci95=[r["ci_lo"], r["ci_hi"]],
                 ci_method="aoi_cluster_bootstrap_B10000_seed_stratum",
                 baseline=f"{r['arch']}-none-upper-r3500-{r['fold']}-hr (3 seeds)")
            for r in asym],
        tests=[dict(name="AOI-level two-sided sign-flip permutation on the seed-averaged "
                         "per-image delta",
                    compares=[f"{r['arch']}-vanilla-loco-r3500-{r['fold']}-hr",
                              f"{r['arch']}-none-upper-r3500-{r['fold']}-hr"],
                    family=r["family"], statistic=r["mean"], p=r["p_raw"], p_holm=r["p_holm"],
                    B=B, two_tailed=True, holm_over=12)
               for r in pair_rows],
        inference_noise_bands=dict(
            n_cells=len(cells),
            n_with_band=sum(1 for r in cells if r["e9_band"] is not None),
            n_without_band=sum(1 for r in cells if r["e9_band"] is None),
            pass_counts=sorted({r["e9_band"]["n_passes"] for r in cells
                                if r["e9_band"] is not None}),
            with_band=[dict(run_id=r["run_id"], file=r["e9_band"]["file"],
                            n_passes=r["e9_band"]["n_passes"],
                            miou_pct_sd=r["e9_band"]["miou_pct_sd"])
                       for r in cells if r["e9_band"] is not None],
            without_band=[r["run_id"] for r in cells if r["e9_band"] is None],
            scope="every band on this grid measures two-class mIoU spread across repeated "
                  "inference passes. No band of any kind exists for an AURC, a terminal point or "
                  "a risk-coverage curve, so no number in this file carries a run-to-run "
                  "interval. The 3-pass bands are undersampled as gates (D-B4-3).",
            note_zero_sd="crossearth-vanilla-loco-r3500-oma-hr-s0's band is exactly 0.0 because "
                         "its probe verdict is BIT-EXACT across the 3 passes, not because the "
                         "band failed to measure anything; its sidecar's published_e9_sd_upper "
                         "field is OGLANet's constant carried for key compatibility and is not a "
                         "CrossEarth band. The other 18 bands run 0.0014 to 0.0275 mIoU.",
            why_it_matters="E-9: OGLANet inference is nondeterministic. Its own probe on this "
                           "grid records up to 0.63 of per-pixel probability movement between "
                           "passes and about 5.8M pixels changing prediction, which is precisely "
                           "what a confidence-ranked metric reads; every OGLANet AURC, terminal "
                           "point and curve here is one draw.",
        ),
        crosscheck_vs_08_task1=crosscheck,
        task1_cells=cells,
        task2_pairs=[r for r in pair_rows if r["quantity"] in ("aurc", "ce")],
        task3_terminal_point=[r for r in pair_rows if r["quantity"] == "terminal"],
        task4_pooled=pooled_rows,
        counts=counts,
        task5_internal_vs_external=internal,
        task5_recoverable_threshold=recoverable,
        task5_ce_factor=ce_factor,
        task6_method_cells_sectioned_out=method_section,
        sanity_flags=flags,
        headline=headline,
        caveats=[
            "The AURC definition is results/06's ratified trapz19: numpy trapezoid of the "
            "19-point risk-coverage curve over linspace(0.10, 1.00, 19), unnormalised over a span "
            "of 0.9. The terminal point rc(1.00) is reported under its own name and is never "
            "called an AURC, because that is the quantity Fig 3(a) printed under a Delta-AURC "
            "label (results/06, E-7 addendum).",
            "SP gap direction is LOCO minus the within-city upper bound, matching CONVENTIONS.md. "
            "Positive means LOCO carries the higher selective risk. This is the opposite sign "
            "convention to results/08's mIoU gaps, which are upper minus LOCO.",
            "Vanilla cells only. The 36 LOCO cells are source_only and the 36 upper cells are "
            "target_labels. Adaptation-method cells are in task 6 and are never pooled with any "
            "headline number.",
            f"Images whose population is below {MIN_POP} pixels are dropped, because "
            "metrics.per_image_rc returns None for them. The drop is a ground-truth property, "
            "identical in every cell of a fold and in both conditions, so the per-image pairing "
            "is exact; per-row n_images states what was used. This includes the 108 fully "
            "DSM-unseen patches (n_valid_px == 0), which results/08's mIoU numbers DO include as "
            "exact-zero deltas and which cannot enter any risk-coverage quantity.",
            "The three-population columns are not on the same denominators: the foreground "
            "population is the GT shadow pixels of the valid region, the background population "
            "its complement, and all_pixels the whole valid region. The fg-minus-bg asymmetry is "
            "computed on the images where both populations are defined, so fg, bg and asymmetry "
            "rows of task 2 add up exactly; sp_gap_fg_own_mask and sp_gap_bg_own_mask give the "
            "same gaps on each population's own larger image set.",
            "Three seeds per cell. Every interval is an AOI cluster bootstrap with seed as a "
            "stratum (AOIs resampled, all three seeds kept inside each sampled AOI), which is "
            "08 task 1's own function; it covers test-patch and AOI sampling and seed variation "
            "only through the seed average. Per-seed point estimates and the across-seed SD "
            "(n = 3) sit beside every estimate.",
            "Holm families are 12 (architecture x fold) pairs, applied separately within each of "
            "the nine quantity families; the family name is on every row. No correction is "
            "applied across families, and the terminal-point family is not independent of the "
            "AURC family on the foreground population (see the sanity flag).",
            "53 of the 72 cells carry no inference-noise band. The 19 that do (all 18 OGLANet "
            "cells and crossearth-vanilla-loco-oma-s0) are 3-pass and record two-class mIoU "
            "spread only, so no AURC or terminal-point number in this file carries a run-to-run "
            "band. E-9's own probe records up to 0.63 of per-pixel probability movement between "
            "OGLANet passes, which a probability-ranked metric reads directly.",
            "S-EO labels are auto-generated from a DSM. Both sides of every paired delta use the "
            "same labels on the same patches, so the gap is a difference under common label "
            "noise; absolute AURC levels carry that noise.",
            "Budget asymmetry on both sides of every gap, as in results/02 and 08: LOCO trains on "
            "3,500 patches of the two source cities, the upper bound on 3,500 patches of the "
            "held-out city. rung 3500 is the S-EO patch budget and is not a point on the "
            "225/450/900 ladder.",
            "The internal and external columns of task 5 are two numbers, never the same "
            "quantity. No cross-dataset test is computed, no magnitude is merged, and nothing "
            "here is said to replicate, validate or confirm the internal result. The internal "
            "column is the countersigned citable set; the struck values (2.4x, the printed "
            "CE-exception sentence and the per-city 20%/22% medians) are not cited.",
            "The internal and external units differ inside task 5: internal counts 63 cells of 7 "
            "methods at one seed on 3 NAIP cities, external counts 12 (architecture x fold) "
            "vanilla pairs over 3 seeds on 3 WorldView-3 cities. The seed-resolved external "
            "counts are given beside the pair counts so the unit is visible.",
            "The recoverable-threshold rule mirrors internal F6, including F6's recorded absence "
            "of an mIoU sanity floor, so neither column is filtered on cell quality.",
            "The cross-entropy column uses the stored aurc_ce_* scalars; its curve and terminal "
            "point are not reconstructible from the dump, so no CE terminal point is reported and "
            "the CE factor is a ratio of integrals only.",
            "Cells that stopped on the epoch ceiling rather than on patience, carried over from "
            "08's caveat: crossearth-vanilla-loco-jax-s1 and -s2, dinov3-vanilla-loco-oma-s1, "
            "oglanet-none-upper-ucsd-s0, oglanet-vanilla-loco-oma-s0. The two OGLANet -e200 "
            "retrains of vanilla and upper cells exist and are NOT used; the inventory-named "
            "originals are the primary universe, as in 08.",
            "No pre-committed branch is applied. No arm A row definitions exist in either tree "
            "(REVIEWS.md provenance note, 2026-10-24), so this file reports numbers and no "
            "verdict; the reading is the operator's and HLT's.",
            "INDEX.parquet was read and its sha gated against the one 08 and 09 cite; it was not "
            "rebuilt. results/08, results/09 and final_paper/ were not edited.",
        ],
        figure=None,
        figure_note="No plot. CS renders from this file; the per-cell 19-point curves are in "
                    "task1_cells under rc_curve_fg / rc_curve_bg, and plots/10_* is reserved for "
                    "that figure.",
        caption=None,
    )

    with open(RESULTS_PATH, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=False, allow_nan=False)
    print(f"\nwrote {RESULTS_PATH}  ({os.path.getsize(RESULTS_PATH)/1e3:.1f} kB)")
    print(f"sha256 {sha256_file(RESULTS_PATH)}")
    print(f"\nHEADLINE: {headline}")
    for f in flags:
        print(f"  FLAG: {f}")


if __name__ == "__main__":
    main()
