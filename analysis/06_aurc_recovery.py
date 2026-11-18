"""
06_aurc_recovery.py — recover the rule behind Fig 3(a)'s published +0.31.

Fig 3 (`figures/fig_diagnostics.tex`, label `fig:diagnostics`) panel (a) prints
"LOCO accumulates $\\Delta$AURC$=+0.31$ over the upper bound" for MAMNet with
Phoenix held out, on the foreground pixel population. Two candidate AURC
definitions live in the tree and neither reproduces it: `aurc_trapz19` (+0.2615)
and `aurc_mean20` (+0.2869), measured by P0a (EXP-01).

This script enumerates the definition space systematically, computes every
variant on the caption's own cell scope, and records the number each one
produces whether it matches or not.

Everything is recomputed from first principles: the contract dumps' stored
probability maps plus the benchmark's ground-truth masks, driven through the
lifted metric module. The published per-image risk-coverage arrays
(`cell_rcs.pkl`) and the published per-cell gaps
(`inria_sp_battery_results.json`) are read as independent cross-checks, never
as the source of a swept number.

Run:
      PYTHONNOUSERSITE=1 \
      python \
      analysis/06_aurc_recovery.py

DRY_RUN prints the resolved schema, the cells selected, their row counts and the
gate results, and writes nothing.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# The system dist-packages on this box holds numpy-1.x-compiled numexpr and
# bottleneck that abort under the venv's numpy 2.x. Drop it; every package this
# script needs resolves from the interpreter's own environment.
sys.path = [p for p in sys.path if p != "/usr/lib/python3/dist-packages"]

import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------- config
DRY_RUN = "--dry-run" in sys.argv

BASE = os.path.join(os.environ["PROJECT_ROOT"], "")
# NOT dumps/st/. The arm C ladder reused the P0a run_ids, so
# dumps/st/mamnet-{none-upper,vanilla-loco}-r450-phoenix-hr-s0 now hold arm C
# retrains whose probability maps are not the published ones. The published
# checkpoints' dumps survive under st_r450_published_recompute/, which is what
# the Fig 3 curves were built from. Gate [5] below proves it.
DUMPS = BASE + "dumps/st_r450_published_recompute/"
INDEX_PARQUET = DUMPS + "INDEX.parquet"
OUT_JSON = BASE + "results/06_aurc_recovery.json"

METRIC_MODULE_DIR = os.path.join(os.environ.get("ST_METRICS_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))), "shadowtransfer")
PROV = os.path.join(os.environ["PROJECT_ROOT"], "")
FIG3_DATA = PROV + "data/figure3_data/"
FIG3_CODE = PROV + "Python/FINAL-fig3.ipynb"
GT_ROOT = PROV + "data/Final_data_test/"

# The caption's scope: MAMNet, Phoenix held out, foreground pixels.
CELL_UPPER = "mamnet-none-upper-r450-phoenix-hr-s0"
CELL_LOCO = "mamnet-vanilla-loco-r450-phoenix-hr-s0"
CELL_FDA_PUB = "mamnet_fda_phoenix"          # published-curve id, cross-check only
PUB_UPPER = "mamnet_upper_phoenix"
PUB_LOCO = "mamnet_vanilla_phoenix"
GT_CITY, GT_RES, GT_SPLIT = "phoenix", "highres", "test"

TARGET = 0.31                 # the printed value we are trying to reproduce
MATCH_TOL = 0.005             # rounds to +0.31 at two decimals
NEAR_TOL = 0.010              # reported as a near-miss, not a match

B_BOOT = 10000                # CLAUDE.md standard
B_BOOT_PUB = 1000             # the published figure's own N_BOOT
SEED = 42                     # the published figure's own RNG_SEED

sys.path.insert(0, METRIC_MODULE_DIR)

# metrics.py imports `utils.contrast_utils` for DetailedEvaluator's boundary
# analysis. That helper lives in the SLURM tree's python/mamnet/ and is not on
# this box. Nothing this script calls touches it, so stub it with a function
# that raises: if the sweep ever reaches boundary analysis the run dies loudly
# rather than returning a wrong contrast.
import types                   # noqa: E402
if "utils" not in sys.modules:
    _u = types.ModuleType("utils")
    _cu = types.ModuleType("utils.contrast_utils")

    def _no_contrast(*a, **k):
        raise RuntimeError(
            "compute_shadow_contrast is stubbed on the analysis host; this analysis must not "
            "reach DetailedEvaluator's boundary analysis")
    _cu.compute_shadow_contrast = _no_contrast
    _u.contrast_utils = _cu
    sys.modules["utils"] = _u
    sys.modules["utils.contrast_utils"] = _cu

import metrics as M           # noqa: E402  (lifted metric module; never reimplemented)

PUB_GRID = M.COVERAGE_GRID    # np.linspace(0.10, 1.00, 19)


# ------------------------------------------------------------- utilities
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cell(run_id):
    """Per-image frame, probability maps and run.json for one dump cell."""
    d = os.path.join(DUMPS, run_id)
    if not os.path.isdir(d):
        raise SystemExit(f"missing dump cell: {d}")
    side = os.path.join(d, "run.json")
    if not os.path.isfile(side):
        raise SystemExit(f"run.json absent next to perimage.parquet: {d}")
    with open(side) as f:
        run = json.load(f)
    df = pd.read_parquet(os.path.join(d, "perimage.parquet"))
    # npz members re-decompress on every __getitem__; the sweep touches each
    # image hundreds of times, so materialise once.
    with np.load(os.path.join(d, "probs.npz")) as z:
        probs = {k: z[k].astype(np.float32) for k in z.files}
    return df, probs, run


def load_gt(image_ids):
    """Binary ground-truth shadow masks for the held-out city's test split."""
    mdir = os.path.join(GT_ROOT, GT_CITY, GT_RES, GT_SPLIT, "masks")
    gt = {}
    for iid in image_ids:
        p = os.path.join(mdir, iid + ".png")
        if not os.path.isfile(p):
            raise SystemExit(f"missing GT mask: {p}")
        gt[iid] = np.array(Image.open(p)) > 127
    return gt


# --------------------------------------------------- variant RC generator
def rc_variant(gt, prob, mode, grid, conf_kind="published", k_rule="ceil"):
    """
    The published `metrics.per_image_rc` body with the coverage grid, the
    confidence score and the coverage-to-rank rule hoisted into arguments.

    conf_kind: 'published' (fg: p, bg: 1-p, all: max(p,1-p))
               'margin'    (|2p-1| on every population)
               'maxprob'   (max(p,1-p) on every population)
    k_rule:    'ceil' (published) or 'int' (the aurc_mean20 truncation)

    Gated in main() against `metrics.per_image_rc` on the published grid.
    """
    gt_flat = gt.ravel().astype(bool)
    p_flat = prob.ravel().astype(np.float32)
    if mode == "all_pixels":
        pred = p_flat > 0.5
        err = (pred != gt_flat).astype(np.float32)
        conf_pub = np.maximum(p_flat, 1.0 - p_flat)
        p_sel = p_flat
    elif mode == "gt_shadow":
        if gt_flat.sum() < 50:
            return None
        p_sel = p_flat[gt_flat]
        err = (p_sel <= 0.5).astype(np.float32)
        conf_pub = p_sel
    elif mode == "gt_bg":
        if (~gt_flat).sum() < 50:
            return None
        p_sel = p_flat[~gt_flat]
        err = (p_sel > 0.5).astype(np.float32)
        conf_pub = 1.0 - p_sel
    else:
        raise ValueError(mode)

    if conf_kind == "published":
        conf = conf_pub
    elif conf_kind == "margin":
        conf = np.abs(2.0 * p_sel - 1.0)
    elif conf_kind == "maxprob":
        conf = np.maximum(p_sel, 1.0 - p_sel)
    else:
        raise ValueError(conf_kind)

    n = len(conf)
    if n < 50:
        return None
    order = np.argsort(-conf)
    cum_err = np.cumsum(err[order])
    rc = np.empty(len(grid), dtype=np.float64)
    for i, c in enumerate(grid):
        k = max(int(np.ceil(c * n)), 1) if k_rule == "ceil" else max(1, int(c * n))
        rc[i] = cum_err[k - 1] / k
    return rc


def rc_ce_variant(gt, prob, mode, grid):
    """`metrics.per_image_rc_ce` with the grid hoisted."""
    gt_flat = gt.ravel().astype(bool)
    p_flat = np.clip(prob.ravel().astype(np.float32), M.EPS_CE, 1 - M.EPS_CE)
    if mode == "gt_shadow":
        if gt_flat.sum() < 50:
            return None
        p = p_flat[gt_flat]
        loss = -np.log(p)
        conf = p
    elif mode == "gt_bg":
        if (~gt_flat).sum() < 50:
            return None
        p = p_flat[~gt_flat]
        loss = -np.log(1 - p)
        conf = 1 - p
    elif mode == "all_pixels":
        t = gt_flat.astype(np.float32)
        loss = -(t * np.log(p_flat) + (1 - t) * np.log(1 - p_flat))
        conf = np.maximum(p_flat, 1 - p_flat)
    else:
        raise ValueError(mode)
    n = len(conf)
    if n < 50:
        return None
    order = np.argsort(-conf)
    cum = np.cumsum(loss[order])
    rc = np.empty(len(grid), dtype=np.float64)
    for i, c in enumerate(grid):
        k = max(int(np.ceil(c * n)), 1)
        rc[i] = cum[k - 1] / k
    return rc


# ------------------------------------------------------- integration rules
def integrate(rc, grid, rule):
    rc = np.asarray(rc, dtype=float)
    g = np.asarray(grid, dtype=float)
    span = g[-1] - g[0]
    if rule == "trapz":
        return float(np.trapezoid(rc, g))
    if rule == "trapz_norm":
        return float(np.trapezoid(rc, g) / span) if span > 0 else float("nan")
    if rule == "mean":
        return float(rc.mean())
    if rule == "left":
        return float(np.sum(rc[:-1] * np.diff(g)))
    if rule == "right":
        return float(np.sum(rc[1:] * np.diff(g)))
    if rule == "simpson":
        n = len(rc)
        if n % 2 == 0:
            return float("nan")          # Simpson needs an odd point count
        h = g[1] - g[0]
        w = np.ones(n)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return float(h / 3.0 * np.dot(w, rc))
    raise ValueError(rule)


def paired_bootstrap(delta, B, seed, stat="mean"):
    """Image-level paired bootstrap over per-image deltas."""
    v = np.asarray(delta, dtype=float)
    v = v[np.isfinite(v)]
    rng = np.random.RandomState(seed)
    n = len(v)
    f = np.mean if stat == "mean" else np.median
    boots = np.empty(B)
    for b in range(B):
        boots[b] = f(v[rng.randint(0, n, size=n)])
    point = float(f(v))
    return point, float(boots.mean()), float(np.percentile(boots, 2.5)), \
        float(np.percentile(boots, 97.5))


# ------------------------------------------------------------------- main
def main():
    print("=" * 78)
    print("06 AURC RULE RECOVERY — Fig 3(a) published +0.31")
    print("=" * 78)

    # ---- provenance -----------------------------------------------------
    print("\n[0] Provenance")
    print(f"  metric module : {METRIC_MODULE_DIR}/metrics.py")
    print(f"                  sha256 {sha256_file(METRIC_MODULE_DIR + '/metrics.py')[:16]}")
    print(f"  figure code   : {FIG3_CODE}")
    print(f"                  sha256 {sha256_file(FIG3_CODE)[:16]}")
    print(f"  published grid: {np.round(PUB_GRID, 3)}  (n={len(PUB_GRID)})")

    # ---- dumps ----------------------------------------------------------
    print("\n[1] Dump cells")
    df_u, pr_u, run_u = load_cell(CELL_UPPER)
    df_l, pr_l, run_l = load_cell(CELL_LOCO)
    for tag, df, run, pr in (("upper", df_u, run_u, pr_u), ("loco", df_l, run_l, pr_l)):
        print(f"  {tag:5s} {run['run_id']}")
        print(f"        rows={len(df)}  n_test(run.json)={run.get('n_test')}  "
              f"probs keys={len(pr)}  arch={run.get('arch')} "
              f"method={run.get('method')} holdout={run.get('holdout')}")
        print(f"        legacy_run_dir={run.get('legacy_run_dir')}")
        print(f"        ckpt={str(run.get('ckpt_path'))[-72:]}")
        if run.get("n_test") is not None and len(df) != run["n_test"]:
            raise SystemExit(f"{tag}: perimage rows {len(df)} != n_test {run['n_test']}")
        if len(df) != 150:
            raise SystemExit(f"{tag}: expected the 150-image Phoenix test pool, got {len(df)}")
        if run.get("legacy_run_dir") is None:
            raise SystemExit(
                f"{tag}: legacy_run_dir is None — this is a retrain, not the published "
                f"checkpoint. Fig 3's curves came from the published checkpoints.")
        if set(df["image_id"]) != set(pr):
            raise SystemExit(f"{tag}: probs.npz keys != perimage image_id set")

    required = ["image_id", "tp_str", "fn_str", "n_gt_fg_valid", "empty_fg_valid",
                "aurc_fg", "aurc_bg", "aurc_all", "aurc_ce_fg",
                "aurc_fg_mean20", "iou_tol"]
    for c in required:
        for tag, df in (("upper", df_u), ("loco", df_l)):
            if c not in df.columns:
                raise SystemExit(f"{tag}: contract column absent: {c}")
    print(f"  contract columns present: {len(required)}/{len(required)}")
    print(f"  dtypes: " + ", ".join(f"{c}={df_u[c].dtype}" for c in
                                    ["tp_str", "n_gt_fg_valid", "empty_fg_valid", "aurc_fg"]))

    ids = sorted(set(df_u["image_id"]) & set(df_l["image_id"]))
    if len(ids) != 150:
        raise SystemExit(f"paired image set is {len(ids)}, expected 150")
    print(f"  paired image ids: {len(ids)}")

    # ---- ground truth ---------------------------------------------------
    print("\n[2] Ground truth")
    gt = load_gt(ids)
    fg_counts = np.array([gt[i].sum() for i in ids])
    print(f"  masks loaded: {len(gt)} from {GT_ROOT}{GT_CITY}/{GT_RES}/{GT_SPLIT}/masks")
    print(f"  GT shadow px per image: min={fg_counts.min()} median={int(np.median(fg_counts))} "
          f"max={fg_counts.max()}")
    print(f"  images with <50 GT shadow px (dropped by per_image_rc): "
          f"{int((fg_counts < 50).sum())}")
    print(f"  images with empty_fg_valid: "
          f"{int(df_u.set_index('image_id').loc[ids, 'empty_fg_valid'].sum())}")

    # ---- gate: variant generator == published implementation -------------
    print("\n[3] Gate — rc_variant() reproduces metrics.per_image_rc on the published grid")
    worst = 0.0
    for tag, pr in (("upper", pr_u), ("loco", pr_l)):
        for iid in ids:
            p = pr[iid].astype(np.float32)
            for mode in ("gt_shadow", "gt_bg", "all_pixels"):
                a = M.per_image_rc(gt[iid], p, mode)
                b = rc_variant(gt[iid], p, mode, PUB_GRID)
                if (a is None) != (b is None):
                    raise SystemExit(f"gate: None mismatch {tag}/{iid}/{mode}")
                if a is not None:
                    worst = max(worst, float(np.abs(np.asarray(a, float) - b).max()))
    print(f"  max |rc_variant - per_image_rc| = {worst:.3e}")
    if worst > 1e-6:
        raise SystemExit("GATE FAILED: variant generator diverges from the metric module")

    # ---- recompute the published curves ---------------------------------
    print("\n[4] Recompute per-image risk-coverage from dumps + GT")
    RC = {}          # RC[(cond, mode)][iid] -> rc on PUB_GRID
    for cond, pr in (("upper", pr_u), ("loco", pr_l)):
        for mode in ("gt_shadow", "gt_bg", "all_pixels"):
            d = {}
            for iid in ids:
                r = M.per_image_rc(gt[iid], pr[iid].astype(np.float32), mode)
                if r is not None:
                    d[iid] = np.asarray(r, dtype=float)
            RC[(cond, mode)] = d
            print(f"  {cond:5s} {mode:11s} n={len(d)}")

    fg_ids = sorted(set(RC[("upper", "gt_shadow")]) & set(RC[("loco", "gt_shadow")]))
    print(f"  paired foreground image set: n={len(fg_ids)}")

    # ---- cross-check against the published artefacts ---------------------
    print("\n[5] Cross-check against published figure data")
    import pickle
    with open(FIG3_DATA + "cell_rcs.pkl", "rb") as f:
        pub = pickle.load(f)["cell_rcs"]
    checks = {}
    for cond, pid in (("upper", PUB_UPPER), ("loco", PUB_LOCO)):
        for mode in ("gt_shadow", "gt_bg", "all_pixels"):
            ours, theirs = RC[(cond, mode)], pub[pid][mode]
            common = sorted(set(ours) & set(theirs))
            dev = max(float(np.abs(ours[i] - np.asarray(theirs[i], float)).max())
                      for i in common)
            checks[f"{cond}|{mode}"] = {"n_ours": len(ours), "n_published": len(theirs),
                                        "n_common": len(common), "max_abs_dev": dev}
            print(f"  {cond:5s} {mode:11s} ours={len(ours):3d} pub={len(theirs):3d} "
                  f"common={len(common):3d} max|dev|={dev:.3e}")

    fg_dev = max(checks["upper|gt_shadow"]["max_abs_dev"],
                 checks["loco|gt_shadow"]["max_abs_dev"])
    if fg_dev > 1e-9:
        raise SystemExit(
            f"GATE FAILED: recomputed foreground curves differ from the published "
            f"cell_rcs.pkl by {fg_dev:.3e}. These dumps are not the cells Fig 3 was "
            f"built from; the sweep would enumerate the wrong model.")
    print(f"  GATE PASS: foreground curves reproduce the published arrays exactly "
          f"(max|dev|={fg_dev:.1e})")

    with open(FIG3_DATA + "inria_sp_battery_results.json") as f:
        s1 = json.load(f)["shadow"]["s1"]["phoenix|gt_pos"]
    print(f"  published s1 record: aurc_upper={s1['aurc_upper']:.6f} "
          f"aurc_loco={s1['aurc_loco']:.6f} gap={s1['gap']:.6f} n={s1['n_images']}")

    # ---- uniqueness: is FINAL-fig3.ipynb the only generator? --------------
    print("\n[6] Generator uniqueness across the provenance tree")
    import glob
    import re
    nb_hits = {}
    for nbp in sorted(glob.glob(PROV + "Python/**/*.ipynb", recursive=True)):
        try:
            with open(nbp) as f:
                cells = json.load(f).get("cells", [])
        except Exception:
            continue
        src = "\n".join("".join(c.get("source", [])) for c in cells)
        hits = [t for t in ("gap_at_1", "def aurc", "COVERAGE_GRID", "per_image_rc")
                if t in src]
        if hits:
            nb_hits[os.path.relpath(nbp, PROV)] = hits
    for k, v in nb_hits.items():
        print(f"  {k}: {v}")
    # the caption's number is transcribed from the printed gap_at_1, not hardcoded
    with open(FIG3_CODE) as f:
        fig_src = "\n".join("".join(c.get("source", []))
                            for c in json.load(f).get("cells", []))
    lit = [ln.strip() for ln in fig_src.splitlines() if re.search(r"0\.31\b", ln)]
    print(f"  literal '0.31' occurrences in the generator: {len(lit)} -> {lit}")
    panel_png = PROV + "data/final_figs/panel_a_main_riskcoverage.png"
    print(f"  generator's panel (a) output on disk: {os.path.isfile(panel_png)}")
    uniqueness = {
        "notebooks_containing_rc_or_aurc_code": nb_hits,
        "n_notebooks_with_gap_at_1": sum(1 for v in nb_hits.values() if "gap_at_1" in v),
        "literal_0_31_lines_in_generator": lit,
        "generator_panel_a_output_present": os.path.isfile(panel_png),
        "conclusion": "FINAL-fig3.ipynb is the only notebook in the provenance tree "
                      "that contains gap_at_1, def aurc or COVERAGE_GRID, so no "
                      "competing generator could have produced a different +0.31. The "
                      "only literal 0.31 in it is a bar-tick half-width, so the "
                      "caption's value was transcribed from the printed gap_at_1 "
                      "output rather than hardcoded.",
    }

    if DRY_RUN:
        print("\nDRY_RUN — nothing written.")
        return

    # =====================================================================
    # THE SWEEP
    # =====================================================================
    variants = []

    def add(vid, family, recipe, value, note=""):
        v = float(value)
        d = abs(v - TARGET)
        if d < MATCH_TOL:
            verdict = "MATCH"
        elif d <= NEAR_TOL:
            verdict = "NEAR"
        else:
            verdict = "no"
        variants.append({"id": vid, "family": family, "recipe": recipe,
                         "value": round(v, 6), "abs_delta_vs_target": round(d, 6),
                         "verdict": verdict, "note": note})
        return v

    def gap_percurve(mode, grid, rule, conf_kind="published", k_rule="ceil",
                     direction="loco_minus_upper", ids_sel=None, ce=False):
        """Per-image AURC, then mean of the paired per-image deltas."""
        per = {}
        for cond, pr in (("upper", pr_u), ("loco", pr_l)):
            dd = {}
            for iid in ids:
                p = pr[iid].astype(np.float32)
                r = (rc_ce_variant(gt[iid], p, mode, grid) if ce else
                     rc_variant(gt[iid], p, mode, grid, conf_kind, k_rule))
                if r is not None:
                    dd[iid] = integrate(r, grid, rule)
            per[cond] = dd
        common = sorted(set(per["upper"]) & set(per["loco"])) if ids_sel is None else ids_sel
        du = np.array([per["upper"][i] for i in common])
        dl = np.array([per["loco"][i] for i in common])
        delta = dl - du if direction == "loco_minus_upper" else du - dl
        return float(delta.mean()), delta, len(common)

    GRIDS = {
        "0.10-1.00 x19 (published)": np.linspace(0.10, 1.00, 19),
        "0.10-1.00 x20": np.linspace(0.10, 1.00, 20),
        "0.10-1.00 x21": np.linspace(0.10, 1.00, 21),
        "0.05-1.00 x20": np.linspace(0.05, 1.00, 20),
        "0.05-0.95 x19": np.linspace(0.05, 0.95, 19),
        "0.00-1.00 x21": np.linspace(0.00, 1.00, 21),
        "0.10-0.95 x18": np.linspace(0.10, 0.95, 18),
    }

    # ---- family A: integration rule x grid, 0/1 risk, foreground ---------
    print("\n[A] integration rule x coverage grid (0/1 risk, foreground, LOCO-upper)")
    for gname, g in GRIDS.items():
        for rule in ("trapz", "trapz_norm", "mean", "left", "right", "simpson"):
            if rule == "simpson" and len(g) % 2 == 0:
                continue
            v, _, n = gap_percurve("gt_shadow", g, rule)
            vid = f"A_{rule}_{gname.split()[0]}x{len(g)}"
            add(vid, "integration_rule",
                {"risk": "0/1", "population": "gt_shadow_conf_p", "grid": gname,
                 "n_points": len(g), "rule": rule, "aggregation": "per_image_AURC_mean",
                 "direction": "loco_minus_upper", "n_images": n,
                 "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
            print(f"  {vid:34s} {v:+.6f}")

    # aurc_mean20, exactly as the metric module implements it
    v20, d20, n20 = gap_percurve("gt_shadow", np.linspace(0.10, 1.0, 20), "mean",
                                 k_rule="int")
    add("A_mean20_metrics_module", "integration_rule",
        {"risk": "0/1", "population": "gt_shadow_conf_p",
         "grid": "0.10-1.00 x20", "n_points": 20, "rule": "mean",
         "k_rule": "int(c*n) truncation (metrics.aurc_mean20)",
         "aggregation": "per_image_AURC_mean", "direction": "loco_minus_upper",
         "n_images": n20, "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v20,
        "the tree's second live definition")
    print(f"  {'A_mean20_metrics_module':34s} {v20:+.6f}")

    # the two stored dump columns, read not recomputed
    du = df_u.set_index("image_id"); dl = df_l.set_index("image_id")
    for col, vid in (("aurc_fg", "A_stored_aurc_fg_trapz19"),
                     ("aurc_fg_mean20", "A_stored_aurc_fg_mean20")):
        m = np.isfinite(du.loc[ids, col].values) & np.isfinite(dl.loc[ids, col].values)
        v = float((dl.loc[ids, col].values[m] - du.loc[ids, col].values[m]).mean())
        add(vid, "integration_rule",
            {"risk": "0/1", "population": "gt_shadow_conf_p", "source": f"dump column {col}",
             "aggregation": "per_image_AURC_mean", "direction": "loco_minus_upper",
             "n_images": int(m.sum()),
             "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v,
            "stored contract column, read not recomputed")
        print(f"  {vid:34s} {v:+.6f}")

    # ---- family B: pointwise / terminal-value readouts -------------------
    print("\n[B] pointwise readouts of the same curves (no integration)")
    Mu = np.stack([RC[("upper", "gt_shadow")][i] for i in fg_ids])
    Ml = np.stack([RC[("loco", "gt_shadow")][i] for i in fg_ids])
    mu, ml = Mu.mean(0), Ml.mean(0)
    for label, idx in (("c=1.00", 18), ("c=0.90", 16), ("c=0.50", 8), ("c=0.10", 0)):
        v = float(ml[idx] - mu[idx])
        add(f"B_risk_at_{label.replace('c=', '')}", "pointwise",
            {"risk": "0/1", "population": "gt_shadow_conf_p",
             "readout": f"mean risk at coverage {label}, LOCO minus upper",
             "integration": "none", "aggregation": "mean curve then difference",
             "direction": "loco_minus_upper", "n_images": len(fg_ids),
             "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
        print(f"  {'B_risk_at_' + label:34s} {v:+.6f}")
    v = float(np.max(ml - mu))
    add("B_max_pointwise_gap", "pointwise",
        {"risk": "0/1", "population": "gt_shadow_conf_p",
         "readout": "max over the coverage grid of the pointwise gap",
         "integration": "none", "direction": "loco_minus_upper",
         "n_images": len(fg_ids),
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
    print(f"  {'B_max_pointwise_gap':34s} {v:+.6f}")

    # ---- family C: risk definition --------------------------------------
    print("\n[C] risk definition")
    v, _, n = gap_percurve("gt_shadow", PUB_GRID, "trapz", ce=True)
    add("C_ce_trapz19", "risk_definition",
        {"risk": "cross-entropy (App G.5)", "population": "gt_shadow_conf_p",
         "grid": "0.10-1.00 x19 (published)", "rule": "trapz",
         "aggregation": "per_image_AURC_mean", "direction": "loco_minus_upper",
         "n_images": n, "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
    print(f"  {'C_ce_trapz19':34s} {v:+.6f}")

    ce_u = {i: rc_ce_variant(gt[i], pr_u[i].astype(np.float32), "gt_shadow", PUB_GRID)
            for i in fg_ids}
    ce_l = {i: rc_ce_variant(gt[i], pr_l[i].astype(np.float32), "gt_shadow", PUB_GRID)
            for i in fg_ids}
    v = float(np.mean([ce_l[i][-1] - ce_u[i][-1] for i in fg_ids]))
    add("C_ce_risk_at_1.00", "risk_definition",
        {"risk": "cross-entropy", "population": "gt_shadow_conf_p",
         "readout": "mean CE risk at coverage 1.00, LOCO minus upper",
         "integration": "none", "direction": "loco_minus_upper", "n_images": len(fg_ids),
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
    print(f"  {'C_ce_risk_at_1.00':34s} {v:+.6f}")

    # 1-IoU risk, images ranked by mean foreground confidence
    def iou_curve(dfx, prx):
        rows = []
        for i in fg_ids:
            p = prx[i].astype(np.float32)
            g = gt[i]
            pred = p > 0.5
            inter = np.logical_and(pred, g).sum()
            union = np.logical_or(pred, g).sum()
            rows.append((float(p[g].mean()), 1.0 - (inter / union if union else 1.0)))
        conf = np.array([r[0] for r in rows]); risk = np.array([r[1] for r in rows])
        order = np.argsort(-conf); cum = np.cumsum(risk[order])
        out = np.empty(len(PUB_GRID))
        for j, c in enumerate(PUB_GRID):
            k = max(int(np.ceil(c * len(risk))), 1)
            out[j] = cum[k - 1] / k
        return out
    cu, cl = iou_curve(df_u, pr_u), iou_curve(df_l, pr_l)
    for lbl, val in (("C_1minusIoU_trapz19", integrate(cl, PUB_GRID, "trapz") -
                      integrate(cu, PUB_GRID, "trapz")),
                     ("C_1minusIoU_at_1.00", float(cl[-1] - cu[-1]))):
        add(lbl, "risk_definition",
            {"risk": "1 - foreground IoU per image", "population": "images ranked by mean p on GT shadow",
             "grid": "0.10-1.00 x19 (published)",
             "readout": "trapz19" if "trapz" in lbl else "value at coverage 1.00",
             "direction": "loco_minus_upper", "n_images": len(fg_ids),
             "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, val)
        print(f"  {lbl:34s} {val:+.6f}")

    # ---- family D: population -------------------------------------------
    print("\n[D] pixel population")
    for mode in ("gt_bg", "all_pixels"):
        v, _, n = gap_percurve(mode, PUB_GRID, "trapz")
        add(f"D_{mode}_trapz19", "population",
            {"risk": "0/1", "population": mode, "grid": "0.10-1.00 x19 (published)",
             "rule": "trapz", "aggregation": "per_image_AURC_mean",
             "direction": "loco_minus_upper", "n_images": n,
             "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
        print(f"  {'D_' + mode + '_trapz19':34s} {v:+.6f}")
        Mu2 = np.stack([RC[("upper", mode)][i] for i in sorted(
            set(RC[("upper", mode)]) & set(RC[("loco", mode)]))])
        Ml2 = np.stack([RC[("loco", mode)][i] for i in sorted(
            set(RC[("upper", mode)]) & set(RC[("loco", mode)]))])
        v = float(Ml2.mean(0)[-1] - Mu2.mean(0)[-1])
        add(f"D_{mode}_risk_at_1.00", "population",
            {"risk": "0/1", "population": mode,
             "readout": "mean risk at coverage 1.00, LOCO minus upper",
             "integration": "none", "direction": "loco_minus_upper", "n_images": len(Mu2),
             "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
        print(f"  {'D_' + mode + '_risk_at_1.00':34s} {v:+.6f}")

    # the foreground-minus-background asymmetry reading
    fgv, _, _ = gap_percurve("gt_shadow", PUB_GRID, "trapz")
    bgv, _, _ = gap_percurve("gt_bg", PUB_GRID, "trapz")
    add("D_fg_minus_bg_gap_trapz19", "population",
        {"risk": "0/1", "readout": "(fg AURC gap) minus (bg AURC gap)",
         "grid": "0.10-1.00 x19 (published)", "rule": "trapz",
         "direction": "loco_minus_upper",
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, fgv - bgv,
        "the foreground-vs-background asymmetry reading of the caption")
    print(f"  {'D_fg_minus_bg_gap_trapz19':34s} {fgv - bgv:+.6f}")

    # ---- family E: confidence score --------------------------------------
    print("\n[E] confidence score on the foreground population")
    for ck in ("margin", "maxprob"):
        for rule, lab in (("trapz", "trapz19"), ("mean", "mean19")):
            v, _, n = gap_percurve("gt_shadow", PUB_GRID, rule, conf_kind=ck)
            add(f"E_conf_{ck}_{lab}", "confidence_score",
                {"risk": "0/1", "population": "gt_shadow",
                 "confidence": {"margin": "|2p-1|", "maxprob": "max(p,1-p)"}[ck],
                 "grid": "0.10-1.00 x19 (published)", "rule": rule,
                 "aggregation": "per_image_AURC_mean", "direction": "loco_minus_upper",
                 "n_images": n, "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
            print(f"  {'E_conf_' + ck + '_' + lab:34s} {v:+.6f}")
        rcu = {i: rc_variant(gt[i], pr_u[i].astype(np.float32), "gt_shadow", PUB_GRID, ck)
               for i in fg_ids}
        rcl = {i: rc_variant(gt[i], pr_l[i].astype(np.float32), "gt_shadow", PUB_GRID, ck)
               for i in fg_ids}
        v = float(np.mean([rcl[i][-1] - rcu[i][-1] for i in fg_ids]))
        add(f"E_conf_{ck}_risk_at_1.00", "confidence_score",
            {"risk": "0/1", "population": "gt_shadow",
             "confidence": {"margin": "|2p-1|", "maxprob": "max(p,1-p)"}[ck],
             "readout": "mean risk at coverage 1.00", "integration": "none",
             "direction": "loco_minus_upper", "n_images": len(fg_ids),
             "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
        print(f"  {'E_conf_' + ck + '_risk_at_1.00':34s} {v:+.6f}")

    # ---- family F: aggregation -------------------------------------------
    print("\n[F] aggregation")
    v = integrate(ml, PUB_GRID, "trapz") - integrate(mu, PUB_GRID, "trapz")
    add("F_aurc_of_mean_curve_trapz19", "aggregation",
        {"risk": "0/1", "population": "gt_shadow_conf_p",
         "grid": "0.10-1.00 x19 (published)", "rule": "trapz",
         "aggregation": "mean risk-coverage curve, then integrate (panel b's rule)",
         "direction": "loco_minus_upper", "n_images": len(fg_ids),
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
    print(f"  {'F_aurc_of_mean_curve_trapz19':34s} {v:+.6f}")

    # pooled-pixel AURC: one global risk-coverage curve over all foreground pixels
    def pooled_rc(prx):
        confs, errs = [], []
        for i in fg_ids:
            p = prx[i].astype(np.float32).ravel()[gt[i].ravel()]
            confs.append(p); errs.append((p <= 0.5).astype(np.float32))
        conf = np.concatenate(confs); err = np.concatenate(errs)
        order = np.argsort(-conf); cum = np.cumsum(err[order]); n = len(conf)
        out = np.empty(len(PUB_GRID))
        for j, c in enumerate(PUB_GRID):
            k = max(int(np.ceil(c * n)), 1)
            out[j] = cum[k - 1] / k
        return out
    pu, pl = pooled_rc(pr_u), pooled_rc(pr_l)
    add("F_pooled_pixel_trapz19", "aggregation",
        {"risk": "0/1", "population": "gt_shadow_conf_p, pooled across images",
         "grid": "0.10-1.00 x19 (published)", "rule": "trapz",
         "aggregation": "pooled-pixel single curve", "direction": "loco_minus_upper",
         "n_images": len(fg_ids),
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"},
        integrate(pl, PUB_GRID, "trapz") - integrate(pu, PUB_GRID, "trapz"))
    add("F_pooled_pixel_risk_at_1.00", "aggregation",
        {"risk": "0/1", "population": "gt_shadow_conf_p, pooled across images",
         "readout": "pooled risk at coverage 1.00", "integration": "none",
         "direction": "loco_minus_upper", "n_images": len(fg_ids),
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"},
        float(pl[-1] - pu[-1]))
    print(f"  {'F_pooled_pixel_trapz19':34s} "
          f"{integrate(pl, PUB_GRID, 'trapz') - integrate(pu, PUB_GRID, 'trapz'):+.6f}")
    print(f"  {'F_pooled_pixel_risk_at_1.00':34s} {float(pl[-1] - pu[-1]):+.6f}")

    # all 150 images, dropped ones imputed at risk 1.0
    def with_all_150(prx):
        out = {}
        for i in ids:
            r = M.per_image_rc(gt[i], prx[i].astype(np.float32), "gt_shadow")
            out[i] = np.asarray(r, float) if r is not None else np.ones(len(PUB_GRID))
        return out
    au, al = with_all_150(pr_u), with_all_150(pr_l)
    add("F_all150_impute1_trapz19", "aggregation",
        {"risk": "0/1", "population": "gt_shadow_conf_p",
         "grid": "0.10-1.00 x19 (published)", "rule": "trapz",
         "aggregation": "all 150 images, the 21 sub-50-px images imputed at risk 1.0",
         "direction": "loco_minus_upper", "n_images": 150,
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"},
        float(np.mean([integrate(al[i], PUB_GRID, "trapz") -
                       integrate(au[i], PUB_GRID, "trapz") for i in ids])))
    add("F_all150_impute1_risk_at_1.00", "aggregation",
        {"risk": "0/1", "population": "gt_shadow_conf_p",
         "readout": "mean risk at coverage 1.00 over all 150, dropped imputed at 1.0",
         "integration": "none", "direction": "loco_minus_upper", "n_images": 150,
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"},
        float(np.mean([al[i][-1] - au[i][-1] for i in ids])))

    # ---- family G: cell scope --------------------------------------------
    print("\n[G] cell scope (published curves; the caption names one cell)")
    def pub_gap(uid, lid, mode, readout):
        U, L = pub[uid][mode], pub[lid][mode]
        common = sorted(set(U) & set(L))
        if len(common) < 5:
            return None, 0
        mU = np.stack([np.asarray(U[i], float) for i in common]).mean(0)
        mL = np.stack([np.asarray(L[i], float) for i in common]).mean(0)
        if readout == "trapz19":
            return integrate(mL, PUB_GRID, "trapz") - integrate(mU, PUB_GRID, "trapz"), len(common)
        return float(mL[-1] - mU[-1]), len(common)

    for readout in ("trapz19", "risk_at_1.00"):
        vals = []
        for arch in ("mamnet", "oglanet", "dinov3"):
            for city in ("chicago", "miami", "phoenix"):
                g, n = pub_gap(f"{arch}_upper_{city}", f"{arch}_vanilla_{city}",
                               "gt_shadow", readout)
                if g is not None:
                    vals.append(g)
        add(f"G_mean_over_9_vanilla_cells_{readout}", "cell_scope",
            {"risk": "0/1", "population": "gt_shadow_conf_p", "readout": readout,
             "aggregation": "mean over 9 vanilla LOCO cells (3 arch x 3 city)",
             "direction": "loco_minus_upper", "n_cells": len(vals),
             "cell_scope": "9 cells — NOT the caption scope"}, float(np.mean(vals)),
            "wrong scope: the caption names MAMNet/Phoenix alone")
        print(f"  {'G_mean_over_9_' + readout:34s} {np.mean(vals):+.6f}  (n_cells={len(vals)})")

        vals3 = [pub_gap(f"mamnet_upper_{c}", f"mamnet_vanilla_{c}", "gt_shadow", readout)[0]
                 for c in ("chicago", "miami", "phoenix")]
        add(f"G_mean_over_3_mamnet_cities_{readout}", "cell_scope",
            {"risk": "0/1", "population": "gt_shadow_conf_p", "readout": readout,
             "aggregation": "mean over MAMNet's 3 held-out cities",
             "direction": "loco_minus_upper", "n_cells": 3,
             "cell_scope": "3 cells — NOT the caption scope"}, float(np.mean(vals3)),
            "wrong scope: the caption names Phoenix alone")
        print(f"  {'G_mean_over_3_mamnet_' + readout:34s} {np.mean(vals3):+.6f}")

    g, n = pub_gap(PUB_UPPER, CELL_FDA_PUB, "gt_shadow", "risk_at_1.00")
    add("G_fda_risk_at_1.00", "cell_scope",
        {"risk": "0/1", "population": "gt_shadow_conf_p",
         "readout": "risk at coverage 1.00", "condition": "LOCO+FDA minus upper",
         "direction": "fda_minus_upper", "n_images": n,
         "cell_scope": "mamnet/fda/phoenix — the caption's third curve"}, g,
        "the caption's 'FDA narrows it'")
    print(f"  {'G_fda_risk_at_1.00':34s} {g:+.6f}")

    # ---- family H: direction and bootstrap --------------------------------
    print("\n[H] direction and bootstrap")
    v, _, n = gap_percurve("gt_shadow", PUB_GRID, "trapz", direction="upper_minus_loco")
    add("H_direction_upper_minus_loco_trapz19", "direction",
        {"risk": "0/1", "population": "gt_shadow_conf_p",
         "grid": "0.10-1.00 x19 (published)", "rule": "trapz",
         "direction": "upper_minus_loco (sign flip)", "n_images": n,
         "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, v)
    print(f"  {'H_upper_minus_loco_trapz19':34s} {v:+.6f}")

    d_trapz = np.array([integrate(RC[("loco", "gt_shadow")][i], PUB_GRID, "trapz") -
                        integrate(RC[("upper", "gt_shadow")][i], PUB_GRID, "trapz")
                        for i in fg_ids])
    d_at1 = np.array([RC[("loco", "gt_shadow")][i][-1] -
                      RC[("upper", "gt_shadow")][i][-1] for i in fg_ids])
    boots = {}
    for lbl, d in (("trapz19", d_trapz), ("risk_at_1.00", d_at1)):
        for B in (B_BOOT_PUB, B_BOOT):
            pt, bm, lo, hi = paired_bootstrap(d, B, SEED)
            boots[f"{lbl}_B{B}"] = {"point": pt, "boot_mean": bm, "ci95": [lo, hi], "B": B}
            add(f"H_bootstrap_mean_{lbl}_B{B}", "bootstrap",
                {"risk": "0/1", "population": "gt_shadow_conf_p", "readout": lbl,
                 "aggregation": f"image-level paired bootstrap mean, B={B}, seed={SEED}",
                 "direction": "loco_minus_upper", "n_images": len(fg_ids),
                 "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, bm)
            print(f"  {'H_boot_' + lbl + '_B' + str(B):34s} {bm:+.6f}  "
                  f"point={pt:+.6f} CI95=[{lo:+.4f},{hi:+.4f}]")
        pt, bm, lo, hi = paired_bootstrap(d, B_BOOT, SEED, stat="median")
        add(f"H_bootstrap_median_{lbl}_B{B_BOOT}", "bootstrap",
            {"risk": "0/1", "population": "gt_shadow_conf_p", "readout": lbl,
             "aggregation": f"image-level paired bootstrap median, B={B_BOOT}, seed={SEED}",
             "direction": "loco_minus_upper", "n_images": len(fg_ids),
             "cell_scope": "mamnet/vanilla/phoenix (caption scope)"}, bm)

    # =====================================================================
    # VERDICT
    # =====================================================================
    matches = [v for v in variants if v["verdict"] == "MATCH"]
    nears = [v for v in variants if v["verdict"] == "NEAR"]
    print("\n" + "=" * 78)
    print(f"SWEPT {len(variants)} variants.  MATCH={len(matches)}  NEAR={len(nears)}")
    for v in matches:
        print(f"  MATCH {v['id']:34s} {v['value']:+.6f}  (|d|={v['abs_delta_vs_target']:.6f})")
    for v in nears:
        print(f"  NEAR  {v['id']:34s} {v['value']:+.6f}  (|d|={v['abs_delta_vs_target']:.6f})")
    print("=" * 78)

    on_scope = [v for v in matches
                if "caption scope" in str(v["recipe"].get("cell_scope", ""))]
    verdict = "RECOVERED" if on_scope else ("NEAR" if nears else "NOT RECOVERED")

    # Six variants land inside the match band, but they are not six recipes.
    # Four are the same quantity re-expressed (the confidence score cannot matter
    # at full coverage; a bootstrap mean of a statistic is that statistic), and
    # one is a different coverage level that happens to sit near +0.31. The
    # recovery rests on the figure generator's own source line, not on which
    # number is numerically closest — B_risk_at_0.50 is closer and is wrong.
    PRIMARY = "B_risk_at_1.00"
    primary = next(v for v in variants if v["id"] == PRIMARY)
    alias_ids = ["E_conf_margin_risk_at_1.00", "E_conf_maxprob_risk_at_1.00",
                 "H_bootstrap_mean_risk_at_1.00_B1000",
                 "H_bootstrap_mean_risk_at_1.00_B10000"]
    match_analysis = {
        "n_in_match_band": len(matches),
        "n_distinct_quantities_matching": 2,
        "primary": {
            "id": PRIMARY, "value": primary["value"],
            "why": "This is the quantity the published figure generator computes on "
                   "the line that prints the caption's number. Recovery is by source, "
                   "confirmed by value.",
        },
        "aliases_of_primary": [
            {"id": i, "value": next(v for v in variants if v["id"] == i)["value"],
             "why": "the same quantity as " + PRIMARY + ": at coverage 1.00 the whole "
                    "population is covered, so the confidence ranking cannot change "
                    "the value; a bootstrap mean estimates the same statistic"}
            for i in alias_ids],
        "numerical_coincidences": [
            {"id": "B_risk_at_0.50",
             "value": next(v for v in variants if v["id"] == "B_risk_at_0.50")["value"],
             "why": "closer to +0.31 than the recovered recipe (|d|=0.000069 vs "
                    "0.001852) but a different quantity: the risk gap at coverage "
                    "0.50. Nothing in the figure code, the caption or the paper reads "
                    "the curve at c=0.50. Listed so that a proximity argument cannot "
                    "be mistaken for the recovery."}],
        "note": "A search that ranked variants by closeness to +0.31 would return the "
                "wrong recipe. The recovery is established by reading the generator.",
    }

    # ---- results JSON ----------------------------------------------------
    payload = {
        "id": "06_aurc_recovery",
        "arm": "AURC rule recovery (HLT-ordered; not a lettered arm)",
        "answers": ["E-7"],
        "question": "Which definition reproduces Fig 3(a)'s published foreground "
                    "selective-prediction number, +0.31?",
        "runs": [run_u["run_id"], run_l["run_id"]],
        "index_sha256": sha256_file(INDEX_PARQUET),
        "index_sha256_note": f"INDEX.parquet of {DUMPS}",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/06_aurc_recovery.py",
        "git_hash": None,
        "git_hash_note": "project root is not a git repository (CLAUDE.md)",
        "seed": SEED,

        "target": {
            "printed_value": TARGET,
            "source": "final_paper/figures/fig_diagnostics.tex, caption of "
                      "\\label{fig:diagnostics} panel (a)",
            "caption_text": "Risk-coverage curves on foreground pixels (MAMNet, "
                            "Phoenix held out): LOCO accumulates $\\Delta$AURC$=+0.31$ "
                            "over the upper bound; FDA narrows it.",
            "caption_scope": "MAMNet, Phoenix held out, foreground pixels, "
                             "LOCO (vanilla) minus upper bound, single cell",
            "match_criterion": f"|value - {TARGET}| < {MATCH_TOL} (rounds to +0.31 at "
                               f"two decimals) on the caption's own cell scope",
            "near_criterion": f"{MATCH_TOL} <= |value - {TARGET}| <= {NEAR_TOL}",
        },

        "provenance": {
            "figure_code": FIG3_CODE,
            "figure_code_sha256": sha256_file(FIG3_CODE),
            "metric_module": METRIC_MODULE_DIR + "/metrics.py",
            "metric_module_sha256": sha256_file(METRIC_MODULE_DIR + "/metrics.py"),
            "published_curves": FIG3_DATA + "cell_rcs.pkl",
            "published_curves_sha256": sha256_file(FIG3_DATA + "cell_rcs.pkl"),
            "published_s1_record": FIG3_DATA + "inria_sp_battery_results.json",
            "gt_masks": GT_ROOT + f"{GT_CITY}/{GT_RES}/{GT_SPLIT}/masks",
        },

        "generator_uniqueness": uniqueness,

        "gate_checks": {
            "variant_generator_vs_metric_module_max_abs_dev": worst,
            "recomputed_vs_published_curves": checks,
            "published_s1_record": s1 and {k: s1[k] for k in
                                           ("aurc_upper", "aurc_loco", "gap", "gap_ci",
                                            "n_images")},
        },

        "variants": variants,
        "n_variants": len(variants),

        "verdict": verdict,
        "match_analysis": match_analysis,
        "recovered_recipe": ({
            "id": primary["id"],
            "value": primary["value"],
            "statement": "Fig 3(a)'s +0.31 is not an AURC. It is the difference in "
                         "mean per-image 0/1 risk at coverage c = 1.00 — the terminal "
                         "point of the risk-coverage curve, i.e. the plain "
                         "full-coverage foreground error rate — between LOCO "
                         "(mamnet_vanilla_phoenix) and the upper bound "
                         "(mamnet_upper_phoenix), averaged over the 129 test images "
                         "with at least 50 GT shadow pixels. No integration over the "
                         "coverage grid enters it.",
            "source_line": "Shade Maps/Python/FINAL-fig3.ipynb: "
                           "gap_at_1 = curves['vanilla'][0][-1] - curves['upper'][0][-1]; "
                           "print(f\"  Panel (a) DeltaAURC at coverage=1.0: {gap_at_1:+.3f}\")",
        } if on_scope else None),

        "conditions": [],
        "tests": [],
        "headline": "",
        "caveats": [],
        "figure": None,
        "caption": None,
        "figure_note": "No plot. This analysis recovers a definition; its output is "
                       "the variant enumeration, not a figure. Nothing here is "
                       "intended for a float in the paper.",
    }

    # conditions / tests, in CONTRACT §6 shape
    for lbl, d in (("trapz19", d_trapz), ("risk_at_1.00", d_at1)):
        pt, bm, lo, hi = paired_bootstrap(d, B_BOOT, SEED)
        payload["conditions"].append({
            "label": f"MAMNet / Phoenix held out / foreground / {lbl}",
            "metric": ("aurc_trapz19 gap (LOCO - upper)" if lbl == "trapz19"
                       else "risk at coverage 1.00, gap (LOCO - upper)"),
            "mean": round(pt, 6),
            "stderr": round(float(d.std(ddof=1) / np.sqrt(len(d))), 6),
            "n": int(len(d)),
            "ci95": [round(lo, 6), round(hi, 6)],
            "ci_method": f"paired_image_bootstrap_B{B_BOOT}",
            "baseline": "mamnet-none-upper-r450-phoenix-hr-s0",
        })
        payload["tests"].append({
            "name": "image-level paired bootstrap",
            "compares": [run_l["run_id"], run_u["run_id"]],
            "statistic": round(pt, 6), "B": B_BOOT, "two_tailed": True,
            "readout": lbl,
            "p": None,
            "p_note": "no null hypothesis is under test here; this analysis recovers a "
                      "definition, it does not make a claim about the gap",
        })

    payload["bootstraps"] = boots
    pub_trapz19 = next(v for v in variants
                       if v["recipe"].get("grid") == "0.10-1.00 x19 (published)"
                       and v["recipe"].get("rule") == "trapz"
                       and v["recipe"].get("population") == "gt_shadow_conf_p"
                       and v["family"] == "integration_rule")
    payload["published_definition_value"] = {
        "id": pub_trapz19["id"], "value": pub_trapz19["value"],
        "note": "aurc_trapz19 on the caption's cell — the paper's own stated AURC",
    }
    payload["headline"] = (
        f"Fig 3(a)'s printed ΔAURC = +0.31 reproduces as the full-coverage "
        f"foreground risk gap ({primary['value']:+.6f}), the terminal point of the "
        f"risk-coverage curve, and not as any integral of that curve; the same cell's "
        f"AURC gap under the paper's own stated definition is "
        f"{pub_trapz19['value']:+.6f}."
        if on_scope else
        f"No definition among {len(variants)} plausible variants reproduces Fig 3(a)'s "
        f"printed +0.31 on the caption's cell scope."
    )
    payload["caveats"] = [
        "The recovered rule is read off the published figure generator "
        "(Shade Maps/Python/FINAL-fig3.ipynb), not inferred from the number alone. "
        "The notebook's own print statement labels the quantity 'DeltaAURC at "
        "coverage=1.0'; the value it prints is the risk at coverage 1.00, and the "
        "notebook's aurc() helper is never applied to it.",
        "The notebook's aurc() helper IS applied in panel (b) (lines 249-250), which "
        "therefore plots a genuine trapz19 AURC gap. Panel (a)'s caption number and "
        "panel (b)'s bars are two different quantities under one axis label.",
        "Panel (a)'s inset is also a genuine trapz19 AURC gap: the project's own "
        f"inria_sp_battery_results.json stores this cell's gap as {s1['gap']:.6f} with "
        f"95% CI {[round(x, 6) for x in s1['gap_ci']]}. The caption's +0.31 sits above "
        "the upper end of that CI.",
        "trapz19 is an unnormalised np.trapezoid over a grid spanning 0.9, so it is "
        "about 0.9x the mean risk over the grid. Dividing it by 0.9 does not reach "
        "+0.31 either; that variant is in the sweep.",
        "n = 129, not 150: metrics.per_image_rc returns None for images with fewer "
        "than 50 GT shadow pixels. 21 of the 150 Phoenix test images are dropped, "
        "identically in both conditions.",
        "The foreground 0/1 risk-coverage curve carries no ranking information. "
        "Errors on the GT-shadow population are exactly the pixels with p <= 0.5 and "
        "the confidence used is p, so sorting by confidence puts every correct pixel "
        "before every error and rc(c) = max(0, 1 - recall/c) up to ceil() rounding. "
        "Every foreground 0/1 AURC in the paper is a deterministic function of "
        "per-image recall alone.",
        "Scope of this analysis: the Fig 3(a) number only. No Section 4.3 claim was "
        "recomputed; the +0.136 / +0.006 pooled gaps, the 2.4x CE under-reporting, "
        "and the 59/63 recoverable-threshold count are untouched here.",
        "The cross-entropy variants (C_ce_trapz19, C_ce_risk_at_1.00) are definition "
        "probes on this one cell. They are NOT a check of Appendix E's pooled 2.4x "
        "CE-over-0/1 under-reporting ratio, which is a different quantity pooled over "
        "63 cells, and must not be read as one.",
        "Six variants land inside the match band but they are not six recipes: four "
        "are the recovered quantity re-expressed and one (B_risk_at_0.50, +0.309931) "
        "is a different coverage level that happens to sit nearer +0.31 than the "
        "recovered recipe does. See match_analysis. The recovery is established by "
        "reading the figure generator, not by numerical proximity.",
        "Three of the swept variants are read from the published curve archive rather "
        "than recomputed, because their cells have no dump on this box: the two "
        "G_mean_over_* families and G_fda_risk_at_1.00. They are marked with their "
        "source in the variant records and none of them is the recovered recipe.",
        "Cells used: mamnet-none-upper-r450-phoenix-hr-s0 and "
        "mamnet-vanilla-loco-r450-phoenix-hr-s0 from "
        "dumps/st_r450_published_recompute/, the published checkpoints' dumps. The "
        "MAMNet+FDA cell has no dump on this box; the one number taken for it "
        "(G_fda_risk_at_1.00) is read from the published curves and is marked as such.",
        "Cell-selection hazard, found while running this analysis and reported "
        "separately: dumps/st/ carries the same two run_ids, but those directories now "
        "hold arm C RETRAINS (run.json arm=C, legacy_run_dir=None, ckpt under "
        "outputs/armC/). Their foreground trapz19 gap is +0.357587, not +0.261510, and "
        "their per-image curves differ from the published ones on 123 of 129 images. "
        "Any analysis that reaches for dumps/st/{cell} by name to touch a published "
        "Fig 3 quantity gets the wrong model. EXP-01's AURC row is recorded against "
        "run_ids that now resolve to the retrains.",
        "CONTRACT.md was not edited. The AURC blank in section 3 stays open; "
        "ratification is the operator's and HLT's.",
    ]

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {OUT_JSON}")
    print(f"VERDICT: {verdict}")
    print(f"HEADLINE: {payload['headline']}")


if __name__ == "__main__":
    main()
