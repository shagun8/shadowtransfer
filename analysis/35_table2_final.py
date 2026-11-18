#!/usr/bin/env python3
"""
results/35: Table 2 (tab:loco-main) and Appendix F (app:loco-full, Tables
30-35 of the origin/main build) rebuilt from per-cell dumps at K=2.

Answers E-2 (the two upper-bound tables disagree on the architecture ranking)
and H2 (number-consistency audit of Table 2 / Appendix F / section 3.4).

DECISIONS TAKEN BY THE AUTHOR BEFORE THIS JOB (not re-litigated here):
  1. The paper's metric is two-class boundary-tolerant mIoU at K=2 (the *_tol
     count columns, CONTRACT.md section 3). The printed Table 2 and Appendix F
     are K=5 (the *_k5 columns) and are rebuilt.
  2. Table 2's per-architecture block reproduces from no per-cell source and is
     replaced by a rebuild from per-cell values. Its origin is not searched for.

INPUTS (all K=2):
  * high-res LOCO (63): the 24 Table-2 cells among results/21 finding 1's 27
    budget-asymmetric cells take their matched-450 retrains, resolved by
    analysis/31's resolve_n450 (OGLANet SegDeSiC Phoenix from its -bs4
    re-score). The other 3 of the 27 are OGLANet-SIB, which is not a Table 2
    row. The remaining 39 come from dumps/st_r450_published_png.
  * high-res upper (9), mid-res upper (9): dumps/st_r450_published_png.
  * mid-res LOCO (63): dumps/st_r450_published_png/*-loco-r450-*-mr-s0, written
    by Delta's job (delta_runs/midres_png_score/); PHASE B only, gate-passed cells only.
  * SAR, BSM, CrossEarth rows: analysis/31's resolve_row, unchanged.

REGRESSION CHECK kept from the Step-0 gate: every published PNG cell that
Appendix F prints is re-scored at K=5 and must equal the printed Upper/LOCO to
2 dp (|diff| <= 0.005): 81 cells. The 63 mid-res LOCO cells pass a separate
gate (operator addendum): perimage sha256 equal to Delta's manifest and
|K=5 - Appendix F| < 0.01; failing cells are excluded, never adjusted.

SIGNIFICANCE: python/statistical_analysis.py:bootstrap_comparison (Delta only,
not on the analysis host), whose behaviour python/armc_ladder_stats.py documents and
replicates: B resamples WITH REPLACEMENT OF THE 150 TEST IMAGES of the cell's
per-image delta vector (LOCO minus within-city upper, paired on image_id),
percentile CI at 2.5/97.5, two-tailed p = min(2 * prop_wrong, 1) where
prop_wrong is the share of replicates on the far side of zero from the
observed mean. Resampling unit: the test image. The unit is confirmed on the
K=5 side against the printed CIs and daggers (and against an unpaired
alternative) inside this script.

Run:  PYTHONNOUSERSITE=1 \\
      python \\
      analysis/35_table2_final.py [--dry-run]
"""

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
# "A": everything except mid-res LOCO; prints tables, writes nothing.
# "B": full run including mid-res LOCO; writes the results JSON and LaTeX.
PHASE = "B"
B = 10000
BOOT_SEED = 0                       # one fixed seed, reused for every cell
ALPHA = 0.05
NN, NAME = 35, "table2_final"

PAPER = os.path.join(BASE, "final_paper")
# Pinned for release to the commit results/35 records in paper.commit. The original run read
# "origin/main", which a later rewrite of Table 2 moved past the printed K=5 anchors checked below.
PAPER_REF = "686a50f2e89e08054bce51674527315edba8de96"
PNG = os.path.join(BASE, "dumps", "st_r450_published_png")
RECOMPUTE = os.path.join(BASE, "dumps", "st_r450_published_recompute")
INV = os.path.join(BASE, "analysis", "_inv")
OUT = os.path.join(BASE, "analysis", "out")
RESULTS = os.path.join(BASE, "results", f"{NN}_{NAME}.json")
R31_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "31_internal_sib_worst_case_tgtfda.py")
STATS_CSV = os.path.join(os.environ["PROJECT_ROOT"], "data",
                         "model_results/statistical_results.csv")
# Mid-res LOCO gate (operator addendum, 2026-11-03). Delta writes the 63 dumps
# with appendix_f_gate.pass = null; the gate runs HERE: (1) perimage.parquet
# sha256 equals the value Delta reports, (2) the K=5 two-class cell mean from
# the dump's *_k5 counts is within MIDRES_GATE_TOL of the parsed Appendix F
# LOCO value. A cell failing either is excluded from every output and listed
# in the caveats with its diff. Nothing is adjusted to make a cell pass.
MIDRES_GATE_TOL = 0.01
# Where Delta's sha256 list may land, searched in order; the first present is
# used and its path and sha256 are recorded.
MIDRES_SHA_CANDIDATES = [
    # Delta's handoff manifest (operator, 2026-11-03): "sha256  size_bytes  path",
    # one line per perimage.parquet and run.json, '#' header lines.
    os.path.join(BASE, "delta_runs", "midres_png_score", "sync_manifest_midres63.txt"),
    os.path.join(BASE, "runs", "midres_png_score", "results.json"),
    os.path.join(BASE, "delta_runs", "midres_png_score", "results.json"),
    os.path.join(BASE, "runs", "midres_png_score", "SHA256SUMS"),
    os.path.join(BASE, "delta_runs", "midres_png_score", "SHA256SUMS"),
]
MIDRES_LIST = os.path.join(BASE, "delta_runs", "midres_png_score", "midres63.list")
DELTA_RESULTS = os.path.join(BASE, "delta_runs", "midres_png_score", "results.json")
SHA_KEYS = ("perimage_sha256", "perimage_parquet_sha256", "parquet_sha256", "dump_sha256")

ARCHS_APPF = ("mamnet", "oglanet", "dinov3")      # Appendix F table order
ARCHS_T2 = ("dinov3", "mamnet", "oglanet")        # Table 2 column order D M O
ARCH_LABEL = {"mamnet": "MAMNet", "oglanet": "OGLANet", "dinov3": "DINOv3",
              "crossearth": "CrossEarth"}
CITIES = ("chicago", "miami", "phoenix")
RESES = ("hr", "mr")
METHODS = ("vanilla", "fda", "segdesic", "iim", "mrfp_plus", "fada", "isw")
ADAPT = METHODS[1:]
MLABEL = {"vanilla": "Vanilla", "fda": "FDA", "segdesic": "SegDeSiC",
          "iim": "IIM", "mrfp_plus": "MRFP+", "fada": "FADA", "isw": "ISW",
          "sar": "SAR", "bsm": "BSM", "sib": "SIB", "upper": "Upper"}
APPF_LABEL_TO_METHOD = {v: k for k, v in MLABEL.items()}

# The 24 Table-2 cells that must be matched-450 retrains (results/21 finding 1
# minus the 3 OGLANet-SIB cells, which are not Table 2 rows).
EXPECTED_N450 = ({("mamnet", m, c) for m in METHODS for c in CITIES}
                 | {("oglanet", "segdesic", c) for c in CITIES})
FINDING1_NOT_IN_TABLE2 = [f"oglanet/sib/{c}" for c in CITIES]

# BSM DINOv3/Chicago collapsed in training; its dump dir carries only .FAILED.
BSM_EXCLUDED = {("dinov3", "chicago"):
                "training collapse at seed 0 (best validation mIoU 5.51 "
                "against the codebase floor of 10.0); the dump directory "
                "carries only .FAILED; excluded as results/13 and results/31 "
                "exclude it, never imputed"}


def fail(msg):
    raise SystemExit(f"FATAL: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_r31():
    spec = importlib.util.spec_from_file_location("r31_rows", R31_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for flag in ("N450", "SAR_N450"):
        if getattr(mod, flag) is not True:
            fail(f"analysis/31 {flag} is not True; its row resolver is not the "
                 "one results/31 reports")
    return mod


R31 = load_r31()


# ----------------------------------------------------------------------------
# metric
# ----------------------------------------------------------------------------

def two_class_k2(df):
    """Per-image two-class tolerant mIoU at K=2: analysis/31's implementation,
    which asserts itself against the stored columns inside read_cell."""
    return R31.two_class_tol_k2(df)[0]


def two_class_k5(df):
    """Same formula on the K=5 count columns (Appendix F's printed basis)."""
    tp, fp, fn, tn = (df[f"{c}_k5"].to_numpy(np.float64)
                      for c in ("tp", "fp", "fn", "tn"))
    dfg, dbg = tp + fp + fn, tn + fp + fn
    if (dbg <= 0).any():
        fail("K=5 background IoU denominator is zero on some image")
    fg = np.where(dfg > 0, tp / np.where(dfg > 0, dfg, 1.0), 0.0) * 100.0
    bg = tn / dbg * 100.0
    return (fg + bg) / 2.0


# ----------------------------------------------------------------------------
# bootstrap: the published protocol
# ----------------------------------------------------------------------------

def boot_published(delta, b=B, seed=BOOT_SEED):
    """statistical_analysis.py:bootstrap_comparison as armc_ladder_stats.py
    documents it. Resampling unit: the test image (paired delta vector)."""
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    rng = np.random.default_rng(seed)
    boots = delta[rng.integers(0, n, size=(b, n))].mean(axis=1)
    obs = float(delta.mean())
    prop_wrong = float((boots <= 0).mean()) if obs >= 0 else float((boots >= 0).mean())
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return dict(mean=obs, stderr=float(delta.std(ddof=1) / np.sqrt(n)), n=int(n),
                ci95=[lo, hi], p=min(2.0 * prop_wrong, 1.0), B=b, seed=seed,
                two_tailed=True, ci_method=f"paired_image_bootstrap_B{b}")


def boot_unpaired(a, u, b=B, seed=BOOT_SEED):
    """The rejected alternative unit: LOCO and upper images resampled
    independently. Used only to show which unit the printed CIs came from."""
    rng = np.random.default_rng(seed)
    n = a.size
    ra = a[rng.integers(0, n, size=(b, n))].mean(axis=1)
    ru = u[rng.integers(0, n, size=(b, n))].mean(axis=1)
    d = ra - ru
    obs = float(a.mean() - u.mean())
    pw = float((d <= 0).mean()) if obs >= 0 else float((d >= 0).mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(ci95=[float(lo), float(hi)], p=min(2 * pw, 1.0))


# ----------------------------------------------------------------------------
# printed paper (origin/main)
# ----------------------------------------------------------------------------

def git_show(path):
    r = subprocess.run(["git", "-C", PAPER, "show", f"{PAPER_REF}:{path}"],
                       capture_output=True, text=True)
    if r.returncode:
        fail(f"git show {PAPER_REF}:{path}: {r.stderr.strip()}")
    return r.stdout


def paper_commit():
    return subprocess.run(["git", "-C", PAPER, "rev-parse", PAPER_REF],
                          capture_output=True, text=True).stdout.strip()


def parse_num(s):
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    return float(s.replace("$", "").replace("+", "").strip())


def parse_appf():
    """Every per-cell row of app_loco_extended.tex: Upper, LOCO, Delta, CI, p,
    dagger. Returns a DataFrame keyed by (arch, method, city, res)."""
    tex = git_show("appendix/app_loco_extended.tex")
    rows, arch, meth = [], None, None
    for ln, line in enumerate(tex.splitlines(), 1):
        m = re.search(r"label\{tab:loco-full-(\w+)\}", line)
        if m:
            arch = m.group(1)
        parts = [p.strip() for p in line.rstrip("\\ ").split("&")]
        if (arch in ARCHS_APPF and len(parts) == 8
                and parts[1] in ("Chicago", "Miami", "Phoenix")):
            if parts[0]:
                meth = APPF_LABEL_TO_METHOD[parts[0]]
            ci = re.match(r"\$\[([+\-\d.]+),\\,([+\-\d.]+)\]\$", parts[6])
            pv = parts[7]
            dag = "dagger" in pv
            pnum = re.search(r"(<\\!)?\.(\d+)", pv)
            rows.append(dict(arch=arch, method=meth, city=parts[1].lower(),
                             res=parts[2].lower(), up=float(parts[3]),
                             loco=float(parts[4]), delta=parse_num(parts[5]),
                             ci_lo=float(ci.group(1)), ci_hi=float(ci.group(2)),
                             p=(0.0 if pnum.group(1) else float("." + pnum.group(2))),
                             p_floor=bool(pnum.group(1)), dagger=dag,
                             sig=not dag, bold="mathbf" in parts[5], line=ln))
    F = pd.DataFrame(rows)
    if len(F) != 126:
        fail(f"parsed {len(F)} Appendix F cell rows, expected 126")
    if F.duplicated(["arch", "method", "city", "res"]).any():
        fail("duplicate Appendix F cell rows")
    return F, tex


def parse_table2():
    tex = git_show("tables/tab_loco_main.tex")
    out = {}
    for line in tex.splitlines():
        parts = [p.strip() for p in line.rstrip("\\ ").split("&")]
        if len(parts) == 12 and parts[0] in APPF_LABEL_TO_METHOD:
            v = [parse_num(p) for p in parts[1:]]
            out[APPF_LABEL_TO_METHOD[parts[0]]] = dict(
                loco=dict(zip(ARCHS_T2, v[0:3])), delta=dict(zip(ARCHS_T2, v[3:6])),
                sig6=dict(zip(ARCHS_T2, v[6:9])), pooled_delta=v[9], sig18=v[10])
        if parts[0].startswith(r"\textit{Upper bound}"):
            nums = re.findall(r"\\mathit\{([\d.]+)\}", line)
            out["upper"] = dict(zip(ARCHS_T2, map(float, nums)))
        if parts[0].startswith(r"\textit{Avg.\ $\Delta$}"):
            out["avg_delta"] = dict(zip(ARCHS_T2, (parse_num(p) for p in parts[4:7])))
        if parts[0].startswith(r"\textit{\# Sig."):
            out["sig42"] = dict(zip(ARCHS_T2, (float(p) for p in parts[7:10])))
    if len([k for k in out if k in METHODS]) != 7:
        fail("Table 2 parse did not find the 7 method rows")
    return out, tex


# ----------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------

def png_rid(arch, method, city, res):
    if method == "upper":
        return f"{arch}-none-upper-r450-{city}-{res}-s0"
    return f"{arch}-{method}-loco-r450-{city}-{res}-s0"


def require_markers(root, rid):
    d = os.path.join(root, rid)
    if not os.path.isfile(os.path.join(d, ".DONE")):
        fail(f"{rid}: no .DONE marker")
    if os.path.isfile(os.path.join(d, ".FAILED")):
        fail(f"{rid}: .FAILED marker present")


STORED_TOL = 1e-6   # analysis/31 uses 1e-9; mamnet-none-upper-r450-phoenix-mr-s0
                    # (never read by 31) differs from its stored column by 2e-9


def read_cell(root, rid):
    """analysis/31's read_cell validation, unchanged except the stored-column
    tolerance (STORED_TOL), and returning the measured max deviation."""
    d = os.path.join(root, rid)
    pq, sj = os.path.join(d, "perimage.parquet"), os.path.join(d, "run.json")
    if not os.path.isfile(pq):
        fail(f"{rid}: no perimage.parquet")
    if not os.path.isfile(sj):
        fail(f"{rid}: perimage.parquet with no run.json sidecar (CONTRACT.md section 7)")
    df = pd.read_parquet(pq)
    meta = json.load(open(sj))
    missing = [c for c in R31.CONTRACT_COLS if c not in df.columns]
    if missing:
        fail(f"{rid}: CONTRACT.md section 3 columns absent: {missing}")
    for col, kind in R31.CONTRACT_COLS.items():
        if df[col].isna().all():
            continue
        ok = {"int": pd.api.types.is_integer_dtype, "float": pd.api.types.is_float_dtype,
              "bool": pd.api.types.is_bool_dtype, "object": lambda t: t == object}[kind](df[col].dtype)
        if not ok:
            fail(f"{rid}: column {col} dtype {df[col].dtype} is not {kind}; never coerce")
    for col in R31.LOAD_BEARING:
        if col not in df.columns or df[col].isna().any():
            fail(f"{rid}: load-bearing column {col} absent or null")
    if len(df) != 150:
        fail(f"{rid}: n_images {len(df)} != 150")
    m2, fg, bg = R31.two_class_tol_k2(df)
    maxdev = 0.0
    for name, got, stored in (("fg", fg, df.iou_fg_tol_pct), ("bg", bg, df.iou_bg_tol_pct),
                              ("2class", m2, df.miou_tol_pct)):
        mx = float(np.abs(got - stored.to_numpy()).max())
        maxdev = max(maxdev, mx)
        if mx > STORED_TOL:
            fail(f"{rid}: recomputed {name} IoU disagrees with the stored column by {mx:.3e}")
    return df, meta, maxdev


def load_vec(root, rid, idx, with_k5):
    """Read one cell through analysis/31's validator (CONTRACT section 3 columns
    and dtypes, run.json sidecar, n_images 150, K=2 recomputed and asserted
    against the stored columns). Returns a per-image Series keyed by image_id."""
    if rid not in set(idx.run_id):
        fail(f"{rid}: not in the index built for {os.path.relpath(root, BASE)}")
    require_markers(root, rid)
    df, meta, maxdev = read_cell(root, rid)
    if df.image_id.duplicated().any():
        fail(f"{rid}: duplicate image_id")
    out = dict(k2=pd.Series(two_class_k2(df), index=df.image_id.values).sort_index(),
               meta=meta, empty_fg=int(df.empty_fg_valid.sum()), stored_maxdev=maxdev)
    if with_k5:
        if "tp_k5" not in df.columns:
            fail(f"{rid}: no *_k5 columns, cannot run the K=5 regression")
        out["k5"] = pd.Series(two_class_k5(df), index=df.image_id.values).sort_index()
    return out


MR_RID = re.compile(r"^[a-z0-9]+-[a-z_]+-loco-r450-(chicago|miami|phoenix)-mr-s0$")


def load_delta_sha_list():
    """Delta's reported perimage.parquet sha256 per mid-res LOCO run_id.

    Accepted forms, nothing else: a sha256sum text file whose paths contain the
    run_id directory; a JSON object {run_id: sha256}; or any JSON whose nested
    dicts carry a `run_id` and exactly one of SHA_KEYS. A run_id reported twice
    with different values is fatal."""
    path = next((p for p in MIDRES_SHA_CANDIDATES if os.path.isfile(p)), None)
    if path is None:
        fail("no Delta sha256 list for the mid-res dumps on the analysis host; looked at "
             + ", ".join(os.path.relpath(p, BASE) for p in MIDRES_SHA_CANDIDATES))
    out, aux = {}, {}

    def put(rid, h):
        if not MR_RID.match(rid):
            return
        if not re.fullmatch(r"[0-9a-f]{64}", str(h)):
            fail(f"{path}: {rid} sha256 {h!r} is not a sha256")
        if rid in out and out[rid] != h:
            fail(f"{path}: {rid} reported with two different sha256 values")
        out[rid] = h

    if path.endswith(".json"):
        J = json.load(open(path))

        def walk(o):
            if isinstance(o, dict):
                if all(isinstance(k, str) and MR_RID.match(k) for k in o) and o and \
                        all(isinstance(v, str) for v in o.values()):
                    for k, v in o.items():
                        put(k, v)
                    return
                if "run_id" in o:
                    ks = [k for k in SHA_KEYS if o.get(k)]
                    if len(ks) > 1 and len({o[k] for k in ks}) > 1:
                        fail(f"{path}: {o['run_id']} carries conflicting sha keys {ks}")
                    if ks:
                        put(o["run_id"], o[ks[0]])
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(J)
    else:
        for line in open(path):
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) == 3 and parts[1].isdigit():       # sha256  size  path
                h, size, fp = parts[0], int(parts[1]), parts[2]
            elif len(parts) == 2:                            # sha256sum
                h, size, fp = parts[0], None, parts[1].lstrip("*")
            else:
                fail(f"{path}: unparseable line {line.rstrip()!r}")
            comps = fp.split("/")
            if len(comps) < 2 or not MR_RID.match(comps[-2]):
                continue
            rid, fname = comps[-2], comps[-1]
            if fname == "perimage.parquet":
                put(rid, h)
                aux.setdefault(rid, {})["perimage_size"] = size
            elif fname == "run.json":
                if not re.fullmatch(r"[0-9a-f]{64}", h):
                    fail(f"{path}: {rid}/run.json sha256 {h!r} is not a sha256")
                if aux.get(rid, {}).get("run_json_sha256") not in (None, h):
                    fail(f"{path}: {rid}/run.json reported with two different sha256 values")
                aux.setdefault(rid, {}).update(run_json_sha256=h, run_json_size=size)
            else:
                fail(f"{path}: unexpected file {fname!r} for {rid}")
    if not out:
        fail(f"{path}: no mid-res LOCO run_id with a sha256 could be read; the "
             "format is not one this function accepts")
    return out, dict(path=os.path.relpath(path, BASE), sha256=sha256_file(path), n=len(out)), aux


def midres_gate(rid, F, sha_list, aux, delta_cells, idx):
    """One mid-res LOCO cell through the addendum gate. Returns the gate row and,
    if the dump is readable, its loaded vectors. Pass = perimage.parquet sha256
    equals Delta's manifest AND |K=5 recomputed - Appendix F| < MIDRES_GATE_TOL.
    run.json sha256, file sizes and Delta's own K=5 are recorded, not gated."""
    arch, method, _, _, city, res, _ = rid.split("-")[:7]
    pr = F[(F.arch == arch) & (F.method == method) & (F.city == city) & (F.res == res)]
    a = aux.get(rid, {})
    dc = delta_cells.get(rid, {})
    row = dict(run_id=rid, printed=float(pr.loco.iloc[0]), recomputed_k5=None, diff=None,
               sha_expected=sha_list.get(rid), sha_got=None, pass_sha=None, pass_k5=None,
               perimage_size_expected=a.get("perimage_size"), perimage_size_got=None,
               run_json_sha_expected=a.get("run_json_sha256"), run_json_sha_got=None,
               run_json_sha_match=None,
               delta_reported_k5=(dc.get("gate") or {}).get("k5"),
               delta_reported_k2=dc.get("png_miou_k2_twoclass"),
               recomputed_k2=None, k5_minus_delta_reported=None, k2_minus_delta_reported=None,
               appendix_f_gate_pass_in_run_json=None, passed=False, reason=None)
    d = os.path.join(PNG, rid)
    pq = os.path.join(d, "perimage.parquet")
    if rid not in set(idx.run_id):
        row["reason"] = "not in the index built for dumps/st_r450_published_png"
        return row, None
    if not os.path.isfile(pq):
        row["reason"] = "no dump on the analysis host"
        return row, None
    if not os.path.isfile(os.path.join(d, ".DONE")) or os.path.isfile(os.path.join(d, ".FAILED")):
        row["reason"] = "no .DONE marker, or a .FAILED marker present"
        return row, None
    row["sha_got"] = sha256_file(pq)
    row["pass_sha"] = row["sha_expected"] is not None and row["sha_got"] == row["sha_expected"]
    row["perimage_size_got"] = os.path.getsize(pq)
    rj = os.path.join(d, "run.json")
    row["run_json_sha_got"] = sha256_file(rj) if os.path.isfile(rj) else None
    row["run_json_sha_match"] = (row["run_json_sha_expected"] is not None
                                 and row["run_json_sha_got"] == row["run_json_sha_expected"])
    df, meta, maxdev = read_cell(PNG, rid)       # schema problems stay fatal
    row["appendix_f_gate_pass_in_run_json"] = (meta.get("appendix_f_gate") or {}).get("pass")
    k5 = pd.Series(two_class_k5(df), index=df.image_id.values).sort_index()
    row["recomputed_k5"] = float(k5.mean())
    row["diff"] = row["recomputed_k5"] - row["printed"]
    row["pass_k5"] = abs(row["diff"]) < MIDRES_GATE_TOL
    row["recomputed_k2"] = float(two_class_k2(df).mean())
    if row["delta_reported_k5"] is not None:
        row["k5_minus_delta_reported"] = row["recomputed_k5"] - row["delta_reported_k5"]
    if row["delta_reported_k2"] is not None:
        row["k2_minus_delta_reported"] = row["recomputed_k2"] - row["delta_reported_k2"]
    row["passed"] = bool(row["pass_sha"] and row["pass_k5"])
    if not row["passed"]:
        row["reason"] = "; ".join(
            ([] if row["pass_sha"] else
             ["sha256 not in Delta's list" if row["sha_expected"] is None else
              "sha256 differs from Delta's list"])
            + ([] if row["pass_k5"] else [f"|K=5 diff| {abs(row['diff']):.4f} >= {MIDRES_GATE_TOL}"]))
    if df.image_id.duplicated().any():
        fail(f"{rid}: duplicate image_id")
    vec = dict(k2=pd.Series(two_class_k2(df), index=df.image_id.values).sort_index(), k5=k5,
               meta=meta, empty_fg=int(df.empty_fg_valid.sum()), stored_maxdev=maxdev)
    return row, vec


def build_cells(idx_png, idx_rec, F):
    """All main-block cells. Returns (cells, upper, excluded, gate_rows, sha_info)."""
    gate_rows, sha_list, sha_info, aux, delta_cells = [], None, None, {}, {}
    if PHASE == "B":
        sha_list, sha_info, aux = load_delta_sha_list()
        expected = {png_rid(a, m, c, "mr") for a in ARCHS_APPF for m in METHODS for c in CITIES}
        listed = {l.strip() for l in open(MIDRES_LIST) if l.strip()}
        DJ = json.load(open(DELTA_RESULTS))
        delta_cells = {c["run_id"]: c for c in DJ["cells"]}
        in_idx = set(idx_png.run_id[idx_png.run_id.str.match(MR_RID.pattern)])
        sha_info.update(
            midres_list=dict(path=os.path.relpath(MIDRES_LIST, BASE), sha256=sha256_file(MIDRES_LIST),
                             n=len(listed), equals_expected_63=listed == expected),
            delta_results=dict(path=os.path.relpath(DELTA_RESULTS, BASE),
                               sha256=sha256_file(DELTA_RESULTS), n_cells=len(delta_cells),
                               slurm_job_id=DJ.get("slurm_job_id"), n_gated=DJ.get("n_gated"),
                               gate_tol=DJ.get("gate_tol"), scorer=DJ.get("scorer"),
                               metric_module_sha=DJ.get("metric_module_sha"),
                               equals_expected_63=set(delta_cells) == expected),
            manifest_perimage_ids_equal_expected_63=set(sha_list) == expected,
            manifest_run_json_ids_equal_expected_63={r for r, v in aux.items()
                                                     if v.get("run_json_sha256")} == expected,
            index_midres_loco_ids_equal_expected_63=in_idx == expected)
        for k in ("midres_list", "delta_results"):
            if not sha_info[k]["equals_expected_63"]:
                fail(f"{sha_info[k]['path']}: run_id set differs from the 63 expected mid-res LOCO cells")
        if not sha_info["manifest_perimage_ids_equal_expected_63"]:
            fail("Delta's manifest does not list exactly the 63 expected perimage.parquet files")
    upper = {}
    for arch in ARCHS_APPF:
        for city in CITIES:
            for res in RESES:
                rid = png_rid(arch, "upper", city, res)
                v = load_vec(PNG, rid, idx_png, with_k5=True)
                upper[(arch, city, res)] = dict(run_id=rid, tree="st_r450_published_png", **v)
    cells, excluded, n450_used = {}, [], set()
    for arch in ARCHS_APPF:
        for method in METHODS:
            for city in CITIES:
                for res in RESES:
                    key = (arch, method, city, res)
                    pub_rid = png_rid(arch, method, city, res)
                    pub = None
                    if res == "mr" and PHASE == "A":
                        excluded.append(dict(cell="/".join(key), reason="PHASE A: mid-res LOCO not run"))
                        continue
                    if res == "mr":
                        grow, vec = midres_gate(pub_rid, F, sha_list, aux, delta_cells, idx_png)
                        gate_rows.append(grow)
                        if not grow["passed"]:
                            excluded.append(dict(cell="/".join(key), run_id=pub_rid,
                                                 reason="mid-res gate: " + grow["reason"],
                                                 printed=grow["printed"],
                                                 recomputed_k5=grow["recomputed_k5"],
                                                 diff=grow["diff"]))
                            continue
                        cells[key] = dict(run_id=pub_rid, tree="st_r450_published_png",
                                          source="published cell, PNG-scored from saved masks "
                                                 "(mid-res, published training budget)", **vec)
                        continue
                    pub = load_vec(PNG, pub_rid, idx_png, with_k5=True)
                    rec = dict(run_id=pub_rid, tree="st_r450_published_png",
                               source="published cell, PNG-scored", **pub)
                    if res == "hr":
                        found, refused = R31.resolve_n450(method, arch, city)
                        if found:
                            f0 = found[0]
                            idx_here = idx_rec if f0["tree"] == "st_r450_published_recompute" else None
                            if idx_here is None:
                                fail(f"{f0['run_id']}: n450 cell outside the recompute tree")
                            v = load_vec(f0["root"], f0["run_id"], idx_here, with_k5=False)
                            rec = dict(run_id=f0["run_id"], tree=f0["tree"],
                                       source=("matched-450 retrain, inference batch "
                                               f"{f0['inference_batch_size']}"),
                                       n450_refused=[r["run_id"] for r in refused],
                                       published_run_id=pub_rid, published=pub, **v)
                            n450_used.add((arch, method, city))
                    cells[key] = rec
    if n450_used != EXPECTED_N450:
        fail(f"n450 substitutions {sorted(n450_used ^ EXPECTED_N450)} differ from "
             "the 24 expected Table-2 cells")
    return cells, upper, excluded, gate_rows, sha_info


def cell_stats(loco, up, metric):
    a, u = loco[metric], up[metric]
    if list(a.index) != list(u.index):
        fail("LOCO and upper cells do not share the 150 test image_ids")
    d = (a - u).to_numpy()
    bt = boot_published(d)
    return dict(loco=float(a.mean()), upper=float(u.mean()), **bt,
                sig=bool(bt["p"] < ALPHA),
                sig_degrade=bool(bt["p"] < ALPHA and bt["mean"] < 0))


# ----------------------------------------------------------------------------
# K=5 regression and unit confirmation
# ----------------------------------------------------------------------------

def k5_regression(cells, upper, F):
    rows = []
    for (arch, city, res), u in upper.items():
        pr = F[(F.arch == arch) & (F.city == city) & (F.res == res)].up.unique()
        if len(pr) != 1:
            fail(f"Appendix F prints {len(pr)} upper values for {arch}/{city}/{res}")
        rows.append(dict(cell=u["run_id"], got=float(u["k5"].mean()), printed=float(pr[0])))
    for key, c in cells.items():
        if key[3] == "mr":
            continue    # mid-res LOCO is gated by midres_gate at |diff| < 0.01, not here
        pub = c.get("published", c)
        arch, method, city, res = key
        pr = F[(F.arch == arch) & (F.method == method) & (F.city == city) & (F.res == res)]
        rows.append(dict(cell=png_rid(*key), got=float(pub["k5"].mean()),
                         printed=float(pr.loco.iloc[0])))
    R = pd.DataFrame(rows)
    R["diff"] = R.got.round(2) - R.printed
    R["absdiff_raw"] = (R.got - R.printed).abs()
    bad = R[R["diff"].abs() > 1e-9]
    return R, bad


def k5_unit_check(cells, upper, F):
    """Printed Appendix F CI and dagger vs the published-protocol bootstrap at
    K=5 (paired, image unit) and vs the unpaired alternative."""
    out = []
    for key, c in cells.items():
        arch, method, city, res = key
        pub = c.get("published", c)
        up = upper[(arch, city, res)]
        a, u = pub["k5"], up["k5"]
        if list(a.index) != list(u.index):
            fail(f"{key}: K=5 pairing failed")
        bt = boot_published((a - u).to_numpy())
        un = boot_unpaired(a.to_numpy(), u.to_numpy())
        pr = F[(F.arch == arch) & (F.method == method) & (F.city == city) & (F.res == res)].iloc[0]
        out.append(dict(cell="/".join(key), printed_lo=pr.ci_lo, printed_hi=pr.ci_hi,
                        printed_p=pr.p, printed_sig=bool(pr.sig),
                        paired_lo=bt["ci95"][0], paired_hi=bt["ci95"][1], paired_p=bt["p"],
                        paired_sig=bt["p"] < ALPHA,
                        unpaired_lo=un["ci95"][0], unpaired_hi=un["ci95"][1],
                        unpaired_p=un["p"], unpaired_sig=un["p"] < ALPHA))
    U = pd.DataFrame(out)
    U["printed_w"] = U.printed_hi - U.printed_lo
    U["paired_w"] = U.paired_hi - U.paired_lo
    U["unpaired_w"] = U.unpaired_hi - U.unpaired_lo
    summ = dict(
        n_cells=int(len(U)),
        paired=dict(max_abs_ci_end_diff=float(max((U.paired_lo - U.printed_lo).abs().max(),
                                                  (U.paired_hi - U.printed_hi).abs().max())),
                    mean_abs_width_diff=float((U.paired_w - U.printed_w).abs().mean()),
                    dagger_agree=int((U.paired_sig == U.printed_sig).sum()),
                    max_abs_p_diff=float((U.paired_p - U.printed_p).abs().max())),
        unpaired=dict(max_abs_ci_end_diff=float(max((U.unpaired_lo - U.printed_lo).abs().max(),
                                                    (U.unpaired_hi - U.printed_hi).abs().max())),
                      mean_abs_width_diff=float((U.unpaired_w - U.printed_w).abs().mean()),
                      dagger_agree=int((U.unpaired_sig == U.printed_sig).sum())))
    return U, summ


def stats_csv_check(F):
    """statistical_results.csv (the file behind Appendix F and Figure 1):
    confirm its loco/tolerant/mIOU rows equal the printed Appendix F."""
    S = pd.read_csv(STATS_CSV)
    S = S[(S.test_type == "loco") & (S.evaluation == "tolerant") & (S.metric == "mIOU")]
    S = S.assign(res=S.resolution.map({"highres": "hr", "midres": "mr"}),
                 arch=S.model, city=S.city)
    M = F.merge(S, on=["arch", "method", "city", "res"], how="left", validate="1:1")
    if M.delta_mean.isna().any():
        fail("statistical_results.csv lacks some Appendix F cells")
    chk = dict(
        n=int(len(M)),
        max_abs_upper=float((M.upper_bound_mean.round(2) - M.up).abs().max()),
        max_abs_loco=float((M.method_mean.round(2) - M.loco).abs().max()),
        max_abs_ci=float(max((M.ci_lower.round(2) - M.ci_lo).abs().max(),
                             (M.ci_upper.round(2) - M.ci_hi).abs().max())),
        dagger_agree=int(((M.p_value < ALPHA) == M.sig).sum()),
        significant_col_agree=int((M.significant.astype(bool) == M.sig).sum()))
    return chk, M


# ----------------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------------

def aggregate(stats, upper_stats, cell_keys_by_arch_method):
    """Table 2 definitions. stats[(arch, method, city, res)] -> cell stats."""
    T = {}
    for arch in ARCHS_T2:
        ups = [upper_stats[k] for k in upper_stats if k[0] == arch]
        T[("upper", arch)] = dict(value=float(np.mean(ups)), n=len(ups))
    for method in METHODS:
        for arch in ARCHS_T2:
            ks = cell_keys_by_arch_method(arch, method)
            T[(method, arch)] = dict(
                loco=float(np.mean([stats[k]["loco"] for k in ks])),
                delta=float(np.mean([stats[k]["mean"] for k in ks])),
                sig=int(sum(stats[k]["sig"] for k in ks)),
                sig_degrade=int(sum(stats[k]["sig_degrade"] for k in ks)),
                n=len(ks))
        ks = [k for a in ARCHS_T2 for k in cell_keys_by_arch_method(a, method)]
        T[(method, "pooled")] = dict(
            delta=float(np.mean([stats[k]["mean"] for k in ks])),
            sig=int(sum(stats[k]["sig"] for k in ks)),
            sig_degrade=int(sum(stats[k]["sig_degrade"] for k in ks)), n=len(ks))
    for arch in ARCHS_T2:
        T[("avg_delta", arch)] = float(np.mean([T[(m, arch)]["delta"] for m in METHODS]))
        T[("sig_all", arch)] = int(sum(T[(m, arch)]["sig"] for m in METHODS))
        T[("n_all", arch)] = int(sum(T[(m, arch)]["n"] for m in METHODS))
    return T


def extra_row(method, arch_list, R, ups):
    """SAR/BSM/CrossEarth rows from analysis/31's resolve_row (high-res)."""
    per = {}
    for arch in arch_list:
        cs = []
        for city in CITIES:
            r = R[(method, arch, city)]
            u = ups[(arch, city)]
            if r.get("per_image") is None:
                cs.append(dict(city=city, value=None, blocker=r.get("blocker")))
                continue
            a = pd.Series(r["per_image"], index=r["image_id"]).sort_index()
            uu = pd.Series(u["per_image"], index=u["image_id"]).sort_index()
            if list(a.index) != list(uu.index):
                fail(f"{method}/{arch}/{city}: rows do not share the 150 image_ids")
            bt = boot_published((a - uu).to_numpy())
            cs.append(dict(city=city, run_id=r.get("run_id"), tree=r.get("tree"),
                           source_kind=r.get("source_kind"), loco=float(a.mean()),
                           upper=float(uu.mean()), upper_run_id=u.get("run_id"),
                           **bt, sig=bool(bt["p"] < ALPHA),
                           sig_degrade=bool(bt["p"] < ALPHA and bt["mean"] < 0)))
        per[arch] = cs
    return per


def summarise_extra(per):
    out = {}
    for arch, cs in per.items():
        ok = [c for c in cs if c.get("loco") is not None]
        out[arch] = dict(loco=float(np.mean([c["loco"] for c in ok])) if ok else None,
                         delta=float(np.mean([c["mean"] for c in ok])) if ok else None,
                         sig=int(sum(c["sig"] for c in ok)), n=len(ok),
                         missing=[c["city"] for c in cs if c.get("loco") is None])
    allc = [c for cs in per.values() for c in cs if c.get("loco") is not None]
    out["pooled"] = dict(delta=float(np.mean([c["mean"] for c in allc])),
                         sig=int(sum(c["sig"] for c in allc)), n=len(allc))
    return out


def sec34(stats, T):
    keys = list(stats)
    n = len(keys)
    bonf = ALPHA / n
    worst = min(keys, key=lambda k: stats[k]["mean"])
    best = max(keys, key=lambda k: stats[k]["mean"])
    am = {(m, a): T[(m, a)]["delta"] for m in METHODS for a in ARCHS_T2}
    lo, hi = min(am, key=am.get), max(am, key=am.get)
    pooled = {m: T[(m, "pooled")]["delta"] for m in METHODS}
    every = {m: dict(all_delta_negative=all(stats[k]["mean"] < 0 for k in keys if k[1] == m),
                     all_sig_degrade=all(stats[k]["sig_degrade"] for k in keys if k[1] == m))
             for m in METHODS}
    return dict(
        n_cells=n,
        n_sig_p05=int(sum(stats[k]["sig"] for k in keys)),
        n_sig_degrade_p05=int(sum(stats[k]["sig_degrade"] for k in keys)),
        bonferroni_threshold=bonf,
        n_sig_bonferroni=int(sum(stats[k]["p"] < bonf for k in keys)),
        n_sig_degrade_bonferroni=int(sum(stats[k]["p"] < bonf and stats[k]["mean"] < 0 for k in keys)),
        mean_delta=float(np.mean([stats[k]["mean"] for k in keys])),
        worst_cell=dict(cell="/".join(worst), delta=stats[worst]["mean"]),
        best_cell=dict(cell="/".join(best), delta=stats[best]["mean"]),
        arch_method_delta_min=dict(method=lo[0], arch=lo[1], delta=am[lo]),
        arch_method_delta_max=dict(method=hi[0], arch=hi[1], delta=am[hi]),
        n_adapt_more_negative_than_vanilla=int(sum(pooled[m] < pooled["vanilla"] for m in ADAPT)),
        adapt_more_negative_than_vanilla=[m for m in ADAPT if pooled[m] < pooled["vanilla"]],
        methods_every_cell=every)


# ----------------------------------------------------------------------------
# console tables
# ----------------------------------------------------------------------------

def print_main(T, label, ncell):
    print(f"\n=== {label} ===")
    print(f"{'Method':10s} | {'LOCO D':>7s} {'M':>6s} {'O':>6s} | {'Δ D':>6s} {'M':>6s} {'O':>6s} | "
          f"Sig/{ncell} D M O | pooled Δ  Sig/{3 * ncell}")
    print(f"{'Upper':10s} | " + " ".join(f"{T[('upper', a)]['value']:6.2f}" for a in ARCHS_T2))
    for m in METHODS:
        r = [T[(m, a)] for a in ARCHS_T2]
        p = T[(m, "pooled")]
        print(f"{MLABEL[m]:10s} | " + " ".join(f"{x['loco']:6.2f}" for x in r) + " | "
              + " ".join(f"{x['delta']:+6.2f}" for x in r) + " | "
              + " ".join(f"{x['sig']}" for x in r) + f"     | {p['delta']:+7.2f}  {p['sig']:2d}")
    print(f"{'Avg Δ':10s} | {'':20s} | " + " ".join(f"{T[('avg_delta', a)]:+6.2f}" for a in ARCHS_T2))
    print(f"{'#Sig':10s} | {'':20s} | {'':20s} | " + " ".join(str(T[('sig_all', a)]) for a in ARCHS_T2)
          + f"  (of {T[('n_all', ARCHS_T2[0])]} per arch)")


def print_extra(name, s, per):
    cols = [a for a in ARCHS_T2 if a in s] + [a for a in s if a not in ARCHS_T2 and a != "pooled"]
    print(f"{name:10s} | " + " ".join(f"{ARCH_LABEL[a][:2]} loco {s[a]['loco']:6.2f} Δ {s[a]['delta']:+6.2f} "
                                       f"sig {s[a]['sig']}/{s[a]['n']}" if s[a]['loco'] is not None
                                       else f"{ARCH_LABEL[a][:2]} n/a" for a in cols)
          + f" | pooled Δ {s['pooled']['delta']:+6.2f} sig {s['pooled']['sig']}/{s['pooled']['n']}")
    for a in cols:
        for c in per[a]:
            if c.get("loco") is None:
                print(f"    {a}/{c['city']}: NO VALUE ({c.get('blocker')})")



# ----------------------------------------------------------------------------
# Step 6: K=5 inventory of printed numbers that come from Table 2 / Appendix F
# ----------------------------------------------------------------------------
# Curated from a scan of origin/main sections/*.tex, the abstract, every
# \caption, and the figure data files; coincidental numeric matches (p
# thresholds, nats, SIB-ablation deltas) were dropped by reading each hit.
# Line numbers are resolved at run time from each entry's anchor regex.

def _fig1(F_or_cells, arch, kind):
    """Figure 1 chart: median over the 3 HIGH-RES Vanilla cells (what the
    JPEG shows; the notebook docstring and caption say 6 cells)."""
    return F_or_cells(arch, kind)


INVENTORY = [
    ("neurips_2026.tex", r"all adaptation methods degrade under geographic transfer",
     "all adaptation methods degrade", "every method's pooled Delta < 0 (qualitative)", "all_pooled_negative"),
    ("sections/benchmark.tex", r"\$118\$ of \$126\$", "118", "cells significant at p<0.05, of 126", "n_sig"),
    ("sections/benchmark.tex", r"robust to Bonferroni", "robust to Bonferroni (=118)",
     "cells significant after Bonferroni over 126", "n_bonf"),
    ("sections/benchmark.tex", r"-7\.21", "-7.21", "mean Delta over 126 cells", "mean126"),
    ("sections/benchmark.tex", r"worst single cell of \$-24\.08", "-24.08", "worst single-cell Delta", "worst"),
    ("sections/benchmark.tex", r"-1\.20\$ on DINOv3", "-1.20 (DINOv3)", "max of 21 arch x method 6-cell mean Deltas", "am_max"),
    ("sections/benchmark.tex", r"-15\.30\$ on OGLANet", "-15.30 (OGLANet)", "min of 21 arch x method 6-cell mean Deltas", "am_min"),
    ("sections/benchmark.tex", r"Five of six methods", "5 of 6", "adaptation methods with pooled Delta below Vanilla's", "n_below_vanilla"),
    ("sections/benchmark.tex", r"ISW and FADA degrade every cell", "ISW, FADA",
     "methods significantly degrading all 18 cells", "every_cell"),
    ("sections/intro.tex", r"averages 75", "75%", "mean of the 3 per-arch Upper values", "upper_mean"),
    ("sections/intro.tex", r"dropping to 71", "71%", "mean of the 3 per-arch Vanilla LOCO values", "vanilla_loco_mean"),
    ("sections/intro.tex", r"-\s*1\.7\$ \(DINOv3\)", "-1.7", "Vanilla Delta, DINOv3", "T2:vanilla:dinov3"),
    ("sections/intro.tex", r"-\s*15\.3\$ \(OGLANet\)", "-15.3", "ISW Delta, OGLANet (sentence calls it Vanilla; E-3)", "T2:isw:oglanet"),
    ("sections/evaluation.tex", r"costs OGLANet \$-10\.1", "-10.1", "MRFP+ Delta, OGLANet", "T2:mrfp_plus:oglanet"),
    ("sections/evaluation.tex", r"costs DINOv3 \$-7\.9", "-7.9", "ISW Delta, DINOv3", "T2:isw:dinov3"),
    ("appendix/app_loco_extended.tex", r"differs by \$0\.20\$", "0.20", "|mean Delta HR - mean Delta MR|", "res_gap"),
    ("appendix/app_loco_extended.tex", r"five of the eight", "5 of 8",
     "DINOv3 mid-res cells among the non-significant cells", "dino_mr_nonsig"),
    ("appendix/app_loco_extended.tex", r"best \(\$\+5\.15\$\)", "+5.15", "best single-cell Delta", "best"),
    ("appendix/app_loco_extended.tex", r"^\(\$-24\.08\$\) cells", "-24.08", "worst single-cell Delta", "worst"),
    # added in PHASE B from a scan of appendix/*.tex for sentences citing Table 2
    ("appendix/app_eval.tex", r"costs DINOv3 \$-7\.9\$ mIoU because", "-7.9",
     "MRFP+ Delta, DINOv3", "T2:mrfp_plus:dinov3"),
    ("appendix/app_eval.tex", r"DINOv3 mean \$\\Delta = -7\.9\$ mIoU under ISW", "-7.9",
     "ISW Delta, DINOv3", "T2:isw:dinov3"),
    ("appendix/app_splits.tex", r"^Phoenix, the city with the largest LOCO degradation", "Phoenix",
     "held-out city with the most negative mean Delta over its 42 cells", "city_most_negative"),
    ("appendix/app_splits.tex", r"Phoenix's lower Upper Bound", "Phoenix (lowest)",
     "held-out city with the lowest mean Upper over its 6 upper cells", "city_lowest_upper"),
    ("appendix/app_splits.tex", r"Chicago already posts the highest Upper Bound", "Chicago (highest)",
     "held-out city with the highest mean Upper over its 6 upper cells", "city_highest_upper"),
    ("appendix/app_metric.tex", r"\(\$K = 2\$\) numbers side-by-side in Appendix", "K=0 and K=2 side by side",
     "band width(s) printed in Appendix F", "appf_metric"),
    ("figures/fig_teaser.tex (figure1 final.001.jpeg)", None, "Delta -11.1 / -2.7 / -5.9 (D/M/O)",
     "Figure 1 chart: median Upper and median Vanilla LOCO over the 3 high-res cells per arch, Delta = difference",
     "fig1"),
]


def inventory_values(stats_like, upper_like, F, csvM, full):
    """One function, two inputs: called with the printed K=5 Appendix F cells
    and with this job's K=2 cells. full=False means mid-res LOCO is absent, so
    6-cell and 126-cell quantities are not computed."""
    d = stats_like
    v = {}
    v["fig1"] = {}
    for a in ARCHS_T2:
        u = np.median([upper_like[(a, c, "hr")] for c in CITIES])
        l = np.median([d[(a, "vanilla", c, "hr")]["loco"] for c in CITIES])
        v["fig1"][a] = dict(upper_median=round(float(u), 2), loco_median=round(float(l), 2),
                           delta=round(float(l - u), 2))
    if not full:
        return v
    keys = list(d)
    n = len(keys)
    am = {(m, a): np.mean([d[k]["mean"] for k in keys if k[0] == a and k[1] == m])
          for m in METHODS for a in ARCHS_T2}
    pooled = {m: np.mean([d[k]["mean"] for k in keys if k[1] == m]) for m in METHODS}
    w = min(keys, key=lambda k: d[k]["mean"])
    b = max(keys, key=lambda k: d[k]["mean"])
    hr = np.mean([d[k]["mean"] for k in keys if k[3] == "hr"])
    mr = np.mean([d[k]["mean"] for k in keys if k[3] == "mr"])
    nonsig = [k for k in keys if not d[k]["sig"]]
    v.update(
        all_pooled_negative=all(x < 0 for x in pooled.values()),
        n_sig=int(sum(d[k]["sig"] for k in keys)),
        n_bonf=int(sum(d[k]["p"] < ALPHA / n for k in keys)),
        mean126=round(float(np.mean([d[k]["mean"] for k in keys])), 4),
        worst=dict(cell="/".join(w), delta=round(float(d[w]["mean"]), 2)),
        best=dict(cell="/".join(b), delta=round(float(d[b]["mean"]), 2)),
        am_max=dict(zip(("method", "arch"), max(am, key=am.get)), delta=round(float(max(am.values())), 2)),
        am_min=dict(zip(("method", "arch"), min(am, key=am.get)), delta=round(float(min(am.values())), 2)),
        n_below_vanilla=int(sum(pooled[m] < pooled["vanilla"] for m in ADAPT)),
        every_cell=[m for m in METHODS if all(d[k]["sig"] and d[k]["mean"] < 0 for k in keys if k[1] == m)],
        upper_mean=round(float(np.mean([np.mean([upper_like[(a, c, r)] for c in CITIES for r in RESES])
                                        for a in ARCHS_T2])), 2),
        vanilla_loco_mean=round(float(np.mean([am_loco for am_loco in
                                               [np.mean([d[k]["loco"] for k in keys
                                                         if k[0] == a and k[1] == "vanilla"])
                                                for a in ARCHS_T2]])), 2),
        res_gap=round(abs(hr - mr), 2),
        dino_mr_nonsig=f"{sum(1 for k in nonsig if k[0] == 'dinov3' and k[3] == 'mr')} of {len(nonsig)}")
    cd = {c: float(np.mean([d[k]["mean"] for k in keys if k[2] == c])) for c in CITIES}
    cu = {c: float(np.mean([upper_like[(a, c, r)] for a in ARCHS_T2 for r in RESES])) for c in CITIES}
    v.update(city_most_negative=dict(city=min(cd, key=cd.get), mean_delta_by_city={c: round(x, 2) for c, x in cd.items()}),
             city_lowest_upper=dict(city=min(cu, key=cu.get), mean_upper_by_city={c: round(x, 2) for c, x in cu.items()}),
             city_highest_upper=dict(city=max(cu, key=cu.get), mean_upper_by_city={c: round(x, 2) for c, x in cu.items()}),
             n_cells=n)
    for m in METHODS:
        for a in ARCHS_T2:
            v[f"T2:{m}:{a}"] = round(float(am[(m, a)]), 2)
    if csvM is not None:   # K=5 side: the unrounded p-values behind the daggers
        v["n_bonf"] = int((csvM.p_value < ALPHA / n).sum())
        v["mean126"] = round(float(csvM.delta_mean.mean()), 4)
    return v


def print_inventory(v5, v2):
    print("\n=== STEP 6 INVENTORY (origin/main): printed | K=5 recomputed from Appendix F | K=2 this job ===")
    rows = []
    for f, anchor, printed, what, key in INVENTORY:
        ln = None
        if anchor:
            src = git_show(f.split(" ")[0])
            hits = [i for i, l in enumerate(src.splitlines(), 1) if re.search(anchor, l)]
            if len(hits) != 1:
                fail(f"inventory anchor {anchor!r} hits {len(hits)} lines of {f}")
            ln = hits[0]
        k5, k2 = v5.get(key, "n/a"), v2.get(key, "PHASE B")
        rows.append(dict(file=f, line=ln, printed=printed, quantity=what, key=key, k5=k5, k2=k2))
        print(f"  {f}:{ln}  printed {printed!r}  [{what}]\n      K=5 {k5}\n      K=2 {k2}")
    return rows

# ----------------------------------------------------------------------------
# PHASE B: LaTeX rows (no bold anywhere), format-checked against origin/main
# ----------------------------------------------------------------------------

CITY_LABEL = {"chicago": "Chicago", "miami": "Miami", "phoenix": "Phoenix"}
RES_LABEL_POOLED = {"hr": r"High-res ($0.3$\,m/px)", "mr": r"Mid-res  ($0.6$\,m/px)"}


def _sg(x, dp):
    return f"${x:+.{dp}f}$"


def _p_tex(p, sig):
    if p == 0:
        if not sig:
            fail("p = 0 on a non-significant cell")
        return r"$<\!.0002$"
    t = f"{p:.4f}"
    t = t[1:] if t.startswith("0") else t
    return f"${t}$" if sig else f"${t}^{{\\dagger}}$"


def appf_row(first, method, city, res, up, loco, delta, lo, hi, p, sig):
    m = f"{MLABEL[method]:<10s}" if first else " " * 10
    ci = f"$[{lo:+.2f},\\,{hi:+.2f}]$"
    return (f"{m}& {CITY_LABEL[city]:<7s} & {res.upper()} & {up:.2f} & {loco:.2f} & "
            f"{_sg(delta, 2):<8s} & {ci:<19s} & {_p_tex(p, sig)} \\\\")


def appf_excluded_row(first, method, city, res, up, why):
    m = f"{MLABEL[method]:<10s}" if first else " " * 10
    return (f"{m}& {CITY_LABEL[city]:<7s} & {res.upper()} & {up:.2f} & --- & --- & --- & --- \\\\"
            f"  % excluded: {why}")


def _norm(line):
    line = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


def appf_format_check(F, appf_tex):
    """Render the printed K=5 values through appf_row and compare with the
    origin/main lines, whitespace-normalised and with \\mathbf removed."""
    src = appf_tex.splitlines()
    bad = []
    for arch in ARCHS_APPF:
        G = F[F.arch == arch].sort_values("line")
        order = [(m, c, r) for m in METHODS for c in CITIES for r in RESES]
        if [(r.method, r.city, r.res) for r in G.itertuples()] != order:
            fail(f"Appendix F {arch}: printed row order differs from METHODS x CITIES x RESES")
        for i, r in enumerate(G.itertuples()):
            mine = appf_row(i % 6 == 0, r.method, r.city, r.res, r.up, r.loco, r.delta,
                            r.ci_lo, r.ci_hi, r.p, r.sig)
            if _norm(mine) != _norm(src[r.line - 1]):
                bad.append((r.line, src[r.line - 1], mine))
    return bad


def appf_tables_tex(stats, upper_k2, excluded_keys):
    out = {}
    for arch in ARCHS_APPF:
        L = []
        for mi, m in enumerate(METHODS):
            if mi:
                L.append(r"\cmidrule(l){1-8}")
            for i, (c, r) in enumerate([(c, r) for c in CITIES for r in RESES]):
                k = (arch, m, c, r)
                if k in stats:
                    s = stats[k]
                    L.append(appf_row(i == 0, m, c, r, s["upper"], s["loco"], s["mean"],
                                      s["ci95"][0], s["ci95"][1], s["p"], s["sig"]))
                else:
                    L.append(appf_excluded_row(i == 0, m, c, r, upper_k2[(arch, c, r)],
                                               excluded_keys.get(k, "absent")))
        out[arch] = L
    return out


def pooled_tables(stats):
    """Appendix F pooled summaries (Tables 33-35 in the origin/main build)."""
    def summ(keys):
        d = [stats[k]["mean"] for k in keys]
        wk = min(keys, key=lambda k: stats[k]["mean"])
        bk = max(keys, key=lambda k: stats[k]["mean"])
        return dict(mean=float(np.mean(d)), best=float(stats[bk]["mean"]), best_cell="/".join(bk),
                    worst=float(stats[wk]["mean"]), worst_cell="/".join(wk),
                    sig=int(sum(stats[k]["sig"] for k in keys)),
                    sig_degrade=int(sum(stats[k]["sig_degrade"] for k in keys)), n=len(keys))
    K = list(stats)
    return dict(per_city={c: summ([k for k in K if k[2] == c]) for c in CITIES},
                per_resolution={r: summ([k for k in K if k[3] == r]) for r in RESES},
                per_method={m: summ([k for k in K if k[1] == m]) for m in METHODS})


def pooled_tex(P):
    L = ["% Table: LOCO degradation by held-out city (tab:loco-per-city) -- rows",
         r"% columns: Held-out city & $\Delta_{\text{mean}}$ & $\Delta_{\text{best}}$ & $\Delta_{\text{worst}}$ & Sig./n"]
    for c in CITIES:
        s = P["per_city"][c]
        L.append(f"{CITY_LABEL[c]:<7s} & {_sg(s['mean'], 2)} & {_sg(s['best'], 2):<8s} & "
                 f"{_sg(s['worst'], 2)} & ${s['sig']}/{s['n']}$ \\\\")
    L += ["", "% Table: LOCO degradation by native resolution (tab:loco-per-resolution) -- rows",
          r"% columns: Resolution & $\Delta_{\text{mean}}$ & $\Delta_{\text{best}}$ & $\Delta_{\text{worst}}$ & Sig./n"]
    for r in RESES:
        s = P["per_resolution"][r]
        L.append(f"{RES_LABEL_POOLED[r]} & {_sg(s['mean'], 2)} & {_sg(s['best'], 2)} & "
                 f"{_sg(s['worst'], 2)} & ${s['sig']}/{s['n']}$ \\\\")
    L += ["", "% Table: method-level safety summary (tab:loco-method-summary) -- rows",
          r"% columns: Method & Sig./18 & $\Delta_{\text{mean}}$ & $\Delta_{\text{best}}$ & $\Delta_{\text{worst}}$"]
    for m in METHODS:
        s = P["per_method"][m]
        sg = f"${s['sig']}$" if s["n"] == 18 else f"${s['sig']}/{s['n']}$"
        L.append(f"{MLABEL[m]:<10s}& {sg} & {_sg(s['mean'], 2):<9s}& {_sg(s['best'], 2)} & "
                 f"{_sg(s['worst'], 2)} \\\\")
    return L


def _frac(k, n, full):
    return f"{k}" if n == full else f"{k}/{n}"


def t2_main_row(label, loco, delta, sig, n, pooled, psig, pn, full=6):
    return (f"{label:<10s}& " + " & ".join(f"{loco[a]:.1f}" for a in ARCHS_T2) + " & "
            + " & ".join(_sg(delta[a], 1) for a in ARCHS_T2) + " & "
            + " & ".join(_frac(sig[a], n[a], full) for a in ARCHS_T2)
            + f" & {_sg(pooled, 2)} & {_frac(psig, pn, 3 * full)} \\\\")


def t2_format_check(T2p, t2_tex):
    src = {_norm(l) for l in t2_tex.splitlines()}
    bad = []
    for m in METHODS:
        r = T2p[m]
        mine = t2_main_row(MLABEL[m], r["loco"], r["delta"], {a: int(r["sig6"][a]) for a in ARCHS_T2},
                           {a: 6 for a in ARCHS_T2}, r["pooled_delta"], int(r["sig18"]), 18)
        if _norm(mine) not in src:
            bad.append((m, mine))
    return bad


def t2_tex(T, extra, ce, ce_upper):
    L = [r"\textit{Upper bound} & " + " & ".join(f"$\\mathit{{{T[('upper', a)]['value']:.1f}}}$" for a in ARCHS_T2)
         + r" & --- & --- & ---  & --- & --- & ---  & --- & --- \\",
         r"\midrule"]
    for m in METHODS:
        p = T[(m, "pooled")]
        L.append(t2_main_row(MLABEL[m], {a: T[(m, a)]["loco"] for a in ARCHS_T2},
                             {a: T[(m, a)]["delta"] for a in ARCHS_T2},
                             {a: T[(m, a)]["sig"] for a in ARCHS_T2},
                             {a: T[(m, a)]["n"] for a in ARCHS_T2}, p["delta"], p["sig"], p["n"]))
    L += [r"\midrule",
          r"\textit{Avg.\ $\Delta$} & --- & --- & --- & "
          + " & ".join(_sg(T[("avg_delta", a)], 1) for a in ARCHS_T2)
          + r" & --- & --- & --- & --- & --- \\",
          r"\textit{\# Sig./$42$} & --- & --- & --- & --- & --- & ---  & "
          + " & ".join(_frac(T[("sig_all", a)], T[("n_all", a)], 42) for a in ARCHS_T2)
          + r" & --- & --- \\"]
    L += ["", r"% ---- SAR and BSM: HIGH-RESOLUTION ONLY (3 cells per architecture; Sig. as k/n) ----",
          r"\midrule"]
    for m in ("sar", "bsm"):
        s = extra[m]["summary"]
        cells = []
        for a in ARCHS_T2:
            cells.append(s[a])
        L.append(f"{MLABEL[m] + ' (HR)':<10s}& "
                 + " & ".join(f"{x['loco']:.1f}" for x in cells) + " & "
                 + " & ".join(_sg(x["delta"], 1) for x in cells) + " & "
                 + " & ".join(f"{x['sig']}/{x['n']}" for x in cells)
                 + f" & {_sg(s['pooled']['delta'], 2)} & {s['pooled']['sig']}/{s['pooled']['n']} \\\\")
    L += ["", r"% ---- CrossEarth: HIGH-RESOLUTION ONLY (3 cells per row) ----",
          r"\midrule",
          r"\multicolumn{12}{l}{\textit{CrossEarth (high-res only, 3 held-out cities)}} \\",
          r"\textit{Upper bound} & \multicolumn{3}{c|}{$\mathit{" + f"{ce_upper:.1f}" + r"}$}"
          r" & \multicolumn{3}{c|}{---} & \multicolumn{3}{c|}{---} & --- & --- \\"]
    for m in ("vanilla", "fda", "sib", "bsm"):
        s = ce[m]["summary"]["crossearth"]
        L.append(f"{MLABEL[m]:<10s}& \\multicolumn{{3}}{{c|}}{{{s['loco']:.1f}}} & "
                 f"\\multicolumn{{3}}{{c|}}{{{_sg(s['delta'], 1)}}} & "
                 f"\\multicolumn{{3}}{{c|}}{{{s['sig']}/{s['n']}}} & "
                 f"{_sg(s['delta'], 2)} & {s['sig']}/{s['n']} \\\\")
    return L


def write_new(path, text):
    """Never overwrite: every output of this job is new."""
    if os.path.exists(path):
        fail(f"{os.path.relpath(path, BASE)} exists; this job never overwrites")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return sha256_file(path)


def jclean(o):
    """JSON-safe copy: numpy scalars to python, pandas Series dropped."""
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, tuple) else "/".join(map(str, k)): jclean(v)
                for k, v in o.items() if not isinstance(v, (pd.Series, pd.DataFrame))}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    return o


def extreme(rows, label):
    """rows: list of (cell_label, delta, p, sig). Worst = most negative Delta."""
    rows = [r for r in rows if r[1] is not None]
    w = min(rows, key=lambda r: r[1])
    b = max(rows, key=lambda r: r[1])
    return dict(table=label, n_cells=len(rows),
                worst=dict(cell=w[0], delta=w[1], p=w[2], sig=w[3]),
                best=dict(cell=b[0], delta=b[1], p=b[2], sig=b[3]))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    print(f"PHASE {PHASE}  DRY_RUN {DRY_RUN}  B {B}  seed {BOOT_SEED}  paper {PAPER_REF} "
          f"= {paper_commit()[:8]}")
    F, appf_tex = parse_appf()
    T2p, t2src = parse_table2()
    print(f"Appendix F parsed: {len(F)} cells; Table 2 parsed: {sorted(T2p)}")

    idx_png, info_png = R31.build_index("st_r450_published_png", PNG)
    idx_rec, info_rec = R31.build_index("st_r450_published_recompute", RECOMPUTE)
    print(f"index st_r450_published_png: {info_png['n_rows']} rows, "
          f"{info_png['n_unindexable']} unindexable; recompute: {info_rec['n_rows']} rows, "
          f"{info_rec['n_unindexable']} unindexable")
    mr_loco = idx_png[(idx_png.run_id.str.contains("-loco-")) & (idx_png.run_id.str.contains("-mr-"))]
    print(f"mid-res LOCO cells present in the PNG tree: {len(mr_loco)}")
    if DRY_RUN:
        sel = idx_png[idx_png.run_id.str.contains("-hr-|-upper-", regex=True)]
        print(f"DRY RUN: would read {len(sel)} PNG cells + 24 n450 cells; "
              f"n_images values {sorted(sel.n_images.unique())}")
        if PHASE == "B":
            shal, shai, aux = load_delta_sha_list()
            print(f"DRY RUN: Delta manifest {shai['path']}: {shai['n']} perimage sha256, "
                  f"{sum(1 for v in aux.values() if v.get('run_json_sha256'))} run.json sha256; "
                  f"mid-res LOCO in index {len(mr_loco)}, n_images {sorted(mr_loco.n_images.unique())}; "
                  f"in manifest and index: {len(set(shal) & set(mr_loco.run_id))}")
            dfm = pd.read_parquet(os.path.join(PNG, mr_loco.run_id.iloc[0], "perimage.parquet"))
            print("schema of", mr_loco.run_id.iloc[0], f"({dfm.shape[1]} columns)")
            print(dfm.dtypes.to_string())
        df = pd.read_parquet(os.path.join(PNG, sel.run_id.iloc[0], "perimage.parquet"))
        print("schema of", sel.run_id.iloc[0])
        print(df.dtypes.to_string())
        return

    cells, upper, excluded, gate_rows, sha_info = build_cells(idx_png, idx_rec, F)
    print(f"cells: {len(cells)} LOCO, {len(upper)} upper; excluded {len(excluded)}")
    if PHASE == "B":
        print(f"\n=== MID-RES GATE (sha256 vs {sha_info['path']}; |K=5 - Appendix F| < {MIDRES_GATE_TOL}) ===")
        print(f"{'run_id':44s} {'printed':>8s} {'K=5':>9s} {'diff':>8s} sha  k5  pass")
        for g in gate_rows:
            rk = "" if g["recomputed_k5"] is None else f"{g['recomputed_k5']:9.4f}"
            df_ = "" if g["diff"] is None else f"{g['diff']:+8.4f}"
            print(f"{g['run_id']:44s} {g['printed']:8.2f} {rk:>9s} {df_:>8s} {str(g['pass_sha'])[0]:4s} "
                  f"{str(g['pass_k5'])[0]:3s} {'PASS' if g['passed'] else 'FAIL: ' + g['reason']}")
        print(f"gate: {sum(g['passed'] for g in gate_rows)} of {len(gate_rows)} pass")

    # ---- K=5 regression on every published PNG cell Appendix F prints -------
    R, bad = k5_regression(cells, upper, F)
    print(f"\nK=5 REGRESSION: {len(R)} published cells, max |round2(K5) - printed| = "
          f"{R['diff'].abs().max():.2f} (raw max {R.absdiff_raw.max():.4f}); failures {len(bad)}")
    if len(bad):
        print(bad.to_string())
        fail("K=5 regression broke: a published cell no longer reproduces Appendix F")

    # ---- resampling-unit confirmation at K=5 --------------------------------
    U, usumm = k5_unit_check(cells, upper, F)
    csv_chk, _ = stats_csv_check(F)
    print(f"K=5 BOOTSTRAP UNIT CHECK on {usumm['n_cells']} cells (printed CI / dagger vs recomputed):")
    print(f"  paired, image unit : max|CI end diff| {usumm['paired']['max_abs_ci_end_diff']:.3f}, "
          f"mean|width diff| {usumm['paired']['mean_abs_width_diff']:.3f}, daggers agree "
          f"{usumm['paired']['dagger_agree']}/{usumm['n_cells']}, max|p diff| {usumm['paired']['max_abs_p_diff']:.4f}")
    print(f"  unpaired (rejected): max|CI end diff| {usumm['unpaired']['max_abs_ci_end_diff']:.3f}, "
          f"mean|width diff| {usumm['unpaired']['mean_abs_width_diff']:.3f}, daggers agree "
          f"{usumm['unpaired']['dagger_agree']}/{usumm['n_cells']}")
    print(f"  statistical_results.csv vs printed Appendix F: {csv_chk}")
    dis = U[U.paired_sig != U.printed_sig]
    if len(dis):
        print("  dagger disagreements (paired):")
        print(dis[["cell", "printed_p", "paired_p", "printed_lo", "paired_lo",
                   "printed_hi", "paired_hi"]].to_string(index=False))

    # ---- K=2 cell statistics ------------------------------------------------
    stats = {k: dict(cell_stats(c, upper[(k[0], k[2], k[3])], "k2"),
                     run_id=c["run_id"], tree=c["tree"], source=c["source"],
                     upper_run_id=upper[(k[0], k[2], k[3])]["run_id"])
             for k, c in cells.items()}
    upper_k2 = {k: float(v["k2"].mean()) for k, v in upper.items()}
    res_in = ("hr",) if PHASE == "A" else RESES

    def keys_for(arch, method):
        return [(arch, method, c, r) for c in CITIES for r in res_in if (arch, method, c, r) in stats]

    up_sel = {k: v for k, v in upper_k2.items() if k[2] in res_in}
    T = aggregate(stats, up_sel, keys_for)
    tag = "HIGH-RES ONLY (3 cells per arch)" if PHASE == "A" else "FULL (6 cells per arch)"
    print_main(T, f"K=2 MAIN BLOCK, {tag}", 3 if PHASE == "A" else 6)

    # same definitions at K=5 over the same published cells, for the shift
    stats5 = {}
    for k, c in cells.items():
        pub = c.get("published", c)
        stats5[k] = dict(loco=float(pub["k5"].mean()),
                         mean=float(pub["k5"].mean() - upper[(k[0], k[2], k[3])]["k5"].mean()))
        pr = F[(F.arch == k[0]) & (F.method == k[1]) & (F.city == k[2]) & (F.res == k[3])].iloc[0]
        stats5[k].update(sig=bool(pr.sig), sig_degrade=bool(pr.sig and pr.delta < 0))
    up5 = {k: float(v["k5"].mean()) for k, v in upper.items() if k[2] in res_in}
    T5 = aggregate(stats5, up5, keys_for)
    print_main(T5, f"K=5 SAME DEFINITIONS, published cells (printed daggers), {tag}",
               3 if PHASE == "A" else 6)

    # ---- SAR / BSM / CrossEarth rows (analysis/31's resolver) -------------
    R31rows = {}
    for method in ("upper", "vanilla", "sar", "bsm"):
        for arch in ARCHS_APPF:
            for city in CITIES:
                if method == "bsm" and (arch, city) in BSM_EXCLUDED:
                    R31rows[(method, arch, city)] = dict(per_image=None, blocker=BSM_EXCLUDED[(arch, city)])
                    continue
                R31rows[(method, arch, city)] = R31.resolve_row(method, arch, city)
    for method in ("upper", "vanilla", "fda", "sib", "bsm"):
        for city in CITIES:
            R31rows[(method, "crossearth", city)] = R31.resolve_row(method, "crossearth", city)
    # the upper rows analysis/31 resolves must be the same cells used above
    for arch in ARCHS_APPF:
        for city in CITIES:
            r = R31rows[("upper", arch, city)]
            mine = upper[(arch, city, "hr")]
            if r["run_id"] != mine["run_id"] or abs(r["value"] - mine["k2"].mean()) > 1e-9:
                fail(f"upper/{arch}/{city}: analysis/31 resolves a different upper cell")
            # Vanilla: analysis/31 vs this job's cell (n450 for MAMNet)
            v = R31rows[("vanilla", arch, city)]
            if v["run_id"] != cells[(arch, "vanilla", city, "hr")]["run_id"]:
                fail(f"vanilla/{arch}/{city}: analysis/31 resolves {v['run_id']}")
    ups = {(a, c): R31rows[("upper", a, c)] for a in ARCHS_APPF + ("crossearth",) for c in CITIES}
    extra = {}
    for method in ("sar", "bsm"):
        per = extra_row(method, ARCHS_APPF, R31rows, ups)
        extra[method] = dict(per_cell=per, summary=summarise_extra(per))
    ce = {}
    for method in ("vanilla", "fda", "sib", "bsm"):
        per = extra_row(method, ("crossearth",), R31rows, ups)
        ce[method] = dict(per_cell=per, summary=summarise_extra(per))
    ce_upper = float(np.mean([R31rows[("upper", "crossearth", c)]["value"] for c in CITIES]))

    print("\n=== SAR / BSM ROWS, K=2, high-res only (Sig./3; pooled over available cells) ===")
    for m in ("sar", "bsm"):
        print_extra(MLABEL[m], extra[m]["summary"], extra[m]["per_cell"])
    print(f"\n=== CROSSEARTH BLOCK, K=2, high-res (upper mean {ce_upper:.2f}) ===")
    for m in ("vanilla", "fda", "sib", "bsm"):
        print_extra(MLABEL[m], ce[m]["summary"], ce[m]["per_cell"])

    # ---- CrossEarth test-list check ---------------------------------------
    print("\n=== CROSSEARTH TEST-LIST CHECK vs the canonical high-res held-out list ===")
    ce_list = []
    for city in CITIES:
        canon = list(upper[("mamnet", city, "hr")]["k2"].index)
        cdf = pd.read_parquet(os.path.join(PNG, png_rid("mamnet", "upper", city, "hr"), "perimage.parquet"),
                              columns=["image_id", "n_gt_fg_valid", "n_valid_px"]).set_index("image_id").sort_index()
        for method, tok in (("upper", "vanilla-upper"), ("vanilla", "vanilla-loco"), ("fda", "fda-loco"),
                            ("sib", "sib-loco"), ("bsm", "bsm-loco")):
            rid = f"crossearth-{tok}-r450-{city}-hr-s0"
            d = pd.read_parquet(os.path.join(BASE, "dumps", "st", rid, "perimage.parquet"),
                                columns=["image_id", "n_gt_fg_valid", "n_valid_px"]).set_index("image_id").sort_index()
            same_ids = sorted(d.index) == canon
            gt_eq = int((d.n_gt_fg_valid == cdf.n_gt_fg_valid.reindex(d.index)).sum()) if same_ids else None
            vp_eq = int((d.n_valid_px == cdf.n_valid_px.reindex(d.index)).sum()) if same_ids else None
            ok = same_ids and gt_eq == 150 and vp_eq == 150
            ce_list.append(dict(run_id=rid, n_ids=int(len(d)), overlap=len(set(d.index) & set(canon)),
                                gt_fg_equal=gt_eq, valid_px_equal=vp_eq, correct_list=ok))
            print(f"  {rid:42s} ids {len(d)} overlap {len(set(d.index) & set(canon))} "
                  f"gt_fg= {gt_eq} valid_px= {vp_eq} -> {'CORRECT' if ok else 'WRONG LIST: UNUSABLE'}")
    armclists = [f"crossearth-vanilla-loco-r450-{c}-hr-s0-armclists" for c in CITIES]
    present = [r for r in armclists if os.path.isdir(os.path.join(BASE, "dumps", "st", r))]
    print(f"  -armclists Vanilla cells on the analysis host: {present or 'none'}")
    if not all(r["correct_list"] for r in ce_list):
        fail("a CrossEarth cell carries a wrong test list; the block must drop it (not wired)")

    # ---- section 3.4 preview over the cells present --------------------------
    s34 = sec34(stats, T)
    print(f"\n=== SECTION 3.4 QUANTITIES, K=2, over the {s34['n_cells']} cells present ===")
    for k, v in s34.items():
        print(f"  {k}: {v}")
    s34_5 = sec34({k: dict(v, p=float(F[(F.arch == k[0]) & (F.method == k[1]) & (F.city == k[2])
                                        & (F.res == k[3])].p.iloc[0])) for k, v in stats5.items()}, T5)
    print(f"  [K=5 same cells, printed p]: n_sig {s34_5['n_sig_p05']}, bonf {s34_5['n_sig_bonferroni']}, "
          f"mean {s34_5['mean_delta']:.2f}, worst {s34_5['worst_cell']}, "
          f"range {s34_5['arch_method_delta_min']} .. {s34_5['arch_method_delta_max']}")

    # ---- Appendix F rows (console, high-res in PHASE A) ----------------------
    print("\n=== APPENDIX F PER-CELL, K=2 (printed K=5 beside) ===")
    print(f"{'cell':34s} {'up2':>6s} {'loco2':>6s} {'Δ2':>7s} {'CI2':>17s} {'p2':>7s} s | "
          f"{'up5':>6s} {'loco5':>6s} {'Δ5':>7s} s5 | source")
    for arch in ARCHS_APPF:
        for m in METHODS:
            for city in CITIES:
                for res in res_in:
                    k = (arch, m, city, res)
                    s = stats[k]
                    pr = F[(F.arch == arch) & (F.method == m) & (F.city == city) & (F.res == res)].iloc[0]
                    print(f"{'/'.join(k):34s} {s['upper']:6.2f} {s['loco']:6.2f} {s['mean']:+7.2f} "
                          f"[{s['ci95'][0]:+6.2f},{s['ci95'][1]:+6.2f}] {s['p']:7.4f} {'*' if s['sig'] else '†'} | "
                          f"{pr.up:6.2f} {pr.loco:6.2f} {pr.delta:+7.2f} {'*' if pr.sig else '†'}  | "
                          f"{'n450' if 'n450' in s['run_id'] else 'png'}")

    # ---- Step 6 inventory ----------------------------------------------------
    _, csvM = stats_csv_check(F)
    F5 = {(r.arch, r.method, r.city, r.res): dict(loco=r.loco, mean=r.delta, sig=bool(r.sig), p=r.p)
          for r in F.itertuples()}
    up5_printed = {(r.arch, r.city, r.res): r.up for r in F.itertuples()}
    v5 = inventory_values(F5, up5_printed, F, csvM, full=True)
    v2 = inventory_values(stats, upper_k2, F, None, full=(PHASE == "B"))
    v5["appf_metric"] = "K=5 two-class tolerant only; no K=0 column (Appendix F values reproduce at K=5)"
    v2["appf_metric"] = "K=2 two-class tolerant only in the rebuilt rows; no K=0 column"
    inv_rows = print_inventory(v5, v2)

    if PHASE == "A":
        print("\nPHASE A: nothing written.")
        return

    # ---- format checks: printed K=5 values through this job's formatters -----
    fbad = appf_format_check(F, appf_tex)
    tbad = t2_format_check(T2p, t2src)
    print(f"\nFORMAT CHECK: Appendix F rows re-rendered from printed values: "
          f"{126 - len(fbad)}/126 identical (whitespace-normalised, \\mathbf removed); "
          f"Table 2 method rows: {7 - len(tbad)}/7")
    for b in fbad[:5] + tbad:
        print("   ", b)
    if fbad or tbad:
        fail("a formatter does not reproduce the printed row format")

    # ---- Appendix F pooled tables, K=2 and K=5 (printed-cell recompute) ------
    P2 = pooled_tables(stats)
    P5 = pooled_tables({(r.arch, r.method, r.city, r.res): dict(mean=float(r.delta), sig=bool(r.sig),
                                                                 sig_degrade=bool(r.sig and r.delta < 0))
                        for r in F.itertuples()})
    print("\n=== APPENDIX F POOLED SUMMARIES, K=2 (K=5 from printed cells beside) ===")
    for grp in ("per_city", "per_resolution", "per_method"):
        for k in P2[grp]:
            a, b = P2[grp][k], P5[grp][k]
            print(f"  {grp:15s} {k:10s} mean {a['mean']:+7.2f} best {a['best']:+6.2f} worst {a['worst']:+7.2f} "
                  f"sig {a['sig']}/{a['n']}   | K=5 {b['mean']:+7.2f} {b['best']:+6.2f} {b['worst']:+7.2f} {b['sig']}/{b['n']}")

    # ---- Figure 1 data block -----------------------------------------------
    fig1 = dict(
        what=("Vanilla upper, LOCO and Delta per architecture at K=2. table2_definition: means over "
              "the 6 cells (3 held-out cities x 2 resolutions), Table 2's definition. high_res_only: "
              "the 3 high-resolution cells, as mean and as median (the printed Figure 1 JPEG's values "
              "equal the K=5 high-res medians). CrossEarth has high-resolution cells only."),
        printed_figure1_jpeg_delta=dict(dinov3=-11.1, mamnet=-2.7, oglanet=-5.9),
        per_arch={})
    for a in ARCHS_T2:
        hu = [upper_k2[(a, c, "hr")] for c in CITIES]
        hl = [stats[(a, "vanilla", c, "hr")]["loco"] for c in CITIES]
        hd = [stats[(a, "vanilla", c, "hr")]["mean"] for c in CITIES]
        fig1["per_arch"][a] = dict(
            table2_definition=dict(upper=T[("upper", a)]["value"], loco=T[("vanilla", a)]["loco"],
                                   delta=T[("vanilla", a)]["delta"], n_cells=T[("vanilla", a)]["n"],
                                   n_upper_cells=T[("upper", a)]["n"]),
            high_res_only=dict(upper_mean=float(np.mean(hu)), loco_mean=float(np.mean(hl)),
                               delta_mean=float(np.mean(hd)), upper_median=float(np.median(hu)),
                               loco_median=float(np.median(hl)),
                               delta_median_of_medians=float(np.median(hl) - np.median(hu)),
                               n_cells=3),
            k5_printed_cells=dict(
                table2_definition_delta=T5[("vanilla", a)]["delta"],
                high_res_median_delta=v5["fig1"][a]["delta"]),
            flag=None)
    ceu = [R31rows[("upper", "crossearth", c)]["value"] for c in CITIES]
    cel = [c_["loco"] for c_ in ce["vanilla"]["per_cell"]["crossearth"]]
    ced = [c_["mean"] for c_ in ce["vanilla"]["per_cell"]["crossearth"]]
    fig1["per_arch"]["crossearth"] = dict(
        table2_definition=dict(upper=float(np.mean(ceu)), loco=float(np.mean(cel)), delta=float(np.mean(ced)),
                               n_cells=3, n_upper_cells=3),
        high_res_only=dict(upper_mean=float(np.mean(ceu)), loco_mean=float(np.mean(cel)),
                           delta_mean=float(np.mean(ced)), upper_median=float(np.median(ceu)),
                           loco_median=float(np.median(cel)),
                           delta_median_of_medians=float(np.median(cel) - np.median(ceu)), n_cells=3),
        k5_printed_cells=None,
        flag=("HIGH-RES ONLY, 3 cells: CrossEarth has no mid-resolution cells, so its table2_definition "
              "entry is the 3-cell high-res mean, not a 6-cell mean. Its Vanilla cells are arm F's "
              "(crossearth-vanilla-loco-r450-*-hr-s0), whose training draw differs from arm C's."))
    print("\n=== FIGURE 1 DATA, K=2 ===")
    for a, v in fig1["per_arch"].items():
        t, h = v["table2_definition"], v["high_res_only"]
        print(f"  {a:10s} T2-def ({t['n_cells']} cells) up {t['upper']:.2f} loco {t['loco']:.2f} Δ {t['delta']:+.2f} | "
              f"HR mean up {h['upper_mean']:.2f} loco {h['loco_mean']:.2f} Δ {h['delta_mean']:+.2f} | "
              f"HR median up {h['upper_median']:.2f} loco {h['loco_median']:.2f} Δ {h['delta_median_of_medians']:+.2f}")

    # ---- extremes: worst and best per table and overall (no bold printed) ------
    def lab(k):
        return "/".join(k)
    main_rows = [(lab(k), s["mean"], s["p"], s["sig"]) for k, s in stats.items()]
    sar_rows = [(f"sar/{a}/{c['city']}/hr", c.get("mean"), c.get("p"), c.get("sig"))
                for a, cs in extra["sar"]["per_cell"].items() for c in cs]
    bsm_rows = [(f"bsm/{a}/{c['city']}/hr", c.get("mean"), c.get("p"), c.get("sig"))
                for a, cs in extra["bsm"]["per_cell"].items() for c in cs]
    ce_rows = [(f"crossearth/{m}/{c['city']}/hr", c.get("mean"), c.get("p"), c.get("sig"))
               for m in ce for c in ce[m]["per_cell"]["crossearth"]]
    am = {(m, a): T[(m, a)]["delta"] for m in METHODS for a in ARCHS_T2}
    extremes = dict(
        rule="worst = most negative Delta (LOCO minus within-city upper), best = largest Delta; "
             "no bold is printed in the LaTeX snippets",
        table2_main_block_arch_method=dict(
            worst=dict(method=min(am, key=am.get)[0], arch=min(am, key=am.get)[1], delta=min(am.values())),
            best=dict(method=max(am, key=am.get)[0], arch=max(am, key=am.get)[1], delta=max(am.values()))),
        table2_main_block_cells=extreme(main_rows, "Table 2 main block, the cells it pools"),
        table2_sar_row_cells=extreme(sar_rows, "Table 2 SAR row (high-res)"),
        table2_bsm_row_cells=extreme(bsm_rows, "Table 2 BSM row (high-res)"),
        table2_crossearth_block_cells=extreme(ce_rows, "Table 2 CrossEarth block (high-res)"),
        appendix_f={a: extreme([r for r in main_rows if r[0].startswith(a + "/")],
                               f"Appendix F per-cell table, {ARCH_LABEL[a]}") for a in ARCHS_APPF},
        appendix_f_pooled=jclean(P2),
        overall_126_grid=extreme(main_rows, "the 126-cell grid (Table 2 main block = Appendix F)"),
        overall_all_rows=extreme(main_rows + sar_rows + bsm_rows + ce_rows,
                                 "every cell in either output, incl. SAR/BSM/CrossEarth"),
        printed_bold_k5=[dict(line=int(r.line), cell=f"{r.arch}/{r.method}/{r.city}/{r.res}", delta=float(r.delta))
                         for r in F[F.bold].itertuples()])
    print("\n=== EXTREMES ===")
    for k in ("table2_main_block_cells", "table2_sar_row_cells", "table2_bsm_row_cells",
              "table2_crossearth_block_cells", "overall_126_grid", "overall_all_rows"):
        e = extremes[k]
        print(f"  {k:30s} worst {e['worst']['cell']} {e['worst']['delta']:+.2f}   best {e['best']['cell']} "
              f"{e['best']['delta']:+.2f} (p {e['best']['p']:.4f})")
    for a in ARCHS_APPF:
        e = extremes["appendix_f"][a]
        print(f"  appendix_f/{a:20s} worst {e['worst']['cell']} {e['worst']['delta']:+.2f}   best "
              f"{e['best']['cell']} {e['best']['delta']:+.2f}")

    # ---- arm C p1histmatch degenerate cell (operator item 5; read only) --------
    man = json.load(open(os.path.join(BASE, "logs", "armC_probes_manifest.json")))
    p1 = []
    for r in (x for x in man["runs"] if x["probe"] == 1):
        pq = os.path.join(BASE, "dumps", "st", r["run_id"], "perimage.parquet")
        if not os.path.isfile(pq):
            p1.append(dict(run_id=r["run_id"], present=False))
            continue
        d = pd.read_parquet(pq, columns=["n_pred_pos", "n_valid_px"])
        p1.append(dict(run_id=r["run_id"], present=True, parent=r["parent"],
                       pooled_pred_pos_fraction=float(d.n_pred_pos.sum() / d.n_valid_px.sum())))
    p1 = sorted(p1, key=lambda x: -x.get("pooled_pred_pos_fraction", -1))
    r01 = json.load(open(os.path.join(BASE, "results", "01_scaling_ladder.json")))
    deg = [x for x in p1 if x.get("pooled_pred_pos_fraction", 0) > r01["collapse_rule"]["threshold"]]
    p1hist = dict(
        what=("operator item 5: which arm C probe-1 (*-p1histmatch) cell is the degenerate one results/20 "
              "excludes from its 0-of-26. Read only; no results file is changed."),
        rule=(f"results/01's collapse rule: pooled n_pred_pos / n_valid_px over the cell's 150 images > "
              f"{r01['collapse_rule']['threshold']}"),
        manifest=dict(path="logs/armC_probes_manifest.json",
                      sha256=sha256_file(os.path.join(BASE, "logs", "armC_probes_manifest.json")),
                      n_probe1=sum(1 for x in man["runs"] if x["probe"] == 1)),
        n_present=sum(1 for x in p1 if x["present"]),
        degenerate=deg,
        degenerate_parent_in_results_01_known_collapsed=[x["parent"] in r01["collapse_rule"]["known_collapsed"]
                                                         for x in deg],
        next_highest=[x for x in p1 if x not in deg][:1],
        all_27=p1)
    print("\n=== ARM C p1histmatch DEGENERATE CELL ===")
    for x in deg:
        print(f"  {x['run_id']}  pooled pred-pos fraction {x['pooled_pred_pos_fraction']:.6f}  parent {x['parent']}")
    print(f"  next highest: {p1hist['next_highest']}")

    if DRY_RUN:
        print("DRY RUN: nothing written.")
        return

    # ---- write -----------------------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()
    excl_keys = {tuple(e["cell"].split("/")): e["reason"] for e in excluded}
    A = appf_tables_tex(stats, upper_k2, excl_keys)
    head = [f"% Generated by analysis/{NN}_{NAME}.py at {now}; numbers in results/{NN}_{NAME}.json.",
            "% Metric: two-class boundary-tolerant mIoU at K=2 (per image mean of fg and bg IoU, mean over 150).",
            "% Significance: paired image bootstrap, B=10,000, seed 0, two-tailed, p<0.05 (Appendix F dagger test).",
            "% No bold anywhere (operator); worst/best cells are recorded in the JSON's `extremes`."]
    t2lines = head + ["% tab:loco-main body rows, origin/main column layout l|ccc|ccc|ccc|cc (D M O).",
                      "% Main block: 6 cells per arch x method (3 held-out cities x 2 resolutions);",
                      "% Sig./6 counts cells significant in either direction (the Appendix F dagger convention).", ""] \
        + t2_tex(T, extra, ce, ce_upper)
    alines = list(head)
    for arch in ARCHS_APPF:
        alines += ["", f"% ==== tab:loco-full-{arch} ({ARCH_LABEL[arch]}) body rows: "
                       "Method & City & Res. & Upper & LOCO & Delta & 95% CI & p ===="] + A[arch]
    alines += ["", "% ==== pooled summaries (Tables 33-35 in the origin/main build) ===="] + pooled_tex(P2)
    sha_t2 = write_new(os.path.join(OUT, "tab_loco_main_rows.tex"), "\n".join(t2lines) + "\n")
    sha_af = write_new(os.path.join(OUT, "app_loco_extended_rows.tex"), "\n".join(alines) + "\n")

    idx_paths = {}
    for label, idx in (("st_r450_published_png", idx_png), ("st_r450_published_recompute", idx_rec)):
        pth = os.path.join(INV, f"INDEX_{label}_{NN}.parquet")
        if os.path.exists(pth):
            fail(f"{os.path.relpath(pth, BASE)} exists; never overwrite a cited index")
        idx.to_parquet(pth + ".tmp.parquet", index=False)
        os.replace(pth + ".tmp.parquet", pth)
        idx_paths[label] = dict(path=os.path.relpath(pth, BASE), sha256=sha256_file(pth),
                                n_rows=int(len(idx)), tree=os.path.relpath(PNG if "png" in label else RECOMPUTE, BASE))

    runs = set()
    for k, s in stats.items():
        runs |= {s["run_id"], s["upper_run_id"]}
    for (m, a, c), r in R31rows.items():
        for kk, vv in r.items():
            if kk.endswith("run_id") and isinstance(vv, str):
                runs.add(vv)
    runs |= {x["run_id"] for x in ce_list}
    runs_excluded = list(excluded) + [dict(run_id="dinov3-bsm-loco-r450-chicago-hr-s0",
                                           reason=BSM_EXCLUDED[("dinov3", "chicago")])]
    # cells whose run_id is excluded must not also appear in runs
    runs -= {e.get("run_id") for e in excluded if e.get("run_id")}

    n = s34["n_cells"]
    gate_fail = [g for g in gate_rows if not g["passed"]]
    n_sig_pos = sum(1 for s in stats.values() if s["sig"] and s["mean"] > 0)
    ce_ok = all(x["correct_list"] for x in ce_list)
    headline = (f"On two-class boundary-tolerant mIoU at K=2, {s34['n_sig_p05']} of {n} LOCO cells differ "
                f"from the within-city upper bound at p<0.05 (paired image bootstrap, B=10,000), the mean "
                f"Delta over the {n} cells is {s34['mean_delta']:+.2f} mIoU, and the worst cell is "
                f"{s34['worst_cell']['cell']} at {s34['worst_cell']['delta']:+.2f}.")
    caveats = [
        "Metric: two-class boundary-tolerant mIoU at K=2 (per image, mean of foreground and background IoU from "
        "tp_tol/fp_tol/fn_tol/tn_tol; Appendix E's degenerate rule: no foreground and no predicted positive -> "
        "foreground IoU per the stored convention; cell = mean over 150 images). The printed Table 2 and Appendix F "
        "are K=5 two-class; every printed magnitude in them changes.",
        "Mid-resolution LOCO cells (63) are the published runs at their published training budget; no matched-450 "
        "retrain exists for any mid-resolution cell. They were scored by Delta (job "
        f"{sha_info['delta_results']['slurm_job_id']}, python/armB_png_score.py:score_cell) from the saved "
        "prediction PNG masks. The 6-cell means therefore pool matched-450 retrains (high-res) with published-budget "
        "cells (mid-res) for MAMNet (all 7 methods) and OGLANet-SegDeSiC.",
        f"Mid-resolution gate: {sum(g['passed'] for g in gate_rows)} of {len(gate_rows)} cells passed "
        f"(perimage.parquet sha256 equal to Delta's {sha_info['path']} AND |K=5 recomputed - Appendix F LOCO| < "
        f"{MIDRES_GATE_TOL}). "
        + ("No cell failed." if not gate_fail else
           "Excluded from every output: " + "; ".join(
               f"{g['run_id']} (printed {g['printed']:.2f}, recomputed K=5 "
               f"{'n/a' if g['recomputed_k5'] is None else format(g['recomputed_k5'], '.4f')}, diff "
               f"{'n/a' if g['diff'] is None else format(g['diff'], '+.4f')}; {g['reason']})" for g in gate_fail)),
        "SAR and BSM rows are high-resolution only (Sig./3 per architecture; pooled over 9 cells, BSM over 8). "
        "They have no mid-resolution cells and are not comparable in scope to the 6-cell main block.",
        "The SAR row is built as results/31 built it: Vanilla's per-image values plus SAR's within-pass change "
        "(MAMNet SAR from its n450 runs in delta_runs/).",
        "BSM DINOv3/Chicago is excluded: " + BSM_EXCLUDED[("dinov3", "chicago")] + ".",
        "One training seed per matched-450 cell; no seed band exists for any n450 substitution.",
        "Every OGLANet cell is a single inference pass at the published batch size; OGLANet inference is "
        "nondeterministic (REVIEWS.md E-9: cell-mean SD up to 0.4247 mIoU over repeated passes), so OGLANet "
        "cell values and their p-values carry pass-to-pass variation the bootstrap does not include.",
        "The high-resolution substitutions are 24, not 27: results/21 finding 1 names 27 budget-asymmetric cells, "
        "and 3 of them are OGLANet-SIB (" + ", ".join(FINDING1_NOT_IN_TABLE2) + "), which is not a Table 2 row.",
        "CrossEarth Vanilla comes from arm F (crossearth-vanilla-loco-r450-{city}-hr-s0), whose training draw "
        "differs from arm C's. The -armclists Vanilla cells are not on the analysis host"
        + (f" (present: {present})" if present else "") + ".",
        f"CrossEarth test lists: {sum(x['correct_list'] for x in ce_list)} of {len(ce_list)} CrossEarth cells carry "
        "the canonical 150 high-resolution test image_ids with equal per-image GT foreground and valid-pixel counts. "
        "This contradicts delta_logs/session_logs/2026-11-01-crossearth-ladder-release-step1-2.md, which records "
        "150/150 different ids.",
        "The CrossEarth block is high-resolution only (3 cells per row); it is not pooled with the main block.",
        "Validator tolerance: stored-vs-recomputed K=2 columns are asserted to 1e-6 (analysis/31 uses 1e-9) "
        "because mamnet-none-upper-r450-phoenix-mr-s0 deviates by 2e-9; the largest deviation of every cell "
        "is recorded per cell.",
        "Sig. counts cells significant in either direction (the Appendix F dagger convention, which the printed "
        f"Sig. columns follow); {n_sig_pos} of {n} cells are significant with Delta > 0. sig_degrade counts are "
        "recorded beside every Sig. count.",
        "p = 0 (no bootstrap replicate on the far side of zero) is printed as <.0002, the resolution of "
        "p = min(2*prop, 1) at B = 10,000. One bootstrap seed (0) is used for every cell.",
        f"Bonferroni over the {n}-cell family leaves {s34['n_sig_bonferroni']} significant "
        f"(threshold {s34['bonferroni_threshold']:.3g}); §3.4's 'robust to Bonferroni' is checked against this.",
        "Figure 1 data: the printed JPEG's Delta values (-11.1 / -2.7 / -5.9) equal the K=5 high-resolution "
        "medians, not the 6-cell pooling its caption and notebook docstring describe; both K=2 versions are in "
        "figure1_data. CrossEarth's entry is high-resolution only (3 cells).",
        "Appendix F's mid-resolution LOCO values enter only through the gate; the 81-cell K=5 regression (upper "
        "18 + high-res published LOCO 63) stays at 0.00 to 2 dp.",
        "REVIEWS.md E-9 (OGLANet nondeterminism) is carried as a caveat, not answered.",
        "app_metric.tex states Appendix F prints strict (K=0) and tolerant numbers side by side; neither the "
        "printed Appendix F nor these rebuilt rows carry a K=0 column.",
    ]
    sanity = [
        f"K=5 inventory: printed numbers that do not equal their K=5 recompute from Appendix F: "
        + "; ".join(f"{r['file']}:{r['line']} printed {r['printed']} vs K=5 {r['k5']}" for r in inv_rows
                    if r["key"] in ("n_bonf", "am_max", "am_min", "n_below_vanilla", "every_cell",
                                    "T2:vanilla:dinov3", "T2:isw:oglanet", "T2:mrfp_plus:oglanet",
                                    "T2:isw:dinov3", "T2:mrfp_plus:dinov3", "dino_mr_nonsig", "fig1")),
        "Mid-res gate cross-check against Delta's results.json: max |recomputed K=5 - Delta's K=5| = "
        + (f"{max(abs(g['k5_minus_delta_reported']) for g in gate_rows if g['k5_minus_delta_reported'] is not None):.2e}"
           if any(g['k5_minus_delta_reported'] is not None for g in gate_rows) else "n/a")
        + ", max |recomputed K=2 - Delta's K=2| = "
        + (f"{max(abs(g['k2_minus_delta_reported']) for g in gate_rows if g['k2_minus_delta_reported'] is not None):.2e}"
           if any(g['k2_minus_delta_reported'] is not None for g in gate_rows) else "n/a")
        + "; the same parquet is read on both sides, so agreement to float precision is expected, not evidence.",
        f"run.json sha256 vs Delta's manifest: {sum(bool(g['run_json_sha_match']) for g in gate_rows)} of "
        f"{len(gate_rows)} match (recorded, not gated); perimage sizes match on "
        f"{sum(g['perimage_size_got'] == g['perimage_size_expected'] for g in gate_rows)} of {len(gate_rows)}.",
        f"Upper bound ranking at K=2 (6-cell means): "
        + " > ".join(f"{ARCH_LABEL[a]} {T[('upper', a)]['value']:.2f}" for a in
                     sorted(ARCHS_T2, key=lambda a: -T[("upper", a)]["value"]))
        + "; printed Table 2 (K=5 claim): " + " > ".join(f"{ARCH_LABEL[a]} {T2p['upper'][a]}" for a in
                                                       sorted(ARCHS_T2, key=lambda a: -T2p["upper"][a]))
        + "; same definitions on the published K=5 cells: " + " > ".join(
            f"{ARCH_LABEL[a]} {T5[('upper', a)]['value']:.2f}" for a in sorted(ARCHS_T2, key=lambda a: -T5[("upper", a)]["value"])),
        "Printed Table 2 upper row (" + ", ".join(f"{ARCH_LABEL[a]} {T2p['upper'][a]}" for a in ARCHS_T2) + ") "
        f"is not reproduced by the 6-cell K=5 mean of Appendix F's printed uppers; this job does not search for "
        "its origin (author decision 2).",
    ]
    shift = []
    for (a, c, r), u in upper.items():
        d = pd.read_parquet(os.path.join(PNG, u["run_id"], "perimage.parquet"),
                            columns=["tp_tol", "fn_tol", "fp_tol", "tp_k5", "fn_k5", "fp_k5"])
        shift.append(dict(run_id=u["run_id"], k2=float(u["k2"].mean()), k5=float(u["k5"].mean()),
                          k2_minus_k5=float(u["k2"].mean() - u["k5"].mean()),
                          no_gt_fg_images_k2=int((d.tp_tol + d.fn_tol == 0).sum()),
                          no_gt_fg_images_k5=int((d.tp_k5 + d.fn_k5 == 0).sum()),
                          no_gt_fg_with_pred_pos_k2=int(((d.tp_tol + d.fn_tol == 0) & (d.fp_tol > 0)).sum()),
                          no_gt_fg_with_pred_pos_k5=int(((d.tp_k5 + d.fn_k5 == 0) & (d.fp_k5 > 0)).sum())))
    shift = sorted(shift, key=lambda x: x["k2_minus_k5"])
    dl = [stats[k]["loco"] - cells[k].get("published", cells[k])["k5"].mean() for k in stats
          if "published" not in cells[k]]
    sanity.append(
        f"K=2 minus K=5 on the 18 upper cells ranges from {shift[0]['k2_minus_k5']:+.2f} ({shift[0]['run_id']}) to "
        f"{shift[-1]['k2_minus_k5']:+.2f} ({shift[-1]['run_id']}); on the {len(dl)} LOCO cells scored from the "
        f"same dump at both widths, from {min(dl):+.2f} to {max(dl):+.2f}. Images with no ground-truth foreground "
        "after the band: " + "; ".join(f"{x['run_id']} {x['no_gt_fg_images_k2']} at K=2 vs {x['no_gt_fg_images_k5']} "
                                        f"at K=5" for x in shift[-2:]) + ". Per-cell counts in upper_k2_k5_shift.")
    conditions, tests = [], []
    for k, s in stats.items():
        conditions.append(dict(label=f"{lab(k)} LOCO minus within-city upper", metric="delta_two_class_mIoU_tol_K2_pct",
                               mean=s["mean"], stderr=s["stderr"], n=s["n"], ci95=s["ci95"],
                               ci_method="paired_image_bootstrap_B10000_percentile", baseline=s["upper_run_id"]))
        tests.append(dict(name="paired bootstrap (published protocol)", compares=[s["run_id"], s["upper_run_id"]],
                          statistic=s["mean"], p=s["p"], B=B, two_tailed=True, seed=BOOT_SEED,
                          resampling_unit="test image (150 per cell), paired on image_id"))
    for rowname, rows_ in (("sar", extra["sar"]["per_cell"]), ("bsm", extra["bsm"]["per_cell"]),
                           *((f"crossearth_{m}", ce[m]["per_cell"]) for m in ce)):
        for a, cs in rows_.items():
            for c in cs:
                if c.get("loco") is None:
                    continue
                conditions.append(dict(label=f"{rowname}/{a}/{c['city']}/hr LOCO minus within-city upper",
                                       metric="delta_two_class_mIoU_tol_K2_pct", mean=c["mean"], stderr=c["stderr"],
                                       n=c["n"], ci95=c["ci95"], ci_method="paired_image_bootstrap_B10000_percentile",
                                       baseline=c["upper_run_id"]))
                tests.append(dict(name="paired bootstrap (published protocol)", compares=[c["run_id"], c["upper_run_id"]],
                                  statistic=c["mean"], p=c["p"], B=B, two_tailed=True, seed=BOOT_SEED,
                                  resampling_unit="test image (150 per cell), paired on image_id"))

    def t2_json(T_):
        out = dict(upper={a: T_[("upper", a)] for a in ARCHS_T2},
                   rows={m: dict({a: T_[(m, a)] for a in ARCHS_T2}, pooled=T_[(m, "pooled")]) for m in METHODS},
                   avg_delta={a: T_[("avg_delta", a)] for a in ARCHS_T2},
                   sig_all={a: T_[("sig_all", a)] for a in ARCHS_T2},
                   n_all={a: T_[("n_all", a)] for a in ARCHS_T2})
        return out

    appf_cells = []
    for k, s in stats.items():
        pr = F[(F.arch == k[0]) & (F.method == k[1]) & (F.city == k[2]) & (F.res == k[3])].iloc[0]
        c = cells[k]
        appf_cells.append(dict(
            arch=k[0], method=k[1], city=k[2], res=k[3], run_id=s["run_id"], tree=s["tree"], source=s["source"],
            upper_run_id=s["upper_run_id"], upper=s["upper"], loco=s["loco"], delta=s["mean"], stderr=s["stderr"],
            ci95=s["ci95"], p=s["p"], sig=s["sig"], sig_degrade=s["sig_degrade"],
            empty_fg_images=c.get("empty_fg"), stored_column_maxdev=c.get("stored_maxdev"),
            published_run_id=c.get("published_run_id"),
            printed_k5=dict(upper=float(pr.up), loco=float(pr.loco), delta=float(pr.delta),
                            ci95=[float(pr.ci_lo), float(pr.ci_hi)], p=float(pr.p), p_floor=bool(pr.p_floor),
                            dagger=bool(pr.dagger), bold=bool(pr.bold), line=int(pr.line))))

    J = dict(
        id=f"{NN}_{NAME}", arm="H2",
        answers=["AC1", "E-2", "E-3", "E-4", "E-7"],
        answers_detail=("E-7: the printed Table 2 and Appendix F are rebuilt on the K=2 metric the author fixed. "
                        "E-2: the upper-bound row is rebuilt from per-cell values (ranking in sanity_flags). "
                        "E-3, E-4: the section 3.4 / intro / abstract quantities are recomputed in sec34 and "
                        "inventory. AC1: the 24 budget-asymmetric high-res cells take matched-450 retrains. "
                        "E-9 (OGLANet nondeterminism) is carried as a caveat, not answered."),
        question=("What do Table 2 (tab:loco-main) and Appendix F (Tables 30-35 of the origin/main build) read "
                  "when rebuilt from per-cell dumps at K=2, with the Appendix F significance test?"),
        status="COMPLETED",
        metric=dict(name="two_class_mIoU_tol_K2_pct",
                    definition=("per image, mean of foreground IoU tp/(tp+fp+fn) and background IoU tn/(tn+fp+fn) "
                                "from the *_tol counts (+-2 px band); cell = mean over 150 test images; K=5 uses "
                                "the *_k5 counts with the same formula"),
                    implementation="analysis/31_internal_sib_worst_case_tgtfda.py:two_class_tol_k2 (imported)"),
        significance=dict(
            protocol="python/statistical_analysis.py:bootstrap_comparison as documented in python/armc_ladder_stats.py",
            resampling_unit="the test image: 150 per-image deltas (LOCO minus within-city upper, paired on image_id) "
                            "resampled with replacement",
            B=B, seed=BOOT_SEED, seed_note="numpy default_rng(0), re-seeded for every cell", ci="percentile 2.5/97.5",
            p="two-tailed, min(2 * share of replicates on the far side of zero from the observed mean, 1)",
            alpha=ALPHA,
            unit_confirmed_at_k5=usumm,
            statistical_results_csv_vs_printed_appendix_f=csv_chk),
        k5_regression=dict(n_cells=int(len(R)), max_abs_round2_diff=float(R["diff"].abs().max()),
                           max_abs_raw_diff=float(R.absdiff_raw.max()), failures=int(len(bad)),
                           scope="18 upper + 63 high-res published LOCO cells (PNG tree); mid-res LOCO is gated"),
        midres_gate=dict(rule=f"perimage.parquet sha256 equals Delta's manifest AND |K=5 recomputed - parsed "
                              f"Appendix F LOCO| < {MIDRES_GATE_TOL}; failing cells excluded from every output; "
                              "nothing adjusted",
                         tol=MIDRES_GATE_TOL, sources=sha_info, n_cells=len(gate_rows),
                         n_pass=sum(g["passed"] for g in gate_rows), rows=gate_rows),
        table2=dict(
            definitions=("Upper, LOCO, Delta: means over the 6 cells per architecture (3 held-out cities x 2 "
                         "resolutions); Sig./6 = cells with p<0.05 in either direction; pooled Delta over 18 cells, "
                         "Sig./18; Avg. Delta = mean over the 7 methods"),
            main_k2=t2_json(T), main_k5_same_definitions_published_cells=t2_json(T5),
            printed_k5=T2p,
            sar_hr=extra["sar"], bsm_hr=extra["bsm"],
            crossearth_hr=dict(upper_mean=ce_upper,
                               upper_cells={c: R31rows[("upper", "crossearth", c)]["run_id"] for c in CITIES},
                               rows=ce)),
        appendix_f=dict(cells=appf_cells, pooled_k2=P2, pooled_k5_from_printed_cells=P5,
                        is_tables_30_35="yes: per-cell Tables 30-32 (tab:loco-full-{mamnet,oglanet,dinov3}) and "
                                        "pooled Tables 33-35 (tab:loco-per-city, -per-resolution, -method-summary) "
                                        "in the origin/main build"),
        extremes=extremes,
        upper_k2_k5_shift=shift,
        sec34=dict(k2=s34, k5_same_cells_printed_p=s34_5),
        inventory=inv_rows,
        figure1_data=fig1,
        crossearth_test_list=dict(rows=ce_list, all_correct=ce_ok, armclists_present=present,
                                  contradicts="delta_logs/session_logs/2026-11-01-crossearth-ladder-release-step1-2.md "
                                              "('150/150 different')"),
        p1histmatch_degenerate=p1hist,
        format_check=dict(appendix_f_rows_identical=126 - len(fbad), table2_method_rows_identical=7 - len(tbad),
                          rule="printed K=5 values rendered through this job's formatters equal the origin/main "
                               "lines after whitespace normalisation and \\mathbf removal"),
        outputs=dict(tab_loco_main_rows=dict(path=f"analysis/out/tab_loco_main_rows.tex", sha256=sha_t2),
                     app_loco_extended_rows=dict(path=f"analysis/out/app_loco_extended_rows.tex", sha256=sha_af)),
        conditions=conditions, tests=tests,
        runs=sorted(runs), runs_excluded=runs_excluded,
        index_sha256=idx_paths["st_r450_published_png"]["sha256"], index_sha256_all=idx_paths,
        index_info=dict(png={k: v for k, v in info_png.items() if k != "unindexable"},
                        recompute={k: v for k, v in info_rec.items() if k != "unindexable"}),
        paper=dict(ref=PAPER_REF, commit=paper_commit()),
        generated_utc=now, script=f"analysis/{NN}_{NAME}.py",
        script_sha256=sha256_file(os.path.abspath(__file__)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 conda_env="myenv (conda env)", pythonnousersite="1"),
        seed=BOOT_SEED,
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; only final_paper/ is git-connected "
                        "and this job does not touch it",
        compute_note="read-only CPU analysis on the analysis host; no training, no GPU inference; nothing under dumps/, "
                     "results/ (other than this file) or final_paper/ modified",
        headline=headline, caveats=caveats, sanity_flags=sanity,
        figure=None, caption=None)
    J = jclean(J)
    txt = json.dumps(J, indent=1, allow_nan=False)
    sha_j = write_new(RESULTS, txt + "\n")
    print(f"\nWROTE {os.path.relpath(RESULTS, BASE)} sha256 {sha_j[:8]}; {len(caveats)} caveats; "
          f"{len(J['runs'])} runs; index {idx_paths['st_r450_published_png']['sha256'][:8]}")
    print(f"WROTE analysis/out/tab_loco_main_rows.tex {sha_t2[:8]}, analysis/out/app_loco_extended_rows.tex {sha_af[:8]}")
    print(f"HEADLINE: {headline}")
    return dict(J=J, sha_j=sha_j)


if __name__ == "__main__":
    main()
