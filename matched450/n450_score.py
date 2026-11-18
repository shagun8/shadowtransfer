#!/usr/bin/env python3
"""
n450 retrain -- score one retrained cell from its saved probability maps.

SAME RE-SCORING CODE AS THE ORIGINALS. The originals' per-image outputs in
dumps/st_r450_published_recompute/ were written by armc_r450_rescore.py (and,
for the three MAMNet vanilla cells, by its predecessor armc_r450.py), whose
per-image loop is: threshold_and_filter(prob, min_pixels=10) ->
armc_dump.perimage_row(...) for every row, plus a second no-filter recipe
(p > 0.5) summarised in run.json. This file runs that loop over the new cell's
maps by importing those exact functions; it computes no metric itself.
armc_r450_rescore.main() is not called because it hard-codes the original
run_id (`...-s0`), the original's probability tree and the original's
provenance into run.json.

Writes <dump_dir> atomically: everything goes to <dump_dir>.partial, the
parquet is read back and checked against the originals' schema, then the
directory is renamed into place. .DONE is inside, written last.
"""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BASE = Path(os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from shadowtransfer import armc_dump                       # noqa: E402
from shadowtransfer.armc_dump import sha256_file           # noqa: E402
from shadowtransfer.metrics import per_image_counts, threshold_and_filter  # noqa
import armc_r450_rescore as R450                           # noqa: E402

METRICS_PY = BASE / "shadowtransfer/metrics.py"
D19 = "d19d5e948f79c5498bf21a0e47d1302b4b13e8a97ff7d2c2a48033d9f1514e13"
D19_COMMIT = "06021b6"
SCHEMA_REF = (Path(os.environ["ST_SCRATCH_ROOT"]) / ("dumps/"
                   "st_r450_published_recompute/mamnet-fda-loco-r450-chicago-hr-s0"
                   "/perimage.parquet"))


def code_shas():
    return {
        "metric_module_sha": sha256_file(METRICS_PY),
        "dump_module_sha": sha256_file(BASE / "shadowtransfer/armc_dump.py"),
        "rescore_module_sha": sha256_file(BASE / "support/armc_r450_rescore.py"),
        "scorer_sha": sha256_file(Path(__file__).resolve()),
    }


def metric_sha_relation(orig_sha):
    """How the current metrics.py relates to the one an original was scored
    with. EXACT, or PREFIX (the original's revision is a byte-prefix of the
    current file and the appended text defines no name the prefix defines), or
    MISMATCH."""
    cur = sha256_file(METRICS_PY)
    if orig_sha is None:
        return {"relation": "ORIGINAL_DUMP_PENDING", "current": cur}
    if orig_sha == cur:
        return {"relation": "EXACT", "current": cur, "original": orig_sha}
    if orig_sha != D19:
        return {"relation": "MISMATCH", "current": cur, "original": orig_sha}
    old = subprocess.run(["git", "-C", str(BASE), "show",
                          f"{D19_COMMIT}:shadowtransfer/metrics.py"],
                         capture_output=True, check=True).stdout
    import hashlib
    if hashlib.sha256(old).hexdigest() != D19:
        return {"relation": "MISMATCH", "current": cur, "original": orig_sha,
                "note": f"git blob at {D19_COMMIT} is not {D19}"}
    new = METRICS_PY.read_bytes()
    if not new.startswith(old):
        return {"relation": "MISMATCH", "current": cur, "original": orig_sha}

    def names(src):
        return {n.name for n in ast.parse(src).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))} | {
            t.id for n in ast.parse(src).body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}
    appended = new[len(old):].decode()
    clash = names(old.decode()) & names(appended) if appended.strip() else set()
    rel = "PREFIX" if not clash else "MISMATCH"
    ruling = ("operator ruling 2026-10-26 (n450 decision 1): PREFIX accepted "
              "for the three MAMNet standard originals") if rel == "PREFIX" else None
    return {"relation": rel, "current": cur, "original": orig_sha,
            "appended_bytes": len(new) - len(old),
            "appended_defines": sorted(names(appended)) if appended.strip() else [],
            "redefined_names": sorted(clash), "operator_ruling": ruling,
            "note": "literal sha assertion FAILS; the original's metrics.py is a "
                    "byte-prefix of the current one and the appended block "
                    "redefines nothing, so every function the scorer calls is "
                    "byte-identical"}


def original_scores(cell):
    out = {"orig_run_id": cell["orig_run_id"]}
    rj = Path(cell["orig_dump_dir"]) / "run.json"
    if (Path(cell["orig_dump_dir"]) / ".DONE").exists() and rj.exists():
        r = json.loads(rj.read_text())
        out.update(orig_dump_fg_iou_tol_pct=r.get("headline_fg_iou_tol_pct"),
                   orig_dump_metric_module_sha=r.get("metric_module_sha"),
                   orig_dump_module_sha=r.get("dump_module_sha"),
                   orig_dump_written_utc=r.get("written_utc"))
    else:
        out["orig_dump_fg_iou_tol_pct"] = None
        out["orig_dump_note"] = "original's re-score dump not written yet"
    fam = "sib" if cell["method"] == "sibc4" else "classic"
    try:
        v, field, p, s = R450.repro_target(Path(cell["orig_dir"]), fam)
        out.update(orig_test_results_fg_k2=v, orig_test_results_field=field,
                   orig_test_results_path=p, orig_test_results_sha256=s)
    except SystemExit as e:
        out["orig_test_results_error"] = str(e)
    return out


def own_test_results_fg(path, family):
    t = json.loads(Path(path).read_text())
    if family == "sib":
        return float(t["tolerant_2px"]["Shadow_IOU"]), "tolerant_2px.Shadow_IOU"
    return (float(t["detailed"]["boundary_tolerant"]["tolerant_2px"]["iou"]),
            "detailed.boundary_tolerant.tolerant_2px.iou")


def score(run_id, cell, probs_dir, dump_dir, run, limit=0, own_test_json=None):
    t0 = time.time()
    probs_dir, dump_dir = Path(probs_dir), Path(dump_dir)
    mask_dir = R450.mask_dir_for("loco", cell["city"], "highres")
    ps = {p.stem for p in probs_dir.glob("*.npy")}
    ms = {p.stem for p in mask_dir.glob("*.png")}
    if ps != ms or len(ps) != 150:
        raise SystemExit(f"FATAL stem sets: probs {len(ps)} masks {len(ms)} "
                         f"sym-diff {sorted(ps ^ ms)[:5]}")
    stems = sorted(ps)[: (limit or None)]
    geo = armc_dump.load_geo_meta(BASE)
    rows, probs, nf_fg, nf_mi = [], {}, [], []
    for stem in stems:
        p16 = np.load(probs_dir / f"{stem}.npy")
        prob = p16.astype(np.float32)
        raw = cv2.imread(str(mask_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != prob.shape:
            raise SystemExit(f"FATAL mask {stem}: {None if raw is None else raw.shape}")
        gt = (raw > 127).astype(np.uint8)
        pred = threshold_and_filter(prob, min_pixels=10)
        rows.append(armc_dump.perimage_row(run_id, stem, cell["city"], "hr", gt,
                                           prob, pred, geo.get(f"{stem}.png")))
        probs[stem] = p16
        c = per_image_counts((prob > 0.5).astype(np.uint8), gt, boundary_tolerance=2)
        f_, _, m_ = R450.two_class(c["tp_tol"], c["fp_tol"], c["tn_tol"], c["fn_tol"])
        nf_fg.append(f_)
        nf_mi.append(m_)
    import pandas as pd
    df = pd.DataFrame(rows)
    fg = float(df["iou_tol"].mean()) * 100.0
    fam = "sib" if cell["method"] == "sibc4" else "classic"
    sc = {
        "headline_metric": "foreground-only IoU, K=2 don't-care band, per-image "
                           "mean (Appendix E; CONTRACT section 3 iou_tol)",
        "headline_fg_iou_tol_pct": fg,
        "miou_k2_twoclass": float(df["miou_tol_pct"].mean()),
        "miou_k0_twoclass": float(df["miou_str_pct"].mean()),
        "prediction_recipe": "threshold_and_filter(prob, min_pixels=10) -- the "
                             "originals' primary recipe",
        "nofilter_recipe": {"recipe": "p > 0.5, no size filter",
                            "fg_iou_k2_pct": float(np.mean(nf_fg)),
                            "miou_k2_twoclass": float(np.mean(nf_mi))},
        "n_images_scored": len(df), "smoke_limit": limit or None,
    }
    if own_test_json and Path(own_test_json).exists():
        v, field = own_test_results_fg(own_test_json, fam)
        cmp_ = sc["nofilter_recipe"]["fg_iou_k2_pct"] if fam == "sib" else fg
        sc["self_check"] = {
            "own_test_results": str(own_test_json), "field": field, "value": v,
            "compared_recipe": "nofilter" if fam == "sib" else "filtered",
            "delta": cmp_ - v, "pass_0p01": abs(cmp_ - v) <= 0.01,
            "note": "the new cell's own trainer test() vs this scorer, same "
                    "checkpoint; n<150 in a smoke makes this uninformative"}
    sc["original"] = original_scores(cell)
    o = sc["original"].get("orig_dump_fg_iou_tol_pct")
    sc["delta_new_minus_orig_dump"] = (fg - o) if (o is not None and not limit) else None
    sc["metric_sha_check"] = metric_sha_relation(
        sc["original"].get("orig_dump_metric_module_sha"))
    run = dict(run)
    run.update(code_shas())
    run["scoring"] = sc
    run["headline_fg_iou_tol_pct"] = fg
    run["source_probs_dir"] = str(probs_dir)
    run["source_mask_dir"] = str(mask_dir)
    run["score_wall_s"] = time.time() - t0
    part = Path(str(dump_dir) + ".partial")
    if part.exists():
        part.rename(str(part) + f".stale_{os.environ.get('SLURM_JOB_ID', 'x')}")
    armc_dump.write_dump(part, rows, probs, run)
    back = pd.read_parquet(part / "perimage.parquet")
    if len(back) != len(rows) or list(back.columns) != list(df.columns):
        raise SystemExit("FATAL parquet read-back mismatch")
    if SCHEMA_REF.exists():
        ref = pd.read_parquet(SCHEMA_REF)
        if list(ref.columns) != list(back.columns):
            raise SystemExit(f"FATAL schema != originals': "
                             f"{set(ref.columns) ^ set(back.columns)}")
    if dump_dir.exists():
        left = [x.name for x in dump_dir.iterdir()]
        if ".DONE" in left or any(not n.startswith(".FAILED") for n in left):
            raise SystemExit(f"FATAL {dump_dir} exists with {left}; not replacing")
        dump_dir.rename(str(dump_dir) + f".failed_{os.environ.get('SLURM_JOB_ID','x')}")
    part.rename(dump_dir)
    print(f"[n450_score] {run_id}: fg-only IoU K=2 = {fg:.4f}  "
          f"(n={len(df)})  -> {dump_dir}", flush=True)
    return run
