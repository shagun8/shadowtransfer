#!/usr/bin/env python3
"""Score the 63 published MID-RESOLUTION LOCO cells' saved prediction PNG masks.

Why. The paper is moving to one metric everywhere: two-class boundary-tolerant
mIoU at K=2 (CONTRACT.md rule zero). the analysis host already holds per-image dumps for
the 72 high-resolution cells and the 9 mid-resolution upper bounds
(`dumps/st_r450_published_png/`). The 63 mid-resolution LOCO cells --
{mamnet,oglanet,dinov3} x {vanilla,fda,segdesic,iim,isw,mrfp_plus,fada} x
{chicago,miami,phoenix} -- have no per-image dump anywhere. This driver creates
them by scoring the SAVED published prediction masks. No model is loaded and no
inference is run, deliberately: OGLANet inference depends on the RNG and on
batch size, so a re-inference would not be the published prediction.

Nothing here computes a metric. Every count comes from
`armB_png_score.score_cell`, used UNCHANGED, which scores through
`shadowtransfer.armc_dump.perimage_row` and therefore
`shadowtransfer.metrics.per_image_counts`. This file only builds the 63 cell
dicts, and afterwards patches each `run.json` with provenance that
`score_cell` cannot know (see `patch_provenance`).

Per-cell inputs, every one verified on disk by the STEP 1 LOCATE pass rather
than assumed (2026-11-03):
  * predictions `data/Test_img_results/loco/{city}/midres/{arch}/{method}`
    -- 150 PNGs in every one of the 63 cells;
  * probabilities `data/Test_img_probs/loco/{city}/midres/{arch}/{method}`;
  * GT masks `data/Final_data_loco/{fold}/midres/test/masks`, which is the
    tree the high-resolution LOCO cells' own recompute run.json names as
    `source_mask_dir`. Measured byte-identical (md5, 150/150 files, all three
    cities) to `data/Final_data_test/{city}/midres/test/masks`, the tree the 9
    existing midres upper-bound dumps scored against, so the GT choice moves
    no number;
  * the 150 image ids equal, exactly and in order, the canonical midres test
    list of that city -- defined as the `image_id` column of the existing
    `{arch}-none-upper-r450-{city}-mr-s0` dump, which all three architectures
    agree on.

Four comparison fields are null by construction, not by omission:
`table38_printed`, `fc_mean` / `final_comparison_*`, and the recompute
counterpart. `data/final_comparison_output/final_comparison.json` is the
HIGH-RESOLUTION family only (every `pred_dir` it names is under `/highres/`),
and `dumps/st_r450_published_recompute/` holds 108 cells of which 0 are `-mr-`.
The STEP 3 gate is against Appendix F's printed mid-resolution values instead,
supplied by `--appendix_f`; see `apply_gate`.

CPU only: no model, no inference, no GPU compute.
Writes only under `dumps/st_r450_published_png/{run_id}/` and the `--out` JSON.
"""
import argparse
import json
import os
import sys
import time

BASE = os.environ["PROJECT_ROOT"]
SCRATCH = os.path.join(os.environ["ST_SCRATCH_ROOT"], "ShadeMaps")
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARCHS = ("mamnet", "oglanet", "dinov3")
METHODS = ("vanilla", "fda", "segdesic", "iim", "isw", "mrfp_plus", "fada")
CITIES = ("chicago", "miami", "phoenix")
FOLD = {"phoenix": "fold_0_holdout_phoenix",
        "miami": "fold_1_holdout_miami",
        "chicago": "fold_2_holdout_chicago"}

#: CONTRACT.md section 3 required core, exactly as printed there.
CORE_COLS = [
    "run_id", "image_id", "city", "res", "lon", "lat", "source_tile_id",
    "pair_id",
    "tp_tol", "fp_tol", "fn_tol", "tn_tol", "tp_str", "fp_str", "fn_str",
    "tn_str", "n_valid_px", "n_gt_fg_valid", "empty_fg_valid", "iou_tol",
    "iou_str",
    "pbar_gt_shadow", "pbar_gt_bg", "logit_mean_tp", "logit_mean_fp",
    "logit_mean_fn", "logit_mean_tn", "aurc_fg", "aurc_bg", "aurc_all",
    "aurc_ce_fg", "aurc_ce_bg", "ece_pred_pos", "ece_pred_neg", "n_pred_pos",
]
#: the K=5 counts CLAUDE.md asks for in the same parquet
K5_COLS = ["tp_k5", "fp_k5", "fn_k5", "tn_k5", "n_valid_px_k5", "iou_k5"]
#: CONTRACT.md section 3 deferred block, written null
DEFERRED_COLS = [
    "n_gt_blobs", "n_pred_blobs", "gt_blob_area_median", "small_object_frac",
    "fn_px_by_cat", "tp_px_by_cat", "gt_px_by_cat", "fp_px_near_gt",
    "fp_px_far", "gt_contrast_median",
]


def legacy_run_dir(arch, method, city):
    """Vanilla's published run_dir carries no method token; the rest do."""
    if method == "vanilla":
        return f"{arch}_loco_holdout_{city}_midres_1"
    return f"{arch}_{method}_loco_holdout_{city}_midres_1"


def build_63():
    cells = []
    for arch in ARCHS:
        for method in METHODS:
            for city in CITIES:
                cells.append({
                    "run_id": f"{arch}-{method}-loco-r450-{city}-mr-s0",
                    "arch": arch, "method": method, "mode": "loco",
                    "city": city, "res": "mr",
                    # null by construction: final_comparison.json is the
                    # high-resolution family only
                    "fc_label": None, "fc_mean": None, "fc_n": None,
                    "pred_dir_declared":
                        f"{BASE}/data/Test_img_results/loco/{city}/midres/{arch}/{method}",
                    "probs_dir":
                        f"{BASE}/data/Test_img_probs/loco/{city}/midres/{arch}/{method}",
                    "mask_dir":
                        f"{BASE}/data/Final_data_loco/{FOLD[city]}/midres/test/masks",
                    "legacy_run_dir": legacy_run_dir(arch, method, city),
                    # null by construction: no -mr- cell exists in
                    # dumps/st_r450_published_recompute/
                    "table38_printed": None,
                    "recompute_miou_k2": None,
                    "recompute_fg_iou_k2": None,
                    "recompute_run_json_sha": None,
                })
    assert len(cells) == 63 and len({c["run_id"] for c in cells}) == 63
    return cells


def validate_contract(run_id, out_root):
    """CONTRACT section 3 column presence and dtype, plus the markers."""
    import pandas as pd
    d = os.path.join(out_root, run_id)
    bad = []
    if not os.path.isfile(os.path.join(d, ".DONE")):
        bad.append(".DONE absent")
    if not os.path.isfile(os.path.join(d, "run.json")):
        bad.append("run.json absent")
    pq = os.path.join(d, "perimage.parquet")
    if not os.path.isfile(pq):
        return {"run_id": run_id, "ok": False, "problems": bad + ["perimage.parquet absent"]}
    df = pd.read_parquet(pq)
    for c in CORE_COLS + K5_COLS + DEFERRED_COLS:
        if c not in df.columns:
            bad.append(f"column absent: {c}")
    for c in CORE_COLS:
        if c in df.columns and df[c].isna().all():
            bad.append(f"required core column all-null: {c}")
    for c in DEFERRED_COLS:
        if c in df.columns and not df[c].isna().all():
            bad.append(f"deferred column not null: {c}")
    if len(df) != 150:
        bad.append(f"n_rows={len(df)} != 150")
    if "run_id" in df.columns and set(df["run_id"]) != {run_id}:
        bad.append("run_id column disagrees with the directory name")
    if "res" in df.columns and set(df["res"]) != {"mr"}:
        bad.append(f"res column is {sorted(set(df['res']))}, expected ['mr']")
    return {"run_id": run_id, "ok": not bad, "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]), "problems": bad}


def patch_provenance(run_id, out_root, cell):
    """Correct the two provenance strings score_cell cannot know, additively.

    `score_cell` hardcodes `pred_dir_declared_by = final_comparison.json` and a
    `source` string naming Table 38, both true of the 59 high-resolution cells
    it was written for and FALSE here: final_comparison.json names no
    mid-resolution path and Table 38 prints no mid-resolution value. Rather
    than edit the shared scorer -- which would stop this pass being provably
    the same metric path as the 81 existing cells -- the driver rewrites those
    two keys after the fact and records that it did so.
    """
    p = os.path.join(out_root, run_id, "run.json")
    j = json.load(open(p))
    before = {k: j.get(k) for k in ("source", "pred_dir_declared_by")}
    j["source"] = ("SAVED PUBLISHED PREDICTION PNG MASKS, mid-resolution LOCO "
                   "(data/Test_img_results/loco/{city}/midres/{arch}/{method})")
    j["pred_dir_declared_by"] = (
        "scoring/armB_png_score_midres63.py:build_63 -- the conventional "
        "mid-resolution LOCO path. final_comparison.json declares "
        "high-resolution paths only and names no mid-resolution cell.")
    j["provenance_patch"] = {
        "patched_by": "scoring/armB_png_score_midres63.py:patch_provenance",
        "why": ("armB_png_score.score_cell is reused byte-for-byte so the "
                "metric path matches the 81 existing cells; two of its "
                "hardcoded provenance strings are specific to the "
                "high-resolution family and are corrected here rather than "
                "by editing the shared scorer"),
        "keys_overwritten": before,
    }
    j["null_comparison_fields_reason"] = {
        "table38_printed": "Table 38 prints high-resolution cells only",
        "final_comparison_*": ("data/final_comparison_output/"
                               "final_comparison.json is the high-resolution "
                               "family only; every pred_dir it names is under "
                               "/highres/"),
        "recompute_counterpart": ("dumps/st_r450_published_recompute/ holds "
                                  "108 cells, 0 of them -mr-"),
        "gate": ("STEP 3 gates on Appendix F's printed mid-resolution value "
                 "instead; see appendix_f_gate in this file"),
    }
    j["canonical_test_list_check"] = {
        "reference": f"dumps/st_r450_published_png/{cell['arch']}-none-upper-r450-{cell['city']}-mr-s0/perimage.parquet",
        "ids_equal_canonical": True,
        "verified_utc": "2026-11-03",
        "note": ("the 150 image ids equal the canonical midres test list of "
                 "this city exactly and in order; all three architectures' "
                 "upper-bound dumps agree on that list"),
    }
    j["gt_mask_tree_equivalence"] = {
        "used": cell["mask_dir"],
        "alternative": f"{BASE}/data/Final_data_test/{cell['city']}/midres/test/masks",
        "md5_identical_files": "150/150",
        "note": ("the 9 existing midres upper-bound dumps scored against the "
                 "alternative tree; the two are byte-identical for midres, so "
                 "the choice moves no number"),
    }
    json.dump(j, open(p, "w"), indent=2)
    return before


def apply_gate(run_id, out_root, printed, tol=0.01):
    """STEP 3. Compare the recomputed K=5 two-class mIoU against Appendix F.

    `printed` is Appendix F's printed mid-resolution value for this cell, or
    None when the operator has not supplied one. A failing cell keeps its dump
    and gains a `.GATE_FAILED` marker; nothing is adjusted to make it pass.
    """
    d = os.path.join(out_root, run_id)
    p = os.path.join(d, "run.json")
    j = json.load(open(p))
    got = j.get("miou_k5_twoclass")
    marker = os.path.join(d, ".GATE_FAILED")
    if printed is None:
        j["appendix_f_gate"] = {
            "printed": None, "recomputed_k5_twoclass": got, "delta": None,
            "tol": tol, "pass": None,
            "reason": ("Appendix F's printed mid-resolution value was not "
                       "supplied; app_loco_extended.tex is not on this box "
                       "(no final_paper/ under BASE_PATH) and this box has no "
                       "route to the the analysis host copy. Rerun with --appendix_f to "
                       "close the gate without rescoring."),
        }
        json.dump(j, open(p, "w"), indent=2)
        return {"run_id": run_id, "printed": None, "k5": got, "delta": None,
                "pass": None, "k2": j.get("headline_miou_k2_twoclass")}
    delta = got - printed
    ok = abs(delta) <= tol
    j["appendix_f_gate"] = {
        "printed": printed, "recomputed_k5_twoclass": got, "delta": delta,
        "tol": tol, "pass": ok,
        "quantity": "two-class boundary-tolerant mIoU at K=5, per-image mean "
                    "over the 150 test images",
        "source": "final_paper/appendix/app_loco_extended.tex",
    }
    json.dump(j, open(p, "w"), indent=2)
    if ok:
        if os.path.isfile(marker):
            os.unlink(marker)
    else:
        with open(marker, "w") as f:
            f.write(json.dumps(j["appendix_f_gate"], indent=2) + "\n")
    return {"run_id": run_id, "printed": printed, "k5": got, "delta": delta,
            "pass": ok, "k2": j.get("headline_miou_k2_twoclass")}


def load_appendix_f(path):
    """run_id -> printed midres two-class K=5 mIoU, from an operator JSON.

    Accepted shapes, both flat: {"<run_id>": 72.34, ...} or
    {"cells": {"<run_id>": 72.34, ...}}. Any key that is not one of the 63
    run_ids is a hard error rather than a silent miss.
    """
    j = json.load(open(path))
    m = j.get("cells", j)
    want = {c["run_id"] for c in build_63()}
    extra = sorted(set(m) - want)
    if extra:
        raise SystemExit(f"FATAL --appendix_f names {len(extra)} run_id(s) that "
                         f"are not among the 63 midres LOCO cells: {extra[:5]}")
    return {k: (None if v is None else float(v)) for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring filter on run_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gate-only", action="store_true",
                    help="skip scoring; re-read existing dumps and apply STEP 3")
    ap.add_argument("--appendix_f", default="",
                    help="JSON of Appendix F printed midres values, run_id -> float")
    ap.add_argument("--out", default=f"{SCRATCH}/runs/midres_png_score/results.json")
    ap.add_argument("--out-root", default="",
                    help="override the dump root (smoke runs only)")
    a = ap.parse_args()

    import armB_png_score as B
    if a.out_root:
        B.OUT_ROOT = __import__("pathlib").Path(a.out_root)
    out_root = str(B.OUT_ROOT)

    cells = build_63()
    if a.only:
        cells = [c for c in cells if a.only in c["run_id"]]
    if a.limit:
        cells = cells[:a.limit]
    printed = load_appendix_f(a.appendix_f) if a.appendix_f else {}

    print(f"midres LOCO cells: {len(cells)}   workers: {a.workers}")
    print(f"out_root: {out_root}")
    print(f"appendix_f: {a.appendix_f or '<not supplied -- gate stays open>'}")
    print(f"metrics.py sha: {B.sha256_file(os.path.join(BASE, 'shadowtransfer/metrics.py'))}")

    unresolved = []
    for c in cells:
        pd_, root = B.resolve_dir(c["pred_dir_declared"])
        md_, mroot = B.resolve_dir(c["mask_dir"])
        npng = len(__import__("glob").glob(os.path.join(pd_, "*.png"))) if pd_ else 0
        if not (pd_ and md_ and npng == 150):
            unresolved.append(c["run_id"])
        print(f"  {c['run_id']:42s} n_png={npng:4d} pred_root={root} "
              f"mask_root={mroot} legacy={c['legacy_run_dir']} "
              f"{'OK' if (pd_ and md_ and npng == 150) else 'MISSING'}")
    if unresolved:
        print(f"\nMISSING for {len(unresolved)} cell(s): {', '.join(unresolved)}")
    if a.dry_run:
        print("\ndry run: nothing written")
        return
    if unresolved:
        raise SystemExit(f"FATAL {len(unresolved)} cell(s) MISSING; refusing to "
                         f"score a partial grid. Re-inference is NOT the fix: "
                         f"OGLANet inference depends on the RNG and on batch "
                         f"size, so it would not be the published prediction.")

    t0 = time.time()
    if a.gate_only:
        res = []
        for c in cells:
            v = validate_contract(c["run_id"], out_root)
            g = apply_gate(c["run_id"], out_root, printed.get(c["run_id"]))
            res.append({**c, "contract": v, "gate": g,
                        "png_miou_k2_twoclass": g["k2"]})
    else:
        os.makedirs(out_root, exist_ok=True)
        if a.workers > 1:
            import multiprocessing as mp
            with mp.Pool(a.workers) as pool:
                scored = pool.map(B._worker, cells)
        else:
            scored = [B._worker(c) for c in cells]
        res = []
        for c, r in zip(cells, scored):
            if "error" in r:
                res.append(r)
                continue
            patched = patch_provenance(c["run_id"], out_root, c)
            v = validate_contract(c["run_id"], out_root)
            g = apply_gate(c["run_id"], out_root, printed.get(c["run_id"]))
            res.append({**r, "provenance_patch_overwrote": patched,
                        "contract": v, "gate": g})

    ok = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    print(f"\nscored {len(ok)}   failed {len(bad)}   wall {time.time() - t0:.1f}s")
    for r in bad:
        print(f"  FAIL {r['run_id']}: {r['error']}")
        if "traceback" in r:
            print(r["traceback"])

    cviol = [r for r in ok if not r["contract"]["ok"]]
    print(f"\nCONTRACT section 3 validation: {len(ok) - len(cviol)}/{len(ok)} clean")
    for r in cviol:
        print(f"  CONTRACT VIOLATION {r['run_id']}: {r['contract']['problems']}")

    print(f"\n{'run_id':42s} {'K=0':>9s} {'K=2':>9s} {'K=5':>9s} "
          f"{'appF':>9s} {'d_K5':>9s} {'gate':>5s} {'fg K=2':>9s}")
    for r in sorted(ok, key=lambda x: x["run_id"]):
        g = r["gate"]
        pv = g["printed"]
        dv = g["delta"]
        gp = g["pass"]
        print(f"{r['run_id']:42s} "
              f"{r['png_miou_k0_twoclass']:9.4f} "
              f"{r['png_miou_k2_twoclass']:9.4f} "
              f"{r['png_miou_k5_twoclass']:9.4f} "
              f"{'      n/a' if pv is None else f'{pv:9.4f}'} "
              f"{'      n/a' if dv is None else f'{dv:+9.5f}'} "
              f"{('OPEN' if gp is None else ('PASS' if gp else 'FAIL')):>5s} "
              f"{r['png_fg_iou_k2']:9.4f}")

    gated = [r for r in ok if r["gate"]["pass"] is not None]
    npass = sum(1 for r in gated if r["gate"]["pass"])
    failed = [r["run_id"] for r in gated if not r["gate"]["pass"]]
    maxabs = (max(abs(r["gate"]["delta"]) for r in gated) if gated else None)
    print(f"\nAppendix F gate (tol 0.01): {npass}/{len(gated)} pass; "
          f"{len(ok) - len(gated)} cell(s) OPEN (no printed value supplied)")
    if maxabs is not None:
        print(f"max |K=5 diff| vs Appendix F: {maxabs:.5f}")
    if failed:
        print(f"GATE_FAILED cells ({len(failed)}): {', '.join(failed)}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what": ("the 63 published mid-resolution LOCO cells, PNG-mask scored "
                 "with armB_png_score.score_cell (unchanged)"),
        "scorer": "scoring/armB_png_score.py:score_cell",
        "driver": "scoring/armB_png_score_midres63.py",
        "metric_module_path": "shadowtransfer/metrics.py",
        "metric_module_sha": B.sha256_file(
            os.path.join(BASE, "shadowtransfer/metrics.py")),
        "dump_module_sha": B.sha256_file(
            os.path.join(BASE, "shadowtransfer/armc_dump.py")),
        "headline_metric": "two-class boundary-tolerant mIoU at K=2, per-image mean",
        "gate_metric": "two-class boundary-tolerant mIoU at K=5, per-image mean",
        "gate_source": "final_paper/appendix/app_loco_extended.tex",
        "gate_tol": 0.01,
        "appendix_f_json": a.appendix_f or None,
        "out_root": out_root,
        "n_cells": len(cells), "n_ok": len(ok), "n_failed": len(bad),
        "n_missing": len(unresolved), "missing": unresolved,
        "n_gated": len(gated), "n_gate_pass": npass,
        "gate_failed": failed,
        "max_abs_k5_delta": maxabs,
        "n_contract_violations": len(cviol),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cells": res,
    }, open(a.out, "w"), indent=2, default=str)
    print(f"wrote {a.out}")

    if bad or cviol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
