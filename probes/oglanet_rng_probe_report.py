#!/usr/bin/env python3
"""Step-3 table for the OGLANet inference-RNG probe.

Assembles the per-pass numbers each probe job already computed, and adds the
only cross-job quantities that cannot be computed inside one job: bit-identity
and per-pixel max |diff| of every pass against the C1a pass-0 reference for the
same checkpoint. delta_stats is imported from oglanet_probsource, the same
function the earlier finding-4 work used, so the 0.80-0.88 swing it reports is
measured the same way.

This is a determinism check -- the exception CLAUDE.md names to the
no-aggregation rule -- not an analysis. It prints facts and the pre-stated
comparison values; it draws no conclusion.

  python oglanet_rng_probe_report.py --out_root ${ST_SCRATCH_ROOT}/oglanet_rng_probe
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BASE = os.environ.get("BASE_PATH") or os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONDS = ["C1a", "C1b", "C2s0", "C2s1", "C3", "C4"]
CKPTS = {
    "A": {"label": "SIB oglanet-sibc4-loco-r450-chicago-hr-s42-n450 "
                   "(batch 8, fp16)",
          "stored_miou": 68.87062690608916,
          "stored_fg": 43.19055777026932,
          "stored_source": "the cell's own test_results.json, written in "
                           "process at training time",
          "reentry_miou": 68.531270, "reentry_fg": 42.598234,
          "reentry_source": "the five 2026-10-31 epochs-0 reps, five nodes"},
    "B": {"label": "published vanilla LOCO chicago highres (batch 4, fp32)",
          "stored_miou": None, "stored_fg": None, "stored_source": None,
          "reentry_miou": None, "reentry_fg": None,
          "reentry_source": "erratum E-9 nine-pass span 1.1158 mIoU; "
                            "results/21 finding 4 per-pixel swing 0.80-0.88, "
                            "0 of 150 images identical in any pass"},
}


def load_npz(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--report_dir", default=None)
    a = ap.parse_args()
    from oglanet_probsource import delta_stats

    root = os.path.abspath(a.out_root)
    rep_dir = a.report_dir or os.path.join(root, "report")
    os.makedirs(rep_dir, exist_ok=True)

    cells, missing = {}, []
    for ck in CKPTS:
        for c in CONDS:
            cid = f"{ck}-{c}"
            pj = os.path.join(root, cid, "probe.json")
            if not (os.path.exists(os.path.join(root, cid, ".DONE"))
                    and os.path.exists(pj)):
                missing.append(cid)
                continue
            cells[cid] = json.load(open(pj))

    out = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "out_root": root, "missing_cells": missing,
           "pre_stated_comparisons": {k: {kk: vv for kk, vv in v.items()}
                                      for k, v in CKPTS.items()},
           "tables": {}, "facts": {}}

    for ck, meta in CKPTS.items():
        ref_id = f"{ck}-C1a"
        ref = None
        if ref_id in cells:
            ref = load_npz(os.path.join(root, ref_id, "pass0", "probs.npz"))
        rows = []
        for c in CONDS:
            cid = f"{ck}-{c}"
            if cid not in cells:
                continue
            rec = cells[cid]
            for p in rec["passes"]:
                k = p["pass"]
                d = None
                if ref is not None:
                    probs = load_npz(os.path.join(root, cid, f"pass{k}",
                                                  "probs.npz"))
                    d = delta_stats(ref, probs)
                rows.append({
                    "condition": c, "cell": cid,
                    "job": p.get("slurm_job_id"), "node": p.get("node"),
                    "seed": rec["seed"], "pass": k,
                    "miou_k2_twoclass_metricmod": p["miou_k2_twoclass_pct"],
                    "fg_iou_k2_metricmod": p["fg_iou_k2_pct"],
                    "miou_k2_twoclass_trainer":
                        p.get("trainer_miou_k2_twoclass"),
                    "fg_iou_k2_trainer": p.get("trainer_fg_iou_k2"),
                    "torch_initial_seed": p["torch_initial_seed"],
                    "n_identical_vs_C1a_p0":
                        (None if d is None else d["n_images_bit_identical"]),
                    "max_abs_dp_vs_C1a_p0":
                        (None if d is None else d["max_over_images"]),
                    "mean_abs_dp_vs_C1a_p0":
                        (None if d is None else d["mean_over_images"]),
                    "rng_sha_cpu_before": p["rng_sha_before"]["cpu"][:16],
                    "rng_sha_cpu_after": p["rng_sha_after"]["cpu"][:16],
                    "rng_sha_cuda_before": (p["rng_sha_before"]["cuda"] or "")[:16],
                    "rng_sha_cuda_after": (p["rng_sha_after"]["cuda"] or "")[:16],
                    "rng_advanced": p["rng_advanced_during_pass"],
                    "wall_s": p["wall_s"], "n_images": p["n_images"],
                })
        out["tables"][ck] = rows
        out["facts"][ck] = facts_for(ck, meta, rows)

    with open(os.path.join(rep_dir, "report.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    md = render(out)
    with open(os.path.join(rep_dir, "report.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"\n[report] {rep_dir}/report.json and report.md")
    if missing:
        print(f"[report] MISSING (no .DONE): {missing}")
    return 0


def _g(rows, cond, p=0, key="miou_k2_twoclass_metricmod"):
    for r in rows:
        if r["condition"] == cond and r["pass"] == p:
            return r.get(key)
    return None


def facts_for(ck, meta, rows):
    """The raw answers to Q1-Q6. Numbers and row references only."""
    f = {}
    # Q1  fixed seed across jobs/nodes: C1a vs C1b, both seed 42, 1 pass
    a, b = _g(rows, "C1a"), _g(rows, "C1b")
    fa, fb = (_g(rows, "C1a", key="fg_iou_k2_metricmod"),
              _g(rows, "C1b", key="fg_iou_k2_metricmod"))
    nid = _g(rows, "C1b", key="n_identical_vs_C1a_p0")
    dmx = _g(rows, "C1b", key="max_abs_dp_vs_C1a_p0")
    f["Q1"] = {"rows": ["C1a p0", "C1b p0"],
               "nodes": [_g(rows, "C1a", key="node"),
                         _g(rows, "C1b", key="node")],
               "miou": [a, b],
               "miou_delta": (None if None in (a, b) else b - a),
               "fg": [fa, fb],
               "fg_delta": (None if None in (fa, fb) else fb - fa),
               "n_images_bit_identical_of_150": nid,
               "max_abs_dp": dmx,
               "bit_identical": (nid == 150 and dmx == 0.0)
               if nid is not None else None}
    # Q2  changing the seed: C1 (42) vs C2 (0, 1)
    s0, s1 = _g(rows, "C2s0"), _g(rows, "C2s1")
    vals = [v for v in (a, s0, s1) if v is not None]
    f["Q2"] = {"rows": ["C1a p0 seed42", "C2s0 p0 seed0", "C2s1 p0 seed1"],
               "miou": {"seed42": a, "seed0": s0, "seed1": s1},
               "fg": {"seed42": fa,
                      "seed0": _g(rows, "C2s0", key="fg_iou_k2_metricmod"),
                      "seed1": _g(rows, "C2s1", key="fg_iou_k2_metricmod")},
               "miou_span_over_three_seeds":
                   (None if len(vals) < 2 else max(vals) - min(vals)),
               "n_identical_vs_C1a_p0":
                   {"seed0": _g(rows, "C2s0", key="n_identical_vs_C1a_p0"),
                    "seed1": _g(rows, "C2s1", key="n_identical_vs_C1a_p0")},
               "max_abs_dp_vs_C1a_p0":
                   {"seed0": _g(rows, "C2s0", key="max_abs_dp_vs_C1a_p0"),
                    "seed1": _g(rows, "C2s1", key="max_abs_dp_vs_C1a_p0")}}
    # Q3  two consecutive passes in one process: C3 (seeded), C4 (unseeded)
    f["Q3"] = {}
    for c in ("C3", "C4"):
        p0, p1 = _g(rows, c, 0), _g(rows, c, 1)
        r1 = [r for r in rows if r["condition"] == c and r["pass"] == 1]
        f["Q3"][c] = {
            "miou_pass0": p0, "miou_pass1": p1,
            "miou_delta": (None if None in (p0, p1) else p1 - p0),
            "fg_pass0": _g(rows, c, 0, "fg_iou_k2_metricmod"),
            "fg_pass1": _g(rows, c, 1, "fg_iou_k2_metricmod"),
            "pass1_vs_C1a_p0_identical":
                (r1[0]["n_identical_vs_C1a_p0"] if r1 else None),
            "pass1_vs_C1a_p0_max_dp":
                (r1[0]["max_abs_dp_vs_C1a_p0"] if r1 else None),
            "torch_initial_seed": _g(rows, c, 0, "torch_initial_seed")}
    # Q4  does the RNG state advance across a pass
    f["Q4"] = [{"cell": r["cell"], "pass": r["pass"],
                "cpu_before": r["rng_sha_cpu_before"],
                "cpu_after": r["rng_sha_cpu_after"],
                "cpu_changed": r["rng_sha_cpu_before"] != r["rng_sha_cpu_after"],
                "cuda_before": r["rng_sha_cuda_before"],
                "cuda_after": r["rng_sha_cuda_after"],
                "cuda_changed":
                    r["rng_sha_cuda_before"] != r["rng_sha_cuda_after"]}
               for r in rows]
    # Q5 / Q6  against the pre-stated values
    if meta["stored_miou"] is not None:
        cand = [(r["condition"], r["pass"], r["miou_k2_twoclass_trainer"])
                for r in rows if r["miou_k2_twoclass_trainer"] is not None]
        hit = [c for c in cand if abs(c[2] - meta["stored_miou"]) < 1e-6]
        best = (min(cand, key=lambda c: abs(c[2] - meta["stored_miou"]))
                if cand else None)
        f["Q5"] = {"stored_miou": meta["stored_miou"],
                   "stored_fg": meta["stored_fg"],
                   "stored_source": meta["stored_source"],
                   "reproduced_exactly_by": hit,
                   "closest": best,
                   "closest_offset": (None if best is None
                                      else best[2] - meta["stored_miou"]),
                   "reentry_value_2026_10_31": meta["reentry_miou"],
                   "trainer_values_all": cand}
    else:
        vals = [r["miou_k2_twoclass_metricmod"] for r in rows
                if r["condition"] in ("C2s0", "C2s1", "C3", "C4")]
        dps = [r["max_abs_dp_vs_C1a_p0"] for r in rows
               if r["condition"] in ("C2s0", "C2s1", "C3", "C4")
               and r["max_abs_dp_vs_C1a_p0"] is not None]
        f["Q6"] = {"miou_values_C2_C3_C4": vals,
                   "miou_span_C2_C3_C4":
                       (None if len(vals) < 2 else max(vals) - min(vals)),
                   "E9_nine_pass_miou_span_for_comparison": 1.1158,
                   "max_abs_dp_values": dps,
                   "max_abs_dp_over_C2_C3_C4":
                       (None if not dps else max(dps)),
                   "finding4_per_pixel_swing_for_comparison": "0.80-0.88"}
    return f


def _f(v, n=6):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


COLS = [("condition", "cond"), ("job", "job"), ("pass", "pass"),
        ("node", "node"), ("seed", "seed"),
        ("miou_k2_twoclass_metricmod", "mIoU K=2 2cls (metric mod)"),
        ("fg_iou_k2_metricmod", "fg IoU K=2 (metric mod)"),
        ("miou_k2_twoclass_trainer", "mIoU K=2 2cls (trainer)"),
        ("fg_iou_k2_trainer", "fg IoU K=2 (trainer)"),
        ("n_identical_vs_C1a_p0", "ident/150 vs C1a p0"),
        ("max_abs_dp_vs_C1a_p0", "max |dp| vs C1a p0"),
        ("rng_sha_cpu_before", "cpu rng sha before"),
        ("rng_sha_cpu_after", "cpu rng sha after")]


def render(out):
    L = [f"# OGLANet inference-RNG probe -- step 3",
         f"",
         f"out_root `{out['out_root']}`  written {out['written_utc']}  ",
         f"missing cells: {out['missing_cells'] or 'none'}", ""]
    for ck, meta in CKPTS.items():
        rows = out["tables"].get(ck, [])
        L += [f"## Checkpoint {ck} -- {meta['label']}", ""]
        if not rows:
            L += ["_no completed cells_", ""]
            continue
        L.append("| " + " | ".join(h for _, h in COLS) + " |")
        L.append("|" + "|".join("---" for _ in COLS) + "|")
        for r in rows:
            L.append("| " + " | ".join(_f(r.get(k)) for k, _ in COLS) + " |")
        L.append("")
        f = out["facts"][ck]
        L += [f"### Answers, checkpoint {ck}", ""]
        q1 = f["Q1"]
        L += [f"**Q1 fixed seed across jobs and nodes (C1a vs C1b, both seed "
              f"42).** nodes {q1['nodes']}; two-class mIoU "
              f"{_f(q1['miou'][0])} vs {_f(q1['miou'][1])} "
              f"(delta {_f(q1['miou_delta'])}); fg IoU {_f(q1['fg'][0])} vs "
              f"{_f(q1['fg'][1])} (delta {_f(q1['fg_delta'])}); "
              f"{_f(q1['n_images_bit_identical_of_150'])}/150 images "
              f"bit-identical; max |dp| {_f(q1['max_abs_dp'])}. "
              f"Bit-identical: **{_f(q1['bit_identical'])}**.", ""]
        q2 = f["Q2"]
        L += [f"**Q2 changing the seed (C1 seed 42 vs C2 seed 0 / seed 1).** "
              f"two-class mIoU seed42 {_f(q2['miou']['seed42'])}, seed0 "
              f"{_f(q2['miou']['seed0'])}, seed1 {_f(q2['miou']['seed1'])}; "
              f"span {_f(q2['miou_span_over_three_seeds'])}. "
              f"fg IoU seed42 {_f(q2['fg']['seed42'])}, seed0 "
              f"{_f(q2['fg']['seed0'])}, seed1 {_f(q2['fg']['seed1'])}. "
              f"bit-identical to C1a p0: seed0 "
              f"{_f(q2['n_identical_vs_C1a_p0']['seed0'])}/150, seed1 "
              f"{_f(q2['n_identical_vs_C1a_p0']['seed1'])}/150; max |dp| "
              f"seed0 {_f(q2['max_abs_dp_vs_C1a_p0']['seed0'])}, seed1 "
              f"{_f(q2['max_abs_dp_vs_C1a_p0']['seed1'])}.", ""]
        for c in ("C3", "C4"):
            q = f["Q3"][c]
            L += [f"**Q3 {c} -- two consecutive passes in one process"
                  f"{' (seed 42)' if c == 'C3' else ' (unseeded)'}.** "
                  f"two-class mIoU pass0 {_f(q['miou_pass0'])}, pass1 "
                  f"{_f(q['miou_pass1'])} (delta {_f(q['miou_delta'])}); "
                  f"fg IoU pass0 {_f(q['fg_pass0'])}, pass1 "
                  f"{_f(q['fg_pass1'])}; torch.initial_seed() "
                  f"{q['torch_initial_seed']}; pass1 vs C1a p0 "
                  f"{_f(q['pass1_vs_C1a_p0_identical'])}/150 identical, "
                  f"max |dp| {_f(q['pass1_vs_C1a_p0_max_dp'])}.", ""]
        n_cpu = sum(1 for r in f["Q4"] if r["cpu_changed"])
        n_cuda = sum(1 for r in f["Q4"] if r["cuda_changed"])
        L += [f"**Q4 does the RNG state advance during a pass.** CPU "
              f"generator sha changed in {n_cpu}/{len(f['Q4'])} passes; CUDA "
              f"generator sha changed in {n_cuda}/{len(f['Q4'])}. Per-pass "
              f"shas are in the table and in report.json.", ""]
        if "Q5" in f:
            q5 = f["Q5"]
            L += [f"**Q5 does any condition reproduce the stored "
                  f"training-time value {_f(q5['stored_miou'])}?** "
                  f"({q5['stored_source']}) exact matches: "
                  f"{q5['reproduced_exactly_by'] or 'NONE'}. closest: "
                  f"{q5['closest']}, offset {_f(q5['closest_offset'])}. "
                  f"The 2026-10-31 five-node epochs-0 re-entry value was "
                  f"{_f(q5['reentry_value_2026_10_31'])}.", ""]
        if "Q6" in f:
            q6 = f["Q6"]
            L += [f"**Q6 spread across C2/C3/C4 vs the earlier finding.** "
                  f"two-class mIoU values {[round(v, 6) for v in q6['miou_values_C2_C3_C4']]}, "
                  f"span {_f(q6['miou_span_C2_C3_C4'])} "
                  f"(E-9 nine-pass span for comparison: "
                  f"{q6['E9_nine_pass_miou_span_for_comparison']}). "
                  f"max per-pixel |dp| vs C1a p0 over C2/C3/C4: "
                  f"{_f(q6['max_abs_dp_over_C2_C3_C4'])} "
                  f"(finding 4 for comparison: "
                  f"{q6['finding4_per_pixel_swing_for_comparison']}).", ""]
    L += ["---", "",
          "Facts only. The reading (R-a / R-b / R-c) is stated in the session "
          "log, not here."]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
