#!/usr/bin/env python
"""
OGLANet inference-RNG probe of 2026-11-01, promoted into results/.

Answers E-9, AC2, and the weakness ids results/21 carries (E-7, AC1, BxRx-W4).

results/21 finding 4 (ten unseeded passes of run_inference_probs.py, never
bit-identical) and the 2026-10-31 diagnosis (five seeded SIB passes,
bit-identical, 0.339357 two-class mIoU below the stored training-time value)
conflicted. The Delta probe of 2026-11-01 ran a seed x pass grid on two OGLANet
checkpoints and read R-a (RNG-dependent). This job copies that probe into
results/30_oglanet_rng_dependence.json so the paper can cite it. NN 30 is the
next free results number after 29.

Nothing is recomputed. Every score, delta and sha is copied from the probe's
own files under delta_runs/oglanet_rng_probe/. The only arithmetic is the two
cross-checks the brief allows:
  (a) per-image count identity between two passes, from perimage_probe.parquet,
      against the log's 150/150 and 0/150 claims. The parquet holds pixel
      counts, not probabilities (probs.npz was not copied), so count identity
      is a NECESSARY condition for bit identity: it can refute a 150/150 claim
      and cannot refute a 0/150 claim.
  (b) min / max / range over passes of the stored two-class mIoU and fg IoU.
Every cross-check is asserted and its pass/fail recorded.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/30_oglanet_rng_dependence.py
"""

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _promote_lib import (BASE, env_block, fail, load_json,  # noqa: E402
                          read_text, source, write_results)

DRY_RUN = "--dry-run" in sys.argv
NAME = "30_oglanet_rng_dependence"

PROBE = "delta_runs/oglanet_rng_probe"
REPORT = PROBE + "/report/report.json"
REPORT_MD = PROBE + "/report/report.md"
LOG = "delta_logs/session_logs/2026-11-01-oglanet-rng-probe.md"
R21 = "results/21_disclosure_record.json"
R27 = "results/27_internal_sib_worst_case_n450_sar.json"

CKPTS = ("A", "B")
CELLS = ("C1a", "C1b", "C2s0", "C2s1", "C3", "C4")
COND_LABEL = {"C1a": "C1a", "C1b": "C1b", "C2s0": "C2 seed 0",
              "C2s1": "C2 seed 1", "C3": "C3", "C4": "C4"}
COUNT_COLS = ["tp_tol", "fp_tol", "fn_tol", "tn_tol",
              "tp_str", "fp_str", "fn_str", "tn_str",
              "n_valid_px", "n_gt_fg_valid"]
# Output-path guard: set ST_FORBIDDEN_OUT_PREFIX to a read-only source tree that results must never land under.
FORBIDDEN = os.environ.get("ST_FORBIDDEN_OUT_PREFIX", "")


def r6(x):
    return None if x is None else round(float(x), 6)


def section(text, start, end):
    i = text.find(start)
    j = text.find(end, i + 1)
    if i < 0 or j < 0:
        fail(f"log section not found: {start!r} .. {end!r}")
    return text[i:j]


def log_line_of(text, needle):
    i = text.find(needle)
    if i < 0:
        fail(f"citation not in log: {needle!r}")
    return text.count("\n", 0, i) + 1


def parse_log_table(sec):
    """Rows of a markdown table: dict cond/pass -> list of cells."""
    rows = {}
    for ln in sec.splitlines():
        m = re.match(r"\|\s*(C1a|C1b|C2s0|C2s1|C3|C4)\s*\|", ln)
        if m:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows[(cells[0], int(cells[2]))] = cells
    return rows


def main():
    out_path = os.path.realpath(os.path.join(BASE, "results", NAME + ".json"))
    if FORBIDDEN and out_path.startswith(FORBIDDEN):
        fail(f"output path {out_path} is under the forbidden tree")

    log = read_text(LOG)
    rep = load_json(REPORT)
    if rep["missing_cells"]:
        fail(f"report lists missing cells {rep['missing_cells']}")

    # ---- per-pass rows: report.json tables, verified against probe.json,
    #      summary.json, test_results.json and the log's printed tables ----
    log_tab = {
        "A": parse_log_table(section(log, "### Checkpoint A", "### Checkpoint B")),
        "B": parse_log_table(section(log, "### Checkpoint B", "## 6. Q1-Q6")),
    }
    probes, rows, copy_checks = {}, {}, []
    ckpt_info = {}
    for ck in CKPTS:
        rep_rows = {(r["condition"], r["pass"]): r for r in rep["tables"][ck]}
        if len(rep_rows) != 8:
            fail(f"{ck}: report table has {len(rep_rows)} rows, expected 8")
        for cell in CELLS:
            pj = load_json(f"{PROBE}/{ck}-{cell}/probe.json")
            probes[(ck, cell)] = pj
            if pj["condition"] != f"{ck}-{cell}":
                fail(f"{ck}-{cell}: probe.json condition {pj['condition']}")
            if pj["n_passes_recorded"] != len(pj["passes"]):
                fail(f"{ck}-{cell}: n_passes_recorded mismatch")
            ckpt_info.setdefault(ck, []).append(
                dict(cell=f"{ck}-{cell}", job=pj["passes"][0]["slurm_job_id"],
                     written_utc=pj["written_utc"], **pj["checkpoint"]))
            for p in pj["passes"]:
                k = (cell, p["pass"])
                if k not in rep_rows:
                    fail(f"{ck}-{cell} pass {p['pass']} absent from report")
                rr = rep_rows[k]
                sm = load_json(f"{PROBE}/{ck}-{cell}/pass{p['pass']}/summary.json")
                # report.json == probe.json == summary.json, exactly
                pairs = [
                    ("miou_k2_twoclass_metricmod", "miou_k2_twoclass_pct"),
                    ("fg_iou_k2_metricmod", "fg_iou_k2_pct"),
                    ("miou_k2_twoclass_trainer", "trainer_miou_k2_twoclass"),
                    ("fg_iou_k2_trainer", "trainer_fg_iou_k2"),
                    ("torch_initial_seed", "torch_initial_seed"),
                ]
                for rk, pk in pairs:
                    if rr[rk] != p.get(pk) or sm.get(pk) != p.get(pk):
                        fail(f"{ck}-{cell} p{p['pass']} {rk}: report "
                             f"{rr[rk]} probe {p.get(pk)} summary {sm.get(pk)}")
                if (rr["job"] != p["slurm_job_id"] or rr["node"] != p["node"]
                        or rr["seed"] != pj["seed"]):
                    fail(f"{ck}-{cell} p{p['pass']}: job/node/seed mismatch")
                for dev in ("cpu", "cuda"):
                    if not (p["rng_sha_before"][dev].startswith(rr[f"rng_sha_{dev}_before"])
                            and p["rng_sha_after"][dev].startswith(rr[f"rng_sha_{dev}_after"])):
                        fail(f"{ck}-{cell} p{p['pass']}: {dev} rng sha mismatch")
                # trainer field == the pass's own test_results.json (A only)
                tr_src = None
                if ck == "A":
                    tr_src = f"{PROBE}/{ck}-{cell}/pass{p['pass']}/test_results.json"
                    tr = load_json(tr_src)["tolerant_2px"]
                    if (tr["mIOU"] != p["trainer_miou_k2_twoclass"]
                            or tr["Shadow_IOU"] != p["trainer_fg_iou_k2"]):
                        fail(f"{ck}-{cell} p{p['pass']}: test_results.json "
                             f"tolerant_2px differs from trainer field")
                else:
                    if p.get("trainer_miou_k2_twoclass") is not None:
                        fail(f"B-{cell}: unexpected trainer field")
                # log printed table == report rounded to 6 dp
                lt = log_tab[ck].get(k)
                if lt is None:
                    fail(f"log table {ck} lacks {k}")
                if ck == "A":
                    log_vals = dict(job=lt[1], node=lt[3], miou_tr=lt[5],
                                    fg_tr=lt[6], miou_mm=lt[7], ident=lt[8],
                                    maxdp=lt[9])
                    mine = dict(job=rr["job"], node=rr["node"],
                                miou_tr=f"{rr['miou_k2_twoclass_trainer']:.6f}",
                                fg_tr=f"{rr['fg_iou_k2_trainer']:.6f}",
                                miou_mm=f"{rr['miou_k2_twoclass_metricmod']:.6f}",
                                ident=str(rr["n_identical_vs_C1a_p0"]),
                                maxdp=f"{rr['max_abs_dp_vs_C1a_p0']:.6f}")
                else:
                    log_vals = dict(job=lt[1], node=lt[3], miou_mm=lt[5],
                                    fg_mm=lt[6], ident=lt[7], maxdp=lt[8])
                    mine = dict(job=rr["job"], node=rr["node"],
                                miou_mm=f"{rr['miou_k2_twoclass_metricmod']:.6f}",
                                fg_mm=f"{rr['fg_iou_k2_metricmod']:.6f}",
                                ident=str(rr["n_identical_vs_C1a_p0"]),
                                maxdp=f"{rr['max_abs_dp_vs_C1a_p0']:.6f}")
                if log_vals != mine:
                    fail(f"log table {ck} {k}: {log_vals} != {mine}")
                copy_checks.append(f"{ck}-{cell} p{p['pass']}")

                cond = COND_LABEL[cell]
                if cell in ("C3", "C4"):
                    cond = f"{cell} pass {p['pass']}"
                rows[(ck, cell, p["pass"])] = dict(
                    row_id=f"{ck}-{cell}-p{p['pass']}",
                    checkpoint=ck,
                    condition=cond,
                    cell=f"{ck}-{cell}",
                    pass_in_process=p["pass"],
                    seed_set_before_main=pj["seed"],
                    torch_initial_seed=p["torch_initial_seed"],
                    slurm_job_id=p["slurm_job_id"],
                    node=p["node"],
                    gpu_model=p["gpu_model"],
                    miou_k2_twoclass_pct__trainer_field=p.get("trainer_miou_k2_twoclass"),
                    fg_iou_k2_pct__trainer_field=p.get("trainer_fg_iou_k2"),
                    miou_k2_twoclass_pct__metric_module_field=p["miou_k2_twoclass_pct"],
                    fg_iou_k2_pct__metric_module_field=p["fg_iou_k2_pct"],
                    n_images=p["n_images"],
                    n_images_bit_identical_to_C1a_pass0_of_150=rr["n_identical_vs_C1a_p0"],
                    max_abs_dp_vs_C1a_pass0=rr["max_abs_dp_vs_C1a_p0"],
                    mean_abs_dp_vs_C1a_pass0=rr["mean_abs_dp_vs_C1a_p0"],
                    rng_sha256_cpu_before=p["rng_sha_before"]["cpu"],
                    rng_sha256_cpu_after=p["rng_sha_after"]["cpu"],
                    rng_sha256_cuda_before=p["rng_sha_before"]["cuda"],
                    rng_sha256_cuda_after=p["rng_sha_after"]["cuda"],
                    rng_advanced_during_pass=p["rng_advanced_during_pass"],
                    determinism=p["determinism"],
                    sources=[f"{PROBE}/{ck}-{cell}/probe.json",
                             f"{PROBE}/{ck}-{cell}/pass{p['pass']}/summary.json",
                             REPORT] + ([tr_src] if tr_src else []),
                )

    # ---- checkpoint identity: one sha per job, all six jobs per checkpoint ----
    ckpt_block = {}
    for ck in CKPTS:
        shas = {c["sha256"] for c in ckpt_info[ck]}
        if len(shas) != 1:
            fail(f"checkpoint {ck}: {len(shas)} distinct sha256 across jobs")
        by_job = sorted(ckpt_info[ck], key=lambda c: int(c["job"]))
        ckpt_block[ck] = dict(
            path=by_job[0]["path"], size_bytes=by_job[0]["size_bytes"],
            sha256=by_job[0]["sha256"],
            sha256_first_job=dict(job=by_job[0]["job"], cell=by_job[0]["cell"],
                                  sha256=by_job[0]["sha256"]),
            sha256_last_job=dict(job=by_job[-1]["job"], cell=by_job[-1]["cell"],
                                 sha256=by_job[-1]["sha256"]),
            n_jobs_recording_sha=len(by_job),
            n_distinct_sha=len(shas),
            sha256_before_after_note=(
                "The probe records the checkpoint sha256 once per job in "
                "probe.json['checkpoint']; it does not record a separate "
                "before-pass and after-pass hash. 'Before' and 'after' here are "
                "the first and last of the six jobs by job id; all six agree. "
                "Checkpoints were symlinked into each cell, never copied or "
                "written (log section 9)."))
    pa, pb = probes[("A", "C1a")], probes[("B", "C1a")]
    ckpt_block["A"].update(
        label="A", run_id="oglanet-sibc4-loco-r450-chicago-hr-s42-n450",
        description="SIB (C4) OGLANet, n450 matched-450 retrain, chicago, "
                    "high resolution",
        inference_path="SIB trainer path: train_oglanet_sib.main() with "
                       "epochs=0, test_once (published test_and_save_predictions "
                       "eval body)",
        batch_size=pa["batch_size"], precision="fp16 autocast",
        build=pa["build"], cell_meta=pa["cell"])
    ckpt_block["B"].update(
        label="B", run_id="oglanet-vanilla-loco-r450-chicago-hr-s0",
        legacy_run_dir="oglanet_loco_holdout_chicago_highres_1",
        description="published OGLANet vanilla LOCO chicago high resolution",
        inference_path="run_inference_probs.py (load_model strict=False, "
                       "_extract_logits) via oglanet_probsource.t1_build/t1_pass",
        batch_size=pb["batch_size"], precision="fp32",
        build=pb.get("build"), cell_meta=pb.get("cell"))
    if pa["batch_size"] != 8 or pa["fp16"] is not True:
        fail("checkpoint A is not batch 8 fp16")
    if pb["batch_size"] != 4 or pb["fp16"] is not False:
        fail("checkpoint B is not batch 4 fp32")
    r21 = load_json(R21)
    r21_b_sha = (r21["finding_4_oglanet_model_source"]["per_run"]
                 ["t1/chicago"]["checkpoint"]["sha256"])
    if r21_b_sha != ckpt_block["B"]["sha256"]:
        fail("checkpoint B sha differs from results/21 finding 4 t1/chicago")
    ckpt_block["B"]["same_sha_as_results21_finding4_t1_chicago"] = True

    # ---- cross-check (a): per-image count identity vs C1a pass 0 ----
    cc_a = []
    for ck in CKPTS:
        ref = pd.read_parquet(os.path.join(
            BASE, PROBE, f"{ck}-C1a/pass0/perimage_probe.parquet"))
        ref = ref.set_index("image_id").sort_index()
        if len(ref) != 150 or not ref.index.is_unique:
            fail(f"{ck}-C1a p0 parquet: {len(ref)} rows / non-unique ids")
        for (k2, cell, pss), row in rows.items():
            if k2 != ck:
                continue
            df = pd.read_parquet(os.path.join(
                BASE, PROBE, f"{ck}-{cell}/pass{pss}/perimage_probe.parquet"))
            df = df.set_index("image_id").sort_index()
            if not df.index.equals(ref.index):
                fail(f"{ck}-{cell} p{pss}: image ids differ from C1a p0")
            for c in COUNT_COLS:
                if df[c].dtype != "int64":
                    fail(f"{ck}-{cell} p{pss}: {c} dtype {df[c].dtype}")
            n_cnt = int((df[COUNT_COLS] == ref[COUNT_COLS]).all(axis=1).sum())
            n_score = int(((df["miou_k2_twoclass_pct"] == ref["miou_k2_twoclass_pct"])
                           & (df["fg_iou_k2_pct"] == ref["fg_iou_k2_pct"])).sum())
            claim = row["n_images_bit_identical_to_C1a_pass0_of_150"]
            if claim == 150:
                ok = n_cnt == 150
                test = "claim 150/150 requires all 150 images count-identical"
            else:
                ok = n_cnt >= claim
                test = (f"claim {claim}/150: count identity cannot refute it "
                        f"(bit-identical implies count-identical, not the "
                        f"reverse); the check only requires count-identical "
                        f">= claim")
            cc_a.append(dict(row_id=row["row_id"],
                             log_claim_bit_identical_of_150=claim,
                             n_images_count_identical_of_150=n_cnt,
                             n_images_perimage_score_identical_of_150=n_score,
                             test=test, refutable=claim == 150,
                             result="PASS" if ok else "FAIL"))
            row["n_images_count_identical_to_C1a_pass0_of_150__analysis_host_parquet"] = n_cnt
    # C3 pass 1 vs C3 pass 0 is also in probe.json vs_pass0 (Delta, from probs);
    # C3 pass 0 is the C1a pass 0 state, so it must agree with the report's 0.
    for ck in CKPTS:
        v = probes[(ck, "C3")]["passes"][1]["vs_pass0"]
        rr = rows[(ck, "C3", 1)]
        ok = (v["n_images_bit_identical"] ==
              rr["n_images_bit_identical_to_C1a_pass0_of_150"] == 0
              and v["max_over_images"] == rr["max_abs_dp_vs_C1a_pass0"])
        cc_a.append(dict(
            row_id=f"{ck}-C3-p1",
            test="probe.json vs_pass0 (C3 pass 1 vs C3 pass 0, from probs on "
                 "Delta) agrees with report.json vs C1a pass 0, since C3 pass 0 "
                 "is bit-identical to C1a pass 0",
            probe_vs_pass0_n_bit_identical=v["n_images_bit_identical"],
            probe_vs_pass0_max_abs_dp=v["max_over_images"],
            refutable=True, result="PASS" if ok else "FAIL"))
    if any(c["result"] != "PASS" for c in cc_a):
        fail("cross-check (a) failed: " +
             json.dumps([c for c in cc_a if c["result"] != "PASS"]))

    # ---- cross-check (b): min / max / range over passes ----
    def rng(ck, field, cells=None):
        vals = [r[field] for (k, c, _), r in rows.items()
                if k == ck and (cells is None or c in cells)
                and r[field] is not None]
        return dict(n_passes=len(vals), min=min(vals), max=max(vals),
                    range=max(vals) - min(vals))

    MM, MMF = ("miou_k2_twoclass_pct__metric_module_field",
               "fg_iou_k2_pct__metric_module_field")
    TR, TRF = ("miou_k2_twoclass_pct__trainer_field",
               "fg_iou_k2_pct__trainer_field")
    ranges = {
        "A": {"all_8_passes": {TR: rng("A", TR), TRF: rng("A", TRF),
                               MM: rng("A", MM), MMF: rng("A", MMF)},
              "three_seeds_C1a_C2s0_C2s1": {
                  TR: rng("A", TR, ("C1a", "C2s0", "C2s1")),
                  MM: rng("A", MM, ("C1a", "C2s0", "C2s1"))}},
        "B": {"all_8_passes": {MM: rng("B", MM), MMF: rng("B", MMF)},
              "three_seeds_C1a_C2s0_C2s1": {
                  MM: rng("B", MM, ("C1a", "C2s0", "C2s1"))},
              "C2_C3_C4_six_passes": {
                  MM: rng("B", MM, ("C2s0", "C2s1", "C3", "C4"))}},
    }
    fa, fb = rep["facts"]["A"], rep["facts"]["B"]
    cc_b = []

    def chk(name, got, want_log, want_report=None):
        ok = r6(got) == want_log and (want_report is None or got == want_report)
        cc_b.append(dict(check=name, computed=got, log_value=want_log,
                         report_json_value=want_report,
                         result="PASS" if ok else "FAIL"))

    chk("A trainer two-class mIoU range over 8 passes (log s8: 1.466038)",
        ranges["A"]["all_8_passes"][TR]["range"], 1.466038)
    chk("A trainer two-class mIoU min over 8 passes (log Q5: 68.531270)",
        ranges["A"]["all_8_passes"][TR]["min"], 68.531270)
    chk("A trainer two-class mIoU max over 8 passes (log Q5: 69.997308)",
        ranges["A"]["all_8_passes"][TR]["max"], 69.997308)
    chk("A metric-module mIoU span over seeds 42/0/1 (log Q2: 0.660029)",
        ranges["A"]["three_seeds_C1a_C2s0_C2s1"][MM]["range"], 0.660029,
        fa["Q2"]["miou_span_over_three_seeds"])
    chk("B metric-module mIoU span over seeds 42/0/1 (log Q2: 1.372855)",
        ranges["B"]["three_seeds_C1a_C2s0_C2s1"][MM]["range"], 1.372855,
        fb["Q2"]["miou_span_over_three_seeds"])
    chk("B metric-module mIoU span over C2/C3/C4 (log Q6: 2.279582)",
        ranges["B"]["C2_C3_C4_six_passes"][MM]["range"], 2.279582,
        fb["Q6"]["miou_span_C2_C3_C4"])
    chk("B metric-module mIoU range over all 8 passes equals the C2/C3/C4 "
        "span (caveat 3 says 'over all conditions')",
        ranges["B"]["all_8_passes"][MM]["range"], 2.279582)
    if any(c["result"] != "PASS" for c in cc_b):
        fail("cross-check (b) failed: " +
             json.dumps([c for c in cc_b if c["result"] != "PASS"]))

    # ---- stored training-time value of checkpoint A ----
    q5 = fa["Q5"]
    stored = q5["stored_miou"]
    if r6(stored) != 68.870627:
        fail(f"stored value {stored} != 68.870627")
    a_tr = sorted(r[TR] for (k, _, _), r in rows.items() if k == "A")
    rA = ranges["A"]["all_8_passes"][TR]
    inside = rA["min"] <= stored <= rA["max"]
    if not inside:
        fail("stored value outside the observed trainer range")
    stored_block = dict(
        checkpoint="A",
        miou_k2_twoclass_pct__trainer_field=stored,
        fg_iou_k2_pct__trainer_field=q5["stored_fg"],
        source=q5["stored_source"] + " (copied via report.json facts.A.Q5)",
        observed_trainer_range_over_8_passes=[rA["min"], rA["max"]],
        inside_observed_range=inside,
        n_passes_below_stored=sum(v < stored for v in a_tr),
        n_passes_above_stored=sum(v > stored for v in a_tr),
        reproduced_exactly_by=q5["reproduced_exactly_by"],
        closest_pass=dict(row_id="A-C2s1-p0", value=q5["closest"][2],
                          offset=q5["closest_offset"]),
        seed42_value=rows[("A", "C1a", 0)][TR],
        seed42_offset_log="-0.339357 (log Q5); reproduces the 2026-10-31 "
                          "five-node value to every printed digit",
        reentry_value_2026_10_31=q5["reentry_value_2026_10_31"],
        field_used="trainer field (test_results.json tolerant_2px mIOU), the "
                   "field the stored value lives in")

    n_mm_below = sum(r[MM] < r[TR] for (k, _, _), r in rows.items() if k == "A")
    above = [r["row_id"] for (k, _, _), r in rows.items()
             if k == "A" and r[MM] > r[TR]]

    # ---- Q1-Q6 and the reading, copied verbatim from the log ----
    qsec = section(log, "## 6. Q1-Q6", "## 7. The reading")
    qparas = re.split(r"\n(?=\*\*Q[1-6] )", qsec)
    qtext = {}
    for para in qparas:
        m = re.match(r"\*\*(Q[1-6]) ", para)
        if m:
            qtext[m.group(1)] = " ".join(para.split())
    if sorted(qtext) != [f"Q{i}" for i in range(1, 7)]:
        fail(f"log Q paragraphs found: {sorted(qtext)}")
    q_rows = {
        "Q1": ["A-C1a-p0", "A-C1b-p0", "B-C1a-p0", "B-C1b-p0"],
        "Q2": ["A-C1a-p0", "A-C2s0-p0", "A-C2s1-p0",
               "B-C1a-p0", "B-C2s0-p0", "B-C2s1-p0"],
        "Q3": ["A-C3-p0", "A-C3-p1", "A-C4-p0", "A-C4-p1",
               "B-C3-p0", "B-C3-p1", "B-C4-p0", "B-C4-p1"],
        "Q4": sorted(r["row_id"] for r in rows.values()),
        "Q5": sorted(r["row_id"] for r in rows.values()
                     if r["checkpoint"] == "A"),
        "Q6": ["B-C2s0-p0", "B-C2s1-p0", "B-C3-p0", "B-C3-p1",
               "B-C4-p0", "B-C4-p1"],
    }
    q_short = {
        "Q1": "Does a fixed seed reproduce bit-identically across jobs and "
              "nodes? YES, on both checkpoints.",
        "Q2": "Does changing the seed change the score? YES, by roughly "
              "0.7-1.4 mIoU.",
        "Q3": "Do two consecutive passes in one process differ? YES, seeded "
              "and unseeded alike.",
        "Q4": "Does the RNG state advance during a pass? The CPU generator "
              "yes, in 16/16 passes; the CUDA generator no, in 0/16.",
        "Q5": "Does any condition reproduce checkpoint A's stored 68.870627? "
              "NO, as expected.",
        "Q6": "Does checkpoint B's spread match the earlier finding? The "
              "per-pixel swing matches; the mIoU span is larger, and for a "
              "known reason.",
    }
    for q, s in q_short.items():
        if " ".join(s.split()[:6]).lower() not in qtext[q].replace("**", "").lower():
            fail(f"{q} short answer does not match the log")
    # Q4 copy check against the rows
    n_cpu_adv = sum(r["rng_sha256_cpu_before"] != r["rng_sha256_cpu_after"]
                    for r in rows.values())
    n_cuda_adv = sum(r["rng_sha256_cuda_before"] != r["rng_sha256_cuda_after"]
                     for r in rows.values())
    if (n_cpu_adv, n_cuda_adv) != (16, 0):
        fail(f"Q4 rows: cpu advanced {n_cpu_adv}, cuda {n_cuda_adv}")
    for ck in CKPTS:
        if (rows[(ck, "C3", 1)]["rng_sha256_cpu_before"]
                != rows[(ck, "C3", 0)]["rng_sha256_cpu_after"]):
            fail(f"{ck}: C3 pass 1 does not start where pass 0 ended")
    answers_q = {q: dict(short=q_short[q], log_text_verbatim=qtext[q],
                         supporting_rows=q_rows[q],
                         log_line=log_line_of(log, f"**{q} "))
                 for q in sorted(qtext)}

    rsec = section(log, "## 7. The reading", "## 8. Flagged")
    if "**R-a — RNG-dependent.**" not in rsec:
        fail("log reading is not R-a")
    reading = dict(
        reading="R-a",
        label="RNG-dependent",
        justification_verbatim=" ".join(
            rsec.split("\n", 1)[1].split("Two contract-level")[0].split()),
        contract_consequences_verbatim=" ".join(
            ("Two contract-level" + rsec.split("Two contract-level")[1]).split()),
        log_line=log_line_of(log, "## 7. The reading"))

    # ---- mechanism facts, each citation asserted present in the log ----
    mech = []
    for fact, cites, needle in [
        ("DFFM assigns a lambda to self.downsample_conv, so the factory is a "
         "plain Python attribute and not a submodule",
         ["dffm.py:34-38"], "`dffm.py:34-38` assigns a **lambda**"),
        ("Each per-forward downsample layer is BUILT on CPU inside forward()",
         ["dffm.py:81"], ":81  built HERE"),
        ("... then COPIED to CUDA if available",
         ["dffm.py:82-83"], ":83  copied, not built"),
        ("The initialisers therefore draw from the torch CPU default generator "
         "only; the CUDA generator is not involved (confirmed by measurement, "
         "and at runtime in 16/16 passes, Q4)",
         ["dffm.py:80-85"], "torch CPU default generator only"),
        ("20 layers per forward: 4+3+2+1 for s4_d, 3+2+1 for s3_d, 2+1 for "
         "s2_d, 1 for s1_d",
         ["dffm.py:105-108", "dffm.py:130-132", "dffm.py:153-154",
          "dffm.py:176"], "**20 layers per forward**"),
        ("BatchNorm2d of these layers stays in training mode (built after "
         "model.eval()) and normalises with batch statistics; it consumes no "
         "RNG, all RNG goes to the Conv2d weight and bias",
         [], "normalise with **batch statistics**"),
        ("OGLANetSIB uses the same module, with exactly one DFFM call per "
         "forward at eval",
         ["oglanet_sib.py:32", "oglanet_sib.py:115", "oglanet_sib.py:240",
          "oglanet_sib.py:256-271"], "exactly one DFFM call per forward"),
        ("run_inference_probs.py seeds nothing: a grep for manual_seed|"
         "np.random.seed|random.seed|deterministic|benchmark|CUBLAS|"
         "set_default_device|use_deterministic over all 473 lines returns "
         "zero matches",
         ["run_inference_probs.py (473 lines, 0 matches)"],
         "over all 473 lines returns **zero matches**"),
        ("Creating a DataLoader iterator draws one int64 from the CPU default "
         "generator, so pass 2 of a two-pass job starts from a different "
         "generator position even before any forward",
         ["torch/utils/data/dataloader.py:706-710 (torch 2.10.0+cu128)"],
         "dataloader.py:706-710"),
        ("The SIB trainer never seeds torch; its only seed call is "
         "np.random.seed(42) inside _print_bootstrap_comparison, which runs "
         "after the test pass",
         ["train_oglanet_sib.py:1187", "train_oglanet_sib.py:1180"],
         "`train_oglanet_sib.py:1187` `np.random.seed(42)`"),
        ("The step-5 re-entry harness seeds 42 (random, numpy, "
         "torch.manual_seed, torch.cuda.manual_seed_all) immediately before "
         "mod.main()",
         ["n450_sib_noise.py:154", "n450_sib_noise.py:51-57",
          "n450_sib_noise.py:166", "n450_roster.py:33"],
         "`:154` `seed_all(R.SEED)`"),
        ("The production driver has the same ordering, so the original "
         "training job and the re-entry enter main() at the identical "
         "generator state; they reach the test pass at different positions "
         "because training (100 epochs) consumes the generator first",
         ["n450_train.py:226", "n450_train.py:239",
          "train_oglanet_sib.py:1646", "train_oglanet_sib.py:1779"],
         "`n450_train.py:226`"),
        ("The checkpoint load uses load_state_dict with default strict=True "
         "and does not error, because every registered key matches; the 20 "
         "per-forward layers are never registered",
         ["train_oglanet_sib.py:1770-1774"],
         "`load_state_dict` at `train_oglanet_sib.py:1774`"),
        ("No determinism setting in either path (measured in all 16 passes: "
         "cudnn_deterministic False, deterministic_algorithms False, "
         "CUBLAS_WORKSPACE_CONFIG None)",
         [], "**None set in either path.**"),
    ]:
        mech.append(dict(fact=fact, citations=cites,
                         log_line=log_line_of(log, needle)))
    det = {json.dumps(r["determinism"], sort_keys=True) for r in rows.values()}
    if len(det) != 1:
        fail("determinism block differs across passes")

    # ---- headline ----
    sA = ranges["A"]["three_seeds_C1a_C2s0_C2s1"][MM]["range"]
    sB = ranges["B"]["three_seeds_C1a_C2s0_C2s1"][MM]["range"]
    headline = (
        f"On two OGLANet Chicago high-resolution checkpoints (A: SIB n450, "
        f"batch 8, fp16; B: published vanilla LOCO, batch 4, fp32), inference "
        f"at a fixed torch seed of 42 was bit-identical across two jobs on two "
        f"nodes (150 of 150 images, max per-pixel |dp| 0), changing the seed "
        f"among 42, 0 and 1 changed two-class mIoU at K=2 by up to {sA:.6f} "
        f"on A and {sB:.6f} on B (metric-module field), and checkpoint A's "
        f"stored training-time value {stored:.6f} lies inside the "
        f"{rA['min']:.6f}-{rA['max']:.6f} trainer-field range of its eight "
        f"passes.")

    caveats = [
        "Two checkpoints, one city (Chicago), high resolution only. Nothing is "
        "measured for the other OGLANet cells.",
        "Eight passes per checkpoint over three seeds (42, 0, 1) plus "
        "unseeded draws. This is not a distribution estimate, and no per-cell "
        "noise band is claimed; a band would need at least 10 seeds.",
        f"The across-seed span here, 2.279582 two-class mIoU on checkpoint B "
        f"over C2/C3/C4 (equal to B's range over all eight passes; A's "
        f"trainer-field range over all eight is "
        f"{ranges['A']['all_8_passes'][TR]['range']:.6f}), is about twice "
        f"E-9's nine-pass span of 1.1158. E-9's passes were unseeded draws "
        f"from one process family; this grid spans three seeds. The two are "
        f"not the same quantity.",
        "Every published OGLANet number in the paper is one unseeded draw of "
        "this RNG-dependent pass, at its stated inference batch size (DFFM "
        "batch dependence, results/21 finding 3).",
        "A dump-only re-entry cannot reproduce an original OGLANet dump. "
        "load_state_dict(strict=True) cannot detect the defect because the "
        "20 per-forward layers are never registered.",
        f"Metric-module values sit about 0.014 below the trainer field on "
        f"checkpoint A ({n_mm_below} of 8 passes below; on {', '.join(above)} "
        f"the metric-module value is above), from the 10-pixel component "
        f"filter and the two tolerant implementations. Comparisons to stored "
        f"training-time values use the trainer field. Checkpoint B has no "
        f"trainer field (run_inference_probs.py writes no test_results.json).",
        "Promotion only: no inference and no recomputation of any score on "
        "this box. Every value is copied from the probe's files; the only "
        "arithmetic is the count-identity and min/max/range cross-checks.",
        "Bit identity (150/150, 0/150) and per-pixel max |dp| are copied from "
        "the Delta report, which computed them from probs.npz. Those files "
        "were not copied to the analysis host, so the analysis host can check only per-image count "
        "identity from perimage_probe.parquet. That check can refute a "
        "150/150 claim and cannot refute a 0/150 claim.",
        "The probe records each checkpoint's sha256 once per job, not before "
        "and after each pass. All six jobs per checkpoint record the same "
        "sha256, and the log states the checkpoints were symlinked and never "
        "written.",
    ]

    sanity = [
        "The Q1 zeros are exact (150/150, max |dp| 0.000000 across two nodes). "
        "The log flags this as expected, and the CPU generator sha256 before "
        "and after also matches across the two jobs, so the generator "
        "trajectory itself is identical.",
        "Line-number drift in citations: results/21 finding 3 cites the DFFM "
        "lambda at dffm.py:33-37 and the call site at 66-82; the probe log "
        "cites 34-38 and 80-85, and results/21's public-release entry gives "
        "lambda line 34 and call site 81. This JSON uses the probe log's "
        "numbers. Not resolved here.",
        "REVIEWS.md E-9 says 'No stochastic module exists on the vanilla "
        "forward path' and 'Mechanism unidentified'. This probe contradicts "
        "both: the 20 per-forward DFFM layers draw from the CPU generator on "
        "the vanilla path (checkpoint B). REVIEWS.md was not edited.",
        "results/27's caveat on the 0.339357 offset offered as its leading "
        "hypothesis that the unregistered layers carried trained state in "
        "the original in-process pass. The probe log instead attributes the "
        "offset to generator position: the layers are re-randomised on every "
        "forward, and training consumed the generator before the test pass. "
        "results/27 was not edited.",
        "On checkpoint A, the C2s1 pass has its metric-module mIoU above the "
        "trainer field while the other seven are below; caveat 6's 'about "
        "0.014 below' holds for the seed-42 passes, not every pass.",
        "The log's file table lists the log at "
        "summaries/session_logs/2026-11-01-oglanet-rng-probe.md (Delta path); "
        "on the analysis host it is at " + LOG + ".",
    ]

    rows_list = [rows[k] for k in sorted(
        rows, key=lambda k: (k[0], CELLS.index(k[1]), k[2]))]
    conditions = [dict(
        label=f"OGLANet ckpt {r['checkpoint']} / {r['condition']}",
        row_id=r["row_id"],
        metric=("miou_k2_twoclass_pct__trainer_field" if r["checkpoint"] == "A"
                else "miou_k2_twoclass_pct__metric_module_field"),
        value=(r[TR] if r["checkpoint"] == "A" else r[MM]),
        n=r["n_images"], stderr=None, ci95=None, ci_method=None,
        baseline=None) for r in rows_list]

    srcs = [source(LOG, "the probe log: step 0 citations, tables, Q1-Q6, "
                        "reading R-a"),
            source(REPORT, "per-pass tables and Q facts (Delta report)"),
            source(REPORT_MD, "the same tables in markdown"),
            source(R21, "findings 3 and 4; answers ids; checkpoint B sha"),
            source(R27, "the caveat on the 0.339357 offset")]
    for ck in CKPTS:
        for cell in CELLS:
            srcs.append(source(f"{PROBE}/{ck}-{cell}/probe.json",
                               f"{ck}-{cell} per-pass record"))
            for p in probes[(ck, cell)]["passes"]:
                d = f"{PROBE}/{ck}-{cell}/pass{p['pass']}"
                srcs.append(source(d + "/summary.json", f"{ck}-{cell} p{p['pass']}"))
                srcs.append(source(d + "/perimage_probe.parquet",
                                   f"{ck}-{cell} p{p['pass']} cross-check (a)"))
                if ck == "A":
                    srcs.append(source(d + "/test_results.json",
                                       f"{ck}-{cell} p{p['pass']} trainer field"))

    ans = ["E-9", "AC2"] + [a for a in r21["answers"] if a not in ("E-9", "AC2")]
    out = dict(
        id=NAME,
        arm="H2",
        answers=ans,
        answers_detail={
            "E-9": "names the mechanism of OGLANet inference nondeterminism "
                   "for the erratum and the reproducibility appendix",
            "AC2": "separates the measured observations (Q1-Q6) from the "
                   "hypothesis results/27 carried about the 0.339357 offset",
            "inherited_from_results_21": [a for a in r21["answers"]
                                          if a not in ("E-9", "AC2")],
        },
        question="Does OGLANet inference output depend on the torch RNG state "
                 "at inference time, and does that reconcile results/21 "
                 "finding 4 with the 2026-10-31 seeded diagnosis?",
        status="PROMOTED_EVIDENCE",
        promotion_note="This JSON computes no score. Every value is copied "
                       "from the Delta probe under " + PROBE + "/ and its "
                       "session log. Checks performed on the copy: report.json "
                       "== probe.json == summary.json per pass, exactly; "
                       "trainer field == the pass's own test_results.json; "
                       "the log's printed tables == report.json at 6 dp; plus "
                       "the two allowed cross-checks (a) and (b).",
        checkpoints=ckpt_block,
        per_pass={"A": [r for r in rows_list if r["checkpoint"] == "A"],
                  "B": [r for r in rows_list if r["checkpoint"] == "B"]},
        per_pass_field_note="trainer_field = the SIB trainer's own "
                            "test_results.json tolerant_2px (A only); "
                            "metric_module_field = shadowtransfer/metrics.py "
                            "(sha256 " + pa["metric_module_sha256"] + ") via "
                            "the probe's _agg_check-asserted roll-up.",
        answers_q1_q6=answers_q,
        reading=reading,
        mechanism=mech,
        stored_training_time_value=stored_block,
        ranges_over_passes=ranges,
        cross_checks={
            "a_count_identity_vs_C1a_pass0": cc_a,
            "b_min_max_range": cc_b,
            "copy_checks_rows_verified": copy_checks,
            "all_pass": True,
        },
        probe=dict(script="python/oglanet_rng_probe.py (Delta)",
                   sha256=pa["probe_sha256"],
                   metric_module_sha256=pa["metric_module_sha256"],
                   torch=probes[("A", "C1a")]["passes"][0]["torch_version"],
                   gpu=rows_list[0]["gpu_model"],
                   determinism_all_passes=rows_list[0]["determinism"]),
        conditions=conditions,
        conditions_note="one pass per row over the same 150 images; no "
                        "stderr or CI because each value is a single pass",
        tests=[],
        tests_note="no statistical test: a promotion of a deterministic "
                   "seed x pass grid",
        runs=[ckpt_block["A"]["run_id"], ckpt_block["B"]["run_id"]],
        runs_note="A by its run_id; B is the published cell in legacy run dir "
                  "oglanet_loco_holdout_chicago_highres_1, named by the "
                  "run_id results/21 uses for it, and identified by checkpoint "
                  "sha256 " + ckpt_block["B"]["sha256"] + " (same as results/21 "
                  "finding 4 t1/chicago).",
        index_sha256=None,
        index_sha256_reason="every source is a Delta probe artifact under "
                            "delta_runs/, not a dump tree",
        source_artifacts=srcs,
        script="analysis/" + NAME + ".py",
        seed=0,
        seed_note="no stochastic step on the analysis host; the probe's own seeds "
                  "(42, 0, 1, unseeded) are recorded per pass",
        headline=headline,
        caveats=caveats,
        sanity_flags=sanity,
        figure=None,
        caption=None,
    )
    out.update(env_block())

    print("per-pass rows:", len(rows_list))
    for r in rows_list:
        print(f"  {r['row_id']:10s} {r['condition']:10s} {r['slurm_job_id']} "
              f"{r['node']} tr={r[TR]} mm={r[MM]:.6f} "
              f"ident={r['n_images_bit_identical_to_C1a_pass0_of_150']} "
              f"cnt={r['n_images_count_identical_to_C1a_pass0_of_150__analysis_host_parquet']} "
              f"dp={r['max_abs_dp_vs_C1a_pass0']:.6f}")
    print("cross-check (a):", sum(c["result"] == "PASS" for c in cc_a), "/",
          len(cc_a), "PASS")
    print("cross-check (b):", sum(c["result"] == "PASS" for c in cc_b), "/",
          len(cc_b), "PASS")
    print("headline:", headline)
    write_results(out, NAME, DRY_RUN)


if __name__ == "__main__":
    main()
