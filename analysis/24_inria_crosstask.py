#!/usr/bin/env python
"""
Arm H2 -- INRIA cross-task replication: source of the printed numbers.

Answers BxRx-W2, AC1.

Section 4.3 and App. "INRIA Cross-Task Replication" print five cross-task
numbers. This job finds where each is printed, finds the output they come from,
and recomputes each one with the test the paper names.

WHAT EXISTS ON THIS BOX. Exactly one INRIA output:
  Shade Maps/data/figure3_data/inria_sp_battery_results.json
It is a PER-CELL record: three shadow cells and three INRIA cells, one row per
(task, held-out city, signature). There is no per-image INRIA output anywhere on
the analysis host and no INRIA dump in CONTRACT.md's schema. The search is recorded in full
in the output JSON's `search_record`.

WHAT THAT MEANS FOR THE RECOMPUTATION. Every printed test is a CELL-LEVEL test
over n = 6 cells, so the recomputation is exact and complete at the layer the
paper tests: the six per-cell statistics are read from the battery file and the
sign test, the Wilcoxon test and Cohen's d are recomputed here. The per-cell
statistics THEMSELVES cannot be re-derived on this box, because the per-image
INRIA risk-coverage inputs do not exist here. This job checks the test layer,
not the measurement layer, and says so in the caveats.

Read-only. No training, no GPU inference, no dump read, nothing modified. Writes
exactly one file: results/24_inria_crosstask.json.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/24_inria_crosstask.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

BASE = os.environ["PROJECT_ROOT"]
SHADE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
OUT_NAME = "24_inria_crosstask.json"

BATTERY = os.path.join(SHADE, "data", "figure3_data",
                       "inria_sp_battery_results.json")
SUBMITTED_COMMIT = "21151b8"
PAPER = os.path.join(BASE, "final_paper")

# Every tree the brief names, plus the two the paper's own figure code reads.
SEARCH_ROOTS = [
    ("runs", os.path.join(BASE, "runs")),
    ("dumps", os.path.join(BASE, "dumps")),
    ("delta_dumps", os.path.join(BASE, "delta_dumps")),
    ("results", os.path.join(BASE, "results")),
    ("logs", os.path.join(BASE, "logs")),
    ("Shade Maps/data", os.path.join(SHADE, "data")),
]

SHADOW_CITIES = ("chicago", "miami", "phoenix")
INRIA_CITIES = ("austin", "chicago", "vienna")

# The five printed claims, each with the verdict key it rests on, the quantity
# the verdict pools, and the values the paper prints. `printed` is what the
# paper says; nothing below is rounded to reach it.
CLAIMS = [
    dict(key="S1_gt_pos_gap",
         name="foreground SP gap transfers in direction",
         quantity="per-cell foreground AURC gap, LOCO minus upper, on GT "
                  "foreground pixels (s1 '<city>|gt_pos' -> gap)",
         printed=dict(sign_test_p=0.016, cohens_d_gt=1.1, n_same_direction=6,
                      n_cells=6),
         printed_in=["sections/evaluation.tex", "appendix/app_eval.tex"]),
    dict(key="F2_class_asymmetry",
         name="foreground-vs-background asymmetry transfers in direction",
         quantity="per-cell foreground gap minus background gap "
                  "(f2 -> delta)",
         printed=dict(sign_test_p=0.016, cohens_d_gt=1.1, n_same_direction=6,
                      n_cells=6),
         printed_in=["sections/evaluation.tex", "appendix/app_eval.tex"]),
    dict(key="F5_CE_underreporting",
         name="cross-entropy under-reporting transfers in direction",
         quantity="per-cell cross-entropy gap minus 0/1 gap "
                  "(f5 -> gap_ce - gap_01)",
         printed=dict(sign_test_p=0.016, cohens_d_gt=1.1, n_same_direction=6,
                      n_cells=6),
         printed_in=["appendix/app_eval.tex"]),
    dict(key="A1_logit_asymmetry",
         name="predicted-positive logit erosion transfers in direction on 5 "
              "of 6 cells",
         quantity="per-cell logit asymmetry, delta_TP minus delta_FP "
                  "(a1 -> asymmetry)",
         printed=dict(wilcoxon_p=0.031, cohens_d=1.5, n_same_direction=5,
                      n_cells=6),
         printed_in=["appendix/app_eval.tex"]),
    dict(key="E1_ECE_asymmetry",
         name="class-stratified ECE flips direction between tasks",
         quantity="per-cell ECE rise on predicted-positive minus the rise on "
                  "predicted-negative pixels (e1 -> rise_asymmetry)",
         printed=dict(direction_shadow="positive on all 3 cells",
                      direction_inria="not positive on all 3 cells",
                      vienna_rise_neg=0.138, vienna_rise_pos=0.034),
         printed_in=["appendix/app_eval.tex"]),
]

# An extra printed INRIA number in the same appendix paragraph block. It is NOT
# one of the five the brief enumerates; it is checked because this job is
# already holding the file that would settle it, and reported separately.
EXTRA_PRINTED_MIOU_RANGE = [0.74, 0.81]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fail(msg):
    raise SystemExit("ASSERTION FAILED: " + msg)


def rnd(x, nd):
    """Round half away from zero, the way a printed value is read."""
    return float(np.sign(x) * np.floor(abs(x) * 10 ** nd + 0.5) / 10 ** nd)


# ----------------------------------------------------------------------------
# step 1 -- where each value is printed now
# ----------------------------------------------------------------------------

def find_printed():
    """Every line mentioning INRIA in the paper's sections and appendix, with
    file, line number and the sentence verbatim."""
    hits = []
    for sub in ("sections", "appendix"):
        d = os.path.join(PAPER, sub)
        for name in sorted(os.listdir(d)):
            if not name.endswith(".tex"):
                continue
            rel = f"{sub}/{name}"
            for i, line in enumerate(open(os.path.join(d, name)).read()
                                     .splitlines(), start=1):
                if "inria" not in line.lower():
                    continue
                hits.append(dict(file=rel, line=i, commented=line.lstrip()
                                 .startswith("%"), text=line.strip()))
    return hits


def in_submitted(rel, needle):
    """Is this sentence in the submitted commit, or only in the worktree?"""
    try:
        blob = subprocess.run(
            ["git", "-C", PAPER, "show", f"{SUBMITTED_COMMIT}:{rel}"],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return dict(checked=False, why=str(e))
    return dict(checked=True, present=needle in blob,
                commit=SUBMITTED_COMMIT)


# ----------------------------------------------------------------------------
# step 2 -- what INRIA outputs exist
# ----------------------------------------------------------------------------

def search_for_outputs():
    """grep -ril inria over the roots the brief names. Recorded in full: a
    search that found nothing is the finding, so the roots that were searched
    have to be in the output, not only the one that hit."""
    record = []
    for label, root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            record.append(dict(root=label, path=root, exists=False,
                               by_name=[], by_content=[]))
            continue
        by_name = [l for l in subprocess.run(
            ["find", root, "-iname", "*inria*"],
            capture_output=True, text=True).stdout.splitlines() if l.strip()]
        by_content = subprocess.run(
            ["grep", "-ril", "--", "inria", root],
            capture_output=True, text=True).stdout.splitlines()

        def rel(p, _root=root):
            # paths containing a space must never be split; relative to the
            # tree that owns them, so the two roots do not produce ../../..
            anchor = BASE if p.startswith(BASE) else SHADE
            return os.path.relpath(p, anchor)
        record.append(dict(
            root=label, path=root, exists=True,
            by_name=sorted(rel(p) for p in by_name),
            by_content=sorted(rel(p) for p in by_content if p.strip())))
    return record


# ----------------------------------------------------------------------------
# step 3 -- recompute
# ----------------------------------------------------------------------------

def cell_values(bat):
    """The six per-cell values behind each verdict, in a fixed order:
    the three shadow cells then the three INRIA cells."""
    out = {}
    s1 = []
    for task, cities in (("shadow", SHADOW_CITIES), ("inria", INRIA_CITIES)):
        for c in cities:
            k = f"{c}|gt_pos"
            if k not in bat[task]["s1"]:
                fail(f"{task}/s1: no '{k}' record")
            s1.append(dict(task=task, city=c,
                           value=bat[task]["s1"][k]["gap"],
                           n_images=bat[task]["s1"][k]["n_images"]))
    out["S1_gt_pos_gap"] = s1

    def rows(probe, fn):
        r = []
        for task, cities in (("shadow", SHADOW_CITIES),
                             ("inria", INRIA_CITIES)):
            by_city = {x["city"]: x for x in bat[task][probe]}
            for c in cities:
                if c not in by_city:
                    fail(f"{task}/{probe}: no row for {c}")
                r.append(dict(task=task, city=c, value=fn(by_city[c]),
                              n_images=by_city[c].get("n_images")))
        return r

    out["F2_class_asymmetry"] = rows("f2", lambda x: x["delta"])
    out["F5_CE_underreporting"] = rows(
        "f5", lambda x: x["gap_ce"] - x["gap_01"])
    out["A1_logit_asymmetry"] = rows("a1", lambda x: x["asymmetry"])
    out["E1_ECE_asymmetry"] = rows("e1", lambda x: x["rise_asymmetry"])
    return out


def recompute(vals):
    x = np.asarray([v["value"] for v in vals], dtype=np.float64)
    n = x.size
    n_pos = int((x > 0).sum())
    n_neg = int((x < 0).sum())
    if n_pos + n_neg != n:
        fail("a per-cell value is exactly zero; the sign test is not defined "
             "without a stated tie rule")
    sign_one = float(stats.binomtest(n_pos, n, 0.5,
                                     alternative="greater").pvalue)
    sign_two = float(stats.binomtest(n_pos, n, 0.5,
                                     alternative="two-sided").pvalue)
    w_one = float(stats.wilcoxon(x, alternative="greater").pvalue)
    w_two = float(stats.wilcoxon(x, alternative="two-sided").pvalue)
    sd = float(x.std(ddof=1))
    return dict(
        n_cells=n, n_same_direction_positive=n_pos,
        n_opposite_direction=n_neg,
        values=[dict(task=v["task"], city=v["city"], value=v["value"],
                     n_images=v["n_images"], sign=("+" if v["value"] > 0
                                                   else "-")) for v in vals],
        mean=float(x.mean()),
        mean_shadow=float(x[:3].mean()), mean_inria=float(x[3:].mean()),
        sd_ddof1=sd,
        cohens_d_one_sample=float(x.mean() / sd),
        cohens_d_definition="one-sample Cohen's d, mean over the 6 cells "
                            "divided by their sample standard deviation "
                            "(ddof=1); reproduces the stored pooled_d exactly",
        sign_test_p_one_tailed=sign_one,
        sign_test_p_two_tailed=sign_two,
        wilcoxon_p_one_tailed=w_one,
        wilcoxon_p_two_tailed=w_two,
        smallest_possible_sign_p_one_tailed=float(0.5 ** n),
        smallest_possible_sign_p_two_tailed=float(2 * 0.5 ** n))


def main():
    hr = "=" * 100
    print(hr)
    print("Arm H2 -- INRIA cross-task replication, source of the printed "
          "numbers")
    print(f"DRY_RUN = {DRY_RUN}")
    print(hr)

    # ---- step 1 ----------------------------------------------------------
    printed_lines = find_printed()
    print(f"\n--- step 1: INRIA mentions in final_paper ({len(printed_lines)}) "
          "---")
    for h in printed_lines:
        tag = " (commented out)" if h["commented"] else ""
        print(f"  {h['file']}:{h['line']}{tag}  {h['text'][:110]}...")

    # ---- step 2 ----------------------------------------------------------
    search = search_for_outputs()
    print("\n--- step 2: search for INRIA outputs ---")
    for r in search:
        if not r["exists"]:
            print(f"  {r['root']:20s} ABSENT")
            continue
        found = sorted(set(r["by_name"]) | set(r["by_content"]))
        print(f"  {r['root']:20s} by name {len(r['by_name']):2d}, by content "
              f"{len(r['by_content']):2d} -> "
              + (", ".join(found) if found else "nothing"))
    if not os.path.isfile(BATTERY):
        fail(f"{BATTERY}: the one INRIA output is absent; nothing can be "
             "recomputed")
    bat = json.load(open(BATTERY))
    bat_sha = sha256_file(BATTERY)
    for k in ("shadow", "inria", "verdicts"):
        if k not in bat:
            fail(f"{BATTERY}: top-level key {k} absent")
    if bat.get("shadow_excluded") or bat.get("inria_excluded"):
        fail(f"{BATTERY}: records a non-empty exclusion list; the pooled n=6 "
             "the paper prints assumes none")

    # ---- step 3 ----------------------------------------------------------
    vals = cell_values(bat)
    comparisons, mismatches = [], []
    print("\n--- step 3: printed against recomputed ---")
    for claim in CLAIMS:
        key = claim["key"]
        rec = recompute(vals[key])
        stored = bat["verdicts"][key]

        # the stored verdict block must agree with the recomputation, or the
        # battery file and this job disagree about its own contents
        for field, got in (("pooled_mean", rec["mean"]),
                           ("shadow_mean", rec["mean_shadow"]),
                           ("inria_mean", rec["mean_inria"]),
                           ("pooled_d", rec["cohens_d_one_sample"]),
                           ("sign_p", rec["sign_test_p_one_tailed"]),
                           ("wilcoxon_p", rec["wilcoxon_p_one_tailed"])):
            if abs(float(stored[field]) - got) > 1e-9:
                fail(f"{key}: recomputed {field} {got!r} disagrees with the "
                     f"stored {stored[field]!r} by more than 1e-9")

        p = claim["printed"]
        checks = []

        def chk(label, printed_v, recomputed_v, nd, note=None):
            ok = (printed_v is None or
                  rnd(recomputed_v, nd) == rnd(printed_v, nd))
            c = dict(value=label, printed=printed_v,
                     recomputed=recomputed_v,
                     recomputed_rounded_to_printed_precision=rnd(
                         recomputed_v, nd),
                     agrees_to_printed_precision=bool(ok))
            if note:
                c["note"] = note
            checks.append(c)
            if not ok:
                mismatches.append(dict(claim=key, value=label,
                                       printed=printed_v,
                                       recomputed=recomputed_v))
            return ok

        if "sign_test_p" in p:
            chk("sign test p", p["sign_test_p"],
                rec["sign_test_p_one_tailed"], 3,
                note="the printed value matches the ONE-TAILED sign test. The "
                     "two-tailed value is "
                     f"{rec['sign_test_p_two_tailed']:.6f}.")
        if "wilcoxon_p" in p:
            chk("Wilcoxon p", p["wilcoxon_p"],
                rec["wilcoxon_p_one_tailed"], 3,
                note="the printed value matches the ONE-TAILED Wilcoxon "
                     "signed-rank test. The two-tailed value is "
                     f"{rec['wilcoxon_p_two_tailed']:.6f}.")
        if "cohens_d" in p:
            chk("Cohen's d", p["cohens_d"], rec["cohens_d_one_sample"], 1)
        if "cohens_d_gt" in p:
            ok = rec["cohens_d_one_sample"] > p["cohens_d_gt"]
            checks.append(dict(
                value="Cohen's d > threshold", printed=f"d > "
                f"{p['cohens_d_gt']}", recomputed=rec["cohens_d_one_sample"],
                agrees_to_printed_precision=bool(ok)))
            if not ok:
                mismatches.append(dict(claim=key,
                                       value="Cohen's d > threshold",
                                       printed=p["cohens_d_gt"],
                                       recomputed=rec["cohens_d_one_sample"]))
        if "n_same_direction" in p:
            ok = rec["n_same_direction_positive"] == p["n_same_direction"]
            checks.append(dict(
                value="cells in the same direction",
                printed=f"{p['n_same_direction']} of {p['n_cells']}",
                recomputed=f"{rec['n_same_direction_positive']} of "
                           f"{rec['n_cells']}",
                agrees_to_printed_precision=bool(ok)))
            if not ok:
                mismatches.append(dict(claim=key,
                                       value="cells in the same direction",
                                       printed=p["n_same_direction"],
                                       recomputed=rec[
                                           "n_same_direction_positive"]))

        # E1's two printed per-cell numbers
        if key == "E1_ECE_asymmetry":
            vien = next(r for r in bat["inria"]["e1"] if r["city"] == "vienna")
            chk("INRIA Vienna ECE rise, predicted-negative",
                p["vienna_rise_neg"], vien["rise_neg"], 3)
            chk("INRIA Vienna ECE rise, predicted-positive",
                p["vienna_rise_pos"], vien["rise_pos"], 3)
            shadow_all_pos = all(r["rise_asymmetry"] > 0
                                 for r in bat["shadow"]["e1"])
            inria_all_pos = all(r["rise_asymmetry"] > 0
                                for r in bat["inria"]["e1"])
            flips = bool(shadow_all_pos and not inria_all_pos)
            checks.append(dict(
                value="direction flips between tasks",
                printed="flips", recomputed=(
                    "shadow rise_asymmetry positive on 3 of 3 cells "
                    f"({shadow_all_pos}); INRIA positive on "
                    f"{sum(1 for r in bat['inria']['e1'] if r['rise_asymmetry'] > 0)}"
                    " of 3"),
                agrees_to_printed_precision=flips))
            if not flips:
                mismatches.append(dict(claim=key,
                                       value="direction flips between tasks",
                                       printed="flips",
                                       recomputed="does not flip"))

        comparisons.append(dict(
            claim=key, name=claim["name"], quantity=claim["quantity"],
            printed_in=claim["printed_in"],
            stored_verdict=stored["verdict"],
            checks=checks, recomputed=rec))

        print(f"\n  {key}  [{stored['verdict']}]")
        for c in checks:
            mark = "OK " if c["agrees_to_printed_precision"] else "MISMATCH"
            print(f"    {mark}  {c['value']:46s} printed {c['printed']!s:>12s} "
                  f" recomputed {c['recomputed']!s:>22s}")

    # ---- the extra printed number in the same paragraph ------------------
    inria_miou = {r["city"]: r["miou"] for r in bat["inria"]["f3v2"]}
    lo, hi = min(inria_miou.values()), max(inria_miou.values())
    inside = [c for c, v in inria_miou.items()
              if EXTRA_PRINTED_MIOU_RANGE[0] - 0.005 <= v
              <= EXTRA_PRINTED_MIOU_RANGE[1] + 0.005]
    extra = dict(
        value="INRIA upper-bound mIoU range across cities",
        printed=EXTRA_PRINTED_MIOU_RANGE,
        printed_sentence_file="appendix/app_eval.tex",
        in_scope_of_the_brief=False,
        in_scope_note="not one of the five values this job was asked to "
                      "recompute; checked because the file that would settle "
                      "it was already open, and reported apart from the five.",
        battery_field="inria.f3v2[].miou",
        per_city=inria_miou,
        observed_range=[lo, hi],
        cities_inside_the_printed_range=sorted(inside),
        cities_outside_the_printed_range=sorted(
            set(inria_miou) - set(inside)),
        agrees=bool(len(inside) == len(inria_miou)),
        unresolved=("the battery file does not label f3v2.miou as the upper "
                    "bound or as the LOCO cell. If it is the upper bound, the "
                    "printed range covers two of three cities and omits "
                    f"vienna at {inria_miou.get('vienna'):.4f}. The generator "
                    "that wrote this field is not on this box (the only "
                    "notebook that reads the file, "
                    "Shade Maps/Python/FINAL-fig3.ipynb, consumes f3v2 "
                    "nowhere), so the label cannot be settled here and no "
                    "verdict is recorded."),
        shadow_counterpart={r["city"]: r["miou"]
                            for r in bat["shadow"]["f3v2"]},
        miou_floor=bat.get("miou_floor"))
    print("\n--- extra printed number, outside the five (reported, not "
          "adjudicated) ---")
    print(f"  INRIA f3v2 mIoU per city: {inria_miou}; printed range "
          f"{EXTRA_PRINTED_MIOU_RANGE}; inside: {sorted(inside)}")

    # ---- family-level multiplicity --------------------------------------
    fam_p = [(c["claim"],
              c["recomputed"]["sign_test_p_one_tailed"]
              if c["claim"] != "A1_logit_asymmetry"
              else c["recomputed"]["wilcoxon_p_one_tailed"])
             for c in comparisons]
    order = sorted(range(len(fam_p)), key=lambda i: fam_p[i][1])
    m, holm_adj, running = len(fam_p), [0.0] * len(fam_p), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * fam_p[i][1])
        running = max(running, adj)
        holm_adj[i] = running
    multiplicity = dict(
        family="the five cross-task signatures, as the paper groups them",
        correction="Holm-Bonferroni, m = 5",
        applied_in_the_paper=False,
        note="The paper prints each signature's p unadjusted. CONVENTIONS.md "
             "asks for a correction when a claim rests on a family of cells. "
             "The adjusted values are computed here as a check and are NOT a "
             "replacement for the printed ones; which family these five tests "
             "belong to is a scoping question for the paper, not a number.",
        per_claim=[dict(claim=k, test=("Wilcoxon one-tailed"
                                       if k == "A1_logit_asymmetry"
                                       else "sign test one-tailed"),
                        p_as_printed=p, p_holm=holm_adj[i],
                        still_below_0_05=bool(holm_adj[i] < 0.05))
                   for i, (k, p) in enumerate(fam_p)])
    print("\n--- Holm over the family of five (a check, not a replacement) ---")
    for e in multiplicity["per_claim"]:
        print(f"  {e['claim']:24s} p {e['p_as_printed']:.6f} -> Holm "
              f"{e['p_holm']:.6f}  below 0.05: {e['still_below_0_05']}")

    # ---- how much of the 'three signatures' is one signature -------------
    s1 = np.array([v["value"] for v in vals["S1_gt_pos_gap"]])
    f2 = np.array([v["value"] for v in vals["F2_class_asymmetry"]])
    f5 = np.array([v["value"] for v in vals["F5_CE_underreporting"]])
    redundancy = dict(
        note="S1, F2 and F5 are counted as three signatures in the printed "
             "sentence. They are not three independent measurements of the "
             "cross-task question.",
        f2_minus_s1_per_cell=[float(a - b) for a, b in zip(f2, s1)],
        max_abs_f2_minus_s1=float(np.abs(f2 - s1).max()),
        why_f2_is_nearly_s1="F2's delta is the foreground gap minus the "
                            "background gap, and the background gap is at most "
                            f"{max(abs(a - b) for a, b in zip(f2, s1)):.4f} in "
                            "absolute value on any of the six cells, so F2 is "
                            "S1 shifted by a near-zero quantity.",
        f5_shares_s1="F5's quantity is gap_ce - gap_01, and gap_01 is the same "
                     "number as S1's gap on every cell, so F5's sign depends "
                     "on the same measurement plus the cross-entropy gap.",
        f5_gap01_equals_s1=bool(all(
            abs(r["gap_01"] - g) < 1e-12
            for r, g in zip(
                [x for c in SHADOW_CITIES
                 for x in bat["shadow"]["f5"] if x["city"] == c]
                + [x for c in INRIA_CITIES
                   for x in bat["inria"]["f5"] if x["city"] == c], s1))),
        pearson_s1_f2=float(np.corrcoef(s1, f2)[0, 1]),
        pearson_s1_f5=float(np.corrcoef(s1, f5)[0, 1]))

    # ---- assemble --------------------------------------------------------
    all_agree = not mismatches
    headline = (
        "All five printed INRIA cross-task values recompute exactly from "
        "Shade Maps/data/figure3_data/inria_sp_battery_results.json, the only "
        "INRIA output on this box: sign-test p = 0.015625 and Cohen's d of "
        "1.1448, 1.1413 and 1.1403 on the three 6-of-6 signatures, Wilcoxon "
        "p = 0.03125 with d = 1.5031 on the 5-of-6 logit erosion, and the "
        "class-stratified ECE asymmetry positive on 3 of 3 shadow cells and on "
        "1 of 3 INRIA cells with INRIA Vienna at 0.1376 predicted-negative "
        "against 0.0338 predicted-positive; every p the paper prints is "
        "one-tailed, and no per-image INRIA output exists on this box."
        if all_agree else
        f"{len(mismatches)} of the printed INRIA cross-task values do not "
        "reproduce from the battery file beyond rounding; see mismatches.")

    sanity = [
        "Every p-value the paper prints for this family is ONE-TAILED. The "
        "sign-test 0.016 is the one-tailed 0.015625 (two-tailed 0.03125) and "
        "the logit-erosion 0.031 is the one-tailed Wilcoxon 0.03125 "
        "(two-tailed 0.0625). CONVENTIONS.md asks for two-tailed unless a "
        "directional hypothesis is stated; the appendix states the hypothesis "
        "directionally ('transfer in direction') and its own parenthetical, "
        "'the smallest p-value any sign test can return is 0.016', is only "
        "true one-tailed. Both tails are recorded on every claim. The logit "
        "erosion is the one that moves: two-tailed it is 0.0625, above 0.05.",

        "The test is not the same across the five signatures. Three use the "
        "sign test, the logit erosion uses Wilcoxon, and the battery file "
        "stores both for all five. For the logit erosion the sign test gives "
        "0.109375 and Wilcoxon gives 0.03125, so the reported test is the one "
        "of the two that crosses 0.05 on that signature. Flagged as a "
        "selection the paper does not state; the file shows both values and "
        "both are carried here.",

        "Three of the five signatures are close to one measurement counted "
        "three times. F2's per-cell value differs from S1's by at most "
        "0.00200 on any of the six cells (Pearson r = 0.99999), "
        "because the background AURC gap it subtracts is near zero, and F5's "
        "gap_01 term is byte-identical to S1's gap on every cell. 'Three "
        "findings transfer cleanly, sign-test p = 0.016 each' is therefore "
        "three readings that share a numerator, not three independent tests, "
        "and their three identical p-values are one p-value.",

        "n = 6 is the whole family. With six cells the sign test cannot return "
        "anything below 0.015625 one-tailed, and the paper says so. Every "
        "'p = 0.016' in this family is the floor of the test, not a measured "
        "separation, and three of the five land exactly on it.",

        "No multiplicity correction is applied in the paper to this family of "
        "five. Under Holm-Bonferroni over the five as printed, all four "
        "signatures the paper calls transferred adjust to 0.078125 and the "
        "ECE signature to 0.34375, so none is below 0.05. Recorded as a "
        "check: whether these five form one family is a scoping decision, and "
        "this job does not make it.",

        "The INRIA Vienna cell is the one carrying most of the cross-task "
        "effect. Its foreground gap is 0.3319 against 0.0688 and 0.0778 for "
        "austin and chicago, a factor of 4.3 to 4.8, and its f3v2 mIoU is "
        "0.5686 against 0.8060 and 0.7382. The pooled means and Cohen's d "
        "values are not robust to dropping it, and no leave-one-cell-out "
        "check is printed in the paper or stored in the battery file.",

        "The paper's n for this test and the n of the underlying cells are "
        "different quantities. The sign and Wilcoxon tests run on n = 6 cells; "
        "each cell's own gap rests on 112 to 142 images, not 150, because the "
        "risk-coverage helper drops images with too few GT foreground pixels. "
        "The per-cell image counts are carried on every value in this JSON.",
    ]

    caveats = [
        "The test layer is recomputed; the measurement layer is not. The six "
        "per-cell statistics behind every test here are READ from "
        "Shade Maps/data/figure3_data/inria_sp_battery_results.json and cannot "
        "be re-derived on the analysis host: there is no per-image INRIA output anywhere "
        "on this box, no INRIA dump in CONTRACT.md's schema, and no INRIA "
        "checkpoint. A search by name and by content over runs/, dumps/, "
        "delta_dumps/, results/, logs/ and Shade Maps/data returned exactly "
        "one INRIA file, the battery file itself, plus one results JSON that "
        "cites it (results/06_aurc_recovery.json). The full search is in "
        "search_record.",

        "The battery file is the paper's own artifact and is not independent "
        "evidence about itself. results/06_aurc_recovery.json already names it "
        "as published_s1_record, and the figure-3 notebook reads it to draw "
        "the panel the main text points at. Recomputing the tests from it "
        "confirms that the printed numbers are the numbers that file carries, "
        "not that the file's per-cell measurements are correct.",

        "The generator of the battery file is not on this box. "
        "Shade Maps/Python/FINAL-fig3.ipynb reads the file but does not write "
        "it, and nothing else on the analysis host mentions it. The per-cell definitions "
        "used here (which field each signature pools) were read off the file's "
        "own structure and verified by reproducing its stored verdicts block "
        "to under 1e-9 on all six fields of all five signatures; they were not "
        "read from the code that produced them.",

        "The INRIA cells are single-seed, MAMNet only, and at one resolution. "
        "The battery file carries no seed field, no architecture field and no "
        "repeated run for any INRIA cell, so no noise floor exists for the "
        "six per-cell values and the cross-task claim rests on one draw per "
        "cell. The appendix states MAMNet and the matched recipe; the seed "
        "count is not stated there and is not recoverable here.",

        "Every p-value in the printed family is one-tailed and no multiplicity "
        "correction is applied. Both are recorded on every claim in this JSON, "
        "with the two-tailed value and the Holm-adjusted value beside the "
        "printed one. Neither is offered as a replacement for a printed "
        "number; both are the checks CONVENTIONS.md asks for.",

        "This job recomputes five values. A sixth INRIA number in the same "
        "appendix paragraph, the upper-bound mIoU range [0.74, 0.81] across "
        "cities, is reported in extra_printed_value_not_in_scope and is NOT "
        "adjudicated: the battery file's f3v2.miou is 0.8060 austin, 0.7382 "
        "chicago and 0.5686 vienna, which covers the printed range on two "
        "cities and not on the third, but the file does not label that field "
        "as the upper bound and the generator that would settle it is not on "
        "this box.",

        "No figure is produced and no number here is rounded to match a "
        "printed value. The printed values are carried verbatim as the paper "
        "prints them and every recomputed value is carried at full precision "
        "beside a copy rounded only for the agreement test.",
    ]

    submitted = {}
    for claim in CLAIMS:
        for rel in claim["printed_in"]:
            submitted.setdefault(rel, None)
    needles = {
        "sections/evaluation.tex": "sign test at the strongest result the "
                                   "sample size permits",
        "appendix/app_eval.tex": "The predicted-positive logit erosion "
                                 "transfers in direction on $5$ of $6$ cells",
    }
    submitted_state = {rel: in_submitted(rel, needles[rel])
                       for rel in needles}

    out = dict(
        id="24_inria_crosstask",
        arm="H2",
        answers=["BxRx-W2", "AC1"],
        answers_note="BxRx-W2 is the task-scope weakness this cross-task test "
                     "answers; AC1 is the AC's evidence-for-every-claim note, "
                     "which is why the values have to come from a results file "
                     "before the paper prints them.",
        question=("Where are the INRIA cross-task values printed, what output "
                  "do they come from, and do they recompute with the tests the "
                  "paper names?"),
        status="RECOMPUTED",
        verdict=("every printed value reproduces to the precision it is "
                 "printed at" if all_agree else
                 f"{len(mismatches)} printed values do not reproduce"),
        printed_locations=printed_lines,
        printed_sentences_in_the_submitted_commit=dict(
            commit=SUBMITTED_COMMIT, per_file=submitted_state,
            note="the two files carrying the five values differ from the "
                 "submitted commit overall, so each sentence is checked for "
                 "presence in that commit individually rather than by file "
                 "hash"),
        search_record=search,
        source=dict(
            path=os.path.relpath(BATTERY, SHADE),
            absolute=BATTERY,
            sha256=bat_sha,
            kind="per-cell record: 3 shadow cells and 3 INRIA cells, one row "
                 "per (task, held-out city, signature). Not per-image.",
            probes_present=sorted(bat["shadow"].keys()),
            miou_floor=bat.get("miou_floor"),
            shadow_excluded=bat.get("shadow_excluded"),
            inria_excluded=bat.get("inria_excluded"),
            also_cited_by="results/06_aurc_recovery.json, as "
                          "provenance.published_s1_record"),
        comparisons=comparisons,
        mismatches=mismatches,
        mismatches_note="a mismatch is a disagreement beyond the precision the "
                        "paper prints at; empty means every printed value "
                        "reproduces",
        multiplicity_check=multiplicity,
        signature_redundancy=redundancy,
        extra_printed_value_not_in_scope=extra,
        conditions=[dict(
            label=f"{c['claim']} / 6 cells / cross-task direction",
            metric=c["quantity"],
            mean=c["recomputed"]["mean"],
            stderr=float(c["recomputed"]["sd_ddof1"] / np.sqrt(6)),
            n=6, n_images_per_cell=[v["n_images"]
                                    for v in c["recomputed"]["values"]],
            ci95=None,
            ci_method="none; the printed family reports no interval and the "
                      "per-image inputs needed to form one do not exist on "
                      "this box",
            baseline="the cell's own within-city upper bound, inside each "
                     "per-cell statistic",
            cohens_d=c["recomputed"]["cohens_d_one_sample"])
            for c in comparisons],
        tests=[dict(
            name=("Wilcoxon signed-rank, one-tailed"
                  if c["claim"] == "A1_logit_asymmetry"
                  else "sign test (exact binomial), one-tailed"),
            compares=[c["claim"], "no cross-task direction"],
            statistic=c["recomputed"]["mean"],
            p=(c["recomputed"]["wilcoxon_p_one_tailed"]
               if c["claim"] == "A1_logit_asymmetry"
               else c["recomputed"]["sign_test_p_one_tailed"]),
            p_two_tailed=(c["recomputed"]["wilcoxon_p_two_tailed"]
                          if c["claim"] == "A1_logit_asymmetry"
                          else c["recomputed"]["sign_test_p_two_tailed"]),
            p_holm=next(e["p_holm"] for e in multiplicity["per_claim"]
                        if e["claim"] == c["claim"]),
            B=None, two_tailed=False, n=6,
            correction="none as printed; Holm-Bonferroni over the family of "
                       "five carried beside it")
            for c in comparisons],
        runs=[],
        runs_note="no run_id is involved: no INRIA dump exists in CONTRACT.md's "
                  "schema on this box and this job reads none. The six cells "
                  "are identified by (task, held-out city) as the battery file "
                  "names them.",
        runs_excluded=[],
        source_artifacts=[
            dict(path=os.path.relpath(BATTERY, SHADE), sha256=bat_sha,
                 role="the one INRIA output on this box; every per-cell value "
                      "here is read from it"),
            dict(path="final_paper/appendix/app_eval.tex",
                 role="the appendix subsection printing four of the five "
                      "values"),
            dict(path="final_paper/sections/evaluation.tex",
                 role="section 4.3, printing the first two values"),
            dict(path="results/06_aurc_recovery.json",
                 role="the one results JSON already citing the battery file; "
                      "read for provenance, no number taken from it"),
            dict(path="Shade Maps/Python/FINAL-fig3.ipynb",
                 role="reads the battery file to draw the figure the main text "
                      "points at; does not write it"),
        ],
        index_sha256=bat_sha,
        index_sha256_note="no INDEX.parquet was read or built; this job reads "
                          "no dump tree. The value is the sha256 of the one "
                          "source file every number comes from.",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/24_inria_crosstask.py",
        env=dict(python=sys.version.split()[0],
                 conda_env="myenv "
                           "(conda env)",
                 pythonnousersite=os.environ.get("PYTHONNOUSERSITE"),
                 numpy=np.__version__, scipy=stats.__name__ and __import__(
                     "scipy").__version__),
        seed=None,
        seed_note="no stochastic step runs here; the sign test and the "
                  "Wilcoxon signed-rank test are exact at n = 6",
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; "
                        "only final_paper/ is git-connected and this job only "
                        "reads it",
        compute_note="read-only CPU analysis on the analysis host. No training, no GPU "
                     "inference, no dump read, nothing under dumps/, results/ "
                     "or final_paper/ modified.",
        headline=headline,
        sanity_flags=sanity,
        caveats=caveats,
        figure=None,
        caption=None,
    )

    print("\n" + hr)
    print(f"mismatches: {len(mismatches)}")
    for m in mismatches:
        print(f"  {m}")
    print("headline: " + headline)
    print(f"caveats {len(caveats)}   sanity flags {len(sanity)}")
    print(hr)

    if DRY_RUN:
        print("\nDRY_RUN: nothing written.")
        return

    outpath = os.path.join(BASE, "results", OUT_NAME)
    if os.path.exists(outpath):
        fail(f"{outpath} already exists; this job writes exactly one new file")

    def enc(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")

    tmp = outpath + ".new"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=enc, allow_nan=False)
    os.replace(tmp, outpath)
    print(f"\nwrote {outpath}")
    print(f"  sha256 {sha256_file(outpath)}")


if __name__ == "__main__":
    main()
