#!/usr/bin/env python
"""
E-12 seed table -- the countersigned MAMNet-SIB clean-pool retrains, promoted
into results/.

Answers E-12, BxRx-W3.

Erratum E-12 found that the published MAMNet-SIB cells' FDA pool walked the city
root, so the pool was 750 images plus 1,500 label-mask PNGs and a 100-draw took
roughly 33 images and 67 binary masks. The three cells were retrained under the
recipe as described, three seeds each, and the HLT countersigned the resulting
table in summaries/HLT_COUNTERSIGNS_armA_e12.md. That file is drafting authority,
not a results JSON, so CS cannot cite its numbers. This job promotes the final
table and the G1 refinement into results/19_e12_seed_table.json. NN 19 is the
next free results number above the 15-18 block the arm D, arm L and arms G/J
figures already own.

Nothing is recomputed. The source is markdown, so its table cells are carried as
the strings it prints and the file is also carried verbatim.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/19_e12_seed_table.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _promote_lib import (env_block, fail, read_text, source,  # noqa: E402
                          write_results)

DRY_RUN = "--dry-run" in sys.argv

HLT = "summaries/HLT_COUNTERSIGNS_armA_e12.md"


def main():
    text = read_text(HLT)

    # the FINAL table, located by its own heading rather than by position
    m = re.search(r"^## E-12 seed table — FINAL \((?P<when>[^)]*)\)\n"
                  r"(?P<preamble>(?:(?!\|).*\n)*)"
                  r"(?P<table>(?:\|.*\n)+)", text, re.M)
    if not m:
        fail("could not locate the FINAL E-12 seed table in " + HLT)
    lines = [l for l in m.group("table").strip().splitlines()]
    cells = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    header = cells[0]
    # drop the markdown separator row, which holds only dashes, colons and
    # spaces; every other row is data
    rows = [r for r in cells[1:] if not set("".join(r)) <= set("-: ")]
    if len(rows) != 3:
        fail(f"the FINAL seed table has {len(rows)} data rows, expected 3 cities")
    if [r[0] for r in rows] != ["phoenix", "miami", "chicago"]:
        fail(f"unexpected city order in the FINAL seed table: "
             f"{[r[0] for r in rows]}")

    g1 = re.search(r"^## G1 REFINEMENT under the seed mean\n(?P<body>(?:.|\n)*)",
                   text, re.M)
    if not g1:
        fail("could not locate the G1 REFINEMENT section in " + HLT)
    g1_body = g1.group("body").strip()
    if "7.48" not in g1_body or "7.18" not in g1_body:
        fail("the G1 REFINEMENT section no longer carries the 7.48 and 7.18 "
             "margins this job promotes")

    e12 = re.search(r"^## E-12 disclosure \(§5\.1/App H\)\n(?P<body>(?:- .*\n|  .*\n)+)",
                    text, re.M)
    if not e12:
        fail("could not locate the E-12 disclosure section in " + HLT)

    # Parse the three cities' numbers into typed fields. The strings stay beside
    # them so a reader can see what was parsed; nothing is rounded or derived.
    parsed = []
    for city, two_class, pub2, fg, pubfg in rows:
        mm = re.match(r"([\d.]+)\s*±\s*([\d.]+)", two_class)
        mf = re.match(r"([\d.]+)\s*±\s*([\d.]+)", fg)
        if not (mm and mf):
            fail(f"{city}: could not parse the seed-table cells "
                 f"{two_class!r} / {fg!r}")
        parsed.append(dict(
            city=city,
            two_class_tol_k2_clean_mean=float(mm.group(1)),
            two_class_tol_k2_clean_sd=float(mm.group(2)),
            two_class_tol_k2_published=float(pub2),
            fg_only_tol_k2_clean_mean=float(mf.group(1)),
            fg_only_tol_k2_clean_sd=float(mf.group(2)),
            fg_only_tol_k2_published=float(pubfg),
            clean_minus_published_two_class=float(mm.group(1)) - float(pub2),
            clean_minus_published_fg=float(mf.group(1)) - float(pubfg),
            verbatim=dict(two_class=two_class, published_two_class=pub2,
                          fg_only=fg, published_fg=pubfg)))
    for p in parsed:
        if p["clean_minus_published_two_class"] >= 0:
            fail(f"{p['city']}: the clean-pool mean is not below the published "
                 "value, which contradicts the countersigned printable claim")

    conditions = []
    for p in parsed:
        for key, metric in (("two_class_tol_k2", "miou_2class_tolerant_K2_pct"),
                            ("fg_only_tol_k2", "iou_fg_tolerant_K2_pct")):
            conditions.append(dict(
                label=f"MAMNet-SIB / {p['city']} / clean-pool retrain "
                      f"({metric})",
                metric=metric,
                mean=p[f"{key}_clean_mean"], stderr=None,
                sd=p[f"{key}_clean_sd"], n=3, n_unit="seeds",
                seeds=[42, 43, 44],
                ci95=None,
                ci_method="none; the source reports mean +- across-seed "
                          "standard deviation over three seeds and the "
                          "countersigned claim rests on that spread, not on an "
                          "interval",
                baseline="the published MAMNet-SIB cell, re-evaluated from its "
                         "own checkpoint",
                baseline_value=p[f"{key}_published"],
                clean_minus_published=p[f"clean_minus_published_"
                                        f"{'two_class' if key.startswith('two') else 'fg'}"]))

    out = dict(
        id="19_e12_seed_table",
        arm="A",
        answers=["E-12", "BxRx-W3"],
        question="Under the FDA recipe as described, what does MAMNet-SIB score "
                 "on the three held-out cities, and how does that compare with "
                 "the published cells whose FDA pool was contaminated?",
        status="PROMOTED_EVIDENCE",
        promotion_note="This JSON computes nothing. The source is a "
                       "countersigned markdown table; its cells are carried "
                       "verbatim as strings and parsed into typed fields beside "
                       "them. The two differences per row "
                       "(clean_minus_published) are subtractions of two copied "
                       "numbers, included because the countersigned printable "
                       "claim is about their sign; the script asserts that sign "
                       "rather than discovering it.",
        metric=dict(
            primary="miou_2class_tolerant_K2_pct, the Table 38 family",
            secondary="iou_fg_tolerant_K2_pct",
            published_column="the published checkpoint re-evaluated under the "
                             "same path, which reproduced its own stored "
                             "test_results.json at +0.000000; that exact zero "
                             "is the designed check of the E-12 wave, not a "
                             "result"),
        seed_table=dict(
            table_dated=m.group("when"),
            preamble=m.group("preamble").strip(),
            header=header, rows=rows, parsed=parsed,
            seeds=[42, 43, 44], n_seeds=3,
            n_cells=3,
            scope="MAMNet-SIB high-resolution LOCO cells only; the 18 FDA "
                  "cells are clean, OGLANet-SIB is source_only with a "
                  "cross-source FDA pool and DINOv3-SIB has no FDA"),
        g1_refinement=dict(
            verbatim=g1_body,
            what_it_settles="the internal pooled worst cell under the clean "
                            "pool",
            flip="MAMNet/Phoenix replaces OGLANet/Miami as the worst internal "
                 "cell",
            margin_at_seed_42=7.63,
            margin_at_seed_mean=7.48,
            margin_at_seed_mean_arithmetic="73.75 - 66.2665",
            published_margin_it_displaces=7.18,
            flip_holds_at_every_seed=True,
            citable_form="when paper text touches the internal worst case, "
                         "cite the seed-mean margin of about 7.5 mIoU "
                         "(MAMNet/Phoenix, clean pool, with the seed band), "
                         "not the single-seed 7.63"),
        countersigned_disclosure=dict(
            verbatim=e12.group("body").strip(),
            note="HLT drafting authority: these sentences are fixed and must "
                 "not be paraphrased. Reproduced here so that the results JSON "
                 "carries the wording the numbers may be used with."),
        printable_claims=dict(
            allowed=["all three clean cells sit below their published value",
                     "both phoenix seeds sit below the published value",
                     "the gap is not attributable among pool split, pool "
                     "content and RNG"],
            struck=["any pool-versus-seed ratio sentence (HLT struck it)"],
            over_two_miou_flags="phoenix seeds 42 and 44, miami seeds 43 and "
                                "44, as recorded unsmoothed in those runs' own "
                                "run.json"),
        conditions=conditions,
        tests=[dict(
            name="none",
            compares=["MAMNet-SIB clean-pool retrains", "the published cell"],
            p=None,
            p_absent_reason="the source computes no significance test. Its "
                            "claim is a sign claim over three seeds against a "
                            "single published draw, and the countersigned "
                            "wording says the difference is not attributable "
                            "among pool split, pool content and RNG. A paired "
                            "test over the clean retrains against the published "
                            "cell is in results/12_internal_sib_worst_case_v2."
        )],
        runs=sorted([f"mamnet-sibc4-loco-r450-{c}-hr-s{t}"
                     for c in ("chicago", "miami", "phoenix")
                     for t in ("42-e12clean", "43-e12clean", "44-e12clean",
                               "PUB-e12orig")]),
        runs_note="the nine clean-pool retrains and the three re-evaluations of "
                  "the published checkpoints, as dumped under "
                  "delta_dumps/dumps/e12/",
        runs_excluded=[dict(
            run_id="mamnet-sibc4-loco-r450-{city}-hr-s0 (published)",
            reason="the published MAMNet-SIB cells are contaminated (E-12) and "
                   "are never used as a value; they appear only as the "
                   "published column, re-evaluated from their own checkpoints")],
        index_sha256=None,
        index_sha256_reason="the source is a markdown countersign file; the "
                            "underlying dumps are indexed in "
                            "results/12_internal_sib_worst_case_v2",
        source_artifacts=[
            source(HLT, "the countersigned FINAL E-12 seed table, the E-12 "
                        "disclosure wording and the G1 refinement"),
        ],
        script="analysis/19_e12_seed_table.py",
        figure=None,
        caption=None,
        seed=42,
        seed_note="the retrains' own seeds are 42, 43 and 44; this job runs no "
                  "stochastic step",
        headline=(
            "Under the FDA recipe as described, MAMNet-SIB scores "
            + "; ".join(f"{p['city']} {p['two_class_tol_k2_clean_mean']:.4f} "
                        f"+- {p['two_class_tol_k2_clean_sd']:.4f}"
                        for p in parsed)
            + " two-class tolerant mIoU at K=2 over seeds {42, 43, 44}, each "
              "below its published contaminated value ("
            + "; ".join(f"{p['city']} {p['two_class_tol_k2_published']:.4f}"
                        for p in parsed)
            + "), and the internal pooled worst cell moves to MAMNet/Phoenix "
              "with a seed-mean margin of 7.48 mIoU in place of OGLANet/Miami's "
              "7.18."),
        sanity_flags=[
            "The published column reproduces each cell's own stored "
            "test_results.json at exactly +0.000000. That exact zero is the "
            "designed check of the E-12 wave and is what makes the clean "
            "retrains comparable; it is not a result and should not be read as "
            "one.",
            "Four of the nine clean cells carry a >2 mIoU flag in their own "
            "run.json, unsmoothed: phoenix seeds 42 and 44, miami seeds 43 and "
            "44. The three-seed standard deviations here (0.55 to 0.81 "
            "two-class) are therefore not the whole spread of the retrain "
            "wave.",
            "Three seeds is the smallest number from which a standard "
            "deviation can be quoted at all, and the published side is a "
            "single draw. The direction of every difference is consistent "
            "across cities, which is the countersigned claim; the magnitude of "
            "any one city's difference is not separable from seed noise with "
            "n = 3 against n = 1.",
            "Promotion only: if any number here disagrees with "
            + HLT + ", that file is right and this JSON is stale. Its "
            "sentences are HLT drafting authority and must not be paraphrased."],
        caveats=[
            "Scope is three MAMNet-SIB high-resolution LOCO cells and nothing "
            "else. E-12's own scope note: the 18 FDA cells are clean, "
            "OGLANet-SIB is source_only because its FDA root was never "
            "dereferenced (0 of 111 logs), and DINOv3-SIB has no FDA "
            "component at all.",
            "The contamination is not reconstructible. The published pool's "
            "2,250 candidates were 750 images plus 1,500 label-mask PNGs, a "
            "uniform 100-draw took about 33 images and 67 binary masks, and "
            "roughly 6 to 7 test images entered through the amplitude channel "
            "with P(zero) of about 1e-10. The exact draw cannot be recovered, "
            "so the clean retrain is the recipe as described run for the first "
            "time, not a corrected version of the published run.",
            "The clean-versus-published difference has three candidate causes "
            "at once: the pool split, the pool content and the RNG. The "
            "countersigned wording forbids attributing it to any one of them, "
            "and this JSON carries no attribution.",
            "The published cells are single draws and the retrains are three "
            "seeds, so the two columns are not matched on seed count. No "
            "paired test over the same images is in the source.",
            "The retrains are MAMNet at the published 450-image budget, and "
            "MAMNet-SIB was measured at 450 both published and retrained, so "
            "the 900-image training-count finding that affects MAMNet's "
            "standard and six adaptation cells does not apply to these rows.",
            "High resolution only, rung 450 only, two-class and "
            "foreground-only boundary-tolerant IoU at K=2 only.",
            "The G1 margin of 7.48 is arithmetic the source states (73.75 - "
            "66.2665) against Table 38's printed MAMNet/Phoenix upper bound. "
            "The same margin recomputed on the declared PNG path in "
            "results/12_internal_sib_worst_case_v2 is 7.4850 against a "
            "declared-path upper bound of 73.751570, so the two agree; the "
            "7.48 here is the countersigned figure, not that recomputation.",
        ],
        **env_block())

    write_results(out, "19_e12_seed_table", DRY_RUN)


if __name__ == "__main__":
    main()
