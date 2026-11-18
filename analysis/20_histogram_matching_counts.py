#!/usr/bin/env python
"""
Histogram-matching counts, internal and external, promoted into results/.

Answers AC1, v1UJ-W4, zNBu-W2, v1UJ-W3.

Section 4.2's feature-space finding rests in part on histogram matching failing
to recover the LOCO gap. Two independent counts exist: arm C's internal probe
over the published scaling ladder, and arm A's external S-EO probe. The external
count is already in results/08_seo_full_grid.json; the internal count is not in
any results JSON, so the pair cannot be cited together. This job promotes both
into results/20_histogram_matching_counts.json as two separate fields with their
own sources, plus their sum in a third field explicitly labelled as a sum. NN 20
is the next free results number after 19.

Nothing is recomputed. The one arithmetic operation is the sum the operator's
brief permits, and the field says so in its own name and note.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/20_histogram_matching_counts.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _promote_lib import (env_block, fail, load_json, read_text,  # noqa: E402
                          source, write_results)

DRY_RUN = "--dry-run" in sys.argv

MANIFEST = "logs/armC_probes_manifest.json"
PAPER_STATE = "PAPER_STATE.md"
SEO = "results/08_seo_full_grid.json"


def main():
    man = load_json(MANIFEST)
    ps_text = read_text(PAPER_STATE)
    seo = load_json(SEO)
    t5 = seo["task5"]
    hl = t5["headline"]
    comb = hl["combined_with_armC_internal"]

    probe1 = [r for r in man["runs"] if r["probe"] == 1]
    if man["by_probe"]["1"] != len(probe1):
        fail(f"manifest disagrees with itself on the probe-1 count: "
             f"by_probe 1 = {man['by_probe']['1']}, runs = {len(probe1)}")
    if len(probe1) != 27:
        fail(f"expected 27 arm C histogram-matching dumps, manifest holds "
             f"{len(probe1)}")

    # the internal verdict itself lives in PAPER_STATE.md's arm C row; the
    # manifest supplies the run_ids, not the count of improving cells
    m = re.search(r"^\| \*\*C — §4\.2 probe\*\* \| (?P<body>.*?) \| "
                  r"`logs/armC_probes_manifest\.json`.*?\|$", ps_text, re.M)
    if not m:
        fail("could not locate the arm C §4.2 probe row in " + PAPER_STATE)
    ps_row = m.group("body").strip()
    if "0 of 26 usable cells improve" not in ps_row:
        fail("the arm C §4.2 probe row no longer carries the 0-of-26 verdict")
    rung_r = dict(re.findall(r"(\d{3}) (−[\d.]+)", ps_row))
    if sorted(rung_r) != ["225", "450", "900"]:
        fail(f"could not recover the per-rung mean R from the arm C row: "
             f"{rung_r}")

    if comb["armC_usable"] != 26 or comb["armC_improving"] != 0:
        fail(f"results/08's record of the internal count changed: {comb}")
    if hl["n_cells"] != 36 or hl["n_improving_miou"] != 0:
        fail(f"results/08's external count changed: {hl}")

    internal_usable, internal_improving = comb["armC_usable"], comb["armC_improving"]
    external_usable, external_improving = hl["n_cells"], hl["n_improving_miou"]

    out = dict(
        id="20_histogram_matching_counts",
        arm="C + A",
        answers=["AC1", "v1UJ-W4", "zNBu-W2", "v1UJ-W3"],
        question="Under the published grayscale histogram-matching rule, how "
                 "many cells improve, internally on the scaling ladder and "
                 "externally on the S-EO grid?",
        status="PROMOTED_EVIDENCE",
        promotion_note="This JSON computes nothing except the one sum the brief "
                       "permits, which is carried in a field whose name and "
                       "note both say it is a sum. The internal and external "
                       "counts are kept in separate fields with separate "
                       "sources and are never merged into a single reported "
                       "quantity.",
        rule=t5["rule"],
        internal=dict(
            scope="arm C, the published ShadowTransfer scaling ladder, "
                  "vanilla LOCO cells at rungs 225, 450 and 900, high "
                  "resolution",
            n_cells_usable=internal_usable,
            n_cells_improving=internal_improving,
            n_dumps_written=len(probe1),
            n_dumps_note="the manifest names 27 *-p1histmatch dumps; one is "
                         "degenerate and excluded, which is why the usable "
                         "denominator is 26 and not 27",
            mean_recovery_ratio_R_by_rung={k: v for k, v in rung_r.items()},
            mean_recovery_ratio_R_note="R as PAPER_STATE.md's arm C row prints "
                                       "it, with the minus signs the file uses; "
                                       "rung 225 is 8 cells after the "
                                       "degenerate exclusion",
            verdict_verbatim=ps_row,
            count_source=dict(
                path=PAPER_STATE,
                locator="the 'C — §4.2 probe' row",
                why="the manifest supplies the probe's 27 run_ids and their "
                    "parents and comparators, not the number of cells that "
                    "improved. The 0-of-26 verdict is PAPER_STATE.md's arm C "
                    "row, which is also the source results/08 cites for it."),
            run_ids=[r["run_id"] for r in probe1],
            run_id_parents={r["run_id"]: dict(parent=r["parent"],
                                              comparator=r["comparator"])
                            for r in probe1}),
        external=dict(
            scope="arm A, the S-EO grid, vanilla cells at rung 3500, high "
                  "resolution, three folds",
            n_cells_usable=external_usable,
            n_cells_improving=external_improving,
            n_cells_improving_foreground_only=hl["n_improving_fg"],
            n_cells_improving_and_significant_holm36=hl[
                "n_improving_holm36_sig"],
            n_cells_degrading_and_significant_holm36=hl[
                "n_degrading_holm36_sig"],
            by_arch=hl["by_arch"],
            per_cell=t5["headline_cells"],
            method_family=dict(
                summary=t5["method_family"]["summary"],
                mixed_denominator_disclosures=t5["method_family"][
                    "mixed_denominator_disclosures"],
                note=t5["method_family"]["note"]),
            e200_probes=t5["e200_probes"],
            deferred_or_failed=t5["deferred_or_failed"],
            count_source=dict(path=SEO, locator="task5.headline")),
        sum_of_the_two=dict(
            is_a_sum=True,
            label="internal plus external, as a sum of two separately measured "
                  "counts",
            n_cells_usable=internal_usable + external_usable,
            n_cells_improving=internal_improving + external_improving,
            as_printed=comb["combined"],
            arithmetic=f"{internal_improving} + {external_improving} improving "
                       f"of {internal_usable} + {external_usable} usable",
            agrees_with_results_08=(comb["combined"]
                                    == f"{internal_improving + external_improving}"
                                       f" of "
                                       f"{internal_usable + external_usable}"),
            what_it_is_not="not a pooled estimate and not a test over 62 "
                           "cells. The two halves are different benchmarks, "
                           "different rungs, different test-pool sizes and "
                           "different architectures in part, and the external "
                           "half carries Holm-36 significance while the "
                           "internal half carries none. The sum is reportable "
                           "as a count and nothing else."),
        conditions=[
            dict(label="internal (arm C ladder) / cells improving under the "
                       "published grayscale matching rule",
                 metric="n_cells_improving_under_histogram_matching",
                 mean=internal_improving, stderr=None,
                 n=internal_usable, n_unit="usable cells",
                 ci95=None,
                 ci_method="none; a count, and the source computed no interval "
                           "or test for it",
                 baseline="the same cells' unmatched re-inference"),
            dict(label="external (arm A S-EO) / cells improving under the "
                       "published grayscale matching rule",
                 metric="n_cells_improving_under_histogram_matching",
                 mean=external_improving, stderr=None,
                 n=external_usable, n_unit="cells",
                 ci95=None,
                 ci_method="none; a count. The per-cell deltas behind it do "
                           "carry Holm-36 significance, recorded in external."
                           "n_cells_degrading_and_significant_holm36",
                 baseline="each cell's own in-run unmatched re-inference"),
            dict(label="sum of the two counts",
                 metric="n_cells_improving_under_histogram_matching",
                 mean=internal_improving + external_improving, stderr=None,
                 n=internal_usable + external_usable,
                 n_unit="cells, summed across two benchmarks",
                 ci95=None, ci_method="none; this field is a sum, not an "
                                      "estimate",
                 baseline=None, is_a_sum=True)],
        tests=[dict(
            name="Holm-Bonferroni over the 36 external cells",
            compares=["matched pass", "in-run unmatched re-inference"],
            statistic=f"{hl['n_degrading_holm36_sig']} of {external_usable} "
                      "cells degrade significantly; "
                      f"{hl['n_improving_holm36_sig']} improve significantly",
            p=None,
            p_absent_reason="the per-cell p-values are in results/08's own "
                            "task5 block and are not flattened here",
            correction="Holm-Bonferroni over 36 tests"),
            dict(name="none for the internal half",
                 compares=["matched pass", "unmatched comparator"],
                 p=None,
                 p_absent_reason="arm C's probe reports a per-rung mean "
                                 "recovery ratio and an improve/not-improve "
                                 "verdict per cell, with no interval, test or "
                                 "multiplicity correction")],
        runs=sorted([r["run_id"] for r in probe1]
                    + [c["probe"] for c in t5["headline_cells"]]),
        runs_excluded=[dict(
            run_id="one arm C rung-225 histogram-matching cell",
            reason="degenerate; excluded by arm C, which is why the usable "
                   "internal denominator is 26 of 27 dumps. PAPER_STATE.md's "
                   "arm C row records the exclusion but not which cell it is, "
                   "and the manifest does not carry a per-cell verdict, so the "
                   "cell cannot be named from either source on this box.")]
        + [dict(run_id=d["probe"], reason=d["reason"])
           for d in t5["deferred_or_failed"]],
        index_sha256=None,
        index_sha256_reason="the internal count comes from a manifest and a "
                            "state file, the external count from an existing "
                            "results JSON; no dump tree is read",
        source_artifacts=[
            source(MANIFEST, "the 27 arm C histogram-matching run_ids with "
                             "their parents and comparators"),
            source(PAPER_STATE, "the arm C §4.2 probe row, which carries the "
                                "0-of-26 verdict and the per-rung mean "
                                "recovery ratio"),
            source(SEO, "task 5: the external 36-cell count, its per-cell "
                        "rows, the method-family probes and the deferred "
                        "cells"),
        ],
        script="analysis/20_histogram_matching_counts.py",
        figure=None,
        caption=None,
        seed=0,
        seed_note="no stochastic step; every value is a count or a per-cell row "
                  "copied from a source file",
        headline=(
            f"Under the published grayscale histogram-matching rule "
            f"{internal_improving} of {internal_usable} usable internal "
            f"scaling-ladder cells improve and {external_improving} of "
            f"{external_usable} external S-EO cells improve, and the sum of "
            f"the two counts is {comb['combined']} cells."),
        sanity_flags=[
            "Two independently measured counts both landing on exactly zero "
            "improving cells is the kind of result CONVENTIONS.md asks to be "
            "flagged. Three things make it less surprising than it looks: the "
            "external half additionally reports 36 of 36 cells degrading "
            "significantly after Holm-36, so the zero is the tail of a "
            "one-directional effect rather than a coin landing on its edge; "
            "the internal half is a verdict over a mean recovery ratio that is "
            "negative at every rung (-0.418, -0.583, -0.674); and the two "
            "halves share the published matching rule and its grayscale "
            "definition, so they are not fully independent tests of the same "
            "hypothesis.",
            "The internal denominator is 26 while the manifest holds 27 dumps. "
            "The excluded cell is degenerate, and neither source on this box "
            "names it, so a reader cannot check that exclusion from this JSON.",
            "results/16_armL_ssl_regime.json reports a different count on a "
            "different grid: 7 of 36 arm L cells improve under the same "
            "published rule at two-class K=2, and 4 of 36 foreground-only. "
            "That grid is not part of either half here and must not be folded "
            "into the sum.",
            "Promotion only: if any number here disagrees with its source "
            "file, the source file is right and this JSON is stale."],
        caveats=[
            "The internal and external counts are two separate fields on "
            "purpose. The sum field is a sum of counts, labelled as one, and "
            "is not a pooled estimate: the two halves differ in benchmark, "
            "rung (225/450/900 against 3500), test-pool size, architecture mix "
            "and significance treatment.",
            "The internal 0-of-26 verdict is not recomputed here and cannot be "
            "recomputed on this box. results/08 cites the same source for it "
            "(PAPER_STATE.md's arm C row) and records that the internal count "
            "is not recomputed there either, so this JSON inherits that "
            "limitation rather than removing it.",
            "The external half's method-family probes are a different quantity "
            "and are carried separately: their recovery ratio is a method "
            "numerator over a vanilla denominator, because methods are "
            "LOCO-only project-wide and have no method upper cell, so they "
            "borrow the matching fold and seed's vanilla upper. The source "
            "forbids pooling them into the headline count and they are not "
            "pooled here.",
            "39 external probes are deferred or failed, almost all of them SAR "
            "cells that need a build running test-time adaptation on the "
            "matched images. Those cells are in neither count.",
            "Arm C's internal probe carries no interval, no p-value and no "
            "multiplicity correction. Anything the paper says about the "
            "internal half's significance needs a new analysis.",
            "Both halves are high resolution only. Neither covers the "
            "mid-resolution cells.",
            "The arm A external probe's headline uses the published "
            "grayscale-plus-ratio matched value (miou_tol_pct_matched_gray). "
            "The S-EO method-family dumps also carry a per-channel matched "
            "variant, and the two are different interventions; only the "
            "published grayscale rule is counted.",
        ],
        **env_block())

    write_results(out, "20_histogram_matching_counts", DRY_RUN)


if __name__ == "__main__":
    main()
