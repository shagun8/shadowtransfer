#!/usr/bin/env python
"""
Appendix E placeholder -- the empty-ground-truth-foreground rate, promoted into
results/.

Answers BxRx-W5, zNBu-M4.

Appendix E (L731) carries an unfinished "X.X%" placeholder for the share of test
images whose eroded evaluation band holds no ground-truth shadow. The number was
measured by arm H and written up in summaries/empty_denominator_rate.md, and arm
D's pre-grid gate measured the same quantity per cell in
runs/D_seedfloor/seed_floor.md. Neither is a results JSON, so CS cannot cite
either. This job promotes both into
results/14_appE_empty_foreground_rate.json. NN 14 is the next free results
number after 13.

Nothing is recomputed. Both sources are markdown, so their table cells are
carried as the strings the summaries print and each file is also carried
verbatim, which makes a silent re-rounding impossible.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/14_appE_empty_foreground_rate.py
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _promote_lib import (env_block, fail, read_text, source,  # noqa: E402
                          write_results)

DRY_RUN = "--dry-run" in sys.argv

EMPTY = "summaries/empty_denominator_rate.md"
SEEDFLOOR = "runs/D_seedfloor/seed_floor.md"


def md_tables(text):
    out, title, header, rows = [], None, None, []

    def flush():
        if header:
            out.append(dict(title=title, header=header, rows=rows))
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            flush()
            title, header, rows = s.lstrip("#").strip(), None, []
            continue
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if header is None:
                header = cells
            else:
                rows.append(cells)
        elif s and header:
            flush()
            header, rows = None, []
    flush()
    return [t for t in out if t["rows"]]


def main():
    empty_text = read_text(EMPTY)
    sf_text = read_text(SEEDFLOOR)
    empty_tables = md_tables(empty_text)
    sf_tables = md_tables(sf_text)

    pooled = next((t for t in empty_tables if t["header"][:2] == ["Band",
                                                                 "Empty images"]),
                  None)
    per_split = next((t for t in empty_tables
                      if t["header"][:2] == ["City", "Res"]), None)
    coverage = next((t for t in empty_tables if t["header"] == ["", "Count"]),
                    None)
    sf_empty = next((t for t in sf_tables
                     if t["header"][:3] == ["arch", "city", "condition"]), None)
    for name, t in (("pooled", pooled), ("per-split", per_split),
                    ("coverage", coverage), ("seed-floor empty-fg", sf_empty)):
        if t is None:
            fail(f"could not locate the {name} table in its source markdown")
    if len(pooled["rows"]) != 3:
        fail(f"pooled table has {len(pooled['rows'])} rows, expected 3 bands")
    if len(per_split["rows"]) != 6:
        fail(f"per-split table has {len(per_split['rows'])} rows, expected 6")
    if len(sf_empty["rows"]) != 54:
        fail(f"seed-floor empty-fg table has {len(sf_empty['rows'])} rows, "
             "expected 54")

    m = re.search(r"^> (Of the 900 test images.*?)$", empty_text,
                  re.M | re.S)
    if not m:
        fail("could not locate the Appendix E sentence in "
             + EMPTY)
    app_e_sentence = m.group(1).strip()

    # the two sources must agree on the three high-resolution splits; this is a
    # check on the copy, not a recomputation
    sf_by_city = {}
    for arch, city, cond, n_empty, n_test, pct in sf_empty["rows"]:
        sf_by_city.setdefault(city, set()).add((n_empty, n_test, pct))
    split_k2 = {}
    for city, res, n, k0, k2, k5, cells in per_split["rows"]:
        split_k2[(city, res)] = k2.strip("*")
    for city, vals in sf_by_city.items():
        if len(vals) != 1:
            fail(f"seed_floor.md reports more than one empty-fg value for "
                 f"{city}: {vals}")
        (n_empty, n_test, pct), = vals
        if pct != split_k2[(city, "hr")]:
            fail(f"{city}: seed_floor.md says {pct} at K=2 and "
                 f"empty_denominator_rate.md says {split_k2[(city, 'hr')]}")

    conditions = []
    for band, n_empty, n, rate in pooled["rows"]:
        conditions.append(dict(
            label=f"pooled over the LOCO benchmark / band {band}",
            metric="empty_gt_foreground_on_valid_region_rate",
            mean=rate, mean_is_a_string=True,
            empty_images=n_empty, n=n, n_unit="distinct test images",
            stderr=None, ci95=None,
            ci_method="none; the quantity is a count over a fixed image set, "
                      "not an estimate",
            baseline=None))
    for city, res, n, k0, k2, k5, cells in per_split["rows"]:
        conditions.append(dict(
            label=f"split {city}/{res} / band K=2",
            metric="empty_gt_foreground_on_valid_region_rate",
            mean=k2.strip("*"), mean_is_a_string=True,
            n=n, n_unit="test images", k0=k0, k5=k5, cells_on_split=cells,
            stderr=None, ci95=None,
            ci_method="none; a count over a fixed image set",
            baseline=None))

    out = dict(
        id="14_appE_empty_foreground_rate",
        arm="H",
        answers=["BxRx-W5", "zNBu-M4"],
        question="What share of the benchmark's test images has an eroded "
                 "evaluation band containing no ground-truth shadow pixels, so "
                 "that their per-image foreground IoU is undefined?",
        status="PROMOTED_EVIDENCE",
        promotion_note="This JSON computes nothing. Both sources are markdown "
                       "summaries and their table cells are carried as the "
                       "strings those files print, with each file also carried "
                       "verbatim. The script's checks are that the two sources "
                       "agree on the three high-resolution splits and that each "
                       "table has the expected number of rows; neither is a new "
                       "statistic.",
        metric=dict(
            name="empty_gt_foreground_on_valid_region_rate",
            definition="the share of test images for which the eroded "
                       "evaluation band at band width K holds zero "
                       "ground-truth shadow pixels, so the per-image "
                       "foreground IoU has a zero denominator",
            bands=[0, 2, 5],
            depends_on_model=False,
            depends_on_model_note="the band is a function of the ground-truth "
                                  "mask and the band width alone, so the "
                                  "quantity does not vary with architecture or "
                                  "method and the 144-cell grid carries only "
                                  "six distinct values, one per (city, "
                                  "resolution) split",
            k0_note="K=0 is not an erosion (metrics.py uses a 1x1 kernel), so "
                    "the K=0 count is the number of test images with no "
                    "ground-truth shadow at all",
            mask_read="cv2.IMREAD_GRAYSCALE, foreground = raw > 127, as the "
                      "dump generator reads them"),
        appendix_E_sentence=app_e_sentence,
        appendix_E_sentence_note="the sentence as written for Appendix E in "
                                 + EMPTY + "; it is the authored wording, not "
                                 "something this job composed",
        pooled=dict(header=pooled["header"], rows=pooled["rows"],
                    n_distinct_test_images=900,
                    n_distinct_note="900 = 6 splits x 150; the 144 cells "
                                    "divide uniformly at 24 per split, so "
                                    "pooling over all 144 x 150 = 21600 "
                                    "cell-image evaluations gives the identical "
                                    "rate"),
        per_split=dict(header=per_split["header"], rows=per_split["rows"]),
        coverage=dict(header=coverage["header"], rows=coverage["rows"]),
        per_cell_from_arm_D_seed_floor=dict(
            header=sf_empty["header"], rows=sf_empty["rows"],
            n_rows=len(sf_empty["rows"]),
            note="arm D's pre-grid gate measured the same quantity on the 54 "
                 "arm C rung-450 high-resolution cells (3 architectures x 3 "
                 "cities x {loco, upper} x 3 seeds). Every cell on a given city "
                 "carries that city's value exactly, which is the expected "
                 "behaviour for a mask-only quantity and is what makes it a "
                 "check rather than 54 measurements."),
        agreement_between_sources=dict(
            checked="the K=2 rate per high-resolution city",
            chicago="7.33% in both sources",
            miami="25.33% in both sources",
            phoenix="14.00% in both sources",
            all_agree=True),
        other_source_tables=dict(
            empty_denominator_rate_md=empty_tables,
            seed_floor_md=sf_tables),
        verbatim=dict(empty_denominator_rate_md=empty_text,
                      seed_floor_md=sf_text),
        conditions=conditions,
        tests=[dict(
            name="monotonicity check, K=0 <= K=2 <= K=5",
            compares=["band K=0", "band K=2", "band K=5"],
            result="pass on every split", p=None,
            p_absent_reason="an ordering check over counts, not a hypothesis "
                            "test"),
            dict(name="border-convention sensitivity",
                 compares=["cv2 erosion (canonical, metrics.py)",
                           "scipy erosion with border_value=0"],
                 statistic="1 of 900 images flips at K=2, 28 at K=5, always "
                           "toward more empties",
                 p=None,
                 p_absent_reason="a sensitivity count, not a hypothesis test",
                 note="cv2 is canonical per CONTRACT.md rule zero. The K=2 "
                      "headline is robust to this choice; the K=5 figure is "
                      "not.")],
        runs=sorted({f"{a}-{t}-r450-{c}-hr-s{s}"
                     for a in ("mamnet", "oglanet", "dinov3")
                     for t in ("vanilla-loco", "none-upper")
                     for c in ("chicago", "miami", "phoenix")
                     for s in (0, 1, 2)}),
        runs_note="the 54 arm C rung-450 high-resolution cells arm D's gate "
                  "read. The pooled and per-split numbers rest on the "
                  "ground-truth masks of the 900 distinct test images and on "
                  "the 144-cell manifest logs/p0_manifest.json, not on these "
                  "dumps.",
        runs_excluded=[dict(
            run_id="oglanet-none-upper-r450-miami-hr-s0",
            reason="the arm H pass found this dump directory present but "
                   ".FAILED with no perimage.parquet, so it carries the "
                   "GT-direct value for its split rather than a dump-backed "
                   "one. The cell has since been scored from its prediction "
                   "PNG masks (dumps/st_r450_published_png), which the arm H "
                   "pass predates.")],
        index_sha256=None,
        index_sha256_reason="both sources are markdown summaries of work done "
                            "on the SLURM box; no INDEX.parquet enters this job",
        source_artifacts=[
            source(EMPTY, "the arm H measurement: pooled, per-split, per-cell "
                          "coverage, sanity checks and the Appendix E sentence"),
            source(SEEDFLOOR, "arm D's pre-grid gate, which measured the same "
                              "quantity on 54 cells"),
        ],
        script="analysis/14_appE_empty_foreground_rate.py",
        figure=None,
        caption=None,
        seed=0,
        seed_note="no stochastic step; every value is a count copied from a "
                  "summary",
        headline=(
            "Of the 900 distinct test images in the LOCO benchmark, 117 "
            "(13.00%) have an eroded evaluation band containing no "
            "ground-truth shadow pixels at K=2, against 111 (12.33%) at K=0 "
            "and 160 (17.78%) at K=5, and the per-split K=2 rate ranges from "
            "2.00% (phoenix/mr) to 25.33% (miami/hr)."),
        sanity_flags=[
            "Flagged as too clean in the source and carried here: the pooled "
            "K=2 count is 117 of 900, which lands on exactly 13.00%. That is "
            "arithmetic coincidence of the six per-split counts "
            "(11 + 10 + 38 + 34 + 21 + 3), not a rounded or imputed figure.",
            "The border convention moves the K=5 number materially and the "
            "K=2 number barely: recomputing the band with scipy and "
            "border_value=0 instead of cv2's default flips the empty verdict "
            "on 1 of 900 images at K=2 and 28 at K=5, always toward more "
            "empties, because cv2 treats outside-image as foreground during "
            "erosion. The K=2 headline is robust to the choice; the K=5 figure "
            "is not.",
            "Every cell on a given (city, resolution) split carries that "
            "split's value exactly, in both sources independently. That is the "
            "expected behaviour for a mask-only quantity, so the 144-cell and "
            "54-cell tables are one measurement repeated, not 144 or 54 "
            "measurements.",
            "Promotion only: if any number here disagrees with its source "
            "file, the source file is right and this JSON is stale."],
        caveats=[
            "Both sources are markdown. The arm H summary names "
            "results/empty_denominator_rate.json as its own source, and that "
            "file does not exist on the analysis host, so the full 144-cell per_cell table "
            "it refers to is not reachable from this box. What is promoted here "
            "is the pooled table, the six per-split values, the coverage table "
            "and arm D's independent 54-cell measurement.",
            "The quantity is a property of the ground-truth masks and the band "
            "width alone. It does not vary with architecture or method, so "
            "nothing here is evidence about any model.",
            "Coverage at the time of the arm H pass: 17 of 144 cells had a "
            "usable regenerated dump and agreed exactly with the GT-direct "
            "recomputation on n, image-id set and empty count at all three "
            "bands; the other 127 carried the GT-direct value for their own "
            "split and no dump was fabricated. One dump directory was "
            "unusable, oglanet-none-upper-r450-miami-hr-s0.",
            "K=0 is not an erosion, so the K=0 row answers a different "
            "question from the K=2 and K=5 rows: it counts images with no "
            "ground-truth shadow anywhere, not images whose eroded band is "
            "empty.",
            "Appendix E's placeholder sits in a sentence about the "
            "foreground-only average. The published tables are not "
            "foreground-only (erratum E-7), so the share of images this "
            "quantity excludes from a foreground-only mean is not the share it "
            "affects in any printed table. Table 38 and Tables 30-32 are "
            "two-class averages in which an empty-foreground image scores "
            "exactly 50 under the Appendix E degenerate rule rather than being "
            "dropped.",
            "Both sources cover the published LOCO grid. They say nothing "
            "about the external datasets or about any arm whose test pool "
            "differs from the 150-image splits.",
        ],
        **env_block())

    write_results(out, "14_appE_empty_foreground_rate", DRY_RUN)


if __name__ == "__main__":
    main()
