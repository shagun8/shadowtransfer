#!/usr/bin/env python
"""
Arm E -- frozen-backbone DINOv3 decoder-retraining results, PROMOTED.

Answers zNBu-W1, zNBu-W5, AC1.

Two decoder-retraining runs exist on the frozen-backbone DINOv3 cells and the
paper must report both accurately:

  arme_part2          the LITERAL App G.9 Experiment A on the frozen backbone:
                      the decoder is REINITIALIZED. run_ids
                      dinov3frozen-expa-expa-r450-{city}-hr-s0.
  arme_part2_revised  a revised variant: the decoder is LOADED from arm E Part
                      1's own LOCO checkpoint for the same city and then
                      fine-tuned, not reinitialized. run_ids
                      dinov3frozen-adapt-adapt-r450-{city}-hr-s0.

Neither is a reproduction of the published Experiment A (Table 36, DINOv3
R = 0.69 [0.60, 0.75], pooled over cities, decoder reinitialized on a
LOCO-trained encoder).

THIS JOB PROMOTES, IT DOES NOT RECOMPUTE. Every number here is copied from the
two Delta results files. No parquet is read, no R is recomputed, no bootstrap is
run. The per-image R vectors do not exist on any machine (see caveats), so a
recomputation here would not be a check of these numbers but a different
computation, and the brief forbids it.

Read-only. No training, no GPU inference, no checkpoint loaded. Nothing under
dumps/, delta_runs/, delta_logs/ or results/ is modified. Writes exactly one
file: results/23_frozen_backbone_decoder_retrain.json.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/23_frozen_backbone_decoder_retrain.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
OUT_NAME = "23_frozen_backbone_decoder_retrain.json"

CITIES = ("chicago", "miami", "phoenix")
MIN_GAP_EXPECTED = 0.02

SRC = {
    "literal": "delta_runs/arme_part2/results.json",
    "revised": "delta_runs/arme_part2_revised/results.json",
    "part1": "delta_runs/arme_part1/results.json",
    "aggregate_script": "delta_runs/arme_scripts/arme_part2_revised_aggregate.py",
    "log_literal": "delta_logs/session_logs/2026-07-28-arme-part2.md",
    "log_revised": "delta_logs/session_logs/2026-07-28-arme-part2-revised.md",
    "log_part1": "delta_logs/session_logs/2026-07-28-arme-part1.md",
}

# The published number this JSON is NOT comparable to, recorded so the
# non-comparability is machine-readable rather than only prose.
PUBLISHED_TABLE36 = dict(
    table="Table 36 / App G.9",
    arch="dinov3",
    condition="vanilla full fine-tune; decoder REINITIALIZED on a LOCO-trained "
              "encoder",
    pooling="pooled over the three held-out cities",
    median_R=0.69, ci95=[0.60, 0.75], n_valid_images="335-351")

# The per-image R vectors are computed and discarded inside the aggregate
# script. These are the two lines the brief names; both are asserted present
# and verbatim below rather than quoted from memory.
VECTOR_LINES = {90: "R, valid_idx, excluded = compute_per_image_recovery(",
                175: "out_path.write_text(json.dumps(out, indent=2))"}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fail(msg):
    raise SystemExit("ASSERTION FAILED: " + msg)


def load(key):
    p = os.path.join(BASE, SRC[key])
    if not os.path.isfile(p):
        fail(f"{SRC[key]}: input absent; this job promotes it and cannot "
             "proceed without it")
    return json.load(open(p)), sha256_file(p)


# ----------------------------------------------------------------------------
# promotion
# ----------------------------------------------------------------------------

def per_city_rows(doc, cell_key, run_label):
    """Copy, do not recompute, each city's stored R block."""
    rows = []
    by_city = {c["city"]: c for c in doc["per_city"]}
    for city in CITIES:
        if city not in by_city:
            fail(f"{run_label}: no per_city block for {city}")
        c = by_city[city]
        if not c.get("done"):
            fail(f"{run_label}/{city}: source block is not marked done")
        if c["min_gap_threshold"] != MIN_GAP_EXPECTED:
            fail(f"{run_label}/{city}: min_gap_threshold "
                 f"{c['min_gap_threshold']} is not the expected "
                 f"{MIN_GAP_EXPECTED}")
        if c["n_valid_R"] + c["n_excluded_small_gap_or_nan"] != \
                c["n_test_images_joined"]:
            fail(f"{run_label}/{city}: n_valid_R + n_excluded does not sum to "
                 "n_test_images_joined")
        if not (c["R_ci95_lo"] <= c["median_R"] <= c["R_ci95_hi"]):
            fail(f"{run_label}/{city}: the stored 95% CI "
                 f"[{c['R_ci95_lo']}, {c['R_ci95_hi']}] does not contain its "
                 f"own point estimate {c['median_R']}")
        rows.append(dict(
            run=run_label, city=city,
            retrained_cell_run_id=c[cell_key],
            loco_run_id=c["loco_run_id"], upper_run_id=c["upper_run_id"],
            median_per_image_R=c["median_R"],
            R_ci95=[c["R_ci95_lo"], c["R_ci95_hi"]],
            R_ci_method=f"image-level bootstrap of the median, B="
                        f"{c['R_bootstrap_n_boot']}, seed "
                        f"{c['R_bootstrap_seed']}",
            R_bootstrap_B=c["R_bootstrap_n_boot"],
            R_bootstrap_seed=c["R_bootstrap_seed"],
            n_valid=c["n_valid_R"],
            n_excluded=c["n_excluded_small_gap_or_nan"],
            n_excluded_small_gap=c["n_excluded_small_gap"],
            n_excluded_nan=c["n_excluded_nan"],
            n_test_images_joined=c["n_test_images_joined"],
            n_not_joined_across_dumps=c["n_not_joined_across_dumps"],
            min_gap_threshold=c["min_gap_threshold"],
            epochs_run=c.get("epochs_run"),
            decoder_initialized_from_part1_loco_epoch=c.get(
                "decoder_initialized_from_part1_loco_epoch"),
            collapse_detected=c.get("collapse_detected"),
            collapse_start_epoch=c.get("collapse_start_epoch"),
            previous_reinit_attempt_run_id=c.get(
                "previous_reinit_attempt_run_id"),
            previous_reinit_attempt_collapsed=c.get(
                "previous_reinit_attempt_collapsed"),
            previous_reinit_attempt_collapse_start_epoch=c.get(
                "previous_reinit_attempt_collapse_start_epoch"),
            source_field="per_city[] of " + SRC[
                "literal" if run_label == "literal_experiment_a" else
                "revised"]))
    return rows


def main():
    hr = "=" * 100
    print(hr)
    print("Arm E -- frozen-backbone DINOv3 decoder retraining, PROMOTION")
    print(f"DRY_RUN = {DRY_RUN}")
    print(hr)

    lit, lit_sha = load("literal")
    rev, rev_sha = load("revised")
    p1, p1_sha = load("part1")
    agg_path = os.path.join(BASE, SRC["aggregate_script"])
    agg_sha = sha256_file(agg_path)
    shas = {k: sha256_file(os.path.join(BASE, v)) for k, v in SRC.items()}

    # ---- the aggregate script's two named lines, read not remembered -----
    agg_lines = open(agg_path).read().splitlines()
    vector_evidence = []
    for ln, expected in VECTOR_LINES.items():
        if len(agg_lines) < ln:
            fail(f"{SRC['aggregate_script']}: only {len(agg_lines)} lines; "
                 f"line {ln} was named by the brief")
        text = agg_lines[ln - 1].strip()
        if expected not in text:
            fail(f"{SRC['aggregate_script']}:{ln} reads {text!r}, which does "
                 f"not contain the expected {expected!r}; the brief's line "
                 "reference no longer points at the code it described")
        vector_evidence.append(dict(file=SRC["aggregate_script"], line=ln,
                                    text=text))
    print("\n--- the per-image R vector is computed and discarded ---")
    for v in vector_evidence:
        print(f"  {v['file']}:{v['line']}  {v['text']}")

    # ---- the 'spec error' label, quoted verbatim, not adopted ------------
    label_field = "divergence_from_published_experiment_a"
    if label_field not in rev:
        fail(f"{SRC['revised']}: field {label_field} absent; it is the field "
             "that carries the 'spec error' label")
    quote = rev[label_field]
    if "spec error" not in quote:
        fail(f"{SRC['revised']}.{label_field} does not contain the phrase "
             "'spec error'")
    if "spec error" in json.dumps(lit):
        fail(f"{SRC['literal']} also contains the phrase 'spec error'; the "
             "brief states the label lives in the revised file and the "
             "attribution has to be re-read")
    log_rev_lines = open(os.path.join(BASE, SRC["log_revised"])).read(
        ).splitlines()
    log_hits = [dict(file=SRC["log_revised"], line=i + 1,
                     text=l.strip())
                for i, l in enumerate(log_rev_lines) if "spec error" in l]
    spec_error_label = dict(
        labelled_run="literal_experiment_a "
                     "(dinov3frozen-expa-expa-r450-{city}-hr-s0)",
        label="a spec error",
        labelled_by_file=SRC["revised"],
        labelled_by_field=label_field,
        reason_verbatim=quote,
        also_labelled_in=log_hits,
        label_not_adjudicated_here=(
            "This JSON records who applied the label and that file's stated "
            "reason verbatim. It neither adopts nor rejects the label. The "
            "label is a claim about which run implements App G.9's Experiment "
            "A, which is a specification question for the paper's authors, not "
            "a number this box can compute."))
    print("\n--- the 'spec error' label ---")
    print(f"  applied to : {spec_error_label['labelled_run']}")
    print(f"  by         : {SRC['revised']} -> {label_field}")
    print(f"  also in    : "
          + (", ".join(f"{h['file']}:{h['line']}" for h in log_hits) or "none"))

    # ---- the two runs ----------------------------------------------------
    lit_rows = per_city_rows(lit, "expa_run_id", "literal_experiment_a")
    rev_rows = per_city_rows(rev, "adapt_run_id", "revised_decoder_adaptation")

    print("\n--- per-city median per-image recovery ratio R (copied) ---")
    print(f"  {'run':28s} {'city':9s} {'median R':>9s} "
          f"{'95% CI':>22s} {'n_valid':>8s} {'n_excl':>7s} {'min_gap':>8s}")
    for r in lit_rows + rev_rows:
        print(f"  {r['run']:28s} {r['city']:9s} "
              f"{r['median_per_image_R']:+9.4f} "
              f"[{r['R_ci95'][0]:+9.4f}, {r['R_ci95'][1]:+9.4f}] "
              f"{r['n_valid']:8d} {r['n_excluded']:7d} "
              f"{r['min_gap_threshold']:8.2f}")

    # The two runs share their LOCO and upper cells, so the exclusion mask is
    # the same quantity in both and the counts must match city by city. A
    # mismatch would mean the two runs did not join the same test pool.
    excl_check = []
    for a, b in zip(lit_rows, rev_rows):
        same = (a["n_valid"] == b["n_valid"]
                and a["n_excluded"] == b["n_excluded"])
        if not same:
            fail(f"{a['city']}: the two runs report different valid/excluded "
                 "counts although both exclude on |IoU_upper - IoU_LOCO| from "
                 "the same two cells")
        excl_check.append(dict(city=a["city"], n_valid=a["n_valid"],
                               n_excluded=a["n_excluded"], identical=True))

    # ---- part 1 context, copied, not contrasted --------------------------
    part1_context = dict(
        note="arm E Part 1's frozen-backbone LOCO and upper levels, copied for "
             "provenance. Part 1's own gap is already promoted in "
             "results/02_frozen_backbone_gap.json on a different metric "
             "(two-class boundary-tolerant mIoU at K=2); the values here are "
             "Part 1's own foreground-only headline metric and are carried as "
             "context, not re-derived and not contrasted with anything.",
        headline_metric=p1["headline_metric"],
        per_city=[dict(city=c["city"],
                       loco_fg_iou_K2_per_image_mean_pct=c[
                           "loco_fg_iou_K2_per_image_mean_pct"],
                       upper_fg_iou_K2_per_image_mean_pct=c[
                           "upper_fg_iou_K2_per_image_mean_pct"],
                       loco_minus_upper_point_estimate_pct=c[
                           "loco_minus_upper_point_estimate_pct"],
                       loco_epoch_of_best=c["loco_epoch_of_best"])
                  for c in p1["per_city"]],
        part1_not_a_tested_claim=p1["not_a_tested_claim"])

    # ---- runs ------------------------------------------------------------
    runs = sorted(set(lit["runs"]) | set(rev["runs"]) | set(p1["runs"]))

    # ---- headline --------------------------------------------------------
    def med(rows):
        return ", ".join(f"{r['city']} {r['median_per_image_R']:+.3f}"
                         for r in rows)

    headline = (
        "On the frozen-backbone DINOv3 cells at rung 450, high resolution, the "
        "literal Experiment A run, which REINITIALIZED the decoder, has "
        "per-city median per-image recovery ratios " + med(lit_rows)
        + ", and the revised run, which loaded arm E Part 1's LOCO-trained "
          "decoder instead of reinitializing it, has " + med(rev_rows)
        + "; both are single-seed and neither reproduces Table 36's pooled "
          "DINOv3 R of 0.69.")

    # ---- caveats ---------------------------------------------------------
    caveats = [
        # the four mandatory ones, in the brief's order
        "The revised run loads arm E Part 1's LOCO-trained decoder rather than "
        "reinitializing it, so it is not Experiment A as published and is not "
        "comparable to Table 36's 0.69. This JSON makes no frozen-versus-"
        "fine-tuned comparison: no number here is differenced against, pooled "
        "with, or tested against the published Table 36 value or any "
        "fine-tuned cell.",

        "With the encoder frozen, the decoder is the only trained component, so "
        "decoder recovery is expected by construction. This experiment does not "
        "test the failure locus. Under full backbone freezing the encoder is "
        "byte-identical across the LOCO, upper and retrained cells of a city, "
        "which removes the encoder as a confound and at the same time removes "
        "the only component the locus question contrasts the decoder against.",

        "Per-image R vectors were computed in memory and not saved "
        "(delta_runs/arme_scripts/arme_part2_revised_aggregate.py lines 90 and "
        "175: line 90 computes the vector, line 175 writes only the per-city "
        "summary). The per-city medians and CIs are the only stored values, so "
        "no interval here can be recomputed, re-pooled across cities, or "
        "re-tested, and no paired test between the two runs is available. The "
        "trained adapt-adapt decoders have no checkpoint on Delta. "
        "DISAGREEMENT RECORDED: two records on this box state the opposite of "
        "that last sentence. "
        "delta_logs/session_logs/2026-07-28-arme-part2-revised.md lines "
        "126-127 list 'Legacy-style training artifacts (decoder-only "
        "checkpoints, ~22.3 MB each): data/dinov3/outputs/armE_part2_revised/"
        "dinov3_{city}_highres_1/', and each adapt-adapt dump's own run.json "
        "carries ckpt_path=$PROJECT_ROOT/data/dinov3/"
        "outputs/armE_part2_revised/dinov3_{city}_highres_1/"
        "checkpoint_latest.pth. Whether those files still exist on Delta "
        "cannot be checked from the analysis host, which does not reach the SLURM box "
        "(CLAUDE.md). The brief's statement is recorded as stated and the two "
        "contradicting records are recorded beside it; neither is adopted.",

        "One seed per cell. High resolution, rung 450. Every cell in both runs "
        "is seed 0, so neither run carries a noise floor and no per-city "
        "difference here can be read against run-to-run variance.",

        # this job's own limitations
        "PROMOTION ONLY. Every number in this JSON is copied from "
        "delta_runs/arme_part2/results.json and "
        "delta_runs/arme_part2_revised/results.json. No dump was read, no R "
        "was recomputed and no bootstrap was run on this box, so nothing here "
        "is an independent check of those two files. What was checked is "
        "internal consistency of what they state: valid plus excluded equals "
        "the joined pool in all six city blocks, each stored CI contains its "
        "own median, the exclusion threshold is 0.02 in all six, and the two "
        "runs' valid/excluded counts agree city by city.",

        "The literal Experiment A run collapsed to majority-class background "
        "prediction in chicago and miami, which is why all three of its "
        "per-city medians are negative: the retrained decoder scored below the "
        "LOCO cell it is measured against, so R is negative by definition. The "
        "collapse epochs recorded for it (chicago 13, miami 11) were diagnosed "
        "by hand from per-epoch logs after the fact, because that run predates "
        "the collapse_flag the revised run carries; phoenix is recorded as "
        "sustained degradation rather than a strict collapse under that rule. "
        "The source for all three is "
        "delta_logs/session_logs/2026-07-28-arme-part2.md, not a stored field "
        "of the literal run's own results file.",

        "The exclusion rule drops images whose |IoU_upper - IoU_LOCO| is below "
        "0.02, and it drops a large and uneven share of the pool: 22 of 150 in "
        "chicago, 57 of 150 in miami, 43 of 150 in phoenix. The surviving "
        "denominators are 128, 93 and 107, so the three per-city medians are "
        "not computed over comparable fractions of their cities' test pools, "
        "and the miami median rests on 62 percent of that city's images.",

        "The two runs are not independent of each other. They share both "
        "reference cells of every city, so the exclusion mask, the denominator "
        "and the gap in R's denominator are the same quantities in both, and "
        "only the numerator differs. A difference between the two runs' "
        "medians is therefore a statement about the two retrained decoders "
        "only, and no test of that difference exists on this box.",

        "The published Experiment A number this is not comparable to is "
        "recorded in the published_table36 field for exactness: median R 0.69, "
        "95% CI [0.60, 0.75], pooled over the three cities, n 335-351 valid "
        "images, decoder reinitialized on a LOCO-trained encoder, full "
        "fine-tune. Both runs here are per-city rather than pooled, so even "
        "the literal run's numbers are a different statistic from that one, "
        "before the frozen encoder is considered.",

        "Gate E-3 in REVIEWS.md pre-commits a reading on R under freezing. It "
        "is NOT adjudicated here. Its three branches compare R against the "
        "published 0.69, which both of these runs' own source files state they "
        "are not a reproduction of, and the branch boundaries ('R "
        "substantially lower / higher') carry no threshold. Adjudicating it "
        "needs either a pooled R on the literal frozen condition or a declared "
        "threshold; neither exists.",

        "No INDEX.parquet was read or built: this job reads no dump tree. The "
        "index_sha256 field carries a sha256 over the sorted list of source "
        "file paths and their own sha256 values, the substitution "
        "results/02_frozen_backbone_gap.json used for the same reason.",
    ]

    # ---- sanity flags ----------------------------------------------------
    sanity = [
        "The two runs report IDENTICAL valid and excluded counts in every city "
        "(128/22, 93/57, 107/43). Flagged per CONVENTIONS.md as too clean, and "
        "explained: the exclusion depends only on |IoU_upper - IoU_LOCO| from "
        "the two reference cells, which both runs share, so the counts are the "
        "same quantity measured twice and are expected to be identical. It is "
        "a consistency check that passed, not three coincidences.",

        "All three of the literal run's medians are negative and all three of "
        "the revised run's are positive, with no overlap between the two runs' "
        "CIs in any city. The sign flip is the collapse: a decoder that "
        "predicts background everywhere scores below the LOCO cell, which puts "
        "R below zero by construction. The magnitudes of the negative medians "
        "(-0.956 to -7.087) are ratios with a small denominator and should not "
        "be read as effect sizes.",

        "The revised run's three medians, 0.743, 0.602 and 0.646, straddle the "
        "published 0.69 and the chicago CI [0.687, 0.831] has the published "
        "0.69 just inside its lower end. CONVENTIONS.md asks for a flag when a "
        "number lands near a published value. It is recorded as a flag only: "
        "the conditions differ (frozen encoder, decoder loaded not "
        "reinitialized, per-city not pooled), so proximity to 0.69 is not "
        "evidence of agreement with it and this JSON does not compare the two.",

        "The revised run's phoenix CI [0.575, 0.888] is 0.313 wide against "
        "0.145 for its chicago CI [0.687, 0.831], a factor of 2.2 on 107 valid "
        "images against 128. Both are visibly asymmetric about their own "
        "median. Reported as-is: the bootstrap is of a median of a ratio whose "
        "denominator is thresholded at 0.02, so a symmetric interval is not "
        "expected and the width is not a function of n alone.",

        "The literal run's results file does not record its own collapse. It "
        "records done=true for all three cities and carries no collapse field "
        "at all; the collapse is recorded only in the session log and in the "
        "revised run's per-city previous_reinit_attempt_* fields. A reader of "
        "the literal file alone would see three negative medians and no stated "
        "cause.",

        "Neither source file carries a config_sha256 or a git_hash, and both "
        "record the same reason: the project root on the SLURM box is not a "
        "git repository. The CONTRACT.md section 7 duplicate-run_id check on "
        "config hash therefore cannot be run on these cells. Substituted: the "
        "twelve run_ids are distinct by construction and every source file's "
        "sha256 is recorded in this JSON.",
    ]

    # ---- CONTRACT section 6 conditions -----------------------------------
    conditions = []
    for r in lit_rows + rev_rows:
        conditions.append(dict(
            label=f"DINOv3 frozen backbone / {r['run']} / {r['city']} / "
                  "median per-image recovery ratio",
            metric="median_per_image_recovery_ratio_R",
            mean=r["median_per_image_R"],
            statistic="median", stderr=None,
            stderr_note="not stored by the source; the per-image R vector was "
                        "not saved, so no stderr can be formed here",
            n=r["n_valid"], n_excluded=r["n_excluded"],
            ci95=r["R_ci95"], ci_method=r["R_ci_method"],
            baseline="R is itself a ratio against the within-city upper bound "
                     "and the LOCO cell of the same city: "
                     "R_i = (IoU_retrained_i - IoU_LOCO_i) / "
                     "(IoU_upper_i - IoU_LOCO_i)",
            promoted_from=r["source_field"]))

    tests = [dict(
        name="none",
        compares=[],
        statistic=None, p=None, B=None, two_tailed=None,
        note="No test is reported. The two runs' per-image R vectors were not "
             "saved, so no paired comparison between them, and no pooled "
             "interval across cities, can be computed on this box. The only "
             "intervals available are the six per-city bootstrap CIs the "
             "source files stored, carried on the conditions above.")]

    index_payload = "\n".join(f"{SRC[k]}:{shas[k]}" for k in sorted(SRC))

    out = dict(
        id="23_frozen_backbone_decoder_retrain",
        arm="E",
        answers=["zNBu-W1", "zNBu-W5", "AC1"],
        answers_note=(
            "The arm E row in REVIEWS.md answers zNBu-W1, zNBu-W5 and AC1, and "
            "both source files carry the same three IDs. Gate E-3 is the "
            "pre-commitment that covers R under freezing and it is NOT "
            "adjudicated by this JSON; the reason is in the caveats and in "
            "precommitted_reading."),
        question=("What do the two frozen-backbone DINOv3 decoder-retraining "
                  "runs report, per city, and which of them reinitialized the "
                  "decoder?"),
        status="PROMOTED",
        promotion_only=(
            "Every number is copied from the two Delta results files. No dump "
            "was read, no recovery ratio was recomputed and no bootstrap was "
            "run on this box."),
        published_table36=PUBLISHED_TABLE36,
        not_a_reproduction=(
            "Neither run reproduces the published Experiment A. The literal "
            "run reinitializes the decoder as published but does so on a "
            "frozen pretrained encoder rather than a LOCO-trained one, and "
            "reports per-city rather than pooled medians. The revised run "
            "additionally loads arm E Part 1's LOCO-trained decoder instead of "
            "reinitializing it. Each source file states this in its own "
            "published_reference field, quoted below."),
        published_reference_verbatim=dict(
            literal=lit["published_reference"],
            revised=rev["published_reference"]),
        runs_compared=dict(
            literal_experiment_a=dict(
                source=SRC["literal"],
                source_sha256=lit_sha,
                source_id=lit["id"],
                decoder_initialization="REINITIALIZED",
                question_verbatim=lit["question"],
                recovery_ratio_definition=lit["recovery_ratio_definition"],
                bootstrap_method=lit["bootstrap_method"],
                min_gap_exclusion=lit["min_gap_exclusion"],
                metric_module_sha256=lit["metric_module_sha256"],
                generated_utc=lit["generated_utc"],
                script=lit["script"],
                run_id_pattern="dinov3frozen-expa-expa-r450-{city}-hr-s0",
                encoder_confound_note=lit["encoder_confound_note"]),
            revised_decoder_adaptation=dict(
                source=SRC["revised"],
                source_sha256=rev_sha,
                source_id=rev["id"],
                decoder_initialization="LOADED from arm E Part 1's LOCO "
                                       "checkpoint, then fine-tuned",
                question_verbatim=rev["question"],
                recovery_ratio_definition=rev["recovery_ratio_definition"],
                bootstrap_method=rev["bootstrap_method"],
                min_gap_exclusion=rev["min_gap_exclusion"],
                metric_module_sha256=rev["metric_module_sha256"],
                generated_utc=rev["generated_utc"],
                script=rev["script"],
                run_id_pattern="dinov3frozen-adapt-adapt-r450-{city}-hr-s0",
                encoder_confound_note=rev["encoder_confound_note"])),
        which_run_reinitialized_the_decoder="literal_experiment_a "
                                            "(dinov3frozen-expa-expa-r450-"
                                            "{city}-hr-s0)",
        exclusion_rule=dict(
            min_gap_threshold=MIN_GAP_EXPECTED,
            definition="an image is excluded when |IoU_upper_i - IoU_LOCO_i| is "
                       "below the threshold, which is R's denominator, or when "
                       "R_i is NaN",
            applied_identically_in_both_runs=True,
            per_city=excl_check),
        per_city_R=lit_rows + rev_rows,
        spec_error_label=spec_error_label,
        per_image_R_not_saved=dict(
            statement="Per-image R vectors were computed in memory and not "
                      "saved. The per-city medians and CIs are the only stored "
                      "values.",
            evidence=vector_evidence,
            consequence="No pooled-over-cities R, no stderr, no paired test "
                        "between the two runs and no re-pooling at another "
                        "exclusion threshold can be produced from the stored "
                        "artifacts.",
            recomputation_not_attempted=(
                "The per-image foreground IoU inputs do exist on the analysis host, in "
                "dumps/st/dinov3frozen-*-r450-*-hr-s0/perimage.parquet, so R "
                "could in principle be rebuilt. The brief forbids recomputing "
                "it and this job did not read those parquets. A rebuilt R "
                "would be a new computation, not a check of these two files.")),
        part1_context=part1_context,
        precommitted_reading=dict(
            gate="E-3",
            source="REVIEWS.md, Arm E, Gate E-3",
            branches_listed=["R approximately 0.69, matching published",
                             "R substantially lower",
                             "R substantially higher"],
            adjudicated=False,
            why_not=("Every branch is a comparison against the published 0.69, "
                     "and both source files state they are not a reproduction "
                     "of that number. The branch boundaries also carry no "
                     "threshold for 'substantially'. This JSON reports both "
                     "runs' per-city medians and makes no frozen-versus-"
                     "fine-tuned comparison, so no branch is selected.")),
        conditions=conditions,
        tests=tests,
        runs=runs,
        runs_note=("twelve run_ids: the three frozen LOCO cells, the three "
                   "frozen within-city upper cells, the three literal "
                   "Experiment A cells and the three revised "
                   "decoder-adaptation cells. All are Delta cells; this job "
                   "read none of their dumps."),
        runs_excluded=[dict(
            run_id="dinov3frozen-*-r450-chicago-hr-s0-smoke",
            reason="smoke-test cells; not part of either reported run")],
        source_artifacts=[
            dict(path=SRC["literal"], sha256=shas["literal"],
                 role="the literal Experiment A run's stored per-city R"),
            dict(path=SRC["revised"], sha256=shas["revised"],
                 role="the revised run's stored per-city R, and the field that "
                      "calls the literal run a spec error"),
            dict(path=SRC["part1"], sha256=shas["part1"],
                 role="the frozen-backbone LOCO and upper levels, carried as "
                      "context"),
            dict(path=SRC["aggregate_script"], sha256=shas["aggregate_script"],
                 role="lines 90 and 175: the per-image R vector is computed "
                      "and never written"),
            dict(path=SRC["log_literal"], sha256=shas["log_literal"],
                 role="the collapse diagnosis for the literal run"),
            dict(path=SRC["log_revised"], sha256=shas["log_revised"],
                 role="the revised run's outputs list, including the "
                      "decoder-only checkpoint paths"),
            dict(path=SRC["log_part1"], sha256=shas["log_part1"],
                 role="arm E Part 1's session record"),
            dict(path="results/02_frozen_backbone_gap.json",
                 role="arm E Part 1's gap, already promoted; not read or "
                      "contrasted here, named so the two are not double-"
                      "counted"),
        ],
        index_sha256=hashlib.sha256(index_payload.encode()).hexdigest(),
        index_sha256_note=("no INDEX.parquet was read or built; this job reads "
                           "no dump tree. The value is sha256 over the sorted "
                           "'path:sha256' lines of the source files, the "
                           "substitution results/02_frozen_backbone_gap.json "
                           "used."),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/23_frozen_backbone_decoder_retrain.py",
        env=dict(python=sys.version.split()[0],
                 conda_env="myenv "
                           "(conda env)",
                 pythonnousersite=os.environ.get("PYTHONNOUSERSITE")),
        seed=None,
        seed_note=("no stochastic step runs here; the bootstrap seeds of the "
                   "promoted intervals are 42 in both source runs and are "
                   "carried per city"),
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; "
                        "only final_paper/ is git-connected and this job does "
                        "not touch it",
        compute_note="read-only CPU promotion on the analysis host. No training, no GPU "
                     "inference, no checkpoint loaded, no parquet read, "
                     "nothing under dumps/, delta_runs/, delta_logs/ or "
                     "results/ modified.",
        headline=headline,
        sanity_flags=sanity,
        caveats=caveats,
        figure=None,
        caption=None,
    )

    print("\nheadline: " + headline)
    print(f"\ncaveats {len(caveats)}   sanity flags {len(sanity)}   "
          f"conditions {len(conditions)}   runs {len(runs)}")

    if DRY_RUN:
        print("\nDRY_RUN: nothing written.")
        return

    outpath = os.path.join(BASE, "results", OUT_NAME)
    if os.path.exists(outpath):
        fail(f"{outpath} already exists; this job writes exactly one new file")
    tmp = outpath + ".new"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    os.replace(tmp, outpath)
    print(f"\nwrote {outpath}")
    print(f"  sha256 {sha256_file(outpath)}")


if __name__ == "__main__":
    main()
