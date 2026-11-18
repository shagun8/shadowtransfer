#!/usr/bin/env python
"""
Disclosure record -- five Delta findings of 2026-10-25 to 2026-10-28, promoted
into results/.

Answers E-7, E-9, AC1, BxRx-W4.

Five findings from the Delta audit sessions bear on numbers that are already in
the paper, and none of them is in a results JSON, so none can be cited:
  1. the training-image-count audit,
  2. the six published cells retrained after their March 2026 runs collapsed,
  3. the DFFM defect, its provenance, and the inference batch size of every
     published OGLANet artifact family,
  4. the OGLANet model-source verdict,
  5. the finding that the old test_results.json files match no scoring
     definition.
This job promotes all five into results/21_disclosure_record.json. NN 21 is the
next free results number after 20.

Nothing is recomputed. Every value is copied from the Delta evidence under
delta_runs/ and every source carries its path and sha256. Where the operator
copied an artifact to the analysis host under two roots, both copies are compared and the
newer complete one is used, and this JSON says which.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/21_disclosure_record.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _promote_lib import (BASE, env_block, fail, load_json,  # noqa: E402
                          source, write_results)

DRY_RUN = "--dry-run" in sys.argv

TCA = "delta_runs/train_count_audit_projects"
TCA_SCRATCH = "delta_runs/train_count_audit"
RETRAIN_LIST = TCA + "/retrain_at_450_list.json"
HISTMATCH_LIST = TCA + "/histmatch_on_sar_list.json"
TCA_CELL = TCA + "/cells/P-mamnet-vanilla-loco-chicago-hr.json"
RETRAINED = "delta_runs/retrained_cells_record/results.json"
MRFP = "delta_runs/mrfp_rerun_audit/results.json"
DFFM = "delta_runs/dffm_defect/provenance.json"
NONDET = "delta_runs/oglanet_nondeterminism/results.json"
DFFM_CONFIRM = "delta_runs/oglanet_nondeterminism/dffm_confirm.json"
STATEMUT = "delta_runs/oglanet_nondeterminism/state_mutation.json"
PROBSRC = "delta_runs/oglanet_saved_prediction_source"
PROBSRC_PROJECTS = "delta_runs/oglanet_saved_prediction_source_projects"
STRICTFALSE = PROBSRC + "/strictfalse_load_audit.json"
VANREF = PROBSRC + "/vanilla_reference_scores.json"
VARIANTS = "delta_runs/test_results_metric_variant_audit/results.json"
VARIANTS_MANIFEST = "delta_runs/test_results_metric_variant_audit/manifest.json"


def listing(rel):
    root = os.path.join(BASE, rel)
    out = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            out.append(dict(path=os.path.relpath(p, BASE),
                            bytes=os.path.getsize(p)))
    return sorted(out, key=lambda r: r["path"])


def main():
    # ---- the two duplicated copies, compared before anything is read -------
    dup = {}
    for name, a, b in (("train_count_audit", TCA_SCRATCH, TCA),
                       ("oglanet_saved_prediction_source",
                        PROBSRC, PROBSRC_PROJECTS)):
        la, lb = listing(a), listing(b)
        dup[name] = dict(
            copy_a=dict(path=a, n_files=len(la), files=la),
            copy_b=dict(path=b, n_files=len(lb), files=lb))
    # train_count_audit: the two copies are complementary, not duplicates. The
    # /scratch copy holds the SAR weight-drift probe outputs; the /projects copy
    # holds the roster cell artifact and the two prepared lists, and its own
    # sar_drift entries are .MOVED pointers left behind when those files were
    # relocated to /scratch. Both are used, each for what only it has.
    dup["train_count_audit"].update(
        verdict="COMPLEMENTARY, not duplicated",
        used=dict(lists_and_cells=TCA,
                  sar_drift=TCA_SCRATCH),
        why="the /projects copy is the only one carrying "
            "retrain_at_450_list.json, histmatch_on_sar_list.json and "
            "cells/; its sar_drift entries are .MOVED pointers, and the real "
            "sar_drift JSONs are only in the /scratch copy. Neither copy is a "
            "superset of the other.")
    # oglanet_saved_prediction_source: the /scratch copy is the newer complete
    # one. The /projects copy holds only the 5-image, 2-pass smoke.
    dup["oglanet_saved_prediction_source"].update(
        verdict="the /scratch copy is the newer complete one",
        used=PROBSRC,
        why="the /scratch copy holds the four 10-pass, 150-image runs of "
            "2026-10-27 plus the strict=False load audit and the vanilla "
            "reference scores. The _projects copy holds only the 2026-10-26 "
            "5-image, 2-pass smoke under smoke/, which the real runs supersede.")

    tca_cell = load_json(TCA_CELL)
    retrain_list = load_json(RETRAIN_LIST)
    histmatch_list = load_json(HISTMATCH_LIST)
    retrained = load_json(RETRAINED)
    mrfp = load_json(MRFP)
    dffm = load_json(DFFM)
    nondet = load_json(NONDET)
    dffm_confirm = load_json(DFFM_CONFIRM)
    statemut = load_json(STATEMUT)
    strictfalse = load_json(STRICTFALSE)
    vanref = load_json(VANREF)
    variants = load_json(VARIANTS)
    variants_manifest = load_json(VARIANTS_MANIFEST)

    probsrc = {}
    for city in ("chicago", "miami"):
        for test in ("t1", "t2"):
            rel = f"{PROBSRC}/{city}_hr_{test}.json"
            d = load_json(rel)
            probsrc[f"{test}/{city}"] = dict(
                source=rel, test=d["test"], pipeline=d["pipeline"],
                checkpoint=d["checkpoint"], config_path=d["config_path"],
                saved_probs_dir=d["saved_probs_dir"], n_images=d["n_images"],
                passes=d["passes"], saved_score=d["saved_score"],
                range_check_fg_iou_k2=d["range_check_fg_iou_k2"],
                range_check_miou_k2_twoclass=d["range_check_miou_k2_twoclass"],
                per_image_delta_vs_saved={
                    k: v for k, v in d["per_image_delta_vs_saved"].items()
                    if k != "worst_across_passes"
                    and k != "best_across_passes"},
                determinism_across_passes=d["determinism_across_passes"],
                env=d["env"])
    if len(probsrc) != 4:
        fail(f"expected 4 model-source runs, found {len(probsrc)}")

    # identity checks on the copy
    if len(retrained["cells"]) != 6:
        fail(f"the retrained-cells record holds {len(retrained['cells'])} "
             "cells, expected 6")
    if variants["n_cells_with_a_match"] + variants["n_cells_with_no_match"] \
            != variants["n_cells"]:
        fail("the metric-variant audit's match counts do not sum to its cell "
             "count")
    fams = dffm["inference_batch_size_by_family"]
    bs = {f["family"]: f["inference_batch_size"] for f in fams}
    if sorted({v for v in bs.values()}) != [1, 4]:
        fail(f"the batch-size families are no longer just 1 and 4: {bs}")
    n4 = sum(1 for v in bs.values() if v == 4)
    n1 = sum(1 for v in bs.values() if v == 1)
    if n4 != 2 or n1 != 5:
        fail(f"expected 2 batch-4 families and 5 batch-1 families, found "
             f"{n4} and {n1}")
    for k, v in probsrc.items():
        if v["test"] == "t1":
            if not (v["range_check_fg_iou_k2"]["saved_inside_range"]
                    and v["range_check_miou_k2_twoclass"]["saved_inside_range"]):
                fail(f"{k}: the copied t1 range check no longer brackets the "
                     "saved value, which the promoted verdict rests on")
        else:
            if (v["range_check_fg_iou_k2"]["saved_inside_range"]
                    or v["range_check_miou_k2_twoclass"]["saved_inside_range"]):
                fail(f"{k}: the copied t2 range check now brackets the saved "
                     "value, which the promoted verdict rests on not doing")

    out = dict(
        id="21_disclosure_record",
        arm="H2",
        answers=["E-7", "E-9", "AC1", "BxRx-W4"],
        question="What do the 2026-10-25 to 2026-10-28 Delta audits establish "
                 "about how the published numbers were produced, and which "
                 "published numbers does each finding reach?",
        status="PROMOTED_EVIDENCE",
        promotion_note="This JSON computes nothing. Every value is copied from "
                       "the Delta evidence under delta_runs/ and each source "
                       "carries its path and sha256. The script's checks (the "
                       "six retrained cells, the metric-variant match counts "
                       "summing to the cell count, two batch-4 and five "
                       "batch-1 artifact families, and the four model-source "
                       "range checks still falling the way the verdict says) "
                       "are checks on the copy, not new statistics.",
        duplicate_copies=dup,

        # ---- finding 1 -------------------------------------------------
        finding_1_training_counts=dict(
            what="the published cells' training-image counts, measured rather "
                 "than declared",
            measured=dict(
                trained_on_900=["MAMNet standard (vanilla) LOCO, 3 cities",
                                "MAMNet FDA, SegDeSiC, IIM, ISW, MRFP+ and "
                                "FADA, 3 cities each (18 cells)",
                                "OGLANet SIB (C4clean), 3 cities",
                                "OGLANet SegDeSiC, 3 cities"],
                trained_on_900_n_cells=27,
                trained_on_900_per_source_city=450,
                trained_on_450="every other cross-city cell, and every "
                               "within-city upper bound",
                validated_on_300="every pooled-LOCO cell",
                validated_on_150="every within-city upper bound, and the BSM "
                                 "and internal CrossEarth cells",
                contract_and_paper_say=150,
                contract_and_paper_say_note="paper section 3.3 and "
                                            "CONTRACT.md's own example both "
                                            "record n_val 150; no published "
                                            "LOCO cell used 150"),
            how_measured=dict(
                method_a="rebuild each original's dataset with its own dataset "
                         "code and count",
                method_b="read the checkpoint's stored epoch index and "
                         "optimizer step count and divide: 112 batches per "
                         "epoch at batch_size 8 with drop_last is 900, 56 is "
                         "450",
                agreement="the two methods agreed on the one cell audited end "
                          "to end, within drop_last's 0..7",
                hard_limit="61 of the 123 roster cells have no checkpoint on "
                           "disk (archived to Purdue Fortress 2026-09-23), so "
                           "method b cannot run on them; method a still can",
                grid_status="the 123-cell audit grid was prepared and left "
                            "gated, so no per-cell results table exists. What "
                            "is promoted here is the 24 cells read directly "
                            "plus the code audit behind the rest."),
            stage_2_cell_artifact=tca_cell,
            retrain_list=retrain_list,
            histmatch_on_sar_list=dict(
                n_cells_already_done=histmatch_list.get("n_done"),
                summary="prepared, not submitted; carried for completeness "
                        "because it is the other half of the same audit",
                verbatim=histmatch_list),
            matched_450_retrain_status=(
                "in progress on the SLURM box. As of 2026-10-28, 2 of the 27 "
                "cells had a scored dump (OGLANet-SIB phoenix and "
                "OGLANet-SegDeSiC phoenix), 1 was running, 19 were pending and "
                "5 had no live job: the 3 MAMNet-ISW cells were cancelled "
                "behind a smoke that timed out twice, and the 2 "
                "OGLANet-SegDeSiC fans were cancelled by their own gate after "
                "the real cell's self-check delta came in at 0.7196 against a "
                "0.25 threshold. No n450 dump has reached the analysis host."),
            consequence_for_the_paper=(
                "MAMNet's standard LOCO cells and its six adaptation cells, "
                "and OGLANet's SIB and SegDeSiC cells, are compared against "
                "450-image within-city upper bounds at twice the training "
                "budget. The premise that the method cells deviate from their "
                "architecture's standard is only true for OGLANet: MAMNet's "
                "own LOCO standard is 900, so MAMNet method cells at 900 match "
                "their architecture's baseline. A matched-450 MAMNet retrain "
                "also cannot restore 'the standard models' exact image lists', "
                "because MAMNet never drew a 450-image subset; adopting "
                "OGLANet and DINOv3's draw would be a new decision.")),

        # ---- finding 2 -------------------------------------------------
        finding_2_retrained_cells=dict(
            what="six published cells were retrained after 2026-03-19 because "
                 "their March runs collapsed; the printed tables use the later "
                 "runs",
            n_cells=len(retrained["cells"]),
            cells=retrained["cells"],
            how_cells_were_identified=retrained["how_cells_were_identified"],
            decision_metric=retrained["decision_metric"],
            caveat_on_settings=retrained["caveat_on_settings"],
            attribution=retrained["attribution"],
            settings_changed_on_one_cell=dict(
                cell="oglanet mrfp_plus miami highres",
                change="learning rate 1.0e-4 -> 3.0e-5",
                evidence="args.json mtime 2026-04-29T01:17:34Z is the exact "
                         "start of the successful run; the first April attempt "
                         "kept 1e-4 and collapsed again",
                other_five="retrained at the same learning rate and simply did "
                           "not collapse the second time"),
            discontinuity=dict(
                what="two cells' published numbers moved between the March and "
                     "April artifacts",
                oglanet_mrfp_plus_chicago_delta_miou=26.6812,
                oglanet_mrfp_plus_miami_delta_miou=39.3567,
                measured_as="eval_sib_results (2026-03-27) against "
                            "final_comparison.json (2026-05-06), two-class "
                            "boundary-tolerant mIoU at K=2",
                control="OGLANet MRFP+ phoenix is the one MRFP+ cell never "
                        "retrained, and on it the two artifacts agree "
                        "bit-for-bit (58.027756061170805 against "
                        "58.027756061171), which is what establishes that the "
                        "jump is a model change and not a scoring defect"),
            mrfp_rerun_audit=mrfp,
            consequence_for_the_paper=(
                "Three of the six are high-resolution cells whose published "
                "Table 38 rows are the post-retrain values, and one is the "
                "DINOv3 vanilla miami cell whose 72.6112 the paper prints. The "
                "reruns are explained and dated; who authorised the one "
                "learning-rate change is unrecorded, because no ledger line, "
                "commit or DECISIONS entry covers 2026-04 and the only actor "
                "evidence is the SLURM job owner.")),

        # ---- finding 3 -------------------------------------------------
        finding_3_dffm_defect=dict(
            what=nondet.get("finding")
                 or dffm_confirm["finding"],
            code=dffm_confirm["code"],
            measured=dict(
                layers_built_per_forward=dffm_confirm[
                    "layers_built_per_forward"],
                in_channel_sequence=dffm_confirm[
                    "in_channel_sequence_per_forward"],
                same_sequence_both_forwards=dffm_confirm[
                    "same_sequence_both_forwards"],
                n_with_identical_weights_across_forwards=dffm_confirm[
                    "n_with_identical_weights_across_forwards"],
                all_in_training_mode=dffm_confirm["all_in_training_mode"],
                all_bn_use_batch_statistics=dffm_confirm[
                    "all_bn_use_batch_statistics"],
                n_params_in_state_dict=dffm_confirm[
                    "n_dffm_downsample_params_in_model_state_dict"],
                model_keys=dffm_confirm["n_model_keys"],
                checkpoint_keys=dffm_confirm["n_checkpoint_keys"],
                settings=dffm_confirm["settings"],
                reading=dffm_confirm["reading"]),
            provenance=dffm["dffm_source_provenance"],
            provenance_summary=(
                "an own reimplementation from a published figure, not a copy "
                "of upstream code. The file carries no source URL, no upstream "
                "repository, no license, no author and no copy-origin "
                "statement; a case-insensitive grep for http, github, arxiv, "
                "'adapted from', copied, 'source:', origin, author, license, "
                "paper and reference returns exactly one hit, line 4, 'Based "
                "on Figure 5 in the paper'. The class docstring also carries "
                "an explicit ASSUMPTION note about the 1x1 channel-reduction "
                "conv. So the defect appears to originate in this project "
                "rather than to have been inherited."),
            public_release=dffm["public_release_copy"],
            public_release_summary=(
                "the public release's copy is byte-identical to the working "
                "tree (same sha256) and carries the same lambda factory at "
                "line 34-38 with the same call site at line 81, so the defect "
                "is in the released code as well."),
            batch_size_by_artifact_family=fams,
            batch_size_summary=dict(
                batch_4=dffm["summary"]["batch_4_families"],
                batch_1=dffm["summary"]["batch_1_families"],
                n_families_at_4=n4, n_families_at_1=n1,
                consequence=dffm["summary"]["consequence"],
                provenance_gap=dffm["summary"]["provenance_gap"]),
            why_batch_size_matters=dffm["why_batch_size_matters"],
            nondeterminism=nondet,
            state_mutation=dict(
                summary="a full-state diff shows nothing, because the 20 "
                        "modules are locals discarded after use, so no "
                        "persistent tensor changes. That earlier negative "
                        "result is consistent with this finding, not evidence "
                        "against it.",
                verbatim=statemut),
            supersedes_erratum_E9=(
                "E-9 records OGLANet inference as nondeterministic with a "
                "published cell-mean standard deviation up to 0.4247 mIoU and "
                "attributes it to cudnn left at defaults. That attribution is "
                "wrong. torch.use_deterministic_algorithms(True) raises nothing "
                "and does not fix it, and passes are still not bit-identical "
                "under it. E-9 names the phenomenon; this is its cause."),
            consequence_for_the_paper=(
                "Every OGLANet number in the project, in training and in "
                "inference, was produced with these 20 random batch-statistic "
                "layers in the forward path. The published tables are batch 4; "
                "every dump-based OGLANet family (arm C's ladder, arm D, S-EO, "
                "Exp A/B and every SAR pass) is batch 1, and the measured "
                "spread on one cell between those two regimes is about 6.6 "
                "foreground-IoU points. An OGLANet number is only meaningful "
                "with its inference batch size stated, and only five of the "
                "seven families record it anywhere but in the script.")),

        # ---- finding 4 -------------------------------------------------
        finding_4_oglanet_model_source=dict(
            what="which model file and which pipeline wrote the saved OGLANet "
                 "predictions",
            design="two pipelines, 10 passes each, all 150 images, on the "
                   "OGLANet vanilla LOCO chicago and miami high-resolution "
                   "cells. Test 1 is python/run_inference_probs.py, the script "
                   "the published jobs ran (17821359 chicago, 17821357 miami). "
                   "Test 2 is the SAR harness with adaptation off "
                   "(armf_sar_pass.unadapted_spec(), steps 0).",
            verdict=(
                "This model file reproduces the saved artifacts only through "
                "the original inference script. Test 1's 10-pass range brackets "
                "both the stored probability maps and the PNG masks on both "
                "cities and both metrics, 8 of 8 times. Test 2 brackets "
                "neither, 8 of 8 times, falling short by 2.59 (chicago) and "
                "7.60 (miami) foreground-IoU points. But no pass of either test "
                "reproduces any saved artifact per pixel: 0 of 150 images are "
                "bit-identical in any pass on any cell, and the pass-to-pass "
                "probability swing reaches 0.80 to 0.88. These OGLANet "
                "artifacts are reproducible in aggregate score and not "
                "reproducible per pixel."),
            per_run=probsrc,
            checkpoint_sha256={
                "chicago": "aacf0edd6ebf196210c1ae2576c31e81b68c23eea8ac2f5048"
                           "202f564f2b3047",
                "miami": "aca8225d7c109bb304bac12c8a31303f345176268ebc7ae3b540"
                         "3ba4d8ae6fd8"},
            checkpoint_sha256_note="computed twice by independent routes and "
                                   "agreeing; both /projects paths are "
                                   "symlinks into /scratch, so the 'which "
                                   "root' question does not arise for these "
                                   "two files",
            strict_false_load_audit=dict(
                summary="a hypothesis that was formed and then falsified, "
                        "recorded so it is not re-derived: both published "
                        "inference scripts load with strict=False and discard "
                        "the returned missing and unexpected keys, so unloaded "
                        "parameters would keep a random init. Measured: 634 of "
                        "634 keys present, 0 missing, 0 unexpected, 0 of "
                        "68,711,566 elements left at constructor init. The "
                        "loader is not the explanation. The silent strict=False "
                        "remains a live way to get a wrong number without "
                        "noticing.",
                verbatim=strictfalse),
            png_vs_probability_map=dict(
                summary="the OGLANet PNG masks and the OGLANet probability maps "
                        "are different predictions, not two scorings of one "
                        "prediction. 20 of 20 OGLANet cells disagree, on 0.76 "
                        "to 3.05 percent of pixels and up to 2.03 two-class "
                        "mIoU; the 10-pixel small-component filter explains 0.1 "
                        "percent of it and the disagreeing pixels sit far from "
                        "p = 0.5. MAMNet and DINOv3 disagree by at most 0.0147. "
                        "The two trees were written five weeks apart by two "
                        "different jobs, each minutes after its own script was "
                        "last modified, and the measured instability above is "
                        "the same size as the gap, so the two trees are best "
                        "read as two independent runs of a nondeterministic "
                        "pipeline.",
                vanilla_reference_scores=vanref),
            consequence_for_the_paper=(
                "The published OGLANet tables can be reproduced in aggregate "
                "from the artifacts on disk through the original script, so no "
                "printed OGLANet value is unsupported. But no OGLANet value is "
                "reproducible per pixel, and the saved probability maps are not "
                "the predictions the tables were scored from, so any "
                "probability-space analysis of an OGLANet cell describes a "
                "different run from the one in the table.")),

        # ---- finding 5 -------------------------------------------------
        finding_5_test_results_match_no_definition=dict(
            what="the old test_results.json files match no scoring definition",
            question=variants["question"],
            axes=variants["axes"],
            n_combinations_per_cell=96,
            n_combinations_note="quantity (two-class mIoU, foreground IoU, "
                                "background IoU, foreground F1 and two-class "
                                "mean F1) x band (K = 0, 2, 5) x filter "
                                "(10-pixel minimum, none) x aggregation "
                                "(per-image mean, pooled counts) x "
                                "empty-foreground (included, excluded)",
            old_value_field=variants["old_value_field"],
            metric_provenance=variants["metric_provenance"],
            tolerance=variants["tolerance"],
            n_cells=variants["n_cells"],
            n_cells_in_the_new_59=variants["n_cells_in_new_59"],
            n_cells_with_a_match=variants["n_cells_with_a_match"],
            n_cells_with_no_match=variants["n_cells_with_no_match"],
            combination_explains_n_cells=variants[
                "combination_explains_n_cells"],
            the_one_match=dict(
                cell="dinov3-segdesic-loco-r450-chicago-hr-s0",
                combination="iou_fg | K0 | filter10px | pooled | "
                            "empty_included",
                diff=0.0023,
                reading="no other cell matches that combination, so it is best "
                        "read as coincidence rather than as a recovered "
                        "definition"),
            self_check="for all 44 cells the audit's own arithmetic reproduces "
                       "each cell's stored foreground IoU at K=2 under the "
                       "cell's own definition to a maximum of 7.1e-15, so "
                       "'none matches' is a statement about the hypothesis "
                       "space and not about the implementation",
            structure="the 34 cells in the new set split in two: a near-miss "
                      "tail of about 8 cells with |diff| <= 0.05 whose closest "
                      "combination IS the current definition, where there is no "
                      "alternative definition to find; and the large-delta "
                      "cells (+33.6, +13.9 and a cluster of about +6 on "
                      "OGLANet) where no combination comes within 2 to 25 "
                      "points. The large-delta cells' closest combinations "
                      "drift toward K=0 and no filter but fall 2 to 3 points "
                      "short, so a definition change does not account for them.",
            per_cell=variants["cells"],
            manifest=variants_manifest,
            consequence_for_the_paper=(
                "The stored test_results.json files are not a usable "
                "comparator for any recomputation: 43 of 44 audited cells match "
                "no definition in a 96-combination sweep, and the tolerance was "
                "not widened to make that verdict. Any reproduction gate that "
                "reads them will fail for reasons that have nothing to do with "
                "the recomputation. The declared path for the internal "
                "adjudication is the prediction PNG tree for exactly this "
                "reason, and the two cells whose stored files are off by -7.91 "
                "and +12.01 foreground mIoU match their PNG-scored published "
                "value to 0.0012 and 0.0057.")),

        conditions=[
            dict(label="published OGLANet artifact trees / inference batch size",
                 metric="inference_batch_size",
                 mean=4, stderr=None, n=n4, n_unit="artifact families",
                 ci95=None, ci_method="none; a configuration value read from "
                                      "the worker script",
                 baseline=None),
            dict(label="dump-based OGLANet artifact families / inference batch "
                       "size",
                 metric="inference_batch_size",
                 mean=1, stderr=None, n=n1, n_unit="artifact families",
                 ci95=None, ci_method="none; a configuration value read from "
                                      "the worker script",
                 baseline=None),
            dict(label="old test_results.json files matching some scoring "
                       "definition",
                 metric="n_cells_with_a_matching_definition",
                 mean=variants["n_cells_with_a_match"], stderr=None,
                 n=variants["n_cells"], n_unit="cells",
                 ci95=None, ci_method="none; a count over a 96-combination "
                                      "sweep at a 0.01 tolerance",
                 baseline="each cell's own re-score under the current "
                          "definition"),
            dict(label="published cells retrained after 2026-03-19",
                 metric="n_cells_retrained",
                 mean=len(retrained["cells"]), stderr=None, n=144,
                 n_unit="published cells",
                 ci95=None, ci_method="none; a count over tensorboard event "
                                      "files",
                 baseline=None),
        ]
        + [dict(label=f"OGLANet model-source {k} / 10-pass "
                      f"{m.replace('range_check_', '')}",
                metric=m.replace("range_check_", ""),
                mean=v[m]["mean"], stderr=None, sd=v[m]["sd"],
                n=v["passes"], n_unit="passes",
                ci95=[v[m]["min"], v[m]["max"]],
                ci_method="observed min and max over 10 passes, not a "
                          "bootstrap interval",
                baseline="the saved artifact's own score",
                baseline_value=v[m]["saved_value"],
                saved_inside_range=v[m]["saved_inside_range"])
           for k, v in sorted(probsrc.items())
           for m in ("range_check_fg_iou_k2",
                     "range_check_miou_k2_twoclass")],

        tests=[dict(
            name="96-combination definition sweep at a 0.01 tolerance",
            compares=["each cell's stored test_results.json value",
                      "every combination of quantity, band, filter, "
                      "aggregation and empty-foreground handling"],
            statistic=f"{variants['n_cells_with_no_match']} of "
                      f"{variants['n_cells']} cells match no combination",
            p=None,
            p_absent_reason="an exhaustive search over a hypothesis space, not "
                            "a hypothesis test",
            self_check="the sweep reproduces each cell's own stored value "
                       "under the cell's own definition to 7.1e-15"),
            dict(name="10-pass bracketing check",
                 compares=["each pipeline's 10-pass range",
                           "the saved artifact's score"],
                 statistic="Test 1 brackets both saved artifacts 8 of 8 times; "
                           "Test 2 brackets neither 8 of 8 times",
                 p=None,
                 p_absent_reason="a bracketing check over observed passes, not "
                                 "a hypothesis test"),
            dict(name="per-pixel reproduction check",
                 compares=["each pass's probability map",
                           "the saved probability map"],
                 statistic="0 of 150 images bit-identical in any pass on any "
                           "of four cells; pass-to-pass maximum absolute "
                           "probability difference 0.797 to 0.884",
                 p=None,
                 p_absent_reason="a byte-identity count, not a hypothesis "
                                 "test")],

        runs=sorted(
            {c["run_id"] for c in variants["cells"] if c.get("run_id")}
            | {f"{c['arch']}-{c['method']}-{c['mode']}-{c['city']}-{c['res']}"
               for c in retrained["cells"]}
            | {"oglanet-probsource-t1-chicago-hr",
               "oglanet-probsource-t1-miami-hr",
               "oglanet-probsource-t2-chicago-hr",
               "oglanet-probsource-t2-miami-hr"}),
        runs_note="the 44 cells of the metric-variant audit, the 6 retrained "
                  "cells named by their published run directory, and the 4 "
                  "model-source probe runs. Findings 1 and 3 are code and "
                  "configuration audits and name no run.",
        runs_excluded=[],
        index_sha256=None,
        index_sha256_reason="every source is a Delta audit artifact under "
                            "delta_runs/, not a dump tree",
        source_artifacts=[
            source(TCA_CELL, "finding 1: the one training-count cell audited "
                             "end to end by both methods"),
            source(RETRAIN_LIST, "finding 1: the measured per-cell counts and "
                                 "the 27-cell matched-450 retrain list"),
            source(HISTMATCH_LIST, "finding 1: the other half of the same "
                                   "audit, prepared and not submitted"),
            source(RETRAINED, "finding 2: the six retrained cells, their "
                              "March attempts and their later runs"),
            source(MRFP, "finding 2: the MRFP+ rerun audit the record extends"),
            source(DFFM, "finding 3: the DFFM provenance and the inference "
                         "batch size of every published OGLANet family"),
            source(DFFM_CONFIRM, "finding 3: the 20-layer confirmation under "
                                 "deterministic algorithms"),
            source(NONDET, "finding 3: the determinism probe and batch sweep"),
            source(STATEMUT, "finding 3: the state-mutation probe and the "
                             "corrected module hook"),
            source(STRICTFALSE, "finding 4: the strict=False load audit"),
            source(VANREF, "finding 4: PNG and probability-map scores for the "
                           "two OGLANet vanilla reference cells"),
            source(VARIANTS, "finding 5: the 96-combination definition sweep "
                             "over 44 cells"),
            source(VARIANTS_MANIFEST, "finding 5: that sweep's manifest"),
        ] + [source(v["source"], f"finding 4: model-source run {k}")
             for k, v in sorted(probsrc.items())],
        script="analysis/21_disclosure_record.py",
        figure=None,
        caption=None,
        seed=0,
        seed_note="no stochastic step; every value is copied from a Delta "
                  "audit artifact",
        headline=(
            "Five Delta audits establish that 27 published cells trained on 900 "
            "images rather than 450 and every pooled-LOCO cell validated on "
            "300 rather than 150; that six published cells were retrained after "
            "their March 2026 runs collapsed, with one learning-rate change "
            "whose author is unrecorded; that OGLANet's DFFM reimplementation "
            "builds 20 unregistered random batch-statistic layers inside "
            "forward() on every call, which makes every OGLANet number one draw "
            "and batch-size dependent across two artifact families at batch 4 "
            "and five at batch 1; that the saved OGLANet predictions are "
            "reproducible in aggregate score through the original script and "
            "not reproducible per pixel; and that 43 of 44 audited "
            "test_results.json files match no scoring definition in a "
            "96-combination sweep."),
        sanity_flags=[
            "Finding 1's strongest numbers are a code audit, not a measurement, "
            "on 21 of the 24 cells reported at 900: their checkpoints are on "
            "Fortress, so only method a (rebuild the dataset and count) could "
            "run. The three OGLANet SIB cells are the exception and have their "
            "own train.log lines.",
            "Finding 1's 15-of-18 cells landing on exactly 448 images per epoch "
            "is expected under samples_per_dir 225 with drop_last, but it is "
            "one value repeating across every architecture, city and mode, so "
            "it is not 15 independent measurements. The Delta session flagged "
            "it as such.",
            "Finding 2's two large discontinuities (+26.68 and +39.36 mIoU) are "
            "too large to be noise and are reported without adjustment. The "
            "phoenix control, where the two artifacts agree bit-for-bit, is "
            "what separates 'a different model' from 'a different scoring'.",
            "Finding 3 corrects two of the Delta session's own earlier probes, "
            "recorded so the wrong answers are not re-derived: a module hook "
            "first named a ReLU because a torchvision BasicBlock reuses one "
            "ReLU module for two call sites, and a norm inventory on a freshly "
            "constructed OGLANet reported no batch-statistic layers at all, "
            "which is true and blind to the 20 that exist only during a "
            "forward.",
            "Finding 5's single match is one cell of 44 under a combination no "
            "other cell matches, at a diff of 0.0023 against a 0.01 tolerance. "
            "It is reported as coincidence; treating it as a recovered "
            "definition would be reading one cell as a result, which "
            "CONVENTIONS.md forbids.",
            "Two of the operator's copied evidence directories exist under two "
            "roots. They were compared file by file before anything was read: "
            "train_count_audit's two copies are complementary rather than "
            "duplicated, and oglanet_saved_prediction_source's /scratch copy is "
            "the newer complete one while the _projects copy holds only the "
            "superseded 5-image smoke. Both verdicts are in duplicate_copies.",
            "Promotion only: if any number here disagrees with its source file, "
            "the source file is right and this JSON is stale."],
        caveats=[
            "This is a disclosure record, not an analysis. It restates no "
            "published number and corrects none. What each finding reaches is "
            "named per finding in its consequence_for_the_paper field; whether "
            "any printed value changes is not decided here.",
            "Finding 1's 123-cell audit grid was prepared and left gated, so "
            "there is no per-cell counts table. The counts promoted here rest "
            "on 24 checkpoints read directly plus a code audit of the trainers "
            "for the rest, and 61 of the 123 roster cells have no checkpoint on "
            "disk at all.",
            "Finding 1 also records an instruction conflict: a 2026-07-28 "
            "operator instruction says on-disk directory contents are not "
            "authoritative for n_train and n_val, and the 2026-10-26 "
            "instruction asks for the split files. The newer instruction was "
            "followed, train.log was preferred where it exists, and nothing "
            "downstream gates on the field.",
            "Finding 2's settings are recovered from each attempt's own "
            "Train/LearningRate at step 1 and its epoch count, not from "
            "args.json, because only the latest config survives per cell. One "
            "config file (oglanet mrfp_plus chicago) additionally has a "
            "data_root pointing at phoenix and an mtime later than the run it "
            "describes; that was reported and not adjusted.",
            "Finding 2's attribution is a negative result: no ledger line, "
            "commit or DECISIONS entry covers 2026-04, so who changed the "
            "learning rate is unrecorded for all six cells and the only actor "
            "evidence is the SLURM job owner.",
            "Finding 3 names the cause of OGLANet's nondeterminism but does not "
            "fix it. python/oglanet/models/dffm.py was deliberately not edited: "
            "every OGLANet number in the project was produced with it, and "
            "editing it would change all of them. The public release carries "
            "the same code.",
            "Finding 3's batch-size table is a provenance gap as much as a "
            "finding. Only the SAR runs record their inference batch size in "
            "the artifact; for the other five families the value exists only in "
            "the script, and the Exp A and Exp B trees carry no "
            "inference_config.json at all.",
            "Finding 4 covers two cells, OGLANet vanilla LOCO chicago and miami "
            "at high resolution. It says nothing about the other OGLANet cells, "
            "and its midres fallback was never triggered because Test 1 "
            "reproduced both artifacts.",
            "Finding 5 audited the 44 cells of "
            "dumps/st_r450_published_recompute whose reproduction gate against "
            "their own stored test_results.json fails: 34 in the 2026-10-25/26 "
            "set of 59 and 10 in the older 17-cell recompute. It says nothing "
            "about the 32 cells whose gate passes.",
            "Finding 5's no-filter arm was recomputed from each cell's own "
            "probs.npz rather than from the stored counts, because a filter "
            "changes the prediction mask and therefore the counts. That is "
            "still CPU work over stored arrays and no inference was run, but it "
            "means the no-filter numbers come from the probability maps, which "
            "for OGLANet are a different prediction from the PNG masks the "
            "tables were scored from (finding 4).",
            "Every finding here was produced on the SLURM box and copied to "
            "the analysis host by the operator. This box did not re-run any of them and "
            "cannot: findings 3 and 4 need a GPU and findings 1, 2 and 5 need "
            "the published run directories, checkpoints, tensorboard event "
            "files and probability maps, none of which cross the machine "
            "boundary.",
        ],
        **env_block())

    write_results(out, "21_disclosure_record", DRY_RUN)


if __name__ == "__main__":
    main()
