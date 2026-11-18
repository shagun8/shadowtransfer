#!/usr/bin/env python
"""
Arm D -- appearance-augmentation stress, promoted into results/.

Answers v1UJ-W4, zNBu-W2, AC1.

Arm D ran on the SLURM box and its numbers have lived in runs/D_S3/,
runs/D_R/correlation.json and runs/D_seedfloor/ ever since. Figure 15 and
plots/captions.md already read from those files directly, which breaks
CONTRACT.md section 6: CS may not compute, or read, a number that is not in a
results JSON. This job promotes the arm's recorded values into
results/15_armD_appearance_augmentation.json. NN 15 is free in results/ and is
the figure number arm D already owns (plots/15_armD_appearance_augmentation.png),
so it satisfies CONVENTIONS.md's "NN matches the figure number".

Nothing is recomputed. Every value is copied from its source file and every
source carries its path and sha256.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/15_armD_appearance_augmentation_results.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _promote_lib import (env_block, fail, load_json, source,  # noqa: E402
                          write_results)

DRY_RUN = "--dry-run" in sys.argv

S3 = "runs/D_S3/summary.json"
S3_LEVELS = "runs/D_S3/levels.json"
S3_E200 = "runs/D_S3/decision_e200.json"
S3_PHX = "runs/D_S3/decision_phoenix_final.json"
S3_PHX_REP = "runs/D_S3/decision_phoenix_replicated.json"
DR = "runs/D_R/correlation.json"
SF = "runs/D_seedfloor/results.json"


def main():
    s3 = load_json(S3)
    lev = load_json(S3_LEVELS)
    e200 = load_json(S3_E200)
    phx = load_json(S3_PHX)
    phxr = load_json(S3_PHX_REP)
    dr = load_json(DR)
    sf = load_json(SF)

    # The decision rule is the arm's own; it is reproduced here as an identity
    # check on the copied values, not recomputed into a new verdict.
    for c in s3["cells"]:
        if abs(c["abs_delta"] - abs(c["gap_aug_mean"] - c["gap_vanilla_mean"])) > 1e-6:
            fail(f"{c['arch']}/{c['city']}: copied abs_delta does not match the "
                 "copied gap means")
        if (c["abs_delta"] > c["threshold"]) != c["changed"]:
            fail(f"{c['arch']}/{c['city']}: copied verdict does not follow from "
                 "the copied rule")
    if s3["grid"]["n_changed"] + s3["grid"]["n_unchanged"] != 9:
        fail("the copied grid counts do not sum to 9 cells")

    changed = [f"{c['arch']}/{c['city']}" for c in s3["cells"] if c["changed"]]

    conditions = []
    for c in s3["cells"]:
        conditions.append(dict(
            label=f"{c['arch']} / {c['city']} / augmented (S3) gap",
            metric="gap_miou_2class_tolerant_K2_pct",
            mean=c["gap_aug_mean"], stderr=c["gap_aug_sd_paired"],
            n=len(c["aug_loco"]["seeds_found"]),
            n_unit="seeds",
            ci95=None,
            ci_method="none; arm D reports across-seed spread, and CLAUDE.md "
                      "routes bootstraps to the SSH box, which arm D never ran",
            baseline="vanilla across-seed mean gap, same cell",
            baseline_mean=c["gap_vanilla_mean"],
            baseline_sd=c["gap_vanilla_sd"],
            noise_floor_sd=c["gap_sd_ref"],
            decision_threshold=c["threshold"],
            abs_delta=c["abs_delta"], changed=c["changed"]))
    conditions.append(dict(
        label="grid / augmented (S3) gap, mean over the 9 cells",
        metric="gap_miou_2class_tolerant_K2_pct",
        mean=s3["grid"]["gap_aug_mean_over_cells"]["mean"],
        stderr=None,
        sd=s3["grid"]["gap_aug_mean_over_cells"]["sd"],
        n=s3["grid"]["gap_aug_mean_over_cells"]["n"], n_unit="cells",
        ci95=None, ci_method="none; copied from runs/D_S3/summary.json",
        baseline="vanilla gap, mean over the same 9 cells",
        baseline_mean=s3["grid"]["gap_vanilla_mean_over_cells"]["mean"],
        baseline_sd=s3["grid"]["gap_vanilla_mean_over_cells"]["sd"]))

    tests = [dict(
        name="arm D decision rule (not a significance test)",
        compares=["augmented (S3) across-seed mean gap",
                  "vanilla across-seed mean gap"],
        rule=s3["decision_rule"], rule_prose=s3["decision_rule_prose"],
        n_cells=9, n_changed=s3["grid"]["n_changed"],
        n_unchanged=s3["grid"]["n_unchanged"],
        changed_cells=changed, p=None,
        p_absent_reason="arm D computed no p-value, no CI and no multiplicity "
                        "correction; its own caveat records that as the SSH "
                        "box's job, and this job promotes values rather than "
                        "computing new ones")]
    for key, blk in dr["block1_correlations"].items():
        tests.append(dict(
            name=f"Spearman rho, outcome {key} against the block-1 predictors",
            compares=[key, "block-1 predictors"],
            correction=dr["block2_multiplicity"]["correction_method"],
            family_scope=dr["block2_multiplicity"]["family_scope"],
            B=dr["B_bootstrap"], n_cells=dr["n_cells"],
            rows=blk))

    out = dict(
        id="15_armD_appearance_augmentation",
        arm="D",
        answers=s3["weakness_ids"],
        question=s3["question"],
        status="PROMOTED_EVIDENCE",
        promotion_note="This JSON computes nothing. Every value is copied from "
                       "the arm D artifacts listed in source_artifacts. The "
                       "two identity checks the script performs (that each "
                       "copied abs_delta follows from the copied gap means, and "
                       "that each copied verdict follows from the copied rule) "
                       "are checks on the copy, not new statistics.",
        metric=dict(name=s3["metric_name"],
                    secondary=s3["secondary_metric_name"],
                    definition=sf["metric_definition"],
                    gap_definition=s3["gap_definition"],
                    eps_iou=sf["eps_iou"],
                    metrics_sha256=s3["metrics_sha256"],
                    metrics_sha256_recorded_in_dumps=sf[
                        "metrics_sha256_recorded_in_dumps"]),
        decision_rule=dict(rule=s3["decision_rule"],
                           prose=s3["decision_rule_prose"],
                           sd_note=s3["sd_note"]),
        per_cell=s3["cells"],
        grid=s3["grid"],
        levels=lev,
        decision_e200=e200,
        decision_phoenix_final=phx,
        decision_phoenix_replicated=phxr,
        seed_floor=dict(
            per_cell=sf["cells"],
            sanity_checks=sf["sanity_checks"],
            sanity_flag_count=sf["sanity_flag_count"],
            recompute_vs_stored_max_abs_delta_pct=sf[
                "recompute_vs_stored_max_abs_delta_pct"],
            caveats=sf["caveats"]),
        appearance_correlation=dict(
            question=dr["question"],
            n_cells=dr["n_cells"],
            cells=dr["cells"],
            outcome_definitions=dr["outcome_definitions"],
            predictor_definitions=dr["predictor_definitions"],
            block1_correlations=dr["block1_correlations"],
            block2_multiplicity=dr["block2_multiplicity"],
            block3_rank_ordering=dr["block3_rank_ordering"],
            block4_predictor_collinearity=dr["block4_predictor_collinearity"],
            block5_per_city_aggregate=dr["block5_per_city_aggregate"],
            appearance_distance_full=dr["appearance_distance_full"],
            appearance_distance_train_vs_train=dr[
                "appearance_distance_train_vs_train"],
            fig6_appC2_side_by_side=dr["fig6_appC2_side_by_side"],
            headline=dr["headline"],
            sanity=dr["sanity"],
            sanity_flags=dr["sanity_flags"],
            caveats=dr["caveats"],
            B_bootstrap=dr["B_bootstrap"],
            B_permutation=dr["B_permutation"],
            seed=dr["seed"]),
        runs=sorted({c["run_id"] for cell in sf["cells"]
                     for side in ("loco", "upper")
                     for c in cell[side]["per_seed"].values()}
                    | {rid for cell in s3["cells"]
                       for side in ("aug_loco", "aug_upper")
                       for rid in [f"{cell['arch']}-augs3-"
                                   f"{'loco' if side == 'aug_loco' else 'upper'}"
                                   f"-r450-{cell['city']}-hr-s{s}"
                                   for s in cell[side]["seeds_found"]]}),
        runs_planned=s3["runs_planned"],
        runs_found=s3["runs_found"],
        cells_patched=s3["cells_patched"],
        index_sha256=None,
        index_sha256_reason="arm D ran on the SLURM box and this job reads its "
                            "results JSONs, not a dump tree, so no "
                            "INDEX.parquet enters it",
        source_artifacts=[
            source(S3, "the S3 stress grid: per-cell augmented and vanilla "
                       "gaps, the decision rule and its verdicts"),
            source(S3_LEVELS, "per-cell LOCO and upper levels behind the gaps"),
            source(S3_E200, "the 200-epoch decision variant"),
            source(S3_PHX, "the OGLANet/Phoenix final decision"),
            source(S3_PHX_REP, "the OGLANet/Phoenix replication"),
            source(DR, "appearance-distance and seed-variance correlations"),
            source(SF, "the across-seed noise floor the decision rule is read "
                       "against"),
        ],
        script="analysis/15_armD_appearance_augmentation_results.py",
        figure="plots/15_armD_appearance_augmentation.png",
        caption="See plots/captions.md, Figure 15 (revision 2). The caption is "
                "already written and is not restated here.",
        seed=dr["seed"],
        seed_note="arm D's own seeds; this job runs no stochastic step",
        headline=(
            f"Under aggressive appearance-only augmentation applied to both "
            f"sides, {s3['grid']['n_changed']} of "
            f"{s3['grid']['n_cells_evaluable']} cells change their LOCO-versus-"
            f"upper gap by more than twice the vanilla across-seed gap "
            f"standard deviation ({', '.join(changed)}), the mean gap moves "
            f"from {s3['grid']['gap_vanilla_mean_over_cells']['mean']:+.3f} to "
            f"{s3['grid']['gap_aug_mean_over_cells']['mean']:+.3f} two-class "
            f"tolerant mIoU at K=2, and the worst augmented cell is "
            f"{s3['grid']['worst_case_gap_aug']['arch']}/"
            f"{s3['grid']['worst_case_gap_aug']['city']} at "
            f"{s3['grid']['worst_case_gap_aug']['gap_aug_mean']:+.3f}."),
        sanity_flags=list(s3.get("sanity_flags", []))
        + list(dr["sanity_flags"])
        + list(sf["sanity_flags"])
        + ["Promotion only: if any number here disagrees with its source file, "
           "the source file is right and this JSON is stale."],
        caveats=[
            # the two the operator's brief requires
            "CrossEarth was not run through arm D. It joined the project after "
            "arm D had already run, so its absence is a scheduling fact and "
            "not an implementation barrier: arm D's augmentation transform "
            "(armd_aug.py) was later shown byte-identical in effect across the "
            "CrossEarth stack, and arm C/D CrossEarth cells were specced with "
            "an augmentation branch. The 9-cell arm D grid is therefore the "
            "three original architectures only.",
            "The OGLANet cells in arm D were inferred at batch size 1. That "
            "matters because OGLANet's DFFM reimplementation builds 20 "
            "unregistered, randomly initialised, train-mode Conv2d+BatchNorm2d "
            "layers inside forward() on every call, so its BatchNorms "
            "normalise with batch statistics and the cell score depends on the "
            "inference batch size (measured on one cell: foreground IoU at K=2 "
            "of 33.05 at batch 1 against 39.62 at batch 4). Every arm D "
            "OGLANet number, and every arm C r450 vanilla number arm D reads "
            "as its noise floor, is in the batch-1 regime; the published "
            "OGLANet tables are batch 4. See "
            "delta_runs/dffm_defect/provenance.json.",
        ]
        + [f"Carried from {S3}: {c}" for c in s3["caveats"]]
        + [f"Carried from {DR}: {c}" for c in dr["caveats"]]
        + [f"Carried from {SF}: {c}" for c in sf["caveats"]]
        + ["Arm D computed no confidence interval, p-value or multiplicity "
           "correction, and this job adds none. Its verdicts rest on a "
           "two-standard-deviation rule against the vanilla across-seed gap "
           "spread, with one to three seeds per augmented cell. Anything the "
           "paper says about significance here needs a new analysis, not this "
           "JSON.",
           "Single augmentation strength (S3), high resolution only, rung 450 "
           "only, so an unchanged gap cannot be attributed to too weak a dose.",
           "The three ride-along arm D diagnostics (city-identity probe, SP "
           "gap, histogram-matching probe) are not in the source files and are "
           "not in this JSON."],
        **env_block())

    write_results(out, "15_armD_appearance_augmentation", DRY_RUN)


if __name__ == "__main__":
    main()
