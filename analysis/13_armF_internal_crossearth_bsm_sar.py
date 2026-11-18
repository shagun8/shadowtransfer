#!/usr/bin/env python
"""
Internal arm F -- CrossEarth, BSM and SAR, promoted into results/.

Answers AC3, v1UJ-W3, v1UJ-W2, AC1.

The three internal arm F method families have lived in runs/armF_* ever since
they ran on the SLURM box, and nothing under results/ carries their numbers, so
CS cannot cite them without breaking CONTRACT.md section 6. This job promotes
them into results/13_armF_internal_crossearth_bsm_sar.json. NN 13 is the lowest
free results number.

Nothing is recomputed. Every value is copied from its source file and every
source carries its path and sha256. The CrossEarth locus tables come from a
markdown summary rather than a JSON, so that file is carried verbatim as text
beside the parsed rows.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/13_armF_internal_crossearth_bsm_sar.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _promote_lib import (env_block, fail, load_json, read_text,  # noqa: E402
                          source, write_results)

DRY_RUN = "--dry-run" in sys.argv

CE_PB = "runs/armF_crossearth_analysis/paired_bootstrap.json"
CE_SP = "runs/armF_crossearth_analysis/sp_gap.json"
CE_NE = "runs/armF_crossearth_analysis/nonempty_subset.json"
CE_LOCUS = "summaries/armF_crossearth_locus.md"
BSM = "runs/armF_bsm_analysis/results.json"
ARCHS = ("mamnet", "oglanet", "dinov3")
CITIES = ("chicago", "miami", "phoenix")


def md_tables(text):
    """Carry the locus markdown's pipe tables through as rows, keeping the
    header spelling the file uses. No value is parsed into a number: the cells
    are copied as the strings the summary prints, so nothing can be silently
    re-rounded."""
    out, title, header, rows = [], None, None, []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("##"):
            if header:
                out.append(dict(title=title, header=header, rows=rows))
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
            out.append(dict(title=title, header=header, rows=rows))
            header, rows = None, []
    if header:
        out.append(dict(title=title, header=header, rows=rows))
    return out


def main():
    ce_pb = load_json(CE_PB)
    ce_sp = load_json(CE_SP)
    ce_ne = load_json(CE_NE)
    locus_text = read_text(CE_LOCUS)
    bsm = load_json(BSM)

    # SAR: the nine internal cells, each its own run
    sar = {}
    sar_missing = []
    for arch in ARCHS:
        for city in CITIES:
            rel = f"runs/armF_sar_{arch}_{city}_highres/results.json"
            p = os.path.join(os.environ["PROJECT_ROOT"], rel)
            if not os.path.isfile(p):
                sar_missing.append(f"{arch}/{city}")
                continue
            d = load_json(rel)
            sar[f"{arch}/{city}"] = dict(
                source=rel,
                arch=d["arch"], city=d["holdout_city"], res=d["resolution"],
                target_access=d["target_access"],
                target_access_detail=d["target_access_detail"],
                checkpoint_path=d["checkpoint_path"],
                checkpoint_sha256=d["checkpoint_sha256"],
                parent_run_dir=d["parent_run_dir"],
                metric=d["metric"], sar_config=d["sar_config"],
                pilot_verdict=d["pilot_verdict"],
                unadapted=d["unadapted"], sar=d["sar"],
                per_band=d["per_band"], per_pass=d["per_pass"],
                sanity_checks=d["sanity_checks"],
                inference_batch_size=d["provenance"].get("batch_size"),
                metric_module_sha256=d["provenance"]["metric_module_sha256"],
                wall_clock_s=d["wall_clock_s"],
                caveats=d["caveats"],
                metric_naming_correction=d["metric_naming_correction"])
    if sar_missing:
        fail(f"arm F SAR cells absent on this box: {sar_missing}")
    if len(sar) != 9:
        fail(f"expected 9 internal SAR cells, found {len(sar)}")

    # identity checks on the copy, not new statistics
    for k, v in sar.items():
        a = v["sar"]["delta_adaptation_only_iou_fg_tol2"]
        b = v["sar"]["iou_fg_tol2_final"] - v["sar"]["iou_fg_tol2_step0"]
        if abs(a - b) > 1e-9:
            fail(f"{k}: copied SAR foreground delta does not follow from the "
                 "copied step-0 and final values")
    if len(bsm["input_cells"]) != 11:
        fail(f"BSM promoted as {len(bsm['input_cells'])} cells, expected 11")
    if bsm["collapsed_cells"] != ["dinov3_chicago"]:
        fail(f"BSM collapsed-cell list changed: {bsm['collapsed_cells']}")

    conditions, tests = [], []
    for e in ce_pb["per_method_summary"]:
        for metric_key, metric_name in (
                ("mIoU_boundary_tolerant_K2_pct",
                 "gap_miou_2class_tolerant_K2_pct"),
                ("IoU_foreground_only_K2_pct",
                 "gap_iou_fg_tolerant_K2_pct")):
            blk = e[metric_key]
            conditions.append(dict(
                label=f"CrossEarth / {e['method']} / gap to its own "
                      f"within-city upper bound ({metric_key})",
                metric=metric_name, mean=blk["mean_gap_over_3_cities"],
                stderr=None, n=blk["n_cities"], n_unit="cities",
                ci95=None,
                ci_method="none at the per-method level; the per-cell paired "
                          "bootstraps are in crossearth.all_150.join and in "
                          "tests",
                baseline="crossearth-vanilla-upper-r450-{city}-hr-s0",
                worst_case_single_cell_gap=blk["worst_case_single_cell_gap"],
                worst_case_city=blk["worst_case_city"],
                sign_note=blk["note"]))
    for e in bsm["per_architecture_summary"]:
        conditions.append(dict(
            label=f"BSM / {e['arch']} / mean over its available cells",
            metric="miou_2class_tolerant_K2_pct",
            mean=e["bsm_mIoU_tol_K2_mean"], stderr=None,
            n=e["bsm_mIoU_tol_K2_n_cells"], n_unit="cells",
            ci95=None, ci_method="none at the per-architecture level",
            baseline="vanilla, same architecture and city",
            delta_vs_vanilla_mean=e["delta_vs_vanilla_mIoU_tol_K2_mean"],
            worst_gap_to_upper=e["worst_gap_to_upper_mIoU_tol_K2"],
            worst_gap_to_upper_city=e["worst_gap_to_upper_mIoU_tol_K2_city"],
            collapsed_cities=e["collapsed_cities"]))
    for k, v in sorted(sar.items()):
        conditions.append(dict(
            label=f"SAR / {k} / unadapted level and adaptation change",
            metric="iou_fg_tolerant_K2_pct",
            mean=v["unadapted"]["iou_fg_tol2"], stderr=None, n=150,
            n_unit="images", ci95=None,
            ci_method="none in the source; arm F reports the change, not a CI",
            baseline="the published cell's own stored test_results.json",
            baseline_value=v["sanity_checks"][
                "unadapted_vs_stored_test_results"]["stored"],
            baseline_difference=v["sanity_checks"][
                "unadapted_vs_stored_test_results"]["difference"],
            adaptation_change=v["sar"]["delta_adaptation_only_iou_fg_tol2"],
            adaptation_change_two_class=v["sar"][
                "delta_adaptation_only_miou_tol2"],
            diverged=v["sar"]["diverged"],
            inference_batch_size=v["inference_batch_size"]))

    for fam, label in (("family_1_bsm_vs_vanilla", "BSM vs Vanilla"),
                       ("family_2_bsm_vs_upper_bound",
                        "BSM vs the within-city upper bound")):
        for metric_key, rows in bsm[fam].items():
            for r in rows:
                tests.append(dict(
                    name=r["test_name"], family=f"{label} / {metric_key}",
                    compares=[r["bsm_run_id"], r["comparator_run_id"]],
                    statistic=r["delta_mean"], stderr=r["stderr"], n=r["n"],
                    ci95=r["ci95"], ci_method=r["ci_method"],
                    p=r["p_raw"], p_holm=r["p_holm"],
                    correction=bsm["bootstrap"]["multiplicity"],
                    B=r["B"], two_tailed=True))
    tests.append(dict(
        name="CrossEarth per-cell paired bootstraps",
        family="CrossEarth LOCO vs its own within-city upper bound",
        compares=["crossearth-{method}-loco-r450-{city}-hr-s0",
                  "crossearth-vanilla-upper-r450-{city}-hr-s0"],
        B=ce_pb["bootstrap"]["B"], two_tailed=True,
        correction=ce_pb["multiplicity"],
        rows_in="crossearth.all_150 and crossearth.nonempty_subset",
        p=None,
        p_absent_reason="the per-cell p-values live inside the copied "
                        "per_method_summary and join_assertions blocks of the "
                        "source files and are not flattened here"))

    out = dict(
        id="13_armF_internal_crossearth_bsm_sar",
        arm="F",
        answers=sorted(set(ce_pb["weakness_ids"]) | set(bsm["weakness_ids"])
                       | {"AC3"}),
        question="On the internal benchmark, how far do CrossEarth, batch style "
                 "mixing and test-time SAR close the LOCO-versus-upper gap, and "
                 "how far do they remain from the within-city upper bound?",
        status="PROMOTED_EVIDENCE",
        promotion_note="This JSON computes nothing. Every value is copied from "
                       "the arm F artifacts listed in source_artifacts. The "
                       "identity checks the script performs (each SAR cell's "
                       "copied adaptation change following from its copied "
                       "step-0 and final values, the BSM cell count and the "
                       "collapsed-cell list) are checks on the copy, not new "
                       "statistics.",
        metric=dict(
            crossearth=dict(
                empty_foreground_handling=ce_pb["empty_foreground_handling"],
                metric_module_sha=ce_pb["metric_module_sha"]),
            bsm=dict(metrics=bsm["metrics"],
                     empty_foreground_handling=bsm["empty_foreground_handling"],
                     bootstrap=bsm["bootstrap"]),
            sar=dict(name="iou_fg_tolerant_K2 (arm F's reported metric), with "
                          "the two-class K=2 value derived from the same stored "
                          "counts",
                     naming_correction_applied="2026-07-31; see each cell's "
                                               "metric_naming_correction block")),
        crossearth=dict(
            architecture=ce_pb["architecture"],
            n_cells=15,
            cells_present=["crossearth-vanilla-upper-r450-{city}-hr-s0",
                           "crossearth-vanilla-loco-r450-{city}-hr-s0",
                           "crossearth-fda-loco-r450-{city}-hr-s0",
                           "crossearth-sib-loco-r450-{city}-hr-s0",
                           "crossearth-bsm-loco-r450-{city}-hr-s0"],
            per_method_summary=ce_pb["per_method_summary"],
            stored_vs_degenerate_rule=ce_pb["stored_vs_degenerate_rule"],
            probs_npz_roundtrip_check=ce_pb["probs_npz_roundtrip_check"],
            join_assertions=ce_pb["join_assertions"],
            bootstrap=ce_pb["bootstrap"],
            multiplicity=ce_pb["multiplicity"],
            sp_gap=dict(per_cell_aurc=ce_sp["per_cell_aurc"],
                        sp_gaps=ce_sp["sp_gaps"],
                        logit_drift=ce_sp["logit_drift"],
                        bootstrap=ce_sp["bootstrap"],
                        join_assertions=ce_sp["join_assertions"]),
            nonempty_subset=dict(
                per_cell_subset_report=ce_ne["per_cell_subset_report"],
                all_150=ce_ne["all_150"],
                nonempty_subset=ce_ne["nonempty_subset"],
                all_150_vs_stored_analysis=ce_ne["all_150_vs_stored_analysis"],
                empty_foreground_handling=ce_ne["empty_foreground_handling"],
                bootstrap=ce_ne["bootstrap"]),
            locus=dict(source=CE_LOCUS, tables=md_tables(locus_text),
                       verbatim=locus_text,
                       note="App G.8 city-identity probe (part A) and App G.9 "
                            "recovery ratio R (part B) for CrossEarth. The "
                            "summary is markdown, so its cells are carried as "
                            "the strings it prints rather than reparsed into "
                            "floats.")),
        bsm=dict(
            input_cells=bsm["input_cells"],
            comparator_cells=bsm["comparator_cells"],
            comparator_resolution=bsm["comparator_resolution"],
            collapsed_cells=bsm["collapsed_cells"],
            per_cell=bsm["per_cell"],
            family_1_bsm_vs_vanilla=bsm["family_1_bsm_vs_vanilla"],
            family_2_bsm_vs_upper_bound=bsm["family_2_bsm_vs_upper_bound"],
            per_architecture_summary=bsm["per_architecture_summary"],
            join_checks=bsm["join_checks"],
            not_computed=bsm["not_computed"],
            target_access=bsm["target_access"]),
        sar=dict(n_cells=len(sar), per_cell=sar,
                 target_access="deployment_time",
                 note="each SAR row is one GPU pass over the published "
                      "checkpoint with adaptation off and on; the reported "
                      "change is the within-pass difference between the final "
                      "step and step 0"),
        conditions=conditions,
        tests=tests,
        runs=sorted({c["run_id"] for c in bsm["input_cells"]}
                    | {c["vanilla_run_id"] for c in bsm["comparator_cells"]}
                    | {c.get("upper_run_id") for c in bsm["comparator_cells"]
                       if c.get("upper_run_id")}
                    | {f"crossearth-{t}-r450-{c}-hr-s0"
                       for t in ("vanilla-upper", "vanilla-loco", "fda-loco",
                                 "sib-loco", "bsm-loco") for c in CITIES}
                    | {f"armF_sar_{a}_{c}_highres" for a in ARCHS
                       for c in CITIES}),
        runs_excluded=[dict(run_id="dinov3-bsm-loco-r450-chicago-hr-s0",
                            reason="training collapse at seed 0, best "
                                   "validation mIoU 5.51 against the codebase "
                                   "floor of 10.0; no test result, never "
                                   "imputed")],
        index_sha256=None,
        index_sha256_reason="this job reads arm F results JSONs, not a dump "
                            "tree, so no INDEX.parquet enters it",
        source_artifacts=[
            source(CE_PB, "CrossEarth per-method gaps and per-cell paired "
                          "bootstraps"),
            source(CE_SP, "CrossEarth selective-prediction (AURC) gaps and "
                          "logit drift"),
            source(CE_NE, "CrossEarth non-empty-foreground subset recompute"),
            source(CE_LOCUS, "CrossEarth App G.8 city-identity probe and App "
                             "G.9 recovery ratio R"),
            source(BSM, "batch style mixing: 11 cells, both comparison "
                        "families, per-architecture summary"),
        ] + [source(v["source"], f"internal SAR pass, {k}")
             for k, v in sorted(sar.items())],
        script="analysis/13_armF_internal_crossearth_bsm_sar.py",
        figure=None,
        caption=None,
        seed=0,
        seed_note="arm F's own seeds; this job runs no stochastic step. The "
                  "bootstraps copied here were run at B = 10,000 (CrossEarth "
                  "mIoU families, BSM) and B = 1,000 (CrossEarth SP gaps).",
        headline=(
            "On the internal benchmark CrossEarth's vanilla LOCO cells sit "
            f"{abs(ce_pb['per_method_summary'][0]['mIoU_boundary_tolerant_K2_pct']['mean_gap_over_3_cities']):.2f} "
            "two-class tolerant mIoU below CrossEarth's own within-city upper "
            "bound on average over three cities, with a worst single cell of "
            f"{abs(ce_pb['per_method_summary'][0]['mIoU_boundary_tolerant_K2_pct']['worst_case_single_cell_gap']):.2f} "
            f"on {ce_pb['per_method_summary'][0]['mIoU_boundary_tolerant_K2_pct']['worst_case_city']}; "
            f"batch style mixing is an 11-cell row whose worst gap to the upper "
            "bound is "
            f"{min(e['worst_gap_to_upper_mIoU_tol_K2'] for e in bsm['per_architecture_summary']):.2f} "
            "two-class mIoU at "
            f"{min(bsm['per_architecture_summary'], key=lambda e: e['worst_gap_to_upper_mIoU_tol_K2'])['arch']}/"
            f"{min(bsm['per_architecture_summary'], key=lambda e: e['worst_gap_to_upper_mIoU_tol_K2'])['worst_gap_to_upper_mIoU_tol_K2_city']} "
            "(gap is LOCO minus upper, so more negative is worse); and "
            "test-time SAR changes the foreground IoU of "
            "its nine internal cells by between "
            f"{min(v['sar']['delta_adaptation_only_iou_fg_tol2'] for v in sar.values()):+.4f} "
            f"and {max(v['sar']['delta_adaptation_only_iou_fg_tol2'] for v in sar.values()):+.4f} "
            "points."),
        sanity_flags=list(ce_pb.get("sanity_flags", []))
        + list(bsm.get("sanity_flags", []))
        + ["SAR's measured adaptation change on the nine internal cells is "
           f"between "
           f"{min(v['sar']['delta_adaptation_only_iou_fg_tol2'] for v in sar.values()):+.4f} "
           f"and "
           f"{max(v['sar']['delta_adaptation_only_iou_fg_tol2'] for v in sar.values()):+.4f} "
           "foreground-IoU points. Those are smaller than erratum E-9's "
           "OGLANet cell-mean standard deviation of up to 0.4247 mIoU, so on "
           "the three OGLANet cells SAR is not separable from doing nothing.",
           "Arm F's own SAR sanity check reports the unadapted pass matching "
           "each cell's stored test_results.json exactly on several cells and "
           "missing by -7.91 (dinov3/chicago) and +12.01 (dinov3/phoenix). "
           "Those two stored files are the artifacts that disagree with "
           "everything else: the same two cells match their PNG-scored "
           "published value to 0.0012 and 0.0057. The 2026-07-30 operator "
           "decision was to WARN and not stop, which is why it was never "
           "chased.",
           "The BSM per-cell block carries a deliberate definitional offset: "
           "each cell's stored test.miou_tol2 does not apply the Appendix E "
           "degenerate rule, so the promoted mIoU differs from the stored one "
           "by exactly 50 * n_degenerate / 150. The source asserts that "
           "identity at run time; it is a definition difference, not a "
           "reproduction failure.",
           "Promotion only: if any number here disagrees with its source file, "
           "the source file is right and this JSON is stale."],
        caveats=[
            # the one the operator's brief requires
            "CrossEarth was not run through the scaling ladder (arm C), for the "
            "same reason it was not run through arm D: it joined the project "
            "after those arms had run. That is a scheduling fact and not an "
            "implementation barrier. A CrossEarth ladder and augmentation grid "
            "was specced and submitted on 2026-10-27 as 30 cells; as of "
            "2026-10-28 three of the 30 had completed and no full-length "
            "CrossEarth cell had ever finished in that arm, so no CrossEarth "
            "rung-225 or rung-900 number exists. Every CrossEarth number here "
            "is rung 450 only.",
        ]
        + [f"Carried from {CE_PB}: {c}" for c in ce_pb["caveats"]]
        + [f"Carried from {CE_NE}: {c}" for c in ce_ne["caveats"]]
        + [f"Carried from {BSM}: {c}" for c in bsm["caveats"]]
        + [f"Carried from runs/armF_sar_*/results.json: {c}"
           for c in sar["mamnet/chicago"]["caveats"]]
        + ["BSM is an 11-cell row, not 12. DINOv3/chicago collapsed in "
           "training and is never imputed, so the DINOv3 BSM row is two "
           "cities.",
           "BSM and internal CrossEarth used a different 450-image draw from "
           "the published standard cells, with 150 validation images rather "
           "than 300: BSM's wrapper draws its own 450 (225 per source city) "
           "and CrossEarth's 450 comes from the Final_data_loco split "
           "directory, overlapping the shared list on 222, 234 and 225 images "
           "of 450. No BSM or CrossEarth cell is above 450.",
           "Every internal SAR pass ran at inference batch size 1, which is "
           "forced by SAR being episodic and per-image. For OGLANet that puts "
           "the whole SAR row in a different regime from the published "
           "batch-4 tables, because DFFM's 20 unregistered batch-statistic "
           "layers make the OGLANet score batch-size dependent: arm F's "
           "unadapted OGLANet pass sits 3.02, 8.14 and 0.21 foreground-IoU "
           "points below the same cells' PNG-scored published values, against "
           "0.0009 for MAMNet and 0.0057 for DINOv3. See "
           "delta_runs/dffm_defect/provenance.json.",
           "CrossEarth's within-city upper cells record target_access "
           "source_only in their own sidecars while arm C and arm D upper "
           "cells record target_labels. Erratum I-1's taxonomy makes every "
           "within-city upper bound target_labels; the sidecar disagreement is "
           "recorded and not edited.",
           "The CrossEarth SP gaps were bootstrapped at B = 1,000 while every "
           "other interval here is B = 10,000, so those intervals are not "
           "directly comparable in width with the rest.",
           "High resolution only, rung 450 only. The nine mid-resolution "
           "cells of the published grid have no counterpart in any of these "
           "three families.",
           "BSM's and SAR's gaps to the within-city upper bound here are NOT "
           "the same numbers as in results/12_internal_sib_worst_case_v2, and "
           "the difference is the comparator, not the method. Arm F compared "
           "each BSM cell against the arm C r450 retrained upper bound in "
           "dumps/st; results/12 compares it against the published upper bound "
           "scored from its prediction PNG masks. The two upper-bound "
           "populations differ by up to about 10 two-class mIoU on OGLANet, so "
           "for example arm F records BSM's worst gap as -16.16 at "
           "oglanet/chicago while results/12 records +17.40 (upper minus "
           "method) for the same cell. Cite one or the other and say which; "
           "results/12 is the declared-path number."],
        **env_block())

    write_results(out, "13_armF_internal_crossearth_bsm_sar", DRY_RUN)


if __name__ == "__main__":
    main()
