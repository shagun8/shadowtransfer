"""
Arm F / E-12 / E-2 -- results/31: results/27 with the OGLANet-SIB rows taken
from the HELD-OUT-CITY FDA pool retrains.

Answers BxRx-W3, AC3, v1UJ-W3, AC1 (results/27's IDs; no new weakness).

WHAT CHANGES AGAINST results/27, and nothing else:
  1. The three OGLANet-SIB rows. results/27 took them from
     oglanet-sibc4-loco-r450-{city}-hr-s42-n450, whose SIB FDA style pool was
     the other SOURCE city's train split (225 images after n450_inject). The
     paper's SIB draws that pool from the HELD-OUT city's unlabeled train
     split. The cells were rerun that way:
     oglanet-sibc4tgt-loco-r450-{city}-hr-s42-n450 (450 donor images, runtime
     donor assertions PASS; delta_logs/session_logs/
     2026-11-02-oglanet-sib-tgtfda-results.md). Same value field as results/27
     (two-class K=2 from raw counts, asserted against miou_tol_pct).
  2. OGLANet-SIB's access class: source_only -> unlabeled_target, from the new
     cells' run.json. The within-access-class rankings and the M12 source_only
     restriction downstream (analysis/32) follow it.
Every other row, bootstrap, reading rule and inherited caveat is results/27's
code, unchanged; vs_results_27 asserts that only the three OGLANet-SIB rows
moved.

Writes results/31_internal_sib_worst_case_tgtfda.json and analysis/_inv/
INDEX_{tree}_31.parquet. Never rewrites the _v3, _26, _27 or _28 indexes, any
dump, or any existing results JSON.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/31_internal_sib_worst_case_tgtfda.py [--dry-run [--dry-dump=PATH]]

=============================================================================
THE REST OF THIS DOCSTRING IS results/27's, CARRIED VERBATIM. Where it says
"this job" it describes the computation 31 inherits unchanged.
=============================================================================

Arm F / E-12 / E-2 -- internal SIB worst-case adjudication on the matched-450
retrains, with the MAMNet SAR rows MEASURED on the n450 checkpoints and the
OGLANet-SIB published comparator relabelled.

Answers BxRx-W3, AC3, v1UJ-W3, and AC1 on the budget confound.

This is analysis/26_internal_sib_worst_case_n450.py run with --sar-n450, as a
separate script writing a separate results JSON. results/26 is never
overwritten; it is read once, read-only, to build the vs_results_26 side table,
exactly as results/22 is read to build v3_vs_n450.

WHAT THIS JOB CHANGES AGAINST results/26, and nothing else:

  1. The three MAMNet SAR rows. results/26 applied SAR's within-pass change as
     measured on the PUBLISHED 900-image checkpoints to the matched-450
     unadapted level, an ASSUMPTION recorded as its caveat 4b. The SAR rerun on
     the n450 checkpoints has since landed
     (delta_runs/armF_sar_mamnet_{city}_highres-n450, 3/3 cells .DONE, hard
     n450 gate passed on all three; delta_logs/session_logs/
     2026-10-31-n450-sar-closed.md), so those three rows now carry a change
     measured on the same checkpoint their unadapted level comes from and
     caveat 4b's budget assumption is DROPPED for them. It is KEPT for the
     OGLANet and DINOv3 SAR rows, which have no n450 rerun and need none: their
     Vanilla cells were already at 450, so no budget assumption was ever made
     on them. Caveat 4 -- the batch-1 versus batch-4 regime mismatch -- applies
     to all nine SAR rows exactly as before and is untouched by this job.

  2. The OGLANet-SIB published comparator. results/26 carried the published
     OGLANet-SIB miami and phoenix values as Table 38 BOUNDS, because no dump
     exists for either cell. They are now relabelled as MEASURED fp16
     trainer-pass values, 66.7057610118262 and 63.9668489688177 two-class K=2,
     read from each published cell's own test_results.json (tolerant_2px),
     which is the published Table 38 source
     (oglanet/train_oglanet_sib.py:test_and_save_predictions, batch 8, under
     autocast). Source: delta_logs/session_logs/
     2026-10-31-oglanet-sib-n450-diagnosis.md section 12, step 3. The published
     comparator rule for OGLANet-SIB is fp16 against fp16 on all three cities;
     chicago's fp32 re-score of its stored probability maps, 71.7940216, is
     recorded beside its fp16 value and is NOT used as the comparator.

  3. One new caveat on the OGLANet inference offset measured in that log's
     section 12.1, which is a finding about the OGLANet code path rather than
     about any number here.

Nothing else moves. Every row value, every gap, every test and every other
caveat is results/26's computation unchanged, which is why the two JSONs are
comparable cell by cell in vs_results_26.

WHY A SEPARATE SCRIPT. v3's --n450 switch was written before any matched-450
dump existed and it REFUSES any OGLANet n450 dump whose directory name does not
carry a `bs4` suffix. That rule is wrong for the dumps that actually landed:
oglanet-segdesic chicago and miami were already scored from a batch-4 pass and
record `inference_batch_size: 4` in run.json without a `-bs4` directory name, and
the three oglanet-sibc4 cells were scored at batch 8 under fp16 autocast, which
is the regime the published Table 38 OGLANet-SIB row itself was produced in. So
the comparability test here reads the recorded batch size and compares it against
the batch size the cell's own published comparator was produced at, rather than
pattern-matching a directory name. Everything else in the computation is v3's,
unchanged.

WHAT --n450 CHANGES. 27 of the grid's rows are substituted, exactly the 27 cells
results/21_disclosure_record.json finding 1 measured at 900 source images:
MAMNet standard and its six adaptation methods (21 cells), OGLANet SegDeSiC (3)
and OGLANet SIB (3). Every within-city upper bound already trained on 450 and is
unchanged, which is the point of the exercise. Two consequences worth stating up
front rather than discovering in a table:
  * SIB becomes a 9-of-9 row. OGLANet-SIB miami and phoenix had no dump anywhere
    on the published path and entered v3 only as Table 38 bounds; their n450
    retrains exist, so the matched-cell comparison in this job is over all nine
    cells and no longer favours SIB by construction.
  * the internal 9-cell grid becomes budget-matched end to end. The 27 retrains
    are at 450, MAMNet-SIB's E-12 clean retrains are at 450, and every other
    cross-city cell and every upper bound was already at 450.

SAR. --sar-n450 is ON and not optional in this job; running it off would
duplicate results/26 under a new number. The flag is resolved PER ARCHITECTURE,
which is the only resolution that is correct here:

  * an architecture whose Vanilla cell IS substituted by a matched-450 retrain
    (MAMNet, all three cities) REQUIRES an n450 SAR run, because otherwise its
    SAR row mixes a 450-image unadapted level with a change measured on a
    900-image checkpoint. If one is absent the job fails loudly.
  * an architecture whose Vanilla cell is NOT substituted (OGLANet, DINOv3 --
    they were already at 450) takes its published SAR run, because there is no
    budget mismatch to remove and no n450 SAR run exists or is needed.

The n450 SAR runs landed under delta_runs/, not runs/, so both roots are
searched and the one that supplied each row is recorded on it. The two run
families also use different key names for the same quantities
(delta_adaptation_only_miou_k2_twoclass against delta_adaptation_only_miou_tol2,
unadapted.miou_k2_twoclass against unadapted.miou_tol2_derived_from_counts.
value); both spellings are read, neither is guessed, and a results.json carrying
neither is a hard stop.

DECLARED PATH
  metric      two-class average boundary-tolerant mIoU at K=2, in percent (the
              Table 38 / revised Table 42 family), computed per image from raw
              tp/fp/fn/tn on the valid region and macro-averaged over the cell's
              150 test images
  degenerate  App E rule: an image with tp+fp+fn == 0 at K=2 scores foreground
              IoU 0 and background IoU 100, so its two-class value is exactly 50
  resolution  high resolution only
  rung        450
  published   dumps/st_r450_published_png/ -- the saved prediction PNG masks,
              which are the tree Table 38 and final_comparison.json were scored
              from. 85 cells, the nine high-resolution cross-city Vanilla cells
              included.
  n450        dumps/st_r450_published_recompute/*-s42-n450[-bs4] -- the
              matched-450 retrains, scored in-job by n450_score.py from each
              cell's own probability maps, one scorer sha on all 27.
  fallback    dumps/st_r450_published_recompute/ (published cells re-passed over
              their own stored float16 probability maps), used ONLY where the PNG
              tree does not carry a published cell, and recorded per row.

GAP SIGN. CONVENTIONS.md fixes `gap` = LOCO minus the matched within-city upper
bound, so the paper's gap is negative. The operator's brief asks for `upper minus
method`, which is the same quantity with the sign flipped and is the one the
worst-case claim is phrased in. Both are emitted on every row:
  gap                       = method - upper   (CONVENTIONS.md sign, negative)
  gap_upper_minus_method    = upper - method   (the brief's sign, positive)
"worst case" always means the largest `gap_upper_minus_method` over the cells of
a scope, i.e. the deployment-relevant shortfall.

Read-only on every dump tree and on every SAR run directory. No training, no
GPU inference, no checkpoint loaded. Nothing under dumps/ or delta_runs/ and no
existing file under results/ is modified.

Writes results/27_internal_sib_worst_case_n450_sar.json and four tree indexes
under analysis/_inv/ with a _27 suffix (CLAUDE.md requires selection from an
index; the brief forbids writing inside dumps/, and the trees' own
INDEX.parquet, the _v3 indexes results/22 cites by sha256 and the _26 indexes
results/26 cites by sha256 are all left untouched).

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/27_internal_sib_worst_case_n450_sar.py [--dry-run]
"""

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
# results/31: a dry run may dump the would-be JSON to a path OUTSIDE the project
# (the session scratchpad) so the carried prose can be reviewed before the full
# pass. Never under results/.
DRY_DUMP = next((a.split("=", 1)[1] for a in sys.argv
                 if a.startswith("--dry-dump=")), None)
if DRY_DUMP and "/ShadowTransfer/" in os.path.abspath(DRY_DUMP):
    raise SystemExit("--dry-dump must point outside the project tree")
# This IS the --n450 job. The switch is not optional here: running it off would
# duplicate results/22 under a new number.
N450 = True
# This IS the --sar-n450 job. Also not optional: running it off would duplicate
# results/26. Resolved per architecture in resolve_sar_run().
SAR_N450 = True
B = 10000                            # paired bootstrap replicates, everywhere
BOOT_SEED = 0

PNG = os.path.join(BASE, "dumps", "st_r450_published_png")
RECOMPUTE = os.path.join(BASE, "dumps", "st_r450_published_recompute")
ST = os.path.join(BASE, "dumps", "st")
E12 = os.path.join(BASE, "delta_dumps", "dumps", "e12")
INV = os.path.join(BASE, "analysis", "_inv")

# Roots searched for an arm F SAR run, in order. The published grid is under
# runs/; the n450 rerun landed under delta_runs/. Both are searched and the root
# that supplied a row is recorded on it, so no row's provenance depends on which
# directory the operator happened to copy into.
SAR_ROOTS = [("runs", os.path.join(BASE, "runs")),
             ("delta_runs", os.path.join(BASE, "delta_runs"))]

# Trees searched for an n450 counterpart, in order. The 27 that landed are all
# under st_r450_published_recompute; the other roots are kept so a later copy
# needs no code change.
N450_TREES = [("st_r450_published_png", PNG),
              ("st_r450_published_recompute", RECOMPUTE),
              ("st", ST),
              ("st_n450", os.path.join(BASE, "dumps", "st_n450"))]

# The inference batch size each n450 cell's PUBLISHED comparator was produced
# at, and therefore the batch size the n450 cell's own scored pass must match for
# the pair to be comparable. OGLANet scores are batch-size dependent
# (results/21_disclosure_record.json finding 3: DFFM rebuilds 20 unregistered
# batch-statistic BatchNorms inside forward()), so for OGLANet this is a
# comparability requirement and for MAMNet and DINOv3 it is a record.
#   * the published OGLANet classic tables are batch-4 fp32
#     (run_inference.sh:73, run_inference_probs.sh:77)
#   * the published Table 38 OGLANet-SIB row came from train_oglanet_sib.py's own
#     test pass at args.batch_size 8 under fp16 autocast, not from
#     run_inference_probs.py at batch 4. run_inference_probs.py cannot load a SIB
#     checkpoint at all (no SIB model_variant; prob_variant is None), which is
#     why job 23273276 failed, so no batch-4 OGLANet-SIB pass exists or can be
#     produced with the released code.
REQUIRED_INFER_BS = {("oglanet", "segdesic"): 4, ("oglanet", "sib"): 8}
REQUIRED_INFER_BS_DEFAULT = 4

ARCHS = ("mamnet", "oglanet", "dinov3")
CITIES = ("chicago", "miami", "phoenix")
CE = "crossearth"

E9_MAX_SD5 = 0.4247                  # erratum E-9's measured OGLANet band
E9_MAX_SPAN9 = 1.1158

# Printed Table 38 values, used ONLY in agreement checks and in the explicit
# missing-cell bound. Nothing is rounded to match them.
TABLE38 = {
    ("upper", "mamnet"): (86.31, 79.11, 73.75),
    ("upper", "oglanet"): (79.12, 73.89, 66.15),
    ("upper", "dinov3"): (80.27, 76.00, 67.82),
    ("vanilla", "mamnet"): (79.72, 74.64, 62.84),
    ("vanilla", "oglanet"): (68.47, 66.18, 61.18),
    ("vanilla", "dinov3"): (67.26, 72.61, 65.24),
    ("fda", "mamnet"): (81.88, 74.21, 67.77),
    ("fda", "oglanet"): (71.07, 67.30, 63.27),
    ("fda", "dinov3"): (72.26, 70.81, 58.72),
    ("segdesic", "mamnet"): (82.44, 73.14, 58.80),
    ("segdesic", "oglanet"): (66.47, 59.90, 61.17),
    ("segdesic", "dinov3"): (75.52, 71.18, 68.04),
    ("iim", "mamnet"): (81.06, 70.05, 55.43),
    ("iim", "oglanet"): (70.01, 66.36, 55.25),
    ("iim", "dinov3"): (77.43, 72.23, 56.51),
    ("isw", "mamnet"): (75.23, 67.74, 51.08),
    ("isw", "oglanet"): (64.19, 67.78, 59.69),
    ("isw", "dinov3"): (57.60, 59.86, 55.68),
    ("mrfp_plus", "mamnet"): (75.76, 72.80, 53.35),
    ("mrfp_plus", "oglanet"): (63.82, 57.30, 58.03),
    ("mrfp_plus", "dinov3"): (77.46, 73.20, 60.78),
    ("fada", "mamnet"): (79.59, 60.90, 62.30),
    ("fada", "oglanet"): (66.17, 65.20, 60.26),
    ("fada", "dinov3"): (70.15, 71.43, 62.81),
    ("sib", "mamnet"): (81.65, 74.54, 68.64),
    ("sib", "oglanet"): (73.05, 66.71, 63.97),
    ("sib", "dinov3"): (76.57, 71.71, 61.45),
}

# The PUBLISHED OGLANet-SIB comparator, relabelled by this job.
#
# results/26 carried miami and phoenix as Table 38 BOUNDS, because neither cell
# has a dump in the PNG tree or the probability-map fallback. The operator's
# brief for this job relabels them as MEASURED fp16 trainer-pass values. They
# come from each published cell's own test_results.json field tolerant_2px,
# which IS the published Table 38 source: oglanet/train_oglanet_sib.py's
# test_and_save_predictions runs the test loader at args.batch_size 8 under
# autocast(enabled=args.use_amp) with use_amp defaulting True.
#   source: delta_logs/session_logs/2026-10-31-oglanet-sib-n450-diagnosis.md
#           section 12.3 (step 3), which prints the full-precision values and
#           the absolute path of each test_results.json
#   path  : <scratch>/.../oglanet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_
#           {city}__highres/test_results.json
#
# COMPARATOR RULE, as the brief states it: fp16 against fp16 for OGLANet-SIB on
# all three cities. chicago is the only one of the three with an fp32 re-score
# of its stored probability maps; that value is recorded beside its fp16 value
# and is never the comparator. The n450 cells on the other side of the contrast
# were themselves scored at batch 8 under fp16 autocast by the SIB trainer's own
# eval path, so the contrast is like-for-like on all three cities.
PUBLISHED_OGLANET_SIB_FP16 = {
    "chicago": dict(two_class=73.0475762379333, fg=50.827375740052645),
    "miami": dict(two_class=66.7057610118262, fg=38.165101266461726),
    "phoenix": dict(two_class=63.9668489688177, fg=31.376845952601848),
}
PUBLISHED_OGLANET_SIB_FP16_BASIS = (
    "measured fp16 trainer-pass value, read from the published cell's own "
    "test_results.json field tolerant_2px (mIOU / Shadow_IOU), which is the "
    "published Table 38 source: oglanet/train_oglanet_sib.py's "
    "test_and_save_predictions evaluates the test loader at args.batch_size 8 "
    "under autocast(enabled=args.use_amp), use_amp default True. Printed to "
    "full precision in delta_logs/session_logs/"
    "2026-10-31-oglanet-sib-n450-diagnosis.md section 12.3. This is a "
    "MEASUREMENT, not a bound: results/26 carried the miami and phoenix "
    "published values as Table 38 2-dp bounds because neither cell has a dump, "
    "and that framing is withdrawn here.")
# chicago's fp32 re-score of the stored sp_gap probability maps. Recorded, never
# used as the comparator. Two recipes exist and both are carried: nofilter is
# the one that matches the SIB test path, and the primary dump columns add
# filter_small_predictions(min_pixels=10).
PUBLISHED_OGLANET_SIB_FP32_RESCORE = {
    "chicago": dict(
        two_class_nofilter=71.79402162754555, fg_nofilter=48.50400555962078,
        two_class_dump_columns=71.7948714691302,
        fg_dump_columns=48.50426007665757,
        basis=("fp32 CPU re-score of the cell's stored sp_gap probability maps "
               "(sp_gap_analysis.py, no autocast), recorded in "
               "dumps/st_r450_published_recompute/"
               "oglanet-sibc4-loco-r450-chicago-hr-s0/run.json. NOT the "
               "comparator: the comparator rule is fp16 against fp16. The two "
               "regimes differ by 1.2536 two-class and 2.3234 fg on this cell, "
               "which is why that dump records table38_gate_pass false and "
               "repro_gate_pass false."),
        note="miami and phoenix have NO fp32 re-score at all "
             "(orig_dump_fg_iou_tol_pct null, metric_sha_check "
             "ORIGINAL_DUMP_PENDING): the section 4 AMP hold plus the blocked "
             "Fortress recall. The fp16 trainer pass is the only comparator "
             "available for those two, and by the rule above it is also the "
             "right one."),
}
# The n450 cells' OWN trainer-pass values, from the same log section, so the
# like-for-like 2x-data-against-budget-matched contrast can be read in the
# trainer's own regime as well as against this job's scored row values. The
# scored row value is what enters every number in this JSON; these are a record.
# results/31: these are now the HELD-OUT-CITY FDA pool cells'
# (oglanet-sibc4tgt-loco-r450-{city}-hr-s42-n450) own trainer-pass values. fg is
# run.json scoring.self_check.value (test_results.json tolerant_2px.Shadow_IOU),
# asserted against the dump in resolve_row. The trainer's TWO-CLASS value lives
# only in test_results.json on Delta, which was not copied to the analysis host, so it is
# None here and every field derived from it is None. results/27's values for the
# cross-source cells were 68.8706 / 64.2807 / 60.3659 two-class and
# 43.1906 / 35.3340 / 25.6316 fg.
N450_OGLANET_SIB_TRAINER = {
    "chicago": dict(two_class=None, fg=43.57054216836842),
    "miami": dict(two_class=None, fg=33.909206408994116),
    "phoenix": dict(two_class=None, fg=28.4824162946347),
}
N450_OGLANET_SIB_TRAINER_SUPERSEDED_R27 = {
    "chicago": dict(two_class=68.8706, fg=43.1906),
    "miami": dict(two_class=64.2807, fg=35.3340),
    "phoenix": dict(two_class=60.3659, fg=25.6316),
}

# Access classes: the I-1 countersigned taxonomy as the operator's brief states
# it. These OVERRIDE each dump's own target_access field, which on the PNG tree
# describes the scoring pass ("source_only; scoring pass over stored artifacts")
# and on arm F's CrossEarth upper cells reads source_only where arm C/D uppers
# read target_labels. Every disagreement is recorded per row.
ACCESS = {
    "upper": "target_labels",
    "vanilla": "source_only",
    "fda": "unlabeled_target",
    "segdesic": "unlabeled_target",
    "iim": "source_only",
    "isw": "source_only",
    "mrfp_plus": "source_only",
    "fada": "source_only",
    "bsm": "source_only",
    "sar": "deployment_time",
    "sib_mamnet": "unlabeled_target",
    # results/31: the OGLANet-SIB cells now draw their FDA style pool from the
    # held-out city's unlabeled train images (run.json target_access
    # unlabeled_target on all three), so the class follows the cells.
    "sib_oglanet": "unlabeled_target",
    "sib_dinov3": "source_only",
    "sib_crossearth": "source_only",
}

# results/31: inherited prose items that describe the superseded cross-source
# OGLANet-SIB cells. Each must hit exactly `expect` carried items.
STALE_MARKERS_31 = [
    dict(match="I-1 is later and makes OGLANet-SIB source_only, which is what is used here",
         expect=1,
         correction="results/31: OGLANet-SIB is unlabeled_target here. The "
                    "new cells' FDA style pool is the held-out city's "
                    "unlabeled train split, and their run.json records "
                    "unlabeled_target; I-1's source_only described the "
                    "cross-source donor, which these cells no longer use."),
    dict(match="MAMNet-SIB is unlabeled_target while OGLANet-SIB and DINOv3-SIB are source_only",
         expect=1,
         correction="results/31: MAMNet-SIB and OGLANet-SIB are "
                    "unlabeled_target; DINOv3-SIB and CrossEarth-SIB are "
                    "source_only. SIB still spans two access classes."),
    dict(match="n4. All three OGLANet SIB n450 cells stopped at the 100-epoch maximum",
         expect=1,
         correction="results/31: describes the superseded cross-source cells. "
                    "The sibc4tgt cells: chicago early-stopped at epoch 75 "
                    "(best 50, stopped_on early_stopping); miami 100 of 100 "
                    "(best 92) and phoenix 100 of 100 (best 97), both "
                    "stopped_on epochs_max."),
    dict(match="(-0.0099 chicago, -0.0097 miami, +0.0050 phoenix)",
         expect=1,
         correction="results/31: those self-check deltas and the five-pass "
                    "repeat-inference measurement are of the superseded "
                    "cross-source chicago cell. The sibc4tgt cells' "
                    "self-check deltas are -0.0161 chicago, -0.0104 miami, "
                    "+0.0073 phoenix (new caveat (d)); no repeat-inference "
                    "harness has been run on any sibc4tgt cell."),
    dict(match="sibc4 25.639580 against segdesic's superseded batch-4 pass 25.637529",
         expect=1,
         correction="results/31: the near-tie involves the superseded "
                    "cross-source OGLANet-SIB phoenix cell, which enters no "
                    "number here. The sibc4tgt phoenix cell scores 28.4987 "
                    "foreground IoU at K=2, so no SIB-SegDeSiC phoenix "
                    "near-tie exists in this JSON; the SegDeSiC side of the "
                    "flag is unchanged."),
]


# Pre-committed readings, verbatim from results/11_internal_sib_worst_case.json.
PRECOMMITTED = {
    "R1": ("SIB has the smallest pooled worst-case gap on the 9 original cells "
           "AND on the 12 cells incl. CrossEarth -> claim stands as written; "
           "print with the MAMNet seed band."),
    "R2": ("SIB smallest on 9, not on 12 -> claim scoped to the three "
           "architectures SIB was developed on (the posted AC wording). "
           "CrossEarth reported beside it."),
    "R3": ("Any other method's worst-case gap <= SIB's on the 9 cells -> "
           "\"smallest worst-case of any method\" is withdrawn. HLT call on the "
           "replacement sentence."),
    "R4": ("SIB's worst-case gap >= vanilla's on either scope -> \"smaller than "
           "doing nothing\" (contribution 4) is withdrawn."),
}

CONTRACT_COLS = {
    "run_id": "object", "image_id": "object", "city": "object", "res": "object",
    "lon": "float", "lat": "float", "source_tile_id": "object",
    "pair_id": "object",
    "tp_tol": "int", "fp_tol": "int", "fn_tol": "int", "tn_tol": "int",
    "tp_str": "int", "fp_str": "int", "fn_str": "int", "tn_str": "int",
    "n_valid_px": "int", "n_gt_fg_valid": "int", "empty_fg_valid": "bool",
    "iou_tol": "float", "iou_str": "float",
    "pbar_gt_shadow": "float", "pbar_gt_bg": "float",
    "logit_mean_tp": "float", "logit_mean_fp": "float",
    "logit_mean_fn": "float", "logit_mean_tn": "float",
    "aurc_fg": "float", "aurc_bg": "float", "aurc_all": "float",
    "aurc_ce_fg": "float", "aurc_ce_bg": "float",
    "ece_pred_pos": "float", "ece_pred_neg": "float", "n_pred_pos": "int",
    "n_gt_blobs": "int", "n_pred_blobs": "int",
    "gt_blob_area_median": "float", "small_object_frac": "float",
    "fn_px_by_cat": "object", "tp_px_by_cat": "object",
    "gt_px_by_cat": "object",
    "fp_px_near_gt": "int", "fp_px_far": "int",
    "gt_contrast_median": "float",
}

LOAD_BEARING = ("run_id", "image_id", "city", "res",
                "tp_tol", "fp_tol", "fn_tol", "tn_tol",
                "n_valid_px", "n_gt_fg_valid", "empty_fg_valid",
                "iou_tol", "iou_fg_tol_pct", "iou_bg_tol_pct",
                "miou_tol_pct")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fail(msg):
    raise SystemExit("ASSERTION FAILED: " + msg)


def _fmt_or_none(v):
    # results/31: a published cell with no dump has no published gap
    return ("unavailable (no published dump for this cell on the analysis host)"
            if v is None else f"{v:+.4f}")


# ----------------------------------------------------------------------------
# the one metric implementation on this box
# ----------------------------------------------------------------------------

def two_class_tol_k2(df):
    """Per-image two-class boundary-tolerant mIoU at K=2, percent.

    CONTRACT.md rule zero puts the canonical implementation in
    shadowtransfer/metrics.py on the SLURM box. No copy of that file is
    reachable from the analysis host (searched the project base, analysis/, python/ and the
    Shade Maps checkout), so the metric is recomputed here from the raw
    tp/fp/fn/tn counts that CONTRACT.md section 3 calls the source of truth, and
    is then asserted against the stored iou_fg_tol_pct / iou_bg_tol_pct /
    miou_tol_pct columns on every cell read. Any disagreement is a loud failure.
    """
    tp = df.tp_tol.to_numpy(dtype=np.float64)
    fp = df.fp_tol.to_numpy(dtype=np.float64)
    fn = df.fn_tol.to_numpy(dtype=np.float64)
    tn = df.tn_tol.to_numpy(dtype=np.float64)
    den_fg = tp + fp + fn
    den_bg = tn + fp + fn
    if (den_bg <= 0).any():
        fail("background IoU denominator is zero on some image")
    fg = np.where(den_fg > 0, tp / np.where(den_fg > 0, den_fg, 1.0), 0.0) * 100.0
    bg = tn / den_bg * 100.0
    return (fg + bg) / 2.0, fg, bg


def two_class_from_counts(tp, fp, fn, tn):
    """Same formula over bare count arrays, for the SAR perstep parquet."""
    tp, fp, fn, tn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn, tn))
    den_fg, den_bg = tp + fp + fn, tn + fp + fn
    if (den_bg <= 0).any():
        fail("background IoU denominator is zero on some SAR image")
    fg = np.where(den_fg > 0, tp / np.where(den_fg > 0, den_fg, 1.0), 0.0) * 100.0
    bg = tn / den_bg * 100.0
    return (fg + bg) / 2.0


def read_cell(root, run_id):
    d = os.path.join(root, run_id)
    pq = os.path.join(d, "perimage.parquet")
    sj = os.path.join(d, "run.json")
    if not os.path.isdir(d):
        return None, None, "no dump directory"
    if not os.path.isfile(pq):
        extra = (" (.FAILED marker present)"
                 if os.path.isfile(os.path.join(d, ".FAILED")) else "")
        return None, None, "no perimage.parquet" + extra
    if not os.path.isfile(sj):
        fail(f"{run_id}: perimage.parquet with no run.json sidecar "
             "(CONTRACT.md section 7)")
    df = pd.read_parquet(pq)
    meta = json.load(open(sj))
    missing = [c for c in CONTRACT_COLS if c not in df.columns]
    if missing:
        fail(f"{run_id}: CONTRACT.md section 3 columns absent: {missing}")
    deferred = []
    for col, kind in CONTRACT_COLS.items():
        dt = df[col].dtype
        if df[col].isna().all():
            deferred.append(col)
            continue
        ok = {"int": pd.api.types.is_integer_dtype,
              "float": pd.api.types.is_float_dtype,
              "bool": pd.api.types.is_bool_dtype,
              "object": lambda t: t == object}[kind](dt)
        if not ok:
            fail(f"{run_id}: column {col} dtype {dt} is not {kind}; never "
                 "coerce a mismatched dtype (CLAUDE.md)")
    for col in LOAD_BEARING:
        if col not in df.columns:
            fail(f"{run_id}: load-bearing column {col} absent")
        if df[col].isna().any():
            fail(f"{run_id}: load-bearing column {col} carries "
                 f"{int(df[col].isna().sum())} nulls")
    for col in ("tp_tol", "fp_tol", "fn_tol", "tn_tol"):
        if not pd.api.types.is_integer_dtype(df[col].dtype):
            fail(f"{run_id}: count column {col} dtype {df[col].dtype} is not "
                 "integer; never coerce a mismatched dtype")
    if len(df) != 150:
        fail(f"{run_id}: n_images {len(df)} != the expected test-pool size 150")
    m2, fg, bg = two_class_tol_k2(df)
    for name, got, stored in (("fg", fg, df.iou_fg_tol_pct.to_numpy()),
                              ("bg", bg, df.iou_bg_tol_pct.to_numpy()),
                              ("2class", m2, df.miou_tol_pct.to_numpy())):
        mx = float(np.abs(got - stored).max())
        if mx > 1e-9:
            fail(f"{run_id}: recomputed {name} IoU disagrees with the stored "
                 f"column by {mx:.3e}")
    meta["_deferred_cols"] = deferred
    return df, meta, None


# ----------------------------------------------------------------------------
# index
# ----------------------------------------------------------------------------

SCALAR = (str, int, float, bool, type(None))


def build_index(label, root, write_to=None):
    """CONTRACT.md section 5: one row per run_id, every scalar run.json field
    plus n_images, mean_iou_tol, parquet_mtime, dump_sha256.

    Every index this job selects from is built here and written outside the dump
    tree. The PNG tree arrived from Delta with no INDEX.parquet at all, and the
    recompute tree's on-disk index is stale (it predates the three trial cells
    copied after it was written). CLAUDE.md requires a current index and the
    operator's brief forbids writing inside dumps/, so the two instructions are
    satisfied by building the index here. The on-disk index of each tree that has
    one is still read, hashed and reported, including its staleness.
    """
    rows, unindexable = [], []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        pq, sj = os.path.join(d, "perimage.parquet"), os.path.join(d, "run.json")
        if not (os.path.isfile(pq) and os.path.getsize(pq) > 0
                and os.path.isfile(sj)):
            unindexable.append(dict(
                run_id=name, has_parquet=os.path.isfile(pq),
                parquet_bytes=os.path.getsize(pq) if os.path.isfile(pq) else 0,
                has_run_json=os.path.isfile(sj),
                failed_marker=os.path.isfile(os.path.join(d, ".FAILED"))))
            continue
        meta = json.load(open(sj))
        df = pd.read_parquet(pq, columns=["iou_tol"])
        r = {k: v for k, v in meta.items() if isinstance(v, SCALAR)}
        r.update(run_id=name, n_images=int(len(df)),
                 mean_iou_tol=float(df.iou_tol.mean()),
                 parquet_mtime=float(os.path.getmtime(pq)),
                 dump_sha256=sha256_file(pq))
        rows.append(r)
    idx = pd.DataFrame(rows)
    # run.json keys are not type-stable across arms (`probe` is an int on the
    # arm C probes and the string "noise" elsewhere), and parquet needs one type
    # per column. Columns carrying more than one python type are written as
    # strings, and which ones were cast is recorded rather than hidden.
    mixed = []
    for col in idx.columns:
        if idx[col].dtype != object:
            continue
        kinds = {type(v).__name__ for v in idx[col] if v is not None}
        if len(kinds) > 1:
            mixed.append(dict(column=col, types=sorted(kinds)))
            idx[col] = idx[col].map(lambda v: None if v is None else str(v))
    for rid, g in idx.groupby("run_id"):
        if "config_sha256" in g.columns and g.config_sha256.nunique() > 1:
            fail(f"{label}: run_id {rid} carries "
                 f"{g.config_sha256.nunique()} config_sha256 values")
    info = dict(label=label, root=os.path.relpath(root, BASE),
                n_rows=int(len(idx)), n_unindexable=len(unindexable),
                unindexable=unindexable, built_here=True,
                columns_cast_to_str=mixed)
    if write_to and not DRY_RUN:
        os.makedirs(os.path.dirname(write_to), exist_ok=True)
        tmp = write_to + ".new"
        idx.to_parquet(tmp, index=False)
        os.replace(tmp, write_to)
        info.update(path=os.path.relpath(write_to, BASE),
                    sha256=sha256_file(write_to))
    return idx, info


def on_disk_index_state(label, root):
    """Read, hash and age-check a tree's own INDEX.parquet. Reported, never
    selected from: this job selects from the indexes it builds itself, because
    it may not write inside dumps/ and a stale index may not be selected from.
    """
    p = os.path.join(root, "INDEX.parquet")
    if not os.path.isfile(p):
        return dict(label=label, path=os.path.relpath(p, BASE), present=False,
                    sha256=None, n_rows=None, stale=None,
                    note="no INDEX.parquet in this tree")
    idx = pd.read_parquet(p)
    mt = os.path.getmtime(p)
    newer = [n for n in sorted(os.listdir(root))
             if os.path.isdir(os.path.join(root, n))
             and os.path.isfile(os.path.join(root, n, "perimage.parquet"))
             and os.path.getmtime(os.path.join(root, n, "perimage.parquet")) > mt]
    absent = [n for n in sorted(os.listdir(root))
              if os.path.isdir(os.path.join(root, n))
              and os.path.isfile(os.path.join(root, n, "perimage.parquet"))
              and n not in set(idx.run_id.astype(str))]
    return dict(label=label, path=os.path.relpath(p, BASE), present=True,
                sha256=sha256_file(p), n_rows=int(len(idx)),
                stale=bool(newer or absent),
                dumps_newer_than_the_index=newer,
                dumps_absent_from_the_index=absent,
                note=("current" if not (newer or absent) else
                      "STALE: this job does not select from it and does not "
                      "rewrite it (the brief forbids writing inside dumps/)"))


# ----------------------------------------------------------------------------
# bootstrap
# ----------------------------------------------------------------------------

def paired_bootstrap(delta, b=B, seed=BOOT_SEED):
    """Paired image-level bootstrap on per-image deltas, two-tailed.

    Returns the point estimate, its standard error, the percentile 95% CI and a
    two-tailed p-value floored at 1/B (the arm F convention: p at 1e-04 reads as
    p <= 1/B).
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = delta.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    reps = delta[idx].mean(axis=1)
    point = float(delta.mean())
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    p = 2.0 * min(float((reps <= 0).mean()), float((reps >= 0).mean()))
    return dict(mean=point,
                stderr=float(delta.std(ddof=1) / np.sqrt(n)),
                n=int(n), ci95=[lo, hi], ci_method=f"paired_bootstrap_B{b}",
                p=float(min(1.0, max(1.0 / b, p))), B=b, two_tailed=True,
                bootstrap_sd=float(reps.std(ddof=1)))


def cluster_bootstrap_mean_over_cells(per_cell_deltas, b=B, seed=BOOT_SEED):
    """Mean over cells of a per-image paired delta, with an image-cluster
    bootstrap.

    per_cell_deltas maps a cluster key (the held-out city, whose 150 test images
    are shared by every cell of that city in identical order) to a list of
    per-image delta arrays. Images are resampled once per city and the same
    resampled index vector is applied to every cell of that city, so the
    clustering matches Appendix G.9's image-cluster bootstrap rather than
    treating two cells of one city as independent.
    """
    cells = [(k, a) for k, arrs in per_cell_deltas.items() for a in arrs]
    if not cells:
        return None
    point = float(np.mean([a.mean() for _, a in cells]))
    rng = np.random.default_rng(seed)
    keys = sorted(per_cell_deltas)
    n_by_key = {k: per_cell_deltas[k][0].size for k in keys}
    reps = np.empty(b, dtype=np.float64)
    for r in range(b):
        draw = {k: rng.integers(0, n_by_key[k], size=n_by_key[k]) for k in keys}
        reps[r] = np.mean([a[draw[k]].mean() for k, a in cells])
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    p = 2.0 * min(float((reps <= 0).mean()), float((reps >= 0).mean()))
    return dict(mean=point, stderr=float(reps.std(ddof=1)), n_cells=len(cells),
                n_images_total=int(sum(a.size for _, a in cells)),
                ci95=[lo, hi],
                ci_method=f"paired_image_cluster_bootstrap_B{b}",
                p=float(min(1.0, max(1.0 / b, p))), B=b, two_tailed=True)


def holm(pvals):
    """Holm-Bonferroni, returning adjusted p-values in input order.
    python/armL_results.py::holm's rule, reimplemented (that file is on Delta).
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


# ----------------------------------------------------------------------------
# row resolution
# ----------------------------------------------------------------------------
# One row per (method, arch, city). Each row records which tree supplied it and
# whether that is the declared PNG tree or the recorded fallback.

METHOD_ORDER = ["upper", "vanilla", "fda", "segdesic", "iim", "isw",
                "mrfp_plus", "fada", "sib", "bsm", "sar"]
METHOD_LABEL = {"upper": "Upper bound", "vanilla": "Vanilla", "fda": "FDA",
                "segdesic": "SegDeSiC", "iim": "IIM", "isw": "ISW",
                "mrfp_plus": "MRFP+", "fada": "FADA", "sib": "SIB",
                "bsm": "BSM", "sar": "SAR"}
BASELINES = ["fda", "segdesic", "iim", "isw", "mrfp_plus", "fada", "bsm", "sar"]


def declared_run_id(method, arch, city):
    """The run_id the declared path looks for, before any n450 substitution."""
    if arch == CE:
        tok = {"upper": "vanilla-upper", "vanilla": "vanilla-loco",
               "fda": "fda-loco", "sib": "sib-loco", "bsm": "bsm-loco"}[method]
        return f"{CE}-{tok}-r450-{city}-hr-s0"
    if method == "upper":
        return f"{arch}-none-upper-r450-{city}-hr-s0"
    if method == "vanilla":
        return f"{arch}-vanilla-loco-r450-{city}-hr-s0"
    if method == "sib":
        return f"{arch}-sibc4-loco-r450-{city}-hr-s0"
    return f"{arch}-{method}-loco-r450-{city}-hr-s0"


def infer_bs_of(run_id, meta):
    """The cell's scored-pass inference batch size, with its provenance.

    Two of the 27 n450 run.json files record `inference_batch_size: null`; both
    are single-cell or interactive resubmissions that predate the change which
    added the field. For those the value comes from the session log line below,
    cited rather than assumed. A cell with neither is a hard stop.
    """
    v = meta.get("inference_batch_size")
    if v is not None:
        return int(v), ("run.json field `inference_batch_size`"
                        + (f"; source: {meta['inference_batch_size_source']}"
                           if meta.get("inference_batch_size_source") else ""))
    log = {
        "mamnet-vanilla-loco-r450-phoenix-hr-s42-n450": (4,
            "run.json field is NULL; value 4 from "
            "delta_logs/session_logs/2026-10-28-n450-bs4-isw.md, STEP 5 table: "
            "\"| mamnet-vanilla-loco-r450-phoenix | **30.8823** | 28.4405 | "
            "**+2.4418** | 25 | 15 | early_stopping | **4** "
            "(`run_inference_probs.py`) |\""),
        "oglanet-segdesic-loco-r450-phoenix-hr-s42-n450": (4,
            "run.json field is NULL; value 4 from "
            "delta_logs/session_logs/2026-10-28-n450-bs4-isw.md, STEP 1 1a "
            "table. Not used for a value here: this cell is superseded by its "
            "-bs4 re-score, which records the field directly."),
    }
    if run_id in log:
        return log[run_id]
    fail(f"{run_id}: inference_batch_size is null in run.json and no session "
         "log line is registered for it")


def resolve_n450(method, arch, city):
    """Find the matched-450 retrain for this cell, if one exists.

    The retrain's run_id is the published cell's with its seed token replaced and
    a trailing `-n450` suffix, optionally `-n450-bs4` for a batch-4 re-score.

    This REPLACES v3's rule, which refused any OGLANet n450 dump whose directory
    name lacked a `bs4` suffix. That name test is wrong for the dumps that
    landed: oglanet-segdesic chicago and miami were scored from a batch-4 pass
    and record `inference_batch_size: 4` without a `-bs4` directory, and the
    three oglanet-sibc4 cells were scored at batch 8 fp16 because that is the
    regime their own published Table 38 row was produced in. The test here is on
    the RECORDED batch size against the batch size the cell's published
    comparator was produced at (REQUIRED_INFER_BS), which is what comparability
    actually requires. A candidate that fails it is reported as present and
    refused, never silently substituted. A `-bs4` re-score is preferred over its
    parent, as the operator's brief directs.
    """
    base = declared_run_id(method, arch, city)
    stem = re.sub(r"-s\d+$", "", base)
    found, refused = [], []
    # results/31: OGLANet-SIB's matched-450 cell is the HELD-OUT-CITY FDA pool
    # retrain (method tag sibc4tgt). The cross-source n450 cell results/27 used
    # is recorded as refused, by name, and is never substituted.
    if (arch, method) == ("oglanet", "sib"):
        old_stem = stem
        stem = f"oglanet-sibc4tgt-loco-r450-{city}-hr"
        for label, root in N450_TREES:
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                if re.match(re.escape(old_stem) + r"-s\d+-n450[-\w]*$", name):
                    refused.append(dict(
                        tree=label, run_id=name,
                        reason=("superseded in results/31: cross-SOURCE FDA "
                                "donor (the other source city's train split, "
                                "225 images after n450_inject). The paper's "
                                "SIB draws the FDA style pool from the "
                                "held-out city's train split; the sibc4tgt "
                                "cell is that configuration.")))
    pat = re.compile(re.escape(stem) + r"-s\d+-n450(?P<suffix>[-\w]*)$")
    need = REQUIRED_INFER_BS.get((arch, method), REQUIRED_INFER_BS_DEFAULT)
    for label, root in N450_TREES:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            m = pat.match(name)
            if not m:
                continue
            d = os.path.join(root, name)
            if not os.path.isfile(os.path.join(d, "perimage.parquet")):
                continue
            if not os.path.isfile(os.path.join(d, ".DONE")):
                refused.append(dict(tree=label, run_id=name,
                                    reason="no .DONE marker"))
                continue
            if os.path.isfile(os.path.join(d, ".FAILED")):
                refused.append(dict(tree=label, run_id=name,
                                    reason=".FAILED marker present"))
                continue
            meta = json.load(open(os.path.join(d, "run.json")))
            got, prov = infer_bs_of(name, meta)
            if got != need:
                refused.append(dict(
                    tree=label, run_id=name, inference_batch_size=got,
                    required_inference_batch_size=need,
                    batch_size_provenance=prov,
                    reason=(f"scored at inference batch size {got} but this "
                            f"cell's published comparator is batch {need}; "
                            "OGLANet scores are batch-size dependent "
                            "(results/21 finding 3) and a mismatch is not "
                            "substituted silently")))
                continue
            found.append(dict(tree=label, root=root, run_id=name,
                              suffix=m.group("suffix") or "",
                              inference_batch_size=got,
                              batch_size_provenance=prov))
    # a -bs4 re-score wins over its parent
    bs4 = [f for f in found if "bs4" in f["suffix"]]
    if bs4:
        for f in found:
            if f not in bs4:
                refused.append(dict(
                    tree=f["tree"], run_id=f["run_id"],
                    inference_batch_size=f["inference_batch_size"],
                    reason=("superseded by its -bs4 re-score, which is the pass "
                            "the operator's brief names")))
        found = bs4
    if len(found) > 1:
        fail(f"{stem}: {len(found)} equally preferred n450 dumps "
             f"{[f['run_id'] for f in found]}; the choice is not determined")
    return found, refused


def find_sar_run(arch, city, n450):
    """Locate an arm F SAR run directory, searching both roots.

    Returns (root_label, abs_path) for the first root that holds a complete run,
    or (None, None). A directory that exists but carries no results.json is NOT
    a run and is reported as absent by the caller, which is a hard stop wherever
    the run is required.
    """
    name = f"armF_sar_{arch}_{city}_highres" + ("-n450" if n450 else "")
    for label, root in SAR_ROOTS:
        p = os.path.join(root, name)
        if (os.path.isfile(os.path.join(p, "results.json"))
                and os.path.isfile(os.path.join(p, "perstep.parquet"))):
            return label, p
    return None, None


def sar_stated_delta(d, where):
    """SAR's own within-pass two-class delta, under either key spelling.

    The published grid writes delta_adaptation_only_miou_tol2; the n450 rerun
    writes delta_adaptation_only_miou_k2_twoclass. They are the same quantity
    under two names. Neither is guessed: a results.json carrying neither key is
    a hard stop, because the recomputed delta is asserted against this value and
    an absent assertion is not an assertion.
    """
    for k in ("delta_adaptation_only_miou_tol2",
              "delta_adaptation_only_miou_k2_twoclass"):
        if k in d["sar"]:
            return float(d["sar"][k]), k
    fail(f"{where}: results.json carries neither "
         "delta_adaptation_only_miou_tol2 nor "
         "delta_adaptation_only_miou_k2_twoclass, so the recomputed within-pass "
         "delta cannot be asserted against the run's own value")


def sar_own_unadapted(d, where):
    """The run's own unadapted two-class level, under either key spelling."""
    u = d["unadapted"]
    if isinstance(u.get("miou_tol2_derived_from_counts"), dict):
        return float(u["miou_tol2_derived_from_counts"]["value"]), \
            "unadapted.miou_tol2_derived_from_counts.value"
    if u.get("miou_k2_twoclass") is not None:
        return float(u["miou_k2_twoclass"]), "unadapted.miou_k2_twoclass"
    fail(f"{where}: results.json carries neither "
         "unadapted.miou_tol2_derived_from_counts.value nor "
         "unadapted.miou_k2_twoclass")


def resolve_row(method, arch, city):
    """Return the row's per-image two-class K=2 vector plus its provenance."""
    row = dict(method=method, method_label=METHOD_LABEL[method], arch=arch,
               city=city,
               access_class=ACCESS.get(f"{method}_{arch}", ACCESS.get(method)),
               table38_printed=(TABLE38[(method, arch)][CITIES.index(city)]
                                if (method, arch) in TABLE38 else None))
    n450_found, n450_refused = resolve_n450(method, arch, city)
    row["n450_candidates"] = [f"{f['tree']}/{f['run_id']}" for f in n450_found]
    row["n450_refused"] = n450_refused
    row["n450_substituted"] = False

    if N450 and n450_found:
        f0 = n450_found[0]
        label, root, rid = f0["tree"], f0["root"], f0["run_id"]
        df, meta, why = read_cell(root, rid)
        if df is None:
            fail(f"{rid}: selected as the n450 substitution but {why}")
        m2, fg, _ = two_class_tol_k2(df)
        # The published comparator, kept beside the n450 value on the row so the
        # v3-vs-n450 contrast is readable per cell without re-reading v3.
        prid = declared_run_id(method, arch, city)
        pdf, pmeta = None, None
        ptree = None
        for plabel, proot in (("st_r450_published_png", PNG),
                              ("st_r450_published_recompute", RECOMPUTE)):
            pdf, pmeta, _ = read_cell(proot, prid)
            if pdf is not None:
                ptree = plabel
                break
        pm2 = pfg = None
        if pdf is not None:
            pm2, pfg, _ = two_class_tol_k2(pdf)
        row.update(run_id=rid, tree=label,
                   source_kind=("matched-450 retrain, scored in-job by "
                                "n450_score.py from the cell's own probability "
                                f"maps at inference batch size "
                                f"{f0['inference_batch_size']}"),
                   on_declared_path=False, fell_back=False,
                   n450_substituted=True, n=int(len(df)),
                   value=float(m2.mean()), value_fg=float(fg.mean()),
                   per_image=m2, image_id=list(df.image_id),
                   n450_inference_batch_size=f0["inference_batch_size"],
                   n450_batch_size_provenance=f0["batch_size_provenance"],
                   n450_n_train=meta.get("n_train"),
                   n450_n_val=meta.get("n_val"),
                   n450_seed=meta.get("seed"),
                   n450_epochs_run=meta.get("epochs_run"),
                   n450_epochs_max=meta.get("epochs_max"),
                   n450_stopped_on=meta.get("stopped_on"),
                   n450_scorer_sha=meta.get("scorer_sha"),
                   n450_self_check_delta=(meta.get("scoring", {})
                                          .get("self_check", {}).get("delta")),
                   published_run_id=prid,
                   published_tree=ptree,
                   published_value=(float(pm2.mean()) if pm2 is not None
                                    else None),
                   published_value_fg=(float(pfg.mean()) if pfg is not None
                                       else None),
                   published_per_image=(pm2 if pm2 is not None else None),
                   published_n_train_measured=900,
                   published_n_train_basis=("results/21_disclosure_record.json "
                                            "finding 1, measured from the "
                                            "cells' own checkpoints"),
                   sidecar_target_access=meta.get("target_access"),
                   metric_module_sha=meta.get("metric_module_sha"),
                   empty_fg_images=int(df.empty_fg_valid.sum()),
                   blocker=None)
        # ---- the relabelled OGLANet-SIB published comparator -------------
        # Only these three cells. The comparator is the fp16 trainer pass on all
        # three cities, which is the published Table 38 source; for miami and
        # phoenix it is the ONLY published value that exists, and this job
        # records it as a MEASUREMENT rather than as the Table 38 bound
        # results/26 carried. The value is never a row value and never enters a
        # gap: the row is the n450 retrain, and the gap is against the
        # within-city upper bound. It is the published side of the
        # 2x-data-against-budget-matched contrast.
        if method == "sib" and arch == "oglanet":
            fp16 = PUBLISHED_OGLANET_SIB_FP16[city]
            # results/31: the trainer fg constant must be this dump's own
            sc_v = meta["scoring"]["self_check"]["value"]
            if abs(sc_v - N450_OGLANET_SIB_TRAINER[city]["fg"]) > 1e-12:
                fail(f"{rid}: run.json self_check value {sc_v!r} != "
                     f"N450_OGLANET_SIB_TRAINER fg "
                     f"{N450_OGLANET_SIB_TRAINER[city]['fg']!r}")
            if meta.get("fda_donor") != "heldout_city_train_images":
                fail(f"{rid}: fda_donor {meta.get('fda_donor')!r}, expected "
                     "heldout_city_train_images")
            if meta.get("target_access") != "unlabeled_target":
                fail(f"{rid}: target_access {meta.get('target_access')!r}")
            row.update(
                fda_donor=meta.get("fda_donor"),
                fda_donor_n=meta.get("fda_donor_n"),
                fda_donor_runtime_assertions_verdict=(
                    (meta.get("fda_donor_runtime_assertions") or {})
                    .get("verdict")),
                n450_self_check_pass_0p01=(meta.get("scoring", {})
                                           .get("self_check", {})
                                           .get("pass_0p01")),
                superseded_cross_source_run_id=(
                    f"oglanet-sibc4-loco-r450-{city}-hr-s42-n450"))
            row.update(
                published_comparator_value=fp16["two_class"],
                published_comparator_value_fg=fp16["fg"],
                published_comparator_regime="fp16 trainer pass, batch 8, "
                                            "autocast enabled",
                published_comparator_basis=PUBLISHED_OGLANET_SIB_FP16_BASIS,
                published_comparator_rule=(
                    "fp16 against fp16 for OGLANet-SIB. The n450 row was itself "
                    "scored at batch 8 under fp16 autocast by the SIB trainer's "
                    "own eval path, so the contrast is like-for-like on all "
                    "three cities."),
                published_comparator_is_a_bound=False,
                published_comparator_was_a_bound_in_results_26=(
                    city in ("miami", "phoenix")),
                published_fp32_rescore=PUBLISHED_OGLANET_SIB_FP32_RESCORE.get(
                    city),
                n450_trainer_pass=N450_OGLANET_SIB_TRAINER[city],
                delta_n450_scored_minus_published_comparator=float(
                    m2.mean() - fp16["two_class"]),
                delta_n450_trainer_minus_published_comparator=(
                    None if N450_OGLANET_SIB_TRAINER[city]["two_class"] is None
                    else float(N450_OGLANET_SIB_TRAINER[city]["two_class"]
                               - fp16["two_class"])),
                n450_trainer_two_class_absent_why=(
                    "results/31: the tgtfda cell's trainer two-class value is "
                    "in its test_results.json on Delta only; run.json carries "
                    "the trainer's fg value (scoring.self_check.value), which "
                    "is asserted here"),
                delta_definition=(
                    "delta_n450_scored_minus_published_comparator uses this "
                    "job's scored row value, which is the number every other "
                    "field here is computed from. "
                    "delta_n450_trainer_minus_published_comparator uses the "
                    "n450 cell's own trainer-pass value instead, so both sides "
                    "come from the trainer; that is the figure section 12.3 of "
                    "the diagnosis log prints. The two differ by the cell's "
                    "scorer self-check delta, which is recorded on this row as "
                    "n450_self_check_delta."))
            # chicago is the one city with a published dump, so its fp16
            # comparator is checkable against the PNG tree's own scoring of the
            # published prediction masks. Asserted, not assumed.
            if row.get("published_value") is not None:
                d16 = abs(row["published_value"] - fp16["two_class"])
                if d16 > 1e-6:
                    fail(f"sib/oglanet/{city}: the PNG tree scores the "
                         f"published cell at {row['published_value']:.9f} but "
                         f"the fp16 trainer-pass comparator is "
                         f"{fp16['two_class']:.9f}, |delta| {d16:.3e}. The "
                         "comparator rule says these are the same regime, so a "
                         "disagreement means one of them is not what it is "
                         "labelled.")
                row["published_comparator_png_agreement"] = dict(
                    png_tree_value=row["published_value"],
                    fp16_trainer_value=fp16["two_class"],
                    abs_delta=d16, tolerance=1e-6, agrees=True,
                    note="the PNG tree's scoring of this cell's published "
                         "prediction masks reproduces the fp16 trainer-pass "
                         "value, which is what makes the comparator rule "
                         "checkable on the one city that has a dump.")
        return row

    # --- MAMNet SIB: the E-12 clean-pool retrains, three seeds -------------
    if method == "sib" and arch == "mamnet":
        seeds, per_seed = {}, []
        for tag, seed in (("s42-e12clean", 42), ("s43-e12clean", 43),
                          ("s44-e12clean", 44)):
            rid = f"mamnet-sibc4-loco-r450-{city}-hr-{tag}"
            df, meta, why = read_cell(E12, rid)
            if df is None:
                fail(f"{rid}: {why}")
            m2, fg, _ = two_class_tol_k2(df)
            stated = meta.get("two_class_tol_k2_pct")
            if stated is not None and abs(float(m2.mean()) - stated) > 1e-4:
                fail(f"{rid}: declared path {m2.mean():.6f} disagrees with the "
                     f"sidecar's two_class_tol_k2_pct {stated:.6f}")
            seeds[seed] = m2
            per_seed.append(dict(seed=seed, run_id=rid,
                                 value=float(m2.mean()), value_fg=float(fg.mean()),
                                 target_access=meta.get("target_access")))
            ids = list(df.image_id)
        pub = f"mamnet-sibc4-loco-r450-{city}-hr-sPUB-e12orig"
        dfp, metap, _ = read_cell(E12, pub)
        m2p, _, _ = two_class_tol_k2(dfp)
        stack = np.vstack([seeds[s] for s in (42, 43, 44)])
        row.update(run_id=[r["run_id"] for r in per_seed],
                   tree="delta_dumps/dumps/e12",
                   source_kind="E-12 clean-pool retrain, seed mean over "
                               "{42, 43, 44}",
                   on_declared_path=True, fell_back=False,
                   n=int(stack.shape[1]),
                   value=float(stack.mean(axis=0).mean()),
                   value_fg=float(np.mean([r["value_fg"] for r in per_seed])),
                   per_image=stack.mean(axis=0), image_id=ids,
                   per_seed=per_seed,
                   seed_values=[r["value"] for r in per_seed],
                   seed_sd=float(np.std([r["value"] for r in per_seed], ddof=1)),
                   published_contaminated_value=float(m2p.mean()),
                   published_contaminated_repro_delta=metap.get(
                       "repro_delta_miou"),
                   sidecar_target_access=per_seed[0]["target_access"],
                   metric_module_sha=metap.get("metric_module_sha"),
                   empty_fg_images=int(dfp.empty_fg_valid.sum()),
                   blocker=None)
        return row

    # --- SAR: declared-path unadapted published cell + SAR's own change ----
    if method == "sar":
        # --sar-n450 is resolved PER ARCHITECTURE. The unadapted level a SAR row
        # sits on is the matched-450 Vanilla cell wherever one exists, so the
        # within-pass change added to it must have been measured on the same
        # checkpoint family. An architecture whose Vanilla cell IS substituted
        # therefore REQUIRES an n450 SAR run and a missing one is a hard stop;
        # an architecture whose Vanilla cell is not substituted (OGLANet,
        # DINOv3, already at 450) takes its published run, because there is no
        # budget mismatch to remove.
        vfound, vrefused = resolve_n450("vanilla", arch, city)
        needs_n450_sar = bool(vfound)
        pub_label, pub_dir = find_sar_run(arch, city, n450=False)
        n450_label, n450_dir = find_sar_run(arch, city, n450=True)
        row["sar_n450_required"] = needs_n450_sar
        row["sar_n450_required_why"] = (
            "this architecture's Vanilla cell IS replaced by a matched-450 "
            "retrain, so the SAR row's unadapted level is at 450 and the "
            "within-pass change added to it must be measured on the same "
            "checkpoint family" if needs_n450_sar else
            "this architecture's Vanilla cell is NOT replaced -- it was already "
            "at 450 -- so there is no budget mismatch for an n450 SAR run to "
            "remove, and none exists or is needed")
        row["sar_n450_dir_present"] = n450_dir is not None
        row["sar_published_dir_present"] = pub_dir is not None
        if needs_n450_sar:
            if n450_dir is None:
                fail(f"sar/{arch}/{city}: this cell's Vanilla row is a "
                     "matched-450 retrain, so an n450 SAR run is REQUIRED, and "
                     "no armF_sar_" + f"{arch}_{city}" + "_highres-n450 "
                     "directory with both a results.json and a perstep.parquet "
                     "exists under "
                     + " or ".join(lbl for lbl, _ in SAR_ROOTS)
                     + ". This job does not fall back to the published-model "
                       "run: that fallback IS results/26's caveat 4b and is "
                       "what this job exists to remove.")
            p, root_label = n450_dir, n450_label
            sar_kind = "SAR rerun on the n450 checkpoints"
        else:
            if pub_dir is None:
                row.update(run_id=None, tree=None, source_kind=None,
                           on_declared_path=False, fell_back=False, value=None,
                           per_image=None,
                           blocker="no arm F SAR run on this box")
                return row
            p, root_label = pub_dir, pub_label
            sar_kind = "published-model SAR run"
        row["sar_run_dir"] = os.path.relpath(p, BASE)
        row["sar_run_root"] = root_label
        row["sar_run_kind"] = sar_kind
        row["sar_n450_measured"] = bool(needs_n450_sar)
        rj, pp = os.path.join(p, "results.json"), os.path.join(p, "perstep.parquet")
        if not (os.path.isfile(rj) and os.path.isfile(pp)):
            row.update(run_id=None, tree=None, source_kind=None,
                       on_declared_path=False, fell_back=False, value=None,
                       per_image=None, blocker="no arm F SAR run on this box")
            return row
        d = json.load(open(rj))
        ps = pd.read_parquet(pp)
        lab0 = ps[ps.pass_id == "unadapted"]
        if lab0.empty:
            fail(f"armF_sar_{arch}_{city}: no unadapted pass in perstep.parquet")
        sarp = ps[ps.pass_id != "unadapted"]
        if sarp.empty:
            fail(f"armF_sar_{arch}_{city}: no adapted pass in perstep.parquet")
        # step 0 and the final step of the SAR pass itself: the within-pass delta
        step0 = sarp[sarp.step == sarp.step.min()].sort_values("image_id")
        stepN = sarp[sarp.step == sarp.step.max()].sort_values("image_id")
        if list(step0.image_id) != list(stepN.image_id):
            fail(f"armF_sar_{arch}_{city}: SAR step-0 and final-step image sets "
                 "differ; a paired delta is not defined")
        m0 = two_class_from_counts(step0.tp_k2, step0.fp_k2, step0.fn_k2,
                                   step0.tn_k2)
        mN = two_class_from_counts(stepN.tp_k2, stepN.fp_k2, stepN.fn_k2,
                                   stepN.tn_k2)
        delta_per_image = mN - m0
        stated, stated_key = sar_stated_delta(d, f"armF_sar_{arch}_{city}")
        if abs(float(delta_per_image.mean()) - stated) > 1e-6:
            fail(f"armF_sar_{arch}_{city}: recomputed within-pass two-class "
                 f"delta {delta_per_image.mean():.8f} disagrees with the run's "
                 f"own {stated_key} {stated:.8f}")
        # The published unadapted level. v3 change: the declared PNG tree now
        # carries all nine cross-city Vanilla cells, so the SAR unadapted level
        # is read from the PNG tree and the probability-map tree is only a
        # recorded fallback, as it is for every other row.
        # n450 change: the unadapted level is the MATCHED-450 Vanilla cell
        # wherever one exists, so a SAR row sits on the same training budget as
        # every other row of its architecture. Only MAMNet has one; OGLANet and
        # DINOv3 Vanilla were already 450 and are unchanged.
        base_rid = declared_run_id("vanilla", arch, city)
        bdf = bmeta = None
        base_tree, base_fell_back, bwhy = None, None, None
        base_is_n450 = False
        # vfound / vrefused were resolved at the top of this branch, where they
        # decided whether an n450 SAR run is required. Reused, not re-resolved.
        row["sar_unadapted_n450_candidates"] = [
            f"{f['tree']}/{f['run_id']}" for f in vfound]
        row["sar_unadapted_n450_refused"] = vrefused
        if vfound:
            f0 = vfound[0]
            bdf, bmeta, bwhy = read_cell(f0["root"], f0["run_id"])
            if bdf is None:
                fail(f"{f0['run_id']}: selected as the SAR unadapted level but "
                     f"{bwhy}")
            base_rid, base_tree = f0["run_id"], f0["tree"]
            base_fell_back, base_is_n450 = False, True
        else:
            for blabel, broot in (("st_r450_published_png", PNG),
                                  ("st_r450_published_recompute", RECOMPUTE)):
                bdf, bmeta, bwhy = read_cell(broot, base_rid)
                if bdf is not None:
                    base_tree = blabel
                    base_fell_back = (blabel == "st_r450_published_recompute")
                    break
        if bdf is None:
            row.update(run_id=None, tree=None, source_kind=None,
                       on_declared_path=False, fell_back=True, value=None,
                       per_image=None,
                       blocker=f"no unadapted published cell: {bwhy}")
            return row
        bm2, bfg, _ = two_class_tol_k2(bdf)
        # The SAR change and the published unadapted level come from two
        # artifacts with their own row orders, and the row below adds them per
        # image, so both sides are put in image_id order and the alignment is
        # asserted rather than assumed.
        base_sorted = bdf.sort_values("image_id").reset_index(drop=True)
        if list(base_sorted.image_id) != list(step0.image_id):
            fail(f"armF_sar_{arch}_{city}: the SAR pass and the published "
                 "unadapted cell do not carry the same 150 image_ids in "
                 "image_id order, so SAR's change cannot be added per image")
        bm2_sorted, _, _ = two_class_tol_k2(base_sorted)
        per_image = bm2_sorted + delta_per_image
        if base_is_n450 != needs_n450_sar:
            fail(f"sar/{arch}/{city}: the unadapted level is "
                 f"{'' if base_is_n450 else 'NOT '}a matched-450 cell but the "
                 f"SAR run selected is the "
                 f"{'n450' if needs_n450_sar else 'published'} one; the two "
                 "sides of a SAR row must sit on the same checkpoint family")
        row.update(run_id=os.path.basename(p),
                   tree=f"{root_label}/armF_sar_*",
                   source_kind=(("matched-450 unadapted cell" if base_is_n450
                                 else "published unadapted cell")
                                + f" ({base_tree}) plus the within-pass change "
                                + ("measured on the n450 checkpoints"
                                   if needs_n450_sar else
                                   "measured on the PUBLISHED checkpoints")
                                + "; caveat 4"
                                + ("" if needs_n450_sar else " and caveat 4b")),
                   sar_unadapted_is_n450=base_is_n450,
                   sar_change_measured_on=("n450 checkpoints" if needs_n450_sar
                                           else "published checkpoints"),
                   sar_stated_delta_key=stated_key,
                   sar_assumption=(
                       "MEASURED, not assumed, on the budget axis: the "
                       "unadapted level is the matched-450 Vanilla cell and the "
                       "within-pass change added to it was measured on the same "
                       "matched-450 checkpoint "
                       f"({os.path.relpath(p, BASE)}). results/26's caveat 4b, "
                       "which assumed SAR's effect is budget-invariant, is "
                       "DROPPED for this row. Caveat 4 still applies: the "
                       "change is measured at inference batch size 1 while the "
                       "unadapted level is a batch-4 scored pass."
                       if needs_n450_sar else
                       "no budget assumption is made on this row and none is "
                       "needed: this architecture's Vanilla cell was already at "
                       "450, so the published SAR run's checkpoint and the "
                       "unadapted level are the same training budget. Caveat 4 "
                       "applies as in results/22 and results/26: the change is "
                       "measured at inference batch size 1 against a batch-4 "
                       "unadapted value, and for OGLANet that regime difference "
                       "is not small."),
                   on_declared_path=False, fell_back=bool(base_fell_back),
                   unadapted_base_tree=base_tree,
                   n=int(per_image.size), value=float(per_image.mean()),
                   value_fg=None, per_image=per_image,
                   image_id=list(base_sorted.image_id),
                   unadapted_base_run_id=base_rid,
                   unadapted_base_value=float(bm2.mean()),
                   sar_own_unadapted_value=sar_own_unadapted(
                       d, f"armF_sar_{arch}_{city}")[0],
                   sar_own_unadapted_key=sar_own_unadapted(
                       d, f"armF_sar_{arch}_{city}")[1],
                   sar_own_unadapted_minus_declared_base=float(
                       sar_own_unadapted(d, f"armF_sar_{arch}_{city}")[0]
                       - bm2.mean()),
                   sar_n450_gate=((d.get("sanity_checks") or {})
                                  .get("n450_gate")),
                   sar_delta_per_image=delta_per_image,
                   sar_delta_cell_mean=float(delta_per_image.mean()),
                   sar_diverged=bool(d["sar"]["diverged"]),
                   sar_batch_size=d["provenance"].get("batch_size"),
                   sidecar_target_access=d.get("target_access"),
                   metric_module_sha=d["provenance"].get("metric_module_sha256"),
                   empty_fg_images=int(bdf.empty_fg_valid.sum()),
                   blocker=None)
        return row

    # --- everything else: PNG tree first, recompute as the fallback --------
    rid = declared_run_id(method, arch, city)
    trees = ([("st_r450_published_png", PNG),
              ("st_r450_published_recompute", RECOMPUTE)]
             if arch != CE else [("st", ST)])
    if method == "bsm" and arch != CE:
        trees = [("st", ST)]
    for i, (label, root) in enumerate(trees):
        df, meta, why = read_cell(root, rid)
        if df is None:
            continue
        m2, fg, _ = two_class_tol_k2(df)
        row.update(run_id=rid, tree=label,
                   source_kind=(meta.get("source")
                                or ("arm F / arm C-D training run"
                                    if label == "st" else None)),
                   on_declared_path=(label == "st_r450_published_png"),
                   fell_back=(label == "st_r450_published_recompute"),
                   n=int(len(df)), value=float(m2.mean()),
                   value_fg=float(fg.mean()), per_image=m2,
                   image_id=list(df.image_id),
                   png_table38_delta=meta.get("table38_delta"),
                   png_table38_gate_pass=meta.get("table38_gate_pass"),
                   png_final_comparison_delta=meta.get(
                       "final_comparison_delta"),
                   png_vs_recompute_delta=(
                       (meta.get("recompute_counterpart") or {}).get(
                           "delta_png_minus_recompute_miou_k2")),
                   sidecar_target_access=meta.get("target_access"),
                   metric_module_sha=meta.get("metric_module_sha"),
                   empty_fg_images=int(df.empty_fg_valid.sum()),
                   blocker=None)
        return row
    row.update(run_id=rid, tree=None, source_kind=None,
               on_declared_path=False, fell_back=False, value=None,
               per_image=None,
               blocker=("no dump in the declared PNG tree and none in the "
                        "recompute fallback"))
    return row



# ----------------------------------------------------------------------------
# the v3-vs-n450 side table
# ----------------------------------------------------------------------------

V3_PATH = os.path.join(BASE, "results", "22_internal_sib_worst_case_v3.json")


def v3_side_table(per_method, matched, margins, vanilla_worst, reading, adj,
                  per_cell, fell_back, excluded, index_sha256=None):
    """Every headline quantity of results/22, read from that JSON, beside this
    job's value for the same quantity.

    results/22 is opened read-only and is never written. Its numbers are read
    from the file rather than recomputed, so a difference here is a difference
    between two published artifacts and not between two runs of one script.
    """
    if not os.path.isfile(V3_PATH):
        fail(f"{os.path.relpath(V3_PATH, BASE)} is absent; the v3-vs-n450 side "
             "table is a required output of this job")
    v3 = json.load(open(V3_PATH))
    v3pm = {(e["method"], e["scope"]): e for e in v3["per_method"]}
    v3mt = {(e["method"], e["scope"]): e for e in v3["matched_cell_comparison"]}
    nowpm = {(e["method"], e["scope"]): e for e in per_method}
    nowmt = {(e["method"], e["scope"]): e for e in matched}

    def pair(a, b, key):
        va = (a or {}).get(key)
        vb = (b or {}).get(key)
        d = (None if (va is None or vb is None
                      or not isinstance(va, (int, float))
                      or not isinstance(vb, (int, float)))
             else float(vb - va))
        return dict(v3=va, n450=vb, change_n450_minus_v3=d)

    rows = []
    for scope in ("9_cell", "12_cell"):
        for m in METHOD_ORDER:
            if m == "upper":
                continue
            a, b = v3pm.get((m, scope)), nowpm.get((m, scope))
            am, bm = v3mt.get((m, scope)), nowmt.get((m, scope))
            rows.append(dict(
                scope=scope, method=m, method_label=METHOD_LABEL[m],
                as_available_worst_case_gap=pair(a, b, "worst_case_gap"),
                as_available_worst_cell=dict(v3=(a or {}).get("worst_cell"),
                                             n450=(b or {}).get("worst_cell")),
                as_available_mean_gap=pair(a, b, "mean_gap"),
                as_available_n_cells=dict(
                    v3=(a or {}).get("n_cells_available"),
                    n450=(b or {}).get("n_cells_available")),
                as_available_cells_missing=dict(
                    v3=(a or {}).get("cells_missing"),
                    n450=(b or {}).get("cells_missing")),
                matched_cell_worst_case_gap=pair(am, bm, "worst_case_gap"),
                matched_cell_worst_cell=dict(
                    v3=(am or {}).get("worst_cell"),
                    n450=(bm or {}).get("worst_cell")),
                matched_cell_mean_gap=pair(am, bm, "mean_gap"),
                matched_cell_n_cells=dict(v3=(am or {}).get("n_cells"),
                                          n450=(bm or {}).get("n_cells")),
                access_class=dict(v3=(a or {}).get("access_class"),
                                  n450=(b or {}).get("access_class")),
                n_cells_substituted_in_this_method=sum(
                    1 for r in per_cell
                    if r["method"] == m and r.get("n450_substituted")
                    and (r["arch"] != CE if scope == "9_cell" else True))))

    sib = dict()
    for scope in ("9_cell", "12_cell"):
        a, b = v3["sib_margin"].get(scope, {}), margins.get(scope, {})
        sib[scope] = dict(
            sib_mean_gap=pair(a, b, "sib_mean_gap"),
            sib_mean_gap_ci95=dict(v3=a.get("sib_mean_gap_ci95"),
                                   n450=b.get("sib_mean_gap_ci95")),
            sib_n_cells=dict(v3=a.get("sib_n_cells"), n450=b.get("sib_n_cells")),
            strongest_baseline=dict(v3=a.get("strongest_baseline"),
                                    n450=b.get("strongest_baseline")),
            strongest_baseline_mean_gap=pair(a, b,
                                             "strongest_baseline_mean_gap"),
            margin_mean_gap_difference=pair(a, b,
                                            "margin_mean_gap_difference"),
            paired_margin_mean=dict(
                v3=(a.get("paired_margin") or {}).get("mean"),
                n450=(b.get("paired_margin") or {}).get("mean")),
            paired_margin_ci95=dict(
                v3=(a.get("paired_margin") or {}).get("ci95"),
                n450=(b.get("paired_margin") or {}).get("ci95")),
            paired_margin_p=dict(
                v3=(a.get("paired_margin") or {}).get("p"),
                n450=(b.get("paired_margin") or {}).get("p")),
            paired_margin_cells=dict(v3=a.get("paired_margin_cells"),
                                     n450=b.get("paired_margin_cells")),
            vanilla_mean_gap=pair(a, b, "vanilla_mean_gap"),
            crosses_access_class=dict(v3=a.get("crosses_access_class"),
                                      n450=b.get("crosses_access_class")))

    van = {scope: dict(
        worst_case_gap=pair(v3["vanilla_worst_case"].get(scope, {}),
                            vanilla_worst.get(scope, {}), "worst_case_gap"),
        worst_cell=dict(
            v3=v3["vanilla_worst_case"].get(scope, {}).get("worst_cell"),
            n450=vanilla_worst.get(scope, {}).get("worst_cell")),
        mean_gap=pair(v3["vanilla_worst_case"].get(scope, {}),
                      vanilla_worst.get(scope, {}), "mean_gap"),
        n_cells=dict(
            v3=v3["vanilla_worst_case"].get(scope, {}).get("n_cells"),
            n450=vanilla_worst.get(scope, {}).get("n_cells")))
        for scope in ("9_cell", "12_cell")}

    v3r = v3["reading_fired"]
    read = dict(
        basis=dict(v3=v3r["basis"], n450=reading["basis"]),
        scope_9_cell=dict(v3=v3r["scope_9_cell"],
                          n450=reading["scope_9_cell"]),
        scope_12_cell=dict(v3=v3r["scope_12_cell"],
                           n450=reading["scope_12_cell"]),
        scope_9_cell_as_available=dict(
            v3=v3r["scope_9_cell_as_available"],
            n450=reading["scope_9_cell_as_available"]),
        scope_12_cell_as_available=dict(
            v3=v3r["scope_12_cell_as_available"],
            n450=reading["scope_12_cell_as_available"]),
        bases_agree=dict(v3=v3r["bases_agree"], n450=reading["bases_agree"]),
        boolean_evaluation=dict(v3=v3r["boolean_evaluation"],
                                n450=reading["boolean_evaluation"]),
        r4_also_fires=dict(v3=v3r["r4_also_fires"],
                           n450=reading["r4_also_fires"]),
        sib_worst_case_matched_9=dict(
            v3=v3r["adjudication"]["matched_cell"]["9_cell"]["sib_worst_case"],
            n450=adj["matched_cell"]["9_cell"]["sib_worst_case"]),
        sib_worst_cell_matched_9=dict(
            v3=v3r["adjudication"]["matched_cell"]["9_cell"]["sib_worst_cell"],
            n450=adj["matched_cell"]["9_cell"]["sib_worst_cell"]),
        methods_at_or_below_sib_matched_9=dict(
            v3=v3r["adjudication"]["matched_cell"]["9_cell"][
                "methods_with_worst_case_at_or_below_sib"],
            n450=adj["matched_cell"]["9_cell"][
                "methods_with_worst_case_at_or_below_sib"]),
        missing_cell_bound=dict(
            v3=v3r["missing_cell_bound"], n450=reading["missing_cell_bound"]),
        missing_cell_bound_changes_the_worst_case=dict(
            v3=v3r["missing_cell_bound_changes_the_worst_case"],
            n450=reading["missing_cell_bound_changes_the_worst_case"]))

    coverage = dict(
        rows_without_a_value=dict(
            v3=[f"{e['method']}/{e['arch']}/{e['city']}"
                for e in v3["rows_without_a_value"]],
            n450=[f"{e['method']}/{e['arch']}/{e['city']}"
                  for e in excluded]),
        rows_fell_back_from_png_tree=dict(
            v3=len(v3["rows_fell_back_from_png_tree"]),
            n450=len(fell_back)),
        n_cells_substituted=sum(1 for r in per_cell
                                if r.get("n450_substituted")),
        index_sha256=dict(
            v3=v3["index_sha256"], n450=index_sha256,
            same=(v3["index_sha256"] == index_sha256),
            note=("both are the declared PNG tree's index. They are equal when "
                  "that tree has not changed between the two jobs, which is "
                  "the case here: the matched-450 dumps live in the recompute "
                  "tree, whose index is carried separately in "
                  "index_sha256_all.")))

    return dict(
        what=("every headline quantity results/22_internal_sib_worst_case_v3"
              ".json reports, read from that file, beside this job's value for "
              "the same quantity. v3 is opened read-only and never written."),
        v3_path="results/22_internal_sib_worst_case_v3.json",
        v3_generated_utc=v3.get("generated_utc"),
        v3_headline=v3.get("headline"),
        per_method=rows,
        sib_margin=sib,
        vanilla_worst_case=van,
        reading=read,
        coverage=coverage)

# ----------------------------------------------------------------------------
# the results/26 side table
# ----------------------------------------------------------------------------

R26_PATH = os.path.join(BASE, "results", "26_internal_sib_worst_case_n450.json")


def r26_side_table(per_method, matched, margins, vanilla_worst, reading, adj,
                   per_cell):
    """results/26's headline quantities, read from that file, beside this job's.

    This job changes two inputs and nothing else, so the useful thing this table
    does is show WHICH rows moved. Any row other than a MAMNet SAR row that
    differs from results/26 is a defect in this script, and the table makes that
    checkable rather than asserted. results/26 is opened read-only and never
    written; its numbers are read from the file, not recomputed.
    """
    if not os.path.isfile(R26_PATH):
        fail(f"{os.path.relpath(R26_PATH, BASE)} is absent; the results/26 side "
             "table is a required output of this job")
    r26 = json.load(open(R26_PATH))
    a_pm = {(e["method"], e["scope"]): e for e in r26["per_method"]}
    a_mt = {(e["method"], e["scope"]): e for e in r26["matched_cell_comparison"]}
    b_pm = {(e["method"], e["scope"]): e for e in per_method}
    b_mt = {(e["method"], e["scope"]): e for e in matched}
    a_pc = {(e["method"], e["arch"], e["city"]): e for e in r26["per_cell"]}

    def pair(a, b, key):
        va, vb = (a or {}).get(key), (b or {}).get(key)
        d = (None if (va is None or vb is None
                      or not isinstance(va, (int, float))
                      or not isinstance(vb, (int, float)))
             else float(vb - va))
        return dict(r26=va, this=vb, change_this_minus_r26=d)

    rows = []
    for scope in ("9_cell", "12_cell"):
        for m in METHOD_ORDER:
            if m == "upper":
                continue
            rows.append(dict(
                scope=scope, method=m, method_label=METHOD_LABEL[m],
                as_available_worst_case_gap=pair(a_pm.get((m, scope)),
                                                 b_pm.get((m, scope)),
                                                 "worst_case_gap"),
                as_available_mean_gap=pair(a_pm.get((m, scope)),
                                           b_pm.get((m, scope)), "mean_gap"),
                matched_cell_worst_case_gap=pair(a_mt.get((m, scope)),
                                                 b_mt.get((m, scope)),
                                                 "worst_case_gap"),
                matched_cell_mean_gap=pair(a_mt.get((m, scope)),
                                           b_mt.get((m, scope)), "mean_gap")))

    # the per-cell diff, which is the real check
    cell_changes, cell_identical = [], 0
    for r in per_cell:
        a = a_pc.get((r["method"], r["arch"], r["city"]))
        if a is None:
            cell_changes.append(dict(
                method=r["method"], arch=r["arch"], city=r["city"],
                r26=None, this=r.get("gap_upper_minus_method"),
                change=None, note="cell absent from results/26"))
            continue
        va, vb = a.get("gap_upper_minus_method"), r.get("gap_upper_minus_method")
        if va is None and vb is None:
            cell_identical += 1
            continue
        if va is not None and vb is not None and va == vb:
            cell_identical += 1
            continue
        cell_changes.append(dict(
            method=r["method"], arch=r["arch"], city=r["city"],
            r26_gap_upper_minus_method=va, this_gap_upper_minus_method=vb,
            change=(None if (va is None or vb is None) else float(vb - va)),
            r26_run_id=a.get("run_id"), this_run_id=r.get("run_id"),
            r26_sar_change_measured_on=a.get("sar_change_measured_on"),
            this_sar_change_measured_on=r.get("sar_change_measured_on")))

    expected = ({("sar", "mamnet", c) for c in CITIES}
                | {("sib", "oglanet", c) for c in CITIES})   # results/31
    got = {(c["method"], c["arch"], c["city"]) for c in cell_changes}
    unexpected = sorted(got - expected)
    if unexpected:
        fail("rows other than the three MAMNet SAR rows differ from "
             "results/26: " + ", ".join("/".join(u) for u in unexpected)
             + ". This job is results/26 with two inputs upgraded; any other "
               "row moving is a defect in this script, not a finding.")

    sib = {}
    for scope in ("9_cell", "12_cell"):
        a = r26["sib_margin"].get(scope, {})
        b = margins.get(scope, {})
        sib[scope] = dict(
            sib_mean_gap=pair(a, b, "sib_mean_gap"),
            strongest_baseline=dict(r26=a.get("strongest_baseline"),
                                    this=b.get("strongest_baseline")),
            strongest_baseline_mean_gap=pair(a, b,
                                             "strongest_baseline_mean_gap"),
            margin_mean_gap_difference=pair(a, b,
                                            "margin_mean_gap_difference"),
            paired_margin_mean=dict(
                r26=(a.get("paired_margin") or {}).get("mean"),
                this=(b.get("paired_margin") or {}).get("mean")),
            paired_margin_p=dict(r26=(a.get("paired_margin") or {}).get("p"),
                                 this=(b.get("paired_margin") or {}).get("p")))

    r26r = r26["reading_fired"]
    read = dict(
        basis=dict(r26=r26r["basis"], this=reading["basis"]),
        scope_9_cell=dict(r26=r26r["scope_9_cell"],
                          this=reading["scope_9_cell"]),
        scope_12_cell=dict(r26=r26r["scope_12_cell"],
                           this=reading["scope_12_cell"]),
        boolean_evaluation=dict(r26=r26r["boolean_evaluation"],
                                this=reading["boolean_evaluation"]),
        sib_worst_case_matched_9=dict(
            r26=r26r["adjudication"]["matched_cell"]["9_cell"][
                "sib_worst_case"],
            this=adj["matched_cell"]["9_cell"]["sib_worst_case"]),
        methods_at_or_below_sib_matched_9=dict(
            r26=r26r["adjudication"]["matched_cell"]["9_cell"][
                "methods_with_worst_case_at_or_below_sib"],
            this=adj["matched_cell"]["9_cell"][
                "methods_with_worst_case_at_or_below_sib"]))

    oglanet_sib = [dict(
        city=r["city"],
        results_26_published_framing="Table 38 printed 2-dp bound"
            if r.get("published_comparator_was_a_bound_in_results_26")
            else "measured published dump value (PNG tree)",
        results_26_published_value=(
            (a_pc.get((r["method"], r["arch"], r["city"])) or {})
            .get("published_counterpart_value")),
        results_26_table38_printed=(
            (a_pc.get((r["method"], r["arch"], r["city"])) or {})
            .get("table38_printed")),
        this_published_comparator=r.get("published_comparator_value"),
        this_published_comparator_regime=r.get("published_comparator_regime"),
        this_is_a_bound=r.get("published_comparator_is_a_bound"),
        n450_scored_value=r.get("method_value"),
        delta_n450_scored_minus_published=r.get(
            "delta_n450_scored_minus_published_comparator"),
        delta_n450_trainer_minus_published=r.get(
            "delta_n450_trainer_minus_published_comparator"),
        fp32_rescore=r.get("published_fp32_rescore"))
        for r in per_cell
        if r["method"] == "sib" and r["arch"] == "oglanet"]

    return dict(
        what=("results/26's headline quantities, read from that file, beside "
              "this job's value for the same quantity, plus a per-cell diff "
              "that asserts no row other than the three MAMNet SAR rows moved. "
              "results/26 is opened read-only and never written."),
        r26_path="results/26_internal_sib_worst_case_n450.json",
        r26_generated_utc=r26.get("generated_utc"),
        r26_headline=r26.get("headline"),
        what_this_job_changed=[
            "the three MAMNet SAR rows: the within-pass change is now measured "
            "on the n450 checkpoints instead of the published ones (caveat 4b "
            "dropped for those rows)",
            "the published OGLANet-SIB comparator on miami and phoenix: "
            "relabelled from a Table 38 2-dp bound to the measured fp16 "
            "trainer-pass value, and the fp16-against-fp16 comparator rule is "
            "stated with chicago's fp32 re-score recorded beside it",
            "one new caveat on the OGLANet repeat-inference offset"],
        per_method=rows,
        sib_margin=sib,
        reading=read,
        per_cell_diff=dict(
            n_cells_identical=cell_identical,
            n_cells_changed=len(cell_changes),
            changed=cell_changes,
            rows_allowed_to_change=(["sar/mamnet/" + c for c in CITIES]
                                    + ["sib/oglanet/" + c for c in CITIES]),
            assertion="any row outside rows_allowed_to_change that differs "
                      "from results/26 is a hard failure of this script"),
        oglanet_sib_published_comparator=oglanet_sib,
        vanilla_worst_case={
            scope: dict(
                worst_case_gap=pair(r26["vanilla_worst_case"].get(scope, {}),
                                    vanilla_worst.get(scope, {}),
                                    "worst_case_gap"),
                mean_gap=pair(r26["vanilla_worst_case"].get(scope, {}),
                              vanilla_worst.get(scope, {}), "mean_gap"))
            for scope in ("9_cell", "12_cell")})



# results/31: the same table against results/27 (generated from r26_side_table)

R27_PATH = os.path.join(BASE, "results", "27_internal_sib_worst_case_n450_sar.json")

def r27_side_table(per_method, matched, margins, vanilla_worst, reading, adj,
                   per_cell):
    """results/27's headline quantities, read from that file, beside this job's.

    This job changes two inputs and nothing else, so the useful thing this table
    does is show WHICH rows moved. Any row other than a MAMNet SAR row that
    differs from results/27 is a defect in this script, and the table makes that
    checkable rather than asserted. results/27 is opened read-only and never
    written; its numbers are read from the file, not recomputed.
    """
    if not os.path.isfile(R27_PATH):
        fail(f"{os.path.relpath(R27_PATH, BASE)} is absent; the results/27 side "
             "table is a required output of this job")
    r27 = json.load(open(R27_PATH))
    a_pm = {(e["method"], e["scope"]): e for e in r27["per_method"]}
    a_mt = {(e["method"], e["scope"]): e for e in r27["matched_cell_comparison"]}
    b_pm = {(e["method"], e["scope"]): e for e in per_method}
    b_mt = {(e["method"], e["scope"]): e for e in matched}
    a_pc = {(e["method"], e["arch"], e["city"]): e for e in r27["per_cell"]}

    def pair(a, b, key):
        va, vb = (a or {}).get(key), (b or {}).get(key)
        d = (None if (va is None or vb is None
                      or not isinstance(va, (int, float))
                      or not isinstance(vb, (int, float)))
             else float(vb - va))
        return dict(r27=va, this=vb, change_this_minus_r27=d)

    rows = []
    for scope in ("9_cell", "12_cell"):
        for m in METHOD_ORDER:
            if m == "upper":
                continue
            rows.append(dict(
                scope=scope, method=m, method_label=METHOD_LABEL[m],
                as_available_worst_case_gap=pair(a_pm.get((m, scope)),
                                                 b_pm.get((m, scope)),
                                                 "worst_case_gap"),
                as_available_mean_gap=pair(a_pm.get((m, scope)),
                                           b_pm.get((m, scope)), "mean_gap"),
                matched_cell_worst_case_gap=pair(a_mt.get((m, scope)),
                                                 b_mt.get((m, scope)),
                                                 "worst_case_gap"),
                matched_cell_mean_gap=pair(a_mt.get((m, scope)),
                                           b_mt.get((m, scope)), "mean_gap")))

    # the per-cell diff, which is the real check
    cell_changes, cell_identical = [], 0
    for r in per_cell:
        a = a_pc.get((r["method"], r["arch"], r["city"]))
        if a is None:
            cell_changes.append(dict(
                method=r["method"], arch=r["arch"], city=r["city"],
                r27=None, this=r.get("gap_upper_minus_method"),
                change=None, note="cell absent from results/27"))
            continue
        va, vb = a.get("gap_upper_minus_method"), r.get("gap_upper_minus_method")
        if va is None and vb is None:
            cell_identical += 1
            continue
        if va is not None and vb is not None and va == vb:
            cell_identical += 1
            continue
        cell_changes.append(dict(
            method=r["method"], arch=r["arch"], city=r["city"],
            r27_gap_upper_minus_method=va, this_gap_upper_minus_method=vb,
            change=(None if (va is None or vb is None) else float(vb - va)),
            r27_run_id=a.get("run_id"), this_run_id=r.get("run_id"),
            r27_sar_change_measured_on=a.get("sar_change_measured_on"),
            this_sar_change_measured_on=r.get("sar_change_measured_on")))

    expected = {("sib", "oglanet", c) for c in CITIES}   # results/31
    got = {(c["method"], c["arch"], c["city"]) for c in cell_changes}
    unexpected = sorted(got - expected)
    if unexpected:
        fail("rows other than the three OGLANet-SIB rows differ from "
             "results/27: " + ", ".join("/".join(u) for u in unexpected)
             + ". This job is results/27 with the OGLANet-SIB cells swapped; any other "
               "row moving is a defect in this script, not a finding.")

    sib = {}
    for scope in ("9_cell", "12_cell"):
        a = r27["sib_margin"].get(scope, {})
        b = margins.get(scope, {})
        sib[scope] = dict(
            sib_mean_gap=pair(a, b, "sib_mean_gap"),
            strongest_baseline=dict(r27=a.get("strongest_baseline"),
                                    this=b.get("strongest_baseline")),
            strongest_baseline_mean_gap=pair(a, b,
                                             "strongest_baseline_mean_gap"),
            margin_mean_gap_difference=pair(a, b,
                                            "margin_mean_gap_difference"),
            paired_margin_mean=dict(
                r27=(a.get("paired_margin") or {}).get("mean"),
                this=(b.get("paired_margin") or {}).get("mean")),
            paired_margin_p=dict(r27=(a.get("paired_margin") or {}).get("p"),
                                 this=(b.get("paired_margin") or {}).get("p")))

    r27r = r27["reading_fired"]
    read = dict(
        basis=dict(r27=r27r["basis"], this=reading["basis"]),
        scope_9_cell=dict(r27=r27r["scope_9_cell"],
                          this=reading["scope_9_cell"]),
        scope_12_cell=dict(r27=r27r["scope_12_cell"],
                           this=reading["scope_12_cell"]),
        boolean_evaluation=dict(r27=r27r["boolean_evaluation"],
                                this=reading["boolean_evaluation"]),
        sib_worst_case_matched_9=dict(
            r27=r27r["adjudication"]["matched_cell"]["9_cell"][
                "sib_worst_case"],
            this=adj["matched_cell"]["9_cell"]["sib_worst_case"]),
        methods_at_or_below_sib_matched_9=dict(
            r27=r27r["adjudication"]["matched_cell"]["9_cell"][
                "methods_with_worst_case_at_or_below_sib"],
            this=adj["matched_cell"]["9_cell"][
                "methods_with_worst_case_at_or_below_sib"]))

    oglanet_sib = [dict(
        city=r["city"],
        results_27_published_framing="Table 38 printed 2-dp bound"
            if r.get("published_comparator_was_a_bound_in_results_27")
            else "measured published dump value (PNG tree)",
        results_27_published_value=(
            (a_pc.get((r["method"], r["arch"], r["city"])) or {})
            .get("published_counterpart_value")),
        results_27_table38_printed=(
            (a_pc.get((r["method"], r["arch"], r["city"])) or {})
            .get("table38_printed")),
        this_published_comparator=r.get("published_comparator_value"),
        this_published_comparator_regime=r.get("published_comparator_regime"),
        this_is_a_bound=r.get("published_comparator_is_a_bound"),
        n450_scored_value=r.get("method_value"),
        delta_n450_scored_minus_published=r.get(
            "delta_n450_scored_minus_published_comparator"),
        delta_n450_trainer_minus_published=r.get(
            "delta_n450_trainer_minus_published_comparator"),
        fp32_rescore=r.get("published_fp32_rescore"))
        for r in per_cell
        if r["method"] == "sib" and r["arch"] == "oglanet"]

    return dict(
        what=("results/27's headline quantities, read from that file, beside "
              "this job's value for the same quantity, plus a per-cell diff "
              "that asserts no row other than the three OGLANet-SIB rows moved. "
              "results/27 is opened read-only and never written."),
        r27_path="results/27_internal_sib_worst_case_n450.json",
        r27_generated_utc=r27.get("generated_utc"),
        r27_headline=r27.get("headline"),
        what_this_job_changed=[
            "the three OGLANet-SIB rows: taken from the held-out-city FDA pool "
            "retrains oglanet-sibc4tgt-loco-r450-{chicago,miami,phoenix}-hr-"
            "s42-n450 instead of the cross-source n450 cells "
            "oglanet-sibc4-loco-r450-{city}-hr-s42-n450",
            "OGLANet-SIB's access class: source_only -> unlabeled_target"],
        per_method=rows,
        sib_margin=sib,
        reading=read,
        per_cell_diff=dict(
            n_cells_identical=cell_identical,
            n_cells_changed=len(cell_changes),
            changed=cell_changes,
            rows_allowed_to_change=["sib/oglanet/" + c for c in CITIES],
            assertion="any row outside rows_allowed_to_change that differs "
                      "from results/27 is a hard failure of this script"),
        oglanet_sib_published_comparator=oglanet_sib,
        vanilla_worst_case={
            scope: dict(
                worst_case_gap=pair(r27["vanilla_worst_case"].get(scope, {}),
                                    vanilla_worst.get(scope, {}),
                                    "worst_case_gap"),
                mean_gap=pair(r27["vanilla_worst_case"].get(scope, {}),
                              vanilla_worst.get(scope, {}), "mean_gap"))
            for scope in ("9_cell", "12_cell")})


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

PNG_VANILLA_CELLS = [f"{a}-vanilla-loco-r450-{c}-hr-s0"
                     for a in ARCHS for c in CITIES]


def assert_png_vanilla_present():
    """v3's entry condition: the nine cross-city Vanilla cells must be in the
    declared PNG tree, each complete and each marked .DONE. Any absence stops
    the job; v3 exists only to move those nine rows, and the nine SAR rows
    built on them, off the probability-map fallback.
    """
    state, missing = [], []
    for rid in PNG_VANILLA_CELLS:
        d = os.path.join(PNG, rid)
        rec = dict(run_id=rid,
                   dir_present=os.path.isdir(d),
                   done_marker=os.path.isfile(os.path.join(d, ".DONE")),
                   failed_marker=os.path.isfile(os.path.join(d, ".FAILED")),
                   has_perimage=os.path.isfile(
                       os.path.join(d, "perimage.parquet")),
                   has_run_json=os.path.isfile(os.path.join(d, "run.json")))
        if rec["has_run_json"]:
            m = json.load(open(os.path.join(d, "run.json")))
            rec.update(
                final_comparison_delta=m.get("final_comparison_delta"),
                final_comparison_gate_pass=m.get("final_comparison_gate_pass"),
                png_miou_k2_twoclass=m.get("headline_miou_k2_twoclass"),
                delta_png_minus_recompute_miou_k2=(
                    (m.get("recompute_counterpart") or {}).get(
                        "delta_png_minus_recompute_miou_k2")))
        bad = [k for k in ("dir_present", "done_marker", "has_perimage",
                           "has_run_json") if not rec[k]]
        if bad or rec["failed_marker"]:
            missing.append(f"{rid}: " + ", ".join(bad + (
                [".FAILED marker present"] if rec["failed_marker"] else [])))
        state.append(rec)
    if missing:
        fail("the nine PNG-scored cross-city Vanilla cells are the entry "
             "condition for v3 and are not all present:\n  "
             + "\n  ".join(missing))
    return state


def main():
    hr = "=" * 100
    print(hr)
    print("Arm F / E-12 / E-2 -- internal SIB worst-case adjudication, n450 "
          "+ measured SAR")
    print(f"DRY_RUN = {DRY_RUN}   N450 = {N450}   SAR_N450 = {SAR_N450}   "
          f"B = {B}")
    print(hr)

    # ---- v3 entry condition ---------------------------------------------
    png_vanilla_state = assert_png_vanilla_present()
    print("\n--- v3 entry condition: the nine PNG-scored Vanilla cells ---")
    for r in png_vanilla_state:
        print(f"  {r['run_id']:42s} .DONE  final_comparison_delta "
              f"{r['final_comparison_delta']}  PNG minus recompute "
              f"{r['delta_png_minus_recompute_miou_k2']:+.4f}")

    # ---- indexes ---------------------------------------------------------
    trees = [("st_r450_published_png", PNG),
             ("st_r450_published_recompute", RECOMPUTE),
             ("st", ST),
             ("e12", E12)]
    indexes, built = {}, {}
    for label, root in trees:
        # Every prior version's index files are cited by sha256 in its own
        # results JSON, so none of them may be rewritten: results/12 cites
        # INDEX_{label}.parquet (a 76-cell PNG tree), results/22 cites
        # INDEX_{label}_v3.parquet (85 cells, and a recompute tree without the
        # 27 n450 dumps) and results/26 cites INDEX_{label}_26.parquet. This job
        # writes INDEX_{label}_27.parquet and touches nothing else.
        idx, info = build_index(
            label, root,
            write_to=os.path.join(INV, f"INDEX_{label}_31.parquet"))
        built[label] = idx
        indexes[label] = info
    on_disk = {label: on_disk_index_state(label, root)
               for label, root in trees}
    png_idx = built["st_r450_published_png"]
    png_info = indexes["st_r450_published_png"]
    print("\n--- indexes (CLAUDE.md: start from the index, never from a glob) ---")
    for lbl, i in indexes.items():
        o = on_disk[lbl]
        print(f"  {lbl:32s} built rows {i['n_rows']:4d}  sha256 "
              f"{(i.get('sha256') or 'not written in a dry run')[:16]}...  "
              f"on-disk index "
              f"{'absent' if not o['present'] else ('STALE' if o['stale'] else 'current')}")
    print(f"  PNG tree unindexable dirs: {png_info['n_unindexable']}")

    png_names = set(png_idx.run_id.astype(str))
    print(f"  PNG tree cells: {len(png_names)} "
          f"(hr {sum('-hr-' in n for n in png_names)}, "
          f"mr {sum('-mr-' in n for n in png_names)})")

    # ---- declared-path reproduction check on the PNG tree ----------------
    print("\n--- declared-path reproduction, as recorded by each PNG dump ---")
    t38d, fcd = [], []
    for n in sorted(png_names):
        meta = json.load(open(os.path.join(PNG, n, "run.json")))
        if meta.get("table38_delta") is not None:
            t38d.append((n, abs(meta["table38_delta"])))
        if meta.get("final_comparison_delta") is not None:
            fcd.append((n, abs(meta["final_comparison_delta"])))
    repro = dict(
        n_cells_with_table38=len(t38d),
        max_abs_delta_vs_table38=max(v for _, v in t38d),
        worst_cell_vs_table38=max(t38d, key=lambda x: x[1])[0],
        n_cells_with_final_comparison=len(fcd),
        max_abs_delta_vs_final_comparison=max(v for _, v in fcd),
        n_final_comparison_exact=sum(1 for _, v in fcd if v == 0.0))
    print(f"  Table 38: {repro['n_cells_with_table38']} cells, max |delta| "
          f"{repro['max_abs_delta_vs_table38']:.4f} "
          f"({repro['worst_cell_vs_table38']})")
    print(f"  final_comparison.json: {repro['n_cells_with_final_comparison']} "
          f"cells, max |delta| {repro['max_abs_delta_vs_final_comparison']:.6f}, "
          f"exact on {repro['n_final_comparison_exact']}")

    # ---- resolve every row -----------------------------------------------
    print("\n--- resolving rows ---")
    rows = {}
    for method in METHOD_ORDER:
        for arch in ARCHS:
            for city in CITIES:
                rows[(method, arch, city)] = resolve_row(method, arch, city)
    for method in ("upper", "vanilla", "fda", "sib", "bsm"):
        for city in CITIES:
            rows[(method, CE, city)] = resolve_row(method, CE, city)

    # image-id identity across trees, asserted not assumed
    by_city = {}
    for k, r in rows.items():
        if r.get("image_id"):
            by_city.setdefault(r["city"], []).append((k, r["image_id"]))
    # Every paired delta below is a positional subtraction of two per-image
    # vectors, so the ORDER has to match, not only the set. Both are checked.
    id_check = {}
    for city, lst in by_city.items():
        ref = lst[0][1]
        bad_set = [f"{m}/{a}" for (m, a, _), ids in lst
                   if sorted(ids) != sorted(ref)]
        if bad_set:
            fail(f"{city}: image_id set differs on {bad_set}; a paired "
                 "bootstrap across trees is not defined")
        bad_order = [f"{m}/{a}" for (m, a, _), ids in lst if ids != ref]
        if bad_order:
            fail(f"{city}: image_id ORDER differs on {bad_order}; the paired "
                 "deltas here are positional subtractions, so a reordered "
                 "cell would pair the wrong images")
        id_check[city] = dict(n_cells=len(lst), n_images=len(ref),
                              same_set=True, same_order=True,
                              reference_cell=f"{lst[0][0][0]}/{lst[0][0][1]}",
                              order_is_lexicographic=(ref == sorted(ref)))
    print("  image_id sets AND order identical within every city: "
          + ", ".join(f"{c} ({v['n_cells']} cells, n={v['n_images']})"
                      for c, v in sorted(id_check.items())))

    # ---- per-cell gaps ---------------------------------------------------
    per_cell, cell_pvals = [], {}
    for (method, arch, city), r in sorted(rows.items()):
        if method == "upper":
            continue
        u = rows[("upper", arch, city)]
        rec = dict(method=method, method_label=METHOD_LABEL[method], arch=arch,
                   city=city, access_class=r["access_class"],
                   scope=("9_cell" if arch != CE else "crossearth"),
                   run_id=r.get("run_id"), tree=r.get("tree"),
                   source_kind=r.get("source_kind"),
                   unadapted_base_tree=r.get("unadapted_base_tree"),
                   on_declared_path=r.get("on_declared_path"),
                   fell_back=r.get("fell_back"),
                   n450_substituted=r.get("n450_substituted"),
                   n450_candidates=r.get("n450_candidates"),
                   n450_refused=r.get("n450_refused"),
                   n450_inference_batch_size=r.get("n450_inference_batch_size"),
                   n450_batch_size_provenance=r.get(
                       "n450_batch_size_provenance"),
                   n450_n_train=r.get("n450_n_train"),
                   n450_n_val=r.get("n450_n_val"),
                   n450_epochs_run=r.get("n450_epochs_run"),
                   n450_epochs_max=r.get("n450_epochs_max"),
                   n450_stopped_on=r.get("n450_stopped_on"),
                   n450_self_check_delta=r.get("n450_self_check_delta"),
                   published_counterpart_run_id=r.get("published_run_id"),
                   published_counterpart_tree=r.get("published_tree"),
                   published_counterpart_value=r.get("published_value"),
                   published_counterpart_value_fg=r.get("published_value_fg"),
                   sar_unadapted_is_n450=r.get("sar_unadapted_is_n450"),
                   sar_change_measured_on=r.get("sar_change_measured_on"),
                   sar_assumption=r.get("sar_assumption"),
                   sar_run_dir=r.get("sar_run_dir"),
                   sar_run_root=r.get("sar_run_root"),
                   sar_run_kind=r.get("sar_run_kind"),
                   sar_n450_required=r.get("sar_n450_required"),
                   sar_n450_required_why=r.get("sar_n450_required_why"),
                   sar_n450_measured=r.get("sar_n450_measured"),
                   sar_n450_dir_present=r.get("sar_n450_dir_present"),
                   sar_stated_delta_key=r.get("sar_stated_delta_key"),
                   sar_own_unadapted_value=r.get("sar_own_unadapted_value"),
                   sar_own_unadapted_minus_declared_base=r.get(
                       "sar_own_unadapted_minus_declared_base"),
                   sar_delta_cell_mean=r.get("sar_delta_cell_mean"),
                   sar_diverged=r.get("sar_diverged"),
                   sar_batch_size=r.get("sar_batch_size"),
                   sar_n450_gate=r.get("sar_n450_gate"),
                   published_comparator_value=r.get(
                       "published_comparator_value"),
                   published_comparator_value_fg=r.get(
                       "published_comparator_value_fg"),
                   published_comparator_regime=r.get(
                       "published_comparator_regime"),
                   published_comparator_basis=r.get(
                       "published_comparator_basis"),
                   published_comparator_rule=r.get("published_comparator_rule"),
                   published_comparator_is_a_bound=r.get(
                       "published_comparator_is_a_bound"),
                   published_comparator_was_a_bound_in_results_26=r.get(
                       "published_comparator_was_a_bound_in_results_26"),
                   published_comparator_png_agreement=r.get(
                       "published_comparator_png_agreement"),
                   published_fp32_rescore=r.get("published_fp32_rescore"),
                   n450_trainer_pass=r.get("n450_trainer_pass"),
                   delta_n450_scored_minus_published_comparator=r.get(
                       "delta_n450_scored_minus_published_comparator"),
                   delta_n450_trainer_minus_published_comparator=r.get(
                       "delta_n450_trainer_minus_published_comparator"),
                   upper_run_id=u.get("run_id"), upper_tree=u.get("tree"),
                   method_value=r.get("value"), upper_value=u.get("value"),
                   table38_printed=r.get("table38_printed"),
                   sidecar_target_access=r.get("sidecar_target_access"),
                   sidecar_access_disagrees=(
                       r.get("sidecar_target_access") is not None
                       and r.get("sidecar_target_access") != r["access_class"]),
                   blocker=r.get("blocker"))
        if r.get("per_image") is None or u.get("per_image") is None:
            rec.update(gap=None, gap_upper_minus_method=None, test=None)
            per_cell.append(rec)
            continue
        d = np.asarray(r["per_image"]) - np.asarray(u["per_image"])
        bs = paired_bootstrap(d)
        rec.update(gap=float(d.mean()),
                   gap_upper_minus_method=float(-d.mean()),
                   n=int(d.size),
                   stderr=bs["stderr"], ci95_gap=bs["ci95"],
                   ci95_gap_upper_minus_method=[-bs["ci95"][1], -bs["ci95"][0]],
                   ci_method=bs["ci_method"], p=bs["p"],
                   test="paired image-level bootstrap on per-image two-class "
                        "K=2 deltas, method minus its within-city upper bound, "
                        "two-tailed",
                   paired_across_trees=(r.get("tree") != u.get("tree")),
                   d_vs_table38=(None if r.get("table38_printed") is None
                                 else float(r["value"] - r["table38_printed"])))
        # the matched-450 contrast on this cell, where both vectors exist
        if (r.get("n450_substituted")
                and r.get("published_per_image") is not None):
            dn = np.asarray(r["per_image"]) - np.asarray(r["published_per_image"])
            bn = paired_bootstrap(dn)
            u_pub = np.asarray(u["per_image"])
            rec.update(
                delta_n450_minus_published=float(dn.mean()),
                delta_n450_minus_published_ci95=bn["ci95"],
                delta_n450_minus_published_p=bn["p"],
                delta_n450_minus_published_test=(
                    "paired image-level bootstrap on per-image two-class K=2 "
                    "deltas, n450 minus published, two-tailed"),
                gap_upper_minus_method_published=float(
                    (u_pub - np.asarray(r["published_per_image"])).mean()),
                gap_change_n450_minus_published=float(
                    (np.asarray(r["published_per_image"])
                     - np.asarray(r["per_image"])).mean()))
        cell_pvals.setdefault(method, []).append((len(per_cell), bs["p"]))
        per_cell.append(rec)

    # Holm-Bonferroni within each method's cell family
    for method, lst in cell_pvals.items():
        adj = holm([p for _, p in lst])
        for (i, _), a in zip(lst, adj):
            per_cell[i]["p_holm_within_method_family"] = a
            per_cell[i]["holm_family"] = (
                f"{METHOD_LABEL[method]} cells vs their own upper bounds, "
                f"m = {len(lst)}")

    pc = pd.DataFrame(per_cell)
    pd.set_option("display.width", 240)
    print("\n--- per-cell gap, upper minus method, two-class K=2 (percent) ---")
    show = pc[pc.scope == "9_cell"][
        ["method", "arch", "city", "method_value", "upper_value",
         "gap_upper_minus_method", "p", "on_declared_path", "fell_back"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n--- CrossEarth cells (its own upper bound) ---")
    print(pc[pc.scope == "crossearth"][
        ["method", "city", "method_value", "upper_value",
         "gap_upper_minus_method", "p"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---- the missing-cell bound ------------------------------------------
    # OGLANet-SIB miami and phoenix have no dump in either tree. Their only
    # machine-readable evidence is Table 38's printed 2-dp value. Scoring the
    # declared tree reproduces Table 38 to <= 0.0072 on all 59 method cells, so
    # the printed value bounds those two cells to about +-0.01. The bound is
    # reported so the reading can be adjudicated; it is never a row value.
    bound = []
    for (method, arch, city), r in sorted(rows.items()):
        if method == "upper" or r.get("per_image") is not None:
            continue
        if r.get("table38_printed") is None:
            continue
        u = rows[("upper", arch, city)]
        if u.get("value") is None:
            continue
        bound.append(dict(
            method=method, arch=arch, city=city,
            table38_printed_method=r["table38_printed"],
            declared_path_upper=u["value"],
            bound_gap_upper_minus_method=float(u["value"]
                                               - r["table38_printed"]),
            basis="Table 38's printed 2-dp value for the method cell against "
                  "the declared-path PNG upper bound; the PNG tree reproduces "
                  "Table 38 to at most 0.0072 on 59 of 59 method cells, so the "
                  "bound is tight to about +-0.01",
            is_a_row_value=False))
    if bound:
        print("\n--- missing-cell bound from Table 38 (not a row value) ---")
        print(pd.DataFrame(bound)[
            ["method", "arch", "city", "table38_printed_method",
             "declared_path_upper", "bound_gap_upper_minus_method"]].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))

    # ---- per-method, per-scope summaries ---------------------------------
    def cells_of(method, scope):
        if scope == "9_cell":
            return [r for r in per_cell
                    if r["method"] == method and r["arch"] != CE]
        return [r for r in per_cell if r["method"] == method]

    def cluster_input(recs):
        out = {}
        for r in recs:
            rr = rows[(r["method"], r["arch"], r["city"])]
            u = rows[("upper", r["arch"], r["city"])]
            if rr.get("per_image") is None or u.get("per_image") is None:
                continue
            out.setdefault(r["city"], []).append(
                np.asarray(u["per_image"]) - np.asarray(rr["per_image"]))
        return out

    per_method = []
    for scope in ("9_cell", "12_cell"):
        n_expected = 9 if scope == "9_cell" else 12
        for method in METHOD_ORDER:
            if method == "upper":
                continue
            recs = cells_of(method, scope)
            have = [r for r in recs if r["gap_upper_minus_method"] is not None]
            miss = [f"{r['arch']}/{r['city']}" for r in recs
                    if r["gap_upper_minus_method"] is None]
            no_ce = (scope == "12_cell"
                     and not any(r["arch"] == CE for r in recs))
            entry = dict(
                method=method, method_label=METHOD_LABEL[method], scope=scope,
                access_class=sorted({r["access_class"] for r in recs}),
                n_cells_expected=(9 if no_ce else n_expected),
                n_cells_available=len(have),
                cells_missing=miss,
                has_crossearth_counterpart=not no_ce,
                archs=sorted({r["arch"] for r in have}))
            if not have:
                entry.update(worst_case_gap=None, worst_cell=None,
                             mean_gap=None, mean_gap_ci95=None,
                             mean_gap_stderr=None, mean_gap_p=None)
                per_method.append(entry)
                continue
            w = max(have, key=lambda r: r["gap_upper_minus_method"])
            cb = cluster_bootstrap_mean_over_cells(cluster_input(have))
            entry.update(
                worst_case_gap=w["gap_upper_minus_method"],
                worst_cell=f"{w['arch']}/{w['city']}",
                worst_cell_run_id=w["run_id"],
                worst_cell_ci95=w["ci95_gap_upper_minus_method"],
                worst_cell_p=w["p"],
                best_case_gap=min(r["gap_upper_minus_method"] for r in have),
                mean_gap=cb["mean"], mean_gap_stderr=cb["stderr"],
                mean_gap_ci95=cb["ci95"], mean_gap_ci_method=cb["ci_method"],
                mean_gap_p=cb["p"],
                per_cell_gaps={f"{r['arch']}/{r['city']}":
                               r["gap_upper_minus_method"] for r in have},
                n_cells_on_declared_path=sum(1 for r in have
                                             if r["on_declared_path"]),
                n_cells_fell_back=sum(1 for r in have if r["fell_back"]))
            per_method.append(entry)

    pm = pd.DataFrame(per_method)
    print("\n--- worst case and mean gap per method, per scope "
          "(upper minus method) ---")
    print(pm[["scope", "method", "n_cells_available", "n_cells_expected",
              "worst_case_gap", "worst_cell", "mean_gap", "mean_gap_p",
              "access_class"]].to_string(index=False,
                                         float_format=lambda v: f"{v:+.4f}"))

    # ---- matched-cell comparison -----------------------------------------
    # SIB is a 7-of-9 row, and the two absent cells are OGLANet/Miami and
    # OGLANet/Phoenix, the architecture with the largest gaps. Comparing a
    # 7-cell worst case against a 9-cell worst case favours SIB by construction,
    # so the primary comparison restricts every method to the cells where SIB
    # exists, and the as-available comparison is reported beside it.
    sib_cells_9 = {(r["arch"], r["city"]) for r in cells_of("sib", "9_cell")
                   if r["gap_upper_minus_method"] is not None}
    sib_cells_12 = {(r["arch"], r["city"]) for r in cells_of("sib", "12_cell")
                    if r["gap_upper_minus_method"] is not None}
    matched = []
    for scope, keep in (("9_cell", sib_cells_9), ("12_cell", sib_cells_12)):
        for method in METHOD_ORDER:
            if method == "upper":
                continue
            have = [r for r in cells_of(method, scope)
                    if (r["arch"], r["city"]) in keep
                    and r["gap_upper_minus_method"] is not None]
            if not have:
                continue
            w = max(have, key=lambda r: r["gap_upper_minus_method"])
            cb = cluster_bootstrap_mean_over_cells(cluster_input(have))
            matched.append(dict(
                scope=scope, method=method, method_label=METHOD_LABEL[method],
                n_cells=len(have), n_cells_in_matched_set=len(keep),
                cells=[f"{r['arch']}/{r['city']}" for r in have],
                worst_case_gap=w["gap_upper_minus_method"],
                worst_cell=f"{w['arch']}/{w['city']}",
                mean_gap=cb["mean"], mean_gap_ci95=cb["ci95"],
                mean_gap_stderr=cb["stderr"], mean_gap_p=cb["p"],
                access_class=sorted({r["access_class"] for r in have})))
    mt = pd.DataFrame(matched)
    print("\n--- matched-cell comparison: every method restricted to the "
          "cells where SIB exists ---")
    print(mt[["scope", "method", "n_cells", "worst_case_gap", "worst_cell",
              "mean_gap"]].to_string(index=False,
                                     float_format=lambda v: f"{v:+.4f}"))

    # ---- SIB's margin over the strongest baseline ------------------------
    margins = {}
    for scope in ("9_cell", "12_cell"):
        sib = next((e for e in per_method
                    if e["method"] == "sib" and e["scope"] == scope), None)
        cand = [e for e in per_method
                if e["scope"] == scope and e["method"] in BASELINES
                and e["mean_gap"] is not None]
        van = next((e for e in per_method
                    if e["method"] == "vanilla" and e["scope"] == scope), None)
        if sib is None or sib["mean_gap"] is None or not cand:
            margins[scope] = dict(strongest_baseline=None,
                                  why="no comparable baseline")
            continue
        strongest = min(cand, key=lambda e: e["mean_gap"])
        # per-image paired margin on the cells both rows cover
        keep = (set(strongest["per_cell_gaps"]) & set(sib["per_cell_gaps"]))
        per_city = {}
        for key in sorted(keep):
            arch, city = key.split("/")
            s = rows[("sib", arch, city)]
            b = rows[(strongest["method"], arch, city)]
            if s.get("per_image") is None or b.get("per_image") is None:
                continue
            per_city.setdefault(city, []).append(
                np.asarray(s["per_image"]) - np.asarray(b["per_image"]))
        cb = cluster_bootstrap_mean_over_cells(per_city) if per_city else None
        margins[scope] = dict(
            sib_mean_gap=sib["mean_gap"], sib_mean_gap_ci95=sib["mean_gap_ci95"],
            sib_n_cells=sib["n_cells_available"],
            strongest_baseline=strongest["method"],
            strongest_baseline_label=strongest["method_label"],
            strongest_baseline_mean_gap=strongest["mean_gap"],
            strongest_baseline_n_cells=strongest["n_cells_available"],
            strongest_baseline_access_class=strongest["access_class"],
            margin_mean_gap_difference=float(strongest["mean_gap"]
                                             - sib["mean_gap"]),
            margin_definition="strongest baseline's mean gap minus SIB's mean "
                              "gap; positive means SIB sits closer to the "
                              "within-city upper bound",
            paired_margin_cells=sorted(keep),
            paired_margin=cb,
            paired_margin_definition="per image, SIB minus the strongest "
                                     "baseline on the cells both rows cover, "
                                     "averaged over cells, image-cluster "
                                     "bootstrap by held-out city",
            vanilla_mean_gap=(van or {}).get("mean_gap"),
            baseline_mean_gaps={e["method"]: e["mean_gap"] for e in cand},
            crosses_access_class=(sorted({a for e in (sib, strongest)
                                          for a in e["access_class"]})
                                  if len({tuple(sib["access_class"]),
                                          tuple(strongest["access_class"])}) > 1
                                  else None))
    print("\n--- SIB mean gap and margin over the strongest baseline ---")
    for scope, m in margins.items():
        if not m.get("strongest_baseline"):
            print(f"  {scope}: {m['why']}")
            continue
        pm_ = m["paired_margin"]
        print(f"  {scope}: SIB mean gap {m['sib_mean_gap']:+.4f} "
              f"(n_cells {m['sib_n_cells']}); strongest baseline "
              f"{m['strongest_baseline_label']} {m['strongest_baseline_mean_gap']:+.4f}; "
              f"margin {m['margin_mean_gap_difference']:+.4f}")
        if pm_:
            print(f"      paired margin {pm_['mean']:+.4f} "
                  f"CI95 [{pm_['ci95'][0]:+.4f}, {pm_['ci95'][1]:+.4f}] "
                  f"p {pm_['p']:.4f} over {pm_['n_cells']} cells")

    # ---- within-access-class rankings -----------------------------------
    rankings = {}
    for scope in ("9_cell", "12_cell"):
        for cls in ("source_only", "unlabeled_target", "deployment_time"):
            ent = [e for e in per_method
                   if e["scope"] == scope and e["mean_gap"] is not None
                   and e["access_class"] == [cls]]
            if not ent:
                continue
            rankings[f"{scope}/{cls}"] = dict(
                access_class=cls, scope=scope, n_methods=len(ent),
                by_worst_case=[dict(method=e["method"],
                                    worst_case_gap=e["worst_case_gap"],
                                    worst_cell=e["worst_cell"],
                                    n_cells=e["n_cells_available"])
                               for e in sorted(ent,
                                               key=lambda x: x["worst_case_gap"])],
                by_mean_gap=[dict(method=e["method"], mean_gap=e["mean_gap"],
                                  ci95=e["mean_gap_ci95"],
                                  n_cells=e["n_cells_available"])
                             for e in sorted(ent, key=lambda x: x["mean_gap"])])
    # SIB's cells do not share one access class: MAMNet-SIB is
    # unlabeled_target through FDA while OGLANet-SIB and DINOv3-SIB are
    # source_only, so the whole-method ranking above cannot place SIB at all.
    # The second view splits every method by its CELLS' access class, which is
    # the only way a same-class comparison can include SIB.
    rankings_by_cell = {}
    for scope in ("9_cell", "12_cell"):
        for cls in ("source_only", "unlabeled_target", "deployment_time",
                    "target_labels"):
            ent = []
            for method in METHOD_ORDER:
                if method == "upper":
                    continue
                have = [r for r in cells_of(method, scope)
                        if r["access_class"] == cls
                        and r["gap_upper_minus_method"] is not None]
                if not have:
                    continue
                w = max(have, key=lambda r: r["gap_upper_minus_method"])
                cb = cluster_bootstrap_mean_over_cells(cluster_input(have))
                ent.append(dict(
                    method=method, method_label=METHOD_LABEL[method],
                    n_cells=len(have),
                    cells=[f"{r['arch']}/{r['city']}" for r in have],
                    worst_case_gap=w["gap_upper_minus_method"],
                    worst_cell=f"{w['arch']}/{w['city']}",
                    mean_gap=cb["mean"], mean_gap_ci95=cb["ci95"],
                    mean_gap_stderr=cb["stderr"], mean_gap_p=cb["p"]))
            if not ent:
                continue
            rankings_by_cell[f"{scope}/{cls}"] = dict(
                access_class=cls, scope=scope, n_methods=len(ent),
                by_worst_case=sorted(ent, key=lambda e: e["worst_case_gap"]),
                by_mean_gap=sorted(ent, key=lambda e: e["mean_gap"]),
                note="each method is restricted to the cells whose own access "
                     "class is this one, so a method whose cells span classes "
                     "(SIB) appears in more than one ranking with different "
                     "cells, and the n_cells column is not constant down a "
                     "ranking")
    print("\n--- within-access-class ranking, whole methods, by worst case "
          "(smallest first) ---")
    for k, v in sorted(rankings.items()):
        print(f"  {k}: " + " < ".join(
            f"{e['method']} {e['worst_case_gap']:+.3f}"
            for e in v["by_worst_case"]))
    print("\n--- within-access-class ranking, by CELL access class "
          "(places SIB) ---")
    for k, v in sorted(rankings_by_cell.items()):
        print(f"  {k}: " + " < ".join(
            f"{e['method']}[{e['n_cells']}] {e['worst_case_gap']:+.3f}"
            for e in v["by_worst_case"]))

    # ---- vanilla's worst case -------------------------------------------
    vanilla_worst = {}
    for scope in ("9_cell", "12_cell"):
        e = next((x for x in per_method
                  if x["method"] == "vanilla" and x["scope"] == scope), None)
        vanilla_worst[scope] = dict(
            worst_case_gap=e["worst_case_gap"], worst_cell=e["worst_cell"],
            worst_cell_ci95=e.get("worst_cell_ci95"),
            worst_cell_p=e.get("worst_cell_p"),
            mean_gap=e["mean_gap"], mean_gap_ci95=e["mean_gap_ci95"],
            n_cells=e["n_cells_available"],
            all_cells_fell_back=bool(e["n_cells_fell_back"]
                                     and not e["n_cells_on_declared_path"]))
    print("\n--- Vanilla worst case ---")
    for scope, v in vanilla_worst.items():
        print(f"  {scope}: {v['worst_case_gap']:+.4f} at {v['worst_cell']} "
              f"(n_cells {v['n_cells']})")

    # ---- the pre-committed readings --------------------------------------
    def adjudicate(basis_rows, scope):
        sib = next((e for e in basis_rows
                    if e["method"] == "sib" and e["scope"] == scope), None)
        others = [e for e in basis_rows
                  if e["scope"] == scope and e["method"] != "sib"
                  and e.get("worst_case_gap") is not None]
        if sib is None or sib.get("worst_case_gap") is None:
            return dict(sib_worst_case=None,
                        verdict="not adjudicable; SIB has no worst case")
        ties = [dict(method=e["method"], worst_case_gap=e["worst_case_gap"],
                     worst_cell=e["worst_cell"])
                for e in others
                if e["worst_case_gap"] <= sib["worst_case_gap"]]
        van = next((e for e in others if e["method"] == "vanilla"), None)
        return dict(
            sib_worst_case=sib["worst_case_gap"],
            sib_worst_cell=sib["worst_cell"],
            sib_n_cells=sib.get("n_cells", sib.get("n_cells_available")),
            methods_with_worst_case_at_or_below_sib=ties,
            sib_is_strictly_smallest=(len(ties) == 0),
            vanilla_worst_case=(van or {}).get("worst_case_gap"),
            vanilla_worst_cell=(van or {}).get("worst_cell"),
            sib_at_or_above_vanilla=(
                None if van is None
                else bool(sib["worst_case_gap"] >= van["worst_case_gap"])),
            ranking=[dict(method=e["method"],
                          worst_case_gap=e["worst_case_gap"],
                          worst_cell=e["worst_cell"])
                     for e in sorted(others + [sib],
                                     key=lambda x: x["worst_case_gap"])])

    adj = dict(
        matched_cell={s: adjudicate(matched, s)
                      for s in ("9_cell", "12_cell")},
        as_available={s: adjudicate(per_method, s)
                      for s in ("9_cell", "12_cell")})

    def fire(a):
        """R1-R4 from one basis. R1/R2 are joint over the two scopes; R3 is the
        9-cell scope; R4 is either scope."""
        s9, s12 = a["9_cell"], a["12_cell"]
        out = {}
        out["R1"] = bool(s9.get("sib_is_strictly_smallest")
                         and s12.get("sib_is_strictly_smallest"))
        out["R2"] = bool(s9.get("sib_is_strictly_smallest")
                         and not s12.get("sib_is_strictly_smallest"))
        out["R3"] = bool(s9.get("methods_with_worst_case_at_or_below_sib"))
        out["R4"] = bool(s9.get("sib_at_or_above_vanilla")
                         or s12.get("sib_at_or_above_vanilla"))
        return out

    fired = {b: fire(adj[b]) for b in adj}
    primary = "matched_cell"

    def one_reading(basis, scope):
        f = fired[basis]
        a = adj[basis][scope]
        if a.get("sib_worst_case") is None:
            return None
        if scope == "9_cell":
            if f["R3"]:
                return "R3"
            return "R1" if f["R1"] else ("R2" if f["R2"] else "R3")
        return "R1" if f["R1"] else "R2"

    reading = dict(
        basis=primary,
        basis_why=("SIB is a 7-of-9 row and the two absent cells are OGLANet/"
                   "Miami and OGLANet/Phoenix. Comparing a 7-cell worst case "
                   "against a 9-cell worst case favours SIB by construction, so "
                   "the reading rests on the matched-cell comparison, in which "
                   "every method is restricted to the cells where SIB exists. "
                   "The as-available comparison is reported beside it and the "
                   "two are checked for agreement."),
        scope_9_cell=one_reading(primary, "9_cell"),
        scope_12_cell=one_reading(primary, "12_cell"),
        scope_12_cell_label_is_a_fallback=(
            "READ THIS BEFORE QUOTING scope_12_cell. v3's reading function, "
            "kept unchanged here, scopes R3 to the 9-cell comparison and "
            "returns only R1 or R2 for the 12-cell scope, so a 12-cell label of "
            "R2 means \"not R1\" and NOT that R2's own condition fired. R2's "
            "condition is \"SIB smallest on 9, not on 12\", and "
            "boolean_evaluation records it as false on both bases. What "
            "actually holds on the 12-cell scope is the same thing that holds "
            "on the 9-cell scope: FDA's worst case is at or below SIB's, so "
            "R3's condition is satisfied on both scopes. The adjudication block "
            "carries the 12-cell numbers that show it."),
        scope_12_cell_note=("Only Upper, Vanilla, FDA, SIB and BSM have a "
                            "CrossEarth counterpart, so on the 12-cell scope "
                            "SIB's worst case is taken over 10 cells while "
                            "SegDeSiC, IIM, ISW, MRFP+, FADA and SAR are still "
                            "9-cell rows. Adding cells can only raise a worst "
                            "case, so SIB being smallest over a superset of "
                            "those methods' cells is the conservative "
                            "direction of the asymmetry."),
        scope_9_cell_as_available=one_reading("as_available", "9_cell"),
        scope_12_cell_as_available=one_reading("as_available", "12_cell"),
        bases_agree=(one_reading(primary, "9_cell")
                     == one_reading("as_available", "9_cell")
                     and one_reading(primary, "12_cell")
                     == one_reading("as_available", "12_cell")),
        r4_also_fires={b: fired[b]["R4"] for b in fired},
        boolean_evaluation=fired,
        adjudication=adj,
        missing_cell_bound=bound,
        missing_cell_bound_changes_the_worst_case=None)

    # does the Table 38 bound on the missing cells change SIB's worst case?
    sib_m = adj[primary]["9_cell"]
    sib_bound = [b for b in bound if b["method"] == "sib"]
    if sib_m.get("sib_worst_case") is not None and sib_bound:
        worst_bound = max(b["bound_gap_upper_minus_method"] for b in sib_bound)
        reading["missing_cell_bound_changes_the_worst_case"] = dict(
            sib_worst_case_observed=sib_m["sib_worst_case"],
            sib_worst_case_observed_cell=sib_m["sib_worst_cell"],
            largest_bounded_missing_cell_gap=worst_bound,
            missing_cells=[f"{b['arch']}/{b['city']}" for b in sib_bound],
            changes_it=bool(worst_bound > sib_m["sib_worst_case"]),
            statement=("The two absent OGLANet-SIB cells are bounded by Table "
                       "38's printed values at "
                       + ", ".join(f"{b['arch']}/{b['city']} "
                                   f"{b['bound_gap_upper_minus_method']:+.2f}"
                                   for b in sib_bound)
                       + f". The largest of those, {worst_bound:+.2f}, is "
                       + ("ABOVE" if worst_bound > sib_m["sib_worst_case"]
                          else "below")
                       + f" the observed SIB worst case of "
                         f"{sib_m['sib_worst_case']:+.2f} at "
                         f"{sib_m['sib_worst_cell']}, so filling those two "
                         "cells "
                       + ("WOULD" if worst_bound > sib_m["sib_worst_case"]
                          else "would not")
                       + " move SIB's worst case."))

    print("\n" + hr)
    print("PRE-COMMITTED READING")
    print(f"  basis               : {reading['basis']}")
    print(f"  9-cell scope        : {reading['scope_9_cell']}")
    print(f"  12-cell scope       : {reading['scope_12_cell']}")
    print(f"  as-available 9 / 12 : {reading['scope_9_cell_as_available']} / "
          f"{reading['scope_12_cell_as_available']}")
    print(f"  bases agree         : {reading['bases_agree']}")
    print(f"  R4 also fires       : {reading['r4_also_fires']}")
    for b in ("matched_cell", "as_available"):
        a = adj[b]["9_cell"]
        print(f"  [{b}] SIB worst case {a['sib_worst_case']:+.4f} at "
              f"{a['sib_worst_cell']}; methods at or below SIB: "
              + (", ".join(f"{t['method']} {t['worst_case_gap']:+.3f}"
                           for t in a["methods_with_worst_case_at_or_below_sib"])
                 or "none"))
    print(hr)

    # ---- CONTRACT section 6 conditions and tests -------------------------
    conditions, tests = [], []
    for e in per_method:
        if e["mean_gap"] is None:
            continue
        conditions.append(dict(
            label=f"{e['method_label']} / {e['scope']} / mean gap to the "
                  f"within-city upper bound",
            metric="gap_upper_minus_method_mIoU_tol_K2_pct",
            mean=e["mean_gap"], stderr=e["mean_gap_stderr"],
            n=e["n_cells_available"],
            n_images=sum(150 for _ in range(e["n_cells_available"])),
            ci95=e["mean_gap_ci95"], ci_method=e["mean_gap_ci_method"],
            baseline="within-city upper bound, same architecture and city",
            access_class=e["access_class"],
            worst_case_gap=e["worst_case_gap"], worst_cell=e["worst_cell"]))
        tests.append(dict(
            name="paired image-cluster bootstrap",
            compares=[f"{e['method']}/{e['scope']}", "upper"],
            statistic=e["mean_gap"], p=e["mean_gap_p"], B=B,
            two_tailed=True,
            clustering="images resampled within held-out city, shared across "
                       "the cells of that city"))
    for r in per_cell:
        if r.get("p") is None:
            continue
        tests.append(dict(
            name="paired image-level bootstrap",
            compares=[r["run_id"] if isinstance(r["run_id"], str)
                      else ",".join(r["run_id"]), r["upper_run_id"]],
            statistic=r["gap_upper_minus_method"], p=r["p"],
            p_holm=r.get("p_holm_within_method_family"),
            holm_family=r.get("holm_family"), B=B, two_tailed=True,
            correction="Holm-Bonferroni within the method's cell family"))
    for scope, m in margins.items():
        if m.get("paired_margin"):
            tests.append(dict(
                name="paired image-cluster bootstrap, SIB minus the strongest "
                     "baseline",
                compares=["sib", m["strongest_baseline"]],
                statistic=m["paired_margin"]["mean"],
                p=m["paired_margin"]["p"], B=B, two_tailed=True,
                scope=scope,
                note="no multiplicity correction: one pre-specified comparison "
                     "per scope"))
    for r in per_cell:
        if r["method"] != "sar":
            continue
        rr = rows[("sar", r["arch"], r["city"])]
        if rr.get("sar_delta_per_image") is None:
            continue
        bs = paired_bootstrap(rr["sar_delta_per_image"])
        tests.append(dict(
            name="paired image-level bootstrap on SAR's within-pass change",
            compares=[f"{rr['run_id']} final step",
                      f"{rr['run_id']} step 0"],
            statistic=bs["mean"], p=bs["p"], ci95=bs["ci95"], B=B,
            two_tailed=True,
            note="the only part of a SAR row that is measured on one forward "
                 "path; the row's level is the published unadapted cell plus "
                 "this change (caveat 4)"))

    # ---- n450 switch state ----------------------------------------------
    n450_block = dict(
        enabled=N450,
        rule=("the matched-450 retrain replaces the published cell wherever one "
              "exists. The retrain's run_id is the published cell's with its "
              "seed token replaced and a trailing -n450 suffix, optionally "
              "-n450-bs4 for a batch-4 re-score. A candidate must carry a "
              ".DONE marker and a perimage.parquet, and its RECORDED inference "
              "batch size must equal the batch size its own published "
              "comparator was produced at: 4 for the OGLANet classic families "
              "(run_inference.sh:73, run_inference_probs.sh:77), 8 for OGLANet "
              "SIB (train_oglanet_sib.py:1554 at args.batch_size 8 under fp16 "
              "autocast, which is the regime Table 38's OGLANet-SIB row itself "
              "came from). A -bs4 re-score is preferred over its parent. This "
              "REPLACES v3's rule, which tested the directory NAME for a bs4 "
              "suffix and would have refused all six OGLANet n450 cells, five "
              "of which are at the right batch size and one of which carries "
              "the -bs4 name."),
        rule_changed_from_v3=True,
        rule_change_why=("v3's switch was written before any matched-450 dump "
                         "existed. Its OGLANet refusal assumed a plain -n450 "
                         "dump would be inference-batch-1; the dumps that "
                         "landed record batch 4 and batch 8 in run.json, so "
                         "the name test refuses comparable cells and would "
                         "leave the OGLANet rows on the published values."),
        trees_searched=[t for t, r in N450_TREES if os.path.isdir(r)],
        trees_absent=[t for t, r in N450_TREES if not os.path.isdir(r)],
        n_cells_substituted=sum(1 for r in rows.values()
                                if r.get("n450_substituted")),
        n_cells_with_a_candidate=sum(1 for r in rows.values()
                                     if r.get("n450_candidates")),
        candidates={f"{m}/{a}/{c}": r["n450_candidates"]
                    for (m, a, c), r in sorted(rows.items())
                    if r.get("n450_candidates")},
        refused={f"{m}/{a}/{c}": r["n450_refused"]
                 for (m, a, c), r in sorted(rows.items())
                 if r.get("n450_refused")},
        retrain_status=("COMPLETE. All 27 matched-450 cells are on this box "
                        "under dumps/st_r450_published_recompute with a .DONE "
                        "marker and one scorer sha "
                        "e8a28bcba1f45ca3daaf240961451b62c8ae4b9d2ce78507c086f"
                        "1daae286bbd. Their record is "
                        "results/25_matched450_retrain_record.json."),
        required_inference_batch_size={f"{a}/{m}": v
                                       for (a, m), v in
                                       REQUIRED_INFER_BS.items()},
        required_inference_batch_size_default=REQUIRED_INFER_BS_DEFAULT,
        sar_n450_requested=SAR_N450,
        sar_n450_rule=("resolved PER ARCHITECTURE. An architecture whose "
                       "Vanilla cell IS substituted by a matched-450 retrain "
                       "requires an n450 SAR run and an absent one is a hard "
                       "stop; an architecture whose Vanilla cell is not "
                       "substituted takes its published SAR run, because its "
                       "checkpoint and its unadapted level are already the same "
                       "budget. Both SAR roots are searched: runs/ and "
                       "delta_runs/."),
        sar_runs_resolved={
            f"{a}/{c}": dict(
                n450_required=bool(resolve_n450("vanilla", a, c)[0]),
                n450_dir=(lambda t: None if t[1] is None
                          else os.path.relpath(t[1], BASE))(
                              find_sar_run(a, c, True)),
                published_dir=(lambda t: None if t[1] is None
                               else os.path.relpath(t[1], BASE))(
                                   find_sar_run(a, c, False)))
            for a in ARCHS for c in CITIES})
    print("\n--- --n450 switch ---")
    print(f"  enabled {N450}; candidates {n450_block['n_cells_with_a_candidate']}; "
          f"substituted {n450_block['n_cells_substituted']}; "
          f"trees absent {n450_block['trees_absent']}")

    # ---- fallbacks and exclusions ---------------------------------------
    fell_back = [dict(method=r["method"], arch=r["arch"], city=r["city"],
                      run_id=r["run_id"], tree=r["tree"],
                      unadapted_base_tree=r.get("unadapted_base_tree"),
                      reason="the declared PNG tree carries no dump for this "
                             "cell, so it is taken from the probability-map "
                             "fallback tree and recorded here")
                 for r in per_cell if r.get("fell_back")]
    excluded = [dict(method=r["method"], arch=r["arch"], city=r["city"],
                     run_id=r["run_id"], reason=r["blocker"])
                for r in per_cell if r.get("blocker")]
    print(f"\n  rows that fell back from the PNG tree: {len(fell_back)}")
    for f in fell_back:
        print(f"    {f['method']}/{f['arch']}/{f['city']} -> {f['tree']}")
    print(f"  rows with no value: {len(excluded)}")
    for e in excluded:
        print(f"    {e['method']}/{e['arch']}/{e['city']}: {e['reason']}")

    # ---- the v3-vs-n450 side table --------------------------------------
    side = v3_side_table(per_method, matched, margins, vanilla_worst, reading,
                         adj, per_cell, fell_back, excluded,
                         index_sha256=png_info.get("sha256"))
    side26 = r26_side_table(per_method, matched, margins, vanilla_worst,
                            reading, adj, per_cell)
    print("\n--- vs results/26: which rows moved ---")
    print(f"  cells identical to results/26: "
          f"{side26['per_cell_diff']['n_cells_identical']}; changed: "
          f"{side26['per_cell_diff']['n_cells_changed']}")
    for c in side26["per_cell_diff"]["changed"]:
        print(f"    {c['method']}/{c['arch']}/{c['city']} "
              f"{c['r26_gap_upper_minus_method']:+.6f} -> "
              f"{c['this_gap_upper_minus_method']:+.6f}  change "
              f"{c['change']:+.6f}  ({c['r26_sar_change_measured_on']} -> "
              f"{c['this_sar_change_measured_on']})")
    print("  OGLANet-SIB published comparator:")
    for e in side26["oglanet_sib_published_comparator"]:
        print(f"    {e['city']:8s} results/26 {e['results_26_published_framing']}"
              f"  -> fp16 trainer pass {e['this_published_comparator']:.7f}; "
              f"n450 scored {e['n450_scored_value']:.7f}; delta "
              f"{e['delta_n450_scored_minus_published']:+.4f}")
    print("\n--- v3 vs n450: worst case per method, matched-cell basis, "
          "9-cell scope ---")
    for r in side["per_method"]:
        if r["scope"] != "9_cell":
            continue
        a = r["matched_cell_worst_case_gap"]
        if a["v3"] is None and a["n450"] is None:
            continue
        print(f"  {r['method']:10s} v3 "
              + (f"{a['v3']:+.4f}" if a["v3"] is not None else "   none  ")
              + f" ({r['matched_cell_worst_cell']['v3']}, n="
              + f"{r['matched_cell_n_cells']['v3']})  ->  n450 "
              + (f"{a['n450']:+.4f}" if a["n450"] is not None else "   none  ")
              + f" ({r['matched_cell_worst_cell']['n450']}, n="
              + f"{r['matched_cell_n_cells']['n450']})  change "
              + (f"{a['change_n450_minus_v3']:+.4f}"
                 if a["change_n450_minus_v3"] is not None else "n/a"))
    print("\n--- v3 vs n450: SIB margin and the reading ---")
    for scope in ("9_cell", "12_cell"):
        sm = side["sib_margin"][scope]
        print(f"  {scope}: SIB mean gap v3 {sm['sib_mean_gap']['v3']} -> n450 "
              f"{sm['sib_mean_gap']['n450']}; strongest baseline "
              f"{sm['strongest_baseline']['v3']} -> "
              f"{sm['strongest_baseline']['n450']}; margin "
              f"{sm['margin_mean_gap_difference']['v3']} -> "
              f"{sm['margin_mean_gap_difference']['n450']}")
    rd = side["reading"]
    print(f"  reading 9-cell  v3 {rd['scope_9_cell']['v3']} -> n450 "
          f"{rd['scope_9_cell']['n450']}")
    print(f"  reading 12-cell v3 {rd['scope_12_cell']['v3']} -> n450 "
          f"{rd['scope_12_cell']['n450']}")

    # ---- sanity flags ----------------------------------------------------
    sanity = []
    exact_t38 = [r for r in per_cell
                 if r.get("d_vs_table38") is not None
                 and abs(r["d_vs_table38"]) < 1e-9]
    sanity.append(
        "Every OGLANet row is one draw of a nondeterministic inference "
        "pipeline whose cause is now named: DFFM._apply_downsample_n_times "
        "rebuilds 20 freshly random, unregistered, train-mode Conv2d+BatchNorm2d "
        "layers inside forward() on every call (dffm.py:33-37 factory, 78-82 "
        "call site; 0 of 20 carry the same weights across two forwards, 20 of "
        "20 normalise with batch statistics, 0 of their parameters appear in "
        "state_dict). Erratum E-9 names the phenomenon and attributes it to "
        "cudnn; the attribution is wrong, and "
        "torch.use_deterministic_algorithms(True) neither raises nor fixes it.")
    sanity.append(
        "The OGLANet PNG masks and the OGLANet probability maps are different "
        "predictions, not two scorings of one prediction: 20 of 20 OGLANet "
        "cells disagree, 0.76 to 3.05 percent of pixels, up to 2.03 two-class "
        "mIoU, with the 10-pixel small-component filter explaining 0.1 percent "
        "of it and the disagreeing pixels sitting far from p = 0.5 (median "
        "|p - 0.5| 0.074 to 0.088). MAMNet and DINOv3 disagree by at most "
        "0.0147. In v2 the Vanilla and SAR rows came from the probability-map "
        "tree while every other row came from the PNG tree, which put the three "
        "OGLANet Vanilla rows and the three OGLANet SAR rows on a different "
        "population from their own upper bounds. In v3 every Vanilla row, and "
        "every SAR row's unadapted level, is PNG-scored, so no row here is "
        "paired across the two prediction populations. The measured PNG minus "
        "probability-map difference on the nine Vanilla cells is "
        + ", ".join(f"{r['run_id'].split('-')[0]}/{r['run_id'].split('-')[4]} "
                    f"{r['delta_png_minus_recompute_miou_k2']:+.4f}"
                    for r in png_vanilla_state)
        + ", so the three OGLANet Vanilla rows move by that much between v2 "
          "and v3 and the six MAMNet and DINOv3 rows do not.")
    sanity.append(
        f"The declared tree's own reproduction record: Table 38 to "
        f"{repro['max_abs_delta_vs_table38']:.4f} on "
        f"{repro['n_cells_with_table38']} cells and final_comparison.json "
        f"exactly (0.000000) on {repro['n_final_comparison_exact']} of "
        f"{repro['n_cells_with_final_comparison']}. An exact match on "
        f"{repro['n_final_comparison_exact']} cells "
        "is expected rather than suspicious -- it is the same metric code over "
        "the same files -- but it means the PNG tree and final_comparison.json "
        "are not independent evidence about each other.")
    if exact_t38:
        sanity.append(
            f"{len(exact_t38)} row(s) land exactly on their printed Table 38 "
            "value to 1e-9: " + ", ".join(f"{r['method']}/{r['arch']}/{r['city']}"
                                          for r in exact_t38)
            + ". Flagged per CONVENTIONS.md; the printed values are 2-dp so an "
              "exact tie is arithmetic coincidence, not a recovered number.")
    else:
        sanity.append(
            "No row lands exactly on a printed Table 38 value; the printed "
            "values are 2-dp and every declared-path value carries more digits.")
    sib9 = next(e for e in matched if e["method"] == "sib"
                and e["scope"] == "9_cell")
    sanity.append(
        f"SIB's worst cell on the 9-cell scope is {sib9['worst_cell']} at "
        f"{sib9['worst_case_gap']:+.4f}. The published claim's 7.18 is OGLANet/"
        "Miami (Table 38: 73.89 - 66.71), a cell that has no dump on this box. "
        "The G1 flip recorded in summaries/HLT_COUNTERSIGNS_armA_e12.md "
        "(MAMNet/Phoenix ~7.48 under the clean pool, computed as 73.75 - "
        "66.2665) reproduces here on the declared path, which is a check of "
        "this job rather than a new result.")
    sanity.append(
        "TOO CLEAN, carried forward and now measured. The OGLANet Phoenix n450 "
        "pair lands 0.002051 foreground-IoU points apart -- sibc4 25.639580 "
        "against segdesic's superseded batch-4 pass 25.637529, two different "
        "methods at 100 against 37 epochs. Both cells enter numbers in this "
        "JSON, so the coincidence is load-bearing here in a way it was not in "
        "v3. It is NOT one prediction scored twice: "
        "results/25_matched450_retrain_record.json measures the pair's "
        "per-image foreground-IoU correlation at +0.8066, its mean absolute "
        "per-image difference at 9.1599 IoU points with a maximum of 50.8900, "
        "its thresholded masks agreeing on 0.942632 of pixels and its mean "
        "absolute probability difference at 0.067100, against 0.977037 and "
        "0.027490 for two batch-4 passes of the SAME SegDeSiC checkpoint. The "
        "same two methods also nearly tie on chicago (0.2179 apart, "
        "correlation +0.5697), so two of three cities show a cell-mean near-tie "
        "with substantially disagreeing per-image predictions. No cause is "
        "identified and no number here is adjusted for it.")
    above_upper = [r for r in per_cell
                   if r.get("gap_upper_minus_method") is not None
                   and r["gap_upper_minus_method"] < 0]
    if above_upper:
        sanity.append(
            "Cell(s) scoring ABOVE their own within-city upper bound: "
            + ", ".join(f"{r['method']}/{r['arch']}/{r['city']} "
                        f"{r['gap_upper_minus_method']:+.4f} (p {r['p']:.4f})"
                        for r in above_upper)
            + ". The published Table 38 already prints this pair the same way "
              "(SegDeSiC/DINOv3/Phoenix 68.04 against the DINOv3/Phoenix upper "
              "bound 67.82), so it is a property of the published grid rather "
              "than of this recomputation, and the paired bootstrap does not "
              "separate it from zero.")
    sar_v = [(r, next(x for x in per_cell if x["method"] == "vanilla"
                      and x["arch"] == r["arch"] and x["city"] == r["city"]))
             for r in per_cell if r["method"] == "sar"
             and r.get("gap_upper_minus_method") is not None]
    if sar_v:
        mx = max(abs(a["gap_upper_minus_method"] - b["gap_upper_minus_method"])
                 for a, b in sar_v)
        sanity.append(
            f"The SAR rows and the Vanilla rows differ by at most {mx:.4f} "
            "two-class mIoU. That is the size of SAR's measured adaptation "
            "effect on this metric, and it is far below the OGLANet "
            "nondeterminism band (E-9 cell-mean SD up to "
            f"{E9_MAX_SD5}, nine-pass span up to {E9_MAX_SPAN9}). SAR is "
            "therefore not separable from doing nothing on the three OGLANet "
            "cells, and its rows are reported as a separate access class "
            "rather than as a distinct level.")
    sanity.append(
        "The operator's brief states the 59 method cells reproduce printed "
        "Table 38 to a maximum |diff| of 0.0072. The maximum recorded in the "
        f"dumps themselves is {repro['max_abs_delta_vs_table38']:.4f} at "
        f"{repro['worst_cell_vs_table38']}, i.e. tighter than stated, and the "
        "Table 38 gate passes on 59 of 59. The premise checks out; the stated "
        "figure is not reproduced and the measured one is used.")
    cross_tree = [r for r in per_cell if r.get("paired_across_trees")]
    sanity.append(
        f"{len(cross_tree)} of {sum(1 for r in per_cell if r.get('p') is not None)} "
        "per-cell bootstraps pair a method cell against an upper-bound cell "
        "from a different dump tree. The pairing itself is sound -- the "
        "image_id sets are identical and in identical order within every city, "
        "asserted here, not assumed -- but a cross-tree delta carries both "
        "trees' provenance, which for OGLANet means two different predictions.")

    subbed = [r for r in per_cell if r.get("n450_substituted")]
    sanity.append(
        f"{len(subbed)} rows are matched-450 retrains and every within-city "
        "upper bound is unchanged, so the internal 9-cell grid in this JSON is "
        "budget-matched end to end: the 27 retrains are at n_train 450, "
        "MAMNet-SIB's E-12 clean retrains are at 450, and every other "
        "cross-city cell and every upper bound was already at 450. The "
        "budget-asymmetry caveat that v3 carried as its caveat 1 does not apply "
        "to these numbers. What replaces it is a single-seed caveat: each "
        "retrain is one draw of a non-reproducible training job (caveat n1).")
    moved = [r for r in subbed
             if r.get("delta_n450_minus_published") is not None]
    if moved:
        big = sorted(moved, key=lambda r: -abs(r["delta_n450_minus_published"]))
        sanity.append(
            "Per-cell movement from the published value to the matched-450 "
            f"retrain, over the {len(moved)} substituted rows with both "
            "per-image vectors: mean "
            f"{np.mean([r['delta_n450_minus_published'] for r in moved]):+.4f}, "
            f"range {min(r['delta_n450_minus_published'] for r in moved):+.4f} "
            f"to {max(r['delta_n450_minus_published'] for r in moved):+.4f} "
            "two-class mIoU, largest absolute moves "
            + ", ".join(f"{r['method']}/{r['arch']}/{r['city']} "
                        f"{r['delta_n450_minus_published']:+.2f}"
                        for r in big[:4])
            + ". Every one of these is a single-seed contrast and none of them "
              "separates a budget effect from training nondeterminism.")
    sib_now = [r for r in per_cell if r["method"] == "sib"
               and r.get("gap_upper_minus_method") is not None]
    sanity.append(
        f"SIB is a {len([r for r in sib_now if r['arch'] != CE])}-of-9 row here "
        "against 7 of 9 in v3, because the two OGLANet-SIB cells that had no "
        "dump anywhere on the published path (miami, phoenix) now have "
        "matched-450 retrains. The matched-cell comparison therefore no longer "
        "restricts the other methods to a 7-cell subset that excluded "
        "OGLANet's two largest-gap cities, which is the single structural "
        "change most likely to move a reading. The three OGLANet-SIB rows are "
        "the only rows in this JSON scored at inference batch size 8 rather "
        "than 4, matched to their own published comparator and not to the rest "
        "of the OGLANet grid.")
    sar_meas = [r for r in per_cell if r["method"] == "sar"
                and r.get("sar_n450_measured")]
    sar_pub = [r for r in per_cell if r["method"] == "sar"
               and r.get("sar_n450_measured") is False]
    sanity.append(
        f"{len(sar_meas)} of {len(sar_meas) + len(sar_pub)} SAR rows now take "
        "their within-pass change from a run on the SAME checkpoint their "
        "unadapted level comes from: "
        + ", ".join(f"{r['arch']}/{r['city']} ({r['sar_run_dir']})"
                    for r in sar_meas)
        + ". The remaining "
        + ", ".join(f"{r['arch']}/{r['city']}" for r in sar_pub)
        + " take the published run, which is correct for them because their "
          "Vanilla cells were already at 450 and no n450 SAR run exists or is "
          "needed. results/26's caveat 4b is therefore dropped for the first "
          "group and kept for the second, and caveat 4 -- batch 1 against "
          "batch 4 -- is untouched on all nine. The SAR adaptation effect "
          "itself is near zero on every row and on both checkpoint families "
          "(largest |change| 0.0237 two-class on the published side, 0.0094 on "
          "the n450 side), and that is expected by construction: the inherited "
          "selected_config carries pilot_verdict no_meaningful_adaptation and "
          "was not re-selected on the n450 models.")
    gates = [(r["arch"], r["city"], (r.get("sar_n450_gate") or {}))
             for r in sar_meas]
    if gates:
        sanity.append(
            "The n450 SAR runs' own hard gate -- the harness's unadapted pass "
            "against the n450 cell's stored two-class mIoU, tolerance 0.01, "
            "HARD STOP on failure -- passed on all three: "
            + ", ".join(f"{a}/{c} {g.get('difference'):+.6f}"
                        for a, c, g in gates)
            + ". Those differences are also this job's cross-check that the "
              "SAR harness and n450_score.py agree on the unadapted level, "
              "measured by two independent code paths, and the largest of them "
              "is 0.005 two-class mIoU.")
    # What the reading turns on, stated rather than left in a table. R3 fires
    # when some method's worst case is at or below SIB's, so the decisive
    # quantity is the slack between SIB's worst cell and the next method's, and
    # whether that slack is larger than the known instability on those cells.
    m9 = adj["matched_cell"]["9_cell"]
    ties = m9.get("methods_with_worst_case_at_or_below_sib") or []
    if m9.get("sib_worst_case") is not None:
        sibrow = next((r for r in per_cell
                       if r["method"] == "sib"
                       and f"{r['arch']}/{r['city']}" == m9["sib_worst_cell"]),
                      None)
        parts = [
            f"The 9-cell reading turns on one comparison: SIB's worst case "
            f"{m9['sib_worst_case']:+.4f} at {m9['sib_worst_cell']} against "
            + (", ".join(f"{t['method']} {t['worst_case_gap']:+.4f} at "
                         f"{t['worst_cell']}" for t in ties)
               if ties else "no method at or below it")
            + "."]
        if ties:
            slack = min(m9["sib_worst_case"] - t["worst_case_gap"]
                        for t in ties)
            parts.append(
                f"The slack is {slack:.4f} two-class mIoU. For scale on the "
                "same metric: erratum E-9's measured OGLANet nine-pass span is "
                f"{E9_MAX_SPAN9} and its cell-mean SD reaches {E9_MAX_SD5}, and "
                "the matched-450 retrains are one seed each against one "
                "published draw.")
        if sibrow is not None and sibrow.get("n450_substituted"):
            parts.append(
                f"SIB's worst cell {m9['sib_worst_cell']} IS one of the "
                "substituted matched-450 rows: its gap moved from "
                + _fmt_or_none(sibrow.get('gap_upper_minus_method_published'))
                + " published to "
                f"{sibrow.get('gap_upper_minus_method'):+.4f} at n450, a change "
                "of " + _fmt_or_none(sibrow.get('gap_change_n450_minus_published'))
                + ". So "
                "the reading rests on a single OGLANet cell that carries the "
                "single-seed caveat (n1), the DFFM one-draw caveat (3, n3), the "
                "batch-8 fp16 scoring regime, and the probability-map versus "
                "PNG population mixing all at once. That is stated, not "
                "discounted: the measurement is what it is, and it is one cell.")
        sanity.append(" ".join(parts))

        s12 = adj["matched_cell"]["12_cell"]
    if (s12.get("methods_with_worst_case_at_or_below_sib")
            and reading["scope_12_cell"] == "R2"):
        sanity.append(
            "Label trap, flagged so it is not quoted wrongly. The 12-cell scope "
            "reads R2, but that is v3's reading function returning \"not R1\": "
            "it scopes R3 to the 9-cell comparison and can only return R1 or R2 "
            "for 12 cells. R2's own condition, SIB smallest on 9 and not on 12, "
            "is FALSE here (boolean_evaluation.R2 is false on both bases). On "
            "the 12-cell scope "
            + ", ".join(f"{t['method']} {t['worst_case_gap']:+.4f}"
                        for t in s12["methods_with_worst_case_at_or_below_sib"])
            + f" also sits at or below SIB's {s12['sib_worst_case']:+.4f}, so "
              "R3's condition holds on both scopes and R1 holds on neither. "
              "The reading function was left unchanged rather than edited, "
              "because changing how a pre-commitment is evaluated after seeing "
              "the numbers is exactly what the pre-commitment exists to "
              "prevent; the correct reading is stated here instead.")

    mixed = sorted({(r["arch"], r["method"]) for r in per_cell
                    if r.get("n450_substituted")})
    sanity.append(
        "The grid in this JSON is not uniformly a retrain: "
        + ", ".join(f"{a}/{m}" for a, m in mixed)
        + " are matched-450 retrains and every other row is as published. That "
          "is the correct construction, because only those cells trained on 900 "
          "(results/21 finding 1), but it means a method's row can be a retrain "
          "on one architecture and a published cell on another, and a "
          "cross-architecture comparison of one method mixes the two.")

    # ---- caveats ---------------------------------------------------------
    caveats = [
        # 1 -- v3's caveat 1, updated for the 450-image budget
        "Training budgets as measured (delta_runs/train_count_audit and the "
        "2026-10-25/26 Delta audits behind it): MAMNet standard and its six "
        "adaptation methods (FDA, SegDeSiC, IIM, ISW, MRFP+, FADA) trained on "
        "900 images, 450 per source city; OGLANet SIB and OGLANet SegDeSiC on "
        "900; every other cross-city cell on 450; every within-city upper bound "
        "on 450. In THIS JSON all 27 of those cells are replaced by matched-450 "
        "retrains at n_train 450 (225 per source city) and the upper bounds are "
        "unchanged, so no row here is compared against an upper bound at half "
        "its own training budget. What the retrain did NOT change: every "
        "cross-city cell, retrains included, validated on 300 images, not the "
        "150 that section 3.3 and CONTRACT.md state, and the retrains could not "
        "restore the published cells' image draw because MAMNet's LOCO standard "
        "trained on all 900 source images and so never drew a 450-image subset "
        "(results/21_disclosure_record.json finding 1). All 27 retrains share "
        "one new 450-image list, recorded as train_list_sha256, which is a new "
        "decision rather than a restoration. The per-cell record of the "
        "substitution is results/25_matched450_retrain_record.json.",
        # 2
        "BSM and internal CrossEarth used a different 450-image draw from the "
        "published standard cells, with 150 validation images rather than 300. "
        "Measured on Delta: BSM draws its own 450 (225 per source city) through "
        "its wrapper; CrossEarth's 450 comes from the Final_data_loco split "
        "directory and overlaps the shared list on 222, 234 and 225 images of "
        "450. No BSM or CrossEarth cell is above 450.",
        # 3
        "OGLANet inference is nondeterministic and inference-batch-size "
        "dependent because of a defect in our own DFFM reimplementation: "
        "python/oglanet/models/dffm.py builds 20 freshly random-initialised, "
        "unregistered Conv2d+BatchNorm2d layers inside forward() on every call, "
        "so model.eval() never reaches them and they normalise with batch "
        "statistics (delta_runs/dffm_defect/, delta_runs/oglanet_nondeterminism/"
        "dffm_confirm.json). Measured spread on one OGLANet cell: foreground "
        "IoU at K=2 of 33.05 at batch 1, 37.33 at batch 2, 39.62 at batch 4, "
        "40.55 at batch 8. The published OGLANet tables, and therefore every "
        "OGLANet row on the declared path here, were produced at batch size 4. "
        "Every OGLANet value is one draw.",
        # 4
        "Every SAR row is an unadapted cell's level plus the change SAR "
        "measured inside its own pass, and that change was measured at "
        "inference batch size 1 while the unadapted value is a batch-4 scored "
        "pass. That regime mismatch is unchanged by this job and applies to all "
        "nine SAR rows. For OGLANet it is not small: "
        "arm F's unadapted pass sits 3.02, 8.14 and 0.21 foreground-IoU points "
        "below the same cells' PNG-scored published values on chicago, miami "
        "and phoenix, against 0.0009 for MAMNet and 0.0057 for DINOv3. SAR is "
        "episodic and per-image, so its baseline is the batch-1 regime by "
        "construction and no harness change makes the OGLANet comparison sound. "
        "The within-pass change itself is measured on one forward path and is "
        "the only part of a SAR row that carries a paired test.",
        # 4b -- DROPPED for the MAMNet rows in this job, kept for the others
        "Caveat 4b of results/26 is DROPPED for the three MAMNet SAR rows and "
        "KEPT for the six OGLANet and DINOv3 rows. results/26 added SAR's "
        "within-pass change as measured on the PUBLISHED 900-image MAMNet "
        "checkpoint to the matched-450 unadapted level, which assumed SAR's "
        "effect does not depend on the training budget. The SAR rerun on the "
        "n450 MAMNet checkpoints has landed -- "
        "delta_runs/armF_sar_mamnet_{chicago,miami,phoenix}_highres-n450, three "
        "of three cells COMPLETED with .DONE on disk, and the hard n450 gate "
        "(the harness's unadapted pass against the n450 cell's own stored "
        "two-class mIoU, tolerance 0.01, HARD STOP on failure) passed on all "
        "three at -0.00253 / +0.00260 / +0.00502 -- so those three rows now "
        "take their change from the same checkpoint family their unadapted "
        "level comes from and the assumption is a measurement. The assumption "
        "was never made on the OGLANet and DINOv3 rows and is not made now: "
        "their Vanilla cells were already at 450, so there is no budget "
        "mismatch for an n450 SAR run to remove and none exists. What the "
        "measurement shows about the dropped assumption: SAR's change on the "
        "n450 MAMNet checkpoints is +0.000142 (chicago), -0.003814 (miami) and "
        "-0.009416 (phoenix) two-class mIoU, against -0.004356, +0.002867 and "
        "-0.023722 on the published checkpoints. Both sets are within 0.024 of "
        "zero, so the assumption results/26 made was in fact close to harmless, "
        "which is a statement we can now make rather than hope for. The "
        "selected (lr, steps) config was NOT re-selected on the n450 models "
        "(the arm F brief changed only the checkpoint), and it carries "
        "pilot_verdict no_meaningful_adaptation, chosen on the published "
        "checkpoints' source-city validation. So the near-zero SAR effect on "
        "these rows is expected by construction and is not evidence that SAR "
        "cannot adapt these models; a SAR result that is not predetermined "
        "needs the (lr, steps) ladder re-selected on the n450 models, which is "
        "a new arm and is not done here.",
        # 5
        "Six published cells were retrained after their March 2026 runs "
        "collapsed (delta_runs/retrained_cells_record/): OGLANet MRFP+ chicago "
        "hr and mr, OGLANet MRFP+ miami hr, DINOv3 MRFP+ chicago mr and miami "
        "hr, and DINOv3 vanilla miami hr. Table 38 uses the later runs, "
        "confirmed from the dates of the prediction trees it scores. Two of "
        "those cells' published numbers moved by +26.68 and +39.36 mIoU between "
        "the March and April artifacts. One cell, OGLANet MRFP+ miami, changed "
        "its learning rate from 1e-4 to 3e-5; the other five were retrained at "
        "the same learning rate and simply did not collapse again. Who made "
        "that change is unrecorded: no ledger line, no commit and no DECISIONS "
        "entry covers 2026-04, and the only actor evidence is the SLURM job "
        "owner.",
        # 6
        "Seeds: every published cell on this grid is a single draw. MAMNet-SIB "
        "is the exception and carries three clean-pool seeds {42, 43, 44}; its "
        "row is the seed mean and the per-seed values are in the side table. "
        "The published MAMNet-SIB cell is contaminated (E-12) and is never used "
        "as a value, only reported beside the retrains.",
        # the rest are this job's own limitations
        "SIB is a 9-of-9 row here, against 7 of 9 in v3. The two cells v3 "
        "could not fill, OGLANet-SIB miami and OGLANet-SIB phoenix, still have "
        "no PUBLISHED dump in the PNG tree or the recompute fallback (the "
        "2026-10-25 56-cell re-score deliberately held them pending the OGLANet "
        "SIB AMP question), so no per-image published vector exists for either "
        "and neither can enter a paired test on the published side. What this "
        "job changes against results/26 is the LABEL on their published "
        "cell-mean values: results/26 carried them as Table 38 2-dp bounds, and "
        "they are now the measured fp16 trainer-pass values 66.7057610118262 "
        "(miami) and 63.9668489688177 (phoenix), read from each published "
        "cell's own test_results.json field tolerant_2px, which is the "
        "published Table 38 source. The bound framing is withdrawn. The "
        "comparator rule is fp16 against fp16 on all three cities, which the "
        "n450 rows satisfy because they were scored at batch 8 under fp16 "
        "autocast by the SIB trainer's own eval path; chicago's fp32 re-score "
        "of its stored probability maps, 71.79402162754555, is recorded beside "
        "its fp16 value 73.0475762379333 and is not used as the comparator. "
        "What fills the rows is still their matched-450 "
        "retrains, which exist. The consequence is structural, not a rounding "
        "matter: v3's matched-cell comparison restricted every method to the 7 "
        "cells where SIB existed, which excluded OGLANet miami and phoenix, two "
        "of the largest-gap cells on the grid; this job's matched-cell "
        "comparison is over all 9. A worst case over 9 cells can only be larger "
        "than over a 7-cell subset of them, so part of any increase in SIB's "
        "worst case here is coverage and not budget. The two effects are not "
        "separated, because no published OGLANet-SIB dump exists for those "
        "cities to separate them with.",
        "The three OGLANet SIB cells' published Table 38 rows are not "
        "reproducible from the stored probability maps at any scoring setting: "
        "Table 38's SIB row was written under fp16 autocast by "
        "train_oglanet_sib.py while the stored maps were written in fp32 by "
        "sp_gap_analysis.py. The PNG tree sidesteps this -- the one available "
        "OGLANet SIB cell reproduces Table 38 from its PNG masks -- but it is "
        "why the other two cells were held. The three n450 OGLANet SIB rows "
        "inherit the same regime: they were scored at inference batch size 8 "
        "under fp16 autocast by the SIB trainer's own test loader, which is what "
        "makes them comparable to Table 38's SIB row and NOT comparable to the "
        "batch-4 fp32 regime of every other published OGLANet table. No "
        "batch-4 OGLANet-SIB pass exists or can be produced with the released "
        "code: run_inference_probs.py has no SIB model_variant and prob_variant "
        "is None for every SIB cell, which is why the batch-4 attempt (job "
        "23273276) failed with a TypeError rather than producing a number.",
        "RESOLVED in v3. v2's tree-mixing caveat read: Vanilla rows and SAR "
        "rows come from the recompute fallback tree (stored probability maps), "
        "every other row from the declared PNG tree; the PNG tree carries no "
        "vanilla cell at all; for OGLANet the two trees are different "
        "predictions and OGLANet phoenix vanilla had no measured PNG score "
        "anywhere on this box. It is resolved by the nine PNG-scored "
        "cross-city Vanilla dumps at "
        "dumps/st_r450_published_png/{mamnet,oglanet,dinov3}-vanilla-loco-r450-"
        "{chicago,miami,phoenix}-hr-s0, written 2026-10-28 by the same scoring "
        "pass that produced the rest of the tree, each carrying .DONE and each "
        "reproducing final_comparison.json to 0.000000. Every Vanilla row here, "
        "and every SAR row's unadapted level, is now PNG-scored, so no row is "
        "paired against an upper bound from a different prediction population "
        "and no row in this JSON falls back to the probability-map tree. The "
        "nine cells move by " + ", ".join(
            f"{r['run_id'].split('-')[0]}/{r['run_id'].split('-')[4]} "
            f"{r['delta_png_minus_recompute_miou_k2']:+.4f}"
            for r in png_vanilla_state)
        + " two-class mIoU against their v2 probability-map values. The SAR "
          "rows remain an ASSUMPTION for the separate reason in caveat 4: "
          "their within-pass change is measured at inference batch size 1 "
          "while the PNG-scored unadapted level is batch 4.",
        "BSM is an 8-of-9 row on the three original architectures and 11 of 12 "
        "with CrossEarth. DINOv3/chicago collapsed in training at best "
        "validation mIoU 5.51 against the codebase floor of 10.0 and its dump "
        "directory is empty. It is never imputed.",
        "CrossEarth exists for Upper, Vanilla, FDA, SIB and BSM only. SegDeSiC, "
        "IIM, ISW, MRFP+, FADA and SAR have no CrossEarth counterpart, so on "
        "the 12-cell scope those six methods are still 9-cell rows and their "
        "n_cells_expected is recorded as 9. CrossEarth is measured against its "
        "own within-city upper bound in every case.",
        "Access classes follow erratum I-1's countersigned taxonomy as the "
        "operator's brief states it, which OVERRIDES each dump's own "
        "target_access field. Two systematic disagreements are recorded per "
        "row: every PNG-tree dump records source_only because the PNG pass is a "
        "scoring pass over stored artifacts, and arm F's CrossEarth "
        "within-city upper cells record source_only where arm C and arm D "
        "uppers record target_labels. "
        "summaries/target_access_audit.md v3 also lists SIB as "
        "unlabeled_target on OGLANet; I-1 is later and makes OGLANet-SIB "
        "source_only, which is what is used here.",
        "Comparing SIB against a baseline crosses access classes on two of "
        "three architectures: MAMNet-SIB is unlabeled_target while OGLANet-SIB "
        "and DINOv3-SIB are source_only, and the strongest baseline may sit in "
        "either class. The within-access-class rankings are reported so a "
        "same-class comparison is available; no pooled number here is corrected "
        "for the class mix, and the 2026-08-03 pooling ruling applies to the "
        "external criterion, not internally.",
        "CONTRACT.md rule zero puts the one metric implementation in "
        "shadowtransfer/metrics.py on the SLURM box. No copy of that file is "
        "reachable from the analysis host, so the metric is recomputed from the raw "
        "tp/fp/fn/tn counts and asserted on every cell read against the stored "
        "iou_fg_tol_pct / iou_bg_tol_pct / miou_tol_pct columns (max |delta| "
        "under 1e-9 everywhere), against the E-12 sidecars' own "
        "two_class_tol_k2_pct, and against each arm F SAR run's own "
        "delta_adaptation_only_miou_tol2 (max |delta| under 1e-6).",
        "The declared PNG tree arrived from Delta without an INDEX.parquet and "
        "the brief forbids writing inside dumps/, so the index was rebuilt here "
        "and written to analysis/_inv/INDEX_st_r450_published_png_26.parquet. "
        "Its sha256 is the index_sha256 of this JSON; the other three tree "
        "indexes are carried in index_sha256_all. The _v3 suffix is "
        "deliberate: results/12 cites analysis/_inv/INDEX_"
        "st_r450_published_png.parquet by sha256 and that file describes a "
        "76-cell tree, so it is left on disk unchanged rather than rewritten "
        "to describe the 85-cell tree v3 reads.",
        "High resolution only, rung 450 only, two-class average "
        "boundary-tolerant mIoU at K=2 only. The nine mid-resolution standard "
        "cells in the PNG tree are scored but are not used here and have no "
        "LOCO counterpart, so Table 2's pooled upper bounds still cannot be "
        "recomputed and erratum E-2's Table 2 side is still unchecked.",
        "No value anywhere in this JSON is rounded to match a printed value. "
        "Table 38 appears only inside agreement checks and inside the explicitly "
        "labelled missing-cell bound.",

        # ---- the matched-450 retrains' own caveats, carried verbatim in
        # substance from results/25_matched450_retrain_record.json ----------
        "n1. One seed (s42) per matched-450 cell, so every substituted row here "
        "is a single draw and carries no seed band. Training is not reproducible "
        "across jobs at a fixed seed: the rerun of mamnet-segdesic-phoenix "
        "stopped at epoch 39 (best 29) where the timed-out first attempt reached "
        "52 (best 42) at the same seed, the same 450-image list and the same "
        "recipe (delta_logs/session_logs/2026-10-29-n450-final-sar.md, step 1). "
        "A per-cell difference of that kind cannot be separated from a "
        "training-budget effect on one seed, which bounds how much of any "
        "movement between v3 and this job can be attributed to the budget.",

        "n2. MAMNet Phoenix IIM, ISW and MRFP+ score 2.82, 4.40 and 13.36 "
        "foreground IoU at n450 with normal validation (n_val 300, as every "
        "published pooled-LOCO cell). The published ISW and MRFP+ Phoenix cells "
        "were also low, at 5.51 and 9.59 foreground IoU, so on those two the "
        "condition is not introduced by the retrain; published IIM Phoenix was "
        "13.98. Kept as measured by operator decision; no cell was reseeded "
        "after test scores were seen, and none is dropped or imputed here. "
        "These three are not collapse in results/01's sense, which catches a "
        "cell predicting almost everything, not one predicting almost nothing: "
        "no cell on either side of this grid exceeds a pooled "
        "predicted-positive fraction of 0.308.",

        "n3. OGLANet values remain subject to the DFFM defect "
        "(results/21_disclosure_record.json finding 3), on the retrains exactly "
        "as on the published cells: every OGLANet number is one draw at one "
        "inference batch size. The OGLANet SegDeSiC self-check deltas +0.58 "
        "(chicago), +0.46 (miami) and +0.72 (phoenix) compare the trainer's "
        "batch-1 test() with the batch-4 scorer "
        "(delta_logs/session_logs/2026-10-28-n450-bs4-isw.md step 2: the same "
        "delta is 0.000000 when both sides are computed at batch 4); the "
        "batch-4 value is the scored one and is what enters here. "
        "mamnet-mrfp_plus-phoenix's self-check delta is -0.0136 against a 0.01 "
        "tolerance, a separate and unexplained disagreement on an architecture "
        "with no DFFM. oglanet-segdesic phoenix is taken from its -bs4 "
        "re-score, as the brief directs; its superseded original batch-4 pass "
        "scored 0.1144 foreground IoU lower, which is pass-to-pass "
        "nondeterminism at a fixed batch size and sits inside erratum E-9's "
        "measured nine-pass span of 1.1158.",

        "n4. All three OGLANet SIB n450 cells stopped at the 100-epoch maximum, "
        "not on patience (epochs_run 100 of 100, best epoch 90 / 97 / 84 for "
        "chicago / miami / phoenix). Their validation metric was still "
        "improving when the budget ran out, so those three rows are a lower "
        "bound on what the recipe reaches at 450 images rather than a converged "
        "result, and SIB's worst case here is correspondingly conservative on "
        "the OGLANet cells.",

        # ---- this job's own additional limitations ------------------------
        "The grid in this JSON is a MIXED grid by construction. Only the 27 "
        "cells that trained on 900 are substituted, because only those had a "
        "budget asymmetry to remove; every other row is the published cell. A "
        "method's row can therefore be a retrain on MAMNet and a published cell "
        "on DINOv3, and any comparison of one method across architectures mixes "
        "the two. n450_switch.candidates and the per_cell rows' "
        "n450_substituted field record which is which, cell by cell.",

        "Every substituted OGLANet row pairs a probability-map-scored method "
        "cell against a PNG-scored upper bound, and for OGLANet those are "
        "different predictions rather than two scorings of one prediction (20 "
        "of 20 OGLANet cells disagree, 0.76 to 3.05 percent of pixels, up to "
        "2.03 two-class mIoU). The matched-450 retrains are scored from their "
        "own probability maps by n450_score.py and no PNG-scored retrain exists, "
        "so for the six OGLANet substituted rows the population mixing v3 "
        "resolved for Vanilla comes back, in the other direction. On the 21 "
        "MAMNet substituted rows the two populations differ by at most 0.0147 "
        "two-class mIoU and the question does not arise. "
        "results/25_matched450_retrain_record.json carries the published "
        "probability-map value of every one of these cells so the size of the "
        "mixing can be read off.",

        "The n450 substitution rule in this job is NOT v3's. v3 refused any "
        "OGLANet n450 dump whose directory name lacked a bs4 suffix, on the "
        "assumption that a plain -n450 OGLANet dump would be inference-batch-1. "
        "The dumps that landed are batch 4 (OGLANet SegDeSiC) and batch 8 "
        "(OGLANet SIB) and record it in run.json, so the name test would have "
        "refused five comparable cells. The rule here reads the recorded batch "
        "size and requires it to equal the batch size the cell's own published "
        "comparator was produced at. Every refusal is recorded in "
        "n450_switch.refused with its measured and required batch size.",

        "This job adjudicates the pre-committed readings on a grid that is not "
        "the published grid. v3 remains the record of the published grid and is "
        "not superseded. Where a reading differs between the two, which reading "
        "the paper rests on is a scoping decision for the operator and the HLT, "
        "not something this JSON settles: both are reported, side by side, in "
        "v3_vs_n450.",

        # ---- new in this job ---------------------------------------------
        "The OGLANet inference pass carries NO run-to-run nondeterminism at a "
        "fixed inference batch size, and it nonetheless does not reproduce the "
        "value stored at training time. Five repeat inference passes of "
        "oglanet-sibc4-loco-r450-chicago-hr-s42-n450, each a separate SLURM job "
        "on a DIFFERENT node (jobs 23382471 / 23385066 / 23385067 / 23385068 / "
        "23385069, batch 8, fp16 autocast, epochs_override 0, seed 42), all "
        "returned two-class mIoU at K=2 of 68.531270 and foreground IoU of "
        "42.598234, bit-identical to every printed digit: the five-rep range is "
        "exactly 0.000000 on both metrics. All five sit 0.339357 two-class and "
        "0.592324 foreground BELOW the value stored in that cell's own run.json "
        "at training time (68.870627 / 43.190558), and the predictions differ "
        "too (865 against 827 argmax-versus-probability disagreement pixels). "
        "A reproducible offset of the same size five times is not noise. Its "
        "mechanism is NOT established. The leading hypothesis, flagged as a "
        "hypothesis: OGLANet's DFFM rebuilds about 20 unregistered "
        "Conv2d+BatchNorm2d layers inside forward(), anything constructed "
        "inside forward() is by definition absent from state_dict, so the "
        "trainer's own load_state_dict(strict=True) matches every registered "
        "key without restoring those layers, and a fresh process starts that "
        "part of the network from construction defaults where the original "
        "in-process pass carried its trained state. Three consequences for the "
        "numbers here. First, the three OGLANet-SIB n450 row values in this "
        "JSON come from n450_score.py's own pass, not from the trainer's stored "
        "value, so an offset of this kind is inside them; this JSON records the "
        "per-cell self-check delta (-0.0099 chicago, -0.0097 miami, +0.0050 "
        "phoenix) which is the same comparison at a much smaller size, and the "
        "0.3394 figure above is from a separate repeat-inference harness on one "
        "cell. Second, the explanation results/26's lineage offered for the "
        "three oglanet-segdesic self-check failures (+0.5814 / +0.4608 / "
        "+0.7196), that they are re-run noise, is REFUTED by the zero range and "
        "is withdrawn; that anomaly is open. Third, if the hypothesis holds it "
        "is a contract-level finding, because CLAUDE.md requires that a failed "
        "dump never require retraining and for the SIB family a dump-only "
        "re-entry would not reproduce the original dump. The decisive test -- "
        "compare state_dict keys against named_modules() for an OGLANetSIB and "
        "count what is built inside forward() -- has not been run. Source: "
        "delta_logs/session_logs/2026-10-31-oglanet-sib-n450-diagnosis.md "
        "section 12.1.",

        "The published OGLANet-SIB comparator is the fp16 trainer pass on all "
        "three cities, and chicago is the only city on which that choice is "
        "checkable. The rule matters because the two regimes disagree by 1.2536 "
        "two-class and 2.3234 foreground on chicago, the one cell where both "
        "exist: fp16 trainer pass 73.0475762379333 / 50.827375740052645 against "
        "fp32 re-score of the stored probability maps 71.79402162754555 / "
        "48.50400555962078. Table 38's SIB row came from the fp16 pass, the "
        "n450 cells were scored in the same regime, and miami and phoenix have "
        "no fp32 re-score at all, so fp16 against fp16 is both the like-for-like "
        "rule and the only rule all three cities can satisfy. On chicago the "
        "PNG tree's scoring of the published prediction masks reproduces the "
        "fp16 value and this job asserts that agreement to 1e-6; on miami and "
        "phoenix there is no second measurement of the published value and the "
        "comparator rests on that single test_results.json file.",

        "The published-against-n450 contrast on the three OGLANet-SIB cells is "
        "a cell-mean contrast only, on all three cities. The published side has "
        "no per-image vector anywhere on this box for miami and phoenix, and "
        "for chicago the per-image vector that exists is the PNG tree's, which "
        "this job does use for the paired bootstrap on that one cell. So the "
        "miami and phoenix published-against-n450 differences carry no interval "
        "and no test, and the three cities' contrasts are not computed the same "
        "way. They are recorded per row as "
        "delta_n450_scored_minus_published_comparator and are reported as "
        "differences of two single draws, not as an effect.",
    ]

    v3sib9 = side["reading"]["sib_worst_case_matched_9"]["v3"]
    r26sib9 = side26["reading"]["sib_worst_case_matched_9"]["r26"]
    headline = (
        f"On the declared path of two-class boundary-tolerant mIoU at K=2, high "
        f"resolution, rung 450, with the 27 cells that trained on 900 source "
        f"images replaced by their matched-450 retrains, the three MAMNet SAR "
        f"rows taking their within-pass change from runs on the n450 "
        f"checkpoints rather than the published ones, and the published "
        f"OGLANet-SIB comparator read as the measured fp16 trainer-pass values "
        f"73.0475762 / 66.7057610 / 63.9668490 two-class, SIB's worst-case gap "
        f"to the within-city upper bound is {sib9['worst_case_gap']:.2f} mIoU "
        f"at {sib9['worst_cell']} over {sib9['n_cells']} of 9 cells "
        f"(results/26: {r26sib9:.2f}; v3: {v3sib9:.2f} over 7 of 9), its mean "
        f"gap is {margins['9_cell']['sib_mean_gap']:.2f} mIoU, its mean margin "
        f"over {margins['9_cell']['strongest_baseline_label']} is "
        f"{margins['9_cell']['margin_mean_gap_difference']:+.2f} mIoU, reading "
        f"{reading['scope_9_cell']} fires on the 9-cell scope and "
        f"{reading['scope_12_cell']} on the 12-cell scope "
        f"(v3: {side['reading']['scope_9_cell']['v3']} and "
        f"{side['reading']['scope_12_cell']['v3']}), and "
        f"{side26['per_cell_diff']['n_cells_changed']} of "
        f"{side26['per_cell_diff']['n_cells_identical'] + side26['per_cell_diff']['n_cells_changed']} "
        f"cells differ from results/26.")

    # ---- assemble --------------------------------------------------------
    def strip(r):
        return {k: v for k, v in r.items()
                if k not in ("per_image", "image_id", "sar_delta_per_image",
                             "published_per_image")}

    out = dict(
        id="31_internal_sib_worst_case_tgtfda",
        arm="F",
        answers=["BxRx-W3", "AC3", "v1UJ-W3", "AC1"],
        answers_detail=(
            "BxRx-W3, AC3 and v1UJ-W3 are v3's, re-adjudicated on the "
            "matched-450 retrains. AC1 is added because the budget asymmetry "
            "this job removes was v3's caveat 1 and "
            "results/21_disclosure_record.json finding 1: 27 of the grid's "
            "cross-city cells trained on 900 source images against 450-image "
            "within-city upper bounds. This job carries the same four IDs as "
            "results/26 because it is the same adjudication with two inputs "
            "upgraded from assumption to measurement; it does not answer a new "
            "weakness."),
        question=("Does SIB hold the smallest largest-single-cell gap to the "
                  "within-city upper bound of any evaluated method on the "
                  "internal grid, and what is its mean-mIoU margin over the "
                  "strongest baseline, once the 27 cells that trained on 900 "
                  "source images are replaced by their matched-450 retrains?"),
        status="ADJUDICATED",
        supersedes=["results/26_internal_sib_worst_case_n450.json"],
        supersedes_why=(
            "results/26 is superseded on exactly two points and on nothing "
            "else. (1) Its three MAMNet SAR rows applied a within-pass change "
            "measured on the PUBLISHED 900-image checkpoints to a matched-450 "
            "unadapted level, which it recorded as caveat 4b, an assumption; "
            "the n450 SAR rerun has landed and those rows are now measured on "
            "the same checkpoint family, so the assumption is gone rather than "
            "restated. (2) Its published OGLANet-SIB miami and phoenix values "
            "were Table 38 bounds; they are now measured fp16 trainer-pass "
            "values. Every other number in results/26 is reproduced here "
            "unchanged, and vs_results_26 is the cell-by-cell record of that. "
            "results/26 is not deleted and remains readable; it is the record "
            "of the grid as it stood before the SAR rerun landed."),
        companion_to=["results/22_internal_sib_worst_case_v3.json",
                      "results/26_internal_sib_worst_case_n450.json"],
        companion_why=("This job does NOT supersede v3. v3 is the published "
                       "grid as published and remains the record of it; this "
                       "job is the same computation with 27 of its rows "
                       "replaced by matched-450 retrains, which is a different "
                       "grid and a different question. The two are reported "
                       "side by side in v3_vs_n450 and neither replaces the "
                       "other. v3's own lineage is unchanged: v3 superseded "
                       "results/12_internal_sib_worst_case_v2.json, which "
                       "superseded results/11 (GATE_FAILED)."),
        matched_450_record="results/25_matched450_retrain_record.json",
        png_vanilla_entry_condition=png_vanilla_state,
        declared_path=dict(
            metric="two_class_mIoU_boundary_tolerant_K2_pct",
            definition=("per image, the mean of foreground IoU and background "
                        "IoU computed from raw tp/fp/fn/tn on the valid region "
                        "with a +-2 px don't-care band, then macro-averaged "
                        "over the cell's 150 test images"),
            degenerate_rule=("App E: an image with tp+fp+fn == 0 at K=2 scores "
                             "foreground 0 and background 100, so its "
                             "two-class value is exactly 50"),
            gap_sign=dict(
                gap="method minus its within-city upper bound (CONVENTIONS.md "
                    "sign; negative when the method is below the upper bound)",
                gap_upper_minus_method="upper bound minus method (the brief's "
                                       "sign; positive is a shortfall). "
                                       "\"Worst case\" always means the largest "
                                       "gap_upper_minus_method over a scope."),
            table_family="Table 38 / revised Table 42",
            resolution="hr", rung=450,
            source=("dumps/st_r450_published_png (the saved prediction PNG "
                    "masks, the tree Table 38 and final_comparison.json were "
                    f"scored from), {png_info['n_rows']} cells"),
            fallback=("dumps/st_r450_published_recompute (published cells "
                      "re-passed over their own stored float16 probability "
                      "maps), used only where the PNG tree carries no dump and "
                      f"recorded per row; {len(fell_back)} rows fall back in "
                      "v3"),
            other_trees=dict(
                bsm="dumps/st (arm F BSM training runs)",
                crossearth="dumps/st (arm F / arm C-D CrossEarth runs)",
                mamnet_sib="delta_dumps/dumps/e12 (E-12 clean-pool retrains)",
                sar="runs/armF_sar_{arch}_{city}_highres/perstep.parquet for "
                    "OGLANet and DINOv3 (their Vanilla cells were already at "
                    "450), delta_runs/armF_sar_mamnet_{city}_highres-n450/"
                    "perstep.parquet for MAMNet (whose Vanilla cells are "
                    "matched-450 retrains, so the change must be measured on "
                    "the same checkpoint family)"),
            reproduction=repro,
            validated_against=[
                "stored per-image iou_fg_tol_pct / iou_bg_tol_pct / "
                "miou_tol_pct columns, max |delta| < 1e-9 on every cell read",
                "the PNG dumps' own recorded Table 38 and "
                "final_comparison.json deltas",
                "the E-12 sidecars' two_class_tol_k2_pct",
                "each arm F SAR run's own delta_adaptation_only_miou_tol2, "
                "max |delta| < 1e-6"],
            image_id_identity=id_check),
        scopes=dict(
            scope_9_cell="the three original architectures (MAMNet, OGLANet, "
                         "DINOv3) x three held-out cities",
            scope_12_cell="the same plus CrossEarth, for the five methods "
                          "CrossEarth was run with (Upper, Vanilla, FDA, SIB, "
                          "BSM); the other six methods have no CrossEarth "
                          "counterpart and stay 9-cell rows"),
        precommitted_readings=PRECOMMITTED,
        reading_fired=reading,
        per_method=per_method,
        matched_cell_comparison=matched,
        per_cell=[strip(r) for r in per_cell],
        vanilla_worst_case=vanilla_worst,
        sib_margin=margins,
        within_access_class_rankings=dict(
            whole_method=rankings,
            whole_method_note="methods whose cells all share one access class. "
                              "SIB is absent from this view because its cells "
                              "do not: MAMNet-SIB is unlabeled_target through "
                              "FDA while OGLANet-SIB and DINOv3-SIB are "
                              "source_only.",
            by_cell_access_class=rankings_by_cell,
            by_cell_access_class_note="every method restricted to the cells "
                                      "whose own access class is the one named, "
                                      "which is the only view a same-class "
                                      "comparison can place SIB in. Cell counts "
                                      "differ within a ranking, so these are "
                                      "comparisons of different cell sets and "
                                      "the matched-cell comparison is the one "
                                      "the reading rests on."),
        mamnet_sib_seed_table=[
            dict(city=c,
                 per_seed=rows[("sib", "mamnet", c)]["per_seed"],
                 seed_mean=rows[("sib", "mamnet", c)]["value"],
                 seed_sd=rows[("sib", "mamnet", c)]["seed_sd"],
                 published_contaminated=rows[("sib", "mamnet", c)][
                     "published_contaminated_value"],
                 published_contaminated_repro_delta=rows[
                     ("sib", "mamnet", c)]["published_contaminated_repro_delta"])
            for c in CITIES],
        conditions=conditions,
        tests=tests,
        n450_switch=n450_block,
        v3_vs_n450=side,
        vs_results_26=side26,
        rows_fell_back_from_png_tree=fell_back,
        rows_without_a_value=excluded,
        runs=sorted({r["run_id"] for r in rows.values()
                     if isinstance(r.get("run_id"), str)}
                    | {x for r in rows.values()
                       if isinstance(r.get("run_id"), list)
                       for x in r["run_id"]}),
        runs_excluded=[dict(run_id=e["run_id"], reason=e["reason"])
                       for e in excluded]
        + [dict(run_id=f"{a}-{t}-r450-{c}-mr-s0",
                reason="mid-resolution; the declared path is high resolution "
                       "only")
           for a in ARCHS for t in ("none-upper",) for c in CITIES],
        index_sha256=png_info.get("sha256"),
        index_sha256_all={k: dict(path=v.get("path"), tree=v.get("root"),
                                  sha256=v.get("sha256"), n_rows=v["n_rows"],
                                  built_here=v.get("built_here"),
                                  n_unindexable=v.get("n_unindexable"),
                                  unindexable=v.get("unindexable"))
                          for k, v in indexes.items()},
        index_on_disk_state=on_disk,
        source_artifacts=[
            dict(path="delta_runs/dffm_defect/provenance.json",
                 role="the DFFM defect and the inference batch size of every "
                      "published OGLANet artifact family (caveats 3 and 4)"),
            dict(path="delta_runs/retrained_cells_record/results.json",
                 role="the six cells retrained after 2026-03-19 (caveat 5)"),
            dict(path="delta_runs/train_count_audit_projects/"
                      "retrain_at_450_list.json",
                 role="the measured training counts and the 27-cell matched-450 "
                      "retrain list (caveat 1)"),
            dict(path="delta_runs/oglanet_saved_prediction_source/"
                      "vanilla_reference_scores.json",
                 role="PNG-scored OGLANet vanilla values for chicago and "
                      "miami, used only to quantify the fallback mixing"),
            dict(path="summaries/HLT_COUNTERSIGNS_armA_e12.md",
                 role="the countersigned E-12 seed table and the G1 flip"),
            dict(path="final_paper.md",
                 role="Table 38 printed values, used only in agreement checks "
                      "and in the labelled missing-cell bound"),
            dict(path="results/25_matched450_retrain_record.json",
                 role="the per-cell record of the 27 matched-450 retrains "
                      "substituted here: their published counterparts, their "
                      "inference batch sizes, their collapse check, the "
                      "OGLANet Phoenix near-tie measurements, and caveats n1-n4"),
            dict(path="results/21_disclosure_record.json",
                 role="finding 1 (which 27 cells trained on 900 and why a "
                      "matched-450 retrain cannot restore the published image "
                      "draw) and finding 3 (the DFFM defect and the inference "
                      "batch size of every published OGLANet artifact family)"),
            dict(path="delta_logs/session_logs/2026-10-28-n450-bs4-isw.md",
                 role="the batch-4 re-score of oglanet-segdesic phoenix, the "
                      "batch-size explanation of the g3 self-check, the two "
                      "cells whose inference_batch_size field is null, and the "
                      "failure of the batch-4 OGLANet-SIB attempt"),
            dict(path="delta_logs/session_logs/2026-10-29-n450-final-sar.md",
                 role="the mamnet-segdesic-phoenix epoch counts behind caveat "
                      "n1, the four failing self-checks, and the arm F SAR "
                      "n450 build this job reads"),
            dict(path="delta_logs/session_logs/2026-10-31-n450-sar-closed.md",
                 role="the arm F SAR n450 closure: three of three cells "
                      "COMPLETED, the hard n450 gate's measured differences, "
                      "the weight-drift pass, the inherited "
                      "no_meaningful_adaptation config, and the published "
                      "versus n450 change table behind caveat 4b"),
            dict(path="delta_logs/session_logs/"
                      "2026-10-31-oglanet-sib-n450-diagnosis.md",
                 role="section 12.0 (450 per fold is the intended budget, so "
                      "the published counterparts are 2x-data cells and there "
                      "is no drop to explain), section 12.1 (the five-rep "
                      "zero-range repeat inference and the 0.3394 offset "
                      "behind this job's new caveat), section 12.2 (the "
                      "corrected 27-of-27 count), section 12.3 (the fp16 "
                      "trainer-pass published OGLANet-SIB values this job "
                      "relabels, chicago's fp32 re-score, and the "
                      "fp16-against-fp16 comparator rule) and section 12.4 "
                      "(target_access source_only, resolved by supersession)"),
            dict(path="delta_runs/armF_sar_mamnet_{city}_highres-n450/",
                 role="the SAR rerun on the matched-450 MAMNet checkpoints: "
                      "results.json and perstep.parquet per city, read for the "
                      "within-pass change of the three MAMNet SAR rows"),
        ],
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/31_internal_sib_worst_case_tgtfda.py",
        env=dict(python=sys.version.split()[0],
                 conda_env="myenv "
                           "(conda env)",
                 pythonnousersite=os.environ.get("PYTHONNOUSERSITE"),
                 numpy=np.__version__, pandas=pd.__version__),
        seed=BOOT_SEED,
        seed_note=f"the only stochastic step is the bootstrap, B = {B}, "
                  f"numpy default_rng seeded at {BOOT_SEED}",
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; "
                        "only final_paper/ is git-connected and this job does "
                        "not touch it",
        compute_note="read-only CPU analysis on the analysis host. No training, no GPU "
                     "inference, no checkpoint loaded, nothing under dumps/ or "
                     "results/ modified.",
        headline=headline,
        sanity_flags=sanity,
        caveats=caveats,
        figure=None,
        caption=None,
    )

    # =========================================================================
    # results/31 AMENDMENT. Everything above is results/27's computation run on
    # the swapped OGLANet-SIB cells. This block adds the results/27 side table,
    # the explicit SIB-minus-FDA paired difference, the ten-method ranks, the
    # new caveats, and replaces the headline. It recomputes no row.
    # =========================================================================
    side27 = r27_side_table(per_method, matched, margins, vanilla_worst,
                            reading, adj, per_cell)
    print("\n--- vs results/27: which rows moved ---")
    for c in side27["per_cell_diff"]["changed"]:
        print(f"    {c['method']}/{c['arch']}/{c['city']} "
              f"{c['r27_gap_upper_minus_method']:+.6f} -> "
              f"{c['this_gap_upper_minus_method']:+.6f}  change "
              f"{c['change']:+.6f}  ({c['r27_run_id']} -> {c['this_run_id']})")

    # SIB minus FDA, paired, per image, image-cluster bootstrap by held-out
    # city: the same function and seed results/27 used for its paired_margin,
    # with the comparator FIXED to FDA rather than chosen as the strongest
    # baseline, so the quantity stays defined if the strongest baseline moves.
    sib_minus_fda = {}
    for scope in ("9_cell", "12_cell"):
        se = next(e for e in per_method
                  if e["method"] == "sib" and e["scope"] == scope)
        fe = next(e for e in per_method
                  if e["method"] == "fda" and e["scope"] == scope)
        keep = sorted(set(se["per_cell_gaps"]) & set(fe["per_cell_gaps"]))
        per_city = {}
        for k in keep:
            a_, c_ = k.split("/")
            per_city.setdefault(c_, []).append(
                np.asarray(rows[("sib", a_, c_)]["per_image"])
                - np.asarray(rows[("fda", a_, c_)]["per_image"]))
        cb = cluster_bootstrap_mean_over_cells(per_city)
        sib_minus_fda[scope] = dict(
            cells=keep, definition=(
                "per image, SIB minus FDA on the cells both rows cover, "
                "averaged over cells, image-cluster bootstrap by held-out "
                "city (results/27's paired_margin function and seed). Positive "
                "means SIB scores higher than FDA."), **cb)
        if margins[scope].get("strongest_baseline") == "fda":
            d_ = abs(margins[scope]["paired_margin"]["mean"] - cb["mean"])
            if d_ > 1e-12 or margins[scope]["paired_margin"]["ci95"] != cb["ci95"]:
                fail(f"{scope}: SIB-minus-FDA recomputation disagrees with "
                     "paired_margin although FDA is the strongest baseline")

    r27 = json.load(open(R27_PATH))

    def ranks(pm_list, basis_key):
        out_ = {}
        for scope in ("9_cell", "12_cell"):
            ents = [e for e in pm_list if e["scope"] == scope
                    and e["method"] != "upper"]
            for fld in ("worst_case_gap", "mean_gap"):
                have = [e for e in ents if e.get(fld) is not None]
                srt = sorted(have, key=lambda e: e[fld])
                sib_v = next(e[fld] for e in have if e["method"] == "sib")
                out_[f"{scope}/{fld}"] = dict(
                    basis=basis_key, n_methods_ranked=len(srt),
                    sib_rank=1 + sum(1 for e in have if e[fld] < sib_v),
                    n_ties_with_sib=sum(1 for e in have if e[fld] == sib_v
                                        and e["method"] != "sib"),
                    rank_definition="1 = smallest gap of the methods ranked; "
                                    "Upper excluded",
                    order=[dict(method=e["method"], value=e[fld],
                                n_cells=e.get("n_cells_available",
                                              e.get("n_cells")))
                           for e in srt])
        return out_

    rank_this = dict(as_available=ranks(per_method, "per_method"),
                     matched_cell=ranks(matched, "matched_cell_comparison"))
    rank_r27 = dict(as_available=ranks(r27["per_method"], "per_method"),
                    matched_cell=ranks(r27["matched_cell_comparison"],
                                       "matched_cell_comparison"))

    def pick(pm_list, scope, method):
        return next(e for e in pm_list
                    if e["scope"] == scope and e["method"] == method)

    moved = []
    for scope in ("9_cell", "12_cell"):
        for basis, a_list, b_list in (
                ("as_available", r27["per_method"], per_method),
                ("matched_cell", r27["matched_cell_comparison"], matched)):
            a_, b_ = pick(a_list, scope, "sib"), pick(b_list, scope, "sib")
            moved.append(dict(
                scope=scope, basis=basis,
                sib_worst_case_gap=dict(r27=a_["worst_case_gap"],
                                        this=b_["worst_case_gap"],
                                        change=b_["worst_case_gap"]
                                        - a_["worst_case_gap"]),
                sib_worst_cell=dict(r27=a_["worst_cell"], this=b_["worst_cell"]),
                sib_mean_gap=dict(r27=a_["mean_gap"], this=b_["mean_gap"],
                                  change=b_["mean_gap"] - a_["mean_gap"]),
                sib_mean_gap_ci95=dict(r27=a_.get("mean_gap_ci95"),
                                       this=b_.get("mean_gap_ci95")),
                sib_rank_worst_case=dict(
                    r27=rank_r27[basis][f"{scope}/worst_case_gap"]["sib_rank"],
                    this=rank_this[basis][f"{scope}/worst_case_gap"]["sib_rank"],
                    of=rank_this[basis][f"{scope}/worst_case_gap"]["n_methods_ranked"]),
                sib_rank_mean_gap=dict(
                    r27=rank_r27[basis][f"{scope}/mean_gap"]["sib_rank"],
                    this=rank_this[basis][f"{scope}/mean_gap"]["sib_rank"],
                    of=rank_this[basis][f"{scope}/mean_gap"]["n_methods_ranked"])))
    r27_pm = {s_: r27["sib_margin"][s_] for s_ in ("9_cell", "12_cell")}
    sib_fda_moved = {}
    for scope in ("9_cell", "12_cell"):
        a_ = r27_pm[scope]
        if a_.get("strongest_baseline") != "fda":
            fail(f"results/27 {scope}: strongest baseline is "
                 f"{a_.get('strongest_baseline')}, so its paired_margin is not "
                 "SIB minus FDA and cannot be the predecessor value")
        sib_fda_moved[scope] = dict(
            r27=dict(mean=a_["paired_margin"]["mean"],
                     ci95=a_["paired_margin"]["ci95"],
                     p=a_["paired_margin"]["p"],
                     source="results/27 sib_margin[scope].paired_margin "
                            "(strongest baseline FDA)"),
            this=dict(mean=sib_minus_fda[scope]["mean"],
                      ci95=sib_minus_fda[scope]["ci95"],
                      p=sib_minus_fda[scope]["p"]),
            change=sib_minus_fda[scope]["mean"] - a_["paired_margin"]["mean"])
    reading_moved = dict(
        scope_9_cell=dict(r27=r27["reading_fired"]["scope_9_cell"],
                          this=reading["scope_9_cell"]),
        scope_12_cell=dict(r27=r27["reading_fired"]["scope_12_cell"],
                           this=reading["scope_12_cell"]),
        scope_12_cell_label_is_a_fallback=reading.get(
            "scope_12_cell_label_is_a_fallback"),
        boolean_evaluation=dict(r27=r27["reading_fired"]["boolean_evaluation"],
                                this=reading["boolean_evaluation"]),
        r4_also_fires=dict(r27=r27["reading_fired"].get("r4_also_fires"),
                           this=reading.get("r4_also_fires")))
    print("\n--- results/31 vs results/27 ---")
    for e in moved:
        print(f"  {e['scope']:8s} {e['basis']:13s} SIB worst "
              f"{e['sib_worst_case_gap']['r27']:.4f} ({e['sib_worst_cell']['r27']})"
              f" -> {e['sib_worst_case_gap']['this']:.4f} "
              f"({e['sib_worst_cell']['this']}); mean "
              f"{e['sib_mean_gap']['r27']:.4f} -> {e['sib_mean_gap']['this']:.4f};"
              f" rank wc {e['sib_rank_worst_case']['r27']}->"
              f"{e['sib_rank_worst_case']['this']}/{e['sib_rank_worst_case']['of']}"
              f" mean {e['sib_rank_mean_gap']['r27']}->"
              f"{e['sib_rank_mean_gap']['this']}/{e['sib_rank_mean_gap']['of']}")
    for scope, e in sib_fda_moved.items():
        print(f"  {scope} SIB-FDA paired {e['r27']['mean']:+.4f} "
              f"{e['r27']['ci95']} p {e['r27']['p']} -> {e['this']['mean']:+.4f} "
              f"{e['this']['ci95']} p {e['this']['p']}")
    print(f"  reading 9: {reading_moved['scope_9_cell']}  12: "
          f"{reading_moved['scope_12_cell']}")

    # ---- caveats -------------------------------------------------------------
    og = {c: rows[("sib", "oglanet", c)] for c in CITIES}
    og_pc = {r["city"]: r for r in per_cell
             if r["method"] == "sib" and r["arch"] == "oglanet"}
    sc = {c: og[c].get("n450_self_check_delta") for c in CITIES}
    new_caveats = [
        "(a) One seed per new cell. Each of the three OGLANet-SIB rows is one "
        "training run at seed 42 (oglanet-sibc4tgt-loco-r450-{chicago,miami,"
        "phoenix}-hr-s42-n450); no second seed of the held-out-city FDA pool "
        "configuration exists, so no seed band accompanies these rows and a "
        "seed-to-seed change of the size measured on MAMNet-SIB (E-12) is not "
        "separable from the swap.",
        "(b) The internal FDA donor pool of the three new cells is the held-out "
        "city's 450 training images (unlabeled; fda_donor "
        "heldout_city_train_images, fda_donor_n 450, runtime donor assertions "
        "PASS with zero test or val overlap by id and by content). The "
        "cross-source n450 cells results/27 used drew from 225 images of the "
        "other source city, so against results/27 the donor differs in city "
        "AND in pool size; against the published cell only the city differs "
        "(run.json fda_donor_pool_size_note).",
        "(c) OGLANet inference depends on batch size and on RNG state "
        "(results/30_oglanet_rng_dependence.json): at a fixed torch seed two "
        "jobs were bit-identical, while changing the seed among 42, 0 and 1 "
        "moved two-class mIoU at K=2 by up to 0.660029 on an OGLANet-SIB n450 "
        "checkpoint at batch 8 fp16. That checkpoint is the superseded "
        "cross-source chicago cell; no RNG or repeat-inference measurement "
        "exists for any sibc4tgt cell. Each OGLANet row here is one draw at "
        "batch 8 under fp16 autocast with test-pass seed 42 (run.json "
        "test_pass_rng), and the miami and phoenix cells share a test-pass CPU "
        "RNG state sha while chicago's differs (session log 2026-11-02 "
        "section 7).",
        f"(d) The scorer self-check delta (n450_score.py nofilter foreground "
        f"IoU at K=2 minus the trainer's tolerant_2px.Shadow_IOU, same "
        f"checkpoint) exceeds its hard-coded 0.01 on chicago ({sc['chicago']:+.4f}) "
        f"and miami ({sc['miami']:+.4f}); phoenix is {sc['phoenix']:+.4f}. The "
        "0.01 is a literal at n450_score.py:179, not a band derived from "
        "passes. The recorded mechanism: the trainer averages per-image "
        "tolerant IoU from argmax under autocast, the scorer recomputes from "
        "the saved float16 probability maps at p > 0.5, and float16 rounding "
        "at the 0.5 boundary is the difference; the cross-source comparators "
        "sat at -0.0099 / -0.0097 / +0.0050. The check gated nothing. Source: "
        "delta_logs/session_logs/2026-11-02-oglanet-sib-tgtfda-results.md "
        "section 6.",
        "Value field. Every OGLANet-SIB row value is the field results/27 used "
        "for its OGLANet-SIB n450 rows: the per-image two-class "
        "boundary-tolerant mIoU at K=2 recomputed from the dump's tp_tol / "
        "fp_tol / fn_tol / tn_tol (App E degenerate rule), asserted to 1e-9 "
        "against the stored miou_tol_pct column, macro-averaged over 150 "
        "images. That is the primary prediction recipe "
        "(threshold_and_filter, min_pixels=10) and equals run.json "
        "scoring.miou_k2_twoclass (69.1742 / 63.4838 / 61.9742).",
        "Access class. OGLANet-SIB is unlabeled_target in this JSON, from each "
        "new cell's run.json, replacing the source_only class results/27 "
        "applied under erratum I-1. SIB's source_only cells are therefore "
        "DINOv3 only on the 9-cell scope (3 cells) and DINOv3 plus CrossEarth "
        "on the 12-cell scope (6 cells); MAMNet-SIB and OGLANet-SIB are "
        "unlabeled_target. Every within-access-class ranking below follows "
        "that class.",
        "Trainer two-class values absent. The new cells' trainer-pass two-class "
        "mIoU is in test_results.json on Delta and was not copied to the analysis host, so "
        "n450_trainer_pass.two_class and "
        "delta_n450_trainer_minus_published_comparator are null on the three "
        "OGLANet-SIB rows; the trainer's foreground value is carried and "
        "asserted against run.json.",
        "Second difference in the training run, not only the donor. chicago's "
        "new cell early-stopped at epoch 75 (best 50) where the cross-source "
        "cell ran to the 100-epoch maximum (best 90); miami and phoenix ran to "
        "100 in both. Wall clock and the cell's own early-stopping trajectory "
        "therefore differ between the two chicago cells, in addition to the "
        "donor pool.",
    ]
    STALE_MARKERS = STALE_MARKERS_31
    stale = []
    for kind, lst in (("caveats", caveats), ("sanity_flags", sanity)):
        for i, c in enumerate(lst):
            hits = [m_ for m_ in STALE_MARKERS if m_["match"] in c]
            if hits:
                stale.append(dict(list=kind, index_in_carried_list=i,
                                  excerpt=c[:160], correction=[h["correction"]
                                                               for h in hits]))
    for m_ in STALE_MARKERS:
        n_hit = sum(1 for kind, lst in (("caveats", caveats),
                                        ("sanity_flags", sanity))
                    for c in lst if m_["match"] in c)
        if n_hit != m_["expect"]:
            fail(f"stale marker {m_['match'][:50]!r} hit {n_hit} carried "
                 f"items, expected {m_['expect']}")

    sib9 = next(e for e in moved if e["scope"] == "9_cell"
                and e["basis"] == "matched_cell")
    new_headline = (
        f"With the three OGLANet-SIB cells taken from the held-out-city FDA "
        f"pool retrains (sibc4tgt, seed 42) instead of results/27's "
        f"cross-source n450 cells, SIB's worst-case gap to the within-city "
        f"upper bound on the 9-cell n450 grid is "
        f"{sib9['sib_worst_case_gap']['this']:.2f} mIoU at "
        f"{sib9['sib_worst_cell']['this']} (results/27: "
        f"{sib9['sib_worst_case_gap']['r27']:.2f} at "
        f"{sib9['sib_worst_cell']['r27']}), its mean gap is "
        f"{sib9['sib_mean_gap']['this']:.2f} mIoU (results/27: "
        f"{sib9['sib_mean_gap']['r27']:.2f}), SIB minus FDA is "
        f"{sib_minus_fda['9_cell']['mean']:+.2f} mIoU per image, 95% CI "
        f"[{sib_minus_fda['9_cell']['ci95'][0]:+.2f}, "
        f"{sib_minus_fda['9_cell']['ci95'][1]:+.2f}], p "
        f"{sib_minus_fda['9_cell']['p']:.4f}, and reading "
        f"{reading['scope_9_cell']} fires on the 9-cell scope and "
        f"{reading['scope_12_cell']} on the 12-cell scope (results/27: "
        f"{r27['reading_fired']['scope_9_cell']} and "
        f"{r27['reading_fired']['scope_12_cell']}).")

    out.update(
        status="ADJUDICATED",
        headline=new_headline,
        headline_in_results_27_form=headline,
        headline_in_results_27_form_note=(
            "results/27's own headline sentence, regenerated by the inherited "
            "code on this job's rows. Its fixed phrases (the fp16 comparator "
            "values, 'results/26', 'v3') describe the inherited comparisons and "
            "are correct as statements about those files; its N-of-M cell count "
            "is against results/26."),
        supersedes=["results/27_internal_sib_worst_case_n450_sar.json"],
        supersedes_why=(
            "results/27 is superseded on exactly one input: its three "
            "OGLANet-SIB rows came from n450 retrains whose FDA style pool was "
            "the other SOURCE city, while the paper's SIB draws it from the "
            "HELD-OUT city's unlabeled train split. Those rows now come from "
            "the held-out-city pool retrains, and OGLANet-SIB's access class "
            "follows the cells to unlabeled_target. Every other row is "
            "results/27's, asserted cell by cell in vs_results_27. results/27 "
            "is not deleted and remains readable."),
        companion_to=["results/22_internal_sib_worst_case_v3.json",
                      "results/26_internal_sib_worst_case_n450.json",
                      "results/27_internal_sib_worst_case_n450_sar.json"],
        vs_results_27=side27,
        moved_from_results_27=dict(
            sib_per_scope_and_basis=moved,
            sib_minus_fda_paired=sib_fda_moved,
            reading=reading_moved,
            ranks_among_ten_methods=dict(this=rank_this, r27=rank_r27)),
        sib_minus_fda_paired=sib_minus_fda,
        oglanet_sib_cells_used=[dict(
            city=c, run_id=og[c]["run_id"], tree=og[c]["tree"],
            value_two_class_k2=og[c]["value"], value_fg_k2=og[c]["value_fg"],
            gap_upper_minus_method=og_pc[c]["gap_upper_minus_method"],
            ci95_gap_upper_minus_method=og_pc[c]["ci95_gap_upper_minus_method"],
            upper_value=og_pc[c]["upper_value"],
            fda_donor=og[c].get("fda_donor"), fda_donor_n=og[c].get("fda_donor_n"),
            fda_donor_runtime_assertions_verdict=og[c].get(
                "fda_donor_runtime_assertions_verdict"),
            sidecar_target_access=og[c].get("sidecar_target_access"),
            access_class=og[c]["access_class"],
            n450_epochs_run=og[c].get("n450_epochs_run"),
            n450_stopped_on=og[c].get("n450_stopped_on"),
            n450_self_check_delta=og[c].get("n450_self_check_delta"),
            n450_self_check_pass_0p01=og[c].get("n450_self_check_pass_0p01"),
            superseded_cross_source_run_id=og[c].get(
                "superseded_cross_source_run_id"))
            for c in CITIES],
        value_field=("per-image two-class boundary-tolerant mIoU at K=2 "
                     "recomputed from tp_tol/fp_tol/fn_tol/tn_tol, asserted "
                     "against the stored miou_tol_pct column, mean over 150 "
                     "images: the field results/27 used for its OGLANet-SIB "
                     "n450 rows (row key `value`)"),
        caveats=new_caveats + caveats,
        caveats_layout=(
            f"caveats[0:{len(new_caveats)}] are new in results/31. "
            f"caveats[{len(new_caveats)}:] are results/27's caveats, "
            "regenerated by the inherited code on this job's rows: f-string "
            "numbers are recomputed, fixed prose is carried verbatim. Where "
            "the fixed prose describes the superseded cross-source OGLANet-SIB "
            "cells, carried_items_stale_for_oglanet_sib names the item and "
            "states the correction; the item itself is kept unedited."),
        carried_items_stale_for_oglanet_sib=stale,
        carried_note_stale_for_oglanet_sib=dict(
            field="within_access_class_rankings.whole_method_note",
            excerpt=(out.get("within_access_class_rankings") or {})
            .get("whole_method_note"),
            correction="results/31: MAMNet-SIB and OGLANet-SIB are "
                       "unlabeled_target and DINOv3-SIB source_only, so SIB "
                       "still spans two classes and stays absent from the "
                       "whole-method view; the parenthetical class list in "
                       "the note is results/27's."),
        script_lineage=dict(
            built_from="analysis/27_internal_sib_worst_case_n450_sar.py",
            how=("a copy with count-asserted edits: the OGLANet-SIB n450 stem "
                 "(sibc4 -> sibc4tgt), its access class, the trainer-pass "
                 "constants, the _31 index suffix, the output path, a results/27 "
                 "side table generated from the results/26 one, and this "
                 "amendment block. resolve_row, every bootstrap, every reading "
                 "rule and every inherited caveat are unchanged code.")),
    )

    print("\nheadline: " + out["headline"])

    if DRY_RUN:
        if DRY_DUMP:
            json.dump(out, open(DRY_DUMP, "w"), indent=1, default=lambda o: (
                o.tolist() if isinstance(o, np.ndarray) else
                (o.item() if isinstance(o, np.generic) else str(o))))
            print(f"DRY_RUN: would-be JSON dumped to {DRY_DUMP} (outside the "
                  "project tree)")
        print("\nDRY_RUN: nothing written.")
        return

    outpath = os.path.join(BASE, "results",
                           "31_internal_sib_worst_case_tgtfda.json")
    if os.path.exists(outpath):
        fail(f"{outpath} already exists; this job writes exactly one new file")

    def enc(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")

    def clean(o):
        """NaN is pandas' way of saying 'no value'; JSON says null. A
        non-finite value that is NOT a missing value still raises."""
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.floating):
            return clean(float(o))
        if isinstance(o, float) and not np.isfinite(o):
            if np.isnan(o):
                return None
            fail(f"non-finite, non-missing value in the output: {o}")
        return o

    tmp = outpath + ".new"
    with open(tmp, "w") as fh:
        json.dump(clean(out), fh, indent=1, default=enc, allow_nan=False)
    os.replace(tmp, outpath)
    print(f"\nwrote {outpath}")
    print(f"  sha256 {sha256_file(outpath)}")


def resolve_all_rows():
    """Every row of this job's grid, as a dict keyed by (method, arch, city).

    Exported so a downstream analysis reads the SAME rows this JSON reports
    rather than resolving them again with its own rules. CONTRACT.md section 6
    makes a results JSON the CS boundary, not the analysis boundary; two
    analyses on this box sharing one row resolver is the only way their cell
    values can be asserted identical, which analysis/28 does.

    Runs the full validation of every cell read: CONTRACT.md section 3 column
    presence and dtype, the 150-image test-pool size, and the recomputed metric
    against the stored columns.
    """
    rows = {}
    for method in METHOD_ORDER:
        for arch in ARCHS:
            for city in CITIES:
                rows[(method, arch, city)] = resolve_row(method, arch, city)
    for method in ("upper", "vanilla", "fda", "sib", "bsm"):
        for city in CITIES:
            rows[(method, CE, city)] = resolve_row(method, CE, city)
    return rows


if __name__ == "__main__":
    main()
