#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_sib_ablation_submit.sh
#
# SIB Component Ablation Experiments (A1–A10) for MAMNet.
#
# All ablations are relative to SIB-Full (C4):
#   C4 = Haar + VIB + Aug + AB + SAG + aFDA(0.005) + Contrast
#   Flags: HAAR=1, VIB=1, AUG=1, AB=1, FDA=1, SAG=1, CTR=1
#
# ═══════════════════════════════════════════════════════════════════════
# Ablation Table (Paper §5.3 / Table 4)
# ═══════════════════════════════════════════════════════════════════════
#
#  ID   Name                      What changes from C4          Diagnostic
#  ──   ────                      ──────────────────────         ──────────
#  A1   No-VIB on F_LL            VIB=0                         D2
#  A2   Uniform-β VIB             AB=0 (fixed β)                D1
#  A3   Symmetric VIB             β_content for LH/HL too       D1+D2
#  A4   No content augmentation   AUG=0                         D3
#  A5   Aug all subbands          Aug on LH/HL/HH (MRFP+ analog) D1+D3
#  A6   No-SAG                    SAG=0                         D2
#  A7   No-FDA-preproc            FDA=0, CTR=0                  confound
#  A8   No-WT (no Haar)           HAAR=0 (uniform VIB)          D2
#  A9   No edge VIB               VIB on LL only, skip LH/HL   D2
#  A10  VIB wrong subband         VIB on HL only (inverse)      D2
#
# ═══════════════════════════════════════════════════════════════════════
# Key predictions (if diagnostics are correct):
#   A1  drops OGLANet > DINOv3 (validates D2 architecture specificity)
#   A2  drops high-intensity decile mIoU > low-intensity (validates D1)
#   A3  collapses thin-shadow F1 (boundary degradation from over-compression)
#   A4  drops DINOv3 > OGLANet (validates D3 decoder robustness)
#   A5  reproduces MRFP+ OGLANet-Miami collapse (validates asymmetric design)
#   A10 worse than C4 everywhere (inverse evidence: wrong subband = wrong fix)
# ═══════════════════════════════════════════════════════════════════════
#
# Total: 10 ablations × 3 folds × 1 resolution = 30 SLURM jobs
#
# Usage:
#   bash mamnet_sib_ablation_submit.sh           # submit all
#   bash mamnet_sib_ablation_submit.sh --dry-run  # preview only

# ---- Server paths (uncomment the one you need) ----

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
MAMNET_OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
LOG_DIR="${BASE_PATH}/data/mamnet/sib_ablation_logs"
BOUNDARY_TOLERANCE=2

# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
FOLD_NAMES=("phoenix" "miami" "chicago")

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
    esac
done

# ═════════════════════════════════════════════════════════════════════════════
# Helper: submit one ablation configuration across all folds
#
# Args:
#   $1   TAG               — experiment tag for naming
#   $2   USE_HAAR           — 0 or 1
#   $3   USE_VIB            — 0 or 1
#   $4   USE_CONTENT_AUG    — 0 or 1
#   $5   ADAPTIVE_BETA      — 0 or 1
#   $6   USE_FDA            — 0 or 1
#   $7   USE_SAG            — 0 or 1
#   $8   USE_MULTISCALE     — 0 or 1
#   $9   USE_CONTRAST       — 0 or 1
#   $10  FDA_L              — float
#   $11  BETA_CONTENT       — float (empty = use script default)
#   $12  BOUNDARY_TOLERANCE — int
#   $13  USE_PASSTHROUGH_GATE — 0 or 1
#   $14  USE_MODULE_BYPASS  — 0 or 1
#   $15  BETA_MAX_MULTIPLIER — float (empty = default)
#   $16  SYMMETRIC_VIB      — 0 or 1
#   $17  AUG_ALL_SUBBANDS   — 0 or 1
#   $18  VIB_WRONG_SUBBAND  — 0 or 1
#   $19  NO_EDGE_VIB        — 0 or 1
# ═════════════════════════════════════════════════════════════════════════════
submit_loco() {
    local TAG=$1
    local HAAR=$2
    local VIB=$3
    local AUG=$4
    local AB=$5
    local FDA=$6
    local SAG=$7
    local MS=$8
    local CTR=$9
    local FL=${10}
    local BC=${11}
    local BT=${12}
    local GATE=${13}
    local MOD_BYPASS=${14}
    local BMM=${15}
    local SYM_VIB=${16}
    local AUG_ALL=${17}
    local VIB_WRONG=${18}
    local NO_EDGE=${19}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}"
    echo "  SAG=${SAG}  MS=${MS}  CTR=${CTR}  FDA_L=${FL}  BETA=${BC:-default}"
    echo "  GATE=${GATE}  MOD_BYPASS=${MOD_BYPASS}  BMM=${BMM:-default}"
    echo "  SYM_VIB=${SYM_VIB}  AUG_ALL=${AUG_ALL}  VIB_WRONG=${VIB_WRONG}  NO_EDGE=${NO_EDGE}"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="mamnet_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"

            # FDA target root = holdout city images (unlabeled)
            fda_tgt="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            if [ "${DRY_RUN}" -eq 1 ]; then
                echo "    [DRY RUN] name=${name}"
                echo "              out=${outfile}"
                continue
            fi

            mkdir -p "${LOG_DIR}"

            sbatch \
                --output="${outfile}" \
                --job-name="${name:0:60}" \
                --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR}/${name},FDA_TARGET_ROOT=${fda_tgt},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},MAMNET_OUTPUT_DIR=${MAMNET_OUTPUT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},USE_SAG=${SAG},USE_MULTISCALE_SIB=${MS},USE_CONTRAST=${CTR},FDA_L=${FL},BETA_CONTENT=${BC},BOUNDARY_TOLERANCE=${BT},USE_PASSTHROUGH_GATE=${GATE},USE_MODULE_BYPASS=${MOD_BYPASS},BETA_MAX_MULTIPLIER=${BMM},SYMMETRIC_VIB=${SYM_VIB},AUG_ALL_SUBBANDS=${AUG_ALL},VIB_WRONG_SUBBAND=${VIB_WRONG},NO_EDGE_VIB=${NO_EDGE} \
                mamnet_sib.sh
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════"
echo "  MAMNet + SIB — Component Ablation Experiments"
echo "  A1–A10 (all relative to C4 = SIB-Full)"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "════════════════════════════════════════════════════"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Logs:    ${LOG_DIR}/mamnet_sib_*.out"
echo ""

# ═════════════════════════════════════════════════════════════════════════════
#                              C4 BASE REFERENCE
# ═════════════════════════════════════════════════════════════════════════════
# C4 (SIB-Full) = M1 from main experiments:
#   HAAR=1 VIB=1 AUG=1 AB=1 FDA=1 SAG=1 MS=0 CTR=1
#   FDA_L=0.005  BC=""  BT=2  GATE=0  MOD=0  BMM=""
#   SYM=0  AUG_ALL=0  VIB_WRONG=0  NO_EDGE=0
#
# Already run as M1. Not re-run here — results serve as ablation baseline.
# ═════════════════════════════════════════════════════════════════════════════

#                        HAAR VIB AUG AB  FDA SAG MS  CTR FDA_L  BC   BT GATE MOD BMM SYM AUG_ALL VIB_WRONG NO_EDGE

# ═════════════════════════════════════════════════════════════════════════════
# A1: No-VIB on F_LL — Remove all VIB compression
#   Tests D2: Is domain compression via VIB necessary?
#   Prediction: OGLANet drops more than DINOv3
#   Change from C4: VIB=0, AB=0 (AB irrelevant without VIB)
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A1_no_vib"                                                    \
            1   0   1   0   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A2: Uniform-β VIB — Fixed β (no intensity conditioning)
#   Tests D1: Does intensity-adaptive compression target the right stratum?
#   Prediction: High-intensity decile mIoU drops more than low-intensity
#   Change from C4: AB=0
#   Note: Equivalent to M11 — re-run for clean ablation naming
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A2_uniform_beta"                                              \
            1   1   1   0   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A3: Symmetric VIB — Apply β_content to edge subbands too
#   Tests D1+D2: Does subband asymmetry matter?
#   Prediction: Thin-shadow F1 collapses; boundary degradation
#   Change from C4: SYMMETRIC_VIB=1
#   NEW FLAG — requires updated sib.py
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A3_symmetric_vib"                                             \
            1   1   1   1   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  1  0  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A4: No content augmentation — Remove noise perturbation on LL
#   Tests D3: Does content augmentation force decoder robustness?
#   Prediction: DINOv3 degrades most (decoder-locus failure)
#   Change from C4: AUG=0
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A4_no_content_aug"                                            \
            1   1   0   1   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A5: Augment all subbands — MRFP+ analog
#   Tests D1+D3: Why SIB's asymmetric augmentation is correct
#   Prediction: Reproduces MRFP+ OGLANet-Miami collapse
#   Change from C4: AUG_ALL_SUBBANDS=1
#   NEW FLAG — requires updated sib.py
#   This is the most rhetorically powerful ablation: "by changing one
#   design choice, we reproduce the failure mode of the best-competing
#   prior method."
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A5_aug_all_subbands"                                          \
            1   1   1   1   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  1  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A6: No-SAG — Remove Skip Attention Gates
#   Tests D2: Do skip connections bypass the bottleneck?
#   Prediction: U-Net-based architectures show encoder-gain leakage
#   Change from C4: SAG=0
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A6_no_sag"                                                    \
            1   1   1   1   1   0   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A7: No-FDA-preproc — Remove FDA + contrast channel
#   Tests confound: Are SIB gains from FDA/contrast, not the module?
#   Prediction: If MAMNet/OGLANet gains come from FDA not SIB, large drop;
#               if SIB is doing the work, drop is small
#   Change from C4: FDA=0, CTR=0
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A7_no_fda_preproc"                                            \
            1   1   1   1   0   1   0   0   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A8: No-WT — Replace Haar decomposition with identity (uniform VIB)
#   Tests D2: Does subband separation itself matter?
#   Note: With HAAR=0, content aug and adaptive β naturally don't apply
#         (they require frequency decomposition). This tests the FULL
#         contribution of Haar: decomposition + content-only aug + adaptive β.
#   Prediction: Worse than A3 — without separation, can't attempt targeting
#   Change from C4: HAAR=0
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A8_no_haar"                                                   \
            0   1   1   1   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  0  0

# ═════════════════════════════════════════════════════════════════════════════
# A9: No edge VIB — VIB on LL only, LH/HL pass through unchanged
#   Tests: Does gentle edge regularization via β_edge matter?
#   Prediction: Small change — edge VIB is gentle (β_edge << β_content)
#               so removing it should have modest effect
#   Change from C4: NO_EDGE_VIB=1
#   NEW FLAG — requires updated sib.py
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A9_no_edge_vib"                                               \
            1   1   1   1   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  0  1

# ═════════════════════════════════════════════════════════════════════════════
# A10: VIB on wrong subband (HL only) — Inverse evidence
#   Tests D2: Is the LL target choice principled or arbitrary?
#   Prediction: Degradation everywhere — LL domain info leaks through,
#               HL task-relevant edges are destroyed by compression
#   Change from C4: VIB_WRONG_SUBBAND=1
#   NEW FLAG — requires updated sib.py
#   Small experiment, large rhetorical payoff: "the design choice isn't
#   arbitrary because the wrong choice is demonstrably worse."
# ═════════════════════════════════════════════════════════════════════════════
submit_loco "A10_vib_wrong_subband"                                        \
            1   1   1   1   1   1   0   1   0.005  ""   ${BOUNDARY_TOLERANCE}  0  0  ""  0  0  1  0

echo ""
echo "════════════════════════════════════════════════════"
echo "  Summary: 10 ablations × 3 folds = 30 SLURM jobs"
echo "════════════════════════════════════════════════════"
echo ""
echo "  Critical ablations (§5.3 Table 4):"
echo "    A1:  No-VIB           — validates D2 (domain compression)"
echo "    A2:  Uniform-β        — validates D1 (intensity targeting)"
echo "    A3:  Symmetric VIB    — validates spectral asymmetry"
echo "    A4:  No content aug   — validates D3 (decoder robustness)"
echo "    A5:  Aug all subbands — reproduces MRFP+ failure mode"
echo ""
echo "  Extended ablations (Appendix):"
echo "    A6:  No-SAG           — skip connection bypass"
echo "    A7:  No-FDA-preproc   — controls for FDA/contrast confound"
echo "    A8:  No-Haar          — full spectral decomposition contribution"
echo "    A9:  No edge VIB      — edge regularization value"
echo "    A10: VIB wrong subband — inverse evidence (wrong target)"
echo ""
echo "  Reference baseline (already run, not re-submitted):"
echo "    C4 = M1: Haar+VIB+Aug+AB+SAG+aFDA+Ctr (SIB-Full)"
echo ""
echo "  Analysis after completion:"
echo "    1. For each Ax, compute per-cell Δ mIoU vs C4/M1"
echo "    2. For A2: report per-intensity-decile mIoU (high vs low)"
echo "    3. For A3: report thin-shadow-category F1"
echo "    4. For A5: check OGLANet-Miami specifically (MRFP+ failure cell)"
echo "    5. For A10: confirm degradation everywhere (inverse evidence)"
echo ""
echo "  Monitor jobs  :  squeue -u \$USER"
echo "  Watch a log   :  tail -f ${LOG_DIR}/mamnet_sib_A*.out"
echo "  Check outputs :  ls ${OUTPUT_DIR}/mamnet_sib_A*/comparison_results.json"
echo ""