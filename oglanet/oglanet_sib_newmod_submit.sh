#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_sib_newmod_submit.sh
#
# Queues OGLANet+SIB diagnostic module ablation jobs on SLURM.
#
# All experiments use the C4/M1 base config as the foundation:
#   Haar + VIB + Aug + AB + aFDA(0.005) + SAG + Ctr
#
# NEW MODULE EXPERIMENTS:
#
# N0:  C4 baseline (no new modules) — control for fair comparison
# N1:  C4 + CACR only
# N2:  C4 + CE-AURC only
# N3:  C4 + CACR + CE-AURC
# N4:  C4 + TENT only (test-time, no training change)
# N5:  C4 + CACR + CE-AURC + TENT (full stack)
#
# CACR SENSITIVITY:
# N6:  C4 + CACR (weight=0.05)  — lower CACR weight
# N7:  C4 + CACR (weight=0.5)   — higher CACR weight
# N8:  C4 + CACR (neg_weight=0.1) — also penalize background shift
#
# CE-AURC SENSITIVITY:
# N9:  C4 + CE-AURC (weight=0.05)  — higher CE-AURC weight
# N10: C4 + CE-AURC (weight=0.001) — lower CE-AURC weight
#
# TENT SENSITIVITY:
# N11: C4 + TENT (steps=3)  — more adaptation steps
# N12: C4 + TENT (steps=5)  — aggressive adaptation
#
# Each config: 3 folds × 1 resolution = 3 SLURM jobs
#
# Decision tree:
#   1. N0 vs N1 → Does CACR help mIoU? Does it reduce SP gap?
#   2. N0 vs N2 → Does CE-AURC improve calibration on shadow pixels?
#   3. N3 vs N1/N2 → Do CACR + CE-AURC compose or interfere?
#   4. N4 vs N0 → Does TENT alone help at test time?
#   5. N5 vs N3 → Does TENT on top of training-time modules add value?
#   6. N6/N7/N8 → CACR weight sensitivity
#   7. N9/N10 → CE-AURC weight sensitivity
#   8. N11/N12 → TENT steps sensitivity
#
# Test runs: override EPOCHS / EARLY_STOPPING_PATIENCE from the command line:
#   EPOCHS=5 EARLY_STOPPING_PATIENCE=10 ./oglanet_sib_newmod_submit.sh --dry-run

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
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OGLANET_OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
LOG_DIR="${BASE_PATH}/data/oglanet/sib_newmod_logs"
BOUNDARY_TOLERANCE=2

# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
FOLD_NAMES=("phoenix" "miami" "chicago")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults: epochs and patience can be overridden from the command line
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS=${EPOCHS:-100}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-25}

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
    esac
done

# =====================================================================
# Helper: submit one configuration across all folds
#
# Args (positional, same shape as MAMNet's submit script):
#   $1   TAG              — experiment tag for naming
#   $2   USE_HAAR         — 0 or 1
#   $3   USE_VIB          — 0 or 1
#   $4   USE_CONTENT_AUG  — 0 or 1
#   $5   ADAPTIVE_BETA    — 0 or 1
#   $6   USE_FDA          — 0 or 1
#   $7   USE_SAG          — 0 or 1
#   $8   USE_MULTISCALE   — 0 or 1
#   $9   USE_CONTRAST     — 0 or 1
#   $10  FDA_L            — float
#   $11  BETA_CONTENT     — float (empty = default)
#   $12  BOUNDARY_TOLERANCE
#   $13  USE_PASSTHROUGH_GATE — 0 or 1
#   $14  USE_MODULE_BYPASS    — 0 or 1
#   $15  BETA_MAX_MULTIPLIER  — float (empty = default)
#   $16  USE_CACR             — 0 or 1
#   $17  CACR_WEIGHT          — float (empty = default 0.1)
#   $18  CACR_NEG_WEIGHT      — float (empty = default 0.0)
#   $19  USE_CE_AURC          — 0 or 1
#   $20  CE_AURC_WEIGHT       — float (empty = default 0.01)
#   $21  USE_TENT             — 0 or 1
#   $22  TENT_STEPS           — int (empty = default 1)
#   $23  TENT_LR              — float (empty = default 0.001)
# =====================================================================
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
    local L_CACR=${16}
    local L_CACR_W=${17}
    local L_CACR_NW=${18}
    local L_CE_AURC=${19}
    local L_CE_AURC_W=${20}
    local L_TENT=${21}
    local L_TENT_S=${22}
    local L_TENT_LR=${23}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  SIB:  HAAR=${HAAR} VIB=${VIB} AUG=${AUG} AB=${AB} FDA=${FDA} SAG=${SAG}"
    echo "  NEW:  CACR=${L_CACR}(w=${L_CACR_W:-def}) CE_AURC=${L_CE_AURC}(w=${L_CE_AURC_W:-def}) TENT=${L_TENT}(s=${L_TENT_S:-def})"

    for fold_id in 1; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="oglanet_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"

            # FDA target root = holdout city images (unlabeled)
            fda_tgt="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            if [ "${DRY_RUN}" -eq 1 ]; then
                echo "    [DRY RUN] name=${name}"
                continue
            fi

            mkdir -p "${LOG_DIR}"

            sbatch \
                --output="${outfile}" \
                --job-name="${name:0:60}" \
                --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR}/${name},FDA_TARGET_ROOT=${fda_tgt},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},OGLANET_OUTPUT_DIR=${OGLANET_OUTPUT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},USE_SAG=${SAG},USE_MULTISCALE_SIB=${MS},USE_CONTRAST=${CTR},FDA_L=${FL},BETA_CONTENT=${BC},BOUNDARY_TOLERANCE=${BT},USE_PASSTHROUGH_GATE=${GATE},USE_MODULE_BYPASS=${MOD_BYPASS},BETA_MAX_MULTIPLIER=${BMM},USE_CACR=${L_CACR},CACR_WEIGHT=${L_CACR_W},CACR_NEG_WEIGHT=${L_CACR_NW},USE_CE_AURC=${L_CE_AURC},CE_AURC_WEIGHT=${L_CE_AURC_W},USE_TENT=${L_TENT},TENT_STEPS=${L_TENT_S},TENT_LR=${L_TENT_LR},EPOCHS=${EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE} \
                oglanet_sib_newmod.sh
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  OGLANet + SIB — Diagnostic Module Ablation Submission"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "========================================================"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Logs:    ${LOG_DIR}/oglanet_sib_N*.out"
echo "  Epochs:  ${EPOCHS}    Patience: ${EARLY_STOPPING_PATIENCE}"
echo ""

# ── C4 BASE CONFIG ──────────────────────────────────────────────────────────
# All experiments below use identical SIB settings (C4/M1):
#   HAAR=1  VIB=1  AUG=1  AB=1  FDA=1  SAG=1  MS=0  CTR=1
#   FDA_L=0.005  GATE=0  MOD_BYPASS=0
# Only the new diagnostic modules (CACR, CE-AURC, TENT) vary.
#
# submit_loco args:
#   TAG  HAAR VIB AUG AB FDA SAG MS CTR  FL     BC BT GATE BYPASS BMM
#   CACR CACR_W CACR_NW  CE_AURC CE_AURC_W  TENT TENT_S TENT_LR
# ─────────────────────────────────────────────────────────────────────────────

# # =====================================================================
# # N0: C4 baseline — no new modules (control)
# # =====================================================================
# submit_loco "N0_c4_baseline" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 0 "" ""  0 ""  0 "" ""

# # =====================================================================
# # N1: C4 + CACR only (weight=0.1, default)
# #   Tests: Does CACR reduce the class-asymmetric SP gap?
# #   Ablation prediction: removing CACR increases gt_shadow AURC gap
# # =====================================================================
# submit_loco "N1_c4_cacr" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 1 "0.1" ""  0 ""  0 "" ""

# # =====================================================================
# # N2: C4 + CE-AURC only (weight=0.01, default)
# #   Tests: Does CE-AURC improve calibration on shadow pixels?
# #   Ablation prediction: removing CE-AURC increases CE-AURC gap > 0/1 gap
# # =====================================================================
# submit_loco "N2_c4_ceaurc" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 0 "" ""  1 "0.01"  0 "" ""

# # =====================================================================
# # N3: C4 + CACR + CE-AURC (both training-time modules)
# #   Tests: Do they compose or interfere?
# # =====================================================================
# submit_loco "N3_c4_cacr_ceaurc" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 1 "0.1" ""  1 "0.01"  0 "" ""

# # =====================================================================
# # N4: C4 + TENT only (test-time adaptation, 1 step)
# #   Tests: Does TENT alone help? (no training change)
# #   §4.4 prediction: OGLANet has encoder-locus failure; TENT on BN
# #   affine may help recover some of it without target labels.
# # =====================================================================
# submit_loco "N4_c4_tent" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 0 "" ""  0 ""  1 "1" "0.001"

# # =====================================================================
# # N5: C4 + CACR + CE-AURC + TENT (full stack)
# #   Tests: All three modules combined
# # =====================================================================
# submit_loco "N5_c4_cacr_ceaurc_tent" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 1 "0.1" ""  1 "0.01"  1 "1" "0.001"

# =====================================================================
# N6: CACR sensitivity — lower weight (0.05)
# =====================================================================
submit_loco "N6_c4_cacr_low" \
    1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    1 "0.05" ""  0 ""  0 "" ""

# # =====================================================================
# # N7: CACR sensitivity — higher weight (0.5)
# # =====================================================================
# submit_loco "N7_c4_cacr_high" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 1 "0.5" ""  0 ""  0 "" ""

# # =====================================================================
# # N8: CACR with background penalty (neg_weight=0.1)
# #   Tests: Does also penalizing background logit shift help?
# # =====================================================================
# submit_loco "N8_c4_cacr_negpen" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 1 "0.1" "0.1"  0 ""  0 "" ""

# # =====================================================================
# # N9: CE-AURC sensitivity — higher weight (0.05)
# # =====================================================================
# submit_loco "N9_c4_ceaurc_high" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 0 "" ""  1 "0.05"  0 "" ""

# # =====================================================================
# # N10: CE-AURC sensitivity — lower weight (0.001)
# # =====================================================================
# submit_loco "N10_c4_ceaurc_low" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 0 "" ""  1 "0.001"  0 "" ""

# # =====================================================================
# # N11: TENT sensitivity — 3 steps
# # =====================================================================
# submit_loco "N11_c4_tent_3step" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 0 "" ""  0 ""  1 "3" "0.001"

# # =====================================================================
# # N12: TENT sensitivity — 5 steps (aggressive)
# # =====================================================================
# submit_loco "N12_c4_tent_5step" \
    # 1 1 1 1 1 1 0 1  0.005 "" ${BOUNDARY_TOLERANCE} 0 0 "" \
    # 0 "" ""  0 ""  1 "5" "0.001"

echo ""
echo "========================================================"
echo "  Summary: ${DRY_RUN:+DRY RUN — }13 configs × 3 folds = 39 jobs"
echo "========================================================"
echo ""
echo "  Core experiments:"
echo "    N0:  C4 baseline (control)"
echo "    N1:  C4 + CACR (class-asymmetric confidence regularizer)"
echo "    N2:  C4 + CE-AURC (calibration-aware auxiliary loss)"
echo "    N3:  C4 + CACR + CE-AURC (combined training-time)"
echo "    N4:  C4 + TENT (test-time adaptation only)"
echo "    N5:  C4 + CACR + CE-AURC + TENT (full stack)"
echo ""
echo "  Sensitivity:"
echo "    N6:  CACR weight=0.05  |  N7: CACR weight=0.5"
echo "    N8:  CACR neg_weight=0.1"
echo "    N9:  CE-AURC weight=0.05  |  N10: CE-AURC weight=0.001"
echo "    N11: TENT steps=3  |  N12: TENT steps=5"
echo ""
echo "  Decision tree:"
echo "    1. Compare N0 vs N1/N2/N3 on mIoU + SP gap metrics"
echo "    2. If N1 helps: check N6/N7/N8 for optimal CACR weight"
echo "    3. If N2 helps: check N9/N10 for optimal CE-AURC weight"
echo "    4. If N4 helps: check N11/N12 for TENT step sensitivity"
echo "    5. N5 = best training combo + TENT → final method candidate"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Logs:     tail -f ${LOG_DIR}/oglanet_sib_N*.out"
echo "  Results:  ls ${OUTPUT_DIR}/oglanet_sib_N*/comparison_results.json"
echo ""