#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_sib_newmod_submit.sh
#
# Queues DINOv3 + SIB + diagnostic-motivated module experiments
# (CACR, CE-AURC, TENT) on SLURM.
#
# Base configuration: D1 (best DINOv3 SIB config from Round 1)
#   HAAR=1 VIB=1 AUG=1 AB=1 FDA=0 GATE=0 MOD_BYPASS=0
#   BC=0.01 (vib_beta_content)
#   BE=0.0001 (vib_beta_edge)
#   BS=0.02 (vib_beta_scale)
#   LC=1.0 (lambda_content)
#   LE=0.1 (lambda_edge)
#   WF=0.1 (vib_warmup_fraction)
#
# Experiment matrix (N0-N12), 13 configs × 3 folds = 39 jobs:
#   N0:  Baseline control (D1 alone, no new modules)
#   N1:  CACR only (w=0.1)
#   N2:  CE-AURC only (w=0.01)
#   N3:  CACR + CE-AURC (training-only stack)
#   N4:  TENT only (steps=1, lr=0.001)
#   N5:  Full stack: CACR + CE-AURC + TENT
#   --- CACR sensitivity ---
#   N6:  CACR w=0.05 (gentler)
#   N7:  CACR w=0.5  (stronger)
#   N8:  CACR w=0.1, neg_weight=0.1 (penalize bg too)
#   --- CE-AURC sensitivity ---
#   N9:  CE-AURC w=0.05
#   N10: CE-AURC w=0.001
#   --- TENT sensitivity ---
#   N11: TENT steps=3
#   N12: TENT steps=5
#
# Test runs:
#   EPOCHS=5 EARLY_STOPPING_PATIENCE=10 ./dinov3_sib_newmod_submit.sh
#
# Dry run (print sbatch commands without submitting):
#   ./dinov3_sib_newmod_submit.sh --dry-run

# =====================================================================
# Server paths — uncomment the block for your target server
# =====================================================================

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

# =====================================================================
# Derived paths (shared across all servers — do not edit)
# =====================================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
DINOV3_OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
LOG_DIR="${BASE_PATH}/data/dinov3/sib_newmod_logs"

# --- Gilbreth ---
# WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# --- Anvil ---
# WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# --- NCSA Delta ---
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Boundary tolerance in pixels
BOUNDARY_TOLERANCE=2

mkdir -p ${LOG_DIR}

FOLD_NAMES=("phoenix" "miami" "chicago")

# Defaults (overridable via env)
EPOCHS=${EPOCHS:-100}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-25}

# Dry-run flag
DRY_RUN=0
for arg in "$@"; do
    if [ "${arg}" == "--dry-run" ]; then
        DRY_RUN=1
    fi
done

# =====================================================================
# Helper: submit one SIB+NewMod configuration across all folds
#
# Args (positional):
#   $1   TAG                  — experiment tag for naming
#   $2   USE_HAAR             — 0 or 1
#   $3   USE_VIB              — 0 or 1
#   $4   USE_CONTENT_AUG      — 0 or 1
#   $5   ADAPTIVE_BETA        — 0 or 1
#   $6   USE_FDA              — 0 or 1
#   $7   VIB_BETA_CONTENT     — float
#   $8   VIB_BETA_EDGE        — float   (only when Haar=1)
#   $9   VIB_BETA_SCALE       — float   (only when AdaptiveBeta=1)
#   $10  LAMBDA_CONTENT       — float
#   $11  LAMBDA_EDGE          — float
#   $12  VIB_WARMUP_FRAC      — float
#   $13  FDA_L                — float   (only when FDA=1)
#   $14  USE_PASSTHROUGH_GATE — 0 or 1
#   $15  EXP_TAG              — string
#   $16  USE_MODULE_BYPASS    — 0 or 1
#   $17  USE_CACR             — 0 or 1
#   $18  CACR_WEIGHT          — float
#   $19  CACR_NEG_WEIGHT      — float
#   $20  USE_CE_AURC          — 0 or 1
#   $21  CE_AURC_WEIGHT       — float
#   $22  USE_TENT             — 0 or 1
#   $23  TENT_STEPS           — int
#   $24  TENT_LR              — float
# =====================================================================
submit_loco() {
    local TAG=$1
    local HAAR=$2
    local VIB=$3
    local AUG=$4
    local AB=$5
    local FDA=$6
    local BC=$7
    local BE=$8
    local BS=$9
    local LC=${10}
    local LE=${11}
    local WF=${12}
    local FL=${13}
    local GATE=${14}
    local EXP_TAG=${15}
    local MOD_BYPASS=${16}
    local USE_CACR=${17}
    local CACR_WEIGHT=${18}
    local CACR_NEG_WEIGHT=${19}
    local USE_CE_AURC=${20}
    local CE_AURC_WEIGHT=${21}
    local USE_TENT=${22}
    local TENT_STEPS=${23}
    local TENT_LR=${24}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}  GATE=${GATE}  MOD_BYPASS=${MOD_BYPASS}"
    echo "  beta_content=${BC}  beta_edge=${BE}  beta_scale=${BS}"
    echo "  lambda_content=${LC}  lambda_edge=${LE}  warmup=${WF}"
    echo "  CACR=${USE_CACR} (w=${CACR_WEIGHT}, neg_w=${CACR_NEG_WEIGHT})"
    echo "  CE-AURC=${USE_CE_AURC} (w=${CE_AURC_WEIGHT})"
    echo "  TENT=${USE_TENT} (steps=${TENT_STEPS}, lr=${TENT_LR})"
    echo "  EPOCHS=${EPOCHS}  PATIENCE=${EARLY_STOPPING_PATIENCE}"
    echo "  EXP_TAG=${EXP_TAG}"

    for fold_id in 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="dinov3_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"

            # FDA target = held-out test city images (the domain we adapt toward)
            FDA_TARGET_ROOT="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            local EXPORT_VARS="MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},WEIGHT_DIR=${WEIGHT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},FDA_TARGET_ROOT=${FDA_TARGET_ROOT},VIB_BETA_CONTENT=${BC},VIB_BETA_EDGE=${BE},VIB_BETA_SCALE=${BS},LAMBDA_CONTENT=${LC},LAMBDA_EDGE=${LE},VIB_WARMUP_FRAC=${WF},FDA_L=${FL},BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},DINOV3_OUTPUT_DIR=${DINOV3_OUTPUT_DIR},USE_PASSTHROUGH_GATE=${GATE},EXP_TAG=${EXP_TAG},USE_MODULE_BYPASS=${MOD_BYPASS},USE_CACR=${USE_CACR},CACR_WEIGHT=${CACR_WEIGHT},CACR_NEG_WEIGHT=${CACR_NEG_WEIGHT},USE_CE_AURC=${USE_CE_AURC},CE_AURC_WEIGHT=${CE_AURC_WEIGHT},USE_TENT=${USE_TENT},TENT_STEPS=${TENT_STEPS},TENT_LR=${TENT_LR},EPOCHS=${EPOCHS},EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"

            if [ "${DRY_RUN}" == "1" ]; then
                echo "    [DRY-RUN] sbatch --output=${outfile} --job-name=${name} --export=PROJECT_ROOT=${PROJECT_ROOT},${EXPORT_VARS} dinov3_sib_newmod.sh"
            else
                sbatch --output=${outfile} \
                       --job-name=${name} \
                       --export=PROJECT_ROOT=${PROJECT_ROOT},${EXPORT_VARS} \
                       dinov3_sib_newmod.sh
            fi
        done
    done
}

echo "========================================"
echo "  DINOv3 + SIB + NewMod — Experiment Submission"
echo "  CACR + CE-AURC + TENT (N0-N12)"
echo "========================================"
echo "  Output:              ${OUTPUT_DIR}"
echo "  Log dir:             ${LOG_DIR}"
echo "  Weights:             ${WEIGHT_DIR}"
echo "  Donor dir:           ${DINOV3_OUTPUT_DIR}"
echo "  Boundary tolerance:  ±${BOUNDARY_TOLERANCE}px"
echo "  EPOCHS:              ${EPOCHS}"
echo "  PATIENCE:            ${EARLY_STOPPING_PATIENCE}"
echo "  DRY_RUN:             ${DRY_RUN}"
echo ""
echo "  Base config: D1 = HAAR + VIB + AUG + AB"
echo "    BC=0.01  BE=0.0001  BS=0.02"
echo "    LC=1.0   LE=0.1     WF=0.1"
echo ""

# =====================================================================
# Argument table for submit_loco:
#
# Position 1 :  TAG
# Positions 2-6 :  HAAR VIB AUG AB FDA
# Positions 7-13 :  BC BE BS LC LE WF FL
# Positions 14-16 :  GATE EXP_TAG MOD_BYPASS
# Positions 17-19 :  USE_CACR CACR_WEIGHT CACR_NEG_WEIGHT
# Positions 20-21 :  USE_CE_AURC CE_AURC_WEIGHT
# Positions 22-24 :  USE_TENT TENT_STEPS TENT_LR
# =====================================================================

# # ─────────────────────────────────────────────────────────────────────
# # N0: Baseline control (D1 alone — no diagnostic modules)
# # ─────────────────────────────────────────────────────────────────────
# submit_loco "N0_baseline" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N0" 0 \
            # 0 0.0 0.0 \
            # 0 0.0 \
            # 0 0 0.0

# # ─────────────────────────────────────────────────────────────────────
# # N1: CACR only (w=0.1)
# # ─────────────────────────────────────────────────────────────────────
# submit_loco "N1_cacr" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N1" 0 \
            # 1 0.1 0.0 \
            # 0 0.0 \
            # 0 0 0.0

# ─────────────────────────────────────────────────────────────────────
# N2: CE-AURC only (w=0.01)
# ─────────────────────────────────────────────────────────────────────
submit_loco "N2_ceaurc" \
            1 1 1 1 0 \
            0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            0 "N2" 0 \
            0 0.0 0.0 \
            1 0.01 \
            0 0 0.0

# ─────────────────────────────────────────────────────────────────────
# N3: CACR + CE-AURC (training-only stack)
# ─────────────────────────────────────────────────────────────────────
# submit_loco "N3_cacr_ceaurc" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N3" 0 \
            # 1 0.1 0.0 \
            # 1 0.01 \
            # 0 0 0.0

# ─────────────────────────────────────────────────────────────────────
# N4: TENT only (steps=1, lr=0.001)
# ─────────────────────────────────────────────────────────────────────
submit_loco "N4_tent" \
            1 1 1 1 0 \
            0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            0 "N4" 0 \
            0 0.0 0.0 \
            0 0.0 \
            1 1 0.001

# ─────────────────────────────────────────────────────────────────────
# N5: Full stack — CACR + CE-AURC + TENT
# ─────────────────────────────────────────────────────────────────────
submit_loco "N5_full" \
            1 1 1 1 0 \
            0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            0 "N5" 0 \
            1 0.1 0.0 \
            1 0.01 \
            1 1 0.001

# # ─────────────────────────────────────────────────────────────────────
# # CACR sensitivity
# # ─────────────────────────────────────────────────────────────────────

# # N6: CACR w=0.05 (gentler)
# submit_loco "N6_cacr_w005" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N6" 0 \
            # 1 0.05 0.0 \
            # 0 0.0 \
            # 0 0 0.0

# # N7: CACR w=0.5 (stronger)
# submit_loco "N7_cacr_w050" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N7" 0 \
            # 1 0.5 0.0 \
            # 0 0.0 \
            # 0 0 0.0

# # N8: CACR w=0.1, neg_weight=0.1 (penalize background too)
# submit_loco "N8_cacr_neg" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N8" 0 \
            # 1 0.1 0.1 \
            # 0 0.0 \
            # 0 0 0.0

# # ─────────────────────────────────────────────────────────────────────
# # CE-AURC sensitivity
# # ─────────────────────────────────────────────────────────────────────

# # N9: CE-AURC w=0.05 (stronger)
# submit_loco "N9_ceaurc_w005" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N9" 0 \
            # 0 0.0 0.0 \
            # 1 0.05 \
            # 0 0 0.0

# # N10: CE-AURC w=0.001 (gentler)
# submit_loco "N10_ceaurc_w0001" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N10" 0 \
            # 0 0.0 0.0 \
            # 1 0.001 \
            # 0 0 0.0

# # ─────────────────────────────────────────────────────────────────────
# # TENT sensitivity
# # ─────────────────────────────────────────────────────────────────────

# # N11: TENT steps=3
# submit_loco "N11_tent_s3" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N11" 0 \
            # 0 0.0 0.0 \
            # 0 0.0 \
            # 1 3 0.001

# # N12: TENT steps=5
# submit_loco "N12_tent_s5" \
            # 1 1 1 1 0 \
            # 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 \
            # 0 "N12" 0 \
            # 0 0.0 0.0 \
            # 0 0.0 \
            # 1 5 0.001

# =====================================================================
# Summary
# =====================================================================
echo ""
echo "========================================"
echo "  Queued: 13 configs × 3 folds = 39 jobs"
echo "========================================"
echo ""
echo "  N0:  Baseline control (D1 alone)"
echo "  N1:  CACR only (w=0.1)"
echo "  N2:  CE-AURC only (w=0.01)"
echo "  N3:  CACR + CE-AURC"
echo "  N4:  TENT only (steps=1)"
echo "  N5:  Full stack: CACR + CE-AURC + TENT"
echo "  --- CACR sensitivity ---"
echo "  N6:  CACR w=0.05"
echo "  N7:  CACR w=0.5"
echo "  N8:  CACR w=0.1, neg_w=0.1"
echo "  --- CE-AURC sensitivity ---"
echo "  N9:  CE-AURC w=0.05"
echo "  N10: CE-AURC w=0.001"
echo "  --- TENT sensitivity ---"
echo "  N11: TENT steps=3"
echo "  N12: TENT steps=5"
echo ""
echo "  Decision tree (after N0-N5 complete):"
echo ""
echo "    1. If N0 ≈ baseline (validates pipeline), proceed."
echo ""
echo "    2. Compare N1, N2, N4 vs N0 → which single module helps most?"
echo "       - N1>N0: CACR helps. Tune w via N6/N7. Check N8 (neg_w)."
echo "       - N2>N0: CE-AURC helps. Tune w via N9/N10."
echo "       - N4>N0: TENT helps. Tune steps via N11/N12."
echo ""
echo "    3. Compare N3, N5 vs single-module winners:"
echo "       - N3>N1+N2: CACR and CE-AURC stack additively."
echo "       - N5>N3: TENT stacks on top."
echo "       - N5<N3: TENT may be hurting under domain shift; drop or tune."
echo ""
echo "    4. Phoenix-specific check (mirroring D11 module-bypass story):"
echo "       Phoenix is the bright/regression-prone holdout. Compare"
echo "       fold=0 (Phoenix) vs Miami/Chicago — modules should help"
echo "       more on Phoenix than on the easy folds."
echo ""
echo "  What to check after runs:"
echo "    cat <output_dir>/test_results.json"
echo "    cat <output_dir>/training_history.json | python -c 'import json,sys; h=json.load(sys.stdin); print(h[-1])'"
echo "    For CACR: look at cacr_diag.cacr_pos_shift trend in training_history.json"
echo "    For CE-AURC: look at ce_aurc_diag.ce_aurc_mean_shadow_conf trend"
echo "    For TENT: compare test_results.json with tent_active=true vs N0"
echo ""