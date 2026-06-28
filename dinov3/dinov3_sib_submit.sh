#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_sib_submit.sh
#
# Queues DINOv3+SIB training jobs on SLURM.
#
# Round 1 (D1-D6): 6 configurations — original ablation study
# Round 2 (D7-D10): 4 configurations — fix DINOv3 Phoenix regression
# Round 3 (D11): Module-level residual bypass gate on C4
#
# D11: Module bypass gate wrapping entire SIB pipeline
#      C4 base (Haar + VIB + Aug + AB) + learned per-sample α gate
#      F_out = α · F_sib + (1−α) · F_encoder
#      α = sigmoid(linear(GAP(F_encoder))), bias init = +2.0
#      Expected: Phoenix α → 0 (bypass), Chicago/Miami α → 1 (full SIB)

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
LOG_DIR="${BASE_PATH}/data/dinov3/sib_logs"

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

# =====================================================================
# Helper: submit one SIB configuration across all folds and resolutions
#
# Args:
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
#   $14  USE_PASSTHROUGH_GATE — 0 or 1  (optional, defaults to 0)
#   $15  EXP_TAG              — string  (optional)
#   $16  USE_MODULE_BYPASS    — 0 or 1  (optional, defaults to 0)
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
    local GATE=${14:-0}
    local EXP_TAG=${15:-}
    local MOD_BYPASS=${16:-0}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}  GATE=${GATE}  MOD_BYPASS=${MOD_BYPASS}"
    echo "  beta_content=${BC}  beta_edge=${BE}  beta_scale=${BS}"
    echo "  lambda_content=${LC}  lambda_edge=${LE}  warmup=${WF}"
    echo "  boundary_tolerance=±${BOUNDARY_TOLERANCE}px"
    echo "  EXP_TAG=${EXP_TAG:-none}"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="dinov3_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"

            # FDA target = held-out test city images (the domain we adapt toward)
            FDA_TARGET_ROOT="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"
            [ "${FDA}" == "1" ] && echo "    FDA_TARGET_ROOT=${FDA_TARGET_ROOT}"

            sbatch --output=${outfile} \
                   --job-name=${name} \
                   --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},WEIGHT_DIR=${WEIGHT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},FDA_TARGET_ROOT=${FDA_TARGET_ROOT},VIB_BETA_CONTENT=${BC},VIB_BETA_EDGE=${BE},VIB_BETA_SCALE=${BS},LAMBDA_CONTENT=${LC},LAMBDA_EDGE=${LE},VIB_WARMUP_FRAC=${WF},FDA_L=${FL},BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},DINOV3_OUTPUT_DIR=${DINOV3_OUTPUT_DIR},USE_PASSTHROUGH_GATE=${GATE},EXP_TAG=${EXP_TAG},USE_MODULE_BYPASS=${MOD_BYPASS} \
                   dinov3_sib.sh
        done
    done
}

echo "========================================"
echo "  DINOv3 + SIB — Experiment Submission"
echo "  Round 3: Module Bypass Gate"
echo "========================================"
echo "  Output:              ${OUTPUT_DIR}"
echo "  Log dir:             ${LOG_DIR}"
echo "  Weights:             ${WEIGHT_DIR}"
echo "  Donor dir:           ${DINOV3_OUTPUT_DIR}"
echo "  Boundary tolerance:  ±${BOUNDARY_TOLERANCE}px"
echo ""

# # =====================================================================
# # Round 1 experiments (D1-D6) — all completed
# # =====================================================================

# # D1: SIB-full — Haar + DiffVIB + ContentAug + AdaptiveBeta
# submit_loco "D1_full" 1 1 1 1 0 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D1" 0

# # D2: SIB-noAug — Haar + DiffVIB + AdaptiveBeta (no augmentation)
# submit_loco "D2_noAug" 1 1 0 1 0 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D2" 0

# # D3: SIB-uniformVIB — no Haar, VIB on all features + Aug + AdaptiveBeta
# submit_loco "D3_uniformVIB" 0 1 1 1 0 0.001 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D3" 0

# # D4: SIB-fixedBeta — Haar + DiffVIB + Aug + Fixed beta=0.01
# submit_loco "D4_fixedBeta" 1 1 1 0 0 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D4" 0

# # D5: SIB+FDA — Full SIB + FDA (β_FDA=0.01)
# submit_loco "D5_withFDA" 1 1 1 1 1 0.01 0.0001 0.02 1.0 0.1 0.1 0.005 0 "D5" 0

# # D6: VIB-only — no Haar, no Aug, uniform VIB, fixed beta=0.005
# submit_loco "D6_VIBonly" 0 1 0 0 0 0.005 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D6" 0

# # =====================================================================
# # Round 2 experiments (D7-D10) — all completed
# # =====================================================================

# # D7: Low beta (β=0.002) — 5× lower than default 0.01
# submit_loco "D7_lowBeta" 1 1 1 0 0 0.002 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D7" 0

# # D8: Aug only (NO VIB) — Haar + ContentAug, no compression
# submit_loco "D8_augOnly" 1 0 1 0 0 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D8" 0

# # D9: Passthrough gate — Haar + VIB + Aug + learned gate
# submit_loco "D9_gate" 1 1 1 0 0 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 1 "D9" 0

# # D10: Intermediate beta (β=0.005) — 2× lower than default 0.01
# submit_loco "D10_midBeta" 1 1 1 0 0 0.005 0.0001 0.02 1.0 0.1 0.1 0.01 0 "D10" 0

# # =====================================================================
# # Round 3: Module Bypass Gate (D11)
# #
# # Base = C4 = D1: Haar + VIB + Aug + AdaptiveBeta
# # Added: Module-level residual bypass gate wrapping entire SIB pipeline
# #   F_out = α · F_sib + (1−α) · F_encoder  (post-projection)
# #   α = sigmoid(linear(GAP(F_encoder))), bias init = +2.0 → α ≈ 0.88
# #
# # Expected behaviour:
# #   Phoenix holdout → α converges near 0 (bypass, ≈ vanilla)
# #   Chicago/Miami   → α stays near 1 (full SIB applied)
# #
# # Diagnostic: bypass_gate_alpha.json saved per test image
# #
# #   TAG            HAAR VIB AUG AB  FDA BC    BE      BS   LC  LE  WF  FL    GATE EXP   BYPASS
# # =====================================================================
# submit_loco "D11_bypass" \
            # 1    1   1   1   0   0.01  0.0001  0.02 1.0 0.1 0.1 0.01  0    "D11" 1
			
# C4-clean: Haar + VIB (no aug, no AB) — base for tempscale
# DINOv3 has no SAG/FDA in the existing submit signature; SIB on DINOv3
# is just Haar + per-band VIB on encoder features (§4.2 component).
#   TAG               HAAR VIB AUG AB FDA BC    BE     BS   LC  LE  WF  FL    GATE EXP        BYPASS
submit_loco "C4clean_haar_vib" 1 1 0 0 0 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 0    "C4clean" 0

echo ""
echo "========================================"
echo "  Round 3: 3 jobs queued (1 config × 3 folds)"
echo "========================================"
echo ""
echo "  D11 (module bypass) — C4 + learned per-sample bypass gate"
echo "    Base: Haar + VIB + Aug + AdaptiveBeta (= D1)"
echo "    Added: ModuleBypassGate wrapping SIB post-projection"
echo "    Gate init: bias=+2.0 → α≈0.88 (SIB on by default)"
echo ""
echo "  What to check after runs:"
echo "    cat <output_dir>/bypass_gate_alpha.json | python -m json.tool | head -10"
echo "    Expected: Phoenix → low mean α; Chicago/Miami → high mean α"
echo ""