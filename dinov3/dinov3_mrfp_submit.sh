#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_mrfp_submit.sh
#
# Queues DINOv3 + MRFP/MRFP+ LOCO training jobs on SLURM.
# Only LOCO analysis (3 folds × highres).
#
# ---- Server paths (uncomment ONE block) ----

# ==================== Gilbreth ====================
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# ==================== Anvil ====================
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# ==================== NCSA Delta ====================
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

# ---- Derived paths (no changes needed below) ----
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

# ---- Evaluation config ----
# Don't-care band half-width in pixels.
# DetailedEvaluator always runs with this value.
# When EVAL_TOLERANT=1 it also selects the decision metric.
BOUNDARY_TOLERANCE=2
EVAL_TOLERANT=1           # 1 = tolerant mIOU drives decisions; 0 = strict
EARLY_STOPPING_PATIENCE=25

# ---- MRFP configuration ----
# MRFP+ is enabled by default (best variant from paper Table 5).
# Set USE_MRFP_PLUS=0 to use plain MRFP (HRFP+NP+ only, no HRFP+).
USE_MRFP_PLUS=1

# Perturbation probabilities (paper defaults: 0.5 each)
HRFP_PROB=0.5
NP_PROB=0.5
HRFP_PLUS_PROB=0.5
HRFP_BN_STD=0.5

# LOCO fold → held-out city mapping
FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# LOCO models (3 folds × highres)
# ============================================================
echo "========================================"
echo "Queueing DINOv3 + MRFP/MRFP+ LOCO models"
echo "========================================"
echo "  USE_MRFP_PLUS=${USE_MRFP_PLUS}"
echo "  HRFP_PROB=${HRFP_PROB}  NP_PROB=${NP_PROB}"
echo "  HRFP_PLUS_PROB=${HRFP_PLUS_PROB}  BN_STD=${HRFP_BN_STD}"
echo "  BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE}  EVAL_TOLERANT=${EVAL_TOLERANT}"
echo "========================================"

# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
for fold_id in 2
do
    for res in midres
    do
        holdout="${FOLD_NAMES[$fold_id]}"

        if [ "${USE_MRFP_PLUS}" == "1" ]; then
            variant_tag="mrfp_plus"
        else
            variant_tag="mrfp"
        fi

        name="dinov3_${variant_tag}__loco_holdout_${holdout}__${res}"
        outfile="${BASE_PATH}/data/dinov3/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout}) ${res}"

        sbatch --output="${outfile}" \
               --job-name="${name}" \
               --export=PROJECT_ROOT=${PROJECT_ROOT},\
MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
OUTPUT_DIR=${OUTPUT_DIR},\
WEIGHT_DIR=${WEIGHT_DIR},\
EVAL_TOLERANT=${EVAL_TOLERANT},\
BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},\
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},\
USE_MRFP_PLUS=${USE_MRFP_PLUS},\
HRFP_PROB=${HRFP_PROB},\
NP_PROB=${NP_PROB},\
HRFP_PLUS_PROB=${HRFP_PLUS_PROB},\
HRFP_BN_STD=${HRFP_BN_STD},\
COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},\
COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               dinov3_mrfp.sh
    done
done

echo ""
echo "All LOCO jobs queued!"
echo ""
echo "Notes:"
echo "  • Variant: $([ "${USE_MRFP_PLUS}" == "1" ] && echo 'MRFP+' || echo 'MRFP')"
echo "  • Decision metric: $([ "${EVAL_TOLERANT}" == "1" ] && echo "Tolerant ±${BOUNDARY_TOLERANCE}px mIOU" || echo 'Strict per-image mIOU')"
echo "  • Early stopping patience: ${EARLY_STOPPING_PATIENCE} epochs"
echo "  • HRFP/NP+ attachment: feat_block3 (HF) / feat_block11 (LF/style)"