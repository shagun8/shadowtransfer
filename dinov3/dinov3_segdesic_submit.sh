#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_segdesic_submit.sh
#
# Queues DINOv3+SegDesic LOCO training jobs on SLURM.
# LOCO: 3 jobs (3 folds x midres)
#
# =====================================================================
# CLUSTER SWITCH — uncomment ONE block, comment the other
# =====================================================================
# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

# =====================================================================
# Derived paths (cluster-agnostic)
# =====================================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
GEO_METADATA_PATH="${BASE_PATH}/data/Final_data_test/metadata/mapping_segdesic.json"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
# =====================================================================
# CHANGED: Evaluation settings — now configurable via env vars
# (previously --eval_boundary_tolerant was hardcoded in dinov3_segdesic.sh
#  and BOUNDARY_TOLERANCE was never passed at all, meaning the DetailedEvaluator
#  always used its own default of 2px while key lookups used hardcoded
#  'tolerant_5px' → KeyError at runtime).
#
# EVAL_TOLERANT=1  → pass --eval_boundary_tolerant to training script
#                    (uses tolerant mIOU for best checkpoint / early stopping)
# BOUNDARY_TOLERANCE=K → pass --boundary_tolerance K to training script
#                    (sets the ±K px don't-care band in DetailedEvaluator)
# =====================================================================
EVAL_TOLERANT=1
BOUNDARY_TOLERANCE=2

FOLD_NAMES=("phoenix" "miami" "chicago")
# ============================================================
# LOCO models (3 folds x midres = 3 jobs)
# ============================================================
echo "Queueing LOCO SegDesic models..."
for fold_id in 0 1 2; do
    for res in midres; do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="segdesic_dinov3__loco_holdout_${holdout_city}__${res}"
        outfile="${BASE_PATH}/data/dinov3/${name}.out"
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        sbatch --output=${outfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},WEIGHT_DIR=${WEIGHT_DIR},GEO_METADATA_PATH=${GEO_METADATA_PATH},EVAL_TOLERANT=${EVAL_TOLERANT},BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               dinov3_segdesic.sh
    done
done
echo ""
echo "All jobs queued!"