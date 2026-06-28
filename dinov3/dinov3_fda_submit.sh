#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_fda_submit.sh
#
# Queues DINOv3 + FDA LOCO training jobs on SLURM.
# FDA adapts source training images toward the holdout city's style.
#
# Total: 6 jobs (3 folds x 2 res)

# ---- Cluster paths (uncomment ONE block) ----

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

# ---- Derived paths (no changes needed) ----
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# Train LOCO + FDA models (6 jobs)
# ============================================================
echo "Queueing LOCO + FDA models..."

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout="${FOLD_NAMES[$fold_id]}"
        name="fda_dinov3__loco_holdout_${holdout}__${res}"
        outfile="${BASE_PATH}/data/dinov3/${name}.out"
        TARGET_CITY_ROOT="${BASE_DATA_ROOT}${holdout}/${res}/train/images"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout}) ${res}"
        sbatch --output=${outfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},WEIGHT_DIR=${WEIGHT_DIR},TARGET_CITY_ROOT=${TARGET_CITY_ROOT},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               dinov3_fda.sh
    done
done

echo ""
echo "All jobs queued!"