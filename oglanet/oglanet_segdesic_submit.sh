#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_segdesic_submit.sh
#
# Queues OGLANet + SegDesic training jobs on SLURM (LOCO folds).
# Uncomment the server paths you need.

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
GEO_METADATA_PATH="${BASE_PATH}/data/Final_data_test/metadata/mapping_segdesic.json"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# LOCO models
# ============================================================
echo "Queueing LOCO models..."

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="segdesic_oglanet__loco_holdout_${holdout_city}__${res}"

        # --- Gilbreth / Anvil ---
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"

        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},GEO_METADATA_PATH=${GEO_METADATA_PATH},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=2,EARLY_STOPPING_PATIENCE=10 \
               oglanet_segdesic.sh
    done
done

echo "All jobs queued!"