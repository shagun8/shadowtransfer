#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_isw_submit.sh
#
# Queues MAMNet + ISW training jobs for LOCO analysis.
# Prerequisites: ISW masks must already be precomputed
#                (run compute_isw_masks_submit.sh first).

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
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
ISW_MASK_BASE_DIR="${BASE_PATH}/data/mamnet/isw_masks"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

BOUNDARY_TOLERANCE=2
ISW_LAMBDA=0.6

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# LOCO models with ISW
# ============================================================
echo "Queueing MAMNet + ISW LOCO models..."

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        mask_dir="${ISW_MASK_BASE_DIR}/loco_holdout_${holdout_city}_${res}"
        name="mamnet_isw__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet/${name}.out"

        # Check that masks exist
        if [ ! -d "${mask_dir}" ]; then
            echo "  WARNING: Mask dir not found: ${mask_dir}"
            echo "           Run compute_isw_masks_submit.sh first!"
            echo "           Skipping fold ${fold_id}."
            continue
        fi

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        echo "    mask dir: ${mask_dir}"

        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},ISW_MASK_DIR=${mask_dir},ISW_LAMBDA=${ISW_LAMBDA},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10,COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               mamnet_isw.sh
    done
done

echo ""
echo "All ISW LOCO training jobs queued!"