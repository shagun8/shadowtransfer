#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_submit.sh
#
# Queues MAMNet training jobs on SLURM.
# Uncomment the server block you need below.

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
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

# Don't-care band half-width in pixels.
# DetailedEvaluator always runs with this value.
# When EVAL_TOLERANT=1, also selects which metric drives decisions.
BOUNDARY_TOLERANCE=2

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# PART 1: Train individual city models
# ============================================================
echo "Queueing individual city models..."
for city in miami chicago phoenix
do
    for res in midres
    do
        name="mamnet__${city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet/${name}.out"
        data_root="${BASE_DATA_ROOT}${city}/${res}/"
        echo "  - ${city} ${res} (single mode)"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=single,DATA_ROOT=${data_root},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10,COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               mamnet.sh
    done
done

# ============================================================
# PART 2: Train LOCO models
# ============================================================
echo "Queueing LOCO models..."
# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="mamnet__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet/${name}.out"
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10,COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               mamnet.sh
    done
done

echo "All jobs queued!"