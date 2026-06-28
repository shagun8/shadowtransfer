#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_fda_submit.sh
#
# Queues OGLANet+FDA LOCO training jobs on SLURM.
# FDA target domain = holdout city's train images.
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
FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# Train LOCO models with FDA
# ============================================================
echo "Queueing FDA LOCO models..."
# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="fda_oglanet__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"
        TARGET_CITY_ROOT="${BASE_DATA_ROOT}${holdout_city}/${res}/train/images"
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},TARGET_CITY_ROOT=${TARGET_CITY_ROOT},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=2,EARLY_STOPPING_PATIENCE=10 \
               oglanet_fda.sh
    done
done

echo "All jobs queued!"