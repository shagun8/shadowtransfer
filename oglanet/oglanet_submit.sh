#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_submit.sh
#
# Queues OGLANet training jobs on SLURM.
#
# PART 1 — Individual city models:  single-city training
# PART 2 — LOCO models:             leave-one-city-out folds
#
# Uncomment the parts you need.
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
# PART 1: Train individual city models
# ============================================================
echo "Queueing individual city models..."
for city in chicago miami phoenix
do
    for res in midres
    do
        name="oglanet__${city}__${res}"
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"
        data_root="${BASE_DATA_ROOT}${city}/${res}/"
        echo "  - ${city} ${res} (single mode)"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=single,DATA_ROOT=${data_root},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=2,EARLY_STOPPING_PATIENCE=10 \
               oglanet.sh
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
        name="oglanet__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=2,EARLY_STOPPING_PATIENCE=10 \
               oglanet.sh
    done
done

echo "All jobs queued!"