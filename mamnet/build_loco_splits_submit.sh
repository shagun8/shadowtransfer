#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: build_loco_splits_submit.sh
#
# Submits one sbatch job per (fold, resolution) for materializing the
# LOCO splits. Total jobs = 3 folds x N resolutions.
#
# Each job writes a self-contained subtree under OUTPUT_ROOT (real PNG
# copies plus a manifest.json and metadata_{train,val,test}.json), so
# parallel jobs don't share any output paths.

# ---- Server paths (uncomment the one you need) ----
# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test"
OUTPUT_ROOT="${BASE_PATH}/data/Final_data_loco"
LOG_DIR="${BASE_PATH}/data/mamnet"

# Defaults match the paper:
#   - 225 train + 75 val per training city  (450 train, 150 val total)
#   - test = held-out city's full 150-image test pool
TRANSFER_MODE="copy"          # copy | symlink | hardlink
N_TRAIN_PER_CITY=225
N_VAL_PER_CITY=75
NO_MULTICLASS=0               # 1 to skip masks_multiclass/
SEED=42

RESOLUTIONS=("highres" "midres")
FOLD_NAMES=("phoenix" "miami" "chicago")    # fold_id -> holdout city

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_ROOT}"

echo "Queueing LOCO split-build jobs..."
echo "  BASE_DATA_ROOT : ${BASE_DATA_ROOT}"
echo "  OUTPUT_ROOT    : ${OUTPUT_ROOT}"
echo "  RESOLUTIONS    : ${RESOLUTIONS[*]}"
echo "  FOLDS          : 0 1 2  (holdouts: ${FOLD_NAMES[*]})"
echo "  TRANSFER_MODE  : ${TRANSFER_MODE}"
echo "  MULTICLASS     : $([ "${NO_MULTICLASS}" == "1" ] && echo "OFF" || echo "ON")"
echo

for fold_id in 0 1 2
do
    holdout_city="${FOLD_NAMES[$fold_id]}"
    for res in "${RESOLUTIONS[@]}"
    do
        name="build_loco__holdout_${holdout_city}__${res}"
        outputfile="${LOG_DIR}/${name}.out"
        echo "  - fold ${fold_id} (holdout: ${holdout_city}), ${res}"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},FOLD_ID=${fold_id},RESOLUTION=${res},TRANSFER_MODE=${TRANSFER_MODE},N_TRAIN_PER_CITY=${N_TRAIN_PER_CITY},N_VAL_PER_CITY=${N_VAL_PER_CITY},NO_MULTICLASS=${NO_MULTICLASS},SEED=${SEED} \
               build_loco_splits.sh
    done
done

echo
echo "All build jobs queued (3 folds x ${#RESOLUTIONS[@]} resolutions = $((3 * ${#RESOLUTIONS[@]})) jobs)."