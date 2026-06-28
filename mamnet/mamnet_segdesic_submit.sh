#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_segdesic_submit.sh
#
# Queues MAMNet+SegDesic LOCO training jobs on SLURM.
# Uncomment the server block you need.

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
GEO_METADATA_PATH="${BASE_PATH}/data/Final_data_test/metadata/mapping_segdesic.json"

# ---- Boundary-tolerant zone half-width in pixels ----
# Controls the ±Kpx don't-care band in DetailedEvaluator.
# Set to 0 to disable tolerance (strict only). Default: 5.
BOUNDARY_TOLERANCE=2

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
        name="segdesic_mamnet__loco_holdout_${holdout_city}__${res}"
        outfile="${BASE_PATH}/data/mamnet/${name}.out"
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}  boundary_tolerance=${BOUNDARY_TOLERANCE}"
        sbatch --output=${outfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},GEO_METADATA_PATH=${GEO_METADATA_PATH},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE} \
               mamnet_segdesic.sh
    done
done
echo "All jobs queued!"