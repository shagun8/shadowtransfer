#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: split_diag_submit.sh
#
# Queues split distribution diagnostics on SLURM.
# Uncomment the server block you need below.

# ---- Server paths (uncomment the one you need) ----

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"

# ---- Common paths ----
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test"
OUTPUT_DIR="${BASE_PATH}/data/split_diagnostics_output"
GEO_METADATA_PATH="${BASE_PATH}/data/Final_data_test/metadata/mapping.json"
LOG_DIR="${BASE_PATH}/data/split_diagnostics"

# Create output dirs
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# ============================================================
# PART 1: Full run — all cities, both resolutions, all 4 tests
# ============================================================
echo "Queueing full split diagnostics (all cities, both resolutions)..."

name="split_diag__full"
outputfile="${LOG_DIR}/${name}.out"

echo "  - Full diagnostics (highres + midres, all cities)"
echo "    Log: ${outputfile}"

sbatch --output=${outputfile} \
       --job-name=${name} \
       --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR},GEO_METADATA_PATH=${GEO_METADATA_PATH},RESOLUTIONS="highres midres",CITIES="chicago miami phoenix",BATCH_SIZE=32 \
       split_diagnostics.sh

# ============================================================
# PART 2 (optional): Label-only quick run (no GPU needed, ~2 min)
# Uncomment if you want a fast check before the full run.
# ============================================================
# echo ""
# echo "Queueing label-stats-only quick run..."
# 
# name="split_diag__labels_only"
# outputfile="${LOG_DIR}/${name}.out"
# 
# echo "  - Label stats only (no feature extraction)"
# echo "    Log: ${outputfile}"
# 
# sbatch --output=${outputfile} \
#        --job-name=${name} \
#        --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR}_labels_only,GEO_METADATA_PATH=${GEO_METADATA_PATH},RESOLUTIONS="highres midres",CITIES="chicago miami phoenix",SKIP_FEATURES=1,BATCH_SIZE=32 \
#        split_diag.sh

echo ""
echo "Job queued! Check logs at: ${LOG_DIR}/"
echo "Results will appear in: ${OUTPUT_DIR}/"