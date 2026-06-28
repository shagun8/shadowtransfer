#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: submit_loco_eval.sh
# Submit LOCO evaluation jobs for both resolutions in parallel

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

LOG_DIR="${BASE_PATH}/data/Test_img_results/inference_logs/"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

echo "======================================"
echo "Submitting Jobs"
echo "======================================"
echo "Log Dir: ${LOG_DIR}"
echo ""

# Submit jobs for all resolution × fold combinations
FOLD_NAMES=("phoenix" "miami" "chicago")

name="statistical_analysis"
outputfile="${LOG_DIR}/${name}.out"
sbatch --output=${outputfile} \
--job-name=${name} \
runonefile.sh

echo "======================================"
echo "All LOCO evaluation jobs submitted!"
echo "======================================"
echo ""