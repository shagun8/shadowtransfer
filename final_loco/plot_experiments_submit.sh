#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# Submit script for plot_experiments.sh
#
# ============================================================
# SERVER PATHS — uncomment the block for your server
# ============================================================

# --- Gilbreth ---
# LOG_DIR="${PROJECT_ROOT}/data/loco_diagnostic_results/logs"

# --- NCSA Delta ---
LOG_DIR="${PROJECT_ROOT}/data/loco_diagnostic_results/logs"

mkdir -p "${LOG_DIR}"

name="plot_experiments"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       plot_experiments.sh

echo "Submitted: ${name}"
echo "Log: ${outputfile}"