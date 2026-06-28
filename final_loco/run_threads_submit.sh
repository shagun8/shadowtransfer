#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# Submit the diagnostic analysis job.
#
# All extra arguments are forwarded to run_diagnostics.py via run_threads.sh.
#
# ---- Uncomment the block for your active server ----

# --- Gilbreth ---
# LOG_DIR="${PROJECT_ROOT}/data/loco_diagnostic_results/logs"

# --- NCSA Delta ---
LOG_DIR="${PROJECT_ROOT}/data/loco_diagnostic_results/logs"

mkdir -p "${LOG_DIR}"

name="plots"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       run_threads.sh "$@"

echo "Submitted: ${name}"
echo "Log: ${outputfile}"