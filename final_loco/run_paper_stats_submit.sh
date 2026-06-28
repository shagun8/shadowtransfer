#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# Submit the paper statistics computation job.
#
# Usage:
#   ./run_paper_stats_submit.sh                  # full run (bootstrap + permutation)
#   ./run_paper_stats_submit.sh --skip_bootstrap  # quick run (~1 min, CV CIs only)
#
# ---- Uncomment the block for your active server ----

# --- Gilbreth ---
# LOG_DIR="${PROJECT_ROOT}/data/loco_diagnostic_results/logs"

# --- NCSA Delta ---
LOG_DIR="${PROJECT_ROOT}/data/loco_diagnostic_results/logs"

mkdir -p "${LOG_DIR}"

name="paper_statistics"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       run_paper_stats.sh "$@"

echo "Submitted: ${name}"
echo "Log: ${outputfile}"
echo ""
echo "Quick check: squeue -u \$USER"
echo "Results: loco_diagnostic_results/paper_statistics/"