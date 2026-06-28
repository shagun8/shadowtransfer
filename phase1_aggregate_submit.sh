#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: phase1_aggregate_submit.sh

BASE_PATH="${PROJECT_ROOT}"
LOG_DIR="${BASE_PATH}/data/eval_sib_logs"
mkdir -p "${LOG_DIR}"

name="phase1_aggregate"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       phase1_aggregate.sh "$@"

echo "Submitted: ${name}"
echo "Log:       ${outputfile}"