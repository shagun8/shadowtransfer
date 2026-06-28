#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: tempscale_aggregate_submit.sh

BASE_PATH="${PROJECT_ROOT}"
LOG_DIR="${BASE_PATH}/data/eval_sib_logs"
mkdir -p "${LOG_DIR}"

name="tempscale_aggregate"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       tempscale_aggregate.sh "$@"

echo "Submitted: ${name}"
echo "Log:       ${outputfile}"
echo ""
echo "When complete, check:"
echo "  ${BASE_PATH}/data/case_study_output/case_study_table.txt"
echo "  ${BASE_PATH}/data/case_study_output/case_study_table.tex"
echo "  ${BASE_PATH}/data/case_study_output/case_study_report.json"