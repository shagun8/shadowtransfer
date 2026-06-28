#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: verify_c_star_val_submit.sh
#
# Submits verify_c_star_val.sh — runs the c*_val visual sanity check.
# Reads the same per-cell SP-gap JSONs as the coverage-recovery aggregator.

BASE_PATH="${PROJECT_ROOT}/"
MAMNET_RESULTS_DIR="${BASE_PATH}/data/mamnet/sp_gap_results"
OGLANET_RESULTS_DIR="${BASE_PATH}/data/oglanet/sp_gap_results"
DINOV3_RESULTS_DIR="${BASE_PATH}/data/dinov3/sp_gap_results"
OUTPUT_DIR="${BASE_PATH}/data/coverage_recover_aggregate/verification"
LOG_DIR="${BASE_PATH}/data/coverage_recover_aggregate_logs"
RESOLUTION="highres"
BETA="0.5"

DRY_RUN=0
LOCAL_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        --local)   LOCAL_RUN=1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "  c*_val Verification — Submit"
echo "  beta = ${BETA}"
echo "========================================"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  [DRY RUN] would submit verify_c_star_val.sh"; exit 0
fi
if [ "${LOCAL_RUN}" -eq 1 ]; then
    export MAMNET_RESULTS_DIR OGLANET_RESULTS_DIR DINOV3_RESULTS_DIR
    export OUTPUT_DIR RESOLUTION BETA
    bash verify_c_star_val.sh; exit $?
fi

outfile="${LOG_DIR}/verify_c_star_val.out"
sbatch \
    --output="${outfile}" \
    --job-name="verify_cstar" \
    --export=PROJECT_ROOT=${PROJECT_ROOT},MAMNET_RESULTS_DIR=${MAMNET_RESULTS_DIR},OGLANET_RESULTS_DIR=${OGLANET_RESULTS_DIR},DINOV3_RESULTS_DIR=${DINOV3_RESULTS_DIR},OUTPUT_DIR=${OUTPUT_DIR},RESOLUTION=${RESOLUTION},BETA=${BETA} \
    verify_c_star_val.sh
echo "Submitted → ${outfile}"
echo ""
echo "After completion:"
echo "  ${OUTPUT_DIR}/verify_c_star_val_grid_beta${BETA}.png"
echo "  ${OUTPUT_DIR}/verify_c_star_val_summary_beta${BETA}.md"