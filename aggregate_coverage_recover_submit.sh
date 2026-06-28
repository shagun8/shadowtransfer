#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: aggregate_coverage_recover_submit.sh
#
# Submits aggregate_coverage_recover.sh after the 9 per-cell sp_gap JSONs
# have been (re-)produced with the val_rc_records / test_rc_records fields.

BASE_PATH="${PROJECT_ROOT}/"
MAMNET_RESULTS_DIR="${BASE_PATH}/data/mamnet/sp_gap_results"
OGLANET_RESULTS_DIR="${BASE_PATH}/data/oglanet/sp_gap_results"
DINOV3_RESULTS_DIR="${BASE_PATH}/data/dinov3/sp_gap_results"
OUTPUT_DIR="${BASE_PATH}/data/coverage_recover_aggregate"
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
echo "  Phase 2 Coverage Recovery — Submit"
echo "  beta = ${BETA}"
echo "========================================"

# Sanity check — verify each per-cell JSON has the new fields
echo "Checking per-cell JSONs for val_rc_records / test_rc_records..."
all_ok=1
for arch_dir in "${MAMNET_RESULTS_DIR}" "${OGLANET_RESULTS_DIR}" "${DINOV3_RESULTS_DIR}"; do
    arch=$(basename $(dirname ${arch_dir}))
    for city in phoenix miami chicago; do
        for fname in "sp_gap_${arch}_${city}_${RESOLUTION}.json" \
                     "sp_gap_${arch}_c4clean_${city}_${RESOLUTION}.json"; do
            f="${arch_dir}/${fname}"
            if [ -f "${f}" ]; then
                if grep -q '"val_rc_records"' "${f}" && grep -q '"test_rc_records"' "${f}"; then
                    echo "  ✓ ${arch}/${city} (has val_rc_records + test_rc_records)"
                else
                    echo "  ✗ ${arch}/${city} (JSON exists but missing rc_records — re-run sp_gap)"
                    all_ok=0
                fi
                break
            fi
        done
    done
done

if [ "${all_ok}" -eq 0 ]; then
    echo ""
    echo "  WARNING: some cells missing rc_records. Re-run sp_gap_analysis with the patch."
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  [DRY RUN] would submit aggregate_coverage_recover.sh"; exit 0
fi
if [ "${LOCAL_RUN}" -eq 1 ]; then
    export MAMNET_RESULTS_DIR OGLANET_RESULTS_DIR DINOV3_RESULTS_DIR
    export OUTPUT_DIR RESOLUTION BETA
    bash aggregate_coverage_recover.sh; exit $?
fi

outfile="${LOG_DIR}/aggregate_coverage_recover.out"
sbatch \
    --output="${outfile}" \
    --job-name="cov_recover_agg" \
    --export=PROJECT_ROOT=${PROJECT_ROOT},MAMNET_RESULTS_DIR=${MAMNET_RESULTS_DIR},OGLANET_RESULTS_DIR=${OGLANET_RESULTS_DIR},DINOV3_RESULTS_DIR=${DINOV3_RESULTS_DIR},OUTPUT_DIR=${OUTPUT_DIR},RESOLUTION=${RESOLUTION},BETA=${BETA} \
    aggregate_coverage_recover.sh
echo "Submitted → ${outfile}"
echo ""
echo "After completion:"
echo "  ${OUTPUT_DIR}/coverage_recover_summary.md"
echo "  ${OUTPUT_DIR}/coverage_recover_results.json"