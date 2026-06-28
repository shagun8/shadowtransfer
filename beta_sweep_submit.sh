#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: beta_sweep_submit.sh
#
# Submits beta_sweep.sh — runs aggregate_coverage_recover.py at
# β ∈ {0.3, 0.5, 0.7} sequentially and produces a unified summary.
# All three runs reuse the same per-cell sp_gap_*.json files (which must
# contain val_rc_records / test_rc_records).

BASE_PATH="${PROJECT_ROOT}/"
MAMNET_RESULTS_DIR="${BASE_PATH}/data/mamnet/sp_gap_results"
OGLANET_RESULTS_DIR="${BASE_PATH}/data/oglanet/sp_gap_results"
DINOV3_RESULTS_DIR="${BASE_PATH}/data/dinov3/sp_gap_results"
SWEEP_OUTPUT_DIR="${BASE_PATH}/data/coverage_recover_aggregate/beta_sweep"
LOG_DIR="${BASE_PATH}/data/coverage_recover_aggregate_logs"
RESOLUTION="highres"

DRY_RUN=0
LOCAL_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
        --local)   LOCAL_RUN=1 ;;
    esac
done

mkdir -p "${SWEEP_OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "  β Sweep — Submit"
echo "  β ∈ {0.3, 0.5, 0.7}"
echo "========================================"
echo "  Sweep output: ${SWEEP_OUTPUT_DIR}"
echo "  Resolution  : ${RESOLUTION}"
echo ""

# Sanity check — verify each per-cell JSON has rc_records fields
echo "Checking per-cell JSONs for val_rc_records / test_rc_records..."
all_ok=1
for arch_dir in "${MAMNET_RESULTS_DIR}" "${OGLANET_RESULTS_DIR}" "${DINOV3_RESULTS_DIR}"; do
    arch=$(basename $(dirname ${arch_dir}))
    for city in phoenix miami chicago; do
        found=0
        for fname in "sp_gap_${arch}_${city}_${RESOLUTION}.json" \
                     "sp_gap_${arch}_c4clean_${city}_${RESOLUTION}.json"; do
            f="${arch_dir}/${fname}"
            if [ -f "${f}" ]; then
                if grep -q '"val_rc_records"' "${f}" && grep -q '"test_rc_records"' "${f}"; then
                    echo "  ✓ ${arch}/${city}"
                else
                    echo "  ✗ ${arch}/${city} (JSON exists but missing rc_records)"
                    all_ok=0
                fi
                found=1
                break
            fi
        done
        if [ "${found}" -eq 0 ]; then
            echo "  ✗ ${arch}/${city} (JSON not found)"
            all_ok=0
        fi
    done
done

if [ "${all_ok}" -eq 0 ]; then
    echo ""
    echo "  WARNING: some cells missing rc_records — sweep will skip those cells."
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  [DRY RUN] would submit beta_sweep.sh"; exit 0
fi
if [ "${LOCAL_RUN}" -eq 1 ]; then
    export MAMNET_RESULTS_DIR OGLANET_RESULTS_DIR DINOV3_RESULTS_DIR
    export SWEEP_OUTPUT_DIR RESOLUTION
    bash beta_sweep.sh; exit $?
fi

outfile="${LOG_DIR}/beta_sweep.out"
sbatch \
    --output="${outfile}" \
    --job-name="beta_sweep" \
    --export=PROJECT_ROOT=${PROJECT_ROOT},MAMNET_RESULTS_DIR=${MAMNET_RESULTS_DIR},OGLANET_RESULTS_DIR=${OGLANET_RESULTS_DIR},DINOV3_RESULTS_DIR=${DINOV3_RESULTS_DIR},SWEEP_OUTPUT_DIR=${SWEEP_OUTPUT_DIR},RESOLUTION=${RESOLUTION} \
    beta_sweep.sh
echo "Submitted → ${outfile}"
echo ""
echo "After completion:"
echo "  ${SWEEP_OUTPUT_DIR}/beta_sweep_summary.md      (unified table across β)"
echo "  ${SWEEP_OUTPUT_DIR}/beta_sweep_summary.json    (machine-readable)"
echo "  ${SWEEP_OUTPUT_DIR}/beta_0.3/coverage_recover_results.json"
echo "  ${SWEEP_OUTPUT_DIR}/beta_0.5/coverage_recover_results.json"
echo "  ${SWEEP_OUTPUT_DIR}/beta_0.7/coverage_recover_results.json"