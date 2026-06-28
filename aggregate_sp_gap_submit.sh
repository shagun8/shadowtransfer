#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: aggregate_sp_gap_submit.sh
#
# Submits aggregate_sp_gap.sh — runs the cross-architecture aggregator
# after all 9 per-cell SP-gap jobs (3 archs × 3 cities) have completed.

# ---- Server paths ----
BASE_PATH="${PROJECT_ROOT}/"

MAMNET_RESULTS_DIR="${BASE_PATH}/data/mamnet/sp_gap_results"
OGLANET_RESULTS_DIR="${BASE_PATH}/data/oglanet/sp_gap_results"
DINOV3_RESULTS_DIR="${BASE_PATH}/data/dinov3/sp_gap_results"

PROBS_ROOT="${BASE_PATH}/data/Test_img_probs/"
DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

OUTPUT_DIR="${BASE_PATH}/data/sp_gap_aggregate"
LOG_DIR="${BASE_PATH}/data/sp_gap_aggregate_logs"

RESOLUTION="highres"

# ---- Parse flags ----
DRY_RUN=0
LOCAL_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run)   DRY_RUN=1 ;;
        --local)     LOCAL_RUN=1 ;;   # run inline instead of sbatch
    esac
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "  SP-Gap Phase 1 Aggregator — Submit"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN"
fi
if [ "${LOCAL_RUN}" -eq 1 ]; then
echo "  MODE: LOCAL (no sbatch)"
fi
echo "========================================"
echo "  MAMNet results : ${MAMNET_RESULTS_DIR}"
echo "  OGLANet results: ${OGLANET_RESULTS_DIR}"
echo "  DINOv3 results : ${DINOV3_RESULTS_DIR}"
echo "  Probs root     : ${PROBS_ROOT}"
echo "  Data root      : ${DATA_ROOT}"
echo "  Output         : ${OUTPUT_DIR}"
echo "  Resolution     : ${RESOLUTION}"
echo ""

# Sanity check — list expected per-cell JSONs
echo "Expected per-cell JSONs:"
all_present=1
for arch_dir in "${MAMNET_RESULTS_DIR}" "${OGLANET_RESULTS_DIR}" "${DINOV3_RESULTS_DIR}"; do
    arch=$(basename $(dirname ${arch_dir}))
    for city in phoenix miami chicago; do
        # MAMNet uses "sp_gap_mamnet_{city}_{res}.json" (no c4clean in name)
        # OGLANet/DINOv3 use "sp_gap_{arch}_c4clean_{city}_{res}.json"
        f1="${arch_dir}/sp_gap_${arch}_${city}_${RESOLUTION}.json"
        f2="${arch_dir}/sp_gap_${arch}_c4clean_${city}_${RESOLUTION}.json"
        if [ -f "${f1}" ]; then
            echo "  ✓ ${f1}"
        elif [ -f "${f2}" ]; then
            echo "  ✓ ${f2}"
        else
            echo "  ✗ MISSING: ${arch}/${city}"
            all_present=0
        fi
    done
done

if [ "${all_present}" -eq 0 ]; then
    echo ""
    echo "  WARNING: some per-cell JSONs are missing. Aggregator will skip those cells."
    echo "  Continue anyway? (set --local to run interactively, or just submit)"
fi

if [ "${DRY_RUN}" -eq 1 ]; then
    echo ""
    echo "  [DRY RUN] would submit aggregate_sp_gap.sh"
    exit 0
fi

if [ "${LOCAL_RUN}" -eq 1 ]; then
    echo ""
    echo "Running aggregator locally..."
    export MAMNET_RESULTS_DIR OGLANET_RESULTS_DIR DINOV3_RESULTS_DIR
    export PROBS_ROOT DATA_ROOT OUTPUT_DIR RESOLUTION
    bash aggregate_sp_gap.sh
    exit $?
fi

# ---- Submit ----
outfile="${LOG_DIR}/aggregate_sp_gap.out"
sbatch \
    --output="${outfile}" \
    --job-name="sp_gap_aggregate" \
    --export=PROJECT_ROOT=${PROJECT_ROOT},MAMNET_RESULTS_DIR=${MAMNET_RESULTS_DIR},OGLANET_RESULTS_DIR=${OGLANET_RESULTS_DIR},DINOV3_RESULTS_DIR=${DINOV3_RESULTS_DIR},PROBS_ROOT=${PROBS_ROOT},DATA_ROOT=${DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR},RESOLUTION=${RESOLUTION} \
    aggregate_sp_gap.sh

echo ""
echo "Submitted → ${outfile}"
echo ""
echo "After completion, inspect:"
echo "  ${OUTPUT_DIR}/sp_gap_summary.md       (publication-ready Markdown)"
echo "  ${OUTPUT_DIR}/sp_gap_aggregate.json   (full results JSON)"
echo ""