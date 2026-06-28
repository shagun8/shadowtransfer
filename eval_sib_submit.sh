#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: eval_sib_submit.sh
#
# Submits SIB evaluation jobs — one job per model type (3 total).
#
# Each job runs eval_sib.py which:
#   1. Scores all LOCO baselines (vanilla, fda, segdesic, iim, isw, mrfp_plus, fada)
#      and Upper Bound for each city
#   2. Auto-discovers and scores all SIB variant folders for that model
#   3. Prints per-city comparison tables + ranking
#   4. Saves eval_{model}_{resolution}.json
#   5. When all 3 model JSONs are present, prints + saves cross-model table
#
# Usage:
#   bash eval_sib_submit.sh           # submit all 3 jobs
#   bash eval_sib_submit.sh --dry-run # preview without submitting

# =====================================================================
# Server paths — uncomment the block for your target server
# =====================================================================

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"

# =====================================================================
# Derived paths (shared across all servers — do not edit)
# =====================================================================
DATA_PATH="${BASE_PATH}/data"

TEST_IMG_RESULTS_DIR="${DATA_PATH}/Test_img_results"
GT_BASE_DIR="${DATA_PATH}/Final_data_test"
EVAL_OUTPUT_DIR="${DATA_PATH}/eval_sib_results"
LOG_DIR="${DATA_PATH}/eval_sib_logs"

RESOLUTION="highres"
BOUNDARY_TOLERANCE=2
IMG_SIZE=384

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
    [ "$arg" == "--dry-run" ] && DRY_RUN=1
done

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  SIB Evaluation — Job Submission"
[ "${DRY_RUN}" -eq 1 ] && echo "  MODE: DRY RUN (no jobs submitted)"
echo "========================================"
echo "  Resolution       : ${RESOLUTION}"
echo "  Boundary tol     : ±${BOUNDARY_TOLERANCE}px"
echo "  GT base dir      : ${GT_BASE_DIR}"
echo "  Test img results : ${TEST_IMG_RESULTS_DIR}"
echo "  Eval output dir  : ${EVAL_OUTPUT_DIR}"
echo "  Logs             : ${LOG_DIR}"
echo ""

mkdir -p "${EVAL_OUTPUT_DIR}" "${LOG_DIR}"

# =====================================================================
# Helper: submit one model evaluation job
#
# Args:
#   $1  MODEL_TYPE      — mamnet | oglanet | dinov3
#   $2  SIB_OUTPUT_DIR  — model-specific SIB prediction outputs dir
# =====================================================================
submit_eval() {
    local MODEL_TYPE=$1
    local SIB_OUTPUT_DIR=$2

    local JOB_NAME="eval_sib_${MODEL_TYPE}_${RESOLUTION}"
    local OUTFILE="${LOG_DIR}/${JOB_NAME}.out"

    echo "  Queueing: ${JOB_NAME}"
    echo "    sib_output_dir: ${SIB_OUTPUT_DIR}"

    # Existence check (warn only — prediction dirs may be partially complete)
    [ ! -d "${SIB_OUTPUT_DIR}" ] && \
        echo "    WARNING: SIB_OUTPUT_DIR missing — ${SIB_OUTPUT_DIR}"
    [ ! -d "${TEST_IMG_RESULTS_DIR}" ] && \
        echo "    WARNING: TEST_IMG_RESULTS_DIR missing — ${TEST_IMG_RESULTS_DIR}"
    [ ! -d "${GT_BASE_DIR}" ] && \
        echo "    WARNING: GT_BASE_DIR missing — ${GT_BASE_DIR}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "    [DRY RUN] would submit: ${JOB_NAME}"
        echo "              log: ${OUTFILE}"
        return
    fi

    sbatch \
        --output="${OUTFILE}" \
        --job-name="${JOB_NAME}" \
        --export=PROJECT_ROOT=${PROJECT_ROOT},MODEL_TYPE="${MODEL_TYPE}",\
SIB_OUTPUT_DIR="${SIB_OUTPUT_DIR}",\
TEST_IMG_RESULTS_DIR="${TEST_IMG_RESULTS_DIR}",\
GT_BASE_DIR="${GT_BASE_DIR}",\
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}",\
RESOLUTION="${RESOLUTION}",\
BOUNDARY_TOLERANCE="${BOUNDARY_TOLERANCE}",\
IMG_SIZE="${IMG_SIZE}" \
        eval_sib.sh

    if [ $? -eq 0 ]; then
        echo "    ✓ Submitted: ${JOB_NAME}"
    else
        echo "    ✗ Submission failed: ${JOB_NAME}"
    fi
}

# =====================================================================
# Submit one job per model type
# =====================================================================

submit_eval "mamnet"  "${DATA_PATH}/mamnet/outputs"
echo ""
submit_eval "oglanet" "${DATA_PATH}/oglanet/outputs"
echo ""
submit_eval "dinov3"  "${DATA_PATH}/dinov3/outputs"

echo ""
echo "========================================"
echo "  3 jobs submitted (1 per model type)"
echo ""
echo "  Each job evaluates all cities:"
echo "    chicago / miami / phoenix"
echo ""
echo "  Outputs saved to:"
echo "    ${EVAL_OUTPUT_DIR}/eval_mamnet_${RESOLUTION}.json"
echo "    ${EVAL_OUTPUT_DIR}/eval_oglanet_${RESOLUTION}.json"
echo "    ${EVAL_OUTPUT_DIR}/eval_dinov3_${RESOLUTION}.json"
echo "    ${EVAL_OUTPUT_DIR}/cross_model_comparison_${RESOLUTION}.json"
echo "      (cross-model table auto-printed when all 3 jobs finish)"
echo ""
echo "  Monitor jobs  :  squeue -u \$USER"
echo "  Watch a log   :  tail -f ${LOG_DIR}/eval_sib_<model>_${RESOLUTION}.out"
echo "========================================"