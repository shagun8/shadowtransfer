#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: eval_mamnet_sib_submit.sh
#
# Submits evaluation jobs for MAMNet+SIB experiments.
# Each job reports all three aggregation methods (A/B/C) for:
#   - SIB (this run)
#   - Upper Bound (if --ub_pred_dir is set)
#   - LOCO Vanilla (if --loco_pred_dir is set, for recovery ratios)
#
# Usage:
#   bash eval_mamnet_sib_submit.sh           # submit jobs
#   bash eval_mamnet_sib_submit.sh --dry-run # preview without submitting

# =====================================================================
# Server paths — Anvil (active)
# =====================================================================
BASE_PATH="${PROJECT_ROOT}"

# =====================================================================
# Server paths — Gilbreth (uncomment to swap)
# =====================================================================
# BASE_PATH="${PROJECT_ROOT}"

# =====================================================================
# Shared paths
# =====================================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
LOG_DIR="${BASE_PATH}/data/mamnet"

# Root of pre-computed baseline inference results.
# Expected structure:
#   upper/{city}/{res}/mamnet/base/       <- upper bound predictions
#   loco/{city}/{res}/mamnet/vanilla/     <- LOCO vanilla predictions
#   loco/{city}/{res}/mamnet/fda/
#   loco/{city}/{res}/mamnet/segdesic/
#   loco/{city}/{res}/mamnet/mcl/
TEST_IMG_RESULTS="${BASE_PATH}/data/Test_img_results"

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
    [ "$arg" == "--dry-run" ] && DRY_RUN=1
done

# =====================================================================
# Helper: submit one evaluation job
#
# Args:
#   $1  EXP_NAME    — experiment directory name inside OUTPUT_DIR
#   $2  CITY        — test city  (chicago | miami | phoenix)
#   $3  RES         — resolution (highres | midres)
#   $4  EVAL_MODE   — predictions | checkpoint | both
# =====================================================================
submit_eval() {
    local EXP_NAME=$1
    local CITY=$2
    local RES=$3
    local MODE=$4

    local EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
    local GT_DIR="${BASE_DATA_ROOT}/${CITY}/${RES}/test/masks"
    local DATA_ROOT="${BASE_DATA_ROOT}/${CITY}/${RES}"

    # Upper bound and LOCO vanilla prediction directories
    local UB_PRED_DIR="${TEST_IMG_RESULTS}/upper/${CITY}/${RES}/mamnet/base"
    local LOCO_PRED_DIR="${TEST_IMG_RESULTS}/loco/${CITY}/${RES}/mamnet/vanilla"

    local JOB_NAME="eval_${EXP_NAME:0:55}"
    local OUTFILE="${LOG_DIR}/eval_${EXP_NAME}.out"

    echo ""
    echo "===== Queueing eval: ${EXP_NAME} ====="
    echo "  city=${CITY}  res=${RES}  mode=${MODE}"
    echo "  exp_dir      : ${EXP_DIR}"
    echo "  gt_dir       : ${GT_DIR}"
    echo "  ub_pred_dir  : ${UB_PRED_DIR}"
    echo "  loco_pred_dir: ${LOCO_PRED_DIR}"

    # Existence checks (warn only — don't abort)
    [ ! -d "${EXP_DIR}"      ] && echo "  WARNING: EXP_DIR missing      — ${EXP_DIR}"
    [ ! -d "${GT_DIR}"       ] && echo "  WARNING: GT_DIR missing        — ${GT_DIR}"
    [ ! -d "${UB_PRED_DIR}"  ] && echo "  WARNING: UB_PRED_DIR missing   — ${UB_PRED_DIR}"
    [ ! -d "${LOCO_PRED_DIR}"] && echo "  WARNING: LOCO_PRED_DIR missing — ${LOCO_PRED_DIR}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "  [DRY RUN] would submit: ${JOB_NAME}"
        return
    fi

    sbatch \
        --output="${OUTFILE}" \
        --job-name="${JOB_NAME}" \
        --export=PROJECT_ROOT=${PROJECT_ROOT},EVAL_MODE=${MODE},EXP_DIR=${EXP_DIR},GT_DIR=${GT_DIR},DATA_ROOT=${DATA_ROOT},UB_PRED_DIR=${UB_PRED_DIR},LOCO_PRED_DIR=${LOCO_PRED_DIR},IMG_SIZE=384,BATCH_SIZE=4 \
        eval_mamnet_sib.sh
}

echo "========================================"
echo "  MAMNet + SIB — Evaluation Submission"
[ "${DRY_RUN}" -eq 1 ] && echo "  MODE: DRY RUN"
echo "========================================"
echo "  Outputs : ${OUTPUT_DIR}"
echo "  Logs    : ${LOG_DIR}/eval_*.out"
echo "  UB root : ${TEST_IMG_RESULTS}/upper/{city}/{res}/mamnet/base"
echo "  LOCO    : ${TEST_IMG_RESULTS}/loco/{city}/{res}/mamnet/vanilla"
echo ""

# =====================================================================
# Add entries below — one line per experiment.
# Format:
#   submit_eval  <exp_dir_name>  <city>  <resolution>  <eval_mode>
#
# eval_mode:
#   predictions  — score saved PNGs in exp_dir/predictions/
#   checkpoint   — reload best_model.pth and run fresh inference
#   both         — predictions first, then checkpoint (recommended)
# =====================================================================

# ── M2 miami highres — the run in question ────────────────────────────
submit_eval \
    "mamnet_sib_M2_haar_vib_aug_ab_fda_ctr__loco_holdout_miami__highres" \
    "miami" "highres" "both"

# ── Uncomment to evaluate other folds / configs ───────────────────────
# submit_eval \
#     "mamnet_sib_M2_haar_vib_aug_ab_fda_ctr__loco_holdout_phoenix__highres" \
#     "phoenix" "highres" "both"

# submit_eval \
#     "mamnet_sib_M2_haar_vib_aug_ab_fda_ctr__loco_holdout_chicago__highres" \
#     "chicago" "highres" "both"

# submit_eval \
#     "mamnet_sib_M6_vib_aug_ab_fda_ctr__loco_holdout_miami__highres" \
#     "miami" "highres" "both"

# submit_eval \
#     "mamnet_sib_M7_haar_vib_aug_fda_ctr__loco_holdout_miami__highres" \
#     "miami" "highres" "both"

echo ""
echo "========================================"
echo "  Monitor : squeue -u \$USER"
echo "  Logs    : tail -f ${LOG_DIR}/eval_*.out"
echo "  Results : cat ${OUTPUT_DIR}/*/eval_results*.json | python -m json.tool"
echo ""
echo "  Output JSON has three sections per method:"
echo "    strict_dataset_level   (A — old method)"
echo "    strict_per_image_mean  (B — new method)"
echo "    tolerant_5px_per_image (C — tolerant)"
echo "========================================"