#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: complete_eval_submit.sh
#
# Submits complete_eval.sh jobs for any OGLANet+SIB experiments whose
# training was interrupted — i.e. checkpoints exist but evaluation
# outputs are missing (no predictions/ dir or no comparison_results.json).
#
# Usage:
#   bash complete_eval_submit.sh              # auto-discover all incomplete
#   bash complete_eval_submit.sh --dry-run    # print jobs without submitting
#
# What it checks:
#   For each experiment dir under OUTPUT_DIR:
#     - Has checkpoints/best_model.pth       → training ran at least partially
#     - Missing predictions/                 → inference never completed
#     - OR missing comparison_results.json   → comparison never completed
#   Jobs that already have both are skipped (idempotent).
#
# ─────────────────────────────────────────────────────────────────────────────

# =====================================================================
# Server paths — Anvil (active)
# =====================================================================
BASE_PATH="${PROJECT_ROOT}"
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
LOG_DIR="${BASE_PATH}/data/oglanet"

# =====================================================================
# Server paths — Gilbreth (uncomment to use)
# =====================================================================
# BASE_PATH="${PROJECT_ROOT}"
# BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test"
# OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
# LOG_DIR="${BASE_PATH}/data/oglanet"

# Fold mapping: 0=phoenix, 1=miami, 2=chicago
FOLD_NAMES=("phoenix" "miami" "chicago")

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Helper: submit one completion job
#
# Args:
#   $1  EXP_DIR    — full path to experiment output dir
#   $2  CITY       — holdout city name (phoenix/miami/chicago)
#   $3  RES        — resolution (midres/highres)
# ─────────────────────────────────────────────────────────────────────────────
submit_completion() {
    local EXP_DIR=$1
    local CITY=$2
    local RES=$3

    local CKPT="${EXP_DIR}/checkpoints/best_model.pth"
    local EXP_NAME
    EXP_NAME=$(basename "${EXP_DIR}")
    local OUTFILE="${LOG_DIR}/complete_eval_${EXP_NAME}.out"

    # Vanilla baseline paths (same structure as original training submission)
    local COMP_INF="${OUTPUT_DIR}/oglanet_loco_holdout_${CITY}_${RES}_1"
    local COMP_DATA="${BASE_DATA_ROOT}/${CITY}/${RES}"

    echo ""
    echo "  → Submitting completion for: ${EXP_NAME}"
    echo "      city=${CITY}  res=${RES}"
    echo "      checkpoint: ${CKPT}"
    echo "      log:        ${OUTFILE}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "      [DRY RUN — not submitted]"
        return
    fi

    sbatch \
        --output="${OUTFILE}" \
        --job-name="complete_${EXP_NAME:0:48}" \
        --export=PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT_PATH="${CKPT}",OUTPUT_DIR="${EXP_DIR}",BASE_DATA_ROOT="${BASE_DATA_ROOT}",COMPARISON_INFERENCE_DIR="${COMP_INF}",COMPARISON_DATA_ROOT="${COMP_DATA}" \
        complete_eval.sh
}

# ─────────────────────────────────────────────────────────────────────────────
# Main: scan output dir and find incomplete experiments
# ─────────────────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  OGLANet+SIB — Completing Interrupted Evaluations"
echo "========================================================"
echo "  Output dir : ${OUTPUT_DIR}"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE       : DRY RUN (no jobs submitted)"
fi
echo ""

n_submitted=0
n_skipped=0
n_no_ckpt=0

for EXP_DIR in "${OUTPUT_DIR}"/oglanet_sib_*; do
    [ -d "${EXP_DIR}" ] || continue

    EXP_NAME=$(basename "${EXP_DIR}")
    CKPT="${EXP_DIR}/checkpoints/best_model.pth"
    PRED_DIR="${EXP_DIR}/predictions"
    COMP_FILE="${EXP_DIR}/comparison_results.json"

    # ── No checkpoint → training never started or completely failed ──
    if [ ! -f "${CKPT}" ]; then
        echo "  SKIP (no checkpoint):  ${EXP_NAME}"
        (( n_no_ckpt++ )) || true
        continue
    fi

    # ── Already complete → skip ──
    if [ -d "${PRED_DIR}" ] && [ -f "${COMP_FILE}" ]; then
        N_PREDS=$(find "${PRED_DIR}" -maxdepth 1 \
                  \( -name "*.png" -o -name "*.jpg" -o -name "*.tif" \) \
                  2>/dev/null | wc -l)
        if [ "${N_PREDS}" -gt 0 ]; then
            echo "  DONE (${N_PREDS} preds):    ${EXP_NAME}"
            (( n_skipped++ )) || true
            continue
        fi
    fi

    # ── Needs completion — extract city and resolution from name ──
    CITY=""
    RES=""
    for FNAME in "${FOLD_NAMES[@]}"; do
        if [[ "${EXP_NAME}" == *"${FNAME}"* ]]; then
            CITY="${FNAME}"
            break
        fi
    done
    for RNAME in "midres" "highres"; do
        if [[ "${EXP_NAME}" == *"${RNAME}"* ]]; then
            RES="${RNAME}"
            break
        fi
    done

    if [ -z "${CITY}" ] || [ -z "${RES}" ]; then
        echo "  WARN (cannot parse city/res): ${EXP_NAME}"
        continue
    fi

    submit_completion "${EXP_DIR}" "${CITY}" "${RES}"
    (( n_submitted++ )) || true
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Summary"
echo "  Submitted : ${n_submitted}"
echo "  Skipped   : ${n_skipped} (already complete)"
echo "  No ckpt   : ${n_no_ckpt} (training never ran)"
echo "========================================================"
echo ""
echo "  Monitor jobs  :  squeue -u \$USER"
echo "  Watch a log   :  tail -f ${LOG_DIR}/complete_eval_<name>.out"
echo "  Check outputs :  ls ${OUTPUT_DIR}/oglanet_sib_*/comparison_results.json"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Specific override: force-submit for the known interrupted M3 experiment.
# Uncomment this block if auto-discovery misses it or you want to re-run it.
# ─────────────────────────────────────────────────────────────────────────────
# submit_completion \
#     "${OUTPUT_DIR}/oglanet_sib_M3_haar_vib_aug_ab_sfda_ctr__loco_holdout_phoenix__midres" \
#     "phoenix" \
#     "midres"