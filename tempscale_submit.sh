#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: tempscale_submit.sh
#
# Post-hoc class-conditional temperature scaling on already-trained
# C4-clean checkpoints. Loads best model, fits (T_pos, T_neg) on
# source-city val, applies at test on held-out city. No retraining.
#
# Each job runs ~5 minutes (val + test inference, LBFGS fit, metrics).

BASE_PATH="${PROJECT_ROOT}"
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test"
LOG_DIR="${BASE_PATH}/data/tempscale_logs"
WEIGHT_DIR_DINOV3="${BASE_PATH}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# C4-clean output directory roots (where the trained checkpoints live)
MAMNET_ROOT="${BASE_PATH}/data/mamnet/outputs"
OGLANET_ROOT="${BASE_PATH}/data/oglanet/outputs"
DINOV3_ROOT="${BASE_PATH}/data/dinov3/outputs"

# C4-clean experiment tag prefixes (the names you used in the submit scripts)
MAMNET_TAG="mamnet_sib_C4clean_haar_vib_sag_fda_ctr"
OGLANET_TAG="oglanet_sib_C4clean_haar_vib_sag_fda_ctr"
DINOV3_TAG="dinov3_sib_C4clean_haar_vib"   # adjust if your tag differs

FOLD_NAMES=("phoenix" "miami" "chicago")

mkdir -p ${LOG_DIR}

DRY_RUN=0
for arg in "$@"; do
    case $arg in --dry-run) DRY_RUN=1 ;; esac
done

# ─────────────────────────────────────────────────────────────────────
# Helper: submit one tempscale job
# Args: ARCH FOLD_ID CKPT_DIR EXTRA_ARGS
# ─────────────────────────────────────────────────────────────────────
submit_one() {
    local ARCH=$1 FOLD_ID=$2 CKPT_DIR=$3 EXTRA=$4
    local HOLDOUT="${FOLD_NAMES[$FOLD_ID]}"
    local NAME="tempscale_${ARCH}_${HOLDOUT}"
    local LOG="${LOG_DIR}/${NAME}.out"

    if [ ! -d "${CKPT_DIR}" ]; then
        echo "  ! SKIP ${NAME}: checkpoint dir not found: ${CKPT_DIR}"
        return
    fi

    echo "  - ${ARCH} fold=${FOLD_ID} (${HOLDOUT})"
    echo "      ckpt: ${CKPT_DIR}"

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "      [DRY RUN]"
        return
    fi

    sbatch --output=${LOG} --job-name=${NAME} \
           --export=PROJECT_ROOT=${PROJECT_ROOT},ARCH=${ARCH},FOLD_ID=${FOLD_ID},CKPT_DIR=${CKPT_DIR},BASE_DATA_ROOT=${BASE_DATA_ROOT},WEIGHT_DIR_DINOV3=${WEIGHT_DIR_DINOV3},EXTRA="${EXTRA}" \
           tempscale.sh
}

echo "========================================"
echo "  Tempscale Eval — 3 archs × 3 folds"
[ "${DRY_RUN}" -eq 1 ] && echo "  MODE: DRY RUN"
echo "========================================"

# MAMNet
echo ""
echo "MAMNet:"
for fold_id in 0 1 2; do
    holdout="${FOLD_NAMES[$fold_id]}"
    ckpt_dir="${MAMNET_ROOT}/${MAMNET_TAG}__loco_holdout_${holdout}__highres"
    submit_one "mamnet" $fold_id "${ckpt_dir}" ""
done

# OGLANet
echo ""
echo "OGLANet:"
for fold_id in 0 1 2; do
    holdout="${FOLD_NAMES[$fold_id]}"
    ckpt_dir="${OGLANET_ROOT}/${OGLANET_TAG}__loco_holdout_${holdout}__highres"
    submit_one "oglanet" $fold_id "${ckpt_dir}" ""
done

# DINOv3
echo ""
echo "DINOv3:"
for fold_id in 0 1 2; do
    holdout="${FOLD_NAMES[$fold_id]}"
    # DINOv3 uses a different naming convention — check your actual dir layout
    ckpt_dir="${DINOV3_ROOT}/${DINOV3_TAG}_loco_holdout_${holdout}_highres_1"
    submit_one "dinov3" $fold_id "${ckpt_dir}" ""
done

echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Results: <ckpt_dir>/tempscale_results.json"
echo ""