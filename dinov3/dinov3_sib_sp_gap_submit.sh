#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_sib_sp_gap_submit.sh
#
# Fans out SP-gap Phase 1 analysis across 3 LOCO holdout folds for DINOv3
# C4-clean.
#
# C4-clean tag from dinov3_sib_submit.sh: "C4clean_haar_vib"
# DINOv3 has no SAG, no FDA per §4.4 (encoder is already domain-invariant).

# ---- Server paths ----
BASE_PATH="${PROJECT_ROOT}/"
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
DINOV3_OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/sp_gap_results"
LOG_DIR="${BASE_PATH}/data/dinov3/sp_gap_logs"

WEIGHT_DIR="${BASE_PATH}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# C4-clean experiment tag for DINOv3 — matches dinov3_sib_submit.sh:
#   submit_loco "C4clean_haar_vib" 1 1 0 0 0 0.01 0.0001 0.02 1.0 0.1 0.1 0.01 0 "C4clean" 0
# Output dir name follows the train_dinov3_sib.py exp_name template:
#   dinov3_sib{sib_tag}{fda_suffix}_loco_holdout_{test_city}_{resolution}_1
# With use_haar=1, use_vib=1, use_content_aug=0, adaptive_beta=0:
#   sib_tag = "_haar_vib"
#   fda_suffix = ""
#   plus exp_tag prepended: "_C4clean_haar_vib"
EXP_DIR_TAG="dinov3_sib_C4clean_haar_vib"

FOLD_NAMES=("phoenix" "miami" "chicago")
RESOLUTION="highres"

# ---- Parse flags ----
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "  DINOv3 SP-Gap Phase 1 — Submission"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "========================================"
echo "  C4-clean dir tag: ${EXP_DIR_TAG}"
echo "  Results dir:      ${OUTPUT_DIR}"
echo "  Log dir:          ${LOG_DIR}"
echo "  Backbone weights: ${WEIGHT_DIR}"
echo ""

for fold_id in 0 1 2; do
    holdout="${FOLD_NAMES[$fold_id]}"

    # DINOv3 train_dinov3_sib.py output dir naming convention:
    #   {exp_dir_tag}_loco_holdout_{holdout}_{resolution}_1
    C4CLEAN_DIR="${DINOV3_OUTPUT_DIR}/${EXP_DIR_TAG}_loco_holdout_${holdout}_${RESOLUTION}_1"
    C4CLEAN_CKPT="${C4CLEAN_DIR}/checkpoint_best.pth"

    name="dinov3_sp_gap_${holdout}_${RESOLUTION}"
    outfile="${LOG_DIR}/${name}.out"

    echo "  fold_id=${fold_id}  holdout=${holdout}"
    echo "    C4-clean: ${C4CLEAN_CKPT}"

    if [ ! -f "${C4CLEAN_CKPT}" ]; then
        echo "    WARNING: not found — trying alternative names..."
        # Try a few common variations of the directory name
        for alt_dir in \
            "${DINOV3_OUTPUT_DIR}/${EXP_DIR_TAG}_loco_holdout_${holdout}_${RESOLUTION}" \
            "${DINOV3_OUTPUT_DIR}/dinov3_sib_C4clean__loco_holdout_${holdout}__${RESOLUTION}"
        do
            if [ -f "${alt_dir}/checkpoint_best.pth" ]; then
                C4CLEAN_CKPT="${alt_dir}/checkpoint_best.pth"
                echo "    Found alt: ${C4CLEAN_CKPT}"
                break
            elif [ -f "${alt_dir}/best_model.pth" ]; then
                C4CLEAN_CKPT="${alt_dir}/best_model.pth"
                echo "    Found alt: ${C4CLEAN_CKPT}"
                break
            fi
        done
        if [ ! -f "${C4CLEAN_CKPT}" ]; then
            echo "    SKIPPING fold ${fold_id} — checkpoint not found"
            continue
        fi
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "    [DRY RUN] would submit: ${name}"
        continue
    fi

    sbatch \
        --output="${outfile}" \
        --job-name="${name}" \
        --export=PROJECT_ROOT=${PROJECT_ROOT},FOLD_ID=${fold_id},RESOLUTION=${RESOLUTION},C4CLEAN_CHECKPOINT=${C4CLEAN_CKPT},BASE_DATA_ROOT=${BASE_DATA_ROOT},WEIGHT_DIR=${WEIGHT_DIR},OUTPUT_DIR=${OUTPUT_DIR} \
        dinov3_sib_sp_gap.sh

    echo "    Submitted → ${outfile}"
done

echo ""
echo "========================================"
echo "  Summary: 3 jobs (C4-clean inference only)"
echo "========================================"
echo ""
echo "  Monitor : squeue -u \$USER"
echo "  Watch   : tail -f ${LOG_DIR}/dinov3_sp_gap_phoenix_${RESOLUTION}.out"
echo "  Results : ls ${OUTPUT_DIR}/sp_gap_dinov3_c4clean_*_${RESOLUTION}.json"
echo ""