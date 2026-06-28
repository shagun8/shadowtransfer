#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_isw_submit.sh
#
# Queues DINOv3 + ISW training jobs for LOCO analysis.
#
# Prerequisites: ISW masks must already be precomputed.
#   Run: bash compute_isw_masks_dinov3_submit.sh
#   Wait for those jobs to finish, then run this script.
#
# Queues: 3 jobs  (fold 0, 1, 2 × highres)

# ─────────────────────────────────────────────────────────────────────────────
# Server paths  — uncomment ONE block
# ─────────────────────────────────────────────────────────────────────────────

# --- Gilbreth (ACTIVE) ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"
BASE_PATH2="${PROJECT_ROOT}"

# ─────────────────────────────────────────────────────────────────────────────
# Derived paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
ISW_MASK_BASE_DIR="${BASE_PATH}/data/dinov3/isw_masks"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

# Evaluation settings
BOUNDARY_TOLERANCE=2   # ±K px don't-care band
ISW_LAMBDA=0.6         # weight for ISW loss term

FOLD_NAMES=("phoenix" "miami" "chicago")

# ─────────────────────────────────────────────────────────────────────────────
# Queue LOCO training jobs
# ─────────────────────────────────────────────────────────────────────────────

echo "Queueing DINOv3 + ISW LOCO training jobs..."
echo ""

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        mask_dir="${ISW_MASK_BASE_DIR}/loco_holdout_${holdout_city}_${res}"

        # Verify masks exist before submitting
        if [ ! -d "${mask_dir}" ]; then
            echo "  WARNING: Mask directory not found: ${mask_dir}"
            echo "           Run compute_isw_masks_dinov3_submit.sh first!"
            echo "           Skipping fold ${fold_id}."
            echo ""
            continue
        fi
        # Quick sanity check: expect at least 3 .npy mask files
        n_masks=$(ls "${mask_dir}"/*.npy 2>/dev/null | grep -c "_mask.npy" || true)
        if [ "${n_masks}" -lt 3 ]; then
            echo "  WARNING: Found only ${n_masks} mask file(s) in ${mask_dir}"
            echo "           Expected at least 3 (block3, block6, block9)."
            echo "           Skipping fold ${fold_id}."
            echo ""
            continue
        fi

        name="dinov3_isw__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/dinov3/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        echo "    mask dir:   ${mask_dir}"
        echo "    output dir: ${OUTPUT_DIR}"

        sbatch --output="${outputfile}" \
               --job-name="${name}" \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
OUTPUT_DIR=${OUTPUT_DIR},\
WEIGHT_DIR=${WEIGHT_DIR},\
ISW_MASK_DIR=${mask_dir},\
ISW_LAMBDA=${ISW_LAMBDA},\
EVAL_TOLERANT=1,\
BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},\
COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},\
COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               dinov3_isw.sh

        echo ""
    done
done

echo "All DINOv3 + ISW LOCO jobs queued!"
echo ""
echo "Output directories will be created under: ${OUTPUT_DIR}"
echo "Look for: dinov3_isw_loco_holdout_{city}_{res}_1/"