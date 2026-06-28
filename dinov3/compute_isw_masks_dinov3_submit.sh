#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: compute_isw_masks_dinov3_submit.sh
#
# Queues DINOv3 ISW mask precomputation jobs for all LOCO folds.
# Run this ONCE before training with ISW.
#
# Output: one mask directory per fold under ISW_MASK_BASE_DIR.
# After these jobs finish, run dinov3_isw_submit.sh to start ISW training.

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
ISW_MASK_BASE_DIR="${BASE_PATH}/data/dinov3/isw_masks"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

FOLD_NAMES=("phoenix" "miami" "chicago")

# ─────────────────────────────────────────────────────────────────────────────
# Queue precomputation jobs (LOCO only, as per user request)
# ─────────────────────────────────────────────────────────────────────────────

echo "Queueing DINOv3 ISW mask precomputation jobs (LOCO)..."
echo "  Mask base dir: ${ISW_MASK_BASE_DIR}"
echo ""

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        mask_dir="${ISW_MASK_BASE_DIR}/loco_holdout_${holdout_city}_${res}"
        name="dinov3_isw_masks__loco_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/dinov3/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        echo "    mask dir: ${mask_dir}"

        sbatch --output="${outputfile}" \
               --job-name="${name}" \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
WEIGHT_DIR=${WEIGHT_DIR},\
ISW_MASK_OUTPUT_DIR=${mask_dir} \
               compute_isw_masks_dinov3.sh
    done
done

echo ""
echo "All precomputation jobs queued!"
echo "Mask directories will be created under: ${ISW_MASK_BASE_DIR}"
echo ""
echo "After these jobs finish, run: bash dinov3_isw_submit.sh"