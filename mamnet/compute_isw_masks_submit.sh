#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: compute_isw_masks_submit.sh
#
# Queues ISW mask precomputation jobs for LOCO folds.
# Run this ONCE before training with ISW.
#
# Output: one mask directory per fold under ISW_MASK_BASE_DIR.

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
ISW_MASK_BASE_DIR="${BASE_PATH}/data/mamnet/isw_masks"

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# Precompute ISW masks for each LOCO fold
# ============================================================
echo "Queueing ISW mask precomputation jobs..."

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        mask_dir="${ISW_MASK_BASE_DIR}/loco_holdout_${holdout_city}_${res}"
        name="isw_mask__loco_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        echo "    mask dir: ${mask_dir}"

        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},ISW_MASK_OUTPUT_DIR=${mask_dir},USE_CONTRAST=1 \
               compute_isw_masks.sh
    done
done

echo ""
echo "All precompute jobs queued!"
echo "Mask directories will be created under: ${ISW_MASK_BASE_DIR}"
echo ""
echo "After these jobs finish, run mamnet_isw_submit.sh to start training."