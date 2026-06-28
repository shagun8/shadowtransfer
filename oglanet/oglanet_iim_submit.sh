#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_iim_submit.sh
#
# Queues OGLANet-IIM LOCO training jobs on SLURM.
# Only LOCO folds are submitted here (3 folds × resolutions).
#
# ============================================================
# Server paths  (uncomment the block for your active server)
# ============================================================

# ---- Gilbreth (UNCOMMENTED — active server) ----
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# ---- Anvil (commented out) ----
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# ---- NCSA Delta (commented out) ----
BASE_PATH="${PROJECT_ROOT}"
BASE_PATH2="${PROJECT_ROOT}"

# ============================================================
# Shared paths
# ============================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"

# ============================================================
# Evaluation settings
# ============================================================
BOUNDARY_TOLERANCE=2        # don't-care band half-width in pixels
EARLY_STOPPING_PATIENCE=10

# ============================================================
# IIM hyper-parameters
# ============================================================
IIM_NUM_KERNELS=8
IIM_KERNEL_SIZE=5
II_LOSS_MODE=adaptive       # adaptive | fixed
II_TARGET_RATIO=0.01        # II loss = 1 % of task loss (adaptive mode)
II_LOSS_WEIGHT=0.01         # used only in fixed mode
GAMMA_RANGE_LO=0.5
GAMMA_RANGE_HI=2.0

# ============================================================
# Fold → holdout city mapping
# ============================================================
FOLD_NAMES=("phoenix" "miami" "chicago")
# fold 0 → test on phoenix  (train: miami + chicago)
# fold 1 → test on miami    (train: phoenix + chicago)
# fold 2 → test on chicago  (train: phoenix + miami)

# ============================================================
# Submit LOCO jobs
# ============================================================
echo "Queueing OGLANet-IIM LOCO jobs..."

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="oglanet_iim__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"

        sbatch \
            --output="${outputfile}" \
            --job-name="${name}" \
            --export=PROJECT_ROOT=${PROJECT_ROOT},\
MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
OUTPUT_DIR=${OUTPUT_DIR},\
USE_CONTRAST=1,\
EVAL_TOLERANT=1,\
BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},\
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},\
IIM_NUM_KERNELS=${IIM_NUM_KERNELS},\
IIM_KERNEL_SIZE=${IIM_KERNEL_SIZE},\
II_LOSS_MODE=${II_LOSS_MODE},\
II_TARGET_RATIO=${II_TARGET_RATIO},\
II_LOSS_WEIGHT=${II_LOSS_WEIGHT},\
GAMMA_RANGE_LO=${GAMMA_RANGE_LO},\
GAMMA_RANGE_HI=${GAMMA_RANGE_HI} \
            oglanet_iim.sh

    done
done

echo "All OGLANet-IIM LOCO jobs queued!"