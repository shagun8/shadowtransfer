#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_iim_submit.sh
#
# Queues DINOv3-IIM LOCO training jobs on SLURM.
# Only LOCO mode is submitted here (all 3 folds × highres).
#
# Total: 3 jobs (holdout_phoenix, holdout_miami, holdout_chicago)

# ================================================================
# Cluster paths — uncomment ONE block
# ================================================================

# ---- Gilbreth (ACTIVE) ----
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# ---- Anvil ----
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# ---- NCSA Delta ----
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

# ================================================================
# Derived paths (no changes needed below this line)
# ================================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

# Don't-care band half-width in pixels
BOUNDARY_TOLERANCE=2

# IIM hyper-parameters (YOLA paper defaults)
IIM_NUM_KERNELS=8
IIM_KERNEL_SIZE=5
II_LOSS_MODE=adaptive
II_TARGET_RATIO=0.01
II_LOSS_WEIGHT=0.01      # only used in fixed mode
GAMMA_RANGE_LO=0.5
GAMMA_RANGE_HI=2.0

# Fold → held-out city mapping (0=phoenix, 1=miami, 2=chicago)
FOLD_NAMES=("phoenix" "miami" "chicago")

# ================================================================
# Submit LOCO jobs — all 3 folds × highres
# ================================================================
echo "========================================"
echo "Queueing DINOv3-IIM LOCO jobs"
echo "========================================"

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout="${FOLD_NAMES[$fold_id]}"
        name="dinov3_iim__loco_holdout_${holdout}__${res}"
        outfile="${BASE_PATH}/data/dinov3/${name}.out"

        echo "  - LOCO fold ${fold_id}  (holdout: ${holdout})  res=${res}"

        sbatch --output=${outfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},\
MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
OUTPUT_DIR=${OUTPUT_DIR},\
WEIGHT_DIR=${WEIGHT_DIR},\
EVAL_TOLERANT=1,\
BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},\
IIM_NUM_KERNELS=${IIM_NUM_KERNELS},\
IIM_KERNEL_SIZE=${IIM_KERNEL_SIZE},\
II_LOSS_MODE=${II_LOSS_MODE},\
II_TARGET_RATIO=${II_TARGET_RATIO},\
II_LOSS_WEIGHT=${II_LOSS_WEIGHT},\
GAMMA_RANGE_LO=${GAMMA_RANGE_LO},\
GAMMA_RANGE_HI=${GAMMA_RANGE_HI},\
COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},\
COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               dinov3_iim.sh
    done
done

echo ""
echo "All DINOv3-IIM LOCO jobs queued!"
echo "Output logs → ${BASE_PATH}/data/dinov3/"