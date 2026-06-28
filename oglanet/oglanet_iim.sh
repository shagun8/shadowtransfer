#!/bin/bash
# FILENAME: oglanet_iim.sh
#
# SLURM job script for OGLANet-IIM training.
# Accepts MODE, DATA_ROOT / BASE_DATA_ROOT, RESOLUTION, FOLD_ID,
# OUTPUT_DIR and IIM hyper-parameters via --export.
#
# ============================================================
# SBATCH — common resource settings (shared across servers)
# ============================================================

# ---- Gilbreth (UNCOMMENTED — active server) ----
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:29:59

# ---- Anvil (commented out) ----
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=5:59:59

# ---- NCSA Delta (commented out) ----
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59

# ============================================================
# Module loading + paths (one block per server)
# ============================================================

# ---- Gilbreth (UNCOMMENTED — active server) ----
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12

# ---- Anvil (commented out) ----
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# ---- NCSA Delta (commented out) ----
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/oglanet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ============================================================
# Distributed training env vars (single-node, single-GPU)
# ============================================================
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# ============================================================
# Build optional flag strings from exported env vars
# ============================================================

CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

BOUNDARY_TOL_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOL_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

EARLY_STOP_FLAG=""
if [ -n "${EARLY_STOPPING_PATIENCE}" ] && [ "${EARLY_STOPPING_PATIENCE}" != "0" ]; then
    EARLY_STOP_FLAG="--early_stopping_patience ${EARLY_STOPPING_PATIENCE}"
fi

# ============================================================
# IIM hyper-parameters  (defaults match paper recommendations)
# ============================================================
IIM_NUM_KERNELS=${IIM_NUM_KERNELS:-8}
IIM_KERNEL_SIZE=${IIM_KERNEL_SIZE:-5}
II_LOSS_MODE=${II_LOSS_MODE:-adaptive}
II_TARGET_RATIO=${II_TARGET_RATIO:-0.01}
II_LOSS_WEIGHT=${II_LOSS_WEIGHT:-0.01}
GAMMA_LO=${GAMMA_RANGE_LO:-0.5}
GAMMA_HI=${GAMMA_RANGE_HI:-2.0}

IIM_FLAGS="--num_kernels ${IIM_NUM_KERNELS} \
           --kernel_size ${IIM_KERNEL_SIZE} \
           --ii_loss_mode ${II_LOSS_MODE} \
           --ii_target_ratio ${II_TARGET_RATIO} \
           --ii_loss_weight ${II_LOSS_WEIGHT} \
           --gamma_range_lo ${GAMMA_LO} \
           --gamma_range_hi ${GAMMA_HI}"

# ============================================================
# Diagnostic echo
# ============================================================
echo "============================================"
echo "OGLANet-IIM Training"
echo "============================================"
echo "MODE:               ${MODE}"
echo "Contrast flag:      ${CONTRAST_FLAG}"
echo "Tolerant eval:      ${TOLERANT_FLAG}"
echo "Boundary tolerance: ${BOUNDARY_TOL_FLAG}"
echo "Early stopping:     ${EARLY_STOP_FLAG}"
echo "IIM num_kernels:    ${IIM_NUM_KERNELS}"
echo "IIM kernel_size:    ${IIM_KERNEL_SIZE}"
echo "II loss mode:       ${II_LOSS_MODE}"
echo "II target ratio:    ${II_TARGET_RATIO}"
echo "II loss weight:     ${II_LOSS_WEIGHT}  (fixed mode only)"
echo "Gamma range:        [${GAMMA_LO}, ${GAMMA_HI}]"
echo "============================================"

# ============================================================
# Run
# ============================================================

if [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_iim.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --weight_decay 1e-4 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${IIM_FLAGS}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_iim.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --weight_decay 1e-4 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${IIM_FLAGS}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_iim.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --weight_decay 1e-4 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${IIM_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi