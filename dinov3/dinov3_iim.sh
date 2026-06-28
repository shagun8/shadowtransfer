#!/bin/bash
# FILENAME: dinov3_iim.sh
#
# SLURM runner for DINOv3-IIM shadow detection training.
# Supports Gilbreth, Anvil, and NCSA Delta clusters.
# Uncomment the ONE block matching your cluster.

# ================================================================
# SLURM — Gilbreth  (ACTIVE — uncomment the block below)
# ================================================================
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:39:59

# ================================================================
# SLURM — Anvil  (comment out if not on Anvil)
# ================================================================
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=5:59:59

# ================================================================
# SLURM — NCSA Delta  (comment out if not on the primary cluster)
# ================================================================
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59
#SBATCH --job-name=dinov3_iim_run

# ================================================================
# Modules & conda — Gilbreth  (ACTIVE)
# ================================================================
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# cd ${PROJECT_ROOT}/python/dinov3

# ================================================================
# Modules & conda — Anvil  (comment out if not on Anvil)
# ================================================================
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python
# cd ${PROJECT_ROOT}/python/dinov3

# ================================================================
# Modules & conda — NCSA Delta  (comment out if not on the primary cluster)
# ================================================================
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python
cd ${PROJECT_ROOT}/python/dinov3

# ================================================================
# Distributed training env vars (single node)
# ================================================================
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# ================================================================
# Build optional flags from env vars
# ================================================================

TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

BOUNDARY_TOL_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOL_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

COMPARISON_FLAGS=""
if [ -n "${COMPARISON_INFERENCE_DIR}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_inference_dir ${COMPARISON_INFERENCE_DIR}"
fi
if [ -n "${COMPARISON_DATA_ROOT}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_data_root ${COMPARISON_DATA_ROOT}"
fi

# ----------------------------------------------------------------
# IIM-specific flags (defaults match YOLA paper recommendations)
# ----------------------------------------------------------------
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

echo "============================================"
echo "DINOv3-IIM Training"
echo "============================================"
echo "Mode:               ${MODE}"
echo "Tolerant eval:      ${TOLERANT_FLAG}"
echo "Boundary tolerance: ${BOUNDARY_TOL_FLAG}"
echo "IIM num_kernels:    ${IIM_NUM_KERNELS}"
echo "IIM kernel_size:    ${IIM_KERNEL_SIZE}"
echo "II loss mode:       ${II_LOSS_MODE}"
echo "II target ratio:    ${II_TARGET_RATIO} (adaptive)"
echo "II loss weight:     ${II_LOSS_WEIGHT}  (fixed mode only)"
echo "Gamma range:        [${GAMMA_LO}, ${GAMMA_HI}]"
echo "Comparison:         ${COMPARISON_FLAGS}"
echo "============================================"

# ================================================================
# Run training
# ================================================================

if [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_dinov3_iim.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 50 \
        --lr 0.00005 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --min_lr 1e-6 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 15 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${IIM_FLAGS} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_dinov3_iim.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --min_lr 1e-6 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 15 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${IIM_FLAGS} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_dinov3_iim.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --weight_decay 0.05 \
        --warmup_epochs 5 \
        --min_lr 1e-6 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 15 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${IIM_FLAGS} \
        ${COMPARISON_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi