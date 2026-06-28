#!/bin/bash
# FILENAME: dinov3_fda.sh
#
# SLURM runner for DINOv3 + FDA shadow detection training.
# Supports Gilbreth and Anvil clusters — uncomment the block you need.
#
# ---- SLURM common settings ----
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --exclude=${NODE}
# #SBATCH --time=3:59:59
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu

# ---- SLURM cluster-specific (uncomment ONE block) ----
# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=0:20:59

# ---- NCSA Delta ----
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59


# ---- Working directory (uncomment ONE) ----
# ---- Python binary (uncomment ONE) ----

# ---- Module & conda setup (uncomment ONE block) ----
# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# ---- NCSA Delta ----
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/dinov3
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ---- Distributed training env vars (single node) ----
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# ---- Build optional flags from env vars ----

# CHANGED: tolerant flag now driven by EVAL_TOLERANT env var (was hardcoded --eval_boundary_tolerant)
TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

# CHANGED: boundary tolerance K (don't-care band half-width in pixels).
# DetailedEvaluator always runs with this width.
# When EVAL_TOLERANT=1, also selects which metric drives decisions.
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

echo "Tolerant eval:      ${TOLERANT_FLAG}"
echo "Boundary tolerance: ${BOUNDARY_TOL_FLAG}"
echo "Comparison:         ${COMPARISON_FLAGS}"

# ---- Run training ----
if [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_dinov3.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 10 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_dinov3.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --use_fda \
        --fda_target_root ${TARGET_CITY_ROOT} \
        --fda_L 0.01 \
        --early_stopping_patience 10 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_dinov3.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 10 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi