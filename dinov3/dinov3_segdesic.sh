#!/bin/bash
# FILENAME: dinov3_segdesic.sh
#
# =====================================================================
# CLUSTER SWITCH — uncomment ONE SBATCH block, comment the other
# =====================================================================
# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=128G
# #SBATCH --time=3:59:59

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --exclude=${NODE}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:20:59

# ---- NCSA Delta ----
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59

# =====================================================================
# CLUSTER SWITCH — cd, python binary, modules
# =====================================================================
# --- Anvil ---
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12

# --- Gilbreth ---
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12

# ---- NCSA Delta ----
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/dinov3
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# =====================================================================
# Distributed training env vars (single node)
# =====================================================================
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1
# =====================================================================
# CHANGED: Build optional flags from env vars (matches dinov3.sh pattern).
# Previously --eval_boundary_tolerant was hardcoded in each branch below.
# Now controlled via EVAL_TOLERANT=1 env var from the submit script so the
# same .sh file works for both tolerant and strict evaluation runs.
# =====================================================================
TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

# CHANGED: --boundary_tolerance K replaces the hardcoded 'tolerant_5px' key
# that was causing a KeyError at runtime (DetailedEvaluator defaulted to 2px
# but lookups were hardcoded to 'tolerant_5px').  Value comes from submit script.
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
# =====================================================================
# Run training
# =====================================================================
if [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_dinov3_segdesic.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 4 \
        --epochs 100 \
        --lr 0.0003 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --geo_metadata ${GEO_METADATA_PATH} \
        --early_stopping_patience 10 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_dinov3_segdesic.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 4 \
        --epochs 100 \
        --lr 0.0003 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --geo_metadata ${GEO_METADATA_PATH} \
        --early_stopping_patience 10 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_dinov3_segdesic.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 4 \
        --epochs 100 \
        --lr 0.0003 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --geo_metadata ${GEO_METADATA_PATH} \
        --early_stopping_patience 10 \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi