#!/bin/bash
# FILENAME: dinov3_fada.sh
#
# SLURM runner for DINOv3-FADA shadow detection training.
# Supports single-city, all-city, and LOCO modes.
#
# ════════════════════════════════════════════════════════════════════════════
# SERVER BLOCKS — uncomment exactly ONE block
# ════════════════════════════════════════════════════════════════════════════

# ── Gilbreth ──────────────────────────────────────────────────────────────
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:29:59

# ── Anvil ─────────────────────────────────────────────────────────────────
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=128G
# #SBATCH --time=5:59:59

# ── NCSA Delta ────────────────────────────────────────────────────────────
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59

# ════════════════════════════════════════════════════════════════════════════
# Module + conda setup — uncomment the matching server block
# ════════════════════════════════════════════════════════════════════════════

# ── Gilbreth ──────────────────────────────────────────────────────────────
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# ── Anvil ─────────────────────────────────────────────────────────────────
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# ── NCSA Delta ────────────────────────────────────────────────────────────
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/dinov3
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ════════════════════════════════════════════════════════════════════════════
# Distributed training env vars (single node)
# ════════════════════════════════════════════════════════════════════════════
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# ════════════════════════════════════════════════════════════════════════════
# Build optional flags from environment variables
# ════════════════════════════════════════════════════════════════════════════

# Boundary-tolerant evaluation
TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

# Don't-care band half-width (DetailedEvaluator always uses this)
BOUNDARY_TOL_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOL_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

# FADA hyperparameters (defaults follow Bi et al., NeurIPS 2024)
#   rank=16  (Table 3 best at 16–32)
#   m=100    (Fig 8 stable at 75–125)
#   stages=3 6 9 11 (all DINOv3 feature-extraction blocks)
FADA_RANK_FLAG=""
if [ -n "${FADA_RANK}" ]; then
    FADA_RANK_FLAG="--fada_rank ${FADA_RANK}"
fi

FADA_TOKEN_FLAG=""
if [ -n "${FADA_TOKEN_LENGTH}" ]; then
    FADA_TOKEN_FLAG="--fada_token_length ${FADA_TOKEN_LENGTH}"
fi

FADA_STAGES_FLAG=""
if [ -n "${FADA_STAGES}" ]; then
    FADA_STAGES_FLAG="--fada_stages ${FADA_STAGES}"
fi

# Learning rate override
LR_FLAG=""
if [ -n "${LR}" ]; then
    LR_FLAG="--lr ${LR}"
fi

# Comparison inference results (optional)
COMPARISON_FLAGS=""
if [ -n "${COMPARISON_INFERENCE_DIR}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_inference_dir ${COMPARISON_INFERENCE_DIR}"
fi
if [ -n "${COMPARISON_DATA_ROOT}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_data_root ${COMPARISON_DATA_ROOT}"
fi

echo "=========================================="
echo "DINOv3-FADA Configuration"
echo "=========================================="
echo "  Mode              : ${MODE}"
echo "  Tolerant eval     : ${TOLERANT_FLAG:-disabled}"
echo "  Boundary tolerance: ${BOUNDARY_TOL_FLAG:-default(2)}"
echo "  FADA rank         : ${FADA_RANK_FLAG:-default(16)}"
echo "  FADA token length : ${FADA_TOKEN_FLAG:-default(100)}"
echo "  FADA stages       : ${FADA_STAGES_FLAG:-default(3 6 9 11)}"
echo "  LR                : ${LR_FLAG:-default(1e-4)}"
echo "  Comparison        : ${COMPARISON_FLAGS:-none}"
echo "=========================================="

# ════════════════════════════════════════════════════════════════════════════
# Run training
# ════════════════════════════════════════════════════════════════════════════

if [ "$MODE" == "single" ]; then

    $PYTHON_BIN -u train_dinov3_fada.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs ${EPOCHS:-100} \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE:-15} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${FADA_RANK_FLAG} \
        ${FADA_TOKEN_FLAG} \
        ${FADA_STAGES_FLAG} \
        ${LR_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "loco" ]; then

    $PYTHON_BIN -u train_dinov3_fada.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs ${EPOCHS:-100} \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE:-15} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${FADA_RANK_FLAG} \
        ${FADA_TOKEN_FLAG} \
        ${FADA_STAGES_FLAG} \
        ${LR_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "all" ]; then

    $PYTHON_BIN -u train_dinov3_fada.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs ${EPOCHS:-100} \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience ${EARLY_STOPPING_PATIENCE:-15} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${FADA_RANK_FLAG} \
        ${FADA_TOKEN_FLAG} \
        ${FADA_STAGES_FLAG} \
        ${LR_FLAG} \
        ${COMPARISON_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}  (expected: single | loco | all)"
    exit 1
fi