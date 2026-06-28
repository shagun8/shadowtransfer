#!/bin/bash
# FILENAME: dinov3_isw.sh
#
# SLURM runner for DINOv3 + ISW shadow detection training.
# Prerequisites: ISW masks must be precomputed first
#                (run compute_isw_masks_dinov3_submit.sh).

# ─────────────────────────────────────────────────────────────────────────────
# SBATCH: Gilbreth  (ACTIVE)
# ─────────────────────────────────────────────────────────────────────────────
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:29:59

# ─────────────────────────────────────────────────────────────────────────────
# SBATCH: Anvil  (comment out if using Gilbreth)
# ─────────────────────────────────────────────────────────────────────────────
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=128G
# #SBATCH --time=5:59:59

# ─────────────────────────────────────────────────────────────────────────────
# SBATCH: NCSA Delta  (comment out if using Gilbreth)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59
#SBATCH --job-name=dinov3_isw

# ─────────────────────────────────────────────────────────────────────────────
# Module & conda setup
# ─────────────────────────────────────────────────────────────────────────────

# --- Gilbreth (ACTIVE) ---
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

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/dinov3
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ─────────────────────────────────────────────────────────────────────────────
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────────────
# Optional flags from environment variables
# ─────────────────────────────────────────────────────────────────────────────

TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

BOUNDARY_TOL_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOL_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

ISW_LAMBDA_FLAG=""
if [ -n "${ISW_LAMBDA}" ]; then
    ISW_LAMBDA_FLAG="--isw_lambda ${ISW_LAMBDA}"
fi

ISW_LAYERS_FLAG=""
if [ -n "${ISW_LAYERS}" ]; then
    ISW_LAYERS_FLAG="--isw_layers ${ISW_LAYERS}"
fi

COMPARISON_FLAGS=""
if [ -n "${COMPARISON_INFERENCE_DIR}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_inference_dir ${COMPARISON_INFERENCE_DIR}"
fi
if [ -n "${COMPARISON_DATA_ROOT}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_data_root ${COMPARISON_DATA_ROOT}"
fi

echo "============================================================"
echo "DINOv3 + ISW Training"
echo "============================================================"
echo "Mode:               ${MODE}"
echo "Tolerant eval:      ${TOLERANT_FLAG}"
echo "Boundary tolerance: ${BOUNDARY_TOL_FLAG}"
echo "ISW mask dir:       ${ISW_MASK_DIR}"
echo "ISW lambda:         ${ISW_LAMBDA_FLAG}"
echo "ISW layers:         ${ISW_LAYERS_FLAG}"
echo "Comparison:         ${COMPARISON_FLAGS}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Run training
# ─────────────────────────────────────────────────────────────────────────────

if [ "$MODE" == "single" ]; then
    ${PYTHON_BIN} -u train_dinov3_isw.py \
        --mode single \
        --data_root "${DATA_ROOT}" \
        --model_name dinov3_vits16 \
        --weights_path "${WEIGHT_DIR}" \
        --batch_size 8 \
        --epochs 50 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --early_stopping_patience 15 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${ISW_LAYERS_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "loco" ]; then
    ${PYTHON_BIN} -u train_dinov3_isw.py \
        --mode loco \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --fold_id "${FOLD_ID}" \
        --model_name dinov3_vits16 \
        --weights_path "${WEIGHT_DIR}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --early_stopping_patience 15 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${ISW_LAYERS_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "all" ]; then
    ${PYTHON_BIN} -u train_dinov3_isw.py \
        --mode all \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --model_name dinov3_vits16 \
        --weights_path "${WEIGHT_DIR}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --early_stopping_patience 15 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${ISW_LAYERS_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${COMPARISON_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi

TRAIN_RC=$?
echo ""
echo "=== Python exited with code: ${TRAIN_RC} ==="
if [ ${TRAIN_RC} -ne 0 ]; then
    echo "ERROR: Training failed!"
fi
exit ${TRAIN_RC}