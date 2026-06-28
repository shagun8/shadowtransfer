#!/bin/bash
# FILENAME: compute_isw_masks_dinov3.sh
#
# SLURM script to precompute ISW sensitivity masks for DINOv3.
# One-time offline computation — run ONCE before ISW training.
#
# Output: one .npy mask per hooked layer + metadata.json
# Pass --output_dir to train_dinov3_isw.py as --isw_mask_dir

# ─────────────────────────────────────────────────────────────────────────────
# SBATCH: Gilbreth  (ACTIVE)
# ─────────────────────────────────────────────────────────────────────────────
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=2:59:59

# ─────────────────────────────────────────────────────────────────────────────
# SBATCH: Anvil  (comment out if using Gilbreth)
# ─────────────────────────────────────────────────────────────────────────────
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=2:59:59

# ─────────────────────────────────────────────────────────────────────────────
# SBATCH: NCSA Delta  (comment out if using Gilbreth)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:29:59

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
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────────────
# Optional flags
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_FLAG=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_FLAG="--checkpoint ${CHECKPOINT}"
fi

NUM_SAMPLES_FLAG=""
if [ -n "${NUM_SAMPLES}" ] && [ "${NUM_SAMPLES}" != "0" ]; then
    NUM_SAMPLES_FLAG="--num_samples ${NUM_SAMPLES}"
fi

echo "============================================================"
echo "DINOv3 ISW Mask Precomputation"
echo "============================================================"
echo "Mode:           ${MODE}"
echo "Resolution:     ${RESOLUTION}"
echo "Fold ID:        ${FOLD_ID}"
echo "Output dir:     ${ISW_MASK_OUTPUT_DIR}"
echo "Checkpoint:     ${CHECKPOINT_FLAG}"
echo "Num samples:    ${NUM_SAMPLES_FLAG}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Run precomputation
# ─────────────────────────────────────────────────────────────────────────────

if [ "$MODE" == "single" ]; then
    ${PYTHON_BIN} -u compute_isw_masks_dinov3.py \
        --mode single \
        --data_root "${DATA_ROOT}" \
        --img_size 384 \
        --weights_path "${WEIGHT_DIR}" \
        --model_name dinov3_vits16 \
        --layers block3,block6,block9 \
        --output_dir "${ISW_MASK_OUTPUT_DIR}" \
        --num_workers 2 \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG}

elif [ "$MODE" == "loco" ]; then
    ${PYTHON_BIN} -u compute_isw_masks_dinov3.py \
        --mode loco \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --fold_id "${FOLD_ID}" \
        --img_size 384 \
        --weights_path "${WEIGHT_DIR}" \
        --model_name dinov3_vits16 \
        --layers block3,block6,block9 \
        --output_dir "${ISW_MASK_OUTPUT_DIR}" \
        --num_workers 2 \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG}

elif [ "$MODE" == "all" ]; then
    ${PYTHON_BIN} -u compute_isw_masks_dinov3.py \
        --mode all \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --img_size 384 \
        --weights_path "${WEIGHT_DIR}" \
        --model_name dinov3_vits16 \
        --layers block3,block6,block9 \
        --output_dir "${ISW_MASK_OUTPUT_DIR}" \
        --num_workers 2 \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi

PRECOMPUTE_RC=$?
echo ""
echo "=== Precomputation exited with code: ${PRECOMPUTE_RC} ==="
if [ ${PRECOMPUTE_RC} -ne 0 ]; then
    echo "ERROR: ISW mask precomputation failed!"
fi
exit ${PRECOMPUTE_RC}