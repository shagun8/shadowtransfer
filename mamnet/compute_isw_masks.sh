#!/bin/bash
# FILENAME: compute_isw_masks.sh
# Precompute ISW sensitivity masks (one-time, run before training).

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=1:59:59

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=1:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:29:59

# ---- Modules (uncomment the one you need) ----

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/mamnet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ---- Build optional flags ----
CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

CHECKPOINT_FLAG=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_FLAG="--checkpoint ${CHECKPOINT}"
fi

NUM_SAMPLES_FLAG=""
if [ -n "${NUM_SAMPLES}" ] && [ "${NUM_SAMPLES}" != "0" ]; then
    NUM_SAMPLES_FLAG="--num_samples ${NUM_SAMPLES}"
fi

echo "Mode:           ${MODE}"
echo "Output dir:     ${ISW_MASK_OUTPUT_DIR}"
echo "Contrast:       ${CONTRAST_FLAG}"
echo "Checkpoint:     ${CHECKPOINT_FLAG}"
echo "Num samples:    ${NUM_SAMPLES_FLAG}"

# ---- Run precomputation ----
if [ "$MODE" == "single" ]; then
    $PYTHON_BIN compute_isw_masks.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --img_size 384 \
        --output_dir ${ISW_MASK_OUTPUT_DIR} \
        --num_workers 2 \
        ${CONTRAST_FLAG} \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN compute_isw_masks.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --img_size 384 \
        --output_dir ${ISW_MASK_OUTPUT_DIR} \
        --num_workers 2 \
        ${CONTRAST_FLAG} \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN compute_isw_masks.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --img_size 384 \
        --output_dir ${ISW_MASK_OUTPUT_DIR} \
        --num_workers 2 \
        ${CONTRAST_FLAG} \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi