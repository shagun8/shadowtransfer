#!/bin/bash
# FILENAME: mamnet_inria.sh
#
# SLURM job script for training MAMNet on INRIA building footprints.
# Mirrors mamnet.sh structure exactly — only the python entry point
# changes (train_inria.py instead of train.py) and output_dir uses
# the mamnet_inria/ subtree.
#
# ---- SBATCH: Common settings ----
# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=5:59:59
# ---- SBATCH: Server-specific (uncomment the one you need) ----
# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=standby
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=3:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:59:59


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

# Distributed training env vars (single node)
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# ---- Build optional flags from env vars ----

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

COMPARISON_FLAGS=""
if [ -n "${COMPARISON_INFERENCE_DIR}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_inference_dir ${COMPARISON_INFERENCE_DIR}"
fi
if [ -n "${COMPARISON_DATA_ROOT}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_data_root ${COMPARISON_DATA_ROOT}"
fi

echo "=== INRIA MAMNet Training ==="
echo "Contrast:          ${CONTRAST_FLAG}"
echo "Tolerant eval:     ${TOLERANT_FLAG}"
echo "Boundary tolerance:${BOUNDARY_TOL_FLAG}"
echo "Early stopping:    ${EARLY_STOP_FLAG}"
echo "Comparison:        ${COMPARISON_FLAGS}"
echo "Mode:              ${MODE}"

# ---- Run training via train_inria.py (not train.py) ----
if [ "$MODE" == "single" ]; then
    $PYTHON_BIN train_inria.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${COMPARISON_FLAGS}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN train_inria.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${COMPARISON_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE} (expected 'single' or 'loco')"
    exit 1
fi