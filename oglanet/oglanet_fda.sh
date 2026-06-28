#!/bin/bash
# FILENAME: oglanet_fda.sh
# ---- SBATCH: Common settings ----
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=128G
# #SBATCH --time=3:59:59
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --exclude=${NODE}

# ---- SBATCH: Server-specific (uncomment the one you need) ----
# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:20:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59

# ---- Server paths (uncomment the one you need) ----
# --- Gilbreth ---
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/oglanet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# Distributed training env vars (single node)
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# ---- Build optional flags from environment variables ----
OPTIONAL_FLAGS=""

if [ "${USE_CONTRAST}" == "1" ]; then
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --use_contrast"
fi

if [ "${EVAL_TOLERANT}" == "1" ]; then
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --eval_boundary_tolerant"
fi

# Boundary tolerance K — controls the don't-care band half-width in pixels.
# DetailedEvaluator always runs; this sets the band width.
# When EVAL_TOLERANT=1, also controls which metric drives decisions.
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

if [ -n "${EARLY_STOPPING_PATIENCE}" ]; then
    OPTIONAL_FLAGS="${OPTIONAL_FLAGS} --early_stopping_patience ${EARLY_STOPPING_PATIENCE}"
fi

# FDA flags (always on for loco in this script; target root passed from submit)
FDA_FLAGS=""
if [ -n "${TARGET_CITY_ROOT}" ]; then
    FDA_FLAGS="--use_fda --fda_target_root ${TARGET_CITY_ROOT} --fda_L 0.01"
fi

echo "Optional flags: ${OPTIONAL_FLAGS}"
echo "FDA flags:      ${FDA_FLAGS}"

# ---- Run training ----
if [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${FDA_FLAGS} \
        ${OPTIONAL_FLAGS}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${FDA_FLAGS} \
        ${OPTIONAL_FLAGS}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${FDA_FLAGS} \
        ${OPTIONAL_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi