#!/bin/bash
# FILENAME: mamnet_segdesic.sh

# ---- SBATCH common ----
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=3:59:59
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu

# ---- Server-specific SBATCH (uncomment the one you need) ----
# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=0:15:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59


# ---- Module loading (uncomment the one you need) ----
# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

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

CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

EVAL_TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    EVAL_TOLERANT_FLAG="--eval_boundary_tolerant"
fi

# Boundary tolerance K — controls the don't-care band half-width in pixels.
# DetailedEvaluator always runs; this sets the band width.
# When EVAL_TOLERANT=1, also controls which metric drives decisions.
BOUNDARY_TOLERANCE_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOLERANCE_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

EARLY_STOP_FLAG=""
if [ -n "${EARLY_STOPPING_PATIENCE}" ]; then
    EARLY_STOP_FLAG="--early_stopping_patience ${EARLY_STOPPING_PATIENCE}"
fi

echo "Contrast flag:           ${CONTRAST_FLAG}"
echo "Eval tolerant flag:      ${EVAL_TOLERANT_FLAG}"
echo "Boundary tolerance flag: ${BOUNDARY_TOLERANCE_FLAG}"
echo "Early stop flag:         ${EARLY_STOP_FLAG}"

# ---- Run training ----

if [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_segdesic.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 4 \
        --epochs 100 \
        --lr 0.0003 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --geo_metadata ${GEO_METADATA_PATH} \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${CONTRAST_FLAG} \
        ${EVAL_TOLERANT_FLAG} \
        ${BOUNDARY_TOLERANCE_FLAG} \
        ${EARLY_STOP_FLAG}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_segdesic.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 4 \
        --epochs 100 \
        --lr 0.0003 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --geo_metadata ${GEO_METADATA_PATH} \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${CONTRAST_FLAG} \
        ${EVAL_TOLERANT_FLAG} \
        ${BOUNDARY_TOLERANCE_FLAG} \
        ${EARLY_STOP_FLAG}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_segdesic.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --batch_size 4 \
        --epochs 100 \
        --lr 0.0003 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --geo_metadata ${GEO_METADATA_PATH} \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${CONTRAST_FLAG} \
        ${EVAL_TOLERANT_FLAG} \
        ${BOUNDARY_TOLERANCE_FLAG} \
        ${EARLY_STOP_FLAG}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi