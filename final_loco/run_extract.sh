#!/bin/bash
# SLURM job template for encoder feature extraction (linear probe diagnostic 1d).
# Receives parameters via environment variables from run_extract_submit.sh.
#
# ============================================================
# SERVER: uncomment the #SBATCH block for your active server
# ============================================================

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --exclude=${NODE}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=0:29:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:29:59

# ============================================================
# SERVER PATHS — uncomment the block for your active server
# ============================================================

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/final_loco/
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# OUTPUT_DIR=${PROJECT_ROOT}/data/extracted_features
# DINOV3_WEIGHTS=${PROJECT_ROOT}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/final_loco/
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python
OUTPUT_DIR=${PROJECT_ROOT}/data/extracted_features
DINOV3_WEIGHTS=${PROJECT_ROOT}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

echo "=========================================="
echo "FEATURE EXTRACTION"
echo "=========================================="
echo "Model:         ${MODEL_TYPE}/${MODEL_VARIANT}"
echo "Checkpoint ID: ${CHECKPOINT_ID}"
echo "City:          ${CITY}"
echo "Res:           ${RES}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Data:          ${DATA_ROOT}"
echo "Output:        ${OUTPUT_DIR}"
echo "=========================================="

$PYTHON_BIN extract_features.py \
    --model_type      ${MODEL_TYPE} \
    --model_variant   ${MODEL_VARIANT} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_root       ${DATA_ROOT} \
    --city            ${CITY} \
    --res             ${RES} \
    --checkpoint_id   ${CHECKPOINT_ID} \
    --output_dir      ${OUTPUT_DIR} \
    --dinov3_weights_path ${DINOV3_WEIGHTS} \
    --batch_size 4 \
    --num_workers 4

echo "Job complete: $(date)"