#!/bin/bash
# SLURM job template for shadow detection probability inference.
# Parallel to run_inference.sh but calls run_inference_probs.py, which
# saves per-pixel shadow-class probability maps (float16 .npy) instead
# of binary PNG masks.

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
# #SBATCH --time=0:15:00

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:19:59

# Job parameters (set by submission script):
#   MODEL_TYPE, MODEL_VARIANT, TEST_TYPE, CITY, TRAIN_RES, TEST_RES
#   CHECKPOINT_PATH, DATA_ROOT, OUTPUT_DIR

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
# cd ${PROJECT_ROOT}/python
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# DINOV3_WEIGHTS=${PROJECT_ROOT}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python
DINOV3_WEIGHTS=${PROJECT_ROOT}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

echo "=========================================="
echo "PROBABILITY INFERENCE JOB"
echo "=========================================="
echo "Model: ${MODEL_TYPE}/${MODEL_VARIANT}"
echo "Test: ${TEST_TYPE}"
echo "City: ${CITY}"
echo "Train res: ${TRAIN_RES}"
echo "Test res: ${TEST_RES}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data: ${DATA_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "DINOv3 weights: ${DINOV3_WEIGHTS}"
echo "=========================================="

$PYTHON_BIN run_inference_probs.py \
    --model_type ${MODEL_TYPE} \
    --model_variant ${MODEL_VARIANT} \
    --test_type ${TEST_TYPE} \
    --city ${CITY} \
    --train_res ${TRAIN_RES} \
    --test_res ${TEST_RES} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --img_size 384 \
    --batch_size 4 \
    --num_workers 4 \
    --device cuda \
    --dinov3_weights_path ${DINOV3_WEIGHTS}

echo "Job complete!"