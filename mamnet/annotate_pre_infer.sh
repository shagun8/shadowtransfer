#!/bin/bash
# FILENAME: mamnet.sh
#SBATCH -A <SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --qos=standby
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=3:59:59

cd ${PROJECT_ROOT}/python/mamnet
module load conda
module load cuda/12.1.1  # or cuda/12.6.0 (which is the default)
module load cudnn/9.2.0.82-12
conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12

# Set environment variables for single-node distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# Run inference
${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python inference_annotation.py \
    --checkpoint ${CHECKPOINT} \
    --image_dir ${IMAGE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --city ${CITY} \
    --resolution ${RESOLUTION} \
    --session_num ${SESSION_NUM} \
    --img_size 384 \
    --device cuda

echo "Inference completed for session ${SESSION_NUM}, ${CITY} ${RESOLUTION}"