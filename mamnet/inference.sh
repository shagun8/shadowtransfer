#!/bin/bash
# FILENAME: mamnet.sh
#SBATCH -A <SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --qos=normal
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=5:59:59

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

# Build the python command based on MODE
PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# Print configuration
echo "=================================="
echo "Inference Analysis:"
echo "=================================="
echo ""

# $PYTHON_BIN run_inference.py \
			# --config ${PROJECT_ROOT}/python/mamnet/config.yaml \
			# --eval_type all --city chicago \
			# --target_resolution highres --source_resolution midres
			# --device cuda
			
			
$PYTHON_BIN run_inference.py \
    --config ${PROJECT_ROOT}/python/mamnet/config.yaml \
    --eval_type ${EVAL_TYPE} \
    --city ${CITY} \
    --target_resolution ${TARGET_RES} \
    --source_resolution ${SOURCE_RES} \
    --device cuda
	

echo ""
echo "Completed: ${EVAL_TYPE} for ${CITY}/${TARGET_RES} (source: ${SOURCE_RES})"


# # 1. Within-city baseline
# python run_inference.py --eval_type within --city chicago --target_resolution highres

# # 2. LOCO geographic transfer
# python run_inference.py --eval_type loco --city chicago --target_resolution highres

# # 3. Cross-resolution: midres → highres
# python run_inference.py --eval_type cross_res --city chicago \
    # --source_resolution midres --target_resolution highres

# # 4. Cross-resolution: highres → midres
# python run_inference.py --eval_type cross_res --city chicago \
    # --source_resolution highres --target_resolution midres

# # 5. Run all three at once
# python run_inference.py --eval_type all --city chicago \
    # --target_resolution highres --source_resolution midres