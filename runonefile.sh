#!/bin/bash
# SLURM job template for shadow detection inference

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
# #SBATCH --time=0:10:00

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:59:59

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
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
module purge
module load cudatoolkit
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python
DINOV3_WEIGHTS=${PROJECT_ROOT}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"


BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR_BASE="${BASE_PATH}/data/Test_img_results/analysis_results"
INFERENCE_OUTPUT_DIR="${BASE_PATH}/data/Test_img_results/"

# Set environment variables for single-node distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# python analyze_inference_results.py \
    # --inference_dir ${INFERENCE_OUTPUT_DIR} \
    # --data_root ${BASE_DATA_ROOT} \
    # --output_dir ${OUTPUT_DIR_BASE}
	
python statistical_analysis.py \
        --inference_dir ${INFERENCE_OUTPUT_DIR} \
        --data_root ${BASE_DATA_ROOT} \
        --output_csv "${OUTPUT_DIR_BASE}/statistical_results.csv" \
        --n_bootstrap 10000
		

# python aggregate_ddib_results.py --output_dir ${PROJECT_ROOT}/data/dinov3/outputs