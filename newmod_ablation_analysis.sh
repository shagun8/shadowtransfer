#!/bin/bash
# FILENAME: newmod_ablation_analysis.sh
#
# Runs the diagnostic-module (CACR + CE-AURC + TENT) ablation analysis
# over MAMNet, OGLANet, and DINOv3 outputs.
#
# =====================================================================
# SBATCH directives — uncomment the block for your target server
# =====================================================================

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

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=5:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=${NODE}
#SBATCH --time=2:29:59

# =====================================================================
# Modules + paths — uncomment the block for your target server
# =====================================================================

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# =====================================================================
# Distributed training env vars (single node — analysis is CPU-only,
# but match the format of the SIB-ablation runner for consistency)
# =====================================================================
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# =====================================================================
# Run analysis
# =====================================================================
python newmod_ablation_analysis.py \
    --mamnet_root  ${PROJECT_ROOT}/data/mamnet/outputs \
    --oglanet_root ${PROJECT_ROOT}/data/oglanet/outputs \
    --dinov3_root  ${PROJECT_ROOT}/data/dinov3/outputs \
    --output_dir   ${PROJECT_ROOT}/data/newmod_analysis \
    --eval_type    tolerant \
    --n_bootstrap  10000 \
    --alpha        0.05 \
    --boundary_tolerance 2