#!/bin/bash
# SBATCH script for generating experiment evaluation plots
#
# ============================================================
# SBATCH HEADERS — uncomment the block for your server
# ============================================================
#
# --- Gilbreth ---
##SBATCH -A ${SLURM_ACCOUNT}
##SBATCH --partition=${SLURM_PARTITION}
##SBATCH --exclude=${NODE}
##SBATCH --gres=gpu:1
##SBATCH --qos=normal
##SBATCH --constraint=a100
##SBATCH --nodes=1
##SBATCH --gpus-per-node=1
##SBATCH --cpus-per-task=4
##SBATCH --mem=64G
##SBATCH --time=1:30:00
#
# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:30:00

# ============================================================
# SERVER SETUP — uncomment the block for your server
# ============================================================

# --- Gilbreth ---
# cd ${PROJECT_ROOT}/python/final_loco/
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load cudatoolkit
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/final_loco/
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

echo "=========================================="
echo "PLOT EXPERIMENT RESULTS"
echo "Start: $(date)"
echo "=========================================="

$PYTHON_BIN -u plot_experiment_results.py

echo ""
echo "End: $(date)"
echo "Job complete!"