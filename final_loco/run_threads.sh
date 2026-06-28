#!/bin/bash
# SLURM job script for cross-location diagnostic analysis.
# Runs run_diagnostics.py (all threads by default) then generate_plots.py.
#
# NOTE: Diagnostics are CPU-only (numpy / sklearn / scipy).
#       GPU resources are allocated here for queue compatibility — they are
#       not used by the diagnostic code itself.
#
# ============================================================
# USAGE (via submit_diagnostics.sh, or direct sbatch):
#   sbatch run_threads.sh                          # all diagnostics
#   sbatch run_threads.sh --threads 1 3            # threads 1 and 3
#   sbatch run_threads.sh --diagnostics 1a 1b 3e   # specific diagnostics
#   sbatch run_threads.sh --threads all --resolutions highres
#   sbatch run_threads.sh --threads all --cities chicago phoenix
# ============================================================

# ---- Gilbreth — commented out ----
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --exclude=${NODE}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=9
# #SBATCH --mem=64G
# #SBATCH --time=15:59:59

# ---- NCSA Delta — active ----
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:59:59

# ============================================================
# ENVIRONMENT SETUP
# ============================================================

# --- Gilbreth ---
# cd ${PROJECT_ROOT}/python/final_loco
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
cd ${PROJECT_ROOT}/python/final_loco
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ============================================================
# RUN
# ============================================================

echo "=========================================="
echo "DIAGNOSTIC JOB"
echo "Start: $(date)"
echo "Arguments: $@"
echo "Host: $(hostname)"
echo "=========================================="

# Run all diagnostics (or forward specific flags from submit_diagnostics.sh).
# Threads available: 1 (1a, 1b, 1c, 1d), 3 (3a, 3b, 3e), 4 (4a, 4b)
# 1d (linear probe) runs automatically inside thread 1 if extracted features exist.

# $PYTHON_BIN -u run_diagnostics.py --threads all "$@"
# $PYTHON_BIN -u run_diagnostics.py --threads all "$@"

echo ""
echo "=========================================="
echo "Diagnostics complete: $(date)"
echo "=========================================="

echo ""
echo "Generating plots..."
$PYTHON_BIN -u generate_plots.py

echo ""
echo "=========================================="
echo "All done: $(date)"
echo "=========================================="