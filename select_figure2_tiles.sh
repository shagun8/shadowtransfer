#!/bin/bash
# FILENAME: select_figure2_tiles.sh
#
# SLURM job that runs select_figure2_tiles.py on a CPU node.
# Submitted by select_figure2_tiles_submit.sh, which exports BASE_PATH
# and OUTPUT_DIR into the environment.
#
# CPU-only — selection logic is I/O-bound, no GPU needed.
# ============================================================
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:14:59

set -eo pipefail

echo "============================================================"
echo "  Figure 2 tile selection"
echo "  Started:    $(date)"
echo "  Host:       $(hostname)"
echo "  Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "  BASE_PATH:  ${BASE_PATH}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "============================================================"

# ============================================================
# Environment activation — adjust to match your conda/venv setup.
# Uncomment whichever block applies on your target server.
# ============================================================
# --- Conda ---
# source ~/.bashrc
# conda activate shademaps

# --- venv ---
# source ${HOME}/envs/shademaps/bin/activate

# --- Module loads (NCSA Delta example) ---
# module load anaconda3
# conda activate shademaps

# ============================================================
# Run the selection
# ============================================================
# --- NCSA Delta ---
module purge
module load cudatoolkit
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

python -u select_figure2_tiles.py

echo ""
echo "Finished:  $(date)"
echo "============================================================"