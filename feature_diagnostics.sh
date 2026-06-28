#!/bin/bash
# FILENAME: feat_diag.sh
#
# SLURM job script for task-relevant feature distribution diagnostics.
# Called by feat_diag_submit.sh — do not run directly.
# NOTE: No GPU needed — this is pure numpy/scipy on images and masks.

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- NCSA Delta (CPU partition — no GPU needed) ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:59:59

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=standby
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=32G
# #SBATCH --time=1:59:59

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=32G
# #SBATCH --time=1:59:59


# ---- Modules (uncomment the one you need) ----

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

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


echo "============================================"
echo "Feature Distribution Diagnostics"
echo "============================================"
echo "Base data root:  ${BASE_DATA_ROOT}"
echo "Resolutions:     ${RESOLUTIONS}"
echo "Cities:          ${CITIES}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Splits:          ${SPLIT_A} vs ${SPLIT_B}"
echo "============================================"

# ---- Run ----
$PYTHON_BIN feature_diagnostics.py \
    --base_data_root ${BASE_DATA_ROOT} \
    --resolutions ${RESOLUTIONS} \
    --cities ${CITIES} \
    --output_dir ${OUTPUT_DIR} \
    --splits ${SPLIT_A:-train} ${SPLIT_B:-test}

echo "Done!"