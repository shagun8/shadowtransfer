#!/bin/bash
# FILENAME: sib_ablation_analysis_v2.sh
#
# SLURM job script for sib_ablation_analysis_v2.py.
# Called by sib_ablation_analysis_v2_submit.sh.

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
# #SBATCH --time=3:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:29:59

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
# Distributed env vars (parity with other jobs; not strictly needed here)
# =====================================================================
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# =====================================================================
# Defaults if not provided via --export
# =====================================================================
: "${BASE_PATH:=${PROJECT_ROOT}/}"
: "${OUTPUT_DIR:=${BASE_PATH}/data/ablation_analysis_v2}"

echo "============================================="
echo "  SIB Ablation Analysis V2"
echo "============================================="
echo "  BASE_PATH   : ${BASE_PATH}"
echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "  SLURM job   : ${SLURM_JOB_ID}"
echo "  Node        : $(hostname)"
echo "============================================="
echo ""

$PYTHON_BIN -u sib_ablation_analysis_v2.py \
    --base_path "${BASE_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --boundary_tolerance 2 \
    --img_size 384 \
    --n_bootstrap 10000 \
    --alpha 0.05 \
    "$@"