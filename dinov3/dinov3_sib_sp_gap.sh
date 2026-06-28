#!/bin/bash
# FILENAME: dinov3_sib_sp_gap.sh
#
# SLURM job script for Phase 1 SP-gap measurement on DINOv3 C4-clean
# (inference only, no training).

# ---- SBATCH directives ----

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --exclude=${NODE}
#SBATCH --time=0:15:00

# --- Modules ---
module purge
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/dinov3
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export PYTHONUNBUFFERED=1

# ---- Validate required variables ----
if [ -z "${FOLD_ID}" ] || [ -z "${C4CLEAN_CHECKPOINT}" ]; then
    echo "ERROR: FOLD_ID and C4CLEAN_CHECKPOINT must be set via --export"
    exit 1
fi

echo "============================================="
echo "  DINOv3 SP-Gap Phase 1 Analysis (C4-clean only)"
echo "============================================="
echo "  FOLD_ID:            ${FOLD_ID}"
echo "  RESOLUTION:         ${RESOLUTION}"
echo "  C4CLEAN_CHECKPOINT: ${C4CLEAN_CHECKPOINT}"
echo "  BASE_DATA_ROOT:     ${BASE_DATA_ROOT}"
echo "  WEIGHT_DIR:         ${WEIGHT_DIR}"
echo "  OUTPUT_DIR:         ${OUTPUT_DIR}"
echo "  SLURM job:          ${SLURM_JOB_ID}"
echo "  Node:               $(hostname)"
echo "  GPU:                $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================="

$PYTHON_BIN -u sp_gap_analysis.py \
    --c4clean_checkpoint ${C4CLEAN_CHECKPOINT} \
    --base_data_root ${BASE_DATA_ROOT} \
    --fold_id ${FOLD_ID} \
    --resolution ${RESOLUTION} \
    --output_dir ${OUTPUT_DIR} \
    --weights_path ${WEIGHT_DIR} \
    --batch_size 8 \
    --num_workers 1 \
    --bootstrap_B 10000 \
    --min_shadow_pixels 5 \
    --n_coverage 20 \
    --device cuda