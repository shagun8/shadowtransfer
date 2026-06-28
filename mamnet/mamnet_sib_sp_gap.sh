#!/bin/bash
# FILENAME: mamnet_sib_sp_gap.sh
#
# SLURM job script for Phase 1 SP-gap measurement (inference only, no training).
# Called by mamnet_sib_sp_gap_submit.sh with env vars for each fold.
#
# Required env vars (set via --export in sbatch):
#   FOLD_ID               — 0 (phoenix), 1 (miami), 2 (chicago)
#   RESOLUTION            — highres or midres
#   VANILLA_CHECKPOINT    — path to checkpoint_best.pth from Vanilla LOCO run
#   C4CLEAN_CHECKPOINT    — path to best_model.pth from C4-clean SIB run
#   BASE_DATA_ROOT        — path to Final_data_test/
#   OUTPUT_DIR            — where to write sp_gap_mamnet_{city}_{res}.json

# ---- SBATCH directives ----

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0:45:00

# --- Modules ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/mamnet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

export MASTER_ADDR=localhost
export MASTER_PORT=12355
export PYTHONUNBUFFERED=1

# ---- Validate required variables ----
if [ -z "${FOLD_ID}" ] || [ -z "${VANILLA_CHECKPOINT}" ] || [ -z "${C4CLEAN_CHECKPOINT}" ]; then
    echo "ERROR: FOLD_ID, VANILLA_CHECKPOINT, C4CLEAN_CHECKPOINT must be set via --export"
    exit 1
fi

echo "============================================="
echo "  MAMNet SP-Gap Phase 1 Analysis"
echo "============================================="
echo "  FOLD_ID:            ${FOLD_ID}"
echo "  RESOLUTION:         ${RESOLUTION}"
echo "  VANILLA_CHECKPOINT: ${VANILLA_CHECKPOINT}"
echo "  C4CLEAN_CHECKPOINT: ${C4CLEAN_CHECKPOINT}"
echo "  OUTPUT_DIR:         ${OUTPUT_DIR}"
echo "  SLURM job:          ${SLURM_JOB_ID}"
echo "  Node:               $(hostname)"
echo "  GPU:                $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================="

$PYTHON_BIN -u sp_gap_analysis.py \
    --vanilla_checkpoint ${VANILLA_CHECKPOINT} \
    --c4clean_checkpoint ${C4CLEAN_CHECKPOINT} \
    --base_data_root ${BASE_DATA_ROOT} \
    --fold_id ${FOLD_ID} \
    --resolution ${RESOLUTION} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_workers 1 \
    --bootstrap_B 10000 \
    --min_shadow_pixels 5 \
    --n_coverage 20 \
    --device cuda