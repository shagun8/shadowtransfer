#!/bin/bash
# FILENAME: verify_c_star_val.sh
#
# Visual sanity check on c*_val operating points.
# Reads the augmented sp_gap_*_c4clean_*.json files (must contain
# val_rc_records and coverage_grid) and writes a 3x3 grid PNG showing
# the source-val RC curve with c*_val and the beta-target overlaid.
#
# Pure CPU work — no GPU forward passes. Should complete in under a minute.

#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0:30:00

module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python
export PYTHONUNBUFFERED=1

if [ -z "${MAMNET_RESULTS_DIR}" ] || [ -z "${OGLANET_RESULTS_DIR}" ] \
   || [ -z "${DINOV3_RESULTS_DIR}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: Required env vars missing"
    exit 1
fi

echo "============================================="
echo "  c*_val Verification"
echo "============================================="
echo "  MAMNET_RESULTS_DIR : ${MAMNET_RESULTS_DIR}"
echo "  OGLANET_RESULTS_DIR: ${OGLANET_RESULTS_DIR}"
echo "  DINOV3_RESULTS_DIR : ${DINOV3_RESULTS_DIR}"
echo "  OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "  RESOLUTION         : ${RESOLUTION:-highres}"
echo "  BETA               : ${BETA:-0.5}"
echo "  SLURM job          : ${SLURM_JOB_ID}"
echo "  Node               : $(hostname)"
echo "============================================="

$PYTHON_BIN -u verify_c_star_val.py \
    --mamnet_results_dir  ${MAMNET_RESULTS_DIR} \
    --oglanet_results_dir ${OGLANET_RESULTS_DIR} \
    --dinov3_results_dir  ${DINOV3_RESULTS_DIR} \
    --output_dir          ${OUTPUT_DIR} \
    --resolution          ${RESOLUTION:-highres} \
    --beta                ${BETA:-0.5} \
    --fallback_c          1.0