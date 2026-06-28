#!/bin/bash
# FILENAME: aggregate_coverage_recover.sh
#
# Phase 2 coverage-recovery aggregator — pure analysis, no GPU forward passes.
# Reads val_rc_records + test_rc_records from per-cell SP-gap JSONs (which
# must have been produced after applying the per-cell patch).

#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00

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
echo "  Phase 2 Coverage Recovery Aggregator"
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

$PYTHON_BIN -u aggregate_coverage_recover.py \
    --mamnet_results_dir  ${MAMNET_RESULTS_DIR} \
    --oglanet_results_dir ${OGLANET_RESULTS_DIR} \
    --dinov3_results_dir  ${DINOV3_RESULTS_DIR} \
    --output_dir          ${OUTPUT_DIR} \
    --resolution          ${RESOLUTION:-highres} \
    --bootstrap_B         10000 \
    --beta                ${BETA:-0.5} \
    --fallback_c          1.0