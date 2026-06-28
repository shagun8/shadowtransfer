#!/bin/bash
# FILENAME: beta_sweep.sh
#
# Robustness sweep: runs aggregate_coverage_recover.py at β ∈ {0.3, 0.5, 0.7}
# sequentially in one job, then summarizes all three with beta_sweep_summarize.py.
# Output dir layout:
#   ${SWEEP_OUTPUT_DIR}/beta_0.3/coverage_recover_results.json
#   ${SWEEP_OUTPUT_DIR}/beta_0.5/coverage_recover_results.json
#   ${SWEEP_OUTPUT_DIR}/beta_0.7/coverage_recover_results.json
#   ${SWEEP_OUTPUT_DIR}/beta_sweep_summary.md
#   ${SWEEP_OUTPUT_DIR}/beta_sweep_summary.json
#
# Pure CPU work; should complete in 5-10 minutes total.

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
   || [ -z "${DINOV3_RESULTS_DIR}" ] || [ -z "${SWEEP_OUTPUT_DIR}" ]; then
    echo "ERROR: Required env vars missing"
    exit 1
fi

echo "============================================="
echo "  β Robustness Sweep — Coverage Recovery"
echo "============================================="
echo "  MAMNET_RESULTS_DIR : ${MAMNET_RESULTS_DIR}"
echo "  OGLANET_RESULTS_DIR: ${OGLANET_RESULTS_DIR}"
echo "  DINOV3_RESULTS_DIR : ${DINOV3_RESULTS_DIR}"
echo "  SWEEP_OUTPUT_DIR   : ${SWEEP_OUTPUT_DIR}"
echo "  RESOLUTION         : ${RESOLUTION:-highres}"
echo "  BETAS              : 0.3, 0.5, 0.7"
echo "  SLURM job          : ${SLURM_JOB_ID}"
echo "  Node               : $(hostname)"
echo "============================================="

# --- Loop over beta values ---
for BETA in 0.3 0.5 0.7; do
    echo ""
    echo "=========================================="
    echo "  Running aggregator at β=${BETA}"
    echo "=========================================="
    OUT_SUB="${SWEEP_OUTPUT_DIR}/beta_${BETA}"
    mkdir -p "${OUT_SUB}"

    $PYTHON_BIN -u aggregate_coverage_recover.py \
        --mamnet_results_dir  ${MAMNET_RESULTS_DIR} \
        --oglanet_results_dir ${OGLANET_RESULTS_DIR} \
        --dinov3_results_dir  ${DINOV3_RESULTS_DIR} \
        --output_dir          ${OUT_SUB} \
        --resolution          ${RESOLUTION:-highres} \
        --bootstrap_B         10000 \
        --beta                ${BETA} \
        --fallback_c          1.0

    if [ $? -ne 0 ]; then
        echo "ERROR: aggregator failed at β=${BETA}"
        exit 1
    fi
done

# --- Build unified summary ---
echo ""
echo "=========================================="
echo "  Building β-sweep unified summary"
echo "=========================================="

$PYTHON_BIN -u beta_sweep_summarize.py \
    --sweep_dir ${SWEEP_OUTPUT_DIR} \
    --betas 0.3 0.5 0.7

echo ""
echo "Done. See ${SWEEP_OUTPUT_DIR}/beta_sweep_summary.md"