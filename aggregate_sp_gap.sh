#!/bin/bash
# FILENAME: aggregate_sp_gap.sh
#
# Runs aggregate_sp_gap.py — pure analysis, no GPU needed.
# Reads C4-clean JSONs from each architecture's sp_gap_results dir,
# loads pre-saved .npy probability maps for Vanilla + 6 adaptation methods,
# computes paired bootstrap CIs and population-level Wilcoxon, writes JSON
# + Markdown summary.

# ---- SBATCH directives ----

# --- NCSA Delta (CPU partition is fine) ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1:00:00

# --- Modules ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

export PYTHONUNBUFFERED=1

# ---- Validate ----
if [ -z "${MAMNET_RESULTS_DIR}" ] || [ -z "${OGLANET_RESULTS_DIR}" ] \
   || [ -z "${DINOV3_RESULTS_DIR}" ] || [ -z "${PROBS_ROOT}" ] \
   || [ -z "${DATA_ROOT}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: Required env vars missing"
    exit 1
fi

echo "============================================="
echo "  Phase 1 SP-Gap Aggregator"
echo "============================================="
echo "  MAMNET_RESULTS_DIR : ${MAMNET_RESULTS_DIR}"
echo "  OGLANET_RESULTS_DIR: ${OGLANET_RESULTS_DIR}"
echo "  DINOV3_RESULTS_DIR : ${DINOV3_RESULTS_DIR}"
echo "  PROBS_ROOT         : ${PROBS_ROOT}"
echo "  DATA_ROOT          : ${DATA_ROOT}"
echo "  OUTPUT_DIR         : ${OUTPUT_DIR}"
echo "  RESOLUTION         : ${RESOLUTION:-highres}"
echo "  SLURM job          : ${SLURM_JOB_ID}"
echo "  Node               : $(hostname)"
echo "============================================="

$PYTHON_BIN -u aggregate_sp_gap.py \
    --mamnet_results_dir ${MAMNET_RESULTS_DIR} \
    --oglanet_results_dir ${OGLANET_RESULTS_DIR} \
    --dinov3_results_dir ${DINOV3_RESULTS_DIR} \
    --probs_root ${PROBS_ROOT} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --resolution ${RESOLUTION:-highres} \
    --bootstrap_B 10000 \
    --n_coverage 20 \
    --min_shadow_pixels 5 \
    --img_size 384