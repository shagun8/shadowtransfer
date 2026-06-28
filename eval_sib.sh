#!/bin/bash
# FILENAME: eval_sib.sh
#
# SLURM job script for SIB evaluation — one job per model type.
# Invoked by eval_sib_submit.sh via sbatch --export.
#
# NOTE: eval_sib.py does no model loading (predictions-only mode).
#       A GPU node is used for queue simplicity, but only CPU/RAM is needed.
#
# Required env vars (set by eval_sib_submit.sh):
#   MODEL_TYPE             — mamnet | oglanet | dinov3
#   SIB_OUTPUT_DIR         — model-specific SIB outputs dir
#   TEST_IMG_RESULTS_DIR   — root of Test_img_results/
#   GT_BASE_DIR            — root of Final_data_test/
#   EVAL_OUTPUT_DIR        — where to save eval JSON files
#   RESOLUTION             — highres | midres
#   BOUNDARY_TOLERANCE     — int, default 2
#   IMG_SIZE               — int, default 384

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=32G
# #SBATCH --time=0:59:59

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=32G
# #SBATCH --time=0:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=0:59:59


# ---- Modules + environment (uncomment the one you need) ----

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────────────
# Validate required variables
# ─────────────────────────────────────────────────────────────────────────────
for var in MODEL_TYPE SIB_OUTPUT_DIR TEST_IMG_RESULTS_DIR \
           GT_BASE_DIR EVAL_OUTPUT_DIR RESOLUTION; do
    if [ -z "${!var}" ]; then
        echo "ERROR: ${var} must be set via --export"
        exit 1
    fi
done

BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE:-2}
IMG_SIZE=${IMG_SIZE:-384}

# ─────────────────────────────────────────────────────────────────────────────
# Echo configuration
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================="
echo "  SIB Evaluation Job"
echo "============================================="
echo "  MODEL_TYPE           : ${MODEL_TYPE}"
echo "  RESOLUTION           : ${RESOLUTION}"
echo "  BOUNDARY_TOLERANCE   : ${BOUNDARY_TOLERANCE}"
echo "  IMG_SIZE             : ${IMG_SIZE}"
echo "  SIB_OUTPUT_DIR       : ${SIB_OUTPUT_DIR}"
echo "  TEST_IMG_RESULTS_DIR : ${TEST_IMG_RESULTS_DIR}"
echo "  GT_BASE_DIR          : ${GT_BASE_DIR}"
echo "  EVAL_OUTPUT_DIR      : ${EVAL_OUTPUT_DIR}"
echo "  SLURM job            : ${SLURM_JOB_ID}"
echo "  Node                 : $(hostname)"
echo "============================================="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────────────────────────────────────
$PYTHON_BIN -u eval_sib.py \
    --model_type          "${MODEL_TYPE}" \
    --sib_output_dir      "${SIB_OUTPUT_DIR}" \
    --test_img_results_dir "${TEST_IMG_RESULTS_DIR}" \
    --gt_base_dir         "${GT_BASE_DIR}" \
    --output_dir          "${EVAL_OUTPUT_DIR}" \
    --resolution          "${RESOLUTION}" \
    --boundary_tolerance  "${BOUNDARY_TOLERANCE}" \
    --img_size            "${IMG_SIZE}"

echo ""
echo "Job complete!"