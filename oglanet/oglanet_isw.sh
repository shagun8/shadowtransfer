#!/bin/bash
# FILENAME: oglanet_isw.sh
# OGLANet + ISW (Instance Selective Whitening) training job.
# Prerequisites: ISW masks must be precomputed
#                (run compute_isw_masks_oglanet_submit.sh first).

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:29:59

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=128G
# #SBATCH --time=5:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59
#SBATCH --job-name=oglanet_isw

# ---- Modules (uncomment the one you need) ----

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/oglanet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# Distributed env vars (single node)
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

export PYTHONUNBUFFERED=1

# ---- Build optional flags from env vars ----
CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

BOUNDARY_TOL_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOL_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

EARLY_STOP_FLAG=""
if [ -n "${EARLY_STOPPING_PATIENCE}" ] && [ "${EARLY_STOPPING_PATIENCE}" != "0" ]; then
    EARLY_STOP_FLAG="--early_stopping_patience ${EARLY_STOPPING_PATIENCE}"
fi

ISW_LAMBDA_FLAG=""
if [ -n "${ISW_LAMBDA}" ]; then
    ISW_LAMBDA_FLAG="--isw_lambda ${ISW_LAMBDA}"
fi

echo "Mode:               ${MODE}"
echo "Contrast:           ${CONTRAST_FLAG}"
echo "Tolerant eval:      ${TOLERANT_FLAG}"
echo "Boundary tolerance: ${BOUNDARY_TOL_FLAG}"
echo "Early stopping:     ${EARLY_STOP_FLAG}"
echo "ISW mask dir:       ${ISW_MASK_DIR}"
echo "ISW lambda:         ${ISW_LAMBDA_FLAG}"

# ---- Debug checks ----
echo ""
echo "=== DEBUG CHECKS ==="
echo "Working dir: $(pwd)"
echo "Python:      ${PYTHON_BIN}"
echo "Python version: $(${PYTHON_BIN} --version 2>&1)"

if [ ! -f "train_oglanet_isw.py" ]; then
    echo "ERROR: train_oglanet_isw.py not found in $(pwd)"
    exit 1
fi
echo "train_oglanet_isw.py: OK"

if [ ! -f "utils/isw_loss.py" ]; then
    echo "ERROR: utils/isw_loss.py not found (copy from mamnet/utils/)"
    exit 1
fi
echo "utils/isw_loss.py: OK"

if [ ! -f "utils/visualization_oglanet_isw.py" ]; then
    echo "ERROR: utils/visualization_oglanet_isw.py not found"
    exit 1
fi
echo "utils/visualization_oglanet_isw.py: OK"

if [ -n "${ISW_MASK_DIR}" ] && [ ! -d "${ISW_MASK_DIR}" ]; then
    echo "ERROR: ISW mask dir not found: ${ISW_MASK_DIR}"
    echo "       Run compute_isw_masks_oglanet_submit.sh first!"
    exit 1
fi
echo "ISW mask dir found: OK"
echo "Mask files:"
ls -la ${ISW_MASK_DIR}/*.npy 2>&1

# Full import check
echo ""
echo "Testing all imports for train_oglanet_isw.py..."
${PYTHON_BIN} -c "
import sys; sys.path.insert(0, '.')
print('  importing models.oglanet...')
from models.oglanet import OGLANet
print('  importing data.dataset...')
from data.dataset import get_dataloaders
print('  importing utils.evaluation_detailed...')
from utils.evaluation_detailed import DetailedEvaluator
print('  importing utils.losses...')
from utils.losses import OGLANetLoss
print('  importing utils.metrics...')
from utils.metrics import ShadowMetrics
print('  importing utils.postprocessing...')
from utils.postprocessing import filter_small_predictions
print('  importing utils.isw_loss...')
from utils.isw_loss import ISWLoss, EncoderFeatureHooks
print('  importing utils.visualization_oglanet_isw...')
from utils.visualization_oglanet_isw import (
    plot_loss_curves_oglanet_isw, plot_metrics_curves,
    save_best_worst_visualizations)
print('ALL imports OK')
" 2>&1
IMPORT_RC=$?
if [ $IMPORT_RC -ne 0 ]; then
    echo "ERROR: Import check failed (exit code ${IMPORT_RC})"
    exit 1
fi
echo "=== END DEBUG CHECKS ==="
echo ""

# ---- Smoke test ----
echo "Smoke test..."
${PYTHON_BIN} -u -c "print('SMOKE TEST OK')" 2>&1
echo "Smoke test exit code: $?"
echo ""

# ---- Run training ----
echo "Launching OGLANet + ISW training..."

if [ "$MODE" == "single" ]; then
    ${PYTHON_BIN} -u train_oglanet_isw.py \
        --mode single \
        --data_root "${DATA_ROOT}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --optimizer adamax \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} 2>&1
    TRAIN_RC=$?

elif [ "$MODE" == "loco" ]; then
    ${PYTHON_BIN} -u train_oglanet_isw.py \
        --mode loco \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --fold_id "${FOLD_ID}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --optimizer adamax \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} 2>&1
    TRAIN_RC=$?

elif [ "$MODE" == "all" ]; then
    ${PYTHON_BIN} -u train_oglanet_isw.py \
        --mode all \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --optimizer adamax \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} 2>&1
    TRAIN_RC=$?

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi

echo ""
echo "=== Python exited with code: ${TRAIN_RC} ==="
if [ ${TRAIN_RC} -ne 0 ]; then
    echo "ERROR: Training failed!"
fi
exit ${TRAIN_RC}