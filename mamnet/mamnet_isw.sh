#!/bin/bash
# FILENAME: mamnet_isw.sh
# MAMNet + ISW (Instance Selective Whitening) training job.

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=0:59:59

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=5:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59
#SBATCH --job-name=satmae_run

# ---- Modules (uncomment the one you need) ----

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/mamnet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# Distributed training env vars (single node)
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# Force unbuffered Python output (critical for SLURM log capture)
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

COMPARISON_FLAGS=""
if [ -n "${COMPARISON_INFERENCE_DIR}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_inference_dir ${COMPARISON_INFERENCE_DIR}"
fi
if [ -n "${COMPARISON_DATA_ROOT}" ]; then
    COMPARISON_FLAGS="${COMPARISON_FLAGS} --comparison_data_root ${COMPARISON_DATA_ROOT}"
fi

ISW_LAMBDA_FLAG=""
if [ -n "${ISW_LAMBDA}" ]; then
    ISW_LAMBDA_FLAG="--isw_lambda ${ISW_LAMBDA}"
fi

echo "Mode:              ${MODE}"
echo "Contrast:          ${CONTRAST_FLAG}"
echo "Tolerant eval:     ${TOLERANT_FLAG}"
echo "Boundary tolerance:${BOUNDARY_TOL_FLAG}"
echo "Early stopping:    ${EARLY_STOP_FLAG}"
echo "ISW mask dir:      ${ISW_MASK_DIR}"
echo "ISW lambda:        ${ISW_LAMBDA_FLAG}"
echo "Comparison:        ${COMPARISON_FLAGS}"

# ---- Debug checks ----
echo ""
echo "=== DEBUG CHECKS ==="
echo "Working dir: $(pwd)"
echo "Python bin:  ${PYTHON_BIN}"
echo "Python version: $(${PYTHON_BIN} --version 2>&1)"

if [ ! -f "train_isw.py" ]; then
    echo "ERROR: train_isw.py not found in $(pwd)"
    ls -la *.py 2>/dev/null | head -20
    exit 1
fi
echo "train_isw.py found: OK"

if [ ! -f "utils/isw_loss.py" ]; then
    echo "ERROR: utils/isw_loss.py not found"
    exit 1
fi
echo "utils/isw_loss.py found: OK"

if [ ! -f "utils/visualization_isw.py" ]; then
    echo "ERROR: utils/visualization_isw.py not found"
    exit 1
fi
echo "utils/visualization_isw.py found: OK"

if [ -n "${ISW_MASK_DIR}" ] && [ ! -d "${ISW_MASK_DIR}" ]; then
    echo "ERROR: ISW mask dir not found: ${ISW_MASK_DIR}"
    exit 1
fi
echo "ISW mask dir found: OK"
echo "Mask files:"
ls -la ${ISW_MASK_DIR}/*.npy 2>&1

# Full import check — test ALL imports that train_isw.py uses at top level
echo ""
echo "Testing ALL train_isw.py imports..."
${PYTHON_BIN} -c "
import sys; sys.path.insert(0, '.')
print('  importing models.mamnet...')
from models.mamnet import MAMNet
print('  importing data.dataset...')
from data.dataset import get_dataloaders
print('  importing data.dataset_enhanced...')
from data.dataset_enhanced import ShadowDatasetEnhanced
print('  importing utils.evaluation_detailed...')
from utils.evaluation_detailed import DetailedEvaluator
print('  importing utils.losses...')
from utils.losses import MAMNetLoss
print('  importing utils.metrics...')
from utils.metrics import ShadowMetrics
print('  importing utils.postprocessing...')
from utils.postprocessing import filter_small_predictions
print('  importing utils.isw_loss...')
from utils.isw_loss import ISWLoss, EncoderFeatureHooks
print('  importing utils.visualization_isw...')
from utils.visualization_isw import plot_loss_curves_isw, plot_metrics_curves, save_best_worst_visualizations
print('ALL imports OK')
" 2>&1
IMPORT_RC=$?
if [ $IMPORT_RC -ne 0 ]; then
    echo "ERROR: Import check failed with exit code ${IMPORT_RC}"
    exit 1
fi

echo "=== END DEBUG CHECKS ==="
echo ""

# ---- Smoke test: can Python produce ANY output at all? ----
echo "Smoke test..."
${PYTHON_BIN} -u -c "print('SMOKE TEST OK')" 2>&1
echo "Smoke test exit code: $?"

echo "Script smoke test..."
${PYTHON_BIN} -u -c "
import sys; sys.path.insert(0, '.')
import train_isw
print('MODULE LOADED - has main:', hasattr(train_isw, 'main'))
" 2>&1
echo "Script smoke test exit code: $?"
echo ""

# ---- Run training (NO eval — direct execution) ----
echo "Launching training..."

if [ "$MODE" == "single" ]; then
    ${PYTHON_BIN} -u train_isw.py \
        --mode single \
        --data_root "${DATA_ROOT}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${COMPARISON_FLAGS} 2>&1
    TRAIN_RC=$?

elif [ "$MODE" == "loco" ]; then
    ${PYTHON_BIN} -u train_isw.py \
        --mode loco \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --fold_id "${FOLD_ID}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${COMPARISON_FLAGS} 2>&1
    TRAIN_RC=$?

elif [ "$MODE" == "all" ]; then
    ${PYTHON_BIN} -u train_isw.py \
        --mode all \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers 1 \
        --isw_mask_dir "${ISW_MASK_DIR}" \
        ${ISW_LAMBDA_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${COMPARISON_FLAGS} 2>&1
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