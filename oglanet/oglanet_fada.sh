#!/bin/bash
# FILENAME: oglanet_fada.sh
#
# SLURM batch script for OGLANetFADA training.
# Supports single-city, all-city, and LOCO modes.
#
# Environment variables consumed (set via --export in submit script):
#   MODE                     single | all | loco
#   DATA_ROOT                (single mode) path to city/resolution/ dir
#   BASE_DATA_ROOT           (all / loco)  path to data root
#   RESOLUTION               highres | midres
#   FOLD_ID                  0 | 1 | 2     (loco mode only)
#   OUTPUT_DIR               where to write checkpoints / plots
#   USE_CONTRAST             1 = use 4-ch RGBC input  (default: off)
#   EVAL_TOLERANT            1 = tolerant mIOU as decision metric
#   BOUNDARY_TOLERANCE       don't-care band half-width in pixels (default: 2)
#   EARLY_STOPPING_PATIENCE  patience in epochs  (0 / unset = disabled)
#   FADA_RANK                LoRA rank r  (default: 16)
#   FADA_TOKEN_LENGTH        token length m  (default: 100)
#   FADA_STAGES              space-separated stage list  (default: "3 4 5")
#   LR                       learning rate  (default: 1e-4)
#   LR_FADA                  separate LR for FADA adapters  (default: same as LR)
#   LR_DECODER               separate LR for DFFM/Dec/OAM  (default: same as LR)
#   WEIGHT_DECAY             Adam weight decay  (default: 1e-4)

# ============================================================================
# SBATCH directives — uncomment the server block you are targeting
# ============================================================================

# ---- Gilbreth (ACTIVE) ----
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=128G
# #SBATCH --time=0:29:59

# ---- Anvil ----
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=5:59:59

# ---- NCSA Delta ----
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=5:29:59

# ============================================================================
# Module loads & paths — uncomment the server block you are targeting
# ============================================================================

# ---- Gilbreth (ACTIVE) ----
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# ---- Anvil ----
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/oglanet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# ---- NCSA Delta ----
module purge
module load cudatoolkit
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/oglanet
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# ============================================================================
# Distributed training env (single node / single GPU)
# ============================================================================
export MASTER_ADDR=localhost
export MASTER_PORT=12356
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# ============================================================================
# Build optional flags from environment variables
# ============================================================================

# --- Contrast channel ---
CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

# --- Boundary-tolerant evaluation ---
TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

BOUNDARY_TOL_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOL_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

# --- Early stopping ---
EARLY_STOP_FLAG=""
if [ -n "${EARLY_STOPPING_PATIENCE}" ] && [ "${EARLY_STOPPING_PATIENCE}" != "0" ]; then
    EARLY_STOP_FLAG="--early_stopping_patience ${EARLY_STOPPING_PATIENCE}"
fi

# --- FADA hyperparameters ---
# Defaults follow Bi et al. NeurIPS 2024:
#   rank=16   (Table 3: best at 16-32)
#   m=100     (Fig 8: stable in 75-125)
#   stages="3 4 5" (feat3=256ch, feat4=512ch, feat5=1024ch in OGLANet)
FADA_RANK_FLAG=""
if [ -n "${FADA_RANK}" ]; then
    FADA_RANK_FLAG="--fada_rank ${FADA_RANK}"
fi

FADA_TOKEN_FLAG=""
if [ -n "${FADA_TOKEN_LENGTH}" ]; then
    FADA_TOKEN_FLAG="--fada_token_length ${FADA_TOKEN_LENGTH}"
fi

FADA_STAGES_FLAG=""
if [ -n "${FADA_STAGES}" ]; then
    FADA_STAGES_FLAG="--fada_stages ${FADA_STAGES}"
fi

# --- Learning rates ---
LR_FLAG=""
if [ -n "${LR}" ]; then
    LR_FLAG="--lr ${LR}"
fi

LR_FADA_FLAG=""
if [ -n "${LR_FADA}" ]; then
    LR_FADA_FLAG="--lr_fada ${LR_FADA}"
fi

LR_DECODER_FLAG=""
if [ -n "${LR_DECODER}" ]; then
    LR_DECODER_FLAG="--lr_decoder ${LR_DECODER}"
fi

WEIGHT_DECAY_FLAG=""
if [ -n "${WEIGHT_DECAY}" ]; then
    WEIGHT_DECAY_FLAG="--weight_decay ${WEIGHT_DECAY}"
fi

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "OGLANetFADA Training Configuration"
echo "=========================================="
echo "Mode:               ${MODE}"
echo "Contrast:           ${CONTRAST_FLAG:-off}"
echo "Tolerant eval:      ${TOLERANT_FLAG:-off}"
echo "Boundary tolerance: ${BOUNDARY_TOL_FLAG:-default(2)}"
echo "Early stopping:     ${EARLY_STOP_FLAG:-disabled}"
echo "FADA rank:          ${FADA_RANK_FLAG:-default(16)}"
echo "FADA token length:  ${FADA_TOKEN_FLAG:-default(100)}"
echo "FADA stages:        ${FADA_STAGES_FLAG:-default(3 4 5)}"
echo "LR (base):          ${LR_FLAG:-default(1e-4)}"
echo "LR (FADA):          ${LR_FADA_FLAG:-same as base}"
echo "LR (decoder):       ${LR_DECODER_FLAG:-same as base}"
echo "Weight decay:       ${WEIGHT_DECAY_FLAG:-default(1e-4)}"
echo "=========================================="

# ============================================================================
# Run training
# ============================================================================

if [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_fada.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 100 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${LR_FLAG} \
        ${LR_FADA_FLAG} \
        ${LR_DECODER_FLAG} \
        ${WEIGHT_DECAY_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${FADA_RANK_FLAG} \
        ${FADA_TOKEN_FLAG} \
        ${FADA_STAGES_FLAG}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_fada.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 8 \
        --epochs 100 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${LR_FLAG} \
        ${LR_FADA_FLAG} \
        ${LR_DECODER_FLAG} \
        ${WEIGHT_DECAY_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${FADA_RANK_FLAG} \
        ${FADA_TOKEN_FLAG} \
        ${FADA_STAGES_FLAG}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_fada.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --batch_size 8 \
        --epochs 100 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        ${LR_FLAG} \
        ${LR_FADA_FLAG} \
        ${LR_DECODER_FLAG} \
        ${WEIGHT_DECAY_FLAG} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG} \
        ${BOUNDARY_TOL_FLAG} \
        ${EARLY_STOP_FLAG} \
        ${FADA_RANK_FLAG} \
        ${FADA_TOKEN_FLAG} \
        ${FADA_STAGES_FLAG}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi