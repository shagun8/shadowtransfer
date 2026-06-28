#!/bin/bash
# FILENAME: eval_mamnet_sib.sh
#SBATCH -A <SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=0:29:59

# ─────────────────────────────────────────────────────────────────────────────
# Environment — Anvil (active)
# ─────────────────────────────────────────────────────────────────────────────
cd ${PROJECT_ROOT}/python/mamnet
PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

module purge
module load modtree/gpu
module load cuda/12.6.1
module load anaconda
conda activate ${PROJECT_ROOT}/satmae_cuda12

# ─────────────────────────────────────────────────────────────────────────────
# Environment — Gilbreth (uncomment to swap)
# ─────────────────────────────────────────────────────────────────────────────
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12

export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────────────
# Validate required variables
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "${EVAL_MODE}" ]; then
    echo "ERROR: EVAL_MODE must be set (predictions | checkpoint | both)"
    exit 1
fi

IMG_SIZE=${IMG_SIZE:-384}
BATCH_SIZE=${BATCH_SIZE:-4}

# Optional flags — upper bound and LOCO vanilla
UB_FLAG=""
[ -n "${UB_PRED_DIR}" ]   && UB_FLAG="--ub_pred_dir ${UB_PRED_DIR}"

LOCO_FLAG=""
[ -n "${LOCO_PRED_DIR}" ] && LOCO_FLAG="--loco_pred_dir ${LOCO_PRED_DIR}"

echo "============================================="
echo "  MAMNet + SIB Evaluation"
echo "============================================="
echo "  EVAL_MODE    : ${EVAL_MODE}"
echo "  EXP_DIR      : ${EXP_DIR}"
echo "  GT_DIR       : ${GT_DIR}"
echo "  DATA_ROOT    : ${DATA_ROOT}"
echo "  UB_PRED_DIR  : ${UB_PRED_DIR:-not set}"
echo "  LOCO_PRED_DIR: ${LOCO_PRED_DIR:-not set}"
echo "  IMG_SIZE     : ${IMG_SIZE}"
echo "  SLURM job    : ${SLURM_JOB_ID}"
echo "  Node         : $(hostname)"
echo "  GPU(s)       : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────────────────────────────────────────
if [ "${EVAL_MODE}" == "predictions" ]; then

    $PYTHON_BIN -u eval_mamnet_sib.py \
        --mode predictions \
        --pred_dir "${EXP_DIR}/predictions" \
        --gt_dir   "${GT_DIR}" \
        --img_size ${IMG_SIZE} \
        ${UB_FLAG} \
        ${LOCO_FLAG}

elif [ "${EVAL_MODE}" == "checkpoint" ]; then

    $PYTHON_BIN -u eval_mamnet_sib.py \
        --mode checkpoint \
        --ckpt_path  "${EXP_DIR}/best_model.pth" \
        --data_root  "${DATA_ROOT}" \
        --img_size   ${IMG_SIZE} \
        --batch_size ${BATCH_SIZE} \
        ${UB_FLAG} \
        ${LOCO_FLAG}

elif [ "${EVAL_MODE}" == "both" ]; then

    echo "--- Step 1: Score saved predictions ---"
    $PYTHON_BIN -u eval_mamnet_sib.py \
        --mode predictions \
        --pred_dir "${EXP_DIR}/predictions" \
        --gt_dir   "${GT_DIR}" \
        --img_size ${IMG_SIZE} \
        ${UB_FLAG} \
        ${LOCO_FLAG}

    echo ""
    echo "--- Step 2: Fresh inference from checkpoint ---"
    $PYTHON_BIN -u eval_mamnet_sib.py \
        --mode checkpoint \
        --ckpt_path  "${EXP_DIR}/best_model.pth" \
        --data_root  "${DATA_ROOT}" \
        --img_size   ${IMG_SIZE} \
        --batch_size ${BATCH_SIZE} \
        ${UB_FLAG} \
        ${LOCO_FLAG}

else
    echo "ERROR: Unknown EVAL_MODE=${EVAL_MODE}  (use: predictions | checkpoint | both)"
    exit 1
fi