#!/bin/bash
# FILENAME: mamnet_sib.sh

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- Anvil ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=${NODE}
# #SBATCH --mem=64G
# #SBATCH --time=5:59:59

# --- Gilbreth ---
# #SBATCH -A ${SLURM_ACCOUNT}
# #SBATCH --partition=${SLURM_PARTITION}
# #SBATCH --gres=gpu:1
# #SBATCH --qos=standby
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=3:59:59

# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=${NODE}
#SBATCH --time=4:29:59


# ---- Modules (uncomment the one you need) ----

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/mamnet
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

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
export PYTHONUNBUFFERED=1

# ─────────────────────────────────────────────────────────────────────────────
# Validate required variables
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "${MODE}" ] || [ -z "${OUTPUT_DIR}" ]; then
    echo "ERROR: MODE and OUTPUT_DIR must be set via --export"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Build SIB component flags from env vars
# ─────────────────────────────────────────────────────────────────────────────
SIB_FLAGS=""
[ "${USE_HAAR}"              == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_haar"
[ "${USE_VIB}"               == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_vib"
[ "${USE_CONTENT_AUG}"       == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_content_aug"
[ "${ADAPTIVE_BETA}"         == "1" ] && SIB_FLAGS="${SIB_FLAGS} --adaptive_beta"
[ "${USE_SAG}"               == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_sag"
[ "${USE_MULTISCALE_SIB}"    == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_multiscale_sib"
[ "${USE_PASSTHROUGH_GATE}"  == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_passthrough_gate"
[ "${USE_MODULE_BYPASS}"     == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_module_bypass"

# Ablation flags (A3, A5, A9, A10)
[ "${SYMMETRIC_VIB}"         == "1" ] && SIB_FLAGS="${SIB_FLAGS} --symmetric_vib"
[ "${AUG_ALL_SUBBANDS}"      == "1" ] && SIB_FLAGS="${SIB_FLAGS} --aug_all_subbands"
[ "${NO_EDGE_VIB}"           == "1" ] && SIB_FLAGS="${SIB_FLAGS} --no_edge_vib"
[ "${VIB_WRONG_SUBBAND}"     == "1" ] && SIB_FLAGS="${SIB_FLAGS} --vib_wrong_subband"
[ "${USE_CLASS_COND_TEMPSCALE}" == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_class_cond_tempscale"

# ─────────────────────────────────────────────────────────────────────────────
# SIB hyperparameter flags (only passed when explicitly overridden)
# ─────────────────────────────────────────────────────────────────────────────
HYPER_FLAGS=""
[ -n "${BETA_CONTENT}"         ] && HYPER_FLAGS="${HYPER_FLAGS} --beta_content ${BETA_CONTENT}"
[ -n "${BETA_EDGE}"            ] && HYPER_FLAGS="${HYPER_FLAGS} --beta_edge ${BETA_EDGE}"
[ -n "${NOISE_SCALE}"          ] && HYPER_FLAGS="${HYPER_FLAGS} --noise_scale ${NOISE_SCALE}"
[ -n "${BETA_MAX_MULTIPLIER}"  ] && HYPER_FLAGS="${HYPER_FLAGS} --beta_max_multiplier ${BETA_MAX_MULTIPLIER}"
[ -n "${MULTISCALE_BETA_BASE}" ] && HYPER_FLAGS="${HYPER_FLAGS} --multiscale_beta_base ${MULTISCALE_BETA_BASE}"
[ -n "${VIB_WARMUP_FRACTION}"  ] && HYPER_FLAGS="${HYPER_FLAGS} --vib_warmup_fraction ${VIB_WARMUP_FRACTION}"
[ -n "${BOUNDARY_TOLERANCE}"   ] && HYPER_FLAGS="${HYPER_FLAGS} --boundary_tolerance ${BOUNDARY_TOLERANCE}"

# ─────────────────────────────────────────────────────────────────────────────
# Data flags
# ─────────────────────────────────────────────────────────────────────────────
DATA_FLAGS=""
[ "${USE_CONTRAST}" == "1" ] && DATA_FLAGS="${DATA_FLAGS} --use_contrast"

if [ "${USE_FDA}" == "1" ]; then
    DATA_FLAGS="${DATA_FLAGS} --use_fda"
    [ -n "${FDA_L}"           ] && DATA_FLAGS="${DATA_FLAGS} --fda_L ${FDA_L}"
    [ -n "${FDA_TARGET_ROOT}" ] && DATA_FLAGS="${DATA_FLAGS} --fda_target_root ${FDA_TARGET_ROOT}"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Echo configuration
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================="
echo "  MAMNet + SIB Training Job"
echo "============================================="
echo "  MODE        : ${MODE}"
echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "  RESOLUTION  : ${RESOLUTION}"
echo "  FOLD_ID     : ${FOLD_ID}"
echo "  COMP_INF    : ${COMPARISON_INFERENCE_DIR}"
echo "  MAMNET_OUT  : ${MAMNET_OUTPUT_DIR}"
echo "  SIB_FLAGS   : ${SIB_FLAGS}"
echo "  HYPER_FLAGS : ${HYPER_FLAGS}"
echo "  DATA_FLAGS  : ${DATA_FLAGS}"
echo "  SLURM job   : ${SLURM_JOB_ID}"
echo "  Node        : $(hostname)"
echo "  GPU(s)      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================="
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Run training
# ─────────────────────────────────────────────────────────────────────────────
if [ "${MODE}" == "loco" ]; then

    $PYTHON_BIN -u train_mamnet_sib.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 25 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        --mamnet_output_dir ${MAMNET_OUTPUT_DIR} \
        ${SIB_FLAGS} \
        ${HYPER_FLAGS} \
        ${DATA_FLAGS}

elif [ "${MODE}" == "all" ]; then

    $PYTHON_BIN -u train_mamnet_sib.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 25 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        --mamnet_output_dir ${MAMNET_OUTPUT_DIR} \
        ${SIB_FLAGS} \
        ${HYPER_FLAGS} \
        ${DATA_FLAGS}

elif [ "${MODE}" == "single" ]; then

    $PYTHON_BIN -u train_mamnet_sib.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 25 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        --mamnet_output_dir ${MAMNET_OUTPUT_DIR} \
        ${SIB_FLAGS} \
        ${HYPER_FLAGS} \
        ${DATA_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi