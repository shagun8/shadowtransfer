#!/bin/bash
# FILENAME: dinov3_sib.sh
#
# SLURM job script for DINOv3 + SIB training.
# Called by dinov3_sib_submit.sh / dinov3_sib_ablation_submit.sh
# with env vars for each configuration.
#
# SIB component flags (passed as env vars, 0 or 1):
#   USE_HAAR             — Haar wavelet decomposition
#   USE_VIB              — Variational Information Bottleneck
#   USE_CONTENT_AUG      — Content augmentation (training only)
#   ADAPTIVE_BETA        — Intensity-adaptive beta in VIB
#   USE_PASSTHROUGH_GATE — Learned passthrough gate on VIB output
#   USE_MODULE_BYPASS    — Module-level residual bypass gate
#   USE_FDA              — Fourier Domain Adaptation (input-level)
#
# SIB ablation flags (§5.3 component ablation, 0 or 1):
#   DISABLE_CONTENT_VIB  — A1: skip content VIB on F_LL
#   SYMMETRIC_VIB        — A3: high-β VIB on LL, LH, HL
#   AUG_ALL_SUBBANDS     — A6: augment all subbands
#   VIB_ON_HL_ONLY       — A10: content VIB on F_HL (wrong subband)
#
# SIB hyper-parameters (passed as env vars):
#   VIB_BETA_CONTENT — beta for content/uniform VIB   (default 0.01)
#   VIB_BETA_EDGE    — beta for edge VIB              (default 0.0001)
#   VIB_BETA_SCALE   — adaptive beta range            (default 0.02)
#   LAMBDA_CONTENT   — weight for content KL loss     (default 1.0)
#   LAMBDA_EDGE      — weight for edge KL loss        (default 0.1)
#   VIB_WARMUP_FRAC  — fraction of epochs for warmup  (default 0.1)
#   FDA_L            — FDA low-freq swap range         (default 0.01)
#   AUG_P_MIX        — cross-domain mixing prob       (default 0.3)
#
# BOUNDARY_TOLERANCE — ±K px don't-care zone          (default 2)
# DINOV3_OUTPUT_DIR  — root scanned for completed donor experiment

# =====================================================================
# SBATCH directives — uncomment the block for your target server
# =====================================================================

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
#SBATCH --exclude=${NODE}
#SBATCH --time=4:29:59

# =====================================================================
# Modules + paths — uncomment the block for your target server
# =====================================================================

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${PROJECT_ROOT}/satmae_cuda12
# cd ${PROJECT_ROOT}/python/dinov3
# PYTHON_BIN=${PROJECT_ROOT}/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load cudatoolkit/25.3_12.8
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/dinov3
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python

# =====================================================================
# Distributed training env vars (single node)
# =====================================================================
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# =====================================================================
# Build SIB flags from env vars
# =====================================================================
SIB_FLAGS=""

# Core component toggles
[ "${USE_HAAR}"              == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_haar"
[ "${USE_VIB}"               == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_vib"
[ "${USE_CONTENT_AUG}"       == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_content_aug"
[ "${ADAPTIVE_BETA}"         == "1" ] && SIB_FLAGS="${SIB_FLAGS} --adaptive_beta"
[ "${USE_PASSTHROUGH_GATE}"  == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_passthrough_gate"
[ "${USE_MODULE_BYPASS}"     == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_module_bypass"

# Ablation flags (§5.3)
[ "${DISABLE_CONTENT_VIB}"   == "1" ] && SIB_FLAGS="${SIB_FLAGS} --disable_content_vib"
[ "${SYMMETRIC_VIB}"         == "1" ] && SIB_FLAGS="${SIB_FLAGS} --symmetric_vib"
[ "${AUG_ALL_SUBBANDS}"      == "1" ] && SIB_FLAGS="${SIB_FLAGS} --aug_all_subbands"
[ "${VIB_ON_HL_ONLY}"        == "1" ] && SIB_FLAGS="${SIB_FLAGS} --vib_on_hl_only"
[ "${USE_CLASS_COND_TEMPSCALE}" == "1" ] && SIB_FLAGS="${SIB_FLAGS} --use_class_cond_tempscale"

# ---- VIB hyper-parameters ----
HYPER_FLAGS=""

[ -n "${VIB_BETA_CONTENT}" ] && HYPER_FLAGS="${HYPER_FLAGS} --vib_beta_content ${VIB_BETA_CONTENT}"
[ -n "${VIB_BETA_EDGE}"    ] && HYPER_FLAGS="${HYPER_FLAGS} --vib_beta_edge ${VIB_BETA_EDGE}"
[ -n "${VIB_BETA_SCALE}"   ] && HYPER_FLAGS="${HYPER_FLAGS} --vib_beta_scale ${VIB_BETA_SCALE}"
[ -n "${LAMBDA_CONTENT}"   ] && HYPER_FLAGS="${HYPER_FLAGS} --lambda_content ${LAMBDA_CONTENT}"
[ -n "${LAMBDA_EDGE}"      ] && HYPER_FLAGS="${HYPER_FLAGS} --lambda_edge ${LAMBDA_EDGE}"
[ -n "${VIB_WARMUP_FRAC}"  ] && HYPER_FLAGS="${HYPER_FLAGS} --vib_warmup_fraction ${VIB_WARMUP_FRAC}"
[ -n "${AUG_P_MIX}"        ] && HYPER_FLAGS="${HYPER_FLAGS} --aug_p_mix ${AUG_P_MIX}"

# Boundary tolerance (defaults to 2 in Python if not set)
[ -n "${BOUNDARY_TOLERANCE}" ] && HYPER_FLAGS="${HYPER_FLAGS} --boundary_tolerance ${BOUNDARY_TOLERANCE}"
[ -n "${EXP_TAG}" ] && HYPER_FLAGS="${HYPER_FLAGS} --exp_tag ${EXP_TAG}"

# ---- FDA flags ----
FDA_FLAGS=""

if [ "${USE_FDA}" == "1" ]; then
    FDA_FLAGS="${FDA_FLAGS} --use_fda"
    [ -n "${FDA_TARGET_ROOT}" ] && FDA_FLAGS="${FDA_FLAGS} --fda_target_root ${FDA_TARGET_ROOT}"
    [ -n "${FDA_L}"           ] && FDA_FLAGS="${FDA_FLAGS} --fda_L ${FDA_L}"
fi

# Donor directory flag (Strategy 1 comparison)
DONOR_FLAGS=""
[ -n "${DINOV3_OUTPUT_DIR}" ] && DONOR_FLAGS="${DONOR_FLAGS} --dinov3_output_dir ${DINOV3_OUTPUT_DIR}"

echo "================================================="
echo "DINOv3 + SIB Training"
echo "================================================="
echo "MODE:             ${MODE}"
echo "FOLD_ID:          ${FOLD_ID}"
echo "RESOLUTION:       ${RESOLUTION}"
echo "SIB flags:        ${SIB_FLAGS}"
echo "Hyper flags:      ${HYPER_FLAGS}"
echo "FDA flags:        ${FDA_FLAGS}"
echo "Donor flags:      ${DONOR_FLAGS}"
echo "BOUNDARY_TOL:     ${BOUNDARY_TOLERANCE:-2 (default)}"
echo "DINOV3_OUT:       ${DINOV3_OUTPUT_DIR}"
echo "GATE:             ${USE_PASSTHROUGH_GATE:-0}"
echo "MODULE_BYPASS:    ${USE_MODULE_BYPASS:-0}"
echo "EXP_TAG:          ${EXP_TAG:-none}"
echo "--- Ablation flags ---"
echo "DISABLE_CONTENT_VIB: ${DISABLE_CONTENT_VIB:-0}"
echo "SYMMETRIC_VIB:       ${SYMMETRIC_VIB:-0}"
echo "AUG_ALL_SUBBANDS:    ${AUG_ALL_SUBBANDS:-0}"
echo "VIB_ON_HL_ONLY:      ${VIB_ON_HL_ONLY:-0}"
echo "AUG_P_MIX:           ${AUG_P_MIX:-0.3 (default)}"
echo "================================================="

# =====================================================================
# Run training
# =====================================================================

if [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_dinov3_sib.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 25 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${SIB_FLAGS} \
        ${HYPER_FLAGS} \
        ${FDA_FLAGS} \
        ${DONOR_FLAGS}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_dinov3_sib.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 25 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${SIB_FLAGS} \
        ${HYPER_FLAGS} \
        ${FDA_FLAGS} \
        ${DONOR_FLAGS}

elif [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_dinov3_sib.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --model_name dinov3_vits16 \
        --weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 25 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${SIB_FLAGS} \
        ${HYPER_FLAGS} \
        ${FDA_FLAGS} \
        ${DONOR_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi