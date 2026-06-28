#!/bin/bash
# FILENAME: dinov3_sib_newmod.sh
#
# SLURM job script for DINOv3 + SIB with diagnostic-motivated modules.
# Extends dinov3_sib.sh with CACR, CE-AURC, and TENT support.
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
# NEW — Diagnostic-motivated module flags (0 or 1):
#   USE_CACR             — Class-Asymmetric Confidence Regularizer
#   USE_CE_AURC          — CE-AURC auxiliary loss on gt_shadow pixels
#   USE_TENT             — Test-time entropy minimization
#
# NEW — Diagnostic hyperparameters (passed only when explicitly set):
#   CACR_WEIGHT          — CACR loss weight (default 0.1)
#   CACR_NEG_WEIGHT      — CACR background weight (default 0.0)
#   CE_AURC_WEIGHT       — CE-AURC loss weight (default 0.01)
#   CE_AURC_FLOOR        — CE-AURC floor weight (default 0.5)
#   TENT_STEPS           — TENT adaptation steps per batch (default 1)
#   TENT_LR              — TENT optimizer lr (default 0.001)
#
# Test runs: override EPOCHS / EARLY_STOPPING_PATIENCE from submit:
#   EPOCHS=5 EARLY_STOPPING_PATIENCE=10 ./dinov3_sib_newmod_submit.sh

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

# ─────────────────────────────────────────────────────────────────────────────
# Defaults: epochs and patience can be overridden from the submit script
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS_VAL=${EPOCHS:-100}
PATIENCE_VAL=${EARLY_STOPPING_PATIENCE:-25}

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

# =====================================================================
# NEW: Diagnostic module flags (CACR, CE-AURC, TENT)
# =====================================================================
DIAG_FLAGS=""
[ "${USE_CACR}"      == "1" ] && DIAG_FLAGS="${DIAG_FLAGS} --use_cacr"
[ "${USE_CE_AURC}"   == "1" ] && DIAG_FLAGS="${DIAG_FLAGS} --use_ce_aurc"
[ "${USE_TENT}"      == "1" ] && DIAG_FLAGS="${DIAG_FLAGS} --use_tent"

# Diagnostic hyperparameters (only passed when explicitly set)
[ -n "${CACR_WEIGHT}"      ] && DIAG_FLAGS="${DIAG_FLAGS} --cacr_weight ${CACR_WEIGHT}"
[ -n "${CACR_NEG_WEIGHT}"  ] && DIAG_FLAGS="${DIAG_FLAGS} --cacr_neg_weight ${CACR_NEG_WEIGHT}"
[ -n "${CE_AURC_WEIGHT}"   ] && DIAG_FLAGS="${DIAG_FLAGS} --ce_aurc_weight ${CE_AURC_WEIGHT}"
[ -n "${CE_AURC_FLOOR}"    ] && DIAG_FLAGS="${DIAG_FLAGS} --ce_aurc_floor ${CE_AURC_FLOOR}"
[ -n "${TENT_STEPS}"       ] && DIAG_FLAGS="${DIAG_FLAGS} --tent_steps ${TENT_STEPS}"
[ -n "${TENT_LR}"          ] && DIAG_FLAGS="${DIAG_FLAGS} --tent_lr ${TENT_LR}"

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
echo "DINOv3 + SIB + NewMod Training"
echo "================================================="
echo "MODE:             ${MODE}"
echo "FOLD_ID:          ${FOLD_ID}"
echo "RESOLUTION:       ${RESOLUTION}"
echo "EPOCHS:           ${EPOCHS_VAL}"
echo "PATIENCE:         ${PATIENCE_VAL}"
echo "SIB flags:        ${SIB_FLAGS}"
echo "DIAG flags:       ${DIAG_FLAGS}"
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
echo "--- New module flags ---"
echo "USE_CACR:            ${USE_CACR:-0} (w=${CACR_WEIGHT:-def}, neg_w=${CACR_NEG_WEIGHT:-def})"
echo "USE_CE_AURC:         ${USE_CE_AURC:-0} (w=${CE_AURC_WEIGHT:-def}, floor=${CE_AURC_FLOOR:-def})"
echo "USE_TENT:            ${USE_TENT:-0} (steps=${TENT_STEPS:-def}, lr=${TENT_LR:-def})"
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
        --epochs ${EPOCHS_VAL} \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience ${PATIENCE_VAL} \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${SIB_FLAGS} \
        ${DIAG_FLAGS} \
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
        --epochs ${EPOCHS_VAL} \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience ${PATIENCE_VAL} \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${SIB_FLAGS} \
        ${DIAG_FLAGS} \
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
        --epochs ${EPOCHS_VAL} \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience ${PATIENCE_VAL} \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${SIB_FLAGS} \
        ${DIAG_FLAGS} \
        ${HYPER_FLAGS} \
        ${FDA_FLAGS} \
        ${DONOR_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi