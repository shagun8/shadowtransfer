#!/bin/bash
# SBATCH script for running Experiments A, B, C
# Parameters passed via environment variables from submit script:
#   EXPERIMENT, MODEL_TYPE, MODEL_VARIANT, HOLDOUT_CITY, RES,
#   CHECKPOINT_PATH, TEST_DATA_ROOT, [TRAIN_DATA_ROOT], [SOURCE_IMAGE_DIRS],
#   [OUTPUT_BASE], [TRAIN_FRACTION], [EPOCHS], [LR], [DATA_EFFICIENCY]
#
# ============================================================
# SBATCH HEADERS — uncomment the block for your server
# ============================================================
#
# --- Gilbreth ---
##SBATCH -A ${SLURM_ACCOUNT}
##SBATCH --partition=${SLURM_PARTITION}
##SBATCH --exclude=${NODE}
##SBATCH --gres=gpu:1
##SBATCH --qos=normal
##SBATCH --constraint=a100
##SBATCH --nodes=1
##SBATCH --gpus-per-node=1
##SBATCH --cpus-per-task=4
##SBATCH --mem=64G
##SBATCH --time=4:00:00
#
# --- NCSA Delta ---
#SBATCH --account=<SLURM_ACCOUNT>   # fill in your account/partition
#SBATCH --partition=<SLURM_PARTITION>   # fill in your account/partition
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# ============================================================
# SERVER SETUP — uncomment the block for your server
# ============================================================

# --- Gilbreth ---
# cd ${PROJECT_ROOT}/python/final_loco
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${PROJECT_ROOT}/conda_envs/satmae_cuda12
# PYTHON_BIN=${PROJECT_ROOT}/conda_envs/satmae_cuda12/bin/python
# DINOV3_WEIGHTS=${PROJECT_ROOT}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# --- NCSA Delta ---
module purge
module load cudatoolkit
module load pytorch-conda/2.8
conda activate ${PROJECT_ROOT}/envs/satmae_cuda12
cd ${PROJECT_ROOT}/python/final_loco
PYTHON_BIN=${PROJECT_ROOT}/envs/satmae_cuda12/bin/python
DINOV3_WEIGHTS=${PROJECT_ROOT}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

echo "=========================================="
echo "EXPERIMENT ${EXPERIMENT} — ${MODEL_TYPE}/${MODEL_VARIANT}"
echo "Holdout: ${HOLDOUT_CITY} | Res: ${RES}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Start: $(date)"
echo "=========================================="

if [ "${EXPERIMENT}" == "a" ]; then
    # Experiment A: Decoder Retraining on Holdout City
    # Longer time limit set via --time in submit script for Exp A
    $PYTHON_BIN -u experiment_a_decoder_retrain.py \
        --model_type      ${MODEL_TYPE} \
        --model_variant   ${MODEL_VARIANT} \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --holdout_city    ${HOLDOUT_CITY} \
        --res             ${RES} \
        --train_data_root ${TRAIN_DATA_ROOT} \
        --test_data_root  ${TEST_DATA_ROOT} \
        --output_base     ${OUTPUT_BASE} \
        --train_fraction  ${TRAIN_FRACTION:-0.25} \
        --epochs          ${EPOCHS:-30} \
        --lr              ${LR:-0.001} \
        --batch_size      4 \
        --num_workers     4 \
        --dinov3_weights_path ${DINOV3_WEIGHTS} \
        ${DATA_EFFICIENCY:+--data_efficiency}

elif [ "${EXPERIMENT}" == "b" ]; then
    # Experiment B: BN Statistics Swap (DINOv3 skipped in Python)
    # excluded variant — not in release: experiment_b_bn_swap.py is not shipped.
    echo "Experiment B (BN-stat swap) is an excluded variant and not in this release; skipping."
    # $PYTHON_BIN -u experiment_b_bn_swap.py \
    #     --model_type      ${MODEL_TYPE} \
    #     --model_variant   ${MODEL_VARIANT} \
    #     --checkpoint_path ${CHECKPOINT_PATH} \
    #     --holdout_city    ${HOLDOUT_CITY} \
    #     --res             ${RES} \
    #     --test_data_root  ${TEST_DATA_ROOT} \
    #     --output_base     ${OUTPUT_BASE} \
    #     --layerwise \
    #     --batch_size      4 \
    #     --num_workers     4
		
elif [ "${EXPERIMENT}" == "b2" ]; then
    # Experiment B2: Encoder Retraining on Holdout City
    # Uses architecture-specific LR if LR env var is not set:
    #   MAMNet/OGLANet: 1e-4, DINOv3: 1e-5
    LR_FLAG=""
    if [ -n "${LR}" ]; then
        LR_FLAG="--lr ${LR}"
    fi
    # ^ If LR is unset, experiment_b2 uses its internal defaults
 
    $PYTHON_BIN -u experiment_b2_encoder_retrain.py \
        --model_type      ${MODEL_TYPE} \
        --model_variant   ${MODEL_VARIANT} \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --holdout_city    ${HOLDOUT_CITY} \
        --res             ${RES} \
        --train_data_root ${TRAIN_DATA_ROOT} \
        --test_data_root  ${TEST_DATA_ROOT} \
        --output_base     ${OUTPUT_BASE} \
        --train_fraction  ${TRAIN_FRACTION:-0.25} \
        --epochs          ${EPOCHS:-30} \
        ${LR_FLAG} \
        --batch_size      4 \
        --num_workers     4 \
        --dinov3_weights_path ${DINOV3_WEIGHTS}

elif [ "${EXPERIMENT}" == "c" ]; then
    # Experiment C: Test-Time Histogram Matching
    $PYTHON_BIN -u experiment_c_histogram_match.py \
        --model_type      ${MODEL_TYPE} \
        --model_variant   ${MODEL_VARIANT} \
        --checkpoint_path ${CHECKPOINT_PATH} \
        --holdout_city    ${HOLDOUT_CITY} \
        --res             ${RES} \
        --test_data_root  ${TEST_DATA_ROOT} \
        --source_image_dirs ${SOURCE_IMAGE_DIRS} \
        --output_base     ${OUTPUT_BASE} \
        --save_samples \
        --batch_size      4 \
        --num_workers     4 \
        --dinov3_weights_path ${DINOV3_WEIGHTS}

elif [ "${EXPERIMENT}" == "eval" ]; then
    # Evaluation: compute recovery ratios and diagnostics
    $PYTHON_BIN -u evaluate_experiments.py \
        --experiments ${EVAL_EXPERIMENTS:-a b c} \
        --models      ${EVAL_MODELS:-mamnet oglanet dinov3} \
        --resolutions ${EVAL_RES:-highres}

elif [ "${EXPERIMENT}" == "eval_robust" ]; then
    # Robust recovery analysis: per-image R, bootstrap CIs, cross-arch tests
    # excluded variant — not in release: evaluate_recovery_robust.py is not shipped.
    echo "eval_robust (evaluate_recovery_robust.py) is an excluded variant and not in this release; skipping."
    # $PYTHON_BIN -u evaluate_recovery_robust.py \
    #     --experiments ${EVAL_EXPERIMENTS:-a b2} \
    #     --models      ${EVAL_MODELS:-mamnet oglanet dinov3} \
    #     --resolutions ${EVAL_RES:-highres} \
    #     --n_boot      ${N_BOOT:-10000}
		
else
    echo "ERROR: Unknown experiment: ${EXPERIMENT}"
    exit 1
fi

echo ""
echo "End: $(date)"
echo "Job complete!"