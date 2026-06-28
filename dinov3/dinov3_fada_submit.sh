#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_fada_submit.sh
#
# Queues DINOv3-FADA LOCO training jobs on SLURM.
# Runs Leave-One-City-Out analysis: 3 folds × highres.
#
# ════════════════════════════════════════════════════════════════════════════
# SERVER PATHS — uncomment exactly ONE block
# ════════════════════════════════════════════════════════════════════════════

# ── Gilbreth ──────────────────────────────────────────────────────────────
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# ── Anvil ─────────────────────────────────────────────────────────────────
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# ── NCSA Delta ────────────────────────────────────────────────────────────
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

# ════════════════════════════════════════════════════════════════════════════
# Derived paths  (no changes needed below this line)
# ════════════════════════════════════════════════════════════════════════════
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

# ════════════════════════════════════════════════════════════════════════════
# Evaluation settings
# ════════════════════════════════════════════════════════════════════════════
# DetailedEvaluator always runs with this band width.
# When EVAL_TOLERANT=1, tolerant mIOU also drives all decisions.
BOUNDARY_TOLERANCE=2

# ════════════════════════════════════════════════════════════════════════════
# FADA hyperparameters  (Bi et al., NeurIPS 2024 defaults)
# ════════════════════════════════════════════════════════════════════════════
#   FADA_RANK=16          Table 3 — best performance at 16-32
#   FADA_TOKEN_LENGTH=100 Fig  8 — stable in 75-125 range
#   FADA_STAGES="3 6 9 11"  all DINOv3 decoder feature-extraction blocks
#   LR=0.0001             paper default for Adam with frozen backbone
FADA_RANK=16
FADA_TOKEN_LENGTH=100
FADA_STAGES="3 6 9 11"
LR=0.0001
EARLY_STOPPING_PATIENCE=15
EPOCHS=100

# Fold → held-out city mapping
FOLD_NAMES=("phoenix" "miami" "chicago")

# ════════════════════════════════════════════════════════════════════════════
# LOCO: queue one job per fold
# Fold 0 → holdout Phoenix  (train: Chicago + Miami)
# Fold 1 → holdout Miami    (train: Chicago + Phoenix)
# Fold 2 → holdout Chicago  (train: Miami + Phoenix)
# ════════════════════════════════════════════════════════════════════════════
echo "Queueing DINOv3-FADA LOCO jobs …"
echo ""
echo "  FADA config: rank=${FADA_RANK}  m=${FADA_TOKEN_LENGTH}"
echo "               stages=${FADA_STAGES}  lr=${LR}"
echo "  Early stopping patience: ${EARLY_STOPPING_PATIENCE}"
echo "  Max epochs: ${EPOCHS}"
echo ""

for fold_id in 0 1 2; do
    for res in midres; do
        holdout="${FOLD_NAMES[$fold_id]}"
        name="dinov3_fada__loco_holdout_${holdout}__${res}"
        outfile="${BASE_PATH}/data/dinov3/${name}.out"

        echo "  Submitting fold ${fold_id}  (holdout: ${holdout})  ${res}"

        sbatch \
            --output="${outfile}" \
            --job-name="${name}" \
            --export=PROJECT_ROOT=${PROJECT_ROOT},\
MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
OUTPUT_DIR=${OUTPUT_DIR},\
WEIGHT_DIR=${WEIGHT_DIR},\
EVAL_TOLERANT=1,\
BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},\
FADA_RANK=${FADA_RANK},\
FADA_TOKEN_LENGTH=${FADA_TOKEN_LENGTH},\
"FADA_STAGES=${FADA_STAGES}",\
LR=${LR},\
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},\
EPOCHS=${EPOCHS},\
COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},\
COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
            dinov3_fada.sh
    done
done

echo ""
echo "All DINOv3-FADA LOCO jobs queued!"
echo ""
echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo "  Backbone     : DINOv3 ViT-S/16  (FROZEN ~22 M params)"
echo "  FADA rank    : ${FADA_RANK}       (LoRA r, paper best 16-32)"
echo "  FADA tokens  : ${FADA_TOKEN_LENGTH}      (m, paper stable 75-125)"
echo "  FADA stages  : ${FADA_STAGES}  (all decoder feature blocks)"
echo "  Trainable    : ~3.4 M  (4×FADABlock + decoder)"
echo "  LR           : ${LR}  (Adam, paper default)"
echo "  Eval metric  : Tolerant ±${BOUNDARY_TOLERANCE}px mIOU (decisions)"
echo "  Early stop   : patience=${EARLY_STOPPING_PATIENCE}"
echo "  Max epochs   : ${EPOCHS}"
echo "  Output dir   : ${OUTPUT_DIR}"
echo "=========================================="