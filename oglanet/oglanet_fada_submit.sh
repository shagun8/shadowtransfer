#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_fada_submit.sh
#
# Queues OGLANetFADA LOCO training jobs on SLURM.
# Runs 3-fold Leave-One-City-Out analysis for cross-city shadow detection.
#
# Usage: bash oglanet_fada_submit.sh
#
# Fold mapping:
#   fold 0 → holdout Phoenix   (train: Chicago + Miami)
#   fold 1 → holdout Miami     (train: Chicago + Phoenix)
#   fold 2 → holdout Chicago   (train: Miami   + Phoenix)

# ============================================================================
# Server paths — uncomment the block for the server you are targeting
# ============================================================================

# ---- Gilbreth (ACTIVE) ----
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# ---- Anvil ----
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# ---- NCSA Delta ----
BASE_PATH="${PROJECT_ROOT}"
BASE_PATH2="${PROJECT_ROOT}"

# ============================================================================
# Paths
# ============================================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"

# ============================================================================
# Evaluation settings
# ============================================================================
BOUNDARY_TOLERANCE=2
EARLY_STOPPING_PATIENCE=20

# ============================================================================
# FADA hyperparameters
# Paper defaults (Bi et al., NeurIPS 2024):
#   FADA_RANK=16           Table 3: best performance at r=16–32
#   FADA_TOKEN_LENGTH=100  Fig 8:   stable in 75–125 range
#   FADA_STAGES="3 4 5"   feat3 (256ch) / feat4 (512ch) / feat5 (1024ch)
#                         NOTE: OGLANet feat5 is 1024ch (not 512ch as in MAMNet)
#                               because GLAMEncoder adds a stride-2 512→1024 block.
#   LR=1e-4               FADA paper default for Adam with frozen backbone
# ============================================================================
FADA_RANK=16
FADA_TOKEN_LENGTH=100
# Stages 3 and 4 only (ResNet layer3: 256ch, layer4: 512ch).
# Stage 5 (glam5_conv output, 1024ch) produces only 9 Haar tokens after DWT
# making the attention map degenerate — exclude it.
# Stages 1/2 can be added if GPU memory allows.
FADA_STAGES="3 4"
LR=0.0001
# Uncomment to set separate LRs (default: both equal to LR):
# LR_FADA=0.0001
# LR_DECODER=0.0001
WEIGHT_DECAY=0.0001

# ============================================================================
# Fold name lookup
# ============================================================================
FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================================
# Queue LOCO jobs
# ============================================================================
echo "=========================================="
echo "Queueing OGLANetFADA LOCO jobs"
echo "=========================================="
echo "  Encoder:          GLAMEncoder (ResNet-34 + GFEM) — FROZEN"
echo "  FADA rank (r):    ${FADA_RANK}"
echo "  FADA tokens (m):  ${FADA_TOKEN_LENGTH}"
echo "  FADA stages:      ${FADA_STAGES}"
echo "  Learning rate:    ${LR}"
echo "  Weight decay:     ${WEIGHT_DECAY}"
echo "  Contrast channel: YES (4-ch RGBC)"
echo "  Decision metric:  Tolerant ±${BOUNDARY_TOLERANCE}px mIOU"
echo "  Early stopping:   patience=${EARLY_STOPPING_PATIENCE}"
echo "  Epochs:           100 (max)"
echo "=========================================="
echo ""

for fold_id in 0 1 2
do
    for res in highres midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="oglanet_fada__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"

        echo "  Queuing fold ${fold_id} — holdout: ${holdout_city} / ${res}"

        sbatch \
            --output="${outputfile}" \
            --job-name="${name}" \
            --export=PROJECT_ROOT=${PROJECT_ROOT},\
MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
OUTPUT_DIR=${OUTPUT_DIR},\
USE_CONTRAST=1,\
EVAL_TOLERANT=1,\
BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},\
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},\
FADA_RANK=${FADA_RANK},\
FADA_TOKEN_LENGTH=${FADA_TOKEN_LENGTH},\
FADA_STAGES="${FADA_STAGES}",\
LR=${LR},\
WEIGHT_DECAY=${WEIGHT_DECAY} \
            oglanet_fada.sh
    done
done

echo ""
echo "All OGLANetFADA LOCO jobs queued!"
echo ""
echo "Output logs → ${BASE_PATH}/data/oglanet/"
echo "Checkpoints → ${OUTPUT_DIR}/"