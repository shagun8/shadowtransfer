#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_fada_submit.sh
#
# Queues MAMNet-FADA training jobs on SLURM.
# Runs LOCO analysis (3 folds) for cross-city shadow detection.
# Uncomment the server block you need below.

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

# ---- Evaluation settings ----
BOUNDARY_TOLERANCE=2

# ---- FADA hyperparameters ----
# Paper defaults (Bi et al., NeurIPS 2024):
#   FADA_RANK=16       (Table 3: best at 16-32)
#   FADA_TOKEN_LENGTH=100  (Fig 8: stable in 75-125)
#   FADA_STAGES="3 4 5"   (feat3, feat4, feat5)
#   LR=0.0001          (paper default for Adam with frozen backbone)
FADA_RANK=16
FADA_TOKEN_LENGTH=100
FADA_STAGES="3 4 5"
LR=0.0001

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# LOCO Analysis (Leave-One-City-Out)
# ============================================================
echo "Queueing MAMNet-FADA LOCO models..."
echo "  FADA config: rank=${FADA_RANK}, m=${FADA_TOKEN_LENGTH}, stages=${FADA_STAGES}, lr=${LR}"
echo ""

# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="mamnet_fada__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"

        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10,FADA_RANK=${FADA_RANK},FADA_TOKEN_LENGTH=${FADA_TOKEN_LENGTH},FADA_STAGES="${FADA_STAGES}",LR=${LR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               mamnet_fada.sh
    done
done

echo ""
echo "All MAMNet-FADA LOCO jobs queued!"
echo ""
echo "=========================================="
echo "Configuration Summary"
echo "=========================================="
echo "  Encoder:           ResNet-34 (FROZEN)"
echo "  FADA rank (r):     ${FADA_RANK}"
echo "  FADA tokens (m):   ${FADA_TOKEN_LENGTH}"
echo "  FADA stages:       ${FADA_STAGES}"
echo "  Learning rate:     ${LR}"
echo "  Contrast channel:  YES (4ch RGBC)"
echo "  Eval metric:       Tolerant ±${BOUNDARY_TOLERANCE}px mIOU"
echo "  Early stopping:    patience=10"
echo "  Epochs:            100 (max)"
echo "=========================================="