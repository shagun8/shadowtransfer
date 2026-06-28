#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_mrfp_submit.sh
#
# Queues MAMNet + MRFP+ LOCO training jobs on SLURM.
# Only LOCO analysis (3 folds).
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

# ---- MRFP configuration ----
# Don't-care band half-width in pixels
BOUNDARY_TOLERANCE=2

# MRFP+ is enabled by default (best variant from paper Table 5).
# Set USE_MRFP_PLUS=0 to use plain MRFP (HRFP+NP+ only).
USE_MRFP_PLUS=1

# Perturbation probabilities (paper default: 0.5 each)
HRFP_PROB=0.5
NP_PROB=0.5
HRFP_PLUS_PROB=0.5
HRFP_BN_STD=0.5

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# LOCO models (3 folds)
# ============================================================
echo "Queueing MAMNet + MRFP+ LOCO models..."

# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="mamnet_mrfp_plus__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"

        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10,USE_MRFP_PLUS=${USE_MRFP_PLUS},HRFP_PROB=${HRFP_PROB},NP_PROB=${NP_PROB},HRFP_PLUS_PROB=${HRFP_PLUS_PROB},HRFP_BN_STD=${HRFP_BN_STD},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               mamnet_mrfp.sh
    done
done

echo "All LOCO jobs queued!"