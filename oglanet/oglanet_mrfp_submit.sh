#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_mrfp_submit.sh
#
# Queues OGLANet + MRFP+ LOCO training jobs on SLURM.
# Runs all 3 leave-one-city-out folds.
#
# Fold mapping:
#   fold 0 → holdout: phoenix  (train: chicago + miami)
#   fold 1 → holdout: miami    (train: chicago + phoenix)
#   fold 2 → holdout: chicago  (train: miami + phoenix)

# ============================================================
# Server paths (uncomment the one you need)
# ============================================================

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"
BASE_PATH2="${PROJECT_ROOT}"

# ============================================================
# Dataset and output paths
# ============================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"

# ============================================================
# MRFP configuration
# MRFP+ is enabled by default (best variant from paper Table 5).
# Set USE_MRFP_PLUS=0 to ablate to plain MRFP (HRFP+NP+ only).
# ============================================================
USE_MRFP_PLUS=1

# Perturbation probabilities (paper default: 0.5 each)
HRFP_PROB=0.5
NP_PROB=0.5
HRFP_PLUS_PROB=0.5
HRFP_BN_STD=0.5

# Boundary tolerance (don't-care band half-width in pixels)
BOUNDARY_TOLERANCE=2

# Early stopping patience (epochs without improvement)
EARLY_STOPPING_PATIENCE=30

# ============================================================
# Fold name lookup
# ============================================================
FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# Queue LOCO jobs — all 3 folds
# ============================================================
echo "Queueing OGLANet + MRFP+ LOCO models..."

for fold_id in 1
do
    for res in highres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="oglanet_mrfp_plus__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"

        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,\
BASE_DATA_ROOT=${BASE_DATA_ROOT},\
RESOLUTION=${res},\
FOLD_ID=${fold_id},\
OUTPUT_DIR=${OUTPUT_DIR},\
USE_CONTRAST=1,\
EVAL_TOLERANT=1,\
BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},\
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE},\
USE_MRFP_PLUS=${USE_MRFP_PLUS},\
HRFP_PROB=${HRFP_PROB},\
NP_PROB=${NP_PROB},\
HRFP_PLUS_PROB=${HRFP_PLUS_PROB},\
HRFP_BN_STD=${HRFP_BN_STD} \
               oglanet_mrfp.sh
    done
done

echo "All LOCO jobs queued!"
echo ""
echo "Summary:"
echo "  Model:       OGLANet + MRFP+ (HRFP + HRFP+ + NP+)"
echo "  Folds:       3 (phoenix / miami / chicago holdouts)"
# echo "  Resolution:  highres"
echo "  lr:          0.0001"
echo "  epochs:      100"
echo "  MRFP probs:  HRFP=${HRFP_PROB}  NP+=${NP_PROB}  HRFP+=${HRFP_PLUS_PROB}"
echo "  BN std:      ${HRFP_BN_STD}"
echo "  Boundary K:  ${BOUNDARY_TOLERANCE}px"
echo "  Outputs:     ${OUTPUT_DIR}"