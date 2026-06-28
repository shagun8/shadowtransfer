#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_submit.sh
#
# Queues DINOv3 training jobs on SLURM.
#
# Part 1 — Individual city models:  3 jobs (3 cities x midres)
# Part 2 — LOCO models:             3 jobs (3 folds x midres)
#
# Total: 6 jobs

# ---- Cluster paths (uncomment ONE block) ----

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}/"
# BASE_PATH2="${PROJECT_ROOT}/"

# ---- NCSA Delta ----
BASE_PATH="${PROJECT_ROOT}"
BASE_PATH2="${PROJECT_ROOT}"

# ---- Derived paths (no changes needed) ----
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

# CHANGED: Don't-care band half-width in pixels.
# DetailedEvaluator always runs with this value.
# When EVAL_TOLERANT=1, also selects which metric drives decisions.
BOUNDARY_TOLERANCE=2

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# PART 1: Train individual city models
# ============================================================
# echo "Queueing individual city models..."
# for city in chicago miami phoenix
# do
    # for res in midres
    # do
        # name="dinov3__${city}__${res}"
        # outfile="${BASE_PATH}/data/dinov3/${name}.out"
        # data_root="${BASE_DATA_ROOT}${city}/${res}/"
        # echo "  - ${city} ${res} (single mode)"
        # sbatch --output=${outfile} \
               # --job-name=${name} \
               # --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=single,DATA_ROOT=${data_root},OUTPUT_DIR=${OUTPUT_DIR},WEIGHT_DIR=${WEIGHT_DIR},EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               # dinov3.sh
    # done
# done

# ============================================================
# PART 2: Train LOCO models
# ============================================================
echo "Queueing LOCO models..."
# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
for fold_id in 1
do
    for res in highres
    do
        holdout="${FOLD_NAMES[$fold_id]}"
        name="dinov3__loco_holdout_${holdout}__${res}"
        outfile="${BASE_PATH}/data/dinov3/${name}.out"
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout}) ${res}"
        sbatch --output=${outfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},WEIGHT_DIR=${WEIGHT_DIR},EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               dinov3.sh
	done
done

echo ""
echo "All jobs queued!"