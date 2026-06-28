#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_inria_submit.sh
#
# Queues INRIA MAMNet training jobs on SLURM.
# Mirrors mamnet_submit.sh — 3 upper-bound models + 3 LOCO models.
# Uncomment the server block you need below.
#
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

# INRIA data root — written by inria_prep.py
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/inria/"
OUTPUT_DIR="${BASE_PATH}/data/mamnet_inria/outputs"

# No comparison_inference_dir for INRIA — we'll compute alpha from
# probability maps generated post-hoc via run_inference_probs.py
COMPARISON_INFERENCE_DIR=""
COMPARISON_DATA_ROOT=""

BOUNDARY_TOLERANCE=2
FOLD_NAMES=("austin" "chicago" "vienna")

# Make output log dir if it doesn't exist
mkdir -p "${BASE_PATH}/data/mamnet_inria"

# ============================================================
# PART 1: Train individual city upper-bound models
# ============================================================
echo "Queueing INRIA upper-bound models (single mode)..."
for city in austin chicago vienna
do
    for res in highres
    do
        name="mamnet_inria__${city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet_inria/${name}.out"
        data_root="${BASE_DATA_ROOT}${city}/${res}/"
        echo "  - ${city} ${res} (single mode)"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=single,DATA_ROOT=${data_root},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10,COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               mamnet_inria.sh
    done
done

# ============================================================
# PART 2: Train LOCO models
# ============================================================
echo "Queueing INRIA LOCO models..."
# Fold mapping: 0=holdout austin, 1=holdout chicago, 2=holdout vienna
for fold_id in 0 1 2
do
    for res in highres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="mamnet_inria__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet_inria/${name}.out"
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10,COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
               mamnet_inria.sh
    done
done

echo "All INRIA jobs queued!"
echo ""
echo "Expected outputs:"
echo "  Upper-bound:  ${OUTPUT_DIR}/mamnet_inria_{austin,chicago,vienna}_highres_1/"
echo "  LOCO:         ${OUTPUT_DIR}/mamnet_inria_loco_holdout_{austin,chicago,vienna}_highres_1/"
echo ""
echo "After training completes, run inference on each test split using"
echo "run_inference_probs.py to generate probability maps, then compute"
echo "alpha via test4v2_affine_decomposition.py for the INRIA-vs-shadow"
echo "comparison."