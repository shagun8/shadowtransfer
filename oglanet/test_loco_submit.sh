#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: submit_loco_eval.sh
# Submit LOCO evaluation jobs for both resolutions in parallel

BASE_DATA_ROOT="${PROJECT_ROOT}/data/Final_data_test/"
OUTPUT_DIR="${PROJECT_ROOT}/data/oglanet/outputs"
LOG_DIR="${PROJECT_ROOT}/data/oglanet/loco_eval_logs"

# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

echo "======================================"
echo "Submitting LOCO Evaluation Jobs"
echo "======================================"
echo ""
echo "Data Root: ${BASE_DATA_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Log Dir: ${LOG_DIR}"
echo ""

# Submit jobs for all resolution × fold combinations
FOLD_NAMES=("phoenix" "miami" "chicago")

for res in highres midres
do
    for fold_id in 0 1 2
    do
        fold_name="${FOLD_NAMES[$fold_id]}"
        name="loco_eval_${res}_fold${fold_id}_${fold_name}"
        outputfile="${LOG_DIR}/${name}.out"
        
        echo "Submitting: ${name}"
        echo "  Resolution: ${res}"
        echo "  Fold: ${fold_id} (holdout: ${fold_name})"
        echo "  Log file: ${outputfile}"
        
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=PROJECT_ROOT=${PROJECT_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR} \
               test_loco.sh
        
        echo ""
    done
done

echo "======================================"
echo "All LOCO evaluation jobs submitted!"
echo "======================================"
echo ""