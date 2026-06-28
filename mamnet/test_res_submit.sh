#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: submit_loco_eval.sh
# Submit LOCO evaluation jobs for both resolutions in parallel

BASE_DATA_ROOT="${PROJECT_ROOT}/data/Final_data_test/"
OUTPUT_DIR="${PROJECT_ROOT}/data/mamnet/outputs"
LOG_DIR="${PROJECT_ROOT}/data/mamnet/res_eval_logs"

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

CITIES=("chicago" "miami" "phoenix")

for source_res in midres highres
do
    for target_res in midres highres
    do
		# Skip same-resolution combinations if desired
        if [ "$source_res" == "$target_res" ]; then
            continue
        fi
		
        for city in "${CITIES[@]}"
        do
            name="res_eval_${source_res}to${target_res}_${city}"
            outputfile="${LOG_DIR}/${name}.out"
            
            echo "Submitting: ${name}"
            echo "  Source Resolution: ${source_res}"
            echo "  Target Resolution: ${target_res}"
            echo "  City: ${city}"
            echo "  Log file: ${outputfile}"
            
            sbatch --output=${outputfile} \
                   --job-name=${name} \
                   --export=PROJECT_ROOT=${PROJECT_ROOT},SOURCE_RES=${source_res},TARGET_RES=${target_res},CITY=${city},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR} \
                   test_res.sh
            
            echo ""
        done
    done
done

echo "======================================"
echo "All Resolution evaluation jobs submitted!"
echo "======================================"
echo ""