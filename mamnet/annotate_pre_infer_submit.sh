#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"

# Base paths
CHECKPOINT_BASE="${PROJECT_ROOT}/data/mamnet/outputs/"
ANNOTATION_BASE="${PROJECT_ROOT}/data/NAIP_Data_Process/Images/"
OUTPUT_BASE="${PROJECT_ROOT}/data/NAIP_Data_Process/Prelabels/"

# Cities and resolutions
#CITIES=("chicago" "miami")
CITIES=("phoenix")
RESOLUTIONS=("highres" "midres")

# Function to find the best checkpoint for a given city and resolution
find_best_checkpoint() {
    local city=$1
    local res=$2
    
    # Find the most recent training output directory
    #local checkpoint_dir=$(ls -dt ${CHECKPOINT_BASE}/mamnet_${city}_${res}_* 2>/dev/null | head -n 1)
	local checkpoint_dir=$(ls -dt ${CHECKPOINT_BASE}/mamnet_all_${res}_* 2>/dev/null | head -n 1)
    
    if [ -d "$checkpoint_dir" ]; then
        echo "${checkpoint_dir}/checkpoint_best.pth"
    else
        echo ""
    fi
}

# Loop through all sessions (1 to 30)
for session_num in $(seq 31 45)
do
    # Format session number with leading zero
    session_str=$(printf "annotation_session_%02d" $session_num)
    
    echo "========================================="
    echo "Processing ${session_str}"
    echo "========================================="
    
    # Create output directory for this session
    SESSION_OUTPUT_DIR="${OUTPUT_BASE}/${session_str}"
    mkdir -p ${SESSION_OUTPUT_DIR}
    
    # Loop through all city/resolution combinations
    for city in "${CITIES[@]}"
    do
        for res in "${RESOLUTIONS[@]}"
        do
            echo ""
            echo "Processing: ${city} ${res}"
            
            # Find checkpoint
            CHECKPOINT=$(find_best_checkpoint $city $res)
            
            if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
                echo "ERROR: Cannot find checkpoint for ${city} ${res}"
                echo "Expected pattern: ${CHECKPOINT_BASE}/mamnet_${city}_${res}_*"
                continue
            fi
            
            # Image directory
            IMAGE_DIR="${ANNOTATION_BASE}/${session_str}/${city}_${res}"
            
            if [ ! -d "$IMAGE_DIR" ]; then
                echo "WARNING: Image directory not found: ${IMAGE_DIR}"
                echo "Skipping..."
                continue
            fi
            
            # Count images
            num_images=$(ls -1 ${IMAGE_DIR}/*.png 2>/dev/null | wc -l)
            
            if [ $num_images -eq 0 ]; then
                echo "WARNING: No images found in ${IMAGE_DIR}"
                echo "Skipping..."
                continue
            fi
            
            echo "  Checkpoint: ${CHECKPOINT}"
            echo "  Images:     ${IMAGE_DIR} \(${num_images} images\)"
            echo "  Output:     ${SESSION_OUTPUT_DIR}"
            
            # Job name
            JOB_NAME="inf_s${session_num}_${city}_${res}"
            OUTPUT_FILE="${SESSION_OUTPUT_DIR}/slurm_${city}_${res}.out"
            
            # Submit job
            sbatch \
                --output=${OUTPUT_FILE} \
                --job-name=${JOB_NAME} \
                --export=PROJECT_ROOT=${PROJECT_ROOT},CHECKPOINT=${CHECKPOINT},IMAGE_DIR=${IMAGE_DIR},OUTPUT_DIR=${SESSION_OUTPUT_DIR},CITY=${city},RESOLUTION=${res},SESSION_NUM=${session_num} \
                annotate_pre_infer.sh
            
            echo "  Job submitted: ${JOB_NAME}"
        done
    done
    
    echo " "
done