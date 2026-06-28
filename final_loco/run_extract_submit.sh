#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# Submit feature extraction jobs for linear probe diagnostic (1d).
#
# Strategy:
#   1. Upper-bound: take the chicago upper checkpoint for each model and
#      extract features on all 3 cities.
#   2. LOCO vanilla: for each holdout fold, extract features on all 3 cities.
#
# ============================================================
# SERVER PATHS — uncomment the block for your active server
# ============================================================

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# LOG_DIR="${BASE_PATH}/data/extracted_features/logs"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"
LOG_DIR="${BASE_PATH}/data/extracted_features/logs"

BASE_DATA="${BASE_PATH}/data/Final_data_test"
OUTPUT_BASE="${BASE_PATH}/data"
FEATURE_OUT="${BASE_PATH}/data/extracted_features"

mkdir -p "$LOG_DIR"

CITIES=("chicago" "miami" "phoenix")
RES="midres"   # feature extraction focused on highres first
MODELS=("mamnet" "oglanet" "dinov3")

declare -A LOCO_HOLDOUT
LOCO_HOLDOUT[0]="phoenix"
LOCO_HOLDOUT[1]="miami"
LOCO_HOLDOUT[2]="chicago"

SUBMITTED=0
MISSING=0

find_checkpoint() {
    local model_type=$1
    local pattern=$2
    local search_dir="${OUTPUT_BASE}/${model_type}/outputs"
    [ ! -d "$search_dir" ] && echo "" && return
    local dirs
    dirs=$(find "$search_dir" -maxdepth 1 -type d -name "$pattern" \
           -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2-)
    for dir in $dirs; do
        local ckpt="${dir}/checkpoint_best.pth"
        [ -f "$ckpt" ] && echo "$ckpt" && return
    done
    echo ""
}

submit() {
    local model_type=$1 variant=$2 ckpt=$3 city=$4 res=$5 ckpt_id=$6
    local data_root="${BASE_DATA}/${city}/${res}"
    local job_name="feat_${model_type}_${ckpt_id}_on_${city}"
    local log="${LOG_DIR}/${job_name}.out"

    sbatch --output="$log" --job-name="$job_name" \
        --export=PROJECT_ROOT=${PROJECT_ROOT},MODEL_TYPE="$model_type",MODEL_VARIANT="$variant",\
CHECKPOINT_PATH="$ckpt",DATA_ROOT="$data_root",CITY="$city",RES="$res",\
CHECKPOINT_ID="$ckpt_id" \
        run_extract.sh

    if [ $? -eq 0 ]; then
        echo "  ✓ $job_name"
        ((SUBMITTED++))
    fi
}

# ================================================================
# SCENARIO 1: Upper-bound models (chicago → all cities)
# ================================================================
echo "========================================"
echo "UPPER-BOUND FEATURE EXTRACTION"
echo "========================================"

SOURCE_CITY="chicago"
for model_type in "${MODELS[@]}"; do
    pattern="${model_type}_${SOURCE_CITY}_${RES}_*"
    ckpt=$(find_checkpoint "$model_type" "$pattern")
    if [ -z "$ckpt" ]; then
        echo "  ⚠ Missing: ${model_type} upper ${SOURCE_CITY} ${RES}"
        ((MISSING++)); continue
    fi
    ckpt_id="upper_${SOURCE_CITY}_${RES}"
    echo ""
    echo "${model_type^^} upper ${SOURCE_CITY} → all cities"
    for city in "${CITIES[@]}"; do
        submit "$model_type" "base" "$ckpt" "$city" "$RES" "$ckpt_id"
    done
done

# ================================================================
# SCENARIO 2: LOCO vanilla models (trained on 2 cities → test all 3)
# ================================================================
echo ""
echo "========================================"
echo "LOCO VANILLA FEATURE EXTRACTION"
echo "========================================"

for model_type in "${MODELS[@]}"; do
    for fold_id in 0 1 2; do
        holdout="${LOCO_HOLDOUT[$fold_id]}"
        pattern="${model_type}_loco_holdout_${holdout}_${RES}_*"
        ckpt=$(find_checkpoint "$model_type" "$pattern")
        if [ -z "$ckpt" ]; then
            echo "  ⚠ Missing: ${model_type} loco vanilla holdout=${holdout} ${RES}"
            ((MISSING++)); continue
        fi
        ckpt_id="loco_vanilla_holdout_${holdout}_${RES}"
        echo ""
        echo "${model_type^^} LOCO vanilla (holdout=${holdout}) → all cities"
        for city in "${CITIES[@]}"; do
            submit "$model_type" "vanilla" "$ckpt" "$city" "$RES" "$ckpt_id"
        done
    done
done

# ================================================================
echo ""
echo "========================================"
echo "SUMMARY: Submitted=$SUBMITTED  Missing=$MISSING"
echo "========================================"
echo "Monitor with: squeue -u \$USER"
echo "========================================"