#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
#
# Master submission script for INRIA probability inference.
# Paralleling submit_inference_probs.sh but for MAMNet-INRIA only.
# Launches run_inference_probs_inria.sh for every upper + LOCO cell.

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/inria/"
# INRIA checkpoints live under mamnet_inria/outputs/ rather than mamnet/outputs/
CHECKPOINT_ROOT="${BASE_PATH}/data/mamnet_inria/outputs"
INFERENCE_OUTPUT_DIR="${BASE_PATH}/data/Test_img_probs_inria/"

CITIES=("austin" "chicago" "vienna")
RESOLUTIONS=("highres")

declare -A LOCO_TEST_CITY
LOCO_TEST_CITY[0]="austin"
LOCO_TEST_CITY[1]="chicago"
LOCO_TEST_CITY[2]="vienna"

SUBMITTED=0
MISSING=0
FAILED=0

find_checkpoint() {
    local pattern=$1
    local search_dir="${CHECKPOINT_ROOT}"

    if [ ! -d "$search_dir" ]; then
        echo ""
        return
    fi

    local dirs
    dirs=$(find "$search_dir" -maxdepth 1 -type d -name "$pattern" \
           -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)

    for dir in $dirs; do
        local checkpoint="${dir}/checkpoint_best.pth"
        if [ -f "$checkpoint" ]; then
            echo "$checkpoint"
            return
        fi
    done

    echo ""
}

submit_job() {
    local model_variant=$1
    local test_type=$2
    local city=$3
    local train_res=$4
    local test_res=$5
    local checkpoint_path=$6
    local data_root=$7
    local job_name=$8

    local log_dir="${INFERENCE_OUTPUT_DIR}/inference_logs"
    mkdir -p "$log_dir"

    sbatch --output="${log_dir}/${job_name}.out" \
           --job-name="$job_name" \
           --export=PROJECT_ROOT=${PROJECT_ROOT},MODEL_VARIANT="$model_variant",\
TEST_TYPE="$test_type",\
CITY="$city",\
TRAIN_RES="$train_res",\
TEST_RES="$test_res",\
CHECKPOINT_PATH="$checkpoint_path",\
DATA_ROOT="$data_root",\
OUTPUT_DIR="$INFERENCE_OUTPUT_DIR" \
           run_inference_probs_inria.sh

    if [ $? -eq 0 ]; then
        echo "  ✓ Submitted: $job_name"
        ((SUBMITTED++))
    else
        echo "  ✗ Failed to submit: $job_name"
        ((FAILED++))
    fi
}

# ===========================================================================
# SCENARIO 1 — INRIA UPPER-BOUNDS (within-city)
# ===========================================================================
echo "========================================================================"
echo "INRIA SCENARIO 1: UPPER-BOUNDS"
echo "========================================================================"

for city in "${CITIES[@]}"; do
    for train_res in "${RESOLUTIONS[@]}"; do
        echo ""
        echo "MAMNet — $city — $train_res"
        echo "----------------------------------------------------------------"

        # Upper-bound directory follows the InriaTrainer rename pattern:
        # mamnet_inria_{city}_{res}_1
        checkpoint=$(find_checkpoint "mamnet_${city}_${train_res}_*")

        if [ -z "$checkpoint" ]; then
            echo "  ⚠ Checkpoint not found: mamnet_inria_${city}_${train_res}_*"
            ((MISSING++))
            continue
        fi
        echo "  ✓ Found checkpoint: $checkpoint"

        data_root="${BASE_DATA_ROOT}/${city}/${train_res}"
        job_name="prob_inria_upper_mamnet_${city}_${train_res}"
        submit_job "base" "upper" "$city" "$train_res" "$train_res" \
                   "$checkpoint" "$data_root" "$job_name"
    done
done

# ===========================================================================
# SCENARIO 2 — INRIA LOCO
# ===========================================================================
echo ""
echo "========================================================================"
echo "INRIA SCENARIO 2: LOCO"
echo "========================================================================"

# Only one method to evaluate — "vanilla" in the shadow naming convention.
# (No adaptation methods trained on INRIA; control experiment is shape-only.)
for fold_id in 0 1 2; do
    test_city="${LOCO_TEST_CITY[$fold_id]}"

    for res in "${RESOLUTIONS[@]}"; do
        echo ""
        echo "MAMNet — VANILLA — fold $fold_id ($test_city) — $res"
        echo "----------------------------------------------------------------"

        checkpoint=$(find_checkpoint \
            "mamnet_loco_holdout_${test_city}_${res}_*")

        if [ -z "$checkpoint" ]; then
            echo "  ⚠ Checkpoint not found: mamnet_inria_loco_holdout_${test_city}_${res}_*"
            ((MISSING++))
            continue
        fi
        echo "  ✓ Found checkpoint: $checkpoint"

        data_root="${BASE_DATA_ROOT}/${test_city}/${res}"
        job_name="prob_inria_loco_mamnet_vanilla_${test_city}_${res}"
        submit_job "vanilla" "loco" "$test_city" "$res" "$res" \
                   "$checkpoint" "$data_root" "$job_name"
    done
done

# ===========================================================================
# SUMMARY
# ===========================================================================
echo ""
echo "========================================================================"
echo "SUBMISSION SUMMARY"
echo "========================================================================"
echo "Submitted: $SUBMITTED"
echo "Missing  : $MISSING"
echo "Failed   : $FAILED"
echo ""
echo "Expected: 3 upper + 3 LOCO = 6 jobs total"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "========================================================================"