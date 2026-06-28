#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"

# Master submission script for probability-saving inference.
# Parallel to the run_inference master script; launches run_inference_probs.sh
# for every upper and LOCO cell so we can build confidence distributions.
#
# Only Scenario 1 (upper) and Scenario 2 (LOCO) are launched here. Cross-res
# and HRDA/GSDPE are commented out in the original workflow; we preserve that.

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR_BASE="${BASE_PATH}/data"

# NEW: separate output tree for probability maps so we don't clobber the
# existing binary-mask predictions in Test_img_results/
INFERENCE_OUTPUT_DIR="${BASE_PATH}/data/Test_img_probs/"

CITIES=("chicago" "miami" "phoenix")
RESOLUTIONS=("highres" "midres")

CITIES=("chicago" "miami" "phoenix")
RESOLUTIONS=("highres" "midres")
MODELS=("mamnet" "oglanet" "dinov3")

declare -A LOCO_TEST_CITY
LOCO_TEST_CITY[0]="phoenix"
LOCO_TEST_CITY[1]="miami"
LOCO_TEST_CITY[2]="chicago"

SUBMITTED=0
MISSING=0
FAILED=0

find_checkpoint() {
    local model_type=$1
    local pattern=$2
    local search_dir="${OUTPUT_DIR_BASE}/${model_type}/outputs"

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
    local model_type=$1
    local model_variant=$2
    local test_type=$3
    local city=$4
    local train_res=$5
    local test_res=$6
    local checkpoint_path=$7
    local data_root=$8
    local job_name=$9

    local log_dir="${INFERENCE_OUTPUT_DIR}/inference_logs/${model_type}"
    mkdir -p "$log_dir"

    sbatch --output="${log_dir}/${job_name}.out" \
           --job-name="$job_name" \
           --export=PROJECT_ROOT=${PROJECT_ROOT},MODEL_TYPE="$model_type",\
MODEL_VARIANT="$model_variant",\
TEST_TYPE="$test_type",\
CITY="$city",\
TRAIN_RES="$train_res",\
TEST_RES="$test_res",\
CHECKPOINT_PATH="$checkpoint_path",\
DATA_ROOT="$data_root",\
OUTPUT_DIR="$INFERENCE_OUTPUT_DIR" \
           run_inference_probs.sh

    if [ $? -eq 0 ]; then
        echo "  ✓ Submitted: $job_name"
        ((SUBMITTED++))
    else
        echo "  ✗ Failed to submit: $job_name"
        ((FAILED++))
    fi
}

# ===========================================================================
# SCENARIO 1 — WITHIN-CITY MODELS (upper bounds)
# ===========================================================================
# echo "========================================================================"
# echo "SCENARIO 1: WITHIN-CITY MODELS (upper bounds) — PROBABILITY OUTPUT"
# echo "========================================================================"

# CITIES=("miami")
# RESOLUTIONS=("highres")
# MODELS=("dinov3")

# for model_type in "${MODELS[@]}"; do
    # for city in "${CITIES[@]}"; do
        # for train_res in "${RESOLUTIONS[@]}"; do
            # echo ""
            # echo "${model_type^^} — $city — $train_res"
            # echo "----------------------------------------------------------------"

            # checkpoint=$(find_checkpoint "$model_type" \
                         # "${model_type}_${city}_${train_res}_*")

            # if [ -z "$checkpoint" ]; then
                # echo "  ⚠ Checkpoint not found: ${model_type}_${city}_${train_res}_*"
                # ((MISSING++))
                # continue
            # fi
            # echo "  ✓ Found checkpoint"

            # data_root="${BASE_DATA_ROOT}/${city}/${train_res}"
            # job_name="prob_upper_${model_type}_${city}_${train_res}"
            # submit_job "$model_type" "base" "upper" \
                       # "$city" "$train_res" "$train_res" \
                       # "$checkpoint" "$data_root" "$job_name"
        # done
    # done
# done

# ===========================================================================
# SCENARIO 2 — LOCO MODELS
# ===========================================================================
echo ""
echo "========================================================================"
echo "SCENARIO 2: LOCO MODELS — PROBABILITY OUTPUT"
echo "========================================================================"

LOCO_METHODS=("vanilla" "fda" "segdesic" "iim" "isw" "mrfp_plus" "fada")
LOCO_METHODS=("mrfp_plus")
RESOLUTIONS=("highres")
MODELS=("oglanet")

for model_type in "${MODELS[@]}"; do
    for method in "${LOCO_METHODS[@]}"; do
        for fold_id in 1 2; do
            test_city="${LOCO_TEST_CITY[$fold_id]}"

            for res in "${RESOLUTIONS[@]}"; do
                echo ""
                echo "${model_type^^} — ${method^^} — fold $fold_id ($test_city) — $res"
                echo "----------------------------------------------------------------"

                if [ "$method" = "vanilla" ]; then
                    pattern="${model_type}_loco_holdout_${test_city}_${res}_*"
                else
                    pattern="${model_type}_${method}_loco_holdout_${test_city}_${res}_*"
                fi

                checkpoint=$(find_checkpoint "$model_type" "$pattern")

                if [ -z "$checkpoint" ]; then
                    echo "  ⚠ Checkpoint not found: $pattern"
                    ((MISSING++))
                    continue
                fi
                echo "  ✓ Found checkpoint"

                data_root="${BASE_DATA_ROOT}/${test_city}/${res}"
                job_name="prob_loco_${model_type}_${method}_${test_city}_${res}"

                submit_job "$model_type" "$method" "loco" \
                           "$test_city" "$res" "$res" \
                           "$checkpoint" "$data_root" "$job_name"
            done
        done
    done
done

# ===========================================================================
# SUMMARY
# ===========================================================================
echo ""
echo "========================================================================"
echo "SUBMISSION SUMMARY"
echo "========================================================================"
echo "✓ Submitted : $SUBMITTED"
echo "⚠ Missing   : $MISSING"
echo "✗ Failed    : $FAILED"
echo ""
echo "Expected:"
echo "  Upper : 3 models × 3 cities × 2 resolutions = 18 jobs"
echo "  LOCO  : 3 models × ${#LOCO_METHODS[@]} methods × 3 folds × 2 resolutions = $(( 3 * ${#LOCO_METHODS[@]} * 3 * 2 )) jobs"
echo "  Total : $(( 18 + 3 * ${#LOCO_METHODS[@]} * 3 * 2 )) jobs"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "========================================================================"