#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"

# Master script to submit all shadow detection inference jobs.
# Handles all test scenarios: within-city (upper bounds), LOCO, and
# cross-resolution.
#
# LOCO methods (updated):
#   removed  : mcl
#   added    : iim, isw, mrfp_plus, fada
#   retained : vanilla, fda, segdesic
#
# ISW note: 'isw' uses the same base model architecture as 'vanilla'.
#   The ISW auxiliary loss and feature hooks are training-only; they leave
#   no trace in the saved weights.  run_inference.py handles this correctly.

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR_BASE="${BASE_PATH}/data"
INFERENCE_OUTPUT_DIR="${BASE_PATH}/data/Test_img_results/"

CITIES=("chicago" "miami" "phoenix")
RESOLUTIONS=("highres" "midres")
MODELS=("mamnet" "oglanet" "dinov3")

# LOCO fold mapping: fold 0 = phoenix, fold 1 = miami, fold 2 = chicago
declare -A LOCO_TEST_CITY
LOCO_TEST_CITY[0]="phoenix"
LOCO_TEST_CITY[1]="miami"
LOCO_TEST_CITY[2]="chicago"

SUBMITTED=0
MISSING=0
FAILED=0

# ---------------------------------------------------------------------------
# find_checkpoint MODEL_TYPE GLOB_PATTERN
#   Searches OUTPUT_DIR_BASE/MODEL_TYPE/outputs/ for a directory matching
#   GLOB_PATTERN and returns the path of the most-recently-modified
#   checkpoint_best.pth found, or an empty string if none exists.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# submit_job  MODEL_TYPE MODEL_VARIANT TEST_TYPE CITY TRAIN_RES TEST_RES
#             CHECKPOINT_PATH DATA_ROOT JOB_NAME
# ---------------------------------------------------------------------------
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
           run_inference.sh

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
# echo "SCENARIO 1: WITHIN-CITY MODELS (upper bounds)"
# echo "========================================================================"

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
                # ((MISSING+=2))
                # continue
            # fi
            # echo "  ✓ Found checkpoint"

            # # Same-resolution inference (upper bound for LOCO comparisons)
            # data_root="${BASE_DATA_ROOT}/${city}/${train_res}"
            # job_name="inf_upper_${model_type}_${city}_${train_res}"
            # submit_job "$model_type" "base" "upper" \
                       # "$city" "$train_res" "$train_res" \
                       # "$checkpoint" "$data_root" "$job_name"

            # # Cross-resolution inference (upper bound for cross-res comparisons)
            # if [ "$train_res" = "highres" ]; then
                # other_res="midres"
            # else
                # other_res="highres"
            # fi
            # data_root="${BASE_DATA_ROOT}/${city}/${other_res}"
            # job_name="inf_crossres_upper_${model_type}_${city}_${train_res}_to_${other_res}"
            # submit_job "$model_type" "base" "cross-res" \
                       # "$city" "$train_res" "$other_res" \
                       # "$checkpoint" "$data_root" "$job_name"
        # done
    # done
# done

# ===========================================================================
# SCENARIO 2 — LOCO MODELS
#
# Methods:
#   vanilla   — base MAMNet/OGLANet/DINOv3, trained without augmentation
#   fda       — trained with FDA input augmentation (same model arch as vanilla)
#   segdesic  — geographic domain adaptation (SegDesic module)
#   iim       — Illumination-Invariant Module
#   isw       — Instance Selective Whitening (training-only reg.; base arch)
#   mrfp_plus — Multi-Resolution Feature Perturbation+ (training-only perturb.)
#   fada      — Frequency-Adapted Domain Adaptation (FADA adapters active at infer.)
#
# Checkpoint naming convention (exp_name from training scripts):
#   vanilla   : {model}_loco_holdout_{test_city}_{res}_*
#   all others: {model}_{method}_loco_holdout_{test_city}_{res}_*
# ===========================================================================
echo ""
echo "========================================================================"
echo "SCENARIO 2: LOCO MODELS"
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

                # Build checkpoint glob pattern
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
                job_name="inf_loco_${model_type}_${method}_${test_city}_${res}"

                submit_job "$model_type" "$method" "loco" \
                           "$test_city" "$res" "$res" \
                           "$checkpoint" "$data_root" "$job_name"
            done
        done
    done
done

# # ===========================================================================
# # SCENARIO 3 — HRDA CROSS-RESOLUTION
# # ===========================================================================
# echo ""
# echo "========================================================================"
# echo "SCENARIO 3: HRDA CROSS-RESOLUTION"
# echo "========================================================================"

# HRDA_MODELS=("mamnet" "dinov3")   # OGLANet has no HRDA variant

# for model_type in "${HRDA_MODELS[@]}"; do
    # for city in "${CITIES[@]}"; do
        # for direction in "midres:highres" "highres:midres"; do
            # IFS=':' read -r train_res test_res <<< "$direction"

            # echo ""
            # echo "${model_type^^} — HRDA — $city — ${train_res}→${test_res}"
            # echo "----------------------------------------------------------------"

            # if [ "$model_type" = "dinov3" ]; then
                # pattern="dinov3_hrda_${city}_*"
            # else
                # pattern="mamnet_hrda_${city}_${train_res}_*"
            # fi

            # checkpoint=$(find_checkpoint "$model_type" "$pattern")

            # if [ -z "$checkpoint" ]; then
                # echo "  ⚠ Checkpoint not found: $pattern"
                # ((MISSING++))
                # continue
            # fi
            # echo "  ✓ Found checkpoint"

            # data_root="${BASE_DATA_ROOT}/${city}/${test_res}"
            # job_name="inf_hrda_${model_type}_${city}_${train_res}_to_${test_res}"

            # submit_job "$model_type" "hrda" "cross-res" \
                       # "$city" "$train_res" "$test_res" \
                       # "$checkpoint" "$data_root" "$job_name"
        # done
    # done
# done

# # ===========================================================================
# # SCENARIO 4 — GSDPE CROSS-RESOLUTION
# # ===========================================================================
# echo ""
# echo "========================================================================"
# echo "SCENARIO 4: GSDPE CROSS-RESOLUTION"
# echo "========================================================================"

# for model_type in "${MODELS[@]}"; do
    # for city in "${CITIES[@]}"; do
        # for direction in "midres:highres" "highres:midres"; do
            # IFS=':' read -r train_res test_res <<< "$direction"

            # echo ""
            # echo "${model_type^^} — GSDPE — $city — ${train_res}→${test_res}"
            # echo "----------------------------------------------------------------"

            # pattern="${model_type}_gsdpe_${city}_${train_res}_*"
            # checkpoint=$(find_checkpoint "$model_type" "$pattern")

            # if [ -z "$checkpoint" ]; then
                # echo "  ⚠ Checkpoint not found: $pattern"
                # ((MISSING++))
                # continue
            # fi
            # echo "  ✓ Found checkpoint"

            # data_root="${BASE_DATA_ROOT}/${city}/${test_res}"
            # job_name="inf_gsdpe_${model_type}_${city}_${train_res}_to_${test_res}"

            # submit_job "$model_type" "gsdpe" "cross-res" \
                       # "$city" "$train_res" "$test_res" \
                       # "$checkpoint" "$data_root" "$job_name"
        # done
    # done
# done

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
echo "Expected LOCO jobs:"
echo "  ${#LOCO_METHODS[@]} methods × 3 folds × 2 resolutions × 3 models"
echo "  = $(( ${#LOCO_METHODS[@]} * 3 * 2 * 3 )) jobs"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "========================================================================"