#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# Submit all Experiment A, B, B2, C jobs and the evaluation job.
#
# Usage:
#   ./submit_experiments.sh                     # all experiments
#   ./submit_experiments.sh --exp a             # only Exp A
#   ./submit_experiments.sh --exp b2            # only Exp B2 (encoder retrain)
#   ./submit_experiments.sh --exp a b2          # A and B2
#   ./submit_experiments.sh --eval-only         # only evaluation
#   ./submit_experiments.sh --eval-robust       # only robust recovery evaluation
#   ./submit_experiments.sh --res midres        # change resolution
#   ./submit_experiments.sh --variants vanilla  # only vanilla (default for b2)

# ============================================================
# SERVER PATHS — uncomment the block for your server
# ============================================================

# --- Gilbreth ---
# TEST_BASE="${PROJECT_ROOT}/data/Final_data_test"
# TRAIN_BASE="${PROJECT_ROOT}/data/Final_data"
# OUTPUT_DATA="${PROJECT_ROOT}/data"
# PRED_BASE="${OUTPUT_DATA}/Test_img_results"
# LOG_DIR="${PRED_BASE}/logs"

# --- NCSA Delta ---
TEST_BASE="${PROJECT_ROOT}/data/Final_data_test"
TRAIN_BASE="${PROJECT_ROOT}/data/Final_data"
OUTPUT_DATA="${PROJECT_ROOT}/data"
PRED_BASE="${OUTPUT_DATA}/Test_img_results"
LOG_DIR="${PRED_BASE}/logs"

# ============================================================
# CONFIGURATION
# ============================================================
declare -A LOCO_HOLDOUT
LOCO_HOLDOUT[0]="phoenix"
LOCO_HOLDOUT[1]="miami"
LOCO_HOLDOUT[2]="chicago"

CITIES=("chicago" "miami" "phoenix")
MODELS=("mamnet" "oglanet" "dinov3")

# All LOCO variants
LOCO_VARIANTS=("vanilla" "fda" "segdesic" "iim" "isw" "mrfp_plus" "fada")

# B2 defaults to vanilla only (diagnostic, not DA-method comparison)
B2_VARIANTS=("vanilla")

# Parse arguments
RES="highres"
EXPERIMENTS=("a" "b" "b2" "c")
EVAL_ONLY=false
EVAL_ROBUST=false
CUSTOM_VARIANTS=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp)       shift; EXPERIMENTS=()
                     while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                         EXPERIMENTS+=("$1"); shift
                     done ;;
        --eval-only)  EVAL_ONLY=true; shift ;;
        --eval-robust) EVAL_ROBUST=true; shift ;;
        --res)        RES="$2"; shift 2 ;;
        --variants)   shift; CUSTOM_VARIANTS=()
                      while [[ "$#" -gt 0 && ! "$1" =~ ^-- ]]; do
                          CUSTOM_VARIANTS+=("$1"); shift
                      done ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# If custom variants specified, override B2_VARIANTS too
if [ -n "$CUSTOM_VARIANTS" ]; then
    B2_VARIANTS=("${CUSTOM_VARIANTS[@]}")
    LOCO_VARIANTS=("${CUSTOM_VARIANTS[@]}")
fi

mkdir -p "${LOG_DIR}"

SUBMITTED=0
MISSING=0
JOB_IDS=""

# ============================================================
# HELPER: find best checkpoint
# ============================================================
find_checkpoint() {
    local model_type=$1
    local pattern=$2
    local search_dir="${OUTPUT_DATA}/${model_type}/outputs"
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

# ============================================================
# HELPER: submit a single experiment job
# ============================================================
submit_job() {
    local exp=$1
    local model_type=$2
    local model_variant=$3
    local holdout_city=$4
    local ckpt=$5
    local time_limit=${6:-2:00:00}
    shift 6
    local extra_env="$*"

    local test_data_root="${TEST_BASE}/${holdout_city}/${RES}"
    local job_name="exp${exp}_${model_type}_${model_variant}_${holdout_city}_${RES}"
    local log="${LOG_DIR}/${job_name}.out"

    local env_str
    env_str="EXPERIMENT=${exp}"
    env_str="${env_str},MODEL_TYPE=${model_type}"
    env_str="${env_str},MODEL_VARIANT=${model_variant}"
    env_str="${env_str},CHECKPOINT_PATH=${ckpt}"
    env_str="${env_str},HOLDOUT_CITY=${holdout_city}"
    env_str="${env_str},RES=${RES}"
    env_str="${env_str},TEST_DATA_ROOT=${test_data_root}"
    [ -n "$extra_env" ] && env_str="${env_str},${extra_env}"

    local job_id
    job_id=$(sbatch --parsable \
                    --output="${log}" \
                    --job-name="${job_name}" \
                    --time="${time_limit}" \
                    --export=PROJECT_ROOT=${PROJECT_ROOT},"${env_str}" \
                    run_experiments.sh)

    if [ $? -eq 0 ]; then
        echo "  ✓ Submitted: ${job_name} (Job ${job_id})"
        JOB_IDS="${JOB_IDS}:${job_id}"
        ((SUBMITTED++))
    else
        echo "  ✗ Failed to submit: ${job_name}"
    fi
}

# # ============================================================
# # SKIP EXPERIMENTS IF EVAL-ONLY
# # ============================================================
# if [ "${EVAL_ONLY}" = false ] && [ "${EVAL_ROBUST}" = false ]; then

# # ============================================================
# # EXPERIMENT A: Decoder Retraining
# # ============================================================
# if [[ " ${EXPERIMENTS[*]} " =~ " a " ]]; then
    # echo ""
    # echo "========================================================"
    # echo "EXPERIMENT A: Decoder Retraining"
    # echo "========================================================"

    # EXP_A_BASE="${PRED_BASE}/experiment_a"

    # for model_type in "${MODELS[@]}"; do
        # for fold_id in 0 1 2; do
            # holdout="${LOCO_HOLDOUT[$fold_id]}"

            # for variant in "${LOCO_VARIANTS[@]}"; do
                # if [ "$variant" = "vanilla" ]; then
                    # pattern="${model_type}_loco_holdout_${holdout}_${RES}_*"
                # else
                    # pattern="${model_type}_${variant}_loco_holdout_${holdout}_${RES}_*"
                # fi

                # ckpt=$(find_checkpoint "$model_type" "$pattern")
                # if [ -z "$ckpt" ]; then
                    # echo "  ⚠ Missing: ${model_type}/${variant} holdout=${holdout}"
                    # ((MISSING++))
                    # continue
                # fi

                # train_root="${TRAIN_BASE}/${holdout}/${RES}"
                # [ ! -d "$train_root" ] && train_root="${TEST_BASE}/${holdout}/${RES}"

                # extra="TRAIN_DATA_ROOT=${train_root}"
                # extra="${extra},OUTPUT_BASE=${EXP_A_BASE}"
                # extra="${extra},TRAIN_FRACTION=0.25"
                # extra="${extra},EPOCHS=30"
                # extra="${extra},LR=0.001"

                # submit_job "a" "$model_type" "$variant" "$holdout" "$ckpt" \
                           # "6:00:00" "$extra"
            # done
        # done
    # done
# fi

# # ============================================================
# # EXPERIMENT B: BN Statistics Swap (CNN models only)
# # ============================================================
# if [[ " ${EXPERIMENTS[*]} " =~ " b " ]]; then
    # echo ""
    # echo "========================================================"
    # echo "EXPERIMENT B: BN Statistics Swap (MAMNet, OGLANet only)"
    # echo "========================================================"

    # EXP_B_BASE="${PRED_BASE}/experiment_b"

    # for model_type in "mamnet" "oglanet"; do
        # for fold_id in 0 1 2; do
            # holdout="${LOCO_HOLDOUT[$fold_id]}"

            # for variant in "${LOCO_VARIANTS[@]}"; do
                # if [ "$variant" = "vanilla" ]; then
                    # pattern="${model_type}_loco_holdout_${holdout}_${RES}_*"
                # else
                    # pattern="${model_type}_${variant}_loco_holdout_${holdout}_${RES}_*"
                # fi

                # ckpt=$(find_checkpoint "$model_type" "$pattern")
                # if [ -z "$ckpt" ]; then
                    # echo "  ⚠ Missing: ${model_type}/${variant} holdout=${holdout}"
                    # ((MISSING++))
                    # continue
                # fi

                # extra="OUTPUT_BASE=${EXP_B_BASE}"
                # submit_job "b" "$model_type" "$variant" "$holdout" "$ckpt" \
                           # "1:30:00" "$extra"
            # done
        # done
    # done
    # echo "  (DINOv3 skipped — uses LayerNorm, no BN running stats to swap)"
# fi

# # ============================================================
# # EXPERIMENT B2: Encoder Retraining
# # ============================================================
# if [[ " ${EXPERIMENTS[*]} " =~ " b2 " ]]; then
    # echo ""
    # echo "========================================================"
    # echo "EXPERIMENT B2: Encoder Retraining on Holdout City"
    # echo "========================================================"
    # echo "  Variants: ${B2_VARIANTS[*]}"
    # echo "  LR: architecture-specific defaults (1e-4 CNN, 1e-5 ViT)"

    # EXP_B2_BASE="${PRED_BASE}/experiment_b2"

    # for model_type in "${MODELS[@]}"; do
        # for fold_id in 0 1 2; do
            # holdout="${LOCO_HOLDOUT[$fold_id]}"

            # for variant in "${B2_VARIANTS[@]}"; do
                # if [ "$variant" = "vanilla" ]; then
                    # pattern="${model_type}_loco_holdout_${holdout}_${RES}_*"
                # else
                    # pattern="${model_type}_${variant}_loco_holdout_${holdout}_${RES}_*"
                # fi

                # ckpt=$(find_checkpoint "$model_type" "$pattern")
                # if [ -z "$ckpt" ]; then
                    # echo "  ⚠ Missing: ${model_type}/${variant} holdout=${holdout}"
                    # ((MISSING++))
                    # continue
                # fi

                # train_root="${TRAIN_BASE}/${holdout}/${RES}"
                # [ ! -d "$train_root" ] && train_root="${TEST_BASE}/${holdout}/${RES}"

                # # LR is NOT set here — experiment_b2 uses architecture-specific
                # # defaults (1e-4 CNN, 1e-5 DINOv3) unless overridden
                # extra="TRAIN_DATA_ROOT=${train_root}"
                # extra="${extra},OUTPUT_BASE=${EXP_B2_BASE}"
                # extra="${extra},TRAIN_FRACTION=0.25"
                # extra="${extra},EPOCHS=30"

                # submit_job "b2" "$model_type" "$variant" "$holdout" "$ckpt" \
                           # "6:00:00" "$extra"
            # done
        # done
    # done
# fi

# # ============================================================
# # EXPERIMENT C: Histogram Matching
# # ============================================================
# if [[ " ${EXPERIMENTS[*]} " =~ " c " ]]; then
    # echo ""
    # echo "========================================================"
    # echo "EXPERIMENT C: Test-Time Histogram Matching"
    # echo "========================================================"

    # EXP_C_BASE="${PRED_BASE}/experiment_c"

    # for model_type in "${MODELS[@]}"; do
        # for fold_id in 0 1 2; do
            # holdout="${LOCO_HOLDOUT[$fold_id]}"

            # for variant in "${LOCO_VARIANTS[@]}"; do
                # if [ "$variant" = "vanilla" ]; then
                    # pattern="${model_type}_loco_holdout_${holdout}_${RES}_*"
                # else
                    # pattern="${model_type}_${variant}_loco_holdout_${holdout}_${RES}_*"
                # fi

                # ckpt=$(find_checkpoint "$model_type" "$pattern")
                # if [ -z "$ckpt" ]; then
                    # echo "  ⚠ Missing: ${model_type}/${variant} holdout=${holdout}"
                    # ((MISSING++))
                    # continue
                # fi

                # source_dirs=""
                # for city in "${CITIES[@]}"; do
                    # if [ "$city" != "$holdout" ]; then
                        # img_dir="${TEST_BASE}/${city}/${RES}/test/images"
                        # [ -d "$img_dir" ] && source_dirs="${source_dirs} ${img_dir}"
                    # fi
                # done
                # source_dirs=$(echo "$source_dirs" | xargs)

                # extra="OUTPUT_BASE=${EXP_C_BASE}"
                # extra="${extra},SOURCE_IMAGE_DIRS=${source_dirs}"
                # submit_job "c" "$model_type" "$variant" "$holdout" "$ckpt" \
                           # "1:30:00" "$extra"
            # done
        # done
    # done
# fi

# fi  # end EVAL_ONLY / EVAL_ROBUST check

# # ============================================================
# # EVALUATION JOBS
# # ============================================================
# echo ""
# echo "========================================================"
# echo "EVALUATION"
# echo "========================================================"

# DEPENDENCY=""
# if [ -n "${JOB_IDS}" ] && [ "${EVAL_ONLY}" = false ] && [ "${EVAL_ROBUST}" = false ]; then
    # JOB_IDS="${JOB_IDS#:}"
    # DEPENDENCY="--dependency=afterany:${JOB_IDS//:/:}"
# fi

# # Standard evaluation (existing)
# if [ "${EVAL_ROBUST}" = false ]; then
    # eval_name="eval_experiments"
    # eval_log="${LOG_DIR}/${eval_name}.out"

    # eval_job=$(sbatch --parsable \
        # --output="${eval_log}" \
        # --job-name="${eval_name}" \
        # --time=4:00:00 \
        # --mem=128G \
        # --cpus-per-task=9 \
        # ${DEPENDENCY} \
        # --export=PROJECT_ROOT=${PROJECT_ROOT},"EXPERIMENT=eval,\
# EVAL_EXPERIMENTS=${EXPERIMENTS[*]},\
# EVAL_MODELS=${MODELS[*]},\
# EVAL_RES=${RES}" \
        # run_experiments.sh)

    # if [ $? -eq 0 ]; then
        # echo "  ✓ Standard evaluation submitted (Job ${eval_job})"
        # [ -n "${DEPENDENCY}" ] && echo "    Waits for: ${JOB_IDS//:/, }"
    # fi
# fi

# Robust recovery evaluation (new)
if [ "${EVAL_ROBUST}" = true ] || [[ " ${EXPERIMENTS[*]} " =~ " b2 " ]]; then
    eval_robust_name="eval_robust_recovery"
    eval_robust_log="${LOG_DIR}/${eval_robust_name}.out"

    # If running after other jobs, add dependency on eval_job too
    ROBUST_DEP="${DEPENDENCY}"
    if [ -n "${eval_job}" ]; then
        if [ -n "${ROBUST_DEP}" ]; then
            ROBUST_DEP="${ROBUST_DEP}:${eval_job}"
        else
            ROBUST_DEP="--dependency=afterany:${eval_job}"
        fi
    fi

    eval_robust_job=$(sbatch --parsable \
        --output="${eval_robust_log}" \
        --job-name="${eval_robust_name}" \
        --time=2:00:00 \
        --mem=128G \
        --cpus-per-task=9 \
        ${ROBUST_DEP} \
        --export=PROJECT_ROOT=${PROJECT_ROOT},"EXPERIMENT=eval_robust,\
EVAL_EXPERIMENTS=a b2,\
EVAL_MODELS=${MODELS[*]},\
EVAL_RES=${RES},\
N_BOOT=10000" \
        run_experiments.sh)

    if [ $? -eq 0 ]; then
        echo "  ✓ Robust recovery evaluation submitted (Job ${eval_robust_job})"
    fi
fi

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "========================================================"
echo "SUBMISSION SUMMARY"
echo "========================================================"
echo "  Submitted : ${SUBMITTED} jobs"
echo "  Missing   : ${MISSING} checkpoints"
echo "  Experiments: ${EXPERIMENTS[*]}"
if [[ " ${EXPERIMENTS[*]} " =~ " b2 " ]]; then
    echo "  B2 variants: ${B2_VARIANTS[*]}"
fi
echo "  Resolution: ${RES}"
echo "  Logs      : ${LOG_DIR}"
echo ""
echo "  Monitor with: squeue -u \$USER"
echo "========================================================"