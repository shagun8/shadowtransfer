#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_sib_tempscale_submit.sh
#
# Queues C4-clean + class-conditional temperature scaling jobs.
#
# C4-clean: surviving SIB components only (Haar + VIB + SAG + FDA + Ctr).
#   Drops content-aug (failed A4) and adaptive-β (failed A2).
# Tempscale: post-hoc fit on source-city val, applied at test on held-out city.
#
# Configurations:
#   TS1: Vanilla + tempscale          (no SIB, just tempscale on Vanilla logits)
#   TS2: V+mod1 (Haar+VIB) + tempscale          (§4.2 + §4.3)
#   TS3: V+mod2 (SAG+FDA) + tempscale            (§4.4 + §4.3)
#   TS4: V+SIB  (C4-clean) + tempscale           (§4.2 + §4.4 + §4.3) — headline

# ---- Server paths ----
BASE_PATH="${PROJECT_ROOT}/"
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
MAMNET_OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
LOG_DIR="${BASE_PATH}/data/mamnet/sib_logs"
BOUNDARY_TOLERANCE=2

FOLD_NAMES=("phoenix" "miami" "chicago")

DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
    esac
done

# Args: TAG HAAR VIB AUG AB FDA SAG MS CTR FDA_L BETA BT GATE MOD_BYPASS BMM
submit_tempscale() {
    local TAG=$1
    local HAAR=$2 VIB=$3 AUG=$4 AB=$5 FDA=$6 SAG=$7 MS=$8 CTR=$9
    local FL=${10} BC=${11} BT=${12} GATE=${13} MOD_BYPASS=${14} BMM=${15}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO + tempscale) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}"
    echo "  SAG=${SAG}  MS=${MS}  CTR=${CTR}  FDA_L=${FL}  BETA=${BC:-default}"
    echo "  TEMPSCALE=1"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="mamnet_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"
            fda_tgt="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            if [ "${DRY_RUN}" -eq 1 ]; then
                echo "    [DRY RUN] name=${name}"
                continue
            fi

            mkdir -p "${LOG_DIR}"

            sbatch \
                --output="${outfile}" \
                --job-name="${name:0:60}" \
                --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR}/${name},FDA_TARGET_ROOT=${fda_tgt},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},MAMNET_OUTPUT_DIR=${MAMNET_OUTPUT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},USE_SAG=${SAG},USE_MULTISCALE_SIB=${MS},USE_CONTRAST=${CTR},FDA_L=${FL},BETA_CONTENT=${BC},BOUNDARY_TOLERANCE=${BT},USE_PASSTHROUGH_GATE=${GATE},USE_MODULE_BYPASS=${MOD_BYPASS},BETA_MAX_MULTIPLIER=${BMM},USE_CLASS_COND_TEMPSCALE=1 \
                mamnet_sib.sh
        done
    done
}

echo "========================================"
echo "  MAMNet + SIB — C4-clean + Tempscale"
if [ "${DRY_RUN}" -eq 1 ]; then echo "  MODE: DRY RUN"; fi
echo "========================================"

# ─────────────────────────────────────────────────────────────────────
# TS4: V + SIB (C4-clean) + tempscale  — headline configuration
#   Haar + VIB + SAG + FDA + Ctr, no aug, no AB.
#   3 folds × 1 res = 3 jobs.
# ─────────────────────────────────────────────────────────────────────
submit_tempscale "TS4_C4clean_tempscale" 1 1 0 0 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0 ""

# Optional standalone-module rows for the §5.3 table (uncomment to run):
# TS1: Vanilla + tempscale                — pure §4.3 effect
# submit_tempscale "TS1_vanilla_tempscale"     0 0 0 0 0 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0 ""
# TS2: V + mod1 (Haar+VIB) + tempscale    — §4.2 + §4.3
# submit_tempscale "TS2_haarvib_tempscale"     1 1 0 0 0 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0 ""
# TS3: V + mod2 (SAG+FDA) + tempscale     — §4.4 + §4.3
# submit_tempscale "TS3_sagfda_tempscale"      0 0 0 0 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0 ""

echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Results:  ${OUTPUT_DIR}/mamnet_sib_TS4_*/tempscale_results.json"
echo ""