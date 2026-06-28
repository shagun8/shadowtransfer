#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_sib_submit.sh
#
# Queues MAMNet+SIB training jobs on SLURM.
#
# M1:  SIB+aFDA+SAG/ctr    Full method (M2 + SAG)
# M2:  SIB+aFDA/ctr         Reference config (Haar+VIB+Aug+AB+aFDA+Ctr)
# M3:  SIB+sFDA/ctr         Standard FDA (M2 but β_FDA=0.01)
# M4:  SIB/ctr              No FDA (M2 − FDA)
# M5:  SIB-noAug+aFDA/ctr   No content aug (M2 − Aug)
# M6:  SIB-uniformVIB       No Haar (M2 − Haar)
# M7:  SIB-fixedBeta        No adaptive β (M2 − AB)
# M8:  SIB-highBeta         High β_content (M2 + high β)
# M9:  SIB-multiScale       Multi-scale SIB (M2 + ms-SIB)
# M10: SIB+aFDA             No contrast (M2 − Ctr)
#
# Round 2 (fixing unified variant):
# M11: SAG without adaptive β
# M12: Lower β_content (0.0005)
# M13: Passthrough gate (VIB-level auto-disable)
#
# Round 3 (module-level bypass):
# M14: C4 + module bypass gate (entire SIB wrapped in learned residual)
#      = M1 config + --use_module_bypass
#
# Decision logic:
#   Step 1:  M2 vs M6 → Haar decomposition value
#   Step 2:  M2 vs M7 → Intensity adaptation value
#   Step 3:  M2 vs M5 → Content augmentation value
#   Step 4a: M2 vs M4 → FDA necessity
#   Step 4b: M2 vs M3 → FDA attenuation
#   Step 4c: M1 vs M2 → SAG value
#   Step 4d: M2 vs M9 → Multi-scale SIB
#   Step 4e: M2 vs M10 → Contrast channel
#   Step 4f: M2 vs M8 → Beta sensitivity
#   Step 5:  Assemble final method
#
# Round 2 decision logic:
#   M11 vs M1 → Is SAG benefit from AB or from SAG itself?
#   M12 vs M7 → Does lower β beat default β?
#   M13 vs M7 → Does passthrough gate help everywhere?
#
# Round 3 decision logic:
#   M14 vs M1 → Does module bypass gate fix regressions?
#   Check per-city α values: Phoenix α << Chicago/Miami α → gate adapting

# ---- Server paths (uncomment the one you need) ----

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
MAMNET_OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
LOG_DIR="${BASE_PATH}/data/mamnet/sib_logs"
BOUNDARY_TOLERANCE=2

# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
FOLD_NAMES=("phoenix" "miami" "chicago")

# ─────────────────────────────────────────────────────────────────────────────
# Parse flags
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
    esac
done

# =====================================================================
# Helper: submit one SIB configuration across all folds and resolutions
#
# Args:
#   $1   TAG              — experiment tag for naming
#   $2   USE_HAAR         — 0 or 1
#   $3   USE_VIB          — 0 or 1
#   $4   USE_CONTENT_AUG  — 0 or 1
#   $5   ADAPTIVE_BETA    — 0 or 1
#   $6   USE_FDA          — 0 or 1
#   $7   USE_SAG          — 0 or 1
#   $8   USE_MULTISCALE   — 0 or 1
#   $9   USE_CONTRAST     — 0 or 1
#   $10  FDA_L            — float
#   $11  BETA_CONTENT     — float  (empty = use script default)
#   $12  BOUNDARY_TOLERANCE
#   $13  USE_PASSTHROUGH_GATE — 0 or 1
#   $14  USE_MODULE_BYPASS   — 0 or 1
# =====================================================================
submit_loco() {
    local TAG=$1
    local HAAR=$2
    local VIB=$3
    local AUG=$4
    local AB=$5
    local FDA=$6
    local SAG=$7
    local MS=$8
    local CTR=$9
    local FL=${10}
    local BC=${11}
    local BT=${12}
    local GATE=${13}
    local MOD_BYPASS=${14}
	local BMM=${15}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}"
    echo "  SAG=${SAG}  MS=${MS}  CTR=${CTR}  FDA_L=${FL}  BETA=${BC:-default}"
    echo "  GATE=${GATE}  MOD_BYPASS=${MOD_BYPASS}"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="mamnet_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"

            # FDA target root = holdout city images (unlabeled)
            fda_tgt="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            if [ "${DRY_RUN}" -eq 1 ]; then
                echo "    [DRY RUN] name=${name}"
                echo "              out=${outfile}"
                continue
            fi

            mkdir -p "${LOG_DIR}"

            sbatch \
                --output="${outfile}" \
                --job-name="${name:0:60}" \
                --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR}/${name},FDA_TARGET_ROOT=${fda_tgt},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},MAMNET_OUTPUT_DIR=${MAMNET_OUTPUT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},USE_SAG=${SAG},USE_MULTISCALE_SIB=${MS},USE_CONTRAST=${CTR},FDA_L=${FL},BETA_CONTENT=${BC},BOUNDARY_TOLERANCE=${BT},USE_PASSTHROUGH_GATE=${GATE},USE_MODULE_BYPASS=${MOD_BYPASS},BETA_MAX_MULTIPLIER=${BMM} \
                mamnet_sib.sh
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  MAMNet + SIB — Experiment Submission"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "========================================"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Logs:    ${LOG_DIR}/mamnet_sib_*.out"
echo ""

# # =====================================================================
# # M1: SIB+aFDA+SAG/ctr — Full method (reference + SAG)
# # =====================================================================
# submit_loco "M1_haar_vib_aug_ab_fda_sag_ctr" 1 1 1 1 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M2: SIB+aFDA/ctr — Reference config
# #   Haar + VIB + Aug + AB + aFDA + Ctr (no SAG, no ms-SIB)
# # =====================================================================
# submit_loco "M2_haar_vib_aug_ab_fda_ctr" 1 1 1 1 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M3: SIB+sFDA/ctr — Standard FDA (β_FDA=0.01 instead of 0.005)
# #   Tests: Is FDA attenuation beneficial?
# # =====================================================================
# submit_loco "M3_haar_vib_aug_ab_sfda_ctr" 1 1 1 1 1 0 0 1 0.01 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M4: SIB/ctr — No FDA
# #   Tests: Is FDA necessary at all?
# # =====================================================================
# submit_loco "M4_haar_vib_aug_ab_ctr" 1 1 1 1 0 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M5: SIB-noAug+aFDA/ctr — No content augmentation
# #   Tests: Content augmentation value
# # =====================================================================
# submit_loco "M5_haar_vib_ab_fda_ctr" 1 1 0 1 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M6: SIB-uniformVIB+aFDA/ctr — No Haar (uniform VIB)
# #   Tests: Haar decomposition value
# # =====================================================================
# submit_loco "M6_vib_aug_ab_fda_ctr" 0 1 1 1 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M7: SIB-fixedBeta+aFDA/ctr — No adaptive β
# #   Tests: Intensity adaptation value
# # =====================================================================
# submit_loco "M7_haar_vib_aug_fda_ctr" 1 1 1 0 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M8: SIB-highBeta+aFDA/ctr — High β_content (sensitivity test)
# # =====================================================================
# submit_loco "M8_haar_vib_aug_ab_fda_highbeta_ctr" 1 1 1 1 1 0 0 1 0.005 "0.01" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M9: SIB-multiScale+aFDA/ctr — Multi-scale SIB at encoder stages
# #   Tests: Multi-scale SIB value
# # =====================================================================
# submit_loco "M9_haar_vib_aug_ab_fda_multiscale_ctr" 1 1 1 1 1 0 1 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M10: SIB+aFDA — No contrast channel
# #   Tests: Contrast channel value
# # =====================================================================
# submit_loco "M10_haar_vib_aug_ab_fda" 1 1 1 1 1 0 0 0 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M11: SIB+aFDA+SAG/ctr — SAG without adaptive β
# # =====================================================================
# submit_loco "M11_haar_vib_aug_fda_sag_ctr" 1 1 1 0 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M12: SIB+aFDA/ctr — Low β_content (0.0005)
# # =====================================================================
# submit_loco "M12_haar_vib_aug_fda_lowbeta_ctr" 1 1 1 0 1 0 0 1 0.005 "0.0005" ${BOUNDARY_TOLERANCE} 0 0

# # =====================================================================
# # M13: SIB+aFDA+Gate/ctr — Passthrough gate (VIB-level auto-disable)
# # =====================================================================
# submit_loco "M13_haar_vib_aug_fda_gate_ctr" 1 1 1 0 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 1 0

# =====================================================================
# M14: C4 + Module Bypass Gate — entire SIB wrapped in learned residual
#   Base = M1 (Haar+VIB+Aug+AB+aFDA+SAG+Ctr) + module-level bypass gate.
#   Gate: α = sigmoid(linear(GAP(F_encoder))), init bias=+2.0 (SIB "on").
#   F_out = α·F_sib + (1−α)·F_encoder.
#
#   Tests: Can the gate learn to bypass SIB when domain gap is small?
#   Compare: M14 vs M1 (bypass gate effect)
#   Diagnostic: per-city α values in bypass_gate_alpha.json
#
#   HAAR=1  VIB=1  AUG=1  AB=1  FDA=1  SAG=1  MS=0  CTR=1
#   FDA_L=0.005  BETA=default  GATE=0  MOD_BYPASS=1
# =====================================================================
# submit_loco "M14_haar_vib_aug_ab_fda_sag_ctr_modbypass" 1 1 1 1 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 1 ""

# =====================================================================
# M15: C4 + Standard FDA + Lower β_content + Reduced adaptive range
#   Goal: Close MAMNet gap vs standalone FDA.
#   Base = C4/M1 (Haar+VIB+Aug+AB+FDA+SAG+Ctr)
#   Changes from M1:
#     - FDA_L=0.01 (standard, not attenuated) — ablation M3 vs M2 showed +0.69
#     - BETA_CONTENT=0.005 (5× default) — less VIB compression on FDA-corrected features
#     - BETA_MAX_MULTIPLIER=2.0 (reduced from 3.0) — less aggressive adaptive range
#
#   HAAR=1  VIB=1  AUG=1  AB=1  FDA=1  SAG=1  MS=0  CTR=1
#   FDA_L=0.01  BETA=0.005  GATE=0  MOD_BYPASS=0  BMM=2.0
# =====================================================================
# submit_loco "M15_haar_vib_aug_ab_sfda_sag_lowbeta_ctr" 1 1 1 1 1 1 0 1 0.01 "0.005" ${BOUNDARY_TOLERANCE} 0 0 "2.0"

# C4-clean: Haar + VIB + SAG + FDA + Ctr (no aug, no AB) — base for tempscale
submit_loco "C4clean_haar_vib_sag_fda_ctr" 1 1 0 0 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0 ""

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo ""
echo "  Round 1 (ablation — uncomment to re-run):"
echo "    M1-M10: Component ablation study"
echo ""
echo "  Round 2 (fixing unified variant — uncomment to re-run):"
echo "    M11: SAG without adaptive β"
echo "    M12: Lower β_content (0.0005)"
echo "    M13: Passthrough gate (VIB-level)"
echo ""
echo "  Round 3 (module bypass — ACTIVE):"
echo "    M14: C4 + module bypass gate (3 folds × 1 res = 3 jobs)"
echo "         Wraps entire SIB in α·SIB + (1−α)·encoder residual"
echo "         α logged per-image → bypass_gate_alpha.json"
echo ""
echo "  Decision tree after M14 results:"
echo "    1. M14 Phoenix α << M14 Chicago/Miami α → gate adapts to gap"
echo "    2. M14 >= M1 everywhere → adopt module bypass as final method"
echo "    3. M14 Phoenix ≈ vanilla → gate successfully prevents regression"
echo ""
echo "  Monitor jobs  :  squeue -u \$USER"
echo "  Watch a log   :  tail -f ${LOG_DIR}/mamnet_sib_<n>.out"
echo "  Check outputs :  ls ${OUTPUT_DIR}/mamnet_sib_*/comparison_results.json"
echo "  Check alphas  :  cat ${OUTPUT_DIR}/mamnet_sib_M14_*/bypass_gate_alpha.json"
echo ""