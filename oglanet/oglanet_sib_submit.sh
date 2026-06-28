#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_sib_submit.sh
#
# Queues OGLANet+SIB training jobs on SLURM.
#
# Round 1 (8 configs × 3 folds = 24 jobs):
# M1:  SIB+aFDA+SAG/ctr   Full method (M2 + SAG)
# M2:  SIB+aFDA/ctr        Reference config (Haar+VIB+Aug+AB+aFDA+Ctr)
# M3:  SIB+sFDA/ctr        Standard FDA (M2 but β_FDA=0.01)
# M4:  SIB/ctr             No FDA (M2 − FDA)
# M5:  SIB-noAug+aFDA/ctr  No content aug (M2 − Aug)
# M6:  SIB-uniformVIB      No Haar (M2 − Haar)
# M7:  SIB-fixedBeta       No adaptive β (M2 − AB)
# M8:  aFDA-only           Baseline (no SIB)
#
# Round 2 (4 new configs × 3 folds = 12 jobs):
# O9:  SAG without adaptive β   — Tests if SAG alone (not AB) drove M1's win
# O10: Lower β (0.0005)         — Tests gentler compression
# O11: Passthrough gate + SAG   — Tests VIB auto-disable mechanism
# O12: Standard FDA + SAG, no AB — Tests sFDA+SAG combination
#
# Round 3 (1 new config × 3 folds = 3 jobs):
# O13: Module bypass gate       — Full C4 + module-level residual bypass

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OGLANET_OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
LOG_DIR="${BASE_PATH}/data/oglanet/sib_logs"

# Fold mapping: 0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago
FOLD_NAMES=("phoenix" "miami" "chicago")
BOUNDARY_TOLERANCE=2

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
#   $13  USE_PASSTHROUGH_GATE — 0 or 1 (default 0 if omitted)
#   $14  USE_MODULE_BYPASS   — 0 or 1 (default 0 if omitted)
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
    local GATE=${13:-0}
    local MOD_BYPASS=${14:-0}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}"
    echo "  SAG=${SAG}  MS=${MS}  CTR=${CTR}  FDA_L=${FL}  BETA=${BC:-default}"
    echo "  GATE=${GATE}  MODULE_BYPASS=${MOD_BYPASS}"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="oglanet_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"

            # FDA target root = holdout city images (unlabeled)
            fda_tgt="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            if [ "${DRY_RUN}" -eq 1 ]; then
                echo "    [DRY RUN] name=${name}"
                echo "              out=${outfile}"
                continue
            fi

            sbatch \
                --output="${outfile}" \
                --job-name="${name:0:60}" \
                --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR}/${name},FDA_TARGET_ROOT=${fda_tgt},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},OGLANET_OUTPUT_DIR=${OGLANET_OUTPUT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},USE_SAG=${SAG},USE_MULTISCALE_SIB=${MS},USE_CONTRAST=${CTR},FDA_L=${FL},BETA_CONTENT=${BC},BOUNDARY_TOLERANCE=${BT},USE_PASSTHROUGH_GATE=${GATE},USE_MODULE_BYPASS=${MOD_BYPASS} \
                oglanet_sib.sh
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  OGLANet + SIB — Experiment Submission"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "========================================"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Logs:    ${LOG_DIR}/oglanet_sib_*.out"
echo ""

# =====================================================================
# ROUND 1 — Original 8 experiments (M1–M8)
# Uncomment these if you need to re-run them.
# =====================================================================

# # M1: SIB+aFDA+SAG/ctr — Full method (reference + SAG)
# submit_loco "M1_haar_vib_aug_ab_fda_sag_ctr" 1 1 1 1 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # M2: SIB+aFDA/ctr — Reference config
# submit_loco "M2_haar_vib_aug_ab_fda_ctr" 1 1 1 1 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # M3: SIB+sFDA/ctr — Standard FDA (β_FDA=0.01)
# submit_loco "M3_haar_vib_aug_ab_sfda_ctr" 1 1 1 1 1 0 0 1 0.01 "" ${BOUNDARY_TOLERANCE} 0 0

# # M4: SIB/ctr — No FDA
# submit_loco "M4_haar_vib_aug_ab_ctr" 1 1 1 1 0 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # M5: SIB-noAug+aFDA/ctr — No content augmentation
# submit_loco "M5_haar_vib_ab_fda_ctr" 1 1 0 1 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # M6: SIB-uniformVIB+aFDA/ctr — No Haar (uniform VIB)
# submit_loco "M6_vib_aug_ab_fda_ctr" 0 1 1 1 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # M7: SIB-fixedBeta+aFDA/ctr — No adaptive β
# submit_loco "M7_haar_vib_aug_fda_ctr" 1 1 1 0 1 0 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # M8: aFDA-only — Baseline (no SIB)
# submit_loco "M8_fda_only_baseline" 0 0 0 0 1 0 0 0 0.005 "" ${BOUNDARY_TOLERANCE} 0 0


# =====================================================================
# ROUND 2 — Experiments (O9–O12)
# Uncomment these if you need to re-run them.
# =====================================================================

# # O9: SAG without adaptive β
# submit_loco "O9_haar_vib_aug_fda_sag_ctr" 1 1 1 0 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0

# # O10: Lower compression (β_content=0.0005)
# submit_loco "O10_haar_vib_aug_fda_lowbeta_ctr" 1 1 1 0 1 0 0 1 0.005 0.0005 ${BOUNDARY_TOLERANCE} 0 0

# # O11: Passthrough gate + SAG
# submit_loco "O11_haar_vib_aug_fda_sag_gate_ctr" 1 1 1 0 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 1 0

# # O12: Standard FDA + SAG, no adaptive β
# submit_loco "O12_haar_vib_aug_sfda_sag_ctr" 1 1 1 0 1 1 0 1 0.01 "" ${BOUNDARY_TOLERANCE} 0 0


# =====================================================================
# ROUND 3 — Module Bypass Gate (O13)
#
# Full C4 config + module-level residual bypass gate.
# C4 = Haar + VIB + Aug + AB + SAG + aFDA + Ctr
#
# The bypass gate wraps the entire SIB pipeline:
#   F_out = α · F_sib + (1 − α) · F_encoder
# where α = sigmoid(linear(GAP(F_encoder))), per-sample.
# Initialized with bias=+2.0 → α ≈ 0.88 (SIB "on" by default).
#
# Expected behaviour:
#   - Cities where SIB helps: α stays high (~0.9)
#   - Cities where SIB hurts: α drops toward 0 (auto-bypass)
#
# Diagnostic: bypass_gate_alpha.json saved in output dir.
#   cat <output_dir>/bypass_gate_alpha.json | python -m json.tool | head
#
# Decision logic:
#   O13 vs M1  → Does bypass gate preserve M1 gains where SIB helps?
#   O13 vs M7  → Does bypass gate + SAG + AB beat fixed-beta baseline?
#   Check per-city α values to confirm gate is modulating appropriately.
# =====================================================================
# submit_loco "O13_haar_vib_aug_ab_fda_sag_bypass_ctr" 1 1 1 1 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 1

# C4-clean: Haar + VIB + SAG + FDA + Ctr (no aug, no AB) — base for tempscale
submit_loco "C4clean_haar_vib_sag_fda_ctr" 1 1 0 0 1 1 0 1 0.005 "" ${BOUNDARY_TOLERANCE} 0 0


echo ""
echo "========================================"
echo "  Summary — Round 3 (1 new experiment)"
echo "========================================"
echo ""
echo "  O13:  Module bypass gate on C4 base"
echo "        (Haar+VIB+Aug+AB+SAG+aFDA+Ctr + module bypass)"
echo ""
echo "  Decision tree:"
echo "    O13 vs M1   → Bypass gate value (preserves gains, avoids regressions?)"
echo "    O13 α vals  → Per-city gate behaviour (publishable diagnostic)"
echo ""
echo "  Total new jobs: 1 config × 3 folds = 3 SLURM jobs"
echo ""
echo "  Monitor jobs  :  squeue -u \$USER"
echo "  Watch a log   :  tail -f ${LOG_DIR}/oglanet_sib_O13_*.out"
echo "  Check outputs :  ls ${OUTPUT_DIR}/oglanet_sib_O13_*/comparison_results.json"
echo "  Check α values:  cat ${OUTPUT_DIR}/oglanet_sib_O13_*/bypass_gate_alpha.json | python -m json.tool | head"
echo ""