#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_sib_ablation_submit.sh
#
# Submits SIB component ablation experiments A1–A10 for OGLANet.
#
# Base config: C4 = Haar + VIB + Aug + AB + SAG + aFDA(0.005) + Ctr
# (= M1 config from Round 1)
#
# Each ablation modifies EXACTLY ONE component from C4:
#
#   A1:  No-VIB on F_LL         — skip VIB on LL, keep VIB on LH/HL/HH
#   A2:  Uniform-β              — remove intensity-adaptive β (AB=0)
#   A3:  Symmetric VIB          — force all subbands to use beta_content
#   A4:  No content augmentation — remove ContentAugmentation entirely
#   A5:  Aug-only-AdaIN         — keep style perturbation, remove cross-city mixing
#   A6:  Aug all subbands       — apply augmentation to LH/HL/HH too (MRFP+ analog)
#   A7:  No-SAG                 — remove Skip Attention Gates
#   A8:  No-FDA/contrast        — remove CNN preprocessing (FDA + contrast channel)
#   A9:  No-WT                  — remove Haar decomposition (VIB on full features)
#   A10: VIB on wrong subband   — VIB only on HL (inverse evidence)
#
# Total: 10 ablations × 3 LOCO folds = 30 SLURM jobs

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
# Helper: submit one ablation config across all 3 LOCO folds
#
# All parameters default to C4 base values. Each ablation overrides
# exactly one parameter.
#
# Args:
#   $1   TAG               — experiment tag
#   $2   HAAR               (default 1)
#   $3   VIB                (default 1)
#   $4   AUG                (default 1)
#   $5   AB                 (default 1)
#   $6   FDA                (default 1)
#   $7   SAG                (default 1)
#   $8   CTR                (default 1)
#   $9   FDA_L              (default 0.005)
#   $10  SKIP_LL_VIB        (default 0)
#   $11  SYMMETRIC_BETA     (default 0)
#   $12  AUG_ALL_SUBBANDS   (default 0)
#   $13  VIB_ONLY_BAND      (default "" = all bands)
#   $14  AUG_P_MIX          (default "" = use script default 0.3)
# =====================================================================
submit_ablation() {
    local TAG=${1}
    local HAAR=${2:-1}
    local VIB=${3:-1}
    local AUG=${4:-1}
    local AB=${5:-1}
    local FDA=${6:-1}
    local SAG=${7:-1}
    local CTR=${8:-1}
    local FL=${9:-0.005}
    local SKIP_LL=${10:-0}
    local SYM_BETA=${11:-0}
    local AUG_ALL=${12:-0}
    local VIB_BAND=${13:-}
    local P_MIX=${14:-}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}"
    echo "  SAG=${SAG}  CTR=${CTR}  FDA_L=${FL}"
    echo "  SKIP_LL_VIB=${SKIP_LL}  SYM_BETA=${SYM_BETA}  AUG_ALL=${AUG_ALL}"
    echo "  VIB_ONLY_BAND=${VIB_BAND:-all}  AUG_P_MIX=${P_MIX:-default}"

    for fold_id in 0 1 2; do
        holdout="${FOLD_NAMES[$fold_id]}"
        name="oglanet_sib_${TAG}__loco_holdout_${holdout}__highres"
        outfile="${LOG_DIR}/${name}.out"
        fda_tgt="${BASE_DATA_ROOT}/${holdout}/highres"

        # Build export string — no duplicate keys
        EXPORTS="MODE=loco"
        EXPORTS="${EXPORTS},BASE_DATA_ROOT=${BASE_DATA_ROOT}"
        EXPORTS="${EXPORTS},RESOLUTION=highres"
        EXPORTS="${EXPORTS},FOLD_ID=${fold_id}"
        EXPORTS="${EXPORTS},OUTPUT_DIR=${OUTPUT_DIR}/${name}"
        EXPORTS="${EXPORTS},FDA_TARGET_ROOT=${fda_tgt}"
        EXPORTS="${EXPORTS},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR}"
        EXPORTS="${EXPORTS},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT}"
        EXPORTS="${EXPORTS},OGLANET_OUTPUT_DIR=${OGLANET_OUTPUT_DIR}"
        EXPORTS="${EXPORTS},BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE}"
        # C4 base flags (with per-ablation overrides)
        EXPORTS="${EXPORTS},USE_HAAR=${HAAR}"
        EXPORTS="${EXPORTS},USE_VIB=${VIB}"
        EXPORTS="${EXPORTS},USE_CONTENT_AUG=${AUG}"
        EXPORTS="${EXPORTS},ADAPTIVE_BETA=${AB}"
        EXPORTS="${EXPORTS},USE_FDA=${FDA}"
        EXPORTS="${EXPORTS},USE_SAG=${SAG}"
        EXPORTS="${EXPORTS},USE_CONTRAST=${CTR}"
        EXPORTS="${EXPORTS},FDA_L=${FL}"
        # Ablation-specific flags
        EXPORTS="${EXPORTS},SKIP_LL_VIB=${SKIP_LL}"
        EXPORTS="${EXPORTS},SYMMETRIC_BETA=${SYM_BETA}"
        EXPORTS="${EXPORTS},AUG_ALL_SUBBANDS=${AUG_ALL}"
        [ -n "${VIB_BAND}" ] && EXPORTS="${EXPORTS},VIB_ONLY_BAND=${VIB_BAND}"
        [ -n "${P_MIX}" ]    && EXPORTS="${EXPORTS},AUG_P_MIX=${P_MIX}"

        echo "  - fold=${fold_id} (holdout: ${holdout})"

        if [ "${DRY_RUN}" -eq 1 ]; then
            echo "    [DRY RUN] ${name}"
            continue
        fi

        sbatch \
            --output="${outfile}" \
            --job-name="${name:0:60}" \
            --export=PROJECT_ROOT=${PROJECT_ROOT},"${EXPORTS}" \
            oglanet_sib.sh
    done
}

# ─────────────────────────────────────────────────────────────────────────────
echo "========================================"
echo "  OGLANet + SIB — Ablation Submission"
echo "  (A1–A10, each vs C4 base)"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "========================================"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Logs:    ${LOG_DIR}/oglanet_sib_A*.out"
echo ""

# =====================================================================
# C4 base for reference (NOT submitted — already run as M1):
#   HAAR=1 VIB=1 AUG=1 AB=1 FDA=1 SAG=1 CTR=1 FL=0.005
#   All ablation flags = 0/empty
# =====================================================================

#                         TAG                    HAAR VIB AUG AB FDA SAG CTR FL    SKIP SYM AALL VIB_BAND P_MIX
# ─────────────────────────────────────────────────────────────────────────────

# A1: No-VIB on F_LL — skip VIB on LL subband, keep VIB on LH/HL/HH
#     Tests D2: is content compression on LL load-bearing?
#     Predicted: large drop on OGLANet (encoder-level failure); moderate on others
submit_ablation "A1_no_vib_ll"        1 1 1 1 1 1 1 0.005  1 0 0  ""   ""

# A2: Uniform-β — remove intensity-adaptive β (same as M7 but with SAG)
#     Tests D1: is intensity conditioning important?
#     Predicted: disproportionate loss in high-intensity deciles
submit_ablation "A2_uniform_beta"     1 1 1 0 1 1 1 0.005  0 0 0  ""   ""

# A3: Symmetric VIB — force beta_content on ALL subbands (no asymmetry)
#     Tests D1+D2: does subband-specific β matter?
#     Predicted: thin-shadow F1 collapse; boundary degradation
submit_ablation "A3_symmetric_vib"    1 1 1 1 1 1 1 0.005  0 1 0  ""   ""

# A4: No content augmentation — remove ContentAugmentation entirely
#     Tests D3: is decoder robustness training important?
#     Predicted: DINOv3 degrades most (decoder failure locus)
submit_ablation "A4_no_aug"           1 1 0 1 1 1 1 0.005  0 0 0  ""   ""

# A5: Aug-only-AdaIN — keep style perturbation, remove cross-city mixing
#     Tests D3 variant: how much of augmentation value is from mixing?
#     Predicted: partial loss vs A4; tells us mixing vs noise contribution
submit_ablation "A5_aug_no_mix"       1 1 1 1 1 1 1 0.005  0 0 0  ""   "0.0"

# A6: Aug all subbands — apply augmentation to LH/HL/HH too (MRFP+ analog)
#     Tests D1+D3 interaction: does indiscriminate augmentation hurt?
#     Predicted: should reproduce MRFP+'s OGLANet-Miami collapse
submit_ablation "A6_aug_all"          1 1 1 1 1 1 1 0.005  0 0 1  ""   ""

# A7: No-SAG — remove Skip Attention Gates
#     Tests D2: do skip connections leak domain info around the bottleneck?
#     Predicted: U-Net architectures degrade from encoder-gain leakage
submit_ablation "A7_no_sag"           1 1 1 1 1 0 1 0.005  0 0 0  ""   ""

# A8: No-FDA/contrast — remove CNN preprocessing entirely
#     Tests confound: are SIB gains from FDA/contrast, not the module itself?
#     Predicted: if SIB is doing the work, drop is small; if FDA, drop is large
submit_ablation "A8_no_preproc"       1 1 1 1 0 1 0 0.005  0 0 0  ""   ""

# A9: No-WT — remove Haar decomposition (VIB on full features)
#     Tests D2: is subband separation itself necessary?
#     Predicted: worse than A3 — without separation, can't target compression
submit_ablation "A9_no_haar"          0 1 1 1 1 1 1 0.005  0 0 0  ""   ""

# A10: VIB on wrong subband — apply VIB only to HL (inverse evidence)
#      Tests that the LL targeting is non-arbitrary
#      Predicted: should be clearly worse than C4; validates design choice
submit_ablation "A10_vib_hl_only"     1 1 1 1 1 1 1 0.005  0 0 0  "HL" ""

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Summary — Ablation Experiments"
echo "========================================"
echo ""
echo "  Ablation    What's changed from C4           Diagnostic tested"
echo "  ─────────   ──────────────────────────────   ─────────────────"
echo "  A1          Skip VIB on LL                   D2 (domain compression)"
echo "  A2          No adaptive β                    D1 (intensity conditioning)"
echo "  A3          Same β for all subbands          D1+D2 (spectral targeting)"
echo "  A4          No content augmentation          D3 (decoder robustness)"
echo "  A5          No cross-city mixing             D3 variant"
echo "  A6          Aug on ALL subbands              D1+D3 (MRFP+ analog)"
echo "  A7          No Skip Attention Gates          D2 (domain bypass)"
echo "  A8          No FDA + no contrast             Confound control"
echo "  A9          No Haar (VIB on full F)          D2 (subband separation)"
echo "  A10         VIB on HL only (wrong band)      Inverse evidence"
echo ""
echo "  Total: 10 configs × 3 folds = 30 SLURM jobs"
echo ""
echo "  Monitor    :  squeue -u \$USER"
echo "  Watch logs :  tail -f ${LOG_DIR}/oglanet_sib_A*.out"
echo "  Check done :  ls ${OUTPUT_DIR}/oglanet_sib_A*/comparison_results.json"
echo ""
echo "  Key claims each ablation enables:"
echo "    A1 drops OGLANet > DINOv3 → content compression essential for encoder failures"
echo "    A2 drops high-intensity decile mIoU → intensity conditioning is targeted"
echo "    A3 collapses thin-shadow F1 → subband asymmetry preserves boundaries"
echo "    A4 drops DINOv3 most → augmentation addresses decoder miscalibration"
echo "    A6 reproduces MRFP+ collapse → why SIB's selective augmentation works"
echo "    A10 clearly worse → LL targeting is non-arbitrary"
echo ""