#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: dinov3_sib_ablation_submit.sh
#
# SIB Component Ablation Study (§5.3) — DINOv3
#
# Queues ablation experiments A1–A10 for the SIB module on DINOv3.
# Each ablation removes or modifies exactly ONE design choice from
# the C4 baseline (Haar + VIB + Aug + AdaptiveBeta), testing whether
# that component is load-bearing and for which diagnostic stratum.
#
# C4 baseline (= D1): Haar + VIB + Aug + AB
#   (SAG is N/A for DINOv3; FDA is not used for DINOv3)
#
# Ablation table:
# ┌──────┬───────────────────────────────────┬────────────────────────┐
# │ ID   │ What's changed from C4            │ Diagnostic tested      │
# ├──────┼───────────────────────────────────┼────────────────────────┤
# │ A1   │ No content VIB on F_LL            │ D2 (domain compress.)  │
# │ A2   │ Fixed β (no intensity adaptation) │ D1 (intensity target.) │
# │ A3   │ Same high β on LL, LH, HL        │ D1+D2 (spectral tgt.)  │
# │ A4   │ No content augmentation           │ D3 (decoder robust.)   │
# │ A5   │ No cross-city mixing (AdaIN only) │ D3 variant             │
# │ A6   │ Augment ALL subbands              │ D1+D3 (MRFP+ analog)  │
# │ A7   │ No SAG                            │ N/A for DINOv3         │
# │ A8   │ No FDA preprocessing              │ N/A for DINOv3         │
# │ A9   │ No Haar (uniform VIB on full F)   │ D2 (subband separation)│
# │ A10  │ VIB on F_HL (wrong subband)       │ D2 (inverse evidence)  │
# └──────┴───────────────────────────────────┴────────────────────────┘
#
# Total: 8 active ablations × 3 LOCO folds = 24 SLURM jobs

# =====================================================================
# Server paths — uncomment the block for your target server
# =====================================================================

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"
BASE_PATH2="${PROJECT_ROOT}/"

# =====================================================================
# Derived paths (shared across all servers — do not edit)
# =====================================================================
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
DINOV3_OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
LOG_DIR="${BASE_PATH}/data/dinov3/sib_ablation_logs"

# --- Gilbreth ---
# WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# --- Anvil ---
# WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# --- NCSA Delta ---
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Boundary tolerance in pixels
BOUNDARY_TOLERANCE=2

mkdir -p ${LOG_DIR}

FOLD_NAMES=("phoenix" "miami" "chicago")

# =====================================================================
# C4 baseline hyperparameters (shared by all ablations unless overridden)
# =====================================================================
C4_BC=0.01       # vib_beta_content
C4_BE=0.0001     # vib_beta_edge
C4_BS=0.02       # vib_beta_scale
C4_LC=1.0        # lambda_content
C4_LE=0.1        # lambda_edge
C4_WF=0.1        # vib_warmup_fraction
C4_FL=0.01       # fda_L (not used for DINOv3, kept for consistency)

# =====================================================================
# Helper: submit one ablation configuration across all 3 LOCO folds
#
# Args:
#   $1   TAG                  — experiment tag (e.g. A1, A2)
#   $2   USE_HAAR             — 0 or 1
#   $3   USE_VIB              — 0 or 1
#   $4   USE_CONTENT_AUG      — 0 or 1
#   $5   ADAPTIVE_BETA        — 0 or 1
#   $6   USE_FDA              — 0 or 1
#   $7   VIB_BETA_CONTENT     — float
#   $8   VIB_BETA_EDGE        — float
#   $9   VIB_BETA_SCALE       — float
#   $10  LAMBDA_CONTENT       — float
#   $11  LAMBDA_EDGE          — float
#   $12  VIB_WARMUP_FRAC      — float
#   $13  FDA_L                — float
#   $14  USE_PASSTHROUGH_GATE — 0 or 1
#   $15  EXP_TAG              — string
#   $16  USE_MODULE_BYPASS    — 0 or 1
#   $17  DISABLE_CONTENT_VIB  — 0 or 1
#   $18  SYMMETRIC_VIB        — 0 or 1
#   $19  AUG_ALL_SUBBANDS     — 0 or 1
#   $20  VIB_ON_HL_ONLY       — 0 or 1
#   $21  AUG_P_MIX            — float (empty = default 0.3)
# =====================================================================
submit_ablation() {
    local TAG=$1
    local HAAR=$2
    local VIB=$3
    local AUG=$4
    local AB=$5
    local FDA=$6
    local BC=$7
    local BE=$8
    local BS=$9
    local LC=${10}
    local LE=${11}
    local WF=${12}
    local FL=${13}
    local GATE=${14:-0}
    local EXP_TAG=${15:-}
    local MOD_BYPASS=${16:-0}
    local DIS_CON_VIB=${17:-0}
    local SYM_VIB=${18:-0}
    local AUG_ALL=${19:-0}
    local VIB_HL=${20:-0}
    local PMIX=${21:-}

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  HAAR=${HAAR}  VIB=${VIB}  AUG=${AUG}  AB=${AB}  FDA=${FDA}"
    echo "  beta_content=${BC}  beta_edge=${BE}  beta_scale=${BS}"
    echo "  lambda_content=${LC}  lambda_edge=${LE}  warmup=${WF}"
    echo "  DISABLE_CONTENT_VIB=${DIS_CON_VIB}  SYMMETRIC_VIB=${SYM_VIB}"
    echo "  AUG_ALL_SUBBANDS=${AUG_ALL}  VIB_ON_HL_ONLY=${VIB_HL}"
    [ -n "${PMIX}" ] && echo "  AUG_P_MIX=${PMIX}"
    echo "  EXP_TAG=${EXP_TAG}"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="dinov3_sib_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${LOG_DIR}/${name}.out"

            # FDA target = held-out test city images
            FDA_TARGET_ROOT="${BASE_DATA_ROOT}/${holdout}/${res}"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            sbatch --output=${outfile} \
                   --job-name=${name} \
                   --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},WEIGHT_DIR=${WEIGHT_DIR},USE_HAAR=${HAAR},USE_VIB=${VIB},USE_CONTENT_AUG=${AUG},ADAPTIVE_BETA=${AB},USE_FDA=${FDA},FDA_TARGET_ROOT=${FDA_TARGET_ROOT},VIB_BETA_CONTENT=${BC},VIB_BETA_EDGE=${BE},VIB_BETA_SCALE=${BS},LAMBDA_CONTENT=${LC},LAMBDA_EDGE=${LE},VIB_WARMUP_FRAC=${WF},FDA_L=${FL},BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},DINOV3_OUTPUT_DIR=${DINOV3_OUTPUT_DIR},USE_PASSTHROUGH_GATE=${GATE},EXP_TAG=${EXP_TAG},USE_MODULE_BYPASS=${MOD_BYPASS},DISABLE_CONTENT_VIB=${DIS_CON_VIB},SYMMETRIC_VIB=${SYM_VIB},AUG_ALL_SUBBANDS=${AUG_ALL},VIB_ON_HL_ONLY=${VIB_HL},AUG_P_MIX=${PMIX} \
                   dinov3_sib.sh
        done
    done
}


echo "============================================================"
echo "  DINOv3 + SIB — §5.3 Component Ablation Study"
echo "  Baseline: C4 (Haar + VIB + Aug + AdaptiveBeta)"
echo "============================================================"
echo "  Output:              ${OUTPUT_DIR}"
echo "  Log dir:             ${LOG_DIR}"
echo "  Weights:             ${WEIGHT_DIR}"
echo "  Donor dir:           ${DINOV3_OUTPUT_DIR}"
echo "  Boundary tolerance:  ±${BOUNDARY_TOLERANCE}px"
echo ""


# =====================================================================
# A1: No content VIB on F_LL  (edge VIB only)
#     Tests D2: Is content compression on LL essential?
#     Predicted: OGLANet drops most (encoder-level failure); DINOv3 less
#
#   C4 minus: content VIB on LL
#   C4 kept:  Haar, edge VIB on LH/HL, Aug, AdaptiveBeta
#   New flag: DISABLE_CONTENT_VIB=1
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A1_noConVIB" \
                  1    1   1   1   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A1" 0 \
                  1    0   0   0   ""


# =====================================================================
# A2: Uniform-β VIB  (fixed β, no intensity conditioning)
#     Tests D1: Is intensity-adaptive compression necessary?
#     Predicted: overall IoU nearly preserved; disproportionate loss
#                in high-intensity deciles
#
#   C4 minus: adaptive_beta
#   Everything else identical to C4
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A2_fixedBeta" \
                  1    1   1   0   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A2" 0 \
                  0    0   0   0   ""


# =====================================================================
# A3: Symmetric VIB  (same high β on LL, LH, HL; HH passes through)
#     Tests D1+D2: Is subband asymmetry necessary?
#     Predicted: boundary degradation; thin-shadow F1 collapses;
#                small-object IoU drops sharply
#
#   C4 minus: differential VIB (edge VIB with low β)
#   C4 plus:  content-level VIB applied to LH/HL too
#   New flag: SYMMETRIC_VIB=1
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A3_symVIB" \
                  1    1   1   1   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A3" 0 \
                  0    1   0   0   ""


# =====================================================================
# A4: No content augmentation
#     Tests D3: Does augmentation force decoder robustness?
#     Predicted: DINOv3 should degrade most (its failure locus is
#                decoder); OGLANet least affected
#
#   C4 minus: content augmentation
#   Everything else identical to C4
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A4_noAug" \
                  1    1   0   1   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A4" 0 \
                  0    0   0   0   ""


# =====================================================================
# A5: No cross-city mixing (AdaIN style perturbation only)
#     Tests D3 variant: How much of augmentation's benefit comes
#                       specifically from cross-city mixing?
#
#   C4 minus: cross-domain content mixing (aug_p_mix → 0.0)
#   C4 kept:  AdaIN-style random perturbation (aug_p_aug = 0.5)
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A5_noMix" \
                  1    1   1   1   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A5" 0 \
                  0    0   0   0   "0.0"


# =====================================================================
# A6: Augment ALL subbands (MRFP+ analog)
#     Tests D1+D3 interaction: Does augmenting boundary subbands cause
#                              catastrophic collapse?
#     Predicted: Should replicate MRFP+'s OGLANet-Miami collapse —
#                confirms why MRFP+ fails and why SIB doesn't
#     THIS IS THE MOST RHETORICALLY POWERFUL ABLATION
#
#   C4 minus: content-only augmentation constraint
#   C4 plus:  augmentation applied to F_LH, F_HL, F_HH too
#   New flag: AUG_ALL_SUBBANDS=1
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A6_augAll" \
                  1    1   1   1   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A6" 0 \
                  0    0   1   0   ""


# =====================================================================
# A7: No SAG (Skip Attention Gates)
#     N/A for DINOv3 — DINOv3 has no skip connections; SAG is not
#     applicable.  This ablation applies to MAMNet/OGLANet only.
# =====================================================================
# submit_ablation "A7_noSAG" ...   # N/A for DINOv3


# =====================================================================
# A8: No FDA preprocessing
#     N/A for DINOv3 — DINOv3 does not use FDA or contrast channel
#     in C4.  This ablation applies to MAMNet/OGLANet only.
# =====================================================================
# submit_ablation "A8_noFDA" ...   # N/A for DINOv3


# =====================================================================
# A9: No wavelet transform (uniform VIB on full F)
#     Tests D2: Does the Haar subband separation itself matter?
#     Predicted: worse than A3 — without the separation, you can't
#                even attempt targeted compression
#
#   C4 minus: Haar decomposition (use_haar=0 → UniformVIB path)
#   VIB beta_content matches C4 (0.01) for clean comparison
#   Note: When Haar is off, VIB is uniform and edge VIB doesn't exist.
#         AB still works (UniformVIB supports adaptive beta).
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A9_noHaar" \
                  0    1   1   1   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A9" 0 \
                  0    0   0   0   ""


# =====================================================================
# A10: VIB on wrong subband — content VIB on F_HL instead of F_LL
#      Tests D2: Is the design choice of compressing LL not arbitrary?
#      Predicted: clearly worse — compressing the wrong subband
#                 destroys task-relevant boundary information while
#                 leaving domain-carrying content untouched.
#      INVERSE EVIDENCE: "the design choice isn't arbitrary because
#                         the wrong choice is demonstrably worse."
#
#   C4 minus: content VIB on F_LL
#   C4 plus:  content VIB on F_HL (wrong subband)
#             edge VIB on F_LH only (F_LL untouched)
#   New flag: VIB_ON_HL_ONLY=1
#
#     TAG           HAAR VIB AUG AB  FDA  BC       BE       BS     LC   LE   WF    FL    GATE EXP   BYP  DCV  SYM  AUGA VIBHL PMIX
# =====================================================================
submit_ablation "A10_vibHL" \
                  1    1   1   1   0   ${C4_BC} ${C4_BE} ${C4_BS} ${C4_LC} ${C4_LE} ${C4_WF} ${C4_FL} 0 "A10" 0 \
                  0    0   0   1   ""


echo ""
echo "============================================================"
echo "  §5.3 Ablation Study: 24 jobs queued (8 ablations × 3 folds)"
echo "============================================================"
echo ""
echo "  A1  (No content VIB on LL)         — 3 jobs  [D2]"
echo "  A2  (Uniform-β, no intensity cond) — 3 jobs  [D1]"
echo "  A3  (Symmetric VIB all subbands)   — 3 jobs  [D1+D2]"
echo "  A4  (No content augmentation)      — 3 jobs  [D3]"
echo "  A5  (No cross-city mixing)         — 3 jobs  [D3 variant]"
echo "  A6  (Aug all subbands / MRFP+)     — 3 jobs  [D1+D3]"
echo "  A7  N/A for DINOv3 (no SAG)"
echo "  A8  N/A for DINOv3 (no FDA)"
echo "  A9  (No Haar / uniform VIB)        — 3 jobs  [D2]"
echo "  A10 (VIB on wrong subband HL)      — 3 jobs  [D2 inverse]"
echo ""
echo "  Key predictions to verify after runs:"
echo "    A1:  OGLANet drops > DINOv3 drop (validates D2 architecture specificity)"
echo "    A2:  high-intensity decile mIoU drops > low-intensity (validates D1)"
echo "    A3:  thin-shadow F1 collapses but A1 doesn't (subband asymmetry matters)"
echo "    A4:  DINOv3 drops > OGLANet drop (validates D3A decoder focus)"
echo "    A6:  reproduces MRFP+ collapse (why MRFP+ fails and SIB doesn't)"
echo "    A10: worse than C4 (LL was the right subband — not arbitrary)"
echo ""