#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: sib_ablation_analysis_v2_submit.sh
#
# Submits the redone SIB component ablation analysis.
#
# Recomputes EVERY tolerant mIoU value from prediction PNGs vs GT masks
# (no comparison_results.json shortcuts).
#
# Includes:
#   C4         — original full SIB (Haar + VIB + Aug + AB [+ SAG + aFDA + Ctr])
#   C4_clean   — proposed simplified model (Haar + VIB only)
#   A1–A10     — original component ablations (same as before)
#
# For each (architecture × ablation):
#   - Per-cell tolerant mIoU averaged across 3 LOCO holdout cities
#   - Paired bootstrap (B=10000) of ablation vs C4, pooled over cities
#   - Holm–Bonferroni global correction
#   - Recovery ratio R = (method − Vanilla) / (Upper − Vanilla)

# =====================================================================
# Server paths — uncomment the block for your target server
# =====================================================================

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}/"

# =====================================================================
# Derived paths
# =====================================================================
OUTPUT_DIR="${BASE_PATH}/data/ablation_analysis_v2"
LOG_DIR="${BASE_PATH}/data/eval_sib_logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

name="sib_ablation_analysis_v2"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_PATH=${BASE_PATH},OUTPUT_DIR=${OUTPUT_DIR} \
       sib_ablation_analysis_v2.sh "$@"

echo ""
echo "============================================================"
echo "  SIB Ablation Analysis V2 Job Submitted"
echo "============================================================"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Log:     ${outputfile}"
echo ""
echo "  After completion, check:"
echo "    ${OUTPUT_DIR}/ablation_v2_main_table.txt"
echo "    ${OUTPUT_DIR}/ablation_v2_deltas.txt"
echo "    ${OUTPUT_DIR}/ablation_v2_recovery.txt"
echo "    ${OUTPUT_DIR}/ablation_v2_summary.txt"
echo "    ${OUTPUT_DIR}/ablation_v2_predictions.txt"
echo "    ${OUTPUT_DIR}/ablation_v2_table4.tex"
echo "    ${OUTPUT_DIR}/ablation_v2_report.json"
echo ""
echo "  Predictions to verify:"
echo "    P1: A1 OGLANet drop > DINOv3 drop  → D2 architecture specificity"
echo "    P2: A4 DINOv3 drop > OGLANet drop  → D3 decoder focus"
echo "    P3: A10 worse than C4 everywhere   → wrong subband, inverse evidence"
echo "    P4: A3 hurts boundaries            → subband asymmetry matters"
echo "    P5: A6 OGLANet-Miami collapses     → why MRFP+ fails, SIB doesn't"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Watch:    tail -f ${outputfile}"
echo ""