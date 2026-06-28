#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: final_comparison_submit.sh
#
# Submits the C4_clean vs all-baselines comparison job.
#
# Recomputes tolerant mIoU from scratch using prediction PNGs vs GT masks
# (no shortcuts via comparison_results.json).
#
# Per (architecture, city) cell:
#   - Builds a tolerant-mIoU table for {Upper Bound, LOCO Vanilla, LOCO FDA,
#     LOCO SegDesic, LOCO IIM, LOCO ISW, LOCO MRFP+, LOCO FADA, C4_clean}
#   - Paired bootstrap (B=10000) of C4_clean vs every baseline
#   - Holm–Bonferroni correction (per-cell + global)
#
# Output: ${OUTPUT_DIR}/{report.txt, table.tex, *.json}

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
OUTPUT_DIR="${BASE_PATH}/data/final_comparison_output"
LOG_DIR="${BASE_PATH}/data/eval_sib_logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

name="final_comparison"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_PATH=${BASE_PATH},OUTPUT_DIR=${OUTPUT_DIR} \
       final_comparison.sh "$@"

echo ""
echo "============================================================"
echo "  Final Comparison Job Submitted"
echo "============================================================"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Log:     ${outputfile}"
echo ""
echo "  After completion, check:"
echo "    ${OUTPUT_DIR}/final_comparison_report.txt"
echo "    ${OUTPUT_DIR}/final_comparison_table.tex"
echo "    ${OUTPUT_DIR}/final_comparison.json"
echo "    ${OUTPUT_DIR}/bootstrap_results.json"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Watch:    tail -f ${outputfile}"
echo ""