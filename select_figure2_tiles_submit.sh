#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: select_figure2_tiles_submit.sh
#
# Submits the Figure 2 candidate-tile selection job.
#
# Iterates all 9 (architecture x holdout-city) cells at high resolution,
# loads input / GT / upper / LOCO / SIB for every test tile, and runs the
# four-stage filter chain:
#
#   Stage 1 — image-quality filters (coverage, blob count, luminance)
#   Stage 2 — failure-mode filters  (delta_fg, delta_bg, R, mIoU floor)
#   Stage 3 — closeness to per-cell median delta_fg
#   Stage 4 — visual-quality re-rank (Laplacian variance, contrast, halo FP)
#
# Outputs (in OUTPUT_DIR):
#   selection_metrics_full.csv     all tiles with all metrics + flags
#   selection_top_candidates.csv   tiles that survive every filter, ranked
#   contact_sheet.pdf              paginated thumbnails for visual review
#   selection_summary.txt          plain-text top-K with filenames

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
OUTPUT_DIR="${BASE_PATH}/data/figure2_selection"
LOG_DIR="${BASE_PATH}/data/eval_sib_logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

name="figure2_tile_selection"
outputfile="${LOG_DIR}/${name}.out"

sbatch --output="${outputfile}" \
       --job-name="${name}" \
       --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_PATH=${BASE_PATH},OUTPUT_DIR=${OUTPUT_DIR} \
       select_figure2_tiles.sh "$@"

echo ""
echo "============================================================"
echo "  Figure 2 Tile Selection Job Submitted"
echo "============================================================"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Log:     ${outputfile}"
echo ""
echo "  After completion, check:"
echo "    ${OUTPUT_DIR}/selection_metrics_full.csv"
echo "    ${OUTPUT_DIR}/selection_top_candidates.csv"
echo "    ${OUTPUT_DIR}/contact_sheet.pdf"
echo "    ${OUTPUT_DIR}/selection_summary.txt"
echo ""
echo "  Reading the contact sheet:"
echo "    8 candidates per architecture (DINOv3, MAMNet, OGLANet)"
echo "    Each row: input | GT | upper | LOCO | SIB"
echo "    Header above each row prints model | city | tile_id"
echo "    Plus delta_fg, delta_bg, R, mIoU(U/L/S), p_S(U/L/S)"
echo ""
echo "  Pick one tile per architecture (ideally one per city)"
echo "  and reply with the three filenames."
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Watch:    tail -f ${outputfile}"
echo ""