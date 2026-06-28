#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: feat_diag_submit.sh
#
# Queues task-relevant feature distribution diagnostics on SLURM.
# Uncomment the server block you need below.

# ---- Server paths (uncomment the one you need) ----

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"

# ---- Common paths ----
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test"
OUTPUT_DIR="${BASE_PATH}/data/feature_diagnostics_output"
LOG_DIR="${BASE_PATH}/data/feature_diagnostics"

# Create output dirs
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

# ============================================================
# Full run: all cities, both resolutions, train vs test
# ============================================================
echo "Queueing feature distribution diagnostics..."

name="feat_diag__full"
outputfile="${LOG_DIR}/${name}.out"

echo "  - Full diagnostics: train vs test, all cities × resolutions"
echo "    Log: ${outputfile}"

sbatch --output=${outputfile} \
       --job-name=${name} \
       --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR},RESOLUTIONS="highres midres",CITIES="chicago miami phoenix",SPLIT_A=train,SPLIT_B=test \
       feature_diagnostics.sh

# ============================================================
# Optional: also compare train vs val (uncomment to run)
# ============================================================
# echo ""
# echo "Queueing train-vs-val comparison..."
#
# name="feat_diag__train_vs_val"
# outputfile="${LOG_DIR}/${name}.out"
#
# sbatch --output=${outputfile} \
#        --job-name=${name} \
#        --export=PROJECT_ROOT=${PROJECT_ROOT},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR}_train_vs_val,RESOLUTIONS="highres midres",CITIES="chicago miami phoenix",SPLIT_A=train,SPLIT_B=val \
#        feat_diag.sh

echo ""
echo "Job queued! Check logs at: ${LOG_DIR}/"
echo "Results will appear in: ${OUTPUT_DIR}/"
echo ""
echo "Key outputs:"
echo "  feature_divergence_detailed.txt  — full table with both metrics"
echo "  city_comparison_{res}.png        — bar chart comparing cities (the money plot)"
echo "  heatmap_ks_d.png                 — heatmap of all KS distances"
echo "  distributions_{city}_{res}.png   — per-city histograms"
echo "  divergence_summary.txt           — bottom-line interpretation"