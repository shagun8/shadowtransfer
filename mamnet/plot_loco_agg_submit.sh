#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"

BASE_DATA_ROOT="${PROJECT_ROOT}/data/Final_data_test/"
OUTPUT_DIR="${PROJECT_ROOT}/data/mamnet/outputs"

# ============================================================
# PART 1: Train individual file
# ============================================================
echo "Queueing individual file..."

name="LOCO_agg_plots_file"
outputfile="${PROJECT_ROOT}/data/mamnet/${name}.out"
RESULTS_DIR="${PROJECT_ROOT}/data/mamnet/outputs"
OUTPUT_DIR="${PROJECT_ROOT}/data/mamnet/loco_aggregate_results"

echo "${name}"
sbatch --output=${outputfile} \
	   --job-name=${name} \
	   --export=PROJECT_ROOT=${PROJECT_ROOT},RESULTS_DIR=${RESULTS_DIR},OUTPUT_DIR=${OUTPUT_DIR} \
	   plot_loco_agg.sh

echo "All jobs queued!"