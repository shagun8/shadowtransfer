#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_isw_submit.sh
#
# Queues OGLANet + ISW LOCO training jobs on SLURM.
#
# Prerequisites:
#   ISW masks must already be precomputed — run
#   compute_isw_masks_oglanet_submit.sh first and wait for those jobs
#   to finish before running this script.
#
# Covers: LOCO folds 0, 1, 2 (holdout: phoenix, miami, chicago).

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- Anvil ---
# BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH2="${PROJECT_ROOT}"

# --- NCSA Delta ---
BASE_PATH="${PROJECT_ROOT}"
BASE_PATH2="${PROJECT_ROOT}"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
ISW_MASK_BASE_DIR="${BASE_PATH}/data/oglanet/isw_masks"

BOUNDARY_TOLERANCE=2
ISW_LAMBDA=0.6

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# LOCO models with ISW
# ============================================================
echo "Queueing OGLANet + ISW LOCO training jobs..."
echo ""

for fold_id in 0 1 2
do
    for res in midres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        mask_dir="${ISW_MASK_BASE_DIR}/loco_holdout_${holdout_city}_${res}"
        name="oglanet_isw__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/oglanet/${name}.out"

        # Verify masks exist before queuing
        if [ ! -d "${mask_dir}" ]; then
            echo "  WARNING: Mask dir not found: ${mask_dir}"
            echo "           Run compute_isw_masks_oglanet_submit.sh first!"
            echo "           Skipping fold ${fold_id} (${holdout_city})."
            echo ""
            continue
        fi

        # Quick sanity: at least one .npy file present
        if ! ls "${mask_dir}"/*.npy 1>/dev/null 2>&1; then
            echo "  WARNING: No .npy masks found in ${mask_dir}"
            echo "           Precomputation may not have completed."
            echo "           Skipping fold ${fold_id} (${holdout_city})."
            echo ""
            continue
        fi

        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        echo "    mask dir:  ${mask_dir}"
        echo "    output:    ${outputfile}"

        sbatch --output="${outputfile}" \
               --job-name="${name}" \
               --export=PROJECT_ROOT=${PROJECT_ROOT},MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},ISW_MASK_DIR=${mask_dir},ISW_LAMBDA=${ISW_LAMBDA},USE_CONTRAST=1,EVAL_TOLERANT=1,BOUNDARY_TOLERANCE=${BOUNDARY_TOLERANCE},EARLY_STOPPING_PATIENCE=10 \
               oglanet_isw.sh
        echo ""
    done
done

echo "All OGLANet + ISW LOCO training jobs queued!"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Output logs:  ${BASE_PATH}/data/oglanet/oglanet_isw__loco_holdout_<city>__<res>.out"