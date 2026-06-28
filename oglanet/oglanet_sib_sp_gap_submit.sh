#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: oglanet_sib_sp_gap_submit.sh
#
# Fans out SP-gap Phase 1 analysis across 3 LOCO holdout folds for OGLANet
# C4-clean. Vanilla and 6 adaptation methods are NOT run here — their
# probability maps are already saved in Test_img_probs/ and the aggregator
# loads them directly.
#
# Checkpoint path convention (from oglanet_sib_submit.sh):
#   oglanet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_{city}__highres/best_model.pth

# ---- Server paths ----
BASE_PATH="${PROJECT_ROOT}/"
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OGLANET_OUTPUT_DIR="${BASE_PATH}/data/oglanet/outputs"
OUTPUT_DIR="${BASE_PATH}/data/oglanet/sp_gap_results"
LOG_DIR="${BASE_PATH}/data/oglanet/sp_gap_logs"

# C4-clean experiment tag (matches the submit_loco TAG in oglanet_sib_submit.sh)
C4CLEAN_TAG="C4clean_haar_vib_sag_fda_ctr"

FOLD_NAMES=("phoenix" "miami" "chicago")
RESOLUTION="highres"

# ---- Parse flags ----
DRY_RUN=0
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=1 ;;
    esac
done

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "========================================"
echo "  OGLANet SP-Gap Phase 1 — Submission"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "========================================"
echo "  C4-clean tag: ${C4CLEAN_TAG}"
echo "  Results dir:  ${OUTPUT_DIR}"
echo "  Log dir:      ${LOG_DIR}"
echo ""

for fold_id in 0 1 2; do
    holdout="${FOLD_NAMES[$fold_id]}"

    C4CLEAN_CKPT="${OGLANET_OUTPUT_DIR}/oglanet_sib_${C4CLEAN_TAG}__loco_holdout_${holdout}__${RESOLUTION}/checkpoints/best_model.pth"

    name="oglanet_sp_gap_${holdout}_${RESOLUTION}"
    outfile="${LOG_DIR}/${name}.out"

    echo "  fold_id=${fold_id}  holdout=${holdout}"
    echo "    C4-clean: ${C4CLEAN_CKPT}"

    if [ ! -f "${C4CLEAN_CKPT}" ]; then
        # Try the alternative path (without 'checkpoints/' subdir, in case naming differs)
        C4CLEAN_CKPT_ALT="${OGLANET_OUTPUT_DIR}/oglanet_sib_${C4CLEAN_TAG}__loco_holdout_${holdout}__${RESOLUTION}/best_model.pth"
        if [ -f "${C4CLEAN_CKPT_ALT}" ]; then
            C4CLEAN_CKPT="${C4CLEAN_CKPT_ALT}"
            echo "    Using alt path: ${C4CLEAN_CKPT}"
        else
            echo "    WARNING: C4-clean checkpoint not found at either path — skipping"
            echo "      Tried: ${C4CLEAN_CKPT}"
            echo "      Tried: ${C4CLEAN_CKPT_ALT}"
            continue
        fi
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "    [DRY RUN] would submit: ${name}"
        continue
    fi

    sbatch \
        --output="${outfile}" \
        --job-name="${name}" \
        --export=PROJECT_ROOT=${PROJECT_ROOT},FOLD_ID=${fold_id},RESOLUTION=${RESOLUTION},C4CLEAN_CHECKPOINT=${C4CLEAN_CKPT},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR} \
        oglanet_sib_sp_gap.sh

    echo "    Submitted → ${outfile}"
done

echo ""
echo "========================================"
echo "  Summary: 3 jobs (C4-clean inference only)"
echo "========================================"
echo ""
echo "  Monitor : squeue -u \$USER"
echo "  Watch   : tail -f ${LOG_DIR}/oglanet_sp_gap_phoenix_${RESOLUTION}.out"
echo "  Results : ls ${OUTPUT_DIR}/sp_gap_oglanet_c4clean_*_${RESOLUTION}.json"
echo ""
echo "  After all 3 (× 3 archs total) complete, run aggregate_sp_gap.py."
echo ""