#!/bin/bash

# Fail loudly if the single per-environment variable is unset.
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
# FILENAME: mamnet_sib_sp_gap_submit.sh
#
# Fans out SP-gap Phase 1 analysis across 3 LOCO holdout folds for MAMNet.
# After all 3 jobs finish, inspect the 3 JSON files in OUTPUT_DIR.
# Population-level Wilcoxon is deferred until OGLANet + DINOv3 cells are done.
#
# Checkpoint naming conventions assumed:
#   Vanilla: mamnet_loco_holdout_{city}_highres_1/checkpoint_best.pth
#   C4-clean: mamnet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_{city}__highres/best_model.pth
#
# If your C4-clean output dirs have a different tag, update C4CLEAN_TAG below.

# ---- Server paths ----
BASE_PATH="${PROJECT_ROOT}/"
BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
MAMNET_OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/sp_gap_results"
LOG_DIR="${BASE_PATH}/data/mamnet/sp_gap_logs"

# C4-clean experiment tag as it appears in the output directory name
# (set in mamnet_sib_submit.sh as the TAG argument to submit_loco)
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
echo "  MAMNet SP-Gap Phase 1 — Job Submission"
if [ "${DRY_RUN}" -eq 1 ]; then
echo "  MODE: DRY RUN (no jobs submitted)"
fi
echo "========================================"
echo "  Vanilla base: ${MAMNET_OUTPUT_DIR}/mamnet_loco_holdout_{city}_${RESOLUTION}_1/"
echo "  C4-clean tag: ${C4CLEAN_TAG}"
echo "  Results dir:  ${OUTPUT_DIR}"
echo "  Log dir:      ${LOG_DIR}"
echo ""

for fold_id in 0 1 2; do
    holdout="${FOLD_NAMES[$fold_id]}"

    VANILLA_CKPT="${MAMNET_OUTPUT_DIR}/mamnet_loco_holdout_${holdout}_${RESOLUTION}_1/checkpoint_best.pth"
    C4CLEAN_CKPT="${MAMNET_OUTPUT_DIR}/mamnet_sib_${C4CLEAN_TAG}__loco_holdout_${holdout}__${RESOLUTION}/best_model.pth"

    name="mamnet_sp_gap_${holdout}_${RESOLUTION}"
    outfile="${LOG_DIR}/${name}.out"

    echo "  fold_id=${fold_id}  holdout=${holdout}"
    echo "    Vanilla : ${VANILLA_CKPT}"
    echo "    C4-clean: ${C4CLEAN_CKPT}"

    # Check that both checkpoints exist before submitting
    if [ ! -f "${VANILLA_CKPT}" ]; then
        echo "    WARNING: Vanilla checkpoint not found — skipping fold ${fold_id}"
        continue
    fi
    if [ ! -f "${C4CLEAN_CKPT}" ]; then
        echo "    WARNING: C4-clean checkpoint not found — skipping fold ${fold_id}"
        continue
    fi

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "    [DRY RUN] would submit: ${name}"
        continue
    fi

    sbatch \
        --output="${outfile}" \
        --job-name="${name}" \
        --export=PROJECT_ROOT=${PROJECT_ROOT},FOLD_ID=${fold_id},RESOLUTION=${RESOLUTION},VANILLA_CHECKPOINT=${VANILLA_CKPT},C4CLEAN_CHECKPOINT=${C4CLEAN_CKPT},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR} \
        mamnet_sib_sp_gap.sh

    echo "    Submitted → ${outfile}"
done

echo ""
echo "========================================"
echo "  Summary: 3 jobs (inference only, ~10min each)"
echo "========================================"
echo ""
echo "  Monitor : squeue -u \$USER"
echo "  Watch   : tail -f ${LOG_DIR}/mamnet_sp_gap_phoenix_${RESOLUTION}.out"
echo "  Results : ls ${OUTPUT_DIR}/sp_gap_mamnet_*_${RESOLUTION}.json"
echo ""
echo "  After all 3 complete, check each JSON:"
echo "    cat ${OUTPUT_DIR}/sp_gap_mamnet_phoenix_${RESOLUTION}.json | python -m json.tool"
echo ""
echo "  Decision rule per cell (bootstrap CI):"
echo "    mean_delta < 0 AND CI excludes 0  =>  C4-clean reduces AURC_shadow"
echo "    mean_delta < 0 AND CI includes 0  =>  directional but inconclusive"
echo "    mean_delta >= 0                   =>  no improvement; Phase 2 needed"
echo ""
echo "  NOTE: Population Wilcoxon (needs n>=9 cells) is deferred until"
echo "  OGLANet and DINOv3 cells are added. For MAMNet alone, report"
echo "  the 3 per-cell CIs and sign count (X/3 cells improve)."
echo ""