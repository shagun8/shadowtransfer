#!/bin/bash
# FILENAME: arme_part2_submit.sh
# Arm E Part 2 — submit Experiment A decoder-retraining runs, 3-city grid.
#
#   ./arme_part2_submit.sh --dry-run        print sbatch commands + log paths
#   ./arme_part2_submit.sh --smoke-only     submit ONLY the chicago smoke
#   ./arme_part2_submit.sh                  submit all 3 full cells (resumable)
#
# Per the operator's explicit instruction (matching arm E Part 1's session):
# the full 3-cell grid is gated on the chicago smoke cell completing
# end-to-end (training + dump), not a separate single-full-cell operator
# pause. The smoke job is run and inspected in-session before the grid is
# submitted, never fire-and-forget.
#
# Every sbatch here writes a ledger line (CLAUDE.md). --dry-run skips it.

set -euo pipefail

# ---- BASE_PATH: NCSA Delta (live) ----
BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH="${PROJECT_ROOT}"   # second cluster
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
LOG_PATH="${BASE_PATH}/logs"
DUMP_PATH="${BASE_PATH}/dumps/st"
WORKER="${PYTHON_DIR}/arme_part2_train.sh"
MANIFEST="${LOG_PATH}/arme_part2_manifest.json"

DRY_RUN=0
SMOKE_ONLY=0
for a in "$@"; do
    case "$a" in
        --dry-run)    DRY_RUN=1 ;;
        --smoke-only) SMOKE_ONLY=1 ;;
        *) echo "unknown arg $a"; exit 1 ;;
    esac
done

mkdir -p "${LOG_PATH}"

CITIES=(chicago miami phoenix)

run_id() {
    echo "dinov3frozen-expa-expa-r450-$1-hr-s0"
}

if [ "${SMOKE_ONLY}" -eq 1 ]; then
    RID="$(run_id chicago)-smoke"
    LOG="${LOG_PATH}/${RID}.out"
    CMD=(sbatch --time=00:15:00
         --job-name="${RID}"
         --output="${LOG}"
         --export=ALL,BASE_PATH="${BASE_PATH}",CITY="chicago",SMOKE="1",SMOKE_EPOCHS="3",RUN_ID="${RID}"
         "${WORKER}")
    echo "job shape: 1 smoke job, --time 00:15:00 (override), 3 epochs, own out_dir/dump."
    echo "  log : ${LOG}"
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "DRY RUN — not submitting:"; printf '  %q' "${CMD[@]}"; echo; exit 0
    fi
    JOBID=$("${CMD[@]}" | awk '{print $NF}')
    echo "submitted ${JOBID}"
    python3 "${PYTHON_DIR}/ledger.py" add \
        --event submit --state SUBMITTED --arm E --cell "${RID}" --jobid "${JOBID}" \
        --script "${BASE_PATH}/appendix_h/arme_part2_submit.sh" --log "${LOG}" \
        --note "arm E part 2 smoke; chicago Experiment A, 3 epochs, own out_dir/dump"
    exit 0
fi

# ---- full 3-cell grid ----
MANIFEST_CELLS="["
FIRST=1
N_JOBS=0
for city in "${CITIES[@]}"; do
    RID="$(run_id "$city")"
    [ "$FIRST" -eq 1 ] && FIRST=0 || MANIFEST_CELLS="${MANIFEST_CELLS},"
    MANIFEST_CELLS="${MANIFEST_CELLS}\"${RID}\""

    if [ -f "${DUMP_PATH}/${RID}/.DONE" ]; then
        echo "skip ${RID}: .DONE exists"
        continue
    fi

    LOG="${LOG_PATH}/${RID}.out"
    CMD=(sbatch --time=00:30:00
         --job-name="${RID}"
         --output="${LOG}"
         --export=ALL,BASE_PATH="${BASE_PATH}",CITY="${city}",SMOKE="0",RUN_ID="${RID}"
         "${WORKER}")

    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "DRY RUN — would submit:"; printf '  %q' "${CMD[@]}"; echo
        continue
    fi

    JOBID=$("${CMD[@]}" | awk '{print $NF}')
    echo "submitted ${RID} -> ${JOBID}"
    python3 "${PYTHON_DIR}/ledger.py" add \
        --event submit --state SUBMITTED --arm E --cell "${RID}" --jobid "${JOBID}" \
        --script "${BASE_PATH}/appendix_h/arme_part2_submit.sh" --log "${LOG}" \
        --note "arm E part 2 full grid; Experiment A decoder retrain, frozen backbone"
    N_JOBS=$((N_JOBS + 1))
done
MANIFEST_CELLS="${MANIFEST_CELLS}]"

echo "{\"arm\":\"E\",\"part\":2,\"cells\":${MANIFEST_CELLS}}" > "${MANIFEST}"
echo
echo "job shape: up to 3 jobs, --time 00:30:00 each (~112 images/25 subset, 30"
echo "  epochs, roughly 8-9% of Part 1's per-cell training compute -- generous"
echo "  headroom, not measured from a comparable published wall clock since"
echo "  App G.9 does not report one), ~1.5 GPU-hours worst case for the grid."
echo "  manifest : ${MANIFEST}"
echo "  submitted this call: ${N_JOBS}"
