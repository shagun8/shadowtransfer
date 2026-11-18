#!/bin/bash
# FILENAME: python/armB_png_score_midres63_submit.sh
# Driver: score the saved published prediction PNG masks for the 63
# MID-RESOLUTION LOCO cells. One job, 16 CPU workers over the cells.
#
# Why one job and not a 63-way fan-out, which is what CLAUDE.md's fan-out rule
# would normally ask for:
#   1. it is the recipe this task reuses -- armB_png_score_submit.sh scored the
#      59 high-resolution cells exactly this way, and the task is to reuse that
#      recipe unchanged except for the resolution and the cell list;
#   2. a cell measured 272.9 s single-threaded (dt-login03, 2026-10-27), so 63
#      cells over 16 workers is ~18 min of wall clock in one job;
#   3. the queue already holds ~146 PENDING jobs against a confirmed
#      bf_max_job_user=100, so 63 extra jobs would push the operator's pending
#      CrossEarth and SAR GPU work out of the backfill window to buy nothing.
# Flagged here rather than left implicit.

set -euo pipefail

# --- BASE_PATH (confirmed on disk 2026-07-27) -------------------------------
BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
SCRATCH="${ST_SCRATCH_ROOT}"
LOG_PATH="${SCRATCH}/logs"
OUT_DIR="${SCRATCH}/runs/midres_png_score"
WORKER="${PYTHON_DIR}/armB_png_score_midres63.sh"
PNG_PY="${PYTHON_DIR}/armB_png_score_midres63.py"
SCORER_PY="${PYTHON_DIR}/armB_png_score.py"
PNG_ENV="${BASE_PATH}/envs/satmae_cuda12"
# PNG_ENV="${BASE_PATH}/conda_envs/satmae_cuda12"           # Gilbreth
# PNG_ENV="${CONDA_ENV_PATH}"   # third cluster

CONTRACT="${PYTHON_DIR}/shadowtransfer/worker_contract.py"
ST_DISK_GATE="${PYTHON_DIR}/shadowtransfer/disk_gate.py"
export ST_DISK_GATE

RUN_ID="armB-png-score-midres63"
LEDGER_ARM="C"
WORKERS=16
TIMEBOX_S=3000
DRY_RUN=0
PARTITION=""
TIMELIMIT=""
ONLY=""
APPENDIX_F=""
OUT_ROOT=""
GATE_ONLY=""
INTENDED=63

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)     DRY_RUN=1 ;;
        --interactive) PARTITION="gpuA100x4-interactive"; TIMELIMIT="0:55:00" ;;
        --workers)     shift; WORKERS="$1" ;;
        --only)        shift; ONLY="$1" ;;
        --appendix-f)  shift; APPENDIX_F="$1" ;;
        --out-root)    shift; OUT_ROOT="$1" ;;
        --gate-only)   GATE_ONLY=1 ;;
        --time)        shift; TIMELIMIT="$1" ;;
        --timebox)     shift; TIMEBOX_S="$1" ;;
        --run-id)      shift; RUN_ID="$1" ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "${LOG_PATH}" "${OUT_DIR}" "${OUT_DIR}/.markers"
LOG_FILE="${LOG_PATH}/${RUN_ID}.out"
PNG_OUT="${OUT_DIR}/results.json"
[ -n "${ONLY}" ] && PNG_OUT="${OUT_DIR}/results_only_$(echo "${ONLY}" | tr -c 'A-Za-z0-9_.-' '_').json"
MANIFEST="${LOG_PATH}/armB_png_score_midres63_manifest.json"

echo "=============================================================================="
echo "MIDRES-63 PNG-MASK SCORING submit   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  base       ${BASE_PATH}"
echo "  worker     ${WORKER}"
echo "  driver     ${PNG_PY}"
echo "  scorer     ${SCORER_PY}  (score_cell, reused unchanged)"
echo "  out        ${PNG_OUT}"
echo "  dumps      ${OUT_ROOT:-${SCRATCH}/dumps/st_r450_published_png}/{run_id}/"
echo "  workers    ${WORKERS}    time-box ${TIMEBOX_S}s"
echo "  partition  ${PARTITION:-gpuA100x4}    --time ${TIMELIMIT:-0:55:00}"
echo "  only       ${ONLY:-<all 63 midres LOCO cells>}"
echo "  appendix F ${APPENDIX_F:-<not supplied -- STEP 3 gate stays OPEN>}"
echo "=============================================================================="

# --- the mechanical driver/worker variable contract (CLAUDE.md) -------------
python3 "${CONTRACT}" --selftest

EXPORTS="ALL,BASE_PATH=${BASE_PATH},PNG_PY=${PNG_PY},PNG_OUT=${PNG_OUT}"
EXPORTS="${EXPORTS},PNG_WORKERS=${WORKERS},PNG_ENV=${PNG_ENV}"
EXPORTS="${EXPORTS},TIMEBOX_S=${TIMEBOX_S},ST_LEDGER_ARM=${LEDGER_ARM}"
[ -n "${ONLY}" ]       && EXPORTS="${EXPORTS},PNG_ONLY=${ONLY}"
[ -n "${APPENDIX_F}" ] && EXPORTS="${EXPORTS},PNG_APPENDIX_F=${APPENDIX_F}"
[ -n "${OUT_ROOT}" ]   && EXPORTS="${EXPORTS},PNG_OUT_ROOT=${OUT_ROOT}"
[ -n "${GATE_ONLY}" ]  && EXPORTS="${EXPORTS},PNG_GATE_ONLY=1"

if ! python3 "${CONTRACT}" --worker "${WORKER}" --py "${PNG_PY}" \
        --py "${SCORER_PY}" \
        --base "${BASE_PATH}" --exports "${EXPORTS}" --label "${RUN_ID}"; then
    echo "FATAL driver/worker variable contract REFUSED ${RUN_ID}" >&2
    exit 3
fi

# --- STEP 1 LOCATE, re-run here so a submit never outruns the inventory -----
echo "[locate] dry-run inventory of the cell set"
"${PNG_ENV}/bin/python" -u "${PNG_PY}" --dry-run \
    ${ONLY:+--only "${ONLY}"} | tail -n 70

# --- the disk gate reads Delta's own quota tool, nothing else ---------------
python3 "${ST_DISK_GATE}"
PROBE_DIR="${OUT_ROOT:-${SCRATCH}/dumps}"
mkdir -p "${PROBE_DIR}"
echo "[probe] 600 MiB conv=fsync at ${PROBE_DIR}"
dd if=/dev/zero of="${PROBE_DIR}/.probe" bs=1M count=600 conv=fsync \
    status=none && rm -f "${PROBE_DIR}/.probe" && echo "[probe] OK"

cat > "${MANIFEST}" << MEOF
{
  "arm": "C",
  "task": "score the saved published prediction PNG masks for the 63 mid-resolution LOCO cells",
  "written_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "run_id": "${RUN_ID}",
  "intended_n_cells": ${INTENDED},
  "cell_set": "{mamnet,oglanet,dinov3} x {vanilla,fda,segdesic,iim,isw,mrfp_plus,fada} x {chicago,miami,phoenix}, mode loco, res mr, rung 450",
  "intended_run_ids": $("${PNG_ENV}/bin/python" -c "
import sys; sys.path.insert(0, '${PYTHON_DIR}')
import json
from armB_png_score_midres63 import build_63
print(json.dumps([c['run_id'] for c in build_63()]))"),
  "driver": "${PNG_PY}",
  "scorer": "${SCORER_PY}",
  "worker": "${WORKER}",
  "out": "${PNG_OUT}",
  "dump_root": "${OUT_ROOT:-${SCRATCH}/dumps/st_r450_published_png}",
  "log": "${LOG_FILE}",
  "workers": ${WORKERS},
  "timebox_s": ${TIMEBOX_S},
  "only": "${ONLY}",
  "appendix_f": "${APPENDIX_F}",
  "headline_metric": "two-class boundary-tolerant mIoU at K=2, per-image mean",
  "gate_metric": "two-class boundary-tolerant mIoU at K=5 vs Appendix F, tol 0.01"
}
MEOF
echo "[manifest] ${MANIFEST}"

SB_EXTRA=()
[ -n "${PARTITION}" ] && SB_EXTRA+=(--partition="${PARTITION}")
[ -n "${TIMELIMIT}" ] && SB_EXTRA+=(--time="${TIMELIMIT}")

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "DRY   sbatch ${SB_EXTRA[*]-} --job-name=${RUN_ID} --output=${LOG_FILE} --export=${EXPORTS} ${WORKER}"
    echo "=============================================================================="
    echo "dry run: nothing submitted, no ledger line written"
    exit 0
fi

JOBID=$(sbatch --parsable "${SB_EXTRA[@]}" \
    --job-name="${RUN_ID}" --output="${LOG_FILE}" \
    --export="${EXPORTS}" "${WORKER}")
echo "SUB   ${JOBID}  ${RUN_ID}  -> ${LOG_FILE}"

python3 "${PYTHON_DIR}/ledger.py" add \
    --arm "${LEDGER_ARM}" --cell "${RUN_ID}" --jobid "${JOBID}" \
    --script "$0" --log "${LOG_FILE}" \
    --note "score the saved published prediction PNG masks for the 63 mid-resolution LOCO cells (data/Test_img_results/loco/{city}/midres/{arch}/{method}) with metrics.py sha 730ad4a5 via armB_png_score.score_cell unchanged; two-class mIoU K=2 primary plus K=0/K=5 counts and foreground-only; STEP 3 gate is K=5 two-class vs Appendix F's printed midres value, tol 0.01, OPEN until --appendix-f is supplied; only=${ONLY:-all 63}; CPU-only on ${PARTITION:-gpuA100x4} with the A100 allocated and idle (no CPU association on this allocation); loads no checkpoint, runs no inference, modifies no existing dump, writes only new dump dirs under dumps/st_r450_published_png"

echo "=============================================================================="
echo "submitted 1   job ${JOBID}   --time ${TIMELIMIT:-0:55:00}"
echo "after it finishes:  cat ${PNG_OUT}"
echo "=============================================================================="
