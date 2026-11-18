#!/bin/bash
# FILENAME: python/armf_crossearth_grid_submit.sh
# Arm F / CrossEarth -- the 12-cell gap row plus the SIB insertion-point gate.
# Holds paths, the per-cluster BASE_PATH, the fan-out, and the ledger calls.
# Never sbatch by hand (CLAUDE.md).
#
#   ./armf_crossearth_grid_submit.sh [--dry-run]
#
# 13 jobs:
#   1 SIB insertion-point gate    00:30:00
#   9 vanilla and fda cells       02:45:00, no dependency, submitted immediately
#   3 sib cells                   02:45:00, --dependency=afterok on the gate
#
# The grid is resumable: a cell whose dumps/st/{run_id}/.DONE exists is skipped,
# so a resubmit fills only the holes.

set -euo pipefail

# ---------------- BASE_PATH: NCSA Delta (live) ----------------
BASE_PATH="${PROJECT_ROOT}"

# ---------------- BASE_PATH: Gilbreth (commented unless in use) ----------------
# BASE_PATH="${PROJECT_ROOT}"   # second cluster

# ---------------- BASE_PATH: Anvil (commented unless in use) ----------------
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
LOG_PATH="${BASE_PATH}/logs"
RUNS_PATH="${BASE_PATH}/runs"
DUMP_PATH="${BASE_PATH}/dumps"
REPO_ROOT="${BASE_PATH}/external/CrossEarth"
ENV_PATH="${BASE_PATH}/envs/crossearth_mm"
CKPT="${REPO_ROOT}/checkpoints/dinov2_converted.pth"

# sha256 and byte length taken on the login node 2026-07-30 and re-verified by
# stage 0 on gpua017. Every job re-verifies both at start and dies on mismatch.
CKPT_SHA256="ccac192898ea50c709db7fa003a46105589e8002f745af076cd5a88d6658300c"
CKPT_BYTES="1216917864"

RES="hr"
SEED="0"
ITERS="10000"
BATCH_SIZE="4"
VAL_INTERVAL="500"
NUM_WORKERS="8"
CELL_TIME="02:45:00"
SMOKE_TIME="00:30:00"
SMOKE_ITERS="500"
SMOKE_CITY="phoenix"

DRY_RUN="no"
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY_RUN="yes"; done

mkdir -p "${LOG_PATH}" "${RUNS_PATH}" "${DUMP_PATH}/st"

CELL_WORKER="${PYTHON_DIR}/armf_crossearth_cell.sh"
SMOKE_WORKER="${PYTHON_DIR}/armf_crossearth_sib_smoke.sh"
SMOKE_OUT="${RUNS_PATH}/armF_crossearth_sib_smoke"
SMOKE_LOG="${LOG_PATH}/armF-crossearth-sib-smoke.out"

# 12 cells, spec order: 1-3 upper/vanilla, 4-6 loco/vanilla,
# 7-9 loco/fda, 10-12 loco/sib.
CELLS=(
  "upper:vanilla:chicago" "upper:vanilla:miami" "upper:vanilla:phoenix"
  "loco:vanilla:chicago"  "loco:vanilla:miami"  "loco:vanilla:phoenix"
  "loco:fda:chicago"      "loco:fda:miami"      "loco:fda:phoenix"
  "loco:sib:chicago"      "loco:sib:miami"      "loco:sib:phoenix"
)

# ---------------- manifest, written before anything is submitted ------------
MANIFEST="${LOG_PATH}/armF_crossearth_grid_manifest.json"
{
  echo "{"
  echo "  \"arm\": \"F_crossearth\","
  echo "  \"architecture\": \"CrossEarth\","
  echo "  \"weakness_ids\": [\"v1UJ-W3\", \"AC1\", \"AC3\", \"v1UJ-W2\"],"
  echo "  \"resolution\": \"highres\", \"seed\": ${SEED}, \"rung\": 450,"
  echo "  \"iters\": ${ITERS}, \"batch_size\": ${BATCH_SIZE},"
  echo "  \"val_interval\": ${VAL_INTERVAL},"
  echo "  \"cell_time\": \"${CELL_TIME}\", \"smoke_time\": \"${SMOKE_TIME}\","
  echo "  \"sib_smoke\": {\"out_dir\": \"${SMOKE_OUT}\", \"log\": \"${SMOKE_LOG}\","
  echo "                  \"iters\": ${SMOKE_ITERS}, \"city\": \"${SMOKE_CITY}\"},"
  echo "  \"cells\": ["
  N=${#CELLS[@]}
  for i in "${!CELLS[@]}"; do
    IFS=":" read -r CT ME CI <<< "${CELLS[$i]}"
    RID="crossearth-${ME}-${CT}-r450-${CI}-${RES}-s${SEED}"
    SEP=","; [ "$i" -eq "$((N-1))" ] && SEP=""
    echo "    {\"n\": $((i+1)), \"run_id\": \"${RID}\", \"cell_type\": \"${CT}\","
    echo "     \"method\": \"${ME}\", \"city\": \"${CI}\","
    echo "     \"dump_dir\": \"${DUMP_PATH}/st/${RID}\","
    echo "     \"results_json\": \"${RUNS_PATH}/armF_crossearth_${CT}_${ME}_${CI}_highres/results.json\","
    echo "     \"log\": \"${LOG_PATH}/${RID}.out\","
    echo "     \"depends_on_sib_smoke\": $([ "${ME}" = "sib" ] && echo true || echo false)}${SEP}"
  done
  echo "  ],"
  echo "  \"written_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
  echo "}"
} > "${MANIFEST}"
echo "manifest: ${MANIFEST}"

COMMON="BASE_PATH=${BASE_PATH},REPO_ROOT=${REPO_ROOT},ENV_PATH=${ENV_PATH}"
COMMON="${COMMON},CKPT=${CKPT},CKPT_SHA256=${CKPT_SHA256},CKPT_BYTES=${CKPT_BYTES}"
COMMON="${COMMON},RES=${RES},SEED=${SEED},NUM_WORKERS=${NUM_WORKERS}"

# ---------------- 1. the SIB insertion-point gate ---------------------------
SMOKE_JOBID=""
if [ -f "${SMOKE_OUT}/.DONE" ]; then
    echo "skip: ${SMOKE_OUT}/.DONE exists; sib cells will submit without a dependency"
else
    EX="ALL,${COMMON},OUT_DIR=${SMOKE_OUT},CITY=${SMOKE_CITY}"
    EX="${EX},ITERS=${SMOKE_ITERS},BATCH_SIZE=${BATCH_SIZE}"
    EX="${EX},REQUESTED_TIME=${SMOKE_TIME},LOG_FILE=${SMOKE_LOG}"
    CMD=(sbatch --job-name="armF-crossearth-sib-smoke" --output="${SMOKE_LOG}"
         --time="${SMOKE_TIME}" --kill-on-invalid-dep=yes
         --export="${EX}" "${SMOKE_WORKER}")
    if [ "${DRY_RUN}" = "yes" ]; then
        echo "DRY RUN: ${CMD[*]}"
        echo "log    : ${SMOKE_LOG}"
        SMOKE_JOBID="DRYRUN"
    else
        OUT="$("${CMD[@]}")"; echo "${OUT}"
        SMOKE_JOBID="$(echo "${OUT}" | awk '{print $NF}')"
        python3 "${PYTHON_DIR}/ledger.py" add \
            --arm F --cell "crossearth-sib-insertion-smoke" \
            --jobid "${SMOKE_JOBID}" --script "${PYTHON_DIR}/armf_crossearth_grid_submit.sh" \
            --state SUBMITTED --log "${SMOKE_LOG}" \
            --note "arm F CrossEarth SIB insertion-point gate, 500 iters on cell 10 (loco/phoenix); cells 10-12 depend afterok; requested ${SMOKE_TIME}; env envs/crossearth_mm; no CPU account so GPU work on gpuA100x4"
        echo "jobid  : ${SMOKE_JOBID}  (sib gate)"
        echo "log    : ${SMOKE_LOG}"
    fi
fi

# ---------------- 2. the 12 cells, one job each ------------------------------
SUBMITTED=0; SKIPPED=0
for i in "${!CELLS[@]}"; do
    IFS=":" read -r CT ME CI <<< "${CELLS[$i]}"
    RID="crossearth-${ME}-${CT}-r450-${CI}-${RES}-s${SEED}"
    DDIR="${DUMP_PATH}/st/${RID}"
    LOG_FILE="${LOG_PATH}/${RID}.out"
    if [ -f "${DDIR}/.DONE" ]; then
        echo "skip: ${DDIR}/.DONE exists"
        SKIPPED=$((SKIPPED+1)); continue
    fi
    EX="ALL,${COMMON},RUN_ID=${RID},DUMP_DIR=${DDIR}"
    EX="${EX},CELL_TYPE=${CT},METHOD=${ME},CITY=${CI}"
    EX="${EX},ITERS=${ITERS},BATCH_SIZE=${BATCH_SIZE},VAL_INTERVAL=${VAL_INTERVAL}"
    EX="${EX},REQUESTED_TIME=${CELL_TIME},LOG_FILE=${LOG_FILE}"

    CMD=(sbatch --job-name="${RID}" --output="${LOG_FILE}"
         --time="${CELL_TIME}" --kill-on-invalid-dep=yes)
    DEPNOTE="no dependency"
    if [ "${ME}" = "sib" ] && [ -n "${SMOKE_JOBID}" ] && [ "${SMOKE_JOBID}" != "DRYRUN" ]; then
        CMD+=(--dependency="afterok:${SMOKE_JOBID}")
        DEPNOTE="afterok:${SMOKE_JOBID}"
    elif [ "${ME}" = "sib" ] && [ "${SMOKE_JOBID}" = "DRYRUN" ]; then
        CMD+=(--dependency="afterok:<SMOKE_JOBID>")
        DEPNOTE="afterok:<SMOKE_JOBID>"
    fi
    CMD+=(--export="${EX}" "${CELL_WORKER}")

    if [ "${DRY_RUN}" = "yes" ]; then
        echo "DRY RUN [$((i+1))/12] ${RID}  (${DEPNOTE})"
        echo "  ${CMD[*]}"
        echo "  log: ${LOG_FILE}"
        SUBMITTED=$((SUBMITTED+1)); continue
    fi
    OUT="$("${CMD[@]}")"; echo "${OUT}"
    JOBID="$(echo "${OUT}" | awk '{print $NF}')"
    python3 "${PYTHON_DIR}/ledger.py" add \
        --arm F --cell "${RID}" \
        --jobid "${JOBID}" --script "${PYTHON_DIR}/armf_crossearth_grid_submit.sh" \
        --state SUBMITTED --log "${LOG_FILE}" \
        --note "arm F CrossEarth ${CT}/${ME}/${CI} hr s${SEED}; ${ITERS} iters bs ${BATCH_SIZE} crop 384 bf16 (deform attn fp32); requested ${CELL_TIME}; ${DEPNOTE}; env envs/crossearth_mm; no CPU account so the whole cell including inference runs on gpuA100x4"
    echo "jobid  : ${JOBID}  [$((i+1))/12] ${RID}  (${DEPNOTE})"
    echo "log    : ${LOG_FILE}"
    SUBMITTED=$((SUBMITTED+1))
done

echo "----------------------------------------------------------------"
echo "cells submitted: ${SUBMITTED}   skipped (.DONE): ${SKIPPED}"
echo "sib gate jobid : ${SMOKE_JOBID:-<none, .DONE existed>}"
echo "per-cell --time: ${CELL_TIME}   gate --time: ${SMOKE_TIME}"
echo "manifest       : ${MANIFEST}"
