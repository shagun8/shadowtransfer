#!/bin/bash
# FILENAME: python/armf_crossearth_submit.sh
# Arm F / CrossEarth -- submit script.  Holds paths, per-cluster BASE_PATH, and
# the ledger call.  Never sbatch by hand (CLAUDE.md).
#
#   ./armf_crossearth_submit.sh stage0 [--dry-run]
#
# Stage 1 is NOT wired into this script.  The released CrossEarth code does not
# contain the Earth-Style Injection, the Geospatial Semantic Extractor, the
# cross-attention Injector, or the ASPP masked-image-modelling branch, and
# Million-AID is not on disk.  Protocol deviation 2 of the arm spec says stop
# and report in exactly that case.  Stage 1 goes in once the operator rules.

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
REPO_ROOT="${BASE_PATH}/external/CrossEarth"
ENV_PATH="${BASE_PATH}/envs/crossearth_mm"
CKPT="${REPO_ROOT}/checkpoints/dinov2_converted.pth"

# sha256 and byte length taken on the login node, 2026-07-30, from
# https://huggingface.co/Cusyoung/CrossEarth/resolve/main/dinov2_converted.pth
CKPT_SHA256="ccac192898ea50c709db7fa003a46105589e8002f745af076cd5a88d6658300c"
CKPT_BYTES="1216917864"

HOLDOUT="phoenix"
RES="hr"
SEED="0"
ITERS="50"
BATCH_SIZE="4"
PRECISION="bf16"

STAGE="${1:-}"
DRY_RUN="no"
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY_RUN="yes"; done

mkdir -p "${LOG_PATH}" "${RUNS_PATH}"

case "${STAGE}" in
stage0)
    JOB_NAME="armF-crossearth-stage0"
    OUT_DIR="${RUNS_PATH}/armF_crossearth_stage0"
    LOG_FILE="${LOG_PATH}/${JOB_NAME}.out"
    WORKER="${PYTHON_DIR}/armf_crossearth_stage0.sh"

    MANIFEST="${LOG_PATH}/armF_crossearth_manifest.json"
    cat > "${MANIFEST}" << MEOF
{
  "arm": "F_crossearth",
  "weakness_ids": ["v1UJ-W3", "AC1"],
  "target_access": "source_only",
  "architecture": "CrossEarth (released subset: Rein-DINOv2 + Mask2Former)",
  "stage0": {"run_id": "armF-crossearth-stage0", "out_dir": "${OUT_DIR}",
             "log": "${LOG_FILE}", "time": "00:40:00"},
  "stage1_authorised_but_held": {
     "run_id": "armF-crossearth-loco-r450-phoenix-hr-s0",
     "reason": "protocol deviation 2: Earth-Style Injection absent from the code release and Million-AID absent from disk; operator ruling required"},
  "cells_not_authorised": [
     "armF-crossearth-loco-r450-chicago-hr-s0",
     "armF-crossearth-loco-r450-miami-hr-s0",
     "armF-crossearth-upper-r450-chicago-hr-s0",
     "armF-crossearth-upper-r450-miami-hr-s0",
     "armF-crossearth-upper-r450-phoenix-hr-s0"],
  "written_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
MEOF
    echo "manifest: ${MANIFEST}"

    if [ -f "${OUT_DIR}/.DONE" ]; then
        echo "skip: ${OUT_DIR}/.DONE exists"
        exit 0
    fi

    EXPORTS="ALL,BASE_PATH=${BASE_PATH},REPO_ROOT=${REPO_ROOT},ENV_PATH=${ENV_PATH}"
    EXPORTS="${EXPORTS},CKPT=${CKPT},CKPT_SHA256=${CKPT_SHA256},CKPT_BYTES=${CKPT_BYTES}"
    EXPORTS="${EXPORTS},OUT_DIR=${OUT_DIR},HOLDOUT=${HOLDOUT},RES=${RES},SEED=${SEED}"
    EXPORTS="${EXPORTS},ITERS=${ITERS},BATCH_SIZE=${BATCH_SIZE},PRECISION=${PRECISION}"
    EXPORTS="${EXPORTS},LOG_FILE=${LOG_FILE}"

    CMD=(sbatch --job-name="${JOB_NAME}" --output="${LOG_FILE}"
         --kill-on-invalid-dep=yes --export="${EXPORTS}" "${WORKER}")

    if [ "${DRY_RUN}" = "yes" ]; then
        echo "DRY RUN: ${CMD[*]}"
        echo "log    : ${LOG_FILE}"
        exit 0
    fi

    OUT="$("${CMD[@]}")"
    echo "${OUT}"
    JOBID="$(echo "${OUT}" | awk '{print $NF}')"
    python3 "${PYTHON_DIR}/ledger.py" add \
        --arm F --cell "crossearth-stage0-env-gate" \
        --jobid "${JOBID}" --script "${PYTHON_DIR}/armf_crossearth_submit.sh" \
        --state SUBMITTED --log "${LOG_FILE}" \
        --note "arm F CrossEarth stage 0 env+load gate; separate env envs/crossearth_mm (torch 2.1.0+cu118, mmcv 2.1.0); no CPU account so GPU work on gpuA100x4"
    echo "jobid  : ${JOBID}"
    echo "log    : ${LOG_FILE}"
    ;;
*)
    echo "usage: $0 stage0 [--dry-run]" >&2
    exit 2
    ;;
esac
