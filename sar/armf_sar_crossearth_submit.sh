#!/bin/bash
# FILENAME: python/armf_sar_crossearth_submit.sh
# Arm F SAR on the three INTERNAL CrossEarth vanilla LOCO checkpoints
# (crossearth-vanilla-loco-r450-{chicago,miami,phoenix}-hr-s0), high resolution.
# Test-time adaptation only -- no training, no checkpoint written, no published
# run touched. One cell per job, one city per cell.
#
#   ./armf_sar_crossearth_submit.sh --dry-run        print sbatch lines, no ledger
#   ./armf_sar_crossearth_submit.sh --smoke          5-image smoke on one city
#   ./armf_sar_crossearth_submit.sh --city miami     one real cell
#   ./armf_sar_crossearth_submit.sh                  the three real cells
#
# Recipe: lr 2.5e-4 (1x sar.SAR_LR_BASE), 10 steps -- the recipe the nine S-EO
# CrossEarth SAR cells ran. BORROWED twice over (crossearth was absent from the
# a4 selection pool, and that pool selected on S-EO, not internal, validation);
# declared in every results.json under recipe_provenance.

set -euo pipefail

# --- BASE_PATH: NCSA Delta (default, live) ----------------------------------
BASE_PATH="${PROJECT_ROOT}"
# --- BASE_PATH: Gilbreth (commented unless in use) --------------------------
# BASE_PATH="${PROJECT_ROOT}"   # second cluster
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
WORKER="${PYTHON_DIR}/armf_sar_crossearth_worker.sh"
DRIVER="${PYTHON_DIR}/armf_sar_crossearth.py"
LEDGER="${PYTHON_DIR}/ledger.py"
CONTRACT="${PYTHON_DIR}/shadowtransfer/worker_contract.py"
ST_DISK_GATE="${PYTHON_DIR}/shadowtransfer/disk_gate.py"
export ST_DISK_GATE
LEDGER_ARM="F"

REPO_ROOT="${BASE_PATH}/external/CrossEarth"
ENV_PATH="${BASE_PATH}/envs/crossearth_mm"
BACKBONE_CKPT="${REPO_ROOT}/checkpoints/dinov2_converted.pth"
BACKBONE_SHA256="ccac192898ea50c709db7fa003a46105589e8002f745af076cd5a88d6658300c"
BACKBONE_BYTES="1216917864"

# /scratch only (CLAUDE.md 2026-09-01: the archive root is archive-only)
SCRATCH_ROOT="${ST_SCRATCH_ROOT}"
PARENT_DUMPS="${SCRATCH_ROOT}/dumps/st"
OUT_ROOT="${SCRATCH_ROOT}/runs"
LOG_PATH="${SCRATCH_ROOT}/logs/armF_sar_crossearth"
MANIFEST="${LOG_PATH}/armF_sar_crossearth_manifest.json"

SAR_LR="0.00025"
SAR_STEPS="10"
MOMENTUM="0.9"
RHO="0.05"
SEED="0"
TIMEOUT_S="900"
SMOKE_IMAGES="5"
REAL_TIME="0:40:00"
SMOKE_TIME="0:25:00"
TIME_OVERRIDE=""
# gpuA100x4-interactive, measured 2026-11-03: MaxTime 01:00:00, DefaultTime
# 00:30:00, QOS gpua100x4intqos with MaxJobsPU=1 AND MaxSubmitJobsPU=1. A cell
# is ~16 min, so the wall clock fits, but only ONE cell can be submitted at a
# time -- which is why --interactive refuses a multi-city run instead of
# silently serialising a fan-out (CLAUDE.md: one cell per job).
INT_PARTITION="gpuA100x4-interactive"
INT_MAX_TIME_S=3600
PARTITION=""
MARGIN_GIB=30
FOOTPRINT_GIB=1          # 3 x (results.json + perstep.parquet); no checkpoint

DRY_RUN=0; SMOKE=0; ONLY_CITY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --smoke)   SMOKE=1 ;;
        --city)    shift; ONLY_CITY="$1" ;;
        --time)    shift; TIME_OVERRIDE="$1" ;;
        --interactive) PARTITION="${INT_PARTITION}" ;;
        --partition)   shift; PARTITION="$1" ;;
        --steps)   shift; SAR_STEPS="$1" ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
mkdir -p "${LOG_PATH}" "${OUT_ROOT}"

echo "=============================================================================="
echo "ARM F SAR crossearth submit   $(date -u +%Y-%m-%dT%H:%M:%SZ)   host $(hostname)"
echo "  worker ${WORKER}"
echo "  dry-run=${DRY_RUN}  smoke=${SMOKE}  city=${ONLY_CITY:-all}  steps=${SAR_STEPS}"
echo "=============================================================================="

echo "[contract] selftest"
python3 "${CONTRACT}" --selftest

# ---- every input on disk before anything is submitted ----------------------
python3 - "${PARENT_DUMPS}" "${OUT_ROOT}" "${REPO_ROOT}" "${ENV_PATH}" \
          "${BACKBONE_CKPT}" "${BACKBONE_SHA256}" "${BACKBONE_BYTES}" <<'PY'
import json, os, sys
parent_dumps, out_root, repo, env, bb, bb_sha, bb_bytes = sys.argv[1:8]
bad = []
for p in (repo, os.path.join(repo, "configs", "_base_", "models",
                             "CrossEarth_dinov2_mask2former.py"),
          os.path.join(env, "bin", "python"), bb):
    if not os.path.exists(p):
        bad.append(p)
if os.path.exists(bb):
    n = os.stat(bb).st_size
    if n != int(bb_bytes):
        bad.append(f"{bb}: {n} bytes, expected {bb_bytes}")
for c in ("chicago", "miami", "phoenix"):
    rid = f"crossearth-vanilla-loco-r450-{c}-hr-s0"
    rj = os.path.join(parent_dumps, rid, "run.json")
    if not os.path.exists(rj):
        bad.append(rj); continue
    d = json.load(open(rj))
    ck = d.get("ckpt_path")
    if not ck or not os.path.exists(ck):
        bad.append(f"{rid}: ckpt_path {ck}")
    if (d.get("arch"), d.get("method"), d.get("mode"), d.get("holdout")) != \
            ("crossearth", "vanilla", "loco", c):
        bad.append(f"{rid}: run.json describes the wrong cell")
    tgt = os.path.join(out_root, f"armF_sar_crossearth_{c}_highres")
    _archive = os.environ.get("ST_ARCHIVE_ROOT", "")
    assert not (_archive and os.path.realpath(tgt).startswith(_archive)), \
        f"REFUSED: ST_ARCHIVE_ROOT is archive-only, target {tgt}"
    pub = os.path.join(os.environ["PROJECT_ROOT"], "runs", f"armF_sar_crossearth_{c}_highres")
    assert os.path.realpath(tgt) != os.path.realpath(pub), \
        f"REFUSED: target {tgt} resolves onto {pub}"
    print(f"[inputs] {rid}  ckpt {ck}")
    print(f"[inputs] {rid}  stored test two-class K=2 {d['test']['miou_tol2']:.6f}"
          f"  fg-only {d['test']['iou_fg_tol2']:.6f}")
assert not bad, "missing or wrong inputs: " + "; ".join(bad)
print("[inputs] 3 parent run.json + 3 checkpoints, CrossEarth release, "
      "crossearth_mm env and the backbone are present; no write target is on "
      "/projects and none collides with a published run")
PY

# ---- disk: scratch == quota alloc, asserted, then the quota row ----
python3 - <<'PY'
import os
a, b = os.stat(os.environ["ST_SCRATCH_ROOT"]), os.stat(os.environ["ST_QUOTA_ALLOC"])
assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), "not the same volume"
print(f"[disk] scratch == quota alloc (dev {a.st_dev}, inode {a.st_ino})")
PY
HEAD_GIB=$(python3 "${ST_DISK_GATE}" --alloc "${ST_QUOTA_ALLOC}" --json | python3 -c "import json,sys; d=json.load(sys.stdin); print(round(d['free']/2**30,1))")
echo "[disk] headroom to soft ${HEAD_GIB} GiB; projected footprint ${FOOTPRINT_GIB} GiB; margin floor ${MARGIN_GIB} GiB"
python3 -c "import sys; h=float('${HEAD_GIB}'); sys.exit(0 if h-${FOOTPRINT_GIB} >= ${MARGIN_GIB} else 1)" \
    || { echo "STOP: margin under ${MARGIN_GIB} GiB" >&2; exit 1; }
dd if=/dev/zero of="${OUT_ROOT}/.probe" bs=1M count=600 conv=fsync status=none
rm -f "${OUT_ROOT}/.probe"
echo "[probe] 600 MiB conv=fsync OK at $(readlink -f "${OUT_ROOT}")"

CITIES="chicago miami phoenix"
[ -n "${ONLY_CITY}" ] && CITIES="${ONLY_CITY}"
[ "${SMOKE}" -eq 1 ] && CITIES="${ONLY_CITY:-phoenix}"

# ---- the interactive partition is a one-cell-at-a-time route ---------------
PART_FLAG=""
if [ -n "${PARTITION}" ]; then
    PART_FLAG="--partition=${PARTITION}"
    echo "[partition] ${PARTITION} (worker default gpuA100x4 overridden on the sbatch line)"
    if [ "${PARTITION}" = "${INT_PARTITION}" ]; then
        N_CITY=$(echo ${CITIES} | wc -w)
        if [ "${N_CITY}" -gt 1 ]; then
            echo "STOP: ${INT_PARTITION} has MaxSubmitJobsPU=1, so ${N_CITY} cells cannot be queued together. Re-run with --city <one city>, or drop --interactive to fan out on gpuA100x4." >&2
            exit 1
        fi
        REQ="${TIME_OVERRIDE:-$([ "${SMOKE}" -eq 1 ] && echo "${SMOKE_TIME}" || echo "${REAL_TIME}")}"
        python3 - "${REQ}" "${INT_MAX_TIME_S}" <<'PY2'
import sys
req, cap = sys.argv[1], int(sys.argv[2])
h, m, sec = ([0] * (3 - len(req.split(":"))) + [int(x) for x in req.split(":")])[-3:]
total = h * 3600 + m * 60 + sec
if total > cap:
    raise SystemExit(f"STOP: requested --time {req} ({total}s) exceeds the "
                     f"interactive partition MaxTime of {cap}s")
print(f"[partition] requested --time {req} ({total}s) is inside the "
      f"interactive MaxTime of {cap}s")
PY2
        RUN_INT=$(squeue -u "$USER" -h -p "${INT_PARTITION}" -o "%i" | wc -l)
        if [ "${RUN_INT}" -gt 0 ]; then
            echo "STOP: ${RUN_INT} job(s) of yours are already on ${INT_PARTITION} and MaxSubmitJobsPU=1. Wait for them, or drop --interactive." >&2
            exit 1
        fi
    fi
fi

# ---- manifest BEFORE submitting --------------------------------------------
python3 - "${MANIFEST}" "${OUT_ROOT}" "${SMOKE}" "${SAR_LR}" "${SAR_STEPS}" ${CITIES} <<'PY'
import json, os, sys, time
man, out_root, smoke, lr, steps = sys.argv[1:6]
cities = sys.argv[6:]
role = "smoke" if smoke == "1" else "real"
m = {"arm": "F", "stage": "sar_crossearth_internal",
     "what": "arm F SAR (episodic test-time adaptation) on the three internal "
             "CrossEarth vanilla LOCO r450 hr s0 checkpoints; identical "
             "harness to armf_sar_grid.py, architecture and image source are "
             "the only changed inputs",
     "recipe": {"lr": float(lr), "steps": int(steps), "lr_multiplier": 1.0,
                "source": "the S-EO CrossEarth SAR recipe, BORROWED"},
     "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "role": role, "n_cells": len(cities),
     "cells": [{"city": c,
                "parent_run_id": f"crossearth-vanilla-loco-r450-{c}-hr-s0",
                "out_dir": os.path.join(
                    out_root, f"armF_sar_crossearth_{c}_highres"
                    + ("-smoke" if role == "smoke" else ""))}
               for c in cities]}
tmp = man + ".new"
json.dump(m, open(tmp, "w"), indent=1)
os.replace(tmp, man)
print(f"[manifest] {man} ({len(cities)} cells, role {role})")
PY

N_SUB=0
for CITY in ${CITIES}; do
    if [ "${SMOKE}" -eq 1 ]; then
        ROLE="smoke"; TLIM="${TIME_OVERRIDE:-${SMOKE_TIME}}"; CTAG="-smoke"
        NOTE="arm F SAR crossearth STAGE-1 SMOKE: ${SMOKE_IMAGES} images, exercises the harness end to end including the BRANCH-4 snapshot assertion; requested ${TLIM}"
    else
        ROLE="real"; TLIM="${TIME_OVERRIDE:-${REAL_TIME}}"; CTAG=""
        NOTE="arm F SAR on the internal CrossEarth vanilla LOCO r450 hr s0 checkpoint, ${CITY}: unadapted pass + ${SAR_STEPS}-step SAR pass over the canonical 150 test images, lr ${SAR_LR} (BORROWED from the S-EO CrossEarth cells), D-B3-3 log-score adapter; test-time only, trains nothing, writes no checkpoint; requested ${TLIM} on ${PARTITION:-gpuA100x4}"
    fi
    NAME="armF-sar-ce-${ROLE}-${CITY}-hr"
    CDIR="${OUT_ROOT}/armF_sar_crossearth_${CITY}_highres${CTAG}"
    if [ -f "${CDIR}/.DONE" ]; then echo "SKIP  ${NAME} (.DONE)"; continue; fi
    LOG="${LOG_PATH}/${NAME}.out"
    EXPORTS="ALL,BASE_PATH=${BASE_PATH},CITY=${CITY},ROLE=${ROLE},OUT_ROOT=${OUT_ROOT},TAG=${CTAG},PARENT_DUMPS=${PARENT_DUMPS},REPO_ROOT=${REPO_ROOT},ENV_PATH=${ENV_PATH},BACKBONE_CKPT=${BACKBONE_CKPT},BACKBONE_SHA256=${BACKBONE_SHA256},BACKBONE_BYTES=${BACKBONE_BYTES},SAR_LR=${SAR_LR},SAR_STEPS=${SAR_STEPS},MOMENTUM=${MOMENTUM},RHO=${RHO},SEED=${SEED},TIMEOUT_S=${TIMEOUT_S},SMOKE_IMAGES=${SMOKE_IMAGES},LEDGER_ARM=${LEDGER_ARM},ARMF_TIME_LIMIT=${TLIM},LOG_FILE=${LOG}"
    if ! python3 "${CONTRACT}" --worker "${WORKER}" --py "${DRIVER}" \
            --base "${BASE_PATH}" --exports "${EXPORTS}" --label "${NAME}"; then
        echo "REFUSED ${NAME}: driver/worker variable contract violated" >&2; exit 5
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "DRY   sbatch ${PART_FLAG} --time=${TLIM} --job-name=${NAME} --output=${LOG} --export=${EXPORTS} ${WORKER}"
        N_SUB=$((N_SUB+1)); continue
    fi
    J=$(sbatch --parsable ${PART_FLAG} --time="${TLIM}" --job-name="${NAME}" \
        --output="${LOG}" --export="${EXPORTS}" "${WORKER}")
    echo "SUB   ${J}  ${NAME}  time=${TLIM}  -> ${LOG}"
    python3 "${LEDGER}" add --event submit --state SUBMITTED --arm "${LEDGER_ARM}" \
        --cell "crossearth-vanilla-loco-r450-${CITY}-hr-s0 [sar-${ROLE}]" \
        --jobid "${J}" --script "$(readlink -f "$0")" --log "${LOG}" --note "${NOTE}"
    N_SUB=$((N_SUB+1))
done

echo "=============================================================================="
echo "jobs ${N_SUB} (dry-run=${DRY_RUN})   manifest ${MANIFEST}   logs ${LOG_PATH}"
echo "=============================================================================="
