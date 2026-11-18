#!/bin/bash
# FILENAME: python/armf_sar_n450_submit.sh
# Arm F SAR re-run on the three n450 MAMNet standard (vanilla) LOCO models.
# Test-time adaptation only -- no training, no checkpoint written, no published
# SAR run touched. One cell per job, one city per cell.
#
#   ./armf_sar_n450_submit.sh --dry-run            print sbatch lines, no ledger
#   ./armf_sar_n450_submit.sh --smoke              5-image smoke on one city
#   ./armf_sar_n450_submit.sh                      the three real cells
#   ./armf_sar_n450_submit.sh --city miami         one city only

set -euo pipefail

# --- BASE_PATH: NCSA Delta (default, live) ----------------------------------
BASE_PATH="${PROJECT_ROOT}"
# --- BASE_PATH: Gilbreth (commented unless in use) --------------------------
# BASE_PATH="${PROJECT_ROOT}"   # second cluster
# --- BASE_PATH: Anvil (commented unless in use) -----------------------------
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
WORKER="${PYTHON_DIR}/armf_sar_n450_worker.sh"
DRIVER="${PYTHON_DIR}/armf_sar_n450.py"
LEDGER="${PYTHON_DIR}/ledger.py"
CONTRACT="${PYTHON_DIR}/shadowtransfer/worker_contract.py"
ST_DISK_GATE="${PYTHON_DIR}/shadowtransfer/disk_gate.py"
export ST_DISK_GATE
LEDGER_ARM="F"
# /scratch only (CLAUDE.md 2026-09-01: /projects is archive-only)
SCRATCH_ROOT="${ST_SCRATCH_ROOT:-${PROJECT_ROOT}}"
OUT_ROOT="${SCRATCH_ROOT}/runs"
LOG_PATH="${SCRATCH_ROOT}/logs/armF_sar_n450"
MANIFEST="${LOG_PATH}/armF_sar_n450_manifest.json"
TAG="-n450"
SMOKE_IMAGES="5"
REAL_TIME="0:45:00"
TIME_OVERRIDE=""   # --time applies to whichever role is submitted (2026-10-31)
SMOKE_TIME="0:20:00"
MARGIN_GIB=30
FOOTPRINT_GIB=1          # 3 x (results.json + perstep.parquet); no checkpoint

DRY_RUN=0; SMOKE=0; ONLY_CITY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --smoke)   SMOKE=1 ;;
        --city)    shift; ONLY_CITY="$1" ;;
        --time)    shift; TIME_OVERRIDE="$1" ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
mkdir -p "${LOG_PATH}" "${OUT_ROOT}"

echo "=============================================================================="
echo "ARM F SAR n450 submit   $(date -u +%Y-%m-%dT%H:%M:%SZ)   host $(hostname)"
echo "  worker ${WORKER}   dry-run=${DRY_RUN}  smoke=${SMOKE}  city=${ONLY_CITY:-all}"
echo "=============================================================================="

echo "[contract] selftest"
python3 "${CONTRACT}" --selftest

# ---- every input must be on disk before anything is submitted ---------------
python3 - "${OUT_ROOT}" "${TAG}" <<'PY'
import json, os, sys
out_root, tag = sys.argv[1], sys.argv[2]
N450 = os.path.join(os.environ["ST_SCRATCH_ROOT"], "data/mamnet/outputs/n450")
DUMP = os.path.join(os.environ["ST_SCRATCH_ROOT"], "dumps/st_r450_published_recompute")
PUB = os.path.join(os.environ["PROJECT_ROOT"], "runs")
sel = os.path.join(PUB, "armF_sar_lr_pilot", "selected_config.json")
assert os.path.exists(sel), f"no stage-1 selected_config at {sel}"
d = json.load(open(sel))
print(f"[inputs] selected_config lr={d['lr']} x{d['lr_multiplier']} "
      f"steps={d['steps']} verdict={d.get('pilot_verdict')}")
bad = []
for c in ("chicago", "miami", "phoenix"):
    rid = f"mamnet-vanilla-loco-r450-{c}-hr-s42-n450"
    leg = os.path.join(N450, rid, f"mamnet_loco_holdout_{c}_highres_1")
    for p in (os.path.join(leg, "checkpoint_best.pth"),
              os.path.join(leg, "args.json"),
              os.path.join(DUMP, rid, "run.json")):
        if not os.path.exists(p):
            bad.append(p)
    # the published counterpart must exist and must NOT be the write target
    pub = os.path.join(PUB, f"armF_sar_mamnet_{c}_highres")
    assert os.path.exists(os.path.join(pub, "results.json")), f"no {pub}"
    tgt = os.path.join(out_root, f"armF_sar_mamnet_{c}_highres{tag}")
    assert os.path.realpath(tgt) != os.path.realpath(pub), \
        f"REFUSED: target {tgt} resolves onto the published run {pub}"
    _archive = os.environ.get("ST_ARCHIVE_ROOT", "")
    assert not (_archive and os.path.realpath(tgt).startswith(_archive)), \
        f"REFUSED: ST_ARCHIVE_ROOT is archive-only, target {tgt}"
assert not bad, "missing inputs: " + "; ".join(bad)
print("[inputs] 3 n450 checkpoints, 3 args.json, 3 run.json present; "
      "3 published SAR runs present and distinct from every write target")
PY

# ---- disk: scratch root and quota allocation are one volume, then quota row --
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
[ "${SMOKE}" -eq 1 ] && CITIES="${ONLY_CITY:-chicago}"

# ---- manifest BEFORE submitting ---------------------------------------------
python3 - "${MANIFEST}" "${OUT_ROOT}" "${TAG}" "${SMOKE}" ${CITIES} <<'PY'
import json, os, sys, time
man, out_root, tag, smoke = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cities = sys.argv[5:]
role = "smoke" if smoke == "1" else "real"
m = {"arm": "F", "stage": "sar_n450",
     "what": "arm F SAR re-run on the three n450 MAMNet standard (vanilla) "
             "LOCO checkpoints; test-time adaptation only, identical harness, "
             "checkpoint is the only changed input",
     "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "role": role, "n_cells": len(cities),
     "cells": [{"city": c,
                "parent_run_id": f"mamnet-vanilla-loco-r450-{c}-hr-s42-n450",
                "out_dir": os.path.join(out_root, f"armF_sar_mamnet_{c}_highres{tag}"),
                "published_counterpart":
                    os.path.join(os.environ["PROJECT_ROOT"], "runs",
                                 f"armF_sar_mamnet_{c}_highres")}
               for c in cities]}
tmp = man + ".new"
json.dump(m, open(tmp, "w"), indent=1)
os.replace(tmp, man)
print(f"[manifest] {man} ({len(cities)} cells, role {role})")
PY

N_SUB=0
for CITY in ${CITIES}; do
    if [ "${SMOKE}" -eq 1 ]; then
        ROLE="smoke"; TLIM="${TIME_OVERRIDE:-${SMOKE_TIME}}"; CTAG="${TAG}-smoke"
        NOTE="arm F SAR n450 STAGE-1 SMOKE: ${SMOKE_IMAGES} images, exercises the harness end to end; requested ${TLIM}"
    else
        ROLE="real"; TLIM="${TIME_OVERRIDE:-${REAL_TIME}}"; CTAG="${TAG}"
        NOTE="arm F SAR n450 cell: unadapted + SAR pass + weight-drift pass over 150 test images on the n450 MAMNet vanilla checkpoint; test-time only, no training; requested ${TLIM}"
    fi
    NAME="armF-sar-n450-${ROLE}-mamnet-${CITY}-hr"
    CDIR="${OUT_ROOT}/armF_sar_mamnet_${CITY}_highres${CTAG}"
    if [ -f "${CDIR}/.DONE" ]; then echo "SKIP  ${NAME} (.DONE)"; continue; fi
    LOG="${LOG_PATH}/${NAME}.out"
    EXPORTS="ALL,BASE_PATH=${BASE_PATH},CITY=${CITY},ROLE=${ROLE},OUT_ROOT=${OUT_ROOT},TAG=${CTAG},SMOKE_IMAGES=${SMOKE_IMAGES},LEDGER_ARM=${LEDGER_ARM},ARMF_TIME_LIMIT=${TLIM},LOG_FILE=${LOG}"
    if ! python3 "${CONTRACT}" --worker "${WORKER}" --py "${DRIVER}" \
            --base "${BASE_PATH}" --exports "${EXPORTS}" --label "${NAME}"; then
        echo "REFUSED ${NAME}: driver/worker variable contract violated" >&2; exit 5
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "DRY   sbatch --time=${TLIM} --job-name=${NAME} --output=${LOG} --export=${EXPORTS} ${WORKER}"
        N_SUB=$((N_SUB+1)); continue
    fi
    J=$(sbatch --parsable --time="${TLIM}" --job-name="${NAME}" \
        --output="${LOG}" --export="${EXPORTS}" "${WORKER}")
    echo "SUB   ${J}  ${NAME}  time=${TLIM}  -> ${LOG}"
    python3 "${LEDGER}" add --event submit --state SUBMITTED --arm "${LEDGER_ARM}" \
        --cell "mamnet-vanilla-loco-r450-${CITY}-hr-s42-n450 [sar-${ROLE}]" \
        --jobid "${J}" --script "$(readlink -f "$0")" --log "${LOG}" --note "${NOTE}"
    N_SUB=$((N_SUB+1))
done

echo "=============================================================================="
echo "jobs ${N_SUB} (dry-run=${DRY_RUN})   manifest ${MANIFEST}   logs ${LOG_PATH}"
echo "=============================================================================="
