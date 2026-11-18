#!/bin/bash
# FILENAME: python/n450_tgtfda_submit.sh
# OGLANet-SIB target-FDA retrain -- driver for the three cells
#   oglanet-sibc4tgt-loco-r450-{chicago,miami,phoenix}-hr-s42-n450
#
# THE ONE CHANGE vs oglanet-sibc4-loco-r450-{city}-hr-s42-n450: the FDA style
# pool on the SIB CNN branch is the HELD-OUT city's train-split images (paper
# S5.1 / App. F), not the other SOURCE city's train set that
# dataset_sib.py:291-294 falls back to. Everything else -- the 450-image train
# list, the 300-image val list, seed 42, epochs 100, patience 25, batch 8,
# fp16, lr/schedule/optimiser, SIB loss weights, img_size, highres, the
# checkpoint-selection metric -- is the n450 cell's own config, enforced by
# CHECK 4 (n450_config.diff), which refuses any other difference.
#
# --time ARITHMETIC (step 2a): the three cross-source n450 SIB cells measured
#   chicago 8000.3 s, miami 7083.2 s, phoenix 6439.2 s (their run.json
#   wall_clock_s). max = 8000.302 s; x 1.3 = 10400.39 s = 2 h 53 m 20.39 s;
#   rounded UP to the minute = 2:54:00.
#
# Usage:
#   n450_tgtfda_submit.sh --stage smoke            [--dry-run] [--test-only]
#   n450_tgtfda_submit.sh --stage real --shape one3gpu|three1gpu [--dry-run] [--test-only]
set -euo pipefail

BASE_PATH="${PROJECT_ROOT}"
PYTHON_DIR="${BASE_PATH}"
SHADE="${ST_SCRATCH_ROOT}"
N450_ROOT="${SHADE}/n450"
DUMP_ROOT="${SHADE}/dumps/st_r450_published_recompute"
CKPT_ROOT="${SHADE}/data/oglanet/outputs/n450"
LOG_PATH="${SHADE}/logs/tgtfda"
MANIFEST="${SHADE}/logs/tgtfda_manifest.json"
LEDGER="${PYTHON_DIR}/ledger.py"
LEDGER_ARM="N450"
CONTRACT="${PYTHON_DIR}/shadowtransfer/worker_contract.py"
ST_DISK_GATE="${PYTHON_DIR}/shadowtransfer/disk_gate.py"
export ST_DISK_GATE
WORKER1="${PYTHON_DIR}/n450_tgtfda_worker.sh"
WORKER3="${PYTHON_DIR}/n450_tgtfda_worker3.sh"
DRIVER_PY="${PYTHON_DIR}/n450_tgtfda.py"
TRAIN_PY="${PYTHON_DIR}/n450_train.py"
TIME_REAL="2:54:00"
TIME_SMOKE="0:25:00"
SMOKE_CITY="phoenix"
SMOKE_PART="gpuA100x4-interactive"       # 1:00:00 cap; the smoke fits, real does not
# 3 cells x (one ~509 MB best checkpoint + 150 float16 prob .npy + a ~38 MB
# probs.npz + a ~80 KB parquet) ~= 3 x 0.72 GiB, plus the trainer's prediction
# PNGs. Rounded up hard.
FOOTPRINT_GIB=5
MARGIN_GIB=10

STAGE=""; SHAPE="three1gpu"; DRY_RUN=0; TEST_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --shape) SHAPE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --test-only) TEST_ONLY=1; shift ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done
case "${STAGE}" in smoke|real) ;; *) echo "--stage smoke|real required" >&2; exit 2 ;; esac
case "${SHAPE}" in one3gpu|three1gpu) ;; *) echo "--shape one3gpu|three1gpu" >&2; exit 2 ;; esac
mkdir -p "${LOG_PATH}" "${DUMP_ROOT}" "${CKPT_ROOT}" "${N450_ROOT}"

RUN_IDS=$(python3 -c "
import sys; sys.path.insert(0,'${PYTHON_DIR}')
import n450_tgtfda as T
print(' '.join(T.new_run_id(c) for c in ('chicago','miami','phoenix')))")
SMOKE_RID=$(python3 -c "
import sys; sys.path.insert(0,'${PYTHON_DIR}')
import n450_tgtfda as T; print(T.new_run_id('${SMOKE_CITY}'))")
echo "[cells] ${RUN_IDS}"
echo "[smoke] ${SMOKE_RID}"

# ---- donor lists (step 0b) must exist and PASS -------------------------------
echo "[donor] verifying the step-0b manifest"
python3 - <<'PY'
import json, os, sys
sys.path.insert(0, os.environ["PROJECT_ROOT"])
import n450_tgtfda as T
m = T.donor_expectations()
for city, e in m["cities"].items():
    lp = e["list_path"]
    body = open(lp).read()
    import hashlib
    got = hashlib.sha256(body.encode()).hexdigest()
    assert got == e["file_sha256"], f"{lp}: sha {got} != {e['file_sha256']}"
    n = len([l for l in body.splitlines() if l.strip()])
    assert n == e["n_donor"] == 450, f"{lp}: {n} lines"
    a = e["assertions"]
    assert a["A1_all_under_holdout_train_images"]["n_outside"] == 0
    assert a["A2_no_masks"]["n_in_mask_dir"] == 0
    assert a["A2_no_masks"]["n_maskish_filename"] == 0
    assert a["A2_no_masks"]["n_not_3channel"] == 0
    for sp, v in a["A3_no_test_or_val_overlap"].items():
        assert v["n_id_overlap"] == 0 and v["n_content_sha256_overlap"] == 0, (city, sp)
    assert a["A4_no_source_city_files"]["n_path_hits"] == 0
    assert a["A4_no_source_city_files"]["n_id_hits"] == 0
    print(f"[donor] {city}: 450 files, PASS, file_sha={e['file_sha256'][:16]}")
PY

# ---- CHECK 4 on all three cells, before anything is submitted ----------------
echo "[check4] config diff vs the published cell, all three"
for RID in ${RUN_IDS}; do
    OUT=$("${BASE_PATH}/envs/satmae_cuda12/bin/python" -u "${DRIVER_PY}" \
          --run_id "${RID}" --dry_run 2>&1) || { echo "${OUT}"; exit 4; }
    V=$(echo "${OUT}" | grep -c "VERDICT PERMITTED" || true)
    CH=$(echo "${OUT}" | grep -c "^  CHANGED " || true)
    OO=$(echo "${OUT}" | grep -c "^  ORIG-ONLY " || true)
    NO=$(echo "${OUT}" | grep -c "^  NEW-ONLY " || true)
    echo "[check4] ${RID}: PERMITTED=${V} changed=${CH} orig_only=${OO} new_only=${NO}"
    if [ "${V}" -ne 1 ] || [ "${CH}" -ne 2 ] || [ "${OO}" -ne 0 ] || [ "${NO}" -ne 0 ]; then
        echo "REFUSED ${RID}: CHECK 4 must be PERMITTED with exactly 2 changed keys" >&2
        echo "${OUT}" | grep -E "CHANGED|ORIG-ONLY|NEW-ONLY|VERDICT" >&2
        exit 4
    fi
    echo "${OUT}" | grep -E "^  CHANGED (fda_target_root|output_dir):" | cut -c1-120
done

# ---- driver/worker variable contract ----------------------------------------
echo "[contract] selftest"
python3 "${CONTRACT}" --selftest

# ---- disk ---------------------------------------------------------------------
python3 - <<'PY'
import os
a, b = os.stat(os.environ["ST_SCRATCH_ROOT"]), os.stat(os.environ["ST_QUOTA_ALLOC"])
assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), "not the same volume"
print(f"[disk] scratch == quota alloc (dev {a.st_dev}, inode {a.st_ino})")
PY
HEAD_GIB=$(python3 "${ST_DISK_GATE}" --alloc "${ST_QUOTA_ALLOC}" --json \
    | python3 -c "import json,sys; print(round(json.load(sys.stdin)['free']/2**30,1))")
echo "[disk] headroom to soft ${HEAD_GIB} GiB; footprint ${FOOTPRINT_GIB} GiB; margin floor ${MARGIN_GIB} GiB"
python3 -c "import sys; sys.exit(0 if float('${HEAD_GIB}')-${FOOTPRINT_GIB} >= ${MARGIN_GIB} else 1)" \
    || { echo "STOP: margin under ${MARGIN_GIB} GiB" >&2; exit 1; }
for T in "${DUMP_ROOT}" "${CKPT_ROOT}" "${LOG_PATH}" "${N450_ROOT}"; do
    mkdir -p "${T}"
    dd if=/dev/zero of="${T}/.probe" bs=1M count=600 conv=fsync status=none
    rm -f "${T}/.probe"
    echo "[probe] 600 MiB conv=fsync OK at $(readlink -f "${T}")"
done

# ---- manifest BEFORE submitting -----------------------------------------------
python3 - "${MANIFEST}" "${STAGE}" "${SHAPE}" "${TIME_REAL}" <<'PY'
import json, os, sys, time
sys.path.insert(0, os.environ["PROJECT_ROOT"])
import n450_tgtfda as T
cells = T.cells()
m = {"arm": "N450", "sub_arm": "tgtfda",
     "what": "3 OGLANet-SIB matched-450 LOCO cells retrained with the FDA "
             "style pool set to the HELD-OUT city's train-split images "
             "(paper S5.1 / App. F) instead of the other SOURCE city's "
             "train set (dataset_sib.py:291-294 fallback)",
     "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
     "stage": sys.argv[2], "shape": sys.argv[3], "time_limit": sys.argv[4],
     "time_limit_arithmetic": "max(8000.302, 7083.2, 6439.2) s x 1.3 = "
                              "10400.39 s = 2:53:20.39 -> up to the minute "
                              "= 2:54:00",
     "dump_root": T.R.DUMP_ROOT, "n_cells": len(cells),
     "run_ids": sorted(cells),
     "comparators": {rid: f"oglanet-sibc4-loco-r450-{c['city']}-hr-s42-n450"
                     for rid, c in cells.items()},
     "donor_manifest": T.DONOR_MANIFEST,
     "dump_dirs": {rid: c["dump_dir"] for rid, c in cells.items()},
     "ckpt_dirs": {rid: c["ckpt_dir"] for rid, c in cells.items()},
     "target_access": "unlabeled_target"}
tmp = sys.argv[1] + ".new"
json.dump(m, open(tmp, "w"), indent=1)
assert json.load(open(tmp))["n_cells"] == 3
os.replace(tmp, sys.argv[1])
print(f"[manifest] {sys.argv[1]} ({len(cells)} run_ids)")
PY

# ---- submission ----------------------------------------------------------------
sub_one() {   # run_id role time partition
    local RID="$1" ROLE="$2" T="$3" PART="$4"
    local CDIR NAME LOG EXPORTS EXTRA=()
    CDIR=$(python3 -c "
import sys; sys.path.insert(0,'${PYTHON_DIR}')
import n450_tgtfda as T
from shadowtransfer import n450_roster as R
c=T.cells()['${RID}']; print(R.smoke_dir(c) if '${ROLE}'=='smoke' else c['dump_dir'])")
    NAME="tgtfda-${ROLE}-${RID}"
    LOG="${LOG_PATH}/${NAME}.out"
    if [ -f "${CDIR}/.DONE" ]; then echo "SKIP  ${NAME} (.DONE at ${CDIR})"; return 0; fi
    EXPORTS="ALL,BASE_PATH=${BASE_PATH},RUN_ID=${RID},ROLE=${ROLE},LEDGER_ARM=${LEDGER_ARM},LOG_FILE=${LOG}"
    if ! python3 "${CONTRACT}" --worker "${WORKER1}" --py "${DRIVER_PY}" \
            --py "${TRAIN_PY}" --base "${BASE_PATH}" --exports "${EXPORTS}" \
            --label "${NAME}"; then
        echo "REFUSED ${NAME}: driver/worker variable contract violated" >&2; exit 5
    fi
    local TIME_NOTE
    if [ "${ROLE}" = "smoke" ]; then
        TIME_NOTE="--time ${T}: stage-1 smoke (2 epochs, 5 scored images), sized from the ~6 min the equivalent cross-source smoke took, NOT from the real cells' arithmetic"
    else
        TIME_NOTE="--time ${T} = max measured wall 8000.3 s x 1.3 = 10400.39 s = 2:53:20.39, rounded up to the minute"
    fi
    [ -n "${PART}" ] && EXTRA+=(--partition="${PART}")
    if [ "${TEST_ONLY}" -eq 1 ]; then
        sbatch --test-only "${EXTRA[@]}" --time="${T}" --job-name="${NAME}" \
            --output="${LOG}" --export="${EXPORTS}" "${WORKER1}" 2>&1 | sed "s/^/TEST  ${NAME}: /"
        return 0
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "DRY   sbatch ${EXTRA[*]-} --time=${T} --job-name=${NAME} --output=${LOG} --export=${EXPORTS} ${WORKER1}"
        return 0
    fi
    local J
    J=$(sbatch --parsable "${EXTRA[@]}" --time="${T}" --job-name="${NAME}" \
        --output="${LOG}" --export="${EXPORTS}" "${WORKER1}")
    echo "SUB   ${J}  ${NAME}  time=${T} part=${PART:-default}  -> ${LOG}"
    python3 "${LEDGER}" add --event submit --state SUBMITTED --arm "${LEDGER_ARM}" \
        --cell "${RID} [tgtfda-${ROLE}]" --jobid "${J}" \
        --script "$(readlink -f "$0")" --log "${LOG}" \
        --note "OGLANet-SIB target-FDA retrain (FDA donor = held-out city train images, paper S5.1/App.F); ONE change vs oglanet-sibc4-loco-r450-{city}-hr-s42-n450; ${TIME_NOTE}"
}

sub_three_on_one() {   # all three cities, one job, one GPU each
    local NAME LOG EXPORTS J
    NAME="tgtfda-real-3cities"
    LOG="${LOG_PATH}/${NAME}.out"
    EXPORTS="ALL,BASE_PATH=${BASE_PATH},RUN_IDS=${RUN_IDS},ROLE=real,LEDGER_ARM=${LEDGER_ARM},LOG_PATH=${LOG_PATH},LOG_FILE=${LOG}"
    if ! python3 "${CONTRACT}" --worker "${WORKER3}" --py "${DRIVER_PY}" \
            --py "${TRAIN_PY}" --base "${BASE_PATH}" --exports "${EXPORTS}" \
            --label "${NAME}"; then
        echo "REFUSED ${NAME}: driver/worker variable contract violated" >&2; exit 5
    fi
    if [ "${TEST_ONLY}" -eq 1 ]; then
        sbatch --test-only --time="${TIME_REAL}" --job-name="${NAME}" \
            --output="${LOG}" --export="${EXPORTS}" "${WORKER3}" 2>&1 | sed "s/^/TEST  ${NAME}: /"
        return 0
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "DRY   sbatch --time=${TIME_REAL} --job-name=${NAME} --output=${LOG} --export=${EXPORTS} ${WORKER3}"
        return 0
    fi
    J=$(sbatch --parsable --time="${TIME_REAL}" --job-name="${NAME}" \
        --output="${LOG}" --export="${EXPORTS}" "${WORKER3}")
    echo "SUB   ${J}  ${NAME}  time=${TIME_REAL}  -> ${LOG}"
    python3 "${LEDGER}" add --event submit --state SUBMITTED --arm "${LEDGER_ARM}" \
        --cell "tgtfda 3 cities on one node: ${RUN_IDS}" --jobid "${J}" \
        --script "$(readlink -f "$0")" --log "${LOG}" \
        --note "OGLANet-SIB target-FDA retrain, 3 cities / 3 GPUs / 1 node (operator instruction 2026-11-01: minimise wall clock; breaks one-cell-per-job); --time ${TIME_REAL} = max measured wall 8000.3 s x 1.3 rounded up; per-cell logs ${LOG_PATH}/<run_id>.gpuN.out"
}

if [ "${STAGE}" = "smoke" ]; then
    sub_one "${SMOKE_RID}" smoke "${TIME_SMOKE}" "${SMOKE_PART}"
elif [ "${SHAPE}" = "one3gpu" ]; then
    sub_three_on_one
else
    for RID in ${RUN_IDS}; do sub_one "${RID}" real "${TIME_REAL}" ""; done
fi
echo "[done] stage=${STAGE} shape=${SHAPE} dry_run=${DRY_RUN} test_only=${TEST_ONLY}"
