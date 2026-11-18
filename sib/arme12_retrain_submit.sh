#!/bin/bash
# FILENAME: python/arme12_retrain_submit.sh
# E-12 REMEDIATION -- retrain the 3 contaminated MAMNet SIB C4clean HR cells
# with a clean FDA target pool. HLT-ordered; confirmation in
# summaries/session_logs/2026-08-03-e12-and-riders.md.
#
# THE CONTAMINATION. train_mamnet_sib.py hands fda_target_root to
# load_target_images, which os.walks the root and draws 100 files. The three
# published C4clean HR cells were given the CITY ROOT, whose walk yields 2250
# candidates -- 1350 train + 450 val + 450 test across images and both mask
# subtrees. 20.0% of the candidate list is the test subtree and all 150 test
# filenames are in it, so ~20 of the 100 drawn files were held-out test imagery.
#
# THE ONE CHANGE. fda_target_root -> {city}/highres/train/images, the same
# 450-image directory the 18 CLEAN standalone-FDA cells pin. Draw rule
# unchanged. Every other hyperparameter, flag, split and schedule is lifted
# verbatim from the published cell's own config.json.
#
# SCOPE. MAMNet SIB C4clean highres only, 3 cells. The 18 standalone FDA cells
# are clean -- NOT touched. OGLANet-SIB never reads fda_target_root (inert) and
# DINOv3-SIB has use_fda false -- NOT touched.
#
# THE SEED CANNOT BE MATCHED. The published config.json is json.dump(vars(args))
# and carries no `seed` key, so that script version had no --seed argument and
# no seeding block: those runs were not explicitly seeded and the seed is
# UNRECOVERABLE. This wave uses the current default 42. Old-vs-new therefore
# differs in the FDA pool AND in RNG stream, and no delta is attributable to the
# pool alone. Recorded in every run.json under provenance.seed_caveat.
#
# THE PUBLISHED ARTIFACTS ARE EVIDENCE. Nothing is written into, moved out of,
# or annotated in the published run directories. The worker checksums each
# published dir's listing before and after its job and fails if it changed.
#
# STAGE GATE
#   1  smoke   one cell, 2 epochs, 5 test images, throwaway roots
#   2  repro   all 3 published checkpoints re-evaluated; each must reproduce
#              its OWN stored test_results.json within 0.01 mIoU. This is a
#              stronger stage-2 than a truncated cell: it verifies the whole
#              inference + metric + dump path against a known-good number
#              before any training burns an allocation.
#   3  retrain the 3 cells, --dependency=afterok on all three repro jobs
#
# NO RESUME. train_mamnet_sib.py has no resume path and writes only
# best_model.pth. --time is 02:30:00 against a measured 70 min (0.78 min/epoch,
# 88-96 epochs, early-stopped) -- worst case 100 epochs is ~93 min with startup
# and dump.
#
# DUMPS AND CHECKPOINTS GO TO SCRATCH. the archive root is inside its soft quota.

set -euo pipefail

# --- BASE_PATH (NCSA Delta, live) ---
BASE_PATH="${PROJECT_ROOT}"
SCRATCH_ROOT="${ST_SCRATCH_ROOT}"

# --- BASE_PATH (Gilbreth, commented unless in use) ---
# BASE_PATH="${PROJECT_ROOT}"   # second cluster

# --- BASE_PATH (Anvil, commented unless in use) ---
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
LOG_PATH="${BASE_PATH}/logs"
DUMP_ROOT="${SCRATCH_ROOT}/dumps"
CKPT_ROOT="${SCRATCH_ROOT}/data/e12_ckpt"
SMOKE_TAG="${SMOKE_TAG:-e12a}"
SMOKE_ROOT="${SCRATCH_ROOT}/e12_smoke_${SMOKE_TAG}"
WORKER="${PYTHON_DIR}/arme12_retrain_worker.sh"
LEDGER="${PYTHON_DIR}/ledger.py"
SELF="${PYTHON_DIR}/arme12_retrain_submit.sh"
MANIFEST="${LOG_PATH}/e12_retrain_manifest.json"

CITIES="${CITIES:-phoenix miami chicago}"
SEED="${SEED:-42}"
RETRAIN_TIME="${RETRAIN_TIME:-02:30:00}"
REPRO_TIME="${REPRO_TIME:-00:25:00}"
SMOKE_TIME="${SMOKE_TIME:-00:20:00}"
TIMEOUT_S=3600
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_REPRO="${SKIP_REPRO:-0}"
SMOKE_CITY="${SMOKE_CITY:-phoenix}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

mkdir -p "${LOG_PATH}" "${CKPT_ROOT}" "${SMOKE_ROOT}"

[ -f "${WORKER}" ] || { echo "FATAL missing worker: ${WORKER}" >&2; exit 1; }
[ -f "${PYTHON_DIR}/arme12_retrain.py" ] || { echo "FATAL missing trainer" >&2; exit 1; }

PROBE="${DUMP_ROOT}/.write_probe_$$"
if ! (touch "${PROBE}" && rm -f "${PROBE}") 2>/dev/null; then
    echo "FATAL ${DUMP_ROOT} is not writable; refusing to submit" >&2; exit 1
fi

GIT_HASH="$(git -C "${BASE_PATH}" rev-parse HEAD 2>/dev/null || echo '')"
GIT_DIRTY="$(git -C "${BASE_PATH}" status --porcelain 2>/dev/null | wc -l)"
[ -n "${GIT_HASH}" ] || { echo "FATAL cannot resolve a git hash" >&2; exit 1; }
ARM_SRC="python/arme12_retrain.py python/arme12_retrain_worker.sh python/arme12_retrain_submit.sh python/shadowtransfer/armc_dump.py python/shadowtransfer/armc_runtime.py python/shadowtransfer/metrics.py"
ARM_DIRTY="$(git -C "${BASE_PATH}" status --porcelain -- ${ARM_SRC} 2>/dev/null | wc -l)"
if [ "${ARM_DIRTY}" -ne 0 ]; then
    echo "FATAL ${ARM_DIRTY} of this arm's own source files are uncommitted:" >&2
    git -C "${BASE_PATH}" status --porcelain -- ${ARM_SRC} >&2
    exit 1
fi
echo "git: HEAD ${GIT_HASH}; arm sources clean; ${GIT_DIRTY} other paths dirty"

# ---- preflight: the published cells, and the pool assertion, per city -------
echo
echo "=== preflight ==="
for CITY in ${CITIES}; do
    PUB="${BASE_PATH}/data/mamnet/outputs/mamnet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_${CITY}__highres"
    for f in config.json test_results.json best_model.pth; do
        [ -f "${PUB}/${f}" ] || { echo "FATAL ${CITY}: missing ${PUB}/${f}" >&2; exit 1; }
    done
    POOL="${BASE_PATH}/data/Final_data_test/${CITY}/highres/train/images"
    TEST="${BASE_PATH}/data/Final_data_test/${CITY}/highres/test/images"
    NP="$(ls "${POOL}" | wc -l)"; NT="$(ls "${TEST}" | wc -l)"
    OV="$(comm -12 <(ls "${POOL}" | sort) <(ls "${TEST}" | sort) | wc -l)"
    echo "--- ${CITY}: clean pool ${NP} files, test ${NT} files, filename overlap ${OV}"
    [ "${OV}" -eq 0 ] || { echo "FATAL ${CITY}: pool overlaps test on ${OV} files" >&2; exit 1; }
    [ "${NP}" -eq 450 ] || { echo "FATAL ${CITY}: pool is ${NP} files, expected 450" >&2; exit 1; }
    python3 - "${PUB}/test_results.json" "${CITY}" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d["tolerant_2px"]; s = d["strict"]
print(f"    published stored: n={d['num_images']}  tol2px mIOU {t['mIOU']:.4f} "
      f"ShIOU {t['Shadow_IOU']:.4f}  |  strict mIOU {s['mIOU']:.4f}")
PYEOF
done

COMMON="BASE_PATH=${BASE_PATH},SEED=${SEED},TIMEOUT_S=${TIMEOUT_S}"
COMMON="${COMMON},ARMA_GIT_HASH=${GIT_HASH},ARMA_GIT_DIRTY=${GIT_DIRTY}"

NOTE_BASE="E-12 REMEDIATION: the published mamnet_sib_C4clean HR cells built their FDA target pool from the city ROOT (2250 candidates, 20.0% the test subtree, all 150 test filenames present), so ~20 of the 100 drawn files were held-out test imagery. THE ONE CHANGE is fda_target_root -> {city}/highres/train/images (450 files, 0 test overlap); draw rule, hyperparameters, flags, splits and schedule all verbatim from the published config.json. Published seed is UNRECOVERABLE (no seed key in config.json), so seed=42 and old-vs-new differs in RNG as well as pool. Published run dirs are read-only evidence and are checksummed before/after"

echo
echo "=== E-12 REMEDIATION ==="
echo "cities      : ${CITIES}"
echo "seed        : ${SEED}  (published seed unrecoverable)"
echo "ckpt root   : ${CKPT_ROOT}"
echo "dump root   : ${DUMP_ROOT}/e12/<run_id>"
echo "retrain     : ${RETRAIN_TIME} per cell, NO resume chain"
echo "dry-run     : ${DRY}"
echo

DRYN=0
submit() {   # submit <jobname> <log> <time> <dep> <extra_export>
    local jn="$1" log="$2" tl="$3" dep="$4" ex="$5"
    local -a cmd=(sbatch --job-name="${jn}" --output="${log}" --time="${tl}")
    [ -n "${dep}" ] && cmd+=(--dependency="${dep}")
    cmd+=(--export="ALL,${COMMON},ARMA_TIME_LIMIT=${tl},${ex}" "${WORKER}")
    if [ "${DRY}" -eq 1 ]; then
        # stdout carries ONLY the jobid -- this function is command-substituted.
        { printf 'DRY: '; printf '%q ' "${cmd[@]}"; printf '\n     log -> %s\n' "${log}"; } >&2
        DRYN=$((DRYN + 1)); echo "DRY${DRYN}"; return 0
    fi
    local out; out="$("${cmd[@]}")"
    echo "${out}" | awk '{print $NF}'
}

JOBS=""; SMOKE_JID=""; REPRO_DEP=""

if [ "${SKIP_SMOKE}" != "1" ]; then
    # BOTH task branches, at smoke cost. Job 20826399 died at the first line of
    # the inference phase (a wrong import path), which left everything after it
    # -- metrics, both metric conventions, the dump, run.json -- unexercised.
    # A smoke that stops at the first bug only proves the code before it, so
    # each branch now runs end to end and must reach .DONE.
    echo "=== STAGE 1 -- smokes (throwaway roots, both task branches) ==="
    DEP=""
    for ST in retrain repro; do
        RID="mamnet-sibc4-loco-r450-${SMOKE_CITY}-hr-s${SEED}-e12smoke-${ST}"
        LOG="${LOG_PATH}/e12_smoke_${ST}_${SMOKE_CITY}.out"
        EXTRA="SMOKE=1,SMOKE_TEST=5"
        [ "${ST}" = "retrain" ] && EXTRA="${EXTRA},SMOKE_EPOCHS=2"
        JID="$(submit "e12-smoke-${ST}-${SMOKE_CITY}" "${LOG}" "${SMOKE_TIME}" "${DEP}" \
            "CITY=${SMOKE_CITY},TASK=${ST},RUN_ID=${RID},CELL_DIR=${SMOKE_ROOT}/e12/${RID},CKPT_ROOT=${SMOKE_ROOT}/ckpt,OUT_ROOT=${SMOKE_ROOT},${EXTRA},ST_LEDGER_CELL=${RID}")"
        echo "submitted SMOKE ${ST} jobid=${JID} dep=${DEP:-none} log=${LOG}"
        if [ "${DRY}" -eq 0 ]; then
            python3 "${LEDGER}" add --arm E12 --cell "${RID}" \
                --jobid "${JID}" --script "${SELF}" --log "${LOG}" \
                --note "STAGE-1 SMOKE (${ST} branch) for the E-12 remediation, throwaway roots under ${SMOKE_ROOT}; the repro branch reads the published checkpoint READ-ONLY; requested ${SMOKE_TIME}"
        fi
        DEP="afterok:${JID}"
        SMOKE_JID="${JID}"
        JOBS="${JOBS}${JOBS:+ }${JID}"
    done
    echo
fi

if [ "${SKIP_REPRO}" != "1" ]; then
    echo "=== STAGE 2 -- reproduce the 3 published cells (no training) ==="
    DEP=""; [ -n "${SMOKE_JID}" ] && DEP="afterok:${SMOKE_JID}"
    for CITY in ${CITIES}; do
        RID="mamnet-sibc4-loco-r450-${CITY}-hr-sPUB-e12orig"
        LOG="${LOG_PATH}/e12_repro_${CITY}.out"
        JID="$(submit "e12-repro-${CITY}" "${LOG}" "${REPRO_TIME}" "${DEP}" \
            "CITY=${CITY},TASK=repro,RUN_ID=${RID},CELL_DIR=${DUMP_ROOT}/e12/${RID},CKPT_ROOT=${CKPT_ROOT},OUT_ROOT=${DUMP_ROOT},ST_LEDGER_CELL=${RID}")"
        echo "submitted REPRO ${CITY} jobid=${JID} dep=${DEP:-none} log=${LOG}"
        [ "${DRY}" -eq 0 ] && python3 "${LEDGER}" add --arm E12 --cell "${RID}" \
            --jobid "${JID}" --script "${SELF}" --log "${LOG}" \
            --note "STAGE-2 GATE: re-evaluate the PUBLISHED (contaminated-pool) checkpoint for ${CITY} and reproduce its own stored test_results.json within 0.01 mIoU; also emits a CONTRACT dump so old-vs-new is per-image. No training. Published dir read-only; requested ${REPRO_TIME}"
        REPRO_DEP="${REPRO_DEP}${REPRO_DEP:+:}${JID}"
        JOBS="${JOBS}${JOBS:+ }${JID}"
    done
    echo
fi

echo "=== STAGE 3 -- the 3 retrains with the clean pool ==="
DEP=""
if [ -n "${REPRO_DEP}" ]; then DEP="afterok:${REPRO_DEP}"
elif [ -n "${SMOKE_JID}" ]; then DEP="afterok:${SMOKE_JID}"; fi
MAN=""
for CITY in ${CITIES}; do
    RID="mamnet-sibc4-loco-r450-${CITY}-hr-s${SEED}-e12clean"
    LOG="${LOG_PATH}/e12_retrain_${CITY}.out"
    JID="$(submit "e12-retrain-${CITY}" "${LOG}" "${RETRAIN_TIME}" "${DEP}" \
        "CITY=${CITY},TASK=retrain,RUN_ID=${RID},CELL_DIR=${DUMP_ROOT}/e12/${RID},CKPT_ROOT=${CKPT_ROOT},OUT_ROOT=${DUMP_ROOT},ST_LEDGER_CELL=${RID}")"
    echo "submitted RETRAIN ${CITY} jobid=${JID} dep=${DEP:-none} log=${LOG}"
    [ "${DRY}" -eq 0 ] && python3 "${LEDGER}" add --arm E12 --cell "${RID}" \
        --jobid "${JID}" --script "${SELF}" --log "${LOG}" \
        --note "${NOTE_BASE}; requested ${RETRAIN_TIME}"
    MAN="${MAN}${MAN:+,}\"${CITY}\": {\"retrain_jobid\": \"${JID}\", \"run_id\": \"${RID}\", \"dump\": \"${DUMP_ROOT}/e12/${RID}\"}"
    JOBS="${JOBS}${JOBS:+ }${JID}"
done

if [ "${DRY}" -eq 0 ]; then
    cat > "${MANIFEST}" << MEOF
{"arm": "E12", "wave": "e12_fda_pool_remediation",
 "written_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
 "git_hash": "${GIT_HASH}",
 "the_one_change": "fda_target_root -> {city}/highres/train/images",
 "seed": ${SEED}, "seed_provenance": "published seed unrecoverable; not in config.json",
 "smoke_jobid": "${SMOKE_JID:-none}", "repro_gate": "${REPRO_DEP:-none}",
 "retrain_time": "${RETRAIN_TIME}", "resume": "none (published script has no resume path)",
 "cells": {${MAN}}}
MEOF
    echo "manifest -> ${MANIFEST}"
fi

echo
echo "=== job shape ==="
echo "jobs        : $([ "${SKIP_SMOKE}" = "1" ] && echo 0 || echo 1) smoke + $([ "${SKIP_REPRO}" = "1" ] && echo 0 || echo $(echo ${CITIES} | wc -w)) repro + $(echo ${CITIES} | wc -w) retrain"
echo "per-job     : retrain ${RETRAIN_TIME}, repro ${REPRO_TIME}, smoke ${SMOKE_TIME}"
echo "GPU-hours   : ceiling $(python3 -c "print(round(3*2.5 + 3*0.42 + 0.33, 2))"); expected ~$(python3 -c "print(round(3*1.2 + 3*0.1 + 0.15, 2))") at the measured 0.78 min/epoch and 88-96 epochs"
echo "jobids      : ${JOBS:-none}"
echo "dumps       : ${DUMP_ROOT}/e12/<run_id>/{perimage.parquet,probs.npz,run.json,.DONE}"
