#!/bin/bash
# FILENAME: python/armA_b4_probe_submit.sh
# ARM A BLOCK 4 -- submit the s4.2 probe over tiers 1 and 3 (D-B4-2): 90 cells.
#   tier 1  bsm, isw x mamnet/dinov3/oglanet   54   vanilla model build
#   tier 3  crossearth x vanilla/bsm/fda/sib   36   armA_b4_ce_shim
#   NOT submitted: SAR 36 and tier 2 108 -- deferred, recorded in the inventory.
#
# JOB SHAPE -- the three stages, mechanised as a dependency chain
#   stage 1  5 smokes (one per arch + crossearth/sib), truncated, minutes
#   stage 2  5 real gate cells on the same parents, each afterok its smoke
#   gate     1 sanity job (armA_b4_probe_sanity.py), afterok all 5 gate cells
#   stage 3  the remaining 85, afterok the sanity job
# CLAUDE.md has stage 2 end the turn for the operator. The operator's block-4
# brief (2026-10-17) specifies "smoke per arch -> a few real cells ->
# machine-gated fan-out on a sanity job", so the human gate is REPLACED by that
# job, which fails -- holding all 85 -- unless every gate cell reproduced its
# parent under its arch's rule, kept unseen exclusion live, had the transform
# reach the model, and stayed under 80% of its --time. That is the deviation
# and this is where it is stated.
#
# --time is from MEASURED wall clocks (existing probe run.json, max per
# arch x fold; CrossEarth from the D-B3-2 E-9 3-pass jobs + the <=105 s
# source-CDF step), roughly 2x, never the 3:59:59 ceiling.
#
# Usage:  bash python/armA_b4_probe_submit.sh [--dry-run]
set -euo pipefail

# --- NCSA Delta (default, live) ---
BASE_PATH="${PROJECT_ROOT}"
SCRATCH_ROOT="${ST_SCRATCH_ROOT}"
ENV_SATMAE="${BASE_PATH}/envs/satmae_cuda12"
ENV_CE="${BASE_PATH}/envs/crossearth_mm"

# --- Gilbreth (commented unless in use) ---
# BASE_PATH="${PROJECT_ROOT}"   # second cluster
# ENV_SATMAE="${BASE_PATH}/conda_envs/satmae_cuda12"
# ENV_CE="${BASE_PATH}/conda_envs/crossearth_mm"

# --- Anvil (commented unless in use) ---
# BASE_PATH="${PROJECT_ROOT}"   # third cluster
# ENV_SATMAE="${CONDA_ENV_PATH}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
LOG_ROOT="${SCRATCH_ROOT}/logs/armA_b4"
LEDGER="${PYTHON_DIR}/ledger.py"
WORKER="${PYTHON_DIR}/armA_b4_probe_worker.sh"
SANITY_WORKER="${PYTHON_DIR}/armA_b4_sanity_worker.sh"
DRIVER="${PYTHON_DIR}/armA_b4_probe_driver.py"
HM="${PYTHON_DIR}/armA_seo_histmatch.py"
CONTRACT="${PYTHON_DIR}/shadowtransfer/worker_contract.py"
ST_DISK_GATE="${PYTHON_DIR}/shadowtransfer/disk_gate.py"
SELF="python/armA_b4_probe_submit.sh"
INVENTORY="${LOG_ROOT}/armA_b4_probe_inventory.json"
MANIFEST="${LOG_ROOT}/armA_b4_probes_manifest.json"
GATE_OUT="${LOG_ROOT}/armA_b4_gate.json"

SEO_RAW="${SCRATCH_ROOT}/data/seo_raw"
SPLITS_DIR="${SCRATCH_ROOT}/data/seo_splits_acq3"
BUDGET_DIR="${SCRATCH_ROOT}/data/seo_budget_acq3"
CKPT_ROOT="${SCRATCH_ROOT}/data/armA_ckpt"
IMG_ROOT="${SEO_RAW}/pansharpened"
MASK_ROOT="${SEO_RAW}/shadows_max"
PATCH_INDEX="${BASE_PATH}/logs/armA_seo_patch_index_acq3.json"
OUT_ROOT="${SCRATCH_ROOT}/dumps"
SMOKE_ROOT="${SCRATCH_ROOT}/armA_b4_probe_smoke"
RUNG=3500
NUM_WORKERS=8
SMOKE_TIME="00:20:00"
SANITY_TIME="00:10:00"
NEED_GIB=70              # 90 cells x <=0.65 GiB measured (JAX probe dumps), rounded up
# gate cells: one real cell per arch + crossearth/sib (the explicitly-handled checkpoint)
GATE_IDS="mamnet-bsm-loco-r3500-oma-hr-s0 dinov3-isw-loco-r3500-oma-hr-s0 oglanet-bsm-loco-r3500-oma-hr-s0 crossearth-vanilla-loco-r3500-oma-hr-s0 crossearth-sib-loco-r3500-oma-hr-s0"

# Nodes kept out of this block, each with its reason. gpua075: job 22959224 got
# EIO reading a file on OST006a there at 2026-10-17T23:00Z, while the file read
# cleanly from dt-login02 (stale client import after the 10-16..10-17 outage).
EXCLUDE_NODES="${EXCLUDE_NODES-gpua075}"
echo "excluded nodes: ${EXCLUDE_NODES:-none}"

DRY=0
[ "${1-}" = "--dry-run" ] && DRY=1
echo "dry-run: ${DRY}"

time_for () {   # measured max x ~2, per arch x fold
    case "$1:$2" in
        mamnet:JAX|dinov3:JAX) echo 00:35:00 ;;
        mamnet:OMA|dinov3:OMA|mamnet:UCSD|dinov3:UCSD) echo 00:25:00 ;;
        oglanet:JAX) echo 00:40:00 ;;
        oglanet:OMA) echo 00:25:00 ;;
        oglanet:UCSD) echo 00:30:00 ;;
        crossearth:JAX) echo 00:45:00 ;;
        crossearth:OMA) echo 00:30:00 ;;
        crossearth:UCSD) echo 00:35:00 ;;
        *) echo "FATAL no measured time for $1:$2" >&2; return 1 ;;
    esac
}
secs () { IFS=: read -r h m s <<< "$1"; echo $(( 10#$h * 3600 + 10#$m * 60 + 10#$s )); }
# Idempotent against the queue: a job of the same name already pending/running
# is reused (as the dependency) rather than submitted twice.
queued () { squeue -u "${USER}" -h -o "%i %j" | awk -v n="$1" '$2 == n {print $1; exit}'; }

for p in "${WORKER}" "${SANITY_WORKER}" "${DRIVER}" "${HM}" "${CONTRACT}" "${ST_DISK_GATE}" \
         "${PATCH_INDEX}" "${SPLITS_DIR}" "${BUDGET_DIR}" "${IMG_ROOT}" "${MASK_ROOT}" \
         "${ENV_SATMAE}/bin/python" "${ENV_CE}/bin/python"; do
    [ -e "${p}" ] || { echo "FATAL missing input: ${p}" >&2; exit 1; }
done
mkdir -p "${LOG_ROOT}"

# ---- gate E2: the inventory must resolve every roster cell ------------------
python3 "${PYTHON_DIR}/armA_b4_probe_roster.py" --out "${INVENTORY}" \
    || { echo "FATAL inventory/roster did not resolve (gate E2)" >&2; exit 1; }

# ---- the standing contract preflight, self-tested first --------------------
python3 "${CONTRACT}" --selftest >/dev/null || { echo "FATAL worker_contract selftest" >&2; exit 1; }

# ---- disk: Delta's quota tool via the single gate implementation -----------
GATE_JSON="$(timeout 120 python3 "${ST_DISK_GATE}" --json)" || { echo "FATAL disk gate unreadable -- refusal" >&2; exit 1; }
python3 - "${GATE_JSON}" "${NEED_GIB}" << 'PYEOF' || exit 1
import json, sys
g, need = json.loads(sys.argv[1]), float(sys.argv[2])
free = g["free"] / 2**30
print(f"disk gate: {g['alloc']} used {g['used']/2**30:.1f} GiB, soft {g['soft']/2**30:.0f} GiB, free-to-soft {free:.1f} GiB, need {need:.0f} GiB")
if g["over_soft"] or free < need:
    sys.exit("FATAL disk gate refuses: over soft or insufficient headroom")
PYEOF
if [ "${DRY}" -eq 0 ]; then
    for t in "${OUT_ROOT}/seo" "${SMOKE_ROOT}/seo"; do
        mkdir -p "${t}"
        dd if=/dev/zero of="${t}/.probe_$$" bs=1M count=600 conv=fsync status=none \
            || { rm -f "${t}/.probe_$$"; echo "FATAL 600 MB write probe failed at ${t}" >&2; exit 1; }
        rm -f "${t}/.probe_$$"; echo "write probe 600 MB conv=fsync OK at ${t}"
    done
fi

# ---- git: this block's own sources must be committed -----------------------
GIT_HASH="$(git -C "${BASE_PATH}" rev-parse HEAD)"
GIT_DIRTY="$(git -C "${BASE_PATH}" status --porcelain | wc -l)"
SRC="python/armA_b4_probe_submit.sh python/armA_b4_probe_worker.sh python/armA_b4_sanity_worker.sh python/armA_b4_probe_driver.py python/armA_b4_ce_shim.py python/armA_b4_ce_recipe.json python/armA_b4_probe_roster.py python/armA_b4_probe_sanity.py python/shadowtransfer/worker_contract.py python/armA_seo_histmatch.py"
SRC_DIRTY="$(git -C "${BASE_PATH}" status --porcelain -- ${SRC} | wc -l)"
if [ "${SRC_DIRTY}" -ne 0 ]; then
    git -C "${BASE_PATH}" status --porcelain -- ${SRC} >&2
    if [ "${DRY}" -eq 0 ]; then echo "FATAL ${SRC_DIRTY} block-4 sources uncommitted" >&2; exit 1; fi
    echo "WARN (dry-run) ${SRC_DIRTY} block-4 sources uncommitted"
fi
echo "git HEAD ${GIT_HASH}; ${GIT_DIRTY} paths dirty tree-wide"

COMMON="BASE_PATH=${BASE_PATH},BUDGET_DIR=${BUDGET_DIR},PATCH_INDEX=${PATCH_INDEX},SPLITS_DIR=${SPLITS_DIR}"
COMMON="${COMMON},IMG_ROOT=${IMG_ROOT},MASK_ROOT=${MASK_ROOT},CKPT_ROOT=${CKPT_ROOT},ENV_SATMAE=${ENV_SATMAE}"
COMMON="${COMMON},ENV_CE=${ENV_CE},NUM_WORKERS=${NUM_WORKERS},RUNG=${RUNG},ARMA_GIT_HASH=${GIT_HASH},ARMA_GIT_DIRTY=${GIT_DIRTY}"
NOTE="armA block4 s4.2 probe (D-B4-2), inference only; histmatch both rules + pbar columns"
N=0; JOBS=""

submit () {   # $1 worker $2 name $3 log $4 time $5 dep $6 exports $7 ledger-cell $8 py-for-contract
    local exports="$6"
    python3 "${CONTRACT}" --worker "$1" --py "$8" --py "${HM}" --base "${BASE_PATH}" \
        --exports "ALL,${exports}" --label "$2" >&2 \
        || { echo "FATAL contract violated for $2 -- nothing submitted for it" >&2; exit 1; }
    local cmd=(sbatch --job-name="$2" --output="$3" --time="$4")
    [ -n "${EXCLUDE_NODES}" ] && cmd+=(--exclude="${EXCLUDE_NODES}")
    [ -n "$5" ] && cmd+=(--dependency="afterok:$5")
    cmd+=(--export="ALL,${exports}" "$1")
    if [ "${DRY}" -eq 1 ]; then
        printf 'DRY: %s ... --time=%s dep=%s -> %s\n' "$2" "$4" "${5:-none}" "$3" >&2
        echo "DRY"; return 0
    fi
    local out jid
    out="$("${cmd[@]}")"; jid="$(echo "${out}" | awk '{print $NF}')"
    python3 "${LEDGER}" add --arm A --cell "$7" --jobid "${jid}" --script "${SELF}" \
        --log "$3" --note "${NOTE}; requested $4" >/dev/null
    echo "submitted $2 jobid=${jid} time=$4 dep=${5:-none} log=$3" >&2
    echo "${jid}"
}

cell_exports () {   # $1 task $2 out_root $3 run_id ... (from the inventory TSV row)
    local task=$1 oroot=$2 rid=$3 arch=$4 method=$5 hold=$6 seed=$7 pdir=$8 cdir=$9 tlim=${10}
    local e="${COMMON},TASK=${task},RUN_ID=${rid},ARCH=${arch},METHOD=${method},HOLDOUT=${hold},SEED=${seed}"
    e="${e},CELL_DIR=${oroot}/seo/${rid},OUT_ROOT=${oroot},PARENT_DUMP_DIR=${pdir},COMPARATOR_DUMP_DIR=${cdir}"
    e="${e},ARMA_TIME_LIMIT=${tlim},TIMEOUT_S=$(( $(secs "${tlim}") - 180 )),ST_LEDGER_CELL=${rid}"
    [ "${task}" = "smoke" ] && e="${e},SMOKE_TEST=6,SMOKE_TRAIN=48"
    echo "${e}"
}

ROWS="$(python3 - "${INVENTORY}" << 'PYEOF'
import json, sys
inv = json.load(open(sys.argv[1]))
for c in inv["cells"]:
    if c.get("runnable"):
        print("\t".join(str(c[k]) for k in ("run_id_probe", "arch", "method", "holdout", "seed",
                                            "parent_dump_dir", "comparator_dump_dir", "run_id")))
PYEOF
)"
[ "$(echo "${ROWS}" | grep -c .)" -eq 90 ] || { echo "FATAL expected 90 runnable cells, got $(echo "${ROWS}" | grep -c .)" >&2; exit 1; }

# ---- the manifest, written BEFORE anything is submitted ---------------------
if [ "${DRY}" -eq 0 ]; then
python3 - "${INVENTORY}" "${MANIFEST}" "${GIT_HASH}" "${GATE_IDS}" << 'PYEOF'
import json, sys
inv, out, git, gate = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4].split()
m = {"arm": "A", "block": 4, "authority": "DECISIONS_armA_wave2.md D-B4-2", "git_hash": git,
     "probe": "s4.2 histogram matching (published gray + arm C per-channel) + pbar columns",
     "family_rule": "D-A2-2: vanilla feeds the headline 0-of-N count; method families never do "
                    "and every method-family dump carries mixed_denominator_disclosure",
     "status_counts": inv["status_counts"], "n_submitted": inv["n_runnable"],
     "gate_cells": [g + "-p1histmatch" for g in gate],
     "cells": [{k: c.get(k) for k in ("run_id", "run_id_probe", "arch", "method", "family", "holdout",
                                      "seed", "status", "reason", "parent_dump_dir",
                                      "comparator_dump_dir", "probe_dump_dir")} for c in inv["cells"]]}
open(out + ".new", "w").write(json.dumps(m, indent=1))
import os; os.replace(out + ".new", out)
print(f"manifest written: {out} ({len(m['cells'])} parents, {m['n_submitted']} submitted)")
PYEOF
fi

declare -A SMOKE_JID GATE_JID
GATE_SPEC=""; GATE_DEPS=""

# ---- stage 1 + 2: smoke then one real gate cell, per gate parent -----------
for gid in ${GATE_IDS}; do
    row="$(echo "${ROWS}" | awk -F'\t' -v g="${gid}" '$8 == g')"
    [ -n "${row}" ] || { echo "FATAL gate cell ${gid} is not a runnable roster cell" >&2; exit 1; }
    IFS=$'\t' read -r rid arch method hold seed pdir cdir parent <<< "${row}"
    tl="$(time_for "${arch}" "${hold}")"
    sdep=""
    if [ ! -f "${OUT_ROOT}/seo/${rid}/.DONE" ]; then
        if [ ! -f "${SMOKE_ROOT}/seo/${rid}/.DONE" ] && q="$(queued "armA-b4-smoke-${parent}")" && [ -n "${q}" ]; then
            sdep="${q}"; echo "reusing queued armA-b4-smoke-${parent} ${q}" >&2
        elif [ ! -f "${SMOKE_ROOT}/seo/${rid}/.DONE" ]; then
            sj="$(submit "${WORKER}" "armA-b4-smoke-${parent}" "${LOG_ROOT}/smoke_${rid}.out" "${SMOKE_TIME}" "" \
                "$(cell_exports smoke "${SMOKE_ROOT}" "${rid}" "${arch}" "${method}" "${hold}" "${seed}" "${pdir}" "${cdir}" "${SMOKE_TIME}")" \
                "${rid}-smoke" "${DRIVER}")"
            sdep="${sj}"; N=$((N + 1)); JOBS="${JOBS} ${sj}"
        fi
        q="$(queued "armA-b4-gate-${parent}")"
        if [ -n "${q}" ]; then gj="${q}"; echo "reusing queued armA-b4-gate-${parent} ${q}" >&2; else
        gj="$(submit "${WORKER}" "armA-b4-gate-${parent}" "${LOG_ROOT}/${rid}.out" "${tl}" "${sdep}" \
            "$(cell_exports real "${OUT_ROOT}" "${rid}" "${arch}" "${method}" "${hold}" "${seed}" "${pdir}" "${cdir}" "${tl}")" \
            "${rid}" "${DRIVER}")"
        N=$((N + 1)); fi
        JOBS="${JOBS} ${gj}"
        GATE_DEPS="${GATE_DEPS:+${GATE_DEPS}:}${gj}"
    else
        echo "gate cell ${rid}: .DONE exists, used as-is" >&2
    fi
    GATE_SPEC="${GATE_SPEC:+${GATE_SPEC};}${OUT_ROOT}/seo/${rid}=${tl}"
done

# ---- the machine gate ------------------------------------------------------
SAN_EXP="BASE_PATH=${BASE_PATH},RUN_ID=armA-b4-gate,CELL_DIR=${LOG_ROOT}/gate,GATE_CELLS=${GATE_SPEC},GATE_OUT=${GATE_OUT},ST_LEDGER_CELL=armA-b4-machine-gate"
SANJ="$(submit "${SANITY_WORKER}" "armA-b4-sanity" "${LOG_ROOT}/sanity.out" "${SANITY_TIME}" "${GATE_DEPS}" \
    "${SAN_EXP}" "armA-b4-machine-gate" "${PYTHON_DIR}/armA_b4_probe_sanity.py")"
N=$((N + 1)); JOBS="${JOBS} ${SANJ}"

# ---- stage 3: the fan-out, afterok the machine gate ------------------------
NFAN=0; GPUS_S=0
while IFS=$'\t' read -r rid arch method hold seed pdir cdir parent; do
    case " ${GATE_IDS} " in *" ${parent} "*) continue ;; esac
    if [ -f "${OUT_ROOT}/seo/${rid}/.DONE" ]; then echo "skip ${rid}: .DONE" >&2; continue; fi
    tl="$(time_for "${arch}" "${hold}")"
    jid="$(submit "${WORKER}" "armA-b4-${parent}" "${LOG_ROOT}/${rid}.out" "${tl}" "${SANJ}" \
        "$(cell_exports real "${OUT_ROOT}" "${rid}" "${arch}" "${method}" "${hold}" "${seed}" "${pdir}" "${cdir}" "${tl}")" \
        "${rid}" "${DRIVER}")"
    N=$((N + 1)); NFAN=$((NFAN + 1)); JOBS="${JOBS} ${jid}"; GPUS_S=$((GPUS_S + $(secs "${tl}")))
done <<< "${ROWS}"

echo
echo "=== job shape ==="
echo "jobs        : ${N} (smokes + gate cells + 1 machine gate + ${NFAN} fan-out)"
echo "gate cells  : ${GATE_IDS}"
echo "fan-out     : afterok ${SANJ} (the machine gate); per-cell --time from the measured table"
echo "GPU-h ceil  : fan-out $(python3 -c "print(round(${GPUS_S}/3600, 1))") (sum of --time); expected ~half that"
echo "logs        : ${LOG_ROOT}/<run_id>.out ; gate verdict ${GATE_OUT}"
echo "manifest    : ${MANIFEST}"
echo "inventory   : ${INVENTORY}"
echo "jobids      : ${JOBS}"
echo "cancel list : squeue -u \$USER -h -o '%i %j' | awk '\$2 ~ /^armA-b4-/ {print \$1}'"
