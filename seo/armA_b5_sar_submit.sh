#!/bin/bash
# FILENAME: python/armA_b5_sar_submit.sh
# ARM A BLOCK 5 (D-B5-1) -- SAR s4.2 probe submission.
#   --stage smoke   4 smoke jobs, one per arch, on the OMA s0 SAR parent:
#                   prep + two 8-image slices (offset 0 and mid-stream), the
#                   unmatched SAR pass repeated once. This is the evidence for
#                   the (b) gate question; nothing writes a dump.
#   (grid stage: added only after the operator rules SAR's gate -- D-B5-1 asks
#    for (a)/(b) BEFORE any SAR cell runs.)
# Usage:  bash python/armA_b5_sar_submit.sh --stage smoke [--dry-run]
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
LOG_ROOT="${SCRATCH_ROOT}/logs/armA_b5"
LEDGER="${PYTHON_DIR}/ledger.py"
WORKER="${PYTHON_DIR}/armA_b5_sar_worker.sh"
PROBE="${PYTHON_DIR}/armA_b5_sar_probe.py"
CONTRACT="${PYTHON_DIR}/shadowtransfer/worker_contract.py"
ST_DISK_GATE="${PYTHON_DIR}/shadowtransfer/disk_gate.py"
SELF="python/armA_b5_sar_submit.sh"
INVENTORY="${SCRATCH_ROOT}/logs/armA_b4/armA_b4_probe_inventory_final.json"
SEO_RAW="${SCRATCH_ROOT}/data/seo_raw"
SPLITS_DIR="${SCRATCH_ROOT}/data/seo_splits_acq3"
BUDGET_DIR="${SCRATCH_ROOT}/data/seo_budget_acq3"
IMG_ROOT="${SEO_RAW}/pansharpened"
MASK_ROOT="${SEO_RAW}/shadows_max"
PATCH_INDEX="${BASE_PATH}/logs/armA_seo_patch_index_acq3.json"
VANILLA_DUMPS="${SCRATCH_ROOT}/dumps/seo"
SMOKE_ROOT="${SCRATCH_ROOT}/armA_b5_sar_smoke"
SMOKE_N=8
SMOKE_OFFSET=818          # mid-stream of OMA's 1637 test patches
NEED_GIB=5
EXCLUDE_NODES="${EXCLUDE_NODES-gpua075}"

STAGE=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "FATAL unknown arg $1" >&2; exit 1 ;;
    esac
done
[ "${STAGE}" = "smoke" ] || { echo "FATAL --stage smoke is the only stage built; the grid waits on the operator's SAR gate ruling" >&2; exit 1; }
echo "stage ${STAGE}  dry-run ${DRY}  excluded nodes ${EXCLUDE_NODES:-none}"

smoke_time () { case "$1" in crossearth) echo 00:45:00 ;; *) echo 00:30:00 ;; esac; }
secs () { IFS=: read -r h m s <<< "$1"; echo $(( 10#$h * 3600 + 10#$m * 60 + 10#$s )); }
queued () { squeue -u "${USER}" -h -o "%i %j" | awk -v n="$1" '$2 == n {print $1; exit}'; }

for p in "${WORKER}" "${PROBE}" "${CONTRACT}" "${ST_DISK_GATE}" "${INVENTORY}" "${PATCH_INDEX}" \
         "${SPLITS_DIR}" "${BUDGET_DIR}" "${IMG_ROOT}" "${MASK_ROOT}" "${VANILLA_DUMPS}" \
         "${ENV_SATMAE}/bin/python" "${ENV_CE}/bin/python"; do
    [ -e "${p}" ] || { echo "FATAL missing input: ${p}" >&2; exit 1; }
done
mkdir -p "${LOG_ROOT}"

python3 "${CONTRACT}" --selftest >/dev/null || { echo "FATAL worker_contract selftest" >&2; exit 1; }

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
    mkdir -p "${SMOKE_ROOT}"
    dd if=/dev/zero of="${SMOKE_ROOT}/.probe_$$" bs=1M count=600 conv=fsync status=none \
        || { rm -f "${SMOKE_ROOT}/.probe_$$"; echo "FATAL 600 MB write probe failed at ${SMOKE_ROOT}" >&2; exit 1; }
    rm -f "${SMOKE_ROOT}/.probe_$$"; echo "write probe 600 MB conv=fsync OK at ${SMOKE_ROOT}"
fi

GIT_HASH="$(git -C "${BASE_PATH}" rev-parse HEAD)"
GIT_DIRTY="$(git -C "${BASE_PATH}" status --porcelain | wc -l)"
SRC="python/armA_b5_sar_submit.sh python/armA_b5_sar_worker.sh python/armA_b5_sar_probe.py python/shadowtransfer/armA_b5_sar.py"
SRC_DIRTY="$(git -C "${BASE_PATH}" status --porcelain -- ${SRC} | wc -l)"
if [ "${SRC_DIRTY}" -ne 0 ]; then
    git -C "${BASE_PATH}" status --porcelain -- ${SRC} >&2
    [ "${DRY}" -eq 0 ] && { echo "FATAL ${SRC_DIRTY} block-5 sources uncommitted" >&2; exit 1; }
    echo "WARN (dry-run) ${SRC_DIRTY} block-5 sources uncommitted"
fi
echo "git HEAD ${GIT_HASH}; ${GIT_DIRTY} paths dirty tree-wide"

COMMON="BASE_PATH=${BASE_PATH},VANILLA_DUMPS=${VANILLA_DUMPS},BUDGET_DIR=${BUDGET_DIR},SPLITS_DIR=${SPLITS_DIR}"
COMMON="${COMMON},IMG_ROOT=${IMG_ROOT},MASK_ROOT=${MASK_ROOT},PATCH_INDEX=${PATCH_INDEX},ENV_SATMAE=${ENV_SATMAE}"
COMMON="${COMMON},ENV_CE=${ENV_CE},ARMA_GIT_HASH=${GIT_HASH},ARMA_GIT_DIRTY=${GIT_DIRTY}"
NOTE="armA block5 SAR s4.2 probe (D-B5-1) smoke: prep + 2 slices x ${SMOKE_N}, unmatched SAR repeated"
N=0; JOBS=""; GPUS_S=0

for ARCH in mamnet dinov3 oglanet crossearth; do
    PARENT_ID="${ARCH}-sar-loco-r3500-oma-hr-s0"
    PDIR="$(python3 - "${INVENTORY}" "${PARENT_ID}" << 'PYEOF'
import json, sys
c = [x for x in json.load(open(sys.argv[1]))["cells"] if x["run_id"] == sys.argv[2]]
assert len(c) == 1 and c[0]["final_outcome"] == "deferred_sar", c
print(c[0]["parent_dump_dir"])
PYEOF
)"
    RID="${PARENT_ID}-p1histmatch"
    WORK="${SMOKE_ROOT}/${RID}"
    TL="$(smoke_time "${ARCH}")"
    NAME="armA-b5-smoke-${PARENT_ID}"
    if [ -f "${WORK}/.DONE" ]; then echo "skip ${NAME}: .DONE"; continue; fi
    q="$(queued "${NAME}")"; [ -n "${q}" ] && { echo "skip ${NAME}: already queued ${q}"; continue; }
    EXP="${COMMON},TASK=smoke,RUN_ID=${RID},ARCH=${ARCH},PARENT_DUMP_DIR=${PDIR},WORK_DIR=${WORK},CELL_DIR=${WORK}"
    EXP="${EXP},SMOKE_N=${SMOKE_N},SMOKE_OFFSET=${SMOKE_OFFSET},TIMEOUT_S=$(( $(secs "${TL}") - 240 ))"
    EXP="${EXP},ARMA_TIME_LIMIT=${TL},ST_LEDGER_CELL=${RID}-smoke"
    python3 "${CONTRACT}" --worker "${WORKER}" --py "${PROBE}" --base "${BASE_PATH}" \
        --exports "ALL,${EXP}" --label "${NAME}" >&2 || { echo "FATAL contract violated for ${NAME}" >&2; exit 1; }
    LOG="${LOG_ROOT}/smoke_${RID}.out"
    cmd=(sbatch --job-name="${NAME}" --output="${LOG}" --time="${TL}")
    [ -n "${EXCLUDE_NODES}" ] && cmd+=(--exclude="${EXCLUDE_NODES}")
    cmd+=(--export="ALL,${EXP}" "${WORKER}")
    if [ "${DRY}" -eq 1 ]; then echo "DRY: ${NAME} --time=${TL} -> ${LOG}"; continue; fi
    out="$("${cmd[@]}")"; jid="$(echo "${out}" | awk '{print $NF}')"
    python3 "${LEDGER}" add --arm A --cell "${RID}-smoke" --jobid "${jid}" --script "${SELF}" \
        --log "${LOG}" --note "${NOTE}; requested ${TL}" >/dev/null
    echo "submitted ${NAME} jobid=${jid} time=${TL} log=${LOG}"
    N=$((N + 1)); JOBS="${JOBS} ${jid}"; GPUS_S=$((GPUS_S + $(secs "${TL}")))
done
echo "=== job shape: ${N} smoke jobs; GPU-h ceiling $(python3 -c "print(round(${GPUS_S}/3600, 2))"); jobids:${JOBS}"
echo "cancel list : squeue -u \$USER -h -o '%i %j' | awk '\$2 ~ /^armA-b5-/ {print \$1}'"
