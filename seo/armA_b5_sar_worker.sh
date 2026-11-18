#!/bin/bash
# FILENAME: python/armA_b5_sar_worker.sh
# ARM A BLOCK 5 (D-B5-1) -- s4.2 histogram-matching probe on the SAR cells.
# Drives python/armA_b5_sar_probe.py. INFERENCE + SAR test-time adaptation;
# trains nothing, writes no checkpoint.
#   TASK=prep   per-cell source CDF + test metadata (no GPU work)
#   TASK=shard  one slice [START,STOP) of the held-out test stream
#   TASK=smoke  prep, then two short slices (offset 0 and SMOKE_OFFSET), the
#               unmatched SAR pass repeated once -- the (b) determinism evidence
#   TASK=band   SAR band measurement (D-B5-2/D-B5-3): one slice [START,STOP),
#               unadapted + BAND_PASSES unmatched SAR passes, no matched conditions
#   TASK=bandsmoke  prep, then slice [0,SMOKE_N) in band mode (BAND_PASSES passes)
# Contract checked per cell by shadowtransfer/worker_contract.py before sbatch.
# There is no CPU account on this allocation (a GPU account only).

# --- NCSA Delta (default, live) ---
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# --- Gilbreth (commented unless in use) ---
# #SBATCH -A sukkusur
# #SBATCH --partition=a100-40gb
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G

# --- Anvil (commented unless in use) ---
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G

set -euo pipefail
export PYTHONUNBUFFERED=1

: "${BASE_PATH:?BASE_PATH unset}"
: "${TASK:?TASK unset (prep|shard|smoke)}"
: "${RUN_ID:?RUN_ID unset}"
: "${ARCH:?ARCH unset}"
: "${PARENT_DUMP_DIR:?PARENT_DUMP_DIR unset}"
: "${WORK_DIR:?WORK_DIR unset}"
: "${CELL_DIR:?CELL_DIR unset}"
: "${VANILLA_DUMPS:?VANILLA_DUMPS unset}"
: "${BUDGET_DIR:?BUDGET_DIR unset}"
: "${SPLITS_DIR:?SPLITS_DIR unset}"
: "${IMG_ROOT:?IMG_ROOT unset}"
: "${MASK_ROOT:?MASK_ROOT unset}"
: "${PATCH_INDEX:?PATCH_INDEX unset}"
: "${ENV_SATMAE:?ENV_SATMAE unset}"
: "${ENV_CE:?ENV_CE unset}"
: "${TIMEOUT_S:?TIMEOUT_S unset}"
# BAND_PASSES is branch-only (band|bandsmoke): ${BAND_PASSES-} + an explicit assert there.
: "${ARMA_TIME_LIMIT:?ARMA_TIME_LIMIT unset}"
: "${ARMA_GIT_HASH:?ARMA_GIT_HASH unset}"
: "${ARMA_GIT_DIRTY:?ARMA_GIT_DIRTY unset}"
: "${ST_LEDGER_CELL:?ST_LEDGER_CELL unset}"

case "${ARCH}" in
    mamnet|dinov3|oglanet|crossearth) ;;
    *) echo "FATAL unknown ARCH=${ARCH}" >&2; exit 1 ;;
esac
PARENT_ID="$(basename "${PARENT_DUMP_DIR}")"
case "${PARENT_ID}" in
    "${ARCH}"-sar-loco-r3500-*-hr-s*) ;;
    *) echo "FATAL PARENT_DUMP_DIR ${PARENT_DUMP_DIR} is not a ${ARCH} SAR cell" >&2; exit 1 ;;
esac
[ "${RUN_ID}" = "${PARENT_ID}-p1histmatch" ] || { echo "FATAL RUN_ID=${RUN_ID} != ${PARENT_ID}-p1histmatch" >&2; exit 1; }
[ "$(basename "${WORK_DIR}")" = "${RUN_ID}" ] || { echo "FATAL WORK_DIR ${WORK_DIR} is not named ${RUN_ID}" >&2; exit 1; }

# Branch-only inputs: ${VAR-} plus an explicit assert on the branch (CLAUDE.md).
case "${TASK}" in
    prep) [ "${CELL_DIR}" = "${WORK_DIR}" ] || { echo "FATAL prep CELL_DIR must be WORK_DIR" >&2; exit 1; } ;;
    shard)
        [ -n "${START-}" ] && [ -n "${STOP-}" ] || { echo "FATAL TASK=shard needs START and STOP" >&2; exit 1; }
        [ "${CELL_DIR}" = "${WORK_DIR}/shards/$(printf '%05d_%05d' "${START}" "${STOP}")" ] \
            || { echo "FATAL shard CELL_DIR ${CELL_DIR} is not the shard's own dir" >&2; exit 1; } ;;
    smoke)
        [ -n "${SMOKE_N-}" ] && [ -n "${SMOKE_OFFSET-}" ] || { echo "FATAL TASK=smoke needs SMOKE_N and SMOKE_OFFSET" >&2; exit 1; }
        [ "${CELL_DIR}" = "${WORK_DIR}" ] || { echo "FATAL smoke CELL_DIR must be WORK_DIR" >&2; exit 1; } ;;
    band)
        [ -n "${START-}" ] && [ -n "${STOP-}" ] && [ -n "${BAND_PASSES-}" ] || { echo "FATAL TASK=band needs START, STOP and BAND_PASSES" >&2; exit 1; }
        [ "${BAND_PASSES}" -ge 10 ] || { echo "FATAL BAND_PASSES=${BAND_PASSES} < 10: a band that gates needs >=10 passes (D-B4-3)" >&2; exit 1; }
        [ "${CELL_DIR}" = "${WORK_DIR}/shards/$(printf '%05d_%05d' "${START}" "${STOP}")" ] \
            || { echo "FATAL band CELL_DIR ${CELL_DIR} is not the shard's own dir" >&2; exit 1; } ;;
    bandsmoke)
        [ -n "${SMOKE_N-}" ] && [ -n "${BAND_PASSES-}" ] || { echo "FATAL TASK=bandsmoke needs SMOKE_N and BAND_PASSES" >&2; exit 1; }
        [ "${BAND_PASSES}" -ge 10 ] || { echo "FATAL BAND_PASSES=${BAND_PASSES} < 10 (D-B4-3)" >&2; exit 1; }
        [ "${CELL_DIR}" = "${WORK_DIR}" ] || { echo "FATAL bandsmoke CELL_DIR must be WORK_DIR" >&2; exit 1; } ;;
    sb)
        [ -n "${START-}" ] && [ -n "${STOP-}" ] || { echo "FATAL TASK=sb needs START and STOP" >&2; exit 1; }
        [ "${CELL_DIR}" = "${WORK_DIR}/shards/$(printf '%05d_%05d' "${START}" "${STOP}")" ] \
            || { echo "FATAL sb CELL_DIR ${CELL_DIR} is not the shard's own dir" >&2; exit 1; } ;;
    sbsmoke)
        [ -n "${SMOKE_N-}" ] || { echo "FATAL TASK=sbsmoke needs SMOKE_N" >&2; exit 1; }
        [ "${CELL_DIR}" = "${WORK_DIR}" ] || { echo "FATAL sbsmoke CELL_DIR must be WORK_DIR" >&2; exit 1; } ;;
    *) echo "FATAL TASK=${TASK}; expected prep|shard|smoke|band|bandsmoke|sb|sbsmoke" >&2; exit 1 ;;
esac

mkdir -p "${CELL_DIR}"
export ST_LEDGER_ARM="A"
export ST_LEDGER_NOTE="armA block5 SAR s4.2 probe ${TASK} (D-B5-1)"
# Trap FIRST, before modules and env: an env failure still leaves .FAILED.
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"
if [ -f "${CELL_DIR}/.DONE" ]; then echo "[${RUN_ID}] ${CELL_DIR}/.DONE exists -- nothing to do"; exit 0; fi
rm -f "${CELL_DIR}/.FAILED"
export ST_COMPUTE_NOTE="SAR probe on gpuA100x4 with 1 A100 (prep uses no GPU); there is no CPU association on this allocation (a GPU account only)"

# --- NCSA Delta modules + env (live) ---
module purge
if [ "${ARCH}" = "crossearth" ]; then
    PYTHON_BIN="${ENV_CE}/bin/python"
    export MPLBACKEND=Agg XFORMERS_DISABLED=1 TOKENIZERS_PARALLELISM=false
else
    module load cudatoolkit          # required before pytorch-conda/2.8
    module load pytorch-conda/2.8
    set +u                            # conda's activation hook reads an unbound var
    conda activate "${ENV_SATMAE}"
    set -u
    PYTHON_BIN="${ENV_SATMAE}/bin/python"
fi

# --- Gilbreth modules + env (commented unless in use) ---
# module purge
# if [ "${ARCH}" = "crossearth" ]; then PYTHON_BIN="${ENV_CE}/bin/python"; else
#     module load conda; module load cuda/12.1.1; module load cudnn/9.2.0.82-12
#     set +u; conda activate "${ENV_SATMAE}"; set -u; PYTHON_BIN="${ENV_SATMAE}/bin/python"; fi

# --- Anvil modules + env (commented unless in use) ---
# module purge
# if [ "${ARCH}" = "crossearth" ]; then PYTHON_BIN="${ENV_CE}/bin/python"; else
#     module load modtree/gpu; module load cuda/12.6.1; module load anaconda
#     set +u; conda activate "${ENV_SATMAE}"; set -u; PYTHON_BIN="${ENV_SATMAE}/bin/python"; fi

[ -x "${PYTHON_BIN}" ] || { echo "FATAL no python at ${PYTHON_BIN}" >&2; exit 1; }
# Node health before any read (block-4 lesson, job 22959224 on gpua075).
LFS_OUT="$(timeout 60 lfs check servers 2>&1)" || { echo "FATAL node ${SLURMD_NODENAME:-?}: lfs check servers failed -- node fault" >&2; echo "${LFS_OUT}" >&2; exit 1; }
LFS_BAD="$(echo "${LFS_OUT}" | grep -v -E 'active\.$' || true)"
[ -z "${LFS_BAD}" ] || { echo "FATAL node ${SLURMD_NODENAME:-?}: OST imports in error -- node fault:" >&2; echo "${LFS_BAD}" >&2; exit 1; }
cd "${BASE_PATH}/python"
export ARMA_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

echo "============ ARM A BLOCK 5 -- SAR s4.2 PROBE (D-B5-1) ============"
echo "run_id       : ${RUN_ID}   task ${TASK}"
echo "parent       : ${PARENT_DUMP_DIR}"
echo "work / cell  : ${WORK_DIR} / ${CELL_DIR}"
echo "slice        : ${START-}..${STOP-}   smoke n ${SMOKE_N-} offset ${SMOKE_OFFSET-}   band passes ${BAND_PASSES-}"
echo "jobid / node : ${SLURM_JOB_ID:-none} / ${SLURMD_NODENAME:-none}   gpu ${ARMA_GPU_NAME}"
echo "time limit   : ${ARMA_TIME_LIMIT}   python time-box ${TIMEOUT_S}s"
echo "python       : ${PYTHON_BIN}"
echo "git          : ${ARMA_GIT_HASH} (${ARMA_GIT_DIRTY} paths dirty tree-wide)"
echo "started      : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "==================================================================="

run_probe () {   # $1 task, rest extra args
    local t="$1"; shift
    ${PYTHON_BIN} -u "${BASE_PATH}/seo/armA_b5_sar_probe.py" --task "${t}" \
        --parent_dump_dir "${PARENT_DUMP_DIR}" --work_dir "${WORK_DIR}" \
        --vanilla_dumps "${VANILLA_DUMPS}" --budget_dir "${BUDGET_DIR}" \
        --splits_dir "${SPLITS_DIR}" --img_root "${IMG_ROOT}" --mask_root "${MASK_ROOT}" \
        --patch_index "${PATCH_INDEX}" --timeout_s "${TIMEOUT_S}" "$@"
}

case "${TASK}" in
    prep) run_probe prep ;;
    shard) run_probe shard --start "${START}" --stop "${STOP}" ;;
    smoke)
        [ -f "${WORK_DIR}/prep.DONE" ] || run_probe prep
        run_probe shard --start 0 --stop "${SMOKE_N}" --repeat_unmatched 1
        run_probe shard --start "${SMOKE_OFFSET}" --stop "$(( SMOKE_OFFSET + SMOKE_N ))" --repeat_unmatched 1
        touch "${CELL_DIR}/.DONE" ;;
    band) run_probe shard --start "${START}" --stop "${STOP}" --repeat_unmatched "$(( BAND_PASSES - 1 ))" --band_only ;;
    bandsmoke)
        [ -f "${WORK_DIR}/prep.DONE" ] || run_probe prep
        run_probe shard --start 0 --stop "${SMOKE_N}" --repeat_unmatched "$(( BAND_PASSES - 1 ))" --band_only
        touch "${CELL_DIR}/.DONE" ;;
    # stage (b), operator 2026-10-26: unmatched + gray-matched + per-channel SAR, no unadapted re-run
    sb) run_probe shard --start "${START}" --stop "${STOP}" --skip_unadapted ;;
    sbsmoke)
        [ -f "${WORK_DIR}/prep.DONE" ] || run_probe prep
        run_probe shard --start 0 --stop "${SMOKE_N}" --skip_unadapted
        touch "${CELL_DIR}/.DONE" ;;
esac
[ "${TASK}" = "prep" ] && { [ -f "${WORK_DIR}/prep.DONE" ] || { echo "FATAL prep wrote no prep.DONE" >&2; exit 1; }; touch "${CELL_DIR}/.DONE"; }
[ -f "${CELL_DIR}/.DONE" ] || { echo "FATAL ${TASK} returned without ${CELL_DIR}/.DONE" >&2; exit 1; }
echo "[${RUN_ID}] ${TASK} finished cleanly at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
