#!/bin/bash
# FILENAME: python/armA_b4_probe_worker.sh
# ARM A BLOCK 4 -- the s4.2 histogram-matching probe, tiers 1 and 3 (D-B4-2).
# ONE CELL PER JOB. INFERENCE ONLY: trains nothing, writes no checkpoint.
#
# DERIVED FROM armA_seo_histmatch_worker.sh, whose variable contract was READ IN
# FULL (2026-10-17) and found unable to run block 4 at all:
#   * it had NO METHOD variable, so every method cell would have SILENTLY PROBED
#     A VANILLA PARENT (the probe's --method defaults to vanilla). METHOD is now
#     REQUIRED, validated against D-B4-2's approved (arch, method) pairs, and
#     cross-checked against RUN_ID and the parent directory's name;
#   * it refused oglanet by name on the 2026-08-02 drop. That refusal is LIFTED:
#     D-A2-2 (2026-08-21) superseded it and puts OGLANet and CrossEarth,
#     vanilla AND method, in the s4.2 probe's scope;
#   * it did not know crossearth. Admitted here, in CrossEarth's own env
#     (envs/crossearth_mm: mmcv 2.1.0 ships only a cu118/torch2.1 wheel, so
#     NO cudatoolkit / pytorch-conda module -- armA_b3_ce_e9_worker.sh's
#     stage-0 finding, carried over);
#   * it read parent and comparator from one VANILLA_DUMP_ROOT. Replaced by
#     PARENT_DUMP_DIR + COMPARATOR_DUMP_DIR, resolved per cell by
#     armA_b4_probe_roster.py.
# Every export the submit script sets is checked against this file by
# shadowtransfer/worker_contract.py before sbatch, in both directions.
#
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
: "${TASK:?TASK unset (smoke|real)}"
: "${RUN_ID:?RUN_ID unset}"
: "${ARCH:?ARCH unset}"
: "${METHOD:?FATAL METHOD unset -- required by name; a defaulted METHOD probes a vanilla parent silently}"
: "${HOLDOUT:?HOLDOUT unset}"
: "${SEED:?SEED unset}"
: "${RUNG:?RUNG unset}"
: "${CELL_DIR:?CELL_DIR unset}"
: "${OUT_ROOT:?OUT_ROOT unset}"
: "${PARENT_DUMP_DIR:?PARENT_DUMP_DIR unset}"
: "${COMPARATOR_DUMP_DIR:?COMPARATOR_DUMP_DIR unset}"
: "${BUDGET_DIR:?BUDGET_DIR unset}"
: "${PATCH_INDEX:?PATCH_INDEX unset}"
: "${SPLITS_DIR:?SPLITS_DIR unset}"
: "${IMG_ROOT:?IMG_ROOT unset}"
: "${MASK_ROOT:?MASK_ROOT unset}"
: "${CKPT_ROOT:?CKPT_ROOT unset}"
: "${ENV_SATMAE:?ENV_SATMAE unset}"
: "${ENV_CE:?ENV_CE unset}"
: "${NUM_WORKERS:?NUM_WORKERS unset}"
: "${TIMEOUT_S:?TIMEOUT_S unset}"
: "${ARMA_TIME_LIMIT:?ARMA_TIME_LIMIT unset}"
: "${ARMA_GIT_HASH:?ARMA_GIT_HASH unset}"
: "${ARMA_GIT_DIRTY:?ARMA_GIT_DIRTY unset}"
: "${ST_LEDGER_CELL:?ST_LEDGER_CELL unset}"

case "${ARCH}" in
    mamnet|dinov3|oglanet|crossearth) ;;
    *) echo "FATAL unknown ARCH=${ARCH}" >&2; exit 1 ;;
esac
case "${METHOD}" in
    sar) echo "FATAL ${ARCH}/sar: probe not run -- requires a build that runs SAR's test-time adaptation on the matched images, deferred 2026-10-17 (D-B4-2)" >&2; exit 1 ;;
    fada|iim|mrfp_plus|segdesic) echo "FATAL ${ARCH}/${METHOD}: probe not run -- requires per-method model build (tier 2), deferred 2026-10-17 (D-B4-2)" >&2; exit 1 ;;
esac
case "${ARCH}:${METHOD}" in
    mamnet:bsm|mamnet:isw|dinov3:bsm|dinov3:isw|oglanet:bsm|oglanet:isw) TIER=1 ;;
    crossearth:vanilla|crossearth:bsm|crossearth:fda|crossearth:sib) TIER=3 ;;
    *) echo "FATAL (${ARCH}, ${METHOD}) is not in D-B4-2's approved scope" >&2; exit 1 ;;
esac
case "${HOLDOUT}" in
    JAX|OMA|UCSD) ;;
    *) echo "FATAL unknown HOLDOUT=${HOLDOUT}" >&2; exit 1 ;;
esac
case "${TASK}" in
    smoke|real) ;;
    *) echo "FATAL TASK=${TASK}; expected smoke|real" >&2; exit 1 ;;
esac

# Identity: the labels, the run_id and the parent directory must all agree, so
# a mislabelled export cannot probe a different parent than its name claims.
HOLD_LC="$(echo "${HOLDOUT}" | tr 'A-Z' 'a-z')"
PARENT_ID="${ARCH}-${METHOD}-loco-r${RUNG}-${HOLD_LC}-hr-s${SEED}"
COMP_ID="${ARCH}-none-upper-r${RUNG}-${HOLD_LC}-hr-s${SEED}"
[ "${RUN_ID}" = "${PARENT_ID}-p1histmatch" ] || { echo "FATAL RUN_ID=${RUN_ID} != ${PARENT_ID}-p1histmatch" >&2; exit 1; }
[ "$(basename "${PARENT_DUMP_DIR}")" = "${PARENT_ID}" ] || { echo "FATAL PARENT_DUMP_DIR ${PARENT_DUMP_DIR} is not ${PARENT_ID}" >&2; exit 1; }
[ "$(basename "${COMPARATOR_DUMP_DIR}")" = "${COMP_ID}" ] || { echo "FATAL COMPARATOR_DUMP_DIR ${COMPARATOR_DUMP_DIR} is not ${COMP_ID}" >&2; exit 1; }
[ "${CELL_DIR}" = "${OUT_ROOT}/seo/${RUN_ID}" ] || { echo "FATAL CELL_DIR ${CELL_DIR} != OUT_ROOT/seo/RUN_ID (where the probe writes)" >&2; exit 1; }

mkdir -p "${CELL_DIR}"
export ST_LEDGER_ARM="A"
export ST_LEDGER_NOTE="armA block4 s4.2 probe tier ${TIER} (D-B4-2)"
# Trap FIRST, before modules and env: an env failure still leaves .FAILED.
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

if [ -f "${CELL_DIR}/.DONE" ]; then
    echo "[${RUN_ID}] .DONE exists at ${CELL_DIR} -- nothing to do"
    exit 0
fi
rm -f "${CELL_DIR}/.FAILED"

export ST_COMPUTE_NOTE="inference-only probe on gpuA100x4 with 1 A100 in use; there is no CPU association on this allocation (a GPU account only)"

# D-B4-3: an OGLANet cell may be gated on its parent's 10-pass band instead of
# the n=3 record. Optional, so read with ${VAR-} and asserted on its branch.
E9_ARGS=""
if [ -n "${E9_REPEAT_FILE-}" ]; then
    [ "${ARCH}" = "oglanet" ] || { echo "FATAL E9_REPEAT_FILE is oglanet-only" >&2; exit 1; }
    [ "$(dirname "${E9_REPEAT_FILE}")" = "${PARENT_DUMP_DIR}" ] || { echo "FATAL E9_REPEAT_FILE not in the parent dump dir" >&2; exit 1; }
    [ -f "${E9_REPEAT_FILE}" ] || { echo "FATAL E9_REPEAT_FILE ${E9_REPEAT_FILE} missing" >&2; exit 1; }
    E9_ARGS="--e9_repeat_file ${E9_REPEAT_FILE}"
fi

SMOKE_ARGS=""
if [ "${TASK}" = "smoke" ]; then
    [ -n "${SMOKE_TEST-}" ] || { echo "FATAL TASK=smoke needs SMOKE_TEST" >&2; exit 1; }
    [ -n "${SMOKE_TRAIN-}" ] || { echo "FATAL TASK=smoke needs SMOKE_TRAIN" >&2; exit 1; }
    SMOKE_ARGS="--smoke_test ${SMOKE_TEST} --smoke_train ${SMOKE_TRAIN}"
fi

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

# Node health, before any read. A compute node whose Lustre client holds an OST
# import in error returns EIO on every file with objects on that OST: job
# 22959224 (gpua075, 2026-10-17) died reading the dinov3 recipe args.json, whose
# only object is on OST006a -- the OST NCSA had down 10-16..10-17 -- while the
# same file read cleanly from dt-login02. That is a node fault, not a cell
# fault, so it is reported as one, in seconds, before the cell spends anything.
LFS_OUT="$(timeout 60 lfs check servers 2>&1)" || {
    echo "FATAL node ${SLURMD_NODENAME:-?}: lfs check servers failed or timed out -- node fault, not a cell fault; resubmit excluding this node" >&2
    echo "${LFS_OUT}" >&2; exit 1; }
LFS_BAD="$(echo "${LFS_OUT}" | grep -v -E 'active\.$' || true)"
if [ -n "${LFS_BAD}" ]; then
    echo "FATAL node ${SLURMD_NODENAME:-?}: Lustre client has OST imports in error -- node fault, not a cell fault; resubmit excluding this node:" >&2
    echo "${LFS_BAD}" >&2; exit 1
fi
echo "node health   : lfs check servers clean on ${SLURMD_NODENAME:-?} ($(echo "${LFS_OUT}" | grep -c 'active\.$') imports active)"
cd "${BASE_PATH}/python"
export ARMA_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

echo "============ ARM A BLOCK 4 -- s4.2 PROBE, tier ${TIER} (D-B4-2) ============"
echo "run_id        : ${RUN_ID}"
echo "arch / method : ${ARCH} / ${METHOD}   holdout ${HOLDOUT}   rung ${RUNG}   seed ${SEED}"
echo "task          : ${TASK}   smoke args: ${SMOKE_ARGS:-none}"
echo "jobid / node  : ${SLURM_JOB_ID:-none} / ${SLURMD_NODENAME:-none}   gpu ${ARMA_GPU_NAME}"
echo "time limit    : ${ARMA_TIME_LIMIT}   post-load time-box ${TIMEOUT_S}s"
echo "python        : ${PYTHON_BIN}"
echo "parent dir    : ${PARENT_DUMP_DIR}"
echo "comparator dir: ${COMPARATOR_DUMP_DIR}"
echo "E-9 band file : ${E9_REPEAT_FILE:-parent inference_repeat.json (default)}"
echo "cell dir      : ${CELL_DIR}"
echo "git           : ${ARMA_GIT_HASH} (${ARMA_GIT_DIRTY} other paths dirty)"
echo "compute_note  : ${ST_COMPUTE_NOTE}"
echo "started       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

${PYTHON_BIN} -u "${BASE_PATH}/python/armA_b4_probe_driver.py" \
    --arch "${ARCH}" \
    --method "${METHOD}" \
    --holdout "${HOLDOUT}" \
    --rung "${RUNG}" \
    --seed "${SEED}" \
    --budget_dir "${BUDGET_DIR}" \
    --patch_index "${PATCH_INDEX}" \
    --splits_dir "${SPLITS_DIR}" \
    --img_root "${IMG_ROOT}" \
    --mask_root "${MASK_ROOT}" \
    --ckpt_root "${CKPT_ROOT}" \
    --vanilla_dump_root "$(dirname "${PARENT_DUMP_DIR}")" \
    --out_root "${OUT_ROOT}" \
    --num_workers "${NUM_WORKERS}" \
    --timeout_s "${TIMEOUT_S}" \
    --parent_dump_dir "${PARENT_DUMP_DIR}" \
    --comparator_dump_dir "${COMPARATOR_DUMP_DIR}" \
    --family_json "${CELL_DIR}/probe_family.json" \
    ${E9_ARGS} ${SMOKE_ARGS}

[ -f "${CELL_DIR}/.DONE" ] || { echo "FATAL probe returned without writing ${CELL_DIR}/.DONE" >&2; exit 1; }
echo "[${RUN_ID}] finished cleanly at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
