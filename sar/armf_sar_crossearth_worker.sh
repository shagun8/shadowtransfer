#!/bin/bash
# FILENAME: python/armf_sar_crossearth_worker.sh
# Arm F SAR on the three INTERNAL CrossEarth vanilla LOCO checkpoints (hr, s0).
# ONE CELL PER JOB. Test-time adaptation only: loads an existing checkpoint,
# runs an unadapted pass and one 10-step SAR pass over the held-out city's
# canonical 150-image test list, and writes results.json + perstep.parquet.
# Trains nothing, writes no checkpoint, touches no published artifact.
# Driven by python/armf_sar_crossearth_submit.sh, which checks this file's
# variable contract against each cell's --export string before sbatch.
#
# DERIVED FROM armf_sar_n450_worker.sh (the internal SAR schema) and
# armA_t1_cesar_worker.sh (the CrossEarth environment), both read in full.
# What is carried from the CrossEarth worker and why:
#   * the crossearth_mm env and the OMITTED `module load cudatoolkit`: mmcv
#     2.1.0 ships only a cu118/torch2.1 wheel whose torch carries its own CUDA
#     runtime in nvidia/* wheels. Loading cudatoolkit would put a second
#     libcudart on the path. Arm F stage-0 finding.
#   * XFORMERS_DISABLED=1, so the DINOv2 layers take their plain-attention path.
#   * the verified backbone sha/bytes, which the driver re-checks on disk.
#   * the trap BEFORE the environment block, so an env failure still leaves a
#     .FAILED marker and a FAILED ledger line (CLAUDE.md).
#
# There is no CPU account on this allocation (a GPU account only).

# --- NCSA Delta (default, live) ---------------------------------------------
# gpuA100x4-interactive is also viable for this cell (a cell is ~16 min against
# that partition MaxTime of 01:00:00) and the submit script can select it with
# --interactive, which overrides the directive below on the sbatch line. It is
# NOT the default: QOS gpua100x4intqos carries MaxJobsPU=1 and
# MaxSubmitJobsPU=1 (measured 2026-11-03), so a multi-cell fan-out cannot be
# queued there at all.
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
# #SBATCH --partition=gpuA100x4-interactive   # see the note above
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0:40:00

# --- Gilbreth (commented unless in use) -------------------------------------
# #SBATCH -A sukkusur
# #SBATCH --partition=a100-40gb
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=0:40:00

# --- Anvil (commented unless in use) ----------------------------------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=0:40:00

set -euo pipefail

# ---- required exports: every one of these is read below ---------------------
: "${BASE_PATH:?BASE_PATH not exported}"
: "${CITY:?CITY not exported}"
: "${ROLE:?ROLE not exported -- expected smoke|real. Never defaulted: a grid cell that ran as a smoke would write a truncated dump under a real directory name}"
: "${OUT_ROOT:?OUT_ROOT not exported}"
[ -n "${TAG+x}" ] || { echo "FATAL TAG not exported (it may be empty on a real cell, but it must be set)" >&2; exit 1; }
: "${PARENT_DUMPS:?PARENT_DUMPS not exported -- the dump root of the parent vanilla cell}"
: "${REPO_ROOT:?REPO_ROOT not exported}"
: "${ENV_PATH:?ENV_PATH not exported}"
: "${BACKBONE_CKPT:?BACKBONE_CKPT not exported}"
: "${BACKBONE_SHA256:?BACKBONE_SHA256 not exported}"
: "${BACKBONE_BYTES:?BACKBONE_BYTES not exported}"
: "${SAR_LR:?SAR_LR not exported -- the recipe is passed in, never defaulted}"
: "${SAR_STEPS:?SAR_STEPS not exported -- the recipe is passed in, never defaulted}"
: "${MOMENTUM:?MOMENTUM not exported}"
: "${RHO:?RHO not exported}"
: "${SEED:?SEED not exported}"
: "${TIMEOUT_S:?TIMEOUT_S not exported}"
: "${LEDGER_ARM:?LEDGER_ARM not exported}"
: "${ARMF_TIME_LIMIT:?ARMF_TIME_LIMIT not exported}"
case "${ROLE}" in smoke|real) ;; *) echo "bad ROLE=${ROLE}" >&2; exit 1 ;; esac

LIMIT_FLAG=""
if [ "${ROLE}" = "smoke" ]; then LIMIT_FLAG="--limit ${SMOKE_IMAGES-5}"; fi

CELL_DIR="${OUT_ROOT}/armF_sar_crossearth_${CITY}_highres${TAG}"
export CELL_DIR
export ST_LEDGER_ARM="${LEDGER_ARM}"
export ST_LEDGER_CELL="crossearth-vanilla-loco-r450-${CITY}-hr-s0 [sar-${ROLE}]"
export ST_LEDGER_LOG="${LOG_FILE-}"
export ST_LEDGER_NOTE="arm F SAR on the internal CrossEarth vanilla LOCO checkpoint, test-time only"
export ST_COMPUTE_NOTE="test-time adaptation (no training) on gpuA100x4 with 1 A100 in use; there is no CPU association on this allocation (a GPU account only)"
export ARMF_TIME_LIMIT

# Traps BEFORE the module/conda block: an environment failure must still leave
# a .FAILED marker and a FAILED ledger line (CLAUDE.md).
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

# --- modules + env (Delta, live) --------------------------------------------
# `module load cudatoolkit` is deliberately OMITTED here; see the header.
module purge
PYTHON_BIN="${ENV_PATH}/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "FATAL: no python at ${PYTHON_BIN}" >&2; exit 1; }

# --- modules + env (Gilbreth, commented unless in use) ----------------------
# no crossearth_mm env is built on Gilbreth
# module purge
# PYTHON_BIN="${ENV_PATH}/bin/python"

# --- modules + env (Anvil, commented unless in use) -------------------------
# no crossearth_mm env is built on Anvil
# module purge
# PYTHON_BIN="${ENV_PATH}/bin/python"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg          # no plotting, ever, inside a worker
export XFORMERS_DISABLED=1     # DINOv2 layers take their plain-attention path
export TOKENIZERS_PARALLELISM=false

echo "=============================================================================="
echo "ARM F SAR -- internal CrossEarth vanilla LOCO"
echo "  city          ${CITY}"
echo "  role          ${ROLE}"
echo "  parent        crossearth-vanilla-loco-r450-${CITY}-hr-s0"
echo "  parent dumps  ${PARENT_DUMPS}"
echo "  out dir       ${CELL_DIR}"
echo "  recipe        lr ${SAR_LR}  steps ${SAR_STEPS}  momentum ${MOMENTUM}  rho ${RHO}  seed ${SEED}"
echo "  job id        ${SLURM_JOB_ID:-none}"
echo "  node          ${SLURMD_NODENAME:-none}"
echo "  partition     ${SLURM_JOB_PARTITION:-none}"
echo "  time limit    ${ARMF_TIME_LIMIT}"
echo "  timeout_s     ${TIMEOUT_S}"
echo "  gpu           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo none)"
echo "  repo_root     ${REPO_ROOT}"
echo "  env           ${ENV_PATH}"
echo "  python        ${PYTHON_BIN}"
echo "  backbone      ${BACKBONE_CKPT}"
echo "  backbone sha  ${BACKBONE_SHA256}  ${BACKBONE_BYTES} B"
echo "  compute_note  ${ST_COMPUTE_NOTE}"
echo "  started       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================================="

if [ -f "${CELL_DIR}/.DONE" ]; then
    echo "SKIP ${CELL_DIR}: .DONE exists"
    exit 0
fi

cd "${BASE_PATH}/python"

${PYTHON_BIN} -u "${BASE_PATH}/sar/armf_sar_crossearth.py" \
    --city "${CITY}" \
    --out_root "${OUT_ROOT}" \
    --tag="${TAG}" \
    --parent_dumps "${PARENT_DUMPS}" \
    --repo_root "${REPO_ROOT}" \
    --backbone_ckpt "${BACKBONE_CKPT}" \
    --backbone_sha256 "${BACKBONE_SHA256}" \
    --backbone_bytes "${BACKBONE_BYTES}" \
    --lr "${SAR_LR}" \
    --steps "${SAR_STEPS}" \
    --momentum "${MOMENTUM}" \
    --rho "${RHO}" \
    --seed "${SEED}" \
    --timeout_s "${TIMEOUT_S}" \
    ${LIMIT_FLAG}

echo "[${CITY}/${ROLE}] worker finished rc=0 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
