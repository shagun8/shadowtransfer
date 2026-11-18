#!/bin/bash
# FILENAME: python/armf_sar_n450_worker.sh
# Arm F SAR re-run on the n450 MAMNet standard models -- ONE CELL PER JOB.
# Test-time adaptation only: loads a checkpoint, runs an unadapted pass and one
# SAR pass over the held-out city's 150 test images, measures weight drift, and
# writes results.json + perstep.parquet. Trains nothing and writes no
# checkpoint. Driven by python/armf_sar_n450_submit.sh, which checks this
# file's variable contract against each cell's --export string before sbatch.

# --- NCSA Delta (default, live) ---------------------------------------------
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0:45:00

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
# #SBATCH --time=0:45:00

# --- Anvil (commented unless in use) ----------------------------------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=0:45:00

set -euo pipefail

# ---- required exports: every one of these is read below ---------------------
: "${BASE_PATH:?BASE_PATH not exported}"
: "${CITY:?CITY not exported}"
: "${ROLE:?ROLE not exported}"
: "${OUT_ROOT:?OUT_ROOT not exported}"
: "${TAG:?TAG not exported}"
: "${LEDGER_ARM:?LEDGER_ARM not exported}"
: "${ARMF_TIME_LIMIT:?ARMF_TIME_LIMIT not exported}"
case "${ROLE}" in smoke|real) ;; *) echo "bad ROLE=${ROLE}" >&2; exit 1 ;; esac

LIMIT_FLAG=""
if [ "${ROLE}" = "smoke" ]; then LIMIT_FLAG="--limit ${SMOKE_IMAGES-5}"; fi

CELL_DIR="${OUT_ROOT}/armF_sar_mamnet_${CITY}_highres${TAG}"
export CELL_DIR
export ST_LEDGER_ARM="${LEDGER_ARM}"
export ST_LEDGER_CELL="mamnet-vanilla-loco-r450-${CITY}-hr-s42-n450 [sar-${ROLE}]"
export ST_LEDGER_LOG="${LOG_FILE-}"
export ST_LEDGER_NOTE="arm F SAR on the n450 MAMNet standard model, test-time only"

# Traps BEFORE the module/conda block: an environment failure must still leave
# a .FAILED marker and a FAILED ledger line (CLAUDE.md).
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

# --- modules + env (Delta) --------------------------------------------------
module purge
module load cudatoolkit          # required before pytorch-conda/2.8
module load pytorch-conda/2.8
set +u
conda activate ${BASE_PATH}/envs/satmae_cuda12
set -u
cd "${BASE_PATH}/sar/"
PYTHON_BIN=${BASE_PATH}/envs/satmae_cuda12/bin/python

# --- modules + env (Gilbreth, commented unless in use) ----------------------
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${BASE_PATH}/conda_envs/satmae_cuda12
# cd "${BASE_PATH}/sar/"
# PYTHON_BIN=${BASE_PATH}/conda_envs/satmae_cuda12/bin/python

# --- modules + env (Anvil, commented unless in use) -------------------------
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${CONDA_ENV_PATH}   # third cluster
# cd "${BASE_PATH}/sar/"
# PYTHON_BIN=${CONDA_ENV_PATH}/bin/python   # third cluster

export PYTHONUNBUFFERED=1
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo none)

echo "=============================================================================="
echo "ARM F SAR n450 worker"
echo "  city          ${CITY}"
echo "  role          ${ROLE}"
echo "  out dir       ${CELL_DIR}"
echo "  job id        ${SLURM_JOB_ID:-none}"
echo "  node          ${SLURMD_NODENAME:-none}"
echo "  partition     ${SLURM_JOB_PARTITION:-none}"
echo "  time limit    ${ARMF_TIME_LIMIT}"
echo "  gpu           ${GPU_NAME}"
echo "  python        ${PYTHON_BIN}"
echo "  started       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================================="

if [ -f "${CELL_DIR}/.DONE" ]; then
    echo "SKIP ${CELL_DIR}: .DONE exists"
    exit 0
fi

$PYTHON_BIN -u armf_sar_n450.py --city "${CITY}" --out_root "${OUT_ROOT}" \
    --tag="${TAG}" ${LIMIT_FLAG}

echo "worker finished rc=0 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
