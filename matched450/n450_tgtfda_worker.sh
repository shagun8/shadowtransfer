#!/bin/bash
# FILENAME: python/n450_tgtfda_worker.sh
# OGLANet-SIB target-FDA retrain -- ONE-GPU worker: one cell per job.
# Retrains one matched-450 OGLANet-SIB cell with the FDA style pool set to the
# HELD-OUT city's train-split images (paper S5.1 / App. F) instead of the other
# SOURCE city's train set, then scores and dumps it through the same n450
# harness as the three cross-source cells.
# Driven by python/n450_tgtfda_submit.sh, which checks this file's variable
# contract against each cell's --export string before sbatch.

# --- NCSA Delta (default, live) ---------------------------------------------
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2:54:00

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
# #SBATCH --time=2:54:00

# --- Anvil (commented unless in use) ----------------------------------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=2:54:00

set -euo pipefail

# ---- required exports: every one of these is read below ---------------------
: "${BASE_PATH:?BASE_PATH not exported}"
: "${RUN_ID:?RUN_ID not exported}"
: "${ROLE:?ROLE not exported}"
: "${LEDGER_ARM:?LEDGER_ARM not exported}"
case "${ROLE}" in smoke|real) ;; *) echo "bad ROLE=${ROLE}" >&2; exit 1 ;; esac
# optional: DUMP_ONLY=1 re-enters a trained cell (inference + dump only)
DUMP_ONLY_FLAG=""
if [ "${DUMP_ONLY-0}" = "1" ]; then DUMP_ONLY_FLAG="--dump_only"; fi

CELL_DIR=$(python3 -c "
import sys; sys.path.insert(0, '${BASE_PATH}/python')
import n450_tgtfda as T
from shadowtransfer import n450_roster as R
c = T.cells()['${RUN_ID}']
print(R.smoke_dir(c) if '${ROLE}' == 'smoke' else c['dump_dir'])")
export CELL_DIR
export ST_LEDGER_ARM="${LEDGER_ARM}"
export ST_LEDGER_CELL="${RUN_ID} [${ROLE}]"
export ST_LEDGER_LOG="${LOG_FILE-}"
export ST_LEDGER_NOTE="oglanet-sib target-FDA retrain (held-out train donor)"

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
cd "${BASE_PATH}/matched450/"
PYTHON_BIN=${BASE_PATH}/envs/satmae_cuda12/bin/python

# --- modules + env (Gilbreth, commented unless in use) ----------------------
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${BASE_PATH}/conda_envs/satmae_cuda12
# cd "${BASE_PATH}/matched450/"
# PYTHON_BIN=${BASE_PATH}/conda_envs/satmae_cuda12/bin/python

# --- modules + env (Anvil, commented unless in use) -------------------------
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${CONDA_ENV_PATH}
# cd "${BASE_PATH}/matched450/"
# PYTHON_BIN=${CONDA_ENV_PATH}/bin/python

export PYTHONUNBUFFERED=1
N450_TIME_LIMIT=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null || echo unknown)
export N450_TIME_LIMIT
GPU_NAME=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo none)

echo "=============================================================================="
echo "OGLANet-SIB TARGET-FDA retrain worker (1 GPU)"
echo "  run_id        ${RUN_ID}"
echo "  role          ${ROLE}"
echo "  dump_only     ${DUMP_ONLY-0}"
echo "  cell_dir      ${CELL_DIR}"
echo "  job id        ${SLURM_JOB_ID:-none}"
echo "  node          ${SLURMD_NODENAME:-none}"
echo "  partition     ${SLURM_JOB_PARTITION:-none}"
echo "  time limit    ${N450_TIME_LIMIT}"
echo "  gpu           ${GPU_NAME}"
echo "  cuda visible  ${CUDA_VISIBLE_DEVICES-unset}"
echo "  python        ${PYTHON_BIN}"
echo "  started       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================================="

if [ -f "${CELL_DIR}/.DONE" ]; then
    echo "SKIP ${RUN_ID} [${ROLE}]: .DONE exists (grid is resumable)"
    exit 0
fi

SMOKE_FLAG=""
if [ "${ROLE}" = "smoke" ]; then SMOKE_FLAG="--smoke"; fi

set +e
$PYTHON_BIN -u n450_tgtfda.py --run_id "${RUN_ID}" ${SMOKE_FLAG} ${DUMP_ONLY_FLAG}
rc=$?
set -e
echo "cell python finished rc=${rc} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "${rc}" -ne 0 ]; then exit "${rc}"; fi
echo "worker finished rc=0 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
