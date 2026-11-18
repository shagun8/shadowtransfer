#!/bin/bash
# FILENAME: armd_e200.sh
# Arm D / D_S3_e200 worker — reruns ONE of the seven ceiling cells at
# epochs_max 200 and writes its dump under {run_id}-e200. Submitted by
# armd_e200_submit.sh, never run by hand.
#
# Required exports: RUN_ID ARCH CONDITION HOLDOUT SEED BASE_PATH
# Optional exports: SMOKE_EPOCHS SMOKE_TEST OUT_ROOT DRY_RUN CHECK_ONLY FORCE
#                   ARMD_TIME_LIMIT EPOCHS_MAX
#
# RUN_ID carries the -e200 suffix. The 100-epoch cells are never touched.

# ---- SBATCH: NCSA Delta (live) ----
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:30:00

# ---- SBATCH: Gilbreth (commented unless in use) ----
# #SBATCH -A sukkusur
# #SBATCH --partition=a100-40gb
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=4:30:00

# ---- SBATCH: Anvil (commented unless in use) ----
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=4:30:00

set -euo pipefail

# ERR/EXIT traps sourced BEFORE the module and conda block so an environment
# failure still leaves a .FAILED marker and a FAILED ledger line (CLAUDE.md).
ST_LEDGER_ARM="D"
ST_LEDGER_CELL="${RUN_ID:-unknown}"
ST_LEDGER_LOG=""
CELL_DIR="${OUT_ROOT:-${BASE_PATH:-}/dumps}/st/${RUN_ID:-unknown}"
if [ -n "${BASE_PATH:-}" ] && [ -f "${BASE_PATH}/shadowtransfer/slurm_traps.sh" ]; then
    source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"
fi

# `set -u` is dropped across the module and conda block, then restored: Delta's
# conda deactivate hook and Lmod's shell functions are not set -u clean.
set +u

# ---- modules + env: NCSA Delta (live) ----
# cudatoolkit MUST load before pytorch-conda/2.8 or the modulefile refuses.
module purge
module load cudatoolkit
module load pytorch-conda/2.8
conda activate ${BASE_PATH}/envs/satmae_cuda12
cd "${BASE_PATH}/appendix_h/"
PYTHON_BIN=${BASE_PATH}/envs/satmae_cuda12/bin/python

# ---- modules + env: Gilbreth (commented unless in use) ----
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${BASE_PATH}/conda_envs/satmae_cuda12
# cd "${BASE_PATH}/appendix_h/"
# PYTHON_BIN=${BASE_PATH}/conda_envs/satmae_cuda12/bin/python

# ---- modules + env: Anvil (commented unless in use) ----
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${CONDA_ENV_PATH}
# cd "${BASE_PATH}/appendix_h/"
# PYTHON_BIN=${CONDA_ENV_PATH}/bin/python

set -u

export PYTHONUNBUFFERED=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

for v in RUN_ID ARCH CONDITION HOLDOUT SEED BASE_PATH; do
    if [ -z "${!v:-}" ]; then
        echo "FATAL required export ${v} is unset"
        exit 1
    fi
done

DUMP_DIR="${OUT_ROOT:-${BASE_PATH}/dumps}/st/${RUN_ID}"

echo "=============================================================================="
echo "ARM D / D_S3_e200 worker — convergence control, epochs_max 100 -> 200"
echo "  run_id        ${RUN_ID}"
echo "  arch          ${ARCH}"
echo "  condition     ${CONDITION}"
echo "  holdout       ${HOLDOUT}"
echo "  rung          450 (fixed)"
echo "  seed          ${SEED}"
echo "  epochs_max    ${EPOCHS_MAX:-200}"
echo "  job id        ${SLURM_JOB_ID:-none}"
echo "  node          ${SLURMD_NODENAME:-none}"
echo "  partition     ${SLURM_JOB_PARTITION:-none}"
echo "  time limit    ${ARMD_TIME_LIMIT:-unset}"
echo "  gpu           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "  python        ${PYTHON_BIN}"
echo "  dump dir      ${DUMP_DIR}"
echo "  started       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================================="

EXTRA=""
[ -n "${SMOKE_EPOCHS:-}" ] && EXTRA="${EXTRA} --smoke_epochs ${SMOKE_EPOCHS}"
[ -n "${SMOKE_TEST:-}" ]   && EXTRA="${EXTRA} --smoke_test ${SMOKE_TEST}"
[ -n "${OUT_ROOT:-}" ]     && EXTRA="${EXTRA} --out_root ${OUT_ROOT}"
[ -n "${DRY_RUN:-}" ]      && EXTRA="${EXTRA} --dry_run"
[ -n "${CHECK_ONLY:-}" ]   && EXTRA="${EXTRA} --check_only"
[ -n "${FORCE:-}" ]        && EXTRA="${EXTRA} --force"
[ -n "${EPOCHS_MAX:-}" ]   && EXTRA="${EXTRA} --epochs_max ${EPOCHS_MAX}"

set +e
$PYTHON_BIN -u armd_e200.py \
    --run_id    "${RUN_ID}" \
    --arch      "${ARCH}" \
    --condition "${CONDITION}" \
    --holdout   "${HOLDOUT}" \
    --res       hr \
    --seed      "${SEED}" \
    --aug_strength S3 \
    ${EXTRA}
RC=$?
set -e

if [ ${RC} -ne 0 ]; then
    mkdir -p "${DUMP_DIR}"
    # APPEND, never truncate: armc_runtime.mark_failed may already have written
    # a marker naming the phase the cell died in.
    echo "armd_e200.py exited ${RC} in job ${SLURM_JOB_ID:-none} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "${DUMP_DIR}/.FAILED"
    echo "FAILED rc=${RC}; marker at ${DUMP_DIR}/.FAILED"
    exit ${RC}
fi

echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
