#!/bin/bash
# FILENAME: python/arme12_retrain_worker.sh
# E-12 REMEDIATION -- one MAMNet SIB C4clean HR cell per job.
#
# TASK=retrain : train with the FDA target pool pinned to the held-out city's
#                train/images, then emit the CONTRACT §3 dump.
# TASK=repro   : no training. Load the PUBLISHED best_model.pth, run test
#                inference, reproduce that cell's own stored test_results.json,
#                and dump it so the old-vs-new comparison is per-image.
#
# The published run directories are READ-ONLY. This worker never writes into
# them: checkpoints go to CKPT_ROOT and dumps to OUT_ROOT, both on /scratch.
#
# NO RESUME. The published train_mamnet_sib.py has no resume path and writes no
# periodic checkpoints -- only best_model.pth, rewritten on each improvement. A
# wall-clock kill therefore loses the cell's training. The mitigation is the
# declared --time (02:30:00 against a measured 70 min), not a resume chain.

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
: "${RUN_ID:?RUN_ID unset}"
: "${CITY:?CITY unset}"
: "${TASK:?TASK unset}"
: "${SEED:?SEED unset}"
: "${CELL_DIR:?CELL_DIR unset}"
: "${CKPT_ROOT:?CKPT_ROOT unset}"
: "${OUT_ROOT:?OUT_ROOT unset}"

case "${CITY}" in
    phoenix|miami|chicago) ;;
    *) echo "FATAL unknown CITY=${CITY}; E-12 covers the 3 published LOCO folds" >&2; exit 1 ;;
esac
case "${TASK}" in
    retrain|repro) ;;
    *) echo "FATAL unknown TASK=${TASK}" >&2; exit 1 ;;
esac

mkdir -p "${CELL_DIR}" "${CKPT_ROOT}"
export ST_LEDGER_ARM="E12"
export ST_LEDGER_CELL="${ST_LEDGER_CELL:-${RUN_ID}}"
export ST_LEDGER_NOTE="${ST_LEDGER_NOTE:-E-12 remediation: clean FDA target pool}"
# Trap FIRST, before modules/conda: an environment failure must still leave a
# .FAILED marker and a FAILED ledger line (CLAUDE.md).
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

if [ -f "${CELL_DIR}/.DONE" ]; then
    echo "[${RUN_ID}] .DONE exists at ${CELL_DIR} -- nothing to do"
    exit 0
fi
rm -f "${CELL_DIR}/.FAILED"

export ST_COMPUTE_NOTE="trained on gpuA100x4 with 1 A100 in use; there is no CPU association on this allocation (a GPU account only), so every E-12 job -- GPU-bound or not -- runs on this partition"
export ARMA_TIME_LIMIT="${ARMA_TIME_LIMIT:-unset}"

TIMEOUT_S="${TIMEOUT_S:-3600}"
NW_ARG=""
[ -n "${NUM_WORKERS:-}" ] && NW_ARG="--num_workers ${NUM_WORKERS}"
SMOKE_ARGS=""
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_ARGS="--smoke_epochs ${SMOKE_EPOCHS:-2} --smoke_test ${SMOKE_TEST:-5}"
fi

# The published cell whose artifacts must NOT be touched. Recorded and
# checksummed before and after so an accidental write is caught, not assumed
# away.
PUB_DIR="${BASE_PATH}/data/mamnet/outputs/mamnet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_${CITY}__highres"
[ -d "${PUB_DIR}" ] || { echo "FATAL published dir missing: ${PUB_DIR}" >&2; exit 1; }
PUB_BEFORE="$(cd "${PUB_DIR}" && ls -la --time-style=+%s | sha256sum | cut -c1-16)"

echo "==================== E-12 REMEDIATION ===================="
echo "run_id        : ${RUN_ID}"
echo "task          : ${TASK}"
echo "city (holdout): ${CITY}   seed ${SEED}"
echo "jobid         : ${SLURM_JOB_ID:-none}"
echo "node          : ${SLURMD_NODENAME:-none}"
echo "partition     : ${SLURM_JOB_PARTITION:-none}"
echo "cpus          : ${SLURM_CPUS_PER_TASK:-none}"
echo "time limit    : ${ARMA_TIME_LIMIT}"
echo "started       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "BASE_PATH     : ${BASE_PATH}"
echo "CKPT_ROOT     : ${CKPT_ROOT}   (never the published dir)"
echo "OUT_ROOT      : ${OUT_ROOT}"
echo "CELL_DIR      : ${CELL_DIR}"
echo "published dir : ${PUB_DIR}  (READ-ONLY; listing sha ${PUB_BEFORE})"
echo "git_hash      : ${ARMA_GIT_HASH:-unset}"
echo "compute_note  : ${ST_COMPUTE_NOTE}"
echo "smoke         : ${SMOKE:-0}"
echo "=========================================================="

# --- NCSA Delta modules + env (live) ---
module purge
module load cudatoolkit          # required before pytorch-conda/2.8
module load pytorch-conda/2.8
set +u                            # conda's activation hook reads an unbound var
conda activate ${BASE_PATH}/envs/satmae_cuda12
set -u
PYTHON_BIN=${BASE_PATH}/envs/satmae_cuda12/bin/python

# --- Gilbreth modules + env (commented unless in use) ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# set +u
# conda activate ${BASE_PATH}/conda_envs/satmae_cuda12
# set -u
# PYTHON_BIN=${BASE_PATH}/conda_envs/satmae_cuda12/bin/python

# --- Anvil modules + env (commented unless in use) ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# set +u
# conda activate ${CONDA_ENV_PATH}
# set -u
# PYTHON_BIN=${CONDA_ENV_PATH}/bin/python

cd "${BASE_PATH}/python"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

${PYTHON_BIN} -u "${BASE_PATH}/sib/arme12_retrain.py" \
    --city "${CITY}" \
    --task "${TASK}" \
    --run_id "${RUN_ID}" \
    --ckpt_root "${CKPT_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --seed "${SEED}" \
    --timeout_s "${TIMEOUT_S}" \
    ${NW_ARG} ${SMOKE_ARGS}

PUB_AFTER="$(cd "${PUB_DIR}" && ls -la --time-style=+%s | sha256sum | cut -c1-16)"
if [ "${PUB_BEFORE}" != "${PUB_AFTER}" ]; then
    echo "FATAL the published run dir CHANGED during this job: ${PUB_BEFORE} -> ${PUB_AFTER}" >&2
    echo "      ${PUB_DIR}" >&2
    exit 1
fi
echo "published dir unchanged (listing sha ${PUB_AFTER})"
echo "[${RUN_ID}] worker finished cleanly at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
