#!/bin/bash
# FILENAME: python/armB_png_score_midres63.sh
# Worker: score the 63 published MID-RESOLUTION LOCO cells' saved prediction
# PNG masks. CPU-only pass over stored PNG masks, GT masks and stored
# probability maps. No model, no inference, no GPU compute. Submitted on a GPU
# partition with the allocated A100 left idle, because this allocation has no
# CPU association (a GPU account only).

# --- NCSA Delta (default, live) ---------------------------------------------
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=0:55:00

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
# #SBATCH --time=0:55:00

# --- Anvil (commented unless in use) ----------------------------------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=0:55:00

set -euo pipefail

: "${BASE_PATH:?BASE_PATH not exported}"
: "${PNG_PY:?PNG_PY not exported}"
: "${PNG_OUT:?PNG_OUT not exported}"
: "${PNG_WORKERS:?PNG_WORKERS not exported}"
: "${PNG_ENV:?PNG_ENV not exported}"
: "${TIMEBOX_S:?TIMEBOX_S not exported}"

CELL_DIR="$(dirname "${PNG_OUT}")/.markers"
export CELL_DIR
ST_LEDGER_ARM="${ST_LEDGER_ARM:-C}"
ST_LEDGER_CELL="armB-png-score-midres63"
export ST_LEDGER_ARM ST_LEDGER_CELL

# shellcheck source=/dev/null
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

# --- NCSA Delta modules + env (live) ----------------------------------------
module purge
module load cudatoolkit
module load pytorch-conda/2.8
set +u
conda activate "${PNG_ENV}"
set -u
cd "${BASE_PATH}/scoring/"
PYTHON_BIN="${PNG_ENV}/bin/python"

# --- Gilbreth / Anvil (commented unless in use) -----------------------------
# module load conda; module load cuda/12.1.1; module load cudnn/9.2.0.82-12
# conda activate ${PNG_ENV}; cd "${BASE_PATH}/scoring/"
# PYTHON_BIN=${PNG_ENV}/bin/python

export PYTHONUNBUFFERED=1

echo "===== MIDRES-63 PNG-MASK SCORING (CPU, reads published artifacts) ====="
echo "driver     : ${PNG_PY}"
echo "scorer     : ${BASE_PATH}/scoring/armB_png_score.py (score_cell, unchanged)"
echo "out        : ${PNG_OUT}"
echo "dump root  : ${PNG_OUT_ROOT:-${ST_SCRATCH_ROOT}/dumps/st_r450_published_png}"
echo "workers    : ${PNG_WORKERS}"
echo "time-box   : ${TIMEBOX_S}s"
echo "only       : ${PNG_ONLY:-<all 63 midres LOCO cells>}"
echo "appendix F : ${PNG_APPENDIX_F:-<not supplied -- STEP 3 gate stays OPEN>}"
echo "jobid/node : ${SLURM_JOB_ID:-none} / ${SLURMD_NODENAME:-$(hostname)}"
echo "partition  : ${SLURM_JOB_PARTITION:-unknown}   GPU allocated and LEFT IDLE"
echo "cpus       : ${SLURM_CPUS_PER_TASK:-unknown}"
echo "started    : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "NOTE       : reads PNG masks, GT masks and stored probability maps;"
echo "             loads no checkpoint and runs no inference;"
echo "             writes only new dump dirs under st_r450_published_png/"
echo "======================================================================="

ARGS=(--workers "${PNG_WORKERS}" --out "${PNG_OUT}")
if [ -n "${PNG_ONLY-}" ]; then ARGS+=(--only "${PNG_ONLY}"); fi
if [ -n "${PNG_APPENDIX_F-}" ]; then ARGS+=(--appendix_f "${PNG_APPENDIX_F}"); fi
if [ -n "${PNG_OUT_ROOT-}" ]; then ARGS+=(--out-root "${PNG_OUT_ROOT}"); fi
if [ -n "${PNG_GATE_ONLY-}" ]; then ARGS+=(--gate-only); fi

set +e
timeout --signal=TERM --kill-after=60 "${TIMEBOX_S}" \
    "${PYTHON_BIN}" -u "${PNG_PY}" "${ARGS[@]}"
RC=$?
set -e

if [ "${RC}" -eq 124 ] || [ "${RC}" -eq 137 ]; then
    echo "FATAL time-box ${TIMEBOX_S}s expired during midres PNG scoring" >&2
    exit 124
fi
[ "${RC}" -ne 0 ] && { echo "FATAL scorer exited ${RC}" >&2; exit "${RC}"; }

mkdir -p "${CELL_DIR}"; touch "${CELL_DIR}/.DONE"
echo "MIDRES-63 PNG scoring finished cleanly at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
