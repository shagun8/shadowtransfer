#!/bin/bash
# FILENAME: python/armf_crossearth_stage0.sh
# Arm F / CrossEarth -- stage 0 worker (environment and load gate).
# Answers v1UJ-W3, AC1.  target_access: source_only.
# Submitted only by python/armf_crossearth_submit.sh, which writes the ledger line.

# ---------------- NCSA Delta (default, live) ----------------
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:40:00

# ---------------- Gilbreth (commented unless in use) ----------------
# #SBATCH -A sukkusur
# #SBATCH --partition=a100-40gb
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=00:40:00

# ---------------- Anvil (commented unless in use) ----------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=00:40:00

set -euo pipefail

# Traps FIRST, before any module or conda work, so an environment failure
# still leaves a .FAILED marker and a FAILED ledger line (CLAUDE.md).
CELL_DIR="${OUT_DIR:-}"
export CELL_DIR
ST_LEDGER_ARM="F"
ST_LEDGER_CELL="crossearth-stage0-env-gate"
ST_LEDGER_LOG="${LOG_FILE:-}"
export ST_LEDGER_ARM ST_LEDGER_CELL ST_LEDGER_LOG
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

for v in BASE_PATH REPO_ROOT ENV_PATH CKPT CKPT_SHA256 CKPT_BYTES OUT_DIR \
         HOLDOUT RES SEED ITERS BATCH_SIZE PRECISION; do
    if [ -z "${!v:-}" ]; then
        echo "FATAL: required export ${v} is unset" >&2
        exit 1
    fi
done

# ---------------- modules + env ----------------
# NOTE, deliberate deviation from the CLAUDE.md Delta block: this arm does NOT
# use pytorch-conda/2.8 or envs/satmae_cuda12.  CrossEarth needs mmcv 2.1.0,
# whose only prebuilt wheel is cu118/torch2.1, so it lives in its own env
# (envs/crossearth_mm) built by scripts/build_crossearth_env.sh.  That torch
# ships its own CUDA runtime in nvidia/* wheels, so `module load cudatoolkit`
# is intentionally omitted -- it would put a second libcudart on the path.
# The env is invoked by absolute path rather than `conda activate`; nothing in
# the mm stack reads a conda activation variable.
module purge
PYTHON_BIN="${ENV_PATH}/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "FATAL: no python at ${PYTHON_BIN}" >&2; exit 1; }
cd "${BASE_PATH}/crossearth/"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg          # no plotting, ever, inside a worker
export XFORMERS_DISABLED=1     # DINOv2 layers fall back to plain attention
export TOKENIZERS_PARALLELISM=false

echo "================================================================"
echo "ARM F / CrossEarth  stage 0  environment + load gate"
echo "  jobid       : ${SLURM_JOB_ID:-none}"
echo "  node        : ${SLURMD_NODENAME:-none}"
echo "  gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  base_path   : ${BASE_PATH}"
echo "  repo_root   : ${REPO_ROOT}"
echo "  env_path    : ${ENV_PATH}"
echo "  python      : ${PYTHON_BIN}"
echo "  ckpt        : ${CKPT}"
echo "  ckpt_sha256 : ${CKPT_SHA256}"
echo "  ckpt_bytes  : ${CKPT_BYTES}"
echo "  out_dir     : ${OUT_DIR}"
echo "  holdout/res : ${HOLDOUT} / ${RES}"
echo "  iters/bs/pr : ${ITERS} / ${BATCH_SIZE} / ${PRECISION}"
echo "  seed        : ${SEED}"
echo "  started     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" -u armf_crossearth_stage0.py \
    --base_path   "${BASE_PATH}" \
    --repo_root   "${REPO_ROOT}" \
    --ckpt        "${CKPT}" \
    --ckpt_sha256 "${CKPT_SHA256}" \
    --ckpt_bytes  "${CKPT_BYTES}" \
    --out_dir     "${OUT_DIR}" \
    --holdout     "${HOLDOUT}" \
    --res         "${RES}" \
    --iters       "${ITERS}" \
    --batch_size  "${BATCH_SIZE}" \
    --precision   "${PRECISION}" \
    --seed        "${SEED}"

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
