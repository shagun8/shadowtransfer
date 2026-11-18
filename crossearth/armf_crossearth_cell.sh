#!/bin/bash
# FILENAME: python/armf_crossearth_cell.sh
# Arm F / CrossEarth -- worker for ONE of the 12 cells.
# Answers v1UJ-W3, AC1 (+ AC3, v1UJ-W2 on the fda and sib cells).
# Submitted only by python/armf_crossearth_grid_submit.sh, which writes the
# ledger line. Never sbatch this by hand (CLAUDE.md).

# ---------------- NCSA Delta (default, live) ----------------
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:45:00

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
# #SBATCH --time=02:45:00

# ---------------- Anvil (commented unless in use) ----------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=02:45:00

set -euo pipefail

# Traps FIRST, before any environment work, so an env failure still leaves a
# .FAILED marker and a FAILED ledger line (CLAUDE.md).
CELL_DIR="${DUMP_DIR:-}"
export CELL_DIR
ST_LEDGER_ARM="F"
ST_LEDGER_CELL="${RUN_ID:-unknown}"
ST_LEDGER_LOG="${LOG_FILE:-}"
export ST_LEDGER_ARM ST_LEDGER_CELL ST_LEDGER_LOG
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

for v in BASE_PATH REPO_ROOT ENV_PATH CKPT CKPT_SHA256 CKPT_BYTES RUN_ID \
         DUMP_DIR CELL_TYPE METHOD CITY RES SEED ITERS BATCH_SIZE \
         VAL_INTERVAL NUM_WORKERS REQUESTED_TIME; do
    if [ -z "${!v:-}" ]; then
        echo "FATAL: required export ${v} is unset" >&2
        exit 1
    fi
done

# ---------------- modules + env ----------------
# Deliberate deviation from the CLAUDE.md Delta block, carried over from
# stage 0: this arm does NOT use pytorch-conda/2.8 or envs/satmae_cuda12.
# CrossEarth needs mmcv 2.1.0, whose only prebuilt wheel is cu118/torch2.1, so
# it lives in envs/crossearth_mm. That torch ships its own CUDA runtime in
# nvidia/* wheels, so `module load cudatoolkit` is intentionally omitted: it
# would put a second libcudart on the path. The env is invoked by absolute
# path; nothing in the mm stack reads a conda activation variable.
module purge
PYTHON_BIN="${ENV_PATH}/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "FATAL: no python at ${PYTHON_BIN}" >&2; exit 1; }
cd "${BASE_PATH}/crossearth/"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg          # no plotting, ever, inside a worker
export XFORMERS_DISABLED=1     # DINOv2 layers take their plain-attention path
export TOKENIZERS_PARALLELISM=false
export REQUESTED_TIME

echo "================================================================"
echo "ARM F / CrossEarth  cell"
echo "  jobid       : ${SLURM_JOB_ID:-none}"
echo "  node        : ${SLURMD_NODENAME:-none}"
echo "  gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  run_id      : ${RUN_ID}"
echo "  cell        : ${CELL_TYPE} / ${METHOD} / ${CITY} / ${RES}"
echo "  base_path   : ${BASE_PATH}"
echo "  repo_root   : ${REPO_ROOT}"
echo "  env_path    : ${ENV_PATH}"
echo "  python      : ${PYTHON_BIN}"
echo "  ckpt        : ${CKPT}"
echo "  ckpt_sha256 : ${CKPT_SHA256}"
echo "  ckpt_bytes  : ${CKPT_BYTES}"
echo "  dump_dir    : ${DUMP_DIR}"
echo "  iters/bs/vi : ${ITERS} / ${BATCH_SIZE} / ${VAL_INTERVAL}"
echo "  seed        : ${SEED}"
echo "  requested   : ${REQUESTED_TIME}"
echo "  started     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

mkdir -p "${DUMP_DIR}"

"${PYTHON_BIN}" -u armf_crossearth_train.py \
    --base_path    "${BASE_PATH}" \
    --repo_root    "${REPO_ROOT}" \
    --ckpt         "${CKPT}" \
    --ckpt_sha256  "${CKPT_SHA256}" \
    --ckpt_bytes   "${CKPT_BYTES}" \
    --cell_type    "${CELL_TYPE}" \
    --method       "${METHOD}" \
    --city         "${CITY}" \
    --res          "${RES}" \
    --seed         "${SEED}" \
    --iters        "${ITERS}" \
    --batch_size   "${BATCH_SIZE}" \
    --val_interval "${VAL_INTERVAL}" \
    --num_workers  "${NUM_WORKERS}"

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
