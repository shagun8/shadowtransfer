#!/bin/bash
# FILENAME: python/armf_crossearth_sib_smoke.sh
# Arm F / CrossEarth -- the SIB insertion-point gate. 500 iterations on cell 10
# (LOCO, Phoenix held out). Cells 10-12 carry --dependency=afterok on this job.
# Submitted only by python/armf_crossearth_grid_submit.sh.

# ---------------- NCSA Delta (default, live) ----------------
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00

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
# #SBATCH --time=00:30:00

# ---------------- Anvil (commented unless in use) ----------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=00:30:00

set -euo pipefail

CELL_DIR="${OUT_DIR:-}"
export CELL_DIR
ST_LEDGER_ARM="F"
ST_LEDGER_CELL="crossearth-sib-insertion-smoke"
ST_LEDGER_LOG="${LOG_FILE:-}"
export ST_LEDGER_ARM ST_LEDGER_CELL ST_LEDGER_LOG
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

for v in BASE_PATH REPO_ROOT ENV_PATH CKPT CKPT_SHA256 CKPT_BYTES OUT_DIR \
         CITY RES SEED ITERS BATCH_SIZE NUM_WORKERS REQUESTED_TIME; do
    if [ -z "${!v:-}" ]; then
        echo "FATAL: required export ${v} is unset" >&2
        exit 1
    fi
done

# See armf_crossearth_cell.sh for why cudatoolkit / pytorch-conda are omitted.
module purge
PYTHON_BIN="${ENV_PATH}/bin/python"
[ -x "${PYTHON_BIN}" ] || { echo "FATAL: no python at ${PYTHON_BIN}" >&2; exit 1; }
cd "${BASE_PATH}/crossearth/"

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export XFORMERS_DISABLED=1
export TOKENIZERS_PARALLELISM=false
export REQUESTED_TIME

echo "================================================================"
echo "ARM F / CrossEarth  SIB insertion-point gate"
echo "  jobid       : ${SLURM_JOB_ID:-none}"
echo "  node        : ${SLURMD_NODENAME:-none}"
echo "  gpu         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "  cell        : loco / sib / ${CITY} / ${RES}  (cell 10)"
echo "  out_dir     : ${OUT_DIR}"
echo "  iters/bs    : ${ITERS} / ${BATCH_SIZE}"
echo "  seed        : ${SEED}"
echo "  requested   : ${REQUESTED_TIME}"
echo "  started     : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" -u armf_crossearth_sib_smoke.py \
    --base_path   "${BASE_PATH}" \
    --repo_root   "${REPO_ROOT}" \
    --ckpt        "${CKPT}" \
    --ckpt_sha256 "${CKPT_SHA256}" \
    --ckpt_bytes  "${CKPT_BYTES}" \
    --out_dir     "${OUT_DIR}" \
    --city        "${CITY}" \
    --res         "${RES}" \
    --seed        "${SEED}" \
    --iters       "${ITERS}" \
    --batch_size  "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"

echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
