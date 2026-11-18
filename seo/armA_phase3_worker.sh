#!/bin/bash
# FILENAME: python/armA_phase3_worker.sh
# Arm A / S-EO PHASE 3 -- one method cell per job. LOCO ONLY.
#
# TASK selects smoke or real; ARCH selects the architecture; METHOD selects fda
# or sib. OGLANet is dropped from arm A and is rejected here by name as well as
# in the trainer, so neither layer alone has to be the one that remembers.
# CONDITION is not an input: phase 3 trains no upper bound, so the worker
# hard-codes loco and would reject anything else.
#
# There is no CPU account on this allocation (a GPU account only).
# This worker genuinely uses the GPU, but the compute_note is still written
# into every artifact so the allocation story is on the record.
#
# Resumable: a job whose ${CELL_DIR}/.DONE exists exits 0 immediately. A cell
# is a chain of jobs with --dependency=afterany; each link resumes from the
# newest checkpoint_epoch_*.pth the previous link left behind.

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
: "${ARCH:?ARCH unset}"
: "${METHOD:?METHOD unset}"
: "${HOLDOUT:?HOLDOUT unset}"
: "${SEED:?SEED unset}"
: "${CELL_DIR:?CELL_DIR unset}"
: "${BUDGET_DIR:?BUDGET_DIR unset}"
: "${PATCH_INDEX:?PATCH_INDEX unset}"
: "${SPLITS_DIR:?SPLITS_DIR unset}"
: "${IMG_ROOT:?IMG_ROOT unset}"
: "${MASK_ROOT:?MASK_ROOT unset}"
: "${CKPT_ROOT:?CKPT_ROOT unset}"

case "${ARCH}" in
    mamnet|dinov3) ;;
    oglanet) echo "FATAL oglanet is dropped from arm A (operator decision 2026-08-02)" >&2; exit 1 ;;
    *) echo "FATAL unknown ARCH=${ARCH}" >&2; exit 1 ;;
esac
case "${METHOD}" in
    fda|sib) ;;
    vanilla|none) echo "FATAL METHOD=${METHOD} belongs to phases 1-2, not phase 3" >&2; exit 1 ;;
    *) echo "FATAL unknown METHOD=${METHOD}" >&2; exit 1 ;;
esac
if [ -n "${CONDITION:-}" ] && [ "${CONDITION}" != "loco" ]; then
    echo "FATAL phase 3 is LOCO-only; CONDITION=${CONDITION} rejected. Each" >&2
    echo "      fold's phase-2 vanilla upper cell is the comparator and is" >&2
    echo "      already on disk; no new upper bound is trained." >&2
    exit 1
fi

mkdir -p "${CELL_DIR}" "${CKPT_ROOT}"
export ST_LEDGER_ARM="A"
export ST_LEDGER_CELL="${ST_LEDGER_CELL:-${RUN_ID}}"
export ST_LEDGER_NOTE="${ST_LEDGER_NOTE:-armA S-EO phase 3}"
# Trap FIRST, before modules/conda: an environment failure must still leave a
# .FAILED marker and a FAILED ledger line (CLAUDE.md).
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

if [ -f "${CELL_DIR}/.DONE" ]; then
    echo "[${RUN_ID}] .DONE exists at ${CELL_DIR} -- nothing to do"
    exit 0
fi
rm -f "${CELL_DIR}/.FAILED"

export ST_COMPUTE_NOTE="trained on gpuA100x4 with 1 A100 in use; there is no CPU association on this allocation (a GPU account only), so every arm A job -- GPU-bound or not -- runs on this partition"
export ARMA_TIME_LIMIT="${ARMA_TIME_LIMIT:-unset}"

TASK="${TASK:-real}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
OUT_ROOT_ARG=""
[ -n "${OUT_ROOT:-}" ] && OUT_ROOT_ARG="--out_root ${OUT_ROOT}"
PROBE_ARG=""
[ -n "${PROBE_JSON:-}" ] && PROBE_ARG="--probe_json ${PROBE_JSON}"
VDUMP_ARG=""
[ -n "${VANILLA_DUMP_ROOT:-}" ] && VDUMP_ARG="--vanilla_dump_root ${VANILLA_DUMP_ROOT}"

# The train-side DetailedEvaluator lever. Never on by default. The trainer
# refuses without an authority; this layer additionally verifies the artifact
# on disk is still the one the operator's decision was taken against, BEFORE
# training starts, so a swapped artifact costs seconds and not a cell.
EVAL_ARGS=""
if [ "${DISABLE_TRAIN_EVAL:-0}" = "1" ]; then
    : "${EQUIV_ARTIFACT:?EQUIV_ARTIFACT unset while DISABLE_TRAIN_EVAL=1}"
    : "${TRAIN_EVAL_AUTHORITY:?TRAIN_EVAL_AUTHORITY unset while DISABLE_TRAIN_EVAL=1}"
    [ -f "${EQUIV_ARTIFACT}" ] || { echo "FATAL EQUIV_ARTIFACT missing: ${EQUIV_ARTIFACT}" >&2; exit 1; }
    if [ -n "${EQUIV_SHA_EXPECT:-}" ]; then
        GOT="$(sha256sum "${EQUIV_ARTIFACT}" | cut -c1-${#EQUIV_SHA_EXPECT})"
        if [ "${GOT}" != "${EQUIV_SHA_EXPECT}" ]; then
            echo "FATAL equivalence artifact sha prefix ${GOT} != expected ${EQUIV_SHA_EXPECT}" >&2
            exit 1
        fi
        echo "equivalence artifact sha prefix verified: ${GOT}"
    fi
    EVAL_ARGS="--disable_train_eval --train_eval_authority ${TRAIN_EVAL_AUTHORITY} --equiv_artifact ${EQUIV_ARTIFACT}"
    EVAL_ARGS="${EVAL_ARGS} --train_eval_off_from_epoch ${TRAIN_EVAL_OFF_FROM_EPOCH:-1}"
    EVAL_ARGS="${EVAL_ARGS} --train_eval_off_from_epoch_label ${TRAIN_EVAL_OFF_FROM_EPOCH_LABEL:-1}"
fi

# --dump_only: skip training entirely, load checkpoint_best.pth and re-run the
# inference and dump phase alone. CLAUDE.md requires every training script to
# have this mode precisely so a failed dump never costs a retrain; the
# /projects quota block of 2026-08-03 is what it is for.
DUMP_ONLY_ARG=""
if [ "${DUMP_ONLY:-0}" = "1" ]; then
    DUMP_ONLY_ARG="--dump_only"
    if [ "${TASK}" = "smoke" ]; then
        echo "FATAL DUMP_ONLY=1 with TASK=smoke is meaningless" >&2; exit 1
    fi
fi

SMOKE_ARGS=""
if [ "${TASK}" = "smoke" ]; then
    SMOKE_ARGS="--smoke_epochs ${SMOKE_EPOCHS:-2} --smoke_train ${SMOKE_TRAIN:-64} --smoke_val ${SMOKE_VAL:-32} --smoke_test ${SMOKE_TEST:-5}"
    SMOKE_ARGS="${SMOKE_ARGS} --overhead_steps ${OVERHEAD_STEPS:-12}"
fi

echo "=================== ARM A / S-EO PHASE 3 ==================="
echo "run_id        : ${RUN_ID}"
echo "arch / method : ${ARCH} / ${METHOD}"
echo "task          : ${TASK}"
echo "condition     : loco (phase 3 is LOCO-only)   holdout ${HOLDOUT}   seed ${SEED}"
echo "jobid         : ${SLURM_JOB_ID:-none}"
echo "node          : ${SLURMD_NODENAME:-none}"
echo "partition     : ${SLURM_JOB_PARTITION:-none}"
echo "cpus          : ${SLURM_CPUS_PER_TASK:-none}   num_workers ${NUM_WORKERS}"
echo "time limit    : ${ARMA_TIME_LIMIT}"
echo "started       : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "BASE_PATH     : ${BASE_PATH}"
echo "BUDGET_DIR    : ${BUDGET_DIR}"
echo "PATCH_INDEX   : ${PATCH_INDEX}"
echo "SPLITS_DIR    : ${SPLITS_DIR}"
echo "IMG_ROOT      : ${IMG_ROOT}"
echo "MASK_ROOT     : ${MASK_ROOT}"
echo "CKPT_ROOT     : ${CKPT_ROOT}"
echo "CELL_DIR      : ${CELL_DIR}"
echo "VANILLA_DUMPS : ${VANILLA_DUMP_ROOT:-default}"
echo "git_hash      : ${ARMA_GIT_HASH:-unset}"
echo "compute_note  : ${ST_COMPUTE_NOTE}"
echo "smoke args    : ${SMOKE_ARGS:-none}"
echo "dump_only     : ${DUMP_ONLY:-0}"
echo "OUT_ROOT      : ${OUT_ROOT:-<BASE_PATH>/dumps}"
echo "train-eval    : $([ "${DISABLE_TRAIN_EVAL:-0}" = "1" ] && echo "DISABLED under ${TRAIN_EVAL_AUTHORITY:-unset} (${EQUIV_ARTIFACT:-unset})" || echo "ON (arm C recipe)")"
echo "============================================================"

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

${PYTHON_BIN} -u "${BASE_PATH}/seo/armA_phase3_train.py" \
    --arch "${ARCH}" \
    --method "${METHOD}" \
    --run_id "${RUN_ID}" \
    --holdout "${HOLDOUT}" \
    --seed "${SEED}" \
    --budget_dir "${BUDGET_DIR}" \
    --patch_index "${PATCH_INDEX}" \
    --splits_dir "${SPLITS_DIR}" \
    --img_root "${IMG_ROOT}" \
    --mask_root "${MASK_ROOT}" \
    --ckpt_root "${CKPT_ROOT}" \
    --num_workers "${NUM_WORKERS}" \
    --timeout_s "${TIMEOUT_S}" \
    ${OUT_ROOT_ARG} ${PROBE_ARG} ${VDUMP_ARG} ${SMOKE_ARGS} ${DUMP_ONLY_ARG} ${EVAL_ARGS}

echo "[${RUN_ID}] worker finished cleanly at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
