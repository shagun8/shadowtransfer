#!/bin/bash
# FILENAME: armc_probes.sh
# Arm C — §4.2-4.4 diagnostic probes. Runs ONE probe cell. Submitted by
# armc_probes_submit.sh, never run by hand.
#
# Required exports (all probes):   PROBE BASE_PATH
# Probes 1-3:                      ARCH RUNG CITY SEED
# Probe 3 additionally:            EXPERIMENT (A|B)
# Probe 4 instead:                 RUN_ID ARCH MODE RUNG HOLDOUT SEED
# Optional:                        OUT_ROOT FORCE DRY_RUN ARMC_TIME_LIMIT
#                                  SMOKE_TEST SMOKE_EPOCHS SMOKE_PERM
#                                  CELLS_PER_IMAGE N_PERMUTATIONS TIMEOUT_S
#
# Probe 4 is a real ladder cell at a fresh seed. It is trained by armc_train.py
# with no probe-specific argument at all, so "identical recipe to the ladder"
# is literal; armc_probes.py then only annotates the collapse flag onto the
# finished run.json.

# ---- SBATCH: NCSA Delta (live) ----
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1:15:00

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
# #SBATCH --time=1:15:00

# ---- SBATCH: Anvil (commented unless in use) ----
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=1:15:00

set -euo pipefail

# ---- ERR/EXIT traps, sourced BEFORE the module and conda block so an
# environment failure still leaves a .FAILED marker and a FAILED ledger line.
PROBE="${PROBE:-unknown}"
if [ "${PROBE}" = "4" ]; then
    CELL_ID="${RUN_ID:-unknown}"
elif [ "${PROBE}" = "5" ]; then
    # Probe 2's raw-pixel calibration reference. One run, no model, no arch.
    CELL_ID="rawpixel-none-probe-r0-allcities-hr-s0-p2rawpixel"
else
    _sfx="p${PROBE}"
    case "${PROBE}" in
        1) _sfx="p1histmatch" ;;
        2) _sfx="p2cityprobe" ;;
        3) _sfx="p3exp${EXPERIMENT:-}" ;;
    esac
    CELL_ID="${ARCH:-unknown}-vanilla-loco-r${RUNG:-0}-${CITY:-unknown}-hr-s${SEED:-0}-${_sfx}"
fi
ST_LEDGER_ARM="C"
ST_LEDGER_CELL="${CELL_ID}"
ST_LEDGER_NOTE="probe=${PROBE}"
CELL_DIR="${OUT_ROOT:-${BASE_PATH:-}/dumps}/st/${CELL_ID}"
if [ -n "${BASE_PATH:-}" ] && [ -f "${BASE_PATH}/shadowtransfer/slurm_traps.sh" ]; then
    source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"
fi

# `set -u` is dropped across the module and conda block and restored past all
# three cluster blocks: Delta's conda deactivate hook and Lmod's shell
# functions both read unbound variables (job 20540575 died on it in 9 s).
set +u

# ---- modules + env: NCSA Delta (live) ----
# cudatoolkit must load BEFORE pytorch-conda/2.8; omitting it kills the job.
module purge
module load cudatoolkit
module load pytorch-conda/2.8
conda activate ${BASE_PATH}/envs/satmae_cuda12
cd "${BASE_PATH}/probes/"
PYTHON_BIN=${BASE_PATH}/envs/satmae_cuda12/bin/python

# ---- modules + env: Gilbreth (commented unless in use) ----
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate ${BASE_PATH}/conda_envs/satmae_cuda12
# cd "${BASE_PATH}/probes/"
# PYTHON_BIN=${BASE_PATH}/conda_envs/satmae_cuda12/bin/python

# ---- modules + env: Anvil (commented unless in use) ----
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate ${CONDA_ENV_PATH}
# cd "${BASE_PATH}/probes/"
# PYTHON_BIN=${CONDA_ENV_PATH}/bin/python

set -u

export PYTHONUNBUFFERED=1
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# ---- every required export is validated; an unset one is a hard exit ----
REQUIRED="PROBE BASE_PATH"
case "${PROBE}" in
    1|2) REQUIRED="${REQUIRED} ARCH RUNG CITY SEED" ;;
    3)   REQUIRED="${REQUIRED} ARCH RUNG CITY SEED EXPERIMENT" ;;
    4)   REQUIRED="${REQUIRED} RUN_ID ARCH MODE RUNG HOLDOUT SEED" ;;
    5)   REQUIRED="${REQUIRED}" ;;
    *)   echo "FATAL PROBE must be 1, 2, 3, 4 or 5; got '${PROBE}'"; exit 1 ;;
esac
for v in ${REQUIRED}; do
    if [ -z "${!v:-}" ]; then
        echo "FATAL required export ${v} is unset for probe ${PROBE}"
        exit 1
    fi
done

echo "=============================================================================="
echo "ARM C probe worker"
echo "  probe         ${PROBE}"
echo "  cell          ${CELL_ID}"
echo "  arch          ${ARCH:-none (no model)}"
echo "  rung / city   ${RUNG:-none} / ${CITY:-${HOLDOUT:-none}}"
echo "  seed          ${SEED:-0}"
echo "  experiment    ${EXPERIMENT:-n/a}"
echo "  job id        ${SLURM_JOB_ID:-none}"
echo "  node          ${SLURMD_NODENAME:-none}"
echo "  partition     ${SLURM_JOB_PARTITION:-none}"
echo "  time limit    ${ARMC_TIME_LIMIT:-unset}"
echo "  gpu           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo none)"
echo "  python        ${PYTHON_BIN}"
echo "  dump dir      ${CELL_DIR}"
echo "  started       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=============================================================================="

# Two flag sets, because probe 4 drives armc_train.py, whose parser has never
# heard of --smoke_perm, --cells_per_image or --n_permutations. Passing a probe
# flag to the ladder trainer is an argparse error, not a warning.
EXTRA=""
[ -n "${OUT_ROOT:-}" ]        && EXTRA="${EXTRA} --out_root ${OUT_ROOT}"
[ -n "${DRY_RUN:-}" ]         && EXTRA="${EXTRA} --dry_run"
[ -n "${FORCE:-}" ]           && EXTRA="${EXTRA} --force"
[ -n "${SMOKE_TEST:-}" ]      && EXTRA="${EXTRA} --smoke_test ${SMOKE_TEST}"
[ -n "${SMOKE_EPOCHS:-}" ]    && EXTRA="${EXTRA} --smoke_epochs ${SMOKE_EPOCHS}"
[ -n "${SMOKE_PERM:-}" ]      && EXTRA="${EXTRA} --smoke_perm ${SMOKE_PERM}"
[ -n "${CELLS_PER_IMAGE:-}" ] && EXTRA="${EXTRA} --cells_per_image ${CELLS_PER_IMAGE}"
[ -n "${N_PERMUTATIONS:-}" ]  && EXTRA="${EXTRA} --n_permutations ${N_PERMUTATIONS}"
[ -n "${TIMEOUT_S:-}" ]       && EXTRA="${EXTRA} --timeout_s ${TIMEOUT_S}"

# Probe 5 has no encoder and no epochs, so --smoke_epochs and
# --cells_per_image are not in its parser.
REF_EXTRA=""
[ -n "${OUT_ROOT:-}" ]       && REF_EXTRA="${REF_EXTRA} --out_root ${OUT_ROOT}"
[ -n "${DRY_RUN:-}" ]        && REF_EXTRA="${REF_EXTRA} --dry_run"
[ -n "${FORCE:-}" ]          && REF_EXTRA="${REF_EXTRA} --force"
[ -n "${SMOKE_TEST:-}" ]     && REF_EXTRA="${REF_EXTRA} --smoke_test ${SMOKE_TEST}"
[ -n "${SMOKE_PERM:-}" ]     && REF_EXTRA="${REF_EXTRA} --smoke_perm ${SMOKE_PERM}"
[ -n "${N_PERMUTATIONS:-}" ] && REF_EXTRA="${REF_EXTRA} --n_permutations ${N_PERMUTATIONS}"
[ -n "${TIMEOUT_S:-}" ]      && REF_EXTRA="${REF_EXTRA} --timeout_s ${TIMEOUT_S}"

TRAIN_EXTRA=""
[ -n "${OUT_ROOT:-}" ]     && TRAIN_EXTRA="${TRAIN_EXTRA} --out_root ${OUT_ROOT}"
[ -n "${DRY_RUN:-}" ]      && TRAIN_EXTRA="${TRAIN_EXTRA} --dry_run"
[ -n "${SMOKE_TEST:-}" ]   && TRAIN_EXTRA="${TRAIN_EXTRA} --smoke_test ${SMOKE_TEST}"
[ -n "${SMOKE_EPOCHS:-}" ] && TRAIN_EXTRA="${TRAIN_EXTRA} --smoke_epochs ${SMOKE_EPOCHS}"
[ -n "${TIMEOUT_S:-}" ]    && TRAIN_EXTRA="${TRAIN_EXTRA} --timeout_s ${TIMEOUT_S}"

set +e
if [ "${PROBE}" = "4" ]; then
    # Stage 1: the ladder's own trainer, no probe-specific flag.
    $PYTHON_BIN -u armc_train.py \
        --run_id "${RUN_ID}" --arch "${ARCH}" --mode "${MODE}" \
        --rung "${RUNG}" --holdout "${HOLDOUT}" --res hr --seed "${SEED}" \
        --force ${TRAIN_EXTRA}
    RC=$?
    # Stage 2: annotate the collapse flag onto the finished run.json.
    if [ ${RC} -eq 0 ] && [ -z "${DRY_RUN:-}" ]; then
        $PYTHON_BIN -u armc_probes.py --probe 4 \
            --annotate_run_id "${RUN_ID}" ${OUT_ROOT:+--out_root ${OUT_ROOT}}
        RC=$?
    fi
elif [ "${PROBE}" = "5" ]; then
    # No checkpoint, no encoder, no GPU work: the GPU this job holds sits idle
    # because the allocation has no CPU account (CLAUDE.md). Recorded as
    # compute_note in run.json rather than hidden.
    $PYTHON_BIN -u armc_probes_rawpixel.py ${REF_EXTRA}
    RC=$?
else
    $PYTHON_BIN -u armc_probes.py \
        --probe "${PROBE}" --arch "${ARCH}" --rung "${RUNG}" \
        --city "${CITY}" --seed "${SEED}" \
        ${EXPERIMENT:+--experiment ${EXPERIMENT}} \
        ${EXTRA}
    RC=$?
fi
set -e

if [ ${RC} -ne 0 ]; then
    mkdir -p "${CELL_DIR}"
    # APPEND, never truncate: armc_runtime.mark_failed may already have named
    # the phase the cell died in, and that field is what makes a failure
    # triageable across a 63-cell grid without opening 63 logs.
    echo "armc probe ${PROBE} exited ${RC} in job ${SLURM_JOB_ID:-none} at $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "${CELL_DIR}/.FAILED"
    echo "FAILED rc=${RC}; marker at ${CELL_DIR}/.FAILED"
    exit ${RC}
fi

echo "finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
