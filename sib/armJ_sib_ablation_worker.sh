#!/bin/bash
# FILENAME: python/armJ_sib_ablation_worker.sh
# SIB component-ablation PNG scoring at K=2 (Table 3 / Appendix J), plus the
# provenance gate. CPU-only work; the A100 is allocated and left idle because
# this allocation has no CPU association (CLAUDE.md).
#
# Loads no checkpoint, runs no inference, modifies no existing dump.
#
# ---------------------------------------------------------------------------
# NCSA Delta (ACTIVE)
# ---------------------------------------------------------------------------
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=3:59:59
#
# ---------------------------------------------------------------------------
# Gilbreth (commented unless in use)
# ---------------------------------------------------------------------------
# #SBATCH -A sukkusur
# #SBATCH --partition=a100-40gb
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=3:59:59
#
# ---------------------------------------------------------------------------
# Anvil (commented unless in use)
# ---------------------------------------------------------------------------
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=48G
# #SBATCH --time=3:59:59

set -euo pipefail

# Traps FIRST, before modules and conda: an environment failure must still
# leave a .FAILED marker and a FAILED ledger line (CLAUDE.md).
: "${BASE_PATH:?BASE_PATH must be exported}"
source "${BASE_PATH}/shadowtransfer/slurm_traps.sh"

: "${RUN_TASK:?}"
: "${OUT_ROOT:?}"
: "${RESULTS:?}"
: "${WORKERS:?}"
: "${ONLY:?}"
: "${SMOKE:?}"
: "${TIMEBOX:?}"
: "${INVENTORY:?}"
: "${CELL_DIR:?}"
APPENDIX_J="${APPENDIX_J-}"
# Optional, chain/gate branch only: asserted BEFORE the file is read.
APPENDIX_J_SHA256="${APPENDIX_J_SHA256-}"

export PYTHONUNBUFFERED=1

echo "=============================================================="
echo " ARM J -- SIB component ablations, PNG scoring at K=2"
echo " jobid        : ${SLURM_JOB_ID:-none}"
echo " node         : ${SLURMD_NODENAME:-none}"
echo " partition    : ${SLURM_JOB_PARTITION:-none}"
echo " gpu          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo none) (ALLOCATED AND IDLE -- CPU-only job)"
echo " task         : ${RUN_TASK}"
echo " out_root     : ${OUT_ROOT}"
echo " results      : ${RESULTS}"
echo " workers      : ${WORKERS}"
echo " only         : ${ONLY}"
echo " smoke        : ${SMOKE}"
echo " timebox      : ${TIMEBOX}s"
echo " inventory    : ${INVENTORY}"
echo " cell_dir     : ${CELL_DIR}"
echo " appendix_j   : ${APPENDIX_J:-<none; provenance gate stays OPEN>}"
echo " base_path    : ${BASE_PATH}"
echo "=============================================================="

module purge
module load cudatoolkit
module load pytorch-conda/2.8
set +u
conda activate "${BASE_PATH}/envs/satmae_cuda12"
set -u
cd "${BASE_PATH}/sib/"
PYTHON_BIN="${BASE_PATH}/envs/satmae_cuda12/bin/python"
echo "python: $(${PYTHON_BIN} -V 2>&1)"

case "${RUN_TASK}" in
  selftest)
    echo "--- STAGE 1a: assembler selftest against armc_dump.perimage_row ---"
    ${PYTHON_BIN} -u "${BASE_PATH}/sib/armJ_sib_ablation_score.py" \
        --selftest --inventory "${INVENTORY}"
    ;;
  smoke)
    echo "--- STAGE 1b: smoke, ${SMOKE} images per cell ---"
    ${PYTHON_BIN} -u "${BASE_PATH}/sib/armJ_sib_ablation_score.py" \
        --inventory "${INVENTORY}" --out_root "${OUT_ROOT}" \
        --results "${RESULTS}" --workers "${WORKERS}" --only "${ONLY}" \
        --smoke "${SMOKE}" --timebox "${TIMEBOX}"
    ;;
  grid)
    echo "--- scoring: full test list, ${WORKERS} workers ---"
    ${PYTHON_BIN} -u "${BASE_PATH}/sib/armJ_sib_ablation_score.py" \
        --inventory "${INVENTORY}" --out_root "${OUT_ROOT}" \
        --results "${RESULTS}" --workers "${WORKERS}" --only "${ONLY}" \
        --smoke 0 --timebox "${TIMEBOX}"
    ;;
  chain)
    # ONE interactive allocation, stages back to back, operator ruling
    # 2026-11-05 item 3 ("interactive only, no batch queue").  Stage order and
    # the stage-1a schema assert are preserved; any stage failing aborts the
    # chain because the whole script runs under `set -euo pipefail`.
    RDIR="$(dirname "${RESULTS}")"
    mkdir -p "${RDIR}"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
      python3 "${BASE_PATH}/shadowtransfer/ledger.py" add \
          --jobid "${SLURM_JOB_ID}" --arm J \
          --cell "sib-ablation-k2-chain" \
          --script "${BASE_PATH}/sib/armJ_sib_ablation_submit.sh" \
          --log "${ST_SCRATCH_ROOT}/logs/armJ-sibabl-chain.out" \
          --note "ARM J SIB ablations, srun INTERACTIVE chain: gates + stages 1a/1b/2/3 back to back in one allocation (operator ruling: no batch queue). CPU-only on ${SLURM_JOB_PARTITION:-?} with the A100 allocated and idle." \
          || echo "WARN ledger add failed (chain continues)"
    fi

    echo "--- STAGE 0a: provenance gate, LEVELS vs v2's own main table ---"
    ${PYTHON_BIN} -u \
      "${BASE_PATH}/python/armJ_sib_ablation_provenance_gate.py" \
      --inventory "${INVENTORY}" --out "${RDIR}/provenance_gate.json"

    echo "--- STAGE 0b: DELTA gate, Deltas + stars vs printed Appendix J ---"
    test -n "${APPENDIX_J}" || { echo "FATAL chain needs APPENDIX_J"; exit 1; }
    test -f "${APPENDIX_J}" || { echo "FATAL APPENDIX_J not a file: ${APPENDIX_J}"; exit 1; }
    if [ -n "${APPENDIX_J_SHA256}" ]; then
      ${PYTHON_BIN} -u \
        "${BASE_PATH}/python/armJ_sib_ablation_delta_gate.py" \
        --inventory "${INVENTORY}" --out "${RDIR}/delta_gate.json" \
        --appendix-j "${APPENDIX_J}" \
        --appendix-j-sha256 "${APPENDIX_J_SHA256}"
    else
      ${PYTHON_BIN} -u \
        "${BASE_PATH}/python/armJ_sib_ablation_delta_gate.py" \
        --inventory "${INVENTORY}" --out "${RDIR}/delta_gate.json" \
        --appendix-j "${APPENDIX_J}"
    fi

    echo "--- STAGE 1a: assembler selftest against armc_dump.perimage_row ---"
    ${PYTHON_BIN} -u "${BASE_PATH}/sib/armJ_sib_ablation_score.py" \
        --selftest --inventory "${INVENTORY}"

    echo "--- STAGE 1b: smoke, ${SMOKE} images per cell, ISOLATED out root ---"
    ${PYTHON_BIN} -u "${BASE_PATH}/sib/armJ_sib_ablation_score.py" \
        --inventory "${INVENTORY}" --out_root "${OUT_ROOT}_smoke" \
        --results "${RDIR}/chain_smoke_results.json" \
        --workers "${WORKERS}" --only all \
        --smoke "${SMOKE}" --timebox "${TIMEBOX}"

    echo "--- STAGE 2: one real cell, full 150 images, ISOLATED out root ---"
    S2_CELL="mamnet-siba1-loco-r900-chicago-hr-s0"
    S2_T0=$(date +%s)
    ${PYTHON_BIN} -u "${BASE_PATH}/sib/armJ_sib_ablation_score.py" \
        --inventory "${INVENTORY}" --out_root "${OUT_ROOT}_stage2" \
        --results "${RDIR}/chain_stage2_results.json" \
        --workers 1 --only "${S2_CELL}" --smoke 0 --timebox "${TIMEBOX}"
    S2_WALL=$(( $(date +%s) - S2_T0 ))
    echo "STAGE 2 measured wall clock: ${S2_WALL}s for 1 cell"

    echo "--- STAGE 3: the full grid, ${WORKERS} workers ---"
    S3_T0=$(date +%s)
    ${PYTHON_BIN} -u "${BASE_PATH}/sib/armJ_sib_ablation_score.py" \
        --inventory "${INVENTORY}" --out_root "${OUT_ROOT}" \
        --results "${RESULTS}" --workers "${WORKERS}" --only all \
        --smoke 0 --timebox "${TIMEBOX}"
    echo "STAGE 3 measured wall clock: $(( $(date +%s) - S3_T0 ))s"

    echo "--- determinism: stage-2 cell rescored as part of stage 3 ---"
    A="${OUT_ROOT}_stage2/${S2_CELL}/perimage.parquet"
    B="${OUT_ROOT}/${S2_CELL}/perimage.parquet"
    if [ -f "${A}" ] && [ -f "${B}" ]; then
      HA=$(sha256sum "${A}" | awk '{print $1}')
      HB=$(sha256sum "${B}" | awk '{print $1}')
      echo "  stage2 ${HA}"
      echo "  stage3 ${HB}"
      if [ "${HA}" = "${HB}" ]; then
        echo "  DETERMINISM PASS -- byte-identical across two processes"
      else
        echo "  DETERMINISM FAIL -- parquet differs across processes"
      fi
    else
      echo "  DETERMINISM n/a -- one side missing"
    fi
    ;;
  gate)
    if [ -z "${APPENDIX_J}" ]; then
      echo "NOTE: APPENDIX_J empty -- the provenance gate will report OPEN."
      ${PYTHON_BIN} -u \
        "${BASE_PATH}/python/armJ_sib_ablation_provenance_gate.py" \
        --inventory "${INVENTORY}" --out "${RESULTS}"
    else
      test -f "${APPENDIX_J}" || { echo "FATAL APPENDIX_J not a file: ${APPENDIX_J}"; exit 1; }
      ${PYTHON_BIN} -u \
        "${BASE_PATH}/python/armJ_sib_ablation_provenance_gate.py" \
        --inventory "${INVENTORY}" --out "${RESULTS}" \
        --appendix-j "${APPENDIX_J}"
    fi
    ;;
  *)
    echo "FATAL unknown RUN_TASK: ${RUN_TASK}" >&2
    exit 1
    ;;
esac

echo "WORKER OK  task=${RUN_TASK}"
