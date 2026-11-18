#!/bin/bash
# FILENAME: python/armJ_sib_ablation_submit.sh
# Driver for the SIB component-ablation PNG scoring at K=2 (Table 3 / App J).
#
#   --task selftest|smoke|grid|gate      (default smoke)
#   --only all|<run_id,...>|<row>|<arch> (default all)
#   --smoke N                            images per cell in the smoke task
#   --workers N                          process pool size
#   --time HH:MM:SS                      SBATCH --time
#   --timebox SEC                        python-side time box, < wall limit
#   --appendix-j PATH                    printed Appendix J values for the gate
#   --interactive                        use gpuA100x4-interactive
#   --dep JOBID                          sbatch --dependency=afterok:JOBID
#   --dry-run                            print the sbatch command, submit nothing
#
# Every sbatch goes through here so a ledger line is always written; the ledger
# call is skipped on --dry-run (CLAUDE.md).
set -euo pipefail

BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH="${PROJECT_ROOT}"   # second cluster
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
SCRATCH="${ST_SCRATCH_ROOT}"
LOG_PATH="${SCRATCH}/logs"
RUN_DIR="${SCRATCH}/runs/sib_ablation_k2"
INVENTORY="${SCRATCH}/runs/sib_ablation_inventory/inventory.json"
OUT_ROOT="${SCRATCH}/dumps/st_sib_ablation_png"
ST_DISK_GATE="${PYTHON_DIR}/shadowtransfer/disk_gate.py"
WORKER="${PYTHON_DIR}/armJ_sib_ablation_worker.sh"

TASK="smoke"; ONLY="all"; SMOKE=5; WORKERS=8
TIME_LIMIT="0:25:00"; TIMEBOX=1200; APPENDIX_J=""; APPENDIX_J_SHA256=""
PARTITION="gpuA100x4"; DRY=0; DEP=""; SRUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --task)        TASK="$2"; shift 2 ;;
    --only)        ONLY="$2"; shift 2 ;;
    --smoke)       SMOKE="$2"; shift 2 ;;
    --workers)     WORKERS="$2"; shift 2 ;;
    --time)        TIME_LIMIT="$2"; shift 2 ;;
    --timebox)     TIMEBOX="$2"; shift 2 ;;
    --appendix-j)  APPENDIX_J="$2"; shift 2 ;;
    --appendix-j-sha256) APPENDIX_J_SHA256="$2"; shift 2 ;;
    # --srun runs the work NOW inside one interactive allocation
    # instead of queueing a batch job (operator ruling 2026-11-05:
    # "interactive only, no batch queue").  The ledger line is written
    # by the worker itself, from inside the allocation, because srun
    # does not hand the job id back to the driver before it runs.
    --srun)        SRUN=1; PARTITION="gpuA100x4-interactive"; shift ;;
    --interactive) PARTITION="gpuA100x4-interactive"; shift ;;
    --dep)         DEP="$2"; shift 2 ;;
    --dry-run)     DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# A smoke run is TRUNCATED to --smoke images per cell. It must never land in
# the real dump root: a 5-row parquet carrying a .DONE marker is
# indistinguishable from a finished cell to anything that scans for markers.
if [ "${TASK}" = "smoke" ]; then
  OUT_ROOT="${SCRATCH}/dumps/st_sib_ablation_png_smoke"
fi

mkdir -p "${LOG_PATH}" "${RUN_DIR}"
JOB_NAME="armJ-sibabl-${TASK}"
[ "${TASK}" = "grid" ] && [ "${ONLY}" != "all" ] && JOB_NAME="armJ-sibabl-onecell"
LOG_FILE="${LOG_PATH}/${JOB_NAME}.out"
RESULTS="${RUN_DIR}/${JOB_NAME#armJ-sibabl-}_results.json"
[ "${TASK}" = "gate" ] && RESULTS="${RUN_DIR}/provenance_gate.json"
CELL_DIR="${RUN_DIR}/${TASK}_marker"

test -f "${WORKER}"    || { echo "FATAL worker missing: ${WORKER}"; exit 1; }
test -f "${INVENTORY}" || { echo "FATAL inventory missing: ${INVENTORY}"; exit 1; }
if [ -n "${APPENDIX_J}" ]; then
  test -f "${APPENDIX_J}" || { echo "FATAL --appendix-j not a file: ${APPENDIX_J}"; exit 1; }
fi

# ---- disk gate: Delta's own quota tool, the only authoritative source ----
echo "--- disk gate ---"
python3 "${ST_DISK_GATE}" || { echo "FATAL disk gate refused"; exit 1; }

# ---- driver/worker variable contract, checked mechanically, dry-run too ----
EXPORTS="ALL,BASE_PATH=${BASE_PATH},RUN_TASK=${TASK},OUT_ROOT=${OUT_ROOT},RESULTS=${RESULTS},WORKERS=${WORKERS},ONLY=${ONLY},SMOKE=${SMOKE},TIMEBOX=${TIMEBOX},INVENTORY=${INVENTORY},CELL_DIR=${CELL_DIR},APPENDIX_J=${APPENDIX_J},APPENDIX_J_SHA256=${APPENDIX_J_SHA256}"

echo "--- worker_contract selftest ---"
python3 "${PYTHON_DIR}/shadowtransfer/worker_contract.py" --selftest \
  || { echo "FATAL worker_contract selftest failed"; exit 1; }

echo "--- worker_contract check ---"
python3 "${PYTHON_DIR}/shadowtransfer/worker_contract.py" \
    --worker "${WORKER}" \
    --py "${PYTHON_DIR}/armJ_sib_ablation_score.py" \
    --py "${PYTHON_DIR}/armJ_sib_ablation_provenance_gate.py" \
    --py "${PYTHON_DIR}/armJ_sib_ablation_delta_gate.py" \
    --base "${BASE_PATH}" \
    --exports "${EXPORTS}" \
    --label "armJ-${TASK}" \
  || { echo "FATAL driver/worker contract violated -- refusing this cell"; exit 1; }

NOTE="ARM J SIB component ablations, PNG scoring at K=2 (Table 3 / Appendix J). task=${TASK} only=${ONLY} smoke=${SMOKE} workers=${WORKERS} timebox=${TIMEBOX}s. Scores the SAVED prediction PNG masks of the ablation cells with metrics.py via armJ_png_only.perimage_row_png_only; CONTRACT section 3 probability columns NULL (operator decision 2, these cells have no stored probability maps); K=5 printed gate replaced by the provenance gate (operator decision 3). Loads no checkpoint, runs no inference, modifies no existing dump, writes only under ${OUT_ROOT}. CPU-only on ${PARTITION} with the A100 allocated and idle (no CPU association on this allocation)."

if [ "${SRUN}" -eq 1 ]; then
  echo "--- srun (ONE interactive allocation, nothing queued) ---"
  echo "srun --partition=${PARTITION} --time=${TIME_LIMIT} ${WORKER}"
  echo "log:      ${LOG_FILE}"
  echo "results:  ${RESULTS}"
  echo "out_root: ${OUT_ROOT}"
  if [ "${DRY}" -eq 1 ]; then
    echo "DRY RUN -- nothing run, no ledger line written"
    exit 0
  fi
  # shellcheck disable=SC2086
  srun --account=${SLURM_ACCOUNT} \
       --partition="${PARTITION}" \
       --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=48G \
       --time="${TIME_LIMIT}" \
       --job-name="${JOB_NAME}" \
       --export="${EXPORTS}" \
       bash "${WORKER}" 2>&1 | tee "${LOG_FILE}"
  RC=${PIPESTATUS[0]}
  echo "srun exit ${RC}"
  exit "${RC}"
fi

echo "--- sbatch ---"
echo "sbatch --job-name=${JOB_NAME} --partition=${PARTITION} --time=${TIME_LIMIT}"
echo "       --output=${LOG_FILE}"
echo "       --export=${EXPORTS}"
echo "       ${WORKER}"
echo "log:      ${LOG_FILE}"
echo "results:  ${RESULTS}"
echo "out_root: ${OUT_ROOT}"

if [ "${DRY}" -eq 1 ]; then
  echo "DRY RUN -- nothing submitted, no ledger line written"
  exit 0
fi

DEP_ARG=()
if [ -n "${DEP}" ]; then
  DEP_ARG=(--dependency="afterok:${DEP}")
  echo "dependency: afterok:${DEP}"
fi

JID=$(sbatch --parsable \
      "${DEP_ARG[@]+"${DEP_ARG[@]}"}" \
      --job-name="${JOB_NAME}" \
      --partition="${PARTITION}" \
      --time="${TIME_LIMIT}" \
      --output="${LOG_FILE}" \
      --export="${EXPORTS}" \
      "${WORKER}")
echo "SUBMITTED jobid=${JID}"

python3 "${PYTHON_DIR}/ledger.py" add \
    --event submit --state SUBMITTED \
    --arm J --cell "sib-ablation-k2-${TASK}" \
    --jobid "${JID}" \
    --script "./python/armJ_sib_ablation_submit.sh" \
    --log "${LOG_FILE}" \
    --note "${NOTE} requested ${TIME_LIMIT}"
echo "ledger line written for ${JID}"
