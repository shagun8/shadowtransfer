#!/bin/bash
# FILENAME: armd_e200_submit.sh
# Arm D / D_S3_e200 — convergence control. Reruns, at epochs_max 200, exactly
# the seven arm D cells whose run.json records epochs_run == epochs_max, and
# recomputes the decision rule with those values swapped in.
#
#   ./armd_e200_submit.sh verify                 print the ceiling set, submit nothing
#   ./armd_e200_submit.sh smoke                  stage 1: one truncated cell
#   ./armd_e200_submit.sh cell                   one cell; RUN_ID env (base id)
#   ./armd_e200_submit.sh grid                   the seven reruns + the decision job
#   ./armd_e200_submit.sh decision               the decision recompute alone
#   ... --dry-run                                print sbatch commands, submit nothing
#
# Env: FORCE=1 re-runs a -e200 cell that already carries .DONE.
#      OUT_ROOT redirects the dump.  NO_DECISION=1 skips the chained recompute.
#      EPOCHS_MAX overrides 200 (do not; the spec says one ceiling raise only).
#
# The 100-epoch cells are NEVER touched. Every run_id here carries the -e200
# suffix, writes its own dump directory and its own checkpoint directory, so
# both versions stay readable and the effect of the ceiling is itself
# reportable.
#
# WALL CLOCK. At 100 epochs the oglanet cells ran 01:00-01:39 and the dinov3
# cells 01:00-01:37. Doubling the ceiling puts the worst case near 03:20 if
# early stopping never fires, so --time is 04:30:00. Noted for the record:
# aug-upper cells train longer than aug-LOCO ones, and mamnet's 02:05:00 limit
# in armd_train_submit.sh was set from a LOCO measurement, after which an upper
# cell used 86% of it. No mamnet cell is in this rerun set.

set -euo pipefail

# ---- NCSA Delta (live) ----
BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH="${PROJECT_ROOT}"   # second cluster
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
DUMP_PATH="${BASE_PATH}/dumps"
LOG_PATH="${BASE_PATH}/logs"
WORKER="${PYTHON_DIR}/armd_e200.sh"
DEC_WORKER="${PYTHON_DIR}/armd_e200_decision.sh"
mkdir -p "${LOG_PATH}"

TIME_SMOKE="00:20:00"
TIME_CELL="04:30:00"
TIME_DEC="00:15:00"
SUFFIX="-e200"

STAGE="${1:-}"; shift || true
DRY=""; for a in "$@"; do [ "${a}" = "--dry-run" ] && DRY="1"; done

# The seven ceiling cells, read from armd_e200.py:CEILING_CELLS so the guard in
# the trainer and the list submitted here cannot drift, then re-verified here
# against every completed cell's run.json.
CEILING_CELLS="$(python3 - <<'PYC'
import sys
sys.path.insert(0, os.environ["PROJECT_ROOT"])
import re
src = open(os.path.join(os.environ["PROJECT_ROOT"], "appendix_h/armd_e200.py")).read()
blk = re.search(r"CEILING_CELLS = \(\n(.*?)\n\)", src, re.S).group(1)
for m in re.finditer(r'"([^"]+)"', blk):
    print(m.group(1))
PYC
)"

verify_set () {
    python3 - <<'PYV'
import json, os, re, sys
BASE = os.environ["PROJECT_ROOT"]
src = open(f"{BASE}/python/armd_e200.py").read()
blk = re.search(r"CEILING_CELLS = \(\n(.*?)\n\)", src, re.S).group(1)
declared = set(re.findall(r'"([^"]+)"', blk))
summ = json.load(open(f"{BASE}/runs/D_S3/summary.json"))
observed, rows = set(), []
for rid, _ in summ["cells_patched"]:
    p = f"{BASE}/dumps/st/{rid}/run.json"
    if not os.path.exists(p):
        print(f"  MISSING run.json {rid}"); continue
    d = json.load(open(p))
    hit = d.get("epochs_run") == d.get("epochs_max")
    if hit:
        observed.add(rid)
        rows.append((rid, d.get("epochs_run"), d.get("epochs_max"),
                     d.get("best_epoch"), d.get("early_stopped")))
print(f"  completed cells scanned: {len(summ['cells_patched'])}")
print(f"  epochs_run == epochs_max: {len(observed)}")
for r in sorted(rows):
    print(f"    {r[0]:44s} epochs_run {r[1]}/{r[2]} best {r[3]} early_stopped {r[4]}")
if declared != observed:
    print("  MISMATCH between armd_e200.py:CEILING_CELLS and the completed grid")
    print(f"    declared not observed: {sorted(declared - observed)}")
    print(f"    observed not declared: {sorted(observed - declared)}")
    sys.exit(2)
print("  VERIFIED: the declared set is exactly the observed set.")
PYV
}

JOBIDS=""

submit () {
    local base_id="$1" arch="$2" condition="$3" holdout="$4" seed="$5" tlim="$6"
    local extra="${7:-}" tag="${8:-}"
    local run_id="${base_id}${SUFFIX}"
    local log="${LOG_PATH}/${run_id}${tag}.out"
    local exports="ALL,RUN_ID=${run_id},ARCH=${arch},CONDITION=${condition}"
    exports="${exports},HOLDOUT=${holdout},SEED=${seed},BASE_PATH=${BASE_PATH}"
    exports="${exports},ARMD_TIME_LIMIT=${tlim}"
    exports="${exports},EPOCHS_MAX=${EPOCHS_MAX:-200}${extra:+,${extra}}"
    if [ -n "${DRY}" ]; then
        echo "sbatch --job-name=${run_id} --time=${tlim} --output=${log} --export=${exports}"
        return
    fi
    local sb_out jobid
    sb_out="$(sbatch --job-name="${run_id}" --time="${tlim}" --output="${log}" \
                     ${DEPENDENCY:+--dependency=afterok:${DEPENDENCY}} \
                     --export="${exports}" "${WORKER}")"
    echo "${sb_out}   ${run_id}"
    jobid="$(echo "${sb_out}" | grep -oE '[0-9]+' | tail -1 || true)"
    JOBIDS="${JOBIDS:+${JOBIDS}:}${jobid}"
    python3 "${PYTHON_DIR}/ledger.py" add --arm D --cell "${run_id}" \
        --jobid "${jobid}" --script "${PYTHON_DIR}/armd_e200_submit.sh" \
        --log "${log}" \
        --note "stage=${STAGE} armD_S3_e200 convergence control; epochs_max 100->200, everything else identical to ${base_id}; rerun_of=${base_id}" \
        || echo "  ledger: append failed (non-fatal)"
}

# submit_base <base_run_id>   e.g. oglanet-augs3-upper-r450-chicago-hr-s0
submit_base () {
    local base_id="$1"
    local arch mode city seed condition run_id extra=""
    arch="$(echo "${base_id}" | cut -d- -f1)"
    mode="$(echo "${base_id}" | cut -d- -f3)"
    city="$(echo "${base_id}" | cut -d- -f5)"
    seed="$(echo "${base_id}" | cut -d- -f7 | tr -d 's')"
    condition="aug-loco"; [ "${mode}" = "upper" ] && condition="aug-upper"
    run_id="${base_id}${SUFFIX}"
    [ -n "${FORCE:-}" ] && extra="FORCE=1"
    if [ -z "${FORCE:-}" ] && [ -f "${DUMP_PATH}/st/${run_id}/.DONE" ]; then
        echo "  reused (.DONE exists, not retrained): ${run_id}"
        SKIPPED=$((SKIPPED+1)); return
    fi
    submit "${base_id}" "${arch}" "${condition}" "${city}" "${seed}" \
           "${TIME_CELL}" "${extra}"
    SUBMITTED=$((SUBMITTED+1))
}

submit_decision () {
    local dep="${1:-}"
    local log="${LOG_PATH}/armD_S3_e200_decision.out"
    if [ -n "${DRY}" ]; then
        echo "sbatch --job-name=armD-S3-e200-decision --time=${TIME_DEC} --output=${log}${dep:+ --dependency=afterany:${dep}}"
        return
    fi
    local sb_out jobid
    sb_out="$(sbatch --job-name="armD-S3-e200-decision" --time="${TIME_DEC}" \
                     --output="${log}" \
                     ${dep:+--dependency=afterany:${dep}} \
                     --export="ALL,BASE_PATH=${BASE_PATH},ALLOW_PARTIAL=1" \
                     "${DEC_WORKER}")"
    echo "${sb_out}   armD-S3-e200-decision"
    jobid="$(echo "${sb_out}" | grep -oE '[0-9]+' | tail -1 || true)"
    python3 "${PYTHON_DIR}/ledger.py" add --arm D --cell "armD_S3_e200_decision" \
        --jobid "${jobid}" --script "${PYTHON_DIR}/armd_e200_submit.sh" \
        --log "${log}" \
        --note "armD_S3_e200 decision recompute -> runs/D_S3/decision_e200.json; summary.json left untouched; CPU work on gpuA100x4 with the GPU idle (no CPU account on this allocation)" \
        || echo "  ledger: append failed (non-fatal)"
}

case "${STAGE}" in

  verify)
    echo "--- arm D ceiling set, declared vs observed ---"
    verify_set
    ;;

  smoke)
    # Stage 1: one truncated cell, 2 epochs, 5 test images, dump to scratch.
    SMOKE_ROOT="${SMOKE_ROOT:-${BASE_PATH}/dumps/_smoke_armD_e200}"
    mkdir -p "${SMOKE_ROOT}"
    S_BASE="${RUN_ID:-oglanet-augs3-upper-r450-chicago-hr-s0}"
    S_ARCH="$(echo "${S_BASE}" | cut -d- -f1)"
    S_MODE="$(echo "${S_BASE}" | cut -d- -f3)"
    S_CITY="$(echo "${S_BASE}" | cut -d- -f5)"
    S_SEED="$(echo "${S_BASE}" | cut -d- -f7 | tr -d 's')"
    S_COND="aug-loco"; [ "${S_MODE}" = "upper" ] && S_COND="aug-upper"
    submit "${S_BASE}" "${S_ARCH}" "${S_COND}" "${S_CITY}" "${S_SEED}" \
           "${TIME_SMOKE}" "SMOKE_EPOCHS=2,SMOKE_TEST=5,OUT_ROOT=${SMOKE_ROOT}" \
           ".smoke"
    ;;

  cell)
    C_BASE="${RUN_ID:?RUN_ID must be a base run_id, without the -e200 suffix}"
    if ! echo "${CEILING_CELLS}" | grep -qx "${C_BASE}"; then
        echo "FATAL ${C_BASE} is not one of the seven ceiling cells"; exit 1
    fi
    SUBMITTED=0; SKIPPED=0
    echo "SINGLE CELL — ${C_BASE}${SUFFIX}  time ${TIME_CELL}"
    submit_base "${C_BASE}"
    ;;

  grid)
    echo "--- verifying the ceiling set before submitting anything ---"
    verify_set
    MANIFEST="${LOG_PATH}/armD_S3_e200_manifest.json"
    SUBMITTED=0; SKIPPED=0; N=0; FIRST=1
    {
      echo "{"
      echo "  \"arm\": \"D\", \"arm_id\": \"D_S3\", \"arm_variant\": \"D_S3_e200\","
      echo "  \"weakness_ids\": [\"v1UJ-W4\", \"zNBu-W2\", \"AC1\"],"
      echo "  \"rung\": 450, \"res\": \"hr\", \"aug_strength\": \"S3\","
      echo "  \"epochs_max\": ${EPOCHS_MAX:-200}, \"epochs_max_original\": 100,"
      echo "  \"changed_from_grid\": \"the epoch ceiling and nothing else\","
      echo "  \"cell_set_source\": \"python/armd_e200.py:CEILING_CELLS, verified against epochs_run == epochs_max in every completed cell's run.json\","
      echo "  \"force\": \"${FORCE:-0}\", \"submitted_utc\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
      echo "  \"runs\": ["
    } > "${MANIFEST}"
    echo "--- submitting ---"
    while read -r BASE_ID; do
        [ -z "${BASE_ID}" ] && continue
        [ ${FIRST} -eq 0 ] && echo "," >> "${MANIFEST}"
        printf '    "%s%s"' "${BASE_ID}" "${SUFFIX}" >> "${MANIFEST}"
        FIRST=0; N=$((N+1))
        submit_base "${BASE_ID}"
    done <<< "${CEILING_CELLS}"
    { echo ""; echo "  ]"; echo "}"; } >> "${MANIFEST}"
    echo "manifest: ${MANIFEST}"
    echo "intended ${N}   submitted ${SUBMITTED}   reused(.DONE) ${SKIPPED}"
    if [ -z "${NO_DECISION:-}" ]; then
        echo "--- chaining the decision recompute behind ${SUBMITTED} job(s) ---"
        submit_decision "${JOBIDS}"
    fi
    ;;

  decision)
    submit_decision "${DEPENDENCY:-}"
    ;;

  *)
    echo "usage: $0 {verify|smoke|cell|grid|decision} [--dry-run]"; exit 1 ;;
esac
