#!/bin/bash
# FILENAME: armc_probes_submit.sh
# Arm C — the §4.2-4.4 diagnostic probes on the ladder models. One sbatch job
# per probe cell. Answers AC1 and zNBu-W3 (REVIEWS.md, arm C ladder constraint).
#
#   ./armc_probes_submit.sh smoke     stage 1: every probe truncated, to scratch
#   ./armc_probes_submit.sh pilot     stage 2: one real cell of each probe
#   ./armc_probes_submit.sh grid      stage 3: the remaining 63 cells
#   ./armc_probes_submit.sh manifest  write the manifest, submit nothing
#   ... --dry-run                     print the sbatch commands, submit nothing
#
# Shape: 67 runs total.
#   probe 1  27  3 arch x 3 rungs x 3 cities        0:30:00 each
#   probe 2  18  3 arch x rungs {225,900} x 3 city  0:45:00 each
#   probe 3  18  3 arch x 3 cities x {A,B}, r900    0:45:00 each
#   probe 4   4  named collapsed cells at seed 3    1:15:00 each
# ~16 GPU-hours. The grid is resumable: a cell carrying .DONE is skipped.

set -euo pipefail

# ---- NCSA Delta (live) ----
BASE_PATH="${PROJECT_ROOT}"
# BASE_PATH="${PROJECT_ROOT}"   # second cluster
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
DUMP_PATH="${BASE_PATH}/dumps"
LOG_PATH="${BASE_PATH}/logs"
RUNS_TSV="${BASE_PATH}/runs/RUNS.tsv"
WORKER="${PYTHON_DIR}/armc_probes.sh"
MANIFEST="${LOG_PATH}/armC_probes_manifest.json"
SMOKE_ROOT="${SMOKE_ROOT:-${ST_SCRATCH_ROOT}/dumps_armC_probes_smoke}"
mkdir -p "${LOG_PATH}" "${BASE_PATH}/runs"

TIME_P1="0:30:00"
TIME_P2="0:45:00"
TIME_P3="0:45:00"
TIME_P4="1:15:00"
TIME_SMOKE="0:25:00"

ARCHS="${ARCHS:-mamnet oglanet dinov3}"
CITIES="${CITIES:-chicago miami phoenix}"
SEED="${SEED:-0}"

# Stage-2 pilot cells, fixed by the arm spec.
PILOT_ARCH="mamnet"; PILOT_RUNG="900"; PILOT_CITY="chicago"; PILOT_EXP="A"
PILOT_P4="oglanet-none-upper-r225-phoenix-hr-s3"

# Probe 4 — the four cells the ladder produced a collapsed model for, re-run at
# seed 3 with the ladder recipe untouched (patience 20 flat, cap 100, best
# retained). Format: run_id|arch|mode|rung|holdout|seed
P4_CELLS=(
  "dinov3-none-upper-r225-phoenix-hr-s3|dinov3|upper|225|phoenix|3"
  "dinov3-vanilla-loco-r225-miami-hr-s3|dinov3|loco|225|miami|3"
  "dinov3-none-upper-r225-miami-hr-s3|dinov3|upper|225|miami|3"
  "oglanet-none-upper-r225-phoenix-hr-s3|oglanet|upper|225|phoenix|3"
)

STAGE="${1:-}"; shift || true
DRY=""; for a in "$@"; do [ "${a}" = "--dry-run" ] && DRY="1"; done

n_submitted=0
n_skipped=0

# ---------------------------------------------------------------------------
# One sbatch per cell. Every submission writes a ledger line and a RUNS.tsv
# row; a job with no ledger line is a job nobody can account for later.
# ---------------------------------------------------------------------------
submit_cell () {
    local cell_id="$1"; local tlimit="$2"; local note="$3"; shift 3
    local dump_root="${OUT_ROOT:-${DUMP_PATH}}"
    local log="${LOG_PATH}/${cell_id}.out"

    if [ -f "${dump_root}/st/${cell_id}/.DONE" ] && [ -z "${FORCE:-}" ]; then
        echo "  skip (.DONE)  ${cell_id}"
        n_skipped=$((n_skipped + 1))
        return 0
    fi

    if [ -n "${DRY}" ]; then
        echo "  DRY  sbatch --time=${tlimit} -J ${cell_id} -o ${log} \\"
        echo "         --export=ALL,BASE_PATH=${BASE_PATH},ARMC_TIME_LIMIT=${tlimit},$* ${WORKER}"
        n_submitted=$((n_submitted + 1))
        return 0
    fi

    local sb_out
    sb_out="$(sbatch --job-name="${cell_id}" \
                     --output="${log}" \
                     --time="${tlimit}" \
                     --export="ALL,BASE_PATH=${BASE_PATH},ARMC_TIME_LIMIT=${tlimit},$*" \
                     "${WORKER}")"
    echo "  ${sb_out}   ${cell_id}"
    local jobid
    jobid="$(echo "${sb_out}" | grep -oE '[0-9]+' | tail -1 || true)"

    python3 "${PYTHON_DIR}/ledger.py" add --arm C --cell "${cell_id}" \
        --jobid "${jobid}" --script "${PYTHON_DIR}/armc_probes_submit.sh" \
        --log "${log}" --note "${note}" \
        || echo "  ledger: append failed (non-fatal)"

    if [ ! -s "${RUNS_TSV}" ]; then
        printf 'ts\tarm\tstage\tcell\tjobid\ttime_limit\tscript\tlog\tnote\n' > "${RUNS_TSV}"
    fi
    printf '%s\tC\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${STAGE}" "${cell_id}" "${jobid}" \
        "${tlimit}" "${PYTHON_DIR}/armc_probes_submit.sh" "${log}" "${note}" \
        >> "${RUNS_TSV}"

    n_submitted=$((n_submitted + 1))
}

# Probe 2's raw-pixel calibration reference: one run, no model, no arch, no
# rung. CPU-only work on gpuA100x4 with the GPU idle -- this allocation has no
# CPU account (CLAUDE.md), and run.json records that as compute_note.
RAWPIXEL_ID="rawpixel-none-probe-r0-allcities-hr-s0-p2rawpixel"
TIME_P5="0:20:00"

submit_rawpixel () {
    submit_cell "${RAWPIXEL_ID}" "${TIME_P5}" \
        "probe=2 raw-pixel calibration reference, no model; CPU work on gpuA100x4 with the GPU idle stage=${STAGE}" \
        "PROBE=5${OUT_ROOT:+,OUT_ROOT=${OUT_ROOT}}${SMOKE_TEST:+,SMOKE_TEST=${SMOKE_TEST}}${SMOKE_PERM:+,SMOKE_PERM=${SMOKE_PERM}}${FORCE:+,FORCE=1}"
}

probe_cell_id () {   # probe arch rung city seed [experiment]
    case "$1" in
        1) echo "$2-vanilla-loco-r$3-$4-hr-s$5-p1histmatch" ;;
        2) echo "$2-vanilla-loco-r$3-$4-hr-s$5-p2cityprobe" ;;
        3) echo "$2-vanilla-loco-r$3-$4-hr-s$5-p3exp$6" ;;
    esac
}

submit_probe () {    # probe arch rung city seed [experiment]
    local probe="$1" arch="$2" rung="$3" city="$4" seed="$5" exp="${6:-}"
    local cell_id; cell_id="$(probe_cell_id "$@")"
    local tlimit; case "${probe}" in 1) tlimit="${TIME_P1}";; 2) tlimit="${TIME_P2}";; 3) tlimit="${TIME_P3}";; esac
    [ -n "${SMOKE:-}" ] && tlimit="${TIME_SMOKE}"
    submit_cell "${cell_id}" "${tlimit}" \
        "probe=${probe} rung=${rung} arch=${arch} city=${city}${exp:+ exp=${exp}} stage=${STAGE}" \
        "PROBE=${probe},ARCH=${arch},RUNG=${rung},CITY=${city},SEED=${seed}${exp:+,EXPERIMENT=${exp}}${OUT_ROOT:+,OUT_ROOT=${OUT_ROOT}}${SMOKE_TEST:+,SMOKE_TEST=${SMOKE_TEST}}${SMOKE_EPOCHS:+,SMOKE_EPOCHS=${SMOKE_EPOCHS}}${SMOKE_PERM:+,SMOKE_PERM=${SMOKE_PERM}}${FORCE:+,FORCE=1}"
}

submit_p4 () {       # "run_id|arch|mode|rung|holdout|seed"
    IFS='|' read -r rid arch mode rung hold seed <<< "$1"
    local tlimit="${TIME_P4}"
    [ -n "${SMOKE:-}" ] && tlimit="${TIME_SMOKE}"
    submit_cell "${rid}" "${tlimit}" \
        "probe=4 collapsed-cell re-run at seed ${seed}, ladder recipe unchanged stage=${STAGE}" \
        "PROBE=4,RUN_ID=${rid},ARCH=${arch},MODE=${mode},RUNG=${rung},HOLDOUT=${hold},SEED=${seed}${OUT_ROOT:+,OUT_ROOT=${OUT_ROOT}}${SMOKE_TEST:+,SMOKE_TEST=${SMOKE_TEST}}${SMOKE_EPOCHS:+,SMOKE_EPOCHS=${SMOKE_EPOCHS}}${FORCE:+,FORCE=1}"
}

# ---------------------------------------------------------------------------
# Manifest — every intended run_id, written BEFORE submission so a missing
# cell is detectable without guessing what was supposed to run.
# ---------------------------------------------------------------------------
write_manifest () {
    python3 - "$MANIFEST" "$BASE_PATH" "$ARCHS" "$CITIES" "$SEED" << 'PYEOF'
import json, sys
manifest_path, base, archs, cities, seed = sys.argv[1:6]
archs, cities = archs.split(), cities.split()
runs = []
for a in archs:
    for r in (225, 450, 900):
        for c in cities:
            runs.append({"probe": 1, "run_id": f"{a}-vanilla-loco-r{r}-{c}-hr-s{seed}-p1histmatch",
                         "parent": f"{a}-vanilla-loco-r{r}-{c}-hr-s{seed}",
                         "comparator": f"{a}-none-upper-r{450 if r == 900 else r}-{c}-hr-s{seed}",
                         "time": "0:30:00"})
    for r in (225, 900):
        for c in cities:
            runs.append({"probe": 2, "run_id": f"{a}-vanilla-loco-r{r}-{c}-hr-s{seed}-p2cityprobe",
                         "parent": f"{a}-vanilla-loco-r{r}-{c}-hr-s{seed}",
                         "comparator": f"{a}-none-upper-r{450 if r == 900 else r}-{c}-hr-s{seed}",
                         "time": "0:45:00"})
    for c in cities:
        for e in ("A", "B"):
            runs.append({"probe": 3, "experiment": e,
                         "run_id": f"{a}-vanilla-loco-r900-{c}-hr-s{seed}-p3exp{e}",
                         "parent": f"{a}-vanilla-loco-r900-{c}-hr-s{seed}",
                         "comparator": f"{a}-none-upper-r450-{c}-hr-s{seed}",
                         "time": "0:45:00"})
for rid in ("dinov3-none-upper-r225-phoenix-hr-s3",
            "dinov3-vanilla-loco-r225-miami-hr-s3",
            "dinov3-none-upper-r225-miami-hr-s3",
            "oglanet-none-upper-r225-phoenix-hr-s3"):
    runs.append({"probe": 4, "run_id": rid, "parent": None, "time": "1:15:00"})
doc = {"arm": "C", "name": "armC_probes", "answers": ["AC1", "zNBu-W3"],
       "n_runs": len(runs),
       "by_probe": {str(p): sum(1 for r in runs if r["probe"] == p) for p in (1, 2, 3, 4)},
       "dump_root": f"{base}/dumps/st", "runs": runs}
with open(manifest_path, "w") as f:
    json.dump(doc, f, indent=2)
print(f"manifest: {len(runs)} runs -> {manifest_path}")
PYEOF
}

# ---------------------------------------------------------------------------

case "${STAGE}" in

manifest)
    write_manifest
    ;;

smoke)
    # Stage 1: every probe truncated, to scratch. Never to the real dump root:
    # a 2-epoch probe-3 dump landing where the real one goes would be read as
    # the real result.
    export SMOKE=1
    export OUT_ROOT="${SMOKE_ROOT}"
    export SMOKE_TEST=5
    export SMOKE_EPOCHS=2
    export SMOKE_PERM=3
    export FORCE=1
    mkdir -p "${OUT_ROOT}"
    write_manifest
    echo "STAGE 1 smoke -> ${OUT_ROOT}"
    submit_probe 1 "${PILOT_ARCH}" "${PILOT_RUNG}" "${PILOT_CITY}" "${SEED}"
    submit_probe 2 "${PILOT_ARCH}" "${PILOT_RUNG}" "${PILOT_CITY}" "${SEED}"
    submit_probe 3 "${PILOT_ARCH}" 900 "${PILOT_CITY}" "${SEED}" A
    submit_probe 3 "${PILOT_ARCH}" 900 "${PILOT_CITY}" "${SEED}" B
    ;;

pilot)
    # Stage 2: one real cell of each probe, nothing truncated.
    write_manifest
    echo "STAGE 2 pilot -> ${DUMP_PATH}/st"
    submit_probe 1 "${PILOT_ARCH}" "${PILOT_RUNG}" "${PILOT_CITY}" "${SEED}"
    submit_probe 2 "${PILOT_ARCH}" "${PILOT_RUNG}" "${PILOT_CITY}" "${SEED}"
    submit_probe 3 "${PILOT_ARCH}" 900 "${PILOT_CITY}" "${SEED}" "${PILOT_EXP}"
    for cell in "${P4_CELLS[@]}"; do
        [[ "${cell}" == "${PILOT_P4}|"* ]] && submit_p4 "${cell}"
    done
    ;;

rawpixel)
    # Probe 2's third reference. One job; nothing else depends on it.
    echo "raw-pixel reference -> ${DUMP_PATH}/st"
    submit_rawpixel
    ;;

grid)
    # Stage 3: everything the pilot did not cover. Cells carrying .DONE are
    # skipped, so a resubmission fills only the holes.
    write_manifest
    echo "STAGE 3 grid -> ${DUMP_PATH}/st"
    for arch in ${ARCHS}; do
        for rung in 225 450 900; do
            for city in ${CITIES}; do
                submit_probe 1 "${arch}" "${rung}" "${city}" "${SEED}"
            done
        done
        for rung in 225 900; do
            for city in ${CITIES}; do
                submit_probe 2 "${arch}" "${rung}" "${city}" "${SEED}"
            done
        done
        for city in ${CITIES}; do
            for exp in A B; do
                submit_probe 3 "${arch}" 900 "${city}" "${SEED}" "${exp}"
            done
        done
    done
    for cell in "${P4_CELLS[@]}"; do
        submit_p4 "${cell}"
    done
    ;;

*)
    echo "usage: $0 {smoke|pilot|grid|rawpixel|manifest} [--dry-run]" >&2
    exit 1
    ;;
esac

echo
echo "stage ${STAGE}: ${n_submitted} submitted, ${n_skipped} skipped (.DONE)"
echo "manifest ${MANIFEST}"
echo "logs     ${LOG_PATH}"
[ -n "${DRY}" ] && echo "DRY RUN — nothing was submitted, no ledger line written"
exit 0
