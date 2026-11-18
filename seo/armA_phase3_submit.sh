#!/bin/bash
# FILENAME: python/armA_phase3_submit.sh
# Arm A / S-EO -- PHASE 3: SIB and FDA, LOCO-ONLY. 12 cells.
#
# NO new upper bound is trained. Each fold's phase-2 vanilla upper cell is the
# comparator and is already on disk with a .DONE. Nothing is submitted for
# OGLANet, which is dropped from arm A and is rejected by name in the worker
# and absent from the trainer's --arch choices. No diagnostics.
#
# THE 12 CELLS, all seed 0, all r3500, all hr, all LOCO
#   mamnet fda  OMA JAX UCSD   unlabeled_target
#   mamnet sib  OMA JAX UCSD   unlabeled_target   (SIB on a CNN includes FDA)
#   dinov3 fda  OMA JAX UCSD   unlabeled_target
#   dinov3 sib  OMA JAX UCSD   source_only        (the ViT branch has no FDA)
#
# The code path is phase 2's. python/armA_phase3_train.py imports
# armA_pilot_train's helpers unchanged and owns only its own main(); the
# phase-1/2 file is NOT edited, so the train_script_sha recorded in the twelve
# finished vanilla cells' run.json still resolves. The method layer is
# python/shadowtransfer/armA_methods.py, which lifts every component from the
# published tree and owns only the S-EO mapping.
#
# BUDGET MANIFESTS ARE REUSED VERBATIM. Same seo_budget_acq3/{FOLD}_loco.json,
# same 3500/900 patch ids as the fold's vanilla cells, same budget_rule_sha,
# which the trainer re-checks at the top of every cell and which this script
# re-checks below.
#
# LINK SIZING
#   Vanilla measured rates, +20% headroom (phase 1/2's basis):
#     mamnet  5.1673 min/epoch -> 6.201; 4h - 6 min startup - 40 min dump
#             reserve = 194 min usable = 31 epochs/link -> 4 links to 100
#     dinov3  2.9038 min/epoch -> 3.485; same accounting = 55 epochs/link
#             -> 2 links needed
#   Method overhead is UNMEASURED at submit time -- the smokes that measure it
#   are submitted in this same call and the cells gate on them. Every chain is
#   therefore given ONE EXTRA LINK as the standing allowance: mamnet 5,
#   dinov3 4. At the 1.5x overhead the operator named as the trigger, mamnet
#   falls to 20 epochs/link (5 x 20 = 100, exactly covered) and dinov3 to 37
#   (3 needed + 1 spare). An extra link is close to free: every link after a
#   cell writes .DONE exits in seconds on the marker check.
#   The smokes' measured ratio lands in each smoke's run.json under
#   method_overhead and is reported next session; if any exceeds 1.5x the
#   allowance above already covers it, and a re-run of this script tops up a
#   chain that still needs more.
#
# SUBMISSION ORDER: dinov3 chains first, then mamnet, matching phase 2. If the
# allocation or the day runs out, the casualty is the cheaper architecture.
#
# CHECKPOINTS GO TO SCRATCH. the archive root is inside its soft-quota grace period.
# Dumps land under BASE_PATH per CONTRACT.md 2.

set -euo pipefail

# --- BASE_PATH (NCSA Delta, live) ---
BASE_PATH="${PROJECT_ROOT}"
SCRATCH_ROOT="${ST_SCRATCH_ROOT}"

# --- BASE_PATH (Gilbreth, commented unless in use) ---
# BASE_PATH="${PROJECT_ROOT}"   # second cluster

# --- BASE_PATH (Anvil, commented unless in use) ---
# BASE_PATH="${PROJECT_ROOT}"   # third cluster

PYTHON_DIR="${BASE_PATH}"
LOG_PATH="${BASE_PATH}/logs"
DUMP_ROOT="${BASE_PATH}/dumps"
SEO_RAW="${SCRATCH_ROOT}/data/seo_raw"
SPLITS_DIR="${SCRATCH_ROOT}/data/seo_splits_acq3"
BUDGET_DIR="${SCRATCH_ROOT}/data/seo_budget_acq3"
CKPT_ROOT="${SCRATCH_ROOT}/data/armA_ckpt"
SMOKE_TAG="${SMOKE_TAG:-p3a}"
SMOKE_ROOT="${SCRATCH_ROOT}/data/armA_smoke_${SMOKE_TAG}"
IMG_ROOT="${SEO_RAW}/pansharpened"
MASK_ROOT="${SEO_RAW}/shadows_max"
PATCH_INDEX="${LOG_PATH}/armA_seo_patch_index_acq3.json"
WORKER="${PYTHON_DIR}/armA_phase3_worker.sh"
LEDGER="${PYTHON_DIR}/ledger.py"
SELF="${PYTHON_DIR}/armA_phase3_submit.sh"
VANILLA_DUMP_ROOT="${DUMP_ROOT}/seo"

SEED=0
FOLDS="${FOLDS:-OMA JAX UCSD}"
METHODS="${METHODS:-fda sib}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
CELL_TIME="${CELL_TIME:-03:59:59}"
SMOKE_TIME="00:30:00"
NUM_WORKERS=8
TIMEOUT_S=3600             # post-training time-box: up to 2642 test patches
LINKS_MAMNET="${LINKS_MAMNET:-5}"
LINKS_DINOV3="${LINKS_DINOV3:-4}"
OVERHEAD_STEPS="${OVERHEAD_STEPS:-12}"
SMOKE_FOLD="${SMOKE_FOLD:-OMA}"

# The train-side DetailedEvaluator is OFF for all arm A training, under the
# code-path audit. Every phase-3 cell starts from scratch under the lever, so
# its declared deviation point is epoch 1.
DISABLE_TRAIN_EVAL="${DISABLE_TRAIN_EVAL:-1}"
TRAIN_EVAL_AUTHORITY="${TRAIN_EVAL_AUTHORITY:-code_audit}"
EQUIV_ARTIFACT="${EQUIV_ARTIFACT:-${LOG_PATH}/armA_evalfree_equivalence.json}"
EQUIV_SHA_EXPECT="${EQUIV_SHA_EXPECT:-4d030f64f444d886}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

mkdir -p "${LOG_PATH}" "${CKPT_ROOT}" "${SMOKE_ROOT}"

for p in "${PATCH_INDEX}" "${SPLITS_DIR}" "${IMG_ROOT}" "${MASK_ROOT}"; do
    [ -e "${p}" ] || { echo "FATAL missing input: ${p}" >&2; exit 1; }
done

# Budget manifests: present, assertions passed, and byte-identical to what the
# vanilla cells trained on. The last check is the one phase 3 adds -- the whole
# point of the comparison is that the method cell and the vanilla cell saw the
# same 3500/900/n_test patch ids.
echo "=== budget manifests (REUSED VERBATIM) ==="
for F in ${FOLDS}; do
    for c in loco upper; do
        [ -f "${BUDGET_DIR}/${F}_${c}.json" ] || {
            echo "FATAL missing budget manifest ${BUDGET_DIR}/${F}_${c}.json" >&2
            exit 1; }
    done
    REP="${BUDGET_DIR}/${F}_report.json"
    [ -f "${REP}" ] || { echo "FATAL missing budget report ${REP}" >&2; exit 1; }
    python3 - "${F}" "${BUDGET_DIR}" "${VANILLA_DUMP_ROOT}" <<'PYEOF' || exit 1
import hashlib, json, os, sys
F, BUD, VD = sys.argv[1], sys.argv[2], sys.argv[3]
rep = json.load(open(os.path.join(BUD, f"{F}_report.json")))
if not rep.get("assertions_pass"):
    print(f"FATAL {F} budget assertions FAILED: {rep.get('assertion_failures')}",
          file=sys.stderr)
    sys.exit(1)
p = os.path.join(BUD, f"{F}_loco.json")
raw = open(p, "rb").read()
man = json.loads(raw)
sha = hashlib.sha256(raw).hexdigest()
lc = F.lower()
lines = []
for arch in ("mamnet", "dinov3"):
    rj = os.path.join(VD, f"{arch}-vanilla-loco-r3500-{lc}-hr-s0", "run.json")
    if not os.path.exists(rj):
        lines.append(f"{arch}: vanilla loco dump ABSENT")
        continue
    v = json.load(open(rj))
    same_sha = v.get("budget_manifest_sha256")
    same_rule = v.get("budget_rule_sha") == man["budget_rule_sha"]
    lines.append(f"{arch}: budget_rule_sha {'MATCH' if same_rule else 'MISMATCH'}"
                 + ("" if same_sha is None else
                    f", manifest sha {'MATCH' if same_sha == sha else 'MISMATCH'}"))
    if not same_rule:
        print(f"FATAL {F}/{arch}: the vanilla comparator trained under a "
              f"different budget_rule_sha", file=sys.stderr)
        sys.exit(1)
print(f"  {F}: n_train {man['n_train']} n_val {man['n_val']} n_test {man['n_test']}"
      f"  rule {man['budget_rule_sha'][:16]}  loco-manifest sha {sha[:16]}")
for ln in lines:
    print(f"       vs {ln}")
PYEOF
done

if [ "${DISABLE_TRAIN_EVAL}" = "1" ]; then
    [ -f "${EQUIV_ARTIFACT}" ] || {
        echo "FATAL DISABLE_TRAIN_EVAL=1 but EQUIV_ARTIFACT is missing" >&2; exit 1; }
    GOT="$(sha256sum "${EQUIV_ARTIFACT}" | cut -c1-${#EQUIV_SHA_EXPECT})"
    [ "${GOT}" = "${EQUIV_SHA_EXPECT}" ] || {
        echo "FATAL equivalence artifact sha prefix ${GOT} != ${EQUIV_SHA_EXPECT}" >&2
        exit 1; }
    echo "train-side evaluator: DISABLED under authority '${TRAIN_EVAL_AUTHORITY}'"
    echo "  artifact sha prefix ${GOT} verified"
fi

GIT_HASH="$(git -C "${BASE_PATH}" rev-parse HEAD 2>/dev/null || echo '')"
GIT_DIRTY="$(git -C "${BASE_PATH}" status --porcelain 2>/dev/null | wc -l)"
[ -n "${GIT_HASH}" ] || { echo "FATAL cannot resolve a git hash at ${BASE_PATH}" >&2; exit 1; }
ARM_SRC="python/armA_phase3_train.py python/armA_phase3_worker.sh python/armA_phase3_submit.sh python/armA_pilot_train.py python/shadowtransfer/armA_methods.py python/shadowtransfer/seo_budget.py python/shadowtransfer/seo_dump.py python/shadowtransfer/seo_dataset.py python/shadowtransfer/seo_traineval.py python/shadowtransfer/armc_runtime.py"
ARM_DIRTY="$(git -C "${BASE_PATH}" status --porcelain -- ${ARM_SRC} 2>/dev/null | wc -l)"
if [ "${ARM_DIRTY}" -ne 0 ]; then
    echo "FATAL ${ARM_DIRTY} of this arm's own source files are uncommitted:" >&2
    git -C "${BASE_PATH}" status --porcelain -- ${ARM_SRC} >&2
    exit 1
fi
echo "git: HEAD ${GIT_HASH}; arm sources clean; ${GIT_DIRTY} other paths dirty"

COMMON="BASE_PATH=${BASE_PATH},BUDGET_DIR=${BUDGET_DIR},PATCH_INDEX=${PATCH_INDEX}"
COMMON="${COMMON},SPLITS_DIR=${SPLITS_DIR},IMG_ROOT=${IMG_ROOT},MASK_ROOT=${MASK_ROOT}"
COMMON="${COMMON},SEED=${SEED},NUM_WORKERS=${NUM_WORKERS}"
COMMON="${COMMON},TIMEOUT_S=${TIMEOUT_S},ARMA_GIT_HASH=${GIT_HASH},ARMA_GIT_DIRTY=${GIT_DIRTY}"
COMMON="${COMMON},DISABLE_TRAIN_EVAL=${DISABLE_TRAIN_EVAL}"
COMMON="${COMMON},TRAIN_EVAL_AUTHORITY=${TRAIN_EVAL_AUTHORITY}"
COMMON="${COMMON},EQUIV_ARTIFACT=${EQUIV_ARTIFACT},EQUIV_SHA_EXPECT=${EQUIV_SHA_EXPECT}"
COMMON="${COMMON},VANILLA_DUMP_ROOT=${VANILLA_DUMP_ROOT}"

LEDGER_NOTE_BASE="armA S-EO PHASE 3, SIB+FDA LOCO-ONLY (no new upper bound, no oglanet, no diagnostics); budget manifests reused verbatim, n_train=3500 n_val=900 test=held-out city FULL split; evalOFF under code_audit eq-sha ${EQUIV_SHA_EXPECT}; ckpts on scratch; trained on gpuA100x4 (no CPU association on this allocation)"

submit() {
    local name="$1" tlimit="$2" dep="$3" extra="$4" log="$5"
    local cmd=(sbatch --job-name="${name}" --output="${log}" --time="${tlimit}")
    [ -n "${dep}" ] && cmd+=(--dependency="${dep}")
    cmd+=(--export="ALL,${COMMON},ARMA_TIME_LIMIT=${tlimit},${extra}" "${WORKER}")
    if [ "${DRY}" -eq 1 ]; then
        printf 'DRY: ' >&2; printf '%q ' "${cmd[@]}" >&2; printf '\n' >&2
        echo "DRY: log -> ${log}" >&2
        echo "DRYID"; return 0
    fi
    local out jid
    out="$("${cmd[@]}")"
    jid="$(echo "${out}" | awk '{print $NF}')"
    echo "submitted ${name} jobid=${jid} time=${tlimit} dep=${dep:-none} log=${log}" >&2
    python3 "${LEDGER}" add --arm A --cell "${name}" --jobid "${jid}" \
        --script "${SELF}" --log "${log}" \
        --note "${LEDGER_NOTE_BASE}; requested ${tlimit}" >&2
    echo "${jid}"
}

declare -A CELL_JOBS
declare -A SMOKE_JOB
SKIPPED_CELLS=""

# submit_cell <arch> <method> <fold> <links> <dep>
submit_cell() {
    local arch="$1" method="$2" fold="$3" nlinks="$4" dep0="$5"
    local lc RUN_ID CELL_DIR prev jobs link LOG J
    lc="$(echo "${fold}" | tr 'A-Z' 'a-z')"
    RUN_ID="${arch}-${method}-loco-r3500-${lc}-hr-s${SEED}"
    CELL_DIR="${DUMP_ROOT}/seo/${RUN_ID}"
    mkdir -p "${CELL_DIR}"
    if [ -f "${CELL_DIR}/.DONE" ]; then
        echo "SKIP ${RUN_ID} (.DONE exists)"
        SKIPPED_CELLS="${SKIPPED_CELLS}${SKIPPED_CELLS:+ }${RUN_ID}"
        return 0
    fi
    prev="${dep0}"
    jobs=""
    for link in $(seq 1 "${nlinks}"); do
        LOG="${LOG_PATH}/armA_p3_${arch}_${method}_${lc}_link${link}.out"
        J=$(submit "armA-p3-${arch}-${method}-${lc}-L${link}" "${CELL_TIME}" "${prev}" \
            "TASK=real,ARCH=${arch},METHOD=${method},HOLDOUT=${fold},RUN_ID=${RUN_ID},CONDITION=loco,CELL_DIR=${CELL_DIR},CKPT_ROOT=${CKPT_ROOT},TRAIN_EVAL_OFF_FROM_EPOCH=1,TRAIN_EVAL_OFF_FROM_EPOCH_LABEL=1,ST_LEDGER_CELL=${RUN_ID}-L${link}" \
            "${LOG}")
        case "${J}" in
            "") ;;
            DRYID) ;;
            *) prev="afterany:${J}"; jobs="${jobs}${jobs:+ }${J}" ;;
        esac
    done
    CELL_JOBS["${RUN_ID}"]="${jobs}"
}

echo
echo "=== ARM A / S-EO PHASE 3  folds=${FOLDS}  methods=${METHODS}  seed=${SEED} ==="
echo "BASE_PATH   ${BASE_PATH}"
echo "git         ${GIT_HASH}  (arm sources clean; ${GIT_DIRTY} other paths dirty)"
echo "budget dir  ${BUDGET_DIR}   (manifests REUSED VERBATIM)"
echo "ckpt root   ${CKPT_ROOT}"
echo "dump root   ${DUMP_ROOT}/seo/<run_id>"
echo "comparator  ${VANILLA_DUMP_ROOT}/<arch>-vanilla-loco-... and -none-upper-..."
echo "            (phase 2, already .DONE; phase 3 trains NO new upper bound)"
echo "dry-run     ${DRY}"
echo

# ---- stage-1 smokes: one per (arch, method), 4 total -----------------------
# One fold is enough per (arch, method): all three folds' manifests are already
# gated by phase 2's own four smokes and by the budget-report check above, so
# what is new here is the METHOD, not the manifest. Each smoke gates only its
# own (arch, method) cells, so a dinov3/sib failure cannot hold up mamnet/fda.
# Every attempt gets its own root: a retried smoke must retrain from zero, not
# resume its predecessor and skip straight to the dump.
if [ "${SKIP_SMOKE}" -eq 0 ]; then
    lcS="$(echo "${SMOKE_FOLD}" | tr 'A-Z' 'a-z')"
    for arch in dinov3 mamnet; do
    for method in ${METHODS}; do
        SR="${arch}-${method}-loco-r3500-${lcS}-hr-s${SEED}"
        SO="${SMOKE_ROOT}/${arch}/${method}"
        SL="${LOG_PATH}/armA_p3_smoke_${arch}_${method}_${SMOKE_TAG}.out"
        SMOKE_JOB["${arch}/${method}"]=$(submit \
            "armA-p3-smoke-${arch}-${method}-${SMOKE_TAG}" "${SMOKE_TIME}" "" \
            "TASK=smoke,ARCH=${arch},METHOD=${method},HOLDOUT=${SMOKE_FOLD},RUN_ID=${SR},CONDITION=loco,CELL_DIR=${SO}/seo/${SR},CKPT_ROOT=${SO}/train,OUT_ROOT=${SO},SMOKE_EPOCHS=2,SMOKE_TRAIN=64,SMOKE_VAL=32,SMOKE_TEST=5,OVERHEAD_STEPS=${OVERHEAD_STEPS},TIMEOUT_S=900,TRAIN_EVAL_OFF_FROM_EPOCH=1,TRAIN_EVAL_OFF_FROM_EPOCH_LABEL=1,ST_LEDGER_CELL=armA-p3-smoke-${arch}-${method}-${SMOKE_TAG}" \
            "${SL}")
    done; done
else
    echo "SKIP_SMOKE=1: no stage-1 gate submitted"
    for arch in dinov3 mamnet; do for method in ${METHODS}; do
        SMOKE_JOB["${arch}/${method}"]="SKIPPED"; done; done
fi

gate_of() {
    case "${1}" in DRYID|SKIPPED|"") echo "" ;; *) echo "afterok:${1}" ;; esac
}

# ---- the cells. dinov3 FIRST, then mamnet. ---------------------------------
echo
for method in ${METHODS}; do
    G="$(gate_of "${SMOKE_JOB[dinov3/${method}]}")"
    for F in ${FOLDS}; do submit_cell dinov3 "${method}" "${F}" "${LINKS_DINOV3}" "${G}"; done
done
for method in ${METHODS}; do
    G="$(gate_of "${SMOKE_JOB[mamnet/${method}]}")"
    for F in ${FOLDS}; do submit_cell mamnet "${method}" "${F}" "${LINKS_MAMNET}" "${G}"; done
done

# ---- manifest: every intended run_id, its components and its target_access --
# Written so the audit trail LEADS the cells: which components are active and
# what target_access each cell claims is on disk before any of them trains.
MAN="${LOG_PATH}/armA_phase3_manifest.json"
python3 - "${MAN}.new" "${GIT_HASH}" "${GIT_DIRTY}" "${FOLDS}" "${METHODS}" "${BASE_PATH}" \
    "${LINKS_MAMNET}" "${LINKS_DINOV3}" "${DUMP_ROOT}" "${BUDGET_DIR}" \
    "${SPLITS_DIR}" "${CKPT_ROOT}" "${LOG_PATH}" "${CELL_TIME}" "${SMOKE_TIME}" \
    "${SMOKE_FOLD}" "${OVERHEAD_STEPS}" "${EQUIV_SHA_EXPECT}" \
    "${TRAIN_EVAL_AUTHORITY}" "$(declare -p CELL_JOBS | sed 's/^declare -A CELL_JOBS=//')" \
    "$(declare -p SMOKE_JOB | sed 's/^declare -A SMOKE_JOB=//')" \
    "${SKIPPED_CELLS}" <<'PYEOF'
import datetime, json, os, re, sys

(out, git_hash, git_dirty, folds, methods, base_path, lm, ld, dump_root,
 budget_dir, splits_dir, ckpt_root, log_path, cell_time, smoke_time,
 smoke_fold, ov_steps, eq_sha, authority, cell_jobs_raw, smoke_jobs_raw,
 skipped) = sys.argv[1:23]

sys.path.insert(0, os.path.join(base_path, "python"))
from shadowtransfer import armA_methods as M

def parse_assoc(s):
    return dict(re.findall(r'\[([^\]]+)\]="([^"]*)"', s))

cell_jobs = parse_assoc(cell_jobs_raw)
smoke_jobs = parse_assoc(smoke_jobs_raw)

cells = []
for arch in ("dinov3", "mamnet"):
    for method in methods.split():
        ta, detail = M.method_target_access(arch, method)
        if method == "sib":
            spec = M.SIB_SPEC[arch]
            active, off = spec["components"], spec["components_off"]
            cfg_src, L = spec["published_config"], spec["fda_L"]
        else:
            spec = M.FDA_SPEC[arch]
            active = ["fda_preproc"] + (["contrast_channel"] if arch == "mamnet" else [])
            off = []
            cfg_src, L = spec["published_config"], spec["fda_L"]
        for F in folds.split():
            rid = f"{arch}-{method}-loco-r3500-{F.lower()}-hr-s0"
            cells.append({
                "run_id": rid, "arch": arch, "method": method, "mode": "loco",
                "holdout": F, "rung": 3500, "res": "hr", "seed": 0,
                "target_access": ta, "target_access_detail": detail,
                "components_active": active, "components_off": off,
                "published_method_config": cfg_src,
                "fda_L": L,
                "fda_target_pool": (None if L is None else {
                    "source": f"{budget_dir}/{F}_upper.json train split",
                    "city": F, "split": "train", "max": 100,
                    "rule": ("unlabelled IMAGES only; asserted disjoint from "
                             "this cell's train, val and test id sets"),
                }),
                "links": int(lm) if arch == "mamnet" else int(ld),
                "jobs": cell_jobs.get(rid, ""),
                "dump": f"{dump_root}/seo/{rid}",
                "comparator_loco": f"{arch}-vanilla-loco-r3500-{F.lower()}-hr-s0",
                "comparator_upper": f"{arch}-none-upper-r3500-{F.lower()}-hr-s0",
            })

payload = {
    "arm": "A", "dataset": "seo",
    "kind": ("PHASE 3: SIB and FDA, LOCO-ONLY, 12 cells. No new upper bound "
             "(each fold's phase-2 vanilla upper is the comparator), no "
             "oglanet (dropped from arm A), no diagnostics."),
    "folds": folds, "methods": methods, "seed": 0,
    "n_train": 3500, "n_val": 900,
    "test": "held-out city FULL split, never subsampled",
    "budget_manifests_reused_verbatim": True,
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "git_hash": git_hash, "git_other_dirty_paths": int(git_dirty),
    "train_eval_disabled": True, "train_eval_authority": authority,
    "train_eval_equivalence_sha": eq_sha, "train_eval_disabled_from_epoch": 1,
    "recipe": ("phase-2 recipe: 100-epoch ceiling, flat patience 20, one "
               "published vanilla config per arch as the base. Method "
               "hyperparameters come from the published method configs named "
               "per cell and override nothing else."),
    "sib_variant": "C4-clean (the published SIB)",
    "sib_coverage_threshold": ("NOT applied: it is a post-hoc step over stored "
                               "probabilities, absent from both published SIB "
                               "training scripts. probs.npz is unthresholded."),
    "links_per_cell": {"mamnet": int(lm), "dinov3": int(ld)},
    "epochs_per_link_at_vanilla_rate": {"mamnet": 31, "dinov3": 55},
    "sizing_basis": ("vanilla MEASURED rates + 20% headroom (mamnet 5.1673 -> "
                     "6.201 min/epoch, dinov3 2.9038 -> 3.485; 4h link less 6 "
                     "min startup and a 40 min dump reserve = 194 min usable), "
                     "PLUS one extra link per chain as the standing allowance "
                     "for method overhead, which is unmeasured at submit time. "
                     "At 1.5x overhead mamnet is 20 epochs/link (5 x 20 = 100) "
                     "and dinov3 37 (3 needed + 1 spare)."),
    "cell_time": cell_time, "smoke_time": smoke_time,
    "smoke_fold": smoke_fold, "smoke_overhead_steps": int(ov_steps),
    "smoke_jobids": smoke_jobs, "skipped": skipped,
    "cells": cells,
    "budget_dir": budget_dir, "splits_dir": splits_dir, "ckpt_root": ckpt_root,
    "logs": f"{log_path}/armA_p3_<arch>_<method>_<fold>_link<N>.out",
}
with open(out, "w") as f:
    json.dump(payload, f, indent=1)
print(f"manifest: {len(cells)} cells")
PYEOF
python3 -c "import json;json.load(open('${MAN}.new'))"
mv -f "${MAN}.new" "${MAN}"

NCELL=0
for rid in "${!CELL_JOBS[@]}"; do NCELL=$((NCELL + 1)); done
CEIL=$(( 4 * (6 * LINKS_DINOV3 + 6 * LINKS_MAMNET) ))

echo
echo "=== job shape ==="
echo "cells submitted : ${NCELL} of 12   (skipped: ${SKIPPED_CELLS:-none})"
echo "jobs            : 4 smokes + 6 dinov3 chains x ${LINKS_DINOV3} links"
echo "                            + 6 mamnet chains x ${LINKS_MAMNET} links"
echo "per-job --time  : smoke ${SMOKE_TIME}, cells ${CELL_TIME}"
echo "GPU-hours       : CEILING ${CEIL} if every cell runs every link to the"
echo "                  wall clock. Expected ~54 at vanilla-equivalent rates"
echo "                  (the phase-1 projection's stated assumption), higher in"
echo "                  proportion to whatever the smokes measure the method"
echo "                  overhead to be. Every link after a cell writes .DONE"
echo "                  exits in seconds on the marker check."
echo "manifest        : ${MAN}"
echo "markers         : ${DUMP_ROOT}/seo/<run_id>/.DONE|.FAILED"
