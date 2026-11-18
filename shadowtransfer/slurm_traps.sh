#!/bin/bash
# FILENAME: python/shadowtransfer/slurm_traps.sh
# Sourceable ERR+EXIT traps for SLURM worker scripts.
#
# Source it near the top of a worker — BEFORE the module and conda block. Under
# `set -euo pipefail` a module or conda failure aborts the worker before the
# python call that the marker logic wraps, so an environment failure is
# otherwise indistinguishable from a cell that never ran (CLAUDE.md).
#
# Contract:
#   reads  CELL_DIR        directory the marker belongs in (falls back to DUMP_DIR)
#   reads  BASE_PATH       used to locate python/ledger.py (optional)
#   reads  ST_LEDGER_ARM / ST_LEDGER_CELL / ST_LEDGER_LOG / ST_LEDGER_NOTE
#                          optional labels prefixed onto the ledger line
#   nonzero exit -> ${CELL_DIR}/.FAILED holding exit code, failing line number
#                   and SLURM jobid, plus a FAILED ledger line
#   clean exit   -> writes nothing; .DONE stays with the existing code paths
#
# The marker is never written when ${CELL_DIR}/.DONE exists. An existing .FAILED
# is APPENDED to, never overwritten, so a worker's own message survives.

_ST_ERR_LINE="unknown"
_ST_ERR_CMD="unknown"

_st_on_err() {
    _ST_ERR_LINE="${1:-unknown}"
    _ST_ERR_CMD="${2:-unknown}"
}

_st_on_exit() {
    local rc=$?
    set +e
    trap - EXIT ERR
    if [ "${rc}" -eq 0 ]; then
        return 0
    fi

    local dir="${CELL_DIR:-${DUMP_DIR:-}}"
    local jobid="${SLURM_JOB_ID:-none}"
    local when
    when="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    if [ -n "${dir}" ] && [ ! -f "${dir}/.DONE" ]; then
        mkdir -p "${dir}" 2>/dev/null
        {
            echo "trap: exit_code=${rc} line=${_ST_ERR_LINE} jobid=${jobid}"
            echo "trap: command=${_ST_ERR_CMD}"
            echo "trap: script=${0} node=${SLURMD_NODENAME:-none} at ${when}"
        } >> "${dir}/.FAILED" 2>/dev/null
        echo "trap: wrote ${dir}/.FAILED (rc=${rc} line=${_ST_ERR_LINE})" >&2
    fi

    local ledger="${BASE_PATH:-}/python/ledger.py"
    if [ -n "${BASE_PATH:-}" ] && [ -f "${ledger}" ]; then
        python3 "${ledger}" add \
            --event manual --state FAILED \
            --arm   "${ST_LEDGER_ARM:-unknown}" \
            --cell  "${ST_LEDGER_CELL:-${RUN_ID:-unknown}}" \
            --jobid "${SLURM_JOB_ID:-}" \
            --script "${0}" \
            --log   "${ST_LEDGER_LOG:-}" \
            --note  "${ST_LEDGER_NOTE:+${ST_LEDGER_NOTE}; }trap: rc=${rc} line=${_ST_ERR_LINE}" >&2
    fi
    return 0
}

trap '_st_on_err "${LINENO}" "${BASH_COMMAND}"' ERR
trap '_st_on_exit' EXIT
