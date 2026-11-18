#!/usr/bin/env python3
"""Append-only run ledger for SLURM jobs. Stdlib only.

One JSON object per line in runs/ledger.jsonl; latest line wins per jobid.
Schema: ts, event, arm, cell, jobid, script, state, log_path, note.

Subcommands:
  add     append a SUBMITTED line
  sync    batch sacct query; append a line per job whose state changed
  status  latest state per jobid, grouped by arm, with counts
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER_PATH = os.environ.get(
    "LEDGER_PATH", os.path.join(REPO_ROOT, "runs", "ledger.jsonl")
)

# Closed state enum. Anything sacct reports outside this set is recorded as
# UNKNOWN with the raw string preserved in the note.
STATES = {
    "SUBMITTED", "PENDING", "RUNNING", "COMPLETED", "FAILED",
    "CANCELLED", "TIMEOUT", "NODE_FAIL", "UNKNOWN",
}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"}


def now_iso():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def read_lines():
    """Return every ledger record in file order. Bad lines are skipped."""
    if not os.path.exists(LEDGER_PATH):
        return []
    out = []
    with open(LEDGER_PATH) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except ValueError:
                sys.stderr.write("ledger: skipping malformed line: %s\n" % raw[:80])
    return out


def append(rec):
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    with open(LEDGER_PATH, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def make_record(event, arm, cell, jobid, script, state, log_path, note):
    return {
        "ts": now_iso(),
        "event": event,
        "arm": arm or "",
        "cell": cell or "",
        "jobid": str(jobid) if jobid else None,
        "script": script or "",
        "state": state,
        "log_path": log_path or None,
        "note": note or "",
    }


def latest_by_jobid(records):
    """Latest record per jobid; file order decides. Keyed by jobid string."""
    latest = {}
    for rec in records:
        jid = rec.get("jobid")
        if jid:
            latest[str(jid)] = rec
    return latest


def normalize_state(raw):
    """sacct State -> (enum state, note). Handles 'CANCELLED by 12345'."""
    tok = (raw or "").strip().split()[0] if (raw or "").strip() else ""
    tok = tok.rstrip("+").upper()
    if tok in STATES:
        return tok, ""
    if not tok:
        return "UNKNOWN", "sacct: empty state"
    return "UNKNOWN", "sacct state: %s" % raw.strip()


def query_sacct(jobids):
    """One batch sacct call. Returns {jobid: raw_state}. -X drops .batch steps.

    Non-numeric ids are dropped before the query. `CONTRACT`-side the ledger
    schema allows `"jobid": null` for a bookkeeping line that describes no job,
    and `sacct -j null` exits 1 on "Bad job/step specified" -- which took the
    WHOLE batch down with it, so every genuinely live job in the same call came
    back with no record and was written UNKNOWN. Observed 2026-08-03: one
    bookkeeping line cost 114 arm-A jobids their true state while they were
    running. A caller-supplied id must not be able to blind the query.
    """
    jobids = [j for j in jobids if str(j).isdigit()]
    if not jobids:
        return {}
    cmd = [
        "sacct", "-j", ",".join(jobids), "-X", "--noheader",
        "--parsable2", "--format=JobID,State",
    ]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write("ledger: sacct unavailable (%s)\n" % exc)
        return {}
    if proc.returncode != 0:
        sys.stderr.write("ledger: sacct exit %d: %s\n" % (proc.returncode, proc.stderr.strip()))
        return {}
    seen = {}
    for line in proc.stdout.splitlines():
        if "|" not in line:
            continue
        jid, state = line.split("|", 1)
        jid = jid.strip().split(".")[0]  # defensive: strip .batch/.extern
        if jid:
            seen.setdefault(jid, state.strip())
    return seen


def cmd_add(args):
    rec = make_record(
        args.event, args.arm, args.cell, args.jobid, args.script,
        args.state, args.log, args.note,
    )
    append(rec)
    print("ledger: %s %s %s %s" % (
        rec["arm"], rec["cell"], rec["jobid"], rec["state"]))
    return 0


def cmd_sync(args):
    records = read_lines()
    latest = latest_by_jobid(records)
    pending = [j for j, r in latest.items() if r.get("state") not in TERMINAL]
    if not pending:
        print("ledger: nothing to sync (%d jobs, all terminal)" % len(latest))
        return 0
    raw_states = query_sacct(sorted(pending))
    changes = 0
    for jid in sorted(pending):
        prev = latest[jid]
        if jid in raw_states:
            state, note = normalize_state(raw_states[jid])
        else:
            state, note = "UNKNOWN", "sacct: no record"
        if state == prev.get("state") and note == prev.get("note", ""):
            continue
        if state == prev.get("state"):
            continue
        append(make_record(
            "sync", prev.get("arm"), prev.get("cell"), jid, prev.get("script"),
            state, prev.get("log_path"), note or prev.get("note", ""),
        ))
        print("ledger: %s %s -> %s%s" % (
            jid, prev.get("state"), state, (" (%s)" % note) if note else ""))
        changes += 1
    if not changes:
        print("ledger: no state changes (%d checked)" % len(pending))
    return 0


def cmd_status(args):
    latest = latest_by_jobid(read_lines())
    by_arm = {}
    counts = {}
    for jid, rec in latest.items():
        by_arm.setdefault(rec.get("arm") or "(no arm)", []).append((jid, rec))
        key = rec.get("state", "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    for arm in sorted(by_arm):
        rows = sorted(by_arm[arm], key=lambda kv: kv[0])
        print("== %s (%d)" % (arm, len(rows)))
        for jid, rec in rows:
            note = rec.get("note") or ""
            print("   %-10s %-12s %s%s" % (
                jid, rec.get("state"), rec.get("cell") or "",
                ("  [%s]" % note) if note else ""))
    print("-- totals: %d jobs" % len(latest))
    for state in sorted(counts):
        print("   %-12s %d" % (state, counts[state]))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("add", help="append a submit line")
    a.add_argument("--arm", required=True)
    a.add_argument("--cell", required=True)
    a.add_argument("--jobid", default="")
    a.add_argument("--script", default="")
    a.add_argument("--log", default="")
    a.add_argument("--note", default="")
    # Defaults keep `add` a submit line; slurm_traps.sh overrides both to
    # append a manual FAILED line without a second subcommand.
    a.add_argument("--state", default="SUBMITTED", choices=sorted(STATES))
    a.add_argument("--event", default="submit",
                   choices=["submit", "sync", "manual"])
    a.set_defaults(func=cmd_add)
    s = sub.add_parser("sync", help="batch sacct refresh")
    s.set_defaults(func=cmd_sync)
    t = sub.add_parser("status", help="latest state per jobid")
    t.set_defaults(func=cmd_status)
    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
