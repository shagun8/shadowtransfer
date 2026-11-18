#!/usr/bin/env python3
"""STANDING PREFLIGHT -- the driver/worker variable contract, checked both ways.

Why this exists (2026-10-17, arm A block 4). Six times a submit script and the
worker it drives have disagreed about variable names. Five of those failed
loudly. The sixth would not have: the inherited histmatch worker had no METHOD
variable, so a driver exporting METHOD=bsm would have been ignored and every
method cell would have SILENTLY PROBED A VANILLA PARENT -- a wrong number with
no error. Prompt wording and code review did not catch it reliably, so the
check is mechanical and runs on every cell a submit script builds.

THE CONTRACT, per cell:
  (a) every variable the driver EXPORTS is one the worker chain READS
      -- else the export is dead, and a value the operator believes is in
         force is not;
  (b) every variable the worker REQUIRES (`: "${VAR:?...}"`) is one the
      driver EXPORTS -- else the cell dies at start, or worse, a later
      `${VAR:-default}` papers over it.

"The worker chain" = the worker .sh, every file it `source`s (followed
recursively, `${BASE_PATH}` expanded), and the Python files it launches, whose
`os.environ.get/[]` / `os.getenv` reads also count as reads (the inherited
submit exported ARMA_GIT_DIRTY, which the worker never names but the probe
reads from the environment).

Stdlib only. Usage (a submit script calls this once per cell, dry-run too):
    python3 worker_contract.py --worker W.sh --py a.py --py b.py \
        --base BASE_PATH --exports "ALL,K1=v1,K2=v2"
    python3 worker_contract.py --selftest
Exit 0 = contract holds; 1 = violated (both lists printed); 2 = usage error.
"""
import argparse
import json
import os
import re
import sys
import tempfile

SH_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)|\$([A-Za-z_][A-Za-z0-9_]*)")
SH_REQ = re.compile(r':\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):\?')
SH_SRC = re.compile(r'^\s*(?:source|\.)\s+"?([^"\s]+)"?', re.M)
PY_ENV = re.compile(r"""os\.(?:environ\.get|getenv)\(\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]"""
                    r"""|os\.environ\[\s*['"]([A-Za-z_][A-Za-z0-9_]*)['"]\s*\]""")
#: provided by sbatch's own --export=ALL / the scheduler, never by a driver
IGNORE_EXPORTS = {"ALL", "NONE"}


def _strip_comments(text):
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))


def scan_sh(path, base, seen=None):
    """(reads, required, files) for a shell file and everything it sources."""
    seen = set() if seen is None else seen
    path = os.path.normpath(path)
    if path in seen:
        return set(), set(), []
    seen.add(path)
    text = _strip_comments(open(path).read())
    reads = {a or b for a, b in SH_REF.findall(text)}
    req = set(SH_REQ.findall(text))
    files = [path]
    for src in SH_SRC.findall(text):
        p = src.replace("${BASE_PATH}", base).replace("$BASE_PATH", base)
        if "$" in p:
            raise SystemExit(f"contract: cannot resolve sourced path {src!r} in {path}")
        r2, q2, f2 = scan_sh(p, base, seen)
        reads |= r2
        req |= q2
        files += f2
    return reads, req, files


def scan_py(paths):
    reads = set()
    for p in paths:
        for a, b in PY_ENV.findall(open(p).read()):
            reads.add(a or b)
    return reads


def parse_exports(s):
    keys = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k = tok.split("=", 1)[0]
        if k in IGNORE_EXPORTS:
            continue
        keys.append(k)
    return keys


def check(worker, py, base, exports):
    reads, req, files = scan_sh(worker, base)
    reads |= scan_py(py)
    exp = set(parse_exports(exports))
    dead = sorted(exp - reads)
    unmet = sorted(req - exp)
    return {"ok": not dead and not unmet,
            "exported_but_never_read": dead,
            "required_but_not_exported": unmet,
            "n_exported": len(exp), "n_required": len(req), "n_read": len(reads),
            "files_scanned": files + list(py)}


def selftest():
    d = tempfile.mkdtemp(prefix="wc_selftest_")
    inc = os.path.join(d, "inc.sh")
    open(inc, "w").write('echo "${FROM_SOURCE:-x}"\n')
    w = os.path.join(d, "w.sh")
    open(w, "w").write('#!/bin/bash\n: "${A:?}"\n: "${B:?}"\nsource "${BASE_PATH}/inc.sh"\n'
                       'echo "${OPT:-1}" $C\n# : "${COMMENTED:?}"\n')
    py = os.path.join(d, "p.py")
    open(py, "w").write('import os\nos.environ.get("ENVPY")\n')
    cases = [
        ("exact", "ALL,A=1,B=2", True, [], []),
        ("silent-corruption shape: METHOD exported, never read",
         "ALL,A=1,B=2,METHOD=bsm", False, ["METHOD"], []),
        ("required not exported", "ALL,A=1", False, [], ["B"]),
        ("read via a sourced file", "ALL,A=1,B=2,FROM_SOURCE=y", True, [], []),
        ("read via python environ", "ALL,A=1,B=2,ENVPY=z", True, [], []),
        ("commented-out requirement does not count", "ALL,A=1,B=2", True, [], []),
    ]
    fails = []
    for name, exp, ok, dead, unmet in cases:
        r = check(w, [py], d, exp)
        good = (r["ok"] == ok and r["exported_but_never_read"] == dead
                and r["required_but_not_exported"] == unmet)
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {r['exported_but_never_read']} "
              f"/ {r['required_but_not_exported']}")
        if not good:
            fails.append(name)
    print("worker_contract selftest", "PASS" if not fails else f"FAIL {fails}")
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--py", action="append", default=[])
    ap.add_argument("--base")
    ap.add_argument("--exports")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.worker and a.base and a.exports is not None):
        ap.print_usage()
        return 2
    r = check(a.worker, a.py, a.base, a.exports)
    if a.json:
        print(json.dumps(r))
    elif r["ok"]:
        print(f"contract OK {a.label}: {r['n_exported']} exported all read; "
              f"{r['n_required']} required all exported")
    else:
        print(f"CONTRACT VIOLATED {a.label}", file=sys.stderr)
        print(f"  exported but never read : {r['exported_but_never_read']}", file=sys.stderr)
        print(f"  required but not exported: {r['required_but_not_exported']}", file=sys.stderr)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
