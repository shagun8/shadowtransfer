#!/usr/bin/env python3
"""Authoritative writable-headroom gate for the scratch allocation.

READ THIS BEFORE CHANGING THE INPUT SOURCE. Three separate readings of the
question "how much can we still write" have now been wrong, each one costing
real grids:

  1. `df /projects`                 -- shows the QUOTA view, not the
     filesystem. Reported 100% while space existed. (CLAUDE.md 2026-09-01)

  2. `lfs quota -p 20929 <fs>`      -- on 2026-09-09 reported used 1096.9 GiB
     of a 1500.0 GiB soft limit, i.e. 403 GiB apparently free, while every
     single write failed with Errno 122. Two 27-cell arm-A grids died on it.

  3. per-OST grant arithmetic       -- sum(granted blimit) - sum(granted used)
     over the `lfs quota -v -p` OST lines gave 10.0 GiB, which matched the
     observed wall and so LOOKED like the mechanism. It was a coincidence.
     That 10 GiB was simply the remaining distance to the hard limit. The
     pool/OST-topology reading built on top of it was wrong too.

THE AUTHORITATIVE SOURCE is Delta's own quota tool, /sw/user/scripts/quota
(NOT /usr/bin/quota, which reports only /u and never shows /work/hdd). Its
"Quota usage for allocations" table carries the real enforced per-allocation
limits. Observed 2026-09-22:

    | <ST_QUOTA_ALLOC>  | 1.067T* | 1000G | 1.074T | ...
      used --------------^        ^soft     ^hard

`lfs quota -p 20929 /work/hdd` agrees on USED (1.067T) and disagrees on both
LIMITS (it says soft 1.465T / hard 1.611T). Usage agrees; limits do not.
Trust this tool for the limits.

Units are BINARY (G=GiB, T=TiB). Verified internally: every allocation row
has hard == 1.1 x soft, and 1.1 x 1000 GiB = 1100 GiB = 1.0742 TiB, which is
exactly the "1.074T" printed. A decimal reading does not close.

GATE SEMANTICS. The gate returns headroom against the SOFT limit, not the
hard one. Crossing soft starts a grace timer; /projects hit grace expiry and
TRUNCATED FILES (CLAUDE.md 2026-09-01). A grid must never be planned into
grace. If usage is already above soft the gate refuses and says OVER_SOFT.

AND THE NUMBER IS STILL ONLY A PREDICTION. A `dd conv=fsync` probe at every
real write target is the only proof. This module does not replace the probe.

stdlib only. No imports from the rest of the tree.
"""

import argparse
import json
import os
import re
import subprocess
import sys

QUOTA_BIN = "/sw/user/scripts/quota"
DEFAULT_ALLOC = os.environ.get("ST_QUOTA_ALLOC", "")
GiB = 1024.0 ** 3

_UNITS = {"K": 1024.0,
          "M": 1024.0 ** 2,
          "G": 1024.0 ** 3,
          "T": 1024.0 ** 4,
          "P": 1024.0 ** 5}


def parse_size(tok):
    """'1.067T*' -> bytes. Binary units. The trailing * (over-soft marker) is
    stripped by the caller's regex but tolerated here too."""
    tok = tok.strip().rstrip("*").strip()
    if not tok or tok in ("-", "0"):
        return 0.0
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([KMGTP]?)", tok, re.IGNORECASE)
    if not m:
        raise ValueError("unparseable size %r" % tok)
    val = float(m.group(1))
    suf = m.group(2).upper()
    return val * _UNITS.get(suf, 1.0)


def read_quota(alloc=DEFAULT_ALLOC, timeout=180):
    """Run Delta's quota tool and return the row for `alloc`.

    Returns dict(used, soft, hard, over_soft, raw). Raises RuntimeError with
    the tool's own output attached if the row is absent -- never guesses.
    """
    try:
        proc = subprocess.run([QUOTA_BIN], capture_output=True, text=True,
                              timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(
            "%s not found. This is Delta's quota tool and it is the ONLY "
            "authoritative source for /work/hdd limits; /usr/bin/quota shows "
            "/u only. Refusing to fall back to `lfs quota`, which reported "
            "the wrong limits on 2026-09-09 and cost two grids." % QUOTA_BIN)
    except subprocess.TimeoutExpired:
        raise RuntimeError("%s timed out after %ds" % (QUOTA_BIN, timeout))

    out = proc.stdout
    for ln in out.splitlines():
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < 4 or cells[0] != alloc:
            continue
        used_tok, soft_tok, hard_tok = cells[1], cells[2], cells[3]
        return {
            "alloc": alloc,
            "used": parse_size(used_tok),
            "soft": parse_size(soft_tok),
            "hard": parse_size(hard_tok),
            "over_soft": used_tok.rstrip().endswith("*"),
            "raw": ln.strip(),
            "source": QUOTA_BIN,
        }
    raise RuntimeError(
        "no row for %s in %s output. Refusing to guess.\n--- output ---\n%s"
        % (alloc, QUOTA_BIN, out))


def headroom(alloc=DEFAULT_ALLOC, timeout=180):
    """Writable headroom in bytes against the SOFT limit, plus context.

    `free` is the gate number. It is soft - used and may be negative, which
    the caller must treat as a refusal, not as a small budget.
    """
    q = read_quota(alloc, timeout=timeout)
    q["free"] = q["soft"] - q["used"]
    q["free_to_hard"] = q["hard"] - q["used"]
    return q


def report(q, stream=sys.stdout):
    """Human-readable gate banner. Prints the source so a future reader can
    see which of the four candidate inputs produced the number."""
    p = stream.write
    p("[disk gate] source %s   allocation %s\n" % (q["source"], q["alloc"]))
    p("[disk gate] raw row: %s\n" % q["raw"])
    p("[disk gate] used %.1f GiB   soft %.1f GiB   hard %.1f GiB%s\n"
      % (q["used"] / GiB, q["soft"] / GiB, q["hard"] / GiB,
         "   *** OVER SOFT ***" if q["over_soft"] else ""))
    p("[disk gate] WRITABLE HEADROOM (soft - used) %.1f GiB"
      "   [to hard: %.1f GiB]\n" % (q["free"] / GiB, q["free_to_hard"] / GiB))
    p("[disk gate] NOTE this is a prediction; the dd probe at every real "
      "write target is the proof.\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alloc", default=DEFAULT_ALLOC)
    ap.add_argument("--json", action="store_true",
                    help="emit the row as JSON on stdout (bytes)")
    ap.add_argument("--timeout", type=int, default=180)
    a = ap.parse_args(argv)
    try:
        q = headroom(a.alloc, timeout=a.timeout)
    except RuntimeError as e:
        sys.stderr.write("FATAL disk gate: %s\n" % e)
        return 2
    if a.json:
        sys.stdout.write(json.dumps(q) + "\n")
    else:
        report(q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
