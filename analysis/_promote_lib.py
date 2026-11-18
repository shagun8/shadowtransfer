"""Shared helpers for the evidence-promotion jobs (results/13, 14, 15, 19-21).

These jobs compute nothing. Each one reads an existing arm artifact, copies its
values verbatim into a CONTRACT.md section 6 results JSON, and records the source
path and sha256 of everything it copied. The only arithmetic any of them performs
is the one sum results/20 is explicitly allowed to carry, and that field is
labelled as a sum.

Read-only on every source. Each job writes exactly one new file under results/
and refuses to overwrite an existing one.
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

BASE = os.environ["PROJECT_ROOT"]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def fail(msg):
    raise SystemExit("ASSERTION FAILED: " + msg)


def load_json(rel):
    p = os.path.join(BASE, rel)
    if not os.path.isfile(p):
        fail(f"source absent: {rel}")
    return json.load(open(p))


def read_text(rel):
    p = os.path.join(BASE, rel)
    if not os.path.isfile(p):
        fail(f"source absent: {rel}")
    return open(p).read()


def source(rel, role):
    p = os.path.join(BASE, rel)
    if not os.path.isfile(p):
        fail(f"source absent: {rel}")
    return dict(path=rel, sha256=sha256_file(p),
                bytes=os.path.getsize(p), role=role)


def env_block():
    return dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        env=dict(python=sys.version.split()[0],
                 conda_env="myenv "
                           "(conda env)",
                 pythonnousersite=os.environ.get("PYTHONNOUSERSITE")),
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; "
                        "only final_paper/ is git-connected and this job does "
                        "not touch it",
        compute_note="read-only CPU repackaging on the analysis host. No number is "
                     "recomputed, no training, no GPU inference, no dump read "
                     "or written.")


def write_results(out, name, dry_run=False):
    """Write results/{name}.json, refusing to overwrite."""
    if dry_run:
        print(f"\nDRY_RUN: results/{name}.json not written. "
              f"{len(out)} top-level keys, {len(out.get('caveats', []))} "
              f"caveats.")
        return None
    path = os.path.join(BASE, "results", f"{name}.json")
    if os.path.exists(path):
        fail(f"{path} already exists; this job writes exactly one new file")
    tmp = path + ".new"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, allow_nan=False)
    os.replace(tmp, path)
    print(f"\nwrote {path}")
    print(f"  sha256 {sha256_file(path)}")
    print(f"  caveats {len(out['caveats'])}")
    print(f"  headline: {out['headline']}")
    return path
