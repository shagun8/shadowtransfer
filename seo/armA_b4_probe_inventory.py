#!/usr/bin/env python3
"""ARM A / BLOCK 4 -- the §4.2 probe inventory, DERIVED BY RULE.

`armA_seo_histmatch.py --inventory-only` does not exist and never did; the
2026-10-12 and 2026-10-15 session logs both name it as the "exact next
command". This script is that command, implemented. It follows the precedent of
`armA_b3_ce_e9_roster.py`: the set is derived by a printed rule, not restated
inside a submit script, so the count is auditable and the exclusions visible.

THE RULE. A cell is a BLOCK-4 PROBE PARENT iff:
    arch in (mamnet, dinov3, oglanet, crossearth)
    the run_id is <arch>-<method>-loco-r3500-<city>-hr-s<seed>   (S-EO rung)
and NONE of:
    "smoke" in the dump path             -> smoke tree
    "p1histmatch" in the dir name        -> a probe, not a probe parent
    dir is "<run_id>.<suffix>"           -> superseded snapshot
    method == "expa"                     -> block 2's Experiment A cells
    condition == upper                   -> the COMPARATOR, not a parent

UPPER cells are excluded as parents and counted separately: they are what
`comparator_run_id` resolves for R's denominator, and `parent_run_id` only ever
builds a "-loco-" id.

EXPA cells are excluded and counted separately, on instruction. They are
themselves derived from a vanilla LOCO `parent_run_id`, so probing one would
stack a probe on a retrained decoder and compound the R-denominator question
D-A2-2 already guards.

ALREADY PROBED. A cell is probed iff "<run_id>-p1histmatch" with a .DONE exists
under any scanned root. A probe dir without .DONE is PARTIAL and counted TODO.

WHY EVERY FILE TOUCH HAPPENS IN A CHILD PROCESS
-----------------------------------------------
On 2026-10-16,
  dumps/seo/oglanet-fda-loco-r3500-oma-hr-s1/inference_repeat.json
was found to hang `stat` in UNINTERRUPTIBLE I/O on dt-login01 AND dt-login03,
while its own directory and its sibling run.json stat instantly and `lfs check
servers` reported all 210 servers active. `timeout 6` could not kill it: a
process in Lustre D-state does not die on a signal, so neither a timeout nor a
ReadGuard can rescue the process that issued the read.

The consequence for this script is structural, not cosmetic. The classification
rule above is satisfied ENTIRELY by directory NAMES, and readdir was observed
healthy everywhere. So the parent process does readdir only, and every
file-level access -- .DONE, run.json, inference_repeat.json -- happens in a
per-cell child that the parent ABANDONS on timeout and records as POISONED. One
wedged cell costs one cell and is named in the report; it cannot take the
inventory with it.

THE OGLANET BAND is part of that per-cell payload: an OGLANet probe cell cannot
resolve its reproduction rule without its parent's inference_repeat.json
(D-A2-2 -- `resolve_repro_rule` refuses rather than guess), so a poisoned band
file is a cell that cannot be submitted, and the inventory has to say which.

Stdlib only. Writes the inventory + manifest JSON; prints the report.
"""
import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time

ARCHS = ("mamnet", "dinov3", "oglanet", "crossearth")
CELL_RE = re.compile(r"^(mamnet|dinov3|oglanet|crossearth)-([a-z0-9_]+)-"
                     r"(loco|upper)-r3500-([a-z]+)-hr-s(\d+)$")
VANILLA = "vanilla"
SUFFIX = "p1histmatch"
CITY = {"jax": "JAX", "oma": "OMA", "ucsd": "UCSD"}

#: the per-cell worker, run as `python -I -c PROBE <dir>`. Everything that
#: touches a file is in here, i.e. in the child, never in the parent.
PROBE = r'''
import json, os, sys
d = sys.argv[1]
o = {"done": os.path.exists(os.path.join(d, ".DONE")),
     "repeat": os.path.exists(os.path.join(d, "inference_repeat.json"))}
try:
    j = json.load(open(os.path.join(d, "run.json")))
    o["run_json"] = True
    for k in ("run_id", "arch", "method", "condition", "rung", "smoke",
              "ckpt_path", "block", "wave", "parent_run_id"):
        if k in j:
            o[k] = j[k]
except Exception as e:
    o["run_json"] = False
    o["run_json_error"] = type(e).__name__
sys.stdout.write(json.dumps(o))
'''


def family(method):
    return "vanilla" if method == VANILLA else "method"


def seo_roots(base):
    return sorted(set(glob.glob(os.path.join(base, "*", "seo"))
                      + glob.glob(os.path.join(base, "*", "*", "seo"))))


def probe_many(dirs, timeout, jobs, label=""):
    """{dir: payload|None}. None == POISONED (child abandoned, still in D-state).

    Children are abandoned rather than killed: SIGKILL does not land on a
    process in uninterruptible Lustre I/O, and waiting on it is the hang.
    """
    res, pend, i, t_last = {}, [], 0, time.time()
    while i < len(dirs) or pend:
        while i < len(dirs) and len(pend) < jobs:
            d = dirs[i]; i += 1
            p = subprocess.Popen([sys.executable, "-I", "-c", PROBE, d],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL)
            pend.append([d, p, time.time()])
        still = []
        for item in pend:
            d, p, t0 = item
            if p.poll() is not None:
                try:
                    res[d] = json.loads(p.stdout.read().decode() or "{}")
                except Exception:                              # noqa: BLE001
                    res[d] = {}
            elif time.time() - t0 > timeout:
                res[d] = None                                   # POISONED
                try:
                    p.stdout.close()
                except Exception:                              # noqa: BLE001
                    pass
            else:
                still.append(item)
        pend = still
        if time.time() - t_last > 15:
            t_last = time.time()
            print(f"    [{label}] {len(res)}/{len(dirs)} probed, "
                  f"{len(pend)} in flight", flush=True)
        if pend:
            time.sleep(0.05)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ["ST_SCRATCH_ROOT"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=24)
    args = ap.parse_args()

    # ---- phase 1: readdir ONLY, in the parent ------------------------------
    roots, bad_roots = seo_roots(args.base), []
    cand, probe_dirs = {}, {}
    excl = collections.Counter()
    for R in roots:
        try:
            names = sorted(os.listdir(R))
        except OSError as e:
            bad_roots.append([R, e.errno]); continue
        smoky = "smoke" in R
        for nm in names:
            d = os.path.join(R, nm)
            if SUFFIX in nm:
                base_rid = nm.replace("-" + SUFFIX, "")
                probe_dirs.setdefault(base_rid, []).append(d)
                continue
            m = CELL_RE.match(nm)
            if not m:
                if CELL_RE.match(nm.split(".")[0]):
                    excl["superseded snapshot (<run_id>.<suffix>)"] += 1
                else:
                    excl["dir name does not parse as an S-EO r3500 cell"] += 1
                continue
            arch, method, cond, hold, seed = m.groups()
            if smoky or "smoke" in d:
                excl["smoke tree (path contains 'smoke')"] += 1; continue
            if method == "expa":
                excl["block 2 Experiment A cell (method=expa)"] += 1; continue
            if cond != "loco":
                excl["upper cell (comparator, not a probe parent)"] += 1; continue
            cand.setdefault(nm, []).append(d)
    # ---- the PHASE-1 report, which is the trustworthy one -----------------
    # Phase 1 touches no file: readdir only. On 2026-10-16 the file-probe phase
    # below was contaminated by Lustre contention on a loaded login node (125 of
    # 449 directories timed out, including cells whose run.json had been read
    # instantly seconds earlier), which silently DROPPED cells and made the
    # composition read short -- mamnet/vanilla showed 5 of 9. The classification
    # rule is satisfied entirely by directory names, so phase 1 can and must be
    # reported on its own, independent of whether phase 2 degrades.
    p1_by = collections.Counter(
        (CELL_RE.match(r).group(1), CELL_RE.match(r).group(2)) for r in cand)
    p1_fam = collections.Counter(
        family(CELL_RE.match(r).group(2)) for r in cand)
    p1_archs = sorted({a for a, _ in p1_by})
    p1_meths = sorted({m for _, m in p1_by})
    p1_short = {f"{a}/{m}": n for (a, m), n in sorted(p1_by.items()) if n != 9}
    phase1 = {
        "touches_no_file": True,
        "n_probe_parents": len(cand),
        "n_probe_dirs_by_name": len(probe_dirs),
        "family_totals": dict(sorted(p1_fam.items())),
        "by_arch_method": {f"{a}/{m}": n for (a, m), n in sorted(p1_by.items())},
        "pairs_at_full_9": sum(1 for v in p1_by.values() if v == 9),
        "n_pairs": len(p1_by),
        "pairs_not_at_9": p1_short,
        "n_probed_by_name": len(set(probe_dirs) & set(cand)),
        "crossearth_method_cells": sum(
            n for (a, m), n in p1_by.items() if a == "crossearth" and m != VANILLA),
        "non_crossearth_method_cells": sum(
            n for (a, m), n in p1_by.items() if a != "crossearth" and m != VANILLA),
    }
    W0 = 78
    print("=" * W0)
    print("PHASE 1 -- READDIR ONLY (no stat, no open: cannot be contaminated)")
    print("=" * W0)
    print(f"roots {len(roots)}   PROBE PARENTS {len(cand)}   "
          f"probe dirs {len(probe_dirs)}   probed by name "
          f"{phase1['n_probed_by_name']}")
    print(f"FAMILY totals {dict(sorted(p1_fam.items()))}")
    print()
    print(f"{'method':11s}" + "".join(f"{a:>12s}" for a in p1_archs) + f"{'row':>7s}")
    for m in p1_meths:
        row = [p1_by.get((a, m), 0) for a in p1_archs]
        print(f"{m:11s}" + "".join(f"{v:12d}" for v in row) + f"{sum(row):7d}")
    print(f"{'TOTAL':11s}"
          + "".join(f"{sum(p1_by.get((a, m), 0) for m in p1_meths):12d}"
                    for a in p1_archs)
          + f"{len(cand):7d}")
    print(f"\n(arch,method) pairs at the full 3-fold x 3-seed 9: "
          f"{phase1['pairs_at_full_9']} of {len(p1_by)}")
    for k, n in p1_short.items():
        print(f"   SHORT {k:24s} {n} of 9")
    print(f"crossearth METHOD {phase1['crossearth_method_cells']}   "
          f"non-crossearth METHOD {phase1['non_crossearth_method_cells']}")
    print("=" * W0, flush=True)

    # ---- phase 2: every file touch, in abandonable children ----------------
    todo_probe = [d for ds in probe_dirs.values() for d in ds]
    pay_probe = probe_many(todo_probe, args.timeout, args.jobs, "probes")
    flat = [d for ds in cand.values() for d in ds]
    pay_cell = probe_many(flat, args.timeout, args.jobs, "cells")

    poisoned = sorted([d for d, v in pay_probe.items() if v is None]
                      + [d for d, v in pay_cell.items() if v is None])

    probed, partial_only = {}, {}
    for rid, ds in probe_dirs.items():
        done = any((pay_probe.get(d) or {}).get("done") for d in ds)
        (probed if done else partial_only)[rid] = ds

    cells = {}
    for rid, ds in cand.items():
        best = None
        for d in ds:
            v = pay_cell.get(d)
            if v is None:
                continue
            if v.get("smoke"):
                continue
            if best is None or (v.get("ckpt_path") and not best[1].get("ckpt_path")):
                best = (d, v)
        if best is None:
            excl["every copy poisoned or smoke=true"] += 1
            continue
        d, v = best
        arch, method, cond, hold, seed = CELL_RE.match(rid).groups()
        cells[rid] = {
            "run_id": rid, "arch": arch, "method": method,
            "family": family(method), "holdout": CITY.get(hold, hold.upper()),
            "seed": int(seed), "dump_dir": d,
            "dump_done": bool(v.get("done")),
            "ckpt_path": v.get("ckpt_path"),
            "block": v.get("block"), "wave": v.get("wave"),
            "probed": rid in probed,
            "probe_partial": rid in partial_only,
            "e9_band": ("PRESENT" if v.get("repeat") else "ABSENT"),
            "n_copies": len(ds),
        }

    # A parent whose OWN dump never completed cannot be probed: the probe reads
    # its stored perimage.parquet to reproduce the baseline.
    no_parent_dump = sorted(r for r, v in cells.items() if not v["dump_done"])
    todo = sorted(r for r, v in cells.items()
                  if not v["probed"] and v["dump_done"])
    # OGLANet without its parent's band is BLOCKED, not TODO (D-A2-2).
    blocked_band = sorted(r for r in todo if cells[r]["arch"] == "oglanet"
                          and cells[r]["e9_band"] != "PRESENT")
    ready = [r for r in todo if r not in set(blocked_band)]

    by_am = collections.Counter((v["arch"], v["method"]) for v in cells.values())
    todo_am = collections.Counter((cells[r]["arch"], cells[r]["method"]) for r in todo)
    out = {
        "rule": __doc__,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base": args.base, "n_roots": len(roots), "roots": roots,
        "phase1_readdir_only": phase1,
        "roots_unreadable": bad_roots,
        "n_probe_parents": len(cells),
        "n_probed": sum(1 for v in cells.values() if v["probed"]),
        "n_todo": len(todo), "n_ready": len(ready),
        "by_arch_method": {f"{a}/{m}": c for (a, m), c in sorted(by_am.items())},
        "todo_by_arch_method": {f"{a}/{m}": c for (a, m), c in sorted(todo_am.items())},
        "family_totals": dict(sorted(collections.Counter(
            v["family"] for v in cells.values()).items())),
        "family_todo": dict(sorted(collections.Counter(
            cells[r]["family"] for r in todo).items())),
        "family_ready": dict(sorted(collections.Counter(
            cells[r]["family"] for r in ready).items())),
        "excluded": dict(excl.most_common()),
        "poisoned_paths": poisoned,
        "probe_partial": sorted(partial_only),
        "parents_without_completed_dump": no_parent_dump,
        "oglanet_blocked_no_e9_band": blocked_band,
        "todo": todo, "ready": ready,
        "cells": {k: cells[k] for k in sorted(cells)},
    }
    json.dump(out, open(args.out, "w"), indent=2)
    json.dump({"arm": "A", "block": 4, "probe": "s4.2 histmatch",
               "n_intended": len(ready), "run_ids": ready,
               "held_back_oglanet_no_band": blocked_band},
              open(args.manifest, "w"), indent=2)

    W = 78
    print("=" * W)
    print("ARM A / BLOCK 4 -- §4.2 PROBE INVENTORY (derived by rule)")
    print("=" * W)
    print(f"seo roots scanned            {len(roots)}")
    print(f"PROBE PARENTS (S-EO LOCO)    {len(cells)}")
    print(f"  already probed (.DONE)     {out['n_probed']}")
    print(f"  TO PROBE                   {len(todo)}")
    print(f"  of those, READY to submit  {len(ready)}")
    print("-" * W)
    print("COMPOSITION by (arch, method, family)   total / probed / TODO")
    for (a, m), c in sorted(by_am.items()):
        t = todo_am.get((a, m), 0)
        print(f"  {a:11s} {m:10s} {family(m):8s} {c:5d} / {c - t:5d} / {t:5d}")
    print("-" * W)
    print(f"FAMILY totals   {out['family_totals']}")
    print(f"FAMILY todo     {out['family_todo']}")
    print(f"FAMILY ready    {out['family_ready']}")
    print("-" * W)
    print("EXCLUDED (by the printed rule):")
    for k, v in excl.most_common():
        print(f"  {v:6d}  {k}")
    print("-" * W)
    if poisoned:
        print(f"*** POISONED PATHS -- child abandoned in D-state: {len(poisoned)}")
        for d in poisoned:
            print(f"      {d}")
    else:
        print("poisoned paths: none in this scan")
    if no_parent_dump:
        print(f"*** parents with NO completed dump (cannot be probed): "
              f"{len(no_parent_dump)}")
        for r in no_parent_dump[:12]:
            print(f"      {r}")
    if out["probe_partial"]:
        print(f"PARTIAL probe dumps (no .DONE), counted TODO: "
              f"{len(out['probe_partial'])}")
        for r in out["probe_partial"]:
            print(f"      {r}")
    if blocked_band:
        print(f"*** OGLANET BLOCKED -- parent has no usable "
              f"inference_repeat.json: {len(blocked_band)}")
        for r in blocked_band:
            print(f"      {cells[r]['e9_band']:9s} {r}")
    print("=" * W)
    print(f"inventory -> {args.out}")
    print(f"manifest  -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
