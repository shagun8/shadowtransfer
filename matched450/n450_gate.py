#!/usr/bin/env python3
"""
n450 retrain -- machine gate between stages. Runs at the TAIL of a smoke or
real-cell job (same allocation, no extra queue hop).

All 36 jobs are submitted up front: smoke -> real (afterok: every smoke of the
group) -> fan-out (afterok: real). Real and fan-out jobs are submitted with a
provisional 3:59:59 and are dependency-blocked, so they cannot start before a
gate has run. A gate can only LOWER a limit (SLURM lets a user shorten, never
lengthen), which is why the provisional value is the partition ceiling.

  --stage smoke   once EVERY smoke of the group carries .DONE: validate them
                  all; set the real cell's --time (and a provisional fan-out
                  --time per trainer) from the measured per-epoch rate
  --stage real    validate the real cell; set every fan-out --time from it
  --stage cancel  a smoke/real job failed: cancel everything downstream
On a failed validation the gate cancels downstream jobs (found with squeue by
job-name prefix, never by a guessed id range) and exits 3.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

BASE = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from shadowtransfer import n450_roster as R              # noqa: E402

CEIL = 4 * 3600 - 1
FLOOR = 20 * 60
SHELL_OVERHEAD_S = 300        # module purge/load + conda, not seen by python
GROSS_SELF_CHECK = 0.25       # trainer's own test vs scorer, fg IoU points
GATES = os.path.join(R.N450_ROOT, "gates")


def hms(s):
    s = int(min(CEIL, max(FLOOR, math.ceil(s / 60.0) * 60)))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def my_jobs(prefix):
    out = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i %j %T"],
                         capture_output=True, text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    return [(r[0], r[1], r[2]) for r in rows if r[1].startswith(prefix)]


def cancel(group, roles, why):
    done = []
    for role in roles:
        for jid, name, st in my_jobs(f"n450-g{group}-{role}-"):
            subprocess.run(["scancel", jid], check=False)
            done.append((jid, name, st))
            print(f"[gate] CANCELLED {jid} {name} ({st}): {why}", flush=True)
    return done


def set_time(group, role, run_ids_to_limit):
    upd = []
    for jid, name, st in my_jobs(f"n450-g{group}-{role}-"):
        rid = name.split(f"n450-g{group}-{role}-", 1)[1]
        if rid not in run_ids_to_limit:
            continue
        lim = run_ids_to_limit[rid]
        p = subprocess.run(["scontrol", "update", f"JobId={jid}", f"TimeLimit={lim}"],
                           capture_output=True, text=True)
        upd.append({"jobid": jid, "name": name, "state": st, "time": lim,
                    "rc": p.returncode, "err": p.stderr.strip()})
        print(f"[gate] {jid} {name}: TimeLimit -> {lim} rc={p.returncode} "
              f"{p.stderr.strip()}", flush=True)
    return upd


def validate(dump_dir, n_rows, smoke):
    import pandas as pd
    bad = []
    if not os.path.exists(os.path.join(dump_dir, ".DONE")):
        return None, [f"no .DONE in {dump_dir}"]
    r = json.load(open(os.path.join(dump_dir, "run.json")))
    shared = json.load(open(R.SHARED_LISTS))["lists"][r["holdout"]]["sha256"]
    if r["n_train"] != 450 or sorted(r["n_train_per_source_city"].values()) != [225, 225]:
        bad.append(f"n_train {r['n_train']} {r['n_train_per_source_city']}")
    if r["train_list_sha256"] != shared:
        bad.append("train list sha != shared")
    if r["n_val"] != 300 or r["n_test"] != 150:
        bad.append(f"n_val {r['n_val']} n_test {r['n_test']}")
    if r["config_diff_vs_original"]["verdict"] != "PERMITTED":
        bad.append("config diff not PERMITTED")
    if not r["data_access_observed"]["agree"]:
        bad.append("target access mismatch")
    df = pd.read_parquet(os.path.join(dump_dir, "perimage.parquet"))
    ref = pd.read_parquet(os.path.join(R.DUMP_ROOT,
                                       "mamnet-fda-loco-r450-chicago-hr-s0",
                                       "perimage.parquet"))
    if list(df.columns) != list(ref.columns) or len(df) != n_rows:
        bad.append(f"parquet {len(df)} rows / schema match "
                   f"{list(df.columns) == list(ref.columns)}")
    if df[["tp_tol", "fp_tol", "fn_tol", "tn_tol"]].isna().any().any():
        bad.append("NaN in counts")
    sc = r["scoring"].get("self_check")
    if not smoke and sc and abs(sc["delta"]) > GROSS_SELF_CHECK:
        bad.append(f"self-check delta {sc['delta']:.4f} > {GROSS_SELF_CHECK}")
    if not smoke and not (0.0 < r["headline_fg_iou_tol_pct"] < 100.0):
        bad.append(f"headline {r['headline_fg_iou_tol_pct']}")
    return r, bad


def rate(r):
    ep = max(1, r["epochs_run"])
    per_epoch = r["train_wall_s"] / ep
    post = r["wall_clock_s"] - r["train_wall_s"]
    return per_epoch, post


def worst_case(per_epoch, post, epochs_max=100):
    return SHELL_OVERHEAD_S + post + epochs_max * per_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["smoke", "real", "cancel"])
    ap.add_argument("--group", type=int, required=True)
    ap.add_argument("--role", default=None)
    a = ap.parse_args()
    os.makedirs(GATES, exist_ok=True)
    g = a.group
    rec = {"stage": a.stage, "group": g, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
           "job": os.environ.get("SLURM_JOB_ID")}
    cells = [c for c in R.roster() if c["group"] == g]
    if a.stage == "cancel":
        roles = ["real", "fan"] if a.role == "smoke" else ["fan"]
        rec["cancelled"] = cancel(g, roles, f"{a.role} job failed")
        json.dump(rec, open(os.path.join(GATES, f"g{g}_cancel_{rec['job']}.json"),
                            "w"), indent=1)
        return 0
    if a.stage == "smoke":
        smokes = R.smoke_cells(g)
        if not all(os.path.exists(os.path.join(R.smoke_dir(c), ".DONE")) for c in smokes):
            print(f"[gate] group {g}: not every smoke is done yet; the last one gates")
            return 0
        rates, bad = {}, []
        for c in smokes:
            r, b = validate(R.smoke_dir(c), 5, smoke=True)
            bad += [f"{c['run_id']}: {x}" for x in b]
            if r:
                rates[c["method"]] = rate(r)
        rec.update(rates=rates, problems=bad)
        if bad:
            rec["cancelled"] = cancel(g, ["real", "fan"], "smoke validation failed")
            json.dump(rec, open(os.path.join(GATES, f"g{g}_smoke.json"), "w"), indent=1)
            print("[gate] FAIL", *bad, sep="\n  ")
            return 3
        lim = {c["run_id"]: hms(1.5 * worst_case(*rates[c["method"]]))
               for c in cells}
        rec["limits"] = lim
        rec["updates"] = (set_time(g, "real", lim) + set_time(g, "fan", lim))
        json.dump(rec, open(os.path.join(GATES, f"g{g}_smoke.json"), "w"), indent=1)
        print(f"[gate] group {g} smoke gate PASS; limits {lim}")
        return 0
    # stage real
    real = R.by_run_id()[R.REAL_CELL[g]]
    r, bad = validate(real["dump_dir"], 150, smoke=False)
    rec["problems"] = bad
    if bad:
        rec["cancelled"] = cancel(g, ["fan"], "real-cell validation failed")
        json.dump(rec, open(os.path.join(GATES, f"g{g}_real.json"), "w"), indent=1)
        print("[gate] FAIL", *bad, sep="\n  ")
        return 3
    pe_real, post_real = rate(r)
    sm = json.load(open(os.path.join(GATES, f"g{g}_smoke.json")))["rates"]
    lim = {}
    for c in cells:
        pe = sm[c["method"]][0] * (pe_real / sm[real["method"]][0])
        wc = worst_case(pe, post_real)
        lim[c["run_id"]] = hms(max(1.25 * wc, 2.0 * r["wall_clock_s"] * pe / pe_real))
    rec.update(real_wall_s=r["wall_clock_s"], real_epochs_run=r["epochs_run"],
               real_per_epoch_s=pe_real, limits=lim,
               self_check=r["scoring"].get("self_check"))
    rec["updates"] = set_time(g, "fan", lim)
    json.dump(rec, open(os.path.join(GATES, f"g{g}_real.json"), "w"), indent=1)
    print(f"[gate] group {g} real gate PASS; fan limits {lim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
