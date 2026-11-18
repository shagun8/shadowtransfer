#!/usr/bin/env python3
"""ARM A BLOCK 5 -- resolve the 108 tier-2 probe cells (D-B5-1). Stdlib only.

The block-4 inventory never resolved tier 2 (deferred). Rule, per parent run_id:
search every /scratch dump root (dumps*/seo and dumps*/seo/seo) for a directory
of EXACTLY that name carrying .DONE + run.json + perimage.parquet. A copy is
excluded only if ITS OWN run.json says smoke (or carries a smoke-truncated
test count); the root's name is reported, never trusted. Exactly one real copy
must remain, else the cell is unresolved and refused. Comparator: the vanilla
upper in dumps/seo, as block 4. OGLANet: report the n=3 and any n>=10 band file.
"""
import argparse
import glob
import json
import os
from datetime import datetime, timezone

SCR = os.environ["ST_SCRATCH_ROOT"]
ARCHS = ("mamnet", "dinov3", "oglanet")
METHODS = ("fada", "iim", "mrfp_plus", "segdesic")
FOLDS = ("OMA", "JAX", "UCSD")
FULL = (".DONE", "run.json", "perimage.parquet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    roots = sorted(set(glob.glob(f"{SCR}/dumps*/seo") + glob.glob(f"{SCR}/dumps*/seo/seo")))
    cells, errors = [], 0
    for arch in ARCHS:
        for method in METHODS:
            for fold in FOLDS:
                for seed in (0, 1, 2):
                    rid = f"{arch}-{method}-loco-r3500-{fold.lower()}-hr-s{seed}"
                    comp = f"{arch}-none-upper-r3500-{fold.lower()}-hr-s{seed}"
                    cand, excl = [], []
                    for r in roots:
                        d = os.path.join(r, rid)
                        if not all(os.path.exists(os.path.join(d, f)) for f in FULL):
                            continue
                        run = json.load(open(os.path.join(d, "run.json")))
                        if run.get("smoke"):
                            excl.append({"dir": d, "why": "run.json smoke=true"})
                        else:
                            cand.append({"dir": d, "ckpt": run.get("ckpt_path"),
                                         "train_sun": run.get("train_sun_elev_mean")})
                    cdir = os.path.join(SCR, "dumps/seo", comp)
                    errs = []
                    if len(cand) != 1:
                        errs.append(f"{len(cand)} real complete copies: {[c['dir'] for c in cand]}")
                    elif not (cand[0]["ckpt"] and os.path.exists(cand[0]["ckpt"])):
                        errs.append(f"parent ckpt missing: {cand[0]['ckpt']}")
                    elif cand[0]["train_sun"] is None:
                        errs.append("parent run.json has no train_sun_elev_mean")
                    if not all(os.path.exists(os.path.join(cdir, f)) for f in FULL):
                        errs.append(f"comparator incomplete: {cdir}")
                    pdir = cand[0]["dir"] if len(cand) == 1 else None
                    e9 = None
                    if arch == "oglanet" and pdir:
                        n10 = sorted(glob.glob(os.path.join(pdir, "inference_repeat_n*.json")))
                        e9 = {"n3": os.path.exists(os.path.join(pdir, "inference_repeat.json")),
                              "n10_files": n10}
                    errors += bool(errs)
                    cells.append({"run_id": rid, "arch": arch, "method": method, "holdout": fold,
                                  "seed": seed, "family": "method",
                                  "parent_dump_dir": pdir, "parent_excluded": excl,
                                  "comparator": comp, "comparator_dump_dir": cdir,
                                  "run_id_probe": f"{rid}-p1histmatch",
                                  "probe_dump_dir": os.path.join(SCR, "dumps/seo", f"{rid}-p1histmatch"),
                                  "e9": e9, "errors": errs, "runnable": not errs})
    out = {"authority": "DECISIONS_armA_wave2.md D-B5-1", "n_roots": len(roots),
           "n_cells": len(cells), "n_unresolved": errors,
           "generated_utc": datetime.now(timezone.utc).isoformat(), "cells": cells}
    tmp = args.out + ".new"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, args.out)
    print(f"tier-2 roster: {len(cells)} cells, {errors} unresolved, {len(roots)} roots -> {args.out}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
