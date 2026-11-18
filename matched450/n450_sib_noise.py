#!/usr/bin/env python
"""Repeat-inference noise for an OGLANet SIB (sibc4) n450 cell -- ONE REP PER JOB.

Why this script exists at all. `n450_bs4.py` REFUSES every SIB cell by design
(`n450_bs4.py:100`): `cell["prob_variant"]` is None for the SIB family because
`run_inference_probs.py` has no SIB model variant, so it cannot build an
OGLANetSIB. Job 23273276 (`n450-bs4-sibc4-phoenix`) hit exactly that refusal.
The only pass that can score a SIB checkpoint is the SIB trainer's own
`test_and_save_predictions` -- batch `args.batch_size` (8) under fp16 autocast --
which is what this script drives, through the same `shadowtransfer/n450_sib.py`
harness that wrote the cell's published-recompute dump.

What one rep does: loads the cell's EXISTING `checkpoints/best_model.pth`
(symlinked, never copied, never written), runs `mod.main()` with `epochs=0` so
the training loop (`train_oglanet_sib.py:1646`) does not execute, lets the
trainer load the best checkpoint (`:1770-1774`) and run the test pass
(`:1779`), then records `test_results.json`'s two-class tolerant mIoU at K=2
and foreground Shadow_IOU.

Trains nothing. Writes no checkpoint. Writes no published artifact and touches
no original cell: every write lands under --out_root, and the script refuses if
--out_root resolves inside the cell's dump dir or its checkpoint dir.

Seeding matches the production path exactly (seed 42 for python/numpy/torch/
cuda, `n450_roster.SEED`), so any spread between reps is framework
non-determinism rather than a different RNG draw -- which is the quantity being
measured.

  python n450_sib_noise.py --run_id <sibc4 run_id> --rep N --out_root DIR
"""
import argparse
import json
import os
import random
import sys
import time

BASE = os.environ.get("BASE_PATH") or os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_runtime as RT          # noqa: E402
from shadowtransfer import n450_roster as R            # noqa: E402
from shadowtransfer import n450_inject as INJ          # noqa: E402
from shadowtransfer import n450_config as CFG          # noqa: E402


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def seed_all(s):
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--post_timeout", type=int, default=1800)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    t_start = time.time()
    cell = R.by_run_id().get(a.run_id)
    if cell is None:
        print(f"REFUSED: {a.run_id} is not an n450 run_id")
        return 2
    if cell["method"] != "sibc4":
        print(f"REFUSED: {a.run_id} is method={cell['method']}, not sibc4. "
              f"Non-SIB cells have a prob_variant and go through n450_bs4.py.")
        return 2

    src_ckpt = os.path.join(cell["ckpt_dir"], "checkpoints", "best_model.pth")
    out = os.path.abspath(os.path.join(a.out_root, a.run_id, f"rep{a.rep}"))
    probs_dir = os.path.join(out, "probs")

    # ---- never write into the cell or its dump -----------------------------
    for guarded in (cell["dump_dir"], cell["ckpt_dir"], cell["orig_dir"]):
        g = os.path.normpath(guarded)
        if os.path.normpath(out) == g or os.path.normpath(out).startswith(g + os.sep):
            print(f"REFUSED: --out_root puts this rep inside {guarded}, which is "
                  f"an original/published-recompute path. Pick a fresh root.")
            return 2
    if not os.path.exists(src_ckpt):
        print(f"REFUSED: no checkpoint at {src_ckpt}")
        return 3

    print("=" * 78)
    print(f"n450 SIB repeat-inference  {a.run_id}  rep {a.rep}   ({utc()})")
    print(f"  arch/method/city  {cell['arch']}/{cell['method']}/{cell['city']}")
    print(f"  checkpoint (ro)   {src_ckpt}")
    print(f"  out dir           {out}")
    print(f"  probs dir         {probs_dir}")
    print("=" * 78, flush=True)

    if a.dry_run:
        print("[dry-run] resolved paths only; nothing built, nothing written")
        return 0

    RT.preflight_write_targets(
        BASE, [os.path.join(out, f) for f in
               ("test_results.json", "noise_rep.json", ".DONE")],
        label=f"{a.run_id}-rep{a.rep}")

    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    link = os.path.join(out, "checkpoints", "best_model.pth")
    if not os.path.exists(link):
        os.symlink(src_ckpt, link)          # read-only use; never copied
    print(f"[ckpt] symlink {link} -> {src_ckpt}", flush=True)

    # ---- trainer + config, exactly as n450_train.py resolves them ----------
    tree = os.path.join(BASE, cell["arch"])
    sys.path.insert(0, tree)
    os.chdir(tree)
    orig, orig_path, _sha = RT.read_with_retry(
        120, f"original config of {cell['legacy_run_dir']}",
        lambda: CFG.read_orig_config(cell["orig_dir"], cell["config_source"]))

    mod, trainer_sha = CFG.load_trainer(tree, cell["trainer_module"],
                                        os.path.join(out, "_modsrc"))
    overrides = {"output_dir": out, "epochs": 0}
    permitted = dict(CFG.PERMITTED_CHANGED)
    permitted.update(CFG.DUMP_ONLY_CHANGED)
    ns, _dests = CFG.resolve(mod, "parse_args", orig, overrides)
    d = CFG.diff(orig, dict(vars(ns)), permitted)
    print(f"[config] baseline {orig_path}")
    for k, v in d["changed"].items():
        print(f"  CHANGED {k}: {v['orig']!r} -> {v['new']!r}")
    print(f"  VERDICT {d['verdict']}", *d["refusals"], sep="\n    ")
    if d["verdict"] != "PERMITTED":
        return 4
    if int(ns.epochs) != 0:
        print(f"REFUSED: epochs resolved to {ns.epochs}, not 0 -- this script "
              f"must never train.")
        return 4

    # ---- injection, plot silencing, seeds ----------------------------------
    INJ.configure(R.SHARED_LISTS, cell["city"], cell["source_cities"],
                  ns.base_data_root, ns.resolution)
    import data.dataset_sib as DS
    INJ.wrap_class(DS.ShadowDatasetSIB, "ShadowDatasetSIB")
    silenced = []
    for name in list(vars(mod)):
        if name.startswith("plot_") or name in ("save_best_worst_visualizations",
                                                "save_best_worst_predictions"):
            setattr(mod, name, lambda *x, **k: None)
            silenced.append(name)
    seed_all(R.SEED)

    import torch
    cudnn = dict(deterministic=bool(torch.backends.cudnn.deterministic),
                 benchmark=bool(torch.backends.cudnn.benchmark))

    timebox = RT.PhaseTimeout(a.post_timeout, out, label="repeat-inference")
    from shadowtransfer import n450_sib as SIB
    silenced += SIB.install(mod, ns, probs_dir, timebox)
    print(f"[silenced] {sorted(set(silenced))}", flush=True)

    t0 = time.time()
    mod.main()
    wall = time.time() - t0

    if torch.cuda.is_available():
        print(f"[vram] peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"on {torch.cuda.get_device_name(0)}", flush=True)

    with open(os.path.join(out, "test_results.json")) as f:
        tr = json.load(f)
    tol = tr["tolerant_2px"]
    rec = {
        "run_id": a.run_id, "rep": a.rep, "written_utc": utc(),
        "miou_k2_twoclass_trainer": tol["mIOU"],
        "fg_iou_k2_trainer": tol["Shadow_IOU"],
        "strict_miou_trainer": tr["strict"]["mIOU"],
        "strict_fg_iou_trainer": tr["strict"]["Shadow_IOU"],
        "num_images": tr["num_images"],
        "inference_batch_size": int(ns.batch_size),
        "inference_precision": f"fp16 autocast (use_amp={bool(ns.use_amp)})",
        "seed": R.SEED,
        "seed_note": "seeded identically every rep, as the production path "
                     "does; spread is framework non-determinism, not RNG",
        "cudnn": cudnn,
        "checkpoint": src_ckpt,
        "epochs_override": int(ns.epochs),
        "trainer_module": cell["trainer_module"],
        "trainer_sha256": trainer_sha,
        "inference_block": SIB.STATE.get("test"),
        "wall_s": wall,
        "total_wall_s": time.time() - t_start,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "trains_nothing": True,
        "writes_published_artifact": False,
    }
    with open(os.path.join(out, "noise_rep.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    open(os.path.join(out, ".DONE"), "w").close()
    print(f"\n[rep {a.rep}] two-class mIoU K=2 = {tol['mIOU']:.6f}   "
          f"fg IoU K=2 = {tol['Shadow_IOU']:.6f}   wall {wall:.1f}s")
    print(f"[n450-sib-noise] DONE {a.run_id} rep {a.rep} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
