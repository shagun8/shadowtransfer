#!/usr/bin/env python3
"""
n450 retrain -- one cell: retrain a published LOCO recipe at the matched
450-image budget, then score it into the CONTRACT section 3 dump schema.

Changes relative to the original cell, and nothing else (CHECK 4 enforces it):
  * training set: the shared 450-image list (225 per source city), injected by
    n450_inject into the trainer's own dataset class;
  * output paths (/scratch) and the -n450 run_id suffix;
  * save_freq -> 1e9 (best checkpoint only; bookkeeping);
  * seed 42 for python/numpy/torch/cuda: none of the eight trainers seeds
    anything and no original config records a seed, so the published seed is
    unrecoverable (as in E-12) and 42 is used.
Validation stays at the trainer's own 300; test is the published 150.

  python n450_train.py --run_id <id> [--smoke] [--dry_run] [--dump_only]
"""
import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from shadowtransfer import n450_roster as R              # noqa: E402
from shadowtransfer import n450_inject as INJ            # noqa: E402
from shadowtransfer import n450_config as CFG            # noqa: E402
from shadowtransfer import armc_runtime as RT            # noqa: E402

PY = os.path.join(BASE, "envs/satmae_cuda12/bin/python")

# Scored-pass inference batch size. The published OGLANet tables were inferred
# at batch 4 (run_inference.sh:73, run_inference_probs.sh:77), and OGLANet
# scores are batch-size dependent, so every n450 cell compared against them is
# scored from a batch-4 pass. MAMNet and DINOv3 have no DFFM and are
# unaffected; the value is recorded for them too so the artifact states it.
INFER_BS = 4


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def seed_all(s):
    import numpy as np
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def vram(tag):
    import torch
    if torch.cuda.is_available():
        g = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"[n450] VRAM {tag}: peak {g:.2f} GiB on "
              f"{torch.cuda.get_device_name(0)}", flush=True)
        return round(g, 3)
    return None


def check_access(cell, counts):
    """Observed held-out-city access vs the declared target_access. A
    mismatch refuses the dump: the column is what Table 2 groups by."""
    hold = os.path.normpath(os.path.join(R.DATA_ROOT, cell["city"]))
    hold_p = os.path.normpath(os.path.join(BASE, "data/Final_data_test",
                                           cell["city"]))
    fda_hold = [x["root"] for x in counts["fda_loads"]
                if x["root"].replace("//", "/").startswith((hold, hold_p))]
    observed = bool(counts["n_holdout_unlabeled"] or fda_hold)
    declared = cell["target_access"] == "unlabeled_target"
    out = {"observed_holdout_access": observed,
           "observed_fda_holdout_roots": fda_hold,
           "declared": cell["target_access"], "agree": observed == declared}
    print(f"[n450] target access declared={cell['target_access']} "
          f"observed_holdout_access={observed} -> agree={out['agree']}", flush=True)
    if not out["agree"]:
        raise SystemExit(f"FATAL target_access mismatch: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dump_only", action="store_true")
    ap.add_argument("--construct_only", action="store_true",
                    help="pre-queue CPU test: build loaders+model, assert "
                         "counts and access, stop before the first step")
    ap.add_argument("--post_timeout", type=int, default=2400)
    a = ap.parse_args()
    t_start = time.time()
    cell = R.by_run_id()[a.run_id]
    arch, method, city = cell["arch"], cell["method"], cell["city"]
    sib = method == "sibc4"
    rid = R.smoke_run_id(cell) if a.smoke else cell["run_id"]
    ckpt_root = (os.path.join(R.N450_ROOT, "smoke_ckpt", rid) if a.smoke
                 else cell["ckpt_dir"])
    dump_dir = R.smoke_dir(cell) if a.smoke else cell["dump_dir"]
    if a.construct_only:
        ckpt_root = dump_dir = os.path.join(R.N450_ROOT, "selftest", "construct", rid)
    print("=" * 78, f"\nn450 cell {rid}  ({utc()})\n  {arch}/{method}/{city}  group "
          f"{cell['group']}  smoke={a.smoke} dry_run={a.dry_run} "
          f"dump_only={a.dump_only}\n  ckpt_root {ckpt_root}\n  dump_dir  {dump_dir}"
          f"\n  original  {cell['orig_dir']}\n" + "=" * 78, flush=True)
    if os.path.exists(os.path.join(dump_dir, ".DONE")):
        print(f"SKIP: {dump_dir}/.DONE exists")
        return 0
    if not a.dry_run:
        RT.preflight_write_targets(BASE, [os.path.join(dump_dir, f) for f in
                                          ("perimage.parquet", "probs.npz",
                                           "run.json", ".DONE")], label=rid)
    best = (os.path.join(ckpt_root, "checkpoints", "best_model.pth") if sib else
            (glob.glob(os.path.join(ckpt_root, "*", "checkpoint_best.pth")) or [""])[0])
    if best and os.path.exists(best) and not (a.dump_only or a.dry_run):
        print(f"REFUSED: {best} exists without a .DONE. Re-enter with "
              f"DUMP_ONLY=1, or move the directory aside; this worker never "
              f"silently restarts a trained cell from zero.")
        return 3
    if a.dump_only and not (best and os.path.exists(best)):
        print(f"REFUSED: --dump_only but no checkpoint under {ckpt_root}")
        return 3

    tree = os.path.join(BASE, arch)
    sys.path.insert(0, tree)
    os.chdir(tree)
    # 2026-10-28: both reads go through read_with_retry, which retries ONCE
    # after 120 s on Errno 5 (EIO) or a read timeout, then fails. The
    # 2026-10-27 Lustre episode killed job 23209696 on the 300 s guard below
    # and took its two fan cells with it; these two reads touch the same
    # published tree and fail the same way.
    orig, orig_path, orig_sha = RT.read_with_retry(
        120, f"original config of {cell['legacy_run_dir']}",
        lambda: CFG.read_orig_config(cell["orig_dir"], cell["config_source"]))
    baseline, argsjson_vs_ckpt = orig, None
    if cell["ckpt_args_available"]:
        ck = RT.read_with_retry(
            300, "original checkpoint args",
            lambda: CFG.read_ckpt_args(
                os.path.join(cell["orig_dir"], "checkpoint_best.pth")))
        argsjson_vs_ckpt = CFG.diff(orig, ck, {})["changed"]
        baseline = ck
    overrides = {"output_dir": ckpt_root}
    permitted = dict(CFG.PERMITTED_CHANGED)
    if not sib:
        overrides["save_freq"] = CFG.SAVE_FREQ_OFF
    if a.smoke:
        overrides["epochs"] = 2
        permitted.update(CFG.SMOKE_CHANGED)
    if a.dump_only and sib:
        overrides["epochs"] = 0
        permitted.update(CFG.DUMP_ONLY_CHANGED)
    if method == "isw":
        # Operator ruling 2026-10-26 (decision 2): the published MAMNet ISW
        # masks were built from 900 images; use masks rebuilt from the shared
        # 450-image list (python/n450_isw_masks.py). Refuse without them.
        import n450_isw_masks as ISWM
        md = ISWM.out_dir(city)
        with open(os.path.join(md, "n450_provenance.json")) as f:
            prov = json.load(f)
        shared = json.load(open(R.SHARED_LISTS))["lists"][city]["sha256"]
        if (prov["problems"] or prov["num_images_used"] != R.N_TRAIN
                or prov["shared_list_sha256"] != shared or prov["smoke"]
                or not os.path.exists(os.path.join(md, "metadata.json"))):
            print(f"REFUSED: n450 ISW masks at {md} not valid: {prov}")
            return 5
        overrides["isw_mask_dir"] = md
        permitted["isw_mask_dir"] = ("operator ruling 2026-10-26 (decision 2): "
                                     "ISW masks rebuilt from the shared 450-image "
                                     "list; published masks used 900")
    os.makedirs(ckpt_root, exist_ok=True)
    mod, trainer_sha = CFG.load_trainer(tree, cell["trainer_module"],
                                        os.path.join(ckpt_root, "_modsrc"))
    ns, dests = CFG.resolve(mod, "parse_args" if sib else "get_args",
                            baseline, overrides)
    resolved = dict(vars(ns))
    d = CFG.diff(baseline, resolved, permitted)
    print(f"[CHECK 4] {rid}: baseline = "
          f"{'original checkpoint args' if cell['ckpt_args_available'] else orig_path}")
    for k, v in d["changed"].items():
        print(f"  CHANGED {k}: {v['orig']!r} -> {v['new']!r}   "
              f"[{d['changed_reasons'][k]}]")
    for k, v in d["orig_only"].items():
        print(f"  ORIG-ONLY {k}={v!r}")
    for k, v in d["new_only"].items():
        print(f"  NEW-ONLY {k}={v!r}   [{d['new_only_reasons'][k]}]")
    if argsjson_vs_ckpt:
        print(f"  (args.json vs checkpoint args, informational: {argsjson_vs_ckpt})")
    print(f"  VERDICT {d['verdict']}", *d["refusals"], sep="\n    ")
    with open(os.path.join(ckpt_root, "config_diff.json"), "w") as f:
        json.dump({"diff": d, "resolved": resolved, "baseline_source":
                   "checkpoint args" if cell["ckpt_args_available"] else orig_path,
                   "args_json_vs_ckpt_args": argsjson_vs_ckpt}, f, indent=1,
                  default=str)
    if d["verdict"] != "PERMITTED":
        return 4
    if a.dry_run:
        return 0

    # ---- injection + side effects + seeds -------------------------------
    INJ.configure(R.SHARED_LISTS, city, cell["source_cities"],
                  ns.base_data_root, ns.resolution)
    if sib:
        import data.dataset_sib as DS
        INJ.wrap_class(DS.ShadowDatasetSIB, "ShadowDatasetSIB")
    else:
        import data.dataset_enhanced as DE
        INJ.wrap_class(DE.ShadowDatasetEnhanced, "ShadowDatasetEnhanced")
        if arch == "mamnet":
            import data.fda_transform as FT
            INJ.wrap_fda_loader(FT)
    silenced = []
    for name in list(vars(mod)):
        if name.startswith("plot_") or name in ("save_best_worst_visualizations",
                                                "save_best_worst_predictions"):
            setattr(mod, name, lambda *x, **k: None)
            silenced.append(name)
    seed_all(R.SEED)
    timebox = RT.PhaseTimeout(a.post_timeout, dump_dir, label="post-training")
    probs_dir = os.path.join(ckpt_root, "probs")
    if sib:
        from shadowtransfer import n450_sib as SIB
        silenced += SIB.install(mod, ns, probs_dir, timebox)
        hist_bak = os.path.join(ckpt_root, "n450_training_history.json")
        if a.dump_only and not os.path.exists(hist_bak):
            print("REFUSED: dump-only SIB needs n450_training_history.json")
            return 3
        t0 = time.time()
        SIB.STATE["construct_only"] = a.construct_only
        try:
            mod.main()
        except SIB.ConstructOnly:
            check_access(cell, SIB.STATE["counts"])
            print(f"[n450] CONSTRUCT-ONLY OK {rid}")
            return 0
        t_train = (SIB.STATE.get("t_test_start", time.time()) - t0)
        if not a.dump_only:
            shutil.copy(os.path.join(ckpt_root, "training_history.json"), hist_bak)
        with open(hist_bak) as f:
            ep = SIB.epoch_stats(json.load(f), 2 if a.smoke else baseline["epochs"],
                                 ns.early_stopping_patience, R.N_TRAIN // ns.batch_size)
        counts = SIB.STATE.get("counts") or INJ.assert_counts(n_test=150)
        own_test = os.path.join(ckpt_root, "test_results.json")
        infer = SIB.STATE["test"]
        ckpt_path = os.path.join(ckpt_root, "checkpoints", "best_model.pth")
    else:
        if hasattr(mod, "SummaryWriter"):
            mod.SummaryWriter = RT.NullWriter
            silenced.append("SummaryWriter")
        Cls = getattr(mod, cell["trainer_class"])
        seen = []
        orig_save = Cls.save_checkpoint

        def rec_save(self, epoch, is_best=False, *x, **k):
            seen.append((int(epoch), bool(is_best)))
            return orig_save(self, epoch, is_best, *x, **k)
        Cls.save_checkpoint = rec_save
        t0 = time.time()
        trainer = Cls(ns)
        counts = INJ.assert_counts(n_test=150)
        vram("after model + loaders built")
        if a.construct_only:
            check_access(cell, counts)
            print(f"[n450] CONSTRUCT-ONLY OK {rid}")
            return 0
        sumf = os.path.join(ckpt_root, "n450_train_summary.json")
        if not a.dump_only:
            trainer.train()
            bests = [e for e, b in seen if b]
            spe = R.N_TRAIN // ns.batch_size
            ep = {"epochs_max": ns.epochs, "epochs_run": max(e for e, _ in seen),
                  "best_epoch": bests[-1] if bests else None,
                  "best_val_metric": float(trainer.best_decision_miou),
                  "stopped_on": ("early_stopping" if max(e for e, _ in seen)
                                 < ns.epochs else "epochs_max"),
                  "early_stop_patience": ns.early_stopping_patience,
                  "steps_per_epoch": spe,
                  "steps_to_best": (bests[-1] * spe) if bests else None,
                  "train_wall_s": time.time() - t0}
            with open(sumf, "w") as f:
                json.dump(ep, f, indent=1)
        with open(sumf) as f:
            ep = json.load(f)
        t_train = ep["train_wall_s"]
        timebox.__enter__()
        timebox.phase("trainer.test() -> own test_results.json")
        trainer.test()
        own_test = os.path.join(trainer.output_dir, "test_results.json")
        ckpt_path = os.path.join(trainer.output_dir, "checkpoint_best.pth")
        timebox.phase("inference: run_inference_probs.py (published prob writer)")
        tree_out = os.path.join(ckpt_root, "probs_tree")
        cmd = [PY, "-u", os.path.join(BASE, "python/run_inference_probs.py"),
               "--model_type", arch, "--model_variant", cell["prob_variant"],
               "--test_type", "loco", "--city", city, "--train_res", R.RES_DIR,
               "--test_res", R.RES_DIR, "--checkpoint_path", ckpt_path,
               "--data_root", os.path.join(ns.base_data_root, city, R.RES_DIR),
               "--output_dir", tree_out,
               # 2026-10-28: passed EXPLICITLY. It was previously left to
               # run_inference_probs.py's argparse default, which is also 4, so
               # the number is unchanged -- but an OGLANet score depends on it
               # (DFFM rebuilds 20 unregistered batch-statistic BatchNorms per
               # forward), so it must be stated rather than inherited.
               "--batch_size", str(INFER_BS)]
        ti = time.time()
        subprocess.run(cmd, cwd=BASE, check=True,
                       timeout=max(60, a.post_timeout - 300))
        probs_dir = os.path.join(tree_out, "loco", city, R.RES_DIR, arch,
                                 cell["prob_variant"])
        infer = {"writer": "python/run_inference_probs.py", "cmd": cmd,
                 "precision": "fp32 (no autocast), as the originals' Test_img_probs",
                 "infer_wall_s": time.time() - ti}
    # 2026-10-28 (step 1b) REVERTED, and why.
    # The intent was: for an OGLANet SIB cell, replace the SIB trainer's
    # batch-8 fp16 probability maps with a batch-4 fp32 run_inference_probs.py
    # pass, so the cell is scored at the published OGLANet batch size.
    # That is NOT POSSIBLE with the published inference scripts.
    # `cell["prob_variant"]` is None for every SIB cell by design, because
    # run_inference_probs.py's model-construction block has no SIB variant --
    # its oglanet variants are base/vanilla/fda/isw, segdesic, iim, mrfp_plus
    # and fada. An OGLANet SIB checkpoint is an OGLANetSIB instance that none of
    # those builds, which is exactly why the SIB family writes its own probs.
    # Attempting it crashed job 23273276 with
    #   TypeError: expected str, bytes or os.PathLike object, not NoneType
    # on the None variant. Making it work needs NEW model-loading code for a
    # published architecture, which is an operator decision, not a patch.
    # So OGLANet SIB cells remain scored from the SIB trainer's own pass, and
    # run.json records that batch size honestly rather than claiming 4.

    vr = vram("end of training + test")
    acc = check_access(cell, counts)
    timebox.phase("score + dump (n450_score)")
    import n450_score
    run = {
        "run_id": rid, "dataset": "st", "arm": "n450", "arch": arch,
        "method": method, "mode": "loco", "rung": 450, "n_train": counts["n_train"],
        "n_train_per_source_city": counts["n_train_per_source_city"],
        "n_val": counts["n_val"], "n_test": counts["n_test"], "holdout": city,
        "res": "hr", "seed": R.SEED,
        "seed_source": "set by the n450 wrapper (python, numpy, torch, cuda)",
        "seed_caveat": "the original recorded no seed and its trainer seeds "
                       "nothing: original seed unrecoverable (cf. E-12)",
        "target_access": cell["target_access"],
        "target_access_detail": cell["target_access_detail"],
        "data_access_observed": dict(acc, **{k: counts[k] for k in
                                 ("n_holdout_unlabeled", "fda_loads", "datasets")}),
        "train_list_sha256": counts["train_list_sha256"],
        "val_list_sha256": counts["val_list_sha256"],
        "test_list_sha256": counts["test_list_sha256"],
        "train_list_source": R.SHARED_LISTS, **ep, "train_wall_s": t_train,
        "lr": ns.lr, "batch_size": ns.batch_size,
        "decision_metric": "published trainer's own (eval_boundary_tolerant="
                           f"{getattr(ns, 'eval_boundary_tolerant', None)}, "
                           "tolerant 2px per-image 2-class mIoU), unchanged",
        "reporting_metric": "foreground-only IoU, K=2 (Appendix E)",
        "inference": infer, "peak_vram_gib": vr,
        "inference_batch_size": (ns.batch_size if sib else INFER_BS),
        "inference_batch_size_source": (
            f"the SIB trainer's own test loader at args.batch_size "
            f"({ns.batch_size}), under fp16 autocast; run_inference_probs.py "
            f"cannot load a SIB checkpoint (no SIB model_variant), so a "
            f"batch-{INFER_BS} pass is not available for this family"
            if sib else
            f"run_inference_probs.py --batch_size {INFER_BS}, passed explicitly"),
        "inference_batch_size_why": (
            "the published OGLANet tables were inferred at batch 4 "
            "(run_inference.sh:73, run_inference_probs.sh:77) and OGLANet "
            "scores are batch-size dependent because DFFM rebuilds 20 "
            "unregistered batch-statistic BatchNorms inside forward(); see "
            "runs/dffm_defect/provenance.json"
            if arch == "oglanet" else
            "recorded for completeness; this architecture has no DFFM and its "
            "scoring is unchanged"),
        "config_verbatim": resolved, "config_sha256": CFG.config_sha(resolved),
        "config_diff_vs_original": d, "original_config_path": orig_path,
        "original_config_sha256": orig_sha, "args_json_vs_ckpt_args": argsjson_vs_ckpt,
        "trainer": cell["trainer_path"], "trainer_sha256": trainer_sha,
        "side_effects_disabled": silenced, "ckpt_path": ckpt_path,
        "parent_run_id": None, "legacy_run_dir": cell["legacy_run_dir"],
        "original_run_id": cell["orig_run_id"], "structure_cols": "deferred",
        "git_hash": None, "git_hash_reason": "BASE_PATH working tree is unversioned "
        "for this code (python/{arch}/data/*.py untracked); provenance = shas",
        "cluster": "delta", "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "time_limit": os.environ.get("N450_TIME_LIMIT"), "smoke": a.smoke,
        "dump_only": a.dump_only, "written_utc": utc(),
    }
    run["wall_clock_s"] = time.time() - t_start
    n450_score.score(rid, cell, probs_dir, dump_dir, run,
                     limit=5 if a.smoke else 0, own_test_json=own_test)
    timebox.__exit__(None, None, None)
    if not a.smoke:
        for p in glob.glob(os.path.join(ckpt_root, "**", "checkpoint_epoch_*.pth"),
                           recursive=True):
            raise SystemExit(f"FATAL periodic checkpoint written: {p}")
    print(f"[n450] DONE {rid} wall {time.time()-t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
