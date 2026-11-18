#!/usr/bin/env python3
"""Does OGLANet inference depend on the RNG state at inference time?

THE HYPOTHESIS. `DFFM._apply_downsample_n_times` (oglanet/models/dffm.py:
80-85) calls `self.downsample_conv`, which `__init__` defines as a LAMBDA
FACTORY (dffm.py:34-38) rather than a submodule, so every forward constructs 20
fresh, randomly-initialised, UNREGISTERED Conv2d+BatchNorm2d layers. They are
absent from `state_dict`, so `load_state_dict` cannot restore them, and their
weights come from the torch CPU default generator at the moment of the call.
Under the hypothesis: a fixed seed gives identical passes, an unseeded or
advancing generator gives differing passes, and a training-time pass ran after
training had advanced the generator so a fresh re-run cannot reproduce it.

WHAT THIS SCRIPT CONTROLS, AND NOTHING ELSE: the seed, and the number of
consecutive passes in one process. Both inference paths are IMPORTED and called,
never reimplemented:

  --path run_inference_probs   oglanet_probsource.t1_build / t1_pass, which
                               themselves call run_inference_probs.load_model
                               and ._extract_logits verbatim.
  --path sib_trainer           shadowtransfer.n450_sib.install's `test_once`,
                               the published test_and_save_predictions body,
                               driven through train_oglanet_sib.main() with
                               epochs=0 exactly as n450_sib_noise.py does.

Scoring is the one sanctioned module: shadowtransfer.metrics.per_image_counts +
threshold_and_filter, rolled up with armc_dump.iou_from_counts /
miou_from_counts -- the same calls oglanet_probsource.score_probs makes, and the
per-image roll-up is ASSERTED equal to score_probs' own aggregate so this file
cannot drift from it. For --path sib_trainer the trainer's own
test_results.json is recorded alongside, because that is the field the stored
training-time value (68.870627) and the five-node re-entry value (68.531270)
both live in.

The full 150-image test list runs every pass, never truncated: DFFM's fresh
BatchNorm2d layers are in training mode (bn.training is True because they are
built after model.eval()), so they normalise with BATCH statistics and a
shorter list changes the batches and therefore the measured quantity.

Trains nothing. Edits nothing. Writes no published artifact: every write lands
under --out_dir, and the script refuses if --out_dir resolves inside a cell's
checkpoint, original or dump directory.

  python oglanet_rng_probe.py --path sib_trainer --seed 42 --passes 1 \
      --run_id oglanet-sibc4-loco-r450-chicago-hr-s42-n450 \
      --checkpoint .../checkpoints/best_model.pth --batch_size 8 --fp16 \
      --out_dir ${ST_SCRATCH_ROOT}/runs/oglanet_rng_probe/A-C1a
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = os.environ.get("BASE_PATH") or os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

METRICS_PY = Path(BASE) / "shadowtransfer/metrics.py"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def seed_all(s):
    """Identical to n450_train.py:50-56 and n450_sib_noise.py:51-57."""
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rng_sha():
    """sha256 of the CPU and CUDA generator states. The CPU one is the one that
    matters: dffm.py:81 builds its Conv2d on CPU (device confirmed at
    construction) and only then copies to CUDA at dffm.py:82-83, so the
    initialisers draw from the CPU default generator alone."""
    import torch
    out = {"cpu": hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()).hexdigest()}
    out["cuda"] = (hashlib.sha256(
        torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
        if torch.cuda.is_available() else None)
    return out


def determinism():
    import torch
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "deterministic_algorithms":
            bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


def env_record():
    import torch
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_model": (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None),
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "determinism": determinism(),
    }


def per_image_rows(probs_by_stem, gt_by_stem):
    """One row per image from the sanctioned metric module. No new arithmetic:
    threshold_and_filter + per_image_counts are metrics.py's, iou_from_counts
    and miou_from_counts are armc_dump's, and these are the same four calls
    oglanet_probsource.score_probs makes."""
    from shadowtransfer.armc_dump import iou_from_counts, miou_from_counts
    from shadowtransfer.metrics import per_image_counts, threshold_and_filter
    rows = []
    for s in sorted(probs_by_stem):
        p = np.asarray(probs_by_stem[s], dtype=np.float32)
        pred = threshold_and_filter(p, min_pixels=10)
        c = per_image_counts(pred, gt_by_stem[s], boundary_tolerance=2)
        fg = iou_from_counts(c["tp_tol"], c["fp_tol"], c["fn_tol"])
        two = miou_from_counts(c["tp_tol"], c["fp_tol"],
                               c["tn_tol"], c["fn_tol"])[2]
        rows.append({
            "image_id": s,
            "tp_tol": int(c["tp_tol"]), "fp_tol": int(c["fp_tol"]),
            "fn_tol": int(c["fn_tol"]), "tn_tol": int(c["tn_tol"]),
            "tp_str": int(c["tp_str"]), "fp_str": int(c["fp_str"]),
            "fn_str": int(c["fn_str"]), "tn_str": int(c["tn_str"]),
            "n_valid_px": int(c["n_valid_px"]),
            "n_gt_fg_valid": int(np.asarray(gt_by_stem[s])[
                c["valid_mask"].astype(bool)].sum()),
            "fg_iou_k2_pct": float(fg * 100.0),
            "miou_k2_twoclass_pct": float(two),
        })
    return rows


def write_pass(pass_dir, rows, probs, summary):
    """perimage_probe.parquet + probs.npz + summary.json.

    Named perimage_probe.parquet, NOT perimage.parquet: this is a diagnostic
    probe, not a CONTRACT.md section-3 dump, and must not be picked up as one
    by anything that globs for the contract name.
    """
    import pandas as pd
    os.makedirs(pass_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(
        os.path.join(pass_dir, "perimage_probe.parquet"), index=False)
    np.savez(os.path.join(pass_dir, "probs.npz"),
             **{s: probs[s] for s in sorted(probs)})
    with open(os.path.join(pass_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)


def _agg_check(rows, probs, gt, tag):
    """The per-image roll-up must equal oglanet_probsource.score_probs exactly.
    This is the guard that keeps this file from drifting away from the
    sanctioned aggregate."""
    from oglanet_probsource import score_probs
    ref = score_probs(probs, gt)
    mine = {
        "fg_iou_k2_pct": float(np.mean([r["fg_iou_k2_pct"] for r in rows])),
        "miou_k2_twoclass_pct":
            float(np.mean([r["miou_k2_twoclass_pct"] for r in rows])),
        "n_images": len(rows),
    }
    for k in ("fg_iou_k2_pct", "miou_k2_twoclass_pct"):
        if abs(mine[k] - ref[k]) > 1e-9:
            raise SystemExit(
                f"FATAL {tag}: per-image roll-up {k}={mine[k]!r} != "
                f"score_probs {ref[k]!r}. This file has drifted from the "
                f"sanctioned aggregate; fix it before trusting any number.")
    if mine["n_images"] != ref["n_images"]:
        raise SystemExit(f"FATAL {tag}: n_images {mine} vs {ref}")
    return ref


# ===========================================================================
# PATH B -- run_inference_probs.py, via oglanet_probsource's imported wrappers
# ===========================================================================

def run_path_probs(a, out, rec):
    import torch
    from oglanet_probsource import (t1_build, t1_pass, resolve_ckpt,
                                    load_gt_direct, delta_stats)

    if a.fp16:
        raise SystemExit(
            "REFUSED: --fp16 with --path run_inference_probs. That script has "
            "no autocast anywhere (run_inference_probs.py:332-393); its "
            "published regime is fp32. Inventing an AMP variant would add a "
            "second variable to an RNG experiment.")

    ckpt, root, tried = resolve_ckpt(a.city, a.res_dir)
    if os.path.realpath(str(ckpt)) != os.path.realpath(a.checkpoint):
        raise SystemExit(
            f"REFUSED: --checkpoint {a.checkpoint} is not the cell's resolved "
            f"checkpoint {ckpt}.\n  " + "\n  ".join(tried))
    rec["checkpoint"] = {"path": str(ckpt), "root": root,
                         "sha256": sha256_file(ckpt),
                         "size_bytes": os.path.getsize(ckpt)}
    print(f"[ckpt] {ckpt}  sha {rec['checkpoint']['sha256'][:16]}", flush=True)

    model, device, loader, aa, R = t1_build(
        a.city, a.res_dir, ckpt, 384, a.batch_size, a.num_workers)
    rec["build"] = {
        "builder": "oglanet_probsource.t1_build -> "
                   "run_inference_probs.load_model (strict=False) + "
                   "ShadowDatasetEnhanced(task_id=2, augment=False)",
        "pass_fn": "oglanet_probsource.t1_pass -> model(images) + "
                   "run_inference_probs._extract_logits + softmax[:,1] -> f16",
        "batch_size": a.batch_size, "num_workers": a.num_workers,
        "precision": "fp32 (no autocast; as published)",
        "n_dataset": len(loader.dataset), "shuffle": False,
    }
    print(f"[build] {len(loader.dataset)} test images, batch {a.batch_size}, "
          f"fp32", flush=True)
    if len(loader.dataset) != 150:
        raise SystemExit(f"REFUSED: dataset is {len(loader.dataset)} images, "
                         f"not 150. The list must never be truncated.")

    gt = None
    pass0 = None
    for k in range(a.passes):
        before = rng_sha()
        seed0 = torch.initial_seed()
        t0 = time.time()
        probs = t1_pass(model, device, loader, aa, R)
        wall = time.time() - t0
        after = rng_sha()
        if gt is None:
            gt = load_gt_direct(a.city, a.res_dir, sorted(probs))
        rows = per_image_rows(probs, gt)
        agg = _agg_check(rows, probs, gt, f"pass{k}")
        s = {
            "pass": k, "wall_s": wall,
            "fg_iou_k2_pct": agg["fg_iou_k2_pct"],
            "miou_k2_twoclass_pct": agg["miou_k2_twoclass_pct"],
            "n_images": agg["n_images"],
            "torch_initial_seed": int(seed0),
            "rng_sha_before": before, "rng_sha_after": after,
            "rng_advanced_during_pass": before != after,
            "vs_pass0": (None if k == 0 else delta_stats(pass0, probs)),
            "metric_module_sha256": rec["metric_module_sha256"],
            **env_record(),
        }
        if torch.cuda.is_available():
            s["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        write_pass(os.path.join(out, f"pass{k}"), rows, probs, s)
        rec["passes"].append(s)
        if k == 0:
            pass0 = probs
        print(f"[pass {k}] two-class mIoU K=2 = {s['miou_k2_twoclass_pct']:.6f}"
              f"   fg IoU K=2 = {s['fg_iou_k2_pct']:.6f}   "
              f"initial_seed {seed0}   rng advanced "
              f"{s['rng_advanced_during_pass']}   wall {wall:.1f}s", flush=True)
    return rec


# ===========================================================================
# PATH A -- the SIB trainer's own test pass, via n450_sib.install's test_once
# ===========================================================================

def run_path_sib(a, out, rec):
    """Drives train_oglanet_sib.main() with epochs=0, exactly as
    n450_sib_noise.py:117-166 does, with ONE change: test_and_save_predictions
    is called --passes times in a row instead of once, with no reseed between.
    The pass body itself is n450_sib.install's `test_once`, untouched."""
    import torch
    from shadowtransfer import armc_runtime as RT
    from shadowtransfer import n450_roster as RS
    from shadowtransfer import n450_inject as INJ
    from shadowtransfer import n450_config as CFG

    if not a.fp16:
        raise SystemExit(
            "REFUSED: --path sib_trainer without --fp16. Every C4clean "
            "original trained and tested under autocast (use_amp=True); "
            "forcing fp32 here would add a second variable.")

    cell = RS.by_run_id().get(a.run_id)
    if cell is None or cell["method"] != "sibc4":
        raise SystemExit(f"REFUSED: {a.run_id} is not an n450 sibc4 cell.")
    src_ckpt = os.path.join(cell["ckpt_dir"], "checkpoints", "best_model.pth")
    if os.path.realpath(src_ckpt) != os.path.realpath(a.checkpoint):
        raise SystemExit(f"REFUSED: --checkpoint {a.checkpoint} != cell "
                         f"checkpoint {src_ckpt}")
    for guarded in (cell["dump_dir"], cell["ckpt_dir"], cell["orig_dir"]):
        g = os.path.normpath(guarded)
        if os.path.normpath(out) == g or os.path.normpath(out).startswith(g + os.sep):
            raise SystemExit(f"REFUSED: --out_dir is inside {guarded}")
    rec["checkpoint"] = {"path": src_ckpt, "sha256": sha256_file(src_ckpt),
                         "size_bytes": os.path.getsize(src_ckpt)}
    rec["cell"] = {k: cell[k] for k in
                   ("arch", "method", "city", "legacy_run_dir",
                    "trainer_module", "target_access")}
    print(f"[ckpt] {src_ckpt}  sha {rec['checkpoint']['sha256'][:16]}",
          flush=True)

    RT.preflight_write_targets(
        BASE, [os.path.join(out, f) for f in
               ("probe.json", "test_results.json", ".DONE")],
        label=f"rngprobe-{a.run_id}")

    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    link = os.path.join(out, "checkpoints", "best_model.pth")
    if not os.path.exists(link):
        os.symlink(src_ckpt, link)              # read-only use; never copied
    probs_dir = os.path.join(out, "_probs_scratch")

    tree = os.path.join(BASE, cell["arch"])
    sys.path.insert(0, tree)
    os.chdir(tree)
    orig, orig_path, _sha = RT.read_with_retry(
        120, f"original config of {cell['legacy_run_dir']}",
        lambda: CFG.read_orig_config(cell["orig_dir"], cell["config_source"]))
    mod, trainer_sha = CFG.load_trainer(tree, cell["trainer_module"],
                                        os.path.join(out, "_modsrc"))
    ns, _dests = CFG.resolve(mod, "parse_args", orig,
                             {"output_dir": out, "epochs": 0})
    permitted = dict(CFG.PERMITTED_CHANGED)
    permitted.update(CFG.DUMP_ONLY_CHANGED)
    d = CFG.diff(orig, dict(vars(ns)), permitted)
    print(f"[config] baseline {orig_path}")
    for k, v in d["changed"].items():
        print(f"  CHANGED {k}: {v['orig']!r} -> {v['new']!r}")
    print(f"  VERDICT {d['verdict']}", *d["refusals"], sep="\n    ")
    if d["verdict"] != "PERMITTED":
        return None
    if int(ns.epochs) != 0:
        raise SystemExit(f"REFUSED: epochs resolved to {ns.epochs}, not 0.")
    if int(ns.batch_size) != a.batch_size:
        raise SystemExit(
            f"REFUSED: --batch_size {a.batch_size} != the cell's published "
            f"{ns.batch_size}. Changing it changes the DFFM batch statistics, "
            f"which is a second variable.")
    if not bool(ns.use_amp):
        raise SystemExit("REFUSED: cell has use_amp=False but --fp16 was given.")
    rec["build"] = {
        "builder": "n450_config.load_trainer(train_oglanet_sib) + main() with "
                   "epochs=0; best_model.pth loaded at "
                   "train_oglanet_sib.py:1770-1774",
        "pass_fn": "n450_sib.install -> test_once (the published "
                   "test_and_save_predictions eval body + f16 probs)",
        "batch_size": int(ns.batch_size),
        "precision": f"fp16 autocast (use_amp={bool(ns.use_amp)})",
        "trainer_sha256": trainer_sha, "config_baseline": str(orig_path),
    }

    INJ.configure(RS.SHARED_LISTS, cell["city"], cell["source_cities"],
                  ns.base_data_root, ns.resolution)
    import data.dataset_sib as DS
    INJ.wrap_class(DS.ShadowDatasetSIB, "ShadowDatasetSIB")
    silenced = []
    for name in list(vars(mod)):
        if name.startswith("plot_") or name in ("save_best_worst_visualizations",
                                                "save_best_worst_predictions"):
            setattr(mod, name, lambda *x, **k: None)
            silenced.append(name)

    # ---- the only thing this script controls: the seed -------------------
    if a.seed is not None:
        seed_all(a.seed)
        print(f"[seed] seed_all({a.seed}) -- same call the production path "
              f"makes (n450_train.py:226)", flush=True)
    else:
        print("[seed] NOT SEEDED -- torch's default generator keeps its "
              "per-process nondeterministic seed", flush=True)

    timebox = RT.PhaseTimeout(a.post_timeout, out, label="rng-probe")
    from shadowtransfer import n450_sib as SIB
    silenced += SIB.install(mod, ns, probs_dir, timebox)
    inner = mod.test_and_save_predictions          # = test_once, untouched

    from oglanet_probsource import load_gt_direct, delta_stats
    state = {"gt": None, "pass0": None}

    def multipass(model, loader, device, args, logger):
        last = None
        for k in range(a.passes):
            before = rng_sha()
            seed0 = torch.initial_seed()
            t0 = time.time()
            last = inner(model, loader, device, args, logger)
            wall = time.time() - t0
            after = rng_sha()
            with open(os.path.join(args.output_dir, "test_results.json")) as f:
                tr = json.load(f)
            probs = {Path(f).stem: np.load(os.path.join(probs_dir, f))
                     for f in sorted(os.listdir(probs_dir))
                     if f.endswith(".npy")}
            if state["gt"] is None:
                state["gt"] = load_gt_direct(cell["city"], RS.RES_DIR,
                                             sorted(probs))
            rows = per_image_rows(probs, state["gt"])
            agg = _agg_check(rows, probs, state["gt"], f"pass{k}")
            s = {
                "pass": k, "wall_s": wall,
                "fg_iou_k2_pct": agg["fg_iou_k2_pct"],
                "miou_k2_twoclass_pct": agg["miou_k2_twoclass_pct"],
                "n_images": agg["n_images"],
                "trainer_miou_k2_twoclass": tr["tolerant_2px"]["mIOU"],
                "trainer_fg_iou_k2": tr["tolerant_2px"]["Shadow_IOU"],
                "trainer_strict_miou": tr["strict"]["mIOU"],
                "trainer_num_images": tr["num_images"],
                "inference_block": dict(SIB.STATE.get("test") or {}),
                "torch_initial_seed": int(seed0),
                "rng_sha_before": before, "rng_sha_after": after,
                "rng_advanced_during_pass": before != after,
                "vs_pass0": (None if k == 0
                             else delta_stats(state["pass0"], probs)),
                "metric_module_sha256": rec["metric_module_sha256"],
                **env_record(),
            }
            if torch.cuda.is_available():
                s["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
            pd_dir = os.path.join(out, f"pass{k}")
            write_pass(pd_dir, rows, probs, s)
            with open(os.path.join(pd_dir, "test_results.json"), "w") as f:
                json.dump(tr, f, indent=1)
            rec["passes"].append(s)
            if k == 0:
                state["pass0"] = probs
            print(f"[pass {k}] trainer two-class mIoU K=2 = "
                  f"{tr['tolerant_2px']['mIOU']:.6f}   trainer fg IoU K=2 = "
                  f"{tr['tolerant_2px']['Shadow_IOU']:.6f}   "
                  f"metric-module fg = {agg['fg_iou_k2_pct']:.6f}   "
                  f"initial_seed {seed0}   wall {wall:.1f}s", flush=True)
        return last

    mod.test_and_save_predictions = multipass
    print(f"[silenced] {sorted(set(silenced))}", flush=True)
    mod.main()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True,
                    choices=["sib_trainer", "run_inference_probs"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", required=True,
                    help="an int, or the literal 'none' for no seeding at all")
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--condition", default="",
                    help="free-text label, e.g. C1a; recorded, not acted on")
    # path-specific identifiers
    ap.add_argument("--run_id", help="required for --path sib_trainer")
    ap.add_argument("--city", help="required for --path run_inference_probs")
    ap.add_argument("--res_dir", default="highres", choices=["highres", "midres"])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--post_timeout", type=int, default=1800)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    a.seed = None if str(a.seed).lower() == "none" else int(a.seed)
    if a.passes < 1:
        raise SystemExit("REFUSED: --passes must be >= 1")
    if a.path == "sib_trainer" and not a.run_id:
        raise SystemExit("REFUSED: --path sib_trainer needs --run_id")
    if a.path == "run_inference_probs" and not a.city:
        raise SystemExit("REFUSED: --path run_inference_probs needs --city")

    out = os.path.abspath(a.out_dir)
    t_start = time.time()
    rec = {
        "probe": "oglanet_rng_probe",
        "probe_sha256": sha256_file(os.path.abspath(__file__)),
        "written_utc": utc(),
        "path": a.path, "condition": a.condition,
        "seed": a.seed, "passes": a.passes,
        "batch_size": a.batch_size, "fp16": bool(a.fp16),
        "out_dir": out,
        "metric_module_sha256": sha256_file(METRICS_PY),
        "metric_module": str(METRICS_PY),
        "hypothesis": "OGLANet outputs depend on the RNG state at inference "
                      "time, via the ~20 unregistered Conv2d+BatchNorm2d "
                      "layers DFFM builds inside forward() "
                      "(dffm.py:34-38, 80-85)",
        "trains_nothing": True, "writes_published_artifact": False,
        "passes_detail_note": "one entry per consecutive pass, in order; no "
                              "reseed happens between passes",
        "passes": [],
    }

    print("=" * 78)
    print(f"oglanet_rng_probe   path={a.path}  condition={a.condition or '-'}  "
          f"seed={a.seed}  passes={a.passes}   ({utc()})")
    print(f"  checkpoint  {a.checkpoint}")
    print(f"  batch       {a.batch_size}   fp16={bool(a.fp16)}")
    print(f"  out_dir     {out}")
    print(f"  metrics.py  {rec['metric_module_sha256'][:16]}")
    print(f"  probe sha   {rec['probe_sha256'][:16]}")
    print("=" * 78, flush=True)

    if a.dry_run:
        print("[dry-run] arguments resolved only; nothing built, nothing "
              "written")
        return 0

    os.makedirs(out, exist_ok=True)

    if a.path == "run_inference_probs":
        # Seeded at process start: the production script seeds nowhere
        # (run_inference_probs.py has no manual_seed), so the only honest
        # insertion point is before load_model builds the model.
        if a.seed is not None:
            seed_all(a.seed)
            print(f"[seed] seed_all({a.seed}) before any model construction",
                  flush=True)
        else:
            print("[seed] NOT SEEDED -- torch's default generator keeps its "
                  "per-process nondeterministic seed", flush=True)
        rec["seed_site"] = "process start, before t1_build"
        rec = run_path_probs(a, out, rec)
    else:
        # Seeded immediately before main(), which is where n450_train.py:226
        # seeds relative to its own main() call at :239.
        rec["seed_site"] = "immediately before train_oglanet_sib.main()"
        rec = run_path_sib(a, out, rec)

    if rec is None:
        return 4
    rec["total_wall_s"] = time.time() - t_start
    rec["n_passes_recorded"] = len(rec["passes"])
    with open(os.path.join(out, "probe.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    open(os.path.join(out, ".DONE"), "w").close()
    print(f"\n[oglanet_rng_probe] DONE {a.condition or a.path} "
          f"{rec['n_passes_recorded']} pass(es) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
