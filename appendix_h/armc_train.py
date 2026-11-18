"""
Arm C — the scaling ladder. One cell: train one (arch, mode, rung, holdout,
seed) at high resolution, then emit its CONTRACT.md dump.

Answers zNBu-W3 and AC1: is the leave-one-city-out gap caused by geography, or
by having only a few hundred training images?

What this script changes relative to the published protocol
-----------------------------------------------------------
The TRAINING-SET SIZE, and nothing else. The model, loss, optimizer, scheduler,
augmentation, early-stopping rule and model-selection metric are the published
architecture's own, reached by constructing that architecture's own Trainer from
its own published args.json and then replacing only its dataloaders.

Two deviations from the published cells are forced by the arm spec and are
recorded in run.json rather than hidden:

  1. n_val = 150 at every rung (REVIEWS.md arm C: "Validation stays at 150
     images at every rung. If val scales with train, model selection quality
     moves and the ladder is confounded."). The published LOCO cells validate on
     300 (both source cities' full val pools); the published upper-bound cells
     validate on 150 already.
  2. Training-set membership is an explicit seeded file list, not the
     `samples_per_dir` sampler the published loaders use.

Selection rule
--------------
For a fixed seed, one permutation per (city, split) is drawn from a stable
SHA-256-derived seed and every rung takes a PREFIX of it. Rungs are therefore
nested: the rung-225 set is a subset of the rung-450 set is a subset of the
rung-900 set. That removes between-rung sampling noise from the ladder. The
resolved file lists are hashed into run.json (`train_list_sha256` etc.) so a
reader can verify exactly which images a cell saw.

  LOCO  rung 225 -> 113 / 112 across the two source cities (first-listed city
                    takes the extra image)
        rung 450 -> 225 / 225        (not trained here; see armc_r450.py)
        rung 900 -> 450 / 450        (the full source pools)
  upper rung 225 -> 225 from the held-out city's own train pool
        rung 450 -> 450, the published upper bound (not trained here)

Metric
------
Arm C's headline is FOREGROUND-ONLY IoU at K = 2. It is written by
shadowtransfer.armc_dump, which is the only metric implementation any arm-C job
touches. Raw counts at K = 0 and K = 5 ride along in the parquet.

Usage
-----
    python armc_train.py --run_id mamnet-vanilla-loco-r225-phoenix-hr-s0 \
        --arch mamnet --mode loco --rung 225 --holdout phoenix --seed 0
    ... --dry_run                       resolve and print, write nothing
    ... --smoke_epochs 2 --smoke_test 5 --out_root <scratch>   stage-1 smoke
    ... --dump_only [--ckpt <path>]     no training: load an existing
                                        checkpoint_best.pth and run only the
                                        inference + dump phase

Worker side effects that are switched off
-----------------------------------------
The published trainers draw matplotlib figures and open a tensorboard writer
inside train(). This box does not build figures (CLAUDE.md; CONVENTIONS.md puts
them on the SSH box), and the figure call is what hung job 20540648 for 90
minutes on a 0-byte metrics_curves.png. Both are disabled here, from the
wrapper, by rebinding those names inside the architecture's own train module
after import — the published files are not edited. Periodic checkpointing is
disabled the same way, through the published config's own `save_freq`, so a cell
keeps only checkpoint_best.pth.

Everything after training — inference, metrics, the dump — runs under a hard
timeout, so a hang there fails loudly instead of eating the wall clock.
"""

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_dump                        # noqa: E402
from shadowtransfer.armc_dump import sha256_file            # noqa: E402
from shadowtransfer import armc_runtime                     # noqa: E402

# ---------------------------------------------------------------------------
# Fixed project facts. LOCO_FOLDS is the published mapping, copied from
# python/{arch}/data/dataset.py, where it is byte-identical in all three trees.
# ---------------------------------------------------------------------------

LOCO_FOLDS = {
    0: {"train": ["chicago", "miami"],   "test": "phoenix"},
    1: {"train": ["chicago", "phoenix"], "test": "miami"},
    2: {"train": ["miami",   "phoenix"], "test": "chicago"},
}
FOLD_OF_HOLDOUT = {"phoenix": 0, "miami": 1, "chicago": 2}
RES_DIR = {"hr": "highres", "mr": "midres"}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

ARCH_DIR = {"mamnet": "mamnet", "oglanet": "oglanet", "dinov3": "dinov3"}
ARCH_TRAIN_MODULE = {"mamnet": "train", "oglanet": "train", "dinov3": "train_dinov3"}

DINOV3_WEIGHTS = BASE / "python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

N_VAL = 150            # fixed at every rung, arm C constraint
N_TEST = 150

# ---------------------------------------------------------------------------
# Early stopping: ONE recipe for every architecture at every rung, so that the
# rung is the only variable on the ladder.
#
#     patience = 20 epochs,  epochs capped at 100,  best checkpoint retained
#
# Operator decision, 2026-07-29. It is Table 23's LOCO value and is never below
# what any published cell used, so no cell is stopped earlier than the paper's
# own runs would have stopped it.
#
# Recorded fact, not an objection: every published args.json in this tree
# carries early_stopping_patience=10, for all three architectures (measured
# 2026-07-29). Table 23 states 20. Arm C applies 20 uniformly. At rung 450 that
# is twice the patience the published cells actually ran with, so arm C's r450
# cells are NOT a like-for-like reproduction of the published configuration on
# this axis. That is the intended trade: a rung-comparable rule beats matching
# a per-cell value that itself differs from the paper's table.
#
# Step-equivalents at batch_size 8, for reading steps_to_best across rungs:
#     r225   28 steps/epoch ->  560 steps
#     r450   56 steps/epoch -> 1120 steps
#     r900  112 steps/epoch -> 2240 steps
# The budget in STEPS therefore still scales with the rung. What the flat
# epoch rule guarantees is that no rung is stopped sooner than the published
# rule would have stopped it.
ARMC_PATIENCE_EPOCHS = 20
ARMC_EPOCHS_MAX = 100


def armc_patience_epochs(rung, batch_size):
    """(patience_epochs, steps_per_epoch). Patience is flat; steps_per_epoch is
    returned for the runtime assertion and for run.json."""
    spe = rung // batch_size
    if spe <= 0:
        raise SystemExit(f"FATAL rung {rung} < batch_size {batch_size}")
    return ARMC_PATIENCE_EPOCHS, spe


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------

def stable_rng(*parts):
    """A generator seeded from a stable hash of its parts.

    Python's builtin hash() is salted per process, so it cannot be used: the
    same cell would draw a different training set on a resubmit.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def city_split_files(city, res_dir, split):
    """Sorted filenames present in BOTH images/ and masks/ for one city split."""
    root = BASE / "data" / "Final_data_test" / city / res_dir / split
    img_dir, mask_dir = root / "images", root / "masks"
    if not img_dir.is_dir():
        raise SystemExit(f"FATAL missing image dir {img_dir}")
    names = sorted(f.name for f in img_dir.iterdir()
                   if f.suffix.lower() in IMG_EXTS and (mask_dir / f.name).exists())
    if not names:
        raise SystemExit(f"FATAL no paired files under {root}")
    return names


def take_prefix(names, n, city, split, seed):
    """Seeded permutation, then the first n — so rungs nest for a fixed seed."""
    if n > len(names):
        raise SystemExit(f"FATAL requested {n} from {city}/{split}, only "
                         f"{len(names)} paired files exist")
    perm = stable_rng("armC", city, split, seed).permutation(len(names))
    return sorted(names[i] for i in perm[:n])


def split_rung(rung, n_cities):
    """Per-source-city train counts. 225 -> [113, 112]; the extra image goes to
    the first city in the fold's own listed order."""
    base, extra = divmod(rung, n_cities)
    return [base + (1 if i < extra else 0) for i in range(n_cities)]


def resolve_selection(mode, holdout, rung, res_dir, seed):
    """Return (train, val, test) as lists of (city, filename), plus provenance."""
    if mode == "loco":
        fold = FOLD_OF_HOLDOUT[holdout]
        src = LOCO_FOLDS[fold]["train"]
        if LOCO_FOLDS[fold]["test"] != holdout:
            raise SystemExit("FATAL fold table disagrees with holdout")
        per_city = split_rung(rung, len(src))
        train = [(c, f) for c, n in zip(src, per_city)
                 for f in take_prefix(city_split_files(c, res_dir, "train"),
                                      n, c, "train", seed)]
        n_val_each = split_rung(N_VAL, len(src))
        val = [(c, f) for c, n in zip(src, n_val_each)
               for f in take_prefix(city_split_files(c, res_dir, "val"),
                                    n, c, "val", seed)]
        train_cities, n_per_city = src, dict(zip(src, per_city))
    elif mode == "upper":
        train = [(holdout, f) for f in take_prefix(
            city_split_files(holdout, res_dir, "train"), rung,
            holdout, "train", seed)]
        val = [(holdout, f) for f in take_prefix(
            city_split_files(holdout, res_dir, "val"), N_VAL,
            holdout, "val", seed)]
        train_cities, n_per_city = [holdout], {holdout: rung}
    else:
        raise SystemExit(f"FATAL unknown mode {mode}")

    test = [(holdout, f) for f in city_split_files(holdout, res_dir, "test")]
    return train, val, test, train_cities, n_per_city


def list_sha256(pairs):
    joined = "\n".join(f"{c}/{f}" for c, f in pairs).encode()
    return hashlib.sha256(joined).hexdigest()


def city_roots(pairs, res_dir):
    """Unique city roots, in first-appearance order."""
    seen, roots = set(), []
    for c, _ in pairs:
        if c not in seen:
            seen.add(c)
            roots.append(str(BASE / "data" / "Final_data_test" / c / res_dir))
    return roots


def seed_everything(seed):
    """Seed every RNG a cell touches, and report where each was set.

    The returned dict goes into run.json verbatim. This file's docstring once
    described behaviour the file did not implement; recording the resolved
    sites means a reader checks the artifact instead of trusting prose.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {
        "python_random": f"random.seed({seed})",
        "numpy": f"np.random.seed({seed})",
        "torch_cpu": f"torch.manual_seed({seed})",
        "torch_cuda": f"torch.cuda.manual_seed_all({seed})",
        "dataloader_shuffle": (f"torch.Generator().manual_seed({seed}), passed "
                               f"as DataLoader(generator=)"),
        "dataloader_workers": ("dl_worker_init seeds python random and numpy "
                               "from torch.initial_seed() in each worker"),
        "file_selection": ("stable_rng(): SHA-256 of ('armC', city, split, "
                           f"{seed}), independent of the global RNGs so a "
                           "resubmit draws the same images"),
    }


def dl_worker_init(worker_id):
    """Seed python random and numpy inside a dataloader worker process.

    torch already gives each worker its own torch RNG, seeded base_seed +
    worker_id from the parent's global torch RNG, so torch-side ops are
    reproducible without this. python `random` and numpy are not covered on
    every torch version, and the published augmentation pipelines draw from
    both. torch.initial_seed() inside a worker returns that worker's own seed,
    so each worker gets a distinct, run-seed-derived stream.
    """
    import torch
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s)
    np.random.seed(s)


# ---------------------------------------------------------------------------
# Dataset construction — the arch's own class, then filtered to our file list
# ---------------------------------------------------------------------------

def build_dataset(arch, pairs, split, res_dir, img_size, augment):
    """Construct the architecture's own dataset over the relevant city roots,
    then keep exactly the selected files. Filtering after construction (rather
    than reimplementing the loader) guarantees every transform, augmentation and
    channel is the published one."""
    roots = city_roots(pairs, res_dir)

    if arch == "mamnet":
        from data.dataset_enhanced import ShadowDatasetEnhanced
        ds = ShadowDatasetEnhanced(root_dir=roots, split=split, img_size=img_size,
                                   task_id=2, augment=augment, use_fda=False)
    elif arch == "oglanet":
        from data.dataset_enhanced import ShadowDatasetEnhanced
        ds = ShadowDatasetEnhanced(roots, split=split, img_size=img_size,
                                   augment=augment, use_contrast=True)
    elif arch == "dinov3":
        from data.dataset import Dataset as ShadowDatasetPlain
        ds = ShadowDatasetPlain(roots, split=split, img_size=img_size,
                                augment=augment)
    else:
        raise SystemExit(f"FATAL unknown arch {arch}")

    want = {str(BASE / "data" / "Final_data_test" / c / res_dir / split / "images" / f)
            for c, f in pairs}
    keep = [i for i, p in enumerate(ds.img_paths) if os.path.abspath(p) in want]
    if len(keep) != len(want):
        raise SystemExit(
            f"FATAL {split} selection resolved {len(keep)} of {len(want)} files; "
            f"dataset saw {len(ds.img_paths)} candidates under {roots}")
    ds.img_files = [ds.img_files[i] for i in keep]
    ds.img_paths = [ds.img_paths[i] for i in keep]
    ds.mask_paths = [ds.mask_paths[i] for i in keep]
    return ds


# ---------------------------------------------------------------------------
# Published config
# ---------------------------------------------------------------------------

def legacy_run_dir(arch, mode, holdout, res_dir):
    if mode == "loco":
        return f"{arch}_loco_holdout_{holdout}_{res_dir}_1"
    return f"{arch}_{holdout}_{res_dir}_1"


def published_config(arch, mode, holdout, res_dir):
    p = BASE / "data" / arch / "outputs" / legacy_run_dir(arch, mode, holdout, res_dir) / "args.json"
    if not p.exists():
        raise SystemExit(f"FATAL no published config at {p}")
    return json.loads(p.read_text()), p


def arch_parser_defaults(archtrain, module_name):
    """The architecture's own argparse defaults, as a dict.

    A published args.json records only the flags that existed when it was
    written. DINOv3's trainer reads `args.freeze_backbone`, which was added to
    its parser after its published cells ran, so a config built from args.json
    alone lacks it and Trainer.__init__ dies with AttributeError -- which is
    what killed all 18 DINOv3 cells of the first rung-225 grid.

    Layering these defaults UNDERNEATH the published config restores exactly
    what the published run itself resolved to, and changes nothing wherever
    args.json is complete. Measured 2026-07-28: mamnet 29/29 keys and oglanet
    26/26 present, so this is a no-op for both; DINOv3 is short exactly one,
    freeze_backbone=False, i.e. full fine-tuning. That value matters -- a
    frozen backbone is arm E, and arm C varies the training-set size and
    nothing else.

    get_args() reads sys.argv, so argv is swapped for the call.
    """
    saved = sys.argv
    sys.argv = [module_name]
    try:
        return vars(archtrain.get_args())
    finally:
        sys.argv = saved


def newest_epoch_checkpoint(out_dir):
    cks = sorted(Path(out_dir).glob("checkpoint_epoch_*.pth"),
                 key=lambda q: int(q.stem.split("_")[-1]))
    return str(cks[-1]) if cks else None


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--arch", required=True, choices=sorted(ARCH_DIR))
    ap.add_argument("--mode", required=True, choices=["loco", "upper"])
    ap.add_argument("--rung", required=True, type=int, choices=[225, 450, 900])
    ap.add_argument("--holdout", required=True, choices=sorted(FOLD_OF_HOLDOUT))
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke_epochs", type=int, default=0,
                    help="stage-1 smoke: cap epochs")
    ap.add_argument("--smoke_test", type=int, default=0,
                    help="stage-1 smoke: cap test images")
    ap.add_argument("--force", action="store_true",
                    help="re-run a cell that already carries .DONE, overwriting "
                         "its dump. For a deliberate re-run under a changed "
                         "rule, not for filling holes in a grid.")
    ap.add_argument("--timeout_s", type=int, default=1200,
                    help="hard time-box over everything after training")
    ap.add_argument("--out_root", default=None,
                    help="override dumps/ root; every truncated run must set it "
                         "so a smoke dump never lands on the real path")
    args = ap.parse_args()

    t0 = time.time()
    smoke = bool(args.smoke_epochs or args.smoke_test)
    if smoke and args.out_root is None:
        raise SystemExit("FATAL --smoke_* requires --out_root")

    res_dir = RES_DIR[args.res]
    method = "vanilla" if args.mode == "loco" else "none"
    expect_id = (f"{args.arch}-{method}-{args.mode}-r{args.rung}-"
                 f"{args.holdout}-{args.res}-s{args.seed}")
    if args.run_id != expect_id:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {expect_id}")

    dump_dir = (Path(args.out_root) if args.out_root else BASE / "dumps") / "st" / args.run_id
    if (dump_dir / ".DONE").exists():
        if not args.force:
            print(f"[{args.run_id}] .DONE exists at {dump_dir} — nothing to do")
            return
        print(f"[{args.run_id}] .DONE exists at {dump_dir} but --force is set "
              f"— re-running and OVERWRITING the existing dump", flush=True)

    # Refuse a quarantined output BEFORE spending a training run on it. Job
    # 20556487 trained for 28 minutes and then hung 1200 s writing its parquet
    # to a path an earlier SIGKILL had poisoned. This check costs milliseconds
    # and turns that into an immediate, named failure.
    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet",
        dump_dir / "probs.npz",
        dump_dir / "run.json",
        dump_dir / ".DONE",
    ], label="dump")

    # ---- resolve the data selection before touching a GPU --------------------
    train_pairs, val_pairs, test_pairs, train_cities, n_per_city = \
        resolve_selection(args.mode, args.holdout, args.rung, res_dir, args.seed)

    # The published config for mamnet/loco/chicago is itself a quarantined
    # path: job 20551713 hung writing to it. Reads still work, so this is
    # time-boxed rather than refused -- a stuck read fails in a minute with a
    # name attached instead of eating the cell's wall clock.
    with armc_runtime.ReadGuard(60, "published config"):
        cfg, cfg_path = published_config(args.arch, args.mode, args.holdout, res_dir)
    # A truncated run must not train into the real cell's checkpoint directory:
    # its 2-epoch checkpoint_best.pth would otherwise sit where the real cell
    # writes, and a real run that died before its first improvement would resume
    # or be evaluated off the smoke's weights.
    out_dir_parent = (Path(args.out_root) / "train" / args.run_id if smoke
                      else BASE / "data" / args.arch / "outputs" / "armC" / args.run_id)

    print("=" * 78)
    print(f"ARM C  run_id           {args.run_id}")
    print(f"  arch/mode/rung/seed   {args.arch} / {args.mode} / {args.rung} / {args.seed}")
    print(f"  holdout / res         {args.holdout} / {args.res} ({res_dir})")
    print(f"  train                 {len(train_pairs)}  {n_per_city}")
    print(f"  val                   {len(val_pairs)} (fixed at every rung)")
    print(f"  test                  {len(test_pairs)}")
    print(f"  published config      {cfg_path}")
    print(f"  output_dir            {out_dir_parent}")
    print(f"  dump                  {dump_dir}")
    print(f"  slurm job             {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    print(f"  train_list_sha256     {list_sha256(train_pairs)}")
    print(f"  val_list_sha256       {list_sha256(val_pairs)}")
    print(f"  test_list_sha256      {list_sha256(test_pairs)}")
    if smoke:
        print(f"  SMOKE                 epochs<={args.smoke_epochs} "
              f"test<={args.smoke_test}")
    print("=" * 78, flush=True)

    # ---- hard checks on set sizes -------------------------------------------
    if len(train_pairs) != args.rung:
        raise SystemExit(f"FATAL train set is {len(train_pairs)}, rung says {args.rung}")
    if len(val_pairs) != N_VAL:
        raise SystemExit(f"FATAL val set is {len(val_pairs)}, expected {N_VAL}")
    if len(test_pairs) != N_TEST:
        raise SystemExit(f"FATAL test set is {len(test_pairs)}, expected {N_TEST}")
    if set(train_pairs) & set(val_pairs):
        raise SystemExit("FATAL train/val overlap")
    if set(train_pairs) & set(test_pairs) or set(val_pairs) & set(test_pairs):
        raise SystemExit("FATAL test leakage into train or val")

    if args.dry_run:
        print("\n--dry_run: resolved, nothing written.")
        print(f"  first 3 train: {train_pairs[:3]}")
        print(f"  first 3 val:   {val_pairs[:3]}")
        print(f"  first 3 test:  {test_pairs[:3]}")
        return

    # ---- seeds ---------------------------------------------------------------
    import torch                                            # noqa: E402
    seed_sites = seed_everything(args.seed)
    print("SEED SITES")
    for _k, _v in seed_sites.items():
        print(f"  {_k:20s} {_v}")
    print(flush=True)

    # ---- the architecture's own trainer -------------------------------------
    arch_root = BASE / ARCH_DIR[args.arch]
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    # shadowtransfer.metrics puts python/mamnet on sys.path and imports
    # utils.contrast_utils from it, so `utils` is already bound to MAMNet's copy.
    # Drop it, or the OGLANet and DINOv3 trainers would silently import MAMNet's
    # losses and metrics instead of their own.
    for _m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_m]
    archtrain = __import__(ARCH_TRAIN_MODULE[args.arch])

    # Rebind the published trainer's figure calls and its tensorboard writer
    # BEFORE constructing its Trainer -- the writer is built in __init__, so a
    # later rebind would be too late for it. The published files are not
    # edited. Job 20540648 is why: it trained cleanly, then hung ~90 minutes
    # in plot_metrics_curves and produced no dump at all.
    disabled = armc_runtime.silence_worker_side_effects(archtrain)

    from argparse import Namespace
    # Published config wins wherever it has an opinion; the architecture's own
    # parser defaults fill only the gaps. See arch_parser_defaults().
    _defaults = arch_parser_defaults(archtrain, ARCH_TRAIN_MODULE[args.arch])
    defaults_filled = {k: _defaults[k] for k in sorted(set(_defaults) - set(cfg))}
    if defaults_filled:
        print(f"CONFIG defaults filled from the arch's own parser: "
              f"{defaults_filled}", flush=True)
    a = {**_defaults, **cfg}
    a["output_dir"] = str(out_dir_parent)
    a["device"] = "cuda"
    a["eval_only"] = False
    if args.mode == "loco":
        a["mode"] = "loco"
        a["base_data_root"] = str(BASE / "data" / "Final_data_test") + "/"
        a["resolution"] = res_dir
        a["fold_id"] = FOLD_OF_HOLDOUT[args.holdout]
        a["data_root"] = None
    else:
        a["mode"] = "single"
        a["data_root"] = str(BASE / "data" / "Final_data_test" / args.holdout / res_dir) + "/"
        a["base_data_root"] = None
        a["resolution"] = None
        a["fold_id"] = None
    for k in ("comparison_inference_dir", "comparison_data_root"):
        if k in a and a[k]:
            a[k] = str(BASE / "data" / Path(a[k].rstrip("/")).name) + "/"
    if args.arch == "dinov3":
        a["weights_path"] = str(DINOV3_WEIGHTS)
    if args.smoke_epochs:
        a["epochs"] = args.smoke_epochs
    # Only checkpoint_best.pth survives a cell. mamnet also writes
    # checkpoint_epoch_{10,20} by default; at ~509 MB each, 54 cells would cross
    # the /projects soft quota. Switched off through the published config's own
    # save_freq knob, which train.py already gates on.
    armc_runtime.disable_periodic_checkpoints(a)

    # The published args.json carries no seed, so three seeds of one cell are
    # indistinguishable on disk. All three trainers write vars(args) to
    # {output_dir}/args.json in __init__, so these land there. None of them
    # reads args.seed, so the injection is inert with respect to training.
    # One early-stopping recipe for every arch at every rung. Set BEFORE the
    # Trainer is built so {output_dir}/args.json records what was actually
    # applied rather than the published value.
    patience_epochs, spe_expected = armc_patience_epochs(args.rung, a["batch_size"])
    patience_published = a.get("early_stopping_patience")
    a["early_stopping_patience"] = patience_epochs
    a["epochs"] = ARMC_EPOCHS_MAX
    print(f"EARLY STOPPING  patience={patience_epochs} epochs flat "
          f"({patience_epochs * spe_expected} steps at {spe_expected} "
          f"steps/epoch; this cell's published args.json said "
          f"{patience_published}), epochs_max={ARMC_EPOCHS_MAX}", flush=True)

    a["seed"] = args.seed
    a["armc_run_id"] = args.run_id
    a["armc_rung"] = args.rung
    a["armc_n_train"] = args.rung
    a["armc_n_val"] = N_VAL

    a["resume"] = None
    trainer_args = Namespace(**a)

    trainer = archtrain.Trainer(trainer_args)
    run_out = Path(trainer.output_dir)

    # Resume from the newest periodic checkpoint if this cell was killed before.
    resume_from = newest_epoch_checkpoint(run_out)
    if resume_from:
        print(f"RESUME from {resume_from}", flush=True)
        trainer.load_checkpoint(resume_from)

    # ---- swap in the rung-controlled dataloaders -----------------------------
    from torch.utils.data import DataLoader
    img_size = trainer_args.img_size
    nw = trainer_args.num_workers
    bs = trainer_args.batch_size

    ds_train = build_dataset(args.arch, train_pairs, "train", res_dir, img_size, True)
    ds_val = build_dataset(args.arch, val_pairs, "val", res_dir, img_size, False)
    ds_test = build_dataset(args.arch, test_pairs, "test", res_dir, img_size, False)
    if args.smoke_test:
        for attr in ("img_files", "img_paths", "mask_paths"):
            setattr(ds_test, attr, getattr(ds_test, attr)[:args.smoke_test])

    # generator= makes the shuffle reproducible from the run seed alone rather
    # than from however much global torch RNG happened to be consumed first.
    # worker_init_fn= seeds python random and numpy INSIDE each worker process;
    # torch seeds only its own per-worker RNG, and the published augmentations
    # draw from random and numpy as well.
    g = torch.Generator()
    g.manual_seed(args.seed)

    trainer.dataloaders = {
        "train": DataLoader(ds_train, batch_size=bs, shuffle=True, num_workers=nw,
                            pin_memory=True, drop_last=True,
                            generator=g, worker_init_fn=dl_worker_init),
        "val": DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw,
                          pin_memory=True, worker_init_fn=dl_worker_init),
        "test": DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=nw,
                           pin_memory=True, worker_init_fn=dl_worker_init),
    }
    steps_per_epoch = len(trainer.dataloaders["train"])
    print(f"\nRUNG DATALOADERS  train={len(ds_train)}  val={len(ds_val)}  "
          f"test={len(ds_test)}  steps/epoch={steps_per_epoch}", flush=True)
    if len(ds_train) != args.rung:
        raise SystemExit(f"FATAL loader train size {len(ds_train)} != rung {args.rung}")
    if steps_per_epoch != spe_expected:
        raise SystemExit(f"FATAL steps/epoch {steps_per_epoch} != {spe_expected} "
                         f"assumed when the patience was set; run.json's "
                         f"early_stop_patience_steps would be wrong")

    # ---- train ---------------------------------------------------------------
    t_train = time.time()
    trainer.train()
    train_wall = time.time() - t_train

    best_ckpt = run_out / "checkpoint_best.pth"
    if not best_ckpt.exists():
        raise SystemExit(f"FATAL no best checkpoint at {best_ckpt}")
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    best_epoch = int(ck["epoch"])
    epochs_run = len(trainer.train_losses)
    steps_to_best = best_epoch * steps_per_epoch
    best_val_metric = float(ck.get("best_decision_miou", float("nan")))
    del ck

    print(f"\nBEST epoch {best_epoch}  epochs_run {epochs_run}  "
          f"steps_to_best {steps_to_best}  best_val_decision_mIoU {best_val_metric:.4f}",
          flush=True)

    # ---- inference on the held-out test pool, then the dump ------------------
    # Everything past training runs inside a hard time-box. Job 20540648 trained
    # cleanly, then hung ~90 minutes in the published figure call against a
    # 2-hour wall clock, and the cell produced nothing at all. A hang here now
    # names the phase it was in, dumps every thread's stack, writes .FAILED and
    # exits non-zero rather than eating the wall clock silently.
    from utils.postprocessing import filter_small_predictions

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference")
        trainer.load_checkpoint(str(best_ckpt))
        model = trainer.model
        model.eval()
        device = trainer.device

        geo = armc_dump.load_geo_meta(BASE)
        rows, probs = [], {}
        t_inf = time.time()
        with torch.no_grad():
            for batch in trainer.dataloaders["test"]:
                images = batch["image"].to(device)
                masks = batch["mask"]
                fname = batch["filename"][0]
                stem = Path(fname).stem

                logits = model(images)
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
                filtered = filter_small_predictions(logits, min_pixels=10)
                pred = torch.argmax(filtered, dim=1)[0].cpu().numpy().astype(np.uint8)
                gt = masks[0].numpy().astype(np.uint8)

                rows.append(armc_dump.perimage_row(
                    args.run_id, stem, args.holdout, args.res, gt, prob, pred,
                    geo.get(fname)))
                probs[stem] = prob.astype(np.float16)
        inf_wall = time.time() - t_inf

        tbx.phase("metrics")
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        wall = time.time() - t0

        import pandas as pd                                     # noqa: E402
        df = pd.DataFrame(rows)
        fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
        miou_tol = float(df["miou_tol_pct"].mean())
        early_stopped = bool(epochs_run < trainer_args.epochs)

        # Did the silencing actually hold? This is an execution fact the job
        # checks on itself, not analysis: if a .png or a tensorboard event file
        # appeared, the rebinding did not take and the next cell will hang.
        stray_png = sorted(str(q.relative_to(run_out)) for q in run_out.rglob("*.png"))
        stray_tb = sorted(str(q.relative_to(run_out)) for q in run_out.rglob("events.out.tfevents*"))
        stray_epoch_ckpt = sorted(q.name for q in run_out.glob("checkpoint_epoch_*.pth"))

        run = {
            "run_id": args.run_id,
            "dataset": "st",
            "arm": "C",
            "arch": args.arch,
            "method": method,
            "mode": args.mode,
            "rung": args.rung,
            "n_train": len(ds_train),
            "n_train_per_source_city": n_per_city,
            "n_val": len(ds_val),
            "n_test": len(df),
            "holdout": args.holdout,
            "res": args.res,
            "seed": args.seed,
            "seed_scope": "python random, numpy, torch, torch.cuda, the "
                          "train/val file selection, and the dataloader "
                          "workers (see seed_sites)",
            "seed_sites": seed_sites,
            "worker_seeding": (
                "each DataLoader gets generator= seeded from the run seed and "
                "worker_init_fn=dl_worker_init, which seeds python random and "
                "numpy inside every worker process from torch.initial_seed(). "
                "Without it only torch's own per-worker RNG is reproducible, "
                "and the published augmentations draw from random and numpy."),

            "target_access": "source_only" if args.mode == "loco" else "target_labels",
            "target_access_detail": (
                "training and validation draw only from the two source cities; no "
                "held-out-city image is touched before test-time inference"
                if args.mode == "loco" else
                "within-city upper bound: train and val are labelled images from the "
                "held-out city itself, disjoint from its test pool"),

            # --- the two metrics, kept apart on purpose --------------------
            "decision_metric": "2class_tol_k2",
            "decision_metric_value": round(best_val_metric / 100.0, 6),
            "decision_metric_value_pct": round(best_val_metric, 4),
            "decision_metric_note": (
                "what checkpoint_best.pth was SELECTED on: the published "
                "architecture's own decision metric, per-image DetailedEvaluator "
                "2-class tolerant mIoU at K=2. Unchanged on purpose -- arm C "
                "varies the training-set size and nothing else. It is roughly "
                "twice the reporting metric and must not be read as an arm C "
                "result."),
            "reporting_metric": "fg_only_tol_k2",
            "reporting_metric_value": round(fg_iou_tol, 4),
            "reporting_metric_note": (
                "what the paper and this arm REPORT: foreground-only IoU at "
                "K=2, column iou_tol. CONTRACT.md section 3."),

            "epochs_max": trainer_args.epochs,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "steps_per_epoch": steps_per_epoch,
            "steps_to_best": steps_to_best,
            "early_stopped": early_stopped,
            "best_val_metric": best_val_metric,
            "best_val_metric_definition": (
                "the architecture's own published decision metric: per-image "
                "DetailedEvaluator tolerant mIoU at K=boundary_tolerance, percent"),
            "early_stop_patience": trainer_args.early_stopping_patience,
            "early_stop_rule": (
                f"patience {ARMC_PATIENCE_EPOCHS} epochs flat, epochs capped at "
                f"{ARMC_EPOCHS_MAX}, best checkpoint retained. One recipe for "
                f"every architecture at every rung so the rung is the only "
                f"variable. Table 23's LOCO value; never below what any "
                f"published cell used."),
            "early_stop_patience_published": patience_published,
            "early_stop_patience_steps": patience_epochs * steps_per_epoch,
            "early_stop_note": (
                "every published args.json in this tree records "
                "early_stopping_patience=10 for all three architectures; "
                "Table 23 states 20. Arm C applies 20 uniformly, so at rung 450 "
                "this is twice the patience the published cells actually ran "
                "with and those cells are not a like-for-like reproduction of "
                "the published configuration on this axis."),
            "lr": trainer_args.lr,
            "batch_size": trainer_args.batch_size,

            "metric_module_sha": sha256_file(BASE / "shadowtransfer/metrics.py"),
            "metric_module_path": "shadowtransfer/metrics.py",
            "dump_module_sha": sha256_file(BASE / "shadowtransfer/armc_dump.py"),
            "dump_module_note": (
                "armc_dump.py was truncated to 0 bytes on 2026-07-28 and "
                "restored from its own bytecode, verified equivalent "
                "instruction-for-instruction across all 15 code objects. "
                "Behaviour is unchanged; the sha is not the pre-incident one "
                "and will not match what the arm C r450 cells recorded."),
            "runtime_module_sha": sha256_file(BASE / "shadowtransfer/armc_runtime.py"),
            "train_script_sha": sha256_file(Path(__file__).resolve()),
            "git_hash": None,
            "git_hash_reason": (
                "BASE_PATH became a git repository on 2026-07-28, after this "
                "script was written; this field is left null until the operator "
                "decides whether run.json should carry a commit"),
            "config_sha256": sha256_file(cfg_path),
            "config_verbatim": a,
            "published_config_path": str(cfg_path.relative_to(BASE)),
            "config_defaults_filled": defaults_filled,
            "config_defaults_note": (
                "keys absent from the published args.json and supplied by the "
                "architecture's own argparse defaults. Empty for mamnet and "
                "oglanet. For dinov3 this is freeze_backbone=False, i.e. full "
                "fine-tuning -- a frozen backbone would be arm E, not arm C."),

            "train_list_sha256": list_sha256(train_pairs),
            "val_list_sha256": list_sha256(val_pairs),
            "test_list_sha256": list_sha256(test_pairs),
            "train_selection": (
                "seeded permutation per (city, split), nested prefixes across rungs; "
                "rung 225 splits 113/112 across the two source cities"),
            "n_train_provenance": (
                "the rung constant, not a directory listing. n_val is the N_VAL "
                "constant. Directories are read only to enumerate candidate "
                "filenames, and take_prefix hard-fails if a pool is short. "
                "Directory contents are not authoritative for counts "
                "(CLAUDE.md, n_train/n_val provenance override 2026-07-28)."),
            "val_note": (
                "n_val is pinned to 150 at every rung per REVIEWS.md arm C. The "
                "published LOCO cells validate on 300 (both source cities' full val "
                "pools); the published upper-bound cells validate on 150."),

            "cluster": "delta",
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu": gpu_name,
            "slurm_time_limit": os.environ.get("ARMC_TIME_LIMIT"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "wall_clock_s": round(wall, 2),
            "wall_clock_note": (
                "measured before write_dump, so it excludes the parquet and "
                "probs.npz writes and under-reports the cell. Left as-is on "
                "operator instruction 2026-07-28."),
            "train_wall_clock_s": round(train_wall, 2),
            "inference_wall_clock_s": round(inf_wall, 2),
            "peak_vram_alloc_gb": round(peak_alloc, 3),
            "peak_vram_reserved_gb": round(peak_reserved, 3),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),

            "ckpt_path": str(best_ckpt),
            "parent_run_id": None,
            "legacy_run_dir": None,
            "structure_cols": "deferred",

            "worker_side_effects_disabled": disabled,
            "post_training_timeout_s": args.timeout_s,
            "side_effect_audit": {
                "png_files": stray_png,
                "tensorboard_event_files": stray_tb,
                "periodic_checkpoints": stray_epoch_ckpt,
                "save_freq_effective": a.get("save_freq"),
            },

            "prediction_recipe": (
                "argmax over the two class logits, then "
                "filter_small_predictions(min_pixels=10); identical to the "
                "architecture's own test() loop"),
            "probs_are_prefilter": True,
            "probs_note": "softmax(logits)[:, 1], stored float16, unthresholded",
            "band_width_headline": 2,
            "band_widths_written": [0, 2, 5],
            "headline_metric": "foreground-only IoU at K=2 (column iou_tol)",
            "headline_fg_iou_tol_pct": fg_iou_tol,
            "miou_tol_pct_published_definition": miou_tol,

            "aurc_definition": "aurc_trapz19",
            "aurc_definition_note": (
                "CONTRACT.md section 3 leaves the canonical AURC blank. "
                "aurc_fg/bg/all use aurc_trapz19; aurc_*_mean20 carry the "
                "SIB-trainer plain mean over 20 points. Neither is declared "
                "canonical by this run."),

            "smoke": smoke,
            "forced_rerun": bool(args.force),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # ---- self-checks a job runs on itself (execution facts, not analysis)
        flags = []
        if epochs_run < 10:
            flags.append(f"epochs_run={epochs_run} (<10)")
        if epochs_run >= trainer_args.epochs:
            flags.append(f"ran to the epoch ceiling ({trainer_args.epochs}); "
                         f"early stopping never fired")
        if int(df["empty_fg_valid"].sum()):
            flags.append(f"empty_fg_valid on {int(df['empty_fg_valid'].sum())} images")
        if float(df["iou_tol"].std()) < 1e-6:
            flags.append("per-image iou_tol has ~zero variance")
        if stray_png:
            flags.append(f"figures were written despite the rebind: {stray_png}")
        if stray_tb:
            flags.append(f"tensorboard events were written despite the rebind: {stray_tb}")
        if stray_epoch_ckpt:
            flags.append(f"periodic checkpoints were written: {stray_epoch_ckpt}")
        run["sanity_flags"] = flags

        tbx.phase("write_dump")
        armc_dump.write_dump(dump_dir, rows, probs, run)

    print("\n" + "=" * 78)
    print(f"[{args.run_id}]")
    print(f"  fg IoU K=2 (headline)   {fg_iou_tol:.4f}")
    print(f"  fg IoU K=0 / K=5        {float(df['iou_str'].mean())*100:.4f} / "
          f"{float(df['iou_k5'].mean())*100:.4f}")
    print(f"  2-class mIoU K=2        {miou_tol:.4f}  (published-table definition)")
    print(f"  decision metric         {run['decision_metric']} = "
          f"{run['decision_metric_value_pct']:.4f}%")
    print(f"  images                  {len(df)}")
    print(f"  epochs_run              {epochs_run}   best_epoch {best_epoch}   "
          f"early_stopped {early_stopped}")
    print(f"  steps_to_best           {steps_to_best}  ({steps_per_epoch}/epoch)")
    print(f"  wall clock              {wall:.1f}s  (train {train_wall:.1f}s, "
          f"inference {inf_wall:.1f}s)")
    print(f"  peak VRAM alloc/resv    {peak_alloc:.2f} / {peak_reserved:.2f} GiB")
    print(f"  side effects            png={len(stray_png)} tb={len(stray_tb)} "
          f"epoch_ckpt={len(stray_epoch_ckpt)}")
    print(f"  sanity flags            {flags or 'none'}")
    print(f"  -> {dump_dir}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
