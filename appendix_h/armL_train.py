"""
Arm L — SSL-pretrained CNN under DINOv3's fine-tuning regime.

Answers zNBu-W5 (primary), zNBu-W1, AC1, AC2.

The question
------------
Section 4.4's failure-locus finding separates DINOv3 from the two CNNs. DINOv3
is simultaneously the only transformer, the only self-supervised pretraining and
the only low-LR fine-tune in the paper, so "architecture" and "regime" are
confounded. Arm E answered the DINOv3 half by freezing its backbone. Arm L
answers the CNN half with a 2x2 at fixed capacity:

    pretraining      LR      cell id   what it isolates
    supervised       1e-4    sup_hi    control: R50 under the published regime
    supervised       1e-5    sup_lo    LR alone
    ssl (DINO-R50)   1e-4    ssl_hi    pretraining alone
    ssl (DINO-R50)   1e-5    ssl_lo    zNBu-W5's exact ask

x 3 held-out cities x 3 seeds = 36 LOCO runs, plus 36 matched within-city upper
bounds. High-resolution only; that is a declared scope cut matching arm C.

Two subcommands, run as separate SLURM jobs
-------------------------------------------
    train   one cell: train, infer on the held-out city, write the CONTRACT.md
            dump. Nothing else.
    probes  the Section 4.2-4.4 diagnostics for one LOCO cell, after both that
            cell and its matched upper bound carry .DONE.

They are separate because CLAUDE.md requires it: a probe failure must not be
able to cost a trained model, and the training cell's wall clock has to stay
predictable.

run_id
------
    mamnetr50-{cell}-{mode}-r450-{holdout}-hr-s{seed}
    cell in {sup_hi, sup_lo, ssl_hi, ssl_lo}

CONTRACT.md's `method` slot carries the 2x2 cell id rather than
`vanilla`/`none`, in both modes. The 2x2 is a property of the model, not of an
adaptation method, and the upper bounds vary over it too, so arm C's
mode=upper -> method="none" convention would collide four cells into one id.
Recorded in run.json as `method_note`.

What is NOT the published protocol, and is recorded rather than hidden
----------------------------------------------------------------------
  1. The encoder is ResNet-50, not the published ResNet-34. That is the arm.
  2. Early stopping is flat 20 epochs, arm C's recipe, not Table 9's 10.
  3. The auxiliary heads attach one decoder stage later than the published
     model's; see mamnet_r50.AUX_ATTACH_NOTE.
  4. checkpoint_best.pth carries model weights only, no optimizer state. At
     350M parameters the optimizer state is 2.8 GB per cell and 72 cells of it
     does not fit. A killed cell restarts rather than resuming.
  5. Checkpoints default to /scratch. 72 x 1.4 GB does not fit in the
     /projects headroom below the 500G soft quota. Dumps still go to /projects.

Everything else -- Adam, weight decay 1e-4, batch 8, 100 epochs,
ReduceLROnPlateau(max, 0.5, 3), CE + 0.4 * mean(3 aux CE), h/v-flip, the <10 px
component filter, and 2-class boundary-tolerant mIoU at K=2 as the checkpoint
decision metric -- is the published MAMNet configuration, taken from
data/mamnet/outputs/*/args.json and python/mamnet/train.py.
"""

import argparse
import hashlib
import json
import os
import sys
import time
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

from shadowtransfer import armc_dump                             # noqa: E402
from shadowtransfer import armc_runtime                          # noqa: E402
from shadowtransfer.armc_dump import sha256_file                 # noqa: E402

# ---------------------------------------------------------------------------
# Fixed project facts
# ---------------------------------------------------------------------------

LOCO_FOLDS = {
    0: {"train": ["chicago", "miami"],   "test": "phoenix"},
    1: {"train": ["chicago", "phoenix"], "test": "miami"},
    2: {"train": ["miami",   "phoenix"], "test": "chicago"},
}
FOLD_OF_HOLDOUT = {"phoenix": 0, "miami": 1, "chicago": 2}
RES_DIR = {"hr": "highres", "mr": "midres"}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

ARCH = "mamnetr50"
RUNG = 450
N_VAL = 150
N_TEST = 150
IMG_SIZE = 384
BATCH_SIZE = 8
NUM_WORKERS = 1                 # the published MAMNet value
WEIGHT_DECAY = 1e-4
AUX_WEIGHT = 0.4
BOUNDARY_TOLERANCE = 2          # the decision metric's band, args.json's value
EPOCHS_MAX = 100
PATIENCE_EPOCHS = 20            # arm C's flat rule, NOT Table 9's 10
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 3

CELLS = {
    "sup_hi": {"pretraining": "supervised", "lr": 1e-4, "lr_label": "high"},
    "sup_lo": {"pretraining": "supervised", "lr": 1e-5, "lr_label": "low"},
    "ssl_hi": {"pretraining": "ssl",        "lr": 1e-4, "lr_label": "high"},
    "ssl_lo": {"pretraining": "ssl",        "lr": 1e-5, "lr_label": "low"},
}

# Checkpoints are 1.40 GB each (350,051,886 fp32 parameters, weights only).
# 72 cells is ~101 GB; /projects had 55 G below its 500 G soft quota on
# 2026-07-30. /scratch had 1.3 T free. Dumps are small and stay on /projects.
DEFAULT_CKPT_ROOT = Path(os.environ["ST_SCRATCH_ROOT"]) / "data/armL"

# Probe constants, matching arm C so the two arms' numbers are comparable.
R_EXCLUDE_MIN = 0.02
EXPAB_EPOCHS = 30
EXPAB_SUBSET_FRAC = 0.25
EXPB_ENCODER_LR_MULT = 1.0      # erratum E-10: App G.9's "100x" is wrong
CELLS_PER_IMAGE = 8             # city-identity probe spatial subsample
N_PERM = 100


def stable_rng(*parts):
    """Generator seeded from a stable hash. Python's hash() is salted per
    process, so a resubmitted cell would otherwise draw different images."""
    d = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(d[:8], "big"))


def run_id_for(cell, mode, holdout, seed):
    return f"{ARCH}-{cell}-{mode}-r{RUNG}-{holdout}-hr-s{seed}"


def city_split_files(city, res_dir, split):
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
    """Seeded permutation then the first n. Same construction as arm C, with
    'armL' in the hash so the two arms draw independent selections."""
    if n > len(names):
        raise SystemExit(f"FATAL requested {n} from {city}/{split}, only "
                         f"{len(names)} paired files exist")
    perm = stable_rng("armL", city, split, seed).permutation(len(names))
    return sorted(names[i] for i in perm[:n])


def resolve_selection(mode, holdout, res_dir, seed):
    """(train, val, test) as lists of (city, filename), plus provenance.

    LOCO trains on 225 + 225 from the two source cities and never touches a
    held-out-city image before test-time inference. The upper bound trains on
    450 from the held-out city's own train pool, disjoint from its test pool.
    Validation is 150 in both modes.
    """
    if mode == "loco":
        fold = FOLD_OF_HOLDOUT[holdout]
        src = LOCO_FOLDS[fold]["train"]
        if LOCO_FOLDS[fold]["test"] != holdout:
            raise SystemExit("FATAL fold table disagrees with holdout")
        per_city = RUNG // len(src)
        train = [(c, f) for c in src
                 for f in take_prefix(city_split_files(c, res_dir, "train"),
                                      per_city, c, "train", seed)]
        val = [(c, f) for c in src
               for f in take_prefix(city_split_files(c, res_dir, "val"),
                                    N_VAL // len(src), c, "val", seed)]
        n_per_city = {c: per_city for c in src}
    elif mode == "upper":
        train = [(holdout, f) for f in take_prefix(
            city_split_files(holdout, res_dir, "train"), RUNG,
            holdout, "train", seed)]
        val = [(holdout, f) for f in take_prefix(
            city_split_files(holdout, res_dir, "val"), N_VAL,
            holdout, "val", seed)]
        n_per_city = {holdout: RUNG}
    else:
        raise SystemExit(f"FATAL unknown mode {mode}")

    test = [(holdout, f) for f in city_split_files(holdout, res_dir, "test")]
    return train, val, test, n_per_city


def list_sha256(pairs):
    return hashlib.sha256(
        "\n".join(f"{c}/{f}" for c, f in pairs).encode()).hexdigest()


def city_roots(pairs, res_dir):
    seen, roots = set(), []
    for c, _ in pairs:
        if c not in seen:
            seen.add(c)
            roots.append(str(BASE / "data" / "Final_data_test" / c / res_dir))
    return roots


def seed_everything(seed):
    import random
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
        "dataloader_shuffle": f"torch.Generator().manual_seed({seed})",
        "dataloader_workers": "dl_worker_init seeds random and numpy per worker",
        "file_selection": f"stable_rng('armL', city, split, {seed})",
    }


def dl_worker_init(worker_id):
    import random
    import torch
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s)
    np.random.seed(s)


def enter_mamnet_tree():
    """Put python/mamnet on sys.path and cd into it, as arm C's jobs do.

    Arm L reuses MAMNet's dataset, loss and postprocessing unchanged; only the
    model is new. Any `utils` module already bound is dropped first --
    shadowtransfer.metrics imports MAMNet's utils.contrast_utils, and a stale
    binding is how the wrong arch's losses get imported silently.
    """
    root = BASE / "mamnet"
    sys.path.insert(0, str(root))
    os.chdir(root)
    for m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[m]
    return root


def build_dataset(pairs, split, res_dir, augment):
    """MAMNet's own ShadowDatasetEnhanced, then filtered to the selected files.

    task_id=2 is the published MAMNet setting: it derives the 4th (contrast)
    channel and normalises RGB with ImageNet statistics. augment=True adds
    RandomHorizontalFlip(p=0.5) and RandomVerticalFlip(p=0.5) and nothing else,
    which is exactly the arm spec's training augmentation. Filtering after
    construction rather than reimplementing the loader guarantees every
    transform is the published one.
    """
    from data.dataset_enhanced import ShadowDatasetEnhanced
    roots = city_roots(pairs, res_dir)
    ds = ShadowDatasetEnhanced(root_dir=roots, split=split, img_size=IMG_SIZE,
                               task_id=2, augment=augment, use_fda=False)
    want = {str(BASE / "data" / "Final_data_test" / c / res_dir / split /
                "images" / f) for c, f in pairs}
    keep = [i for i, p in enumerate(ds.img_paths) if os.path.abspath(p) in want]
    if len(keep) != len(want):
        raise SystemExit(
            f"FATAL {split} selection resolved {len(keep)} of {len(want)} "
            f"files; dataset saw {len(ds.img_paths)} candidates under {roots}")
    ds.img_files = [ds.img_files[i] for i in keep]
    ds.img_paths = [ds.img_paths[i] for i in keep]
    ds.mask_paths = [ds.mask_paths[i] for i in keep]
    return ds


def make_loader(ds, batch_size, shuffle, seed, drop_last=False):
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      drop_last=drop_last, generator=g if shuffle else None,
                      worker_init_fn=dl_worker_init)


# ---------------------------------------------------------------------------
# Model, optimiser, loss
# ---------------------------------------------------------------------------

def build_model(cell, device, pretraining_override=None):
    """MAMNetR50 for one 2x2 cell, on `device`, plus its load report.

    Both pretrainings go through mamnet_r50._load_backbone, which aborts on any
    missing or unexpected key outside fc.*. There is no path here that trains on
    a partially loaded encoder.
    """
    import torch
    from shadowtransfer import mamnet_r50 as M
    pre = pretraining_override or CELLS[cell]["pretraining"]
    model = M.MAMNetR50(num_classes=2, pretraining=pre, use_aux=True)
    model = model.to(device)
    report = dict(model.load_report)
    report["param_counts"] = M.param_counts(model)
    if torch.cuda.is_available():
        print(f"[armL] VRAM after model load: "
              f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GiB allocated, "
              f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GiB reserved",
              flush=True)
    return model, report


def build_optimizer(model, lr):
    """The published MAMNet optimiser and scheduler, train.py:190-198."""
    import torch.optim as optim
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=SCHED_FACTOR, patience=SCHED_PATIENCE)
    return opt, sched


def build_criterion():
    """MAMNet's own published loss object: CE(main) + 0.4 * mean(3 aux CE)."""
    from utils.losses import MAMNetLoss
    return MAMNetLoss(aux_weight=AUX_WEIGHT)


def decision_miou(model, loader, device):
    """The published decision metric: per-image 2-class boundary-tolerant mIoU
    at K=2, from the canonical DetailedEvaluator in shadowtransfer.metrics.

    train.py:336 reads detailed['boundary_tolerant'][f'tolerant_{K}px']['iou'],
    where the prediction is argmax of the raw logits -- the <10 px component
    filter is a test-time step and is not applied during model selection. Both
    facts are kept.
    """
    import torch
    from shadowtransfer.metrics import DetailedEvaluator
    ev = DetailedEvaluator(boundary_tolerance=BOUNDARY_TOLERANCE)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"]
            if masks.dim() == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)
            if masks.dtype != torch.long:
                masks = (masks > 0.5).long()
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            # Tensors, and `images` alongside them: this is train.py:488's own
            # call. DetailedEvaluator.update() calls .cpu() on all three, and
            # `images` only feeds the contrast-stratified bins -- the decision
            # metric itself does not read them -- but matching the published
            # call leaves nothing to argue about.
            ev.update(preds.cpu(), masks.cpu(), images.cpu())
    m = ev.compute_metrics()
    return float(m["boundary_tolerant"][f"tolerant_{BOUNDARY_TOLERANCE}px"]["iou"])


def save_best(path, model, epoch, metric, extra):
    """Weights only. 350M fp32 parameters is 1.40 GB; adding Adam's two moment
    buffers would make it 4.2 GB and 72 cells would not fit anywhere. The cost
    is that a killed cell restarts from epoch 0 rather than resuming, which at
    the measured epoch time is the cheaper of the two."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pth.tmp")
    torch.save({"epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "best_decision_miou": float(metric),
                **extra}, tmp)
    tmp.replace(path)
    return path.stat().st_size


def train_cell(model, loaders, lr, device, best_ckpt, extra):
    """Train to the published recipe with arm C's flat 20-epoch patience.

    Returns a dict of training facts for run.json. Model selection, the LR
    schedule and early stopping all read the same decision metric, exactly as
    train.py does.
    """
    import torch
    opt, sched = build_optimizer(model, lr)
    criterion = build_criterion()

    best_metric, best_epoch, since_improve = -1.0, 0, 0
    train_losses, val_metrics, lrs = [], [], []
    steps_per_epoch = len(loaders["train"])
    ckpt_bytes = None

    for epoch in range(1, EPOCHS_MAX + 1):
        model.train()
        running, nb = 0.0, 0
        t_ep = time.time()
        for batch in loaders["train"]:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["mask"].to(device, non_blocking=True)
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype != torch.long:
                labels = (labels > 0.5).long()
            opt.zero_grad(set_to_none=True)
            losses = criterion(model(images), labels)
            losses["total"].backward()
            opt.step()
            running += float(losses["total"].item())
            nb += 1
        epoch_loss = running / max(nb, 1)
        train_losses.append(epoch_loss)

        metric = decision_miou(model, loaders["val"], device)
        val_metrics.append(metric)
        sched.step(metric)
        lrs.append(float(opt.param_groups[0]["lr"]))

        improved = metric > best_metric
        if improved:
            best_metric, best_epoch, since_improve = metric, epoch, 0
            ckpt_bytes = save_best(best_ckpt, model, epoch, metric, extra)
        else:
            since_improve += 1

        print(f"  epoch {epoch:3d}/{EPOCHS_MAX}  loss {epoch_loss:.4f}  "
              f"val 2class-tol-mIoU {metric:.4f}%  best {best_metric:.4f}% "
              f"(ep {best_epoch})  lr {lrs[-1]:.2e}  "
              f"no-improve {since_improve}/{PATIENCE_EPOCHS}  "
              f"{time.time() - t_ep:.0f}s", flush=True)

        if since_improve >= PATIENCE_EPOCHS:
            print(f"  early stop at epoch {epoch}: {PATIENCE_EPOCHS} epochs "
                  f"without improvement", flush=True)
            break

    if best_epoch == 0:
        raise SystemExit("FATAL no epoch improved on the initial metric; "
                         "no checkpoint was written")
    return {
        "epochs_max": EPOCHS_MAX,
        "epochs_run": len(train_losses),
        "best_epoch": best_epoch,
        "steps_per_epoch": steps_per_epoch,
        "steps_to_best": best_epoch * steps_per_epoch,
        "early_stopped": len(train_losses) < EPOCHS_MAX,
        "best_val_metric": best_metric,
        "best_val_metric_definition": (
            "per-image 2-class boundary-tolerant mIoU at K=2, percent, from "
            "shadowtransfer.metrics.DetailedEvaluator -- the published MAMNet "
            "decision metric (train.py:336)"),
        "early_stop_patience": PATIENCE_EPOCHS,
        "early_stop_rule": (
            f"flat {PATIENCE_EPOCHS} epochs without improvement, epochs capped "
            f"at {EPOCHS_MAX}. Arm C's recipe. Table 9 and every published "
            f"args.json say 10; the flat 20 is never below what a published "
            f"cell used and is recorded as a deviation."),
        "early_stop_patience_published": 10,
        "lr_schedule": (f"ReduceLROnPlateau(mode=max, factor={SCHED_FACTOR}, "
                        f"patience={SCHED_PATIENCE}) on the decision metric"),
        "lr_final": lrs[-1] if lrs else None,
        "lr_per_epoch": [round(x, 12) for x in lrs],
        "train_loss_per_epoch": [round(x, 6) for x in train_losses],
        "val_metric_per_epoch": [round(x, 6) for x in val_metrics],
        "checkpoint_bytes": ckpt_bytes,
        "checkpoint_contents": "model_state_dict only; no optimizer state",
    }


# ---------------------------------------------------------------------------
# Inference — the ladder's recipe, so arm L is comparable to arm C
# ---------------------------------------------------------------------------

def infer_rows(model, loader, run_id, city, device, geo):
    """softmax channel 1 for the stored probability, argmax after the <10 px
    filter for the prediction, armc_dump.perimage_row for every metric. Byte
    for byte the loop in armc_train.py; nothing is recomputed inline."""
    import torch
    from utils.postprocessing import filter_small_predictions
    model.eval()
    rows, probs = [], {}
    with torch.no_grad():
        for batch in loader:
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
                run_id, stem, city, "hr", gt, prob, pred, geo.get(fname)))
            probs[stem] = prob.astype(np.float16)
    return rows, probs


# The sha256 of python/shadowtransfer/metrics.py as it stood in the COMMITTED
# tree before 2026-07-31, i.e. before the additive commit that brought
# `bootstrap_ci` (lifted for arm E Part 2, 47 insertions, 0 deletions, no
# existing function changed) under version control. Arm L calls bootstrap_ci
# for its recovery-ratio CIs, so it depends on the newer blob. Both are
# recorded: an artifact that names only one of them cannot be traced by a
# reader holding the other.
METRIC_MODULE_SHA_PREV = (
    "d19d5e948f79c5498bf21a0e47d1302b4b13e8a97ff7d2c2a48033d9f1514e13")


def script_shas():
    return {
        "metric_module_sha": sha256_file(BASE / "python/shadowtransfer/metrics.py"),
        "metric_module_sha_prev": METRIC_MODULE_SHA_PREV,
        "metric_module_sha_prev_note": (
            "the committed blob before the 2026-07-31 additive commit of "
            "bootstrap_ci; metric_module_sha is what this run actually used"),
        "dump_module_sha": sha256_file(BASE / "python/shadowtransfer/armc_dump.py"),
        "runtime_module_sha": sha256_file(BASE / "python/shadowtransfer/armc_runtime.py"),
        "model_module_sha": sha256_file(BASE / "python/shadowtransfer/mamnet_r50.py"),
        "probes_common_sha": sha256_file(BASE / "python/shadowtransfer/armc_probes_common.py"),
        "probes_ops_sha": sha256_file(BASE / "python/shadowtransfer/armc_probes_ops.py"),
        "probe_script_sha": sha256_file(Path(__file__).resolve()),
        "train_script_sha": sha256_file(Path(__file__).resolve()),
    }


def git_hash():
    """The working-tree commit, or null with the reason. CONVENTIONS.md wants
    provenance; CLAUDE.md records that BASE_PATH became a git repo on
    2026-07-28. Arm L's scripts are committed before the grid runs, so this is
    populated rather than null."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            dirty = subprocess.run(
                ["git", "-C", str(BASE), "status", "--porcelain",
                 "python/armL_train.py", "python/shadowtransfer/mamnet_r50.py",
                 "python/armL_worker.sh", "python/armL_submit.sh"],
                capture_output=True, text=True, timeout=30).stdout.strip()
            return out.stdout.strip(), (None if not dirty else
                                        f"arm L scripts uncommitted: {dirty}")
    except Exception as exc:                      # noqa: BLE001
        return None, f"git rev-parse failed: {exc}"
    return None, "git rev-parse returned non-zero"


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args):
    t0 = time.time()
    cell = CELLS[args.cell]
    res_dir = RES_DIR["hr"]
    smoke = bool(args.smoke_epochs or args.smoke_test)
    if smoke and args.out_root is None:
        raise SystemExit("FATAL --smoke_* requires --out_root")

    run_id = run_id_for(args.cell, args.mode, args.holdout, args.seed)
    if args.run_id and args.run_id != run_id:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {run_id}")

    dump_dir = (Path(args.out_root) if args.out_root else BASE / "dumps") / "st" / run_id
    if (dump_dir / ".DONE").exists() and not args.force:
        print(f"[{run_id}] .DONE exists at {dump_dir} — nothing to do")
        return

    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet", dump_dir / "probs.npz",
        dump_dir / "run.json", dump_dir / ".DONE"], label="dump")

    ckpt_root = Path(args.ckpt_root) if args.ckpt_root else DEFAULT_CKPT_ROOT
    if smoke:
        ckpt_root = Path(args.out_root) / "train"
    best_ckpt = ckpt_root / run_id / "checkpoint_best.pth"

    train_pairs, val_pairs, test_pairs, n_per_city = \
        resolve_selection(args.mode, args.holdout, res_dir, args.seed)

    print("=" * 78)
    print(f"ARM L  run_id            {run_id}")
    print(f"  cell                   {args.cell}  "
          f"({cell['pretraining']} pretraining, lr {cell['lr']:g} = "
          f"{cell['lr_label']})")
    print(f"  mode / holdout / res   {args.mode} / {args.holdout} / hr ({res_dir})")
    print(f"  seed                   {args.seed}")
    print(f"  train / val / test     {len(train_pairs)} {n_per_city} / "
          f"{len(val_pairs)} / {len(test_pairs)}")
    print(f"  checkpoint             {best_ckpt}")
    print(f"  dump                   {dump_dir}")
    print(f"  slurm                  {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    print(f"  train_list_sha256      {list_sha256(train_pairs)}")
    print(f"  val_list_sha256        {list_sha256(val_pairs)}")
    print(f"  test_list_sha256       {list_sha256(test_pairs)}")
    if smoke:
        print(f"  SMOKE                  epochs<={args.smoke_epochs} "
              f"test<={args.smoke_test}")
    print("=" * 78, flush=True)

    if len(train_pairs) != RUNG:
        raise SystemExit(f"FATAL train set is {len(train_pairs)}, expected {RUNG}")
    if len(val_pairs) != N_VAL:
        raise SystemExit(f"FATAL val set is {len(val_pairs)}, expected {N_VAL}")
    if len(test_pairs) != N_TEST:
        raise SystemExit(f"FATAL test set is {len(test_pairs)}, expected {N_TEST}")
    if set(train_pairs) & set(val_pairs):
        raise SystemExit("FATAL train/val overlap")
    if set(train_pairs) & set(test_pairs) or set(val_pairs) & set(test_pairs):
        raise SystemExit("FATAL test leakage into train or val")
    if args.mode == "loco" and any(c == args.holdout for c, _ in train_pairs + val_pairs):
        raise SystemExit("FATAL a LOCO cell selected a held-out-city image")

    if args.dry_run:
        print("\n--dry_run: resolved, nothing written.")
        print(f"  first 3 train: {train_pairs[:3]}")
        print(f"  first 3 test:  {test_pairs[:3]}")
        return

    import torch
    seed_sites = seed_everything(args.seed)
    enter_mamnet_tree()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, load_report = build_model(args.cell, device)

    # Sanity: a load that reported clean but changed nothing would give cosine
    # 1.0 against a fresh encoder. Eight fixed images, on the same device.
    ds_probe = build_dataset(test_pairs[:8], "test", res_dir, augment=False)
    fixed = torch.stack([ds_probe[i]["image"] for i in range(len(ds_probe))])
    from shadowtransfer import mamnet_r50 as M
    cos_mean, cos_min, cos_max = M.feat5_cosine_check(
        fixed, cell["pretraining"], device=device)
    print(f"[armL] feat5 cosine fresh-vs-loaded: mean={cos_mean:.4f} "
          f"min={cos_min:.4f} max={cos_max:.4f}", flush=True)
    if cos_mean > 0.999:
        raise SystemExit(f"FATAL loaded encoder is indistinguishable from a "
                         f"fresh one (cosine {cos_mean:.6f}); the checkpoint "
                         f"did not take")
    torch.cuda.empty_cache()

    ds_train = build_dataset(train_pairs, "train", res_dir, augment=True)
    ds_val = build_dataset(val_pairs, "val", res_dir, augment=False)
    ds_test = build_dataset(test_pairs, "test", res_dir, augment=False)
    if args.smoke_test:
        for attr in ("img_files", "img_paths", "mask_paths"):
            setattr(ds_test, attr, getattr(ds_test, attr)[:args.smoke_test])
    loaders = {
        "train": make_loader(ds_train, BATCH_SIZE, True, args.seed, drop_last=True),
        "val": make_loader(ds_val, BATCH_SIZE, False, args.seed),
        "test": make_loader(ds_test, 1, False, args.seed),
    }
    print(f"LOADERS train={len(ds_train)} val={len(ds_val)} test={len(ds_test)} "
          f"steps/epoch={len(loaders['train'])}", flush=True)

    global EPOCHS_MAX
    epochs_ceiling = EPOCHS_MAX
    if args.smoke_epochs:
        EPOCHS_MAX = args.smoke_epochs

    ckpt_extra = {"run_id": run_id, "cell": args.cell,
                  "pretraining": cell["pretraining"], "lr": cell["lr"]}
    t_train = time.time()
    tr = train_cell(model, loaders, cell["lr"], device, best_ckpt, ckpt_extra)
    train_wall = time.time() - t_train
    EPOCHS_MAX = epochs_ceiling

    ck_gb = (tr["checkpoint_bytes"] or 0) / 1024 ** 3
    print(f"\n[armL] checkpoint {ck_gb:.2f} GiB at {best_ckpt}; "
          f"72 cells projects to {ck_gb * 72:.1f} GiB on "
          f"{best_ckpt.parts[1] if len(best_ckpt.parts) > 1 else '?'}",
          flush=True)

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference")
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        geo = armc_dump.load_geo_meta(BASE)
        t_inf = time.time()
        rows, probs = infer_rows(model, loaders["test"], run_id,
                                 args.holdout, device, geo)
        inf_wall = time.time() - t_inf

        # Determinism: a second identical pass over the same checkpoint and the
        # same images. MAMNet-R34 was exactly 0.0000 (erratum E-9 is
        # OGLANet-only). A non-zero here extends E-9 to R50 and is flagged.
        det = {}
        if args.determinism:
            tbx.phase("determinism_pass2")
            rows2, _ = infer_rows(model, loaders["test"], run_id,
                                  args.holdout, device, geo)
            a = np.array([r["iou_tol"] for r in rows], dtype=np.float64)
            b = np.array([r["iou_tol"] for r in rows2], dtype=np.float64)
            det = {
                "determinism_passes": 2,
                "determinism_max_abs_delta_per_image": float(np.abs(a - b).max()),
                "determinism_mean_delta": float(np.abs(a - b).mean()),
                "determinism_cell_mean_delta": float(abs(a.mean() - b.mean())),
                "determinism_n_images_differing": int((a != b).sum()),
            }
            print(f"[armL] determinism: max |delta| "
                  f"{det['determinism_max_abs_delta_per_image']:.6g}, "
                  f"{det['determinism_n_images_differing']} of {len(a)} images "
                  f"differ", flush=True)

        tbx.phase("metrics")
        import pandas as pd
        df = pd.DataFrame(rows)
        fg_iou = float(df["iou_tol"].mean()) * 100.0
        miou2 = float(df["miou_tol_pct"].mean())
        pred_pos_frac = float((df["n_pred_pos"] / df["n_valid_px"]).mean())
        gh, gh_reason = git_hash()

        run = {
            "run_id": run_id, "dataset": "st", "arm": "L",
            "weakness_ids": ["zNBu-W5", "zNBu-W1", "AC1", "AC2"],
            "arch": ARCH, "arch_note": "MAMNet with a ResNet-50 encoder; "
                                       "python/shadowtransfer/mamnet_r50.py",
            "method": args.cell,
            "method_note": (
                "CONTRACT.md's method slot carries the 2x2 cell id in both "
                "modes. The 2x2 is a property of the model, not an adaptation "
                "method, and the upper bounds vary over it too; arm C's "
                "mode=upper -> method='none' convention would collide four "
                "cells into one run_id."),
            "mode": args.mode, "rung": RUNG,
            "n_train": len(ds_train), "n_train_per_source_city": n_per_city,
            "n_val": len(ds_val), "n_test": len(df),
            "n_counts_provenance": (
                "measured from the constructed datasets, not from a directory "
                "listing (CLAUDE.md: directory contents are not authoritative)"),
            "holdout": args.holdout, "res": "hr", "seed": args.seed,
            "seed_sites": seed_sites,

            "pretraining": cell["pretraining"],
            "lr": cell["lr"], "lr_label": cell["lr_label"],
            "cell_2x2": args.cell,
            "ssl_checkpoint": load_report.get("ckpt_name"),
            "ssl_checkpoint_sha256": (
                sha256_file(load_report["ckpt_path"])
                if load_report.get("ckpt_path") else None),
            "pretrained_load_report": load_report,
            "feat5_cosine_fresh_vs_loaded": {
                "mean": cos_mean, "min": cos_min, "max": cos_max,
                "n_images": int(fixed.shape[0]),
                "note": "far below 1.0 means the checkpoint changed the "
                        "encoder; 1.0 would mean the load silently did nothing"},
            "aux_attach_note": M.AUX_ATTACH_NOTE,

            "target_access": ("source_only" if args.mode == "loco"
                              else "target_labels"),
            "target_access_detail": (
                "LOCO trains and validates only on the two source cities and "
                "touches no held-out-city image before test-time inference"
                if args.mode == "loco" else
                "within-city upper bound: train and val are labelled images "
                "from the held-out city itself, disjoint from its test pool. "
                "target_labels by definition, and this AGREES with how arm C "
                "records its matching upper-bound cells. An earlier arm L spec "
                "fixed this field to source_only on every cell; the operator "
                "corrected it on 2026-07-31, BEFORE the grid ran, so no arm L "
                "grid cell carries the wrong value. This is reviewer BxRx-W4 "
                "applied to our own arm rather than left as a disagreement."),

            "decision_metric": "2class_tol_k2",
            "decision_metric_value_pct": round(tr["best_val_metric"], 4),
            "decision_metric_note": (
                "what checkpoint_best.pth was SELECTED on. Published MAMNet's "
                "own rule, kept unchanged."),
            "reporting_metric": "2class_tol_k2 (primary) and fg_only_tol_k2",
            "reporting_metric_note": (
                "the arm L primary table reports two-class tolerant K=2, "
                "comparable to arm C and Table 38; foreground-only K=2 is "
                "reported alongside from the same counts"),
            "headline_miou_2class_k2": miou2,
            "headline_fg_iou_k2_pct": fg_iou,

            "optimizer": f"Adam(lr={cell['lr']:g}, weight_decay={WEIGHT_DECAY:g})",
            "batch_size": BATCH_SIZE, "img_size": IMG_SIZE,
            "loss": f"CrossEntropyLoss + {AUX_WEIGHT} * mean(3 aux CE), "
                    f"MAMNet's own utils.losses.MAMNetLoss",
            "augmentation": "train: RandomHorizontalFlip(0.5), "
                            "RandomVerticalFlip(0.5). val/test: resize + "
                            "ImageNet normalisation only. MAMNet task_id=2.",
            "postprocessing": "connected components < 10 px removed "
                              "(utils.postprocessing.filter_small_predictions)",
            "grad_checkpoint": True,
            "grad_checkpoint_scope": "encoder layer1-layer4 and the four "
                                     "CrissCrossAttention modules",
            "grad_checkpoint_note": (
                "At ResNet-50 widths this model does not fit an A100-40GB at "
                "batch 8 and 384 px. Job 20643780 OOMed at 39.45 of 39.49 GiB "
                "inside decoder3.cca; with the attention blocks checkpointed, "
                "job 20643844 reached decoder4 and OOMed there, so the "
                "encoder's retained activations are the rest of it -- layer1 "
                "runs at full 384x384 with 256 channels, four times "
                "ResNet-34's width. Gradient checkpointing trades compute for "
                "memory and changes NO hyperparameter: batch size, LR, "
                "optimiser, schedule, loss, augmentation and the decision "
                "metric are untouched. It is exact, not approximately exact: "
                "the recompute context (mamnet_r50._BNStatsGuard) restores "
                "BatchNorm running statistics, without which momentum would be "
                "applied twice per optimiser step. Verified on 2026-07-30 "
                "against an uncheckpointed run: identical loss, max |grad "
                "delta| 0.000e+00 over 263 tensors, max |buffer delta| "
                "0.000e+00 over 216 BatchNorm buffers."),
            **tr,

            "train_list_sha256": list_sha256(train_pairs),
            "val_list_sha256": list_sha256(val_pairs),
            "test_list_sha256": list_sha256(test_pairs),
            **script_shas(),
            "git_hash": gh, "git_hash_reason": gh_reason,

            "cluster": "delta",
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_time_limit": os.environ.get("ARML_TIME_LIMIT"),
            "wall_clock_s": round(time.time() - t0, 2),
            "train_wall_clock_s": round(train_wall, 2),
            "inference_wall_clock_s": round(inf_wall, 2),
            "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
                                  if torch.cuda.is_available() else None,
            "peak_vram_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
                                     if torch.cuda.is_available() else None,
            "compute_note": "A100 GPU job on gpuA100x4; no CPU allocation "
                            "exists on this account",

            "ckpt_path": str(best_ckpt),
            "ckpt_bytes": tr["checkpoint_bytes"],
            "ckpt_root_note": (
                "checkpoints live on /scratch: 72 cells x 1.4 GB exceeds the "
                "/projects headroom below the 500G soft quota. Dumps are on "
                "/projects."),
            "parent_run_id": None, "legacy_run_dir": None,
            "structure_cols": "deferred",
            "prediction_recipe": (
                "argmax over the two class logits, then "
                "filter_small_predictions(min_pixels=10)"),
            "probs_note": "softmax(logits)[:, 1], float16, unthresholded",
            "band_width_headline": 2, "band_widths_written": [0, 2, 5],
            "aurc_definition_note": (
                "CONTRACT.md section 3 leaves the canonical AURC blank. "
                "aurc_fg/bg/all are aurc_trapz19; aurc_*_mean20 are the "
                "SIB-trainer plain mean over 20 points. Neither is declared "
                "canonical by this run."),
            "pred_pos_fraction": pred_pos_frac,
            **det,
            "smoke": smoke, "forced_rerun": bool(args.force),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        flags = []
        if pred_pos_frac > 0.5:
            flags.append(f"predicted-positive fraction {pred_pos_frac:.3f} > 0.5 "
                         f"(collapse to all-positive)")
        if tr["best_epoch"] <= 5:
            flags.append(f"best_epoch {tr['best_epoch']} <= 5")
        if not tr["early_stopped"]:
            # tr["epochs_max"] is the ceiling this run actually used. The
            # module global is restored before this point, so reading it here
            # would report 100 for a truncated smoke.
            flags.append(f"ran to the epoch ceiling ({tr['epochs_max']}); "
                         f"early stopping never fired")
        if int(df["empty_fg_valid"].sum()):
            flags.append(f"empty_fg_valid on {int(df['empty_fg_valid'].sum())} images")
        if float(df["iou_tol"].std()) < 1e-6:
            flags.append("per-image iou_tol has ~zero variance")
        if det and det.get("determinism_max_abs_delta_per_image", 0) > 0:
            flags.append(
                f"inference is NOT deterministic: max per-image |delta| "
                f"{det['determinism_max_abs_delta_per_image']:.6g} over 2 "
                f"passes. MAMNet-R34 was exactly 0.0000; this extends "
                f"erratum E-9 beyond OGLANet.")
        run["sanity_flags"] = flags

        tbx.phase("write_dump")
        armc_dump.write_dump(dump_dir, rows, probs, run)

    print("\n" + "=" * 78)
    print(f"[{run_id}]")
    print(f"  2-class mIoU K=2 (primary)  {miou2:.4f}")
    print(f"  fg-only IoU K=2             {fg_iou:.4f}")
    print(f"  decision metric (val)       {tr['best_val_metric']:.4f}%")
    print(f"  epochs_run / best_epoch     {tr['epochs_run']} / {tr['best_epoch']}"
          f"   early_stopped {tr['early_stopped']}")
    print(f"  pred-positive fraction      {pred_pos_frac:.4f}")
    print(f"  checkpoint                  {ck_gb:.2f} GiB")
    print(f"  wall clock                  {time.time() - t0:.0f}s "
          f"(train {train_wall:.0f}s, inference {inf_wall:.0f}s)")
    print(f"  peak VRAM alloc             {run['peak_vram_alloc_gb']} GiB")
    print(f"  sanity flags                {flags or 'none'}")
    print(f"  -> {dump_dir}")
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Probes P1-P5
# ---------------------------------------------------------------------------
#
# Every probe reads a completed LOCO cell and its matched within-city upper
# bound. They run as their own SLURM job, after the grid: a probe failure must
# not be able to cost a trained model (CLAUDE.md).
#
# The probe arch token passed to armc_probes_common is 'mamnet'. That is not a
# shortcut -- MAMNetR50 deliberately keeps the published submodule names
# (encoder / mscaf / decoder / aux_module), so ENCODER_PREFIXES ('encoder.'),
# ENCODER_FEATURE_KEY ('feat5') and the freeze/reinit helpers resolve to the
# same tensors they would on the published R34 model. The §4.4 encoder/decoder
# split is therefore the published one: MSCAF and the aux heads count as
# decoder.

def probe_dump_dir(out_root, run_id, suffix):
    return (Path(out_root) if out_root else BASE / "dumps") / "st" / f"{run_id}-{suffix}"


def load_cell(run_id, dumps_root):
    import pandas as pd
    d = (Path(dumps_root) if dumps_root else BASE / "dumps") / "st" / run_id
    if not (d / ".DONE").exists():
        raise SystemExit(f"FATAL no completed dump for {run_id} at {d}")
    return (json.loads((d / "run.json").read_text()),
            pd.read_parquet(d / "perimage.parquet"), d)


def align_on_image_id(df_probe, df_other, col):
    m = dict(zip(df_other["image_id"], df_other[col]))
    missing = [i for i in df_probe["image_id"] if i not in m]
    if missing:
        raise SystemExit(f"FATAL {len(missing)} image_ids absent from the "
                         f"comparison dump, first: {missing[:3]}")
    return np.array([m[i] for i in df_probe["image_id"]], dtype=np.float64)


def recovery(iou_probe, iou_loco, iou_upper):
    """Per-image R = (probe - LOCO) / (upper - LOCO), NaN where the headroom is
    below 0.02. The exclusion is on the DENOMINATOR: an image the LOCO model
    already scores as well as the upper bound carries no recoverable gap, and a
    near-zero denominator manufactures arbitrarily large |R|."""
    head = np.asarray(iou_upper) - np.asarray(iou_loco)
    ok = head >= R_EXCLUDE_MIN
    r = np.full(len(head), np.nan)
    r[ok] = (np.asarray(iou_probe)[ok] - np.asarray(iou_loco)[ok]) / head[ok]
    return r, ok, head


def r_summary(r, ok, label):
    from shadowtransfer.metrics import bootstrap_ci
    vals = r[ok]
    med, lo, hi = bootstrap_ci(vals, stat_fn=np.median, n_boot=10000, ci=0.95,
                               seed=42) if len(vals) else (np.nan,) * 3
    print(f"  {label}: median R {med:.4f} [{lo:.4f}, {hi:.4f}] "
          f"on {len(vals)} of {len(r)} images "
          f"({int((~ok).sum())} excluded, headroom < {R_EXCLUDE_MIN})",
          flush=True)
    return {"median_R": float(med), "R_ci95": [float(lo), float(hi)],
            "n_used": int(len(vals)), "n_excluded": int((~ok).sum()),
            "mean_R": float(np.nanmean(vals)) if len(vals) else float("nan"),
            "exclusion_rule": f"headroom (upper - LOCO) < {R_EXCLUDE_MIN}",
            "ci_method": "image-cluster bootstrap B=10,000, "
                         "shadowtransfer.metrics.bootstrap_ci"}


# --- P1: Section 4.2 histogram matching ------------------------------------

def probe_p1(model, device, run_id, holdout, seed, df_loco, df_upper,
             train_pairs, res_dir, test_pairs, geo, out_root):
    from shadowtransfer import armc_probes_ops as OPS
    print("\n--- P1  Section 4.2 histogram matching ---", flush=True)
    ch_cdf, gray_cdf, n_src = OPS.source_cdfs(train_pairs, res_dir, BASE)

    iou_loco = df_loco["iou_tol"].to_numpy(dtype=np.float64)
    iou_upper = align_on_image_id(df_loco, df_upper, "iou_tol")

    out = {"n_source_images": n_src,
           "matching_rules": {
               "headline": "grayscale CDF match with the per-pixel channel "
                           "ratio preserved -- the published Section 4.2 rule "
                           "(experiment_utils.histogram_match_image), the same "
                           "rule arm C used",
               "rides_along": "per-channel CDF match"},
           "source_statistics": "the cell's OWN 450 training images, not the "
                                "source cities' whole pools"}

    for label, matcher in (("gray", OPS.GrayMatcher(gray_cdf)),
                           ("per_channel", OPS.PerChannelMatcher(ch_cdf))):
        ds = build_dataset(test_pairs, "test", res_dir, augment=False)
        ds.fda_transform = matcher     # applied to the raw uint8 RGB array
        loader = make_loader(ds, 1, False, seed)
        rows, _ = infer_rows(model, loader, f"{run_id}-p1histmatch-{label}",
                             holdout, device, geo)
        import pandas as pd
        dfm = pd.DataFrame(rows)
        iou_m = align_on_image_id(df_loco, dfm, "iou_tol")
        r, ok, head = recovery(iou_m, iou_loco, iou_upper)
        out[label] = {
            "mean_iou_tol_matched": float(iou_m.mean()),
            "mean_iou_tol_loco": float(iou_loco.mean()),
            "mean_iou_tol_upper": float(iou_upper.mean()),
            "improved": bool(iou_m.mean() > iou_loco.mean()),
            "mean_delta_iou": float((iou_m - iou_loco).mean()),
            **r_summary(r, ok, f"P1 {label}"),
            "per_image_R": [None if np.isnan(x) else float(x) for x in r],
            "per_image_headroom": [float(x) for x in head],
        }
        out[f"{label}_rows"] = rows
    return out


# --- P2: Section 4.4 city-identity probe ------------------------------------

def probe_p2(model, device, seed, res_dir, out_root):
    """Read erratum E-8 first: the published probe taps the WRONG tensor on all
    three architectures, because `extract_features.py` picks its layer by a
    lexicographic sort. This probe taps feat5 -- the tensor MSCAF and the
    decoder actually consume -- by calling the encoder submodule directly.
    StandardScaler is fitted inside each CV fold, never on all data; the leak at
    thread1_1d.py:171 is not reproduced."""
    from shadowtransfer import armc_probes_ops as OPS
    from shadowtransfer import armc_probes_common as C
    from PIL import Image
    print("\n--- P2  Section 4.4 city-identity probe (feat5, E-8 aware) ---",
          flush=True)

    cities = sorted(FOLD_OF_HOLDOUT)
    pairs = [(c, f) for c in cities
             for f in city_split_files(c, res_dir, "test")]
    ds = build_dataset(pairs, "test", res_dir, augment=False)
    loader = make_loader(ds, BATCH_SIZE, False, seed)

    city_of_path = {os.path.abspath(str(BASE / "data" / "Final_data_test" / c /
                                        res_dir / "test" / "images" / f)): c
                    for c, f in pairs}
    img_city = np.array([cities.index(city_of_path[os.path.abspath(p)])
                         for p in ds.img_paths], dtype=np.int64)

    X, ids, shape, n_cells_total = OPS.extract_features(
        model, "mamnet", loader, device, CELLS_PER_IMAGE, seed)
    print(f"  feat5 {shape}, {n_cells_total} cells/image, "
          f"{CELLS_PER_IMAGE} sampled -> X {X.shape}", flush=True)
    res = OPS.city_identity_probe(X, img_city, CELLS_PER_IMAGE,
                                  n_folds=5, n_perm=N_PERM, seed=seed)
    per_img = res.pop("per_image_accuracy")

    # Raw-pixel control: six numbers per image, per-channel mean and std.
    ctrl_feats = []
    for p in ds.img_paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        ctrl_feats.append(np.concatenate([arr.mean(axis=(0, 1)),
                                          arr.std(axis=(0, 1))]))
    ctrl = OPS.city_identity_probe(np.asarray(ctrl_feats, dtype=np.float32),
                                   img_city, 1, n_folds=5, n_perm=0, seed=seed)
    ctrl.pop("per_image_accuracy", None)

    return {
        "tap": "feat5, the encoder's final feature map, called as a submodule "
               "rather than picked by name-sorting (erratum E-8)",
        "feat_shape": list(shape),
        "n_spatial_cells_total": int(n_cells_total),
        "n_spatial_cells_sampled": CELLS_PER_IMAGE,
        "n_images": int(len(img_city)),
        "cities": cities,
        "cv": "image-level GroupKFold(5); StandardScaler fitted inside each "
              "fold (the leak at thread1_1d.py:171 is not reproduced)",
        "probe": res,
        "raw_pixel_control": {
            "features": "per-channel mean and std of the RGB image, 6 numbers",
            **ctrl},
        "per_image_accuracy": [float(x) for x in per_img],
    }


# --- P3: Section 4.3 selective prediction -----------------------------------

def probe_p3(df_loco, dump_dir, test_pairs, res_dir, seed):
    """CONTRACT.md section 3's AURC definition is an OPEN BLANK. This probe does
    not pick one. It writes the risk-coverage ARRAYS per image on the canonical
    19-point grid, and reports both live AURC definitions side by side, so the
    blank can be closed later without rerunning inference.

    Ground truth comes from the same dataset object the dump's inference used,
    not from a re-read of the mask file: a second resize with a different
    interpolation would silently change the population the curve is computed on.
    """
    from shadowtransfer.metrics import COVERAGE_GRID, per_image_rc
    print("\n--- P3  Section 4.3 selective prediction (arrays, both AURCs) ---",
          flush=True)
    npz = np.load(dump_dir / "probs.npz")
    ds = build_dataset(test_pairs, "test", res_dir, augment=False)
    gts = {}
    for i in range(len(ds)):
        item = ds[i]
        gts[Path(item["filename"]).stem] = np.asarray(item["mask"]).astype(np.uint8)

    curves = []
    for stem in df_loco["image_id"]:
        if stem not in gts:
            raise SystemExit(f"FATAL no GT for {stem} in the test dataset")
        prob = npz[stem].astype(np.float32)
        gt = gts[stem]
        for mode, name in (("gt_shadow", "fg"), ("gt_bg", "bg"),
                           ("all_pixels", "all")):
            rc = per_image_rc(gt, prob, mode)
            for k, c in enumerate(COVERAGE_GRID):
                curves.append({"image_id": stem, "population": name,
                               "coverage": float(c),
                               "risk": float("nan") if rc is None else float(rc[k])})
    import pandas as pd
    dfc = pd.DataFrame(curves)
    dfc.to_parquet(dump_dir / "curves.parquet", index=False)
    print(f"  wrote {dump_dir/'curves.parquet'} ({len(dfc)} rows, "
          f"{len(COVERAGE_GRID)} coverage points x 3 populations x "
          f"{len(df_loco)} images)", flush=True)
    return {
        "curves_path": str((dump_dir / "curves.parquet").relative_to(BASE)),
        "coverage_grid": [float(c) for c in COVERAGE_GRID],
        "coverage_grid_note": (
            "metrics.COVERAGE_GRID = np.linspace(0.10, 1.00, 19), the grid the "
            "composition_analysis implementation uses. The arm spec asked for "
            "0.10..1.00 step 0.05, which IS this 19-point grid; it also spoke "
            "of '19 intervals' and '20 grid points', which the two live "
            "implementations do not share -- aurc_mean20 builds its own "
            "20-point grid internally."),
        "populations": {"fg": "gt shadow pixels, conf = p",
                        "bg": "gt background pixels, conf = 1 - p",
                        "all": "all pixels, conf = max(p, 1-p)"},
        "aurc_trapz19": {
            "fg": float(df_loco["aurc_fg"].mean()),
            "bg": float(df_loco["aurc_bg"].mean()),
            "all": float(df_loco["aurc_all"].mean())},
        "aurc_mean20": {
            "fg": float(df_loco["aurc_fg_mean20"].mean()),
            "bg": float(df_loco["aurc_bg_mean20"].mean()),
            "all": float(df_loco["aurc_all_mean20"].mean())},
        "definition_note": "neither definition is declared canonical here",
    }


# --- P4: Section 4.4 Experiments A and B ------------------------------------

def probe_p4(cell_name, device, run_id, holdout, seed, res_dir, df_loco,
             df_upper, test_pairs, geo, best_ckpt, out_root):
    """Exp A: freeze encoder, reinitialise and retrain the decoder.
    Exp B: freeze decoder, reset the encoder to its pretrained init, retrain it.

    Erratum E-10: App G.9's "encoder LR 100x smaller" is wrong; published
    MAMNet used 1x. Exp B here uses 1x the cell's own training LR and records
    exp_b_encoder_lr_multiplier = 1.0 explicitly.
    """
    import torch
    from shadowtransfer import armc_probes_common as C
    from shadowtransfer import armc_probes_ops as OPS
    print("\n--- P4  Section 4.4 Experiments A and B ---", flush=True)

    lr = CELLS[cell_name]["lr"]
    n_sub = int(RUNG * EXPAB_SUBSET_FRAC)
    names = city_split_files(holdout, res_dir, "train")
    idx = stable_rng("armL-expab", holdout, seed).permutation(len(names))[:n_sub]
    sub_pairs = [(holdout, names[i]) for i in sorted(idx)]
    ds_sub = build_dataset(sub_pairs, "train", res_dir, augment=True)
    loader_sub = make_loader(ds_sub, BATCH_SIZE, True, seed, drop_last=True)
    ds_test = build_dataset(test_pairs, "test", res_dir, augment=False)
    loader_test = make_loader(ds_test, 1, False, seed)

    iou_loco = df_loco["iou_tol"].to_numpy(dtype=np.float64)
    iou_upper = align_on_image_id(df_loco, df_upper, "iou_tol")
    ceiling_steps = EXPAB_EPOCHS * len(loader_sub)
    out = {"n_subset_images": len(ds_sub),
           "subset_fraction": EXPAB_SUBSET_FRAC,
           "epochs": EXPAB_EPOCHS,
           "steps_ceiling": ceiling_steps,
           "target_access": "target_labels",
           "target_access_detail": "the component-retraining experiments of "
                                   "Section 4.4 use held-out-city labels; this "
                                   "is the one place arm L does",
           "parent_run_id": run_id}

    for exp in ("A", "B"):
        model, _ = build_model(cell_name, device)
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        if exp == "A":
            C.freeze_encoder(model, "mamnet")
            n_reinit = C.reinit_decoder(model, "mamnet")
            frozen, exp_lr, extra = "encoder", lr, {"n_decoder_modules_reinit": n_reinit}
        else:
            fresh, _ = build_model(cell_name, device)
            snap = C.snapshot_encoder(fresh, "mamnet")
            del fresh
            torch.cuda.empty_cache()
            C.restore_encoder(model, snap)
            C.freeze_decoder(model, "mamnet")
            exp_lr = lr * EXPB_ENCODER_LR_MULT
            frozen, extra = "decoder", {
                "exp_b_encoder_lr_multiplier": EXPB_ENCODER_LR_MULT,
                "exp_b_encoder_lr": exp_lr,
                "exp_b_lr_note": (
                    "erratum E-10: App G.9 says the encoder LR is 100x smaller; "
                    "the published MAMNet runs used 1x. 1x is applied here and "
                    "the multiplier is recorded rather than implied.")}

        stats = OPS.retrain_component(model, "mamnet", loader_sub, device,
                                      frozen, EXPAB_EPOCHS, exp_lr)
        rows, _ = infer_rows(model, loader_test, f"{run_id}-p3exp{exp}",
                             holdout, device, geo)
        import pandas as pd
        dfe = pd.DataFrame(rows)
        iou_e = align_on_image_id(df_loco, dfe, "iou_tol")
        r, ok, _ = recovery(iou_e, iou_loco, iou_upper)
        budget_bound = stats["steps_to_best"] >= 0.9 * ceiling_steps
        out[f"exp{exp}"] = {
            **stats, **extra,
            "lr": exp_lr,
            "evaluated_checkpoint": "final (no validation, no selection) -- the "
                                    "published Section 4.4 recipe",
            "mean_iou_tol": float(iou_e.mean()),
            **r_summary(r, ok, f"P4 Exp {exp}"),
            "budget_bound": bool(budget_bound),
            "budget_bound_note": (
                f"steps_to_best {stats['steps_to_best']} of a "
                f"{ceiling_steps}-step ceiling; within 10% of the ceiling means "
                f"the 30-epoch budget binds and R is a LOWER BOUND"
                if budget_bound else
                f"steps_to_best {stats['steps_to_best']} of {ceiling_steps}; "
                f"the budget does not bind"),
            "per_image_R": [None if np.isnan(x) else float(x) for x in r],
        }
        out[f"exp{exp}_rows"] = rows
        del model, state
        torch.cuda.empty_cache()
    return out


# --- P5: encoder drift ------------------------------------------------------

def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between two n x d feature matrices.

    Not a paper metric and therefore not in shadowtransfer.metrics, which owns
    the tolerant IoU family, the risk-coverage curve and the recovery ratio.
    Written here, once, and its definition recorded in the probe dump.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, ord="fro") ** 2
    den = (np.linalg.norm(X.T @ X, ord="fro") *
           np.linalg.norm(Y.T @ Y, ord="fro"))
    return float(num / den) if den > 0 else float("nan")


def probe_p5(cell_name, device, seed, res_dir, test_pairs, best_ckpt):
    """Relative L2 weight drift and linear CKA between the pretrained init and
    the fine-tuned encoder. This is what lets 'low LR' be a MEASUREMENT rather
    than a label on a hyperparameter."""
    import torch
    from shadowtransfer import arme_e1_lib as E1
    print("\n--- P5  encoder drift (relative L2 + linear CKA) ---", flush=True)

    init_model, _ = build_model(cell_name, device)
    init_enc_sd = {k: v.detach().cpu().clone()
                   for k, v in init_model.encoder.state_dict().items()}
    param_keys = set(dict(init_model.encoder.named_parameters()).keys())

    ft_model, _ = build_model(cell_name, device)
    state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    ft_model.load_state_dict(state["model_state_dict"])
    ft_enc_sd = {k: v.detach().cpu().clone()
                 for k, v in ft_model.encoder.state_dict().items()}

    stages = ("conv1", "layer1", "layer2", "layer3", "layer4")
    drift = {"whole_encoder": E1.group_relative_l2(
        init_enc_sd, ft_enc_sd,
        [k for s in stages for k in E1.resnet_stage_keys(init_enc_sd, s, param_keys)])}
    for s in stages:
        keys = E1.resnet_stage_keys(init_enc_sd, s, param_keys)
        entry = E1.group_relative_l2(init_enc_sd, ft_enc_sd, keys)
        mods = {}
        for m in ("conv1", "conv2", "conv3", "downsample", "bn"):
            mk = [k for k in keys
                  if (f".{m}." in k or k.startswith(f"{m}.")) or
                  (m == "bn" and (".bn" in k or k.startswith("bn")))]
            if mk:
                mods[m] = E1.group_relative_l2(init_enc_sd, ft_enc_sd, mk)
        drift[s] = {**entry, "per_module": mods}
        print(f"  {s:8s} relative L2 {entry['relative_l2']:.6f} "
              f"over {entry['n_params']:,} params", flush=True)

    ds = build_dataset(test_pairs, "test", res_dir, augment=False)
    loader = make_loader(ds, BATCH_SIZE, False, seed)
    feats = {s: {"init": [], "ft": []} for s in ("layer1", "layer2", "layer3", "layer4")}
    init_model.eval()
    ft_model.eval()
    ft_model.to(device)
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            for tag, m in (("init", init_model), ("ft", ft_model)):
                out = m.encoder(x)
                for i, s in enumerate(("layer1", "layer2", "layer3", "layer4"), 1):
                    feats[s][tag].append(
                        out[f"feat{i}"].mean(dim=(2, 3)).cpu().numpy())
    cka = {s: linear_cka(np.concatenate(v["init"]), np.concatenate(v["ft"]))
           for s, v in feats.items()}
    for s, v in cka.items():
        print(f"  {s:8s} linear CKA init-vs-fine-tuned {v:.6f}", flush=True)

    del init_model, ft_model, state
    torch.cuda.empty_cache()
    return {
        "drift_rel_l2_per_stage": drift,
        "cka_per_stage": cka,
        "cka_definition": (
            "linear CKA, Kornblith et al. 2019: "
            "||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) on column-centred "
            "matrices. Each stage's feature map is global-average-pooled over "
            "space to one vector per image, so n = the test pool and d = the "
            "stage's channel count."),
        "cka_images": len(ds),
        "cka_note": "held-out city's test images, fixed order, no augmentation",
        "drift_note": (
            "W_init is a freshly constructed MAMNetR50 encoder carrying the "
            "cell's own pretraining -- the same construction the cell trained "
            "from. BatchNorm buffers are excluded from the weight comparison "
            "(arme_e1_lib.resnet_stage_keys filters on named_parameters)."),
    }


def write_probe(out_root, run_id, suffix, payload, rows=None):
    d = probe_dump_dir(out_root, run_id, suffix)
    armc_runtime.preflight_write_targets(
        BASE, [d / "run.json", d / ".DONE"], label=f"{suffix} dump")
    d.mkdir(parents=True, exist_ok=True)
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(d / "perimage.parquet", index=False)
    (d / "run.json").write_text(json.dumps(payload, indent=2, default=str))
    (d / ".DONE").touch()
    print(f"  -> {d}", flush=True)
    return d


def cmd_probes(args):
    t0 = time.time()
    res_dir = RES_DIR["hr"]
    loco_id = run_id_for(args.cell, "loco", args.holdout, args.seed)
    upper_id = run_id_for(args.cell, "upper", args.holdout, args.seed)
    want = [p.strip() for p in args.probes.split(",") if p.strip()]

    run_loco, df_loco, dump_loco = load_cell(loco_id, args.dumps_root)
    run_upper, df_upper, _ = load_cell(upper_id, args.dumps_root)
    best_ckpt = Path(run_loco["ckpt_path"])
    if not best_ckpt.exists():
        raise SystemExit(f"FATAL checkpoint recorded by {loco_id} is missing: "
                         f"{best_ckpt}")

    train_pairs, _, test_pairs, _ = resolve_selection(
        "loco", args.holdout, res_dir, args.seed)

    print("=" * 78)
    print(f"ARM L PROBES  {loco_id}")
    print(f"  upper comparator  {upper_id}")
    print(f"  checkpoint        {best_ckpt}")
    print(f"  probes            {want}")
    print(f"  slurm             {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    print("=" * 78, flush=True)
    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    import torch
    seed_everything(args.seed)
    enter_mamnet_tree()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    geo = armc_dump.load_geo_meta(BASE)

    common = {
        "arm": "L", "dataset": "st", "parent_run_id": loco_id,
        "upper_run_id": upper_id, "arch": ARCH, "method": args.cell,
        "holdout": args.holdout, "res": "hr", "seed": args.seed,
        "rung": RUNG, "mode": "loco",
        "pretraining": CELLS[args.cell]["pretraining"],
        "lr": CELLS[args.cell]["lr"], "lr_label": CELLS[args.cell]["lr_label"],
        "target_access": "source_only",
        "ckpt_path": str(best_ckpt),
        "probe_arch_token": (
            "armc_probes_common is called with arch='mamnet'. MAMNetR50 keeps "
            "the published submodule names, so ENCODER_PREFIXES, the feat5 tap "
            "and the freeze/reinit helpers resolve to the same tensors they "
            "would on the published R34 model."),
        **script_shas(),
        "git_hash": git_hash()[0], "git_hash_reason": git_hash()[1],
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "compute_note": "A100 GPU job on gpuA100x4",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    model = None
    if {"p1", "p2"} & set(want):
        model, _ = build_model(args.cell, device)
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

    with armc_runtime.PhaseTimeout(args.timeout_s,
                                   probe_dump_dir(args.out_root, loco_id, "probes"),
                                   label="probes") as tbx:
        if "p1" in want:
            tbx.phase("p1_histmatch")
            r = probe_p1(model, device, loco_id, args.holdout, args.seed,
                         df_loco, df_upper, train_pairs, res_dir, test_pairs,
                         geo, args.out_root)
            rows = r.pop("gray_rows", None)
            r.pop("per_channel_rows", None)
            write_probe(args.out_root, loco_id, "p1histmatch",
                        {**common, "probe": "p1_section4.2_histogram_matching",
                         **r}, rows)

        if "p2" in want:
            tbx.phase("p2_cityprobe")
            r = probe_p2(model, device, args.seed, res_dir, args.out_root)
            write_probe(args.out_root, loco_id, "p2cityprobe",
                        {**common, "probe": "p2_section4.4_city_identity",
                         "target_access": "unlabeled_target",
                         "target_access_detail":
                             "the probe reads held-out-city test images to "
                             "measure whether city identity is decodable; it "
                             "is a diagnostic on a finished model and changes "
                             "no weight",
                         **r})

        if "p3" in want:
            tbx.phase("p3_selective_prediction")
            r = probe_p3(df_loco, dump_loco, test_pairs, res_dir, args.seed)
            write_probe(args.out_root, loco_id, "p3selpred",
                        {**common, "probe": "p3_section4.3_selective_prediction",
                         **r})

        if "p4" in want:
            tbx.phase("p4_expAB")
            if model is not None:
                del model
                torch.cuda.empty_cache()
                model = None
            r = probe_p4(args.cell, device, loco_id, args.holdout, args.seed,
                         res_dir, df_loco, df_upper, test_pairs, geo,
                         best_ckpt, args.out_root)
            for exp in ("A", "B"):
                rows = r.pop(f"exp{exp}_rows")
                write_probe(args.out_root, loco_id, f"p3exp{exp}",
                            {**common, "probe": f"p4_section4.4_experiment_{exp}",
                             "target_access": "target_labels",
                             **{k: v for k, v in r.items()
                                if not k.startswith("exp") or k == f"exp{exp}"}},
                            rows)

        if "p5" in want:
            tbx.phase("p5_drift")
            r = probe_p5(args.cell, device, args.seed, res_dir, test_pairs,
                         best_ckpt)
            write_probe(args.out_root, loco_id, "p5drift",
                        {**common, "probe": "p5_encoder_drift", **r})

    print(f"\nPROBES DONE for {loco_id} in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Arm L — SSL-pretrained CNN "
                                             "under DINOv3's fine-tuning regime")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cell", required=True, choices=sorted(CELLS))
        p.add_argument("--holdout", required=True, choices=sorted(FOLD_OF_HOLDOUT))
        p.add_argument("--seed", required=True, type=int)
        p.add_argument("--dry_run", action="store_true")
        p.add_argument("--out_root", default=None,
                       help="override dumps/ root; every truncated run must "
                            "set it so a smoke dump never lands on a real path")
        p.add_argument("--timeout_s", type=int, default=1800)

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--mode", required=True, choices=["loco", "upper"])
    t.add_argument("--run_id", default=None)
    t.add_argument("--ckpt_root", default=None)
    t.add_argument("--force", action="store_true")
    t.add_argument("--smoke_epochs", type=int, default=0)
    t.add_argument("--smoke_test", type=int, default=0)
    t.add_argument("--determinism", action="store_true",
                   help="second identical inference pass; the E-9 check")

    p = sub.add_parser("probes")
    common(p)
    p.add_argument("--probes", default="p1,p2,p3,p4,p5")
    p.add_argument("--dumps_root", default=None)

    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_probes(args)


if __name__ == "__main__":
    main()
