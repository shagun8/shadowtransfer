#!/usr/bin/env python3
"""
Arm A / S-EO — one budget-matched external-LOCO cell, trained and dumped.

PHASE 1 of the REDUCED grid. Two MAMNet OMA cells resumed to convergence plus
one DINOv3 OMA LOCO cell whose first link is also the architecture's timing
probe. This script states no gap, computes no bootstrap, and writes no results
JSON.

OGLANet is DROPPED from arm A. It is deliberately absent from `--arch`'s
choices rather than merely unused, so a future session cannot resurrect it by
passing a string.

What is reused, verbatim
------------------------
  - the published Trainer of the selected architecture, constructed from a
    published `args.json`, with only its dataloaders replaced. Same model,
    loss, optimizer, scheduler, augmentation policy and model-selection metric.
      mamnet  mamnet/train.py        RECIPE data/mamnet/outputs/mamnet_phoenix_highres_1/args.json
      dinov3  dinov3/train_dinov3.py RECIPE data/dinov3/outputs/dinov3_phoenix_highres_1/args.json
  - `shadowtransfer.armc_runtime` for the figure/tensorboard silencing, the
    collapse guard, the post-training time-box and the quarantine preflight.
    An unwired cell hangs ~90 minutes inside the published figure call; that has
    already cost one session.
  - `shadowtransfer.metrics` as the ONLY metric implementation (CONTRACT.md
    rule zero), through `shadowtransfer.seo_dump`.
  - `shadowtransfer.seo_dataset.SEODataset`, unmodified.

What differs from arm C, and why
--------------------------------
  1. The dataset is S-EO, not ShadowTransfer. Patches, not images: 384 px tiles
     cut at native ~0.3 m GSD, no resize.
  2. The budget is 3500 train / 900 val patches in EVERY cell, drawn AOI-first
     by `shadowtransfer.seo_budget`. Test is the held-out city's full test
     split, never subsampled, byte-identical between the fold's two cells.
  3. DSM-unseen pixels leave the metric entirely (numerator and denominator).
     See `seo_dump` for the mechanism -- nothing in metrics.py is edited.
  4. The TRAINING loss also drops unseen pixels, via CrossEntropyLoss
     reconfigured with ignore_index=255 and the loader's own `mask_ignore`
     target. This is the prep phase's committed rule
     (armA_seo_census_acq3.json, valid_region_rule).
     LIMITATION, recorded rather than hidden: the published trainer's own
     VALIDATION decision metric (DetailedEvaluator 2-class tolerant mIoU, which
     selects checkpoint_best.pth) still sees unseen pixels under their raw
     `_shadow` label. Feeding it a 255 sentinel would corrupt the erode/dilate
     that builds its +-K don't-care band, and patching the published evaluator
     is a contract change. Both cells use the identical rule, so it does not
     act differently on the two sides of the comparison. run.json records this
     as `unseen_handling`.
  5. Periodic checkpointing is ON at save_freq=1 and pruned to the newest one,
     because a 3500-patch cell can exceed the 3:59:59 wall clock and CLAUDE.md
     asks long cells to checkpoint and resume rather than request more time.
     Arm C switches periodic checkpoints off; this cell cannot afford to.
     All but checkpoint_best.pth are deleted once the dump is written.
  6. The TRAIN-side DetailedEvaluator is DISABLED. See below.

The train-side evaluator, and the authority it is disabled under
----------------------------------------------------------------
`--disable_train_eval` needs `--train_eval_authority`, and the two authorities
are not interchangeable:

  equivalence_pass  the stage-1 bit-identical proof passed. `--equiv_artifact`
                    is re-read and must carry verdict == "PASS".
  code_audit        OPERATOR DECISION, on the record 2026-08-02. The stage-1
                    proof returned INCONCLUSIVE -- not FAIL -- because the two
                    CONTROL runs, which differ in nothing at all, already
                    diverge at step 1 under torch defaults
                    (cudnn.deterministic and cudnn.benchmark both False, never
                    overridden in this project, and overriding them for the
                    proof would prove a configuration that does not ship). The
                    proof as specified is therefore unexecutable here. The
                    lever is taken on the separability audit instead: train and
                    val evaluators are distinct objects (train.py:215 vs :217),
                    every reference to either is one-sided, and the train-side
                    result is a local that `train_epoch` never returns,
                    reaching two prints and a NullWriter.

Under `code_audit` the artifact is still read and its sha still recorded, so
run.json names the exact evidence the decision was taken against; what changes
is only that verdict != "PASS" no longer refuses. Nothing weakens on the
VALIDATION side: that evaluator computes the boundary-tolerant decision metric
that selects checkpoint_best.pth and drives patience-20, and it is untouched.

Checkpoints go to scratch, not /projects: the archive root is inside its soft-quota
grace period (456 G of a 500 G soft / 550 G hard limit, measured 2026-08-01).

Usage
-----
    python armA_pilot_train.py --arch mamnet \
        --run_id mamnet-vanilla-loco-r3500-oma-hr-s0 \
        --condition loco --holdout OMA --seed 0 ...
    ... --dry_run                     resolve and print, write nothing
    ... --smoke_train 64 --smoke_val 32 --smoke_epochs 2 --smoke_test 5 \
        --out_root <scratch>          stage-1 smoke
    ... --dump_only                   skip training, dump from checkpoint_best
    ... --probe_json logs/x.json      flush a per-epoch timing probe every epoch
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_runtime                        # noqa: E402
from shadowtransfer import seo_budget                          # noqa: E402
from shadowtransfer import seo_dump                            # noqa: E402
from shadowtransfer import seo_traineval                       # noqa: E402
from shadowtransfer.armc_dump import sha256_file               # noqa: E402
from shadowtransfer.seo_dataset import SEODataset, assert_no_leakage  # noqa: E402

RES = "hr"
RUNG = seo_budget.N_TRAIN                 # the budget IS the rung for this arm
PATIENCE_EPOCHS = 20                      # arm C's ladder recipe, flat
EPOCHS_MAX = 100                          # arm C's ladder recipe
IGNORE_INDEX = 255
DATASET = "seo"

#: MAMNet's measured eval-OFF rate on this exact cell shape (3500 train / 900
#: val), from logs/armA_evalfree_equivalence.json arm b. Every other
#: architecture's cost is reported as a multiple of it.
MAMNET_MIN_PER_EPOCH_EVAL_OFF = 5.1673

#: One recipe config per architecture, used for BOTH cells of a fold so they
#: differ ONLY in the data. The two published MAMNet configs (loco and upper)
#: were diffed on 2026-08-01 and are identical on every training knob; they
#: differ only in data-routing keys, which this script overrides anyway because
#: it replaces the dataloaders. The same holds for DINOv3.
#:
#: use_contrast follows the architecture's published input contract, NOT a
#: choice made here: MAMNet's 144 published cells all set use_contrast=true and
#: consume RGBC [4,H,W]; DINOv3's consume plain RGB, which is also what arm C's
#: DINOv3 ladder cells feed it (armc_train.build_dataset -> data.dataset.Dataset).
ARCH_SPEC = {
    "mamnet": {
        "dir": "mamnet",
        "train_module": "train",
        "recipe": "data/mamnet/outputs/mamnet_phoenix_highres_1/args.json",
        "use_contrast": True,
        "in_channels": 4,
    },
    "dinov3": {
        "dir": "dinov3",
        "train_module": "train_dinov3",
        "recipe": "data/dinov3/outputs/dinov3_phoenix_highres_1/args.json",
        "use_contrast": False,
        "in_channels": 3,
    },
}
#: OGLANet is dropped from arm A by operator decision (2026-08-02). Not listed
#: above, so `--arch oglanet` fails at argument parsing rather than silently
#: training a cell nothing downstream expects.
ARCH_DROPPED = {"oglanet": "dropped from arm A by operator decision 2026-08-02"}

DINOV3_WEIGHTS = BASE / "python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def seed_everything(seed):
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
        "dataloader_shuffle": f"torch.Generator().manual_seed({seed}) via DataLoader(generator=)",
        "dataloader_workers": "dl_worker_init seeds python random and numpy per worker",
        "cudnn": ("left at torch defaults (deterministic=False, benchmark=False), "
                  "as everywhere else in this project. Forcing either would "
                  "change the recipe; see logs/armA_evalfree_equivalence.json "
                  "for what that costs the equivalence proof."),
        "patch_selection": ("NOT seeded: seo_budget ranks AOIs by "
                            "sha256(SPLIT_SALT|aoi_id), the same rule the splits "
                            "used, so every seed of a cell sees the identical "
                            "3500/900 patches and the seed varies only the "
                            "optimisation"),
    }


def dl_worker_init(worker_id):
    import torch
    s = torch.initial_seed() % (2 ** 32)
    random.seed(s)
    np.random.seed(s)


BEST_NAME = "checkpoint_best.pth"
BEST_PREV_NAME = "checkpoint_best_prev.pth"


def install_checkpoint_pruner(archtrain, verbose=True):
    """Keep exactly one periodic checkpoint, the newest.

    save_freq=1 makes a killed cell resumable to within one epoch, but the
    published save_checkpoint writes a uniquely-named checkpoint_epoch_N.pth
    every time; at ~509 MB and up to 100 epochs that is ~51 GB per cell against
    a /projects filesystem already inside its soft-quota grace period. This
    wrapper deletes every periodic checkpoint but the newest immediately after
    each save, so a cell costs 2 checkpoints on disk, not 100.

    Wraps whatever is already installed (armc_runtime's collapse guard) rather
    than replacing it.
    """
    T = getattr(archtrain, "Trainer", None)
    if T is None or not hasattr(T, "save_checkpoint"):
        return {"installed": False, "reason": "no Trainer.save_checkpoint"}
    orig = T.save_checkpoint
    if getattr(orig, "_st_ckpt_pruner", False):
        return {"installed": True, "reason": "already installed"}

    def pruning_save(self, epoch, is_best=False, *a, **k):
        out = orig(self, epoch, is_best, *a, **k)
        try:
            cks = sorted(Path(self.output_dir).glob("checkpoint_epoch_*.pth"),
                         key=lambda q: int(q.stem.split("_")[-1]))
            for q in cks[:-1]:
                q.unlink()
        except OSError as e:
            print(f"[armA] checkpoint prune failed (non-fatal): {e}", flush=True)
        # Keep a VERIFIED copy of the best checkpoint. A wall-clock SIGKILL can
        # land inside the 509 MB best-save, and a truncated checkpoint_best.pth
        # is the cell's result. Verifying costs one read per improvement and
        # buys a fallback that is exactly one improvement old.
        if is_best:
            try:
                best = Path(self.output_dir) / BEST_NAME
                if best.exists() and _loads(best):
                    import shutil
                    tmp = Path(self.output_dir) / (BEST_PREV_NAME + ".new")
                    shutil.copyfile(best, tmp)
                    tmp.replace(Path(self.output_dir) / BEST_PREV_NAME)
            except OSError as e:
                print(f"[armA] best-checkpoint mirror failed (non-fatal): {e}",
                      flush=True)
        return out

    pruning_save._st_ckpt_pruner = True
    T.save_checkpoint = pruning_save
    if verbose:
        print("[armA] checkpoint pruner LIVE: keep newest checkpoint_epoch_*.pth "
              "only (save_freq=1 for resume)", flush=True)
    return {"installed": True, "reason": "ok",
            "rule": "keep only the newest checkpoint_epoch_*.pth after every save"}


def _loads(path):
    """True when torch can actually read this checkpoint."""
    import torch
    try:
        torch.load(str(path), map_location="cpu", weights_only=False)
        return True
    except Exception as e:
        print(f"[armA] {Path(path).name} does not load: "
              f"{type(e).__name__}: {str(e)[:120]}", flush=True)
        return False


def newest_epoch_checkpoint(out_dir):
    """Newest periodic checkpoint that ACTUALLY LOADS, newest-first.

    A link that hits the SLURM wall clock is SIGKILLed, and the kill can land
    in the middle of a 509 MB torch.save. That is not hypothetical: job
    20747581 was killed at 2026-08-01T21:29:15 part-way through writing
    checkpoint_epoch_10.pth, leaving a 381,681,664-byte file that raises

        RuntimeError: PytorchStreamReader failed reading zip archive:
        failed finding central directory

    while the intact checkpoint_epoch_9.pth sat beside it. Sorting by epoch
    number alone picks the corpse, so every link of the chain would have died
    on resume and the cell would have lost ~40 GPU-hours silently. The
    truncation recurs at EVERY link boundary, so this check is permanent, not
    a one-off cleanup.

    A file that does not load is renamed to *.corrupt rather than deleted: it
    is the evidence that the kill landed mid-write, and it stops the glob from
    finding it again.
    """
    cks = sorted(Path(out_dir).glob("checkpoint_epoch_*.pth"),
                 key=lambda q: int(q.stem.split("_")[-1]), reverse=True)
    for q in cks:
        if _loads(q):
            return str(q)
        target = q.with_suffix(".pth.corrupt")
        try:
            q.rename(target)
            print(f"[armA] renamed {q.name} -> {target.name} (truncated by a "
                  f"mid-write SIGKILL); falling back to the previous epoch",
                  flush=True)
        except OSError as e:
            print(f"[armA] could not rename {q.name}: {e}", flush=True)
    return None


def resolve_best_checkpoint(run_out):
    """checkpoint_best.pth, or its last verified copy if the kill hit that one.

    Same failure mode as the periodic checkpoints, but far worse: a truncated
    checkpoint_best.pth is the cell's actual result. install_checkpoint_pruner
    keeps a verified copy beside it after every improvement, so the run falls
    back one improvement instead of losing the cell.
    """
    best = Path(run_out) / BEST_NAME
    prev = Path(run_out) / BEST_PREV_NAME
    if best.exists() and _loads(best):
        return best
    if best.exists():
        try:
            best.rename(best.with_suffix(".pth.corrupt"))
        except OSError:
            pass
    if prev.exists() and _loads(prev):
        print(f"[armA] {BEST_NAME} was truncated; falling back to "
              f"{BEST_PREV_NAME} (one improvement older)", flush=True)
        return prev
    return None


def write_json_atomic(path, payload):
    """{path}.new, validate, mv. Never write in place."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".new")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    json.loads(tmp.read_text())          # a file that does not parse never lands
    tmp.replace(p)
    return str(p)


# ---------------------------------------------------------------------------
# Dataset assembly: the arm's own loader, then filtered to the budget manifest
# ---------------------------------------------------------------------------

class PatchSubset:
    """`SEODataset` restricted to an explicit patch-id list, in that order.

    Filtering after construction rather than reimplementing the loader keeps
    every transform, the contrast channel and the valid-region logic exactly as
    `seo_dataset.py` defines them.
    """

    def __init__(self, base, patch_ids, label):
        self.base = base
        self.city, self.split = base.city, base.split
        pos = {}
        for i, r in enumerate(base.rows):
            pos[f"{r['image_id']}_y{r['y']}_x{r['x']}"] = i
        missing = [p for p in patch_ids if p not in pos]
        if missing:
            raise SystemExit(
                f"FATAL {label}: {len(missing)} budget patch ids are absent from "
                f"the {base.city}/{base.split} loader (first: {missing[:3]})")
        self.index = [pos[p] for p in patch_ids]
        self.rows = [base.rows[i] for i in self.index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        return self.base[self.index[i]]


class TrainTargets:
    """Hand the LOSS `mask_ignore` (255 outside the valid region).

    The trainer's CrossEntropyLoss is reconfigured with ignore_index=255 by the
    caller, so unseen pixels contribute nothing to the gradient. Applied to the
    TRAIN split only: the val split keeps its plain {0,1} target because the
    published DetailedEvaluator builds its don't-care band by eroding and
    dilating the target, and a 255 there would corrupt that morphology.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = dict(self.base[i])
        d["mask"] = d["mask_ignore"]
        return d


def city_of(patch_id):
    return patch_id.split("_", 1)[0]


def build_split(img_root, mask_root, patch_index, splits_dir, split, patch_ids,
                augment, label, use_contrast):
    """One PatchSubset per contributing city, concatenated in city order."""
    import torch
    from torch.utils.data import ConcatDataset
    by_city = {}
    for p in patch_ids:
        by_city.setdefault(city_of(p), []).append(p)
    parts, bases = [], []
    for city in sorted(by_city):
        base = SEODataset(
            img_root=img_root, mask_root=mask_root, patch_index=patch_index,
            split_manifest=str(Path(splits_dir) / f"{city}.json"), split=split,
            patch=384, augment=augment, use_contrast=use_contrast,
            exclude_unseen=True, exclude_uncertain=False)
        parts.append(PatchSubset(base, by_city[city], f"{label}/{city}"))
        bases.append(base)
    ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    rows = [r for p in parts for r in p.rows]
    if len(rows) != len(patch_ids):
        raise SystemExit(f"FATAL {label}: assembled {len(rows)} of {len(patch_ids)}")
    del torch
    return ds, rows, bases, parts


def patch_index_map(bases):
    """patch_id -> index of the patch inside its own source image."""
    out = {}
    for b in bases:
        seen = {}
        for r in b.rows:
            k = r["image_id"]
            out[f"{k}_y{r['y']}_x{r['x']}"] = seen.get(k, 0)
            seen[k] = seen.get(k, 0) + 1
    return out


def route_config_to_this_cluster(a, arch):
    """Point every data-routing key at a path that exists on THIS cluster.

    The Trainer builds its own dataloaders in __init__ from these before this
    script replaces them, so they must resolve even though nothing trains on
    them. DINOv3's published args.json was written on Anvil and carries
    /anvil/... paths, which do not exist on Delta -- the constructor would die
    in get_dataloaders long before the S-EO loaders were installed. MAMNet's
    published args.json already carries Delta paths, so for MAMNet this is a
    no-op that is applied anyway rather than branched around.

    Recorded in run.json as `config_rerouted_keys`.
    """
    rerouted = {}
    root = str(BASE / "data" / "Final_data_test") + "/"
    want = {
        "data_root": root + "phoenix/highres/",
        "base_data_root": None,
        "comparison_data_root": root,
        "comparison_inference_dir": str(BASE / "data" / "Test_img_results") + "/",
    }
    for k, v in want.items():
        if k in a and a.get(k) != v:
            if k == "base_data_root" and a.get(k) is None:
                continue
            rerouted[k] = {"was": a.get(k), "now": v}
            a[k] = v
    if arch == "dinov3":
        if a.get("weights_path") != str(DINOV3_WEIGHTS):
            rerouted["weights_path"] = {"was": a.get("weights_path"),
                                        "now": str(DINOV3_WEIGHTS)}
            a["weights_path"] = str(DINOV3_WEIGHTS)
        if not DINOV3_WEIGHTS.exists():
            raise SystemExit(f"FATAL DINOv3 pretrained weights missing: "
                             f"{DINOV3_WEIGHTS}")
    return rerouted


def set_ignore_index_loss(trainer, arch):
    """Drop unseen pixels from the training loss, per architecture.

    MAMNet's `MAMNetLoss` holds an inner CrossEntropyLoss at
    `trainer.criterion.criterion` and is called once per training step;
    `validate()` calls that inner object directly. DINOv3 has no wrapper -- its
    `trainer.criterion` IS the CrossEntropyLoss, and both train_epoch and
    validate call it. Replacing it there is still correct: the val split hands
    a plain {0,1} target, which contains no 255, so ignore_index is inert on
    validation and active on training. Neither published file is edited.
    """
    import torch.nn as nn
    if arch == "mamnet":
        trainer.criterion.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        where = "trainer.criterion.criterion (inner CE inside MAMNetLoss)"
    elif arch == "dinov3":
        trainer.criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        where = ("trainer.criterion (DINOv3 has no loss wrapper; inert on the "
                 "val split, whose target carries no 255)")
    else:
        raise SystemExit(f"FATAL no ignore_index route for arch {arch}")
    print(f"[armA] loss -> CrossEntropyLoss(ignore_index={IGNORE_INDEX}) at "
          f"{where}", flush=True)
    return {"ignore_index": IGNORE_INDEX, "installed_at": where}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mamnet", choices=sorted(ARCH_SPEC),
                    help="oglanet is intentionally absent: dropped from arm A")
    ap.add_argument("--run_id", required=True)
    # ---- WAVE 2 / D-A2-1: from-scratch RETRAIN of a ceiling-bound cell.
    # BOTH DEFAULT TO THE CURRENT BEHAVIOUR: --run_id_suffix "" and
    # --epochs_max 0 resolve every pre-existing invocation exactly as
    # before, so the 100+ cells that record this file as
    # train_script_sha keep the recipe they had. What moves is this
    # file's sha256; a retrained cell records the delta as
    # retrain_script_sha beside the sha the ORIGINAL cell carried.
    ap.add_argument("--run_id_suffix", default="",
                    help="appended to the derived run_id, so a retrain "
                         "takes a NEW dump path and the original dump is "
                         "never opened for writing. Empty by default.")
    ap.add_argument("--epochs_max", type=int, default=0,
                    help="the train-loop ceiling. 0 keeps the recipe's "
                         "EPOCHS_MAX. A from-scratch retrain has ONE "
                         "horizon, so there is no schedule denominator "
                         "to pin and no second variable -- see D-A2-1.")
    ap.add_argument("--condition", required=True, choices=["loco", "upper"])
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--budget_dir", required=True)
    ap.add_argument("--patch_index", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out_root", default=None, help="override dumps/ root")
    ap.add_argument("--keep_latest", action="store_true",
                    help="retain checkpoint_latest.pth on completion "
                         "(D-A4-1); implied when stopped_on==epoch_ceiling")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--timeout_s", type=int, default=3600)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dump_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke_epochs", type=int, default=0)
    ap.add_argument("--smoke_train", type=int, default=0)
    ap.add_argument("--smoke_val", type=int, default=0)
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--disable_train_eval", action="store_true",
                    help="switch off the TRAIN-side DetailedEvaluator")
    ap.add_argument("--train_eval_authority", default=None,
                    choices=["equivalence_pass", "code_audit"],
                    help="required with --disable_train_eval; see the module "
                         "docstring for what each one permits")
    ap.add_argument("--equiv_artifact", default=None,
                    help="stage-1 equivalence artifact; its sha256 is recorded "
                         "in run.json as train_eval_equivalence_sha")
    ap.add_argument("--train_eval_off_from_epoch", type=int, default=1,
                    help="the CELL's first eval-OFF epoch, counted in epochs of "
                         "TRAINING. A cell-level constant, not a per-link one: "
                         "1 for a cell started under the lever, 10 for the two "
                         "MAMNet cells whose first 9 epochs predate it.")
    ap.add_argument("--train_eval_off_from_epoch_label", type=int, default=1,
                    help="the same epoch as the published trainer PRINTS it. "
                         "Differs from the count by one per resume boundary "
                         "that precedes it; see resume_epoch_label_offset_note.")
    ap.add_argument("--probe_json", default=None,
                    help="flush a per-epoch timing/VRAM probe here after EVERY "
                         "epoch, so a link killed on its wall clock still "
                         "leaves its measurement behind")
    args = ap.parse_args()
    # D-A2-1: one horizon. EPOCHS_CEIL is what the loop, the stop rule
    # and the sanity flag all read; EPOCHS_MAX stays the recipe's value
    # and is reported beside it.
    EPOCHS_CEIL = args.epochs_max or EPOCHS_MAX

    t0 = time.time()
    arch = args.arch
    spec = ARCH_SPEC[arch]
    recipe_config = BASE / spec["recipe"]
    smoke = bool(args.smoke_epochs or args.smoke_train or args.smoke_val or args.smoke_test)
    if smoke and args.out_root is None:
        raise SystemExit("FATAL --smoke_* requires --out_root")
    if args.disable_train_eval and not args.train_eval_authority:
        raise SystemExit(
            "FATAL --disable_train_eval requires --train_eval_authority. "
            "Disabling the train-side evaluator without naming the authority "
            "on the record is exactly the undeclared recipe change this arm "
            "may not make.")

    mode = args.condition
    method = "vanilla" if mode == "loco" else "none"
    expect = f"{arch}-{method}-{mode}-r{RUNG}-{args.holdout.lower()}-{RES}-s{args.seed}"
    if args.run_id_suffix:
        expect = f"{expect}-{args.run_id_suffix}"
    if args.run_id != expect:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {expect}")

    dump_dir = (Path(args.out_root) if args.out_root else BASE / "dumps") / DATASET / args.run_id
    if (dump_dir / ".DONE").exists() and not args.force:
        print(f"[{args.run_id}] .DONE exists at {dump_dir} - nothing to do")
        return

    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet", dump_dir / "probs.npz",
        dump_dir / "run.json", dump_dir / ".DONE"], label="dump")
    if args.probe_json:
        armc_runtime.preflight_write_targets(BASE, [Path(args.probe_json)],
                                             label="probe")

    man_path = Path(args.budget_dir) / f"{args.holdout}_{mode}.json"
    man = json.loads(man_path.read_text())
    if man["budget_rule_sha"] != seo_budget.budget_rule_sha():
        raise SystemExit(
            f"FATAL manifest budget_rule_sha {man['budget_rule_sha'][:16]} != the "
            f"live rule {seo_budget.budget_rule_sha()[:16]}; rebuild the manifest")

    idx = json.loads(Path(args.patch_index).read_text())
    out_dir_parent = Path(args.ckpt_root) / args.run_id

    print("=" * 78)
    print(f"ARM A / S-EO  run_id      {args.run_id}")
    print(f"  arch                    {arch}  (train module {spec['train_module']}, "
          f"{spec['in_channels']}-channel input, use_contrast={spec['use_contrast']})")
    print(f"  dropped from this arm   {sorted(ARCH_DROPPED)}")
    print(f"  condition / holdout     {mode} / {args.holdout}  sources={man['source_cities']}")
    print(f"  budget                  train {man['n_train']}  val {man['n_val']}  "
          f"test {man['n_test']} (full, {man['n_test_aoi']} AOIs)")
    print(f"  manifest                {man_path}")
    print(f"  split_rule_sha          {man['split_rule_sha'][:16]}")
    print(f"  budget_rule_sha         {man['budget_rule_sha'][:16]}")
    print(f"  recipe config           {recipe_config}")
    print(f"  checkpoints             {out_dir_parent}")
    print(f"  dump                    {dump_dir}")
    print(f"  probe                   {args.probe_json or 'none'}")
    print(f"  train-side evaluator    "
          f"{'DISABLED under ' + str(args.train_eval_authority) if args.disable_train_eval else 'ON (arm C recipe)'}")
    print(f"  slurm job               {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    for k, d in man["per_quota_detail"].items():
        print(f"  boundary AOI {k:12s} {d['partial_aoi_id']} contributed "
              f"{d['partial_aoi_n']} of {d['partial_aoi_total']}  "
              f"({d['n_aoi_used']} AOIs, {d['quota_fraction_of_pool']:.1%} of pool)")
    if smoke:
        print(f"  SMOKE  epochs<={args.smoke_epochs} train<={args.smoke_train} "
              f"val<={args.smoke_val} test<={args.smoke_test}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    import torch                                                # noqa: E402
    from torch.utils.data import DataLoader                     # noqa: E402
    seed_sites = seed_everything(args.seed)

    tr_ids, va_ids, te_ids = man["train"], man["val"], man["test"]
    if smoke:
        tr_ids = tr_ids[:args.smoke_train or len(tr_ids)]
        va_ids = va_ids[:args.smoke_val or len(va_ids)]
        te_ids = te_ids[:args.smoke_test or len(te_ids)]

    uc = spec["use_contrast"]
    ds_train, rows_tr, bases_tr, parts_tr = build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "train", tr_ids, True, "train", uc)
    ds_val, rows_va, bases_va, parts_va = build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "val", va_ids, False, "val", uc)
    ds_test, rows_te, bases_te, parts_te = build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "test", te_ids, False, "test", uc)

    leak = assert_no_leakage(bases_tr + bases_va + bases_te)
    pidx = patch_index_map(bases_tr + bases_va + bases_te)

    aoi_tr = {r["aoi_id"] for r in rows_tr}
    aoi_va = {r["aoi_id"] for r in rows_va}
    aoi_te = {r["aoi_id"] for r in rows_te}
    for a_, b_, lbl in ((aoi_tr, aoi_va, "train/val"), (aoi_tr, aoi_te, "train/test"),
                        (aoi_va, aoi_te, "val/test")):
        if a_ & b_:
            raise SystemExit(f"FATAL AOI overlap {lbl}: {sorted(a_ & b_)[:5]}")

    train_sun = np.array([r["sun_el"] for r in rows_tr], dtype=np.float64)
    train_sun_mean, train_sun_sd = float(train_sun.mean()), float(train_sun.std(ddof=1))
    print(f"\nTRAIN SUN ELEVATION  mean {train_sun_mean:.3f}  sd {train_sun_sd:.3f}  "
          f"n {len(train_sun)}   (sun_dist is |patch - this mean|, per CELL)", flush=True)
    print(f"LEAKAGE CHECK  {leak}", flush=True)

    cfg = json.loads(recipe_config.read_text())
    arch_root = BASE / "python" / spec["dir"]
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    # shadowtransfer.metrics puts python/mamnet on sys.path and imports
    # utils.contrast_utils from it, so `utils` is already bound to MAMNet's
    # copy. Drop it, or the DINOv3 trainer would silently import MAMNet's
    # losses and metrics instead of its own.
    for _m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_m]
    archtrain = __import__(spec["train_module"])

    disabled = armc_runtime.silence_worker_side_effects(archtrain)
    pruner = install_checkpoint_pruner(archtrain)

    from argparse import Namespace
    saved_argv, sys.argv = sys.argv, [spec["train_module"]]
    try:
        defaults = vars(archtrain.get_args())
    finally:
        sys.argv = saved_argv
    defaults_filled = {k: defaults[k] for k in sorted(set(defaults) - set(cfg))}
    if defaults_filled:
        print(f"CONFIG defaults filled from the arch's own parser: "
              f"{defaults_filled}", flush=True)
    a = {**defaults, **cfg}
    rerouted = route_config_to_this_cluster(a, arch)
    if rerouted:
        print(f"CONFIG rerouted to this cluster: {sorted(rerouted)}", flush=True)
    a["output_dir"] = str(out_dir_parent)
    a["device"] = "cuda"
    a["eval_only"] = False
    a["resume"] = None
    a["num_workers"] = args.num_workers
    a["early_stopping_patience"] = PATIENCE_EPOCHS
    a["epochs"] = args.smoke_epochs or EPOCHS_CEIL
    a["save_freq"] = 1
    a["seed"] = args.seed
    a["armA_arch"] = arch
    a["armA_run_id"] = args.run_id
    a["armA_dataset"] = DATASET
    a["armA_n_train"] = len(ds_train)
    a["armA_n_val"] = len(ds_val)
    trainer_args = Namespace(**a)

    # Construct the Trainer inside a LoadReport so every load_state_dict the
    # constructor performs is recorded with its FULL missing/unexpected counts.
    # arm E found 37 of 188 tensors silently dropped on the DINOv3 constructor;
    # strict=False plus a five-name print is how that stayed invisible.
    with seo_traineval.LoadReport() as load_rep:
        trainer = archtrain.Trainer(trainer_args)
    weight_load = load_rep.summary()
    for c in weight_load:
        print(f"[armA] load_state_dict on {c.get('module')}: offered "
              f"{c.get('n_tensors_offered')}, module holds "
              f"{c.get('n_tensors_in_module')}, missing "
              f"{c.get('n_missing_keys')}, unexpected "
              f"{c.get('n_unexpected_keys')}", flush=True)
        if c.get("n_missing_keys"):
            print(f"           missing[:10]    {c.get('missing_keys_first10')}", flush=True)
        if c.get("n_unexpected_keys"):
            print(f"           unexpected[:10] {c.get('unexpected_keys_first10')}", flush=True)
    run_out = Path(trainer.output_dir)

    loss_note = set_ignore_index_loss(trainer, arch)

    # ---- the train-side evaluator, under a NAMED authority -------------------
    train_eval_state = {"disabled": False,
                        "what": "train-side evaluator left ON (arm C recipe)"}
    equiv_sha = None
    equiv_verdict = None
    equiv_path = None
    if args.disable_train_eval:
        if not args.equiv_artifact:
            raise SystemExit(
                "FATAL --disable_train_eval requires --equiv_artifact under "
                "either authority. The artifact is the evidence the decision "
                "was taken against and its sha goes into run.json.")
        ap_path = Path(args.equiv_artifact)
        if not ap_path.exists():
            raise SystemExit(f"FATAL equivalence artifact missing: {ap_path}")
        equiv = json.loads(ap_path.read_text())
        equiv_verdict = equiv.get("verdict")
        equiv_sha = hashlib.sha256(ap_path.read_bytes()).hexdigest()
        equiv_path = str(ap_path)
        if args.train_eval_authority == "equivalence_pass" and equiv_verdict != "PASS":
            raise SystemExit(
                f"FATAL authority=equivalence_pass but the stage-1 verdict is "
                f"{equiv_verdict!r} -- refusing to disable the train-side "
                f"evaluator. {equiv.get('verdict_reason', '')}")
        if args.train_eval_authority == "code_audit" and equiv_verdict == "FAIL":
            raise SystemExit(
                "FATAL authority=code_audit is for an INCONCLUSIVE proof, not a "
                "FAILED one. A FAIL verdict is positive evidence the lever "
                "changes training and no code-path argument overrides it.")
        train_eval_state = seo_traineval.disable_train_evaluator(trainer)
        train_eval_state["authority"] = args.train_eval_authority
        train_eval_state["equivalence_artifact"] = equiv_path
        train_eval_state["equivalence_sha256"] = equiv_sha
        train_eval_state["equivalence_verdict"] = equiv_verdict
        print(f"[armA] train-side evaluator DISABLED under authority "
              f"'{args.train_eval_authority}'; artifact {ap_path.name} "
              f"verdict {equiv_verdict} sha {equiv_sha[:16]}", flush=True)

    # Stage-3 diagnostic: predicted-positive fraction on val, per epoch.
    # Observation only -- see install_val_predpos_probe.
    val_predpos = []
    predpos_probe = seo_traineval.install_val_predpos_probe(trainer, val_predpos)

    step_losses = []
    step_probe = None
    if args.probe_json:
        step_probe = seo_traineval.install_step_loss_probe(trainer, step_losses)

    resume_from = newest_epoch_checkpoint(run_out)
    resumed_epoch = 0
    if resume_from:
        print(f"RESUME from {resume_from}", flush=True)
        trainer.load_checkpoint(resume_from)
        resumed_epoch = int(getattr(trainer, "start_epoch", 0))
    # First epoch this process will actually run. With the lever on, this is
    # also the first epoch that trained without the train-side evaluator, i.e.
    # the mid-cell deviation point recorded in run.json.
    first_epoch_this_link = resumed_epoch + 1
    # CELL-level, supplied by the caller. It was previously derived from THIS
    # link's resume point, which is a different quantity: the run.json that
    # survives is the last link's, so a 3-link cell recorded the epoch its
    # FINAL link resumed at (59, 62, 32) instead of the epoch the lever came
    # on (10, 10, 1). Metadata only -- no training was affected -- but it is
    # exactly the number the deviation is declared by, so it is now passed in
    # rather than inferred from a link that cannot know the cell's history.
    train_eval_off_from_epoch = (args.train_eval_off_from_epoch
                                 if train_eval_state.get("disabled") else None)
    train_eval_off_from_epoch_label = (args.train_eval_off_from_epoch_label
                                       if train_eval_state.get("disabled") else None)

    g = torch.Generator()
    g.manual_seed(args.seed)
    bs = trainer_args.batch_size
    nw = args.num_workers
    trainer.dataloaders = {
        "train": DataLoader(TrainTargets(ds_train), batch_size=bs, shuffle=True,
                            num_workers=nw, pin_memory=True, drop_last=True,
                            generator=g, worker_init_fn=dl_worker_init),
        "val": DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw,
                          pin_memory=True, worker_init_fn=dl_worker_init),
        "test": DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=nw,
                           pin_memory=True, worker_init_fn=dl_worker_init),
    }
    steps_per_epoch = len(trainer.dataloaders["train"])
    val_batches = len(trainer.dataloaders["val"])
    print(f"\nDATALOADERS  train={len(ds_train)}  val={len(ds_val)}  "
          f"test={len(ds_test)}  steps/epoch={steps_per_epoch}  "
          f"val_batches={val_batches}", flush=True)
    if not smoke:
        if len(ds_train) != seo_budget.N_TRAIN:
            raise SystemExit(f"FATAL train {len(ds_train)} != {seo_budget.N_TRAIN}")
        if len(ds_val) != seo_budget.N_VAL:
            raise SystemExit(f"FATAL val {len(ds_val)} != {seo_budget.N_VAL}")
        if len(ds_test) != man["n_test"]:
            raise SystemExit(f"FATAL test {len(ds_test)} != {man['n_test']}")

    # ---- per-epoch probe, flushed to disk after EVERY epoch -----------------
    # A probe link is EXPECTED to end on its wall clock, i.e. by SIGKILL. A
    # measurement written only at the end of training is a measurement that
    # such a link never writes, so this flushes every epoch, atomically.
    probe_ctx = {"epochs": []}
    probe_state = {"installed": False, "reason": "--probe_json not set"}

    def _rate(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return round(sum(vals) / len(vals) / 60.0, 4)

    def flush_probe(_rec=None, oom=None, note=None):
        if not args.probe_json:
            return
        eps = [e for e in probe_ctx["epochs"] if e.get("val_s") is not None]
        tail = eps[1:] if len(eps) >= 2 else eps          # drop epoch 1: cudnn warm-up
        tr_all, va_all = _rate([e["train_s"] for e in eps]), _rate([e["val_s"] for e in eps])
        tr, va = _rate([e["train_s"] for e in tail]), _rate([e["val_s"] for e in tail])
        tot = None if tr is None or va is None else round(tr + va, 4)
        payload = {
            "run_id": args.run_id, "arch": arch, "condition": mode,
            "holdout": args.holdout, "seed": args.seed,
            "link_role": ("timing probe = link 1 of a real grid cell; this cell "
                          "is NOT thrown away"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_time_limit": os.environ.get("ARMA_TIME_LIMIT"),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "batch_size": bs, "n_train": len(ds_train), "n_val": len(ds_val),
            "steps_per_epoch": steps_per_epoch, "val_batches": val_batches,
            "num_workers": args.num_workers,
            "first_epoch_label_this_link": first_epoch_this_link,
            "epochs": probe_ctx["epochs"],
            "n_epochs_measured": len(eps),
            "measured": {
                "basis": ("epochs after the first, which pays cudnn autotune "
                          "and page-cache warm-up; the all-epoch figures are "
                          "given beside them, not instead of them"),
                "train_min_per_epoch": tr,
                "val_min_per_epoch": va,
                "total_min_per_epoch": tot,
                "train_min_per_epoch_all_epochs": tr_all,
                "val_min_per_epoch_all_epochs": va_all,
                "s_per_step_train": (None if tr is None else
                                     round(tr * 60.0 / steps_per_epoch, 5)),
                "s_per_step_total": (None if tot is None else
                                     round(tot * 60.0 / steps_per_epoch, 5)),
            },
            "mamnet_baseline_min_per_epoch_eval_off": MAMNET_MIN_PER_EPOCH_EVAL_OFF,
            "mamnet_baseline_source": ("logs/armA_evalfree_equivalence.json arm b, "
                                       "same 3500 train / 900 val cell shape"),
            "multiplier_vs_mamnet": (
                None if tot is None else
                round(tot / MAMNET_MIN_PER_EPOCH_EVAL_OFF, 4)),
            "peak_vram_alloc_gb": round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
            "peak_vram_reserved_gb": round(
                torch.cuda.max_memory_reserved() / (1024 ** 3), 3),
            "vram_total_gb": (
                round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3)
                if torch.cuda.is_available() else None),
            "oom": bool(oom) if oom is not None else False,
            "oom_detail": note,
            "oom_policy": ("if this cell OOMs at arm C's batch size the DINOv3 "
                           "chain STOPS and is reported. The batch size is NOT "
                           "cut here -- that is a recipe change, and a recipe "
                           "change is the operator's decision, not this "
                           "script's."),
            "pretrained_weight_load": weight_load,
            "loss_step0": (step_losses[0] if step_losses else None),
            "loss_last": (step_losses[-1] if step_losses else None),
            "n_train_steps_seen": len(step_losses),
            "val_predpos_per_epoch": val_predpos,
            "train_eval_disabled": bool(train_eval_state.get("disabled")),
            "train_eval_authority": args.train_eval_authority,
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        write_json_atomic(args.probe_json, payload)

    if args.probe_json:
        probe_state = seo_traineval.install_epoch_probe(trainer, probe_ctx,
                                                        on_epoch=flush_probe)
        flush_probe()                    # a probe exists from before epoch 1
        print(f"[armA] epoch probe LIVE -> {args.probe_json}", flush=True)

    train_wall = 0.0
    if not args.dump_only:
        t_train = time.time()
        try:
            trainer.train()
        except Exception as e:
            oom = ("out of memory" in str(e).lower()
                   or type(e).__name__ == "OutOfMemoryError")
            flush_probe(oom=oom,
                        note=f"{type(e).__name__}: {str(e)[:400]}" if oom else None)
            if oom:
                print("!" * 78, flush=True)
                print(f"[armA] CUDA OOM at batch_size {bs} on {arch}. STOPPING. "
                      f"The batch size is NOT being cut -- that is a recipe "
                      f"change and it is the operator's call.", flush=True)
                print("!" * 78, flush=True)
            raise
        train_wall = time.time() - t_train
        flush_probe()

    best_ckpt = resolve_best_checkpoint(run_out)
    if best_ckpt is None:
        raise SystemExit(
            f"FATAL no loadable best checkpoint under {run_out} "
            f"(neither {BEST_NAME} nor {BEST_PREV_NAME})")
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    best_epoch = int(ck["epoch"])
    epochs_run = len(ck.get("train_losses", [])) or len(trainer.train_losses)
    best_val_metric = float(ck.get("best_decision_miou", float("nan")))
    del ck
    epochs_run_total = len(getattr(trainer, "train_losses", []) or []) or epochs_run
    steps_to_best = best_epoch * steps_per_epoch
    val_hist = list((getattr(trainer, "val_metrics_history", {}) or {}).get("mIOU", []))
    ewi = int(getattr(trainer, "epochs_without_improvement", -1))
    stopped_on = ("patience" if ewi >= PATIENCE_EPOCHS else
                  "epoch_ceiling" if epochs_run_total >= EPOCHS_CEIL - 1 else
                  "link_boundary_or_incomplete")
    print(f"\nBEST epoch {best_epoch}  epochs_run {epochs_run_total}  "
          f"steps_to_best {steps_to_best}  epochs_without_improvement {ewi}  "
          f"stopped_on {stopped_on}  "
          f"best_val_decision_mIoU {best_val_metric:.4f}", flush=True)
    # ---- inference on the held-out city's FULL test split, then the dump ----
    from utils.postprocessing import filter_small_predictions

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference")
        trainer.load_checkpoint(str(best_ckpt))
        model = trainer.model
        model.eval()
        device = trainer.device

        meta_by_pid = {}
        for r in rows_te:
            pid = f"{r['image_id']}_y{r['y']}_x{r['x']}"
            meta_by_pid[pid] = {
                "image_id": pid,
                "aoi_id": r["aoi_id"],
                "source_image_id": r["image_id"],
                "patch_index": pidx[pid],
                "patch_y": r["y"], "patch_x": r["x"],
                "city": r["city"], "split": "test",
                "acquisition_date": r["acq"],
                "sun_elevation": r["sun_el"], "sun_azimuth": r["sun_az"],
                "lon": r["lon"], "lat": r["lat"],
                "sun_dist": abs(float(r["sun_el"]) - train_sun_mean),
            }

        rows, probs = [], {}
        t_inf = time.time()
        with torch.no_grad():
            for batch in trainer.dataloaders["test"]:
                images = batch["image"].to(device)
                pid = batch["filename"][0]
                logits = model(images)
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
                filtered = filter_small_predictions(logits, min_pixels=10)
                pred = torch.argmax(filtered, dim=1)[0].cpu().numpy().astype(np.uint8)
                gt = batch["mask"][0].numpy().astype(np.uint8)
                seen = batch["valid"][0].numpy().astype(bool)
                rows.append(seo_dump.perimage_row(
                    args.run_id, meta_by_pid[pid], gt, prob, pred, seen))
                probs[pid] = prob.astype(np.float16)
        inf_wall = time.time() - t_inf

        tbx.phase("metrics")
        import pandas as pd                                      # noqa: E402
        df = pd.DataFrame(rows)
        fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
        miou_tol = float(df["miou_tol_pct"].mean())
        pred_pos_frac = float(df["n_pred_pos"].sum() / max(int(df["n_valid_px"].sum()), 1))
        mean_unseen = float(df["unseen_frac"].mean())
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        wall = time.time() - t0

        stray_png = sorted(str(q.relative_to(run_out)) for q in run_out.rglob("*.png"))
        stray_tb = sorted(str(q.relative_to(run_out))
                          for q in run_out.rglob("events.out.tfevents*"))
        stray_epoch_ckpt = sorted(q.name for q in run_out.glob("checkpoint_epoch_*.pth"))

        # The 16 hex characters the operator put on the record for this arm.
        # The full sha is recomputed from the artifact bytes above; the worker
        # checks the prefix BEFORE training so a swapped artifact costs seconds.
        equiv_sha_on_record = "4d030f64f444d886"
        justification = {
            "code_audit": ("code-path audit; empirical proof unexecutable, "
                           "harness nondeterministic at torch defaults"),
            "equivalence_pass": ("stage-1 bit-identical equivalence proof "
                                 "returned PASS"),
        }.get(args.train_eval_authority)

        run = {
            "run_id": args.run_id, "dataset": DATASET, "arm": "A",
            "arm_note": ("PHASE 1 of the REDUCED grid. Not a grid, not a result. "
                         "This box states no gap and computes no bootstrap. "
                         "OGLANet is dropped from arm A."),
            "arch": arch, "method": method, "mode": mode, "rung": RUNG,
            "arch_input": {"in_channels": spec["in_channels"],
                           "use_contrast": spec["use_contrast"],
                           "note": ("follows the architecture's published input "
                                    "contract: MAMNet's 144 published cells set "
                                    "use_contrast=true and consume RGBC; DINOv3 "
                                    "consumes plain RGB, as in arm C's ladder.")},
            "rung_note": ("rung here is the patch budget, 3500. CONTRACT.md 1's "
                          "rung enum (225|450|900) is the ShadowTransfer ladder; "
                          "arm A extends it because the unit is a 384 px patch, "
                          "not a benchmark image."),
            "holdout": args.holdout, "source_cities": man["source_cities"],
            "res": RES, "seed": args.seed,
            "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(df),
            "n_test_expected": man["n_test"],
            "n_train_aoi": len(aoi_tr), "n_val_aoi": len(aoi_va), "n_test_aoi": len(aoi_te),
            "n_train_per_source_city": {c: sum(1 for p in man["train"]
                                               if p.split("_", 1)[0] == c)
                                        for c in sorted({p.split("_", 1)[0]
                                                         for p in man["train"]})},
            "seed_sites": seed_sites,
            "leakage_check": leak,

            "target_access": "source_only" if mode == "loco" else "target_labels",
            "target_access_detail": (
                "train and val draw only from the two source cities' AOIs; no "
                "held-out-city pixel is touched before test-time inference"
                if mode == "loco" else
                "within-city upper bound: train and val are labelled patches from "
                "the held-out city's own AOIs, disjoint from its test AOIs"),

            # ---- the metric, declared before the run ----
            "reporting_metric": "2class_tol_k2_unseen_excluded",
            "reporting_metric_value_pct": round(miou_tol, 4),
            "reporting_metric_note": (
                "arm A's declared headline: two-class boundary-tolerant mIoU at "
                "K=2 with DSM-unseen pixels excluded from numerator and "
                "denominator. Column miou_tol_pct."),
            "sanity_floor_metric": "fg_only_tol_k2_unseen_excluded",
            "sanity_floor_metric_value_pct": round(fg_iou_tol, 4),
            "sanity_floor_metric_note": (
                "foreground-only IoU at K=2, column iou_tol. The quantity the "
                "sanity floor is set on. This box does not set the floor."),
            "decision_metric": "2class_tol_k2_unseen_INCLUDED",
            "decision_metric_value_pct": round(best_val_metric, 4),
            "decision_metric_note": (
                "what checkpoint_best.pth was SELECTED on: the published "
                "trainer's own per-image DetailedEvaluator 2-class tolerant "
                "mIoU at K=2, on validation. It is NOT the reporting metric and "
                "does NOT exclude unseen pixels -- see unseen_handling."),
            "band_width_headline": 2,
            "band_widths_written": [0, 2, 5],

            "unseen_handling": {
                "metric": ("EXCLUDED from numerator and denominator, at K=0, 2 "
                           "and 5, via a prediction-side sentinel that drops the "
                           "pixel from all four confusion cells of the lifted "
                           "DetailedEvaluator. metrics.py is not edited."),
                "loss": ("EXCLUDED: CrossEntropyLoss(ignore_index=255) on the "
                         "loader's mask_ignore target, train split only. "
                         + loss_note["installed_at"]),
                "val_decision_metric": (
                    "NOT excluded. The published DetailedEvaluator builds its "
                    "+-K don't-care band by eroding and dilating the target, so a "
                    "255 sentinel there would corrupt the band's morphology, and "
                    "patching the published evaluator is a contract change. Both "
                    "cells of the fold use the identical rule."),
                "mean_unseen_frac_on_test": round(mean_unseen, 6),
                "rule_source": "logs/armA_seo_census_acq3.json valid_region_rule",
            },

            "budget_rule": man["budget_rule"],
            "budget_rule_sha": man["budget_rule_sha"],
            "split_rule_sha": man["split_rule_sha"],
            "budget_manifest": str(man_path),
            "budget_boundary_aois": {k: {"partial_aoi_id": d["partial_aoi_id"],
                                         "partial_aoi_n": d["partial_aoi_n"],
                                         "partial_aoi_total": d["partial_aoi_total"],
                                         "n_aoi_used": d["n_aoi_used"],
                                         "quota_fraction_of_pool":
                                             d["quota_fraction_of_pool"]}
                                     for k, d in man["per_quota_detail"].items()},
            "test_note": man["test_note"],

            "train_sun_elev_mean": round(train_sun_mean, 4),
            "train_sun_elev_sd": round(train_sun_sd, 4),
            "sun_dist_definition": ("|patch sun_elevation - train_sun_elev_mean| "
                                    "of THIS cell; it differs between the loco "
                                    "and upper cells on the same test patch"),

            "epochs_max": trainer_args.epochs, "epochs_run": epochs_run_total,
            "epochs_run_at_best_ckpt": epochs_run,
            "best_epoch": best_epoch, "steps_per_epoch": steps_per_epoch,
            "steps_to_best": steps_to_best,
            "epochs_without_improvement_at_stop": ewi,
            "stopped_on": stopped_on,
            "early_stopped": bool(ewi >= PATIENCE_EPOCHS),
            "early_stop_patience": PATIENCE_EPOCHS,
            "early_stop_rule": (f"patience {PATIENCE_EPOCHS} epochs flat, epochs "
                                f"capped at {EPOCHS_CEIL}, best checkpoint retained "
                                f"- arm C's ladder recipe, unchanged"),
            "epochs_ceiling_recipe": EPOCHS_MAX,
            "epochs_ceiling_used": EPOCHS_CEIL,
            "run_id_suffix": args.run_id_suffix or None,
            "retrain_script_sha": (sha256_file(Path(__file__).resolve())
                                   if args.run_id_suffix else None),
            "retrain_note": ((
                "FROM-SCRATCH RETRAIN under D-A2-1, not a continuation and "
                "not a patch. The cell trains from initialisation with "
                "ONE horizon, so --epochs_max and the LR schedule "
                "denominator are the same number and no schedule is "
                "reshaped. The ORIGINAL cell's dump is preserved untouched "
                "at its own run_id; this cell writes to a NEW run_id and "
                "never opens the original for writing. retrain_script_sha "
                "is this file AFTER the additive --run_id_suffix / "
                "--epochs_max flags were added; the original cell recorded "
                "the sha before them.") if args.run_id_suffix else None),
            "early_stop_patience_published": cfg.get("early_stopping_patience"),
            "val_decision_metric_history_pooled_miou": val_hist,
            "resume_epoch_label_offset_note": (
                "PUBLISHED off-by-one, recorded not fixed: train() iterates "
                "`range(start_epoch, epochs)` and labels the epoch `epoch+1`, "
                "while load_checkpoint sets `start_epoch = checkpoint['epoch'] "
                "+ 1`. A resume therefore skips exactly one epoch LABEL at each "
                "link boundary (resuming from the checkpoint labelled 9 makes "
                "the next trained epoch label 11, not 10). epochs_run counts "
                "actual epochs and is the number to use; the label ceiling of "
                "100 is reached one epoch short per boundary. Fixing this "
                "would change a published trainer, which is a contract change."),
            "best_val_metric": best_val_metric,
            "lr": trainer_args.lr, "batch_size": trainer_args.batch_size,
            "num_workers": args.num_workers,
            "num_workers_note": ("published args.json records num_workers=1; "
                                 "raised to 8 because each S-EO sample is a 384 px "
                                 "crop out of a ~1500 px PNG plus, for MAMNet, a "
                                 "7x7 RMS contrast pass. Measured dataloader "
                                 "wait 1.4%."),

            "dataset_detail": {
                "name": "S-EO (Masquil et al., CVPRW 2025, arXiv 2504.06920)",
                "unit": "384 px patch at native ~0.3 m GSD",
                "resize_applied": False, "patch_px": 384, "gsd_m": 0.3,
                "shadow_value": 0,
                "shadow_value_note": ("the released _shadow.png stores 1=LIT, "
                                      "0=SHADOW; measured by brightness, not "
                                      "assumed. armA_seo_census re-verifies every "
                                      "prep run."),
                "acq_per_aoi": 3, "bootstrap_unit": "aoi_id",
                "labels": ("simulated from an aligned LiDAR DSM plus solar "
                           "position through the bundle-adjusted RPC camera; not "
                           "human annotations"),
                "radiometry": "uint8, colour-corrected and histogram-equalized at source",
                "category_maps": "empty (CONTRACT.md 3: external datasets)",
            },

            "metric_module_sha": sha256_file(BASE / "shadowtransfer/metrics.py"),
            "dump_module_sha": sha256_file(BASE / "shadowtransfer/seo_dump.py"),
            "budget_module_sha": sha256_file(BASE / "shadowtransfer/seo_budget.py"),
            "loader_module_sha": sha256_file(BASE / "shadowtransfer/seo_dataset.py"),
            "traineval_module_sha": sha256_file(BASE / "shadowtransfer/seo_traineval.py"),
            "runtime_module_sha": sha256_file(BASE / "shadowtransfer/armc_runtime.py"),
            "train_script_sha": sha256_file(Path(__file__).resolve()),
            "git_hash": os.environ.get("ARMA_GIT_HASH"),
            "git_hash_reason": (None if os.environ.get("ARMA_GIT_HASH")
                                else "ARMA_GIT_HASH not exported by the worker"),
            "git_tree_dirty_paths": os.environ.get("ARMA_GIT_DIRTY"),
            "git_hash_scope": (
                "HEAD of BASE_PATH at submit time. The submit script hard-fails "
                "if any of this arm's own source files is uncommitted; the count "
                "above is every OTHER dirty path in the tree (standing untracked "
                "bytecode and scratch output), which this run did not execute."),
            "config_sha256": sha256_file(recipe_config),
            "config_verbatim": a,
            "recipe_config_path": str(recipe_config.relative_to(BASE)),
            "recipe_config_note": (
                "one published config per ARCH for BOTH cells of a fold so they "
                "differ only in the data. The published loco and upper configs "
                "were diffed 2026-08-01 and are identical on every training "
                "knob; they differ only in data-routing keys, which this script "
                "overrides because it replaces the dataloaders wholesale."),
            "config_defaults_filled": defaults_filled,
            "config_rerouted_keys": rerouted,
            "config_rerouted_note": (
                "data-routing keys repointed at this cluster. DINOv3's published "
                "args.json was written on Anvil and carries /anvil/... paths, "
                "which do not exist here; the Trainer builds its own dataloaders "
                "in __init__ from these before this script replaces them, so "
                "they must resolve even though nothing trains on them."),
            "pretrained_weight_load": weight_load,
            "pretrained_weight_load_note": (
                "every load_state_dict performed while the Trainer was "
                "constructed, with FULL missing/unexpected counts. The published "
                "DINOv3 backbone loads with strict=False and prints only five "
                "names of each, which is how a partial load stays invisible."),

            "cluster": "delta", "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu": gpu_name, "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_time_limit": os.environ.get("ARMA_TIME_LIMIT"),
            "wall_clock_s": round(wall, 2),
            "train_wall_clock_s": round(train_wall, 2),
            "inference_wall_clock_s": round(inf_wall, 2),
            "peak_vram_alloc_gb": round(peak_alloc, 3),
            "peak_vram_reserved_gb": round(peak_reserved, 3),
            "epoch_timing_this_link": probe_ctx["epochs"],
            "probe_json": args.probe_json,
            "probe_state": probe_state,
            "compute_note": os.environ.get(
                "ST_COMPUTE_NOTE",
                "trained on gpuA100x4; no CPU association exists on this "
                "allocation (a GPU account only)"),

            "ckpt_path": str(best_ckpt), "parent_run_id": None,
            "legacy_run_dir": None, "structure_cols": "deferred",

            # ---- the declared, mid-cell deviation --------------------------
            "train_eval_disabled": bool(train_eval_state.get("disabled")),
            "train_eval_authority": args.train_eval_authority,
            "train_eval_justification": justification,
            "train_eval_equivalence_sha": (equiv_sha_on_record
                                           if train_eval_state.get("disabled")
                                           else None),
            "train_eval_equivalence_sha_full": equiv_sha,
            "train_eval_equivalence_sha_note": (
                "train_eval_equivalence_sha is the 16-character value the "
                "operator put on the record for this arm; "
                "train_eval_equivalence_sha_full is the sha256 recomputed here "
                "from the artifact's own bytes. The worker checks the prefix "
                "before training starts."),
            "train_eval_equivalence_verdict": equiv_verdict,
            "train_eval_equivalence_artifact": equiv_path,
            "train_eval_disabled_from_epoch": train_eval_off_from_epoch,
            "train_eval_disabled_from_epoch_label": train_eval_off_from_epoch_label,
            "train_eval_link_first_epoch_label": first_epoch_this_link,
            "train_eval_disabled_from_epoch_note": (
                "MID-CELL DEVIATION. train_eval_disabled_from_epoch counts "
                "EPOCHS OF TRAINING: the two MAMNet cells trained epochs 1-9 "
                "with the train-side evaluator ON and every epoch from their "
                "10th onward with it OFF. The *_label field is what the "
                "published trainer prints for that same epoch, which differs "
                "by one at a resume boundary; see "
                "resume_epoch_label_offset_note. Both are CELL-level constants "
                "supplied by the submit script. train_eval_link_first_epoch_label "
                "is the different, per-link quantity: the epoch THIS link "
                "resumed at."),
            "train_eval_state": train_eval_state,
            "train_eval_deviation_note": (
                "DECLARED DEVIATION from arm C's ladder recipe. Arm C runs the "
                "published Trainer.train_epoch with its per-image "
                "DetailedEvaluator active on every training batch; this arm "
                "switches that one evaluator off. It is observational -- the "
                "train-side result is a local that train_epoch never returns, "
                "reaching two prints and a NullWriter. The bit-identical "
                "equivalence proof returned INCONCLUSIVE, not FAIL: the two "
                "CONTROL runs, identical in every respect, already diverge at "
                "step 1 under torch defaults, so the proof as specified is "
                "unexecutable in the shipping configuration. The lever is taken "
                "on the code-path audit instead, by operator decision "
                "2026-08-02. The VALIDATION-side evaluator, which computes the "
                "boundary-tolerant decision metric that selects "
                "checkpoint_best.pth and drives patience-20, is UNTOUCHED."),
            "val_predpos_per_epoch": val_predpos,
            "val_predpos_probe": predpos_probe,
            "step_loss_probe": step_probe,
            "worker_side_effects_disabled": disabled,
            "collapse_guard": dict(armc_runtime.COLLAPSE_GUARD_STATE),
            "checkpoint_pruner": pruner,
            "post_training_timeout_s": args.timeout_s,
            "side_effect_audit": {"png_files": stray_png,
                                  "tensorboard_event_files": stray_tb,
                                  "periodic_checkpoints": stray_epoch_ckpt,
                                  "save_freq_effective": a.get("save_freq")},
            "prediction_recipe": ("argmax over the two class logits, then "
                                  "filter_small_predictions(min_pixels=10); "
                                  "identical to the published test() loop"),
            "probs_note": "softmax(logits)[:, 1], stored float16, unthresholded",
            "aurc_definition": "aurc_trapz19",
            "pred_pos_frac_on_valid": round(pred_pos_frac, 6),
            "smoke": smoke, "forced_rerun": bool(args.force),
            "dump_only": bool(args.dump_only),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        # ---- self-checks the job runs on itself (execution facts, not analysis)
        flags = []
        if len(df) != len(ds_test):
            flags.append(f"dump rows {len(df)} != n_test {len(ds_test)}")
        if not smoke and len(df) != man["n_test"]:
            flags.append(f"dump rows {len(df)} != manifest n_test {man['n_test']}")
        if not (0.001 < pred_pos_frac < 0.9):
            flags.append(f"predicted-positive fraction {pred_pos_frac:.5f} outside "
                         f"(0.001, 0.9)")
        if not smoke and epochs_run_total >= EPOCHS_CEIL - 1:
            flags.append(f"ran to the epoch ceiling ({EPOCHS_CEIL}); early stopping "
                         f"never fired")
        if epochs_run_total < 10 and not smoke:
            flags.append(f"epochs_run={epochs_run_total} (<10)")
        if float(df["iou_tol"].std()) < 1e-6:
            flags.append("per-image iou_tol has ~zero variance")
        if int(df["empty_fg_valid"].sum()):
            flags.append(f"empty_fg_valid on {int(df['empty_fg_valid'].sum())} patches")
        if int(df["all_shadow_flag"].sum()):
            flags.append(f"all_shadow_flag on {int(df['all_shadow_flag'].sum())} patches")
        if stray_png:
            flags.append(f"figures written despite the rebind: {stray_png}")
        if stray_tb:
            flags.append(f"tensorboard events written despite the rebind: {stray_tb}")
        for c in ("aoi_id", "sun_elevation"):
            if df[c].isna().any():
                flags.append(f"{c} is null on {int(df[c].isna().sum())} rows")
        for c in ("aoi_id", "sun_elevation"):
            if df[c].isna().any():
                raise SystemExit(f"FATAL {c} is null on "
                                 f"{int(df[c].isna().sum())} dump rows")
        run["sanity_flags"] = flags

        tbx.phase("write_dump")
        seo_dump.write_dump(dump_dir, rows, probs, run)

    # Best-only on completion (CLAUDE.md: a long run may checkpoint for resume,
    # but deletes all but the best once it is done).
    removed = []
    # ---- D-A4-1 CARVE-OUT (ARM A SESSION A4, 2026-09-03) -------------------
    # D-A4-1 "Recurrence prevention" ruled that `checkpoint_latest.pth`
    # SURVIVES on any cell whose `stopped_on` is `epoch_ceiling`, and on any
    # cell run with `--keep_latest`; a cell that stopped on patience has a
    # resolved ceiling status and still deletes it. That ruling was recorded in
    # DECISIONS_armA_wave2.md but NEVER LANDED IN CODE -- every prune site
    # listed "checkpoint_latest.pth" unconditionally, and the dinov3 BSM
    # stage-2 log of 2026-09-03 shows it in `periodic ckpts removed`. This is
    # the carve-out, implemented as ruled. It does not widen deletion: it only
    # withholds it.
    _keep_latest = bool(getattr(args, "keep_latest", False)) or (
        stopped_on == "epoch_ceiling")
    _pats = ["checkpoint_epoch_*.pth", "checkpoint_epoch_*.pth.corrupt",
             BEST_PREV_NAME]
    if not _keep_latest:
        _pats.insert(2, "checkpoint_latest.pth")
    else:
        print(f"[D-A4-1] checkpoint_latest.pth RETAINED "
              f"(stopped_on={stopped_on}, keep_latest="
              f"{bool(getattr(args, 'keep_latest', False))})", flush=True)
    for pat in _pats:
        for q in run_out.glob(pat):
            q.unlink()
            removed.append(q.name)

    print("\n" + "=" * 78)
    print(f"[{args.run_id}]  arch {arch}")
    print(f"  2-class mIoU K=2 (headline)  {miou_tol:.4f}")
    print(f"  fg-only IoU K=2              {fg_iou_tol:.4f}")
    print(f"  fg-only IoU K=0 / K=5        {float(df['iou_str'].mean())*100:.4f} / "
          f"{float(df['iou_k5'].mean())*100:.4f}")
    print(f"  val decision metric          {best_val_metric:.4f} (unseen INCLUDED)")
    print(f"  patches                      {len(df)}   AOIs {len(aoi_te)}")
    print(f"  mean unseen_frac             {mean_unseen:.5f}")
    print(f"  predicted-positive fraction  {pred_pos_frac:.5f}")
    print(f"  epochs_run {epochs_run_total}  best_epoch {best_epoch}  "
          f"steps_to_best {steps_to_best}  stopped_on {stopped_on}  "
          f"epochs_without_improvement {ewi}")
    print(f"  train-eval disabled from     epoch {train_eval_off_from_epoch} "
          f"(printed label {train_eval_off_from_epoch_label}); this link "
          f"resumed at label {first_epoch_this_link}")
    print(f"  wall clock                   {wall:.1f}s "
          f"(train {train_wall:.1f}s, inference {inf_wall:.1f}s)")
    print(f"  peak VRAM alloc/resv         {peak_alloc:.2f} / {peak_reserved:.2f} GiB")
    print(f"  periodic ckpts removed       {removed or 'none'}")
    print(f"  sanity flags                 {flags or 'none'}")
    print(f"  -> {dump_dir}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
