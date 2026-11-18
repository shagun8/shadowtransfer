"""
Arm E Part 2 — Experiment A (decoder retraining) on the frozen-backbone
DINOv3 LOCO models. Runtime: per-cell args/naming and the 25%-subset train
loader.

Reuses `arme_part1_runtime.patch_trainer_class` UNCHANGED for the freeze
assertions (construction + after-first-optimizer-step, trainable count and
grad-is-None) and the decoder-only checkpoint save/load -- Part 2's freeze
requirement is identical to Part 1's ("same verified full-freeze switch").

WHY NO SEPARATE "REINITIALISE THE DECODER" STEP EXISTS HERE: a freshly
constructed `DINOv3ShadowDetector` already builds `self.decoder =
DINOv3Decoder(...)` from scratch every time (PyTorch's default Conv2d/
ConvTranspose2d/Linear init), and Part 2 never loads Part 1's checkpoint --
it only reads Part 1's per-image dumps, at aggregation time, for the
recovery-ratio join. So "freeze the encoder, reinitialize the decoder" is
exactly what a bare `Trainer(args)` construction with `freeze_backbone=True`
already gives, with no extra code. `common.seed_all(0)` (called before
construction, same as Part 1) makes that fresh init reproducible.

SUBSET SELECTION, verbatim from the published Experiment A implementation.
App G.9's prose says "a randomly sampled 25%... (~110 images)". The actual
code, `final_loco/experiment_a_decoder_retrain.py:164-167`, is NOT
random at all:

    n_use   = max(1, int(n_total * args.train_fraction))
    indices = np.linspace(0, n_total - 1, n_use, dtype=int).tolist()
    train_subset = Subset(full_dataset, indices)

a deterministic, evenly-spaced index list over the sorted file listing, with
no RNG call anywhere -- there is no seed to recover because none is used.
`rebuild_expa_train_loader` below reproduces this exact formula (same
variable names, same Subset/DataLoader call shape, drop_last=True matching
that script's DataLoader), over dinov3's OWN `data.dataset.Dataset` class
(not the published script's mamnet-borrowed one, used there for cross-arch
convenience) so every image_id lines up with arm E Part 1's LOCO/upper dumps
for the recovery-ratio R join -- same directory, same sorted-filename
ordering, same class.
"""

import sys
from pathlib import Path

import numpy as np
import torch

CITIES = ("chicago", "miami", "phoenix")
ARCH = "dinov3frozen"
RUNG = 450
TRAIN_FRACTION = 0.25

# CONTRACT.md's mode enum ({loco, upper}) has no slot for a third,
# limited-target-data condition. method="expa"/mode="expa" is a deliberate,
# clearly-flagged departure (see session log) rather than a forced fit into
# either existing value.
METHOD = "expa"
MODE = "expa"


def run_id(city, seed=0, suffix=""):
    return f"{ARCH}-{METHOD}-{MODE}-r{RUNG}-{city}-hr-s{seed}" + suffix


def rebuild_expa_train_loader(base, city, batch_size=8, num_workers=1,
                              train_fraction=TRAIN_FRACTION):
    """~110-image (25%) train loader for the holdout city, replacing
    Trainer's own full-450-image single-city loader. Returns
    (loader, n_total, n_use, indices) -- the counts and indices are recorded
    into run.json so the exact subset is reconstructible without re-running.
    """
    if str(Path(base) / "python" / "dinov3") not in sys.path:
        sys.path.insert(0, str(Path(base) / "python" / "dinov3"))
    from data.dataset import Dataset

    train_dir = str(Path(base) / "data" / "Final_data_test" / city / "highres")
    full_dataset = Dataset(train_dir, split="train", img_size=384, augment=True)
    n_total = len(full_dataset)
    n_use = max(1, int(n_total * train_fraction))
    indices = np.linspace(0, n_total - 1, n_use, dtype=int).tolist()
    subset = torch.utils.data.Subset(full_dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader, n_total, n_use, indices
