#!/usr/bin/env python3
"""Arm A / S-EO WAVE 0 -- the CrossEarth adapter.

CrossEarth is a first-class arm A architecture column, but it does not fit the
`armA_pilot_train.py` code path at all: it has no `Trainer` class, no epochs,
no `dataloaders` dict and no per-image `DetailedEvaluator`. It is mmseg,
iteration-based, with a frozen DINOv2 trunk plus Rein adapters and a
Mask2Former head, driven by arm F's `armf_crossearth_engine.train`.

This module is the ONLY new code between the two. It holds two dataset
wrappers and a collate function; every metric, every model construction and the
training loop itself come from the existing arm F modules unchanged, and every
patch comes from `seo_dataset.SEODataset` unchanged.

Why the images are recovered rather than re-read
------------------------------------------------
CrossEarth's `SegDataPreProcessor` does its own normalisation from
`PIXEL_MEAN`/`PIXEL_STD`, so it needs the raw uint8 crop; `SEODataset` returns
an ImageNet-normalised float tensor. Three routes were available:

  1. add a `raw` return mode to `seo_dataset.py`. Rejected: that changes
     `loader_module_sha`, which is recorded in the twelve finished seed-0 cells
     and re-checked against them. Phase 3 rejected the same edit for the same
     reason when it needed `intensity_map`.
  2. decode the PNG crop a second time inside this adapter. Rejected: it
     doubles the loader's I/O for one architecture only.
  3. invert the normalisation on the tensor the loader already built.

Route 3 is used, and it is exact rather than approximate: the forward map is
`t = (x/255 - mean)/std` in float32 on integers 0..255, so the inverse lands
within ~1e-4 of an integer and `np.rint` recovers `x` exactly. That is not
assumed -- `recover_uint8` asserts the residual on every call, and the run
fails loudly rather than training on a drifted image. Recorded in run.json as
`image_recovery`.

Why the LOSS sees 255 but Mask2Former still cannot ignore it
-------------------------------------------------------------
Arm A's committed rule is that DSM-unseen pixels leave the loss entirely; for
MAMNet, DINOv3 and OGLANet that is one `CrossEntropyLoss(ignore_index=255)`.
Mask2Former has no per-pixel ignore. `mmseg.models.decode_heads.
mask2former_head._seg_data_to_instance_data` does
`gt_labels = classes[classes != self.ignore_index]` and then
`masks.append(gt_sem_seg == class_id)`, so a 255 pixel forms NO instance of its
own -- but it is also `False` in every other class's target mask, which the
mask loss reads as a negative rather than as absent.

So passing `mask_ignore` gives, exactly:
  * 255 pixels are excluded from the classification target (no query is asked
    to predict them), and
  * 255 pixels are trained as "not shadow and not background" by the mask
    branch, which is NOT the same as being dropped from the loss.

This is the one place arm A's unseen rule does not map onto an architecture,
and it is reported rather than papered over. It is still strictly closer to the
rule than passing the plain {0,1} target would be, which would assert those
pixels ARE background. Both cells of every fold use the identical rule, so it
does not act differently on the two sides of the comparison -- the same
argument the arm already makes for the val decision metric, which no
architecture excludes unseen pixels from.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .seo_dataset import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD

#: The residual the inverse normalisation is allowed to leave before a pixel
#: value is no longer trusted to round back to its original integer.
RECOVERY_TOL = 1e-2

_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float64).reshape(3, 1, 1)
_STD = np.asarray(IMAGENET_STD, dtype=np.float64).reshape(3, 1, 1)

UNSEEN_IN_LOSS = (
    "Mask2Former has no per-pixel ignore. gt_sem_seg carries 255 outside the "
    "valid region, so unseen pixels form no instance and are excluded from the "
    "classification target, but the mask branch reads them as negatives for "
    "every class rather than dropping them. This is the one architecture where "
    "arm A's unseen-from-loss rule does not map exactly; see "
    "seo_crossearth.py. Identical on both cells of every fold."
)

IMAGE_RECOVERY = (
    "raw uint8 recovered by inverting seo_dataset's ImageNet normalisation "
    "(x = rint((t*std + mean)*255)), asserted integral to "
    f"{RECOVERY_TOL} on every item. seo_dataset.py is not edited, so its "
    "loader_module_sha still matches the twelve finished seed-0 cells."
)


def recover_uint8(t):
    """Normalised float CHW tensor -> the uint8 HWC crop it was built from."""
    a = t.detach().cpu().numpy().astype(np.float64)
    if a.shape[0] != 3:
        raise ValueError(
            f"expected a 3-channel RGB tensor, got shape {tuple(a.shape)}. "
            "CrossEarth consumes plain RGB; build the SEODataset with "
            "use_contrast=False.")
    x = (a * _STD + _MEAN) * 255.0
    r = np.rint(x)
    resid = float(np.abs(x - r).max())
    if resid > RECOVERY_TOL:
        raise AssertionError(
            f"inverse ImageNet normalisation left a residual of {resid:.6f}, "
            f"above {RECOVERY_TOL}: the recovered image is not the uint8 crop "
            f"seo_dataset read. Refusing to train on a drifted image.")
    return np.ascontiguousarray(
        np.clip(r, 0, 255).astype(np.uint8).transpose(1, 2, 0)), resid


class SEOCrossEarthTrain(Dataset):
    """S-EO train patches through CrossEarth's released augmentation pipeline.

    `base` is whatever `armA_pilot_train.build_split` assembled -- a
    `PatchSubset` or a `ConcatDataset` of them -- built with
    `use_contrast=False` and `augment=False`. The augmentation is CrossEarth's
    own (arm F's 512->384 port of the released RandomChoiceResize / RandomCrop
    / RandomFlip / PhotoMetricDistortion), NOT seo_dataset's flips, so this
    architecture trains under its own published policy exactly as the other
    three train under theirs.
    """

    def __init__(self, base, pipeline):
        self.base = base
        self.pipeline = pipeline
        self.max_residual = 0.0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        d = self.base[i]
        img, resid = recover_uint8(d["image"])
        self.max_residual = max(self.max_residual, resid)
        gt = d["mask_ignore"].numpy().astype(np.uint8)
        res = self.pipeline({
            "img": img, "gt_seg_map": gt,
            "img_shape": img.shape[:2], "ori_shape": img.shape[:2],
            "seg_fields": ["gt_seg_map"],
        })
        return {
            "img": np.ascontiguousarray(res["img"].transpose(2, 0, 1)),
            "gt": res["gt_seg_map"].astype(np.int64),
            "valid": (res["gt_seg_map"] != IGNORE_INDEX),
            "image_id": d["filename"],
            "img_path": d["filename"],
        }


class SEOCrossEarthEval(Dataset):
    """S-EO val / test patches, untouched. No resize, no crop, no distortion.

    `gt` is the PLAIN {0, 1} target, not `mask_ignore`: the validation decision
    metric that selects the checkpoint does not exclude unseen pixels on any
    arm A architecture, and matching that here is what keeps the four
    architecture columns' model selection comparable. The valid mask travels
    beside it so the DUMP metric -- which does exclude them, per
    CONTRACT.md -- can be computed from the same forward pass.
    """

    def __init__(self, base, limit=None):
        self.base = base
        self.n = len(base) if limit is None else min(limit, len(base))
        self.max_residual = 0.0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        d = self.base[i]
        img, resid = recover_uint8(d["image"])
        self.max_residual = max(self.max_residual, resid)
        return {
            "img": np.ascontiguousarray(img.transpose(2, 0, 1)),
            "gt": d["mask"].numpy().astype(np.int64),
            "valid": d["valid"].numpy().astype(bool),
            "image_id": d["filename"],
            "img_path": d["filename"],
        }


def collate(batch):
    """Arm F's collate plus the valid mask the CONTRACT dump needs."""
    return {
        "img": [torch.from_numpy(b["img"]) for b in batch],
        "gt": [torch.from_numpy(b["gt"]) for b in batch],
        "valid": [b["valid"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "img_path": [b["img_path"] for b in batch],
    }


def predict_with_valid(model, dataset, batch_size=4, num_workers=4,
                       amp_dtype=None):
    """`armf_crossearth_engine.predict_probs`, carrying `valid` through.

    The engine's own version drops everything except ids, gts and probs, and
    the CONTRACT dump needs the valid mask aligned to the same rows. The
    forward pass, the autocast dtype and the probability definition are the
    engine's, imported rather than restated.
    """
    from torch.utils.data import DataLoader
    from .armf_crossearth_runtime import foreground_prob, make_data

    if amp_dtype is None:
        amp_dtype = torch.bfloat16
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate)
    ids, gts, probs, valids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            data = make_data(batch, model, training=False)
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model.predict(data["inputs"], data["data_samples"])
            for k, sample in enumerate(out):
                probs.append(foreground_prob(sample.seg_logits.data))
                gts.append(batch["gt"][k].numpy().astype(np.uint8))
                valids.append(np.asarray(batch["valid"][k], dtype=bool))
                ids.append(batch["image_id"][k])
    return ids, gts, probs, valids
