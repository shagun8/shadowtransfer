#!/usr/bin/env python3
"""
S-EO dataset loader for arm A (external LOCO). Data preparation only.

Returns the same keys as `python/mamnet/data/dataset.py:ShadowDataset`
('image', 'mask', 'filename') under the same conventions -- ImageNet-normalised
float image [3,384,384] (or [4,...] with the RMS-contrast channel, matching the
published MAMNet recipe), integer mask [384,384] in {0,1} -- so an existing
trainer consumes it unchanged. That file is not modified and not imported.
Adds 'image_id', 'aoi_id', 'city', 'patch_yx', 'valid', 'mask_ignore',
'n_valid_px', 'n_gt_fg_valid', 'empty_fg_valid' for the CONTRACT.md 3 schema.

`python/oglanet/data/init.py` exports a name `AISDataset`; no class of that name
exists anywhere in the tree, so that import is dead and this is modelled on
ShadowDataset instead.

Unit of iteration is a PATCH, not an image
--------------------------------------------------------------------------
S-EO crops are ~1500x1500 at ~0.3 m GSD; the published trainers take 384x384.
Resizing 1500 -> 384 would imply ~1.2 m GSD and destroy the resolution match
against ShadowTransfer. This loader instead cuts a centred grid of
non-overlapping 384x384 tiles at **native** resolution, so S-EO patches sit at
0.3 m/px -- the same GSD as ShadowTransfer `hr`. There is no Resize anywhere in
this class.

Leakage
--------------------------------------------------------------------------
Every patch carries its source AOI. Splits are assigned at the AOI level, so
all patches cut from one ground footprint land in exactly one split. An AOI is
one 500x500 m footprint imaged repeatedly. As of the 3-acquisition re-run
(2026-08-01) `armA_seo_prepare.py` keeps THREE acquisitions per AOI, so the
same ground now recurs three times inside the dataset under different sun
positions. Those three images are near-duplicates of each other; AOI-level
splitting is what keeps all three on one side of the boundary.
`assert_no_leakage()` checks no AOI and no source image_id spans two splits and
is called by the smoke test. With multiple acquisitions per AOI it is the check
that matters, not a formality.

Bootstrap unit is the AOI, not the patch
--------------------------------------------------------------------------
Patches from one AOI are not independent -- with 3 acquisitions they share the
same ground three times over. Every row this loader and its manifests emit
carries `aoi_id` so downstream resampling is over AOIs. Nothing in this file
does the resampling; it only guarantees the column is present and non-null.

Unseen pixels are EXCLUDED from loss and metric, not folded into background
--------------------------------------------------------------------------
`_unseen_pixels_mask` flags pixels the DSM could not see. A pixel it flags is
neither foreground nor background: it is absent from the denominator. `valid`
is authoritative -- `n_valid_px` counts only pixels outside it, `n_gt_fg_valid`
and `empty_fg_valid` are computed on the valid region, and `mask_ignore`
carries IGNORE_INDEX (255) there so a `CrossEntropyLoss(ignore_index=255)`
drops them from the loss without any further wiring.

Reason, for the record: DSM coverage differs by city (measured unseen-pixel
fraction differs several-fold between JAX, OMA and UCSD), so folding unseen
pixels into background would make the reported metric partly a measure of DSM
coverage rather than of shadow prediction, and would do so unequally across the
three LOCO folds. The alternative -- counting them as confident background --
would credit a model for 13.3% of pixels whose geometry is unknown.

Provenance recorded because we cite it
--------------------------------------------------------------------------
Pansharpened crops are **colour-corrected and histogram-equalized at source**
(dataset card). The radiometry has already been through a non-linear per-image
transform. Arm F's 4.2 probe is histogram matching; the interaction is an
analyst call.

Labels are **not human annotations**: shadow masks are simulated from an aligned
LiDAR DSM plus solar position through the bundle-adjusted RPC camera. Arm A
measures a *gap*, and LOCO and the within-city upper bound share these labels,
so a stationary label error largely cancels in the difference.

Mask products and label polarity
--------------------------------------------------------------------------
shadows_max ships four single-band PNGs per image, not a multi-band raster.
All four are {0,1}. Measured semantics (24-image brightness test, 2026-08-01):

  _shadow               1 = LIT, 0 = SHADOW      <- inverted vs the obvious read
  _uncertainty_mask     1 = uncertain            34.7% of pixels
  _tree_mask            1 = vegetation (NDVI)    23.1%
  _unseen_pixels_mask   1 = unseen / DSM no-data 13.3%

**The shadow mask is inverted.** Mean image brightness where `_shadow == 1` is
124.2 against 69.9 where it is 0, in the same direction on every one of 24
sampled images. Reading `_shadow > 0` as foreground would train on inverted
labels and would report a shadow fraction of 0.85; the true fraction is 0.156.
`SHADOW_VALUE = 0` encodes this, and `shadow_value` is a constructor argument
so a future S-EO release that flips the convention is a one-line change rather
than a silent regression. `armA_seo_census.py` re-runs the brightness test on
>=50 patches per city on every prep run and fails the job if the direction
flips.

Contrast channel
--------------------------------------------------------------------------
The published MAMNet config (`args.json`, all 144 cells) sets
`use_contrast: true`, so MAMNet consumes RGBC [4,H,W] where the 4th channel is
the RMS contrast map. `use_contrast=True` reproduces that here by importing
`mamnet/utils/contrast_utils.py` by path -- the published implementation,
unmodified and uncopied -- and applying it exactly as
`mamnet/data/dataset_enhanced.py` does: contrast on the uint8 RGB, /255, then
ImageNet normalisation on RGB only and a clamp to [0,1] on the 4th channel.

Categories: S-EO has no six-way shadow taxonomy. Per CONTRACT.md 3, external
datasets write **empty** fn_px_by_cat / tp_px_by_cat / gt_px_by_cat maps.
"""
import importlib.util
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

CITIES = ("JAX", "OMA", "UCSD")
CATEGORY_MAPS_EMPTY = {"fn_px_by_cat": {}, "tp_px_by_cat": {}, "gt_px_by_cat": {}}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
AUX_PRODUCTS = ("_uncertainty_mask", "_tree_mask", "_unseen_pixels_mask")
UNSEEN_PRODUCT = "_unseen_pixels_mask"
# _shadow.png stores 1 = lit, 0 = shadow. Verified by brightness, not assumed.
SHADOW_VALUE = 0
# Pixels outside the valid region. CrossEntropyLoss(ignore_index=255) drops them.
IGNORE_INDEX = 255

_CONTRAST_FN = None


def contrast_fn():
    """`add_contrast_channel` from the published MAMNet tree, imported by path.

    Not reimplemented and not copied: `mamnet/utils/contrast_utils.py` is
    the implementation the published cells used, and it has no package-relative
    imports, so loading it by file path is exact.
    """
    global _CONTRAST_FN
    if _CONTRAST_FN is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "mamnet", "utils", "contrast_utils.py")
        path = os.path.normpath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"contrast_utils.py not found at {path}")
        spec = importlib.util.spec_from_file_location("st_contrast_utils", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CONTRAST_FN = mod.add_contrast_channel
    return _CONTRAST_FN


def assert_no_leakage(datasets):
    """No AOI, and no source image, appears in more than one split."""
    seen_aoi, seen_img = {}, {}
    for ds in datasets:
        tag = f"{ds.city}/{ds.split}"
        for aoi in {r["aoi_id"] for r in ds.rows}:
            if seen_aoi.setdefault(aoi, tag) != tag:
                raise AssertionError(f"AOI {aoi} in {seen_aoi[aoi]} and {tag}")
        for iid in {r["image_id"] for r in ds.rows}:
            if seen_img.setdefault(iid, tag) != tag:
                raise AssertionError(f"image {iid} in {seen_img[iid]} and {tag}")
    return {"n_aoi": len(seen_aoi), "n_images": len(seen_img), "leakage": False}


class SEODataset(Dataset):
    """One item = one 384x384 patch cut at native GSD.

    Args:
        img_root, mask_root: extracted product roots
        patch_index:  logs/armA_seo_patch_index*.json from armA_seo_prepare.py
        split_manifest: seo_splits*/{city}.json (AOI lists)
        split: 'train' | 'val' | 'test'
        use_contrast: append the published RMS-contrast 4th channel (MAMNet)
        exclude_unseen: unseen pixels leave the valid region (default, and the
            arm's committed choice -- see the module docstring)
    """

    def __init__(self, img_root, mask_root, patch_index, split_manifest, split,
                 patch=384, augment=False, return_aux=False,
                 shadow_value=SHADOW_VALUE, exclude_unseen=True,
                 exclude_uncertain=False, use_contrast=False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        idx = patch_index if isinstance(patch_index, dict) else json.load(open(patch_index))
        images = idx["images"]
        cfg = idx.get("config", {})
        if cfg.get("patch") and cfg["patch"] != patch:
            raise ValueError(
                f"patch index was built at {cfg['patch']} px but loader asked for "
                f"{patch}; rebuild the index rather than re-tiling here.")
        man = split_manifest if isinstance(split_manifest, dict) else json.load(open(split_manifest))

        self.city, self.split = man["city"], split
        self.img_root, self.mask_root = img_root, mask_root
        self.patch, self.return_aux = patch, return_aux
        self.shadow_value = shadow_value
        self.exclude_unseen, self.exclude_uncertain = exclude_unseen, exclude_uncertain
        self.use_contrast = use_contrast
        self.augment = augment and split == "train"
        self.split_rule_sha = man["split_rule_sha"]
        self.acq_per_aoi = cfg.get("acq_per_aoi")
        self.radiometry = "uint8_histeq"
        self.gsd_note = (f"{patch}x{patch} tile at native ~0.3 m GSD, no resize "
                         "(matches ShadowTransfer hr)")

        want = set(man[split])
        self.rows = []
        for image_id, rec in sorted(images.items()):
            if rec["aoi_id"] not in want:
                continue
            for (y, x) in rec["patches"]:
                self.rows.append({"image_id": image_id, "aoi_id": rec["aoi_id"],
                                  "city": rec["city"], "image": rec["image"],
                                  "mask": rec["mask"], "aux": rec.get("aux", {}),
                                  "acq": rec.get("acq"), "sun_el": rec.get("sun_el"),
                                  "sun_az": rec.get("sun_az"),
                                  "lon": rec.get("lon"), "lat": rec.get("lat"),
                                  "y": y, "x": x})
        # Bootstrap unit is the AOI: the column must exist on every row.
        bad = [r for r in self.rows if not r.get("aoi_id")]
        if bad:
            raise AssertionError(
                f"{self.city}/{split}: {len(bad)} patch rows have a null aoi_id; "
                "downstream analysis resamples AOIs and cannot proceed")
        self.norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self):
        return len(self.rows)

    def describe(self):
        return {
            "city": self.city, "split": self.split,
            "n_patches": len(self), "n_images": len({r["image_id"] for r in self.rows}),
            "n_aoi": len({r["aoi_id"] for r in self.rows}),
            "patch": self.patch, "radiometry": self.radiometry,
            "gsd_note": self.gsd_note, "split_rule_sha": self.split_rule_sha,
            "acq_per_aoi": self.acq_per_aoi,
            "resize_applied": False,
            "shadow_value": self.shadow_value,
            "use_contrast": self.use_contrast,
            "in_channels": 4 if self.use_contrast else 3,
            "polarity_note": "_shadow.png stores 1=lit, 0=shadow (measured)",
            "valid_region": ("excludes _unseen_pixels_mask" if self.exclude_unseen else "all pixels")
                            + ("; excludes _uncertainty_mask" if self.exclude_uncertain else ""),
            "valid_region_semantics": (
                "authoritative: n_valid_px, n_gt_fg_valid and empty_fg_valid are "
                "computed on it, and mask_ignore carries 255 outside it so the "
                "loss drops those pixels. Unseen pixels are neither foreground "
                "nor background."),
            "bootstrap_unit": "aoi_id",
            "category_maps": "empty (CONTRACT.md 3: external datasets write empty maps)",
        }

    @staticmethod
    def _crop(path, y, x, p, expect_bands):
        with Image.open(path) as im:
            a = np.array(im.crop((x, y, x + p, y + p)))
        if a.ndim == 2:
            a = a[:, :, None]
        if a.shape[0] != p or a.shape[1] != p:
            raise ValueError(f"{path}: crop at ({y},{x}) gave {a.shape[:2]}, want {p}x{p}")
        if expect_bands and a.shape[2] < expect_bands:
            raise ValueError(f"{path}: expected >={expect_bands} bands, got {a.shape[2]}")
        return a

    def _to_tensor(self, arr):
        """uint8 [H,W,3] -> normalised float tensor, RGB or RGBC."""
        if self.use_contrast:
            rgbc = contrast_fn()(np.ascontiguousarray(arr[:, :, :3]))
            t = torch.from_numpy(
                np.ascontiguousarray(rgbc).transpose(2, 0, 1)).float().div_(255.0)
            t[:3] = self.norm(t[:3])
            t[3:4] = t[3:4].clamp(0, 1)
            return t
        return self.norm(torch.from_numpy(
            np.ascontiguousarray(arr[:, :, :3]).transpose(2, 0, 1)).float().div_(255.0))

    def __getitem__(self, i):
        r = self.rows[i]
        y, x, p = r["y"], r["x"], self.patch

        arr = self._crop(os.path.join(self.img_root, r["image"]), y, x, p, 3)
        if arr.dtype != np.uint8:
            raise TypeError(
                f"{r['image']}: expected uint8 pansharpened crop, got {arr.dtype}. "
                "MSI/panchromatic are uint16 and are not supported; rescaling them "
                "here would be an undeclared radiometric decision.")
        image = self._to_tensor(arr)

        m = self._crop(os.path.join(self.mask_root, r["mask"]), y, x, p, 1)[:, :, 0]
        vals = np.unique(m)
        if not set(vals.tolist()) <= {0, 1}:
            raise ValueError(
                f"{r['mask']}: shadow mask is not binary in {{0,1}} (values "
                f"{vals[:8]}). That file is probably an uncertainty or "
                "vegetation product, or the encoding changed.")
        fg = (m == self.shadow_value)

        # valid region: pixels the geometry actually determined. Authoritative.
        valid = np.ones_like(fg, dtype=bool)
        for prod, on in ((UNSEEN_PRODUCT, self.exclude_unseen),
                         ("_uncertainty_mask", self.exclude_uncertain)):
            rel = r["aux"].get(prod)
            if on and rel:
                flag = self._crop(os.path.join(self.mask_root, rel), y, x, p, 1)[:, :, 0]
                valid &= (flag == 0)

        mask = torch.from_numpy(fg.astype(np.int64))
        if self.augment:
            vt = torch.from_numpy(valid)
            if torch.rand(1).item() < 0.5:
                image, mask, vt = (torch.flip(image, [2]), torch.flip(mask, [1]),
                                   torch.flip(vt, [1]))
            if torch.rand(1).item() < 0.5:
                image, mask, vt = (torch.flip(image, [1]), torch.flip(mask, [0]),
                                   torch.flip(vt, [0]))
            valid = vt.numpy()

        vt = torch.from_numpy(np.ascontiguousarray(valid))
        mask_ignore = torch.where(vt, mask, torch.full_like(mask, IGNORE_INDEX))
        n_valid_px = int(vt.sum())
        n_gt_fg_valid = int((mask.bool() & vt).sum())

        out = {"image": image, "mask": mask, "mask_ignore": mask_ignore,
               "valid": vt,
               "n_valid_px": n_valid_px,
               "n_gt_fg_valid": n_gt_fg_valid,
               "empty_fg_valid": bool(n_gt_fg_valid == 0),
               "filename": f"{r['image_id']}_y{y}_x{x}",
               "image_id": r["image_id"], "aoi_id": r["aoi_id"], "city": r["city"],
               "patch_yx": (y, x)}
        if self.return_aux:
            for prod in AUX_PRODUCTS:
                rel = r["aux"].get(prod)
                out["aux" + prod] = (
                    torch.from_numpy(
                        (self._crop(os.path.join(self.mask_root, rel), y, x, p, 1)[:, :, 0] > 0
                         ).astype(np.uint8))
                    if rel else None)
        return out
