"""Arm F / CrossEarth — the 12-cell table, split resolution, FDA, augmentation.

Weakness IDs: v1UJ-W3, AC1 (gap row); AC3, v1UJ-W2 (the fda and sib rows).

Nothing here computes a metric. Metrics come from shadowtransfer.metrics and
the CONTRACT row comes from shadowtransfer.armc_dump (CONTRACT.md rule zero).

SPLIT LAYOUT, verified on disk 2026-07-30
-----------------------------------------
  loco   data/Final_data_loco/fold_{k}_holdout_{city}/highres/{train,val,test}
         train 450 = 225 + 225 from the two source cities
         val   150 =  75 +  75 source-city
         test  150 held-out city
         filenames carry a `{source_city}__` prefix
  upper  data/Final_data_test/{city}/highres/{train,val,test}
         450 / 150 / 150, all that one city, filenames unprefixed

The loco test split is byte-identical in filename order to the matching upper
test split for all three cities, so the gap row is paired per image.
"""

import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

CITIES = ("chicago", "miami", "phoenix")
FOLD_OF_CITY = {"phoenix": 0, "miami": 1, "chicago": 2}
RES_DIR = {"hr": "highres", "mr": "midres"}

# The 12 cells of the arm, in spec order.
CELLS = (
    [{"cell_type": "upper", "method": "vanilla", "city": c} for c in CITIES]
    + [{"cell_type": "loco", "method": "vanilla", "city": c} for c in CITIES]
    + [{"cell_type": "loco", "method": "fda", "city": c} for c in CITIES]
    + [{"cell_type": "loco", "method": "sib", "city": c} for c in CITIES]
)

WEAKNESS_IDS = {
    "vanilla": ["v1UJ-W3", "AC1"],
    "fda": ["v1UJ-W3", "AC1", "AC3", "v1UJ-W2"],
    "sib": ["v1UJ-W3", "AC1", "AC3", "v1UJ-W2"],
}

# Verbatim, on every results.json of this arm. Do not paraphrase.
RELEASE_CAVEAT = (
    "CrossEarth run from the public release (VisionXLab/CrossEarth @ 644a5a1) "
    "with our LOCO splits, source-city validation only, crop 384, batch 4, "
    "bf16 with deformable attention in fp32, and a 10,000-iteration budget set "
    "from our own 100-epoch protocol. The GSE, Injector, ASPP masked-image-"
    "modelling branch and Earth-Style Injection components described in the "
    "paper are not present in the released code."
)


def run_id_of(cell_type, method, city, res="hr", rung=450, seed=0):
    """CONTRACT.md §1 run_id. `crossearth` is the arch slug; the results.json
    `architecture` field carries the display string "CrossEarth"."""
    return f"crossearth-{method}-{cell_type}-r{rung}-{city}-{res}-s{seed}"


def cell_dirname(cell_type, method, city, res="hr"):
    """runs/ directory name fixed by the arm spec."""
    return f"armF_crossearth_{cell_type}_{method}_{city}_{RES_DIR[res]}"


def split_root(base_path, cell_type, city, res="hr"):
    if cell_type == "loco":
        return os.path.join(base_path, "data", "Final_data_loco",
                            f"fold_{FOLD_OF_CITY[city]}_holdout_{city}",
                            RES_DIR[res])
    if cell_type == "upper":
        return os.path.join(base_path, "data", "Final_data_test", city,
                            RES_DIR[res])
    raise ValueError(cell_type)


def image_id_of(filename):
    """Stem of the benchmark's own `original_filename`.

    LOCO filenames are prefixed with the image's SOURCE city
    (`chicago__chicago_session01_highres_random_001.png`); upper-bound
    filenames are not. Stripping the prefix makes the loco and upper rows for
    the same test image share an image_id, which is what the paired gap needs.
    """
    b = os.path.basename(filename)
    if "__" in b:
        b = b.split("__", 1)[1]
    return os.path.splitext(b)[0]


def load_split(base_path, cell_type, city, split, res="hr"):
    """(image_paths, mask_paths, metadata list) in sorted-filename order.

    The order is fixed forever: probs.npz is keyed by image_id, but anything
    positional downstream depends on this being stable.
    """
    root = split_root(base_path, cell_type, city, res)
    with open(os.path.join(root, f"metadata_{split}.json")) as f:
        meta = json.load(f)
    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")
    names = sorted(os.listdir(img_dir))
    imgs = [os.path.join(img_dir, n) for n in names]
    msks = [os.path.join(msk_dir, n) for n in names]
    missing = [m for m in msks if not os.path.exists(m)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} masks missing, first {missing[0]}")
    return imgs, msks, meta


def assert_no_heldout(base_path, cell_type, city, res="hr"):
    """No held-out-city image in any training or validation dataloader.

    Checked two ways on every train and val image: the filename, and the
    `city` field of the split's own metadata record. Raises rather than
    returning False. FDA's unlabelled target pool is the single declared
    exception and does not pass through here.

    For an `upper` cell there is no held-out city; the assertion becomes
    "every train/val/test image is this city and no other", which is the
    property that makes it a within-city upper bound.
    """
    report = {"cell_type": cell_type, "city": city, "heldout_hits": 0}
    for split in ("train", "val", "test"):
        imgs, _, recs = load_split(base_path, cell_type, city, split, res)
        cities = sorted({r.get("city") for r in recs})
        if cell_type == "loco":
            bad_name = [p for p in imgs
                        if city in os.path.basename(p).lower()]
            bad_meta = [r for r in recs if r.get("city") == city]
            if split in ("train", "val"):
                hits = len(bad_name) + len(bad_meta)
                report["heldout_hits"] += hits
                if hits:
                    raise AssertionError(
                        f"held-out city {city} present in {split}: "
                        f"{len(bad_name)} filenames, {len(bad_meta)} metadata")
            elif cities != [city]:
                raise AssertionError(f"loco test split is not all {city}: {cities}")
        else:
            if cities != [city]:
                raise AssertionError(
                    f"upper {city} {split} contains {cities}")
        report[split] = {"n": len(imgs), "cities": cities,
                         "n_meta": len(recs)}
    return report


def fda_pool_dir(base_path, city, res="hr"):
    """Held-out city's own train/images. Unlabelled: masks are never read."""
    return os.path.join(split_root(base_path, "upper", city, res),
                        "train", "images")


def build_fda_pool(base_path, city, pool_size, seed, res="hr"):
    """Up to `pool_size` unlabelled held-out-city training images.

    Sampled uniformly without replacement from the full held-out train split
    under a dedicated RNG so the pool does not depend on training-time draws.
    Returns (paths, dir, n_available).
    """
    d = fda_pool_dir(base_path, city, res)
    names = sorted(os.listdir(d))
    rng = np.random.default_rng(seed)
    k = min(pool_size, len(names))
    idx = rng.choice(len(names), size=k, replace=False)
    return [os.path.join(d, names[i]) for i in sorted(idx)], d, len(names)


def fda_swap(src, tgt, L=0.01):
    """Appendix F.2 FDA, applied at the input stage before resize/normalise.

    Per-channel 2-D FFT; the central amplitude rectangle of half-width
    b = floor(min(H, W) * L) is taken from the target, the source phase is
    preserved, and the reconstruction is clipped to [0, 255]. At 384 with
    L = 0.01, b = 3 and the swapped region is 6x6 per channel.

    Args:
        src, tgt: uint8 HWC arrays, same shape.
    Returns:
        uint8 HWC array.
    """
    h, w = src.shape[:2]
    b = int(np.floor(min(h, w) * L))
    if b < 1:
        return src.copy()
    fs = np.fft.fft2(src.astype(np.float32), axes=(0, 1))
    ft = np.fft.fft2(tgt.astype(np.float32), axes=(0, 1))
    amp_s = np.fft.fftshift(np.abs(fs), axes=(0, 1))
    amp_t = np.fft.fftshift(np.abs(ft), axes=(0, 1))
    cy, cx = h // 2, w // 2
    amp_s[cy - b:cy + b, cx - b:cx + b] = amp_t[cy - b:cy + b, cx - b:cx + b]
    amp_s = np.fft.ifftshift(amp_s, axes=(0, 1))
    out = np.fft.ifft2(amp_s * np.exp(1j * np.angle(fs)), axes=(0, 1)).real
    return np.clip(out, 0, 255).astype(np.uint8)


def fda_half_width(crop=384, L=0.01):
    return int(np.floor(crop * L))


# --- the released train pipeline, ported from 512 to our 384 patch ----------
# configs/CrossEarth_dinov2/CrossEarth_dinov2_mask2former_512x512_bs1x4.py
# Transform types, their order and every other argument are the released ones,
# built through the mmseg registry rather than reimplemented. Two ports:
#
#   1. base length 512 -> 384, which rescales the resize ladder and max_size
#      proportionally.
#   2. the ladder's sub-1.0x rungs are dropped: released
#      `range(5, 21)` spans 0.5x-2.0x, ours spans 1.0x-2.0x.
#
# Port 2 is forced by our data, not chosen. The released datasets are large
# tiles, where a shorter-edge resize to 0.5x still leaves far more than one
# crop's worth of pixels. Our images are exactly one 384x384 patch, so any
# rung below 1.0x makes RandomCrop(384) return a partial crop (measured: a
# 0.5x rung gives 268x268) which SegDataPreProcessor then zero-pads to 384
# with seg_pad_val 255. That trains a large share of iterations on padding
# rather than on imagery, which is a bigger deviation than dropping the rungs.
def train_pipeline_cfg(crop=384):
    return [
        dict(type="RandomChoiceResize",
             scales=[int(crop * x * 0.1) for x in range(10, 21)],
             resize_type="ResizeShortestEdge",
             max_size=int(2048 * crop / 512)),
        dict(type="RandomCrop", crop_size=(crop, crop), cat_max_ratio=0.75),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PhotoMetricDistortion"),
    ]


def build_train_pipeline(crop=384):
    from mmcv.transforms import Compose
    from mmengine.registry import init_default_scope
    from mmseg.registry import TRANSFORMS
    init_default_scope("mmseg")
    return Compose([TRANSFORMS.build(c) for c in train_pipeline_cfg(crop)])


class ShadowTrainDataset(Dataset):
    """Infinite-index training dataset: FDA (optional) then the released
    augmentation pipeline. Returns uint8 CHW RGB and int64 HW in {0, 1}."""

    def __init__(self, img_paths, mask_paths, pipeline, fda_pool=None,
                 fda_L=0.01, seed=0):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.pipeline = pipeline
        self.fda_pool = fda_pool
        self.fda_L = fda_L
        self.rng = np.random.default_rng(seed + 9973)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.img_paths[i]).convert("RGB"))
        msk = np.asarray(Image.open(self.mask_paths[i]).convert("L"))
        gt = (msk > 127).astype(np.uint8)
        if self.fda_pool:
            j = int(self.rng.integers(len(self.fda_pool)))
            tgt = np.asarray(Image.open(self.fda_pool[j]).convert("RGB"))
            img = fda_swap(img, tgt, L=self.fda_L)
        res = self.pipeline({
            "img": img, "gt_seg_map": gt,
            "img_shape": img.shape[:2], "ori_shape": img.shape[:2],
            "seg_fields": ["gt_seg_map"],
        })
        return {"img": np.ascontiguousarray(res["img"].transpose(2, 0, 1)),
                "gt": res["gt_seg_map"].astype(np.int64),
                "image_id": image_id_of(self.img_paths[i]),
                "img_path": self.img_paths[i]}


class ShadowEvalDataset(Dataset):
    """Validation and test: images and masks untouched, per Appendix F.2."""

    def __init__(self, img_paths, mask_paths, limit=None):
        self.img_paths = img_paths[:limit] if limit else img_paths
        self.mask_paths = mask_paths[:limit] if limit else mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.img_paths[i]).convert("RGB"))
        msk = np.asarray(Image.open(self.mask_paths[i]).convert("L"))
        return {"img": np.ascontiguousarray(img.transpose(2, 0, 1)),
                "gt": (msk > 127).astype(np.int64),
                "image_id": image_id_of(self.img_paths[i]),
                "img_path": self.img_paths[i]}


def collate(batch):
    import torch
    return {"img": [torch.from_numpy(b["img"]) for b in batch],
            "gt": [torch.from_numpy(b["gt"]) for b in batch],
            "image_id": [b["image_id"] for b in batch],
            "img_path": [b["img_path"] for b in batch]}
