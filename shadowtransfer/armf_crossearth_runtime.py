"""Arm F / CrossEarth — shared runtime.

Weakness IDs answered: v1UJ-W3, AC1.  target_access: source_only.

Everything here is shared by the stage-0 environment gate and the stage-1
timing pilot.  Nothing in this file computes a metric: metrics come from
`shadowtransfer.metrics` and nowhere else (CONTRACT.md rule zero).

WHAT THE RELEASED CrossEarth CODE ACTUALLY CONTAINS
---------------------------------------------------
Checked against github.com/VisionXLab/CrossEarth @ 644a5a1 on 2026-07-30.
The released package registers exactly four things:

    ReinsDinoVisionTransformer, DinoVisionTransformer, RVSA_MTP  (backbones)
    ReinMask2FormerHead                                          (head)
    PEFTEncoderDecoder                                           (segmentor)
    PEFTOptimWrapperConstructor                                  (optimizer)

There is no Geospatial Semantic Extractor, no cross-attention Injector, no
ASPP / masked-image-modelling branch, and no Earth-Style Injection.  The
repo's README says "Training steps: Coming soon."  What can be trained from
the release is frozen DINOv2 ViT-L/14-at-patch-16 + LoRA Reins + Mask2Former.
That is a subset of the architecture described in the TPAMI paper, and any
run built on this file must record it as such.
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

FOLD_OF_CITY = {"phoenix": 0, "miami": 1, "chicago": 2}
RES_DIR = {"hr": "highres", "mr": "midres"}

# Components named in the CrossEarth paper that the code release does not ship.
# Carried into run.json verbatim so a reader never has to rediscover it.
MISSING_FROM_RELEASE = [
    "geospatial_semantic_extractor",
    "cross_attention_injector",
    "aspp_masked_image_modelling_decoder",
    "earth_style_injection",
]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def add_repo_to_path(repo_root):
    """Make `import CrossEarth` resolve to the cloned release."""
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def split_dir(base_path, holdout, res):
    """Absolute path of one LOCO fold-resolution split directory."""
    fold = FOLD_OF_CITY[holdout]
    return os.path.join(
        base_path, "data", "Final_data_loco",
        f"fold_{fold}_holdout_{holdout}", RES_DIR[res])


def load_split(base_path, holdout, res, split):
    """Return (image_paths, mask_paths, metadata records) for one split.

    Ordering is the sorted image-filename order and is fixed forever: the
    per-image parquet joins on image_id, but probs.npz and any future
    positional read depend on this being stable.
    """
    root = split_dir(base_path, holdout, res)
    meta_path = os.path.join(root, f"metadata_{split}.json")
    with open(meta_path) as f:
        meta = json.load(f)
    img_dir = os.path.join(root, split, "images")
    msk_dir = os.path.join(root, split, "masks")
    names = sorted(os.listdir(img_dir))
    imgs = [os.path.join(img_dir, n) for n in names]
    msks = [os.path.join(msk_dir, n) for n in names]
    for m in msks:
        if not os.path.exists(m):
            raise FileNotFoundError(f"mask missing for {m}")
    return imgs, msks, meta


def assert_source_only(base_path, holdout, res):
    """Protocol deviation 1: no held-out-city image in train or val.

    The released DG configs put the target domain's val split in the
    validation dataloader.  Our splits do not, but that is a property of the
    data we must verify rather than assume, so this asserts it two ways: on
    every train/val filename, and on the `city` field of every metadata
    record.  Raises rather than returning False.
    """
    report = {}
    for split in ("train", "val"):
        imgs, _, meta = load_split(base_path, holdout, res, split)
        with open(os.path.join(split_dir(base_path, holdout, res),
                               f"metadata_{split}.json")) as f:
            recs = json.load(f)
        bad_names = [p for p in imgs if holdout in os.path.basename(p).lower()]
        bad_meta = [r for r in recs if r.get("city") == holdout]
        if bad_names or bad_meta:
            raise AssertionError(
                f"held-out city {holdout} present in {split}: "
                f"{len(bad_names)} filenames, {len(bad_meta)} metadata records")
        cities = sorted({r.get("city") for r in recs})
        report[split] = {"n": len(imgs), "cities": cities,
                         "n_meta": len(recs), "heldout_hits": 0}
    imgs, _, _ = load_split(base_path, holdout, res, "test")
    report["test"] = {"n": len(imgs), "cities": [holdout]}
    return report


class ShadowSegDataset(Dataset):
    """384x384 RGB PNG + {0,255} single-channel mask.

    Returns uint8 CHW RGB and int64 HW in {0,1}.  We hand RGB straight to the
    model and build the data preprocessor with bgr_to_rgb=False; the released
    pipeline loads BGR through mmcv and flips it with bgr_to_rgb=True, so the
    tensor the network sees is identical.  This is an implementation detail,
    not a deviation.
    """

    def __init__(self, img_paths, mask_paths, limit=None):
        if limit is not None:
            img_paths = img_paths[:limit]
            mask_paths = mask_paths[:limit]
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = np.asarray(Image.open(self.img_paths[i]).convert("RGB"))
        msk = np.asarray(Image.open(self.mask_paths[i]).convert("L"))
        gt = (msk > 127).astype(np.int64)
        return {
            "img": torch.from_numpy(img.transpose(2, 0, 1).copy()),
            "gt": torch.from_numpy(gt),
            "image_id": os.path.splitext(os.path.basename(self.img_paths[i]))[0],
            "img_path": self.img_paths[i],
        }


def collate(batch):
    return {
        "img": [b["img"] for b in batch],
        "gt": [b["gt"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "img_path": [b["img_path"] for b in batch],
    }


def build_cfg(repo_root, ckpt_path, num_classes=2, crop=384):
    """Load the released model config and retarget it to our task.

    Deviations from the released config, all deliberate and all recorded:
      crop 512 -> 384   (native patch size; 1024 -> 576 tokens)
      num_classes -> 2  (background, shadow)
      test_cfg mode slide -> whole  (image size equals crop size)
    """
    from mmengine.config import Config

    cfg_path = os.path.join(
        repo_root, "configs", "_base_", "models",
        "CrossEarth_dinov2_mask2former.py")
    cfg = Config.fromfile(cfg_path)
    m = cfg.model

    m.data_preprocessor.size = (crop, crop)
    m.data_preprocessor.bgr_to_rgb = False
    m.backbone.img_size = crop
    m.backbone.init_cfg.checkpoint = ckpt_path
    m.decode_head.num_classes = num_classes
    m.decode_head.loss_cls.class_weight = [1.0] * num_classes + [0.1]
    m.test_cfg = dict(mode="whole")
    return cfg


def build_model(cfg, device="cuda"):
    """Construct, initialise, and move the model. Registers CrossEarth first."""
    import CrossEarth  # noqa: F401  registers backbones/heads/segmentors
    from mmengine.registry import init_default_scope
    from mmseg.registry import MODELS

    # Without the mmseg default scope the nested build of SegDataPreProcessor
    # is looked up in mmengine::model and raises KeyError.
    init_default_scope("mmseg")
    model = MODELS.build(cfg.model)
    model.init_weights()
    model = model.to(device)
    return model


def param_report(model):
    """Counts after the backbone's own freeze rule has been applied.

    ReinsDinoVisionTransformer.train(True) calls set_requires_grad(self,
    ["reins"]), so the freeze only exists once the model has been put in
    train mode.  Call this after model.train().
    """
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bb_total = sum(p.numel() for p in model.backbone.parameters())
    bb_train = sum(p.numel() for n, p in model.backbone.named_parameters()
                   if p.requires_grad)
    bb_train_names = sorted({n.split(".")[0] for n, p
                             in model.backbone.named_parameters()
                             if p.requires_grad})
    return {
        "n_params_total": int(total),
        "n_trainable_params": int(train),
        "n_frozen_params": int(total - train),
        "backbone_n_params": int(bb_total),
        "backbone_n_trainable": int(bb_train),
        "backbone_trainable_top_level": bb_train_names,
        # The published design trains reins only inside the backbone.
        "backbone_frozen_except_reins": bb_train_names == ["reins"],
    }


def make_data(batch, model, training):
    """Pack a collated batch into the form EncoderDecoder.loss/predict want."""
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample

    samples = []
    for gt, path in zip(batch["gt"], batch["img_path"]):
        s = SegDataSample()
        h, w = int(gt.shape[0]), int(gt.shape[1])
        s.set_metainfo(dict(img_shape=(h, w), ori_shape=(h, w),
                            pad_shape=(h, w), scale_factor=(1.0, 1.0),
                            reduce_zero_label=False, img_path=path,
                            seg_map_path=path))
        s.gt_sem_seg = PixelData(data=gt[None])
        samples.append(s)
    data = {"inputs": batch["img"], "data_samples": samples}
    return model.data_preprocessor(data, training=training)


def foreground_prob(seg_logits):
    """P(shadow) from a Mask2Former semantic score map.

    ReinMask2FormerHead.predict returns
    einsum('bqc,bqhw->bchw', softmax(cls)[..., :-1], sigmoid(mask)).  Those
    channels are non-negative but do not sum to exactly 1, so the foreground
    probability is the normalised foreground channel rather than a softmax of
    the two.  Recorded in run.json as prob_source.
    """
    s = seg_logits.float()
    denom = s.sum(dim=0).clamp_min(1e-12)
    return (s[1] / denom).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)


def patch_msda_fp32():
    """Run mmcv's deformable attention in fp32 under a bf16 autocast region.

    mmcv 2.1.0 compiles ms_deform_attn_forward_cuda for float and half only:
    under torch.autocast(bfloat16) the Mask2Former pixel decoder dies with
    RuntimeError: "ms_deform_attn_forward_cuda" not implemented for
    'BFloat16'.  Observed on job 20599586, gpua017, 2026-07-30.

    The op is a small share of the FLOPs -- the frozen ViT-L forward dominates
    -- so forcing this one module to fp32 keeps the requested bf16 autocast on
    the part that pays for it.  This makes the precision "bf16 autocast except
    deformable attention", not plain bf16 autocast, and every run that calls
    this records deform_attn_fp32 = true and carries a caveat.
    """
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttention as MSDA)

    if getattr(MSDA, "_st_fp32_patched", False):
        return False
    original = MSDA.forward

    def _f32(x):
        return x.float() if torch.is_tensor(x) and x.is_floating_point() else x

    def forward(self, *args, **kwargs):
        with torch.autocast("cuda", enabled=False):
            args = tuple(_f32(a) for a in args)
            kwargs = {k: _f32(v) for k, v in kwargs.items()}
            return original(self, *args, **kwargs)

    MSDA.forward = forward
    MSDA._st_fp32_patched = True
    return True


def gpu_banner(tag=""):
    if not torch.cuda.is_available():
        return {"gpu": None}
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    info = {
        "gpu": props.name,
        "gpu_total_gb": round(props.total_memory / 2 ** 30, 2),
        "gpu_alloc_gb": round(torch.cuda.memory_allocated(i) / 2 ** 30, 2),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated(i) / 2 ** 30, 2),
    }
    print(f"[vram{(' ' + tag) if tag else ''}] {json.dumps(info)}", flush=True)
    return info


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {"seed": seed, "seed_numpy": seed, "seed_python": seed,
            "seed_torch": seed, "seed_cuda": seed}
