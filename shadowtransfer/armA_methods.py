#!/usr/bin/env python3
"""Arm A / S-EO PHASE 3 -- the method layer: FDA and SIB, mapped onto S-EO.

Nothing here reimplements a method. Every component is either imported from the
published tree by path or constructed from the published architecture module,
and every hyperparameter is lifted from a published `args.json`/`config.json`
that is named beside it. What this file owns is the MAPPING onto S-EO: where
the FDA target pool comes from, how the intensity map is produced when the
loader does not carry one, and how the published SIB model, optimiser,
scheduler, gradient clipping and KL term are installed into the phase-2 code
path without editing a published file.

Which published cell each number comes from
-------------------------------------------
  FDA, both archs   data/{arch}/outputs/{arch}_fda_loco_holdout_chicago_highres_1/args.json
  SIB, mamnet       data/mamnet/outputs/mamnet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_chicago__highres/config.json
  SIB, dinov3       data/dinov3/outputs/dinov3_sib_C4clean_haar_vib_loco_holdout_chicago_highres_1/args.json

C4-clean IS the published SIB. `Final_Experiments_Results.md` records the
decision: adaptive beta and content augmentation were dropped after their
ablations failed, leaving Haar + content VIB + edge VIB, plus SAG and FDA on
the CNN branch only. Both on-disk configs agree (`adaptive_beta: false`,
`use_content_aug: false`).

Three places where App H's prose and the published configs disagree. The
CONFIGS are used, because they are what produced the published numbers, and
the disagreement is reported rather than reconciled:

  1. App H defines an intensity-conditioned beta_content(x). Both C4-clean
     configs set `adaptive_beta: false`, so no published SIB cell used it. The
     intensity map is still built and still passed (below) so the code path is
     the published one, but it reaches an inactive branch.
  2. Table 37 lists beta_base 0.01 / beta_edge 1e-4 for every architecture.
     That matches DINOv3's config; MAMNet's config is 1e-3 / 1e-5, ten times
     smaller. Per-arch config values are used.
  3. Table 37 lists AdamW wd 0.01 and, for DINOv3, split encoder/head learning
     rates. The published SIB trainers build Adam(wd=1e-4) for MAMNet and
     AdamW(wd=0.05) single-LR for DINOv3. The trainers are followed.

Coverage thresholding is NOT applied here
-----------------------------------------
App H's deployment-time coverage threshold c*_val is absent from both
published SIB training scripts -- it is a post-hoc step over stored
probabilities, computed in `final_loco/composition_analysis/`. Arm A therefore
trains and dumps at full coverage, exactly as its vanilla cells do, and stores
`probs.npz` unthresholded so the threshold is recomputable downstream without
a GPU. A downstream cell that applies it becomes `deployment_time`; a training
run that does not apply it is not.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

from shadowtransfer.seo_dataset import (SEODataset, IMAGENET_MEAN,
                                        IMAGENET_STD)

METHODS = ("fda", "sib")

#: Standalone FDA. Both published cells: L=0.01, <=100 unlabelled target
#: images, applied at the input on the TRAIN split only.
FDA_SPEC = {
    "mamnet": {
        "fda_L": 0.01,
        "target_pool_max": 100,
        "published_config": "data/mamnet/outputs/mamnet_fda_loco_holdout_chicago_highres_1/args.json",
        "published_target_root": "{holdout}/highres/train/images",
        "published_early_stop_patience": 15,
    },
    "dinov3": {
        "fda_L": 0.01,
        "target_pool_max": 100,
        "published_config": "data/dinov3/outputs/dinov3_fda_loco_holdout_chicago_highres_1/args.json",
        "published_target_root": "{holdout}/highres/train/images",
        "published_early_stop_patience": 10,
    },
}

#: SIB = C4-clean. `flags` are consumed verbatim by the published factory.
SIB_SPEC = {
    "mamnet": {
        "published_config": ("data/mamnet/outputs/mamnet_sib_C4clean_haar_vib_sag_fda_ctr"
                             "__loco_holdout_chicago__highres/config.json"),
        "components": ["haar", "content_vib", "edge_vib", "sag", "fda_preproc",
                       "contrast_channel"],
        "components_off": ["content_aug", "adaptive_beta", "multiscale_sib",
                           "passthrough_gate", "module_bypass", "cacr",
                           "ce_aurc", "tent"],
        "flags": {
            "use_haar": True, "use_vib": True, "use_content_aug": False,
            "adaptive_beta": False, "use_sag": True,
            "use_multiscale_sib": False, "use_passthrough_gate": False,
            "use_module_bypass": False, "use_cacr": False, "use_ce_aurc": False,
            "use_tent": False, "symmetric_vib": False, "aug_all_subbands": False,
            "no_edge_vib": False, "vib_wrong_subband": False,
            "beta_content": 0.001, "beta_edge": 1e-05, "noise_scale": 0.1,
            "beta_max_multiplier": 3.0, "multiscale_beta_base": 0.0001,
            "use_contrast": True,
        },
        "vib_warmup_fraction": 0.1,
        "use_fda": True, "fda_L": 0.005, "target_pool_max": 100,
        "grad_clip_max_norm": 5.0,
        "optimizer": "Adam(lr, weight_decay=1e-4)",
        "scheduler": "CosineAnnealingLR(T_max=epochs_max, eta_min=1e-6)",
        "kl_term": "task_loss + vib_warmup_weight(epoch) * kl_total",
    },
    "dinov3": {
        "published_config": ("data/dinov3/outputs/dinov3_sib_C4clean_haar_vib"
                             "_loco_holdout_chicago_highres_1/args.json"),
        "components": ["haar", "content_vib", "edge_vib"],
        "components_off": ["sag", "fda_preproc", "contrast_channel",
                           "content_aug", "adaptive_beta", "passthrough_gate",
                           "module_bypass", "cacr", "ce_aurc", "tent"],
        "flags": {
            "use_haar": True, "use_vib": True, "use_content_aug": False,
            "adaptive_beta": False, "use_passthrough_gate": False,
            "use_module_bypass": False, "disable_content_vib": False,
            "symmetric_vib": False, "aug_all_subbands": False,
            "vib_on_hl_only": False, "use_cacr": False, "use_ce_aurc": False,
            "use_tent": False, "num_domains": 2,
            "vib_beta_content": 0.01, "vib_beta_edge": 0.0001,
            "vib_beta_scale": 0.02, "aug_sigma_style": 0.25,
            "aug_sigma_shift": 0.15, "aug_p_aug": 0.5, "aug_p_mix": 0.3,
        },
        "vib_warmup_fraction": 0.1,
        "lambda_content": 1.0, "lambda_edge": 0.1,
        "use_fda": False, "fda_L": None, "target_pool_max": 0,
        "grad_clip_max_norm": None,
        "optimizer": "AdamW(lr, weight_decay, betas=(0.9,0.999))  [= the vanilla DINOv3 optimiser]",
        "scheduler": "CosineWarmupScheduler(warmup_epochs, epochs_max, lr, min_lr)  [= vanilla]",
        "kl_term": ("task_loss + lambda_content*kl_content + lambda_edge*kl_edge; "
                    "the warmup factor is applied INSIDE the published SIB forward"),
    },
}


def method_target_access(arch, method):
    """(target_access, detail) BEFORE the cell runs. The manifest carries this."""
    if method == "fda":
        return ("unlabeled_target",
                "FDA input preprocessing draws its amplitude spectra from "
                "unlabelled IMAGES of the held-out city's TRAIN split only")
    if method == "sib":
        if SIB_SPEC[arch]["use_fda"]:
            return ("unlabeled_target",
                    "SIB on this CNN includes the FDA input-preprocessing "
                    "component (App H, erratum I-1), whose target pool is "
                    "unlabelled IMAGES of the held-out city's TRAIN split only")
        return ("source_only",
                "SIB on DINOv3 is Haar + content VIB + edge VIB. Per 5.1 and "
                "App H the ViT branch receives neither FDA nor SAG, so no "
                "held-out-city pixel is touched before test-time inference. "
                "App H's deployment-time coverage threshold is a post-hoc step "
                "over stored probabilities and is NOT applied by this run; see "
                "sib_coverage_threshold")
    raise ValueError(f"unknown method {method!r}")


def _published_fda_module():
    """`fda_transform.py` from the published tree, imported BY PATH.

    `python/mamnet/data/fda_transform.py` and `python/dinov3/data/fda_transform.py`
    are byte-identical (verified 2026-08-02), so one copy serves both
    architectures and neither published file is modified or duplicated.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, os.pardir, "mamnet", "data",
                                         "fda_transform.py"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"published fda_transform.py not found at {path}")
    spec = importlib.util.spec_from_file_location("st_fda_transform", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def split_patch_id(pid):
    """'JAX_652_24_y190_x165' -> ('JAX_652_24', 190, 165)."""
    head, x = pid.rsplit("_x", 1)
    image_id, y = head.rsplit("_y", 1)
    return image_id, int(y), int(x)


def build_fda_target_pool(budget_dir, holdout, patch_index, img_root, seed,
                          max_images, forbidden_ids, label):
    """<=`max_images` unlabelled held-out-city TRAIN patches, and its provenance.

    THE POOL RULE, which is stricter than the published SIB cell's
    ----------------------------------------------------------------
    The published standalone-FDA cells point `fda_target_root` at
    `{city}/{res}/train/images`, i.e. the held-out city's train split. The
    published SIB cells point it at `{city}/{res}` -- the city directory ROOT --
    and `load_target_images` walks it, so those runs' target pools could draw
    from val and test as well. Arm A uses the TRAIN-split rule for both
    methods, per the operator's I-1 discipline, and asserts it rather than
    trusting a path string.

    The ids come from the fold's own `{HOLDOUT}_upper.json` budget manifest --
    the same file that defines the fold's within-city upper bound, so the
    target pool is a subset of patches the project has already committed to as
    that city's train split, at the identical `budget_rule_sha`. Nothing new is
    sampled from disk.

    Selection is a DETERMINISTIC seeded choice over the sorted id list, not
    `np.random.choice` over an `os.walk` ordering as the published loader does.
    Declared deviation: it makes the pool reproducible from `run.json` alone.
    """
    man_path = Path(budget_dir) / f"{holdout}_upper.json"
    man = json.loads(man_path.read_text())
    pool_ids = sorted(man["train"])
    if not pool_ids:
        raise SystemExit(f"FATAL {label}: {man_path} carries no train ids")

    bad_city = [p for p in pool_ids if p.split("_", 1)[0] != holdout]
    if bad_city:
        raise SystemExit(
            f"FATAL {label}: {len(bad_city)} target-pool ids are not from the "
            f"held-out city {holdout} (first: {bad_city[:3]})")

    overlap = sorted(set(pool_ids) & set(forbidden_ids))
    if overlap:
        raise SystemExit(
            f"FATAL {label}: {len(overlap)} target-pool candidate ids are in "
            f"this cell's test/train/val sets (first: {overlap[:5]}). The FDA "
            f"target pool may only contain held-out-city TRAIN patches.")

    # hashlib, not the builtin hash(): str hashing is salted per process by
    # PYTHONHASHSEED, so hash() would give a different pool on every run and
    # "deterministic" would be a lie in the sidecar.
    key = f"fda_target_pool|{holdout}|{int(seed)}".encode()
    rng = np.random.RandomState(
        int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % (2 ** 31))
    take = min(int(max_images), len(pool_ids))
    sel = sorted(np.asarray(pool_ids)[rng.choice(len(pool_ids), take,
                                                 replace=False)].tolist())

    idx = patch_index if isinstance(patch_index, dict) else json.loads(
        Path(patch_index).read_text())
    images = idx["images"]
    from PIL import Image
    crops = []
    for pid in sel:
        image_id, y, x = split_patch_id(pid)
        rec = images.get(image_id)
        if rec is None:
            raise SystemExit(f"FATAL {label}: patch index has no image {image_id}")
        with Image.open(os.path.join(img_root, rec["image"])) as im:
            a = np.array(im.crop((x, y, x + 384, y + 384)))
        if a.ndim == 2:
            a = a[:, :, None]
        a = np.ascontiguousarray(a[:, :, :3])
        if a.shape[:2] != (384, 384):
            raise SystemExit(f"FATAL {label}: {pid} cropped to {a.shape[:2]}")
        if a.dtype != np.uint8:
            raise SystemExit(f"FATAL {label}: {pid} is {a.dtype}, expected uint8")
        crops.append(a)

    prov = {
        "fda_target_pool_n": len(crops),
        "fda_target_pool_split": "train",
        "fda_target_pool_city": holdout,
        "fda_target_pool_manifest": str(man_path),
        "fda_target_pool_budget_rule_sha": man["budget_rule_sha"],
        "fda_target_pool_candidates": len(pool_ids),
        "fda_target_pool_max": int(max_images),
        "fda_target_pool_ids_sha256": hashlib.sha256(
            "\n".join(sel).encode()).hexdigest(),
        "fda_target_pool_ids_first5": sel[:5],
        "fda_target_pool_selection": (
            "deterministic seeded choice over the sorted id list of the fold's "
            "own {HOLDOUT}_upper.json train split; DECLARED DEVIATION from the "
            "published load_target_images(), which np.random.choice()s over an "
            "os.walk() ordering and is therefore not reproducible from the "
            "sidecar"),
        "fda_target_pool_disjoint_from_cell_splits": True,
        "fda_target_pool_labels_used": False,
        "fda_target_pool_note": (
            "IMAGES ONLY. No mask, no label and no val/test patch of the "
            "held-out city enters training. The assertion above is over the "
            "cell's own train, val and test id sets."),
    }
    return crops, sel, prov


class SEODatasetFDA(SEODataset):
    """`SEODataset` with the published FDA transform on the uint8 RGB crop.

    Insertion point is `_to_tensor`, which receives the raw uint8 HWC crop
    straight out of `_crop` -- so FDA runs BEFORE the RMS-contrast channel,
    BEFORE ImageNet normalisation and BEFORE the flip augmentation, which is
    the order `mamnet/data/dataset_enhanced.py` applies for the published
    MAMNet+FDA cells (FDA at __getitem__ line ~200, contrast at line ~246).

    `python/mamnet/data/dataset_sib.py` computes its contrast channel from the
    pre-FDA raw image instead, so the published SIB cells' 4th channel does not
    see the amplitude swap. That is an inconsistency inside the published tree;
    arm A follows the FDA-method order for both methods and records it here.
    """

    def __init__(self, *a, fda=None, **kw):
        self._fda = fda
        self._fda_calls = 0
        super().__init__(*a, **kw)

    def _to_tensor(self, arr):
        if self._fda is not None:
            arr = np.ascontiguousarray(
                self._fda(np.ascontiguousarray(arr[:, :, :3])))
            self._fda_calls += 1
        return super()._to_tensor(arr)


# ===========================================================================
# SIB: the published model, optimiser, scheduler, clipping and KL term,
# installed into the phase-2 code path without editing a published file.
# ===========================================================================

def intensity_map_from_input(x):
    """[B,1,H,W] luminance in [0,1], recovered from the normalised input.

    `python/{arch}/data/dataset_sib.py` builds this in the LOADER, from the raw
    PIL image, as 0.299R + 0.587G + 0.114B over [0,1]. S-EO's loader carries no
    such key, and adding one would edit `seo_dataset.py`, whose sha the phase-1
    and phase-2 cells record. It is instead recovered here by inverting the
    ImageNet normalisation on the first three channels of the tensor the model
    is about to see -- the same luminance, over the same [0,1] range.

    One difference, in this version's favour: the published loader computes the
    map from the UNAUGMENTED image while the tensor it ships is flipped, so a
    flipped published sample carries a misaligned intensity map. Recovering the
    map from the input tensor keeps the two aligned.

    Both C4-clean configs set `adaptive_beta: false`, so this map reaches an
    inactive branch of the published SIB forward and changes no number. It is
    computed and passed anyway, so the code path is the published one.
    """
    import torch
    rgb = x[:, :3]
    mean = torch.tensor(IMAGENET_MEAN, device=x.device,
                        dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device,
                       dtype=x.dtype).view(1, 3, 1, 1)
    rgb = rgb * std + mean
    lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    return lum.clamp(0.0, 1.0)


def vib_warmup_weight(epoch, total_epochs, warmup_fraction=0.1):
    """LIFTED VERBATIM from `mamnet/train_mamnet_sib.py:443`."""
    warmup_epochs = max(1, int(total_epochs * warmup_fraction))
    if epoch < warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    return 1.0


def dinov3_vib_warmup_factor(epoch, warmup_epochs, use_vib=True):
    """LIFTED VERBATIM from `train_dinov3_sib.py:1108` (`_get_vib_warmup_factor`).

    Note the boundary differs from MAMNet's (`<=` here, `<` there) and that the
    two scripts derive `warmup_epochs` differently (`int(frac*epochs)` here,
    `max(1, int(frac*epochs))` there). Both are carried as written.
    """
    if warmup_epochs <= 0 or not use_vib:
        return 1.0
    if epoch <= warmup_epochs:
        return float(epoch) / float(warmup_epochs)
    return 1.0


class _PlateauSignatureScheduler:
    """A cosine scheduler behind `ReduceLROnPlateau`'s call signature.

    The published MAMNet+SIB script schedules with
    `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`; the published vanilla
    MAMNet Trainer -- whose `train()` loop this arm runs -- calls
    `self.scheduler.step(decision_miou)`. Passing a metric to
    `CosineAnnealingLR.step` would be read as an epoch index. This adapter
    swallows the metric and forwards state_dict/load_state_dict so a resumed
    link restores the schedule.
    """

    def __init__(self, inner):
        self.inner = inner

    def step(self, *_a, **_k):
        return self.inner.step()

    def state_dict(self):
        return self.inner.state_dict()

    def load_state_dict(self, sd):
        return self.inner.load_state_dict(sd)

    def get_last_lr(self):
        return self.inner.get_last_lr()

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _MamnetSIBLoss:
    """`MAMNetLoss` + `vib_w * kl_total`, per `train_mamnet_sib.py:543`.

    Delegates every other attribute to the wrapped loss, so
    `trainer.criterion.criterion` -- which the published `validate()` calls
    directly, and which arm A has already replaced with
    `CrossEntropyLoss(ignore_index=255)` -- keeps resolving to the real inner
    CE rather than to a stale copy.
    """

    def __init__(self, inner, state, total_epochs, warmup_fraction):
        self.inner = inner
        self.state = state
        self.total_epochs = total_epochs
        self.warmup_fraction = warmup_fraction

    def __call__(self, outputs, targets):
        import torch
        d = self.inner(outputs, targets)
        kl = (self.state.get("sib_losses") or {}).get("kl_total")
        if self.state.get("training") and isinstance(kl, torch.Tensor):
            w = vib_warmup_weight(self.state.get("epoch", 1), self.total_epochs,
                                  self.warmup_fraction)
            d = dict(d)
            d["kl_total"] = kl.detach()
            d["vib_warmup"] = w
            d["total"] = d["total"] + w * kl
            self.state["last_kl"] = float(kl.detach().item())
            self.state["last_warmup"] = w
            self.state["kl_applied"] = self.state.get("kl_applied", 0) + 1
        return d

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _Dinov3SIBLoss:
    """CE + `lambda_content*kl_content + lambda_edge*kl_edge`, per
    `train_dinov3_sib.py:1154`. The warmup factor is already inside the KLs:
    the published DINOv3 SIB forward multiplies them by `vib_warmup_factor`.
    """

    def __init__(self, inner, state, lambda_content, lambda_edge):
        self.inner = inner
        self.state = state
        self.lambda_content = lambda_content
        self.lambda_edge = lambda_edge

    def __call__(self, outputs, targets):
        import torch
        loss = self.inner(outputs, targets)
        sl = self.state.get("sib_losses") or {}
        if not self.state.get("training"):
            return loss
        kc = sl.get("kl_content", sl.get("kl_uniform"))
        ke = sl.get("kl_edge")
        add = 0.0
        if isinstance(kc, torch.Tensor):
            add = add + self.lambda_content * kc
        if isinstance(ke, torch.Tensor):
            add = add + self.lambda_edge * ke
        if isinstance(add, torch.Tensor):
            loss = loss + add
            self.state["last_kl"] = float(add.detach().item())
            self.state["kl_applied"] = self.state.get("kl_applied", 0) + 1
        return loss

    def __getattr__(self, name):
        return getattr(self.inner, name)


def install_sib(trainer, arch, epochs_max, verbose=True):
    """Replace the vanilla model with the published SIB model, in place.

    Returns a provenance dict for `run.json`. Everything it touches is an
    attribute of the constructed Trainer; no published file is edited.

    Order matters and is asserted by the caller: the ignore_index loss is
    installed FIRST (phase 2's `set_ignore_index_loss`), then wrapped here, so
    the KL rides on top of a loss that already drops unseen pixels.
    """
    import torch
    import torch.optim as optim

    spec = SIB_SPEC[arch]
    a = trainer.args
    device = trainer.device
    state = {"epoch": 1, "training": False, "sib_losses": None,
             "kl_applied": 0, "last_kl": None, "last_warmup": None,
             "forward_calls": 0}

    if arch == "mamnet":
        from argparse import Namespace
        from models.mamnet_sib import build_mamnet_sib
        model = build_mamnet_sib(Namespace(**spec["flags"])).to(device)
    elif arch == "dinov3":
        from dinov3_model_sib import DINOv3ShadowDetectorSIB
        ctor = dict(spec["flags"])
        ctor.update(num_classes=getattr(a, "num_classes", 2),
                    model_name=a.model_name, weights_path=a.weights_path,
                    pretrained=bool(getattr(a, "pretrained", True)),
                    frozen_stages=int(getattr(a, "frozen_stages", -1)))
        model = DINOv3ShadowDetectorSIB(**ctor).to(device)
    else:
        raise SystemExit(f"FATAL no SIB route for arch {arch}")

    n_par = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.model = model

    # ---- optimiser and scheduler, as the published SIB script builds them ---
    if arch == "mamnet":
        trainer.optimizer = optim.Adam(model.parameters(), lr=a.lr,
                                       weight_decay=1e-4)
        trainer.scheduler = _PlateauSignatureScheduler(
            optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer,
                                                 T_max=epochs_max, eta_min=1e-6))
    else:
        trainer.optimizer = optim.AdamW(model.parameters(), lr=a.lr,
                                        weight_decay=a.weight_decay,
                                        betas=(0.9, 0.999))
        sched_cls = type(trainer.scheduler)
        trainer.scheduler = sched_cls(trainer.optimizer,
                                      warmup_epochs=a.warmup_epochs,
                                      total_epochs=epochs_max,
                                      base_lr=a.lr, min_lr=a.min_lr)

    # ---- gradient clipping (MAMNet+SIB only; train_mamnet_sib.py:549) -------
    clip = spec["grad_clip_max_norm"]
    if clip:
        real_step = trainer.optimizer.step

        def clipped_step(*ar, **kw):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            return real_step(*ar, **kw)

        trainer.optimizer.step = clipped_step

    # ---- forward: strip the (outputs, sib_losses) tuple ---------------------
    # Patched on the INSTANCE, not on the class and not by subclassing, so
    # `model.state_dict()` keys are unchanged and a link resumes from a
    # checkpoint written by any other link.
    real_forward = model.forward
    warmup_epochs_dinov3 = int(spec["vib_warmup_fraction"] * epochs_max)

    if arch == "mamnet":
        def fwd(x, *ar, **kw):
            out, losses = real_forward(x, intensity_map=intensity_map_from_input(x))
            state["sib_losses"] = losses
            state["training"] = bool(model.training)
            state["forward_calls"] += 1
            return out
    else:
        def fwd(x, *ar, **kw):
            wf = dinov3_vib_warmup_factor(state.get("epoch", 1),
                                          warmup_epochs_dinov3,
                                          use_vib=spec["flags"]["use_vib"])
            out, losses = real_forward(x,
                                       intensity_map=intensity_map_from_input(x),
                                       city_ids=None, vib_warmup_factor=wf)
            state["sib_losses"] = losses
            state["training"] = bool(model.training)
            state["last_warmup"] = wf
            state["forward_calls"] += 1
            return out

    model.forward = fwd

    # ---- loss: add the KL term ---------------------------------------------
    if arch == "mamnet":
        trainer.criterion = _MamnetSIBLoss(trainer.criterion, state, epochs_max,
                                           spec["vib_warmup_fraction"])
    else:
        trainer.criterion = _Dinov3SIBLoss(trainer.criterion, state,
                                           spec["lambda_content"],
                                           spec["lambda_edge"])

    # ---- epoch tracking, for the KL warmup schedule -------------------------
    real_train_epoch = trainer.train_epoch

    def tracked_train_epoch(epoch, *ar, **kw):
        state["epoch"] = int(epoch)
        return real_train_epoch(epoch, *ar, **kw)

    trainer.train_epoch = tracked_train_epoch

    prov = {
        "method": "sib",
        "sib_variant": "C4-clean (the published SIB)",
        "sib_published_config": spec["published_config"],
        "sib_components_active": spec["components"],
        "sib_components_off": spec["components_off"],
        "sib_flags": spec["flags"],
        "sib_vib_warmup_fraction": spec["vib_warmup_fraction"],
        "sib_vib_warmup_epochs": (warmup_epochs_dinov3 if arch == "dinov3"
                                  else max(1, int(spec["vib_warmup_fraction"]
                                                  * epochs_max))),
        "sib_kl_term": spec["kl_term"],
        "sib_optimizer": spec["optimizer"],
        "sib_scheduler": spec["scheduler"],
        "sib_grad_clip_max_norm": clip,
        "sib_lambda_content": spec.get("lambda_content"),
        "sib_lambda_edge": spec.get("lambda_edge"),
        "sib_n_params": n_par,
        "sib_n_params_trainable": n_tr,
        "sib_intensity_map_source": (
            "recovered from the normalised input by inverting the ImageNet "
            "transform on channels 0-2, not carried by the loader; INERT "
            "because adaptive_beta is false in the published C4-clean config"),
        "sib_coverage_threshold": (
            "NOT applied. App H's deployment-time coverage threshold c*_val is "
            "absent from both published SIB training scripts -- it is a "
            "post-hoc step over stored probabilities in "
            "final_loco/composition_analysis/. This cell trains and dumps at "
            "full coverage, as its vanilla comparator does, and probs.npz is "
            "unthresholded so the threshold is recomputable without a GPU."),
        "sib_appendix_h_disagreements": [
            "App H defines an intensity-conditioned beta_content(x); both "
            "published C4-clean configs set adaptive_beta=false, so no "
            "published SIB cell used it",
            "Table 37 lists beta 0.01/1e-4 for every architecture; MAMNet's "
            "published config is 1e-3/1e-5. Per-arch config values are used",
            "Table 37 lists AdamW wd 0.01 and split DINOv3 encoder/head LRs; "
            "the published trainers build Adam(wd=1e-4) for MAMNet and a "
            "single-LR AdamW(wd=0.05) for DINOv3. The trainers are followed",
        ],
        "sib_state": state,
    }
    if verbose:
        print(f"[armA] SIB installed on {arch}: active {spec['components']}, "
              f"off {spec['components_off']}", flush=True)
        print(f"[armA]   params {n_par:,} ({n_tr:,} trainable)  "
              f"optimizer {spec['optimizer']}  scheduler {spec['scheduler']}",
              flush=True)
        print(f"[armA]   KL: {spec['kl_term']}  "
              f"warmup_fraction {spec['vib_warmup_fraction']}  "
              f"grad_clip {clip}", flush=True)
    return prov, state
