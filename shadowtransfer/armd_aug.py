"""
Arm D — the S3 appearance-augmentation pipeline.

Answers v1UJ-W4 and zNBu-W2: the paper's feature-space claim rests mainly on
histogram matching, and a reviewer asks whether other low-level appearance or
illumination factors explain the LOCO gap. Arm D retrains under aggressive
appearance-only augmentation and asks whether the gap moves.

GEOMETRY IS STRICTLY PRESERVED. Every operation below is a per-pixel or
per-channel intensity map, or an isotropic blur. No crop, no rotation, no
cutout, no cross-image mixing. The published H-flip/V-flip p=0.5 stays where it
is, downstream of this module, and is the only geometric augmentation any arm D
cell applies.

THE ORDERING CONSTRAINT — the whole arm turns on it
---------------------------------------------------
MAMNet and OGLANet consume a 4-channel RGBC tensor whose channel 3 is the local
RMS contrast of the image (mamnet/data/contrast_utils.py). That channel is
computed inside the published dataset's __getitem__, from the PIL RGB, at a
fixed point in the pipeline. If the appearance augmentation were applied after
that point, channel 3 would carry clean, un-augmented intensity statistics
straight into the network and the arm would be a no-op on both CNNs.

This module is therefore installed on the dataset's `fda_transform` hook, which
all three published datasets call on the raw PIL RGB *before* the flip
augmentation and *before* add_contrast_channel:

    image = Image.open(...).convert('RGB')
    if self.fda_transform is not None:          <-- S3 lands here
        image = Image.fromarray(self.fda_transform(np.array(image)))
    ... flips ...
    image_rgbc = add_contrast_channel(image)    <-- contrast sees augmented RGB
    ... ToTensor + Normalize ...

verified in mamnet/data/dataset_enhanced.py, oglanet/data/dataset_enhanced.py
and dinov3/data/dataset.py on 2026-07-30. The published files are NOT edited.

THE HOOK NAME IS NOT FDA. `fda_transform` is the attribute the published
loaders happen to call at the one point in the pipeline arm D needs. Arm D sets
`use_fda=False` everywhere, constructs no FDA transform, and reads no
held-out-city image. A target-access audit (BxRx-W4, erratum I-1) should read
this as source-only on the augmentation axis; the hook name is an artifact of
the published loader, not a statement about data access.

Randomness
----------
Every draw comes from numpy's global RNG, which armc/armd_train seed inside each
DataLoader worker via worker_init_fn. That is the same RNG the published
augmentations draw from, so arm D inherits the established reproducibility
story rather than inventing a second one.
"""

import numpy as np
from PIL import Image

# --- the recipe, exactly as the arm spec states it --------------------------
# Recorded verbatim into every cell's run.json as `aug_params`, so a reader
# never has to trust that the code and the spec agree -- they can diff them.
S3 = {
    "strength": "S3",
    "color_jitter": {"brightness": 0.6, "contrast": 0.6,
                     "saturation": 0.6, "hue": 0.10},
    "gamma": {"low": 0.5, "high": 2.0},
    "per_channel_gain": {"low": 0.8, "high": 1.25,
                         "independent_per_channel": True},
    "random_grayscale": {"p": 0.2},
    "gaussian_blur": {"sigma_low": 0.1, "sigma_high": 2.0, "p": 0.5,
                      "kernel_size": 9},
    "gaussian_noise": {"sigma": 0.04, "scale": "[0,1]", "p": 0.5},
    "order": ["color_jitter", "gamma", "per_channel_gain", "random_grayscale",
              "gaussian_blur", "gaussian_noise",
              "<published H/V flip p=0.5>", "<contrast channel>",
              "<ImageNet normalisation>"],
    "geometry_preserved": True,
    "applied_to": "train split only; val and test are never augmented",
}

_GRAY_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


class S3Appearance:
    """numpy uint8 HWC RGB -> numpy uint8 HWC RGB.

    Installed on a published dataset's `fda_transform` hook. Signature and
    dtypes match what that hook is called with, so the published __getitem__
    needs no change.
    """

    def __init__(self, params=None):
        self.p = params or S3
        # torchvision's ColorJitter is the published-stack implementation of
        # brightness/contrast/saturation/hue and draws from torch's RNG, which
        # is seeded per worker alongside numpy's.
        from torchvision import transforms as T
        cj = self.p["color_jitter"]
        self._jitter = T.ColorJitter(brightness=cj["brightness"],
                                     contrast=cj["contrast"],
                                     saturation=cj["saturation"],
                                     hue=cj["hue"])
        gb = self.p["gaussian_blur"]
        self._blur = T.GaussianBlur(kernel_size=gb["kernel_size"],
                                    sigma=(gb["sigma_low"], gb["sigma_high"]))

    def __repr__(self):
        return ("S3Appearance(appearance-only, geometry-preserving; "
                "NOT Fourier Domain Adaptation; touches no held-out-city image)")

    def __call__(self, img_np):
        if img_np.dtype != np.uint8:
            raise ValueError(f"S3Appearance expects uint8, got {img_np.dtype}")
        if img_np.ndim != 3 or img_np.shape[2] != 3:
            raise ValueError(f"S3Appearance expects HWC RGB, got {img_np.shape}")

        # 1. ColorJitter, on PIL, as published code does it
        img = self._jitter(Image.fromarray(img_np))

        x = np.asarray(img, dtype=np.float32) / 255.0

        # 2. Gamma
        g = self.p["gamma"]
        gamma = np.random.uniform(g["low"], g["high"])
        x = np.clip(x, 0.0, 1.0) ** gamma

        # 3. Per-channel gain, drawn independently per channel
        pg = self.p["per_channel_gain"]
        gain = np.random.uniform(pg["low"], pg["high"], size=3).astype(np.float32)
        x = x * gain[None, None, :]

        # 4. Random grayscale (kept 3-channel; this is an appearance op, not a
        #    channel-count change -- the network's input shape is fixed)
        if np.random.rand() < self.p["random_grayscale"]["p"]:
            lum = (x * _GRAY_W[None, None, :]).sum(axis=2, keepdims=True)
            x = np.repeat(lum, 3, axis=2)

        # 5. Gaussian blur, isotropic, geometry-preserving
        if np.random.rand() < self.p["gaussian_blur"]["p"]:
            blurred = self._blur(Image.fromarray(
                (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)))
            x = np.asarray(blurred, dtype=np.float32) / 255.0

        # 6. Additive Gaussian noise on the [0,1] scale
        gn = self.p["gaussian_noise"]
        if np.random.rand() < gn["p"]:
            x = x + np.random.normal(0.0, gn["sigma"], size=x.shape)

        return (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# The contrast-channel verification the arm spec requires in the log
# ---------------------------------------------------------------------------

def contrast_channel_check(build_dataset, arch, pairs, res_dir, img_size,
                           n_batch=8, seed=0):
    """Build the SAME training dataset twice, augmentation off and on, pull one
    batch from each, and report per-channel mean/std of the input tensor.

    For mamnet and oglanet the input is RGBC and channel 3 is the contrast
    channel. If channel 3 is identical with augmentation on and off, the
    augmentation is landing after the contrast computation and the arm is a
    no-op on that architecture -- this check exists to catch exactly that, in
    the log, before any GPU time is spent.

    Returns a dict; the caller prints it and stores it in run.json.
    """
    import torch
    from torch.utils.data import DataLoader

    out = {"arch": arch, "n_images_compared": int(n_batch)}

    def _stats(augment_on):
        torch.manual_seed(seed)
        np.random.seed(seed)
        ds = build_dataset(arch, pairs[:n_batch], "train", res_dir, img_size, True)
        if augment_on:
            if ds.fda_transform is not None:
                raise SystemExit("FATAL fda_transform already set before injection")
            ds.fda_transform = S3Appearance()
        dl = DataLoader(ds, batch_size=n_batch, shuffle=False, num_workers=0)
        batch = next(iter(dl))
        t = batch["image"]
        return t, {"mean": [round(float(v), 6) for v in t.mean(dim=(0, 2, 3))],
                   "std": [round(float(v), 6) for v in t.std(dim=(0, 2, 3))]}

    t_off, s_off = _stats(False)
    t_on, s_on = _stats(True)

    out["n_channels"] = int(t_off.shape[1])
    out["aug_off"] = s_off
    out["aug_on"] = s_on
    out["per_channel_max_abs_delta"] = [
        round(float((t_on[:, c] - t_off[:, c]).abs().max()), 6)
        for c in range(t_off.shape[1])]

    if t_off.shape[1] == 4:
        d3 = float((t_on[:, 3] - t_off[:, 3]).abs().max())
        out["contrast_channel_index"] = 3
        out["contrast_channel_max_abs_delta"] = round(d3, 6)
        out["contrast_channel_differs"] = bool(d3 > 1e-6)
        out["verdict"] = ("PASS: contrast channel is computed from the "
                          "augmented RGB" if d3 > 1e-6 else
                          "FAIL: contrast channel is identical with "
                          "augmentation on and off")
    else:
        out["contrast_channel_index"] = None
        out["contrast_channel_differs"] = None
        out["verdict"] = ("n/a: 3-channel RGB input, no contrast channel "
                          "(dinov3)")
    return out


def assert_contrast_check(chk):
    """Hard-fail a CNN cell whose contrast channel did not move."""
    if chk.get("contrast_channel_index") == 3 and not chk["contrast_channel_differs"]:
        raise SystemExit(
            f"FATAL {chk['arch']}: contrast channel identical with augmentation "
            f"on and off -- augmentation is landing after add_contrast_channel "
            f"and this cell would be a no-op. Refusing to train.")
