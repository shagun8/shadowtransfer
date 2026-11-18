"""
Arm C §4.2-4.4 probes — the three probe operations.

  probe 1  §4.2 test-time histogram matching, source-city statistics only
  probe 2  §4.4 city-identity linear probe on encoder features
  probe 3  §4.4 component retraining, Experiments A and B

Histogram matching: two definitions, both reported
--------------------------------------------------
The arm spec asks for PER-CHANNEL CDF matching. The published §4.2 experiment
(`python/final_loco/experiment_c_histogram_match.py` via
`experiment_utils.compute_histogram` / `histogram_match_image`) matches a
single GRAYSCALE CDF and re-applies each pixel's per-channel ratio to it. The
two are not the same intervention, and the published reading being compared
against — 0 of 9 cells improved at rung 450 — was measured under the grayscale
one.

Both run. The per-channel version is the headline (`iou_tol_matched`,
`recovery_ratio_R`); the grayscale version rides along
(`iou_tol_matched_gray`, `recovery_ratio_R_gray`) at the cost of one extra
forward pass over 150 images, so a rung-900 result can be read against the
published finding without the matching rule confounding it.

The mapping rule `argmin_s |cdf_src[s] - cdf_tgt[t]|` is the published one in
both variants; the per-channel version only applies it per channel instead of
to grayscale.

Where the match is injected
---------------------------
Into `dataset.fda_transform`, the hook all three architectures' datasets apply
to the raw uint8 RGB array immediately after `Image.open`, before any resize,
normalisation or contrast-channel derivation. That is the only injection point
where matched pixels then travel through the published preprocessing unchanged
-- matching a normalised tensor, as the published Experiment C wrapper does,
would apply an intensity map to ImageNet-standardised values.
"""

import time

import numpy as np
from PIL import Image

from shadowtransfer import armc_probes_common as C


# ---------------------------------------------------------------------------
# Probe 1 — histogram matching
# ---------------------------------------------------------------------------

def channel_cdfs(paths):
    """Aggregate per-channel CDF over a list of image paths. (3, 256)."""
    hist = np.zeros((3, 256), dtype=np.float64)
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        for c in range(3):
            hist[c] += np.bincount(arr[:, :, c].ravel(), minlength=256)
    tot = hist.sum(axis=1, keepdims=True)
    if (tot == 0).any():
        raise SystemExit("FATAL empty channel histogram over the source pool")
    return np.cumsum(hist / tot, axis=1)


def gray_cdf(paths):
    """Aggregate grayscale CDF, the published §4.2 statistic.

    `experiment_utils.compute_histogram` takes the uint8 cast of the RGB mean
    and bins it; that cast (truncation, not rounding) is kept.
    """
    hist = np.zeros(256, dtype=np.float64)
    for p in paths:
        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        gray = arr.mean(axis=2).astype(np.uint8)
        hist += np.bincount(gray.ravel(), minlength=256)
    return np.cumsum(hist / hist.sum())


def _mapping(src_cdf, tgt_cdf):
    """mapping[t] = argmin_s |src_cdf[s] - tgt_cdf[t]|, the published rule."""
    return np.abs(src_cdf[None, :] - tgt_cdf[:, None]).argmin(axis=1).astype(np.uint8)


def match_per_channel(img, src_cdf):
    """Per-channel CDF match of one uint8 (H, W, 3) image toward src_cdf."""
    out = np.empty_like(img)
    for c in range(3):
        h = np.bincount(img[:, :, c].ravel(), minlength=256).astype(np.float64)
        out[:, :, c] = _mapping(src_cdf[c], np.cumsum(h / h.sum()))[img[:, :, c]]
    return out


def match_gray(img, src_gray_cdf):
    """Grayscale match with per-pixel channel ratio preserved.

    Verbatim behaviour of experiment_utils.histogram_match_image, with its
    per-image target CDF computed here rather than passed in.
    """
    gray = img.mean(axis=2)
    h = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    mapping = _mapping(src_gray_cdf, np.cumsum(h / h.sum()))

    matched = np.zeros_like(img)
    gray_f = gray.astype(np.float32)
    mapped_gray = mapping[gray.astype(np.uint8).ravel()].reshape(
        gray.shape).astype(np.float32)
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        ratio = np.where(gray_f > 0, channel / (gray_f + 1e-6), 1.0)
        matched[:, :, c] = np.clip(mapped_gray * ratio, 0, 255).astype(np.uint8)
    return matched


def source_cdfs(train_pairs, res_dir, base):
    """(per-channel CDF, grayscale CDF, n_images) over a cell's own training
    file list -- the rung's images, not the city's whole pool. The histogram a
    rung-225 cell could have computed is not the one a rung-900 cell could."""
    paths = [base / "data" / "Final_data_test" / c / res_dir / "train" / "images" / f
             for c, f in train_pairs]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"FATAL source image missing: {p}")
    t0 = time.time()
    ch = channel_cdfs(paths)
    gy = gray_cdf(paths)
    print(f"  source histogram over {len(paths)} training images "
          f"in {time.time() - t0:.1f}s", flush=True)
    return ch, gy, len(paths)


# ---------------------------------------------------------------------------
# Probe 2 — city-identity linear probe
# ---------------------------------------------------------------------------
#
# Spatial-cell subsampling is a compute decision, stated rather than hidden.
# The encoder's final map is 24x24 = 576 cells per image and the probe sees
# 450 images, so "every spatial cell" is 259,200 samples; the permutation null
# refits 5 folds x 100 shuffles = 505 logistic regressions, which at full
# density is tens of TFLOPs on CPU. A seeded uniform sample of
# `cells_per_image` cells is drawn per image instead. The sample is drawn ONCE
# and reused for the observed fit and for every permutation, so the null is
# calibrated against exactly the data the observed accuracy was measured on.
# n_spatial_cells_total and n_spatial_cells_sampled both go into run.json.

def extract_features(model, arch, loader, device, cells_per_image, seed):
    """Encoder features at sampled spatial cells, plus per-image bookkeeping.

    Returns (X [n_img*S, D], image_ids, feat_shape, n_cells_total).
    """
    import torch
    model.eval()
    rng = np.random.default_rng(seed)
    feats, ids, shape = [], [], None
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            f = C.encoder_forward(model, arch, images)          # [B, D, h, w]
            B, D, h, w = f.shape
            shape = (D, h, w)
            f = f.permute(0, 2, 3, 1).reshape(B, h * w, D).cpu().numpy()
            for b in range(B):
                idx = rng.choice(h * w, size=min(cells_per_image, h * w),
                                 replace=False)
                feats.append(f[b, idx].astype(np.float32))
                ids.append(str(batch["filename"][b]).rsplit(".", 1)[0])
    X = np.concatenate(feats, axis=0)
    return X, ids, shape, shape[1] * shape[2]


def city_identity_probe(X, img_city, cells_per_image, n_folds=5,
                        n_perm=100, seed=0, max_iter=200):
    """3-way logistic regression on encoder features, image-level GroupKFold.

    `img_city` is one integer city label per image, in the order the features
    were stacked. Groups are images: a pixel-level split leaks, because two
    cells of the same image are near-duplicates and land on both sides of the
    fold boundary.

    The permutation null shuffles labels at IMAGE level and refits with the
    same fold structure, so it destroys the city signal without destroying the
    within-image correlation that makes the fold structure necessary.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n_img = len(img_city)
    groups = np.repeat(np.arange(n_img), cells_per_image)
    img_city = np.asarray(img_city)
    splits = list(GroupKFold(n_splits=n_folds).split(X, np.repeat(img_city, cells_per_image), groups))

    def run(labels_img):
        y = np.repeat(labels_img, cells_per_image)
        pred = np.empty(len(y), dtype=np.int64)
        for tr, te in splits:
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=max_iter, n_jobs=1)
            clf.fit(sc.transform(X[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(X[te]))
        return float((pred == y).mean()), pred

    t0 = time.time()
    acc, pred = run(img_city)
    print(f"  observed probe accuracy {acc:.4f} in {time.time() - t0:.1f}s",
          flush=True)

    rng = np.random.default_rng(seed + 991)
    null = []
    t0 = time.time()
    for i in range(n_perm):
        a, _ = run(rng.permutation(img_city))
        null.append(a)
        if (i + 1) % 20 == 0:
            print(f"  permutation {i + 1}/{n_perm}  mean so far "
                  f"{np.mean(null):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    null = np.asarray(null)
    has_null = null.size > 0

    # One-sided: the hypothesis is that city identity is DECODABLE, so only
    # nulls at or above the observed accuracy count against it. The +1 on both
    # sides is the standard finite-permutation correction; p is never 0.
    #
    # n_perm=0 is a real call shape, not a misuse: arm L's raw-pixel control
    # (armL_train.py probe_p2) requests it deliberately, to get the control's
    # OWN accuracy without repeating a 100-refit permutation null already
    # computed on the feature probe over the same images. With an empty null,
    # p_value and every null_* statistic are undefined rather than degenerate
    # -- NaN, not a crash (found 2026-08-01, job 20714436: np.percentile on a
    # zero-size array raised IndexError; null.min()/.max() would have failed
    # the same way one line later). This guard changes nothing for any
    # n_perm >= 1 caller (arm C, arm D, arm F, and arm L's own feature probe):
    # every formula below is unchanged on that path.
    p = (float((1 + int((null >= acc).sum())) / (1 + n_perm))
         if has_null else float("nan"))
    y = np.repeat(img_city, cells_per_image)
    per_img = (pred == y).reshape(n_img, cells_per_image).mean(axis=1)
    return {
        "probe_accuracy": acc,
        "null_mean": float(null.mean()) if has_null else float("nan"),
        "null_p99": float(np.percentile(null, 99)) if has_null else float("nan"),
        "null_min": float(null.min()) if has_null else float("nan"),
        "null_max": float(null.max()) if has_null else float("nan"),
        "p_value": p,
        "n_folds": n_folds,
        "n_permutations": n_perm,
        "chance": float(1.0 / len(np.unique(img_city))),
        "per_image_accuracy": per_img,
        "fold_sizes": [int(len(te)) for _, te in splits],
    }


# ---------------------------------------------------------------------------
# Probe 3 — component retraining (§4.4 Experiments A and B)
# ---------------------------------------------------------------------------
#
# The optimiser, schedule, loss and BN handling are the published §4.4 ones,
# lifted from experiment_a_decoder_retrain.train_decoder and
# experiment_b2_encoder_retrain.train_encoder: AdamW(weight_decay=1e-4),
# CosineAnnealingLR(T_max=epochs), CrossEntropyLoss(ignore_index=255), and the
# frozen component's BatchNorm held in eval so its running statistics do not
# drift on target-city batches. Table 36 came from that recipe; a probe that
# optimises differently is not comparable to it.
#
# Neither published experiment validates or selects a checkpoint: both train a
# fixed 30 epochs and evaluate the final weights. That is kept. `steps_to_best`
# is therefore reported against the minimum-mean-training-loss epoch and
# labelled as such -- it is a training-dynamics fact here, not a selection
# rule, and run.json records evaluated_checkpoint="final".

def retrain_component(model, arch, loader, device, frozen, epochs, lr,
                      weight_decay=1e-4, log_interval=20):
    """Train one component with the other frozen. Returns a stats dict."""
    import torch
    import torch.nn as nn

    prefixes = C.ENCODER_PREFIXES.get(arch, ["encoder.", "backbone."])
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  training {len(trainable)} tensors / {n_trainable:,} params "
          f"at lr={lr:g}, {frozen} frozen", flush=True)
    if n_trainable == 0:
        raise SystemExit("FATAL nothing is trainable; the freeze took everything")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    n_bn_eval = 0
    for name, module in model.named_modules():
        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            continue
        is_encoder = any(name.startswith(p) for p in prefixes)
        if (frozen == "encoder" and is_encoder) or \
           (frozen == "decoder" and not is_encoder):
            module.eval()
            n_bn_eval += 1
    print(f"  {n_bn_eval} BatchNorm layers of the frozen component held in eval",
          flush=True)

    losses = []
    steps_per_epoch = len(loader)
    for epoch in range(epochs):
        running, n_batches = 0.0, 0
        for i, batch in enumerate(loader):
            images = batch["image"].to(device)
            labels = batch["mask"].to(device)
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype == torch.float32:
                labels = (labels > 0.5).long()

            optimizer.zero_grad()
            logits = C.extract_logits(model(images))
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear",
                    align_corners=False)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1
            if (i + 1) % log_interval == 0:
                print(f"    epoch {epoch + 1}/{epochs} batch {i + 1} "
                      f"loss {running / n_batches:.4f}", flush=True)

        epoch_loss = running / max(n_batches, 1)
        scheduler.step()
        losses.append(epoch_loss)
        print(f"  epoch {epoch + 1}/{epochs}: loss={epoch_loss:.4f} "
              f"best={min(losses):.4f} lr={scheduler.get_last_lr()[0]:.3e}",
              flush=True)

    best_epoch = int(np.argmin(losses)) + 1
    return {
        "epochs_run": epochs,
        "steps_per_epoch": steps_per_epoch,
        "steps_to_best": best_epoch * steps_per_epoch,
        "best_epoch": best_epoch,
        "best_train_loss": float(min(losses)),
        "final_train_loss": float(losses[-1]),
        "train_loss_per_epoch": [round(float(x), 6) for x in losses],
        "n_trainable_params": int(n_trainable),
        "n_bn_layers_held_eval": n_bn_eval,
        "optimizer": f"AdamW(lr={lr:g}, weight_decay={weight_decay:g})",
        "scheduler": f"CosineAnnealingLR(T_max={epochs})",
        "loss": "CrossEntropyLoss(ignore_index=255)",
    }


# ---------------------------------------------------------------------------
# Dataset-hook adapters
# ---------------------------------------------------------------------------
#
# Classes rather than closures: a DataLoader worker receives its own copy of
# the dataset, and a top-level class stays picklable if the start method ever
# stops being fork.

class PerChannelMatcher:
    """dataset.fda_transform hook: per-channel CDF match toward the source."""

    def __init__(self, src_cdf):
        self.src_cdf = src_cdf

    def __call__(self, arr):
        return match_per_channel(arr, self.src_cdf)


class GrayMatcher:
    """dataset.fda_transform hook: the published grayscale-plus-ratio match."""

    def __init__(self, src_gray_cdf):
        self.src_gray_cdf = src_gray_cdf

    def __call__(self, arr):
        return match_gray(arr, self.src_gray_cdf)
