"""Arm F / CrossEarth — the training / validation / inference engine.

Shared by armf_crossearth_train.py (the 12 cells) and
armf_crossearth_sib_smoke.py (the gate on the SIB insertion point).

No Runner, no hooks, no visualizer, no tensorboard, no matplotlib: the loops
drive model.loss / model.predict directly. That is deliberate, after the
90-minute metrics_curves.png hang on 2026-07-27.

Nothing here computes a metric. Counts and IoU come from
shadowtransfer.metrics via shadowtransfer.armc_dump.
"""

import json
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- numpy 1.26 shim, installed BEFORE shadowtransfer.metrics is used -------
# metrics.py calls np.trapezoid at two sites (aurc_trapz19, aurc_ce_trapz19).
# np.trapezoid is the numpy>=2.0 name for np.trapz: same function, renamed, no
# behaviour change. This arm's env is pinned to numpy 1.26.4 because torch
# 2.1.0+cu118 is not compiled against numpy 2 ("Failed to initialize NumPy:
# _ARRAY_API not found"), so the newer name is absent here.
#
# The alias is installed rather than metrics.py being edited: metrics.py is the
# single canonical metric implementation (CONTRACT.md rule zero) and every other
# arm runs it under numpy 2 in satmae_cuda12. Editing it to np.trapz would be a
# contract change made for one env's convenience. Recorded per run as
# numpy_trapezoid_shim in run.json.
NUMPY_TRAPEZOID_SHIM = not hasattr(np, "trapezoid")
if NUMPY_TRAPEZOID_SHIM:
    np.trapezoid = np.trapz

from shadowtransfer.armc_dump import miou_from_counts  # noqa: E402
from shadowtransfer.metrics import (  # noqa: E402
    per_image_counts, threshold_and_filter)

BAND_TOL = 2


class PhaseTimeout(Exception):
    pass


class phase_timeout:
    """Hard wall on a post-training phase, naming the phase it died in.

    CLAUDE.md: inference, metric computation and dump writing get an explicit
    timeout that prints the phase, so a hang fails loudly instead of eating the
    wall clock.
    """

    def __init__(self, seconds, what):
        self.seconds = int(seconds)
        self.what = what

    def _fire(self, *_):
        raise PhaseTimeout(f"phase '{self.what}' exceeded {self.seconds}s")

    def __enter__(self):
        self.prev = signal.signal(signal.SIGALRM, self._fire)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.prev)
        return False


def build_optimizer(model, cfg_optim=None, lr=1e-4, weight_decay=0.05):
    """The released optim_wrapper, built through mmengine.

    CrossEarth_dinov2_mask2former_512x512_bs1x4.py declares
    PEFTOptimWrapperConstructor + AdamW(1e-4, wd 0.05, eps 1e-8,
    betas (0.9, 0.999)) with decay_mult 0 on norms and on the embedding-like
    keys. Building it through mmengine reuses the release's own param-group
    logic rather than reimplementing it.
    """
    from mmengine.optim import build_optim_wrapper
    import CrossEarth  # noqa: F401  registers PEFTOptimWrapperConstructor

    embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
    cfg = cfg_optim or dict(
        type="OptimWrapper",
        constructor="PEFTOptimWrapperConstructor",
        optimizer=dict(type="AdamW", lr=lr, weight_decay=weight_decay,
                       eps=1e-8, betas=(0.9, 0.999)),
        paramwise_cfg=dict(custom_keys={
            "norm": dict(decay_mult=0.0),
            "query_embed": embed_multi,
            "level_embed": embed_multi,
            "learnable_tokens": embed_multi,
            "reins.scale": embed_multi,
        }, norm_decay_mult=0.0),
    )
    return build_optim_wrapper(model, cfg), cfg


def build_scheduler(optim_wrapper, end=40000):
    """PolyLR(eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False).

    Kept at the released end=40000 rather than rescaled to our 10,000-iteration
    budget: rescaling would change the LR trajectory as well as its length,
    which is a second deviation. At iteration 10,000 the multiplier is
    (1 - 10000/40000)^0.9 = 0.772.
    """
    from mmengine.optim.scheduler import PolyLR
    return PolyLR(optim_wrapper.optimizer, eta_min=0, power=0.9,
                  begin=0, end=end, by_epoch=False)


def infinite_loader(dataset, batch_size, seed, num_workers=8, collate_fn=None):
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True,
                        collate_fn=collate_fn, generator=g,
                        persistent_workers=num_workers > 0)
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def predict_probs(model, dataset, collate_fn, batch_size=4, num_workers=4,
                  amp_dtype=torch.bfloat16):
    """Foreground probability per image. Returns (image_ids, gts, probs)."""
    from shadowtransfer.armf_crossearth_runtime import foreground_prob, make_data
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)
    ids, gts, probs = [], [], []
    for batch in loader:
        data = make_data(batch, model, training=False)
        with torch.autocast("cuda", dtype=amp_dtype):
            out = model.predict(data["inputs"], data["data_samples"])
        for k, sample in enumerate(out):
            probs.append(foreground_prob(sample.seg_logits.data))
            gts.append(batch["gt"][k].numpy().astype(np.uint8))
            ids.append(batch["image_id"][k])
    return ids, gts, probs


def score_split(gts, probs, min_pixels=10):
    """Per-image scalars macro-averaged over the split. No micro-pooling.

    Returns both readings: `miou_tol2` is the 2-class boundary-tolerant mIoU at
    K = 2 that this arm's results.json reports and that selects the
    checkpoint; `iou_fg_tol2` is the Appendix E foreground-only IoU at K = 2
    that CLAUDE.md names as the project's reporting metric. Both come from the
    same counts, so they can never disagree about the underlying prediction.
    """
    mious, ious, fgfracs = [], [], []
    for gt, prob in zip(gts, probs):
        pred = threshold_and_filter(prob, min_pixels=min_pixels)
        c = per_image_counts(pred, gt, boundary_tolerance=BAND_TOL)
        fg, _, mi = miou_from_counts(c["tp_tol"], c["fp_tol"], c["tn_tol"],
                                     c["fn_tol"])
        mious.append(mi)
        ious.append(fg)
        fgfracs.append(float(pred.mean()))
    return {
        "miou_tol2": float(np.mean(mious)),
        "iou_fg_tol2": float(np.mean(ious)),
        "pred_fg_fraction_mean": float(np.mean(fgfracs)),
        "pred_fg_fraction_min": float(np.min(fgfracs)),
        "pred_fg_fraction_max": float(np.max(fgfracs)),
        "n_images": len(mious),
    }


def trainable_state_dict(model):
    """Reins + head (+ SIB). The frozen DINOv2 trunk is never written.

    ReinsDinoVisionTransformer.state_dict already drops every non-rein key, but
    we select explicitly so the saved keys do not depend on that override.
    """
    sd = {}
    for prefix, module in (("backbone.reins", model.backbone.reins),
                           ("decode_head", model.decode_head)):
        for k, v in module.state_dict().items():
            sd[f"{prefix}.{k}"] = v.detach().cpu()
    return sd


def load_trainable_state_dict(model, sd):
    """Restore a trainable_state_dict onto an init_weights()-ed model."""
    incompatible = model.load_state_dict(sd, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected keys in checkpoint: {incompatible.unexpected_keys[:5]}")
    still_missing = [k for k in incompatible.missing_keys
                     if "reins" in k or k.startswith("decode_head")]
    if still_missing:
        raise RuntimeError(f"trainable keys missing: {still_missing[:5]}")
    return {"n_loaded": len(sd), "n_missing_frozen": len(incompatible.missing_keys)}


def train(model, train_ds, val_ds, collate_fn, out, iters, batch_size,
          val_interval, seed, ckpt_path, sib=None, amp_dtype=torch.bfloat16,
          num_workers=8, log_every=100, save_best=True):
    """The iteration loop. Returns (val_curve, best, steps_to_best, stats).

    Validation is source-city only and drives checkpoint selection through
    2-class boundary-tolerant mIoU at K = 2. No save_best against any target
    metric and no held-out-city split in the validation dataloader: the
    released DG configs do both, and this loop deliberately does neither.
    """
    from shadowtransfer.armf_crossearth_runtime import make_data
    from shadowtransfer.armf_crossearth_sib import kl_ramp

    optim_wrapper, optim_cfg = build_optimizer(model)
    scheduler = build_scheduler(optim_wrapper)
    gen = infinite_loader(train_ds, batch_size, seed, num_workers, collate_fn)

    val_curve, best, steps_to_best = [], None, None
    losses, t0 = [], time.time()
    for it in range(iters):
        model.train()
        batch = next(gen)
        data = make_data(batch, model, training=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            loss_dict = model.loss(data["inputs"], data["data_samples"])
        if sib is not None:
            loss_dict["loss_sib_kl"] = (kl_ramp(it, iters) * sib.last_kl)
        loss, log_vars = model.parse_losses(loss_dict)
        optim_wrapper.update_params(loss)
        scheduler.step()
        losses.append(float(loss.detach()))
        if not np.isfinite(losses[-1]):
            raise RuntimeError(f"non-finite loss at iteration {it}: {losses[-1]}")

        if (it + 1) % log_every == 0:
            print(f"[train] iter {it + 1}/{iters} loss {np.mean(losses[-log_every:]):.4f} "
                  f"lr {optim_wrapper.optimizer.param_groups[0]['lr']:.3e} "
                  f"{(time.time() - t0) / (it + 1):.4f} s/iter", flush=True)

        if (it + 1) % val_interval == 0 or (it + 1) == iters:
            _, vg, vp = predict_probs(model, val_ds, collate_fn,
                                      batch_size=batch_size,
                                      num_workers=max(num_workers // 2, 2),
                                      amp_dtype=amp_dtype)
            s = score_split(vg, vp)
            s["iter"] = it + 1
            s["val_miou_tol2"] = s["miou_tol2"]
            val_curve.append(s)
            print(f"[val] {json.dumps(s)}", flush=True)
            if best is None or s["miou_tol2"] > best["miou_tol2"]:
                best, steps_to_best = s, it + 1
                if save_best:
                    torch.save({"state_dict": trainable_state_dict(model),
                                "iter": it + 1, "val": s}, ckpt_path)

    stats = {
        "s_per_iter": float((time.time() - t0) / max(iters, 1)),
        "loss_mean_first10": float(np.mean(losses[:10])),
        "loss_mean_last10": float(np.mean(losses[-10:])),
        "loss_finite": bool(np.all(np.isfinite(losses))),
        "losses": [float(x) for x in losses],
        "optim_cfg": json.loads(json.dumps(optim_cfg, default=str)),
    }
    return val_curve, best, steps_to_best, stats
