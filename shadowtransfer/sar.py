"""
SAR — Sharpness-Aware and Reliable entropy minimisation for test-time
adaptation. Niu et al., "Towards Stable Test-Time Adaptation in Dynamic Wild
World", ICLR 2023.

Written from the specification, not cloned: a search of `python/` on
2026-07-30 found no vendored SAR and no SAM implementation anywhere in the
tree (`find python -iname '*sar*'` and a grep for `first_step`/`second_step`
both returned nothing), and CLAUDE.md forbids depending on cluster network
access. The four components implemented here are exactly the four the arm F
spec names:

  1. entropy minimisation over the class axis of the model's own prediction
  2. a reliability filter dropping pixels with entropy above E_0
  3. a sharpness-aware update: first_step, second forward, second_step
  4. model recovery driven by a moving average of output entropy

TWO CONSTANTS ARE RESCALED FOR C = 2, NOT CARRIED OVER
------------------------------------------------------
E_0 = 0.4 * ln(C). At C = 2 that is 0.2773 nats. This one scales correctly
with class count by construction and is used as upstream writes it.

The model-recovery threshold does NOT scale. Upstream hardcodes 0.2, which is
2.8953% of ln(1000) = 6.9078. Max entropy at C = 2 is ln(2) = 0.6931, so a
confidently correct binary pixel already sits far below 0.2 and recovery would
fire on every step. `reset_constant` holds the same FRACTION of the entropy
bound instead: 0.6931 * 0.028953 = 0.020069. It is derived from the entropy
bound alone. It is not tuned against target performance and must never be
swept against target mIoU.

WHY THE MODEL IS NOT PUT IN train() MODE
----------------------------------------
Upstream `configure_model` calls `model.train()` and then disables BatchNorm
running statistics. Here the model is held in eval() and the running
statistics are disabled directly, which reaches the same BatchNorm behaviour
without the side effects:

  * `torch.nn.modules.batchnorm._BatchNorm.forward` selects batch statistics
    when `self.training` is True OR when both running buffers are None. Setting
    the buffers to None in eval mode therefore gives batch statistics exactly
    as `model.train()` would.
  * MAMNet's forward returns a dict of auxiliary outputs when `self.training`
    is True (`python/mamnet/models/mamnet.py:79`) and a plain logit tensor
    otherwise. `model.train()` would silently change the return type.
  * Dropout — MAMNet's AuxiliaryModule carries p=0.3 — would become active,
    making the adaptation stochastic. The arm F spec requires eval mode apart
    from the parameters SAR updates.

The cost is recorded in run/results provenance as `train_mode: eval`, and the
BatchNorm treatment is recorded per pass as `decoder_bn_mode`.
"""

import math

import torch
import torch.nn as nn

# Upstream constants, kept visible so the rescaling is auditable.
E0_FRACTION_OF_MAX_ENTROPY = 0.4
UPSTREAM_RESET_CONSTANT = 0.2
UPSTREAM_NUM_CLASSES = 1000
RESET_FRACTION_OF_MAX_ENTROPY = UPSTREAM_RESET_CONSTANT / math.log(UPSTREAM_NUM_CLASSES)

EMA_MOMENTUM = 0.9      # ema = 0.9 * ema + 0.1 * new
SAM_RHO = 0.05          # the SAM default, as the arm F spec states

# The SAR learning-rate BASE, upstream SAR's ViT default for SGD inside SAM,
# used unscaled. Upstream additionally scales lr by batch_size/64, which at the
# episodic batch size of 1 gives 3.9e-6 and no adaptation at all; that scaling
# is not applied. This constant is the `lr_base` of the arm F learning-rate
# pilot: the ladder is {1x, 10x, 100x} * SAR_LR_BASE, selected on the
# source-city validation split and never against held-out-city performance.
SAR_LR_BASE = 0.00025

# Upstream's parameter-collection skip list, transferred verbatim. `layer4` is
# the ResNet skip; `blocks.9/10/11` plus the final `norm` is the ViT-Base skip.
# Our DINOv3 is ViT-S/16 with 12 blocks, so the ViT skip transfers as written
# and excludes three of the four blocks the decoder reads (3, 6, 9, 11).
SKIP_SUBSTRINGS = ("layer4", "blocks.9", "blocks.10", "blocks.11")


def e0_margin(num_classes):
    """The reliability filter's entropy threshold, E_0 = 0.4 * ln(C)."""
    return E0_FRACTION_OF_MAX_ENTROPY * math.log(num_classes)


def reset_constant(num_classes):
    """Model-recovery threshold, held at upstream's FRACTION of max entropy."""
    return RESET_FRACTION_OF_MAX_ENTROPY * math.log(num_classes)


def reset_constant_derivation(num_classes):
    return (f"upstream hardcodes {UPSTREAM_RESET_CONSTANT} at C={UPSTREAM_NUM_CLASSES}, "
            f"which is {100 * RESET_FRACTION_OF_MAX_ENTROPY:.4f}% of "
            f"ln({UPSTREAM_NUM_CLASSES})={math.log(UPSTREAM_NUM_CLASSES):.4f}. "
            f"The same fraction of ln({num_classes})={math.log(num_classes):.4f} "
            f"is {reset_constant(num_classes):.6f}. Derived from the entropy "
            f"bound; never swept against target mIoU.")


def softmax_entropy(logits):
    """Per-pixel entropy over the class axis, in nats. [B,C,H,W] -> [B*H*W]."""
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1).flatten()


# ---------------------------------------------------------------------------
# SAM
# ---------------------------------------------------------------------------

class SAM(torch.optim.Optimizer):
    """Sharpness-aware minimisation. Ascend to the local worst case, then take
    the base optimiser's step from the original point using the gradient
    measured there."""

    def __init__(self, params, base_optimizer, rho=SAM_RHO, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        grads = [((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                 for group in self.param_groups
                 for p in group["params"] if p.grad is not None]
        if not grads:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        return torch.norm(torch.stack(grads), p=2)


# ---------------------------------------------------------------------------
# Model configuration and parameter collection
# ---------------------------------------------------------------------------

BN_BATCH_STATS = "batch_stats"          # as-released SAR: BN uses batch statistics
BN_EVAL_SOURCE_STATS = "eval_source_stats"   # BN stays in eval on source running stats


def configure_model(model, bn_mode):
    """Put the model in the state SAR adapts from. See the module docstring for
    why this does not call model.train().

    bn_mode = BN_BATCH_STATS reproduces upstream: every BatchNorm2d forgets its
    source running statistics and normalises with the statistics of the batch
    it is given. At batch size 1 that makes a BatchNorm2d an InstanceNorm.
    bn_mode = BN_EVAL_SOURCE_STATS leaves every BatchNorm2d in eval mode on the
    statistics the source training left behind.
    """
    if bn_mode not in (BN_BATCH_STATS, BN_EVAL_SOURCE_STATS):
        raise ValueError(f"unknown bn_mode {bn_mode}")
    model.eval()
    model.requires_grad_(False)
    touched = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_mode == BN_BATCH_STATS:
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            touched.append(nm)
    return model, touched


def collect_params(model, skip_substrings=SKIP_SUBSTRINGS, skip_final_norm=True,
                   norm_types=(nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
    """Affine parameters of normalisation layers, minus upstream's skip list.

    Returns (params, names, skipped_modules). The name list is logged for every
    pass: the arm F spec stops the run if it is empty or shorter than 4.
    """
    params, names, skipped = [], [], []
    for nm, m in model.named_modules():
        if any(s in nm for s in skip_substrings):
            if isinstance(m, norm_types):
                skipped.append(nm)
            continue
        # Upstream skips the ViT's final `norm` by exact module name. Ours is
        # nested under the backbone, so the match is on the trailing component.
        # ViT blocks name theirs norm1/norm2, which this does not catch.
        if skip_final_norm and (nm == "norm" or nm.endswith(".norm")):
            if isinstance(m, norm_types):
                skipped.append(nm)
            continue
        if isinstance(m, norm_types):
            for pn, p in m.named_parameters(recurse=False):
                if pn in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names, skipped


def make_optimizer(params, lr, momentum=0.9, rho=SAM_RHO):
    """Upstream's optimiser: SAM wrapping SGD with momentum."""
    return SAM(params, torch.optim.SGD, rho=rho, lr=lr, momentum=momentum)


# ---------------------------------------------------------------------------
# Episodic state
# ---------------------------------------------------------------------------

def snapshot_state(model):
    """Detached clone of every parameter and buffer, on the model's device."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restore_state(model, snap):
    model.load_state_dict(snap, strict=True)


def _bytes_equal(a, b):
    """Byte-identity, not value-equality. Falls back to torch.equal for dtypes
    a uint8 view cannot be taken of."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    try:
        return bool(torch.equal(a.contiguous().view(-1).view(torch.uint8),
                                b.contiguous().view(-1).view(torch.uint8)))
    except (RuntimeError, TypeError):
        return bool(torch.equal(a, b))


def state_matches(model, snap):
    """(ok, differing_keys). The episodic-isolation assertion of the arm F
    spec: model state at the start of image i must be byte-identical to the
    state at the start of image i+1, which is this snapshot."""
    cur = model.state_dict()
    if set(cur) != set(snap):
        return False, sorted(set(cur) ^ set(snap))
    bad = [k for k, v in cur.items() if not _bytes_equal(v, snap[k])]
    return (not bad), bad


# ---------------------------------------------------------------------------
# The adaptation loop
# ---------------------------------------------------------------------------

def update_ema(ema, new):
    return new if ema is None else EMA_MOMENTUM * ema + (1.0 - EMA_MOMENTUM) * new


def _observe(logits, margin):
    """Per-step diagnostics from one forward. Detached; nothing here is part of
    the graph."""
    with torch.no_grad():
        ent = softmax_entropy(logits)
        n_pass = int((ent < margin).sum().item())
        prob_fg = logits.softmax(1)[0, 1]
        return {
            "mean_entropy": float(ent.mean().item()),
            "n_pixels": int(ent.numel()),
            "n_pixels_passing_E0": n_pass,
            "frac_pixels_passing_E0": n_pass / float(ent.numel()),
        }, prob_fg.detach().to("cpu", torch.float32).numpy()


def adapt_image(model, x, steps, margin, reset_const, optimizer_factory,
                pristine_state):
    """Adapt on ONE image, episodically. Returns (records, prob_by_step).

    `records[s]` describes the model after s updates: its entropy statistics,
    the reliability-filter counts, the entropy moving average produced by the
    update that reached state s, and whether that update fired model recovery.
    `prob_by_step[s]` is P(shadow) under state s.

    Nothing survives the call: the caller restores `pristine_state` before the
    next image, and this function neither reads nor writes state outside the
    model and the optimiser it builds itself.
    """
    # steps == 0 is the unadapted baseline pass: one forward, no update, so
    # no optimiser is built (SAM over an empty parameter list would raise).
    optimizer = optimizer_factory() if steps > 0 else None
    ema = None
    pending_ema, pending_reset, pending_skip = None, False, None
    records, prob_by_step = [], {}

    for s in range(steps + 1):
        last = (s == steps)
        if last:
            with torch.no_grad():
                logits = model(x)
        else:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)

        obs, prob = _observe(logits, margin)
        obs.update({"step": s, "ema_entropy": pending_ema,
                    "reset_fired": pending_reset, "update_skipped": pending_skip})
        records.append(obs)
        prob_by_step[s] = prob
        if last:
            break

        # ---- the update from state s to state s+1 --------------------------
        entropys = softmax_entropy(logits)
        keep = torch.where(entropys < margin)
        pending_skip, pending_reset = None, False

        if keep[0].numel() == 0:
            # No pixel is reliable enough to learn from. Upstream would take a
            # NaN mean; the update is skipped instead and recorded as skipped.
            pending_skip = "first_filter_empty"
            del logits, entropys
            continue

        loss = entropys[keep].mean(0)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        entropys2 = softmax_entropy(model(x))[keep]
        keep2 = torch.where(entropys2 < margin)
        if keep2[0].numel() == 0:
            # Upstream calls backward() on a NaN here; that would put NaN into
            # the adapted parameters and silently end the pass. The ascent step
            # is rolled back and no base step is taken.
            with torch.no_grad():
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        st = optimizer.state[p]
                        if "old_p" in st:
                            p.data = st["old_p"]
            optimizer.zero_grad(set_to_none=True)
            pending_skip = "second_filter_empty"
            pending_ema = ema
            continue

        loss_second = entropys2[keep2].mean(0)
        ema = update_ema(ema, float(loss_second.item()))
        loss_second.backward()
        optimizer.second_step(zero_grad=True)
        pending_ema = ema

        if ema is not None and ema < reset_const:
            restore_state(model, pristine_state)
            optimizer = optimizer_factory()
            ema = None
            pending_reset = True

    return records, prob_by_step
