#!/usr/bin/env python3
"""
Arm A — switch off the TRAIN-side DetailedEvaluator, and prove it is inert.

Why
---
The pilot measured ~3.02 s/step against the prep phase's 0.4279 s/step. The
difference is not the model and not the loader: `Trainer.train_epoch` runs a
per-image `DetailedEvaluator` over EVERY training batch -- cv2 erode/dilate,
connectedComponents, distance transforms and per-blob contrast on each 384 px
patch -- on top of the forward/backward pass.

Separability, verified before anything was disabled (2026-08-02)
----------------------------------------------------------------
`mamnet/train.py` builds TWO independent evaluators in `Trainer.__init__`:

    :215  self.detailed_evaluator_train = DetailedEvaluator(...)
    :217  self.detailed_evaluator_val   = DetailedEvaluator(...)

and every reference to either is one-sided:

    train-side  :386 update      :433 compute_metrics    :434 reset
    val-side    :488 update      :514 compute_metrics    :515 reset

They share no object and no call site. The value that selects
`checkpoint_best.pth` and drives patience-20 is
`_get_decision_miou(detailed_results)` at :640, and that `detailed_results`
comes from `self.validate(...)` at :637 -- the VAL evaluator. The train-side
`detailed_results` at :433 is a local that `train_epoch` never returns; it
reaches only two `print` calls and four `self.writer.add_scalar` calls, and the
writer is already a NullWriter under armc_runtime.

`disable_train_evaluator` re-checks the two-distinct-objects property at
runtime and refuses to act if it ever stops holding.

What is NOT touched
-------------------
The validation-side evaluator, ShadowMetrics on either side, the loss, the
optimizer, the scheduler and the augmentation. This module swaps exactly one
attribute on one Trainer instance and edits no published file.
"""
import hashlib
import json


#: D-A5-3 (operator ruling, 2026-09-01). The eval-OFF lever applies to the
#: ATTRIBUTE THE TRAINING LOOP READS, not only to the canonical name.
#:
#: `armA_wave2._alias_evaluators` binds `detailed_evaluator_train` to the SAME
#: object the two IIM CNN trainers keep under `detailed_eval_train`
#: (mamnet/train_iim.py:532, oglanet/train_iim.py:597). Before this ruling the
#: lever replaced only the canonical name, so `self.detailed_eval_train.update`
#: at mamnet:691 / oglanet:719 -- the site the loop actually calls -- kept
#: running the LIVE evaluator. The lever was installed, reported, and inert.
#: Measured cost of that inertness: 1070.7 s (mamnet/iim) and 1070.2 s
#: (oglanet/iim) per train epoch, 82% of the epoch, on the wave-2f probes.
#:
#: The sweep below replaces EVERY instance attribute bound to the train-side
#: object, so a trainer's own name and any alias installed over it go inert
#: together. `armA_wave2e_evalcheck.py`'s P5 reads this flag by AST -- no
#: import, no torch -- and REFUSES any trainer whose loop-read attribute the
#: lever would not reach.
LEVER_ALIAS_SWEEP = True


class InertTrainEvaluator:
    """Stands in for the TRAIN-side DetailedEvaluator. Accumulates nothing.

    `compute_metrics` returns the exact shape `train_epoch` reads --
    `['boundary_tolerant']['strict']` and `[tol_key]`, each with
    precision/recall/f1/iou -- so the two print statements still format. The
    zeros it returns are the same zeros the real evaluator's own `_avg([])`
    returns for an empty accumulator (evaluation_detailed.py:434), so this is
    not a new code path for the printing.

    The real class is subclassed rather than duck-typed so that anything else
    reaching for an attribute we have not thought about still finds the
    published implementation.
    """

    def __init__(self, real, boundary_tolerance):
        self._real = real
        self.boundary_tolerance = boundary_tolerance
        self.tol_key = f"tolerant_{boundary_tolerance}px"
        self.n_update_calls = 0

    def update(self, *_a, **_k):
        self.n_update_calls += 1
        return None

    def reset(self):
        return None

    @staticmethod
    def _zeros():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    def compute_metrics(self):
        tol = self._zeros()
        tol["pixels_excluded"] = 0
        tol["pct_excluded"] = 0.0
        return {
            "overall": self._zeros(),
            "size_stratified": {},
            "size_stratified_tolerant": {},
            "contrast_stratified": {},
            "contrast_stratified_tolerant": {},
            "boundary_tolerant": {"strict": self._zeros(), self.tol_key: tol},
            "fp_fn_analysis": {},
        }

    def __getattr__(self, name):
        return getattr(self._real, name)


def disable_train_evaluator(trainer, verbose=True):
    """Swap the train-side evaluator for an inert one. Returns a state dict.

    Refuses, loudly, if the train and val evaluators are ever the same object
    or if either is missing -- in that case the two cannot be separated without
    a refactor of a published file, which is out of scope by construction.
    """
    tr = getattr(trainer, "detailed_evaluator_train", None)
    va = getattr(trainer, "detailed_evaluator_val", None)
    if tr is None or va is None:
        raise RuntimeError(
            "cannot disable the train-side evaluator: trainer is missing "
            f"detailed_evaluator_train={tr is not None} / "
            f"detailed_evaluator_val={va is not None}")
    if tr is va:
        raise RuntimeError(
            "REFUSING to disable: detailed_evaluator_train and "
            "detailed_evaluator_val are the SAME object, so the train-side "
            "evaluator cannot be switched off without also switching off the "
            "validation decision metric that selects checkpoint_best.pth. "
            "Separating them is a published-file refactor and is out of scope.")

    bt = getattr(trainer.args, "boundary_tolerance", 2)
    inert = InertTrainEvaluator(tr, bt)

    # D-A5-3: sweep every instance attribute bound to the train-side object.
    # `vars()` is snapshotted because the loop assigns into it. The val object
    # cannot be caught here -- `tr is va` was refused above -- but the identity
    # test is written against `tr` rather than against a name list so that an
    # alias nobody anticipated is still covered.
    swapped = []
    for _name, _val in list(vars(trainer).items()):
        if _val is tr:
            setattr(trainer, _name, inert)
            swapped.append(_name)
    if "detailed_evaluator_train" not in swapped:
        # no alias existed (the canonical name was never bound); bind it now so
        # every downstream consumer still finds an evaluator.
        trainer.detailed_evaluator_train = inert
        swapped.append("detailed_evaluator_train")

    # HARD ASSERTIONS. A lever that reports success while the loop still holds
    # the live object is the exact failure D-A5-3 exists to close, so it fails
    # loudly here instead of silently costing 82% of every epoch.
    _live = sorted(n for n, v in vars(trainer).items() if v is tr)
    if _live:
        raise RuntimeError(
            "D-A5-3 VIOLATION: after the swap, these Trainer attributes still "
            f"point at the LIVE train-side evaluator: {_live}. The lever would "
            "be inert on the attribute the training loop reads.")
    if getattr(trainer, "detailed_evaluator_val", None) is not va:
        raise RuntimeError(
            "D-A5-3 VIOLATION: the val-side evaluator was rebound by the "
            "train-side sweep. The decision metric must be untouched.")

    state = {
        "disabled": True,
        "what": "every Trainer attribute bound to the train-side evaluator",
        "attrs_swapped": sorted(swapped),
        "lever_alias_sweep": LEVER_ALIAS_SWEEP,
        "ruling": "D-A5-3",
        "val_evaluator_untouched": True,
        "train_val_evaluators_distinct_objects": True,
        "boundary_tolerance": bt,
        "rationale": ("the train-side per-image DetailedEvaluator is "
                      "observational: train.py:433 keeps its result in a local "
                      "that train_epoch never returns, and it reaches only two "
                      "prints and a NullWriter. The decision metric comes from "
                      "the VAL evaluator at train.py:637/640."),
    }
    if verbose:
        print(f"[seo_traineval] TRAIN-side DetailedEvaluator DISABLED on "
              f"{sorted(swapped)} (D-A5-3 alias sweep; val-side untouched, "
              f"distinct objects confirmed)", flush=True)
    return state


# ---------------------------------------------------------------------------
# Equivalence instrumentation
# ---------------------------------------------------------------------------

def weight_sha(model):
    """sha256 over every parameter and buffer, in sorted key order.

    Bytes, not values: two runs that agree here agree bit-for-bit on the model.
    """
    h = hashlib.sha256()
    sd = model.state_dict()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        try:
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        except (AttributeError, RuntimeError, TypeError):
            h.update(repr(v).encode())
    return h.hexdigest()


def install_trace(trainer, trace):
    """Record the four quantities the equivalence proof compares.

    Every hook wraps rather than replaces, and none of them touches a published
    file:

      per-step training loss  <- trainer.criterion.forward. The published
          MAMNetLoss is called once per training step; validate() calls the
          INNER `self.criterion.criterion` instead, so this hook is train-only.
      val decision metric     <- Trainer._get_decision_miou, whose return value
          is exactly what drives the scheduler, the best checkpoint and the
          patience counter.
      weight checksum,
      best epoch, patience    <- Trainer.save_checkpoint, called once per epoch
          AFTER the best/patience update.
    """
    crit = trainer.criterion
    orig_forward = crit.forward

    def traced_forward(outputs, target):
        out = orig_forward(outputs, target)
        try:
            trace["step_loss"].append(float(out["total"].item()))
        except Exception:
            pass
        return out

    crit.forward = traced_forward

    cls = type(trainer)
    orig_decision = cls._get_decision_miou
    orig_save = cls.save_checkpoint

    def traced_decision(self, detailed_results):
        v = orig_decision(self, detailed_results)
        trace["val_decision_miou"].append(float(v))
        return v

    def traced_save(self, epoch, is_best=False, *a, **k):
        out = orig_save(self, epoch, is_best, *a, **k)
        trace["epoch_state"].append({
            "epoch": int(epoch),
            "is_best": bool(is_best),
            "best_decision_miou": float(self.best_decision_miou),
            "epochs_without_improvement": int(self.epochs_without_improvement),
            "weight_sha": weight_sha(self.model),
            "n_steps_so_far": len(trace["step_loss"]),
        })
        return out

    cls._get_decision_miou = traced_decision
    cls.save_checkpoint = traced_save
    return {"hooks": ["criterion.forward", "Trainer._get_decision_miou",
                      "Trainer.save_checkpoint"]}


def new_trace():
    return {"step_loss": [], "val_decision_miou": [], "epoch_state": []}


def trace_sha(trace):
    return hashlib.sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def install_val_predpos_probe(trainer, sink):
    """Record the predicted-positive fraction on the validation set per epoch.

    OBSERVATION ONLY. It wraps `detailed_evaluator_val.update` on the instance,
    counts the positives in `preds`, and then calls straight through with the
    arguments untouched. No returned value changes and no accumulator is
    altered, so the validation decision metric that selects
    checkpoint_best.pth is bit-for-bit what it would have been.

    This exists because the stage-3 inversion diagnostic needs a per-epoch
    predicted-positive fraction, and nothing the published trainer prints lets
    it be recovered exactly.
    """
    ev = trainer.detailed_evaluator_val
    orig = ev.update
    state = {"pos": 0, "tot": 0}

    def probed(predictions, targets, images=None):
        try:
            state["pos"] += int((predictions > 0).sum().item())
            state["tot"] += int(predictions.numel())
        except Exception:
            pass
        return orig(predictions, targets, images)

    ev.update = probed

    cls = type(trainer)
    orig_decision = cls._get_decision_miou

    def traced(self, detailed_results):
        v = orig_decision(self, detailed_results)
        sink.append({"pred_pos": state["pos"], "n_px": state["tot"],
                     "pred_pos_frac": (state["pos"] / state["tot"]
                                       if state["tot"] else None),
                     "val_decision_miou": float(v)})
        state["pos"] = 0
        state["tot"] = 0
        return v

    cls._get_decision_miou = traced
    return {"probe": "val predicted-positive fraction per epoch",
            "observation_only": True}


# ---------------------------------------------------------------------------
# Phase-1 timing / weight-load / step-loss instrumentation (arm A task 2)
# ---------------------------------------------------------------------------
#
# All three are OBSERVATION ONLY. Each wraps a callable and calls straight
# through with its arguments untouched; none of them changes a returned value,
# an accumulator, or a decision. They exist because the DINOv3 timing probe is
# link 1 of a REAL grid cell -- the cell must not be perturbed by being
# measured, and the measurement must survive the link being SIGKILLed on its
# wall clock, which is the expected end of a probe link.


class LoadReport:
    """Every `Module.load_state_dict` call made while this is active.

    Arm E found 37 of 188 tensors silently dropped on the DINOv3 constructor:
    `dinov3_backbone.py:90` calls `load_state_dict(..., strict=False)` and
    prints only the first five names of each list. `strict=False` makes a
    partial load indistinguishable from a complete one in the log, so this
    records the FULL counts and the first ten names of each, per call.
    """

    def __init__(self):
        self.calls = []
        self._orig = None

    def __enter__(self):
        import torch
        self._orig = torch.nn.Module.load_state_dict

        def recording(module_self, state_dict, *a, **k):
            out = self._orig(module_self, state_dict, *a, **k)
            try:
                missing = list(getattr(out, "missing_keys", []) or [])
                unexpected = list(getattr(out, "unexpected_keys", []) or [])
                self.calls.append({
                    "module": type(module_self).__name__,
                    "n_tensors_offered": len(state_dict),
                    "n_tensors_in_module": len(module_self.state_dict()),
                    "n_missing_keys": len(missing),
                    "n_unexpected_keys": len(unexpected),
                    "missing_keys_first10": missing[:10],
                    "unexpected_keys_first10": unexpected[:10],
                    "strict": bool(k.get("strict", a[0] if a else True)),
                })
            except Exception:
                self.calls.append({"module": type(module_self).__name__,
                                   "capture_failed": True})
            return out

        torch.nn.Module.load_state_dict = recording
        return self

    def __exit__(self, *_exc):
        import torch
        if self._orig is not None:
            torch.nn.Module.load_state_dict = self._orig
        return False

    def summary(self):
        """The construction-time loads, worst first by dropped-tensor count."""
        return sorted(self.calls,
                      key=lambda c: -int(c.get("n_missing_keys", 0) or 0))


def install_step_loss_probe(trainer, sink):
    """Append every TRAINING step's loss to `sink`.

    Train-only by construction rather than by flag: `validate()` runs under
    `torch.no_grad()` in all three published trainers, so `is_grad_enabled()`
    separates the two without needing to know which trainer this is. Handles
    both loss shapes in the tree -- MAMNet's `MAMNetLoss` returns a dict with a
    'total' key, DINOv3's `CrossEntropyLoss` returns the tensor itself.

    Install AFTER any criterion surgery: MAMNet's ignore_index patch replaces
    `trainer.criterion.criterion` (the inner object, which this does not wrap),
    but DINOv3's replaces `trainer.criterion` outright and would drop the hook.
    """
    import torch
    crit = trainer.criterion
    orig = crit.forward

    def probed(*a, **k):
        out = orig(*a, **k)
        if torch.is_grad_enabled():
            try:
                v = out["total"] if isinstance(out, dict) else out
                sink.append(float(v.item()))
            except Exception:
                pass
        return out

    crit.forward = probed
    return {"probe": "per-step training loss",
            "hook": f"{type(crit).__name__}.forward (instance attribute)",
            "train_only_mechanism": "torch.is_grad_enabled()",
            "observation_only": True}


def install_epoch_probe(trainer, ctx, on_epoch=None):
    """Time `train_epoch` and `validate` separately, per epoch.

    `on_epoch` is called after every validation with the per-epoch record, so a
    probe artifact can be flushed to disk each epoch. That flush is what makes
    the measurement survive the SIGKILL that ends a wall-clock-bounded probe
    link: a probe written only at the end of training is a probe that a killed
    link never writes.

    Wraps the CLASS, because `train_epoch` and `validate` are called as bound
    methods on self; re-entry is guarded so a second install is a no-op.
    """
    import time
    import torch

    cls = type(trainer)
    if getattr(cls.train_epoch, "_st_epoch_probe", False):
        return {"installed": False, "reason": "already installed on this class"}
    orig_train = cls.train_epoch
    orig_val = cls.validate
    ctx.setdefault("epochs", [])

    def timed_train(self, epoch, *a, **k):
        t0 = time.time()
        out = orig_train(self, epoch, *a, **k)
        ctx["epochs"].append({
            "epoch": int(epoch),
            "train_s": round(time.time() - t0, 3),
            "val_s": None,
            "peak_vram_reserved_gb": round(
                torch.cuda.max_memory_reserved() / (1024 ** 3), 3),
        })
        return out

    def timed_val(self, epoch, *a, **k):
        t0 = time.time()
        out = orig_val(self, epoch, *a, **k)
        rec = None
        for r in reversed(ctx["epochs"]):
            if r["epoch"] == int(epoch):
                rec = r
                break
        if rec is None:
            rec = {"epoch": int(epoch), "train_s": None}
            ctx["epochs"].append(rec)
        rec["val_s"] = round(time.time() - t0, 3)
        rec["peak_vram_reserved_gb"] = round(
            torch.cuda.max_memory_reserved() / (1024 ** 3), 3)
        if on_epoch is not None:
            try:
                on_epoch(rec)
            except Exception as e:                     # never kill a cell to log it
                print(f"[seo_traineval] epoch-probe flush failed "
                      f"(non-fatal): {type(e).__name__}: {e}", flush=True)
        return out

    timed_train._st_epoch_probe = True
    timed_val._st_epoch_probe = True
    cls.train_epoch = timed_train
    cls.validate = timed_val
    return {"installed": True,
            "probe": "per-epoch train and validate wall clock, split",
            "hooks": [f"{cls.__name__}.train_epoch", f"{cls.__name__}.validate"],
            "observation_only": True}
