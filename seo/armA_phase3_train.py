#!/usr/bin/env python3
"""Arm A / S-EO PHASE 3 -- one method cell (FDA or SIB), trained and dumped.

LOCO ONLY. No new upper bound is trained: each fold's phase-2 vanilla upper
cell is the comparator, and it is already on disk. OGLANet is dropped from arm
A and is absent from `--arch`'s choices. No diagnostics.

Relationship to phase 2
-----------------------
This script does NOT re-implement the phase-2 cell driver. It imports
`armA_pilot_train` and reuses its helpers unchanged -- the budget-manifest
`PatchSubset`, the `TrainTargets` ignore-index target, the corrupt-checkpoint
resolver, the cluster re-routing, the `set_ignore_index_loss` installation, the
checkpoint pruner and the seeding. Only `main()` is this file's own, because
six points in it differ. A separate file rather than a `--method` branch inside
`armA_pilot_train.py` because that file's sha256 is recorded in all twelve
finished vanilla cells' `run.json` as `train_script_sha`; editing it would
break that record for artifacts that are already written. Both shas are
recorded here, so the lineage is explicit rather than implied.

The six differences
-------------------
  1. `--method {fda,sib}`; `run_id` carries it; the condition is always `loco`.
  2. The TRAIN split is built from `armA_methods.SEODatasetFDA` when an FDA
     component is active (method=fda on either arch, method=sib on MAMNet).
     Val and test are never FDA-transformed, exactly as the published cells do.
  3. `armA_methods.install_sib` replaces the model, optimiser, scheduler and
     loss with the published SIB ones, AFTER `set_ignore_index_loss`, so the
     KL rides on a loss that already drops unseen pixels.
  4. A method probe, run in smoke mode, that proves the method's transform
     actually executed rather than trusting a flag.
  5. An overhead measurement: N timed train steps with the method against N
     with the vanilla model of the same arch, same batch, same node.
  6. `run.json` gains the method block, the FDA target-pool provenance, and
     the assertion that this cell's test patch-id set equals the fold's
     vanilla cells'.

Everything else -- the 100-epoch ceiling, flat patience 20, the eval-OFF lever
under `code_audit` with its three `run.json` fields, the unseen-pixel rule, the
dump spec, `probs.npz` float16, the quarantine preflight, the post-training
time-box, the figure and tensorboard silencing -- is phase 2's, unchanged.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import armA_pilot_train as P                                   # noqa: E402
from shadowtransfer import armA_methods as M                   # noqa: E402
from shadowtransfer import armc_runtime                        # noqa: E402
from shadowtransfer import seo_budget                          # noqa: E402
from shadowtransfer import seo_dump                            # noqa: E402
from shadowtransfer import seo_traineval                       # noqa: E402
from shadowtransfer.armc_dump import sha256_file               # noqa: E402
from shadowtransfer.seo_dataset import SEODataset, assert_no_leakage  # noqa: E402

RES = P.RES
RUNG = P.RUNG
PATIENCE_EPOCHS = P.PATIENCE_EPOCHS
EPOCHS_MAX = P.EPOCHS_MAX
DATASET = P.DATASET
ARCH_SPEC = P.ARCH_SPEC
ARCH_DROPPED = P.ARCH_DROPPED


def fda_is_active(arch, method):
    """Is the FDA amplitude swap itself active for this (arch, method)?

    NOTE the narrowed docstring. This used to say "touch held-out-city
    imagery at all", and it was the ONLY predicate the target_access guard
    consulted. That conflated two different questions, which was fine while
    FDA was the only route to held-out imagery and is not fine now -- see
    `touches_heldout_imagery` below. The RETURN VALUES ARE UNCHANGED, so
    every FDA-specific site that reads `use_fda` (the L= banner at :680, the
    FDA build at :718, the probe at :1080) behaves exactly as before.
    """
    if method == "fda":
        return True
    if method == "sib":
        return bool(M.SIB_SPEC[arch]["use_fda"])
    return False


#: Methods that reach held-out-city imagery WITHOUT the FDA amplitude swap.
#: segdesic takes a SECOND FORWARD PASS on an unlabeled target batch every
#: step for its domain loss (mamnet/train_segdesic.py:417), and in LOCO mode
#: that target pool is the held-out city's own imagery. It is
#: `unlabeled_target` in the registry on all three architectures -- a finding
#: read off the code path that runs, not off the trainer file's appearance,
#: because on DINOv3 the loader is built inside `data.dataset.get_dataloaders`
#: rather than inside the trainer.
HELDOUT_FORWARD_METHODS = ("segdesic",)


def touches_heldout_imagery(arch, method):
    """Does this (arch, method) reach held-out-city imagery by ANY route?

    ARM A SESSION B, DELIBERATE AND DECLARED WIDENING. The target_access
    guard below asserts that the manifest and the code path agree about
    whether held-out imagery is touched. Its old form compared
    `target_access == 'unlabeled_target'` against `fda_is_active`, which made
    "touches held-out imagery" and "runs the FDA amplitude swap" the same
    proposition. For segdesic they are different propositions and the guard
    would have fired on a correctly-registered cell.

    `shadowtransfer/armA_wave2.py:NOT_INSTALLABLE_HERE['segdesic']` names this
    exact collision and says widening the guard is "a deliberate decision, not
    a silent bypass". This is that decision, taken in the open: the guard gets
    a WIDER predicate, `use_fda` keeps its old value and its old meaning, and
    no FDA code path is reached by a non-FDA method.
    """
    return fda_is_active(arch, method) or method in HELDOUT_FORWARD_METHODS


def fda_L_for(arch, method):
    if method == "fda":
        return M.FDA_SPEC[arch]["fda_L"]
    return M.SIB_SPEC[arch]["fda_L"]


def fda_pool_max_for(arch, method):
    if method == "fda":
        return M.FDA_SPEC[arch]["target_pool_max"]
    return M.SIB_SPEC[arch]["target_pool_max"]


def build_split(img_root, mask_root, patch_index, splits_dir, split, patch_ids,
                augment, label, use_contrast, fda=None):
    """`armA_pilot_train.build_split`, with an optional FDA transform.

    Identical in every other respect: one dataset per contributing city,
    filtered to the budget manifest's ids in manifest order, concatenated in
    city order. `fda` is passed only for the train split and only when an FDA
    component is active; val and test always get the plain loader.
    """
    from torch.utils.data import ConcatDataset
    by_city = {}
    for p in patch_ids:
        by_city.setdefault(P.city_of(p), []).append(p)
    parts, bases = [], []
    for city in sorted(by_city):
        kw = dict(img_root=img_root, mask_root=mask_root,
                  patch_index=patch_index,
                  split_manifest=str(Path(splits_dir) / f"{city}.json"),
                  split=split, patch=384, augment=augment,
                  use_contrast=use_contrast, exclude_unseen=True,
                  exclude_uncertain=False)
        base = (M.SEODatasetFDA(fda=fda, **kw) if fda is not None
                else SEODataset(**kw))
        parts.append(P.PatchSubset(base, by_city[city], f"{label}/{city}"))
        bases.append(base)
    ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    rows = [r for p in parts for r in p.rows]
    if len(rows) != len(patch_ids):
        raise SystemExit(f"FATAL {label}: assembled {len(rows)} of {len(patch_ids)}")
    return ds, rows, bases, parts


def probe_fda(fda_mod, fda_transform, target_crops, ds_train, L, label):
    """Prove the amplitude swap EXECUTED, rather than trusting `--use_fda`.

    The swap is verified WHERE IT HAPPENS -- on the published
    `low_freq_mutate`, against the real source and target spectra of this
    cell's own pool -- and not through the finished image. Verifying it through
    the image cannot work: `fda_source_to_target` reconstructs by inverse FFT,
    clips to [0, 255] and casts back to uint8, and at L=0.005 the swapped
    rectangle is 2x2 coefficients out of 384x384, so quantisation noise is
    larger than the signal being checked. Measured on a real OMA pair,
    2026-08-02: the round-tripped rectangle sits 0.035 (L=0.01) and 0.044
    (L=0.005) away from the target's in relative max-abs terms, with the same
    quantity between source and target only 0.135 and 0.087. Those numbers
    prove nothing either way, which is why they are recorded below as
    information and nothing is gated on them.

    Four checks, of which the first two are exact:
      1. the mutated spectrum's low-frequency rectangle is BIT-EQUAL to the
         target's;
      2. everything outside that rectangle is BIT-EQUAL to the source's, so
         the mutation is confined to where FDA says it is;
      3. the half-width b = floor(min(H,W) * L) is at least 1, so the swap is
         not a silent no-op;
      4. pulling one real training sample increments the loader's own
         `_fda_calls` counter, so the transform is wired into the split that
         trains and not merely constructed.
    """
    base = ds_train.base if hasattr(ds_train, "base") else None
    src = np.array(target_crops[0]).astype(np.float32)
    trg = np.array(target_crops[-1]).astype(np.float32)
    h, w = src.shape[:2]
    b = int(np.floor(min(h, w) * L))
    if b < 1:
        raise SystemExit(f"FATAL {label}: L={L} on a {h}x{w} patch gives a "
                         f"half-width of {b}; the amplitude swap would be a "
                         f"no-op")

    amp_src = np.abs(np.fft.fft2(src, axes=(0, 1)))
    amp_trg = np.abs(np.fft.fft2(trg, axes=(0, 1)))
    mut = fda_mod.low_freq_mutate(amp_src, amp_trg, L=L)

    sh = lambda z: np.fft.fftshift(z, axes=(0, 1))
    m_s, s_s, t_s = sh(mut), sh(amp_src), sh(amp_trg)
    ch, cw = h // 2, w // 2
    rect = (slice(ch - b, ch + b), slice(cw - b, cw + b))
    rect_exact = bool(np.array_equal(m_s[rect], t_s[rect]))
    keep = np.ones((h, w), dtype=bool)
    keep[rect] = False
    outside_exact = bool(np.array_equal(m_s[keep], s_s[keep]))

    out = fda_mod.fda_source_to_target(
        np.array(target_crops[0]), np.array(target_crops[-1]), L=L)
    changed = int(np.abs(out.astype(np.int32)
                         - np.array(target_crops[0]).astype(np.int32)).sum())

    calls0 = getattr(base, "_fda_calls", None)
    _ = ds_train[0]
    calls1 = getattr(base, "_fda_calls", None)
    wired = (calls0 is not None and calls1 is not None and calls1 > calls0)

    def _rel(a, c):
        d = np.abs(a - c).max()
        return round(float(d / max(float(np.abs(c).max()), 1e-6)), 6)

    rep = {"fda_probe_L": L,
           "fda_probe_half_width_b": b,
           "fda_probe_swapped_rect_px": 2 * b,
           "fda_probe_rect_bit_equal_to_target": rect_exact,
           "fda_probe_outside_bit_equal_to_source": outside_exact,
           "fda_probe_pixels_changed_uint8": changed,
           "fda_probe_loader_calls_before_after": [calls0, calls1],
           "fda_probe_wired_into_train_split": bool(wired),
           "fda_probe_roundtrip_info_only": {
               "note": ("uint8 round-trip distances, RECORDED NOT GATED: "
                        "clipping and quantisation dominate a 2b x 2b "
                        "rectangle this small"),
               "rel_out_vs_target": _rel(sh(np.abs(np.fft.fft2(
                   out.astype(np.float32), axes=(0, 1))))[rect], t_s[rect]),
               "rel_source_vs_target": _rel(s_s[rect], t_s[rect]),
           }}
    if not rect_exact:
        raise SystemExit(f"FATAL {label}: low_freq_mutate did not write the "
                         f"target's low-frequency rectangle; {rep}")
    if not outside_exact:
        raise SystemExit(f"FATAL {label}: low_freq_mutate altered coefficients "
                         f"OUTSIDE the low-frequency rectangle; {rep}")
    if changed == 0:
        raise SystemExit(f"FATAL {label}: FDA produced a bit-identical image; "
                         f"the swap did not survive reconstruction; {rep}")
    if not wired:
        raise SystemExit(f"FATAL {label}: FDA transform is not wired into the "
                         f"TRAIN split loader; {rep}")
    print(f"[armA] FDA probe PASS  {rep}", flush=True)
    return rep


def probe_sib(model, arch, state, batch, device, label):
    """Prove each ACTIVE SIB component fired at least once on a real batch.

    Forward hooks count calls on the components that ARE modules. Two
    components are not modules and are proved differently:

      - DINOv3's Haar is the free function `haar_forward`, so it carries no
        hook. It is proved from the loss dict instead: the differential
        (Haar) branch emits `kl_content` AND `kl_edge`, while the no-Haar
        branch emits `kl_uniform`. Seeing the first pair and not the second is
        positive evidence the Haar path executed.
      - MAMNet's Haar IS a module (`sib.haar`) and is hooked.

    A component the published C4-clean config declares OFF must have no module
    at all: a flag that is off while its module was still built would be a
    silent recipe change, so that is a hard failure too.

    Runs in SMOKE MODE ONLY. It pushes one batch through the model in train
    mode, which touches BatchNorm running statistics; a smoke cell's weights
    are discarded, a real cell's must not be perturbed before epoch 1.
    """
    import torch
    spec = M.SIB_SPEC[arch]
    counts = {}
    handles = []

    def hook(name):
        def f(_m, _i, _o):
            counts[name] = counts.get(name, 0) + 1
        return f

    sib = getattr(model, "sib", None)
    if sib is None:
        raise SystemExit(f"FATAL {label}: model has no .sib module")

    wanted = {}
    for nm, attrs in (("haar", ("haar",)),
                      ("content_vib", ("vib_content", "content_vib")),
                      ("edge_vib", ("vib_edge", "edge_vib"))):
        for attr in attrs:
            mod = getattr(sib, attr, None)
            if mod is not None:
                wanted[nm] = mod
                break
    for nm, mod in wanted.items():
        handles.append(mod.register_forward_hook(hook(nm)))
    # SAG is an nn.ModuleList, which has no forward of its own -- a hook on the
    # list never fires however many times its four gates are called. Hook the
    # ELEMENTS, and require all four skip levels rather than merely one.
    n_sag_gates = 0
    if arch == "mamnet" and spec["flags"]["use_sag"]:
        gates = getattr(model, "sag_gates", None)
        if gates is None:
            raise SystemExit(f"FATAL {label}: use_sag is on but model.sag_gates "
                             f"is None")
        n_sag_gates = len(gates)
        if n_sag_gates != 4:
            raise SystemExit(f"FATAL {label}: expected one SAG gate per skip "
                             f"level (4), found {n_sag_gates}")
        for i, gmod in enumerate(gates):
            handles.append(gmod.register_forward_hook(hook(f"sag{i + 1}")))

    # Components C4-clean declares OFF must not have been built.
    absent = {}
    for nm, attr in (("content_aug", "content_aug"),
                     ("intensity_beta", "intensity_beta"),
                     ("passthrough_gate", "passthrough_gate"),
                     ("uniform_vib", "uniform_vib")):
        absent[nm] = getattr(sib, attr, None) is not None
    absent["module_bypass"] = bool(getattr(model, "use_module_bypass", False))
    absent["multiscale_sib"] = bool(getattr(model, "use_multiscale_sib", False))
    built_but_off = sorted(k for k, v in absent.items() if v)

    was = model.training
    model.train()
    with torch.enable_grad():
        _ = model(batch["image"].to(device))
    for h in handles:
        h.remove()
    if not was:
        model.eval()

    kl = (state.get("sib_losses") or {})
    kl_keys = sorted(k for k, v in kl.items() if isinstance(v, torch.Tensor)
                     and v.dim() == 0)
    kl_total = kl.get("kl_total", kl.get("kl_content"))
    kl_val = float(kl_total.detach().item()) if kl_total is not None else None

    need = ["content_vib", "edge_vib"]
    if arch == "mamnet":
        need = ["haar"] + need + [f"sag{i + 1}" for i in range(n_sag_gates)]
    missing = [n for n in need if counts.get(n, 0) < 1]

    haar_via_loss = None
    if arch == "dinov3":
        haar_via_loss = ("kl_content" in kl_keys and "kl_edge" in kl_keys
                         and "kl_uniform" not in kl_keys)

    rep = {"sib_probe_component_calls": counts,
           "sib_probe_required": need,
           "sib_probe_kl_keys": kl_keys,
           "sib_probe_kl_value": kl_val,
           "sib_probe_haar_via_loss_dict": haar_via_loss,
           "sib_probe_modules_built_though_off": built_but_off}
    if missing:
        raise SystemExit(f"FATAL {label}: SIB components did not fire: "
                         f"{missing}; {rep}")
    if built_but_off:
        raise SystemExit(f"FATAL {label}: C4-clean declares these OFF but the "
                         f"model built them: {built_but_off}; {rep}")
    if arch == "dinov3" and not haar_via_loss:
        raise SystemExit(f"FATAL {label}: the DINOv3 Haar branch did not run "
                         f"(loss dict {kl_keys}); {rep}")
    if kl_val is None or not np.isfinite(kl_val) or kl_val <= 0:
        raise SystemExit(f"FATAL {label}: SIB produced no positive KL "
                         f"({kl_val}); {rep}")
    print(f"[armA] SIB probe PASS  {rep}", flush=True)
    return rep


def measure_overhead(trainer, arch, ds_train, bs, nw, seed, steps, device, label):
    """s/step with the method against s/step vanilla, same node, same batches.

    The question the operator asked is a RATIO, and a ratio taken against a
    number measured on another day, on another node, at another dataloader
    width is not the same measurement. So the vanilla side is rebuilt here --
    the published vanilla model of the same architecture, a fresh optimiser,
    the same batches in the same order -- and both sides are timed in the same
    process, back to back.

    The method side is timed FIRST so that any page-cache warming from reading
    the same patches twice works against the method, not for it. Both sides
    discard a warm-up step, which pays cudnn autotune.
    """
    import torch
    from torch.utils.data import DataLoader

    def _loader():
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(P.TrainTargets(ds_train), batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=True, drop_last=True,
                          generator=g, worker_init_fn=P.dl_worker_init)

    def _time(model, criterion, optimizer, n):
        model.train()
        per = []
        it = iter(_loader())
        for i in range(n + 1):
            try:
                b = next(it)
            except StopIteration:
                it = iter(_loader())
                b = next(it)
            torch.cuda.synchronize()
            t = time.time()
            images = b["image"].to(device)
            masks = b["mask"].to(device)
            out = model(images)
            lo = criterion(out, masks)
            lo = lo["total"] if isinstance(lo, dict) else lo
            optimizer.zero_grad()
            lo.backward()
            optimizer.step()
            torch.cuda.synchronize()
            per.append(time.time() - t)
        return per[1:]                      # drop the cudnn warm-up step

    method_s = _time(trainer.model, trainer.criterion, trainer.optimizer, steps)

    # ---- the vanilla side, rebuilt ----------------------------------------
    import torch.optim as optim
    a = trainer.args
    # The vanilla side must carry the SAME ignore_index=255 loss as the method
    # side: the train loader hands `mask_ignore`, whose unseen pixels are 255,
    # and a 2-class CrossEntropyLoss without ignore_index meets that target
    # with a device-side assert rather than an error message.
    import torch.nn as nn
    if arch == "mamnet":
        from models.mamnet import MAMNet
        from utils.losses import MAMNetLoss
        # MAMNet takes `use_contrast`, not `in_channels`: it picks
        # ResNet34Encoder4Ch itself. Built exactly as train.py:176 builds it.
        vm = MAMNet(num_classes=getattr(a, "num_classes", 2),
                    pretrained=bool(getattr(a, "pretrained", True)),
                    use_aux=True,
                    use_contrast=bool(ARCH_SPEC[arch]["use_contrast"])).to(device)
        vc = MAMNetLoss(aux_weight=getattr(a, "aux_weight", 0.4))
        vc.criterion = nn.CrossEntropyLoss(ignore_index=255)
        vo = optim.Adam(vm.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    else:
        from dinov3_model import DINOv3ShadowDetector
        vm = DINOv3ShadowDetector(
            num_classes=getattr(a, "num_classes", 2), model_name=a.model_name,
            weights_path=a.weights_path,
            pretrained=bool(getattr(a, "pretrained", True)),
            frozen_stages=int(getattr(a, "frozen_stages", -1))).to(device)
        vc = nn.CrossEntropyLoss(ignore_index=255)
        vo = optim.AdamW(vm.parameters(), lr=a.lr, weight_decay=a.weight_decay,
                         betas=(0.9, 0.999))
    vanilla_s = _time(vm, vc, vo, steps)
    del vm, vc, vo
    torch.cuda.empty_cache()

    m = float(np.median(method_s))
    v = float(np.median(vanilla_s))
    rep = {
        "overhead_steps_timed": steps,
        "overhead_batch_size": bs,
        "overhead_s_per_step_method": round(m, 5),
        "overhead_s_per_step_vanilla": round(v, 5),
        "overhead_ratio": round(m / v, 4) if v > 0 else None,
        "overhead_s_per_step_method_all": [round(x, 5) for x in method_s],
        "overhead_s_per_step_vanilla_all": [round(x, 5) for x in vanilla_s],
        "overhead_exceeds_1p5x": bool(v > 0 and (m / v) > 1.5),
        "overhead_basis": (
            "median of `steps` timed train steps after one discarded warm-up "
            "step, both sides in ONE process on ONE node over the SAME batches "
            "in the same order, method side timed first. Vanilla side is the "
            "published model of the same architecture with a fresh optimiser; "
            "the FDA-transformed loader is shared by both sides, so this "
            "isolates the model/loss cost and NOT the CPU cost of the FDA "
            "transform -- that shows up in the epoch rate, reported beside it."),
        "overhead_note_vs_recorded_vanilla": {
            "recorded_min_per_epoch_eval_off": {
                "mamnet": P.MAMNET_MIN_PER_EPOCH_EVAL_OFF, "dinov3": 2.9038},
            "recorded_s_per_step_total": {"mamnet": 0.70947, "dinov3": 0.39869},
            "source": ("logs/armA_evalfree_equivalence.json arm b (mamnet) and "
                       "logs/armA_dinov3_probe.json (dinov3), both at the full "
                       "3500/900 cell shape; the in-process ratio above is the "
                       "number to use, these are the link-sizing basis"),
        },
    }
    print(f"[armA] method overhead {label}: method {m:.5f} s/step  vanilla "
          f"{v:.5f} s/step  ratio {rep['overhead_ratio']}", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mamnet", choices=sorted(ARCH_SPEC),
                    help="oglanet is intentionally absent: dropped from arm A")
    ap.add_argument("--method", default=None, choices=sorted(M.METHODS),
                    help="required for every cell except a "
                         "--continue_vanilla continuation")
    ap.add_argument("--run_id", required=True)
    # ---- WAVE 2 / D-A2-1: see armA_pilot_train.py for the full note.
    # Defaults to "" so every pre-existing invocation resolves as before.
    ap.add_argument("--run_id_suffix", default="",
                    help="appended to the derived run_id, so a retrain "
                         "takes a NEW dump path and the original dump is "
                         "never opened for writing.")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--budget_dir", required=True)
    ap.add_argument("--patch_index", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--vanilla_dump_root", default=None,
                    help="where the fold's phase-2 vanilla dumps live; the test "
                         "patch-id identity check reads them")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--timeout_s", type=int, default=3600)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--dump_only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke_epochs", type=int, default=0)
    ap.add_argument("--smoke_train", type=int, default=0)
    ap.add_argument("--smoke_val", type=int, default=0)
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--overhead_steps", type=int, default=0,
                    help="smoke only: time this many train steps with the "
                         "method and the same many vanilla")
    ap.add_argument("--disable_train_eval", action="store_true")
    ap.add_argument("--train_eval_authority", default=None,
                    choices=["equivalence_pass", "code_audit"])
    ap.add_argument("--equiv_artifact", default=None)
    ap.add_argument("--train_eval_off_from_epoch", type=int, default=1)
    ap.add_argument("--train_eval_off_from_epoch_label", type=int, default=1)
    ap.add_argument("--probe_json", default=None)
    # ---- continuation of a ceiling-bound cell (HLT uniform rule) ----------
    ap.add_argument("--epochs_max", type=int, default=0,
                    help="the TRAIN LOOP ceiling. 0 keeps the recipe's "
                         "EPOCHS_MAX. Raising it does NOT reshape the LR "
                         "cosine; see --schedule_epochs.")
    ap.add_argument("--schedule_epochs", type=int, default=0,
                    help="the denominator the LR cosine and the SIB VIB "
                         "warmup are shaped on. 0 keeps the recipe's "
                         "EPOCHS_MAX. Pinning this while --epochs_max rises "
                         "is what keeps the LR schedule the one the resumed "
                         "trajectory was already on.")
    ap.add_argument("--resume_from_best", action="store_true",
                    help="when no checkpoint_epoch_*.pth survives, resume "
                         "from checkpoint_best.pth")
    ap.add_argument("--continuation", action="store_true")
    # D-A4-1 carve-out, ARM A SESSION A4. Default OFF, so every existing
    # invocation is unchanged; epoch_ceiling retains latest without it.
    ap.add_argument("--keep_latest", action="store_true",
                    help="retain checkpoint_latest.pth on completion "
                         "(D-A4-1); implied when stopped_on==epoch_ceiling")
    ap.add_argument("--prev_best_val", type=float, default=None)
    ap.add_argument("--prev_best_epoch", type=int, default=None)
    ap.add_argument("--prev_stopped_on", default=None)
    # ---- W1-C: continuing a cell this script did not originally train ------
    # Phase 3 trains LOCO method cells only, and both facts were hard-coded:
    # `--method` was required in {fda,sib} and `mode` was the literal "loco".
    # The continuation machinery below (--epochs_max / --schedule_epochs /
    # --resume_from_best / --continuation, the pinned LR denominator, the
    # patience reconstruction and the identity-vs-improvement re-dump) exists
    # ONLY here, while two of the five ceiling-bound cells it has to drive are
    # wave-0 VANILLA cells and one of those is an UPPER cell. Splitting the
    # five across two code paths would break the uniform rule the continuation
    # is defined by, so the two facts are made into arguments instead.
    #
    # BOTH DEFAULT TO THE CURRENT BEHAVIOUR. --continue_mode defaults to
    # "loco" and --continue_vanilla is off, so every existing invocation --
    # the twelve seed-0 method cells, W1-B's 24 seed fills, W1-A's 18 oglanet
    # cells -- resolves exactly as before. What changes is this file's sha256,
    # which those cells record as train_script_sha; the delta is recorded in
    # each CONTINUED cell's run.json as continuation_script_sha beside the
    # sha the original cell carried, so the lineage is explicit rather than
    # silently broken.
    ap.add_argument("--continue_mode", default="loco", choices=["loco", "upper"],
                    help="the cell's condition. Phase 3 trains loco only; "
                         "'upper' exists so W1-C can continue the one "
                         "ceiling-bound wave-0 UPPER cell under the same rule "
                         "as the other four. It does NOT train a new upper "
                         "bound: --continuation is required with it.")
    ap.add_argument("--continue_vanilla", action="store_true",
                    help="the cell carries no method: run_id says 'vanilla' "
                         "(loco) or 'none' (upper) and no FDA pool is built "
                         "and no SIB is installed. Requires --continuation.")
    args = ap.parse_args()

    if args.continue_mode != "loco" and not args.continuation:
        raise SystemExit("FATAL --continue_mode upper is only for "
                         "--continuation; phase 3 trains no new upper bound")
    if args.continue_vanilla and not args.continuation:
        raise SystemExit("FATAL --continue_vanilla is only for --continuation; "
                         "phase 3 trains no new vanilla cell")
    if args.method is None and not args.continue_vanilla:
        raise SystemExit("FATAL --method is required unless --continue_vanilla")

    # The loop ceiling and the schedule denominator are separate numbers. The
    # published CosineWarmupScheduler takes total_epochs=args.epochs, so
    # raising the ceiling alone would move every remaining epoch onto a
    # different, much higher, point of the cosine -- a second variable. The
    # continuation pins the denominator at the recipe's value and clamps the
    # cosine's progress at 1.0, so replayed epochs get exactly the LR they
    # would have got, and epochs past the original ceiling sit at min_lr.
    EPOCHS_CEIL = args.epochs_max or EPOCHS_MAX
    SCHED_EPOCHS = args.schedule_epochs or EPOCHS_MAX
    if args.continuation:
        if args.prev_best_val is None or args.prev_best_epoch is None:
            raise SystemExit("FATAL --continuation requires --prev_best_val "
                             "and --prev_best_epoch")
        if args.dump_only:
            raise SystemExit("FATAL --continuation and --dump_only are "
                             "mutually exclusive")

    t0 = time.time()
    arch, method = args.arch, args.method
    spec = ARCH_SPEC[arch]
    recipe_config = BASE / spec["recipe"]
    smoke = bool(args.smoke_epochs or args.smoke_train or args.smoke_val
                 or args.smoke_test)
    if smoke and args.out_root is None:
        raise SystemExit("FATAL --smoke_* requires --out_root")
    if args.overhead_steps and not smoke:
        raise SystemExit("FATAL --overhead_steps is a smoke-only measurement; "
                         "it builds a second model and would perturb a real cell")
    if args.disable_train_eval and not args.train_eval_authority:
        raise SystemExit(
            "FATAL --disable_train_eval requires --train_eval_authority.")

    # Phase 3 is LOCO-only BY CONSTRUCTION for every cell it trains from
    # scratch; --continue_mode can only be something else under --continuation,
    # which is asserted at parse time above.
    mode = args.continue_mode
    if args.continue_vanilla:
        # The run_id grammar the wave-0 vanilla cells were written under.
        method = "vanilla" if mode == "loco" else "none"
    expect = f"{arch}-{method}-{mode}-r{RUNG}-{args.holdout.lower()}-{RES}-s{args.seed}"
    if args.run_id_suffix:
        expect = f"{expect}-{args.run_id_suffix}"
    if args.run_id != expect:
        raise SystemExit(f"FATAL run_id {args.run_id} != derived {expect}")

    if args.continue_vanilla:
        ta = "source_only" if mode == "loco" else "target_labels"
        ta_detail = (
            "train and val draw only from the two source cities' AOIs; no "
            "held-out-city pixel is touched before test-time inference"
            if mode == "loco" else
            "within-city upper bound: train and val are labelled patches from "
            "the held-out city's own AOIs, disjoint from its test AOIs")
        use_fda = False
    else:
        ta, ta_detail = M.method_target_access(arch, method)
        use_fda = fda_is_active(arch, method)
    touches = touches_heldout_imagery(arch, method) if mode != "upper" else False
    if (ta == "unlabeled_target") != touches:
        raise SystemExit(
            f"FATAL target_access {ta!r} disagrees with "
            f"touches_heldout_imagery={touches} (fda_active={use_fda}) for "
            f"{arch}/{method}; the manifest and the code path must not "
            f"disagree about whether held-out imagery is touched")

    dump_dir = ((Path(args.out_root) if args.out_root else BASE / "dumps")
                / DATASET / args.run_id)
    if args.continuation and (dump_dir / ".CONTINUED").exists():
        print(f"[{args.run_id}] .CONTINUED exists at {dump_dir} - the "
              f"continuation is finished; nothing to do")
        return
    if args.continuation and not (dump_dir / ".DONE").exists():
        raise SystemExit(f"FATAL --continuation but there is no prior dump to "
                         f"continue from at {dump_dir}")
    if (dump_dir / ".DONE").exists() and not args.force and not args.continuation:
        print(f"[{args.run_id}] .DONE exists at {dump_dir} - nothing to do")
        return

    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet", dump_dir / "probs.npz",
        dump_dir / "run.json", dump_dir / ".DONE",
        dump_dir / ".CONTINUED"], label="dump")
    if args.probe_json:
        armc_runtime.preflight_write_targets(BASE, [Path(args.probe_json)],
                                             label="probe")

    man_path = Path(args.budget_dir) / f"{args.holdout}_{mode}.json"
    man = json.loads(man_path.read_text())
    if man["budget_rule_sha"] != seo_budget.budget_rule_sha():
        raise SystemExit(
            f"FATAL manifest budget_rule_sha {man['budget_rule_sha'][:16]} != the "
            f"live rule {seo_budget.budget_rule_sha()[:16]}; rebuild the manifest")
    man_sha = hashlib.sha256(man_path.read_bytes()).hexdigest()

    idx = json.loads(Path(args.patch_index).read_text())
    out_dir_parent = Path(args.ckpt_root) / args.run_id

    print("=" * 78)
    print(f"ARM A / S-EO PHASE 3  run_id   {args.run_id}")
    print(f"  arch / method           {arch} / {method}")
    print(f"  dropped from this arm   {sorted(ARCH_DROPPED)}")
    print(f"  condition / holdout     {mode}"
          f"{' (phase 3 trains LOCO only; this is a W1-C continuation of an existing cell)' if mode != 'loco' else ' (phase 3 is LOCO-only)'}"
          f" / {args.holdout}")
    print(f"  sources                 {man['source_cities']}")
    print(f"  budget                  train {man['n_train']}  val {man['n_val']}  "
          f"test {man['n_test']} (full, {man['n_test_aoi']} AOIs)")
    print(f"  manifest                {man_path}")
    print(f"  manifest sha256         {man_sha[:16]}")
    print(f"  budget_rule_sha         {man['budget_rule_sha'][:16]}  MATCHES live rule")
    print(f"  target_access           {ta}")
    print(f"    detail                {ta_detail}")
    print(f"  FDA component           {'ACTIVE  L=' + str(fda_L_for(arch, method)) if use_fda else 'not used'}")
    if method == "sib":
        s = M.SIB_SPEC[arch]
        print(f"  SIB components active   {s['components']}")
        print(f"  SIB components off      {s['components_off']}")
        print(f"  SIB published config    {s['published_config']}")
    print(f"  base recipe config      {recipe_config}  (phase-2 recipe)")
    print(f"  checkpoints             {out_dir_parent}")
    print(f"  dump                    {dump_dir}")
    print(f"  train-side evaluator    "
          f"{'DISABLED under ' + str(args.train_eval_authority) if args.disable_train_eval else 'ON'}")
    print(f"  slurm job               {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    if smoke:
        print(f"  SMOKE  epochs<={args.smoke_epochs} train<={args.smoke_train} "
              f"val<={args.smoke_val} test<={args.smoke_test} "
              f"overhead_steps={args.overhead_steps}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    import torch                                                # noqa: E402
    from torch.utils.data import DataLoader                     # noqa: E402
    seed_sites = P.seed_everything(args.seed)

    tr_ids, va_ids, te_ids = man["train"], man["val"], man["test"]
    if smoke:
        tr_ids = tr_ids[:args.smoke_train or len(tr_ids)]
        va_ids = va_ids[:args.smoke_val or len(va_ids)]
        te_ids = te_ids[:args.smoke_test or len(te_ids)]

    # ---- the FDA target pool, built and asserted BEFORE anything expensive --
    fda_mod = fda_path = None
    fda_transform = None
    fda_prov = {"fda_active": False}
    target_crops, target_ids = [], []
    if use_fda:
        fda_mod, fda_path = M._published_fda_module()
        forbidden = set(man["train"]) | set(man["val"]) | set(man["test"])
        target_crops, target_ids, fda_prov = M.build_fda_target_pool(
            budget_dir=args.budget_dir, holdout=args.holdout,
            patch_index=idx, img_root=args.img_root, seed=args.seed,
            max_images=fda_pool_max_for(arch, method),
            forbidden_ids=forbidden, label=args.run_id)
        fda_transform = fda_mod.FDATransform(
            target_crops, L=fda_L_for(arch, method), target_selection="random")
        fda_prov.update({
            "fda_active": True,
            "fda_L": fda_L_for(arch, method),
            "fda_module_path": str(Path(fda_path).relative_to(BASE)),
            "fda_module_sha256": sha256_file(Path(fda_path)),
            "fda_application_point": (
                "input, on the uint8 RGB crop, BEFORE the RMS-contrast channel, "
                "BEFORE ImageNet normalisation and BEFORE the flip "
                "augmentation; TRAIN split only"),
            "fda_target_selection": "uniform random per __getitem__ call",
            "fda_val_test_transformed": False,
            "fda_published_source": (M.FDA_SPEC[arch]["published_config"]
                                     if method == "fda"
                                     else M.SIB_SPEC[arch]["published_config"]),
        })
        print(f"[armA] FDA target pool: {fda_prov['fda_target_pool_n']} "
              f"unlabelled {args.holdout} TRAIN patches of "
              f"{fda_prov['fda_target_pool_candidates']} candidates; ids sha "
              f"{fda_prov['fda_target_pool_ids_sha256'][:16]}; L="
              f"{fda_prov['fda_L']}", flush=True)

    uc = spec["use_contrast"]
    ds_train, rows_tr, bases_tr, parts_tr = build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "train", tr_ids,
        True, "train", uc, fda=fda_transform)
    ds_val, rows_va, bases_va, parts_va = build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "val", va_ids,
        False, "val", uc)
    ds_test, rows_te, bases_te, parts_te = build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "test", te_ids,
        False, "test", uc)

    leak = assert_no_leakage(bases_tr + bases_va + bases_te)
    pidx = P.patch_index_map(bases_tr + bases_va + bases_te)

    aoi_tr = {r["aoi_id"] for r in rows_tr}
    aoi_va = {r["aoi_id"] for r in rows_va}
    aoi_te = {r["aoi_id"] for r in rows_te}
    for a_, b_, lbl in ((aoi_tr, aoi_va, "train/val"), (aoi_tr, aoi_te, "train/test"),
                        (aoi_va, aoi_te, "val/test")):
        if a_ & b_:
            raise SystemExit(f"FATAL AOI overlap {lbl}: {sorted(a_ & b_)[:5]}")

    train_sun = np.array([r["sun_el"] for r in rows_tr], dtype=np.float64)
    train_sun_mean, train_sun_sd = float(train_sun.mean()), float(train_sun.std(ddof=1))
    print(f"\nTRAIN SUN ELEVATION  mean {train_sun_mean:.3f}  sd {train_sun_sd:.3f}  "
          f"n {len(train_sun)}", flush=True)
    print(f"LEAKAGE CHECK  {leak}", flush=True)

    cfg = json.loads(recipe_config.read_text())
    arch_root = BASE / "python" / spec["dir"]
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    for _m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_m]
    archtrain = __import__(spec["train_module"])

    disabled = armc_runtime.silence_worker_side_effects(archtrain)
    pruner = P.install_checkpoint_pruner(archtrain)

    from argparse import Namespace
    saved_argv, sys.argv = sys.argv, [spec["train_module"]]
    try:
        defaults = vars(archtrain.get_args())
    finally:
        sys.argv = saved_argv
    defaults_filled = {k: defaults[k] for k in sorted(set(defaults) - set(cfg))}
    a = {**defaults, **cfg}
    rerouted = P.route_config_to_this_cluster(a, arch)
    a["output_dir"] = str(out_dir_parent)
    a["device"] = "cuda"
    a["eval_only"] = False
    a["resume"] = None
    a["num_workers"] = args.num_workers
    a["early_stopping_patience"] = PATIENCE_EPOCHS
    a["epochs"] = args.smoke_epochs or EPOCHS_CEIL
    a["save_freq"] = 1
    a["seed"] = args.seed
    a["armA_arch"] = arch
    a["armA_method"] = method
    a["armA_run_id"] = args.run_id
    a["armA_dataset"] = DATASET
    a["armA_n_train"] = len(ds_train)
    a["armA_n_val"] = len(ds_val)
    trainer_args = Namespace(**a)

    with seo_traineval.LoadReport() as load_rep:
        trainer = archtrain.Trainer(trainer_args)
    weight_load_vanilla = load_rep.summary()
    run_out = Path(trainer.output_dir)

    # Order is load-bearing: the ignore_index loss goes in FIRST, then the
    # method wraps it, so the KL rides on a loss that already drops unseen
    # pixels rather than the other way round.
    loss_note = P.set_ignore_index_loss(trainer, arch)

    method_prov = {"method": method}
    sib_state = None
    if args.continue_vanilla:
        method_prov.update({
            "method_installed": False,
            "continue_vanilla": True,
            "continue_vanilla_note": (
                "this cell carries no method: no FDA target pool is built and "
                "no SIB is installed. Only the ignore_index loss above is "
                "applied, which is exactly what the wave-0 vanilla trainer "
                "applied. The continuation changes the cell's LENGTH, not its "
                "recipe."),
        })
    elif method == "sib":
        with seo_traineval.LoadReport() as load_rep2:
            sib_prov, sib_state = M.install_sib(trainer, arch,
                                                epochs_max=SCHED_EPOCHS)
        method_prov.update(sib_prov)
        method_prov["sib_pretrained_weight_load"] = load_rep2.summary()
        for c in method_prov["sib_pretrained_weight_load"]:
            print(f"[armA] SIB load_state_dict on {c.get('module')}: offered "
                  f"{c.get('n_tensors_offered')}, module holds "
                  f"{c.get('n_tensors_in_module')}, missing "
                  f"{c.get('n_missing_keys')}, unexpected "
                  f"{c.get('n_unexpected_keys')}", flush=True)
    else:
        method_prov.update({
            "fda_published_config": M.FDA_SPEC[arch]["published_config"],
            "fda_published_early_stop_patience":
                M.FDA_SPEC[arch]["published_early_stop_patience"],
            "fda_early_stop_patience_used": PATIENCE_EPOCHS,
            "fda_early_stop_patience_note": (
                "the published FDA cells early-stop at 15 (MAMNet) and 10 "
                "(DINOv3); arm A uses the flat patience 20 of its own recipe "
                "so the method cell and its vanilla comparator stop under the "
                "SAME rule. Carried as a declared difference, not silently."),
        })
    method_prov["target_access"] = ta
    method_prov["target_access_detail"] = ta_detail
    method_prov.update(fda_prov)

    train_eval_state = {"disabled": False,
                        "what": "train-side evaluator left ON (arm C recipe)"}
    equiv_sha = equiv_verdict = equiv_path = None
    if args.disable_train_eval:
        if not args.equiv_artifact:
            raise SystemExit("FATAL --disable_train_eval requires --equiv_artifact")
        ap_path = Path(args.equiv_artifact)
        if not ap_path.exists():
            raise SystemExit(f"FATAL equivalence artifact missing: {ap_path}")
        equiv = json.loads(ap_path.read_text())
        equiv_verdict = equiv.get("verdict")
        equiv_sha = hashlib.sha256(ap_path.read_bytes()).hexdigest()
        equiv_path = str(ap_path)
        if args.train_eval_authority == "equivalence_pass" and equiv_verdict != "PASS":
            raise SystemExit(f"FATAL authority=equivalence_pass but verdict is "
                             f"{equiv_verdict!r}")
        if args.train_eval_authority == "code_audit" and equiv_verdict == "FAIL":
            raise SystemExit("FATAL authority=code_audit is for an INCONCLUSIVE "
                             "proof, not a FAILED one.")
        train_eval_state = seo_traineval.disable_train_evaluator(trainer)
        train_eval_state["authority"] = args.train_eval_authority
        train_eval_state["equivalence_artifact"] = equiv_path
        train_eval_state["equivalence_sha256"] = equiv_sha
        train_eval_state["equivalence_verdict"] = equiv_verdict
        print(f"[armA] train-side evaluator DISABLED under authority "
              f"'{args.train_eval_authority}'; sha {equiv_sha[:16]}", flush=True)

    val_predpos = []
    predpos_probe = seo_traineval.install_val_predpos_probe(trainer, val_predpos)
    step_losses = []
    step_probe = None
    if args.probe_json:
        step_probe = seo_traineval.install_step_loss_probe(trainer, step_losses)

    # ---- LR schedule: pinned to the recipe's denominator, progress clamped
    # The published CosineWarmupScheduler is STATELESS -- step(epoch) derives
    # the LR from the epoch index alone, so there is no scheduler state in the
    # checkpoint to restore and none is needed. What DOES have to be
    # reconstructed is its shape: total_epochs comes from args.epochs, so a
    # raised ceiling would silently move the cosine. This pins total_epochs at
    # SCHED_EPOCHS and clamps progress at 1.0, so the replayed epochs get
    # exactly the LR the original trajectory gave them and every epoch beyond
    # the original ceiling holds at min_lr instead of the cosine turning back
    # up (progress > 1 makes cos(pi*progress) rise again).
    import math                                                # noqa: E402
    sched = getattr(trainer, "scheduler", None)
    sched_pin = {"pinned": False, "reason": "not attempted"}
    if EPOCHS_CEIL != SCHED_EPOCHS:
        if sched is None or not hasattr(sched, "total_epochs"):
            raise SystemExit(
                f"FATAL --epochs_max {EPOCHS_CEIL} != --schedule_epochs "
                f"{SCHED_EPOCHS} but this scheduler "
                f"({type(sched).__name__}) exposes no total_epochs to pin. "
                f"Running would reshape the LR schedule, which is a second "
                f"variable. Refusing.")
        was_total = int(getattr(sched, "total_epochs"))
        sched.total_epochs = SCHED_EPOCHS
        _b = float(getattr(sched, "base_lr"))
        _m = float(getattr(sched, "min_lr"))
        _w = int(getattr(sched, "warmup_epochs"))
        _T = SCHED_EPOCHS
        _opt = sched.optimizer

        def _clamped_step(epoch, _s=sched):
            _s.current_epoch = epoch
            if epoch < _w:
                lr = _b * (epoch + 1) / _w
            else:
                prog = min(1.0, (epoch - _w) / float(_T - _w))
                lr = _m + (_b - _m) * 0.5 * (1.0 + math.cos(math.pi * prog))
            for g in _opt.param_groups:
                g["lr"] = lr
            return lr

        sched.step = _clamped_step
        sched_pin = {
            "pinned": True,
            "scheduler": type(sched).__name__,
            "total_epochs_was": was_total,
            "total_epochs_now": _T,
            "loop_ceiling": EPOCHS_CEIL,
            "warmup_epochs": _w, "base_lr": _b, "min_lr": _m,
            "rule": ("lr = min_lr + (base_lr-min_lr)*0.5*(1+cos(pi*progress)), "
                     "progress = min(1.0, (epoch-warmup)/(total_epochs-warmup))"),
            "clamp_note": ("progress is CLAMPED at 1.0. Without the clamp an "
                           "epoch past total_epochs would make the cosine "
                           "rise back toward base_lr. With it, every epoch "
                           "past the original ceiling trains at min_lr."),
            "unchanged_over_replayed_epochs": True,
        }
        print(f"[armA] LR schedule PINNED: total_epochs {was_total} -> {_T} "
              f"while the loop ceiling is {EPOCHS_CEIL}; progress clamped at "
              f"1.0", flush=True)

    resume_from = P.newest_epoch_checkpoint(run_out)
    resume_source = "periodic_checkpoint"
    if resume_from is None and args.resume_from_best:
        _b_ck = P.resolve_best_checkpoint(run_out)
        if _b_ck is None:
            raise SystemExit(f"FATAL --resume_from_best: no loadable best "
                             f"checkpoint under {run_out}")
        resume_from = str(_b_ck)
        resume_source = "checkpoint_best"
    resumed_epoch = 0
    resume_rec = {"resumed": False}
    if resume_from:
        print(f"RESUME from {resume_from}  (source: {resume_source})", flush=True)
        trainer.load_checkpoint(resume_from)
        resumed_epoch = int(getattr(trainer, "start_epoch", 0))
        ck_epoch = resumed_epoch - 1
        ewi_in_ckpt = int(getattr(trainer, "epochs_without_improvement", 0))
        ewi_recon = ewi_in_ckpt
        recon_note = ("resumed from a periodic checkpoint, whose stored "
                      "epochs_without_improvement is already the live count; "
                      "no reconstruction needed")
        if resume_source == "checkpoint_best":
            # Patience CONTINUES FROM THE BEST, it does not reset. Because the
            # only surviving checkpoint IS the best, current_epoch == best
            # epoch, so the reconstruction (current_epoch - best_epoch) is 0 --
            # which is also what the best checkpoint stores, since it was
            # written at the instant the counter was zeroed. The two agree; the
            # reconstruction is applied explicitly anyway so the rule, not the
            # coincidence, is what governs.
            ewi_recon = max(0, ck_epoch - ck_epoch)
            trainer.epochs_without_improvement = ewi_recon
            recon_note = ("epochs_without_improvement reconstructed as "
                          "(current_epoch - best_epoch) at resume. The resume "
                          "source IS checkpoint_best, so current_epoch == "
                          "best_epoch and the reconstruction is 0. The "
                          "17/8 epochs of no-improvement the original "
                          "trajectory had accumulated by its ceiling are NOT "
                          "carried: that trajectory's epochs 84-99 no longer "
                          "exist on disk and are being replayed, not resumed.")
        lr_next = None
        if sched is not None and hasattr(sched, "total_epochs"):
            _wq = int(getattr(sched, "warmup_epochs"))
            _bq = float(getattr(sched, "base_lr"))
            _mq = float(getattr(sched, "min_lr"))
            _Tq = int(getattr(sched, "total_epochs"))
            if resumed_epoch < _wq:
                lr_next = _bq * (resumed_epoch + 1) / _wq
            else:
                _p = min(1.0, (resumed_epoch - _wq) / float(_Tq - _wq))
                lr_next = _mq + (_bq - _mq) * 0.5 * (1.0 + math.cos(math.pi * _p))
        resume_rec = {
            "resumed": True,
            "resume_source": resume_source,
            "resume_source_path": str(resume_from),
            "resume_epoch": ck_epoch,
            "resume_start_epoch_index": resumed_epoch,
            "resume_first_epoch_label": resumed_epoch + 1,
            "epochs_run_at_resume": len(getattr(trainer, "train_losses", []) or []),
            "best_val_metric_at_resume": float(
                getattr(trainer, "best_decision_miou", float("nan"))),
            "epochs_without_improvement_in_checkpoint": ewi_in_ckpt,
            "epochs_without_improvement_reconstructed": ewi_recon,
            "patience_reconstruction_note": recon_note,
            "lr_at_first_continuation_epoch": lr_next,
            "lr_schedule_state_note": (
                "the published CosineWarmupScheduler holds no state beyond "
                "current_epoch: step(epoch) computes the LR from the epoch "
                "index and the four constants. Nothing to restore from the "
                "checkpoint; the schedule is reconstructed by the loop "
                "stepping to the resume epoch, and its SHAPE is pinned -- see "
                "lr_schedule."),
            "resume_note": (
                "the best-only prune that followed this cell's dump-only "
                "recovery removed checkpoint_epoch_100.pth, so "
                "checkpoint_best.pth is the newest surviving checkpoint. The "
                "continuation therefore restarts from the BEST epoch, not "
                "from the epoch the original run stopped at: epochs after the "
                "best are REPLAYED on a new trajectory (new dataloader "
                "shuffle order, new nondeterministic kernels), not resumed."),
        }
        print(f"[armA] resume: epoch {ck_epoch} -> first continuation epoch "
              f"label {resumed_epoch + 1}; epochs_without_improvement "
              f"{ewi_in_ckpt} (checkpoint) -> {ewi_recon} (reconstructed); "
              f"lr {lr_next}", flush=True)
    first_epoch_this_link = resumed_epoch + 1
    train_eval_off_from_epoch = (args.train_eval_off_from_epoch
                                 if train_eval_state.get("disabled") else None)
    train_eval_off_from_epoch_label = (args.train_eval_off_from_epoch_label
                                       if train_eval_state.get("disabled") else None)

    g = torch.Generator()
    g.manual_seed(args.seed)
    bs = trainer_args.batch_size
    nw = args.num_workers
    trainer.dataloaders = {
        "train": DataLoader(P.TrainTargets(ds_train), batch_size=bs, shuffle=True,
                            num_workers=nw, pin_memory=True, drop_last=True,
                            generator=g, worker_init_fn=P.dl_worker_init),
        "val": DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=nw,
                          pin_memory=True, worker_init_fn=P.dl_worker_init),
        "test": DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=nw,
                           pin_memory=True, worker_init_fn=P.dl_worker_init),
    }
    steps_per_epoch = len(trainer.dataloaders["train"])
    val_batches = len(trainer.dataloaders["val"])
    print(f"\nDATALOADERS  train={len(ds_train)}  val={len(ds_val)}  "
          f"test={len(ds_test)}  steps/epoch={steps_per_epoch}  "
          f"val_batches={val_batches}", flush=True)
    if not smoke:
        if len(ds_train) != seo_budget.N_TRAIN:
            raise SystemExit(f"FATAL train {len(ds_train)} != {seo_budget.N_TRAIN}")
        if len(ds_val) != seo_budget.N_VAL:
            raise SystemExit(f"FATAL val {len(ds_val)} != {seo_budget.N_VAL}")
        if len(ds_test) != man["n_test"]:
            raise SystemExit(f"FATAL test {len(ds_test)} != {man['n_test']}")

    # ---- the method probe: SMOKE ONLY --------------------------------------
    probe_rep = {"method_probe": "not run (real cell; the probe pushes a batch "
                                 "through the model and would touch BatchNorm "
                                 "running statistics before epoch 1)"}
    if smoke:
        if use_fda:
            probe_rep = {"method_probe": "fda"}
            probe_rep.update(probe_fda(fda_mod, fda_transform, target_crops,
                                       ds_train, fda_L_for(arch, method),
                                       args.run_id))
        if method == "sib":
            b0 = next(iter(trainer.dataloaders["train"]))
            r = probe_sib(trainer.model, arch, sib_state, b0, trainer.device,
                          args.run_id)
            probe_rep["method_probe"] = ("sib+fda" if use_fda else "sib")
            probe_rep.update(r)

    overhead = {"overhead_measured": False}
    if smoke and args.overhead_steps:
        overhead = measure_overhead(trainer, arch, ds_train, bs, nw, args.seed,
                                    args.overhead_steps, trainer.device,
                                    args.run_id)
        overhead["overhead_measured"] = True

    train_wall = 0.0
    if not args.dump_only:
        t_train = time.time()
        try:
            trainer.train()
        except Exception as e:
            oom = ("out of memory" in str(e).lower()
                   or type(e).__name__ == "OutOfMemoryError")
            if oom:
                print("!" * 78, flush=True)
                print(f"[armA] CUDA OOM at batch_size {bs} on {arch}/{method}. "
                      f"STOPPING. The batch size is NOT being cut -- that is a "
                      f"recipe change and it is the operator's call.", flush=True)
                print("!" * 78, flush=True)
            raise
        train_wall = time.time() - t_train

    best_ckpt = P.resolve_best_checkpoint(run_out)
    if best_ckpt is None:
        raise SystemExit(f"FATAL no loadable best checkpoint under {run_out}")
    ck = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    best_epoch = int(ck["epoch"])
    # ARM A WAVE 2f, ADDITIVE. The three published IIM trainers
    # checkpoint their per-epoch history under `train_total`, not
    # `train_losses` (mamnet/train_iim.py:870, oglanet/train_iim.py:915,
    # dinov3/train_dinov3_iim.py:842). The PUBLISHED key is tried first,
    # so every non-IIM family resolves byte-identically to before; the
    # fallback fires only where the published key is genuinely absent.
    _ck_epoch_hist = ck.get("train_losses") or ck.get("train_total") or []
    epochs_run = len(_ck_epoch_hist) or len(trainer.train_losses)
    best_val_metric = float(ck.get("best_decision_miou", float("nan")))
    del ck
    epochs_run_total = len(getattr(trainer, "train_losses", []) or []) or epochs_run
    steps_to_best = best_epoch * steps_per_epoch
    val_hist = list((getattr(trainer, "val_metrics_history", {}) or {}).get("mIOU", []))
    ewi = int(getattr(trainer, "epochs_without_improvement", -1))
    # ---- ARM A BLOCK 1, ADDITIVE: stopped_on compares a LABEL, not a count.
    # The v1 expression compared `epochs_run_total` -- the LENGTH of the
    # trainer's per-epoch history list -- against EPOCHS_CEIL, which is an
    # epoch LABEL. The two are not the same quantity, and for the three IIM
    # trainers they are routinely far apart, because those trainers append to
    # `train_total` and the length recovered here comes from the BEST
    # checkpoint's history, not from the last epoch actually executed.
    #
    # Worked example, oglanet-iim-loco-r3500-ucsd-hr-s2 (job 21697966,
    # armA_w2g_..._link1.out): the log prints `Epoch 89/100` through
    # `Epoch 100/100` -- the cell ran to the ceiling exactly -- and then
    # `BEST epoch 100  epochs_run 91 ... stopped_on link_boundary_or_incomplete`,
    # because 91 >= 99 is False. `best_epoch 100` with `epochs_run 91` is
    # itself impossible for a true epoch count and is the tell.
    #
    # The published loop is `for epoch in range(self.start_epoch,
    # self.args.epochs)` (mamnet/train_iim.py:954, oglanet/train_iim.py:1012,
    # dinov3/train_dinov3_iim.py:919) and it breaks ONLY on patience. So the
    # last epoch LABEL reached is the loop's own ceiling whenever the loop had
    # anything to run and patience did not fire; and when start_epoch is
    # already at the ceiling the cell is a genuine link boundary that executed
    # no epoch at all.
    #
    # v1's value is preserved verbatim as stopped_on_v1 and is written into
    # run.json under legacy_keys; nothing that already reads epochs_run_total,
    # epochs_run or epochs_max sees any change.
    _loop_ceiling_label = int(
        getattr(getattr(trainer, "args", None), "epochs", 0) or EPOCHS_CEIL)
    _start_idx = int(getattr(trainer, "start_epoch", 0))
    _had_epochs_to_run = _start_idx < _loop_ceiling_label
    last_epoch_label = _loop_ceiling_label if _had_epochs_to_run else _start_idx

    stopped_on_v1 = ("patience" if ewi >= PATIENCE_EPOCHS else
                     "epoch_ceiling" if epochs_run_total >= EPOCHS_CEIL - 1 else
                     "link_boundary_or_incomplete")
    stopped_on = ("patience" if ewi >= PATIENCE_EPOCHS else
                  "epoch_ceiling" if last_epoch_label >= _loop_ceiling_label else
                  "link_boundary_or_incomplete")
    stopped_on_basis = {
        "rule": ("patience if ewi >= PATIENCE_EPOCHS, else epoch_ceiling if the "
                 "last epoch LABEL reached >= the loop's own ceiling, else "
                 "link_boundary_or_incomplete"),
        "last_epoch_label": last_epoch_label,
        "loop_ceiling_label": _loop_ceiling_label,
        "arm_epochs_ceil": int(EPOCHS_CEIL),
        "loop_ceiling_is_arm_ceiling": bool(_loop_ceiling_label == int(EPOCHS_CEIL)),
        "start_epoch_index_at_resume": _start_idx,
        "had_epochs_to_run": bool(_had_epochs_to_run),
        "epochs_without_improvement_at_stop": int(ewi),
        "patience_epochs": int(PATIENCE_EPOCHS),
        "epochs_run_total_is_a_list_length_not_a_label": True,
        "v1_expression": ("epochs_run_total >= EPOCHS_CEIL - 1, which compared a "
                          "history-list length against an epoch label"),
        "changed_vs_v1": bool(stopped_on != stopped_on_v1),
    }
    if stopped_on != stopped_on_v1:
        print(f"[armA/block1] stopped_on RELABELLED {stopped_on_v1!r} -> "
              f"{stopped_on!r}  (last_epoch_label {last_epoch_label} >= "
              f"loop_ceiling {_loop_ceiling_label}; epochs_run_total "
              f"{epochs_run_total} was the list length)", flush=True)
    print(f"\nBEST epoch {best_epoch}  epochs_run {epochs_run_total}  "
          f"steps_to_best {steps_to_best}  epochs_without_improvement {ewi}  "
          f"stopped_on {stopped_on}  "
          f"best_val_decision_mIoU {best_val_metric:.4f}", flush=True)

    # ---- continuation: does the new best supersede the old dump? --------
    cont_record = {}
    cont_swap = None
    if args.continuation:
        prev_best = float(args.prev_best_val)
        improved = bool(best_val_metric > prev_best)
        epochs_at_resume = int(resume_rec.get("epochs_run_at_resume", 0))
        cont = {
            "is_continuation": True,
            "continue_mode": mode,
            "continue_vanilla": bool(args.continue_vanilla),
            "continuation_script_sha": sha256_file(Path(__file__).resolve()),
            "continuation_script_sha_note": (
                "armA_phase3_train.py's sha256 AS IT RAN THIS CONTINUATION. It "
                "differs from the train_script_sha the ORIGINAL cell recorded, "
                "because the continuation path lived only in this file while "
                "two of the five ceiling-bound cells were wave-0 vanilla cells "
                "and one of those was an upper cell. The change is ADDITIVE -- "
                "--continue_mode defaults to 'loco' and --continue_vanilla is "
                "off, so every pre-existing invocation resolves exactly as "
                "before -- but the sha moved, and that delta is recorded here "
                "rather than left for a future reader to discover. The "
                "original cell's own train_script_sha is preserved untouched "
                "in the dump this continuation supersedes or appends to."),
            "original_trainer": (
                "armA_pilot_train.py / armA_wave0_train.py for the wave-0 "
                "vanilla cells; armA_phase3_train.py or armA_wave1_train.py "
                "for the wave-1 method cells. The continuation runs THIS "
                "file for all five, which is what makes the rule uniform."),
            "rule": ("HLT-signed uniform rule: every ceiling-bound cell in "
                     "the SIB-6/FDA-6 family continues to convergence. No "
                     "retraining from scratch."),
            "prev_stopped_on": args.prev_stopped_on,
            "prev_best_epoch": args.prev_best_epoch,
            "prev_best_val_metric": prev_best,
            "new_best_epoch": best_epoch,
            "new_best_val_metric": best_val_metric,
            "best_improved": improved,
            "best_delta": best_val_metric - prev_best,
            "epochs_ceiling_was": 100,
            "epochs_ceiling_now": EPOCHS_CEIL,
            "patience": PATIENCE_EPOCHS,
            "epochs_run_in_continuation": epochs_run_total - epochs_at_resume,
            "epochs_run_total": epochs_run_total,
            "stopped_on": stopped_on,
            "epochs_without_improvement_at_stop": ewi,
            "resume": resume_rec,
            "lr_schedule": sched_pin,
            "slurm_job_id_continuation": os.environ.get("SLURM_JOB_ID"),
            "node_continuation": os.environ.get("SLURMD_NODENAME"),
        }
        print("\n" + "-" * 78, flush=True)
        print(f"[continuation] prev best {prev_best:.6f} @ epoch "
              f"{args.prev_best_epoch}   new best {best_val_metric:.6f} @ "
              f"epoch {best_epoch}   improved={improved}", flush=True)
        print(f"[continuation] epochs in continuation "
              f"{cont['epochs_run_in_continuation']}   stopped_on {stopped_on}"
              f"   epochs_without_improvement {ewi}", flush=True)
        print("-" * 78, flush=True)

        if not improved:
            # The existing dump STANDS. The parquet and probs.npz are the
            # artefacts of the very checkpoint that is still best, so
            # rewriting them would only re-run inference on the same weights.
            # Only run.json gains the continuation record.
            cont["dump_action"] = (
                "none. The best checkpoint never improved during the "
                "continuation, so the existing perimage.parquet and "
                "probs.npz remain the dump of the still-best checkpoint and "
                "are not rewritten. run.json is appended to in place.")
            rj = dump_dir / "run.json"
            d = json.loads(rj.read_text())
            d["continuation"] = cont
            d["stopped_on_continuation"] = stopped_on
            d["epochs_run_total"] = epochs_run_total
            d["epochs_run"] = epochs_run_total
            d["epochs_max"] = EPOCHS_CEIL
            d["epochs_without_improvement_at_stop"] = ewi
            d["early_stopped"] = bool(stopped_on == "patience")
            d["continuation_written_utc"] = datetime.now(
                timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            fl = list(d.get("sanity_flags") or [])
            fl = [f for f in fl if not f.startswith("ran to the epoch ceiling")]
            fl.append(f"continued past the original 100-epoch ceiling to "
                      f"{EPOCHS_CEIL}; stopped_on {stopped_on} after "
                      f"{cont['epochs_run_in_continuation']} further epochs "
                      f"with no improvement on the decision metric")
            d["sanity_flags"] = fl
            tmp = Path(str(rj) + ".new")
            tmp.write_text(json.dumps(d, indent=2, default=str))
            tmp.replace(rj)
            print(f"[continuation] run.json appended in place -> {rj}",
                  flush=True)

            removed = []
            # ---- D-A4-1 CARVE-OUT (ARM A SESSION A4, 2026-09-03) -----------
            # D-A4-1 "Recurrence prevention" ruled that `checkpoint_latest.pth`
            # SURVIVES on any cell whose `stopped_on` is `epoch_ceiling`, and on any
            # cell run with `--keep_latest`; a cell that stopped on patience has a
            # resolved ceiling status and still deletes it. That ruling was recorded in
            # DECISIONS_armA_wave2.md but NEVER LANDED IN CODE -- every prune site
            # listed "checkpoint_latest.pth" unconditionally, and the dinov3 BSM
            # stage-2 log of 2026-09-03 shows it in `periodic ckpts removed`. This is
            # the carve-out, implemented as ruled. It does not widen deletion: it only
            # withholds it.
            _keep_latest = bool(getattr(args, "keep_latest", False)) or (
                stopped_on == "epoch_ceiling")
            _pats = ["checkpoint_epoch_*.pth", "checkpoint_epoch_*.pth.corrupt",
                     P.BEST_PREV_NAME]
            if not _keep_latest:
                _pats.insert(2, "checkpoint_latest.pth")
            else:
                print(f"[D-A4-1] checkpoint_latest.pth RETAINED "
                      f"(stopped_on={stopped_on}, keep_latest="
                      f"{bool(getattr(args, 'keep_latest', False))})", flush=True)
            for pat in _pats:
                for q in run_out.glob(pat):
                    q.unlink()
                    removed.append(q.name)
            (dump_dir / ".CONTINUED").touch()
            print("\n" + "=" * 78)
            print(f"[{args.run_id}]  CONTINUATION -- dump UNCHANGED")
            print(f"  resume source                {resume_rec.get('resume_source')}")
            print(f"  best epoch  old / new        {args.prev_best_epoch} / {best_epoch}")
            print(f"  best val metric old / new    {prev_best:.4f} / {best_val_metric:.4f}")
            print(f"  epochs in continuation       {cont['epochs_run_in_continuation']}")
            print(f"  epochs_run_total             {epochs_run_total}")
            print(f"  stopped_on                   {stopped_on}   ewi {ewi}")
            print(f"  wall clock                   {time.time() - t0:.1f}s "
                  f"(train {train_wall:.1f}s)")
            print(f"  periodic ckpts removed       {removed or 'none'}")
            print(f"  -> {dump_dir} (parquet and probs.npz untouched)")
            print("=" * 78, flush=True)
            return

        # Improved: the converged dump supersedes the truncated one. It is
        # written to a .new sibling first and swapped in only once it is
        # complete, so a killed writer can never destroy the truncated dump
        # without having a converged one to put in its place.
        cont["dump_action"] = (
            "re-dumped. The best improved during the continuation, so the "
            "truncated dump is moved aside to {run_id}.truncated/ and the "
            "converged dump takes the run_id path. Both survive.")
        cont_swap = {"final": dump_dir,
                     "truncated": Path(str(dump_dir) + ".truncated")}
        dump_dir = Path(str(dump_dir) + ".new")
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"[continuation] best improved; staging the converged dump at "
              f"{dump_dir}", flush=True)
        cont_record = {"continuation": cont,
                       "stopped_on_continuation": stopped_on,
                       "supersedes_dump": str(cont_swap["truncated"]),
                       "parent_run_id": args.run_id,
                       "parent_run_id_note": (
                           "self-referential by construction: this run "
                           "CONTINUES its own earlier, ceiling-truncated run "
                           "from that run's checkpoint_best.pth. The "
                           "truncated dump is kept at supersedes_dump.")}

    from utils.postprocessing import filter_small_predictions

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference")
        trainer.load_checkpoint(str(best_ckpt))
        model = trainer.model
        model.eval()
        device = trainer.device

        meta_by_pid = {}
        for r in rows_te:
            pid = f"{r['image_id']}_y{r['y']}_x{r['x']}"
            meta_by_pid[pid] = {
                "image_id": pid, "aoi_id": r["aoi_id"],
                "source_image_id": r["image_id"], "patch_index": pidx[pid],
                "patch_y": r["y"], "patch_x": r["x"],
                "city": r["city"], "split": "test",
                "acquisition_date": r["acq"],
                "sun_elevation": r["sun_el"], "sun_azimuth": r["sun_az"],
                "lon": r["lon"], "lat": r["lat"],
                "sun_dist": abs(float(r["sun_el"]) - train_sun_mean),
            }

        rows, probs = [], {}
        t_inf = time.time()
        with torch.no_grad():
            for batch in trainer.dataloaders["test"]:
                images = batch["image"].to(device)
                pid = batch["filename"][0]
                logits = model(images)
                # ARM A SESSION B-2, ADDITIVE. Every arm A trainer before
                # segdesic returned a plain tensor here. The segdesic models
                # return a dict -- {'main', 'aux1..3', 'geo'} on the CNNs,
                # {'p1'..'p6', 'geo'} on OGLANet -- and in eval the geo head
                # is not computed and the aux branches are gated off, so the
                # segmentation logits are the one meaningful entry.
                #
                # The unwrap is HERE and not on the model's forward because a
                # forward-level unwrap breaks the published trainer's own
                # validate(), which needs the dict
                # (mamnet/train_segdesic.py:549). At this call site it cannot
                # reach the trainer at all. Inert for every arch that already
                # returns a tensor.
                if isinstance(logits, dict):
                    logits = (logits.get("main")
                              if "main" in logits else logits.get("p6"))
                    if logits is None:
                        raise SystemExit(
                            "FATAL the model returned a dict with neither "
                            "'main' nor 'p6'; cannot resolve segmentation "
                            "logits for the dump")
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
                filtered = filter_small_predictions(logits, min_pixels=10)
                pred = torch.argmax(filtered, dim=1)[0].cpu().numpy().astype(np.uint8)
                gt = batch["mask"][0].numpy().astype(np.uint8)
                seen = batch["valid"][0].numpy().astype(bool)
                rows.append(seo_dump.perimage_row(
                    args.run_id, meta_by_pid[pid], gt, prob, pred, seen))
                probs[pid] = prob.astype(np.float16)
        inf_wall = time.time() - t_inf

        tbx.phase("metrics")
        import pandas as pd                                      # noqa: E402
        df = pd.DataFrame(rows)
        fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
        miou_tol = float(df["miou_tol_pct"].mean())
        pred_pos_frac = float(df["n_pred_pos"].sum() / max(int(df["n_valid_px"].sum()), 1))
        mean_unseen = float(df["unseen_frac"].mean())
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        wall = time.time() - t0

        # ---- test patch-id identity against the fold's VANILLA cells --------
        # The comparator is phase 2's vanilla loco and upper cells on this
        # fold. If the method cell's test rows are not the same patches in the
        # same set, no per-image difference against them means anything.
        vroot = Path(args.vanilla_dump_root) if args.vanilla_dump_root else (
            BASE / "dumps" / DATASET)
        mine = set(df["image_id"].tolist())
        vcheck = {}
        for vrid in (f"{arch}-vanilla-loco-r{RUNG}-{args.holdout.lower()}-{RES}-s{args.seed}",
                     f"{arch}-none-upper-r{RUNG}-{args.holdout.lower()}-{RES}-s{args.seed}"):
            vpq = vroot / vrid / "perimage.parquet"
            if not vpq.exists():
                vcheck[vrid] = "absent"
                continue
            try:
                theirs = set(pd.read_parquet(vpq, columns=["image_id"])["image_id"].tolist())
            except Exception as e:                              # noqa: BLE001
                vcheck[vrid] = f"unreadable: {type(e).__name__}"
                continue
            vcheck[vrid] = ("identical" if theirs == mine else
                            f"DIFFERS: mine {len(mine)}, theirs {len(theirs)}, "
                            f"symmetric difference {len(theirs ^ mine)}")
        manifest_identical = (mine == set(te_ids))

        stray_png = sorted(str(q.relative_to(run_out)) for q in run_out.rglob("*.png"))
        stray_tb = sorted(str(q.relative_to(run_out))
                          for q in run_out.rglob("events.out.tfevents*"))
        stray_epoch_ckpt = sorted(q.name for q in run_out.glob("checkpoint_epoch_*.pth"))

        equiv_sha_on_record = "4d030f64f444d886"
        justification = {
            "code_audit": ("code-path audit; empirical proof unexecutable, "
                           "harness nondeterministic at torch defaults"),
            "equivalence_pass": ("stage-1 bit-identical equivalence proof "
                                 "returned PASS"),
        }.get(args.train_eval_authority)

        run = {
            "run_id": args.run_id, "dataset": DATASET, "arm": "A",
            "arm_note": ("PHASE 3 of the REDUCED grid: SIB and FDA, LOCO-ONLY. "
                         "No new upper bound is trained -- each fold's phase-2 "
                         "vanilla upper cell is the comparator. OGLANet is "
                         "dropped from arm A. This box states no gap and "
                         "computes no bootstrap."),
            "arch": arch, "method": method, "mode": mode, "rung": RUNG,
            "arch_input": {"in_channels": spec["in_channels"],
                           "use_contrast": spec["use_contrast"]},
            "rung_note": ("rung here is the patch budget, 3500; the unit is a "
                          "384 px patch, not a benchmark image."),
            "holdout": args.holdout, "source_cities": man["source_cities"],
            "res": RES, "seed": args.seed,
            "n_train": len(ds_train), "n_val": len(ds_val), "n_test": len(df),
            "n_test_expected": man["n_test"],
            "n_train_aoi": len(aoi_tr), "n_val_aoi": len(aoi_va), "n_test_aoi": len(aoi_te),
            "n_train_per_source_city": {c: sum(1 for p in man["train"]
                                               if p.split("_", 1)[0] == c)
                                        for c in sorted({p.split("_", 1)[0]
                                                         for p in man["train"]})},
            "seed_sites": seed_sites, "leakage_check": leak,

            "target_access": ta, "target_access_detail": ta_detail,
            "method_block": method_prov,
            "method_probe": probe_rep,
            "method_overhead": overhead,

            "comparator": {
                "vanilla_loco": f"{arch}-vanilla-loco-r{RUNG}-{args.holdout.lower()}-{RES}-s{args.seed}",
                "vanilla_upper": f"{arch}-none-upper-r{RUNG}-{args.holdout.lower()}-{RES}-s{args.seed}",
                "note": ("phase-2 cells, already on disk. Phase 3 trains no new "
                         "upper bound."),
                "test_patchid_set_vs_vanilla": vcheck,
                "test_patchid_set_equals_manifest": manifest_identical,
            },

            "reporting_metric": "2class_tol_k2_unseen_excluded",
            "reporting_metric_value_pct": round(miou_tol, 4),
            "reporting_metric_note": (
                "two-class boundary-tolerant mIoU at K=2 with DSM-unseen pixels "
                "excluded from numerator and denominator. Column miou_tol_pct."),
            "sanity_floor_metric": "fg_only_tol_k2_unseen_excluded",
            "sanity_floor_metric_value_pct": round(fg_iou_tol, 4),
            "decision_metric": "2class_tol_k2_unseen_INCLUDED",
            "decision_metric_value_pct": round(best_val_metric, 4),
            "decision_metric_note": (
                "what checkpoint_best.pth was SELECTED on: the published "
                "trainer's own per-image DetailedEvaluator 2-class tolerant "
                "mIoU at K=2 on validation. Not the reporting metric."),
            "band_width_headline": 2, "band_widths_written": [0, 2, 5],

            "unseen_handling": {
                "metric": ("EXCLUDED from numerator and denominator at K=0, 2 "
                           "and 5, via a prediction-side sentinel; metrics.py "
                           "is not edited."),
                "loss": ("EXCLUDED: CrossEntropyLoss(ignore_index=255) on the "
                         "loader's mask_ignore target, train split only. "
                         + loss_note["installed_at"]
                         + ("; the method's KL term is added on top of that "
                            "loss, not beside it" if method == "sib" else "")),
                "val_decision_metric": (
                    "NOT excluded; the published DetailedEvaluator's band "
                    "morphology would be corrupted by a 255 sentinel. Both the "
                    "method cell and its vanilla comparator use the identical "
                    "rule."),
                "mean_unseen_frac_on_test": round(mean_unseen, 6),
                "rule_source": "logs/armA_seo_census_acq3.json valid_region_rule",
            },

            "budget_rule": man["budget_rule"],
            "budget_rule_sha": man["budget_rule_sha"],
            "split_rule_sha": man["split_rule_sha"],
            "budget_manifest": str(man_path),
            "budget_manifest_sha256": man_sha,
            "budget_manifest_reused_verbatim": True,
            "test_note": man["test_note"],

            "train_sun_elev_mean": round(train_sun_mean, 4),
            "train_sun_elev_sd": round(train_sun_sd, 4),

            "epochs_max": trainer_args.epochs, "epochs_run": epochs_run_total,
            "epochs_run_at_best_ckpt": epochs_run, "best_epoch": best_epoch,
            "steps_per_epoch": steps_per_epoch, "steps_to_best": steps_to_best,
            "epochs_without_improvement_at_stop": ewi, "stopped_on": stopped_on,
            # ARM A BLOCK 1, ADDITIVE. stopped_on now compares the last epoch
            # LABEL reached against the loop's own ceiling; the basis is
            # recorded so the label is auditable without the log, and v1's
            # value is preserved verbatim.
            "stopped_on_basis": stopped_on_basis,
            "last_epoch_label": last_epoch_label,
            "legacy_keys": {
                "stopped_on_v1": stopped_on_v1,
                "stopped_on_v1_note": (
                    "v1 evaluated `epochs_run_total >= EPOCHS_CEIL - 1`, which "
                    "compared the LENGTH of the trainer's per-epoch history "
                    "list against an epoch LABEL. Where the two disagree the "
                    "v1 value is wrong, not merely differently scaled."),
            },
            "early_stopped": bool(ewi >= PATIENCE_EPOCHS),
            "early_stop_patience": PATIENCE_EPOCHS,
            "early_stop_rule": (f"patience {PATIENCE_EPOCHS} epochs flat, epochs "
                                f"capped at {EPOCHS_MAX} - phase 2's recipe, so "
                                f"the method cell and its comparator stop under "
                                f"the same rule"),
            "early_stop_patience_published_base": cfg.get("early_stopping_patience"),
            "val_decision_metric_history_pooled_miou": val_hist,
            # ARM A WAVE 2f, ADDITIVE. The resume record was built on
            # every link but written only under `continuation`, and a
            # wave-2 cell is not a continuation -- so resume_source lived
            # in the logs and nowhere else. It is a per-cell provenance
            # fact and belongs in the dump. Nothing pre-existing moves;
            # the continuation record still carries its own copy.
            "resume": resume_rec,
            "resume_source": resume_rec.get("resume_source"),
            "resume_source_path": resume_rec.get("resume_source_path"),
            "resume_epoch_label_offset_note": (
                "PUBLISHED off-by-one, recorded not fixed: a resume skips one "
                "epoch LABEL at each link boundary. epochs_run counts actual "
                "epochs and is the number to use."),
            "best_val_metric": best_val_metric,
            "lr": trainer_args.lr, "batch_size": trainer_args.batch_size,
            "num_workers": args.num_workers,

            "dataset_detail": {
                "name": "S-EO (Masquil et al., CVPRW 2025, arXiv 2504.06920)",
                "unit": "384 px patch at native ~0.3 m GSD",
                "resize_applied": False, "patch_px": 384, "gsd_m": 0.3,
                "shadow_value": 0, "acq_per_aoi": 3, "bootstrap_unit": "aoi_id",
                "labels": ("simulated from an aligned LiDAR DSM plus solar "
                           "position through the bundle-adjusted RPC camera"),
                "radiometry": "uint8, colour-corrected and histogram-equalized at source",
                "category_maps": "empty (CONTRACT.md 3: external datasets)",
            },

            "metric_module_sha": sha256_file(BASE / "shadowtransfer/metrics.py"),
            "dump_module_sha": sha256_file(BASE / "shadowtransfer/seo_dump.py"),
            "budget_module_sha": sha256_file(BASE / "shadowtransfer/seo_budget.py"),
            "loader_module_sha": sha256_file(BASE / "shadowtransfer/seo_dataset.py"),
            "traineval_module_sha": sha256_file(BASE / "shadowtransfer/seo_traineval.py"),
            "runtime_module_sha": sha256_file(BASE / "shadowtransfer/armc_runtime.py"),
            "methods_module_sha": sha256_file(BASE / "shadowtransfer/armA_methods.py"),
            "train_script_sha": sha256_file(Path(__file__).resolve()),
            "phase12_train_script_sha": sha256_file(BASE / "support/armA_pilot_train.py"),
            "run_id_suffix": args.run_id_suffix or None,
            "retrain_note": ((
                "FROM-SCRATCH RETRAIN under D-A2-1. The ORIGINAL cell's "
                "dump is preserved untouched at its own run_id; this cell "
                "writes to a NEW run_id ending in the suffix above and "
                "never opens the original for writing.")
                if args.run_id_suffix else None),
            "phase12_train_script_note": (
                "phase 3 imports armA_pilot_train's helpers unchanged and owns "
                "only its own main(); the phase-1/2 file is not edited, so the "
                "train_script_sha recorded in the twelve finished vanilla "
                "cells still resolves."),
            "git_hash": os.environ.get("ARMA_GIT_HASH"),
            "git_hash_reason": (None if os.environ.get("ARMA_GIT_HASH")
                                else "ARMA_GIT_HASH not exported by the worker"),
            "git_tree_dirty_paths": os.environ.get("ARMA_GIT_DIRTY"),
            "config_sha256": sha256_file(recipe_config),
            "config_verbatim": a,
            "recipe_config_path": str(recipe_config.relative_to(BASE)),
            "recipe_config_note": (
                "the BASE recipe is phase 2's -- one published vanilla config "
                "per arch, for the method cell and its comparator alike, so "
                "they differ only in the method. Method-specific "
                "hyperparameters come from the published METHOD configs named "
                "in method_block, and override nothing else."),
            "config_defaults_filled": defaults_filled,
            "config_rerouted_keys": rerouted,
            "pretrained_weight_load": weight_load_vanilla,
            "pretrained_weight_load_note": (
                "loads performed while the VANILLA Trainer was constructed. "
                "For a SIB cell that model is then replaced; the SIB model's "
                "own loads are in method_block.sib_pretrained_weight_load."),

            "cluster": "delta", "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gpu": gpu_name, "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_time_limit": os.environ.get("ARMA_TIME_LIMIT"),
            "wall_clock_s": round(wall, 2),
            "train_wall_clock_s": round(train_wall, 2),
            "inference_wall_clock_s": round(inf_wall, 2),
            "peak_vram_alloc_gb": round(peak_alloc, 3),
            "peak_vram_reserved_gb": round(peak_reserved, 3),
            "compute_note": os.environ.get(
                "ST_COMPUTE_NOTE",
                "trained on gpuA100x4; no CPU association exists on this "
                "allocation (a GPU account only)"),

            "ckpt_path": str(best_ckpt), "parent_run_id": None,
            "legacy_run_dir": None, "structure_cols": "deferred",

            "train_eval_disabled": bool(train_eval_state.get("disabled")),
            "train_eval_authority": args.train_eval_authority,
            "train_eval_justification": justification,
            "train_eval_equivalence_sha": (equiv_sha_on_record
                                           if train_eval_state.get("disabled")
                                           else None),
            "train_eval_equivalence_sha_full": equiv_sha,
            "train_eval_equivalence_verdict": equiv_verdict,
            "train_eval_equivalence_artifact": equiv_path,
            "train_eval_disabled_from_epoch": train_eval_off_from_epoch,
            "train_eval_disabled_from_epoch_label": train_eval_off_from_epoch_label,
            "train_eval_link_first_epoch_label": first_epoch_this_link,
            "train_eval_state": train_eval_state,
            "train_eval_deviation_note": (
                "DECLARED DEVIATION, identical to phase 1 and phase 2: the "
                "TRAIN-side DetailedEvaluator is off under the code-path audit, "
                "operator decision 2026-08-02. The VALIDATION-side evaluator, "
                "which selects checkpoint_best.pth and drives patience-20, is "
                "UNTOUCHED."),
            "val_predpos_per_epoch": val_predpos,
            "val_predpos_probe": predpos_probe,
            "step_loss_probe": step_probe,
            "worker_side_effects_disabled": disabled,
            "collapse_guard": dict(armc_runtime.COLLAPSE_GUARD_STATE),
            "checkpoint_pruner": pruner,
            "post_training_timeout_s": args.timeout_s,
            "side_effect_audit": {"png_files": stray_png,
                                  "tensorboard_event_files": stray_tb,
                                  "periodic_checkpoints": stray_epoch_ckpt,
                                  "save_freq_effective": a.get("save_freq")},
            "prediction_recipe": ("argmax over the two class logits, then "
                                  "filter_small_predictions(min_pixels=10)"),
            "probs_note": "softmax(logits)[:, 1], stored float16, unthresholded",
            "aurc_definition": "aurc_trapz19",
            "aurc_definition_note": ("ratification pending P0a; CONTRACT.md line "
                                     "163 is still [BLANK]. Recomputable from "
                                     "probs.npz without a GPU."),
            "pred_pos_frac_on_valid": round(pred_pos_frac, 6),
            "smoke": smoke, "forced_rerun": bool(args.force),
            "dump_only": bool(args.dump_only),
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        flags = []
        if len(df) != len(ds_test):
            flags.append(f"dump rows {len(df)} != n_test {len(ds_test)}")
        if not smoke and len(df) != man["n_test"]:
            flags.append(f"dump rows {len(df)} != manifest n_test {man['n_test']}")
        if not (0.001 < pred_pos_frac < 0.9):
            flags.append(f"predicted-positive fraction {pred_pos_frac:.5f} outside "
                         f"(0.001, 0.9)")
        if not smoke and epochs_run_total >= EPOCHS_CEIL - 1:
            flags.append(f"ran to the epoch ceiling ({EPOCHS_CEIL}); early stopping "
                         f"never fired")
        if epochs_run_total < 10 and not smoke:
            flags.append(f"epochs_run={epochs_run_total} (<10)")
        if float(df["iou_tol"].std()) < 1e-6:
            flags.append("per-image iou_tol has ~zero variance")
        if int(df["empty_fg_valid"].sum()):
            flags.append(f"empty_fg_valid on {int(df['empty_fg_valid'].sum())} patches")
        if int(df["all_shadow_flag"].sum()):
            flags.append(f"all_shadow_flag on {int(df['all_shadow_flag'].sum())} patches")
        if stray_png:
            flags.append(f"figures written despite the rebind: {stray_png}")
        if stray_tb:
            flags.append(f"tensorboard events written despite the rebind: {stray_tb}")
        if not manifest_identical:
            flags.append("test patch-id set != the fold manifest's test list")
        for k, v in vcheck.items():
            if v not in ("identical", "absent"):
                flags.append(f"test patch-id set vs {k}: {v}")
        # `kl_applied` counts KL additions IN THIS PROCESS. A --dump_only
        # process does no training, so the counter is 0 by construction and
        # the check must not fire: it did on 2026-08-03 for the three SIB
        # cells recovered after the /projects quota block, and the flag was a
        # false positive. Their training logs show the KL plainly -- the
        # mamnet trainer prints total, main and aux, and total exceeded
        # main + 0.4*aux by 3.90 falling to 1.32 over the first six logged
        # batches, against exactly 0.0000 for the FDA cells that carry no KL.
        if (method == "sib" and sib_state is not None and not smoke
                and not args.dump_only):
            if int(sib_state.get("kl_applied", 0)) < 1:
                flags.append("SIB KL term was never added to the loss")
        if method == "sib" and args.dump_only:
            flags.append("dump_only: the SIB KL counter is 0 by construction "
                         "in this process and says nothing about training; see "
                         "method_block.sib_runtime.kl_applied_steps_note")
        for c in ("aoi_id", "sun_elevation"):
            if df[c].isna().any():
                raise SystemExit(f"FATAL {c} is null on "
                                 f"{int(df[c].isna().sum())} dump rows")
        run.update(cont_record)
        run["sanity_flags"] = flags
        if method == "sib" and sib_state is not None:
            run["method_block"]["sib_runtime"] = {
                "kl_applied_steps_note": (
                    "counts KL additions IN THIS PROCESS only. Zero under "
                    "--dump_only, which trains nothing; it is not evidence "
                    "about the run that produced the checkpoint."),
                "kl_applied_steps": int(sib_state.get("kl_applied", 0)),
                "forward_calls": int(sib_state.get("forward_calls", 0)),
                "last_kl": sib_state.get("last_kl"),
                "last_vib_warmup": sib_state.get("last_warmup"),
            }

        tbx.phase("write_dump")
        seo_dump.write_dump(dump_dir, rows, probs, run)

    if cont_swap is not None:
        import shutil                                          # noqa: E402
        if cont_swap["truncated"].exists():
            shutil.rmtree(cont_swap["truncated"])
        cont_swap["final"].rename(cont_swap["truncated"])
        dump_dir.rename(cont_swap["final"])
        dump_dir = cont_swap["final"]
        print(f"[continuation] truncated dump -> {cont_swap['truncated']}",
              flush=True)
        print(f"[continuation] converged dump -> {dump_dir}", flush=True)
    if args.continuation:
        (dump_dir / ".CONTINUED").touch()

    removed = []
    # ---- D-A4-1 CARVE-OUT (ARM A SESSION A4, 2026-09-03) -------------------
    # D-A4-1 "Recurrence prevention" ruled that `checkpoint_latest.pth`
    # SURVIVES on any cell whose `stopped_on` is `epoch_ceiling`, and on any
    # cell run with `--keep_latest`; a cell that stopped on patience has a
    # resolved ceiling status and still deletes it. That ruling was recorded in
    # DECISIONS_armA_wave2.md but NEVER LANDED IN CODE -- every prune site
    # listed "checkpoint_latest.pth" unconditionally, and the dinov3 BSM
    # stage-2 log of 2026-09-03 shows it in `periodic ckpts removed`. This is
    # the carve-out, implemented as ruled. It does not widen deletion: it only
    # withholds it.
    _keep_latest = bool(getattr(args, "keep_latest", False)) or (
        stopped_on == "epoch_ceiling")
    _pats = ["checkpoint_epoch_*.pth", "checkpoint_epoch_*.pth.corrupt",
             P.BEST_PREV_NAME]
    if not _keep_latest:
        _pats.insert(2, "checkpoint_latest.pth")
    else:
        print(f"[D-A4-1] checkpoint_latest.pth RETAINED "
              f"(stopped_on={stopped_on}, keep_latest="
              f"{bool(getattr(args, 'keep_latest', False))})", flush=True)
    for pat in _pats:
        for q in run_out.glob(pat):
            q.unlink()
            removed.append(q.name)

    print("\n" + "=" * 78)
    print(f"[{args.run_id}]  arch {arch}  method {method}")
    print(f"  target_access                {ta}")
    print(f"  2-class mIoU K=2 (headline)  {miou_tol:.4f}")
    print(f"  fg-only IoU K=2              {fg_iou_tol:.4f}")
    print(f"  fg-only IoU K=0 / K=5        {float(df['iou_str'].mean())*100:.4f} / "
          f"{float(df['iou_k5'].mean())*100:.4f}")
    print(f"  val decision metric          {best_val_metric:.4f}")
    print(f"  patches                      {len(df)}   AOIs {len(aoi_te)}")
    print(f"  predicted-positive fraction  {pred_pos_frac:.5f}")
    print(f"  test ids vs vanilla          {vcheck}")
    print(f"  epochs_run {epochs_run_total}  best_epoch {best_epoch}  "
          f"stopped_on {stopped_on}  epochs_without_improvement {ewi}")
    print(f"  wall clock                   {wall:.1f}s "
          f"(train {train_wall:.1f}s, inference {inf_wall:.1f}s)")
    print(f"  peak VRAM alloc/resv         {peak_alloc:.2f} / {peak_reserved:.2f} GiB")
    print(f"  periodic ckpts removed       {removed or 'none'}")
    print(f"  sanity flags                 {flags or 'none'}")
    print(f"  -> {dump_dir}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
