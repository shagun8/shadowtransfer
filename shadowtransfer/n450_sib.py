"""
n450 retrain -- OGLANet SIB (C4clean) adapter for train_oglanet_sib.main().

main() is monolithic (train loop and final test inline), so it is driven with
module-level rebinds instead of a Trainer object. Rebinds, all in-process,
published file untouched:
  parse_args               -> returns the resolved Namespace (n450_config)
  get_dataloaders_sib      -> the original, then n450_inject.assert_counts()
                              BEFORE the first optimizer step
  SummaryWriter            -> armc_runtime.NullWriter
  plot_all_losses          -> no-op (CLAUDE.md: no figures in a worker)
  compare_with_baselines   -> no-op (reads other methods' stored predictions
                              and writes comparison_results.json; not part of
                              the trained model or its test output)
  test_and_save_predictions-> test_once() below

test_once() is the published test_and_save_predictions body, line for line for
the eval path (model.eval(); autocast(enabled=args.use_amp); model(images,
intensity_map=..., city_ids=...); p6 argmax; PNG; _compute_strict_metrics /
_compute_tolerant_metrics / _average_metrics; test_results.json), with ONE
addition: the softmax shadow-class probability of the SAME p6 tensor is kept as
float16 per image. Operator ruling: OGLANet SIB trained under autocast, so test
inference runs ONCE, the probability maps are saved from that pass, and the
scores come from those maps -- scores and saved predictions from one pass at
one precision. TENT is refused (use_tent is false in all three originals).
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from shadowtransfer import armc_runtime as RT
from shadowtransfer import n450_inject as INJ

STATE = {}


class ConstructOnly(Exception):
    """Raised after the loaders are built and counted (pre-queue CPU test)."""


def _noop(*_a, **_k):
    return None


def install(mod, ns, probs_dir, timebox):
    """timebox: an armc_runtime.PhaseTimeout, armed when the test pass starts."""
    orig_gdl = mod.get_dataloaders_sib

    def get_dataloaders_sib(*a, **k):
        data = orig_gdl(*a, **k)
        STATE["counts"] = INJ.assert_counts(n_test=150)
        if STATE.get("construct_only"):
            raise ConstructOnly()
        return data

    mod.get_dataloaders_sib = get_dataloaders_sib
    mod.parse_args = lambda: ns
    mod.SummaryWriter = RT.NullWriter
    mod.plot_all_losses = _noop
    mod.compare_with_baselines = _noop

    def test_once(model, loader, device, args, logger):
        import torch
        from PIL import Image
        STATE["t_test_start"] = time.time()
        timebox.__enter__()
        STATE["timebox_armed"] = True
        timebox.phase("test pass under autocast + probability maps")
        if getattr(args, "use_tent", False):
            raise RuntimeError("use_tent=True: not part of any C4clean recipe")
        if torch.cuda.is_available():
            print(f"[n450_sib] VRAM after best-model load: "
                  f"{torch.cuda.memory_allocated()/2**30:.2f} GiB allocated, "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                  flush=True)
        model.eval()
        pred_save_dir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(pred_save_dir, exist_ok=True)
        os.makedirs(probs_dir, exist_ok=True)
        sib_strict_list, sib_tolerant_list, all_filenames, all_alphas = [], [], [], []
        disagree_px, n_px = 0, 0
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            intensity_map = batch['intensity_map'].to(device)
            city_ids = batch.get('city_id', None)
            if city_ids is not None:
                city_ids = city_ids.to(device)
            with torch.no_grad():
                with mod.autocast(enabled=args.use_amp):
                    out = model(images, intensity_map=intensity_map,
                                city_ids=city_ids)
                    predictions = out['predictions']
                    kl_losses = out['kl_losses']
            pred_p6 = predictions['p6']
            preds = pred_p6.argmax(dim=1)
            # the one addition: probabilities from the SAME p6 tensor
            prob16 = (torch.softmax(pred_p6.float(), dim=1)[:, 1]
                      .cpu().numpy().astype(np.float16))
            if 'bypass_alpha' in kl_losses:
                alpha_batch = kl_losses['bypass_alpha'].cpu().numpy()
                for j in range(alpha_batch.shape[0]):
                    all_alphas.append(float(alpha_batch[j]))
            for i, fname in enumerate(batch['filename']):
                pred_np = preds[i].cpu().numpy().astype(np.uint8)
                gt_np = masks[i].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np * 255).save(
                    os.path.join(pred_save_dir, fname))
                np.save(os.path.join(probs_dir, Path(fname).stem + ".npy"),
                        prob16[i])
                disagree_px += int(((prob16[i] > 0.5).astype(np.uint8)
                                    != pred_np).sum())
                n_px += pred_np.size
                sib_strict_list.append(mod._compute_strict_metrics(pred_np, gt_np))
                sib_tolerant_list.append(mod._compute_tolerant_metrics(
                    pred_np, gt_np, tolerance=args.boundary_tolerance))
                all_filenames.append(fname)
        sib_strict = mod._average_metrics(sib_strict_list)
        sib_tolerant = mod._average_metrics(sib_tolerant_list)
        logger.info(f'\nSIB Results ({len(all_filenames)} images): '
                    f'Tolerant ShIOU={sib_tolerant["Shadow_IOU"]:.2f}  '
                    f'mIOU={sib_tolerant["mIOU"]:.2f}')
        test_results = {
            'num_images': len(all_filenames),
            'strict': sib_strict,
            f'tolerant_{args.boundary_tolerance}px': sib_tolerant,
            'tent_active': False,
        }
        if all_alphas:
            test_results['bypass_alpha_summary'] = {
                'mean': float(np.mean(all_alphas)), 'std': float(np.std(all_alphas)),
                'min': float(np.min(all_alphas)), 'max': float(np.max(all_alphas))}
        with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        STATE["test"] = {
            "n_images": len(all_filenames),
            "precision": ("fp16 autocast (torch.cuda.amp.autocast enabled=use_amp="
                          f"{bool(args.use_amp)}); argmax PNGs, test_results.json "
                          "and probs.npz all from this one pass"),
            "argmax_vs_prob16_gt_half_disagree_px": disagree_px,
            "n_px": n_px,
            "test_wall_s": time.time() - STATE["t_test_start"],
        }
        return (sib_strict, sib_tolerant, sib_strict_list, sib_tolerant_list,
                all_filenames)

    mod.test_and_save_predictions = test_once
    return ["parse_args", "get_dataloaders_sib(+assert_counts)", "SummaryWriter",
            "plot_all_losses", "compare_with_baselines",
            "test_and_save_predictions(+probs)"]


def epoch_stats(history, epochs_max, patience, steps_per_epoch):
    if not history:
        return {"epochs_run": 0}
    vals = [h["val_mIOU"] for h in history]
    best_i = int(np.argmax(vals))                 # first max == strict '>' rule
    run = len(history)
    return {"epochs_max": epochs_max, "epochs_run": run,
            "best_epoch": history[best_i]["epoch"],
            "best_val_metric": float(vals[best_i]),
            "stopped_on": ("early_stopping" if run < epochs_max else "epochs_max"),
            "early_stop_patience": patience,
            "steps_per_epoch": steps_per_epoch,
            "steps_to_best": history[best_i]["epoch"] * steps_per_epoch}
