#!/usr/bin/env python3
"""WHICH MODEL WROTE THE SAVED OGLANET PREDICTIONS?

The question. data/Test_img_probs/loco/{chicago,miami}/highres/oglanet/vanilla/
holds 150 float16 .npy probability maps per city, written 2026-04-23 by jobs
17821359 (chicago) and 17821357 (miami). Their inference_config.json names
data/oglanet/outputs/oglanet_loco_holdout_{city}_highres_1/checkpoint_best.pth.
Neither the config nor the job log records a size, hash or mtime for the file it
loaded, so on-disk evidence alone cannot tie the saved maps to the checkpoint
that is there now (2026-10-25 inventory). This script settles it by measurement.

TWO CANDIDATE PRODUCERS, 10 passes each
---------------------------------------
Test 1  run_inference_probs.py -- the script the logs say wrote them.
        Its load_model and _extract_logits are IMPORTED and called, not
        reimplemented, so the checkpoint-loading and logit-extraction semantics
        are the originals byte-for-byte. Its dataset construction and batch size
        are reproduced from its own main().

Test 2  the test-time adaptation method with adaptation switched off:
        shadowtransfer.armf_sar_pass.unadapted_spec() is steps=0, labelled
        "unadapted (SAR off; published inference configuration)". Its own
        build_model / load_checkpoint_strict / sar.configure_model /
        sar.adapt_image are called, so this is arm F's code path, not an
        imitation of it.

WHY TEST 2 IS A CANDIDATE AT ALL. If Test 1 does not reproduce the saved maps,
the next hypothesis is that a different pipeline wrote them. The unadapted SAR
pass is the strongest alternative: it loads the same vanilla checkpoint and
produces vanilla predictions, but through a different dataset class, a
different batch size and a stricter load.

ERRATUM E-9 IS THE REASON THERE ARE 10 PASSES. OGLANet inference is
nondeterministic: repeated inference of one checkpoint over the same 150 images
moves the cell mean by up to 0.4247 mIoU and the per-pixel probability by up to
0.82, with byte-identical inputs, weights and buffers, on 0 of 15 cells made
bit-exact by cudnn.deterministic. A single pass therefore cannot answer the
question; the test is whether the saved maps' own score falls INSIDE the
10-pass range, and how the per-image probability deltas compare to the spread
the passes show among themselves.

THIS SCRIPT NEVER WRITES INTO data/Test_img_probs. run_inference_probs.py's
main() would write there; it is not called. Probabilities are produced in
memory, and the only outputs are under --out_root.
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
PY = BASE
for p in (str(PY), str(PY / "shadowtransfer")):
    if p not in sys.path:
        sys.path.insert(0, p)

SCRATCH_BASE = Path(os.environ["ST_SCRATCH_ROOT"])
RES_DIR = {"hr": "highres", "mr": "midres"}
SAVED_VARIANT = "vanilla"


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def resolve_ckpt(city, res_dir):
    """The LOCO checkpoint for this cell, from whichever root actually has it.

    /projects carries PARTIAL copies of some published run directories -- its
    oglanet_loco_holdout_chicago_midres_1 holds args.json and tensorboard and
    no checkpoint at all, while /scratch holds the complete set. Picking a root
    by directory existence would hand back the partial one.
    """
    rel = Path("data/oglanet/outputs") / f"oglanet_loco_holdout_{city}_{res_dir}_1"
    tried = []
    for name, root in (("projects", BASE), ("scratch", SCRATCH_BASE)):
        f = root / rel / "checkpoint_best.pth"
        tried.append(f"{name}:{f} exists={f.is_file()}")
        if f.is_file():
            return f, name, tried
    raise SystemExit("FATAL no checkpoint_best.pth on either root:\n  " +
                     "\n  ".join(tried))


def resolve_published_config(city, res_dir):
    rel = Path("data/oglanet/outputs") / f"oglanet_loco_holdout_{city}_{res_dir}_1"
    for root in (BASE, SCRATCH_BASE):
        p = root / rel / "args.json"
        if p.is_file():
            return json.loads(p.read_text()), p
    raise SystemExit(f"FATAL no args.json for {rel} on either root")


def load_saved_probs(city, res_dir, stems):
    """The maps under test. Read-only; this tree is never written."""
    d = (BASE / "data/Test_img_probs/loco" / city / res_dir / "oglanet"
         / SAVED_VARIANT)
    out = {}
    for s in stems:
        f = d / f"{s}.npy"
        if not f.is_file():
            raise SystemExit(f"FATAL saved prob missing: {f}")
        out[s] = np.load(f)            # float16, as written
    return out, d


def load_gt_direct(city, res_dir, stems):
    """GT masks read straight off disk, with no architecture tree involved.

    WHY NOT the arch dataset class, as Test 2 does. armf_sar_cells.test_images
    reaches armc_train.build_dataset, which does `from data.dataset_enhanced
    import ShadowDatasetEnhanced` and passes use_contrast=True. That resolves
    to whichever architecture tree happens to be on sys.path, and only
    armf_sar_cells.enter_arch_tree('oglanet') puts the right one there -- it
    also chdirs and purges cached models/data/utils modules, which is fine for
    Test 2 but not something Test 1 should be made to depend on after it has
    already imported run_inference_probs (which pulls in the MAMNet tree's
    ShadowDatasetEnhanced, whose __init__ has no use_contrast parameter at
    all). Calling it there raised
    `TypeError: ShadowDatasetEnhanced.__init__() got an unexpected keyword
    argument 'use_contrast'`.

    The two routes agree on these masks: they are 384x384 and strictly
    {0, 255}, so the dataset's resize-then-(>0) and this (>127) give the same
    array. Test 2 asserts that equality per image rather than assuming it.
    """
    import cv2
    d = BASE / "data/Final_data_test" / city / res_dir / "test/masks"
    out = {}
    for s_ in stems:
        f = d / f"{s_}.png"
        raw = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise SystemExit(f"FATAL unreadable mask {f}")
        out[s_] = (raw > 127).astype(np.uint8)
    return out


def score_probs(probs_by_stem, gt_by_stem):
    """Cell score from the one sanctioned metric module. Foreground-only IoU at
    K=2 is the headline (Appendix E); the two-class K=2 value is the Table 38
    family and is carried alongside so the cell can be read either way."""
    from shadowtransfer.armc_dump import iou_from_counts, miou_from_counts
    from shadowtransfer.metrics import per_image_counts, threshold_and_filter
    fg, two = [], []
    for s in sorted(probs_by_stem):
        p = np.asarray(probs_by_stem[s], dtype=np.float32)
        pred = threshold_and_filter(p, min_pixels=10)
        c = per_image_counts(pred, gt_by_stem[s], boundary_tolerance=2)
        fg.append(iou_from_counts(c["tp_tol"], c["fp_tol"], c["fn_tol"]))
        two.append(miou_from_counts(c["tp_tol"], c["fp_tol"],
                                    c["tn_tol"], c["fn_tol"])[2])
    return {"fg_iou_k2_pct": float(np.mean(fg) * 100.0),
            "miou_k2_twoclass_pct": float(np.mean(two)),
            "n_images": len(fg)}


def delta_stats(a_by_stem, b_by_stem):
    """Per-image largest probability difference, and the roll-up over images.

    Both sides are cast to float32 before differencing but the produced maps are
    first rounded to float16, because that is the dtype the saved maps were
    written in: comparing a float32 forward against a float16 file would report
    a quantisation step as a disagreement.
    """
    per = {}
    for s in sorted(a_by_stem):
        a = np.asarray(a_by_stem[s], dtype=np.float32)
        b = np.asarray(b_by_stem[s], dtype=np.float32)
        per[s] = float(np.max(np.abs(a - b)))
    v = np.array([per[s] for s in sorted(per)])
    return {
        "per_image_max_abs_delta": per,
        "max_over_images": float(v.max()),
        "mean_over_images": float(v.mean()),
        "median_over_images": float(np.median(v)),
        "n_images_bit_identical": int((v == 0.0).sum()),
        "n_images_differing": int((v > 0.0).sum()),
        "n_images": int(v.size),
    }


# ===========================================================================
# TEST 1 -- run_inference_probs.py, the script the logs name
# ===========================================================================

def t1_build(city, res_dir, ckpt, img_size, batch_size, num_workers):
    """Build exactly what run_inference_probs.py's main() builds.

    load_model and _extract_logits are the imported originals. The dataset and
    DataLoader lines are reproduced from that script's main(): the MAMNet tree's
    ShadowDatasetEnhanced with task_id=2 (its 'Task 2: contrast as 4th channel'
    switch), augment off, shuffle off. data_root is the one its own saved
    inference_config.json records for these cells: Final_data_test/{city}/{res},
    NOT the Final_data_loco fold tree.
    """
    import torch
    from torch.utils.data import DataLoader
    import run_inference_probs as R

    class A:
        pass
    a = A()
    a.model_type = "oglanet"
    a.model_variant = SAVED_VARIANT          # 'vanilla' -> the OGLANet base class
    a.test_type = "loco"
    a.city = city
    a.train_res = res_dir
    a.test_res = res_dir
    a.checkpoint_path = str(ckpt)
    a.data_root = str(BASE / "data/Final_data_test" / city / res_dir)
    a.img_size = img_size
    a.batch_size = batch_size
    a.num_workers = num_workers
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    a.shadow_class_idx = 1
    a.reference_gsd = 1.0
    a.dinov3_model_name = "dinov3_vits16"
    a.dinov3_weights_path = None
    a.dinov3_pretrained = True
    a.dinov3_frozen_stages = -1

    model, device = R.load_model(a)           # the original loader, strict=False
    ds = R.ShadowDatasetEnhanced(
        root_dir=[a.data_root], split="test", img_size=a.img_size,
        task_id=2, augment=False, geo_metadata_path=None)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=True)
    return model, device, loader, a, R


def t1_pass(model, device, loader, a, R):
    """One pass. Returns {stem: float16 prob map}, matching what the script
    saved: softmax over the class dim, shadow channel, cast to float16."""
    import torch
    import torch.nn.functional as F
    out = {}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            filenames = batch["filename"]
            outputs = model(images)
            logits = R._extract_logits(outputs, a)
            if (logits.shape[-2] != a.img_size or logits.shape[-1] != a.img_size):
                logits = F.interpolate(logits, size=(a.img_size, a.img_size),
                                       mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1)[:, a.shadow_class_idx]
            arr = probs.cpu().numpy().astype(np.float16)
            for i in range(arr.shape[0]):
                out[Path(str(filenames[i])).stem] = arr[i]
    return out


def t1_inputs(loader):
    """The input tensors Test 1 feeds the model, so the preprocessing question
    is answered by measurement rather than by reading two dataset classes."""
    out = {}
    for batch in loader:
        x = batch["image"].numpy()
        for i, fn in enumerate(batch["filename"]):
            out[Path(str(fn)).stem] = x[i].astype(np.float32)
    return out


# ===========================================================================
# TEST 2 -- the TTA method's code with adaptation switched off
# ===========================================================================

def t2_build(city, res_dir, ckpt, cfg):
    """arm F's own unadapted path: armf_sar_cells.build_model, its STRICT
    checkpoint load, sar.configure_model in BN_EVAL_SOURCE_STATS, and its
    materialised batch-size-1 image set. Nothing here is reimplemented."""
    import torch
    from shadowtransfer import armf_sar_cells as C
    from shadowtransfer import sar

    C.enter_arch_tree("oglanet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = C.build_model("oglanet", cfg, device)
    ck_epoch = C.load_checkpoint_strict(model, ckpt, device)
    model, bn_modules = sar.configure_model(model, sar.BN_EVAL_SOURCE_STATS)
    params, names, skipped = sar.collect_params(model)
    images = C.test_images("oglanet", city, res_dir)
    return {
        "model": model, "device": device, "images": images,
        "params": params, "names": names, "skipped": skipped,
        "bn_modules": bn_modules, "ck_epoch": ck_epoch,
        "pristine": sar.snapshot_state(model), "sar": sar,
        "num_classes": C.NUM_CLASSES, "norm_mode": C.NORM_MODE,
    }


def t2_pass(h):
    """One unadapted pass: steps=0, so adapt_image does a single forward under
    no_grad, builds no optimiser and applies no update."""
    sar = h["sar"]
    margin = sar.e0_margin(h["num_classes"])
    reset_const = sar.reset_constant(h["num_classes"])
    out = {}
    for im in h["images"]:
        sar.restore_state(h["model"], h["pristine"])
        x = im["x"].to(h["device"], non_blocking=True)
        _, probs = sar.adapt_image(h["model"], x, 0, margin, reset_const,
                                   lambda: None, h["pristine"])
        out[im["image_id"]] = probs[0].astype(np.float16)
        del x
    return out


def t2_inputs(h):
    return {im["image_id"]: im["x"][0].numpy().astype(np.float32)
            for im in h["images"]}


def t2_gt(h, city, res_dir):
    """The dataset-derived GT run_pass itself scores against, cross-checked
    per image against the direct mask read so the two tests are known to be
    scored on the same ground truth rather than assumed to be."""
    gt = {im["image_id"]: im["gt"] for im in h["images"]}
    direct = load_gt_direct(city, res_dir, sorted(gt))
    bad = [k for k in gt if not np.array_equal(gt[k], direct[k])]
    if bad:
        raise SystemExit(
            f"FATAL GT mismatch between the oglanet dataset class and a direct "
            f"mask read on {len(bad)} of {len(gt)} images, e.g. {bad[:5]}. The "
            f"two tests would not be scored on the same ground truth.")
    return gt


def pass_summary(scores, saved_score, key):
    """Does the saved maps' own score fall inside the N-pass range?"""
    v = np.array([s[key] for s in scores])
    lo, hi = float(v.min()), float(v.max())
    return {
        "metric": key,
        "per_pass": [float(x) for x in v],
        "min": lo, "max": hi,
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "range": hi - lo,
        "saved_value": float(saved_score[key]),
        "saved_inside_range": bool(lo <= saved_score[key] <= hi),
        "saved_minus_pass_mean": float(saved_score[key] - v.mean()),
    }


def input_digest(inputs):
    """Exact per-image fingerprint of a pipeline's model input, so the two
    pipelines' preprocessing can be compared across processes without moving
    150 tensors. One test runs per process: Test 1 imports the mamnet tree's
    dataset class and Test 2 chdirs into the oglanet tree and purges cached
    models/data/utils packages, so sharing a process risks a cross-tree import
    producing a plausible wrong number."""
    out = {}
    for s in sorted(inputs):
        a = np.ascontiguousarray(inputs[s], dtype=np.float32)
        out[s] = {
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "shape": list(a.shape),
            "n_channels": int(a.shape[0]) if a.ndim == 3 else None,
            "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()),
        }
    return out


def run_one(args):
    res_dir = RES_DIR[args.res]
    ckpt, ckpt_root, tried = resolve_ckpt(args.city, res_dir)
    cfg, cfg_path = resolve_published_config(args.city, res_dir)
    t_sha = time.time()
    ckpt_sha = sha256_file(ckpt)
    sha_s = time.time() - t_sha

    print("=" * 78)
    print(f"OGLANET SAVED-PREDICTION SOURCE   test={args.test}  "
          f"city={args.city}  res={args.res}")
    print(f"  checkpoint      {ckpt}")
    print(f"  ckpt root       {ckpt_root}")
    print(f"  ckpt size       {ckpt.stat().st_size}")
    print(f"  ckpt sha256     {ckpt_sha}   ({sha_s:.1f}s)")
    print(f"  config          {cfg_path}")
    print(f"  passes          {args.passes}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing run.")
        return None

    import torch
    gpu = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

    if args.test == "t1":
        model, device, loader, a, R = t1_build(
            args.city, res_dir, ckpt, args.img_size, args.batch_size,
            args.num_workers)
        inputs = t1_inputs(loader)
        stems = sorted(inputs)
        if args.limit:
            stems = stems[:args.limit]
        # GT straight off disk: no arch tree, so nothing can collide with the
        # MAMNet-tree imports run_inference_probs has already performed.
        gt = load_gt_direct(args.city, res_dir, stems)
        pass_fn = lambda: t1_pass(model, device, loader, a, R)
        pipeline = {
            "name": "run_inference_probs.py",
            "script": "run_inference_probs.py",
            "dataset_class": "mamnet/data/dataset_enhanced.py "
                             "ShadowDatasetEnhanced",
            "dataset_kwargs": {"task_id": 2, "augment": False,
                               "geo_metadata_path": None,
                               "img_size": args.img_size},
            "data_root": str(BASE / "data/Final_data_test" / args.city / res_dir),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "shuffle": False,
            "checkpoint_load": "load_state_dict(..., strict=False)",
            "model_ctor": "OGLANet(num_classes=2, pretrained=False, "
                          f"img_size={args.img_size}, use_contrast=True)",
            "eval_mode": "model.eval()",
            "logit_extraction": "run_inference_probs._extract_logits "
                                "(tuple/dict-aware: main / p6 / pred_fused)",
            "upsample": "F.interpolate bilinear align_corners=False if the "
                        "logits are not img_size",
            "prob": "F.softmax(logits, dim=1)[:, 1]",
            "output_dtype": "float16",
            "imported_verbatim": ["load_model", "_extract_logits",
                                  "ShadowDatasetEnhanced"],
        }
    else:
        h = t2_build(args.city, res_dir, ckpt, cfg)
        inputs = t2_inputs(h)
        gt = t2_gt(h, args.city, res_dir)
        stems = sorted(inputs)
        if args.limit:
            stems = stems[:args.limit]
            h["images"] = [im for im in h["images"] if im["image_id"] in set(stems)]
        pass_fn = lambda: t2_pass(h)
        pipeline = {
            "name": "armf_sar_pass unadapted (TTA with adaptation off)",
            "script": "shadowtransfer/armf_sar_pass.py + "
                      "shadowtransfer/sar.py",
            "spec": "unadapted_spec(): steps=0, "
                    "'unadapted (SAR off; published inference configuration)'",
            "dataset_class": "oglanet/data/dataset_enhanced.py "
                             "ShadowDatasetEnhanced (via armc_train.build_dataset)",
            "dataset_kwargs": {"use_contrast": True, "augment": False,
                               "img_size": 384},
            "data_root": str(BASE / "data/Final_data_test" / args.city / res_dir),
            "batch_size": 1,
            "num_workers": 0,
            "shuffle": False,
            "checkpoint_load": "load_state_dict(..., strict=True), hard fail on "
                               "any missing or unexpected key",
            "checkpoint_epoch": h["ck_epoch"],
            "model_ctor": "armf_sar_cells.build_model -> OGLANet(num_classes=2, "
                          "pretrained=False, img_size=cfg.img_size, "
                          "use_contrast=cfg.use_contrast)",
            "eval_mode": f"sar.configure_model({h['norm_mode']}): model.eval() "
                         f"+ requires_grad_(False); "
                         f"{len(h['bn_modules'])} BatchNorm2d modules on source "
                         f"running statistics",
            "logit_extraction": "sar._observe: logits.softmax(1)[0, 1] with no "
                                "tuple/dict unwrapping",
            "upsample": "none",
            "prob": "logits.softmax(1)[0, 1]",
            "output_dtype": "float32 in memory, rounded to float16 here to "
                            "match the saved files' dtype",
            "collected_norm_params": len(h["names"]),
            "episodic_restore": "state restored and byte-compared before every "
                                "image",
            "imported_verbatim": ["armf_sar_cells.build_model",
                                  "armf_sar_cells.load_checkpoint_strict",
                                  "armf_sar_cells.test_images",
                                  "sar.configure_model", "sar.adapt_image"],
        }

    saved, saved_dir = load_saved_probs(args.city, res_dir, stems)
    gt = {s: gt[s] for s in stems}
    saved_score = score_probs(saved, gt)
    print(f"saved maps: {saved_dir}")
    print(f"saved score: fg IoU K=2 {saved_score['fg_iou_k2_pct']:.4f}   "
          f"2-class K=2 {saved_score['miou_k2_twoclass_pct']:.4f}   "
          f"n={saved_score['n_images']}", flush=True)

    passes, scores, deltas = [], [], []
    first = None
    for i in range(args.passes):
        t0 = time.time()
        got = pass_fn()
        got = {s: got[s] for s in stems}
        sc = score_probs(got, gt)
        dl = delta_stats(got, saved)
        if first is None:
            first = got
            self_dl = None
        else:
            self_dl = delta_stats(got, first)
        w = time.time() - t0
        scores.append(sc)
        deltas.append(dl)
        passes.append({
            "pass": i, "wall_clock_s": round(w, 2),
            "score": sc,
            "vs_saved": {k: v for k, v in dl.items()
                         if k != "per_image_max_abs_delta"},
            "vs_pass0": (None if self_dl is None else
                         {k: v for k, v in self_dl.items()
                          if k != "per_image_max_abs_delta"}),
        })
        print(f"  pass {i}: fg IoU K=2 {sc['fg_iou_k2_pct']:.4f}  "
              f"2-class {sc['miou_k2_twoclass_pct']:.4f}  "
              f"max|dp| vs saved {dl['max_over_images']:.6f}  "
              f"bit-identical {dl['n_images_bit_identical']}/{dl['n_images']}  "
              f"{w:.0f}s", flush=True)

    # per-image worst delta across all passes, and the best-matching pass
    all_per = [d["per_image_max_abs_delta"] for d in deltas]
    worst_any = {s: max(d[s] for d in all_per) for s in stems}
    best_any = {s: min(d[s] for d in all_per) for s in stems}
    best_pass = int(np.argmin([d["max_over_images"] for d in deltas]))

    result = {
        "test": args.test,
        "pipeline": pipeline,
        "city": args.city, "res": args.res, "arch": "oglanet",
        "mode": "loco", "variant": SAVED_VARIANT,
        "checkpoint": {
            "path": str(ckpt), "root": ckpt_root,
            "size_bytes": ckpt.stat().st_size,
            "sha256": ckpt_sha,
            "mtime_utc": datetime.fromtimestamp(
                ckpt.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "roots_tried": tried,
        },
        "config_path": str(cfg_path),
        "saved_probs_dir": str(saved_dir.relative_to(BASE)),
        "n_images": len(stems),
        "limit_applied": args.limit or None,
        "passes": args.passes,
        "saved_score": saved_score,
        "pass_detail": passes,
        "range_check_fg_iou_k2": pass_summary(scores, saved_score,
                                             "fg_iou_k2_pct"),
        "range_check_miou_k2_twoclass": pass_summary(
            scores, saved_score, "miou_k2_twoclass_pct"),
        "per_image_delta_vs_saved": {
            "worst_across_passes": worst_any,
            "best_across_passes": best_any,
            "max_worst": float(max(worst_any.values())),
            "max_best": float(max(best_any.values())),
            "n_images_bit_identical_in_every_pass": int(sum(
                1 for s in stems if worst_any[s] == 0.0)),
            "n_images_bit_identical_in_some_pass": int(sum(
                1 for s in stems if best_any[s] == 0.0)),
            "best_matching_pass": best_pass,
        },
        "determinism_across_passes": {
            "note": "pass i vs pass 0 on identical inputs and weights; "
                    "non-zero here is erratum E-9, not a provenance signal",
            "max_abs_delta": (None if args.passes < 2 else float(max(
                p["vs_pass0"]["max_over_images"] for p in passes[1:]))),
            "n_passes_bit_identical_to_pass0": (
                None if args.passes < 2 else int(sum(
                    1 for p in passes[1:]
                    if p["vs_pass0"]["max_over_images"] == 0.0))),
        },
        "input_digest": input_digest({s: inputs[s] for s in stems}),
        "env": {
            "gpu": gpu,
            "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "torch": __import__("torch").__version__,
            "cudnn_deterministic": bool(
                __import__("torch").backends.cudnn.deterministic),
            "cudnn_benchmark": bool(__import__("torch").backends.cudnn.benchmark),
        },
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return result


def do_merge(out_root):
    """Fold the per-(city, res, test) JSONs into results.json and decide the
    verdict. The preprocessing comparison happens here, because the two tests
    run in separate processes."""
    parts = {}
    for f in sorted(Path(out_root).glob("*_*_t?.json")):
        d = json.loads(f.read_text())
        parts[(d["city"], d["res"], d["test"])] = d
    if not parts:
        raise SystemExit(f"FATAL no per-test JSONs under {out_root}")

    cells, preproc = {}, {}
    for (city, res, test), d in sorted(parts.items()):
        cells.setdefault(f"{city}_{res}", {})[test] = {
            "saved_score": d["saved_score"],
            "range_check_fg_iou_k2": d["range_check_fg_iou_k2"],
            "range_check_miou_k2_twoclass": d["range_check_miou_k2_twoclass"],
            "per_image_delta_vs_saved": {
                k: v for k, v in d["per_image_delta_vs_saved"].items()
                if not isinstance(v, dict)},
            "determinism_across_passes": d["determinism_across_passes"],
            "checkpoint": d["checkpoint"],
            "pipeline": d["pipeline"],
            "passes": d["passes"],
            "n_images": d["n_images"],
        }

    for key in sorted(cells):
        both = cells[key]
        if "t1" not in both or "t2" not in both:
            continue
        a = parts[(key.rsplit("_", 1)[0], key.rsplit("_", 1)[1], "t1")]["input_digest"]
        b = parts[(key.rsplit("_", 1)[0], key.rsplit("_", 1)[1], "t2")]["input_digest"]
        shared = sorted(set(a) & set(b))
        same = [s for s in shared if a[s]["sha256"] == b[s]["sha256"]]
        ch_a = sorted({a[s]["n_channels"] for s in shared})
        ch_b = sorted({b[s]["n_channels"] for s in shared})
        preproc[key] = {
            "n_images_compared": len(shared),
            "n_inputs_byte_identical": len(same),
            "inputs_identical": len(same) == len(shared) and len(shared) > 0,
            "t1_channels": ch_a, "t2_channels": ch_b,
            "t1_shapes": sorted({tuple(a[s]["shape"]) for s in shared})[:3],
            "t2_shapes": sorted({tuple(b[s]["shape"]) for s in shared})[:3],
            "example_disagreement": next(
                ({"image_id": s,
                  "t1": a[s], "t2": b[s]} for s in shared
                 if a[s]["sha256"] != b[s]["sha256"]), None),
            "declared_differences": {
                "batch_size": [both["t1"]["pipeline"]["batch_size"],
                               both["t2"]["pipeline"]["batch_size"]],
                "dataset_class": [both["t1"]["pipeline"]["dataset_class"],
                                  both["t2"]["pipeline"]["dataset_class"]],
                "checkpoint_load": [both["t1"]["pipeline"]["checkpoint_load"],
                                    both["t2"]["pipeline"]["checkpoint_load"]],
                "logit_extraction": [both["t1"]["pipeline"]["logit_extraction"],
                                     both["t2"]["pipeline"]["logit_extraction"]],
                "resize": ["img_size 384, dataset-level, both",
                           "img_size 384, dataset-level, both"],
                "normalisation": [
                    "fixed ImageNet mean/std; contrast_utils.py is "
                    "byte-identical (md5 8983273da4a390d711b633d2db97aaf4) in "
                    "all four arch trees, so the RGBC 4th channel is the same "
                    "code on both paths",
                    "same"],
                "crop": ["none on either path (augment=False, test split)",
                         "none"],
                "channel_order": ["RGBC", "RGBC"],
                "tiling": ["none; whole 384x384 patch", "none"],
            },
        }

    verdict = {}
    for key in sorted(cells):
        v = {}
        for test, d in cells[key].items():
            rc = d["range_check_fg_iou_k2"]
            pid = d["per_image_delta_vs_saved"]
            v[test] = {
                "saved_inside_10pass_range": rc["saved_inside_range"],
                "saved_fg_iou_k2": rc["saved_value"],
                "pass_range": [rc["min"], rc["max"]],
                "pass_mean": rc["mean"], "pass_sd": rc["sd"],
                "max_per_image_prob_delta_best_pass": pid["max_best"],
                "n_images_bit_identical_in_some_pass":
                    pid["n_images_bit_identical_in_some_pass"],
                "bit_exact_reproduction": pid["max_best"] == 0.0,
            }
        verdict[key] = v

    out = {
        "id": "oglanet_saved_prediction_source",
        "question": ("Which model, and which pipeline, wrote "
                     "data/Test_img_probs/loco/{chicago,miami}/highres/oglanet/"
                     "vanilla/?"),
        "arm": "C",
        "weakness_ids": ["E-9"],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "support/oglanet_probsource.py",
        "git_hash": None,
        "git_hash_reason": "project root is not a git repository (CLAUDE.md)",
        "provenance_jobs": {"chicago": "17821359", "miami": "17821357",
                            "written": "2026-04-23"},
        "metric": ("foreground-only IoU at K=2 (Appendix E) as the headline, "
                   "two-class K=2 (the Table 38 family) alongside; both from "
                   "shadowtransfer/metrics.py"),
        "e9_note": ("erratum E-9: OGLANet inference is nondeterministic, so a "
                    "single pass cannot answer this. The test is whether the "
                    "saved maps' own score falls inside the 10-pass range, "
                    "read next to the pass-to-pass spread on identical inputs."),
        "preprocessing_comparison": preproc,
        "verdict": verdict,
        "cells": cells,
        "caveats": [
            "Produced maps are rounded to float16 before differencing, because "
            "that is the dtype the saved files carry; a float32-vs-float16 "
            "comparison would report quantisation as disagreement.",
            "A score landing inside the 10-pass range is consistent with the "
            "on-disk checkpoint having produced the saved maps; under E-9 it is "
            "not proof of byte identity of the checkpoint file itself.",
            "No hash, size or mtime of the loaded file was recorded in 2026-04, "
            "so the 2026-04-23 run's checkpoint cannot be compared directly "
            "against today's.",
        ],
    }
    p = Path(out_root) / "results.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"merged {len(parts)} per-test JSONs -> {p}")
    for key in sorted(verdict):
        for test in sorted(verdict[key]):
            d = verdict[key][test]
            print(f"  {key:16s} {test}: saved {d['saved_fg_iou_k2']:.4f}  "
                  f"range [{d['pass_range'][0]:.4f}, {d['pass_range'][1]:.4f}]  "
                  f"inside={d['saved_inside_10pass_range']}  "
                  f"bit_exact={d['bit_exact_reproduction']}  "
                  f"max|dp|={d['max_per_image_prob_delta_best_pass']:.6f}")
    for key in sorted(preproc):
        pp = preproc[key]
        print(f"  {key:16s} inputs identical={pp['inputs_identical']} "
              f"({pp['n_inputs_byte_identical']}/{pp['n_images_compared']})  "
              f"channels t1={pp['t1_channels']} t2={pp['t2_channels']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["t1", "t2"],
                    help="t1 = run_inference_probs.py; t2 = TTA code, "
                         "adaptation off")
    ap.add_argument("--city", choices=["chicago", "miami"])
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=4,
                    help="run_inference_probs.py's own default; Test 1 only")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="fold the per-test JSONs into results.json")
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else \
        BASE / "runs/oglanet_saved_prediction_source"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.merge:
        do_merge(out_root)
        return

    if not args.test or not args.city:
        raise SystemExit("FATAL --test and --city are required unless --merge")

    res = run_one(args)
    if res is None:
        return
    p = out_root / f"{args.city}_{args.res}_{args.test}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n-> {p}")

    rc = res["range_check_fg_iou_k2"]
    pid = res["per_image_delta_vs_saved"]
    print(f"\n[{args.test} {args.city} {args.res}]")
    print(f"  saved fg IoU K=2        {rc['saved_value']:.4f}")
    print(f"  10-pass range           [{rc['min']:.4f}, {rc['max']:.4f}]  "
          f"mean {rc['mean']:.4f}  sd {rc['sd']:.4f}")
    print(f"  saved inside range      {rc['saved_inside_range']}")
    print(f"  max |dp| (best pass)    {pid['max_best']:.6f}")
    print(f"  bit-identical images    "
          f"{pid['n_images_bit_identical_in_some_pass']}/{res['n_images']} "
          f"in some pass, "
          f"{pid['n_images_bit_identical_in_every_pass']}/{res['n_images']} "
          f"in every pass")
    print(f"  pass-to-pass max |dp|   "
          f"{res['determinism_across_passes']['max_abs_delta']}")


if __name__ == "__main__":
    main()
