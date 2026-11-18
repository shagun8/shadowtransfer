#!/usr/bin/env python3
"""n450 OGLANet cells re-inferred at batch size 4, and the self-check re-run
at batch 4 on BOTH sides.

WHY. OGLANet's DFFM rebuilds 20 unregistered, randomly-initialised
Conv2d+BatchNorm2d layers inside forward() on every call
(oglanet/models/dffm.py:33-37 factory, 66-82 call site). Those
BatchNorms normalise with BATCH statistics, so an OGLANet score depends on the
inference batch size: 33.05 at batch 1 rising to 40.55 at batch 8 on one
measured cell. The published OGLANet tables were inferred at batch 4
(run_inference.sh:73, run_inference_probs.sh:77), so an n450 OGLANet cell that
is compared against them must be inferred at batch 4 too.

WHAT THIS DOES, per cell, writing only to `<run_id>-bs4` paths:
  1. run_inference.py      --batch_size 4  -> 0/255 PNG masks
  2. run_inference_probs.py --batch_size 4 -> float16 probability maps
  3. side B of the gate's self-check, recomputed at batch 4: the predictions
     from step 2 are fed through the architecture's OWN
     utils.evaluation_detailed.DetailedEvaluator (the lifted source of tolerant
     IoU per CLAUDE.md), and a test_results.json is written in the trainer's
     own shape so n450_score reads it exactly as it reads a real one.
  4. n450_score.score(run_id + "-bs4", ...) -> side A, the dump, and the
     self_check block with both sides now at batch 4.

The original cell, its checkpoint and its dump are never written to.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
PY = sys.executable
E9_SPAN = 1.12          # OGLANet's measured nine-pass span, erratum E-9
OLD_THRESHOLD = 0.25    # the gate's GROSS_SELF_CHECK, which predates DFFM


def detailed_side_b(probs_dir, mask_dir, arch, min_pixels=10):
    """Side B at batch 4: the architecture's own DetailedEvaluator over the
    batch-4 predictions. Returns both prediction recipes so the comparison is
    not resting on a guess about which one the trainer used."""
    import cv2
    import torch
    from shadowtransfer.metrics import threshold_and_filter
    sys.path.insert(0, os.path.join(BASE, arch))
    from utils.evaluation_detailed import DetailedEvaluator

    out = {}
    for label in ("filtered", "argmax"):
        ev = DetailedEvaluator(boundary_tolerance=2)
        for p in sorted(Path(probs_dir).glob("*.npy")):
            prob = np.load(p).astype(np.float32)
            gt = cv2.imread(str(Path(mask_dir) / f"{p.stem}.png"),
                            cv2.IMREAD_GRAYSCALE)
            if gt is None:
                raise SystemExit(f"FATAL unreadable mask for {p.stem}")
            gt = (gt > 127).astype(np.uint8)
            pred = (threshold_and_filter(prob, min_pixels=min_pixels)
                    if label == "filtered" else (prob > 0.5).astype(np.uint8))
            # DetailedEvaluator.update takes BINARY predictions [B,H,W] with
            # values {0,1} and targets [B,H,W] -- NOT class logits. Verified
            # against evaluation_detailed.py:127-135 before use.
            ev.update(torch.from_numpy(pred.astype(np.uint8))[None],
                      torch.from_numpy(gt.astype(np.uint8))[None])
        m = ev.compute_metrics()
        bt = m["boundary_tolerant"]
        key = "tolerant_2px" if "tolerant_2px" in bt else [
            k for k in bt if k.startswith("tolerant")][0]
        out[label] = {"tol_key": key, "iou": float(bt[key]["iou"]),
                      "f1": float(bt[key].get("f1", float("nan"))),
                      "strict_iou": float(bt["strict"]["iou"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--suffix", default="-bs4")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from shadowtransfer import n450_roster as R
    import armc_r450_rescore as R450
    import n450_score

    cell = R.by_run_id()[a.run_id]
    arch, city = cell["arch"], cell["city"]
    if arch != "oglanet":
        print(f"REFUSED {a.run_id}: arch={arch} has no DFFM; its scoring is "
              f"left unchanged by instruction (step 1c).")
        return 2
    if not cell.get("prob_variant"):
        print(f"REFUSED {a.run_id}: prob_variant is None. The SIB family has no "
              f"model_variant in run_inference_probs.py / run_inference.py -- "
              f"their oglanet variants are base/vanilla/fda/isw, segdesic, iim, "
              f"mrfp_plus, fada, none of which builds an OGLANetSIB. A batch-4 "
              f"pass for a SIB cell needs NEW model-loading code for a "
              f"published architecture, which is an operator decision. This "
              f"cell's probability maps come from the SIB trainer's own test "
              f"loader at args.batch_size under fp16 autocast.")
        return 3
    new_id = a.run_id + a.suffix
    ckpt_root = cell["ckpt_dir"]
    # the checkpoint the finished cell actually produced
    cands = [os.path.join(ckpt_root, "checkpoints", "best_model.pth"),
             os.path.join(ckpt_root, "checkpoint_best.pth")]
    cands += [str(p) for p in Path(ckpt_root).rglob("checkpoint_best.pth")]
    cands += [str(p) for p in Path(ckpt_root).rglob("best_model.pth")]
    ckpt = next((c for c in cands if os.path.isfile(c)), None)
    if ckpt is None:
        raise SystemExit(f"FATAL no checkpoint under {ckpt_root}; tried {cands[:4]}")

    bs_root = f"{ckpt_root}_bs{a.batch}"
    png_out = os.path.join(bs_root, "preds_tree")
    prob_out = os.path.join(bs_root, "probs_tree")
    os.makedirs(bs_root, exist_ok=True)
    mask_dir = R450.mask_dir_for("loco", city, "highres")

    print("=" * 78)
    print(f"n450 BATCH-{a.batch} RE-INFERENCE   {a.run_id} -> {new_id}")
    print(f"  arch/city     {arch} / {city}")
    print(f"  checkpoint    {ckpt}")
    print(f"  variant       {cell['prob_variant']}")
    print(f"  bs root       {bs_root}   (the original cell is never written to)")
    print(f"  mask dir      {mask_dir}")
    print("=" * 78)

    common = ["--model_type", arch, "--model_variant", cell["prob_variant"],
              "--test_type", "loco", "--city", city,
              "--train_res", "highres", "--test_res", "highres",
              "--checkpoint_path", ckpt,
              "--data_root", os.path.join(BASE, "data/Final_data_test", city, "highres"),
              "--img_size", "384", "--batch_size", str(a.batch),
              "--num_workers", "4", "--device", "cuda"]

    t = time.time()
    print(f"\n[1/4] run_inference.py  --batch_size {a.batch}  -> PNG masks")
    subprocess.run([PY, "-u", os.path.join(BASE, "run_inference.py")]
                   + common + ["--output_dir", png_out],
                   cwd=os.path.join(BASE, "python"), check=True)
    t_png = time.time() - t

    t = time.time()
    print(f"\n[2/4] run_inference_probs.py  --batch_size {a.batch}  -> prob maps")
    subprocess.run([PY, "-u", os.path.join(BASE, "run_inference_probs.py")]
                   + common + ["--output_dir", prob_out],
                   cwd=os.path.join(BASE, "python"), check=True)
    t_prob = time.time() - t

    probs_dir = os.path.join(prob_out, "loco", city, "highres", arch,
                             cell["prob_variant"])
    preds_dir = os.path.join(png_out, "loco", city, "highres", arch,
                             cell["prob_variant"])
    n_npy = len(list(Path(probs_dir).glob("*.npy")))
    n_png = len(list(Path(preds_dir).glob("*.png")))
    print(f"\n  prob maps {n_npy} in {probs_dir}")
    print(f"  PNG masks {n_png} in {preds_dir}")
    if n_npy != 150 or n_png != 150:
        raise SystemExit(f"FATAL expected 150/150, got {n_npy}/{n_png}")

    print(f"\n[3/4] side B at batch {a.batch}: the arch's own DetailedEvaluator")
    sb = detailed_side_b(probs_dir, mask_dir, arch)
    for k, v in sb.items():
        print(f"      {k:9s} tolerant_2px iou = {v['iou']:.6f}  "
              f"(strict {v['strict_iou']:.6f})")
    fam = "sib" if cell["method"] == "sibc4" else "classic"
    recipe = "nofilter" if fam == "sib" else "filtered"
    chosen = sb["argmax" if recipe == "nofilter" else "filtered"]
    tr = {"detailed": {"boundary_tolerant": {"tolerant_2px": {
              "iou": chosen["iou"], "f1": chosen["f1"]}}},
          "tolerant_2px": {"Shadow_IOU": chosen["iou"]},
          "_provenance": {
              "written_by": "matched450/n450_bs4.py",
              "what": f"side B of the n450 gate self-check, recomputed at batch "
                      f"{a.batch} from the same pass as side A, using "
                      f"python/{arch}/utils/evaluation_detailed.py "
                      f"DetailedEvaluator(boundary_tolerance=2)",
              "not_a_trainer_output": True,
              "prediction_recipe": recipe,
              "both_recipes": sb}}
    tr_path = os.path.join(bs_root, "test_results.json")
    json.dump(tr, open(tr_path, "w"), indent=1)
    print(f"      wrote {tr_path}  (recipe used for the check: {recipe})")

    print(f"\n[4/4] n450_score.score -> {new_id}")
    dump_dir = os.path.join(R.DUMP_ROOT, new_id)
    run = {"run_id": new_id, "parent_run_id": a.run_id,
           "inference_batch_size": a.batch,
           "inference_batch_size_note":
               "re-inferred at batch 4 to match the published OGLANet tables "
               "(run_inference.sh:73, run_inference_probs.sh:77). OGLANet "
               "scores are batch-size dependent because DFFM rebuilds 20 "
               "unregistered batch-statistic BatchNorms inside forward(); see "
               "runs/dffm_defect/provenance.json.",
           "inference_precision": "fp32 (no autocast), as the published trees",
           "png_masks_dir": preds_dir, "probs_dir_bs4": probs_dir,
           "png_inference_wall_s": round(t_png, 1),
           "prob_inference_wall_s": round(t_prob, 1),
           "checkpoint_used": ckpt,
           "self_check_side_b_at_batch4": tr["_provenance"],
           "e9_nine_pass_span": E9_SPAN,
           "old_gate_threshold": OLD_THRESHOLD}
    r = n450_score.score(new_id, cell, probs_dir, dump_dir, run,
                         own_test_json=tr_path)
    sc = r["scoring"]
    print(f"\n  bs4 headline fg IoU K=2 : {sc['headline_fg_iou_tol_pct']:.6f}")
    print(f"  bs4 two-class mIoU K=2  : {sc['miou_k2_twoclass']:.6f}")
    chk = sc.get("self_check") or {}
    if chk:
        d = chk["delta"]
        print(f"  self-check at batch {a.batch}: side A {chk['value']+d:.6f}  "
              f"side B {chk['value']:.6f}  delta {d:+.6f}")
        print(f"    vs old threshold {OLD_THRESHOLD}: "
              f"{'PASS' if abs(d) <= OLD_THRESHOLD else 'FAIL'}")
        print(f"    vs E-9 span {E9_SPAN}      : "
              f"{'PASS' if abs(d) <= E9_SPAN else 'FAIL'}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"run_id": new_id, "parent": a.run_id, "batch": a.batch,
                   "scoring": sc, "side_b_both_recipes": sb,
                   "dump_dir": dump_dir,
                   "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  open(a.out, "w"), indent=2, default=str)
        print(f"  wrote {a.out}")
    print(f"\ndump {dump_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
