#!/usr/bin/env python3
"""
Arm F SAR on the three INTERNAL CrossEarth vanilla LOCO checkpoints, hr, s0.

IDENTICAL HARNESS, TWO THINGS CHANGED
-------------------------------------
`support/armf_sar_grid.py` ran the published nine-cell internal SAR grid
(mamnet / oglanet / dinov3 x chicago / miami / phoenix) and
`sar/armf_sar_n450.py` re-ran it on the n450 MAMNet checkpoints. This
driver runs the SAME two passes through the SAME pass module,
`shadowtransfer/armf_sar_pass.py`, imported UNCHANGED, over the SAME canonical
150-image held-out-city test list in the fixed sorted-filename order, batch
size 1, augmentation off, episodic. Two inputs differ:

  1. the architecture is `crossearth`, which `armf_sar_cells.build_model`
     refuses. The four CrossEarth-shaped branches are NOT re-derived here:
     `shadowtransfer/armA_t1_cesar_adapt.install()` is called verbatim, which
     is the module ARM A TAIL block 1 used for the nine S-EO CrossEarth SAR
     cells. That installs the D-B3-3 log(normalised score) wrapper, the
     engine's own checkpoint loader, and the BRANCH-4 episodic snapshot over
     named_parameters()+named_buffers().
  2. the images are the INTERNAL LOCO test split
     (`armf_crossearth_cells.load_split`), not the S-EO stream. This is the
     one thing block 1's driver cannot supply: `armA_t1_cesar_run.py` resolves
     its images through `armA_a4_sar_seo`.

`armf_sar_grid.py`, `armf_sar_n450.py`, `armf_sar_pass.py`, `armf_sar_cells.py`,
`sar.py` and `armA_t1_cesar_adapt.py` are NOT edited. They produced published
numbers and stay frozen (CLAUDE.md).

TWO DECLARED DEVIATIONS, both in every results.json
---------------------------------------------------
D-B3-3   SAR consumes log(normalised two-class Mask2Former score). CrossEarth
         exposes no logits. Same ruling, same transform, same module as the
         nine S-EO CrossEarth SAR cells.
BRANCH 4 episodic snapshot/restore/compare over named_parameters() +
         named_buffers(), because ReinsDinoVisionTransformer.state_dict pops
         every non-rein key.

THE RECIPE IS THE S-EO CrossEarth RECIPE, AND THAT IS A BORROWING
-----------------------------------------------------------------
lr 2.5e-4 (1x `sar.SAR_LR_BASE`), 10 steps, momentum 0.9, rho 0.05 -- the a4
pilot selection the nine S-EO CrossEarth cells ran, in which CROSSEARTH WAS
ABSENT FROM THE SELECTION POOL (the pilot ran mamnet / oglanet / dinov3) and
which was selected on S-EO source-city validation, not on internal data. The
internal published grid's own stage-1 selection
(`runs/armF_sar_lr_pilot/selected_config.json`) is lr 2.5e-4 x1.0 with
steps=1, so the step count here is the operator's instruction, not that file.
Declared in `recipe_provenance`, never presented as a selection.

PRECISION DIFFERS FROM THE PARENT CELL, BY DESIGN
-------------------------------------------------
The parent cells' own inference ran under `torch.autocast(bfloat16)`
(`armf_crossearth_engine.evaluate:140`). The SAR forward path is fp32, as it
was for the nine S-EO cells, because SAR takes gradients through it. So the
unadapted pass is NOT expected to reproduce the parent's stored `test`
numbers bit-for-bit, and `unadapted_vs_parent_test` is recorded and WARNed,
never a gate. Every SAR number in this cell is a delta against the unadapted
pass on the same checkpoint and the same images, computed in this run, so no
parent value enters any comparison.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
BASE = SCRIPT_PATH.parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                   # noqa: E402

from armf_sar_naming_patch import counts_agg                         # noqa: E402
from shadowtransfer import (armc_runtime, armf_crossearth_cells,     # noqa: E402
                            armf_crossearth_runtime, armf_sar_meta,
                            armf_sar_pass, sar)
from shadowtransfer import armA_t1_cesar_adapt as CE                 # noqa: E402
from shadowtransfer.armc_dump import sha256_file                     # noqa: E402
from shadowtransfer.armf_sar_cells import (ARM, RES_DIR,             # noqa: E402
                                           WEAKNESS_IDS, coverage_band)

STAGE = "grid_crossearth_internal"
ARCH = "crossearth"
PARENT_RUNG = 450
PARENT_SEED = 0
WARN_TOL_MIOU = 0.01
DEFAULT_PARENT_DUMPS = os.path.join(os.environ["ST_SCRATCH_ROOT"], "dumps/st")
DEFAULT_OUT_ROOT = os.path.join(os.environ["ST_SCRATCH_ROOT"], "runs")
PUBLISHED_INTERNAL_SELECTED = BASE / "runs/armF_sar_lr_pilot/selected_config.json"


def parent_run_id(city):
    return f"{ARCH}-vanilla-loco-r{PARENT_RUNG}-{city}-hr-s{PARENT_SEED}"


def resolve_parent_cell(city, parent_dumps, repo_root, backbone_ckpt):
    """The internal CrossEarth vanilla LOCO cell this SAR cell adapts.

    Everything comes from the parent's own run.json: the checkpoint path, the
    `config_verbatim` the cell was built with, and its stored `test` block.
    Nothing is guessed. `_ce_*` keys are added exactly as
    `CE.resolve_vanilla_cell` adds them for the S-EO cells, and num_classes is
    NOT read from config_verbatim (it carries none) -- it comes from the arm's
    own `armf_crossearth_runtime.build_cfg(num_classes=2)` and is asserted on
    the built model inside `CE.build_model_crossearth`.
    """
    rid = parent_run_id(city)
    rj = Path(parent_dumps) / rid / "run.json"
    if not rj.exists():
        raise SystemExit(f"FATAL SAR needs the internal vanilla LOCO cell "
                         f"{rid}; its run.json is absent at {rj}. SAR adapts "
                         f"an existing checkpoint and cannot create one.")
    run = json.loads(rj.read_text())
    ckpt = run.get("ckpt_path")
    if not ckpt or not Path(ckpt).exists():
        raise SystemExit(f"FATAL {rid}: ckpt_path {ckpt!r} from its run.json "
                         f"does not exist on disk.")
    if run.get("arch") != ARCH or run.get("method") != "vanilla" \
            or run.get("mode") != "loco" or run.get("holdout") != city:
        raise SystemExit(f"FATAL {rid} run.json describes "
                         f"{run.get('arch')}/{run.get('method')}/"
                         f"{run.get('mode')}/{run.get('holdout')}, not "
                         f"{ARCH}/vanilla/loco/{city}")
    cfg = dict(run.get("config_verbatim") or {})
    cfg["_ce_repo_root"] = str(repo_root)
    cfg["_ce_backbone_ckpt"] = str(backbone_ckpt)
    cfg["_ce_crop"] = int(cfg.get("crop", CE.CROP))
    stored = dict(run.get("test") or {})
    prov = {
        "parent_run_json": str(rj),
        "parent_ckpt_path": str(ckpt),
        "parent_config_sha256": run.get("config_sha256"),
        "parent_backbone_init": run.get("backbone_init"),
        "parent_metric_module_sha": run.get("metric_module_sha"),
        "parent_best_val_metric": run.get("best_val_metric"),
        "parent_steps_to_best": run.get("steps_to_best"),
        "parent_wall_clock_s": run.get("wall_clock_s"),
        "config_verbatim_keys": sorted(k for k in cfg if not k.startswith("_ce_")),
        "ce_keys_added": ["_ce_repo_root", "_ce_backbone_ckpt", "_ce_crop"],
        "num_classes_source": (
            "armf_crossearth_runtime.build_cfg(num_classes=2) -- the arm's own "
            "CrossEarth constructor, the same call that built the parent cell. "
            "The parent's config_verbatim carries no num_classes key."),
        "parent_inference_precision": (
            "bf16 autocast except deformable attention in fp32 "
            "(armf_crossearth_engine.evaluate); this SAR pass is fp32"),
    }
    return Path(ckpt), cfg, run, stored, prov


def test_images_internal(base_path, city, res_dir, limit=0):
    """The held-out city's canonical 150-image LOCO test split.

    Route: `armf_crossearth_cells.load_split(base, 'loco', city, 'test')`, the
    same function and the same sorted-filename order the parent cell's own
    training script used for its test set, so the image list and its order are
    the parent's, not a new draw. `image_id` is `armf_crossearth_cells.
    image_id_of`, which strips the LOCO source-city prefix exactly as the
    parent dump's `image_id` column does.

    `x` is float32 holding RAW 0..255 RGB and that is deliberate: it is
    `CE.CrossEarthSARModel`'s contract, because CrossEarth's
    SegDataPreProcessor does its own normalisation. GT is plain {0,1} from the
    mask PNG at >127, the parent's own threshold.
    """
    import torch
    from PIL import Image

    imgs, msks, meta = armf_crossearth_cells.load_split(
        str(base_path), "loco", city, "test", "hr" if res_dir == "highres" else "mr")
    if limit:
        imgs, msks = imgs[:limit], msks[:limit]
    cities = sorted({r.get("city") for r in meta})
    if cities != [city]:
        raise SystemExit(f"FATAL loco test split for {city} reports cities "
                         f"{cities}; the test stream must be the held-out city "
                         f"only or the cell is not deployment_time")
    out = []
    for ip, mp in zip(imgs, msks):
        arr = np.asarray(Image.open(ip).convert("RGB"))
        gt = (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.uint8)
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy()).float().unsqueeze(0)
        tf = float(gt.mean())
        out.append({
            "image_id": armf_crossearth_cells.image_id_of(ip),
            "x": x,
            "gt": gt,
            "true_fg_fraction": tf,
            "coverage_band": coverage_band(tf),
            "img_path": ip,
        })
    if not out:
        raise SystemExit(f"FATAL loco test split for {city} resolved to 0 images")
    return out, {
        "route": ("armf_crossearth_cells.load_split(base, 'loco', city, "
                  "'test') -- the parent cell's own test-set route"),
        "split_root": armf_crossearth_cells.split_root(str(base_path), "loco", city),
        "n_images": len(out),
        "image_dtype": "float32 holding raw 0..255 RGB (SegDataPreProcessor normalises)",
        "gt_source": "mask PNG > 127, plain {0,1}",
        "unseen_handling": ("no don't-care class in the internal benchmark; "
                            "the K=2 band is the only exclusion and it comes "
                            "from shadowtransfer.metrics"),
        "first_image_id": out[0]["image_id"],
        "last_image_id": out[-1]["image_id"],
        "order": "sorted image filename, the order fixed by armf_crossearth_cells",
    }


def two_class_from_rows(rows, pass_id, step):
    """Two-class mIoU at K=2 and foreground-only IoU at K=2 from one pass's
    stored per-image counts, via `armf_sar_naming_patch.counts_agg` -- the
    function that wrote the nine published internal cells' value. No metric is
    reimplemented here (CONTRACT.md rule zero)."""
    import pandas as pd
    sub = pd.DataFrame([r for r in rows
                        if r["pass_id"] == pass_id and r["step"] == step])
    miou2, fg, n, n_degen, n_empty_gt = counts_agg(sub)
    return {"miou_k2_twoclass": miou2, "iou_fg_k2": fg, "n_images": int(n),
            "n_images_fg_denominator_zero": n_degen,
            "n_images_empty_ground_truth": n_empty_gt}


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True,
                    choices=list(armf_crossearth_cells.CITIES))
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--lr", type=float, default=sar.SAR_LR_BASE)
    ap.add_argument("--lr_multiplier", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--rho", type=float, default=sar.SAM_RHO)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N images")
    ap.add_argument("--parent_dumps", default=DEFAULT_PARENT_DUMPS)
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--tag", default="")
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--backbone_ckpt", required=True)
    ap.add_argument("--backbone_sha256", required=True)
    ap.add_argument("--backbone_bytes", type=int, required=True)
    ap.add_argument("--timeout_s", type=int, default=900)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def verify_backbone(path, sha256, nbytes):
    """The backbone is re-verified on disk in the job, not trusted from the
    submit script's constant."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"FATAL backbone {p} absent")
    got_bytes = p.stat().st_size
    got_sha = sha256_file(p)
    if got_bytes != int(nbytes) or got_sha != sha256:
        raise SystemExit(f"FATAL backbone mismatch at {p}: bytes {got_bytes} "
                         f"vs {nbytes}, sha256 {got_sha} vs {sha256}")
    return {"path": str(p), "sha256": got_sha, "bytes": got_bytes,
            "reverified_on_disk": True}


def recipe_provenance(args):
    internal = None
    if PUBLISHED_INTERNAL_SELECTED.exists():
        d = json.loads(PUBLISHED_INTERNAL_SELECTED.read_text())
        internal = {k: d.get(k) for k in
                    ("lr", "lr_multiplier", "steps", "pilot_verdict",
                     "selection_split")}
    return {
        "lr": args.lr,
        "lr_multiplier": args.lr_multiplier,
        "steps": args.steps,
        "momentum": args.momentum,
        "rho": args.rho,
        "lr_base": sar.SAR_LR_BASE,
        "source": ("the SAR recipe the nine S-EO CrossEarth SAR cells ran "
                   "(ARM A TAIL block 1): lr 1x lr_base, 10 steps"),
        "deviation": ("the recipe is BORROWED twice over and neither borrowing "
                      "is hidden: (a) the a4 pilot that selected lr1x/s10 ran "
                      "mamnet, oglanet and dinov3 -- CROSSEARTH WAS ABSENT "
                      "FROM THE SELECTION POOL; (b) that pilot selected on "
                      "S-EO source-city validation, not on internal data. No "
                      "internal CrossEarth cell selected this recipe."),
        "internal_published_selection_for_reference": internal,
        "internal_published_selection_path": (
            str(PUBLISHED_INTERNAL_SELECTED)
            if PUBLISHED_INTERNAL_SELECTED.exists() else None),
        "internal_published_selection_note": (
            "the internal published grid's own stage-1 selection carries "
            "steps=1. It is recorded for reference and is NOT what runs here; "
            "the 10-step recipe is the operator's instruction, matching the "
            "S-EO CrossEarth cells."),
        "selected_on_heldout_city": False,
    }


def main():
    args = _cli()
    t_start = time.time()
    res_dir = RES_DIR[args.res]
    install_report = CE.install()
    backbone = verify_backbone(args.backbone_ckpt, args.backbone_sha256,
                               args.backbone_bytes)
    ckpt, cfg, parent_run, parent_test, parent_prov = resolve_parent_cell(
        args.city, args.parent_dumps, args.repo_root, args.backbone_ckpt)
    recipe = recipe_provenance(args)
    specs = [armf_sar_pass.unadapted_spec(),
             armf_sar_pass.sar_spec(args.lr, args.lr_multiplier, args.steps)]

    out_dir = Path(args.out_root) / \
        f"armF_sar_{ARCH}_{args.city}_{res_dir}{args.tag}"
    results_path, dump_path = out_dir / "results.json", out_dir / "perstep.parquet"

    print("=" * 78)
    print(f"ARM F SAR on the INTERNAL CrossEarth vanilla LOCO checkpoint   "
          f"city={args.city}  res={args.res}")
    print(f"  parent run_id   {parent_run_id(args.city)}")
    print(f"  parent run.json {parent_prov['parent_run_json']}")
    print(f"  checkpoint      {ckpt}")
    print(f"  backbone        {backbone['path']}  {backbone['sha256'][:16]}... "
          f"{backbone['bytes']} B (reverified)")
    print(f"  repo_root       {args.repo_root}")
    print(f"  parent stored   two-class mIoU K=2 {parent_test.get('miou_tol2')}  "
          f"fg-only {parent_test.get('iou_fg_tol2')}  "
          f"({parent_prov['parent_inference_precision']})")
    print(f"  recipe          lr {args.lr:g} ({args.lr_multiplier:g}x lr_base "
          f"{sar.SAR_LR_BASE})  steps {args.steps}  momentum {args.momentum}  "
          f"rho {args.rho}   BORROWED, see recipe_provenance")
    print(f"  E_0             {sar.e0_margin(2):.6f} nats   reset constant "
          f"{sar.reset_constant(2):.6f} nats")
    print(f"  norm mode       {armf_sar_meta.NORM_MODE}")
    print(f"  passes          {[s['pass_id'] for s in specs]}")
    print(f"  rebindings      {json.dumps(install_report['rebindings'])}")
    print(f"  out             {out_dir}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return 0
    if (out_dir / ".DONE").exists():
        print(f"skip: {out_dir}/.DONE exists")
        return 0

    armc_runtime.preflight_write_targets(BASE, [results_path, dump_path],
                                         label="armF SAR crossearth internal")
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from shadowtransfer.arme_common import seed_all
    seed_sites = seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CE.enter_arch_tree(args.repo_root)
    images, split_meta = test_images_internal(BASE, args.city, res_dir, args.limit)
    print(f"\nTEST SET {len(images)} images, {split_meta['order']}, "
          f"augmentation off, batch size 1; first {split_meta['first_image_id']}, "
          f"last {split_meta['last_image_id']}", flush=True)

    # ---- BRANCH-4 machine check, before any pass ---------------------------
    probe = armf_sar_pass.build_model(ARCH, cfg, device)
    armf_sar_pass.load_checkpoint_strict(probe, ckpt, device)
    probe, _bn = sar.configure_model(probe, sar.BN_EVAL_SOURCE_STATS)
    branch4 = CE.assert_snapshot_covers_collector(probe)
    collector = CE.collector_report(probe)
    load_report = dict(CE.LOAD_REPORT)
    print(f"[branch4] {json.dumps(branch4)}", flush=True)
    print(f"[collector] {collector['n_collected_params']} params, trunk blocks "
          f"{collector['trunk_blocks_collected']}", flush=True)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summaries, rows = [], []
    base_iou_by_image, base_fg0_by_image = {}, {}
    for spec in specs:
        s, r = armf_sar_pass.run_pass(spec, ARCH, cfg, ckpt, device, images,
                                      rho=args.rho, momentum=args.momentum)
        summaries.append(s)
        rows.extend(r)
        if spec["pass_id"] == "unadapted":
            fin = [x for x in r if x["step"] == 0]
            base_iou_by_image = {x["image_id"]: x["iou_fg_tol2"] for x in fin}
            base_fg0_by_image = {x["image_id"]: x["pred_fg_fraction"] for x in fin}
            s["delta_vs_unadapted"] = 0.0
        else:
            s["delta_vs_unadapted"] = s["miou_fg_tol2"] - summaries[0]["miou_fg_tol2"]

    armf_sar_pass.mark_divergence(summaries)
    s0 = armf_sar_pass.step0_check(rows, summaries, base_fg0_by_image)
    una, sarp = summaries[0], summaries[1]

    two_un = two_class_from_rows(rows, "unadapted", 0)
    two_s0 = two_class_from_rows(rows, sarp["pass_id"], 0)
    two_sf = two_class_from_rows(rows, sarp["pass_id"], args.steps)
    agg_method = (f"macro-average over {two_un['n_images']} images of the mean "
                  f"of foreground and background IoU at K=2, computed from "
                  f"stored tp/fp/fn/tn by armf_sar_naming_patch.counts_agg")

    # ---- the parent comparison: WARN, never a gate -------------------------
    stored_two = parent_test.get("miou_tol2")
    stored_fg = parent_test.get("iou_fg_tol2")
    d_two = None if stored_two is None else two_un["miou_k2_twoclass"] - float(stored_two)
    d_fg = None if stored_fg is None else una["miou_fg_tol2"] - float(stored_fg)
    print(f"WARN unadapted_vs_parent_test  two-class K=2 computed "
          f"{two_un['miou_k2_twoclass']:.6f} stored {stored_two} difference "
          f"{d_two}; fg-only computed {una['miou_fg_tol2']:.6f} stored "
          f"{stored_fg} difference {d_fg}. Recorded, not a gate: the parent's "
          f"inference ran under bf16 autocast and this pass is fp32, so a "
          f"small difference is expected. Every SAR number here is a delta "
          f"against the unadapted pass on the same checkpoint and the same "
          f"{len(images)} images.", flush=True)

    episodic_all = all(s["episodic_state_isolation"] for s in summaries)
    if not episodic_all:
        armc_runtime.mark_failed(out_dir, "episodic_isolation",
                                 "model state was not byte-identical between images")
        raise SystemExit("FATAL episodic state isolation failed; the arm F spec "
                         "stops the run. Per-pass detail is in the log above.")

    caveats = armf_sar_meta.base_caveats(
        lr=args.lr, lr_multiplier=args.lr_multiplier, steps=args.steps,
        rho=args.rho, momentum=args.momentum, config_a_collect="norm_all")
    caveats.append(recipe["deviation"])
    caveats.append(armf_crossearth_cells.RELEASE_CAVEAT)
    caveats.append("D-B3-3: SAR consumes log(normalised two-class Mask2Former "
                   "score) because CrossEarth exposes no logits. Same ruling, "
                   "transform and module as the nine S-EO CrossEarth SAR cells.")
    caveats.append("Episodic snapshot/restore/compare go over "
                   "named_parameters()+named_buffers(), not state_dict(), "
                   "because ReinsDinoVisionTransformer.state_dict pops every "
                   "non-rein key (BRANCH 4).")
    caveats.append(f"The parent cell's own inference ran under bf16 autocast "
                   f"({parent_prov['parent_inference_precision']}); this SAR "
                   f"pass is fp32, so the unadapted pass is not expected to "
                   f"reproduce the parent's stored test numbers bit-for-bit. "
                   f"unadapted_vs_parent_test records the difference and does "
                   f"not gate.")
    caveats.append("Upstream's skip list is ViT-Base shaped; on CrossEarth's "
                   "24-block DINOv2 ViT-L trunk blocks.9/10/11 are mid-trunk "
                   "and blocks 21/22/23 ARE collected. Transferred as written, "
                   "reported, not judged.")
    if args.limit:
        caveats.append(f"SMOKE RUN: {args.limit} images. Not a result.")
    ctx = dict(
        args=args, t_start=t_start, out_dir=out_dir, results_path=results_path,
        dump_path=dump_path, images=images, rows=rows, summaries=summaries,
        una=una, sarp=sarp, s0=s0, two_un=two_un, two_s0=two_s0, two_sf=two_sf,
        agg_method=agg_method, d_two=d_two, d_fg=d_fg, stored_two=stored_two,
        stored_fg=stored_fg, ckpt=ckpt, cfg=cfg, parent_prov=parent_prov,
        parent_test=parent_test, recipe=recipe, backbone=backbone,
        install_report=install_report, branch4=branch4, collector=collector,
        load_report=load_report, split_meta=split_meta,
        base_iou_by_image=base_iou_by_image, seed_sites=seed_sites,
        caveats=caveats, episodic_all=episodic_all)
    return _write(ctx)


def _write(c):
    """Assemble results.json in the INTERNAL arm F SAR schema and write the
    dump. Key names follow the nine published internal cells after
    `armf_sar_naming_patch` (iou_fg_tol2 as the pass headline,
    miou_tol2_derived_from_counts for the two-class value), so a reader can
    put a CrossEarth cell beside a MAMNet one without a mapping."""
    args, una, sarp = c["args"], c["una"], c["sarp"]
    rows, summaries, images = c["rows"], c["summaries"], c["images"]
    two_un, two_s0, two_sf = c["two_un"], c["two_s0"], c["two_sf"]
    out_dir, dump_path = c["out_dir"], c["dump_path"]

    results = {
        "arm": ARM,
        "stage": STAGE,
        "weakness_ids": WEAKNESS_IDS,
        "target_access": armf_sar_meta.TARGET_ACCESS,
        "target_access_detail": armf_sar_meta.TARGET_ACCESS_DETAIL,
        "arch": ARCH,
        "architecture": "CrossEarth",
        "dataset": "st",
        "holdout_city": args.city,
        "resolution": args.res,
        "seed": args.seed,
        "trains": False,
        "trains_note": ("SAR trains nothing; it adapts LayerNorm/BatchNorm "
                        "affine parameters of an existing vanilla LOCO "
                        "checkpoint at test time. No checkpoint is written."),
        "parent_run_id": parent_run_id(args.city),
        "parent_run_dir": None,
        "parent_provenance": c["parent_prov"],
        "parent_stored_test": c["parent_test"],
        "checkpoint_path": str(c["ckpt"]),
        "checkpoint_sha256": sha256_file(c["ckpt"]),
        "backbone_init": c["backbone"],
        "crossearth_missing_from_release": list(
            armf_crossearth_runtime.MISSING_FROM_RELEASE),
        "metric": dict(armf_sar_meta.METRIC_BLOCK),
        "reporting_metric": ("iou_fg_boundary_tolerant_K2 (Appendix E); "
                             "miou_2class_boundary_tolerant_K2 derived from the "
                             "same stored counts and reported beside it"),
        "prob_source": ("softmax(1)[0,1] of log(normalised two-class "
                        "Mask2Former score), which equals "
                        "armf_crossearth_runtime.foreground_prob exactly. See "
                        "declared_deviations.d_b3_3_logit_definition."),
        "prediction_recipe": ("shadowtransfer.metrics.threshold_and_filter at "
                              "0.5 with the lifted filter_small_predictions("
                              "min_pixels=10), via armf_sar_pass.run_pass"),
        "sar_config": armf_sar_meta.sar_config_block(
            lr=args.lr, lr_multiplier=args.lr_multiplier, steps=args.steps,
            rho=args.rho, momentum=args.momentum,
            collected_by_pass={s["pass_id"]: s["collected_params"] for s in summaries},
            names_by_pass={s["pass_id"]: s["n_collected_params"] for s in summaries},
            selection_split="seo_source_city_val (BORROWED; see recipe_provenance)"),
        "recipe_provenance": c["recipe"],
        "declared_deviations": {
            "d_b3_3_logit_definition": c["install_report"]["logit_deviation"],
            "branch4_episodic_snapshot": c["install_report"]["snapshot_deviation"],
            "precision_vs_parent": {
                "deviation": "the SAR forward path is fp32; the parent cell's "
                             "own inference ran under bf16 autocast",
                "why": ("SAR takes gradients through the forward path and the "
                        "nine S-EO CrossEarth SAR cells ran it in fp32; "
                        "matching the parent's autocast would change the "
                        "gradient path and break comparability with those "
                        "cells"),
                "effect": ("the unadapted pass is not expected to reproduce "
                           "the parent's stored test numbers bit-for-bit; "
                           "sanity_checks.unadapted_vs_parent_test records the "
                           "difference and does not gate"),
            },
            "files_unedited": [
                "shadowtransfer/sar.py", "shadowtransfer/armf_sar_pass.py",
                "shadowtransfer/armf_sar_cells.py",
                "shadowtransfer/armA_t1_cesar_adapt.py",
                "armf_sar_grid.py", "armf_sar_n450.py"],
        },
        "arch_branch_resolutions": c["install_report"]["rebindings"],
        "checkpoint_load_report": c["load_report"],
        "sar_collector_exposure": c["collector"],
        "branch4_snapshot_coverage": c["branch4"],
        "split_meta": c["split_meta"],
        "unadapted": {
            "iou_fg_tol2": una["miou_fg_tol2"],
            "mean_pred_fg_fraction": una["mean_pred_fg_fraction_final"],
            "miou_tol2_derived_from_counts": {
                "value": two_un["miou_k2_twoclass"],
                "method": c["agg_method"],
                "n_images": two_un["n_images"],
                "n_images_fg_denominator_zero": two_un["n_images_fg_denominator_zero"],
                "n_images_empty_ground_truth": two_un["n_images_empty_ground_truth"],
            },
        },
        "sar": {
            "iou_fg_tol2": sarp["miou_fg_tol2"],
            "delta_vs_unadapted_iou_fg_tol2": sarp["delta_vs_unadapted"],
            "mean_pred_fg_fraction_step0": sarp["mean_pred_fg_fraction_step0"],
            "mean_pred_fg_fraction_final": sarp["mean_pred_fg_fraction_final"],
            "n_images_with_zero_foreground_prediction":
                sarp["n_images_with_zero_foreground_prediction"],
            "n_reset_fires": sarp["n_reset_fires"],
            "n_images_pred_fg_below_0p01": sarp["n_images_pred_fg_below_0p01"],
            "frac_images_pred_fg_below_0p01": sarp["frac_images_pred_fg_below_0p01"],
            "diverged": sarp["diverged"],
            "divergence_note": sarp["divergence_note"],
            "n_updates_skipped": sarp["n_updates_skipped"],
            "n_images_state_changed_by_adaptation":
                sarp["n_images_state_changed_by_adaptation"],
            "iou_fg_tol2_step0": two_s0["iou_fg_k2"],
            "iou_fg_tol2_final": two_sf["iou_fg_k2"],
            "delta_adaptation_only_iou_fg_tol2":
                two_sf["iou_fg_k2"] - two_s0["iou_fg_k2"],
            "delta_adaptation_only_definition": (
                "final-step minus step-0, both macro-means over the same "
                "images within the same SAR pass; isolates the adaptation "
                "effect from any step-0 forward-path difference"),
            "miou_tol2_derived_from_counts": {
                "value": two_sf["miou_k2_twoclass"],
                "value_step0": two_s0["miou_k2_twoclass"],
                "value_final": two_sf["miou_k2_twoclass"],
                "method": c["agg_method"],
                "n_images": two_sf["n_images"],
            },
            "delta_adaptation_only_miou_tol2":
                two_sf["miou_k2_twoclass"] - two_s0["miou_k2_twoclass"],
            "delta_vs_unadapted_miou_tol2":
                two_sf["miou_k2_twoclass"] - two_un["miou_k2_twoclass"],
        },
        "per_band": armf_sar_pass.per_band_blocks(rows, summaries,
                                                  c["base_iou_by_image"]),
        "per_image_dump_path": str(dump_path),
        "sanity_checks": {
            "episodic_state_isolation": bool(c["episodic_all"]),
            "episodic_state_isolation_detail": {
                s["pass_id"]: {"ok": s["episodic_state_isolation"],
                               "mismatch_keys": s["episodic_mismatch_keys"]}
                for s in summaries},
            "collected_params_nonempty": all(s["n_collected_params"] >= 4
                                             for s in summaries),
            "step0_matches_unadapted": c["s0"],
            "branch4_snapshot_covers_collector": c["branch4"]["verdict"],
            "backbone_sha256_reverified": c["backbone"]["reverified_on_disk"],
            "test_stream_is_heldout_city_only": True,
            "no_label_reaches_adaptation": (
                "sar.adapt_image takes (model, x, steps, ...) and no label; "
                "CE.CrossEarthSARModel builds data_samples with METAINFO ONLY "
                "and never a gt_sem_seg"),
            "unadapted_vs_parent_test": {
                "computed_miou_k2_twoclass": two_un["miou_k2_twoclass"],
                "stored_miou_k2_twoclass": c["stored_two"],
                "difference_miou_k2_twoclass": c["d_two"],
                "computed_iou_fg_k2": una["miou_fg_tol2"],
                "stored_iou_fg_k2": c["stored_fg"],
                "difference_iou_fg_k2": c["d_fg"],
                "stored_source": c["parent_prov"]["parent_run_json"],
                "stored_field": "test.miou_tol2 / test.iou_fg_tol2",
                "warn_tolerance_reported": WARN_TOL_MIOU,
                "within_reported_tolerance": (
                    None if c["d_two"] is None
                    else bool(abs(c["d_two"]) <= WARN_TOL_MIOU)),
                "action": ("WARN only, never a gate: the parent ran bf16 "
                           "autocast and this pass is fp32. No parent value "
                           "enters any comparison; every SAR number is a "
                           "delta against the unadapted pass in this run."),
            },
        },
        "caveats": c["caveats"],
        "compute_note": armf_sar_meta.COMPUTE_NOTE,
        "cluster": "delta",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_time_limit": os.environ.get("ARMF_TIME_LIMIT"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "git_hash": None,
        "git_hash_reason": "project root is not a git repository",
        "wall_clock_s": None,
        "s_per_image": None,
        "peak_gpu_mem_gb": max([s["peak_gpu_mem_gb"] for s in summaries
                                if s["peak_gpu_mem_gb"] is not None] or [None]),
        "per_pass": [{k: s[k] for k in (
            "pass_id", "config_label", "steps", "lr", "lr_multiplier",
            "miou_fg_tol2", "delta_vs_unadapted", "mean_pred_fg_fraction_step0",
            "mean_pred_fg_fraction_final", "n_reset_fires", "n_updates_skipped",
            "n_collected_params", "norm_mode", "episodic_state_isolation",
            "wall_clock_s", "s_per_image", "peak_gpu_mem_gb",
            "checkpoint_epoch")} for s in summaries],
        "provenance": armf_sar_meta.provenance_block(
            SCRIPT_PATH, c["seed_sites"], len(images),
            extra={"smoke": bool(args.limit),
                   "ce_adapter_module_sha256": sha256_file(
                       BASE / "shadowtransfer/armA_t1_cesar_adapt.py"),
                   "crossearth_runtime_sha256": sha256_file(
                       BASE / "shadowtransfer/armf_crossearth_runtime.py"),
                   "crossearth_engine_sha256": sha256_file(
                       BASE / "shadowtransfer/armf_crossearth_engine.py"),
                   "crossearth_cells_sha256": sha256_file(
                       BASE / "shadowtransfer/armf_crossearth_cells.py"),
                   "naming_patch_sha256": sha256_file(
                       BASE / "support/armf_sar_naming_patch.py"),
                   "repo_root": str(args.repo_root),
                   "test_list_order": c["split_meta"]["route"]}),
    }
    results["wall_clock_s"] = round(time.time() - c["t_start"], 2)
    results["s_per_image"] = round(results["wall_clock_s"] /
                                   max(1, len(images) * len(summaries)), 4)

    with armc_runtime.PhaseTimeout(args.timeout_s, out_dir,
                                   label="armF SAR crossearth write") as tbx:
        tbx.phase("write_parquet")
        import pandas as pd
        pd.DataFrame(rows).to_parquet(dump_path, index=False)
        tbx.phase("write_results_json")
        c["results_path"].write_text(
            json.dumps(results, indent=2, default=str) + "\n")
    (out_dir / ".DONE").touch()

    print("\n" + "=" * 78)
    print(f"ARM F SAR crossearth DONE  {args.city}/{args.res}   "
          f"{len(rows)} rows -> {dump_path}")
    print(f"  results        {c['results_path']}")
    print(f"  unadapted      two-class K=2 {two_un['miou_k2_twoclass']:.4f}   "
          f"fg-only {una['miou_fg_tol2']:.4f}   pred_fg "
          f"{una['mean_pred_fg_fraction_final']:.5f}")
    print(f"  sar step{args.steps:<3}   two-class K=2 "
          f"{two_sf['miou_k2_twoclass']:.4f}   fg-only "
          f"{sarp['miou_fg_tol2']:.4f}")
    print(f"  delta          two-class "
          f"{results['sar']['delta_vs_unadapted_miou_tol2']:+.6f}   fg-only "
          f"{sarp['delta_vs_unadapted']:+.6f}   resets "
          f"{sarp['n_reset_fires']}   diverged {sarp['diverged']}")
    print(f"  vs parent      two-class computed {two_un['miou_k2_twoclass']:.4f} "
          f"stored {c['stored_two']} difference {c['d_two']}")
    print(f"  wall           {results['wall_clock_s']:.0f}s   peak VRAM "
          f"{results['peak_gpu_mem_gb']} GB")
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
