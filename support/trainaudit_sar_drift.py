#!/usr/bin/env python3
"""
Does the test-time method actually move any weights, and by how much?
(operator request, 2026-10-25, item 3). CHECK ONLY -- no training, no
checkpoint written, no published artifact touched.

`armF_sar_*/results.json` already records `n_images_state_changed_by_adaptation`,
but that is a BYTE-IDENTITY count: it answers "did anything move at all", not
"by how much". This probe measures the magnitude.

Per image, episodically:
  1. restore the pristine state and byte-check the restore;
  2. run `sar.adapt_image` for the cell's own selected number of steps;
  3. with the adapted state still in place, diff EVERY tensor of the
     state_dict against the pristine snapshot.

Reported per cell:
  * `max_abs_delta`      the largest |adapted - start| over all parameters and
                         all images, with the parameter name and image it came
                         from
  * `max_rel_delta`      the same as a fraction of |start| at that element
  * `n_params_moved`     how many of the collected parameters ever moved
  * `moved_outside_collected`  any tensor that moved but was NOT collected by
                         SAR -- should be empty; a non-empty list is a finding

Two architectures are handled by two branches, because CrossEarth is not one of
`armf_sar_cells.build_model`'s three cases (see
`shadowtransfer/armA_t1_cesar_adapt.py`, branches 1-3): it is built through
mmsegmentation's registry and its checkpoint carries `state_dict`, not
`model_state_dict`. The dinov3 branch is the arm F ShadowTransfer cell; the
crossearth branch is the arm A S-EO cell, which is the only CrossEarth SAR cell
that exists.
"""

import argparse
import json
import os
import platform
import sys
import time

BASE = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

RES_DIR = {"hr": "highres", "mr": "midres"}


#: Set by a branch whose architecture needs a non-default state reader.
CTX = {"state_fn": None, "state_note": None}


def diff_state(model, pristine, state_fn=None):
    """(rows, summary). One row per tensor that moved. Compares the FULL
    state_dict, not just the collected parameters, so a parameter SAR did not
    claim to adapt cannot move unnoticed."""
    import torch
    cur = state_fn(model) if state_fn else model.state_dict()
    rows = []
    for k, v in cur.items():
        # armA_t1_cesar_adapt._full_state prefixes keys "P:"/"B:"; the
        # collected-parameter names are unprefixed. No other arch uses ":".
        base = k.split(":", 1)[1] if ":" in k else k
        p = pristine[k]
        if v.shape != p.shape or v.dtype != p.dtype:
            rows.append({"name": k, "param_name": base,
                         "shape_or_dtype_changed": True})
            continue
        if not v.is_floating_point():
            if not bool(torch.equal(v, p)):
                rows.append({"name": k, "param_name": base,
                             "non_float_changed": True})
            continue
        d = (v.detach().float() - p.detach().float()).abs()
        m = float(d.max().item())
        if m == 0.0:
            continue
        flat = d.flatten()
        i = int(flat.argmax().item())
        start = float(p.detach().float().flatten()[i].item())
        rows.append({
            "name": k,
            "param_name": base,
            "max_abs_delta": m,
            "start_value_at_argmax": start,
            "rel_delta_at_argmax": (m / abs(start)) if start != 0.0 else None,
            "l2_delta": float(d.norm().item()),
            "l2_start": float(p.detach().float().norm().item()),
            "n_elements_moved": int((flat > 0).sum().item()),
            "n_elements": int(flat.numel()),
        })
    return rows


def run_dinov3(args, out):
    from shadowtransfer import armf_sar_cells as cells
    from shadowtransfer import armf_sar_pass as sarpass
    from shadowtransfer import sar
    import torch

    res_dir = RES_DIR[args.res]
    cells.enter_arch_tree(args.arch)
    cell_dir, ckpt, cfg_path, tr_path = cells.resolve_cell(args.arch, args.city, res_dir)
    cfg = json.loads(cfg_path.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = cells.build_model(args.arch, cfg, device)
    ck_epoch = cells.load_checkpoint_strict(model, ckpt, device)
    model, bn_modules = sar.configure_model(model, sar.BN_EVAL_SOURCE_STATS)
    params, names, skipped = sar.collect_params(model)
    for p in params:
        p.requires_grad_(True)
    pristine = sar.snapshot_state(model)

    out.update(
        arch=args.arch, city=args.city, res=args.res,
        cell_dir=str(cell_dir), ckpt_path=str(ckpt),
        checkpoint_epoch=ck_epoch,
        n_bn_modules=len(bn_modules),
        n_collected_params=len(names), collected_params=names,
        skipped_norm_modules=skipped,
        skip_substrings=list(sar.SKIP_SUBSTRINGS),
        norm_mode=sar.BN_EVAL_SOURCE_STATS)

    images = cells.test_images(args.arch, args.city, res_dir, limit=args.limit)
    margin = sar.e0_margin(cells.NUM_CLASSES)
    reset_const = sar.reset_constant(cells.NUM_CLASSES)
    lr = args.lr * args.lr_multiplier

    def optimizer_factory():
        return sar.make_optimizer(params, lr=lr, momentum=0.9, rho=sar.SAM_RHO)

    out.update(sar_steps=args.steps, sar_lr=lr, sar_lr_base=args.lr,
               sar_lr_multiplier=args.lr_multiplier,
               e0_margin=margin, reset_constant=reset_const,
               n_images=len(images))
    return model, pristine, images, margin, reset_const, optimizer_factory, set(names)


def run_crossearth(args, out):
    """The arm A S-EO CrossEarth SAR cell. Three inputs the other three
    architectures do not need -- the CrossEarth repo, its converted DINOv2
    backbone, and the S-EO budget manifest that names the test patch ids --
    are resolved exactly as `python/armA_t1_cesar_run.py` resolves them, so
    this probe adapts the same model on the same images as the cell itself."""
    from shadowtransfer import armA_a4_sar_seo as SEO
    from shadowtransfer import armA_t1_cesar_adapt as ce
    from shadowtransfer import sar
    import armA_pilot_train as P
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt
    if not ckpt or not os.path.isfile(ckpt):
        raise SystemExit(f"FATAL --ckpt required and must exist for crossearth; got {ckpt!r}")
    if not args.fold:
        raise SystemExit("FATAL --fold required for crossearth (OMA|JAX|UCSD)")
    for k, v in (("--ce_repo_root", args.ce_repo_root),
                 ("--ce_backbone", args.ce_backbone)):
        if not v or not os.path.exists(v):
            raise SystemExit(f"FATAL {k} missing or absent: {v!r}")

    # CrossEarth REQUIRES armA_t1_cesar_adapt.install() before any snapshot.
    # ReinsDinoVisionTransformer.state_dict() pops every key without "rein"
    # in it, so sar.snapshot_state omits the whole frozen DINOv2 trunk and
    # sar.restore_state's strict load raises on image 0 (observed: job
    # 23242232). install() rebinds snapshot/restore/state_matches onto
    # named_parameters()+named_buffers(), which the override does not touch.
    install_report = ce.install()
    CTX["state_fn"] = ce._full_state
    CTX["state_note"] = ("armA_t1_cesar_adapt._full_state over "
                         "named_parameters()+named_buffers(); the model's own "
                         "state_dict() omits the frozen DINOv2 trunk")

    cfg = {"_ce_repo_root": args.ce_repo_root,
           "_ce_backbone_ckpt": args.ce_backbone}
    model = ce.build_model_crossearth("crossearth", cfg, device)
    info = ce.load_checkpoint_crossearth(model, ckpt, device)
    model, bn_modules = sar.configure_model(model, sar.BN_EVAL_SOURCE_STATS)
    params, names, skipped = sar.collect_params(model)
    for prm in params:
        prm.requires_grad_(True)
    pristine = sar.snapshot_state(model)
    # The adapter's own gate: every parameter SAR will adapt must be present in
    # the snapshot, or adaptation leaks across images while state_matches still
    # reports isolation intact. Raises rather than returning a wrong number.
    cover = ce.assert_snapshot_covers_collector(model)

    out.update(ce_install_rebindings=install_report.get("rebindings"),
               ce_snapshot_covers_collector=cover,
               ce_snapshot_deviation=install_report.get("snapshot_deviation"),
               full_state_reader=CTX["state_note"],
               arch="crossearth", fold=args.fold, ckpt_path=ckpt,
               checkpoint_info=info, n_bn_modules=len(bn_modules),
               n_collected_params=len(names), collected_params=names,
               skipped_norm_modules=skipped,
               skip_substrings=list(sar.SKIP_SUBSTRINGS),
               norm_mode=sar.BN_EVAL_SOURCE_STATS,
               ce_repo_root=args.ce_repo_root, ce_backbone=args.ce_backbone)

    cell, cell_path = SEO.load_cell_manifest(args.fold)
    ids = list(cell["test"])
    wrong = [q for q in ids if P.city_of(q) != args.fold]
    if wrong:
        raise SystemExit(f"FATAL {args.fold} test split carries {len(wrong)} ids "
                         f"from other cities (first {wrong[:3]})")
    roots = dict(img_root=SEO.DEF_IMG_ROOT, mask_root=SEO.DEF_MASK_ROOT,
                 patch_index=SEO.DEF_PATCH_INDEX, splits_dir=SEO.DEF_SPLITS_DIR)
    images = ce.images_from_patch_ids_crossearth(
        cfg, ids, "test", f"{args.fold}/test", args.limit, **roots)

    margin = sar.e0_margin(2)
    reset_const = sar.reset_constant(2)
    lr = args.lr * args.lr_multiplier

    def optimizer_factory():
        return sar.make_optimizer(params, lr=lr, momentum=0.9, rho=sar.SAM_RHO)

    out.update(sar_steps=args.steps, sar_lr=lr, n_images=len(images),
               budget_manifest=str(cell_path), n_test_ids=len(ids))
    return model, pristine, images, margin, reset_const, optimizer_factory, set(names)


def run_seo(args, out):
    """One arm A S-EO cell for a NON-CrossEarth architecture.

    `run_dinov3` resolves an INTERNAL ShadowTransfer cell by city and cannot
    reach S-EO at all, so an S-EO dinov3 question had no path before this.
    The cell, its checkpoint and its image stream are resolved exactly as
    `armA_a4_sar_run.py` resolves them -- `resolve_vanilla_cell` for the
    checkpoint and recipe config, `target_test_images` for the stream -- so
    this probe adapts the same model on the same images, in the same
    manifest order, as the cell itself. `target_test_images` and the
    CrossEarth branch's `load_cell_manifest` read the same budget manifest's
    `cell["test"]`, which is what makes the two cells' image sets the same
    set in the same order.
    """
    from shadowtransfer import armA_a4_sar_seo as SEO
    from shadowtransfer import sar
    import torch

    fold = args.fold.upper()
    ckpt, cfg, vrun = SEO.resolve_vanilla_cell(args.arch, fold, args.cell_seed)
    SEO.enter_arch_tree(args.arch)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SEO.build_model(args.arch, cfg, device)
    ck_epoch = SEO.load_checkpoint_strict(model, ckpt, device)
    model, bn_modules = sar.configure_model(model, sar.BN_EVAL_SOURCE_STATS)
    params, names, skipped = sar.collect_params(model)
    for p in params:
        p.requires_grad_(True)
    pristine = sar.snapshot_state(model)

    roots = dict(img_root=SEO.DEF_IMG_ROOT, mask_root=SEO.DEF_MASK_ROOT,
                 patch_index=SEO.DEF_PATCH_INDEX, splits_dir=SEO.DEF_SPLITS_DIR)
    images, split_meta = SEO.target_test_images(
        args.arch, cfg, fold, args.limit, **roots)

    out.update(
        arch=args.arch, dataset="seo", fold=fold, cell_seed=args.cell_seed,
        vanilla_run_id=SEO.vanilla_run_id(args.arch, fold, args.cell_seed),
        sar_run_id=SEO.sar_run_id(args.arch, fold, args.cell_seed),
        ckpt_path=str(ckpt), checkpoint_epoch=ck_epoch,
        n_bn_modules=len(bn_modules),
        n_collected_params=len(names), collected_params=names,
        skipped_norm_modules=skipped,
        skip_substrings=list(sar.SKIP_SUBSTRINGS),
        norm_mode=sar.BN_EVAL_SOURCE_STATS,
        split_meta=split_meta)

    margin = sar.e0_margin(SEO.NUM_CLASSES)
    reset_const = sar.reset_constant(SEO.NUM_CLASSES)
    lr = args.lr * args.lr_multiplier

    def optimizer_factory():
        return sar.make_optimizer(params, lr=lr, momentum=0.9, rho=sar.SAM_RHO)

    out.update(sar_steps=args.steps, sar_lr=lr, sar_lr_base=args.lr,
               sar_lr_multiplier=args.lr_multiplier,
               e0_margin=margin, reset_constant=reset_const,
               n_images=len(images))
    return model, pristine, images, margin, reset_const, optimizer_factory, set(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="dinov3",
                    choices=["dinov3", "mamnet", "oglanet", "crossearth"])
    ap.add_argument("--city", default="chicago")
    ap.add_argument("--res", default="hr", choices=["hr", "mr"])
    ap.add_argument("--ckpt", default=None, help="crossearth branch only")
    ap.add_argument("--fold", default=None,
                    help="crossearth branch only: OMA | JAX | UCSD")
    ap.add_argument("--ce_repo_root",
                    default=os.path.join(BASE, "external", "CrossEarth"),
                    help="crossearth branch only")
    ap.add_argument("--ce_backbone",
                    default=os.path.join(BASE, "external", "CrossEarth",
                                         "checkpoints", "dinov2_converted.pth"),
                    help="crossearth branch only")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.00025)
    ap.add_argument("--lr_multiplier", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all test images")
    ap.add_argument("--out_root",
                    default=os.path.join(
                        os.environ.get("ST_SCRATCH_ROOT", ""),
                        "runs/train_count_audit/sar_drift"))
    ap.add_argument("--cell_seed", type=int, default=0,
                    help="S-EO branches: the vanilla cell seed SAR adapts")
    ap.add_argument("--recipe_from", default=None,
                    help="the SAR cell's OWN run.json; its recipe block "
                         "overrides steps/lr/lr_multiplier so the probe "
                         "never runs on its own defaults")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    from shadowtransfer import sar
    import torch

    recipe_source = None
    if args.recipe_from:
        rj = json.load(open(args.recipe_from))
        r = rj.get("recipe")
        if not r:
            raise SystemExit(f"FATAL --recipe_from {args.recipe_from} has no "
                             f"recipe block; refusing to fall back to defaults")
        args.steps = int(r["steps"])
        args.lr = float(r["lr"])
        args.lr_multiplier = float(r.get("lr_multiplier", 1.0))
        recipe_source = {"path": args.recipe_from,
                         "run_id": rj.get("run_id"),
                         "recipe_verbatim": r,
                         "sar_constants_verbatim": rj.get("sar_constants")}

    if args.arch == "crossearth":
        label = f"crossearth-sar-{(args.fold or 'nofold').lower()}"
    elif args.fold:
        label = f"{args.arch}-sar-seo-{args.fold.lower()}"
    else:
        label = f"{args.arch}-sar-{args.city}-{args.res}"
    out = {"id": "sar_weight_drift", "label": label + args.tag,
           "question": ("Does the test-time method change any weights, and what "
                        "is the largest change?"),
           "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "node": platform.node(),
           "sar_module": os.path.join("python", "shadowtransfer", "sar.py"),
           "recipe_source": recipe_source}

    print("=" * 74)
    print(f"SAR WEIGHT-DRIFT PROBE (check only)   {label}")
    print(f"  steps {args.steps}  lr {args.lr} x {args.lr_multiplier}  limit {args.limit}")
    if recipe_source:
        print(f"  recipe read from : {recipe_source['path']}")
        print(f"    run_id         : {recipe_source['run_id']}")
        print(f"    recipe         : {json.dumps(recipe_source['recipe_verbatim'])}")
        print(f"    sar_constants  : {json.dumps(recipe_source['sar_constants_verbatim'])}")
    else:
        print("  recipe           : PROBE DEFAULTS (no --recipe_from given)")
    print("=" * 74)
    sys.stdout.flush()
    if args.dry_run:
        print("dry run: parsed arguments and stopped before loading anything")
        return

    t0 = time.time()
    if args.arch == "crossearth":
        model, pristine, images, margin, rc, ofac, collected = run_crossearth(args, out)
    elif args.fold:
        model, pristine, images, margin, rc, ofac, collected = run_seo(args, out)
    else:
        model, pristine, images, margin, rc, ofac, collected = run_dinov3(args, out)

    print(f"collected {out['n_collected_params']} params; "
          f"skipped {len(out['skipped_norm_modules'])} norm modules; "
          f"{out['n_images']} images", flush=True)
    if out["n_collected_params"] < 4:
        raise SystemExit(f"FATAL collected {out['n_collected_params']} parameters")

    best = {"max_abs_delta": 0.0}
    moved_names, outside, per_image = set(), set(), []
    n_any = 0
    for i, im in enumerate(images):
        sar.restore_state(model, pristine)
        ok, bad = sar.state_matches(model, pristine)
        if not ok:
            raise SystemExit(f"FATAL restore not byte-identical at image {i}: {bad[:5]}")
        x = im["x"].to(model_device(model), non_blocking=True)
        sar.adapt_image(model, x, args.steps, margin, rc, ofac, pristine)
        rows = diff_state(model, pristine, CTX["state_fn"])
        if rows:
            n_any += 1
        im_max = 0.0
        for r in rows:
            nm = r.get("param_name", r["name"])
            moved_names.add(nm)
            if nm not in collected:
                outside.add(nm)
            m = r.get("max_abs_delta", 0.0)
            im_max = max(im_max, m)
            if m > best["max_abs_delta"]:
                best = dict(r, image_index=i,
                            image_id=im.get("image_id", im.get("stem", str(i))))
        per_image.append({"image_index": i, "max_abs_delta": im_max,
                          "n_tensors_moved": len(rows)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(images)}  running max |delta| = "
                  f"{best['max_abs_delta']:.3e}", flush=True)

    import hashlib
    img_ids = [im.get("image_id", im.get("stem", str(i)))
               for i, im in enumerate(images)]
    out["image_ids_sha256"] = hashlib.sha256(
        "\n".join(img_ids).encode()).hexdigest()
    out["image_ids_first10"] = img_ids[:10]
    out["image_ids"] = img_ids
    maxes = sorted((p["max_abs_delta"] for p in per_image), reverse=True)
    out.update(
        state_reader=(CTX["state_note"] or "model.state_dict()"),
        n_images_with_any_change=n_any,
        n_params_moved=len(moved_names),
        params_moved=sorted(moved_names)[:200],
        moved_outside_collected=sorted(outside),
        largest_change=best,
        max_abs_delta=best["max_abs_delta"],
        max_rel_delta=best.get("rel_delta_at_argmax"),
        per_image_max_abs_delta_top10=maxes[:10],
        per_image_max_abs_delta_median=(maxes[len(maxes) // 2] if maxes else None),
        wall_clock_s=round(time.time() - t0, 1),
        peak_vram_gb=(round(torch.cuda.max_memory_allocated() / 2**30, 3)
                      if torch.cuda.is_available() else None))

    os.makedirs(args.out_root, exist_ok=True)
    p = os.path.join(args.out_root, out["label"] + ".json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, p)

    print("-" * 74)
    print(f"images with ANY weight change : {n_any} / {out['n_images']}")
    print(f"parameters that ever moved    : {len(moved_names)} "
          f"of {out['n_collected_params']} collected")
    print(f"moved OUTSIDE collected set   : {sorted(outside) or 'none'}")
    print(f"largest |adapted - start|     : {best['max_abs_delta']:.6e}"
          f"  on {best.get('name')}  (image {best.get('image_id')})")
    print(f"  as a fraction of that start value: {best.get('rel_delta_at_argmax')}")
    print(f"wrote {p}")


def model_device(model):
    for p in model.parameters():
        return p.device
    return "cpu"


if __name__ == "__main__":
    main()
