"""E-12 REMEDIATION — retrain the 3 contaminated MAMNet SIB C4clean HR cells.

THE CONTAMINATION, as confirmed in summaries/session_logs/2026-08-03-e12-and-riders.md
------------------------------------------------------------------------------------
`train_mamnet_sib.py` builds its FDA target pool by handing `fda_target_root` to
`data/fda_transform.load_target_images`, which `os.walk`s the root, keeps every
.png/.jpg/.jpeg/.tif/.tiff it finds, and draws 100 with `np.random.choice`. The
three published C4clean HR cells were given the CITY ROOT
(`Final_data_test/{city}/highres`), whose walk yields 2250 candidates:
1350 train + 450 val + 450 test (images and both mask subtrees). 20% of the
candidate list — and, in expectation, 20 of the 100 drawn files — was held-out
test imagery.

THE ONE CHANGE
--------------
`fda_target_root` points at `Final_data_test/{city}/highres/train/images`, the
same 450-image directory the 18 CLEAN standalone-FDA cells pin. Draw rule
unchanged: same `load_target_images`, same `max_images=100`, same sampler. Every
other hyperparameter, component flag, split and schedule is lifted verbatim from
the published cell's own `config.json`.

Scope, per the HLT order: MAMNet SIB C4clean highres only, one cell per
held-out city. The 18 standalone FDA cells are clean and are not touched.
OGLANet-SIB never reads `fda_target_root` at all (the value is inert) and
DINOv3-SIB has `use_fda: false`; neither is touched.

WHAT COULD NOT BE PINNED FROM THE PUBLISHED CONFIG
--------------------------------------------------
`config.json` is `json.dump(vars(args))` written before the dataloaders are
built, so it records every argument the run's script version declared. The
published files carry 52 keys; the current script declares 57. The five absent
are `seed`, `full_dataset`, `expected_train_total`, `stats_json` and
`skip_final_test` — all later additions.

`seed` is the one that matters. The published cells were produced by a script
version with NO `--seed` argument and no seeding block, so THE PUBLISHED SEED IS
UNRECOVERABLE and those runs were not explicitly seeded at all. This run uses
the script's current default, 42, and records `seed_provenance` as unrecoverable.
The consequence is stated plainly in run.json and must be carried into any
reading of the old-vs-new delta: the comparison is NOT a clean single-variable
contrast. It differs in the FDA pool AND in RNG stream. A small move is
therefore not attributable to the pool alone, and neither is a large one.

The other four are inert here: `full_dataset` False reproduces the published
subsampling (loco sets `samples_per_dir = 450 // 2 = 225`, giving the published
450/300/150 split), `stats_json` and `skip_final_test` touch only post-training
bookkeeping, and `expected_train_total` is used here as a hard count gate.

MODES
-----
    --task retrain   train with the clean pool, then dump (CONTRACT §3)
    --task repro     no training: load the PUBLISHED best_model.pth, run test
                     inference, reproduce that cell's own stored
                     test_results.json, and dump it too so the comparison is
                     per-image rather than headline-only

The published run directories are READ-ONLY here. Nothing is written into them,
nothing is moved, nothing is annotated: the contaminated originals are evidence
and both sets go in the appendix per the E-12 disclosure plan. Every artefact
this script writes lands under /scratch.
"""

import argparse
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

from shadowtransfer import armc_dump                            # noqa: E402
from shadowtransfer import armc_runtime                         # noqa: E402

RES = "highres"
RUNG = 450
ARCH = "mamnet"
METHOD = "sibc4"
DATASET = "shadowtransfer_internal"
PUB_TMPL = ("data/mamnet/outputs/mamnet_sib_C4clean_haar_vib_sag_fda_ctr__"
            "loco_holdout_{city}__highres")
# fold_id -> holdout, from train_mamnet_sib.CITY_FOLDS. Re-asserted at runtime
# against the published config's own fold_id rather than trusted from here.
FOLD_OF = {"phoenix": 0, "miami": 1, "chicago": 2}
VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
DEFAULT_SEED = 42
N_TRAIN_EXPECT = 450
# A regenerated cell must reproduce its own stored test_results.json within
# this many mIoU (CLAUDE.md, "Reusing the published cells"). All three E-12
# repro jobs came in at exactly +0.000000 on 2026-08-03.
REPRO_TOL_MIOU = 0.01
N_VAL_EXPECT = 300
N_TEST_EXPECT = 150

# The five arguments the published config predates, with the current script's
# defaults. Recorded in run.json as `config_defaults_filled`.
LATER_ARGS = {
    "seed": DEFAULT_SEED,
    "full_dataset": False,
    "expected_train_total": None,
    "stats_json": "train_stats.json",
    "skip_final_test": False,
}


def published_dir(city):
    return BASE / PUB_TMPL.format(city=city)


def clean_pool_root(city):
    return BASE / "data" / "Final_data_test" / city / RES / "train" / "images"


def test_images_dir(city):
    return BASE / "data" / "Final_data_test" / city / RES / "test" / "images"


def enumerate_pool(root):
    """The candidate list `load_target_images` would build, in its own order.

    Reproduces `data/fda_transform.py:196-203` exactly: os.walk, the same
    extension filter, no directory excluded. The 100-file draw that follows is
    `np.random.choice` over THIS list, so a list disjoint from the test set
    makes every possible draw disjoint from it. That is the assertion below,
    and it is strictly stronger than checking the drawn 100.
    """
    out = []
    root = str(root)
    if not os.path.isdir(root):
        raise SystemExit(f"FATAL pool root is not a directory: {root}")
    for r, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                out.append(os.path.join(r, f))
    return sorted(out)


def pool_assertion(city):
    """Enumerate the clean pool and prove it is disjoint from the test set.

    Filename-level, per the task spec: the pool and the test set live in
    different directories, so a path comparison would pass trivially. Basenames
    are what identify an image across splits.
    """
    root = clean_pool_root(city)
    tdir = test_images_dir(city)
    pool = enumerate_pool(root)
    pool_names = sorted(os.path.basename(p) for p in pool)
    if not tdir.is_dir():
        raise SystemExit(f"FATAL published test dir missing: {tdir}")
    test_names = sorted(f for f in os.listdir(tdir) if f.lower().endswith(VALID_EXT))
    overlap = sorted(set(pool_names) & set(test_names))
    # The contaminated root, enumerated for the record so run.json carries the
    # before/after rather than only the after.
    contaminated_root = BASE / "data" / "Final_data_test" / city / RES
    contaminated = enumerate_pool(contaminated_root)
    contaminated_names = [os.path.basename(p) for p in contaminated]
    contaminated_overlap = sorted(set(contaminated_names) & set(test_names))
    rec = {
        "fda_target_pool_split": "train",
        "pool_root": str(root),
        "n_pool": len(pool),
        "n_pool_expected": 450,
        "pool_enumeration_rule": ("os.walk + extension filter "
                                  f"{VALID_EXT}, verbatim from "
                                  "data/fda_transform.py:196-203"),
        "draw_rule": ("np.random.choice(candidates, 100, replace=False) — "
                      "unchanged from the published cells"),
        "max_images": 100,
        "test_set_dir": str(tdir),
        "n_test_images": len(test_names),
        "overlap_filenames": overlap,
        "n_overlap": len(overlap),
        "disjoint": len(overlap) == 0,
        "assertion": ("the enumerated pool is disjoint from the published test "
                      "set at filename level; every possible 100-file draw is "
                      "therefore also disjoint"),
        "contaminated_root_for_reference": {
            "root": str(contaminated_root),
            "n_pool": len(contaminated),
            "n_overlap_with_test": len(contaminated_overlap),
            "pct_of_pool_that_is_test_subtree": (
                round(100.0 * sum(1 for p in contaminated
                                  if f"{os.sep}test{os.sep}" in p)
                      / max(len(contaminated), 1), 4)),
            "note": ("this is what the PUBLISHED cells used; enumerated here "
                     "for the record only. Nothing reads it."),
        },
    }
    return rec, pool


def load_published_config(city):
    p = published_dir(city) / "config.json"
    if not p.is_file():
        raise SystemExit(f"FATAL published config missing: {p}")
    return json.loads(p.read_text()), p


def build_args(city, task, out_dir, seed):
    """The published args, verbatim, plus the five later additions.

    The ONLY substantive override is fda_target_root. output_dir is redirected
    to /scratch so nothing is written into the published run directory, and
    device is set for this box.
    """
    from argparse import Namespace
    cfg, cfg_path = load_published_config(city)
    if cfg.get("fold_id") != FOLD_OF[city]:
        raise SystemExit(f"FATAL {city}: published fold_id {cfg.get('fold_id')} "
                         f"!= expected {FOLD_OF[city]}")
    for k, v in (("mode", "loco"), ("resolution", RES), ("use_fda", True),
                 ("eval_boundary_tolerant", True), ("boundary_tolerance", 2)):
        if cfg.get(k) != v:
            raise SystemExit(f"FATAL {city}: published {k}={cfg.get(k)!r} != {v!r}")

    a = dict(cfg)
    filled = {k: v for k, v in LATER_ARGS.items() if k not in a}
    a.update(filled)
    a["seed"] = seed
    a["output_dir"] = str(out_dir)
    a["device"] = "cuda"
    a["skip_final_test"] = True          # our own dump phase replaces it
    a["expected_train_total"] = N_TRAIN_EXPECT
    old_root = a["fda_target_root"]
    a["fda_target_root"] = str(clean_pool_root(city))
    prov = {
        "published_config": str(cfg_path.relative_to(BASE)),
        "published_config_sha256": armc_dump.sha256_file(cfg_path),
        "n_keys_published": len(cfg),
        "config_defaults_filled": filled,
        "config_defaults_filled_note": (
            "absent from the published config because that script version did "
            "not declare them; filled with the CURRENT script's defaults"),
        "overrides": {
            "fda_target_root": {"was": old_root, "now": a["fda_target_root"],
                                "this_is_the_one_change": True},
            "output_dir": {"was": cfg["output_dir"], "now": str(out_dir),
                           "why": "published run dirs are read-only evidence"},
            "skip_final_test": {"now": True,
                                "why": ("the published post-training block "
                                        "writes PNGs and matplotlib figures; "
                                        "CLAUDE.md forbids plotting in a "
                                        "worker and a matplotlib call hung a "
                                        "cell for 90 minutes on 2026-07-27. "
                                        "The dump phase in this script "
                                        "replaces it.")},
            "expected_train_total": {"now": N_TRAIN_EXPECT,
                                     "why": "hard count gate on the loader"},
        },
        "seed_used": seed,
        "seed_provenance": (
            "UNRECOVERABLE for the published cells. config.json is "
            "json.dump(vars(args)) and carries no `seed` key, so the script "
            "version that produced them had no --seed argument and no seeding "
            "block; those runs were not explicitly seeded. This run uses the "
            "current script's default, 42."),
        "seed_caveat": (
            "old-vs-new therefore differs in the FDA pool AND in RNG stream. "
            "The delta is NOT attributable to the pool alone."),
    }
    return Namespace(**a), prov, cfg


def silence_sib_side_effects(mod):
    """No-op the published module's figure and PNG writers.

    --skip_final_test already returns before every one of these, so this is
    belt-and-braces against a future code path reaching them. Recorded either
    way. Nothing is edited on disk; the rebinds live in this process only.
    """
    disabled = []
    for name in ("plot_all_losses", "save_best_worst_predictions",
                 "test_and_save_predictions", "compare_with_baselines"):
        if hasattr(mod, name):
            setattr(mod, name, armc_runtime._noop)
            disabled.append(name)
    return disabled


def build_test_loader(mod, args, city):
    """The held-out city's test split, exactly as get_dataloaders_sib builds it.

    Built directly rather than through get_dataloaders_sib so a repro job never
    touches an FDA pool it does not need. The test dataset never applies FDA in
    either path (`use_fda and split == 'train'`), so this is the same object.
    """
    import torch
    from data.dataset_sib import ShadowDatasetSIB
    test_path = os.path.join(args.base_data_root, city, args.resolution)
    ds = ShadowDatasetSIB(
        root_dir=[test_path], split="test", img_size=args.img_size,
        augment=False, geo_metadata_path=None,
        use_contrast=args.use_contrast)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True)
    return ds, loader


def infer(mod, model, loader, device):
    """Test inference under the PUBLISHED prediction recipe.

    `filter_small_predictions(logits, min_pixels=10)` then argmax — lifted from
    `test_and_save_predictions` (train_mamnet_sib.py:715-717). Returns the
    per-image pieces both metric conventions need.
    """
    import torch
    # NOT a fresh import: `mod.filter_small_predictions` is the very object the
    # published `test_and_save_predictions` calls (train_mamnet_sib.py:72 binds
    # it from utils.postprocessing, NOT utils.metrics -- job 20826399's smoke
    # died on exactly that wrong path). Going through the module guarantees the
    # prediction recipe cannot drift from the published one.
    filter_small_predictions = mod.filter_small_predictions
    model.eval()
    recs = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            intensity = batch["intensity_map"].to(device)
            fname = batch["filename"][0]
            outputs = model(images, intensity_map=intensity)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = (outputs if isinstance(outputs, torch.Tensor)
                      else outputs["main"])
            filtered = filter_small_predictions(logits, min_pixels=10)
            pred = filtered.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
            gt = batch["mask"][0].numpy().astype(np.uint8)
            recs.append({"filename": fname, "gt": gt, "prob": prob, "pred": pred})
    return recs


def published_convention(mod, recs, tol):
    """Reproduce the published cells' own test_results.json convention.

    Their numbers are the per-image mean of `_compute_strict_metrics` and
    `_compute_tolerant_metrics`, via `_average_metrics`. Those three functions
    are imported from the published module and called unchanged, so this is
    their convention and not a re-derivation of it.
    """
    sl = [mod._compute_strict_metrics(r["pred"], r["gt"]) for r in recs]
    tl = [mod._compute_tolerant_metrics(r["pred"], r["gt"], tolerance=tol)
          for r in recs]
    return {"num_images": len(recs),
            "strict": mod._average_metrics(sl),
            f"tolerant_{tol}px": mod._average_metrics(tl)}


def contract_rows(run_id, city, recs):
    """CONTRACT §3 rows via shadowtransfer.armc_dump — the only metric path."""
    geo = armc_dump.load_geo_meta(BASE)
    rows, probs = [], {}
    for r in recs:
        stem = os.path.splitext(r["filename"])[0]
        rows.append(armc_dump.perimage_row(
            run_id, stem, city, "hr", r["gt"], r["prob"], r["pred"],
            geo.get(r["filename"])))
        probs[stem] = r["prob"].astype(np.float16)
    return rows, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=sorted(FOLD_OF))
    ap.add_argument("--task", required=True, choices=["retrain", "repro"])
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--timeout_s", type=int, default=3600)
    ap.add_argument("--num_workers", type=int, default=None,
                    help="default: the published config's own value")
    ap.add_argument("--smoke_epochs", type=int, default=0)
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    city = args.city
    smoke = bool(args.smoke_epochs or args.smoke_test)
    dump_dir = Path(args.out_root) / "e12" / args.run_id
    run_out = Path(args.ckpt_root) / args.run_id

    if (dump_dir / ".DONE").exists() and not args.force:
        print(f"[{args.run_id}] .DONE exists at {dump_dir} - nothing to do")
        return

    # ---- the pool assertion, BEFORE anything expensive ---------------------
    pool_rec, _pool = pool_assertion(city)
    print("=" * 78)
    print(f"[{args.run_id}] FDA TARGET POOL ASSERTION")
    print(f"  pool root         {pool_rec['pool_root']}")
    print(f"  n_pool            {pool_rec['n_pool']} (expected {pool_rec['n_pool_expected']})")
    print(f"  test set          {pool_rec['test_set_dir']} ({pool_rec['n_test_images']} files)")
    print(f"  n_overlap         {pool_rec['n_overlap']}")
    print(f"  DISJOINT          {pool_rec['disjoint']}")
    cr = pool_rec["contaminated_root_for_reference"]
    print(f"  (published root   {cr['root']}: n_pool {cr['n_pool']}, "
          f"{cr['pct_of_pool_that_is_test_subtree']}% of it the test subtree)")
    print("=" * 78, flush=True)
    if not pool_rec["disjoint"]:
        armc_dump.mark_failed(dump_dir, f"FDA pool overlaps the test set on "
                                        f"{pool_rec['n_overlap']} filenames")
        raise SystemExit(f"FATAL {city}: clean pool overlaps the published test "
                         f"set on {pool_rec['n_overlap']} filenames: "
                         f"{pool_rec['overlap_filenames'][:10]}")
    if pool_rec["n_pool"] != pool_rec["n_pool_expected"] and not smoke:
        raise SystemExit(f"FATAL {city}: n_pool {pool_rec['n_pool']} != "
                         f"{pool_rec['n_pool_expected']}")

    trainer_args, prov, cfg = build_args(city, args.task, run_out, args.seed)
    if args.num_workers is not None:
        trainer_args.num_workers = args.num_workers
        prov["overrides"]["num_workers"] = {"was": cfg["num_workers"],
                                            "now": args.num_workers}
    if args.task == "repro":
        # The published cell is loaded and evaluated, not retrained. Its own
        # contaminated root is left in place on the Namespace so nothing here
        # misrepresents which cell this is; no pool is ever loaded, because the
        # test dataset never applies FDA.
        trainer_args.fda_target_root = cfg["fda_target_root"]
        prov["overrides"]["fda_target_root"] = {
            "applied": False,
            "value": cfg["fda_target_root"],
            "why": ("--task repro evaluates the PUBLISHED checkpoint; the "
                    "clean-pool override is not applied and no FDA pool is "
                    "loaded (test split never applies FDA)"),
        }
    if args.smoke_epochs:
        trainer_args.epochs = args.smoke_epochs
        trainer_args.expected_train_total = None

    print(f"[{args.run_id}] task={args.task} city={city} "
          f"fold_id={trainer_args.fold_id} seed={trainer_args.seed} "
          f"epochs={trainer_args.epochs} patience="
          f"{trainer_args.early_stopping_patience} "
          f"num_workers={trainer_args.num_workers}", flush=True)
    print(f"[{args.run_id}] fda_target_root -> {trainer_args.fda_target_root}",
          flush=True)
    if args.dry_run:
        print(json.dumps({"pool": pool_rec, "provenance": prov}, indent=2))
        return

    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet", dump_dir / "probs.npz",
        dump_dir / "run.json", dump_dir / ".DONE"], label="dump")

    import torch                                                # noqa: E402
    arch_root = BASE / "mamnet"
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    for _m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")
               or k == "data" or k.startswith("data.")]:
        del sys.modules[_m]
    import train_mamnet_sib as mod                              # noqa: E402

    disabled = silence_sib_side_effects(mod)
    print(f"[e12] side effects disabled on train_mamnet_sib: {disabled}",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_stats, train_wall = None, 0.0
    run_out.mkdir(parents=True, exist_ok=True)

    if args.task == "retrain":
        t_tr = time.time()
        mod.run_training(trainer_args)
        train_wall = time.time() - t_tr
        sp = run_out / trainer_args.stats_json
        if not sp.is_file():
            raise SystemExit(f"FATAL no {sp} after training")
        train_stats = json.loads(sp.read_text())
        ckpt_path = run_out / "best_model.pth"
    else:
        ckpt_path = published_dir(city) / "best_model.pth"
    if not ckpt_path.is_file():
        raise SystemExit(f"FATAL checkpoint missing: {ckpt_path}")

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("build_model")
        model = mod.build_mamnet_sib(trainer_args).to(device)
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        best_epoch = int(ck.get("epoch", -1))
        best_val = float(ck.get("best_metric", float("nan")))
        print(f"[{args.run_id}] loaded {ckpt_path} (epoch {best_epoch}, "
              f"best_metric {best_val:.4f})", flush=True)

        tbx.phase("inference")
        ds, loader = build_test_loader(mod, trainer_args, city)
        if args.smoke_test:
            ds.img_paths = ds.img_paths[:args.smoke_test]
            ds.img_files = ds.img_files[:args.smoke_test]
            ds.mask_paths = ds.mask_paths[:args.smoke_test]
            ds, loader = ds, torch.utils.data.DataLoader(
                ds, batch_size=1, shuffle=False, num_workers=0)
        t_inf = time.time()
        recs = infer(mod, model, loader, device)
        inf_wall = time.time() - t_inf
        if not smoke and len(recs) != N_TEST_EXPECT:
            raise SystemExit(f"FATAL n_test {len(recs)} != {N_TEST_EXPECT}")

        tbx.phase("metrics")
        tol = int(trainer_args.boundary_tolerance)
        pub_conv = published_convention(mod, recs, tol)
        rows, probs = contract_rows(args.run_id, city, recs)
        import pandas as pd                                     # noqa: E402
        df = pd.DataFrame(rows)
        fg_iou_tol = float(df["iou_tol"].mean()) * 100.0
        two_class_k2 = float(df["miou_tol_pct"].mean())
        pred_pos_frac = float(df["n_pred_pos"].sum()
                              / max(int(df["n_valid_px"].sum()), 1))
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)

        # ---- the published cell's own stored numbers, read not recomputed ---
        stored = json.loads((published_dir(city) / "test_results.json").read_text())
        tol_key = f"tolerant_{tol}px"
        repro_delta = None
        if args.task == "repro":
            repro_delta = (pub_conv[tol_key]["mIOU"] - stored[tol_key]["mIOU"])

        epochs_max = int(trainer_args.epochs)
        epochs_run = int((train_stats or {}).get("epochs_run", 0))
        n_train_total = int((train_stats or {}).get("n_train_total", 0))
        steps_per_epoch = n_train_total // int(trainer_args.batch_size)
        stopped_on = None
        if args.task == "retrain":
            stopped_on = ("patience" if epochs_run < epochs_max
                          else "epoch_ceiling")

        flags = []
        if not (0.001 < pred_pos_frac < 0.9):
            flags.append(f"predicted-positive fraction {pred_pos_frac:.5f} "
                         f"outside (0.001, 0.9)")
        if not smoke and len(df) != N_TEST_EXPECT:
            flags.append(f"row count {len(df)} != published n_test {N_TEST_EXPECT}")
        if float(df["iou_tol"].std()) < 1e-6:
            flags.append("zero variance in per-image tolerant IoU")
        delta_pub = pub_conv[tol_key]["mIOU"] - stored[tol_key]["mIOU"]
        if args.task == "retrain" and abs(delta_pub) > 2.0:
            flags.append(f"MOVED {delta_pub:+.4f} mIoU from the published value "
                         f"under the published convention — >2 mIoU. Reported "
                         f"as-is; not smoothed and not re-run.")
        if repro_delta is not None and abs(repro_delta) > 0.01:
            flags.append(f"repro of the published cell's own test_results.json "
                         f"is off by {repro_delta:+.4f} mIoU (>0.01)")

        run = {
            "run_id": args.run_id, "dataset": DATASET, "arm": "E12",
            "e12_remediation": True,
            "e12_task": args.task,
            "e12_note": (
                "E-12: the published mamnet_sib_C4clean HR cells built their "
                "FDA target pool from the city ROOT, so ~20% of the 100-file "
                "draw was held-out test imagery. This wave retrains the three "
                "affected cells with the pool pinned to the held-out city's "
                "train/images. The 18 standalone FDA cells are clean and "
                "untouched; OGLANet-SIB never reads fda_target_root and "
                "DINOv3-SIB has use_fda false — neither is touched."),
            "arch": ARCH, "method": METHOD, "mode": "loco", "rung": RUNG,
            "res": "hr", "holdout": city, "fold_id": int(trainer_args.fold_id),
            "seed": int(trainer_args.seed),
            "n_test": len(df), "n_test_expected": N_TEST_EXPECT,
            "n_train": n_train_total or None,
            "n_val": int((train_stats or {}).get("n_val_total", 0)) or None,
            "n_train_per_city": (train_stats or {}).get("n_train_per_city"),

            "fda_target_pool_split": pool_rec["fda_target_pool_split"],
            "fda_target_pool": pool_rec,
            "fda_L": trainer_args.fda_L,

            "provenance": prov,
            "published_cell_dir": str(published_dir(city).relative_to(BASE)),
            "published_cell_stored_test_results": stored,
            "published_cell_artifacts_untouched": True,

            "reporting_metric": "fg_only_tol_k2",
            "reporting_metric_value_pct": round(fg_iou_tol, 4),
            "two_class_tol_k2_pct": round(two_class_k2, 4),
            "band_width_headline": 2, "band_widths_written": [0, 2, 5],
            "published_convention_metrics": pub_conv,
            "published_convention_note": (
                "per-image mean of train_mamnet_sib._compute_strict_metrics "
                "and ._compute_tolerant_metrics via ._average_metrics — the "
                "published cells' own test_results.json convention, computed "
                "by importing those three functions unchanged"),
            "delta_vs_published_convention_miou": round(delta_pub, 4),
            "repro_delta_miou": (None if repro_delta is None
                                 else round(repro_delta, 6)),
            "prediction_recipe": ("argmax over filter_small_predictions("
                                  "logits, min_pixels=10) — lifted from "
                                  "train_mamnet_sib.test_and_save_predictions"),
            "decision_metric": f"2class_tol_k{tol}_pooled_val",
            "decision_metric_note": (
                "what best_model.pth was SELECTED on: the published loop's own "
                "validate() tolerant mIOU at K=2 on the source-city val split. "
                "Not the reporting metric."),
            "decision_metric_value": best_val,
            "best_epoch": best_epoch,
            "epochs_max": epochs_max, "epochs_run": epochs_run or None,
            "steps_per_epoch": steps_per_epoch or None,
            "steps_to_best": (best_epoch * steps_per_epoch
                              if steps_per_epoch and best_epoch > 0 else None),
            "stopped_on": stopped_on,
            "early_stop_patience": int(trainer_args.early_stopping_patience),
            "ckpt_path": str(ckpt_path),
            "parent_run_id": None,
            "target_access": "unlabeled_target",
            "target_access_detail": (
                "FDA input preprocessing draws its amplitude spectra from "
                "UNLABELLED images of the held-out city's TRAIN split. The "
                "published cells drew from the city root and so also touched "
                "the held-out city's VAL and TEST images — that is E-12."),

            "metric_module_sha": armc_dump.sha256_file(
                BASE / "shadowtransfer/metrics.py"),
            "dump_module_sha": armc_dump.sha256_file(
                BASE / "shadowtransfer/armc_dump.py"),
            "train_script_sha": armc_dump.sha256_file(
                BASE / "mamnet/train_mamnet_sib.py"),
            "e12_script_sha": armc_dump.sha256_file(Path(__file__).resolve()),
            "worker_side_effects_disabled": disabled,
            "side_effect_note": (
                "skip_final_test=True returns before plot_all_losses, "
                "test_and_save_predictions, compare_with_baselines and "
                "save_best_worst_predictions; those names are additionally "
                "rebound to no-ops in this process. The published file is not "
                "edited."),
            "seed_sites": {
                "python_random": f"random.seed({trainer_args.seed})",
                "numpy": f"np.random.seed({trainer_args.seed})",
                "torch_cpu": f"torch.manual_seed({trainer_args.seed})",
                "torch_cuda": f"torch.cuda.manual_seed_all({trainer_args.seed})",
                "where": "train_mamnet_sib.run_training:1687-1694, before the "
                         "dataloaders (and so before the FDA 100-file draw)",
            },
            "cluster": "delta",
            "compute_note": os.environ.get("ST_COMPUTE_NOTE"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_time_limit": os.environ.get("ARMA_TIME_LIMIT"),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "peak_vram_alloc_gb": round(peak_alloc, 3),
            "peak_vram_reserved_gb": round(peak_reserved, 3),
            "train_wall_clock_s": round(train_wall, 2),
            "inference_wall_clock_s": round(inf_wall, 2),
            "wall_clock_s": round(time.time() - t0, 2),
            "git_hash": os.environ.get("ARMA_GIT_HASH"),
            "git_tree_dirty_paths": os.environ.get("ARMA_GIT_DIRTY"),
            "pred_pos_frac_on_valid": round(pred_pos_frac, 6),
            "smoke": smoke,
            "sanity_flags": flags,
            "written_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
        }

        tbx.phase("write_dump")
        armc_dump.write_dump(dump_dir, rows, probs, run)

    print("\n" + "=" * 78)
    print(f"[{args.run_id}]  task {args.task}  city {city}")
    print(f"  FDA pool                     {pool_rec['pool_root']}")
    print(f"  n_pool / overlap / disjoint  {pool_rec['n_pool']} / "
          f"{pool_rec['n_overlap']} / {pool_rec['disjoint']}")
    print(f"  best_epoch                   {best_epoch}")
    print(f"  epochs_run / stopped_on      {epochs_run or '-'} / {stopped_on or '-'}")
    print(f"  steps_to_best                {run['steps_to_best']}")
    print(f"  val decision metric          {best_val:.4f}")
    print(f"  --- NEW (this run) ---")
    print(f"  fg-only IoU K=2              {fg_iou_tol:.4f}")
    print(f"  two-class K=2 (CONTRACT)     {two_class_k2:.4f}")
    print(f"  published-convention mIOU    {pub_conv[tol_key]['mIOU']:.4f} "
          f"(strict {pub_conv['strict']['mIOU']:.4f})")
    print(f"  --- PUBLISHED (stored, contaminated pool) ---")
    print(f"  published-convention mIOU    {stored[tol_key]['mIOU']:.4f} "
          f"(strict {stored['strict']['mIOU']:.4f})")
    print(f"  DELTA (new - published)      {delta_pub:+.4f} mIoU")
    if repro_delta is not None:
        print(f"  repro_delta vs stored        {repro_delta:+.6f}")
    print(f"  rows / expected              {len(df)} / {N_TEST_EXPECT}")
    print(f"  predicted-positive fraction  {pred_pos_frac:.5f}")
    print(f"  wall clock                   {time.time() - t0:.1f}s "
          f"(train {train_wall:.1f}s, inference {inf_wall:.1f}s)")
    print(f"  sanity flags                 {flags or 'none'}")
    print(f"  -> {dump_dir}")
    print("=" * 78, flush=True)

    # ---- the repro GATE actually gates -------------------------------------
    # A --task repro job exists to prove the inference + metric + dump path
    # against a known-good number BEFORE any retrain builds on it. CLAUDE.md on
    # regenerated cells: a cell that does not reproduce its own stored
    # test_results.json is "a finding: report it, do not adjust it, and do not
    # build on it until the operator has seen it." Flagging it in run.json is
    # not enough -- the retrains hang off --dependency=afterok, so a job that
    # merely flags and exits 0 lets them start anyway. The dump is written
    # first, so the evidence survives; only the exit status fails.
    if (args.task == "repro" and not smoke and repro_delta is not None
            and abs(repro_delta) > REPRO_TOL_MIOU):
        print("!" * 78, flush=True)
        print(f"[{args.run_id}] REPRO GATE FAILED: reproduced "
              f"{pub_conv[tol_key]['mIOU']:.6f} against the cell's own stored "
              f"{stored[tol_key]['mIOU']:.6f}, delta {repro_delta:+.6f} > "
              f"{REPRO_TOL_MIOU}.", flush=True)
        print("  The dump IS written and complete -- it is the comparison that "
              "failed, not the artifact. Nothing downstream may build on this "
              "until the operator has seen it.", flush=True)
        print("!" * 78, flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
