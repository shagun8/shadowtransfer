"""
Arm C — the §4.2-4.4 diagnostic probes on the ladder models.

Answers AC1 and zNBu-W3 through the pre-commitment in REVIEWS.md, arm C:
"Run the §4.2-4.4 diagnostics on the 900 models. If the SP gap is still
foreground-concentrated and the locus is unchanged at 900, the framework is
not a small-data artifact."

Four probes, all off the completed 135-cell ladder, all vanilla LOCO seed 0
high-resolution unless stated:

  1  §4.2 histogram matching                 27 cells  3 rungs x 3 arch x 3 city
  2  §4.4 city-identity linear probe         18 cells  rungs 225 and 900
  3  §4.4 component retraining A and B       18 runs   rung 900 only
  4  collapsed-cell re-runs at seed 3         4 runs   driven by armc_train.py

No base model is retrained. Probes 1 and 2 are inference only; probe 3 trains
one component of a copy of the cell's model and leaves the cell's checkpoint
untouched; probe 4 is a fresh ladder cell at a new seed and goes through
armc_train.py so that "identical recipe to the ladder" is literal rather than
a claim, with this script only annotating the collapse flag afterwards.

Every dump is CONTRACT.md §3. The core columns of a probe dump are always a
real inference over the held-out city's 150 test images, so a probe that
should reproduce its parent ladder cell can be checked against it; the probe's
own quantities ride in extra columns.

Usage
-----
    python3 -u armc_probes.py --probe 1 --arch mamnet --rung 900 --city chicago
    python3 -u armc_probes.py --probe 2 --arch mamnet --rung 900 --city chicago
    python3 -u armc_probes.py --probe 3 --arch mamnet --rung 900 --city chicago \
        --experiment A
    python3 -u armc_probes.py --probe 4 --annotate_run_id <ladder run_id>
    ... --dry_run                 resolve and print, write nothing
    ... --smoke_test 5 --smoke_epochs 2 --smoke_perm 3 --out_root <scratch>
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

import armc_train                                                # noqa: E402
from shadowtransfer import armc_dump                             # noqa: E402
from shadowtransfer import armc_runtime                          # noqa: E402
from shadowtransfer import armc_probes_common as C               # noqa: E402
from shadowtransfer import armc_probes_ops as OPS                # noqa: E402
from shadowtransfer.armc_dump import sha256_file                 # noqa: E402

PROBE_SUFFIX = {1: "p1histmatch", 2: "p2cityprobe", 3: "p3exp"}
CITIES = ("chicago", "miami", "phoenix")
COLLAPSE_PRED_POS_FRAC = 0.50


def probe_run_id(probe, arch, rung, city, seed, experiment=None):
    parent = C.ladder_run_id(arch, "loco", rung, city, seed)
    suffix = PROBE_SUFFIX[probe] + (experiment if probe == 3 else "")
    return f"{parent}-{suffix}"


def module_shas():
    return {
        "metric_module_sha": sha256_file(BASE / "shadowtransfer/metrics.py"),
        "dump_module_sha": sha256_file(BASE / "shadowtransfer/armc_dump.py"),
        "runtime_module_sha": sha256_file(BASE / "shadowtransfer/armc_runtime.py"),
        "probes_common_sha": sha256_file(BASE / "shadowtransfer/armc_probes_common.py"),
        "probes_ops_sha": sha256_file(BASE / "shadowtransfer/armc_probes_ops.py"),
        "probe_script_sha": sha256_file(Path(__file__).resolve()),
    }


def test_loader(arch, cities, res_dir, cfg, batch_size, smoke_test):
    """DataLoader over the test split of one or more cities, in dataset order.

    Built through armc_train.build_dataset, so the transforms, channels and
    file selection are the ladder's own.
    """
    from torch.utils.data import DataLoader
    # The smoke truncation is PER CITY, not over the concatenated list. Probe 2
    # reads all three cities and a global head would hand its 3-way regression
    # a single class.
    pairs = []
    for c in cities:
        names = armc_train.city_split_files(c, res_dir, "test")
        pairs += [(c, f) for f in (names[:smoke_test] if smoke_test else names)]
    ds = armc_train.build_dataset(arch, pairs, "test", res_dir,
                                  cfg["img_size"], False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True,
                        worker_init_fn=armc_train.dl_worker_init)
    return ds, loader


def reproduction_delta(rows, parent_run_id, label):
    """max |Δ| between a probe's re-inference and the parent cell's stored
    per-image foreground IoU at K=2. An execution fact the job checks on
    itself, not analysis: the same weights on the same images must give the
    same number."""
    import pandas as pd
    df = pd.DataFrame(rows)
    parent = C.ladder_perimage(parent_run_id)
    stored = C.align_on_image_id(df, parent, "iou_tol")
    d = np.abs(df["iou_tol"].to_numpy(dtype=np.float64) - stored)
    out = {"repro_against": parent_run_id, "repro_column": "iou_tol",
           "repro_max_abs_delta": float(d.max()),
           "repro_mean_abs_delta": float(d.mean()),
           "repro_n_images": int(len(d))}
    print(f"  REPRO {label}: max|Δ| {out['repro_max_abs_delta']:.3e} "
          f"over {out['repro_n_images']} images vs {parent_run_id}", flush=True)
    return out


def base_run_fields(args, probe, run_id, parent_run_id, cfg, ckpt, t0):
    """The run.json fields every probe writes identically."""
    import torch
    parent = C.ladder_run_json(parent_run_id)
    comparator_rung = C.COMPARATOR_RUNG[args.rung]
    comparator = C.ladder_run_id(args.arch, "upper", comparator_rung, args.city,
                                 args.seed)
    run = {
        "run_id": run_id,
        "dataset": "st",
        "arm": "C",
        "probe": probe,
        "probe_name": {1: "s4.2 histogram matching",
                       2: "s4.4 city-identity linear probe",
                       3: "s4.4 component retraining"}[probe],
        "arch": args.arch,
        "method": "vanilla",
        "mode": "loco",
        "rung": args.rung,
        "holdout": args.city,
        "res": "hr",
        "seed": args.seed,
        "n_test": None,
        "parent_run_id": parent_run_id,
        "parent_ckpt_path": str(ckpt),
        "parent_ckpt_sha256": sha256_file(ckpt),
        "parent_reporting_metric_value": parent.get("reporting_metric_value"),
        "parent_n_train_per_source_city": parent.get("n_train_per_source_city"),
        "comparator": comparator,
        "comparator_rung": comparator_rung,
        "is_budget_asymmetric": bool(comparator_rung != args.rung),
        "budget_asymmetric_note": (
            "rung 900 has no within-city upper bound: every city holds exactly "
            "450 training images. R at rung 900 is measured against Upper-450, "
            "which is the budget-asymmetric comparison REVIEWS.md names."),
        "run_id_note": (
            "the probe run_id carries its parent cell's identity plus a probe "
            "suffix. It is not a new ladder cell and must not be pooled with "
            "one; target_access below is the probe's, not the parent's."),
        "reporting_metric": "fg_only_tol_k2",
        "reporting_metric_note": (
            "foreground-only IoU at K=2, column iou_tol. CONTRACT.md section 3."),
        "prediction_recipe": C.PREDICTION_RECIPE,
        "band_width_headline": 2,
        "band_widths_written": [0, 2, 5],
        "structure_cols": "deferred",
        "config_verbatim": cfg,
        "cluster": "delta",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_time_limit": os.environ.get("ARMC_TIME_LIMIT"),
        "git_hash": None,
        "git_hash_reason": (
            "left null for arm C artifacts pending the operator's decision on "
            "whether run.json should carry a commit"),
        "smoke": bool(args.smoke_test or args.smoke_epochs or args.smoke_perm),
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    run.update(module_shas())
    return run


# ---------------------------------------------------------------------------
# Probe 1 — §4.2 histogram matching
# ---------------------------------------------------------------------------

def run_probe1(args, run_id, parent_run_id, dump_dir, cfg, ckpt, device, t0):
    import pandas as pd
    import torch

    res_dir = armc_train.RES_DIR["hr"]
    model = C.build_model(args.arch, cfg, device)
    load_info = C.load_ladder_weights(model, ckpt, device)
    geo = armc_dump.load_geo_meta(BASE)

    # The source pool is this cell's OWN training list at its OWN rung, drawn
    # by armc_train's seeded selection, not the source cities' whole pools.
    train_pairs, _, _, train_cities, n_per_city = armc_train.resolve_selection(
        "loco", args.city, args.rung, res_dir, args.seed)
    src_ch, src_gray, n_src = OPS.source_cdfs(train_pairs, res_dir, BASE)

    ds, loader = test_loader(args.arch, [args.city], res_dir, cfg, 1,
                             args.smoke_test)

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference_unmatched")
        t = time.time()
        ds.fda_transform = None
        rows_base, _ = C.infer_rows(model, loader, run_id, args.city, device, geo)
        wall_base = time.time() - t

        tbx.phase("inference_matched_per_channel")
        t = time.time()
        ds.fda_transform = OPS.PerChannelMatcher(src_ch)
        rows_match, probs = C.infer_rows(model, loader, run_id, args.city,
                                         device, geo)
        wall_match = time.time() - t

        tbx.phase("inference_matched_gray")
        t = time.time()
        ds.fda_transform = OPS.GrayMatcher(src_gray)
        rows_gray, _ = C.infer_rows(model, loader, run_id, args.city, device, geo)
        wall_gray = time.time() - t
        ds.fda_transform = None

        tbx.phase("recovery_ratio")
        df = pd.DataFrame(rows_base)
        dfm = pd.DataFrame(rows_match)
        dfg = pd.DataFrame(rows_gray)
        if not (list(df["image_id"]) == list(dfm["image_id"]) == list(dfg["image_id"])):
            raise SystemExit("FATAL the three inference passes disagree on image order")

        comparator = C.ladder_run_id(args.arch, "upper",
                                     C.COMPARATOR_RUNG[args.rung], args.city,
                                     args.seed)
        loco = C.ladder_perimage(parent_run_id)
        upper = C.ladder_perimage(comparator)
        iou_loco = C.align_on_image_id(df, loco, "iou_tol")
        iou_upper = C.align_on_image_id(df, upper, "iou_tol")
        fp_loco = C.align_on_image_id(df, loco, "fp_tol")

        r, ok, head = C.recovery_ratio(dfm["iou_tol"].to_numpy(), iou_loco, iou_upper)
        r_gray, _, _ = C.recovery_ratio(dfg["iou_tol"].to_numpy(), iou_loco, iou_upper)

        rows = df.to_dict("records")
        for i, row in enumerate(rows):
            row["probe"] = 1
            row["rung"] = args.rung
            row["comparator"] = comparator
            row["is_budget_asymmetric"] = bool(C.COMPARATOR_RUNG[args.rung] != args.rung)
            row["target_access"] = "source_only"
            row["iou_tol_matched"] = float(dfm["iou_tol"].iloc[i])
            row["iou_tol_matched_gray"] = float(dfg["iou_tol"].iloc[i])
            row["fp_px_loco"] = int(fp_loco[i])
            row["fp_px_matched"] = int(dfm["fp_tol"].iloc[i])
            row["fp_px_matched_gray"] = int(dfg["fp_tol"].iloc[i])
            row["fp_px_loco_reinfer"] = int(df["fp_tol"].iloc[i])
            row["iou_tol_loco_stored"] = float(iou_loco[i])
            row["iou_tol_upper_stored"] = float(iou_upper[i])
            row["headroom"] = float(head[i])
            row["r_included"] = bool(ok[i])
            row["recovery_ratio_R"] = float(r[i])
            row["recovery_ratio_R_gray"] = float(r_gray[i])

        run = base_run_fields(args, 1, run_id, parent_run_id, cfg, ckpt, t0)
        run.update({
            "n_test": len(rows),
            "target_access": "source_only",
            "target_access_detail": (
                "the matching statistic is the aggregate histogram of this "
                "cell's own source-city TRAINING images. No held-out-city "
                "image, labelled or unlabelled, contributes to it; each test "
                "image is matched toward that statistic using only its own "
                "pixels. Model weights are the parent cell's, unchanged."),
            "intervention": "test-time per-channel CDF matching, no parameter update",
            "match_definition_headline": (
                "per-channel: for each RGB channel independently, the source "
                "pool's aggregate channel CDF is the target and the mapping is "
                "argmin_s |cdf_src[s] - cdf_img[t]|. Columns iou_tol_matched, "
                "fp_px_matched, recovery_ratio_R."),
            "match_definition_published": (
                "grayscale-plus-ratio, verbatim from "
                "final_loco/experiment_utils.histogram_match_image: one "
                "grayscale CDF, each pixel's per-channel ratio re-applied to "
                "the mapped grey value. This is what the published section 4.2 "
                "finding of 0 of 9 cells improved was measured under, so it is "
                "carried alongside in iou_tol_matched_gray and "
                "recovery_ratio_R_gray rather than being replaced."),
            "match_injection_point": (
                "dataset.fda_transform -- the raw uint8 RGB array immediately "
                "after Image.open, before resize, normalisation and the "
                "contrast channel"),
            "n_source_images": n_src,
            "source_cities": train_cities,
            "n_train_per_source_city": n_per_city,
            "train_list_sha256": armc_train.list_sha256(train_pairs),
            "r_definition": (
                "per image, R = (IoU_match - IoU_LOCO) / (IoU_upper - IoU_LOCO) "
                "on foreground-only IoU at K=2. IoU_LOCO and IoU_upper are read "
                "from the stored ladder dumps and are NOT re-inferred."),
            "r_exclusion_rule": f"images with (IoU_upper - IoU_LOCO) < {C.R_EXCLUDE_MIN}",
            "r_n_included": int(ok.sum()),
            "r_n_excluded": int((~ok).sum()),
            "core_columns_note": (
                "the core CONTRACT section 3 columns are the UNMATCHED "
                "re-inference, so this dump reproduces its parent ladder cell "
                "and the matched result sits in iou_tol_matched. probs.npz "
                "holds the MATCHED probabilities; the unmatched ones already "
                "exist in the parent cell's dump."),
            "checkpoint_load": load_info,
            "wall_clock_s": round(time.time() - t0, 2),
            "inference_wall_clock_s": {"unmatched": round(wall_base, 2),
                                       "matched": round(wall_match, 2),
                                       "matched_gray": round(wall_gray, 2)},
            "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        })
        run.update(reproduction_delta(rows_base, parent_run_id, "probe1 unmatched"))
        return rows, probs, run


# ---------------------------------------------------------------------------
# Probe 2 — §4.4 city-identity linear probe
# ---------------------------------------------------------------------------

def _city_of(path):
    """data/Final_data_test/{city}/{res}/{split}/images/{f} -> city."""
    return Path(path).resolve().parents[3].name


def run_probe2(args, run_id, parent_run_id, dump_dir, cfg, ckpt, device, t0):
    import pandas as pd
    import torch

    res_dir = armc_train.RES_DIR["hr"]
    model = C.build_model(args.arch, cfg, device)
    load_info = C.load_ladder_weights(model, ckpt, device)
    geo = armc_dump.load_geo_meta(BASE)

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("feature_extraction")
        ds_f, loader_f = test_loader(args.arch, CITIES, res_dir, cfg,
                                     args.feat_batch, args.smoke_test)
        cities_per_image = [_city_of(p) for p in ds_f.img_paths]
        stems = [Path(p).stem for p in ds_f.img_paths]
        t = time.time()
        X, ids, feat_shape, n_cells_total = OPS.extract_features(
            model, args.arch, loader_f, device, args.cells_per_image, args.seed)
        wall_feat = time.time() - t
        if ids != stems:
            raise SystemExit("FATAL feature extraction order does not match the "
                             "dataset's own file order")
        classes = sorted(set(cities_per_image))
        img_city = np.array([classes.index(c) for c in cities_per_image])
        print(f"  features {X.shape} from {len(stems)} images over {classes}; "
              f"encoder map {feat_shape} = {n_cells_total} cells, "
              f"{args.cells_per_image} sampled per image "
              f"({wall_feat:.1f}s)", flush=True)

        tbx.phase("linear_probe")
        t = time.time()
        stats = OPS.city_identity_probe(
            X, img_city, min(args.cells_per_image, n_cells_total),
            n_folds=args.n_folds, n_perm=args.smoke_perm or args.n_permutations,
            seed=args.seed)
        wall_probe = time.time() - t
        acc_by_key = {(cities_per_image[i], stems[i]): float(a)
                      for i, a in enumerate(stats.pop("per_image_accuracy"))}

        tbx.phase("inference")
        ds_t, loader_t = test_loader(args.arch, [args.city], res_dir, cfg, 1,
                                     args.smoke_test)
        rows_base, probs = C.infer_rows(model, loader_t, run_id, args.city,
                                        device, geo)
        df = pd.DataFrame(rows_base)

        tbx.phase("assemble")
        rows = df.to_dict("records")
        for row in rows:
            row["probe"] = 2
            row["rung"] = args.rung
            row["comparator"] = C.ladder_run_id(
                args.arch, "upper", C.COMPARATOR_RUNG[args.rung], args.city,
                args.seed)
            row["is_budget_asymmetric"] = bool(
                C.COMPARATOR_RUNG[args.rung] != args.rung)
            row["target_access"] = "source_only"
            row["probe_accuracy"] = stats["probe_accuracy"]
            row["null_mean"] = stats["null_mean"]
            row["null_p99"] = stats["null_p99"]
            row["p_value"] = stats["p_value"]
            row["n_folds"] = stats["n_folds"]
            row["n_permutations"] = stats["n_permutations"]
            row["probe_accuracy_image"] = acc_by_key.get(
                (args.city, row["image_id"]), float("nan"))

        run = base_run_fields(args, 2, run_id, parent_run_id, cfg, ckpt, t0)
        run.update({
            "n_test": len(rows),
            "target_access": "source_only",
            "target_access_detail": (
                "the probe reads encoder activations on TEST images of all "
                "three cities and predicts which city an activation came from. "
                "No label of any city is used to change a model weight, and no "
                "held-out-city image enters training -- the parent cell's "
                "weights are frozen and unchanged. The city labels the "
                "regression fits are the probe's own supervision, not the "
                "segmentation task's."),
            "n_probe_images": len(stems),
            "probe_cities": classes,
            "probe_classes": len(classes),
            "encoder_feature_key": C.ENCODER_FEATURE_KEY[args.arch],
            "encoder_feature_shape": list(feat_shape),
            "n_spatial_cells_total": int(n_cells_total),
            "n_spatial_cells_sampled": int(min(args.cells_per_image, n_cells_total)),
            "n_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "subsampling_note": (
                "spatial cells are subsampled per image, seeded, because the "
                "permutation null refits 5 folds x 100 shuffles and the full "
                "map is 259,200 samples. The SAME sample is used for the "
                "observed fit and for every permutation, so the null is "
                "calibrated on exactly the data the observed accuracy was "
                "measured on."),
            "cv": (f"GroupKFold(n_splits={args.n_folds}) with the IMAGE as the "
                   "group. A pixel-level split leaks: two cells of one image "
                   "are near-duplicates and would land on both sides of the "
                   "fold boundary, inflating accuracy."),
            "classifier": ("sklearn LogisticRegression(multinomial, lbfgs, "
                           "max_iter=200) on StandardScaler features fitted on "
                           "the training fold only"),
            "null_definition": (
                "city labels shuffled at IMAGE level, refit with the same fold "
                "structure. One-sided p = (1 + #{null >= observed}) / (1 + n)."),
            "probe_stats": stats,
            "wall_clock_s": round(time.time() - t0, 2),
            "feature_wall_clock_s": round(wall_feat, 2),
            "probe_wall_clock_s": round(wall_probe, 2),
            "checkpoint_load": load_info,
            "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        })
        run.update(reproduction_delta(rows_base, parent_run_id, "probe2 inference"))
        for k in ("probe_accuracy", "null_mean", "null_p99", "p_value"):
            run[k] = stats[k]
        return rows, probs, run


# ---------------------------------------------------------------------------
# Probe 3 — §4.4 component retraining, Experiments A and B
# ---------------------------------------------------------------------------

def run_probe3(args, run_id, parent_run_id, dump_dir, cfg, ckpt, device, t0):
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    res_dir = armc_train.RES_DIR["hr"]
    exp = args.experiment

    # Built with pretrained=True, so the encoder carries its pretrained
    # initialisation BEFORE the ladder checkpoint lands on it. Experiment B
    # restores that snapshot rather than constructing a second model.
    model = C.build_model(args.arch, cfg, device)
    pretrained_encoder = C.snapshot_encoder(model, args.arch)
    load_info = C.load_ladder_weights(model, ckpt, device)
    geo = armc_dump.load_geo_meta(BASE)

    # 25% of the held-out city's own training pool, seeded.
    pool = armc_train.city_split_files(args.city, res_dir, "train")
    n_use = int(len(pool) * args.train_fraction)
    rng = armc_train.stable_rng("armC-probe3", args.city, "train", args.seed)
    picked = sorted(np.asarray(pool)[rng.permutation(len(pool))[:n_use]].tolist())
    train_pairs = [(args.city, f) for f in picked]
    print(f"  retraining set: {n_use} of {len(pool)} {args.city} train images "
          f"({args.train_fraction:.0%}, seed {args.seed})", flush=True)

    ds_tr = armc_train.build_dataset(args.arch, train_pairs, "train", res_dir,
                                     cfg["img_size"], True)
    g = torch.Generator(); g.manual_seed(args.seed)
    loader_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True,
                           num_workers=cfg["num_workers"], pin_memory=True,
                           drop_last=True, generator=g,
                           worker_init_fn=armc_train.dl_worker_init)

    if exp == "A":
        frozen, n_frozen = "encoder", C.freeze_encoder(model, args.arch)
        n_reinit = C.reinit_decoder(model, args.arch)
        n_restored = 0
        lr = args.decoder_lr
        enc_lr_applied = None
    else:
        frozen, n_frozen = "decoder", C.freeze_decoder(model, args.arch)
        n_reinit = 0
        n_restored = C.restore_encoder(model, pretrained_encoder)
        lr = float(cfg["lr"]) / 100.0
        enc_lr_applied = lr
    del pretrained_encoder

    epochs = args.smoke_epochs or args.epochs
    t = time.time()
    stats = OPS.retrain_component(model, args.arch, loader_tr, device, frozen,
                                  epochs, lr)
    wall_train = time.time() - t

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("inference")
        _, loader_te = test_loader(args.arch, [args.city], res_dir, cfg, 1,
                                   args.smoke_test)
        rows_base, probs = C.infer_rows(model, loader_te, run_id, args.city,
                                        device, geo)
        df = pd.DataFrame(rows_base)

        tbx.phase("recovery_ratio")
        comparator = C.ladder_run_id(args.arch, "upper",
                                     C.COMPARATOR_RUNG[args.rung], args.city,
                                     args.seed)
        iou_loco = C.align_on_image_id(df, C.ladder_perimage(parent_run_id), "iou_tol")
        iou_upper = C.align_on_image_id(df, C.ladder_perimage(comparator), "iou_tol")
        r, ok, head = C.recovery_ratio(df["iou_tol"].to_numpy(), iou_loco, iou_upper)

        rows = df.to_dict("records")
        for i, row in enumerate(rows):
            row["probe"] = 3
            row["experiment"] = exp
            row["rung"] = args.rung
            row["comparator"] = comparator
            row["is_budget_asymmetric"] = bool(C.COMPARATOR_RUNG[args.rung] != args.rung)
            row["target_access"] = "target_labels"
            row["frozen_component"] = frozen
            row["encoder_lr_applied"] = enc_lr_applied
            row["encoder_lr_published_ratio"] = C.PUBLISHED_ENCODER_LR_RATIO[args.arch]
            row["iou_tol_loco_stored"] = float(iou_loco[i])
            row["iou_tol_upper_stored"] = float(iou_upper[i])
            row["headroom"] = float(head[i])
            row["r_included"] = bool(ok[i])
            row["recovery_ratio_R"] = float(r[i])

        run = base_run_fields(args, 3, run_id, parent_run_id, cfg, ckpt, t0)
        run["run_id"] = run_id
        run.update({
            "n_test": len(rows),
            "experiment": exp,
            "experiment_name": ("A: freeze encoder, reinitialise decoder, "
                                "retrain the decoder"
                                if exp == "A" else
                                "B: freeze decoder, reset the encoder to its "
                                "pretrained initialisation, retrain the encoder"),
            "frozen_component": frozen,
            "n_frozen_params": n_frozen,
            "n_decoder_modules_reinit": n_reinit,
            "n_encoder_entries_restored": n_restored,
            "target_access": "target_labels",
            "target_access_detail": (
                "the retraining set is LABELLED training images of the "
                f"held-out city ({args.city}): {n_use} images, "
                f"{args.train_fraction:.0%} of its 450-image train pool, seed "
                f"{args.seed}. This is the strictest value in CONTRACT.md and "
                "it applies to our own diagnostic, which is what BxRx-W4 asks "
                "for. These runs are not comparable with any source_only cell."),
            "n_retrain_images": n_use,
            "n_retrain_pool": len(pool),
            "train_fraction": args.train_fraction,
            "retrain_list_sha256": armc_train.list_sha256(train_pairs),
            "retrain_selection": (
                "seeded uniform sample without replacement from the held-out "
                "city's train split. The published Experiment A instead takes "
                "an np.linspace stride over the sorted list; the arm spec asks "
                "for a random 25% at seed 0 and that is what ran."),
            "lr_applied": lr,
            "encoder_lr_applied": enc_lr_applied,
            "encoder_lr_published_ratio": C.PUBLISHED_ENCODER_LR_RATIO[args.arch],
            "parent_lr": float(cfg["lr"]),
            "encoder_lr_note": (
                "App G.9 states a 100x reduction of the original training LR "
                "for Experiment B. The published runs actually reduced by "
                f"{C.PUBLISHED_ENCODER_LR_RATIO[args.arch]:g}x for "
                f"{args.arch} (MAMNet 1x, OGLANet 5x, DINOv3 5x). This run "
                "applies the literal 100x, so its encoder LR is "
                f"{(C.PUBLISHED_ENCODER_LR_RATIO[args.arch] / 100.0):g}x what "
                "the published Experiment B used. Both values are recorded; "
                "neither is silently corrected."
                if exp == "B" else
                "Experiment A does not train the encoder; encoder_lr_applied "
                "is null. The decoder LR is the published Experiment A default "
                f"({args.decoder_lr:g})."),
            "decoder_lr_applied": lr if exp == "A" else None,
            "evaluated_checkpoint": "final",
            "evaluated_checkpoint_note": (
                "the published Experiments A and B train a fixed number of "
                "epochs with no validation split and evaluate the final "
                "weights; that is kept. steps_to_best is reported against the "
                "minimum-mean-training-loss epoch and is a training-dynamics "
                "fact, not a selection rule."),
            "checkpoint_saved": False,
            "checkpoint_saved_note": (
                "the retrained model is not written to disk: 18 runs x 509 MB "
                "is 9 GB against a 550G hard quota, and every reported quantity "
                "is in this dump. The parent checkpoint is untouched."),
            "training": stats,
            "steps_to_best": stats["steps_to_best"],
            "epochs_run": stats["epochs_run"],
            "r_definition": (
                "per image, R = (IoU_retrained - IoU_LOCO) / "
                "(IoU_upper - IoU_LOCO) on foreground-only IoU at K=2. "
                "IoU_LOCO and IoU_upper are read from the stored ladder dumps."),
            "r_exclusion_rule": f"images with (IoU_upper - IoU_LOCO) < {C.R_EXCLUDE_MIN}",
            "r_n_included": int(ok.sum()),
            "r_n_excluded": int((~ok).sum()),
            "repro_against": None,
            "repro_note": (
                "no stored quantity is comparable: this dump is a model that "
                "did not exist before. The parent cell's stored IoU enters only "
                "as R's baseline."),
            "checkpoint_load": load_info,
            "wall_clock_s": round(time.time() - t0, 2),
            "train_wall_clock_s": round(wall_train, 2),
            "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        })
        return rows, probs, run


# ---------------------------------------------------------------------------
# Probe 4 — collapsed-cell re-runs: the collapse flag
# ---------------------------------------------------------------------------
#
# The re-run itself goes through armc_train.py, unchanged, so "identical
# recipe to the ladder" is literal. This step only reads the finished dump and
# records whether the fresh seed collapsed too.

def annotate_probe4(run_id, out_root):
    import pandas as pd
    d = (Path(out_root) if out_root else BASE / "dumps") / "st" / run_id
    if not (d / ".DONE").exists():
        raise SystemExit(f"FATAL {run_id} has no .DONE at {d}; nothing to annotate")
    armc_runtime.preflight_write_targets(BASE, [d / "run.json"], label="annotate")

    df = pd.read_parquet(d / "perimage.parquet")
    frac = (df["n_pred_pos"].to_numpy(dtype=np.float64)
            / df["n_valid_px"].to_numpy(dtype=np.float64))
    mean_frac = float(frac.mean())
    collapsed = bool(mean_frac > COLLAPSE_PRED_POS_FRAC)

    run = json.loads((d / "run.json").read_text())
    run["probe"] = 4
    run["probe_name"] = "collapsed-cell re-run at a fresh seed"
    run["probe4_pred_pos_frac_mean"] = mean_frac
    run["probe4_pred_pos_frac_median"] = float(np.median(frac))
    run["probe4_pred_pos_frac_max"] = float(frac.max())
    run["probe4_n_images_over_threshold"] = int((frac > COLLAPSE_PRED_POS_FRAC).sum())
    run["probe4_collapse_threshold"] = COLLAPSE_PRED_POS_FRAC
    run["probe4_collapsed"] = collapsed
    run["probe4_collapse_definition"] = (
        "mean over test images of (n_pred_pos / n_valid_px) exceeding "
        f"{COLLAPSE_PRED_POS_FRAC}. A collapsed cell predicts foreground almost "
        "everywhere, so its foreground IoU degenerates to the ground-truth "
        "foreground rate and is not a detection result.")
    run["sanity_flags"] = list(run.get("sanity_flags", []))
    if collapsed:
        run["sanity_flags"].append(
            f"COLLAPSED at the fresh seed: mean pred-positive fraction "
            f"{mean_frac:.4f} > {COLLAPSE_PRED_POS_FRAC}")
    (d / "run.json").write_text(json.dumps(run, indent=2))

    print("=" * 78)
    print(f"PROBE 4 {run_id}")
    print(f"  fg IoU K=2               {run.get('headline_fg_iou_tol_pct')}")
    print(f"  mean pred-positive frac  {mean_frac:.4f}")
    print(f"  images over {COLLAPSE_PRED_POS_FRAC}          "
          f"{run['probe4_n_images_over_threshold']} of {len(df)}")
    print(f"  steps_to_best            {run.get('steps_to_best')} "
          f"(best_epoch {run.get('best_epoch')})")
    print(f"  COLLAPSED                {collapsed}")
    print("=" * 78, flush=True)
    return collapsed


# ---------------------------------------------------------------------------

def sanity_flags(rows, run, parent_run_id):
    """Assertions the job runs on itself. Every outcome is logged, not only
    the failures."""
    import pandas as pd
    df = pd.DataFrame(rows)
    flags, checks = [], {}

    checks["n_test_is_150"] = (len(df) == C.N_TEST)
    if not checks["n_test_is_150"] and not run["smoke"]:
        flags.append(f"n_test is {len(df)}, expected {C.N_TEST}")

    parent = C.ladder_run_json(parent_run_id)
    checks["prediction_recipe_matches_ladder"] = (
        run["prediction_recipe"] == parent.get("prediction_recipe"))
    if not checks["prediction_recipe_matches_ladder"]:
        flags.append("prediction_recipe differs from the parent ladder cell")

    checks["metric_module_sha_matches_ladder"] = (
        run["metric_module_sha"] == parent.get("metric_module_sha"))
    if not checks["metric_module_sha_matches_ladder"]:
        flags.append(
            f"metric_module_sha {run['metric_module_sha'][:12]} differs from "
            f"the parent cell's {str(parent.get('metric_module_sha'))[:12]}")

    n_empty = int(df["empty_fg_valid"].sum())
    checks["n_empty_fg_valid"] = n_empty
    if n_empty:
        flags.append(f"empty_fg_valid on {n_empty} images")

    if float(df["iou_tol"].std()) < 1e-6:
        flags.append("per-image iou_tol has ~zero variance")

    if "recovery_ratio_R" in df:
        r = df["recovery_ratio_R"].to_numpy(dtype=np.float64)
        r = r[np.isfinite(r)]
        checks["mean_R"] = float(r.mean()) if len(r) else float("nan")
        checks["median_R"] = float(np.median(r)) if len(r) else float("nan")
        checks["n_R"] = int(len(r))
        if len(r) and float(r.mean()) > 0 and run["probe"] == 1:
            # Probe 1 only. The published section 4.2 reading is 0 of 9 cells
            # improved at rung 450, so any improvement is a finding, not an
            # error. Probe 3 is EXPECTED to recover (Table 36 puts Exp A at
            # 0.67 for MAMNet), so a positive R there is not remarkable and
            # flagging it would train the reader to ignore the flag.
            flags.append(f"mean R = {r.mean():.4f} > 0 (any improvement); the "
                         f"published section 4.2 reading is 0 of 9 cells "
                         f"improved at rung 450 -- a finding, not an error")
    if "iou_tol_matched" in df:
        checks["mean_iou_tol"] = float(df["iou_tol"].mean())
        checks["mean_iou_tol_matched"] = float(df["iou_tol_matched"].mean())
        checks["mean_iou_tol_matched_gray"] = float(df["iou_tol_matched_gray"].mean())
        checks["mean_fp_px_loco"] = float(df["fp_px_loco"].mean())
        checks["mean_fp_px_matched"] = float(df["fp_px_matched"].mean())

    d = run.get("repro_max_abs_delta")
    if d is not None and d > 1e-6:
        flags.append(f"re-inference does not reproduce the parent cell: "
                     f"max|Δ| iou_tol = {d:.3e}")

    run["assertion_results"] = checks
    run["sanity_flags"] = flags
    print("ASSERTIONS")
    for k, v in checks.items():
        print(f"  {k:36s} {v}")
    print(f"  sanity_flags                         {flags or 'none'}", flush=True)
    return run


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--arch", choices=sorted(C.ARCH_DIR))
    ap.add_argument("--rung", type=int, choices=[225, 450, 900])
    ap.add_argument("--city", choices=sorted(CITIES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--experiment", choices=["A", "B"],
                    help="probe 3 only")
    ap.add_argument("--annotate_run_id", default=None,
                    help="probe 4 only: the finished ladder cell to flag")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--timeout_s", type=int, default=1800)
    # probe 2
    ap.add_argument("--cells_per_image", type=int, default=32)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_permutations", type=int, default=100)
    ap.add_argument("--feat_batch", type=int, default=8)
    # probe 3
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train_fraction", type=float, default=0.25)
    ap.add_argument("--decoder_lr", type=float, default=1e-3,
                    help="published Experiment A decoder LR")
    # stage-1 smoke
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--smoke_epochs", type=int, default=0)
    ap.add_argument("--smoke_perm", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    if args.probe == 4:
        if not args.annotate_run_id:
            raise SystemExit("FATAL probe 4 needs --annotate_run_id")
        annotate_probe4(args.annotate_run_id, args.out_root)
        return

    for name in ("arch", "rung", "city"):
        if getattr(args, name) is None:
            raise SystemExit(f"FATAL --{name} is required for probe {args.probe}")
    if args.probe == 3:
        if args.experiment is None:
            raise SystemExit("FATAL probe 3 needs --experiment A|B")
        if args.rung != 900:
            raise SystemExit("FATAL probe 3 runs at rung 900 only")
    if args.probe == 2 and args.rung == 450:
        raise SystemExit("FATAL probe 2 runs at rungs 225 and 900 only")
    smoke = bool(args.smoke_test or args.smoke_epochs or args.smoke_perm)
    if smoke and args.out_root is None:
        raise SystemExit("FATAL --smoke_* requires --out_root")

    parent_run_id = C.ladder_run_id(args.arch, "loco", args.rung, args.city,
                                    args.seed)
    run_id = probe_run_id(args.probe, args.arch, args.rung, args.city,
                          args.seed, args.experiment)
    dump_dir = (Path(args.out_root) if args.out_root else BASE / "dumps") / "st" / run_id

    if (dump_dir / ".DONE").exists() and not args.force:
        print(f"[{run_id}] .DONE exists at {dump_dir} — nothing to do")
        return

    # Refuse a quarantined output BEFORE the expensive part, never at the
    # moment of writing (CLAUDE.md, "Quarantined paths").
    armc_runtime.preflight_write_targets(BASE, [
        dump_dir / "perimage.parquet", dump_dir / "probs.npz",
        dump_dir / "run.json", dump_dir / ".DONE"], label="dump")

    parent = C.ladder_run_json(parent_run_id)
    ckpt = C.ladder_checkpoint(parent_run_id)
    cfg = parent["config_verbatim"]
    comparator = C.ladder_run_id(args.arch, "upper",
                                 C.COMPARATOR_RUNG[args.rung], args.city,
                                 args.seed)

    print("=" * 78)
    print(f"ARM C PROBE {args.probe}   run_id  {run_id}")
    print(f"  parent cell           {parent_run_id}")
    print(f"  parent fg IoU K=2     {parent.get('reporting_metric_value')}")
    print(f"  checkpoint            {ckpt}")
    print(f"  comparator            {comparator} "
          f"(budget-asymmetric: {C.COMPARATOR_RUNG[args.rung] != args.rung})")
    print(f"  arch/rung/city/seed   {args.arch} / {args.rung} / {args.city} / {args.seed}")
    if args.probe == 3:
        print(f"  experiment            {args.experiment}  epochs "
              f"{args.smoke_epochs or args.epochs}  fraction {args.train_fraction}")
    if args.probe == 2:
        print(f"  probe                 {args.cells_per_image} cells/image, "
              f"{args.n_folds} folds, "
              f"{args.smoke_perm or args.n_permutations} permutations")
    print(f"  dump                  {dump_dir}")
    print(f"  slurm job             {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    if smoke:
        print(f"  SMOKE                 test<={args.smoke_test} "
              f"epochs<={args.smoke_epochs} perms<={args.smoke_perm}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    seed_sites = armc_train.seed_everything(args.seed)
    C.enter_arch_tree(args.arch)
    device = "cuda"

    fn = {1: run_probe1, 2: run_probe2, 3: run_probe3}[args.probe]
    rows, probs, run = fn(args, run_id, parent_run_id, dump_dir, cfg, ckpt,
                          device, t0)
    run["seed_sites"] = seed_sites
    run["checkpoint_sha256"] = run["parent_ckpt_sha256"]
    run = sanity_flags(rows, run, parent_run_id)

    armc_dump.write_dump(dump_dir, rows, probs, run)

    import pandas as pd
    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"[{run_id}]")
    print(f"  fg IoU K=2 (core)       {float(df['iou_tol'].mean()) * 100:.4f}")
    if "iou_tol_matched" in df:
        print(f"  fg IoU K=2 matched      "
              f"{float(df['iou_tol_matched'].mean()) * 100:.4f}  (gray "
              f"{float(df['iou_tol_matched_gray'].mean()) * 100:.4f})")
    if "recovery_ratio_R" in df:
        r = df["recovery_ratio_R"].to_numpy(dtype=np.float64)
        r = r[np.isfinite(r)]
        print(f"  mean R                  {r.mean():.4f}  (n={len(r)}, "
              f"{run['r_n_excluded']} excluded)")
    if args.probe == 2:
        print(f"  probe accuracy          {run['probe_accuracy']:.4f}  "
              f"(chance {run['probe_stats']['chance']:.4f}, null mean "
              f"{run['null_mean']:.4f}, null p99 {run['null_p99']:.4f}, "
              f"p={run['p_value']:.4f})")
    if args.probe == 3:
        print(f"  steps_to_best           {run['steps_to_best']} "
              f"(best epoch {run['training']['best_epoch']} of "
              f"{run['training']['epochs_run']})")
    print(f"  images                  {len(df)}")
    print(f"  target_access           {run['target_access']}")
    print(f"  metric_module_sha       {run['metric_module_sha'][:16]}")
    print(f"  wall clock              {run['wall_clock_s']:.1f}s")
    print(f"  peak VRAM alloc         {run['peak_vram_alloc_gb']:.2f} GiB")
    print(f"  sanity flags            {run['sanity_flags'] or 'none'}")
    print(f"  -> {dump_dir}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
