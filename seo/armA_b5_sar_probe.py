#!/usr/bin/env python3
"""ARM A BLOCK 5 (D-B5-1) -- s4.2 histogram matching on the SAR cells. Two tasks.

  --task prep   once per cell, no GPU work: the source CDF over the cell's own
                3500 training patches (armA_seo_histmatch.source_cdfs_seo, the
                rule every block-4 probe used), the test-patch metadata, and the
                identity checks (training pool == the checkpoint's, by
                train_sun_elev_mean of the vanilla parent).
  --task shard  one contiguous slice [start, stop) of the held-out test stream.
                SAR is EPISODIC (sar.restore_state before every image; ema and
                optimiser are local to adapt_image), so an image's result does not
                depend on which images share its job -- that is what licenses
                slicing, and the smoke measures it (a slice at a nonzero offset is
                compared per image against the stored full-stream run).

Per shard, four conditions through the SAR parent's own route and recipe:
  unadapted_unmatched   run_pass(unadapted)   -> compared to stored 'unadapted'
  sar_unmatched         run_pass(sar)         -> compared to stored SAR pass
  sar_matched_gray      run_pass(sar) on the PUBLISHED gray-matched images
  sar_matched           run_pass(sar) on arm C's per-channel-matched images
Every condition is scored TWICE from the same prediction: SAR's own convention
(band_counts on the plain mask, unseen counted as background -- what the stored
SAR score is) and arm A's (seo_dump.perimage_row, unseen excluded). The merge
(built after the operator rules the gate) assembles shards into the dump.
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

import armA_pilot_train as P                                     # noqa: E402
import armA_seo_histmatch as hm                                  # noqa: E402
from shadowtransfer import armA_b5_sar as B5                     # noqa: E402
from shadowtransfer import armc_runtime, seo_budget, seo_dump    # noqa: E402
from shadowtransfer import armA_a4_sar_seo as SEO                # noqa: E402
from shadowtransfer.armc_dump import sha256_file                 # noqa: E402


def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["prep", "shard"])
    ap.add_argument("--parent_dump_dir", required=True)
    ap.add_argument("--work_dir", required=True, help="per-cell working dir")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=0)
    ap.add_argument("--repeat_unmatched", type=int, default=0,
                    help="smoke: extra sar_unmatched passes (within-process check)")
    ap.add_argument("--band_only", action="store_true",
                    help="SAR band measurement (D-B5-2/D-B5-3): unadapted + the unmatched SAR "
                         "pass and its repeats only; the matched conditions are not run")
    ap.add_argument("--skip_unadapted", action="store_true",
                    help="stage (b), operator 2026-10-26: run sar_unmatched (in-run comparator), "
                         "sar_matched_gray (published rule) and sar_matched (per-channel, because "
                         "the S-EO method probes report it alongside); the unadapted pass is "
                         "already stored and is not re-run")
    ap.add_argument("--timeout_s", type=int, default=3000)
    ap.add_argument("--vanilla_dumps", default=SEO.DEF_VANILLA_DUMPS)
    ap.add_argument("--budget_dir", default=SEO.DEF_BUDGET_DIR)
    ap.add_argument("--splits_dir", default=SEO.DEF_SPLITS_DIR)
    ap.add_argument("--img_root", default=SEO.DEF_IMG_ROOT)
    ap.add_argument("--mask_root", default=SEO.DEF_MASK_ROOT)
    ap.add_argument("--patch_index", default=SEO.DEF_PATCH_INDEX)
    return ap.parse_args()


def put(path, write):
    """{path}.new -> verify non-empty on disk -> rename."""
    p = Path(path)
    tmp = p.with_name(p.name + ".new")
    write(tmp)
    if not tmp.exists() or tmp.stat().st_size == 0:
        raise SystemExit(f"FATAL {tmp} empty or absent after write")
    tmp.replace(p)


def save_npz(path, compressed, **arrays):
    """np.savez to an explicit handle so the name stays {path}.new (no .npz added)."""
    with open(path, "wb") as f:
        (np.savez_compressed if compressed else np.savez)(f, **arrays)


def prep(args, R):
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    vdir = Path(args.vanilla_dumps) / R["vanilla_run_id"]
    if not (vdir / ".DONE").exists():
        raise SystemExit(f"FATAL vanilla parent {vdir} has no .DONE")
    vrun = json.loads((vdir / "run.json").read_text())
    if vrun["ckpt_path"] != R["ckpt"]:
        raise SystemExit(f"FATAL SAR ckpt {R['ckpt']} != vanilla parent's {vrun['ckpt_path']}")
    man_path = Path(args.budget_dir) / f"{R['fold']}_loco.json"
    if str(man_path) != R["run"]["split_meta"]["budget_manifest"]:
        raise SystemExit(f"FATAL manifest {man_path} != the SAR parent's "
                         f"{R['run']['split_meta']['budget_manifest']}")
    man = json.loads(man_path.read_text())
    if man["budget_rule_sha"] != seo_budget.budget_rule_sha():
        raise SystemExit("FATAL budget manifest rule sha != live rule")
    idx = json.loads(Path(args.patch_index).read_text())
    split = {}
    for name in ("train", "val", "test"):
        _ds, rows, bases, _ = P.build_split(args.img_root, args.mask_root, idx,
                                            args.splits_dir, name, man[name], False,
                                            name, False)
        split[name] = (rows, bases)
    rows_tr, rows_te = split["train"][0], split["test"][0]
    if [B5.pid_of(r) for r in rows_te] != list(man["test"]):
        raise SystemExit("FATAL test rows are not in manifest order")
    sun = float(np.mean([r["sun_el"] for r in rows_tr]))
    if abs(sun - float(vrun["train_sun_elev_mean"])) > 5e-4:
        raise SystemExit(f"FATAL train sun mean {sun} != vanilla parent's "
                         f"{vrun['train_sun_elev_mean']}: wrong training pool")
    pidx = P.patch_index_map(split["train"][1] + split["val"][1] + split["test"][1])
    meta = {}
    for r in rows_te:
        pid = B5.pid_of(r)
        meta[pid] = {"image_id": pid, "aoi_id": r["aoi_id"], "source_image_id": r["image_id"],
                     "patch_index": pidx[pid], "patch_y": r["y"], "patch_x": r["x"],
                     "city": r["city"], "split": "test", "acquisition_date": r["acq"],
                     "sun_elevation": r["sun_el"], "sun_azimuth": r["sun_az"],
                     "lon": r["lon"], "lat": r["lat"],
                     "sun_dist": abs(float(r["sun_el"]) - sun)}
    src_ch, src_gray, prov = hm.source_cdfs_seo(rows_tr, args.img_root, args.mask_root,
                                                384, exclude_unseen=True, label=R["probe_run_id"])
    put(work / "source_cdf.npz", lambda t: save_npz(t, False, src_ch=src_ch, src_gray=src_gray))
    out = {"parent_run_id": R["parent_run_id"], "parent_dump_dir": R["parent_dir"],
           "vanilla_run_id": R["vanilla_run_id"], "vanilla_dump_dir": str(vdir),
           "train_sun_elev_mean": sun, "source_pool": prov,
           "source_cities": man["source_cities"], "budget_manifest": str(man_path),
           "budget_manifest_sha256": hashlib.sha256(man_path.read_bytes()).hexdigest(),
           "budget_rule_sha": man["budget_rule_sha"],
           "source_cdf_sha256": sha256_file(work / "source_cdf.npz"),
           "test_ids": list(man["test"]), "meta": meta,
           "rule_identity_check": hm.assert_rule_identity(),
           "written_utc": datetime.now(timezone.utc).isoformat()}
    put(work / "prep.json", lambda t: t.write_text(json.dumps(out)))
    (work / "prep.DONE").touch()
    print(f"[{R['probe_run_id']}] prep done: n_test {len(out['test_ids'])}, "
          f"train sun {sun:.6f}, source px valid {prov['source_px_valid']}", flush=True)


def route_setup(R, args):
    """The SAR parent's own route, set up exactly as its driver did."""
    if R["arch"] == "crossearth":
        from shadowtransfer import armA_t1_cesar_adapt as CE
        from shadowtransfer import armf_crossearth_runtime as cert
        bb = R["run"]["backbone_init"]
        if cert.sha256_file(bb["path"]) != bb["sha256"] or os.path.getsize(bb["path"]) != bb["bytes"]:
            raise SystemExit("FATAL CrossEarth backbone differs from the SAR parent's")
        repo = R["run"]["arch_branch_resolutions"]["arch_tree_entry"]["repo_root"]
        CE.enter_arch_tree(repo)
        inst = CE.install()
        ckpt, cfg, _v, _nc = CE.resolve_vanilla_cell(R["fold"], R["seed"], args.vanilla_dumps,
                                                     repo, bb["path"])
        cfg_cmp = {k: v for k, v in cfg.items() if not k.startswith("_ce_")}
        route = {"route": "armA_t1_cesar_run (CrossEarth)", "repo_root": repo,
                 "rebindings": inst["rebindings"]}
    else:
        ckpt, cfg, _v = SEO.resolve_vanilla_cell(R["arch"], R["fold"], R["seed"],
                                                 args.vanilla_dumps)
        SEO.enter_arch_tree(R["arch"])
        cfg_cmp = cfg
        route = {"route": "armA_a4_sar_run (mamnet/dinov3/oglanet)"}
    if str(ckpt) != R["ckpt"] or json.loads(json.dumps(cfg_cmp)) != R["run"]["config_verbatim"]:
        raise SystemExit("FATAL route resolved a checkpoint/config other than the SAR parent's")
    route["dataset_rebinds"] = B5.install_dataset(R["arch"])
    return Path(ckpt), cfg, route


SUMMARY_KEYS = ("pass_id", "steps", "lr", "lr_multiplier", "miou_fg_tol2",
                "n_collected_params", "episodic_state_isolation",
                "n_images_state_changed_by_adaptation", "n_reset_fires",
                "n_updates_skipped", "wall_clock_s", "s_per_image", "peak_gpu_mem_gb",
                "checkpoint_epoch", "norm_mode", "n_bn2d_modules")


def compare_stored(fresh_rows, stored, pass_id, steps):
    """Per image: does the fresh pass reproduce the STORED rows, every step?"""
    import pandas as pd
    st = stored[stored["pass_id"] == pass_id]
    m = pd.DataFrame(fresh_rows).merge(st, on=["image_id", "step"], how="left",
                                       suffixes=("", "_st"), validate="one_to_one")
    if m["tp_k2_st"].isna().any():
        raise SystemExit(f"FATAL {int(m['tp_k2_st'].isna().sum())} fresh rows have no "
                         f"stored row in pass {pass_id}")
    eq = np.ones(len(m), dtype=bool)
    for c in B5.COUNT_COLS:
        eq &= m[c].to_numpy() == m[c + "_st"].to_numpy()
    m["_eq"] = eq
    m["_dent"] = (m["mean_entropy"] - m["mean_entropy_st"]).abs()
    out = {}
    for pid, g in m.groupby("image_id", sort=False):
        bad = g.loc[~g["_eq"], "step"]
        fin = g[g["step"] == steps].iloc[0]
        out[pid] = {"repro_counts_equal_all_steps": bool(g["_eq"].all()),
                    "repro_first_divergent_step": int(bad.min()) if len(bad) else -1,
                    "repro_iou_fg_tol2_delta": float(fin["iou_fg_tol2"] - fin["iou_fg_tol2_st"]),
                    "repro_max_abs_mean_entropy_delta": float(g["_dent"].max()),
                    "sar_iou_fg_tol2_stored": float(fin["iou_fg_tol2_st"])}
    return out


def shard(args, R):
    import pandas as pd
    import torch
    work = Path(args.work_dir)
    if not (work / "prep.DONE").exists():
        raise SystemExit(f"FATAL {work}/prep.DONE absent: run --task prep first")
    pr = json.loads((work / "prep.json").read_text())
    if pr["parent_run_id"] != R["parent_run_id"]:
        raise SystemExit("FATAL prep.json belongs to another cell")
    if sha256_file(work / "source_cdf.npz") != pr["source_cdf_sha256"]:
        raise SystemExit("FATAL source_cdf.npz changed since prep")
    cdf = np.load(work / "source_cdf.npz")
    n_test = len(pr["test_ids"])
    if not (0 <= args.start < args.stop <= n_test):
        raise SystemExit(f"FATAL slice [{args.start},{args.stop}) outside [0,{n_test})")
    ids = pr["test_ids"][args.start:args.stop]
    sdir = work / "shards" / f"{args.start:05d}_{args.stop:05d}"
    if (sdir / ".DONE").exists():
        print(f"[{sdir}] .DONE exists -- nothing to do")
        return
    sdir.mkdir(parents=True, exist_ok=True)
    armc_runtime.preflight_write_targets(
        BASE, [sdir / n for n in ("rows.parquet", "probs_gray.npz", "shard.json", ".DONE")],
        label=f"{R['probe_run_id']}[{args.start}:{args.stop}]")
    t0 = time.time()
    seeds = P.seed_everything(R["seed"])
    label_assert = SEO.assert_no_label_tensor_reaches_adaptation()
    ckpt, cfg, route = route_setup(R, args)
    device = torch.device("cuda")
    ctx = {"arch": R["arch"], "fold": R["fold"], "cfg": cfg, "ckpt": ckpt,
           "roots": {"img_root": args.img_root, "mask_root": args.mask_root,
                     "patch_index": args.patch_index, "splits_dir": args.splits_dir},
           "meta": pr["meta"], "recipe": R["recipe"], "probe_run_id": R["probe_run_id"]}
    matchers = {None: None, "gray": hm.SEOGrayMatcher(cdf["src_gray"]),
                "perch": hm.SEOPerChannelMatcher(cdf["src_ch"])}
    stored = pd.read_parquet(Path(R["parent_dir"]) / "perimage.parquet")
    conds = list(B5.CONDITIONS)
    if args.skip_unadapted and args.band_only:
        raise SystemExit("FATAL --skip_unadapted and --band_only are exclusive")
    if args.skip_unadapted and conds[0][0] != "unadapted_unmatched":
        raise SystemExit(f"FATAL CONDITIONS[0] is {conds[0][0]}, not unadapted_unmatched")
    plan = conds[(1 if args.skip_unadapted else 0):2] + [
        (f"sar_unmatched_repeat{i + 1}", None, "sar")
        for i in range(args.repeat_unmatched)] + ([] if args.band_only else conds[2:])
    cudnn = {"deterministic": bool(torch.backends.cudnn.deterministic),
             "benchmark": bool(torch.backends.cudnn.benchmark),
             "note": "left at the SAR routes' own state; neither route sets them"}
    out_rows, summaries, probs_gray, cache, checks = [], {}, {}, {}, {}
    with armc_runtime.PhaseTimeout(args.timeout_s, sdir, label="b5 sar shard") as tbx:
        for name, mk, kind in plan:
            if mk not in cache:
                tbx.phase(f"materialise_{mk or 'unmatched'}")
                cache = {mk: B5.materialise(ctx, ids, matchers[mk])}
            images, seen = cache[mk]
            spec = R["unadapted_spec"] if kind == B5.UNADAPTED else R["sar_spec"]
            tbx.phase(f"pass_{name}")
            summ, rows, arows, probs = B5.run_condition(
                ctx, spec, images, seen, device, keep_probs=(name == "sar_matched_gray"))
            probs_gray.update(probs)
            summaries[name] = {k: summ.get(k) for k in SUMMARY_KEYS}
            print(f"  [{name}] VRAM peak {summ['peak_gpu_mem_gb']} GB", flush=True)
            if kind == "sar" and len(ids) >= 2 and not summ["n_images_state_changed_by_adaptation"]:
                raise SystemExit(f"FATAL {name}: SAR moved no parameter on any image")
            rep = compare_stored(rows, stored, spec["pass_id"], spec["steps"]) if mk is None else {}
            ev = {}
            for r in rows:
                e = ev.setdefault(r["image_id"], [0, 0])
                e[0] += bool(r["reset_fired"])
                e[1] += r["update_skipped"] is not None
            fin = {r["image_id"]: r for r in rows if r["step"] == spec["steps"]}
            for i, a in enumerate(arows):
                pid = a["image_id"]
                if pid != images[i]["image_id"]:
                    raise SystemExit("FATAL unseen-excluded rows out of order")
                row = {"condition": name, "test_index": args.start + i, **a}
                for c in B5.COUNT_COLS + ("iou_fg_tol2", "pred_fg_fraction"):
                    row[f"sar_{c}"] = fin[pid][c]
                row["sar_n_reset_fired"], row["sar_n_update_skipped"] = ev[pid]
                row.update(rep.get(pid, {}))
                out_rows.append(row)
            if rep:
                d = np.array([v["repro_iou_fg_tol2_delta"] for v in rep.values()])
                checks[name] = {
                    "n_images": len(rep),
                    "n_counts_equal_all_steps": int(sum(v["repro_counts_equal_all_steps"]
                                                        for v in rep.values())),
                    "max_abs_iou_fg_tol2_delta": float(np.abs(d).max()),
                    "slice_mean_delta_pct": float(100 * d.mean()),
                    "max_abs_mean_entropy_delta": float(max(
                        v["repro_max_abs_mean_entropy_delta"] for v in rep.values()))}
                print(f"  [{name}] vs stored: {checks[name]}", flush=True)
        tbx.phase("write_shard")
        df = pd.DataFrame(out_rows)
        # seo_dump._write_parquet: the empty category maps must be typed MAP, not
        # inferred struct<> (seo_dump.py docstring; this smoke hit it, 23012249-51).
        put(sdir / "rows.parquet", lambda t: seo_dump._write_parquet(df, t))
        put(sdir / "probs_gray.npz", lambda t: save_npz(t, True, **probs_gray))
        rec = {"probe_run_id": R["probe_run_id"], "parent_run_id": R["parent_run_id"],
               "start": args.start, "stop": args.stop, "n_images": len(ids),
               "plan": [p[0] for p in plan], "band_only": bool(args.band_only),
               "skip_unadapted": bool(args.skip_unadapted),
               "n_sar_unmatched_passes": 1 + args.repeat_unmatched,
               "summaries": summaries, "repro_vs_stored": checks,
               "recipe": R["recipe"], "route": route, "seeds": seeds, "cudnn": cudnn,
               "no_label_assertion": label_assert, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "node": os.environ.get("SLURMD_NODENAME"), "gpu": os.environ.get("ARMA_GPU_NAME"),
               "slurm_time_limit": os.environ.get("ARMA_TIME_LIMIT"),
               "git_hash": os.environ.get("ARMA_GIT_HASH"),
               "git_tree_dirty_paths": os.environ.get("ARMA_GIT_DIRTY"),
               "compute_note": os.environ.get("ST_COMPUTE_NOTE"),
               "peak_vram_alloc_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
               "wall_clock_s": round(time.time() - t0, 1),
               "probe_script_sha": sha256_file(Path(__file__).resolve()),
               "b5_module_sha": sha256_file(BASE / "shadowtransfer/armA_b5_sar.py"),
               "written_utc": datetime.now(timezone.utc).isoformat()}
        put(sdir / "shard.json", lambda t: t.write_text(json.dumps(rec, indent=1, default=str)))
        (sdir / ".DONE").touch()
    print(f"[{R['probe_run_id']}] shard [{args.start},{args.stop}) done in "
          f"{rec['wall_clock_s']}s -> {sdir}", flush=True)


def main():
    args = _cli()
    R = B5.resolve(args.parent_dump_dir)
    print("=" * 78)
    print(f"ARM A BLOCK 5 (D-B5-1) -- SAR s4.2 probe  task {args.task}")
    print(f"  probe run_id   {R['probe_run_id']}   (METHOD FAMILY; never in the headline)")
    print(f"  SAR parent     {R['parent_dir']}")
    print(f"  vanilla ckpt   {R['ckpt']}")
    print(f"  recipe         {R['recipe']}  pass {R['sar_spec']['pass_id']}")
    print(f"  work dir       {args.work_dir}   slice [{args.start},{args.stop})")
    print(f"  band_only      {args.band_only}   unmatched SAR passes {1 + args.repeat_unmatched}")
    print(f"  slurm job      {os.environ.get('SLURM_JOB_ID')} on {os.environ.get('SLURMD_NODENAME')}")
    print("=" * 78, flush=True)
    if args.task == "prep":
        prep(args, R)
    else:
        shard(args, R)


if __name__ == "__main__":
    main()
