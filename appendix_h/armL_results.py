"""
Arm L — assemble results/15_armL_ssl_regime.json from the finished dumps.

READ THIS BEFORE RUNNING IT
---------------------------
CLAUDE.md is explicit that this box "does NOT aggregate across images into a
claim ... anything that pools images into a mean, a CI, a p-value, or a ranking
happens on the SSH box", and that the legacy bootstrap/FDR code here is frozen
and must not be called from a new arm. The arm L spec asks for exactly that:
per-cell CIs, paired bootstrap over per-image deltas at B = 10,000, and
Holm correction over three contrasts.

The two cannot both be honoured. This script is written so the work is
specified and reviewable, and it does NOT call the legacy
`paired_bootstrap_gap` / `bh_fdr` path -- the bootstrap here is a local,
image-clustered resample and Holm is eight lines. Whether it runs on Delta or
whether the dumps are copied to the analysis host and this logic runs there is an OPERATOR
DECISION. It is not wired into armL_submit.sh and no SLURM job calls it.

    python armL_results.py --exp_nn 15 [--dumps_root ...] [--out ...]

Gap sign convention
-------------------
CONVENTIONS.md defines gap = LOCO minus the matched within-city upper bound,
which is NEGATIVE when LOCO is worse. Arm C's tables in REVIEWS.md report the
negation (+6.04 at rung 450). Both are emitted under names that cannot be
confused: `gap_*` follows CONVENTIONS.md, `gap_*_armc_sign` is its negation.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent

CELLS = {
    "sup_hi": ("supervised", 1e-4, "high"),
    "sup_lo": ("supervised", 1e-5, "low"),
    "ssl_hi": ("ssl", 1e-4, "high"),
    "ssl_lo": ("ssl", 1e-5, "low"),
}
CITIES = ("chicago", "miami", "phoenix")
SEEDS = (0, 1, 2)
CONTRASTS = (("ssl_lo", "sup_hi"), ("ssl_lo", "ssl_hi"), ("ssl_lo", "sup_lo"))
B = 10000
METRICS = {"2class_k2": "miou_tol_pct", "fg_k2": "iou_tol"}


def rid(cell, mode, city, seed):
    return f"mamnetr50-{cell}-{mode}-r450-{city}-hr-s{seed}"


def read_dump(root, run_id):
    d = Path(root) / "st" / run_id
    if not (d / ".DONE").exists():
        return None, None
    return (json.loads((d / "run.json").read_text()),
            pd.read_parquet(d / "perimage.parquet"))


def read_probe(root, loco_id, suffix):
    d = Path(root) / "st" / f"{loco_id}-{suffix}"
    if not (d / ".DONE").exists():
        return None
    return json.loads((d / "run.json").read_text())


def cluster_bootstrap(deltas, image_ids, seed=42, n_boot=B):
    """Mean of `deltas` with an image-clustered bootstrap CI and a two-tailed
    p against zero.

    Images are the cluster: the same image appears once per seed, and the three
    seeds' errors on it are not independent. Resampling image IDs with
    replacement and taking every row belonging to a drawn ID keeps that
    correlation inside the resample.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    ids = np.asarray(image_ids)
    uniq, inv = np.unique(ids, return_inverse=True)
    by_id = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    obs = float(deltas.mean())
    boot = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([by_id[k] for k in draw])
        boot[b] = deltas[idx].mean()
    boot.sort()
    lo, hi = float(boot[int(0.025 * n_boot)]), float(boot[int(0.975 * n_boot) - 1])
    centred = boot - boot.mean()
    p = float((1 + int((np.abs(centred) >= abs(obs)).sum())) / (1 + n_boot))
    return {"mean": obs, "ci95": [lo, hi], "p": p, "n": int(len(deltas)),
            "n_clusters": int(len(uniq)), "B": n_boot,
            "test": "paired bootstrap on per-image deltas, clustered by image, "
                    "two-tailed", "ci_method": f"image-cluster bootstrap B={n_boot}"}


def holm(pvals):
    """Holm-Bonferroni over a family. Returns adjusted p in input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_nn", default="15")
    ap.add_argument("--dumps_root", default=str(BASE / "dumps"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow_partial", action="store_true",
                    help="emit a results JSON over the cells that exist, "
                         "listing the missing ones, instead of hard-failing")
    args = ap.parse_args()

    conditions, missing, caveats = [], [], []
    per_image = {}          # (cell, city, seed) -> DataFrame of per-image gaps

    for cell, (pre, lr, lr_label) in CELLS.items():
        for city in CITIES:
            for seed in SEEDS:
                lid, uid = rid(cell, "loco", city, seed), rid(cell, "upper", city, seed)
                rl, dl = read_dump(args.dumps_root, lid)
                ru, du = read_dump(args.dumps_root, uid)
                if rl is None or ru is None:
                    missing.extend([x for x, r in ((lid, rl), (uid, ru)) if r is None])
                    continue

                m = du.set_index("image_id")
                dl = dl.set_index("image_id")
                shared = dl.index.intersection(m.index)
                if len(shared) != len(dl):
                    raise SystemExit(
                        f"FATAL {lid} and {uid} do not share every image_id "
                        f"({len(shared)} of {len(dl)})")
                dl, m = dl.loc[shared], m.loc[shared]

                row = {"cell": cell, "pretraining": pre, "lr": lr,
                       "lr_label": lr_label, "arch": "mamnet_r50",
                       "city": city, "seed": seed,
                       "arm": "L", "target_access": "source_only",
                       "weakness_ids": ["zNBu-W5", "zNBu-W1", "AC1", "AC2"],
                       "loco_run_id": lid, "upper_run_id": uid,
                       "n_train": rl["n_train"], "n_val": rl["n_val"],
                       "n_test": rl["n_test"], "n_images": int(len(dl)),
                       "ssl_checkpoint": rl.get("ssl_checkpoint"),
                       "ssl_checkpoint_sha256": rl.get("ssl_checkpoint_sha256"),
                       "best_epoch": rl["best_epoch"],
                       "steps_to_best": rl["steps_to_best"],
                       "early_stopped": rl["early_stopped"],
                       "pred_pos_fraction": rl["pred_pos_fraction"],
                       "determinism_max_delta": rl.get(
                           "determinism_max_abs_delta_per_image"),
                       "git_hash": rl.get("git_hash"),
                       "metric_module_sha": rl.get("metric_module_sha"),
                       "probe_script_sha": rl.get("probe_script_sha"),
                       "probes_common_sha": rl.get("probes_common_sha"),
                       "probes_ops_sha": rl.get("probes_ops_sha"),
                       "compute_note": rl.get("compute_note"),
                       "caveats": list(rl.get("sanity_flags", []))}

                gaps = {}
                for tag, col in METRICS.items():
                    lo_v = dl[col].to_numpy(dtype=np.float64)
                    up_v = m[col].to_numpy(dtype=np.float64)
                    d = lo_v - up_v
                    gaps[tag] = d
                    bs = cluster_bootstrap(d, dl.index.to_numpy(),
                                           seed=1000 + seed)
                    row[f"loco_miou_{tag}"] = float(lo_v.mean())
                    row[f"upper_miou_{tag}"] = float(up_v.mean())
                    row[f"gap_{tag}"] = float(d.mean())
                    row[f"gap_{tag}_armc_sign"] = float(-d.mean())
                    row[f"per_cell_gap_ci_95_{tag}"] = bs["ci95"]
                    row[f"p_paired_bootstrap_{tag}"] = bs["p"]
                per_image[(cell, city, seed)] = pd.DataFrame(
                    {"image_id": dl.index.to_numpy(),
                     "gap_2class_k2": gaps["2class_k2"],
                     "gap_fg_k2": gaps["fg_k2"]})

                # probes
                p1 = read_probe(args.dumps_root, lid, "p1histmatch")
                p2 = read_probe(args.dumps_root, lid, "p2cityprobe")
                p3 = read_probe(args.dumps_root, lid, "p3selpred")
                pa = read_probe(args.dumps_root, lid, "p3expA")
                pb = read_probe(args.dumps_root, lid, "p3expB")
                p5 = read_probe(args.dumps_root, lid, "p5drift")
                if p1:
                    g = p1["gray"]
                    row.update(histmatch_mean_R=g["mean_R"],
                               histmatch_median_R=g["median_R"],
                               histmatch_cells_improving=int(bool(g["improved"])),
                               histmatch_n_excluded=g["n_excluded"],
                               histmatch_rule=p1["matching_rules"]["headline"])
                if p2:
                    row.update(probe_city_acc=p2["probe"]["probe_accuracy"],
                               probe_null_mean=p2["probe"]["null_mean"],
                               probe_null_p99=p2["probe"]["null_p99"],
                               probe_rawpixel_acc=p2["raw_pixel_control"]["probe_accuracy"])
                if p3:
                    row.update(sp_aurc_fg_trapz=p3["aurc_trapz19"]["fg"],
                               sp_aurc_fg_mean=p3["aurc_mean20"]["fg"],
                               sp_aurc_bg_trapz=p3["aurc_trapz19"]["bg"],
                               sp_aurc_bg_mean=p3["aurc_mean20"]["bg"])
                if pa:
                    e = pa["expA"]
                    row.update(expA_median_R=e["median_R"], expA_R_ci95=e["R_ci95"],
                               expA_steps_to_best=e["steps_to_best"],
                               expA_budget_bound=e["budget_bound"])
                    if e["budget_bound"]:
                        row["caveats"].append(
                            "Exp A steps_to_best is within 10% of the 30-epoch "
                            "ceiling; its R is a LOWER BOUND")
                if pb:
                    e = pb["expB"]
                    row.update(expB_median_R=e["median_R"], expB_R_ci95=e["R_ci95"],
                               expB_encoder_lr_multiplier=e["exp_b_encoder_lr_multiplier"],
                               expB_steps_to_best=e["steps_to_best"],
                               expB_budget_bound=e["budget_bound"])
                if p5:
                    row.update(
                        drift_rel_l2_per_stage={
                            k: v["relative_l2"] if isinstance(v, dict) and
                            "relative_l2" in v else None
                            for k, v in p5["drift_rel_l2_per_stage"].items()},
                        cka_per_stage=p5["cka_per_stage"])
                for name, probe in (("p1", p1), ("p2", p2), ("p3", p3),
                                    ("p3expA", pa), ("p3expB", pb), ("p5", p5)):
                    if probe is None:
                        row["caveats"].append(f"probe {name} has no completed dump")
                conditions.append(row)

    if missing and not args.allow_partial:
        raise SystemExit(f"FATAL {len(missing)} of 72 cells have no completed "
                         f"dump; first: {missing[:5]}. Re-run with "
                         f"--allow_partial to emit over what exists.")
    if missing:
        caveats.append(f"{len(missing)} of 72 training cells were missing at "
                       f"assembly time: {missing}")

    # --- 2x2 aggregate, pooled over 3 cities x 3 seeds ----------------------
    aggregate, pooled = {}, {}
    for cell in CELLS:
        parts = [per_image[k] for k in per_image if k[0] == cell]
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True)
        pooled[cell] = d
        aggregate[cell] = {
            tag: cluster_bootstrap(d[f"gap_{tag}"].to_numpy(),
                                   d["image_id"].to_numpy(), seed=7)
            for tag in METRICS}
        aggregate[cell]["n_cells"] = len(parts)

    tests = []
    for tag in METRICS:
        raw, rows = [], []
        for a, b in CONTRASTS:
            if a not in pooled or b not in pooled:
                continue
            da = pooled[a].groupby("image_id")[f"gap_{tag}"].mean()
            db = pooled[b].groupby("image_id")[f"gap_{tag}"].mean()
            shared = da.index.intersection(db.index)
            delta = (da.loc[shared] - db.loc[shared]).to_numpy()
            bs = cluster_bootstrap(delta, shared.to_numpy(), seed=11)
            rows.append({"compares": [a, b], "metric": tag,
                         "statistic": bs["mean"], "ci95": bs["ci95"],
                         "p_raw": bs["p"], "n": bs["n"],
                         "n_clusters": bs["n_clusters"], "B": B,
                         "test": "paired bootstrap on per-image delta-of-delta "
                                 "(gap_a - gap_b), clustered by image, two-tailed",
                         "two_tailed": True})
            raw.append(bs["p"])
        for r, p in zip(rows, holm(np.array(raw)) if raw else []):
            r["p_holm"] = float(p)
            r["multiplicity"] = "Holm-Bonferroni over the 3 contrasts"
        tests.extend(rows)

    caveats += [
        "High-resolution cells only; a declared scope cut matching arm C.",
        "The encoder is ResNet-50, not the published ResNet-34, so sup_hi is "
        "the control against which the R34->R50 swap is read; it is not the "
        "published MAMNet number.",
        "Early stopping is flat 20 epochs (arm C's recipe), not Table 9's 10.",
        "The auxiliary heads attach one decoder stage later than the published "
        "R34 model's; see mamnet_r50.AUX_ATTACH_NOTE.",
        "gap_* follows CONVENTIONS.md (LOCO minus upper, negative when LOCO is "
        "worse). gap_*_armc_sign is its negation, matching arm C's tables.",
        "Assembled on the SLURM box, which CLAUDE.md reserves for per-image "
        "quantities. See this script's module docstring; the operator decides "
        "whether this file or its analysis-host equivalent is authoritative.",
    ]

    out = {
        "id": f"{args.exp_nn}_armL_ssl_regime",
        "arm": "L",
        "answers": ["zNBu-W5", "zNBu-W1", "AC1", "AC2"],
        "question": "Is the section 4.4 failure-locus finding about "
                    "architecture (ViT vs CNN) or about regime "
                    "(self-supervised pretraining plus low-LR fine-tuning)?",
        "runs": sorted({c["loco_run_id"] for c in conditions} |
                       {c["upper_run_id"] for c in conditions}),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "python/armL_results.py",
        "git_hash": conditions[0].get("git_hash") if conditions else None,
        "primary_metric": "two-class boundary-tolerant mIoU at K=2 "
                          "(the Table 38 family, comparable to arm C)",
        "secondary_metric": "foreground-only IoU at K=2 (Appendix E)",
        "conditions": conditions,
        "aggregate_2x2": aggregate,
        "tests": tests,
        "headline": None,
        "caveats": caveats,
        "figure": None,
        "caption": None,
    }
    path = Path(args.out) if args.out else \
        BASE / "results" / f"{args.exp_nn}_armL_ssl_regime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {path}  ({len(conditions)} of 36 cell-pairs, "
          f"{len(tests)} tests)")
    print("headline is null on purpose: it is one factual sentence and it is "
          "written by a human who has read the numbers.")


if __name__ == "__main__":
    main()
