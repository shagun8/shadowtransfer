"""
Arm E Part 2, revised — aggregate: per-image recovery ratio R, joined
against arm E Part 1's LOCO and upper per-image dumps for the same
city/images, median R per city with an image-cluster bootstrap 95% CI
(B=10,000).

READ-ONLY. Runs on the login node (pandas/JSON only, no torch, no GPU).
Uses `shadowtransfer.metrics.compute_per_image_recovery` and `.bootstrap_ci`
-- see the previous Part 2 aggregate script / metrics.py for the lifting
provenance of `bootstrap_ci` (added there, not duplicated here).

Also carries forward each cell's collapse-detection result (from run.json,
written during training against per-epoch validation Shadow_IoU) so a
reader does not have to open run.json separately to see whether a cell
degenerated the way the previous (reinitialized-decoder) attempt did.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer.metrics import compute_per_image_recovery, bootstrap_ci  # noqa: E402
from shadowtransfer.armc_dump import sha256_file                             # noqa: E402

CITIES = ("chicago", "miami", "phoenix")
MIN_GAP = 0.02
N_BOOT = 10000

# The previous (reinitialized-decoder) Part 2 attempt predates the
# collapse_flag logic added in this revision, so its own run.json carries no
# collapse fields. These are the manually-diagnosed collapse-start epochs
# from 2026-07-28-arme-part2.md's per-epoch log analysis (chicago epoch 13,
# miami epoch 11; phoenix never met the strict "3 consecutive epochs <5%"
# criterion despite sustained 5-8% degradation -- recorded as such, not
# rounded up to "collapsed").
PREVIOUS_REINIT_COLLAPSE = {
    "chicago": {"collapsed": True, "start_epoch": 13},
    "miami": {"collapsed": True, "start_epoch": 11},
    "phoenix": {"collapsed": False, "start_epoch": None,
               "note": "sustained 5-8% Shadow_IoU degradation without ever "
                       "meeting the strict 3-consecutive-epochs-under-5% rule"},
}


def load_iou(run_id):
    d = BASE / "dumps" / "st" / run_id
    if not (d / ".DONE").exists():
        return None
    df = pd.read_parquet(d / "perimage.parquet", columns=["image_id", "iou_tol"])
    return dict(zip(df["image_id"].astype(str), df["iou_tol"].astype(float)))


def load_run_json(run_id):
    p = BASE / "dumps" / "st" / run_id / "run.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    per_city = []
    for city in CITIES:
        loco_rid = f"dinov3frozen-vanilla-loco-r450-{city}-hr-s0"
        upper_rid = f"dinov3frozen-none-upper-r450-{city}-hr-s0"
        adapt_rid = f"dinov3frozen-adapt-adapt-r450-{city}-hr-s0"
        prev_expa_rid = f"dinov3frozen-expa-expa-r450-{city}-hr-s0"

        loco_iou, upper_iou, adapt_iou = (load_iou(r) for r in
                                          (loco_rid, upper_rid, adapt_rid))
        missing = [n for n, d in (("loco", loco_iou), ("upper", upper_iou),
                                  ("adapt", adapt_iou)) if d is None]
        if missing:
            per_city.append({"city": city, "done": False, "missing_dumps": missing})
            continue

        run = load_run_json(adapt_rid)

        image_ids = sorted(set(loco_iou) & set(upper_iou) & set(adapt_iou))
        union_ids = set(loco_iou) | set(upper_iou) | set(adapt_iou)
        not_joined = sorted(union_ids - set(image_ids))

        exp_ious = [adapt_iou[i] for i in image_ids]
        loco_ious = [loco_iou[i] for i in image_ids]
        upper_ious = [upper_iou[i] for i in image_ids]

        R, valid_idx, excluded = compute_per_image_recovery(
            exp_ious, loco_ious, upper_ious, min_gap=MIN_GAP)
        median_R, ci_lo, ci_hi = bootstrap_ci(R, n_boot=N_BOOT)

        prev = PREVIOUS_REINIT_COLLAPSE[city]

        per_city.append({
            "city": city, "done": True,
            "loco_run_id": loco_rid, "upper_run_id": upper_rid, "adapt_run_id": adapt_rid,
            "n_test_images_joined": len(image_ids),
            "n_not_joined_across_dumps": len(not_joined),
            "not_joined_image_ids": not_joined,
            "n_valid_R": excluded["valid_images"],
            "n_excluded_small_gap_or_nan": (excluded["excluded_nan"]
                                            + excluded["excluded_small_gap"]),
            "n_excluded_nan": excluded["excluded_nan"],
            "n_excluded_small_gap": excluded["excluded_small_gap"],
            "min_gap_threshold": excluded["min_gap_threshold"],
            "median_R": median_R, "R_ci95_lo": ci_lo, "R_ci95_hi": ci_hi,
            "R_bootstrap_n_boot": N_BOOT, "R_bootstrap_seed": 42,
            "collapse_detected": run.get("collapse_detected") if run else None,
            "collapse_start_epoch": run.get("collapse_start_epoch") if run else None,
            "epochs_run": run.get("epochs_run") if run else None,
            "decoder_initialized_from_part1_loco_epoch": (
                run.get("decoder_initialized_from_part1_loco_epoch") if run else None),
            "previous_reinit_attempt_run_id": prev_expa_rid,
            "previous_reinit_attempt_collapsed": prev["collapsed"],
            "previous_reinit_attempt_collapse_start_epoch": prev["start_epoch"],
            "previous_reinit_attempt_collapse_source": (
                "manually diagnosed from per-epoch logs in "
                "2026-07-28-arme-part2.md; that run predates collapse_flag"),
        })

    out = {
        "id": "arme_part2_revised_decoder_adaptation_recovery", "arm": "E", "part": "2-revised",
        "answers": ["zNBu-W1", "zNBu-W5", "AC1", "App G.9 Experiment A, adapted "
                    "for the frozen-backbone (Part 1) baselines: decoder LOADED "
                    "from Part 1's LOCO checkpoint, not reinitialized"],
        "question": ("Does fine-tuning arm E Part 1's own LOCO-trained decoder "
                    "on 25% of the holdout city's own data recover the "
                    "LOCO-vs-upper gap, when the encoder is fully frozen at "
                    "its pretrained state (so the encoder cannot be a "
                    "confound)?"),
        "runs": [row.get(k) for row in per_city
                for k in ("loco_run_id", "upper_run_id", "adapt_run_id") if row.get(k)],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "python/arme_part2_revised_aggregate.py",
        "git_hash": None, "git_hash_reason": "project root is not a git repository",
        "metric_module_sha256": sha256_file(BASE / "shadowtransfer" / "metrics.py"),
        "recovery_ratio_definition": "R_i = (IoU_adapted_i - IoU_LOCO_i) / (IoU_upper_i - IoU_LOCO_i)",
        "min_gap_exclusion": MIN_GAP,
        "bootstrap_method": ("image-level resampling with replacement, "
                            "stat_fn=np.median, B=10000, seed=42 -- "
                            "shadowtransfer.metrics.bootstrap_ci"),
        "divergence_from_published_experiment_a": (
            "App G.9's published Experiment A reinitializes the decoder. This "
            "run loads arm E Part 1's LOCO-trained decoder instead. Reason: "
            "the published version reinitializes atop an encoder that WAS "
            "LOCO-trained on 450 source images, so a task-adapted component "
            "survives reinitialization there; under full backbone freezing "
            "(arm E's premise) the encoder is never LOCO-trained, so "
            "reinitializing the decoder would leave no task-adapted component "
            "anywhere in the model. The previous attempt did exactly that and "
            "collapsed to majority-class background prediction in all 3 "
            "cities (run_ids dinov3frozen-expa-expa-r450-*-hr-s0, "
            "2026-07-28-arme-part2.md); that was a spec error, not a finding."),
        "encoder_confound_note": (
            "Under full backbone freezing, the encoder is byte-identical, by "
            "construction, across the LOCO, upper, and this decoder-adaptation "
            "cell for a given city -- none of the three ever trains it. Any "
            "recovery measured here is attributable to the decoder alone."),
        "published_reference": (
            "App G.9 / Table 36 reports R POOLED ACROSS THE THREE CITIES for "
            "DINOv3 (vanilla, full fine-tune, reinitialized decoder atop a "
            "LOCO-trained encoder): median R=0.69 [0.60, 0.75], n=335-351. "
            "This JSON reports PER-CITY medians for a materially different "
            "condition -- frozen-pretrained encoder, decoder ADAPTED (not "
            "reinitialized) from Part 1's own LOCO checkpoint -- not a "
            "reproduction of that published number."),
        "per_city": per_city,
    }

    out_dir = BASE / "runs" / "arme_part2_revised"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"WROTE {out_path}")
    for row in per_city:
        if not row.get("done"):
            print(f"  {row['city']:8s} NOT DONE, missing: {row['missing_dumps']}")
            continue
        print(f"  {row['city']:8s} n_valid={row['n_valid_R']:3d}  "
              f"median_R={row['median_R']:+.3f}  "
              f"CI95=[{row['R_ci95_lo']:+.3f}, {row['R_ci95_hi']:+.3f}]  "
              f"collapse={row['collapse_detected']}")


if __name__ == "__main__":
    main()
