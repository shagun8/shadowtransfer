"""
Arm E Part 2 — aggregate: per-image recovery ratio R, joined against arm E
Part 1's LOCO and upper per-image dumps for the same city/images, median R
per city with an image-cluster bootstrap 95% CI (B=10,000).

READ-ONLY. Runs on the login node (pandas/JSON only, no torch, no GPU).
Uses `shadowtransfer.metrics.compute_per_image_recovery` and `.bootstrap_ci`
-- the only implementations of these quantities for new work (CLAUDE.md:
"Import the metric module... never reimplement inline"). `bootstrap_ci` was
lifted into metrics.py for this task from
python/final_loco/evaluate_recovery_robust.py (same file CLAUDE.md already
cites for compute_per_image_recovery); see that function's docstring in
metrics.py for the lifting note.

Unlike arme_part1_aggregate.py's plain point-estimate difference, THIS is a
properly-powered statistical claim (median + bootstrap CI over n=100-150
per-image R values per city) and is reported as one, per CONVENTIONS.md.
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


def load_iou(run_id):
    d = BASE / "dumps" / "st" / run_id
    if not (d / ".DONE").exists():
        return None
    df = pd.read_parquet(d / "perimage.parquet", columns=["image_id", "iou_tol"])
    return dict(zip(df["image_id"].astype(str), df["iou_tol"].astype(float)))


def main():
    per_city = []
    for city in CITIES:
        loco_rid = f"dinov3frozen-vanilla-loco-r450-{city}-hr-s0"
        upper_rid = f"dinov3frozen-none-upper-r450-{city}-hr-s0"
        expa_rid = f"dinov3frozen-expa-expa-r450-{city}-hr-s0"

        loco_iou, upper_iou, expa_iou = (load_iou(r) for r in (loco_rid, upper_rid, expa_rid))
        missing = [n for n, d in (("loco", loco_iou), ("upper", upper_iou), ("expa", expa_iou))
                  if d is None]
        if missing:
            per_city.append({"city": city, "done": False, "missing_dumps": missing})
            continue

        image_ids = sorted(set(loco_iou) & set(upper_iou) & set(expa_iou))
        union_ids = set(loco_iou) | set(upper_iou) | set(expa_iou)
        not_joined = sorted(union_ids - set(image_ids))

        exp_ious = [expa_iou[i] for i in image_ids]
        loco_ious = [loco_iou[i] for i in image_ids]
        upper_ious = [upper_iou[i] for i in image_ids]

        R, valid_idx, excluded = compute_per_image_recovery(
            exp_ious, loco_ious, upper_ious, min_gap=MIN_GAP)
        median_R, ci_lo, ci_hi = bootstrap_ci(R, n_boot=N_BOOT)

        per_city.append({
            "city": city, "done": True,
            "loco_run_id": loco_rid, "upper_run_id": upper_rid, "expa_run_id": expa_rid,
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
        })

    out = {
        "id": "arme_part2_experiment_a_recovery", "arm": "E", "part": 2,
        "answers": ["zNBu-W1", "zNBu-W5", "AC1", "App G.9 Experiment A, "
                    "re-run against the frozen-backbone (Part 1) baselines"],
        "question": ("Does retraining only the decoder on 25% of the holdout "
                    "city's own data recover the LOCO-vs-upper gap, when the "
                    "encoder is fully frozen at its pretrained state (so the "
                    "encoder cannot be a confound)?"),
        "runs": [row.get(k) for row in per_city
                for k in ("loco_run_id", "upper_run_id", "expa_run_id") if row.get(k)],
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "appendix_h/arme_part2_aggregate.py",
        "git_hash": None, "git_hash_reason": "project root is not a git repository",
        "metric_module_sha256": sha256_file(BASE / "shadowtransfer" / "metrics.py"),
        "recovery_ratio_definition": "R_i = (IoU_expA_i - IoU_LOCO_i) / (IoU_upper_i - IoU_LOCO_i)",
        "min_gap_exclusion": MIN_GAP,
        "bootstrap_method": ("image-level resampling with replacement, "
                            "stat_fn=np.median, B=10000, seed=42 -- "
                            "shadowtransfer.metrics.bootstrap_ci, lifted from "
                            "python/final_loco/evaluate_recovery_robust.py"),
        "encoder_confound_note": (
            "Under full backbone freezing, the encoder is byte-identical, by "
            "construction, across the LOCO, upper, and Experiment-A cells for "
            "a given city -- none of the three ever trains it. Any recovery "
            "measured here is attributable to the decoder alone."),
        "published_reference": (
            "App G.9 / Table 36 reports R POOLED ACROSS THE THREE CITIES for "
            "DINOv3 (vanilla, full fine-tune, not frozen-backbone): "
            "median R=0.69 [0.60, 0.75], n=335-351 valid images. This JSON "
            "reports PER-CITY medians for the FROZEN-backbone condition "
            "instead -- a finer breakdown of a different (frozen, not "
            "vanilla) baseline, not a reproduction of that published number."),
        "per_city": per_city,
    }

    out_dir = BASE / "runs" / "arme_part2"
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
              f"CI95=[{row['R_ci95_lo']:+.3f}, {row['R_ci95_hi']:+.3f}]")


if __name__ == "__main__":
    main()
