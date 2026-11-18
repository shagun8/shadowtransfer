#!/usr/bin/env python
"""
Arm D / D_S3_diag — assemble runs/D_S3/diagnostics.json from the per-group
probe results.

This is a CONCATENATION, not an analysis. Every number in the output was
computed inside the group job that owns it. Nothing here pools across cells,
computes a cross-cell mean, ranks anything, or correlates a probe against a
city appearance statistic -- the last is arm D-R and is not specced.

Two gates before it writes:

  1  runs/D_S3/decision_phoenix_replicated.json.part_b_aggregation_blocked.
     The arm spec: if a replicated e200 cell's value differs from its
     100-epoch value by more than that cell's gap_sd_ref, report it and stop
     before this aggregation. Overridable only with --ignore_block, which is
     recorded in the output.
  2  every group named by --require must be present, unless --allow_partial.

CPU-only. No model is loaded, no image is read.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armd_diag_lib as L                    # noqa: E402
from shadowtransfer.armc_dump import sha256_file                 # noqa: E402

RESULTS_DIR = BASE / "runs" / "D_S3" / "diag"
OUT_PATH = BASE / "runs" / "D_S3" / "diagnostics.json"
DECISION_PATH = BASE / "runs" / "D_S3" / "decision_phoenix_replicated.json"


def check_block_gate(ignore):
    """Part A's gate. Returns the block record; raises unless ignored."""
    rec = {"decision_file": str(DECISION_PATH.relative_to(BASE)),
           "present": DECISION_PATH.exists(),
           "blocked": None, "exceeding_cells": None, "ignored": bool(ignore)}
    if not DECISION_PATH.exists():
        rec["note"] = ("Part A's decision recompute has not run. The gate "
                       "cannot be evaluated; recorded as not-evaluated rather "
                       "than as passed.")
        print(f"  GATE: {rec['note']}")
        return rec
    d = json.loads(DECISION_PATH.read_text())
    rec["blocked"] = bool(d.get("part_b_aggregation_blocked"))
    rec["exceeding_cells"] = d.get("prior_ceiling_exceeds_gap_sd_ref")
    rec["prior_ceiling_comparisons"] = d.get("prior_ceiling_comparisons")
    rec["decision_file_sha256"] = sha256_file(DECISION_PATH)
    if rec["blocked"] and not ignore:
        raise SystemExit(
            "FATAL Part A reports part_b_aggregation_blocked=true for "
            f"{rec['exceeding_cells']}: an e200 cell's value differs from its "
            "100-epoch value by more than that cell's gap_sd_ref. The arm "
            "spec says report this and stop before the Part B aggregation. "
            "Re-run with --ignore_block only on the operator's instruction.")
    if rec["blocked"]:
        print("  GATE: BLOCKED but --ignore_block set; recorded in the output.")
    else:
        print("  GATE: passed (part_b_aggregation_blocked=false)")
    return rec


def aggregate(args):
    ignore = bool(getattr(args, "ignore_block", False))
    allow_partial = bool(getattr(args, "allow_partial", False))
    src_dir = Path(getattr(args, "out_root", None) or RESULTS_DIR)

    print(f"ARM D / {L.ARM_ID} aggregate")
    print(f"  reading {src_dir}")
    gate = check_block_gate(ignore)

    expected = {f"{a}-{c}-s{s}": modes
                for (a, c, s), modes in L.discover_groups().items()}
    found, unreadable = {}, []
    for p in sorted(src_dir.glob("*.json")):
        try:
            found[p.stem] = json.loads(p.read_text())
        except Exception as e:                                  # noqa: BLE001
            unreadable.append({"file": str(p), "error": str(e)})
    missing = sorted(set(expected) - set(found))

    print(f"  groups discoverable from dumps/: {len(expected)}")
    print(f"  group result files found:        {len(found)}")
    print(f"  missing:                         {missing or 'none'}")
    if unreadable:
        print(f"  UNREADABLE: {unreadable}")
    if missing and not allow_partial:
        raise SystemExit(f"FATAL {len(missing)} group(s) missing and "
                         f"--allow_partial not set: {missing}")

    by_probe = {"city": {}, "sp": {}, "histmatch": {}}
    for gid, rec in sorted(found.items()):
        for probe in by_probe:
            blk = (rec.get("probes") or {}).get(probe)
            if blk is None:
                continue
            by_probe[probe][gid] = {
                "group_id": gid,
                "arch": rec.get("arch"), "city": rec.get("city"),
                "seed": rec.get("seed"), "rung": rec.get("rung"),
                "res": rec.get("res"),
                "model_variant": rec.get("model_variant"),
                "cells": rec.get("cells"),
                "metric_name": rec.get("metric_name"),
                "metrics_sha256": rec.get("metrics_sha256"),
                "slurm_job_id": rec.get("slurm_job_id"),
                "wall_clock_s": (rec.get("probe_wall_clock_s") or {}).get(probe),
                "block": blk,
            }

    n_cells = sum(1 for r in found.values()
                  for k in ("aug_loco", "aug_upper")
                  if (r.get("cells") or {}).get(k))

    out = {
        "id": "D_S3_diagnostics",
        "arm": L.ARM, "arm_id": L.ARM_ID, "weakness_ids": L.WEAKNESS_IDS,
        "question": ("Did the S3 appearance augmentation change what the "
                     "encoder encodes, how the model's confidence ranks "
                     "foreground errors, and how much it depends on pixel "
                     "statistics?"),
        "rung": L.RUNG, "res": L.RES,
        "metric_name": L.METRIC_NAME,
        "secondary_metric_name": L.SECONDARY_METRIC_NAME,
        "metrics_sha256": sha256_file(BASE / "shadowtransfer/metrics.py"),
        "metrics_sha256_expected": "730ad4a5",
        "reporting_metric": "fg_only_tol_k2",
        "band_width_headline": 2,
        "band_widths_written": [0, 2, 5],

        "part_a_gate": gate,
        "groups_expected": expected,
        "groups_found": sorted(found),
        "groups_missing": missing,
        "groups_unreadable": unreadable,
        "partial": bool(missing),
        "n_groups": len(found),
        "n_aug_cells_covered": n_cells,
        "n_aug_cells_note": (
            "the arm spec names 30 aug cells. The set discovered on disk is "
            "what is reported here; a difference between the two is a finding "
            "for the operator, not something this job reconciles."),

        "probes": by_probe,
        "probe_order": ["city", "sp", "histmatch"],
        "probe_names": {
            "city": "encoder city-identity linear probe (App G.8)",
            "sp": "foreground selective-prediction gap / AURC",
            "histmatch": "histogram-matching recovery ratio R (s4.2 / App G.3)",
        },
        "probe_protocol_deviations": L.PROBE_PROTOCOL_DEVIATIONS,
        "aurc_caveat": L.AURC_CAVEAT,

        "caveats": [
            L.AURC_CAVEAT,
            "The city probe's null mean is MEASURED per model and is never "
            "assumed to sit at 1/3. A previous run on this project measured "
            "0.363-0.381 and that discrepancy is unresolved; the arm C App "
            "G.8 cells measured 0.329-0.333.",
            "REVIEWS.md erratum E-8: the published App G.8 probe tapped the "
            "wrong encoder tensor on all three architectures and leaked its "
            "StandardScaler across folds. This run uses the corrected arm C "
            "path, so its accuracies are NOT comparable to the published "
            "0.33 / 0.43 / 0.45. See probe_protocol_deviations.",
            "No number here is compared against any value printed in the PDF "
            "or in Other_Analysis.md.",
            "No pooled cross-cell statistic, no ranking, and no correlation "
            "against city appearance statistics is computed. The last is arm "
            "D-R and is not specced.",
            "No figure is built. CONVENTIONS.md puts figures on the SSH box.",
            "This file is a concatenation of per-group results. Every "
            "interval in it was computed inside the group job that owns it, "
            "over that group's own 150 held-out-city test images.",
            "Image-level bootstrap intervals are computed on this box, which "
            "CLAUDE.md normally reserves for the SSH box. The arm spec "
            "requires B=1000 paired intervals inside the probe job; flagged "
            "rather than resolved silently.",
            "aug-upper and vanilla-upper cells are target_labels runs. This "
            "job only reads their stored per-image numbers; it trains "
            "nothing.",
            "REVIEWS.md carries no pre-committed reading for arm D. "
            "CONVENTIONS.md requires one before the numbers are read.",
        ],

        "bootstrap_B": L.BOOTSTRAP_B,
        "diag_lib_sha": sha256_file(BASE / "shadowtransfer/armd_diag_lib.py"),
        "aggregate_script_sha": sha256_file(Path(__file__).resolve()),
        "diag_script_sha": sha256_file(BASE / "support/armd_diag.py"),
        "compute_note": ("CPU-only concatenation submitted on gpuA100x4 with "
                         "the GPU idle; there is no CPU account on this "
                         "allocation."),
        "cluster": "delta",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_hash": None,
        "git_hash_reason": "left null for arm D artifacts, matching arm C",
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    if getattr(args, "dry_run", False):
        print("--dry_run: nothing written.")
        print(json.dumps({k: out[k] for k in
                          ("n_groups", "groups_missing", "partial",
                           "n_aug_cells_covered")}, indent=2))
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"-> {OUT_PATH}")
    print(f"   groups {len(found)}  aug cells covered {n_cells}  "
          f"partial {out['partial']}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--allow_partial", action="store_true")
    ap.add_argument("--ignore_block", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    aggregate(ap.parse_args())


if __name__ == "__main__":
    main()
