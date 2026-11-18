"""Arm D / D_S3 — variance-tiered grid support.

Everything the 25-run S3 grid needs that arm D's single-cell script did not:

  * the SEED TIER table — how many seeds each (arch, city, condition) gets,
    set from the D_seedfloor across-seed gap sd rather than flat at one seed;
  * gap_sd_ref — that cell's D_seedfloor gap sd, in 2-class tolerant mIoU K=2;
  * the arm C rung-450 vanilla BASELINES in the primary metric, per seed, with
    the vanilla gap per seed, its across-seed mean and its sd;
  * the DECISION RULE block, pre-committed here before any arm D number is
    read, and written into every cell.

The primary metric for all of the above is 2-class boundary-tolerant mIoU at
K=2, which is the metric D_seedfloor measured the noise floor in. Arm D also
reports foreground-only IoU at K=2 per CLAUDE.md; both travel together.

The gap sds are NOT recomputed here. They are read from
runs/D_seedfloor/results.json, the gate job that produced them, so the number
the seed tiers were chosen from and the number the decision rule is tested
against cannot drift apart. read_baselines_2class independently recomputes the
same sd from the arm C dumps and the two are cross-checked; a mismatch is a
sanity flag, not a silent correction.
"""

import json
import statistics
from pathlib import Path

SEEDFLOOR_REL = "runs/D_seedfloor/results.json"
PRIMARY_METRIC = "miou_2class_tolerant_K2"
SECONDARY_METRIC = "iou_fg_tolerant_K2"
RUNG = 450
BASELINE_SEEDS = (0, 1, 2)

# run.json scalar carrying the primary metric on an arm C cell.
_PRIMARY_KEY = "miou_tol_pct_published_definition"
_SECONDARY_KEY = "headline_fg_iou_tol_pct"

# ---------------------------------------------------------------------------
# Seed tiers. Operator-fixed for this grid on 2026-07-30; the thresholds below
# are the rule those counts came from, asserted against the table so a future
# edit to either one fails loudly instead of quietly disagreeing.
#
#   aug-loco  seeds by the cell's LOCO-vs-upper GAP sd:   <1.0 -> 1, <2.0 -> 2, else 3
#   aug-upper seeds by the cell's UPPER-bound sd:         >1.0 -> 2, else 1
# ---------------------------------------------------------------------------
SEED_TIER = {
    ("mamnet",  "chicago"): {"aug-loco": 1, "aug-upper": 1},
    ("mamnet",  "miami"):   {"aug-loco": 2, "aug-upper": 1},
    ("mamnet",  "phoenix"): {"aug-loco": 3, "aug-upper": 1},
    ("oglanet", "chicago"): {"aug-loco": 3, "aug-upper": 2},
    ("oglanet", "miami"):   {"aug-loco": 1, "aug-upper": 1},
    ("oglanet", "phoenix"): {"aug-loco": 1, "aug-upper": 1},
    ("dinov3",  "chicago"): {"aug-loco": 1, "aug-upper": 1},
    ("dinov3",  "miami"):   {"aug-loco": 1, "aug-upper": 1},
    ("dinov3",  "phoenix"): {"aug-loco": 2, "aug-upper": 1},
}

DECISION_RULE_TEXT = (
    "|gap_aug_mean - gap_vanilla_mean| > 2 * gap_sd_ref")

GAP_DEFINITION = (
    "LOCO across-seed mean minus within-city-upper across-seed mean, same "
    "architecture, same held-out city, same test pool, 2-class boundary-"
    "tolerant mIoU at K=2. Each side is averaged over the seeds its tier "
    "allots, which for the vanilla baseline (3 seeds both sides) is identical "
    "to averaging the per-seed gaps.")


def _tier_from_sds(gap_sd, upper_sd):
    loco = 1 if gap_sd < 1.0 else (2 if gap_sd < 2.0 else 3)
    return {"aug-loco": loco, "aug-upper": 2 if upper_sd > 1.0 else 1}


def load_seedfloor(base):
    """Read the D_seedfloor gate results. Hard-fails if absent."""
    p = Path(base) / SEEDFLOOR_REL
    if not p.exists():
        raise SystemExit(
            f"FATAL arm D S3 needs the seed-floor gate at {p}; it is the only "
            f"source of gap_sd_ref and the seed tiers. Run D_seedfloor first.")
    d = json.loads(p.read_text())
    out = {}
    for c in d.get("cells", []):
        out[(c["arch"], c["city"])] = c
    if not out:
        raise SystemExit(f"FATAL {p} carries no cells")
    return {"path": str(p), "metrics_sha256": d.get("metrics_sha256"),
            "written_utc": d.get("written_utc"), "cells": out}


def seedfloor_refs(base, arch, city):
    """gap_sd_ref and the supporting spreads for one (arch, city), 2-class K=2."""
    sf = load_seedfloor(base)
    key = (arch, city)
    if key not in sf["cells"]:
        raise SystemExit(f"FATAL no D_seedfloor cell for {arch}/{city}")
    c = sf["cells"][key]
    g = c["gap"]["spread_miou_2class_tol_K2"]
    u = c["upper"]["spread_miou_2class_tol_K2"]
    l = c["loco"]["spread_miou_2class_tol_K2"]
    return {
        "gap_sd_ref": round(float(g["std_sample"]), 6),
        "gap_vanilla_mean_seedfloor": round(float(g["mean"]), 6),
        "vanilla_upper_sd_seedfloor": round(float(u["std_sample"]), 6),
        "vanilla_loco_sd_seedfloor": round(float(l["std_sample"]), 6),
        "n_seeds": int(g["n"]),
        "metric": PRIMARY_METRIC,
        "source": (f"{SEEDFLOOR_REL} cells[{arch}/{city}]"
                   ".gap.spread_miou_2class_tol_K2.std_sample"),
        "seedfloor_path": sf["path"],
        "seedfloor_metrics_sha256": sf["metrics_sha256"],
        "seedfloor_written_utc": sf["written_utc"],
    }


def seed_tier(base, arch, city, condition=None):
    """Planned seed COUNT for a cell, with the table checked against the rule."""
    key = (arch, city)
    if key not in SEED_TIER:
        raise SystemExit(f"FATAL no seed tier for {arch}/{city}")
    refs = seedfloor_refs(base, arch, city)
    implied = _tier_from_sds(refs["gap_sd_ref"], refs["vanilla_upper_sd_seedfloor"])
    if implied != SEED_TIER[key]:
        raise SystemExit(
            f"FATAL seed tier table disagrees with the D_seedfloor sds for "
            f"{arch}/{city}: table {SEED_TIER[key]}, rule implies {implied} "
            f"(gap sd {refs['gap_sd_ref']}, upper sd "
            f"{refs['vanilla_upper_sd_seedfloor']}). Reconcile before running.")
    tier = dict(SEED_TIER[key])
    return tier if condition is None else tier[condition]


def planned_seeds(base, arch, city, condition):
    """Seed VALUES for a cell: 0, then 1, then 2 — arm C's seed values."""
    return list(range(seed_tier(base, arch, city, condition)))


def baseline_run_id(arch, kind, city, res, seed):
    """Arm C rung-450 vanilla cell ids. The upper cells carry method token
    `none`, not `vanilla`; getting that wrong resolves to a missing directory,
    so it lives in exactly one function."""
    tok = "vanilla-loco" if kind == "loco" else "none-upper"
    return f"{arch}-{tok}-r{RUNG}-{city}-{res}-s{seed}"


def read_baselines_2class(base, arch, city, res="hr"):
    """Arm C rung-450 vanilla baselines in the primary metric.

    Per-seed LOCO, per-seed upper, their across-seed means, the vanilla gap per
    seed, its across-seed mean and its sd — all read from the arm C dumps'
    own run.json scalars, not recomputed from parquet and not taken from any
    number printed in the PDF. `missing` names any cell that did not resolve;
    the caller stops on a non-empty list.
    """
    base = Path(base)
    out = {"metric": PRIMARY_METRIC,
           "secondary_metric": SECONDARY_METRIC,
           "source": (f"arm C rung-{RUNG} vanilla dumps, run.json scalars "
                      f"{_PRIMARY_KEY} (primary) and {_SECONDARY_KEY}"),
           "missing": []}
    per = {"loco": {}, "upper": {}}
    per2 = {"loco": {}, "upper": {}}
    for kind in ("loco", "upper"):
        for s in BASELINE_SEEDS:
            rid = baseline_run_id(arch, kind, city, res, s)
            d = base / "dumps" / "st" / rid
            if not (d / ".DONE").exists() or not (d / "run.json").exists():
                out["missing"].append(rid)
                continue
            try:
                rj = json.loads((d / "run.json").read_text())
            except Exception as e:                              # noqa: BLE001
                out["missing"].append(f"{rid} (unreadable: {e})")
                continue
            v = rj.get(_PRIMARY_KEY)
            if v is None:
                out["missing"].append(f"{rid} (no {_PRIMARY_KEY})")
                continue
            per[kind][s] = float(v)
            if rj.get(_SECONDARY_KEY) is not None:
                per2[kind][s] = float(rj[_SECONDARY_KEY])
        out[f"vanilla_{kind}_run_ids"] = [
            baseline_run_id(arch, kind, city, res, s) for s in BASELINE_SEEDS]
        out[f"vanilla_{kind}_per_seed"] = {f"s{s}": round(v, 6)
                                           for s, v in sorted(per[kind].items())}
        out[f"vanilla_{kind}_mean"] = (
            round(statistics.fmean(per[kind].values()), 6) if per[kind] else None)
        out[f"vanilla_{kind}_per_seed_fg"] = {f"s{s}": round(v, 6)
                                              for s, v in sorted(per2[kind].items())}

    paired = sorted(set(per["loco"]) & set(per["upper"]))
    gaps = {s: per["loco"][s] - per["upper"][s] for s in paired}
    out["vanilla_gap_per_seed"] = {f"s{s}": round(v, 6) for s, v in gaps.items()}
    out["vanilla_gap_seeds_paired"] = paired
    out["vanilla_gap_mean"] = (round(statistics.fmean(gaps.values()), 6)
                               if gaps else None)
    out["vanilla_gap_sd"] = (round(statistics.stdev(gaps.values()), 6)
                             if len(gaps) >= 2 else None)
    out["gap_definition"] = GAP_DEFINITION
    out["note"] = (
        "the noise floor arm D is read against. These are armc_train rung-450 "
        "cells — same seeded nested-prefix selection rule, same early-stopping "
        "recipe, same dump code path this arm uses — so aug vs vanilla at a "
        "fixed seed differs by the S3 augmentation and nothing else.")
    return out


def decision_rule_block(base, arch, city, condition, baselines):
    """The pre-committed decision rule, fixed before any arm D number is read.

    Everything computable from the vanilla side is filled here, at cell time.
    gap_aug_mean needs every seed of BOTH aug conditions for this (arch, city)
    and therefore cannot exist until the grid finishes; armd_s3_aggregate.py
    fills it and flips `changed`. `status` says which of the two states the
    block is in, so a half-filled rule can never be mistaken for a verdict.
    """
    refs = seedfloor_refs(base, arch, city)
    sd = refs["gap_sd_ref"]
    gv = baselines.get("vanilla_gap_mean")
    return {
        "rule": DECISION_RULE_TEXT,
        "rule_prose": (
            "An augmented cell's gap counts as CHANGED only if the absolute "
            "difference between its across-seed mean gap and the vanilla "
            "across-seed mean gap exceeds twice that cell's D_seedfloor "
            "across-seed gap sd. Fixed in the arm spec before the grid ran."),
        "metric": PRIMARY_METRIC,
        "gap_definition": GAP_DEFINITION,
        "gap_vanilla_mean": gv,
        "gap_sd_ref": sd,
        "gap_sd_ref_source": refs["source"],
        "threshold": round(2.0 * sd, 6),
        "gap_aug_mean": None,
        "abs_delta": None,
        "changed": None,
        "status": "pending_grid_completion",
        "filled_by": "python/armd_s3_aggregate.py",
        "seed_tier": seed_tier(base, arch, city),
        "this_cell_condition": condition,
        "seedfloor_gap_mean_crosscheck": {
            "from_seedfloor": refs["gap_vanilla_mean_seedfloor"],
            "from_arm_c_dumps": gv,
            "agree_1e4": (gv is not None and
                          abs(gv - refs["gap_vanilla_mean_seedfloor"]) < 1e-4),
        },
        "seedfloor_gap_sd_crosscheck": {
            "from_seedfloor": sd,
            "from_arm_c_dumps": baselines.get("vanilla_gap_sd"),
            "agree_1e4": (baselines.get("vanilla_gap_sd") is not None and
                          abs(baselines["vanilla_gap_sd"] - sd) < 1e-4),
        },
        "note": ("computed and recorded, not interpreted. Whether a CHANGED "
                 "cell weakens section 4.2 is the analyst's call on the SSH box."),
    }
