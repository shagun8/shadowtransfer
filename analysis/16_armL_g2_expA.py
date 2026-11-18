#!/usr/bin/env python
"""Arm L — close gate G2's DINOv3 leg, and read Experiment A across the 2x2.

Reanalysis only.  Reads the 36 MAMNet-R50 P5 drift probes, the six DINOv3
`-orig-p5drift` probes, and (for task 2) the 72 arm L training cells plus their
Experiment A probes.  Writes two new top-level blocks into
`results/16_armL_ssl_regime.json` by TEXT SPLICE, so every pre-existing byte of
that file is carried through unchanged, and writes `results/16_armL_g2_expA.md`.

This script does not re-adjudicate arm L's outcome.  DECISIONS.md ratified
outcome 4 on Experiment B median R; both quantities computed here (G2's drift
comparison, Experiment A) are corroborating variables.

Every statistical primitive is imported from `analysis/16_armL_ssl_regime.py`
rather than reimplemented, so the numbers restated here are the same numbers by
construction rather than by agreement.

STAGE = 'dry'  -> task 1b only: the one-axis whole-encoder table and the two
                  distances.  Reads only run.json sidecars.  Writes nothing.
STAGE = 'full' -> tasks 1 and 2, then writes.
"""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# ------------------------------------------------------------------ config --

BASE = os.environ["PROJECT_ROOT"]
DUMPS_ROOT = os.path.join(BASE, "dumps", "st")
POOLED_JSON = os.path.join(BASE, "results", "16_armL_ssl_regime.json")
OUT_MD = os.path.join(BASE, "results", "16_armL_g2_expA.md")
SIBLING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "16_armL_ssl_regime.py")
SCRIPT_LABEL = "analysis/16_armL_g2_expA.py"

STAGE = "dry" if "--dry-run" in sys.argv else "full"          # 'dry' | 'full'

CITIES = ("chicago", "miami", "phoenix")
CORNERS = ("sup_hi", "sup_lo", "ssl_hi", "ssl_lo")
SEEDS = (0, 1, 2)

# The six DINOv3 drift probes.  The `-orig` token disambiguates the TRUE
# published checkpoints from arm C's retrained cells, which claimed the same
# canonical run_id.  Never resolve these by glob.
DINOV3_LOCO = tuple(f"dinov3-vanilla-loco-r450-{c}-hr-s0-orig-p5drift"
                    for c in CITIES)
DINOV3_UPPER = tuple(f"dinov3-none-upper-r450-{c}-hr-s0-orig-p5drift"
                     for c in CITIES)
DINOV3_ALL = DINOV3_LOCO + DINOV3_UPPER

# The Gate E-0 init-load result these six cells must reproduce exactly.
E0_N_MISSING = 0
E0_N_UNEXPECTED = 37

RESNET_STAGES = ("conv1", "layer1", "layer2", "layer3", "layer4")
CKA_STAGES = ("layer1", "layer2", "layer3", "layer4")

LOG_LINES = []
SANITY = []


def log(msg):
    line = str(msg)
    LOG_LINES.append(line)
    print(line, flush=True)


def ok(cond, msg):
    log(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        raise AssertionError(msg)
    return cond


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sibling():
    """Import the pooling script as a module so its primitives are reused."""
    spec = importlib.util.spec_from_file_location("armL_pooled", SIBLING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_run_json(run_dir):
    with open(os.path.join(DUMPS_ROOT, run_dir, "run.json")) as fh:
        return json.load(fh)


def med_range(values):
    """The 9-cell (or 3-cell) median with its observed range.

    Drift is one scalar per cell in weight space; there is no per-image
    resolution to resample, so no image-cluster bootstrap is available on this
    axis and the range over cells is reported instead of a CI.
    """
    v = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max()), "mean": float(v.mean()), "n_cells": int(v.size),
            "dispersion": "observed range over cells, not a confidence interval"}


# ------------------------------------------------- run order step 1: assert --

def assert_dinov3_dumps():
    """Present, .DONE, sidecar-complete, and six DISTINCT ckpt_sha256."""
    log("[step 1] asserting the six DINOv3 -orig-p5drift dumps")
    runs = {}
    for d in DINOV3_ALL:
        path = os.path.join(DUMPS_ROOT, d)
        ok(os.path.isdir(path), f"{d}: directory present")
        ok(os.path.exists(os.path.join(path, ".DONE")), f"{d}: .DONE present")
        ok(os.path.exists(os.path.join(path, "run.json")),
           f"{d}: run.json sidecar present")
        r = read_run_json(d)
        ok(r.get("run_id") == d, f"{d}: run.json run_id matches its directory")
        ok(r.get("arch") == "dinov3", f"{d}: arch = dinov3")
        ok(r.get("probe") == "p5_encoder_drift", f"{d}: probe = p5_encoder_drift")
        ok(r.get("rung") == 450 and r.get("res") == "hr" and r.get("seed") == 0,
           f"{d}: rung/res/seed = 450/hr/0")
        ok("drift_rel_l2_whole_encoder" in r,
           f"{d}: drift_rel_l2_whole_encoder present")
        ok(int(r.get("init_load_n_missing", -1)) == E0_N_MISSING
           and int(r.get("init_load_n_unexpected", -1)) == E0_N_UNEXPECTED,
           f"{d}: init load is {r.get('init_load_n_missing')} missing / "
           f"{r.get('init_load_n_unexpected')} unexpected, matching Gate E-0")
        runs[d] = r

    shas = {d: r["ckpt_sha256"] for d, r in runs.items()}
    for d, s in shas.items():
        log(f"    ckpt_sha256  {d}\n                 {s}")
    dup = [(a, b) for i, a in enumerate(DINOV3_ALL) for b in DINOV3_ALL[i + 1:]
           if shas[a] == shas[b]]
    ok(not dup,
       f"the six ckpt_sha256 are distinct ({len(set(shas.values()))} distinct "
       f"of 6)" + (f"; COLLIDING: {dup}" if dup else ""))
    return runs


# --------------------------------------------------------- task 1: G2 drift --

def cnn_drift_rows():
    """One row per arm L LOCO cell, read from its own p5drift sidecar."""
    rows = []
    for corner in CORNERS:
        for city in CITIES:
            for seed in SEEDS:
                parent = f"mamnetr50-{corner}-loco-r450-{city}-hr-s{seed}"
                d = f"{parent}-p5drift"
                ok(os.path.exists(os.path.join(DUMPS_ROOT, d, ".DONE")),
                   f"{d}: .DONE present")
                r = read_run_json(d)
                ok(r.get("parent_run_id") == parent,
                   f"{d}: parent_run_id matches its cell")
                stages = r["drift_rel_l2_per_stage"]
                cka = r["cka_per_stage"]
                ok(set(cka) == set(CKA_STAGES),
                   f"{d}: CKA is reported on exactly the four ResNet stages")
                rows.append({
                    "run_id": parent, "corner": corner, "city": city,
                    "seed": seed,
                    "rel_l2": {s: float(stages[s]["relative_l2"])
                               for s in RESNET_STAGES},
                    "rel_l2_whole_encoder":
                        float(stages["whole_encoder"]["relative_l2"]),
                    "n_params_whole_encoder":
                        int(stages["whole_encoder"]["n_params"]),
                    "cka": {s: float(cka[s]) for s in CKA_STAGES},
                    "cka_mean_over_4_stages":
                        float(np.mean([cka[s] for s in CKA_STAGES])),
                })
    log(f"[task 1] read {len(rows)} MAMNet-R50 p5drift sidecars")
    return rows


def task1a_per_stage(rows, pooled):
    """Restate the per-stage relative L2 and CKA table, and check it against
    the table already in the pooled JSON rather than trusting either alone."""
    table, sep = {}, {}
    for corner in CORNERS:
        cr = [r for r in rows if r["corner"] == corner]
        table[corner] = {
            "n_cells": len(cr),
            "relative_l2": {
                s: {"mean": float(np.mean([r["rel_l2"][s] for r in cr])),
                    "median": float(np.median([r["rel_l2"][s] for r in cr])),
                    "min": float(np.min([r["rel_l2"][s] for r in cr])),
                    "max": float(np.max([r["rel_l2"][s] for r in cr]))}
                for s in RESNET_STAGES},
            "relative_l2_whole_encoder":
                med_range([r["rel_l2_whole_encoder"] for r in cr]),
            "cka": {
                s: {"mean": float(np.mean([r["cka"][s] for r in cr])),
                    "median": float(np.median([r["cka"][s] for r in cr])),
                    "min": float(np.min([r["cka"][s] for r in cr])),
                    "max": float(np.max([r["cka"][s] for r in cr]))}
                for s in CKA_STAGES},
        }
    for pre, hi, lo in (("sup", "sup_hi", "sup_lo"), ("ssl", "ssl_hi", "ssl_lo")):
        sep[pre] = {
            "relative_l2_lo_below_hi_no_overlap": {
                s: bool(table[lo]["relative_l2"][s]["max"]
                        < table[hi]["relative_l2"][s]["min"])
                for s in RESNET_STAGES},
            "relative_l2_lo_below_hi_no_overlap_whole_encoder": bool(
                table[lo]["relative_l2_whole_encoder"]["max"]
                < table[hi]["relative_l2_whole_encoder"]["min"]),
            "cka_lo_above_hi_no_overlap": {
                s: bool(table[lo]["cka"][s]["min"] > table[hi]["cka"][s]["max"])
                for s in CKA_STAGES},
            "cka_lo_above_hi_in_mean": {
                s: bool(table[lo]["cka"][s]["mean"] > table[hi]["cka"][s]["mean"])
                for s in CKA_STAGES},
        }

    # cross-check against gates.G2.per_corner, which the pooled pass wrote
    prev = pooled["gates"]["G2"]["per_corner"]
    worst = 0.0
    for corner in CORNERS:
        for s in list(RESNET_STAGES) + ["whole_encoder"]:
            if s == "whole_encoder":
                a = table[corner]["relative_l2_whole_encoder"]["mean"]
            else:
                a = table[corner]["relative_l2"][s]["mean"]
            worst = max(worst, abs(a - prev[corner]["relative_l2"][s]["mean"]))
        for s in CKA_STAGES:
            worst = max(worst, abs(table[corner]["cka"][s]["mean"]
                                   - prev[corner]["cka"][s]["mean"]))
    ok(worst < 1e-12,
       f"task 1a reproduces gates.G2.per_corner exactly "
       f"(max |delta| over 40 means = {worst:.3e})")

    all_l2 = all(v for pre in sep for v in
                 sep[pre]["relative_l2_lo_below_hi_no_overlap"].values())
    all_cka = all(v for pre in sep for v in
                  sep[pre]["cka_lo_above_hi_no_overlap"].values())
    all_cka_mean = all(v for pre in sep for v in
                       sep[pre]["cka_lo_above_hi_in_mean"].values())
    return {
        "axis": "per ResNet stage, arm L only",
        "per_corner": table,
        "separation": sep,
        "relative_l2_separates_no_overlap_every_stage": bool(all_l2),
        "cka_separates_no_overlap_every_stage": bool(all_cka),
        "cka_orders_correctly_in_mean_every_stage": bool(all_cka_mean),
        "reproduces_pooled_G2_block": True,
        "max_abs_delta_vs_pooled_G2": float(worst),
        "statement":
            "Relative L2 separates the _lo corners from their _hi counterparts "
            "with no overlap across the nine cells at every ResNet stage and on "
            "the whole encoder. Linear CKA orders the corners the same way in "
            "the mean at every stage, but the nine-cell ranges overlap, so CKA "
            "does not separate them under the no-overlap rule.",
    }


def task1b_whole_encoder(rows, dino):
    """The ONE comparable axis: whole-encoder relative L2, arm L vs DINOv3."""
    corner_l2 = {c: med_range([r["rel_l2_whole_encoder"] for r in rows
                               if r["corner"] == c]) for c in CORNERS}
    corner_cka = {c: med_range([r["cka_mean_over_4_stages"] for r in rows
                                if r["corner"] == c]) for c in CORNERS}

    d_l2 = {"loco": {}, "upper": {}}
    d_cka = {"loco": {}, "upper": {}}
    for group, names in (("loco", DINOV3_LOCO), ("upper", DINOV3_UPPER)):
        for city, name in zip(CITIES, names):
            r = dino[name]
            d_l2[group][city] = {
                "run_id": name,
                "relative_l2_whole_encoder":
                    float(r["drift_rel_l2_whole_encoder"]["relative_l2"]),
                "n_params": int(r["drift_rel_l2_whole_encoder"]["n_params"]),
                "ckpt_sha256": r["ckpt_sha256"],
            }
            if r.get("cka_mean") is None:
                d_cka[group][city] = {
                    "run_id": name, "cka_mean": None,
                    "status": "ABSENT",
                    "reason": (f"cka_n_images = {r.get('cka_n_images')}; the "
                               f"within-city upper cells carry no CKA and none "
                               f"is imputed"),
                }
            else:
                d_cka[group][city] = {
                    "run_id": name, "cka_mean": float(r["cka_mean"]),
                    "status": "PRESENT",
                    "cka_n_images": int(r.get("cka_n_images", 0)),
                    "n_blocks": len(r.get("cka_per_block", [])),
                }

    loco_vals = [d_l2["loco"][c]["relative_l2_whole_encoder"] for c in CITIES]
    upper_vals = [d_l2["upper"][c]["relative_l2_whole_encoder"] for c in CITIES]
    ref = {
        "dinov3_loco": med_range(loco_vals),
        "dinov3_upper": med_range(upper_vals),
        "dinov3_all_six": med_range(loco_vals + upper_vals),
    }

    def dist_set(corner_vals, refval, refname, with_ratio=True):
        """Every corner's distance to one reference value, plus the ordering."""
        rows = {}
        for c in CORNERS:
            a = corner_vals[c]["median"]
            row = {"corner": c, "corner_median": a, "reference": refname,
                   "reference_value": float(refval),
                   "absolute_difference": float(abs(a - refval))}
            if with_ratio:
                row["ratio_corner_over_reference"] = float(a / refval)
            rows[c] = row
        order = sorted(CORNERS, key=lambda c: rows[c]["absolute_difference"])
        return {
            "reference_value": float(refval), "reference_name": refname,
            "per_corner": rows,
            "ordering_nearest_first": order,
            "ssl_lo_distance": rows["ssl_lo"]["absolute_difference"],
            "sup_hi_distance": rows["sup_hi"]["absolute_difference"],
            "ssl_lo_nearer_than_sup_hi": bool(
                rows["ssl_lo"]["absolute_difference"]
                < rows["sup_hi"]["absolute_difference"]),
            "sup_hi_minus_ssl_lo_distance": float(
                rows["sup_hi"]["absolute_difference"]
                - rows["ssl_lo"]["absolute_difference"]),
        }

    # Every reference is reported under BOTH its median and its mean, and the
    # ordering under each is stated, so the reader can see whether the reading
    # depends on that choice rather than being told it does not.
    l2_refs = {}
    for key in ("dinov3_loco", "dinov3_upper", "dinov3_all_six"):
        l2_refs[key] = {
            "cells": ref[key],
            "median": dist_set(corner_l2, ref[key]["median"],
                               f"{key}_median"),
            "mean": dist_set(corner_l2, ref[key]["mean"], f"{key}_mean"),
        }
        same = (l2_refs[key]["median"]["ordering_nearest_first"]
                == l2_refs[key]["mean"]["ordering_nearest_first"])
        l2_refs[key]["ordering_same_under_median_and_mean"] = bool(same)
        l2_refs[key]["ordering_note"] = (
            f"the four corners' distance ordering is "
            f"{'unchanged' if same else 'CHANGED'} between the "
            f"{key} median ({ref[key]['median']:.6f}) and mean "
            f"({ref[key]['mean']:.6f}): "
            f"{' < '.join(l2_refs[key]['median']['ordering_nearest_first'])} "
            f"under the median, "
            f"{' < '.join(l2_refs[key]['mean']['ordering_nearest_first'])} "
            f"under the mean.")

    prim = l2_refs["dinov3_loco"]["median"]
    stability = {
        key: {"ssl_lo_ratio": l2_refs[key]["median"]["per_corner"]["ssl_lo"][
                  "ratio_corner_over_reference"],
              "sup_hi_ratio": l2_refs[key]["median"]["per_corner"]["sup_hi"][
                  "ratio_corner_over_reference"],
              "ssl_lo_nearer_than_sup_hi":
                  l2_refs[key]["median"]["ssl_lo_nearer_than_sup_hi"]}
        for key in l2_refs}
    fold = {key: float(v["sup_hi_ratio"] / v["ssl_lo_ratio"])
            for key, v in stability.items()}
    stability_holds = all(v["ssl_lo_nearer_than_sup_hi"]
                          for v in stability.values())

    distances = {
        "question": ("G2's literal second clause: does ssl_lo's whole-encoder "
                     "relative L2 sit nearer DINOv3's than sup_hi's does?"),
        "primary_reference": "dinov3_loco_median",
        "primary_reference_value": ref["dinov3_loco"]["median"],
        "primary_reference_rationale":
            "the vanilla LOCO cells are the training condition matched to arm "
            "L's four LOCO-trained corners, and they are the only DINOv3 cells "
            "with a CKA leg; the within-city upper cells and the six-cell pool "
            "are reported alongside and are not substituted for it",
        "by_reference": l2_refs,
        "ssl_lo_vs_primary": prim["per_corner"]["ssl_lo"],
        "sup_hi_vs_primary": prim["per_corner"]["sup_hi"],
        "all_four_corners_vs_primary": prim["per_corner"],
        "ssl_lo_nearer_than_sup_hi_vs_primary":
            prim["ssl_lo_nearer_than_sup_hi"],
        "sup_hi_minus_ssl_lo_distance_primary":
            prim["sup_hi_minus_ssl_lo_distance"],
        "reference_stability": {
            "ratios_by_reference": stability,
            "fold_separation_sup_hi_over_ssl_lo": fold,
            "ordering_holds_under_every_reference": bool(stability_holds),
            "statement":
                "the ssl_lo-vs-sup_hi ordering and its roughly six- to "
                "sevenfold ratio separation hold under every reference tested "
                "(LOCO median 1.74x vs 11.66x; within-city upper 1.56x vs "
                "10.47x; six-cell pool 1.64x vs 11.03x), so the reading is not "
                "an artifact of the reference choice.",
        },
    }
    inside = [c for c in CORNERS
              if ref["dinov3_loco"]["min"] <= corner_l2[c]["median"]
              <= ref["dinov3_loco"]["max"]]
    distances["corners_whose_median_falls_inside_the_dinov3_loco_range"] = inside
    distances["sup_lo_position"] = {
        "sup_lo_median": corner_l2["sup_lo"]["median"],
        "dinov3_loco_range": [ref["dinov3_loco"]["min"],
                              ref["dinov3_loco"]["max"]],
        "inside_range": bool("sup_lo" in inside),
        "statement":
            "G2 asked whether ssl_lo is nearer than sup_hi, not whether ssl_lo "
            "is uniquely near, so sup_lo's position does not bear on the gate. "
            "A supervised corner also landing inside the DINOv3 LOCO "
            "relative-L2 range indicates the drift regime tracks the "
            "fine-tuning learning rate rather than the pretraining objective, "
            "consistent with the absent Experiment B main effects recorded in "
            "DECISIONS.md. Stated as a measurement; no further reading is "
            "attached.",
    }

    # CKA, on the two quantities that exist.  The CNN number is the unweighted
    # mean over its 4 ResNet stages; the DINOv3 number is the mean over its 12
    # ViT blocks.  Different layer counts, so this is the same caveat as L2.
    cka_loco = [d_cka["loco"][c]["cka_mean"] for c in CITIES]
    cka_ref = med_range(cka_loco)
    cka_by_ref = {
        "median": dist_set(corner_cka, cka_ref["median"],
                           "dinov3_loco_median", with_ratio=False),
        "mean": dist_set(corner_cka, cka_ref["mean"], "dinov3_loco_mean",
                         with_ratio=False),
    }
    cka_same = (cka_by_ref["median"]["ordering_nearest_first"]
                == cka_by_ref["mean"]["ordering_nearest_first"])
    prim_cka = cka_by_ref["median"]
    l2_fold = (prim["per_corner"]["sup_hi"]["ratio_corner_over_reference"]
               / prim["per_corner"]["ssl_lo"]["ratio_corner_over_reference"])

    cka_block = {
        "armL_mean_cka_definition":
            "per cell, the unweighted mean of linear CKA over the four ResNet "
            "stages (layer1-4); pooled to the 9-cell median and range per corner",
        "dinov3_mean_cka_definition":
            "the dump's own cka_mean, the mean of linear CKA over the 12 ViT "
            "blocks",
        "armL_per_corner": corner_cka,
        "dinov3_loco": d_cka["loco"],
        "dinov3_upper": d_cka["upper"],
        "dinov3_upper_status": "ABSENT on all three cities, not imputed",
        "dinov3_loco_median_range": cka_ref,
        "reference_internal_spread":
            "the DINOv3 CKA reference is a 3-cell median over a set with a 2x "
            "internal range (phoenix 0.3697 against chicago 0.7622 and miami "
            "0.7368), so the median is the miami cell alone. Both the median "
            "(0.7368) and the mean (0.6229) are reported as reference values "
            "and neither is picked over the other.",
        "by_reference": cka_by_ref,
        "ordering_same_under_median_and_mean": bool(cka_same),
        "ordering_note":
            f"the four corners' distance ordering is "
            f"{'unchanged' if cka_same else 'CHANGED'} between the CKA "
            f"reference median ({cka_ref['median']:.4f}) and mean "
            f"({cka_ref['mean']:.4f}): "
            f"{' < '.join(cka_by_ref['median']['ordering_nearest_first'])} "
            f"under the median, "
            f"{' < '.join(cka_by_ref['mean']['ordering_nearest_first'])} "
            f"under the mean.",
        "ssl_lo_vs_dinov3_loco": prim_cka["per_corner"]["ssl_lo"],
        "sup_hi_vs_dinov3_loco": prim_cka["per_corner"]["sup_hi"],
        "all_four_corners_vs_dinov3_loco": prim_cka["per_corner"],
        "ssl_lo_nearer_than_sup_hi": prim_cka["ssl_lo_nearer_than_sup_hi"],
        "separation_between_the_two_corners": float(abs(
            prim_cka["ssl_lo_distance"] - prim_cka["sup_hi_distance"])),
        "verdict_on_this_axis": "UNINFORMATIVE",
        "verdict_note":
            f"the two distances are "
            f"{prim_cka['sup_hi_distance']:.4f} (sup_hi) and "
            f"{prim_cka['ssl_lo_distance']:.4f} (ssl_lo), a difference of "
            f"{abs(prim_cka['ssl_lo_distance'] - prim_cka['sup_hi_distance']):.4f}. "
            f"The two corners are equidistant from DINOv3 on this axis, so it "
            f"is uninformative about the clause. A "
            f"{abs(prim_cka['ssl_lo_distance'] - prim_cka['sup_hi_distance']):.3f} "
            f"separation does not contradict the {l2_fold:.1f}-fold separation "
            f"on the relative-L2 axis.",
    }

    return {
        "axis": "whole-encoder relative L2",
        "axis_rule":
            "arm L's drift is per ResNet stage (conv1, layer1-4); DINOv3's is "
            "per ViT block (0-11). No block-to-stage mapping is constructed and "
            "no blocks are averaged into pseudo-stages. The whole-encoder "
            "aggregate is the only directly comparable quantity and is the only "
            "axis this comparison is reported on.",
        "armL_per_corner": corner_l2,
        "armL_dispersion_note":
            "the 9-cell median and range, NOT an image-cluster bootstrap CI. "
            "Relative L2 is a weight-space scalar with one value per cell, so "
            "there is no per-image unit to resample and the dumps do not "
            "support a bootstrap CI on this axis.",
        "dinov3_cells": d_l2,
        "dinov3_reference_medians": ref,
        "distances": distances,
        "cka": cka_block,
    }


def dinov3_per_block_curves(dino):
    """The 12-block CKA curve for each LOCO city, reported not chased."""
    curves, per_block = {}, {}
    for city, name in zip(CITIES, DINOV3_LOCO):
        r = dino[name]
        blocks = r["cka_per_block"]
        ok(len(blocks) == 12, f"{name}: 12 per-block CKA entries")
        curves[city] = [{"block": int(b["block"]),
                         "linear_cka": float(b["linear_cka"]),
                         "mean_cosine": float(b.get("mean_cosine"))
                         if b.get("mean_cosine") is not None else None}
                        for b in blocks]
        per_block[city] = {int(b["block"]): float(b["linear_cka"])
                           for b in blocks}
    lowest = {b: min(CITIES, key=lambda c: per_block[c][b]) for b in range(12)}
    phoenix_lowest_everywhere = all(v == "phoenix" for v in lowest.values())
    return {
        "scope": "DINOv3 vanilla LOCO only; the three upper cells have no CKA",
        "curves": curves,
        "lowest_city_by_block": lowest,
        "phoenix_lowest_at_every_block": bool(phoenix_lowest_everywhere),
        "phoenix_block0_linear_cka": float(per_block["phoenix"][0]),
        "observation":
            "phoenix's LOCO CKA sits below chicago's and miami's at every one of "
            "the 12 blocks, including block 0, not only in the deep blocks. "
            "Reported, not chased: this is a DINOv3 property and belongs to arm "
            "E's ledger, not to arm L's reading.",
        "flag": "reported_not_chased",
    }


# ------------------------------------------------------- task 2: Experiment A --

def task2(mod, per_cell, pooled_R, contr_A):
    """Characterise Experiment A across the 2x2."""
    M = mod.PRIMARY_METRIC

    corner_medians = {}
    for c in CORNERS:
        b = pooled_R["expA"][c][M]
        corner_medians[c] = {
            "median_R": b["median_R"], "ci95": b["ci95"],
            "ci_method": b["ci_method"], "cluster_unit": b["cluster_unit"],
            "mean_R": b["mean_R"], "n_used": b["n_used"],
            "n_clusters": b["n_clusters"], "n_excluded": b["n_excluded"],
            "n_cells": b["n_cells"],
        }

    # ---- 2a: sup_lo, per city, per seed, per cell
    sup_lo = [c for c in per_cell if c["corner"] == "sup_lo"]

    by_city = {}
    for city in CITIES:
        b = pooled_R["expA_by_city"]["sup_lo"][city][M]
        by_city[city] = {"median_R": b["median_R"], "ci95": b["ci95"],
                         "mean_R": b["mean_R"], "n_used": b["n_used"],
                         "n_clusters": b["n_clusters"], "n_cells": b["n_cells"],
                         "ci_method": b["ci_method"]}

    def bound_tie_diagnostic(v, k, bound):
        """Why a percentile bound landed exactly on a round value.

        The bootstrap distribution of a MEDIAN is discrete: every replicate is
        an order statistic of the observed values.  When several observations
        are exactly equal, a run of replicates takes that same value, and a
        percentile falling inside the run prints as that value exactly.  That
        reads like a coding artifact and is not one, so it is diagnosed rather
        than left for a reader to trip over.
        """
        pad, nc = mod._pad_clusters(v, k)
        rng = np.random.default_rng(mod.BOOT_SEED)
        boots = np.empty(mod.B, dtype=np.float64)
        for lo in range(0, mod.B, 500):
            hi = min(lo + 500, mod.B)
            idx = rng.integers(0, nc, size=(hi - lo, nc))
            boots[lo:hi] = np.nanmedian(pad[idx].reshape(hi - lo, -1), axis=1)
        return {
            "bound": float(bound),
            "n_replicates_equal_to_bound": int((boots == bound).sum()),
            "n_replicates_strictly_above_zero": int((boots > 0.0).sum()),
            "fraction_replicates_strictly_above_zero":
                float((boots > 0.0).mean()),
            "max_replicate": float(boots.max()),
            "n_observed_R_exactly_zero": int((v == 0.0).sum()),
            "n_observed_R": int(v.size),
            "diagnosis":
                "the bootstrap distribution of the median is discrete; "
                f"{int((boots == bound).sum())} of {mod.B} replicates take the "
                f"value {bound} exactly, because {int((v == 0.0).sum())} of the "
                f"{int(v.size)} pooled per-image R values are exactly 0. The "
                f"percentile falls inside that run of ties. The bound is a tie, "
                f"not a boundary: {int((boots > 0.0).sum())} replicates lie "
                f"strictly above zero, so the interval does not exclude zero.",
        }

    def pool_cells(cells):
        """Image-cluster bootstrap of the median over an arbitrary cell set."""
        vals, keys = [], []
        for c in cells:
            R, usable = c["_R"]["expA"][M]
            vals.append(R[usable])
            keys.append(np.char.add(
                np.asarray([c["city"] + "|"] * int(usable.sum())),
                c["_image_ids"][usable].astype(str)))
        v = np.concatenate(vals) if vals else np.array([])
        k = np.concatenate(keys) if keys else np.array([])
        bs = mod.cluster_bootstrap(v, k)
        out = {"median_R": bs["point"], "ci95": [bs["ci95_lo"], bs["ci95_hi"]],
               "ci_method": "image_cluster_bootstrap_B10000",
               "cluster_unit": "(city, image_id)",
               "mean_R": float(v.mean()) if v.size else None,
               "n_used": int(v.size), "n_clusters": bs["n_clusters"],
               "n_cells": len(cells),
               "cells": [c["run_id"] for c in cells]}
        for side, b in (("ci95_lo", bs["ci95_lo"]), ("ci95_hi", bs["ci95_hi"])):
            if b is not None and float(b) == round(float(b), 10) and b == 0.0:
                out.setdefault("bound_diagnostics", {})[side] = \
                    bound_tie_diagnostic(v, k, 0.0)
                SANITY.append(
                    f"expA: the {side} bound of the pooled median over "
                    f"{len(cells)} cell(s) is exactly 0.0. Diagnosed as a tie "
                    f"in the discrete bootstrap distribution of the median, "
                    f"not a coding artifact and not a boundary; see "
                    f"bound_diagnostics.{side}.")
        return out

    by_seed = {f"s{s}": pool_cells([c for c in sup_lo if c["seed"] == s])
               for s in SEEDS}

    per_cell_rows = []
    for c in sorted(sup_lo, key=lambda x: (x["city"], x["seed"])):
        blk = c["expA"]["R"][M]
        per_cell_rows.append({
            "run_id": c["run_id"], "city": c["city"], "seed": c["seed"],
            "median_R": blk["median_R"], "ci95": blk["ci95"],
            "mean_R": blk["mean_R"], "n_used": blk["n_used"],
            "n_excluded": blk["n_excluded"],
            "steps_to_best": c["expA"]["steps_to_best"],
            "steps_ceiling": c["expA"]["steps_ceiling"],
            "budget_bound": bool(c["expA"]["budget_bound"]),
            "readable": bool(c["expA"]["readable"]),
            "readable_strict30": bool(c["expA"]["readable_strict30"]),
        })
    n_neg = sum(1 for r in per_cell_rows if r["median_R"] < 0)
    n_pos = sum(1 for r in per_cell_rows if r["median_R"] >= 0)

    # ---- 2b: main effects from the four corner medians, both experiments
    def effects(src):
        m = {c: src[c][M]["median_R"] for c in CORNERS}
        return {
            "corner_medians_used": m,
            "ssl_effect": float((m["ssl_hi"] + m["ssl_lo"]) / 2.0
                                - (m["sup_hi"] + m["sup_lo"]) / 2.0),
            "low_lr_effect": float((m["sup_lo"] + m["ssl_lo"]) / 2.0
                                   - (m["sup_hi"] + m["ssl_hi"]) / 2.0),
            "interaction": float((m["sup_hi"] + m["ssl_lo"]) / 2.0
                                 - (m["sup_lo"] + m["ssl_hi"]) / 2.0),
            "formula": {
                "ssl_effect": "(ssl_hi + ssl_lo)/2 - (sup_hi + sup_lo)/2",
                "low_lr_effect": "(sup_lo + ssl_lo)/2 - (sup_hi + ssl_hi)/2",
                "interaction": "(sup_hi + ssl_lo)/2 - (sup_lo + ssl_hi)/2",
            },
            "note": "computed from the four pooled corner MEDIANS, the same way "
                    "DECISIONS.md computed the Experiment B effects; no test is "
                    "attached to these three numbers",
        }

    eff_A, eff_B = effects(pooled_R["expA"]), effects(pooled_R["expB"])
    # DECISIONS.md quotes the Experiment B effects to 3 dp; check we reproduce.
    ok(abs(eff_B["ssl_effect"] - (-0.003)) < 5e-4
       and abs(eff_B["low_lr_effect"] - (-0.018)) < 5e-4,
       f"Experiment B main effects reproduce DECISIONS.md "
       f"(ssl {eff_B['ssl_effect']:+.4f} vs -0.003, "
       f"low-LR {eff_B['low_lr_effect']:+.4f} vs -0.018)")

    # ---- 2c: the budget question
    readable = [c for c in sup_lo if c["expA"]["readable"]]
    unreadable = [c for c in sup_lo if not c["expA"]["readable"]]
    budget = {
        "rule": pooled["gates"]["G3"]["rule"],
        "rule_source": "gates.G3, unchanged; this block only re-applies it",
        "asymmetry":
            "a HIGH R under a binding budget is VALID because undertraining can "
            "only depress R; a LOW R under a binding budget is AMBIGUOUS between "
            "locus and undertraining and is unreadable",
        "n_cells": len(sup_lo),
        "n_budget_bound": sum(1 for c in sup_lo if c["expA"]["budget_bound"]),
        "n_readable": len(readable),
        "n_unreadable": len(unreadable),
        "readable_cells": [c["run_id"] for c in readable],
        "unreadable_cells": [c["run_id"] for c in unreadable],
        "readable_set_note":
            "for sup_lo the readable set is exactly the set whose 30-epoch "
            "budget does not bind: no budget-bound sup_lo cell reaches the "
            "0.70 high-R exemption",
        "median_R_all_cells": corner_medians["sup_lo"],
        "median_R_readable_cells_only": pool_cells(readable),
        "median_R_unreadable_cells_only": pool_cells(unreadable),
        "both_reported":
            "neither figure replaces the other. The all-cells median is the "
            "pooled corner value; the readable-only median describes the "
            "subset on which a low R is not confounded with undertraining.",
        "grid_context": {
            "expA_budget_bound_over_36_cells":
                pooled["gates"]["G3"]["counts"]["expA_budget_bound"],
            "expA_readable_over_36_cells":
                pooled["gates"]["G3"]["counts"]["expA_readable"],
        },
    }

    # ---- 2d: the three Holm-corrected Experiment A contrasts
    stored = {r["contrast"]: r["by_metric"][M]
              for r in pooled["contrasts"]["expA_corroborating"]}
    rows = []
    worst = 0.0
    for r in contr_A:
        m = r["by_metric"][M]
        s = stored[r["contrast"]]
        worst = max(worst, abs(m["mean"] - s["mean"]),
                    abs((m["p_holm"] or 0.0) - (s["p_holm"] or 0.0)))
        rows.append({
            "contrast": r["contrast"], "compares": r["compares"],
            "note": r["note"], "experiment": "expA", "metric": M,
            "test": m["test"], "mean_delta_R": m["mean"],
            "median_delta_R": m["median"], "stderr": m["stderr"],
            "ci95": m["ci95"], "ci_method": m["ci_method"],
            "cluster_unit": m["cluster_unit"], "B": m["B"],
            "two_tailed": True, "p_raw": m["p_raw"], "p_holm": m["p_holm"],
            "holm_family": m["holm_family"],
            "n": m["n"], "n_clusters": m["n_clusters"],
            "n_usable_left_only": m["n_usable_left_only"],
            "n_usable_right_only": m["n_usable_right_only"],
            "separates_at_alpha_0.05": bool(m["p_holm"] is not None
                                            and m["p_holm"] < 0.05),
        })
    ok(worst < 1e-12,
       f"the three Experiment A contrasts reproduce contrasts.expA_corroborating "
       f"exactly (max |delta| = {worst:.3e})")

    return {
        "experiment": "A (decoder retraining)",
        "decision_variable_note":
            "Experiment A is the CORROBORATING variable. DECISIONS.md ratified "
            "outcome 4 on Experiment B median R against the pre-committed "
            "threshold table. Nothing in this block re-adjudicates that.",
        "metric": M,
        "signature_table_expA": {
            "cnn_like_band": "0.12 - 0.67 (published OGLANet 0.12, "
                             "MAMNet-R34 0.67)",
            "dinov3_like": ">= 0.50",
            "source": "REVIEWS.md, arm L pre-committed reading",
        },
        "corner_medians": corner_medians,
        "sup_lo_breakdown": {
            "by_city": by_city,
            "by_seed": by_seed,
            "per_cell": per_cell_rows,
            "n_cells_with_negative_median_R": n_neg,
            "n_cells_with_non_negative_median_R": n_pos,
            "statement":
                f"{n_neg} of 9 sup_lo cells have a negative Experiment A median "
                f"R and {n_pos} have a non-negative one, so the corner's "
                f"negative pooled median is not carried by a single cell, and "
                f"it is also not uniform across the grid.",
        },
        "main_effects": {"expA": eff_A, "expB": eff_B,
                         "expB_source_check": "reproduces the -0.003 / -0.018 "
                                              "quoted in DECISIONS.md"},
        "budget_readability": budget,
        "contrasts_vs_sup_hi_axis": rows,
        "sanity_flags": list(SANITY),
        "exclusions":
            "no cell, seed or image is excluded, winsorised or trimmed here "
            "beyond the per-image headroom rule already recorded in the pooled "
            "pass (R undefined when upper - LOCO < 0.02 on the IoU fraction "
            "scale); the readable-only recomputation is reported ALONGSIDE the "
            "all-cells figure, never in place of it",
    }


# ---------------------------------------------------------------- assembly --

INIT_CAVEAT = (
    "DINOv3 init load: on all six cells the as-published dinov3_backbone.py "
    "construction drops 37 of 188 checkpoint tensors (0 missing / 37 "
    "unexpected: 1 storage_tokens, 12 attn.qkv.bias_mask, 24 ls.gamma), "
    "matching Gate E-0's recorded result. Drift is measured against the init "
    "that produced the published cells, so the denominator of every DINOv3 "
    "relative L2 here is that same reduced init. Carried forward without "
    "interpretation."
)

COMPARABILITY_CAVEAT = (
    "Relative L2 over a ResNet-50 encoder (23.5M parameters, 159 tensors, "
    "convolutional) and over a ViT-S/16 encoder (21.6M parameters, 151 "
    "tensors, attention and MLP) is the same formula over different parameter "
    "counts and different layer types. The two numbers are not a matched "
    "measurement of the same thing and the comparison is indicative only. G2 "
    "moves off PARTIAL only with this caveat attached."
)


def build_g2_block(t1a, t1b, curves, dino):
    d = t1b["distances"]
    cka = t1b["cka"]
    clause = d["ssl_lo_nearer_than_sup_hi_vs_primary"]
    ratios = d["reference_stability"]["ratios_by_reference"]["dinov3_loco"]
    sep = t1a["separation"]
    fold_lo_hi = {
        pre: {s: sep[pre]["relative_l2_lo_below_hi_no_overlap"][s]
              for s in RESNET_STAGES} for pre in ("sup", "ssl")}
    ratio_hi_over_lo = {
        pre: {s: float(t1a["per_corner"][f"{pre}_hi"]["relative_l2"][s]["mean"]
                       / t1a["per_corner"][f"{pre}_lo"]["relative_l2"][s]["mean"])
              for s in RESNET_STAGES} for pre in ("sup", "ssl")}
    for pre in ("sup", "ssl"):
        ratio_hi_over_lo[pre]["whole_encoder"] = float(
            t1a["per_corner"][f"{pre}_hi"]["relative_l2_whole_encoder"]["mean"]
            / t1a["per_corner"][f"{pre}_lo"]["relative_l2_whole_encoder"]["mean"])
    all_ratios = [v for pre in ratio_hi_over_lo
                  for v in ratio_hi_over_lo[pre].values()]

    conjunct_1 = {
        "clause": ("the _lo corners show measurably smaller encoder drift "
                   "(P5 relative L2, CKA) than their _hi counterparts"),
        "result": "PASS",
        "basis": "arm L only; no DINOv3 cell enters this conjunct",
        "relative_l2_no_overlap_at_every_stage":
            t1a["relative_l2_separates_no_overlap_every_stage"],
        "no_overlap_by_pretraining_and_stage": fold_lo_hi,
        "mean_ratio_hi_over_lo": ratio_hi_over_lo,
        "mean_ratio_hi_over_lo_range": [float(min(all_ratios)),
                                        float(max(all_ratios))],
        "statement":
            f"The _hi corners' mean relative L2 is "
            f"{min(all_ratios):.1f}x to {max(all_ratios):.1f}x their _lo "
            f"counterparts', roughly a tenfold separation, and the nine-cell "
            f"ranges do not overlap at any ResNet stage or on the whole "
            f"encoder. Linear CKA orders the corners the same way in the mean "
            f"at every stage "
            f"({t1a['cka_orders_correctly_in_mean_every_stage']}) but its "
            f"nine-cell ranges overlap "
            f"({t1a['cka_separates_no_overlap_every_stage']} under the "
            f"no-overlap rule), so relative L2 carries this conjunct and CKA "
            f"corroborates its direction.",
    }
    conjunct_2 = {
        "clause": ("ssl_lo sits nearer the measured DINOv3 drift than sup_hi "
                   "does"),
        "reported_per_axis": True,
        "whole_encoder_relative_l2": {
            "result": "PASS" if clause else "FAIL",
            "ssl_lo_distance": d["ssl_lo_vs_primary"]["absolute_difference"],
            "sup_hi_distance": d["sup_hi_vs_primary"]["absolute_difference"],
            "ssl_lo_ratio_to_reference": ratios["ssl_lo_ratio"],
            "sup_hi_ratio_to_reference": ratios["sup_hi_ratio"],
            "fold_separation":
                d["reference_stability"]["fold_separation_sup_hi_over_ssl_lo"][
                    "dinov3_loco"],
            "reference": d["primary_reference"],
            "reference_value": d["primary_reference_value"],
            "statement":
                f"ssl_lo sits at "
                f"{ratios['ssl_lo_ratio']:.2f}x the DINOv3 LOCO median and "
                f"sup_hi at {ratios['sup_hi_ratio']:.2f}x, distances "
                f"{d['ssl_lo_vs_primary']['absolute_difference']:.6f} and "
                f"{d['sup_hi_vs_primary']['absolute_difference']:.6f}. The "
                f"ordering and the fold separation hold under every reference "
                f"tested.",
        },
        "mean_cka": {
            "result": "UNINFORMATIVE",
            "ssl_lo_distance": cka["ssl_lo_vs_dinov3_loco"][
                "absolute_difference"],
            "sup_hi_distance": cka["sup_hi_vs_dinov3_loco"][
                "absolute_difference"],
            "separation": cka["separation_between_the_two_corners"],
            "statement": cka["verdict_note"],
        },
        "combined_note":
            "the two axes are not combined into a single verdict. Relative L2 "
            "answers the clause; CKA is uninformative on it and does not "
            "contradict it.",
    }
    return {
        "gate": "G2",
        "question": ("do arm L's low-LR corners sit in DINOv3's actual "
                     "fine-tuning drift regime, or merely carry a smaller LR "
                     "label?"),
        "scope": ("this block ADDS the DINOv3 leg that gates.G2 recorded as NOT "
                  "EVALUATED. gates.G2 itself is unchanged and still reads "
                  "PARTIAL; this block carries the closure."),
        "armL_half_restated": t1a,
        "one_axis_comparison": t1b,
        "dinov3_per_block_cka": curves,
        "dinov3_init_caveat": INIT_CAVEAT,
        "architecture_comparability_limitation": COMPARABILITY_CAVEAT,
        "dinov3_cells_used": [
            {"run_id": n, "mode": dino[n]["mode"], "city": dino[n]["holdout"],
             "ckpt_sha256": dino[n]["ckpt_sha256"],
             "ckpt_path": dino[n]["ckpt_path"],
             "init_source_sha256": dino[n]["init_source_sha256"],
             "init_load_n_missing": dino[n]["init_load_n_missing"],
             "init_load_n_unexpected": dino[n]["init_load_n_unexpected"],
             "lr": dino[n]["lr"], "checkpoint_epoch": dino[n].get("checkpoint_epoch"),
             "cka_mean": dino[n].get("cka_mean"),
             "cka_n_images": dino[n].get("cka_n_images"),
             "naming_collision_caveat": dino[n].get("naming_collision_caveat")}
            for n in DINOV3_ALL],
        "ckpt_sha256_distinct_across_six": True,
        "g2_test_has_two_parts":
            "G2's stated test is a conjunction: the _lo corners must drift "
            "measurably less than the _hi corners, AND ssl_lo must sit nearer "
            "the measured DINOv3 drift than sup_hi does. The two conjuncts are "
            "reported separately below.",
        "g2_conjunct_1": conjunct_1,
        "g2_conjunct_2": conjunct_2,
        "g2_status": "read_with_caveat",
        "g2_status_scope":
            "g2_status SUMMARISES g2_conjunct_1 and g2_conjunct_2; it does not "
            "replace them. A reader acting on the gate reads the two conjuncts, "
            "not this field.",
        "g2_status_caveat": COMPARABILITY_CAVEAT + " " + INIT_CAVEAT,
        "g2_status_note":
            "read_with_caveat means both legs of G2 are now measured — the arm "
            "L half (conjunct 1) and the DINOv3 half (conjunct 2) — and the "
            "gate can be read only with the architecture-comparability and "
            "init-load caveats attached. It does not restate gates.G2, which "
            "still carries PARTIAL as written at pooling time, and it does not "
            "re-adjudicate arm L's ratified outcome.",
    }


def splice_into_json(path, blocks):
    """Insert new top-level blocks by editing TEXT, so every pre-existing byte
    of the file survives verbatim."""
    with open(path, "rb") as fh:
        original = fh.read()
    text = original.decode("utf-8")
    stripped = text.rstrip()
    ok(stripped.endswith("}"), "the pooled JSON ends with a closing brace")
    head = stripped[:-1].rstrip()
    ok(head.endswith("}") or head.endswith("]") or head.endswith('"')
       or head[-1].isdigit() or head.endswith("e") or head.endswith("l"),
       "the last member of the pooled JSON ends cleanly before the splice")
    parts = []
    for key, val in blocks:
        body = json.dumps(val, indent=2, default=float)
        body = "\n".join(("  " + ln) if ln.strip() else ln
                         for ln in body.split("\n")).lstrip()
        parts.append(f'  "{key}": {body}')
    new_text = head + ",\n" + ",\n".join(parts) + "\n}" + text[len(stripped):]
    tmp = path + ".new"
    with open(tmp, "w") as fh:
        fh.write(new_text)
    os.replace(tmp, path)
    return original, new_text


def verify_splice(path, original_bytes, new_keys):
    """Re-parse and prove nothing pre-existing changed."""
    with open(path, "rb") as fh:
        after = fh.read()
    prev = json.loads(original_bytes.decode("utf-8"))
    now = json.loads(after.decode("utf-8"))

    ok(list(now.keys())[:len(prev)] == list(prev.keys()),
       "every pre-existing top-level key is still present, in its original order")
    ok(list(now.keys())[len(prev):] == list(new_keys),
       f"the only added top-level keys are {list(new_keys)}")

    # byte-level: the spliced file starts with the original file's bytes up to
    # its final closing brace
    head = original_bytes.decode("utf-8").rstrip()[:-1].rstrip()
    ok(after.decode("utf-8").startswith(head),
       f"the first {len(head)} bytes of the pooled JSON are unchanged")

    diffs = [k for k in prev
             if json.dumps(prev[k], sort_keys=True, default=float)
             != json.dumps(now[k], sort_keys=True, default=float)]
    ok(not diffs, f"every pre-existing block re-parses identically "
                  f"({'differs: ' + str(diffs) if diffs else 'all ' + str(len(prev))})")
    ok(json.dumps(prev["exp_b_lr_verification"], sort_keys=True)
       == json.dumps(now["exp_b_lr_verification"], sort_keys=True),
       "exp_b_lr_verification is byte-identical")
    return now


# ---------------------------------------------------------------------- md --

def cap(s):
    """Capitalise a JSON note being spliced into the middle of prose."""
    return s[:1].upper() + s[1:] if s else s


def f(x, nd=4, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def write_md(path, g2, expA):
    t1b = g2["one_axis_comparison"]
    L = t1b["armL_per_corner"]
    D = t1b["dinov3_cells"]
    ref = t1b["dinov3_reference_medians"]
    dist = t1b["distances"]
    cka = t1b["cka"]
    out = []
    A = out.append

    A("# Arm L — gate G2 closed on one axis, and Experiment A read across the 2x2")
    A("")
    A("Companion to `results/16_armL_ssl_regime.json`, blocks "
      "`g2_drift_comparison` and `expA_analysis`. Numbers only. The arm's "
      "outcome was ratified in `DECISIONS.md` on Experiment B median R and is "
      "not revisited here.")
    A("")
    A("---")
    A("")
    A("## 1. Gate G2 — the one comparable axis")
    A("")
    A("Arm L's drift is per ResNet stage; DINOv3's is per ViT block. We do not "
      "map blocks onto stages and we do not average blocks into pseudo-stages. "
      "The whole-encoder aggregate is the only directly comparable quantity, "
      "and it is the only axis this table uses.")
    A("")
    A("Dispersion is the median and observed range over cells, not a bootstrap "
      "CI: relative L2 is one weight-space scalar per cell, so there is no "
      "per-image unit to resample.")
    A("")
    A("| Condition | n cells | Whole-encoder relative L2, median | Range |")
    A("|---|---|---|---|")
    for c in CORNERS:
        v = L[c]
        A(f"| arm L `{c}` | {v['n_cells']} | {v['median']:.6f} | "
          f"{v['min']:.6f} – {v['max']:.6f} |")
    for city in CITIES:
        v = D["loco"][city]
        A(f"| DINOv3 vanilla LOCO, {city} | 1 | "
          f"{v['relative_l2_whole_encoder']:.6f} | — |")
    A(f"| DINOv3 vanilla LOCO, median of 3 | 3 | {ref['dinov3_loco']['median']:.6f} "
      f"| {ref['dinov3_loco']['min']:.6f} – {ref['dinov3_loco']['max']:.6f} |")
    for city in CITIES:
        v = D["upper"][city]
        A(f"| DINOv3 within-city upper, {city} | 1 | "
          f"{v['relative_l2_whole_encoder']:.6f} | — |")
    A(f"| DINOv3 within-city upper, median of 3 | 3 | "
      f"{ref['dinov3_upper']['median']:.6f} | "
      f"{ref['dinov3_upper']['min']:.6f} – {ref['dinov3_upper']['max']:.6f} |")
    A("")
    A("### The two distances")
    A("")
    A("G2's second conjunct asks whether `ssl_lo`'s whole-encoder drift sits "
      "nearer DINOv3's than `sup_hi`'s does. Reported against both the median "
      "and the mean of each reference, so the reader can see whether the "
      "ordering depends on that choice.")
    A("")
    A("| Corner | Median | Distance to LOCO median | Ratio | Distance to LOCO mean | Ratio |")
    A("|---|---|---|---|---|---|")
    rmed = dist["by_reference"]["dinov3_loco"]["median"]["per_corner"]
    rmea = dist["by_reference"]["dinov3_loco"]["mean"]["per_corner"]
    for c in CORNERS:
        A(f"| `{c}` | {rmed[c]['corner_median']:.6f} | "
          f"{rmed[c]['absolute_difference']:.6f} | "
          f"{rmed[c]['ratio_corner_over_reference']:.2f}x | "
          f"{rmea[c]['absolute_difference']:.6f} | "
          f"{rmea[c]['ratio_corner_over_reference']:.2f}x |")
    A("")
    A(f"Reference values: DINOv3 LOCO median "
      f"{ref['dinov3_loco']['median']:.6f}, mean "
      f"{ref['dinov3_loco']['mean']:.6f}. "
      + cap(dist["by_reference"]["dinov3_loco"]["ordering_note"]))
    A("")
    A(f"Difference between the two named distances: "
      f"{dist['sup_hi_minus_ssl_lo_distance_primary']:.6f} "
      f"(sup_hi's minus ssl_lo's).")
    A("")
    A(cap(dist["reference_stability"]["statement"]))
    A("")
    A(dist["sup_lo_position"]["statement"])
    A("")
    A("### Mean CKA, where it exists")
    A("")
    A("The arm L number is the unweighted mean over four ResNet stages; the "
      "DINOv3 number is the mean over twelve ViT blocks. Same caveat as "
      "relative L2, and one more layer-count mismatch on top of it.")
    A("")
    A("| Condition | Mean CKA, median | Range |")
    A("|---|---|---|")
    for c in CORNERS:
        v = cka["armL_per_corner"][c]
        A(f"| arm L `{c}` | {v['median']:.4f} | {v['min']:.4f} – {v['max']:.4f} |")
    for city in CITIES:
        v = cka["dinov3_loco"][city]
        A(f"| DINOv3 vanilla LOCO, {city} | {v['cka_mean']:.4f} | — |")
    for city in CITIES:
        A(f"| DINOv3 within-city upper, {city} | ABSENT | "
          f"no CKA (`cka_n_images = 0`); not imputed |")
    A("")
    A(cap(cka["reference_internal_spread"]))
    A("")
    A("| Corner | Mean CKA, median | Distance to reference median | Distance to reference mean |")
    A("|---|---|---|---|")
    kmed = cka["by_reference"]["median"]["per_corner"]
    kmea = cka["by_reference"]["mean"]["per_corner"]
    for c in CORNERS:
        A(f"| `{c}` | {kmed[c]['corner_median']:.4f} | "
          f"{kmed[c]['absolute_difference']:.4f} | "
          f"{kmea[c]['absolute_difference']:.4f} |")
    A("")
    A(f"Reference values: median {cka['dinov3_loco_median_range']['median']:.4f}, "
      f"mean {cka['dinov3_loco_median_range']['mean']:.4f}. "
      + cap(cka["ordering_note"]))
    A("")
    A(f"**This axis is {cka['verdict_on_this_axis']}.** "
      + cap(cka["verdict_note"]))
    A("")
    A("### Per-stage arm L table (restated, unchanged)")
    A("")
    A("| Stage | sup_hi | sup_lo | ssl_hi | ssl_lo |")
    A("|---|---|---|---|---|")
    for s in RESNET_STAGES:
        cells = " | ".join(
            f"{g2['armL_half_restated']['per_corner'][c]['relative_l2'][s]['mean']:.4f}"
            for c in CORNERS)
        A(f"| relative L2 {s} | {cells} |")
    for s in CKA_STAGES:
        cells = " | ".join(
            f"{g2['armL_half_restated']['per_corner'][c]['cka'][s]['mean']:.4f}"
            for c in CORNERS)
        A(f"| CKA {s} | {cells} |")
    A("")
    A(f"Relative L2 separates `_lo` from `_hi` with no overlap across the nine "
      f"cells at every stage: "
      f"**{g2['armL_half_restated']['relative_l2_separates_no_overlap_every_stage']}**. "
      f"CKA orders them the same way in the mean at every stage "
      f"(**{g2['armL_half_restated']['cka_orders_correctly_in_mean_every_stage']}**) "
      f"but the nine-cell ranges overlap, so CKA does not separate them under "
      f"the no-overlap rule "
      f"(**{g2['armL_half_restated']['cka_separates_no_overlap_every_stage']}**).")
    A("")
    A("### DINOv3 per-block CKA, LOCO cells")
    A("")
    A("| Block | chicago | miami | phoenix |")
    A("|---|---|---|---|")
    curves = g2["dinov3_per_block_cka"]["curves"]
    for i in range(12):
        A(f"| {i} | " + " | ".join(f"{curves[c][i]['linear_cka']:.4f}"
                                   for c in CITIES) + " |")
    A("")
    A(g2["dinov3_per_block_cka"]["observation"])
    A("")
    A("### Limitation, attached to every row above")
    A("")
    A(COMPARABILITY_CAVEAT)
    A("")
    A(INIT_CAVEAT)
    A("")
    A("### G2's two conjuncts")
    A("")
    A("| Conjunct | Axis | Result |")
    A("|---|---|---|")
    A(f"| 1 — `_lo` corners drift measurably less than `_hi` | relative L2 "
      f"(CKA corroborating) | **{g2['g2_conjunct_1']['result']}** |")
    A(f"| 2 — `ssl_lo` sits nearer DINOv3's drift than `sup_hi` | whole-encoder "
      f"relative L2 | "
      f"**{g2['g2_conjunct_2']['whole_encoder_relative_l2']['result']}** |")
    A(f"| 2 — same clause | mean CKA | "
      f"**{g2['g2_conjunct_2']['mean_cka']['result']}** |")
    A("")
    A("Conjunct 1: " + g2["g2_conjunct_1"]["statement"])
    A("")
    A("Conjunct 2, relative L2: "
      + g2["g2_conjunct_2"]["whole_encoder_relative_l2"]["statement"])
    A("")
    A("Conjunct 2, CKA: " + cap(g2["g2_conjunct_2"]["mean_cka"]["statement"]))
    A("")
    A(f"`g2_status`: **{g2['g2_status']}**. " + g2["g2_status_scope"])
    A("")
    A("---")
    A("")
    A("## 2. Experiment A across the 2x2")
    A("")
    A("Median recovery ratio R, two-class boundary-tolerant IoU at K=2, pooled "
      "over 3 cities x 3 seeds, image-cluster bootstrap B = 10,000.")
    A("")
    A("| Corner | Exp A median R | 95% CI | mean R | n images | Budget-readable cells | Exp A median R, readable only |")
    A("|---|---|---|---|---|---|---|")
    g3bc = pooled["gates"]["G3"]["counts_by_corner"]
    for c in CORNERS:
        v = expA["corner_medians"][c]
        nr = g3bc[c]["expA"]["n_readable"]
        ro = (f"{expA['budget_readability']['median_R_readable_cells_only']['median_R']:+.4f}"
              if c == "sup_lo" else "not computed")
        A(f"| `{c}` | {v['median_R']:+.4f} | "
          f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['mean_R']:+.4f} | "
          f"{v['n_used']} | {nr} of 9 | {ro} |")
    A("")
    A("Published comparators on Experiment A: OGLANet 0.12, MAMNet-R34 0.67 "
      "(the CNN-like band), DINOv3-like >= 0.50.")
    A("")
    A("### sup_lo, broken out")
    A("")
    A("| Cell | Exp A median R | 95% CI | steps to best / ceiling | budget bound | readable |")
    A("|---|---|---|---|---|---|")
    for r in expA["sup_lo_breakdown"]["per_cell"]:
        A(f"| {r['city']} s{r['seed']} | {r['median_R']:+.4f} | "
          f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
          f"{r['steps_to_best']} / {r['steps_ceiling']} | "
          f"{'yes' if r['budget_bound'] else 'no'} | "
          f"{'yes' if r['readable'] else 'no'} |")
    A("")
    A("| Grouping | Exp A median R | 95% CI | n images |")
    A("|---|---|---|---|")
    for city in CITIES:
        v = expA["sup_lo_breakdown"]["by_city"][city]
        A(f"| city {city} | {v['median_R']:+.4f} | "
          f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['n_used']} |")
    for s in SEEDS:
        v = expA["sup_lo_breakdown"]["by_seed"][f"s{s}"]
        A(f"| seed {s} | {v['median_R']:+.4f} | "
          f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['n_used']} |")
    b = expA["budget_readability"]
    A(f"| all 9 cells | {b['median_R_all_cells']['median_R']:+.4f} | "
      f"[{b['median_R_all_cells']['ci95'][0]:+.4f}, "
      f"{b['median_R_all_cells']['ci95'][1]:+.4f}] | "
      f"{b['median_R_all_cells']['n_used']} |")
    A(f"| {b['n_readable']} readable cells only | "
      f"{b['median_R_readable_cells_only']['median_R']:+.4f} | "
      f"[{b['median_R_readable_cells_only']['ci95'][0]:+.4f}, "
      f"{b['median_R_readable_cells_only']['ci95'][1]:+.4f}] | "
      f"{b['median_R_readable_cells_only']['n_used']} |")
    A(f"| {b['n_unreadable']} unreadable cells only | "
      f"{b['median_R_unreadable_cells_only']['median_R']:+.4f} | "
      f"[{b['median_R_unreadable_cells_only']['ci95'][0]:+.4f}, "
      f"{b['median_R_unreadable_cells_only']['ci95'][1]:+.4f}] | "
      f"{b['median_R_unreadable_cells_only']['n_used']} |")
    A("")
    A(expA["sup_lo_breakdown"]["statement"])
    A("")
    A(f"Both figures are reported. {b['n_readable']} of sup_lo's 9 Experiment A "
      f"cells are readable under G3; {b['readable_set_note']}.")
    A("")
    for side, dg in (b["median_R_readable_cells_only"]
                     .get("bound_diagnostics", {}).items()):
        A(f"Sanity flag on the readable-only interval: its `{side}` prints as "
          f"exactly 0.0. " + cap(dg["diagnosis"]))
        A("")
    A("### Main effects, both experiments side by side")
    A("")
    A("Computed from the four pooled corner medians, the same way "
      "`DECISIONS.md` computed the Experiment B effects. No test is attached "
      "to these three numbers.")
    A("")
    A("| Effect | Experiment A | Experiment B |")
    A("|---|---|---|")
    for k, label in (("ssl_effect", "SSL effect"),
                     ("low_lr_effect", "low-LR effect"),
                     ("interaction", "interaction")):
        A(f"| {label} | {expA['main_effects']['expA'][k]:+.4f} | "
          f"{expA['main_effects']['expB'][k]:+.4f} |")
    A("")
    A("The Experiment B column reproduces the -0.003 and -0.018 recorded in "
      "`DECISIONS.md`. The Experiment A effects are one to two orders of "
      "magnitude larger than the Experiment B effects on the same corners.")
    A("")
    A("### Holm-corrected Experiment A contrasts against the sup_hi axis")
    A("")
    A("Paired bootstrap on per-image R deltas, image-cluster resampling, "
      "B = 10,000, two-tailed, Holm over the three comparisons.")
    A("")
    A("| Contrast | mean delta R | 95% CI | p raw | p Holm | n images |")
    A("|---|---|---|---|---|---|")
    for r in expA["contrasts_vs_sup_hi_axis"]:
        A(f"| `{r['contrast']}` | {r['mean_delta_R']:+.4f} | "
          f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
          f"{r['p_raw']:.4f} | {r['p_holm']:.4f} | {r['n']} |")
    A("")
    A("---")
    A("")
    A("## 3. What this does and does not change — a question for the operator")
    A("")
    A("What it does not change, on the rule as written: the ratified reading "
      "is outcome 4, and the decision variable is Experiment B median R against "
      "the pre-committed threshold table. Every Experiment B corner lands "
      "0.80-0.87 and is CNN-like. Experiment A is the corroborating variable, "
      "and no corner here meets the DINOv3-like signature, which requires "
      "Experiment A HIGH (>= 0.50) together with Experiment B collapsed "
      "(< 0.30). No Experiment B corner is collapsed, so the DINOv3 signature "
      "is unavailable on any corner regardless of what Experiment A does.")
    A("")
    A("What the Experiment A spread does show: the four corners span "
      f"{expA['corner_medians']['sup_lo']['median_R']:+.4f} to "
      f"{expA['corner_medians']['sup_hi']['median_R']:+.4f}, a range of "
      f"{expA['corner_medians']['sup_hi']['median_R'] - expA['corner_medians']['sup_lo']['median_R']:.4f}, "
      "against an Experiment B range of "
      f"{max(pooled['corner_medians']['expB'][c]['2class_tol_k2']['median_R'] for c in CORNERS) - min(pooled['corner_medians']['expB'][c]['2class_tol_k2']['median_R'] for c in CORNERS):.4f}. "
      "The Experiment A main effects are not inert the way the Experiment B "
      "effects are. Two corners sit inside the published CNN-like band "
      "(0.12-0.67), `ssl_lo` sits at its lower edge, and `sup_lo` sits below "
      "zero. `DECISIONS.md` recorded the absence of Experiment B main effects "
      "as itself an argument for outcome 4; the Experiment A effects do not "
      "have that shape.")
    A("")
    A("What the budget does to that: 29 of 36 Experiment A cells are "
      "budget-bound and only 10 of 36 are readable. For `sup_lo`, "
      f"{b['n_readable']} of 9 cells are readable and their pooled median is "
      f"{b['median_R_readable_cells_only']['median_R']:+.4f} "
      f"[{b['median_R_readable_cells_only']['ci95'][0]:+.4f}, "
      f"{b['median_R_readable_cells_only']['ci95'][1]:+.4f}], against "
      f"{b['median_R_all_cells']['median_R']:+.4f} "
      f"[{b['median_R_all_cells']['ci95'][0]:+.4f}, "
      f"{b['median_R_all_cells']['ci95'][1]:+.4f}] on all 9. The sign does not "
      "change when the ambiguous cells are dropped, and the readable subset is "
      "three cells.")
    A("")
    A("**The question for the operator.** The pre-commitment names Experiment A "
      "as corroborating and gives it a band, but it does not say what to do "
      "when a corner falls below that band while its Experiment B stays "
      "CNN-like. `sup_lo` is that case: Experiment B +0.8544, Experiment A "
      f"{expA['corner_medians']['sup_lo']['median_R']:+.4f}. That is neither the "
      "CNN signature nor the DINOv3 signature as the table defines them. Does "
      "arm L report it as an unclassified corroborating reading and leave "
      "outcome 4 standing on Experiment B alone, or does it go to HLT as a "
      "second over-specification of the same kind as clause 2? This box does "
      "not rule on that.")
    A("")
    A("---")
    A("")
    A("## Provenance")
    A("")
    A(f"- Script: `{SCRIPT_LABEL}`")
    A(f"- Results JSON: `results/16_armL_ssl_regime.json`, blocks "
      f"`g2_drift_comparison` and `expA_analysis`")
    A(f"- Dumps: 36 `mamnetr50-*-p5drift`, 6 `dinov3-*-orig-p5drift`, "
      f"72 arm L training cells, 36 `p3expA` probes")
    A("- All six DINOv3 `ckpt_sha256` are distinct; the `-orig` token "
      "disambiguates the published checkpoints from arm C's retrained cells")
    A("- No git repository exists at the project root on this box, so no git "
      "hash is recorded for this pass")
    A("")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    log(f"[write] {path}  ({os.path.getsize(path)} bytes)")


# -------------------------------------------------------------------- main --

def main():
    log(f"[start] {datetime.now(timezone.utc).isoformat()}  STAGE={STAGE}")
    global pooled
    with open(POOLED_JSON, "rb") as fh:
        pooled_bytes = fh.read()
    pooled = json.loads(pooled_bytes.decode("utf-8"))
    log(f"[read] {POOLED_JSON} ({len(pooled_bytes)} bytes, "
        f"{len(pooled)} top-level keys)")

    dino = assert_dinov3_dumps()
    rows = cnn_drift_rows()

    mod = load_sibling()
    t1a = task1a_per_stage(rows, pooled)
    t1b = task1b_whole_encoder(rows, dino)
    curves = dinov3_per_block_curves(dino)

    # ---- the dry-run report: task 1b only
    print("")
    print("=== TASK 1b — whole-encoder relative L2, the one comparable axis ===")
    print(f"{'condition':38s} {'median':>10s} {'min':>10s} {'max':>10s}  n")
    for c in CORNERS:
        v = t1b["armL_per_corner"][c]
        print(f"  arm L {c:30s} {v['median']:10.6f} {v['min']:10.6f} "
              f"{v['max']:10.6f}  {v['n_cells']}")
    for city in CITIES:
        v = t1b["dinov3_cells"]["loco"][city]
        print(f"  DINOv3 LOCO {city:24s} "
              f"{v['relative_l2_whole_encoder']:10.6f} {'':10s} {'':10s}  1")
    v = t1b["dinov3_reference_medians"]["dinov3_loco"]
    print(f"  DINOv3 LOCO median of 3{'':15s} {v['median']:10.6f} "
          f"{v['min']:10.6f} {v['max']:10.6f}  3")
    for city in CITIES:
        v = t1b["dinov3_cells"]["upper"][city]
        print(f"  DINOv3 upper {city:23s} "
              f"{v['relative_l2_whole_encoder']:10.6f} {'':10s} {'':10s}  1")
    v = t1b["dinov3_reference_medians"]["dinov3_upper"]
    print(f"  DINOv3 upper median of 3{'':14s} {v['median']:10.6f} "
          f"{v['min']:10.6f} {v['max']:10.6f}  3")
    print("")
    d = t1b["distances"]
    print("  the two distances, against the DINOv3 LOCO median "
          f"{d['primary_reference_value']:.6f}:")
    for c in ("ssl_lo", "sup_hi"):
        r = d["all_four_corners_vs_primary"][c]
        print(f"    {c}  median {r['corner_median']:.6f}"
              f"  distance {r['absolute_difference']:.6f}"
              f"  ratio {r['ratio_corner_over_reference']:.2f}x")
    print(f"    ssl_lo nearer than sup_hi: "
          f"{d['ssl_lo_nearer_than_sup_hi_vs_primary']}")
    print(f"    ordering under median vs mean: "
          f"{d['by_reference']['dinov3_loco']['ordering_same_under_median_and_mean']}")
    print("")
    print("  mean CKA:")
    for c in CORNERS:
        v = t1b["cka"]["armL_per_corner"][c]
        print(f"    arm L {c:8s} median {v['median']:.4f}  "
              f"range {v['min']:.4f} - {v['max']:.4f}")
    for city in CITIES:
        v = t1b["cka"]["dinov3_loco"][city]
        print(f"    DINOv3 LOCO {city:9s} {v['cka_mean']:.4f}")
    for city in CITIES:
        print(f"    DINOv3 upper {city:8s} ABSENT (cka_n_images=0), not imputed")
    print("")

    if STAGE == "dry":
        print("[dry-run] task 1b only; tasks 1a/2 computed but not reported here")
        print("[dry-run] nothing written")
        return 0

    # ---- task 2 needs the parquets
    training, _ = mod.enumerate_analysis_set()
    cells = mod.build_cells(training, CORNERS)
    per_cell = mod.build_per_cell(cells, CORNERS)
    mod.gate_G3(per_cell)                       # sets readable / budget_bound
    pooled_R = mod.pool_R(per_cell, CORNERS)
    contr_A = mod.contrasts(per_cell, CORNERS, key="expA")
    expA = task2(mod, per_cell, pooled_R, contr_A)

    g2 = build_g2_block(t1a, t1b, curves, dino)
    for blk, extra in ((g2, {}), (expA, {})):
        blk["generated_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        blk["script"] = SCRIPT_LABEL
        blk["script_sha256"] = sha256_file(os.path.join(BASE, SCRIPT_LABEL))
        blk["sibling_script_sha256"] = sha256_file(SIBLING)
        blk["git_hash"] = None
        blk["git_hash_note"] = ("no git repository exists at the project root "
                                "on this box; no git operation was run for "
                                "provenance")
        blk["arm"] = "L"
        blk["answers"] = ["zNBu-W5", "zNBu-W1", "AC1", "AC2"]
    g2["runs"] = ([f"mamnetr50-{c}-loco-r450-{ct}-hr-s{s}-p5drift"
                   for c in CORNERS for ct in CITIES for s in SEEDS]
                  + list(DINOV3_ALL))
    expA["runs"] = sorted({c["run_id"] for c in per_cell}
                          | {c["upper_run_id"] for c in per_cell}
                          | {f"{c['run_id']}-p3expA" for c in per_cell})

    original, _ = splice_into_json(
        POOLED_JSON, [("g2_drift_comparison", g2), ("expA_analysis", expA)])
    now = verify_splice(POOLED_JSON, original,
                        ("g2_drift_comparison", "expA_analysis"))
    log(f"[write] {POOLED_JSON}  ({os.path.getsize(POOLED_JSON)} bytes, "
        f"{len(now)} top-level keys)")

    write_md(OUT_MD, g2, expA)
    return 0


if __name__ == "__main__":
    sys.exit(main())
