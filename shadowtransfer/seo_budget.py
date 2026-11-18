#!/usr/bin/env python3
"""
Arm A / S-EO — budget-matched, AOI-aware subsampling.

Every cell in arm A trains on the SAME number of patches so that "external
LOCO gap" cannot be read as "the LOCO cell simply saw less data". Fixed for
the whole arm:

    n_train = 3500 patches        n_val = 900 patches
    n_test  = the held-out city's FULL test split, never subsampled

The unit of the quota is the PATCH, but the unit of SELECTION is the AOI, for
the same reason splits are AOI-level: with acq_per_aoi=3 the same 500x500 m
footprint appears three times under different sun positions, so patches inside
an AOI are near-duplicates. Taking patches at random would put two views of one
roof on both sides of a train/val boundary.

The rule, verbatim
------------------
Within a (city, split):
  1. rank AOIs by (sha256("ShadowTransfer/armA/S-EO/v1|<aoi_id>")[:8] as uint64,
     aoi_id) -- byte-for-byte the rule armA_seo_splits.py used, so the budget
     ordering is a refinement of the split ordering rather than a second,
     independent draw;
  2. take WHOLE AOIs in that order while the running total plus the next AOI's
     patch count stays <= the quota;
  3. the next AOI is the BOUNDARY AOI: take its first (quota - running) patches
     in deterministic index order (image_id ascending, then the patch grid order
     stored in the patch index);
  4. stop. Exactly one AOI per (city, split, quota) is partial, and its id and
     count are recorded.

Per condition, for a fold holding out city H with source cities S1, S2:
    loco   train 1750 from S1 + 1750 from S2      val 450 + 450
    upper  train 3500 from H                      val 900
    test   H's full test split, byte-identical between the two cells

Nothing here reads an image. It reads the patch index and the AOI split
manifests and emits patch id lists.
"""
import hashlib
import json
from pathlib import Path

SPLIT_SALT = "ShadowTransfer/armA/S-EO/v1"

N_TRAIN = 3500
N_VAL = 900

BUDGET_RULE = (
    f"{SPLIT_SALT}|sha256-first8|rank-AOI-whole-then-one-partial|"
    f"n_train={N_TRAIN}|n_val={N_VAL}|test=full|"
    "partial-AOI-patch-order=(image_id asc, patch-index-order)"
)


def budget_rule_sha():
    return hashlib.sha256(BUDGET_RULE.encode()).hexdigest()


def hash_u(aoi_id):
    """uint64 from the first 8 bytes of sha256(SALT|aoi_id). Identical to
    armA_seo_splits.hash_u -- do not re-derive it differently here."""
    h = hashlib.sha256(f"{SPLIT_SALT}|{aoi_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def patch_id(image_id, y, x):
    """Matches SEODataset's `filename`, which is the dump's image_id."""
    return f"{image_id}_y{y}_x{x}"


def aoi_patches(patch_index, aoi_ids):
    """{aoi_id: [patch_id, ...]} in deterministic order.

    Order inside an AOI is image_id ascending, then the patch grid order the
    patch index stores (row-major over the centred crop grid). That is a total
    order fixed by the prep run, so the boundary AOI's prefix is reproducible.
    """
    want = set(aoi_ids)
    out = {a: [] for a in want}
    for image_id in sorted(patch_index["images"]):
        rec = patch_index["images"][image_id]
        if rec["aoi_id"] not in want:
            continue
        out[rec["aoi_id"]].extend(patch_id(image_id, y, x) for (y, x) in rec["patches"])
    return out


def take_budget(patch_index, aoi_ids, quota, label):
    """Whole AOIs in rank order, then one partial AOI. Returns a dict."""
    per_aoi = aoi_patches(patch_index, aoi_ids)
    ordered = sorted(aoi_ids, key=lambda a: (hash_u(a), a))
    total = sum(len(per_aoi[a]) for a in ordered)
    if total < quota:
        raise SystemExit(
            f"FATAL {label}: quota {quota} exceeds the {total} patches available "
            f"across {len(ordered)} AOIs. The budget cannot be met from this pool.")

    picked, aois_used, running = [], [], 0
    partial_id, partial_n = None, 0
    for a in ordered:
        n = len(per_aoi[a])
        if running + n <= quota:
            picked.extend(per_aoi[a])
            aois_used.append(a)
            running += n
            if running == quota:
                break
        else:
            partial_id, partial_n = a, quota - running
            picked.extend(per_aoi[a][:partial_n])
            aois_used.append(a)
            running = quota
            break

    if running != quota:
        raise SystemExit(f"FATAL {label}: took {running} patches, quota {quota}")
    if len(picked) != quota:
        raise SystemExit(f"FATAL {label}: list is {len(picked)}, quota {quota}")
    if len(set(picked)) != quota:
        raise SystemExit(f"FATAL {label}: duplicate patch ids in the selection")
    return {
        "patches": picked,
        "aois_used": aois_used,
        "n_aoi_used": len(aois_used),
        "n_aoi_available": len(ordered),
        "n_patches_available": total,
        "partial_aoi_id": partial_id,
        "partial_aoi_n": partial_n,
        "partial_aoi_total": (len(per_aoi[partial_id]) if partial_id else 0),
        "quota_fraction_of_pool": round(quota / total, 6),
    }


def full_split(patch_index, aoi_ids, label):
    """The whole pool, unsubsampled. Used for test."""
    per_aoi = aoi_patches(patch_index, aoi_ids)
    ordered = sorted(aoi_ids, key=lambda a: (hash_u(a), a))
    picked = [p for a in ordered for p in per_aoi[a]]
    if len(set(picked)) != len(picked):
        raise SystemExit(f"FATAL {label}: duplicate patch ids")
    return {"patches": picked, "aois_used": ordered, "n_aoi_used": len(ordered)}


def build_cell(patch_index, splits, holdout, condition, source_cities):
    """One cell's manifest dict. condition is 'loco' or 'upper'."""
    if condition == "loco":
        per_city_train = N_TRAIN // len(source_cities)
        per_city_val = N_VAL // len(source_cities)
        if per_city_train * len(source_cities) != N_TRAIN:
            raise SystemExit("FATAL n_train does not divide evenly across sources")
        if per_city_val * len(source_cities) != N_VAL:
            raise SystemExit("FATAL n_val does not divide evenly across sources")
        draw = [(c, per_city_train, per_city_val) for c in source_cities]
    elif condition == "upper":
        draw = [(holdout, N_TRAIN, N_VAL)]
    else:
        raise SystemExit(f"FATAL unknown condition {condition}")

    train, val, detail = [], [], {}
    for city, n_tr, n_va in draw:
        for split, quota, acc in (("train", n_tr, train), ("val", n_va, val)):
            r = take_budget(patch_index, splits[city][split], quota,
                            f"{holdout}/{condition}/{city}/{split}")
            acc.extend(r["patches"])
            detail[f"{city}/{split}"] = {k: v for k, v in r.items() if k != "patches"}

    test = full_split(patch_index, splits[holdout]["test"], f"{holdout}/test")

    train_aois = {a for k, d in detail.items() if k.endswith("/train")
                  for a in d["aois_used"]}
    val_aois = {a for k, d in detail.items() if k.endswith("/val")
                for a in d["aois_used"]}
    overlap = train_aois & val_aois
    if overlap:
        raise SystemExit(f"FATAL {holdout}/{condition}: AOIs span train and val: "
                         f"{sorted(overlap)[:5]}")
    if set(train) & set(val) or set(train) & set(test["patches"]) \
            or set(val) & set(test["patches"]):
        raise SystemExit(f"FATAL {holdout}/{condition}: patch id overlap between splits")

    return {
        "arm": "A", "dataset": "seo", "fold": holdout, "condition": condition,
        "holdout": holdout, "source_cities": list(source_cities),
        "train": train, "val": val, "test": test["patches"],
        "n_train": len(train), "n_val": len(val), "n_test": len(test["patches"]),
        "n_test_aoi": test["n_aoi_used"],
        "aois_used": {k: d["aois_used"] for k, d in detail.items()},
        "partial_aoi_id": {k: d["partial_aoi_id"] for k, d in detail.items()},
        "partial_aoi_n": {k: d["partial_aoi_n"] for k, d in detail.items()},
        "per_quota_detail": detail,
        "test_aois": test["aois_used"],
        "budget_rule": BUDGET_RULE,
        "budget_rule_sha": budget_rule_sha(),
        "split_rule_sha": splits[holdout]["split_rule_sha"],
        "test_note": ("the held-out city's FULL test split; never subsampled and "
                      "byte-identical between this fold's loco and upper cells"),
    }
