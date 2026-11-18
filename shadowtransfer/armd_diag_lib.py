"""
Arm D / D_S3_diag — the diagnostics ride-along. Shared runtime.

Answers v1UJ-W4, zNBu-W2 and AC1. REVIEWS.md arm D.

The grid says the gap shrank on Phoenix. It does not say why. These are the
three §4.2-4.4 diagnostics, pre-specced as part of arm D and run as a separate
inference job after the grid, per CLAUDE.md, so a probe failure cannot cost a
trained model.

NOTHING HERE IS A NEW METRIC OR A NEW PROBE
-------------------------------------------
Every probe operation is imported from the arm C implementation and called
unchanged. This module resolves cells, pairs conditions and assembles the
results block; it does not reimplement a probe.

  probe   operation                          imported from
  ------  ---------------------------------  --------------------------------
  city    extract_features,                  shadowtransfer.armc_probes_ops
          city_identity_probe                  (App G.8 protocol, 5-fold
                                               GroupKFold, image-level
                                               permutation null)
  sp      aurc_fg / aurc_fg_mean20 columns   already written into every
                                             CONTRACT.md section 3 dump by
                                             shadowtransfer.armc_dump via
                                             metrics.aurc_trapz19 /
                                             metrics.aurc_mean20
  hist    source_cdfs, PerChannelMatcher,    shadowtransfer.armc_probes_ops
          GrayMatcher, recovery_ratio        shadowtransfer.armc_probes_common
  ci      bootstrap_ci                       shadowtransfer.metrics

Known erratum, carried not silently absorbed
--------------------------------------------
REVIEWS.md erratum E-8 records that App G.8's published probe taps the wrong
tensor on all three architectures and that its StandardScaler is fitted before
cross-validation. armc_probes_ops fixed both: it taps
ENCODER_FEATURE_KEY (feat5 / feat5 / feat_block11) and fits the scaler inside
each fold. The arm spec asks for "App G.8 protocol exactly", which is
ambiguous between the published code and the published prose. This module uses
armc_probes_ops as-is -- the arm C path, the prose protocol -- and records
`probe_protocol_deviations` on every block so the choice is visible rather
than assumed. Accuracies here are therefore NOT comparable to the published
0.33 / 0.43 / 0.45.

Erratum E-8 also records that a raw-pixel control using only per-channel mean
and std reaches 0.871. That control is python/armc_probes_rawpixel.py and is
not re-run here.
"""

import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shadowtransfer import armc_probes_common as C                # noqa: E402
from shadowtransfer import armc_probes_ops as OPS                 # noqa: E402
from shadowtransfer import metrics as M                           # noqa: E402
from shadowtransfer.armc_dump import sha256_file                  # noqa: E402

ARM = "D"
ARM_ID = "D_S3_diag"
WEAKNESS_IDS = ["v1UJ-W4", "zNBu-W2", "AC1"]
RUNG = 450
RES = "hr"
CITIES = ("chicago", "miami", "phoenix")
AUG_SUFFIX = "-e200"
BOOTSTRAP_B = 1000
METRIC_NAME = "miou_2class_tolerant_K2"
SECONDARY_METRIC_NAME = "iou_fg_tolerant_K2"

DUMP_ROOT = BASE / "dumps" / "st"

AURC_CAVEAT = (
    "CONTRACT.md section 3 leaves the canonical AURC definition BLANK pending "
    "the P0a discriminating test, and neither live candidate reproduces the "
    "published Fig 3(a) +0.31. Both are emitted here: aurc_fg is "
    "metrics.aurc_trapz19 (np.trapezoid over a 19-point coverage grid "
    "spanning 0.9, the composition_analysis definition) and aurc_fg_mean20 is "
    "metrics.aurc_mean20 (plain mean over a 20-point grid, the SIB-trainer "
    "definition). The headline used below is aurc_trapz19. Neither is "
    "declared canonical by this run and no value here is compared against any "
    "published AURC number.")

PROBE_PROTOCOL_DEVIATIONS = [
    "Encoder tensor: armc_probes_ops taps ENCODER_FEATURE_KEY (mamnet feat5, "
    "oglanet feat5, dinov3 feat_block11). REVIEWS.md erratum E-8 records that "
    "the published App G.8 code tapped feat4 and blocks.9 instead. This is "
    "the corrected tensor, which is what App G.8's prose describes.",
    "StandardScaler is fitted inside each fold. Erratum E-8 records that the "
    "published thread1_1d.py:171 fitted it on all data before "
    "cross-validation, a leak that inflated the published number.",
    "Spatial cells are subsampled per image (seeded, 32 of the map) because "
    "the permutation null refits n_folds x n_permutations logistic "
    "regressions. The same sample is used for the observed fit and for every "
    "permutation.",
    "Accuracies here are therefore NOT comparable to the published 0.33 / "
    "0.43 / 0.45 for dinov3 / mamnet / oglanet.",
]


# ---------------------------------------------------------------------------
# Cell resolution
# ---------------------------------------------------------------------------

def aug_base_rid(arch, mode, city, seed):
    return f"{arch}-augs3-{mode}-r{RUNG}-{city}-{RES}-s{seed}"


def van_rid(arch, mode, city, seed):
    method = "vanilla" if mode == "loco" else "none"
    return f"{arch}-{method}-{mode}-r{RUNG}-{city}-{RES}-s{seed}"


def done(rid):
    return (DUMP_ROOT / rid / ".DONE").exists()


def resolve_aug(arch, mode, city, seed):
    """The aug cell to use, preferring the -e200 variant. Returns
    (run_id, model_variant) or (None, None)."""
    base = aug_base_rid(arch, mode, city, seed)
    if done(base + AUG_SUFFIX):
        return base + AUG_SUFFIX, "e200"
    if done(base):
        return base, "100epoch"
    return None, None


def discover_groups():
    """Every (arch, city, seed) that carries at least one completed arm D aug
    cell. The group, not the cell, is the job unit: a group's vanilla LOCO
    comparator is probed ONCE and shared by its aug-loco and aug-upper cells,
    instead of once per aug cell."""
    import re
    groups = {}
    for p in sorted(DUMP_ROOT.iterdir()):
        m = re.fullmatch(r"(\w+)-augs3-(loco|upper)-r450-(\w+)-hr-s(\d+)"
                         r"(?:-e200)?", p.name)
        if not m or not (p / ".DONE").exists():
            continue
        arch, mode, city, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        g = groups.setdefault((arch, city, seed), set())
        g.add(mode)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def group_cells(arch, city, seed):
    """Everything a group's job needs, resolved and checked."""
    al, al_var = resolve_aug(arch, "loco", city, seed)
    au, au_var = resolve_aug(arch, "upper", city, seed)
    vl = van_rid(arch, "loco", city, seed)
    vu = van_rid(arch, "upper", city, seed)
    out = {
        "arch": arch, "city": city, "seed": seed, "rung": RUNG, "res": RES,
        "aug_loco": al, "aug_loco_variant": al_var,
        "aug_upper": au, "aug_upper_variant": au_var,
        "vanilla_loco": vl if done(vl) else None,
        "vanilla_upper": vu if done(vu) else None,
        "missing": [],
    }
    for k, v in (("vanilla_loco", vl), ("vanilla_upper", vu)):
        if out[k] is None:
            out["missing"].append(v)
    if al is None and au is None:
        out["missing"].append(f"{arch}-augs3-*-r{RUNG}-{city}-{RES}-s{seed}")
    return out


def run_json(rid):
    return json.loads((DUMP_ROOT / rid / "run.json").read_text())


def perimage(rid):
    import pandas as pd
    return pd.read_parquet(DUMP_ROOT / rid / "perimage.parquet")


def checkpoint(rid):
    p = Path(run_json(rid)["ckpt_path"])
    if not p.exists():
        raise SystemExit(f"FATAL checkpoint recorded by {rid} is missing: {p}")
    return p


def module_shas():
    return {
        "metrics_sha256": sha256_file(BASE / "shadowtransfer/metrics.py"),
        "probes_common_sha": sha256_file(BASE / "shadowtransfer/armc_probes_common.py"),
        "probes_ops_sha": sha256_file(BASE / "shadowtransfer/armc_probes_ops.py"),
        "diag_lib_sha": sha256_file(Path(__file__).resolve()),
        "dump_module_sha": sha256_file(BASE / "shadowtransfer/armc_dump.py"),
    }


# ---------------------------------------------------------------------------
# Probe 2 — foreground selective prediction / AURC. Parquet only, no GPU.
# ---------------------------------------------------------------------------

def paired_bootstrap_delta(a, b, n_boot=BOOTSTRAP_B, seed=0, ci=0.95):
    """Image-level paired bootstrap of mean(a) - mean(b).

    a and b are aligned per-image arrays over the SAME 150 held-out-city test
    images, so the resample index is shared: LOCO and upper are evaluated on
    one test pool and the pairing is the whole point of the interval.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise SystemExit(f"FATAL paired bootstrap on unequal lengths "
                         f"{len(a)} vs {len(b)}")
    d = a - b
    rng = np.random.RandomState(seed)
    n = len(d)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = d[idx].mean()
    boots.sort()
    alpha = (1 - ci) / 2
    return {
        "statistic": float(d.mean()),
        "ci95": [float(boots[int(alpha * n_boot)]),
                 float(boots[int((1 - alpha) * n_boot) - 1])],
        "ci_method": f"image_level_paired_bootstrap_B{n_boot}",
        "B": n_boot,
        "n": int(n),
        "seed": seed,
    }


def sp_gap(loco_rid, upper_rid, column, seed=0):
    """AURC gap = mean AURC under LOCO minus mean AURC under the upper bound,
    on the foreground pixel population, paired by image_id.

    CONVENTIONS.md fixes the name: SP gap. Positive means the LOCO model's
    risk-coverage curve integrates higher, i.e. its confidence ranks
    foreground errors worse than the within-city model's.
    """
    dl, du = perimage(loco_rid), perimage(upper_rid)
    a = dl[column].to_numpy(dtype=np.float64)
    b = C.align_on_image_id(dl, du, column)
    ok = np.isfinite(a) & np.isfinite(b)
    out = paired_bootstrap_delta(a[ok], b[ok], seed=seed)
    out.update({
        "loco_run_id": loco_rid, "upper_run_id": upper_rid,
        "column": column,
        "loco_mean": float(a[ok].mean()), "upper_mean": float(b[ok].mean()),
        "n_images_total": int(len(a)),
        "n_images_nonfinite_dropped": int((~ok).sum()),
    })
    return out


def sp_block(cells, seed=0):
    """Both conditions, both AURC definitions, and the aug-minus-vanilla
    difference of the SP gaps."""
    cols = {"aurc_trapz19": "aurc_fg", "aurc_mean20": "aurc_fg_mean20"}
    out = {"definitions": cols, "headline_definition": "aurc_trapz19",
           "population": "foreground", "aurc_caveat": AURC_CAVEAT,
           "conditions": {}, "difference": {}, "skipped": []}
    pairs = {"aug": (cells["aug_loco"], cells["aug_upper"]),
             "vanilla": (cells["vanilla_loco"], cells["vanilla_upper"])}
    for cond, (lo, up) in pairs.items():
        if not lo or not up:
            out["skipped"].append(
                {"condition": cond, "reason": f"loco={lo} upper={up}"})
            continue
        out["conditions"][cond] = {
            name: sp_gap(lo, up, col, seed=seed) for name, col in cols.items()}
    for name in cols:
        a = out["conditions"].get("aug", {}).get(name)
        v = out["conditions"].get("vanilla", {}).get(name)
        if a and v:
            out["difference"][name] = {
                "aug_sp_gap_minus_vanilla_sp_gap":
                    round(a["statistic"] - v["statistic"], 8),
                "aug_sp_gap": a["statistic"],
                "vanilla_sp_gap": v["statistic"],
                "note": ("difference of two paired statistics measured on the "
                         "same 150 images but with different model pairs; no "
                         "joint interval is computed for it here."),
            }
    return out


# ---------------------------------------------------------------------------
# Probe 3 — histogram matching. Median R with an image-cluster bootstrap CI.
# ---------------------------------------------------------------------------

def r_summary(r, seed=0, n_boot=BOOTSTRAP_B):
    """Median per-image R with a 95% bootstrap CI over images.

    The resample unit is the image, which is the cluster: R is one value per
    image and nothing below the image is resampled. metrics.bootstrap_ci is
    the only implementation called.
    """
    r = np.asarray(r, dtype=np.float64)
    ok = np.isfinite(r)
    if ok.sum() == 0:
        return {"median": None, "ci95": [None, None], "n": 0,
                "ci_method": f"image_cluster_bootstrap_B{n_boot}"}
    med, lo, hi = M.bootstrap_ci(r[ok], stat_fn=np.median, n_boot=n_boot,
                                 ci=0.95, seed=seed)
    return {
        "median": float(med), "ci95": [float(lo), float(hi)],
        "mean": float(r[ok].mean()), "n": int(ok.sum()),
        "n_excluded_headroom": int((~ok).sum()),
        "ci_method": f"image_cluster_bootstrap_B{n_boot}",
        "stat_fn": "np.median", "B": n_boot, "seed": seed,
    }
