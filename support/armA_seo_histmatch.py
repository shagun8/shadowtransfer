#!/usr/bin/env python3
"""
Arm A / S-EO -- the section 4.2 histogram-matching probe over the 6 vanilla
LOCO cells. INFERENCE ONLY.

No training, no new checkpoint, no method cell, no Experiment A, nothing
AURC-shaped beyond the CONTRACT.md 3 columns `seo_dump` already emits. This
script loads a finished vanilla cell's `checkpoint_best.pth`, runs its test
split three times, and writes one dump.

What it answers
---------------
REVIEWS.md v1UJ-W4 / zNBu-W2: the section 4.2 reading is that test-time
histogram matching does not recover the LOCO gap. Arm C measured that on
ShadowTransfer. This is the external replication on S-EO.

Comparability with arm C is the whole point of the run
------------------------------------------------------
The matching rule and the R statistic are IMPORTED from arm C's own probe
modules, never reimplemented:

    armc_probes_ops._mapping             argmin_s |cdf_src[s] - cdf_tgt[t]|
    armc_probes_ops.match_gray           the PUBLISHED grayscale-plus-ratio rule
    armc_probes_ops.match_per_channel    arm C's per-channel headline variant
    armc_probes_common.recovery_ratio    per-image R, excluded at headroom < 0.02
    armc_probes_common.align_on_image_id

`probes_ops_sha` and `probes_common_sha` are written into run.json from those
same files, beside the arm C values, so the two arms are checkable against each
other rather than merely asserted to agree.

The PUBLISHED rule is the grayscale one
(`final_loco/experiment_utils.histogram_match_image`), and it is this probe's
headline because the arm C statistic being replicated was measured under it.
Arm C's per-channel variant rides along in `iou_tol_matched` /
`recovery_ratio_R`, column for column with arm C.

The one S-EO difference, and it is a DATA difference, not a rule difference
---------------------------------------------------------------------------
S-EO carries a DSM-unseen mask. Those pixels are outside the metric and outside
the loss everywhere else in arm A, so they are outside the matching STATISTICS
here: a histogram accumulated over pixels the geometry never determined would
let DSM coverage -- which differs several-fold between JAX, OMA and UCSD --
drive the intervention, unequally across the three folds. The mapping is still
APPLIED to every pixel; the network is handed a whole patch.

Mechanically that is `match_gray` / `match_per_channel` with `[valid]` inserted
into one bincount and nothing else touched. Where a patch is ENTIRELY unseen
there is no valid-restricted histogram at all, and the target CDF falls back to
the whole patch -- the arm C rule verbatim. That is metric-neutral, because such
a patch is already dropped from every confusion cell. `assert_rule_identity()` PROVES it
at startup: on random uint8 patches with `valid` all-True, the valid-aware
functions must return arrays byte-identical to the arm C originals. It runs on
every job and lands in run.json as `rule_identity_check`.

S-EO radiometry is histogram-equalized AT SOURCE
------------------------------------------------
The pansharpened crops are colour-corrected and histogram-equalized by the
dataset authors (dataset card, quoted in `seo_dataset.py`). Nothing here
corrects for it. It is a property of the data and part of what makes the
replication informative. Stated in run.json as `source_radiometry_note` and in
the probe manifest; not adjusted for.

Where it writes
---------------
`${ST_SCRATCH_ROOT}/dumps/seo/{run_id}-p1histmatch/`. the archive root is
archive-only since the 2026-08-03 quota block (Errno 122).
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
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import armA_pilot_train as P                                      # noqa: E402
from shadowtransfer import armc_runtime                           # noqa: E402
from shadowtransfer import armc_probes_common as C                # noqa: E402
from shadowtransfer import armc_probes_ops as OPS                 # noqa: E402
from shadowtransfer import seo_budget                             # noqa: E402
from shadowtransfer import seo_dump                               # noqa: E402
from shadowtransfer.armc_dump import sha256_file                  # noqa: E402
from shadowtransfer.seo_dataset import SEODataset, UNSEEN_PRODUCT  # noqa: E402

DATASET = "seo"
PROBE_SUFFIX = "p1histmatch"
CITY_OF_HOLDOUT = {"JAX": "jax", "OMA": "oma", "UCSD": "ucsd"}
ARCHS = ("mamnet", "dinov3", "oglanet", "crossearth")

#: D-A2-2 (operator, 2026-08-21): the section 4.2 probe extends to the method
#: cells and to OGLANet and CrossEarth. Implemented 2026-09-22 -- until then
#: this script hardcoded `"method": "vanilla"` and ARCHS was mamnet+dinov3,
#: which is why block 5 could not run.
#:
#: THE TWO FAMILIES NEVER MERGE. D-A2-2's operative half:
#:   vanilla family -- vanilla LOCO cells. May join the headline 0-of-N count.
#:   method family  -- fda / sib / wave-2 method LOCO cells. A SEPARATE family,
#:                     reported beside the headline and NEVER pooled into it.
#: The reason is not bookkeeping. R is defined on a (loco, upper) pair and
#: methods are LOCO-only project-wide, so there is no method upper cell. A
#: method cell must borrow the same fold/seed VANILLA upper as its denominator,
#: making its R a method numerator over a vanilla denominator -- a different
#: quantity from the R arm C published. Every method-family dump carries that
#: fact in its own run.json so a reader who meets one of these rows without
#: DECISIONS_armA_wave2.md still cannot mistake it for a vanilla R.
VANILLA_FAMILY_METHOD = "vanilla"


def probe_family(method):
    """'vanilla' or 'method'. The only place the split is decided."""
    return "vanilla" if method == VANILLA_FAMILY_METHOD else "method"

#: Arm C's headroom exclusion, imported rather than restated.
R_EXCLUDE_MIN = C.R_EXCLUDE_MIN

#: The baseline pass must reproduce the parent cell's stored dump. MAMNet and
#: DINOv3 are bit-exact under the arm's E-9 controls, so this is a hard gate and
#: not a tolerance to be widened: a mismatch is a finding.
REPRO_TOL_MIOU_PCT = 1e-4

PRED_POS_MIN, PRED_POS_MAX = 0.001, 0.9


# ---------------------------------------------------------------------------
# The matching rule -- arm C's, with the histogram restricted to valid pixels
# ---------------------------------------------------------------------------

def _cdf(hist):
    tot = hist.sum()
    if tot <= 0:
        raise SystemExit("FATAL empty histogram: no valid pixel contributed")
    return np.cumsum(hist / tot)


def match_gray_valid(img, src_gray_cdf, valid):
    """`OPS.match_gray` with the per-image target CDF taken over `valid` only.

    Line for line the arm C function. The single edit is `[valid]` inside the
    bincount; the uint8 truncation cast, the published `_mapping`, the per-pixel
    channel ratio, the 1e-6 guard and the clip are untouched, and
    `assert_rule_identity` checks that at startup.
    """
    gray = img.mean(axis=2)
    g8 = gray.astype(np.uint8)
    h = np.bincount(g8[valid].ravel(), minlength=256).astype(np.float64)
    if h.sum() == 0:
        # Every pixel of this patch is DSM-unseen, so there is no
        # valid-restricted statistic to take. Fall back to the whole patch --
        # which IS the arm C rule, because arm C has no valid mask at all.
        # Metric-neutral by construction: a patch with no seen pixel is dropped
        # from all four confusion cells at every band width, so no reported
        # number can move. 29 OMA and 79 UCSD test patches are in this state and
        # job 20821985 died on the first of them. Counted per cell in run.json
        # as n_patches_valid_mask_empty.
        h = np.bincount(g8.ravel(), minlength=256).astype(np.float64)
    mapping = OPS._mapping(src_gray_cdf, _cdf(h))

    matched = np.zeros_like(img)
    gray_f = gray.astype(np.float32)
    mapped_gray = mapping[g8.ravel()].reshape(gray.shape).astype(np.float32)
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        ratio = np.where(gray_f > 0, channel / (gray_f + 1e-6), 1.0)
        matched[:, :, c] = np.clip(mapped_gray * ratio, 0, 255).astype(np.uint8)
    return matched


def match_per_channel_valid(img, src_cdf, valid):
    """`OPS.match_per_channel` with each channel's target CDF over `valid` only."""
    out = np.empty_like(img)
    for c in range(3):
        h = np.bincount(img[:, :, c][valid].ravel(), minlength=256).astype(np.float64)
        if h.sum() == 0:                  # fully unseen; see match_gray_valid
            h = np.bincount(img[:, :, c].ravel(), minlength=256).astype(np.float64)
        out[:, :, c] = OPS._mapping(src_cdf[c], _cdf(h))[img[:, :, c]]
    return out


def assert_rule_identity(n=8, seed=0):
    """With `valid` all-True the valid-aware rule must be the arm C rule, exactly.

    Byte-identity, not closeness. This is the evidence that the S-EO probe is
    the published intervention and not a variant of it, and it is cheap enough
    to run on every job rather than once in a session log.
    """
    rng = np.random.default_rng(seed)
    src_gray = np.cumsum(rng.random(256))
    src_gray /= src_gray[-1]
    src_ch = np.cumsum(rng.random((3, 256)), axis=1)
    src_ch /= src_ch[:, -1:]
    for _ in range(n):
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        allv = np.ones(img.shape[:2], dtype=bool)
        if not np.array_equal(match_gray_valid(img, src_gray, allv),
                              OPS.match_gray(img, src_gray)):
            raise SystemExit("FATAL grayscale rule diverges from armc_probes_ops")
        if not np.array_equal(match_per_channel_valid(img, src_ch, allv),
                              OPS.match_per_channel(img, src_ch)):
            raise SystemExit("FATAL per-channel rule diverges from armc_probes_ops")
        # The fully-unseen patch. With an EMPTY valid mask the fallback must
        # reproduce the arm C rule exactly, because the arm C rule is the
        # whole-patch one. This is the case that killed job 20821985.
        nonev = np.zeros(img.shape[:2], dtype=bool)
        if not np.array_equal(match_gray_valid(img, src_gray, nonev),
                              OPS.match_gray(img, src_gray)):
            raise SystemExit("FATAL empty-valid-mask grayscale fallback is not "
                             "the arm C rule")
        if not np.array_equal(match_per_channel_valid(img, src_ch, nonev),
                              OPS.match_per_channel(img, src_ch)):
            raise SystemExit("FATAL empty-valid-mask per-channel fallback is not "
                             "the arm C rule")
    return {
        "empty_valid_mask_fallback_checked": True,
        "checked": True, "n_random_patches": n, "identical_to_armC": True,
        "what": ("match_gray_valid / match_per_channel_valid return arrays "
                 "BYTE-IDENTICAL to armc_probes_ops.match_gray / "
                 "match_per_channel when the valid mask is all-True; the only "
                 "difference in the S-EO probe is which pixels enter the "
                 "histogram, never how the mapping is built or applied"),
    }


class SEOGrayMatcher:
    """The PUBLISHED rule as a dataset hook. Arm C's `GrayMatcher`, valid-aware."""

    def __init__(self, src_gray_cdf):
        self.src_gray_cdf = src_gray_cdf

    def __call__(self, arr, valid):
        return match_gray_valid(arr, self.src_gray_cdf, valid)


class SEOPerChannelMatcher:
    """Arm C's per-channel headline variant, valid-aware."""

    def __init__(self, src_cdf):
        self.src_cdf = src_cdf

    def __call__(self, arr, valid):
        return match_per_channel_valid(arr, self.src_cdf, valid)


# ---------------------------------------------------------------------------
# The injection point
# ---------------------------------------------------------------------------

class MatchedSEODataset(SEODataset):
    """`SEODataset` with a matcher on the raw uint8 RGB crop.

    Injection is `_to_tensor`, which the parent calls on the crop straight out
    of `_crop`: BEFORE the RMS-contrast channel, BEFORE ImageNet normalisation
    and BEFORE any flip. That is arm C's `fda_transform` point exactly -- the
    only place where matched pixels then travel through the published
    preprocessing unchanged.

    `matcher = None` makes this the parent class in every respect, so the
    baseline pass is not a different loader from the one that produced the
    parent cell's dump; `armA_pilot_train.build_split` is the same function,
    reached through a module-level rebind of the name it constructs.

    `_cur` carries the row index into `_to_tensor` because the matcher needs the
    patch's valid mask, which the parent computes later. A DataLoader worker
    holds its own copy of the dataset and runs `__getitem__` to completion
    before the next, so the stash is not shared across concurrent items.
    """

    matcher = None
    _cur = None

    def _unseen_valid(self, i):
        r = self.rows[i]
        v = np.ones((self.patch, self.patch), dtype=bool)
        rel = r["aux"].get(UNSEEN_PRODUCT)
        if self.exclude_unseen and rel:
            flag = self._crop(os.path.join(self.mask_root, rel),
                              r["y"], r["x"], self.patch, 1)[:, :, 0]
            v &= (flag == 0)
        return v

    def __getitem__(self, i):
        self._cur = i
        return super().__getitem__(i)

    def _to_tensor(self, arr):
        if self.matcher is not None:
            valid = self._unseen_valid(self._cur)
            arr = self.matcher(np.ascontiguousarray(arr[:, :, :3]), valid)
        return super()._to_tensor(arr)


def set_matcher(bases, matcher):
    for b in bases:
        b.matcher = matcher


# ---------------------------------------------------------------------------
# The source statistic
# ---------------------------------------------------------------------------

def source_cdfs_seo(rows, img_root, mask_root, patch, exclude_unseen=True,
                    limit=0, label=""):
    """Per-channel and grayscale CDFs over a cell's OWN training patches.

    The pool is the cell's training list -- the budget's 3500 patches from the
    two source cities, not the cities' whole pools -- which is what arm C's
    `source_cdfs` uses at its rung, and it is `source_only` access by
    construction: no held-out-city pixel is opened here.

    DSM-unseen pixels do not enter either histogram (module docstring). The
    grayscale statistic keeps `experiment_utils.compute_histogram`'s uint8 cast
    of the RGB mean, truncation and all.

    Patches of one source image are contiguous in `rows`, so the full PNG is
    decoded once and sliced; that is the difference between a few minutes and
    an hour over 3500 patches.
    """
    t0 = time.time()
    ch = np.zeros((3, 256), dtype=np.float64)
    gy = np.zeros(256, dtype=np.float64)
    use = rows[:limit] if limit else rows
    cache_img = {"path": None, "arr": None}
    cache_msk = {"path": None, "arr": None}
    n_px_total = n_px_valid = 0

    def load(cache, path):
        if cache["path"] != path:
            with Image.open(path) as im:
                cache["arr"] = np.array(im)
            cache["path"] = path
        return cache["arr"]

    for r in use:
        y, x = r["y"], r["x"]
        full = load(cache_img, os.path.join(img_root, r["image"]))
        arr = np.ascontiguousarray(full[y:y + patch, x:x + patch, :3])
        if arr.dtype != np.uint8:
            raise SystemExit(
                f"FATAL {r['image']}: source pool image is {arr.dtype}, not uint8. "
                "The published matching rule is defined on 256 uint8 bins and "
                "cannot be applied as written; STOPPING rather than improvising "
                "a value-range variant.")
        if arr.shape[:2] != (patch, patch):
            raise SystemExit(f"FATAL {r['image']} crop at ({y},{x}) gave {arr.shape[:2]}")
        valid = np.ones(arr.shape[:2], dtype=bool)
        rel = r["aux"].get(UNSEEN_PRODUCT)
        if exclude_unseen and rel:
            m = load(cache_msk, os.path.join(mask_root, rel))
            if m.ndim == 3:
                m = m[:, :, 0]
            valid &= (m[y:y + patch, x:x + patch] == 0)
        n_px_total += valid.size
        n_px_valid += int(valid.sum())
        for c in range(3):
            ch[c] += np.bincount(arr[:, :, c][valid].ravel(), minlength=256)
        gray = arr.mean(axis=2).astype(np.uint8)
        gy += np.bincount(gray[valid].ravel(), minlength=256)

    if (ch.sum(axis=1) == 0).any() or gy.sum() == 0:
        raise SystemExit("FATAL empty source histogram over the training pool")
    prov = {
        "n_source_patches": len(use),
        "n_source_images": len({r["image"] for r in use}),
        "source_px_total": int(n_px_total),
        "source_px_valid": int(n_px_valid),
        "source_unseen_frac": round(1.0 - n_px_valid / max(n_px_total, 1), 6),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  [{label}] source histogram over {prov['n_source_patches']} training "
          f"patches from {prov['n_source_images']} images in {prov['seconds']}s; "
          f"unseen fraction of the source pool {prov['source_unseen_frac']}",
          flush=True)
    return np.cumsum(ch / ch.sum(axis=1, keepdims=True), axis=1), _cdf(gy), prov


# ---------------------------------------------------------------------------
# Cell resolution
# ---------------------------------------------------------------------------

def parent_run_id(arch, holdout, rung, seed, method=VANILLA_FAMILY_METHOD):
    """The LOCO cell this probe rides on.

    `method` was hardcoded "vanilla" until 2026-09-22; D-A2-2 extends the probe
    to method cells, so it is now a parameter. The default reproduces the old
    string exactly, so every existing vanilla invocation is unchanged.
    """
    return (f"{arch}-{method}-loco-r{rung}-"
            f"{CITY_OF_HOLDOUT[holdout]}-hr-s{seed}")


def resolve_repro_rule(arch, parent_dump_dir):
    """(rule, tolerance_pct, detail) for the baseline-reproduces-parent gate.

    D-A2-2 states this rule DIFFERS BY ARCHITECTURE and must be written into
    every probe cell's run.json:

      MAMNet / DINOv3 -- bit-exact. REPRO_TOL_MIOU_PCT is a hard gate, not a
                         tolerance to widen; a mismatch is a FINDING.
      OGLANet         -- NOT exact (E-9: inference is nondeterministic).
                         Asserting exactness would FAIL ON CORRECT RUNS. The
                         band is taken from the parent cell's OWN 3-pass
                         inference_repeat.json rather than from the published
                         0.42 SD, so the assertion is against what this cell
                         actually did.
      CrossEarth      -- bit-exact, MEASURED not assumed (D-B3-2, 2026-10-10).
                         Three cells spanning three folds and three methods,
                         3 passes each: SD exactly 0.0 and ZERO pixels changing
                         prediction. Was UNSPECIFIED, and this function used to
                         refuse crossearth by name; it no longer does. The
                         measurement is WITHIN-process and the gate it licenses
                         is CROSS-process -- that scope limit travels into every
                         crossearth probe's banner and run.json rather than
                         living only in DECISIONS_armA_wave2.md.
    """
    if arch in ("mamnet", "dinov3"):
        return "bit_exact", REPRO_TOL_MIOU_PCT, {
            "rule": "bit_exact",
            "why": ("D-A2-2: MAMNet and DINOv3 are bit-exact under the arm's "
                    "E-9 controls, so this is a hard gate and a mismatch is a "
                    "finding, not a tolerance to widen."),
        }

    if arch == "oglanet":
        rp = Path(parent_dump_dir) / "inference_repeat.json"
        if not rp.exists():
            raise SystemExit(
                f"FATAL OGLANet probe cell needs its parent's E-9 band and "
                f"{rp} is absent. D-A2-2 requires the OGLANet reproduction to "
                f"be asserted WITHIN the E-9 band; asserting bit-exactness "
                f"would fail on a correct run, and inventing a band would make "
                f"the gate meaningless. Run the 3-pass repeat for the parent "
                f"cell first.")
        rep = json.loads(rp.read_text())
        sd_block = rep.get("across_pass_sd") or {}
        sd = sd_block.get("miou_tol_pct_sd")
        rng = sd_block.get("miou_tol_pct_range")
        if sd is None or not rng:
            raise SystemExit(
                f"FATAL {rp} carries no across_pass_sd.miou_tol_pct_sd / "
                f"_range; cannot resolve the E-9 band without guessing.")
        observed_range = float(rng[1]) - float(rng[0])
        # Band = the wider of the parent's observed across-pass spread and
        # 3 SD of it. Both come from the parent's own measurement.
        tol = max(observed_range, 3.0 * float(sd))
        return "e9_band", tol, {
            "rule": "e9_band",
            "why": ("D-A2-2: OGLANet inference is nondeterministic (E-9), so "
                    "the reproduction is asserted within the E-9 band and the "
                    "delta against that band is reported. Asserting exactness "
                    "would fail on correct runs."),
            "source": str(rp),
            "parent_miou_tol_pct_sd": float(sd),
            "parent_miou_tol_pct_range": [float(rng[0]), float(rng[1])],
            "band_pct": tol,
            "band_definition": "max(parent observed across-pass range, 3 x SD)",
            "published_e9_sd_upper": sd_block.get("published_e9_sd_upper"),
        }

    if arch == "crossearth":
        # D-B3-2 (operator, 2026-10-10, commit 1e2b4cb) RESOLVED this. It used
        # to refuse by name because there was nothing to derive a rule from:
        # 41 crossearth dumps on /scratch and ZERO inference_repeat.json among
        # them. The 3-pass repeat has since been MEASURED on three cells
        # spanning folds and methods -- vanilla/OMA (n=1637), sib/JAX (n=2642)
        # and fda/UCSD (n=1870) -- and all three were bit-exact: SD exactly
        # 0.0, max |prob delta| 0.0, and ZERO pixels changing prediction over
        # 6,149 patches in total. So CrossEarth joins MAMNet and DINOv3 on the
        # exact rule and no band is created; OGLANet remains the sole
        # nondeterministic architecture. The alternative could not be derived
        # from this data anyway: max(range, 3 x SD) is 0.0, so a "band" would
        # degenerate to the exact rule.
        return "bit_exact", REPRO_TOL_MIOU_PCT, {
            "rule": "bit_exact",
            "why": ("D-B3-2: MEASURED bit-exact on three crossearth cells "
                    "spanning three folds and three methods. CrossEarth is a "
                    "ViT whose eval path has no stochastic component and "
                    "seo_crossearth.predict_with_valid calls model.eval() "
                    "before every pass."),
            "source": "DECISIONS_armA_wave2.md D-B3-2 (commit 1e2b4cb)",
            "measured_on": [
                "crossearth-vanilla-loco-r3500-oma-hr-s0  (n=1637)",
                "crossearth-sib-loco-r3500-jax-hr-s0      (n=2642)",
                "crossearth-fda-loco-r3500-ucsd-hr-s0     (n=1870)",
            ],
            "measured_sd_miou_tol_pct": 0.0,
            "measured_max_abs_prob_delta": 0.0,
            "measured_px_changing_prediction": 0,
            "n_patches_measured": 6149,
            # THE SCOPE LIMIT, carried into the probe's own output so an
            # auditor meeting one of these dumps sees it without this file.
            "scope_limit": (
                "D-B3-2 measured WITHIN-PROCESS pass-to-pass identity: three "
                "passes inside one job, on one node and one GPU. The gate it "
                "licenses here is a CROSS-PROCESS comparison -- this probe "
                "recomputing a baseline and checking it against a parent dump "
                "written by a different job on a possibly different node. "
                "Within-process identity does not by itself establish "
                "cross-process identity and D-B3-2 does not claim it does. It "
                "is nonetheless the right rule because it is exactly the "
                "evidence base MAMNet's and DINOv3's rules rest on, and "
                "because a band cannot be derived from a measurement whose "
                "range and SD are both exactly 0.0."),
            "scope_limit_consequence": (
                "THE FIRST crossearth probe cell whose baseline misses its "
                "parent by more than REPRO_TOL_MIOU_PCT is a FINDING, to be "
                "reported and NOT absorbed. The candidate explanation to "
                "check first is cross-process or cross-node variation, which "
                "D-B3-2 has NOT excluded. This tolerance must never be "
                "widened retroactively."),
            "positive_control": (
                "the same probe, schema and metric path returned "
                "bit_exact=False with max |delta| 0.40-0.51 on three OGLANet "
                "smokes, and the three crossearth cells returned three "
                "DIFFERENT mIoUs (64.47 / 67.17 / 68.43), so the measurement "
                "is not dead."),
        }

    raise SystemExit(f"FATAL no reproduction rule known for arch {arch!r}")


def comparator_run_id(arch, holdout, rung, seed):
    return f"{arch}-none-upper-r{rung}-{CITY_OF_HOLDOUT[holdout]}-hr-s{seed}"


def stored_dump(root, run_id):
    d = Path(root) / run_id
    if not (d / ".DONE").exists():
        raise SystemExit(f"FATAL no completed dump for {run_id} at {d}")
    return d


def stored_perimage(root, run_id):
    import pandas as pd
    return pd.read_parquet(stored_dump(root, run_id) / "perimage.parquet")


def stored_run_json(root, run_id):
    return json.loads((stored_dump(root, run_id) / "run.json").read_text())


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    # D-A2-2 (2026-09-22). Default reproduces the vanilla-only behaviour this
    # script had before, so every existing invocation is unchanged.
    ap.add_argument("--method", default=VANILLA_FAMILY_METHOD,
                    help="method label of the parent LOCO cell. Default "
                         "'vanilla'. Anything else puts this probe in the "
                         "METHOD family, which is never pooled into the "
                         "headline 0-of-N count (D-A2-2).")
    ap.add_argument("--holdout", required=True, choices=sorted(CITY_OF_HOLDOUT))
    ap.add_argument("--rung", type=int, default=seo_budget.N_TRAIN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget_dir", required=True)
    ap.add_argument("--patch_index", required=True)
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--mask_root", required=True)
    ap.add_argument("--ckpt_root", required=True)
    ap.add_argument("--vanilla_dump_root", required=True,
                    help="where the phase-2 vanilla and upper dumps live")
    ap.add_argument("--out_root", required=True, help="dump root; /scratch")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--timeout_s", type=int, default=3600)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--smoke_test", type=int, default=0)
    ap.add_argument("--smoke_train", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    smoke = bool(args.smoke_test or args.smoke_train)
    arch, holdout = args.arch, args.holdout
    spec = P.ARCH_SPEC[arch]
    method = args.method
    family = probe_family(method)
    parent = parent_run_id(arch, holdout, args.rung, args.seed, method)
    # The comparator (upper) is ALWAYS vanilla: methods are LOCO-only
    # project-wide, so there is no method upper cell. For a method-family probe
    # this makes R a method numerator over a VANILLA denominator -- a different
    # quantity from the published R. That is disclosed in run.json below and is
    # the reason the two families never merge.
    comparator = comparator_run_id(arch, holdout, args.rung, args.seed)
    mixed_denominator = (family == "method")
    run_id = f"{parent}-{PROBE_SUFFIX}"
    dump_dir = Path(args.out_root) / DATASET / run_id
    recipe_config = BASE / spec["recipe"]

    # The rule identity check runs BEFORE anything expensive: a probe that is
    # not the published intervention should cost seconds, not a GPU hour.
    rule_check = assert_rule_identity()

    man_path = Path(args.budget_dir) / f"{holdout}_loco.json"
    man = json.loads(man_path.read_text())
    man_sha = hashlib.sha256(man_path.read_bytes()).hexdigest()
    if man["budget_rule_sha"] != seo_budget.budget_rule_sha():
        raise SystemExit(
            f"FATAL budget manifest rule {man['budget_rule_sha'][:16]} != live "
            f"rule {seo_budget.budget_rule_sha()[:16]}")

    parent_run = stored_run_json(args.vanilla_dump_root, parent)
    # Resolved HERE, before the GPU is touched: a crossearth cell or an
    # OGLANet cell with no E-9 band should cost seconds, not an inference pass.
    repro_rule, repro_tol, repro_detail = resolve_repro_rule(
        arch, stored_dump(args.vanilla_dump_root, parent))
    ckpt = Path(parent_run["ckpt_path"])
    if not ckpt.exists():
        raise SystemExit(f"FATAL parent checkpoint missing: {ckpt}")

    print("=" * 78)
    print(f"ARM A / S-EO  section 4.2 HISTOGRAM-MATCHING PROBE   {run_id}")
    print(f"  arch / holdout          {arch} / {holdout}")
    print(f"  parent (LOCO) cell      {parent}")
    print(f"  method / family         {method} / {family}"
          f"{'   *** METHOD FAMILY: never pooled into the headline ***' if mixed_denominator else ''}")
    print(f"  reproduction rule       {repro_rule}  tol {repro_tol:.6g} mIoU pct")
    print(f"  repro rule source       {repro_detail.get('source', 'D-A2-2')}")
    # D-B3-2 asks for the scope limit to be legible to an auditor who has only
    # this cell's artifacts. run.json carries it under sanity.repro_rule_detail;
    # the banner carries it here so the .out log is self-describing too.
    for _k in ("scope_limit", "scope_limit_consequence"):
        _v = repro_detail.get(_k)
        if _v:
            print(f"  {_k:22s}  {_v}")
    print(f"  comparator (upper)      {comparator}")
    print(f"  checkpoint              {ckpt}")
    print(f"  sources                 {man['source_cities']}")
    print(f"  budget                  train {man['n_train']}  test {man['n_test']}")
    print(f"  manifest sha256         {man_sha[:16]}")
    print(f"  headline match rule     PUBLISHED grayscale-plus-ratio "
          f"(experiment_utils.histogram_match_image)")
    print(f"  alongside               arm C per-channel variant")
    print(f"  rule identity vs arm C  {rule_check['identical_to_armC']}")
    print(f"  dump                    {dump_dir}")
    print(f"  slurm job               {os.environ.get('SLURM_JOB_ID')} on "
          f"{os.environ.get('SLURMD_NODENAME')}")
    if smoke:
        print(f"  SMOKE  test<={args.smoke_test}  source_patches<={args.smoke_train}")
    print("=" * 78, flush=True)

    if args.dry_run:
        print("--dry_run: resolved, nothing written.")
        return

    if (dump_dir / ".DONE").exists():
        print(f"[{run_id}] .DONE exists -- nothing to do")
        return
    armc_runtime.preflight_write_targets(
        BASE, [dump_dir / "perimage.parquet", dump_dir / "probs.npz",
               dump_dir / "run.json", dump_dir / ".DONE"], label=run_id)

    import torch                                                  # noqa: E402
    from torch.utils.data import DataLoader                       # noqa: E402
    seed_sites = P.seed_everything(args.seed)

    # The loader gains a matcher hook and nothing else. Rebinding the name
    # build_split constructs keeps the split assembly byte-identical to the one
    # that produced the parent cell rather than forking it.
    P.SEODataset = MatchedSEODataset

    idx = json.loads(Path(args.patch_index).read_text())
    tr_ids, va_ids, te_ids = man["train"], man["val"], man["test"]
    if args.smoke_test:
        te_ids = te_ids[:args.smoke_test]
    uc = spec["use_contrast"]
    ds_train, rows_tr, bases_tr, _ = P.build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "train", tr_ids,
        False, "train", uc)
    ds_val, rows_va, bases_va, _ = P.build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "val", va_ids,
        False, "val", uc)
    ds_test, rows_te, bases_te, _ = P.build_split(
        args.img_root, args.mask_root, idx, args.splits_dir, "test", te_ids,
        False, "test", uc)
    pidx = P.patch_index_map(bases_tr + bases_va + bases_te)

    # sun_dist is defined against THIS cell's training pool, so it has to be the
    # parent cell's number or the column means something different here.
    train_sun = np.array([r["sun_el"] for r in rows_tr], dtype=np.float64)
    train_sun_mean = float(train_sun.mean())
    if abs(train_sun_mean - float(parent_run["train_sun_elev_mean"])) > 5e-4:
        raise SystemExit(
            f"FATAL train sun elevation mean {train_sun_mean:.6f} differs from the "
            f"parent cell's {parent_run['train_sun_elev_mean']}; the training pool "
            "this probe resolved is not the one the checkpoint was trained on")

    src_ch, src_gray, src_prov = source_cdfs_seo(
        rows_tr, args.img_root, args.mask_root, 384, exclude_unseen=True,
        limit=args.smoke_train, label=run_id)

    # ---- the model: the parent cell's, unchanged ---------------------------
    cfg = json.loads(recipe_config.read_text())
    arch_root = BASE / "python" / spec["dir"]
    sys.path.insert(0, str(arch_root))
    os.chdir(arch_root)
    for _m in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_m]
    archtrain = __import__(spec["train_module"])
    disabled = armc_runtime.silence_worker_side_effects(archtrain)

    from argparse import Namespace
    saved_argv, sys.argv = sys.argv, [spec["train_module"]]
    try:
        defaults = vars(archtrain.get_args())
    finally:
        sys.argv = saved_argv
    a = {**defaults, **cfg}
    rerouted = P.route_config_to_this_cluster(a, arch)
    # The probe never trains, so output_dir is its own scratch directory: no
    # published checkpoint tree is opened for writing by this job at all.
    a["output_dir"] = str(Path(args.ckpt_root) / run_id)
    a["device"] = "cuda"
    a["eval_only"] = False
    a["resume"] = None
    a["num_workers"] = args.num_workers
    a["seed"] = args.seed
    trainer = archtrain.Trainer(Namespace(**a))

    g = torch.Generator()
    g.manual_seed(args.seed)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             worker_init_fn=P.dl_worker_init)
    trainer.dataloaders = {"train": test_loader, "val": test_loader,
                           "test": test_loader}

    from utils.postprocessing import filter_small_predictions

    with armc_runtime.PhaseTimeout(args.timeout_s, dump_dir) as tbx:
        tbx.phase("load_checkpoint")
        trainer.load_checkpoint(str(ckpt))
        model = trainer.model
        model.eval()
        device = trainer.device
        print(f"  VRAM after model load {torch.cuda.memory_allocated() / 2**30:.3f} GB",
              flush=True)

        meta_by_pid = {}
        for r in rows_te:
            pid = f"{r['image_id']}_y{r['y']}_x{r['x']}"
            meta_by_pid[pid] = {
                "image_id": pid, "aoi_id": r["aoi_id"],
                "source_image_id": r["image_id"], "patch_index": pidx[pid],
                "patch_y": r["y"], "patch_x": r["x"],
                "city": r["city"], "split": "test",
                "acquisition_date": r["acq"],
                "sun_elevation": r["sun_el"], "sun_azimuth": r["sun_az"],
                "lon": r["lon"], "lat": r["lat"],
                "sun_dist": abs(float(r["sun_el"]) - train_sun_mean),
            }

        def one_pass(name, matcher):
            """The parent cell's inference loop, verbatim, under one matcher."""
            set_matcher(bases_te, matcher)
            tbx.phase(f"inference_{name}")
            t = time.time()
            rows, probs = [], {}
            with torch.no_grad():
                for batch in trainer.dataloaders["test"]:
                    images = batch["image"].to(device)
                    pid = batch["filename"][0]
                    logits = model(images)
                    prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy().astype(np.float32)
                    filtered = filter_small_predictions(logits, min_pixels=10)
                    pred = torch.argmax(filtered, dim=1)[0].cpu().numpy().astype(np.uint8)
                    gt = batch["mask"][0].numpy().astype(np.uint8)
                    seen = batch["valid"][0].numpy().astype(bool)
                    rows.append(seo_dump.perimage_row(
                        run_id, meta_by_pid[pid], gt, prob, pred, seen))
                    probs[pid] = prob.astype(np.float16)
            set_matcher(bases_te, None)
            wall = time.time() - t
            print(f"  pass {name}: {len(rows)} patches in {wall:.1f}s", flush=True)
            return rows, probs, wall

        rows_base, _, wall_base = one_pass("unmatched", None)
        rows_gray, probs_gray, wall_gray = one_pass(
            "matched_gray_published", SEOGrayMatcher(src_gray))
        rows_perch, _, wall_perch = one_pass(
            "matched_per_channel", SEOPerChannelMatcher(src_ch))

        tbx.phase("recovery_ratio")
        import pandas as pd                                       # noqa: E402
        df = pd.DataFrame(rows_base)
        dfg = pd.DataFrame(rows_gray)
        dfp = pd.DataFrame(rows_perch)
        if not (list(df["image_id"]) == list(dfg["image_id"]) == list(dfp["image_id"])):
            raise SystemExit("FATAL the three inference passes disagree on image order")

        loco = stored_perimage(args.vanilla_dump_root, parent)
        upper = stored_perimage(args.vanilla_dump_root, comparator)
        iou_loco = C.align_on_image_id(df, loco, "iou_tol")
        iou_upper = C.align_on_image_id(df, upper, "iou_tol")
        fp_loco = C.align_on_image_id(df, loco, "fp_tol")
        miou_loco = C.align_on_image_id(df, loco, "miou_tol_pct")

        # R is arm C's, imported: (probe - LOCO) / (upper - LOCO) on
        # foreground-only IoU at K=2, undefined where the headroom is < 0.02.
        # LOCO and upper are the STORED dumps, never re-inferred.
        r_gray, ok, head = C.recovery_ratio(dfg["iou_tol"].to_numpy(),
                                            iou_loco, iou_upper)
        r_perch, _, _ = C.recovery_ratio(dfp["iou_tol"].to_numpy(),
                                         iou_loco, iou_upper)

        rows = df.to_dict("records")
        for i, row in enumerate(rows):
            row["probe"] = 1
            row["probe_name"] = "s4.2 histogram matching"
            row["rung"] = args.rung
            row["comparator"] = comparator
            row["target_access"] = "source_only"
            row["iou_tol_matched_gray"] = float(dfg["iou_tol"].iloc[i])
            row["iou_tol_matched"] = float(dfp["iou_tol"].iloc[i])
            row["miou_tol_pct_matched_gray"] = float(dfg["miou_tol_pct"].iloc[i])
            row["miou_tol_pct_matched"] = float(dfp["miou_tol_pct"].iloc[i])
            for band, col in ((0, "str"), (2, "tol"), (5, "k5")):
                for cell in ("tp", "fp", "fn", "tn"):
                    k = f"{cell}_{col}"
                    if k in dfg.columns:
                        row[f"{k}_matched_gray"] = int(dfg[k].iloc[i])
                        row[f"{k}_matched"] = int(dfp[k].iloc[i])
            row["n_pred_pos_matched_gray"] = int(dfg["n_pred_pos"].iloc[i])
            row["n_pred_pos_matched"] = int(dfp["n_pred_pos"].iloc[i])
            row["fp_px_loco_stored"] = int(fp_loco[i])
            row["iou_tol_loco_stored"] = float(iou_loco[i])
            row["iou_tol_upper_stored"] = float(iou_upper[i])
            row["miou_tol_pct_loco_stored"] = float(miou_loco[i])
            row["headroom"] = float(head[i])
            row["r_included"] = bool(ok[i])
            row["recovery_ratio_R_gray"] = float(r_gray[i])
            row["recovery_ratio_R"] = float(r_perch[i])

        # ---- sanity, all of it inside the job ---------------------------
        tbx.phase("sanity")

        def predpos(d):
            return float(d["n_pred_pos"].sum() / max(int(d["n_valid_px"].sum()), 1))

        def n_changed(other):
            cols = [c for c in ("tp_tol", "fp_tol", "fn_tol", "tn_tol") if c in df]
            diff = np.zeros(len(df), dtype=bool)
            for c in cols:
                diff |= (df[c].to_numpy() != other[c].to_numpy())
            return int(diff.sum())

        stored_miou = float(loco["miou_tol_pct"].mean())
        base_miou = float(df["miou_tol_pct"].mean())
        repro_delta = base_miou - stored_miou
        max_abs_img_delta = float(np.abs(
            df["miou_tol_pct"].to_numpy() - miou_loco).max())
        checks = {
            "baseline_reproduces_parent_dump": bool(abs(repro_delta) < repro_tol),
            "repro_delta_miou_pct": round(repro_delta, 8),
            "repro_max_abs_per_image_miou_pct": round(max_abs_img_delta, 8),
            "repro_tolerance_miou_pct": repro_tol,
            "repro_rule": repro_rule,
            "repro_rule_detail": repro_detail,
            # D-A2-2 asks for the delta against the band, not just pass/fail.
            "repro_delta_over_band": (round(abs(repro_delta) / repro_tol, 6)
                                      if repro_tol else None),
            "n_rows": len(rows),
            "n_test_expected": len(te_ids),
            "row_count_matches_n_test": bool(len(rows) == len(te_ids)),
            "row_count_matches_all_conditions": bool(
                len(df) == len(dfg) == len(dfp) == len(te_ids)),
            "pred_pos_frac_unmatched": round(predpos(df), 6),
            "pred_pos_frac_matched_gray": round(predpos(dfg), 6),
            "pred_pos_frac_matched": round(predpos(dfp), 6),
            "n_images_changed_by_matched_gray": n_changed(dfg),
            "n_images_changed_by_matched": n_changed(dfp),
            "mean_unseen_frac": round(float(df["unseen_frac"].mean()), 6),
            "n_patches_valid_mask_empty": int((df["unseen_frac"] >= 1.0).sum()),
            "unseen_exclusion_live": bool(float(df["unseen_frac"].max()) > 0.0),
        }
        for lab, v in (("matched_gray", checks["pred_pos_frac_matched_gray"]),
                       ("matched", checks["pred_pos_frac_matched"])):
            checks[f"pred_pos_in_range_{lab}"] = bool(PRED_POS_MIN < v < PRED_POS_MAX)

        fatal = []
        if not checks["row_count_matches_all_conditions"]:
            fatal.append(f"row counts {len(df)}/{len(dfg)}/{len(dfp)} != "
                         f"n_test {len(te_ids)}")
        for lab in ("matched_gray", "matched"):
            if not checks[f"pred_pos_in_range_{lab}"]:
                fatal.append(f"{lab} predicted-positive fraction "
                             f"{checks['pred_pos_frac_' + lab]} outside "
                             f"({PRED_POS_MIN}, {PRED_POS_MAX})")
            if checks[f"n_images_changed_by_{lab}"] == 0:
                fatal.append(f"{lab} is identical to the baseline on every image: "
                             "the transform did not reach the model")
        empty_valid = (df["unseen_frac"].to_numpy() >= 1.0)
        if empty_valid.any():
            for lab, d in (("unmatched", df), ("matched_gray", dfg), ("matched", dfp)):
                if int(d.loc[empty_valid, "n_valid_px"].sum()) != 0:
                    fatal.append(
                        f"{lab}: a fully-unseen patch reports n_valid_px > 0, so "
                        "the whole-patch fallback is NOT metric-neutral")
        if not smoke and not checks["baseline_reproduces_parent_dump"]:
            if repro_rule == "bit_exact" and arch == "crossearth":
                # D-B3-2's scope limit is the operative half here: it measured
                # WITHIN-process identity and this gate is CROSS-process, so
                # the first candidate explanation is named rather than left to
                # the reader -- and the tolerance is explicitly not widenable.
                why = (repro_detail.get("scope_limit_consequence", "")
                       + " Rule source: "
                       + str(repro_detail.get("source", "D-B3-2"))
                       + ". Scope limit: "
                       + str(repro_detail.get("scope_limit", "")))
            elif repro_rule == "bit_exact":
                why = ("This is a FINDING, not a tolerance to widen -- MAMNet "
                       "and DINOv3 are bit-exact under the arm's E-9 controls "
                       "(D-A2-2).")
            else:
                why = (f"The band is this parent cell's OWN measured E-9 "
                       f"spread ({repro_detail.get('band_definition')}), not a "
                       f"published constant, so exceeding it is a FINDING "
                       f"about this cell and not E-9 noise.")
            fatal.append(
                f"baseline mIoU {base_miou:.6f} does not reproduce the parent "
                f"dump's {stored_miou:.6f} (delta {repro_delta:.2e}, rule "
                f"{repro_rule}, tolerance {repro_tol:.6g}); max per-image "
                f"|delta| {max_abs_img_delta:.2e}. {why}")
        if fatal:
            for f in fatal:
                print(f"FATAL SANITY {f}", flush=True)
            raise SystemExit("FATAL sanity gate failed; no dump written")
        if smoke:
            checks["baseline_reproduction_note"] = (
                "SKIPPED: a smoke pass scores a truncated test split, which "
                "cannot reproduce the parent cell's full-split mean")

        r_ok_gray = r_gray[ok]
        r_ok_perch = r_perch[ok]
        summary = {
            "n_images": len(rows),
            "miou_tol_pct_unmatched": round(base_miou, 4),
            "miou_tol_pct_matched_gray": round(float(dfg["miou_tol_pct"].mean()), 4),
            "miou_tol_pct_matched": round(float(dfp["miou_tol_pct"].mean()), 4),
            "delta_miou_gray": round(float(dfg["miou_tol_pct"].mean()) - base_miou, 4),
            "delta_miou_per_channel": round(
                float(dfp["miou_tol_pct"].mean()) - base_miou, 4),
            "fg_iou_tol_pct_unmatched": round(float(df["iou_tol"].mean()) * 100, 4),
            "fg_iou_tol_pct_matched_gray": round(float(dfg["iou_tol"].mean()) * 100, 4),
            "fg_iou_tol_pct_matched": round(float(dfp["iou_tol"].mean()) * 100, 4),
            "R_gray_mean": round(float(np.mean(r_ok_gray)), 6) if ok.any() else None,
            "R_gray_median": round(float(np.median(r_ok_gray)), 6) if ok.any() else None,
            "R_mean": round(float(np.mean(r_ok_perch)), 6) if ok.any() else None,
            "R_median": round(float(np.median(r_ok_perch)), 6) if ok.any() else None,
            "r_n_included": int(ok.sum()),
            "r_n_excluded": int((~ok).sum()),
            "summary_note": (
                "descriptive per-cell means over this cell's own images, an "
                "execution fact for the report. No pooling across cells, no CI, "
                "no test: per-image values are the columns in this parquet and "
                "the SSH box does the statistics."),
        }

        # ---- run.json ---------------------------------------------------
        tbx.phase("run_json")
        run = {
            "run_id": run_id,
            "dataset": DATASET,
            "arm": "A",
            "probe": 1,
            "probe_name": "s4.2 histogram matching",
            "probe_note": (
                "external replication of the section 4.2 histogram-matching "
                "probe that arm C ran on ShadowTransfer. Answers v1UJ-W4 / "
                "zNBu-W2. INFERENCE ONLY: no training, no new checkpoint, no "
                "new method cell, no Experiment A. NOTE the probe itself may "
                "RIDE ON a method cell since D-A2-2 (2026-08-21); see "
                "probe_family. It still trains nothing."),
            "arch": arch,
            "method": method,
            "probe_family": family,
            "probe_family_rule": (
                "D-A2-2. The vanilla family may join the headline 0-of-N "
                "count. The method family is a SEPARATE family, reported "
                "beside the headline and NEVER pooled into it."),
            "mixed_denominator": mixed_denominator,
            "mixed_denominator_disclosure": (
                ("THIS IS A METHOD-FAMILY PROBE. R here is a METHOD numerator "
                 "over a VANILLA denominator: methods are LOCO-only "
                 "project-wide, so there is no method upper cell and this "
                 "probe borrows the same fold/seed vanilla upper "
                 f"({comparator}) as its denominator. That is a DIFFERENT "
                 "QUANTITY from the R arm C published and from the R the "
                 "vanilla cells report. It must never be pooled into the "
                 "headline 0-of-N count. See DECISIONS_armA_wave2.md D-A2-2.")
                if mixed_denominator else
                ("vanilla family: numerator and denominator are both vanilla, "
                 "so R is the same quantity arm C published.")),
            "mode": "loco",
            "rung": args.rung,
            "holdout": holdout,
            "res": "hr",
            "seed": args.seed,
            "n_test": len(rows),
            "parent_run_id": parent,
            "parent_ckpt_path": str(ckpt),
            "parent_ckpt_sha256": sha256_file(ckpt),
            "parent_reporting_metric_value_pct": parent_run.get(
                "reporting_metric_value_pct"),
            "comparator": comparator,
            "comparator_rung": args.rung,
            "is_budget_asymmetric": False,
            "run_id_note": (
                "the probe run_id carries its parent cell's identity plus a "
                "probe suffix. It is not a new arm A cell and must not be "
                "pooled with one."),
            "reporting_metric": "2class_tol_k2_unseen_excluded",
            "reporting_metric_note": (
                "arm A's declared headline, column miou_tol_pct. The MATCHED "
                "conditions are miou_tol_pct_matched_gray (published rule) and "
                "miou_tol_pct_matched (arm C per-channel variant)."),
            "r_metric": "fg_only_tol_k2",
            "r_metric_note": (
                "R is computed on foreground-only IoU at K=2, column iou_tol, "
                "because that is what arm C's recovery_ratio was computed on."),
            "band_width_headline": 2,
            "band_widths_written": [0, 2, 5],
            "prediction_recipe": (
                "argmax over the two class logits, then "
                "filter_small_predictions(min_pixels=10); the parent cell's own "
                "loop, unchanged"),
            "probs_note": (
                "probs.npz holds the PUBLISHED-GRAY MATCHED softmax(logits)[:,1], "
                "float16, unthresholded. The unmatched probabilities already "
                "exist in the parent cell's dump and are not duplicated."),
            "target_access": "source_only",
            "target_access_detail": (
                "the matching statistic is the aggregate histogram of THIS "
                "cell's own source-city TRAINING patches. No held-out-city "
                "pixel, labelled or unlabelled, contributes to it; each test "
                "patch is matched toward that statistic using only its own "
                "pixels. Model weights are the parent cell's, unchanged, and "
                "no parameter is updated."),
            "intervention": "test-time CDF matching, no parameter update",
            "match_definition_headline": (
                "PUBLISHED grayscale-plus-ratio, verbatim from "
                "final_loco/experiment_utils.histogram_match_image via "
                "shadowtransfer/armc_probes_ops.match_gray: one grayscale CDF, "
                "mapping argmin_s |cdf_src[s] - cdf_img[t]|, each pixel's "
                "per-channel ratio re-applied to the mapped grey value. This is "
                "what the published section 4.2 finding was measured under and "
                "what arm C's replicated statistic used. Columns "
                "iou_tol_matched_gray, miou_tol_pct_matched_gray, "
                "recovery_ratio_R_gray."),
            "match_definition_alongside": (
                "arm C's per-channel variant (armc_probes_ops.match_per_channel), "
                "carried in iou_tol_matched, miou_tol_pct_matched and "
                "recovery_ratio_R so the two arms line up column for column."),
            "match_injection_point": (
                "MatchedSEODataset._to_tensor -- the raw uint8 RGB crop straight "
                "out of SEODataset._crop, BEFORE the RMS-contrast channel, "
                "BEFORE ImageNet normalisation and BEFORE any flip. Arm C's "
                "fda_transform point exactly."),
            "match_seo_deviation": (
                "the ONLY S-EO difference is which pixels enter a histogram: "
                "DSM-unseen pixels are excluded from the source-pool statistic "
                "and from each test patch's own target CDF, the same rule the "
                "metric and the loss use everywhere else in arm A. A histogram "
                "over pixels the geometry never determined would let DSM "
                "coverage, which differs several-fold between JAX, OMA and UCSD, "
                "drive the intervention unequally across folds. The mapping is "
                "still APPLIED to every pixel. Mechanically this is the arm C "
                "function with [valid] inside one bincount."),
            "match_seo_empty_valid_fallback": (
                "a test patch whose valid mask is EMPTY -- every pixel DSM-unseen "
                "-- has no valid-restricted histogram, so its target CDF is taken "
                "over the whole patch, which is the arm C rule unchanged. "
                "Metric-neutral: such a patch is dropped from all four confusion "
                "cells at every band width, and the sanity block asserts its "
                "n_valid_px is 0 in all three conditions. Count in "
                "sanity.n_patches_valid_mask_empty; 29 of 1637 OMA, 79 of 1870 "
                "UCSD, 0 of 2642 JAX."),
            "rule_identity_check": rule_check,
            "source_radiometry_note": (
                "S-EO pansharpened RGB is colour-corrected and HISTOGRAM-"
                "EQUALIZED AT SOURCE by the dataset authors (dataset card; "
                "seo_dataset.py docstring). This probe does NOT correct for "
                "that. It is a property of the data and part of what makes the "
                "replication informative; the analyst reads the interaction."),
            "source_pool": src_prov,
            "source_cities": man["source_cities"],
            "source_pool_note": (
                "the cell's own training list at its own budget, not the source "
                "cities' whole pools -- arm C's source_cdfs rule."),
            "train_sun_elev_mean": train_sun_mean,
            "r_definition": (
                "per image, R = (IoU_match - IoU_LOCO) / (IoU_upper - IoU_LOCO) "
                "on foreground-only IoU at K=2. IoU_LOCO and IoU_upper are read "
                "from the stored arm A dumps and are NOT re-inferred. "
                "shadowtransfer/armc_probes_common.recovery_ratio, imported."),
            "r_exclusion_rule": f"images with (IoU_upper - IoU_LOCO) < {R_EXCLUDE_MIN}",
            "r_n_included": int(ok.sum()),
            "r_n_excluded": int((~ok).sum()),
            "core_columns_note": (
                "the core CONTRACT.md 3 columns are the UNMATCHED re-inference, "
                "so this dump reproduces its parent cell and the matched results "
                "sit in the *_matched / *_matched_gray columns."),
            "unseen_handling": parent_run.get("unseen_handling"),
            "budget_manifest": str(man_path),
            "budget_manifest_sha256": man_sha,
            "budget_rule_sha": man["budget_rule_sha"],
            "split_rule_sha": parent_run.get("split_rule_sha"),
            "dataset_detail": parent_run.get("dataset_detail"),
            "arch_input": parent_run.get("arch_input"),
            "recipe_config_path": str(recipe_config.relative_to(BASE)),
            "config_sha256": sha256_file(recipe_config),
            "config_rerouted_keys": rerouted,
            "worker_side_effects_disabled": disabled,
            "seed_sites": seed_sites,
            "sanity": checks,
            "cell_summary": summary,
            "cluster": "delta",
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "gpu": os.environ.get("ARMA_GPU_NAME"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_time_limit": os.environ.get("ARMA_TIME_LIMIT"),
            "compute_note": os.environ.get("ST_COMPUTE_NOTE"),
            "git_hash": os.environ.get("ARMA_GIT_HASH"),
            "git_hash_scope": (
                "HEAD of BASE_PATH at submit time; the submit script hard-fails "
                "if this probe's own source files are uncommitted"),
            "git_tree_dirty_paths": os.environ.get("ARMA_GIT_DIRTY"),
            "smoke": smoke,
            "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metric_module_sha": sha256_file(BASE / "shadowtransfer/metrics.py"),
            "dump_module_sha": sha256_file(BASE / "shadowtransfer/seo_dump.py"),
            "loader_module_sha": sha256_file(BASE / "shadowtransfer/seo_dataset.py"),
            "budget_module_sha": sha256_file(BASE / "shadowtransfer/seo_budget.py"),
            "runtime_module_sha": sha256_file(BASE / "shadowtransfer/armc_runtime.py"),
            "probes_common_sha": sha256_file(BASE / "shadowtransfer/armc_probes_common.py"),
            "probes_ops_sha": sha256_file(BASE / "shadowtransfer/armc_probes_ops.py"),
            "probe_script_sha": sha256_file(Path(__file__).resolve()),
            "armC_probe_script_sha": sha256_file(BASE / "probes/armc_probes.py"),
            "comparability_note": (
                "probes_ops_sha and probes_common_sha are the SAME files arm C's "
                "probe 1 hashed into its own run.json. armC_probe_script_sha is "
                "arm C's driver, which this script does not run: the matching "
                "rule and R come from the two shared modules, the S-EO data "
                "plumbing does not."),
            "wall_clock_s": round(time.time() - t0, 2),
            "inference_wall_clock_s": {
                "unmatched": round(wall_base, 2),
                "matched_gray_published": round(wall_gray, 2),
                "matched_per_channel": round(wall_perch, 2),
            },
            "peak_vram_alloc_gb": round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
        }

        tbx.phase("write_dump")
        seo_dump.write_dump(dump_dir, rows, probs_gray, run)

    print("\n" + "=" * 78)
    print(f"DONE {run_id}")
    for k, v in summary.items():
        if k != "summary_note":
            print(f"  {k:34s} {v}")
    print(f"  mean unseen_frac                   {checks['mean_unseen_frac']}")
    print(f"  pred-pos unmatched/gray/perchannel {checks['pred_pos_frac_unmatched']} "
          f"{checks['pred_pos_frac_matched_gray']} {checks['pred_pos_frac_matched']}")
    print(f"  images changed gray/perchannel     "
          f"{checks['n_images_changed_by_matched_gray']} / "
          f"{checks['n_images_changed_by_matched']}")
    print(f"  repro delta miou_pct               {checks['repro_delta_miou_pct']}")
    print(f"  dump                               {dump_dir}")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
