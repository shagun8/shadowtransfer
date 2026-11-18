"""
Arm F -- results/32: the method comparison profile of results/28, recomputed on
results/31's rows (OGLANet-SIB from the HELD-OUT-CITY FDA pool retrains).

Answers AC3, v1UJ-W3, BxRx-W3, AC1 (results/28's IDs).

WHAT CHANGES AGAINST results/28, and nothing else:
  1. The row resolver. This script imports
     analysis/31_internal_sib_worst_case_tgtfda.py instead of analysis/27, so
     the three OGLANet-SIB rows are oglanet-sibc4tgt-loco-r450-{city}-hr-s42-
     n450 and every cell value is asserted against results/31's per_cell.
  2. M12's source_only restriction follows the access class results/31
     applies: OGLANet-SIB is unlabeled_target, so SIB's source_only cells are
     DINOv3 only on the 9-cell scope (3 cells) and DINOv3 plus CrossEarth on
     the 12-cell scope (6 cells). The restriction RULE is unchanged; only the
     class of three cells moved.
The metric list, method list, scopes, ALPHA and multiplicity family are
results/28's, unchanged. This remains POST HOC / EXPLORATORY.

Writes results/32_method_comparison_profile_tgtfda.json and analysis/_inv/
INDEX_{tree}_32.parquet.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/32_method_comparison_profile_tgtfda.py [--dry-run [--dry-dump=PATH]]

=============================================================================
THE REST OF THIS DOCSTRING IS results/28's, CARRIED VERBATIM. Where it names
results/27 or analysis/27, this script reads results/31 and analysis/31.
=============================================================================

Arm F -- method comparison profile on the n450 grid.

Answers AC3, v1UJ-W3, BxRx-W3 and AC1.

PURPOSE. A complete, PRE-DECLARED comparison of every method on the n450 grid.
The metric list below is fixed before any value is computed, and every metric is
reported for EVERY method: Vanilla, FDA, SegDeSiC, IIM, ISW, MRFP+, FADA, SIB,
BSM and SAR. No metric is added, dropped or reordered after seeing values, and
no method is reported selectively. The whole analysis is POST HOC: it was
commissioned after results/26's reading fired R3, i.e. after we had seen that
FDA's worst case sits at or below SIB's. That is stated in the headline and in
the caveats, and it is the reason the metric list is frozen in code rather than
chosen as the numbers arrive.

ROWS. Exactly results/27_internal_sib_worst_case_n450_sar.json's rows. This
script does not resolve cells with its own rules: it imports
analysis/27_internal_sib_worst_case_n450_sar.py and calls its resolve_all_rows(),
so every cell value, every tree choice, every n450 substitution, every SAR run
selection and every CONTRACT.md section 3 validation is literally the same code
that wrote results/27. Each cell's mean is then asserted against results/27's
own per_cell record, so a divergence is a loud failure rather than a second
opinion.

DECLARED PATH, inherited unchanged from results/27:
  metric      two-class average boundary-tolerant mIoU at K=2, in percent,
              computed per image from raw tp/fp/fn/tn on the valid region and
              macro-averaged over the cell's 150 test images
  resolution  high resolution only
  rung        450, with the 27 cells that trained on 900 source images replaced
              by their matched-450 retrains
  gap sign    gap = upper bound minus method (positive is a shortfall), the sign
              the worst-case claim is phrased in. "Worst case" always means the
              largest such gap over the cells of a scope.

SCOPES, declared:
  9_cell   the three original architectures (MAMNet, OGLANet, DINOv3) x three
           held-out cities
  12_cell  the same plus CrossEarth, for the five methods CrossEarth was run
           with (Upper, Vanilla, FDA, SIB, BSM). The other methods have no
           CrossEarth counterpart and stay 9-cell rows; n is stated per method
           on every row of every metric.

Read-only. No training, no GPU inference, no checkpoint loaded. Nothing under
dumps/, delta_runs/ or results/ is modified; results/26 and results/27 are read
as JSON and never written.

Run:  PYTHONNOUSERSITE=1 \
      python \
      analysis/28_method_comparison_profile.py [--dry-run]
"""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
B = 10000            # every bootstrap in this file, matching the project
BOOT_SEED = 0
ALPHA = 0.05         # the significance level every count in M5, M6, M8 and M10
                     # is taken at, declared before computing
INV = os.path.join(BASE, "analysis", "_inv")
# results/32: the rows and the per_cell record come from results/31
R27 = os.path.join(BASE, "results", "31_internal_sib_worst_case_tgtfda.json")
R28 = os.path.join(BASE, "results", "28_method_comparison_profile.json")
DRY_DUMP = next((a.split("=", 1)[1] for a in sys.argv
                 if a.startswith("--dry-dump=")), None)
if DRY_DUMP and "/ShadowTransfer/" in os.path.abspath(DRY_DUMP):
    raise SystemExit("--dry-dump must point outside the project tree")

# ----------------------------------------------------------------------------
# THE PRE-DECLARED METRIC LIST. Frozen before any value is computed.
# ----------------------------------------------------------------------------
# Nothing is added to this list, removed from it or reordered after the numbers
# are seen, and every entry is reported for every method in METHODS below even
# where it is degenerate (Vanilla's M5-M7 against itself) or empty (a method
# with no cell in the access class M12 restricts to). Where a metric is not
# defined for a method, the row says so and carries null, rather than being
# omitted.
METRICS = [
    dict(id="M1", name="worst-case absolute gap to upper",
         definition="max over the cells of a scope of the cell's mean gap "
                    "(upper bound minus method, two-class mIoU at K=2, "
                    "percent), reported with the cell it falls on",
         units="mIoU percentage points"),
    dict(id="M2", name="mean gap, cell-level bootstrap CI",
         definition="mean over the cells of a scope of the per-cell mean gap, "
                    "with a 95% percentile CI from resampling the CELLS with "
                    "replacement, B = 10000. The image-cluster bootstrap "
                    "results/27 uses for the same quantity (images resampled "
                    "within held-out city, shared across the cells of that "
                    "city) is reported beside it on every row as "
                    "mean_gap_image_cluster_*, declared here rather than added "
                    "later: the two resample different things and a reader "
                    "comparing this JSON against results/27 needs both.",
         units="mIoU percentage points"),
    dict(id="M3", name="worst-case relative gap",
         definition="max over the cells of a scope of 100 * (cell mean gap) / "
                    "(cell mean upper-bound value), reported with its cell",
         units="percent of the within-city upper bound"),
    dict(id="M4", name="mean relative gap with CI",
         definition="mean over cells of the per-cell relative gap of M3, with "
                    "a 95% percentile CI from resampling the cells with "
                    "replacement, B = 10000",
         units="percent of the within-city upper bound"),
    dict(id="M5", name="harm count",
         definition="number of cells where the method is significantly WORSE "
                    "than Vanilla: per cell, a paired image-level bootstrap on "
                    "the per-image difference (method minus Vanilla), B = "
                    "10000, two-tailed, Holm-Bonferroni over that method's "
                    "cells in the scope; a cell counts when its Holm-adjusted "
                    "p < 0.05 and the difference is negative. Not defined for "
                    "Vanilla, which is the comparator.",
         units="cells"),
    dict(id="M6", name="benefit count",
         definition="same test and same Holm family as M5; a cell counts when "
                    "its Holm-adjusted p < 0.05 and the difference is "
                    "positive. M5 and M6 share ONE two-tailed test per cell "
                    "and the sign decides which count the cell lands in, so a "
                    "cell contributes one p-value, not two.",
         units="cells"),
    dict(id="M7", name="largest harm",
         definition="min over the cells of a scope of (method minus Vanilla) "
                    "at the cell mean, reported with its cell. Negative means "
                    "the method is below Vanilla on that cell. Not defined for "
                    "Vanilla.",
         units="mIoU percentage points"),
    dict(id="M8", name="cells whose gap to upper is not significant",
         definition="per cell, a paired image-level bootstrap on the per-image "
                    "gap (method minus its within-city upper bound), B = "
                    "10000, two-tailed, Holm-Bonferroni over that method's "
                    "cells in the scope; a cell counts when its Holm-adjusted "
                    "p >= 0.05. Reported as a count and as the list of cells.",
         units="cells"),
    dict(id="M9", name="spread of the per-cell gap",
         definition="sample standard deviation (ddof = 1) of the per-cell mean "
                    "gap across the cells of a scope",
         units="mIoU percentage points"),
    dict(id="M10", name="SIB against every other method, per cell",
         definition="per cell, a paired image-level bootstrap on the per-image "
                    "difference (SIB minus the other method), B = 10000, "
                    "two-tailed, Holm-Bonferroni over the cells both methods "
                    "hold in the scope; reported as sign counts (SIB "
                    "significantly better / significantly worse / not "
                    "significant at Holm-adjusted p < 0.05) plus the per-cell "
                    "detail",
         units="cells"),
    dict(id="M11", name="worst-case difference with uncertainty",
         definition="for SIB against each other method, the difference of "
                    "worst-case gaps (SIB's worst case minus the other "
                    "method's) with a bootstrap: images are resampled with "
                    "replacement B = 10000 times, once per held-out city with "
                    "the same index vector applied to every cell of that city "
                    "(the Appendix G.9 image-cluster scheme, which keeps the "
                    "two methods paired on the same images and keeps the upper "
                    "bound paired with both), and BOTH methods' worst cases "
                    "are recomputed inside every replicate. Reported as the "
                    "observed point, a 95% percentile CI, a two-tailed p "
                    "floored at 1/B, and the share of replicates in which "
                    "SIB's worst case is the smaller of the two. Computed on "
                    "the cells both methods hold, with n stated.",
         units="mIoU percentage points"),
    dict(id="M12", name="the same profile restricted by target-access class",
         definition="M1 to M11 recomputed over the source_only cells only, "
                    "with n stated per method. A method's cells are classed by "
                    "the erratum I-1 taxonomy as results/27 applies it, so a "
                    "method whose cells span classes contributes only its "
                    "source_only ones: SIB contributes its DINOv3 cells on "
                    "the 9-cell scope and its DINOv3 and CrossEarth cells on "
                    "the 12-cell scope (results/32: OGLANet-SIB is "
                    "unlabeled_target since its FDA pool is the held-out "
                    "city's; results/28 read 'its OGLANet and DINOv3 cells'), "
                    "and FDA, SegDeSiC and SAR contribute none and are "
                    "reported with n = 0. Reported on both scopes.",
         units="as the metric restricted"),
]

METHODS = ["vanilla", "fda", "segdesic", "iim", "isw", "mrfp_plus", "fada",
           "sib", "bsm", "sar"]
SCOPES = ["9_cell", "12_cell"]
ACCESS_RESTRICTIONS = ["source_only"]      # M12; declared, not discovered

# The one multiplicity family the brief specifies, declared before computing.
# results/32: results/28 prose items that describe OGLANet-SIB as source_only
# or the superseded cross-source cells. Each must hit exactly `expect` items.
STALE_MARKERS_32 = [
    dict(match="FDA touches the target domain in every cell; SIB in three of nine.",
         expect=1,
         correction="results/32: SIB touches the target domain in six of nine "
                    "9-cell cells (MAMNet and OGLANet, both through FDA "
                    "preprocessing with a held-out-city pool); its three "
                    "DINOv3 cells are source_only. M12 still removes FDA "
                    "entirely, so there is still no access-matched "
                    "SIB-against-FDA comparison on this grid."),
    dict(match="the three MAMNet cells, through the FDA stage of its own recipe, "
               "and its OGLANet and DINOv3 cells are source_only",
         expect=1,
         correction="results/32: the count printed in this flag (6 of 9) is "
                    "recomputed and correct; the cells are the three MAMNet "
                    "and the three OGLANet cells, and only DINOv3-SIB is "
                    "source_only. The phrase 'a third of its cells' is "
                    "results/28's and now reads two thirds."),
    dict(match="SIB 6 (OGLANet and DINOv3 only)",
         expect=1,
         correction="results/32: on the 9-cell scope SIB has 3 source_only "
                    "cells (DINOv3 only); on the 12-cell scope 6 (DINOv3 and "
                    "CrossEarth). The other methods' counts are unchanged. "
                    "SIB's three exclude MAMNet and OGLANet."),
    dict(match="five repeat passes of oglanet-sibc4-loco-r450-chicago-hr-s42-n450",
         expect=1,
         correction="results/32: the five-pass measurement is of the "
                    "superseded cross-source chicago cell, which enters no "
                    "number here. No repeat-inference harness has been run "
                    "on a sibc4tgt cell; results/30 measures RNG dependence "
                    "on the same cross-source checkpoint."),
]


MULTIPLICITY_FAMILY = dict(
    comparison="SIB against FDA",
    why=("FDA is the method whose worst case sits at or below SIB's in "
         "results/26 and results/27, which is what fired reading R3, so it is "
         "the comparison this profile exists to characterise. Characterising "
         "it draws on five of the metrics above, which is a family of tests "
         "and not one test."),
    declared_member_classes=[
        "every M5/M6 cell test for SIB and for FDA (one two-tailed test per "
        "cell per method; M5 and M6 share it)",
        "every M8 cell test for SIB and for FDA",
        "every M10 per-cell SIB-against-FDA test",
        "the M11 SIB-against-FDA worst-case-difference test"],
    correction="Holm-Bonferroni across the whole family",
    reported="raw p and Holm-adjusted p for every member",
    scope="9_cell")

# ----------------------------------------------------------------------------
# import the row resolver that wrote results/27
# ----------------------------------------------------------------------------

R27_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "31_internal_sib_worst_case_tgtfda.py")   # results/32


def load_r27():
    spec = importlib.util.spec_from_file_location("r27_rows", R27_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load_r27()
fail = M.fail
CE = M.CE
ARCHS, CITIES = M.ARCHS, M.CITIES
METHOD_LABEL = M.METHOD_LABEL
paired_bootstrap = M.paired_bootstrap
holm = M.holm
sha256_file = M.sha256_file


# ----------------------------------------------------------------------------
# bootstraps this file adds
# ----------------------------------------------------------------------------

def cell_bootstrap(values, b=B, seed=BOOT_SEED):
    """Resample the CELLS of a scope with replacement.

    M2 and M4 ask for a cell-level interval, which is a different question from
    results/27's image-cluster interval: this one treats the cells as the
    sampling unit and so carries the between-cell spread, where the image-cluster
    one treats the images as the sampling unit and carries the within-cell noise.
    With 9 cells it is a coarse interval and it is reported as such; the
    image-cluster value is carried beside it on every row.
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    reps = v[rng.integers(0, n, size=(b, n))].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    return dict(mean=float(v.mean()), n_cells=int(n), ci95=[lo, hi],
                ci_method=f"cell_level_bootstrap_B{b}",
                bootstrap_sd=float(reps.std(ddof=1)),
                sd_across_cells=(float(v.std(ddof=1)) if n > 1 else None))


def city_index_draws(cells_by_city, b=B, seed=BOOT_SEED):
    """One resampled image index vector per held-out city per replicate.

    The 150 test images of a city are shared, in identical order, by every cell
    of that city -- asserted in results/27's declared_path.image_id_identity --
    so resampling them once per city and applying the same vector to every cell
    of that city is what keeps two methods paired with each other and with the
    upper bound inside a replicate. Resampling each cell independently would
    break that pairing and would widen every interval for the wrong reason.
    """
    rng = np.random.default_rng(seed)
    sizes = {c: a.size for c, a in cells_by_city.items()}
    for _ in range(b):
        yield {c: rng.integers(0, n, size=n) for c, n in sizes.items()}


def worst_case_diff_bootstrap(cells_a, cells_b, b=B, seed=BOOT_SEED):
    """M11. Difference of two methods' worst-case gaps, with its uncertainty.

    cells_a / cells_b are lists of (city, gap_per_image) over the SAME cell set,
    so the difference is of two worst cases taken over the same cells. Inside a
    replicate the images of a city are resampled once and that one vector is
    applied to every cell of that city on both sides, then each side's worst
    case -- the max over its cells of the resampled per-cell mean gap -- is
    recomputed. The statistic is a max of means, so it is not a mean of
    anything and its bootstrap distribution is skewed by construction; the
    percentile interval is reported without a normal-theory alternative.
    """
    if not cells_a or not cells_b:
        return None
    one = {}
    for city, arr in cells_a + cells_b:
        one.setdefault(city, arr)
    point_a = max(a.mean() for _, a in cells_a)
    point_b = max(a.mean() for _, a in cells_b)
    reps = np.empty(b, dtype=np.float64)
    a_smaller = 0
    for i, draw in enumerate(city_index_draws(one, b=b, seed=seed)):
        wa = max(arr[draw[city]].mean() for city, arr in cells_a)
        wb = max(arr[draw[city]].mean() for city, arr in cells_b)
        reps[i] = wa - wb
        if wa < wb:
            a_smaller += 1
    lo, hi = (float(x) for x in np.percentile(reps, [2.5, 97.5]))
    p = 2.0 * min(float((reps <= 0).mean()), float((reps >= 0).mean()))
    return dict(worst_case_a=float(point_a), worst_case_b=float(point_b),
                difference=float(point_a - point_b),
                ci95=[lo, hi], ci_method=f"paired_image_cluster_bootstrap_B{b}",
                p=float(min(1.0, max(1.0 / b, p))), B=b, two_tailed=True,
                bootstrap_sd=float(reps.std(ddof=1)),
                bootstrap_median=float(np.median(reps)),
                share_a_worst_case_smaller=float(a_smaller / b),
                share_definition="fraction of the B replicates in which the "
                                 "first method's worst case is strictly "
                                 "smaller, i.e. its largest single-cell "
                                 "shortfall is the lesser of the two")


def cluster_mean(cells):
    """results/27's image-cluster bootstrap on a mean over cells, reused."""
    by_city = {}
    for city, arr in cells:
        by_city.setdefault(city, []).append(arr)
    return M.cluster_bootstrap_mean_over_cells(by_city)


# ----------------------------------------------------------------------------
# the cell table
# ----------------------------------------------------------------------------

def build_cells(rows):
    """One record per (method, arch, city) that has a value, with the per-image
    vectors every metric below is computed from."""
    out = []
    for method in METHODS:
        for arch in list(ARCHS) + [CE]:
            for city in CITIES:
                r = rows.get((method, arch, city))
                if r is None:
                    continue
                u = rows[("upper", arch, city)]
                v = rows[("vanilla", arch, city)]
                if r.get("per_image") is None or u.get("per_image") is None:
                    out.append(dict(
                        method=method, arch=arch, city=city,
                        scope=("9_cell" if arch != CE else "crossearth"),
                        access_class=r["access_class"], has_value=False,
                        run_id=r.get("run_id"), blocker=r.get("blocker"),
                        mean_gap=None, upper_mean=None, rel_gap=None))
                    continue
                m = np.asarray(r["per_image"], dtype=np.float64)
                uu = np.asarray(u["per_image"], dtype=np.float64)
                gap = uu - m
                vv = (None if v.get("per_image") is None
                      else np.asarray(v["per_image"], dtype=np.float64))
                out.append(dict(
                    method=method, arch=arch, city=city,
                    scope=("9_cell" if arch != CE else "crossearth"),
                    access_class=r["access_class"], has_value=True,
                    run_id=r.get("run_id"), tree=r.get("tree"),
                    n450_substituted=r.get("n450_substituted"),
                    n=int(m.size),
                    method_per_image=m, upper_per_image=uu,
                    vanilla_per_image=vv, gap_per_image=gap,
                    method_mean=float(m.mean()), upper_mean=float(uu.mean()),
                    mean_gap=float(gap.mean()),
                    rel_gap=float(100.0 * gap.mean() / uu.mean()),
                    vs_vanilla_per_image=(None if vv is None else m - vv),
                    vs_vanilla_mean=(None if vv is None
                                     else float((m - vv).mean())),
                    blocker=None))
    return out


def select(cells, method, scope, access=None):
    keep = [c for c in cells if c["method"] == method and c["has_value"]]
    if scope == "9_cell":
        keep = [c for c in keep if c["arch"] != CE]
    if access is not None:
        keep = [c for c in keep if c["access_class"] == access]
    return sorted(keep, key=lambda c: (c["arch"], c["city"]))


def key(c):
    return f"{c['arch']}/{c['city']}"


# ----------------------------------------------------------------------------
# M1 - M9, one method, one cell set
# ----------------------------------------------------------------------------

def profile(method, cells, scope, access=None):
    lbl = f"{method}/{scope}" + (f"/{access}" if access else "")
    e = dict(method=method, method_label=METHOD_LABEL[method], scope=scope,
             access_restriction=access, n_cells=len(cells),
             cells=[key(c) for c in cells],
             has_crossearth_counterpart=any(c["arch"] == CE for c in cells),
             access_classes=sorted({c["access_class"] for c in cells}),
             n_cells_n450_substituted=sum(1 for c in cells
                                          if c.get("n450_substituted")))
    if not cells:
        for k in ("M1_worst_case_gap", "M1_worst_cell", "M2_mean_gap",
                  "M3_worst_case_relative_gap", "M3_worst_cell",
                  "M4_mean_relative_gap", "M5_harm_count", "M6_benefit_count",
                  "M7_largest_harm", "M7_largest_harm_cell",
                  "M8_n_cells_gap_not_significant", "M9_gap_sd_across_cells"):
            e[k] = None
        e["why_empty"] = ("no cell of this method falls in this scope and "
                          "access class")
        return e, []

    tests = []

    # ---- M1 -------------------------------------------------------------
    w = max(cells, key=lambda c: c["mean_gap"])
    e.update(M1_worst_case_gap=w["mean_gap"], M1_worst_cell=key(w),
             M1_worst_cell_run_id=w["run_id"],
             M1_best_case_gap=min(c["mean_gap"] for c in cells),
             M1_best_cell=key(min(cells, key=lambda c: c["mean_gap"])))

    # ---- M2 -------------------------------------------------------------
    cb = cell_bootstrap([c["mean_gap"] for c in cells])
    icb = cluster_mean([(c["city"], c["gap_per_image"]) for c in cells])
    e.update(M2_mean_gap=cb["mean"], M2_mean_gap_ci95=cb["ci95"],
             M2_ci_method=cb["ci_method"],
             M2_bootstrap_sd=cb["bootstrap_sd"],
             M2_mean_gap_image_cluster_ci95=icb["ci95"],
             M2_mean_gap_image_cluster_stderr=icb["stderr"],
             M2_mean_gap_image_cluster_p=icb["p"],
             M2_mean_gap_image_cluster_ci_method=icb["ci_method"])
    tests.append(dict(
        name="paired image-cluster bootstrap on the mean gap to the upper bound",
        metric="M2", compares=[lbl, "upper"], statistic=icb["mean"],
        p=icb["p"], B=B, two_tailed=True,
        clustering="images resampled within held-out city, shared across the "
                   "cells of that city"))

    # ---- M3, M4 ---------------------------------------------------------
    wr = max(cells, key=lambda c: c["rel_gap"])
    rb = cell_bootstrap([c["rel_gap"] for c in cells])
    e.update(M3_worst_case_relative_gap=wr["rel_gap"], M3_worst_cell=key(wr),
             M3_worst_cell_upper_value=wr["upper_mean"],
             M4_mean_relative_gap=rb["mean"],
             M4_mean_relative_gap_ci95=rb["ci95"],
             M4_ci_method=rb["ci_method"])

    # ---- M5, M6, M7 -----------------------------------------------------
    if method == "vanilla":
        e.update(M5_harm_count=None, M5_harm_cells=None,
                 M6_benefit_count=None, M6_benefit_cells=None,
                 M7_largest_harm=None, M7_largest_harm_cell=None,
                 M5_M6_M7_why_none=("Vanilla is the comparator of M5, M6 and "
                                    "M7, so these are not defined for it. The "
                                    "test would be Vanilla against itself: a "
                                    "per-image difference identically zero, a "
                                    "degenerate bootstrap and a p-value with "
                                    "no meaning. Reported as null rather than "
                                    "silently skipped."),
                 M5_M6_vanilla_cells=None)
    else:
        usable = [c for c in cells if c["vs_vanilla_per_image"] is not None]
        missing_v = [key(c) for c in cells
                     if c["vs_vanilla_per_image"] is None]
        bs = [paired_bootstrap(c["vs_vanilla_per_image"]) for c in usable]
        adj = holm([x["p"] for x in bs]) if bs else []
        detail = []
        for c, x, a in zip(usable, bs, adj):
            detail.append(dict(cell=key(c), run_id=c["run_id"],
                               method_minus_vanilla=x["mean"], ci95=x["ci95"],
                               p=x["p"], p_holm=a,
                               significant=bool(a < ALPHA),
                               direction=("worse" if x["mean"] < 0
                                          else "better")))
            tests.append(dict(
                name="paired image-level bootstrap, method minus Vanilla",
                metric="M5/M6", compares=[c["run_id"], "vanilla " + key(c)],
                cell=key(c), scope=scope, access_restriction=access,
                statistic=x["mean"], ci95=x["ci95"], p=x["p"], p_holm=a,
                holm_family=f"{METHOD_LABEL[method]} cells against Vanilla, "
                            f"m = {len(usable)}, scope {scope}"
                            + (f", access {access}" if access else ""),
                B=B, two_tailed=True))
        harm = [d["cell"] for d in detail
                if d["significant"] and d["method_minus_vanilla"] < 0]
        ben = [d["cell"] for d in detail
               if d["significant"] and d["method_minus_vanilla"] > 0]
        lh = min(usable, key=lambda c: c["vs_vanilla_mean"]) if usable else None
        e.update(M5_harm_count=len(harm), M5_harm_cells=harm,
                 M6_benefit_count=len(ben), M6_benefit_cells=ben,
                 M5_M6_per_cell=detail,
                 M5_M6_cells_without_a_vanilla_comparator=missing_v,
                 M5_M6_holm_m=len(usable),
                 M7_largest_harm=(None if lh is None else lh["vs_vanilla_mean"]),
                 M7_largest_harm_cell=(None if lh is None else key(lh)),
                 M7_definition_note=("min over cells of the cell-mean "
                                     "difference, so it is the largest "
                                     "shortfall against Vanilla whether or not "
                                     "that cell's test is significant; the "
                                     "cell's own p and Holm-adjusted p are in "
                                     "M5_M6_per_cell"))

    # ---- M8 -------------------------------------------------------------
    gbs = [paired_bootstrap(-c["gap_per_image"]) for c in cells]   # method-upper
    gadj = holm([x["p"] for x in gbs])
    g8 = []
    for c, x, a in zip(cells, gbs, gadj):
        g8.append(dict(cell=key(c), run_id=c["run_id"], gap=c["mean_gap"],
                       p=x["p"], p_holm=a,
                       significant=bool(a < ALPHA)))
        tests.append(dict(
            name="paired image-level bootstrap, method minus its within-city "
                 "upper bound",
            metric="M8", compares=[c["run_id"], "upper " + key(c)],
            cell=key(c), scope=scope, access_restriction=access,
            statistic=x["mean"], ci95=x["ci95"], p=x["p"], p_holm=a,
            holm_family=f"{METHOD_LABEL[method]} cells against their own upper "
                        f"bounds, m = {len(cells)}, scope {scope}"
                        + (f", access {access}" if access else ""),
            B=B, two_tailed=True))
    ns = [d["cell"] for d in g8 if not d["significant"]]
    e.update(M8_n_cells_gap_not_significant=len(ns),
             M8_cells_gap_not_significant=ns, M8_per_cell=g8,
             M8_holm_m=len(cells), M8_alpha=ALPHA)

    # ---- M9 -------------------------------------------------------------
    gaps = np.asarray([c["mean_gap"] for c in cells], dtype=np.float64)
    e.update(M9_gap_sd_across_cells=(float(gaps.std(ddof=1))
                                     if gaps.size > 1 else None),
             M9_gap_range=[float(gaps.min()), float(gaps.max())],
             M9_n_cells=int(gaps.size))
    return e, tests


# ----------------------------------------------------------------------------
# M10, M11: SIB against every other method
# ----------------------------------------------------------------------------

def sib_head_to_heads(cells, scope, access=None):
    rows, tests = [], []
    sib = {key(c): c for c in select(cells, "sib", scope, access)}
    for method in METHODS:
        if method == "sib":
            continue
        other = {key(c): c for c in select(cells, method, scope, access)}
        shared = sorted(set(sib) & set(other))
        rec = dict(
            comparison=f"sib_vs_{method}", other_method=method,
            other_method_label=METHOD_LABEL[method], scope=scope,
            access_restriction=access,
            n_cells_shared=len(shared), cells=shared,
            sib_cells_not_shared=sorted(set(sib) - set(other)),
            other_cells_not_shared=sorted(set(other) - set(sib)))
        if not shared:
            rec.update(M10_sib_better=None, M10_sib_worse=None,
                       M10_not_significant=None, M11=None,
                       why_empty="the two methods share no cell in this scope "
                                 "and access class")
            rows.append(rec)
            continue
        # ---- M10 --------------------------------------------------------
        diffs = [sib[k]["method_per_image"] - other[k]["method_per_image"]
                 for k in shared]
        bs = [paired_bootstrap(d) for d in diffs]
        adj = holm([x["p"] for x in bs])
        detail = []
        for k, x, a in zip(shared, bs, adj):
            detail.append(dict(cell=k, sib_minus_other=x["mean"],
                               ci95=x["ci95"], p=x["p"], p_holm=a,
                               significant=bool(a < ALPHA),
                               direction=("sib_better" if x["mean"] > 0
                                          else "sib_worse")))
            tests.append(dict(
                name="paired image-level bootstrap, SIB minus the other method",
                metric="M10", compares=[sib[k]["run_id"], other[k]["run_id"]],
                comparison=f"sib_vs_{method}", cell=k, scope=scope,
                access_restriction=access,
                statistic=x["mean"], ci95=x["ci95"], p=x["p"], p_holm=a,
                holm_family=f"SIB against {METHOD_LABEL[method]}, m = "
                            f"{len(shared)} cells, scope {scope}"
                            + (f", access {access}" if access else ""),
                B=B, two_tailed=True))
        rec.update(
            M10_sib_better=sum(1 for d in detail
                               if d["significant"] and d["sib_minus_other"] > 0),
            M10_sib_worse=sum(1 for d in detail
                              if d["significant"] and d["sib_minus_other"] < 0),
            M10_not_significant=sum(1 for d in detail if not d["significant"]),
            M10_per_cell=detail, M10_holm_m=len(shared), M10_alpha=ALPHA,
            M10_mean_sib_minus_other=float(np.mean(
                [d["sib_minus_other"] for d in detail])))
        # ---- M11 --------------------------------------------------------
        wd = worst_case_diff_bootstrap(
            [(sib[k]["city"], sib[k]["gap_per_image"]) for k in shared],
            [(other[k]["city"], other[k]["gap_per_image"]) for k in shared])
        rec["M11"] = dict(
            sib_worst_case_gap=wd["worst_case_a"],
            other_worst_case_gap=wd["worst_case_b"],
            difference_sib_minus_other=wd["difference"],
            ci95=wd["ci95"], ci_method=wd["ci_method"], p=wd["p"],
            bootstrap_sd=wd["bootstrap_sd"],
            bootstrap_median=wd["bootstrap_median"],
            share_sib_worst_case_smaller=wd["share_a_worst_case_smaller"],
            share_definition=wd["share_definition"], B=B, n_cells=len(shared),
            note="both worst cases are taken over the same cell set, so the "
                 "difference is not a coverage artefact; the statistic is a "
                 "max of means and its bootstrap distribution is skewed by "
                 "construction, so only the percentile interval is given")
        tests.append(dict(
            name="image-cluster bootstrap on the difference of worst-case gaps",
            metric="M11", comparison=f"sib_vs_{method}", scope=scope,
            access_restriction=access, compares=["sib", method],
            statistic=wd["difference"], ci95=wd["ci95"], p=wd["p"], B=B,
            two_tailed=True, n_cells=len(shared),
            note="no Holm correction inside M11 itself: one comparison per "
                 "method pair. The SIB-against-FDA member is corrected inside "
                 "the declared multiplicity family."))
        rows.append(rec)
    return rows, tests


# ----------------------------------------------------------------------------
# the declared multiplicity family
# ----------------------------------------------------------------------------

def sib_fda_family(profiles, h2h, tests, scope="9_cell"):
    """Every p-value bearing on SIB against FDA, in one Holm family."""
    members = []

    def add(source, label, cell, raw, extra=None):
        m = dict(source=source, label=label, cell=cell, p_raw=raw)
        if extra:
            m.update(extra)
        members.append(m)

    for meth in ("sib", "fda"):
        p = next(e for e in profiles
                 if e["method"] == meth and e["scope"] == scope
                 and e["access_restriction"] is None)
        for d in p["M5_M6_per_cell"]:
            add("M5/M6", f"{meth} minus vanilla", d["cell"], d["p"],
                dict(statistic=d["method_minus_vanilla"],
                     p_holm_within_metric=d["p_holm"]))
        for d in p["M8_per_cell"]:
            add("M8", f"{meth} minus its upper bound", d["cell"], d["p"],
                dict(statistic=d["gap"], p_holm_within_metric=d["p_holm"]))
    hh = next(r for r in h2h
              if r["comparison"] == "sib_vs_fda" and r["scope"] == scope
              and r["access_restriction"] is None)
    for d in hh["M10_per_cell"]:
        add("M10", "sib minus fda", d["cell"], d["p"],
            dict(statistic=d["sib_minus_other"],
                 p_holm_within_metric=d["p_holm"]))
    add("M11", "sib worst case minus fda worst case", "all cells of the scope",
        hh["M11"]["p"], dict(statistic=hh["M11"]["difference_sib_minus_other"],
                             p_holm_within_metric=None))

    adjusted = holm([m["p_raw"] for m in members])
    for m, a in zip(members, adjusted):
        m["p_holm_family"] = a
        m["significant_after_family_holm"] = bool(a < ALPHA)
    return dict(
        **MULTIPLICITY_FAMILY,
        alpha=ALPHA,
        m=len(members),
        n_members_by_source={s: sum(1 for m in members if m["source"] == s)
                             for s in ("M5/M6", "M8", "M10", "M11")},
        double_counting_note=("M5 and M6 are one two-tailed test per cell -- "
                              "the sign decides which count a cell lands in -- "
                              "so each cell contributes ONE p-value to this "
                              "family and not two. Entering it twice would "
                              "inflate m and make every adjusted p "
                              "conservative for no reason."),
        n_significant_after_family_holm=sum(
            1 for m in members if m["significant_after_family_holm"]),
        n_significant_before=sum(1 for m in members if m["p_raw"] < ALPHA),
        members=members)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    hr = "=" * 100
    print(hr)
    print("Arm F -- method comparison profile on the n450 grid "
          "(POST HOC / EXPLORATORY)")
    print(f"DRY_RUN = {DRY_RUN}   B = {B}   ALPHA = {ALPHA}")
    print(hr)
    print("\nPRE-DECLARED METRICS (frozen before any value was computed):")
    for m in METRICS:
        print(f"  {m['id']:4s} {m['name']}")
    print(f"  methods: {', '.join(METHODS)}")
    print(f"  scopes : {', '.join(SCOPES)}")

    # ---- indexes --------------------------------------------------------
    trees = [("st_r450_published_png", M.PNG),
             ("st_r450_published_recompute", M.RECOMPUTE),
             ("st", M.ST), ("e12", M.E12)]
    indexes = {}
    for label, root in trees:
        _, info = M.build_index(
            label, root,
            write_to=(None if DRY_RUN
                      else os.path.join(INV, f"INDEX_{label}_32.parquet")))
        indexes[label] = info
    png_info = indexes["st_r450_published_png"]
    print("\n--- indexes (built here; the _v3, _26 and _27 indexes prior "
          "results JSONs cite by sha256 are untouched) ---")
    for lbl, i in indexes.items():
        print(f"  {lbl:32s} rows {i['n_rows']:4d}  sha256 "
              f"{(i.get('sha256') or 'not written in a dry run')[:16]}...")

    # ---- rows, from the resolver that wrote results/31 ------------------
    print("\n--- resolving rows through "
          "analysis/31_internal_sib_worst_case_tgtfda.py ---")
    rows = M.resolve_all_rows()
    cells = build_cells(rows)
    have = [c for c in cells if c["has_value"]]
    print(f"  {len(have)} cells with a value, "
          f"{len(cells) - len(have)} without")
    for c in cells:
        if not c["has_value"]:
            print(f"    no value: {c['method']}/{c['arch']}/{c['city']}: "
                  f"{c['blocker']}")

    # ---- assert every cell against results/31 ---------------------------
    if not os.path.isfile(R27):
        fail(f"{os.path.relpath(R27, BASE)} is absent; this analysis is defined "
             "as the profile of that JSON's rows and will not run without it")
    r27 = json.load(open(R27))
    r27pc = {(e["method"], e["arch"], e["city"]): e for e in r27["per_cell"]}
    checked, worst = 0, 0.0
    for c in have:
        a = r27pc.get((c["method"], c["arch"], c["city"]))
        if a is None:
            fail(f"{c['method']}/{c['arch']}/{c['city']}: this cell has a value "
                 "here and no row in results/31; the two must be the same grid")
        for fld, got in (("method_value", c["method_mean"]),
                         ("upper_value", c["upper_mean"]),
                         ("gap_upper_minus_method", c["mean_gap"])):
            if a.get(fld) is None:
                fail(f"{c['method']}/{c['arch']}/{c['city']}: results/31 "
                     f"carries no {fld} for a cell this job has a value for")
            d = abs(float(a[fld]) - got)
            worst = max(worst, d)
            if d > 1e-9:
                fail(f"{c['method']}/{c['arch']}/{c['city']}: {fld} "
                     f"{got:.12f} here against results/31's {a[fld]:.12f}, "
                     f"|delta| {d:.3e}. These rows are supposed to be the same "
                     "resolver on the same dumps; any disagreement is a defect "
                     "in this script.")
            checked += 1
    r27_agreement = dict(
        n_values_checked=checked, max_abs_delta=worst, tolerance=1e-9,
        fields=["method_value", "upper_value", "gap_upper_minus_method"],
        what="every cell's method mean, upper-bound mean and mean gap, "
             "asserted against results/31's per_cell record. The rows are "
             "resolved by importing that job's own resolve_all_rows(), so this "
             "is a check that the import did what it claims, not an "
             "independent measurement.")
    print(f"  asserted {checked} cell values against results/31, max |delta| "
          f"{worst:.2e}")

    # ---- the profile ----------------------------------------------------
    profiles, tests = [], []
    for scope in SCOPES:
        for method in METHODS:
            e, t = profile(method, select(cells, method, scope), scope)
            profiles.append(e)
            tests.extend(t)
    for scope in SCOPES:
        for access in ACCESS_RESTRICTIONS:
            for method in METHODS:
                e, t = profile(method, select(cells, method, scope, access),
                               scope, access)
                profiles.append(e)
                tests.extend(t)

    h2h, t2 = [], []
    for scope in SCOPES:
        a, b = sib_head_to_heads(cells, scope)
        h2h.extend(a)
        t2.extend(b)
    for scope in SCOPES:
        for access in ACCESS_RESTRICTIONS:
            a, b = sib_head_to_heads(cells, scope, access)
            h2h.extend(a)
            t2.extend(b)
    tests.extend(t2)

    # ---- print the profile ----------------------------------------------
    def show(scope, access):
        sel = [e for e in profiles if e["scope"] == scope
               and e["access_restriction"] == access]
        cols = ["method", "n_cells", "M1_worst_case_gap", "M1_worst_cell",
                "M2_mean_gap", "M3_worst_case_relative_gap",
                "M4_mean_relative_gap", "M5_harm_count", "M6_benefit_count",
                "M7_largest_harm", "M8_n_cells_gap_not_significant",
                "M9_gap_sd_across_cells"]
        df = pd.DataFrame([{c: e.get(c) for c in cols} for e in sel])
        ttl = f"M1-M9, scope {scope}" + (f", {access} cells only" if access
                                         else ", all cells")
        print(f"\n--- {ttl} ---")
        print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    for scope in SCOPES:
        show(scope, None)
    for scope in SCOPES:
        for access in ACCESS_RESTRICTIONS:
            show(scope, access)

    for scope in SCOPES:
        for access in [None] + ACCESS_RESTRICTIONS:
            sel = [r for r in h2h if r["scope"] == scope
                   and r["access_restriction"] == access]
            ttl = (f"M10 / M11, SIB against every other method, scope {scope}"
                   + (f", {access} cells only" if access else ""))
            print(f"\n--- {ttl} ---")
            for r in sel:
                if r.get("M11") is None:
                    print(f"  sib vs {r['other_method']:10s} n=0  "
                          f"{r.get('why_empty')}")
                    continue
                m11 = r["M11"]
                print(f"  sib vs {r['other_method']:10s} n={r['n_cells_shared']:2d}  "
                      f"M10 better/worse/ns "
                      f"{r['M10_sib_better']}/{r['M10_sib_worse']}/"
                      f"{r['M10_not_significant']}  "
                      f"M11 {m11['sib_worst_case_gap']:+.4f} vs "
                      f"{m11['other_worst_case_gap']:+.4f} diff "
                      f"{m11['difference_sib_minus_other']:+.4f} "
                      f"CI95 [{m11['ci95'][0]:+.4f}, {m11['ci95'][1]:+.4f}] "
                      f"p {m11['p']:.4f} share_sib_smaller "
                      f"{m11['share_sib_worst_case_smaller']:.4f}")

    # ---- the declared multiplicity family -------------------------------
    family = sib_fda_family(profiles, h2h, tests)
    print(f"\n--- the SIB-against-FDA multiplicity family "
          f"(m = {family['m']}, Holm) ---")
    for m in family["members"]:
        print(f"  {m['source']:6s} {m['label']:34s} {m['cell']:22s} "
              f"stat {m['statistic']:+9.4f}  p {m['p_raw']:.4f}  "
              f"Holm {m['p_holm_family']:.4f}  "
              f"{'sig' if m['significant_after_family_holm'] else 'n.s.'}")
    print(f"  significant before correction "
          f"{family['n_significant_before']} of {family['m']}; after Holm "
          f"{family['n_significant_after_family_holm']} of {family['m']}")

    # ---- sanity flags ---------------------------------------------------
    sanity = []
    sanity.append(
        "This whole analysis is POST HOC. It was commissioned after "
        "results/26's adjudication fired reading R3, which is to say after we "
        "had already seen that FDA's worst-case gap sits at or below SIB's. No "
        "p-value here was pre-committed and no reading of any of these twelve "
        "metrics was registered in REVIEWS.md before the numbers existed. What "
        "the pre-declaration in this script buys is narrower and worth stating "
        "exactly: the metric list, the method list, the scopes, the "
        "significance level and the multiplicity family were fixed in code "
        "before any value was computed, and every metric is reported for every "
        "method, so the selection of WHICH comparison to show is not "
        "data-dependent. That is protection against cherry-picking within this "
        "job. It is not protection against the choice to run this job at all, "
        "and it does not convert any number here into confirmatory evidence.")
    # SAR against Vanilla is an identity, not a measurement
    sar_ident = []
    for c in select(cells, "sar", "9_cell"):
        r = rows[("sar", c["arch"], c["city"])]
        if r.get("sar_delta_per_image") is None:
            continue
        d = float(np.abs(np.asarray(r["sar_delta_per_image"])
                         - c["vs_vanilla_per_image"]).max())
        sar_ident.append((key(c), d))
    if sar_ident:
        sanity.append(
            "M5, M6 and M7 are NOT independent measurements on the nine SAR "
            "rows, and the reader must not read them as such. A SAR row is "
            "constructed as its Vanilla cell's per-image value plus SAR's own "
            "within-pass change, so (SAR minus Vanilla) per image IS that "
            "within-pass change, exactly and by construction. Measured here: "
            "the largest absolute difference between the two quantities is "
            + ", ".join(f"{k} {d:.2e}" for k, d in sar_ident)
            + ". So SAR's M5/M6/M7 restate the size of SAR's adaptation step "
              "and nothing else, and its M8 and its gap metrics inherit the "
              "Vanilla cell's level. SAR's own near-zero effect "
              "(pilot_verdict no_meaningful_adaptation, not re-selected on the "
              "n450 models) is why its whole profile sits within a few "
              "hundredths of Vanilla's.")
    # FDA and SIB access classes, which the brief asks be stated plainly
    sib_cells = select(cells, "sib", "9_cell")
    fda_cells = select(cells, "fda", "9_cell")
    sanity.append(
        "The headline comparison is not within one access class. FDA uses "
        "unlabelled held-out-city imagery in "
        f"{sum(1 for c in fda_cells if c['access_class'] == 'unlabeled_target')}"
        f" of its {len(fda_cells)} cells on the 9-cell scope; SIB uses it in "
        f"{sum(1 for c in sib_cells if c['access_class'] == 'unlabeled_target')}"
        f" of {len(sib_cells)}, the three MAMNet cells, through the FDA stage "
        "of its own recipe, and its OGLANet and DINOv3 cells are source_only "
        "because the SIB loader never dereferences fda_target_root. So every "
        "pooled SIB-against-FDA number in this JSON compares a method that "
        "touches the target domain everywhere against one that touches it in a "
        "third of its cells. M12 is the restriction that removes the mix, and "
        "it removes FDA entirely rather than equalising it: FDA has zero "
        "source_only cells.")
    # a method scoring above its own upper bound
    above = [c for c in have if c["mean_gap"] < 0]
    if above:
        sanity.append(
            "Cell(s) with a NEGATIVE gap, i.e. scoring above their own "
            "within-city upper bound: "
            + ", ".join(f"{c['method']}/{key(c)} {c['mean_gap']:+.4f}"
                        for c in above)
            + ". Carried from results/31, where the same pair is flagged: the "
              "published Table 38 prints it the same way and the paired "
              "bootstrap does not separate it from zero. It matters for M3 and "
              "M4, where it enters as a negative relative gap and pulls a mean "
              "toward zero rather than being clipped.")
    # M11's skew
    skews = []
    for r in h2h:
        if r["scope"] != "9_cell" or r["access_restriction"] is not None:
            continue
        if r.get("M11") is None:
            continue
        m11 = r["M11"]
        if abs(m11["bootstrap_median"] - m11["difference_sib_minus_other"]) > 0.25:
            skews.append((r["other_method"], m11["difference_sib_minus_other"],
                          m11["bootstrap_median"]))
    sanity.append(
        "M11's statistic is a difference of two maxima, so its bootstrap "
        "distribution is skewed by construction and the observed point is not "
        "the centre of its own replicates. "
        + (("Measured gap between point and bootstrap median above 0.25 mIoU "
            "on: " + ", ".join(f"sib_vs_{m} point {p:+.4f} median {md:+.4f}"
                               for m, p, md in skews) + ". ")
           if skews else
           "Measured: no comparison on the 9-cell scope shows a point-minus-"
           "median gap above 0.25 mIoU. ")
        + "A max over cells can only move up when a replicate makes any one "
          "cell worse, which biases the replicate maxima upward relative to "
          "the observed max; the percentile interval is reported as-is and no "
          "bias correction is applied.")
    # how often a cell count is driven by one architecture
    og = [c for c in have if c["arch"] == "oglanet"]
    wc9 = {e["method"]: e["M1_worst_cell"] for e in profiles
           if e["scope"] == "9_cell" and e["access_restriction"] is None
           and e["M1_worst_cell"] is not None}
    wc_og = sorted(m for m, k in wc9.items() if k.startswith("oglanet/"))
    wc_mamph = sorted(m for m, k in wc9.items() if k == "mamnet/phoenix")
    sanity.append(
        f"OGLANet supplies {len(og)} of the {len(have)} cells with a value, and "
        "every OGLANet number is one draw of a nondeterministic, "
        "inference-batch-size-dependent pipeline (the DFFM defect, "
        "results/21_disclosure_record.json finding 3). Measured concentration "
        f"of the worst-case column on the 9-cell scope: {len(wc_og)} of "
        f"{len(wc9)} methods take their worst case on an OGLANet cell "
        + (f"({', '.join(wc_og)})" if wc_og else "(none)")
        + f" and {len(wc_mamph)} take it on MAMNet/Phoenix "
        + (f"({', '.join(wc_mamph)})" if wc_mamph else "(none)")
        + f", so {len(wc_og) + len(wc_mamph)} of {len(wc9)} worst cases sit on "
          "one of those two. The distinct worst cells and their counts are "
        + ", ".join(f"{k} x{sum(1 for v in wc9.values() if v == k)}"
                    for k in sorted(set(wc9.values())))
        + ". The worst-case column is also the column with the least averaging "
          "in it: one cell, one seed, one inference pass, and for OGLANet one "
          "draw of a pipeline whose measured nine-pass span is 1.1158 two-class "
          "mIoU.")
    sanity.append(
        "The counts in M5, M6, M8 and M10 are counts of significance "
        "decisions, so they move with the Holm family they sit in and with "
        "ALPHA, neither of which is a property of the data. Each count is "
        "reported beside the per-cell raw p, the per-cell Holm-adjusted p and "
        "the point estimate, so a reader who prefers a different threshold or "
        "a different family can recount without rerunning anything. The "
        "SIB-against-FDA family is the one place where this job corrects "
        "across metrics rather than within one.")

    # ---- CONTRACT section 6 conditions ---------------------------------
    conditions = []
    for e in profiles:
        if e["M2_mean_gap"] is None:
            continue
        conditions.append(dict(
            label=f"{e['method_label']} / {e['scope']}"
                  + (f" / {e['access_restriction']} cells only"
                     if e["access_restriction"] else "")
                  + " / mean gap to the within-city upper bound",
            metric="gap_upper_minus_method_mIoU_tol_K2_pct",
            mean=e["M2_mean_gap"], stderr=e["M2_bootstrap_sd"],
            n=e["n_cells"], n_images=150 * e["n_cells"],
            ci95=e["M2_mean_gap_ci95"], ci_method=e["M2_ci_method"],
            ci95_image_cluster=e["M2_mean_gap_image_cluster_ci95"],
            baseline="within-city upper bound, same architecture and city",
            access_class=e["access_classes"],
            worst_case_gap=e["M1_worst_case_gap"],
            worst_cell=e["M1_worst_cell"]))

    # ---- headline -------------------------------------------------------
    p9 = {e["method"]: e for e in profiles
          if e["scope"] == "9_cell" and e["access_restriction"] is None}
    rank = sorted((e for e in p9.values()), key=lambda e: e["M1_worst_case_gap"])
    sibfda = next(r for r in h2h if r["comparison"] == "sib_vs_fda"
                  and r["scope"] == "9_cell"
                  and r["access_restriction"] is None)
    headline = (
        "POST HOC / EXPLORATORY profile, commissioned after results/26's "
        "reading fired R3 and reporting all twelve pre-declared metrics for "
        "all ten methods: on the 9-cell n450 grid the smallest worst-case gap "
        "to the within-city upper bound is "
        f"{rank[0]['M1_worst_case_gap']:.2f} mIoU "
        f"({rank[0]['method_label']}, at {rank[0]['M1_worst_cell']}, n="
        f"{rank[0]['n_cells']}) against SIB's "
        f"{p9['sib']['M1_worst_case_gap']:.2f} at "
        f"{p9['sib']['M1_worst_cell']} and Vanilla's "
        f"{p9['vanilla']['M1_worst_case_gap']:.2f} at "
        f"{p9['vanilla']['M1_worst_cell']}; SIB's worst case minus FDA's is "
        f"{sibfda['M11']['difference_sib_minus_other']:+.2f} mIoU, 95% CI "
        f"[{sibfda['M11']['ci95'][0]:+.2f}, {sibfda['M11']['ci95'][1]:+.2f}], "
        f"p {sibfda['M11']['p']:.4f}, with SIB's worst case the smaller in "
        f"{sibfda['M11']['share_sib_worst_case_smaller'] * 100:.1f}% of "
        f"{B} replicates; per cell SIB is significantly better than FDA on "
        f"{sibfda['M10_sib_better']} of {sibfda['n_cells_shared']} cells, "
        f"significantly worse on {sibfda['M10_sib_worse']} and not separable "
        f"on {sibfda['M10_not_significant']}; and across the 46-test "
        "SIB-against-FDA family "
        f"{family['n_significant_after_family_holm']} of {family['m']} "
        "p-values survive Holm.")

    # ---- caveats --------------------------------------------------------
    r27_caveats = r27["caveats"]
    caveats = [
        "POST HOC. This profile was chosen after results/26's adjudication "
        "fired reading R3, i.e. after the result it characterises was already "
        "known. It is exploratory and it is not confirmatory evidence for or "
        "against any claim in the paper. CONVENTIONS.md's rule that designing "
        "for an outcome and then observing it is the design and not the "
        "evidence applies to this JSON in full.",

        "ALL METRICS REPORTED. The twelve metrics, the ten methods, the two "
        "scopes, the access restriction, ALPHA = 0.05 and the multiplicity "
        "family were fixed in code before any value was computed (the METRICS "
        "list in analysis/28_method_comparison_profile.py), and every metric is "
        "emitted for every method including where it is degenerate or empty. "
        "Nothing was added, dropped or reordered after the values appeared. "
        "Vanilla's M5, M6 and M7 are null because Vanilla is their comparator; "
        "FDA, SegDeSiC and SAR have n = 0 in M12 because none of their cells is "
        "source_only. Those are declared absences, not omissions.",

        "No pre-committed reading exists for any of these twelve metrics. "
        "REVIEWS.md's pre-commitments R1-R4 are about SIB's worst case against "
        "the other methods' worst cases, which is M1 and M11 here, and they "
        "were adjudicated in results/22, results/26 and results/27. This job "
        "does not re-adjudicate them and does not fire a reading. Whether any "
        "number here changes a sentence in the paper is a scoping decision for "
        "the operator and the HLT.",

        "Single seed per n450 cell. Each of the 27 matched-450 retrains is one "
        "draw (s42) and training is not reproducible across jobs at a fixed "
        "seed on this codebase, so every worst-case cell, every harm count and "
        "every M11 point rests on one training run per cell. The image-level "
        "bootstraps here resample images, never seeds, so no interval in this "
        "JSON contains any seed variance. The one exception is MAMNet-SIB, "
        "whose row is the mean over three E-12 clean-pool seeds {42, 43, 44}; "
        "that row therefore has a different noise structure from every other "
        "row and its per-image vector is a seed mean, which makes its "
        "within-cell variance smaller than a single-seed row's for reasons "
        "unrelated to the method.",

        "DFFM. OGLANet inference is nondeterministic and "
        "inference-batch-size dependent because python/oglanet/models/dffm.py "
        "builds about 20 freshly random-initialised, unregistered "
        "Conv2d+BatchNorm2d layers inside forward() on every call, so "
        "model.eval() never reaches them and they normalise with batch "
        "statistics. Measured spread on one OGLANet cell: foreground IoU at "
        "K=2 of 33.05 at batch 1, 37.33 at batch 2, 39.62 at batch 4, 40.55 at "
        "batch 8. Every OGLANet value in this profile is one draw at one batch "
        "size. How much of this profile's worst-case column sits on OGLANet, "
        "and on MAMNet/Phoenix, is measured and reported in the sanity flags "
        "rather than asserted here. "
        "Erratum E-9's measured OGLANet nine-pass span is 1.1158 two-class "
        "mIoU and its cell-mean SD reaches 0.4247, which is larger than "
        "several of the M11 differences reported here.",

        "DFFM, second part. The OGLANet inference pass carries no run-to-run "
        "nondeterminism at a FIXED batch size -- five repeat passes of "
        "oglanet-sibc4-loco-r450-chicago-hr-s42-n450 on five different nodes "
        "returned 68.531270 two-class mIoU bit-identically, range exactly "
        "0.000000 -- and all five nonetheless sit 0.339357 two-class below the "
        "value stored in that cell's own run.json at training time. A "
        "reproducible offset of that size is not noise and its mechanism is not "
        "established; the leading hypothesis is that the layers DFFM builds "
        "inside forward() cannot be restored by load_state_dict because they "
        "never enter state_dict, so a fresh process starts them from "
        "construction defaults. Full statement in results/31's caveats.",

        "FDA touches the target domain in every cell; SIB in three of nine. "
        "FDA's nine 9-cell cells are all unlabeled_target. SIB's three MAMNet "
        "cells are unlabeled_target through the FDA stage of its own recipe and "
        "its OGLANet and DINOv3 cells are source_only, because the SIB loader "
        "never dereferences fda_target_root (the 2026-10-26 operator ruling, "
        "recorded in those cells' own run.json under target_access_superseded). "
        "So every pooled SIB-against-FDA number in this JSON -- M10's sign "
        "counts, M11's worst-case difference, and the whole multiplicity family "
        "-- compares a method with uniform target access against one with "
        "mixed access. M12 is the restriction that removes the mix, and it "
        "removes FDA completely rather than equalising it: FDA has zero "
        "source_only cells, so there is no access-matched SIB-against-FDA "
        "comparison on this grid at all. That is a property of the grid, not a "
        "choice made here.",

        "M12 is not an access-matched comparison of all ten methods. It is each "
        "method restricted to its own source_only cells, so the cell SETS "
        "differ down the table: Vanilla 9, IIM 9, ISW 9, MRFP+ 9, FADA 9, BSM "
        "8, SIB 6 (OGLANet and DINOv3 only), FDA 0, SegDeSiC 0, SAR 0 on the "
        "9-cell scope. Comparing two rows of M12 compares different cells "
        "unless both are 9-of-9, and SIB's six exclude MAMNet, which is the "
        "architecture with the smallest gaps on this grid. M10 and M11 inside "
        "M12 are computed on the intersection of the two methods' source_only "
        "cells and state n, which is the only version of that comparison that "
        "is not a coverage artefact.",

        "M2 and M4 are cell-level intervals over 9 to 12 cells. A bootstrap "
        "over 9 units is coarse: its percentile interval is built from at most "
        "9 distinct values per replicate and it cannot resolve the tails. The "
        "image-cluster interval results/31 uses for the same mean is carried "
        "beside every M2 row as M2_mean_gap_image_cluster_ci95; the two answer "
        "different questions (between-cell spread against within-cell noise) "
        "and the cell-level one is wider wherever the cells disagree with each "
        "other more than the images inside them do. Neither is corrected for "
        "the other.",

        "M11's statistic is a difference of two maxima. It has no unbiased "
        "bootstrap: a replicate that makes any single cell worse raises that "
        "method's max, so the replicate maxima sit above the observed max on "
        "both sides and the difference inherits a skew that depends on how "
        "close each method's second-worst cell is to its worst. The observed "
        "point, the bootstrap median and the percentile interval are all "
        "reported so the skew is visible; no bias correction is applied and no "
        "normal-theory interval is given.",

        "SAR's M5, M6 and M7 are an identity, not a measurement. A SAR row is "
        "built as its Vanilla cell's per-image value plus SAR's own within-pass "
        "change, so (SAR minus Vanilla) per image is that change exactly, to "
        "floating-point. SAR's harm and benefit counts therefore test whether "
        "SAR's adaptation step is distinguishable from zero and nothing else, "
        "and its gap metrics inherit the Vanilla cell's level. The step is "
        "near zero by construction: the selected (lr, steps) config carries "
        "pilot_verdict no_meaningful_adaptation and was not re-selected on the "
        "n450 models.",

        "BSM is an 8-of-9 row. DINOv3/chicago collapsed in training at best "
        "validation mIoU 5.51 against the codebase floor of 10.0 and its dump "
        "directory is empty; it is never imputed, so BSM's worst case, mean, "
        "spread and every count are over 8 cells and its M11 against SIB is "
        "over the 8 cells both hold.",

        "Six of the ten methods have no CrossEarth counterpart. Only Upper, "
        "Vanilla, FDA, SIB and BSM were run with CrossEarth, so on the 12-cell "
        "scope SegDeSiC, IIM, ISW, MRFP+, FADA and SAR are still 9-cell rows "
        "and their n_cells says 9. Their 12-cell row is their 9-cell row, not a "
        "larger-scope value, and the two must not be read as a like-for-like "
        "column. Every row carries n_cells and has_crossearth_counterpart.",

        "The counts are threshold-dependent. M5, M6, M8 and M10 count "
        "significance decisions at ALPHA = 0.05 under a named Holm family, and "
        "both the threshold and the family are analyst choices rather than "
        "properties of the data. Every count is published beside the per-cell "
        "point estimate, raw p and Holm-adjusted p, so any count here can be "
        "rebuilt at another threshold or under another family without rerunning "
        "the job.",

        "Three bootstrap schemes appear in this JSON and they are not "
        "interchangeable: a paired image-level bootstrap on per-image deltas "
        "within one cell (M5, M6, M8, M10), an image-cluster bootstrap that "
        "resamples a city's images once and applies the draw to every cell of "
        "that city (M2's companion, M11), and a cell-level bootstrap that "
        "resamples the cells themselves (M2, M4). All three use B = 10000 and "
        "numpy default_rng seeded at 0, so intervals are comparable in width "
        "across this file and against results/22, results/26 and results/27, "
        "but a CI from one scheme must not be quoted as if it came from "
        "another.",

        "p-values are floored at 1/B = 0.0001 by the arm F convention, so a "
        "reported p of 0.0001 reads as p <= 1/B and not as a point value. Holm "
        "adjustment of a family of floored p-values inherits that floor: a "
        "member whose adjusted p is m/B is at the floor, not measured there.",

        "This job's rows ARE results/31's rows. It imports that script and "
        "calls its resolve_all_rows(), so there is no second cell resolver, no "
        "second tree-preference rule and no second n450 substitution rule on "
        "this box. Every cell's mean, its upper bound's mean and its gap are "
        "then asserted against results/31's per_cell record to 1e-9 "
        "(r27_agreement in this JSON). The consequence to keep in view is that "
        "this profile is not an independent check of results/31 in any respect "
        "and shares every one of its limitations, which is why they are "
        "reproduced below in full.",
    ] + [f"Inherited from results/31, caveat {i + 1}: {c}"
         for i, c in enumerate(r27_caveats)]

    out = dict(
        id="32_method_comparison_profile_tgtfda",
        arm="F",
        answers=["AC3", "v1UJ-W3", "BxRx-W3", "AC1"],
        answers_detail=(
            "AC3 asks for other domain-adaptation strategies as baselines to "
            "SIB and v1UJ-W3 for a comparison with existing methods that is "
            "not selective; this job is the complete, pre-declared form of "
            "that comparison on the n450 grid, every metric for every method. "
            "BxRx-W3 asks whether the improvement over the strongest baseline "
            "is modest, which M10 and M11 quantify with intervals rather than "
            "with a point. AC1 is the overstatement note: the metrics that "
            "most constrain a claim here are M8, which counts the cells whose "
            "gap is not separable from zero, and M11's CI and resample share, "
            "which say how little a single-cell worst-case ordering is pinned "
            "down. This job fires no reading and does not settle any of them."),
        question=("On the matched-450 grid, how does every evaluated method "
                  "compare to the within-city upper bound and to Vanilla, and "
                  "how large and how certain is SIB's worst-case and per-cell "
                  "difference against each of them?"),
        status="EXPLORATORY",
        post_hoc=True,
        post_hoc_detail=("declared after results/26's reading fired R3. The "
                         "metric list, method list, scopes, ALPHA and "
                         "multiplicity family were frozen in code before any "
                         "value was computed; the decision to compute them at "
                         "all was made after the result they characterise was "
                         "known."),
        supersedes=[],
        companion_to=["results/31_internal_sib_worst_case_tgtfda.json",
                      "results/26_internal_sib_worst_case_n450.json",
                      "results/22_internal_sib_worst_case_v3.json"],
        companion_why=("results/31 is the adjudication and this is the profile "
                       "of the same rows. Neither supersedes the other: "
                       "results/31 answers a pre-committed question about SIB's "
                       "worst case and fires a reading, this JSON answers none "
                       "and fires none."),
        rows_source=dict(
            what="every row is results/31's row, resolved by importing that "
                 "job's resolve_all_rows()",
            script="analysis/31_internal_sib_worst_case_tgtfda.py",
            results_json="results/31_internal_sib_worst_case_tgtfda.json",
            agreement=r27_agreement),
        declared_path=r27["declared_path"],
        declared_metrics=METRICS,
        declared_methods=METHODS,
        declared_scopes=r27["scopes"],
        declared_access_restrictions=ACCESS_RESTRICTIONS,
        alpha=ALPHA,
        profile=profiles,
        sib_head_to_head=h2h,
        sib_vs_fda_multiplicity_family=family,
        per_cell=[{k: v for k, v in c.items()
                   if k not in ("method_per_image", "upper_per_image",
                                "vanilla_per_image", "gap_per_image",
                                "vs_vanilla_per_image")}
                  for c in cells],
        conditions=conditions,
        tests=tests,
        runs=sorted({r["run_id"] for r in rows.values()
                     if isinstance(r.get("run_id"), str)}
                    | {x for r in rows.values()
                       if isinstance(r.get("run_id"), list)
                       for x in r["run_id"]}),
        runs_excluded=[dict(run_id=c["run_id"], reason=c["blocker"])
                       for c in cells if not c["has_value"]],
        index_sha256=png_info.get("sha256"),
        index_sha256_all={k: dict(path=v.get("path"), tree=v.get("root"),
                                  sha256=v.get("sha256"), n_rows=v["n_rows"])
                          for k, v in indexes.items()},
        index_sha256_matches_results_31=(png_info.get("sha256")
                                         == r27.get("index_sha256")),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script="analysis/32_method_comparison_profile_tgtfda.py",
        env=dict(python=sys.version.split()[0],
                 conda_env="myenv "
                           "(conda env)",
                 pythonnousersite=os.environ.get("PYTHONNOUSERSITE"),
                 numpy=np.__version__, pandas=pd.__version__),
        seed=BOOT_SEED,
        seed_note=f"the only stochastic steps are the bootstraps, B = {B}, "
                  f"numpy default_rng seeded at {BOOT_SEED} in all three "
                  f"schemes",
        git_hash=None,
        git_hash_reason="the project base is not a git repository on the analysis host; "
                        "only final_paper/ is git-connected and this job does "
                        "not touch it",
        compute_note="read-only CPU analysis on the analysis host. No training, no GPU "
                     "inference, no checkpoint loaded, nothing under dumps/, "
                     "delta_runs/ or results/ modified.",
        headline=headline,
        sanity_flags=sanity,
        caveats=caveats,
        figure=None,
        caption=None,
    )

    # =========================================================================
    # results/32 AMENDMENT. Everything above is results/28's computation on
    # results/31's rows. This block adds the moved-from-results/28 record, the
    # new caveats and the stale-prose record, and replaces the headline. It
    # recomputes nothing.
    # =========================================================================
    r28 = json.load(open(R28))

    def prof(lst, scope, acc, method):
        return next((e for e in lst if e["scope"] == scope
                     and e["access_restriction"] == acc
                     and e["method"] == method), None)

    def h2h_of(lst, scope, acc, method):
        return next((r for r in lst if r["scope"] == scope
                     and r["access_restriction"] == acc
                     and r["other_method"] == method), None)

    def rank_of(lst, scope, acc, fld):
        have_ = [e for e in lst if e["scope"] == scope
                 and e["access_restriction"] == acc and e.get(fld) is not None]
        sib_e = next((e for e in have_ if e["method"] == "sib"), None)
        if sib_e is None:
            return dict(sib_rank=None, n_methods_ranked=len(have_))
        return dict(sib_rank=1 + sum(1 for e in have_ if e[fld] < sib_e[fld]),
                    n_methods_ranked=len(have_),
                    n_ties_with_sib=sum(1 for e in have_ if e[fld] == sib_e[fld]
                                        and e["method"] != "sib"),
                    order=[dict(method=e["method"], value=e[fld],
                                n_cells=e["n_cells"])
                           for e in sorted(have_, key=lambda e: e[fld])])

    moved = []
    for scope in SCOPES:
        for acc in (None, "source_only"):
            a_, b_ = prof(r28["profile"], scope, acc, "sib"), \
                prof(profiles, scope, acc, "sib")
            rk_a = {f: rank_of(r28["profile"], scope, acc, f)
                    for f in ("M1_worst_case_gap", "M2_mean_gap")}
            rk_b = {f: rank_of(profiles, scope, acc, f)
                    for f in ("M1_worst_case_gap", "M2_mean_gap")}
            m11 = []
            for meth in METHODS:
                if meth == "sib":
                    continue
                ra = (h2h_of(r28["sib_head_to_head"], scope, acc, meth) or {})
                rb = (h2h_of(h2h, scope, acc, meth) or {})
                ma, mb = ra.get("M11") or {}, rb.get("M11") or {}
                m11.append(dict(
                    other_method=meth,
                    n_cells=dict(r28=ra.get("n_cells_shared"),
                                 this=rb.get("n_cells_shared")),
                    difference_sib_minus_other=dict(
                        r28=ma.get("difference_sib_minus_other"),
                        this=mb.get("difference_sib_minus_other")),
                    ci95=dict(r28=ma.get("ci95"), this=mb.get("ci95")),
                    p_raw=dict(r28=ma.get("p"), this=mb.get("p")),
                    share_sib_worst_case_smaller=dict(
                        r28=ma.get("share_sib_worst_case_smaller"),
                        this=mb.get("share_sib_worst_case_smaller"))))
            moved.append(dict(
                scope=scope, access_restriction=acc,
                sib_n_cells=dict(r28=a_["n_cells"], this=b_["n_cells"]),
                sib_cells=dict(r28=a_["cells"], this=b_["cells"]),
                M1_sib_worst_case_gap=dict(r28=a_["M1_worst_case_gap"],
                                           this=b_["M1_worst_case_gap"]),
                M1_sib_worst_cell=dict(r28=a_["M1_worst_cell"],
                                       this=b_["M1_worst_cell"]),
                M2_sib_mean_gap=dict(r28=a_["M2_mean_gap"],
                                     this=b_["M2_mean_gap"]),
                M2_sib_mean_gap_ci95_cell_level=dict(
                    r28=a_.get("M2_mean_gap_ci95"),
                    this=b_.get("M2_mean_gap_ci95")),
                M2_sib_mean_gap_ci95_image_cluster=dict(
                    r28=a_.get("M2_mean_gap_image_cluster_ci95"),
                    this=b_.get("M2_mean_gap_image_cluster_ci95")),
                sib_rank_M1_worst_case=dict(
                    r28=rk_a["M1_worst_case_gap"]["sib_rank"],
                    this=rk_b["M1_worst_case_gap"]["sib_rank"],
                    of=dict(r28=rk_a["M1_worst_case_gap"]["n_methods_ranked"],
                            this=rk_b["M1_worst_case_gap"]["n_methods_ranked"])),
                sib_rank_M2_mean_gap=dict(
                    r28=rk_a["M2_mean_gap"]["sib_rank"],
                    this=rk_b["M2_mean_gap"]["sib_rank"],
                    of=dict(r28=rk_a["M2_mean_gap"]["n_methods_ranked"],
                            this=rk_b["M2_mean_gap"]["n_methods_ranked"])),
                rank_orders_this=rk_b,
                M11=m11))
    fam28 = r28["sib_vs_fda_multiplicity_family"]
    family_moved = dict(
        m=dict(r28=fam28["m"], this=family["m"]),
        n_significant_after_family_holm=dict(
            r28=fam28["n_significant_after_family_holm"],
            this=family["n_significant_after_family_holm"]),
        M11_member_p_holm_family=dict(
            r28=next(m_["p_holm_family"] for m_ in fam28["members"]
                     if m_["source"] == "M11"),
            this=next(m_["p_holm_family"] for m_ in family["members"]
                      if m_["source"] == "M11")))
    print("\n--- results/32 vs results/28 (SIB) ---")
    for e in moved:
        print(f"  {e['scope']:8s} {str(e['access_restriction']):12s} n "
              f"{e['sib_n_cells']['r28']}->{e['sib_n_cells']['this']}  M1 "
              f"{e['M1_sib_worst_case_gap']['r28']:.4f} "
              f"({e['M1_sib_worst_cell']['r28']}) -> "
              f"{e['M1_sib_worst_case_gap']['this']:.4f} "
              f"({e['M1_sib_worst_cell']['this']})  M2 "
              f"{e['M2_sib_mean_gap']['r28']:.4f} -> "
              f"{e['M2_sib_mean_gap']['this']:.4f}  rank M1 "
              f"{e['sib_rank_M1_worst_case']['r28']}->"
              f"{e['sib_rank_M1_worst_case']['this']}  rank M2 "
              f"{e['sib_rank_M2_mean_gap']['r28']}->"
              f"{e['sib_rank_M2_mean_gap']['this']}")
        for x in e["M11"]:
            if x["difference_sib_minus_other"]["this"] is None:
                continue
            print(f"      M11 vs {x['other_method']:9s} "
                  f"{x['difference_sib_minus_other']['r28']} -> "
                  f"{x['difference_sib_minus_other']['this']:+.4f} "
                  f"{x['ci95']['this']} p {x['p_raw']['this']}")

    STALE_MARKERS = STALE_MARKERS_32
    stale = []
    for kind, lst in (("caveats", caveats), ("sanity_flags", sanity)):
        for i, c in enumerate(lst):
            hits = [m_ for m_ in STALE_MARKERS if m_["match"] in c]
            if hits:
                stale.append(dict(list=kind, index_in_carried_list=i,
                                  excerpt=c[:160],
                                  correction=[h["correction"] for h in hits]))
    for m_ in STALE_MARKERS:
        n_hit = sum(1 for lst in (caveats, sanity) for c in lst
                    if m_["match"] in c)
        if n_hit != m_["expect"]:
            fail(f"stale marker {m_['match'][:50]!r} hit {n_hit} carried "
                 f"items, expected {m_['expect']}")
    r31 = r27
    n_inherited = len(r31["caveats"])
    new_caveats = [
        "results/32 is results/28's profile on results/31's rows. The four "
        "caveats the brief adds for the new OGLANet-SIB cells -- (a) one seed "
        "per new cell, (b) the internal donor pool is the held-out city's 450 "
        "training images, (c) OGLANet inference depends on batch size and RNG "
        "state (results/30), (d) the scorer self-check delta exceeds its "
        "hard-coded 0.01 on chicago (-0.0161) and miami (-0.0104) -- are "
        "carried in full below as 'Inherited from results/31, caveat 1' to "
        "'caveat 4'.",
        "M12 moved by construction. OGLANet-SIB is unlabeled_target here, so "
        "SIB's source_only cells are DINOv3 only on the 9-cell scope (3 cells; "
        "results/28: 6, OGLANet and DINOv3) and DINOv3 plus CrossEarth on the "
        "12-cell scope (6 cells; results/28: 9). Every M12 row for SIB, and "
        "every M10/M11 SIB head-to-head inside M12, is over that smaller set; "
        "a change in an M12 SIB number against results/28 mixes the swap of "
        "the OGLANet cells with their removal from the restriction.",
        "M11 differences here are raw, one comparison per method pair, exactly "
        "as in results/28. The Holm correction across the nine is "
        "results/33_m11_holm_tgtfda.json.",
    ]
    sib9 = next(e for e in moved if e["scope"] == "9_cell"
                and e["access_restriction"] is None)
    fda9 = next(x for x in sib9["M11"] if x["other_method"] == "fda")
    rank9 = sib9["rank_orders_this"]["M1_worst_case_gap"]["order"]
    new_headline = (
        "POST HOC / EXPLORATORY profile on results/31's rows (OGLANet-SIB from "
        "the held-out-city FDA pool retrains): on the 9-cell n450 grid SIB's "
        f"worst-case gap is {sib9['M1_sib_worst_case_gap']['this']:.2f} mIoU at "
        f"{sib9['M1_sib_worst_cell']['this']} (results/28: "
        f"{sib9['M1_sib_worst_case_gap']['r28']:.2f} at "
        f"{sib9['M1_sib_worst_cell']['r28']}), rank "
        f"{sib9['sib_rank_M1_worst_case']['this']} of "
        f"{sib9['sib_rank_M1_worst_case']['of']['this']} with "
        f"{rank9[0]['method']} smallest at {rank9[0]['value']:.2f}; its mean "
        f"gap is {sib9['M2_sib_mean_gap']['this']:.2f} mIoU (results/28: "
        f"{sib9['M2_sib_mean_gap']['r28']:.2f}), rank "
        f"{sib9['sib_rank_M2_mean_gap']['this']} of "
        f"{sib9['sib_rank_M2_mean_gap']['of']['this']}; SIB's worst case minus "
        f"FDA's is {fda9['difference_sib_minus_other']['this']:+.2f} mIoU, 95% "
        f"CI [{fda9['ci95']['this'][0]:+.2f}, {fda9['ci95']['this'][1]:+.2f}], "
        f"raw p {fda9['p_raw']['this']:.4f} (results/28: "
        f"{fda9['difference_sib_minus_other']['r28']:+.2f}, p "
        f"{fda9['p_raw']['r28']:.4f}); and {family['n_significant_after_family_holm']}"
        f" of {family['m']} SIB-against-FDA family p-values survive Holm "
        f"(results/28: {fam28['n_significant_after_family_holm']} of "
        f"{fam28['m']}).")
    out.update(
        headline=new_headline,
        headline_in_results_28_form=headline,
        supersedes=["results/28_method_comparison_profile.json"],
        supersedes_why=(
            "results/28 is superseded on one input: its rows were results/27's, "
            "whose OGLANet-SIB cells used a cross-SOURCE FDA style pool. These "
            "rows are results/31's. Metric list, methods, scopes, ALPHA and "
            "the multiplicity family are unchanged. results/28 remains "
            "readable."),
        companion_to=["results/31_internal_sib_worst_case_tgtfda.json",
                      "results/28_method_comparison_profile.json",
                      "results/27_internal_sib_worst_case_n450_sar.json"],
        moved_from_results_28=dict(sib=moved,
                                   sib_vs_fda_multiplicity_family=family_moved),
        caveats=new_caveats + caveats,
        caveats_layout=(
            f"caveats[0:{len(new_caveats)}] are new in results/32. Then "
            "results/28's own caveats, regenerated by the inherited code. "
            f"The last {n_inherited} are 'Inherited from results/31, caveat i', "
            "which are results/31's caveats in order: its eight new ones "
            "first, then results/27's. results/31's "
            "carried_items_stale_for_oglanet_sib names the inherited items "
            "whose fixed prose describes the superseded cross-source cells."),
        carried_items_stale_for_oglanet_sib=stale,
        script_lineage=dict(
            built_from="analysis/28_method_comparison_profile.py",
            how=("a copy with count-asserted edits: the imported resolver "
                 "(analysis/31), the per_cell record it is asserted against "
                 "(results/31), the M12 definition's parenthetical, the _32 "
                 "index suffix, the output path, and this amendment block.")),
    )

    print("\nheadline: " + out["headline"])

    if DRY_RUN:
        if DRY_DUMP:
            json.dump(out, open(DRY_DUMP, "w"), indent=1, default=lambda o: (
                o.tolist() if isinstance(o, np.ndarray) else
                (o.item() if isinstance(o, np.generic) else str(o))))
            print(f"DRY_RUN: would-be JSON dumped to {DRY_DUMP}")
        print("\nDRY_RUN: nothing written.")
        return

    outpath = os.path.join(BASE, "results",
                           "32_method_comparison_profile_tgtfda.json")
    if os.path.exists(outpath):
        fail(f"{outpath} already exists; this job writes exactly one new file")

    def enc(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.floating):
            return clean(float(o))
        if isinstance(o, float) and not np.isfinite(o):
            if np.isnan(o):
                return None
            fail(f"non-finite, non-missing value in the output: {o}")
        return o

    tmp = outpath + ".new"
    with open(tmp, "w") as fh:
        json.dump(clean(out), fh, indent=1, default=enc, allow_nan=False)
    os.replace(tmp, outpath)
    print(f"\nwrote {outpath}")
    print(f"  sha256 {sha256_file(outpath)}")


if __name__ == "__main__":
    main()
