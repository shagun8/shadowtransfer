"""Arm F -- results/33: results/29's Holm correction across the nine M11
SIB-against-method comparisons, re-run on results/32 (the profile on results/31's
rows, whose three OGLANet-SIB cells are the HELD-OUT-CITY FDA pool retrains
oglanet-sibc4tgt-loco-r450-{chicago,miami,phoenix}-hr-s42-n450). Source JSON,
output path, id and the moved-from-results/29 record change; the family, the
Holm rule and every assertion are results/29's. Answers BxRx-W3, AC1, AC3.

Run:  PYTHONNOUSERSITE=1 python \
          analysis/33_m11_holm_tgtfda.py [--dry-run]

=============================================================================
THE REST OF THIS DOCSTRING IS results/29's, CARRIED VERBATIM. results/28 there
reads results/32 here.
=============================================================================

Arm F -- Holm correction across the nine M11 SIB-against-method comparisons.

WHAT THIS IS. results/28_method_comparison_profile.json reports M11 -- the difference of
worst-case gaps between SIB and each other method, with an image-cluster bootstrap -- as
NINE separate tests, one per method pair, with no correction between them. Its own note
says so: "no Holm correction inside M11 itself: one comparison per method pair." This job
treats those nine as ONE FAMILY and applies Holm-Bonferroni across them, on each scope
separately, and reports raw p, adjusted p, the difference and its CI per comparison.

NO NEW BOOTSTRAP. Every statistic, CI and raw p below is COPIED from results/28. This
script resamples nothing, reads no dump and recomputes no metric: it reads nine numbers per
scope out of a results JSON, sorts them, and multiplies. Each copied value is asserted
against results/28 field by field, so a divergence is a loud failure rather than a second
opinion. The only new numbers in the output are the adjusted p-values.

WHY THIS IS A DIFFERENT FAMILY FROM THE ONE results/28 ALREADY HAS. results/28 declares a
multiplicity family of m = 46 (`sib_vs_fda_multiplicity_family`) and corrects within it.
That family is every test characterising ONE comparison, SIB against FDA, across five
metrics. This family is one metric, M11, across NINE comparisons. They overlap in exactly
one member -- the M11 SIB-against-FDA test -- and that member therefore carries two
different adjusted p-values, one per family. Both are reported here side by side. Neither
is wrong; they answer different questions, and a reader must not quote one as the other.

SCOPES. Both, as the brief asks, each corrected as its own family of nine:
  9_cell   three architectures x three held-out cities
  12_cell  the same plus CrossEarth, for the five methods CrossEarth was run with. The
           other six methods have no CrossEarth counterpart and stay 9-cell rows, so n is
           stated per comparison and the two tables are NOT the same nine cells.

Run:  PYTHONNOUSERSITE=1 python \
          analysis/29_m11_holm.py
DRY_RUN = True prints the resolved family and writes nothing.
"""
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# ---- config ----
BASE = os.environ["PROJECT_ROOT"]
SRC = f"{BASE}/results/32_method_comparison_profile_tgtfda.json"
PRED = f"{BASE}/results/29_m11_holm.json"   # results/33: the predecessor
OUT = f"{BASE}/results/33_m11_holm_tgtfda.json"
SCOPES = ["9_cell", "12_cell"]
ALPHA = 0.05
DRY_RUN = False or "--dry-run" in sys.argv
# ----------------


def fail(msg):
    raise SystemExit(f"FAIL: {msg}")


def holm(pvals):
    """Holm-Bonferroni, returning adjusted p-values in input order.

    Byte-for-byte the rule analysis/27_internal_sib_worst_case_n450_sar.py::holm uses, which
    is python/armL_results.py::holm's rule. Reimplemented here rather than imported because
    importing that module runs its row resolution, which reads every dump: this job must not
    touch dumps/ at all. The two are checked against each other in `holm_implementation`
    below on the very vector this job corrects.
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, out, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_hash():
    try:
        r = subprocess.run(["git", "-C", BASE, "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() or None
    except OSError:
        return None


def main():
    if not os.path.exists(SRC):
        fail(f"no source results JSON at {SRC}")
    src = json.load(open(SRC))
    if src.get("id") != "32_method_comparison_profile_tgtfda":
        fail(f"source id is {src.get('id')!r}, expected "
             "32_method_comparison_profile_tgtfda")

    # ---- the nine M11 comparisons per scope, copied, not recomputed ----
    tables = {}
    for scope in SCOPES:
        rows = [t for t in src["tests"]
                if t.get("metric") == "M11" and t.get("scope") == scope
                and t.get("access_restriction") is None]
        if len(rows) != 9:
            fail(f"{scope}: found {len(rows)} unrestricted M11 tests, expected 9 "
                 f"(SIB against each of the nine other methods)")
        comps = sorted(r["comparison"] for r in rows)
        if len(set(comps)) != 9:
            fail(f"{scope}: duplicate comparisons {comps}")
        if any(set(r["compares"]) & {"sib"} == set() for r in rows):
            fail(f"{scope}: a row does not involve sib")
        rows = sorted(rows, key=lambda r: r["comparison"])
        p_raw = [r["p"] for r in rows]
        for r, p in zip(rows, p_raw):
            if not (0.0 < p <= 1.0):
                fail(f"{scope}/{r['comparison']}: p = {p!r} is not a probability")
            if r.get("B") != 10000 or r.get("two_tailed") is not True:
                fail(f"{scope}/{r['comparison']}: not the declared B=10000 two-tailed test")
        p_adj = holm(p_raw)
        table = []
        for r, pa in zip(rows, p_adj):
            other = [m for m in r["compares"] if m != "sib"]
            table.append({
                "comparison": r["comparison"],
                "other_method": other[0] if len(other) == 1 else r["compares"],
                "n_cells": r["n_cells"],
                "difference_sib_worst_minus_other_worst": r["statistic"],
                "ci95": r["ci95"],
                "ci_method": "image_cluster_bootstrap_B10000_percentile",
                "p_raw": r["p"],
                "p_holm_family_of_9": pa,
                "significant_after_holm": bool(pa < ALPHA),
                "significant_before_holm": bool(r["p"] < ALPHA),
                "p_at_the_bootstrap_floor": bool(abs(r["p"] - 1.0 / 10000) < 1e-12),
                "sign": ("SIB's worst case is SMALLER (better)" if r["statistic"] < 0
                         else "SIB's worst case is LARGER (worse)"),
                "copied_from": "results/32_method_comparison_profile_tgtfda.json tests[] M11",
            })
        tables[scope] = table

    # ---- the copy is asserted, field by field, against the source ----
    checked = 0
    for scope in SCOPES:
        by_comp = {t["comparison"]: t for t in src["tests"]
                   if t.get("metric") == "M11" and t.get("scope") == scope
                   and t.get("access_restriction") is None}
        for row in tables[scope]:
            s = by_comp[row["comparison"]]
            for ours, theirs in (("difference_sib_worst_minus_other_worst", "statistic"),
                                 ("p_raw", "p"), ("n_cells", "n_cells")):
                if row[ours] != s[theirs]:
                    fail(f"{scope}/{row['comparison']}: {ours} {row[ours]!r} != source "
                         f"{theirs} {s[theirs]!r}")
                checked += 1
            if list(row["ci95"]) != list(s["ci95"]):
                fail(f"{scope}/{row['comparison']}: ci95 differs from the source")
            checked += 1

    # ---- the two scopes' raw p vectors, compared rather than assumed different ----
    p9 = {t["comparison"]: t["p_raw"] for t in tables["9_cell"]}
    p12 = {t["comparison"]: t["p_raw"] for t in tables["12_cell"]}
    same_p = sorted(c for c in p9 if p9[c] == p12[c])
    n9 = {t["comparison"]: t["n_cells"] for t in tables["9_cell"]}
    n12 = {t["comparison"]: t["n_cells"] for t in tables["12_cell"]}
    diff_n = sorted(c for c in n9 if n9[c] != n12[c])

    # ---- the one member this family shares with results/32's own family ----
    fam = src.get("sib_vs_fda_multiplicity_family") or {}
    shared = next((m for m in (fam.get("members") or []) if m.get("source") == "M11"), None)

    # ---- holm, cross-checked against the canonical implementation's own output ----
    # results/32 ran Holm over its M10 per-cell families with the same function. Re-running
    # this file's holm() on one of those recorded vectors and comparing to the recorded
    # adjusted values proves the two implementations agree, without importing the module
    # that reads dumps.
    xcheck = {"checked": False}
    for t in src["tests"]:
        if t.get("metric") == "M10" and t.get("p_holm") is not None:
            fam_rows = [x for x in src["tests"]
                        if x.get("metric") == "M10" and x.get("scope") == t.get("scope")
                        and x.get("comparison") == t.get("comparison")
                        and x.get("access_restriction") == t.get("access_restriction")]
            if len(fam_rows) < 2:
                continue
            got = holm([x["p"] for x in fam_rows])
            want = [x["p_holm"] for x in fam_rows]
            xcheck = {
                "checked": True,
                "on": f"M10 {t.get('comparison')} / {t.get('scope')}",
                "m": len(fam_rows),
                "max_abs_delta": max(abs(a - b) for a, b in zip(got, want)),
                "agrees": all(abs(a - b) < 1e-12 for a, b in zip(got, want)),
                "why": ("this file's holm() is a reimplementation, so it is checked against "
                        "a vector the canonical implementation already adjusted and "
                        "recorded in results/32, rather than trusted"),
            }
            if not xcheck["agrees"]:
                fail(f"holm() disagrees with the recorded adjusted p on {xcheck['on']}")
            break

    if DRY_RUN:
        print(f"source     {SRC}")
        print(f"source sha {sha256_file(SRC)[:16]}")
        print(f"holm cross-check {xcheck}")
        for scope in SCOPES:
            print(f"\n=== {scope}: family of {len(tables[scope])} ===")
            for t in tables[scope]:
                print(f"  {t['comparison']:20s} n={t['n_cells']:2d} "
                      f"diff={t['difference_sib_worst_minus_other_worst']:+8.4f} "
                      f"p_raw={t['p_raw']:.4f} p_holm={t['p_holm_family_of_9']:.4f} "
                      f"sig={t['significant_after_holm']}")
        print(f"\nraw p identical across scopes for {len(same_p)} of 9: {same_p}")
        print(f"n differs between scopes for: {diff_n}")
        print(f"\nasserted {checked} copied fields. DRY_RUN: nothing written.")
        return

    sig9 = [t["comparison"] for t in tables["9_cell"] if t["significant_after_holm"]]
    ns9 = [t["comparison"] for t in tables["9_cell"] if not t["significant_after_holm"]]
    sig12 = [t["comparison"] for t in tables["12_cell"] if t["significant_after_holm"]]
    ns12 = [t["comparison"] for t in tables["12_cell"] if not t["significant_after_holm"]]
    fda9 = next(t for t in tables["9_cell"] if t["comparison"] == "sib_vs_fda")

    rec = {
        "id": "33_m11_holm_tgtfda",
        "arm": "F",
        "answers": ["BxRx-W3", "AC1", "AC3"],
        "answers_detail": (
            "BxRx-W3 asks whether the improvement over the strongest baseline is modest; "
            "this job says what survives when the nine M11 comparisons are corrected as "
            "one family instead of read one at a time. AC1 is the overstatement note: the "
            "adjusted column is the one that bears on it. AC3 is the baseline-comparison "
            "row, of which this is the multiplicity-corrected form. It fires no reading and "
            "re-adjudicates none: R1-R4 were adjudicated in results/22, results/26 and "
            "results/27."),
        "question": ("Treating the nine M11 SIB-against-method worst-case-difference tests "
                     "as one family per scope, which survive Holm-Bonferroni at alpha = "
                     "0.05, and what are the difference and CI for each?"),
        "status": "EXPLORATORY",
        "post_hoc": True,
        "post_hoc_detail": (
            "Doubly post hoc. results/32 was itself commissioned after results/26's reading "
            "fired R3, i.e. after the result it characterises was known; this job was "
            "commissioned after results/32's values were in hand. The FAMILY definition -- "
            "the nine M11 comparisons of one scope, corrected together -- was fixed by the "
            "brief before any adjusted p was computed, and no comparison was added, dropped "
            "or reordered after the adjusted values appeared. But the choice to correct "
            "these nine and not some other set was made with the raw p-values already "
            "visible in results/32, and CONVENTIONS.md's rule that designing for an outcome "
            "and then observing it is the design and not the evidence applies in full."),
        "companion_to": "results/32_method_comparison_profile_tgtfda.json",
        "companion_why": (
            "every statistic, CI and raw p here is copied from that file and asserted "
            "against it; the only new numbers are the Holm-adjusted p-values. This file "
            "adds no measurement and supersedes nothing."),
        "source": {
            "results_json": "results/32_method_comparison_profile_tgtfda.json",
            "sha256": sha256_file(SRC),
            "metric": "M11",
            "metric_definition": next(m["definition"] for m in src["declared_metrics"]
                                      if m["id"] == "M11"),
            "source_note_on_correction": next(
                (t.get("note") for t in src["tests"] if t.get("metric") == "M11"
                 and t.get("note")), None),
            "fields_asserted_against_the_source": checked,
            "no_new_bootstrap": (
                "this script resamples nothing, opens no dump and recomputes no metric. It "
                "reads nine numbers per scope out of results/32, sorts them and multiplies."),
        },
        "family": {
            "definition": ("the nine M11 tests of one scope -- SIB against vanilla, fda, "
                           "segdesic, iim, isw, mrfp_plus, fada, bsm and sar -- corrected "
                           "together"),
            "m": 9,
            "correction": "Holm-Bonferroni",
            "alpha": ALPHA,
            "one_family_per_scope": True,
            "why_per_scope": ("the two scopes are two readings of the same nine "
                              "comparisons, not eighteen comparisons; pooling them would "
                              "double-count every method that has no CrossEarth counterpart "
                              "and whose two rows are literally the same test"),
            "access_restricted_rows_excluded": (
                "results/32's six source_only M11 rows per scope (metric M12) are NOT in "
                "this family. They are a different restriction over different cell sets, "
                "three methods have n = 0 there, and the brief scopes this job to the "
                "unrestricted nine."),
        },
        "holm_implementation": {
            "rule": ("sort ascending; adjusted p for the rank-r value is min(1, (m - r) * p) "
                     "made monotone non-decreasing by a running maximum"),
            "same_as": ("analysis/27_internal_sib_worst_case_n450_sar.py::holm, which is "
                        "python/armL_results.py::holm's rule"),
            "reimplemented_why": ("importing that module runs its row resolution, which "
                                  "reads every dump; this job must not touch dumps/"),
            "cross_check": xcheck,
        },
        "overlap_with_results_32_own_family": {
            "their_family": {
                "what": fam.get("comparison"),
                "m": fam.get("m"),
                "scope": fam.get("scope"),
                "member_classes": fam.get("declared_member_classes"),
            },
            "shared_members": 1,
            "shared_member": "the M11 SIB-against-FDA test",
            "the_same_test_two_adjusted_p": {
                "p_raw": fda9["p_raw"],
                "p_holm_in_this_family_of_9": fda9["p_holm_family_of_9"],
                "p_holm_in_results_32_family_of_46": (shared or {}).get("p_holm_family"),
                "p_holm_within_metric_in_results_32": (shared or {}).get(
                    "p_holm_within_metric"),
            },
            "how_to_read_it": (
                "one test, two families, two adjusted p-values, neither wrong. results/32's "
                "family asks how certain the SIB-against-FDA comparison is across five "
                "metrics; this family asks how certain the M11 comparison is across nine "
                "methods. A reader must not quote one as the other, and the two must not be "
                "combined into a third number."),
        },
        "scope_9_cell": {
            "definition": src["declared_scopes"]["scope_9_cell"],
            "n_comparisons": 9,
            "comparisons": tables["9_cell"],
            "n_significant_after_holm": len(sig9),
            "significant_after_holm": sig9,
            "not_significant_after_holm": ns9,
            "n_significant_before_holm": sum(
                1 for t in tables["9_cell"] if t["significant_before_holm"]),
            "changed_by_the_correction": [
                t["comparison"] for t in tables["9_cell"]
                if t["significant_before_holm"] != t["significant_after_holm"]],
        },
        "scope_12_cell": {
            "definition": src["declared_scopes"]["scope_12_cell"],
            "n_comparisons": 9,
            "comparisons": tables["12_cell"],
            "n_significant_after_holm": len(sig12),
            "significant_after_holm": sig12,
            "not_significant_after_holm": ns12,
            "n_significant_before_holm": sum(
                1 for t in tables["12_cell"] if t["significant_before_holm"]),
            "changed_by_the_correction": [
                t["comparison"] for t in tables["12_cell"]
                if t["significant_before_holm"] != t["significant_after_holm"]],
        },
        "the_two_scopes_compared": {
            "raw_p_identical_for": same_p,
            "n_raw_p_identical": len(same_p),
            "n_cells_differs_for": diff_n,
            "n_cells_9_cell": n9,
            "n_cells_12_cell": n12,
            "what_this_means": (
                "the nine raw p-values are identical in both scopes, so the Holm-adjusted "
                "p-values are identical too and the two tables differ only in the n column "
                "and in three confidence intervals. Six of the nine methods have no "
                "CrossEarth counterpart, so their 12-cell row IS their 9-cell row -- the "
                "same test, listed twice. For the three that do gain cells the worst case "
                "stays on a cell both scopes hold, so the point estimate does not move "
                "either. The 12-cell table is therefore not an independent replication of "
                "the 9-cell table and must not be read as one."),
        },
        "conditions": [
            {
                "label": f"M11 SIB vs {t['other_method']} / {scope}",
                "metric": "M11_worst_case_difference_two_class_mIoU_tol",
                "mean": t["difference_sib_worst_minus_other_worst"],
                "stderr": None,
                "n": t["n_cells"],
                "ci95": t["ci95"],
                "ci_method": "image_cluster_bootstrap_B10000_percentile",
                "baseline": t["other_method"],
                "n_note": "n is CELLS both methods hold in the scope, not images",
                "stderr_note": ("none: M11's interval is a bootstrap percentile interval on "
                                "a difference of maxima and no normal-theory standard error "
                                "is defined for it"),
            }
            for scope in SCOPES for t in tables[scope]
        ],
        "tests": [
            {
                "name": "Holm-Bonferroni across the nine M11 comparisons of one scope",
                "metric": "M11",
                "compares": ["sib", t["other_method"]],
                "scope": scope,
                "statistic": t["difference_sib_worst_minus_other_worst"],
                "ci95": t["ci95"],
                "p": t["p_raw"],
                "p_holm": t["p_holm_family_of_9"],
                "holm_family": f"the nine M11 comparisons of the {scope} scope, m = 9",
                "B": 10000,
                "two_tailed": True,
                "n_cells": t["n_cells"],
                "underlying_test": ("image-cluster bootstrap on the difference of "
                                    "worst-case gaps, computed in results/32 and copied "
                                    "here unchanged"),
            }
            for scope in SCOPES for t in tables[scope]
        ],
        "runs": src["runs"],
        "index_sha256": src["index_sha256"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "analysis/33_m11_holm_tgtfda.py",
        "git_hash": git_hash(),
        "git_hash_reason": ("the project base is not a git repository on the analysis host; only "
                            "final_paper/ is git-connected and this job does not touch it. "
                            "Same reason results/32 records for its own null hash"),
        "env": {"python": sys.version.split()[0], "stdlib_only": True,
                "note": "no numpy, no pandas, no dump read: nine numbers and a sort"},
        "seed": None,
        "seed_note": ("no seed: this job draws nothing. The bootstraps whose p-values it "
                      "corrects were run in results/32 under numpy default_rng seeded at 0."),
        "compute_note": "under a second, CPU only, no dump read",
        "headline": (
            f"Correcting the nine M11 SIB-against-method worst-case-difference tests as one "
            f"family, {len(sig9)} of 9 survive Holm-Bonferroni at alpha = 0.05 on the 9-cell "
            f"scope and {len(sig12)} of 9 on the 12-cell scope, with the same nine raw "
            f"p-values in both; the SIB-against-FDA comparison is not significant either "
            f"before or after correction (difference "
            f"{fda9['difference_sib_worst_minus_other_worst']:+.2f} mIoU, 95% CI "
            f"[{fda9['ci95'][0]:.2f}, {fda9['ci95'][1]:.2f}], raw p = {fda9['p_raw']:.4f}, "
            f"Holm p = {fda9['p_holm_family_of_9']:.4f})."),
        "caveats": [],
        "figure": None,
        "caption": (
            "M11, the difference between SIB's worst-case gap to the within-city upper bound "
            "and each other method's, on the matched-450 grid, two-class boundary-tolerant "
            "mIoU at K=2 in percentage points. Negative means SIB's worst case is the "
            "smaller of the two. Error bars are 95% percentile intervals from an "
            "image-cluster bootstrap (B = 10,000; a held-out city's images are resampled "
            "once and the draw applied to every cell of that city, so the two methods stay "
            "paired on the same images). p-values are Holm-Bonferroni adjusted across the "
            "nine comparisons of the scope as one family, alpha = 0.05; raw p is shown "
            "beside each. n is the number of cells both methods hold in the scope, not "
            "images. The statistic is a difference of two maxima and its bootstrap is "
            "skewed with no bias correction applied. Every cell is a single training seed. "
            "POST HOC and exploratory: both this correction and the profile it corrects were "
            "commissioned after the result they characterise was known."),
    }

    # ---- caveats: the four the brief names, carried verbatim where they exist ----
    n_floor = sum(1 for t in tables["9_cell"] if t["p_at_the_bootstrap_floor"])

    def carry(pred, label):
        for c in src["caveats"]:
            if pred(c.lower()):
                return f"{label} Carried verbatim from results/32: {c}"
        return None

    cav = [
        ("POST HOC, TWICE OVER. " + rec["post_hoc_detail"]),
        carry(lambda c: "difference of two maxima" in c,
              "M11 IS A DIFFERENCE OF TWO MAXIMA WITH NO BIAS CORRECTION."),
        carry(lambda c: c.startswith("single seed per n450 cell"), "SINGLE SEED."),
        carry(lambda c: c.startswith("dffm. oglanet inference is nondeterministic"),
              "DFFM, FIRST PART."),
        carry(lambda c: c.startswith("dffm, second part"), "DFFM, SECOND PART."),
        (f"HOLM MAKES NO NUMBER SMALLER AND SOME p SIT ON THE BOOTSTRAP FLOOR. "
         f"{n_floor} of the nine raw p-values are 0.0001, which is 1/B and the floor of a "
         f"B = 10,000 bootstrap, not a measured value: the true p could be any value below "
         f"it. Holm multiplies the smallest by m = 9, so those become 0.0009, and a floored "
         f"p that is multiplied is still a floor. A larger B would move these numbers and "
         f"no larger B was run."),
        ("THE 12-CELL TABLE IS NOT AN INDEPENDENT REPLICATION. " +
         rec["the_two_scopes_compared"]["what_this_means"]),
        ("ONE TEST, TWO FAMILIES. " +
         rec["overlap_with_results_32_own_family"]["how_to_read_it"]),
        ("WHAT THE CORRECTION DOES NOT FIX. Holm controls the family-wise error rate over "
         "these nine comparisons. It does not address the post-hoc selection of the family, "
         "the skew of a difference of maxima, the single seed per cell, the DFFM "
         "nondeterminism on every OGLANet value, or the fact that the nine comparisons are "
         "not independent -- they all share SIB's worst case, and SIB's max enters every one "
         "of the nine statistics. Holm is valid without independence, so the adjusted "
         "p-values stand, but a reader should not take nine corrected comparisons as nine "
         "pieces of evidence."),
        ("MIXED TARGET ACCESS, in the comparisons that have it. " +
         (next((c for c in src["caveats"]
                if c.startswith("FDA touches the target domain")), ""))),
        ("BSM IS AN 8-OF-9 ROW. " + (next((c for c in src["caveats"]
                                           if c.startswith("BSM is an 8-of-9 row")), ""))),
        ("NO READING FIRES. This job adjudicates no pre-commitment. REVIEWS.md's R1-R4 "
         "were adjudicated in results/22, results/26 and results/27, and whether any "
         "adjusted p here changes a sentence in the paper is a scoping decision for the "
         "operator and the HLT."),
    ]
    missing = [i for i, c in enumerate(cav) if c is None]
    if missing:
        fail(f"a brief-named caveat could not be carried from results/32: indices {missing}")
    rec["caveats"] = cav

    rec["sanity_flags"] = [
        {"flag": f"{n_floor} of nine raw p at the bootstrap floor (1/B = 0.0001)",
         "detail": [t["comparison"] for t in tables["9_cell"]
                    if t["p_at_the_bootstrap_floor"]],
         "why_flagged": ("1/B is not a measured p. CONVENTIONS.md asks for anything too "
                         "clean to be named before being asked")},
        {"flag": "the nine raw p-values are identical across the two scopes",
         "detail": f"{len(same_p)} of 9 identical; n differs for {diff_n}",
         "why_flagged": ("two tables that look like two results are one result listed "
                         "twice for six of the nine rows")},
        {"flag": "the nine comparisons are not independent",
         "detail": ("SIB's worst case enters all nine statistics, so the nine tests share "
                    "one side of every difference"),
         "why_flagged": ("Holm is valid without independence, but the family is far from "
                         "nine independent questions and should not be read as such")},
        {"flag": "no comparison changes significance under the correction"
                 if not (rec["scope_9_cell"]["changed_by_the_correction"]
                         or rec["scope_12_cell"]["changed_by_the_correction"])
                 else "the correction changes significance for some comparisons",
         "detail": {"9_cell": rec["scope_9_cell"]["changed_by_the_correction"],
                    "12_cell": rec["scope_12_cell"]["changed_by_the_correction"]},
         "why_flagged": ("a correction that changes no verdict is worth saying plainly: it "
                         "means the raw readings were not close to the threshold, not that "
                         "the correction was unnecessary")},
    ]

    # ========================================================================
    # results/33 AMENDMENT: moved-from-results/29 record, the four brief-named
    # caveats, stale-prose record, new headline. Recomputes nothing.
    # ========================================================================
    pred = json.load(open(PRED))
    moved = {}
    for scope in SCOPES:
        a_ = {t["comparison"]: t for t in pred[f"scope_{scope}"]["comparisons"]}
        b_ = {t["comparison"]: t for t in rec[f"scope_{scope}"]["comparisons"]}
        moved[scope] = dict(
            n_significant_after_holm=dict(
                r29=pred[f"scope_{scope}"]["n_significant_after_holm"],
                this=rec[f"scope_{scope}"]["n_significant_after_holm"]),
            comparisons=[dict(
                comparison=c, n_cells=dict(r29=a_[c]["n_cells"],
                                           this=b_[c]["n_cells"]),
                difference_sib_worst_minus_other_worst=dict(
                    r29=a_[c]["difference_sib_worst_minus_other_worst"],
                    this=b_[c]["difference_sib_worst_minus_other_worst"]),
                ci95=dict(r29=a_[c]["ci95"], this=b_[c]["ci95"]),
                p_raw=dict(r29=a_[c]["p_raw"], this=b_[c]["p_raw"]),
                p_holm_family_of_9=dict(r29=a_[c]["p_holm_family_of_9"],
                                        this=b_[c]["p_holm_family_of_9"]),
                significant_after_holm=dict(r29=a_[c]["significant_after_holm"],
                                            this=b_[c]["significant_after_holm"]))
                for c in sorted(b_)])
    print("\n--- results/33 vs results/29 ---")
    for scope in SCOPES:
        print(f"  {scope}: significant after Holm "
              f"{moved[scope]['n_significant_after_holm']}")
        for x in moved[scope]["comparisons"]:
            print(f"    {x['comparison']:20s} "
                  f"{x['difference_sib_worst_minus_other_worst']['r29']:+.4f} -> "
                  f"{x['difference_sib_worst_minus_other_worst']['this']:+.4f}  "
                  f"CI {x['ci95']['this']}  p {x['p_raw']['this']:.4f}  Holm "
                  f"{x['p_holm_family_of_9']['r29']:.4f} -> "
                  f"{x['p_holm_family_of_9']['this']:.4f}")
    stale_markers = [
        dict(match="five repeat passes of oglanet-sibc4-loco-r450-chicago-hr-s42-n450",
             correction="results/33: the five-pass measurement is of the "
                        "superseded cross-source chicago cell, which enters no "
                        "number here; no repeat-inference harness has been run "
                        "on a sibc4tgt cell."),
        dict(match="FDA touches the target domain in every cell; SIB in three of nine.",
             correction="results/33: SIB touches the target domain in six of "
                        "nine 9-cell cells (MAMNet and OGLANet); only "
                        "DINOv3-SIB is source_only. results/32's "
                        "carried_items_stale_for_oglanet_sib records the same "
                        "correction."),
    ]
    stale = []
    for m_ in stale_markers:
        hits = [i for i, c in enumerate(rec["caveats"]) if m_["match"] in c]
        if len(hits) != 1:
            fail(f"stale marker {m_['match'][:50]!r} hit {len(hits)} caveats, "
                 "expected 1")
        stale.append(dict(index_in_carried_list=hits[0] + 4,
                          excerpt=rec["caveats"][hits[0]][:160],
                          correction=m_["correction"]))
    src_cav = src["caveats"]
    brief4 = [c for c in src_cav if c.startswith("Inherited from results/31, caveat ")
              and c.split(":", 1)[0].rsplit(" ", 1)[1] in ("1", "2", "3", "4")]
    if len(brief4) != 4:
        fail(f"expected results/31 caveats 1-4 inside results/32, found {len(brief4)}")
    new_cav = [c.replace("Inherited from results/31", "Carried from results/31 via "
                         "results/32", 1) for c in brief4]
    old_headline = rec["headline"]
    f9 = next(x for x in moved["9_cell"]["comparisons"]
              if x["comparison"] == "sib_vs_fda")
    rec.update(
        headline=(
            f"Correcting the nine M11 SIB-against-method worst-case-difference "
            f"tests of results/32 (OGLANet-SIB from the held-out-city FDA pool "
            f"retrains) as one family, "
            f"{moved['9_cell']['n_significant_after_holm']['this']} of 9 survive "
            f"Holm-Bonferroni at alpha = 0.05 on the 9-cell scope and "
            f"{moved['12_cell']['n_significant_after_holm']['this']} of 9 on the "
            f"12-cell scope (results/29: "
            f"{moved['9_cell']['n_significant_after_holm']['r29']} and "
            f"{moved['12_cell']['n_significant_after_holm']['r29']}); SIB minus "
            f"FDA is {f9['difference_sib_worst_minus_other_worst']['this']:+.2f} "
            f"mIoU, 95% CI [{f9['ci95']['this'][0]:+.2f}, "
            f"{f9['ci95']['this'][1]:+.2f}], raw p "
            f"{f9['p_raw']['this']:.4f}, Holm p "
            f"{f9['p_holm_family_of_9']['this']:.4f} (results/29: "
            f"{f9['difference_sib_worst_minus_other_worst']['r29']:+.2f}, Holm p "
            f"{f9['p_holm_family_of_9']['r29']:.4f})."),
        headline_in_results_29_form=old_headline,
        supersedes=["results/29_m11_holm.json"],
        supersedes_why=("results/29 corrected results/28's M11 tests, whose rows "
                        "used the cross-source OGLANet-SIB cells. This file "
                        "corrects results/32's, whose rows are results/31's. "
                        "Family, rule and assertions unchanged."),
        moved_from_results_29=moved,
        caveats=new_cav + rec["caveats"],
        caveats_layout=("caveats[0:4] are the brief's (a)-(d) for the new "
                        "OGLANet-SIB cells, carried from results/31 through "
                        "results/32. caveats[4:] are results/29's caveats, "
                        "regenerated by its code on results/32."),
        carried_items_stale_for_oglanet_sib=stale,
        script_lineage=dict(
            built_from="analysis/29_m11_holm.py",
            how=("a copy with count-asserted edits: source JSON (results/32), "
                 "output path, id, and this amendment block")),
    )

    if os.path.exists(OUT):
        fail(f"{OUT} already exists; this job writes exactly one new file")
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {OUT}")
    for scope in SCOPES:
        s = rec[f"scope_{scope}"]
        print(f"\n=== {scope}: {s['n_significant_after_holm']} of 9 significant after Holm "
              f"({s['n_significant_before_holm']} before) ===")
        print(f"  {'comparison':20s} {'n':>3s} {'diff':>9s} {'ci95':>20s} "
              f"{'p_raw':>8s} {'p_holm':>8s}  sig")
        for t in s["comparisons"]:
            print(f"  {t['comparison']:20s} {t['n_cells']:3d} "
                  f"{t['difference_sib_worst_minus_other_worst']:+9.4f} "
                  f"[{t['ci95'][0]:+8.4f},{t['ci95'][1]:+8.4f}] "
                  f"{t['p_raw']:8.4f} {t['p_holm_family_of_9']:8.4f}  "
                  f"{'YES' if t['significant_after_holm'] else 'no'}")
    print(f"\nheadline: {rec['headline']}")


if __name__ == "__main__":
    main()
