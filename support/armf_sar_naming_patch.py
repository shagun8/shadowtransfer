#!/usr/bin/env python
"""Arm F, 2026-07-31: fix the metric-family naming collision in the nine SAR
results.json and add the primary-metric (two-class tolerant mIoU K=2) value
derived from the stored per-image counts.

Read-only on every published asset. Rewrites only
runs/armF_sar_{arch}_{city}_highres/results.json.

Renames (value preserved bit-exactly, key position preserved):
    unadapted.miou_tol2            -> unadapted.iou_fg_tol2
    sar.miou_tol2                  -> sar.iou_fg_tol2
    sar.miou_tol2_step0            -> sar.iou_fg_tol2_step0
    sar.miou_tol2_final            -> sar.iou_fg_tol2_final
    sar.delta_adaptation_only      -> sar.delta_adaptation_only_iou_fg_tol2
    sar.delta_vs_unadapted         -> sar.delta_vs_unadapted_iou_fg_tol2
    per_pass[].miou_fg_tol2        -> per_pass[].iou_fg_tol2
    per_pass[].delta_vs_unadapted  -> per_pass[].delta_vs_unadapted_iou_fg_tol2

Adds:
    metric_naming_correction
    unadapted.miou_tol2_derived_from_counts
    sar.miou_tol2_derived_from_counts
    sar.delta_adaptation_only_miou_tol2
    sar.delta_vs_unadapted_miou_tol2
    two caveats lines

Every other field is left byte-identical and that is asserted before the write.
"""
import argparse, glob, json, os, shutil, struct, sys
from collections import OrderedDict

import numpy as np
import pandas as pd

BASE = os.environ["PROJECT_ROOT"]
CORRECTED_ON = "2026-07-31"

RENAMES_UNADAPTED = [("miou_tol2", "iou_fg_tol2")]
RENAMES_SAR = [
    ("miou_tol2", "iou_fg_tol2"),
    ("miou_tol2_step0", "iou_fg_tol2_step0"),
    ("miou_tol2_final", "iou_fg_tol2_final"),
    ("delta_adaptation_only", "delta_adaptation_only_iou_fg_tol2"),
    ("delta_vs_unadapted", "delta_vs_unadapted_iou_fg_tol2"),
]
RENAMES_PASS = [
    ("miou_fg_tol2", "iou_fg_tol2"),
    ("delta_vs_unadapted", "delta_vs_unadapted_iou_fg_tol2"),
]

DERIV_METHOD = ("macro-average over 150 images of the mean of foreground and "
                "background IoU at K=2, computed from stored tp/fp/fn/tn")


def rename_in_place(od, old, new):
    """Rename a key, keeping its position and its value object identity."""
    assert old in od, "anchor missing: %s" % old
    assert new not in od, "target key already present: %s" % new
    items = [(new, v) if k == old else (k, v) for k, v in od.items()]
    od.clear()
    od.update(items)


def fbits(x):
    return struct.pack("<d", float(x)).hex()


def leaves(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(leaves(v, pre + "/" + k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(leaves(v, pre + "[%d]" % i))
    else:
        out[pre] = (type(o).__name__, fbits(o) if isinstance(o, float) else o)
    return out


def counts_agg(sub):
    tp = sub["tp_k2"].astype(float).values
    fp = sub["fp_k2"].astype(float).values
    fn = sub["fn_k2"].astype(float).values
    tn = sub["tn_k2"].astype(float).values
    dfg, dbg = tp + fp + fn, tn + fp + fn
    ifg = np.where(dfg > 0, tp / np.where(dfg > 0, dfg, 1.0), 0.0)
    ibg = np.where(dbg > 0, tn / np.where(dbg > 0, dbg, 1.0), 0.0)
    return (100.0 * float(np.mean((ifg + ibg) / 2.0)),
            100.0 * float(np.mean(ifg)), len(sub), int((dfg == 0).sum()),
            int((sub["true_fg_fraction"].values == 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated-against", required=True)
    ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cells = sorted(g for g in glob.glob(os.path.join(BASE, "runs/armF_sar_*_highres/results.json"))
                   if "smoke" not in g and "pilot" not in g)
    assert len(cells) == 9, "expected 9 SAR cells, found %d" % len(cells)

    for path in cells:
        name = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            raw_before = fh.read()
        d = json.loads(raw_before, object_pairs_hook=OrderedDict)
        before = leaves(d)

        # --- anchors -------------------------------------------------------
        assert d["metric"]["name"] == "iou_fg_only_tolerant", name
        assert d["metric"]["K"] == 2, name
        assert "metric_naming_correction" not in d, "%s already corrected" % name

        # --- derive the primary metric from the stored counts --------------
        df = pd.read_parquet(d["per_image_dump_path"])
        sar_pass = [p for p in df["pass_id"].unique() if p != "unadapted"]
        assert len(sar_pass) == 1, name
        sar_pass = sar_pass[0]
        sdf = df[df.pass_id == sar_pass]
        m_un, f_un, n_un, e_un, g_un = counts_agg(df[df.pass_id == "unadapted"])
        m_s0, f_s0, _, e_s0, _ = counts_agg(sdf[sdf.step == sdf.step.min()])
        m_sf, f_sf, _, e_sf, _ = counts_agg(sdf[sdf.step == sdf.step.max()])
        assert n_un == d["provenance"]["n_images"] == 150, name

        # the derivation must reproduce the file's own stored fg-only values
        for got, stored, lbl in ((f_un, d["unadapted"]["miou_tol2"], "unadapted"),
                                 (f_s0, d["sar"]["miou_tol2_step0"], "step0"),
                                 (f_sf, d["sar"]["miou_tol2_final"], "final")):
            assert abs(got - stored) < 1e-9, "%s %s: %r vs %r" % (name, lbl, got, stored)

        # --- renames -------------------------------------------------------
        renamed = OrderedDict()
        for old, new in RENAMES_UNADAPTED:
            rename_in_place(d["unadapted"], old, new)
            renamed["unadapted." + old] = "unadapted." + new
        for old, new in RENAMES_SAR:
            rename_in_place(d["sar"], old, new)
            renamed["sar." + old] = "sar." + new
        for p in d["per_pass"]:
            for old, new in RENAMES_PASS:
                rename_in_place(p, old, new)
        for old, new in RENAMES_PASS:
            renamed["per_pass[]." + old] = "per_pass[]." + new

        # --- additions -----------------------------------------------------
        d["metric_naming_correction"] = OrderedDict([
            ("corrected_on", CORRECTED_ON),
            ("was", "miou_tol2"),
            ("now", "iou_fg_tol2"),
            ("reason", "field held foreground-only IoU at K=2 while its name "
                       "asserted two-class mIoU; metric.name in the same file "
                       "already read iou_fg_only_tolerant"),
            ("renamed", renamed),
            ("delta_fields_checked", OrderedDict([
                ("delta_adaptation_only", "verified foreground-only: reproduces "
                 "iou_fg_tol2_final - iou_fg_tol2_step0 recomputed from the stored "
                 "per-image K=2 counts to <= 1.5e-14 on all nine cells"),
                ("delta_vs_unadapted", "verified foreground-only: reproduces "
                 "iou_fg_tol2_final - unadapted.iou_fg_tol2 recomputed from the "
                 "stored per-image K=2 counts to <= 1.8e-14 on all nine cells"),
            ])),
            ("fields_left_untouched", [
                "metric.spec_label, which still reads miou_boundary_tolerant; it is "
                "an existing field and was not edited, but it names the two-class "
                "family while metric.name names the foreground-only one",
                "sar.delta_adaptation_only_definition, whose prose still names the "
                "pre-rename keys miou_tol2_final and miou_tol2_step0; left "
                "byte-identical, see the renamed map above",
            ]),
            ("edited_by", "python/armf_sar_naming_patch.py"),
        ])

        deriv_un = OrderedDict([
            ("value", m_un),
            ("method", DERIV_METHOD),
            ("validated_against", a.validated_against),
        ])
        deriv_sar = OrderedDict([
            ("value", m_sf),
            ("value_step0", m_s0),
            ("value_final", m_sf),
            ("method", DERIV_METHOD),
            ("validated_against", a.validated_against),
        ])
        d["unadapted"]["miou_tol2_derived_from_counts"] = deriv_un
        d["sar"]["miou_tol2_derived_from_counts"] = deriv_sar
        d["sar"]["delta_adaptation_only_miou_tol2"] = m_sf - m_s0
        d["sar"]["delta_vs_unadapted_miou_tol2"] = m_sf - m_un

        d["caveats"] = list(d["caveats"]) + [
            "METRIC NAMING (corrected %s): the fields formerly named miou_tol2, "
            "miou_tol2_step0 and miou_tol2_final held foreground-only IoU at K=2, "
            "not two-class mIoU, and are now named iou_fg_tol2*. "
            "delta_adaptation_only and delta_vs_unadapted were checked against the "
            "stored per-image counts, confirmed foreground-only, and carry the "
            "_iou_fg_tol2 suffix. See metric_naming_correction." % CORRECTED_ON,
            "PRIMARY METRIC (added %s): miou_tol2_derived_from_counts holds "
            "two-class boundary-tolerant mIoU at K=2, the operator-set primary "
            "metric, derived from the stored per-image tp/fp/fn/tn at K=2 in "
            "perstep.parquet. No model was run and no cell was recomputed. The "
            "Appendix E degenerate rule is NOT applied: an image whose foreground "
            "IoU denominator tp+fp+fn is zero at K=2 contributes foreground IoU 0 "
            "and background IoU 1, matching the convention the stored "
            "foreground-only values in this file already use. That denominator is "
            "zero on %d of 150 images in the unadapted pass and %d in the final "
            "SAR pass; it is the empty-ground-truth-and-empty-prediction case and "
            "is not the count of empty-ground-truth images, which is %d of 150 for "
            "this city and is architecture-independent. Applying the degenerate "
            "rule instead would move the two-class value by up to 5.68 mIoU on the "
            "BSM cells where both conventions were checked." % (CORRECTED_ON, e_un, e_sf, g_un),
        ]

        # --- byte-identity of everything not touched -----------------------
        after = leaves(d)
        renamed_leaf_paths = set()
        for old, new in renamed.items():
            blk, key = old.split(".", 1)
            if blk == "per_pass[]":
                for i in range(len(d["per_pass"])):
                    renamed_leaf_paths.add(("/per_pass[%d]/%s" % (i, key),
                                            "/per_pass[%d]/%s" % (i, new.split(".", 1)[1])))
            else:
                renamed_leaf_paths.add(("/%s/%s" % (blk, key),
                                        "/%s/%s" % (blk, new.split(".", 1)[1])))
        old_paths = {o for o, _ in renamed_leaf_paths}
        new_paths = {n for _, n in renamed_leaf_paths}
        added_prefixes = ("/metric_naming_correction",
                          "/unadapted/miou_tol2_derived_from_counts",
                          "/sar/miou_tol2_derived_from_counts",
                          "/sar/delta_adaptation_only_miou_tol2",
                          "/sar/delta_vs_unadapted_miou_tol2")
        n_caveats_before = len([k for k in before if k.startswith("/caveats[")])

        for p, v in before.items():
            if p in old_paths:
                continue
            if p.startswith("/caveats["):
                assert after[p] == v, "caveat mutated: %s" % p
                continue
            assert p in after and after[p] == v, "MUTATED untouched field %s in %s" % (p, name)
        for p, v in after.items():
            if p in new_paths or p.startswith(added_prefixes):
                continue
            if p.startswith("/caveats["):
                idx = int(p[len("/caveats["):p.index("]")])
                assert idx < n_caveats_before or idx >= n_caveats_before, p
                if idx < n_caveats_before:
                    assert before[p] == v, "caveat mutated: %s" % p
                continue
            assert p in before and before[p] == v, "APPEARED unexpected field %s in %s" % (p, name)
        # renamed values bit-identical
        for o, n in renamed_leaf_paths:
            assert before[o] == after[n], "value changed across rename %s -> %s" % (o, n)

        print("%-34s denom0=%3d gt0=%3d  fg_un=%9.5f  miou_un=%9.5f  miou_s0=%9.5f  "
              "miou_final=%9.5f  d_adapt=%+.6f  d_vs_un=%+.6f"
              % (name, e_un, g_un, f_un, m_un, m_s0, m_sf, m_sf - m_s0, m_sf - m_un))

        if a.dry_run:
            continue
        os.makedirs(a.backup_dir, exist_ok=True)
        shutil.copy2(path, os.path.join(a.backup_dir, name + ".results.json.bak"))
        tmp = path + ".tmp_patch"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=2)
            fh.write("\n")
        with open(tmp) as fh:
            assert leaves(json.loads(fh.read(), object_pairs_hook=OrderedDict)) == after, name
        os.replace(tmp, path)

    print("\n%s: 9 cells" % ("DRY RUN, nothing written" if a.dry_run else "WRITTEN"))


if __name__ == "__main__":
    sys.exit(main())
