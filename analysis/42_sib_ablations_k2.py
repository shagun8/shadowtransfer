#!/usr/bin/env python3
"""
results/42: SIB component ablations (Appendix J, tab_sib_ablations_full) at the
paper's metric, two-class boundary-tolerant mIoU at K=2, from Delta's 102-cell
PNG-scored tree dumps/st_sib_ablation_png (job 23560247).

INPUTS
  dumps/st_sib_ablation_png/                       102 cells, 150 images each
  delta_runs/sib_ablation_k2/chain_results_sha256_manifest.txt
                                                   sha256 of every perimage.parquet and run.json
  handoff/appendixJ_printed_values.json            printed table, sha256 pinned (gate only)
  results/41_figure1_ci.json                       context baselines only (no number formed from them)

STEPS
  1. Every perimage.parquet and run.json is verified against the manifest (102 cells).
  2. INDEX built here (CONTRACT section 5) through analysis/31's build_index and written to
     analysis/_inv/INDEX_st_sib_ablation_png_42.parquet (outside dumps/, never overwritten).
  3. Every cell is read through analysis/31's read_cell (CONTRACT section 3 columns and dtypes,
     run.json sidecar, n_images 150, K=2 recomputed from tp/fp/fn/tn_tol and asserted against
     the stored columns at 1e-9). The all-null columns must be exactly the declared PNG-only
     probability columns plus the declared deferred structure columns.
  4. Row mapping from run.json config_verbatim flags only. The reference of each (arch, city) is
     the unique cell with every component switched on and no ablation switch set; every other
     cell's toggled-flag set is matched against FLAG_RULES. The directory A-number, Delta's
     run.json `row` and the operator's MAMNet inventory are cross-checks, never inputs.
  5. Per (row, arch): per-image Delta = cell - batch reference (M1 for MAMNet/OGLANet, D1 for
     DINOv3), paired on image_id within city, pooled over 3 cities (n = 450).
     Paired image bootstrap B = 10,000, numpy default_rng(0), resampled within city: cities drawn
     in order chicago, miami, phoenix, each one (B, 150) block of rng.integers(0, 150) applied to
     the city's image_ids sorted ascending; one draw shared by every test. Percentile 95% CI.
     Two-tailed p = min(1, 2 (k + 1) / (B + 1)), k = replicates on the far side of zero; its floor
     is 2/(B+1). The published protocol's min(1, 2k/B) is stored alongside.
     Holm over the family of all (row, arch) tests. Cohen's d = mean / sd(ddof=1) of the 450 Deltas.
  6. Reference offset (C4clean - M1/D1) per (arch, city), with the flags that differ.
  7. Recovery ratio R: only from a budget-matched Vanilla/Upper pair; null with the reason if none.
  8. Gate: the directory-number (v2) mapping's pooled Deltas must reproduce the printed
     tab_sib_ablations_full to 0.01 on 28/28, else nothing is written.

Run:  PYTHONNOUSERSITE=1 \\
      python \\
      analysis/42_sib_ablations_k2.py [--dry-run]
"""

import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------

BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
B = 10000
BOOT_SEED = 0
ALPHA = 0.05
NN, NAME = 42, "sib_ablations_k2"
ARCHS = ("mamnet", "oglanet", "dinov3")
CITIES = ("chicago", "miami", "phoenix")
N_PER_CELL = 150
N_CELLS = 102
GATE_TOL = 0.01

LABEL = "st_sib_ablation_png"
DUMPS = os.path.join(BASE, "dumps", LABEL)
MANIFEST = os.path.join(BASE, "delta_runs", "sib_ablation_k2", "chain_results_sha256_manifest.txt")
PRINTED = os.path.join(BASE, "handoff", "appendixJ_printed_values.json")
PRINTED_SHA = "d88c6cc273855609710cb10f47fc2d4007b78029bac9674d65affebe3e0df9a3"
R41_JSON = os.path.join(BASE, "results", "41_figure1_ci.json")
R31_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "31_internal_sib_worst_case_tgtfda.py")
INV = os.path.join(BASE, "analysis", "_inv")
INDEX_OUT = os.path.join(INV, f"INDEX_{LABEL}_{NN}.parquet")
RESULTS = os.path.join(BASE, "results", f"{NN}_{NAME}.json")
CONTEXT_TREES = {"st": os.path.join(BASE, "dumps", "st"),
                 "st_r450_published_png": os.path.join(BASE, "dumps", "st_r450_published_png"),
                 "st_r450_published_recompute": os.path.join(BASE, "dumps", "st_r450_published_recompute")}

ARCH_LABEL = {"mamnet": "MAMNet", "oglanet": "OGLANet", "dinov3": "DINOv3"}

# Per-city or naming keys: never a component toggle. Asserted equal to the same-city
# reference where present on both (exp_tag is a name and _n_runs_in_log a parse count).
PER_CITY_KEYS = ("fda_target_root", "fold_id", "base_data_root")
IGNORED_KEYS = ("exp_tag", "_n_runs_in_log")

# Every component switch at its "on" value, per architecture. The reference is the unique
# cell of an (arch, city) whose config equals this on every key and sets no ablation switch.
ON = {
    "mamnet": dict(use_haar=True, use_vib=True, use_content_aug=True, adaptive_beta=True,
                   use_sag=True, use_fda=True, use_contrast=True),
    "oglanet": dict(use_haar=True, use_vib=True, use_content_aug=True, adaptive_beta=True,
                    use_sag=True, use_fda=True, use_contrast=True, aug_p_mix=0.3),
    "dinov3": dict(use_haar=True, use_vib=True, use_content_aug=True, adaptive_beta=True,
                   use_fda=False, aug_p_mix=0.3),
}
# Ablation switches: absent from a config == off (False / None). An absent key compared
# with an off value is NOT a toggle; absent vs a truthy value is.
SWITCHES = ("symmetric_vib", "symmetric_beta", "aug_all_subbands", "no_edge_vib",
            "vib_wrong_subband", "skip_ll_vib", "disable_content_vib", "vib_on_hl_only",
            "vib_only_band", "use_module_bypass", "use_passthrough_gate", "use_multiscale_sib")

# (flag, value in the ablated cell) -> row key. A flag's value in the reference is ON[] or off.
FLAG_RULES = {
    ("skip_ll_vib", True): "content_vib_ll",
    ("disable_content_vib", True): "content_vib_ll",
    ("use_vib", False): "all_vib",
    ("adaptive_beta", False): "uniform_beta",
    ("symmetric_vib", True): "symmetric_vib",
    ("symmetric_beta", True): "symmetric_vib",
    ("use_content_aug", False): "content_aug",
    ("aug_p_mix", 0.0): "cross_source_mix",
    ("aug_all_subbands", True): "aug_all_subbands",
    ("use_sag", False): "sag",
    ("use_fda", False): "fda",
    ("use_haar", False): "haar",
    ("vib_wrong_subband", True): "vib_hl",
    ("vib_only_band", "HL"): "vib_hl",
    ("vib_on_hl_only", True): "vib_hl",
    ("no_edge_vib", True): "edge_vib",
}
# Toggles that are allowed to ride along with a primary toggle. Each is recorded on the cell and
# reported; none is silently dropped.
SECONDARY = {
    "early_stopping_patience": "training hyperparameter, not a component",
    "fda_L": "FDA band width; moot when use_fda=False",
    "fda_target_root": "FDA style-pool root; None when use_fda=False",
    "use_contrast": "contrast stage ('ctr' in the directory names); toggled together with use_fda",
}
# Combinations of primary rows that resolve to one row, with the rule stated.
COMBINED = {
    frozenset({"all_vib", "uniform_beta"}): (
        "all_vib", "use_vib=False removes every VIB; adaptive_beta=False is recorded as a "
                   "secondary toggle on the same cell"),
    frozenset({"uniform_beta", "content_aug"}): (
        "c4clean", "the published SIB configuration (C4clean): adaptive_beta=False and "
                   "use_content_aug=False"),
}

# Row key -> (printed row id it fills under the flag-based mapping, canonical label).
ROWS = {
    "content_vib_ll": ("A1", "-Content VIB on F_LL"),
    "all_vib": ("A1", "-all VIB (use_vib=False)"),
    "uniform_beta": ("A2", "Uniform beta (no intensity adapt.)"),
    "symmetric_vib": ("A3", "Symmetric VIB (same beta on F_LL/F_LH/F_HL)"),
    "content_aug": ("A4", "-Content augmentation"),
    "cross_source_mix": ("A5", "-Cross-source content mixing"),
    "aug_all_subbands": ("A6", "Augmentation on all subbands"),
    "sag": ("A7", "-SAG"),
    "fda": ("A8", "-FDA preprocessing"),
    "haar": ("A9", "-Haar (uniform VIB on F)"),
    "vib_hl": ("A10", "VIB on F_HL instead of F_LL (inv.)"),
    "edge_vib": (None, "-Edge VIB on F_LH/F_HL (not a printed row)"),
}
ROW_ORDER = list(ROWS)
PRINTED_ROWS = [f"A{i}" for i in range(1, 11)]
# The row each printed label names (A1 is '-Content VIB on F_LL', not '-all VIB').
PRINTED_CANON = {ROWS[k][0]: k for k in ROW_ORDER if ROWS[k][0] and k != "all_vib"}

# Operator brief / Delta inventory for MAMNet: directory A-number -> row. Cross-check only.
MAMNET_INVENTORY = {"A1": "all_vib", "A5": "aug_all_subbands", "A6": "sag", "A7": "fda",
                    "A8": "haar", "A9": "edge_vib"}
# Brief: N/A by construction.
NA_BY_CONSTRUCTION = {("mamnet", "cross_source_mix"): "MAMNet's SIB has no cross-source mixing flag",
                      ("dinov3", "sag"): "DINOv3's SIB has no SAG",
                      ("dinov3", "fda"): "DINOv3's SIB has no FDA stage"}
# Delta's six (step-2 log, FINDING 3): starred on raw p, dropped by v2's own Holm.
DELTA_SIX = [("A1", "dinov3"), ("A3", "mamnet"), ("A3", "oglanet"), ("A5", "dinov3"),
             ("A8", "oglanet"), ("A9", "oglanet")]
# Delta's per-city reference offsets, C4clean - M1/D1 (step-2 log, sanity flag 3). Cross-check.
DELTA_OFFSETS = {"mamnet": (0.36, 0.88, 2.80), "oglanet": (0.64, -0.62, 0.33),
                 "dinov3": (0.64, -1.12, -0.18)}

HEADLINE_ROWS = (("sag", "-SAG"), ("haar", "-Haar"),
                 ("content_vib_ll", "-Content VIB on F_LL"),
                 ("vib_hl", "VIB on F_HL instead of F_LL"))


def fail(msg):
    raise SystemExit(f"FATAL: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_r31():
    spec = importlib.util.spec_from_file_location("r31_rows", R31_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R31 = load_r31()


# ----------------------------------------------------------------------------
# 1. manifest
# ----------------------------------------------------------------------------

def verify_manifest():
    lines = [ln.split() for ln in open(MANIFEST).read().splitlines() if ln.strip()]
    if any(len(p) != 2 for p in lines):
        fail("manifest lines are not 'sha256 relpath'")
    shas = {rel: sha for sha, rel in lines}
    if len(shas) != len(lines):
        fail("manifest lists a path twice")
    cells = sorted({rel.split("/")[0] for rel in shas})
    on_disk = sorted(n for n in os.listdir(DUMPS) if os.path.isdir(os.path.join(DUMPS, n)))
    if len(cells) != N_CELLS:
        fail(f"manifest names {len(cells)} cells, expected {N_CELLS}")
    if cells != on_disk:
        fail(f"manifest cells != directories on disk: only-manifest "
             f"{sorted(set(cells) - set(on_disk))}, only-disk {sorted(set(on_disk) - set(cells))}")
    bad = []
    for c in cells:
        for f in ("perimage.parquet", "run.json"):
            rel = f"{c}/{f}"
            if rel not in shas:
                fail(f"manifest has no line for {rel}")
            got = sha256_file(os.path.join(DUMPS, rel))
            if got != shas[rel]:
                bad.append(rel)
    if bad:
        fail(f"{len(bad)} files differ from the manifest: {bad[:5]}")
    return dict(path=os.path.relpath(MANIFEST, BASE), sha256=sha256_file(MANIFEST),
                n_lines=len(lines), n_cells=len(cells), n_verified=2 * len(cells), mismatches=0), shas


# ----------------------------------------------------------------------------
# 3. read cells
# ----------------------------------------------------------------------------

STRUCTURE_DEFERRED = ("n_gt_blobs", "n_pred_blobs", "gt_blob_area_median", "small_object_frac",
                      "fn_px_by_cat", "tp_px_by_cat", "gt_px_by_cat", "fp_px_near_gt", "fp_px_far",
                      "gt_contrast_median")


def read_all(idx, shas):
    cells = {}
    for rid in idx.run_id:
        d = os.path.join(DUMPS, rid)
        if not os.path.isfile(os.path.join(d, ".DONE")) or os.path.isfile(os.path.join(d, ".FAILED")):
            fail(f"{rid}: .DONE missing or .FAILED present")
        df, meta, err = R31.read_cell(DUMPS, rid)
        if err:
            fail(f"{rid}: {err}")
        if meta["run_id"] != rid or (df.run_id != rid).any():
            fail(f"{rid}: run_id in run.json / parquet does not match the directory")
        if (df.res != "hr").any() or meta["res"] != "hr" or (df.city != meta["holdout"]).any():
            fail(f"{rid}: res or city disagrees with run.json")
        if df.image_id.duplicated().any():
            fail(f"{rid}: duplicate image_id")
        # null columns must be exactly the declared deviation
        nulls = set(df.columns[df.isna().all()])
        declared = set(meta["deviation_png_only"]["columns_nulled"])
        if meta.get("structure_cols") != "deferred":
            fail(f"{rid}: structure_cols is {meta.get('structure_cols')!r}, expected 'deferred'")
        expected_null = declared | set(STRUCTURE_DEFERRED)
        if nulls != expected_null:
            fail(f"{rid}: all-null columns differ from the declared PNG-only + deferred set: "
                 f"unexpected {sorted(nulls - expected_null)}, declared-but-filled "
                 f"{sorted(expected_null - nulls)}")
        if meta.get("prob_source") != "none (PNG only)":
            fail(f"{rid}: prob_source {meta.get('prob_source')!r}")
        k2 = pd.Series(R31.two_class_tol_k2(df)[0], index=df.image_id.values).sort_index()
        if abs(float(k2.mean()) - float(meta["cell_mean_miou_k2_twoclass"])) > 1e-9:
            fail(f"{rid}: recomputed cell mean {k2.mean()} != run.json {meta['cell_mean_miou_k2_twoclass']}")
        npx = df.n_valid_px.to_numpy(np.float64)
        npred = (df.tp_tol + df.fp_tol).to_numpy(np.float64)
        cells[rid] = dict(run_id=rid, meta=meta, k2=k2, city=meta["holdout"], arch=meta["arch"],
                          dump_sha256=shas[f"{rid}/perimage.parquet"],
                          pred_pos_frac=float((npred / npx).mean()),
                          gt_fg_frac=float((df.n_gt_fg_valid.to_numpy(np.float64) / npx).mean()),
                          share_images_no_pred=float((npred == 0).mean()),
                          fg_iou_mean=float(R31.two_class_tol_k2(df)[1].mean()),
                          cfg=meta["config_verbatim"])
    return cells


# ----------------------------------------------------------------------------
# 4. flag mapping
# ----------------------------------------------------------------------------

def _off(v):
    return v is None or v is False


def is_reference(arch, cfg):
    for k, v in ON[arch].items():
        if k not in cfg or cfg[k] != v:
            return False
    return all(_off(cfg.get(k)) for k in SWITCHES)


def toggles(cfg, ref):
    """Keys whose value differs from the same-city reference. Absent vs an off value is not a toggle."""
    out = {}
    for k in sorted(set(cfg) | set(ref)):
        if k in IGNORED_KEYS:
            continue
        a, b = ref.get(k, "<absent>"), cfg.get(k, "<absent>")
        if a == b:
            continue
        if k in SWITCHES and (a == "<absent>" or _off(a)) and (b == "<absent>" or _off(b)):
            continue
        if k == "fda_target_root" and b is None and cfg.get("use_fda") is False:
            out[k] = [a, b]
            continue
        if k in PER_CITY_KEYS:
            fail(f"per-city key {k} differs from the same-city reference: {a!r} vs {b!r}")
        out[k] = [a, b]
    return out


def map_rows(cells):
    refs = {}
    for (arch, city) in [(a, c) for a in ARCHS for c in CITIES]:
        hits = [r for r, c in cells.items() if c["arch"] == arch and c["city"] == city
                and is_reference(arch, c["cfg"])]
        if len(hits) != 1:
            fail(f"{arch}/{city}: {len(hits)} cells carry the reference flag state, expected 1: {hits}")
        refs[(arch, city)] = hits[0]
    mapping = {}
    for rid, c in cells.items():
        ref_rid = refs[(c["arch"], c["city"])]
        legacy = re.search(r"_sib_([A-Za-z0-9]+?)_", c["meta"]["legacy_run_dir"])
        dir_tag = legacy.group(1) if legacy else None
        if rid == ref_rid:
            mapping[rid] = dict(row="reference", toggled={}, primary={}, secondary={}, dir_tag=dir_tag,
                                rule="all components on, no ablation switch set")
            continue
        T = toggles(c["cfg"], cells[ref_rid]["cfg"])
        primary, secondary, unknown = {}, {}, {}
        for k, (a, b) in T.items():
            key = (k, b)
            if key in FLAG_RULES:
                primary[k] = FLAG_RULES[key]
            elif k in SECONDARY:
                secondary[k] = [a, b]
            else:
                unknown[k] = [a, b]
        if unknown:
            fail(f"{rid}: toggled flags with no rule: {unknown}")
        if any(k in secondary for k in ("fda_L", "fda_target_root")) and c["cfg"].get("use_fda") is not False:
            fail(f"{rid}: FDA parameters changed with FDA still on: {secondary}")
        prow = frozenset(primary.values())
        if len(prow) == 1:
            row, rule = next(iter(prow)), "single component toggle"
        elif prow in COMBINED:
            row, rule = COMBINED[prow]
            if row != "c4clean":
                for k, r in list(primary.items()):
                    if r != row:
                        secondary[k] = T[k]
                        del primary[k]
        else:
            fail(f"{rid}: component toggles {dict(primary)} match no single row and no declared combination")
        mapping[rid] = dict(row=row, toggled=T, primary=primary, secondary=secondary, dir_tag=dir_tag, rule=rule)
    return refs, mapping


def _citynorm(d):
    """Per-city values replaced by a placeholder so a signature can be compared across cities."""
    return {k: (["<held-out city root>", v[1]] if k in PER_CITY_KEYS else v) for k, v in d.items()}


def mapping_table(cells, refs, mapping):
    """One entry per (arch, row); asserts the mapping is identical across the three cities."""
    table, by = [], {}
    for rid, m in mapping.items():
        by.setdefault((cells[rid]["arch"], m["row"]), []).append(rid)
    for arch in ARCHS:
        for row in ["reference", "c4clean"] + ROW_ORDER:
            rids = sorted(by.get((arch, row), []), key=lambda r: CITIES.index(cells[r]["city"]))
            if not rids:
                continue
            if [cells[r]["city"] for r in rids] != list(CITIES):
                fail(f"{arch}/{row}: cells {rids} do not cover each city exactly once")
            sig = {json.dumps(_citynorm(mapping[r]["toggled"]), sort_keys=True) for r in rids}
            tags = {mapping[r]["dir_tag"] for r in rids}
            if len(sig) != 1 or len(tags) != 1:
                fail(f"{arch}/{row}: toggles or directory tags differ across cities: {sig} {tags}")
            m0 = mapping[rids[0]]
            if row == "reference":
                label = f"batch reference ({m0['dir_tag']})"
            elif row == "c4clean":
                label = "published SIB cell (C4clean)"
            else:
                label = ROWS[row][1]
            delta_rows = sorted({cells[r]["meta"].get("row") for r in rids})
            table.append(dict(
                arch=arch, dir_a_number=m0["dir_tag"], row_key=row,
                printed_row_filled=(ROWS[row][0] if row in ROWS else None),
                row_label=label, flags_toggled_vs_reference=_citynorm(m0["toggled"]),
                component_flags=m0["primary"], secondary_flags=_citynorm(m0["secondary"]),
                mapping_rule=m0["rule"],
                directory_number_matches_printed_row=(
                    None if row not in ROWS or ROWS[row][0] is None else m0["dir_tag"] == ROWS[row][0]),
                delta_run_json_row=delta_rows,
                run_ids=rids))
    return table


def cross_checks(cells, mapping, table):
    out = dict(mamnet_inventory=[], delta_row_disagreements=[], flags_not_single_toggle=[])
    for e in table:
        if e["arch"] == "mamnet" and e["dir_a_number"] in MAMNET_INVENTORY:
            exp = MAMNET_INVENTORY[e["dir_a_number"]]
            out["mamnet_inventory"].append(dict(dir=e["dir_a_number"], inventory_row=exp,
                                                flag_row=e["row_key"], confirmed=exp == e["row_key"]))
        if e["row_key"] != "reference":
            ours = {"c4clean": "C4clean", "edge_vib": "AE"}.get(e["row_key"]) or ROWS[e["row_key"]][0]
            if e["delta_run_json_row"] != [ours]:
                out["delta_row_disagreements"].append(dict(arch=e["arch"], row_key=e["row_key"],
                                                           ours=ours, delta=e["delta_run_json_row"]))
        if e["row_key"] not in ("reference", "c4clean") and (e["secondary_flags"] or len(e["component_flags"]) > 1):
            out["flags_not_single_toggle"].append(dict(
                arch=e["arch"], dir_a_number=e["dir_a_number"], row_label=e["row_label"],
                component_flags=e["component_flags"], secondary_flags=e["secondary_flags"]))
    unconfirmed = [x for x in out["mamnet_inventory"] if not x["confirmed"]]
    if len(out["mamnet_inventory"]) != len(MAMNET_INVENTORY) or unconfirmed:
        fail(f"MAMNet inventory not confirmed by flags: {unconfirmed or out['mamnet_inventory']}")
    if out["delta_row_disagreements"]:
        fail(f"flag-based rows disagree with Delta's run.json row: {out['delta_row_disagreements']}")
    return out


# ----------------------------------------------------------------------------
# 5. statistics
# ----------------------------------------------------------------------------

def draws():
    rng = np.random.default_rng(BOOT_SEED)
    return {city: rng.integers(0, N_PER_CELL, size=(B, N_PER_CELL)) for city in CITIES}


def stack(cells, rids):
    return np.stack([cells[r]["k2"].to_numpy(np.float64) for r in rids])


def boot(X, D):
    """X: (3, 150) per-image values in CITIES order; returns (B,) replicates of the pooled mean."""
    return np.mean([X[i][D[c]].mean(axis=1) for i, c in enumerate(CITIES)], axis=0)


def summarise(X, reps):
    obs = float(X.mean(axis=1).mean())
    lo, hi = (float(v) for v in np.percentile(reps, [2.5, 97.5]))
    se = float(np.sqrt(sum(X[i].var(ddof=1) / X.shape[1] for i in range(X.shape[0]))) / X.shape[0])
    return dict(mean=obs, stderr=se, n=int(X.size), ci95=[lo, hi], bootstrap_se=float(reps.std(ddof=1)),
                ci_method=f"paired_image_bootstrap_B{B}_within_city_percentile",
                stderr_method="stratified analytic: sqrt(sum_city s_city^2 / 150) / 3")


def pvalue(obs, reps):
    k = int((reps <= 0).sum()) if obs >= 0 else int((reps >= 0).sum())
    return dict(k_far_side=k, p_two_tailed=min(1.0, 2.0 * (k + 1) / (B + 1)),
                p_published_protocol=min(1.0, 2.0 * k / B), at_floor=(k == 0),
                p_display=(f"<= 2/(B+1) = {2.0 / (B + 1):.8f}" if k == 0
                           else f"{min(1.0, 2.0 * (k + 1) / (B + 1)):.4f}"))


def stars(p):
    return 3 if p < 0.001 else 2 if p < 0.01 else 1 if p < 0.05 else 0


def city_ids(cells):
    ids = {}
    for city in CITIES:
        sets = {tuple(c["k2"].index) for c in cells.values() if c["city"] == city}
        if len(sets) != 1:
            fail(f"{city}: image_id lists differ across the {city} cells ({len(sets)} distinct)")
        ids[city] = list(next(iter(sets)))
        if len(ids[city]) != N_PER_CELL:
            fail(f"{city}: {len(ids[city])} image_ids")
    return ids


# ----------------------------------------------------------------------------
# 7. recovery ratio inputs
# ----------------------------------------------------------------------------

def recovery_ratio(cells, refs):
    """Budget-matched Vanilla/Upper pairs. Reads the context trees' indexes in memory only."""
    budgets = {}
    for arch in ARCHS:
        m = cells[refs[(arch, "chicago")]]["meta"]
        b = {(cells[refs[(arch, c)]]["meta"].get("n_train"), cells[refs[(arch, c)]]["meta"].get("n_val"),
              cells[refs[(arch, c)]]["meta"].get("n_test")) for c in CITIES}
        if len(b) != 1:
            fail(f"{arch}: reference budgets differ across cities: {b}")
        n_tr, n_va, n_te = next(iter(b))
        budgets[arch] = dict(n_train=n_tr, n_val=n_va, n_test=n_te, rung_written=m.get("rung"),
                             rung_ok=m.get("rung_ok"), n_train_evidence=m.get("n_train_evidence"))
    cand = {}
    for label, root in CONTEXT_TREES.items():
        idx, _ = R31.build_index(label, root)
        for arch in ARCHS:
            sel = idx[(idx.run_id.str.match(rf"^{arch}-(vanilla|none)-(loco|upper)-r\d+-(chicago|miami|phoenix)-hr-s\d+(-n450)?$"))]
            for _, r in sel.iterrows():
                cand.setdefault(arch, []).append(dict(
                    tree=label, run_id=r.run_id, mode=r.get("mode"), rung=r.get("rung"),
                    n_train=(None if pd.isna(r.get("n_train")) else int(r.get("n_train")))
                    if "n_train" in r else None,
                    n_val=(None if pd.isna(r.get("n_val")) else int(r.get("n_val"))) if "n_val" in r else None))
    out = {}
    for arch in ARCHS:
        bud = budgets[arch]
        cs = cand.get(arch, [])
        upper_ntrain = sorted({c["n_train"] for c in cs if c["mode"] == "upper" and c["n_train"] is not None})
        upper_rungs = sorted({int(c["rung"]) for c in cs if c["mode"] == "upper" and pd.notna(c["rung"])})
        van_match = [c["run_id"] for c in cs if c["mode"] == "loco" and bud["n_train"] is not None
                     and c["n_train"] == bud["n_train"] and c["n_val"] == bud["n_val"]]
        up_match = [c["run_id"] for c in cs if c["mode"] == "upper" and bud["n_train"] is not None
                    and c["n_train"] == bud["n_train"] and c["n_val"] == bud["n_val"]]
        r900 = sorted({(c["n_train"], c["n_val"]) for c in cs if c["mode"] == "loco" and c["rung"] == 900},
                      key=str)
        if bud["n_train"] is None:
            reason = (f"no training count is recorded for the {ARCH_LABEL[arch]} ablation batch "
                      f"(run.json n_train null, rung {bud['rung_written']} written with rung_ok "
                      f"{bud['rung_ok']}), so no Vanilla/Upper pair can be shown to match its budget")
        else:
            reason = (f"the {ARCH_LABEL[arch]} ablation batch trained on {bud['n_train']}/{bud['n_val']}/"
                      f"{bud['n_test']} (train/val/test, from its train.log). Budget-matched Vanilla LOCO "
                      f"cells found: {len(van_match)}; budget-matched within-city Upper cells found: "
                      f"{len(up_match)}. The rung-900 Vanilla LOCO cells on this box record "
                      f"(n_train, n_val) {r900}, and every within-city Upper records rung {upper_rungs} "
                      f"(n_train {upper_ntrain or 'not recorded'}); a city's train split is 450 images")
        if van_match and up_match:
            fail(f"{arch}: a budget-matched pair exists ({van_match}, {up_match}); this script does not "
                 "implement R from it and must be extended rather than run")
        out[arch] = dict(R=None, reason=reason, ablation_budget=bud,
                         cells_used="none", budget_matched_vanilla=van_match, budget_matched_upper=up_match,
                         rows=None, rows_note="null for the batch reference and every ablation row, same reason",
                         candidates_examined=len(cs))
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    print(f"DRY_RUN {DRY_RUN}  B {B}  seed {BOOT_SEED}")
    if not DRY_RUN:
        for p in (RESULTS, INDEX_OUT):
            if os.path.exists(p):
                fail(f"{os.path.relpath(p, BASE)} exists; never overwrite")

    if sha256_file(PRINTED) != PRINTED_SHA:
        fail("handoff/appendixJ_printed_values.json sha256 differs from the pinned value")
    printed = json.load(open(PRINTED))["tab_sib_ablations_full"]

    man, shas = verify_manifest()
    print(f"manifest: {man['n_verified']} files over {man['n_cells']} cells verified, 0 mismatches")

    idx, idx_info = R31.build_index(LABEL, DUMPS)
    if len(idx) != N_CELLS or idx_info["n_unindexable"]:
        fail(f"index: {len(idx)} rows, {idx_info['n_unindexable']} unindexable")
    if idx.run_id.duplicated().any():
        fail("index: duplicate run_id")
    if (idx.n_images != N_PER_CELL).any():
        fail(f"index: n_images != 150 on {idx.run_id[idx.n_images != N_PER_CELL].tolist()}")
    for _, r in idx.iterrows():
        if r.dump_sha256 != shas[f"{r.run_id}/perimage.parquet"]:
            fail(f"{r.run_id}: index dump_sha256 differs from the manifest")

    cells = read_all(idx, shas)
    ids = city_ids(cells)
    refs, mapping = map_rows(cells)
    table = mapping_table(cells, refs, mapping)
    xc = cross_checks(cells, mapping, table)

    first = next(iter(cells))
    print(f"\nschema ({first}):")
    print(pd.read_parquet(os.path.join(DUMPS, first, "perimage.parquet")).dtypes.to_string())
    print(f"\n{'arch':8s} {'dir':8s} {'row':18s} {'label':46s} toggled vs reference")
    for e in table:
        print(f"{e['arch']:8s} {str(e['dir_a_number']):8s} {e['row_key']:18s} {e['row_label']:46s} "
              f"{e['flags_toggled_vs_reference']}")
    print(f"\ncells read {len(cells)}, rows x 150 images each; image_ids identical across all cells of a city: "
          + ", ".join(f"{c} {len(ids[c])}" for c in CITIES))
    print("MAMNet inventory confirmed:", [(x["dir"], x["flag_row"]) for x in xc["mamnet_inventory"]])
    print("cells whose flags are not a single component toggle:")
    for x in xc["flags_not_single_toggle"]:
        print("   ", x)

    # ---- (arch, row) -> run_ids in city order -----------------------------
    by = {}
    for rid, m in mapping.items():
        by.setdefault((cells[rid]["arch"], m["row"]), {})[cells[rid]["city"]] = rid
    ablations = [(arch, row) for arch in ARCHS for row in ROW_ORDER if (arch, row) in by]

    # ---- coverage of the printed table --------------------------------------
    coverage, n_printed, n_cov = [], 0, 0
    for prow in PRINTED_ROWS:
        for arch in ARCHS:
            pv = printed[prow][arch]["value"]
            if pv is None:
                coverage.append(dict(printed_row=prow, arch=arch, printed="--", covered=None,
                                     note="printed as '--'"))
                continue
            keys = [r for r in ROW_ORDER if ROWS[r][0] == prow and (arch, r) in by]
            n_printed += 3
            if keys:
                n_cov += 3
            na = [NA_BY_CONSTRUCTION[(arch, r)] for r in ROW_ORDER
                  if ROWS[r][0] == prow and (arch, r) in NA_BY_CONSTRUCTION]
            coverage.append(dict(printed_row=prow, arch=arch, printed=pv, covered=bool(keys),
                                 row_key=keys[0] if keys else None, n_cities=3 if keys else 0,
                                 note=(("covered by " + ROWS[keys[0]][1] + " (not the printed label's component)")
                                       if keys and keys[0] != PRINTED_CANON[prow]
                                       else (na[0] if na else None))))
    extra_rows = [dict(arch=a, row_key=r, label=ROWS[r][1]) for (a, r) in ablations if ROWS[r][0] is None]
    print(f"\ncoverage: {n_cov} of {n_printed} printed (row, arch, city) cells; extra rows {extra_rows}")
    for (a, r), why in NA_BY_CONSTRUCTION.items():
        if (a, r) in by:
            fail(f"{a}/{r} is N/A by construction but a cell maps to it")

    # ---- gate: directory-number (v2) mapping vs printed ---------------------
    gate = []
    for prow in PRINTED_ROWS:
        for arch in ARCHS:
            pv = printed[prow][arch]["value"]
            if pv is None:
                continue
            rids = [rid for rid, m in mapping.items() if cells[rid]["arch"] == arch and m["dir_tag"] == prow]
            if len(rids) != 3:
                fail(f"gate: {arch} directory {prow} has {len(rids)} cells")
            rids = sorted(rids, key=lambda r: CITIES.index(cells[r]["city"]))
            refr = [refs[(arch, c)] for c in CITIES]
            d = float(np.mean([cells[r]["k2"].mean() - cells[f]["k2"].mean() for r, f in zip(rids, refr)]))
            gate.append(dict(printed_row=prow, arch=arch, printed=pv, recomputed=d,
                             abs_diff=abs(d - pv), passed=abs(d - pv) <= GATE_TOL,
                             printed_stars=printed[prow][arch]["stars"], run_ids=rids,
                             flag_row=mapping[rids[0]]["row"]))
    n_pass = sum(g["passed"] for g in gate)
    print(f"\nGATE (v2 directory mapping vs printed): {n_pass}/{len(gate)} within {GATE_TOL}; "
          f"max |diff| {max(g['abs_diff'] for g in gate):.4f}")
    if len(gate) != 28 or n_pass != 28:
        for g in gate:
            if not g["passed"]:
                print("   FAIL", g)
        fail("gate did not reproduce the printed table 28/28; nothing written")

    if DRY_RUN:
        print("\nselected cells and row counts:")
        for (arch, row), d in sorted(by.items()):
            print(f"   {arch:8s} {row:18s} " + "  ".join(f"{c}:{d[c]} n={len(cells[d[c]]['k2'])}" for c in CITIES))
        print("\nDRY RUN: nothing written.")
        return

    # ---- bootstrap ------------------------------------------------------------
    D = draws()
    flags = []
    per_row, per_city, conditions, tests = [], [], [], []
    levels = {}
    for arch in ARCHS:
        for row in ["reference", "c4clean"] + ROW_ORDER:
            if (arch, row) not in by:
                continue
            X = stack(cells, [by[(arch, row)][c] for c in CITIES])
            levels[(arch, row)] = summarise(X, boot(X, D))
    for arch, row in ablations:
        rids = [by[(arch, row)][c] for c in CITIES]
        refr = [by[(arch, "reference")][c] for c in CITIES]
        for r, f in zip(rids, refr):
            if list(cells[r]["k2"].index) != list(cells[f]["k2"].index):
                fail(f"{r} / {f}: image_ids not paired")
        Xa, Xr = stack(cells, rids), stack(cells, refr)
        Dl = Xa - Xr
        reps = boot(Dl, D)
        if float(np.abs(reps - (boot(Xa, D) - boot(Xr, D))).max()) > 1e-9:
            fail(f"{arch}/{row}: Delta replicates are not cell minus reference replicates")
        s = summarise(Dl, reps)
        s.update(pvalue(s["mean"], reps))
        flat = Dl.ravel()
        s["cohens_d"] = float(flat.mean() / flat.std(ddof=1))
        s["share_images_negative"] = float((flat < 0).mean())
        s["share_images_zero"] = float((flat == 0).mean())
        s["share_images_positive"] = float((flat > 0).mean())
        if not (s["ci95"][0] <= s["mean"] <= s["ci95"][1]):
            flags.append(f"{arch}/{row}: CI {s['ci95']} excludes its point estimate {s['mean']:.4f}")
        if s["share_images_negative"] in (0.0, 1.0) or s["share_images_positive"] in (0.0, 1.0):
            flags.append(f"{arch}/{row}: every per-image Delta has the same sign (perfect separation)")
        per_row.append(dict(arch=arch, row_key=row, printed_row=ROWS[row][0], row_label=ROWS[row][1],
                            dir_a_number=mapping[rids[0]]["dir_tag"], run_ids=rids, reference_run_ids=refr,
                            level=levels[(arch, row)], reference_level=levels[(arch, "reference")],
                            delta=s))
        for i, c in enumerate(CITIES):
            x = Dl[i]
            rc = x[D[c]].mean(axis=1)
            per_city.append(dict(arch=arch, row_key=row, printed_row=ROWS[row][0], row_label=ROWS[row][1],
                                 city=c, run_id=rids[i], reference_run_id=refr[i],
                                 level=float(Xa[i].mean()), reference_level=float(Xr[i].mean()),
                                 mean_delta=float(x.mean()), stderr=float(x.std(ddof=1) / np.sqrt(x.size)),
                                 n=int(x.size),
                                 ci95=[float(v) for v in np.percentile(rc, [2.5, 97.5])],
                                 ci_method=f"paired_image_bootstrap_B{B}_percentile (same draw as pooled)"))
    # Holm over the family
    praw = [r["delta"]["p_two_tailed"] for r in per_row]
    padj = R31.holm(praw)
    for r, pa in zip(per_row, padj):
        r["delta"]["p_holm"] = pa
        r["delta"]["holm_family_size"] = len(per_row)
        r["delta"]["sig_raw"] = r["delta"]["p_two_tailed"] < ALPHA
        r["delta"]["sig_holm"] = pa < ALPHA
        r["delta"]["stars_raw"] = stars(r["delta"]["p_two_tailed"])
        r["delta"]["stars_holm"] = stars(pa)
    fam = len(per_row)
    print(f"\nHolm family: {fam} (row, arch) tests")

    # printed significance vs ours, by flag row of each printed cell's run
    by_runs = {tuple(r["run_ids"]): r for r in per_row}
    sig_vs_printed = []
    for g in gate:
        r = by_runs[tuple(g["run_ids"])]
        sig_vs_printed.append(dict(
            printed_row=g["printed_row"], arch=g["arch"], printed_label=printed[g["printed_row"]]["printed_label"],
            flag_row_label=r["row_label"], printed_stars=g["printed_stars"],
            p_raw=r["delta"]["p_two_tailed"], p_holm=r["delta"]["p_holm"],
            stars_raw=r["delta"]["stars_raw"], stars_holm=r["delta"]["stars_holm"],
            starred_in_print=g["printed_stars"] > 0, sig_holm=r["delta"]["sig_holm"]))
    lose = [x for x in sig_vs_printed if x["starred_in_print"] and not x["sig_holm"]]
    gain = [x for x in sig_vs_printed if not x["starred_in_print"] and x["sig_holm"]]
    star_raw_agree = sum(x["printed_stars"] == x["stars_raw"] for x in sig_vs_printed)
    lose_keys = sorted((x["printed_row"], x["arch"]) for x in lose)
    print(f"printed-starred cells: {sum(x['starred_in_print'] for x in sig_vs_printed)}; "
          f"lose significance under Holm: {len(lose)} {lose_keys}; Delta's six: {sorted(DELTA_SIX)}")
    if lose_keys != sorted(DELTA_SIX):
        flags.append(f"cells losing significance under Holm here {lose_keys} differ from Delta's six "
                     f"{sorted(DELTA_SIX)} (different bootstrap: within-city resampling, add-one p, seed 0)")

    # ---- 6. reference offset ---------------------------------------------------
    offsets = []
    for arch in ARCHS:
        for i, c in enumerate(CITIES):
            cr, rr = by[(arch, "c4clean")][c], by[(arch, "reference")][c]
            v_c, v_r = float(cells[cr]["k2"].mean()), float(cells[rr]["k2"].mean())
            offsets.append(dict(arch=arch, city=c, c4clean_run_id=cr, reference_run_id=rr,
                                reference_tag=mapping[rr]["dir_tag"], c4clean=v_c, reference=v_r,
                                c4clean_minus_reference=v_c - v_r,
                                flags_differing=mapping[cr]["toggled"],
                                delta_reported=DELTA_OFFSETS[arch][i],
                                abs_diff_vs_delta_reported=abs(v_c - v_r - DELTA_OFFSETS[arch][i])))
    off_vals = [o["c4clean_minus_reference"] for o in offsets]
    off_bad = [o for o in offsets if o["abs_diff_vs_delta_reported"] > 0.0051]
    if off_bad:
        flags.append(f"{len(off_bad)} reference offsets differ from Delta's reported 2 dp values by > 0.005")
    offset_summary = dict(
        min=min(off_vals), max=max(off_vals),
        per_arch={a: dict(values=[o["c4clean_minus_reference"] for o in offsets if o["arch"] == a],
                          mean=float(np.mean([o["c4clean_minus_reference"] for o in offsets if o["arch"] == a])),
                          sign_varies=len({np.sign(o["c4clean_minus_reference"]) for o in offsets
                                           if o["arch"] == a}) > 1)
                  for a in ARCHS},
        flags_differing_per_arch={a: next(o["flags_differing"] for o in offsets if o["arch"] == a) for a in ARCHS})

    # ---- 7. recovery ratio -----------------------------------------------------
    rec = recovery_ratio(cells, refs)
    rec_def = ("R = (level - Vanilla) / (Upper - Vanilla), as a ratio of pooled means (each pooled over the "
               "three cities, 450 images); not the mean of per-city ratios. Not computed: see per-arch reason.")

    # ---- context baselines ---------------------------------------------------
    r41 = json.load(open(R41_JSON))
    context = {a: dict(within_city_upper_k2=r41["per_arch"][a]["upper"]["mean"],
                       within_city_upper_ci95=r41["per_arch"][a]["upper"]["ci95"],
                       vanilla_loco_k2=r41["per_arch"][a]["loco"]["mean"],
                       vanilla_loco_ci95=r41["per_arch"][a]["loco"]["ci95"],
                       cells=r41["per_arch"][a]["cells"])
               for a in ARCHS}

    # ---- too-clean / sanity -------------------------------------------------
    # collapse: >= 95% of images with no predicted shadow, or predicted-positive rate below a tenth of
    # the GT foreground rate. Low rate (< 0.01) without collapse is flagged for information only.
    collapsed = {}
    for rid, c in sorted(cells.items()):
        ref = cells[refs[(c["arch"], c["city"])]]
        desc = (f"predicted-positive fraction {c['pred_pos_frac']:.4f} (GT {c['gt_fg_frac']:.4f}, reference "
                f"{ref['pred_pos_frac']:.4f}), images with no predicted shadow {c['share_images_no_pred']:.0%}, "
                f"foreground IoU {c['fg_iou_mean']:.1f} (reference {ref['fg_iou_mean']:.1f})")
        if c["share_images_no_pred"] >= 0.95 or c["pred_pos_frac"] < 0.1 * c["gt_fg_frac"] or c["pred_pos_frac"] > 0.99:
            collapsed.setdefault((c["arch"], mapping[rid]["row"]), []).append((c["city"], c, ref))
            flags.append(f"{rid}: COLLAPSED prediction: {desc}")
        elif c["pred_pos_frac"] < 0.01:
            flags.append(f"{rid}: low predicted-positive rate, not collapsed: {desc}")
    for r in per_row:
        if r["delta"]["share_images_zero"] > 0.5:
            flags.append(f"{r['arch']}/{r['row_key']}: {r['delta']['share_images_zero']:.2%} of per-image Deltas are exactly 0")
    n_floor = sum(r["delta"]["at_floor"] for r in per_row)
    flags.append(f"{n_floor} of {fam} (row, arch) tests sit at the bootstrap floor (no replicate crossed zero); "
                 f"their p is <= 2/(B+1) = {2 / (B + 1):.8f}, resolution-limited.")
    flags.append(f"gate: {n_pass}/28 printed Deltas reproduce (max |diff| {max(g['abs_diff'] for g in gate):.4f}). "
                 "Expected by construction: the PNGs are the stored masks v2 scored. Not independent evidence.")
    flags.append(f"printed stars vs this script's raw-p stars on the same runs: {star_raw_agree}/28 agree "
                 "(different bootstrap from v2's; informational).")
    n_pat = sum("early_stopping_patience" in x["secondary_flags"] for x in xc["flags_not_single_toggle"])
    flags.append(f"{n_pat} CNN (arch, row) entries differ from their M1 reference in early_stopping_patience "
                 "(15 -> 25) as well as in the component (caveat).")
    for x in xc["flags_not_single_toggle"]:
        extra = {k: v for k, v in x["secondary_flags"].items() if k != "early_stopping_patience"}
        if extra or len(x["component_flags"]) > 1:
            flags.append(f"{x['arch']} dir {x['dir_a_number']} ({x['row_label']}): toggles beyond one component: "
                         f"components {x['component_flags']}, secondary {extra}")

    # ---- conditions / tests ------------------------------------------------------
    for arch in ARCHS:
        for row in ["reference", "c4clean"] + ROW_ORDER:
            if (arch, row) not in by:
                continue
            s = levels[(arch, row)]
            lab = (f"{ARCH_LABEL[arch]} / " +
                   ("batch reference " + mapping[by[(arch, row)]['chicago']]['dir_tag'] if row == "reference"
                    else "published SIB cell C4clean" if row == "c4clean" else ROWS[row][1]) + " / level")
            conditions.append(dict(label=lab, metric="two_class_tol_k2_mIoU", mean=s["mean"], stderr=s["stderr"],
                                   n=s["n"], ci95=s["ci95"], ci_method=s["ci_method"],
                                   baseline=None if row == "reference" else "batch reference, same architecture"))
    for r in per_row:
        s = r["delta"]
        conditions.append(dict(label=f"{ARCH_LABEL[r['arch']]} / {r['row_label']} / Delta vs batch reference",
                               metric="delta_two_class_tol_k2_mIoU (ablation minus batch reference)",
                               mean=s["mean"], stderr=s["stderr"], n=s["n"], ci95=s["ci95"],
                               ci_method=s["ci_method"], baseline="batch reference (M1/D1), same architecture"))
        tests.append(dict(name="paired image bootstrap, resampled within city", arch=r["arch"], row_key=r["row_key"],
                          compares=[r["row_label"], "batch reference"], statistic=s["mean"],
                          p=s["p_two_tailed"], p_display=s["p_display"], p_published_protocol=s["p_published_protocol"],
                          p_holm=s["p_holm"], holm_family_size=fam, cohens_d=s["cohens_d"],
                          B=B, seed=BOOT_SEED, two_tailed=True))

    # ---- headline ----------------------------------------------------------------
    def hl(arch, row):
        if (arch, row) in NA_BY_CONSTRUCTION:
            return f"{ARCH_LABEL[arch]} n/a"
        if (arch, row) not in by:
            return f"{ARCH_LABEL[arch]} no cell"
        s = next(r for r in per_row if r["arch"] == arch and r["row_key"] == row)["delta"]
        ph = (f"<= {s['p_holm']:.4f}" if s["at_floor"] else f"{s['p_holm']:.2f}" if s["p_holm"] >= 0.01
              else f"{s['p_holm']:.4f}")
        note = ("; its cells predict almost no foreground" if (arch, row) in collapsed else "")
        return (f"{ARCH_LABEL[arch]} {s['mean']:+.2f} (Holm p {ph}"
                + ("" if s["sig_holm"] else ", not significant") + note + ")")
    parts = []
    for row, lab in HEADLINE_ROWS:
        segs = [hl(a, row) for a in ARCHS]
        if row == "content_vib_ll":
            segs[0] = hl("mamnet", "all_vib").replace("MAMNet", "MAMNet (-all VIB)")
        parts.append(f"{lab}: " + ", ".join(segs))
    headline = ("Under the flag-based row mapping, the K=2 two-class tolerant mIoU change against each "
                "architecture's ablation-batch reference (n = 450 paired images, Holm over "
                f"{fam} tests) is " + "; ".join(parts) + ".")

    # ---- caveats -----------------------------------------------------------------
    lab_fix = [e for e in table if e["arch"] == "mamnet" and e["dir_a_number"] in PRINTED_CANON
               and PRINTED_CANON[e["dir_a_number"]] != e["row_key"]]
    lab_fix_txt = "; ".join(
        f"directory {e['dir_a_number']} printed as '{printed[e['dir_a_number']]['printed_label']}' is "
        f"{e['row_label']}" for e in lab_fix if e["dir_a_number"] in printed)
    om = offset_summary["per_arch"]
    caveats = [
        ("Reference: every Delta is against the ablation batch's own reference (MAMNet/OGLANet M1, DINOv3 D1), "
         "not the published SIB cell (C4clean). C4clean minus the reference, per city (chicago/miami/phoenix): "
         + "; ".join(f"{ARCH_LABEL[a]} " + " / ".join(f"{v:+.2f}" for v in om[a]["values"]) for a in ARCHS)
         + f"; range {offset_summary['min']:+.2f} to {offset_summary['max']:+.2f}. The sign varies within "
         + ", ".join(ARCH_LABEL[a] for a in ARCHS if om[a]["sign_varies"]) + ". Flags that differ: "
         + "; ".join(f"{ARCH_LABEL[a]} " + ", ".join(f"{k} {v[0]}->{v[1]}" for k, v in
                                                       offset_summary["flags_differing_per_arch"][a].items())
                     for a in ARCHS) + " (reference -> C4clean)."),
        (f"The submission's stars were raw p (no multiplicity correction). Under Holm over the {fam} (row, arch) "
         f"tests here, {len(lose)} of the {sum(x['starred_in_print'] for x in sig_vs_printed)} printed-starred "
         "cells are not significant at 0.05: "
         + ", ".join(f"printed {x['printed_row']} {ARCH_LABEL[x['arch']]} ({x['flag_row_label']}, raw p "
                     f"{x['p_raw']:.4f}, Holm p {x['p_holm']:.4f})" for x in lose)
         + (f". Cells unstarred in print but significant under Holm: "
            + ", ".join(f"{x['printed_row']} {ARCH_LABEL[x['arch']]}" for x in gain) if gain else
            ". No unstarred printed cell becomes significant under Holm") + "."),
        ("The submission's MAMNet row labels were assigned by directory number, not by configuration. "
         "Corrected by flags: " + lab_fix_txt + ". MAMNet has no cell for '-Content VIB on F_LL' alone and no "
         "'-Cross-source content mixing' cell."),
        "One training run per cell; no seed is recorded in any artifact (seed 0 in the run_id is nominal). The "
        "intervals resample test images only and exclude training variation.",
        "OGLANet ablations (and their M1/C4clean) trained on a 900/300/150 pool (train.log), not the 450-image "
        "protocol of the published tables. MAMNet and DINOv3 record no training count (rung 900 written, "
        "rung_ok false).",
        "OGLANet PNGs come from fp16 autocast inference at batch size 8: one draw from a nondeterministic "
        "inference distribution (E-9: cell-mean SD up to 0.4247 mIoU over repeated passes).",
        "MAMNet and OGLANet run with use_fda=True and fda_target_root naming the held-out city. OGLANet's trainer "
        "never passes the root to the dataset (source_only per I-1). Whether MAMNet's M1 reference and ablations "
        "are contaminated in the E-12 manner is not ruled on; E-12's scope is the 3 published MAMNet-SIB cells.",
        "Scored from the stored prediction PNG masks only; the CONTRACT section 3 probability columns are null "
        "(declared deviation) and the structure columns are deferred.",
        "Checkpoints are archived and deleted (all cells except MAMNet C4clean), so no re-inference was possible.",
        "The gate (28/28 printed Deltas reproduced within 0.01 under the directory-number mapping) confirms the "
        "metric lift from Delta's evaluation code is faithful. It is not an independent check of the metric.",
        ("Every MAMNet and OGLANet ablation cell and C4clean records early_stopping_patience 25; the M1 reference "
         "records 15. Each CNN Delta therefore also carries a patience change. DINOv3 cells all record 25."),
        ("The '-FDA preprocessing' cells of MAMNet and OGLANet also set use_contrast=False (and fda_L 0.01 vs "
         "0.005); the row removes FDA and the contrast stage together."),
        ("The MAMNet '-all VIB' cells also set adaptive_beta=False; the row is labelled by use_vib=False."),
        ("Recovery ratio R is null for every architecture: no budget-matched Vanilla/Upper pair exists on this box "
         "(see recovery_ratio.*.reason). The printed tab_sib_recovery values are not reproduced or checked here."),
        (f"p is two-tailed, min(1, 2(k+1)/(B+1)) with k replicates past zero; {n_floor} of {fam} tests sit at "
         "the floor 2/(B+1). The published protocol's min(1, 2k/B) is stored as p_published_protocol. Holm "
         "uses the former."),
        "Interval and p resample images within city (150 per city per replicate); one draw is shared by all tests, "
        "so intervals of different rows are correlated.",
        "DINOv3 '-SAG' and '-FDA preprocessing' are N/A (DINOv3's SIB has neither); MAMNet '-Cross-source "
        "content mixing' is N/A (no such flag in MAMNet's SIB).",
        "Context baselines (within-city upper, Vanilla) are the 450-image-protocol cells of results/41; they are "
        "not budget-matched to these ablations and no number here is formed from them. The strongest published "
        "adaptation method per architecture is not in this JSON.",
    ]

    for (arch, row), lst in sorted(collapsed.items()):
        lab = ROWS[row][1] if row in ROWS else row
        caveats.append(
            f"{ARCH_LABEL[arch]} '{lab}': the cells predict almost no foreground. Per city "
            + "; ".join(f"{city}: predicted-positive fraction {c['pred_pos_frac']:.4f} vs reference "
                        f"{ref['pred_pos_frac']:.4f} (GT {c['gt_fg_frac']:.4f}), images with no predicted shadow "
                        f"{c['share_images_no_pred']:.0%}, foreground IoU {c['fg_iou_mean']:.1f} vs reference "
                        f"{ref['fg_iou_mean']:.1f}" for city, c, ref in lst)
            + ". The row's Delta measures this collapsed run against the reference.")

    # ---- write ----------------------------------------------------------------------
    os.makedirs(INV, exist_ok=True)
    idx.to_parquet(INDEX_OUT + ".tmp.parquet", index=False)
    os.replace(INDEX_OUT + ".tmp.parquet", INDEX_OUT)
    index_sha = sha256_file(INDEX_OUT)
    runs = sorted(cells)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    J = dict(
        id=f"{NN}_{NAME}", arm="H2", dump_arm_label="J (Delta's label in every run.json)",
        answers=["BxRx-W3"],
        answers_detail=("No REVIEWS.md row names the SIB ablation table. This JSON is an H2 recomputation of "
                        "Appendix J's tab_sib_ablations_full at K=2 with corrected row labels and Holm; the "
                        "nearest reviewer ask is BxRx-W3 (what SIB's components contribute). Operator to confirm."),
        question=("Per architecture, how much does each SIB component change two-class tolerant mIoU at K=2 "
                  "when removed, with rows mapped by configuration flags and Holm over the table?"),
        metric=("two-class boundary-tolerant mIoU at K=2: per image, mean of foreground and background IoU from "
                "tp/fp/fn/tn_tol (analysis/31 two_class_tol_k2, asserted against the stored columns); cell = mean "
                "over 150 images; pooled = mean over 3 cities"),
        mapping_table=table, mapping_cross_checks=xc,
        coverage=dict(printed_cells_covered=n_cov, printed_cells_total=n_printed, extra_rows=extra_rows,
                      per_printed_cell=coverage, na_by_construction=[dict(arch=a, row_key=r, reason=w)
                                                                     for (a, r), w in NA_BY_CONSTRUCTION.items()]),
        per_row=per_row, per_city=per_city, reference_offset=offsets, reference_offset_summary=offset_summary,
        recovery_ratio=dict(definition=rec_def, per_arch=rec),
        significance_vs_printed=dict(cells=sig_vs_printed, lose_under_holm=lose_keys, gain_under_holm=
                                     sorted((x["printed_row"], x["arch"]) for x in gain),
                                     delta_six=sorted(DELTA_SIX), printed_star_raw_agreement=star_raw_agree),
        gate=dict(definition=("directory-number (v2) mapping: printed row Ak <- the cells whose legacy directory "
                              "carries Ak; pooled Delta = mean over 3 cities of (cell mean - reference cell mean)"),
                  tolerance=GATE_TOL, passed=n_pass, total=len(gate), cells=gate,
                  printed_source=dict(path=os.path.relpath(PRINTED, BASE), sha256=PRINTED_SHA)),
        context_baselines=dict(source=dict(path=os.path.relpath(R41_JSON, BASE), sha256=sha256_file(R41_JSON)),
                               note="450-image protocol, high resolution, K=2; not budget-matched; context only",
                               per_arch=context),
        conditions=conditions, tests=tests,
        holm=dict(family_size=fam, family="all (row, arch) Delta tests in the table, including MAMNet -Edge VIB",
                  method="Holm-Bonferroni (analysis/31 holm)", alpha=ALPHA),
        bootstrap=dict(B=B, seed=BOOT_SEED, rng="numpy.random.default_rng(0)",
                       draw=("per replicate, 150 positions with replacement within each city; cities drawn in "
                             "order chicago, miami, phoenix, one (B, 150) block each of rng.integers(0, 150), "
                             "applied to the city's image_ids sorted ascending; one draw shared by every test"),
                       statistic="mean over the 3 cities of the resampled city mean",
                       ci="percentile 2.5 / 97.5",
                       p="two-tailed min(1, 2 (k+1)/(B+1)); floor 2/(B+1)"),
        manifest=man, runs=runs, n_runs=len(runs),
        cells=[dict(run_id=r, arch=c["arch"], city=c["city"], row_key=mapping[r]["row"],
                    dir_a_number=mapping[r]["dir_tag"], dump_sha256=c["dump_sha256"],
                    cell_mean_k2=float(c["k2"].mean()), pred_pos_frac=c["pred_pos_frac"],
                    gt_fg_frac=c["gt_fg_frac"], share_images_no_pred=c["share_images_no_pred"],
                    fg_iou_mean=c["fg_iou_mean"],
                    target_access=c["meta"].get("target_access"), pred_dir=c["meta"].get("pred_dir"))
               for r, c in sorted(cells.items())],
        index_sha256=index_sha,
        index=dict(path=os.path.relpath(INDEX_OUT, BASE), n_rows=int(len(idx)),
                   built_by="analysis/31 build_index", info={k: v for k, v in idx_info.items() if k != "unindexable"}),
        generated_utc=now, script=f"analysis/{NN}_{NAME}.py", script_sha256=sha256_file(os.path.abspath(__file__)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 conda_env="myenv (conda env)", pythonnousersite="1"),
        reader=dict(script=os.path.relpath(R31_SCRIPT, BASE), sha256=sha256_file(R31_SCRIPT),
                    functions=["read_cell", "two_class_tol_k2", "build_index", "holm"]),
        seed=BOOT_SEED, git_hash=None,
        git_hash_reason=("the project base is not a git repository on the analysis host; only final_paper/ is git-connected "
                         "and this job does not touch it"),
        headline=headline, caveats=caveats, sanity_flags=flags, figure=None, caption=None)
    txt = json.dumps(J, indent=1, allow_nan=False)
    with open(RESULTS + ".tmp", "w") as fh:
        fh.write(txt + "\n")
    os.replace(RESULTS + ".tmp", RESULTS)

    print(f"\n{'arch':8s} {'row':18s} {'Delta':>8s} {'CI95':>18s} {'p':>9s} {'Holm':>9s} {'d':>7s}")
    for r in per_row:
        s = r["delta"]
        print(f"{r['arch']:8s} {r['row_key']:18s} {s['mean']:+8.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] "
              f"{s['p_two_tailed']:9.5f} {s['p_holm']:9.5f} {s['cohens_d']:+7.3f}")
    print(f"\nWROTE {os.path.relpath(RESULTS, BASE)} sha256 {sha256_file(RESULTS)[:8]}; {len(runs)} runs; "
          f"index {index_sha[:8]}; flags {len(flags)}")
    print(f"HEADLINE: {headline}")


if __name__ == "__main__":
    main()
