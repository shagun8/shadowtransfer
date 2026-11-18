#!/usr/bin/env python3
"""STEP 2 -- score the SIB component-ablation PNG masks at K=2 (Table 3 / App J).

Operator decisions, 2026-11-05:
  1. Score the existing row-cells; do NOT re-infer.  MAMNet x A5 is N/A by
     construction (MAMNet's SIB has no cross-source mixing flag).  MAMNet's
     on-disk A9_no_edge_vib is scored as an extra row,
     "-Edge VIB on F_LH/F_HL".  Rows come from config flags, never from the
     directory number.
  2. PNG-only path approved as a DECLARED DEVIATION: the CONTRACT section 3
     probability columns are null, prob_source = "none (PNG only)".
  3. The K=5 gate is replaced by a provenance gate.  Levels vs v2's main
     table: armJ_sib_ablation_provenance_gate.py.  DELTAS and stars vs the
     printed Appendix J values: armJ_sib_ablation_delta_gate.py.  Neither
     lives here.

CORRECTION 2026-11-05, after the operator's decision was issued.  The decision
said "OGLANet x A4 x Chicago is absent ... pool over Miami and Phoenix only,
n_cities = 2".  THAT PREMISE WAS MY STEP-1 SCAN BUG, not a fact about the
runs.  <archive>/data/{arch}/outputs entries are SYMLINKS into /scratch
(CLAUDE.md 2026-09-09) and the promote step did not create one for every
directory: 11 oglanet_sib_* directories live on /scratch with no /projects
symlink.  Two of them are this arm's: A4_no_aug/chicago (150 PNGs, stored
tolerant_2px.mIOU 70.27993129076216 -- byte-equal to the value v2's own
report records for that cell) and C4clean/chicago (the "reference hole").
The cell was never missing, so scoring it is not imputation.  The row is
scored over all THREE cities, n = 450; the 2-city pooling is ALSO reported
per the decision as issued, so the operator can see both.

Delta against the reference, the paired bootstrap (B = 10,000, seed 0), Holm
and the recovery ratios are NOT computed here.  CLAUDE.md keeps every pooled
mean, CI, p-value and ranking on the SSH box; this box writes the per-image
parquets that results/42_sib_ablations_k2.json is built from on the analysis host.  The
only pooled numbers written here are each cell's own mean, which is a
reproduction/self-check quantity and is what the provenance gate compares.

Reads nothing but stored PNGs, GT masks and config artifacts.  Loads no
checkpoint, runs no inference, modifies no existing dump.  Writes only under
--out_root plus the manifest.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = (os.environ.get("BASE_PATH") or os.environ["PROJECT_ROOT"])
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

CITIES = ("chicago", "miami", "phoenix")
ARCHES = ("mamnet", "oglanet", "dinov3")
RUNG = 900
SEED_NOMINAL = 0

ROW_LABEL = {
    "A1": "-Content VIB on F_LL", "A2": "uniform beta",
    "A3": "symmetric VIB", "A4": "-content augmentation",
    "A5": "-cross-source content mixing", "A6": "augmentation on all subbands",
    "A7": "-SAG (CNN only)", "A8": "-FDA preprocessing (CNN only)",
    "A9": "-Haar (uniform VIB on F)", "A10": "VIB on F_HL instead of F_LL",
    "AE": "-Edge VIB on F_LH/F_HL",
}
# Operator decision 1: record MAMNet A1 with its label mismatch stated.
ROW_LABEL_OVERRIDE = {("mamnet", "A1"): "-all VIB (use_vib=False)"}

METHOD_TOKEN = {"A1": "siba1", "A2": "siba2", "A3": "siba3", "A4": "siba4",
                "A5": "siba5", "A6": "siba6", "A7": "siba7", "A8": "siba8",
                "A9": "siba9", "A10": "siba10", "AE": "sibaedge",
                "M1": "sibm1", "D1": "sibd1", "C4clean": "sibc4clean"}

REF_TAG = {"mamnet": "M1", "oglanet": "M1", "dinov3": "D1"}

# target_access per CONTRACT section 4, from the audited verdicts.
TARGET_ACCESS = {
    "mamnet": ("unlabeled_target",
               "use_fda=True with fda_target_root = the HELD-OUT city's "
               "data/Final_data_test/{city}/highres root. This is the E-12 "
               "pattern (PAPER_STATE.md:149). E-12's stated scope is the 3 "
               "published MAMNet-SIB HR cells; whether this batch's M1 "
               "reference is contaminated is NOT ruled on. Strictest "
               "applicable value recorded, per CONTRACT section 4."),
    "oglanet": ("source_only",
                "use_fda=True and fda_target_root names the held-out city, "
                "but train_oglanet_sib.py parses it at :150 and does NOT pass "
                "it to the dataset at :1520/:1532, so it is never "
                "dereferenced. Matches PAPER_STATE's I-1 verdict for "
                "OGLANet-SIB: source_only, cross-source FDA, 0/111 logs."),
    "dinov3": ("source_only", "use_fda=False; no FDA stage at all."),
}

CAVEATS = [
    "No seed is recorded on any of these cells, in any artifact. seed=0 in the "
    "run_id is NOMINAL. (E-7: no args.json records a seed.)",
    "OGLANet's own train.log states Train: 900 samples Val: 300 Test: 150 on "
    "every cell -- a 900-image source pool, NOT the 450-image published "
    "protocol of section 3.3.",
    "MAMNet and DINOv3 record no training count in any artifact. rung=900 is "
    "carried from OGLANet's log plus the shared split construction over the "
    "same data root; CLAUDE.md's 2026-07-28 ruling makes directory counts "
    "non-authoritative, so rung_ok=false for those two.",
    "OGLANet PNGs were written by test_and_save_predictions at batch_size=8 "
    "under fp16 autocast -- the sib_trainer path of the E-9 probe. They are "
    "ONE DRAW from a nondeterministic inference distribution.",
    "Checkpoints for every cell but MAMNet C4clean are Fortress-archived and "
    "deleted, so re-inference was not possible and none was attempted.",
    "CONTRACT section 3 probability columns are null: these cells have no "
    "stored probability maps. prob_source = 'none (PNG only)'.",
    "CONTRACT section 2 probs.npz is NOT written, for the same reason.",
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


SCRATCH_BASE = os.environ.get("SCRATCH_BASE",
                              os.path.join(os.environ["ST_SCRATCH_ROOT"], "ShadeMaps"))


def arch_dirs(arch):
    """BOTH output roots, /projects first.

    <archive>/data/{arch}/outputs entries are SYMLINKS into /scratch
    (CLAUDE.md 2026-09-09) and the promote step did not create one for every
    directory -- 11 oglanet_sib_* dirs are /scratch-only, two of them this
    arm's.  Scanning /projects alone silently drops them.
    """
    out = {}
    for root in (os.path.join(BASE, "data", arch, "outputs"),
                 os.path.join(SCRATCH_BASE, "data", arch, "outputs")):
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            if d in out:
                continue
            if (d.startswith(arch + "_sib_") and "loco_holdout" in d
                    and "highres" in d):
                out[d] = os.path.join(root, d)
    return {d: out[d] for d in sorted(out)}


def city_of(d):
    return next((c for c in CITIES if c in d), None)


def tag_of(arch, d):
    return d.split("_sib_", 1)[1].split("_")[0] if "_sib_" in d else None


def build_cells(inventory):
    """Every cell to score: 80 flag-mapped rows + 3 AE + M1/D1 + C4clean."""
    inv = json.load(open(inventory))
    byname = {os.path.basename(c["dir_path"]): c for c in inv["cells"]}
    cells = []

    def mk(arch, city, row, d, cinv, kind):
        lbl = ROW_LABEL_OVERRIDE.get((arch, row), ROW_LABEL.get(row))
        rid = "%s-%s-loco-r%d-%s-hr-s%d" % (
            arch, METHOD_TOKEN[row], RUNG, city, SEED_NOMINAL)
        f = cinv["config_flags"]
        return {
            "run_id": rid, "arch": arch, "city": city, "res": "hr",
            "row": row, "row_kind": kind,
            "row_label_printed": ROW_LABEL.get(row),
            "row_label_actual": lbl,
            "row_label_mismatch": (lbl != ROW_LABEL.get(row)),
            "method_token": METHOD_TOKEN[row],
            "legacy_run_dir": os.path.basename(d),
            "dir_path": d,
            "pred_dir": os.path.join(d, "predictions"),
            "mask_dir": os.path.join(BASE, "data/Final_data_test", city,
                                     "highres/test/masks"),
            "config_source": cinv["config_source"],
            "config_kind": cinv["config_kind"],
            "config_flags": f,
            "row_evidence": cinv["row_evidence"].get(row),
            "n_train": (cinv["split_counts"] or {}).get("n_train"),
            "n_val": (cinv["split_counts"] or {}).get("n_val"),
            "n_test_declared": (cinv["split_counts"] or {}).get("n_test"),
            "n_train_evidence": (cinv["split_counts"] or {}).get("evidence"),
            "checkpoint_state": cinv["checkpoint"]["state"],
            "stored_tol2px_miou_twoclass":
                (cinv["test_results"] or {}).get("tolerant_2px_mIOU"),
            "stored_tol2px_fg_iou":
                (cinv["test_results"] or {}).get("tolerant_2px_Shadow_IOU"),
            "stored_tolerant_keys":
                (cinv["test_results"] or {}).get("tolerant_keys"),
            "batch_size": f.get("batch_size"),
        }

    for arch in ARCHES:
        ds = arch_dirs(arch)
        for d, full in ds.items():
            cinv = byname.get(d)
            if cinv is None:
                continue
            city, tag = city_of(d), tag_of(arch, d)
            if city is None:
                continue
            rows = cinv["rows_satisfied_by_flags"]
            # the orphan: MAMNet's edge-VIB run, scored as the extra row AE
            if arch == "mamnet" and tag == "A9" and not rows:
                cells.append(mk(arch, city, "AE", full, cinv, "extra_row"))
                continue
            if tag in ("M1", "D1") and tag == REF_TAG[arch]:
                cells.append(mk(arch, city, tag, full, cinv, "reference"))
                continue
            if tag == "C4clean":
                cells.append(mk(arch, city, "C4clean", full, cinv,
                                "published_sib_cell"))
                continue
            if not (tag and tag.startswith("A") and tag[1:].isdigit()):
                continue
            # Flag-based row resolution. MAMNet A1 satisfies A2 vacuously
            # (use_vib=False implies adaptive_beta=False); the dedicated
            # A2_uniform_beta run is row A2, so drop the vacuous hit.
            picked = [r for r in rows
                      if not (arch == "mamnet" and tag == "A1" and r == "A2")]
            for r in picked:
                cells.append(mk(arch, city, r, full, cinv, "ablation"))
    return cells


def ref_run_id(arch, city):
    return "%s-%s-loco-r%d-%s-hr-s%d" % (
        arch, METHOD_TOKEN[REF_TAG[arch]], RUNG, city, SEED_NOMINAL)


def c4clean_run_id(arch, city):
    return "%s-%s-loco-r%d-%s-hr-s%d" % (
        arch, METHOD_TOKEN["C4clean"], RUNG, city, SEED_NOMINAL)


def score_cell(c, out_root, smoke=0):
    """One cell.  Writes the dump; returns a summary dict."""
    import cv2
    import numpy as np
    import pandas as pd
    from shadowtransfer import armc_dump, armc_runtime
    from shadowtransfer.armJ_png_only import (PROB_ONLY_COLS, PROB_SOURCE,
                                              perimage_row_png_only)

    t0 = time.time()
    out_dir = os.path.join(out_root, c["run_id"])
    pq = os.path.join(out_dir, "perimage.parquet")
    rj_path = os.path.join(out_dir, "run.json")
    done, failed = os.path.join(out_dir, ".DONE"), os.path.join(out_dir, ".FAILED")

    # Refuse a quarantined write target up front, before the expensive part.
    armc_runtime.preflight_write_targets(
        BASE, [pq, rj_path, done, failed], label="armJ:" + c["run_id"])

    os.makedirs(out_dir, exist_ok=True)
    for m in (done, failed):
        if os.path.exists(m):
            os.unlink(m)

    if not os.path.isdir(c["pred_dir"]):
        open(failed, "w").write("pred_dir absent: %s\n" % c["pred_dir"])
        return {**c, "error": "pred_dir absent"}

    geo = armc_dump.load_geo_meta(BASE)
    stems = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(c["pred_dir"], "*.png")))
    if smoke:
        stems = stems[:smoke]

    rows, shape_mismatch = [], []
    for stem in stems:
        png = cv2.imread(os.path.join(c["pred_dir"], stem + ".png"),
                         cv2.IMREAD_GRAYSCALE)
        raw_gt = cv2.imread(os.path.join(c["mask_dir"], stem + ".png"),
                            cv2.IMREAD_GRAYSCALE)
        if png is None or raw_gt is None:
            open(failed, "w").write("missing input for %s\n" % stem)
            return {**c, "error": "missing input for %s" % stem}
        if png.shape != raw_gt.shape:
            shape_mismatch.append({"stem": stem, "png": list(png.shape),
                                   "gt": list(raw_gt.shape)})
            open(failed, "w").write("shape mismatch %s\n" % stem)
            return {**c, "error": "shape mismatch %s" % stem,
                    "shape_mismatch": shape_mismatch}
        pred = (png > 127).astype(np.uint8)
        gt = (raw_gt > 127).astype(np.uint8)
        rows.append(perimage_row_png_only(
            c["run_id"], stem, c["city"], c["res"], gt, pred,
            geo.get(stem + ".png")))

    df = pd.DataFrame(rows)
    df.to_parquet(pq, index=False)

    # Per-cell means: self-check / provenance quantities only. No Delta, no
    # bootstrap, no Holm, no recovery ratio -- those are the analysis host's.
    miou_k2 = float(df["miou_tol_pct"].mean())
    miou_k0 = float(df["miou_str_pct"].mean())
    miou_k5 = float(
        (df["tp_k5"] / (df["tp_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100
         + df["tn_k5"] / (df["tn_k5"] + df["fp_k5"] + df["fn_k5"] + 1e-10) * 100)
        .mean() / 2.0)
    fg_k2 = float(df["iou_tol"].mean()) * 100.0
    fg_k0 = float(df["iou_str"].mean()) * 100.0
    fg_k5 = float(df["iou_k5"].mean()) * 100.0

    stored = c["stored_tol2px_miou_twoclass"]
    d_stored = None if stored is None else miou_k2 - stored

    ta, tad = TARGET_ACCESS[c["arch"]]
    rung_ok = c["n_train"] == RUNG
    run = {
        "run_id": c["run_id"], "dataset": "st", "arm": "J",
        "arch": c["arch"], "method": c["method_token"], "mode": "loco",
        "rung": RUNG,
        "rung_ok": rung_ok,
        "rung_label_anomaly": (None if rung_ok else {
            "what": "no artifact of this cell records a training count",
            "rung_written": RUNG,
            "basis": "OGLANet's train.log states 900/300/150 for the same "
                     "data root and LOCO fold construction; the per-city "
                     "directories under data/Final_data_test hold 450 train "
                     "/ 150 val, so a 2-city pool is 900/300",
            "why_not_authoritative": "CLAUDE.md 2026-07-28: directory "
                                     "contents are not authoritative for "
                                     "n_train/n_val",
        }),
        "n_train": c["n_train"], "n_val": c["n_val"],
        "n_train_evidence": c["n_train_evidence"],
        "n_test": int(len(df)),
        "holdout": c["city"], "res": c["res"],
        "seed": SEED_NOMINAL,
        "seed_source": "NOMINAL -- no seed is recorded on this cell",
        "row": c["row"], "row_kind": c["row_kind"],
        "row_label_printed": c["row_label_printed"],
        "row_label_actual": c["row_label_actual"],
        "row_label_mismatch": c["row_label_mismatch"],
        "row_resolved_by": "config flags, not the directory A-number",
        "row_evidence": c["row_evidence"],
        "reference_run_id": ref_run_id(c["arch"], c["city"]),
        "reference_tag": REF_TAG[c["arch"]],
        "reference_rationale": "the batch's own reference: the configuration "
                               "each ablation toggles from",
        "published_sib_cell_run_id": c4clean_run_id(c["arch"], c["city"]),
        "published_sib_cell_note":
            "the published SIB cell is C4clean, NOT this batch's reference. "
            "For MAMNet the two differ on flags the ablations toggle: "
            "C4clean adaptive_beta=False use_content_aug=False; "
            "M1 adaptive_beta=True use_content_aug=True.",
        "target_access": ta, "target_access_detail": tad,
        "parent_run_id": None,
        "legacy_run_dir": c["legacy_run_dir"],
        "source": "SAVED PREDICTION PNG MASKS",
        "pred_dir": c["pred_dir"], "source_mask_dir": c["mask_dir"],
        "prediction_recipe": "(png > 127); the stored 0/255 argmax mask, "
                             "scored as-is, no thresholding, no size filter",
        "prob_source": PROB_SOURCE,
        "deviation_png_only": {
            "approved_by": "operator decision 2, 2026-11-05",
            "what": "CONTRACT section 3 probability columns are null",
            "columns_nulled": PROB_ONLY_COLS,
            "why": "this cell has no stored probability maps; they exist only "
                   "for the C4clean reference, under "
                   "data/{arch}/sp_gap_results/c4clean_probs_*",
            "probs_npz_written": False,
            "metric_unaffected": "two-class boundary-tolerant mIoU at K=2 is "
                                 "computed from the masks alone",
        },
        "row_assembler": "shadowtransfer.armJ_png_only.perimage_row_png_only "
                         "-- every count and IoU from the same canonical "
                         "metrics.py functions armc_dump.perimage_row calls; "
                         "armc_dump.py is NOT modified",
        "metric_module_path": "shadowtransfer/metrics.py",
        "metric_module_sha": sha256_file(
            os.path.join(BASE, "shadowtransfer/metrics.py")),
        "reporting_metric": "two-class boundary-tolerant mIoU, K=2 "
                            "(miou_tol_pct), per Appendix E's band",
        "decision_metric": "unchanged from the published run; this job "
                           "selects no checkpoint",
        "cell_mean_miou_k2_twoclass": miou_k2,
        "cell_mean_miou_k0_twoclass": miou_k0,
        "cell_mean_miou_k5_twoclass": miou_k5,
        "cell_mean_fg_iou_k2": fg_k2,
        "cell_mean_fg_iou_k0": fg_k0,
        "cell_mean_fg_iou_k5": fg_k5,
        "stored_tol2px_miou_twoclass": stored,
        "delta_recomputed_minus_stored_miou_k2": d_stored,
        "stored_repro_gate_pass": (None if d_stored is None
                                   else abs(d_stored) <= 0.01),
        "stored_tolerant_keys_available": c["stored_tolerant_keys"],
        "k5_printed_gate": {
            "state": "N/A -- replaced by the provenance gate",
            "why": "no K=5 value exists for any of these cells on Delta; "
                   "every stored artifact carries strict and tolerant_2px "
                   "only",
            "gate_script": "python/armJ_sib_ablation_provenance_gate.py",
        },
        "config_source": c["config_source"], "config_kind": c["config_kind"],
        "config_sha256": (sha256_file(c["config_source"])
                          if c["config_source"]
                          and os.path.isfile(c["config_source"]) else None),
        "config_verbatim": c["config_flags"],
        "checkpoint_state": c["checkpoint_state"],
        "ckpt_path": None,
        "batch_size": c["batch_size"],
        "inference_batch_size": c["batch_size"],
        "inference_note": (
            "OGLANet: test_and_save_predictions at batch_size=%s under fp16 "
            "autocast (train_oglanet_sib.py:662,:732,:1554) -- the "
            "sib_trainer path of E-9; these masks are ONE DRAW from a "
            "nondeterministic distribution" % c["batch_size"]
            if c["arch"] == "oglanet" else
            "no inference was run by this job; the stored masks were scored"),
        "structure_cols": "deferred",
        "caveats": CAVEATS,
        "smoke_truncated_to_n_images": (smoke or None),
        "cluster": "delta",
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "compute_note": "CPU-only work on gpuA100x4 with the A100 allocated "
                        "and idle; no CPU association exists on this "
                        "allocation",
        "git_hash": None,
        "git_hash_reason": "per CLAUDE.md, BASE_PATH provenance rests on "
                           "config_sha256 and metric_module_sha",
        "wall_clock_s": round(time.time() - t0, 2),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(run, open(rj_path, "w"), indent=1)
    open(done, "w").write(run["written_utc"] + "\n")
    return {**c, "n_images": int(len(df)),
            "miou_k2_twoclass": miou_k2, "miou_k5_twoclass": miou_k5,
            "fg_iou_k2": fg_k2,
            "delta_recomputed_minus_stored_miou_k2": d_stored,
            "stored_repro_gate_pass": run["stored_repro_gate_pass"],
            "parquet": pq, "parquet_sha256": sha256_file(pq),
            "run_json_sha256": sha256_file(rj_path),
            "wall_clock_s": run["wall_clock_s"]}


def _work(args):
    c, out_root, smoke = args
    try:
        return score_cell(c, out_root, smoke)
    except Exception as e:
        return {**c, "error": "%s: %s" % (type(e).__name__, e),
                "traceback": traceback.format_exc()}


def do_selftest():
    """Prove the PNG-only assembler == armc_dump.perimage_row off-prob.

    Uses a REAL cell that has probabilities: the C4clean reference.
    """
    import cv2
    import numpy as np
    from shadowtransfer.armJ_png_only import selftest_against_perimage_row

    arch, city = "mamnet", "chicago"
    pdir = os.path.join(BASE, "data", arch, "sp_gap_results",
                        "c4clean_probs_%s_%s_highres" % (arch, city))
    mdir = os.path.join(BASE, "data/Final_data_test", city,
                        "highres/test/masks")
    preds = glob.glob(os.path.join(
        BASE, "data", arch, "outputs",
        "%s_sib_C4clean_*%s*highres" % (arch, city), "predictions"))
    if not preds:
        raise SystemExit("FATAL selftest: no C4clean predictions dir")
    npys = sorted(glob.glob(os.path.join(pdir, "*.npy")))[:3]
    if not npys:
        raise SystemExit("FATAL selftest: no probability maps at %s" % pdir)
    outs = []
    for npy in npys:
        stem = os.path.splitext(os.path.basename(npy))[0]
        prob = np.load(npy).astype(np.float32)
        gt = (cv2.imread(os.path.join(mdir, stem + ".png"),
                         cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        png = cv2.imread(os.path.join(preds[0], stem + ".png"),
                         cv2.IMREAD_GRAYSCALE)
        pred = (png > 127).astype(np.uint8)
        outs.append(selftest_against_perimage_row(gt, prob, pred))
        print("  selftest image %s OK" % stem)
    print("SELFTEST PASS on %d real images from %s/%s C4clean"
          % (len(outs), arch, city))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=os.path.join(
        os.environ["ST_SCRATCH_ROOT"], "runs/sib_ablation_inventory/inventory.json"))
    ap.add_argument("--out_root", default=os.path.join(
        os.environ["ST_SCRATCH_ROOT"], "dumps/st_sib_ablation_png"))
    ap.add_argument("--results", required=False)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="all")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--timebox", type=int, default=1800)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        do_selftest()
        return

    cells = build_cells(a.inventory)
    if a.only != "all":
        want = set(a.only.split(","))
        cells = [c for c in cells
                 if c["run_id"] in want or c["row"] in want
                 or c["arch"] in want]
    print("CELLS: %d  (ablation %d, extra_row %d, reference %d, "
          "published_sib_cell %d)" % (
              len(cells),
              sum(1 for c in cells if c["row_kind"] == "ablation"),
              sum(1 for c in cells if c["row_kind"] == "extra_row"),
              sum(1 for c in cells if c["row_kind"] == "reference"),
              sum(1 for c in cells if c["row_kind"] == "published_sib_cell")))
    for c in cells:
        print("  %-46s %-8s row=%-4s %-18s %s" % (
            c["run_id"], c["arch"], c["row"], c["row_kind"],
            c["legacy_run_dir"]))
    if a.dry_run:
        print("DRY RUN -- nothing scored")
        return

    t0 = time.time()
    out, errs = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_work, (c, a.out_root, a.smoke)): c for c in cells}
        for fu in as_completed(futs):
            r = fu.result()
            if r.get("error"):
                errs.append(r)
                print("FAILED %s: %s" % (r["run_id"], r["error"]))
            else:
                out.append(r)
                print("ok %-46s n=%-4d miouK2=%8.4f  dStored=%s" % (
                    r["run_id"], r["n_images"], r["miou_k2_twoclass"],
                    ("%+.4f" % r["delta_recomputed_minus_stored_miou_k2"])
                    if r["delta_recomputed_minus_stored_miou_k2"] is not None
                    else "n/a"))
            if time.time() - t0 > a.timebox:
                print("TIMEBOX %ds exceeded during scoring" % a.timebox)
                break

    res = {"n_cells": len(cells), "n_done": len(out), "n_failed": len(errs),
           "out_root": a.out_root, "smoke": a.smoke,
           "wall_clock_s": round(time.time() - t0, 2),
           "cells": out, "errors": errs}
    if a.results:
        os.makedirs(os.path.dirname(a.results), exist_ok=True)
        json.dump(res, open(a.results, "w"), indent=1, default=str)
        man = a.results.replace(".json", "_sha256_manifest.txt")
        with open(man, "w") as f:
            for r in sorted(out, key=lambda x: x["run_id"]):
                f.write("%s  %s/perimage.parquet\n"
                        % (r["parquet_sha256"], r["run_id"]))
                f.write("%s  %s/run.json\n"
                        % (r["run_json_sha256"], r["run_id"]))
        print("manifest %s" % man)
        print("results  %s" % a.results)
    print("DONE %d/%d  failed %d  wall %.1fs"
          % (len(out), len(cells), len(errs), res["wall_clock_s"]))
    if errs:
        sys.exit(1)


if __name__ == "__main__":
    main()
