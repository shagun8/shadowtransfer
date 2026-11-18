"""ARM A BLOCK 5 (D-B5-1) -- the s4.2 histogram-matching probe on the SAR cells.

D-B4-2 stopped the 36 SAR probes because a SAR cell's checkpoint IS its vanilla
parent's and SAR adapts normalisation affines PER IMAGE AT TEST TIME: a
vanilla-build probe would re-probe the vanilla model. This build runs SAR's own
test-time adaptation ON THE MATCHED IMAGES, through the route that wrote each
SAR parent dump, so the probe constructs the object the SAR cell scored.

IMPORTED UNCHANGED, never copied: shadowtransfer/sar.py; armf_sar_pass.run_pass
(episodic per-image restore, byte-compare, SAM step, reset); armA_a4_sar_seo
(the mamnet/dinov3/oglanet route); armA_t1_cesar_adapt (the CrossEarth route:
D-B3-3 log(normalised score), branch-4 full-state snapshot); armA_seo_histmatch
(the matching rule, its MatchedSEODataset injection point, the source CDF).

WHAT THIS MODULE ADDS, all as in-memory rebinds; no file above is edited:
  1. SeenCaptureDataset -- MatchedSEODataset that also RECORDS each patch's
     DSM-seen `valid` mask. Both SAR routes compute it and then discard it
     (they score on the plain `mask`, unseen counted as background, D-B3-3).
  2. FinalStepCapture -- rebinds the two names run_pass scores with
     (armf_sar_pass.threshold_and_filter / band_counts, bound in ITS namespace)
     to OBSERVE the exact prob and pred of the final step, so the SAME
     prediction is also scored by seo_dump.perimage_row with unseen excluded.
     Both wrappers return the originals' values unchanged.
  3. resolve() -- the SAR parent cell, its recipe and route inputs, from its
     own run.json and nowhere else.
"""
import json
from pathlib import Path

import numpy as np

from shadowtransfer import armf_sar_pass, sar, seo_dump

import armA_seo_histmatch as hm

RUNG = 3500
CNN_ROUTE = ("mamnet", "dinov3", "oglanet")
ARCHS = CNN_ROUTE + ("crossearth",)
FOLDS = ("OMA", "JAX", "UCSD")
UNADAPTED = "unadapted"
#: name -> (matcher kind, pass kind). Order is execution order.
CONDITIONS = (("unadapted_unmatched", None, UNADAPTED),
              ("sar_unmatched", None, "sar"),
              ("sar_matched_gray", "gray", "sar"),
              ("sar_matched", "perch", "sar"))
COUNT_COLS = tuple(f"{c}_k{k}" for k in (0, 2, 5) for c in ("tp", "fp", "fn", "tn")) \
    + ("n_valid_px_k2",)


def pid_of(r):
    return f"{r['image_id']}_y{r['y']}_x{r['x']}"


def sar_parent_id(arch, fold, seed):
    return f"{arch}-sar-loco-r{RUNG}-{fold.lower()}-hr-s{seed}"


class SeenCaptureDataset(hm.MatchedSEODataset):
    """The histmatch injection point, plus a record of each patch's seen mask.

    `matcher` is a CLASS attribute so a route that constructs the dataset
    internally still sees it; materialise() sets and clears it."""

    SEEN = {}

    def __getitem__(self, i):
        if not self.exclude_unseen or self.exclude_uncertain:
            raise SystemExit("FATAL the route built SEODataset with exclude_unseen="
                             f"{self.exclude_unseen} exclude_uncertain="
                             f"{self.exclude_uncertain}; `valid` would not be the "
                             "DSM-seen mask the arm A metric uses")
        out = super().__getitem__(i)
        pid = pid_of(self.rows[i])
        if out.get("filename") != pid:
            raise SystemExit(f"FATAL filename {out.get('filename')} != {pid}")
        SeenCaptureDataset.SEEN[pid] = out["valid"].numpy().astype(bool).copy()
        return out


def install_dataset(arch):
    """Rebind the class name each route's build_split constructs."""
    import armA_pilot_train as P
    P.SEODataset = SeenCaptureDataset
    names = ["armA_pilot_train.SEODataset"]
    if arch in CNN_ROUTE:
        import armA_phase3_train as p3
        p3.SEODataset = SeenCaptureDataset
        names.append("armA_phase3_train.SEODataset")
    return names


def materialise(ctx, ids, matcher):
    """The route's own image materialisation over `ids`, under `matcher`."""
    from shadowtransfer import armA_a4_sar_seo as SEO
    from shadowtransfer import armA_t1_cesar_adapt as CE
    SeenCaptureDataset.SEEN = {}
    SeenCaptureDataset.matcher = matcher
    label = f"{ctx['fold']}/test"
    try:
        if ctx["arch"] == "crossearth":
            imgs = CE.images_from_patch_ids_crossearth(
                ctx["cfg"], list(ids), "test", label, 0, **ctx["roots"])
        else:
            imgs = SEO._images_from_patch_ids(
                ctx["arch"], ctx["cfg"], list(ids), "test", label, 0, **ctx["roots"])
    finally:
        SeenCaptureDataset.matcher = None
    got = [im["image_id"] for im in imgs]
    if got != list(ids):
        raise SystemExit(f"FATAL route returned {len(got)} images out of order / "
                         f"incomplete against {len(ids)} ids")
    seen = dict(SeenCaptureDataset.SEEN)
    missing = [p for p in ids if p not in seen]
    if missing:
        raise SystemExit(f"FATAL {len(missing)} patches never passed through "
                         f"SeenCaptureDataset (first {missing[:3]}): the route did "
                         "not construct the injected class, so the matcher did "
                         "not reach these images")
    return imgs, seen


class FinalStepCapture:
    """Observe run_pass's final-step prob/pred; score them unseen-excluded too.

    run_pass loops images in order and, per image, steps 0..S, calling
    threshold_and_filter(prob) then band_counts(pred, gt) once per step. Call k
    is therefore image k // (S+1), step k % (S+1); `gt is images[idx]['gt']`
    is asserted on every final step, so a drift in that loop fails loudly."""

    def __init__(self, run_id, images, seen, meta, steps, keep_probs):
        self.run_id, self.images, self.seen, self.meta = run_id, images, seen, meta
        self.steps, self.keep_probs = int(steps), keep_probs
        self.k, self._prob, self.rows, self.probs = 0, None, [], {}
        self._orig = None

    def install(self):
        self._orig = (armf_sar_pass.threshold_and_filter, armf_sar_pass.band_counts)
        armf_sar_pass.threshold_and_filter = self._taf
        armf_sar_pass.band_counts = self._bc

    def uninstall(self):
        armf_sar_pass.threshold_and_filter, armf_sar_pass.band_counts = self._orig

    def _taf(self, prob, min_pixels=10):
        self._prob = prob
        return self._orig[0](prob, min_pixels=min_pixels)

    def _bc(self, pred, gt):
        c = self._orig[1](pred, gt)
        idx, step = divmod(self.k, self.steps + 1)
        self.k += 1
        if step == self.steps:
            im = self.images[idx]
            if gt is not im["gt"]:
                raise SystemExit(f"FATAL capture out of step with run_pass at call "
                                 f"{self.k - 1} (image {idx})")
            pid = im["image_id"]
            prob = np.asarray(self._prob, dtype=np.float32)
            pred_np = np.asarray(pred).astype(np.uint8)
            self.rows.append(seo_dump.perimage_row(
                self.run_id, self.meta[pid], gt, prob, pred_np, self.seen[pid]))
            if self.keep_probs:
                self.probs[pid] = prob.astype(np.float16)
        return c


def run_condition(ctx, spec, images, seen, device, keep_probs=False):
    """One run_pass over `images` with the capture installed."""
    cap = FinalStepCapture(ctx["probe_run_id"], images, seen, ctx["meta"],
                           spec["steps"], keep_probs)
    cap.install()
    try:
        summ, rows = armf_sar_pass.run_pass(
            spec, ctx["arch"], ctx["cfg"], ctx["ckpt"], device, images,
            rho=ctx["recipe"]["rho"], momentum=ctx["recipe"]["momentum"])
    finally:
        cap.uninstall()
    if cap.k != len(images) * (spec["steps"] + 1) or len(cap.rows) != len(images):
        raise SystemExit(f"FATAL capture saw {cap.k} calls / {len(cap.rows)} final "
                         f"rows for {len(images)} images x {spec['steps'] + 1} steps")
    fin = [r["image_id"] for r in rows if r["step"] == spec["steps"]]
    if fin != [im["image_id"] for im in images]:
        raise SystemExit("FATAL run_pass final-step rows out of image order")
    if not summ["episodic_state_isolation"]:
        raise SystemExit(f"FATAL episodic isolation failed on {spec['pass_id']}: "
                         f"{summ['episodic_mismatch_keys']}")
    return summ, rows, cap.rows, cap.probs


def resolve(parent_dir):
    """The SAR parent, from its own run.json. Refuses anything not a SAR grid cell."""
    pdir = Path(parent_dir)
    if not (pdir / ".DONE").exists():
        raise SystemExit(f"FATAL SAR parent {pdir} has no .DONE")
    run = json.loads((pdir / "run.json").read_text())
    arch, fold, seed = run["arch"], run["holdout"], int(run["seed"])
    want = sar_parent_id(arch, fold, seed)
    bad = [k for k, ok in (("method", run.get("method") == "sar"),
                           ("stage", run.get("stage") == "grid"),
                           ("trains", run.get("trains") is False),
                           ("run_id", run.get("run_id") == want == pdir.name),
                           ("arch", arch in ARCHS), ("fold", fold in FOLDS)) if not ok]
    if bad:
        raise SystemExit(f"FATAL {pdir} is not a SAR grid cell: {bad}")
    rec = run["recipe"]
    spec = armf_sar_pass.sar_spec(float(rec["lr"]), float(rec["lr_multiplier"]),
                                  int(rec["steps"]))
    if abs(float(rec["lr"]) - sar.SAR_LR_BASE * float(rec["lr_multiplier"])) > 1e-15:
        raise SystemExit(f"FATAL recipe lr {rec['lr']} != lr_base x multiplier")
    stored_ids = [s["pass_id"] for s in run["summaries"]]
    if stored_ids != [UNADAPTED, spec["pass_id"]]:
        raise SystemExit(f"FATAL stored passes {stored_ids} != "
                         f"[{UNADAPTED}, {spec['pass_id']}]")
    return {"parent_dir": str(pdir), "parent_run_id": want, "run": run,
            "arch": arch, "fold": fold, "seed": seed,
            "vanilla_run_id": run["parent_run_id"], "ckpt": run["ckpt_path"],
            "recipe": {"lr": float(rec["lr"]), "lr_multiplier": float(rec["lr_multiplier"]),
                       "steps": int(rec["steps"]), "momentum": float(rec["momentum"]),
                       "rho": float(rec["rho"])},
            "sar_spec": spec, "unadapted_spec": armf_sar_pass.unadapted_spec(),
            "probe_run_id": f"{want}-{hm.PROBE_SUFFIX}"}
