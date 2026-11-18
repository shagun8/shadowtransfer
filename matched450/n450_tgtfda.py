#!/usr/bin/env python3
"""n450 OGLANet-SIB target-FDA retrain -- one cell, through the n450 harness.

Three cells: oglanet-sibc4tgt-loco-r450-{chicago,miami,phoenix}-hr-s42-n450.
Each is the matched-450 OGLANet-SIB cell with ONE change -- the FDA style pool
on the CNN branch is the held-out city's train-split images (paper S5.1 /
App. F) instead of the other SOURCE city's train set (dataset_sib.py:291-294).

HOW IT REUSES THE HARNESS, AND WHAT IT PATCHES
----------------------------------------------
Nothing in python/n450_*.py, python/shadowtransfer/n450_*.py,
train_oglanet_sib.py, dataset_sib.py or dffm.py is edited. This driver rebinds
four things in-process and then calls n450_train.main():

  1 n450_roster.by_run_id  -- adds the three sibc4tgt run_ids. Each derived
    cell is the sibc4 cell with ONLY run_id, ckpt_dir, dump_dir, target_access
    and target_access_detail changed. `method` stays "sibc4" on purpose:
    n450_train branches on it (`sib = method == "sibc4"`), n450_score picks the
    SIB test_results field off it, and orig_dir / orig_dump_dir / orig_run_id
    must keep pointing at the SAME published cell so the dump's "original"
    comparison is the published -s0 one, exactly as the three n450 sibc4 cells
    have it. The new-vs-cross-source-n450 comparison is reported separately.

  2 n450_config.PERMITTED_CHANGED -- permits fda_target_root. The published SIB
    config declares it as `.../{holdout}/highres` (the held-out CITY ROOT, all
    splits -- the same spelling E-12 found contaminating MAMNet) and the OGLANet
    trainer never dereferences it. This run narrows it to `.../{holdout}/
    highres/train/images`, matching arme12_retrain.clean_pool_root. It stays
    inert; the donor that actually acts is the dataset object in 3.

  3 n450_sib.install -- wrapped so train_oglanet_sib_tgtfda.install runs FIRST.
    Order matters: the donor dataset must be constructed, asserted and recorded
    by n450_inject BEFORE n450_sib's wrapper calls INJ.assert_counts(), so the
    donor shows up as n_holdout_unlabeled=450 and check_access() sees the
    held-out access this cell declares. It also wraps the test pass to capture
    the torch RNG seed and state hash at that point (OGLANet inference is
    RNG-sensitive: DFFM rebuilds unregistered batch-statistic BatchNorms per
    forward).

  4 n450_score.score -- adds the tgtfda block to run.json before the dump is
    written: fda_donor, fda_donor_list_sha256, the donor assertions, the
    cross-source donors replaced, and the test-pass RNG record.

  python n450_tgtfda.py --run_id <id> [--smoke] [--dry_run] [--dump_only]
                        [--construct_only] [--list]
"""
import argparse
import hashlib
import json
import os
import sys

BASE = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from shadowtransfer import n450_roster as R                  # noqa: E402
from shadowtransfer import n450_config as CFG                # noqa: E402
import train_oglanet_sib_tgtfda as TGT                       # noqa: E402

# n450_sib imports numpy and n450_score imports cv2/pandas. Both are deferred
# into the patch functions that need them, so `import n450_tgtfda` stays
# stdlib-only: the worker computes CELL_DIR with the SYSTEM python3, BEFORE the
# module/conda block (the traps have to be armed before conda can fail), and a
# numpy-less interpreter there must not break the marker path.

METHOD_TAG = "sibc4tgt"
DONOR_MANIFEST = os.path.join(R.N450_ROOT, "fda_donor_manifest.json")
# Set by install(); read by the score wrapper.
STATE = {"donor": None, "rng": None, "run_id": None}


def new_run_id(city):
    return f"oglanet-{METHOD_TAG}-loco-r450-{city}-{R.RES}-s{R.SEED}-n450"


def derive(city):
    """The sibc4 cell with only run_id / paths / target_access changed."""
    base = {c["city"]: c for c in R.roster()
            if c["arch"] == "oglanet" and c["method"] == "sibc4"}[city]
    rid = new_run_id(city)
    cell = dict(base)
    cell.update(
        run_id=rid,
        ckpt_dir=os.path.join(R.SHADE, "data", "oglanet", "outputs", "n450", rid),
        dump_dir=os.path.join(R.DUMP_ROOT, rid),
        target_access="unlabeled_target",
        target_access_detail=(
            "FDA style pool on the SIB CNN branch = the HELD-OUT city's "
            "train-split images (450, unlabeled; paper S5.1 / App. F), "
            "installed as dataset_sib's fda_target_dataset. Replaces the "
            "cross-SOURCE donor get_dataloaders_sib:291-294 falls back to. "
            "No held-out label, val or test file is read in training."),
        tgtfda=True,
        method_tag=METHOD_TAG,
    )
    return cell


def cells():
    return {new_run_id(c): derive(c) for c in R.CITIES}


def donor_expectations():
    with open(DONOR_MANIFEST) as f:
        m = json.load(f)
    for city in R.CITIES:
        e = m["cities"][city]
        if e["verdict"] != "PASS" or e["n_donor"] != TGT.N_DONOR_EXPECT:
            raise SystemExit(f"FATAL donor manifest for {city} is not a "
                             f"passing {TGT.N_DONOR_EXPECT}-file list: "
                             f"{e['verdict']} n={e['n_donor']}")
    return m


# ------------------------------------------------------------------ patches --
def patch_roster():
    extra = cells()
    orig = R.by_run_id

    def by_run_id():
        d = orig()
        for k, v in extra.items():
            if k in d:
                raise RuntimeError(f"run_id collision: {k}")
            d[k] = v
        return d

    R.by_run_id = by_run_id
    return extra


def patch_permits(city, donor_dir):
    CFG.PERMITTED_CHANGED["fda_target_root"] = (
        "THE ONE SUBSTANTIVE CHANGE, declared. The published OGLANet-SIB "
        f"config carries fda_target_root = .../{city}/highres (the held-out "
        "CITY ROOT: train+val+test, images and both mask subtrees) and "
        "train_oglanet_sib.py never dereferences it -- neither "
        "get_dataloaders_sib call passes it and dataset_sib."
        "get_dataloaders_sib has no such parameter. This run narrows the "
        f"recorded value to {donor_dir} (= arme12_retrain.clean_pool_root, the "
        "pool the clean MAMNet-SIB retrains pin) and installs that directory "
        "as the dataset-level FDA donor, which is the mechanism OGLANet "
        "actually reads. The string itself remains inert.")


def patch_resolve(city):
    """Put fda_target_root into the trainer's resolved config, so CHECK 4's
    config_diff.json names the donor as a permitted difference rather than
    leaving it invisible (the diff is computed before the dataloaders exist)."""
    real = CFG.resolve

    def resolve(mod, fn, baseline, overrides):
        o = dict(overrides)
        o["fda_target_root"] = TGT.split_dir(
            baseline["base_data_root"], city, baseline["resolution"], "train")
        return real(mod, fn, baseline, o)

    CFG.resolve = resolve


def patch_sib_install(city, expected_basename_sha, content_check=True):
    from shadowtransfer import n450_sib as SIB
    real = SIB.install

    def install(mod, ns, probs_dir, timebox):
        st = TGT.install(mod, ns, city, expected_basename_sha,
                         content_check=content_check)
        names = real(mod, ns, probs_dir, timebox)
        inner = mod.test_and_save_predictions

        def test_and_save_predictions(model, loader, device, args, logger):
            import torch
            s = torch.get_rng_state()
            cu = (torch.cuda.get_rng_state_all()
                  if torch.cuda.is_available() else [])
            STATE["rng"] = {
                "where": "immediately before the SIB trainer's test pass",
                "torch_initial_seed": int(torch.initial_seed()),
                "wrapper_seed": R.SEED,
                "cpu_state_sha256": hashlib.sha256(
                    s.cpu().numpy().tobytes()).hexdigest(),
                "cuda_state_sha256": [
                    hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()
                    for x in cu],
                "why": "OGLANet inference is RNG-sensitive: DFFM rebuilds 20 "
                       "unregistered batch-statistic BatchNorms inside "
                       "forward() (runs/dffm_defect/provenance.json), so the "
                       "test-pass RNG is part of the result's provenance",
            }
            print(f"[tgtfda] test-pass RNG: initial_seed="
                  f"{STATE['rng']['torch_initial_seed']} cpu_state_sha="
                  f"{STATE['rng']['cpu_state_sha256'][:16]}", flush=True)
            return inner(model, loader, device, args, logger)

        mod.test_and_save_predictions = test_and_save_predictions
        STATE["donor_state"] = st
        return names + ["fda_target_dataset -> held-out city train split",
                        "test_and_save_predictions(+RNG record)"]

    SIB.install = install


def patch_score(manifest_entry):
    import n450_score
    real = n450_score.score

    def score(run_id, cell, probs_dir, dump_dir, run, **kw):
        st = STATE.get("donor_state") or {}
        run = dict(run)
        run["method_tag"] = METHOD_TAG
        run["fda_donor"] = "heldout_city_train_images"
        run["fda_donor_list_sha256"] = manifest_entry["donor_list_sha256"]
        run["fda_donor_basename_sha256"] = manifest_entry["donor_basename_sha256"]
        run["fda_donor_content_sha256"] = manifest_entry["donor_content_sha256"]
        run["fda_donor_list_path"] = manifest_entry["list_path"]
        run["fda_donor_list_file_sha256"] = manifest_entry["file_sha256"]
        run["fda_donor_n"] = manifest_entry["n_donor"]
        run["fda_donor_root"] = manifest_entry["donor_root"]
        run["fda_donor_definition_source"] = (
            "sib/arme12_retrain.py:114-115 clean_pool_root + :122-139 "
            "enumerate_pool (= mamnet/data/fda_transform.py:196-203 walk + "
            "ext filter); step 0b builder matched450/n450_tgtfda_donor.py")
        run["fda_donor_runtime_assertions"] = st.get("donor")
        run["fda_donor_draw_rule"] = (st.get("donor") or {}).get("draw_rule")
        run["tgtfda_module_sha256"] = CFG.sha256_file(
            os.path.join(BASE, "sib/train_oglanet_sib_tgtfda.py"))
        run["tgtfda_driver_sha256"] = CFG.sha256_file(
            os.path.join(BASE, "matched450/n450_tgtfda.py"))
        run["donor_builder_sha256"] = CFG.sha256_file(
            os.path.join(BASE, "matched450/n450_tgtfda_donor.py"))
        run["test_pass_rng"] = STATE.get("rng")
        run["fda_donor_pool_size_note"] = {
            "this_cell_donor_n": 450,
            "n450_cross_source_comparator_donor_n_per_city": 225,
            "published_cell_donor_n_per_city": 450,
            "why": (
                "A DECLARED SECOND DIFFERENCE against the n450 comparator, "
                "flagged rather than hidden. In the cross-source cells the "
                "donor IS the other source city's train dataset, and "
                "n450_inject rewrites that dataset's img_paths down to the "
                "shared 225 (n450_inject.py:99-101), so each training sample "
                "drew its style from 225 candidates (session log "
                "2026-10-31-oglanet-sib-n450-diagnosis.md, table row 'FDA "
                "donor pool / city', and its finding 7). This cell's donor is "
                "the held-out city's train split, which the injector records "
                "but does NOT filter (role holdout_train_unlabeled), so each "
                "sample draws from 450. So vs the n450 comparator the donor "
                "differs in CITY and in POOL SIZE; vs the PUBLISHED cell the "
                "pool size matches at 450 and only the city differs."),
            "why_not_matched_at_225": (
                "The 450-image budget is a TRAINING budget, not a statement "
                "about the unlabeled donor pool. The paper (S5.1 / App. F) "
                "defines the pool as the held-out city's train split, and "
                "arme12_retrain.clean_pool_root pins that whole 450-file "
                "directory for the clean MAMNet-SIB retrains. Subsampling the "
                "held-out donor to 225 would require choosing WHICH 225 with "
                "no provenance in the paper or in E-12, which is a larger "
                "invention than leaving it whole."),
        }
        run["cross_source_n450_cell"] = {
            "run_id": f"oglanet-sibc4-loco-r450-{cell['city']}-hr-s42-n450",
            "dump_dir": os.path.join(
                R.DUMP_ROOT,
                f"oglanet-sibc4-loco-r450-{cell['city']}-hr-s42-n450"),
            "what": "the same cell with the cross-SOURCE FDA donor; the "
                    "single-variable comparator for this run",
        }
        if run["fda_donor_runtime_assertions"] is None:
            raise SystemExit("FATAL no donor assertion record: the donor was "
                             "never installed, so this cell is not a tgtfda "
                             "cell. Refusing to write the dump.")
        if run["fda_donor_runtime_assertions"]["verdict"] != "PASS":
            raise SystemExit("FATAL donor assertions did not pass")
        return real(run_id, cell, probs_dir, dump_dir, run, **kw)

    n450_score.score = score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no_content_check", action="store_true",
                    help="skip the sha256 byte-overlap half of the in-job "
                         "donor assertion (the id/dir/raster halves still run)")
    a, rest = ap.parse_known_args()
    if a.list:
        for rid, c in cells().items():
            print(rid, c["city"], c["target_access"], c["dump_dir"])
        return 0
    if not a.run_id:
        ap.error("--run_id is required")
    extra = patch_roster()
    if a.run_id not in extra:
        raise SystemExit(f"FATAL {a.run_id} is not a tgtfda cell. "
                         f"Known: {sorted(extra)}")
    cell = extra[a.run_id]
    city = cell["city"]
    man = donor_expectations()
    e = man["cities"][city]
    STATE["run_id"] = a.run_id
    patch_permits(city, e["donor_root"])
    patch_resolve(city)
    patch_sib_install(city, e["donor_basename_sha256"],
                      content_check=not a.no_content_check)
    patch_score(e)
    print(f"[tgtfda] driver armed for {a.run_id}\n"
          f"  donor      {e['donor_root']}  n={e['n_donor']}\n"
          f"  donor sha  {e['donor_list_sha256']}\n"
          f"  list file  {e['list_path']}\n"
          f"  ckpt       {cell['ckpt_dir']}\n"
          f"  dump       {cell['dump_dir']}\n"
          f"  comparator oglanet-sibc4-loco-r450-{city}-hr-s42-n450",
          flush=True)
    import n450_train
    sys.argv = ["n450_train.py", "--run_id", a.run_id] + rest
    return n450_train.main()


if __name__ == "__main__":
    sys.exit(main())
