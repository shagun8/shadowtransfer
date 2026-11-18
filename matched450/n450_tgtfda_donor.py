#!/usr/bin/env python3
"""n450 OGLANet-SIB target-FDA retrain -- STEP 0b: build and prove the donor list.

THE DONOR DEFINITION, lifted from the E-12 clean MAMNet-SIB retrains
---------------------------------------------------------------------
`sib/arme12_retrain.py:114-115` (`clean_pool_root`):

    BASE / "data" / "Final_data_test" / {city} / "highres" / "train" / "images"

enumerated by `enumerate_pool` (`arme12_retrain.py:122-139`), which reproduces
`mamnet/data/fda_transform.py:196-203` exactly: os.walk the root, keep every
file whose lowercased name ends in .png/.jpg/.jpeg/.tif/.tiff, no directory
excluded, sorted. That is the same 450-image directory the 18 clean standalone
FDA cells pin. This module enumerates the identical thing for OGLANet.

Only the ENUMERATION is shared with E-12. MAMNet then draws 100 files once
with np.random.choice inside load_target_images; OGLANet's SIB dataset instead
draws a fresh donor per training sample, uniformly over the whole donor list
(`oglanet/data/dataset_sib.py:200-202`). The draw rule stays each
architecture's own -- changing OGLANet's sampler would be a second variable.

ASSERTIONS (all must pass or nothing is written)
  A1  every path is under {holdout}/highres/train/images
  A2  zero label/mask files: by directory component, by filename pattern, and
      by opening each file and requiring a 3-channel 8-bit RGB raster
  A3  zero overlap with that city's test list and val list, by image id (stem)
      AND by sha256 of the file bytes
  A4  zero files from either source city, by path and by image id
  A5  the count is the expected 450

  python n450_tgtfda_donor.py [--out_dir ...] [--write]
"""
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

BASE = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth", "seo", "sib"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from shadowtransfer import n450_roster as R              # noqa: E402

VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
N_DONOR_EXPECT = 450
MASKISH = re.compile(r"mask|label|gt|seg|annot", re.I)
OUT_DEFAULT = os.path.join(R.N450_ROOT)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def city_split_dir(city, split, sub="images"):
    return os.path.join(R.DATA_ROOT, city, R.RES_DIR, split, sub)


def enumerate_pool(root):
    """`arme12_retrain.enumerate_pool`, behaviour for behaviour."""
    out = []
    root = str(root)
    if not os.path.isdir(root):
        raise SystemExit(f"FATAL pool root is not a directory: {root}")
    for r, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                out.append(os.path.join(r, f))
    return sorted(out)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def list_sha(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def raster_kind(p):
    """(mode, n_channels, dtype) as PIL reports it, without a full decode."""
    from PIL import Image
    with Image.open(p) as im:
        mode, size = im.mode, im.size
    nch = {"RGB": 3, "RGBA": 4, "L": 1, "P": 1, "1": 1, "I": 1, "F": 1,
           "I;16": 1, "LA": 2}.get(mode, -1)
    return mode, nch, size


def build(city, verbose=True):
    src = R.source_cities(city)
    root = city_split_dir(city, "train")
    pool = enumerate_pool(root)
    rec = {"holdout": city, "source_cities": src, "donor_root": root,
           "n_donor": len(pool), "assertions": {}, "failures": []}
    fail = rec["failures"]

    # ---- A1 every path under the holdout train/images dir -----------------
    nroot = os.path.normpath(root) + os.sep
    bad = [p for p in pool if not os.path.normpath(p).startswith(nroot)]
    rec["assertions"]["A1_all_under_holdout_train_images"] = {
        "root": root, "n_outside": len(bad), "examples": bad[:5]}
    if bad:
        fail.append(f"A1: {len(bad)} paths outside {root}")

    # ---- A2 no masks: directory, filename, raster -------------------------
    dir_bad = [p for p in pool
               if "masks" in os.path.normpath(p).split(os.sep)
               or "masks_multiclass" in os.path.normpath(p).split(os.sep)]
    name_bad = [p for p in pool if MASKISH.search(os.path.basename(p))]
    modes, raster_bad = {}, []
    for p in pool:
        mode, nch, size = raster_kind(p)
        modes[mode] = modes.get(mode, 0) + 1
        if nch != 3:
            raster_bad.append((p, mode, nch))
    rec["assertions"]["A2_no_masks"] = {
        "n_in_mask_dir": len(dir_bad), "n_maskish_filename": len(name_bad),
        "filename_pattern": MASKISH.pattern,
        "pil_modes": modes, "n_not_3channel": len(raster_bad),
        "not_3channel_examples": raster_bad[:5]}
    if dir_bad:
        fail.append(f"A2: {len(dir_bad)} files inside a mask directory")
    if name_bad:
        fail.append(f"A2: {len(name_bad)} mask-like filenames")
    if raster_bad:
        fail.append(f"A2: {len(raster_bad)} files are not 3-channel RGB")

    # ---- A3 no overlap with the holdout city's test and val ---------------
    donor_ids = {os.path.splitext(os.path.basename(p))[0] for p in pool}
    donor_hashes = {sha256_file(p): p for p in pool}
    if len(donor_hashes) != len(pool):
        rec["assertions"]["A3_donor_internal_dupes"] = len(pool) - len(donor_hashes)
    ov = {}
    for split in ("test", "val"):
        other = enumerate_pool(city_split_dir(city, split))
        oids = {os.path.splitext(os.path.basename(p))[0] for p in other}
        ohash = {sha256_file(p) for p in other}
        id_ov = sorted(donor_ids & oids)
        h_ov = sorted(set(donor_hashes) & ohash)
        ov[split] = {"n_in_split": len(other), "n_id_overlap": len(id_ov),
                     "id_overlap_examples": id_ov[:5],
                     "n_content_sha256_overlap": len(h_ov),
                     "content_overlap_examples":
                         [donor_hashes[h] for h in h_ov[:5]]}
        if id_ov:
            fail.append(f"A3: {len(id_ov)} donor ids also in {split}")
        if h_ov:
            fail.append(f"A3: {len(h_ov)} donor files byte-identical to a "
                        f"{split} file")
    rec["assertions"]["A3_no_test_or_val_overlap"] = ov

    # ---- A4 nothing from either source city -------------------------------
    src_hit = [p for p in pool
               if any(c in os.path.normpath(p).split(os.sep) for c in src)]
    src_id_hit = sorted(
        donor_ids & {os.path.splitext(os.path.basename(p))[0]
                     for c in src for s in ("train", "val", "test")
                     for p in enumerate_pool(city_split_dir(c, s))})
    rec["assertions"]["A4_no_source_city_files"] = {
        "source_cities": src, "n_path_hits": len(src_hit),
        "path_examples": src_hit[:5], "n_id_hits": len(src_id_hit),
        "id_examples": src_id_hit[:5]}
    if src_hit:
        fail.append(f"A4: {len(src_hit)} donor paths mention a source city")
    if src_id_hit:
        fail.append(f"A4: {len(src_id_hit)} donor ids also exist in a source city")

    # ---- A5 count ---------------------------------------------------------
    rec["assertions"]["A5_count"] = {"n_donor": len(pool),
                                     "expected": N_DONOR_EXPECT}
    if len(pool) != N_DONOR_EXPECT:
        fail.append(f"A5: n_donor {len(pool)} != {N_DONOR_EXPECT}")

    rec["donor_paths"] = pool
    rec["donor_basenames"] = [os.path.basename(p) for p in pool]
    rec["donor_list_sha256"] = list_sha(pool)
    rec["donor_basename_sha256"] = list_sha(sorted(rec["donor_basenames"]))
    rec["donor_content_sha256"] = hashlib.sha256(
        "\n".join(f"{sha256_file(p)}  {os.path.basename(p)}"
                  for p in pool).encode()).hexdigest()
    rec["verdict"] = "PASS" if not fail else "FAIL"
    if verbose:
        print(f"[donor {city}] n={len(pool)} verdict={rec['verdict']} "
              f"list_sha={rec['donor_list_sha256'][:16]} modes={modes}", flush=True)
        for f in fail:
            print(f"   FAILURE {f}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=OUT_DEFAULT)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    recs, allpass = {}, True
    for city in R.CITIES:
        rec = build(city)
        recs[city] = rec
        allpass = allpass and rec["verdict"] == "PASS"
    if not allpass:
        print("\nREFUSED: at least one donor list failed its assertions; "
              "nothing written.")
        return 1
    manifest = {"what": "OGLANet-SIB target-FDA donor lists (step 0b)",
                "written_utc": utc(), "definition_source":
                    "sib/arme12_retrain.py:114-115 clean_pool_root + "
                    ":122-139 enumerate_pool (= mamnet/data/fda_transform.py"
                    ":196-203 walk+filter)",
                "donor_split": "train", "donor_subdir": "images",
                "valid_ext": list(VALID_EXT), "cities": {}}
    for city, rec in recs.items():
        lp = os.path.join(a.out_dir, f"fda_donor_{city}_heldout_train.txt")
        body = "\n".join(rec["donor_paths"]) + "\n"
        if a.write:
            with open(lp, "w") as f:
                f.write(body)
            with open(lp + ".sha256", "w") as f:
                f.write(hashlib.sha256(body.encode()).hexdigest()
                        + f"  {os.path.basename(lp)}\n")
        manifest["cities"][city] = {
            "list_path": lp,
            "file_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "donor_root": rec["donor_root"], "n_donor": rec["n_donor"],
            "donor_list_sha256": rec["donor_list_sha256"],
            "donor_basename_sha256": rec["donor_basename_sha256"],
            "donor_content_sha256": rec["donor_content_sha256"],
            "source_cities": rec["source_cities"],
            "verdict": rec["verdict"],
            "assertions": rec["assertions"]}
    mp = os.path.join(a.out_dir, "fda_donor_manifest.json")
    if a.write:
        with open(mp, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"\nwrote {mp}")
        for city in R.CITIES:
            print(f"  {manifest['cities'][city]['list_path']}  "
                  f"file_sha256={manifest['cities'][city]['file_sha256']}")
    else:
        print("\n(dry: --write not given)")
        print(json.dumps({c: {k: v for k, v in m.items() if k != "assertions"}
                          for c, m in manifest["cities"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
