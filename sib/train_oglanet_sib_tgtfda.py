#!/usr/bin/env python3
"""OGLANet-SIB target-FDA pass-through -- the smallest change that gives the
SIB CNN branch the FDA style pool the paper describes.

WHY A DATASET OBJECT AND NOT `fda_target_root`
----------------------------------------------
`train_oglanet_sib.py:150` declares `--fda_target_root`, and nothing reads it:
neither `get_dataloaders_sib` call (`:1512-1523`, `:1524-1535`) passes it, and
`oglanet/data/dataset_sib.py:get_dataloaders_sib` (`:241-252`) has no such
parameter to receive it. So "set fda_target_root" is not a mechanism here --
it is an inert string. OGLANet's donor is a DATASET OBJECT:

    dataset_sib.py:291-294   if use_fda and len(train_datasets) >= 2:
                                 train_datasets[0].fda_target_dataset = train_datasets[1]
                                 train_datasets[1].fda_target_dataset = train_datasets[0]
    dataset_sib.py:200-202   tgt_idx  = np.random.randint(len(self.fda_target_dataset))
                             tgt_path = self.fda_target_dataset.img_paths[tgt_idx]

i.e. each SOURCE city's train set is the other SOURCE city's style donor --
the cross-source fallback this retrain exists to replace. The only thing read
off the donor is `len()` and `img_paths`.

THE ONE CHANGE
--------------
After `get_dataloaders_sib` returns, both source train datasets'
`fda_target_dataset` is rebound to a `ShadowDatasetSIB` built on the HELD-OUT
city's `train/images` -- the pool `arme12_retrain.clean_pool_root` pins for the
clean MAMNet-SIB retrains, and the pool paper S5.1 / App. F describe. Nothing
else is touched: not the loaders, not the splits, not `fda_L`, not the model,
not the schedule. `ns.fda_target_root` is also set to the donor root so the
config record states the intent, but it remains inert in the trainer.

The DRAW RULE stays OGLANet's own: a fresh donor per training sample, uniform
over the whole 450-image list. MAMNet instead draws 100 files once inside
`load_target_images`. Imposing MAMNet's sampler on OGLANet would be a second
variable, so it is not imposed.

`train_oglanet_sib.py`, `dataset_sib.py` and `dffm.py` are not edited. This
module rebinds attributes in-process only.
"""
import hashlib
import os
import re

MASKISH = re.compile(r"mask|label|gt|seg|annot", re.I)
VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
N_DONOR_EXPECT = 450


def list_sha(lines):
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def donor_city_root(base_data_root, city, resolution):
    """Exactly how dataset_sib.get_dataloaders_sib:282 spells a city root, so
    n450_inject recognises the donor as the held-out city's train split."""
    return os.path.join(base_data_root, city, resolution)


def split_dir(base_data_root, city, resolution, split, sub="images"):
    return os.path.join(base_data_root, city, resolution, split, sub)


def assert_donor(ds, city, base_data_root, resolution, expected_basename_sha,
                 content_check=True, verbose=True):
    """Step-0b assertions re-run on the donor list the DATASET ACTUALLY HOLDS.

    Raises RuntimeError on any failure. Returns the record for run.json.
    """
    paths = list(ds.img_paths)
    names = [os.path.basename(p) for p in paths]
    real = [os.path.realpath(p) for p in paths]
    want_dir = os.path.realpath(split_dir(base_data_root, city, resolution,
                                          "train"))
    fail, rec = [], {"holdout": city, "n_donor": len(paths),
                     "donor_dir_realpath": want_dir,
                     "donor_basename_sha256": list_sha(sorted(names)),
                     "expected_basename_sha256": expected_basename_sha,
                     "dataset_class": type(ds).__name__,
                     "dataset_len": len(ds),
                     "content_check": content_check}

    # A5 count
    if len(paths) != N_DONOR_EXPECT:
        fail.append(f"count {len(paths)} != {N_DONOR_EXPECT}")
    if len(ds) != len(paths):
        fail.append(f"len(dataset) {len(ds)} != len(img_paths) {len(paths)}")

    # A1 directory
    outside = [p for p, r in zip(paths, real)
               if os.path.dirname(r) != want_dir]
    rec["n_outside_donor_dir"] = len(outside)
    if outside:
        fail.append(f"{len(outside)} donor paths not in {want_dir} "
                    f"(e.g. {outside[:3]})")

    # A2 no masks -- directory component, filename, raster
    from PIL import Image
    in_mask_dir = [r for r in real
                   if {"masks", "masks_multiclass"} & set(r.split(os.sep))]
    maskish = [n for n in names if MASKISH.search(n)]
    modes, not_rgb = {}, []
    for p in paths:
        with Image.open(p) as im:
            m = im.mode
        modes[m] = modes.get(m, 0) + 1
        if m != "RGB":
            not_rgb.append((p, m))
    rec.update(n_in_mask_dir=len(in_mask_dir), n_maskish_name=len(maskish),
               pil_modes=modes, n_not_rgb=len(not_rgb))
    if in_mask_dir:
        fail.append(f"{len(in_mask_dir)} donor paths inside a mask directory")
    if maskish:
        fail.append(f"{len(maskish)} mask-like donor filenames: {maskish[:3]}")
    if not_rgb:
        fail.append(f"{len(not_rgb)} donor files are not 3-channel RGB: "
                    f"{not_rgb[:3]}")

    # A3 no overlap with the held-out city's test and val
    ids = {os.path.splitext(n)[0] for n in names}
    hashes = {_sha_file(p) for p in paths} if content_check else None
    ov = {}
    for sp in ("test", "val"):
        d = split_dir(base_data_root, city, resolution, sp)
        other = sorted(f for f in os.listdir(d)
                       if f.lower().endswith(VALID_EXT))
        oids = {os.path.splitext(f)[0] for f in other}
        id_ov = sorted(ids & oids)
        h_ov = []
        if content_check:
            oh = {_sha_file(os.path.join(d, f)) for f in other}
            h_ov = sorted(hashes & oh)
        ov[sp] = {"n_in_split": len(other), "n_id_overlap": len(id_ov),
                  "n_content_overlap": len(h_ov),
                  "examples": id_ov[:3] or h_ov[:3]}
        if id_ov:
            fail.append(f"{len(id_ov)} donor ids also in {city}/{sp}")
        if h_ov:
            fail.append(f"{len(h_ov)} donor files byte-identical to a "
                        f"{city}/{sp} file")
    rec["no_test_val_overlap"] = ov

    # A4 nothing from a source city
    src_hit = [r for r in real if city not in r.split(os.sep)]
    rec["n_not_under_holdout_city"] = len(src_hit)
    if src_hit:
        fail.append(f"{len(src_hit)} donor paths not under city {city}: "
                    f"{src_hit[:3]}")

    # the list is the one step 0b wrote
    if expected_basename_sha and \
            rec["donor_basename_sha256"] != expected_basename_sha:
        fail.append(f"donor basename sha {rec['donor_basename_sha256'][:12]} "
                    f"!= step-0b {expected_basename_sha[:12]}")

    rec["verdict"] = "PASS" if not fail else "FAIL"
    rec["failures"] = fail
    if verbose:
        print(f"[tgtfda] donor assertions {city}: {rec['verdict']}  "
              f"n={len(paths)} modes={modes} "
              f"sha={rec['donor_basename_sha256'][:16]} "
              f"dir={want_dir}", flush=True)
        for sp, v in ov.items():
            print(f"[tgtfda]   no overlap with {city}/{sp}: "
                  f"ids {v['n_id_overlap']}/{v['n_in_split']}, "
                  f"content {v['n_content_overlap']}/{v['n_in_split']}",
                  flush=True)
    if fail:
        raise RuntimeError("[tgtfda] DONOR ASSERTION FAILED for "
                           f"{city}: " + "; ".join(fail))
    return rec


def install(mod, ns, city, expected_basename_sha=None, content_check=True):
    """Wrap mod.get_dataloaders_sib so the FDA donor is the held-out city's
    train split. Call BEFORE n450_sib.install, so that module's count
    assertion runs after the donor dataset has been constructed and recorded.

    Returns a dict filled in once the loaders are built (STATE["donor"]).
    """
    state = {"installed_utc": None, "donor": None}
    orig = mod.get_dataloaders_sib
    if getattr(orig, "_tgtfda_wrapped", False):
        raise RuntimeError("[tgtfda] already installed on this module")
    donor_root = donor_city_root(ns.base_data_root, city, ns.resolution)
    ns.fda_target_root = split_dir(ns.base_data_root, city, ns.resolution,
                                   "train")

    def get_dataloaders_sib(*a, **k):
        data = orig(*a, **k)
        tr = data["train_dataset"]
        subs = list(getattr(tr, "datasets", [tr]))
        if len(subs) != 2:
            raise RuntimeError(f"[tgtfda] expected 2 source train datasets, "
                               f"got {len(subs)}")
        before = []
        for d in subs:
            tgt = getattr(d, "fda_target_dataset", None)
            if tgt is None:
                raise RuntimeError(
                    "[tgtfda] no cross-source donor was set: use_fda is false "
                    "in this config, so there is no FDA branch to redirect")
            before.append({"city_id": getattr(d, "city_id", None),
                           "donor_city_id": getattr(tgt, "city_id", None),
                           "donor_n": len(tgt.img_paths),
                           "donor_dir": os.path.dirname(
                               os.path.realpath(tgt.img_paths[0]))})
        cls = type(subs[0])
        donor = cls(donor_root, split="train", img_size=ns.img_size,
                    use_contrast=ns.use_contrast, fda_L=ns.fda_L)
        rec = assert_donor(donor, city, ns.base_data_root, ns.resolution,
                           expected_basename_sha, content_check=content_check)
        for d in subs:
            d.fda_target_dataset = donor
        after = [{"city_id": getattr(d, "city_id", None),
                  "donor_is_the_same_object": d.fda_target_dataset is donor,
                  "donor_n": len(d.fda_target_dataset.img_paths)}
                 for d in subs]
        if not all(x["donor_is_the_same_object"] for x in after):
            raise RuntimeError("[tgtfda] donor rebind did not take")
        rec.update(donor_root=donor_root,
                   fda_L=ns.fda_L, use_fda=bool(ns.use_fda),
                   fda_target_root_recorded=ns.fda_target_root,
                   fda_target_root_is_read_by_trainer=False,
                   draw_rule="per training sample, np.random.randint over the "
                             "whole donor list (dataset_sib.py:200-202); "
                             "OGLANet's own rule, unchanged",
                   replaced_cross_source_donors=before, after=after)
        state["donor"] = rec
        print(f"[tgtfda] FDA donor rebound: both source train sets -> "
              f"{donor_root} (n={len(donor.img_paths)}); replaced "
              f"{[b['donor_dir'] for b in before]}", flush=True)
        return data

    get_dataloaders_sib._tgtfda_wrapped = True
    get_dataloaders_sib._tgtfda_orig = orig
    mod.get_dataloaders_sib = get_dataloaders_sib
    return state
