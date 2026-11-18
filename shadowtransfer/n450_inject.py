"""
n450 retrain -- feed a published trainer the SHARED 450-image training list.

The OGLANet and DINOv3 standard cells draw 225 images per source city through
`samples_per_dir = 450 // 2` under random_seed=42. The trainers retrained here
(MAMNet's seven, OGLANet SegDeSiC, OGLANet SIB) build their training set with
no subsampling and load all 900. The operator's ruling: feed them the shared
list; do NOT re-seed anything until it happens to match.

Mechanism. After importing the architecture tree, each dataset class's
__init__ is wrapped. The original __init__ runs unchanged; afterwards a dataset
whose split is 'train' and whose roots are exactly source-city roots has its
three parallel per-sample lists (img_files, img_paths, mask_paths -- the only
per-sample state those classes hold, read 2026-10-26) filtered to the shared
list. Nothing else is touched:
  * val / test datasets are recorded, never filtered;
  * a train-split dataset rooted at the HELD-OUT city (SegDeSiC's unlabeled
    target set) is recorded as target access and left as built;
  * any other root raises.
`assert_counts()` is called after the trainer builds its loaders and BEFORE
the first optimizer step, so a wrong count costs seconds, not a training run.
"""

import hashlib
import json
import os

RECORD = []          # one dict per dataset the trainer constructed
FDA_LOADS = []       # every load_target_images(root) call observed
_STATE = {}


def _norm(p):
    return os.path.normpath(str(p))


def list_sha(names):
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def configure(shared_lists_path, holdout, source_cities, data_root, res_dir):
    with open(shared_lists_path) as f:
        shared = json.load(f)
    entry = shared["lists"][holdout]
    per_city = {c: list(entry["per_city"][c]) for c in source_cities}
    allowed = {}
    for c, names in per_city.items():
        for n in names:
            if n in allowed:
                raise RuntimeError(f"name {n} listed for two cities")
            allowed[n] = c
    _STATE.update(
        holdout=holdout, per_city=per_city, allowed=allowed,
        expected_sha=entry["sha256"],
        src_roots={_norm(os.path.join(data_root, c, res_dir)): c
                   for c in source_cities},
        holdout_root=_norm(os.path.join(data_root, holdout, res_dir)),
        shared_path=shared_lists_path,
    )
    if list_sha(allowed) != entry["sha256"]:
        raise RuntimeError("shared list file is internally inconsistent")
    return _STATE


def _roots_of(ds):
    r = getattr(ds, "root_dirs", None)
    if r is None:
        r = getattr(ds, "root_dir", None)
    if isinstance(r, (str, os.PathLike)):
        r = [r]
    return [_norm(x) for x in (r or [])]


def _after_init(ds, cls_name):
    split = getattr(ds, "split", None)
    roots = _roots_of(ds)
    n0 = len(ds.img_files)
    if not (len(ds.img_paths) == n0 == len(ds.mask_paths)):
        raise RuntimeError(f"{cls_name}: parallel lists disagree before filter")
    rec = {"cls": cls_name, "split": split, "roots": roots, "n_built": n0}
    src = _STATE["src_roots"]
    if split == "train" and roots and all(r in src for r in roots):
        cities = {src[r] for r in roots}
        keep = [i for i, f in enumerate(ds.img_files)
                if _STATE["allowed"].get(f) in cities]
        want = {n for n, c in _STATE["allowed"].items() if c in cities}
        got = [ds.img_files[i] for i in keep]
        if len(got) != len(set(got)) or set(got) != want:
            missing = sorted(want - set(got))[:5]
            raise RuntimeError(f"{cls_name}: shared list not reproducible from "
                               f"roots {roots}; missing e.g. {missing}")
        for i in keep:
            d = _norm(os.path.dirname(os.path.dirname(os.path.dirname(
                ds.img_paths[i]))))
            if src.get(d) != _STATE["allowed"][ds.img_files[i]]:
                raise RuntimeError(f"{cls_name}: {ds.img_paths[i]} is not under "
                                   f"its listed city's root")
        ds.img_files = [ds.img_files[i] for i in keep]
        ds.img_paths = [ds.img_paths[i] for i in keep]
        ds.mask_paths = [ds.mask_paths[i] for i in keep]
        rec.update(role="source_train", filtered=True, n_after=len(keep),
                   cities=sorted(cities), names=sorted(got))
    elif split == "train" and roots == [_STATE["holdout_root"]]:
        rec.update(role="holdout_train_unlabeled", filtered=False, n_after=n0)
    elif split in ("val", "test"):
        role = "test" if split == "test" else "val"
        if split == "test" and roots != [_STATE["holdout_root"]]:
            raise RuntimeError(f"{cls_name}: test set not the held-out city: {roots}")
        if split == "val" and not all(r in src for r in roots):
            raise RuntimeError(f"{cls_name}: val set not source-city: {roots}")
        rec.update(role=role, filtered=False, n_after=n0,
                   names=sorted(ds.img_files))
    else:
        raise RuntimeError(f"{cls_name}: unexpected dataset split={split} "
                           f"roots={roots}")
    RECORD.append(rec)
    print(f"[n450_inject] {cls_name} split={split} role={rec['role']} "
          f"n_built={n0} n_after={rec['n_after']}", flush=True)


def wrap_class(cls, label):
    orig = cls.__init__
    if getattr(orig, "_n450_wrapped", False):
        return

    def __init__(self, *a, **k):
        orig(self, *a, **k)
        if not getattr(self, "_n450_seen", False):   # once per instance
            self._n450_seen = True
            _after_init(self, label)

    __init__._n450_wrapped = True
    __init__._n450_orig = orig
    cls.__init__ = __init__


def unwrap_class(cls):
    orig = getattr(cls.__init__, "_n450_orig", None)
    if orig is not None:
        cls.__init__ = orig


def wrap_fda_loader(fda_module):
    """Record every FDA donor-pool load; behaviour unchanged."""
    orig = fda_module.load_target_images
    if getattr(orig, "_n450_wrapped", False):
        return

    def load_target_images(root, *a, **k):
        FDA_LOADS.append({"root": _norm(root), "args": [repr(x) for x in a],
                          "kwargs": {kk: repr(v) for kk, v in k.items()}})
        print(f"[n450_inject] FDA donor load from {root}", flush=True)
        return orig(root, *a, **k)

    load_target_images._n450_wrapped = True
    fda_module.load_target_images = load_target_images


def summary():
    tr = [r for r in RECORD if r["role"] == "source_train"]
    va = [r for r in RECORD if r["role"] == "val"]
    te = [r for r in RECORD if r["role"] == "test"]
    tg = [r for r in RECORD if r["role"] == "holdout_train_unlabeled"]
    names = sorted(n for r in tr for n in r["names"])
    per_city = {}
    for r in tr:
        for n in r["names"]:
            c = _STATE["allowed"][n]
            per_city[c] = per_city.get(c, 0) + 1
    return {
        "n_train": len(names), "n_train_per_source_city": per_city,
        "train_list_sha256": list_sha(names),
        "expected_train_list_sha256": _STATE.get("expected_sha"),
        "n_val": sum(r["n_after"] for r in va),
        "val_list_sha256": list_sha([n for r in va for n in r["names"]]),
        "n_test": sum(r["n_after"] for r in te),
        "test_list_sha256": list_sha([n for r in te for n in r["names"]]),
        "n_holdout_unlabeled": sum(r["n_after"] for r in tg),
        "fda_loads": list(FDA_LOADS),
        "datasets": [{k: v for k, v in r.items() if k != "names"} for r in RECORD],
        "shared_lists_path": _STATE.get("shared_path"),
    }


def assert_counts(n_train=450, per_city=225, n_val=300, n_test=150):
    s = summary()
    bad = []
    if s["n_train"] != n_train:
        bad.append(f"n_train {s['n_train']} != {n_train}")
    if any(v != per_city for v in s["n_train_per_source_city"].values()) or \
            len(s["n_train_per_source_city"]) != 2:
        bad.append(f"per-city {s['n_train_per_source_city']} != {per_city} x 2")
    if s["train_list_sha256"] != s["expected_train_list_sha256"]:
        bad.append("train list sha != shared list sha")
    if s["n_val"] != n_val:
        bad.append(f"n_val {s['n_val']} != {n_val}")
    if n_test is not None and s["n_test"] != n_test:
        bad.append(f"n_test {s['n_test']} != {n_test}")
    if bad:
        raise RuntimeError("n450 COUNT ASSERTION FAILED: " + "; ".join(bad))
    print(f"[n450_inject] counts OK: train {s['n_train']} "
          f"{s['n_train_per_source_city']} sha {s['train_list_sha256'][:12]} | "
          f"val {s['n_val']} | test {s['n_test']} | holdout-unlabeled "
          f"{s['n_holdout_unlabeled']} | fda loads {len(s['fda_loads'])}",
          flush=True)
    return s
