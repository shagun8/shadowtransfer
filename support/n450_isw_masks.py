#!/usr/bin/env python3
"""
n450 -- rebuild the MAMNet ISW whitening masks from the SHARED 450-image list
(operator ruling 2026-10-26, decision 2), for the three MAMNet ISW retrains.

The published masks (data/mamnet/isw_masks/loco_holdout_{city}_highres) were
built from the fold's full source train split: metadata num_images_used 900.
OGLANet's and DINOv3's published masks record 450 (their computes route
through get_dataloaders' seed-42 sampler) and are not touched.

THE PUBLISHED COMPUTE IS NOT EDITED. mamnet/compute_isw_masks.py's
precompute_masks(args) is imported and called with the published settings
(read from the published metadata.json: layers feat1,feat2,feat3, kmeans_k 3,
kmeans_m 1, use_contrast true, img_size 384, mode loco, checkpoint null =
untrained encoder, num_samples 0 = all). Its function-local
`from data.dataset_enhanced import ShadowDatasetEnhanced` resolves to the class
n450_inject has wrapped, so the dataset it builds over the two source roots is
filtered to the shared list -- the same hook the retrain uses.

Declared differences from the published run: the image set (450 shared, not
900), the output directory, and seeds (python/numpy/torch = 42; the published
run seeded nothing, so its photometric-augmentation draw is unrecoverable).

  python n450_isw_masks.py --city phoenix [--num_samples N  (smoke only)]
Output: ${ST_SCRATCH_ROOT}/data/mamnet/isw_masks_n450/loco_holdout_{city}_highres/
"""
import argparse
import json
import os
import random
import sys
import time

BASE = os.environ["PROJECT_ROOT"]
sys.path.insert(0, str(BASE))

# release layout: the drivers live in sibling directories rather than one flat
# tree, so every sibling that exports an importable module goes on sys.path.
for _d in ("appendix_h", "support", "matched450", "sar", "bsm", "scoring", "probes", "crossearth"):
    _p = str(BASE / _d) if hasattr(BASE, "__truediv__") else os.path.join(BASE, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from shadowtransfer import n450_roster as R              # noqa: E402
from shadowtransfer import n450_inject as INJ            # noqa: E402

PUB = os.path.join(BASE, "data/mamnet/isw_masks/loco_holdout_{city}_highres")
OUT = os.path.join(R.SHADE, "data/mamnet/isw_masks_n450/loco_holdout_{city}_highres")


def out_dir(city, smoke=False):
    return OUT.format(city=city) + ("_smoke" if smoke else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=list(R.CITIES))
    ap.add_argument("--num_samples", type=int, default=0)
    a = ap.parse_args()
    smoke = a.num_samples > 0
    t0 = time.time()
    cell = R.by_run_id()[R.new_run_id("mamnet", "isw", a.city)]
    with open(os.path.join(cell["orig_dir"], "args.json")) as f:
        cell_cfg = json.load(f)
    with open(os.path.join(PUB.format(city=a.city), "metadata.json")) as f:
        pub = json.load(f)
    od = out_dir(a.city, smoke)
    if os.path.exists(os.path.join(od, "metadata.json")):
        print(f"SKIP: {od}/metadata.json exists")
        return 0
    tree = os.path.join(BASE, "mamnet")
    sys.path.insert(0, tree)
    os.chdir(tree)
    INJ.configure(R.SHARED_LISTS, a.city, cell["source_cities"],
                  cell_cfg["base_data_root"], cell_cfg["resolution"])
    import data.dataset_enhanced as DE
    INJ.wrap_class(DE.ShadowDatasetEnhanced, "ShadowDatasetEnhanced")
    import numpy as np
    import torch
    random.seed(R.SEED)
    np.random.seed(R.SEED)
    torch.manual_seed(R.SEED)
    torch.cuda.manual_seed_all(R.SEED)
    import compute_isw_masks as CIM
    args = argparse.Namespace(
        data_root=None, base_data_root=cell_cfg["base_data_root"], mode=pub["mode"],
        cities=None, resolution=cell_cfg["resolution"], fold_id=cell["fold_id"],
        img_size=pub["img_size"], num_workers=2, num_samples=a.num_samples,
        num_classes=2, use_contrast=pub["use_contrast"], checkpoint=pub["checkpoint"],
        layers=",".join(pub["layers"]), kmeans_k=pub["kmeans_k"],
        kmeans_m=pub["kmeans_m"], output_dir=od, device="cuda")
    print("[n450_isw] args", vars(args), flush=True)
    CIM.precompute_masks(args)
    s = INJ.summary()
    tr = [d for d in s["datasets"] if d["role"] == "source_train"]
    with open(os.path.join(od, "metadata.json")) as f:
        meta = json.load(f)
    want = a.num_samples if smoke else 450
    bad = []
    if len(tr) != 1 or s["n_train"] != 450:
        bad.append(f"source-train datasets {len(tr)}, n {s['n_train']}")
    if s["train_list_sha256"] != s["expected_train_list_sha256"]:
        bad.append("image list sha != shared")
    if meta["num_images_used"] != want:
        bad.append(f"num_images_used {meta['num_images_used']} != {want}")
    if any(d["role"] not in ("source_train",) for d in s["datasets"]):
        bad.append(f"unexpected datasets {[d['role'] for d in s['datasets']]}")
    prov = {"city": a.city, "smoke": smoke, "shared_list_sha256":
            s["train_list_sha256"], "n_images_listed": s["n_train"],
            "num_images_used": meta["num_images_used"],
            "published_masks": PUB.format(city=a.city),
            "published_num_images_used": pub["num_images_used"],
            "seeds": {"python": R.SEED, "numpy": R.SEED, "torch": R.SEED,
                      "note": "published run seeded nothing"},
            "fraction_selected": {k: v.get("fraction_selected")
                                  for k, v in meta.get("masks", {}).items()},
            "compute": "mamnet/compute_isw_masks.py precompute_masks, unedited",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "wall_s": time.time() - t0, "problems": bad}
    for k, fr in prov["fraction_selected"].items():
        if fr is None or not (0.0 < fr < 1.0):
            bad.append(f"{k} fraction_selected {fr} not in (0,1)")
    with open(os.path.join(od, "n450_provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps(prov, indent=1), flush=True)
    if bad:
        os.replace(os.path.join(od, "metadata.json"),
                   os.path.join(od, "metadata.json.REJECTED"))
        raise SystemExit("FATAL " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
