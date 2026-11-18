"""
n450 retrain -- the 27-cell roster (operator request 2026-10-26).

Retrain 27 internal-dataset LOCO cells at a matched 450-image training budget
(225 per source city, the OGLANet/DINOv3 sampler draw under seed 42), with
validation at 300 and the published test set. Source of the list:
runs/train_count_audit/retrain_at_450_list.json (session log
2026-10-25-train-count-audit-sar-drift.md). stdlib only, so the submit script,
the gate and the worker read one definition of the grid.

Groups, in the operator's submission priority order:
  1  MAMNet standard (vanilla) LOCO          3 cells   mamnet/train.py
  2  OGLANet SIB (C4clean)                   3 cells   oglanet/train_oglanet_sib.py
  3  OGLANet SegDeSiC                        3 cells   oglanet/train_segdesic.py
  4  MAMNet fda/segdesic/iim/isw/mrfp_plus/fada  18    six MAMNet trainers
"""

import os

BASE = os.environ["PROJECT_ROOT"]
SCRATCH = os.environ["ST_SCRATCH_ROOT"]
SHADE = SCRATCH + "/ShadeMaps"
N450_ROOT = SHADE + "/n450"                      # checks, logs, smoke dumps
DUMP_ROOT = SHADE + "/dumps/st_r450_published_recompute"   # next to originals
SHARED_LISTS = N450_ROOT + "/checks/shared_train_lists.json"

CITIES = ("chicago", "miami", "phoenix")
FOLD_OF_HOLDOUT = {"phoenix": 0, "miami": 1, "chicago": 2}
# data/dataset.py LOCO_FOLDS, byte-identical in all three trees
LOCO_TRAIN = {0: ["chicago", "miami"], 1: ["chicago", "phoenix"],
              2: ["miami", "phoenix"]}
RES, RES_DIR = "hr", "highres"
SEED = 42
N_TRAIN, N_PER_CITY, N_VAL, N_TEST = 450, 225, 300, 150
DATA_ROOT = SHADE + "/data/Final_data_test"

MAMNET_METHODS = ("fda", "segdesic", "iim", "isw", "mrfp_plus", "fada")

# (arch, method) -> (trainer module, trainer class or None for main()-style)
TRAINER = {
    ("mamnet", "vanilla"): ("train", "Trainer"),
    ("mamnet", "fda"): ("train", "Trainer"),
    ("mamnet", "segdesic"): ("train_segdesic", "SegDesicTrainer"),
    ("mamnet", "iim"): ("train_iim", "Trainer"),
    ("mamnet", "isw"): ("train_isw", "TrainerISW"),
    ("mamnet", "mrfp_plus"): ("train_mrfp", "Trainer"),
    ("mamnet", "fada"): ("train_fada", "TrainerFADA"),
    ("oglanet", "segdesic"): ("train_segdesic", "SegDesicTrainer"),
    ("oglanet", "sibc4"): ("train_oglanet_sib", None),
}

# What each method's code does with held-out-city data, from the code paths
# read 2026-10-26 (and the 2026-08-03 OGLANet-SIB access verdict). The worker
# also OBSERVES every dataset its trainer builds and every FDA donor load, and
# refuses to write a dump if observation and this table disagree.
TARGET_ACCESS = {
    "vanilla": ("source_only",
                "train/val built from the two source cities only; use_fda=false"),
    "fda": ("unlabeled_target",
            "FDA amplitude donors: load_target_images(fda_target_root, "
            "max_images=100) on the held-out city's train/images"),
    "segdesic": ("unlabeled_target",
                 "loader builds an unlabeled target set on the held-out city's "
                 "train split for the geo domain loss (no labels consumed)"),
    "iim": ("source_only", "no held-out-city data in the trainer's loaders"),
    "isw": ("source_only", "no held-out-city data; ISW masks were precomputed "
                           "from the source cities' train images"),
    "mrfp_plus": ("source_only", "no held-out-city data in the trainer's loaders"),
    "fada": ("source_only", "no held-out-city data in the trainer's loaders"),
    "sibc4": ("source_only",
              "FDA cross-mixes the two SOURCE cities (get_dataloaders_sib sets "
              "each source train set as the other's donor); fda_target_root is "
              "declared and never dereferenced (2026-08-03 access verdict, 0/111 "
              "logs). Preserved, not 'fixed'."),
}

# run_inference_probs.py --model_variant token, = the Test_img_probs dir name
PROB_VARIANT = {m: m for m in ("vanilla",) + MAMNET_METHODS}


def legacy_run_dir(arch, method, city):
    if method == "vanilla":
        return f"{arch}_loco_holdout_{city}_{RES_DIR}_1"
    if method == "sibc4":
        return (f"oglanet_sib_C4clean_haar_vib_sag_fda_ctr__"
                f"loco_holdout_{city}__{RES_DIR}")
    return f"{arch}_{method}_loco_holdout_{city}_{RES_DIR}_1"


def orig_run_id(arch, method, city):
    return f"{arch}-{method}-loco-r450-{city}-{RES}-s0"


def new_run_id(arch, method, city):
    return f"{arch}-{method}-loco-r450-{city}-{RES}-s{SEED}-n450"


def source_cities(city):
    return list(LOCO_TRAIN[FOLD_OF_HOLDOUT[city]])


def _cell(group, arch, method, city):
    mod, cls = TRAINER[(arch, method)]
    rd = legacy_run_dir(arch, method, city)
    ta, tad = TARGET_ACCESS[method]
    return {
        "group": group, "arch": arch, "method": method, "city": city,
        "run_id": new_run_id(arch, method, city),
        "orig_run_id": orig_run_id(arch, method, city),
        "legacy_run_dir": rd,
        "orig_dir": os.path.join(SHADE, "data", arch, "outputs", rd),
        "config_source": ("train.log:Args" if method == "sibc4" else "args.json"),
        "ckpt_args_available": method == "vanilla",
        "trainer_module": mod, "trainer_class": cls,
        "trainer_path": f"python/{arch}/{mod}.py",
        "prob_variant": PROB_VARIANT.get(method),
        "target_access": ta, "target_access_detail": tad,
        "source_cities": source_cities(city),
        "fold_id": FOLD_OF_HOLDOUT[city],
        "ckpt_dir": os.path.join(SHADE, "data", arch, "outputs", "n450",
                                 new_run_id(arch, method, city)),
        "dump_dir": os.path.join(DUMP_ROOT, new_run_id(arch, method, city)),
        "orig_dump_dir": os.path.join(DUMP_ROOT, orig_run_id(arch, method, city)),
    }


def roster():
    out = []
    for c in CITIES:
        out.append(_cell(1, "mamnet", "vanilla", c))
    for c in CITIES:
        out.append(_cell(2, "oglanet", "sibc4", c))
    for c in CITIES:
        out.append(_cell(3, "oglanet", "segdesic", c))
    for m in MAMNET_METHODS:
        for c in CITIES:
            out.append(_cell(4, "mamnet", m, c))
    return out


def by_run_id():
    return {c["run_id"]: c for c in roster()}


# Stage design per group. Group 4 runs six different trainers, so one smoke
# cannot gate the other five: it gets one smoke PER TRAINER, then one real cell.
SMOKE_CITY = "phoenix"
REAL_CELL = {
    1: new_run_id("mamnet", "vanilla", "phoenix"),
    2: new_run_id("oglanet", "sibc4", "phoenix"),
    3: new_run_id("oglanet", "segdesic", "phoenix"),
    4: new_run_id("mamnet", "segdesic", "phoenix"),
}


def smoke_cells(group):
    return [c for c in roster()
            if c["group"] == group and c["city"] == SMOKE_CITY]


def smoke_run_id(cell):
    return cell["run_id"] + "-smoke"


def smoke_dir(cell):
    return os.path.join(N450_ROOT, "smoke", smoke_run_id(cell))


def role_of(run_id):
    """'real' or 'fan' for a roster run_id."""
    c = by_run_id()[run_id]
    return "real" if REAL_CELL[c["group"]] == run_id else "fan"


def job_name(role, cell):
    rid = smoke_run_id(cell) if role == "smoke" else cell["run_id"]
    return f"n450-g{cell['group']}-{role}-{rid}"


if __name__ == "__main__":
    import sys
    if "--list" in sys.argv:
        for c in roster():
            print(c["run_id"], c["group"], c["arch"], c["method"], c["city"])
    else:
        print(len(roster()))
