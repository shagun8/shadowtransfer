"""
Materialize LOCO splits from a per-city dataset.

For each fold and resolution this script copies (by default) the train +
val pools from the two training cities and the full test pool from the
held-out city into a single self-contained directory tree. After it runs
a LOCO experiment becomes equivalent to:

    --mode single --data_root <output_root>/fold_<id>_holdout_<city>/<resolution>

Source layout expected
----------------------
    base_data_root/
        chicago/
            highres/
                {train,val,test}/
                    images/
                    masks/
                    masks_multiclass/      # optional, brought along when present
                metadata_train.json        # optional, merged into LOCO metadata
                metadata_val.json
                metadata_test.json
            midres/...
        miami/...
        phoenix/...

Output layout
-------------
    output_root/
        fold_0_holdout_phoenix/
            highres/
                manifest.json
                metadata_train.json
                metadata_val.json
                metadata_test.json
                {train,val,test}/
                    images/
                    masks/
                    masks_multiclass/      # only where it exists upstream
            midres/
                ...
        fold_1_holdout_miami/...
        fold_2_holdout_chicago/...

Filename convention
-------------------
* train/ and val/ files are renamed `{source_city}__{filename}` so the
  two source cities can't collide and so any reader can see at a glance
  which city a sample came from.
* test/ files keep their original names (single source city).

Metadata
--------
Each fold/resolution gets its own `metadata_{split}.json` whose entries
mirror the source metadata one-for-one and add LOCO context:

    loco_filename            : str   filename as stored in the LOCO tree
    loco_split               : str   "train" / "val" / "test" in this fold
    loco_fold_id             : int
    loco_holdout_city        : str
    loco_resolution          : str
    source_city              : str   the city the sample originated from
    source_split             : str   the split it came from upstream
    has_masks_multiclass     : bool  whether a multiclass mask was copied

Usage
-----
    python build_loco_splits.py \\
        --base_data_root /path/to/Final_data_test \\
        --output_root    /path/to/Final_data_loco \\
        --resolutions    highres midres \\
        --folds          0 1 2 \\
        --mode           copy

Defaults match the paper: 225 train + 75 val per training city, full
test pool from the held-out city. Pass `-1` to either count to take all
available files.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


LOCO_FOLDS = {
    0: {'train': ['chicago', 'miami'],   'test': 'phoenix'},
    1: {'train': ['chicago', 'phoenix'], 'test': 'miami'},
    2: {'train': ['miami',   'phoenix'], 'test': 'chicago'},
}

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


# ----------------------------------------------------------------------
# File-system helpers
# ----------------------------------------------------------------------

def list_split(city_root: Path, split: str, kind: str = 'images'):
    """Return sorted filenames in city_root/split/{kind}/."""
    d = city_root / split / kind
    if not d.exists():
        return []
    return sorted(f.name for f in d.iterdir()
                  if f.suffix.lower() in IMG_EXTS)


def transfer(src: Path, dst: Path, mode: str):
    """Place `src` at `dst` according to `mode`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'symlink':
        dst.symlink_to(src.resolve())
    elif mode == 'hardlink':
        os.link(src, dst)
    else:
        raise ValueError(f"Unknown transfer mode: {mode}")


def collect_paired_files(city_root: Path, split: str):
    """Files that have BOTH images/ and masks/ entries in the given split."""
    imgs = list_split(city_root, split, 'images')
    msks = set(list_split(city_root, split, 'masks'))
    return [f for f in imgs if f in msks]


# ----------------------------------------------------------------------
# Metadata helpers
# ----------------------------------------------------------------------

def load_split_metadata(city_root: Path, split: str):
    """Load metadata_{split}.json if present, else []."""
    p = city_root / f'metadata_{split}.json'
    if not p.exists():
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"    WARN: {p} is not a JSON list; ignoring.")
            return []
        return data
    except json.JSONDecodeError as e:
        print(f"    WARN: could not parse {p}: {e}")
        return []


def build_metadata_lookup(metadata_list, files_on_disk):
    """Map filename_on_disk -> metadata entry.

    The source metadata stores two filename fields: `random_filename` is
    the anonymized name typically present on disk, and `original_filename`
    is the human-readable one. We try both, in that order.
    """
    files_set = set(files_on_disk)
    lookup = {}
    for entry in metadata_list:
        for key in ('random_filename', 'original_filename'):
            fname = entry.get(key)
            if fname and fname in files_set and fname not in lookup:
                lookup[fname] = entry
                break
    return lookup


# ----------------------------------------------------------------------
# Build pipeline
# ----------------------------------------------------------------------

def build_split_specs(city_root: Path, city: str, split_name: str,
                      n_per_city, rng: random.Random):
    """Sample files from one source (city, split). Returns specs and counts."""
    files = collect_paired_files(city_root, split_name)
    metadata_list = load_split_metadata(city_root, split_name)
    meta_lookup = build_metadata_lookup(metadata_list, files)

    if n_per_city is None or n_per_city >= len(files):
        if n_per_city is not None and n_per_city > len(files):
            print(f"    WARN: {city}/{split_name} has only "
                  f"{len(files)} files, requested {n_per_city}")
        chosen = files
    else:
        chosen = sorted(rng.sample(files, n_per_city))

    specs = []
    for f in chosen:
        specs.append({
            'city':           city,
            'src_root':       city_root,
            'split_name':     split_name,
            'orig_filename':  f,
            'metadata_entry': meta_lookup.get(f),
        })
    return specs, len(files), len(chosen)


def materialize(split_dir: Path, source_specs, mode: str,
                prefix_with_city: bool, copy_multiclass: bool):
    """Place files for one output split. Returns per-image records."""
    records = []
    for spec in source_specs:
        city  = spec['city']
        src_r = spec['src_root']
        sname = spec['split_name']
        f     = spec['orig_filename']

        new_name = f"{city}__{f}" if prefix_with_city else f
        transfer(src_r / sname / 'images' / f,
                 split_dir / 'images' / new_name, mode)
        transfer(src_r / sname / 'masks' / f,
                 split_dir / 'masks'  / new_name, mode)

        had_multiclass = False
        if copy_multiclass:
            src_mmc = src_r / sname / 'masks_multiclass' / f
            if src_mmc.exists():
                transfer(src_mmc,
                         split_dir / 'masks_multiclass' / new_name, mode)
                had_multiclass = True

        rec = {
            'city':                 city,
            'source_split':         sname,
            'orig_filename':        f,
            'loco_filename':        new_name,
            'has_masks_multiclass': had_multiclass,
            'metadata':             spec['metadata_entry'],
        }
        records.append(rec)
    return records


def write_loco_metadata(fold_dir: Path, split_name: str, records,
                        fold_id: int, holdout_city: str, resolution: str):
    """Write metadata_{split}.json describing this fold/resolution split."""
    out = []
    for rec in records:
        base = dict(rec['metadata']) if rec['metadata'] is not None else {}
        base['loco_filename']        = rec['loco_filename']
        base['loco_split']           = split_name
        base['loco_fold_id']         = fold_id
        base['loco_holdout_city']    = holdout_city
        base['loco_resolution']      = resolution
        base['source_city']          = rec['city']
        base['source_split']         = rec['source_split']
        base['has_masks_multiclass'] = rec['has_masks_multiclass']
        # Preserve the original on-disk filename even if the upstream
        # metadata entry was missing.
        base.setdefault('original_filename', rec['orig_filename'])
        out.append(base)

    out_path = fold_dir / f'metadata_{split_name}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out_path


def build_fold(base_data_root: Path, out_root: Path, fold_id: int,
               resolution: str, n_train_per_city, n_val_per_city,
               mode: str, seed: int, copy_multiclass: bool):
    fold = LOCO_FOLDS[fold_id]
    train_cities = fold['train']
    test_city    = fold['test']

    fold_dir = out_root / f"fold_{fold_id}_holdout_{test_city}" / resolution
    fold_seed = seed + fold_id * 100  # deterministic, distinct per fold
    rng = random.Random(fold_seed)

    fold_man = {
        'fold_id':       fold_id,
        'holdout_city':  test_city,
        'train_cities':  train_cities,
        'resolution':    resolution,
        'seed':          fold_seed,
        'splits':        {},
    }

    # ----- train + val: sample from each training city -----
    for split_name, n_per_city in [('train', n_train_per_city),
                                   ('val',   n_val_per_city)]:
        all_specs, per_city = [], {}
        for city in train_cities:
            city_root = base_data_root / city / resolution
            specs, n_avail, n_chosen = build_split_specs(
                city_root, city, split_name, n_per_city, rng)
            all_specs.extend(specs)
            per_city[city] = {'available': n_avail, 'kept': n_chosen}

        recs = materialize(fold_dir / split_name, all_specs, mode,
                           prefix_with_city=True,
                           copy_multiclass=copy_multiclass)
        meta_path = write_loco_metadata(fold_dir, split_name, recs,
                                        fold_id, test_city, resolution)
        fold_man['splits'][split_name] = {
            'per_city_counts':   per_city,
            'total':             len(recs),
            'multiclass_count':  sum(1 for r in recs if r['has_masks_multiclass']),
            'metadata_file':     meta_path.name,
        }

    # ----- test: held-out city, full test pool, original filenames -----
    test_root = base_data_root / test_city / resolution
    test_specs, n_avail, n_chosen = build_split_specs(
        test_root, test_city, 'test', None, rng)

    recs = materialize(fold_dir / 'test', test_specs, mode,
                       prefix_with_city=False,
                       copy_multiclass=copy_multiclass)
    meta_path = write_loco_metadata(fold_dir, 'test', recs,
                                    fold_id, test_city, resolution)
    fold_man['splits']['test'] = {
        'per_city_counts':   {test_city: {'available': n_avail, 'kept': n_chosen}},
        'total':             len(recs),
        'multiclass_count':  sum(1 for r in recs if r['has_masks_multiclass']),
        'metadata_file':     meta_path.name,
    }

    return fold_man


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_data_root', required=True, type=Path)
    p.add_argument('--output_root',    required=True, type=Path)
    p.add_argument('--resolutions',    nargs='+', default=['highres', 'midres'])
    p.add_argument('--folds',          nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--n_train_per_city', type=int, default=225,
                   help="Train images per training city. "
                        "Default 225 matches paper (450 total). "
                        "Use -1 to take all available.")
    p.add_argument('--n_val_per_city',   type=int, default=75,
                   help="Val images per training city. "
                        "Default 75 matches paper (150 total). "
                        "Use -1 to take all available.")
    p.add_argument('--mode', choices=['copy', 'symlink', 'hardlink'],
                   default='copy',
                   help="copy (default): portable, self-contained dataset. "
                        "symlink: saves disk space but stays tied to the "
                        "source tree. hardlink: same FS, no extra storage.")
    p.add_argument('--no_multiclass', action='store_true',
                   help="Skip masks_multiclass/ even where it exists.")
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if not args.base_data_root.exists():
        raise SystemExit(f"base_data_root does not exist: {args.base_data_root}")

    n_train = None if args.n_train_per_city == -1 else args.n_train_per_city
    n_val   = None if args.n_val_per_city   == -1 else args.n_val_per_city

    args.output_root.mkdir(parents=True, exist_ok=True)
    common_meta = {
        'base_data_root':    str(args.base_data_root.resolve()),
        'output_root':       str(args.output_root.resolve()),
        'mode':              args.mode,
        'seed':              args.seed,
        'n_train_per_city':  n_train,
        'n_val_per_city':    n_val,
        'copy_multiclass':   not args.no_multiclass,
    }

    for fold_id in args.folds:
        for resolution in args.resolutions:
            holdout = LOCO_FOLDS[fold_id]['test']
            print(f"Fold {fold_id} (holdout={holdout}), {resolution}:")
            fm = build_fold(
                base_data_root=args.base_data_root,
                out_root=args.output_root,
                fold_id=fold_id,
                resolution=resolution,
                n_train_per_city=n_train,
                n_val_per_city=n_val,
                mode=args.mode,
                seed=args.seed,
                copy_multiclass=not args.no_multiclass,
            )
            tr, va, te = fm['splits']['train'], fm['splits']['val'], fm['splits']['test']
            print(f"  train={tr['total']} (mc {tr['multiclass_count']})  "
                  f"val={va['total']} (mc {va['multiclass_count']})  "
                  f"test={te['total']} (mc {te['multiclass_count']})")

            # Self-contained manifest per fold/resolution -- no shared
            # files, so parallel sbatch jobs don't race.
            fold_dir = args.output_root / f"fold_{fold_id}_holdout_{holdout}" / resolution
            with open(fold_dir / 'manifest.json', 'w') as f:
                json.dump({**common_meta, 'fold': fm}, f, indent=2)
            print(f"  manifest -> {fold_dir / 'manifest.json'}")


if __name__ == '__main__':
    main()