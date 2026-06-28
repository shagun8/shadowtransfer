"""
INRIA training wrapper for the shape-control experiment.

Reuses mamnet/train.py's Trainer class in full. Only the LOCO fold lookup
differs — nothing about optimizer, loss, scheduler, augmentation, decision
metric, or early stopping changes.

To keep train.py untouched, this file:
  (1) swaps data.dataset.LOCO_FOLDS in-place with INRIA_LOCO_FOLDS
      BEFORE importing Trainer
  (2) imports Trainer and get_args from train.py
  (3) runs training with whatever args the user passed

Directory naming: we rely on the parent --output_dir being distinct
(e.g. .../mamnet_inria/outputs/ vs .../mamnet/outputs/) so the internal
experiment-name scheme from Trainer (mamnet_{city}_{res}_1) doesn't
need to change. No rename required.

Usage (matches train.py argument surface):
  # Upper-bound (single city)
  python train_inria.py --mode single \
      --data_root {BASE}/data/Final_data_test/inria/austin/highres \
      --batch_size 8 --epochs 100 --lr 0.0001 --img_size 384 \
      --output_dir {BASE}/data/mamnet_inria/outputs \
      --use_contrast --eval_boundary_tolerant --boundary_tolerance 2 \
      --early_stopping_patience 10

  # LOCO holdout Austin (fold 0)
  python train_inria.py --mode loco \
      --base_data_root {BASE}/data/Final_data_test/inria \
      --resolution highres --fold_id 0 \
      --batch_size 8 --epochs 100 --lr 0.0001 --img_size 384 \
      --output_dir {BASE}/data/mamnet_inria/outputs \
      --use_contrast --eval_boundary_tolerant --boundary_tolerance 2 \
      --early_stopping_patience 10
"""

import os
import sys

# Make sure mamnet/ is on sys.path. Assumes this file lives in mamnet/.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# --- fold substitution: swap shadow LOCO_FOLDS for INRIA version ---
# Must happen BEFORE importing train.py (which imports LOCO_FOLDS from
# data.dataset). We mutate the dict in-place so already-imported
# references to data.dataset.LOCO_FOLDS see the INRIA structure.

from data import dataset as _shadow_ds
from inria_dataset import INRIA_LOCO_FOLDS

_shadow_ds.LOCO_FOLDS.clear()
_shadow_ds.LOCO_FOLDS.update(INRIA_LOCO_FOLDS)

# Now safe to import train.py
from train import Trainer, get_args  # noqa: E402

# Belt-and-suspenders: if train.py already imported LOCO_FOLDS as a
# local name (via 'from data.dataset import LOCO_FOLDS'), patch that too.
import train as _train_mod  # noqa: E402
if hasattr(_train_mod, 'LOCO_FOLDS'):
    _train_mod.LOCO_FOLDS = INRIA_LOCO_FOLDS


def main():
    args = get_args()

    # Sanity log so the job output shows which fold config is active
    print('=' * 60)
    print('INRIA TRAINING — using INRIA_LOCO_FOLDS:')
    for fid, cfg in INRIA_LOCO_FOLDS.items():
        print(f'  fold {fid}: train={cfg["train"]}, test={cfg["test"]}')
    if args.mode == 'loco':
        print(f'  ACTIVE: fold_id={args.fold_id} '
              f'-> holdout {INRIA_LOCO_FOLDS[args.fold_id]["test"]}')
    elif args.mode == 'single':
        print(f'  ACTIVE: single mode on {args.data_root}')
    print('=' * 60)

    trainer = Trainer(args)

    if args.eval_only:
        trainer.test()
    else:
        trainer.train()
        trainer.test()


if __name__ == '__main__':
    main()