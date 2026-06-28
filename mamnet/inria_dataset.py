"""
INRIA LOCO fold configuration for the shape-control experiment.

Mirrors the structure of mamnet/data/dataset.py's LOCO_FOLDS — 3 folds,
train on 2 cities, hold out 1. Reuses ShadowDatasetEnhanced without
modification; only the fold lookup differs.

Usage pattern (see train_inria.py):
    from inria_dataset import INRIA_LOCO_FOLDS
    fold_config = INRIA_LOCO_FOLDS[args.fold_id]
    train_cities = fold_config['train']
    test_city    = fold_config['test']
"""

# Mirrors shadow LOCO structure.
#   Shadow fold 0 holds out phoenix (alphabetically last)
#   INRIA fold 0 holds out austin  (alphabetically first)
# Paper table will report all three folds per task for clean matched comparison.
INRIA_LOCO_FOLDS = {
    0: {'train': ['chicago', 'vienna'],  'test': 'austin'},
    1: {'train': ['austin',  'vienna'],  'test': 'chicago'},
    2: {'train': ['austin',  'chicago'], 'test': 'vienna'},
}

INRIA_CITIES = ['austin', 'chicago', 'vienna']
INRIA_RESOLUTION = 'highres'    # INRIA is 0.3m GSD; we only run highres