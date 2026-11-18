# `appendix_h/` — alternative-explanation experiment drivers (App. H)

App. H tests four explanations that could account for the LOCO gap without a
geographic-transfer story. Each driver trains or evaluates cells; the statistics
that turn their dumps into the appendix's claims are in `analysis/`.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `armc_train.py` | training-set scale ladder: trains one cell at a given rung (225/450/900) | submit wrapper | App. H.1 |
| `armc_r450.py` | recomputes the published cells at rung 450 and reproduces each cell's stored result before emitting | submit wrapper | App. H.1, §3.3 |
| `armd_train.py` | appearance augmentation: trains one cell with the augmentation stack from `shadowtransfer/armd_aug.py` | submit wrapper | App. H.2 |
| `arme_part1_train.py` | frozen DINOv3: trains the decoder on a frozen backbone | submit wrapper | App. H.3 |
| `arme_part2_aggregate.py` | aggregates the frozen-backbone part-2 cells | operator | App. H.3 |
| `arme_part2_revised_aggregate.py` | the revised part-2 aggregation used for the camera-ready | operator | App. H.3 |
| `armL_train.py` | pretraining / fine-tuning regime: trains one regime cell | submit wrapper | App. H.4 |
| `armL_results.py` | collects the regime cells into a per-cell record | operator | App. H.4 |

The ladder statistics script (`analysis/armc_ladder_stats.py`) and the
frozen-backbone gap statistics (`analysis/arme_frozen_gap_stats.py`) are in
`analysis/`, not here: this directory produces per-image dumps, and nothing in
it pools images into a mean, an interval or a p-value.

`armc_r450.py` carries a `rung_ok` / `rung_label_anomaly` mechanism that writes
`rung: null` plus an anomaly block when a cell's published `n_train` is not 450.
Its docstring records evidence, from the published runs' own logs, that MAMNet's
three LOCO cells trained on 900 images rather than 450. Read that docstring
before treating the `rung` field of those three cells as authoritative.
