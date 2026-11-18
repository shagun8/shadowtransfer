# `sib/` — SIB retrains and ablation scoring (App. J, K, L)

Released because these runs are inputs to `results/19` (App. K) and
`results/42` (App. L), both cited by the camera-ready.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `arme12_retrain.py` | RUN script: the E-12 FDA-recipe MAMNet-SIB retrains that App. K reports | `arme12_retrain_worker.sh` | App. K |
| `arme12_retrain_worker.sh` | worker: modules, env, `.FAILED` trap | `arme12_retrain_submit.sh` | App. K |
| `arme12_retrain_submit.sh` | SUBMIT wrapper: fan-out over cells and seeds; `--dry-run` | operator | App. K |
| `train_oglanet_sib_tgtfda.py` | OGLANet-SIB trainer variant whose FDA donor pool is the held-out city's train images | `matched450/n450_tgtfda.py` | §5, App. K |
| `armJ_sib_ablation_score.py` | scores the SIB component-ablation cells at K=2 and writes the per-image parquets `results/42` is built from | `armJ_sib_ablation_worker.sh` | App. L |
| `armJ_sib_ablation_worker.sh`, `armJ_sib_ablation_submit.sh` | worker and SUBMIT wrapper for the ablation chain | operator | App. L |

## Target access

`train_oglanet_sib_tgtfda.py` reaches for **held-out-city images** to build its
FDA donor pool, so every cell it trains is `unlabeled_target`, not
`source_only`. The paired cells that use a source-city donor pool are
`source_only`. `docs/CHECKPOINT_INVENTORY.json` records which is which per
checkpoint. This distinction is a reviewer request and it applies to our own
method, so do not compare the two families as though they had equal access.

## Ablation scoring metric

`armJ_sib_ablation_score.py` reports the **two-class** boundary-tolerant mIoU at
K=2 (App. E), computed from `shadowtransfer/metrics.py`. It computes no
bootstrap, no Holm correction and no recovery ratio; those are in `analysis/`.
