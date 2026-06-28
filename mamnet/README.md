# MAMNet: Multi-Scale Shadow Detection Network

Implementation of "A Full-Scale Shadow Detection Network Based on Multiple Attention
Mechanisms for Remote-Sensing Images" (Zhang et al., 2024), used as one of the three
backbones in the ShadowTransfer study.

> **Start here.** The entry points that matter: train the baselines with `train.py`
> (submit via `mamnet_submit.sh`) and our method with `train_mamnet_sib.py` (submit via
> `mamnet_sib_submit.sh`); evaluate a LOCO sweep with `loco_evaluation.py` (run via
> `test_loco.sh`). Everything else in the tables below is a per-method variant or a helper
> imported by these.

## Installation
```bash
pip install -r requirements.txt
```

## Navigating this directory
Each method has a **RUN** script (`*.sh`, holds the hyperparameters + `python …` call) and a
**SUBMIT** wrapper (`*_submit.sh`, SBATCH headers + `sbatch`). The submit wrapper exports the
env vars the run script reads. See the repo-root [`CALLGRAPH.md`](../CALLGRAPH.md) for the full
SUBMIT → RUN → python → modules chain.

### Method launch scripts (SUBMIT → RUN → python entry)
| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `mamnet_submit.sh` | queues Vanilla LOCO jobs | → `mamnet.sh` | Vanilla baseline |
| `mamnet.sh` | RUN: source-only training | invokes `train.py` | Vanilla baseline |
| `mamnet_fda_submit.sh` | queues FDA jobs | → `mamnet_fda.sh` | FDA |
| `mamnet_fda.sh` | RUN: Fourier domain adaptation (`--use_fda`) | invokes `train.py` | FDA |
| `mamnet_segdesic_submit.sh` | queues SegDeSiC jobs | → `mamnet_segdesic.sh` | SegDeSiC |
| `mamnet_segdesic.sh` | RUN: SegDeSiC training | invokes `train_segdesic.py` | SegDeSiC |
| `mamnet_iim_submit.sh` | queues IIM jobs | → `mamnet_iim.sh` | IIM |
| `mamnet_iim.sh` | RUN: illumination-invariant module | invokes `train_iim.py` | IIM |
| `mamnet_isw_submit.sh` | queues ISW jobs | → `mamnet_isw.sh` | ISW |
| `mamnet_isw.sh` | RUN: instance-selective whitening | invokes `train_isw.py` | ISW |
| `mamnet_mrfp_submit.sh` | queues MRFP+ jobs | → `mamnet_mrfp.sh` | MRFP+ |
| `mamnet_mrfp.sh` | RUN: multi-resolution feature perturbation | invokes `train_mrfp.py` | MRFP+ |
| `mamnet_fada_submit.sh` | queues FADA jobs | → `mamnet_fada.sh` | FADA |
| `mamnet_fada.sh` | RUN: FADA adversarial adaptation | invokes `train_fada.py` | FADA |
| `mamnet_sib_submit.sh` | queues SIB (ours) jobs | → `mamnet_sib.sh` | **SIB** (Sec 5) |
| `mamnet_sib_ablation_submit.sh` | queues SIB ablation grid | → `mamnet_sib.sh` | App. H ablations |
| `mamnet_sib.sh` | RUN: SIB training | invokes `train_mamnet_sib.py` | **SIB** (Sec 5) |
| `mamnet_sib_newmod_submit.sh` | queues SIB CACR/CE-AURC/TENT variants | → `mamnet_sib_newmod.sh` | App. H |
| `mamnet_sib_newmod.sh` | RUN: SIB new-module variants | invokes `train_mamnet_sib.py` | App. H |
| `mamnet_sib_tempscale_submit.sh` | queues temp-scaling eval | → `mamnet_sib.sh` | §5.3 temp scaling |
| `mamnet_sib_tempscale.sh` | RUN: temperature-scaling sweep | invokes `tempscale_eval.py` | §5.3 |
| `mamnet_sib_sp_gap_submit.sh` | queues selective-prediction-gap eval | → `mamnet_sib_sp_gap.sh` | 4.3 SP-gap |
| `mamnet_sib_sp_gap.sh` | RUN: SP-gap analysis | invokes `sp_gap_analysis.py` | 4.3 SP-gap |
| `mamnet_inria_submit.sh` | queues INRIA cross-task jobs | → `mamnet_inria.sh` | App. G.7 |
| `mamnet_inria.sh` | RUN: INRIA building-footprint training | invokes `train_inria.py` | App. G.7 |

### Data-prep, eval, inference & aggregation scripts
| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `build_loco_splits_submit.sh` / `build_loco_splits.sh` | build LOCO city splits | RUN invokes `build_loco_splits.py` | LOCO protocol |
| `compute_isw_masks_submit.sh` / `compute_isw_masks.sh` | precompute ISW whitening masks | RUN invokes `compute_isw_masks.py` | ISW |
| `inference_submit.sh` / `inference.sh` | run inference on test cities | RUN invokes `run_inference.py` | Sec 3.4 |
| `annotate_pre_infer_submit.sh` / `annotate_pre_infer.sh` | pre-inference annotation pass | RUN invokes `inference_annotation.py` | inference utility |
| `eval_mamnet_sib_submit.sh` / `eval_mamnet_sib.sh` | evaluate SIB checkpoints | RUN invokes `eval_mamnet_sib.py` | Sec 5 |
| `test_loco_submit.sh` / `test_loco.sh` | LOCO evaluation sweep | RUN invokes `loco_evaluation.py`, `aggregate_loco_results.py` | LOCO results |
| `test_res_submit.sh` / `test_res.sh` | resolution-transfer evaluation | RUN invokes `res_evaluation.py`, `aggregate_loco_results.py` | resolution study |
| `one_file_submit.sh` | single-file LOCO eval helper | → `test_loco.sh` | utility |
| `plot_loco_agg_submit.sh` / `plot_loco_agg.sh` | plot aggregated LOCO results | RUN invokes `aggregate_loco_results.py` | figures |
| `plots_res_test_submit.sh` / `plots_res_test.sh` | plot resolution-test results | RUN invokes `aggregate_res_results.py` | figures |

### Python files
| file | what it does | invoked by / imports | paper ref |
|---|---|---|---|
| `train.py` | Vanilla + FDA training entry | called by `mamnet.sh`,`mamnet_fda.sh`; imports `models/mamnet`, `data/dataset`, `utils/losses`,`metrics` | Vanilla / FDA |
| `train_segdesic.py` | SegDeSiC training entry | called by `mamnet_segdesic.sh`; imports `models/mamnet_segdesic`, `utils/geo_losses` | SegDeSiC |
| `train_iim.py` | IIM training entry | called by `mamnet_iim.sh`; imports `models/mamnet_iim`,`models/iim` | IIM |
| `train_isw.py` | ISW training entry | called by `mamnet_isw.sh`; imports `models/mamnet`, `utils/isw_loss` | ISW |
| `train_mrfp.py` | MRFP+ training entry | called by `mamnet_mrfp.sh`; imports `models/mamnet_mrfp`, `utils/losses_mrfp` | MRFP+ |
| `train_fada.py` | FADA training entry | called by `mamnet_fada.sh`; imports `models/mamnet_fada` | FADA |
| `train_mamnet_sib.py` | **SIB** training entry | called by `mamnet_sib.sh`,`mamnet_sib_newmod.sh`; imports `models/mamnet_sib`,`models/sib`, `data/dataset_sib`, `utils/losses`(CACR/CE-AURC) | **SIB** (Sec 5 / App. H) |
| `train_inria.py` | INRIA cross-task training | called by `mamnet_inria.sh`; imports `inria_dataset` | App. G.7 |
| `eval_mamnet_sib.py` | evaluate SIB model | called by `eval_mamnet_sib.sh` | Sec 5 |
| `sp_gap_analysis.py` | selective-prediction gap analysis | called by `mamnet_sib_sp_gap.sh` | 4.3 SP-gap |
| `tempscale_eval.py` | temperature-scaling evaluation | called by `mamnet_sib_tempscale.sh` & root `tempscale.sh` | §5.3 |
| `run_inference.py` | inference over test set (reads `config.yaml`) | called by `inference.sh` | Sec 3.4 |
| `inference_annotation.py` | annotation helper for inference | called by `annotate_pre_infer.sh` | utility |
| `loco_evaluation.py` | LOCO test-set evaluation | called by `test_loco.sh` | LOCO results |
| `res_evaluation.py` | resolution-transfer evaluation | called by `test_res.sh` | resolution study |
| `aggregate_loco_results.py` | aggregate LOCO metrics across folds | called by `test_loco.sh`,`plot_loco_agg.sh` | LOCO results |
| `aggregate_res_results.py` | aggregate resolution-test metrics | called by `plots_res_test.sh` | resolution study |
| `build_loco_splits.py` | construct leave-one-city-out splits | called by `build_loco_splits.sh` | LOCO protocol |
| `generate_split.py` | spatial-strategy patch split generator | standalone; imports `utils/spatial_sampling` | data prep |
| `compute_isw_masks.py` | precompute ISW covariance masks | called by `compute_isw_masks.sh` | ISW |
| `inria_dataset.py` | INRIA dataset loader | imported by `train_inria.py` | App. G.7 |
| `visualize_splits_spatial_metrics.py` | visualize spatial split metrics | standalone | data prep / diagnostics |

### `models/` — network and per-method modules
| file | what it does |
|---|---|
| `mamnet.py` | base MAMNet (encoder + decoder + attention) |
| `encoder.py` / `encoder_4ch.py` | backbone encoder (3-ch / 4-ch input) |
| `decoder.py` | segmentation decoder head |
| `attention.py` / `mscaf.py` | multi-scale attention blocks |
| `auxiliary.py` | auxiliary heads / weight init |
| `mamnet_iim.py` / `iim.py` | IIM model + illumination-invariance loss |
| `mamnet_segdesic.py` / `segdesic_module.py` | SegDeSiC model + module |
| `mamnet_mrfp.py` / `mrfp_modules.py` | MRFP+ model + perturbation modules |
| `mamnet_fada.py` / `fada.py` | FADA model + adversarial modules |
| `mamnet_sib.py` / `sib.py` | **SIB** model (`build_mamnet_sib`) + SIB/TENT core |

### `data/` — datasets and transforms
| file | what it does |
|---|---|
| `dataset.py` | base shadow dataset + `get_dataloaders` + `LOCO_FOLDS` |
| `dataset_enhanced.py` | augmentation-enhanced dataset |
| `dataset_sib.py` | SIB dataset + `get_dataloaders_sib` |
| `augmentation_enhanced.py` | enhanced augmentation pipeline |
| `fda_transform.py` | Fourier domain adaptation transform |
| `contrast_utils.py` | contrast-based preprocessing |

### `utils/` — losses, metrics, evaluation, viz
| file | what it does |
|---|---|
| `losses.py` | MAMNet loss + CACR / CE-AURC losses |
| `losses_mrfp.py` | MRFP+ loss |
| `geo_losses.py` | SegDeSiC geometric loss |
| `isw_loss.py` | ISW whitening loss + encoder hooks |
| `metrics.py` | shadow metrics + `evaluate_model` |
| `evaluation_detailed.py` | detailed per-class evaluator |
| `postprocessing.py` | small-prediction filtering |
| `spatial_sampling.py` | spatial patch sampling (`select_patches_by_strategy`) |
| `generate_geo_metadata.py` | per-tile geo metadata generation |
| `pseudo_cloud_aug.py` | pseudo-cloud augmentation |
| `convert_mapping.py` | label/index mapping helper |
| `contrast_utils.py` | contrast utilities (shared) |
| `visualization.py` / `visualization_isw.py` / `visualization_segdesic.py` | result visualizations |
| `utils.py` | misc helpers (incl. `config.yaml` loading + `${PROJECT_ROOT}` expansion) |
