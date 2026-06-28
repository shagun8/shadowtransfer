# DINOv3 architecture

DINOv3 (ViT-S/16) backbone — the third backbone in the ShadowTransfer study. Unlike the two
CNN backbones, the model/module files live flat in this directory (no `models/` subdir).
See the repo-root [`CALLGRAPH.md`](../CALLGRAPH.md) for the full SUBMIT → RUN → python →
modules chain.

> **Backbone setup:** `dinov3_backbone.py` inserts `<this dir>/dinov3/` onto `sys.path` and
> imports `from dinov3.models import vision_transformer`. Clone the upstream repo to
> `dinov3/dinov3/` and place weights under `dinov3/weights/` — see the top-level README's
> "DINOv3 backbone" section. Neither the vendored source nor the weights are redistributed here.

> **Start here.** The entry points that matter: train the baselines with `train_dinov3.py`
> (submit via `dinov3_submit.sh`) and our method with `train_dinov3_sib.py` (submit via
> `dinov3_sib_submit.sh`); evaluate a LOCO sweep with `loco_evaluation.py` (run via
> `test_loco.sh`). Everything else in the tables below is a per-method variant or a helper.

### Method launch scripts (SUBMIT → RUN → python entry)
| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `dinov3_submit.sh` | queues Vanilla LOCO jobs | → `dinov3.sh` | Vanilla baseline |
| `dinov3.sh` | RUN: source-only training | invokes `train_dinov3.py` | Vanilla baseline |
| `dinov3_fda_submit.sh` | queues FDA jobs | → `dinov3_fda.sh` | FDA |
| `dinov3_fda.sh` | RUN: Fourier domain adaptation | invokes `train_dinov3.py` | FDA |
| `dinov3_segdesic_submit.sh` | queues SegDeSiC jobs | → `dinov3_segdesic.sh` | SegDeSiC |
| `dinov3_segdesic.sh` | RUN: SegDeSiC training | invokes `train_dinov3_segdesic.py` | SegDeSiC |
| `dinov3_iim_submit.sh` | queues IIM jobs | → `dinov3_iim.sh` | IIM |
| `dinov3_iim.sh` | RUN: illumination-invariant module | invokes `train_dinov3_iim.py` | IIM |
| `dinov3_isw_submit.sh` | queues ISW jobs | → `dinov3_isw.sh` | ISW |
| `dinov3_isw.sh` | RUN: instance-selective whitening | invokes `train_dinov3_isw.py` | ISW |
| `dinov3_mrfp_submit.sh` | queues MRFP+ jobs | → `dinov3_mrfp.sh` | MRFP+ |
| `dinov3_mrfp.sh` | RUN: multi-resolution feature perturbation | invokes `train_dinov3_mrfp.py` | MRFP+ |
| `dinov3_fada_submit.sh` | queues FADA jobs | → `dinov3_fada.sh` | FADA |
| `dinov3_fada.sh` | RUN: FADA adversarial adaptation | invokes `train_dinov3_fada.py` | FADA |
| `dinov3_sib_submit.sh` | queues SIB (ours) jobs | → `dinov3_sib.sh` | **SIB** (Sec 5) |
| `dinov3_sib_ablation_submit.sh` | queues SIB ablation grid | → `dinov3_sib.sh` | App. H ablations |
| `dinov3_sib.sh` | RUN: SIB training | invokes `train_dinov3_sib.py` | **SIB** (Sec 5) |
| `dinov3_sib_newmod_submit.sh` | queues SIB CACR/CE-AURC/TENT variants | → `dinov3_sib_newmod.sh` | App. H |
| `dinov3_sib_newmod.sh` | RUN: SIB new-module variants | invokes `train_dinov3_sib.py` | App. H |
| `dinov3_sib_sp_gap_submit.sh` | queues selective-prediction-gap eval | → `dinov3_sib_sp_gap.sh` | 4.3 SP-gap |
| `dinov3_sib_sp_gap.sh` | RUN: SP-gap analysis | invokes `sp_gap_analysis.py` | 4.3 SP-gap |

### Data-prep, eval, inference & aggregation scripts
| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `compute_isw_masks_dinov3_submit.sh` / `compute_isw_masks_dinov3.sh` | precompute ISW whitening masks | RUN invokes `compute_isw_masks_dinov3.py` | ISW |
| `inference_submit.sh` / `inference.sh` | run inference on test cities | RUN invokes `run_inference.py` | Sec 3.4 |
| `test_loco_submit.sh` / `test_loco.sh` | LOCO evaluation sweep | RUN invokes `loco_evaluation.py`, `aggregate_loco_results.py` | LOCO results |
| `plot_loco_agg_submit.sh` / `plot_loco_agg.sh` | plot aggregated LOCO results | RUN invokes `aggregate_loco_results.py` | figures |

### Python files — training entries & analysis
| file | what it does | invoked by / imports | paper ref |
|---|---|---|---|
| `train_dinov3.py` | Vanilla + FDA training entry | called by `dinov3.sh`,`dinov3_fda.sh`; imports `dinov3_model` | Vanilla / FDA |
| `train_dinov3_segdesic.py` | SegDeSiC training entry | called by `dinov3_segdesic.sh`; imports `dinov3_segdesic` | SegDeSiC |
| `train_dinov3_iim.py` | IIM training entry | called by `dinov3_iim.sh`; imports `dinov3_iim_model`,`iim` | IIM |
| `train_dinov3_isw.py` | ISW training entry | called by `dinov3_isw.sh`,`compute_isw_masks_dinov3.sh`; imports `dinov3_model`,`utils/isw_loss_dinov3` | ISW |
| `train_dinov3_mrfp.py` | MRFP+ training entry | called by `dinov3_mrfp.sh`; imports `dinov3_model_mrfp`,`mrfp_modules` | MRFP+ |
| `train_dinov3_fada.py` | FADA training entry | called by `dinov3_fada.sh`; imports `dinov3_model_fada` | FADA |
| `train_dinov3_sib.py` | **SIB** training entry | called by `dinov3_sib.sh`,`dinov3_sib_newmod.sh`; imports `dinov3_model_sib`, `data/dataset_sib`, `utils/losses`(CACR/CE-AURC) | **SIB** (Sec 5 / App. H) |
| `sp_gap_analysis.py` | selective-prediction gap analysis | called by `dinov3_sib_sp_gap.sh` | 4.3 SP-gap |
| `tempscale_eval.py` | temperature-scaling evaluation | called by root `tempscale.sh` | §5.3 |
| `run_inference.py` | inference over test set (reads `config.yaml`) | called by `inference.sh` | Sec 3.4 |
| `loco_evaluation.py` | LOCO test-set evaluation | called by `test_loco.sh` | LOCO results |
| `aggregate_loco_results.py` | aggregate LOCO metrics across folds | called by `test_loco.sh`,`plot_loco_agg.sh` | LOCO results |
| `compute_isw_masks_dinov3.py` | precompute ISW covariance masks | called by `compute_isw_masks_dinov3.sh` | ISW |

### Model / module files (flat, this directory)
| file | what it does |
|---|---|
| `dinov3_backbone.py` | wraps upstream DINOv3 ViT-S/16; extracts multi-scale features (needs `dinov3/` submodule) |
| `dinov3_decoder.py` | segmentation decoder (`DINOv3Decoder`, `ConvBlock`) |
| `dinov3_model.py` | base detector (`DINOv3ShadowDetector`) = backbone + decoder |
| `dinov3_model_sib.py` | **SIB** detector (`DINOv3ShadowDetectorSIB`) |
| `dinov3_iim_model.py` | IIM detector variant |
| `dinov3_model_fada.py` | FADA detector variant |
| `dinov3_model_mrfp.py` | MRFP+ detector variant |
| `dinov3_segdesic.py` | SegDeSiC detector variant |
| `iim.py` | illumination-invariance loss (`compute_ii_loss`) |
| `mrfp_modules.py` | MRFP+ perturbation modules |
| `scale_attention_dinov3.py` | multi-scale attention for ViT features |
| `sib.py` | SIB / TENT core (Haar + VIB + adaptation) |
| `__init__.py` | package init |

### `data/` and `utils/`
| file | what it does |
|---|---|
| `data/dataset.py` | base shadow dataset + `get_dataloaders` |
| `data/dataset_sib.py` | SIB dataset + `get_dataloaders_sib` |
| `data/fda_transform.py` | Fourier domain adaptation transform |
| `utils/losses.py` | cross-entropy + CACR / CE-AURC losses |
| `utils/losses_mrfp.py` | MRFP+ loss |
| `utils/geo_losses.py` | SegDeSiC geometric loss |
| `utils/isw_loss_dinov3.py` | ISW whitening loss for ViT features |
| `utils/metrics.py` | shadow metrics |
| `utils/evaluation_detailed.py` | detailed per-class evaluator |
| `utils/postprocessing.py` | small-prediction filtering |
| `utils/pseudo_cloud_aug.py` | pseudo-cloud augmentation |
| `utils/contrast_utils.py` | contrast utilities |
| `utils/visualization.py` / `utils/visualization_dinov3_isw.py` / `utils/visualization_segdesic.py` | result visualizations |
| `utils/utils.py` | misc helpers (incl. `config.yaml` loading + `${PROJECT_ROOT}` expansion) |
