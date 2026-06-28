# OGLANet architecture

OGLANet (object-/global-local attention) CNN backbone — the second of the three backbones in
the ShadowTransfer study. Same RUN/SUBMIT convention as the other architectures; see the
repo-root [`CALLGRAPH.md`](../CALLGRAPH.md) for the full SUBMIT → RUN → python → modules chain.

> **Start here.** The entry points that matter: train the baselines with `train.py`
> (submit via `oglanet_submit.sh`) and our method with `train_oglanet_sib.py` (submit via
> `oglanet_sib_submit.sh`); evaluate a LOCO sweep with `loco_evaluation.py` (run via
> `test_loco.sh`). Everything else in the tables below is a per-method variant or a helper.

### Method launch scripts (SUBMIT → RUN → python entry)
| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `oglanet_submit.sh` | queues Vanilla LOCO jobs | → `oglanet.sh` | Vanilla baseline |
| `oglanet.sh` | RUN: source-only training | invokes `train.py` | Vanilla baseline |
| `oglanet_fda_submit.sh` | queues FDA jobs | → `oglanet_fda.sh` | FDA |
| `oglanet_fda.sh` | RUN: Fourier domain adaptation | invokes `train.py` | FDA |
| `oglanet_segdesic_submit.sh` | queues SegDeSiC jobs | → `oglanet_segdesic.sh` | SegDeSiC |
| `oglanet_segdesic.sh` | RUN: SegDeSiC training | invokes `train_segdesic.py` | SegDeSiC |
| `oglanet_iim_submit.sh` | queues IIM jobs | → `oglanet_iim.sh` | IIM |
| `oglanet_iim.sh` | RUN: illumination-invariant module | invokes `train_iim.py` | IIM |
| `oglanet_isw_submit.sh` | queues ISW jobs | → `oglanet_isw.sh` | ISW |
| `oglanet_isw.sh` | RUN: instance-selective whitening | invokes `train_oglanet_isw.py` | ISW |
| `oglanet_mrfp_submit.sh` | queues MRFP+ jobs | → `oglanet_mrfp.sh` | MRFP+ |
| `oglanet_mrfp.sh` | RUN: multi-resolution feature perturbation | invokes `train_oglanet_mrfp.py` | MRFP+ |
| `oglanet_fada_submit.sh` | queues FADA jobs | → `oglanet_fada.sh` | FADA |
| `oglanet_fada.sh` | RUN: FADA adversarial adaptation | invokes `train_fada.py` | FADA |
| `oglanet_sib_submit.sh` | queues SIB (ours) jobs | → `oglanet_sib.sh` | **SIB** (Sec 5) |
| `oglanet_sib_ablation_submit.sh` | queues SIB ablation grid | → `oglanet_sib.sh` | App. H ablations |
| `oglanet_sib.sh` | RUN: SIB training | invokes `train_oglanet_sib.py` | **SIB** (Sec 5) |
| `oglanet_sib_newmod_submit.sh` | queues SIB CACR/CE-AURC/TENT variants | → `oglanet_sib_newmod.sh` | App. H |
| `oglanet_sib_newmod.sh` | RUN: SIB new-module variants | invokes `train_oglanet_sib.py` | App. H |
| `oglanet_sib_sp_gap_submit.sh` | queues selective-prediction-gap eval | → `oglanet_sib_sp_gap.sh` | 4.3 SP-gap |
| `oglanet_sib_sp_gap.sh` | RUN: SP-gap analysis | invokes `sp_gap_analysis.py` | 4.3 SP-gap |

### Data-prep, eval, inference & aggregation scripts
| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `compute_isw_masks_oglanet_submit.sh` / `compute_isw_masks_oglanet.sh` | precompute ISW whitening masks | RUN invokes `compute_isw_masks_oglanet.py` | ISW |
| `inference_submit.sh` / `inference.sh` | run inference on test cities | RUN invokes `run_inference.py` | Sec 3.4 |
| `complete_eval_submit.sh` / `complete_eval.sh` | full evaluation sweep | RUN invokes `complete_eval.py` | LOCO results |
| `test_loco_submit.sh` / `test_loco.sh` | LOCO evaluation sweep | RUN invokes `loco_evaluation.py`, `aggregate_loco_results.py` | LOCO results |
| `plot_loco_agg_submit.sh` / `plot_loco_agg.sh` | plot aggregated LOCO results | RUN invokes `aggregate_loco_results.py` | figures |

### Python files
| file | what it does | invoked by / imports | paper ref |
|---|---|---|---|
| `train.py` | Vanilla + FDA training entry | called by `oglanet.sh`,`oglanet_fda.sh`; imports `models/oglanet`, `data/dataset`, `utils/losses` | Vanilla / FDA |
| `train_segdesic.py` | SegDeSiC training entry | called by `oglanet_segdesic.sh`; imports `models/oglanet_segdesic` | SegDeSiC |
| `train_iim.py` | IIM training entry | called by `oglanet_iim.sh`; imports `models/oglanet_iim`,`models/iim` | IIM |
| `train_oglanet_isw.py` | ISW training entry | called by `oglanet_isw.sh`; imports `utils/isw_loss` | ISW |
| `train_oglanet_mrfp.py` | MRFP+ training entry | called by `oglanet_mrfp.sh`; imports `models/oglanet_mrfp`,`utils/losses_oglanet_mrfp` | MRFP+ |
| `train_fada.py` | FADA training entry | called by `oglanet_fada.sh`; imports `models/oglanet_fada` | FADA |
| `train_oglanet_sib.py` | **SIB** training entry | called by `oglanet_sib.sh`,`oglanet_sib_newmod.sh`; imports `models/oglanet_sib`,`models/sib`, `data/dataset_sib`, `utils/losses`(CACR/CE-AURC) | **SIB** (Sec 5 / App. H) |
| `sp_gap_analysis.py` | selective-prediction gap analysis | called by `oglanet_sib_sp_gap.sh` | 4.3 SP-gap |
| `tempscale_eval.py` | temperature-scaling evaluation | called by root `tempscale.sh` | §5.3 |
| `run_inference.py` | inference over test set (reads `config.yaml`) | called by `inference.sh` | Sec 3.4 |
| `complete_eval.py` | full evaluation routine | called by `complete_eval.sh` | LOCO results |
| `loco_evaluation.py` | LOCO test-set evaluation | called by `test_loco.sh` | LOCO results |
| `aggregate_loco_results.py` | aggregate LOCO metrics across folds | called by `test_loco.sh`,`plot_loco_agg.sh` | LOCO results |
| `compute_isw_masks_oglanet.py` | precompute ISW covariance masks | called by `compute_isw_masks_oglanet.sh` | ISW |

### `models/` — network and per-method modules
| file | what it does |
|---|---|
| `oglanet.py` | base OGLANet |
| `encoder.py` / `encoder_4ch.py` | backbone encoder (3-ch / 4-ch input) |
| `decoder.py` | segmentation decoder head |
| `oam.py` / `glam.py` / `gfem.py` / `dffm.py` | OGLANet attention / fusion modules |
| `oglanet_iim.py` / `iim.py` | IIM model + illumination-invariance loss |
| `oglanet_segdesic.py` / `segdesic_module.py` | SegDeSiC model + module |
| `oglanet_mrfp.py` / `mrfp_modules.py` | MRFP+ model + perturbation modules |
| `oglanet_fada.py` / `fada.py` | FADA model + adversarial modules |
| `oglanet_sib.py` / `sib.py` | **SIB** model (`OGLANetSIB`) + SIB/TENT core |

### `data/` — datasets and transforms
| file | what it does |
|---|---|
| `dataset.py` | base shadow dataset + `get_dataloaders` |
| `dataset_enhanced.py` | augmentation-enhanced dataset |
| `dataset_sib.py` | SIB dataset + `get_dataloaders_sib` |
| `fda_transform.py` | Fourier domain adaptation transform |

### `utils/` — losses, metrics, evaluation, viz
| file | what it does |
|---|---|
| `losses.py` | OGLANet loss + CACR / CE-AURC losses |
| `losses_oglanet_mrfp.py` | MRFP+ loss (OGLANet-specific; this is the one OGLANet MRFP+ uses) |
| `geo_losses.py` | SegDeSiC geometric loss |
| `isw_loss.py` | ISW whitening loss + encoder hooks |
| `metrics.py` | shadow metrics |
| `evaluation_detailed.py` | detailed per-class evaluator |
| `postprocessing.py` | small-prediction filtering |
| `contrast_utils.py` | contrast utilities |
| `visualization.py` / `visualization_oglanet_isw.py` / `visualization_segdesic.py` | result visualizations |
| `utils.py` | misc helpers (incl. `config.yaml` loading + `${PROJECT_ROOT}` expansion) |
