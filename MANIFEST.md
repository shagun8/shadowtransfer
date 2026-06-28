# MANIFEST — ShadowTransfer public-repo inventory

**Source (READ-ONLY):** the original source tree.
**Status:** STEP 1 (inventory). **Nothing copied yet.** Awaiting your approval of this
table before any KEEP file is copied into the clean repo.

## Legend
- **KEEP → \<paper section\>** — maps to the paper; will be copied + scrubbed.
- **EXCLUDE → extra method** — one of the 4 non-paper methods (`gsdpe`, `hrda`, `mcl`, `ddib`).
- **EXCLUDE → cruft** — logs, `__pycache__`, weights, vendored repo, figure-output data, working notes.
- **ASK** — I can't confidently map it; my recommendation is given, you decide.

Paper's 7 evaluated configs: **Vanilla, FDA, SegDeSiC, IIM, ISW, MRFP+, FADA** + the paper's
own method **SIB** (ShadowTransfer, Sec 5 / App. H).

> **Orphan sweep (cleanup pass 3):** a follow-up pass removed helper files that became orphans
> after the method-name exclusion (their only importer was an excluded method) plus a few unused
> per-arch copies and inert mis-named `init.py` files. The KEEP lists below have been pruned
> accordingly. Full rationale + the approved removal set: [`ORPHANS.md`](ORPHANS.md).

---

## ✅ DECISIONS — RESOLVED (applied to the copy)

| # | Item | **Resolution** |
|---|------|----------------|
| D1 | Target-city fine-tuning cluster (`*finetuning*`, `analyze_finetune_results.py`, `loco_ft*.sh`, ×3 archs) | **EXCLUDE all** (user) |
| D2 | SIB ablation version | **KEEP `sib_ablation_analysis_v2.*` only; EXCLUDE v1** (`sib_ablation_analysis.py/.sh/_submit.sh`) (default) |
| D3 | NewMod case study (`newmod_ablation_analysis.*`, `*_sib_newmod.sh`) | **KEEP** (user) |
| D4 | final_loco thread suite (`thread*`, `run_diagnostics`, `run_threads*`, `extract_features`, `run_extract*`, `compute_paper_statistics`, `run_paper_stats*`) | **KEEP all** (user) |
| D5 | `experiment_b_bn_swap.py` | **EXCLUDE** (user) |
| D6 | Robust-recovery (`evaluate_recovery_robust.py`, `robust_recovery/*.json`) | **EXCLUDE** (default; JSONs are results data) |
| D7 | `data_efficiency_submit.sh` | **EXCLUDE** (default; ties to D1) |
| D8 | `composition_analysis/` (entire dir) | **EXCLUDE all** (default) |
| D9 | `select_figure2_tiles.*` | **KEEP → Fig 2** (default) |
| D10 | `mamnet/check_job_status.py` | **EXCLUDE → cruft** (default) |

---

## TOP-LEVEL scripts (`python/*.py`, `*.sh`)

Each analysis script has a RUN `.sh` (the actual `python … --args`, KEEP) and a `_submit.sh`
SBATCH wrapper (KEEP, **scrub hard**). Both listed once via the `.py` row unless noted.

| File(s) | Classification |
|---------|----------------|
| `run_inference.py` (+`.sh`,+`_submit.sh`) | KEEP → Sec 3.4 inference (binary masks) |
| `run_inference_probs.py` (+`.sh`,+`_submit.sh`) | KEEP → 4.3 SP-gap (per-pixel probabilities) |
| `run_inference_probs_inria.py` (+`.sh`,+`_submit.sh`) | KEEP → App. G.7 cross-task (INRIA) |
| `runonefile.sh` (+`_submit.sh`) | KEEP → single-image inference utility |
| `analyze_inference_results.py` | KEEP → Sec 3.4 boundary-tolerant eval |
| `statistical_analysis.py` | KEEP → Sec 3.4 bootstrap stats |
| `final_comparison.py` (+`.sh`,+`_submit.sh`) | KEEP → main results: 7 baselines + SIB per (arch,city) |
| `aggregate_sp_gap.py` (+`.sh`,+`_submit.sh`) | KEEP → 4.3 selective-prediction gap |
| `aggregate_coverage_recover.py` (+`.sh`,+`_submit.sh`) | KEEP → 4.3 / §5 coverage-recovery (Phase 2, c\*_val) |
| `phase1_aggregate.py` (+`.sh`,+`_submit.sh`) | KEEP → 4.3 / §5 Phase-1 AURC decision rule |
| `beta_sweep_summarize.py` (+`beta_sweep.sh`,+`_submit.sh`) | KEEP → App. K coverage-threshold stability |
| `verify_c_star_val.py` (+`.sh`,+`_submit.sh`) | KEEP → App. G.6/K c\*_val sanity check |
| `tempscale_aggregate.py` (+`tempscale.sh`,+`tempscale_aggregate.sh`,+submits) | KEEP → §5.3 temperature-scaling case study |
| `split_diagnostics.py` (+`.sh`,+`_submit.sh`) | KEEP → App. D split diagnostics |
| `feature_diagnostics.py` (+`.sh`,+`_submit.sh`) | KEEP → App. D feature-divergence (KS/Wasserstein) |
| `eval_sib.py` (+`.sh`,+`_submit.sh`) | KEEP → Sec 5 / App. H (SIB eval) |
| `sib_ablation_analysis_v2.py` (+`.sh`,+`_submit.sh`) | KEEP → App. H / Table 4 (see **D2**) |
| `sib_ablation_analysis.py` (+`.sh`,+`_submit.sh`) | ASK → superseded by v2? (**D2**) |
| `newmod_ablation_analysis.py` (+`.sh`,+`_submit.sh`) | ASK → diagnostic-module case study (**D3**) |
| `select_figure2_tiles.py` (+`.sh`,+`_submit.sh`) | KEEP → Fig 2 (**D9**) |

---

## `mamnet/`

### KEEP — base architecture & shared infra
| File | Maps to |
|------|---------|
| `__init__.py`, `config.yaml`, `requirements.txt`, `README.md` | package / config (README = upstream MAMNet paper citation, no identity) |
| `train.py` | Vanilla **and** FDA (`--use_fda`) |
| `models/{mamnet,encoder,encoder_4ch,decoder,attention,auxiliary,mscaf}.py` | base network |
| `data/{dataset,dataset_enhanced,augmentation_enhanced,contrast_utils,fda_transform}.py` | datasets + FDA transform |
| `utils/{losses,metrics,utils,postprocessing,evaluation_detailed,geo_losses,spatial_sampling,pseudo_cloud_aug,contrast_utils,convert_mapping,generate_geo_metadata,visualization}.py` | shared utils |
| `build_loco_splits.py`(+sh,+submit), `generate_split.py` | LOCO split construction |
| `loco_evaluation.py`, `res_evaluation.py` | LOCO / cross-res evaluation |
| `aggregate_loco_results.py`, `aggregate_res_results.py` | result aggregation |
| `run_inference.py` | per-arch inference |
| `sp_gap_analysis.py` | 4.3 SP-gap (per arch) |
| `tempscale_eval.py` | §5.3 temp-scaling (per arch) |
| `inria_dataset.py`, `train_inria.py`, `mamnet_inria.sh`(+submit) | App. G.7 cross-task (INRIA) |
| `inference_annotation.py`, `annotate_pre_infer.sh`(+submit) | inference/annotation prep |
| `visualize_splits_spatial_metrics.py` | App. D split visualization |
| `inference.sh`(+submit), `test_loco.sh`(+submit), `test_res.sh`(+submit), `plot_loco_agg.sh`(+submit), `plots_res_test.sh`(+submit), `one_file_submit.sh` | run/eval/plot wrappers |

### KEEP — per-method (forked base files are expected)
| Method | Files |
|--------|-------|
| FDA | `data/fda_transform.py` (above), `mamnet_fda.sh`(+submit) |
| IIM | `train_iim.py`, `models/{iim,mamnet_iim}.py`, `mamnet_iim.sh`(+submit) |
| ISW | `train_isw.py`, `utils/isw_loss.py`, `utils/visualization_isw.py`, `compute_isw_masks.py`(+sh,+submit), `mamnet_isw.sh`(+submit) |
| MRFP+ | `train_mrfp.py`, `models/{mrfp_modules,mamnet_mrfp}.py`, `utils/losses_mrfp.py`, `mamnet_mrfp.sh`(+submit) |
| SegDeSiC | `train_segdesic.py`, `models/{segdesic_module,mamnet_segdesic}.py`, `utils/visualization_segdesic.py`, `mamnet_segdesic.sh`(+submit) |
| FADA | `train_fada.py`, `models/{fada,mamnet_fada}.py`, `mamnet_fada.sh`(+submit) |
| SIB (paper) | `train_mamnet_sib.py`, `models/{sib,mamnet_sib}.py`, `data/dataset_sib.py`, `eval_mamnet_sib.py`(+sh,+submit), `mamnet_sib.sh`(+submit), `mamnet_sib_ablation_submit.sh`, `mamnet_sib_sp_gap.sh`(+submit), `mamnet_sib_tempscale.sh`(+submit) |

### EXCLUDE → extra method
`ddib.py`, `models/{ddib,mamnet_ddib}.py`, `train_mamnet_ddib.py`, `data/dataset_ddib.py`, `mamnet_ddib.sh`(+submit) ·
`models/{gsdpe,mamnet_gsdpe}.py`, `train_gsdpe.py`, `data/dataset_gsdpe.py`, `mamnet_gsdpe.sh`(+submit) ·
`models/mamnet_hrda.py`, `train_hrda.py`, `data/dataset_hrda.py`, `utils/hrda_losses.py`, `mamnet_hrda.sh`(+submit) ·
`models/mamnet_mcl.py`, `train_mcl.py`, `utils/{contrastive_losses,visualization_mcl}.py`, `mamnet_mcl.sh`(+submit)

### EXCLUDE → cruft
`train_isw_debug.log`, `__pycache__/`, `models/__pycache__/`, `data/__pycache__/`, `utils/__pycache__/`

### ASK (mamnet)
- `train_finetuning.py`, `data/dataset_finetuning.py`, `analyze_finetune_results.py`, `loco_ft.sh`(+submit), `loco_ft_agg.sh`(+submit) → **D1**
- `mamnet_sib_newmod.sh`(+submit) → **D3**
- `check_job_status.py` → **D10**

---

## `oglanet/`

### KEEP — base & infra
`__init__.py`, `config.yaml`, `train.py`, `models/{oglanet,encoder,encoder_4ch,decoder,dffm,gfem,glam,oam,__init__}.py`,
`data/{dataset,dataset_enhanced,fda_transform}.py`,
`utils/{losses,metrics,utils,postprocessing,evaluation_detailed,geo_losses,contrast_utils,visualization}.py`,
`aggregate_loco_results.py`, `loco_evaluation.py`, `run_inference.py`, `sp_gap_analysis.py`, `tempscale_eval.py`,
`complete_eval.py`(+sh,+submit), `inference.sh`(+submit), `test_loco.sh`(+submit), `plot_loco_agg.sh`(+submit)

### KEEP — per-method
| Method | Files |
|--------|-------|
| FDA | `data/fda_transform.py` (above), `oglanet_fda.sh`(+submit) |
| IIM | `train_iim.py`, `models/{iim,oglanet_iim}.py`, `oglanet_iim.sh`(+submit) |
| ISW | `train_oglanet_isw.py`, `utils/isw_loss.py`, `utils/visualization_oglanet_isw.py`, `compute_isw_masks_oglanet.py`(+sh,+submit), `oglanet_isw.sh`(+submit) |
| MRFP+ | `train_oglanet_mrfp.py`, `models/{mrfp_modules,oglanet_mrfp}.py`, `utils/losses_oglanet_mrfp.py`, `oglanet_mrfp.sh`(+submit) |
| SegDeSiC | `train_segdesic.py`, `models/{segdesic_module,oglanet_segdesic}.py`, `utils/visualization_segdesic.py`, `oglanet_segdesic.sh`(+submit) |
| FADA | `train_fada.py`, `models/{fada,oglanet_fada}.py`, `oglanet_fada.sh`(+submit) |
| SIB (paper) | `train_oglanet_sib.py`, `models/{sib,oglanet_sib}.py`, `data/dataset_sib.py`, `oglanet_sib.sh`(+submit), `oglanet_sib_ablation_submit.sh`, `oglanet_sib_sp_gap.sh`(+submit) |

### EXCLUDE → extra method
`ddib.py`, `models/{ddib,oglanet_ddib}.py`, `train_oglanet_ddib.py`, `data/dataset_ddib.py`, `oglanet_ddib.sh`(+submit) ·
`models/{gsdpe,oglanet_gsdpe}.py`, `train_gsdpe.py`, `data/dataset_gsdpe.py`, `oglanet_gsdpe.sh`(+submit) ·
`models/oglanet_hrda.py`, `train_hrda.py`, `data/dataset_hrda.py`, `utils/hrda_losses.py`, `oglanet_hrda.sh`(+submit) ·
`models/oglanet_mcl.py`, `train_mcl.py`, `utils/contrastive_losses.py`, `oglanet_mcl.sh`(+submit)

### EXCLUDE → cruft
`train_oglanet_isw_debug.log`, all `__pycache__/`

### ASK (oglanet)
- `data/dataset_finetuning.py` → **D1**
- `oglanet_sib_newmod.sh`(+submit) → **D3**

---

## `dinov3/`  (my files only — upstream `dinov3/dinov3/` is vendored, see below)

### KEEP — base & infra
`__init__.py` (**scrub the `__author__` line → `'Anonymous'`**), `config.yaml`,
`dinov3_model.py`, `dinov3_backbone.py`, `dinov3_decoder.py`, `scale_attention_dinov3.py`,
`data/{dataset,fda_transform}.py`,
`utils/{losses,metrics,utils,postprocessing,evaluation_detailed,geo_losses,pseudo_cloud_aug,contrast_utils,visualization}.py`,
`aggregate_loco_results.py`, `loco_evaluation.py`, `run_inference.py`, `sp_gap_analysis.py`, `tempscale_eval.py`,
`train_dinov3.py` (Vanilla + FDA), `dinov3.sh`(+submit), `dinov3_fda.sh`(+submit),
`inference.sh`(+submit), `test_loco.sh`(+submit), `plot_loco_agg.sh`(+submit)

### KEEP — per-method
| Method | Files |
|--------|-------|
| IIM | `train_dinov3_iim.py`, `dinov3_iim_model.py`, `iim.py`, `dinov3_iim.sh`(+submit) |
| ISW | `train_dinov3_isw.py`, `utils/isw_loss_dinov3.py`, `utils/visualization_dinov3_isw.py`, `compute_isw_masks_dinov3.py`(+sh,+submit), `dinov3_isw.sh`(+submit) |
| MRFP+ | `train_dinov3_mrfp.py`, `dinov3_model_mrfp.py`, `mrfp_modules.py`, `utils/losses_mrfp.py`, `dinov3_mrfp.sh`(+submit) |
| SegDeSiC | `train_dinov3_segdesic.py`, `dinov3_segdesic.py`, `utils/visualization_segdesic.py`, `dinov3_segdesic.sh`(+submit) |
| FADA | `train_dinov3_fada.py`, `dinov3_fada.py`, `dinov3_model_fada.py`, `dinov3_fada.sh`(+submit) |
| SIB (paper) | `train_dinov3_sib.py`, `dinov3_model_sib.py`, `sib.py`, `data/dataset_sib.py`, `dinov3_sib.sh`(+submit), `dinov3_sib_ablation_submit.sh`, `dinov3_sib_sp_gap.sh`(+submit) |

### EXCLUDE → extra method
`ddib.py`, `dinov3_model_ddib.py`, `train_dinov3_ddib.py`, `data/dataset_ddib.py`, `dinov3_ddib.sh`(+submit) ·
`gsdpe.py`, `dinov3_gsdpe.py`, `dataset_gsdpe.py`, `data/dataset_gsdpe.py`, `train_dinov3_gsdpe.py`, `dinov3_gsdpe.sh`(+submit) ·
`dinov3_hrda.py`, `train_hrda_dinov3.py`, `data/dataset_hrda.py`, `utils/hrda_losses.py`, `dinov3_hrda.sh`(+submit) ·
`dinov3_mcl.py`, `train_dinov3_mcl.py`, `utils/contrastive_losses.py`, `dinov3_mcl.sh`(+submit)

### EXCLUDE → cruft
- `dinov3/dinov3/` — **vendored upstream facebookresearch/DINOv3** (incl. `.git/`, `.github/`, `dinov3.egg-info/`, `notebooks/`, `LICENSE.md`). **Do NOT redistribute** — see license note below.
- `weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (82 MB pretrained weights)
- `models_test/` (`dinov3_gsdpe.py`, `gsdpe.py`, `segdesic_module.py`, `init.py`) — scratch dir
- all `__pycache__/`

### EXCLUDE from repo, but LISTED so you can read them (possible identity / messy history)
`dinov3/CORRECTION_SUMMARY.md`, `dinov3/FINAL_SUMMARY.md`, `dinov3/IMPLEMENTATION_SUMMARY.md`,
`dinov3/QUICKSTART.md`, `dinov3/README_DINOv3.md`

### ASK (dinov3)
- `data/dataset_finetuning.py` → **D1**
- `dinov3_sib_newmod.sh`(+submit) → **D3**

> **DINOv3 license constraint (flag):** `dinov3/dinov3/LICENSE.md` §1.b.i requires that any
> redistribution of DINO Materials or derivatives be under the *same* agreement **with a copy
> of the agreement included**, and §1.b.ii requires acknowledging DINOv3 use in publications.
> Recommendation: do **not** vendor it; depend on upstream via pip/git-submodule and cite it.
> `dinov3_backbone.py` already imports it via `sys.path` → `from dinov3.models import vision_transformer`,
> so a submodule at the expected path works unchanged.

---

## `final_loco/`  (three-probe framework = 4.2/4.3/4.4)

### KEEP
| File | Maps to |
|------|---------|
| `experiment_c_histogram_match.py` | 4.2 pixel-space test (Probe C) |
| `experiment_a_decoder_retrain.py` | 4.4 decoder retrain (Exp A) |
| `experiment_b2_encoder_retrain.py` | 4.4 encoder retrain (Exp B) |
| `evaluate_experiments.py` | evaluator for Exp A/B/C |
| `experiment_utils.py`, `config.py`, `utils.py` | shared support |
| `run_experiments.sh`(+submit) | runs Exp A/B/C |
| `generate_plots.py`, `plot_experiment_results.py`, `plot_experiments.sh`(+submit) | experiment figures |

### ASK
| File | Decision |
|------|----------|
| `experiment_b_bn_swap.py` | **D5** (alt Exp B) |
| `evaluate_recovery_robust.py`, `robust_recovery/robust_recovery_results.json`, `robust_recovery/robust_recovery_full.json` | **D6** |
| `data_efficiency_submit.sh` | **D7** |
| `thread1_1d.py`, `thread1_entanglement.py`, `thread3_geometry.py`, `thread4_position.py`, `run_diagnostics.py`, `run_threads.sh`(+submit), `extract_features.py`, `run_extract.sh`(+submit), `compute_paper_statistics.py`, `run_paper_stats.sh`(+submit) | **D4** |

### EXCLUDE → cruft
`__pycache__/`, `robust_recovery/*.json` (results data — never to GitHub regardless)

---

## `final_loco/composition_analysis/`  → **EXCLUDE all (D8 — confirm)**

Brief: exploratory, not in paper. All default **EXCLUDE → exploratory**. Listed for your confirm:

`scene_composition_shared.py`, `run_composition_analysis.py`, `run_composition.sh`(+submit),
`f1_coverage_strat_gap.py`, `f2_shadow_v_bg.py`, `f3_select_predict_v_miou.py`,
`f4_arch_condition_method_rank.py`, `f5_cross_entropy_aurc.py`, `f6_coverage_recover_ub_perform.py`,
`f7_label_free_dist_grad.py`,
`s1_risk_coverage_select_predict_gap.py`, `s2_f1_logit_mean_decomp.py`, `s2_f2_reliability_diagrams.py`,
`s2_f3_thresh_robust_sweep.py`, `s2_f4_assym_diagnostics.py`, `s2_f5_pop_stratified_ece.py`,
`s2_f6_cross_city_transfer_class_cond_T.py`, `s2_f7_hist_bin_n_platt_scale.py`,
`s2_temp_scaling_inadequate.py`, `s3_dist_conf_scaling.py`,
`test1_size_recall.py`, `test1b_category_and_mechanism.py`, `test2_coverage_calibration.py`,
`test2v2_coverage_error.py`, `test2v3_followups.py`, `test3_1_confidence_distribution.py`,
`test3_1v2_affine_decomposition.py`, `test3_1v3_correlation_and_residual.py`,
`test3_1v4_tail_preservation.py`, `test3_count_recall.py`, `test4_permutation_fix.py`,
`test4v5_logit_analysis.py`, `test5_d1_rewrite.py`, `test6_d3_correction.py`, `test8_sib_reframe.py`,
`test_inria_sp_battery.py`, `test_shadow_vs_inria.py`, `test_tp_fp_shadow_v_inria.py`, `__pycache__/`

---

## `fig2_final/`  → **EXCLUDE → cruft (figure-output data)**
15 files (`col1`–`col5`/*.png, *.npy) — rendered Figure-2 tile data, not code.

---

## Identity scrubbing (summary)

All identifying tokens were removed from the copied code before release. Categories handled:
- An author string in one package `__init__.py` → replaced with `'Anonymous'`.
- Cluster usernames embedded in absolute paths → genericized.
- SLURM allocation/account and partition names → replaced with `<SLURM_ACCOUNT>` / `<SLURM_PARTITION>` placeholders.
- Absolute cluster paths (project / scratch / home trees) → replaced with `<PROJECT_ROOT>` or made relative.
- **No** wandb entity/project, API/HF tokens, SSH keys, or `--mail-user`/`--mail-type` found.
