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

---

## `analysis/` and `docs/` — camera-ready results producers (added for the camera-ready)

Source: the analysis project that turned prediction dumps into the results JSONs cited by the
camera-ready. One script per results JSON, named by the JSON's `script` field. Absolute paths were
replaced by `$PROJECT_ROOT`-relative ones and host names removed; computation is unchanged. The
per-script inputs and paper elements are in [`analysis/README.md`](analysis/README.md) and
[`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md).

| File(s) | Classification |
|---------|----------------|
| `analysis/armc_ladder_stats.py`, `analysis/arme_frozen_gap_stats.py` | KEEP → App. H.1, H.3 (recorded in their JSONs as `python/…`) |
| `analysis/02_seo_loco.py` … `analysis/43_provenance_backfill.py` (39 files) | KEEP → producers of results 02–43 cited by the camera-ready, plus the superseded producers 02, 03, 26–29, 36, 37 whose JSON a cited producer reads |
| `analysis/_promote_lib.py` | KEEP → shared results-envelope writer (13–15, 19–21, 30) |
| `analysis/gj_analysis/gj_analysis.py`, `gj_blob_miss.py` | KEEP → results/17 (failure characterisation) |
| `analysis/fig01/fig01_teaser.py`, `_helpers.py` | KEEP → Figure 1 |
| `analysis/fig03/fig03_relabel.py` | KEEP → Figure 3 (two relabelled values) |
| `analysis/fig04/fig04b_profile.py` | KEEP → Figure 4(b) |
| `analysis/README.md`, `docs/RESULTS_MAP.md` | NEW → documentation |
| `docs/colors.md` | KEEP → figure colour assignments (was `plots/colors.md`) |
| `05_internal_pbar.py`, `05a_build_internal_index.py` | EXCLUDE → not cited; nothing released reads results/05 |
| `08c_probe_sidecar_repairs.py` | EXCLUDE → one-off repair of dump sidecars on the analysis host, not an analysis |
| `11_internal_sib_worst_case.py`, `12_internal_sib_worst_case_v2.py` | EXCLUDE → superseded by 22, not read by any released producer |
| `15_armD_appearance_augmentation.py`, `16_armL_recovery_locus.py`, `17_gj_error_taxonomy.py`, `17_gj_failure_panels.py`, `18_gj_error_axes.py`, `18b_gj_category_table.py`, `gj_analysis/gj_width_bins.py`, `gj_analysis/gj_figure_support_dinov3.py` | EXCLUDE → figures not used in the camera-ready |
| `gj_fill_pass.py`, `gj_inspect_inputs.py`, `gj_iou_mismatch_*.py`, `gj_threshold_diagnostic.py` | EXCLUDE → one-off diagnostics; no results JSON names them |
| `fig01/orig_fig1_*.py`, `fig01/_glyph_src.py`, `fig04/orig_fig4_right.ipynb` | EXCLUDE → copies of the submitted-version figure code kept as provenance |
| `_cache/`, `_inv/`, `out/`, `*.parquet`, `results/`, `plots/*.png`, `fig04/fig4_sib.png` | EXCLUDE → data and outputs |

---

# Camera-ready additions

Added for the NeurIPS 2026 camera-ready so that the code behind every reported
number is released. Source tree: the cluster `python/` tree the paper's runs
were produced from. Same KEEP / EXCLUDE discipline as the June pass.

## `shadowtransfer/`  → **KEEP** (metric package + shared run-time modules)

31 files. Imported as `shadowtransfer`; `ST_METRICS_ROOT` defaults to the
repository root. `metrics.py` ships byte-identical at sha256
`730ad4a5bc039e3151599fb86bd411e2816c0897`, which `analysis/07_sib_mechanism.py`
and `analysis/gj_analysis/gj_blob_miss.py` both verify before use. Includes the
CrossEarth / BSM / SAR / matched-450 runtime modules and four cluster
infrastructure modules (`ledger.py`, `worker_contract.py`, `disk_gate.py`,
`fortress_recall.py`) released so the submit wrappers are complete. Per-file
table in `shadowtransfer/README.md`.

## `crossearth/`  → **KEEP** (App. F.22)

4 files. Overlay around upstream VisionXLab/CrossEarth, pinned at
`644a5a1b3c01b2e5531820b5291d4397597f75de` in `upstream_pin.json`. Upstream is
**not vendored**; clone instructions and the seven-item deviations table are in
that file and in the README.

## `sar/`  → **KEEP** (App. F.24)

3 files. SAR test-time adaptation on the matched-450 models. The submit
wrapper's preflight is site-specific; see `sar/README.md`.

## `bsm/`  → **KEEP** (App. F.23)

1 file. Our BSM reimplementation; no official implementation exists. AdaIN
style-network weights are downloaded, not committed.

## `matched450/`  → **KEEP** (§3.3, Table 2)

4 files. Matched-450 retrains, their scorer, the gate/manifest logic, and the
SIB noise cells.

## `appendix_h/`  → **KEEP** (App. H.1–H.4)

8 files. Scale ladder, appearance augmentation, frozen DINOv3, and
pretraining/fine-tuning regime drivers. The statistics scripts for H.1 and H.3
are released by the analysis pass under `analysis/`.

## `scoring/`  → **KEEP** (App. E)

1 file. K=2 boundary-tolerant scoring of saved prediction PNGs; the documented
App. E path. `scoring/README.md` states the K and variant of every other
evaluator in the repository.

## `probes/`  → **KEEP** (App. F.8)

2 files. OGLANet inference nondeterminism probe and its report.

## `support/`  → **KEEP** (imported by the above)

6 files. Not entry points; released because a released driver imports them.

## EXCLUDED from this pass

| Path | Reason |
|---|---|
| checkpoints (`checkpoint_best.pth`, 144 published cells, 58.58 GiB) | inventory only, per the brief; the user decides whether these go to a model repository |
| prediction dumps (`perimage.parquet`), `Test_img_probs/*.npy` | data, not code; over the size limit |
| `armA_*` block diagnostics, `armT*` harness probes, `arm[BCEFJ]` variant drivers not behind a camera-ready number | internal scaffolding; listed as ASK in the session record rather than released silently |
| `arme_frozen_gap_stats.py`, `armc_ladder_stats.py` | released under `analysis/` by the analysis pass |
| arm-A modules reachable only from unused functions of `support/trainaudit_sar_drift.py` | not camera-ready code; those functions raise `ImportError` in this release, which `support/README.md` states |

## Identity scrubbing (camera-ready pass)

The scrub grep specified in the release brief (cluster allocation and user
names, absolute filesystem roots, HF and GitHub token prefixes, and the
document-host identifiers) returns empty on this branch. 42 substitutions were applied across 16 files with
asserted anchors: cluster roots → `${PROJECT_ROOT}` / `${ST_SCRATCH_ROOT}`,
SLURM account → `<SLURM_ACCOUNT>`, alternate-cluster blocks genericized, and the
analysis-host name removed from two comment strings. Five new environment
variables are documented in `.env.example`.

---

# Camera-ready additions, second pass

Applied the operator's release rule: **a script ships if any run it produced
appears in the runs/inputs of a results JSON listed in
`docs/RESULTS_MAP.md`; otherwise it stays out and is listed here.**

## Released under the rule

| Script | Released to | Cited by |
|---|---|---|
| `armA_phase3_train.py` + worker + submit | `seo/` | results/04, 07, 08, 08b, 09, 10, 34 |
| `armA_b5_sar_probe.py` + worker + submit | `seo/` | results/08 |
| `armA_b4_probe_inventory.py` + worker + submit | `seo/` | results/08 (`armA_b4_probe_inventory_final.json`) |
| `armA_b5_t2_roster.py` | `seo/` | results/08 (`armA_b5_t2_roster.json`) |
| `arme12_retrain.py` + worker + submit | `sib/` | results/19 |
| `armJ_sib_ablation_score.py` + worker + submit | `sib/` | results/42 |
| `train_oglanet_sib_tgtfda.py` | `sib/` | results/31 (imported by `n450_tgtfda.py`) |
| `n450_tgtfda.py` + worker + submit | `matched450/` | results/31–43, which include Table 2 |
| `n450_tgtfda_donor.py` | `matched450/` | results/31 (hashed by `n450_tgtfda.py`) |
| `n450_bs4.py` | `matched450/` | results/25 (the `-bs4` cells) |
| `armB_png_score_midres63.py` + worker + submit | `scoring/` | results/35, 38 (`midres_png_score`) |
| `armf_sar_crossearth.py` + worker + submit | `sar/` | results/38 |
| `armd_e200.py`, `armd_diag_aggregate.py` + wrappers | `appendix_h/` | results/15 |
| `arme_part2_train.py` + submit | `appendix_h/` | results/23 |

Closure dependencies pulled in by the above: `shadowtransfer/`
`armA_a4_sar_seo.py`, `armA_b5_sar.py`, `armA_methods.py`,
`armA_t1_cesar_adapt.py`, `armJ_png_only.py`, `armd_diag_lib.py`,
`arme_part2_runtime.py`, `seo_budget.py`, `seo_crossearth.py`,
`seo_dataset.py`, `seo_dump.py`, `seo_traineval.py`; `support/`
`armA_pilot_train.py`, `armA_seo_histmatch.py`, `armd_diag.py`.

Two citations are by **output path**, not filename, and the rule still selects
them: `results/25` names `*-s42-n450[-bs4]/` (so `n450_bs4.py`) and `results/08`
names two `delta_logs/*.json` files (so the two scripts that write them).
`armf_sar_n450.py`, released in the first pass, is likewise cited only as
`delta_runs/armF_sar_mamnet_*_highres-n450/` in `results/31`.

## Left out under the rule

No run these produced is an input to any results JSON in `docs/RESULTS_MAP.md`:

| Script | Note |
|---|---|
| `armB_png_score_std18.py`, `armB_png_score_vanilla9.py` | only the midres-63 re-score is cited (results/35, 38) |
| `n450_tgtfda_report.py` | reporting helper; no cited JSON reads its output |
| `n450_lists.py`, `n450_sar_rerun.py`, `n450_isw_submit.sh` | matched-450 scaffolding |
| `armA_b1_segdesic*`, `armA_b2_isw*`, `armA_b3_ce*`, `armA_b4_*` other than the inventory, `armA_b5_*` other than the probe/roster | arm-A block diagnostics |
| `armA_wave0_*`, `armA_wave1_*`, `armA_wave2_*` | superseded S-EO waves |
| `armT2_*`, `armT3_*`, `armT4_*` | harness-factor and nondeterminism probes; `results/21` reads the promoted `delta_runs/` audits, not these scripts |
| `armL_dinov3_p5drift.py`, `arme_e0_layerscale.py`, `arme_e1_drift.py` | drift and layerscale side probes |
| `armc_r450_rescore.py`, `armf_sar_grid.py`, `armf_sar_naming_patch.py`, `oglanet_probsource.py`, `n450_isw_masks.py`, `trainaudit_sar_drift.py` | shipped in `support/` as dependencies, not as entry points |
| `armJ_sib_ablation_provenance_gate.py`, `armd_s3_aggregate.py`, `armd_levels.py`, `armf_sar_lr_pilot.py`, `armf_sar_pilot.py`, `armc_probes_rawpixel.py`, `armA_t1_cesar_run.py` | named only in recorded provenance strings; no cited JSON reads their output. Those strings keep their original `python/...` spelling, which is the Delta-side path, since the file is not in this repository |

## Checkpoints

`docs/CHECKPOINT_INVENTORY.json` now carries a `method_families` block beside
the 144 published cells: **33 entries, 29 present (9.41 GiB), 4 absent.**
Inventory only; nothing is uploaded.

- **SIB** — mamnet C4clean LOCO (3), oglanet matched-450 `sibc4` (3,
  `source_only`) and `sibc4tgt` (3, `unlabeled_target`). The oglanet cells are
  under the scratch `n450` tree, not the LOCO tree.
- **BSM** — mamnet (3), oglanet (3), dinov3 (2), crossearth (3).
- **CrossEarth** — vanilla (3), fda (3), sib (3), under `armF/`.
- **SAR** — **none by design.** SAR trains nothing and writes no checkpoint; it
  adapts a parent episodically and discards the adapted weights. The block names
  the parent instead.
- **Absent and listed, not omitted:** dinov3 SIB for all three cities (the run
  dirs hold `args.json`, predictions and `test_results.json` but no `.pth`) and
  dinov3 BSM chicago.

Probing followed symlinks: `data/{arch}/outputs/{run_dir}` on the project root
are symlinks into the scratch root, and a scan that does not follow them reports
zero checkpoints.

## Module shas

`shadowtransfer/metrics.py` ships byte-identical (sha256 `730ad4a5...ab8a6e83`);
`analysis/07`, `analysis/10` and `analysis/gj_analysis/gj_blob_miss.py` all
hard-check it. Every other module in `shadowtransfer/` was repointed for the
release layout, so its sha256 computed here will not match the value recorded in
a published run's `run.json`. That difference is the path rewrite, not a change
in behaviour. During this pass the layout rewrite did briefly alter `metrics.py`
provenance comments; it was restored byte-identical and re-verified against all
three consumers before the push.

## Metric wording

The paper's metric is the **two-class** boundary-tolerant mIoU at K=2 (App. E).
Where a scorer also emits a foreground-only column at the same band, that column
is the scorer **self-check** (results/31, caveat d) and carries shadow-class
statements only. `matched450/README.md` and `scoring/README.md` state this. No
code and no number changed.
