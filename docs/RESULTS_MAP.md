# RESULTS_MAP — results JSON → paper element → producer → inputs

Every number in the camera-ready that comes from the post-submission analysis is read from a
results JSON `results/NN_<name>.json`. The JSONs themselves are **not** in this repository; each
names its producer in its `script` field, and this table maps them. Columns:

- **results id** — the JSON's `id` field (file `results/<id>.json`).
- **paper element** — camera-ready numbering (section / appendix / table / figure).
- **producer** — the script in this repo that writes the JSON (`analysis/`). Where the JSON's
  `script` field names a path that differs from the repo path, the recorded path is given too.
- **inputs** — what the producer reads, as `Delta-side writer → layout under $PROJECT_ROOT`.
  Delta-side writers are named by the filename recorded in the run logs; their repository paths
  are filled in when this branch is rebased on `camera-ready-delta` (marked *(Delta path TBD)*).
  Upstream results JSONs that a producer reads are listed as `results/NN`.

The layouts are described in [`analysis/README.md`](../analysis/README.md#input-layout). The
mapping was built from each JSON's `id`, `script` and `figure` fields, its headline, and a search
of the camera-ready LaTeX for the values it reports; no number is restated here.

## Cited by the camera-ready

| results id | paper element (camera-ready) | producer | inputs (Delta-side writer → layout) |
|---|---|---|---|
| `01_scaling_ladder` | App. H.1 training-set scale ladder (225/450/900) | `analysis/armc_ladder_stats.py` (recorded as `python/armc_ladder_stats.py`) | `armc_train.py` + `armc_dump.py` *(Delta path TBD)* → `dumps/armC/st/<run_id>/{perimage.parquet,run.json}` |
| `02_frozen_backbone_gap` | App. H.3 frozen DINOv3 encoder (gap) | `analysis/arme_frozen_gap_stats.py` (recorded as `python/arme_frozen_gap_stats.py`) | `arme_part1_train.py` + `arme_part1_dump.py` *(Delta path TBD)* → `dumps/st/dinov3frozen-*-r450-*-hr-s0/`; `results/01` |
| `04_seo_probspace` | §4.5, App. I (p̄_S / p̄_B on S-EO, MAMNet + DINOv3) | `analysis/04_seo_probspace.py` | `armA_phase3_train.py` *(Delta path TBD)* → `dumps/seo/<run_id>/{perimage.parquet,run.json,probs.npz}`; metric module `shadowtransfer/metrics.py`; `results/02_seo_loco` |
| `06_aurc_recovery` | §4.3 Figure 3(a) ΔAURC label; App. G.5 | `analysis/06_aurc_recovery.py` | `armc_r450.py` (CPU re-pass of published probability maps) *(Delta path TBD)* → `dumps/st_r450_published_recompute/`; `data/figure3_data/`, `data/Final_data_test/`; metric module |
| `07_sib_mechanism` | App. I "Decomposition of SIB's S-EO deficit" | `analysis/07_sib_mechanism.py` | `armA_phase3_train.py` → `dumps/seo/`; `armc_r450.py` → `dumps/st_r450_published_recompute/`; `logs/armA_phase3_manifest.json`; `data/data_comparisons/SEO/`; `shadowtransfer/armf_crossearth_cells.py` (FDA op); `results/03` |
| `08_seo_full_grid` | §4.5, App. I (S-EO vanilla grid, 4 archs × 3 cities × 3 seeds; method grid) | `analysis/08_seo_full_grid.py` | `armA_phase3_train.py`, `armA_b5_sar_probe.py` *(Delta path TBD)* → `delta_dumps/<tree>/<run_id>/`, `delta_logs/armA_b4_probe_inventory_final.json`, `delta_logs/armA_b5_t2_roster.json`; `results/02_seo_loco`, `results/03` |
| `08_seo_full_grid` (append) | App. I p̄_S over 12 pairs | `analysis/08b_seo_pbar.py` (appends a block to results/08) | as 08, plus `probs.npz`; imports `04_seo_probspace.py` and `08_seo_full_grid.py` |
| `09_seo_diagnostics` | App. I (failure-to-train / collapse / Exp A on S-EO) | `analysis/09_seo_diagnostics.py` | `delta_dumps/`; imports 08 and 08b; `results/08` |
| `10_seo_selective_prediction` | App. I selective-prediction gap on S-EO | `analysis/10_seo_selective_prediction.py` | `delta_dumps/` (coverage grid); imports 08; `results/06`, `results/08`, `results/09` |
| `13_armF_internal_crossearth_bsm_sar` | Table 2 CrossEarth / BSM / SAR rows (§3.4); App. F.22–F.24 | `analysis/13_armF_internal_crossearth_bsm_sar.py` | `armf_crossearth_train.py`, `armf_bsm_train.py`, `armf_sar_grid.py` *(Delta path TBD)* → `runs/armF_*/results.json`, `delta_runs/dffm_defect/provenance.json` |
| `14_appE_empty_foreground_rate` | App. E (empty-foreground rate, 117 of 900 at K=2) | `analysis/14_appE_empty_foreground_rate.py` | `runs/D_seedfloor/seed_floor.md`, `summaries/empty_denominator_rate.md` (promotion of a Delta audit) |
| `15_armD_appearance_augmentation` | App. H.2 appearance augmentation | `analysis/15_armD_appearance_augmentation_results.py` | `armd_train.py`, `armd_e200.py`, `armd_diag_aggregate.py` *(Delta path TBD)* → `runs/D_S3/*.json`, `runs/D_R/correlation.json`, `runs/D_seedfloor/results.json` |
| `16_armL_ssl_regime` | App. H.4 pretraining / fine-tuning regime (Exp B) | `analysis/16_armL_ssl_regime.py` | `armL_train.py` + probes *(Delta path TBD)* → `dumps/st/mamnetr50-*-r450-*-hr-s*[-probes]/`; `results/02_frozen_backbone_gap` |
| `16_armL_g2_expA` (Markdown) | App. H.4 Exp A 2×2 | `analysis/16_armL_g2_expA.py` (writes `results/16_armL_g2_expA.md`) | as 16; imports `16_armL_ssl_regime.py` |
| `19_e12_seed_table` | App. K (MAMNet-SIB cells are the E-12 FDA-recipe retrains) | `analysis/19_e12_seed_table.py` | E-12 retrain driver *(Delta path TBD)* → `delta_dumps/dumps/e12/<run_id>/` |
| `20_histogram_matching_counts` | §4.2, App. G.3; App. I appearance paragraph | `analysis/20_histogram_matching_counts.py` | `armc_probes.py` *(Delta path TBD)* → `logs/armC_probes_manifest.json`; `results/08`, `results/16_armL_ssl_regime` |
| `21_disclosure_record` | App. F.8 DFFM reimplementation defect; App. G.1 cell provenance | `analysis/21_disclosure_record.py` | Delta audits → `delta_runs/{dffm_defect,oglanet_nondeterminism,retrained_cells_record,train_count_audit*,mrfp_rerun_audit,test_results_metric_variant_audit}/` |
| `22_internal_sib_worst_case_v3` | App. K (published-budget adjudication; input to 25, 26, 31) | `analysis/22_internal_sib_worst_case_v3.py` | `armB_png_score.py` *(Delta path TBD)* → `dumps/st_r450_published_png/`; `dumps/st/` (CrossEarth, BSM); `delta_dumps/dumps/e12/`; `delta_runs/*` |
| `23_frozen_backbone_decoder_retrain` | App. H.3 frozen DINOv3 encoder (decoder retrain, R) | `analysis/23_frozen_backbone_decoder_retrain.py` | `arme_part2_train.py` *(Delta path TBD)* → `delta_runs/arme_part1/`, `delta_runs/arme_part2/`, `delta_runs/arme_part2_revised/` `results.json`; `results/02_frozen_backbone_gap` |
| `24_inria_crosstask` | App. G.7 INRIA cross-task replication | `analysis/24_inria_crosstask.py` | `run_inference_probs_inria.py` + the §4.3 battery → `data/figure3_data/inria_sp_battery_results.json`; `results/06` |
| `25_matched450_retrain_record` | App. G.1 cell provenance (450-image retrains); App. K | `analysis/25_matched450_retrain_record.py` | `n450_train.py` + `n450_score.py` *(Delta path TBD)* → `dumps/st_r450_published_recompute/*-s42-n450[-bs4]/`; `dumps/st_r450_published_png/`; `results/01`, `results/21`, `results/22` |
| `30_oglanet_rng_dependence` | App. F.8 / App. G.1 OGLANet inference nondeterminism | `analysis/30_oglanet_rng_dependence.py` | `oglanet_rng_probe_report.py` *(Delta path TBD)* → `delta_runs/oglanet_rng_probe/`; `results/21`, `results/27` |
| `31_internal_sib_worst_case_tgtfda` | §5 Results "Gap to the upper bound"; App. K table | `analysis/31_internal_sib_worst_case_tgtfda.py` | as 22 plus `n450_tgtfda.py` → `dumps/st_r450_published_recompute/oglanet-sibc4tgt-loco-r450-*-hr-s42-n450/`, `delta_runs/armF_sar_mamnet_*_highres-n450/`; `results/22`, `results/26`, `results/27` |
| `32_method_comparison_profile_tgtfda` | App. K method profile; Figure 4(b) | `analysis/32_method_comparison_profile_tgtfda.py` | imports `31_internal_sib_worst_case_tgtfda.py` (row builder); `results/28` |
| `33_m11_holm_tgtfda` | App. K (M11 Holm family) | `analysis/33_m11_holm_tgtfda.py` | `results/32`, `results/29` |
| `34_seo_sib_family_tgtfda` | App. I (SIB on S-EO, held-out-city FDA pool) | `analysis/34_seo_sib_family_tgtfda.py` | `armA_phase3_train.py` (tgtfda cells) → `dumps/seo/oglanet-sib-loco-r3500-*-hr-s0-tgtfda/`; imports `08_seo_full_grid.py`; `results/08`, `results/30` |
| `35_table2_final` | Table 2 at K=2 (§3.4); Figure 1 chart values (`figure1_data`) | `analysis/35_table2_final.py` | `dumps/st_r450_published_png/`, `dumps/st_r450_published_recompute/`, `dumps/st/`, `delta_runs/midres_png_score/`; `data/model_results/statistical_results.csv`; camera-ready LaTeX via `git show` (see README); imports 31; `results/01` |
| `38_table2_final_v2` | Table 2 at K=2 incl. CrossEarth SAR (§3.4) | `analysis/38_table2_final_v2.py` | as 35 plus `delta_runs/armF_sar_crossearth_*`; imports 31 and 35 |
| `39_table2_vs_vanilla_v2` | §3.4 methods vs Vanilla; App. G.2 pooled summaries | `analysis/39_table2_vs_vanilla_v2.py` | imports 38; `results/36`, `results/38` |
| `40_loco_all_cells_v2` | App. G.1 per-cell tables, App. G.2 pooled summaries | `analysis/40_loco_all_cells_v2.py` | imports 38; `results/37`, `results/38` |
| `41_figure1_ci` | Figure 1 error bars (95% CIs) | `analysis/41_figure1_ci.py` | imports 35; `results/35`, `results/38` |
| `42_sib_ablations_k2` | App. L SIB component ablations at K=2 | `analysis/42_sib_ablations_k2.py` | SIB ablation chain → `dumps/st_sib_ablation_png/`, `delta_runs/sib_ablation_k2/`; `handoff/appendixJ_printed_values.json`; imports 31; `results/41` |
| `43_provenance_backfill` | App. J SIB implementation (hyperparameter provenance); Figure 3 "~24×" label | `analysis/43_provenance_backfill.py` | `dumps/st*`, `delta_dumps/dumps/e12/`, `data/figure3_data/`, run logs; imports 31 |

## Read by a cited producer (superseded or upstream; shipped so the chain runs)

| results id | role | producer | read by |
|---|---|---|---|
| `02_seo_loco` | superseded by 08 | `analysis/02_seo_loco.py` | 04, 08 |
| `03_seo_methods` | superseded by 08 | `analysis/03_seo_methods.py` | 07, 08 |
| `26_internal_sib_worst_case_n450` | superseded by 27 → 31 | `analysis/26_internal_sib_worst_case_n450.py` | 25, 27, 31 |
| `27_internal_sib_worst_case_n450_sar` | superseded by 31 | `analysis/27_internal_sib_worst_case_n450_sar.py` | 28, 30, 31 |
| `28_method_comparison_profile` | superseded by 32 | `analysis/28_method_comparison_profile.py` | 29, 32 |
| `29_m11_holm` | superseded by 33 | `analysis/29_m11_holm.py` | 33 |
| `36_table2_vs_vanilla` | superseded by 39 | `analysis/36_table2_vs_vanilla.py` | 39 |
| `37_loco_all_cells` | superseded by 40 | `analysis/37_loco_all_cells.py` | 40 |

## Not cited in the camera-ready text, shipped by request

| results id | role | producer | inputs |
|---|---|---|---|
| `17_gj_error_taxonomy` | failure characterisation (error type, category, width); no camera-ready section prints it | `analysis/gj_analysis/gj_analysis.py` | `dumps/st/{arch}-{vanilla-loco,none-upper}-r450-{city}-hr-s0/`, `data/Final_data_test/`, `analysis/category_legend.json` |
| `17_gj_error_taxonomy_ext` | whole-blob miss rate (extends 17) | `analysis/gj_analysis/gj_blob_miss.py` | as 17; metric module; `results/17_gj_error_taxonomy` |

## Not released

| results id | reason |
|---|---|
| `05_internal_pbar` | not cited in the camera-ready (the printed p̄_S figures are S-EO, from 04/08b) and no released producer reads it |
| `11_internal_sib_worst_case` | superseded by 12 → 22; GATE_FAILED record with no number; no released producer reads it |
| `12_internal_sib_worst_case_v2` | superseded by 22; no released producer reads it |

## Figures

| camera-ready figure | script | reads |
|---|---|---|
| Figure 1 (teaser), right panel | `analysis/fig01/fig01_teaser.py` (+ `_helpers.py`) | `results/35_table2_final.json` (`figure1_data`), `results/41_figure1_ci.json` (CIs, first argument); tiles from `data/Final_data_test/` |
| Figure 3 (diagnostics), two labels | `analysis/fig03/fig03_relabel.py` | the submitted composite JPEG (argument); values from `results/06_aurc_recovery.json` (ΔAURC +0.26) and `results/43_provenance_backfill.json` item 2b (~24×) |
| Figure 4(b) (SIB method profile) | `analysis/fig04/fig04b_profile.py` | `results/32_method_comparison_profile_tgtfda.json`; panel (a) is cropped from the submitted `fig4_final.001.jpeg` |

Figure 2 (qualitative) is produced by `select_figure2_tiles.py` at the repo root and does not
read a results JSON. The dataset figures (annotation pipeline, per-city statistics) are not
generated from results JSONs. Colour assignments shared by the figures are in
[`docs/colors.md`](colors.md).
