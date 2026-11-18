# analysis/ — results producers and figure scripts

Every camera-ready number from the post-submission analysis is read from a results JSON
`results/NN_<name>.json`, and every results JSON names the script that wrote it in its `script`
field. This directory holds those scripts. [`docs/RESULTS_MAP.md`](../docs/RESULTS_MAP.md) maps
each JSON to its camera-ready section, table or figure.

The scripts are CPU-only. They read prediction dumps, run outputs and earlier results JSONs; they
do not train or run inference. The dumps are written by the training, inference and scoring code
elsewhere in this repository (see the *inputs* column of `docs/RESULTS_MAP.md`); eight input
writers are not released and are listed there. Neither the dumps nor the
results JSONs are distributed here.

## Configuration

| variable | required | meaning |
|---|---|---|
| `PROJECT_ROOT` | yes | root of the data/output tree described in [Input layout](#input-layout). Set it via `.env` (see `.env.example`). |
| `ST_METRICS_ROOT` | no | directory that contains the `shadowtransfer/` package (the K=2 metric module [`shadowtransfer/metrics.py`](../shadowtransfer/metrics.py)) and its sibling `mamnet/` (for `mamnet/utils/contrast_utils.py`). Defaults to the repository root, where both ship. |
| `ST_FORBIDDEN_OUT_PREFIX` | no | `30_oglanet_rng_dependence.py` refuses to write under this prefix. Unset means no guard. |

The metric module is imported, never reimplemented. Several producers check its sha256 before
use (`07_sib_mechanism.py` and `gj_analysis/gj_blob_miss.py` require `730ad4a5…`, the revision
recorded in the S-EO and arm C/G/J dumps); a different revision stops them with a message.

## Running

```bash
source .env                                   # sets PROJECT_ROOT
python analysis/31_internal_sib_worst_case_tgtfda.py --dry-run   # resolve inputs, write nothing
python analysis/31_internal_sib_worst_case_tgtfda.py             # full pass, writes results/31_*.json
```

Every producer accepts `--dry-run`, which prints the resolved inputs (dump schema, cells selected,
row counts) and writes nothing. In the arm L scripts (`16_*`) `--dry-run` sets `STAGE = "dry"`,
which still runs the analysis on one corner (`DRY_CORNERS`) and prints it. `armc_ladder_stats.py`
and `arme_frozen_gap_stats.py` also take `--help`. The remaining configuration (cell selection,
`B`, seeds) is a block of constants at the top of each script.

Producers must be run in dependency order when rebuilding from scratch, because many read earlier
results JSONs (listed per script below). The full order is the numeric order, with
`armc_ladder_stats.py` (→ 01) and `arme_frozen_gap_stats.py` (→ 02_frozen_backbone_gap) first and
`08b_seo_pbar.py` immediately after `08_seo_full_grid.py`.

## Input layout

All paths are relative to `$PROJECT_ROOT`.

| path | contents | written by |
|---|---|---|
| `dumps/<tree>/<run_id>/perimage.parquet` + `run.json` | one row per test image: raw `tp/fp/fn/tn` counts (tolerant and strict), per-image IoU, AURC fields, category pixel counts; `run.json` records the run's config, `config_sha256`, `target_access` and the metric module sha | the dump writers (below) |
| `dumps/<tree>/<run_id>/probs.npz` | per-image probability maps (S-EO and probe cells only) | as above |
| `dumps/<tree>/INDEX.parquet` | index of a tree, rebuilt after every copy; producers select cells from it, never from a glob | the producer's own index step or the copy step |
| `delta_dumps/<tree>/…` | the same dump format, for trees copied in bulk (S-EO method grid, E-12 retrains) | the dump writers (below) |
| `delta_runs/<audit>/results.json`, `runs/<arm>/…` | small per-run summaries from Delta audits and training arms | the drivers in `appendix_h/`, `crossearth/`, `bsm/`, `sar/`, `matched450/`, `probes/` |
| `logs/*.json`, `delta_logs/*.json` | manifests and rosters that list which cells exist | the same drivers |
| `results/NN_<name>.json` | outputs of this directory; also inputs to later producers | this directory |
| `analysis/_inv/`, `analysis/_cache/`, `analysis/out/` | tree indexes cited by sha256, caches, scratch outputs | this directory |
| `data/Final_data_test/`, `data/figure3_data/`, `data/model_results/`, `data/data_comparisons/` | dataset tree, submitted-figure data, per-cell CSVs | dataset release and the submission-era scripts |
| `final_paper/` | a git checkout of the paper's LaTeX source | not distributed; see Limitations |

Dump trees and the scripts that wrote them. A writer marked *(not released)* is named by the
filename in the run logs and is not in this repository ([`docs/RESULTS_MAP.md`](../docs/RESULTS_MAP.md#input-writers-not-in-this-repository)):

| tree | cells | writer |
|---|---|---|
| `dumps/armC/st` | arm C scaling ladder (225/450/900) | `appendix_h/armc_train.py`, `shadowtransfer/armc_dump.py` |
| `dumps/st` | arm C probes, arm L (MAMNet-R50 SSL regime), arm E (frozen DINOv3), arm F (BSM, CrossEarth) | `armc_probes.py` *(not released)* (ops in `shadowtransfer/armc_probes_*.py`), `appendix_h/armL_train.py`, `appendix_h/arme_part1_train.py` + `shadowtransfer/arme_part1_dump.py`, `bsm/armf_bsm_train.py`, `crossearth/armf_crossearth_train.py` |
| `dumps/st_r450_published_recompute` | CPU re-pass over the published cells' stored probability maps; matched-450 retrains (`*-s42-n450[-bs4]`); OGLANet-SIB held-out-city FDA retrains (`oglanet-sibc4tgt-*`) | `appendix_h/armc_r450.py`; `matched450/n450_train.py` + `matched450/n450_score.py`; `n450_tgtfda.py` *(not released)* |
| `dumps/st_r450_published_png` | published cells scored from their saved prediction PNGs | `scoring/armB_png_score.py` |
| `dumps/st_sib_ablation_png` | SIB ablation cells scored from PNGs | SIB ablation K=2 chain *(not released)* (`delta_runs/sib_ablation_k2/`) |
| `dumps/seo`, `delta_dumps/…/seo` | S-EO external benchmark (vanilla grid, method grid, SIB `-tgtfda`) | `armA_phase3_train.py`, `armA_b5_sar_probe.py` *(not released)* |
| `delta_dumps/dumps/e12` | MAMNet-SIB FDA-recipe retrains (seeds 42–44) | E-12 retrain driver *(not released)* |

## Producers

`reads` lists inputs under `$PROJECT_ROOT` plus earlier results JSONs; `imports` lists sibling
scripts loaded with `importlib`. "Paper" uses camera-ready numbering.

| script | writes | reads | imports | paper |
|---|---|---|---|---|
| `armc_ladder_stats.py` | `results/01_scaling_ladder.json` | `dumps/armC/st` | — | App. H.1 |
| `arme_frozen_gap_stats.py` | `results/02_frozen_backbone_gap.json` | `dumps/st` (`dinov3frozen-*`); `results/01` | — | App. H.3 |
| `02_seo_loco.py` | `results/02_seo_loco.json` | `dumps/seo` | — | superseded by 08; read by 04, 08 |
| `03_seo_methods.py` | `results/03_seo_methods.json` | `dumps/seo`, `logs/armA_phase3_manifest.json` | — | superseded by 08; read by 07, 08 |
| `04_seo_probspace.py` | `results/04_seo_probspace.json` | `dumps/seo` (`probs.npz`); `results/02_seo_loco` | metric module | §4.5, App. I |
| `06_aurc_recovery.py` | `results/06_aurc_recovery.json` | `dumps/st_r450_published_recompute`, `data/figure3_data`, `data/Final_data_test`, `Python/FINAL-fig3.ipynb` (hashed) | metric module | §4.3 Fig. 3(a), App. G.5 |
| `07_sib_mechanism.py` | `results/07_sib_mechanism.json` | `dumps/seo`, `dumps/st_r450_published_recompute`, `logs/armA_phase3_manifest.json`, `data/data_comparisons/SEO`, `data/Final_data_test`; `results/03` | metric module, `shadowtransfer/armf_crossearth_cells.py` | App. I |
| `08_seo_full_grid.py` | `results/08_seo_full_grid.json` | `delta_dumps`, `delta_logs`; `results/02_seo_loco`, `results/03` | metric module (hashed) | §4.5, App. I |
| `08b_seo_pbar.py` | appends a p̄ block to `results/08_seo_full_grid.json` (backs it up first) | as 08, `probs.npz` | 04, 08 | App. I |
| `09_seo_diagnostics.py` | `results/09_seo_diagnostics.json` | `delta_dumps`; `results/08` | 08, 08b | App. I |
| `10_seo_selective_prediction.py` | `results/10_seo_selective_prediction.json` | `delta_dumps`; `results/06`, `results/08`, `results/09` | 08, metric module | App. I |
| `13_armF_internal_crossearth_bsm_sar.py` | `results/13_…json` | `runs/armF_*`, `delta_runs/dffm_defect` | `_promote_lib` | Table 2 (CrossEarth/BSM/SAR), App. F.22–F.24 |
| `14_appE_empty_foreground_rate.py` | `results/14_…json` | `runs/D_seedfloor`, `summaries/empty_denominator_rate.md` | `_promote_lib` | App. E |
| `15_armD_appearance_augmentation_results.py` | `results/15_…json` | `runs/D_S3`, `runs/D_R`, `runs/D_seedfloor` | `_promote_lib` | App. H.2 |
| `16_armL_ssl_regime.py` | `results/16_armL_ssl_regime.json` (+ `results/16_armL_summary.md`) | `dumps/st` (`mamnetr50-*`); `results/02_frozen_backbone_gap` | — | App. H.4 |
| `16_armL_g2_expA.py` | `results/16_armL_g2_expA.md` | `dumps/st`; `results/16_armL_ssl_regime` | 16_armL_ssl_regime | App. H.4 (Exp A) |
| `gj_analysis/gj_analysis.py` | `results/17_gj_error_taxonomy.json` | `dumps/st`, `data/Final_data_test`, `analysis/category_legend.json`, `analysis/fill_pass_report.json` | — | not printed in the camera-ready |
| `gj_analysis/gj_blob_miss.py` | `results/17_gj_error_taxonomy_ext.json` | as above; `results/17` | metric module | not printed in the camera-ready |
| `19_e12_seed_table.py` | `results/19_e12_seed_table.json` | `delta_dumps/dumps/e12` | `_promote_lib` | App. K |
| `20_histogram_matching_counts.py` | `results/20_…json` | `logs/armC_probes_manifest.json`; `results/08`, `results/16_armL_ssl_regime` | `_promote_lib` | §4.2, App. G.3, App. I |
| `21_disclosure_record.py` | `results/21_disclosure_record.json` | `delta_runs/{dffm_defect,oglanet_nondeterminism,retrained_cells_record,train_count_audit*,mrfp_rerun_audit,test_results_metric_variant_audit}` | `_promote_lib` | App. F.8, App. G.1 |
| `22_internal_sib_worst_case_v3.py` | `results/22_…json`, `analysis/_inv/INDEX_*_v3.parquet` | `dumps/st_r450_published_png`, `dumps/st`, `delta_dumps/dumps/e12`, `delta_runs/*` | — | App. K (input to 25, 26, 31) |
| `23_frozen_backbone_decoder_retrain.py` | `results/23_…json` | `delta_runs/arme_part1`, `delta_runs/arme_part2`, `delta_runs/arme_part2_revised`; `results/02_frozen_backbone_gap` | — | App. H.3 |
| `24_inria_crosstask.py` | `results/24_inria_crosstask.json` | `data/figure3_data/inria_sp_battery_results.json`, `final_paper/` text; `results/06` | — | App. G.7 |
| `25_matched450_retrain_record.py` | `results/25_…json` | `dumps/st_r450_published_recompute` (`*-n450*`), `dumps/st_r450_published_png`; `results/01`, `21`, `22` | — | App. G.1, App. K |
| `26_internal_sib_worst_case_n450.py` | `results/26_…json` | as 22 plus n450 cells; `results/22`, `25` | — | superseded by 27 → 31; read by 25, 27, 31 |
| `27_internal_sib_worst_case_n450_sar.py` | `results/27_…json` | as 26 plus `delta_runs/armF_sar_mamnet_*_highres-n450` (written by `sar/armf_sar_n450.py`); `results/26` | — | superseded by 31; read by 28, 30, 31 |
| `28_method_comparison_profile.py` | `results/28_…json` | `results/22`, `26`, `27` | 27 | superseded by 32; read by 29, 32 |
| `29_m11_holm.py` | `results/29_m11_holm.json` | `results/28` | — | superseded by 33; read by 33 |
| `30_oglanet_rng_dependence.py` | `results/30_…json` | `delta_runs/oglanet_rng_probe`; `results/21`, `27` | `_promote_lib` | App. F.8, App. G.1 |
| `31_internal_sib_worst_case_tgtfda.py` | `results/31_…json`, tree indexes | as 27 plus `oglanet-sibc4tgt-*` cells; `results/22`, `26`, `27` | — | §5, App. K |
| `32_method_comparison_profile_tgtfda.py` | `results/32_…json` | `results/28`, `31` | 31 | App. K, Fig. 4(b) |
| `33_m11_holm_tgtfda.py` | `results/33_m11_holm_tgtfda.json` | `results/29`, `32` | — | App. K |
| `34_seo_sib_family_tgtfda.py` | `results/34_…json` | `dumps/seo` (`*-tgtfda`), `dumps/st_r450_published_recompute`; `results/08`, `30` | 08 | App. I |
| `35_table2_final.py` | `results/35_table2_final.json` | `dumps/st_r450_published_png`, `dumps/st_r450_published_recompute`, `dumps/st`, `delta_runs/midres_png_score`, `data/model_results/statistical_results.csv`, `logs/armC_probes_manifest.json`, `final_paper/` (`git show`); `results/01` | 31 | Table 2; Fig. 1 values |
| `36_table2_vs_vanilla.py` | `results/36_…json` | `results/35` | 35 | superseded by 39; read by 39 |
| `37_loco_all_cells.py` | `results/37_…json` | `results/35` | 35 | superseded by 40; read by 40 |
| `38_table2_final_v2.py` | `results/38_table2_final_v2.json` | as 35 plus `delta_runs/armF_sar_crossearth_*` | 31, 35 | Table 2 (§3.4) |
| `39_table2_vs_vanilla_v2.py` | `results/39_…json` | `results/36`, `38` | 38 | §3.4, App. G.2 |
| `40_loco_all_cells_v2.py` | `results/40_…json` | `results/37`, `38` | 38 | App. G.1, G.2 |
| `41_figure1_ci.py` | `results/41_figure1_ci.json` | `results/35`, `38` | 35 | Fig. 1 error bars |
| `42_sib_ablations_k2.py` | `results/42_sib_ablations_k2.json` | `dumps/st_sib_ablation_png`, `dumps/st*`, `delta_runs/sib_ablation_k2`, `handoff/appendixJ_printed_values.json`; `results/41` | 31 | App. L |
| `43_provenance_backfill.py` | `results/43_provenance_backfill.json` | `dumps/st*`, `delta_dumps/dumps/e12`, `data/figure3_data`, run logs under `summaries/`, `delta_logs/`, `final_paper/` | 31 | App. J; Fig. 3 label |
| `_promote_lib.py` | — | — | — | shared envelope writer for the promotion scripts (13–15, 19–21, 30) |

## Figures

| script | output | reads |
|---|---|---|
| `fig01/fig01_teaser.py [ci_json] [out.png]` | Figure 1 | `results/35_table2_final.json` (`figure1_data`); CIs from the `ci_json` argument (`results/41_figure1_ci.json`); tiles via `fig01/_helpers.py` from `data/Final_data_test/` |
| `fig03/fig03_relabel.py <in.jpeg> <out.jpeg>` | Figure 3 | the submitted Figure 3 composite; replaces two printed values with those of `results/06` (ΔAURC +0.26) and `results/43` item 2b (~24×). Uses the URW Nimbus Sans Bold font at `/usr/share/fonts/opentype/urw-base35/` |
| `fig04/fig04b_profile.py [panel.png] [full.png]` | Figure 4 | `results/32_method_comparison_profile_tgtfda.json`; panel (a) is cropped from the submitted `final_paper/fig4_final.001.jpeg` |

Colours are fixed in [`docs/colors.md`](../docs/colors.md).

## Limitations of this release

- **Not runnable from public data alone.** The producers read Delta-side dumps and run outputs
  that are not distributed (they are derived from trained checkpoints). The scripts document the
  analysis exactly as run; reproducing a JSON needs the dumps.
- **Paper source.** `35_*`–`40_*` and `43_*` parse the camera-ready LaTeX with `git show <ref>:…`
  from `$PROJECT_ROOT/final_paper` to cross-check printed values, and pin the commit they checked
  against (`PAPER_REF`). That repository is not public. Other producers only record its `HEAD`
  as provenance and run without it.
- **Internal records.** A few producers read Markdown session records as provenance
  (`10_*`: `PAPER_STATE.md`, `REVIEWS.md`, `Final_Experiments_Results.md`; `20_*`:
  `PAPER_STATE.md`; `43_*`: `DECISIONS_armF.md`, `Final_Experiments_Results.md`, session logs).
  These are not distributed.
- **Pinned hashes.** Several producers assert the sha256 of an input (tree indexes, earlier
  results JSONs, the metric module) and stop on a mismatch. This is intended: a mismatch means
  the input differs from the one the published number was computed on.
- **Index rebuild in 22.** `22_internal_sib_worst_case_v3.py` rebuilds
  `analysis/_inv/INDEX_{e12,st,st_r450_published_recompute,st_r450_published_png}_v3.parquet`
  before it checks `DRY_RUN`, so `--dry-run` also writes those four files. On trees that have
  gained cells since results/22 was written (the n450 and `sibc4tgt` cells), the rebuilt
  `recompute` and `png` indexes differ from the ones results/22 cites by sha256. Keep the cited
  indexes if you need to reproduce results/22 byte for byte.
- **`PAPER_REF` in 35.** `35_table2_final.py` originally read the paper at `origin/main`; the
  release pins it to `686a50f2…`, the commit recorded in results/35's `paper.commit`, because the
  K=5 anchors it checks were later rewritten. 36, 37 and 41 import 35 and inherit the pin.
- **Superseded producers** (02, 03, 26–29, 36, 37) are kept because a cited producer reads their
  JSON; their headlines are superseded as listed in `docs/RESULTS_MAP.md`.
