# ShadowTransfer

Training and evaluation code for **ShadowTransfer: A Geographic Transfer Benchmark for Overhead
Shadow Detection** (NeurIPS 2026, Track on Evaluations and Datasets). The repository reproduces the shadow-detection transfer study across
three segmentation backbones (**MAMNet**, **OGLANet**, **DINOv3**) under a leave-one-city-out
(LOCO) geographic-transfer protocol, the seven domain-generalization baselines evaluated in
the paper, our method (**SIB**), and the diagnostic analyses behind Sections 4–5 and the
appendices.

- **Dataset:** <https://huggingface.co/datasets/shagunmittal/ShadowTransfer>
  (DOI [10.57967/hf/11483](https://doi.org/10.57967/hf/11483))
- **Citation:** see [Citation](#citation)

> **Configuration:** all cluster-specific settings resolve from
> **four environment variables** — `PROJECT_ROOT`, `SLURM_ACCOUNT`, `SLURM_PARTITION`, `NODE` —
> rather than hard-coded paths. Set them once (see [Configuration](#configuration)) instead of
> editing every script. 

## Configuration
Every path, SLURM account, and partition resolves from a single set of environment variables.
Copy the example file, fill in your four values, and source it before running anything:
```bash
cp .env.example .env
# edit .env: set PROJECT_ROOT, SLURM_ACCOUNT, SLURM_PARTITION, NODE
source .env
```
- In Python, paths come from `os.environ["PROJECT_ROOT"]`.
- In shell scripts, paths use `${PROJECT_ROOT}` (and `${SLURM_ACCOUNT}` / `${SLURM_PARTITION}` /
  `${NODE}`). Every `*_submit.sh` wrapper starts with `: "${PROJECT_ROOT:?set PROJECT_ROOT}"`,
  so it fails loudly if you forgot to `source .env`. The wrappers also pass `PROJECT_ROOT`
  through `sbatch --export`, so it reaches the job environment.
- **Note:** SLURM `#SBATCH` header lines are read by the scheduler *before* shell expansion, so
  `${SLURM_ACCOUNT}` / `${SLURM_PARTITION}` in those header comments are **not** auto-expanded —
  set the value in `.env` and, if your site requires it, also pass `--account`/`--partition` on
  the `sbatch` command line (or edit the header).
- **Note:** the three `*/config.yaml` files still contain the literal `<PROJECT_ROOT>`
  placeholder (YAML has no env-var expansion); edit those by hand if you use the YAML-driven
  inference path.

## Navigating the repo
Per-directory READMEs give a one-line table for every file (what it does, what calls/invokes it,
paper ref). The repo-root [`CALLGRAPH.md`](CALLGRAPH.md) traces the full
SUBMIT → RUN → python → modules chain per architecture and method.
- [`mamnet/README.md`](mamnet/README.md) · [`oglanet/README.md`](oglanet/README.md) ·
  [`dinov3/README.md`](dinov3/README.md) · [`final_loco/README.md`](final_loco/README.md)
- Top-level analysis/aggregation scripts are tabulated in
  [Top-level scripts](#top-level-scripts) below.
- Camera-ready additions: [`analysis/README.md`](analysis/README.md) (results producers and
  figure scripts) and [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) (results JSON → paper element →
  script); training/scoring code in `appendix_h/`, `crossearth/`, `bsm/`, `sar/`, `matched450/`,
  `scoring/`, `probes/`, with the metric package in `shadowtransfer/` (each has a README).

## Repository layout
See [`REPO_SKELETON.md`](REPO_SKELETON.md) for the full tree and
[`MANIFEST.md`](MANIFEST.md) for the provenance of every file. In short: one directory per
architecture (`mamnet/`, `oglanet/`, `dinov3/`), the transfer-probe experiments in
`final_loco/`, and the aggregation/analysis scripts at the repository root.

The annotated tree below marks the **entry points you run** (`▶`); unmarked items are helpers,
imported modules, or generated config that the entry points pull in.

```
shadowtransfer/
├── README.md  MANIFEST.md  CALLGRAPH.md  REPO_SKELETON.md  LICENSE
├── .env.example
│
├── *_submit.sh                 ▶ SLURM submit wrappers (one per analysis script)
├── *.sh                          RUN scripts (hold the hyperparameters; invoked by the wrappers)
├── run_inference.py / run_inference_probs.py / run_inference_probs_inria.py   ▶ inference entries
├── final_comparison.py  analyze_inference_results.py  statistical_analysis.py   analysis (Sec 3.4)
├── aggregate_sp_gap.py  phase1_aggregate.py  aggregate_coverage_recover.py      SP-gap pipeline (§4.3)
├── eval_sib.py  sib_ablation_analysis_v2.py  newmod_ablation_analysis.py        SIB (Sec 5 / App. H)
├── tempscale_aggregate.py  verify_c_star_val.py  beta_sweep_summarize.py        calibration / coverage
├── split_diagnostics.py  feature_diagnostics.py  select_figure2_tiles.py        App. D / Fig 2
│
├── mamnet/                     ▶ architecture 1  (see mamnet/README.md)
│   ├── train.py                ▶ entry: Vanilla / FDA
│   ├── train_{segdesic,iim,isw,mrfp,fada}.py   ▶ entry: per-baseline training
│   ├── train_mamnet_sib.py     ▶ entry: SIB (ours)
│   ├── train_inria.py          ▶ entry: App. G.7 cross-task
│   ├── *_submit.sh             ▶ SLURM submit wrappers   →   *.sh (RUN scripts)
│   ├── loco_evaluation.py  res_evaluation.py  eval_mamnet_sib.py  run_inference.py   eval & inference
│   ├── models/  data/  utils/    helper modules (imported by the entries above)
│   └── config.yaml  README.md  requirements.txt
│
├── oglanet/                    ▶ architecture 2  (same layout: train.py, train_oglanet_sib.py, …)
├── dinov3/                     ▶ architecture 3  (flat module files: train_dinov3*.py, dinov3_*_submit.sh)
│   └── dinov3/                   upstream DINOv3 repo — vendored by the user, not redistributed
│
└── final_loco/                  transfer probes & diagnostics  (see final_loco/README.md)
    ├── run_experiments.sh      ▶ entry: Exp A / B2 / C  →  experiment_{a_decoder_retrain,b2_encoder_retrain,c_histogram_match}.py
    ├── run_threads.sh          ▶ entry: diagnostics     →  run_diagnostics.py, thread*.py
    ├── evaluate_experiments.py   recovery-ratio R evaluation
    └── config.py  utils.py  experiment_utils.py    shared helpers

▶ = entry point (run these).  Unmarked = helper / imported module / generated config.
```

## Methods covered
| Config | Train entry point | Notes |
|--------|-------------------|-------|
| Vanilla | `train.py` | source-only baseline |
| FDA | `train.py --use_fda --fda_target_root …` | Fourier domain adaptation |
| SegDeSiC | `train_segdesic.py` | |
| IIM | `train_iim.py` | illumination-invariant module |
| ISW | `train_isw.py` (+ `compute_isw_masks*.py`) | instance-selective whitening |
| MRFP+ | `train_mrfp.py` / `train_*_mrfp.py` | multi-resolution feature perturbation |
| FADA | `train_fada.py` | |
| **SIB (ours)** | `train_mamnet_sib.py` / `train_oglanet_sib.py` / `train_dinov3_sib.py` | |

(Each architecture provides its own copy of the per-method training scripts and model files.)

## Installation
```bash
# 1. clone
git clone https://github.com/shagun8/shadowtransfer.git && cd shadowtransfer

# 2. python env (per-architecture requirements are identical; mamnet/requirements.txt suffices)
python -m venv .venv && source .venv/bin/activate
pip install -r mamnet/requirements.txt   # torch>=2.0, torchvision, numpy, pillow,
                                          # tensorboard, tqdm, matplotlib, seaborn
```

### DINOv3 backbone (required only for the `dinov3/` architecture)
The upstream [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) repository
is **not** redistributed here (its license restricts redistribution). `dinov3/dinov3_backbone.py`
adds the directory `dinov3/dinov3/` (a `dinov3/` folder next to that file) onto `sys.path` and
then does `from dinov3.models import vision_transformer`. So the upstream repo must be cloned to
`dinov3/dinov3/` (giving the import path `dinov3/dinov3/dinov3/models/…`). Add it there, then
drop in the pretrained weights:
```bash
# clone upstream to dinov3/dinov3/ (the path dinov3_backbone.py adds to sys.path)
git submodule add https://github.com/facebookresearch/dinov3 dinov3/dinov3
# obtain ViT-S/16 LVD-1689M weights from the official DINOv3 release and place at:
#   dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```
If you use DINOv3, comply with its license and **cite it** in any publication.

## Data and checkpoints
The dataset is hosted on Hugging Face:
- Dataset: <https://huggingface.co/datasets/shagunmittal/ShadowTransfer>
  (DOI [10.57967/hf/11483](https://doi.org/10.57967/hf/11483); paper results use revision `5bef90e`)
- Layout, counts, and metadata fields are documented on the dataset card.

Point `PROJECT_ROOT` (in `.env`) at a directory laid out as the scripts expect:
```
$PROJECT_ROOT/
  data/Final_data_test/{city}/{highres,midres}/{train,val,test}/{images,masks}/
  data/<arch>/outputs/        # checkpoints land here
```
Cities: `chicago`, `miami`, `phoenix`. LOCO folds hold out one city at a time.

## Running
Each method has a **RUN** script (the canonical record of hyperparameters, e.g.
`mamnet/mamnet_iim.sh`) and a **SUBMIT** wrapper (SBATCH headers + `sbatch` calls, e.g.
`mamnet/mamnet_iim_submit.sh`). The submit wrapper passes the env vars the run script reads.
First set up your environment once (see [Configuration](#configuration)), then launch a LOCO
sweep on SLURM:
```bash
source .env            # sets PROJECT_ROOT, SLURM_ACCOUNT, SLURM_PARTITION, NODE
cd mamnet
bash mamnet_iim_submit.sh
```
To run a single config without SLURM, `source .env` first, then read the `python … --args`
block inside the RUN script and invoke it directly (paths come from `${PROJECT_ROOT}`).

## Reproducing the paper
Section numbers follow the camera-ready.

**Metric.** "mIoU" in the paper is the two-class boundary-tolerant mIoU at K=2 (App. E),
computed by [`shadowtransfer/metrics.py`](shadowtransfer/metrics.py); foreground (shadow-class)
IoU is used only where a statement concerns the shadow class.

Every number added for the camera-ready is read from a results JSON `results/NN_*.json` written by
a script in [`analysis/`](analysis/README.md); [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md) maps
each JSON to its section, producer and inputs. The table below gives the code path per paper
element: training/inference → scoring → analysis.

| Paper element | Training / inference | Scoring and analysis |
|---|---|---|
| §3.3 LOCO protocol; Table 2 (§3.4 baselines, K=2) | per-arch `train*.py` (`mamnet/`, `oglanet/`, `dinov3/`); matched-450 retrains `matched450/n450_train.py` | `scoring/armB_png_score.py`, `matched450/n450_score.py`, `appendix_h/armc_r450.py` → `analysis/35_table2_final.py`, `analysis/38_table2_final_v2.py`, `analysis/39_table2_vs_vanilla_v2.py` |
| Table 2 CrossEarth / BSM / SAR rows; App. F.22–F.24 | `crossearth/armf_crossearth_train.py` (upstream pinned in `crossearth/upstream_pin.json`), `bsm/armf_bsm_train.py`, `sar/armf_sar_n450.py` | `analysis/13_armF_internal_crossearth_bsm_sar.py`, `analysis/38_table2_final_v2.py` |
| Figure 1 | as Table 2 | `analysis/41_figure1_ci.py`, `analysis/fig01/fig01_teaser.py` |
| §4.2, App. G.3 pixel-space test (histogram matching) | `final_loco/experiment_c_histogram_match.py` | `final_loco/evaluate_experiments.py`; `analysis/20_histogram_matching_counts.py` |
| §4.3, App. G.4–G.6 selective-prediction gap, coverage recovery, calibration | `run_inference_probs.py`, `<arch>/tempscale_eval.py` | `aggregate_sp_gap.py`, `phase1_aggregate.py`, `aggregate_coverage_recover.py`, `tempscale_aggregate.py`; `analysis/06_aurc_recovery.py`, `analysis/fig03/fig03_relabel.py` (Figure 3) |
| §4.4, App. G.8 architecture-dependent failure locus (component retraining) | `final_loco/experiment_a_decoder_retrain.py`, `final_loco/experiment_b2_encoder_retrain.py` | `final_loco/evaluate_experiments.py` |
| §4.5, App. I S-EO external benchmark | not released (see [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md#input-writers-not-in-this-repository)) | `analysis/02_*`–`04_*`, `07_*`–`10_*`, `34_seo_sib_family_tgtfda.py` |
| App. G.7 INRIA cross-task replication | `mamnet/train_inria.py`, `run_inference_probs_inria.py` | §4.3 aggregation scripts; `analysis/24_inria_crosstask.py` |
| App. G.1–G.2 per-cell LOCO tables and pooled summaries | as Table 2 | `analysis/40_loco_all_cells_v2.py`, `analysis/25_matched450_retrain_record.py` |
| §5 SIB case study; App. J implementation | `mamnet/train_mamnet_sib.py`, `oglanet/train_oglanet_sib.py`, `dinov3/train_dinov3_sib.py`; `matched450/n450_sib_noise.py` | `eval_sib.py`; `analysis/31_internal_sib_worst_case_tgtfda.py`, `analysis/43_provenance_backfill.py` |
| App. K gap-to-upper-bound comparison; Figure 4(b) | as Table 2 and §5 | `analysis/31_*`, `analysis/32_method_comparison_profile_tgtfda.py`, `analysis/33_m11_holm_tgtfda.py`, `analysis/19_e12_seed_table.py`, `analysis/fig04/fig04b_profile.py` |
| App. L SIB component ablations (K=2) | per-arch `*_sib.sh` ablation flags | `analysis/42_sib_ablations_k2.py` (`sib_ablation_analysis_v2.py` is the submission-era K=5 table) |
| App. M coverage threshold τ sweep | — | `aggregate_coverage_recover.py` via `beta_sweep.sh`, `beta_sweep_summarize.py`, `verify_c_star_val.py` |
| App. E metric | — | `shadowtransfer/metrics.py`; `scoring/armB_png_score.py` (documented K=2 path; `scoring/README.md` lists every evaluator's K); `analysis/14_appE_empty_foreground_rate.py` |
| App. F training configurations (F.1 MAMNet, F.8 OGLANet incl. DFFM note, F.15 DINOv3) | RUN scripts `<arch>/*.sh`; DFFM defect in `oglanet/README.md` | `probes/oglanet_rng_probe.py`, `probes/oglanet_rng_probe_report.py` → `analysis/30_oglanet_rng_dependence.py`; `analysis/21_disclosure_record.py` |
| App. H.1 training-set scale ladder | `appendix_h/armc_train.py` | `analysis/armc_ladder_stats.py` |
| App. H.2 appearance augmentation | `appendix_h/armd_train.py` | `analysis/15_armD_appearance_augmentation_results.py` |
| App. H.3 frozen DINOv3 | `appendix_h/arme_part1_train.py` | `appendix_h/arme_part2_revised_aggregate.py`; `analysis/arme_frozen_gap_stats.py`, `analysis/23_frozen_backbone_decoder_retrain.py` |
| App. H.4 pretraining / fine-tuning regime | `appendix_h/armL_train.py` | `appendix_h/armL_results.py`; `analysis/16_armL_ssl_regime.py`, `analysis/16_armL_g2_expA.py` |
| App. D split diagnostics | — | `split_diagnostics.py`, `feature_diagnostics.py` |
| App. A–C datasheet, annotation, taxonomy, sampling (C.3) | — | dataset card; no code beyond the split lists |
| Figure 2 (qualitative) | `run_inference.py` | `select_figure2_tiles.py` |

> Not covered by released code: the S-EO training/dump pipeline, the OGLANet-SIB held-out-city
> FDA retrains, the E-12 MAMNet-SIB retrains, the SIB-ablation K=2 scoring chain and the mid-res
> PNG scoring run. Their outputs are read by the `analysis/` producers listed above; see
> [`docs/RESULTS_MAP.md`](docs/RESULTS_MAP.md#input-writers-not-in-this-repository).
> `tempscale*.py` refers to the per-architecture `<arch>/tempscale_eval.py` plus the root
> `tempscale_aggregate.py`; bare names such as `run_inference.py` / `eval_sib.py` also have
> per-architecture namesakes, disambiguated in [Top-level scripts](#top-level-scripts) and the
> per-directory READMEs.

### Diagnostics & probes — result → script → produces
The diagnostic and probe rows above, spelled out so a reviewer can follow each result without
reading code. Multi-step pipelines are numbered in run order.

**§4.2 / App. G.3 — pixel-space test (histogram matching).**
| result | script | produces |
|---|---|---|
| Is the transfer gap a pixel-intensity shift? Test-time histogram matching of the held-out city to the source cities. | `final_loco/experiment_c_histogram_match.py` (RUN: `run_experiments.sh` with `EXPERIMENT=c`) | matched-input predictions; `evaluate_experiments.py` turns them into the recovery ratio **R** reported in §4.2. |
| How many cells improve under the published matching rule, internal ladder and S-EO? | `analysis/20_histogram_matching_counts.py` | `results/20_histogram_matching_counts.json` |

**§4.3 / App. G.4–G.6 — selective-prediction gap, coverage recovery, calibration.** Run in order:
1. `run_inference_probs.py` — saves per-image `P(shadow)` `.npy` maps under `Test_img_probs/` for every cell × method.
2. `<arch>/sp_gap_analysis.py` (RUN: `<arch>_sib_sp_gap.sh`) — per-cell SP-gap JSON `sp_gap_<arch>_<city>_<res>.json` (per-image AURC_shadow / ECE / mIoU + val & test risk-coverage records).
3. `aggregate_sp_gap.py` — aggregates the 9 cells (3 arch × 3 city) → the §4.3 / App. G.4 SP-gap tables.
4. `phase1_aggregate.py` — applies the pre-registered decision rule on AURC_shadow (PASS / PASS-STRAT / FAIL).
5. `aggregate_coverage_recover.py` — coverage recovery (App. G.6): fits `c*_val`, computes per-image selective error at `c*_val` vs `c=1.0`, image-level cluster bootstrap (B=10000) + Wilcoxon over the 9 cell means.
6. `<arch>/tempscale_eval.py` → `tempscale_aggregate.py` — post-hoc temperature scaling (App. G.6).
7. `analysis/06_aurc_recovery.py` — recovers the Figure 3(a) ΔAURC under the paper's AURC definition (App. G.5); `analysis/fig03/fig03_relabel.py` applies it to Figure 3.

**§4.4 / App. G.8 — component retraining (Exp A / Exp B).**
| result | script | produces |
|---|---|---|
| Decoder retrain on the held-out city, encoder frozen — R≈1 ⇒ the failure was at the decoder | `final_loco/experiment_a_decoder_retrain.py` (`run_experiments.sh EXPERIMENT=a`) | held-out predictions → `evaluate_experiments.py` computes **R = (exp_IoU − LOCO_IoU)/(upper_IoU − LOCO_IoU)** for Exp A. |
| Encoder retrain on the held-out city, decoder frozen — R≈1 ⇒ the failure was at the encoder; both low ⇒ encoder–decoder coupling | `final_loco/experiment_b2_encoder_retrain.py` (`run_experiments.sh EXPERIMENT=b2`) | held-out predictions → `evaluate_experiments.py` computes the Exp B **R** values. |

(The exploratory Exp-B BN-swap and `eval_robust` variants are excluded from this release —
their invocations are commented out in `run_experiments.sh`; see `final_loco/README.md`.)

**App. H — alternative explanations for the LOCO gap.**
| result | script | produces |
|---|---|---|
| H.1 Does the gap close with more training data (225 / 450 / 900)? | `appendix_h/armc_train.py` → `analysis/armc_ladder_stats.py` | `results/01_scaling_ladder.json` |
| H.2 Does training-time appearance augmentation close it? | `appendix_h/armd_train.py` → `analysis/15_armD_appearance_augmentation_results.py` | `results/15_armD_appearance_augmentation.json` |
| H.3 Does it persist with a frozen DINOv3 encoder? | `appendix_h/arme_part1_train.py`, `appendix_h/arme_part2_revised_aggregate.py` → `analysis/arme_frozen_gap_stats.py`, `analysis/23_frozen_backbone_decoder_retrain.py` | `results/02_frozen_backbone_gap.json`, `results/23_frozen_backbone_decoder_retrain.json` |
| H.4 Does the pretraining / fine-tuning regime explain it? | `appendix_h/armL_train.py`, `appendix_h/armL_results.py` → `analysis/16_armL_ssl_regime.py`, `analysis/16_armL_g2_expA.py` | `results/16_armL_ssl_regime.json`, `results/16_armL_g2_expA.md` |

**App. F.8 — OGLANet inference nondeterminism.**
| result | script | produces |
|---|---|---|
| How much do OGLANet cell means move across repeated inference passes and seeds? | `probes/oglanet_rng_probe.py` → `probes/oglanet_rng_probe_report.py` → `analysis/30_oglanet_rng_dependence.py` | `results/30_oglanet_rng_dependence.json` |

**App. M — coverage-threshold τ robustness.**
| result | script | produces |
|---|---|---|
| `c*_val` sits at a genuine "halve the source-val full-coverage error" point, not a coverage-grid artifact | `verify_c_star_val.py` | a 3×3 (arch × city) risk-coverage summary-grid PNG with `c*_val` + β-target overlaid. |
| The population result is stable as τ varies | `beta_sweep_summarize.py` (after `aggregate_coverage_recover.py` at several values via `beta_sweep.sh`) | `beta_sweep_summary.md` / `.json` showing the result across the sweep. |

**App. D — split & feature diagnostics.**
| result | script | produces |
|---|---|---|
| Train/val/test splits within a city share a distribution (label-stat KS, domain-classifier covariate-shift, deep-feature MMD, spatial coverage) | `split_diagnostics.py` | per-city split-diagnostic statistics/tables. |
| Task-relevant feature divergence, train vs test, per city × resolution (KS-D + normalized Wasserstein) | `feature_diagnostics.py` | one comparison table (rows = cities, columns = features). |

**App. G.7 — cross-task control (INRIA).**
| result | script | produces |
|---|---|---|
| Does the SP-gap behavior reproduce on a non-shadow task (INRIA building footprints)? | `mamnet/train_inria.py` (train) → `run_inference_probs_inria.py` (infer) → `analysis/24_inria_crosstask.py` | per-image `P(building)` `.npy` maps in the same layout as the shadow probs; `results/24_inria_crosstask.json` |

## Top-level scripts
Analysis / aggregation scripts at the repo root. Each has a RUN script (`*.sh`) and a SUBMIT
wrapper (`*_submit.sh`); the table lists the python entry point each RUN invokes. The camera-ready
directories have their own READMEs: [`analysis/`](analysis/README.md),
[`appendix_h/`](appendix_h/README.md), [`crossearth/`](crossearth/README.md),
[`bsm/`](bsm/README.md), [`sar/`](sar/README.md), [`matched450/`](matched450/README.md),
[`scoring/`](scoring/README.md), [`probes/`](probes/README.md),
[`shadowtransfer/`](shadowtransfer/README.md), [`support/`](support/README.md).

| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `run_inference.py` | inference over a backbone's checkpoints | `run_inference.sh` ← `run_inference_submit.sh` | §3.4, Figure 2 |
| `analyze_inference_results.py` | analyze inference outputs | invoked by `runonefile.sh` | §3.4 (submission-era) |
| `statistical_analysis.py` | significance tests on results | invoked by `runonefile.sh` | §3.4 (submission-era) |
| `final_comparison.py` | C4_clean vs all baselines, paired bootstrap (`--base_path` ← `PROJECT_ROOT`) | `final_comparison.sh` ← `final_comparison_submit.sh` | §3.4 submission-era comparison; camera-ready Table 2 values come from `analysis/38_table2_final_v2.py` |
| `run_inference_probs.py` | inference with per-pixel probabilities | `run_inference_probs.sh` ← `run_inference_probs_submit.sh` | §4.3, App. G.4 |
| `aggregate_sp_gap.py` | aggregate selective-prediction gap | `aggregate_sp_gap.sh` ← `aggregate_sp_gap_submit.sh` | §4.3, App. G.4 |
| `phase1_aggregate.py` | phase-1 SP-gap aggregation | `phase1_aggregate.sh` ← `phase1_aggregate_submit.sh` | §4.3 |
| `aggregate_coverage_recover.py` | aggregate coverage/recovery curves | `aggregate_coverage_recover.sh`, `beta_sweep.sh` | §4.3, App. G.6, App. M |
| `eval_sib.py` | evaluate SIB across architectures | `eval_sib.sh` ← `eval_sib_submit.sh` | §5 |
| `sib_ablation_analysis_v2.py` | SIB component ablation, recomputed from PNGs (submission-era K=5 table) | `sib_ablation_analysis_v2.sh` ← `sib_ablation_analysis_v2_submit.sh` | App. L (K=2 values: `analysis/42_sib_ablations_k2.py`) |
| `newmod_ablation_analysis.py` | CACR + CE-AURC + TENT ablation | `newmod_ablation_analysis.sh` ← `newmod_ablation_analysis_submit.sh` | not reported in the camera-ready |
| `tempscale_aggregate.py` | aggregate temp-scaling results across archs | `tempscale_aggregate.sh` ← `tempscale_aggregate_submit.sh` | App. G.6 |
| `split_diagnostics.py` | train/val/test split diagnostics | `split_diagnostics.sh` ← `split_diagnostics_submit.sh` | App. D |
| `feature_diagnostics.py` | feature-space diagnostics | `feature_diagnostics.sh` ← `feature_diagnostics_submit.sh` | App. D |
| `verify_c_star_val.py` | verify coverage threshold c* on val | `verify_c_star_val.sh` ← `verify_c_star_val_submit.sh` | App. G.6, App. M |
| `beta_sweep_summarize.py` | summarize the coverage-threshold sweep | `beta_sweep.sh` ← `beta_sweep_submit.sh` | App. M |
| `run_inference_probs_inria.py` | INRIA cross-task inference w/ probs | `run_inference_probs_inria.sh` ← `run_inference_probs_inria_submit.sh` | App. G.7 |
| `select_figure2_tiles.py` | select qualitative Figure-2 tiles (`BASE_PATH` ← `PROJECT_ROOT`) | `select_figure2_tiles.sh` ← `select_figure2_tiles_submit.sh` | Figure 2 |
| `tempscale.sh` / `tempscale_submit.sh` | temp-scaling RUN/SUBMIT (cd's into an arch dir) | invokes `<arch>/tempscale_eval.py` | App. G.6 |
| `runonefile.sh` / `runonefile_submit.sh` | one-shot driver for inference analysis | invokes `analyze_inference_results.py`, `statistical_analysis.py` | §3.4 (submission-era) |

## Citation
If you use this code or the dataset, please cite the paper:
```bibtex
@inproceedings{mittal2026shadowtransfer,
  title     = {ShadowTransfer: A Geographic Transfer Benchmark for Overhead Shadow Detection},
  author    = {Mittal, Shagun and Yeh, Raymond A. and Ukkusuri, Satish},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026}
}
```
To cite the dataset release itself:
```bibtex
@misc{mittal2026shadowtransferdata,
  author    = {Mittal, Shagun and Yeh, Raymond A. and Ukkusuri, Satish},
  title     = {{ShadowTransfer} (Revision 5bef90e)},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/11483},
  url       = {https://huggingface.co/datasets/shagunmittal/ShadowTransfer}
}
```
Please also credit USDA-FSA NAIP for the underlying imagery.

## License
Repository code: MIT (see [`LICENSE`](LICENSE)). The DINOv3 backbone is governed by the
[DINOv3 license](https://github.com/facebookresearch/dinov3) and is not included here.
The dataset (masks and metadata) is released under CC BY 4.0; see the dataset card.
