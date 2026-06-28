# ShadowTransfer

Training and evaluation code for **ShadowTransfer** (NeurIPS 2026 submission, under
double-blind review). The repository reproduces the shadow-detection transfer study across
three segmentation backbones (**MAMNet**, **OGLANet**, **DINOv3**) under a leave-one-city-out
(LOCO) geographic-transfer protocol, the seven domain-generalization baselines evaluated in
the paper, our method (**SIB**), and the diagnostic analyses behind Sections 4–5 and the
appendices.

> **Anonymity:** this is an anonymized release. All cluster-specific settings now resolve from
> **four environment variables** — `PROJECT_ROOT`, `SLURM_ACCOUNT`, `SLURM_PARTITION`, `NODE` —
> rather than hard-coded paths. Set them once (see [Configuration](#configuration)) instead of
> editing every script. See [`ANONYMITY_REPORT.md`](ANONYMITY_REPORT.md).

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

## Repository layout
See [`REPO_SKELETON.md`](REPO_SKELETON.md) for the full tree and
[`MANIFEST.md`](MANIFEST.md) for the provenance of every file. In short: one directory per
architecture (`mamnet/`, `oglanet/`, `dinov3/`), the transfer-probe experiments in
`final_loco/`, and the aggregation/analysis scripts at the repository root.

The annotated tree below marks the **entry points you run** (`▶`); unmarked items are helpers,
imported modules, or generated config that the entry points pull in.

```
shadowtransfer-release/
├── README.md  MANIFEST.md  CALLGRAPH.md  ORPHANS.md  ANONYMITY_REPORT.md  LICENSE
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
git clone <ANON_REPO_URL> shadowtransfer && cd shadowtransfer

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
The dataset and trained checkpoints are hosted on Hugging Face (siblings):
- Dataset: `https://huggingface.co/datasets/shadow-transfer-bench/ShadowTransfer`
- Checkpoints: `https://huggingface.co/datasets/shadow-transfer-bench/ShadowTransfer`

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
| Paper element | Scripts |
|---|---|
| Sec 3.4 baselines / main comparison | `run_inference.py`, `analyze_inference_results.py`, `final_comparison.py`, `statistical_analysis.py` |
| 4.2 pixel-space test (Probe C) | `final_loco/experiment_c_histogram_match.py` |
| 4.3 selective-prediction gap | `run_inference_probs.py`, `aggregate_sp_gap.py`, `phase1_aggregate.py`, `aggregate_coverage_recover.py` |
| 4.4 decoder / encoder retrain (Exp A / B) | `final_loco/experiment_a_decoder_retrain.py`, `final_loco/experiment_b2_encoder_retrain.py`, `final_loco/evaluate_experiments.py` |
| Sec 5 / App. H — SIB | `train_*_sib.py`, `eval_sib.py`, `sib_ablation_analysis_v2.py`, `newmod_ablation_analysis.py` |
| §5.3 temperature scaling | `tempscale*.py`, `tempscale_aggregate.py` |
| App. D split / feature diagnostics | `split_diagnostics.py`, `feature_diagnostics.py` |
| App. G.6 / K coverage threshold | `verify_c_star_val.py`, `beta_sweep_summarize.py` |
| App. G.7 cross-task (INRIA) | `run_inference_probs_inria.py`, `mamnet/train_inria.py` |
| Cross-location diagnostic threads | `final_loco/run_diagnostics.py`, `final_loco/thread*.py` |
| Figure 2 (qualitative) | `select_figure2_tiles.py` |

> All scripts named above exist at the listed paths (verified). Two clarifications:
> `tempscale*.py` refers to the per-architecture `<arch>/tempscale_eval.py` plus the root
> `tempscale_aggregate.py` (there is no `tempscale.py`); and bare names like `run_inference.py`
> / `eval_sib.py` exist at the repo root but also have per-architecture namesakes —
> [Top-level scripts](#top-level-scripts) and the per-directory READMEs disambiguate which.

### Diagnostics & probes — result → script → produces
The diagnostic/probe rows above, spelled out so a reviewer can follow each result without
reading code. Multi-step pipelines are numbered in run order.

**§4.2 — pixel-space test (Probe C).**
| result | script | produces |
|---|---|---|
| Is the transfer gap just a pixel-intensity shift? Test-time intensity standardization (histogram-match the holdout city to source). | `final_loco/experiment_c_histogram_match.py` (RUN: `run_experiments.sh` with `EXPERIMENT=c`) | matched-input predictions on the holdout test set; `evaluate_experiments.py` turns them into the Probe-C recovery ratio **R** reported in §4.2. |

**§4.3 — selective-prediction (SP) gap + coverage-recovery.** Run in order:
1. `run_inference_probs.py` — saves per-image `P(shadow)` `.npy` maps under `Test_img_probs/` for every cell × method.
2. `<arch>/sp_gap_analysis.py` (RUN: `<arch>_sib_sp_gap.sh`) — per-cell SP-gap JSON `sp_gap_<arch>_<city>_<res>.json` (per-image AURC_shadow / ECE / mIoU + val & test risk-coverage records).
3. `aggregate_sp_gap.py` — aggregates the 9 cells (3 arch × 3 city), compares **C4-clean vs Vanilla + 6 baselines** → the §4.3 SP-gap tables.
4. `phase1_aggregate.py` — applies the pre-registered decision rule on AURC_shadow (PASS / PASS-STRAT / FAIL).
5. `aggregate_coverage_recover.py` — Phase-2 coverage-recovery: fits `c*_val`, computes per-image selective error at `c*_val` vs `c=1.0`, image-level cluster bootstrap (B=10000) + Wilcoxon over the 9 cell means.

**§4.4 — decoder / encoder retrain (Exp A / Exp B2).**
| result | script | produces |
|---|---|---|
| Decoder retrain on holdout, encoder frozen — R≈1 ⇒ failure was at the decoder | `final_loco/experiment_a_decoder_retrain.py` (`run_experiments.sh EXPERIMENT=a`) | holdout-test predictions → `evaluate_experiments.py` computes **R = (exp_IoU − LOCO_IoU)/(upper_IoU − LOCO_IoU)** for Exp A. |
| Encoder retrain on holdout, decoder frozen — R≈1 ⇒ failure was at the encoder; both low ⇒ encoder–decoder coupling | `final_loco/experiment_b2_encoder_retrain.py` (`run_experiments.sh EXPERIMENT=b2`) | holdout-test predictions → `evaluate_experiments.py` computes the Exp B2 **R** values. |

(The exploratory Exp-B BN-swap and `eval_robust` variants are excluded from this release —
their invocations are commented out in `run_experiments.sh`; see `final_loco/README.md`.)

**App. G.6 / K — coverage-threshold validity.**
| result | script | produces |
|---|---|---|
| `c*_val` sits at a genuine "halve the source-val full-coverage error" point, not a coverage-grid artifact | `verify_c_star_val.py` | a 3×3 (arch × city) risk-coverage summary-grid PNG with `c*_val` + β-target overlaid. |
| The population result is stable as β varies ("fixed once, results stable") | `beta_sweep_summarize.py` (after `aggregate_coverage_recover.py` at several β via `beta_sweep.sh`) | `beta_sweep_summary.md` / `.json` showing the result across β. |

**App. D — split & feature diagnostics.**
| result | script | produces |
|---|---|---|
| Train/val/test splits within a city share a distribution (label-stat KS, domain-classifier covariate-shift, deep-feature MMD, spatial coverage) | `split_diagnostics.py` | per-city split-diagnostic statistics/tables. |
| Task-relevant feature divergence, train vs test, per city × resolution (KS-D + normalized Wasserstein) | `feature_diagnostics.py` | one comparison table (rows = cities, columns = features). |

**App. G.7 — cross-task control (INRIA).**
| result | script | produces |
|---|---|---|
| Does the SP-gap behavior reproduce on a non-shadow task (INRIA building footprints)? | `mamnet/train_inria.py` (train) → `run_inference_probs_inria.py` (infer) | per-image `P(building)` `.npy` maps in the same layout as the shadow probs, so the §4.3 aggregation scripts can be re-run on them. |

## Top-level scripts
Analysis / aggregation scripts at the repo root. Each has a RUN script (`*.sh`) and a SUBMIT
wrapper (`*_submit.sh`); the table lists the python entry point each RUN invokes.

| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `run_inference.py` | inference over a backbone's checkpoints | `run_inference.sh` ← `run_inference_submit.sh` | Sec 3.4 |
| `analyze_inference_results.py` | analyze inference outputs | invoked by `runonefile.sh` | Sec 3.4 |
| `statistical_analysis.py` | significance tests on results | invoked by `runonefile.sh` | Sec 3.4 |
| `final_comparison.py` | C4_clean vs all baselines, paired bootstrap (`--base_path` ← `PROJECT_ROOT`) | `final_comparison.sh` ← `final_comparison_submit.sh` | Sec 3.4 main comparison |
| `run_inference_probs.py` | inference with per-pixel probabilities | `run_inference_probs.sh` ← `run_inference_probs_submit.sh` | 4.3 SP-gap |
| `aggregate_sp_gap.py` | aggregate selective-prediction gap | `aggregate_sp_gap.sh` ← `aggregate_sp_gap_submit.sh` | 4.3 SP-gap |
| `phase1_aggregate.py` | phase-1 SP-gap aggregation | `phase1_aggregate.sh` ← `phase1_aggregate_submit.sh` | 4.3 SP-gap |
| `aggregate_coverage_recover.py` | aggregate coverage/recovery curves | `aggregate_coverage_recover.sh`, `beta_sweep.sh` | 4.3 / App. G.6 |
| `eval_sib.py` | evaluate SIB across architectures | `eval_sib.sh` ← `eval_sib_submit.sh` | Sec 5 / App. H |
| `sib_ablation_analysis_v2.py` | SIB component ablation (Table 4), recomputed from PNGs | `sib_ablation_analysis_v2.sh` ← `sib_ablation_analysis_v2_submit.sh` | App. H |
| `newmod_ablation_analysis.py` | CACR + CE-AURC + TENT ablation | `newmod_ablation_analysis.sh` ← `newmod_ablation_analysis_submit.sh` | App. H |
| `tempscale_aggregate.py` | aggregate temp-scaling results across archs | `tempscale_aggregate.sh` ← `tempscale_aggregate_submit.sh` | §5.3 |
| `split_diagnostics.py` | train/val/test split diagnostics | `split_diagnostics.sh` ← `split_diagnostics_submit.sh` | App. D |
| `feature_diagnostics.py` | feature-space diagnostics | `feature_diagnostics.sh` ← `feature_diagnostics_submit.sh` | App. D |
| `verify_c_star_val.py` | verify coverage threshold c* on val | `verify_c_star_val.sh` ← `verify_c_star_val_submit.sh` | App. G.6 / K |
| `beta_sweep_summarize.py` | summarize β coverage sweep | `beta_sweep.sh` ← `beta_sweep_submit.sh` | App. G.6 / K |
| `run_inference_probs_inria.py` | INRIA cross-task inference w/ probs | `run_inference_probs_inria.sh` ← `run_inference_probs_inria_submit.sh` | App. G.7 |
| `select_figure2_tiles.py` | select qualitative Figure-2 tiles (`BASE_PATH` ← `PROJECT_ROOT`) | `select_figure2_tiles.sh` ← `select_figure2_tiles_submit.sh` | Figure 2 |
| `tempscale.sh` / `tempscale_submit.sh` | temp-scaling RUN/SUBMIT (cd's into an arch dir) | invokes `<arch>/tempscale_eval.py` | §5.3 |
| `runonefile.sh` / `runonefile_submit.sh` | one-shot driver for inference analysis | invokes `analyze_inference_results.py`, `statistical_analysis.py` | Sec 3.4 |

## License
Repository code: MIT (see [`LICENSE`](LICENSE)). The DINOv3 backbone is governed by the
[DINOv3 license](https://github.com/facebookresearch/dinov3) and is not included here.
