# final_loco — three-probe transfer analysis & diagnostics

The cross-location transfer probes (Sec 4.2/4.3/4.4) and the feature-level diagnostic suite.
Shared paths/constants live in `config.py`; shared helpers in `experiment_utils.py` (Exp A/B/C)
and `utils.py` (diagnostics). See the repo-root [`CALLGRAPH.md`](../CALLGRAPH.md) for the full
SUBMIT → RUN → python chain.

> **Start here.** Two entry points: `run_experiments.sh` runs the transfer probes — Exp A
> (`experiment_a_decoder_retrain.py`), Exp B2 (`experiment_b2_encoder_retrain.py`), Exp C
> (`experiment_c_histogram_match.py`) — then `evaluate_experiments.py`; `run_threads.sh` runs the
> diagnostic suite via `run_diagnostics.py`. `config.py` / `utils.py` / `experiment_utils.py` are
> shared helpers. Everything else below is a plotting or stats helper.

### Launch scripts (SUBMIT → RUN → python entry)
| file | what it does | invoked by / invokes | paper ref |
|---|---|---|---|
| `run_experiments_submit.sh` / `run_experiments.sh` | run transfer probes A/B/C + evaluation | RUN invokes `experiment_a_decoder_retrain.py`, `experiment_b2_encoder_retrain.py`, `experiment_c_histogram_match.py`, `evaluate_experiments.py` | 4.2 / 4.4 |
| `plot_experiments_submit.sh` / `plot_experiments.sh` | plot Exp A/B/C results | RUN invokes `plot_experiment_results.py` | 4.2 / 4.4 figures |
| `run_extract_submit.sh` / `run_extract.sh` | extract encoder features for diagnostics | RUN invokes `extract_features.py` | 4.3 / App. D |
| `run_paper_stats_submit.sh` / `run_paper_stats.sh` | compute paper statistics from features | RUN invokes `compute_paper_statistics.py` | Sec 4 stats |
| `run_threads_submit.sh` / `run_threads.sh` | run diagnostic threads + plots | RUN invokes `run_diagnostics.py`, `generate_plots.py` | App. D diagnostics |

> **Excluded variants (now commented out):** `run_experiments.sh` originally invoked
> `experiment_b_bn_swap.py` (the `EXPERIMENT=b` branch) and `evaluate_recovery_robust.py` (the
> `EXPERIMENT=eval_robust` branch), which are **not present** in this release (excluded exploratory
> Exp-B variants). Those invocation lines are now **commented out** with
> `# excluded variant — not in release` and each branch prints a "skipping" message, so the script
> parses and runs to completion. The kept probes are Exp A (`experiment_a_decoder_retrain.py`),
> Exp B2 (`experiment_b2_encoder_retrain.py`) and Exp C (`experiment_c_histogram_match.py`).

### Python files — experiments (Sec 4.2 / 4.4)
| file | what it does | invoked by / imports | paper ref |
|---|---|---|---|
| `experiment_c_histogram_match.py` | Probe C: pixel-space histogram matching | called by `run_experiments.sh`; imports `experiment_utils` | 4.2 (Probe C) |
| `experiment_a_decoder_retrain.py` | Exp A: decoder retrain on target | called by `run_experiments.sh`; imports `experiment_utils` | 4.4 (Exp A) |
| `experiment_b2_encoder_retrain.py` | Exp B: encoder retrain on target | called by `run_experiments.sh`; imports `experiment_utils` | 4.4 (Exp B) |
| `evaluate_experiments.py` | evaluate Exp A/B/C outputs | called by `run_experiments.sh`; imports `config`,`utils` | 4.4 |
| `experiment_utils.py` | shared Exp A/B/C utilities | imported by `experiment_*` | — |
| `plot_experiment_results.py` | plots for Exp A/B/C | called by `plot_experiments.sh` | 4.4 figures |

### Python files — diagnostics (Sec 4.3 / App. D)
| file | what it does | invoked by / imports | paper ref |
|---|---|---|---|
| `run_diagnostics.py` | master runner for the diagnostic threads | called by `run_threads.sh` | App. D |
| `extract_features.py` | extract encoder features per checkpoint | called by `run_extract.sh` | 4.3 / App. D |
| `compute_paper_statistics.py` | aggregate feature stats → paper numbers | called by `run_paper_stats.sh`; imports `config` | Sec 4 stats |
| `thread1_1d.py` | Diagnostic 1d: linear probe for city-identity in features | standalone; imports `os.environ["PROJECT_ROOT"]` paths | App. D (city-identity) |
| `thread1_entanglement.py` | Thread 1: causal vs. spurious feature entanglement | standalone | App. D (entanglement) |
| `thread3_geometry.py` | Thread 3: content- vs. geometry-directed attention | standalone | App. D (geometry) |
| `thread4_position.py` | Thread 4: positional-encoding fragility under shift | standalone | App. D (position) |
| `generate_plots.py` | plots from diagnostic results | called by `run_threads.sh` | App. D figures |
| `utils.py` | core diagnostic utilities | imported across diagnostics | — |
| `config.py` | paths (from `PROJECT_ROOT`), constants, city/variant lists, path helpers | imported across `final_loco` | — |
