# CALLGRAPH — entry-point map

Trace any paper result to the one script that produces it, and see the whole chain at a glance:

```
SUBMIT *_submit.sh   →   RUN *.sh   →   python entry (train_*.py / *.py)   →   core modules
(SBATCH + sbatch)        (hyperparams)   (the actual program)                 (models/ data/ utils/)
```

- **SUBMIT** `*_submit.sh` — SBATCH headers + `sbatch` calls; sets per-environment vars. Starts
  with `: "${PROJECT_ROOT:?set PROJECT_ROOT}"` and passes `PROJECT_ROOT` via `--export`.
- **RUN** `*.sh` — the canonical hyperparameter record + the `python …` invocation.
- **python entry** — the training/analysis program.
- **core modules** — `models/*` (network), `data/*` (datasets/transforms), `utils/*` (losses,
  metrics, eval).

For the paper-element → script index see the top README's
[Reproducing the paper](README.md#reproducing-the-paper) and
[Top-level scripts](README.md#top-level-scripts) tables. Per-file details are in each directory's
README. **Conventions below:** `M_LOSS` = main MAMNet/OGLANet loss + `utils/metrics`,
`utils/postprocessing`, `utils/evaluation_detailed` (shared by every training entry; omitted per
method to keep the chain readable).

---

## Per architecture × method

The three architectures share an identical method set and an identical SUBMIT → RUN → python
shape. The differences are the python entry and the model/loss modules it pulls in.

### Generic shape (all archs)
```
<arch>_<method>_submit.sh
  └─ sbatch → <arch>_<method>.sh
       └─ python <train entry>.py  --mode loco --fold_id {0,1,2} ...
            ├─ models/<arch>[_<method>]   (network)
            ├─ data/dataset[_sib]         (dataloaders; LOCO folds = hold out 1 city)
            └─ utils/losses[...], metrics, postprocessing, evaluation_detailed
```
LOCO folds: `0=holdout_phoenix, 1=holdout_miami, 2=holdout_chicago`.

### MAMNet (`mamnet/`)
```
Vanilla   mamnet_submit.sh          → mamnet.sh          → train.py             → models/mamnet,          data/dataset,      utils/losses(MAMNetLoss)
FDA       mamnet_fda_submit.sh      → mamnet_fda.sh      → train.py --use_fda   → models/mamnet,          data/fda_transform
SegDeSiC  mamnet_segdesic_submit.sh → mamnet_segdesic.sh → train_segdesic.py    → models/mamnet_segdesic, utils/geo_losses
IIM       mamnet_iim_submit.sh      → mamnet_iim.sh      → train_iim.py         → models/mamnet_iim, models/iim
ISW       mamnet_isw_submit.sh      → mamnet_isw.sh      → train_isw.py         → models/mamnet,          utils/isw_loss   (masks: compute_isw_masks.sh → compute_isw_masks.py)
MRFP+     mamnet_mrfp_submit.sh     → mamnet_mrfp.sh     → train_mrfp.py        → models/mamnet_mrfp,     utils/losses_mrfp
FADA      mamnet_fada_submit.sh     → mamnet_fada.sh     → train_fada.py        → models/mamnet_fada
SIB       mamnet_sib_submit.sh      → mamnet_sib.sh      → train_mamnet_sib.py  → models/mamnet_sib, models/sib, data/dataset_sib, utils/losses(CACR,CEAURC)
  ablation  mamnet_sib_ablation_submit.sh  → mamnet_sib.sh        → train_mamnet_sib.py   (App. H grid)
  newmod    mamnet_sib_newmod_submit.sh    → mamnet_sib_newmod.sh → train_mamnet_sib.py   (CACR/CE-AURC/TENT)
  tempscale mamnet_sib_tempscale_submit.sh → mamnet_sib_tempscale.sh → tempscale_eval.py  (§5.3)
  sp_gap    mamnet_sib_sp_gap_submit.sh    → mamnet_sib_sp_gap.sh → sp_gap_analysis.py    (4.3)
INRIA     mamnet_inria_submit.sh    → mamnet_inria.sh    → train_inria.py       → inria_dataset   (App. G.7)
```

### OGLANet (`oglanet/`)
```
Vanilla   oglanet_submit.sh          → oglanet.sh          → train.py            → models/oglanet, data/dataset, utils/losses(OGLANetLoss)
FDA       oglanet_fda_submit.sh      → oglanet_fda.sh      → train.py            → data/fda_transform
SegDeSiC  oglanet_segdesic_submit.sh → oglanet_segdesic.sh → train_segdesic.py   → models/oglanet_segdesic
IIM       oglanet_iim_submit.sh      → oglanet_iim.sh      → train_iim.py        → models/oglanet_iim, models/iim
ISW       oglanet_isw_submit.sh      → oglanet_isw.sh      → train_oglanet_isw.py→ utils/isw_loss   (masks: compute_isw_masks_oglanet.sh → compute_isw_masks_oglanet.py)
MRFP+     oglanet_mrfp_submit.sh     → oglanet_mrfp.sh     → train_oglanet_mrfp.py → models/oglanet_mrfp, utils/losses_oglanet_mrfp
FADA      oglanet_fada_submit.sh     → oglanet_fada.sh     → train_fada.py       → models/oglanet_fada
SIB       oglanet_sib_submit.sh      → oglanet_sib.sh      → train_oglanet_sib.py→ models/oglanet_sib, models/sib, data/dataset_sib, utils/losses(CACR,CEAURC)
  ablation  oglanet_sib_ablation_submit.sh → oglanet_sib.sh        → train_oglanet_sib.py
  newmod    oglanet_sib_newmod_submit.sh   → oglanet_sib_newmod.sh → train_oglanet_sib.py
  sp_gap    oglanet_sib_sp_gap_submit.sh   → oglanet_sib_sp_gap.sh → sp_gap_analysis.py
```

### DINOv3 (`dinov3/`)  — model files are flat (no `models/` subdir); needs the `dinov3/dinov3/` submodule
```
Vanilla   dinov3_submit.sh          → dinov3.sh          → train_dinov3.py          → dinov3_model (→ dinov3_backbone, dinov3_decoder)
FDA       dinov3_fda_submit.sh      → dinov3_fda.sh      → train_dinov3.py          → data/fda_transform
SegDeSiC  dinov3_segdesic_submit.sh → dinov3_segdesic.sh → train_dinov3_segdesic.py → dinov3_segdesic
IIM       dinov3_iim_submit.sh      → dinov3_iim.sh      → train_dinov3_iim.py      → dinov3_iim_model, iim
ISW       dinov3_isw_submit.sh      → dinov3_isw.sh      → train_dinov3_isw.py      → dinov3_model, utils/isw_loss_dinov3   (masks: compute_isw_masks_dinov3.sh → compute_isw_masks_dinov3.py)
MRFP+     dinov3_mrfp_submit.sh     → dinov3_mrfp.sh     → train_dinov3_mrfp.py     → dinov3_model_mrfp, mrfp_modules
FADA      dinov3_fada_submit.sh     → dinov3_fada.sh     → train_dinov3_fada.py     → dinov3_model_fada
SIB       dinov3_sib_submit.sh      → dinov3_sib.sh      → train_dinov3_sib.py      → dinov3_model_sib, data/dataset_sib, utils/losses(CACR,CEAURC)
  ablation  dinov3_sib_ablation_submit.sh → dinov3_sib.sh        → train_dinov3_sib.py
  newmod    dinov3_sib_newmod_submit.sh   → dinov3_sib_newmod.sh → train_dinov3_sib.py
  sp_gap    dinov3_sib_sp_gap_submit.sh   → dinov3_sib_sp_gap.sh → sp_gap_analysis.py
```

### Shared per-arch eval / inference (all three archs)
```
inference_submit.sh   → inference.sh   → run_inference.py        (reads config.yaml)
test_loco_submit.sh   → test_loco.sh   → loco_evaluation.py + aggregate_loco_results.py
plot_loco_agg_submit.sh → plot_loco_agg.sh → aggregate_loco_results.py
```

---

## final_loco — transfer probes & diagnostics
```
run_experiments_submit.sh → run_experiments.sh → experiment_c_histogram_match.py   (4.2 Probe C)
                                                → experiment_a_decoder_retrain.py   (4.4 Exp A)
                                                → experiment_b2_encoder_retrain.py  (4.4 Exp B)
                                                → evaluate_experiments.py           (→ config, utils)
                                                  [all Exp scripts import experiment_utils]
plot_experiments_submit.sh → plot_experiments.sh → plot_experiment_results.py
run_extract_submit.sh      → run_extract.sh      → extract_features.py
run_paper_stats_submit.sh  → run_paper_stats.sh  → compute_paper_statistics.py      (→ config)
run_threads_submit.sh      → run_threads.sh      → run_diagnostics.py → thread1_1d / thread1_entanglement / thread3_geometry / thread4_position
                                                 → generate_plots.py
```
> Stale refs: `run_experiments.sh` also invokes `experiment_b_bn_swap.py` and
> `evaluate_recovery_robust.py`, which are not in this release (excluded Exp-B variants).

---

## Top-level analysis / aggregation (repo root)
```
run_inference_submit.sh            → run_inference.sh            → run_inference.py
run_inference_probs_submit.sh      → run_inference_probs.sh      → run_inference_probs.py          (4.3)
run_inference_probs_inria_submit.sh→ run_inference_probs_inria.sh→ run_inference_probs_inria.py    (App. G.7)
final_comparison_submit.sh         → final_comparison.sh         → final_comparison.py             (Sec 3.4 main comparison)
runonefile_submit.sh               → runonefile.sh               → analyze_inference_results.py, statistical_analysis.py
aggregate_sp_gap_submit.sh         → aggregate_sp_gap.sh         → aggregate_sp_gap.py             (4.3)
phase1_aggregate_submit.sh         → phase1_aggregate.sh         → phase1_aggregate.py             (4.3)
aggregate_coverage_recover_submit.sh → aggregate_coverage_recover.sh → aggregate_coverage_recover.py (4.3 / App. G.6)
beta_sweep_submit.sh               → beta_sweep.sh               → aggregate_coverage_recover.py, beta_sweep_summarize.py (App. G.6/K)
verify_c_star_val_submit.sh        → verify_c_star_val.sh        → verify_c_star_val.py            (App. G.6/K)
eval_sib_submit.sh                 → eval_sib.sh                 → eval_sib.py                     (Sec 5 / App. H)
sib_ablation_analysis_v2_submit.sh → sib_ablation_analysis_v2.sh → sib_ablation_analysis_v2.py     (App. H, Table 4)
newmod_ablation_analysis_submit.sh → newmod_ablation_analysis.sh → newmod_ablation_analysis.py     (App. H)
tempscale_submit.sh                → tempscale.sh                → <arch>/tempscale_eval.py        (§5.3)
tempscale_aggregate_submit.sh      → tempscale_aggregate.sh      → tempscale_aggregate.py          (§5.3)
split_diagnostics_submit.sh        → split_diagnostics.sh        → split_diagnostics.py            (App. D)
feature_diagnostics_submit.sh      → feature_diagnostics.sh      → feature_diagnostics.py          (App. D)
select_figure2_tiles_submit.sh     → select_figure2_tiles.sh     → select_figure2_tiles.py         (Figure 2)
```
