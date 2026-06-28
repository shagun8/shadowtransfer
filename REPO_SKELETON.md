# REPO_SKELETON

Layout of the release repo (the repo root corresponds to the original `python/` directory).

```
shadowtransfer/
├── README.md
├── MANIFEST.md                # provenance: every source file -> KEEP/EXCLUDE/ASK
├── ANONYMITY_REPORT.md        # double-blind scrub log
├── REPO_SKELETON.md           # this file
├── .gitignore
│
├── mamnet/                    # MAMNet architecture (CNN)
│   ├── train.py               # Vanilla + FDA (--use_fda)
│   ├── train_iim.py train_isw.py train_mrfp.py train_segdesic.py train_fada.py
│   ├── train_mamnet_sib.py    # SIB (ours)
│   ├── train_inria.py         # App. G.7 cross-task
│   ├── models/  data/  utils/ # base net + per-method forks + shared code
│   ├── *.sh / *_submit.sh     # RUN scripts (hyperparams) + SBATCH wrappers
│   └── config.yaml requirements.txt README.md
│
├── oglanet/                   # OGLANet architecture (CNN)  — same layout
│
├── dinov3/                    # DINOv3 architecture (ViT)
│   ├── dinov3_model.py dinov3_backbone.py dinov3_decoder.py dinov3_model_sib.py
│   ├── train_dinov3*.py       # one per kept method + SIB
│   ├── data/  utils/
│   ├── dinov3/                # << git submodule: facebookresearch/dinov3 (NOT committed)
│   └── weights/               # << place DINOv3 pretrained .pth here (gitignored)
│
├── final_loco/                # three-probe transfer analysis (Sec 4.2/4.3/4.4)
│   ├── experiment_c_histogram_match.py      # 4.2 pixel-space (Probe C)
│   ├── experiment_a_decoder_retrain.py      # 4.4 decoder retrain (Exp A)
│   ├── experiment_b2_encoder_retrain.py     # 4.4 encoder retrain (Exp B)
│   ├── evaluate_experiments.py compute_paper_statistics.py
│   ├── thread*.py run_diagnostics.py extract_features.py   # diagnostic suite
│   └── *.sh / *_submit.sh
│
└── (top level) analysis / aggregation scripts
    ├── run_inference.py run_inference_probs.py            # inference
    ├── final_comparison.py statistical_analysis.py        # main results + stats
    ├── aggregate_sp_gap.py aggregate_coverage_recover.py phase1_aggregate.py   # 4.3 SP-gap
    ├── tempscale*.py verify_c_star_val.py beta_sweep_summarize.py              # App. G.6/H/K
    ├── split_diagnostics.py feature_diagnostics.py                            # App. D
    ├── eval_sib.py sib_ablation_analysis_v2.py newmod_ablation_analysis.py    # Sec 5 / App. H
    ├── run_inference_probs_inria.py                                           # App. G.7
    └── select_figure2_tiles.py                                               # Fig 2
```

**Not in the repo** (see MANIFEST / ANONYMITY_REPORT): the 4 non-paper methods
(gsdpe/hrda/mcl/ddib), the vendored DINOv3 source, pretrained/trained weights, logs,
`__pycache__`, figure-output data, and exploratory branches the authors excluded.
