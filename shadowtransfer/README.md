# `shadowtransfer/` — metric package and shared run-time modules

Imported as `shadowtransfer`. `ST_METRICS_ROOT` selects the import root and
defaults to the repository root, so no configuration is needed in a normal
checkout. `analysis/07_sib_mechanism.py` and
`analysis/gj_analysis/gj_blob_miss.py` both import from here and both verify
`metrics.py` by sha256 before use.

**`metrics.py` must stay byte-identical.** Its sha256 is
`730ad4a5bc039e3151599fb86bd411e2816c0897`; the two analysis scripts abort if
it changes. It is assembled by lifting functions verbatim from the
per-architecture evaluators, so a change here is a change to every reported
number.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `metrics.py` | the single implementation of boundary-tolerant IoU, the ±K don't-care band, confusion counts, risk-coverage/AURC, CE-AURC, stratified ECE, recovery ratio R | everything that scores | App. E |
| `armc_dump.py` | per-image dump writer (`perimage.parquet` + `run.json`) and `sha256_file` | every driver | App. E |
| `armc_runtime.py` | quarantined-path preflight, time-boxed reads, `.DONE`/`.FAILED` markers | every driver | — (infrastructure) |
| `armc_probes_common.py`, `armc_probes_ops.py` | shared probe helpers for the §4.2–4.4 probes | `appendix_h/armL_train.py` | §4.2–4.4 |
| `armd_aug.py`, `armd_s3.py` | appearance-augmentation transforms and stage-3 helpers | `appendix_h/armd_train.py` | App. H.2 |
| `arme_common.py`, `arme_part1_dump.py`, `arme_part1_runtime.py`, `arme_e1_lib.py` | frozen-backbone probe helpers | `appendix_h/arme_*` | App. H.3 |
| `mamnet_r50.py` | ResNet-50 MAMNet variant used by the regime experiments | `appendix_h/armL_train.py` | App. H.4 |
| `armf_crossearth_runtime.py` | mmseg config overlay and split guards around upstream CrossEarth | `crossearth/` | App. F.22 |
| `armf_crossearth_engine.py` | train/validate/predict loop for CrossEarth | `crossearth/` | App. F.22 |
| `armf_crossearth_cells.py` | CrossEarth cell roster, dataset, FDA band | `crossearth/`, `analysis/07` | App. F.22 |
| `armf_crossearth_sib.py` | SIB module wiring for CrossEarth cells | `crossearth/` | §5, App. J |
| `armf_bsm_module.py`, `armf_bsm_arch_lib.py` | BSM module and AdaIN style-network loading | `bsm/` | App. F.23 |
| `sar.py` | SAR test-time adaptation (entropy objective, episodic reset) | `sar/` | App. F.24 |
| `armf_sar_pass.py` | one SAR adaptation pass over a cell | `sar/` | App. F.24 |
| `armf_sar_cells.py`, `armf_sar_meta.py` | SAR cell roster and metadata | `sar/` | App. F.24 |
| `n450_config.py`, `n450_roster.py`, `n450_inject.py`, `n450_sib.py` | matched-450 config, roster, config injection, SIB variant | `matched450/` | §3.3, Table 2 |
| `ledger.py` | append-only run ledger (`runs/ledger.jsonl`) | submit wrappers | — (infrastructure) |
| `worker_contract.py` | checks each cell's `--export` string against its worker, both directions | submit wrappers | — (infrastructure) |
| `disk_gate.py` | writable-headroom gate; reads a site-specific quota tool | `sar/..._submit.sh` | — (infrastructure) |
| `fortress_recall.py` | raises a recall instruction when an input has been archived to tape | `armc_runtime.py`, `armf_sar_cells.py` | — (infrastructure) |

`metrics.py` imports `compute_shadow_contrast` from `mamnet/utils/contrast_utils.py`
by resolving `../mamnet` relative to its own location. That works unchanged in
this layout because `shadowtransfer/` sits at the repository root next to
`mamnet/`.

The four infrastructure modules are specific to the cluster the paper's runs
were produced on. They are released so the submit wrappers are complete and
auditable, not because they are needed to reproduce a number.
