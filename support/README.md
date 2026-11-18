# `support/` — modules the released drivers import

Not entry points. These are here because a released driver imports them, so the
release would not run without them.

| File | What it does | Imported by |
|---|---|---|
| `trainaudit_sar_drift.py` | `diff_state`, the weight-drift diff over a full `state_dict` | `sar/armf_sar_n450.py` |
| `armf_sar_grid.py` | reads the SAR stage-1 selected configuration | `sar/armf_sar_n450.py` |
| `armf_sar_naming_patch.py` | count aggregation under the SAR naming scheme | `sar/armf_sar_n450.py` |
| `armc_r450_rescore.py` | `two_class` class averaging, and the rescore path | `matched450/n450_score.py` |
| `n450_isw_masks.py` | ISW mask preparation for the matched-450 ISW cells | `matched450/n450_train.py` |
| `oglanet_probsource.py` | resolves which probability source an OGLANet probe reads | `probes/oglanet_rng_probe*.py` |

`trainaudit_sar_drift.py` is released whole, as run. `diff_state`, the function
the SAR driver uses, needs only `torch`. Its other functions import
`shadowtransfer.armA_a4_sar_seo`, `shadowtransfer.armA_t1_cesar_adapt` and
`armA_pilot_train`, which **are** now shipped (the S-EO release pulled them in),
so those paths resolve as well.

## Added in the second release pass

| File | What it does | Imported / read by |
|---|---|---|
| `armd_diag.py` | the appearance-augmentation diagnostic driver | `appendix_h/armd_diag_aggregate.py` hashes it for provenance |
| `armA_pilot_train.py` | the S-EO phase-1/2 trainer the phase-3 driver builds on | `seo/armA_phase3_train.py`, `support/trainaudit_sar_drift.py` |
| `armA_seo_histmatch.py` | the S-EO histogram-matching intervention | the §4.2 probe chain |
