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

`trainaudit_sar_drift.py` is released whole, as run. Only `diff_state` is
reachable from this release: its other functions import arm-A diagnostic
modules that are **not** part of the camera-ready release and are not shipped,
so calling them here raises `ImportError`. `diff_state` itself needs only
`torch`.
