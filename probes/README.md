# `probes/` — OGLANet inference nondeterminism probe (App. F.8)

Behind the nondeterminism statement in App. F.8. Check-only: loads an existing
checkpoint read-only, trains nothing, writes no checkpoint and no published
artifact.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `oglanet_rng_probe.py` | runs repeated inference passes over the full test list under controlled seeds and records the spread | submit wrapper | App. F.8 |
| `oglanet_rng_probe_report.py` | collects the probe cells into the reported range | operator | App. F.8 |

The probe always uses the **full 150-image test list**. It is never truncated:
OGLANet's DFFM reimplementation contains fresh `BatchNorm` layers that use batch
statistics at inference, so a shorter list changes the quantity being measured.
For the same reason the probe records the inference batch size, which is 4 for
the published OGLANet tables.

## Added in the second release pass

| File | What it does | Cited by | Camera-ready section |
|---|---|---|---|
| `armc_probes.py` | the §4.2–4.4 probe driver: histogram-matching intervention, selective prediction, city-identity linear probe | `results/20` | §4.2–4.4, App. G.3 |
| `armc_probes.sh`, `armc_probes_submit.sh` | worker and SUBMIT wrapper | — | §4.2–4.4 |

The probes run as a **separate inference job after a grid completes**, never
inside a training cell, so a probe failure cannot cost a trained model.
