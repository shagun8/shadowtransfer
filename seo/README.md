# `seo/` — S-EO external benchmark (§4.5, App. I)

Released because these runs are inputs to `results/04`, `results/07`,
`results/08`, `results/08b`, `results/09`, `results/10` and `results/34`, all of
which the camera-ready cites (see `docs/RESULTS_MAP.md`).

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `armA_phase3_train.py` | RUN script: trains or evaluates one S-EO cell and writes the dump | `armA_phase3_worker.sh` | §4.5, App. I |
| `armA_phase3_worker.sh` | worker: modules, env, exported-variable validation, `.FAILED` trap | `armA_phase3_submit.sh` | §4.5, App. I |
| `armA_phase3_submit.sh` | SUBMIT wrapper: fan-out over the S-EO grid; `--dry-run` | operator | §4.5, App. I |
| `armA_b5_sar_probe.py` | SAR probe on the S-EO cells | `armA_b5_sar_worker.sh` | App. I |
| `armA_b5_sar_worker.sh`, `armA_b5_sar_submit.sh` | worker and SUBMIT wrapper for the SAR probe | operator | App. I |
| `armA_b5_t2_roster.py` | writes the tier-2 roster JSON that `results/08` reads | operator | App. I |
| `armA_b4_probe_inventory.py` | writes the probe inventory JSON that `results/08` reads | `armA_b4_probe_worker.sh` | App. I |
| `armA_b4_probe_worker.sh`, `armA_b4_probe_submit.sh` | worker and SUBMIT wrapper for the probe inventory | operator | App. I |

The S-EO loaders, budget logic and dump writer are in
`shadowtransfer/seo_{dataset,budget,traineval,dump,crossearth}.py`. The
histogram-matching variant and the phase-1/2 trainer these drivers build on are
in `support/`.

`armA_b4_probe_inventory.py` reads candidate paths from a **child process the
parent abandons**. That is deliberate: it was written against a filesystem
condition where a read could block uninterruptibly, so a time-box in the calling
process could not recover. Keep that structure if you adapt it.
