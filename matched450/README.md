# `matched450/` — matched-450 retrains and their scorer (§3.3, Table 2)

Every method is retrained at a matched budget of 450 training images so Table 2
compares methods at equal data, not equal epochs. The roster, config injection
and SIB variant live in `shadowtransfer/n450_*.py`.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `n450_train.py` | RUN script: retrains one cell at n=450 with the published per-method config injected | submit wrapper | §3.3, Table 2 |
| `n450_score.py` | scores a matched-450 cell and writes the dump; reproduces the cell's own stored result before emitting | `n450_train.py`, operator | §3.3, App. E |
| `n450_gate.py` | manifest/gate logic: which cells are intended, which are done, which are holes | operator | §3.3 |
| `n450_sib_noise.py` | the matched-450 SIB seed-noise cells | submit wrapper | §5, App. L |

## Which metric this scorer reports

**The paper's metric is the two-class column (App. E).** `n450_score.py` writes
`miou_k2_twoclass` and `miou_k0_twoclass`, and those are the numbers the
camera-ready reports for these cells.

The scorer's `headline_metric` field says "foreground-only IoU, K=2" and it
prints `fg-only IoU K=2`. That foreground-only column is the **scorer
self-check** (results/31, caveat d). It is used only for shadow-class
statements, never as the headline. Read `miou_k2_twoclass` unless you
specifically want the shadow-class quantity.

Raw counts at K=0/2/5 are in the parquet, so the band and the class averaging
can both be revisited without rescoring. See `scoring/README.md` for the K and
variant of every evaluator in the repository.

## Added in the second release pass

| File | What it does | Cited by | Camera-ready section |
|---|---|---|---|
| `n450_tgtfda.py` | matched-450 retrain whose FDA donor pool is the **held-out city's** train images, so these cells are `unlabeled_target` | `results/31`–`results/43`, which include Table 2 | §5, App. K |
| `n450_tgtfda_donor.py` | builds that donor pool; `n450_tgtfda.py` hashes it for provenance | as above | App. K |
| `n450_bs4.py` | the batch-size-4 inference variant behind the `-bs4` cells | `results/25` | App. G.1 |
| `n450_tgtfda_submit.sh`, `n450_tgtfda_worker.sh` | SUBMIT wrapper and worker | — | App. K |

See `sib/README.md` on target access: the `tgtfda` family touches held-out-city
images and must not be compared against `source_only` cells as if it did not.
