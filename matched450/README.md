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

`n450_score.py` sets `headline_metric` to **foreground-only IoU with a K=2
don't-care band** and prints `fg-only IoU K=2`. It *also* writes
`miou_k2_twoclass` and `miou_k0_twoclass` to the same output, and raw counts at
K=0/2/5 in the parquet, so the band and the class averaging can both be
revisited without rescoring.

**Note the variant.** App. E is the two-class boundary-tolerant mIoU at K=2,
which is what `scoring/armB_png_score.py` reports as its headline. This scorer
headlines the foreground-only quantity at the same K=2 band. The two are
different numbers from the same counts; both are emitted here. Read App. E and
pick the column deliberately rather than taking `headline_metric` as the
paper's metric.
