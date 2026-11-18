# `scoring/` — K=2 boundary-tolerant scoring of saved prediction PNGs (App. E)

This is the documented scoring path for the camera-ready tables. It re-scores
saved prediction PNGs rather than re-running inference, so a table can be
rebuilt without a GPU.

| File | What it does | Metric and K | Invoked by | Camera-ready section |
|---|---|---|---|---|
| `armB_png_score.py` | scores saved prediction PNGs against GT, emits per-image counts and the dump | headline **two-class boundary-tolerant mIoU at K=2**; also emits K=0 and K=5 | operator | App. E |

All counts come from `shadowtransfer/metrics.py` (`per_image_counts` at
K=0/2/5). The class averaging is `armc_dump.miou_from_counts`, identical to
`support/armc_r450_rescore.py:two_class` to within eps 1e-10.

**The paper's metric is the two-class column at K=2 (App. E).** Where a scorer
in this repository also emits a foreground-only column, that column is a
self-check and carries only shadow-class statements.

## K in the other evaluators in this repository

The repository contains earlier evaluators that are kept because they produced
published numbers. They are **not** the documented path:

| Evaluator | Headline K and variant |
|---|---|
| `scoring/armB_png_score.py` | **K=2, two-class** — the documented App. E path |
| `matched450/n450_score.py` | emits the **two-class** K=2 column that the paper reports; its `headline_metric` field names the foreground-only K=2 quantity, which is the scorer self-check (results/31 caveat d) and is used only for shadow-class statements |
| `mamnet/utils/evaluation_detailed.py` and the per-arch copies | tolerant IoU at the K passed by the caller; the published MAMNet config selects checkpoints on two-class tolerant mIoU |
| `mamnet/utils/metrics.py` and the per-arch copies | strict IoU, i.e. K=0 |

Where a training script selects checkpoints on a different metric than it
reports, both are recorded in that run's `run.json` as `decision_metric` and
`reporting_metric`. Changing the decision metric changes which checkpoint is
chosen, so the published decision metric is left as it was.

## Added in the second release pass

| File | What it does | Metric and K | Cited by |
|---|---|---|---|
| `armB_png_score_midres63.py` | the mid-resolution (63-cell) PNG re-score | two-class boundary-tolerant mIoU at K=2 | `results/35`, `results/38` (Table 2) |
| `armB_png_score_midres63.sh`, `armB_png_score_midres63_submit.sh` | worker and SUBMIT wrapper | — | — |
