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

## K in the other evaluators in this repository

The repository contains earlier evaluators that are kept because they produced
published numbers. They are **not** the documented path:

| Evaluator | Headline K and variant |
|---|---|
| `scoring/armB_png_score.py` | **K=2, two-class** — the documented App. E path |
| `matched450/n450_score.py` | K=2, **foreground-only** headline; also emits two-class at K=0/2 |
| `mamnet/utils/evaluation_detailed.py` and the per-arch copies | tolerant IoU at the K passed by the caller; the published MAMNet config selects checkpoints on two-class tolerant mIoU |
| `mamnet/utils/metrics.py` and the per-arch copies | strict IoU, i.e. K=0 |

Where a training script selects checkpoints on a different metric than it
reports, both are recorded in that run's `run.json` as `decision_metric` and
`reporting_metric`. Changing the decision metric changes which checkpoint is
chosen, so the published decision metric is left as it was.
