# `crossearth/` — CrossEarth baseline (App. F.22)

Upstream is **not vendored**. `upstream_pin.json` carries the pinned commit
(`644a5a1b3c01b2e5531820b5291d4397597f75de`, VisionXLab/CrossEarth), clone
instructions, and the deviations table. Clone upstream yourself and keep its
license headers; this directory is only the overlay we ran around it.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `armf_crossearth_train.py` | RUN script: holds hyperparameters, builds the cell, trains, scores, writes the dump | `armf_crossearth_cell.sh` | App. F.22 |
| `armf_crossearth_cell.sh` | worker: modules, env, `PYTHON_BIN`, calls the RUN script | `armf_crossearth_submit.sh` | App. F.22 |
| `armf_crossearth_submit.sh` | SUBMIT wrapper: paths, SBATCH block, fan-out over cells; `--dry-run` prints without submitting | operator | App. F.22 |
| `upstream_pin.json` | pinned upstream commit, clone instructions, deviations from the published config | documentation | App. F.22 |

## Deviations from the published CrossEarth configuration

Copied verbatim from `armf_crossearth_train.py`'s
`run_cfg["deviations_from_published"]`, which is what each run's `run.json`
records. These are deliberate and reported, not defects:

- crop 512 → 384 (our native patch)
- batch 1 → 4
- iterations 30000 → 10000 (~89 epochs at 450 images, our 100-epoch protocol;
  30000 projects to 5.3 h and exceeds the 3:59:59 partition ceiling)
- bf16 autocast with mmcv deformable attention forced to fp32 (mmcv 2.1.0 has
  no bf16 `ms_deform_attn_forward_cuda` kernel)
- validation is source-city only; the released DG configs put the target domain
  in the validation dataloader
- `test_cfg` slide → whole (image size equals crop size)
- `num_classes` → 2

Checkpoint selection is on the **source-city validation split only**, which is
the point of the fifth item: no held-out-city image reaches training or
validation, so these cells are `source_only` on target access.
