# `sar/` — SAR test-time adaptation on the matched-450 models (App. F.24)

Test-time adaptation only: no training, no checkpoint written, no published SAR
run touched. One cell per job, one city per cell. The adaptation itself lives in
`shadowtransfer/sar.py` and `shadowtransfer/armf_sar_pass.py`.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `armf_sar_n450.py` | RUN script: loads an n450 checkpoint, runs SAR episodically, records weight drift, writes the dump | `armf_sar_n450_worker.sh` | App. F.24 |
| `armf_sar_n450_worker.sh` | worker: modules, env, `PYTHON_BIN`, exported-variable validation, `.FAILED` trap | `armf_sar_n450_submit.sh` | App. F.24 |
| `armf_sar_n450_submit.sh` | SUBMIT wrapper: input preflight, disk gate, manifest, fan-out; `--dry-run`, `--smoke`, `--city` | operator | App. F.24 |

Weight drift is measured with `diff_state`, lifted from
`support/trainaudit_sar_drift.py` rather than reimplemented, and the driver
records that module's sha256 in its output.

## Site-specific parts of the submit wrapper

The wrapper's preflight is from our cluster and will not run elsewhere unchanged:

- it asserts `ST_SCRATCH_ROOT` and `ST_QUOTA_ALLOC` are the same volume;
- it calls `shadowtransfer/disk_gate.py`, which shells out to a site-specific
  quota tool, and refuses to submit under a margin floor;
- it writes a `dd conv=fsync` probe at the real write target;
- it appends a line to `runs/ledger.jsonl` for every submitted job.

Leave `ST_QUOTA_ALLOC` and `ST_ARCHIVE_ROOT` blank to skip the quota gate and
the archive-root assertion. The RUN script and worker need only
`PROJECT_ROOT` and `ST_SCRATCH_ROOT`.

## Added in the second release pass

| File | What it does | Cited by | Camera-ready section |
|---|---|---|---|
| `armf_sar_crossearth.py` | SAR applied to the CrossEarth cells | `results/38` (Table 2 incl. CrossEarth SAR) | App. F.24 |
| `armf_sar_crossearth_submit.sh`, `armf_sar_crossearth_worker.sh` | SUBMIT wrapper and worker | — | App. F.24 |

**SAR writes no checkpoint.** It adapts a parent checkpoint episodically and
discards the adapted weights, which is why `docs/CHECKPOINT_INVENTORY.json`
lists no SAR checkpoint and names the parent instead.
