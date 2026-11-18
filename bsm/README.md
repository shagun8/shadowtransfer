# `bsm/` — BSM baseline, our reimplementation (App. F.23)

**No official implementation exists.** This is our reimplementation, released as
run. The module and the AdaIN style-network loading live in
`shadowtransfer/armf_bsm_module.py` and
`shadowtransfer/armf_bsm_arch_lib.py`.

| File | What it does | Invoked by | Camera-ready section |
|---|---|---|---|
| `armf_bsm_train.py` | RUN script: builds the BSM cell, loads the AdaIN style network, trains, scores, writes the dump | operator / submit wrapper | App. F.23 |

Style-network weights are **downloaded, not committed**. The loader resolves
them at run time; see the path arguments in `armf_bsm_train.py --help`.

Because no upstream reference exists, the appendix describes this
implementation and not a published one. Read App. F.23 before comparing these
numbers against any other BSM result.
