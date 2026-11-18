"""Arm F / BSM — shared helpers for the mamnet/oglanet/dinov3 BSM trainer.

Kept out of armf_bsm_train.py so that file stays under the ~300-line Write
limit this environment enforces (CLAUDE.md: Write hangs past that on this box).
"""

import json
from pathlib import Path

BSM_CAVEAT = (
    "BSM is our reimplementation from Luo et al., JSTARS 2023; no official "
    "code has been released. Style mixing uses a frozen pretrained AdaIN "
    "encoder-decoder and the training batch itself as the style source, so no "
    "external or held-out-city imagery enters training."
)

WEAKNESS_IDS = ["AC3", "v1UJ-W3"]


def bsm_run_id(arch, holdout, res="hr", rung=450, seed=0):
    return f"{arch}-bsm-loco-r{rung}-{holdout}-{res}-s{seed}"


def comparator_paths(base, arch, holdout):
    """Resolved paths of the cells BSM is compared against downstream.

    This job computes no comparison (CONVENTIONS.md: that is CC-SSH work); it
    only records where the comparator cells live. `fda_cell_path` points at the
    legacy pre-CONTRACT directory because no arm has regenerated the published
    FDA cells into dumps/st/ yet (CONTRACT.md §8) -- confirmed by directory
    listing at write time, not assumed.
    """
    base = Path(base)
    st = base / "dumps" / "st"
    vanilla = st / f"{arch}-vanilla-loco-r450-{holdout}-hr-s0"
    upper = st / f"{arch}-none-upper-r450-{holdout}-hr-s0"
    fda_legacy = (base / "data" / arch / "outputs"
                  / f"{arch}_fda_loco_holdout_{holdout}_highres_1")
    return {
        "vanilla_cell_path": str(vanilla) if (vanilla / ".DONE").exists() else None,
        "upper_cell_path": str(upper) if (upper / ".DONE").exists() else None,
        "fda_cell_path": str(fda_legacy) if (fda_legacy / "args.json").exists() else None,
        "fda_cell_path_note": ("legacy published directory, pre-CONTRACT.md "
                               "dump format (test_results.json, not "
                               "perimage.parquet); not yet regenerated under "
                               "dumps/st/, per CONTRACT.md section 8"),
    }


def method_config(bsm_report, alpha, batch_size, self_pair_log, aug_note):
    return {
        "mixing_coefficient": alpha,
        "augmentation_probs": {
            "base_pipeline": aug_note,
            "bsm_application_probability": 1.0,
            "bsm_application_note": ("BSM's own per-batch application "
                                     "probability is not specified in the "
                                     "accessible paper summary; applied to "
                                     "every training batch at a fixed mixing "
                                     "coefficient, the simplest faithful "
                                     "reading of 'per batch, at training time "
                                     "only'."),
        },
        "batch_size": batch_size,
        "hyperparams_set_on": "source_city_val",
        "hyperparams_note": ("alpha fixed at a single value and checked for "
                             "non-degeneracy (finite loss, sane predicted "
                             "foreground fraction) on source-city validation "
                             "at the stage-1 smoke gate; not swept over a "
                             "grid of alpha values, which was out of scope "
                             "for the 3-stage pilot-then-grid plan."),
        "style_net": bsm_report,
        "shuffle_self_pair_log": self_pair_log,
    }


def write_results_json(runs_dir, results):
    (runs_dir / "results.json").write_text(json.dumps(results, indent=2))
