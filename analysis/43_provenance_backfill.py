"""EXP-43: provenance backfill -- SIB configuration check (Part 1) and a source
lookup for printed values no results JSON holds (Part 2).

Pure promotion and recomputation. CPU only. Nothing is trained or inferred.

Part 1 reads the run.json of every SIB run_id listed in results/31, results/32
and results/34, and the companion C4clean run.json from results/42's ablation
tree where a published-path dump carries no config of its own. It compares
the recorded flags with App J's tab:sib-hparams and Fig. 4(a).

Part 2 resolves items 2a-2g against:
  - Shade Maps/data/figure3_data/{robust_recovery_results,robust_recovery_full,
    paper_statistics,experiment_results,test_s1_v2_results}.json, rows.npy and
    cell_rcs.pkl (the files FINAL-fig3.ipynb reads to build Figure 3)
  - the PNG-tree dumps (dumps/st_r450_published_png): aurc_fg / aurc_ce_fg per
    image, from the stored probability maps, for the 63 HR LOCO cells and the
    9 HR upper bounds. Selected from an index built here.
  - Final_Experiments_Results.md, the only record of items 2f and 2g.

Every printed string is asserted to occur in final_paper at PAPER_REF; the
script fails if any is absent.

Run:  python analysis/43_provenance_backfill.py --dry-run   (prints, writes nothing)
      python analysis/43_provenance_backfill.py
"""

import hashlib
import importlib.util
import json
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
BASE = os.environ["PROJECT_ROOT"]
DRY_RUN = "--dry-run" in sys.argv
B = 10000
BOOT_SEED = 0
NN, NAME = 43, "provenance_backfill"

PAPER = os.path.join(BASE, "final_paper")
PAPER_REF = "a20cddbf04e7ce2b528633442fec3d0086fad103"   # origin/main on 2026-11-06

SM = os.environ["PROJECT_ROOT"]
FIG3 = os.path.join(SM, "data", "figure3_data")
FIG3_NB = os.path.join(SM, "Python", "FINAL-fig3.ipynb")
FER = os.path.join(BASE, "Final_Experiments_Results.md")

PNG = os.path.join(BASE, "dumps", "st_r450_published_png")
ABL = os.path.join(BASE, "dumps", "st_sib_ablation_png")
INV = os.path.join(BASE, "analysis", "_inv")
INDEX_OUT = os.path.join(INV, f"INDEX_st_r450_published_png_{NN}.parquet")
RESULTS = os.path.join(BASE, "results", f"{NN}_{NAME}.json")
R31_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "31_internal_sib_worst_case_tgtfda.py")
SEO_INDEX = os.path.join(INV, "INDEX_seo_34.parquet")

E12_LOG = os.path.join(BASE, "summaries", "session_logs", "2026-08-03-e12-retrain.md")
OGL_DIAG_LOG = os.path.join(BASE, "delta_logs", "session_logs", "2026-10-31-oglanet-sib-n450-diagnosis.md")
CE_LOG = os.path.join(BASE, "summaries", "session_logs", "2026-07-30-armf-crossearth-grid.md")
DEC_F = os.path.join(BASE, "DECISIONS_armF.md")

ARCHS = ("mamnet", "oglanet", "dinov3")
CITIES = ("chicago", "miami", "phoenix")
METHODS = ("vanilla", "fda", "segdesic", "iim", "isw", "mrfp_plus", "fada")
METHOD_LABEL = {"vanilla": "Vanilla", "fda": "FDA", "segdesic": "SegDeSiC", "iim": "IIM",
                "isw": "ISW", "mrfp_plus": "MRFP+", "fada": "FADA"}


def fail(msg):
    raise SystemExit(f"FATAL: {msg}")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, BASE) if p.startswith(BASE) else p


def load_r31():
    spec = importlib.util.spec_from_file_location("r31_rows", R31_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_show(path):
    out = subprocess.run(["git", "-C", PAPER, "show", f"{PAPER_REF}:{path}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        fail(f"git show {PAPER_REF}:{path} failed: {out.stderr.strip()}")
    return out.stdout


def r2(x, dp=2):
    """Round half away from zero at dp, as a printed table does."""
    if x is None:
        return None
    q = 10 ** dp
    return float(np.sign(x) * np.floor(abs(x) * q + 0.5) / q)


def find_line(path, needle):
    hits = [i + 1 for i, ln in enumerate(open(path, encoding="utf-8")) if needle in ln]
    if not hits:
        fail(f"{rel(path)}: no line contains {needle!r}")
    return hits


# ----------------------------------------------------------------------------
# printed strings, asserted at PAPER_REF
# ----------------------------------------------------------------------------
PRINTED_STRINGS = {
    "part1": [("appendix/app_sib_impl.tex", r"\beta_{\text{content}}(x) = \beta_{\text{base}} + \beta_{\text{scale}} \cdot \sigma"),
              ("appendix/app_sib_impl.tex", r"\label{tab:sib-hparams}"),
              ("appendix/app_sib_impl.tex", r"$\beta_{\text{base}}$        & $0.01$"),
              ("appendix/app_sib_impl.tex", r"$\beta_{\text{scale}}$       & $0.02$"),
              ("appendix/app_sib_impl.tex", r"$\beta_{\text{edge}}$        & $10^{-4}$"),
              ("appendix/app_sib_impl.tex", r"FDA spectral radius          & $0.005$"),
              ("appendix/app_sib_impl.tex", r"AdamW, weight decay          & $0.01$"),
              ("appendix/app_sib_impl.tex", r"MAMNet/OGLANet               & $1\!\times\!10^{-4}$"),
              ("appendix/app_sib_impl.tex", r"DINOv3 (encoder/head)        & $1\!\times\!10^{-5}$ / $1\!\times\!10^{-4}$"),
              ("appendix/app_sib_impl.tex", r"cosine, warmup epochs        & $5$"),
              ("appendix/app_sib_impl.tex", r"max (early stop on val mIoU) & $100$"),
              ("figures/fig_sib.tex", r"\includegraphics[width=\linewidth]{fig4_2.png}")],
    "2a": [("appendix/app_eval.tex", r"DINOv3   & $0.69\ [0.60,\ 0.75]$ & $-0.13\ [-0.27,\ -0.03]$"),
           ("appendix/app_eval.tex", r"MAMNet   & $0.67\ [0.63,\ 0.72]$ & $0.88\ [0.83,\ 0.90]$"),
           ("appendix/app_eval.tex", r"OGLANet  & $0.12\ [0.05,\ 0.21]$ & $0.86\ [0.81,\ 0.89]$"),
           ("appendix/app_eval.tex", r"$335$--$351$ valid $R$ values per architecture per experiment"),
           ("appendix/app_eval.tex", r"$\text{IoU}^{\text{upper}} - \text{IoU}^{\text{LOCO}} < 0.02$"),
           ("appendix/app_eval.tex", r"Image-cluster bootstrap ($B = 10{,}000$)"),
           ("sections/evaluation.tex", r"\textbf{Mirror-image recovery pattern.}")],
    "2b": [("figures/fig_diagnostics.tex", r"$\sim\!50\times$ smaller background gap")],
    "2c": [("appendix/app_eval.tex", r"all three medians lie in $[-2.72, -1.55]$"),
           ("appendix/app_eval.tex", r"$1.5$--$2.7\times$ further from the upper bound"),
           ("appendix/app_eval.tex", r"grow by factors of $5\times$ to $76\times$"),
           ("appendix/app_eval.tex", r"from $201$K LOCO false positives to $15.4$M after matching")],
    "2d": [("appendix/app_eval.tex", r"$60$ of $63$ cells have $\text{gap}_{\text{CE}} > \text{gap}_{0/1}$ (paired Wilcoxon $p = 2.3 \times 10^{-11}$)"),
           ("appendix/app_eval.tex", r"by a factor of $2.3\times$ on the foreground population ($+0.315$ against $+0.136$)")],
    "2e": [("appendix/app_eval.tex", r"$W = 0.79$ ($p = 0.027$)"),
           ("appendix/app_eval.tex", r"FDA achieves the smallest mean gap ($+0.05$ pooled across architectures and cities) and ISW the largest ($+0.22$)"),
           ("appendix/app_eval.tex", r"FDA $<$ Vanilla $<$ FADA $<$ SegDeSiC $<$ MRFP$^+$ $<$ IIM $<$ ISW"),
           ("appendix/app_eval.tex", r"FDA ranks first on all three architectures and ISW ranks sixth or seventh on all three"),
           ("appendix/app_eval.tex", r"with IIM, ISW, and MRFP$^+$ all clustering near $+0.22$")],
    "2f": [("appendix/app_eval.tex", r"(paired Wilcoxon against zero recovery, $p = 0.19$); on $26$ of $63$ cells")],
    "2g": [("appendix/app_eval.tex", r"(Pearson $r = -0.06$, $p = 0.67$)"),
           ("appendix/app_eval.tex", r"reveals a $2.4\times$ under-reporting")],
}


def check_printed():
    cache, out = {}, {}
    for item, pairs in PRINTED_STRINGS.items():
        out[item] = []
        for path, s in pairs:
            if path not in cache:
                cache[path] = git_show(path)
            n = cache[path].count(s)
            if n == 0:
                fail(f"printed string for {item} not found in {path} at {PAPER_REF[:8]}: {s!r}")
            line = next(i + 1 for i, ln in enumerate(cache[path].splitlines()) if s in ln)
            out[item].append(dict(file=path, line=line, text=s))
    return out


# ----------------------------------------------------------------------------
# Part 1 -- SIB configuration
# ----------------------------------------------------------------------------
MISSING = object()


def get(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return MISSING
        cur = cur[k]
    return cur


def pick(d, paths, src):
    """first present path -> {value, source}; none present -> value None with the paths tried."""
    for p in paths:
        v = get(d, p)
        if v is not MISSING:
            return dict(value=v, source=f"{src}:{p}")
    return dict(value=None, source=None, not_recorded_in=[f"{src}:{p}" for p in paths])


def secondary(value, path, needle, note=None):
    lines = find_line(path, needle)
    r = dict(value=value, source=f"{rel(path)}:{lines[0]}", source_kind="session log / decision record, not a run.json field")
    if note:
        r["note"] = note
    return r


def sib_runs():
    sets = {}
    for nn, f in (("31", "31_internal_sib_worst_case_tgtfda"), ("32", "32_method_comparison_profile_tgtfda"),
                  ("34", "34_seo_sib_family_tgtfda")):
        d = json.load(open(os.path.join(BASE, "results", f + ".json")))
        odd = [r for r in d["runs"] if r.count("-") < 6]
        if odd:
            print(f"  results/{nn} runs entries that are not run_ids (ignored): {odd}")
        sets[nn] = sorted(r for r in d["runs"] if r.count("-") >= 6 and r.split("-")[1].startswith("sib"))
    if len(sets["31"]) != 18 or sets["31"] != sets["32"] or len(sets["34"]) != 30:
        fail(f"SIB run counts changed: 31 {len(sets['31'])}, 32 {len(sets['32'])}, 34 {len(sets['34'])}")
    return sets


def seo_dirs():
    idx = pd.read_parquet(SEO_INDEX)
    out = {}
    for _, r in idx.iterrows():
        if not isinstance(r.run_id, str):
            continue
        d = os.path.join(BASE, "delta_dumps", r.dir) if r.tree == "delta_dumps" else os.path.join(BASE, r.dir)
        out.setdefault(r.run_id, []).append(dict(dir=d, category=r.category, dump_sha256=r.dump_sha256))
    return out


def resolve_dir(rid, seo):
    if rid.startswith("dinov3-sibc4-loco-r450"):
        return os.path.join(PNG, rid)
    if rid.startswith("mamnet-sibc4-loco-r450"):
        return os.path.join(BASE, "delta_dumps", "dumps", "e12", rid)
    if rid.startswith("oglanet-sibc4tgt"):
        return os.path.join(BASE, "dumps", "st_r450_published_recompute", rid)
    if rid.startswith("crossearth-sib-loco-r450"):
        return os.path.join(BASE, "dumps", "st", rid)
    cands = seo.get(rid, [])
    cell = [c for c in cands if c["category"] == "cell"] or cands
    if not cell:
        fail(f"{rid}: not in {rel(SEO_INDEX)}")
    shas = {c["dump_sha256"] for c in cands if c["dump_sha256"]}
    if len(shas) > 1:
        fail(f"{rid}: two S-EO copies with different dump_sha256")
    return cell[0]["dir"]


def companion_c4clean(arch, city):
    p = os.path.join(ABL, f"{arch}-sibc4clean-loco-r900-{city}-hr-s0", "run.json")
    return json.load(open(p)), p


FIELDS = ("use_haar", "content_vib", "edge_vib", "adaptive_beta", "beta_base", "beta_scale", "beta_edge",
          "use_content_aug", "cross_source_mixing_p", "use_sag", "use_fda", "fda_L", "fda_donor_root",
          "fda_donor_n", "use_contrast", "early_stopping_patience", "epochs_max", "epochs_run", "lr",
          "optimizer", "kl_warmup_fraction", "batch_size")


def neg(r):
    if r.get("value") is None:
        return r
    return dict(r, value=(not r["value"]), note=f"negation of {r['source']}")


def cfg_diff(a, b):
    keys = sorted(set(a) | set(b))
    return {k: [a.get(k, "<absent>"), b.get(k, "<absent>")] for k in keys if a.get(k, "<absent>") != b.get(k, "<absent>")}


def part1_record(rid, d, rj, rj_path):
    arch, city = rid.split("-")[0], rid.split("-")[4]
    rec = dict(run_id=rid, dump_dir=rel(d), run_json_sha256=sha256_file(rj_path),
               run_json_run_id=rj.get("run_id"), seed=rj.get("seed"),
               target_access=rj.get("target_access"))
    if rj.get("run_id") != rid:
        fail(f"{rid}: run.json records run_id {rj.get('run_id')!r}")
    f = {}
    src = "run.json"
    if rid.startswith("dinov3-sibc4-loco-r450"):
        comp, cp = companion_c4clean("dinov3", city)
        if comp["legacy_run_dir"] != rj["legacy_run_dir"]:
            fail(f"{rid}: legacy_run_dir differs from results/42's C4clean companion")
        cv, csrc = comp["config_verbatim"], f"{rel(cp)}:config_verbatim"
        rec.update(family="internal_published_png", is_c4clean=True,
                   c4clean_basis=f"same legacy_run_dir ({rj['legacy_run_dir']}) as results/42's {comp['run_id']}; "
                                 "this run_id scores that run's saved prediction PNGs",
                   config_source=comp.get("config_source"), config_kind=comp.get("config_kind"),
                   n_config_keys_on_analysis_host=len(cv), checkpoint_state=comp.get("checkpoint_state"))
        f["use_haar"] = pick(cv, ["use_haar"], csrc)
        f["content_vib"] = neg(pick(cv, ["disable_content_vib"], csrc))
        f["edge_vib"] = pick(cv, ["use_vib"], csrc)
        f["edge_vib"]["note"] = "no separate edge-VIB switch is recorded; use_vib covers both; vib_on_hl_only=" + str(cv.get("vib_on_hl_only"))
        f["adaptive_beta"] = pick(cv, ["adaptive_beta"], csrc)
        for k in ("beta_base", "beta_scale", "beta_edge"):
            f[k] = dict(value=None, not_recorded_in=[csrc],
                        reason=f"the {len(cv)}-key args.json record on the analysis host carries no beta key; the full args.json is "
                               f"at {comp.get('config_source')} on Delta (checkpoint {comp.get('checkpoint_state')})")
        f["use_content_aug"] = pick(cv, ["use_content_aug"], csrc)
        f["cross_source_mixing_p"] = pick(cv, ["aug_p_mix"], csrc)
        f["use_sag"] = dict(value=None, not_recorded_in=[csrc], reason="no SAG key in the DINOv3 args.json record; App J applies SAG to U-Nets only")
        f["use_fda"] = pick(cv, ["use_fda"], csrc)
        f["fda_L"] = pick(cv, ["fda_L"], csrc)
        f["fda_donor_root"] = pick(cv, ["fda_target_root"], csrc)
        f["fda_donor_n"] = dict(value=None, reason="use_fda is false")
        f["use_contrast"] = dict(value=None, not_recorded_in=[csrc], reason="no contrast key in the DINOv3 args.json record")
        f["early_stopping_patience"] = pick(cv, ["early_stopping_patience"], csrc)
        f["epochs_max"] = pick(cv, ["epochs"], csrc)
        f["epochs_run"] = dict(value=None, reason="not recorded in any artifact on the analysis host")
        f["lr"] = pick(cv, ["lr"], csrc)
        f["batch_size"] = pick(cv, ["batch_size"], csrc)
        f["optimizer"] = dict(value=None, reason="not recorded")
        f["kl_warmup_fraction"] = dict(value=None, reason="not recorded")
    elif rid.startswith("mamnet-sibc4-loco-r450"):
        comp, cp = companion_c4clean("mamnet", city)
        if rj["provenance"]["published_config_sha256"] != comp["config_sha256"]:
            fail(f"{rid}: published_config_sha256 != results/42 C4clean companion config_sha256")
        cv = dict(comp["config_verbatim"])
        ov = rj["provenance"]["overrides"]
        if cv.get("fda_target_root") != ov["fda_target_root"]["was"]:
            fail(f"{rid}: override 'was' does not equal the C4clean companion fda_target_root")
        csrc = f"{rel(cp)}:config_verbatim (same config_sha256 as run.json:provenance.published_config_sha256)"
        rec.update(family="internal_e12_retrain", is_c4clean=False,
                   c4clean_basis="a retrain of the published C4clean config (config_sha256 "
                                 f"{comp['config_sha256'][:12]}..., the config results/42's {comp['run_id']} carries) "
                                 "with one override, fda_target_root -> the held-out city's train/images, and an explicit "
                                 f"seed {rj['seed']}. Same configuration, different weights from the C4clean run results/42 read.",
                   config_overrides=ov, n_config_keys_on_analysis_host=len(cv))
        f["use_haar"] = pick(cv, ["use_haar"], csrc)
        f["content_vib"] = pick(cv, ["use_vib"], csrc)
        f["edge_vib"] = neg(pick(cv, ["no_edge_vib"], csrc))
        f["adaptive_beta"] = pick(cv, ["adaptive_beta"], csrc)
        f["beta_base"] = secondary(0.001, E12_LOG, "`beta_content=1e-3`",
                                   "MAMNet's key is beta_content; the 26-key config on the analysis host omits it; read from the E-12 log's "
                                   "verbatim list of the published config.json")
        f["beta_scale"] = dict(value=None, reason="no beta_scale key recorded for MAMNet; adaptive_beta is false so a scale term would be inactive")
        f["beta_edge"] = secondary(1e-5, E12_LOG, "`beta_edge=1e-5`")
        f["use_content_aug"] = pick(cv, ["use_content_aug"], csrc)
        f["cross_source_mixing_p"] = dict(value=None, reason="no mixing key in MAMNet's config; App app_sib_ablation states MAMNet's SIB has no cross-source mixing")
        f["use_sag"] = pick(cv, ["use_sag"], csrc)
        f["use_fda"] = pick(cv, ["use_fda"], csrc)
        f["fda_L"] = pick(rj, ["fda_L"], src)
        f["fda_donor_root"] = pick(rj, ["fda_target_pool.pool_root"], src)
        f["fda_donor_n"] = dict(value=f"{rj['fda_target_pool']['max_images']} drawn from a pool of {rj['fda_target_pool']['n_pool']}",
                                source="run.json:fda_target_pool.{max_images,n_pool}")
        f["use_contrast"] = pick(cv, ["use_contrast"], csrc)
        f["early_stopping_patience"] = pick(rj, ["early_stop_patience"], src)
        f["epochs_max"] = pick(rj, ["epochs_max"], src)
        f["epochs_run"] = pick(rj, ["epochs_run"], src)
        f["lr"] = pick(cv, ["lr"], csrc)
        f["batch_size"] = pick(cv, ["batch_size"], csrc)
        f["optimizer"] = dict(value=None, reason="not recorded in run.json or the config on the analysis host")
        f["kl_warmup_fraction"] = secondary(0.1, E12_LOG, "`vib_warmup_fraction=0.1`")
    elif rid.startswith("oglanet-sibc4tgt"):
        comp, cp = companion_c4clean("oglanet", city)
        cv = rj["config_verbatim"]
        csrc = "run.json:config_verbatim"
        shared = {k: cv[k] for k in comp["config_verbatim"] if k in cv}
        diff = cfg_diff({k: comp["config_verbatim"][k] for k in shared}, shared)
        rec.update(family="internal_n450_tgtfda_retrain", is_c4clean=False,
                   c4clean_basis="a 450-image retrain (seed 42) of the published C4clean config; FDA donor pool replaced by "
                                 "the held-out city's train images (results/31-34 lineage). Different weights from "
                                 f"results/42's {comp['run_id']}.",
                   diff_vs_c4clean_companion_on_shared_keys=diff,
                   config_diff_vs_original=rj.get("config_diff_vs_original", {}).get("changed"))
        f["use_haar"] = pick(cv, ["use_haar"], csrc)
        f["content_vib"] = neg(pick(cv, ["skip_ll_vib"], csrc))
        f["edge_vib"] = pick(cv, ["use_vib"], csrc)
        f["edge_vib"]["note"] = "vib_only_band=" + str(cv.get("vib_only_band"))
        f["adaptive_beta"] = pick(cv, ["adaptive_beta"], csrc)
        f["beta_base"] = pick(cv, ["beta_content"], csrc)
        f["beta_base"]["note"] = "OGLANet's key is beta_content"
        f["beta_scale"] = dict(value=None, not_recorded_in=[csrc],
                               reason=f"no beta_scale key; beta_max_multiplier={cv.get('beta_max_multiplier')} is recorded, "
                                      "and adaptive_beta is false")
        f["beta_edge"] = pick(cv, ["beta_edge"], csrc)
        f["use_content_aug"] = pick(cv, ["use_content_aug"], csrc)
        f["cross_source_mixing_p"] = pick(cv, ["aug_p_mix"], csrc)
        f["use_sag"] = pick(cv, ["use_sag"], csrc)
        f["use_fda"] = pick(cv, ["use_fda"], csrc)
        f["fda_L"] = pick(cv, ["fda_L"], csrc)
        f["fda_donor_root"] = pick(rj, ["fda_donor_root"], src)
        f["fda_donor_root"]["note"] = ("config_verbatim.fda_target_root is never dereferenced by train_oglanet_sib.py "
                                       "(run.json config_diff_vs_original); the donor is installed at dataset level")
        f["fda_donor_n"] = pick(rj, ["fda_donor_n"], src)
        f["use_contrast"] = pick(cv, ["use_contrast"], csrc)
        f["early_stopping_patience"] = pick(rj, ["early_stop_patience"], src)
        f["epochs_max"] = pick(rj, ["epochs_max"], src)
        f["epochs_run"] = pick(rj, ["epochs_run"], src)
        f["lr"] = pick(cv, ["lr"], csrc)
        f["batch_size"] = pick(cv, ["batch_size"], csrc)
        f["optimizer"] = secondary("Adamax (weight_decay %s), ReduceLROnPlateau" % cv.get("weight_decay"), OGL_DIAG_LOG,
                                   "| optimizer | **Adamax** |")
        f["kl_warmup_fraction"] = pick(cv, ["vib_warmup_fraction"], csrc)
    elif rid.startswith("crossearth-sib-loco-r450"):
        mc, cv = rj["method_config"], rj["config_verbatim"]
        rec.update(family="internal_crossearth_armF", is_c4clean=False,
                   c4clean_basis="arm F training run on a fourth architecture (CrossEarth); C4clean was never defined for it")
        f["use_haar"] = secondary(True, DEC_F, "D-10 — SIB on CrossEarth is the Haar+VIB path only")
        f["content_vib"] = dict(value=get(mc, "hidden_widths.content") is not MISSING,
                                source="run.json:method_config.hidden_widths.content",
                                note=f"content-VIB hidden width {get(mc, 'hidden_widths.content')} recorded, so the block exists")
        f["edge_vib"] = dict(value=get(mc, "hidden_widths.edge") is not MISSING,
                             source="run.json:method_config.hidden_widths.edge",
                             note=f"edge-VIB hidden width {get(mc, 'hidden_widths.edge')} recorded, so the block exists")
        f["adaptive_beta"] = secondary(True, CE_LOG, "`I_local` is recovered back to raw",
                                       "run.json records beta_base/beta_scale but no adaptive flag; the session log "
                                       "describes the I_local-conditioned beta as implemented")
        f["beta_base"] = pick(mc, ["beta_base"], "run.json:method_config")
        f["beta_scale"] = pick(mc, ["beta_scale"], "run.json:method_config")
        f["beta_edge"] = pick(mc, ["beta_edge"], "run.json:method_config")
        f["use_content_aug"] = dict(value=None, reason="not recorded in run.json, the arm F results.json or D-10")
        f["cross_source_mixing_p"] = dict(value=None, reason="not recorded")
        f["use_sag"] = secondary(False, DEC_F, "D-10 — SIB on CrossEarth is the Haar+VIB path only")
        f["use_fda"] = secondary(False, DEC_F, "D-10 — SIB on CrossEarth is the Haar+VIB path only")
        f["fda_L"] = dict(value=None, reason="FDA off")
        f["fda_donor_root"] = dict(value=None, reason="FDA off")
        f["fda_donor_n"] = dict(value=None, reason="FDA off")
        f["use_contrast"] = secondary(False, DEC_F, "D-10 — SIB on CrossEarth is the Haar+VIB path only")
        f["early_stopping_patience"] = pick(rj, ["early_stop_patience"], src)
        f["epochs_max"] = dict(value=f"iters_max {cv.get('iters_max')} (val_interval {cv.get('val_interval')})",
                               source="run.json:config_verbatim.iters_max", note="iteration budget, not epochs")
        f["epochs_run"] = pick(rj, ["epochs_run"], src)
        f["lr"] = pick(cv, ["lr"], "run.json:config_verbatim")
        f["batch_size"] = pick(cv, ["batch_size"], "run.json:config_verbatim")
        f["optimizer"] = pick(cv, ["optimizer"], "run.json:config_verbatim")
        f["kl_warmup_fraction"] = pick(mc, ["kl_warmup_frac"], "run.json:method_config")
    else:  # S-EO
        mb = rj["method_block"]
        sf = mb.get("sib_flags", {})
        is_tgt = rid.endswith("-tgtfda")
        rec.update(family="seo", is_c4clean=False,
                   c4clean_basis="S-EO training run (rung 3500). Its SIB flag set was registered from the published "
                                 "C4clean config" + (f" ({mb.get('fda_published_source')})" if mb.get("fda_published_source") else "")
                                 + "; it is not one of the C4clean runs results/42 read.",
                   headline_seed0=rid.split("-")[-1] == "s0" or rid.endswith("-s0-tgtfda"))
        bs = "run.json:method_block"
        if arch == "crossearth":
            off = set(mb.get("components_off", []))
            on = set(mb.get("components_active", []))
            f["use_haar"] = dict(value="haar" in on, source=f"{bs}.components_active")
            f["content_vib"] = dict(value="content_vib" in on, source=f"{bs}.components_active")
            f["edge_vib"] = dict(value="edge_vib" in on, source=f"{bs}.components_active")
            f["adaptive_beta"] = dict(value=("adaptive_beta" not in off), source=f"{bs}.components_off")
            f["beta_base"] = pick(mb, ["beta_base"], bs)
            f["beta_scale"] = pick(mb, ["beta_scale"], bs)
            f["beta_edge"] = pick(mb, ["beta_edge"], bs)
            f["use_content_aug"] = dict(value=("content_aug" not in off), source=f"{bs}.components_off")
            f["cross_source_mixing_p"] = dict(value=None, reason="not recorded")
            f["use_sag"] = dict(value=("sag" not in off), source=f"{bs}.components_off")
            f["use_fda"] = pick(mb, ["fda_active"], bs)
            f["fda_L"] = dict(value=None, reason="FDA off")
            f["fda_donor_root"] = dict(value=None, reason="FDA off")
            f["fda_donor_n"] = dict(value=None, reason="FDA off")
            f["use_contrast"] = dict(value=("contrast_channel" not in off), source=f"{bs}.components_off")
            f["kl_warmup_fraction"] = pick(mb, ["kl_warmup_frac"], bs)
            f["optimizer"] = pick(rj, ["config_verbatim.optimizer.optimizer.type"], src)
        else:
            sfs = f"{bs}.sib_flags"
            f["use_haar"] = pick(sf, ["use_haar"], sfs)
            if arch == "dinov3":
                f["content_vib"] = neg(pick(sf, ["disable_content_vib"], sfs))
                f["beta_base"] = pick(sf, ["vib_beta_content"], sfs)
                f["beta_scale"] = pick(sf, ["vib_beta_scale"], sfs)
                f["beta_edge"] = pick(sf, ["vib_beta_edge"], sfs)
            else:
                f["content_vib"] = (neg(pick(sf, ["skip_ll_vib"], sfs)) if "skip_ll_vib" in sf
                                    else pick(sf, ["use_vib"], sfs))
                f["beta_base"] = pick(sf, ["beta_content"], sfs)
                f["beta_scale"] = dict(value=None, not_recorded_in=[sfs], reason="no beta_scale key; adaptive_beta is false")
                f["beta_edge"] = pick(sf, ["beta_edge"], sfs)
            f["edge_vib"] = (neg(pick(sf, ["no_edge_vib"], sfs)) if "no_edge_vib" in sf else pick(sf, ["use_vib"], sfs))
            f["adaptive_beta"] = pick(sf, ["adaptive_beta"], sfs)
            f["use_content_aug"] = pick(sf, ["use_content_aug"], sfs)
            f["cross_source_mixing_p"] = (pick(sf, ["aug_p_mix"], sfs) if "aug_p_mix" in sf
                                          else dict(value=None, reason="no mixing key recorded"))
            f["use_sag"] = (pick(sf, ["use_sag"], sfs) if "use_sag" in sf
                            else dict(value=None, not_recorded_in=[sfs], reason="no SAG key for this architecture"))
            f["use_fda"] = pick(mb, ["fda_active"], bs)
            f["use_fda"]["note"] = (f"config_verbatim.use_fda={get(rj, 'config_verbatim.use_fda')!r}; "
                                    "method_block.fda_active is the runtime state")
            f["fda_L"] = pick(mb, ["fda_L"], bs) if mb.get("fda_active") else dict(value=None, reason="FDA off")
            if mb.get("fda_active"):
                f["fda_donor_root"] = (pick(rj, ["fda_donor_selected_path"], src) if is_tgt
                                       else pick(mb, ["fda_target_pool_manifest"], bs))
                f["fda_donor_root"]["donor_city"] = mb.get("fda_target_pool_city") if not is_tgt else rj.get("fda_donor_city")
                f["fda_donor_root"]["donor_rule"] = rj.get("fda_rule", "fda_target_pool (held-out city train split)")
                f["fda_donor_n"] = (pick(rj, ["fda_donor_n_selected"], src) if is_tgt else pick(mb, ["fda_target_pool_n"], bs))
            else:
                f["fda_donor_root"] = dict(value=None, reason="FDA off")
                f["fda_donor_n"] = dict(value=None, reason="FDA off")
            f["use_contrast"] = pick(rj, ["arch_input.use_contrast"], src)
            f["kl_warmup_fraction"] = pick(mb, ["sib_vib_warmup_fraction"], bs)
            f["optimizer"] = dict(value=None, reason="not recorded in run.json")
        f["early_stopping_patience"] = pick(rj, ["early_stop_patience"], src)
        f["epochs_max"] = pick(rj, ["epochs_max"], src)
        f["epochs_run"] = pick(rj, ["epochs_run_total", "epochs_run"], src)
        f["lr"] = pick(rj, ["config_verbatim.lr"], src)
        f["batch_size"] = pick(rj, ["config_verbatim.batch_size"], src)
    missing = [k for k in FIELDS if k not in f]
    if missing:
        fail(f"{rid}: fields not resolved: {missing}")
    rec["flags"] = {k: f[k] for k in FIELDS}
    return rec


# App J (tab:sib-hparams + equation) and Fig. 4(a), the reference each run is compared with.
APPJ = {
    "adaptive_beta": (True, "App J equation beta_content(x) = beta_base + beta_scale * sigma(MLP(I_local))"),
    "use_content_aug": (True, "Fig. 4(a) 'Content Aug [train only]'"),
    "beta_base": (0.01, "tab:sib-hparams"),
    "beta_scale": (0.02, "tab:sib-hparams"),
    "beta_edge": (1e-4, "tab:sib-hparams and App J prose"),
    "use_haar": (True, "App J / Fig. 4(a)"),
    "content_vib": (True, "App J / Fig. 4(a)"),
    "edge_vib": (True, "App J / Fig. 4(a)"),
    "kl_warmup_fraction": (0.10, "tab:sib-hparams"),
    "batch_size": (8, "tab:sib-hparams"),
    "epochs_max": (100, "tab:sib-hparams"),
    "optimizer": ("AdamW, weight decay 0.01", "tab:sib-hparams"),
}
APPJ_CNN = {"use_sag": (True, "App J, Fig. 4(a): SAG on U-Nets"), "use_fda": (True, "App J, Fig. 4(a): FDA on CNN inputs"),
            "fda_L": (0.005, "tab:sib-hparams"), "use_contrast": (True, "App J: contrast channel on CNNs"),
            "lr": (1e-4, "tab:sib-hparams MAMNet/OGLANet")}
APPJ_VIT = {"use_sag": (False, "App J: SAG not applied for DINOv3"), "use_fda": (False, "App J: ViT inputs receive no FDA"),
            "use_contrast": (False, "App J: ViT inputs receive no contrast channel"),
            "lr": ("1e-5 encoder / 1e-4 head", "tab:sib-hparams DINOv3")}
TRUTHY_ABSENT = {"content_vib", "edge_vib", "use_haar"}


def compare_to_appj(arch, recs):
    ref = dict(APPJ)
    ref.update(APPJ_CNN if arch in ("mamnet", "oglanet") else APPJ_VIT)
    differ, agree, unverifiable = {}, [], []
    for k, (want, why) in ref.items():
        vals = []
        for r in recs:
            fl = r["flags"][k]
            vals.append(fl.get("value"))
        known = [v for v in vals if v is not None]
        if not known:
            unverifiable.append(k)
            continue
        uniq = sorted({json.dumps(v) for v in known})
        if k == "lr" and arch not in ("mamnet", "oglanet"):
            differ[k] = dict(recorded=[json.loads(u) for u in uniq], app_j=want, where=why)
            continue
        if k == "optimizer":
            okv = all("AdamW" in str(v) and re.search(r"(wd|weight_decay)\W*0\.01\b", str(v)) for v in known)
            (agree.append(k) if okv else differ.__setitem__(k, dict(recorded=[json.loads(u) for u in uniq], app_j=want, where=why)))
            if len(known) < len(vals):
                unverifiable.append(f"{k} (recorded on {len(known)} of {len(vals)} runs)")
            continue
        if k == "epochs_max" and all(isinstance(v, str) for v in known):
            differ[k] = dict(recorded=[json.loads(u) for u in uniq], app_j=want, where=why)
            continue
        ok = all((abs(v - want) < 1e-12) if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(want, (int, float))
                 else v == want for v in known)
        if ok:
            agree.append(k)
        else:
            differ[k] = dict(recorded=[json.loads(u) for u in uniq], app_j=want, where=why)
        if len(known) < len(vals):
            unverifiable.append(f"{k} (recorded on {len(known)} of {len(vals)} runs)")
    return dict(differ=differ, agree=agree, unverifiable=unverifiable)


def part1(sets):
    seo = seo_dirs()
    union = sorted(set(sets["31"]) | set(sets["34"]))
    recs = []
    for rid in union:
        d = resolve_dir(rid, seo)
        p = os.path.join(d, "run.json")
        if not os.path.isfile(p) or not os.path.isfile(os.path.join(d, "perimage.parquet")):
            fail(f"{rid}: run.json or perimage.parquet missing in {rel(d)}")
        rec = part1_record(rid, d, json.load(open(p)), p)
        rec["in_results"] = [nn for nn in ("31", "32", "34") if rid in sets[nn]]
        recs.append(rec)
    groups = {
        "internal_9cell_and_12cell (results/31, /32; section 5.2, Fig. 4(b))": {
            "mamnet": [r for r in recs if r["family"] == "internal_e12_retrain"],
            "oglanet": [r for r in recs if r["family"] == "internal_n450_tgtfda_retrain"],
            "dinov3": [r for r in recs if r["family"] == "internal_published_png"],
            "crossearth": [r for r in recs if r["family"] == "internal_crossearth_armF"]},
        "seo (results/34; S-EO SIB row)": {
            a: [r for r in recs if r["family"] == "seo" and r["run_id"].startswith(a + "-")]
            for a in ("mamnet", "oglanet", "dinov3", "crossearth")},
    }
    verdicts = {}
    for g, by_arch in groups.items():
        verdicts[g] = {}
        for a, rs in by_arch.items():
            if not rs:
                fail(f"{g}/{a}: no runs")
            cmp = compare_to_appj("dinov3" if a == "crossearth" else a, rs)
            names = sorted(cmp["differ"])
            line = (f"{a}: {'does not match' if names else 'matches'} App J tab:sib-hparams + Fig. 4(a) on the recorded "
                    f"fields over {len(rs)} run(s)"
                    + (" (App J lists no CrossEarth row; compared with its ViT row)" if a == "crossearth" else "")
                    + (f"; differing flags: {', '.join(names)}" if names else "")
                    + (f"; not verifiable on the analysis host: {', '.join(cmp['unverifiable'])}" if cmp["unverifiable"] else ""))
            verdicts[g][a] = dict(n_runs=len(rs), run_ids=[r["run_id"] for r in rs], one_line=line, **cmp)
    return recs, verdicts


# ----------------------------------------------------------------------------
# Part 2
# ----------------------------------------------------------------------------
def boot_median(v, b, seed):
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    meds = np.empty(b)
    for i in range(b):
        meds[i] = np.median(v[rng.integers(0, len(v), len(v))])
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def notebook_boot_median(values, n_boot=1000, seed=42):
    """FINAL-fig3.ipynb bootstrap_median, verbatim logic, for Fig. 3(c)'s error bars."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.RandomState(seed)
    boots = [np.median(v[rng.randint(0, len(v), size=len(v))]) for _ in range(n_boot)]
    return float(np.median(v)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def item_2a(png_cells):
    rr = json.load(open(os.path.join(FIG3, "robust_recovery_results.json")))
    full = json.load(open(os.path.join(FIG3, "robust_recovery_full.json")))
    ps = json.load(open(os.path.join(FIG3, "paper_statistics.json")))
    printed = {"a": {"dinov3": (0.69, 0.60, 0.75), "mamnet": (0.67, 0.63, 0.72), "oglanet": (0.12, 0.05, 0.21)},
               "b2": {"dinov3": (-0.13, -0.27, -0.03), "mamnet": (0.88, 0.83, 0.90), "oglanet": (0.86, 0.81, 0.89)}}
    rows, all_match = [], True
    for exp in ("a", "b2"):
        for arch in ("dinov3", "mamnet", "oglanet"):
            s = rr[exp]["cross_architecture"]["per_arch_summary"][arch]
            pooled, excl = [], []
            for c in CITIES:
                cell = full[exp]["cells"][f"{exp}_{arch}_{c}_highres"]
                pooled += cell["per_image_R"]["raw_values"]
                excl.append(dict(city=c, **cell["exclusion_info"]))
            pooled = np.asarray(pooled, float)
            med = float(np.median(pooled))
            if abs(med - s["median_R"]) > 1e-12 or len(pooled) != s["n"]:
                fail(f"2a {exp}/{arch}: median of raw_values {med} / n {len(pooled)} != stored {s['median_R']} / {s['n']}")
            lo, hi = boot_median(pooled, B, BOOT_SEED)
            nb = notebook_boot_median(pooled)
            pr = printed[exp][arch]
            stored = (s["median_R"], *s["ci_95"])
            m = [r2(x) == p for x, p in zip(stored, pr)]
            all_match &= all(m)
            row = dict(experiment="A (decoder)" if exp == "a" else "B (encoder, key b2)", arch=arch,
                       printed=dict(median=pr[0], ci95=[pr[1], pr[2]]),
                       stored=dict(median=s["median_R"], ci95=s["ci_95"], n=s["n"], mean=s["mean_R"]),
                       stored_rounded=[r2(x) for x in stored],
                       matches_printed=dict(median=m[0], ci_lo=m[1], ci_hi=m[2]),
                       recomputed_here=dict(median=med, ci95=[lo, hi], B=B, seed=BOOT_SEED,
                                            method="percentile bootstrap of the median over pooled per-image R "
                                                   "(one R per image, so the image is the cluster)"),
                       fig3c_errorbar=dict(median=nb[0], ci95=[nb[1], nb[2]], B=1000, seed=42,
                                           method="FINAL-fig3.ipynb bootstrap_median on the same pooled raw_values"),
                       exclusions_per_city=excl)
            if exp == "a":
                pm = ps["diagnostic_3"]["decoder_retraining"]["per_model"][arch]
                row["paper_statistics_same"] = (abs(pm["median_R"] - s["median_R"]) < 1e-12 and pm["ci95"] == s["ci_95"])
            rows.append(row)
    ns = sorted({r["stored"]["n"] for r in rows})
    thr = ps["metadata"]["min_gap_for_R"]
    # metric/K: compare the per-image IoU behind R with the dump columns, all 18 vanilla/upper HR cells
    er = json.load(open(os.path.join(FIG3, "experiment_results.json")))
    kcheck = []
    for arch in ARCHS:
        for c in CITIES:
            g = er[f"a_{arch}_{c}_highres"]["global"]["per_image_iou"]
            for role, rid in (("loco", f"{arch}-vanilla-loco-r450-{c}-hr-s0"), ("upper", f"{arch}-none-upper-r450-{c}-hr-s0")):
                df = png_cells[rid]
                ref = np.array([np.nan if x is None else x for x in g[role]], float)
                if len(ref) != len(df):
                    fail(f"2a K check {rid}: {len(ref)} per-image IoUs vs {len(df)} rows")
                out = dict(cell=rid, n_null_in_record=int(np.isnan(ref).sum()),
                           n_empty_denominator_k2=int(((df.tp_tol + df.fp_tol + df.fn_tol) == 0).sum()))
                for k, col in ((0, "iou_str"), (2, "iou_tol"), (5, "iou_k5")):
                    v = df[col].to_numpy(float)
                    v = v / 100 if np.nanmax(v) > 1.5 else v
                    ok = ~np.isnan(ref) & ~np.isnan(v)
                    out[f"K{k}"] = dict(max_abs_diff=float(np.max(np.abs(v[ok] - ref[ok]))),
                                        pearson_r=float(np.corrcoef(v[ok], ref[ok])[0, 1]))
                kcheck.append(out)
    drift = max(abs(a - b) for r in rows for a, b in zip(r["recomputed_here"]["ci95"], r["stored"]["ci95"]))
    drift_at = max(((abs(a - b), f"{r['arch']} Exp {r['experiment'][0]}") for r in rows
                    for a, b in zip(r["recomputed_here"]["ci95"], r["stored"]["ci95"])))[1]
    best = {k: max(x[k]["max_abs_diff"] for x in kcheck) for k in ("K0", "K2", "K5")}
    rmin = {k: min(x[k]["pearson_r"] for x in kcheck) for k in ("K0", "K2", "K5")}
    return dict(
        item="2a", where="section 4.4 'Mirror-image recovery pattern'; App G tab:locus-summary; Fig. 3(c)",
        source=dict(kind="existing artifact (not a results JSON)",
                    path=rel(os.path.join(FIG3, "robust_recovery_results.json")),
                    key_path="{a,b2}.cross_architecture.per_arch_summary.{dinov3,mamnet,oglanet}.{median_R,ci_95,n}",
                    sha256=sha256_file(os.path.join(FIG3, "robust_recovery_results.json")),
                    also=[dict(path=rel(os.path.join(FIG3, "robust_recovery_full.json")),
                               key_path="{a,b2}.cells.*.per_image_R.raw_values (per-image R; recomputed here)",
                               sha256=sha256_file(os.path.join(FIG3, "robust_recovery_full.json"))),
                          dict(path=rel(os.path.join(FIG3, "paper_statistics.json")),
                               key_path="diagnostic_3.decoder_retraining.per_model (Exp A only; identical)")]),
        printed="Exp A: DINOv3 0.69 [0.60, 0.75], MAMNet 0.67 [0.63, 0.72], OGLANet 0.12 [0.05, 0.21]; "
                "Exp B: DINOv3 -0.13 [-0.27, -0.03], MAMNet 0.88 [0.83, 0.90], OGLANet 0.86 [0.81, 0.89]; "
                "335-351 valid R per architecture per experiment; exclusion IoU gap < 0.02",
        value=dict(rows=rows, n_valid_per_arch=ns, exclusion_threshold=thr,
                   metric=("per-image foreground IoU with a boundary-tolerant band, from the published scorer "
                           "(experiment_results.json global.per_image_iou); R = (IoU_exp - IoU_LOCO)/(IoU_upper - IoU_LOCO)"),
                   metric_K_check=dict(
                       cells=len(kcheck), max_abs_diff_by_K=best, min_pearson_r_by_K=rmin,
                       reading=("closest to K=2 (iou_tol) on all 18 vanilla/upper HR cells but not identical to it; "
                                "the published per-image IoU is not reproduced by any of K=0/2/5 in this metric module"),
                       per_cell=kcheck)),
        matches_printed=dict(
            all=bool(all_match),
            n_valid=ns == [335, 338, 351], threshold=thr == 0.02,
            mismatches=[f"{r['arch']} Exp {r['experiment'][0]} {k}: printed {p} vs stored {s:.4f}"
                        for r in rows for k, p, s in (("ci_lo", r["printed"]["ci95"][0], r["stored"]["ci95"][0]),
                                                      ("ci_hi", r["printed"]["ci95"][1], r["stored"]["ci95"][1]),
                                                      ("median", r["printed"]["median"], r["stored"]["median"]))
                        if r2(s) != p]),
        caveats=[
            f"The CI's second decimal is draw-dependent: a fresh B={B} percentile bootstrap (seed {BOOT_SEED}) on the same "
            f"pooled R moves a bound by up to {drift:.4f} ({drift_at}) against the stored CI, and the stored generator's "
            "resampling unit and seed are not recorded. The medians reproduce exactly.",
            "Not a results JSON before this one: the values sit in Shade Maps/data/figure3_data, written by a generator "
            "that is not on the analysis host (paper_statistics.json metadata.version v2_robustness_hardened_no_floor, 2026-04-29).",
            "The table's CIs are the stored ones; Fig. 3(c)'s error bars are a separate B=1000, seed-42 re-bootstrap in "
            "FINAL-fig3.ipynb, so the figure and the table show different intervals for the same medians.",
            "Metric is per-image foreground IoU, not the two-class mIoU the paper reports elsewhere at K=2. Its band is "
            "nearest K=2 but not identical (see metric_K_check).",
            "OGLANet carries E-9 (inference nondeterminism): every OGLANet IoU behind R is one draw.",
            "Single seed per cell; the Exp A / Exp B learning rates are unrecorded in any config artifact (E-11).",
            "No per-image R from these runs exists in CONTRACT dump form; the per-image IoU inputs of Exp A/B are not dumps.",
        ])


def item_2b(s1):
    fg = {(a, m, c): s1[f"{a}|{m}|{c}|gt_shadow"]["gap"] for a in ARCHS for m in METHODS for c in CITIES}
    bg = {(a, m, c): s1[f"{a}|{m}|{c}|gt_bg"]["gap"] for a in ARCHS for m in METHODS for c in CITIES}
    per_m = {}
    for m in METHODS:
        f = float(np.mean([fg[(a, m, c)] for a in ARCHS for c in CITIES]))
        b = float(np.mean([bg[(a, m, c)] for a in ARCHS for c in CITIES]))
        per_m[m] = dict(fg_gap=f, bg_gap=b, ratio=f / b)
    pf, pb = float(np.mean(list(fg.values()))), float(np.mean(list(bg.values())))
    ratios = [v["ratio"] for v in per_m.values()]
    # the notebook's own computation from cell_rcs.pkl
    with open(os.path.join(FIG3, "cell_rcs.pkl"), "rb") as fh:
        cell_rcs = pickle.load(fh)["cell_rcs"]
    grid = np.linspace(0.10, 1.00, 19)
    trap = getattr(np, "trapezoid", None) or np.trapz
    nb_diff = 0.0
    for m in METHODS:
        for mode, ref in (("gt_shadow", fg), ("gt_bg", bg)):
            for a in ARCHS:
                for c in CITIES:
                    U, L = cell_rcs[f"{a}_upper_{c}"][mode], cell_rcs[f"{a}_{m}_{c}"][mode]
                    common = sorted(set(U) & set(L))
                    g = (float(trap(np.stack([L[s] for s in common]).mean(0), grid))
                         - float(trap(np.stack([U[s] for s in common]).mean(0), grid)))
                    nb_diff = max(nb_diff, abs(g - ref[(a, m, c)]))
    n_bg_nonpos = int(sum(v <= 0 for v in bg.values()))
    cands = dict(ratio_of_pooled_means_63_cells=pf / pb,
                 mean_of_per_method_ratios=float(np.mean(ratios)),
                 median_of_per_method_ratios=float(np.median(ratios)),
                 per_method_ratio_range=[float(min(ratios)), float(max(ratios))],
                 median_fg_over_median_bg=float(np.median(list(fg.values())) / np.median(list(bg.values()))))
    return dict(
        item="2b", where="Fig. 3(b) image ('~50x asymmetry') and caption ('~50x smaller background gap')",
        source=dict(kind="new computation from an existing artifact",
                    path=rel(os.path.join(FIG3, "test_s1_v2_results.json")),
                    key_path="{arch}|{method}|{city}|{gt_shadow,gt_bg}.gap (63 cells x 2 populations)",
                    sha256=sha256_file(os.path.join(FIG3, "test_s1_v2_results.json")),
                    cross_check=dict(path=rel(os.path.join(FIG3, "cell_rcs.pkl")),
                                     what="FINAL-fig3.ipynb per_method_pooled_gaps recomputed on common images",
                                     max_abs_diff_vs_test_s1_v2=nb_diff),
                    note="no artifact records a 50x figure; the '~50x' text is on the composed figure only"),
        printed="~50x", value=dict(primary=cands["ratio_of_pooled_means_63_cells"],
                                    primary_definition="mean foreground SP gap over 63 cells / mean background SP gap over 63 cells",
                                    pooled_fg_gap=pf, pooled_bg_gap=pb, candidates=cands, per_method=per_m,
                                    n_cells_bg_gap_nonpositive=n_bg_nonpos, n_cells=63),
        matches_printed=False,
        caveats=[f"No definition tried reproduces 50x: ratio of pooled means {pf / pb:.2f}x, mean of the seven per-method "
                 f"ratios {np.mean(ratios):.1f}x, median {np.median(ratios):.1f}x, per-method range "
                 f"{min(ratios):.1f}x-{max(ratios):.1f}x.",
                 "'across all 7 methods' does not hold for a single factor: FADA is 6.1x and IIM 86.3x.",
                 f"{n_bg_nonpos} of 63 cells have a background gap <= 0, so a per-cell ratio is undefined or negative there.",
                 "AURC here is 0/1 risk on a 19-point coverage grid (trapezoid), the S1 v2 definition; App G's bootstrap is B=1,000."])


def item_2c():
    ps = json.load(open(os.path.join(FIG3, "paper_statistics.json")))
    er = json.load(open(os.path.join(FIG3, "experiment_results.json")))
    pm = ps["diagnostic_3"]["histogram_matching"]["per_model"]
    meds = {a: pm[a]["median_R"] for a in ARCHS}
    cis = {a: pm[a]["ci95"] for a in ARCHS}
    lo, hi = min(meds.values()), max(meds.values())
    fp = []
    for a in ARCHS:
        for c in CITIES:
            g = er[f"c_{a}_{c}_highres"]["global"]
            fp.append(dict(cell=f"{a}/{c}", fp_loco=g["loco"]["fp"], fp_matched=g["experiment"]["fp"],
                           fp_upper=g["upper"]["fp"], inflation=g["experiment"]["fp"] / g["loco"]["fp"],
                           iou_loco=g["loco"]["iou"], iou_upper=g["upper"]["iou"], iou_matched=g["experiment"]["iou"],
                           distance_ratio=(g["upper"]["iou"] - g["experiment"]["iou"]) / (g["upper"]["iou"] - g["loco"]["iou"])))
    infl = [x["inflation"] for x in fp]
    mc = next(x for x in fp if x["cell"] == "mamnet/chicago")
    dist_from_R = sorted(1 - v for v in meds.values())
    return dict(
        item="2c", where="App G.3 'Recovery and inflation'",
        source=dict(kind="existing artifacts (not results JSONs)",
                    path=rel(os.path.join(FIG3, "paper_statistics.json")),
                    key_path="diagnostic_3.histogram_matching.per_model.{mamnet,oglanet,dinov3}.{median_R,ci95,n}",
                    sha256=sha256_file(os.path.join(FIG3, "paper_statistics.json")),
                    also=dict(path=rel(os.path.join(FIG3, "experiment_results.json")),
                              key_path="c_{arch}_{city}_highres.global.{loco,experiment,upper}.{fp,iou}  (experiment c = histogram matching)",
                              sha256=sha256_file(os.path.join(FIG3, "experiment_results.json")))),
        printed="median R in [-2.72, -1.55], CIs below zero; '1.5-2.7x further from the upper bound'; FP inflation 5x-76x per cell; "
                "MAMNet/Chicago 201K -> 15.4M FP pixels",
        value=dict(median_R=meds, ci95=cis, n={a: pm[a]["n"] for a in ARCHS}, median_R_range=[lo, hi],
                   all_ci_below_zero=all(c[1] < 0 for c in cis.values()),
                   distance_to_upper_ratio_at_median_R=dict(definition="1 - median R = (upper - matched)/(upper - LOCO) at the median",
                                                            range=[dist_from_R[0], dist_from_R[-1]]),
                   abs_median_R_range=[abs(hi), abs(lo)],
                   fp_per_cell=fp, fp_inflation_range=[min(infl), max(infl)],
                   n_cells_fp_increase=int(sum(v > 1 for v in infl)), n_cells_fp_ge_5x=int(sum(v >= 5 for v in infl)),
                   mamnet_chicago=dict(fp_loco=mc["fp_loco"], fp_matched=mc["fp_matched"], inflation=mc["inflation"]),
                   cells_improving_context=dict(source="paper_statistics.json diagnostic_1.histogram_matching",
                                                n_improved=ps["diagnostic_1"]["histogram_matching"]["n_improved"],
                                                n=ps["diagnostic_1"]["histogram_matching"]["n"])),
        matches_printed=dict(
            median_range=[r2(lo), r2(hi)] == [-2.72, -1.55],
            ci_below_zero=all(c[1] < 0 for c in cis.values()),
            further_from_upper_1p5_to_2p7=False,
            further_from_upper_note=(f"printed 1.5-2.7 equals |median R| ({abs(hi):.2f}-{abs(lo):.2f}) at 1 dp; the distance to "
                                     f"the upper bound relative to LOCO at the median R is 1 - R = {dist_from_R[0]:.2f}-{dist_from_R[-1]:.2f}"),
            fp_5x_to_76x=False,
            fp_note=(f"inflation over the 9 cells spans {min(infl):.2f}x-{max(infl):.2f}x; FP rise on "
                     f"{int(sum(v > 1 for v in infl))} of 9 and reach >=5x on {int(sum(v >= 5 for v in infl))} "
                     "(MAMNet/Chicago 76.39x, DINOv3/Chicago 5.97x); 4 cells lose FP pixels"),
            mamnet_chicago_201K_to_15p4M=(round(mc["fp_loco"] / 1e3) == 201 and round(mc["fp_matched"] / 1e5) / 10 == 15.4)),
        caveats=["results/20 holds only the 0-of-26 / 0-of-36 counts; the values here come from the published figure-3 "
                 "data files, not from a dump.",
                 "FP counts are pooled over 150 images from the published scorer's tolerant band; they are not the dumps' "
                 "K=2 counts (MAMNet/Chicago Vanilla LOCO: 201,346 here, fp_tol 194,311 and fp_str 322,151 in the PNG-tree dump).",
                 "R excludes images with upper-minus-LOCO IoU < 0.02 (min_gap_for_R) and images with undefined IoU.",
                 "OGLANet cells carry E-9."])


def item_2d(png_cells):
    rows = list(np.load(os.path.join(FIG3, "rows.npy"), allow_pickle=True))
    if len(rows) != 63:
        fail(f"rows.npy has {len(rows)} rows, expected 63")
    rec = {(r["model"], r["method"], r["city"]): r for r in rows}
    g01, gce, r01, rce = [], [], [], []
    for a in ARCHS:
        for c in CITIES:
            U = png_cells[f"{a}-none-upper-r450-{c}-hr-s0"].set_index("image_id")
            for m in METHODS:
                L = png_cells[f"{a}-{m}-loco-r450-{c}-hr-s0"].set_index("image_id")
                ids = U.index.intersection(L.index)
                u, l = U.loc[ids], L.loc[ids]
                ok = u.aurc_fg.notna() & l.aurc_fg.notna() & u.aurc_ce_fg.notna() & l.aurc_ce_fg.notna()
                g01.append(float(l.aurc_fg[ok].mean() - u.aurc_fg[ok].mean()))
                gce.append(float(l.aurc_ce_fg[ok].mean() - u.aurc_ce_fg[ok].mean()))
                r01.append(rec[(a, m, c)]["gap_01"])
                rce.append(rec[(a, m, c)]["gap_ce"])
    g01, gce, r01, rce = map(np.asarray, (g01, gce, r01, rce))
    n_rec, n_dump = int((rce > r01).sum()), int((gce > g01).sum())
    w1 = stats.wilcoxon(rce, r01, alternative="greater")
    w2 = stats.wilcoxon(rce, r01)
    wd1 = stats.wilcoxon(gce, g01, alternative="greater")
    return dict(
        item="2d", where="App G.5",
        source=dict(kind="existing artifact + recomputation from dumps",
                    path=rel(os.path.join(FIG3, "rows.npy")), key_path="[i].{model,method,city,gap_ce,gap_01} (63 rows, F5)",
                    sha256=sha256_file(os.path.join(FIG3, "rows.npy")),
                    recomputed_from=("dumps/st_r450_published_png: mean aurc_fg and aurc_ce_fg over images with both "
                                     "defined, LOCO minus upper on matched image_ids, 63 HR LOCO cells and 9 HR uppers"),
                    write_up=f"{rel(FER)}:{find_line(FER, 'Paired Wilcoxon: p = 2.26e-11')[0]}"),
        printed="60 of 63 cells with gap_CE > gap_0/1, paired Wilcoxon p = 2.3e-11",
        value=dict(n_cells=63, n_ce_exceeds_01_record=n_rec, n_ce_exceeds_01_dumps=n_dump,
                   wilcoxon_one_sided_greater=dict(statistic=float(w1.statistic), p=float(w1.pvalue)),
                   wilcoxon_two_sided=dict(statistic=float(w2.statistic), p=float(w2.pvalue)),
                   wilcoxon_one_sided_dumps_p=float(wd1.pvalue),
                   pooled_gap_01=float(r01.mean()), pooled_gap_ce=float(rce.mean()), pooled_ratio=float(rce.mean() / r01.mean()),
                   dumps_vs_record_max_abs_diff=dict(gap_01=float(np.max(np.abs(g01 - r01))),
                                                     gap_ce=float(np.max(np.abs(gce - rce))))),
        matches_printed=bool(n_rec == 60 and r2(w1.pvalue * 1e11, 1) == 2.3),
        caveats=["The printed p is ONE-tailed (alternative gap_CE > gap_0/1, as the F5 write-up states); two-tailed it is "
                 f"{w2.pvalue:.2e}. The paper does not say which.",
                 "The paper states the under-reporting factor as 2.3x in App G.5 and as 2.4x in the 'No deterministic per-cell "
                 "mechanism' paragraph (app_eval.tex) for the same quantity; the record gives "
                 f"{rce.mean() / r01.mean():.3f}.",
                 "AURC on the 19-point grid; image sets matched per cell. 63 cells = 7 methods x 3 architectures x 3 cities, HR, single seed.",
                 "OGLANet cells carry E-9."])


def item_2e(s1, png_cells):
    fg = {(a, m, c): s1[f"{a}|{m}|{c}|gt_shadow"]["gap"] for a in ARCHS for m in METHODS for c in CITIES}
    per_arch = {a: {m: float(np.mean([fg[(a, m, c)] for c in CITIES])) for m in METHODS} for a in ARCHS}
    ranks = {a: dict(zip(METHODS, stats.rankdata([per_arch[a][m] for m in METHODS]).astype(int).tolist())) for a in ARCHS}
    R = np.array([[ranks[a][m] for m in METHODS] for a in ARCHS], float)
    k, n = R.shape
    Rs = R.sum(0)
    W = float(12 * ((Rs - Rs.mean()) ** 2).sum() / (k ** 2 * (n ** 3 - n)))
    chi = k * (n - 1) * W
    p = float(stats.chi2.sf(chi, n - 1))
    pooled = {m: float(np.mean([fg[(a, m, c)] for a in ARCHS for c in CITIES])) for m in METHODS}
    order = sorted(METHODS, key=lambda m: pooled[m])
    printed_order = ["fda", "vanilla", "fada", "segdesic", "mrfp_plus", "iim", "isw"]
    # dumps cross-check of the 0/1 foreground gap
    dmax = 0.0
    for a in ARCHS:
        for c in CITIES:
            U = png_cells[f"{a}-none-upper-r450-{c}-hr-s0"].set_index("image_id")
            for m in METHODS:
                L = png_cells[f"{a}-{m}-loco-r450-{c}-hr-s0"].set_index("image_id")
                ids = U.index.intersection(L.index)
                ok = U.loc[ids].aurc_fg.notna() & L.loc[ids].aurc_fg.notna()
                g = float(L.loc[ids].aurc_fg[ok].mean() - U.loc[ids].aurc_fg[ok].mean())
                dmax = max(dmax, abs(g - fg[(a, m, c)]))
    isw = {a: ranks[a]["isw"] for a in ARCHS}
    top2 = {a: sorted(METHODS, key=lambda m: ranks[a][m])[:2] for a in ARCHS}
    bot2 = {a: sorted(METHODS, key=lambda m: ranks[a][m])[-2:] for a in ARCHS}
    mam = {m: per_arch["mamnet"][m] for m in ("iim", "isw", "mrfp_plus")}
    return dict(
        item="2e", where="App G.6 'Method-ranking stability'",
        source=dict(kind="new computation from an existing artifact, cross-checked against dumps",
                    path=rel(os.path.join(FIG3, "test_s1_v2_results.json")),
                    key_path="{arch}|{method}|{city}|gt_shadow.gap",
                    sha256=sha256_file(os.path.join(FIG3, "test_s1_v2_results.json")),
                    dumps_max_abs_diff=dmax,
                    write_up=f"{rel(FER)}:{find_line(FER, 'Kendall' + chr(39) + 's W = 0.794')[0]}"),
        printed="Kendall W = 0.79 (p = 0.027) over 7 methods; FDA +0.05 and ISW +0.22 pooled; FDA < Vanilla < FADA < SegDeSiC < "
                "MRFP+ < IIM < ISW; FDA first and ISW sixth or seventh on every architecture; MAMNet IIM/ISW/MRFP+ near +0.22",
        value=dict(kendall_W=W, chi2=chi, df=n - 1, p=p, ranks_1_is_smallest_gap=ranks,
                   per_arch_mean_fg_gap=per_arch, pooled_fg_gap=pooled, pooled_order=[METHOD_LABEL[m] for m in order],
                   isw_rank=isw, mrfp_plus_rank={a: ranks[a]["mrfp_plus"] for a in ARCHS},
                   top_two=top2, bottom_two=bot2, mamnet_iim_isw_mrfp=mam),
        matches_printed=dict(
            kendall_W=r2(W) == 0.79, p=r2(p, 3) == 0.027,
            fda_pooled=r2(pooled["fda"]) == 0.05, isw_pooled=r2(pooled["isw"]) == 0.22,
            ordering=order == printed_order,
            ordering_note=f"data order {[METHOD_LABEL[m] for m in order]}; printed swaps IIM ({pooled['iim']:.4f}) and MRFP+ ({pooled['mrfp_plus']:.4f}). "
                          "Fig. 3(b)'s bars follow the data order.",
            fda_first_everywhere=all(ranks[a]["fda"] == 1 for a in ARCHS),
            isw_sixth_or_seventh_everywhere=all(isw[a] >= 6 for a in ARCHS),
            isw_note=f"ISW ranks {isw}; MRFP+ ranks {dict((a, ranks[a]['mrfp_plus']) for a in ARCHS)}",
            mamnet_near_0p22={m: r2(v) for m, v in mam.items()}),
        caveats=["The App G.6 sentence 'top-two and bottom-two placements are stable' does not hold in the record: top two "
                 f"{top2}, bottom two {bot2}.",
                 "Kendall W is over 3 architectures x 7 methods, each method's gap averaged over 3 cities; n=7 gives the "
                 "chi-square approximation little accuracy.",
                 "The F4 write-up's rank table agrees with the ranks computed here.",
                 "OGLANet cells carry E-9."])


def item_2f():
    l3 = find_line(FER, "F3 null. Mean rec_donor = −0.017")
    lsum = find_line(FER, "F3 donor > raw: sign p = 0.19, null")
    return dict(
        item="2f", where="App G.6 'Post-hoc temperature scaling'",
        source=None,
        source_reason=("No per-cell output of any temperature-scaling test is on the analysis host (searched figure3_data, delta_runs, "
                       "runs, results). The only record is the markdown write-up "
                       f"{rel(FER)} 'Follow-up F' (lines {l3[0]}, {lsum[0]}). A recomputation would need the stored "
                       "probability maps (Test_img_probs on Delta) for the temperature fits; that is CPU work but the maps are not on this box."),
        printed="paired Wilcoxon p = 0.19 against zero recovery; 26 of 63 cells where the source-fitted T worsens held-out performance",
        value=None,
        nearest_record=dict(
            text="F3 null. Mean rec_donor = -0.017 across 108 (cell x donor) pairs; 59/108 with donor recovery > 0; sign test p = 0.193.",
            test="sign test, not Wilcoxon", n="108 cell x donor pairs over the n=54 trustworthy pool, not 63 cells",
            temperature="class-conditional (T_pos, T_neg) fitted on another city's calibration split, not a single T on the source-city validation split"),
        matches_printed=False,
        caveats=["'26 of 63' does not appear in any record on the analysis host. From the write-up, donor recovery is <= 0 on 49 of 108 "
                 "cell x donor pairs.",
                 "The printed p = 0.19 matches the F3 sign test (0.193) at 2 dp, but the paper names a Wilcoxon test and a "
                 "different calibration protocol."])


def item_2g():
    l = find_line(FER, "A2 null. Pearson r = −0.055 (p = 0.67)")
    return dict(
        item="2g", where="App G 'No deterministic per-cell mechanism' (app_eval.tex)",
        source=None,
        source_reason=("The per-cell log(T_pos/T_neg) from S2 v3 is not on the analysis host; only the markdown write-up "
                       f"{rel(FER)}:{l[0]} records the correlation. The logit-asymmetry side is computable from the dumps "
                       "(logit_mean_tp / logit_mean_fp), the temperature-ratio side needs the stored probability maps on Delta."),
        printed="Pearson r = -0.06, p = 0.67",
        value=None,
        nearest_record=dict(text="A2 null. Pearson r = -0.055 (p = 0.67); Spearman r = -0.062 (p = 0.63); n = 63 cells"),
        matches_printed="the write-up's -0.055 and 0.67 round to the printed -0.06 and 0.67; not verifiable against data",
        caveats=["Unverified: no per-cell record behind the correlation is on the analysis host."])


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    if os.path.exists(RESULTS):
        fail(f"{rel(RESULTS)} exists; move it out before re-running")
    if os.path.exists(INDEX_OUT) and not DRY_RUN:
        fail(f"{rel(INDEX_OUT)} exists; move it out before re-running")
    R31 = load_r31()
    printed = check_printed()
    print(f"printed strings: {sum(len(v) for v in printed.values())} found at {PAPER_REF[:8]}")

    sets = sib_runs()
    print(f"Part 1: SIB runs results/31 {len(sets['31'])}, /32 {len(sets['32'])}, /34 {len(sets['34'])}")

    idx_info, idx = None, None
    built = R31.build_index("st_r450_published_png", PNG, write_to=INDEX_OUT)
    idx, idx_info = built if isinstance(built, tuple) else (built, None)
    want = [f"{a}-{m}-loco-r450-{c}-hr-s0" for a in ARCHS for m in METHODS for c in CITIES] + \
           [f"{a}-none-upper-r450-{c}-hr-s0" for a in ARCHS for c in CITIES]
    sel = idx[idx.run_id.isin(want)]
    if len(sel) != 72 or sel.run_id.duplicated().any():
        fail(f"index selection: {len(sel)} of 72 cells")
    print(f"PNG index: {len(idx)} rows; selected {len(sel)} (63 LOCO + 9 upper); n_images {sorted(sel.n_images.unique())}")
    if DRY_RUN:
        df, meta, err = R31.read_cell(PNG, want[0])
        print(f"schema sample {want[0]}: {len(df)} rows, columns {len(df.columns)}; aurc_fg null {int(df.aurc_fg.isna().sum())}")
        seo = seo_dirs()
        for rid in sorted(set(sets["31"]) | set(sets["34"])):
            print(f"  {rid:52s} -> {rel(resolve_dir(rid, seo))}")
        print("DRY RUN: nothing written")
        return
    png_cells = {}
    for rid in want:
        df, meta, err = R31.read_cell(PNG, rid)
        if err:
            fail(f"{rid}: {err}")
        if not os.path.isfile(os.path.join(PNG, rid, ".DONE")):
            fail(f"{rid}: no .DONE")
        png_cells[rid] = df

    recs, verdicts = part1(sets)
    s1 = json.load(open(os.path.join(FIG3, "test_s1_v2_results.json")))
    items = [item_2a(png_cells), item_2b(s1), item_2c(), item_2d(png_cells), item_2e(s1, png_cells), item_2f(), item_2g()]

    p1_lines = {g: {a: v["one_line"] for a, v in by.items()} for g, by in verdicts.items()}
    item1 = dict(
        item="1", where="App J tab:sib-hparams and equation; Fig. 4(a); section 5.1-5.2",
        source=dict(kind="run.json of every SIB run in results/31, /32, /34; C4clean companion run.json from results/42's tree "
                         "where the published-path dump records no config; three secondary records named per field"),
        printed="App J: beta_content(x) = beta_base + beta_scale * sigma(MLP(I_local)); tab:sib-hparams beta_base 0.01, "
                "beta_scale 0.02, beta_edge 1e-4, FDA 0.005, AdamW wd 0.01, LR 1e-4 CNN / 1e-5,1e-4 DINOv3, cosine warmup 5, "
                "batch 8, 100 epochs; Fig. 4(a) 'Content Aug [train only]'",
        value=dict(per_run=recs, verdicts=verdicts, one_line_per_architecture=p1_lines,
                   c4clean_summary={r["run_id"]: r["is_c4clean"] for r in recs}),
        matches_printed=False,
        caveats=["Only DINOv3's three internal SIB cells are the C4clean runs results/42 read (same legacy_run_dir). MAMNet's "
                 "nine are E-12 retrains of the C4clean config; OGLANet's three are n450 retrains with the held-out-city FDA pool; "
                 "CrossEarth and all S-EO SIB runs are separate training runs.",
                 "MAMNet and DINOv3 beta values are not in any run.json on the analysis host. MAMNet's come from the E-12 session log's "
                 "verbatim list of the published config; DINOv3's internal betas are unrecorded here.",
                 "The CNN SIB code's beta_content is compared with App J's beta_base; beta_max_multiplier (OGLANet) is not mapped "
                 "to beta_scale because its semantics are not recorded on the analysis host.",
                 "aug_p_mix = 0.3 is recorded on OGLANet and DINOv3 runs with use_content_aug = false. Whether the trainer applies "
                 "mixing when content augmentation is off is not determinable here: the SIB trainer code is not on the analysis host.",
                 "CrossEarth's internal SIB records beta_base/beta_scale but no adaptive flag; the arm F session log describes "
                 "I_local-conditioned beta. Its S-EO runs list adaptive_beta and content_aug as off.",
                 "OGLANet's optimizer (Adamax, ReduceLROnPlateau) comes from the n450 diagnosis log's reading of "
                 "train_oglanet_sib.py, not from run.json."])

    entries = [item1] + items
    runs = sorted(set(want) | {r["run_id"] for r in recs})
    n_mismatch_items = [e["item"] for e in entries
                        if e["matches_printed"] is False
                        or (isinstance(e["matches_printed"], dict) and not all(
                            v for k, v in e["matches_printed"].items() if isinstance(v, bool)))]
    out = dict(
        id=f"{NN}_{NAME}", arm="H2",
        answers=["AC1"],
        answers_detail="H2 number-consistency audit (REVIEWS.md row H2 answers AC1). Part 1 bears on the SIB description "
                       "(App J, Fig. 4(a)) behind section 5; no REVIEWS.md weakness ID names it, so only AC1 is listed.",
        question="For SIB's reported runs and for seven printed values with no results JSON: what is the source, what does it say, "
                 "and does it match the print?",
        paper=dict(ref=PAPER_REF, printed_strings_found=printed),
        entries=entries,
        items_not_matching_printed=n_mismatch_items,
        runs=runs,
        index_sha256=sha256_file(INDEX_OUT),
        index=dict(path=rel(INDEX_OUT), label="st_r450_published_png", n_rows=int(len(idx)), n_selected=72,
                   info=idx_info, seo_index=dict(path=rel(SEO_INDEX), sha256=sha256_file(SEO_INDEX))),
        source_artifacts={rel(p): sha256_file(p) for p in [
            os.path.join(FIG3, f) for f in ("robust_recovery_results.json", "robust_recovery_full.json", "paper_statistics.json",
                                            "experiment_results.json", "test_s1_v2_results.json", "rows.npy", "cell_rcs.pkl")]
            + [FIG3_NB, FER, E12_LOG, OGL_DIAG_LOG, CE_LOG, DEC_F]},
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        script=rel(os.path.abspath(__file__)), script_sha256=sha256_file(os.path.abspath(__file__)),
        env=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 scipy=__import__("scipy").__version__, conda_env=os.environ.get("CONDA_DEFAULT_ENV")),
        seed=BOOT_SEED, B=B,
        git_hash=None, git_hash_reason="project root is not a git repository (CLAUDE.md)",
        headline=("Of the 18 internal SIB runs in results/31-32 only DINOv3's 3 are results/42's C4clean runs, all 36 "
                  "MAMNet/OGLANet/DINOv3 SIB runs in results/31-34 record adaptive_beta=False and use_content_aug=False, "
                  "and of items 2a-2g, 2d reproduces, 2a reproduces except two CI bounds, 2b, 2c and 2e each hold a value "
                  "that does not reproduce, and 2f and 2g have no data source on the analysis host."),
        caveats=[c for e in entries for c in (f"[{e['item']}] {x}" for x in e["caveats"])],
        figure=None, caption=None)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    tmp = RESULTS + ".new"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, RESULTS)
    print(f"wrote {rel(RESULTS)}")
    for g, by in p1_lines.items():
        print(g)
        for a, ln in by.items():
            print("   ", ln)
    for e in items:
        print(e["item"], "matches_printed:", json.dumps(e["matches_printed"], default=str)[:300])


if __name__ == "__main__":
    main()
