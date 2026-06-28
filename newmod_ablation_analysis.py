#!/usr/bin/env python3
"""
NewMod Diagnostic Module Ablation Analysis — Publication-Ready Statistical Report

Analyzes the N0-N12 ablation matrix for the diagnostic-motivated case-study
modules (CACR, CE-AURC, TENT) layered on top of the SIB-Full (C4) base config.

Experiment matrix:
  N0   C4 baseline (control)
  N1   C4 + CACR (w=0.1)                       [class-asym confidence reg.]
  N2   C4 + CE-AURC (w=0.01)                   [calibration-aware aux loss]
  N3   C4 + CACR + CE-AURC                     [training-time stack]
  N4   C4 + TENT (steps=1, lr=0.001)           [test-time entropy min]
  N5   C4 + CACR + CE-AURC + TENT              [full stack]
  N6   CACR w=0.05    (sensitivity, gentler)
  N7   CACR w=0.5     (sensitivity, stronger)
  N8   CACR w=0.1, neg_weight=0.1              (background penalty)
  N9   CE-AURC w=0.05 (sensitivity, stronger)
  N10  CE-AURC w=0.001 (sensitivity, gentler)
  N11  TENT steps=3   (sensitivity)
  N12  TENT steps=5   (sensitivity, aggressive)

Architectures:
  MAMNet, OGLANet, DINOv3

LOCO folds:
  fold 0 → holdout_phoenix
  fold 1 → holdout_miami
  fold 2 → holdout_chicago

Statistical methodology:
  - Paired bootstrap (B=10,000) for CIs and two-sided p-values
  - Holm–Bonferroni correction across the family of (arch × ablation) tests
  - Cohen's d effect size for each ablation vs N0
  - Composition decomposition: N3 = N1 ⊕ N2, N5 = N3 ⊕ N4

Produces:
  1. Coverage matrix
  2. Main results table (Table 4 candidate): per-cell mIoU for N0-N12
  3. Per-architecture deltas vs N0 with bootstrap CIs and HB-corrected sig
  4. Recovery ratios R = (method − Vanilla) / (Upper − Vanilla)
  5. Composition analysis (additivity, interaction, redundancy)
  6. Hyperparameter sensitivity curves (CACR, CE-AURC, TENT)
  7. Worst-case safety (does any new module create catastrophic failures?)
  8. Per-architecture verdict table → final method recommendation
  9. LaTeX-ready tables and machine-readable JSON dump

Usage:
    python newmod_ablation_analysis.py \
        --mamnet_root  /path/to/mamnet/outputs \
        --oglanet_root /path/to/oglanet/outputs \
        --dinov3_root  /path/to/dinov3/outputs \
        --output_dir   /path/to/analysis_output \
        [--boundary_tolerance 2] \
        [--n_bootstrap 10000] \
        [--alpha 0.05] \
        [--eval_type tolerant]
"""

import os
import sys
import json
import re
import argparse
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════

ARCHITECTURES = ['MAMNet', 'OGLANet', 'DINOv3']
CITIES = ['chicago', 'miami', 'phoenix']
CITY_ABBREV = {'chicago': 'CHI', 'miami': 'MIA', 'phoenix': 'PHX'}
FOLD_MAP = {0: 'phoenix', 1: 'miami', 2: 'chicago'}

# Canonical experiment metadata.
# 'category' partitions experiments for downstream tables.
# 'modules' lists which diagnostic modules are active.
# 'cacr_w', 'ceaurc_w', 'tent_steps' record the (only) varied hyperparameter
# for sensitivity sweeps; None for experiments that don't activate that module.
EXPERIMENT_META = OrderedDict([
    ('N0',  {'name': 'C4 baseline (control)',
             'category': 'control', 'modules': [],
             'cacr_w': None, 'ceaurc_w': None, 'tent_steps': None}),
    ('N1',  {'name': 'C4 + CACR (w=0.1)',
             'category': 'core',    'modules': ['CACR'],
             'cacr_w': 0.10, 'ceaurc_w': None, 'tent_steps': None}),
    ('N2',  {'name': 'C4 + CE-AURC (w=0.01)',
             'category': 'core',    'modules': ['CE-AURC'],
             'cacr_w': None, 'ceaurc_w': 0.01, 'tent_steps': None}),
    ('N3',  {'name': 'C4 + CACR + CE-AURC',
             'category': 'compose', 'modules': ['CACR', 'CE-AURC'],
             'cacr_w': 0.10, 'ceaurc_w': 0.01, 'tent_steps': None}),
    ('N4',  {'name': 'C4 + TENT (s=1)',
             'category': 'core',    'modules': ['TENT'],
             'cacr_w': None, 'ceaurc_w': None, 'tent_steps': 1}),
    ('N5',  {'name': 'C4 + CACR + CE-AURC + TENT',
             'category': 'full',    'modules': ['CACR', 'CE-AURC', 'TENT'],
             'cacr_w': 0.10, 'ceaurc_w': 0.01, 'tent_steps': 1}),
    ('N6',  {'name': 'CACR w=0.05',
             'category': 'sens-cacr',   'modules': ['CACR'],
             'cacr_w': 0.05, 'ceaurc_w': None, 'tent_steps': None}),
    ('N7',  {'name': 'CACR w=0.5',
             'category': 'sens-cacr',   'modules': ['CACR'],
             'cacr_w': 0.50, 'ceaurc_w': None, 'tent_steps': None}),
    ('N8',  {'name': 'CACR w=0.1, neg_w=0.1',
             'category': 'sens-cacr',   'modules': ['CACR+neg'],
             'cacr_w': 0.10, 'ceaurc_w': None, 'tent_steps': None}),
    ('N9',  {'name': 'CE-AURC w=0.05',
             'category': 'sens-ceaurc', 'modules': ['CE-AURC'],
             'cacr_w': None, 'ceaurc_w': 0.05, 'tent_steps': None}),
    ('N10', {'name': 'CE-AURC w=0.001',
             'category': 'sens-ceaurc', 'modules': ['CE-AURC'],
             'cacr_w': None, 'ceaurc_w': 0.001, 'tent_steps': None}),
    ('N11', {'name': 'TENT steps=3',
             'category': 'sens-tent',   'modules': ['TENT'],
             'cacr_w': None, 'ceaurc_w': None, 'tent_steps': 3}),
    ('N12', {'name': 'TENT steps=5',
             'category': 'sens-tent',   'modules': ['TENT'],
             'cacr_w': None, 'ceaurc_w': None, 'tent_steps': 5}),
])

BASELINE_LABELS = [
    'Upper Bound', 'LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
    'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA',
]

# ═════════════════════════════════════════════════════════════════════════════
# Directory discovery
# ═════════════════════════════════════════════════════════════════════════════

# Matches `_N0_`, `_N12_` etc. — the unique key embedded in every dir name
# regardless of architecture-specific suffix conventions.
EXPERIMENT_RE = re.compile(r'_N(\d+)_(?=[a-z])', re.IGNORECASE)


def identify_experiment(dirname: str) -> Optional[str]:
    """Match a directory name to an experiment ID (N0..N12)."""
    m = EXPERIMENT_RE.search(dirname)
    if m is None:
        return None
    n = int(m.group(1))
    if n < 0 or n > 12:
        return None
    return f'N{n}'


def identify_fold(dirname: str) -> Optional[int]:
    """Extract fold/holdout city from directory name."""
    dn = dirname.lower()
    if 'holdout_phoenix' in dn or 'fold_0' in dn or 'fold0' in dn:
        return 0
    elif 'holdout_miami' in dn or 'fold_1' in dn or 'fold1' in dn:
        return 1
    elif 'holdout_chicago' in dn or 'fold_2' in dn or 'fold2' in dn:
        return 2
    for fold_id, city in FOLD_MAP.items():
        if city in dn:
            return fold_id
    return None


def identify_architecture(dirname: str) -> Optional[str]:
    """Extract architecture from directory prefix."""
    dn = dirname.lower()
    if dn.startswith('mamnet'):
        return 'MAMNet'
    if dn.startswith('oglanet'):
        return 'OGLANet'
    if dn.startswith('dinov3'):
        return 'DINOv3'
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Result loading
# ═════════════════════════════════════════════════════════════════════════════

def load_experiment(exp_dir: str, boundary_tolerance: int = 2) -> Optional[Dict]:
    """
    Load results from a single experiment directory.
    Returns dict or None if no usable results were found.
    """
    tol_key = f'tolerant_{boundary_tolerance}px'
    test_path = os.path.join(exp_dir, 'test_results.json')
    comp_path = os.path.join(exp_dir, 'comparison_results.json')

    result: Dict[str, Any] = {}

    if os.path.isfile(test_path):
        with open(test_path) as f:
            test_data = json.load(f)
        result['strict']    = test_data.get('strict', {}) or {}
        result['tolerant']  = test_data.get(tol_key, {}) or {}
        result['num_images'] = test_data.get('num_images', 0)
        # TENT-specific surface (if reported in test_results.json)
        result['tent_active'] = test_data.get('tent_active', False)
    elif os.path.isfile(comp_path):
        with open(comp_path) as f:
            comp_data = json.load(f)
        sib_data = comp_data.get('sib', {})
        result['strict']   = sib_data.get('strict', {}) or {}
        result['tolerant'] = sib_data.get(tol_key, {}) or {}
    else:
        return None

    # Comparison results (carries the baselines)
    if os.path.isfile(comp_path):
        with open(comp_path) as f:
            comp_data = json.load(f)
        result['baselines'] = comp_data.get('baselines', {}) or {}
        if not result.get('strict'):
            sib_data = comp_data.get('sib', {})
            result['strict']   = sib_data.get('strict', {}) or {}
            result['tolerant'] = sib_data.get(tol_key, {}) or {}
    else:
        result['baselines'] = {}

    # Config (for sanity checking which flags were on)
    cfg_path = os.path.join(exp_dir, 'config.json')
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path) as f:
                result['config'] = json.load(f)
        except json.JSONDecodeError:
            result['config'] = {}
    else:
        result['config'] = {}

    # Optional: training-time diagnostic traces (CACR, CE-AURC)
    # Layout follows submit-script comments: cacr_diag.cacr_pos_shift,
    # ce_aurc_diag.ce_aurc_mean_shadow_conf
    th_path = os.path.join(exp_dir, 'training_history.json')
    if os.path.isfile(th_path):
        try:
            with open(th_path) as f:
                th = json.load(f)
            # training_history may be a list-of-epochs or a dict
            if isinstance(th, list) and th:
                last = th[-1]
                result['cacr_diag']   = last.get('cacr_diag', {}) or {}
                result['ceaurc_diag'] = last.get('ce_aurc_diag', {}) or {}
            elif isinstance(th, dict):
                result['cacr_diag']   = th.get('cacr_diag', {}) or {}
                result['ceaurc_diag'] = th.get('ce_aurc_diag', {}) or {}
        except (json.JSONDecodeError, KeyError):
            pass

    # Optional: SP-gap / calibration metrics if computed offline
    sp_path = os.path.join(exp_dir, 'sp_gap_metrics.json')
    if os.path.isfile(sp_path):
        try:
            with open(sp_path) as f:
                result['sp_metrics'] = json.load(f)
        except json.JSONDecodeError:
            pass

    return result


def scan_experiments(root_dir: str, arch: str,
                     boundary_tolerance: int = 2) -> Dict:
    """Scan an architecture's output dir, organize by (exp_id, fold_id)."""
    results: Dict[str, Dict[int, Dict]] = defaultdict(dict)

    if not os.path.isdir(root_dir):
        print(f'  WARNING: {arch} root not found: {root_dir}')
        return results

    for entry in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, entry)
        if not os.path.isdir(full_path):
            continue

        # Architecture sanity check (helps if all archs share one root)
        dir_arch = identify_architecture(entry)
        if dir_arch is not None and dir_arch != arch:
            continue

        exp_id  = identify_experiment(entry)
        fold_id = identify_fold(entry)
        if exp_id is None or fold_id is None:
            continue

        exp_data = load_experiment(full_path, boundary_tolerance)
        if exp_data is None:
            print(f'  WARNING: No results in {entry}')
            continue

        # Keep the highest-mIoU run if duplicates exist (re-runs)
        existing = results[exp_id].get(fold_id)
        if existing:
            new_miou = exp_data.get('tolerant', {}).get('mIOU', 0) or 0
            old_miou = existing.get('tolerant', {}).get('mIOU', 0) or 0
            if new_miou <= old_miou:
                continue

        results[exp_id][fold_id] = exp_data
        city   = FOLD_MAP[fold_id]
        miou_s = exp_data.get('strict', {}).get('mIOU', 0) or 0
        miou_t = exp_data.get('tolerant', {}).get('mIOU', 0) or 0
        print(f'  {arch:8s} {exp_id:4s} fold={fold_id} ({city:8s}) '
              f'strict={miou_s:6.2f}  tolerant={miou_t:6.2f}  [{entry}]')

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Statistical primitives
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_paired(vals_a: np.ndarray, vals_b: np.ndarray,
                     n_bootstrap: int = 10000, seed: int = 42) -> Dict:
    """Paired bootstrap for mean(A) − mean(B). Returns CI + two-sided p."""
    rng = np.random.RandomState(seed)
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return {'delta': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                'p_value': np.nan, 'n': 0}

    diff = vals_a[:n] - vals_b[:n]
    obs  = float(np.mean(diff))

    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot[i] = np.mean(diff[idx])

    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    if obs >= 0:
        p_val = 2 * max(np.mean(boot <= 0), 1.0 / n_bootstrap)
    else:
        p_val = 2 * max(np.mean(boot >= 0), 1.0 / n_bootstrap)
    p_val = float(min(p_val, 1.0))

    return {'delta': obs, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'p_value': p_val, 'n': int(n)}


def cohens_d(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    """Paired Cohen's d. Returns NaN if undefined."""
    diff = vals_a - vals_b
    if len(diff) < 2:
        return float('nan')
    sd = float(np.std(diff, ddof=1))
    if sd < 1e-10:
        return float('nan')
    return float(np.mean(diff) / sd)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm–Bonferroni correction. Returns list of reject-null booleans."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            reject[orig_idx] = True
        else:
            break
    return reject


def sig_stars(p: float) -> str:
    if np.isnan(p):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


# ═════════════════════════════════════════════════════════════════════════════
# Master table & deltas vs N0
# ═════════════════════════════════════════════════════════════════════════════

def build_master_table(all_results: Dict, metric_key: str = 'mIOU',
                       eval_type: str = 'tolerant') -> Dict:
    """table[arch][exp_id] = {city: val, ..., 'mean': val, 'values': arr}"""
    table: Dict[str, Dict[str, Dict]] = {}

    for arch in ARCHITECTURES:
        table[arch] = {}
        arch_results = all_results.get(arch, {})

        for exp_id in EXPERIMENT_META:
            exp_results = arch_results.get(exp_id, {})
            cell_values: Dict[str, float] = {}

            for fold_id, city in FOLD_MAP.items():
                exp = exp_results.get(fold_id)
                if exp is None:
                    cell_values[city] = float('nan')
                else:
                    metrics = exp.get(eval_type, {}) or {}
                    val = metrics.get(metric_key, float('nan'))
                    cell_values[city] = float(val) if val is not None else float('nan')

            vals  = np.array([cell_values[c] for c in CITIES], dtype=np.float64)
            valid = vals[~np.isnan(vals)]
            mean_val = float(np.mean(valid)) if valid.size else float('nan')

            table[arch][exp_id] = {
                **cell_values,
                'mean': mean_val,
                'values': vals,
                'n_valid': int(valid.size),
            }

    return table


def extract_baselines(all_results: Dict, eval_type: str = 'tolerant',
                      metric_key: str = 'mIOU',
                      boundary_tolerance: int = 2) -> Dict:
    """baselines[arch][label][city] = val. Pulled from N0's comparison_results."""
    tol_key = f'tolerant_{boundary_tolerance}px'
    baselines: Dict[str, Dict[str, Dict[str, float]]] = {}

    for arch in ARCHITECTURES:
        baselines[arch] = defaultdict(dict)
        arch_results = all_results.get(arch, {})
        # Prefer N0 (control), but fall back to any N-config that has baselines
        candidate_ids = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5']
        for cand in candidate_ids:
            cand_results = arch_results.get(cand, {})
            if not cand_results:
                continue
            if any(exp.get('baselines') for exp in cand_results.values()):
                use_id = cand
                break
        else:
            continue

        ref_results = arch_results.get(use_id, {})
        for fold_id, city in FOLD_MAP.items():
            exp = ref_results.get(fold_id)
            if exp is None:
                continue
            bl_data = exp.get('baselines', {})
            for bl_label in BASELINE_LABELS:
                if bl_label not in bl_data:
                    continue
                bl_metrics = bl_data[bl_label]
                src = (bl_metrics.get(eval_type)
                       or bl_metrics.get(tol_key)
                       or bl_metrics.get('strict')
                       or {})
                val = src.get(metric_key, float('nan')) if src else float('nan')
                baselines[arch][bl_label][city] = float(val) if val is not None else float('nan')

    return baselines


def compute_deltas_vs_n0(table: Dict, n_bootstrap: int = 10000,
                         alpha: float = 0.05) -> Dict:
    """For each (arch, exp), Δ = exp − N0 with paired bootstrap + Holm correction."""
    deltas: Dict[str, Dict[str, Optional[Dict]]] = {arch: {} for arch in ARCHITECTURES}
    p_list: List[float] = []
    key_list: List[Tuple[str, str]] = []

    for arch in ARCHITECTURES:
        n0_entry = table[arch].get('N0')
        if n0_entry is None or n0_entry['n_valid'] == 0:
            for exp_id in EXPERIMENT_META:
                if exp_id != 'N0':
                    deltas[arch][exp_id] = None
            continue

        n0_vals = n0_entry['values']

        for exp_id in EXPERIMENT_META:
            if exp_id == 'N0':
                continue
            exp_entry = table[arch].get(exp_id)
            if exp_entry is None or exp_entry['n_valid'] == 0:
                deltas[arch][exp_id] = None
                continue

            exp_vals = exp_entry['values']
            mask = ~np.isnan(n0_vals) & ~np.isnan(exp_vals)
            n0_v  = n0_vals[mask]
            exp_v = exp_vals[mask]
            if n0_v.size == 0:
                deltas[arch][exp_id] = None
                continue

            delta_mean = float(np.mean(exp_v) - np.mean(n0_v))
            boot = bootstrap_paired(exp_v, n0_v, n_bootstrap=n_bootstrap)
            d = cohens_d(exp_v, n0_v)

            deltas[arch][exp_id] = {
                'delta_mean': delta_mean,
                'bootstrap':  boot,
                'cohens_d':   d,
                'n_folds':    int(n0_v.size),
            }
            p_list.append(boot['p_value'])
            key_list.append((arch, exp_id))

    # Family-wise correction across (arch × exp) tests
    if p_list:
        corrected = holm_bonferroni(p_list, alpha=alpha)
        for i, (arch, exp_id) in enumerate(key_list):
            d = deltas[arch][exp_id]
            if d is not None:
                d['significant_raw']       = bool(p_list[i] < alpha)
                d['significant_corrected'] = bool(corrected[i])

    return deltas


# ═════════════════════════════════════════════════════════════════════════════
# Recovery ratios
# ═════════════════════════════════════════════════════════════════════════════

def compute_recovery_ratios(table: Dict, baselines: Dict) -> Dict:
    """R = (method − Vanilla) / (Upper − Vanilla)  per (arch, exp, city)."""
    recovery: Dict[str, Dict[str, Dict]] = {}

    for arch in ARCHITECTURES:
        recovery[arch] = {}
        ub = baselines.get(arch, {}).get('Upper Bound', {})
        lv = baselines.get(arch, {}).get('LOCO Vanilla', {})

        for exp_id in EXPERIMENT_META:
            entry = table[arch].get(exp_id)
            if entry is None:
                continue

            R_vals: List[float] = []
            R_per_city: Dict[str, float] = {}
            for city in CITIES:
                ub_v  = ub.get(city, float('nan'))
                lv_v  = lv.get(city, float('nan'))
                exp_v = entry.get(city, float('nan'))
                if (np.isnan(ub_v) or np.isnan(lv_v) or np.isnan(exp_v)):
                    R_per_city[city] = float('nan')
                    continue
                gap = ub_v - lv_v
                if abs(gap) < 0.01:
                    R_per_city[city] = float('nan')
                    continue
                R = (exp_v - lv_v) / gap
                R_per_city[city] = float(R)
                R_vals.append(R)

            recovery[arch][exp_id] = {
                **R_per_city,
                'mean': float(np.mean(R_vals)) if R_vals else float('nan'),
                'min':  float(np.min(R_vals))  if R_vals else float('nan'),
                'max':  float(np.max(R_vals))  if R_vals else float('nan'),
            }

    return recovery


# ═════════════════════════════════════════════════════════════════════════════
# Composition tests (do modules stack? do they interfere?)
# ═════════════════════════════════════════════════════════════════════════════

def compute_composition_tests(table: Dict, deltas: Dict) -> Dict:
    """
    Two formal additivity decompositions:
      C1.  N3 ≟ N1 ⊕ N2     (do CACR and CE-AURC stack additively?)
      C2.  N5 ≟ N3 ⊕ N4     (does TENT add value on top of training-time stack?)

    For each, we report:
      - observed_combined  = mean(N3) − mean(N0)              (or N5 − N0)
      - sum_individual     = (mean(N1)−mean(N0)) + (mean(N2)−mean(N0))
      - interaction        = observed_combined − sum_individual
      - direction: 'super-additive' if interaction > 0,
                   'sub-additive'   if interaction < 0,
                   'redundant'      if observed_combined ≈ max(individual).
    """
    composition: Dict[str, Dict[str, Dict]] = {}

    def _delta(arch: str, exp: str) -> float:
        d = deltas.get(arch, {}).get(exp)
        if d is None:
            return float('nan')
        return d['delta_mean']

    for arch in ARCHITECTURES:
        composition[arch] = {}

        # ── C1: N3 vs N1 ⊕ N2 ────────────────────────────────────────
        d1, d2, d3 = _delta(arch, 'N1'), _delta(arch, 'N2'), _delta(arch, 'N3')
        if not any(np.isnan([d1, d2, d3])):
            sum_indiv   = d1 + d2
            interaction = d3 - sum_indiv
            best_indiv  = max(d1, d2)
            if abs(interaction) < 0.5:
                verdict = 'additive'
            elif interaction > 0:
                verdict = 'super-additive (synergy)'
            elif d3 < best_indiv - 0.5:
                verdict = 'interfering (worse than best individual)'
            else:
                verdict = 'sub-additive (partial overlap)'
            composition[arch]['C1_N3_vs_N1+N2'] = {
                'd_N1': d1, 'd_N2': d2, 'd_N3': d3,
                'sum_individual': sum_indiv,
                'interaction':    interaction,
                'best_individual': best_indiv,
                'verdict': verdict,
            }

        # ── C2: N5 vs N3 ⊕ N4 ────────────────────────────────────────
        d3, d4, d5 = _delta(arch, 'N3'), _delta(arch, 'N4'), _delta(arch, 'N5')
        if not any(np.isnan([d3, d4, d5])):
            sum_indiv   = d3 + d4
            interaction = d5 - sum_indiv
            best_indiv  = max(d3, d4)
            if abs(interaction) < 0.5:
                verdict = 'additive'
            elif interaction > 0:
                verdict = 'super-additive (synergy)'
            elif d5 < best_indiv - 0.5:
                verdict = 'TENT interfering with training-time modules'
            else:
                verdict = 'sub-additive (TENT partially redundant)'
            composition[arch]['C2_N5_vs_N3+N4'] = {
                'd_N3': d3, 'd_N4': d4, 'd_N5': d5,
                'sum_individual': sum_indiv,
                'interaction':    interaction,
                'best_individual': best_indiv,
                'verdict': verdict,
            }

    return composition


# ═════════════════════════════════════════════════════════════════════════════
# Sensitivity curves
# ═════════════════════════════════════════════════════════════════════════════

def compute_sensitivity(table: Dict, deltas: Dict) -> Dict:
    """Builds three sensitivity curves: CACR weight, CE-AURC weight, TENT steps."""
    sens: Dict[str, Dict[str, List[Dict]]] = {arch: {} for arch in ARCHITECTURES}

    cacr_points = [
        ('N6', 0.05),  ('N1', 0.10),  ('N7', 0.50),
    ]
    ceaurc_points = [
        ('N10', 0.001), ('N2', 0.01),  ('N9', 0.05),
    ]
    tent_points = [
        ('N4', 1),  ('N11', 3),  ('N12', 5),
    ]

    for arch in ARCHITECTURES:
        sens[arch]['CACR_weight'] = []
        for exp, w in cacr_points:
            d = deltas.get(arch, {}).get(exp)
            if d is None:
                continue
            entry = table[arch].get(exp)
            sens[arch]['CACR_weight'].append({
                'experiment': exp, 'value': w,
                'mean_miou':  entry['mean'] if entry else float('nan'),
                'delta_vs_N0': d['delta_mean'],
                'p_value':     d['bootstrap']['p_value'],
            })

        sens[arch]['CE_AURC_weight'] = []
        for exp, w in ceaurc_points:
            d = deltas.get(arch, {}).get(exp)
            if d is None:
                continue
            entry = table[arch].get(exp)
            sens[arch]['CE_AURC_weight'].append({
                'experiment': exp, 'value': w,
                'mean_miou':  entry['mean'] if entry else float('nan'),
                'delta_vs_N0': d['delta_mean'],
                'p_value':     d['bootstrap']['p_value'],
            })

        sens[arch]['TENT_steps'] = []
        for exp, s in tent_points:
            d = deltas.get(arch, {}).get(exp)
            if d is None:
                continue
            entry = table[arch].get(exp)
            sens[arch]['TENT_steps'].append({
                'experiment': exp, 'value': s,
                'mean_miou':  entry['mean'] if entry else float('nan'),
                'delta_vs_N0': d['delta_mean'],
                'p_value':     d['bootstrap']['p_value'],
            })

    return sens


# ═════════════════════════════════════════════════════════════════════════════
# Worst-case safety (do new modules ever cause catastrophic failures?)
# ═════════════════════════════════════════════════════════════════════════════

def worst_case_safety(table: Dict, baselines: Dict) -> Dict:
    """For each (arch, exp), find the worst single-cell deg vs N0 and vs Vanilla."""
    safety: Dict[str, Dict[str, Dict]] = {arch: {} for arch in ARCHITECTURES}

    for arch in ARCHITECTURES:
        n0_entry = table[arch].get('N0', {})
        lv = baselines.get(arch, {}).get('LOCO Vanilla', {})

        for exp_id in EXPERIMENT_META:
            entry = table[arch].get(exp_id)
            if entry is None or n0_entry.get('n_valid', 0) == 0:
                continue

            d_vs_n0:      List[float] = []
            d_vs_vanilla: List[float] = []
            for city in CITIES:
                ev   = entry.get(city, float('nan'))
                n0v  = n0_entry.get(city, float('nan'))
                lvv  = lv.get(city, float('nan'))
                if not (np.isnan(ev) or np.isnan(n0v)):
                    d_vs_n0.append(ev - n0v)
                if not (np.isnan(ev) or np.isnan(lvv)):
                    d_vs_vanilla.append(ev - lvv)

            safety[arch][exp_id] = {
                'worst_vs_N0':      float(np.min(d_vs_n0))      if d_vs_n0      else float('nan'),
                'worst_vs_Vanilla': float(np.min(d_vs_vanilla)) if d_vs_vanilla else float('nan'),
                'n_catastrophic':   int(sum(1 for d in d_vs_vanilla if d < -10)),
                'n_regression':     int(sum(1 for d in d_vs_n0     if d < -1)),
            }

    return safety


# ═════════════════════════════════════════════════════════════════════════════
# Per-architecture verdicts (final method recommendation)
# ═════════════════════════════════════════════════════════════════════════════

def per_arch_verdict(deltas: Dict, safety: Dict) -> Dict:
    """
    Decision tree for each architecture:
      1. Find the best-performing config (largest positive Δ vs N0)
      2. Confirm it's HB-significant (or at minimum raw-significant)
      3. Confirm it doesn't introduce catastrophic failures vs N0
      4. Output a final method recommendation per architecture
    """
    verdicts: Dict[str, Dict] = {}

    for arch in ARCHITECTURES:
        best_id   = None
        best_d    = -float('inf')
        best_p    = float('nan')
        best_sig  = False
        for exp_id in EXPERIMENT_META:
            if exp_id == 'N0':
                continue
            d = deltas.get(arch, {}).get(exp_id)
            if d is None:
                continue
            if d['delta_mean'] > best_d:
                best_d   = d['delta_mean']
                best_id  = exp_id
                best_p   = d['bootstrap']['p_value']
                best_sig = bool(d.get('significant_corrected', False))

        worst_n0 = (safety.get(arch, {}).get(best_id, {}).get('worst_vs_N0', float('nan'))
                    if best_id else float('nan'))

        if best_id is None or best_d <= 0:
            recommendation = 'Keep N0 (no module additionally helps)'
        elif not best_sig and best_p > 0.05:
            recommendation = f'Inconclusive — best is {best_id} (Δ={best_d:+.2f}) but not significant'
        elif worst_n0 < -3:
            recommendation = (f'{best_id} on average, BUT worst-cell Δ={worst_n0:+.2f}'
                              f' vs N0 — flag in paper')
        else:
            mods = EXPERIMENT_META[best_id]['modules']
            mod_str = ' + '.join(mods) if mods else '(none)'
            recommendation = f'Use {best_id} = N0 + {mod_str} (Δ={best_d:+.2f}, p={best_p:.4f})'

        verdicts[arch] = {
            'best_experiment': best_id,
            'best_delta':      float(best_d) if best_d != -float('inf') else float('nan'),
            'best_p':          float(best_p),
            'best_hb_sig':     bool(best_sig),
            'worst_vs_N0':     float(worst_n0) if not np.isnan(worst_n0) else float('nan'),
            'recommendation':  recommendation,
        }

    return verdicts


# ═════════════════════════════════════════════════════════════════════════════
# Diagnostic-prediction validation
# ═════════════════════════════════════════════════════════════════════════════

def validate_predictions(deltas: Dict, table: Dict) -> List[str]:
    """Test the §5 predictions documented in the submit-script comments."""
    out: List[str] = []

    def _delta(arch, exp):
        d = deltas.get(arch, {}).get(exp)
        return d['delta_mean'] if d else float('nan')

    # P1: CACR (N1) helps recall-side / shadow-positive accuracy.
    # Coarse proxy: N1 should improve mean mIoU vs N0 on at least 2/3 archs.
    n1_helps = [_delta(a, 'N1') for a in ARCHITECTURES]
    n1_helps_arch = sum(1 for d in n1_helps if not np.isnan(d) and d > 0)
    out.append(
        f'  P1 (CACR helps mean mIoU on majority of archs): '
        f'{n1_helps_arch}/3 archs show Δ>0  '
        f'{"✓ PASS" if n1_helps_arch >= 2 else "✗ FAIL"}')

    # P2: CE-AURC alone (N2) doesn't strongly hurt mIoU
    # (it's a calibration loss, not a quality loss)
    n2_deltas = [_delta(a, 'N2') for a in ARCHITECTURES]
    n2_no_collapse = all(np.isnan(d) or d > -3.0 for d in n2_deltas)
    out.append(
        f'  P2 (CE-AURC does not collapse mIoU on any arch): '
        f'min Δ = {min(d for d in n2_deltas if not np.isnan(d)):+.2f}  '
        f'{"✓ PASS" if n2_no_collapse else "✗ FAIL"}')

    # P3: TENT (N4) helps DINOv3 most (clean encoder + correctable decoder)
    d_dino = _delta('DINOv3',  'N4')
    d_mam  = _delta('MAMNet',  'N4')
    d_ogla = _delta('OGLANet', 'N4')
    if not any(np.isnan([d_dino, d_mam, d_ogla])):
        passed = (d_dino >= max(d_mam, d_ogla))
        out.append(
            f'  P3 (TENT helps DINOv3 most): '
            f'DINOv3={d_dino:+.2f}, MAMNet={d_mam:+.2f}, OGLANet={d_ogla:+.2f}  '
            f'{"✓ PASS" if passed else "✗ FAIL"}')
    else:
        out.append(f'  P3 (TENT helps DINOv3 most): insufficient data')

    # P4: TENT (N4) does not catastrophically fail OGLANet (encoder-locus)
    d_ogla4 = _delta('OGLANet', 'N4')
    out.append(
        f'  P4 (TENT does not catastrophically fail OGLANet): '
        f'Δ={d_ogla4:+.2f}  '
        f'{"✓ PASS" if (np.isnan(d_ogla4) or d_ogla4 > -3.0) else "✗ FAIL"}')

    # P5: Full stack N5 ≥ best single (N1, N2, or N4) on majority of archs
    pass_count = 0
    for arch in ARCHITECTURES:
        d5 = _delta(arch, 'N5')
        d_singles = [_delta(arch, x) for x in ['N1', 'N2', 'N4']]
        d_singles = [d for d in d_singles if not np.isnan(d)]
        if not np.isnan(d5) and d_singles and d5 >= max(d_singles) - 0.5:
            pass_count += 1
    out.append(
        f'  P5 (N5 full stack ≥ best single on majority of archs): '
        f'{pass_count}/3  '
        f'{"✓ PASS" if pass_count >= 2 else "✗ FAIL"}')

    return out


# ═════════════════════════════════════════════════════════════════════════════
# Pretty-print tables
# ═════════════════════════════════════════════════════════════════════════════

def print_main_table(table: Dict, deltas: Dict, eval_type: str = 'tolerant'):
    print(f'\n{"="*120}')
    print(f'  TABLE: Per-Cell LOCO {eval_type.upper()} mIoU — '
          f'N0 (control) and Diagnostic-Module Variants (N1-N12)')
    print(f'{"="*120}')

    header = f'  {"ID":<4} {"Description":<30}'
    for arch in ARCHITECTURES:
        for city in CITIES:
            header += f' {CITY_ABBREV[city]:>5}'
        header += f' {"Avg":>6}'
    header += f'  {"ΔAvg":>7} {"p":>7} {"d":>6}'
    print(header)
    print('  ' + '-' * 116)

    for exp_id, meta in EXPERIMENT_META.items():
        row = f'  {exp_id:<4} {meta["name"]:<30}'

        for arch in ARCHITECTURES:
            entry = table[arch].get(exp_id)
            for city in CITIES:
                if entry and not np.isnan(entry.get(city, float('nan'))):
                    row += f' {entry[city]:5.1f}'
                else:
                    row += f' {"—":>5}'
            if entry and not np.isnan(entry.get('mean', float('nan'))):
                row += f' {entry["mean"]:6.2f}'
            else:
                row += f' {"—":>6}'

        if exp_id == 'N0':
            row += f'  {"—":>7} {"—":>7} {"—":>6}'
        else:
            arch_deltas, arch_ps, arch_ds = [], [], []
            for arch in ARCHITECTURES:
                d = deltas.get(arch, {}).get(exp_id)
                if d:
                    arch_deltas.append(d['delta_mean'])
                    arch_ps.append(d['bootstrap']['p_value'])
                    arch_ds.append(d['cohens_d'])
            if arch_deltas:
                avg_d = float(np.mean(arch_deltas))
                min_p = float(min(arch_ps))
                valid_ds = [d for d in arch_ds if not np.isnan(d)]
                avg_es = float(np.mean(valid_ds)) if valid_ds else float('nan')
                stars = sig_stars(min_p)
                row += f'  {avg_d:+5.2f}{stars:<2}'
                row += f' {min_p:7.4f}'
                row += f' {avg_es:6.2f}' if not np.isnan(avg_es) else f' {"—":>6}'
            else:
                row += f'  {"—":>7} {"—":>7} {"—":>6}'

        print(row)

        # Visual separators between categories
        if exp_id == 'N0' or exp_id == 'N5':
            print('  ' + '-' * 116)
        elif exp_id == 'N8' or exp_id == 'N10':
            print('  ' + '·' * 116)


def print_per_arch_deltas(deltas: Dict):
    print(f'\n{"="*100}')
    print('  PER-ARCHITECTURE DELTAS vs N0 (paired bootstrap, 95% CI, '
          'Holm–Bonferroni corrected)')
    print(f'{"="*100}')
    print(f'  {"ID":<4} {"Arch":<9} {"Δ mIoU":>8} '
          f'{"95% CI":>17} {"p-value":>9} {"d":>7} {"raw":>4} {"HB":>4}')
    print('  ' + '-' * 96)

    for exp_id in EXPERIMENT_META:
        if exp_id == 'N0':
            continue
        for arch in ARCHITECTURES:
            d = deltas.get(arch, {}).get(exp_id)
            if d is None:
                print(f'  {exp_id:<4} {arch:<9} {"N/A":>8}')
                continue
            boot = d['bootstrap']
            ci = f'[{boot["ci_lo"]:+.2f}, {boot["ci_hi"]:+.2f}]'
            cd = d['cohens_d']
            cd_s = f'{cd:+.2f}' if not np.isnan(cd) else '—'
            raw_s = '✓' if d.get('significant_raw', False) else ''
            hb_s  = '✓' if d.get('significant_corrected', False) else ''
            print(f'  {exp_id:<4} {arch:<9} {d["delta_mean"]:+8.2f} '
                  f'{ci:>17} {boot["p_value"]:9.4f} '
                  f'{cd_s:>7} {raw_s:>4} {hb_s:>4}')
        print()


def print_recovery_table(recovery: Dict):
    print(f'\n{"="*86}')
    print('  RECOVERY RATIOS:  R = (method − Vanilla) / (Upper − Vanilla)')
    print(f'{"="*86}')
    print(f'  {"ID":<4} {"Description":<30}', end='')
    for arch in ARCHITECTURES:
        print(f'{arch:>10}', end='')
    print(f'  {"Mean":>8}')
    print('  ' + '-' * 82)

    for exp_id, meta in EXPERIMENT_META.items():
        row = f'  {exp_id:<4} {meta["name"]:<30}'
        arch_means = []
        for arch in ARCHITECTURES:
            r = recovery.get(arch, {}).get(exp_id)
            if r and not np.isnan(r['mean']):
                row += f'{r["mean"]:10.3f}'
                arch_means.append(r['mean'])
            else:
                row += f'{"—":>10}'
        ov = float(np.mean(arch_means)) if arch_means else float('nan')
        row += f'  {ov:8.3f}' if not np.isnan(ov) else f'  {"—":>8}'
        print(row)


def print_composition_tests(composition: Dict):
    print(f'\n{"="*100}')
    print('  COMPOSITION TESTS:  Do diagnostic modules stack additively?')
    print(f'{"="*100}')

    for arch in ARCHITECTURES:
        comp = composition.get(arch, {})
        if not comp:
            continue
        print(f'\n  --- {arch} ---')

        c1 = comp.get('C1_N3_vs_N1+N2')
        if c1:
            print(f'    C1: N3 (CACR+CE-AURC) vs N1 ⊕ N2')
            print(f'        Δ(N1)        = {c1["d_N1"]:+6.2f}')
            print(f'        Δ(N2)        = {c1["d_N2"]:+6.2f}')
            print(f'        Δ(N3) actual = {c1["d_N3"]:+6.2f}')
            print(f'        Sum (additive) = {c1["sum_individual"]:+6.2f}')
            print(f'        Interaction  = {c1["interaction"]:+6.2f}')
            print(f'        Verdict: {c1["verdict"]}')

        c2 = comp.get('C2_N5_vs_N3+N4')
        if c2:
            print(f'\n    C2: N5 (full stack) vs N3 ⊕ N4')
            print(f'        Δ(N3)        = {c2["d_N3"]:+6.2f}')
            print(f'        Δ(N4)        = {c2["d_N4"]:+6.2f}')
            print(f'        Δ(N5) actual = {c2["d_N5"]:+6.2f}')
            print(f'        Sum (additive) = {c2["sum_individual"]:+6.2f}')
            print(f'        Interaction  = {c2["interaction"]:+6.2f}')
            print(f'        Verdict: {c2["verdict"]}')


def print_sensitivity(sens: Dict):
    print(f'\n{"="*86}')
    print('  HYPERPARAMETER SENSITIVITY CURVES (Δ vs N0, all hyperparams varied)')
    print(f'{"="*86}')

    for sweep in ['CACR_weight', 'CE_AURC_weight', 'TENT_steps']:
        print(f'\n  --- {sweep} ---')
        print(f'  {"Arch":<9} {"exp":<5} {"value":>8} '
              f'{"mIoU":>8} {"Δ vs N0":>10} {"p":>8} {"sig":>4}')
        for arch in ARCHITECTURES:
            curve = sens.get(arch, {}).get(sweep, [])
            for pt in curve:
                stars = sig_stars(pt['p_value'])
                print(f'  {arch:<9} {pt["experiment"]:<5} '
                      f'{pt["value"]:>8} '
                      f'{pt["mean_miou"]:8.2f} '
                      f'{pt["delta_vs_N0"]:+10.2f} '
                      f'{pt["p_value"]:8.4f} {stars:>4}')


def print_safety(safety: Dict):
    print(f'\n{"="*100}')
    print('  WORST-CASE SAFETY ANALYSIS: minimum single-cell Δ vs N0 and vs Vanilla')
    print(f'{"="*100}')
    print(f'  {"ID":<4} {"Arch":<9} {"Worst Δ vs N0":>14} '
          f'{"Worst Δ vs Vanilla":>20} {"# regress (>1↓)":>17} '
          f'{"# catastrophic (>10↓)":>22}')
    print('  ' + '-' * 96)
    for exp_id in EXPERIMENT_META:
        for arch in ARCHITECTURES:
            s = safety.get(arch, {}).get(exp_id)
            if s is None:
                continue
            print(f'  {exp_id:<4} {arch:<9} '
                  f'{s["worst_vs_N0"]:+14.2f} '
                  f'{s["worst_vs_Vanilla"]:+20.2f} '
                  f'{s["n_regression"]:>17} '
                  f'{s["n_catastrophic"]:>22}')
        if exp_id != list(EXPERIMENT_META)[-1]:
            print()


def print_verdicts(verdicts: Dict):
    print(f'\n{"="*100}')
    print('  PER-ARCHITECTURE VERDICT — final method recommendation')
    print(f'{"="*100}')
    for arch in ARCHITECTURES:
        v = verdicts.get(arch)
        if v is None:
            print(f'\n  --- {arch} ---  (insufficient data)')
            continue
        print(f'\n  --- {arch} ---')
        print(f'    Best config:   {v["best_experiment"]}  (Δ = {v["best_delta"]:+.2f})')
        print(f'    p-value:       {v["best_p"]:.4f}')
        print(f'    HB-significant: {"yes" if v["best_hb_sig"] else "no"}')
        print(f'    Worst-cell Δ vs N0: {v["worst_vs_N0"]:+.2f}')
        print(f'    →  {v["recommendation"]}')


# ═════════════════════════════════════════════════════════════════════════════
# LaTeX table
# ═════════════════════════════════════════════════════════════════════════════

def generate_latex_table(table: Dict, deltas: Dict, eval_type: str) -> str:
    lines: List[str] = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{')
    lines.append(r'    \textbf{Diagnostic-module ablation (' + eval_type + r' mIoU).}')
    lines.append(r'    N0 is the C4 SIB-Full baseline; N1--N5 add CACR, CE-AURC,')
    lines.append(r'    and TENT in the indicated combinations; N6--N12 sweep their')
    lines.append(r'    hyperparameters. $\Delta$ = mean change vs N0 across the 9')
    lines.append(r'    LOCO cells. Significance from paired bootstrap with')
    lines.append(r'    Holm--Bonferroni correction.')
    lines.append(r'  }')
    lines.append(r'  \label{tab:newmod_ablations}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{@{}llccccccccccc@{}}')
    lines.append(r'    \toprule')
    lines.append(r'    & & \multicolumn{3}{c}{MAMNet} & '
                 r'\multicolumn{3}{c}{OGLANet} & '
                 r'\multicolumn{3}{c}{DINOv3} & Avg & $\Delta$ \\')
    lines.append(r'    \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}')
    lines.append(r'    ID & Description & CHI & MIA & PHX & '
                 r'CHI & MIA & PHX & CHI & MIA & PHX & & \\')
    lines.append(r'    \midrule')

    for exp_id, meta in EXPERIMENT_META.items():
        parts = [f'    {exp_id}', meta['name']]
        all_vals: List[float] = []
        for arch in ARCHITECTURES:
            entry = table[arch].get(exp_id)
            for city in CITIES:
                v = entry.get(city, float('nan')) if entry else float('nan')
                if not np.isnan(v):
                    all_vals.append(v)
                    parts.append(f'{v:.1f}')
                else:
                    parts.append('--')
        avg = float(np.mean(all_vals)) if all_vals else float('nan')
        parts.append(f'{avg:.1f}' if not np.isnan(avg) else '--')

        if exp_id == 'N0':
            parts.append('--')
        else:
            ds = []
            ps = []
            for arch in ARCHITECTURES:
                d = deltas.get(arch, {}).get(exp_id)
                if d:
                    ds.append(d['delta_mean'])
                    ps.append(d['bootstrap']['p_value'])
            if ds:
                avg_d = float(np.mean(ds))
                min_p = float(min(ps))
                stars = sig_stars(min_p)
                if stars:
                    parts.append(f'{avg_d:+.2f}$^{{{stars}}}$')
                else:
                    parts.append(f'{avg_d:+.2f}')
            else:
                parts.append('--')

        lines.append(' & '.join(parts) + r' \\')
        if exp_id == 'N0' or exp_id == 'N5':
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table*}')
    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# JSON report
# ═════════════════════════════════════════════════════════════════════════════

def _clean(val):
    if isinstance(val, np.floating):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.ndarray):
        return [_clean(v) for v in val]
    if isinstance(val, dict):
        return {k: _clean(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_clean(v) for v in val]
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


def generate_json_report(table, baselines, deltas, recovery,
                         composition, sens, safety, verdicts,
                         eval_type: str) -> Dict:
    return {
        'generated': datetime.now().isoformat(),
        'eval_type': eval_type,
        'architectures': ARCHITECTURES,
        'cities': CITIES,
        'experiment_meta': {k: dict(v) for k, v in EXPERIMENT_META.items()},
        'main_table':       _clean(table),
        'baselines':        _clean(dict(baselines)),
        'deltas_vs_N0':     _clean(deltas),
        'recovery_ratios':  _clean(recovery),
        'composition':      _clean(composition),
        'sensitivity':      _clean(sens),
        'safety':           _clean(safety),
        'verdicts':         _clean(verdicts),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='NewMod (CACR + CE-AURC + TENT) Ablation Analysis')

    default_base = os.path.join(os.environ["PROJECT_ROOT"], 'data')
    p.add_argument('--mamnet_root', type=str,
                   default=os.path.join(default_base, 'mamnet/outputs'))
    p.add_argument('--oglanet_root', type=str,
                   default=os.path.join(default_base, 'oglanet/outputs'))
    p.add_argument('--dinov3_root', type=str,
                   default=os.path.join(default_base, 'dinov3/outputs'))
    p.add_argument('--output_dir', type=str, default='./newmod_analysis')
    p.add_argument('--boundary_tolerance', type=int, default=2)
    p.add_argument('--n_bootstrap', type=int, default=10000)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--eval_type', type=str, default='tolerant',
                   choices=['strict', 'tolerant'])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    arch_roots = {
        'MAMNet':  args.mamnet_root,
        'OGLANet': args.oglanet_root,
        'DINOv3':  args.dinov3_root,
    }

    print('=' * 80)
    print('  NEWMOD DIAGNOSTIC-MODULE ABLATION ANALYSIS')
    print('  CACR + CE-AURC + TENT  (N0-N12, 13 configs × 3 archs × 3 folds)')
    print(f'  Eval: {args.eval_type}  |  Boundary: ±{args.boundary_tolerance}px  |  '
          f'Bootstrap B={args.n_bootstrap}  |  α={args.alpha}')
    print('=' * 80)

    # ── 1. Scan ────────────────────────────────────────────────────────────
    print('\n[1/8] Scanning experiment directories...')
    all_results = {}
    for arch in ARCHITECTURES:
        print(f'\n  === {arch} === ({arch_roots[arch]})')
        all_results[arch] = scan_experiments(
            arch_roots[arch], arch, args.boundary_tolerance)

    # Coverage matrix
    print(f'\n  Coverage matrix (experiment × architecture):')
    print(f'  {"":4s}', end='')
    for arch in ARCHITECTURES:
        print(f' {arch:>9}', end='')
    print()
    for exp_id in EXPERIMENT_META:
        print(f'  {exp_id:4s}', end='')
        for arch in ARCHITECTURES:
            n = len(all_results.get(arch, {}).get(exp_id, {}))
            marker = f'{n}/3' if n > 0 else '—'
            print(f' {marker:>9}', end='')
        print()

    # ── 2. Master table ────────────────────────────────────────────────────
    print(f'\n[2/8] Building master results table ({args.eval_type} mIoU)...')
    table = build_master_table(all_results, metric_key='mIOU',
                               eval_type=args.eval_type)

    # ── 3. Baselines ───────────────────────────────────────────────────────
    print('\n[3/8] Extracting baselines from N0 comparison_results.json...')
    baselines = extract_baselines(
        all_results, eval_type=args.eval_type,
        boundary_tolerance=args.boundary_tolerance)
    for arch in ARCHITECTURES:
        for label, cities in baselines.get(arch, {}).items():
            vals = [cities.get(c, float('nan')) for c in CITIES]
            valid = [v for v in vals if not np.isnan(v)]
            if valid:
                print(f'  {arch:8s} {label:<16} mean={np.mean(valid):.2f}  '
                      + ", ".join(f"{CITY_ABBREV[c]}={cities.get(c, float('nan')):.1f}"
                                  for c in CITIES if not np.isnan(cities.get(c, float('nan')))))

    # ── 4. Deltas vs N0 ────────────────────────────────────────────────────
    print(f'\n[4/8] Computing per-architecture deltas vs N0 '
          f'(bootstrap B={args.n_bootstrap})...')
    deltas = compute_deltas_vs_n0(table, n_bootstrap=args.n_bootstrap,
                                  alpha=args.alpha)

    # ── 5. Recovery ratios ─────────────────────────────────────────────────
    print('\n[5/8] Computing recovery ratios...')
    recovery = compute_recovery_ratios(table, baselines)

    # ── 6. Composition / sensitivity / safety ──────────────────────────────
    print('\n[6/8] Composition, sensitivity, and safety analysis...')
    composition = compute_composition_tests(table, deltas)
    sens        = compute_sensitivity(table, deltas)
    safety      = worst_case_safety(table, baselines)

    # ── 7. Verdicts ────────────────────────────────────────────────────────
    print('\n[7/8] Per-architecture verdicts...')
    verdicts = per_arch_verdict(deltas, safety)

    # ── 8. Print + save ────────────────────────────────────────────────────
    print('\n[8/8] Generating reports...\n')

    print_main_table(table, deltas, eval_type=args.eval_type)
    print_per_arch_deltas(deltas)
    print_recovery_table(recovery)
    print_composition_tests(composition)
    print_sensitivity(sens)
    print_safety(safety)
    print_verdicts(verdicts)

    # Diagnostic-prediction validation
    print(f'\n{"="*80}')
    print('  PREDICTION VALIDATION (from §5 / submit-script comments)')
    print(f'{"="*80}')
    for v in validate_predictions(deltas, table):
        print(v)

    # LaTeX
    latex = generate_latex_table(table, deltas, args.eval_type)
    latex_path = os.path.join(args.output_dir, 'table_newmod_ablations.tex')
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f'\n  LaTeX table → {latex_path}')

    # JSON dump
    report = generate_json_report(table, baselines, deltas, recovery,
                                  composition, sens, safety, verdicts,
                                  args.eval_type)
    json_path = os.path.join(args.output_dir, 'newmod_report.json')
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'  JSON report → {json_path}')

    # Supplementary metrics: F1, Shadow_IOU, BER
    print(f'\n{"="*80}')
    print('  SUPPLEMENTARY METRIC TABLES')
    print(f'{"="*80}')
    supp: Dict[str, Dict] = {}
    for extra in ['F1', 'Shadow_IOU', 'BER']:
        sup_table = build_master_table(all_results, metric_key=extra,
                                       eval_type=args.eval_type)
        print(f'\n  --- {extra} ({args.eval_type}) ---')
        print(f'  {"ID":<4}', end='')
        for arch in ARCHITECTURES:
            print(f' {arch:>10}', end='')
        print(f' {"Overall":>10}')
        supp[extra] = {}
        for exp_id in EXPERIMENT_META:
            print(f'  {exp_id:<4}', end='')
            all_v = []
            supp[extra][exp_id] = {}
            for arch in ARCHITECTURES:
                entry = sup_table[arch].get(exp_id)
                if entry and not np.isnan(entry['mean']):
                    print(f' {entry["mean"]:10.2f}', end='')
                    all_v.append(entry['mean'])
                    supp[extra][exp_id][arch] = {
                        c: float(entry[c]) if not np.isnan(entry.get(c, float('nan'))) else None
                        for c in CITIES
                    }
                    supp[extra][exp_id][arch]['mean'] = float(entry['mean'])
                else:
                    print(f' {"—":>10}', end='')
            if all_v:
                print(f' {np.mean(all_v):10.2f}')
            else:
                print(f' {"—":>10}')

    supp_path = os.path.join(args.output_dir, 'newmod_supplementary.json')
    with open(supp_path, 'w') as f:
        json.dump(supp, f, indent=2)
    print(f'\n  Supplementary metrics → {supp_path}')

    # Headline summary
    print(f'\n{"="*80}')
    print('  HEADLINE NUMBERS FOR PAPER')
    print(f'{"="*80}')
    n0_overall = []
    for arch in ARCHITECTURES:
        e = table[arch].get('N0')
        if e and not np.isnan(e['mean']):
            n0_overall.append(e['mean'])
    if n0_overall:
        print(f'  N0 (C4 baseline) overall mIoU:  {np.mean(n0_overall):.2f}')

    for arch in ARCHITECTURES:
        v = verdicts.get(arch, {})
        if v.get('best_experiment'):
            print(f'  {arch:<10} → {v["recommendation"]}')

    print(f'\n  Done. All outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()