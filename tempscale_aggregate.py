#!/usr/bin/env python3
"""
§5 Case Study Aggregator — Class-Conditional Temperature Scaling Results

Reads tempscale_results.json from each (arch, fold) C4-clean cell and
the corresponding comparison_results.json (for Vanilla / Upper Bound),
then produces the four-row §5.3 case-study table:

    Upper Bound  →  LOCO Vanilla  →  C4-clean (T=1)  →  C4-clean + tempscale

Outputs (in --output_dir):
    case_study_table.txt    — printed summary
    case_study_table.tex    — LaTeX-ready
    case_study_report.json  — machine-readable

Usage:
    python tempscale_aggregate.py \
        --mamnet_root  /path/to/mamnet/outputs \
        --oglanet_root /path/to/oglanet/outputs \
        --dinov3_root  /path/to/dinov3/outputs \
        --output_dir   /path/to/case_study_output
"""

import os
import json
import argparse
import re
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


ARCHITECTURES = ['MAMNet', 'OGLANet', 'DINOv3']
CITIES = ['chicago', 'miami', 'phoenix']
CITY_ABBREV = {'chicago': 'CHI', 'miami': 'MIA', 'phoenix': 'PHX'}
FOLD_MAP = {0: 'phoenix', 1: 'miami', 2: 'chicago'}

# C4-clean directory tag prefixes per architecture. Adjust to match your
# actual experiment naming.
C4CLEAN_TAGS = {
    'MAMNet':  ['C4clean_haar_vib_sag_fda_ctr', 'C4clean'],
    'OGLANet': ['C4clean_haar_vib_sag_fda_ctr', 'C4clean'],
    'DINOv3':  ['C4clean_haar_vib', 'C4clean'],
}


# ═════════════════════════════════════════════════════════════════════════════
# Directory discovery
# ═════════════════════════════════════════════════════════════════════════════

def identify_fold(dirname: str) -> Optional[int]:
    """Match dir name to fold ID (0=phoenix, 1=miami, 2=chicago)."""
    dn = dirname.lower()
    if 'holdout_phoenix' in dn or 'phoenix' in dn:
        return 0
    if 'holdout_miami' in dn or 'miami' in dn:
        return 1
    if 'holdout_chicago' in dn or 'chicago' in dn:
        return 2
    return None


def is_c4clean_dir(dirname: str, arch: str) -> bool:
    """Check if a directory is a C4-clean experiment for this architecture."""
    for tag in C4CLEAN_TAGS.get(arch, []):
        if tag.lower() in dirname.lower():
            return True
    return False


def find_c4clean_cells(root: str, arch: str) -> Dict[int, str]:
    """
    Scan an arch root directory for C4-clean experiment dirs.

    Returns {fold_id: full_dir_path} for the (up to) 3 folds.
    If multiple matches per fold, keeps the most recently modified.
    """
    if not os.path.isdir(root):
        return {}

    cells = {}  # fold_id -> (mtime, full_path)
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if not os.path.isdir(full):
            continue
        if not is_c4clean_dir(entry, arch):
            continue
        fold_id = identify_fold(entry)
        if fold_id is None:
            continue
        mtime = os.path.getmtime(full)
        if fold_id not in cells or mtime > cells[fold_id][0]:
            cells[fold_id] = (mtime, full)

    return {fid: path for fid, (_, path) in cells.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Result loading
# ═════════════════════════════════════════════════════════════════════════════

def load_cell_results(cell_dir: str, boundary_tolerance: int = 2) -> Optional[Dict]:
    """
    Load tempscale_results.json + comparison_results.json from one cell.

    Returns:
        {
            'tempscale': {T_pos, T_neg, baseline_T1, tempscale, sp_gap_reduction},
            'baselines': {Upper Bound: {strict, tolerant_Npx}, LOCO Vanilla: ...},
        }
        or None if either file is missing.
    """
    tol_key = f'tolerant_{boundary_tolerance}px'

    ts_path = os.path.join(cell_dir, 'tempscale_results.json')
    comp_path = os.path.join(cell_dir, 'comparison_results.json')

    if not os.path.isfile(ts_path):
        return None

    with open(ts_path) as f:
        tempscale = json.load(f)

    baselines = {}
    if os.path.isfile(comp_path):
        with open(comp_path) as f:
            comp = json.load(f)
        baselines = comp.get('baselines', {})

    return {
        'tempscale': tempscale,
        'baselines': baselines,
        'tol_key': tol_key,
    }


def extract_metric(metrics: Dict, key: str = 'mIOU') -> float:
    """Safe metric extraction."""
    if not isinstance(metrics, dict):
        return float('nan')
    return metrics.get(key, float('nan'))


# ═════════════════════════════════════════════════════════════════════════════
# Statistics
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 10000,
                  seed: int = 42) -> Tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for a list of per-fold values."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return float('nan'), float('nan'), float('nan')
    if len(valid) == 1:
        return float(valid[0]), float('nan'), float('nan')

    rng = np.random.RandomState(seed)
    boot_means = np.array([
        np.mean(valid[rng.choice(len(valid), len(valid), replace=True)])
        for _ in range(n_bootstrap)
    ])
    return (float(np.mean(valid)),
            float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


def paired_bootstrap_delta(vals_a: np.ndarray, vals_b: np.ndarray,
                            n_bootstrap: int = 10000,
                            seed: int = 42) -> Dict:
    """Paired bootstrap: mean(A − B) with CI and two-sided p-value."""
    n = min(len(vals_a), len(vals_b))
    a, b = vals_a[:n], vals_b[:n]
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]

    if len(a) == 0:
        return {'delta': float('nan'), 'ci_lo': float('nan'),
                'ci_hi': float('nan'), 'p_value': float('nan'), 'n': 0}

    diff = a - b
    obs = float(np.mean(diff))
    rng = np.random.RandomState(seed)
    boot = np.array([
        np.mean(diff[rng.choice(len(diff), len(diff), replace=True)])
        for _ in range(n_bootstrap)
    ])
    ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    if obs >= 0:
        p = 2 * max(np.mean(boot <= 0), 1.0 / n_bootstrap)
    else:
        p = 2 * max(np.mean(boot >= 0), 1.0 / n_bootstrap)
    return {'delta': obs, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'p_value': float(min(p, 1.0)), 'n': len(diff)}


def recovery_ratio(method: float, vanilla: float, upper: float) -> float:
    """R = (method − vanilla) / (upper − vanilla)."""
    if any(np.isnan([method, vanilla, upper])):
        return float('nan')
    gap = upper - vanilla
    if abs(gap) < 0.01:
        return float('nan')
    return float((method - vanilla) / gap)


# ═════════════════════════════════════════════════════════════════════════════
# Build per-architecture row data
# ═════════════════════════════════════════════════════════════════════════════

def build_arch_results(cells: Dict[int, Dict], boundary_tolerance: int) -> Dict:
    """
    For one architecture's 3 cells, build the 4-row table data.

    Returns:
        {
            'cells_loaded': [fold_ids],
            'rows': {
                'Upper Bound':       {city → mIoU, mean, ci_lo, ci_hi},
                'LOCO Vanilla':      {city → mIoU, mean, ci_lo, ci_hi},
                'C4-clean (T=1)':    {city → mIoU, mean, ci_lo, ci_hi},
                'C4-clean + tempscale': {city → mIoU, mean, ci_lo, ci_hi,
                                         + AURC_shadow, ECE_pred_pos, T_pos, T_neg},
            },
            'deltas': {
                'C4_clean_vs_vanilla':    bootstrap dict,
                'tempscale_vs_C4_clean':  bootstrap dict,
                'tempscale_vs_vanilla':   bootstrap dict,
            },
            'recovery': {
                'C4_clean_R':  per-city + mean,
                'tempscale_R': per-city + mean,
            },
            'sp_gap': {
                'baseline_aurc_shadow': mean across folds,
                'tempscale_aurc_shadow': mean across folds,
                'delta_aurc_shadow': baseline - tempscale,
                'baseline_ece_pos': mean,
                'tempscale_ece_pos': mean,
                'delta_ece_pos': baseline - tempscale,
                'T_pos_per_fold': [...],
                'T_neg_per_fold': [...],
            },
        }
    """
    tol_key = f'tolerant_{boundary_tolerance}px'

    # Per-row × per-city values
    rows = OrderedDict([
        ('Upper Bound',           {c: float('nan') for c in CITIES}),
        ('LOCO Vanilla',          {c: float('nan') for c in CITIES}),
        ('C4-clean (T=1)',        {c: float('nan') for c in CITIES}),
        ('C4-clean + tempscale',  {c: float('nan') for c in CITIES}),
    ])

    # SP-gap accumulators (per fold)
    sp_acc = {
        'baseline_aurc_shadow': [], 'tempscale_aurc_shadow': [],
        'baseline_ece_pos': [],     'tempscale_ece_pos': [],
        'T_pos_per_fold': [],       'T_neg_per_fold': [],
    }

    cells_loaded = []
    for fold_id, cell in cells.items():
        if cell is None:
            continue
        cells_loaded.append(fold_id)
        city = FOLD_MAP[fold_id]

        # Baselines from comparison_results
        bl = cell['baselines']
        ub = bl.get('Upper Bound', {})
        lv = bl.get('LOCO Vanilla', {})

        ub_metrics = ub.get(tol_key, ub.get('strict', {}))
        lv_metrics = lv.get(tol_key, lv.get('strict', {}))
        rows['Upper Bound'][city]  = extract_metric(ub_metrics)
        rows['LOCO Vanilla'][city] = extract_metric(lv_metrics)

        # C4-clean (T=1) and tempscale rows from tempscale_results.json
        ts = cell['tempscale']
        base_T1 = ts.get('baseline_T1', {})
        ts_app  = ts.get('tempscale', {})

        rows['C4-clean (T=1)'][city]       = extract_metric(
            base_T1.get(tol_key, base_T1.get('strict', {})))
        rows['C4-clean + tempscale'][city] = extract_metric(
            ts_app.get(tol_key, ts_app.get('strict', {})))

        # SP-gap metrics
        base_sp = base_T1.get('sp_metrics', {})
        ts_sp   = ts_app.get('sp_metrics', {})
        sp_acc['baseline_aurc_shadow'].append(base_sp.get('aurc_shadow', float('nan')))
        sp_acc['tempscale_aurc_shadow'].append(ts_sp.get('aurc_shadow', float('nan')))
        sp_acc['baseline_ece_pos'].append(base_sp.get('ece_pred_pos', float('nan')))
        sp_acc['tempscale_ece_pos'].append(ts_sp.get('ece_pred_pos', float('nan')))
        sp_acc['T_pos_per_fold'].append(ts.get('T_pos', float('nan')))
        sp_acc['T_neg_per_fold'].append(ts.get('T_neg', float('nan')))

    # Per-row mean + bootstrap CI across folds
    for row_name, city_dict in rows.items():
        vals = np.array([city_dict[c] for c in CITIES])
        mean, lo, hi = bootstrap_ci(vals)
        city_dict['mean']  = mean
        city_dict['ci_lo'] = lo
        city_dict['ci_hi'] = hi

    # Paired deltas across the same 3 folds
    upper_vals    = np.array([rows['Upper Bound'][c]          for c in CITIES])
    vanilla_vals  = np.array([rows['LOCO Vanilla'][c]         for c in CITIES])
    c4clean_vals  = np.array([rows['C4-clean (T=1)'][c]       for c in CITIES])
    ts_vals       = np.array([rows['C4-clean + tempscale'][c] for c in CITIES])

    deltas = {
        'C4_clean_vs_vanilla':    paired_bootstrap_delta(c4clean_vals, vanilla_vals),
        'tempscale_vs_C4_clean':  paired_bootstrap_delta(ts_vals, c4clean_vals),
        'tempscale_vs_vanilla':   paired_bootstrap_delta(ts_vals, vanilla_vals),
    }

    # Recovery ratios (per city, then mean)
    R_c4   = {c: recovery_ratio(rows['C4-clean (T=1)'][c],
                                 rows['LOCO Vanilla'][c],
                                 rows['Upper Bound'][c])
              for c in CITIES}
    R_ts   = {c: recovery_ratio(rows['C4-clean + tempscale'][c],
                                 rows['LOCO Vanilla'][c],
                                 rows['Upper Bound'][c])
              for c in CITIES}
    R_c4['mean'] = float(np.nanmean(list(R_c4.values())))
    R_ts['mean'] = float(np.nanmean(list(R_ts.values())))

    # Worst single-cell ΔmIoU vs Vanilla
    worst_c4 = float(np.nanmin(c4clean_vals - vanilla_vals)) if len(c4clean_vals) else float('nan')
    worst_ts = float(np.nanmin(ts_vals - vanilla_vals))      if len(ts_vals)      else float('nan')

    # SP-gap aggregates
    sp_gap = {
        'baseline_aurc_shadow_mean':  float(np.nanmean(sp_acc['baseline_aurc_shadow'])),
        'tempscale_aurc_shadow_mean': float(np.nanmean(sp_acc['tempscale_aurc_shadow'])),
        'baseline_ece_pos_mean':      float(np.nanmean(sp_acc['baseline_ece_pos'])),
        'tempscale_ece_pos_mean':     float(np.nanmean(sp_acc['tempscale_ece_pos'])),
        'T_pos_mean':                 float(np.nanmean(sp_acc['T_pos_per_fold'])),
        'T_neg_mean':                 float(np.nanmean(sp_acc['T_neg_per_fold'])),
        'T_pos_per_fold':             sp_acc['T_pos_per_fold'],
        'T_neg_per_fold':             sp_acc['T_neg_per_fold'],
    }
    sp_gap['delta_aurc_shadow'] = (sp_gap['baseline_aurc_shadow_mean']
                                    - sp_gap['tempscale_aurc_shadow_mean'])
    sp_gap['delta_ece_pos']     = (sp_gap['baseline_ece_pos_mean']
                                    - sp_gap['tempscale_ece_pos_mean'])

    return {
        'cells_loaded': cells_loaded,
        'rows': rows,
        'deltas': deltas,
        'recovery': {'C4_clean_R': R_c4, 'tempscale_R': R_ts},
        'worst_vs_vanilla': {'C4_clean': worst_c4, 'tempscale': worst_ts},
        'sp_gap': sp_gap,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Pretty-print
# ═════════════════════════════════════════════════════════════════════════════

def sig_stars(p: float) -> str:
    if np.isnan(p):     return ''
    if p < 0.001:       return '***'
    if p < 0.01:        return '**'
    if p < 0.05:        return '*'
    return ''


def print_arch_block(arch: str, data: Dict, boundary_tolerance: int):
    """Print the 4-row table for one architecture."""
    print()
    print('=' * 90)
    print(f'  {arch}  (cells loaded: {sorted(data["cells_loaded"])})')
    print('=' * 90)

    # Header
    print(f'  {"Row":<26}', end='')
    for c in CITIES:
        print(f' {CITY_ABBREV[c]:>7}', end='')
    print(f' {"Mean":>7} {"95%CI":>16}')
    print('  ' + '-' * 86)

    for row_name, city_dict in data['rows'].items():
        line = f'  {row_name:<26}'
        for c in CITIES:
            v = city_dict[c]
            line += f' {v:7.2f}' if not np.isnan(v) else f' {"—":>7}'
        m = city_dict['mean']
        line += f' {m:7.2f}' if not np.isnan(m) else f' {"—":>7}'
        if not np.isnan(city_dict['ci_lo']):
            line += f' [{city_dict["ci_lo"]:5.2f},{city_dict["ci_hi"]:5.2f}]'
        else:
            line += f' {"—":>16}'
        print(line)

    print()
    print(f'  Paired deltas (3-fold bootstrap, n={data["deltas"]["C4_clean_vs_vanilla"]["n"]}):')
    for label, key in [('C4-clean − Vanilla        ', 'C4_clean_vs_vanilla'),
                       ('Tempscale − C4-clean      ', 'tempscale_vs_C4_clean'),
                       ('Tempscale − Vanilla       ', 'tempscale_vs_vanilla')]:
        d = data['deltas'][key]
        if d['n'] == 0:
            continue
        stars = sig_stars(d['p_value'])
        print(f'    Δ {label} = {d["delta"]:+6.2f}  '
              f'95%CI=[{d["ci_lo"]:+5.2f},{d["ci_hi"]:+5.2f}]  '
              f'p={d["p_value"]:6.4f}{stars}')

    print()
    print(f'  Recovery ratios R = (method − Vanilla) / (Upper − Vanilla):')
    for label, key in [('C4-clean (T=1)         ', 'C4_clean_R'),
                       ('C4-clean + tempscale   ', 'tempscale_R')]:
        r = data['recovery'][key]
        per_city_str = '  '.join(
            f'{CITY_ABBREV[c]}={r[c]:+.3f}' if not np.isnan(r[c]) else f'{CITY_ABBREV[c]}=—'
            for c in CITIES)
        print(f'    R[{label}] = {r["mean"]:+.3f}  ({per_city_str})')

    print()
    print(f'  Worst single-cell Δ vs Vanilla:')
    print(f'    C4-clean (T=1):       {data["worst_vs_vanilla"]["C4_clean"]:+.2f} mIoU')
    print(f'    C4-clean + tempscale: {data["worst_vs_vanilla"]["tempscale"]:+.2f} mIoU')

    print()
    print(f'  §4.3 SP-gap metrics (mean across folds):')
    sp = data['sp_gap']
    print(f'    AURC_shadow:  baseline={sp["baseline_aurc_shadow_mean"]:.4f}  '
          f'tempscale={sp["tempscale_aurc_shadow_mean"]:.4f}  '
          f'Δ={sp["delta_aurc_shadow"]:+.4f}')
    print(f'    ECE_pred_pos: baseline={sp["baseline_ece_pos_mean"]:.4f}  '
          f'tempscale={sp["tempscale_ece_pos_mean"]:.4f}  '
          f'Δ={sp["delta_ece_pos"]:+.4f}')
    print(f'    Fitted temperatures (mean): T_pos={sp["T_pos_mean"]:.3f}  '
          f'T_neg={sp["T_neg_mean"]:.3f}')
    print(f'    Per fold T_pos: {[f"{t:.3f}" for t in sp["T_pos_per_fold"]]}')
    print(f'    Per fold T_neg: {[f"{t:.3f}" for t in sp["T_neg_per_fold"]]}')


def print_headline(report: Dict, boundary_tolerance: int):
    """Print cross-architecture headline numbers for the abstract / conclusion."""
    print()
    print('=' * 90)
    print(f'  HEADLINE NUMBERS (cross-architecture)')
    print('=' * 90)

    # Recovery range across archs
    R_c4  = [report[a]['recovery']['C4_clean_R']['mean'] for a in ARCHITECTURES
             if a in report and not np.isnan(report[a]['recovery']['C4_clean_R']['mean'])]
    R_ts  = [report[a]['recovery']['tempscale_R']['mean'] for a in ARCHITECTURES
             if a in report and not np.isnan(report[a]['recovery']['tempscale_R']['mean'])]
    if R_c4:
        print(f'  C4-clean gap closure R: mean={np.mean(R_c4):.3f}  '
              f'range=[{min(R_c4):.3f}, {max(R_c4):.3f}]  '
              f'({100*np.mean(R_c4):.0f}% on average)')
    if R_ts:
        print(f'  Tempscale  gap closure R: mean={np.mean(R_ts):.3f}  '
              f'range=[{min(R_ts):.3f}, {max(R_ts):.3f}]  '
              f'({100*np.mean(R_ts):.0f}% on average)')

    # Worst-case across archs
    worst_c4 = [report[a]['worst_vs_vanilla']['C4_clean']
                for a in ARCHITECTURES if a in report]
    worst_ts = [report[a]['worst_vs_vanilla']['tempscale']
                for a in ARCHITECTURES if a in report]
    if worst_c4:
        print(f'  Worst single-cell Δ vs Vanilla (across {len(worst_c4)*3} cells):')
        print(f'    C4-clean (T=1):       min={np.nanmin(worst_c4):+.2f} mIoU')
        print(f'    C4-clean + tempscale: min={np.nanmin(worst_ts):+.2f} mIoU')

    # SP-gap reduction
    print()
    print(f'  §4.3 SP-gap reductions (tempscale vs T=1, mean across 3 folds per arch):')
    for arch in ARCHITECTURES:
        if arch not in report:
            continue
        sp = report[arch]['sp_gap']
        print(f'    {arch:8s} ΔAURC_shadow={sp["delta_aurc_shadow"]:+.4f}  '
              f'ΔECE_pos={sp["delta_ece_pos"]:+.4f}  '
              f'(T_pos≈{sp["T_pos_mean"]:.2f}, T_neg≈{sp["T_neg_mean"]:.2f})')

    # Tempscale safety property
    print()
    n_hurt_c4 = 0
    n_hurt_ts = 0
    for arch in ARCHITECTURES:
        if arch not in report:
            continue
        for c in CITIES:
            v = report[arch]['rows']['LOCO Vanilla'][c]
            c4 = report[arch]['rows']['C4-clean (T=1)'][c]
            ts = report[arch]['rows']['C4-clean + tempscale'][c]
            if not np.isnan(c4 - v) and (c4 - v) < 0:
                n_hurt_c4 += 1
            if not np.isnan(ts - v) and (ts - v) < 0:
                n_hurt_ts += 1
    print(f'  Cells where method hurts vs Vanilla:')
    print(f'    C4-clean (T=1):       {n_hurt_c4}/9')
    print(f'    C4-clean + tempscale: {n_hurt_ts}/9')


# ═════════════════════════════════════════════════════════════════════════════
# LaTeX
# ═════════════════════════════════════════════════════════════════════════════

def generate_latex(report: Dict, boundary_tolerance: int) -> str:
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{')
    lines.append(rf'    \textbf{{\S5 case study --- diagnostic-grounded module build (\textpm{boundary_tolerance}px tolerant mIoU).}}')
    lines.append(r'    Each row adds the next diagnostic-motivated module.')
    lines.append(r'    C4-clean = Haar+VIB+SAG+FDA (\S4.2 + \S4.4); tempscale adds \S4.3.')
    lines.append(r'    For DINOv3 the \S4.4 module is omitted because \S4.4 diagnostics show')
    lines.append(r'    its encoder is city-agnostic; C4-clean for DINOv3 = Haar+VIB only.')
    lines.append(r'    $R$ = (method $-$ Vanilla) / (Upper $-$ Vanilla); $\Delta$AURC and $\Delta$ECE on gt-shadow / pred-pos.')
    lines.append(r'  }')
    lines.append(r'  \label{tab:case_study}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{@{}llcccccc@{}}')
    lines.append(r'    \toprule')
    lines.append(r'    Arch & Row & CHI & MIA & PHX & Mean & $R$ & $\Delta$AURC$_{\text{sh}}$ \\')
    lines.append(r'    \midrule')

    for arch in ARCHITECTURES:
        if arch not in report:
            continue
        d = report[arch]
        sp = d['sp_gap']
        for row_name in ['Upper Bound', 'LOCO Vanilla',
                         'C4-clean (T=1)', 'C4-clean + tempscale']:
            r = d['rows'][row_name]
            row_parts = [arch if row_name == 'Upper Bound' else '',
                         row_name]
            for c in CITIES:
                v = r[c]
                row_parts.append(f'{v:.1f}' if not np.isnan(v) else '--')
            m = r['mean']
            row_parts.append(f'{m:.1f}' if not np.isnan(m) else '--')

            if row_name == 'C4-clean (T=1)':
                R = d['recovery']['C4_clean_R']['mean']
                row_parts.append(f'{R:.2f}' if not np.isnan(R) else '--')
                row_parts.append('--')
            elif row_name == 'C4-clean + tempscale':
                R = d['recovery']['tempscale_R']['mean']
                row_parts.append(f'{R:.2f}' if not np.isnan(R) else '--')
                row_parts.append(f'{sp["delta_aurc_shadow"]:+.3f}')
            else:
                row_parts.extend(['--', '--'])

            lines.append('    ' + ' & '.join(row_parts) + r' \\')
        lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# JSON serialization helper
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
    if isinstance(val, (list, tuple)):
        return [_clean(v) for v in val]
    if isinstance(val, float) and np.isnan(val):
        return None
    return val


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='§5 Case Study Aggregator — tempscale results across archs')
    base = os.path.join(os.environ["PROJECT_ROOT"], 'data')
    p.add_argument('--mamnet_root', type=str,
                   default=os.path.join(base, 'mamnet/outputs'))
    p.add_argument('--oglanet_root', type=str,
                   default=os.path.join(base, 'oglanet/outputs'))
    p.add_argument('--dinov3_root', type=str,
                   default=os.path.join(base, 'dinov3/outputs'))
    p.add_argument('--output_dir', type=str,
                   default='./case_study_output')
    p.add_argument('--boundary_tolerance', type=int, default=2)
    p.add_argument('--n_bootstrap', type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 90)
    print(f'  §5 CASE STUDY AGGREGATION  —  Tempscale Results')
    print(f'  Tolerance: ±{args.boundary_tolerance}px  |  '
          f'Bootstrap: {args.n_bootstrap}')
    print('=' * 90)

    arch_roots = {'MAMNet':  args.mamnet_root,
                  'OGLANet': args.oglanet_root,
                  'DINOv3':  args.dinov3_root}

    # ── Step 1: Discover and load cells ──────────────────────────────────
    print('\n[1/3] Scanning C4-clean experiment directories...')
    raw_cells = {}
    for arch, root in arch_roots.items():
        cell_dirs = find_c4clean_cells(root, arch)
        if not cell_dirs:
            print(f'  {arch:8s}  ✗ no C4-clean cells found in {root}')
            continue
        loaded = {}
        for fold_id, cell_dir in cell_dirs.items():
            cell = load_cell_results(cell_dir, args.boundary_tolerance)
            if cell is None:
                print(f'  {arch:8s}  fold={fold_id} ({FOLD_MAP[fold_id]:8s})  '
                      f'✗ no tempscale_results.json in {os.path.basename(cell_dir)}')
                continue
            loaded[fold_id] = cell
            print(f'  {arch:8s}  fold={fold_id} ({FOLD_MAP[fold_id]:8s})  '
                  f'✓ {os.path.basename(cell_dir)}')
        raw_cells[arch] = loaded

    if not any(raw_cells.values()):
        print('\n  ! No tempscale results found. Did the eval jobs complete?')
        print('  Check: <output_dir>/tempscale_results.json should exist for each cell.')
        return

    # ── Step 2: Build per-arch report ─────────────────────────────────────
    print('\n[2/3] Building per-architecture results...')
    report = {}
    for arch, cells in raw_cells.items():
        if not cells:
            continue
        report[arch] = build_arch_results(cells, args.boundary_tolerance)
        print(f'  {arch:8s}  built ({len(cells)}/3 cells)')

    # ── Step 3: Print, save ───────────────────────────────────────────────
    print('\n[3/3] Generating outputs...')

    # Console summary per arch
    for arch in ARCHITECTURES:
        if arch in report:
            print_arch_block(arch, report[arch], args.boundary_tolerance)

    # Headline numbers
    print_headline(report, args.boundary_tolerance)

    # Save text report (capture stdout-equivalent into file)
    txt_path = os.path.join(args.output_dir, 'case_study_table.txt')
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for arch in ARCHITECTURES:
            if arch in report:
                print_arch_block(arch, report[arch], args.boundary_tolerance)
        print_headline(report, args.boundary_tolerance)
    with open(txt_path, 'w') as f:
        f.write(buf.getvalue())
    print(f'\n  Text report → {txt_path}')

    # LaTeX
    latex = generate_latex(report, args.boundary_tolerance)
    tex_path = os.path.join(args.output_dir, 'case_study_table.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f'  LaTeX table → {tex_path}')

    # JSON
    full_report = {
        'generated': datetime.now().isoformat(),
        'boundary_tolerance': args.boundary_tolerance,
        'n_bootstrap': args.n_bootstrap,
        'architectures': list(report.keys()),
        'per_arch': _clean(report),
    }
    json_path = os.path.join(args.output_dir, 'case_study_report.json')
    with open(json_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    print(f'  JSON report → {json_path}')

    print(f'\n  Done. All outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()