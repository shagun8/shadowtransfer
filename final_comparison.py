#!/usr/bin/env python3
"""
Final Comparison: C4_clean vs all baselines, per (architecture, city).

Recomputes tolerant mIoU FROM SCRATCH using prediction PNGs vs GT masks —
no shortcuts via comparison_results.json files.

For each (architecture, city) cell:
  - Computes per-image tolerant mIoU for every method:
      Upper Bound, LOCO Vanilla, LOCO FDA, LOCO SegDesic,
      LOCO IIM, LOCO ISW, LOCO MRFP+, LOCO FADA, C4_clean
  - Builds a master table of mean tolerant mIoU
  - Performs paired bootstrap (B=10000) of C4_clean vs each baseline
  - Reports observed Δ mIoU, 95% CI, two-sided p-value
  - Applies Holm-Bonferroni correction within each (arch, city) family
  - Also reports global correction across all (arch × city × baseline) tests

Sources (NCSA Delta defaults):
  Test image roots (baseline predictions):
    $PROJECT_ROOT/data/Test_img_results/loco/{city}/highres/{arch_lower}/{method}/
    $PROJECT_ROOT/data/Test_img_results/upper/{city}/highres/{arch_lower}/base/
  C4_clean predictions (per architecture):
    DINOv3 :  data/dinov3/outputs/dinov3_sib_C4clean_haar_vib_loco_holdout_{city}_highres_1/predictions/
    MAMNet :  data/mamnet/outputs/mamnet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_{city}__highres/predictions/
    OGLANet:  data/oglanet/outputs/oglanet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_{city}__highres/predictions/
  Ground-truth masks:
    $PROJECT_ROOT/data/Final_data_test/{city}/highres/test/masks/
    (with fallback candidates)

Outputs (in --output_dir):
  final_comparison_report.txt   — human-readable text report
  final_comparison_table.tex    — LaTeX table for the paper
  final_comparison.json         — full machine-readable dump
  bootstrap_results.json        — per-comparison bootstrap details

Usage:
    python final_comparison.py \\
        --base_path $PROJECT_ROOT \\
        --output_dir ./final_comparison_output \\
        --boundary_tolerance 2 \\
        --img_size 384 \\
        --n_bootstrap 10000

If --base_path is not passed, uses the NCSA Delta default.
"""

import os
import sys
import json
import glob
import argparse
import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

warnings.filterwarnings('ignore', category=FutureWarning)


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

ARCHITECTURES = ['MAMNet', 'OGLANet', 'DINOv3']
ARCH_LOWER = {'MAMNet': 'mamnet', 'OGLANet': 'oglanet', 'DINOv3': 'dinov3'}
CITIES = ['chicago', 'miami', 'phoenix']
CITY_ABBREV = {'chicago': 'CHI', 'miami': 'MIA', 'phoenix': 'PHX'}

# Method labels in the order they should appear in tables
METHOD_ORDER = [
    'Upper Bound',
    'LOCO Vanilla',
    'LOCO FDA',
    'LOCO SegDesic',
    'LOCO IIM',
    'LOCO ISW',
    'LOCO MRFP+',
    'LOCO FADA',
    'C4_clean',
]

# Map method label → subdirectory name under {root}/{loco|upper}/{city}/highres/{arch}/
METHOD_TO_SUBDIR = {
    'Upper Bound':   ('upper', 'base'),
    'LOCO Vanilla':  ('loco',  'vanilla'),
    'LOCO FDA':      ('loco',  'fda'),
    'LOCO SegDesic': ('loco',  'segdesic'),
    'LOCO IIM':      ('loco',  'iim'),
    'LOCO ISW':      ('loco',  'isw'),
    'LOCO MRFP+':    ('loco',  'mrfp_plus'),
    'LOCO FADA':     ('loco',  'fada'),
}

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


# ════════════════════════════════════════════════════════════════════════════
# Per-image tolerant metric
# ════════════════════════════════════════════════════════════════════════════

_TOLERANCE_KERNEL_CACHE = {}


def _get_tolerance_kernel(tolerance):
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1))
    return _TOLERANCE_KERNEL_CACHE[tolerance]


def compute_tolerant_miou(pred, gt, tolerance=2):
    """Compute tolerant mIoU for a single (pred, gt) image pair.

    pred, gt are uint8 arrays of the same shape, with values in {0, 1}.
    Returns tolerant mIoU as a float in [0, 100].
    """
    kernel   = _get_tolerance_kernel(tolerance)
    gt_uint8 = gt.astype(np.uint8)
    eroded   = cv2.erode(gt_uint8, kernel)
    dilated  = cv2.dilate(gt_uint8, kernel)
    valid    = ~((dilated - eroded) > 0)

    p  = pred[valid]
    g  = gt[valid]
    tp = np.logical_and(p == 1, g == 1).sum()
    fp = np.logical_and(p == 1, g == 0).sum()
    tn = np.logical_and(p == 0, g == 0).sum()
    fn = np.logical_and(p == 0, g == 1).sum()

    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou          = (shadow_iou + nonshadow_iou) / 2
    return float(miou * 100)


# ════════════════════════════════════════════════════════════════════════════
# Directory + file discovery
# ════════════════════════════════════════════════════════════════════════════

def _stem_map(directory):
    """{filename_stem: full_path} for all images in a directory."""
    m = {}
    if not os.path.isdir(directory):
        return m
    for fn in os.listdir(directory):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            m[os.path.splitext(fn)[0]] = os.path.join(directory, fn)
    return m


def find_gt_dir(base_path: str, city: str, res: str = 'highres') -> Optional[str]:
    """Find the GT mask directory for a given city, trying common layouts."""
    candidates = [
        os.path.join(base_path, 'data', 'Final_data_test', city, res, 'test', 'masks'),
        os.path.join(base_path, 'data', 'Final_data_test', city, res, 'masks'),
        os.path.join(base_path, 'data', 'Final_data_test', 'test', 'masks'),
        os.path.join(base_path, 'data', 'Final_data_test', 'masks'),
    ]
    for c in candidates:
        if os.path.isdir(c) and len(_stem_map(c)) > 0:
            return c
    return None


def find_baseline_pred_dir(base_path: str, arch: str, city: str,
                           method_label: str, res: str = 'highres') -> Optional[str]:
    """Find the prediction directory for a baseline method."""
    if method_label not in METHOD_TO_SUBDIR:
        return None
    split, subdir = METHOD_TO_SUBDIR[method_label]
    arch_lower = ARCH_LOWER[arch]
    pred_dir = os.path.join(
        base_path, 'data', 'Test_img_results', split, city, res, arch_lower, subdir)
    if os.path.isdir(pred_dir) and len(_stem_map(pred_dir)) > 0:
        return pred_dir
    return None


def find_c4clean_pred_dir(base_path: str, arch: str, city: str,
                          res: str = 'highres') -> Optional[str]:
    """
    Find the C4_clean predictions directory for a given (arch, city).

    Try the canonical experiment names first, then glob as a fallback.
    """
    arch_lower = ARCH_LOWER[arch]
    canonical = {
        'DINOv3':  f'dinov3_sib_C4clean_haar_vib_loco_holdout_{city}_{res}_1',
        'MAMNet':  f'mamnet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_{city}__{res}',
        'OGLANet': f'oglanet_sib_C4clean_haar_vib_sag_fda_ctr__loco_holdout_{city}__{res}',
    }
    base_outputs = os.path.join(base_path, 'data', arch_lower, 'outputs')

    # Canonical name first
    canon_path = os.path.join(base_outputs, canonical[arch], 'predictions')
    if os.path.isdir(canon_path) and len(_stem_map(canon_path)) > 0:
        return canon_path

    # Fallback: glob for any C4clean experiment matching this city
    pattern = os.path.join(
        base_outputs, f'*C4clean*holdout_{city}*', 'predictions')
    candidates = glob.glob(pattern)
    candidates = [c for c in candidates if os.path.isdir(c)
                  and len(_stem_map(c)) > 0]
    if candidates:
        # Prefer the most-recently modified
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    return None


# ════════════════════════════════════════════════════════════════════════════
# Per-image tolerant mIoU computation for a (pred_dir, gt_dir) pair
# ════════════════════════════════════════════════════════════════════════════

def per_image_tolerant_mious(pred_dir: str, gt_dir: str, img_size: int,
                              tolerance: int) -> Tuple[List[float], List[str]]:
    """
    Compute tolerant mIoU for every (pred, gt) pair matched by stem.

    Returns:
        ious: list of per-image tolerant mIoU values
        stems: list of corresponding image stems (for pairing across methods)
    """
    pred_map = _stem_map(pred_dir)
    gt_map = _stem_map(gt_dir)

    if not pred_map or not gt_map:
        return [], []

    # Pair by stem
    pairs = []
    for stem, pred_path in sorted(pred_map.items()):
        if stem in gt_map:
            pairs.append((stem, pred_path, gt_map[stem]))

    ious = []
    stems = []
    sz = (img_size, img_size)

    for stem, pred_path, gt_path in pairs:
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
        if pred_img is None or gt_img is None:
            continue

        if pred_img.shape != (img_size, img_size):
            pred_img = cv2.resize(pred_img, sz, interpolation=cv2.INTER_NEAREST)
        if gt_img.shape != (img_size, img_size):
            gt_img = cv2.resize(gt_img, sz, interpolation=cv2.INTER_NEAREST)

        pred_bin = (pred_img > 127).astype(np.uint8)
        gt_bin   = (gt_img   > 127).astype(np.uint8)

        ious.append(compute_tolerant_miou(pred_bin, gt_bin, tolerance=tolerance))
        stems.append(stem)

    return ious, stems


# ════════════════════════════════════════════════════════════════════════════
# Bootstrap statistics
# ════════════════════════════════════════════════════════════════════════════

def paired_bootstrap(vals_a: np.ndarray, vals_b: np.ndarray,
                     n_bootstrap: int = 10000, seed: int = 42) -> Dict:
    """
    Paired bootstrap: is mean(A) - mean(B) different from 0?

    Both inputs are arrays of per-image values from the SAME images.
    Returns observed delta, 95% CI, two-sided p-value.
    """
    rng = np.random.RandomState(seed)
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return {'delta': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                'p_value': np.nan, 'n': 0}

    diff = vals_a[:n] - vals_b[:n]
    obs_delta = float(np.mean(diff))

    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_deltas[i] = np.mean(diff[idx])

    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))

    if obs_delta >= 0:
        p_val = 2 * max(np.mean(boot_deltas <= 0), 1.0 / n_bootstrap)
    else:
        p_val = 2 * max(np.mean(boot_deltas >= 0), 1.0 / n_bootstrap)
    p_val = float(min(p_val, 1.0))

    return {
        'delta': obs_delta,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'p_value': p_val,
        'n': int(n),
    }


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Holm–Bonferroni correction. Returns reject-null booleans."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted = alpha / (n - rank)
        if p <= adjusted:
            reject[orig_idx] = True
        else:
            break
    return reject


def sig_stars(p_val: float) -> str:
    if np.isnan(p_val):
        return ''
    if p_val < 0.001:
        return '***'
    if p_val < 0.01:
        return '**'
    if p_val < 0.05:
        return '*'
    return ''


# ════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ════════════════════════════════════════════════════════════════════════════

def collect_all_metrics(base_path: str, img_size: int, tolerance: int,
                        verbose: bool = True) -> Dict:
    """
    For every (arch, city, method) cell:
      - Resolve prediction directory
      - Pair with GT
      - Compute per-image tolerant mIoU

    Returns:
        results[arch][city][method] = {
            'ious':        np.ndarray of per-image tolerant mIoU,
            'stems':       list of image stems
            'mean':        float mean tolerant mIoU
            'n':           number of images
            'pred_dir':    source directory
        }
    """
    results = defaultdict(lambda: defaultdict(dict))

    for city in CITIES:
        gt_dir = find_gt_dir(base_path, city)
        if gt_dir is None:
            print(f'  WARNING: GT dir not found for {city}; skipping')
            continue
        if verbose:
            print(f'\n  GT for {city}: {gt_dir} ({len(_stem_map(gt_dir))} images)')

        for arch in ARCHITECTURES:
            if verbose:
                print(f'  --- {arch} / {city} ---')

            for method in METHOD_ORDER:
                if method == 'C4_clean':
                    pred_dir = find_c4clean_pred_dir(base_path, arch, city)
                else:
                    pred_dir = find_baseline_pred_dir(base_path, arch, city, method)

                if pred_dir is None:
                    if verbose:
                        print(f'    {method:<14}  ✗  no prediction dir found')
                    results[arch][city][method] = {
                        'ious': np.array([]),
                        'stems': [],
                        'mean': np.nan,
                        'n': 0,
                        'pred_dir': None,
                    }
                    continue

                ious, stems = per_image_tolerant_mious(
                    pred_dir, gt_dir, img_size, tolerance)

                if not ious:
                    if verbose:
                        print(f'    {method:<14}  ✗  no matched (pred, gt) pairs '
                              f'({pred_dir})')
                    results[arch][city][method] = {
                        'ious': np.array([]),
                        'stems': [],
                        'mean': np.nan,
                        'n': 0,
                        'pred_dir': pred_dir,
                    }
                    continue

                ious_arr = np.array(ious)
                results[arch][city][method] = {
                    'ious': ious_arr,
                    'stems': stems,
                    'mean': float(ious_arr.mean()),
                    'n': len(ious),
                    'pred_dir': pred_dir,
                }
                if verbose:
                    print(f'    {method:<14}  ✓  n={len(ious):3d}  '
                          f'tol mIoU={ious_arr.mean():6.2f}')

    return dict(results)


def align_for_pairing(method_a: Dict, method_b: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align per-image scores from two methods to a common set of stems.

    Each input dict has keys 'ious' and 'stems'. Returns aligned numpy
    arrays of the same length.
    """
    if method_a['n'] == 0 or method_b['n'] == 0:
        return np.array([]), np.array([])
    stems_a = method_a['stems']
    ious_a = method_a['ious']
    map_b = {s: i for i, s in enumerate(method_b['stems'])}
    a_aligned = []
    b_aligned = []
    for stem, val_a in zip(stems_a, ious_a):
        if stem in map_b:
            a_aligned.append(val_a)
            b_aligned.append(method_b['ious'][map_b[stem]])
    return np.array(a_aligned), np.array(b_aligned)


def run_bootstrap_comparisons(results: Dict, n_bootstrap: int,
                              alpha: float = 0.05) -> Dict:
    """
    Bootstrap C4_clean vs every other method for each (arch, city).

    Per (arch, city) family: Holm–Bonferroni across the comparisons in
    that cell.

    Global family: Holm–Bonferroni across ALL (arch × city × baseline)
    comparisons.
    """
    boots = defaultdict(lambda: defaultdict(dict))
    global_keys = []
    global_pvals = []

    other_methods = [m for m in METHOD_ORDER if m != 'C4_clean']

    for arch in ARCHITECTURES:
        for city in CITIES:
            cell = results.get(arch, {}).get(city, {})
            c4 = cell.get('C4_clean')
            if c4 is None or c4['n'] == 0:
                # All comparisons unavailable in this cell
                for method in other_methods:
                    boots[arch][city][method] = {
                        'delta': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                        'p_value': np.nan, 'n': 0,
                        'cell_sig_corrected': False,
                        'global_sig_corrected': False,
                    }
                continue

            # Collect raw p-values within this cell
            cell_pvals = []
            cell_methods_used = []

            for method in other_methods:
                other = cell.get(method)
                if other is None or other['n'] == 0:
                    boots[arch][city][method] = {
                        'delta': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                        'p_value': np.nan, 'n': 0,
                        'cell_sig_corrected': False,
                        'global_sig_corrected': False,
                    }
                    continue

                # Pair on shared stems
                c4_a, other_a = align_for_pairing(c4, other)
                if len(c4_a) == 0:
                    boots[arch][city][method] = {
                        'delta': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                        'p_value': np.nan, 'n': 0,
                        'cell_sig_corrected': False,
                        'global_sig_corrected': False,
                    }
                    continue

                # Bootstrap of (C4_clean - other)
                boot = paired_bootstrap(c4_a, other_a,
                                        n_bootstrap=n_bootstrap)
                boots[arch][city][method] = boot
                cell_pvals.append(boot['p_value'])
                cell_methods_used.append(method)
                global_keys.append((arch, city, method))
                global_pvals.append(boot['p_value'])

            # Per-cell Holm–Bonferroni
            if cell_pvals:
                cell_reject = holm_bonferroni(cell_pvals, alpha=alpha)
                for m, r in zip(cell_methods_used, cell_reject):
                    boots[arch][city][m]['cell_sig_corrected'] = bool(r)

    # Global Holm–Bonferroni
    if global_pvals:
        global_reject = holm_bonferroni(global_pvals, alpha=alpha)
        for (arch, city, method), r in zip(global_keys, global_reject):
            boots[arch][city][method]['global_sig_corrected'] = bool(r)

    return {arch: dict(d) for arch, d in boots.items()}


# ════════════════════════════════════════════════════════════════════════════
# Reporting
# ════════════════════════════════════════════════════════════════════════════

def render_main_table(results: Dict, lines: List[str]):
    """Per (arch, city) tolerant mIoU table for every method."""
    lines.append('')
    lines.append('=' * 100)
    lines.append('  TABLE 1: TOLERANT mIoU per (Architecture × City × Method)')
    lines.append('=' * 100)

    header = f'  {"Method":<16}'
    for arch in ARCHITECTURES:
        for city in CITIES:
            header += f' {arch[:3]}-{CITY_ABBREV[city]:<3}'
        header += f' {arch[:3]}-Avg'
    header += '  Overall'
    lines.append(header)
    lines.append('  ' + '-' * 96)

    for method in METHOD_ORDER:
        row = f'  {method:<16}'
        all_vals = []
        for arch in ARCHITECTURES:
            arch_vals = []
            for city in CITIES:
                cell = results.get(arch, {}).get(city, {}).get(method)
                if cell and not np.isnan(cell['mean']):
                    row += f'  {cell["mean"]:6.2f}'
                    arch_vals.append(cell['mean'])
                    all_vals.append(cell['mean'])
                else:
                    row += f'  {"—":>6}'
            if arch_vals:
                row += f'  {np.mean(arch_vals):6.2f}'
            else:
                row += f'  {"—":>6}'
        if all_vals:
            row += f'  {np.mean(all_vals):6.2f}'
        else:
            row += f'  {"—":>6}'
        lines.append(row)
        if method == 'LOCO FADA':
            lines.append('  ' + '-' * 96)


def render_bootstrap_table(results: Dict, boots: Dict, lines: List[str]):
    """
    For each (arch, city) cell, show C4_clean vs each baseline:
    Δ mIoU, 95% CI, p-value, raw / cell-corrected / global-corrected sig.
    """
    lines.append('')
    lines.append('=' * 100)
    lines.append('  TABLE 2: BOOTSTRAP — C4_clean vs each baseline (per arch × city cell)')
    lines.append('  Δ = C4_clean − method   (positive = C4_clean wins)')
    lines.append('  Sig: raw=p<0.05  cell=Holm-Bonferroni within (arch,city)  glob=Holm-Bonferroni global')
    lines.append('  Stars on Δ: * p<.05  ** p<.01  *** p<.001  (raw p)')
    lines.append('=' * 100)

    other_methods = [m for m in METHOD_ORDER if m != 'C4_clean']

    for arch in ARCHITECTURES:
        for city in CITIES:
            lines.append('')
            cell_c4 = results.get(arch, {}).get(city, {}).get('C4_clean')
            c4_mean = (f'{cell_c4["mean"]:.2f}'
                       if cell_c4 and not np.isnan(cell_c4['mean'])
                       else '—')
            n = cell_c4['n'] if cell_c4 else 0
            lines.append(f'  --- {arch} / {city.upper()} '
                         f'(C4_clean tol mIoU = {c4_mean}, n={n}) ---')
            lines.append(f'    {"vs":<16} {"Δ mIoU":>10} {"95% CI":>20} '
                         f'{"p":>9}  {"raw":>3} {"cell":>4} {"glob":>4}')
            lines.append('    ' + '-' * 76)

            for method in other_methods:
                b = boots.get(arch, {}).get(city, {}).get(method)
                if b is None or b.get('n', 0) == 0 or np.isnan(b.get('delta', np.nan)):
                    lines.append(f'    {method:<16} {"—":>10} {"—":>20} '
                                 f'{"—":>9}  {"—":>3} {"—":>4} {"—":>4}')
                    continue
                stars = sig_stars(b['p_value'])
                ci = f'[{b["ci_lo"]:+.2f}, {b["ci_hi"]:+.2f}]'
                raw_sig = '*' if b['p_value'] < 0.05 else '·'
                cell_sig = '*' if b.get('cell_sig_corrected', False) else '·'
                glob_sig = '*' if b.get('global_sig_corrected', False) else '·'
                lines.append(
                    f'    {method:<16} '
                    f'{b["delta"]:+7.2f}{stars:<3} {ci:>20} '
                    f'{b["p_value"]:9.4f}  '
                    f'{raw_sig:>3} {cell_sig:>4} {glob_sig:>4}'
                )


def render_summary(results: Dict, boots: Dict, lines: List[str]):
    """Headline summary statistics."""
    lines.append('')
    lines.append('=' * 100)
    lines.append('  SUMMARY HEADLINES')
    lines.append('=' * 100)

    # Overall C4_clean mean tolerant mIoU
    all_c4 = []
    for arch in ARCHITECTURES:
        for city in CITIES:
            cell = results.get(arch, {}).get(city, {}).get('C4_clean')
            if cell and not np.isnan(cell['mean']):
                all_c4.append(cell['mean'])
    overall = np.mean(all_c4) if all_c4 else np.nan
    lines.append(f'  C4_clean overall tol mIoU (mean of {len(all_c4)} cells): '
                 f'{overall:.2f}')

    # Wins per baseline
    other_methods = [m for m in METHOD_ORDER if m != 'C4_clean']
    lines.append('')
    lines.append('  C4_clean win/tie/loss vs each baseline (mean Δ across cells):')
    for method in other_methods:
        deltas = []
        n_sig_raw = 0
        n_sig_cell = 0
        n_sig_glob = 0
        n_total = 0
        for arch in ARCHITECTURES:
            for city in CITIES:
                b = boots.get(arch, {}).get(city, {}).get(method)
                if b and not np.isnan(b.get('delta', np.nan)):
                    deltas.append(b['delta'])
                    n_total += 1
                    if b['p_value'] < 0.05:
                        n_sig_raw += 1
                    if b.get('cell_sig_corrected', False):
                        n_sig_cell += 1
                    if b.get('global_sig_corrected', False):
                        n_sig_glob += 1
        if deltas:
            mean_d = np.mean(deltas)
            n_wins = sum(1 for d in deltas if d > 0)
            lines.append(
                f'    vs {method:<16} mean Δ = {mean_d:+6.2f}  '
                f'wins {n_wins}/{n_total}  '
                f'sig (raw/cell/glob) = {n_sig_raw}/{n_sig_cell}/{n_sig_glob}'
            )


def write_latex_table(results: Dict, boots: Dict, output_path: str,
                      eval_label: str = 'tolerant mIoU'):
    """LaTeX table for the paper: rows = methods, cols = (arch, city) cells."""
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{')
    lines.append(r'    \textbf{Final comparison (' + eval_label + r').} ')
    lines.append(r'    Per-cell tolerant mIoU for every method on the 9 LOCO ')
    lines.append(r'    cells (3 architectures $\times$ 3 hold-out cities). ')
    lines.append(r'    \textbf{Bold} = best method in cell. Significance ')
    lines.append(r'    against C4\_clean: $^{*}$$p<.05$, $^{**}$$p<.01$, ')
    lines.append(r'    $^{***}$$p<.001$ (paired bootstrap, $B=10000$).')
    lines.append(r'  }')
    lines.append(r'  \label{tab:final_comparison}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{@{}l' + 'ccc|' * len(ARCHITECTURES) + r'c@{}}')
    lines.append(r'    \toprule')

    header_arch = r'    Method'
    for arch in ARCHITECTURES:
        header_arch += r' & \multicolumn{3}{c|}{' + arch + '}'
    header_arch += r' & Overall \\'
    lines.append(header_arch)

    cm_parts = []
    col = 2
    for _ in ARCHITECTURES:
        cm_parts.append(rf'\cmidrule(lr){{{col}-{col+2}}}')
        col += 3
    lines.append('    ' + ''.join(cm_parts))

    cities_row = r'    '
    for arch in ARCHITECTURES:
        for city in CITIES:
            cities_row += f' & {CITY_ABBREV[city]}'
    cities_row += r' & \\'
    lines.append(cities_row)
    lines.append(r'    \midrule')

    # Best per cell for bolding
    best_in_cell = {}
    for arch in ARCHITECTURES:
        for city in CITIES:
            best_method = None
            best_val = -np.inf
            for method in METHOD_ORDER:
                cell = results.get(arch, {}).get(city, {}).get(method)
                if cell and not np.isnan(cell['mean']) and cell['mean'] > best_val:
                    best_val = cell['mean']
                    best_method = method
            best_in_cell[(arch, city)] = best_method

    for method in METHOD_ORDER:
        parts = [f'    {method.replace("_", " ")}']
        all_vals = []
        for arch in ARCHITECTURES:
            for city in CITIES:
                cell = results.get(arch, {}).get(city, {}).get(method)
                if cell is None or np.isnan(cell['mean']):
                    parts.append('--')
                    continue
                val = cell['mean']
                all_vals.append(val)
                cell_str = f'{val:.1f}'

                # Significance star (vs C4_clean) — only if this is a baseline
                if method != 'C4_clean':
                    b = boots.get(arch, {}).get(city, {}).get(method)
                    if b and not np.isnan(b.get('p_value', np.nan)):
                        # If C4_clean is significantly BETTER than this method,
                        # the baseline gets a star (indicating significant diff).
                        s = sig_stars(b['p_value'])
                        if s:
                            cell_str = cell_str + f'$^{{{s}}}$'

                # Bold if best
                if best_in_cell.get((arch, city)) == method:
                    cell_str = r'\textbf{' + cell_str + '}'
                parts.append(cell_str)
        if all_vals:
            parts.append(f'{np.mean(all_vals):.1f}')
        else:
            parts.append('--')
        lines.append(' & '.join(parts) + r' \\')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table*}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ════════════════════════════════════════════════════════════════════════════
# JSON serialization
# ════════════════════════════════════════════════════════════════════════════

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def build_json_report(results: Dict, boots: Dict, args) -> Dict:
    # Strip per-image arrays from results before dumping (keeps file sane);
    # they're re-derivable from disk if needed and full per-image bootstrap
    # state lives in bootstrap_results.json instead.
    summary_results = {}
    for arch in ARCHITECTURES:
        summary_results[arch] = {}
        for city in CITIES:
            summary_results[arch][city] = {}
            for method in METHOD_ORDER:
                cell = results.get(arch, {}).get(city, {}).get(method, {})
                if not cell:
                    summary_results[arch][city][method] = None
                    continue
                summary_results[arch][city][method] = {
                    'mean':     cell.get('mean'),
                    'n':        cell.get('n'),
                    'pred_dir': cell.get('pred_dir'),
                }

    return {
        'generated':          datetime.now().isoformat(),
        'base_path':          args.base_path,
        'boundary_tolerance': args.boundary_tolerance,
        'img_size':           args.img_size,
        'n_bootstrap':        args.n_bootstrap,
        'alpha':              args.alpha,
        'architectures':      ARCHITECTURES,
        'cities':             CITIES,
        'methods':            METHOD_ORDER,
        'results':            _to_jsonable(summary_results),
        'bootstrap':          _to_jsonable(boots),
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Final comparison of C4_clean vs all baselines '
                    '(per arch × city), with paired bootstrap.')
    p.add_argument('--base_path', type=str,
                   default=os.environ["PROJECT_ROOT"],
                   help='Project base path containing data/ subdir '
                        '(defaults to the PROJECT_ROOT env var)')
    p.add_argument('--output_dir', type=str,
                   default='./final_comparison_output')
    p.add_argument('--boundary_tolerance', type=int, default=2)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--n_bootstrap', type=int, default=10000)
    p.add_argument('--alpha', type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 80)
    print('  FINAL COMPARISON: C4_clean vs all baselines')
    print(f'  base_path:           {args.base_path}')
    print(f'  output_dir:          {args.output_dir}')
    print(f'  boundary_tolerance:  ±{args.boundary_tolerance} px')
    print(f'  img_size:            {args.img_size}')
    print(f'  bootstrap B:         {args.n_bootstrap}')
    print(f'  alpha:               {args.alpha}')
    print('=' * 80)

    # ── Step 1: collect all per-image tolerant mIoU values ────────────────
    print('\n[1/4] Computing per-image tolerant mIoU for every cell...')
    results = collect_all_metrics(
        args.base_path, args.img_size, args.boundary_tolerance, verbose=True)

    # ── Step 2: paired bootstrap C4_clean vs each baseline ────────────────
    print('\n[2/4] Running paired bootstrap (C4_clean vs each baseline)...')
    boots = run_bootstrap_comparisons(
        results, n_bootstrap=args.n_bootstrap, alpha=args.alpha)

    # ── Step 3: render text report ─────────────────────────────────────────
    print('\n[3/4] Rendering reports...')
    text_lines = []
    text_lines.append('FINAL COMPARISON REPORT')
    text_lines.append(f'Generated: {datetime.now().isoformat()}')
    text_lines.append(f'Base path: {args.base_path}')
    text_lines.append(f'Eval: tolerant mIoU (±{args.boundary_tolerance} px)')
    text_lines.append(f'Bootstrap: B={args.n_bootstrap}, α={args.alpha}')
    render_main_table(results, text_lines)
    render_bootstrap_table(results, boots, text_lines)
    render_summary(results, boots, text_lines)

    text_path = os.path.join(args.output_dir, 'final_comparison_report.txt')
    with open(text_path, 'w') as f:
        f.write('\n'.join(text_lines))
    # Also echo to stdout
    print('\n'.join(text_lines))
    print(f'\n  Text report saved → {text_path}')

    # ── Step 4: LaTeX + JSON dumps ────────────────────────────────────────
    print('\n[4/4] Saving LaTeX + JSON outputs...')

    latex_path = os.path.join(args.output_dir, 'final_comparison_table.tex')
    write_latex_table(results, boots, latex_path)
    print(f'  LaTeX table saved → {latex_path}')

    json_report = build_json_report(results, boots, args)
    json_path = os.path.join(args.output_dir, 'final_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f'  JSON report saved → {json_path}')

    # Detailed bootstrap dump (with per-image stems for full reproducibility)
    bootstrap_dump = {}
    for arch in ARCHITECTURES:
        bootstrap_dump[arch] = {}
        for city in CITIES:
            bootstrap_dump[arch][city] = {
                method: _to_jsonable(boots.get(arch, {}).get(city, {}).get(method, {}))
                for method in METHOD_ORDER if method != 'C4_clean'
            }
    boot_path = os.path.join(args.output_dir, 'bootstrap_results.json')
    with open(boot_path, 'w') as f:
        json.dump(bootstrap_dump, f, indent=2, default=str)
    print(f'  Bootstrap dump → {boot_path}')

    print(f'\n  Done. All outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()