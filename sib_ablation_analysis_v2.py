#!/usr/bin/env python3
"""
SIB Component Ablation Analysis V2 — Recomputed From Scratch

Same downstream analysis as the original sib_ablation_analysis.py, but
EVERY tolerant mIoU value is recomputed from per-image prediction PNGs vs
GT masks. No comparison_results.json shortcuts.

Setup
-----
For each architecture × city × ablation cell, we look for prediction
images at:

  {arch_root}/<dir matching ablation pattern + holdout city>/predictions/

For C4 baseline (the reference for ablations) and C4_clean we look for:
  - C4:        the original full-SIB experiment (M1 for MAMNet/OGLANet,
               D1 for DINOv3) — Haar + VIB + Aug + AB [+ SAG + aFDA + Ctr]
  - C4_clean:  Haar + VIB only (the new "final" model, included for
               completeness)

For each baseline (Upper Bound, LOCO Vanilla, LOCO FDA, …, LOCO FADA):
  $PROJECT_ROOT/data/Test_img_results/{loco|upper}/
  {city}/highres/{arch_lower}/{method_subdir}/

GT masks: {base}/data/Final_data_test/{city}/highres/test/masks/

Outputs (in --output_dir)
-------------------------
  ablation_v2_main_table.txt        — per-cell mIoU (Table 4 in paper)
  ablation_v2_deltas.txt            — Δ vs C4 with bootstrap CIs
  ablation_v2_recovery.txt          — recovery ratios
  ablation_v2_summary.txt           — headline stats
  ablation_v2_predictions.txt       — §5.3 prediction validation
  ablation_v2_table4.tex            — LaTeX-ready Table 4
  ablation_v2_report.json           — full machine-readable dump

Usage
-----
    python sib_ablation_analysis_v2.py \\
        --base_path $PROJECT_ROOT \\
        --output_dir ./ablation_analysis_v2 \\
        --boundary_tolerance 2 \\
        --img_size 384 \\
        --n_bootstrap 10000 \\
        --alpha 0.05
"""

import os
import re
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
FOLD_MAP = {0: 'phoenix', 1: 'miami', 2: 'chicago'}

# Canonical ablation IDs and their diagnostic mapping
ABLATION_META = OrderedDict([
    ('C4',       {'name': 'SIB-Full (C4)',         'diagnostic': '-',        'critical': True}),
    ('C4_clean', {'name': 'C4_clean (Haar+VIB)',   'diagnostic': '-',        'critical': True}),
    ('A1',       {'name': 'No VIB on F_LL',        'diagnostic': 'D2',       'critical': True}),
    ('A2',       {'name': 'Uniform-β VIB',         'diagnostic': 'D1',       'critical': True}),
    ('A3',       {'name': 'Symmetric VIB',         'diagnostic': 'D1+D2',    'critical': True}),
    ('A4',       {'name': 'No content aug',         'diagnostic': 'D3',       'critical': True}),
    ('A5',       {'name': 'No cross-city mix',      'diagnostic': 'D3v',      'critical': False}),
    ('A6',       {'name': 'Aug all subbands',       'diagnostic': 'D1+D3',    'critical': True}),
    ('A7',       {'name': 'No SAG',                 'diagnostic': 'D2',       'critical': False}),
    ('A8',       {'name': 'No FDA preproc',         'diagnostic': 'confound', 'critical': False}),
    ('A9',       {'name': 'No Haar (uniform VIB)',  'diagnostic': 'D2',       'critical': False}),
    ('A10',      {'name': 'VIB wrong subband (HL)', 'diagnostic': 'D2inv',    'critical': True}),
])

# Architecture-specific directory-name patterns mapping to ablation IDs.
# Order matters: the FIRST matching pattern wins. Place more specific
# patterns earlier.
MAMNET_PATTERNS = OrderedDict([
    ('C4_clean', [r'mamnet_sib_C4clean[_\b]']),
    ('C4',       [r'mamnet_sib_M1[_\b]', r'mamnet_sib_C4[_\b]']),
    ('A1',       [r'A1_no_vib']),
    ('A2',       [r'A2_uniform_beta']),
    ('A3',       [r'A3_symmetric_vib']),
    ('A4',       [r'A4_no_content_aug']),
    ('A5',       [r'A5_aug_all_subbands']),  # MAMNet's A5 doubles as MRFP+ analog
    ('A6',       [r'A6_no_sag']),
    ('A7',       [r'A7_no_fda_preproc']),
    ('A8',       [r'A8_no_haar']),
    ('A9',       [r'A9_no_edge_vib']),
    ('A10',      [r'A10_vib_wrong_subband']),
])

OGLANET_PATTERNS = OrderedDict([
    ('C4_clean', [r'oglanet_sib_C4clean[_\b]']),
    ('C4',       [r'oglanet_sib_M1[_\b]', r'oglanet_sib_C4[_\b]']),
    ('A1',       [r'A1_no_vib_ll', r'A1[_\b]']),
    ('A2',       [r'A2_uniform_beta', r'A2[_\b]']),
    ('A3',       [r'A3_symmetric_vib', r'A3[_\b]']),
    ('A4',       [r'A4_no_aug', r'A4[_\b]']),
    ('A5',       [r'A5_aug_no_mix', r'A5[_\b]']),
    ('A6',       [r'A6_aug_all', r'A6[_\b]']),
    ('A7',       [r'A7_no_sag', r'A7[_\b]']),
    ('A8',       [r'A8_no_preproc', r'A8[_\b]']),
    ('A9',       [r'A9_no_haar', r'A9[_\b]']),
    ('A10',      [r'A10_vib_hl_only', r'A10[_\b]']),
])

DINOV3_PATTERNS = OrderedDict([
    ('C4_clean', [r'dinov3_sib_C4clean[_\b]']),
    ('C4',       [r'dinov3_sib_D1[_\b]', r'dinov3_sib_C4[_\b]']),
    ('A1',       [r'A1_noConVIB', r'A1[_\b]']),
    ('A2',       [r'A2_fixedBeta', r'A2[_\b]']),
    ('A3',       [r'A3_symVIB', r'A3[_\b]']),
    ('A4',       [r'A4_noAug', r'A4[_\b]']),
    ('A5',       [r'A5_noMix', r'A5[_\b]']),
    ('A6',       [r'A6_augAll', r'A6[_\b]']),
    # A7, A8 are N/A for DINOv3 — no jobs submitted for them.
    ('A9',       [r'A9_noHaar', r'A9[_\b]']),
    ('A10',      [r'A10_vibHL', r'A10[_\b]']),
])

ARCH_PATTERNS = {
    'MAMNet':  MAMNET_PATTERNS,
    'OGLANet': OGLANET_PATTERNS,
    'DINOv3':  DINOV3_PATTERNS,
}

BASELINE_LABELS = [
    'Upper Bound', 'LOCO Vanilla', 'LOCO FDA', 'LOCO SegDesic',
    'LOCO IIM', 'LOCO ISW', 'LOCO MRFP+', 'LOCO FADA',
]

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
# Tolerant mIoU
# ════════════════════════════════════════════════════════════════════════════

_TOLERANCE_KERNEL_CACHE = {}


def _get_tolerance_kernel(tolerance):
    if tolerance not in _TOLERANCE_KERNEL_CACHE:
        _TOLERANCE_KERNEL_CACHE[tolerance] = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1))
    return _TOLERANCE_KERNEL_CACHE[tolerance]


def compute_tolerant_miou(pred, gt, tolerance=2):
    """Tolerant mIoU for one (pred, gt) pair, returned as percent."""
    kernel  = _get_tolerance_kernel(tolerance)
    gt_u8   = gt.astype(np.uint8)
    eroded  = cv2.erode(gt_u8, kernel)
    dilated = cv2.dilate(gt_u8, kernel)
    valid   = ~((dilated - eroded) > 0)

    p = pred[valid]
    g = gt[valid]
    tp = np.logical_and(p == 1, g == 1).sum()
    fp = np.logical_and(p == 1, g == 0).sum()
    tn = np.logical_and(p == 0, g == 0).sum()
    fn = np.logical_and(p == 0, g == 1).sum()

    shadow_iou    = tp / (tp + fp + fn + 1e-10)
    nonshadow_iou = tn / (tn + fp + fn + 1e-10)
    miou          = (shadow_iou + nonshadow_iou) / 2
    return float(miou * 100)


# ════════════════════════════════════════════════════════════════════════════
# Directory + image discovery
# ════════════════════════════════════════════════════════════════════════════

def _stem_map(directory):
    m = {}
    if not os.path.isdir(directory):
        return m
    for fn in os.listdir(directory):
        ext = os.path.splitext(fn)[1].lower()
        if ext in IMG_EXTS:
            m[os.path.splitext(fn)[0]] = os.path.join(directory, fn)
    return m


def find_gt_dir(base_path: str, city: str, res: str = 'highres') -> Optional[str]:
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
    if method_label not in METHOD_TO_SUBDIR:
        return None
    split, subdir = METHOD_TO_SUBDIR[method_label]
    arch_lower = ARCH_LOWER[arch]
    pred_dir = os.path.join(
        base_path, 'data', 'Test_img_results', split, city, res, arch_lower, subdir)
    if os.path.isdir(pred_dir) and len(_stem_map(pred_dir)) > 0:
        return pred_dir
    return None


def identify_ablation(dirname: str, arch: str) -> Optional[str]:
    """Match a directory name to the canonical ablation ID."""
    patterns = ARCH_PATTERNS.get(arch, {})
    for abl_id, pats in patterns.items():
        for pat in pats:
            if re.search(pat, dirname, re.IGNORECASE):
                return abl_id
    return None


def identify_holdout_city(dirname: str) -> Optional[str]:
    """Extract the held-out city from a directory name."""
    dn = dirname.lower()
    for city in CITIES:
        if city in dn:
            return city
    return None


def scan_arch_for_ablations(base_path: str, arch: str) -> Dict[str, Dict[str, str]]:
    """
    Scan an architecture's outputs/ tree.

    Returns nested dict:
        result[abl_id][city] = path-to-predictions-dir

    If multiple matching dirs exist for the same (abl_id, city), pick the
    one whose `predictions/` has the most images (most likely the
    completed run); on tie, prefer most-recently modified.
    """
    arch_root = os.path.join(base_path, 'data', ARCH_LOWER[arch], 'outputs')
    found = defaultdict(dict)

    if not os.path.isdir(arch_root):
        print(f'  WARNING: {arch} outputs root not found: {arch_root}')
        return found

    candidates_per_cell = defaultdict(list)
    for entry in sorted(os.listdir(arch_root)):
        full_path = os.path.join(arch_root, entry)
        if not os.path.isdir(full_path):
            continue
        abl_id = identify_ablation(entry, arch)
        city = identify_holdout_city(entry)
        if abl_id is None or city is None:
            continue
        pred_dir = os.path.join(full_path, 'predictions')
        n_imgs = len(_stem_map(pred_dir))
        if n_imgs == 0:
            continue
        try:
            mtime = os.path.getmtime(pred_dir)
        except OSError:
            mtime = 0
        candidates_per_cell[(abl_id, city)].append(
            (pred_dir, n_imgs, mtime, entry))

    # Resolve duplicates
    for (abl_id, city), cands in candidates_per_cell.items():
        cands.sort(key=lambda x: (x[1], x[2]), reverse=True)
        pred_dir, n_imgs, _, entry = cands[0]
        found[abl_id][city] = pred_dir

    return found


# ════════════════════════════════════════════════════════════════════════════
# Per-image computation per cell
# ════════════════════════════════════════════════════════════════════════════

def per_image_tolerant(pred_dir: str, gt_dir: str, img_size: int,
                       tolerance: int) -> Tuple[List[float], List[str]]:
    pred_map = _stem_map(pred_dir)
    gt_map = _stem_map(gt_dir)
    if not pred_map or not gt_map:
        return [], []

    pairs = []
    for stem, pred_path in sorted(pred_map.items()):
        if stem in gt_map:
            pairs.append((stem, pred_path, gt_map[stem]))

    ious, stems = [], []
    sz = (img_size, img_size)

    for stem, pred_path, gt_path in pairs:
        p = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)
        if p is None or g is None:
            continue
        if p.shape != (img_size, img_size):
            p = cv2.resize(p, sz, interpolation=cv2.INTER_NEAREST)
        if g.shape != (img_size, img_size):
            g = cv2.resize(g, sz, interpolation=cv2.INTER_NEAREST)
        pb = (p > 127).astype(np.uint8)
        gb = (g > 127).astype(np.uint8)
        ious.append(compute_tolerant_miou(pb, gb, tolerance=tolerance))
        stems.append(stem)
    return ious, stems


def collect_ablation_metrics(base_path: str, img_size: int, tolerance: int,
                              verbose: bool = True) -> Dict:
    """
    Collect tolerant mIoU per (arch, abl_id, city).

    Returns:
        results[arch][abl_id][city] = {
            'ious':     np.ndarray of per-image tolerant mIoU,
            'stems':    list of image stems,
            'mean':     float mean tolerant mIoU,
            'n':        number of images,
            'pred_dir': source directory
        }
    """
    results = defaultdict(lambda: defaultdict(dict))

    # Pre-resolve GT dirs
    gt_dirs = {}
    for city in CITIES:
        gt = find_gt_dir(base_path, city)
        if gt is None:
            print(f'  WARNING: GT dir not found for {city}; skipping')
        gt_dirs[city] = gt

    for arch in ARCHITECTURES:
        if verbose:
            print(f'\n  === {arch} ===')
        arch_index = scan_arch_for_ablations(base_path, arch)
        for abl_id in ABLATION_META:
            for city in CITIES:
                gt = gt_dirs.get(city)
                if gt is None:
                    results[arch][abl_id][city] = {
                        'ious': np.array([]), 'stems': [],
                        'mean': np.nan, 'n': 0, 'pred_dir': None}
                    continue

                pred_dir = arch_index.get(abl_id, {}).get(city)
                if pred_dir is None:
                    if verbose:
                        print(f'    {abl_id:<10} {city:<8} ✗  no preds found')
                    results[arch][abl_id][city] = {
                        'ious': np.array([]), 'stems': [],
                        'mean': np.nan, 'n': 0, 'pred_dir': None}
                    continue

                ious, stems = per_image_tolerant(
                    pred_dir, gt, img_size, tolerance)
                if not ious:
                    if verbose:
                        print(f'    {abl_id:<10} {city:<8} ✗  no matched pairs')
                    results[arch][abl_id][city] = {
                        'ious': np.array([]), 'stems': [],
                        'mean': np.nan, 'n': 0, 'pred_dir': pred_dir}
                    continue

                arr = np.array(ious)
                results[arch][abl_id][city] = {
                    'ious': arr, 'stems': stems,
                    'mean': float(arr.mean()),
                    'n': len(ious), 'pred_dir': pred_dir}
                if verbose:
                    print(f'    {abl_id:<10} {city:<8} ✓  '
                          f'n={len(ious):3d}  tol mIoU={arr.mean():6.2f}')

    return {arch: dict(d) for arch, d in results.items()}


def collect_baseline_metrics(base_path: str, img_size: int, tolerance: int,
                              verbose: bool = True) -> Dict:
    """
    Collect Upper Bound, LOCO Vanilla, … LOCO FADA tolerant mIoU per
    (arch, baseline, city). Same per-image pairing strategy.
    """
    results = defaultdict(lambda: defaultdict(dict))

    gt_dirs = {city: find_gt_dir(base_path, city) for city in CITIES}

    for arch in ARCHITECTURES:
        if verbose:
            print(f'\n  --- baselines for {arch} ---')
        for bl in BASELINE_LABELS:
            for city in CITIES:
                gt = gt_dirs.get(city)
                if gt is None:
                    results[arch][bl][city] = {
                        'ious': np.array([]), 'stems': [],
                        'mean': np.nan, 'n': 0, 'pred_dir': None}
                    continue
                pred_dir = find_baseline_pred_dir(base_path, arch, city, bl)
                if pred_dir is None:
                    results[arch][bl][city] = {
                        'ious': np.array([]), 'stems': [],
                        'mean': np.nan, 'n': 0, 'pred_dir': None}
                    if verbose:
                        print(f'    {bl:<14} {city:<8} ✗')
                    continue
                ious, stems = per_image_tolerant(
                    pred_dir, gt, img_size, tolerance)
                if not ious:
                    results[arch][bl][city] = {
                        'ious': np.array([]), 'stems': [],
                        'mean': np.nan, 'n': 0, 'pred_dir': pred_dir}
                    if verbose:
                        print(f'    {bl:<14} {city:<8} ✗ no pairs')
                    continue
                arr = np.array(ious)
                results[arch][bl][city] = {
                    'ious': arr, 'stems': stems,
                    'mean': float(arr.mean()),
                    'n': len(ious), 'pred_dir': pred_dir}
                if verbose:
                    print(f'    {bl:<14} {city:<8} ✓  '
                          f'n={len(ious):3d}  tol mIoU={arr.mean():6.2f}')

    return {arch: dict(d) for arch, d in results.items()}


# ════════════════════════════════════════════════════════════════════════════
# Master tables (mean across cities for cross-cell deltas)
# ════════════════════════════════════════════════════════════════════════════

def build_master_table(results: Dict) -> Dict:
    """
    table[arch][abl_id] = {
        'chicago': mean tolerant mIoU,
        'miami':   …,
        'phoenix': …,
        'mean':    average across the 3 cities,
        'values':  np.ndarray of [chicago, miami, phoenix] means (NaN if missing),
        'n_valid': number of cities with valid data,
    }
    """
    table = {}
    for arch in ARCHITECTURES:
        table[arch] = {}
        arch_results = results.get(arch, {})
        for abl_id in ABLATION_META:
            cells = arch_results.get(abl_id, {})
            entry = {}
            vals = []
            for city in CITIES:
                m = cells.get(city, {}).get('mean', np.nan)
                entry[city] = m
                vals.append(m)
            valid = [v for v in vals if not np.isnan(v)]
            entry['mean'] = float(np.mean(valid)) if valid else np.nan
            entry['values'] = np.array(vals)
            entry['n_valid'] = len(valid)
            table[arch][abl_id] = entry
    return table


def build_baseline_table(baseline_results: Dict) -> Dict:
    """baselines[arch][label] = {city: mean, 'mean': overall mean}."""
    out = {}
    for arch in ARCHITECTURES:
        out[arch] = {}
        for bl in BASELINE_LABELS:
            entry = {}
            vals = []
            for city in CITIES:
                m = baseline_results.get(arch, {}).get(bl, {}).get(city, {}).get('mean', np.nan)
                entry[city] = m
                vals.append(m)
            valid = [v for v in vals if not np.isnan(v)]
            entry['mean'] = float(np.mean(valid)) if valid else np.nan
            out[arch][bl] = entry
    return out


# ════════════════════════════════════════════════════════════════════════════
# Bootstrap deltas (per-image, paired, vs C4)
# ════════════════════════════════════════════════════════════════════════════

def align_pair(a: Dict, b: Dict) -> Tuple[np.ndarray, np.ndarray]:
    if a['n'] == 0 or b['n'] == 0:
        return np.array([]), np.array([])
    map_b = {s: i for i, s in enumerate(b['stems'])}
    aa, bb = [], []
    for stem, va in zip(a['stems'], a['ious']):
        if stem in map_b:
            aa.append(va)
            bb.append(b['ious'][map_b[stem]])
    return np.array(aa), np.array(bb)


def paired_bootstrap(vals_a: np.ndarray, vals_b: np.ndarray,
                     n_bootstrap: int, seed: int = 42) -> Dict:
    """Returns {'delta', 'ci_lo', 'ci_hi', 'p_value', 'n'}."""
    rng = np.random.RandomState(seed)
    n = min(len(vals_a), len(vals_b))
    if n == 0:
        return {'delta': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                'p_value': np.nan, 'n': 0}
    diff = vals_a[:n] - vals_b[:n]
    obs = float(np.mean(diff))
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boots[i] = np.mean(diff[idx])
    if obs >= 0:
        p = 2 * max(np.mean(boots <= 0), 1.0 / n_bootstrap)
    else:
        p = 2 * max(np.mean(boots >= 0), 1.0 / n_bootstrap)
    return {
        'delta': obs,
        'ci_lo': float(np.percentile(boots, 2.5)),
        'ci_hi': float(np.percentile(boots, 97.5)),
        'p_value': float(min(p, 1.0)),
        'n': int(n),
    }


def cohens_d_paired(vals_a: np.ndarray, vals_b: np.ndarray) -> float:
    if len(vals_a) < 2 or len(vals_b) < 2:
        return np.nan
    n = min(len(vals_a), len(vals_b))
    diff = vals_a[:n] - vals_b[:n]
    sd = np.std(diff, ddof=1)
    if sd < 1e-10:
        return np.nan
    return float(np.mean(diff) / sd)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = alpha / (n - rank)
        if p <= adj:
            reject[orig_idx] = True
        else:
            break
    return reject


def sig_stars(p_val: float) -> str:
    if np.isnan(p_val):
        return ''
    if p_val < 0.001: return '***'
    if p_val < 0.01:  return '**'
    if p_val < 0.05:  return '*'
    return ''


def compute_deltas_vs_c4(results: Dict, n_bootstrap: int,
                         alpha: float = 0.05) -> Dict:
    """
    Per-image paired bootstrap (pooled across the 3 cities of each arch)
    of every ablation vs C4 (the "full SIB" reference).

    The delta is computed pixel/image-paired. We pool across cities by
    concatenating the per-image scores from each city, after pairing
    within each city. This gives the largest sample size (one fold per
    city × ~30 images / fold ≈ 90 pairs).

    Returns:
        deltas[arch][abl_id] = {
            'delta_mean': float Δ in tol mIoU vs C4 (pooled),
            'bootstrap': {...},
            'cohens_d': float,
            'n_folds': int (cities used),
            'significant_raw': bool,
            'significant_corrected': bool (Holm-Bonferroni global),
        }
    """
    deltas = {}
    keys, pvals = [], []

    for arch in ARCHITECTURES:
        deltas[arch] = {}
        c4_per_city = results.get(arch, {}).get('C4', {})

        for abl_id in ABLATION_META:
            if abl_id == 'C4':
                continue

            abl_per_city = results.get(arch, {}).get(abl_id, {})

            pooled_c4, pooled_abl = [], []
            n_cities_used = 0
            for city in CITIES:
                c4_cell = c4_per_city.get(city)
                abl_cell = abl_per_city.get(city)
                if (c4_cell is None or abl_cell is None
                        or c4_cell['n'] == 0 or abl_cell['n'] == 0):
                    continue
                c4_a, abl_a = align_pair(c4_cell, abl_cell)
                if len(c4_a) == 0:
                    continue
                pooled_c4.extend(c4_a.tolist())
                pooled_abl.extend(abl_a.tolist())
                n_cities_used += 1

            if not pooled_c4:
                deltas[arch][abl_id] = None
                continue

            pa = np.array(pooled_abl)
            pc = np.array(pooled_c4)
            boot = paired_bootstrap(pa, pc, n_bootstrap=n_bootstrap)
            d = cohens_d_paired(pa, pc)
            deltas[arch][abl_id] = {
                'delta_mean': boot['delta'],
                'bootstrap': boot,
                'cohens_d': d,
                'n_folds': n_cities_used,
                'n_pairs': boot['n'],
            }
            keys.append((arch, abl_id))
            pvals.append(boot['p_value'])

    if pvals:
        rejects = holm_bonferroni(pvals, alpha=alpha)
        for i, (arch, abl_id) in enumerate(keys):
            d = deltas[arch][abl_id]
            d['significant_raw'] = bool(pvals[i] < alpha)
            d['significant_corrected'] = bool(rejects[i])

    return deltas


# ════════════════════════════════════════════════════════════════════════════
# Recovery ratios
# ════════════════════════════════════════════════════════════════════════════

def compute_recovery(table: Dict, baselines: Dict) -> Dict:
    recovery = {}
    for arch in ARCHITECTURES:
        recovery[arch] = {}
        ub = baselines.get(arch, {}).get('Upper Bound', {})
        lv = baselines.get(arch, {}).get('LOCO Vanilla', {})
        for abl_id in ABLATION_META:
            entry = table[arch].get(abl_id)
            if entry is None:
                continue
            R_per_city = {}
            R_vals = []
            for city in CITIES:
                ub_v = ub.get(city, np.nan)
                lv_v = lv.get(city, np.nan)
                a_v = entry.get(city, np.nan)
                if np.isnan(ub_v) or np.isnan(lv_v) or np.isnan(a_v):
                    R_per_city[city] = np.nan
                    continue
                gap = ub_v - lv_v
                if abs(gap) < 0.01:
                    R_per_city[city] = np.nan
                    continue
                R = (a_v - lv_v) / gap
                R_per_city[city] = float(R)
                R_vals.append(R)
            recovery[arch][abl_id] = {
                **R_per_city,
                'mean': float(np.mean(R_vals)) if R_vals else np.nan,
                'min':  float(np.min(R_vals))  if R_vals else np.nan,
                'max':  float(np.max(R_vals))  if R_vals else np.nan,
            }
    return recovery


# ════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ════════════════════════════════════════════════════════════════════════════

def compute_summary(table: Dict, baselines: Dict, deltas: Dict) -> Dict:
    summary = {}

    def _mean_overall(abl_id):
        vals = []
        for arch in ARCHITECTURES:
            e = table[arch].get(abl_id)
            if e:
                vals.extend([v for v in e['values'] if not np.isnan(v)])
        return float(np.mean(vals)) if vals else np.nan

    summary['c4_overall_miou']       = _mean_overall('C4')
    summary['c4_clean_overall_miou'] = _mean_overall('C4_clean')

    # Gap closure for C4 and C4_clean
    closure = {}
    for ref_id in ['C4', 'C4_clean']:
        per_arch = {}
        for arch in ARCHITECTURES:
            ub = baselines.get(arch, {}).get('Upper Bound', {})
            lv = baselines.get(arch, {}).get('LOCO Vanilla', {})
            ref = table[arch].get(ref_id)
            if ref is None:
                continue
            R_vals = []
            for city in CITIES:
                u, l, r = ub.get(city, np.nan), lv.get(city, np.nan), ref.get(city, np.nan)
                if np.isnan(u) or np.isnan(l) or np.isnan(r) or abs(u - l) < 0.01:
                    continue
                R_vals.append((r - l) / (u - l))
            if R_vals:
                per_arch[arch] = {
                    'mean_R': float(np.mean(R_vals)),
                    'min_R':  float(np.min(R_vals)),
                    'max_R':  float(np.max(R_vals)),
                }
        closure[ref_id] = per_arch
    summary['gap_closure'] = closure

    # Worst-case Δ vs vanilla for C4 and C4_clean
    for ref_id in ['C4', 'C4_clean']:
        worst = np.nan
        for arch in ARCHITECTURES:
            lv = baselines.get(arch, {}).get('LOCO Vanilla', {})
            ref = table[arch].get(ref_id)
            if ref is None:
                continue
            for city in CITIES:
                lv_v, r_v = lv.get(city, np.nan), ref.get(city, np.nan)
                d = r_v - lv_v
                if not np.isnan(d):
                    worst = min(worst, d) if not np.isnan(worst) else d
        summary[f'{ref_id}_worst_vs_vanilla'] = (
            float(worst) if not np.isnan(worst) else None)

    # Significance counts
    n_sig, n_total = 0, 0
    for arch in ARCHITECTURES:
        for abl_id in ABLATION_META:
            if abl_id in ('C4',):
                continue
            d = deltas.get(arch, {}).get(abl_id)
            if d is None:
                continue
            n_total += 1
            if d.get('significant_corrected') and d['delta_mean'] < 0:
                n_sig += 1
    summary['n_significant_drops']   = n_sig
    summary['n_total_comparisons']   = n_total

    # Largest drop per architecture
    largest = {}
    for arch in ARCHITECTURES:
        worst_abl, worst_delta = None, 0.0
        for abl_id in ABLATION_META:
            if abl_id in ('C4', 'C4_clean'):
                continue
            d = deltas.get(arch, {}).get(abl_id)
            if d and d['delta_mean'] < worst_delta:
                worst_abl = abl_id
                worst_delta = d['delta_mean']
        largest[arch] = {
            'ablation': worst_abl,
            'delta': worst_delta,
            'name': ABLATION_META[worst_abl]['name'] if worst_abl else 'N/A',
        }
    summary['largest_drops'] = largest

    # Per-diagnostic validation
    diag = {}
    for code in ['D1', 'D2', 'D3', 'D1+D2', 'D1+D3', 'D2inv']:
        abls = [a for a, m in ABLATION_META.items()
                if m['diagnostic'] == code and a not in ('C4', 'C4_clean')]
        if not abls:
            continue
        all_d = []
        for arch in ARCHITECTURES:
            for a in abls:
                d = deltas.get(arch, {}).get(a)
                if d and not np.isnan(d['delta_mean']):
                    all_d.append(d['delta_mean'])
        diag[code] = {
            'ablations': abls,
            'mean_delta': float(np.mean(all_d)) if all_d else np.nan,
            'confirms_diagnostic': bool(np.mean(all_d) < 0) if all_d else None,
        }
    summary['diagnostic_validation'] = diag

    return summary


# ════════════════════════════════════════════════════════════════════════════
# Prediction validation
# ════════════════════════════════════════════════════════════════════════════

def validate_predictions(deltas: Dict, table: Dict) -> List[str]:
    out = []

    def _delta(arch, abl):
        d = deltas.get(arch, {}).get(abl)
        return d['delta_mean'] if d else np.nan

    # P1: A1 drops OGLANet > DINOv3
    d_o, d_d = _delta('OGLANet', 'A1'), _delta('DINOv3', 'A1')
    if not np.isnan(d_o) and not np.isnan(d_d):
        passed = d_o < d_d
        out.append(f'  P1 (A1: OGLANet drop > DINOv3 drop): '
                   f'OGLANet Δ={d_o:+.2f}  DINOv3 Δ={d_d:+.2f}  '
                   f'{"✓ PASS" if passed else "✗ FAIL"}')

    # P2: A4 drops DINOv3 > OGLANet
    d_d4, d_o4 = _delta('DINOv3', 'A4'), _delta('OGLANet', 'A4')
    if not np.isnan(d_d4) and not np.isnan(d_o4):
        passed = d_d4 < d_o4
        out.append(f'  P2 (A4: DINOv3 drop > OGLANet drop): '
                   f'DINOv3 Δ={d_d4:+.2f}  OGLANet Δ={d_o4:+.2f}  '
                   f'{"✓ PASS" if passed else "✗ FAIL"}')

    # P3: A10 worse than C4 everywhere
    a10_neg = True
    for arch in ARCHITECTURES:
        d = _delta(arch, 'A10')
        if not np.isnan(d) and d >= 0:
            a10_neg = False
    out.append(f'  P3 (A10: VIB on wrong subband degrades all archs): '
               f'{"✓ PASS" if a10_neg else "✗ FAIL (some non-negative)"}')

    # P4: A3 collapses thin-shadow (avg Δ < 0)
    a3 = [_delta(a, 'A3') for a in ARCHITECTURES]
    a3 = [x for x in a3 if not np.isnan(x)]
    if a3:
        avg = np.mean(a3)
        out.append(f'  P4 (A3: Symmetric VIB hurts boundaries): '
                   f'avg Δ={avg:+.2f}  '
                   f'{"✓ PASS" if avg < 0 else "✗ FAIL"}')

    # P5: A6 reproduces MRFP+ pattern (OGLANet-Miami collapse)
    a6 = table.get('OGLANet', {}).get('A6')
    c4 = table.get('OGLANet', {}).get('C4')
    if a6 and c4:
        m_a = a6.get('miami', np.nan)
        m_c = c4.get('miami', np.nan)
        if not np.isnan(m_a) and not np.isnan(m_c):
            drop = m_a - m_c
            cat = drop < -10
            out.append(f'  P5 (A6: OGLANet-Miami collapse like MRFP+): '
                       f'Δ={drop:+.2f}  '
                       f'{"✓ PASS (catastrophic)" if cat else "? PARTIAL/FAIL"}')

    return out


# ════════════════════════════════════════════════════════════════════════════
# Rendering
# ════════════════════════════════════════════════════════════════════════════

def render_main_table(table: Dict, deltas: Dict, lines: List[str]):
    lines.append('')
    lines.append('=' * 110)
    lines.append('  MAIN TABLE: Per-cell tolerant mIoU (C4 + C4_clean + ablations)')
    lines.append('=' * 110)

    header = f'  {"ID":<10} {"Name":<26}'
    for arch in ARCHITECTURES:
        for city in CITIES:
            header += f' {CITY_ABBREV[city]:>5}'
        header += f' {"Avg":>6}'
    header += f'  {"Δ(Avg)":>7} {"p":>7} {"d":>6}'
    lines.append(header)
    lines.append('  ' + '-' * 106)

    for abl_id, meta in ABLATION_META.items():
        row = f'  {abl_id:<10} {meta["name"]:<26}'
        for arch in ARCHITECTURES:
            entry = table[arch].get(abl_id)
            for city in CITIES:
                if entry and not np.isnan(entry.get(city, np.nan)):
                    row += f' {entry[city]:5.1f}'
                else:
                    row += f'   {"—":>3}'
            if entry and not np.isnan(entry.get('mean', np.nan)):
                row += f' {entry["mean"]:6.2f}'
            else:
                row += f'    {"—":>3}'

        if abl_id in ('C4', 'C4_clean'):
            row += f'  {"—":>7} {"—":>7} {"—":>6}'
        else:
            arch_d, arch_p, arch_cd = [], [], []
            for arch in ARCHITECTURES:
                d = deltas.get(arch, {}).get(abl_id)
                if d:
                    arch_d.append(d['delta_mean'])
                    arch_p.append(d['bootstrap']['p_value'])
                    arch_cd.append(d['cohens_d'])
            if arch_d:
                avg_d = np.mean(arch_d)
                min_p = min(arch_p)
                avg_cd = np.mean([c for c in arch_cd if not np.isnan(c)]) if any(not np.isnan(c) for c in arch_cd) else np.nan
                stars = sig_stars(min_p)
                row += f'  {avg_d:+6.2f}{stars:1s}'
                row += f' {min_p:7.4f}'
                row += f' {avg_cd:6.2f}' if not np.isnan(avg_cd) else f'    {"—":>3}'
            else:
                row += f'  {"—":>7} {"—":>7} {"—":>6}'
        lines.append(row)

        if abl_id in ('C4', 'C4_clean'):
            lines.append('  ' + '-' * 106)


def render_per_arch_deltas(deltas: Dict, lines: List[str]):
    lines.append('')
    lines.append('=' * 100)
    lines.append('  PER-ARCH ABLATION DELTAS vs C4 (paired bootstrap, pooled across cities)')
    lines.append('=' * 100)
    lines.append(f'  {"ID":<10} {"Arch":<9} {"Δ mIoU":>8} '
                 f'{"95% CI":>16} {"p-value":>8} {"raw":>4} '
                 f'{"Cohen d":>8} {"HB-glob":>7}')
    lines.append('  ' + '-' * 96)

    for abl_id in ABLATION_META:
        if abl_id in ('C4', 'C4_clean'):
            continue
        for arch in ARCHITECTURES:
            d = deltas.get(arch, {}).get(abl_id)
            if d is None:
                lines.append(f'  {abl_id:<10} {arch:<9} {"N/A":>8}')
                continue
            boot = d['bootstrap']
            ci = f'[{boot["ci_lo"]:+.2f}, {boot["ci_hi"]:+.2f}]'
            cd = d['cohens_d']
            cd_s = f'{cd:+.2f}' if not np.isnan(cd) else '—'
            raw_s = '*' if d.get('significant_raw') else ''
            hb_s = '*' if d.get('significant_corrected') else ''
            lines.append(
                f'  {abl_id:<10} {arch:<9} {d["delta_mean"]:+8.2f} '
                f'{ci:>16} {boot["p_value"]:8.4f} '
                f'{raw_s:>4} {cd_s:>8} {hb_s:>7}')
        lines.append('')


def render_recovery_table(recovery: Dict, lines: List[str]):
    lines.append('')
    lines.append('=' * 90)
    lines.append('  RECOVERY RATIOS:  R = (method − Vanilla) / (Upper − Vanilla)')
    lines.append('=' * 90)
    header = f'  {"ID":<10} {"Name":<26}'
    for arch in ARCHITECTURES:
        header += f' {arch:>10}'
    header += f'  {"Mean":>8}'
    lines.append(header)
    lines.append('  ' + '-' * 86)

    for abl_id in ABLATION_META:
        meta = ABLATION_META[abl_id]
        row = f'  {abl_id:<10} {meta["name"]:<26}'
        means = []
        for arch in ARCHITECTURES:
            r = recovery.get(arch, {}).get(abl_id)
            if r and not np.isnan(r['mean']):
                row += f' {r["mean"]:10.3f}'
                means.append(r['mean'])
            else:
                row += f' {"—":>10}'
        if means:
            row += f'  {np.mean(means):8.3f}'
        else:
            row += f'  {"—":>8}'
        lines.append(row)


def render_summary(summary: Dict, lines: List[str]):
    lines.append('')
    lines.append('=' * 80)
    lines.append('  CONDENSED SUMMARY (for abstract / conclusion)')
    lines.append('=' * 80)

    lines.append(f'  C4       overall tol mIoU: {summary["c4_overall_miou"]:.2f}')
    lines.append(f'  C4_clean overall tol mIoU: {summary["c4_clean_overall_miou"]:.2f}')

    for ref in ['C4', 'C4_clean']:
        per_arch = summary['gap_closure'].get(ref, {})
        if not per_arch:
            continue
        Rs = [v['mean_R'] for v in per_arch.values()]
        if Rs:
            lines.append(f'  {ref} gap closure (mean R): {np.mean(Rs):.3f}  '
                         f'range {min(Rs)*100:.0f}–{max(Rs)*100:.0f}%')

    if summary.get('C4_worst_vs_vanilla') is not None:
        lines.append(f'  C4 worst single-cell Δ vs Vanilla:       '
                     f'{summary["C4_worst_vs_vanilla"]:+.2f} mIoU')
    if summary.get('C4_clean_worst_vs_vanilla') is not None:
        lines.append(f'  C4_clean worst single-cell Δ vs Vanilla: '
                     f'{summary["C4_clean_worst_vs_vanilla"]:+.2f} mIoU')

    lines.append(f'  Significant ablation drops: '
                 f'{summary["n_significant_drops"]}/{summary["n_total_comparisons"]}')

    lines.append('')
    lines.append('  Largest ablation drop per architecture:')
    for arch, info in summary['largest_drops'].items():
        if info['ablation']:
            lines.append(f'    {arch:<10} {info["ablation"]:5s} ({info["name"]:<26}) '
                         f'Δ = {info["delta"]:+.2f} mIoU')

    lines.append('')
    lines.append('  Diagnostic validation:')
    for code, info in summary['diagnostic_validation'].items():
        cm = '✓' if info['confirms_diagnostic'] else '✗'
        lines.append(f'    {code:<6} ablations {info["ablations"]}  '
                     f'mean Δ = {info["mean_delta"]:+.2f}  {cm}')


def render_predictions(verdicts: List[str], lines: List[str]):
    lines.append('')
    lines.append('=' * 80)
    lines.append('  §5.3 PREDICTION VALIDATION')
    lines.append('=' * 80)
    lines.extend(verdicts)


# ════════════════════════════════════════════════════════════════════════════
# LaTeX
# ════════════════════════════════════════════════════════════════════════════

def write_latex_table(table: Dict, deltas: Dict, output_path: str):
    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{')
    lines.append(r'    \textbf{SIB component ablations (tolerant mIoU).}')
    lines.append(r'    Each row removes one design choice from SIB-Full (C4).')
    lines.append(r'    C4\_clean is the simplified Haar+VIB-only variant. ')
    lines.append(r'    $\Delta$ = change from C4 (negative = component was needed).')
    lines.append(r'    $^{*}$$p<.05$, $^{**}$$p<.01$, $^{***}$$p<.001$ ')
    lines.append(r'    (paired bootstrap $B$=10000).')
    lines.append(r'  }')
    lines.append(r'  \label{tab:sib_ablations_v2}')
    lines.append(r'  \small')
    lines.append(r'  \begin{tabular}{@{}llccccccccccc@{}}')
    lines.append(r'    \toprule')
    lines.append(r'    & & \multicolumn{3}{c}{MAMNet} & '
                 r'\multicolumn{3}{c}{OGLANet} & '
                 r'\multicolumn{3}{c}{DINOv3} & Avg & $\Delta$ \\')
    lines.append(r'    \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}')
    lines.append(r'    ID & Diag. & CHI & MIA & PHX & CHI & MIA & PHX & '
                 r'CHI & MIA & PHX & & \\')
    lines.append(r'    \midrule')

    for abl_id, meta in ABLATION_META.items():
        diag = meta['diagnostic']
        abl_id_tex = abl_id.replace('_', r'\_')
        parts = [f'    {abl_id_tex}', diag]
        all_v = []
        for arch in ARCHITECTURES:
            entry = table[arch].get(abl_id)
            for city in CITIES:
                if entry and not np.isnan(entry.get(city, np.nan)):
                    val = entry[city]
                    all_v.append(val)
                    parts.append(f'{val:.1f}')
                else:
                    parts.append('--')
        if all_v:
            parts.append(f'{np.mean(all_v):.1f}')
        else:
            parts.append('--')

        if abl_id in ('C4', 'C4_clean'):
            parts.append('--')
        else:
            arch_d, arch_p = [], []
            for arch in ARCHITECTURES:
                d = deltas.get(arch, {}).get(abl_id)
                if d:
                    arch_d.append(d['delta_mean'])
                    arch_p.append(d['bootstrap']['p_value'])
            if arch_d:
                avg = np.mean(arch_d)
                stars = sig_stars(min(arch_p))
                parts.append(f'{avg:+.1f}$^{{{stars}}}$' if stars else f'{avg:+.1f}')
            else:
                parts.append('--')

        lines.append(' & '.join(parts) + r' \\')
        if abl_id == 'C4_clean':
            lines.append(r'    \midrule')

    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


# ════════════════════════════════════════════════════════════════════════════
# JSON
# ════════════════════════════════════════════════════════════════════════════

def _to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def build_json_report(table, baselines_table, deltas, recovery, summary, args):
    return _to_jsonable({
        'generated':          datetime.now().isoformat(),
        'base_path':          args.base_path,
        'boundary_tolerance': args.boundary_tolerance,
        'img_size':           args.img_size,
        'n_bootstrap':        args.n_bootstrap,
        'alpha':              args.alpha,
        'architectures':      ARCHITECTURES,
        'cities':             CITIES,
        'ablation_meta':      {k: dict(v) for k, v in ABLATION_META.items()},
        'main_table':         table,
        'baselines':          baselines_table,
        'deltas_vs_c4':       deltas,
        'recovery_ratios':    recovery,
        'summary':            summary,
    })


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='SIB Ablation Analysis V2 — recompute everything '
                    'from prediction PNGs.')
    p.add_argument('--base_path', type=str,
                   default=os.environ["PROJECT_ROOT"])
    p.add_argument('--output_dir', type=str,
                   default='./ablation_analysis_v2')
    p.add_argument('--boundary_tolerance', type=int, default=2)
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--n_bootstrap', type=int, default=10000)
    p.add_argument('--alpha', type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=' * 80)
    print('  SIB ABLATION ANALYSIS V2 — recomputed from prediction PNGs')
    print(f'  base_path:           {args.base_path}')
    print(f'  output_dir:          {args.output_dir}')
    print(f'  boundary_tolerance:  ±{args.boundary_tolerance} px')
    print(f'  img_size:            {args.img_size}')
    print(f'  bootstrap B:         {args.n_bootstrap}')
    print(f'  alpha:               {args.alpha}')
    print('=' * 80)

    # ── Step 1: per-image tolerant mIoU per (arch, abl, city) ─────────────
    print('\n[1/6] Collecting tolerant mIoU for ablation cells...')
    abl_results = collect_ablation_metrics(
        args.base_path, args.img_size, args.boundary_tolerance, verbose=True)

    # ── Step 2: per-image tolerant mIoU for baselines ──────────────────────
    print('\n[2/6] Collecting tolerant mIoU for baselines...')
    baseline_results = collect_baseline_metrics(
        args.base_path, args.img_size, args.boundary_tolerance, verbose=True)

    # ── Step 3: master tables ──────────────────────────────────────────────
    print('\n[3/6] Building master tables...')
    table = build_master_table(abl_results)
    baselines_table = build_baseline_table(baseline_results)

    # ── Step 4: deltas vs C4 ───────────────────────────────────────────────
    print('\n[4/6] Computing deltas vs C4 (paired bootstrap)...')
    deltas = compute_deltas_vs_c4(
        abl_results, n_bootstrap=args.n_bootstrap, alpha=args.alpha)

    # ── Step 5: recovery ratios + summary ──────────────────────────────────
    print('\n[5/6] Computing recovery + summary...')
    recovery = compute_recovery(table, baselines_table)
    summary = compute_summary(table, baselines_table, deltas)

    # ── Step 6: render reports ─────────────────────────────────────────────
    print('\n[6/6] Rendering reports...')

    main_lines = ['SIB ABLATION ANALYSIS V2 — REPORT',
                  f'Generated: {datetime.now().isoformat()}',
                  f'Eval: tolerant mIoU (±{args.boundary_tolerance} px)']
    render_main_table(table, deltas, main_lines)
    main_path = os.path.join(args.output_dir, 'ablation_v2_main_table.txt')
    with open(main_path, 'w') as f:
        f.write('\n'.join(main_lines))

    delta_lines = []
    render_per_arch_deltas(deltas, delta_lines)
    delta_path = os.path.join(args.output_dir, 'ablation_v2_deltas.txt')
    with open(delta_path, 'w') as f:
        f.write('\n'.join(delta_lines))

    rec_lines = []
    render_recovery_table(recovery, rec_lines)
    rec_path = os.path.join(args.output_dir, 'ablation_v2_recovery.txt')
    with open(rec_path, 'w') as f:
        f.write('\n'.join(rec_lines))

    sum_lines = []
    render_summary(summary, sum_lines)
    sum_path = os.path.join(args.output_dir, 'ablation_v2_summary.txt')
    with open(sum_path, 'w') as f:
        f.write('\n'.join(sum_lines))

    pred_lines = []
    verdicts = validate_predictions(deltas, table)
    render_predictions(verdicts, pred_lines)
    pred_path = os.path.join(args.output_dir, 'ablation_v2_predictions.txt')
    with open(pred_path, 'w') as f:
        f.write('\n'.join(pred_lines))

    # Print everything to stdout too
    for chunk in [main_lines, delta_lines, rec_lines, sum_lines, pred_lines]:
        print('\n'.join(chunk))

    # LaTeX
    latex_path = os.path.join(args.output_dir, 'ablation_v2_table4.tex')
    write_latex_table(table, deltas, latex_path)
    print(f'\n  LaTeX table → {latex_path}')

    # JSON
    json_report = build_json_report(table, baselines_table, deltas,
                                     recovery, summary, args)
    json_path = os.path.join(args.output_dir, 'ablation_v2_report.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f'  JSON report → {json_path}')

    print(f'\n  Done. All outputs in: {args.output_dir}')


if __name__ == '__main__':
    main()