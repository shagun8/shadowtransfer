"""
aggregate_sp_gap.py — Aggregate Phase 1 SP-gap results across 3 architectures
                     and compare C4-clean against Vanilla and 6 adaptation
                     methods using saved probability maps.

Inputs (per architecture):
  1. Per-cell C4-clean SP-gap JSON files from sp_gap_analysis.py
     (e.g. sp_gap_mamnet_phoenix_highres.json)
     These contain per-image AURC_shadow / ECE_pred_pos / mIoU for C4-clean.

  2. Saved .npy probability maps under Test_img_probs/:
     {probs_root}/loco/{city}/{res}/{model}/{method}/{stem}.npy
     One file per test image, float16 H×W, raw P(shadow) (no filtering).

  3. Ground-truth masks at:
     {data_root}/{city}/{res}/test/masks/{stem}.png
     (binary 0/255, threshold at 127 -> {0, 1})

Methodology:
  Per-image AURC_shadow on gt-shadow pixels (sort by P(shadow) desc, error
  = fraction of top-c predicted as background, integrate over coverage grid).
  Identical to sp_gap_analysis.py's per-image function — same coverage grid,
  same min_shadow_pixels, same threshold (0.5).

Statistical design:
  Per-cell paired bootstrap (B=10,000, image-level cluster bootstrap).
  Population-level test: Wilcoxon signed-rank on the 9 cell-mean deltas
  (3 archs × 3 cities), one-sided H1: C4-clean < competitor (lower AURC).

Comparisons computed:
  A. C4-clean vs. Vanilla         — primary (does §4.2+§4.4 reduce SP-gap?)
  B. C4-clean vs. BestDA-per-cell — strongest baseline (per-cell oracle pick)
  C. C4-clean vs. MeanDA          — average over 6 adaptation methods
  D. C4-clean vs. each method     — for the full table
  E. Per-architecture and per-city breakdowns

Output: a comprehensive JSON + Markdown summary.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

try:
    from scipy.stats import wilcoxon
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print('WARNING: scipy not available; Wilcoxon test will be skipped.')


ARCHITECTURES = ['mamnet', 'oglanet', 'dinov3']
CITIES = ['phoenix', 'miami', 'chicago']
ADAPTATION_METHODS = ['fda', 'segdesic', 'iim', 'isw', 'mrfp_plus', 'fada']
ALL_METHODS = ['vanilla'] + ADAPTATION_METHODS    # methods loaded from .npy


# =====================================================================
# Per-image metric — IDENTICAL to sp_gap_analysis.py
# =====================================================================

def compute_aurc_shadow_per_image(shadow_prob_hw, gt_label_hw,
                                   n_coverage=20, min_pixels=5):
    """
    AURC_shadow on gt-shadow pixels. Lower = better.
    MUST match the formula in sp_gap_analysis.py exactly.
    """
    shadow_mask = (gt_label_hw.ravel() == 1)
    n_shadow = int(shadow_mask.sum())
    if n_shadow < min_pixels:
        return float('nan')

    confs = shadow_prob_hw.ravel()[shadow_mask].astype(np.float32)
    correct = (confs > 0.5).astype(np.float32)

    sort_idx = np.argsort(-confs)
    sorted_correct = correct[sort_idx]

    coverage_grid = np.linspace(0.10, 1.0, n_coverage)
    aurc = 0.0
    for c in coverage_grid:
        k = max(1, int(round(c * n_shadow)))
        aurc += 1.0 - float(sorted_correct[:k].mean())
    return aurc / n_coverage


def compute_miou_per_image(shadow_prob_hw, gt_label_hw):
    pred = (shadow_prob_hw > 0.5).astype(np.int32)
    gt = gt_label_hw.astype(np.int32)
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    shadow_iou = tp / (tp + fp + fn + 1e-10)
    bg_iou = tn / (tn + fp + fn + 1e-10)
    return float((shadow_iou + bg_iou) / 2)


# =====================================================================
# Bootstrap
# =====================================================================

def bootstrap_delta_ci(deltas, B=10000, seed=42):
    """
    Image-level cluster bootstrap on per-image deltas.
    Returns: (mean_delta, ci_lo, ci_hi, p_two_sided, n_valid)
    """
    valid = deltas[~np.isnan(deltas)]
    n = len(valid)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), float('nan'), 0

    obs_mean = float(np.mean(valid))
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        float(np.mean(valid[rng.choice(n, n, replace=True)]))
        for _ in range(B)
    ])
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    if obs_mean < 0:
        p = 2.0 * max(float(np.mean(boot_means >= 0)), 1.0 / B)
    else:
        p = 2.0 * max(float(np.mean(boot_means <= 0)), 1.0 / B)
    p = min(p, 1.0)
    return obs_mean, ci_lo, ci_hi, p, n


# =====================================================================
# Loading: GT masks + saved .npy probability maps
# =====================================================================

def load_gt_mask(mask_path, expected_size=None):
    """Load GT mask, threshold to {0, 1}."""
    img = Image.open(mask_path).convert('L')
    if expected_size is not None and img.size != (expected_size, expected_size):
        img = img.resize((expected_size, expected_size), Image.NEAREST)
    arr = np.array(img)
    return (arr > 127).astype(np.int32)


def load_npy_method_results(probs_root, data_root, arch, method, city,
                             resolution, expected_size, n_coverage,
                             min_shadow_pixels):
    """
    Load saved .npy probability maps for one (arch, method, city) cell,
    pair with GT masks, compute per-image AURC_shadow and mIoU.

    Returns: dict {filename: (aurc, miou)}
    """
    probs_dir = Path(probs_root) / 'loco' / city / resolution / arch / method
    masks_dir = Path(data_root) / city / resolution / 'test' / 'masks'

    if not probs_dir.is_dir():
        # Try the single-fold-output convention used in some past runs
        return None
    if not masks_dir.is_dir():
        # Try without the test/ subdir
        masks_dir = Path(data_root) / city / resolution / 'masks'
        if not masks_dir.is_dir():
            print(f'    WARNING: GT masks dir not found for {city}/{resolution}')
            return None

    # Build filename -> path maps
    npy_files = {f.stem: f for f in probs_dir.glob('*.npy')}
    mask_files = {f.stem: f for f in masks_dir.glob('*.png')}

    common_stems = sorted(set(npy_files) & set(mask_files))
    if not common_stems:
        print(f'    WARNING: no matching stems in {probs_dir}')
        return None

    results = {}
    for stem in common_stems:
        prob = np.load(npy_files[stem]).astype(np.float32)
        gt = load_gt_mask(mask_files[stem], expected_size=expected_size)

        # Resize prob if needed (some methods may have saved at different size)
        if prob.shape != gt.shape:
            from PIL import Image as _PI
            # Treat prob as grayscale (in [0,1]) -> resize via PIL
            prob_img = _PI.fromarray((prob * 255).astype(np.uint8))
            prob_img = prob_img.resize(gt.shape[::-1], _PI.BILINEAR)
            prob = np.array(prob_img).astype(np.float32) / 255.0

        aurc = compute_aurc_shadow_per_image(
            prob, gt, n_coverage=n_coverage, min_pixels=min_shadow_pixels)
        miou = compute_miou_per_image(prob, gt)
        results[stem] = (aurc, miou)

    return results


def load_c4clean_per_image(c4_json_path):
    """Load C4-clean per-image AURC/mIoU from sp_gap_analysis.py output."""
    with open(c4_json_path) as f:
        data = json.load(f)

    out = {}
    for rec in data['per_image']:
        stem = os.path.splitext(rec['filename'])[0]
        aurc = rec['aurc_shadow'] if rec['aurc_shadow'] is not None else float('nan')
        miou = rec['miou']
        out[stem] = (aurc, miou)
    return out, data


# =====================================================================
# Per-cell comparison
# =====================================================================

def compute_per_cell_deltas(c4_per_image, method_per_image):
    """
    Pair C4-clean and method per image by stem.
    Returns: dict with delta_aurc, delta_miou, n_paired arrays.
    """
    common_stems = sorted(set(c4_per_image) & set(method_per_image))
    if not common_stems:
        return None

    delta_aurc, delta_miou = [], []
    c4_aurc_arr, m_aurc_arr = [], []
    c4_miou_arr, m_miou_arr = [], []
    for stem in common_stems:
        c4_a, c4_m = c4_per_image[stem]
        mt_a, mt_m = method_per_image[stem]
        delta_aurc.append(c4_a - mt_a)   # negative = C4-clean better (lower AURC)
        delta_miou.append(c4_m - mt_m)   # positive = C4-clean better (higher mIoU)
        c4_aurc_arr.append(c4_a)
        m_aurc_arr.append(mt_a)
        c4_miou_arr.append(c4_m)
        m_miou_arr.append(mt_m)

    return {
        'n_paired':    len(common_stems),
        'delta_aurc':  np.array(delta_aurc, dtype=np.float64),
        'delta_miou':  np.array(delta_miou, dtype=np.float64),
        'c4_aurc':     np.array(c4_aurc_arr, dtype=np.float64),
        'method_aurc': np.array(m_aurc_arr, dtype=np.float64),
        'c4_miou':     np.array(c4_miou_arr, dtype=np.float64),
        'method_miou': np.array(m_miou_arr, dtype=np.float64),
    }


def summarize_paired_comparison(paired, B=10000):
    """Bootstrap CIs + summary stats from a paired dict."""
    if paired is None:
        return None

    m_aurc, lo_a, hi_a, p_a, n_a = bootstrap_delta_ci(paired['delta_aurc'], B)
    m_miou, lo_m, hi_m, p_m, n_m = bootstrap_delta_ci(paired['delta_miou'], B)

    n_pairs = paired['n_paired']
    delta_a = paired['delta_aurc']
    valid_a = delta_a[~np.isnan(delta_a)]
    n_improve_aurc = int(np.sum(valid_a < 0))   # C4-clean strictly better

    return {
        'n_paired': n_pairs,
        'aurc_shadow': {
            'c4clean_mean': float(np.nanmean(paired['c4_aurc'])),
            'method_mean':  float(np.nanmean(paired['method_aurc'])),
            'mean_delta':   m_aurc,
            'ci_lo':        lo_a,
            'ci_hi':        hi_a,
            'p_two_sided':  p_a,
            'n_valid':      n_a,
            'n_improve':    n_improve_aurc,    # # images where C4-clean lower
        },
        'miou': {
            'c4clean_mean': float(np.nanmean(paired['c4_miou'])),
            'method_mean':  float(np.nanmean(paired['method_miou'])),
            'mean_delta':   m_miou,
            'ci_lo':        lo_m,
            'ci_hi':        hi_m,
            'p_two_sided':  p_m,
        },
    }


# =====================================================================
# Best-DA per cell + Mean-DA per cell
# =====================================================================

def build_bestda_per_image(method_results_dict):
    """
    For one cell (arch × city), given {method: {stem: (aurc, miou)}},
    pick per-image the lowest-AURC method (oracle per-image).

    Reports both per-image (oracle) and per-cell (single-method-per-cell)
    versions. The per-cell version is what's reported in the table — it
    picks the single method that gave the lowest mean AURC on this cell.

    Returns:
      bestda_per_image: {stem: (best_aurc, best_miou)} — per-image oracle
      bestda_method:    name of the method with lowest cell-mean AURC
      bestda_per_image_method_choice: {stem: method_name}
    """
    # Per-image oracle: pick lowest AURC across the 6 methods per image
    all_stems = set()
    for m, r in method_results_dict.items():
        if r is not None:
            all_stems.update(r.keys())

    bestda_per_image = {}
    bestda_choice = {}
    for stem in all_stems:
        best_method = None
        best_aurc = float('inf')
        best_miou = None
        for m, r in method_results_dict.items():
            if r is None or stem not in r:
                continue
            a, mi = r[stem]
            if not np.isnan(a) and a < best_aurc:
                best_aurc = a
                best_miou = mi
                best_method = m
        if best_method is not None:
            bestda_per_image[stem] = (best_aurc, best_miou)
            bestda_choice[stem] = best_method

    # Per-cell single-method: method with lowest cell-mean AURC
    cell_means = {}
    for m, r in method_results_dict.items():
        if r is None:
            continue
        aurcs = np.array([a for a, _ in r.values()], dtype=np.float64)
        valid = aurcs[~np.isnan(aurcs)]
        if len(valid) > 0:
            cell_means[m] = float(np.mean(valid))
    bestda_method = min(cell_means, key=cell_means.get) if cell_means else None

    return bestda_per_image, bestda_method, bestda_choice


def build_meanda_per_image(method_results_dict):
    """
    For one cell, average AURC across the 6 methods per image.
    Skips methods missing for that image.
    """
    all_stems = set()
    for m, r in method_results_dict.items():
        if r is not None:
            all_stems.update(r.keys())

    meanda_per_image = {}
    for stem in all_stems:
        aurcs, mious = [], []
        for m, r in method_results_dict.items():
            if r is None or stem not in r:
                continue
            a, mi = r[stem]
            if not np.isnan(a):
                aurcs.append(a)
                mious.append(mi)
        if aurcs:
            meanda_per_image[stem] = (float(np.mean(aurcs)), float(np.mean(mious)))
    return meanda_per_image


# =====================================================================
# Population-level Wilcoxon
# =====================================================================

def population_wilcoxon(cell_deltas_array, label):
    """
    cell_deltas_array: 1-D array of cell-mean deltas (one per cell, n=9).
    H1: C4-clean better => mean delta < 0 (for AURC) or > 0 (for mIoU).
    Returns dict with statistic, p-value, n.
    """
    arr = np.asarray(cell_deltas_array)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    out = {'metric': label, 'n_cells': n,
           'mean_delta_across_cells': float(np.mean(arr)) if n > 0 else float('nan')}

    if n < 6:
        out['note'] = 'n < 6 cells; Wilcoxon underpowered (min p ~0.03 at n=6)'
        return out
    if not SCIPY_OK:
        out['note'] = 'scipy not installed; install for population-level test'
        return out

    # Two-sided Wilcoxon
    try:
        stat_two, p_two = wilcoxon(arr, alternative='two-sided',
                                    zero_method='wilcox')
    except ValueError:
        # All zeros or other degeneracy
        return {**out, 'wilcoxon_two_sided_p': float('nan'),
                'note': 'Wilcoxon failed (possibly all-zero deltas)'}

    # One-sided alternative: deltas < 0 (C4-clean better) for AURC,
    # or > 0 for mIoU. The caller determined the metric direction.
    # We always report both directions and let the reader interpret.
    try:
        stat_less, p_less = wilcoxon(arr, alternative='less',
                                      zero_method='wilcox')
    except ValueError:
        p_less = float('nan')
    try:
        stat_greater, p_greater = wilcoxon(arr, alternative='greater',
                                            zero_method='wilcox')
    except ValueError:
        p_greater = float('nan')

    out['wilcoxon_statistic'] = float(stat_two)
    out['wilcoxon_two_sided_p'] = float(p_two)
    out['wilcoxon_p_less']     = float(p_less)     # C4-clean < competitor
    out['wilcoxon_p_greater']  = float(p_greater)  # C4-clean > competitor
    out['n_negative'] = int(np.sum(arr < 0))
    out['n_positive'] = int(np.sum(arr > 0))
    out['n_zero']     = int(np.sum(arr == 0))
    return out


# =====================================================================
# Main aggregator
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Aggregate SP-gap results')

    p.add_argument('--mamnet_results_dir', required=True,
                   help='Dir with sp_gap_mamnet_*.json files')
    p.add_argument('--oglanet_results_dir', required=True)
    p.add_argument('--dinov3_results_dir', required=True)

    p.add_argument('--probs_root', required=True,
                   help='Test_img_probs root')
    p.add_argument('--data_root', required=True,
                   help='Final_data_test root (for GT masks)')
    p.add_argument('--output_dir', required=True)

    p.add_argument('--resolution', default='highres', choices=['highres', 'midres'])
    p.add_argument('--bootstrap_B', type=int, default=10000)
    p.add_argument('--n_coverage', type=int, default=20)
    p.add_argument('--min_shadow_pixels', type=int, default=5)
    p.add_argument('--img_size', type=int, default=384)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Note: MAMNet's JSON has slightly different shape (it computed Vanilla
    # internally too). For uniformity, the aggregator re-loads Vanilla from
    # .npy for ALL three architectures so the comparison methodology is
    # identical. C4-clean per-image arrays are loaded from each arch's JSON.
    arch_results_dirs = {
        'mamnet':  args.mamnet_results_dir,
        'oglanet': args.oglanet_results_dir,
        'dinov3':  args.dinov3_results_dir,
    }

    # ============================================================
    # PASS 1 — Load per-image results for every (arch, city, method)
    # ============================================================
    print(f'\n{"="*70}')
    print('PASS 1: Loading per-image results')
    print(f'{"="*70}')

    # data[arch][city][method] = {stem: (aurc, miou)}
    data = defaultdict(lambda: defaultdict(dict))
    c4_meta = {}  # data[arch][city] -> raw C4 JSON metadata

    for arch in ARCHITECTURES:
        results_dir = arch_results_dirs[arch]
        for city in CITIES:
            print(f'\n  {arch} / {city}')

            # ---- C4-clean: load from per-arch JSON ----
            # MAMNet's JSON is named sp_gap_mamnet_{city}_{res}.json (no "c4clean")
            # OGLANet/DINOv3: sp_gap_{arch}_c4clean_{city}_{res}.json
            candidates = [
                Path(results_dir) / f'sp_gap_{arch}_c4clean_{city}_{args.resolution}.json',
                Path(results_dir) / f'sp_gap_{arch}_{city}_{args.resolution}.json',
            ]
            c4_path = None
            for c in candidates:
                if c.exists():
                    c4_path = c
                    break

            if c4_path is None:
                print(f'    WARNING: C4-clean JSON not found in {results_dir}')
                print(f'    Tried: {[str(c) for c in candidates]}')
                continue

            c4_per_image, c4_raw = load_c4clean_per_image(c4_path)
            data[arch][city]['c4clean'] = c4_per_image
            c4_meta[(arch, city)] = {
                'json_path':           str(c4_path),
                'n_images':            c4_raw.get('n_images_total', len(c4_per_image)),
                'aurc_shadow_mean':    c4_raw.get('aurc_shadow_mean'),
                'miou_mean':           c4_raw.get('miou_mean'),
            }
            print(f'    c4clean: {len(c4_per_image)} images')

            # ---- Vanilla + 6 adaptation methods: load from .npy ----
            for method in ALL_METHODS:
                res = load_npy_method_results(
                    args.probs_root, args.data_root,
                    arch, method, city, args.resolution,
                    expected_size=args.img_size,
                    n_coverage=args.n_coverage,
                    min_shadow_pixels=args.min_shadow_pixels,
                )
                if res is not None:
                    data[arch][city][method] = res
                    valid_aurcs = [a for a, _ in res.values() if not np.isnan(a)]
                    mean_a = float(np.mean(valid_aurcs)) if valid_aurcs else float('nan')
                    print(f'    {method:10s}: {len(res)} images  '
                          f'mean AURC={mean_a:.4f}')
                else:
                    print(f'    {method:10s}: NOT FOUND (skipped)')

    # ============================================================
    # PASS 2 — Per-cell comparisons + per-cell BestDA / MeanDA
    # ============================================================
    print(f'\n{"="*70}')
    print('PASS 2: Per-cell comparisons')
    print(f'{"="*70}')

    # cell_results[arch][city] = comprehensive comparison dict
    cell_results = defaultdict(dict)
    # cell_means[arch][city][method] = mean AURC for that method on that cell
    cell_means_aurc = defaultdict(lambda: defaultdict(dict))
    cell_means_miou = defaultdict(lambda: defaultdict(dict))

    for arch in ARCHITECTURES:
        for city in CITIES:
            cell_data = data[arch][city]
            if 'c4clean' not in cell_data:
                continue
            c4 = cell_data['c4clean']

            cell_summary = {
                'arch': arch, 'city': city, 'resolution': args.resolution,
                'comparisons': {},
            }

            # Cell means for every method (used later by BestDA cell-level)
            cell_means_aurc[arch][city]['c4clean'] = float(
                np.nanmean([a for a, _ in c4.values()]))
            cell_means_miou[arch][city]['c4clean'] = float(
                np.nanmean([m for _, m in c4.values()]))

            # ---- Comparisons against each individual method ----
            method_data_dict = {m: cell_data.get(m) for m in ALL_METHODS}
            for method in ALL_METHODS:
                m_per_img = method_data_dict[method]
                if m_per_img is None:
                    continue
                paired = compute_per_cell_deltas(c4, m_per_img)
                summary = summarize_paired_comparison(paired, B=args.bootstrap_B)
                if summary is not None:
                    cell_summary['comparisons'][method] = summary
                    # Cache cell mean
                    cell_means_aurc[arch][city][method] = float(
                        np.nanmean([a for a, _ in m_per_img.values()]))
                    cell_means_miou[arch][city][method] = float(
                        np.nanmean([m for _, m in m_per_img.values()]))

            # ---- BestDA-per-cell (oracle within the 6 adaptation methods) ----
            # 'vanilla' is excluded from the BestDA pool — that's the source-only
            # baseline we want to beat separately.
            adapt_methods_data = {m: cell_data.get(m)
                                  for m in ADAPTATION_METHODS}
            bestda_per_img, bestda_name, _ = build_bestda_per_image(
                adapt_methods_data)
            if bestda_per_img:
                paired = compute_per_cell_deltas(c4, bestda_per_img)
                summary = summarize_paired_comparison(paired, B=args.bootstrap_B)
                if summary is not None:
                    cell_summary['comparisons']['bestda_per_image_oracle'] = summary
                cell_summary['bestda_cell_method'] = bestda_name

                # Also pair against the SINGLE best DA on this cell
                # (not per-image oracle; one method picked for the whole cell)
                if bestda_name and adapt_methods_data[bestda_name] is not None:
                    paired_cell = compute_per_cell_deltas(
                        c4, adapt_methods_data[bestda_name])
                    summ_cell = summarize_paired_comparison(
                        paired_cell, B=args.bootstrap_B)
                    if summ_cell is not None:
                        cell_summary['comparisons']['bestda_cell_winner'] = summ_cell

            # ---- MeanDA-per-cell (average across 6 adaptation methods) ----
            meanda_per_img = build_meanda_per_image(adapt_methods_data)
            if meanda_per_img:
                paired = compute_per_cell_deltas(c4, meanda_per_img)
                summary = summarize_paired_comparison(paired, B=args.bootstrap_B)
                if summary is not None:
                    cell_summary['comparisons']['meanda'] = summary

            cell_results[arch][city] = cell_summary

            # ---- Print quick per-cell summary ----
            print(f'\n  {arch} / {city}:')
            van = cell_summary['comparisons'].get('vanilla', {}).get('aurc_shadow', {})
            if van:
                print(f'    vs Vanilla : ΔAURC={van["mean_delta"]:+.4f}  '
                      f'CI [{van["ci_lo"]:+.4f}, {van["ci_hi"]:+.4f}]  '
                      f'p={van["p_two_sided"]:.4f}  '
                      f'{van["n_improve"]}/{van["n_valid"]} images improve')
            bd = cell_summary['comparisons'].get('bestda_cell_winner', {}).get('aurc_shadow', {})
            if bd:
                bdname = cell_summary.get('bestda_cell_method', '?')
                print(f'    vs BestDA  : ΔAURC={bd["mean_delta"]:+.4f}  '
                      f'CI [{bd["ci_lo"]:+.4f}, {bd["ci_hi"]:+.4f}]  '
                      f'(winner: {bdname})')
            md = cell_summary['comparisons'].get('meanda', {}).get('aurc_shadow', {})
            if md:
                print(f'    vs MeanDA  : ΔAURC={md["mean_delta"]:+.4f}  '
                      f'CI [{md["ci_lo"]:+.4f}, {md["ci_hi"]:+.4f}]')

    # ============================================================
    # PASS 3 — Population-level Wilcoxon across 9 cells
    # ============================================================
    print(f'\n{"="*70}')
    print('PASS 3: Population-level Wilcoxon (9 cells = 3 archs × 3 cities)')
    print(f'{"="*70}')

    population_results = {}
    competitor_names = ['vanilla'] + ADAPTATION_METHODS + [
        'bestda_per_image_oracle', 'bestda_cell_winner', 'meanda']

    for competitor in competitor_names:
        cell_aurc_deltas = []
        cell_miou_deltas = []
        cell_labels = []
        for arch in ARCHITECTURES:
            for city in CITIES:
                cs = cell_results.get(arch, {}).get(city)
                if cs is None:
                    continue
                comp = cs['comparisons'].get(competitor)
                if comp is None:
                    continue
                a = comp['aurc_shadow']['mean_delta']
                m = comp['miou']['mean_delta']
                cell_aurc_deltas.append(a)
                cell_miou_deltas.append(m)
                cell_labels.append(f'{arch}/{city}')

        if len(cell_aurc_deltas) >= 3:
            wilc_aurc = population_wilcoxon(
                np.array(cell_aurc_deltas),
                f'C4clean_minus_{competitor}_AURC')
            wilc_miou = population_wilcoxon(
                np.array(cell_miou_deltas),
                f'C4clean_minus_{competitor}_mIoU')
            population_results[competitor] = {
                'cell_aurc_deltas':   cell_aurc_deltas,
                'cell_miou_deltas':   cell_miou_deltas,
                'cell_labels':        cell_labels,
                'wilcoxon_aurc':      wilc_aurc,
                'wilcoxon_miou':      wilc_miou,
            }

            mean_a = wilc_aurc.get('mean_delta_across_cells', float('nan'))
            p_less = wilc_aurc.get('wilcoxon_p_less', float('nan'))
            n_neg  = wilc_aurc.get('n_negative', '-')
            n_cells = wilc_aurc.get('n_cells', 0)
            print(f'\n  vs {competitor}:')
            print(f'    n_cells = {n_cells}, mean ΔAURC = {mean_a:+.4f}')
            print(f'    Wilcoxon (H1: C4 < competitor) p = {p_less:.4f}')
            print(f'    {n_neg}/{n_cells} cells improve')

    # ============================================================
    # PASS 4 — Per-architecture and per-city aggregates
    # ============================================================
    print(f'\n{"="*70}')
    print('PASS 4: Per-architecture and per-city descriptive aggregates')
    print(f'{"="*70}')

    # Per-architecture mean of (cell-mean AURC) across 3 cities
    per_arch_means = defaultdict(dict)
    for arch in ARCHITECTURES:
        for method in ['c4clean'] + ALL_METHODS:
            vals = [cell_means_aurc[arch][city].get(method)
                    for city in CITIES
                    if method in cell_means_aurc[arch].get(city, {})]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            if vals:
                per_arch_means[arch][method] = float(np.mean(vals))

    print(f'\n  Per-architecture mean AURC_shadow (mean over 3 cities, lower=better):')
    print(f'  {"Arch":<10} | {"Vanilla":>8} | {"Best-of-DA":>10} | {"MeanDA":>8} | {"C4-clean":>9}')
    print(f'  {"-"*60}')
    for arch in ARCHITECTURES:
        am = per_arch_means.get(arch, {})
        van = am.get('vanilla', float('nan'))
        c4  = am.get('c4clean', float('nan'))
        # Best-of-DA = lowest among the 6 DA methods at the per-arch level
        da_vals = {m: am[m] for m in ADAPTATION_METHODS if m in am}
        best_da = min(da_vals.values()) if da_vals else float('nan')
        mean_da = float(np.mean(list(da_vals.values()))) if da_vals else float('nan')
        print(f'  {arch:<10} | {van:>8.4f} | {best_da:>10.4f} | '
              f'{mean_da:>8.4f} | {c4:>9.4f}')

    # Per-city mean across 3 architectures
    per_city_means = defaultdict(dict)
    for city in CITIES:
        for method in ['c4clean'] + ALL_METHODS:
            vals = [cell_means_aurc[arch][city].get(method)
                    for arch in ARCHITECTURES
                    if method in cell_means_aurc[arch].get(city, {})]
            vals = [v for v in vals if v is not None and not np.isnan(v)]
            if vals:
                per_city_means[city][method] = float(np.mean(vals))

    print(f'\n  Per-city mean AURC_shadow (mean over 3 architectures, lower=better):')
    print(f'  {"City":<10} | {"Vanilla":>8} | {"Best-of-DA":>10} | {"MeanDA":>8} | {"C4-clean":>9}')
    print(f'  {"-"*60}')
    for city in CITIES:
        cm = per_city_means.get(city, {})
        van = cm.get('vanilla', float('nan'))
        c4  = cm.get('c4clean', float('nan'))
        da_vals = {m: cm[m] for m in ADAPTATION_METHODS if m in cm}
        best_da = min(da_vals.values()) if da_vals else float('nan')
        mean_da = float(np.mean(list(da_vals.values()))) if da_vals else float('nan')
        print(f'  {city:<10} | {van:>8.4f} | {best_da:>10.4f} | '
              f'{mean_da:>8.4f} | {c4:>9.4f}')

    # Overall (9 cells) mean
    all_c4 = [cell_means_aurc[a][c].get('c4clean')
              for a in ARCHITECTURES for c in CITIES
              if 'c4clean' in cell_means_aurc[a].get(c, {})]
    all_van = [cell_means_aurc[a][c].get('vanilla')
               for a in ARCHITECTURES for c in CITIES
               if 'vanilla' in cell_means_aurc[a].get(c, {})]

    overall = {
        'c4clean_overall_mean_aurc': float(np.mean(all_c4)) if all_c4 else float('nan'),
        'vanilla_overall_mean_aurc': float(np.mean(all_van)) if all_van else float('nan'),
        'n_c4clean_cells':           len(all_c4),
        'n_vanilla_cells':           len(all_van),
    }

    print(f'\n  OVERALL (9 cells = 3 archs × 3 cities):')
    print(f'    C4-clean mean AURC: {overall["c4clean_overall_mean_aurc"]:.4f}')
    print(f'    Vanilla  mean AURC: {overall["vanilla_overall_mean_aurc"]:.4f}')

    # ============================================================
    # SAVE: comprehensive JSON
    # ============================================================
    final_results = {
        'config': {
            'resolution':         args.resolution,
            'bootstrap_B':        args.bootstrap_B,
            'n_coverage':         args.n_coverage,
            'min_shadow_pixels':  args.min_shadow_pixels,
            'architectures':      ARCHITECTURES,
            'cities':             CITIES,
            'adaptation_methods': ADAPTATION_METHODS,
        },
        'cell_results':         dict(cell_results),
        'cell_means_aurc':      {a: dict(d) for a, d in cell_means_aurc.items()},
        'cell_means_miou':      {a: dict(d) for a, d in cell_means_miou.items()},
        'per_arch_mean_aurc':   dict(per_arch_means),
        'per_city_mean_aurc':   dict(per_city_means),
        'population_wilcoxon':  population_results,
        'overall':              overall,
        'c4_meta':              {f'{k[0]}|{k[1]}': v for k, v in c4_meta.items()},
    }

    out_path = os.path.join(args.output_dir, 'sp_gap_aggregate.json')
    with open(out_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f'\nSaved JSON → {out_path}')

    # ============================================================
    # SAVE: Markdown summary
    # ============================================================
    md_path = os.path.join(args.output_dir, 'sp_gap_summary.md')
    write_markdown_summary(final_results, md_path)
    print(f'Saved Markdown → {md_path}')


# =====================================================================
# Markdown summary writer
# =====================================================================

def _fmt(x, prec=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x:.{prec}f}'


def write_markdown_summary(results, path):
    lines = []
    lines.append('# SP-Gap Phase 1 — Aggregate Results')
    lines.append('')
    lines.append(f'**Resolution:** {results["config"]["resolution"]}  '
                 f'**Bootstrap B:** {results["config"]["bootstrap_B"]}  '
                 f'**Cells:** {len(ARCHITECTURES)} archs × {len(CITIES)} cities = 9')
    lines.append('')

    # ---- Headline table ----
    lines.append('## 1. Headline: per-architecture mean AURC_shadow (lower = better)')
    lines.append('')
    lines.append('| Architecture | Vanilla | Best-of-DA | MeanDA | **C4-clean** |')
    lines.append('|---|---|---|---|---|')
    pa = results['per_arch_mean_aurc']
    for arch in ARCHITECTURES:
        am = pa.get(arch, {})
        van = am.get('vanilla', float('nan'))
        c4 = am.get('c4clean', float('nan'))
        da_vals = {m: am[m] for m in ADAPTATION_METHODS if m in am}
        best_da = min(da_vals.values()) if da_vals else float('nan')
        mean_da = float(np.mean(list(da_vals.values()))) if da_vals else float('nan')
        lines.append(f'| {arch} | {_fmt(van)} | {_fmt(best_da)} | '
                     f'{_fmt(mean_da)} | **{_fmt(c4)}** |')
    lines.append('')

    # ---- Per-city table ----
    lines.append('## 2. Per-city mean AURC_shadow (averaged over 3 archs)')
    lines.append('')
    lines.append('| City | Vanilla | Best-of-DA | MeanDA | **C4-clean** |')
    lines.append('|---|---|---|---|---|')
    pc = results['per_city_mean_aurc']
    for city in CITIES:
        cm = pc.get(city, {})
        van = cm.get('vanilla', float('nan'))
        c4 = cm.get('c4clean', float('nan'))
        da_vals = {m: cm[m] for m in ADAPTATION_METHODS if m in cm}
        best_da = min(da_vals.values()) if da_vals else float('nan')
        mean_da = float(np.mean(list(da_vals.values()))) if da_vals else float('nan')
        lines.append(f'| {city} | {_fmt(van)} | {_fmt(best_da)} | '
                     f'{_fmt(mean_da)} | **{_fmt(c4)}** |')
    lines.append('')

    # ---- Population Wilcoxon ----
    lines.append('## 3. Population-level Wilcoxon test (n=9 cells, one-sided)')
    lines.append('')
    lines.append('H1: C4-clean reduces AURC_shadow (lower = better).')
    lines.append('Reports `wilcoxon_p_less` = P(observed delta is "less" extreme '
                 'under H0 of zero median).')
    lines.append('')
    lines.append('| Competitor | Mean ΔAURC | n improve | Wilcoxon p (less) |')
    lines.append('|---|---|---|---|')
    pop = results['population_wilcoxon']
    for competitor in ['vanilla', 'fda', 'segdesic', 'iim', 'isw',
                        'mrfp_plus', 'fada', 'meanda',
                        'bestda_cell_winner', 'bestda_per_image_oracle']:
        if competitor not in pop:
            continue
        w = pop[competitor]['wilcoxon_aurc']
        mean_d = w.get('mean_delta_across_cells', float('nan'))
        n_neg = w.get('n_negative', '—')
        n_c   = w.get('n_cells', 0)
        p_less = w.get('wilcoxon_p_less', float('nan'))
        lines.append(f'| vs {competitor} | {_fmt(mean_d)} | '
                     f'{n_neg}/{n_c} | {_fmt(p_less)} |')
    lines.append('')

    # ---- Per-cell C4 vs Vanilla detail ----
    lines.append('## 4. Per-cell detail: C4-clean vs Vanilla')
    lines.append('')
    lines.append('| Arch | City | C4 AURC | Van AURC | ΔAURC | 95% CI | n_improve | p |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for arch in ARCHITECTURES:
        for city in CITIES:
            cs = results['cell_results'].get(arch, {}).get(city)
            if not cs:
                continue
            van = cs['comparisons'].get('vanilla', {}).get('aurc_shadow', {})
            if not van:
                continue
            lines.append(f'| {arch} | {city} | '
                         f'{_fmt(van["c4clean_mean"])} | {_fmt(van["method_mean"])} | '
                         f'{_fmt(van["mean_delta"])} | '
                         f'[{_fmt(van["ci_lo"])}, {_fmt(van["ci_hi"])}] | '
                         f'{van["n_improve"]}/{van["n_valid"]} | '
                         f'{_fmt(van["p_two_sided"])} |')
    lines.append('')

    # ---- Per-cell C4 vs Best-DA-cell-winner ----
    lines.append('## 5. Per-cell detail: C4-clean vs Best-DA-on-cell')
    lines.append('')
    lines.append('"Best-DA-on-cell" = the single adaptation method with lowest '
                 'cell-mean AURC on that cell. This is the strongest single-method '
                 'baseline.')
    lines.append('')
    lines.append('| Arch | City | Best-DA winner | C4 AURC | BestDA AURC | ΔAURC | 95% CI |')
    lines.append('|---|---|---|---|---|---|---|')
    for arch in ARCHITECTURES:
        for city in CITIES:
            cs = results['cell_results'].get(arch, {}).get(city)
            if not cs:
                continue
            bd = cs['comparisons'].get('bestda_cell_winner', {}).get('aurc_shadow', {})
            if not bd:
                continue
            winner = cs.get('bestda_cell_method', '?')
            lines.append(f'| {arch} | {city} | {winner} | '
                         f'{_fmt(bd["c4clean_mean"])} | {_fmt(bd["method_mean"])} | '
                         f'{_fmt(bd["mean_delta"])} | '
                         f'[{_fmt(bd["ci_lo"])}, {_fmt(bd["ci_hi"])}] |')
    lines.append('')

    # ---- Method-by-method full table ----
    lines.append('## 6. Population-level deltas by method')
    lines.append('')
    lines.append('Mean ΔAURC across 9 cells (negative = C4-clean better).')
    lines.append('"n improve" = number of cells (out of 9) where C4-clean reduces AURC.')
    lines.append('')
    lines.append('| Method | Mean ΔAURC | n improve | Wilcoxon p (less) |')
    lines.append('|---|---|---|---|')
    for competitor in ['vanilla'] + ADAPTATION_METHODS + ['meanda',
                       'bestda_cell_winner', 'bestda_per_image_oracle']:
        if competitor not in pop:
            continue
        w = pop[competitor]['wilcoxon_aurc']
        mean_d = w.get('mean_delta_across_cells', float('nan'))
        n_neg = w.get('n_negative', '—')
        n_c = w.get('n_cells', 0)
        p_less = w.get('wilcoxon_p_less', float('nan'))
        lines.append(f'| {competitor} | {_fmt(mean_d)} | {n_neg}/{n_c} | '
                     f'{_fmt(p_less)} |')
    lines.append('')

    # ---- Notes ----
    lines.append('## Notes on interpretation')
    lines.append('')
    lines.append('- **AURC_shadow** is the area under the risk-coverage curve '
                 'on ground-truth shadow pixels. Lower means the model is '
                 'more confidently correct on shadow.')
    lines.append('- **Bootstrap CIs** are image-level cluster bootstrap '
                 '(B=10,000); pixel correlation within an image is absorbed.')
    lines.append('- **Wilcoxon test** runs on n=9 cell-mean deltas. Smallest '
                 'achievable p ≈ 0.002 (one-sided, n=9). With n=3 per '
                 'architecture, per-architecture Wilcoxon would be '
                 'underpowered (min p ~0.25), so per-architecture results '
                 'are reported descriptively only.')
    lines.append('- **Best-DA-on-cell** picks the single best adaptation '
                 'method per cell; this is post-hoc selection used as a '
                 'strong oracle baseline. C4-clean beating BestDA-on-cell '
                 'means we can\'t be beaten even by an oracle that knows '
                 'which DA method to pick per city.')
    lines.append('')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()