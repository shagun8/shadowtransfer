#!/usr/bin/env python3
"""
Task-Relevant Feature Distribution Diagnostics
================================================
For each city × resolution, computes a panel of features that mechanically
affect shadow detection mIoU, then measures train-vs-test divergence using
KS distance and normalized Wasserstein distance.

The output is a single comparison table: rows = cities, columns = features,
cells = KS-D (and Wasserstein in normalized units). If Phoenix's row is
systematically larger than Chicago's and Miami's, that's structural evidence
of within-city heterogeneity.

Features computed per image:
  Label-derived:
    - shadow_coverage: fraction of shadow pixels
    - num_objects: number of connected shadow components
    - mean_obj_size: mean shadow object area (pixels)
    - median_obj_size: median shadow object area (pixels)
    - small_obj_frac: fraction of objects < 200px (hurts mIoU disproportionately)
    - boundary_density: shadow perimeter pixels / total pixels
  
  Image-derived:
    - intensity_mean: mean grayscale intensity (0-255)
    - intensity_std: std of grayscale intensity
    - r_mean, g_mean, b_mean: per-channel means
    - saturation_mean: mean HSV saturation
    - contrast: Michelson contrast (max-min)/(max+min) on grayscale
    - shadow_intensity_mean: mean intensity inside shadow regions
    - nonshadow_intensity_mean: mean intensity outside shadow regions
    - intensity_ratio: shadow_intensity / nonshadow_intensity (shadow darkness)

Usage:
  python feature_diagnostics.py \
      --base_data_root /path/to/Final_data_test \
      --resolutions highres midres \
      --output_dir ./feature_diagnostics_output
"""

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats
from scipy.ndimage import label as ndimage_label, binary_dilation, binary_erosion
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore', category=UserWarning)

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def list_files(directory):
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(VALID_EXTENSIONS)])


# ─────────────────────────────────────────────────────────────
# Feature extraction per image
# ─────────────────────────────────────────────────────────────

def compute_image_features(img_path, mask_path):
    """
    Compute all task-relevant features for a single image + mask pair.
    Returns a dict of scalar features, or None if files are missing.
    """
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None

    # Load image and mask
    img = np.array(Image.open(img_path).convert('RGB'))
    mask_raw = np.array(Image.open(mask_path).convert('L'))
    mask = (mask_raw > 127).astype(np.uint8)

    h, w = mask.shape
    total_pixels = h * w
    shadow_pixels = mask.sum()

    features = {}

    # ── Label-derived features ──

    # Shadow coverage
    features['shadow_coverage'] = shadow_pixels / total_pixels

    # Connected components
    labeled, num_features = ndimage_label(mask)
    features['num_objects'] = num_features

    if num_features > 0:
        obj_sizes = np.array([np.sum(labeled == i) for i in range(1, num_features + 1)])
        features['mean_obj_size'] = float(obj_sizes.mean())
        features['median_obj_size'] = float(np.median(obj_sizes))
        features['small_obj_frac'] = float(np.sum(obj_sizes < 200) / num_features)
    else:
        features['mean_obj_size'] = 0.0
        features['median_obj_size'] = 0.0
        features['small_obj_frac'] = 0.0

    # Boundary density: perimeter pixels / total pixels
    if shadow_pixels > 0:
        eroded = binary_erosion(mask, structure=np.ones((3, 3)))
        boundary = mask.astype(bool) & ~eroded
        features['boundary_density'] = float(boundary.sum()) / total_pixels
    else:
        features['boundary_density'] = 0.0

    # ── Image-derived features ──

    gray = img.mean(axis=2)  # Simple grayscale

    features['intensity_mean'] = float(gray.mean())
    features['intensity_std'] = float(gray.std())

    # Per-channel
    features['r_mean'] = float(img[:, :, 0].mean())
    features['g_mean'] = float(img[:, :, 1].mean())
    features['b_mean'] = float(img[:, :, 2].mean())

    # Saturation (from HSV)
    img_float = img.astype(np.float32) / 255.0
    cmax = img_float.max(axis=2)
    cmin = img_float.min(axis=2)
    delta = cmax - cmin
    sat = np.where(cmax > 0, delta / cmax, 0)
    features['saturation_mean'] = float(sat.mean())

    # Contrast: Michelson on grayscale
    gmax, gmin = gray.max(), gray.min()
    if gmax + gmin > 0:
        features['contrast'] = float((gmax - gmin) / (gmax + gmin))
    else:
        features['contrast'] = 0.0

    # Shadow vs non-shadow intensity
    if shadow_pixels > 0 and shadow_pixels < total_pixels:
        shadow_mask_bool = mask.astype(bool)
        features['shadow_intensity_mean'] = float(gray[shadow_mask_bool].mean())
        features['nonshadow_intensity_mean'] = float(gray[~shadow_mask_bool].mean())
        ns_mean = features['nonshadow_intensity_mean']
        if ns_mean > 0:
            features['intensity_ratio'] = features['shadow_intensity_mean'] / ns_mean
        else:
            features['intensity_ratio'] = 1.0
    elif shadow_pixels == 0:
        features['shadow_intensity_mean'] = 0.0
        features['nonshadow_intensity_mean'] = float(gray.mean())
        features['intensity_ratio'] = 0.0
    else:
        features['shadow_intensity_mean'] = float(gray.mean())
        features['nonshadow_intensity_mean'] = 0.0
        features['intensity_ratio'] = 1.0

    return features


def extract_all_features(city_root, split_name):
    """
    Extract features for all images in a split.
    Returns list of feature dicts and list of filenames.
    """
    img_dir = os.path.join(city_root, split_name, 'images')
    mask_dir = os.path.join(city_root, split_name, 'masks')

    if not os.path.exists(img_dir):
        return [], []

    filenames = list_files(img_dir)
    all_features = []
    valid_filenames = []

    for i, fname in enumerate(filenames):
        img_path = os.path.join(img_dir, fname)
        # Try matching mask filename
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            base = os.path.splitext(fname)[0]
            for ext in VALID_EXTENSIONS:
                alt = os.path.join(mask_dir, base + ext)
                if os.path.exists(alt):
                    mask_path = alt
                    break

        feats = compute_image_features(img_path, mask_path)
        if feats is not None:
            all_features.append(feats)
            valid_filenames.append(fname)

        if (i + 1) % 100 == 0:
            print(f"      {i+1}/{len(filenames)} images processed")

    return all_features, valid_filenames


def features_to_arrays(feature_list):
    """Convert list of feature dicts to dict of numpy arrays."""
    if not feature_list:
        return {}
    keys = feature_list[0].keys()
    return {k: np.array([f[k] for f in feature_list]) for k in keys}


# ─────────────────────────────────────────────────────────────
# Divergence statistics
# ─────────────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    'shadow_coverage': 'Shadow Coverage',
    'num_objects': '# Objects',
    'mean_obj_size': 'Mean Obj Size',
    'median_obj_size': 'Median Obj Size',
    'small_obj_frac': 'Small Obj Frac',
    'boundary_density': 'Boundary Density',
    'intensity_mean': 'Intensity μ',
    'intensity_std': 'Intensity σ',
    'r_mean': 'R Mean',
    'g_mean': 'G Mean',
    'b_mean': 'B Mean',
    'saturation_mean': 'Saturation μ',
    'contrast': 'Contrast',
    'shadow_intensity_mean': 'Shadow Int μ',
    'nonshadow_intensity_mean': 'Non-Shadow Int μ',
    'intensity_ratio': 'Intensity Ratio',
}

# Features ordered by importance for shadow detection mIoU
FEATURE_ORDER = [
    'shadow_coverage', 'num_objects', 'mean_obj_size', 'median_obj_size',
    'small_obj_frac', 'boundary_density',
    'intensity_mean', 'intensity_std', 'r_mean', 'g_mean', 'b_mean',
    'saturation_mean', 'contrast',
    'shadow_intensity_mean', 'nonshadow_intensity_mean', 'intensity_ratio',
]


def compute_divergences(train_arrays, test_arrays, feature_name):
    """
    Compute KS distance and normalized 1-Wasserstein distance
    between train and test for a single feature.
    
    Returns dict with ks_d, ks_p, wasserstein_norm, train_mean, test_mean, 
    train_std, test_std, relative_shift.
    """
    a = train_arrays[feature_name]
    b = test_arrays[feature_name]

    if len(a) < 2 or len(b) < 2:
        return None

    # KS test
    ks_stat, ks_p = stats.ks_2samp(a, b)

    # 1-Wasserstein distance, normalized by train std
    w_dist = wasserstein_distance(a, b)
    train_std = a.std()
    if train_std > 0:
        w_norm = w_dist / train_std
    else:
        w_norm = 0.0

    # Descriptive stats
    train_mean = a.mean()
    test_mean = b.mean()
    if train_mean != 0:
        relative_shift = (test_mean - train_mean) / abs(train_mean)
    else:
        relative_shift = 0.0

    return {
        'ks_d': float(ks_stat),
        'ks_p': float(ks_p),
        'wasserstein_raw': float(w_dist),
        'wasserstein_norm': float(w_norm),
        'train_mean': float(train_mean),
        'test_mean': float(test_mean),
        'train_std': float(train_std),
        'test_std': float(b.std()),
        'train_median': float(np.median(a)),
        'test_median': float(np.median(b)),
        'relative_shift': float(relative_shift),
        'n_train': len(a),
        'n_test': len(b),
    }


# ─────────────────────────────────────────────────────────────
# Comparison table and plotting
# ─────────────────────────────────────────────────────────────

def print_comparison_table(all_divergences, metric='ks_d'):
    """
    Print a single table: rows = city×resolution, columns = features.
    Cells = KS-D (or wasserstein_norm).
    """
    metric_label = 'KS-D' if metric == 'ks_d' else 'W₁/σ'

    # Collect all city×res keys
    keys = list(all_divergences.keys())
    if not keys:
        print("No divergence data to display.")
        return

    # Header
    features = FEATURE_ORDER
    short_names = [FEATURE_DISPLAY_NAMES.get(f, f)[:14] for f in features]

    header = f"{'City/Res':<20}"
    for sn in short_names:
        header += f" {sn:>14}"
    
    print("\n" + "=" * (20 + 15 * len(short_names)))
    print(f"CROSS-CITY COMPARISON TABLE — {metric_label} (train vs test)")
    print("  Higher values = larger distribution mismatch.")
    print("  ⚠ marks cells where KS p < 0.05")
    print("=" * (20 + 15 * len(short_names)))
    print(header)
    print("-" * (20 + 15 * len(short_names)))

    for key in sorted(keys):
        city, res = key
        row = f"{city}/{res:<12}"
        divs = all_divergences[key]
        for feat in features:
            if feat in divs and divs[feat] is not None:
                val = divs[feat][metric]
                flag = '*' if divs[feat]['ks_p'] < 0.05 else ' '
                row += f" {val:>13.4f}{flag}"
            else:
                row += f" {'—':>14}"
        print(row)

    print("=" * (20 + 15 * len(short_names)))


def print_detailed_table(all_divergences, output_dir):
    """Print and save a detailed per-feature table with both metrics + descriptive stats."""
    lines = []
    lines.append("=" * 130)
    lines.append("DETAILED FEATURE DIVERGENCE TABLE — Train vs Test within each city")
    lines.append("  KS-D: Kolmogorov-Smirnov distance (0-1, max CDF gap)")
    lines.append("  W₁/σ: 1-Wasserstein normalized by train σ (test shifted by X std-devs)")
    lines.append("  Rel%: (test_mean - train_mean) / |train_mean| × 100")
    lines.append("=" * 130)

    for key in sorted(all_divergences.keys()):
        city, res = key
        divs = all_divergences[key]

        lines.append(f"\n{'─' * 130}")
        lines.append(f"  {city.upper()} ({res})")
        lines.append(f"{'─' * 130}")
        header = (f"  {'Feature':<22} {'Train μ':>10} {'Test μ':>10} {'Train σ':>10} "
                  f"{'Test σ':>10} {'Rel%':>8} {'KS-D':>8} {'KS-p':>10} "
                  f"{'W₁/σ':>8} {'Flag':>6}")
        lines.append(header)
        lines.append("  " + "-" * 126)

        for feat in FEATURE_ORDER:
            if feat not in divs or divs[feat] is None:
                continue
            d = divs[feat]
            flag = '⚠️' if d['ks_p'] < 0.05 else '✓'
            line = (f"  {FEATURE_DISPLAY_NAMES.get(feat, feat):<22} "
                    f"{d['train_mean']:>10.4f} {d['test_mean']:>10.4f} "
                    f"{d['train_std']:>10.4f} {d['test_std']:>10.4f} "
                    f"{d['relative_shift']*100:>7.1f}% "
                    f"{d['ks_d']:>8.4f} {d['ks_p']:>10.4f} "
                    f"{d['wasserstein_norm']:>8.4f} {flag:>6}")
            lines.append(line)

    report = "\n".join(lines)
    print(report)

    path = os.path.join(output_dir, 'feature_divergence_detailed.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f"\nSaved to: {path}")
    return report


def plot_feature_heatmap(all_divergences, metric, output_dir):
    """
    Heatmap: rows = city×resolution, columns = features, cells = divergence value.
    """
    keys = sorted(all_divergences.keys())
    features = FEATURE_ORDER
    
    # Build matrix
    matrix = np.zeros((len(keys), len(features)))
    row_labels = []
    for i, key in enumerate(keys):
        city, res = key
        row_labels.append(f"{city.title()} ({res})")
        for j, feat in enumerate(features):
            if feat in all_divergences[key] and all_divergences[key][feat] is not None:
                matrix[i, j] = all_divergences[key][feat][metric]
            else:
                matrix[i, j] = np.nan

    col_labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]
    metric_label = 'KS Distance' if metric == 'ks_d' else 'Wasserstein / σ'

    fig, ax = plt.subplots(figsize=(20, 4 + 0.5 * len(keys)))

    # Use a diverging colormap centered near 0 for wasserstein, sequential for KS
    cmap = 'YlOrRd'
    vmax = np.nanmax(matrix) if np.nanmax(matrix) > 0 else 1.0
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Annotate cells
    for i in range(len(keys)):
        for j in range(len(features)):
            val = matrix[i, j]
            if not np.isnan(val):
                # Color text based on background
                text_color = 'white' if val > vmax * 0.6 else 'black'
                # Add asterisk if KS significant
                feat = features[j]
                key = keys[i]
                sig = ''
                if (feat in all_divergences[key] and 
                    all_divergences[key][feat] is not None and
                    all_divergences[key][feat]['ks_p'] < 0.05):
                    sig = '*'
                ax.text(j, i, f'{val:.3f}{sig}', ha='center', va='center',
                       fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, label=metric_label, shrink=0.8)
    ax.set_title(f'Train vs Test Divergence — {metric_label}\n(* = KS p < 0.05)', fontsize=13)

    plt.tight_layout()
    path = os.path.join(output_dir, f'heatmap_{metric}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_distributions(train_arrays, test_arrays, city, resolution, output_dir):
    """Plot side-by-side distributions of key features for one city."""
    # Select most important features for plotting
    plot_features = [
        'shadow_coverage', 'mean_obj_size', 'num_objects', 'small_obj_frac',
        'intensity_mean', 'intensity_std', 'saturation_mean',
        'shadow_intensity_mean', 'nonshadow_intensity_mean', 'intensity_ratio',
        'boundary_density', 'contrast',
    ]

    n_feats = len(plot_features)
    n_cols = 4
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()
    fig.suptitle(f'{city.title()} ({resolution}) — Train vs Test Feature Distributions',
                 fontsize=14, fontweight='bold', y=1.01)

    for idx, feat in enumerate(plot_features):
        ax = axes[idx]
        if feat not in train_arrays or feat not in test_arrays:
            ax.set_visible(False)
            continue

        train_data = train_arrays[feat]
        test_data = test_arrays[feat]

        # Compute stats for annotation
        ks_stat, ks_p = stats.ks_2samp(train_data, test_data)

        ax.hist(train_data, bins=30, alpha=0.5, label=f'Train (n={len(train_data)})',
                color='#2196F3', density=True)
        ax.hist(test_data, bins=30, alpha=0.5, label=f'Test (n={len(test_data)})',
                color='#FF5722', density=True)

        # Add vertical lines for medians
        ax.axvline(np.median(train_data), color='#1565C0', linestyle='--',
                   linewidth=1.5, alpha=0.8)
        ax.axvline(np.median(test_data), color='#D84315', linestyle='--',
                   linewidth=1.5, alpha=0.8)

        display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        sig_marker = ' ⚠️' if ks_p < 0.05 else ''
        ax.set_title(f'{display_name}\nKS={ks_stat:.3f}, p={ks_p:.3f}{sig_marker}',
                     fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylabel('Density')

    # Hide unused axes
    for idx in range(len(plot_features), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, f'distributions_{city}_{resolution}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_city_comparison_bars(all_divergences, output_dir):
    """
    Bar chart comparing KS-D across cities for each feature.
    This is the money plot — if Phoenix bars are systematically taller,
    the within-city heterogeneity is the answer.
    """
    # Group by resolution
    res_groups = defaultdict(dict)
    for (city, res), divs in all_divergences.items():
        res_groups[res][city] = divs

    for res, city_divs in res_groups.items():
        cities = sorted(city_divs.keys())
        features = FEATURE_ORDER
        n_features = len(features)
        n_cities = len(cities)

        fig, ax = plt.subplots(figsize=(20, 6))

        x = np.arange(n_features)
        width = 0.8 / n_cities
        colors = {'chicago': '#2196F3', 'miami': '#4CAF50', 'phoenix': '#FF5722'}

        for i, city in enumerate(cities):
            vals = []
            for feat in features:
                if feat in city_divs[city] and city_divs[city][feat] is not None:
                    vals.append(city_divs[city][feat]['ks_d'])
                else:
                    vals.append(0)

            offset = (i - n_cities / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=city.title(),
                         color=colors.get(city, 'gray'), alpha=0.8,
                         edgecolor='white', linewidth=0.5)

            # Mark significant bars
            for j, (bar, feat) in enumerate(zip(bars, features)):
                if (feat in city_divs[city] and city_divs[city][feat] is not None
                        and city_divs[city][feat]['ks_p'] < 0.05):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                           '*', ha='center', va='bottom', fontsize=12, fontweight='bold',
                           color=colors.get(city, 'gray'))

        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_DISPLAY_NAMES.get(f, f) for f in features],
                           rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('KS Distance (train vs test)')
        ax.set_title(f'Per-Feature Train-vs-Test Divergence by City — {res}\n'
                     f'(* = p < 0.05. Taller bars = larger within-city mismatch)',
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

        plt.tight_layout()
        path = os.path.join(output_dir, f'city_comparison_{res}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {path}")


def generate_summary(all_divergences, output_dir):
    """
    Generate a compact summary: which city has the most divergent features?
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("SUMMARY: Per-City Divergence Ranking")
    lines.append("  Mean KS-D across all features (higher = more mismatched splits)")
    lines.append("=" * 80)

    city_scores = defaultdict(list)
    for (city, res), divs in all_divergences.items():
        for feat in FEATURE_ORDER:
            if feat in divs and divs[feat] is not None:
                city_scores[(city, res)].append(divs[feat]['ks_d'])

    lines.append(f"\n  {'City/Res':<25} {'Mean KS-D':>10} {'Max KS-D':>10} {'# Sig':>8} {'# Feats':>8}")
    lines.append("  " + "-" * 65)

    for key in sorted(city_scores.keys()):
        city, res = key
        scores = city_scores[key]
        divs = all_divergences[key]
        n_sig = sum(1 for feat in FEATURE_ORDER
                    if feat in divs and divs[feat] is not None and divs[feat]['ks_p'] < 0.05)
        lines.append(f"  {city}/{res:<18} {np.mean(scores):>10.4f} {np.max(scores):>10.4f} "
                     f"{n_sig:>8} {len(scores):>8}")

    # Phoenix vs others comparison
    lines.append(f"\n  INTERPRETATION:")
    phoenix_means = {res: np.mean(city_scores[('phoenix', res)])
                     for res in ['highres', 'midres']
                     if ('phoenix', res) in city_scores}
    for res, phx_mean in phoenix_means.items():
        others = [(c, np.mean(city_scores[(c, res)]))
                  for c in ['chicago', 'miami']
                  if (c, res) in city_scores]
        others_mean = np.mean([v for _, v in others])
        if phx_mean > others_mean * 1.3:
            lines.append(f"  ⚠️ Phoenix ({res}): mean KS-D = {phx_mean:.4f}, "
                        f"others = {others_mean:.4f} — Phoenix has {phx_mean/others_mean:.1f}× "
                        f"larger within-city mismatch")
        else:
            lines.append(f"  ✓ Phoenix ({res}): mean KS-D = {phx_mean:.4f}, "
                        f"others = {others_mean:.4f} — no systematic excess divergence")

    summary = "\n".join(lines)
    print(summary)

    path = os.path.join(output_dir, 'divergence_summary.txt')
    with open(path, 'w') as f:
        f.write(summary)
    return summary


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Task-relevant feature distribution diagnostics')
    parser.add_argument('--base_data_root', type=str, required=True)
    parser.add_argument('--resolutions', nargs='+', default=['highres', 'midres'])
    parser.add_argument('--cities', nargs='+', default=['chicago', 'miami', 'phoenix'])
    parser.add_argument('--output_dir', type=str, default='./feature_diagnostics_output')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                        help='Which splits to compare (default: train test)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    split_a, split_b = args.splits[0], args.splits[1]
    print(f"Comparing: {split_a} vs {split_b}")

    all_divergences = {}  # (city, res) -> {feature_name: divergence_dict}
    all_feature_cache = {}  # (city, res) -> {split: arrays}

    for resolution in args.resolutions:
        for city in args.cities:
            print(f"\n{'=' * 60}")
            print(f"Processing: {city.title()} — {resolution}")
            print(f"{'=' * 60}")

            city_root = os.path.join(args.base_data_root, city, resolution)
            if not os.path.exists(city_root):
                print(f"  SKIP: {city_root} does not exist")
                continue

            # Extract features for both splits
            split_arrays = {}
            for split_name in [split_a, split_b]:
                print(f"\n  Extracting features for {split_name}...")
                features_list, fnames = extract_all_features(city_root, split_name)
                if not features_list:
                    print(f"    No images found for {split_name}")
                    continue
                arrays = features_to_arrays(features_list)
                split_arrays[split_name] = arrays
                print(f"    {len(features_list)} images, "
                      f"shadow_coverage: mean={arrays['shadow_coverage'].mean():.4f}, "
                      f"median={np.median(arrays['shadow_coverage']):.4f}")

            if split_a not in split_arrays or split_b not in split_arrays:
                print(f"  SKIP: missing split data")
                continue

            all_feature_cache[(city, resolution)] = split_arrays

            # Compute divergences
            print(f"\n  Computing divergences ({split_a} vs {split_b})...")
            divs = {}
            for feat in FEATURE_ORDER:
                if feat in split_arrays[split_a] and feat in split_arrays[split_b]:
                    divs[feat] = compute_divergences(
                        split_arrays[split_a], split_arrays[split_b], feat)
                    if divs[feat] is not None:
                        flag = '⚠️' if divs[feat]['ks_p'] < 0.05 else '✓'
                        print(f"    {FEATURE_DISPLAY_NAMES.get(feat, feat):<22} "
                              f"KS={divs[feat]['ks_d']:.4f} "
                              f"W₁/σ={divs[feat]['wasserstein_norm']:.4f} "
                              f"shift={divs[feat]['relative_shift']*100:+.1f}% {flag}")

            all_divergences[(city, resolution)] = divs

            # Per-city distribution plots
            plot_feature_distributions(
                split_arrays[split_a], split_arrays[split_b],
                city, resolution, args.output_dir)

    # Cross-city comparison outputs
    if all_divergences:
        print("\n\n" + "=" * 80)
        print("CROSS-CITY COMPARISON")
        print("=" * 80)

        print_comparison_table(all_divergences, metric='ks_d')
        print_comparison_table(all_divergences, metric='wasserstein_norm')
        print_detailed_table(all_divergences, args.output_dir)
        plot_feature_heatmap(all_divergences, 'ks_d', args.output_dir)
        plot_feature_heatmap(all_divergences, 'wasserstein_norm', args.output_dir)
        plot_city_comparison_bars(all_divergences, args.output_dir)
        generate_summary(all_divergences, args.output_dir)

    # Save raw data
    raw = {}
    for (city, res), divs in all_divergences.items():
        raw[f"{city}_{res}"] = divs
    json_path = os.path.join(args.output_dir, 'feature_divergences_raw.json')
    with open(json_path, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to: {json_path}")


if __name__ == '__main__':
    main()