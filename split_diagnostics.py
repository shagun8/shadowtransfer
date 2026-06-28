#!/usr/bin/env python3
"""
Split Distribution Diagnostics for ShadowTransfer Benchmark
============================================================
Tests whether train/val/test splits within each city have similar distributions.

Tests implemented:
  1. Label-statistics KS test (shadow coverage %, mean object size, object count)
  2. Domain-classifier covariate-shift test (logistic regression on ResNet-50 features)
  3. MMD on deep features (bootstrap MMD with RBF kernel)
  4. Spatial coverage analysis (requires --geo_metadata_path)

Usage:
  python split_diagnostics.py \
      --base_data_root /path/to/Final_data_test \
      --resolutions highres midres \
      --output_dir ./split_diagnostics_output \
      [--geo_metadata_path /path/to/geo_metadata.json] \
      [--feature_extractor resnet50]  # or dinov2, clip
      [--batch_size 32] \
      [--device cuda]
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
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

warnings.filterwarnings('ignore', category=UserWarning)


def load_geo_metadata(path):
    """
    Load geo metadata from the ShadowTransfer mapping.json format.
    
    Expected format: a JSON list of objects, each with at least:
        original_filename, city, resolution, center_lat, center_lon
    
    Returns:
        dict mapping original_filename -> {lat, lon, city, resolution, ...}
    """
    with open(path, 'r') as f:
        raw = json.load(f)
    
    # Handle both list-of-dicts and dict-of-dicts formats
    if isinstance(raw, list):
        lookup = {}
        for entry in raw:
            fname = entry.get('original_filename')
            if fname is None:
                continue
            lookup[fname] = {
                'lat': entry.get('center_lat'),
                'lon': entry.get('center_lon'),
                'city': entry.get('city'),
                'resolution': entry.get('resolution'),
                'type': entry.get('type'),
                'image_type': entry.get('image_type'),
                'pair_id': entry.get('pair_id'),
                'tile_name': entry.get('tile_name'),
            }
        return lookup
    elif isinstance(raw, dict):
        # Already keyed by filename
        return raw
    else:
        raise ValueError(f"Unexpected metadata format: {type(raw)}")

# ─────────────────────────────────────────────────────────────
# Utility: load images/masks from a split directory
# ─────────────────────────────────────────────────────────────

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def list_files(directory):
    """List image files in a directory, sorted."""
    if not os.path.exists(directory):
        return []
    return sorted([f for f in os.listdir(directory) if f.lower().endswith(VALID_EXTENSIONS)])


def load_mask_as_binary(path):
    """Load a mask file and return a binary numpy array (1=shadow, 0=bg)."""
    mask = np.array(Image.open(path).convert('L'))
    return (mask > 127).astype(np.uint8)


def connected_components_stats(binary_mask):
    """
    Compute connected-component statistics on a binary mask.
    Returns (num_objects, mean_object_area, object_areas_list).
    Uses scipy.ndimage to avoid OpenCV dependency.
    """
    from scipy.ndimage import label as ndimage_label
    labeled, num_features = ndimage_label(binary_mask)
    if num_features == 0:
        return 0, 0.0, []
    areas = []
    for i in range(1, num_features + 1):
        areas.append(np.sum(labeled == i))
    return num_features, float(np.mean(areas)), areas


# ─────────────────────────────────────────────────────────────
# TEST 1: Label-statistics KS tests
# ─────────────────────────────────────────────────────────────

def compute_label_stats(mask_dir, filenames):
    """
    For each image, compute:
      - shadow_coverage: fraction of shadow pixels
      - num_objects: number of connected shadow components
      - mean_object_area: mean area of connected components (pixels)
    Returns dict of arrays.
    """
    coverages = []
    num_objects_list = []
    mean_areas = []

    for fname in filenames:
        mask_path = os.path.join(mask_dir, fname)
        if not os.path.exists(mask_path):
            # Try common extension swaps
            base = os.path.splitext(fname)[0]
            for ext in VALID_EXTENSIONS:
                alt = os.path.join(mask_dir, base + ext)
                if os.path.exists(alt):
                    mask_path = alt
                    break
            else:
                print(f"  Warning: mask not found for {fname}, skipping")
                continue

        mask = load_mask_as_binary(mask_path)
        total_pixels = mask.shape[0] * mask.shape[1]
        shadow_pixels = mask.sum()
        coverages.append(shadow_pixels / total_pixels)

        n_obj, mean_area, _ = connected_components_stats(mask)
        num_objects_list.append(n_obj)
        mean_areas.append(mean_area)

    return {
        'shadow_coverage': np.array(coverages),
        'num_objects': np.array(num_objects_list, dtype=float),
        'mean_object_area': np.array(mean_areas),
    }


def run_ks_tests(city_data, city, resolution):
    """
    Run two-sample KS tests between every split pair for each label statistic.
    city_data: dict  split_name -> label_stats_dict
    Returns list of result dicts.
    """
    splits = list(city_data.keys())
    metrics = ['shadow_coverage', 'num_objects', 'mean_object_area']
    results = []

    for metric in metrics:
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                s1, s2 = splits[i], splits[j]
                a = city_data[s1][metric]
                b = city_data[s2][metric]
                if len(a) < 2 or len(b) < 2:
                    continue
                ks_stat, p_value = stats.ks_2samp(a, b)
                results.append({
                    'city': city,
                    'resolution': resolution,
                    'metric': metric,
                    'split_pair': f'{s1} vs {s2}',
                    'n1': len(a),
                    'n2': len(b),
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'flag': '⚠️' if p_value < 0.05 else '✓',
                })
    return results


# ─────────────────────────────────────────────────────────────
# TEST 2: Domain-classifier covariate-shift test
# ─────────────────────────────────────────────────────────────

class ImageOnlyDataset(Dataset):
    """Simple dataset that loads images and returns features-ready tensors."""

    def __init__(self, img_dir, filenames, transform):
        self.img_dir = img_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        img = Image.open(path).convert('RGB')
        return self.transform(img)


def extract_features(img_dir, filenames, model, transform, device, batch_size=32):
    """Extract feature vectors from a pretrained model (global avg pool)."""
    dataset = ImageOnlyDataset(img_dir, filenames, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_feats = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feats = model(batch)
            if feats.dim() > 2:
                feats = feats.mean(dim=[2, 3])  # global avg pool if spatial
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def build_feature_extractor(name='resnet50', device='cpu'):
    """Build a feature extractor (removes final FC layer)."""
    if name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final FC layer, keep avgpool
        model = nn.Sequential(*list(model.children())[:-1])  # outputs (B, 2048, 1, 1)
    else:
        raise ValueError(f"Unsupported feature extractor: {name}. Use 'resnet50'.")

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return model, transform


def domain_classifier_test(feats_a, feats_b, n_repeats=5):
    """
    Train a logistic regression to distinguish split A from split B.
    Returns mean ROC-AUC across repeated random 50/50 train/test splits.
    AUC ≈ 0.5 means no detectable shift; AUC > 0.7 is notable; > 0.8 is strong.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([feats_a, feats_b], axis=0)
    y = np.concatenate([np.zeros(len(feats_a)), np.ones(len(feats_b))])

    aucs = []
    splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.3, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, y[train_idx])
        probs = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y[test_idx], probs))

    return float(np.mean(aucs)), float(np.std(aucs))


# ─────────────────────────────────────────────────────────────
# TEST 3: MMD on deep features
# ─────────────────────────────────────────────────────────────

def compute_mmd_rbf(X, Y, gamma=None, n_bootstrap=1000):
    """
    Compute MMD² with RBF kernel between X and Y, with bootstrap p-value.
    If gamma is None, uses median heuristic.
    """
    XY = np.concatenate([X, Y], axis=0)
    n_x, n_y = len(X), len(Y)

    # Median heuristic for bandwidth
    if gamma is None:
        dists = cdist(XY, XY, metric='sqeuclidean')
        median_dist = np.median(dists[np.triu_indices(len(XY), k=1)])
        if median_dist == 0:
            median_dist = 1.0
        gamma = 1.0 / median_dist

    def mmd_statistic(idx_x, idx_y, all_data):
        Kxx = np.exp(-gamma * cdist(all_data[idx_x], all_data[idx_x], 'sqeuclidean'))
        Kyy = np.exp(-gamma * cdist(all_data[idx_y], all_data[idx_y], 'sqeuclidean'))
        Kxy = np.exp(-gamma * cdist(all_data[idx_x], all_data[idx_y], 'sqeuclidean'))
        # Unbiased estimator
        np.fill_diagonal(Kxx, 0)
        np.fill_diagonal(Kyy, 0)
        nx, ny = len(idx_x), len(idx_y)
        mmd2 = (Kxx.sum() / (nx * (nx - 1)) +
                Kyy.sum() / (ny * (ny - 1)) -
                2 * Kxy.mean())
        return mmd2

    idx_x = np.arange(n_x)
    idx_y = np.arange(n_x, n_x + n_y)

    observed_mmd = mmd_statistic(idx_x, idx_y, XY)

    # Permutation test
    rng = np.random.RandomState(42)
    count_greater = 0
    for _ in range(n_bootstrap):
        perm = rng.permutation(n_x + n_y)
        perm_mmd = mmd_statistic(perm[:n_x], perm[n_x:], XY)
        if perm_mmd >= observed_mmd:
            count_greater += 1

    p_value = (count_greater + 1) / (n_bootstrap + 1)
    return observed_mmd, p_value


def mmd_test_subsample(feats_a, feats_b, max_samples=200, n_bootstrap=500):
    """
    Run MMD with subsampling if datasets are large (MMD is O(n²)).
    """
    rng = np.random.RandomState(42)
    if len(feats_a) > max_samples:
        idx = rng.choice(len(feats_a), max_samples, replace=False)
        feats_a = feats_a[idx]
    if len(feats_b) > max_samples:
        idx = rng.choice(len(feats_b), max_samples, replace=False)
        feats_b = feats_b[idx]

    return compute_mmd_rbf(feats_a, feats_b, n_bootstrap=n_bootstrap)


# ─────────────────────────────────────────────────────────────
# TEST 4: Spatial coverage analysis
# ─────────────────────────────────────────────────────────────

def spatial_coverage_analysis(geo_metadata, filenames_by_split, city, resolution):
    """
    Analyze spatial distribution of splits using geocoordinates.
    Also checks image_type distribution (random vs paired) per split.
    Returns stats dict and data for plotting.
    """
    results = {}
    coords_by_split = {}

    for split_name, fnames in filenames_by_split.items():
        lats, lons = [], []
        image_types = defaultdict(int)
        matched, unmatched = 0, 0
        for f in fnames:
            if f in geo_metadata:
                entry = geo_metadata[f]
                if entry.get('lat') is not None and entry.get('lon') is not None:
                    lats.append(entry['lat'])
                    lons.append(entry['lon'])
                if entry.get('image_type'):
                    image_types[entry['image_type']] += 1
                matched += 1
            else:
                unmatched += 1
        
        if unmatched > 0:
            print(f"    {split_name}: {matched} matched, {unmatched} unmatched in metadata")
        
        if len(lats) > 0:
            coords_by_split[split_name] = {
                'lat': np.array(lats),
                'lon': np.array(lons),
            }
        
        results[split_name] = {
            'n_with_coords': len(lats),
            'n_total': len(fnames),
            'image_type_counts': dict(image_types),
        }
        if len(lats) > 0:
            lat_arr, lon_arr = np.array(lats), np.array(lons)
            results[split_name].update({
                'lat_range': (float(lat_arr.min()), float(lat_arr.max())),
                'lon_range': (float(lon_arr.min()), float(lon_arr.max())),
                'lat_std': float(lat_arr.std()),
                'lon_std': float(lon_arr.std()),
                'lat_mean': float(lat_arr.mean()),
                'lon_mean': float(lon_arr.mean()),
            })

    if len(coords_by_split) < 2:
        return results, None

    # Mean nearest-neighbor distance from test to train
    if 'test' in coords_by_split and 'train' in coords_by_split:
        test_coords = np.column_stack([coords_by_split['test']['lat'],
                                        coords_by_split['test']['lon']])
        train_coords = np.column_stack([coords_by_split['train']['lat'],
                                         coords_by_split['train']['lon']])
        dists = cdist(test_coords, train_coords, metric='euclidean')
        min_dists = dists.min(axis=1)
        results['test_to_train_nn'] = {
            'mean': float(min_dists.mean()),
            'median': float(np.median(min_dists)),
            'max': float(min_dists.max()),
            'std': float(min_dists.std()),
        }

    # KS tests on lat/lon distributions between splits
    split_names = list(coords_by_split.keys())
    results['spatial_ks'] = []
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            for coord_name in ['lat', 'lon']:
                a = coords_by_split[s1][coord_name]
                b = coords_by_split[s2][coord_name]
                ks_stat, p_val = stats.ks_2samp(a, b)
                results['spatial_ks'].append({
                    'pair': f'{s1} vs {s2}',
                    'coord': coord_name,
                    'ks': float(ks_stat),
                    'p': float(p_val),
                    'flag': '⚠️' if p_val < 0.05 else '✓',
                })

    return results, coords_by_split


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_label_distributions(city_label_data, city, resolution, output_dir):
    """Plot histograms of label statistics per split."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{city.title()} ({resolution}) — Label Statistics by Split',
                 fontsize=14, fontweight='bold')

    metrics = ['shadow_coverage', 'num_objects', 'mean_object_area']
    titles = ['Shadow Coverage (%)', 'Number of Shadow Objects', 'Mean Object Area (px)']
    colors = {'train': '#2196F3', 'val': '#4CAF50', 'test': '#FF5722'}

    for ax, metric, title in zip(axes, metrics, titles):
        for split_name in ['train', 'val', 'test']:
            if split_name not in city_label_data:
                continue
            data = city_label_data[split_name][metric]
            if metric == 'shadow_coverage':
                data = data * 100  # Convert to percentage
            if len(data) > 0:
                ax.hist(data, bins=25, alpha=0.5, label=split_name,
                       color=colors.get(split_name, 'gray'), density=True)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, f'label_stats_{city}_{resolution}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_spatial_coverage(coords_by_split, city, resolution, output_dir):
    """Plot lat/lon of train/val/test on a scatter plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = {'train': '#2196F3', 'val': '#4CAF50', 'test': '#FF5722'}
    markers = {'train': 'o', 'val': 's', 'test': '^'}

    for split_name in ['train', 'val', 'test']:
        if split_name not in coords_by_split:
            continue
        lat = coords_by_split[split_name]['lat']
        lon = coords_by_split[split_name]['lon']
        ax.scatter(lon, lat, c=colors.get(split_name, 'gray'),
                  marker=markers.get(split_name, 'o'),
                  label=f'{split_name} (n={len(lat)})',
                  alpha=0.6, s=20, edgecolors='white', linewidth=0.3)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{city.title()} ({resolution}) — Spatial Coverage by Split')
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    path = os.path.join(output_dir, f'spatial_{city}_{resolution}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_domain_classifier_summary(dc_results, output_dir):
    """Bar chart of domain classifier AUC per city × resolution × split pair."""
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = []
    aucs = []
    errs = []
    colors_list = []
    
    color_map = {'chicago': '#2196F3', 'miami': '#4CAF50', 'phoenix': '#FF5722'}

    for r in dc_results:
        label = f"{r['city'].title()}\n{r['resolution']}\n{r['split_pair']}"
        labels.append(label)
        aucs.append(r['auc_mean'])
        errs.append(r['auc_std'])
        colors_list.append(color_map.get(r['city'], 'gray'))

    x = np.arange(len(labels))
    bars = ax.bar(x, aucs, yerr=errs, color=colors_list, alpha=0.8,
                  edgecolor='white', linewidth=0.5, capsize=3)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Chance (0.5)')
    ax.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Notable (0.7)')
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Strong shift (0.8)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Domain Classifier: Covariate Shift Between Splits')
    ax.set_ylim(0.35, 1.0)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'domain_classifier_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mmd_summary(mmd_results, output_dir):
    """Bar chart of MMD values per city × resolution × split pair."""
    fig, ax = plt.subplots(figsize=(12, 5))

    labels = []
    mmds = []
    pvals = []
    colors_list = []
    color_map = {'chicago': '#2196F3', 'miami': '#4CAF50', 'phoenix': '#FF5722'}

    for r in mmd_results:
        label = f"{r['city'].title()}\n{r['resolution']}\n{r['split_pair']}"
        labels.append(label)
        mmds.append(r['mmd2'])
        pvals.append(r['p_value'])
        colors_list.append(color_map.get(r['city'], 'gray'))

    x = np.arange(len(labels))
    bars = ax.bar(x, mmds, color=colors_list, alpha=0.8,
                  edgecolor='white', linewidth=0.5)

    # Mark significant ones
    for i, (bar, pv) in enumerate(zip(bars, pvals)):
        if pv < 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'p={pv:.3f}*', ha='center', va='bottom', fontsize=7, color='red')
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f'p={pv:.3f}', ha='center', va='bottom', fontsize=7, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('MMD² (RBF kernel)')
    ax.set_title('Maximum Mean Discrepancy Between Splits')

    plt.tight_layout()
    path = os.path.join(output_dir, 'mmd_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────

def generate_report(ks_results, dc_results, mmd_results, spatial_results, output_dir):
    """Generate a text summary report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SPLIT DISTRIBUTION DIAGNOSTICS — SHADOWTRANSFER BENCHMARK")
    report_lines.append("=" * 80)

    # Test 1: KS
    report_lines.append("\n" + "─" * 80)
    report_lines.append("TEST 1: Label-Statistics KS Tests")
    report_lines.append("  H0: Train and test label distributions are drawn from the same population.")
    report_lines.append("  Interpretation: KS > 0.15 with p < 0.05 is a flag.")
    report_lines.append("─" * 80)
    report_lines.append(f"{'City':<10} {'Res':<8} {'Metric':<22} {'Pair':<16} {'N1':>4} {'N2':>4} {'KS':>7} {'p':>9} {'Flag':>4}")
    report_lines.append("-" * 90)
    for r in ks_results:
        report_lines.append(
            f"{r['city']:<10} {r['resolution']:<8} {r['metric']:<22} "
            f"{r['split_pair']:<16} {r['n1']:>4} {r['n2']:>4} "
            f"{r['ks_statistic']:>7.4f} {r['p_value']:>9.4f} {r['flag']:>4}"
        )

    # Test 2: Domain classifier
    report_lines.append("\n" + "─" * 80)
    report_lines.append("TEST 2: Domain-Classifier Covariate-Shift Test")
    report_lines.append("  Logistic regression on ResNet-50 features to distinguish splits.")
    report_lines.append("  AUC ≈ 0.5 = no shift, > 0.7 = notable, > 0.8 = strong shift.")
    report_lines.append("─" * 80)
    report_lines.append(f"{'City':<10} {'Res':<8} {'Pair':<16} {'AUC mean':>10} {'AUC std':>10} {'Flag':>6}")
    report_lines.append("-" * 65)
    for r in dc_results:
        flag = '⚠️' if r['auc_mean'] > 0.7 else '✓'
        report_lines.append(
            f"{r['city']:<10} {r['resolution']:<8} {r['split_pair']:<16} "
            f"{r['auc_mean']:>10.4f} {r['auc_std']:>10.4f} {flag:>6}"
        )

    # Test 3: MMD
    report_lines.append("\n" + "─" * 80)
    report_lines.append("TEST 3: Maximum Mean Discrepancy (RBF kernel)")
    report_lines.append("  Permutation test (500 iterations). p < 0.05 = significant shift.")
    report_lines.append("─" * 80)
    report_lines.append(f"{'City':<10} {'Res':<8} {'Pair':<16} {'MMD²':>12} {'p-value':>10} {'Flag':>6}")
    report_lines.append("-" * 65)
    for r in mmd_results:
        flag = '⚠️' if r['p_value'] < 0.05 else '✓'
        report_lines.append(
            f"{r['city']:<10} {r['resolution']:<8} {r['split_pair']:<16} "
            f"{r['mmd2']:>12.6f} {r['p_value']:>10.4f} {flag:>6}"
        )

    # Test 4: Spatial
    if spatial_results:
        report_lines.append("\n" + "─" * 80)
        report_lines.append("TEST 4: Spatial Coverage Analysis")
        report_lines.append("  Checks geographic spread and image-type composition per split.")
        report_lines.append("─" * 80)
        for key, result in spatial_results.items():
            city, res = key
            report_lines.append(f"\n  {city.title()} ({res}):")
            if result is None:
                report_lines.append("    No geocoordinate data available.")
                continue
            for split_name in ['train', 'val', 'test']:
                if split_name not in result:
                    continue
                s = result[split_name]
                line = f"    {split_name}: n={s.get('n_total', '?')}"
                if s.get('n_with_coords', 0) > 0:
                    line += (f", coords={s['n_with_coords']}, "
                             f"lat=[{s['lat_range'][0]:.4f}, {s['lat_range'][1]:.4f}] "
                             f"(std={s['lat_std']:.5f}), "
                             f"lon=[{s['lon_range'][0]:.4f}, {s['lon_range'][1]:.4f}] "
                             f"(std={s['lon_std']:.5f})")
                if s.get('image_type_counts'):
                    types_str = ', '.join(f"{k}={v}" for k, v in sorted(s['image_type_counts'].items()))
                    line += f"\n           image_types: {types_str}"
                report_lines.append(line)
            
            if 'test_to_train_nn' in result:
                nn = result['test_to_train_nn']
                report_lines.append(f"    Test→Train nearest-neighbor distance (deg):")
                report_lines.append(f"      Mean: {nn['mean']:.5f}  Median: {nn['median']:.5f}  "
                                   f"Max: {nn['max']:.5f}  Std: {nn['std']:.5f}")
            
            if 'spatial_ks' in result:
                report_lines.append(f"    Spatial KS tests (lat/lon distributions):")
                for sk in result['spatial_ks']:
                    report_lines.append(f"      {sk['pair']} ({sk['coord']}): "
                                       f"KS={sk['ks']:.4f}, p={sk['p']:.4f} {sk['flag']}")

    # Summary
    report_lines.append("\n" + "=" * 80)
    report_lines.append("SUMMARY & INTERPRETATION")
    report_lines.append("=" * 80)

    # Count flags per city
    city_flags = defaultdict(int)
    city_total = defaultdict(int)
    for r in ks_results:
        city_total[r['city']] += 1
        if r['flag'] == '⚠️':
            city_flags[r['city']] += 1
    for r in dc_results:
        city_total[r['city']] += 1
        if r['auc_mean'] > 0.7:
            city_flags[r['city']] += 1
    for r in mmd_results:
        city_total[r['city']] += 1
        if r['p_value'] < 0.05:
            city_flags[r['city']] += 1

    for city in sorted(city_total.keys()):
        n_flags = city_flags[city]
        n_total = city_total[city]
        status = "CLEAN" if n_flags == 0 else f"{n_flags}/{n_total} FLAGGED"
        report_lines.append(f"  {city.title()}: {status}")

    report_lines.append("\n  If Phoenix shows materially more flags than Chicago/Miami,")
    report_lines.append("  the LOCO results for Phoenix conflate cross-city shift with")
    report_lines.append("  within-city train/test heterogeneity. See §3 and §6 of the paper plan.")
    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, 'diagnostics_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n{'=' * 60}")
    print(report_text)
    print(f"\nFull report saved to: {report_path}")
    return report_text


# ─────────────────────────────────────────────────────────────
# Cross-city comparison: train vs. train
# ─────────────────────────────────────────────────────────────

def cross_city_comparison(features_cache, output_dir):
    """
    Bonus: compare train distributions ACROSS cities.
    This measures how different the cities actually are from each other,
    providing context for interpreting within-city split differences.
    """
    results = []
    cities_resolutions = list(features_cache.keys())

    for i in range(len(cities_resolutions)):
        for j in range(i + 1, len(cities_resolutions)):
            key_i = cities_resolutions[i]
            key_j = cities_resolutions[j]
            city_i, res_i = key_i
            city_j, res_j = key_j

            # Only compare same resolution
            if res_i != res_j:
                continue

            feats_i = features_cache[key_i].get('train')
            feats_j = features_cache[key_j].get('train')

            if feats_i is None or feats_j is None:
                continue

            auc_mean, auc_std = domain_classifier_test(feats_i, feats_j)
            mmd2, p_val = mmd_test_subsample(feats_i, feats_j)

            results.append({
                'pair': f'{city_i} vs {city_j}',
                'resolution': res_i,
                'dc_auc': auc_mean,
                'mmd2': mmd2,
                'mmd_p': p_val,
            })

    if results:
        print("\n" + "─" * 80)
        print("CROSS-CITY COMPARISON (train vs train, same resolution)")
        print("  Context: how different ARE the cities from each other?")
        print("─" * 80)
        print(f"{'Pair':<25} {'Res':<8} {'DC-AUC':>8} {'MMD²':>12} {'MMD-p':>8}")
        print("-" * 65)
        for r in results:
            print(f"{r['pair']:<25} {r['resolution']:<8} {r['dc_auc']:>8.4f} "
                  f"{r['mmd2']:>12.6f} {r['mmd_p']:>8.4f}")

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Split distribution diagnostics for ShadowTransfer benchmark')
    parser.add_argument('--base_data_root', type=str, required=True,
                        help='Base directory: {root}/{city}/{resolution}/{split}/images/')
    parser.add_argument('--resolutions', nargs='+', default=['highres', 'midres'],
                        help='Resolutions to test (default: highres midres)')
    parser.add_argument('--cities', nargs='+', default=['chicago', 'miami', 'phoenix'],
                        help='Cities to test (default: chicago miami phoenix)')
    parser.add_argument('--output_dir', type=str, default='./split_diagnostics_output',
                        help='Output directory for results and plots')
    parser.add_argument('--geo_metadata_path', type=str, default=None,
                        help='Path to JSON with lat/lon per image filename')
    parser.add_argument('--feature_extractor', type=str, default='resnet50',
                        choices=['resnet50'],
                        help='Feature extractor for domain classifier and MMD')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect)')
    parser.add_argument('--skip_features', action='store_true',
                        help='Skip feature-based tests (2 & 3) — runs label tests only')
    parser.add_argument('--skip_cross_city', action='store_true',
                        help='Skip cross-city comparison')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {args.device}")

    # Load geo metadata if provided
    geo_metadata = None
    if args.geo_metadata_path and os.path.exists(args.geo_metadata_path):
        geo_metadata = load_geo_metadata(args.geo_metadata_path)
        print(f"Loaded geo metadata: {len(geo_metadata)} entries")
        # Print a sample to verify parsing
        sample_key = next(iter(geo_metadata))
        print(f"  Sample entry: {sample_key} -> lat={geo_metadata[sample_key]['lat']}, "
              f"lon={geo_metadata[sample_key]['lon']}")

    # Build feature extractor once
    feat_model, feat_transform = None, None
    if not args.skip_features:
        print(f"\nLoading feature extractor: {args.feature_extractor}...")
        feat_model, feat_transform = build_feature_extractor(
            args.feature_extractor, args.device)

    # Storage for all results
    all_ks_results = []
    all_dc_results = []
    all_mmd_results = []
    all_spatial_results = {}
    features_cache = {}  # (city, res) -> {split: features_array}

    for resolution in args.resolutions:
        for city in args.cities:
            print(f"\n{'=' * 60}")
            print(f"Processing: {city.title()} — {resolution}")
            print(f"{'=' * 60}")

            city_root = os.path.join(args.base_data_root, city, resolution)
            if not os.path.exists(city_root):
                print(f"  SKIP: {city_root} does not exist")
                continue

            # Discover splits
            splits_data = {}
            filenames_by_split = {}
            for split_name in ['train', 'val', 'test']:
                img_dir = os.path.join(city_root, split_name, 'images')
                mask_dir = os.path.join(city_root, split_name, 'masks')
                if not os.path.exists(img_dir):
                    print(f"  SKIP split '{split_name}': {img_dir} not found")
                    continue
                fnames = list_files(img_dir)
                if len(fnames) == 0:
                    print(f"  SKIP split '{split_name}': no images found")
                    continue
                filenames_by_split[split_name] = fnames
                print(f"  {split_name}: {len(fnames)} images")

            if len(filenames_by_split) < 2:
                print(f"  SKIP: need at least 2 splits, found {len(filenames_by_split)}")
                continue

            # ── TEST 1: Label statistics ──
            print(f"\n  [Test 1] Computing label statistics...")
            city_label_data = {}
            for split_name, fnames in filenames_by_split.items():
                mask_dir = os.path.join(city_root, split_name, 'masks')
                city_label_data[split_name] = compute_label_stats(mask_dir, fnames)
                cov = city_label_data[split_name]['shadow_coverage']
                print(f"    {split_name}: shadow_coverage mean={cov.mean():.4f}, "
                      f"std={cov.std():.4f}, median={np.median(cov):.4f}")

            ks_res = run_ks_tests(city_label_data, city, resolution)
            all_ks_results.extend(ks_res)
            plot_label_distributions(city_label_data, city, resolution, args.output_dir)

            # ── TEST 2 & 3: Feature-based tests ──
            if not args.skip_features:
                print(f"\n  [Test 2&3] Extracting features...")
                split_features = {}
                for split_name, fnames in filenames_by_split.items():
                    img_dir = os.path.join(city_root, split_name, 'images')
                    feats = extract_features(img_dir, fnames, feat_model,
                                            feat_transform, args.device,
                                            args.batch_size)
                    split_features[split_name] = feats
                    print(f"    {split_name}: {feats.shape}")

                features_cache[(city, resolution)] = split_features

                # Domain classifier: test all split pairs
                split_names = list(split_features.keys())
                for i in range(len(split_names)):
                    for j in range(i + 1, len(split_names)):
                        s1, s2 = split_names[i], split_names[j]
                        print(f"  [Test 2] Domain classifier: {s1} vs {s2}...")
                        auc_mean, auc_std = domain_classifier_test(
                            split_features[s1], split_features[s2])
                        all_dc_results.append({
                            'city': city, 'resolution': resolution,
                            'split_pair': f'{s1} vs {s2}',
                            'auc_mean': auc_mean, 'auc_std': auc_std,
                        })
                        flag = '⚠️' if auc_mean > 0.7 else '✓'
                        print(f"    AUC = {auc_mean:.4f} ± {auc_std:.4f} {flag}")

                # MMD: test all split pairs
                for i in range(len(split_names)):
                    for j in range(i + 1, len(split_names)):
                        s1, s2 = split_names[i], split_names[j]
                        print(f"  [Test 3] MMD: {s1} vs {s2}...")
                        mmd2, p_val = mmd_test_subsample(
                            split_features[s1], split_features[s2])
                        all_mmd_results.append({
                            'city': city, 'resolution': resolution,
                            'split_pair': f'{s1} vs {s2}',
                            'mmd2': mmd2, 'p_value': p_val,
                        })
                        flag = '⚠️' if p_val < 0.05 else '✓'
                        print(f"    MMD² = {mmd2:.6f}, p = {p_val:.4f} {flag}")

            # ── TEST 4: Spatial coverage ──
            if geo_metadata is not None:
                print(f"\n  [Test 4] Spatial coverage...")
                sp_result, sp_coords = spatial_coverage_analysis(
                    geo_metadata, filenames_by_split, city, resolution)
                all_spatial_results[(city, resolution)] = sp_result
                if sp_coords:
                    plot_spatial_coverage(sp_coords, city, resolution, args.output_dir)
            else:
                all_spatial_results[(city, resolution)] = None

    # Summary plots
    if all_dc_results:
        plot_domain_classifier_summary(all_dc_results, args.output_dir)
    if all_mmd_results:
        plot_mmd_summary(all_mmd_results, args.output_dir)

    # Cross-city comparison
    if not args.skip_features and not args.skip_cross_city and features_cache:
        cross_city_comparison(features_cache, args.output_dir)

    # Generate report
    generate_report(all_ks_results, all_dc_results, all_mmd_results,
                   all_spatial_results, args.output_dir)

    # Save raw results as JSON
    raw = {
        'ks_tests': all_ks_results,
        'domain_classifier': all_dc_results,
        'mmd': all_mmd_results,
    }
    json_path = os.path.join(args.output_dir, 'diagnostics_raw.json')
    with open(json_path, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to: {json_path}")


if __name__ == '__main__':
    main()