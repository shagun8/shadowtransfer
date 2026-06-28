"""
Diagnostic 1d: Linear Probe for City-Identity in Encoder Features
(v2 — GroupKFold fix)

Changes from v1:
  - load_features() now loads image_indices from features.npz and returns them
  - run_probe() uses GroupKFold (image as group) instead of StratifiedKFold
    → eliminates same-image pixel leakage across CV folds
  - Permutation control in compute_paper_statistics.py must also be updated
    (see that file's _run_permutation function)

Requires: pre-extracted features from extract_features.py
  (features.npz must contain 'image_indices' key — already present in current extraction code)

Usage:
    python thread1_1d.py
"""

import os
import json
import glob
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

# --- NCSA Delta --- (paths resolve from the PROJECT_ROOT env var)
FEATURE_BASE = os.path.join(os.environ["PROJECT_ROOT"], "data", "extracted_features")
OUTPUT_BASE  = os.path.join(os.environ["PROJECT_ROOT"], "data", "loco_diagnostic_results")

CITIES = ["chicago", "miami", "phoenix"]
CITY_COLORS = {"chicago": "#1f77b4", "miami": "#ff7f0e", "phoenix": "#2ca02c"}
MODELS = ["mamnet", "oglanet", "dinov3"]
N_INTENSITY_BINS = 5
MAX_SAMPLES_PER_CITY = 30000


# ============================================================
# FEATURE LOADING
# ============================================================

def discover_probes(feature_base=FEATURE_BASE):
    """
    Scan the feature directory and find all (model_type, checkpoint_id) combos
    that have features extracted for at least 2 cities.
    """
    probes = []
    if not os.path.isdir(feature_base):
        print(f"  Feature base not found: {feature_base}")
        return probes

    for model_type in os.listdir(feature_base):
        mdir = os.path.join(feature_base, model_type)
        if not os.path.isdir(mdir):
            continue

        for ckpt_id in os.listdir(mdir):
            cdir = os.path.join(mdir, ckpt_id)
            if not os.path.isdir(cdir):
                continue

            city_paths = {}
            for entry in os.listdir(cdir):
                feat_file = os.path.join(cdir, entry, "features.npz")
                if os.path.exists(feat_file):
                    city_name = entry.split("_")[0]
                    if city_name in CITIES:
                        city_paths[city_name] = feat_file

            if len(city_paths) >= 2:
                probes.append({
                    'model_type': model_type,
                    'checkpoint_id': ckpt_id,
                    'cities': city_paths,
                })

    return probes


def load_features(city_paths, max_per_city=MAX_SAMPLES_PER_CITY):
    """
    Load and concatenate features from multiple cities.

    Returns:
        X: (N, D) float32 feature matrix
        y_city: (N,) int city label (0, 1, 2 for chicago, miami, phoenix)
        intensities: (N,) float intensity per cell
        gt_labels: (N,) int8 ground-truth shadow label
        pred_labels: (N,) int8 model prediction label
        image_ids: (N,) int32 globally-unique image identifier per pixel
                   (offset across cities so IDs are unique globally)
    """
    city_to_idx = {c: i for i, c in enumerate(CITIES)}

    all_X, all_y, all_int, all_gt, all_pred, all_img_ids = [], [], [], [], [], []
    global_image_offset = 0

    for city, path in city_paths.items():
        data = np.load(path)
        feat = data['features'].astype(np.float32)
        inten = data['intensities'].astype(np.float32)
        gt = data['gt_labels']
        pred = data['pred_labels']

        # Load image_indices (per-pixel image ID within this city)
        if 'image_indices' in data:
            img_ids = data['image_indices'].astype(np.int32)
        else:
            # Fallback: if features.npz was extracted with older code
            # that didn't save image_indices, assign all pixels image 0
            # (degrades to StratifiedKFold behavior — print warning)
            print(f"    WARNING: {path} missing 'image_indices'. "
                  f"GroupKFold will not properly separate images for {city}.")
            img_ids = np.zeros(feat.shape[0], dtype=np.int32)

        n = feat.shape[0]
        if n > max_per_city:
            # Deterministic thinning: evenly spaced indices
            idx = np.linspace(0, n - 1, max_per_city, dtype=int)
            feat    = feat[idx]
            inten   = inten[idx]
            gt      = gt[idx]
            pred    = pred[idx]
            img_ids = img_ids[idx]
            n = max_per_city

        # Offset image_ids to be globally unique across cities
        img_ids = img_ids + global_image_offset
        global_image_offset = int(img_ids.max()) + 1

        all_X.append(feat)
        all_y.append(np.full(n, city_to_idx[city], dtype=int))
        all_int.append(inten)
        all_gt.append(gt)
        all_pred.append(pred)
        all_img_ids.append(img_ids)

        n_unique_images = len(np.unique(img_ids))
        print(f"    {city}: {n} vectors ({feat.shape[1]}D), {n_unique_images} unique images")

    return (np.concatenate(all_X), np.concatenate(all_y),
            np.concatenate(all_int), np.concatenate(all_gt),
            np.concatenate(all_pred), np.concatenate(all_img_ids))


# ============================================================
# PROBE TRAINING + EVALUATION
# ============================================================

def run_probe(X, y, n_classes, image_ids, n_folds=5):
    """
    Train logistic regression probe with GroupKFold CV (image as group).

    GroupKFold ensures all pixels from the same image stay in the same fold,
    preventing spatial-correlation leakage between train and test sets.

    Returns dict with:
        train_acc, cv_acc, cv_std, per_class_acc, confusion_matrix, n_folds,
        n_unique_images, cv_method
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Full-data train accuracy (capacity check)
    clf_full = LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=42, C=1.0,
    )
    clf_full.fit(X_scaled, y)
    train_acc = accuracy_score(y, clf_full.predict(X_scaled))

    # Check we have enough unique images for the requested n_folds
    unique_images = np.unique(image_ids)
    n_unique = len(unique_images)

    if n_unique < n_folds:
        print(f"    WARNING: only {n_unique} unique images, need {n_folds} for GroupKFold. "
              f"Reducing n_folds to {n_unique}.")
        n_folds = n_unique

    if n_unique < 2:
        print(f"    WARNING: only {n_unique} unique image(s). Cannot do CV.")
        return {
            'train_acc': float(train_acc), 'cv_acc': float('nan'),
            'cv_std': float('nan'), 'per_class_acc': {},
            'confusion_matrix': [], 'n_folds': 0,
            'n_unique_images': n_unique, 'cv_method': 'none',
        }

    # GroupKFold CV — images never split across folds
    gkf = GroupKFold(n_splits=n_folds)
    fold_accs = []
    all_preds_cv = np.full_like(y, -1)

    for train_idx, test_idx in gkf.split(X_scaled, y, groups=image_ids):
        clf = LogisticRegression(
            max_iter=1000, solver='lbfgs', random_state=42, C=1.0,
        )
        clf.fit(X_scaled[train_idx], y[train_idx])
        pred = clf.predict(X_scaled[test_idx])
        all_preds_cv[test_idx] = pred
        fold_accs.append(accuracy_score(y[test_idx], pred))

    cv_acc = float(np.mean(fold_accs))
    cv_std = float(np.std(fold_accs))

    # Per-class accuracy from CV predictions
    per_class = {}
    for c in range(n_classes):
        mask = (y == c) & (all_preds_cv >= 0)
        if mask.sum() > 0:
            per_class[CITIES[c]] = float(accuracy_score(y[mask], all_preds_cv[mask]))

    # Confusion matrix from CV
    valid_mask = all_preds_cv >= 0
    cm = confusion_matrix(y[valid_mask], all_preds_cv[valid_mask],
                          labels=list(range(n_classes)))

    return {
        'train_acc': float(train_acc),
        'cv_acc': cv_acc,
        'cv_std': cv_std,
        'per_class_acc': per_class,
        'confusion_matrix': cm.tolist(),
        'n_folds': n_folds,
        'n_unique_images': n_unique,
        'cv_method': 'GroupKFold_image',
    }


def intensity_conditioned_probe(X, y, intensities, n_classes,
                                n_bins=N_INTENSITY_BINS):
    """
    Train probe on ALL data, then evaluate accuracy per intensity bin.
    Uses quantile-based bins so each bin has roughly equal samples.

    Note: This intentionally trains on all data (no CV) because the question
    is "does the full-information probe's accuracy vary with intensity?",
    not "how well does it generalize." The CV accuracy is handled by run_probe().
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=42, C=1.0,
    )
    clf.fit(X_scaled, y)
    y_pred = clf.predict(X_scaled)

    # Quantile-based bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(intensities, quantiles))
    actual_bins = len(edges) - 1
    if actual_bins < 2:
        return {'n_bins': 0, 'bins': []}

    bin_idx = np.digitize(intensities, edges) - 1
    bin_idx = np.clip(bin_idx, 0, actual_bins - 1)

    bins = []
    for b in range(actual_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count < 20:
            bins.append({
                'bin_low': float(edges[b]), 'bin_high': float(edges[b + 1]),
                'bin_center': float((edges[b] + edges[b + 1]) / 2),
                'count': count, 'accuracy': None,
            })
        else:
            acc = float(accuracy_score(y[mask], y_pred[mask]))
            bins.append({
                'bin_low': float(edges[b]), 'bin_high': float(edges[b + 1]),
                'bin_center': float((edges[b] + edges[b + 1]) / 2),
                'count': count, 'accuracy': acc,
            })

    return {'n_bins': actual_bins, 'bin_edges': edges.tolist(), 'bins': bins}


def correct_vs_incorrect_probe(X, y, gt_labels, pred_labels, n_classes):
    """
    Split features into correctly-predicted and incorrectly-predicted cells.
    Train probe on all, evaluate accuracy separately on each group.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=1000, solver='lbfgs', random_state=42, C=1.0,
    )
    clf.fit(X_scaled, y)
    y_city_pred = clf.predict(X_scaled)

    correct_mask   = gt_labels == pred_labels
    incorrect_mask = ~correct_mask

    result = {}
    for label, mask in [('correct', correct_mask), ('incorrect', incorrect_mask)]:
        n = int(mask.sum())
        if n < 20:
            result[label] = {'count': n, 'probe_accuracy': None}
        else:
            acc = float(accuracy_score(y[mask], y_city_pred[mask]))
            result[label] = {'count': n, 'probe_accuracy': acc}

    return result


# ============================================================
# MAIN DIAGNOSTIC
# ============================================================

def diagnostic_1d(feature_base=FEATURE_BASE, output_base=OUTPUT_BASE):
    """
    Run all 1d probe analyses on available extracted features.
    Saves results to output_base/thread1/1d_linear_probe/
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1d: Linear Probe for City Identity (GroupKFold)")
    print("=" * 70)

    probes = discover_probes(feature_base)
    if not probes:
        print("  No extracted features found. Run extract_features.py first.")
        return {}

    print(f"  Found {len(probes)} probe-ready (model, checkpoint) combos\n")

    results = {}

    for p in probes:
        model_type = p['model_type']
        ckpt_id    = p['checkpoint_id']
        city_paths = p['cities']
        n_cities   = len(city_paths)
        chance     = 1.0 / n_cities

        key = f"{model_type}_{ckpt_id}"
        print(f"  === {key} ({n_cities} cities) ===")

        # Load features (now includes image_ids)
        X, y_city, intensities, gt_labels, pred_labels, image_ids = \
            load_features(city_paths)
        n_unique_images = len(np.unique(image_ids))
        print(f"    Total: {X.shape[0]} vectors, {X.shape[1]}D, "
              f"{n_cities} classes, {n_unique_images} unique images")

        # 1. Overall probe (GroupKFold)
        probe_result = run_probe(X, y_city, n_cities, image_ids)
        print(f"    Probe CV accuracy ({probe_result['cv_method']}): "
              f"{probe_result['cv_acc']:.3f} ± {probe_result['cv_std']:.3f}  "
              f"(chance={chance:.3f})")

        # 2. Intensity-conditioned (trains on all data, no CV)
        intensity_result = intensity_conditioned_probe(
            X, y_city, intensities, n_cities)

        # 3. Correct vs incorrect (trains on all data, no CV)
        corr_incorr = correct_vs_incorrect_probe(
            X, y_city, gt_labels, pred_labels, n_cities)
        for label, info in corr_incorr.items():
            if info['probe_accuracy'] is not None:
                print(f"    Probe on {label} pixels: {info['probe_accuracy']:.3f}")

        results[key] = {
            'model_type':       model_type,
            'checkpoint_id':    ckpt_id,
            'n_cities':         n_cities,
            'n_samples':        int(X.shape[0]),
            'n_unique_images':  n_unique_images,
            'feature_dim':      int(X.shape[1]),
            'chance_accuracy':  float(chance),
            'cities_available': list(city_paths.keys()),
            'probe':                probe_result,
            'intensity_conditioned': intensity_result,
            'correct_vs_incorrect':  corr_incorr,
        }

    # Save results
    out_dir  = os.path.join(output_base, "thread1", "1d_linear_probe")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "probe_results.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    return results


# ============================================================
# PLOTTING (unchanged from v1)
# ============================================================

def plot_1d_probe(results_dir=OUTPUT_BASE):
    data_path = os.path.join(
        results_dir, "thread1", "1d_linear_probe", "probe_results.json")
    if not os.path.exists(data_path):
        print("  1d: no probe results found, skipping plots")
        return

    with open(data_path) as f:
        results = json.load(f)

    if not results:
        return

    out_dir = os.path.join(
        results_dir, "thread1", "1d_linear_probe", "plots")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13,
        'axes.labelsize': 12, 'legend.fontsize': 9, 'figure.dpi': 150,
    })

    _plot_overall_accuracy(results, out_dir)
    _plot_intensity_conditioned(results, out_dir)
    _plot_correct_incorrect(results, out_dir)
    _plot_confusion_matrices(results, out_dir)

    print("  1d plots saved")


def _plot_overall_accuracy(results, out_dir):
    upper_keys = [k for k in results if 'upper' in k]
    loco_keys  = [k for k in results if 'loco'  in k]

    for group_name, keys in [('upper', upper_keys), ('loco_vanilla', loco_keys)]:
        if not keys:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 2), 5))

        x      = np.arange(len(keys))
        accs   = [results[k]['probe']['cv_acc']  for k in keys]
        stds   = [results[k]['probe']['cv_std']  for k in keys]
        chances = [results[k]['chance_accuracy'] for k in keys]
        labels = [results[k]['model_type'].upper() + "\n" + results[k]['checkpoint_id']
                  for k in keys]

        model_colors = {'mamnet': '#d62728', 'oglanet': '#1f77b4', 'dinov3': '#2ca02c'}
        colors = [model_colors.get(results[k]['model_type'], 'gray') for k in keys]

        ax.bar(x, accs, yerr=stds, width=0.6, color=colors,
               edgecolor='black', linewidth=0.5, capsize=5, alpha=0.85)
        ax.axhline(y=chances[0], color='gray', linestyle='--', linewidth=1.5,
                   label=f'Chance ({chances[0]:.2f})')

        for xi, acc, std in zip(x, accs, stds):
            ax.text(xi, acc + std + 0.015, f"{acc:.3f}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, ha='center')
        ax.set_ylabel("Probe CV Accuracy (GroupKFold, 5-fold)")
        ax.set_title(
            f"City-Identity Probe — {group_name.replace('_', ' ').title()} Models\n"
            f"(Higher = more city-specific features = more entanglement)")
        ax.legend(loc='lower right')
        ax.set_ylim(0, min(1.0, max(accs) + 0.15))
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"probe_accuracy_{group_name}.png"),
                    bbox_inches='tight')
        plt.close()


def _plot_intensity_conditioned(results, out_dir):
    for key, r in results.items():
        ic   = r.get('intensity_conditioned', {})
        bins = ic.get('bins', [])
        valid_bins = [b for b in bins if b.get('accuracy') is not None]
        if len(valid_bins) < 2:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Intensity-Conditioned Probe — "
            f"{r['model_type'].upper()} / {r['checkpoint_id']}")

        centers = [b['bin_center'] for b in valid_bins]
        accs    = [b['accuracy']   for b in valid_bins]
        counts  = [b['count']      for b in valid_bins]
        chance  = r['chance_accuracy']

        ax1.plot(centers, accs, 'o-', color='#d62728', linewidth=2, markersize=7)
        ax1.axhline(y=chance, color='gray', linestyle='--', linewidth=1.5,
                    label=f'Chance ({chance:.2f})')
        ax1.axhline(y=r['probe']['cv_acc'], color='blue', linestyle=':',
                    linewidth=1.5, label=f"Overall CV ({r['probe']['cv_acc']:.3f})")
        ax1.set_xlabel("Boundary-Band Cell Intensity")
        ax1.set_ylabel("Probe Accuracy")
        ax1.set_title("Does city-specificity increase\nwith surface intensity?")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 255)

        widths = [b['bin_high'] - b['bin_low'] for b in valid_bins]
        ax2.bar(centers, counts, width=[w * 0.9 for w in widths],
                color='steelblue', alpha=0.7, edgecolor='black')
        for c, cnt in zip(centers, counts):
            ax2.text(c, cnt + max(counts) * 0.02, str(cnt),
                     ha='center', va='bottom', fontsize=8)
        ax2.set_xlabel("Boundary-Band Cell Intensity")
        ax2.set_ylabel("Number of Feature Cells")
        ax2.set_title("Sample count per bin")
        ax2.set_xlim(0, 255)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_key = key.replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"intensity_probe_{safe_key}.png"),
                    bbox_inches='tight')
        plt.close()


def _plot_correct_incorrect(results, out_dir):
    keys_with_data = [
        k for k in results
        if results[k].get('correct_vs_incorrect', {})
               .get('correct', {}).get('probe_accuracy') is not None
    ]
    if not keys_with_data:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(keys_with_data) * 2.5), 5))
    x     = np.arange(len(keys_with_data))
    width = 0.35

    correct_accs   = []
    incorrect_accs = []
    labels = []
    for k in keys_with_data:
        ci = results[k]['correct_vs_incorrect']
        correct_accs.append(ci['correct']['probe_accuracy'])
        inc_acc = ci.get('incorrect', {}).get('probe_accuracy')
        incorrect_accs.append(inc_acc if inc_acc is not None else 0)
        labels.append(
            results[k]['model_type'].upper() + "\n" + results[k]['checkpoint_id'])

    ax.bar(x - width / 2, correct_accs, width,
           label='Correctly predicted cells',
           color='#2ca02c', alpha=0.8, edgecolor='black')
    ax.bar(x + width / 2, incorrect_accs, width,
           label='Incorrectly predicted cells',
           color='#d62728', alpha=0.8, edgecolor='black')

    chance = results[keys_with_data[0]]['chance_accuracy']
    ax.axhline(y=chance, color='gray', linestyle='--',
               label=f'Chance ({chance:.2f})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Probe Accuracy")
    ax.set_title(
        "City Probe: Correct vs Incorrect Predictions\n"
        "(If correct pixels still encode city → deep entanglement)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correct_vs_incorrect.png"),
                bbox_inches='tight')
    plt.close()


def _plot_confusion_matrices(results, out_dir):
    for key, r in results.items():
        cm     = np.array(r['probe']['confusion_matrix'])
        if cm.size == 0:
            continue
        cities = r['cities_available']
        n      = len(cities)

        fig, ax = plt.subplots(figsize=(5, 4))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([c.capitalize() for c in cities])
        ax.set_yticklabels([c.capitalize() for c in cities])
        ax.set_xlabel("Predicted City")
        ax.set_ylabel("True City")
        ax.set_title(
            f"Probe Confusion — {r['model_type'].upper()}\n{r['checkpoint_id']}")

        for i in range(n):
            for j in range(n):
                ax.text(j, i,
                        f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                        ha='center', va='center', fontsize=9,
                        color='white' if cm_norm[i, j] > 0.5 else 'black')

        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        safe_key = key.replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"confusion_{safe_key}.png"),
                    bbox_inches='tight')
        plt.close()


# ============================================================
# STANDALONE ENTRY POINT
# ============================================================

if __name__ == '__main__':
    results = diagnostic_1d()
    if results:
        plot_1d_probe()
    else:
        print("No results to plot. Run extract_features.py first.")