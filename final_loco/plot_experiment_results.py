"""
Generate plots for Experiment A/B/C evaluation results.

Creates:
  - Recovery ratio heatmap (models × experiments)
  - Intensity-conditioned recovery curves
  - Per-class recovery bar charts
  - FP composition shift analysis
  - Cross-experiment comparison overlays on 1b curves
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from config import (OUTPUT_BASE)
EVAL_DIR = os.path.join(OUTPUT_BASE, "experiment_evaluation")

SHADOW_TYPE_MAP = {
    1: "Building/canyon",
    2: "Under-structure",
    3: "Tree-canopy",
    4: "Topography-cast",
    5: "Vehicle-cast",
    6: "Thin-linear",
}

EXPERIMENT_LABELS = {
    'a': 'Exp A: Decoder Retrain',
    'b': 'Exp B: BN Swap',
    'c': 'Exp C: Histogram Match',
    'b_lw_early': 'B: Early BN Only',
    'b_lw_mid': 'B: Mid BN Only',
    'b_lw_late': 'B: Late BN Only',
}

MODEL_COLORS = {
    'mamnet': '#d62728',
    'oglanet': '#1f77b4',
    'dinov3': '#2ca02c',
}
CITY_MARKERS = {
    'chicago': 'o',
    'miami': 's',
    'phoenix': '^',
}
FRACTION_MAP = {
    'a_de5pct': 0.05,
    'a_de10pct': 0.10,
    'a_de15pct': 0.15,
    'a_de20pct': 0.20,
    'a': 0.25,
}


def load_results():
    """Load experiment evaluation results."""
    path = os.path.join(EVAL_DIR, 'experiment_results.json')
    if not os.path.exists(path):
        print(f"  Results not found at {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_summary():
    """Load summary table."""
    path = os.path.join(EVAL_DIR, 'summary_table.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# PLOT 1: Recovery Ratio Heatmap
# ============================================================

def plot_recovery_heatmap(results, out_dir):
    """
    Heatmap: rows = (model, city), columns = experiments.
    Cell value = R_IoU. Color: green (>0.5 = good recovery), red (<0 = harmful).
    """
    # Organize data
    rows = []
    row_labels = []
    experiments = set()

    for key, r in results.items():
        exp = r['experiment']
        model = r['model']
        city = r['holdout_city']
        experiments.add(exp)
        row_key = f"{model}_{city}"
        if row_key not in [rl for rl, _ in rows]:
            rows.append((row_key, {}))
            row_labels.append(f"{model.upper()}\n{city.capitalize()}")
        # Find row and add data
        for rl, data in rows:
            if rl == row_key:
                data[exp] = r['global']['recovery'].get('iou')
                break

    experiments = sorted(experiments)
    n_rows = len(rows)
    n_cols = len(experiments)

    if n_rows == 0 or n_cols == 0:
        return

    # Build matrix
    matrix = np.full((n_rows, n_cols), np.nan)
    for i, (_, data) in enumerate(rows):
        for j, exp in enumerate(experiments):
            val = data.get(exp)
            if val is not None:
                matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 2.5), max(4, n_rows * 0.8)))

    # Use diverging colormap centered at 0
    vmin = min(-0.2, np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else -0.2
    vmax = max(1.2, np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 1.2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    im = ax.imshow(matrix, cmap='RdYlGn', norm=norm, aspect='auto')

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([EXPERIMENT_LABELS.get(e, e) for e in experiments],
                       fontsize=9, rotation=15, ha='right')
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                        fontsize=10, fontweight='bold', color=color)
            else:
                ax.text(j, i, "N/A", ha='center', va='center',
                        fontsize=9, color='gray')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Recovery Ratio (R)')
    ax.set_title("Recovery Ratio: R = (Exp - LOCO) / (Upper - LOCO)\n"
                 "Green = gap closed, Red = made worse", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'recovery_heatmap.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved recovery_heatmap.png")


# ============================================================
# PLOT 2: Intensity-Conditioned Recovery
# ============================================================

def plot_intensity_recovery(results, out_dir):
    """
    For each (model, city): overlay upper/loco/exp IoU vs intensity.
    Shows WHERE the experiment helps — low vs high intensity.
    """
    # Group results by (model, city)
    grouped = {}
    for key, r in results.items():
        mk = f"{r['model']}_{r['holdout_city']}"
        if mk not in grouped:
            grouped[mk] = {'model': r['model'], 'city': r['holdout_city'], 'experiments': {}}
        grouped[mk]['experiments'][r['experiment']] = r

    for mk, g in grouped.items():
        model, city = g['model'], g['city']
        exps = g['experiments']

        # We need at least one experiment with intensity data
        has_data = False
        for exp_name, r in exps.items():
            if r.get('intensity_conditioned', {}).get('recovery_bins'):
                has_data = True
                break
        if not has_data:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Intensity Analysis — {model.upper()} on {city.capitalize()}", fontsize=13)

        # Left: IoU vs intensity (absolute curves)
        # Get baseline curves from first experiment result
        first_r = list(exps.values())[0]
        upper_bins = first_r.get('intensity_conditioned', {}).get('upper_1b', {}).get('bins', [])
        loco_bins = first_r.get('intensity_conditioned', {}).get('loco_1b', {}).get('bins', [])

        if upper_bins:
            centers = [b['bin_center'] for b in upper_bins if b.get('iou_mean') is not None]
            vals = [b['iou_mean'] for b in upper_bins if b.get('iou_mean') is not None]
            ax1.plot(centers, vals, 'k-o', linewidth=2, markersize=6, label='Upper', zorder=10)

        if loco_bins:
            centers = [b['bin_center'] for b in loco_bins if b.get('iou_mean') is not None]
            vals = [b['iou_mean'] for b in loco_bins if b.get('iou_mean') is not None]
            ax1.plot(centers, vals, 'k--s', linewidth=2, markersize=5, label='LOCO', alpha=0.7)

        exp_colors = {'a': '#d62728', 'b': '#1f77b4', 'c': '#ff7f0e',
                      'b_lw_early': '#9467bd', 'b_lw_mid': '#8c564b', 'b_lw_late': '#e377c2'}

        for exp_name, r in exps.items():
            exp_bins = r.get('intensity_conditioned', {}).get('experiment_1b', {}).get('bins', [])
            if not exp_bins:
                continue
            centers = [b['bin_center'] for b in exp_bins if b.get('iou_mean') is not None]
            vals = [b['iou_mean'] for b in exp_bins if b.get('iou_mean') is not None]
            color = exp_colors.get(exp_name, 'gray')
            label = EXPERIMENT_LABELS.get(exp_name, exp_name)
            ax1.plot(centers, vals, '-^', color=color, linewidth=1.5, markersize=5, label=label)

        ax1.set_xlabel("Shadow Interior Median Intensity")
        ax1.set_ylabel("Boundary IoU")
        ax1.set_title("Absolute Performance")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Right: Recovery ratio vs intensity
        for exp_name, r in exps.items():
            recovery_bins = r.get('intensity_conditioned', {}).get('recovery_bins', [])
            if not recovery_bins:
                continue
            centers = [b['bin_center'] for b in recovery_bins
                       if b.get('recovery_iou') is not None and not np.isnan(b['recovery_iou'])]
            vals = [b['recovery_iou'] for b in recovery_bins
                    if b.get('recovery_iou') is not None and not np.isnan(b['recovery_iou'])]
            if not centers:
                continue
            color = exp_colors.get(exp_name, 'gray')
            label = EXPERIMENT_LABELS.get(exp_name, exp_name)
            ax2.plot(centers, vals, '-o', color=color, linewidth=2, markersize=6, label=label)

        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel("Shadow Interior Median Intensity")
        ax2.set_ylabel("Recovery Ratio (R)")
        ax2.set_title("Recovery by Intensity\n(R=1 = gap closed, R=0 = no help)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'intensity_recovery_{model}_{city}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved intensity_recovery_*.png")


# ============================================================
# PLOT 3: Per-Class Recovery
# ============================================================

def plot_class_recovery(results, out_dir):
    """Bar chart: recovery ratio per shadow class per experiment."""
    from config import SHADOW_TYPE_MAP

    # Group by (model, city)
    grouped = {}
    for key, r in results.items():
        mk = f"{r['model']}_{r['holdout_city']}"
        if mk not in grouped:
            grouped[mk] = {}
        grouped[mk][r['experiment']] = r.get('per_class', {}).get('recovery', {})

    for mk, exp_data in grouped.items():
        model, city = mk.split('_', 1)

        # Collect all shadow types
        all_types = set()
        for exp_recovery in exp_data.values():
            all_types.update(exp_recovery.keys())
        all_types = sorted(all_types)

        if not all_types:
            continue

        experiments = sorted(exp_data.keys())
        n_types = len(all_types)
        n_exps = len(experiments)

        fig, ax = plt.subplots(figsize=(max(8, n_types * 2), 5))

        x = np.arange(n_types)
        width = 0.8 / max(n_exps, 1)
        exp_colors = {'a': '#d62728', 'b': '#1f77b4', 'c': '#ff7f0e'}

        for i, exp_name in enumerate(experiments):
            vals = []
            for st in all_types:
                cr = exp_data[exp_name].get(str(st), exp_data[exp_name].get(st, {}))
                r_val = cr.get('recovery_recall')
                vals.append(r_val if r_val is not None else 0)

            color = exp_colors.get(exp_name, 'gray')
            label = EXPERIMENT_LABELS.get(exp_name, exp_name)
            ax.bar(x + i * width - (n_exps - 1) * width / 2, vals, width,
                   color=color, alpha=0.8, label=label, edgecolor='black', linewidth=0.5)

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

        type_labels = [SHADOW_TYPE_MAP.get(int(st), f"Type {st}") for st in all_types]
        ax.set_xticks(x)
        ax.set_xticklabels(type_labels, fontsize=9, rotation=15)
        ax.set_ylabel("Recovery Ratio (Recall)")
        ax.set_title(f"Per-Class Recovery — {model.upper()} on {city.capitalize()}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'class_recovery_{model}_{city}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved class_recovery_*.png")


# ============================================================
# PLOT 4: FP Composition Shift
# ============================================================

def plot_fp_shift(results, out_dir):
    """
    For each (model, city): show FP cluster proportions for
    upper / loco / each experiment.
    """
    grouped = {}
    for key, r in results.items():
        mk = f"{r['model']}_{r['holdout_city']}"
        if mk not in grouped:
            grouped[mk] = {'model': r['model'], 'city': r['holdout_city']}
        grouped[mk][r['experiment']] = r.get('fp_composition', {})

    for mk, g in grouped.items():
        model, city = g['model'], g['city']

        # Get cluster info from any experiment
        variants = {}
        for exp_name, fp_data in g.items():
            if exp_name in ('model', 'city'):
                continue

            for source, label in [('upper_1a', 'Upper'), ('loco_1a', 'LOCO'),
                                  ('experiment_1a', f'Exp {exp_name.upper()}')]:
                if source not in fp_data:
                    continue
                d = fp_data[source]
                if not d.get('cluster_proportions'):
                    continue
                vkey = label if source != 'experiment_1a' else label
                if vkey not in variants or source == 'experiment_1a':
                    variants[vkey] = {
                        'proportions': d['cluster_proportions'],
                        'intensities': d.get('cluster_mean_intensity', []),
                        'total_fp': d.get('total_fp_pixels', 0),
                    }

        if len(variants) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        n_clusters = max(len(v['proportions']) for v in variants.values())
        x = np.arange(n_clusters)
        width = 0.8 / len(variants)

        colors_map = {'Upper': 'black', 'LOCO': 'gray'}
        exp_c = {'Exp A': '#d62728', 'Exp B': '#1f77b4', 'Exp C': '#ff7f0e'}

        for i, (vname, vdata) in enumerate(variants.items()):
            props = vdata['proportions']
            if len(props) < n_clusters:
                props = props + [0] * (n_clusters - len(props))

            color = colors_map.get(vname, exp_c.get(vname, 'gray'))
            alpha = 0.5 if vname in ('Upper', 'LOCO') else 0.85
            ax.bar(x + i * width - (len(variants) - 1) * width / 2,
                   props[:n_clusters], width, color=color, alpha=alpha,
                   label=f"{vname} ({vdata['total_fp']:,} FP)", edgecolor='black', linewidth=0.3)

        # X-axis labels
        ref_intensities = list(variants.values())[0].get('intensities', [])
        if ref_intensities:
            ax.set_xticks(x)
            ax.set_xticklabels([f"~{int(i)}" for i in ref_intensities[:n_clusters]], fontsize=9)
            ax.set_xlabel("Cluster Mean Intensity")
        else:
            ax.set_xlabel("Cluster Index")

        ax.set_ylabel("Proportion of FP Pixels")
        ax.set_title(f"FP Composition — {model.upper()} on {city.capitalize()}\n"
                     "(Does the experiment diversify FPs away from 'dark=shadow'?)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'fp_shift_{model}_{city}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved fp_shift_*.png")


# ============================================================
# PLOT 5: Cross-Experiment Model Comparison
# ============================================================

def plot_model_comparison(results, out_dir):
    """
    For each city: grouped bar chart comparing R_IoU across models and experiments.
    """
    city_data = {}
    for key, r in results.items():
        city = r['holdout_city']
        if city not in city_data:
            city_data[city] = {}
        mk = f"{r['model']}_{r['experiment']}"
        city_data[city][mk] = r['global']['recovery'].get('iou', float('nan'))

    for city, data in city_data.items():
        models = sorted(set(k.split('_')[0] for k in data.keys()))
        experiments = sorted(set(k.split('_', 1)[1] for k in data.keys()))

        if not models or not experiments:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(models) * len(experiments)), 5))

        x = np.arange(len(models))
        width = 0.8 / max(len(experiments), 1)
        exp_colors = {'a': '#d62728', 'b': '#1f77b4', 'c': '#ff7f0e'}

        for i, exp in enumerate(experiments):
            vals = []
            for model in models:
                mk = f"{model}_{exp}"
                val = data.get(mk, float('nan'))
                vals.append(val if val is not None else 0)

            color = exp_colors.get(exp, 'gray')
            label = EXPERIMENT_LABELS.get(exp, exp)
            ax.bar(x + i * width - (len(experiments) - 1) * width / 2,
                   vals, width, color=color, alpha=0.85, label=label,
                   edgecolor='black', linewidth=0.5)

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(y=1, color='green', linestyle=':', alpha=0.3, label='Full recovery')

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models], fontsize=11)
        ax.set_ylabel("Recovery Ratio (IoU)")
        ax.set_title(f"Model × Experiment Comparison — {city.capitalize()}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'model_comparison_{city}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved model_comparison_*.png")



"""
Data Efficiency Curve Plot — Add to plot_experiment_results.py

Plots R_IoU and per-class R_recall as a function of training data fraction
(5%, 10%, 15%, 20%, 25%) for decoder retraining experiments.
"""


def extract_de_data(results):
    """
    Extract data efficiency results organized by (model, city).

    Returns:
        dict: {(model, city): {fraction: {metrics}}}
    """
    de_data = {}

    for key, r in results.items():
        exp = r.get('experiment', '')
        if exp not in FRACTION_MAP:
            continue

        model = r['model']
        city = r['holdout_city']
        frac = FRACTION_MAP[exp]

        mk = (model, city)
        if mk not in de_data:
            de_data[mk] = {}

        de_data[mk][frac] = {
            'global': r.get('global', {}),
            'per_class': r.get('per_class', {}),
            'intensity_conditioned': r.get('intensity_conditioned', {}),
        }

    return de_data


def plot_data_efficiency_global(de_data, out_dir):
    """
    Plot global R_IoU vs data fraction for each (model, city).

    One figure per city with all models overlaid.
    """
    # Group by city
    cities = sorted(set(city for _, city in de_data.keys()))

    for city in cities:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Data Efficiency — Decoder Retraining on {city.capitalize()}", fontsize=13)

        has_data = False

        for model in ['mamnet', 'oglanet', 'dinov3']:
            mk = (model, city)
            if mk not in de_data or len(de_data[mk]) < 2:
                continue

            has_data = True
            fracs = sorted(de_data[mk].keys())
            color = MODEL_COLORS.get(model, 'gray')

            # Global R_IoU
            r_iou = []
            abs_iou = []
            loco_iou = None
            upper_iou = None

            for frac in fracs:
                gd = de_data[mk][frac].get('global', {})
                recovery = gd.get('recovery', {})
                r_val = recovery.get('iou')
                r_iou.append(r_val if r_val is not None else float('nan'))

                exp_g = gd.get('experiment', {})
                abs_iou.append(exp_g.get('iou', float('nan')))

                if loco_iou is None:
                    loco_g = gd.get('loco', {})
                    loco_iou = loco_g.get('iou')
                    upper_g = gd.get('upper', {})
                    upper_iou = upper_g.get('iou')

            # Left: Absolute IoU
            axes[0].plot([f * 100 for f in fracs], abs_iou, '-o', color=color,
                        linewidth=2, markersize=7, label=model.upper())

            # Right: Recovery ratio
            axes[1].plot([f * 100 for f in fracs], r_iou, '-o', color=color,
                        linewidth=2, markersize=7, label=model.upper())

            # Add baselines (from last available fraction)
            if loco_iou is not None:
                axes[0].axhline(y=loco_iou, color=color, linestyle=':', alpha=0.4)
            if upper_iou is not None:
                axes[0].axhline(y=upper_iou, color=color, linestyle='--', alpha=0.4)

        if not has_data:
            plt.close()
            continue

        # Add general baselines
        axes[0].set_xlabel("Training Data Fraction (%)")
        axes[0].set_ylabel("IoU")
        axes[0].set_title("Absolute IoU\n(dashed = upper, dotted = LOCO)")
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks([5, 10, 15, 20, 25])

        axes[1].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        axes[1].axhline(y=1, color='green', linestyle=':', alpha=0.3, label='Full recovery')
        axes[1].set_xlabel("Training Data Fraction (%)")
        axes[1].set_ylabel("Recovery Ratio (R)")
        axes[1].set_title("Recovery Ratio\n(R=1 = gap closed)")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks([5, 10, 15, 20, 25])

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'data_efficiency_global_{city}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved data_efficiency_global_*.png")


def plot_data_efficiency_per_class(de_data, out_dir):
    """
    Plot per-class R_recall vs data fraction.

    One figure per (model, city) with all shadow classes.
    """
    class_colors = {
        '1': '#d62728',  # Building
        '2': '#ff7f0e',  # Under-structure
        '3': '#2ca02c',  # Tree
        '4': '#9467bd',  # Topography
        '5': '#8c564b',  # Vehicle
        '6': '#e377c2',  # Thin-linear
    }

    for (model, city), frac_data in de_data.items():
        if len(frac_data) < 2:
            continue

        fracs = sorted(frac_data.keys())

        # Collect all shadow types across fractions
        all_types = set()
        for frac in fracs:
            pc = frac_data[frac].get('per_class', {}).get('recovery', {})
            all_types.update(pc.keys())

        if not all_types:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Per-Class Data Efficiency — {model.upper()} on {city.capitalize()}", fontsize=13)

        for stype in sorted(all_types):
            r_recall_vals = []
            abs_recall_vals = []
            type_name = SHADOW_TYPE_MAP.get(int(stype), f"Type {stype}")
            color = class_colors.get(str(stype), 'gray')

            for frac in fracs:
                pc = frac_data[frac].get('per_class', {}).get('recovery', {})
                cls_data = pc.get(stype, pc.get(str(stype), {}))

                r_recall = cls_data.get('recovery_recall')
                r_recall_vals.append(r_recall if r_recall is not None else float('nan'))

                abs_recall = cls_data.get('exp_recall')
                abs_recall_vals.append(abs_recall if abs_recall is not None else float('nan'))

            # Filter out all-NaN series
            if all(np.isnan(v) for v in r_recall_vals):
                continue

            instance_count = 0
            for frac in fracs:
                pc = frac_data[frac].get('per_class', {}).get('recovery', {})
                cls_data = pc.get(stype, pc.get(str(stype), {}))
                ic = cls_data.get('instance_count', 0)
                if ic > instance_count:
                    instance_count = ic

            label = f"{type_name} (n={instance_count})"

            # Left: Absolute recall
            ax1.plot([f * 100 for f in fracs], abs_recall_vals, '-o', color=color,
                    linewidth=2, markersize=6, label=label)

            # Right: Recovery ratio
            # Clip extreme values for readability
            r_clipped = [max(-5, min(5, v)) if not np.isnan(v) else float('nan')
                        for v in r_recall_vals]
            ax2.plot([f * 100 for f in fracs], r_clipped, '-o', color=color,
                    linewidth=2, markersize=6, label=label)

        ax1.set_xlabel("Training Data Fraction (%)")
        ax1.set_ylabel("Recall")
        ax1.set_title("Absolute Recall by Class")
        ax1.legend(fontsize=8, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([5, 10, 15, 20, 25])
        ax1.set_ylim(0, 1)

        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=1, color='green', linestyle=':', alpha=0.3)
        ax2.set_xlabel("Training Data Fraction (%)")
        ax2.set_ylabel("Recovery Ratio (Recall)")
        ax2.set_title("Recovery Ratio by Class\n(clipped to [-5, 5])")
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([5, 10, 15, 20, 25])
        ax2.set_ylim(-5.5, 5.5)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'data_efficiency_class_{model}_{city}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved data_efficiency_class_*.png")


def plot_data_efficiency_intensity(de_data, out_dir):
    """
    Plot intensity-conditioned recovery at different data fractions.

    Shows whether high-intensity gap closes with more target data.
    """
    for (model, city), frac_data in de_data.items():
        if len(frac_data) < 2:
            continue

        fracs = sorted(frac_data.keys())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Intensity × Data Efficiency — {model.upper()} on {city.capitalize()}", fontsize=13)

        cmap = plt.cm.viridis
        frac_colors = {f: cmap(i / max(len(fracs) - 1, 1)) for i, f in enumerate(fracs)}

        # Plot upper/LOCO baselines from first available fraction
        first_frac = fracs[0]
        ic_data = frac_data[first_frac].get('intensity_conditioned', {})

        upper_1b = ic_data.get('upper_1b', {})
        loco_1b = ic_data.get('loco_1b', {})

        if upper_1b and upper_1b.get('bins'):
            centers = [b['bin_center'] for b in upper_1b['bins']
                      if b.get('iou_mean') is not None]
            vals = [b['iou_mean'] for b in upper_1b['bins']
                   if b.get('iou_mean') is not None]
            if centers:
                ax1.plot(centers, vals, 'k-', linewidth=2.5, label='Upper', zorder=10)

        if loco_1b and loco_1b.get('bins'):
            centers = [b['bin_center'] for b in loco_1b['bins']
                      if b.get('iou_mean') is not None]
            vals = [b['iou_mean'] for b in loco_1b['bins']
                   if b.get('iou_mean') is not None]
            if centers:
                ax1.plot(centers, vals, 'k--', linewidth=2, label='LOCO', alpha=0.7)

        # Plot each fraction's curves
        for frac in fracs:
            ic = frac_data[frac].get('intensity_conditioned', {})
            color = frac_colors[frac]
            label = f"{int(frac*100)}%"

            # Absolute IoU curve
            exp_1b = ic.get('experiment_1b', {})
            if exp_1b and exp_1b.get('bins'):
                centers = [b['bin_center'] for b in exp_1b['bins']
                          if b.get('iou_mean') is not None]
                vals = [b['iou_mean'] for b in exp_1b['bins']
                       if b.get('iou_mean') is not None]
                if centers:
                    ax1.plot(centers, vals, '-^', color=color, linewidth=1.5,
                            markersize=5, label=label)

            # Recovery by bin
            recovery_bins = ic.get('recovery_bins', [])
            if recovery_bins:
                centers = [b['bin_center'] for b in recovery_bins
                          if b.get('recovery_iou') is not None
                          and not np.isnan(b['recovery_iou'])]
                vals = [b['recovery_iou'] for b in recovery_bins
                       if b.get('recovery_iou') is not None
                       and not np.isnan(b['recovery_iou'])]
                # Clip for readability
                vals_clipped = [max(-3, min(3, v)) for v in vals]
                if centers:
                    ax2.plot(centers, vals_clipped, '-o', color=color,
                            linewidth=1.5, markersize=5, label=label)

        ax1.set_xlabel("Shadow Interior Median Intensity")
        ax1.set_ylabel("Boundary IoU")
        ax1.set_title("Absolute Performance")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=1, color='green', linestyle=':', alpha=0.3)
        ax2.set_xlabel("Shadow Interior Median Intensity")
        ax2.set_ylabel("Recovery Ratio (clipped [-3, 3])")
        ax2.set_title("Recovery by Intensity at Each Fraction")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'data_efficiency_intensity_{model}_{city}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()

    print("  Saved data_efficiency_intensity_*.png")

# ============================================================
# MAIN
# ============================================================

def main():
    results = load_results()
    if results is None:
        print("No results to plot. Run evaluate_experiments.py first.")
        return

    out_dir = os.path.join(EVAL_DIR, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13,
        'axes.labelsize': 12, 'legend.fontsize': 9, 'figure.dpi': 150,
    })

    print("Generating experiment evaluation plots...")
    plot_recovery_heatmap(results, out_dir)
    plot_intensity_recovery(results, out_dir)
    plot_class_recovery(results, out_dir)
    plot_fp_shift(results, out_dir)
    plot_model_comparison(results, out_dir)

    de_data = extract_de_data(results)
    if not de_data:
        print("No data efficiency results found in experiment_results.json.")
        print("Make sure you ran Experiment A with --data_efficiency flag")
        print("and then ran evaluate_experiments.py.")
        return

    print(f"Found data efficiency results for: "
          f"{[(m, c) for m, c in de_data.keys()]}")
    for mk, fracs in de_data.items():
        print(f"  {mk}: fractions = {sorted(fracs.keys())}")

    print("\nGenerating data efficiency plots...")
    plot_data_efficiency_global(de_data, out_dir)
    plot_data_efficiency_per_class(de_data, out_dir)
    plot_data_efficiency_intensity(de_data, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == '__main__':
    main()