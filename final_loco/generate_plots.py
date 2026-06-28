"""
Generate plots from diagnostic results.

Usage:
    python generate_plots.py [--results_dir /path/to/diagnostic_results]
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*")
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CITIES, RESOLUTIONS, MODELS, LOCO_VARIANTS,
    SHADOW_TYPE_MAP, SHADOW_TYPE_SHORT, OUTPUT_BASE,
)
from utils import DIRECTION_BINS

try:
    from thread1_1d import plot_1d_probe
    HAS_1D_PLOT = True
except ImportError:
    HAS_1D_PLOT = False

# Consistent styling
CITY_COLORS = {"chicago": "#1f77b4", "miami": "#ff7f0e", "phoenix": "#2ca02c"}
MODEL_MARKERS = {"mamnet": "o", "oglanet": "s", "dinov3": "D"}
VARIANT_LINESTYLES = {"vanilla": "-", "fda": "--", "segdesic": "-.", "mcl": ":"}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def load_json(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ============================================================
# THREAD 1 PLOTS
# ============================================================

def plot_1a_fp_clustering(results_dir):
    """
    Plot FP cluster proportions: grouped bar chart comparing cluster distributions
    across cities for each model.
    """
    data = load_json(os.path.join(results_dir, "thread1", "1a_fp_clustering", "fp_clustering_results.json"))
    if not data:
        return
    
    out = os.path.join(results_dir, "thread1", "1a_fp_clustering", "plots")
    os.makedirs(out, exist_ok=True)
    
    for res in RESOLUTIONS:
        for model in MODELS:
            # Compare upper bound across cities
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            fig.suptitle(f"FP Cluster Composition — {model.upper()} @ {res}\n(Upper Bound: Same-City)", fontsize=13)
            
            for ax, city in zip(axes, CITIES):
                key = f"upper_{model}_{city}_{res}"
                if key not in data or data[key].get("total_fp_pixels", 0) == 0:
                    ax.set_title(f"{city} (no data)")
                    continue
                
                r = data[key]
                proportions = r["cluster_proportions"]
                intensities = r["cluster_mean_intensity"]
                n_clusters = len(proportions)
                
                colors = [plt.cm.gray(int(v) / 255) for v in intensities]
                bars = ax.bar(range(n_clusters), proportions, color=colors, edgecolor="black", linewidth=0.5)
                
                ax.set_title(f"{city.capitalize()} ({r['total_fp_pixels']} FPs)")
                ax.set_xlabel("Cluster (sorted by intensity →)")
                ax.set_xticks(range(n_clusters))
                ax.set_xticklabels([f"{v:.0f}" for v in intensities], fontsize=8)
                
                if ax == axes[0]:
                    ax.set_ylabel("Proportion of FP pixels")
            
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"fp_clusters_upper_{model}_{res}.png"), bbox_inches="tight")
            plt.close()
            
            # LOCO comparison: for each city, compare across variants
            for city in CITIES:
                fig, axes = plt.subplots(1, len(LOCO_VARIANTS) + 1, figsize=(4 * (len(LOCO_VARIANTS) + 1), 4), sharey=True)
                fig.suptitle(f"FP Cluster Composition — {model.upper()} on {city.capitalize()} @ {res}", fontsize=13)
                
                all_keys = [f"upper_{model}_{city}_{res}"] + [f"loco_{model}_{v}_{city}_{res}" for v in LOCO_VARIANTS]
                all_labels = ["Upper\n(same-city)"] + [v.upper() for v in LOCO_VARIANTS]
                
                for ax, key, label in zip(axes, all_keys, all_labels):
                    if key not in data or data[key].get("total_fp_pixels", 0) == 0:
                        ax.set_title(f"{label}\n(no data)")
                        continue
                    
                    r = data[key]
                    proportions = r["cluster_proportions"]
                    intensities = r["cluster_mean_intensity"]
                    colors = [plt.cm.gray(int(v) / 255) for v in intensities]
                    
                    ax.bar(range(len(proportions)), proportions, color=colors, edgecolor="black", linewidth=0.5)
                    ax.set_title(f"{label}\n({r['total_fp_pixels']} FPs)")
                    ax.set_xticks(range(len(proportions)))
                    ax.set_xticklabels([f"{v:.0f}" for v in intensities], fontsize=8)
                    
                    if ax == axes[0]:
                        ax.set_ylabel("Proportion")
                
                plt.tight_layout()
                plt.savefig(os.path.join(out, f"fp_clusters_loco_{model}_{city}_{res}.png"), bbox_inches="tight")
                plt.close()
    
    print("  1a plots saved")


def plot_1b_intensity_curves(results_dir):
    """
    Plot boundary IoU/recall vs. shadow interior intensity.
    Compare same-city (upper) vs cross-city (LOCO) curves.
    Annotates sample counts per bin.
    """
    data = load_json(os.path.join(results_dir, "thread1", "1b_intensity_curves", "intensity_curve_results.json"))
    if not data:
        return
    
    out = os.path.join(results_dir, "thread1", "1b_intensity_curves", "plots")
    os.makedirs(out, exist_ok=True)
    
    for res in RESOLUTIONS:
        for model in MODELS:
            for city in CITIES:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
                fig.suptitle(f"Intensity-Conditioned Performance — {model.upper()} on {city.capitalize()} @ {res}")
                
                # Upper bound
                ukey = f"upper_{model}_{city}_{res}"
                upper_counts = []
                if ukey in data and data[ukey].get("n_instances", 0) > 0:
                    bins = data[ukey]["bins"]
                    centers = [b["bin_center"] for b in bins if b["count"] > 0]
                    ious = [b["iou_mean"] for b in bins if b["count"] > 0]
                    recalls = [b["recall_mean"] for b in bins if b["count"] > 0]
                    upper_counts = [(b["bin_center"], b["count"]) for b in bins if b["count"] > 0]
                    
                    ax1.plot(centers, ious, 'k-o', label="Upper (same-city)", linewidth=2, markersize=6)
                    ax2.plot(centers, recalls, 'k-o', label="Upper (same-city)", linewidth=2, markersize=6)
                
                # LOCO variants
                for variant in LOCO_VARIANTS:
                    lkey = f"loco_{model}_{variant}_{city}_{res}"
                    if lkey not in data or data[lkey].get("n_instances", 0) == 0:
                        continue
                    
                    bins = data[lkey]["bins"]
                    centers = [b["bin_center"] for b in bins if b["count"] > 0]
                    ious = [b["iou_mean"] for b in bins if b["count"] > 0]
                    recalls = [b["recall_mean"] for b in bins if b["count"] > 0]
                    
                    ls = VARIANT_LINESTYLES.get(variant, "-")
                    ax1.plot(centers, ious, ls, marker="^", label=variant.upper(), linewidth=1.5)
                    ax2.plot(centers, recalls, ls, marker="^", label=variant.upper(), linewidth=1.5)
                
                ax1.set_xlabel("Shadow Interior Median Intensity")
                ax1.set_ylabel("Boundary IoU")
                ax1.set_title("IoU vs. Surface Intensity")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, 255)
                
                ax2.set_xlabel("Shadow Interior Median Intensity")
                ax2.set_ylabel("Boundary Recall")
                ax2.set_title("Recall vs. Surface Intensity")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(0, 255)
                
                # ax3: sample count distribution (from upper, since GT instances are same)
                if upper_counts:
                    bin_centers_c = [c for c, _ in upper_counts]
                    bin_counts_c = [n for _, n in upper_counts]
                    # Use bin edges for proper bar widths
                    ukey_data = data.get(ukey, {})
                    bin_edges_list = ukey_data.get("bin_edges", [])
                    u_bins = ukey_data.get("bins", [])
                    
                    if u_bins:
                        for b_info in u_bins:
                            if b_info["count"] > 0:
                                bw = b_info["bin_high"] - b_info["bin_low"]
                                ax3.bar(b_info["bin_center"], b_info["count"], width=bw * 0.9,
                                        color=CITY_COLORS.get(city, 'gray'),
                                        alpha=0.7, edgecolor='black')
                                ax3.text(b_info["bin_center"], b_info["count"] + max(bin_counts_c) * 0.02,
                                        str(b_info["count"]),
                                        ha='center', va='bottom', fontsize=7)
                ax3.set_xlabel("Shadow Interior Median Intensity")
                ax3.set_ylabel("Number of Instances")
                ax3.set_title("Sample Count per Bin")
                ax3.set_xlim(0, 255)
                ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out, f"intensity_{model}_{city}_{res}.png"), bbox_inches="tight")
                plt.close()
        
        # --- Cross-model comparison for each city ---
        for city in CITIES:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Intensity-Conditioned IoU — All Models on {city.capitalize()} @ {res}\n(Solid=Upper, Dashed=LOCO Vanilla)")
            
            model_colors = {"mamnet": "#d62728", "oglanet": "#1f77b4", "dinov3": "#2ca02c"}
            
            for ax_idx, (ax, metric_key, metric_label) in enumerate(zip(
                axes, ["iou_mean", "recall_mean", "precision_mean"], ["IoU", "Recall", "Precision"]
            )):
                for model in MODELS:
                    color = model_colors.get(model, 'gray')
                    
                    # Upper
                    ukey = f"upper_{model}_{city}_{res}"
                    if ukey in data and data[ukey].get("n_instances", 0) > 0:
                        bins = data[ukey]["bins"]
                        centers = [b["bin_center"] for b in bins if b["count"] > 0 and b.get(metric_key) is not None]
                        vals = [b[metric_key] for b in bins if b["count"] > 0 and b.get(metric_key) is not None]
                        if centers:
                            ax.plot(centers, vals, '-o', color=color, label=f"{model.upper()} Upper",
                                   linewidth=2, markersize=5)
                    
                    # LOCO vanilla
                    lkey = f"loco_{model}_vanilla_{city}_{res}"
                    if lkey in data and data[lkey].get("n_instances", 0) > 0:
                        bins = data[lkey]["bins"]
                        centers = [b["bin_center"] for b in bins if b["count"] > 0 and b.get(metric_key) is not None]
                        vals = [b[metric_key] for b in bins if b["count"] > 0 and b.get(metric_key) is not None]
                        if centers:
                            ax.plot(centers, vals, '--^', color=color, label=f"{model.upper()} LOCO",
                                   linewidth=1.5, markersize=4, alpha=0.7)
                
                ax.set_xlabel("Shadow Interior Median Intensity")
                ax.set_ylabel(f"Boundary {metric_label}")
                ax.set_title(metric_label)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 255)
            
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"intensity_crossmodel_{city}_{res}.png"), bbox_inches="tight")
            plt.close()
        
        # --- GT intensity distribution comparison across cities ---
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(f"GT Shadow Instance Intensity Distribution — {res}")
        
        for city in CITIES:
            # Use upper bound data (GT instances are same regardless of model)
            ukey = f"upper_{MODELS[0]}_{city}_{res}"
            if ukey not in data or data[ukey].get("n_instances", 0) == 0:
                continue
            
            raw = data[ukey].get("raw_intensities", [])
            if not raw:
                continue
            
            total = len(raw)
            hist_bins = np.linspace(0, 255, 12)  # equal-width bins for GT distribution
            counts, edges = np.histogram(raw, bins=hist_bins)
            centers = (edges[:-1] + edges[1:]) / 2
            proportions = counts / total
            
            ax.plot(centers, proportions, '-o', color=CITY_COLORS[city],
                   label=f"{city.capitalize()} (n={total})", linewidth=2, markersize=6)
        
        ax.set_xlabel("Shadow Interior Median Intensity")
        ax.set_ylabel("Proportion of Instances")
        ax.set_title("Where do shadows fall on the intensity spectrum?")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 255)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"gt_intensity_distribution_{res}.png"), bbox_inches="tight")
        plt.close()
    
    print("  1b plots saved")


def plot_1c_class_recall_drop(results_dir):
    """
    Plot per-class recall drop: grouped bar charts.
    """
    drops = load_json(os.path.join(results_dir, "thread1", "1c_class_recall_drop", "per_class_drops.json"))
    if not drops:
        return
    
    out = os.path.join(results_dir, "thread1", "1c_class_recall_drop", "plots")
    os.makedirs(out, exist_ok=True)
    
    for res in RESOLUTIONS:
        for model in MODELS:
            for city in CITIES:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle(f"Per-Class Recall: Upper vs LOCO — {model.upper()} on {city.capitalize()} @ {res}")
                
                # Collect data for grouped bars
                # ax1: absolute recalls (upper vs loco variants)
                # ax2: normalized recall drops
                
                for vi, variant in enumerate(LOCO_VARIANTS):
                    dkey = f"drop_{model}_{variant}_{city}_{res}"
                    if dkey not in drops:
                        continue
                    
                    d = drops[dkey]
                    # Filter: only show classes with >= 15 instances in BOTH upper and loco
                    stypes = sorted([int(k) for k in d.keys()
                                    if d[k].get("upper_instances", 0) >= 15
                                    and d[k].get("loco_instances", 0) >= 15])
                    
                    if not stypes:
                        continue
                    
                    upper_counts = [d[str(st)]["upper_instances"] for st in stypes]
                    loco_counts = [d[str(st)]["loco_instances"] for st in stypes]
                    names = [f"{SHADOW_TYPE_SHORT.get(st, str(st))}\nU:{uc} L:{lc}"
                             for st, uc, lc in zip(stypes, upper_counts, loco_counts)]
                    upper_recalls = [d[str(st)]["upper_recall"] for st in stypes]
                    loco_recalls = [d[str(st)]["loco_recall"] for st in stypes]
                    # Clip normalized deltas to [-1, 1]
                    norm_deltas = [np.clip(d[str(st)]["normalized_delta"], -1.0, 1.0) for st in stypes]
                    
                    x = np.arange(len(stypes))
                    width = 0.18
                    
                    if vi == 0:
                        ax1.bar(x - 2 * width, upper_recalls, width, label="Upper", color="gray", edgecolor="black")
                    ax1.bar(x + (vi - 1) * width, loco_recalls, width, label=variant.upper(), alpha=0.8)
                    
                    ax2.bar(x + (vi - 1.5) * width, norm_deltas, width, label=variant.upper(), alpha=0.8)
                
                ax1.set_xticks(x)
                ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
                ax1.set_ylabel("Boundary Recall")
                ax1.set_title("Absolute Recall")
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3, axis="y")
                
                ax2.set_xticks(x)
                ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
                ax2.set_ylabel("Normalized Recall Drop")
                ax2.set_title("(Upper - LOCO) / Upper")
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3, axis="y")
                ax2.axhline(y=0, color='black', linewidth=0.5)
                ax2.set_ylim(-1.1, 1.1)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out, f"class_recall_{model}_{city}_{res}.png"), bbox_inches="tight")
                plt.close()
    
    print("  1c plots saved")


# ============================================================
# THREAD 3 PLOTS
# ============================================================

def plot_3a_elongation(results_dir):
    """
    Plot boundary IoU vs elongation, per category and area x elongation.
    """
    data = load_json(os.path.join(results_dir, "thread3", "3a_elongation_performance", "elongation_results.json"))
    if not data:
        return
    
    out = os.path.join(results_dir, "thread3", "3a_elongation_performance", "plots")
    os.makedirs(out, exist_ok=True)
    
    for res in RESOLUTIONS:
        for model in MODELS:
            for city in CITIES:
                # Per-category elongation curves: upper vs LOCO vanilla
                ukey = f"upper_{model}_{city}_{res}"
                lkey = f"loco_{model}_vanilla_{city}_{res}"
                
                for key, label_prefix in [(ukey, "Upper"), (lkey, "LOCO-Vanilla")]:
                    if key not in data or data[key].get("n_instances", 0) == 0:
                        continue
                    
                    r = data[key]
                    cats = r.get("per_category", {})
                    if not cats:
                        continue
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for stype_str, cat_data in cats.items():
                        bins = cat_data["bins"]
                        centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in bins if b["count"] > 0 and b["iou_mean"] is not None]
                        ious = [b["iou_mean"] for b in bins if b["count"] > 0 and b["iou_mean"] is not None]
                        
                        if len(centers) < 2:
                            continue
                        
                        name = cat_data["shadow_type_name"]
                        ax.plot(centers, ious, '-o', label=f"{name} (n={cat_data['instance_count']})")
                    
                    ax.set_xlabel("Elongation Ratio (major/minor axis)")
                    ax.set_ylabel("Boundary IoU")
                    ax.set_title(f"{label_prefix} — {model.upper()} on {city.capitalize()} @ {res}\nIoU vs Elongation by Category")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    tag = "upper" if "upper" in key else "loco_vanilla"
                    plt.savefig(os.path.join(out, f"elong_category_{tag}_{model}_{city}_{res}.png"), bbox_inches="tight")
                    plt.close()
                
                # Area x Elongation 2D view (upper only for clarity)
                if ukey in data and data[ukey].get("n_instances", 0) > 0:
                    axe = data[ukey].get("area_x_elongation", {})
                    if axe:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        for area_label, ae_data in axe.items():
                            bins = ae_data["bins"]
                            centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in bins if b["count"] > 0 and b["iou_mean"] is not None]
                            ious = [b["iou_mean"] for b in bins if b["count"] > 0 and b["iou_mean"] is not None]
                            
                            if len(centers) >= 2:
                                ax.plot(centers, ious, '-o', label=f"{area_label} area (n={ae_data['instance_count']})")
                        
                        ax.set_xlabel("Elongation Ratio")
                        ax.set_ylabel("Boundary IoU")
                        ax.set_title(f"Upper — {model.upper()} on {city.capitalize()} @ {res}\nIoU vs Elongation by Area Size")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(out, f"elong_area_{model}_{city}_{res}.png"), bbox_inches="tight")
                        plt.close()
    
    print("  3a plots saved")


def plot_3b_orientation(results_dir):
    """
    Polar plots of boundary IoU by direction (8 bins).
    Key comparison: axis-aligned vs diagonal for MAMNet.
    """
    data = load_json(os.path.join(results_dir, "thread3", "3b_orientation_performance", "orientation_results.json"))
    if not data:
        return
    
    out = os.path.join(results_dir, "thread3", "3b_orientation_performance", "plots")
    os.makedirs(out, exist_ok=True)
    
    direction_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    angles_rad = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    
    for res in RESOLUTIONS:
        for city in CITIES:
            # Compare all 3 models on the same city (upper bound)
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(projection='polar'))
            fig.suptitle(f"Boundary IoU by Shadow Orientation — {city.capitalize()} @ {res} (Upper Bound)")
            
            for ax, model in zip(axes, MODELS):
                key = f"upper_{model}_{city}_{res}"
                if key not in data or data[key].get("n_instances", 0) == 0:
                    ax.set_title(f"{model.upper()} (no data)")
                    continue
                
                dr = data[key]["direction_results"]
                ious = [dr.get(d, {}).get("iou_mean") for d in direction_order]
                counts = [dr.get(d, {}).get("count", 0) for d in direction_order]
                
                # Replace None with 0
                ious_plot = [v if v is not None else 0 for v in ious]
                
                # Close the polygon
                ious_closed = ious_plot + [ious_plot[0]]
                angles_closed = np.append(angles_rad, angles_rad[0])
                
                ax.plot(angles_closed, ious_closed, '-o', linewidth=2, markersize=5)
                ax.fill(angles_closed, ious_closed, alpha=0.15)
                ax.set_xticks(angles_rad)
                ax.set_xticklabels(direction_order)
                ax.set_title(f"{model.upper()}")
                ax.set_ylim(0, max(ious_plot) * 1.2 if max(ious_plot) > 0 else 1)
                
                # Highlight axis vs diagonal
                avd = data[key].get("axis_vs_diagonal", {})
                diff = avd.get("axis_minus_diag")
                if diff is not None:
                    ax.annotate(f"Axis-Diag: {diff:+.3f}", xy=(0.5, -0.1),
                              xycoords='axes fraction', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"orientation_polar_{city}_{res}.png"), bbox_inches="tight")
            plt.close()
        
        # Cross-model axis vs diagonal comparison table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Axis-Aligned vs Diagonal IoU Gap — {res} (Upper Bound)\n(Positive = axis-aligned performs better)")
        
        table_data = []
        for model in MODELS:
            row = []
            for city in CITIES:
                key = f"upper_{model}_{city}_{res}"
                if key in data:
                    diff = data[key].get("axis_vs_diagonal", {}).get("axis_minus_diag")
                    row.append(f"{diff:+.4f}" if diff is not None else "N/A")
                else:
                    row.append("N/A")
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        rowLabels=[m.upper() for m in MODELS],
                        colLabels=[c.capitalize() for c in CITIES],
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"axis_vs_diagonal_table_{res}.png"), bbox_inches="tight")
        plt.close()
    
    print("  3b plots saved")


def plot_3e_distributions(results_dir):
    """
    Compare shadow geometry distributions across cities.
    """
    data = load_json(os.path.join(results_dir, "thread3", "3e_geometry_distributions", "geometry_distributions.json"))
    if not data:
        return
    
    out = os.path.join(results_dir, "thread3", "3e_geometry_distributions", "plots")
    os.makedirs(out, exist_ok=True)
    
    for res in RESOLUTIONS:
        # Orientation histograms across cities
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(f"Shadow Orientation Distribution — {res}")
        
        dir_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        
        for ax, city in zip(axes, CITIES):
            key = f"{city}_{res}"
            if key not in data:
                continue
            
            if "overall" not in data[key]:
                continue
            
            hist = data[key]["overall"]["orientation_histogram"]
            # Normalize to proportions
            total = sum(hist)
            props = [h / total if total > 0 else 0 for h in hist]
            
            ax.bar(range(8), props, color=CITY_COLORS[city], alpha=0.7, edgecolor="black")
            ax.set_xticks(range(8))
            ax.set_xticklabels(dir_labels)
            ax.set_title(f"{city.capitalize()} (n={data[key]['n_instances']})")
            if ax == axes[0]:
                ax.set_ylabel("Proportion")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"orientation_dist_{res}.png"), bbox_inches="tight")
        plt.close()
        
        # Summary statistics table
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(f"Shadow Geometry Summary — {res}")
        
        table_data = []
        for city in CITIES:
            key = f"{city}_{res}"
            if key not in data or "overall" not in data[key]:
                table_data.append([city, "N/A"] * 5)
                continue
            
            d = data[key]
            ov = d["overall"]
            sun_az = d.get("empirical_sun_azimuth_from_buildings", "N/A")
            sun_str = f"{sun_az:.1f}°" if isinstance(sun_az, float) else sun_az
            
            table_data.append([
                city.capitalize(),
                f"{ov['major_axis']['median']:.1f}",
                f"{ov['elongation']['median']:.2f}",
                f"{ov['area']['median']:.0f}",
                sun_str,
                str(d["n_instances"]),
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=["City", "Median Major Axis", "Median Elongation", "Median Area", "Est. Sun Az.", "N Instances"],
            loc='center', cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"geometry_summary_{res}.png"), bbox_inches="tight")
        plt.close()
    
    print("  3e plots saved")


# ============================================================
# THREAD 4 PLOTS
# ============================================================

def plot_4a_spatial_heatmaps(results_dir):
    """
    Plot ΔIoU heatmaps (2x2 grid) and radial bar charts.
    """
    deltas = load_json(os.path.join(results_dir, "thread4", "4a_spatial_error_maps", "spatial_deltas.json"))
    if not deltas:
        return
    
    out = os.path.join(results_dir, "thread4", "4a_spatial_error_maps", "plots")
    os.makedirs(out, exist_ok=True)
    
    for res in RESOLUTIONS:
        for model in MODELS:
            for city in CITIES:
                # Collect all variants for this (model, city, res)
                variant_grids = {}
                variant_radials = {}
                
                for variant in LOCO_VARIANTS:
                    dkey = f"delta_{model}_{variant}_{city}_{res}"
                    if dkey not in deltas:
                        continue
                    
                    grid = deltas[dkey]["grid"]
                    radial = deltas[dkey]["radial"]
                    
                    # Build 2x2 grid
                    grid_arr = np.full((2, 2), np.nan)
                    for cell in grid:
                        if cell.get("sufficient_data") is not False and "delta_iou" in cell:
                            grid_arr[cell["row"], cell["col"]] = cell["delta_iou"]
                    
                    variant_grids[variant] = grid_arr
                    variant_radials[variant] = radial
                
                if not variant_grids:
                    continue
                
                # Plot heatmaps
                n_variants = len(variant_grids)
                fig, axes = plt.subplots(1, n_variants, figsize=(4 * n_variants, 4))
                if n_variants == 1:
                    axes = [axes]
                fig.suptitle(f"Spatial ΔIoU (Upper - LOCO) — {model.upper()} on {city.capitalize()} @ {res}\n(Red = transfer hurts more)")
                
                # Find global vmin/vmax for consistent colorbar
                all_vals = []
                for g in variant_grids.values():
                    all_vals.extend(g[~np.isnan(g)].tolist())
                
                if all_vals:
                    vmax = max(abs(min(all_vals)), abs(max(all_vals)), 0.01)
                    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                else:
                    norm = None
                
                for ax, (variant, grid_arr) in zip(axes, variant_grids.items()):
                    im = ax.imshow(grid_arr, cmap='RdBu', norm=norm, interpolation='nearest')
                    ax.set_title(variant.upper())
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["Left", "Right"])
                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(["Top", "Bottom"])
                    
                    # Annotate cells
                    for r in range(2):
                        for c in range(2):
                            val = grid_arr[r, c]
                            if not np.isnan(val):
                                ax.text(c, r, f"{val:.3f}", ha='center', va='center', fontsize=11, fontweight='bold')
                            else:
                                ax.text(c, r, "N/A", ha='center', va='center', fontsize=10, color='gray')
                
                fig.colorbar(im, ax=axes, shrink=0.8, label="ΔIoU")
                plt.tight_layout()
                plt.savefig(os.path.join(out, f"spatial_delta_{model}_{city}_{res}.png"), bbox_inches="tight")
                plt.close()
                
                # Radial bar chart
                fig, ax = plt.subplots(figsize=(8, 4))
                ring_labels = ["Center", "Mid-ring", "Periphery"]
                x = np.arange(3)
                width = 0.2
                
                for vi, (variant, radial) in enumerate(variant_radials.items()):
                    vals = []
                    for ring in radial:
                        if ring.get("sufficient_data") is not False and "delta_iou" in ring:
                            vals.append(ring["delta_iou"])
                        else:
                            vals.append(0)
                    ax.bar(x + vi * width, vals, width, label=variant.upper(), alpha=0.8)
                
                ax.set_xticks(x + width * (len(variant_radials) - 1) / 2)
                ax.set_xticklabels(ring_labels)
                ax.set_ylabel("ΔIoU (Upper - LOCO)")
                ax.set_title(f"Radial ΔIoU — {model.upper()} on {city.capitalize()} @ {res}")
                ax.legend()
                ax.grid(True, alpha=0.3, axis="y")
                ax.axhline(y=0, color='black', linewidth=0.5)
                
                plt.tight_layout()
                plt.savefig(os.path.join(out, f"radial_delta_{model}_{city}_{res}.png"), bbox_inches="tight")
                plt.close()
    
    print("  4a plots saved")


def plot_4b_density(results_dir):
    """
    Plot shadow density maps and JSD comparisons.
    """
    density_data = load_json(os.path.join(results_dir, "thread4", "4b_spatial_density", "density_maps.json"))
    jsd_data = load_json(os.path.join(results_dir, "thread4", "4b_spatial_density", "jsd_results.json"))
    
    if not density_data:
        return
    
    out = os.path.join(results_dir, "thread4", "4b_spatial_density", "plots")
    os.makedirs(out, exist_ok=True)
    
    for res in RESOLUTIONS:
        # Side-by-side density maps
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Shadow Spatial Density — {res}\n(Brighter = more shadow pixels)")
        
        vmax = 0
        for city in CITIES:
            key = f"{city}_{res}"
            if key in density_data:
                vmax = max(vmax, np.array(density_data[key]).max())
        
        for ax, city in zip(axes, CITIES):
            key = f"{city}_{res}"
            if key not in density_data:
                continue
            
            density = np.array(density_data[key])
            im = ax.imshow(density, cmap='hot', vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_title(f"{city.capitalize()}")
            ax.axis('off')
        
        fig.colorbar(im, ax=axes, shrink=0.8, label="Shadow pixel fraction")
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"density_maps_{res}.png"), bbox_inches="tight")
        plt.close()
    
    # JSD table
    if jsd_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Shadow Spatial Density — Jensen-Shannon Divergence")
        
        table_data = []
        row_labels = []
        for key, val in sorted(jsd_data.items()):
            row_labels.append(f"{val['city_a'].capitalize()} vs {val['city_b'].capitalize()} ({val['res']})")
            table_data.append([f"{val['jsd']:.4f}"])
        
        table = ax.table(cellText=table_data, rowLabels=row_labels,
                        colLabels=["JSD"], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out, "jsd_table.png"), bbox_inches="tight")
        plt.close()
    
    print("  4b plots saved")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=OUTPUT_BASE)
    args = parser.parse_args()
    
    print(f"Generating plots from: {args.results_dir}")
    
    plot_1a_fp_clustering(args.results_dir)
    plot_1b_intensity_curves(args.results_dir)
    plot_1c_class_recall_drop(args.results_dir)
    if HAS_1D_PLOT:
        try:
            plot_1d_probe(args.results_dir)
        except Exception as e:
            print(f"  1d plots skipped: {e}")
    plot_3a_elongation(args.results_dir)
    plot_3b_orientation(args.results_dir)
    plot_3e_distributions(args.results_dir)
    plot_4a_spatial_heatmaps(args.results_dir)
    plot_4b_density(args.results_dir)
    
    print("\nAll plots generated!")


if __name__ == "__main__":
    main()