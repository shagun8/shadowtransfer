"""
Aggregate Cross-Resolution Transfer Results and Create Publication Figures
Creates CVPR-quality visualizations for resolution transferability analysis

Usage:
    python aggregate_res_results.py --eval_dirs dir1 dir2 dir3 ... --output_dir ./aggregate_results
"""

import os
import glob
import json
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Colorblind-friendly colors
COLORS = {
    'native': '#2E86AB',      # Blue
    'transfer': '#A23B72',    # Purple/Magenta
    'chicago': '#E63946',     # Red
    'miami': '#06AED5',       # Cyan
    'phoenix': '#F77F00',     # Orange
    'midres_to_highres': '#2A9D8F',  # Teal
    'highres_to_midres': '#E76F51',  # Coral
}

def extract_timestamp_from_dir(dir_path):
    """
    Extract timestamp from directory name like 'res_evaluation_midres_to_highres_phoenix_20251028_123456'
    Returns: timestamp as string 'YYYYMMDD_HHMMSS'
    """
    dir_name = os.path.basename(dir_path)
    parts = dir_name.split('_')
    
    # Look for pattern: YYYYMMDD_HHMMSS at the end
    if len(parts) >= 2:
        try:
            # Last two parts should be date and time
            timestamp = f"{parts[-2]}_{parts[-1]}"
            # Validate format
            from datetime import datetime
            datetime.strptime(timestamp, '%Y%m%d_%H%M%S')
            return timestamp
        except:
            pass
    
    return None


def get_transfer_direction_from_dir(dir_path):
    """
    Extract transfer direction from directory name
    Returns: tuple (source_res, target_res) or None
    """
    dir_name = os.path.basename(dir_path)
    
    # Pattern: res_evaluation_{source}to{target}_phoenix_{timestamp}
    if 'midrestohighres' in dir_name or 'midres_to_highres' in dir_name:
        return ('midres', 'highres')
    elif 'highrestomidres' in dir_name or 'highres_to_midres' in dir_name:
        return ('highres', 'midres')
    
    # Try to load config to get direction
    config_path = os.path.join(dir_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return (config.get('source_resolution'), config.get('target_resolution'))
        except:
            pass
    
    return None


def filter_latest_evaluations(eval_dirs):
    """
    Filter evaluation directories to collect all cities for each transfer direction.
    For each direction, either:
    - Use the multi-city 'all' directory (if exists and most recent)
    - Collect all single-city directories from the most recent timestamp batch
    """
    # Group directories by transfer direction and type
    direction_groups = {}
    
    for eval_dir in eval_dirs:
        direction = get_transfer_direction_from_dir(eval_dir)
        timestamp = extract_timestamp_from_dir(eval_dir)
        
        if direction and timestamp:
            dir_key = f"{direction[0]}→{direction[1]}"
            dir_name = os.path.basename(eval_dir)
            
            # Check if this is a multi-city 'all' directory or single city
            is_all = '_all_' in dir_name
            
            # Extract city if single-city run
            city = None
            if not is_all:
                for c in ['chicago', 'miami', 'phoenix']:
                    if f'_{c}_' in dir_name:
                        city = c
                        break
            
            if dir_key not in direction_groups:
                direction_groups[dir_key] = {
                    'all_runs': [],
                    'single_city_runs': []
                }
            
            entry = {
                'dir': eval_dir,
                'timestamp': timestamp,
                'direction': direction,
                'is_all': is_all,
                'city': city
            }
            
            if is_all:
                direction_groups[dir_key]['all_runs'].append(entry)
            else:
                direction_groups[dir_key]['single_city_runs'].append(entry)
    
    # Select directories for each direction
    selected_dirs = []
    
    for dir_key, groups in direction_groups.items():
        print(f"\n  {dir_key}:")
        
        # Prefer 'all' runs if available
        if groups['all_runs']:
            groups['all_runs'].sort(key=lambda x: x['timestamp'], reverse=True)
            latest_all = groups['all_runs'][0]
            selected_dirs.append(latest_all['dir'])
            print(f"    Using multi-city evaluation from {latest_all['timestamp']}")
            print(f"    Path: {latest_all['dir']}")
        
        # Otherwise, collect all cities from the most recent timestamp batch
        elif groups['single_city_runs']:
            # Sort by timestamp
            groups['single_city_runs'].sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Get the most recent timestamp
            latest_timestamp = groups['single_city_runs'][0]['timestamp']
            
            # Collect all cities from this timestamp (or very close timestamps)
            # Allow 5 minute window for jobs submitted together
            from datetime import datetime, timedelta
            latest_dt = datetime.strptime(latest_timestamp, '%Y%m%d_%H%M%S')
            
            batch_dirs = []
            for entry in groups['single_city_runs']:
                entry_dt = datetime.strptime(entry['timestamp'], '%Y%m%d_%H%M%S')
                time_diff = abs((latest_dt - entry_dt).total_seconds())
                
                # Include if within 10 minutes of latest
                if time_diff <= 600:  # 10 minutes
                    batch_dirs.append(entry)
            
            # Add all directories from this batch
            cities_found = set()
            for entry in batch_dirs:
                selected_dirs.append(entry['dir'])
                cities_found.add(entry['city'])
            
            print(f"    Using single-city evaluations from timestamp ~{latest_timestamp}")
            print(f"    Cities: {', '.join(sorted(cities_found))}")
            print(f"    Directories: {len(batch_dirs)}")
            for entry in batch_dirs:
                print(f"      - {entry['city']}: {entry['timestamp']}")
    
    return selected_dirs


def load_evaluation_results(eval_dir):
    """
    Load results from a single evaluation directory
    Returns: dict with config, bootstrap_data, and summary
    """
    # Load config
    config_path = os.path.join(eval_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load bootstrap samples
    pattern = f"bootstrap_samples_*.pkl"
    bootstrap_files = glob.glob(os.path.join(eval_dir, pattern))
    
    if len(bootstrap_files) == 0:
        raise FileNotFoundError(f"No bootstrap files found in {eval_dir}")
    
    bootstrap_path = bootstrap_files[0]  # Take the first match
    with open(bootstrap_path, 'rb') as f:
        bootstrap_data = pickle.load(f)
    
    # Load summary CSV
    summary_pattern = f"summary_table_*.csv"
    summary_files = glob.glob(os.path.join(eval_dir, summary_pattern))
    
    if len(summary_files) > 0:
        summary_df = pd.read_csv(summary_files[0])
    else:
        summary_df = None
    
    # Load SRI summary if available
    sri_pattern = f"sri_summary_*.csv"
    sri_files = glob.glob(os.path.join(eval_dir, sri_pattern))
    
    if len(sri_files) > 0:
        sri_df = pd.read_csv(sri_files[0])
    else:
        sri_df = None
    
    return {
        'config': config,
        'bootstrap_data': bootstrap_data,
        'summary_df': summary_df,
        'sri_df': sri_df,
        'eval_dir': eval_dir
    }


def aggregate_all_results(eval_dirs):
    """
    Load and aggregate results from multiple evaluation directories
    """
    all_results = []
    
    for eval_dir in eval_dirs:
        print(f"Loading results from: {eval_dir}")
        try:
            results = load_evaluation_results(eval_dir)
            all_results.append(results)
        except Exception as e:
            print(f"  WARNING: Failed to load {eval_dir}: {e}")
            continue
    
    if len(all_results) == 0:
        raise ValueError("No valid evaluation results found!")
    
    print(f"\nLoaded {len(all_results)} evaluation results")
    return all_results


def create_aggregated_dataframe(all_results):
    """
    Create a unified dataframe with all results
    """
    rows = []
    
    for result in all_results:
        config = result['config']
        bootstrap_data = result['bootstrap_data']
        
        source_res = config.get('source_resolution', 'unknown')
        target_res = config.get('target_resolution', 'unknown')
        direction = f"{source_res}→{target_res}"
        
        # Extract results for each city
        for pair_name, metrics in bootstrap_data.items():
            # Parse city from pair_name (e.g., "chicago_midres→highres")
            parts = pair_name.split('_')
            city = parts[0]
            
            row = {
                'city': city,
                'source_resolution': source_res,
                'target_resolution': target_res,
                'direction': direction,
            }
            
            # Add metric means and CIs
            for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER']:
                if metric in metrics:
                    native_samples = metrics[metric]['native']
                    transfer_samples = metrics[metric]['transfer']
                    res_gap_samples = metrics[metric]['res_gap']
                    
                    row[f'{metric}_native_mean'] = np.mean(native_samples)
                    row[f'{metric}_native_ci_lower'] = np.percentile(native_samples, 2.5)
                    row[f'{metric}_native_ci_upper'] = np.percentile(native_samples, 97.5)
                    
                    row[f'{metric}_transfer_mean'] = np.mean(transfer_samples)
                    row[f'{metric}_transfer_ci_lower'] = np.percentile(transfer_samples, 2.5)
                    row[f'{metric}_transfer_ci_upper'] = np.percentile(transfer_samples, 97.5)
                    
                    row[f'{metric}_res_gap_mean'] = np.mean(res_gap_samples)
                    row[f'{metric}_res_gap_ci_lower'] = np.percentile(res_gap_samples, 2.5)
                    row[f'{metric}_res_gap_ci_upper'] = np.percentile(res_gap_samples, 97.5)
            
            # Add SRI metrics if available
            for sri_metric in ['SRI_F1', 'SRI_Shadow_IOU', 'SRI_mIOU']:
                if sri_metric in metrics:
                    sri_samples = metrics[sri_metric]['sri']
                    row[f'{sri_metric}_mean'] = np.mean(sri_samples)
                    row[f'{sri_metric}_ci_lower'] = np.percentile(sri_samples, 2.5)
                    row[f'{sri_metric}_ci_upper'] = np.percentile(sri_samples, 97.5)
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def plot_figure1_resolution_transfer_matrix(df, output_path):
    """
    Figure 1: Resolution Transfer Matrix with Performance Degradation
    Shows transfer performance compared to native upper bound
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics_to_plot = ['F1', 'Shadow_IOU']
    metric_labels = ['F1 Score (%)', 'Shadow IoU (%)']
    
    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        
        # Prepare data
        cities = ['chicago', 'miami', 'phoenix']
        directions = sorted(df['direction'].unique())
        
        x = np.arange(len(cities))
        width = 0.25
        
        # Plot bars for each direction
        for dir_idx, direction in enumerate(directions):
            df_dir = df[df['direction'] == direction]
            
            # Extract source and target
            source_res = direction.split('→')[0]
            target_res = direction.split('→')[1]
            
            native_means = []
            native_errs = []
            transfer_means = []
            transfer_errs = []
            
            for city in cities:
                city_data = df_dir[df_dir['city'] == city]
                if len(city_data) > 0:
                    native_mean = city_data[f'{metric}_native_mean'].values[0]
                    native_ci_lower = city_data[f'{metric}_native_ci_lower'].values[0]
                    native_ci_upper = city_data[f'{metric}_native_ci_upper'].values[0]
                    
                    transfer_mean = city_data[f'{metric}_transfer_mean'].values[0]
                    transfer_ci_lower = city_data[f'{metric}_transfer_ci_lower'].values[0]
                    transfer_ci_upper = city_data[f'{metric}_transfer_ci_upper'].values[0]
                    
                    native_means.append(native_mean)
                    native_errs.append([native_mean - native_ci_lower, native_ci_upper - native_mean])
                    
                    transfer_means.append(transfer_mean)
                    transfer_errs.append([transfer_mean - transfer_ci_lower, transfer_ci_upper - transfer_mean])
                else:
                    native_means.append(0)
                    native_errs.append([0, 0])
                    transfer_means.append(0)
                    transfer_errs.append([0, 0])
            
            native_errs = np.array(native_errs).T
            transfer_errs = np.array(transfer_errs).T
            
            offset = (dir_idx - 0.5) * width
            
            # Color based on direction
            color_key = 'midres_to_highres' if 'midres' in source_res else 'highres_to_midres'
            
            # Upper bound bars (with hatching to distinguish)
            ax.bar(x + offset, native_means, width * 0.45, 
                   label=f'{target_res.capitalize()} Native (Upper Bound)' if dir_idx == 0 else '', 
                   color=COLORS[color_key], alpha=0.9, hatch='///',
                   yerr=native_errs, capsize=3, edgecolor='black', linewidth=1.5)
            
            # Transfer bars
            ax.bar(x + offset + width * 0.5, transfer_means, width * 0.45, 
                   label=f'{source_res.capitalize()}→{target_res.capitalize()} Transfer', 
                   color=COLORS[color_key], alpha=0.5,
                   yerr=transfer_errs, capsize=3, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('City', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in cities])
        ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add title
        ax.set_title(f'{label.split()[0]} Performance: Transfer vs Native', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 1: {output_path}")
    plt.close()


def plot_figure2_sri_violin_plots(all_results, output_path):
    """
    Figure 2: Scale Robustness Index (SRI) Comparison
    Box plots showing SRI distribution across cities using per-pair data
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sri_metrics = ['SRI_F1', 'SRI_Shadow_IOU', 'SRI_mIOU']
    sri_labels = ['SRI (F1)', 'SRI (Shadow IoU)', 'SRI (mIOU)']
    
    for idx, (sri_metric, label) in enumerate(zip(sri_metrics, sri_labels)):
        ax = axes[idx]
        
        # Collect per-pair SRI data from each evaluation result
        plot_data = []
        
        for result in all_results:
            eval_dir = result['eval_dir']
            config = result['config']
            direction = f"{config['source_resolution']}→{config['target_resolution']}"
            
            # Load per-pair SRI CSV files
            sri_csv_pattern = os.path.join(eval_dir, '*', 'sri_per_pair_*.csv')
            sri_files = glob.glob(sri_csv_pattern)
            
            if len(sri_files) == 0:
                # Try without subdirectory
                sri_csv_pattern = os.path.join(eval_dir, 'sri_per_pair_*.csv')
                sri_files = glob.glob(sri_csv_pattern)
            
            for sri_file in sri_files:
                # Extract city from filename
                filename = os.path.basename(sri_file)
                city = None
                for c in ['chicago', 'miami', 'phoenix']:
                    if c in filename:
                        city = c
                        break
                
                if city is None:
                    continue
                
                # Load per-pair data
                try:
                    sri_df = pd.read_csv(sri_file)
                    if sri_metric in sri_df.columns:
                        for value in sri_df[sri_metric]:
                            if pd.notna(value):
                                plot_data.append({
                                    'City': city.capitalize(),
                                    'Direction': direction,
                                    'SRI': value
                                })
                except Exception as e:
                    print(f"Warning: Could not load {sri_file}: {e}")
        
        if len(plot_data) == 0:
            print(f"Warning: No per-pair data found for {sri_metric}")
            continue
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create box plot data structure
        cities = ['Chicago', 'Miami', 'Phoenix']
        directions = sorted(plot_df['Direction'].unique())
        
        positions = []
        data_to_plot = []
        labels_list = []
        colors_list = []
        
        pos = 0
        for city in cities:
            for direction in directions:
                subset = plot_df[(plot_df['City'] == city) & (plot_df['Direction'] == direction)]
                if len(subset) > 0:
                    positions.append(pos)
                    data_to_plot.append(subset['SRI'].values)
                    labels_list.append(f"{city}\n{direction}")
                    color_key = 'midres_to_highres' if 'midres' in direction.split('→')[0] else 'highres_to_midres'
                    colors_list.append(COLORS[color_key])
                    pos += 1
            pos += 0.5  # Add space between cities
        
        if len(data_to_plot) == 0:
            continue
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='black'),
                        meanprops=dict(linewidth=2, color='red', linestyle='--'))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Perfect Robustness')
        
        # Styling
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('City and Transfer Direction', fontsize=11, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_title(f'{label} Distribution', fontweight='bold')
        
        # Add shaded regions for interpretation
        y_min, y_max = ax.get_ylim()
        ax.axhspan(0, y_max, alpha=0.1, color='red', label='Degradation')
        ax.axhspan(y_min, 0, alpha=0.1, color='green', label='Improvement')
        
        if idx == 0:
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 2: {output_path}")
    plt.close()


def plot_figure3_scatter_with_identity(all_results, output_path):
    """
    Figure 3: Scatter Plot with Identity Line and Confidence Ellipses
    Shows per-image performance relationship
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get all unique directions
    directions = []
    for result in all_results:
        config = result['config']
        direction = f"{config['source_resolution']}→{config['target_resolution']}"
        if direction not in directions:
            directions.append(direction)
    
    directions = sorted(directions)
    
    for ax_idx, direction in enumerate(directions):
        if ax_idx >= 2:
            break
        
        ax = axes[ax_idx]
        
        # Collect data for this direction
        for result in all_results:
            config = result['config']
            result_direction = f"{config['source_resolution']}→{config['target_resolution']}"
            
            if result_direction != direction:
                continue
            
            bootstrap_data = result['bootstrap_data']
            
            for pair_name, metrics in bootstrap_data.items():
                city = pair_name.split('_')[0]
                
                if 'F1' in metrics:
                    native_samples = metrics['F1']['native']
                    transfer_samples = metrics['F1']['transfer']
                    
                    # Plot scatter points
                    ax.scatter(native_samples, transfer_samples, 
                             alpha=0.3, s=20, color=COLORS[city], label=city.capitalize())
        
        # Plot identity line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, linewidth=2, label='Perfect Transfer')
        
        # Styling
        ax.set_xlabel('Native Model F1 Score (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Transfer Model F1 Score (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Transfer Performance: {direction}', fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_aspect('equal')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right', 
                 frameon=True, fancybox=True, shadow=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Figure 3: {output_path}")
    plt.close()


def create_table1_comprehensive_metrics(df, output_path_csv, output_path_tex):
    """
    Table 1: Comprehensive Metrics Table
    LaTeX format with all statistics
    """
    # Prepare table data
    table_rows = []
    
    for _, row in df.iterrows():
        city = row['city'].capitalize()
        direction = row['direction']
        
        # Format with mean ± CI width
        def format_metric(mean, ci_lower, ci_upper):
            ci_width = (ci_upper - ci_lower) / 2
            return f"{mean:.2f}±{ci_width:.2f}"
        
        table_row = {
            'City': city,
            'Direction': direction,
        }
        
        for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER']:
            table_row[f'{metric} Native'] = format_metric(
                row[f'{metric}_native_mean'],
                row[f'{metric}_native_ci_lower'],
                row[f'{metric}_native_ci_upper']
            )
            table_row[f'{metric} Transfer'] = format_metric(
                row[f'{metric}_transfer_mean'],
                row[f'{metric}_transfer_ci_lower'],
                row[f'{metric}_transfer_ci_upper']
            )
        
        # Add SRI if available
        for sri_metric in ['SRI_F1', 'SRI_Shadow_IOU', 'SRI_mIOU']:
            if f'{sri_metric}_mean' in row and pd.notna(row[f'{sri_metric}_mean']):
                table_row[sri_metric] = format_metric(
                    row[f'{sri_metric}_mean'],
                    row[f'{sri_metric}_ci_lower'],
                    row[f'{sri_metric}_ci_upper']
                )
        
        table_rows.append(table_row)
    
    # Create DataFrame
    table_df = pd.DataFrame(table_rows)
    
    # Save CSV
    table_df.to_csv(output_path_csv, index=False)
    print(f"Saved Table 1 (CSV): {output_path_csv}")
    
    # Create LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Cross-Resolution Transfer Performance. Native: model trained and tested on target resolution (upper bound). Transfer: model trained on source resolution, tested on target resolution. SRI (Scale Robustness Index): per-pair difference (Native - Transfer), where values closer to 0 indicate better scale robustness. Values shown as mean±CI.}")
    latex_lines.append("\\label{tab:resolution_transfer}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{")
    latex_lines.append("\\begin{tabular}{llcccccccccc}")
    latex_lines.append("\\toprule")
    
    # Header
    latex_lines.append("\\multirow{2}{*}{\\textbf{City}} & \\multirow{2}{*}{\\textbf{Direction}} & " +
                      "\\multicolumn{2}{c}{\\textbf{F1 (\\%)}} & " +
                      "\\multicolumn{2}{c}{\\textbf{Shadow IoU (\\%)}} & " +
                      "\\multicolumn{2}{c}{\\textbf{mIOU (\\%)}} & " +
                      "\\multicolumn{3}{c}{\\textbf{SRI}} \\\\")
    latex_lines.append("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-11}")
    latex_lines.append(" & & Native & Transfer & Native & Transfer & Native & Transfer & F1 & IoU & mIOU \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows
    for _, row in table_df.iterrows():
        city = row['City']
        direction = row['Direction'].replace('→', '$\\rightarrow$')
        
        line = f"{city} & {direction}"
        
        for metric in ['F1', 'Shadow_IOU', 'mIOU']:
            line += f" & {row[f'{metric} Native']}"
            line += f" & {row[f'{metric} Transfer']}"
        
        # Add SRI
        for sri_metric in ['SRI_F1', 'SRI_Shadow_IOU', 'SRI_mIOU']:
            if sri_metric in row:
                line += f" & {row[sri_metric]}"
            else:
                line += " & -"
        
        line += " \\\\"
        latex_lines.append(line)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table*}")
    
    # Save LaTeX
    with open(output_path_tex, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"Saved Table 1 (LaTeX): {output_path_tex}")
    
    return table_df


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate Cross-Resolution Transfer Results and Create Publication Figures'
    )
    parser.add_argument('--eval_dirs', type=str, nargs='+', required=True,
                       help='List of evaluation result directories')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for aggregate results')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Pattern to search for eval directories (e.g., "res_evaluation_*"). Will automatically use latest timestamp for each transfer direction.')
    
    args = parser.parse_args()
    
    # If pattern provided, search for matching directories and filter to latest
    if args.pattern:
        base_dir = args.eval_dirs[0] if len(args.eval_dirs) == 1 else '.'
        all_matching_dirs = glob.glob(os.path.join(base_dir, args.pattern))
        print(f"Found {len(all_matching_dirs)} directories matching pattern: {args.pattern}")
        
        if len(all_matching_dirs) == 0:
            raise ValueError(f"No directories found matching pattern: {args.pattern}")
        
        print("\nFiltering to latest evaluation for each transfer direction...")
        eval_dirs = filter_latest_evaluations(all_matching_dirs)
        
        print(f"\nSelected {len(eval_dirs)} evaluation directory(ies)")
        print("(May include multiple directories for different cities in the same transfer direction)")
    else:
        eval_dirs = args.eval_dirs
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}\n")
    
    # Load all results
    print("="*60)
    print("Loading Evaluation Results")
    print("="*60)
    all_results = aggregate_all_results(eval_dirs)
    
    # Create aggregated dataframe
    print("\n" + "="*60)
    print("Creating Aggregated DataFrame")
    print("="*60)
    df = create_aggregated_dataframe(all_results)
    print(f"Aggregated data shape: {df.shape}")
    print(f"Cities: {df['city'].unique()}")
    print(f"Directions: {df['direction'].unique()}")
    
    # Save aggregated data
    df.to_csv(os.path.join(args.output_dir, 'aggregated_results.csv'), index=False)
    print(f"\nSaved aggregated data: {os.path.join(args.output_dir, 'aggregated_results.csv')}")
    
    # Create figures
    print("\n" + "="*60)
    print("Creating Publication Figures")
    print("="*60 + "\n")
    
    # Figure 1: Resolution Transfer Matrix
    plot_figure1_resolution_transfer_matrix(
        df, 
        os.path.join(args.output_dir, 'figure1_resolution_transfer_matrix.png')
    )
    plot_figure1_resolution_transfer_matrix(
        df, 
        os.path.join(args.output_dir, 'figure1_resolution_transfer_matrix.pdf')
    )
    
    # Figure 2: SRI Violin Plots
    plot_figure2_sri_violin_plots(
        all_results,
        os.path.join(args.output_dir, 'figure2_sri_comparison.png')
    )
    plot_figure2_sri_violin_plots(
        all_results,
        os.path.join(args.output_dir, 'figure2_sri_comparison.pdf')
    )
    
    # Figure 3: Scatter Plots
    plot_figure3_scatter_with_identity(
        all_results,
        os.path.join(args.output_dir, 'figure3_scatter_identity.png')
    )
    plot_figure3_scatter_with_identity(
        all_results,
        os.path.join(args.output_dir, 'figure3_scatter_identity.pdf')
    )
    
    # Table 1: Comprehensive Metrics
    create_table1_comprehensive_metrics(
        df,
        os.path.join(args.output_dir, 'table1_comprehensive_metrics.csv'),
        os.path.join(args.output_dir, 'table1_comprehensive_metrics.tex')
    )
    
    print("\n" + "="*60)
    print("All Figures and Tables Created Successfully!")
    print("="*60)
    print(f"\nOutput location: {args.output_dir}")
    print("\nGenerated files:")
    print("  - figure1_resolution_transfer_matrix.png/pdf")
    print("  - figure2_sri_comparison.png/pdf")
    print("  - figure3_scatter_identity.png/pdf")
    print("  - table1_comprehensive_metrics.csv/tex")
    print("  - aggregated_results.csv")
    print("\n")


if __name__ == '__main__':
    main()