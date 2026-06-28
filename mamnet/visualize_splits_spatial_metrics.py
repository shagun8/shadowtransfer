"""
Quick visualization of spatial metrics from generated splits
Helps validate spatial sampling strategies before training
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse

# Styling
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Configuration
STRATEGIES = ['random', 'clustered', 'dispersed']
STRATEGY_COLORS = {
    'random': '#2E86AB',
    'clustered': '#A23B72',
    'dispersed': '#F18F01'
}
STRATEGY_LABELS = {
    'random': 'Random',
    'clustered': 'Clustered',
    'dispersed': 'Dispersed'
}


def load_spatial_metrics_from_splits(split_dir: str) -> pd.DataFrame:
    """
    Load spatial metrics from all split JSON files
    
    Args:
        split_dir: Directory containing split JSON files
    
    Returns:
        DataFrame with spatial metrics for each configuration
    """
    results = []
    
    # Pattern: {city}_{resolution}_{strategy}_N{n:03d}_seed{seed}.json
    pattern = os.path.join(split_dir, '*_N*_seed*.json')
    split_files = glob.glob(pattern)
    
    # Also include N=600 files (no seed suffix)
    pattern_n600 = os.path.join(split_dir, '*_N600.json')
    split_files.extend(glob.glob(pattern_n600))
    
    print(f"Found {len(split_files)} split files")
    
    for split_file in split_files:
        try:
            with open(split_file, 'r') as f:
                data = json.load(f)
            
            # Extract configuration
            city = data['city']
            resolution = data['resolution']
            strategy = data['strategy']
            n_samples = data['n_samples']
            seed = data['random_seed']
            
            # Extract spatial metrics
            spatial_metrics = data.get('spatial_metrics', {})
            
            if n_samples == 0:
                continue  # Skip N=0
            
            row = {
                'city': city,
                'resolution': resolution,
                'strategy': strategy,
                'n_samples': n_samples,
                'seed': seed,
                'mean_pairwise_distance_km': spatial_metrics.get('mean_pairwise_distance_km', np.nan),
                'mean_min_distance_km': spatial_metrics.get('mean_min_distance_km', np.nan),
                'median_min_distance_km': spatial_metrics.get('median_min_distance_km', np.nan),
                'convex_hull_area_km2': spatial_metrics.get('convex_hull_area_km2', np.nan),
                'unique_tiles': spatial_metrics.get('unique_tiles', np.nan),
                'standard_distance': spatial_metrics.get('standard_distance', np.nan),
            }
            
            results.append(row)
            
        except Exception as e:
            print(f"Error loading {split_file}: {e}")
            continue
    
    df = pd.DataFrame(results)
    print(f"\nLoaded metrics from {len(df)} splits")
    print(f"Cities: {sorted(df['city'].unique())}")
    print(f"Resolutions: {sorted(df['resolution'].unique())}")
    print(f"Strategies: {sorted(df['strategy'].unique())}")
    print(f"N values: {sorted(df['n_samples'].unique())}")
    
    return df


def compute_summary_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Compute mean and std across seeds for each configuration
    
    Args:
        df: DataFrame with spatial metrics
        metric: Metric name to compute stats for
    
    Returns:
        DataFrame with mean and std
    """
    stats = []
    
    for city in df['city'].unique():
        for res in df['resolution'].unique():
            for strategy in df['strategy'].unique():
                for n in df['n_samples'].unique():
                    mask = (
                        (df['city'] == city) &
                        (df['resolution'] == res) &
                        (df['strategy'] == strategy) &
                        (df['n_samples'] == n)
                    )
                    
                    values = df[mask][metric].dropna().values
                    
                    if len(values) > 0:
                        stats.append({
                            'city': city,
                            'resolution': res,
                            'strategy': strategy,
                            'n_samples': n,
                            'mean': np.mean(values),
                            'std': np.std(values, ddof=1) if len(values) > 1 else 0,
                            'min': np.min(values),
                            'max': np.max(values),
                            'n_seeds': len(values)
                        })
    
    return pd.DataFrame(stats)


def plot_metric_vs_n(df: pd.DataFrame, metric: str, metric_label: str, 
                     output_path: str):
    """
    Plot a single spatial metric vs N for all city-resolution combinations
    
    Args:
        df: DataFrame with spatial metrics
        metric: Metric column name
        metric_label: Label for y-axis
        output_path: Path to save figure
    """
    cities = sorted(df['city'].unique())
    resolutions = sorted(df['resolution'].unique())
    
    fig, axes = plt.subplots(len(resolutions), len(cities), 
                            figsize=(5*len(cities), 4*len(resolutions)))
    
    if len(resolutions) == 1 and len(cities) == 1:
        axes = np.array([[axes]])
    elif len(resolutions) == 1:
        axes = axes.reshape(1, -1)
    elif len(cities) == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'Spatial Metric: {metric_label}', fontsize=14, fontweight='bold')
    
    # Compute summary statistics
    stats_df = compute_summary_stats(df, metric)
    
    for i, res in enumerate(resolutions):
        for j, city in enumerate(cities):
            ax = axes[i, j]
            
            # Plot each strategy
            for strategy in STRATEGIES:
                mask = (
                    (stats_df['city'] == city) &
                    (stats_df['resolution'] == res) &
                    (stats_df['strategy'] == strategy)
                )
                
                data = stats_df[mask].sort_values('n_samples')
                
                if len(data) > 0:
                    ax.plot(data['n_samples'], data['mean'],
                           marker='o', linewidth=2, markersize=6,
                           color=STRATEGY_COLORS[strategy],
                           label=STRATEGY_LABELS[strategy])
                    
                    # Add error bars (std)
                    ax.fill_between(data['n_samples'],
                                   data['mean'] - data['std'],
                                   data['mean'] + data['std'],
                                   color=STRATEGY_COLORS[strategy],
                                   alpha=0.2)
            
            # Format
            ax.set_xlabel('N (Training Samples)', fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_title(f'{city.title()} - {res}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_metrics_summary(df: pd.DataFrame, output_path: str):
    """
    Create a summary plot with all key spatial metrics
    
    Args:
        df: DataFrame with spatial metrics
        output_path: Path to save figure
    """
    metrics = [
        ('mean_pairwise_distance_km', 'Mean Pairwise Distance (km)'),
        ('mean_min_distance_km', 'Mean Nearest Neighbor (km)'),
        ('convex_hull_area_km2', 'Convex Hull Area (km²)'),
        ('unique_tiles', 'Unique NAIP Tiles')
    ]
    
    cities = sorted(df['city'].unique())
    resolutions = sorted(df['resolution'].unique())
    
    fig, axes = plt.subplots(len(metrics), len(cities)*len(resolutions),
                            figsize=(18, 12))
    
    fig.suptitle('Spatial Metrics Summary: All Strategies', 
                fontsize=14, fontweight='bold')
    
    for metric_idx, (metric, metric_label) in enumerate(metrics):
        stats_df = compute_summary_stats(df, metric)
        
        col_idx = 0
        for res in resolutions:
            for city in cities:
                ax = axes[metric_idx, col_idx]
                
                # Plot each strategy
                for strategy in STRATEGIES:
                    mask = (
                        (stats_df['city'] == city) &
                        (stats_df['resolution'] == res) &
                        (stats_df['strategy'] == strategy)
                    )
                    
                    data = stats_df[mask].sort_values('n_samples')
                    
                    if len(data) > 0:
                        ax.plot(data['n_samples'], data['mean'],
                               marker='o', linewidth=2, markersize=5,
                               color=STRATEGY_COLORS[strategy],
                               label=STRATEGY_LABELS[strategy])
                        
                        ax.fill_between(data['n_samples'],
                                       data['mean'] - data['std'],
                                       data['mean'] + data['std'],
                                       color=STRATEGY_COLORS[strategy],
                                       alpha=0.2)
                
                # Format
                if metric_idx == len(metrics) - 1:
                    ax.set_xlabel('N', fontsize=9)
                if col_idx == 0:
                    ax.set_ylabel(metric_label, fontsize=9)
                
                if metric_idx == 0:
                    ax.set_title(f'{city.title()}-{res}', fontsize=10)
                
                ax.grid(True, alpha=0.3)
                
                if metric_idx == 0 and col_idx == len(cities)*len(resolutions)-1:
                    ax.legend(loc='best', fontsize=8)
                
                col_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_metric_summary(df: pd.DataFrame):
    """
    Print summary statistics to console
    
    Args:
        df: DataFrame with spatial metrics
    """
    print("\n" + "="*80)
    print("SPATIAL METRICS SUMMARY")
    print("="*80)
    
    metrics = [
        ('mean_pairwise_distance_km', 'Mean Pairwise Distance (km)'),
        ('unique_tiles', 'Unique Tiles')
    ]
    
    for metric, metric_label in metrics:
        print(f"\n{metric_label}:")
        print("-" * 80)
        
        stats_df = compute_summary_stats(df, metric)
        
        for strategy in STRATEGIES:
            strategy_data = stats_df[stats_df['strategy'] == strategy]
            
            if len(strategy_data) > 0:
                avg_across_all = strategy_data['mean'].mean()
                print(f"  {STRATEGY_LABELS[strategy]:12s}: "
                      f"Avg={avg_across_all:.2f}, "
                      f"Min={strategy_data['mean'].min():.2f}, "
                      f"Max={strategy_data['mean'].max():.2f}")
    
    # Strategy comparison at specific N values
    print("\n" + "="*80)
    print("STRATEGY COMPARISON AT KEY N VALUES")
    print("="*80)
    
    for n in [25, 100, 200, 350]:
        print(f"\nN={n}:")
        print("-" * 80)
        
        stats_df = compute_summary_stats(df, 'mean_pairwise_distance_km')
        n_data = stats_df[stats_df['n_samples'] == n]
        
        if len(n_data) > 0:
            for strategy in STRATEGIES:
                strategy_data = n_data[n_data['strategy'] == strategy]
                if len(strategy_data) > 0:
                    avg = strategy_data['mean'].mean()
                    print(f"  {STRATEGY_LABELS[strategy]:12s}: {avg:.2f} km")


def main():
    parser = argparse.ArgumentParser(description='Visualize spatial metrics from splits')
    parser.add_argument('--split_dir', type=str, required=True,
                       help='Directory containing split JSON files')
    parser.add_argument('--output_dir', type=str, default='./spatial_metrics_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("SPATIAL METRICS VISUALIZATION")
    print("="*80)
    
    # Load data
    print("\nLoading spatial metrics from splits...")
    df = load_spatial_metrics_from_splits(args.split_dir)
    
    if len(df) == 0:
        print("ERROR: No split files found!")
        return
    
    # Print summary
    print_metric_summary(df)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    print("\n1. Mean Pairwise Distance vs N...")
    plot_metric_vs_n(df, 'mean_pairwise_distance_km', 
                     'Mean Pairwise Distance (km)',
                     os.path.join(args.output_dir, 'mean_pairwise_distance.png'))
    
    print("\n2. Mean Minimum Distance vs N...")
    plot_metric_vs_n(df, 'mean_min_distance_km',
                     'Mean Nearest Neighbor Distance (km)',
                     os.path.join(args.output_dir, 'mean_min_distance.png'))
    
    print("\n3. Convex Hull Area vs N...")
    plot_metric_vs_n(df, 'convex_hull_area_km2',
                     'Convex Hull Area (km²)',
                     os.path.join(args.output_dir, 'convex_hull_area.png'))
    
    print("\n4. Unique Tiles vs N...")
    plot_metric_vs_n(df, 'unique_tiles',
                     'Number of Unique NAIP Tiles',
                     os.path.join(args.output_dir, 'unique_tiles.png'))
    
    print("\n5. All metrics summary...")
    plot_all_metrics_summary(df, 
                            os.path.join(args.output_dir, 'all_metrics_summary.png'))
    
    print("\n" + "="*80)
    print(f"All plots saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()