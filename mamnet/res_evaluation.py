"""
Resolution Transfer Evaluation with Bootstrap Significance Testing
Evaluates resolution transferability of shadow detection models

Usage:
    python res_evaluation.py --output_dir ./outputs --base_data_root /path/to/Final_data_test/ --city chicago
"""

import os
import glob
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import from existing codebase
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet import MAMNet
from data.dataset import get_dataloaders
from utils.postprocessing import filter_small_predictions

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# City mapping
CITIES = ['chicago', 'miami', 'phoenix']
RESOLUTIONS = ['midres', 'highres']

def extract_session_from_path(path):
    """Extract session number from filename like chicago_session11_midres_paired_004.png"""
    import re
    match = re.search(r'session(\d+)', path)
    return match.group(1) if match else None

def extract_pair_id_from_path(filename):
    """
    Extract full pair identifier from filename.
    Examples: 
        chicago_session11_highres_paired_004.png -> chicago_session11_004
        miami_session05_midres_paired_002.png -> miami_session05_002
    """
    import re
    # Match pattern: city_sessionXX_resolution_paired_NNN
    match = re.search(r'(chicago|miami|phoenix)_session(\d+)_(?:midres|highres)_paired_(\d+)', filename)
    if match:
        return f"{match.group(1)}_session{match.group(2)}_{match.group(3)}"
    return filename  # fallback to full filename if pattern doesn't match


def find_latest_checkpoint(output_dir, pattern):
    """Find the latest checkpoint folder matching the pattern"""
    folders = glob.glob(os.path.join(output_dir, pattern))
    if not folders:
        return None
    
    # Sort by timestamp in folder name (format: YYYYMMDD_HHMMSS)
    folders.sort(key=lambda x: x.split('_')[-2] + '_' + x.split('_')[-1])
    latest = folders[-1]
    
    checkpoint_path = os.path.join(latest, 'checkpoint_best.pth')
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def load_model(checkpoint_path, device):
    """Load MAMNet model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    # Initialize model
    model = MAMNet(num_classes=2, pretrained=False, use_aux=True).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def compute_per_image_metrics(pred, target):
    """
    Compute per-image metrics
    pred, target: [H, W] binary masks (0 or 1)
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Basic confusion matrix elements
    tp = np.sum((pred == 1) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    
    # Compute metrics
    metrics = {}
    
    # Accuracy (OA)
    metrics['OA'] = 100.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Precision
    metrics['Precision'] = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall
    metrics['Recall'] = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    if metrics['Precision'] + metrics['Recall'] > 0:
        metrics['F1'] = 2 * metrics['Precision'] * metrics['Recall'] / (metrics['Precision'] + metrics['Recall'])
    else:
        metrics['F1'] = 0.0
    
    # IoU for shadow class
    metrics['Shadow_IOU'] = 100.0 * tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    # IoU for non-shadow class
    nonshadow_iou = 100.0 * tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    
    # mIOU (mean of shadow and non-shadow IoU)
    metrics['mIOU'] = (metrics['Shadow_IOU'] + nonshadow_iou) / 2.0
    
    # BER (Balanced Error Rate)
    shadow_error = 100.0 * fn / (tp + fn) if (tp + fn) > 0 else 0.0
    non_shadow_error = 100.0 * fp / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['BER'] = (shadow_error + non_shadow_error) / 2.0
    
    # Shadow pixel percentage (for analysis)
    metrics['shadow_percentage'] = 100.0 * np.sum(target == 1) / len(target)
    
    return metrics

def get_paired_dataloader(base_data_root, city, resolution, batch_size, img_size):
    """
    Load paired test data for a city at specified resolution.
    Since all test files are paired, we just load the test set normally.
    
    Args:
        base_data_root: Base directory (e.g., /path/to/Final_data_test/)
        city: City name ('chicago', 'miami', 'phoenix')
        resolution: Resolution ('highres' or 'midres')
        batch_size: Batch size
        img_size: Image size
    
    Returns:
        DataLoader with paired test data
    """
    # Construct data_root for the specific city and resolution
    data_root = os.path.join(base_data_root, city, resolution)
    
    # Use existing get_dataloaders function in 'single' mode
    dataloaders = get_dataloaders(
        data_root=data_root,
        base_data_root=None,
        mode='single',
        cities=None,
        resolution=None,
        fold_id=None,
        batch_size=batch_size,
        num_workers=1,
        img_size=img_size
    )
    
    return dataloaders['test']


def evaluate_model_per_image(model, dataloader, device):
    """
    Evaluate model and return per-image metrics
    Returns: list of dicts, each containing metrics for one image
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions (main output)
            if isinstance(outputs, tuple):
                preds = outputs[0]
            else:
                preds = outputs
            
            # Filter small predictions
            preds = filter_small_predictions(preds, min_pixels=10)
            # Convert to binary predictions
            # preds = torch.sigmoid(preds)
            # preds_binary = (preds > 0.5).float()
            preds_binary = torch.argmax(preds, dim=1).float()
            
            # Process each image in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                # pred_np = preds_binary[i, 0].cpu().numpy()
                pred_np = preds_binary[i].cpu().numpy()  # [H, W]
                target_np = masks[i].cpu().numpy()
                
                # Compute metrics for this image
                img_metrics = compute_per_image_metrics(pred_np, target_np)
                
                # Add image info
                img_metrics['batch_idx'] = batch_idx
                img_metrics['image_idx'] = i
                img_metrics['prediction'] = pred_np
                img_metrics['target'] = target_np
                img_metrics['image'] = images[i].cpu().numpy()
                img_metrics['filename'] = batch['filename'][i] if 'filename' in batch else f'batch{batch_idx}_img{i}'
                img_metrics['session'] = extract_session_from_path(img_metrics['filename'])
                img_metrics['pair_id'] = extract_pair_id_from_path(img_metrics['filename'])
                
                results.append(img_metrics)
    
    return results

def compute_sri_per_pair(native_results, transfer_results):
    """
    Compute Scale Robustness Index per paired image
    SRI = F1(native_res) - F1(transfer_from_other_res)
    
    Returns: list of dicts with SRI values per pair
    """
    sri_results = []
    
    # Match pairs by pair_id
    for native_img, transfer_img in zip(native_results, transfer_results):
        # Ensure they're the same pair
        assert native_img['pair_id'] == transfer_img['pair_id'], \
            f"Pair mismatch: {native_img['pair_id']} vs {transfer_img['pair_id']}"
        
        sri = {
            'pair_id': native_img['pair_id'],
            'session': native_img['session'],
            'SRI_F1': native_img['F1'] - transfer_img['F1'],
            'SRI_Shadow_IOU': native_img['Shadow_IOU'] - transfer_img['Shadow_IOU'],
            'SRI_mIOU': native_img['mIOU'] - transfer_img['mIOU'],
            'native_F1': native_img['F1'],
            'transfer_F1': transfer_img['F1'],
        }
        sri_results.append(sri)
    
    return sri_results

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=95):
    """
    Compute bootstrap confidence interval
    data: array of metric values
    """
    bootstrap_samples = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_samples.append(np.mean(sample))
    
    bootstrap_samples = np.array(bootstrap_samples)
    
    # Compute confidence interval
    alpha = (100 - confidence) / 2
    ci_lower = np.percentile(bootstrap_samples, alpha)
    ci_upper = np.percentile(bootstrap_samples, 100 - alpha)
    
    return {
        'mean': np.mean(data),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'bootstrap_samples': bootstrap_samples  # ADD THIS LINE
    }


def paired_permutation_test(data1, data2, n_permutations=10000):
    """
    Paired permutation test for comparing two models on same test set
    data1, data2: arrays of per-image metrics
    """
    # Observed difference
    observed_diff = np.mean(data1) - np.mean(data2)
    
    # Permutation test
    perm_diffs = []
    n = len(data1)
    
    for _ in range(n_permutations):
        # Randomly swap values between the two arrays
        swap = np.random.randint(0, 2, size=n).astype(bool)
        perm1 = np.where(swap, data2, data1)
        perm2 = np.where(swap, data1, data2)
        perm_diffs.append(np.mean(perm1) - np.mean(perm2))
    
    perm_diffs = np.array(perm_diffs)
    
    # Two-tailed p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def analyze_city_pair(within_results, transfer_results, metric_names, n_bootstrap=1000):
    """
    Analyze a single city pair (within vs transfer)
    Returns statistical analysis for all metrics
    """
    analysis = {}
    
    for metric in metric_names:
        within_values = np.array([r[metric] for r in within_results])
        transfer_values = np.array([r[metric] for r in transfer_results])
        
        # Bootstrap CIs
        within_bootstrap = bootstrap_confidence_interval(within_values, n_bootstrap)
        transfer_bootstrap = bootstrap_confidence_interval(transfer_values, n_bootstrap)
        
        # Paired permutation test
        perm_test = paired_permutation_test(within_values, transfer_values)
        
        # Geo-Gap
        geo_gap = within_bootstrap['mean'] - transfer_bootstrap['mean']
        
        analysis[metric] = {
            'within': within_bootstrap,
            'transfer': transfer_bootstrap,
            'geo_gap': geo_gap,
            'permutation_test': perm_test,
            'within_std': np.std(within_values),
            'transfer_std': np.std(transfer_values)
        }
    
    return analysis


def plot_bootstrap_comparison(analysis_dict, output_path, resolution):
    """
    Create forest plot showing bootstrap CIs for all city pairs
    """
    metric_names = ['F1', 'Shadow_IOU', 'mIOU', 'BER']
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        y_pos = []
        y_labels = []
        
        for i, (pair_name, analysis) in enumerate(analysis_dict.items()):
            y_base = i * 2
            
            # Within-city
            within = analysis[metric]['within']
            ax.plot([within['ci_lower'], within['ci_upper']], [y_base, y_base], 
                   'b-', linewidth=2, label='Within-city' if i == 0 else '')
            ax.plot(within['mean'], y_base, 'bo', markersize=8)
            
            # Transfer
            transfer = analysis[metric]['transfer']
            ax.plot([transfer['ci_lower'], transfer['ci_upper']], [y_base+0.5, y_base+0.5], 
                   'r-', linewidth=2, label='Transfer' if i == 0 else '')
            ax.plot(transfer['mean'], y_base+0.5, 'ro', markersize=8)
            
            y_pos.append(y_base + 0.25)
            y_labels.append(pair_name.replace('_', '\n'))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(f'{metric} (%)')
        ax.set_title(f'{metric} - Bootstrap 95% CI Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved bootstrap comparison: {output_path}")


def plot_scatter_comparison(analysis_dict, results_dict, output_path, resolution):
    """
    Create scatter plots showing per-image comparison
    """
    metric_names = ['F1', 'Shadow_IOU', 'mIOU', 'BER']
    n_metrics = len(metric_names)
    n_pairs = len(analysis_dict)
    
    fig, axes = plt.subplots(n_pairs, n_metrics, figsize=(5*n_metrics, 5*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, (pair_name, (within_results, transfer_results)) in enumerate(results_dict.items()):
        for col_idx, metric in enumerate(metric_names):
            ax = axes[row_idx, col_idx]
            
            within_values = np.array([r[metric] for r in within_results])
            transfer_values = np.array([r[metric] for r in transfer_results])
            shadow_pct = np.array([r['shadow_percentage'] for r in within_results])
            
            # Scatter plot
            scatter = ax.scatter(within_values, transfer_values, c=shadow_pct, 
                               cmap='viridis', alpha=0.6, s=30)
            
            # Diagonal line (perfect transfer)
            max_val = max(within_values.max(), transfer_values.max())
            min_val = min(within_values.min(), transfer_values.min())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect transfer')
            
            # Add colorbar for first column
            if col_idx == n_metrics - 1:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Shadow %')
            
            ax.set_xlabel(f'Within-city {metric}')
            ax.set_ylabel(f'Transfer {metric}')
            ax.set_title(f'{pair_name}\n{metric}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved scatter comparison: {output_path}")


def plot_box_comparison(analysis_dict, results_dict, output_path, resolution):
    """
    Create box plots showing distribution comparison
    """
    metric_names = ['F1', 'Shadow_IOU', 'mIOU', 'BER']
    n_metrics = len(metric_names)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        data_to_plot = []
        labels = []
        
        for pair_name, (within_results, transfer_results) in results_dict.items():
            within_values = [r[metric] for r in within_results]
            transfer_values = [r[metric] for r in transfer_results]
            
            data_to_plot.extend([within_values, transfer_values])
            labels.extend([f'{pair_name}\nWithin', f'{pair_name}\nTransfer'])
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        for i, patch in enumerate(bp['boxes']):
            if i % 2 == 0:
                patch.set_facecolor('lightblue')
            else:
                patch.set_facecolor('lightcoral')
        
        ax.set_ylabel(f'{metric} (%)')
        ax.set_title(f'{metric} Distribution')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved box comparison: {output_path}")


def plot_geo_gap_heatmap(summary_df, output_path, resolution):
    """
    Create heatmap showing Geo-Gap for all city pairs
    """
    metric_names = ['F1', 'Shadow_IOU', 'mIOU', 'BER']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        # Create matrix
        matrix = np.zeros((3, 3))
        
        for _, row in summary_df.iterrows():
            source = row['source_cities']
            target = row['target_city']
            geo_gap = row[f'{metric}_geo_gap']
            
            # Map to indices
            target_idx = CITIES.index(target)
            
            # For LOCO, source is the two cities NOT in target
            if target == 'phoenix':
                source_cities = ['chicago', 'miami']
            elif target == 'miami':
                source_cities = ['chicago', 'phoenix']
            else:  # chicago
                source_cities = ['miami', 'phoenix']
            
            # Average across source cities for simplicity
            for src_city in source_cities:
                src_idx = CITIES.index(src_city)
                matrix[src_idx, target_idx] = geo_gap
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Geo-Gap (%)', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(3))
        ax.set_yticks(np.arange(3))
        ax.set_xticklabels(CITIES)
        ax.set_yticklabels(CITIES)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title(f'{metric} Geo-Gap')
        ax.set_xlabel('Target City (Test)')
        ax.set_ylabel('Source City (Train)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved geo-gap heatmap: {output_path}")


def find_best_worst_images(within_results, transfer_results, metric='F1', 
                           n_best=10, n_worst=10, threshold_best=80, threshold_worst_drop=20):
    """
    Find best and worst transfer cases
    """
    best_images = []
    worst_images = []
    
    for within, transfer in zip(within_results, transfer_results):
        within_metric = within[metric]
        transfer_metric = transfer[metric]
        drop = within_metric - transfer_metric
        
        # Check for non-trivial images (has shadows)
        if within['shadow_percentage'] < 1.0:  # Skip images with <1% shadows
            continue
        
        # Best: both models do well
        if within_metric >= threshold_best and transfer_metric >= threshold_best:
            best_images.append({
                'within': within,
                'transfer': transfer,
                'within_metric': within_metric,
                'transfer_metric': transfer_metric,
                'drop': drop
            })
        
        # Worst: within does well, transfer fails badly
        if within_metric >= threshold_best and drop >= threshold_worst_drop:
            worst_images.append({
                'within': within,
                'transfer': transfer,
                'within_metric': within_metric,
                'transfer_metric': transfer_metric,
                'drop': drop
            })
    
    # Sort and select top N
    best_images = sorted(best_images, key=lambda x: x['transfer_metric'], reverse=True)[:n_best]
    worst_images = sorted(worst_images, key=lambda x: x['drop'], reverse=True)[:n_worst]
    
    return best_images, worst_images


def visualize_qualitative_results(best_images, worst_images, output_dir, pair_name, metric='F1'):
    """
    Create qualitative visualization grids for best and worst cases
    """
    # Best cases
    if best_images:
        n_images = len(best_images)
        fig, axes = plt.subplots(n_images, 5, figsize=(20, 4*n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_data in enumerate(best_images):
            within = img_data['within']
            transfer = img_data['transfer']
            
            # Input image
            img = within['image'].transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())  # Normalize
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title('Input RGB')
            axes[idx, 0].axis('off')
            
            # Ground truth
            axes[idx, 1].imshow(within['target'], cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Within-city prediction
            axes[idx, 2].imshow(within['prediction'], cmap='gray')
            axes[idx, 2].set_title(f'Within\n{metric}={within[metric]:.1f}%')
            axes[idx, 2].axis('off')
            
            # Transfer prediction
            axes[idx, 3].imshow(transfer['prediction'], cmap='gray')
            axes[idx, 3].set_title(f'Transfer\n{metric}={transfer[metric]:.1f}%')
            axes[idx, 3].axis('off')
            
            # Difference map
            diff = np.abs(within['prediction'] - transfer['prediction'])
            axes[idx, 4].imshow(diff, cmap='Reds')
            axes[idx, 4].set_title(f'Difference\nDrop={img_data["drop"]:.1f}%')
            axes[idx, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{pair_name}_best_cases.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved best cases: {pair_name}_best_cases.png")
    
    # Worst cases
    if worst_images:
        n_images = len(worst_images)
        fig, axes = plt.subplots(n_images, 5, figsize=(20, 4*n_images))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_data in enumerate(worst_images):
            within = img_data['within']
            transfer = img_data['transfer']
            
            # Input image
            img = within['image'].transpose(1, 2, 0)
            img = (img - img.min()) / (img.max() - img.min())
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title('Input RGB')
            axes[idx, 0].axis('off')
            
            # Ground truth
            axes[idx, 1].imshow(within['target'], cmap='gray')
            axes[idx, 1].set_title('Ground Truth')
            axes[idx, 1].axis('off')
            
            # Within-city prediction
            axes[idx, 2].imshow(within['prediction'], cmap='gray')
            axes[idx, 2].set_title(f'Within\n{metric}={within[metric]:.1f}%')
            axes[idx, 2].axis('off')
            
            # Transfer prediction
            axes[idx, 3].imshow(transfer['prediction'], cmap='gray')
            axes[idx, 3].set_title(f'Transfer\n{metric}={transfer[metric]:.1f}%')
            axes[idx, 3].axis('off')
            
            # Difference map
            diff = np.abs(within['prediction'] - transfer['prediction'])
            axes[idx, 4].imshow(diff, cmap='Reds')
            axes[idx, 4].set_title(f'Difference\nDrop={img_data["drop"]:.1f}%')
            axes[idx, 4].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{pair_name}_worst_cases.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved worst cases: {pair_name}_worst_cases.png")


def create_summary_table(analysis_dict, output_path):
    """
    Create comprehensive summary table
    """
    rows = []
    
    for pair_name, analysis in analysis_dict.items():
        row = {'city_pair': pair_name}
        
        for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER']:
            m = analysis[metric]
            
            # Within-city
            row[f'{metric}_within_mean'] = m['within']['mean']
            row[f'{metric}_within_ci_lower'] = m['within']['ci_lower']
            row[f'{metric}_within_ci_upper'] = m['within']['ci_upper']
            row[f'{metric}_within_std'] = m['within_std']
            
            # Transfer
            row[f'{metric}_transfer_mean'] = m['transfer']['mean']
            row[f'{metric}_transfer_ci_lower'] = m['transfer']['ci_lower']
            row[f'{metric}_transfer_ci_upper'] = m['transfer']['ci_upper']
            row[f'{metric}_transfer_std'] = m['transfer_std']
            
            # Geo-Gap
            row[f'{metric}_geo_gap'] = m['geo_gap']
            row[f'{metric}_p_value'] = m['permutation_test']['p_value']
            row[f'{metric}_significant'] = m['permutation_test']['significant']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"Saved summary table: {output_path}")
    
    return df


def create_paper_ready_table(summary_df, output_path):
    """
    Create LaTeX-ready table for paper
    """
    with open(output_path, 'w') as f:
        f.write("% LaTeX Table for Paper\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Cross-city Transfer Performance and Geo-Gap Analysis}\n")
        f.write("\\label{tab:loco_results}\n")
        f.write("\\begin{tabular}{l|cccc|cccc|cccc}\n")
        f.write("\\hline\n")
        f.write("& \\multicolumn{4}{c|}{Within-city (Upper Bound)} & \\multicolumn{4}{c|}{Transfer} & \\multicolumn{4}{c}{Geo-Gap} \\\\\n")
        f.write("City Pair & F1 & ShadIoU & mIoU & BER & F1 & ShadIoU & mIoU & BER & F1 & ShadIoU & mIoU & BER \\\\\n")
        f.write("\\hline\n")
        
        for _, row in summary_df.iterrows():
            pair = row['city_pair'].replace('_', ' ')
            
            line = f"{pair} & "
            
            # Within-city
            for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER']:
                mean = row[f'{metric}_within_mean']
                ci_l = row[f'{metric}_within_ci_lower']
                ci_u = row[f'{metric}_within_ci_upper']
                line += f"{mean:.1f} & "
            
            # Transfer
            for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER']:
                mean = row[f'{metric}_transfer_mean']
                ci_l = row[f'{metric}_transfer_ci_lower']
                ci_u = row[f'{metric}_transfer_ci_upper']
                line += f"{mean:.1f} & "
            
            # Geo-Gap
            metrics = ['F1', 'Shadow_IOU', 'mIOU', 'BER']
            for i, metric in enumerate(metrics):
                gap = row[f'{metric}_geo_gap']
                sig = row[f'{metric}_significant']
                if i < len(metrics) - 1:
                    line += f"{gap:.1f}{'*' if sig else ''} & "
                else:
                    line += f"{gap:.1f}{'*' if sig else ''}"
            
            line += " \\\\\n"
            f.write(line)
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table: {output_path}")


def main():
    """Main evaluation function for cross-resolution transferability"""
    parser = argparse.ArgumentParser(
        description='Cross-Resolution Transfer Evaluation with SRI (Scale Robustness Index)'
    )
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory containing trained model checkpoints')
    parser.add_argument('--base_data_root', type=str, required=True,
                       help='Base directory for data (e.g., /path/to/Final_data_test/)')
    parser.add_argument('--source_resolution', type=str, required=True,
                       choices=['midres', 'highres'],
                       help='Source resolution (training resolution of transfer model)')
    parser.add_argument('--target_resolution', type=str, required=True,
                       choices=['midres', 'highres'],
                       help='Target resolution (testing resolution - where paired data is loaded from)')
    parser.add_argument('--city', type=str, default='all',
                       choices=['chicago', 'miami', 'phoenix', 'all'],
                       help='City to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap samples')
    parser.add_argument('--img_size', type=int, default=384,
                       help='Image size')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Validate resolution combination
    if args.source_resolution == args.target_resolution:
        print("WARNING: Source and target resolutions are the same!")
        print("This will compare same-resolution models. Consider using different resolutions for cross-resolution analysis.")
    
    # Create output directory for this evaluation
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Determine if evaluating single city or all cities
    if args.city != 'all':
        # Single city - include in directory name
        eval_dir = os.path.join(
            args.output_dir, 
            f'res_evaluation_{args.source_resolution}to{args.target_resolution}_{args.city}_{timestamp}'
        )
    else:
        # All cities - create main directory, cities will have subdirectories
        eval_dir = os.path.join(
            args.output_dir, 
            f'res_evaluation_{args.source_resolution}to{args.target_resolution}_all_{timestamp}'
        )
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Cross-Resolution Transfer Evaluation")
    print(f"Source (Training): {args.source_resolution}")
    print(f"Target (Testing):  {args.target_resolution}")
    print(f"Output: {eval_dir}")
    print(f"{'='*60}\n")
    
    # Determine cities to evaluate
    cities_to_eval = [args.city] if args.city != 'all' else CITIES
    
    # Store all results
    all_analysis = {}
    all_results = {}
    all_sri_results = {}
    summary_rows = []

    # Evaluate each city
    for city in cities_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {city}")
        print(f"Transfer: {args.source_resolution} → {args.target_resolution}")
        print(f"{'='*60}")

        if len(cities_to_eval) > 1:
            city_eval_dir = os.path.join(eval_dir, city)
            os.makedirs(city_eval_dir, exist_ok=True)
        else:
            city_eval_dir = eval_dir
        
        # Find checkpoints
        # 1. Native resolution model (upper bound) - trained on target resolution
        native_pattern = f'mamnet_{city}_{args.target_resolution}_*'
        native_checkpoint = find_latest_checkpoint(args.output_dir, native_pattern)
        
        if native_checkpoint is None:
            print(f"WARNING: Native checkpoint not found for {city} {args.target_resolution}")
            print(f"Looking for pattern: {native_pattern}")
            print(f"Skipping {city}...\n")
            continue
        
        # 2. Transfer model - trained on source resolution
        transfer_pattern = f'mamnet_{city}_{args.source_resolution}_*'
        transfer_checkpoint = find_latest_checkpoint(args.output_dir, transfer_pattern)
        
        if transfer_checkpoint is None:
            print(f"WARNING: Transfer checkpoint not found for {city} {args.source_resolution}")
            print(f"Looking for pattern: {transfer_pattern}")
            print(f"Skipping {city}...\n")
            continue
        
        print(f"Native model ({args.target_resolution}):  {native_checkpoint}")
        print(f"Transfer model ({args.source_resolution}): {transfer_checkpoint}")
        
        # Load models
        native_model = load_model(native_checkpoint, device)
        transfer_model = load_model(transfer_checkpoint, device)
        
        # Get paired test dataloader for target resolution
        print(f"\nLoading paired test data for {city} at {args.target_resolution}...")
        test_loader = get_paired_dataloader(
            args.base_data_root, 
            city, 
            args.target_resolution,
            args.batch_size,
            args.img_size
        )
        
        if len(test_loader.dataset) == 0:
            print(f"WARNING: No paired test data found for {city} at {args.target_resolution}")
            print(f"Skipping {city}...\n")
            continue
            
        print(f"Paired test samples: {len(test_loader.dataset)}")
        
        # Evaluate both models on same paired test set
        print(f"\nEvaluating native model ({args.target_resolution})...")
        native_results = evaluate_model_per_image(native_model, test_loader, device)
        
        print(f"\nEvaluating transfer model ({args.source_resolution} → {args.target_resolution})...")
        transfer_results = evaluate_model_per_image(transfer_model, test_loader, device)
        
        # Verify same number of images
        assert len(native_results) == len(transfer_results), \
            f"Result count mismatch: {len(native_results)} vs {len(transfer_results)}"
        
        # Compute SRI per pair
        print(f"\nComputing SRI (Scale Robustness Index)...")
        sri_results = compute_sri_per_pair(native_results, transfer_results)
        
        # Statistical analysis
        print(f"\nComputing bootstrap statistics...")
        metric_names = ['F1', 'Shadow_IOU', 'mIOU', 'BER', 'OA', 'Precision', 'Recall']
        analysis = analyze_city_pair(native_results, transfer_results, metric_names, args.n_bootstrap)
        
        # Add SRI-specific statistics
        sri_f1_values = [sri['SRI_F1'] for sri in sri_results]
        sri_shadow_iou_values = [sri['SRI_Shadow_IOU'] for sri in sri_results]
        sri_miou_values = [sri['SRI_mIOU'] for sri in sri_results]
        
        analysis['SRI_F1'] = bootstrap_confidence_interval(sri_f1_values, args.n_bootstrap)
        analysis['SRI_Shadow_IOU'] = bootstrap_confidence_interval(sri_shadow_iou_values, args.n_bootstrap)
        analysis['SRI_mIOU'] = bootstrap_confidence_interval(sri_miou_values, args.n_bootstrap)
        
        # Rename 'within' and 'transfer' to 'native' and 'transfer' for clarity
        # for metric in metric_names:
        #     if metric in analysis:
        #         analysis[metric]['native'] = analysis[metric].pop('within', analysis[metric].get('native', {}))
                # 'transfer' key already exists, no need to rename
        
        # Store results
        pair_name = f"{city}_{args.source_resolution}→{args.target_resolution}"
        all_analysis[pair_name] = analysis
        all_results[pair_name] = (native_results, transfer_results)
        all_sri_results[pair_name] = sri_results
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Results for {pair_name}:")
        print(f"{'='*60}")
        for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER']:
            if metric in analysis:
                m = analysis[metric]
                print(f"\n{metric}:")
                print(f"  Native ({args.target_resolution}):  {m['within']['mean']:.2f}% [{m['within']['ci_lower']:.2f}, {m['within']['ci_upper']:.2f}]")
                print(f"  Transfer ({args.source_resolution}→{args.target_resolution}): {m['transfer']['mean']:.2f}% [{m['transfer']['ci_lower']:.2f}, {m['transfer']['ci_upper']:.2f}]")
                print(f"  Res-Gap:     {m['geo_gap']:.2f}%")
                print(f"  P-value:     {m['permutation_test']['p_value']:.4f} {'***' if m['permutation_test']['significant'] else ''}")
        
        # Print SRI statistics
        print(f"\nScale Robustness Index (SRI):")
        print(f"  SRI_F1:         {analysis['SRI_F1']['mean']:.2f}% [{analysis['SRI_F1']['ci_lower']:.2f}, {analysis['SRI_F1']['ci_upper']:.2f}]")
        print(f"  SRI_Shadow_IOU: {analysis['SRI_Shadow_IOU']['mean']:.2f}% [{analysis['SRI_Shadow_IOU']['ci_lower']:.2f}, {analysis['SRI_Shadow_IOU']['ci_upper']:.2f}]")
        print(f"  SRI_mIOU:       {analysis['SRI_mIOU']['mean']:.2f}% [{analysis['SRI_mIOU']['ci_lower']:.2f}, {analysis['SRI_mIOU']['ci_upper']:.2f}]")
        print(f"  (Closer to 0 is better - indicates scale robustness)")
        
        # Find best/worst transfer cases
        print(f"\nFinding best and worst transfer cases...")
        best_images, worst_images = find_best_worst_images(
            native_results, transfer_results, 
            metric='F1', n_best=10, n_worst=10
        )
        
        print(f"Found {len(best_images)} best cases and {len(worst_images)} worst cases")
        
        # Visualize qualitative results
        visualize_qualitative_results(best_images, worst_images, city_eval_dir, pair_name)
        
        # Add to summary
        summary_rows.append({
            'city': city,
            'source_resolution': args.source_resolution,
            'target_resolution': args.target_resolution,
            'n_paired_samples': len(test_loader.dataset),
            **{f'{metric}_native': analysis[metric]['within']['mean'] 
               for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER'] if metric in analysis},
            **{f'{metric}_transfer': analysis[metric]['transfer']['mean'] 
               for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER'] if metric in analysis},
            **{f'{metric}_res_gap': analysis[metric]['geo_gap'] 
               for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER'] if metric in analysis},
            **{f'SRI_{metric}': analysis[f'SRI_{metric}']['mean'] 
               for metric in ['F1', 'Shadow_IOU', 'mIOU']}
        })
    
    # Check if we have any results
    if len(all_analysis) == 0:
        print("\n" + "="*60)
        print("ERROR: No cities were successfully evaluated!")
        print("Please check that model checkpoints exist for the specified cities and resolutions.")
        print("="*60)
        return
    
    # Create comprehensive visualizations
    print(f"\n{'='*60}")
    print("Creating comprehensive visualizations...")
    print(f"{'='*60}\n")
    
    # Bootstrap comparison
    plot_bootstrap_comparison(all_analysis, 
                             os.path.join(city_eval_dir, f'resolution_comparison_{args.source_resolution}to{args.target_resolution}.png'),
                             f"{args.source_resolution}→{args.target_resolution}")
    
    # Scatter plots
    plot_scatter_comparison(all_analysis, all_results,
                           os.path.join(city_eval_dir, f'scatter_comparison_{args.source_resolution}to{args.target_resolution}.png'),
                           f"{args.source_resolution}→{args.target_resolution}")
    
    # Box plots
    plot_box_comparison(all_analysis, all_results,
                       os.path.join(city_eval_dir, f'box_comparison_{args.source_resolution}to{args.target_resolution}.png'),
                       f"{args.source_resolution}→{args.target_resolution}")
    
    # Summary table
    summary_df = create_summary_table(all_analysis, 
                                     os.path.join(city_eval_dir, f'summary_table_{args.source_resolution}to{args.target_resolution}.csv'))
    
    # LaTeX table
    create_paper_ready_table(summary_df,
                            os.path.join(city_eval_dir, f'paper_table_{args.source_resolution}to{args.target_resolution}.tex'))
    
    # Save SRI results per city
    sri_summary = {}
    for pair_name, sri_results in all_sri_results.items():
        sri_df = pd.DataFrame(sri_results)
        sri_csv_path = os.path.join(city_eval_dir, f'sri_per_pair_{pair_name}.csv')
        sri_df.to_csv(sri_csv_path, index=False)
        print(f"Saved SRI per-pair results: {sri_csv_path}")
        
        # Summary stats
        sri_summary[pair_name] = {
            'mean_SRI_F1': sri_df['SRI_F1'].mean(),
            'std_SRI_F1': sri_df['SRI_F1'].std(),
            'median_SRI_F1': sri_df['SRI_F1'].median(),
            'mean_SRI_Shadow_IOU': sri_df['SRI_Shadow_IOU'].mean(),
            'mean_SRI_mIOU': sri_df['SRI_mIOU'].mean(),
        }
    
    # Save SRI summary
    sri_summary_df = pd.DataFrame(sri_summary).T
    sri_summary_path = os.path.join(city_eval_dir, f'sri_summary_{args.source_resolution}to{args.target_resolution}.csv')
    sri_summary_df.to_csv(sri_summary_path)
    print(f"Saved SRI summary: {sri_summary_path}")
    
    # Save bootstrap samples for aggregate plotting
    bootstrap_data = {}
    for pair_name, analysis in all_analysis.items():
        bootstrap_data[pair_name] = {}
        for metric in ['F1', 'Shadow_IOU', 'mIOU', 'BER']:
            if metric in analysis:
                # Save Res-Gap bootstrap samples (native - transfer)
                native_samples = analysis[metric]['within']['bootstrap_samples']
                transfer_samples = analysis[metric]['transfer']['bootstrap_samples']
                res_gap_samples = native_samples - transfer_samples
                bootstrap_data[pair_name][metric] = {
                    'native': native_samples,
                    'transfer': transfer_samples,
                    'res_gap': res_gap_samples
                }
        # Also save SRI bootstrap samples
        for metric in ['F1', 'Shadow_IOU', 'mIOU']:
            sri_key = f'SRI_{metric}'
            if sri_key in analysis:
                bootstrap_data[pair_name][sri_key] = {
                    'sri': analysis[sri_key]['bootstrap_samples']
                }

    # Save as pickle file
    import pickle
    bootstrap_path = os.path.join(city_eval_dir, f'bootstrap_samples_{args.source_resolution}to{args.target_resolution}.pkl')
    with open(bootstrap_path, 'wb') as f:
        pickle.dump(bootstrap_data, f)
    print(f"Saved bootstrap samples: {bootstrap_path}")
    
    # Save configuration
    config = {
        'source_resolution': args.source_resolution,
        'target_resolution': args.target_resolution,
        'cities_evaluated': cities_to_eval,
        'n_bootstrap': args.n_bootstrap,
        'timestamp': timestamp,
        'device': str(device),
        'base_data_root': args.base_data_root
    }
    
    with open(os.path.join(city_eval_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*60}")
    print("Cross-Resolution Evaluation Complete!")
    print(f"Transfer: {args.source_resolution} → {args.target_resolution}")
    print(f"Cities evaluated: {', '.join(cities_to_eval)}")
    print(f"All results saved to: {city_eval_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()