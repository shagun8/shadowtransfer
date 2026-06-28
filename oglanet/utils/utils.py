"""
Utility functions for shadow attribute extraction
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.stats import entropy
import pandas as pd


def _expandvars_tree(obj):
    """Recursively apply os.path.expandvars to all string values.

    config.yaml path strings use ${PROJECT_ROOT} (and other env vars); YAML has no
    env-var expansion of its own, so we resolve them here after load.
    """
    if isinstance(obj, dict):
        return {k: _expandvars_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expandvars_tree(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file (resolves ${PROJECT_ROOT} etc. via expandvars)."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return _expandvars_tree(config)


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata JSON and convert to DataFrame"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    df = pd.DataFrame(metadata)
    return df


def get_paired_filename(filename: str, target_resolution: str) -> str:
    """
    Convert filename from one resolution to another
    Example: chicago_session01_highres_paired_002.png -> chicago_session01_midres_paired_002.png
    """
    if 'highres' in filename and target_resolution == 'midres':
        return filename.replace('highres', 'midres')
    elif 'midres' in filename and target_resolution == 'highres':
        return filename.replace('midres', 'highres')
    else:
        return filename


def find_test_images(base_data_root: str, city: str, resolution: str,
                    metadata_df: pd.DataFrame, max_images: Optional[int] = None) -> List[Dict]:
    """
    Find all test images for a city/resolution combination
    
    Returns:
        List of dicts with image info: {
            'image_path': path to RGB image,
            'mask_path': path to GT mask,
            'filename': original filename,
            'city': city name,
            'resolution': resolution,
            'pair_id': pair ID if paired, else None,
            'metadata': full metadata row
        }
    """
    test_dir = Path(base_data_root) / city / resolution / "test"
    image_dir = test_dir / "images"
    mask_dir = test_dir / "masks"
    
    if not image_dir.exists():
        print(f"WARNING: Image directory does not exist: {image_dir}")
        return []
    
    image_files = sorted(list(image_dir.glob("*.png")))
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    results = []
    for img_path in image_files:
        filename = img_path.name
        mask_path = mask_dir / filename
        
        if not mask_path.exists():
            print(f"WARNING: Mask not found for {filename}")
            continue
        
        # Get metadata
        meta_row = metadata_df[metadata_df['original_filename'] == filename]
        if len(meta_row) == 0:
            print(f"WARNING: No metadata found for {filename}")
            meta_dict = {'pair_id': None}
        else:
            meta_dict = meta_row.iloc[0].to_dict()
        
        results.append({
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'filename': filename,
            'city': city,
            'resolution': resolution,
            'pair_id': meta_dict.get('pair_id', None),
            'metadata': meta_dict
        })
    
    return results

def find_prediction_mask(pred_dir: Path, filename: str) -> Optional[str]:
    """
    Find prediction mask for a given filename
    
    Args:
        pred_dir: Directory containing predictions (e.g., predictions/within/chicago/highres/)
        filename: Original filename
    
    Returns:
        Path to prediction mask if found, else None
    """
    pred_path = pred_dir / filename
    if pred_path.exists():
        return str(pred_path)
    return None

def load_image_and_mask(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load RGB image and binary mask
    
    Returns:
        image: [H, W, 3] uint8 RGB
        mask: [H, W] binary (0 or 1)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 127).astype(np.uint8)  # Binarize
    
    return image, mask


def compute_connected_components(mask: np.ndarray, min_size: int = 10) -> Tuple[np.ndarray, int]:
    """
    Compute connected components and filter by size
    
    Args:
        mask: Binary mask [H, W]
        min_size: Minimum component size in pixels
    
    Returns:
        labeled: Labeled mask [H, W] where each component has unique ID
        num_components: Number of components after filtering
    """
    # Label connected components
    labeled, num = label(mask, connectivity=2, return_num=True)
    
    # Filter by size
    component_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background (0)
    valid_components = np.where(component_sizes >= min_size)[0] + 1  # +1 because bincount index
    
    # Create new labeled mask with only valid components
    labeled_filtered = np.zeros_like(labeled)
    for new_id, old_id in enumerate(valid_components, start=1):
        labeled_filtered[labeled == old_id] = new_id
    
    return labeled_filtered, len(valid_components)


def compute_metrics_per_image(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics between prediction and ground truth
    
    Args:
        pred_mask: Binary prediction [H, W]
        gt_mask: Binary ground truth [H, W]
    
    Returns:
        Dictionary of metrics
    """
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    tp = np.sum((pred_flat == 1) & (gt_flat == 1))
    tn = np.sum((pred_flat == 0) & (gt_flat == 0))
    fp = np.sum((pred_flat == 1) & (gt_flat == 0))
    fn = np.sum((pred_flat == 0) & (gt_flat == 1))
    
    metrics = {}
    
    # Precision, Recall, F1
    metrics['precision'] = 100.0 * tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0
    
    # IoU
    metrics['shadow_iou'] = 100.0 * tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    nonshadow_iou = 100.0 * tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    metrics['miou'] = (metrics['shadow_iou'] + nonshadow_iou) / 2.0
    
    # BER
    shadow_error = 100.0 * fn / (tp + fn) if (tp + fn) > 0 else 0.0
    non_shadow_error = 100.0 * fp / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['ber'] = (shadow_error + non_shadow_error) / 2.0
    
    # OA
    metrics['oa'] = 100.0 * (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Shadow percentage
    metrics['shadow_percentage_gt'] = 100.0 * np.sum(gt_flat == 1) / len(gt_flat)
    metrics['shadow_percentage_pred'] = 100.0 * np.sum(pred_flat == 1) / len(pred_flat)
    
    return metrics


def create_output_directories(config: dict) -> Dict[str, Path]:
    """Create all necessary output directories"""
    output_root = Path(config['paths']['output_root'])
    
    dirs = {
        'root': output_root,
        'predictions': output_root / 'predictions',
        'attributes': output_root / 'attributes',
        'analysis': output_root / 'analysis',
        'figures': output_root / 'figures',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


# Abbreviation mapping for plots
ATTRIBUTE_ABBREVIATIONS = {
    # Image-level
    'mean_brightness': 'bright_mean',
    'std_brightness': 'bright_std',
    'brightness_dynamic_range': 'bright_range',
    'local_contrast_mean': 'contrast_local',
    'edge_density': 'edge_dens',
    'texture_entropy': 'texture_ent',
    'high_frequency_energy_ratio': 'hf_energy',
    
    # Shadow properties
    'shadow_ratio': 'shad_ratio',
    'mean_shadow_size': 'shad_size_mean',
    'shadow_size_variance': 'shad_size_var',
    'mean_elongation': 'elongation',
    'mean_compactness': 'compact',
    'shadow_background_contrast': 'shad_bg_cont',
    
    # Continuity
    'num_shadow_instances': 'num_inst',
    'fragmentation_index': 'frag_idx',
    'largest_shadow_ratio': 'largest_ratio',
    'small_region_noise_ratio': 'noise_ratio',
    'boundary_smoothness': 'bound_smooth',
    'spatial_autocorrelation': 'spatial_auto',
    
    # Component-level
    'component_area': 'area',
    'component_elongation': 'elong',
    'component_compactness': 'compact',
    'component_solidity': 'solid',
    'component_intensity_contrast': 'contrast',

    # ADD these to the existing dict:
    'geo_gap_f1': 'geo_gap_f1',
    'geo_gap_miou': 'geo_gap_miou',
    'within_f1': 'within_f1',
    'loco_f1': 'loco_f1',
}


def get_abbreviated_name(full_name: str) -> str:
    """Get abbreviated name for plotting"""
    return ATTRIBUTE_ABBREVIATIONS.get(full_name, full_name)