"""
Precompute ISW (Instance Selective Whitening) sensitivity masks.

Offline, one-time computation:
  1. Forward each training image through the encoder → covariance at each layer.
  2. Apply photometric augmentation → forward again → covariance.
  3. Compute per-element variance of the two covariances across all images.
  4. K-means cluster the upper-triangular variance values.
  5. Save binary mask: 1 for high-variance (style-sensitive) elements.

Output: one .npy mask per layer + metadata.json.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF
import random

# Ensure project root is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet import MAMNet
from utils.isw_loss import EncoderFeatureHooks


# ──────────────────────────────────────────────────────────────────
# Photometric augmentation (simulates domain shift)
# ──────────────────────────────────────────────────────────────────

class PhotometricTransform:
    """
    Applies random color jittering + Gaussian blur.
    Operates on a [C, H, W] tensor (normalised).
    We de-normalise → augment → re-normalise so that the augmentation
    happens in pixel space.
    """

    def __init__(self,
                 brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,
                 blur_prob=0.5, blur_kernel=5, blur_sigma=(0.1, 2.0),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.brightness = brightness
        self.contrast   = contrast
        self.saturation = saturation
        self.hue        = hue
        self.blur_prob  = blur_prob
        self.blur_kernel = blur_kernel
        self.blur_sigma  = blur_sigma
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: [C, H, W] normalised tensor.  C may be 3 (RGB) or 4 (RGBC).
        Returns:
            Augmented tensor of the same shape.
        """
        C = img_tensor.shape[0]
        rgb = img_tensor[:3]

        # De-normalise to [0, 1]
        rgb = rgb * self.std + self.mean
        rgb = rgb.clamp(0, 1)

        # Color jitter (operates on [C,H,W] float tensor in [0,1])
        fn_idx = list(range(4))
        random.shuffle(fn_idx)
        for fn_id in fn_idx:
            if fn_id == 0:
                bf = random.uniform(max(0, 1 - self.brightness),
                                    1 + self.brightness)
                rgb = TF.adjust_brightness(rgb, bf)
            elif fn_id == 1:
                cf = random.uniform(max(0, 1 - self.contrast),
                                    1 + self.contrast)
                rgb = TF.adjust_contrast(rgb, cf)
            elif fn_id == 2:
                sf = random.uniform(max(0, 1 - self.saturation),
                                    1 + self.saturation)
                rgb = TF.adjust_saturation(rgb, sf)
            elif fn_id == 3:
                hf = random.uniform(-self.hue, self.hue)
                rgb = TF.adjust_hue(rgb, hf)

        # Gaussian blur
        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma)
            rgb = TF.gaussian_blur(rgb, kernel_size=self.blur_kernel,
                                   sigma=sigma)

        rgb = rgb.clamp(0, 1)

        # Re-normalise
        rgb = (rgb - self.mean) / self.std

        if C == 4:
            # Keep the 4th channel (contrast) unchanged
            return torch.cat([rgb, img_tensor[3:4]], dim=0)
        return rgb


# ──────────────────────────────────────────────────────────────────
# Covariance computation helpers
# ──────────────────────────────────────────────────────────────────

def compute_instance_covariance(feat):
    """
    Compute instance-standardised covariance for a single image.

    Args:
        feat: [C, H, W] feature tensor

    Returns:
        cov: [C, C] covariance matrix of the standardised features
    """
    C, H, W = feat.shape
    N = H * W
    feat_flat = feat.reshape(C, N)                          # [C, N]
    mean = feat_flat.mean(dim=1, keepdim=True)              # [C, 1]
    std  = feat_flat.std(dim=1, keepdim=True) + 1e-5        # [C, 1]
    feat_std = (feat_flat - mean) / std                     # [C, N]
    cov = feat_std @ feat_std.T / N                         # [C, C]
    return cov


# ──────────────────────────────────────────────────────────────────
# Main precompute logic
# ──────────────────────────────────────────────────────────────────

def precompute_masks(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Model ────────────────────────────────────────────────────
    print('Creating model...')
    model = MAMNet(
        num_classes=args.num_classes,
        pretrained=True,
        use_aux=False,
        use_contrast=args.use_contrast,
    ).to(device)

    if args.checkpoint:
        print(f'Loading checkpoint: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError:
            # Partial load for mismatched keys
            sd = ckpt['model_state_dict']
            md = model.state_dict()
            compatible = {k: v for k, v in sd.items()
                          if k in md and v.shape == md[k].shape}
            md.update(compatible)
            model.load_state_dict(md)
            print(f'  Loaded {len(compatible)}/{len(sd)} layers (partial)')

    model.eval()

    # ── Hooks ────────────────────────────────────────────────────
    layer_names = args.layers.split(',')
    hooks = EncoderFeatureHooks(model, layer_names)

    # ── Dataset ──────────────────────────────────────────────────
    print('Loading dataset...')
    if args.use_contrast:
        from data.dataset_enhanced import ShadowDatasetEnhanced
        if args.mode == 'single':
            paths = [args.data_root]
        elif args.mode == 'all':
            cities = args.cities if args.cities else ['chicago', 'miami', 'phoenix']
            paths = [os.path.join(args.base_data_root, c, args.resolution)
                     for c in cities]
        elif args.mode == 'loco':
            from data.dataset import LOCO_FOLDS
            fold_cfg = LOCO_FOLDS[args.fold_id]
            paths = [os.path.join(args.base_data_root, c, args.resolution)
                     for c in fold_cfg['train']]
        else:
            raise ValueError(f'Unknown mode: {args.mode}')

        dataset = ShadowDatasetEnhanced(
            root_dir=paths, split='train', img_size=args.img_size,
            task_id=2, augment=False, use_fda=False)
    else:
        from data.dataset import get_dataloaders
        dataloaders = get_dataloaders(
            data_root=args.data_root,
            base_data_root=args.base_data_root,
            mode=args.mode,
            cities=args.cities,
            resolution=args.resolution,
            fold_id=args.fold_id,
            batch_size=1,
            num_workers=args.num_workers,
            img_size=args.img_size,
            use_fda=False,
        )
        dataset = dataloaders['train'].dataset

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    num_samples = min(args.num_samples, len(dataset)) if args.num_samples > 0 \
                  else len(dataset)
    print(f'Using {num_samples}/{len(dataset)} training images for mask computation.')

    # ── Photometric transform ────────────────────────────────────
    photo_aug = PhotometricTransform()

    # ── Accumulate variance matrices ─────────────────────────────
    # For each layer: running sum of per-element variance σ²_i  (Eq. 15)
    variance_accum = {name: None for name in layer_names}
    count = 0

    print('Computing covariance sensitivity...')
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if count >= num_samples:
                break

            images = batch['image']                         # [1, C, H, W]
            img = images[0]                                 # [C, H, W]

            # --- Original ---
            img_orig = images.to(device)
            _ = model(img_orig)
            cov_orig = {}
            for name in layer_names:
                cov_orig[name] = compute_instance_covariance(
                    hooks.features[name][0])                # [C_l, C_l]

            # --- Augmented ---
            img_aug = photo_aug(img).unsqueeze(0).to(device)
            _ = model(img_aug)
            cov_aug = {}
            for name in layer_names:
                cov_aug[name] = compute_instance_covariance(
                    hooks.features[name][0])

            # --- Per-element variance (Eq. 14-15) ---
            for name in layer_names:
                mu_i  = 0.5 * (cov_orig[name] + cov_aug[name])
                var_i = 0.5 * ((cov_orig[name] - mu_i) ** 2
                               + (cov_aug[name] - mu_i) ** 2)
                if variance_accum[name] is None:
                    variance_accum[name] = var_i.cpu()
                else:
                    variance_accum[name] += var_i.cpu()

            count += 1
            if (count % 100 == 0) or count == num_samples:
                elapsed = time.time() - t0
                print(f'  [{count}/{num_samples}]  {elapsed:.1f}s')

    # Average variance across images  (Eq. 13)
    for name in layer_names:
        variance_accum[name] /= count

    # ── K-means clustering → binary masks ────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    masks_info = {}

    for name in layer_names:
        V = variance_accum[name].numpy()                    # [C, C]
        C = V.shape[0]

        # Extract strict upper-triangular elements
        rows, cols = np.triu_indices(C, k=1)
        values = V[rows, cols]                              # [C*(C-1)/2]

        # K-means (k clusters, m lowest → content, rest → style)
        k = args.kmeans_k
        m = args.kmeans_m
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(values.reshape(-1, 1))
        centroids = kmeans.cluster_centers_.flatten()

        # Sort clusters by centroid value
        sorted_idx = np.argsort(centroids)
        low_clusters  = set(sorted_idx[:m].tolist())
        high_clusters = set(sorted_idx[m:].tolist())

        # Build mask: 1 for high-variance (style-sensitive) elements
        mask = np.zeros((C, C), dtype=np.float32)
        for idx_el in range(len(values)):
            if labels[idx_el] in high_clusters:
                mask[rows[idx_el], cols[idx_el]] = 1.0

        # Save
        mask_path = os.path.join(args.output_dir, f'{name}_mask.npy')
        np.save(mask_path, mask)

        n_selected = int(mask.sum())
        n_total    = len(values)
        masks_info[name] = {
            'channels': C,
            'upper_triangular_elements': n_total,
            'selected_style_elements': n_selected,
            'fraction_selected': round(n_selected / max(n_total, 1), 4),
            'cluster_centroids': centroids[sorted_idx].tolist(),
        }
        print(f'  {name}: {n_selected}/{n_total} elements selected '
              f'({n_selected / max(n_total, 1) * 100:.1f}%)')

    # Variance matrices (useful for debugging / analysis)
    for name in layer_names:
        var_path = os.path.join(args.output_dir, f'{name}_variance.npy')
        np.save(var_path, variance_accum[name].numpy())

    # Metadata
    meta = {
        'timestamp': datetime.now().isoformat(),
        'num_images_used': count,
        'layers': layer_names,
        'kmeans_k': args.kmeans_k,
        'kmeans_m': args.kmeans_m,
        'use_contrast': args.use_contrast,
        'img_size': args.img_size,
        'mode': args.mode,
        'checkpoint': args.checkpoint,
        'masks': masks_info,
    }
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f'\nMasks and metadata saved to {args.output_dir}')

    hooks.remove()


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Precompute ISW sensitivity masks for MAMNet')

    # Data
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--base_data_root', type=str, default=None)
    p.add_argument('--mode', type=str, default='single',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--cities', type=str, nargs='+', default=None)
    p.add_argument('--resolution', type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id', type=int, default=None, choices=[0, 1, 2])
    p.add_argument('--img_size', type=int, default=384)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--num_samples', type=int, default=0,
                   help='Max training images to use (0 = all)')

    # Model
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--use_contrast', action='store_true')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Optional model checkpoint to load before feature extraction')

    # ISW specifics
    p.add_argument('--layers', type=str, default='feat1,feat2,feat3',
                   help='Comma-separated encoder layer names to compute masks for')
    p.add_argument('--kmeans_k', type=int, default=3,
                   help='Number of k-means clusters (default: 3)')
    p.add_argument('--kmeans_m', type=int, default=1,
                   help='Number of lowest clusters assigned to content (default: 1)')

    # Output
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory to save mask .npy files')
    p.add_argument('--device', type=str, default='cuda')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    precompute_masks(args)