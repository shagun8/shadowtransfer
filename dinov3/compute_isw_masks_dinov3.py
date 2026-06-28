"""
Precompute ISW (Instance Selective Whitening) sensitivity masks for DINOv3.

Offline, one-time computation:
  1. Load pretrained DINOv3ShadowDetector.
  2. For each training image: forward original + photometrically augmented version.
  3. Capture intermediate features via DINOv3FeatureHooks at blocks [3, 6, 9].
  4. Compute per-element variance of the two covariances across all images.
  5. K-means cluster upper-triangular variance values.
  6. Save binary mask: 1 for high-variance (style-sensitive) elements.

Output: one .npy mask per layer + metadata.json  →  pass --output_dir to ISW training.
"""

import os
import sys
import argparse
import json
import time
import math
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF

# Ensure project root (dinov3 folder) is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dinov3_model import DINOv3ShadowDetector
from utils.isw_loss_dinov3 import DINOv3FeatureHooks


# ──────────────────────────────────────────────────────────────────────────────
# Photometric augmentation  (simulates domain / city shift)
# ──────────────────────────────────────────────────────────────────────────────

class PhotometricTransform:
    """
    Random color jitter + Gaussian blur on a normalised [C, H, W] tensor.
    De-normalises → augments in pixel space → re-normalises.
    Mimics cross-city illumination / color-palette variation.
    """

    def __init__(self,
                 brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,
                 blur_prob=0.5, blur_kernel=5, blur_sigma=(0.1, 2.0),
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.brightness  = brightness
        self.contrast    = contrast
        self.saturation  = saturation
        self.hue         = hue
        self.blur_prob   = blur_prob
        self.blur_kernel = blur_kernel
        self.blur_sigma  = blur_sigma
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std  = torch.tensor(std).view(3, 1, 1)

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: [3, H, W] normalised tensor.
        Returns:
            Augmented [3, H, W] normalised tensor.
        """
        rgb = img_tensor.clone()

        # De-normalise to [0, 1]
        rgb = rgb * self.std + self.mean
        rgb = rgb.clamp(0, 1)

        # Colour jitter (randomised order)
        fn_idx = list(range(4))
        random.shuffle(fn_idx)
        for fn_id in fn_idx:
            if fn_id == 0:
                bf = random.uniform(max(0.0, 1.0 - self.brightness),
                                    1.0 + self.brightness)
                rgb = TF.adjust_brightness(rgb, bf)
            elif fn_id == 1:
                cf = random.uniform(max(0.0, 1.0 - self.contrast),
                                    1.0 + self.contrast)
                rgb = TF.adjust_contrast(rgb, cf)
            elif fn_id == 2:
                sf = random.uniform(max(0.0, 1.0 - self.saturation),
                                    1.0 + self.saturation)
                rgb = TF.adjust_saturation(rgb, sf)
            elif fn_id == 3:
                hf = random.uniform(-self.hue, self.hue)
                rgb = TF.adjust_hue(rgb, hf)

        # Gaussian blur
        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma)
            rgb   = TF.gaussian_blur(rgb, kernel_size=self.blur_kernel,
                                     sigma=sigma)

        rgb = rgb.clamp(0, 1)

        # Re-normalise
        rgb = (rgb - self.mean) / self.std
        return rgb


# ──────────────────────────────────────────────────────────────────────────────
# Instance covariance helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_instance_covariance(feat):
    """
    Instance-standardised covariance for a single image feature map.

    Args:
        feat: [C, H, W]  (single image from the batch)

    Returns:
        cov: [C, C]  covariance of standardised features
    """
    C, H, W = feat.shape
    N        = H * W
    feat_flat = feat.reshape(C, N)                          # [C, N]
    mean      = feat_flat.mean(dim=1, keepdim=True)         # [C, 1]
    std       = feat_flat.std(dim=1, keepdim=True) + 1e-5   # [C, 1]
    feat_std  = (feat_flat - mean) / std                    # [C, N]
    cov       = feat_std @ feat_std.T / N                   # [C, C]
    return cov


# ──────────────────────────────────────────────────────────────────────────────
# Main precompute logic
# ──────────────────────────────────────────────────────────────────────────────

def precompute_masks(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Model ──────────────────────────────────────────────────────────────────
    print('Loading DINOv3ShadowDetector...')
    model = DINOv3ShadowDetector(
        num_classes=2,
        model_name=args.model_name,
        weights_path=args.weights_path,
        pretrained=True,
        frozen_stages=-1,
    ).to(device)

    if args.checkpoint:
        print(f'Loading fine-tuned checkpoint: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError as e:
            print(f'  Warning: partial load ({e})')
            sd = ckpt['model_state_dict']
            md = model.state_dict()
            compatible = {k: v for k, v in sd.items()
                          if k in md and v.shape == md[k].shape}
            md.update(compatible)
            model.load_state_dict(md)
            print(f'  Loaded {len(compatible)}/{len(sd)} layers')

    model.eval()

    # ── Hooks ──────────────────────────────────────────────────────────────────
    layer_names = args.layers.split(',')
    print(f'Hooking layers: {layer_names}')
    hooks = DINOv3FeatureHooks(
        model,
        layer_names=layer_names,
        img_size=args.img_size,
        patch_size=16,
    )

    # ── Dataset ────────────────────────────────────────────────────────────────
    print('\nLoading dataset...')
    from data.dataset import get_dataloaders, LOCO_FOLDS

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
    train_dataset = dataloaders['train'].dataset

    num_samples = (min(args.num_samples, len(train_dataset))
                   if args.num_samples > 0 else len(train_dataset))
    print(f'Using {num_samples}/{len(train_dataset)} training images.')

    loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # ── Photometric transform ──────────────────────────────────────────────────
    photo_aug = PhotometricTransform()

    # ── Accumulate per-element variance across images ─────────────────────────
    variance_accum = {name: None for name in layer_names}
    count = 0

    print('\nComputing covariance sensitivity...')
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            if count >= num_samples:
                break

            images = batch['image']    # [1, 3, H, W]
            img    = images[0]         # [3, H, W]

            # ---- Original image ----
            img_orig = images.to(device)
            _        = model(img_orig)      # triggers hooks
            cov_orig = {}
            for name in layer_names:
                cov_orig[name] = compute_instance_covariance(
                    hooks.features[name][0])   # [C, C]

            # ---- Photometrically augmented image ----
            img_aug = photo_aug(img).unsqueeze(0).to(device)
            _       = model(img_aug)
            cov_aug = {}
            for name in layer_names:
                cov_aug[name] = compute_instance_covariance(
                    hooks.features[name][0])

            # ---- Per-element variance  (Eq. 14-15 from RobustNet) ----
            for name in layer_names:
                mu_i  = 0.5 * (cov_orig[name] + cov_aug[name])
                var_i = 0.5 * ((cov_orig[name] - mu_i) ** 2
                               + (cov_aug[name] - mu_i) ** 2)
                if variance_accum[name] is None:
                    variance_accum[name] = var_i.cpu()
                else:
                    variance_accum[name] += var_i.cpu()

            count += 1
            if count % 100 == 0 or count == num_samples:
                elapsed = time.time() - t0
                print(f'  [{count}/{num_samples}]  {elapsed:.1f}s elapsed')

    # Average across images
    for name in layer_names:
        variance_accum[name] /= count

    # ── K-means clustering → binary masks ─────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    masks_info = {}

    print('\nClustering covariance elements...')
    for name in layer_names:
        V = variance_accum[name].numpy()    # [C, C]
        C = V.shape[0]

        # Extract strict upper-triangular elements only
        rows, cols = np.triu_indices(C, k=1)
        values     = V[rows, cols]          # [C*(C-1)/2]

        n_elements = len(values)
        print(f'  {name}: {C}×{C} covariance, {n_elements} upper-triangular elements')

        # K-means with k clusters; lowest m clusters → content, rest → style
        k = args.kmeans_k
        m = args.kmeans_m
        kmeans   = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels   = kmeans.fit_predict(values.reshape(-1, 1))
        centroids = kmeans.cluster_centers_.flatten()

        sorted_idx    = np.argsort(centroids)
        high_clusters = set(sorted_idx[m:].tolist())

        # Binary mask: 1 = style-sensitive (whiten), 0 = content (preserve)
        mask = np.zeros((C, C), dtype=np.float32)
        for idx_el, (r, c) in enumerate(zip(rows, cols)):
            if labels[idx_el] in high_clusters:
                mask[r, c] = 1.0

        mask_path = os.path.join(args.output_dir, f'{name}_mask.npy')
        np.save(mask_path, mask)

        n_selected = int(mask.sum())
        masks_info[name] = {
            'channels':                   C,
            'upper_triangular_elements':  n_elements,
            'selected_style_elements':    n_selected,
            'fraction_selected':          round(n_selected / max(n_elements, 1), 4),
            'cluster_centroids':          centroids[sorted_idx].tolist(),
        }
        print(f'    → {n_selected}/{n_elements} elements selected as style-sensitive '
              f'({n_selected / max(n_elements, 1) * 100:.1f}%)')

    # Save variance matrices for optional debugging / re-clustering
    for name in layer_names:
        var_path = os.path.join(args.output_dir, f'{name}_variance.npy')
        np.save(var_path, variance_accum[name].numpy())

    # Metadata
    meta = {
        'timestamp':       datetime.now().isoformat(),
        'num_images_used': count,
        'layers':          layer_names,
        'kmeans_k':        args.kmeans_k,
        'kmeans_m':        args.kmeans_m,
        'model_name':      args.model_name,
        'img_size':        args.img_size,
        'mode':            args.mode,
        'fold_id':         args.fold_id,
        'resolution':      args.resolution,
        'checkpoint':      args.checkpoint,
        'masks':           masks_info,
    }
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)

    print(f'\nAll masks and metadata saved to: {args.output_dir}')
    hooks.remove()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Precompute ISW sensitivity masks for DINOv3')

    # Data
    p.add_argument('--data_root',       type=str, default=None)
    p.add_argument('--base_data_root',  type=str, default=None)
    p.add_argument('--mode',            type=str, default='loco',
                   choices=['single', 'all', 'loco'])
    p.add_argument('--cities',          type=str, nargs='+', default=None)
    p.add_argument('--resolution',      type=str, default=None,
                   choices=['highres', 'midres'])
    p.add_argument('--fold_id',         type=int, default=None,
                   choices=[0, 1, 2])
    p.add_argument('--img_size',        type=int, default=384)
    p.add_argument('--num_workers',     type=int, default=2)
    p.add_argument('--num_samples',     type=int, default=0,
                   help='Max training images to use (0 = all)')

    # Model
    p.add_argument('--model_name',      type=str, default='dinov3_vits16',
                   choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    p.add_argument('--weights_path',    type=str, default=None,
                   help='Path to DINOv3 pretrained weights .pth')
    p.add_argument('--checkpoint',      type=str, default=None,
                   help='Optional fine-tuned checkpoint (before ISW training)')

    # ISW specifics
    p.add_argument('--layers',          type=str, default='block3,block6,block9',
                   help='Comma-separated block names: block3,block6,block9[,block11]')
    p.add_argument('--kmeans_k',        type=int, default=3,
                   help='K-means clusters (default: 3)')
    p.add_argument('--kmeans_m',        type=int, default=1,
                   help='Lowest clusters assigned to content (default: 1)')

    # Output
    p.add_argument('--output_dir',      type=str, required=True,
                   help='Directory to save mask .npy files')
    p.add_argument('--device',          type=str, default='cuda')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    precompute_masks(args)