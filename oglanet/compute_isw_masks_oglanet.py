"""
Precompute ISW (Instance Selective Whitening) sensitivity masks for OGLANet.

Offline, one-time computation:
  1. Forward each training image through OGLANet's GLAMEncoder →
     covariance at glam1, glam2, glam3.
  2. Apply photometric augmentation → forward again → covariance.
  3. Compute per-element variance of the two covariances across all images.
  4. K-means cluster the upper-triangular variance values.
  5. Save binary mask: 1 for high-variance (style-sensitive) elements.

Output: one .npy mask per layer + metadata.json.

Key OGLANet difference vs MAMNet:
  - Model is OGLANet (GLAMEncoder), not MAMNet.
  - Encoder hooks target glam1/glam2/glam3, not layer1/layer2/layer3.
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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.oglanet import OGLANet
from utils.isw_loss import ISWLoss, EncoderFeatureHooks


# ──────────────────────────────────────────────────────────────────
# OGLANet-specific encoder hooks
# ──────────────────────────────────────────────────────────────────

class OGLANetEncoderFeatureHooks(EncoderFeatureHooks):
    """
    Forward hooks targeting OGLANet's GLAMEncoder modules.
    Maps feat1→glam1, feat2→glam2, feat3→glam3, feat4→glam4.
    """
    _LAYER_MAP = {
        'feat1': 'glam1',
        'feat2': 'glam2',
        'feat3': 'glam3',
        'feat4': 'glam4',
    }


# ──────────────────────────────────────────────────────────────────
# Photometric augmentation (simulates domain shift)
# Identical to MAMNet compute_isw_masks.py
# ──────────────────────────────────────────────────────────────────

class PhotometricTransform:
    """
    Random color jittering + Gaussian blur applied in pixel space.
    Operates on a [C, H, W] normalised tensor (C = 3 or 4).
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
        C   = img_tensor.shape[0]
        rgb = img_tensor[:3]

        # De-normalise to [0, 1]
        rgb = rgb * self.std + self.mean
        rgb = rgb.clamp(0, 1)

        # Color jitter (shuffled order)
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

        if random.random() < self.blur_prob:
            sigma = random.uniform(*self.blur_sigma)
            rgb   = TF.gaussian_blur(rgb, kernel_size=self.blur_kernel,
                                     sigma=sigma)

        rgb = rgb.clamp(0, 1)
        rgb = (rgb - self.mean) / self.std

        if C == 4:
            return torch.cat([rgb, img_tensor[3:4]], dim=0)
        return rgb


# ──────────────────────────────────────────────────────────────────
# Covariance computation
# ──────────────────────────────────────────────────────────────────

def compute_instance_covariance(feat):
    """
    Instance-standardised covariance for a single image feature map.

    Args:
        feat: [C, H, W]
    Returns:
        cov: [C, C]
    """
    C, H, W  = feat.shape
    N        = H * W
    feat_flat = feat.reshape(C, N)
    mean      = feat_flat.mean(dim=1, keepdim=True)
    std       = feat_flat.std(dim=1, keepdim=True) + 1e-5
    feat_std  = (feat_flat - mean) / std
    cov       = feat_std @ feat_std.T / N
    return cov


# ──────────────────────────────────────────────────────────────────
# Main precompute logic
# ──────────────────────────────────────────────────────────────────

def precompute_masks(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Model ────────────────────────────────────────────────────
    print('Creating OGLANet model for feature extraction...')
    model = OGLANet(
        num_classes=args.num_classes,
        pretrained=True,
        img_size=args.img_size,
        use_contrast=args.use_contrast,
    ).to(device)

    if args.checkpoint:
        print(f'Loading checkpoint: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location=device,
                          weights_only=False)
        try:
            model.load_state_dict(ckpt['model_state_dict'])
        except RuntimeError:
            sd = ckpt['model_state_dict']
            md = model.state_dict()
            compatible = {k: v for k, v in sd.items()
                          if k in md and v.shape == md[k].shape}
            md.update(compatible)
            model.load_state_dict(md)
            print(f'  Partial load: {len(compatible)}/{len(sd)} layers')

    model.eval()

    # ── OGLANet-specific hooks ────────────────────────────────────
    layer_names = args.layers.split(',')
    print(f'Hooking layers: {layer_names} '
          f'(map: feat1→glam1, feat2→glam2, feat3→glam3)')
    hooks = OGLANetEncoderFeatureHooks(model, layer_names=layer_names)

    # ── Dataset ──────────────────────────────────────────────────
    print('Loading dataset...')
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
        use_contrast=args.use_contrast,
    )
    dataset = dataloaders['train'].dataset

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    num_samples = (min(args.num_samples, len(dataset))
                   if args.num_samples > 0 else len(dataset))
    print(f'Using {num_samples}/{len(dataset)} training images '
          f'for mask computation.')

    # ── Photometric augmentation ──────────────────────────────────
    photo_aug = PhotometricTransform()

    # ── Accumulate variance matrices ─────────────────────────────
    variance_accum = {name: None for name in layer_names}
    count = 0

    print('Computing covariance sensitivity across training set...')
    t0 = time.time()

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if count >= num_samples:
                break

            images = batch['image']          # [1, C, H, W]
            img    = images[0]               # [C, H, W]

            # --- Original image ---
            img_orig = images.to(device)
            _ = model(img_orig)
            cov_orig = {
                name: compute_instance_covariance(hooks.features[name][0])
                for name in layer_names
            }

            # --- Photometric-augmented image ---
            img_aug = photo_aug(img).unsqueeze(0).to(device)
            _ = model(img_aug)
            cov_aug = {
                name: compute_instance_covariance(hooks.features[name][0])
                for name in layer_names
            }

            # --- Per-element variance (Eq. 14-15 in RobustNet) ---
            for name in layer_names:
                mu_i  = 0.5 * (cov_orig[name] + cov_aug[name])
                var_i = 0.5 * ((cov_orig[name] - mu_i) ** 2
                               + (cov_aug[name] - mu_i) ** 2)
                if variance_accum[name] is None:
                    variance_accum[name] = var_i.cpu()
                else:
                    variance_accum[name] += var_i.cpu()

            count += 1
            if (count % 50 == 0) or count == num_samples:
                print(f'  [{count}/{num_samples}]  '
                      f'{time.time() - t0:.1f}s elapsed')

    # Average over images
    for name in layer_names:
        variance_accum[name] /= count

    # ── K-means clustering → binary masks ────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    masks_info = {}

    for name in layer_names:
        V = variance_accum[name].numpy()   # [C, C]
        C = V.shape[0]

        rows, cols = np.triu_indices(C, k=1)
        values     = V[rows, cols]         # [C*(C-1)/2]

        k = args.kmeans_k
        m = args.kmeans_m
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(values.reshape(-1, 1))
        centroids = kmeans.cluster_centers_.flatten()

        sorted_idx    = np.argsort(centroids)
        high_clusters = set(sorted_idx[m:].tolist())

        mask = np.zeros((C, C), dtype=np.float32)
        for el_idx in range(len(values)):
            if labels[el_idx] in high_clusters:
                mask[rows[el_idx], cols[el_idx]] = 1.0

        mask_path = os.path.join(args.output_dir, f'{name}_mask.npy')
        np.save(mask_path, mask)

        n_selected = int(mask.sum())
        n_total    = len(values)
        masks_info[name] = {
            'channels':                   C,
            'upper_triangular_elements':  n_total,
            'selected_style_elements':    n_selected,
            'fraction_selected':          round(n_selected / max(n_total, 1), 4),
            'cluster_centroids':          centroids[sorted_idx].tolist(),
        }
        print(f'  {name} (→{OGLANetEncoderFeatureHooks._LAYER_MAP[name]}): '
              f'{n_selected}/{n_total} elements selected '
              f'({n_selected / max(n_total, 1) * 100:.1f}%)')

    # Save variance matrices (for debugging)
    for name in layer_names:
        var_path = os.path.join(args.output_dir, f'{name}_variance.npy')
        np.save(var_path, variance_accum[name].numpy())

    # Metadata
    meta = {
        'timestamp':       datetime.now().isoformat(),
        'model':           'OGLANet',
        'num_images_used': count,
        'layers':          layer_names,
        'layer_map':       OGLANetEncoderFeatureHooks._LAYER_MAP,
        'kmeans_k':        args.kmeans_k,
        'kmeans_m':        args.kmeans_m,
        'use_contrast':    args.use_contrast,
        'img_size':        args.img_size,
        'mode':            args.mode,
        'checkpoint':      args.checkpoint,
        'masks':           masks_info,
    }
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
    print(f'\nMasks and metadata saved to: {args.output_dir}')

    hooks.remove()


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='Precompute ISW sensitivity masks for OGLANet')

    # Data
    p.add_argument('--data_root',       type=str, default=None)
    p.add_argument('--base_data_root',  type=str, default=None)
    p.add_argument('--mode',            type=str, default='single',
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
    p.add_argument('--num_classes',   type=int, default=2)
    p.add_argument('--use_contrast',  action='store_true')
    p.add_argument('--checkpoint',    type=str, default=None,
                   help='Optional OGLANet checkpoint to load before '
                        'feature extraction (recommended if available)')

    # ISW specifics
    p.add_argument('--layers',    type=str, default='feat1,feat2,feat3',
                   help='Comma-separated layer names '
                        '(feat1→glam1, feat2→glam2, feat3→glam3)')
    p.add_argument('--kmeans_k',  type=int, default=3,
                   help='Number of k-means clusters (default: 3)')
    p.add_argument('--kmeans_m',  type=int, default=1,
                   help='Number of lowest-variance clusters assigned to '
                        'content (default: 1)')

    # Output
    p.add_argument('--output_dir', type=str, required=True,
                   help='Directory to save mask .npy files')
    p.add_argument('--device',     type=str, default='cuda')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    precompute_masks(args)