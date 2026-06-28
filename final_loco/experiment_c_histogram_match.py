"""
Experiment C: Test-Time Intensity Standardization
...  (docstring unchanged)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from experiment_utils import (
    load_model, extract_logits, run_inference,
    compute_histogram, histogram_match_image,
    get_dataset, IMG_SIZE,
)


class HistogramMatchedDataset(Dataset):
    """
    Wraps an existing shadow dataset and applies histogram matching
    to each image at load time so that the test city's intensity
    distribution is shifted toward the training cities' aggregate.
    """

    def __init__(self, base_dataset, source_cdf, model_type):
        self.base       = base_dataset
        self.source_cdf = source_cdf
        self.model_type = model_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample     = self.base[idx]
        img_tensor = sample['image']
        img_np     = img_tensor.numpy()

        n_channels = img_np.shape[0]
        rgb        = img_np[:3] if n_channels >= 3 else img_np

        vmin, vmax = rgb.min(), rgb.max()
        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        if vmax <= 1.5:
            if vmin < -1.5:
                rgb_denorm = rgb[:3] * imagenet_std + imagenet_mean
            else:
                rgb_denorm = rgb[:3]
            rgb_uint8 = np.clip(rgb_denorm * 255, 0, 255).astype(np.uint8)
        else:
            rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)

        rgb_hwc = rgb_uint8.transpose(1, 2, 0)
        gray    = rgb_hwc.mean(axis=2).astype(np.uint8)

        target_hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        target_hist /= (target_hist.sum() + 1e-10)
        target_cdf  = np.cumsum(target_hist)

        matched_hwc = histogram_match_image(
            rgb_hwc, self.source_cdf, target_cdf)

        matched = matched_hwc.transpose(2, 0, 1).astype(np.float32)

        if vmax <= 1.5 and vmin < -1.5:
            matched = matched / 255.0
            matched = (matched - imagenet_mean.astype(np.float32)) \
                      / imagenet_std.astype(np.float32)
        elif vmax <= 1.5:
            matched = matched / 255.0

        if n_channels > 3:
            full = img_np.copy()
            full[:3] = matched
            sample['image'] = torch.from_numpy(full)
        else:
            sample['image'] = torch.from_numpy(matched)

        return sample


def run_experiment_c(args):
    """Main entry point for Experiment C."""
    print("=" * 70)
    print("EXPERIMENT C: Test-Time Intensity Standardization")
    print("=" * 70)
    print(f"  Model: {args.model_type}/{args.model_variant}")
    print(f"  Holdout city: {args.holdout_city}")
    print(f"  Resolution: {args.res}")
    print(f"  Source image dirs: {args.source_image_dirs}")
    print()

    # 1. Compute aggregate source histogram
    print("--- Computing source cities' aggregate histogram ---")
    agg_hist = np.zeros(256, dtype=np.float64)
    for src_dir in args.source_image_dirs:
        if not os.path.isdir(src_dir):
            print(f"  WARNING: source dir not found: {src_dir}")
            continue
        h, _ = compute_histogram(src_dir, max_images=200)
        agg_hist += h
        print(f"  Added: {src_dir}")

    if agg_hist.sum() == 0:
        print("  ERROR: no source histograms computed. "
              "Check --source_image_dirs.")
        return

    agg_hist  /= agg_hist.sum()
    source_cdf = np.cumsum(agg_hist)
    print(f"  Source CDF computed "
          f"(mean intensity ≈ {np.sum(np.arange(256) * agg_hist):.1f})")

    # 2. Load model
    model, device = load_model(
        args.model_type, args.model_variant, args.checkpoint_path,
        device=args.device,
        dinov3_model_name=args.dinov3_model_name,
        dinov3_weights_path=args.dinov3_weights_path,
        dinov3_pretrained=args.dinov3_pretrained,
        dinov3_frozen_stages=args.dinov3_frozen_stages,
    )

    # 3. Create histogram-matched dataset
    print("\n--- Creating histogram-matched dataset ---")
    base_dataset    = get_dataset(
        args.model_type, args.test_data_root, split='test', augment=False)
    matched_dataset = HistogramMatchedDataset(
        base_dataset, source_cdf, args.model_type)

    matched_loader = DataLoader(
        matched_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print(f"  Matched dataset: {len(matched_dataset)} images")

    # 4. Inference on matched images
    print("\n--- Running inference on histogram-matched images ---")
    pred_dir = os.path.join(
        args.output_base, args.holdout_city, args.res,
        args.model_type, args.model_variant)
    run_inference(model, matched_loader, device, pred_dir)

    # 5. Save sample matched images for inspection
    if args.save_samples:
        print("\n--- Saving sample matched images ---")
        sample_dir = os.path.join(
            args.output_base, 'matched_samples',
            f"{args.holdout_city}_{args.res}")
        os.makedirs(sample_dir, exist_ok=True)

        imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        imagenet_std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        for i in range(min(10, len(matched_dataset))):
            sample = matched_dataset[i]
            img_t  = sample['image']
            fname  = sample['filename']
            rgb    = img_t[:3].numpy() if img_t.shape[0] >= 3 else img_t.numpy()
            vmin, vmax = rgb.min(), rgb.max()
            if vmax <= 2.0:
                if vmin < -1.5:
                    rgb = rgb * imagenet_std + imagenet_mean
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            Image.fromarray(rgb.transpose(1, 2, 0)).save(
                os.path.join(sample_dir, f"matched_{fname}"))

        print(f"  Saved {min(10, len(matched_dataset))} samples to {sample_dir}")

    print("\n" + "=" * 70)
    print("EXPERIMENT C COMPLETE")
    print("=" * 70)


def get_args():
    p = argparse.ArgumentParser(
        description="Experiment C: Test-Time Histogram Matching")
    p.add_argument('--model_type', required=True,
                   choices=['mamnet', 'oglanet', 'dinov3'])
    p.add_argument('--model_variant', default='vanilla',
                   choices=['base', 'vanilla', 'fda', 'segdesic',
                            'iim', 'isw', 'mrfp_plus', 'fada'])
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--holdout_city', required=True,
                   choices=['chicago', 'miami', 'phoenix'])
    p.add_argument('--res', required=True, choices=['highres', 'midres'])
    p.add_argument('--test_data_root', required=True)
    p.add_argument('--source_image_dirs', nargs='+', required=True)
    p.add_argument('--output_base',
                   default=os.path.join(os.environ["PROJECT_ROOT"],
                                        'data', 'Test_img_results', 'experiment_c'))
    p.add_argument('--save_samples', action='store_true', default=True)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    # DINOv3
    p.add_argument('--dinov3_model_name', default='dinov3_vits16')
    p.add_argument('--dinov3_weights_path', default=None)
    p.add_argument('--dinov3_pretrained', action='store_true', default=True)
    p.add_argument('--dinov3_frozen_stages', type=int, default=-1)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_experiment_c(args)