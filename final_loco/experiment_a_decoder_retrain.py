"""
Experiment A: Decoder Retraining on Holdout City

Freeze the LOCO-trained encoder. Re-initialize the decoder.
Retrain ONLY the decoder on a fraction of the holdout city's training data.
Run inference on the holdout city's test set. Save predictions.

This tests whether encoder features are sufficient for the holdout city
(decoder was just miscalibrated) or genuinely contaminated.

Usage:
    python experiment_a_decoder_retrain.py \
        --model_type mamnet \
        --model_variant vanilla \
        --checkpoint_path /path/to/loco_checkpoint_best.pth \
        --holdout_city phoenix \
        --res highres \
        --train_data_root /path/to/Final_data/phoenix/highres \
        --test_data_root /path/to/Final_data_test/phoenix/highres \
        --train_fraction 0.25 \
        --epochs 30 \
        --output_base /path/to/Test_img_results/experiment_a
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiment_utils import (
    load_model, freeze_encoder, reinit_decoder,
    classify_parameters, extract_logits, run_inference,
    get_dataset, get_dataloader, IMG_SIZE,
)


def train_decoder(model, train_loader, device, epochs=30, lr=1e-3,
                  weight_decay=1e-4, log_interval=20):
    """
    Train only decoder parameters (encoder is frozen).
    Uses cross-entropy loss on the main output only.
    Metrics are tracked per-image (mean over images, not pixel pooling).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Training {len(trainable)} param groups, "
          f"{sum(p.numel() for p in trainable)} total params")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    # Keep encoder BN in eval mode
    for name, module in model.named_modules():
        is_encoder = any(name.startswith(p)
                         for p in ['encoder.', 'backbone.'])
        if is_encoder and isinstance(module,
                                     (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    best_loss = float('inf')

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches    = 0

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)

            # Resolve label key
            if 'label' in batch:
                labels = batch['label'].to(device)
            elif 'mask' in batch:
                labels = batch['mask'].to(device)
            else:
                for k, v in batch.items():
                    if k not in ('image', 'filename') and isinstance(v, torch.Tensor):
                        if v.dtype in (torch.long, torch.int, torch.int32):
                            labels = v.to(device); break
                        elif v.max() <= 1 and v.min() >= 0:
                            labels = v.long().to(device); break
                else:
                    raise KeyError(
                        f"Cannot find label in batch keys: {list(batch.keys())}")

            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            if labels.dtype == torch.float32:
                labels = (labels > 0.5).long()
            if labels.shape[-2:] != (IMG_SIZE, IMG_SIZE):
                labels = nn.functional.interpolate(
                    labels.unsqueeze(1).float(),
                    size=(IMG_SIZE, IMG_SIZE), mode='nearest'
                ).squeeze(1).long()

            optimizer.zero_grad()
            outputs = model(images)
            logits  = extract_logits(outputs)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:],
                    mode='bilinear', align_corners=False)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

            if (i + 1) % log_interval == 0:
                print(f"    Epoch {epoch+1}/{epochs}, "
                      f"Batch {i+1}, "
                      f"Loss: {running_loss/n_batches:.4f}")

        epoch_loss = running_loss / max(n_batches, 1)
        scheduler.step()
        best_loss  = min(best_loss, epoch_loss)

        print(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, "
              f"best={best_loss:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.6f}")

    return best_loss


def run_experiment_a(args):
    """Main entry point for Experiment A."""
    print("=" * 70)
    print("EXPERIMENT A: Decoder Retraining on Holdout City")
    print("=" * 70)
    print(f"  Model          : {args.model_type}/{args.model_variant}")
    print(f"  Holdout city   : {args.holdout_city}")
    print(f"  Resolution     : {args.res}")
    print(f"  Train fraction : {args.train_fraction}")
    print(f"  Epochs         : {args.epochs}")
    print()

    # 1. Load model with LOCO checkpoint
    model, device = load_model(
        args.model_type, args.model_variant, args.checkpoint_path,
        device=args.device,
        dinov3_model_name=args.dinov3_model_name,
        dinov3_weights_path=args.dinov3_weights_path,
        dinov3_pretrained=args.dinov3_pretrained,
        dinov3_frozen_stages=args.dinov3_frozen_stages,
    )

    # 2. Freeze encoder, re-init decoder
    print("\n--- Freezing encoder ---")
    freeze_encoder(model, args.model_type)
    print("\n--- Re-initializing decoder ---")
    reinit_decoder(model, args.model_type)

    # 3. Load training data from holdout city
    print("\n--- Loading training data ---")
    full_dataset = get_dataset(
        args.model_type, args.train_data_root, split='train', augment=True)

    n_total = len(full_dataset)
    n_use   = max(1, int(n_total * args.train_fraction))
    indices = np.linspace(0, n_total - 1, n_use, dtype=int).tolist()
    train_subset = Subset(full_dataset, indices)
    print(f"  Using {n_use}/{n_total} training images "
          f"({args.train_fraction*100:.0f}%)")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 4. Train decoder
    print("\n--- Training decoder ---")
    train_decoder(model, train_loader, device,
                  epochs=args.epochs, lr=args.lr,
                  weight_decay=args.weight_decay)

    # 5. Inference on test set
    print("\n--- Running inference on test set ---")
    test_loader = get_dataloader(
        args.model_type, args.test_data_root, split='test',
        batch_size=args.batch_size, num_workers=args.num_workers)

    pred_dir = os.path.join(
        args.output_base, args.holdout_city, args.res,
        args.model_type, args.model_variant)
    run_inference(model, test_loader, device, pred_dir)

    # 6. Data efficiency curve (optional)
    if args.data_efficiency:
        print("\n--- Data efficiency curve ---")
        fractions = [0.05, 0.10, 0.15, 0.20]
        for frac in fractions:
            if abs(frac - args.train_fraction) < 0.01:
                continue
            print(f"\n  === Fraction: {frac} ===")

            model_de, _ = load_model(
                args.model_type, args.model_variant, args.checkpoint_path,
                device=args.device,
                dinov3_model_name=args.dinov3_model_name,
                dinov3_weights_path=args.dinov3_weights_path,
                dinov3_pretrained=args.dinov3_pretrained,
                dinov3_frozen_stages=args.dinov3_frozen_stages,
            )
            freeze_encoder(model_de, args.model_type)
            reinit_decoder(model_de, args.model_type)

            n_de    = max(1, int(n_total * frac))
            idx_de  = np.linspace(0, n_total - 1, n_de, dtype=int).tolist()
            loader_de = DataLoader(
                Subset(full_dataset, idx_de),
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)

            train_decoder(model_de, loader_de, device,
                          epochs=args.epochs, lr=args.lr)

            frac_str    = f"{int(frac*100)}pct"
            pred_dir_de = os.path.join(
                args.output_base + f"_de{frac_str}",
                args.holdout_city, args.res,
                args.model_type, args.model_variant)
            test_loader_de = get_dataloader(
                args.model_type, args.test_data_root, split='test',
                batch_size=args.batch_size, num_workers=args.num_workers)
            run_inference(model_de, test_loader_de, device, pred_dir_de)
            del model_de

    print("\n" + "=" * 70)
    print("EXPERIMENT A COMPLETE")
    print("=" * 70)


def get_args():
    p = argparse.ArgumentParser(
        description="Experiment A: Decoder Retraining on Holdout City")
    p.add_argument('--model_type', required=True,
                   choices=['mamnet', 'oglanet', 'dinov3'])
    p.add_argument('--model_variant', default='vanilla',
                   choices=['base', 'vanilla', 'fda', 'segdesic',
                            'iim', 'isw', 'mrfp_plus', 'fada'])
    p.add_argument('--checkpoint_path', required=True)
    p.add_argument('--holdout_city', required=True,
                   choices=['chicago', 'miami', 'phoenix'])
    p.add_argument('--res', required=True, choices=['highres', 'midres'])
    p.add_argument('--train_data_root', required=True)
    p.add_argument('--test_data_root', required=True)
    p.add_argument('--output_base',
                   default=os.path.join(os.environ["PROJECT_ROOT"],
                                        'data', 'Test_img_results', 'experiment_a'))
    p.add_argument('--train_fraction', type=float, default=0.25)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    p.add_argument('--data_efficiency', action='store_true',
                   help='Also run 5/10/15/20%% data-efficiency points')
    # DINOv3
    p.add_argument('--dinov3_model_name', default='dinov3_vits16')
    p.add_argument('--dinov3_weights_path', default=None)
    p.add_argument('--dinov3_pretrained', action='store_true', default=True)
    p.add_argument('--dinov3_frozen_stages', type=int, default=-1)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_experiment_a(args)