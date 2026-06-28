"""
Experiment B2: Encoder Retraining on Holdout City

Freeze the LOCO-trained decoder. Reset the encoder to pretrained weights
(ImageNet for CNNs, DINOv2/v3 for ViT). Fine-tune ONLY the encoder on a
fraction of the holdout city's training data. Run inference on the holdout
city's test set. Save predictions.

This is the complement of Experiment A (decoder retraining). Together they
localize the failure:
  - Exp A (decoder retrain):  R≈1 → failure was at decoder
  - Exp B2 (encoder retrain): R≈1 → failure was at encoder
  - Both low → failure is in encoder-decoder coupling

Scientific design choices (document in paper):
  1. Encoder is reset to PRETRAINED weights, not random init.
     Rationale: 25% of ~150 images (≈37 images) cannot train a deep encoder
     from scratch. Pretrained init gives the encoder its best chance of
     learning target-city features, making a low R a stronger negative result.
     This also mirrors the original training pipeline (pretrained → fine-tuned).

  2. Architecture-specific encoder LRs:
       CNNs (MAMNet, OGLANet): 1e-4
       ViT  (DINOv3):          1e-5
     Rationale: Pretrained ViTs are more sensitive to large LR perturbations
     than pretrained CNNs (Dosovitskiy et al. 2021; He et al. 2022). These
     are 10-100× lower than Exp A's decoder LR (1e-3) because encoder
     fine-tuning adapts pretrained representations, whereas Exp A optimizes
     randomly initialized decoder parameters.

  3. Decoder BN layers remain in eval mode (frozen running stats).
     Only encoder BN layers update during training.

Usage:
    python experiment_b2_encoder_retrain.py \\
        --model_type oglanet \\
        --model_variant vanilla \\
        --checkpoint_path /path/to/loco_checkpoint_best.pth \\
        --holdout_city phoenix \\
        --res highres \\
        --train_data_root /path/to/Final_data/phoenix/highres \\
        --test_data_root /path/to/Final_data_test/phoenix/highres
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from experiment_utils import (
    load_model, classify_parameters, extract_logits, run_inference,
    get_dataset, get_dataloader, IMG_SIZE,
    ENCODER_PREFIXES, SCRIPT_DIR, DINOV3_PATH, MAMNET_PATH, OGLANET_PATH,
    _clean_model_paths, _add_path,
)

# ================================================================
# Architecture-specific encoder fine-tuning LRs
# ================================================================
ENCODER_LR_DEFAULTS = {
    'mamnet':  1e-4,
    'oglanet': 1e-4,
    'dinov3':  1e-5,
}


# ================================================================
# FREEZE DECODER (inverse of freeze_encoder)
# ================================================================

def freeze_decoder(model, model_type):
    """
    Freeze all decoder parameters; leave encoder trainable.
    Returns count of frozen decoder parameters.
    """
    encoder_params, decoder_params = classify_parameters(model, model_type)
    for _, param in decoder_params:
        param.requires_grad = False
    trainable = sum(1 for _, p in encoder_params if p.requires_grad)
    print(f"  Frozen: {len(decoder_params)} decoder params, "
          f"trainable: {trainable} encoder params")
    return len(decoder_params)


# ================================================================
# RESET ENCODER TO PRETRAINED WEIGHTS
# ================================================================

def reinit_encoder_to_pretrained(model, model_type, device='cuda',
                                  dinov3_model_name='dinov3_vits16',
                                  dinov3_weights_path=None):
    """
    Reset encoder weights to their pretrained (pre-fine-tuning) state.

    Strategy:
      - CNNs (MAMNet, OGLANet): instantiate the BASE model class with
        pretrained=True (loads ImageNet ResNet weights internally).
      - DINOv3: instantiate the base DINOv3ShadowDetector with the
        pretrained ViT weights file.

    Only parameters whose names exist in BOTH the fresh pretrained model
    and the current model are copied. This safely handles DA variants
    (FADA adapters, IIM kernels, etc.) whose variant-specific modules
    under the encoder prefix are left untouched.

    Also resets encoder BN running statistics to pretrained values.

    Returns:
        (n_params_reset, n_buffers_reset)
    """
    device = torch.device(device)
    prefixes = ENCODER_PREFIXES.get(model_type, ['encoder.', 'backbone.'])

    # Identify encoder parameter and buffer names in current model
    encoder_param_names = set()
    for name, _ in model.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            encoder_param_names.add(name)

    encoder_buffer_names = set()
    for name, _ in model.named_buffers():
        if any(name.startswith(p) for p in prefixes):
            encoder_buffer_names.add(name)

    # Create fresh pretrained model (BASE class only)
    _clean_model_paths()

    if model_type == 'mamnet':
        _add_path(SCRIPT_DIR)
        from mamnet.models.mamnet import MAMNet
        fresh = MAMNet(num_classes=2, pretrained=True,
                       use_aux=True, use_contrast=True)

    elif model_type == 'oglanet':
        _add_path(SCRIPT_DIR)
        from oglanet.models.oglanet import OGLANet
        fresh = OGLANet(num_classes=2, pretrained=True,
                        img_size=IMG_SIZE, use_contrast=True)

    elif model_type == 'dinov3':
        _add_path(os.path.join(SCRIPT_DIR, 'dinov3'))
        _add_path(DINOV3_PATH)
        from dinov3_model import DINOv3ShadowDetector
        fresh = DINOv3ShadowDetector(
            num_classes=2,
            model_name=dinov3_model_name,
            weights_path=dinov3_weights_path,
            pretrained=True,
            frozen_stages=-1,
        )
    else:
        raise ValueError(f"Unknown model_type for encoder reinit: {model_type}")

    fresh = fresh.to(device)
    fresh_sd = fresh.state_dict()

    # Copy encoder parameters and buffers from fresh → current
    current_sd = model.state_dict()
    n_params = 0
    n_buffers = 0

    for name in encoder_param_names:
        if name in fresh_sd:
            if current_sd[name].shape == fresh_sd[name].shape:
                current_sd[name] = fresh_sd[name].clone()
                n_params += 1
            else:
                print(f"  WARNING: shape mismatch for {name}: "
                      f"current={current_sd[name].shape}, "
                      f"fresh={fresh_sd[name].shape}. Skipping.")

    for name in encoder_buffer_names:
        if name in fresh_sd:
            if current_sd[name].shape == fresh_sd[name].shape:
                current_sd[name] = fresh_sd[name].clone()
                n_buffers += 1

    model.load_state_dict(current_sd)
    del fresh

    total_enc = len(encoder_param_names) + len(encoder_buffer_names)
    print(f"  Reset encoder to pretrained: "
          f"{n_params} params + {n_buffers} buffers "
          f"(of {total_enc} total encoder entries)")
    return n_params, n_buffers


# ================================================================
# ENCODER TRAINING LOOP
# ================================================================

def train_encoder(model, model_type, train_loader, device,
                  epochs=30, lr=None, weight_decay=1e-4, log_interval=20):
    """
    Train only encoder parameters (decoder is frozen).

    Decoder BN layers remain in eval mode (running stats frozen).
    Encoder BN layers are set to train mode (update running stats).

    Uses cross-entropy loss on the main output only.
    """
    if lr is None:
        lr = ENCODER_LR_DEFAULTS.get(model_type, 1e-4)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Training {len(trainable)} param groups, "
          f"{sum(p.numel() for p in trainable):,} total params, lr={lr:.1e}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Set training modes carefully:
    #   - Encoder: train mode (parameters + BN stats update)
    #   - Decoder: eval mode (parameters frozen + BN stats frozen)
    prefixes = ENCODER_PREFIXES.get(model_type, ['encoder.', 'backbone.'])

    model.train()
    # Force decoder modules to eval mode
    for name, module in model.named_modules():
        is_enc = any(name.startswith(p) for p in prefixes)
        if not is_enc:
            module.eval()

    best_loss = float('inf')

    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0

        # Re-assert modes each epoch (safety)
        model.train()
        for name, module in model.named_modules():
            is_enc = any(name.startswith(p) for p in prefixes)
            if not is_enc:
                module.eval()

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)

            # Resolve label key (same logic as Experiment A)
            if 'label' in batch:
                labels = batch['label'].to(device)
            elif 'mask' in batch:
                labels = batch['mask'].to(device)
            else:
                for k, v in batch.items():
                    if k not in ('image', 'filename') and isinstance(v, torch.Tensor):
                        if v.dtype in (torch.long, torch.int, torch.int32):
                            labels = v.to(device)
                            break
                        elif v.max() <= 1 and v.min() >= 0:
                            labels = v.long().to(device)
                            break
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
            logits = extract_logits(outputs)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:],
                    mode='bilinear', align_corners=False)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if (i + 1) % log_interval == 0:
                print(f"    Epoch {epoch+1}/{epochs}, "
                      f"Batch {i+1}, "
                      f"Loss: {running_loss/n_batches:.4f}")

        epoch_loss = running_loss / max(n_batches, 1)
        scheduler.step()
        best_loss = min(best_loss, epoch_loss)

        print(f"  Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, "
              f"best={best_loss:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

    return best_loss


# ================================================================
# MAIN EXPERIMENT
# ================================================================

def run_experiment_b2(args):
    """Main entry point for Experiment B2."""
    print("=" * 70)
    print("EXPERIMENT B2: Encoder Retraining on Holdout City")
    print("=" * 70)
    print(f"  Model          : {args.model_type}/{args.model_variant}")
    print(f"  Holdout city   : {args.holdout_city}")
    print(f"  Resolution     : {args.res}")
    print(f"  Train fraction : {args.train_fraction}")
    print(f"  Epochs         : {args.epochs}")
    lr_used = args.lr if args.lr else ENCODER_LR_DEFAULTS.get(args.model_type, 1e-4)
    print(f"  LR             : {lr_used:.1e} "
          f"({'user-specified' if args.lr else 'architecture default'})")
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

    # 2. Freeze decoder, reset encoder to pretrained
    print("\n--- Freezing decoder ---")
    freeze_decoder(model, args.model_type)

    print("\n--- Resetting encoder to pretrained weights ---")
    reinit_encoder_to_pretrained(
        model, args.model_type,
        device=args.device,
        dinov3_model_name=args.dinov3_model_name,
        dinov3_weights_path=args.dinov3_weights_path,
    )

    # 3. Load training data from holdout city
    print("\n--- Loading training data ---")
    full_dataset = get_dataset(
        args.model_type, args.train_data_root, split='train', augment=True)

    n_total = len(full_dataset)
    n_use = max(1, int(n_total * args.train_fraction))
    indices = np.linspace(0, n_total - 1, n_use, dtype=int).tolist()
    train_subset = Subset(full_dataset, indices)
    print(f"  Using {n_use}/{n_total} training images "
          f"({args.train_fraction*100:.0f}%)")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # 4. Train encoder
    print("\n--- Training encoder ---")
    train_encoder(model, args.model_type, train_loader, device,
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

    print("\n" + "=" * 70)
    print("EXPERIMENT B2 COMPLETE")
    print("=" * 70)


def get_args():
    p = argparse.ArgumentParser(
        description="Experiment B2: Encoder Retraining on Holdout City")
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
                                        'data', 'Test_img_results', 'experiment_b2'))
    p.add_argument('--train_fraction', type=float, default=0.25)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=None,
                   help='Encoder LR. If None, uses architecture-specific '
                        'default (1e-4 CNNs, 1e-5 DINOv3)')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--device', default='cuda')
    # DINOv3-specific
    p.add_argument('--dinov3_model_name', default='dinov3_vits16')
    p.add_argument('--dinov3_weights_path', default=None)
    p.add_argument('--dinov3_pretrained', action='store_true', default=True)
    p.add_argument('--dinov3_frozen_stages', type=int, default=-1)
    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    run_experiment_b2(args)