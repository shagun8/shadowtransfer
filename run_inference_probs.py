"""
Probability-saving inference script for shadow detection models.

Parallel to run_inference.py, but instead of argmax'ing and saving binary
PNG masks, this script:
  - computes per-pixel shadow-class probability via softmax
  - saves each prediction as a float16 .npy file (H x W, values in [0, 1])

Rationale: confidence-distribution analysis (Test 4) requires the raw
continuous output of the model, not the thresholded binary mask.

Key differences from run_inference.py:
  1. We DO NOT call filter_small_predictions. That function suppresses
     low-confidence shadow components by spatial size, which would
     contaminate any downstream claim about "the model's confidence
     distribution." We want the raw softmax output.
  2. Output is a per-image float16 .npy file named {original_filename}.npy
     (e.g. patch_00042.png -> patch_00042.npy). Same directory layout as
     run_inference.py so downstream analysis scripts can find it.
  3. We also write a compact metadata JSON per cell so analysis knows
     what resolution / scheme was used.

Usage (identical to run_inference.py):
    python run_inference_probs.py \
        --model_type mamnet \
        --test_type upper \
        --city chicago \
        --train_res highres \
        --test_res highres \
        --model_variant base \
        --checkpoint_path /path/to/checkpoint_best.pth \
        --data_root /path/to/test/data \
        --output_dir ./Test_img_probs

model_variant choices (same as run_inference.py):
    base / vanilla / fda    — base arch (ISW uses this too — training-only)
    segdesic                — SegDesic module
    iim                     — Illumination-Invariant Module
    isw                     — Instance Selective Whitening (base arch)
    mrfp_plus               — MRFP+ (perturbation disabled in eval)
    fada                    — Frequency-Adapted Domain Adaptation
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# MAMNet imports
# ---------------------------------------------------------------------------
mamnet_path = os.path.join(os.path.dirname(__file__), 'mamnet')
if mamnet_path not in sys.path:
    sys.path.insert(0, mamnet_path)

from mamnet.models.mamnet import MAMNet
from mamnet.models.mamnet_segdesic import MAMNetSegDesic
from mamnet.models.mamnet_iim import MAMNetIIM
from mamnet.models.mamnet_mrfp import MAMNetMRFP
from mamnet.models.mamnet_fada import MAMNetFADA
from mamnet.data.dataset_enhanced import ShadowDatasetEnhanced

sys.path.remove(mamnet_path)

# ---------------------------------------------------------------------------
# OGLANet imports
# ---------------------------------------------------------------------------
oglanet_path = os.path.join(os.path.dirname(__file__), 'oglanet')
if oglanet_path not in sys.path:
    sys.path.insert(0, oglanet_path)

from oglanet.models.oglanet import OGLANet
from oglanet.models.oglanet_segdesic import OGLANetSegDesic
from oglanet.models.oglanet_iim import OGLANetIIM
from oglanet.models.oglanet_mrfp import OGLANetMRFP
from oglanet.models.oglanet_fada import OGLANetFADA

sys.path.remove(oglanet_path)

# ---------------------------------------------------------------------------
# DINOv3 imports
# ---------------------------------------------------------------------------
dinov3_pkg_path = os.path.join(os.path.dirname(__file__), 'dinov3')
if dinov3_pkg_path not in sys.path:
    sys.path.insert(0, dinov3_pkg_path)

from dinov3_model import DINOv3ShadowDetector
from dinov3_segdesic import DINOv3SegDesic
from dinov3_iim_model import DINOv3ShadowDetectorIIM
from dinov3_model_mrfp import DINOv3ShadowDetectorMRFP
from dinov3_model_fada import DINOv3FADAShadowDetector


def get_args():
    parser = argparse.ArgumentParser(
        description='Run inference and save per-pixel shadow probabilities')

    # Model identification
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['mamnet', 'oglanet', 'dinov3'])
    parser.add_argument('--model_variant', type=str, required=True,
                        choices=[
                            'base', 'vanilla', 'fda',
                            'segdesic',
                            'iim', 'isw', 'mrfp_plus', 'fada',
                        ])

    # Test configuration
    parser.add_argument('--test_type', type=str, required=True,
                        choices=['upper', 'loco', 'cross-res'])
    parser.add_argument('--city', type=str, required=True)
    parser.add_argument('--train_res', type=str, required=True,
                        choices=['highres', 'midres'])
    parser.add_argument('--test_res', type=str, required=True,
                        choices=['highres', 'midres'])

    # Paths
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./Test_img_probs',
                        help='Base output directory for probability arrays')

    # Model-specific parameters
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)

    # DINOv3-specific
    parser.add_argument('--dinov3_model_name', type=str,
                        default='dinov3_vits16',
                        choices=['dinov3_vits16', 'dinov3_vitb16',
                                 'dinov3_vitl16'])
    parser.add_argument('--dinov3_weights_path', type=str, default=None)
    parser.add_argument('--dinov3_pretrained', action='store_true',
                        default=True)
    parser.add_argument('--dinov3_frozen_stages', type=int, default=-1)

    # Device
    parser.add_argument('--device', type=str, default='cuda')

    # Shadow class index — for 2-class networks this is class 1 (background=0)
    parser.add_argument('--shadow_class_idx', type=int, default=1,
                        help='Index of the shadow class in the output logits')

    return parser.parse_args()


def load_model(args):
    """Same as run_inference.py. Kept verbatim to guarantee identical
    checkpoint loading semantics — if you change the original, sync here."""
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading {args.model_type}/{args.model_variant} "
          f"from {args.checkpoint_path}")

    if args.model_type == 'mamnet':
        if args.model_variant in ['base', 'vanilla', 'fda', 'isw']:
            model = MAMNet(num_classes=2, pretrained=False,
                           use_aux=True, use_contrast=True)
        elif args.model_variant == 'segdesic':
            model = MAMNetSegDesic(
                num_classes=2, pretrained=False, use_aux=True,
                segdesic_hidden_dim=256, segdesic_num_scales=10,
                use_contrast=True)
        elif args.model_variant == 'iim':
            model = MAMNetIIM(
                num_classes=2, pretrained=False, use_aux=True,
                use_contrast=True, num_kernels=8, kernel_size=5)
        elif args.model_variant == 'mrfp_plus':
            model = MAMNetMRFP(
                num_classes=2, pretrained=False, use_aux=True,
                use_contrast=True, use_mrfp_plus=True)
        elif args.model_variant == 'fada':
            model = MAMNetFADA(
                num_classes=2, pretrained=False, use_aux=True,
                use_contrast=True, fada_rank=16, fada_token_length=100,
                fada_stages=(3, 4, 5))

    elif args.model_type == 'oglanet':
        if args.model_variant in ['base', 'vanilla', 'fda', 'isw']:
            model = OGLANet(num_classes=2, pretrained=False,
                            img_size=args.img_size, use_contrast=True)
        elif args.model_variant == 'segdesic':
            model = OGLANetSegDesic(
                num_classes=2, pretrained=False, img_size=args.img_size,
                segdesic_hidden_dim=256, segdesic_num_scales=10,
                use_contrast=True)
        elif args.model_variant == 'iim':
            model = OGLANetIIM(
                num_classes=2, pretrained=False, img_size=args.img_size,
                use_contrast=True, num_kernels=8, kernel_size=5)
        elif args.model_variant == 'mrfp_plus':
            model = OGLANetMRFP(
                num_classes=2, pretrained=False, img_size=args.img_size,
                use_contrast=True, use_mrfp_plus=True)
        elif args.model_variant == 'fada':
            model = OGLANetFADA(
                num_classes=2, pretrained=False, img_size=args.img_size,
                use_contrast=True, fada_rank=16, fada_token_length=100,
                fada_stages=(3, 4, 5))

    elif args.model_type == 'dinov3':
        if args.model_variant in ['base', 'vanilla', 'fda', 'isw']:
            model = DINOv3ShadowDetector(
                num_classes=2, model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path, pretrained=True,
                frozen_stages=args.dinov3_frozen_stages)
        elif args.model_variant == 'segdesic':
            model = DINOv3SegDesic(
                num_classes=2, model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path, pretrained=True,
                frozen_stages=args.dinov3_frozen_stages,
                segdesic_hidden_dim=256, segdesic_num_scales=10)
        elif args.model_variant == 'iim':
            model = DINOv3ShadowDetectorIIM(
                num_classes=2, model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path, pretrained=True,
                frozen_stages=args.dinov3_frozen_stages,
                num_kernels=8, kernel_size=5)
        elif args.model_variant == 'mrfp_plus':
            model = DINOv3ShadowDetectorMRFP(
                num_classes=2, model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path, pretrained=True,
                frozen_stages=args.dinov3_frozen_stages, use_mrfp_plus=True)
        elif args.model_variant == 'fada':
            model = DINOv3FADAShadowDetector(
                num_classes=2, model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path, pretrained=True,
                fada_rank=16, fada_token_length=100,
                fada_stages=(3, 6, 9, 11))

    model = model.to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint_path}")

    checkpoint = torch.load(
        args.checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()
    print(f"✓ Model loaded successfully")
    return model, device


def create_output_dir(args):
    """Same directory scheme as run_inference.py, but under a
    probability-specific root so binary masks and probability maps
    can coexist."""
    if args.test_type == 'cross-res':
        res_str = f"{args.train_res}_to_{args.test_res}"
    else:
        res_str = args.test_res

    output_path = (Path(args.output_dir) / args.test_type / args.city
                   / res_str / args.model_type / args.model_variant)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _extract_logits(outputs, args):
    """Same as run_inference.py — normalise variable output types into
    a logits tensor [B, C, H, W]."""
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    if isinstance(outputs, dict):
        if 'main' in outputs:
            return outputs['main']
        elif 'p6' in outputs:
            return outputs['p6']
        elif 'pred_fused' in outputs:
            return outputs['pred_fused']
        else:
            return list(outputs.values())[0]

    return outputs


def _stem_to_npy(fname):
    """patch_00042.png  ->  patch_00042.npy"""
    stem = Path(fname).stem
    return f"{stem}.npy"


def run_inference_probs(model, dataloader, output_dir, device, args):
    """Run inference and save per-pixel shadow probability maps as .npy."""
    print(f"\nRunning probability inference...")
    print(f"Output directory: {output_dir}")
    print(f"Shadow class index: {args.shadow_class_idx}")

    n_saved = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images    = batch['image'].to(device)
            filenames = batch['filename']

            # ----- forward pass (identical branching to run_inference.py) -----
            outputs = model(images)

            logits = _extract_logits(outputs, args)

            # If output is at a different spatial size than input (e.g. some
            # ViT decoders), upsample to input resolution with bilinear. This
            # matches what argmax + save would do after filter_small_predictions
            # in run_inference.py for spatial alignment with the GT masks.
            if (logits.shape[-2] != args.img_size
                    or logits.shape[-1] != args.img_size):
                logits = F.interpolate(
                    logits, size=(args.img_size, args.img_size),
                    mode='bilinear', align_corners=False)

            # Softmax over class dim; pull shadow-class channel
            probs = F.softmax(logits, dim=1)
            shadow_probs = probs[:, args.shadow_class_idx]   # [B, H, W]

            # Move to CPU, cast to float16 for disk savings
            shadow_probs_np = shadow_probs.cpu().numpy().astype(np.float16)

            for i in range(shadow_probs_np.shape[0]):
                out_path = output_dir / _stem_to_npy(filenames[i])
                np.save(out_path, shadow_probs_np[i])
                n_saved += 1

    print(f"✓ Saved {n_saved} probability maps.")
    return n_saved


def main():
    args = get_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found: {args.checkpoint_path}")
        output_dir = create_output_dir(args)
        with open(output_dir / "MISSING_MODEL.txt", 'w') as f:
            f.write(f"Model checkpoint not found\n"
                    f"Expected: {args.checkpoint_path}\n"
                    f"Model: {args.model_type}/{args.model_variant}\n"
                    f"Test: {args.test_type}  City: {args.city}\n"
                    f"Train res: {args.train_res}  Test res: {args.test_res}\n")
        sys.exit(1)

    try:
        model, device = load_model(args)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        output_dir = create_output_dir(args)
        with open(output_dir / "MODEL_LOAD_FAILED.txt", 'w') as f:
            f.write(f"Model loading failed\nError: {e}\n"
                    f"Checkpoint: {args.checkpoint_path}\n")
        sys.exit(1)

    print(f"\nLoading test data from: {args.data_root}")

    if args.model_type in ['mamnet', 'oglanet']:
        test_dataset = ShadowDatasetEnhanced(
            root_dir=[args.data_root], split='test',
            img_size=args.img_size, task_id=2,
            augment=False, geo_metadata_path=None)
    else:
        if mamnet_path not in sys.path:
            sys.path.insert(0, mamnet_path)
        from mamnet.data.dataset import ShadowDataset
        sys.path.remove(mamnet_path)

        test_dataset = ShadowDataset(
            root_dir=args.data_root, split='test',
            img_size=args.img_size, augment=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    print(f"Test samples: {len(test_dataset)}")

    output_dir = create_output_dir(args)
    n_saved = run_inference_probs(model, test_loader, output_dir,
                                  device, args)

    config = {
        'model_type':     args.model_type,
        'model_variant':  args.model_variant,
        'test_type':      args.test_type,
        'city':           args.city,
        'train_res':      args.train_res,
        'test_res':       args.test_res,
        'checkpoint_path': args.checkpoint_path,
        'data_root':      args.data_root,
        'num_images':     len(test_dataset),
        'num_saved':      n_saved,
        'shadow_class_idx': args.shadow_class_idx,
        'img_size':       args.img_size,
        'output_format':  'float16_npy_per_image',
        'note': ('filter_small_predictions was NOT applied; these are raw '
                 'softmax probabilities of the shadow class'),
    }
    with open(output_dir / 'inference_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n✓ All done!")


if __name__ == '__main__':
    main()