"""
Inference script for shadow detection models.
Loads trained checkpoints and saves predictions for evaluation.

Usage:
    python run_inference.py \
        --model_type mamnet \
        --test_type upper \
        --city chicago \
        --train_res highres \
        --test_res highres \
        --model_variant base \
        --checkpoint_path /path/to/checkpoint_best.pth \
        --data_root /path/to/test/data \
        --output_dir ./Test_img_results

model_variant choices (mcl removed; iim, isw, mrfp_plus, fada added):
    base / vanilla / fda  — base architecture (ISW is also served here:
                            ISW regularisation is training-only and leaves
                            no trace in the model weights)
    segdesic              — SegDesic geographic adaptation module
    iim                   — Illumination-Invariant Module
    isw                   — Instance Selective Whitening  (same arch as base)
    mrfp_plus             — Multi-Resolution Feature Perturbation+
    fada                  — Frequency-Adapted Domain Adaptation
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
from PIL import Image
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
from mamnet.utils.postprocessing import filter_small_predictions

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
        description='Run inference on shadow detection models')

    # Model identification
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['mamnet', 'oglanet', 'dinov3'],
                        help='Model architecture')
    parser.add_argument('--model_variant', type=str, required=True,
                        choices=[
                            'base', 'vanilla', 'fda',
                            'segdesic',
                            'iim', 'isw', 'mrfp_plus', 'fada',
                        ],
                        help='Model variant')

    # Test configuration
    parser.add_argument('--test_type', type=str, required=True,
                        choices=['upper', 'loco', 'cross-res'],
                        help='Type of evaluation')
    parser.add_argument('--city', type=str, required=True,
                        help='City name (test city for LOCO, city for others)')
    parser.add_argument('--train_res', type=str, required=True,
                        choices=['highres', 'midres'],
                        help='Training resolution')
    parser.add_argument('--test_res', type=str, required=True,
                        choices=['highres', 'midres'],
                        help='Test resolution')

    # Paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of test data')
    parser.add_argument('--output_dir', type=str, default='./Test_img_results',
                        help='Base output directory')

    # Model-specific parameters
    parser.add_argument('--img_size', type=int, default=384,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

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
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def load_model(args):
    """
    Instantiate and load a model from checkpoint.

    ISW note: ISW regularisation is training-only (hooks + auxiliary loss).
    The saved weights are identical in shape to the base model, so 'isw'
    is handled by the same branch as 'base'/'vanilla'/'fda'.

    MRFP note: MAMNetMRFP / OGLANetMRFP / DINOv3ShadowDetectorMRFP disable
    their perturbation modules in eval mode, so inference is equivalent to the
    base model — but we still instantiate the correct class to guarantee the
    state-dict keys match.

    FADA note: FADA adapters ARE active at inference time; the correct class
    must be used.

    Default hyperparameters used for all LOCO runs (confirmed by user):
        IIM  : num_kernels=8, kernel_size=5
        MRFP : use_mrfp_plus=True (perturbation probs irrelevant at inference)
        FADA : fada_rank=16, fada_token_length=100
               MAMNet/OGLANet stages=(3,4,5)  |  DINOv3 stages=(3,6,9,11)
    """
    device = torch.device(
        args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading {args.model_type}/{args.model_variant} "
          f"from {args.checkpoint_path}")

    # ------------------------------------------------------------------
    # MAMNet
    # ------------------------------------------------------------------
    if args.model_type == 'mamnet':

        if args.model_variant in ['base', 'vanilla', 'fda', 'isw']:
            # isw: same base architecture; ISW is training-only
            model = MAMNet(
                num_classes=2, pretrained=False,
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
            # Perturbation probs are irrelevant in eval mode
            model = MAMNetMRFP(
                num_classes=2, pretrained=False, use_aux=True,
                use_contrast=True, use_mrfp_plus=True)

        elif args.model_variant == 'fada':
            model = MAMNetFADA(
                num_classes=2, pretrained=False, use_aux=True,
                use_contrast=True, fada_rank=16, fada_token_length=100,
                fada_stages=(3, 4, 5))

    # ------------------------------------------------------------------
    # OGLANet
    # ------------------------------------------------------------------
    elif args.model_type == 'oglanet':

        if args.model_variant in ['base', 'vanilla', 'fda', 'isw']:
            model = OGLANet(
                num_classes=2, pretrained=False,
                img_size=args.img_size, use_contrast=True)

        elif args.model_variant == 'segdesic':
            model = OGLANetSegDesic(
                num_classes=2, pretrained=False,
                img_size=args.img_size,
                segdesic_hidden_dim=256, segdesic_num_scales=10,
                use_contrast=True)

        elif args.model_variant == 'iim':
            model = OGLANetIIM(
                num_classes=2, pretrained=False,
                img_size=args.img_size, use_contrast=True,
                num_kernels=8, kernel_size=5)

        elif args.model_variant == 'mrfp_plus':
            model = OGLANetMRFP(
                num_classes=2, pretrained=False,
                img_size=args.img_size, use_contrast=True,
                use_mrfp_plus=True)

        elif args.model_variant == 'fada':
            model = OGLANetFADA(
                num_classes=2, pretrained=False,
                img_size=args.img_size, use_contrast=True,
                fada_rank=16, fada_token_length=100,
                fada_stages=(3, 4, 5))

    # ------------------------------------------------------------------
    # DINOv3
    # ------------------------------------------------------------------
    elif args.model_type == 'dinov3':

        if args.model_variant in ['base', 'vanilla', 'fda', 'isw']:
            model = DINOv3ShadowDetector(
                num_classes=2,
                model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path,
                pretrained=True,
                frozen_stages=args.dinov3_frozen_stages)

        elif args.model_variant == 'segdesic':
            model = DINOv3SegDesic(
                num_classes=2,
                model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path,
                pretrained=True,
                frozen_stages=args.dinov3_frozen_stages,
                segdesic_hidden_dim=256, segdesic_num_scales=10)

        elif args.model_variant == 'iim':
            model = DINOv3ShadowDetectorIIM(
                num_classes=2,
                model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path,
                pretrained=True,
                frozen_stages=args.dinov3_frozen_stages,
                num_kernels=8, kernel_size=5)

        elif args.model_variant == 'mrfp_plus':
            model = DINOv3ShadowDetectorMRFP(
                num_classes=2,
                model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path,
                pretrained=True,
                frozen_stages=args.dinov3_frozen_stages,
                use_mrfp_plus=True)

        elif args.model_variant == 'fada':
            # DINOv3 FADA uses ViT block indices, not ResNet stage numbers
            model = DINOv3FADAShadowDetector(
                num_classes=2,
                model_name=args.dinov3_model_name,
                weights_path=args.dinov3_weights_path,
                pretrained=True,
                fada_rank=16, fada_token_length=100,
                fada_stages=(3, 6, 9, 11))

    model = model.to(device)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
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
    """Create the output directory following the standard naming scheme."""
    if args.test_type == 'cross-res':
        res_str = f"{args.train_res}_to_{args.test_res}"
    else:
        res_str = args.test_res

    output_path = (Path(args.output_dir) / args.test_type / args.city
                   / res_str / args.model_type / args.model_variant)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _extract_logits(outputs, args):
    """
    Normalise the many output formats produced by different model variants
    into a single logits tensor [B, C, H, W].

    All new variants (iim, isw, mrfp_plus, fada) return a plain tensor in
    eval mode, so they fall straight through to the final else branch.
    """
    # MCL models historically returned (logits, features)
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    if isinstance(outputs, dict):
        if 'main' in outputs:
            return outputs['main']
        elif 'p6' in outputs:          # OGLANet family
            return outputs['p6']
        elif 'pred_fused' in outputs:  # HRDA
            return outputs['pred_fused']
        else:
            return list(outputs.values())[0]

    return outputs   # plain tensor — iim, isw, mrfp_plus, fada, base, etc.


def run_inference(model, dataloader, output_dir, device, args):
    """Run inference and write binary PNG masks to output_dir."""
    print(f"\nRunning inference...")
    print(f"Output directory: {output_dir}")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images    = batch['image'].to(device)
            filenames = batch['filename']

            # ----- forward pass -----
            outputs = model(images)

            logits   = _extract_logits(outputs, args)
            filtered = filter_small_predictions(logits, min_pixels=10)
            preds    = torch.argmax(filtered, dim=1).cpu().numpy()

            for i in range(preds.shape[0]):
                pred_mask = (preds[i].astype(np.uint8)) * 255
                out_path  = output_dir / filenames[i]
                Image.fromarray(pred_mask, mode='L').save(out_path)

    print(f"✓ Inference complete. Saved {len(dataloader.dataset)} predictions.")


def main():
    args = get_args()

    # ------------------------------------------------------------------
    # Early exit if checkpoint is missing
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    try:
        model, device = load_model(args)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        output_dir = create_output_dir(args)
        with open(output_dir / "MODEL_LOAD_FAILED.txt", 'w') as f:
            f.write(f"Model loading failed\nError: {e}\n"
                    f"Checkpoint: {args.checkpoint_path}\n")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"\nLoading test data from: {args.data_root}")

    # MAMNet and OGLANet were all trained with use_contrast=True (4-ch input).
    # DINOv3 variants use standard 3-channel RGB (no contrast channel).
    if args.model_type in ['mamnet', 'oglanet']:
        test_dataset = ShadowDatasetEnhanced(
            root_dir=[args.data_root],
            split='test',
            img_size=args.img_size,
            task_id=2,          # 4-channel: RGB + contrast
            augment=False,
            geo_metadata_path=None)
    else:
        # Re-add mamnet path temporarily for the ShadowDataset import
        if mamnet_path not in sys.path:
            sys.path.insert(0, mamnet_path)
        from mamnet.data.dataset import ShadowDataset
        sys.path.remove(mamnet_path)

        test_dataset = ShadowDataset(
            root_dir=args.data_root,
            split='test',
            img_size=args.img_size,
            augment=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    print(f"Test samples: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    output_dir = create_output_dir(args)
    run_inference(model, test_loader, output_dir, device, args)

    # Save config alongside predictions
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
    }
    with open(output_dir / 'inference_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print(f"\n✓ All done!")


if __name__ == '__main__':
    main()