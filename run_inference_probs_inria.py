"""
Probability-saving inference for INRIA shape-control experiment.

Parallel to run_inference_probs.py, but:
  - Only supports MAMNet (only architecture trained on INRIA).
  - Expects checkpoints under data/mamnet_inria/outputs/ rather than
    data/mamnet/outputs/.
  - Saves probability of class 1 (= building, in the INRIA setup)
    as float16 .npy per image, identical format and layout to shadow
    so downstream analysis scripts can reuse their I/O helpers.

Output layout (mirrors shadow Test_img_probs/):
    {output_dir}/{test_type}/{city}/{res}/mamnet/{variant}/{stem}.npy

For INRIA this yields exactly 6 cells:
    upper/austin/highres/mamnet/base/
    upper/chicago/highres/mamnet/base/
    upper/vienna/highres/mamnet/base/
    loco/austin/highres/mamnet/vanilla/
    loco/chicago/highres/mamnet/vanilla/
    loco/vienna/highres/mamnet/vanilla/

We use variant="base" for upper-bounds and variant="vanilla" for LOCO
exactly like the shadow setup — this keeps the downstream affine-
decomposition scripts unchanged.

Usage (matches run_inference_probs.py argument surface):
    python run_inference_probs_inria.py \
        --test_type upper \
        --city austin \
        --train_res highres \
        --test_res highres \
        --model_variant base \
        --checkpoint_path /path/to/checkpoint_best.pth \
        --data_root /path/to/test/data \
        --output_dir ./Test_img_probs_inria
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
# MAMNet imports (INRIA uses only MAMNet; no need for OGLANet or DINOv3)
# ---------------------------------------------------------------------------
mamnet_path = os.path.join(os.path.dirname(__file__), 'mamnet')
if mamnet_path not in sys.path:
    sys.path.insert(0, mamnet_path)

from mamnet.models.mamnet import MAMNet
from mamnet.data.dataset_enhanced import ShadowDatasetEnhanced


def get_args():
    parser = argparse.ArgumentParser(
        description='Save per-pixel building-class probability maps '
                    'for INRIA MAMNet checkpoints')

    parser.add_argument('--model_variant', type=str, required=True,
                        choices=['base', 'vanilla'],
                        help='base for upper-bound, vanilla for LOCO — '
                             'paralleling the shadow naming convention')

    parser.add_argument('--test_type', type=str, required=True,
                        choices=['upper', 'loco'])
    parser.add_argument('--city', type=str, required=True,
                        choices=['austin', 'chicago', 'vienna'])
    parser.add_argument('--train_res', type=str, required=True,
                        choices=['highres'])
    parser.add_argument('--test_res', type=str, required=True,
                        choices=['highres'])

    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='./Test_img_probs_inria')

    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--positive_class_idx', type=int, default=1,
                        help='Index of building/positive class (default 1)')

    return parser.parse_args()


def load_model(args):
    """Load MAMNet checkpoint. Matches run_inference_probs.py semantics
    exactly — same constructor args, same load_state_dict mode."""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading MAMNet/{args.model_variant} "
          f"from {args.checkpoint_path}")

    # Match training-time constructor (use_contrast=True, use_aux=True)
    model = MAMNet(num_classes=2, pretrained=False,
                   use_aux=True, use_contrast=True).to(device)

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path,
                            map_location=device, weights_only=False)

    # strict=False matches run_inference_probs.py — tolerates minor
    # state-dict drift in auxiliary heads
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print("  Model loaded OK.")
    return model, device


def create_output_dir(args):
    """Output: {output_dir}/{test_type}/{city}/{test_res}/mamnet/{variant}/"""
    output_path = (Path(args.output_dir) / args.test_type / args.city
                   / args.test_res / 'mamnet' / args.model_variant)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _extract_logits(outputs):
    """MAMNet eval-mode output is a tensor. Training-mode output is a dict
    with 'main'. We set model.eval(), so the tensor path is the normal one,
    but we handle both for defensiveness."""
    if isinstance(outputs, dict):
        return outputs['main']
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _stem_to_npy(fname):
    return f"{Path(fname).stem}.npy"


def run_inference(model, dataloader, output_dir, device, args):
    print(f"\nProbability inference on INRIA...")
    print(f"  Output dir:         {output_dir}")
    print(f"  Positive class idx: {args.positive_class_idx}")

    n_saved = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch['image'].to(device)
            filenames = batch['filename']

            outputs = model(images)
            logits = _extract_logits(outputs)

            if (logits.shape[-2] != args.img_size
                    or logits.shape[-1] != args.img_size):
                logits = F.interpolate(
                    logits, size=(args.img_size, args.img_size),
                    mode='bilinear', align_corners=False)

            probs = F.softmax(logits, dim=1)
            pos_probs = probs[:, args.positive_class_idx]
            pos_probs_np = pos_probs.cpu().numpy().astype(np.float16)

            for i in range(pos_probs_np.shape[0]):
                out_path = output_dir / _stem_to_npy(filenames[i])
                np.save(out_path, pos_probs_np[i])
                n_saved += 1

    print(f"  Saved {n_saved} probability maps.")
    return n_saved


def main():
    args = get_args()

    if not os.path.exists(args.checkpoint_path):
        print(f"ERROR: Checkpoint not found: {args.checkpoint_path}")
        output_dir = create_output_dir(args)
        with open(output_dir / "MISSING_MODEL.txt", 'w') as f:
            f.write(f"Checkpoint missing: {args.checkpoint_path}\n")
        sys.exit(1)

    try:
        model, device = load_model(args)
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        output_dir = create_output_dir(args)
        with open(output_dir / "MODEL_LOAD_FAILED.txt", 'w') as f:
            f.write(f"Error: {e}\n")
        sys.exit(1)

    print(f"\nLoading INRIA test data from: {args.data_root}")

    # task_id=2 gives us the 4-channel RGBC input matching MAMNet training
    test_dataset = ShadowDatasetEnhanced(
        root_dir=[args.data_root], split='test',
        img_size=args.img_size, task_id=2,
        augment=False, geo_metadata_path=None)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print(f"Test samples: {len(test_dataset)}")

    output_dir = create_output_dir(args)
    n_saved = run_inference(model, test_loader, output_dir, device, args)

    config = {
        'dataset':            'INRIA (buildings)',
        'model_type':         'mamnet',
        'model_variant':      args.model_variant,
        'test_type':          args.test_type,
        'city':               args.city,
        'train_res':          args.train_res,
        'test_res':           args.test_res,
        'checkpoint_path':    args.checkpoint_path,
        'data_root':          args.data_root,
        'num_images':         len(test_dataset),
        'num_saved':          n_saved,
        'positive_class_idx': args.positive_class_idx,
        'img_size':           args.img_size,
        'output_format':      'float16_npy_per_image',
        'note': 'filter_small_predictions NOT applied; raw softmax probs.',
    }
    with open(output_dir / 'inference_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    print("\n  Done.")


if __name__ == '__main__':
    main()