"""
Shared utilities for Experiments A, B, C.

Handles:
  - Model loading for all variants (mcl removed; iim, isw, mrfp_plus, fada added)
  - Encoder / decoder parameter separation
  - BN layer discovery and statistics helpers
  - Inference + prediction saving
  - Histogram matching
  - Dataset loading (mamnet dataset classes used for all model types)

ISW note: ISW regularisation is training-only. Saved weights are identical
in shape to the base model, so 'isw' is served by the base branch.

MRFP note: MAMNetMRFP / OGLANetMRFP / DINOv3ShadowDetectorMRFP disable
their perturbation modules in eval mode — instantiate the correct class
to guarantee state-dict key matches.

FADA note: FADA adapters ARE active at inference; correct class required.
DINOv3FADAShadowDetector does not accept frozen_stages — omitted.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

IMG_SIZE = 384

SCRIPT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAMNET_PATH  = os.path.join(SCRIPT_DIR, 'mamnet')
OGLANET_PATH = os.path.join(SCRIPT_DIR, 'oglanet')
DINOV3_PATH  = os.path.join(SCRIPT_DIR, 'dinov3', 'dinov3')


# ================================================================
# INTERNAL HELPERS
# ================================================================

def _add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)

def _remove_path(p):
    if p in sys.path:
        sys.path.remove(p)

def _clean_model_paths():
    for p in [MAMNET_PATH, OGLANET_PATH, DINOV3_PATH,
              os.path.join(SCRIPT_DIR, 'dinov3')]:
        _remove_path(p)


# ================================================================
# MODEL LOADING
# ================================================================

def load_model(model_type, model_variant, checkpoint_path, device='cuda',
               dinov3_model_name='dinov3_vits16', dinov3_weights_path=None,
               dinov3_pretrained=True, dinov3_frozen_stages=-1):
    """
    Load model and checkpoint.  Returns (model, device).

    Default hyperparameters (confirmed consistent with training):
        IIM  : num_kernels=8, kernel_size=5
        MRFP : use_mrfp_plus=True  (perturbation disabled in eval)
        FADA : fada_rank=16, fada_token_length=100
               MAMNet/OGLANet stages=(3,4,5)  |  DINOv3 stages=(3,6,9,11)

    DINOv3 FADA note: DINOv3FADAShadowDetector does not accept frozen_stages;
    a separate kwarg dict is used for that variant.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    _clean_model_paths()

    # ------------------------------------------------------------------
    # MAMNet
    # ------------------------------------------------------------------
    if model_type == 'mamnet':
        _add_path(SCRIPT_DIR)
        from mamnet.models.mamnet          import MAMNet
        from mamnet.models.mamnet_segdesic import MAMNetSegDesic
        from mamnet.models.mamnet_iim      import MAMNetIIM
        from mamnet.models.mamnet_mrfp     import MAMNetMRFP
        from mamnet.models.mamnet_fada     import MAMNetFADA

        if model_variant in ('base', 'vanilla', 'fda', 'isw'):
            model = MAMNet(num_classes=2, pretrained=False,
                           use_aux=True, use_contrast=True)
        elif model_variant == 'segdesic':
            model = MAMNetSegDesic(num_classes=2, pretrained=False,
                                   use_aux=True, use_contrast=True,
                                   segdesic_hidden_dim=256,
                                   segdesic_num_scales=10)
        elif model_variant == 'iim':
            model = MAMNetIIM(num_classes=2, pretrained=False,
                              use_aux=True, use_contrast=True,
                              num_kernels=8, kernel_size=5)
        elif model_variant == 'mrfp_plus':
            model = MAMNetMRFP(num_classes=2, pretrained=False,
                               use_aux=True, use_contrast=True,
                               use_mrfp_plus=True)
        elif model_variant == 'fada':
            model = MAMNetFADA(num_classes=2, pretrained=False,
                               use_aux=True, use_contrast=True,
                               fada_rank=16, fada_token_length=100,
                               fada_stages=(3, 4, 5))
        else:
            raise ValueError(f"Unknown MAMNet variant: {model_variant}")

    # ------------------------------------------------------------------
    # OGLANet
    # ------------------------------------------------------------------
    elif model_type == 'oglanet':
        _add_path(SCRIPT_DIR)
        from oglanet.models.oglanet          import OGLANet
        from oglanet.models.oglanet_segdesic import OGLANetSegDesic
        from oglanet.models.oglanet_iim      import OGLANetIIM
        from oglanet.models.oglanet_mrfp     import OGLANetMRFP
        from oglanet.models.oglanet_fada     import OGLANetFADA

        if model_variant in ('base', 'vanilla', 'fda', 'isw'):
            model = OGLANet(num_classes=2, pretrained=False,
                            img_size=IMG_SIZE, use_contrast=True)
        elif model_variant == 'segdesic':
            model = OGLANetSegDesic(num_classes=2, pretrained=False,
                                    img_size=IMG_SIZE, use_contrast=True,
                                    segdesic_hidden_dim=256,
                                    segdesic_num_scales=10)
        elif model_variant == 'iim':
            model = OGLANetIIM(num_classes=2, pretrained=False,
                               img_size=IMG_SIZE, use_contrast=True,
                               num_kernels=8, kernel_size=5)
        elif model_variant == 'mrfp_plus':
            model = OGLANetMRFP(num_classes=2, pretrained=False,
                                img_size=IMG_SIZE, use_contrast=True,
                                use_mrfp_plus=True)
        elif model_variant == 'fada':
            model = OGLANetFADA(num_classes=2, pretrained=False,
                                img_size=IMG_SIZE, use_contrast=True,
                                fada_rank=16, fada_token_length=100,
                                fada_stages=(3, 4, 5))
        else:
            raise ValueError(f"Unknown OGLANet variant: {model_variant}")

    # ------------------------------------------------------------------
    # DINOv3
    # ------------------------------------------------------------------
    elif model_type == 'dinov3':
        _add_path(os.path.join(SCRIPT_DIR, 'dinov3'))
        _add_path(DINOV3_PATH)
        from dinov3_model      import DINOv3ShadowDetector
        from dinov3_segdesic   import DINOv3SegDesic
        from dinov3_iim_model  import DINOv3ShadowDetectorIIM
        from dinov3_model_mrfp import DINOv3ShadowDetectorMRFP
        from dinov3_model_fada import DINOv3FADAShadowDetector

        # Base kwargs shared by all DINOv3 variants that accept frozen_stages
        kw = dict(
            num_classes=2,
            model_name=dinov3_model_name,
            weights_path=dinov3_weights_path,
            pretrained=dinov3_pretrained,
            frozen_stages=dinov3_frozen_stages,
        )
        # FADA-specific kwargs: DINOv3FADAShadowDetector does NOT accept
        # frozen_stages (confirmed from run_inference.py instantiation)
        kw_fada = dict(
            num_classes=2,
            model_name=dinov3_model_name,
            weights_path=dinov3_weights_path,
            pretrained=dinov3_pretrained,
        )

        if model_variant in ('base', 'vanilla', 'fda', 'isw'):
            model = DINOv3ShadowDetector(**kw)
        elif model_variant == 'segdesic':
            model = DINOv3SegDesic(**kw,
                                   segdesic_hidden_dim=256,
                                   segdesic_num_scales=10)
        elif model_variant == 'iim':
            model = DINOv3ShadowDetectorIIM(**kw,
                                            num_kernels=8,
                                            kernel_size=5)
        elif model_variant == 'mrfp_plus':
            model = DINOv3ShadowDetectorMRFP(**kw, use_mrfp_plus=True)
        elif model_variant == 'fada':
            # ViT block indices, not ResNet stage numbers
            model = DINOv3FADAShadowDetector(**kw_fada,
                                             fada_rank=16,
                                             fada_token_length=100,
                                             fada_stages=(3, 6, 9, 11))
        else:
            raise ValueError(f"Unknown DINOv3 variant: {model_variant}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    print(f"  Model loaded: {model_type}/{model_variant} from {checkpoint_path}")
    return model, device


# ================================================================
# ENCODER / DECODER PARAMETER SEPARATION
# ================================================================

ENCODER_PREFIXES = {
    'mamnet':  ['encoder.resnet_encoder.', 'encoder.resnet.', 'encoder.'],
    'oglanet': ['encoder.resnet_encoder.', 'encoder.resnet.', 'encoder.'],
    'dinov3':  ['backbone.dinov3.', 'backbone.'],
}


def classify_parameters(model, model_type):
    """
    Separate model parameters into encoder and decoder groups.
    Returns (encoder_params, decoder_params) as lists of (name, param) tuples.
    """
    prefixes = ENCODER_PREFIXES.get(model_type, ['encoder.', 'backbone.'])
    encoder_params, decoder_params = [], []

    for name, param in model.named_parameters():
        is_enc = any(name.startswith(p) for p in prefixes)
        (encoder_params if is_enc else decoder_params).append((name, param))

    print(f"  Parameters — encoder: {len(encoder_params)}, "
          f"decoder: {len(decoder_params)}")
    return encoder_params, decoder_params


def freeze_encoder(model, model_type):
    """Freeze all encoder parameters. Returns count frozen."""
    encoder_params, decoder_params = classify_parameters(model, model_type)
    for _, param in encoder_params:
        param.requires_grad = False
    trainable = sum(1 for _, p in decoder_params if p.requires_grad)
    print(f"  Frozen: {len(encoder_params)} encoder params, "
          f"trainable: {trainable} decoder params")
    return len(encoder_params)


def reinit_decoder(model, model_type):
    """Re-initialize decoder parameters with standard init."""
    _, decoder_params = classify_parameters(model, model_type)
    decoder_names = {name for name, _ in decoder_params}

    reinit_count = 0
    for name, module in model.named_modules():
        module_params = {n for n, _ in module.named_parameters(prefix=name)}
        if not module_params.intersection(decoder_names):
            continue
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            reinit_count += 1
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            reinit_count += 1
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            reinit_count += 1

    print(f"  Re-initialized {reinit_count} decoder modules")
    return reinit_count


# ================================================================
# BATCH NORMALIZATION HELPERS
# ================================================================

def find_bn_layers(model):
    """Return list of (name, module) for all BatchNorm layers
    that have running stats (track_running_stats=True)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
            and m.running_mean is not None]


def save_bn_stats(model):
    """Save current BN running stats for layers that track them.
    Layers with track_running_stats=False are silently skipped."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.running_mean is None:
                continue   # track_running_stats=False — nothing to save
            stats[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var':  module.running_var.clone(),
                'num_batches_tracked': (module.num_batches_tracked.clone()
                                        if module.num_batches_tracked is not None
                                        else None),
            }
    return stats


def restore_bn_stats(model, stats):
    """Restore previously saved BN running stats.
    Layers absent from stats dict (e.g. track_running_stats=False) are skipped."""
    for name, module in model.named_modules():
        if name not in stats:
            continue
        if module.running_mean is None:
            continue
        module.running_mean.copy_(stats[name]['running_mean'])
        module.running_var.copy_(stats[name]['running_var'])
        if stats[name]['num_batches_tracked'] is not None:
            module.num_batches_tracked.copy_(
                stats[name]['num_batches_tracked'])


def collect_bn_stats(model, dataloader, device, n_batches=None):
    """
    Forward-pass the dataset with BN in train mode to collect
    fresh running statistics. All other modules stay in eval mode.
    Layers with track_running_stats=False are skipped throughout.

    Returns dict of new BN stats; original stats are restored on exit.
    """
    original_stats = save_bn_stats(model)

    for _, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.running_mean is None:
                continue   # track_running_stats=False — skip
            module.running_mean.zero_()
            module.running_var.fill_(1.0)
            if module.num_batches_tracked is not None:
                module.num_batches_tracked.zero_()
            module.momentum = None  # cumulative moving average

    model.eval()
    for _, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting BN stats")):
            if n_batches is not None and i >= n_batches:
                break
            _ = model(batch['image'].to(device))

    new_stats = save_bn_stats(model)
    restore_bn_stats(model, original_stats)

    model.eval()
    for _, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.momentum = 0.1

    return new_stats


def apply_bn_stats_selective(model, new_stats, layer_names):
    """Apply new BN stats only to the specified layer names.
    Layers with track_running_stats=False are skipped."""
    for name, module in model.named_modules():
        if name not in layer_names or name not in new_stats:
            continue
        if module.running_mean is None:
            continue   # track_running_stats=False — skip
        module.running_mean.copy_(new_stats[name]['running_mean'])
        module.running_var.copy_(new_stats[name]['running_var'])
        if new_stats[name]['num_batches_tracked'] is not None:
            module.num_batches_tracked.copy_(
                new_stats[name]['num_batches_tracked'])


def get_bn_layer_depth(model):
    """
    Categorize BN layers into early / mid / late thirds.
    Returns dict: {'early': [...names], 'mid': [...], 'late': [...]}.
    """
    bn_names = [n for n, m in model.named_modules()
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    if not bn_names:
        return {'early': [], 'mid': [], 'late': []}
    n     = len(bn_names)
    third = max(1, n // 3)
    return {
        'early': bn_names[:third],
        'mid':   bn_names[third:2*third],
        'late':  bn_names[2*third:],
    }


# ================================================================
# INFERENCE + PREDICTION SAVING
# ================================================================

def extract_logits(outputs):
    """Normalise diverse output formats to a single logits tensor [B,C,H,W]."""
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, dict):
        return outputs.get('main',
               outputs.get('p6',
               outputs.get('pred_fused',
               list(outputs.values())[0])))
    return outputs  # plain tensor (base, iim, isw, mrfp_plus, fada)


def run_inference(model, dataloader, device, output_dir):
    """
    Run inference and save binary prediction masks (0 or 255).
    One PNG per image, filename preserved from the batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images = batch['image'].to(device)
            fnames = batch['filename']

            outputs = model(images)
            logits  = extract_logits(outputs)

            if logits.shape[-2:] != (IMG_SIZE, IMG_SIZE):
                logits = nn.functional.interpolate(
                    logits, size=(IMG_SIZE, IMG_SIZE),
                    mode='bilinear', align_corners=False)

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for b in range(len(fnames)):
                pred_mask = (preds[b] * 255).astype(np.uint8)
                base = os.path.splitext(
                    os.path.join(output_dir, fnames[b]))[0]
                Image.fromarray(pred_mask).save(base + '.png')
                total += 1

    print(f"  Saved {total} predictions to {output_dir}")
    return total


# ================================================================
# HISTOGRAM MATCHING
# ================================================================

def compute_histogram(images_dir, n_bins=256, max_images=None):
    """
    Compute aggregate grayscale intensity histogram from an image directory.
    Returns (hist, cdf) both shape (n_bins,), hist normalized.
    """
    hist   = np.zeros(n_bins, dtype=np.float64)
    fnames = sorted([f for f in os.listdir(images_dir)
                     if f.lower().endswith(
                         ('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    if max_images is not None:
        step   = max(1, len(fnames) // max_images)
        fnames = fnames[::step][:max_images]

    for fn in tqdm(fnames,
                   desc=f"Computing histogram: "
                        f"{os.path.basename(images_dir)}"):
        arr  = np.array(
            Image.open(os.path.join(images_dir, fn)).convert('RGB'),
            dtype=np.uint8)
        gray = arr.mean(axis=2).astype(np.uint8)
        vals, counts = np.unique(gray, return_counts=True)
        for v, c in zip(vals, counts):
            hist[v] += c

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist, np.cumsum(hist)


def histogram_match_image(source_img, source_cdf, target_cdf):
    """
    Apply histogram matching to bring source_img's distribution toward
    source_cdf, given that the current image distribution is target_cdf.

    Args:
        source_img  : np.array (H, W, 3) uint8
        source_cdf  : CDF of training cities (distribution to match TO)
        target_cdf  : CDF of the current test image
    Returns:
        matched     : np.array (H, W, 3) uint8
    """
    mapping = np.array(
        [np.argmin(np.abs(source_cdf - target_cdf[t])) for t in range(256)],
        dtype=np.uint8)

    gray    = source_img.mean(axis=2)
    matched = np.zeros_like(source_img)

    for c in range(3):
        channel     = source_img[:, :, c].astype(np.float32)
        gray_f      = gray.astype(np.float32)
        ratio       = np.where(gray_f > 0, channel / (gray_f + 1e-6), 1.0)
        mapped_gray = mapping[
            gray.astype(np.uint8).ravel()
        ].reshape(gray.shape).astype(np.float32)
        matched[:, :, c] = np.clip(
            mapped_gray * ratio, 0, 255).astype(np.uint8)

    return matched


# ================================================================
# DATASET LOADING
# ================================================================

def get_dataset(model_type, data_root, split='test', augment=False):
    """
    Load the appropriate shadow dataset for a given model type.

    MAMNet and OGLANet both use MAMNet's ShadowDatasetEnhanced directly
    (4-channel RGB+contrast input). Using the mamnet copy avoids any
    path/signature mismatch from the oglanet copy.

    DINOv3 uses MAMNet's standard ShadowDataset (3-channel RGB).
    """
    _clean_model_paths()
    _add_path(SCRIPT_DIR)

    if model_type in ('mamnet', 'oglanet'):
        # Both CNN models use the same 4-channel enhanced dataset.
        # Always import from mamnet to avoid copy-related signature mismatches.
        from mamnet.data.dataset_enhanced import ShadowDatasetEnhanced
        return ShadowDatasetEnhanced(
            root_dir=[data_root],
            split=split,
            img_size=IMG_SIZE,
            task_id=2,       # 4-channel: RGB + contrast
            augment=augment,
            geo_metadata_path=None)

    else:  # dinov3
        # DINOv3 uses standard 3-channel RGB dataset from mamnet
        from mamnet.data.dataset import ShadowDataset
        return ShadowDataset(
            root_dir=data_root,
            split=split,
            img_size=IMG_SIZE,
            augment=augment)


def get_dataloader(model_type, data_root, split='test', batch_size=4,
                   num_workers=4, augment=False, shuffle=False):
    """Return a DataLoader for the given model and data root."""
    from torch.utils.data import DataLoader
    ds = get_dataset(model_type, data_root, split=split, augment=augment)
    print(f"  Dataset: {len(ds)} images  split={split}  root={data_root}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# ================================================================
# LOCO FOLD HELPERS
# ================================================================

CITIES           = ['chicago', 'miami', 'phoenix']
LOCO_HOLDOUT_MAP = {0: 'phoenix', 1: 'miami', 2: 'chicago'}


def get_training_cities(holdout_city):
    """Return the two training cities for a LOCO fold."""
    return [c for c in CITIES if c != holdout_city]


def find_checkpoint(output_base, model_type, pattern):
    """Find the most-recently-modified checkpoint matching glob pattern."""
    import glob
    search_dir = os.path.join(output_base, model_type, 'outputs')
    dirs = sorted(glob.glob(os.path.join(search_dir, pattern)),
                  key=os.path.getmtime, reverse=True)
    for d in dirs:
        ckpt = os.path.join(d, 'checkpoint_best.pth')
        if os.path.isfile(ckpt):
            return ckpt
    return None