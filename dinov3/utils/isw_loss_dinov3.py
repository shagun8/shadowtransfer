"""
Instance Selective Whitening (ISW) Loss adapted for DINOv3 ViT backbone.

Based on: Choi et al., "RobustNet: Improving Domain Generalization in Urban-Scene
Segmentation via Instance Selective Whitening", CVPR 2021.

Core ISWLoss is identical to the MAMNet version — operates on [B, C, H, W] features.
DINOv3FeatureHooks replaces EncoderFeatureHooks, handling ViT's token-sequence output
([B, N+1, D]) by stripping the CLS token and reshaping to spatial [B, D, h, w].

Default hooked blocks: 3, 6, 9  (0-indexed out of 12 for ViT-S/16)
  - block3  → early layers, highest style sensitivity
  - block6  → mid layers
  - block9  → mid-late layers
All produce [B, 384, 24, 24] features for 384×384 input (ViT-S embed_dim=384, 24=384/16).

Loss is computed per-image and averaged over the batch (per-image mean level).
"""

import torch
import torch.nn as nn
import numpy as np
import os
import math


# ──────────────────────────────────────────────────────────────────────────────
# ISW Loss (architecture-agnostic core)
# ──────────────────────────────────────────────────────────────────────────────

class ISWLoss(nn.Module):
    """
    Instance Selective Whitening loss.

    For each hooked layer:
      1. Instance-standardise features  (zero mean, unit variance per channel).
      2. Compute per-instance covariance matrix of standardised features.
      3. Mask with precomputed binary mask M̃ (style-sensitive elements only).
      4. L1 norm of the masked covariance → per-image loss.
      5. Average over images in the batch.

    Final ISW loss = (1/L) * Σ_l  per-image-mean L1 of masked covariance at layer l.
    """

    def __init__(self, mask_dir, layer_names=None):
        """
        Args:
            mask_dir:    directory containing <layer>_mask.npy files
            layer_names: list of layer names matching the precomputed masks
                         default: ['block3', 'block6', 'block9']
        """
        super().__init__()
        if layer_names is None:
            layer_names = ['block3', 'block6', 'block9']
        self.layer_names = layer_names

        for name in layer_names:
            mask_path = os.path.join(mask_dir, f'{name}_mask.npy')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f'ISW mask not found: {mask_path}.  '
                    f'Run compute_isw_masks_dinov3.py first.')
            mask_np = np.load(mask_path)
            # Register as buffer so it moves to GPU with .to(device)
            self.register_buffer(f'mask_{name}',
                                 torch.from_numpy(mask_np).float())

        # Log mask statistics
        for name in layer_names:
            m = getattr(self, f'mask_{name}')
            C = m.shape[0]
            total_upper = C * (C - 1) // 2
            selected = int(m.sum().item())
            print(f'  ISW mask {name}: {selected}/{total_upper} '
                  f'upper-triangular elements selected '
                  f'({selected / max(total_upper, 1) * 100:.1f}%)')

    def forward(self, features):
        """
        Compute ISW loss from captured encoder features.

        Args:
            features: dict  {layer_name: tensor [B, C, H, W]}

        Returns:
            Scalar loss (per-image mean, averaged over layers).
        """
        layer_losses = []
        for name in self.layer_names:
            if name not in features:
                continue
            feat = features[name]
            mask = getattr(self, f'mask_{name}')
            layer_losses.append(self._layer_loss(feat, mask))

        if len(layer_losses) == 0:
            buffers = list(self.buffers())
            dev = buffers[0].device if buffers else torch.device('cpu')
            return torch.tensor(0.0, device=dev)
        return sum(layer_losses) / len(layer_losses)

    @staticmethod
    def _layer_loss(feat, mask):
        """
        Per-layer ISW loss (per-image mean L1 of masked instance covariance).

        Args:
            feat: [B, C, H, W]   spatial feature map
            mask: [C, C] float   binary upper-triangular style-sensitivity mask

        Returns:
            Scalar — per-image mean of ||Σ_s ⊙ M̃||_1
        """
        B, C, H, W = feat.shape
        N = H * W

        # 1. Instance standardise  ─────────────────────────────────
        feat_flat = feat.reshape(B, C, N)                       # [B, C, N]
        mean      = feat_flat.mean(dim=2, keepdim=True)         # [B, C, 1]
        std       = feat_flat.std(dim=2, keepdim=True) + 1e-5   # [B, C, 1]
        feat_std  = (feat_flat - mean) / std                    # [B, C, N]

        # 2. Covariance of standardised features  ──────────────────
        cov = torch.bmm(feat_std, feat_std.transpose(1, 2)) / N  # [B, C, C]

        # 3. Mask and L1 (per-image)  ──────────────────────────────
        masked_cov = cov * mask.unsqueeze(0)                    # [B, C, C]
        per_image  = masked_cov.abs().sum(dim=(1, 2))           # [B]

        return per_image.mean()


# ──────────────────────────────────────────────────────────────────────────────
# DINOv3-specific forward hooks
# ──────────────────────────────────────────────────────────────────────────────

class DINOv3FeatureHooks:
    """
    Register forward hooks on DINOv3 ViT transformer blocks to capture
    intermediate spatial feature maps.

    ViT block outputs: [B, N+1, D]   (CLS token at index 0, then patch tokens)
    Hook strips CLS and reshapes to: [B, D, H/patch_size, W/patch_size]

    Default: hooks blocks 3, 6, 9 of DINOv3-S (12 blocks total).

    Usage:
        hooks = DINOv3FeatureHooks(model, ['block3', 'block6', 'block9'],
                                    img_size=384, patch_size=16)
        output = model(images)           # hooks fire during forward
        isw_loss = isw_module(hooks.features)
        ...
        hooks.remove()                   # clean up when done
    """

    # Maps user-facing layer names → 0-indexed block positions in dinov3.blocks
    _LAYER_MAP = {
        'block3':  3,
        'block6':  6,
        'block9':  9,
        'block11': 11,
    }

    def __init__(self, model, layer_names=None, img_size=384, patch_size=16):
        """
        Args:
            model:       DINOv3ShadowDetector instance
            layer_names: list of layer name strings (must be in _LAYER_MAP)
            img_size:    input image size (assumed square)
            patch_size:  ViT patch size (16 for DINOv3)
        """
        if layer_names is None:
            layer_names = ['block3', 'block6', 'block9']
        self.layer_names = layer_names
        self.features    = {}
        self._handles    = []

        # Spatial grid dimensions (fixed for given img_size / patch_size)
        self.h_patches = img_size // patch_size   # 24 for 384/16
        self.w_patches = img_size // patch_size   # 24

        for lname in layer_names:
            if lname not in self._LAYER_MAP:
                raise ValueError(
                    f'Unknown layer name "{lname}". '
                    f'Valid options: {list(self._LAYER_MAP.keys())}')
            block_idx = self._LAYER_MAP[lname]

            # Access: model.backbone.dinov3.blocks[block_idx]
            if not hasattr(model, 'backbone') or \
               not hasattr(model.backbone, 'dinov3') or \
               not hasattr(model.backbone.dinov3, 'blocks'):
                raise AttributeError(
                    'Expected model.backbone.dinov3.blocks to exist. '
                    'Make sure you are passing a DINOv3ShadowDetector instance.')

            num_blocks = len(model.backbone.dinov3.blocks)
            if block_idx >= num_blocks:
                raise IndexError(
                    f'Block index {block_idx} out of range for backbone '
                    f'with {num_blocks} blocks.')

            block  = model.backbone.dinov3.blocks[block_idx]
            handle = block.register_forward_hook(self._make_hook(lname))
            self._handles.append(handle)

    def _make_hook(self, name):
        h = self.h_patches
        w = self.w_patches

        def hook_fn(module, inp, out):
            """
            out: [B, N+1, D]  — first token is CLS, rest are patch tokens
            Reshape patch tokens to [B, D, h, w].
            """
            # Some ViT implementations return a tuple; unwrap if needed
            if isinstance(out, tuple):
                out = out[0]

            B, seq_len, D = out.shape

            # Strip CLS token (index 0)
            patch_tokens = out[:, 1:, :]                             # [B, N, D]
            N = patch_tokens.shape[1]

            # Safety: infer spatial dims from token count if not square
            if N == h * w:
                feat = patch_tokens.transpose(1, 2).reshape(B, D, h, w).contiguous()
            else:
                # Fallback: compute dims from N
                h_inferred = w_inferred = int(math.isqrt(N))
                if h_inferred * w_inferred != N:
                    raise RuntimeError(
                        f'Cannot reshape {N} patch tokens into a square grid. '
                        f'Expected {h * w} = {h}×{w}.')
                feat = patch_tokens.transpose(1, 2).reshape(
                    B, D, h_inferred, w_inferred).contiguous()

            self.features[name] = feat

        return hook_fn

    def remove(self):
        """Remove all hooks and clear stored features."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.features.clear()