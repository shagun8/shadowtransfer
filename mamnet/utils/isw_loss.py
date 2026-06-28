"""
Instance Selective Whitening (ISW) Loss for domain-generalizable shadow detection.

Based on: Choi et al., "RobustNet: Improving Domain Generalization in Urban-Scene
Segmentation via Instance Selective Whitening", CVPR 2021.

Key idea: selectively whiten only the feature covariance elements that are
sensitive to photometric transformations (i.e., domain-specific style),
leaving content-related covariances intact.

Loss is computed per-image and averaged over the batch (per-image mean level).
"""

import torch
import torch.nn as nn
import numpy as np
import os


class ISWLoss(nn.Module):
    """
    Instance Selective Whitening loss.

    For each hooked encoder layer:
      1. Instance-standardise features (zero mean, unit variance per channel).
      2. Compute the per-instance covariance matrix of the standardised features.
      3. Mask with the precomputed binary mask M̃ (style-sensitive elements only).
      4. L1 norm of the masked covariance → per-image loss.
      5. Average over images in the batch.

    Final ISW loss = (1/L) * Σ_l  per-image-mean L1 of masked covariance at layer l.
    """

    def __init__(self, mask_dir, layer_names=None):
        """
        Args:
            mask_dir:    directory containing <layer>_mask.npy files
            layer_names: list of layer names to apply ISW to
                         (default: ['feat1', 'feat2', 'feat3'])
        """
        super().__init__()
        if layer_names is None:
            layer_names = ['feat1', 'feat2', 'feat3']
        self.layer_names = layer_names

        for name in layer_names:
            mask_path = os.path.join(mask_dir, f'{name}_mask.npy')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f'ISW mask not found: {mask_path}.  '
                    f'Run compute_isw_masks.py first.')
            mask_np = np.load(mask_path)
            # register as buffer so it moves with .to(device)
            self.register_buffer(f'mask_{name}',
                                 torch.from_numpy(mask_np).float())

        # Log mask statistics
        for name in layer_names:
            m = getattr(self, f'mask_{name}')
            total_upper = m.shape[0] * (m.shape[0] - 1) // 2
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
            # Fallback: get device from buffers (masks are registered as buffers)
            buffers = list(self.buffers())
            dev = buffers[0].device if buffers else torch.device('cpu')
            return torch.tensor(0.0, device=dev)
        return sum(layer_losses) / len(layer_losses)

    @staticmethod
    def _layer_loss(feat, mask):
        """
        Per-layer ISW loss (Eq. 17 in the paper).

        Args:
            feat: [B, C, H, W]
            mask: [C, C] binary (1 = style-sensitive upper-triangular element)

        Returns:
            Scalar — per-image mean of ||Σ_s ⊙ M̃||_1
        """
        B, C, H, W = feat.shape
        N = H * W

        # 1. Instance standardise  ─────────────────────────────────
        feat_flat = feat.reshape(B, C, N)                      # [B, C, N]
        mean = feat_flat.mean(dim=2, keepdim=True)             # [B, C, 1]
        std  = feat_flat.std(dim=2, keepdim=True) + 1e-5       # [B, C, 1]
        feat_std = (feat_flat - mean) / std                    # [B, C, N]

        # 2. Covariance of standardised features  ──────────────────
        cov = torch.bmm(feat_std, feat_std.transpose(1, 2))   # [B, C, C]
        cov = cov / N

        # 3. Mask and L1  ──────────────────────────────────────────
        masked_cov = cov * mask.unsqueeze(0)                   # [B, C, C]
        per_image  = masked_cov.abs().sum(dim=(1, 2))          # [B]

        return per_image.mean()


# ──────────────────────────────────────────────────────────────────
# Feature hook utilities
# ──────────────────────────────────────────────────────────────────

class EncoderFeatureHooks:
    """
    Register forward hooks on encoder layers to capture intermediate
    feature maps without modifying the base model.

    Usage:
        hooks = EncoderFeatureHooks(model, ['feat1', 'feat2', 'feat3'])
        outputs = model(images)            # hooks capture features
        isw_loss = isw_module(hooks.features)
        ...
        hooks.remove()                     # clean up when done
    """

    # Maps layer names to encoder attribute names
    _LAYER_MAP = {
        'feat1': 'layer1',
        'feat2': 'layer2',
        'feat3': 'layer3',
        'feat4': 'layer4',
    }

    def __init__(self, model, layer_names=None):
        if layer_names is None:
            layer_names = ['feat1', 'feat2', 'feat3']
        self.layer_names = layer_names
        self.features = {}
        self._handles = []

        encoder = model.encoder
        for lname in layer_names:
            attr = self._LAYER_MAP[lname]
            if not hasattr(encoder, attr):
                raise AttributeError(
                    f'Encoder has no attribute "{attr}" for layer "{lname}"')
            handle = getattr(encoder, attr).register_forward_hook(
                self._make_hook(lname))
            self._handles.append(handle)

    def _make_hook(self, name):
        def hook_fn(module, inp, out):
            self.features[name] = out
        return hook_fn

    def remove(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.features.clear()