"""
DINOv3 + IIM (Illumination-Invariant Module) for Shadow Detection.

Pipeline:
    image (RGB) → IIM (0.008 M params) → fused 3-ch → DINOv3 backbone → decoder → prediction

The IIM sits at input level only.  The backbone and decoder are identical to the
base DINOv3ShadowDetector — no architectural changes downstream.

Training mode  → dict  {'main': logits [B, 2, H, W], 'iim_features': [B, D, H, W]}
Eval mode      → logits [B, 2, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import DINOv3Decoder
from iim import IIM          # copy models/iim.py → iim.py in this directory


class DINOv3ShadowDetectorIIM(nn.Module):
    """
    DINOv3-based shadow detector with IIM illumination-invariant front-end.

    Parameters
    ----------
    num_classes   : int   Number of output classes (default 2).
    model_name    : str   DINOv3 variant ('dinov3_vits16' | 'dinov3_vitb16' | 'dinov3_vitl16').
    weights_path  : str   Path to pretrained DINOv3 .pth weights (optional).
    pretrained    : bool  Load pretrained weights.
    frozen_stages : int   Backbone stages to freeze (-1 = train all).
    num_kernels   : int   Number of IIM learnable kernels (default 8).
    kernel_size   : int   Spatial size of each IIM kernel (default 5).
    """

    def __init__(
        self,
        num_classes=2,
        model_name='dinov3_vits16',
        weights_path=None,
        pretrained=True,
        frozen_stages=-1,
        num_kernels=8,
        kernel_size=5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # ---- IIM front-end ----
        print('Initializing IIM front-end...')
        self.iim = IIM(num_kernels=num_kernels, kernel_size=kernel_size)

        # ---- DINOv3 backbone ----
        print('Initializing DINOv3 backbone...')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
        )

        embed_dim = self.backbone.embed_dim

        # ---- Decoder ----
        print('Initializing decoder...')
        self.decoder = DINOv3Decoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
        )

        # Diagnostics
        total_p    = sum(p.numel() for p in self.parameters())
        trainable_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        iim_p      = sum(p.numel() for p in self.iim.parameters())
        backbone_p = sum(p.numel() for p in self.backbone.parameters())
        decoder_p  = sum(p.numel() for p in self.decoder.parameters())

        print(f'\nDINOv3-IIM Shadow Detector initialized:')
        print(f'  Model:            {model_name}')
        print(f'  IIM kernels:      {num_kernels} × {kernel_size}×{kernel_size}')
        print(f'  Input size:       384×384  (24×24 patches, perfect fit)')
        print(f'  Output classes:   {num_classes}')
        print(f'  Total params:     {total_p:,}')
        print(f'  Trainable params: {trainable_p:,}')
        print(f'  IIM params:       {iim_p:,}  ({iim_p / 1e6:.4f} M)')
        print(f'  Backbone params:  {backbone_p:,}')
        print(f'  Decoder params:   {decoder_p:,}')

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args
        ----
        x : [B, 3, H, W]  ImageNet-normalised RGB.

        Returns
        -------
        Training mode : dict
            'main'         : [B, num_classes, H, W]  segmentation logits
            'iim_features' : [B, D, H, W]             raw IIM features (for II loss)
        Eval mode : [B, num_classes, H, W]  segmentation logits
        """
        B, C, H, W = x.shape

        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                f"Input size ({H}×{W}) must be divisible by patch size 16. "
                f"Expected 384×384 or another multiple of 16."
            )

        # 1. IIM: illumination-invariant front-end
        iim_out, iim_features = self.iim(x)   # [B, 3, H, W],  [B, D, H, W]

        # 2. Backbone: operate on fused representation (not raw RGB)
        features = self.backbone(iim_out)       # dict of block features

        # 3. Decoder
        output = self.decoder(features)         # [B, num_classes, H, W]

        # Safety resize (should be a no-op for multiples of 16)
        if output.shape[2] != H or output.shape[3] != W:
            output = F.interpolate(output, size=(H, W),
                                   mode='bilinear', align_corners=False)

        if self.training:
            return {'main': output, 'iim_features': iim_features}
        else:
            return output

    # ------------------------------------------------------------------
    def get_predictions(self, x):
        """Binary predictions for inference / evaluation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)      # eval mode → plain tensor
            return torch.argmax(logits, dim=1)

    # ------------------------------------------------------------------
    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False


# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('=' * 60)
    print('Testing DINOv3-IIM Shadow Detector')
    print('=' * 60)

    model = DINOv3ShadowDetectorIIM(
        num_classes=2,
        model_name='dinov3_vits16',
        weights_path=None,
        pretrained=False,          # set True in real use
        num_kernels=8,
        kernel_size=5,
    )

    x = torch.randn(2, 3, 384, 384)

    model.train()
    out_train = model(x)
    print(f"\n[Train] main:         {out_train['main'].shape}")
    print(f"[Train] iim_features: {out_train['iim_features'].shape}")

    model.eval()
    out_eval = model(x)
    print(f"\n[Eval]  logits:       {out_eval.shape}")

    preds = model.get_predictions(x)
    print(f"[Eval]  preds:        {preds.shape}  unique={torch.unique(preds).tolist()}")

    print('\n✓ All checks passed.')