"""
OGLANet + IIM: Shadow Detection with Illumination-Invariant Front-End.

Inserts the YOLA Illumination-Invariant Module (IIM) between the input image
and the GLAM encoder.  Everything else (DFFM, Decoder, OAM) is identical
to base OGLANet.

Pipeline
--------
    image (3 or 4 ch)
        │
        ├─ RGB (3ch) ──► IIM ──► 3ch fused ─┐
        │                                     ├──► GLAMEncoder ──► DFFM ──► Decoder ──► OAM
        └─ [Contrast (1ch, optional)] ───────┘

The GLAMEncoder already selects between ResNet34Encoder (3-ch) and
ResNet34Encoder4Ch (4-ch) based on the use_contrast flag, so no changes
are needed downstream of the encoder.

Reference
---------
Hong et al., "You Only Look Around: Learning Illumination Invariant Feature
for Low-light Object Detection", NeurIPS 2024.
"""

import torch
import torch.nn as nn

from .iim import IIM
from .glam import GLAMEncoder
from .dffm import DFFM
from .decoder import Decoder
from .oam import OAM


class OGLANetIIM(nn.Module):
    """
    OGLANet with IIM front-end for illumination-invariant shadow detection.

    Architecture
    ------------
    1. IIM         : learnable zero-mean kernels in log-RGB space → 3-ch fused output
    2. GLAMEncoder : 5-stage GLAM encoder (wraps ResNet-34 backbone)
    3. DFFM        : dense feature fusion across encoder scales
    4. Decoder     : progressive upsampling to original resolution
    5. OAM         : omni-scale aggregation → 6 predictions (P1–P6)

    Training outputs : dict {'p1'…'p6', 'iim_features'}
    Eval output      : P6 tensor [B, num_classes, H, W]

    Parameters
    ----------
    num_classes  : int   (default 2)
    pretrained   : bool  load ImageNet weights for ResNet-34 backbone
    img_size     : int   input image spatial size (default 384)
    use_contrast : bool  expect a 4th contrast channel in the input tensor;
                         passed through to GLAMEncoder which selects the
                         appropriate 3-ch or 4-ch encoder variant
    num_kernels  : int   number of IIM learnable kernels (default 8)
    kernel_size  : int   spatial size of each IIM kernel (default 5)
    """

    def __init__(self, num_classes=2, pretrained=True, img_size=384,
                 use_contrast=False, num_kernels=8, kernel_size=5):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.use_contrast = use_contrast

        # ---- IIM front-end (always operates on the 3-ch RGB slice) ----
        self.iim = IIM(num_kernels=num_kernels, kernel_size=kernel_size)

        # ---- GLAM Encoder ---------------------------------------------------
        # GLAMEncoder internally chooses ResNet34Encoder (3-ch) or
        # ResNet34Encoder4Ch (4-ch) based on use_contrast.
        # IIM output is 3-ch; we concat contrast as ch-4 when use_contrast=True,
        # so the encoder receives exactly the channel count it expects.
        self.encoder = GLAMEncoder(pretrained=pretrained,
                                   use_contrast=use_contrast)

        # ---- Dense Feature Fusion Module ----
        self.dffm = DFFM()

        # ---- Decoder ----
        self.decoder = Decoder(target_size=(img_size, img_size))

        # ---- Omni-scale Aggregation Module ----
        self.oam = OAM(num_classes=num_classes,
                       target_size=(img_size, img_size))

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args
        ----
        x : [B, C, H, W]  C = 3 (RGB) or 4 (RGBC).

        Returns
        -------
        Training:
            dict with keys 'p1'–'p6' and 'iim_features'
              'p1'–'p6'       : [B, num_classes, H, W]  all 6 OAM predictions
              'iim_features'  : [B, feat_dim, H, W]     raw IIM features for II loss
        Eval:
            P6 tensor  [B, num_classes, H, W]
        """
        B, C, H, W = x.size()

        # ---- Split RGB from optional contrast channel ----
        rgb = x[:, :3]
        contrast = x[:, 3:4] if (self.use_contrast and C == 4) else None

        # ---- IIM: produces 3-ch illumination-invariant representation ----
        iim_out, iim_features = self.iim(rgb)   # [B,3,H,W], [B,feat_dim,H,W]

        # ---- Build encoder input ----
        if contrast is not None:
            enc_input = torch.cat([iim_out, contrast], dim=1)  # [B, 4, H, W]
        else:
            enc_input = iim_out                                 # [B, 3, H, W]

        # ---- GLAM Encoder ----
        encoder_features = self.encoder(enc_input)
        # returns: {'feat1'…'feat5'}

        # ---- DFFM ----
        dffm_features = self.dffm(encoder_features)
        # returns: {'s4_d', 's3_d', 's2_d', 's1_d'}

        # ---- Decoder ----
        decoder_features = self.decoder(dffm_features)
        # returns: {'s4_d_up', 's3_d_up', 's2_d_up', 's1_d_up'}

        # ---- OAM: 6-output prediction ----
        predictions = self.oam(decoder_features)
        # returns: {'p1'…'p6'}

        if self.training:
            predictions['iim_features'] = iim_features
            return predictions
        else:
            return predictions['p6']

    # ------------------------------------------------------------------
    def get_predictions(self, x):
        """Binary class predictions for evaluation (convenience wrapper)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)          # [B, num_classes, H, W]
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
        return preds


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("OGLANetIIM smoke test")
    print("=" * 60)

    for contrast in [False, True]:
        tag = "RGBC" if contrast else "RGB"
        C = 4 if contrast else 3
        model = OGLANetIIM(num_classes=2, pretrained=False, img_size=384,
                           use_contrast=contrast, num_kernels=8, kernel_size=5)

        # ---- Training mode ----
        model.train()
        x = torch.randn(2, C, 384, 384)
        out = model(x)
        print(f"\n[{tag}] Training outputs:")
        for k, v in out.items():
            shape = v.shape if hasattr(v, 'shape') else type(v)
            print(f"  {k}: {shape}")

        # ---- Eval mode ----
        model.eval()
        out_eval = model(x)
        print(f"[{tag}] Eval output: {out_eval.shape}")

    total_p = sum(p.numel() for p in model.parameters())
    iim_p = sum(p.numel() for p in model.iim.parameters())
    print(f"\nTotal params:     {total_p:,}")
    print(f"IIM params:       {iim_p:,}  ({iim_p / 1e6:.4f} M)")
    print(f"Non-IIM params:   {total_p - iim_p:,}")