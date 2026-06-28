"""
MAMNet + IIM: Shadow Detection with Illumination-Invariant Front-End.

Inserts the YOLA Illumination-Invariant Module (IIM) between the input image
and the ResNet-34 encoder.  Everything else (MSCAF, Decoder, Aux) is identical
to the base MAMNet.

Pipeline
--------
    image (3 or 4 ch)
        │
        ├─ RGB (3ch) ──► IIM ──► 3ch fused
        │                            │
        └─ [Contrast (1ch)] ────────cat──► encoder (3 or 4 ch)
                                              │
                                          MSCAF → Decoder → prediction
"""

import torch
import torch.nn as nn

from .iim import IIM
from .encoder import ResNet34Encoder
from .encoder_4ch import ResNet34Encoder4Ch
from .mscaf import MSCAF
from .decoder import Decoder
from .auxiliary import AuxiliaryModule


class MAMNetIIM(nn.Module):
    """
    MAMNet with IIM front-end.

    Parameters
    ----------
    num_classes  : int   (default 2)
    pretrained   : bool  load ImageNet weights for encoder.
    use_aux      : bool  enable auxiliary deep-supervision branches.
    use_contrast : bool  expect a 4th contrast channel in the input and use
                         the 4-channel encoder.
    num_kernels  : int   number of IIM learnable kernels (default 8).
    kernel_size  : int   spatial size of each IIM kernel (default 5).
    """

    def __init__(self, num_classes=2, pretrained=True, use_aux=True,
                 use_contrast=False, num_kernels=8, kernel_size=5):
        super().__init__()

        self.num_classes = num_classes
        self.use_aux = use_aux
        self.use_contrast = use_contrast

        # ---- IIM front-end ----
        self.iim = IIM(num_kernels=num_kernels, kernel_size=kernel_size)

        # ---- Encoder ----
        if use_contrast:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)
            print("MAMNet-IIM: 4-channel encoder (IIM-fused RGB + Contrast)")
        else:
            self.encoder = ResNet34Encoder(pretrained=pretrained)
            print("MAMNet-IIM: 3-channel encoder (IIM-fused RGB)")

        # ---- MSCAF ----
        self.mscaf = MSCAF(in_channels=512)

        # ---- Decoder with CCA ----
        self.decoder = Decoder(num_classes=num_classes)

        # ---- Auxiliary branches ----
        if use_aux:
            self.aux_module = AuxiliaryModule(
                num_classes=num_classes, dropout_rate=0.3)

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args
        ----
        x : [B, C, H, W]  where C = 3 (RGB) or 4 (RGBC).

        Returns
        -------
        Training (use_aux and self.training):
            dict with keys  'main', 'aux1', 'aux2', 'aux3', 'iim_features'
        Eval:
            logits tensor  [B, num_classes, H, W]
        """
        B, C, H, W = x.size()

        # --- Split channels ---
        if self.use_contrast and C == 4:
            rgb = x[:, :3]
            contrast = x[:, 3:4]
        else:
            rgb = x[:, :3]  # safe even if C > 3
            contrast = None

        # --- IIM ---
        iim_out, iim_features = self.iim(rgb)   # both [B, 3, H, W], [B, D, H, W]

        # --- Encoder input ---
        if contrast is not None:
            enc_input = torch.cat([iim_out, contrast], dim=1)  # [B, 4, H, W]
        else:
            enc_input = iim_out                                # [B, 3, H, W]

        # --- Encoder ---
        enc_features = self.encoder(enc_input)

        # --- MSCAF ---
        mscaf_out = self.mscaf(enc_features['feat5'])

        # --- Decoder ---
        decoder_outputs = self.decoder(mscaf_out, enc_features)
        main_out = decoder_outputs['main']  # [B, num_classes, H, W]

        # --- Auxiliary (training only) ---
        if self.use_aux and self.training:
            aux_outputs = self.aux_module(
                decoder_outputs['dec_feat1'],
                decoder_outputs['dec_feat2'],
                decoder_outputs['dec_feat3'],
                target_size=(H, W),
            )
            return {
                'main': main_out,
                'aux1': aux_outputs['aux1'],
                'aux2': aux_outputs['aux2'],
                'aux3': aux_outputs['aux3'],
                'iim_features': iim_features,
            }
        else:
            return main_out

    # ------------------------------------------------------------------
    def get_predictions(self, x):
        """Binary predictions for inference/evaluation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
        return preds


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # --- Quick smoke test ---
    for contrast in [False, True]:
        tag = "RGBC" if contrast else "RGB"
        C = 4 if contrast else 3
        model = MAMNetIIM(num_classes=2, pretrained=False, use_aux=True,
                          use_contrast=contrast, num_kernels=8, kernel_size=5)

        model.train()
        x = torch.randn(2, C, 256, 256)
        out = model(x)
        print(f"\n[{tag}] Training outputs:")
        for k, v in out.items():
            shape = v.shape if hasattr(v, 'shape') else type(v)
            print(f"  {k}: {shape}")

        model.eval()
        out_eval = model(x)
        print(f"[{tag}] Eval output: {out_eval.shape}")

    total = sum(p.numel() for p in model.parameters())
    iim_params = sum(p.numel() for p in model.iim.parameters())
    print(f"\nTotal params:  {total:,}")
    print(f"IIM params:    {iim_params:,}  ({iim_params/1e6:.3f} M)")
    print(f"Non-IIM params:{total - iim_params:,}")