"""
DINOv3 + MRFP/MRFP+ for Shadow Detection
==========================================
Integrates MRFP (Udupa et al., CVPR 2024) into the DINOv3 shadow-detection
architecture for cross-city domain generalisation.

MRFP adaptation for ViT features (all at H/16 spatial resolution):

    HRFP  → feat_block3   earliest extracted features; most local/texture-like
                           (analogous to CNN stage-0 in the original paper)
    NP+   → feat_block11  deepest features; most semantic/global/style-like
                           (analogous to CNN feat4)
    HRFP+ → decoder up3 output (64 ch) via a frozen random 1×1 projection
             from embed_dim → 64.

NOTE on InstanceNorm (absent here intentionally):
    In MAMNet, IN is applied to CNN encoder features that flow into the decoder
    via separate CCA skip connections — the main trunk BN is insulated.
    In DINOv3, all four ViT blocks feed into a single concat → BatchNorm2d
    (feature_fusion).  Applying IN unconditionally (100% prob) to blocks 6 & 9
    during training causes the feature_fusion BN to accumulate running statistics
    that expect instance-normalised inputs.  At eval the IN is removed, the BN
    sees the natural ViT distribution, its normalisation is wrong, and the decoder
    produces a checkerboard/zigzag artefact.
    Solution: omit IN entirely.  HRFP (HF) + NP+ (LF) already cover both ends
    of the frequency spectrum as the paper intends.

Variants:
    MRFP  = {HRFP, NP+}
    MRFP+ = {HRFP, HRFP+, NP+}   ← best from paper Table 5

Training:  stochastic perturbation branches active each with probability p.
Inference: all branches disabled → architecture identical to DINOv3ShadowDetector.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import DINOv3Decoder
from mrfp_modules import HRFPModule, NormalizedPerturbation

# DINOv3 decoder channel layout (from dinov3_decoder.py):
#   dec_channels = [embed_dim, 256, 128, 64, 32]
#   up3 output (penultimate before up4) → 64 channels
_DEC3_CHANNELS = 64


class DINOv3ShadowDetectorMRFP(nn.Module):
    """
    DINOv3 Shadow Detector with MRFP/MRFP+ training-time perturbation.

    Parameters
    ----------
    num_classes    : int   — output classes (2 for shadow/non-shadow)
    model_name     : str   — 'dinov3_vits16' | 'dinov3_vitb16' | 'dinov3_vitl16'
    weights_path   : str   — path to pretrained .pth (or None for random init)
    pretrained     : bool  — load pretrained DINOv3 weights
    frozen_stages  : int   — backbone stages to freeze (-1 = train all)
    use_mrfp_plus  : bool  — True → MRFP+ (HRFP+HRFP++NP+); False → MRFP
    hrfp_prob      : float — probability of applying HRFP each step
    np_prob        : float — probability of applying NP+ each step
    hrfp_plus_prob : float — probability of applying HRFP+ each step
    bn_std         : float — std of Gaussian for HRFP batch-norm init
    """

    def __init__(self,
                 num_classes: int = 2,
                 model_name: str = 'dinov3_vits16',
                 weights_path: str = None,
                 pretrained: bool = True,
                 frozen_stages: int = -1,
                 use_mrfp_plus: bool = True,
                 hrfp_prob: float = 0.5,
                 np_prob: float = 0.5,
                 hrfp_plus_prob: float = 0.5,
                 bn_std: float = 0.5):
        super().__init__()

        self.num_classes    = num_classes
        self.model_name     = model_name
        self.use_mrfp_plus  = use_mrfp_plus
        self.hrfp_prob      = hrfp_prob
        self.np_prob        = np_prob
        self.hrfp_plus_prob = hrfp_plus_prob

        # ---- Backbone ------------------------------------------------
        print('Initialising DINOv3 backbone...')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
        )
        embed_dim = self.backbone.embed_dim   # 384 (ViT-S), 768 (ViT-B), 1024 (ViT-L)

        # ---- Decoder -------------------------------------------------
        print('Initialising decoder...')
        self.decoder = DINOv3Decoder(num_classes=num_classes, embed_dim=embed_dim)

        # ---- MRFP perturbation modules (all frozen, zero learnable params) ----
        #
        # HRFP: randomly initialised overcomplete autoencoder applied to
        #   feat_block3 — the earliest extracted ViT features, most locally
        #   sensitive (counterpart to CNN stage-0 shallow textures).
        self.hrfp = HRFPModule(in_channels=embed_dim, bn_std=bn_std)

        # NP+: channel-statistic style randomisation applied to
        #   feat_block11 — deepest features encoding global statistics.
        self.np_plus = NormalizedPerturbation()

        # ---- MRFP+ extras --------------------------------------------
        if use_mrfp_plus:
            # HRFP+ channel projection:
            #   max_resolution_feat has embed_dim channels;
            #   dec3 (up3 output) has _DEC3_CHANNELS=64 channels.
            #   Frozen random 1×1 conv — consistent with MRFP's no-learnable-
            #   perturbation-params principle.
            self.hrfp_plus_proj = nn.Conv2d(
                embed_dim, _DEC3_CHANNELS, kernel_size=1, bias=False)
            nn.init.kaiming_normal_(
                self.hrfp_plus_proj.weight, mode='fan_out', nonlinearity='relu')
            for p in self.hrfp_plus_proj.parameters():
                p.requires_grad = False

        # ---- Summary -------------------------------------------------
        variant = 'MRFP+' if use_mrfp_plus else 'MRFP'
        print(f'\nDINOv3 + {variant} initialised:')
        print(f'  Backbone:          {model_name}  (embed_dim={embed_dim})')
        print(f'  Input size:        384×384 (24×24 patches)')
        print(f'  HRFP attachment:   feat_block3  (HF / texture perturbation)')
        print(f'  NP+  attachment:   feat_block11 (LF / style perturbation)')
        if use_mrfp_plus:
            print(f'  HRFP+ injected at: decoder up3 output ({_DEC3_CHANNELS}ch), '
                  f'via frozen proj {embed_dim}→{_DEC3_CHANNELS}')
        print(f'  InstanceNorm:      NOT used (would corrupt feature_fusion BN '
              f'running stats — see module docstring)')
        print(f'  HRFP p={hrfp_prob}  NP+ p={np_prob}'
              + (f'  HRFP+ p={hrfp_plus_prob}' if use_mrfp_plus else '')
              + f'  BN σ={bn_std}')

        total_p = sum(p.numel() for p in self.parameters())
        train_p = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Total parameters:     {total_p:,}')
        print(f'  Trainable parameters: {train_p:,}')
        print(f'  Frozen (HRFP+proj):   {total_p - train_p:,}')

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training  → Tensor [B, num_classes, H, W]  (MRFP branches active)
        Inference → Tensor [B, num_classes, H, W]  (identical to base DINOv3)
        """
        B, C, H, W = x.shape

        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                f'Input size ({H}×{W}) must be divisible by patch size (16). '
                f'Expected 384×384 or another multiple of 16.')

        # ==========================================================
        # 1. Backbone — extract multi-scale ViT features
        # ==========================================================
        features = self.backbone(x)
        # All features at [B, embed_dim, H/16, W/16]:
        #   'feat_block3'  'feat_block6'  'feat_block9'  'feat_block11'

        hrfp_output = None   # will hold HRFP dict if applied this step

        # ==========================================================
        # 2. Training-time MRFP perturbations
        #    (completely skipped at eval — no branches, no overhead)
        # ==========================================================
        if self.training:
            # ---- HRFP on feat_block3 (HF / fine-grained perturbation) ----
            # Weights re-randomised every call via _reinit_weights().
            # torch.no_grad() + .detach() prevents gradient storage through
            # the autoencoder (saves memory; same pattern as MAMNetMRFP).
            if random.random() < self.hrfp_prob:
                with torch.no_grad():
                    hrfp_output = self.hrfp(features['feat_block3'])
                features['feat_block3'] = (
                    features['feat_block3']
                    + hrfp_output['perturbation'].detach())

            # ---- NP+ on feat_block11 (LF / style perturbation) ----
            if random.random() < self.np_prob:
                features['feat_block11'] = self.np_plus(features['feat_block11'])

        # ==========================================================
        # 3. Decoder — stepped through manually for HRFP+ injection
        # ==========================================================
        feat_concat = torch.cat([
            features['feat_block3'],
            features['feat_block6'],
            features['feat_block9'],
            features['feat_block11'],
        ], dim=1)                                      # [B, embed_dim*4, H/16, W/16]

        dec_fused = self.decoder.feature_fusion(feat_concat)  # [B, embed_dim, H/16, W/16]
        dec1 = self.decoder.up1(dec_fused)                    # [B, 256,       H/8,  W/8 ]
        dec2 = self.decoder.up2(dec1)                         # [B, 128,       H/4,  W/4 ]
        dec3 = self.decoder.up3(dec2)                         # [B,  64,       H/2,  W/2 ]

        # ---- HRFP+ injection into penultimate decoder layer ----
        # Only when: MRFP+ enabled, training mode, HRFP was applied this step,
        # and the independent HRFP+ coin flip succeeds.
        if (self.use_mrfp_plus
                and self.training
                and hrfp_output is not None
                and random.random() < self.hrfp_plus_prob):
            # max_resolution_feat: [B, embed_dim, ~2*(H/16), ~2*(W/16)]
            hrfp_plus_feat = hrfp_output['max_resolution_feat'].detach()
            # Project embed_dim → _DEC3_CHANNELS (frozen random 1×1 conv)
            hrfp_plus_feat = self.hrfp_plus_proj(hrfp_plus_feat)
            # Resize to match dec3 spatial dimensions
            hrfp_plus_feat = F.interpolate(
                hrfp_plus_feat,
                size=dec3.shape[2:],
                mode='bilinear', align_corners=False)
            dec3 = dec3 + hrfp_plus_feat

        dec4   = self.decoder.up4(dec3)             # [B,  32, H,   W  ]
        output = self.decoder.final_conv(dec4)      # [B, num_classes, H, W]

        # Safety: guarantee output matches input resolution
        if output.shape[2] != H or output.shape[3] != W:
            output = F.interpolate(
                output, size=(H, W), mode='bilinear', align_corners=False)

        return output

    # ------------------------------------------------------------------
    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Binary predictions [B, H, W] ∈ {0, 1}."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


# ======================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Testing DINOv3ShadowDetectorMRFP')
    print('=' * 60)

    for variant, use_plus in [('MRFP', False), ('MRFP+', True)]:
        print(f'\n--- {variant} ---')
        model = DINOv3ShadowDetectorMRFP(
            num_classes=2,
            model_name='dinov3_vits16',
            weights_path=None,
            pretrained=True,
            use_mrfp_plus=use_plus,
        )

        x = torch.randn(2, 3, 384, 384)

        model.train()
        out_train = model(x)
        print(f'Train output: {out_train.shape}')

        model.eval()
        out_eval = model(x)
        print(f'Eval  output: {out_eval.shape}')

        preds = model.get_predictions(x)
        print(f'Predictions:  {preds.shape}  unique={torch.unique(preds).tolist()}')