"""
MAMNet + MRFP: Shadow Detection with Multi-Resolution Feature Perturbation
============================================================================
Integrates MRFP (Udupa et al., CVPR 2024) into the MAMNet shadow-detection
architecture for cross-city domain generalization.

Variants (controlled by flags):
    MRFP   = {HRFP, NP+}
    MRFP+  = {HRFP, HRFP+, NP+}   ← best-performing (paper Table 5)

Training:
    • HRFP perturbs stage-0 encoder features (high-frequency perturbation).
    • NP+ perturbs deepest encoder features (low-frequency / style perturbation).
    • HRFP+ adds overcomplete max-resolution feature to the penultimate
      decoder layer (O₂ branch).
    • Each module is toggled independently with probability p (default 0.5).
    • Three training-time instance-norm layers on feat1/2/3 (MRFP+ only,
      following the paper's reference to IBN-Net [29] / RobustNet [7]).

Inference:
    Completely identical to the base MAMNet — all perturbation branches
    are disabled, no extra parameters or compute.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResNet34Encoder
from .encoder_4ch import ResNet34Encoder4Ch
from .mscaf import MSCAF
from .decoder import Decoder
from .auxiliary import AuxiliaryModule
from .mrfp_modules import HRFPModule, NormalizedPerturbation


class MAMNetMRFP(nn.Module):
    """
    MAMNet with MRFP / MRFP+ training-time perturbation.

    Parameters
    ----------
    num_classes     : int   — output classes (2 for shadow/non-shadow)
    pretrained      : bool  — ImageNet-pretrained ResNet-34 encoder
    use_aux         : bool  — auxiliary deep-supervision branches
    use_contrast    : bool  — 4-channel RGBC input (Task 2)
    use_mrfp_plus   : bool  — True → MRFP+ (HRFP+HRFP++NP+);
                               False → MRFP (HRFP+NP+)
    hrfp_prob       : float — probability of applying HRFP each step
    np_prob         : float — probability of applying NP+ each step
    hrfp_plus_prob  : float — probability of applying HRFP+ each step
    bn_std          : float — std of Gaussian for HRFP batch-norm init
    """

    def __init__(self, num_classes=2, pretrained=True, use_aux=True,
                 use_contrast=False, use_mrfp_plus=True,
                 hrfp_prob=0.5, np_prob=0.5, hrfp_plus_prob=0.5,
                 bn_std=0.5):
        super().__init__()

        self.num_classes    = num_classes
        self.use_aux        = use_aux
        self.use_contrast   = use_contrast
        self.use_mrfp_plus  = use_mrfp_plus
        self.hrfp_prob      = hrfp_prob
        self.np_prob        = np_prob
        self.hrfp_plus_prob = hrfp_plus_prob

        # ---- Encoder ----
        if use_contrast:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)
            print("Using 4-channel encoder (RGB + Contrast)")
        else:
            self.encoder = ResNet34Encoder(pretrained=pretrained)

        # ---- MSCAF ----
        self.mscaf = MSCAF(in_channels=512)

        # ---- Decoder ----
        self.decoder = Decoder(num_classes=num_classes)

        # ---- Auxiliary deep-supervision ----
        if use_aux:
            self.aux_module = AuxiliaryModule(num_classes=num_classes,
                                              dropout_rate=0.3)

        # ---- MRFP perturbation modules (frozen, no trainable params) ----
        # HRFP: applied to stage-0 output (64ch, H×W)
        self.hrfp = HRFPModule(in_channels=64, bn_std=bn_std)

        # NP+: applied to feat4 (512ch, H/8×W/8) — deepest encoder features
        self.np_plus = NormalizedPerturbation()

        # ---- Instance-norm layers for MRFP+ (training-time only) ----
        # Paper §3.3 HRFP+: "three instance normalization layers [29]
        # have been adopted in a similar fashion as [7]."
        # Applied functionally during training; no-op at eval.
        # affine=False → zero extra learnable params.
        if use_mrfp_plus:
            self.in_norm1 = nn.InstanceNorm2d(64,  affine=False)
            self.in_norm2 = nn.InstanceNorm2d(128, affine=False)
            self.in_norm3 = nn.InstanceNorm2d(256, affine=False)

        variant = "MRFP+" if use_mrfp_plus else "MRFP"
        print(f"[{variant}] HRFP p={hrfp_prob}  NP+ p={np_prob}"
              + (f"  HRFP+ p={hrfp_plus_prob}" if use_mrfp_plus else "")
              + f"  BN σ={bn_std}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Training  → dict {'main', 'aux1', 'aux2', 'aux3'}
        Inference → Tensor [B, num_classes, H, W]
        """
        B, _, H, W = x.size()

        # ============================================================
        # 1.  Encoder — with perturbation injection points
        # ============================================================

        # Stage 0: conv1 → bn1 → relu  (64 ch, H×W)
        stage0 = self.encoder.relu(
            self.encoder.bn1(self.encoder.conv1(x)))

        # ---- HRFP on stage 0 (O₁ branch) ----
        # HRFP weights are frozen & re-randomized each call → no gradients
        # needed through the overcomplete autoencoder.  torch.no_grad()
        # prevents storing activations for backprop (saves ~30 GB at 384²).
        # .detach() severs the graph so gradients flow through the *add*
        # into stage0 but not into the HRFP branch.
        hrfp_output = None
        if self.training and random.random() < self.hrfp_prob:
            with torch.no_grad():
                hrfp_output = self.hrfp(stage0)
            stage0 = stage0 + hrfp_output['perturbation'].detach()

        # Continue through encoder layers
        feat1 = self.encoder.layer1(stage0)     # 64,  H
        feat2 = self.encoder.layer2(feat1)      # 128, H/2
        feat3 = self.encoder.layer3(feat2)      # 256, H/4
        feat4 = self.encoder.layer4(feat3)      # 512, H/8

        # ---- Instance norm (MRFP+ only, training only) ----
        if self.use_mrfp_plus and self.training:
            feat1 = self.in_norm1(feat1)
            feat2 = self.in_norm2(feat2)
            feat3 = self.in_norm3(feat3)

        # ---- NP+ on feat4 (style perturbation) ----
        if self.training and random.random() < self.np_prob:
            feat4 = self.np_plus(feat4)

        feat5 = self.encoder.downsample(feat4)  # 512, H/16

        enc_features = {
            'feat1': feat1, 'feat2': feat2,
            'feat3': feat3, 'feat4': feat4,
            'feat5': feat5,
        }

        # ============================================================
        # 2.  MSCAF on deepest features
        # ============================================================
        mscaf_out = self.mscaf(enc_features['feat5'])

        # ============================================================
        # 3.  Decoder — stepped through manually to inject HRFP+
        # ============================================================
        dec1 = self.decoder.decoder1(enc_features['feat4'], mscaf_out)
        dec2 = self.decoder.decoder2(enc_features['feat3'], dec1)
        dec3 = self.decoder.decoder3(enc_features['feat2'], dec2)

        # ---- HRFP+ on penultimate decoder feature (O₂ branch) ----
        if (self.use_mrfp_plus
                and self.training
                and hrfp_output is not None
                and random.random() < self.hrfp_plus_prob):
            hrfp_plus_feat = hrfp_output['max_resolution_feat'].detach()
            hrfp_plus_feat = F.interpolate(
                hrfp_plus_feat,
                size=dec3.shape[2:],
                mode='bilinear', align_corners=False)
            dec3 = dec3 + hrfp_plus_feat

        dec4 = self.decoder.decoder4(enc_features['feat1'], dec3)
        main_out = self.decoder.classifier(dec4)    # [B, C, H', W']

        # ============================================================
        # 4.  Auxiliary branches (training only)
        # ============================================================
        if self.use_aux and self.training:
            aux_outputs = self.aux_module(
                dec1, dec2, dec3, target_size=(H, W))
            return {
                'main': main_out,
                'aux1': aux_outputs['aux1'],
                'aux2': aux_outputs['aux2'],
                'aux3': aux_outputs['aux3'],
            }

        return main_out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_predictions(self, x):
        """Binary predictions [B, H, W] ∈ {0, 1}."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)


# ======================================================================
if __name__ == "__main__":
    model = MAMNetMRFP(num_classes=2, pretrained=True, use_aux=True,
                       use_contrast=False, use_mrfp_plus=True)

    # Training mode
    model.train()
    x = torch.randn(2, 3, 256, 256)
    out_train = model(x)
    print("Training outputs:")
    for k, v in out_train.items():
        print(f"  {k}: {v.shape}")

    # Inference mode
    model.eval()
    out_eval = model(x)
    print(f"\nInference output: {out_eval.shape}")

    preds = model.get_predictions(x)
    print(f"Predictions: {preds.shape}  unique: {torch.unique(preds)}")

    total   = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {train_p:,}")
    print(f"Frozen (HRFP):    {total - train_p:,}")