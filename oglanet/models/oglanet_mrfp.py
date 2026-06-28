"""
OGLANet + MRFP: Shadow Detection with Multi-Resolution Feature Perturbation
============================================================================
Integrates MRFP (Udupa et al., CVPR 2024) into OGLANet for cross-city
shadow-detection generalization.

Option A diagnostic (active):
    Instance Normalization layers have been REMOVED from this version.
    Motivation: GFEM (self-attention) inside each GLAM stage builds global
    channel statistics deliberately.  Applying InstanceNorm2d(affine=False)
    after GFEM outputs destroys those statistics with no learnable recovery,
    which is the primary suspected cause of the mIOU < 1% class-collapse.
    IN was not part of plain MRFP in the original paper — it was introduced
    only alongside HRFP+ (§3.3, "three instance normalization layers [29]
    have been adopted in a similar fashion as [7]"), and the paper validated
    it on plain ResNet encoders, not self-attention-based ones.

    This version therefore implements:
        MRFP  = {HRFP, NP+}
        MRFP+ = {HRFP, HRFP+, NP+}   (NO instance norm in either variant)

    If mIOU recovers, IN+GFEM interaction is confirmed as the root cause
    and we can explore affine=True IN or other placements in a follow-up.

Injection points (unchanged from original design, see rationale below):

    HRFP  — after stem (conv1→bn1→relu), BEFORE maxpool.
              Stage-0 output at H/2, 64ch — matching the paper's exact
              description and the MAMNet implementation convention.
              The stem is a plain conv+BN+relu (no self-attention), so HRFP
              is safe here regardless of the GLAM architecture.

    NP+   — after glam4 output (512ch, H/32).
              Deepest ResNet-equivalent encoder stage, analogous to feat4 in
              MAMNet.  If mIOU still collapses after removing IN, moving NP+
              is the next diagnostic step (Option B).

    HRFP+ — added to s1_d_up (64ch, 384×384) via O2 branch.
              Finest-resolution decoder feature before OAM.  HRFP input is
              at H/2 (64ch) so max_resolution_feat aligns after interpolate.

Training:
    All perturbation modules toggled independently with probability p.
    HRFP weights re-randomized every forward call (never trained, frozen).

Inference:
    Completely identical to base OGLANet — all perturbation branches off.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glam import GLAMEncoder
from .dffm import DFFM
from .decoder import Decoder
from .oam import OAM
from .mrfp_modules import HRFPModule, NormalizedPerturbation


class OGLANetMRFP(nn.Module):
    """
    OGLANet with MRFP / MRFP+ training-time perturbation.
    Instance Normalization removed (Option A diagnostic).

    Parameters
    ----------
    num_classes     : int   — output classes (2 for shadow/non-shadow)
    pretrained      : bool  — ImageNet-pretrained ResNet-34 encoder
    img_size        : int   — input spatial size (default 384)
    use_contrast    : bool  — 4-channel RGBC input (Task 2)
    use_mrfp_plus   : bool  — True  → MRFP+ (HRFP + HRFP+ + NP+, no IN)
                               False → MRFP  (HRFP + NP+)
    hrfp_prob       : float — probability of applying HRFP each step
    np_prob         : float — probability of applying NP+ each step
    hrfp_plus_prob  : float — probability of applying HRFP+ each step
    bn_std          : float — std of Gaussian for HRFP batch-norm init
    """

    def __init__(self, num_classes=2, pretrained=True, img_size=384,
                 use_contrast=False, use_mrfp_plus=True,
                 hrfp_prob=0.5, np_prob=0.5, hrfp_plus_prob=0.5,
                 bn_std=0.5):
        super().__init__()

        self.num_classes    = num_classes
        self.img_size       = img_size
        self.use_contrast   = use_contrast
        self.use_mrfp_plus  = use_mrfp_plus
        self.hrfp_prob      = hrfp_prob
        self.np_prob        = np_prob
        self.hrfp_plus_prob = hrfp_plus_prob

        # ---- GLAMEncoder ----
        # We access its internal submodules directly to inject HRFP at the
        # stem→maxpool boundary (stage-0 output, H/2, 64ch).
        self.encoder = GLAMEncoder(pretrained=pretrained,
                                   use_contrast=use_contrast)

        # ---- Standard OGLANet modules ----
        self.dffm    = DFFM()
        self.decoder = Decoder(target_size=(img_size, img_size))
        self.oam     = OAM(num_classes=num_classes,
                           target_size=(img_size, img_size))

        # ---- MRFP perturbation modules ----
        # HRFP: stem output (64ch, H/2). Frozen, re-randomized each call.
        self.hrfp    = HRFPModule(in_channels=64, bn_std=bn_std)

        # NP+: glam4 output (512ch, H/32). No learnable parameters.
        self.np_plus = NormalizedPerturbation()

        # NOTE: Instance Normalization layers intentionally omitted.
        # See module docstring for full rationale.

        variant = 'MRFP+ (no IN)' if use_mrfp_plus else 'MRFP (no IN)'
        print(f'[OGLANet {variant}]  '
              f'HRFP p={hrfp_prob}  NP+ p={np_prob}'
              + (f'  HRFP+ p={hrfp_plus_prob}' if use_mrfp_plus else '')
              + f'  BN σ={bn_std}')
        print('[OGLANet MRFP] Instance Normalization: DISABLED (Option A)')

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Training  → dict {'p1', 'p2', 'p3', 'p4', 'p5', 'p6'}
                    each [B, num_classes, H, W]
        Inference → Tensor [B, num_classes, H, W]  (P6 only)
        """
        B, _, H, W = x.size()
        enc = self.encoder.resnet_encoder   # ResNet34Encoder or ResNet34Encoder4Ch

        # ==============================================================
        # 1. Stem: conv1 → bn1 → relu   (64ch, H/2)
        #    HRFP is injected here — the stem is a plain conv+BN+relu
        #    with no self-attention, identical to the DeepLabv3+ stage-0
        #    described in the paper.  Safe regardless of GLAM architecture.
        # ==============================================================
        stem = enc.relu(enc.bn1(enc.conv1(x)))   # [B, 64, H/2, W/2]

        # ---- HRFP on stem (O1 branch) ----
        # torch.no_grad() avoids storing activations through the frozen
        # random autoencoder.  .detach() severs the graph so gradients
        # flow through the *add* into stem, not into the HRFP branch.
        hrfp_output = None
        if self.training and random.random() < self.hrfp_prob:
            with torch.no_grad():
                hrfp_output = self.hrfp(stem)
            stem = stem + hrfp_output['perturbation'].detach()

        # Maxpool → [B, 64, H/4, W/4]
        x_enc = enc.maxpool(stem)

        # ==============================================================
        # 2. GLAM stages — no IN applied anywhere (Option A)
        # ==============================================================
        feat1 = self.encoder.glam1(x_enc)   # [B, 64,  ~H/4,  ~W/4]
        feat2 = self.encoder.glam2(feat1)   # [B, 128, ~H/8,  ~W/8]
        feat3 = self.encoder.glam3(feat2)   # [B, 256, ~H/16, ~W/16]
        feat4 = self.encoder.glam4(feat3)   # [B, 512, ~H/32, ~W/32]

        # ---- NP+ on glam4 output (style perturbation) ----
        # glam4 is the deepest ResNet-equivalent stage (512ch), analogous
        # to feat4 in MAMNet.  If collapse persists with IN removed,
        # moving NP+ is the next diagnostic step (Option B).
        if self.training and random.random() < self.np_prob:
            feat4 = self.np_plus(feat4)

        # glam5: OGLANet-specific extra stage — no MRFP injection here
        feat5 = self.encoder.glam5_conv(feat4)   # [B, 1024, ~H/64, ~W/64]
        feat5 = self.encoder.glam5_gfem(feat5)

        encoder_features = {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'feat5': feat5,
        }

        # ==============================================================
        # 3. DFFM
        # ==============================================================
        dffm_features = self.dffm(encoder_features)

        # ==============================================================
        # 4. Decoder
        # ==============================================================
        decoder_features = self.decoder(dffm_features)

        # ---- HRFP+ on s1_d_up (O2 branch) ----
        # s1_d_up (64ch, 384×384) is the finest-resolution decoder output
        # before OAM.  HRFP input is at H/2 (64ch) so max_resolution_feat
        # aligns spatially and channel-wise after interpolate.
        # Note: HRFP+ is retained even with IN removed — the two are
        # independent additions in the paper (§3.3).  Only IN is dropped.
        if (self.use_mrfp_plus
                and self.training
                and hrfp_output is not None
                and random.random() < self.hrfp_plus_prob):
            hrfp_plus_feat = hrfp_output['max_resolution_feat'].detach()
            hrfp_plus_feat = F.interpolate(
                hrfp_plus_feat,
                size=decoder_features['s1_d_up'].shape[2:],
                mode='bilinear', align_corners=False)
            decoder_features['s1_d_up'] = (decoder_features['s1_d_up']
                                           + hrfp_plus_feat)

        # ==============================================================
        # 5. OAM — 6-output prediction
        # ==============================================================
        predictions = self.oam(decoder_features)
        # predictions = {'p1', 'p2', 'p3', 'p4', 'p5', 'p6'}

        if self.training:
            return predictions          # all 6 for deep supervision
        else:
            return predictions['p6']   # P6 only at inference

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
if __name__ == '__main__':
    model = OGLANetMRFP(num_classes=2, pretrained=False, img_size=384,
                        use_contrast=False, use_mrfp_plus=True)

    x = torch.randn(2, 3, 384, 384)

    # Training mode
    model.train()
    out_train = model(x)
    print('Training outputs:')
    for k, v in out_train.items():
        print(f'  {k}: {v.shape}')

    # Inference mode
    model.eval()
    out_eval = model(x)
    print(f'\nInference output: {out_eval.shape}')

    preds = model.get_predictions(x)
    print(f'Predictions: {preds.shape}  unique: {torch.unique(preds)}')

    total   = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nTotal params:     {total:,}')
    print(f'Trainable params: {train_p:,}')
    print(f'Frozen (HRFP):    {total - train_p:,}')