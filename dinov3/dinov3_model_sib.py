"""
DINOv3 + SIB Model for Shadow Detection

Integrates the Spectral Information Bottleneck between the DINOv3
backbone (encoder) and a lightweight progressive upsampling decoder.

Architecture flow:
    Image [B,3,384,384]
      → DINOv3 Backbone → 4 feature maps [B,384,24,24] each
      → Concatenate → [B,1536,24,24]
      → SIB module → [B,384,24,24]
      → Decoder upsampling → [B, num_classes, 384, 384]

The SIB components (HAAR, VIB, AUG, AdaptiveBeta, PassthroughGate,
ModuleBypass) are independently toggleable via constructor flags,
enabling clean ablation studies.

Ablation flags (§5.3):
  disable_content_vib — A1: skip content VIB on F_LL
  symmetric_vib       — A3: high-β VIB on LL, LH, HL
  aug_all_subbands    — A6: augment all subbands
  vib_on_hl_only      — A10: content VIB on F_HL (wrong subband)

NEW (diagnostic-motivated additions, parity with MAMNet/OGLANet):
  use_cacr      — when True (and training), the SIB module returns a
                  pre-augmentation snapshot in sib_losses['pre_aug_bottleneck'].
                  This wrapper then runs a SECOND decoder pass on that
                  snapshot under no_grad and stores the result in
                  sib_losses['ref_logits'] for the training loop's CACR
                  loss to consume.

  use_ce_aurc   — flag-only here; the actual CE-AURC loss lives in
                  utils/losses.py and is driven by the training script.

  use_tent      — flag-only here; TENT adaptation happens in the test
                  loop in train_dinov3_sib.py via the helpers in sib.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import ConvBlock
from sib import SIB


class DINOv3ShadowDetectorSIB(nn.Module):
    """
    DINOv3-based shadow detector with SIB at the encoder-decoder interface.

    Args:
        num_classes:           output segmentation classes (default 2).
        model_name:            DINOv3 variant.
        weights_path:          path to DINOv3 pretrained backbone weights.
        pretrained:            load DINOv3 pretrained weights.
        frozen_stages:         number of backbone stages to freeze (-1 = all).
        use_haar:              enable Haar wavelet decomposition in SIB.
        use_vib:               enable VIB in SIB.
        use_content_aug:       enable content augmentation in SIB.
        adaptive_beta:         enable intensity-adaptive β in VIB.
        use_passthrough_gate:  enable learned passthrough gate on VIB output.
        use_module_bypass:     enable module-level residual bypass gate.
        disable_content_vib:   A1 ablation — skip content VIB on F_LL.
        symmetric_vib:         A3 ablation — high-β VIB on LL, LH, HL.
        aug_all_subbands:      A6 ablation — augment all subbands.
        vib_on_hl_only:        A10 ablation — content VIB on F_HL (wrong subband).
        num_domains:           number of source cities during training.
        vib_beta_content:      β for content VIB.
        vib_beta_edge:         β for edge VIB.
        vib_beta_scale:        adaptive β range.
        aug_sigma_style:       content aug style perturbation strength.
        aug_sigma_shift:       content aug shift perturbation strength.
        aug_p_aug:             probability of style perturbation.
        aug_p_mix:             probability of cross-domain mixing.
        use_cacr:              NEW — enable CACR ref-path snapshot.
        use_ce_aurc:           NEW — flag for CE-AURC loss (driven externally).
        use_tent:              NEW — flag for TENT (driven externally in test).
    """

    def __init__(
        self,
        num_classes=2,
        model_name='dinov3_vits16',
        weights_path=None,
        pretrained=True,
        frozen_stages=-1,
        # SIB toggles
        use_haar=True,
        use_vib=True,
        use_content_aug=True,
        adaptive_beta=True,
        use_passthrough_gate=False,
        use_module_bypass=False,
        # SIB ablation flags (§5.3)
        disable_content_vib=False,
        symmetric_vib=False,
        aug_all_subbands=False,
        vib_on_hl_only=False,
        # SIB hyper-parameters
        num_domains=2,
        vib_beta_content=0.01,
        vib_beta_edge=0.0001,
        vib_beta_scale=0.02,
        aug_sigma_style=0.25,
        aug_sigma_shift=0.15,
        aug_p_aug=0.5,
        aug_p_mix=0.3,
        # NEW: Diagnostic-motivated module flags
        use_cacr=False,
        use_ce_aurc=False,
        use_tent=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name

        # NEW: Store diagnostic-module flags
        self.use_cacr    = use_cacr
        self.use_ce_aurc = use_ce_aurc
        self.use_tent    = use_tent

        # ---- Backbone (frozen DINOv2/v3 ViT) ----
        print('Initialising DINOv3 backbone …')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages,
        )
        embed_dim = self.backbone.embed_dim
        num_feature_blocks = len(self.backbone.feature_blocks)
        in_channels = embed_dim * num_feature_blocks   # 1536 for ViT-S

        # ---- SIB ----
        print('Initialising SIB …')
        self.sib = SIB(
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_haar=use_haar,
            use_vib=use_vib,
            use_content_aug=use_content_aug,
            adaptive_beta=adaptive_beta,
            use_passthrough_gate=use_passthrough_gate,
            use_module_bypass=use_module_bypass,
            # Ablation flags
            disable_content_vib=disable_content_vib,
            symmetric_vib=symmetric_vib,
            aug_all_subbands=aug_all_subbands,
            vib_on_hl_only=vib_on_hl_only,
            # Hyper-parameters
            num_domains=num_domains,
            vib_beta_content=vib_beta_content,
            vib_beta_edge=vib_beta_edge,
            vib_beta_scale=vib_beta_scale,
            aug_sigma_style=aug_sigma_style,
            aug_sigma_shift=aug_sigma_shift,
            aug_p_aug=aug_p_aug,
            aug_p_mix=aug_p_mix,
        )

        # NEW: When CACR is on, instruct SIB to snapshot pre-aug bottleneck
        # during training forwards.
        if self.use_cacr:
            self.sib.return_pre_aug = True

        # ---- Decoder (upsampling stages — fusion handled by SIB) ----
        print('Initialising decoder …')
        dec = [embed_dim, 256, 128, 64, 32]

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dec[0], dec[1], 2, stride=2),
            nn.BatchNorm2d(dec[1]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[1], dec[1]),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(dec[1], dec[2], 2, stride=2),
            nn.BatchNorm2d(dec[2]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[2], dec[2]),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(dec[2], dec[3], 2, stride=2),
            nn.BatchNorm2d(dec[3]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[3], dec[3]),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(dec[3], dec[4], 2, stride=2),
            nn.BatchNorm2d(dec[4]),
            nn.ReLU(inplace=True),
            ConvBlock(dec[4], dec[4]),
        )
        self.final_conv = nn.Conv2d(dec[4], num_classes, kernel_size=1)

        # ---- Summary ----
        total  = sum(p.numel() for p in self.parameters())
        train_ = sum(p.numel() for p in self.parameters() if p.requires_grad)
        bb     = sum(p.numel() for p in self.backbone.parameters())
        sib_p  = sum(p.numel() for p in self.sib.parameters())
        dec_p  = (sum(p.numel() for p in self.up1.parameters())
                + sum(p.numel() for p in self.up2.parameters())
                + sum(p.numel() for p in self.up3.parameters())
                + sum(p.numel() for p in self.up4.parameters())
                + sum(p.numel() for p in self.final_conv.parameters()))
        print(f'\nDINOv3-SIB Shadow Detector:')
        print(f'  Total params:     {total:,}')
        print(f'  Trainable params: {train_:,}')
        print(f'  Backbone:         {bb:,}')
        print(f'  SIB:              {sib_p:,}')
        print(f'  Decoder:          {dec_p:,}')
        print(f'  PassthroughGate:  {"ON" if use_passthrough_gate else "OFF"}')
        print(f'  ModuleBypass:     {"ON" if use_module_bypass else "OFF"}')
        ablation_flags = []
        if disable_content_vib: ablation_flags.append('A1:NoContentVIB')
        if symmetric_vib:       ablation_flags.append('A3:SymmetricVIB')
        if aug_all_subbands:    ablation_flags.append('A6:AugAllSubbands')
        if vib_on_hl_only:      ablation_flags.append('A10:VIBonHL')
        if ablation_flags:
            print(f'  Ablation:         {", ".join(ablation_flags)}')
        # NEW: log diagnostic-module flags
        diag_flags = []
        if use_cacr:    diag_flags.append('CACR')
        if use_ce_aurc: diag_flags.append('CE-AURC')
        if use_tent:    diag_flags.append('TENT')
        if diag_flags:
            print(f'  NewModules:       {", ".join(diag_flags)}')

    # ------------------------------------------------------------------
    def _decode(self, task_feat, H, W):
        """
        Run the upsampling decoder on bottleneck features.

        Factored out so the CACR reference path can re-use it under
        no_grad without duplicating the upsampling logic.
        """
        out = self.up1(task_feat)
        out = self.up2(out)
        out = self.up3(out)
        out = self.up4(out)
        out = self.final_conv(out)
        if out.shape[2] != H or out.shape[3] != W:
            out = F.interpolate(out, size=(H, W),
                                mode='bilinear', align_corners=False)
        return out

    # ------------------------------------------------------------------
    def forward(self, x, intensity_map=None, city_ids=None,
                vib_warmup_factor=1.0):
        """
        Args:
            x:                  [B, 3, H, W]
            intensity_map:      [B, 1, H, W] in [0,1] (optional)
            city_ids:           [B] int64 (optional)
            vib_warmup_factor:  float in [0,1] for VIB warmup.
        Returns:
            logits:     [B, num_classes, H, W]
            sib_losses: dict.  When use_cacr and training and the SIB
                        produced a pre-aug snapshot, this dict will
                        also contain 'ref_logits' (full-resolution
                        decoder output for the un-augmented path,
                        detached and produced under no_grad).
        """
        B, C, H, W = x.shape

        # 1. Encoder
        features = self.backbone(x)

        feat_concat = torch.cat([
            features['feat_block3'],
            features['feat_block6'],
            features['feat_block9'],
            features['feat_block11'],
        ], dim=1)

        # 2. SIB
        task_feat, sib_losses = self.sib(
            feat_concat, intensity_map, city_ids,
            vib_warmup_factor=vib_warmup_factor,
        )

        # 3. Decoder upsampling (main / augmented path)
        out = self._decode(task_feat, H, W)

        # 4. NEW: CACR reference path
        # If a pre-aug snapshot is available, run a second decoder pass
        # on it under no_grad and store the result for the training loop
        # to consume.  The snapshot is popped from sib_losses so it
        # doesn't leak into KL-loss accumulators downstream.
        if self.use_cacr and self.training and 'pre_aug_bottleneck' in sib_losses:
            pre_aug_feat = sib_losses.pop('pre_aug_bottleneck')
            with torch.no_grad():
                ref_out = self._decode(pre_aug_feat, H, W)
            sib_losses['ref_logits'] = ref_out.detach()

        return out, sib_losses

    # ------------------------------------------------------------------
    def get_predictions(self, x, intensity_map=None):
        """Inference helper — returns [B, H, W] integer predictions."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, intensity_map)
            return torch.argmax(logits, dim=1)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print('Backbone frozen.')

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print('Backbone unfrozen.')


# ======================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Testing DINOv3-SIB Shadow Detector')
    print('=' * 60)

    try:
        model = DINOv3ShadowDetectorSIB(
            num_classes=2,
            model_name='dinov3_vits16',
            pretrained=True,
            use_haar=True,
            use_vib=True,
            use_content_aug=True,
            adaptive_beta=True,
            use_passthrough_gate=True,
            use_module_bypass=True,
            num_domains=2,
            use_cacr=True,
            use_ce_aurc=True,
            use_tent=False,
        )

        x   = torch.randn(4, 3, 384, 384)
        im  = torch.rand(4, 1, 384, 384)
        cid = torch.tensor([0, 1, 0, 1])

        model.train()
        out, losses = model(x, im, cid, vib_warmup_factor=0.5)
        print(f'\nTrain mode  →  output {out.shape}')
        for k, v in losses.items():
            if isinstance(v, torch.Tensor) and v.dim() == 0:
                print(f'  {k}: {v.item():.6f}')
            elif isinstance(v, torch.Tensor):
                print(f'  {k}: shape={v.shape}  mean={v.mean().item():.4f}')

        if 'ref_logits' in losses:
            print(f'  ref_logits shape: {losses["ref_logits"].shape}')

        model.eval()
        out_e, losses_e = model(x)
        print(f'\nEval mode   →  output {out_e.shape}')

        print('\n✓ All tests passed!')

    except Exception as e:
        print(f'\nTest failed: {e}')
        import traceback
        traceback.print_exc()