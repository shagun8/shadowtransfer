"""
MAMNet with Spectral Information Bottleneck (SIB).

Architecture:
  ResNet-34 encoder → MSCAF(feat5) → [SIB] → Decoder(CCA) with skip connections
  Optional SAG on skip connections, optional multi-scale SIB at encoder stages.
  Optional module-level bypass gate wrapping the entire SIB pipeline.

NEW — Diagnostic-motivated additions (§4.3 orphan coverage):
  - CACR: Class-Asymmetric Confidence Regularizer. When enabled, the model
    also runs a reference decoder pass on pre-augmentation bottleneck features
    and returns both logit sets for asymmetric penalty computation.
  - CE-AURC: Cross-entropy AURC auxiliary loss flag (loss computed externally).
  - TENT: Test-time entropy minimization flag (adaptation done externally).

Placement mirrors mamnet_ddib.py (Option B) but replaces DDIB with SIB.
All SIB components are independently toggleable for ablation (M1–M15, A1–A10).

Architecture flow:
    Image [B, 3/4, 384, 384]
      → ResNet-34 Encoder  → enc_features dict {feat1..feat5}
      → MSCAF(feat5)       → [B, 512, H/16, W/16]
      → SIB                → [B, 512, H/16, W/16]  (bottleneck)
      → ModuleBypassGate   → α·SIB + (1−α)·MSCAF   (optional)
      → SAG(feat1..feat4)  → filtered skip connections  (optional)
      → Decoder (CCA with skips)
      → [B, num_classes, H, W]

    When CACR is enabled (training only):
      → SIB also returns pre_aug_bottleneck
      → Decoder runs a second pass on pre_aug_bottleneck → ref_logits
      → Both main_logits and ref_logits returned for CACR loss computation

Usage:
    model = MAMNetSIB(
        num_classes=2, in_channels=3,
        use_haar=True, use_vib=True, use_content_aug=True,
        adaptive_beta=True, use_sag=True, use_multiscale_sib=False,
        use_passthrough_gate=False, use_module_bypass=True,
        use_cacr=True, use_ce_aurc=True,
    )
    outputs, sib_losses = model(images, intensity_map=imap)
    # sib_losses may contain 'ref_logits' when CACR is enabled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Base MAMNet components — same imports as mamnet_ddib.py
from .encoder import ResNet34Encoder
from .encoder_4ch import ResNet34Encoder4Ch
from .mscaf import MSCAF
from .decoder import Decoder
from .auxiliary import AuxiliaryModule

# SIB components (copied to mamnet/models/ alongside this file)
from .sib import (
    SpectralInformationBottleneck,
    SkipAttentionGate,
    MultiScaleSIB,
    ModuleBypassGate,
)


class MAMNetSIB(nn.Module):
    """
    MAMNet + SIB for cross-location shadow detection.

    Args:
        num_classes: Output classes (2 for shadow/non-shadow).
        in_channels: Input channels (3 for RGB, 4 for RGB+contrast).
        pretrained: Use ImageNet-pretrained encoder.
        use_aux: Enable auxiliary branches for deep supervision.
        use_haar: Enable Haar wavelet decomposition in SIB.
        use_vib: Enable VIB in SIB.
        use_content_aug: Enable content augmentation on LL.
        adaptive_beta: Enable intensity-adaptive β.
        use_passthrough_gate: Enable learned passthrough gate on VIB output.
        use_module_bypass: Enable module-level bypass gate wrapping entire SIB.
        beta_content: β weight for content (LL) VIB KL.
        beta_edge: β weight for edge (LH/HL) VIB KL.
        noise_scale: Content augmentation noise scale.
        beta_max_multiplier: Max multiplier for intensity-adaptive β.
        use_sag: Enable Skip Attention Gates on skip connections.
        use_multiscale_sib: Enable lightweight SIB at each encoder stage.
        multiscale_beta_base: Base β for multi-scale SIB.
        use_cacr: Enable CACR reference path (returns pre-aug ref logits).
        use_ce_aurc: Flag only — CE-AURC loss computed in training loop.
        use_tent: Flag only — TENT adaptation done in test loop.

    Ablation-only args (all default False):
        symmetric_vib: A3 — Use β_content for edge VIB too.
        aug_all_subbands: A5 — Augment LH/HL/HH too (MRFP+ analog).
        no_edge_vib: A9 — Skip VIB on LH/HL.
        vib_wrong_subband: A10 — Apply content VIB to HL only.
    """

    def __init__(self, num_classes=2, in_channels=3,
                 pretrained=True, use_aux=True,
                 use_haar=True, use_vib=True, use_content_aug=True,
                 adaptive_beta=True, use_passthrough_gate=False,
                 use_module_bypass=False,
                 beta_content=1e-3, beta_edge=1e-5,
                 noise_scale=0.1, beta_max_multiplier=3.0,
                 use_sag=False, use_multiscale_sib=False,
                 multiscale_beta_base=1e-4,
                 # New diagnostic modules
                 use_cacr=False,
                 use_ce_aurc=False,
                 use_tent=False,
                 # Ablation flags
                 symmetric_vib=False,
                 aug_all_subbands=False,
                 no_edge_vib=False,
                 vib_wrong_subband=False):
        super().__init__()

        self.use_sag = use_sag
        self.use_multiscale_sib = use_multiscale_sib
        self.use_aux = use_aux
        self.use_module_bypass = use_module_bypass
        self.use_cacr = use_cacr
        self.use_ce_aurc = use_ce_aurc
        self.use_tent = use_tent

        # ---- Encoder ----
        if in_channels <= 3:
            self.encoder = ResNet34Encoder(pretrained=pretrained)
        else:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)

        # ---- MSCAF Bottleneck ----
        self.mscaf = MSCAF(in_channels=512)

        # ---- SIB (after MSCAF, before decoder) ----
        # Enable return_pre_aug when CACR is active
        self.sib = SpectralInformationBottleneck(
            in_channels=512,
            embed_dim=512,
            use_haar=use_haar,
            use_vib=use_vib,
            use_content_aug=use_content_aug,
            adaptive_beta=adaptive_beta,
            use_passthrough_gate=use_passthrough_gate,
            beta_content=beta_content,
            beta_edge=beta_edge,
            noise_scale=noise_scale,
            beta_max_multiplier=beta_max_multiplier,
            return_pre_aug=use_cacr,
            # Ablation flags
            symmetric_vib=symmetric_vib,
            aug_all_subbands=aug_all_subbands,
            no_edge_vib=no_edge_vib,
            vib_wrong_subband=vib_wrong_subband,
        )

        # ---- Module Bypass Gate (wraps entire SIB pipeline) ----
        if use_module_bypass:
            self.module_bypass_gate = ModuleBypassGate(channels=512)

        # ---- Skip Attention Gates (SAG) on feat1–feat4 ----
        if use_sag:
            self.sag_gates = nn.ModuleList([
                SkipAttentionGate(64),   # feat1
                SkipAttentionGate(128),  # feat2
                SkipAttentionGate(256),  # feat3
                SkipAttentionGate(512),  # feat4
            ])
        else:
            self.sag_gates = None

        # ---- Multi-scale SIB on encoder stages ----
        if use_multiscale_sib:
            self.ms_sib = MultiScaleSIB(
                stage_channels=(64, 128, 256, 512),
                beta_base=multiscale_beta_base,
            )

        # ---- Decoder with CCA (same as mamnet_ddib.py) ----
        self.decoder = Decoder(num_classes=num_classes)

        # ---- Auxiliary branches (same as mamnet_ddib.py) ----
        if use_aux:
            self.aux_module = AuxiliaryModule(
                num_classes=num_classes, dropout_rate=0.3)

        # ---- Summary ----
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f'\nMAMNet-SIB Shadow Detector:')
        print(f'  Total params:     {total:,}')
        print(f'  Trainable params: {trainable:,}')
        print(f'  SIB config: haar={use_haar} vib={use_vib} '
              f'aug={use_content_aug} ab={adaptive_beta} '
              f'gate={use_passthrough_gate} mod_bypass={use_module_bypass}')
        print(f'  SAG={use_sag}  MultiScaleSIB={use_multiscale_sib}')
        print(f'  CACR={use_cacr}  CE-AURC={use_ce_aurc}  TENT={use_tent}')
        ablation_strs = []
        if symmetric_vib:
            ablation_strs.append('symmetric_vib')
        if aug_all_subbands:
            ablation_strs.append('aug_all_subbands')
        if no_edge_vib:
            ablation_strs.append('no_edge_vib')
        if vib_wrong_subband:
            ablation_strs.append('vib_wrong_subband')
        if ablation_strs:
            print(f'  Ablations: {", ".join(ablation_strs)}')

    def forward(self, x, intensity_map=None):
        """
        Args:
            x: Input images [B, C, H, W].
            intensity_map: [B, 1, H, W] normalised intensity for adaptive β.

        Returns:
            outputs: Dict with 'main' (+ 'aux1','aux2','aux3' during training).
                     Or just main tensor during eval.
            sib_losses: Dict with SIB loss components.
                        If module bypass is enabled, also contains
                        'bypass_alpha' [B] with per-sample gate values.
                        If CACR is enabled (training), also contains
                        'ref_logits' [B, num_classes, H, W] from the
                        pre-augmentation reference decoder pass.
        """
        B, _, H, W = x.size()
        sib_losses = {}

        # ---- 1. Encoder ----
        enc_features = self.encoder(x)

        # ---- 2. Multi-scale SIB on encoder features (optional) ----
        if self.use_multiscale_sib:
            feats_list = [enc_features['feat1'], enc_features['feat2'],
                          enc_features['feat3'], enc_features['feat4']]
            feats_list, ms_kl = self.ms_sib(feats_list)
            enc_features['feat1'] = feats_list[0]
            enc_features['feat2'] = feats_list[1]
            enc_features['feat3'] = feats_list[2]
            enc_features['feat4'] = feats_list[3]
            sib_losses['kl_multiscale'] = ms_kl

        # ---- 3. SAG on skip connections (optional) ----
        if self.sag_gates is not None:
            for i, key in enumerate(['feat1', 'feat2', 'feat3', 'feat4']):
                enc_features[key] = self.sag_gates[i](enc_features[key])

        # ---- 4. MSCAF bottleneck ----
        mscaf_out = self.mscaf(enc_features['feat5'])

        # ---- 5. SIB at bottleneck ----
        bottleneck_sib, sib_kl = self.sib(mscaf_out,
                                           intensity_map=intensity_map)
        sib_losses.update(sib_kl)

        # ---- 5b. Module Bypass Gate (optional) ----
        if self.use_module_bypass:
            bottleneck_sib, bypass_alpha = self.module_bypass_gate(
                bottleneck_sib, mscaf_out)
            sib_losses['bypass_alpha'] = bypass_alpha

        # ---- 6. Decoder with CCA and skip connections ----
        decoder_outputs = self.decoder(bottleneck_sib, enc_features)
        main_out = decoder_outputs['main']

        # ---- 6b. CACR reference decoder pass (training only) ────────
        #  Run a second decoder pass on the pre-augmentation bottleneck
        #  features to produce reference logits.  The asymmetric penalty
        #  between main_out and ref_logits is computed in the training
        #  loop via CACRLoss.
        #
        #  Key design choices:
        #   - pre_aug_bottleneck is detached → no gradient through ref path
        #   - Same skip connections (enc_features) used for both passes
        #   - Only 'main' output extracted (no aux branches for reference)
        # ─────────────────────────────────────────────────────────────
        if self.use_cacr and self.training:
            pre_aug_bneck = sib_kl.get('pre_aug_bottleneck', None)
            if pre_aug_bneck is not None:
                # Remove from sib_losses so it's not treated as a scalar loss
                sib_losses.pop('pre_aug_bottleneck', None)

                # Apply module bypass gate to reference path too (if active)
                if self.use_module_bypass:
                    pre_aug_bneck, _ = self.module_bypass_gate(
                        pre_aug_bneck, mscaf_out.detach())

                # Reference decoder pass (no gradient)
                with torch.no_grad():
                    ref_decoder_outputs = self.decoder(
                        pre_aug_bneck, enc_features)
                    ref_logits = ref_decoder_outputs['main']
                sib_losses['ref_logits'] = ref_logits

        # ---- 7. Auxiliary branches (training only) ----
        if self.use_aux and self.training:
            aux_outputs = self.aux_module(
                decoder_outputs['dec_feat1'],
                decoder_outputs['dec_feat2'],
                decoder_outputs['dec_feat3'],
                target_size=(H, W),
            )
            outputs = {
                'main': main_out,
                'aux1': aux_outputs['aux1'],
                'aux2': aux_outputs['aux2'],
                'aux3': aux_outputs['aux3'],
            }
            return outputs, sib_losses
        else:
            return main_out, sib_losses

    def get_predictions(self, x, intensity_map=None):
        """Inference helper — returns [B, H, W] integer predictions."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x, intensity_map)
            return torch.argmax(logits, dim=1)


def build_mamnet_sib(args):
    """
    Factory function to build MAMNetSIB from argparse args.

    Expected args attributes:
        use_haar, use_vib, use_content_aug, adaptive_beta,
        use_passthrough_gate, use_module_bypass,
        beta_content, beta_edge, noise_scale, beta_max_multiplier,
        use_sag, use_multiscale_sib, multiscale_beta_base,
        use_contrast (determines in_channels=3 or 4),
        symmetric_vib, aug_all_subbands, no_edge_vib, vib_wrong_subband,
        use_cacr, use_ce_aurc, use_tent
    """
    in_ch = 4 if getattr(args, 'use_contrast', False) else 3

    model = MAMNetSIB(
        num_classes=2,
        in_channels=in_ch,
        pretrained=True,
        use_aux=True,
        use_haar=getattr(args, 'use_haar', True),
        use_vib=getattr(args, 'use_vib', True),
        use_content_aug=getattr(args, 'use_content_aug', True),
        adaptive_beta=getattr(args, 'adaptive_beta', True),
        use_passthrough_gate=getattr(args, 'use_passthrough_gate', False),
        use_module_bypass=getattr(args, 'use_module_bypass', False),
        beta_content=getattr(args, 'beta_content', 1e-3),
        beta_edge=getattr(args, 'beta_edge', 1e-5),
        noise_scale=getattr(args, 'noise_scale', 0.1),
        beta_max_multiplier=getattr(args, 'beta_max_multiplier', 3.0),
        use_sag=getattr(args, 'use_sag', False),
        use_multiscale_sib=getattr(args, 'use_multiscale_sib', False),
        multiscale_beta_base=getattr(args, 'multiscale_beta_base', 1e-4),
        # New diagnostic modules
        use_cacr=getattr(args, 'use_cacr', False),
        use_ce_aurc=getattr(args, 'use_ce_aurc', False),
        use_tent=getattr(args, 'use_tent', False),
        # Ablation flags
        symmetric_vib=getattr(args, 'symmetric_vib', False),
        aug_all_subbands=getattr(args, 'aug_all_subbands', False),
        no_edge_vib=getattr(args, 'no_edge_vib', False),
        vib_wrong_subband=getattr(args, 'vib_wrong_subband', False),
    )

    return model


if __name__ == '__main__':
    import argparse
    args = argparse.Namespace(
        use_haar=True, use_vib=True, use_content_aug=True,
        adaptive_beta=True, use_passthrough_gate=False,
        use_module_bypass=True,
        beta_content=1e-3, beta_edge=1e-5,
        noise_scale=0.1, beta_max_multiplier=3.0,
        use_sag=True, use_multiscale_sib=True,
        multiscale_beta_base=1e-4, use_contrast=False,
        symmetric_vib=False, aug_all_subbands=False,
        no_edge_vib=False, vib_wrong_subband=False,
        use_cacr=True, use_ce_aurc=True, use_tent=False,
    )
    model = build_mamnet_sib(args)

    x = torch.randn(2, 3, 384, 384)
    imap = torch.rand(2, 1, 384, 384)

    model.train()
    outputs, losses = model(x, intensity_map=imap)
    print(f'\nTraining outputs:')
    for k, v in outputs.items():
        print(f'  {k}: {v.shape}')
    print(f'SIB losses:')
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            if v.dim() == 0:
                print(f'  {k}: {v.item():.6f}')
            else:
                print(f'  {k}: shape={v.shape} mean={v.mean().item():.4f}')

    model.eval()
    out_e, losses_e = model(x, intensity_map=imap)
    print(f'\nEval output: {out_e.shape}')
    if 'bypass_alpha' in losses_e:
        print(f'Bypass alpha: {losses_e["bypass_alpha"]}')
    if 'ref_logits' in losses_e:
        print(f'Ref logits (should be absent in eval): {losses_e["ref_logits"].shape}')
    else:
        print(f'Ref logits correctly absent in eval mode.')