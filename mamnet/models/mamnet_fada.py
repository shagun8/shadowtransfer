"""
MAMNet-FADA: MAMNet with Frequency-Adapted Domain Generalization

Integrates the FADA module (Bi et al., NeurIPS 2024) into the MAMNet
shadow detection architecture for cross-city transferability.

===============================================================================
ARCHITECTURE OVERVIEW
===============================================================================

  Input Image
       │
       ▼
  ┌─────────────────────┐
  │  ResNet-34 Encoder   │  ◄── FROZEN (all parameters)
  │  (pretrained)        │
  └──┬──┬──┬──┬──┬──────┘
     │  │  │  │  │
  feat1 feat2 │  │  feat5
     │  │  feat3 feat4  │
     │  │  │  │  │
     │  │  ▼  ▼  ▼
     │  │  ┌──────────┐
     │  │  │ FADA     │  ◄── TRAINABLE (LoRA tokens, MLPs)
     │  │  │ Blocks   │      Applied at feat3, feat4, feat5
     │  │  └──────────┘
     │  │  │  │  │
     ▼  ▼  ▼  ▼  ▼
  ┌─────────────────────┐
  │  MSCAF + Decoder    │  ◄── TRAINABLE
  │  + CCA + Aux        │
  └─────────────────────┘
       │
       ▼
  Prediction

===============================================================================
DESIGN DECISIONS
===============================================================================

1. FROZEN ENCODER:
   Following the paper (Sec. 4, Fig. 2), the backbone (encoder) is completely
   frozen. Only FADA modules and the decoder are trained. This prevents the
   pretrained features from drifting during fine-tuning and forces all
   domain adaptation to occur through the frequency decomposition pathway.

2. FADA STAGES:
   Applied at feat3 (256ch), feat4 (512ch), feat5 (512ch). See fada.py for
   the reasoning on why feat1/feat2 are excluded.

3. TRAINABLE COMPONENTS:
   - FADA blocks (LoRA tokens + MLPs): ~2.4M parameters
   - MSCAF module: multi-scale feature fusion
   - Decoder with CCA: criss-cross attention for context
   - Auxiliary branches: deep supervision (training only)
   Total trainable ≈ frozen encoder (~21M) excluded, ~6-7M trainable.

4. 4-CHANNEL SUPPORT:
   When use_contrast=True, the 4-channel encoder (RGB + Contrast) is used.
   FADA operates on encoder output features (64/128/256/512 channels),
   so it is agnostic to the number of input channels.
===============================================================================
"""

import torch
import torch.nn as nn

from .encoder import ResNet34Encoder
from .encoder_4ch import ResNet34Encoder4Ch
from .mscaf import MSCAF
from .decoder import Decoder
from .auxiliary import AuxiliaryModule
from .fada import FADABlock


class MAMNetFADA(nn.Module):
    """
    MAMNet with Frequency-Adapted (FADA) domain generalization.

    The encoder is frozen and FADA blocks are inserted after selected
    encoder stages to adapt features in the frequency domain, removing
    city-specific style while preserving shadow-relevant content.
    """

    def __init__(self, num_classes=2, pretrained=True, use_aux=True,
                 use_contrast=False,
                 fada_rank=16, fada_token_length=100,
                 fada_stages=(3, 4, 5)):
        """
        Args:
            num_classes: Number of output classes (default: 2 for shadow/non-shadow)
            pretrained: Use ImageNet-pretrained encoder weights
            use_aux: Enable auxiliary branches for deep supervision
            use_contrast: Use 4-channel input (RGB + Contrast)
            fada_rank: LoRA rank r for token decomposition (paper default: 16)
            fada_token_length: Base token length m (paper default: 100)
            fada_stages: Tuple of encoder stages to apply FADA
                         (default: (3, 4, 5) for feat3, feat4, feat5)
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_aux = use_aux
        self.use_contrast = use_contrast
        self.fada_stages = fada_stages

        # ------------------------------------------------------------------
        # Encoder (FROZEN)
        # ------------------------------------------------------------------
        if use_contrast:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)
            print("Using 4-channel encoder (RGB + Contrast) — FROZEN")
        else:
            self.encoder = ResNet34Encoder(pretrained=pretrained)
            print("Using 3-channel encoder (RGB) — FROZEN")

        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()  # BN layers stay in eval mode

        # ------------------------------------------------------------------
        # FADA Adapters (TRAINABLE)
        # ------------------------------------------------------------------
        stage_channels = {1: 64, 2: 128, 3: 256, 4: 512, 5: 512}
        self.fada_adapters = nn.ModuleDict()
        for stage in fada_stages:
            assert stage in stage_channels, \
                f"Invalid FADA stage {stage}. Must be in {list(stage_channels.keys())}"
            ch = stage_channels[stage]
            self.fada_adapters[f'feat{stage}'] = FADABlock(
                channels=ch,
                token_length=fada_token_length,
                rank=fada_rank,
            )
            print(f"  FADA block at feat{stage}: "
                  f"ch={ch}, m={fada_token_length}, r={fada_rank}")

        # ------------------------------------------------------------------
        # MSCAF (TRAINABLE) — multi-scale spatial-channel attention fusion
        # ------------------------------------------------------------------
        self.mscaf = MSCAF(in_channels=512)

        # ------------------------------------------------------------------
        # Decoder with CCA (TRAINABLE)
        # ------------------------------------------------------------------
        self.decoder = Decoder(num_classes=num_classes)

        # ------------------------------------------------------------------
        # Auxiliary branches (TRAINABLE, training only)
        # ------------------------------------------------------------------
        if use_aux:
            self.aux_module = AuxiliaryModule(
                num_classes=num_classes, dropout_rate=0.3)

    def train(self, mode=True):
        """
        Override train() to keep encoder in eval mode (frozen BN).

        When the model is set to train mode, the encoder's BatchNorm layers
        must remain in eval mode to use the pretrained running statistics
        rather than batch statistics. This is critical for frozen backbones.
        """
        super().train(mode)
        # Always keep encoder in eval mode (frozen BN)
        self.encoder.eval()
        return self

    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W] or [B, 4, H, W] if use_contrast

        Returns:
            If training with use_aux:
                Dict with 'main', 'aux1', 'aux2', 'aux3'
            If inference:
                Main prediction [B, num_classes, H, W]
        """
        B, _, H, W = x.size()

        # ------------------------------------------------------------------
        # Frozen encoder forward (no gradient computation or storage)
        # ------------------------------------------------------------------
        with torch.no_grad():
            enc_features = self.encoder(x)

        # Detach all encoder features to cut the computation graph.
        # This is redundant with torch.no_grad() but makes the intent
        # explicit: no gradients flow back into the encoder.
        enc_features = {k: v.detach() for k, v in enc_features.items()}

        # ------------------------------------------------------------------
        # Apply FADA frequency adaptation to selected stages
        # ------------------------------------------------------------------
        adapted_features = {}
        for key, feat in enc_features.items():
            if key in self.fada_adapters:
                adapted_features[key] = self.fada_adapters[key](feat)
            else:
                adapted_features[key] = feat

        # ------------------------------------------------------------------
        # MSCAF on deepest adapted features
        # ------------------------------------------------------------------
        mscaf_out = self.mscaf(adapted_features['feat5'])

        # ------------------------------------------------------------------
        # Decoder with CCA
        # ------------------------------------------------------------------
        decoder_outputs = self.decoder(mscaf_out, adapted_features)
        main_out = decoder_outputs['main']

        # ------------------------------------------------------------------
        # Auxiliary branches (training only)
        # ------------------------------------------------------------------
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
            }
        else:
            return main_out

    def get_predictions(self, x):
        """Get binary predictions for inference/evaluation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = torch.argmax(logits, dim=1)
        return preds

    # ------------------------------------------------------------------
    # Parameter inspection utilities
    # ------------------------------------------------------------------

    def count_parameters(self):
        """
        Print a breakdown of total, frozen, and trainable parameters.
        """
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        fada_params = sum(
            p.numel() for n, p in self.named_parameters()
            if 'fada_adapters' in n and p.requires_grad
        )
        decoder_params = trainable - fada_params

        print(f"\n{'='*50}")
        print(f"Parameter Breakdown:")
        print(f"{'='*50}")
        print(f"  Total:         {total:>12,}")
        print(f"  Frozen (enc):  {frozen:>12,}")
        print(f"  Trainable:     {trainable:>12,}")
        print(f"    FADA:        {fada_params:>12,}")
        print(f"    Decoder etc: {decoder_params:>12,}")
        print(f"{'='*50}")

        return {
            'total': total,
            'frozen': frozen,
            'trainable': trainable,
            'fada': fada_params,
            'decoder': decoder_params,
        }

    def get_param_groups(self, lr_fada=1e-4, lr_decoder=1e-4):
        """
        Get parameter groups with potentially different learning rates.

        Useful if you want different LRs for FADA vs. decoder modules.
        By default both use 1e-4 (paper default).

        Args:
            lr_fada: Learning rate for FADA modules
            lr_decoder: Learning rate for decoder/MSCAF/aux modules

        Returns:
            List of parameter group dicts for the optimizer
        """
        fada_params = []
        decoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'fada_adapters' in name:
                fada_params.append(param)
            else:
                decoder_params.append(param)

        groups = []
        if fada_params:
            groups.append({
                'params': fada_params,
                'lr': lr_fada,
                'name': 'fada',
            })
        if decoder_params:
            groups.append({
                'params': decoder_params,
                'lr': lr_decoder,
                'name': 'decoder',
            })

        return groups


if __name__ == "__main__":
    print("Testing MAMNetFADA...")

    model = MAMNetFADA(
        num_classes=2,
        pretrained=True,
        use_aux=True,
        use_contrast=False,
        fada_rank=16,
        fada_token_length=100,
        fada_stages=(3, 4, 5),
    )

    # Parameter breakdown
    counts = model.count_parameters()

    # Training mode
    model.train()
    x = torch.randn(2, 3, 256, 256)
    outputs_train = model(x)
    print("\nTraining outputs:")
    for key, val in outputs_train.items():
        print(f"  {key}: {val.shape}")

    # Verify encoder is frozen
    for name, param in model.named_parameters():
        if 'encoder' in name:
            assert not param.requires_grad, f"Encoder param {name} should be frozen!"
    print("\n✓ All encoder parameters are frozen")

    # Inference mode
    model.eval()
    outputs_eval = model(x)
    print(f"\nInference output: {outputs_eval.shape}")

    # Test 4-channel
    print("\n--- 4-channel (contrast) mode ---")
    model_4ch = MAMNetFADA(
        num_classes=2, pretrained=True, use_aux=True,
        use_contrast=True, fada_stages=(3, 4, 5))
    model_4ch.train()
    x_4ch = torch.randn(2, 4, 256, 256)
    out_4ch = model_4ch(x_4ch)
    print("4ch training outputs:")
    for key, val in out_4ch.items():
        print(f"  {key}: {val.shape}")
    model_4ch.count_parameters()