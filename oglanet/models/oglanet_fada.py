"""
OGLANetFADA: OGLANet with Frequency-Adapted Domain Generalization.

===============================================================================
FIX vs. original oglanet_fada.py
===============================================================================

Original problem:
    The entire GLAMEncoder was frozen. But GLAMEncoder = ResNet-34 (pretrained)
    + GFEM modules (random init — never pretrained on anything). Freezing the
    whole encoder locked in random GFEM transformations of ResNet features.
    The downstream DFFM + Decoder + OAM received effectively random features
    and collapsed to predicting background everywhere (CE loss still decreased
    because background is ~90% of pixels, so loss is minimised by confidently
    predicting background everywhere).

Fix:
    Freeze ONLY the ResNet-34 core inside GLAMEncoder (conv1, bn1, layer1-4).
    The GFEM modules (gfem1, channel_reduce, gfem2 inside each GLAM stage,
    plus glam5_conv and glam5_gfem) remain TRAINABLE, just like MAMNet-FADA
    keeps its MSCAF and decoder trainable.

    FADA is inserted between the frozen LFEM output (ResNet layer) and the
    trainable GFEM modules within each selected GLAM stage — exactly the same
    relative position as in MAMNet-FADA (between frozen backbone output and
    trainable decoder).

===============================================================================
ARCHITECTURE (fixed)
===============================================================================

  Input Image (RGB or RGBC)
         │
         ▼
  ┌──────────────────────────────────────────────────────┐
  │  GLAMEncoder                                          │
  │                                                       │
  │  initial (conv1 + maxpool)  ◄── FROZEN               │
  │       │ x0                                            │
  │  ┌────┴──────────────────────────────┐               │
  │  │  GLAM stage n  (n = 1, 2, 3, 4)   │               │
  │  │                                    │               │
  │  │  x_in ──► frozen LFEM (layerN)     │               │
  │  │                  │ feat_raw         │               │
  │  │                  ▼                  │               │
  │  │            [FADA block] ◄─TRAINABLE │               │
  │  │                  │ feat_adapted     │               │
  │  │  x_in ──► trainable GFEM1 ──┐      │               │
  │  │                              ▼      │               │
  │  │           cat → channel_reduce → GFEM2 → feat_out  │
  │  └────────────────────────────────────┘               │
  │                                                       │
  │  GLAM5: glam5_conv ──► [FADA?] ──► glam5_gfem        │
  │         (fully trainable — no pretrained component)   │
  └──────────────────────────────────────────────────────┘
         │ feat1 … feat5
         ▼
  DFFM → Decoder → OAM  ◄── TRAINABLE
         │
  P1 … P6 predictions (training) / P6 (inference)

===============================================================================
FADA STAGE → CHANNEL MAPPING (OGLANet, 384×384 input)
===============================================================================
  Stage 1: ResNet layer1 output,  64ch, 96×96  → Haar LL: 48×48 = 2304 tokens
  Stage 2: ResNet layer2 output, 128ch, 48×48  → Haar LL: 24×24 =  576 tokens
  Stage 3: ResNet layer3 output, 256ch, 24×24  → Haar LL: 12×12 =  144 tokens
  Stage 4: ResNet layer4 output, 512ch, 12×12  → Haar LL:  6×6  =   36 tokens
  Stage 5: glam5_conv output,   1024ch,  6×6   → Haar LL:  3×3  =    9 tokens
             (note: stage 5 has no frozen pretrained component)

  Default fada_stages = (3, 4).  Stages 3 and 4 are the best balance:
  deep enough to carry style information, small enough for practical attention.
  Stages 1/2 can be added if GPU memory allows. Stage 5 produces only 9 Haar
  tokens which makes the token-attention map degenerate.

===============================================================================
FROZEN vs. TRAINABLE SUMMARY
===============================================================================
  Frozen   (~21.3M):  encoder.resnet_encoder  (conv1, bn1, layer1-4)
  Trainable — 'fada' group:   fada_adapters.*
  Trainable — 'decoder' group: encoder GFEM modules + dffm + decoder + oam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glam import GLAMEncoder
from .dffm import DFFM
from .decoder import Decoder
from .oam import OAM
from .fada import FADABlock


class OGLANetFADA(nn.Module):
    """
    OGLANet with Frequency-Adapted (FADA) domain generalisation.

    Only the ResNet-34 core inside GLAMEncoder is frozen. GFEM modules
    are trained from random initialisation alongside the FADA adapters,
    DFFM, Decoder, and OAM.

    FADA is applied between the frozen LFEM (ResNet layer) output and the
    trainable GFEM modules within each selected GLAM stage.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        img_size=384,
        use_contrast=False,
        fada_rank=16,
        fada_token_length=100,
        fada_stages=(3, 4),
    ):
        """
        Args:
            num_classes:       Output classes (default 2: shadow / non-shadow)
            pretrained:        Load ImageNet weights into the ResNet-34 core
            img_size:          Spatial resolution (default 384)
            use_contrast:      4-channel RGBC input
            fada_rank:         LoRA rank r  (paper default 16)
            fada_token_length: Token length m  (paper default 100)
            fada_stages:       GLAM stages at which FADA is inserted.
                               Channels are raw ResNet-34 layer outputs:
                                 1→64ch, 2→128ch, 3→256ch, 4→512ch,
                                 5→1024ch (glam5_conv, trainable)
                               Default (3, 4) is strongly recommended.
        """
        super().__init__()

        self.num_classes  = num_classes
        self.img_size     = img_size
        self.use_contrast = use_contrast
        self.fada_stages  = tuple(fada_stages)

        # ------------------------------------------------------------------
        # GLAMEncoder — build with pretrained ResNet-34 core
        # ------------------------------------------------------------------
        self.encoder = GLAMEncoder(
            pretrained=pretrained, use_contrast=use_contrast
        )

        # Freeze ONLY the ResNet-34 core inside GLAMEncoder.
        # conv1, bn1, layer1-4 are frozen.
        # GFEM modules (gfem1, channel_reduce, gfem2, glam5_conv, glam5_gfem)
        # are NOT frozen — they train from random init.
        for param in self.encoder.resnet_encoder.parameters():
            param.requires_grad = False

        n_frozen    = sum(p.numel() for p in self.encoder.resnet_encoder.parameters())
        n_gfem_train = sum(
            p.numel() for n, p in self.encoder.named_parameters()
            if p.requires_grad
        )
        print(
            f"GLAMEncoder: ResNet-34 core FROZEN ({n_frozen:,} params). "
            f"GFEM modules TRAINABLE ({n_gfem_train:,} params)."
        )

        # ------------------------------------------------------------------
        # FADA Adapters — TRAINABLE
        # Channels correspond to raw ResNet-34 layer outputs (pre-GFEM).
        # ------------------------------------------------------------------
        stage_channels = {1: 64, 2: 128, 3: 256, 4: 512, 5: 1024}

        self.fada_adapters = nn.ModuleDict()
        for stage in self.fada_stages:
            assert stage in stage_channels, (
                f"Invalid FADA stage {stage}. "
                f"Valid: {sorted(stage_channels.keys())}"
            )
            ch = stage_channels[stage]
            src = f"layer{stage}" if stage < 5 else "glam5_conv"
            self.fada_adapters[f"feat{stage}"] = FADABlock(
                channels=ch,
                token_length=fada_token_length,
                rank=fada_rank,
            )
            print(
                f"  FADA @ feat{stage} ({src} output): "
                f"ch={ch}, m={fada_token_length}, r={fada_rank}"
            )

        # ------------------------------------------------------------------
        # DFFM, Decoder, OAM — all TRAINABLE
        # ------------------------------------------------------------------
        self.dffm    = DFFM()
        self.decoder = Decoder(target_size=(img_size, img_size))
        self.oam     = OAM(num_classes=num_classes, target_size=(img_size, img_size))

    # ------------------------------------------------------------------
    # train() override — keep ResNet-34 core BN in eval mode always
    # ------------------------------------------------------------------

    def train(self, mode=True):
        """
        Keep the ResNet-34 core's BatchNorm layers in eval mode at all times.

        This preserves the pretrained running statistics so they are not
        recomputed on the small, city-specific training set (which would
        re-introduce city-specific bias that FADA is designed to remove).

        GFEM BatchNorm layers are NOT affected; they train normally.
        """
        super().train(mode)
        self.encoder.resnet_encoder.eval()
        return self

    # ------------------------------------------------------------------
    # GLAM stage helper with FADA injection
    # ------------------------------------------------------------------

    def _run_glam_with_fada(self, glam_module, x_input, fada_key):
        """
        Run one GLAM stage (glam1…glam4) with FADA inserted between the
        frozen LFEM output and the trainable GFEM modules.

        Original GLAM forward for reference:
            local_feat  = lfem(x)         # frozen ResNet layer
            global_feat = gfem1(x)        # trainable, takes same x as lfem
            cat(local_feat, global_feat) → channel_reduce → gfem2 → output

        Modified here:
            local_feat  = lfem(x)         # frozen (no grad)
            local_feat  = FADA(local_feat) # trainable adaptation
            global_feat = gfem1(x)        # trainable, same x as before
            cat(local_feat, global_feat) → channel_reduce → gfem2 → output

        The key invariant maintained: gfem1 always receives x_input (the
        stage input), exactly as in the original GLAM. Only the LFEM output
        path is intercepted for FADA adaptation.

        Args:
            glam_module: self.encoder.glam1 … glam4
            x_input:     input to this GLAM stage (detached for glam1,
                         has grad for glam2-4 since it comes from trainable GFEM)
            fada_key:    e.g. 'feat3'; looked up in self.fada_adapters

        Returns:
            output feature map [B, out_channels, H', W']
        """
        # ---- Frozen LFEM: no gradient, no BN stat update ----
        with torch.no_grad():
            feat_raw = glam_module.lfem(x_input)
        feat_raw = feat_raw.detach()

        # ---- FADA adaptation (if this stage is selected) ----
        if fada_key in self.fada_adapters:
            feat_adapted = self.fada_adapters[fada_key](feat_raw)
        else:
            feat_adapted = feat_raw

        # ---- Trainable GFEM1 on the original x_input ----
        # Consistent with original GLAM: gfem1 takes the stage input,
        # not the LFEM output.
        global_feat1 = glam_module.gfem1(x_input)

        # ---- Spatial alignment (same as original GLAM) ----
        if feat_adapted.size(2) != global_feat1.size(2):
            feat_adapted = F.adaptive_avg_pool2d(
                feat_adapted, global_feat1.shape[2:]
            )

        # ---- Trainable channel_reduce + GFEM2 ----
        concat_feat  = torch.cat([feat_adapted, global_feat1], dim=1)
        reduced_feat = glam_module.channel_reduce(concat_feat)
        output       = glam_module.gfem2(reduced_feat)

        return output

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]  or  [B, 4, H, W] if use_contrast

        Returns:
            Training:  dict {'p1' … 'p6'}, each [B, num_classes, H, W]
            Inference: P6 tensor [B, num_classes, H, W]
        """
        # ---- Frozen stem (conv1 + bn1 + relu + maxpool) ----
        with torch.no_grad():
            x0 = self.encoder.initial(x)
        x0 = x0.detach()   # [B, 64, H/4, W/4]  e.g. [B, 64, 96, 96] at 384

        # ---- GLAM stages 1-4 ----
        # Each stage: frozen LFEM → FADA (selected stages) → trainable GFEM
        # Sequential: each stage's output is the next stage's x_input.
        # Gradients flow normally through trainable GFEM modules in glam2-4
        # because those x_inputs are outputs of the previous stage's GFEM.
        feat1 = self._run_glam_with_fada(self.encoder.glam1, x0,    "feat1")
        feat2 = self._run_glam_with_fada(self.encoder.glam2, feat1, "feat2")
        feat3 = self._run_glam_with_fada(self.encoder.glam3, feat2, "feat3")
        feat4 = self._run_glam_with_fada(self.encoder.glam4, feat3, "feat4")

        # ---- GLAM5: fully trainable (no frozen ResNet component) ----
        # glam5_conv: stride-2 Conv (512→1024ch), random init, trainable.
        # Optional FADA between glam5_conv and glam5_gfem (if stage 5 selected).
        feat5_conv = self.encoder.glam5_conv(feat4)     # [B, 1024, H/64, W/64]
        if "feat5" in self.fada_adapters:
            feat5_conv = self.fada_adapters["feat5"](feat5_conv)
        feat5 = self.encoder.glam5_gfem(feat5_conv)     # [B, 1024, H/64, W/64]

        enc_features = {
            "feat1": feat1,   # [B,  64, H/4,  W/4 ]
            "feat2": feat2,   # [B, 128, H/8,  W/8 ]
            "feat3": feat3,   # [B, 256, H/16, W/16]
            "feat4": feat4,   # [B, 512, H/32, W/32]
            "feat5": feat5,   # [B,1024, H/64, W/64]
        }

        # ---- Standard OGLANet decoder pipeline (all trainable) ----
        dffm_features    = self.dffm(enc_features)
        decoder_features = self.decoder(dffm_features)
        predictions      = self.oam(decoder_features)

        if self.training:
            return predictions           # dict {p1 … p6}
        else:
            return predictions["p6"]    # P6 only at inference

    def get_predictions(self, x):
        """Return binary shadow predictions [B, H, W]."""
        self.eval()
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------

    def count_parameters(self):
        """Print a detailed parameter breakdown."""
        total     = sum(p.numel() for p in self.parameters())
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        fada_count = sum(
            p.numel() for n, p in self.named_parameters()
            if "fada_adapters" in n and p.requires_grad
        )
        gfem_count = sum(
            p.numel() for n, p in self.named_parameters()
            if "encoder" in n and p.requires_grad
        )
        rest_count = trainable - fada_count - gfem_count

        print(f"\n{'=' * 56}")
        print("OGLANetFADA — Parameter Breakdown")
        print(f"{'=' * 56}")
        print(f"  Total                     : {total:>12,}")
        print(f"  Frozen  (ResNet-34 core)  : {frozen:>12,}")
        print(f"  Trainable                 : {trainable:>12,}")
        print(f"    FADA adapters           : {fada_count:>12,}")
        print(f"    GFEM modules (in enc.)  : {gfem_count:>12,}")
        print(f"    DFFM + Decoder + OAM    : {rest_count:>12,}")
        print(f"{'=' * 56}\n")

        return {
            "total":     total,
            "frozen":    frozen,
            "trainable": trainable,
            "fada":      fada_count,
            "gfem":      gfem_count,
            "decoder":   rest_count,
        }

    def get_param_groups(self, lr_fada=1e-4, lr_decoder=1e-4):
        """
        Return optimizer parameter groups.

        Group 'fada':    FADA adapter LoRA tokens + MLP projections.
        Group 'decoder': GFEM modules inside encoder (trainable) +
                         DFFM + Decoder + OAM.
                         GFEM modules are grouped here because they start from
                         random init like the decoder, not pretrained like ResNet.

        Args:
            lr_fada:    LR for FADA parameters
            lr_decoder: LR for all other trainable parameters

        Returns:
            List of param-group dicts for torch.optim
        """
        fada_params    = []
        decoder_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "fada_adapters" in name:
                fada_params.append(param)
            else:
                # Includes GFEM (inside encoder, but trainable) + DFFM + Dec + OAM
                decoder_params.append(param)

        groups = []
        if fada_params:
            groups.append(
                {"params": fada_params,    "lr": lr_fada,    "name": "fada"}
            )
        if decoder_params:
            groups.append(
                {"params": decoder_params, "lr": lr_decoder, "name": "decoder"}
            )
        return groups


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing OGLANetFADA (freeze ResNet-34 core only) …\n")

    model = OGLANetFADA(
        num_classes=2,
        pretrained=False,   # skip download in quick test
        img_size=384,
        use_contrast=False,
        fada_rank=16,
        fada_token_length=100,
        fada_stages=(3, 4),
    )
    model.count_parameters()

    # ---- Training mode ----
    model.train()
    x = torch.randn(2, 3, 384, 384)
    out_train = model(x)
    print("Training outputs:")
    for k, v in out_train.items():
        print(f"  {k}: {v.shape}")

    # ---- Verify only ResNet-34 core is frozen ----
    for name, param in model.named_parameters():
        if "resnet_encoder" in name:
            assert not param.requires_grad, \
                f"Param '{name}' inside resnet_encoder should be frozen!"
    print("\n✓ All resnet_encoder parameters are frozen")

    # ---- Verify GFEM modules are trainable ----
    gfem_trainable = [
        n for n, p in model.named_parameters()
        if "encoder" in n and "resnet_encoder" not in n and p.requires_grad
    ]
    assert len(gfem_trainable) > 0, "GFEM modules should be trainable!"
    print(f"✓ {len(gfem_trainable)} GFEM parameter tensors are trainable")

    # ---- Verify ResNet-34 BN stays in eval during train() ----
    for name, m in model.named_modules():
        if "resnet_encoder" in name and isinstance(m, nn.BatchNorm2d):
            assert not m.training, \
                f"ResNet-34 BN '{name}' should be in eval mode during train()!"
    print("✓ All resnet_encoder BatchNorm layers stay in eval mode")

    # ---- Verify GFEM BN IS in training mode ----
    gfem_bn_training = [
        name for name, m in model.named_modules()
        if "encoder" in name
        and "resnet_encoder" not in name
        and isinstance(m, nn.BatchNorm2d)
        and m.training
    ]
    assert len(gfem_bn_training) > 0, "GFEM BN layers should be in train mode!"
    print(f"✓ {len(gfem_bn_training)} GFEM BatchNorm layers are in train mode")

    # ---- Inference mode ----
    model.eval()
    out_eval = model(x)
    print(f"\nInference output: {out_eval.shape}")

    # ---- 4-channel (RGBC) mode ----
    print("\n--- 4-channel (RGBC) mode ---")
    model_4ch = OGLANetFADA(
        num_classes=2, pretrained=False, img_size=384,
        use_contrast=True, fada_stages=(3, 4),
    )
    model_4ch.train()
    out_4ch = model_4ch(torch.randn(2, 4, 384, 384))
    print("4-channel training outputs:")
    for k, v in out_4ch.items():
        print(f"  {k}: {v.shape}")
    model_4ch.count_parameters()