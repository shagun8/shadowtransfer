"""
DINOv3-FADA: DINOv3 ViT-S/16 with Frequency-Adapted Domain Generalisation
for cross-city shadow detection.

Adapted from: Bi et al., "Learning Frequency-Adapted Vision Foundation Model
for Domain Generalized Semantic Segmentation", NeurIPS 2024.

Architecture
─────────────────────────────────────────────────────────────────
  Input [B, 3, 384, 384]
    ├── DINOv3 ViT-S/16  (FROZEN — ~22 M params, patch size 16)
    │     Manual block-by-block forward.  After each of blocks
    │     3, 6, 9, 11 (0-indexed):
    │       patch tokens [B, 576, 384]
    │         → reshape  [B, 384, 24, 24]
    │         → FADABlock (Haar DWT → LL branch → inv Haar
    │                                → HF branch ↗)
    │         → adapted  [B, 384, 24, 24] → LayerNorm → decoder feature
    │         adapted tokens fed BACK into next ViT block
    │
    └── DINOv3Decoder  (TRAINABLE — ~1.5 M params)
              ↓
        Shadow logits [B, 2, 384, 384]

Frozen    : ~22 M  (DINOv3 ViT-S/16)
Trainable : ~3.4 M (4×FADABlock + decoder)

Gradient flow
─────────────
The adapted features are fed back into subsequent frozen ViT blocks.
PyTorch propagates gradients THROUGH frozen blocks (without updating their
parameters, since requires_grad=False) to reach upstream FADA adapters.
This mirrors the paper's "frozen L_i → FADA → L_{i+1}" design exactly.

FADA branch assignment (following paper Section 4.2 / 4.3):
  LL  subband → Low-Frequency Branch  (learnable LoRA tokens stabilise content)
  LH/HL/HH   → High-Frequency Branch (Instance-Norm on similarity map removes
                                       domain-specific style; boundary info kept)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import DINOv3Decoder
from dinov3_fada import FADABlock      # copy of mamnet/models/fada.py


class DINOv3FADAShadowDetector(nn.Module):
    """
    DINOv3 shadow detector with FADA frequency-domain adapters.

    Only the FADA adapters and the decoder are trainable; the ViT backbone
    is fully frozen.  DINOv3 ViT-S uses LayerNorm (not BatchNorm) so no
    eval()-override is needed for normalisation layers.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'dinov3_vits16',
        weights_path: str = None,
        pretrained: bool = True,
        fada_stages: tuple = (3, 6, 9, 11),
        fada_token_length: int = 100,
        fada_rank: int = 16,
    ):
        """
        Args
        ────
        num_classes       : output classes  (2 for binary shadow detection)
        model_name        : DINOv3 variant  (dinov3_vits16 / vitb16 / vitl16)
        weights_path      : path to local .pth weights file
        pretrained        : load pretrained weights
        fada_stages       : ViT block indices (0-based) that receive a FADA adapter.
                            Must include {3, 6, 9, 11} — the keys expected by
                            DINOv3Decoder.  Default (3,6,9,11) covers all of them.
        fada_token_length : base token length m (paper default 100; stable 75-125)
        fada_rank         : LoRA rank r       (paper default 16;  stable 16-32)
        """
        super().__init__()

        self.num_classes         = num_classes
        self.model_name          = model_name
        self.fada_stages         = tuple(sorted(set(fada_stages)))
        self.fada_stages_set     = set(self.fada_stages)
        self._fada_token_length  = fada_token_length
        self._fada_rank          = fada_rank

        # ------------------------------------------------------------------
        # Backbone  (FROZEN)
        # ------------------------------------------------------------------
        print('Initialising DINOv3 backbone (will be frozen) …')
        _loader = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=-1,
        )
        self.vit        = _loader.dinov3       # bare ViT module
        self.embed_dim  = _loader.embed_dim    # 384 for ViT-S
        self.patch_size = _loader.patch_size   # 16

        for p in self.vit.parameters():
            p.requires_grad = False

        frozen_n = sum(p.numel() for p in self.vit.parameters())
        print(f'  Backbone frozen  : {frozen_n:,} parameters')

        # ------------------------------------------------------------------
        # Validate fada_stages against decoder requirements
        # ------------------------------------------------------------------
        # DINOv3Decoder.forward() concatenates feat_block3/6/9/11 exactly.
        required = {3, 6, 9, 11}
        missing  = required - self.fada_stages_set
        if missing:
            raise ValueError(
                f"fada_stages must include decoder feature blocks {sorted(required)}. "
                f"Missing: {sorted(missing)}.  Got: {self.fada_stages}."
            )

        # ------------------------------------------------------------------
        # FADA adapters  (TRAINABLE)
        # ------------------------------------------------------------------
        self.fada_adapters = nn.ModuleDict({
            f'block{i}': FADABlock(
                channels=self.embed_dim,
                token_length=fada_token_length,
                rank=fada_rank,
            )
            for i in self.fada_stages
        })
        fada_n = sum(p.numel() for p in self.fada_adapters.parameters())
        print(f'  FADA adapters    : {fada_n:,} trainable parameters')
        print(f'    stages={self.fada_stages}  embed_dim={self.embed_dim}'
              f'  m={fada_token_length}  r={fada_rank}')

        # ------------------------------------------------------------------
        # Decoder  (TRAINABLE)
        # ------------------------------------------------------------------
        self.decoder = DINOv3Decoder(
            num_classes=num_classes,
            embed_dim=self.embed_dim,
        )
        dec_n = sum(p.numel() for p in self.decoder.parameters())
        print(f'  Decoder          : {dec_n:,} trainable parameters')

        # Summary
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'\nDINOv3-FADA:')
        print(f'  Input    : 384×384  →  24×24 = 576 patch tokens')
        print(f'  Total    : {total:,}')
        print(f'  Trainable: {trainable:,}')

    # ------------------------------------------------------------------
    # Keep backbone in eval mode always
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        """Override to keep frozen ViT eval mode at all times."""
        super().train(mode)
        self.vit.eval()
        return self

    # ------------------------------------------------------------------
    # FADA forward  (manual block-by-block ViT pass)
    # ------------------------------------------------------------------

    def _fada_forward(self, x: torch.Tensor) -> dict:
        """
        Steps through each frozen ViT block.  At every FADA stage:

        1. Separate prefix tokens (CLS ± registers) from spatial patch tokens.
        2. Reshape patch tokens to [B, C, h, w].
        3. Apply FADABlock (Haar DWT → LL-branch + HF-branch → inv Haar).
        4. Apply backbone LayerNorm to adapted tokens → decoder feature.
        5. Feed adapted patch tokens back as input to the next block.

        Returns
        ───────
        features : dict
            {'feat_block3', 'feat_block6', 'feat_block9', 'feat_block11'}
            Each tensor [B, embed_dim, H/patch_size, W/patch_size]
        """
        B, _, H, W = x.shape
        num_h = H // self.patch_size
        num_w = W // self.patch_size
        n_patches = num_h * num_w

        # ---- Token preparation (patch embed + CLS + positional encoding) ----
        # prepare_tokens_with_masks may return either:
        #   - a bare tensor  [B, n_prefix + n_patches, C]          (older builds)
        #   - a tuple        ([B, n_prefix + n_patches, C], masks)  (newer builds)
        # We always want only the token tensor.
        _out     = self.vit.prepare_tokens_with_masks(x)
        x_tokens = _out[0] if isinstance(_out, (tuple, list)) else _out
        n_prefix  = x_tokens.shape[1] - n_patches   # usually 1 (CLS only)

        features: dict = {}

        for i, block in enumerate(self.vit.blocks):
            x_tokens = block(x_tokens)   # frozen block, activations tracked

            if i not in self.fada_stages_set:
                continue

            # ---- Split prefix (CLS + regs) and patch tokens ----
            prefix  = x_tokens[:, :n_prefix, :]   # [B, 1, C]
            patches = x_tokens[:, n_prefix:, :]   # [B, 576, C]

            # ---- Reshape patches → 2-D spatial map ----
            spatial = (patches
                       .reshape(B, num_h, num_w, self.embed_dim)
                       .permute(0, 3, 1, 2)
                       .contiguous())              # [B, C, h, w]

            # ---- FADA frequency adaptation (trainable path) ----
            # LL branch  → learnable LoRA tokens stabilise scene content
            # HF branches→ Instance-Norm on similarity map removes city style
            adapted = self.fada_adapters[f'block{i}'](spatial)  # [B, C, h, w]

            # ---- Flatten adapted spatial → token sequence ----
            adapted_seq = (adapted
                           .permute(0, 2, 3, 1)
                           .reshape(B, n_patches, self.embed_dim)
                           .contiguous())          # [B, 576, C]

            # ---- LayerNorm → decoder feature ----
            # Mirrors get_intermediate_layers(norm=True): apply backbone's
            # final LayerNorm to the full token sequence, then extract patches.
            normed = self.vit.norm(
                torch.cat([prefix, adapted_seq], dim=1)
            )                                      # [B, 1+576, C]
            features[f'feat_block{i}'] = (
                normed[:, n_prefix:, :]
                .reshape(B, num_h, num_w, self.embed_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )                                      # [B, C, h, w]

            # ---- Feed adapted tokens back to next block ----
            # Gradients propagate through subsequent frozen blocks to reach
            # upstream FADA parameters (frozen block parameters ignored).
            x_tokens = torch.cat([prefix, adapted_seq], dim=1)

        return features

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args  :  x  [B, 3, H, W],  H and W must be multiples of 16
        Returns: logits [B, num_classes, H, W]
        """
        B, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input ({H}×{W}) must be divisible by patch_size={self.patch_size}."
            )

        features = self._fada_forward(x)
        output   = self.decoder(features)

        if output.shape[2] != H or output.shape[3] != W:
            output = F.interpolate(output, size=(H, W),
                                   mode='bilinear', align_corners=False)
        return output

    def get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Binary predictions for inference (no grad)."""
        self.eval()
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        fada_p    = sum(p.numel() for p in self.fada_adapters.parameters())
        dec_p     = sum(p.numel() for p in self.decoder.parameters())

        print(f"\n{'='*50}")
        print("Parameter breakdown:")
        print(f"{'='*50}")
        print(f"  Total           : {total:>12,}")
        print(f"  Frozen (ViT)    : {frozen:>12,}")
        print(f"  Trainable       : {trainable:>12,}")
        print(f"    FADA adapters : {fada_p:>12,}")
        print(f"    Decoder       : {dec_p:>12,}")
        print(f"{'='*50}")
        return dict(total=total, frozen=frozen, trainable=trainable,
                    fada=fada_p, decoder=dec_p)

    def get_param_groups(
        self,
        lr_fada: float = 1e-4,
        lr_decoder: float = 1e-4,
    ) -> list:
        """
        Per-group parameter lists for the optimiser.
        Allows separate LRs for FADA adapters and decoder.
        """
        fada_params    = [p for n, p in self.named_parameters()
                          if 'fada_adapters' in n and p.requires_grad]
        decoder_params = [p for n, p in self.named_parameters()
                          if 'decoder' in n and p.requires_grad]
        groups = []
        if fada_params:
            groups.append({'params': fada_params,    'lr': lr_fada,    'name': 'fada'})
        if decoder_params:
            groups.append({'params': decoder_params, 'lr': lr_decoder, 'name': 'decoder'})
        return groups


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Testing DINOv3FADAShadowDetector …')
    model = DINOv3FADAShadowDetector(
        num_classes=2,
        model_name='dinov3_vits16',
        weights_path=None,
        pretrained=True,
        fada_stages=(3, 6, 9, 11),
        fada_token_length=100,
        fada_rank=16,
    )

    counts = model.count_parameters()

    model.train()
    x = torch.randn(2, 3, 384, 384)
    out = model(x)
    print(f'\nTrain forward  : {x.shape} → {out.shape}')

    for n, p in model.named_parameters():
        if 'vit' in n and p.requires_grad:
            raise RuntimeError(f'Backbone parameter {n} is NOT frozen!')
    print('✓ All backbone parameters are frozen.')

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f'Eval  forward  : {x.shape} → {out_eval.shape}')
    print('✓ Self-test passed.')