"""
OGLANet + Spectral Information Bottleneck (SIB) wrapper.

Correctly interfaces with OGLANet components:
  - GLAMEncoder: returns dict {'feat1'...'feat5'}
  - DFFM: expects dict {'feat1'...'feat5'}
  - Decoder: expects DFFM output features
  - OAM: produces list of predictions [P1...P6]
  - OGLANetLoss: expects dict {'p1'...'p6'}

SIB is applied to feat4 AFTER encoder, BEFORE DFFM.
SAG gates are applied to feat1-3 skip connections.
Optional ModuleBypassGate wraps the entire SIB call on feat4.

NEW — Diagnostic-motivated additions (§4.3 orphan coverage):
  - CACR (Class-Asymmetric Confidence Regularizer): when use_cacr=True
    and training, the wrapper runs a SECOND pass through DFFM → Decoder
    → OAM using the post-VIB pre-augmentation feat4 from the SIB module.
    The resulting reference predictions are returned in kl_losses
    under key 'ref_predictions'. The training loop computes the
    asymmetric penalty between main p6 and ref p6.
  - CE-AURC: flag-only at the model level; loss is computed in the
    training loop on main p6 vs targets.
  - TENT: flag-only at the model level; adaptation is performed in the
    test loop via configure_tent + tent_adapt_step.
"""

import torch
import torch.nn as nn

from models.glam import GLAMEncoder
from models.dffm import DFFM
from models.decoder import Decoder
from models.oam import OAM
from utils.losses import OGLANetLoss
from models.sib import SIB, SkipAttentionGate, MultiScaleSIB, ModuleBypassGate


class OGLANetSIB(nn.Module):
    """
    OGLANet with Spectral Information Bottleneck for cross-location transfer.

    Architecture flow:
        RGB(+C) → GLAMEncoder → {feat1..feat5}
                                    ↓
                            SAG on feat1-3 (skip filtering)
                            SIB on feat4   (spectral bottleneck)
                            [ModuleBypassGate on feat4 (optional)]
                                    ↓
                            DFFM → Decoder → OAM
                                    ↓
                            {p1..p6} predictions (deep supervision)

        When use_cacr=True (training only):
                            SIB also returns pre-aug feat4 (post-VIB,
                              before content augmentation)
                            Second forward through DFFM → Decoder → OAM
                              produces ref_predictions {p1..p6}
                            Returned in kl_losses['ref_predictions']
    """

    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 3,
                 pretrained_encoder: bool = True,
                 # SIB config
                 use_sib: bool = True,
                 sib_channels: int = 512,
                 beta_content: float = 0.01,
                 beta_edge: float = 0.001,
                 beta_noise: float = 0.05,
                 adaptive_beta: bool = True,
                 use_haar: bool = True,
                 use_vib: bool = True,
                 use_aug: bool = True,
                 sigma_style: float = 0.1,
                 sigma_shift: float = 0.05,
                 aug_p_aug: float = 0.5,
                 aug_p_mix: float = 0.3,
                 use_passthrough_gate: bool = False,
                 use_module_bypass: bool = False,
                 # SAG config
                 use_sag: bool = True,
                 sag_reduction: int = 16,
                 # Multi-scale SIB (optional)
                 use_multiscale_sib: bool = False,
                 multiscale_channels: list = None,
                 # ── Ablation flags ──
                 skip_ll_vib: bool = False,
                 symmetric_beta: bool = False,
                 aug_all_subbands: bool = False,
                 vib_only_band: str = None,
                 # ── NEW: Diagnostic-motivated modules ──
                 use_cacr: bool = False,
                 use_ce_aurc: bool = False,
                 use_tent: bool = False):
        super().__init__()
        self.use_sib = use_sib
        self.use_sag = use_sag
        self.use_module_bypass = use_module_bypass
        self.use_multiscale_sib = use_multiscale_sib
        self.num_classes = num_classes

        # ── New diagnostic flags ──
        self.use_cacr = use_cacr
        self.use_ce_aurc = use_ce_aurc
        self.use_tent = use_tent

        # ── OGLANet backbone components ──
        self.encoder = GLAMEncoder(
            pretrained=pretrained_encoder,
            use_contrast=(in_channels == 4)
        )

        self.dffm = DFFM()
        self.decoder = Decoder()
        self.oam = OAM(num_classes=num_classes)

        # ── SIB on feat4 ──
        # When CACR is enabled, SIB also returns pre-aug bottleneck features.
        if use_sib:
            self.sib = SIB(
                channels=sib_channels,
                beta_content=beta_content,
                beta_edge=beta_edge,
                beta_noise=beta_noise,
                adaptive_beta=adaptive_beta,
                use_haar=use_haar,
                use_vib=use_vib,
                use_aug=use_aug,
                sigma_style=sigma_style,
                sigma_shift=sigma_shift,
                aug_p_aug=aug_p_aug,
                aug_p_mix=aug_p_mix,
                use_passthrough_gate=use_passthrough_gate,
                # Ablation flags
                skip_ll_vib=skip_ll_vib,
                symmetric_beta=symmetric_beta,
                aug_all_subbands=aug_all_subbands,
                vib_only_band=vib_only_band,
                # NEW: enable pre-aug snapshot when CACR is active
                return_pre_aug=use_cacr,
            )

        # ── Module Bypass Gate (wraps entire SIB pipeline on feat4) ──
        if use_module_bypass and use_sib:
            self.module_bypass_gate = ModuleBypassGate(channels=sib_channels)

        # ── Skip Attention Gates on feat1-3 ──
        if use_sag:
            self.sag1 = SkipAttentionGate(64,  reduction=sag_reduction)
            self.sag2 = SkipAttentionGate(128, reduction=sag_reduction)
            self.sag3 = SkipAttentionGate(256, reduction=sag_reduction)

        # ── Optional multi-scale SIB at feat1-3 ──
        if use_multiscale_sib:
            ms_channels = multiscale_channels or [64, 128, 256]
            self.ms_sib = MultiScaleSIB(
                channel_list=ms_channels,
                bottleneck_ratio=0.25,
                beta_content=beta_content,
                beta_edge=beta_edge,
                beta_noise=beta_noise,
                adaptive_beta=False,
                use_haar=use_haar,
                use_vib=use_vib,
                use_aug=False,
            )

        # ── Print config ──
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'\nOGLANet-SIB Shadow Detector:')
        print(f'  Total params:     {total:,}')
        print(f'  Trainable params: {trainable:,}')
        print(f'  SIB config: haar={use_haar} vib={use_vib} aug={use_aug} '
              f'ab={adaptive_beta} gate={use_passthrough_gate} '
              f'mod_bypass={use_module_bypass}')
        print(f'  SAG={use_sag}  MultiScaleSIB={use_multiscale_sib}')
        print(f'  CACR={use_cacr}  CE-AURC={use_ce_aurc}  TENT={use_tent}')

    def forward(self, x: torch.Tensor, intensity_map: torch.Tensor = None,
                city_ids: torch.Tensor = None):
        """
        Args:
            x:             (B, C, H, W) input image (C=3 or 4 with contrast)
            intensity_map: (B, 1, H, W) raw grayscale intensity [0,1]
            city_ids:      (B,) long tensor

        Returns:
            dict with keys:
                'predictions': dict {'p1'...'p6'}, each (B, num_classes, H_i, W_i)
                'kl_losses':   dict of band → scalar KL losses.
                               May also contain:
                                 'bypass_alpha' [B] — module bypass gate values
                                 'ref_predictions' — dict {p1..p6} from CACR
                                                     reference path (training only)
                'sag_gates':   dict (reserved for analysis)
        """
        # ── Encoder ──
        encoder_out = self.encoder(x)
        feat1 = encoder_out['feat1']
        feat2 = encoder_out['feat2']
        feat3 = encoder_out['feat3']
        feat4 = encoder_out['feat4']
        feat5 = encoder_out['feat5']

        kl_losses = {}

        # ── SIB on feat4 ──
        feat4_pre_sib = feat4  # save for module bypass gate (and CACR ref)
        if self.use_sib:
            feat4, sib_kl = self.sib(feat4, intensity_map, city_ids)
            kl_losses.update(sib_kl)

            # Module bypass gate: wraps entire SIB pipeline (main path)
            if self.use_module_bypass:
                feat4, bypass_alpha = self.module_bypass_gate(feat4, feat4_pre_sib)
                kl_losses['bypass_alpha'] = bypass_alpha

        # ── SAG on feat1-3 (skip connection filtering) ──
        if self.use_sag:
            feat1 = self.sag1(feat1)
            feat2 = self.sag2(feat2)
            feat3 = self.sag3(feat3)

        # ── Optional multi-scale SIB on feat1-3 ──
        if self.use_multiscale_sib:
            [feat1, feat2, feat3], ms_kl = self.ms_sib(
                [feat1, feat2, feat3], intensity_map, city_ids
            )
            kl_losses.update(ms_kl)

        # ── Reassemble feature dict for DFFM (main path) ──
        features = {
            'feat1': feat1, 'feat2': feat2, 'feat3': feat3,
            'feat4': feat4, 'feat5': feat5,
        }

        fused = self.dffm(features)
        decoded = self.decoder(fused)
        predictions = self.oam(decoded)

        # ── CACR reference forward pass (training only) ─────────────
        # Run a second pass through DFFM → Decoder → OAM using the
        # pre-augmentation feat4 from SIB. The asymmetric penalty
        # between main p6 and ref p6 is computed in the training loop
        # via CACRLoss. The reference path uses no_grad — no gradient
        # flows through it.
        #
        # Key design choices:
        #   - pre_aug feat4 is detached → no gradient through ref path
        #   - Same skip features (post-SAG) used for both passes
        #   - All 6 ref predictions returned for symmetry; loss uses p6
        # ────────────────────────────────────────────────────────────
        if self.use_cacr and self.training and 'pre_aug_bottleneck' in kl_losses:
            pre_aug_feat4 = kl_losses.pop('pre_aug_bottleneck')

            # Apply module bypass gate to ref path too (if active),
            # using detached pre-SIB features as the "original" anchor.
            if self.use_module_bypass:
                pre_aug_feat4, _ = self.module_bypass_gate(
                    pre_aug_feat4, feat4_pre_sib.detach())

            with torch.no_grad():
                ref_features = {
                    'feat1': feat1, 'feat2': feat2, 'feat3': feat3,
                    'feat4': pre_aug_feat4, 'feat5': feat5,
                }
                ref_fused = self.dffm(ref_features)
                ref_decoded = self.decoder(ref_fused)
                ref_predictions = self.oam(ref_decoded)
            kl_losses['ref_predictions'] = ref_predictions

        return {
            'predictions': predictions,
            'kl_losses': kl_losses,
            'sag_gates': {},
        }

    def get_trainable_params(self, lr: float = 0.0003, encoder_lr_mult: float = 0.1):
        """Parameter groups with different learning rates."""
        encoder_params = list(self.encoder.parameters())
        encoder_param_ids = {id(p) for p in encoder_params}

        other_params = [p for p in self.parameters()
                        if id(p) not in encoder_param_ids and p.requires_grad]

        return [
            {'params': encoder_params, 'lr': lr * encoder_lr_mult},
            {'params': other_params, 'lr': lr},
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# Loss wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class OGLANetSIBLoss(nn.Module):
    """
    Combined loss: OGLANet deep supervision + KL regularization.

    OGLANetLoss expects dict {'p1'...'p6'} and masks.
    This wrapper adds the weighted KL losses from SIB.

    NEW — Skips 'ref_predictions' (CACR ref path) in addition to the
    existing 'bypass_alpha' diagnostic key.
    """

    def __init__(self, num_classes: int = 2, lambda_kl: float = 0.001,
                 kl_warmup_epochs: int = 10):
        super().__init__()
        self.seg_loss = OGLANetLoss()
        self.lambda_kl = lambda_kl
        self.kl_warmup_epochs = kl_warmup_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    @property
    def kl_weight(self) -> float:
        if self.kl_warmup_epochs <= 0:
            return self.lambda_kl
        progress = min(self._current_epoch / self.kl_warmup_epochs, 1.0)
        return self.lambda_kl * progress

    def forward(self, predictions: dict, masks: torch.Tensor,
                kl_losses: dict = None):
        """
        Args:
            predictions: dict {'p1'...'p6'}, each (B, C, H_i, W_i)
            masks:       (B, H, W) long tensor
            kl_losses:   dict from SIB, band → scalar KL.
                         May contain non-loss keys ('bypass_alpha',
                         'ref_predictions') which are skipped.

        Returns:
            dict with keys:
                'total':       scalar total loss
                'seg':         scalar segmentation loss
                'kl':          scalar total KL loss (raw)
                'kl_weighted': KL × current weight (for monitoring)
                'seg_dict':    per-level seg losses from OGLANetLoss
        """
        seg_out = self.seg_loss(predictions, masks)
        seg_total = seg_out['total']

        # KL loss — skip non-loss keys (bypass_alpha, ref_predictions)
        _NON_LOSS_KEYS = {'bypass_alpha', 'ref_predictions',
                          'pre_aug_bottleneck'}
        kl_total = torch.tensor(0.0, device=masks.device)
        if kl_losses:
            for band, kl_val in kl_losses.items():
                if band in _NON_LOSS_KEYS:
                    continue
                if isinstance(kl_val, torch.Tensor) and kl_val.dim() == 0:
                    kl_total = kl_total + kl_val

        weighted_kl = self.kl_weight * kl_total
        total = seg_total + weighted_kl

        return {
            'total': total,
            'seg': seg_total,
            'kl': kl_total,
            'kl_weighted': weighted_kl,
            'seg_dict': seg_out,
        }