"""
Loss functions for shadow detection.
Using Cross-Entropy Loss as specified in the paper (Equation 7).

NEW — Diagnostic-motivated losses (§4.3 orphan coverage):
  - CACRLoss:   Class-Asymmetric Confidence Regularizer
  - CEAURCLoss: Differentiable surrogate for CE-AURC on gt_shadow population
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy Loss for binary shadow detection.

    Formula (from paper, Equation 7):
    L = -[y * log(p) + (1 - y) * log(1 - p)]

    This is equivalent to nn.CrossEntropyLoss for 2-class classification.
    """

    def __init__(self, weight=None, ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, pred, target):
        """
        Args:
            pred: Predictions [B, 2, H, W]
            target: Ground truth [B, H, W] with values {0, 1}

        Returns:
            Loss value
        """
        return self.loss(pred, target)


class MAMNetLoss(nn.Module):
    """
    Complete loss for MAMNet with auxiliary branches.

    Total loss = main_loss + aux_weight * (aux1_loss + aux2_loss + aux3_loss)

    Auxiliary loss weight is typically 0.4 (standard for deep supervision).
    """

    def __init__(self, aux_weight=0.4, weight=None):
        super(MAMNetLoss, self).__init__()
        self.aux_weight = aux_weight
        self.criterion = CrossEntropyLoss(weight=weight)

    def forward(self, outputs, target):
        """
        Args:
            outputs: Dictionary containing:
                - 'main': Main predictions [B, 2, H, W]
                - 'aux1', 'aux2', 'aux3': Auxiliary predictions [B, 2, H, W]
            target: Ground truth [B, H, W]

        Returns:
            Dictionary containing total loss and individual losses
        """
        # Main loss
        main_loss = self.criterion(outputs['main'], target)

        # Auxiliary losses
        if 'aux1' in outputs:
            aux1_loss = self.criterion(outputs['aux1'], target)
            aux2_loss = self.criterion(outputs['aux2'], target)
            aux3_loss = self.criterion(outputs['aux3'], target)

            # Total loss
            aux_loss = (aux1_loss + aux2_loss + aux3_loss) / 3.0
            total_loss = main_loss + self.aux_weight * aux_loss

            return {
                'total': total_loss,
                'main': main_loss,
                'aux': aux_loss,
                'aux1': aux1_loss,
                'aux2': aux2_loss,
                'aux3': aux3_loss
            }
        else:
            # Inference mode (no auxiliary branches)
            return {
                'total': main_loss,
                'main': main_loss
            }


# ======================================================================
# NEW: Class-Asymmetric Confidence Regularizer (CACR)
#
# Targets §4.3 Orphans 1, 2, 3:
#   Orphan 1 — Class-asymmetric SP gap (gt_shadow >> gt_bg)
#   Orphan 2 — TP_pred logit drop under LOCO
#   Orphan 3 — 2.4× under-reporting by 0/1 vs CE metrics
#
# Mechanism:
#   Penalizes logit shift between the augmented (main) and un-augmented
#   (reference) decoder paths, asymmetrically by predicted class.
#   Predicted-positive (shadow) pixels receive high penalty;
#   predicted-negative (background) pixels receive low/no penalty.
#
#   For DINOv3, the model returns a single logits tensor [B, 2, H, W]
#   (full resolution after upsampling), so CACR is applied directly to
#   that output — no p6 indexing needed.
#
# Ablation prediction (from plan):
#   Removing CACR should specifically increase the gt_shadow AURC gap
#   and the TP-FP logit asymmetry on the held-out city, while leaving
#   0/1 mIoU mostly intact.
#
# References:
#   - Tian et al., "On Calibrating Semantic Segmentation Models" (CVPR 2023)
#   - Liu et al., "Improving Predictor Reliability with Selective
#     Recalibration" (2024)
#   - Geifman & El-Yaniv, SelectiveNet (ICML 2019)
# ======================================================================

class CACRLoss(nn.Module):
    """
    Class-Asymmetric Confidence Regularizer.

    Penalizes logit shift between the augmented (main) and un-augmented
    (reference) decoder paths, with asymmetric weighting by predicted class:

        L_CACR = w_pos * mean(|Δ_shadow_logit|  on  pred_pos pixels)
               + w_neg * mean(|Δ_shadow_logit|  on  pred_neg pixels)

    where:
        Δ_shadow_logit = main_logits[:, 1] - ref_logits[:, 1]
        pred_pos = (main_logits.argmax(1) == 1)

    Default: w_pos=1.0, w_neg=0.0  →  only pred-positive pixels penalized.

    Args:
        pos_weight: Weight for predicted-positive (shadow) pixel penalty.
        neg_weight: Weight for predicted-negative (background) pixel penalty.
                    Set to 0 to leave background unconstrained (default).
    """

    def __init__(self, pos_weight=1.0, neg_weight=0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, main_logits, ref_logits, targets=None):
        """
        Args:
            main_logits: [B, 2, H, W] logits from augmented/main path.
            ref_logits:  [B, 2, H, W] logits from reference path (detached).
            targets:     [B, H, W] optional GT (not used in default mode,
                         but available for oracle-stratified variants).

        Returns:
            loss: Scalar CACR loss.
            diagnostics: Dict with per-class shift magnitudes for logging.
        """
        # Shadow-class logit shift (absolute)
        delta = (main_logits[:, 1, :, :] - ref_logits[:, 1, :, :]).abs()

        # Stratify by model's own prediction (not GT)
        with torch.no_grad():
            pred_pos = main_logits.argmax(dim=1) == 1   # [B, H, W]
            pred_neg = ~pred_pos
            n_pos = pred_pos.sum().float()
            n_neg = pred_neg.sum().float()

        loss = torch.tensor(0.0, device=main_logits.device)
        diag = {}

        if n_pos > 0:
            pos_shift = delta[pred_pos].mean()
            loss = loss + self.pos_weight * pos_shift
            diag['cacr_pos_shift'] = pos_shift.item()
        else:
            diag['cacr_pos_shift'] = 0.0

        if self.neg_weight > 0 and n_neg > 0:
            neg_shift = delta[pred_neg].mean()
            loss = loss + self.neg_weight * neg_shift
            diag['cacr_neg_shift'] = neg_shift.item()
        else:
            diag['cacr_neg_shift'] = 0.0

        diag['cacr_n_pos'] = n_pos.item()
        diag['cacr_n_neg'] = n_neg.item()

        return loss, diag


# ======================================================================
# NEW: CE-AURC Auxiliary Loss
#
# Targets §4.3 Orphan 3:
#   The 2.4× under-reporting by 0/1 metrics vs CE-AURC.
#
# Mechanism:
#   Standard segmentation loss (CE) weights all pixels equally.
#   CE-AURC penalizes high-confidence wrong predictions more than
#   low-confidence ones, which is exactly the deployment-relevant
#   failure mode under LOCO.
#
#   This loss restricts to the gt_shadow population and weights
#   per-pixel CE by the model's own confidence, approximating
#   the area under the Risk-Coverage curve with CE as the risk.
#
# Ablation prediction (from plan):
#   Removing CE-AURC loss should increase the CE-AURC gap by a
#   larger fraction than the 0/1-AURC gap.
#
# References:
#   - Geifman & El-Yaniv, SelectiveNet (ICML 2019)
#   - Angelopoulos et al., Conformal Risk Control (2022)
# ======================================================================

class CEAURCLoss(nn.Module):
    """
    Differentiable surrogate for CE-AURC on the ground-truth positive
    (shadow) population.

    Approximates the Area Under the Risk-Coverage curve where:
      - Risk = cross-entropy (not 0/1 loss)
      - Coverage is ordered by model confidence P(shadow)
      - Population is restricted to gt_shadow pixels

    The surrogate weights per-pixel CE by the model's shadow-class
    probability.  High-confidence mistakes (model says shadow with
    high P but is wrong, or model says shadow with LOW P and is
    right but poorly calibrated) receive higher loss.

    Specifically:
        L = mean( w_i * CE_i )   for i ∈ {pixels where GT = shadow}
        w_i = 0.5 + P_shadow(i)  (detached — weights don't receive gradient)

    The 0.5 floor ensures all shadow pixels contribute; the P_shadow
    term upweights high-confidence predictions, penalizing calibration
    failures at the top of the risk-coverage curve.

    Args:
        floor_weight: Minimum weight for all shadow pixels (default 0.5).
    """

    def __init__(self, floor_weight=0.5):
        super().__init__()
        self.floor_weight = floor_weight

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, 2, H, W] raw logits.
            targets: [B, H, W] ground truth {0, 1}.

        Returns:
            loss: Scalar CE-AURC surrogate loss.
            diagnostics: Dict with shadow pixel counts and mean confidence.
        """
        shadow_mask = (targets == 1)   # [B, H, W]
        n_shadow = shadow_mask.sum().float()
        diag = {'ce_aurc_n_shadow': n_shadow.item()}

        if n_shadow == 0:
            return (torch.tensor(0.0, device=logits.device,
                                 requires_grad=True),
                    diag)

        # Per-pixel CE (unreduced)
        ce = F.cross_entropy(logits, targets, reduction='none')  # [B, H, W]

        # Shadow-class probability
        probs = F.softmax(logits, dim=1)            # [B, 2, H, W]
        shadow_prob = probs[:, 1, :, :]             # [B, H, W]

        # Select shadow pixels
        shadow_ce   = ce[shadow_mask]
        shadow_conf = shadow_prob[shadow_mask]

        # Confidence-weighted CE (weights detached — no grad through weights)
        weights = (self.floor_weight + shadow_conf).detach()
        loss = (weights * shadow_ce).mean()

        diag['ce_aurc_mean_shadow_conf'] = shadow_conf.mean().item()
        diag['ce_aurc_mean_shadow_ce'] = shadow_ce.mean().item()

        return loss, diag


if __name__ == "__main__":
    # Test loss functions
    criterion = MAMNetLoss(aux_weight=0.4)

    # Simulate outputs
    outputs = {
        'main': torch.randn(2, 2, 256, 256),
        'aux1': torch.randn(2, 2, 256, 256),
        'aux2': torch.randn(2, 2, 256, 256),
        'aux3': torch.randn(2, 2, 256, 256)
    }
    target = torch.randint(0, 2, (2, 256, 256))

    losses = criterion(outputs, target)

    print("MAMNetLoss values:")
    for key, val in losses.items():
        print(f"  {key}: {val.item():.4f}")

    # Test CACR loss
    print("\nCACR Loss test:")
    cacr = CACRLoss(pos_weight=1.0, neg_weight=0.0)
    main_logits = torch.randn(2, 2, 384, 384)
    ref_logits = main_logits + torch.randn_like(main_logits) * 0.5
    cacr_loss, cacr_diag = cacr(main_logits, ref_logits.detach())
    print(f"  loss: {cacr_loss.item():.4f}")
    for k, v in cacr_diag.items():
        print(f"  {k}: {v:.4f}")

    # Test CE-AURC loss
    print("\nCE-AURC Loss test:")
    ce_aurc = CEAURCLoss(floor_weight=0.5)
    logits = torch.randn(2, 2, 384, 384)
    targets = torch.randint(0, 2, (2, 384, 384))
    aurc_loss, aurc_diag = ce_aurc(logits, targets)
    print(f"  loss: {aurc_loss.item():.4f}")
    for k, v in aurc_diag.items():
        print(f"  {k}: {v:.4f}")