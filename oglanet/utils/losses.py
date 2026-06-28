"""
Loss functions for OGLANet.
Uses Binary Cross-Entropy Loss for all 6 prediction outputs.
Based on paper Section 3.2 and Figure 7.

NEW — Diagnostic-motivated losses (§4.3 orphan coverage):
  - CACRLoss:   Class-Asymmetric Confidence Regularizer
  - CEAURCLoss: Differentiable surrogate for CE-AURC on gt_shadow population
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OGLANetLoss(nn.Module):
    """
    Complete loss for OGLANet with 6 prediction outputs.

    From Figure 7:
    - Loss1, Loss2, Loss3, Loss4, Loss5, Loss6 correspond to P1-P6
    - Total loss = Loss1 + Loss2 + Loss3 + Loss4 + Loss5 + Loss6

    Following paper Section 3.2 (hyperparameter configuration):
    - Uses Binary Cross-Entropy Loss (Equation 5 and 6)
    """

    def __init__(self, weight=None):
        """
        Args:
            weight: Optional class weights for BCE loss
        """
        super(OGLANetLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def forward(self, predictions, target):
        """
        Args:
            predictions: Dict containing 6 predictions
                        {'p1', 'p2', 'p3', 'p4', 'p5', 'p6'}
                        Each is [B, 2, H, W]
            target: Ground truth [B, H, W] with values {0, 1}

        Returns:
            Dict containing individual losses and total loss
        """
        # Calculate loss for each prediction
        loss1 = self.criterion(predictions['p1'], target)
        loss2 = self.criterion(predictions['p2'], target)
        loss3 = self.criterion(predictions['p3'], target)
        loss4 = self.criterion(predictions['p4'], target)
        loss5 = self.criterion(predictions['p5'], target)
        loss6 = self.criterion(predictions['p6'], target)

        # Total loss (sum of all 6 losses)
        total_loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return {
            'total': total_loss,
            'loss1': loss1,
            'loss2': loss2,
            'loss3': loss3,
            'loss4': loss4,
            'loss5': loss5,
            'loss6': loss6
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
#   For OGLANet, both the main and reference paths produce a dict of
#   predictions {p1..p6}. CACR is computed on p6 (the highest-resolution
#   main output) since that's the deployment-relevant prediction.
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
            main_logits: [B, 2, H, W] logits from augmented/main path (p6).
            ref_logits:  [B, 2, H, W] logits from reference path (p6, detached).
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
            logits:  [B, 2, H, W] raw logits (typically p6).
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
    # Test loss function
    criterion = OGLANetLoss()

    # Simulate predictions
    predictions = {
        'p1': torch.randn(2, 2, 384, 384),
        'p2': torch.randn(2, 2, 384, 384),
        'p3': torch.randn(2, 2, 384, 384),
        'p4': torch.randn(2, 2, 384, 384),
        'p5': torch.randn(2, 2, 384, 384),
        'p6': torch.randn(2, 2, 384, 384)
    }
    target = torch.randint(0, 2, (2, 384, 384))

    losses = criterion(predictions, target)

    print("OGLANetLoss values:")
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