"""
Per-Image Mean Loss for OGLANet + MRFP
=======================================
Computes cross-entropy per pixel → averages over H×W per image →
averages over the batch.

This matches the per-image evaluation convention used in major
shadow-detection papers (BDRAR, MTMT, DSDNet, SCOTCH) and is
consistent with the MAMNetMRFPLoss implementation.

OGLANet uses 6-output deep supervision (P1-P6).
Total loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6,
where every term is a per-image mean CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerImageCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss averaged **per image**, then across the batch.

    Steps:
        1. Pixel-level CE with reduction='none'  →  [B, H, W]
        2. Mean over spatial dims per image       →  [B]
        3. Mean over batch                        →  scalar
    """

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred   : [B, C, H, W]  logits
            target : [B, H, W]     class indices
        """
        loss_map  = F.cross_entropy(
            pred, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none')                    # [B, H, W]

        per_image = loss_map.mean(dim=(1, 2))    # [B]
        return per_image.mean()                  # scalar


class OGLANetMRFPLoss(nn.Module):
    """
    Complete per-image-mean loss for OGLANet + MRFP.

    Training mode  (predictions is a dict of 6 tensors):
        Total = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        All six terms are per-image-mean CE values.

    Eval mode (predictions is a single P6 tensor):
        Returns {'total': ce_loss, 'loss6': ce_loss} for compatibility
        with validation loss tracking.

    The .criterion attribute exposes the underlying PerImageCrossEntropyLoss
    so callers can compute a per-image CE on a raw tensor (used in validate()).
    """

    def __init__(self, weight=None):
        super().__init__()
        self.criterion = PerImageCrossEntropyLoss(weight=weight)

    def forward(self, predictions, target):
        """
        Args:
            predictions : dict {'p1'..'p6'} each [B, 2, H, W]  (training)
                          or   Tensor [B, 2, H, W]               (eval)
            target      : [B, H, W]

        Returns:
            dict {
                'total' : scalar total loss
                'loss1' .. 'loss6' : per-component scalars  (training only)
            }
        """
        if isinstance(predictions, dict):
            loss1 = self.criterion(predictions['p1'], target)
            loss2 = self.criterion(predictions['p2'], target)
            loss3 = self.criterion(predictions['p3'], target)
            loss4 = self.criterion(predictions['p4'], target)
            loss5 = self.criterion(predictions['p5'], target)
            loss6 = self.criterion(predictions['p6'], target)

            total = loss1 + loss2 + loss3 + loss4 + loss5 + loss6

            return {
                'total': total,
                'loss1': loss1,
                'loss2': loss2,
                'loss3': loss3,
                'loss4': loss4,
                'loss5': loss5,
                'loss6': loss6,
            }
        else:
            # Eval mode: predictions is a raw P6 tensor
            loss = self.criterion(predictions, target)
            return {'total': loss, 'loss6': loss}


# ======================================================================
if __name__ == '__main__':
    criterion = OGLANetMRFPLoss()

    # Training mode test
    preds_train = {
        'p1': torch.randn(4, 2, 384, 384),
        'p2': torch.randn(4, 2, 384, 384),
        'p3': torch.randn(4, 2, 384, 384),
        'p4': torch.randn(4, 2, 384, 384),
        'p5': torch.randn(4, 2, 384, 384),
        'p6': torch.randn(4, 2, 384, 384),
    }
    target = torch.randint(0, 2, (4, 384, 384))

    losses = criterion(preds_train, target)
    print('Training per-image-mean losses:')
    for k, v in losses.items():
        print(f'  {k}: {v.item():.4f}')

    # Eval mode test
    p6 = torch.randn(4, 2, 384, 384)
    losses_eval = criterion(p6, target)
    print('\nEval per-image-mean loss:')
    for k, v in losses_eval.items():
        print(f'  {k}: {v.item():.4f}')