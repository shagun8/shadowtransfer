"""
Per-Image Mean Loss Functions for MAMNet + MRFP
================================================
Computes cross-entropy per pixel → averages over H×W per image → averages
over the batch.  This matches the per-image evaluation convention used in
major shadow-detection papers (BDRAR, MTMT, DSDNet, SCOTCH).

For fixed-size images the numerical result equals the default global mean,
but the explicit per-image formulation is kept for clarity and correctness
if image sizes ever vary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerImageCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss averaged **per image**, then across the batch.

    Steps:
        1. Pixel-level CE with reduction='none' → [B, H, W]
        2. Mean over spatial dims per image    → [B]
        3. Mean over batch                     → scalar
    """

    def __init__(self, weight=None, ignore_index=-100):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """
        Args:
            pred   : [B, C, H, W]  logits
            target : [B, H, W]     class indices
        """
        loss_map = F.cross_entropy(
            pred, target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none')                       # [B, H, W]

        per_image = loss_map.mean(dim=(1, 2))       # [B]
        return per_image.mean()                     # scalar


class MAMNetMRFPLoss(nn.Module):
    """
    Complete per-image-mean loss for MAMNet + MRFP with deep supervision.

    Total = main_CE  +  aux_weight × mean(aux1_CE, aux2_CE, aux3_CE)

    All CE terms are per-image means.
    """

    def __init__(self, aux_weight=0.4, weight=None):
        super().__init__()
        self.aux_weight = aux_weight
        self.criterion = PerImageCrossEntropyLoss(weight=weight)

    def forward(self, outputs, target):
        """
        Args:
            outputs : dict  {'main', 'aux1', 'aux2', 'aux3'} (training)
                      or    Tensor [B, C, H, W]               (eval)
            target  : [B, H, W]

        Returns:
            dict {'total', 'main', 'aux', 'aux1', 'aux2', 'aux3'}
            (eval mode only returns 'total' and 'main')
        """
        if isinstance(outputs, dict):
            main_loss = self.criterion(outputs['main'], target)

            if 'aux1' in outputs:
                aux1 = self.criterion(outputs['aux1'], target)
                aux2 = self.criterion(outputs['aux2'], target)
                aux3 = self.criterion(outputs['aux3'], target)
                aux_mean = (aux1 + aux2 + aux3) / 3.0
                total = main_loss + self.aux_weight * aux_mean

                return {
                    'total': total,
                    'main':  main_loss,
                    'aux':   aux_mean,
                    'aux1':  aux1,
                    'aux2':  aux2,
                    'aux3':  aux3,
                }
            return {'total': main_loss, 'main': main_loss}
        else:
            # Eval mode — outputs is a raw tensor
            main_loss = self.criterion(outputs, target)
            return {'total': main_loss, 'main': main_loss}


# ======================================================================
if __name__ == "__main__":
    criterion = MAMNetMRFPLoss(aux_weight=0.4)

    outputs = {
        'main': torch.randn(4, 2, 256, 256),
        'aux1': torch.randn(4, 2, 256, 256),
        'aux2': torch.randn(4, 2, 256, 256),
        'aux3': torch.randn(4, 2, 256, 256),
    }
    target = torch.randint(0, 2, (4, 256, 256))

    losses = criterion(outputs, target)
    print("Per-image-mean losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")