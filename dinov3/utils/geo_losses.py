"""
Geographic Domain Loss for SegDesicNet.
Combines standard segmentation loss with geographic coordinate prediction loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeographicDomainLoss(nn.Module):
    """
    Combined loss for SegDesicNet training.
    
    Loss = segmentation_loss + alpha * domain_loss
    
    where domain_loss = 1 - cosine_similarity(pred_encoding, gt_encoding)
    """
    
    def __init__(self, segmentation_criterion, alpha=0.5):
        """
        Args:
            segmentation_criterion: Base segmentation loss (e.g., CrossEntropyLoss)
            alpha: Weight for domain loss component
        """
        super().__init__()
        self.segmentation_criterion = segmentation_criterion
        self.alpha = alpha
    
    def forward(self, seg_output, target, pred_encoding, gt_encoding):
        """
        Compute combined loss.
        
        Args:
            seg_output: Segmentation output [B, num_classes, H, W]
            target: Ground truth masks [B, H, W]
            pred_encoding: Predicted coordinate encoding [B, D]
            gt_encoding: Ground truth coordinate encoding [B, D]
        
        Returns:
            Dictionary with 'total', 'seg_loss', and 'domain_loss'
        """
        # Segmentation loss
        seg_loss = self.segmentation_criterion(seg_output, target)
        
        # Domain loss: cosine dissimilarity
        # cosine_similarity returns values in [-1, 1]
        # We want to maximize similarity, so minimize (1 - similarity)
        cos_sim = F.cosine_similarity(pred_encoding, gt_encoding, dim=1)  # [B]
        domain_loss = (1 - cos_sim).mean()  # Average over batch
        
        # Combined loss
        total_loss = seg_loss + self.alpha * domain_loss
        
        return {
            'total': total_loss,
            'seg_loss': seg_loss,
            'domain_loss': domain_loss
        }


class SegDesicLoss(nn.Module):
    """
    Complete loss for SegDesicNet with auxiliary branches.
    """
    
    def __init__(self, aux_weight=0.4, alpha=0.5):
        """
        Args:
            aux_weight: Weight for auxiliary losses
            alpha: Weight for geographic domain loss
        """
        super().__init__()
        self.aux_weight = aux_weight
        self.alpha = alpha
        
        # Base criterion (cross-entropy with ignore_index for potential padding)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    def forward(self, outputs, masks, geo_outputs=None):
        """
        Compute loss with optional geographic domain component.
        
        Args:
            outputs: Model outputs dictionary with 'main' and optionally 'aux1', 'aux2', 'aux3'
            masks: Ground truth masks [B, H, W]
            geo_outputs: Geographic outputs dict with 'pred_encoding' and 'gt_encoding' (optional)
        
        Returns:
            Dictionary with loss components
        """
        losses = {}
        
        # Main segmentation loss
        if isinstance(outputs, dict):
            main_output = outputs['main']
        else:
            main_output = outputs
        
        main_loss = self.criterion(main_output, masks)
        losses['seg_main'] = main_loss
        
        # Auxiliary losses (if available)
        aux_loss = 0
        if isinstance(outputs, dict) and 'aux1' in outputs:
            aux1_loss = self.criterion(outputs['aux1'], masks)
            aux2_loss = self.criterion(outputs['aux2'], masks)
            aux3_loss = self.criterion(outputs['aux3'], masks)
            aux_loss = (aux1_loss + aux2_loss + aux3_loss) / 3.0
            losses['seg_aux'] = aux_loss
        
        # Total segmentation loss
        if aux_loss > 0:
            seg_loss = main_loss + self.aux_weight * aux_loss
        else:
            seg_loss = main_loss
        
        losses['seg_total'] = seg_loss
        
        # Geographic domain loss (if provided)
        if geo_outputs is not None:
            cos_sim = F.cosine_similarity(
                geo_outputs['pred_encoding'], 
                geo_outputs['gt_encoding'], 
                dim=1
            )
            domain_loss = (1 - cos_sim).mean()
            losses['domain_loss'] = domain_loss
            
            # Combined total loss
            losses['total'] = seg_loss + self.alpha * domain_loss
        else:
            losses['total'] = seg_loss
        
        return losses


if __name__ == "__main__":
    # Test losses
    criterion = SegDesicLoss(aux_weight=0.4, alpha=0.5)
    
    # Dummy data
    B, C, H, W = 2, 2, 256, 256
    outputs = {
        'main': torch.randn(B, C, H, W),
        'aux1': torch.randn(B, C, H, W),
        'aux2': torch.randn(B, C, H, W),
        'aux3': torch.randn(B, C, H, W)
    }
    masks = torch.randint(0, 2, (B, H, W))
    
    geo_outputs = {
        'pred_encoding': F.normalize(torch.randn(B, 40), p=2, dim=1),
        'gt_encoding': F.normalize(torch.randn(B, 40), p=2, dim=1)
    }
    
    # Compute loss
    losses = criterion(outputs, masks, geo_outputs)
    
    print("Losses:")
    for key, val in losses.items():
        print(f"  {key}: {val.item():.4f}")