"""
Omni-scale Aggregation Module (OAM)
Multi-scale prediction with 6 outputs (P1-P6) and corresponding losses.
Based on Figure 7 in the paper.
"""

import torch
import torch.nn as nn


class OAM(nn.Module):
    """
    Omni-scale Aggregation Module for multi-scale shadow prediction.
    
    From Figure 7:
    - P1, P2, P3, P4: Direct predictions from S1_D, S2_D, S3_D, S4_D
    - P5: Concatenation of [P1, P2, P3, P4]
    - P6: Concatenation of [P1, P2, P3, P4, P5]
    - Each P has a corresponding loss (Loss1-Loss6)
    
    ASSUMPTION: Each prediction head (P1-P6) is a simple Conv 1×1
    to produce 2-channel output (background/shadow classes).
    """
    
    def __init__(self, num_classes=2, target_size=(384, 384)):
        """
        Args:
            num_classes: Number of classes (default: 2 for shadow detection)
            target_size: Target spatial size for all predictions
        """
        super(OAM, self).__init__()
        
        self.num_classes = num_classes
        self.target_size = target_size
        
        # Prediction heads for P1-P4 (from decoder features)
        self.pred_p1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.pred_p2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.pred_p3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.pred_p4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Prediction head for P5 (concatenation of P1-P4)
        # After concatenating 4 predictions, each with num_classes channels
        self.pred_p5 = nn.Conv2d(num_classes * 4, num_classes, kernel_size=1)
        
        # Prediction head for P6 (concatenation of P1-P5)
        self.pred_p6 = nn.Conv2d(num_classes * 5, num_classes, kernel_size=1)
        
    def forward(self, decoder_features):
        """
        Args:
            decoder_features: Dict with upsampled decoder features
                              ['s1_d_up', 's2_d_up', 's3_d_up', 's4_d_up']
        
        Returns:
            Dict with 6 predictions [p1, p2, p3, p4, p5, p6]
        """
        s1_d = decoder_features['s1_d_up']  # [B, 64, 384, 384]
        s2_d = decoder_features['s2_d_up']  # [B, 128, 96, 96]
        s3_d = decoder_features['s3_d_up']  # [B, 256, 48, 48]
        s4_d = decoder_features['s4_d_up']  # [B, 512, 12, 12]
        
        # Generate P1-P4 predictions
        p1 = self.pred_p1(s1_d)  # [B, 2, 384, 384]
        
        # Upsample P2, P3, P4 to target size
        p2_raw = self.pred_p2(s2_d)  # [B, 2, 96, 96]
        p2 = nn.functional.interpolate(p2_raw, size=self.target_size, 
                                       mode='bilinear', align_corners=False)
        
        p3_raw = self.pred_p3(s3_d)  # [B, 2, 48, 48]
        p3 = nn.functional.interpolate(p3_raw, size=self.target_size, 
                                       mode='bilinear', align_corners=False)
        
        p4_raw = self.pred_p4(s4_d)  # [B, 2, 12, 12]
        p4 = nn.functional.interpolate(p4_raw, size=self.target_size, 
                                       mode='bilinear', align_corners=False)
        
        # P5: Concatenate P1-P4 (Formula 3 in paper)
        p5_concat = torch.cat([p1, p2, p3, p4], dim=1)  # [B, 8, 384, 384]
        p5 = self.pred_p5(p5_concat)  # [B, 2, 384, 384]
        
        # P6: Concatenate P1-P5 (Formula 4 in paper)
        p6_concat = torch.cat([p1, p2, p3, p4, p5], dim=1)  # [B, 10, 384, 384]
        p6 = self.pred_p6(p6_concat)  # [B, 2, 384, 384]
        
        return {
            'p1': p1,
            'p2': p2,
            'p3': p3,
            'p4': p4,
            'p5': p5,
            'p6': p6
        }


if __name__ == "__main__":
    # Test OAM
    oam = OAM(num_classes=2, target_size=(384, 384))
    
    # Simulate decoder features
    decoder_feats = {
        's1_d_up': torch.randn(2, 64, 384, 384),
        's2_d_up': torch.randn(2, 128, 96, 96),
        's3_d_up': torch.randn(2, 256, 48, 48),
        's4_d_up': torch.randn(2, 512, 12, 12)
    }
    
    predictions = oam(decoder_feats)
    
    print("OAM Predictions:")
    for name, pred in predictions.items():
        print(f"{name}: {pred.shape}")