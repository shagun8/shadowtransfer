"""
Decoder module for OGLANet.
Progressively upsamples decoder features to original resolution.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder that upsamples features from DFFM to original image resolution.
    
    From Figure 2:
    - Takes S1_D, S2_D, S3_D, S4_D from DFFM
    - Progressively upsamples to original resolution
    - Outputs features for OAM prediction heads
    
    ASSUMPTION: Each decoder stage consists of:
    1. Upsampling (bilinear interpolation)
    2. Convolution block (Conv3×3 + BN + ReLU)
    """
    
    def __init__(self, target_size=(384, 384)):
        """
        Args:
            target_size: Target spatial dimensions (H, W) for final output
        """
        super(Decoder, self).__init__()
        
        self.target_size = target_size
        
        # Decoder convolution blocks after upsampling
        # ASSUMPTION: Use Conv3×3 + BN + ReLU after each upsample
        
        self.conv_s4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.conv_s3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv_s2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.conv_s1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, dffm_features):
        """
        Args:
            dffm_features: Dict with keys ['s4_d', 's3_d', 's2_d', 's1_d']
            
        Returns:
            Dict with upsampled decoder features at multiple scales
        """
        s4_d = dffm_features['s4_d']  # [B, 512, 6, 6]
        s3_d = dffm_features['s3_d']  # [B, 256, 12, 12]
        s2_d = dffm_features['s2_d']  # [B, 128, 48, 48]
        s1_d = dffm_features['s1_d']  # [B, 64, 96, 96]
        
        # Compute intermediate upsampling sizes
        # S4 → S3 resolution
        s4_up = nn.functional.interpolate(s4_d, size=s3_d.shape[2:], 
                                         mode='bilinear', align_corners=False)
        s4_up = self.conv_s4(s4_up)  # [B, 512, 12, 12]
        
        # S3 → S2 resolution
        s3_up = nn.functional.interpolate(s3_d, size=s2_d.shape[2:], 
                                         mode='bilinear', align_corners=False)
        s3_up = self.conv_s3(s3_up)  # [B, 256, 48, 48]
        
        # S2 → S1 resolution
        s2_up = nn.functional.interpolate(s2_d, size=s1_d.shape[2:], 
                                         mode='bilinear', align_corners=False)
        s2_up = self.conv_s2(s2_up)  # [B, 128, 96, 96]
        
        # S1 → Original resolution
        s1_up = nn.functional.interpolate(s1_d, size=self.target_size, 
                                         mode='bilinear', align_corners=False)
        s1_up = self.conv_s1(s1_up)  # [B, 64, 384, 384]
        
        return {
            's4_d_up': s4_up,  # For P4
            's3_d_up': s3_up,  # For P3
            's2_d_up': s2_up,  # For P2
            's1_d_up': s1_up   # For P1
        }


if __name__ == "__main__":
    # Test Decoder
    decoder = Decoder(target_size=(384, 384))
    
    # Simulate DFFM features
    dffm_feats = {
        's4_d': torch.randn(2, 512, 6, 6),
        's3_d': torch.randn(2, 256, 12, 12),
        's2_d': torch.randn(2, 128, 48, 48),
        's1_d': torch.randn(2, 64, 96, 96)
    }
    
    decoder_feats = decoder(dffm_feats)
    
    print("Decoder Outputs:")
    for name, feat in decoder_feats.items():
        print(f"{name}: {feat.shape}")