"""
Decoder for MAMNet
Progressively upsamples features and integrates CCA outputs.
Uses bilinear interpolation for upsampling (standard in semantic segmentation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CrissCrossAttention


class DecoderBlock(nn.Module):
    """
    Single decoder block with upsampling and feature fusion.
    
    Process:
    1. Upsample decoder features
    2. Apply CCA to encoder features
    3. Concatenate upsampled decoder + CCA encoder features
    4. Refine with 3x3 conv
    """
    
    def __init__(self, enc_channels, dec_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # CCA for encoder features
        self.cca = CrissCrossAttention(enc_channels)
        
        # Fusion: concatenate upsampled decoder + CCA encoder
        # Then reduce to out_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(enc_channels + dec_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, enc_feat, dec_feat):
        """
        Args:
            enc_feat: Encoder features from skip connection [B, enc_C, H, W]
            dec_feat: Decoder features from previous stage [B, dec_C, H/2, W/2]
            
        Returns:
            Fused features [B, out_C, H, W]
        """
        # Upsample decoder features to match encoder spatial size
        _, _, H, W = enc_feat.size()
        dec_feat_up = F.interpolate(dec_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Apply CCA to encoder features
        enc_feat_cca = self.cca(enc_feat)
        
        # Concatenate and fuse
        fused = torch.cat([enc_feat_cca, dec_feat_up], dim=1)
        out = self.fusion(fused)
        
        return out


class Decoder(nn.Module):
    """
    Complete Decoder for MAMNet.
    
    Architecture (from Figure 3a):
    - Stage 0: MSCAF output [512, 16, 16] (input to decoder)
    - Stage 1: Upsample + CCA(feat3) -> [256, 32, 32]
    - Stage 2: Upsample + CCA(feat2) -> [128, 64, 64]
    - Stage 3: Upsample + CCA(feat1) -> [64, 128, 128]
    - Stage 4: Upsample + CCA(conv1) -> [64, 256, 256]
    """
    
    def __init__(self, num_classes=2):
        super(Decoder, self).__init__()
        
        # Decoder blocks
        # Stage 1: 512 (MSCAF) + 512 (feat4) -> 256
        self.decoder1 = DecoderBlock(enc_channels=512, dec_channels=512, out_channels=256)

        # Stage 2: 256 + 256 (feat3) -> 128
        self.decoder2 = DecoderBlock(enc_channels=256, dec_channels=256, out_channels=128)

        # Stage 3: 128 + 128 (feat2) -> 64
        self.decoder3 = DecoderBlock(enc_channels=128, dec_channels=128, out_channels=64)

        # Stage 4: 64 + 64 (feat1) -> 64
        self.decoder4 = DecoderBlock(enc_channels=64, dec_channels=64, out_channels=64)
        
        # Final classifier
        self.classifier = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, mscaf_out, enc_feats):
        """
        Args:
            mscaf_out: Output from MSCAF [B, 512, 16, 16]
            enc_feats: Dictionary of encoder features
                - feat1: [B, 64, 64, 64]
                - feat2: [B, 128, 32, 32]
                - feat3: [B, 256, 16, 16]
                
        Returns:
            Dictionary containing:
                - 'main': Main prediction [B, num_classes, 256, 256]
                - 'dec_feat1', 'dec_feat2', 'dec_feat3': Intermediate features for AUX
        """
        # Stage 1: 512 -> 256
        dec1 = self.decoder1(enc_feats['feat4'], mscaf_out)  # Now correct: feat4 is 512x32x32

        # Stage 2: 256 -> 128  
        dec2 = self.decoder2(enc_feats['feat3'], dec1)  # feat3 is 256x64x64

        # Stage 3: 128 -> 64
        dec3 = self.decoder3(enc_feats['feat2'], dec2)  # feat2 is 128x128x128

        # Stage 4: 64 -> 64
        dec4 = self.decoder4(enc_feats['feat1'], dec3)  # feat1 is 64x256x256
        
        # Final classification
        main_out = self.classifier(dec4)
        
        return {
            'main': main_out,
            'dec_feat1': dec1,  # For AUX1
            'dec_feat2': dec2,  # For AUX2
            'dec_feat3': dec3,  # For AUX3
        }


if __name__ == "__main__":
    # Test decoder
    decoder = Decoder(num_classes=2)
    
    # Simulate inputs
    mscaf_out = torch.randn(2, 512, 16, 16)
    enc_feats = {
    'feat4': torch.randn(2, 512, 32, 32),  # ADD THIS
    'feat3': torch.randn(2, 256, 64, 64),  # FIX dimension
    'feat2': torch.randn(2, 128, 128, 128),  # FIX dimension
    'feat1': torch.randn(2, 64, 256, 256),  # FIX dimension
    }
    
    outputs = decoder(mscaf_out, enc_feats)
    
    print("Decoder outputs:")
    for key, val in outputs.items():
        print(f"{key}: {val.shape}")