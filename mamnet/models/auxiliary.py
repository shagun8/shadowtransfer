"""
Auxiliary Branch Module (AUX)
Provides additional supervision during training to preserve
information from different decoder stages.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryBranch(nn.Module):
    """
    Auxiliary Branch for shadow detection.
    
    Structure (from Figure 5):
    Input -> Conv3x3 -> BN -> ReLU -> Dropout -> Conv1x1 -> Upsample -> Output
    
    Dropout rate: 0.3 (standard for segmentation auxiliary heads)
    Output channels: 2 (shadow vs non-shadow binary classification)
    """
    
    def __init__(self, in_channels, num_classes=2, dropout_rate=0.3):
        super(AuxiliaryBranch, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)
        
    def forward(self, x, target_size):
        """
        Args:
            x: Input features [B, C, H, W]
            target_size: Tuple (H, W) for final output size
            
        Returns:
            Prediction map upsampled to target_size [B, num_classes, H, W]
        """
        x = self.conv(x)
        x = self.classifier(x)
        
        # Upsample to target size (original image resolution)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


class AuxiliaryModule(nn.Module):
    """
    Complete auxiliary module with 3 branches as shown in Figure 3a.
    
    Three auxiliary branches connected to decoder stages:
    - AUX1: From decoder stage 3 (128 channels, 64x64)
    - AUX2: From decoder stage 2 (128 channels, 64x64)  
    - AUX3: From decoder stage 1 (256 channels, 32x32)
    """
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(AuxiliaryModule, self).__init__()
        
        # Three auxiliary branches with different input channels
        # Based on decoder channel dimensions from Figure 3a
        self.aux1 = AuxiliaryBranch(256, num_classes, dropout_rate)  # From decoder stage 1: 256x32x32
        self.aux2 = AuxiliaryBranch(128, num_classes, dropout_rate)  # From decoder stage 2: 128x64x64
        self.aux3 = AuxiliaryBranch(64, num_classes, dropout_rate)   # From decoder stage 3: 64x128x128
        
    def forward(self, dec_feat1, dec_feat2, dec_feat3, target_size):
        """
        Args:
            dec_feat1: Decoder features from stage 1 [B, 256, 32, 32]
            dec_feat2: Decoder features from stage 2 [B, 128, 64, 64]
            dec_feat3: Decoder features from stage 3 [B, 64, 128, 128]
            target_size: Target output size (H, W)
            
        Returns:
            Dictionary of auxiliary predictions
        """
        # Match auxiliary branches to correct decoder features by channel count
        aux_out1 = self.aux1(dec_feat1, target_size)  # aux1 (256ch) <- dec_feat1 (256ch)
        aux_out2 = self.aux2(dec_feat2, target_size)  # aux2 (128ch) <- dec_feat2 (128ch)
        aux_out3 = self.aux3(dec_feat3, target_size)  # aux3 (64ch) <- dec_feat3 (64ch)
        
        return {
            'aux1': aux_out1,
            'aux2': aux_out2,
            'aux3': aux_out3
        }


if __name__ == "__main__":
    # Test auxiliary module
    aux_module = AuxiliaryModule(num_classes=2, dropout_rate=0.3)
    
    # Simulate decoder features
    dec_feat1 = torch.randn(2, 256, 32, 32)
    dec_feat2 = torch.randn(2, 128, 64, 64)
    dec_feat3 = torch.randn(2, 64, 128, 128)
    
    target_size = (256, 256)
    
    aux_outputs = aux_module(dec_feat1, dec_feat2, dec_feat3, target_size)
    
    print("Auxiliary outputs:")
    for key, val in aux_outputs.items():
        print(f"{key}: {val.shape}")