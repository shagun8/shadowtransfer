"""
Multi-Scale Spatial Channel Attention Fusion Module (MSCAF)
Captures shadow features at multiple scales with both spatial and channel attention.
"""

import torch
import torch.nn as nn
from .attention import ChannelAttention, SpatialAttention


class MSCAF(nn.Module):
    """
    Multi-Scale Spatial Channel Attention Fusion Module
    
    Uses 5 parallel branches:
    - 4 branches with dilated convolutions (rates: 1, 12, 24, 36)
    - 1 global pooling branch
    
    All branches include channel attention, then features are fused
    and passed through spatial attention.
    
    Input/Output: [B, 512, 16, 16] (for 256x256 input images)
    """
    
    def __init__(self, in_channels=512):
        super(MSCAF, self).__init__()
        
        self.in_channels = in_channels
        
        # Branch 1: Dilated conv with rate=1 (standard conv)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.ca1 = ChannelAttention(in_channels, reduction_ratio=16)
        
        # Branch 2: Dilated conv with rate=12
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.ca2 = ChannelAttention(in_channels, reduction_ratio=16)
        
        # Branch 3: Dilated conv with rate=24
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=24, dilation=24, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.ca3 = ChannelAttention(in_channels, reduction_ratio=16)
        
        # Branch 4: Dilated conv with rate=36
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=36, dilation=36, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.ca4 = ChannelAttention(in_channels, reduction_ratio=16)
        
        # Branch 5: Global Average Pooling branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.ca5 = ChannelAttention(in_channels, reduction_ratio=16)
        
        # Fusion: concatenate 5 branches (5 * in_channels)
        # Then reduce back to in_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Spatial attention after fusion
        self.sa = SpatialAttention(kernel_size=7)
        
        # Final 3x3 conv to refine features
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 512, H, W]
        Returns:
            Multi-scale fused features [B, 512, H, W]
        """
        _, _, H, W = x.size()
        
        # Branch 1: rate=1
        out1 = self.branch1(x)
        out1 = self.ca1(out1)
        
        # Branch 2: rate=12
        out2 = self.branch2(x)
        out2 = self.ca2(out2)
        
        # Branch 3: rate=24
        out3 = self.branch3(x)
        out3 = self.ca3(out3)
        
        # Branch 4: rate=36
        out4 = self.branch4(x)
        out4 = self.ca4(out4)
        
        # Branch 5: Global pooling
        out5 = self.gap(x)  # [B, 512, 1, 1]
        out5 = self.branch5(out5)
        out5 = self.ca5(out5)
        out5 = nn.functional.interpolate(out5, size=(H, W), mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)  # [B, 512*5, H, W]
        
        # Fuse to original channel size
        out = self.fusion(out)  # [B, 512, H, W]
        
        # Apply spatial attention
        out = self.sa(out)
        
        # Final refinement
        out = self.final_conv(out)
        
        return out


if __name__ == "__main__":
    # Test MSCAF module
    mscaf = MSCAF(in_channels=512)
    x = torch.randn(2, 512, 16, 16)
    out = mscaf(x)
    print(f"MSCAF input: {x.shape}")
    print(f"MSCAF output: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in mscaf.parameters())
    print(f"MSCAF parameters: {total_params:,}")