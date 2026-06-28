"""
Global Feature Extraction Module (GFEM)
Self-attention mechanism for capturing global context.
Based on Figure 3(b) in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GFEM(nn.Module):
    """
    Global Feature Extraction Module using self-attention.
    
    Architecture (from Figure 3b):
    1. Key map: Input → Conv 1×1 → Concatenate with Query
    2. Attention: Concatenated → Sigmoid
    3. Value map: Input → Conv 1×1
    4. Weighted features: Attention ⊗ Value
    5. Aggregate: Weighted ⊕ Key map
    6. Downsample: Conv 3×3 stride=2
    """
    
    def __init__(self, in_channels, downsample=True):
        super(GFEM, self).__init__()
        
        # 1×1 convolutions for Key and Value transformations
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Conditionally apply downsampling
        if downsample:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C, H, W]
            
        Returns:
            Output feature map [B, C, H/2, W/2] (downsampled)
        """
        B, C, H, W = x.size()
        
        # Generate Key map with 1×1 conv
        key = self.key_conv(x)  # [B, C, H, W]
        
        # Query is the original input (as shown in Figure 3b)
        query = x  # [B, C, H, W]
        
        # Concatenate Key and Query
        # According to formula (1): Con(Conv1*1(K), Q)
        key_query = torch.cat([key, query], dim=1)  # [B, 2C, H, W]
        
        # Apply Sigmoid to get attention weights
        # Reshape for attention computation
        key_query = key_query.view(B, 2*C, H*W)  # [B, 2C, H*W]
        attention = torch.sigmoid(key_query)  # [B, 2C, H*W]
        
        # Generate Value map with 1×1 conv
        value = self.value_conv(x)  # [B, C, H, W]
        value = value.view(B, C, H*W)  # [B, C, H*W]
        
        # Multiply attention with value (element-wise)
        # Take only first C channels of attention (from Key part)
        attention_key = attention[:, :C, :]  # [B, C, H*W]
        weighted_value = attention_key * value  # [B, C, H*W]
        
        # Reshape back to spatial dimensions
        weighted_value = weighted_value.view(B, C, H, W)  # [B, C, H, W]
        
        # Concatenate with original Key map (as per Figure 3b)
        aggregated = weighted_value + key  # [B, C, H, W]
        
        # Final convolution with stride=2 for downsampling
        output = self.out_conv(aggregated)  # [B, C, H/2, W/2]
        
        return output


if __name__ == "__main__":
    # Test GFEM
    gfem = GFEM(in_channels=64)
    x = torch.randn(2, 64, 96, 96)
    
    output = gfem(x)
    print(f"GFEM Input: {x.shape}")
    print(f"GFEM Output: {output.shape}")