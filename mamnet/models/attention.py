"""
Attention Modules for MAMNet:
- Channel Attention (CA)
- Spatial Attention (SA)
- Criss-Cross Attention (CCA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (from CBAM)
    Uses both max pooling and average pooling followed by shared MLP.
    Reduction ratio r=16 as per CBAM standard.
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP: C -> C/r -> C
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Generate channel attention weights
        avg_out = self.mlp(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.mlp(self.max_pool(x))  # [B, C, 1, 1]
        
        attention = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (from CBAM)
    Uses both max pooling and average pooling along channel dimension,
    followed by 7x7 convolution.
    """
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Attention-weighted features [B, C, H, W]
        """
        # Aggregate along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        attention = self.sigmoid(self.conv(x_cat))  # [B, 1, H, W]
        
        return x * attention


class CrissCrossAttention(nn.Module):
    """
    Criss-Cross Attention Module (CCNet)
    Captures contextual information from all pixels in the same row and column.
    Channel reduction: C' = C/8 as per CCNet paper.
    """
    
    def __init__(self, in_channels):
        super(CrissCrossAttention, self).__init__()
        
        self.in_channels = in_channels
        self.channels = in_channels // 8  # C' = C/8
        
        # 1x1 convolutions for Q, K, V
        self.query_conv = nn.Conv2d(in_channels, self.channels, 1)
        self.key_conv = nn.Conv2d(in_channels, self.channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Output with criss-cross attention [B, C, H, W]
        """
        B, C, H, W = x.size()
        
        # Generate Q, K, V
        query = self.query_conv(x)  # [B, C', H, W]
        key = self.key_conv(x)      # [B, C', H, W]
        value = self.value_conv(x)  # [B, C, H, W]
        
        # Reshape for attention computation
        query = query.view(B, self.channels, -1).permute(0, 2, 1)  # [B, H*W, C']
        key = key.view(B, self.channels, -1)  # [B, C', H*W]
        value = value.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        # For each position, collect features from same row and column
        proj_query_H = query.view(B, H, W, self.channels).permute(0, 3, 1, 2)  # [B, C', H, W]
        proj_query_W = query.view(B, H, W, self.channels).permute(0, 3, 2, 1)  # [B, C', W, H]
        
        proj_key_H = key.view(B, self.channels, H, W)  # [B, C', H, W]
        proj_key_W = key.view(B, self.channels, H, W).permute(0, 1, 3, 2)  # [B, C', W, H]
        
        proj_value_H = value.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        proj_value_W = value.view(B, H, W, C).permute(0, 3, 2, 1)  # [B, C, W, H]
        
        # Attention along height (row-wise)
        energy_H = torch.matmul(
            proj_query_H.permute(0, 2, 3, 1).contiguous().view(B * H, W, self.channels),
            proj_key_H.permute(0, 2, 1, 3).contiguous().view(B * H, self.channels, W)
        )  # [B*H, W, W]
        
        attention_H = self.softmax(energy_H)  # [B*H, W, W]
        
        out_H = torch.matmul(
            attention_H,
            proj_value_H.permute(0, 2, 1, 3).contiguous().view(B * H, W, C)
        )  # [B*H, W, C]
        out_H = out_H.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Attention along width (column-wise)
        energy_W = torch.matmul(
            proj_query_W.permute(0, 2, 3, 1).contiguous().view(B * W, H, self.channels),
            proj_key_W.permute(0, 2, 1, 3).contiguous().view(B * W, self.channels, H)
        )  # [B*W, H, H]
        
        attention_W = self.softmax(energy_W)  # [B*W, H, H]
        
        out_W = torch.matmul(
            attention_W,
            proj_value_W.permute(0, 2, 1, 3).contiguous().view(B * W, H, C)
        )  # [B*W, H, C]
        out_W = out_W.view(B, W, H, C).permute(0, 3, 2, 1)  # [B, C, H, W]
        
        # Combine row and column attention
        out = out_H + out_W
        out = self.gamma * out + x
        
        return out


if __name__ == "__main__":
    # Test attention modules
    x = torch.randn(2, 512, 16, 16)
    
    # Test Channel Attention
    ca = ChannelAttention(512, reduction_ratio=16)
    out_ca = ca(x)
    print(f"Channel Attention output: {out_ca.shape}")
    
    # Test Spatial Attention
    sa = SpatialAttention(kernel_size=7)
    out_sa = sa(x)
    print(f"Spatial Attention output: {out_sa.shape}")
    
    # Test Criss-Cross Attention
    cca = CrissCrossAttention(512)
    out_cca = cca(x)
    print(f"Criss-Cross Attention output: {out_cca.shape}")