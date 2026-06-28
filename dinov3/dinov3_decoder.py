"""
Decoder for DINOv3-based Shadow Detection
Progressive upsampling decoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    Decoder block with skip connection: Upsample -> Concat -> ConvBlock -> ConvBlock
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # Reduce skip connection channels if needed
        self.skip_conv = nn.Conv2d(skip_channels, skip_channels // 2, 1) if skip_channels > 0 else None
        
        # Two conv blocks after concatenation
        concat_channels = in_channels + (skip_channels // 2 if skip_channels > 0 else 0)
        self.conv1 = ConvBlock(concat_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip=None):
        """
        Args:
            x: Input features [B, C, H, W]
            skip: Skip connection features [B, C_skip, 2*H, 2*W] or None
        
        Returns:
            Output features [B, out_channels, 2*H, 2*W]
        """
        # Upsample input
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        if skip is not None and self.skip_conv is not None:
            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)
        
        # Apply conv blocks
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class DINOv3Decoder(nn.Module):
    """
    Lightweight decoder for DINOv3 features.
    
    Architecture:
    - All DINOv3 features are at the same resolution (H/16, W/16)
    - Progressive upsampling without multi-scale skip connections
    - Use concatenated features from multiple blocks
    
    Input: Multi-scale DINOv3 features at 1/16 resolution
    Output: Segmentation mask at original resolution
    """
    
    def __init__(self, num_classes=2, embed_dim=384):
        """
        Args:
            num_classes: Number of output classes (2 for binary shadow detection)
            embed_dim: DINOv3 embedding dimension (384 for ViT-S, 768 for ViT-B)
        """
        super(DINOv3Decoder, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Fuse features from multiple blocks (all at same resolution)
        # We'll concatenate features from blocks 3, 6, 9, 11
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder channels (progressively reduce)
        dec_channels = [embed_dim, 256, 128, 64, 32]
        
        # Upsampling stages: 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dec_channels[0], dec_channels[1], 2, stride=2),
            nn.BatchNorm2d(dec_channels[1]),
            nn.ReLU(inplace=True),
            ConvBlock(dec_channels[1], dec_channels[1])
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(dec_channels[1], dec_channels[2], 2, stride=2),
            nn.BatchNorm2d(dec_channels[2]),
            nn.ReLU(inplace=True),
            ConvBlock(dec_channels[2], dec_channels[2])
        )
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(dec_channels[2], dec_channels[3], 2, stride=2),
            nn.BatchNorm2d(dec_channels[3]),
            nn.ReLU(inplace=True),
            ConvBlock(dec_channels[3], dec_channels[3])
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(dec_channels[3], dec_channels[4], 2, stride=2),
            nn.BatchNorm2d(dec_channels[4]),
            nn.ReLU(inplace=True),
            ConvBlock(dec_channels[4], dec_channels[4])
        )
        
        # Final classification head
        self.final_conv = nn.Conv2d(dec_channels[4], num_classes, kernel_size=1)
        
        print(f'DINOv3 Decoder initialized:')
        print(f'  Embed dim: {embed_dim}')
        print(f'  Decoder channels: {dec_channels}')
        print(f'  Num classes: {num_classes}')
        print(f'  Note: All ViT features at same resolution (H/16, W/16)')
    
    def forward(self, features):
        """
        Args:
            features: Dictionary of DINOv3 features (all at H/16, W/16 resolution)
                - 'feat_block3': [B, 384, H/16, W/16]
                - 'feat_block6': [B, 384, H/16, W/16]
                - 'feat_block9': [B, 384, H/16, W/16]
                - 'feat_block11': [B, 384, H/16, W/16]
        
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        # Concatenate all features (they're all at the same spatial resolution)
        feat_concat = torch.cat([
            features['feat_block3'],
            features['feat_block6'],
            features['feat_block9'],
            features['feat_block11']
        ], dim=1)  # [B, 384*4, H/16, W/16]
        
        # Fuse concatenated features
        x = self.feature_fusion(feat_concat)  # [B, 384, H/16, W/16]
        
        # Progressive upsampling: 1/16 -> 1/1
        x = self.up1(x)  # [B, 256, H/8, W/8]
        x = self.up2(x)  # [B, 128, H/4, W/4]
        x = self.up3(x)  # [B, 64, H/2, W/2]
        x = self.up4(x)  # [B, 32, H, W]
        
        # Final classification
        x = self.final_conv(x)  # [B, num_classes, H, W]
        
        return x


if __name__ == "__main__":
    # Test decoder
    print("Testing DINOv3 Decoder...")
    
    decoder = DINOv3Decoder(num_classes=2, embed_dim=384)
    
    # Simulate DINOv3 features (from 392x392 input -> 28x28 patches)
    features = {
        'feat_block3': torch.randn(2, 384, 28, 28),
        'feat_block6': torch.randn(2, 384, 28, 28),
        'feat_block9': torch.randn(2, 384, 28, 28),
        'feat_block11': torch.randn(2, 384, 28, 28),
    }
    
    output = decoder(features)
    print(f"\nDecoder output: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {total_params:,}")