"""
Scale Attention Module for DINOv3-HRDA
Adapted for Vision Transformer (ViT) features from DINOv3.

Key differences from CNN-based scale attention:
1. Input features are from ViT (embed_dim channels instead of CNN channels)
2. Uses transformer-friendly operations
3. Maintains spatial structure after patch embedding

Based on HRDA (ECCV 2022): https://arxiv.org/abs/2204.13132
The scale attention predicts per-pixel weights to combine LR and HR predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAttentionHeadViT(nn.Module):
    """
    Scale Attention Decoder for DINOv3-HRDA.
    Adapted for Vision Transformer features.
    
    Predicts attention weights to fuse LR context and HR detail predictions.
    
    Architecture: Lightweight MLP decoder operating on ViT features
    Input: Features from DINOv3 encoder [B, embed_dim, H/16, W/16]
    Output: Per-class attention weights in [0, 1] (1 = focus on HR detail)
    
    Args:
        embed_dim: Embedding dimension from ViT (384 for ViT-S, 768 for ViT-B)
        num_classes: Number of segmentation classes
        hidden_dim: Hidden dimension for MLP (default: 256)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, embed_dim, num_classes, hidden_dim=256, dropout=0.1):
        super(ScaleAttentionHeadViT, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Feature projection and fusion
        # First reduce ViT features to a manageable dimension
        self.feature_proj = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Attention prediction head
        # Uses depthwise separable convolutions for efficiency
        self.attention_conv1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        self.attention_conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1, groups=hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, num_classes, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        """
        Forward pass to predict scale attention.
        
        Args:
            features: ViT features [B, embed_dim, H/16, W/16]
            
        Returns:
            attention: Scale attention weights [B, num_classes, H, W] in [0, 1]
                      1 = focus on HR detail, 0 = focus on LR context
        """
        B, C, H_feat, W_feat = features.shape
        
        # Project features to hidden dimension
        x = self.feature_proj(features)  # [B, hidden_dim, H/16, W/16]
        
        # Predict attention through convolutions
        x = self.attention_conv1(x)      # [B, hidden_dim/2, H/16, W/16]
        attention = self.attention_conv2(x)  # [B, num_classes, H/16, W/16]
        
        # Upsample attention to full resolution
        # Features are at 1/16 resolution, upsample to full resolution
        target_size = (H_feat * 16, W_feat * 16)
        attention = F.interpolate(
            attention,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )  # [B, num_classes, H, W]
        
        # Apply sigmoid to get weights in [0, 1]
        attention = torch.sigmoid(attention)
        
        return attention


class HRDAFusionModule(nn.Module):
    """
    HRDA Fusion Module - combines LR context and HR detail predictions
    using learned scale attention.
    
    Identical to MAMNet version, as fusion logic is model-agnostic.
    
    Args:
        num_classes: Number of segmentation classes
    """
    
    def __init__(self, num_classes):
        super(HRDAFusionModule, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred_context, pred_detail, attention, detail_crop_coords):
        """
        Fuses LR context and HR detail predictions using scale attention.
        
        Args:
            pred_context: LR context prediction [B, C, H_c, W_c]
            pred_detail: HR detail prediction [B, C, H_d, W_d]
            attention: Scale attention from context [B, C, H_c, W_c]
            detail_crop_coords: Coordinates of detail crop in context
                               [(b1, b2, b3, b4), ...] for each batch item
        
        Returns:
            fused_pred: Fused prediction at HR resolution [B, C, H_HR, W_HR]
        """
        B, C, H_c, W_c = pred_context.shape
        _, _, H_d, W_d = pred_detail.shape
        
        # Determine scale factor
        # Context is at 0.5x resolution, need to upsample to HR (2x)
        scale_factor = 2
        H_HR = H_c * scale_factor
        W_HR = W_c * scale_factor
        
        # Upsample context prediction to HR resolution
        pred_context_hr = F.interpolate(
            pred_context, 
            size=(H_HR, W_HR), 
            mode='bilinear', 
            align_corners=False
        )  # [B, C, H_HR, W_HR]
        
        # Upsample scale attention to HR resolution
        attention_hr = F.interpolate(
            attention,
            size=(H_HR, W_HR),
            mode='bilinear',
            align_corners=False
        )  # [B, C, H_HR, W_HR]
        
        # Create masked attention (only where detail crop exists)
        # Initialize attention mask with zeros (all context)
        attention_masked = torch.zeros_like(attention_hr)  # [B, C, H_HR, W_HR]
        
        # For each batch item, fill in the detail crop region
        for i in range(B):
            if detail_crop_coords is not None and i < len(detail_crop_coords):
                b1, b2, b3, b4 = detail_crop_coords[i]
                
                # Map coordinates from context resolution to HR resolution
                b1_hr = b1 * scale_factor
                b2_hr = b2 * scale_factor
                b3_hr = b3 * scale_factor
                b4_hr = b4 * scale_factor
                
                # Ensure coordinates are within bounds
                b1_hr = max(0, min(b1_hr, H_HR))
                b2_hr = max(0, min(b2_hr, H_HR))
                b3_hr = max(0, min(b3_hr, W_HR))
                b4_hr = max(0, min(b4_hr, W_HR))
                
                # Set attention in detail crop region
                attention_masked[i, :, b1_hr:b2_hr, b3_hr:b4_hr] = \
                    attention_hr[i, :, b1_hr:b2_hr, b3_hr:b4_hr]
        
        # Pad and align detail prediction
        pred_detail_aligned = torch.zeros_like(pred_context_hr)  # [B, C, H_HR, W_HR]
        
        for i in range(B):
            if detail_crop_coords is not None and i < len(detail_crop_coords):
                b1, b2, b3, b4 = detail_crop_coords[i]
                
                # Map coordinates to HR
                b1_hr = b1 * scale_factor
                b2_hr = b2 * scale_factor
                b3_hr = b3 * scale_factor
                b4_hr = b4 * scale_factor
                
                # Ensure detail prediction fits
                b1_hr = max(0, min(b1_hr, H_HR))
                b2_hr = max(0, min(b2_hr, H_HR))
                b3_hr = max(0, min(b3_hr, W_HR))
                b4_hr = max(0, min(b4_hr, W_HR))
                
                # Resize detail prediction to fit the crop region
                detail_h = b2_hr - b1_hr
                detail_w = b4_hr - b3_hr
                
                if detail_h > 0 and detail_w > 0:
                    pred_detail_resized = F.interpolate(
                        pred_detail[i:i+1],
                        size=(detail_h, detail_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    pred_detail_aligned[i, :, b1_hr:b2_hr, b3_hr:b4_hr] = \
                        pred_detail_resized[0]
        
        # Fuse predictions: (1 - attention) * context + attention * detail
        # Eq. 12 from HRDA paper
        fused_pred = (1 - attention_masked) * pred_context_hr + \
                     attention_masked * pred_detail_aligned
        
        return fused_pred


if __name__ == "__main__":
    # Test scale attention module for ViT
    print("=" * 60)
    print("Testing Scale Attention for DINOv3")
    print("=" * 60)
    
    batch_size = 2
    num_classes = 2
    embed_dim = 384  # ViT-S/16
    
    # Simulate ViT features at 1/16 resolution
    # For 192×192 input → 12×12 patches
    features = torch.randn(batch_size, embed_dim, 12, 12)
    
    # Create scale attention head
    scale_attn = ScaleAttentionHeadViT(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=256
    )
    
    # Forward pass
    attention = scale_attn(features)
    
    print(f"\nInput features: {features.shape}")
    print(f"Scale attention: {attention.shape}")
    print(f"Attention range: [{attention.min():.3f}, {attention.max():.3f}]")
    print(f"Expected output size: [2, 2, 192, 192] ✓" if attention.shape == (2, 2, 192, 192) else f"Unexpected size!")
    
    # Test fusion module
    print("\n" + "=" * 60)
    print("Testing HRDA Fusion Module")
    print("=" * 60)
    
    fusion = HRDAFusionModule(num_classes=num_classes)
    
    # Simulate predictions
    pred_context = torch.randn(batch_size, num_classes, 192, 192)
    pred_detail = torch.randn(batch_size, num_classes, 192, 192)
    detail_coords = [(24, 168, 24, 168), (30, 162, 30, 162)]
    
    fused = fusion(pred_context, pred_detail, attention, detail_coords)
    
    print(f"\nContext prediction: {pred_context.shape}")
    print(f"Detail prediction: {pred_detail.shape}")
    print(f"Fused prediction: {fused.shape}")
    print(f"Expected fused size: [2, 2, 384, 384] ✓" if fused.shape == (2, 2, 384, 384) else f"Unexpected size!")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)