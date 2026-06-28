"""
OGLANet with SegDesic Module for Geographic Domain Adaptation
Integrates geographic coordinate embeddings into OGLANet architecture.

Based on:
- OGLANet: Xie et al. (2022) - ISPRS Journal
- SegDesicNet: Verma et al. (2025) - WACV
"""

import torch
import torch.nn as nn
from .oglanet import OGLANet
from .segdesic_module import SegDesicModule


class OGLANetSegDesic(nn.Module):
    """
    OGLANet with integrated SegDesic module for geographic domain adaptation.
    
    Architecture:
    - Standard OGLANet encoder-decoder with GLAM, DFFM, and OAM
    - SegDesic module attached to encoder features
    - Geographic domain loss during training
    
    Key Design Decision:
    - SegDesic attaches to feat5 (deepest encoder features, 512 channels)
    - This matches the MAMNet implementation for consistency
    - Allows the model to learn geographic patterns from high-level features
    """
    
    def __init__(self, num_classes=2, pretrained=True, img_size=384,
                 segdesic_hidden_dim=256, segdesic_num_scales=10, use_contrast=False):
        """
        Args:
            num_classes: Number of segmentation classes
            pretrained: Use pretrained encoder
            img_size: Input image size (default 384)
            segdesic_hidden_dim: Hidden dimension for SegDesic module
            segdesic_num_scales: Number of scales for GRID encoding
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Base OGLANet
        self.oglanet = OGLANet(
            num_classes=num_classes,
            pretrained=pretrained,
            img_size=img_size,
            use_contrast=use_contrast
        )
        
        # SegDesic module (attached to deepest encoder features)
        # OGLANet encoder (ResNet-34) feat5 has 512 channels
        # Note: 4-channel encoder also outputs 512 channels at feat5
        self.segdesic = SegDesicModule(
            in_channels=1024,
            hidden_dim=segdesic_hidden_dim,
            num_scales=segdesic_num_scales
        )
    
    def forward(self, x, lat=None, lon=None):
        """
        Forward pass with optional geographic coordinates.
        
        Args:
            x: Input images [B, 3, H, W]
            lat: Latitude values [B] (optional, for training)
            lon: Longitude values [B] (optional, for training)
        
        Returns:
            Dictionary containing:
                - Segmentation outputs (training: 'p1'-'p6', inference: 'p6' only)
                - Geographic outputs ('geo') if lat/lon provided
        """
        B, _, H, W = x.size()
        
        # Encoder features
        encoder_features = self.oglanet.encoder(x)
        
        # Geographic prediction from deepest features
        geo_outputs = None
        if lat is not None and lon is not None:
            geo_outputs = self.segdesic(encoder_features['feat5'], lat, lon)
        
        # DFFM
        dffm_features = self.oglanet.dffm(encoder_features)
        
        # Decoder
        decoder_features = self.oglanet.decoder(dffm_features)
        
        # OAM (Omni-scale Aggregation Module)
        predictions = self.oglanet.oam(decoder_features)
        
        # Prepare output
        if self.training:
            # Return all 6 predictions for deep supervision
            output = predictions
        else:
            # Return only P6 for inference
            output = predictions['p6']
        
        # Add geographic outputs if computed
        if geo_outputs is not None:
            output['geo'] = geo_outputs
        
        return output
    
    def get_predictions(self, x):
        """
        Get binary predictions for inference.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Binary predictions [B, H, W]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            logits = outputs['p6']
            preds = torch.argmax(logits, dim=1)
        return preds


if __name__ == "__main__":
    # Test OGLANetSegDesic
    print("="*50)
    print("Testing OGLANet with SegDesic")
    print("="*50)
    
    model = OGLANetSegDesic(
        num_classes=2,
        pretrained=False,
        img_size=384,
        segdesic_hidden_dim=256,
        segdesic_num_scales=10
    )
    
    # Training mode with geographic coordinates
    model.train()
    x = torch.randn(2, 3, 384, 384)
    lat = torch.tensor([33.4484, 25.7617])
    lon = torch.tensor([-112.0740, -80.1918])
    
    outputs = model(x, lat, lon)
    
    print("\nTraining outputs:")
    for key, val in outputs.items():
        if key == 'geo':
            print(f"  {key}:")
            for geo_key, geo_val in val.items():
                print(f"    {geo_key}: {geo_val.shape}")
        else:
            print(f"  {key}: {val.shape}")
    
    # Inference mode
    model.eval()
    outputs_eval = model(x)
    print(f"\nInference output: {outputs_eval['p6'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compare with base OGLANet
    base_model = OGLANet(num_classes=2, pretrained=False, img_size=384)
    base_params = sum(p.numel() for p in base_model.parameters())
    additional_params = total_params - base_params
    print(f"Additional parameters from SegDesic: {additional_params:,} ({100*additional_params/base_params:.2f}% increase)")