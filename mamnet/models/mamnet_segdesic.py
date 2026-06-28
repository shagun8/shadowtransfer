"""
MAMNet with SegDesic Module for Geographic Domain Adaptation
Integrates geographic coordinate embeddings into MAMNet architecture.
"""

import torch
import torch.nn as nn
from .mamnet import MAMNet
from .segdesic_module import SegDesicModule


class MAMNetSegDesic(nn.Module):
    """
    MAMNet with integrated SegDesic module for geographic domain adaptation.
    
    Architecture:
    - Standard MAMNet encoder-decoder
    - SegDesic module attached to encoder features
    - Geographic domain loss during training
    """
    
    def __init__(self, num_classes=2, pretrained=True, use_aux=True, 
                 segdesic_hidden_dim=256, segdesic_num_scales=10, use_contrast=False):
        """
        Args:
            num_classes: Number of segmentation classes
            pretrained: Use pretrained encoder
            use_aux: Use auxiliary branches
            segdesic_hidden_dim: Hidden dimension for SegDesic module
            segdesic_num_scales: Number of scales for GRID encoding
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_aux = use_aux
        
        # Base MAMNet
        self.mamnet = MAMNet(
            num_classes=num_classes,
            pretrained=pretrained,
            use_aux=use_aux,
            use_contrast=use_contrast
        )
        
        # SegDesic module (attached to deepest encoder features)
        # ResNet-34 feat5 has 512 channels (same for 3ch and 4ch)
        self.segdesic = SegDesicModule(
            in_channels=512,
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
                - Segmentation outputs ('main', optionally 'aux1', 'aux2', 'aux3')
                - Geographic outputs ('geo') if lat/lon provided
        """
        B, _, H, W = x.size()
        
        # Encoder features
        enc_features = self.mamnet.encoder(x)
        
        # Geographic prediction from deepest features
        geo_outputs = None
        if lat is not None and lon is not None:
            geo_outputs = self.segdesic(enc_features['feat5'], lat, lon)
        
        # MSCAF
        mscaf_out = self.mamnet.mscaf(enc_features['feat5'])
        
        # Decoder
        decoder_outputs = self.mamnet.decoder(mscaf_out, enc_features)
        main_out = decoder_outputs['main']
        
        # Prepare output
        output = {'main': main_out}
        
        # Auxiliary branches (training only)
        if self.use_aux and self.training:
            aux_outputs = self.mamnet.aux_module(
                decoder_outputs['dec_feat1'],
                decoder_outputs['dec_feat2'],
                decoder_outputs['dec_feat3'],
                target_size=(H, W)
            )
            output.update(aux_outputs)
        
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
            logits = outputs['main']
            preds = torch.argmax(logits, dim=1)
        return preds


if __name__ == "__main__":
    # Test MAMNetSegDesic
    model = MAMNetSegDesic(
        num_classes=2,
        pretrained=False,
        use_aux=True,
        segdesic_hidden_dim=256,
        segdesic_num_scales=10
    )
    
    # Training mode with geographic coordinates
    model.train()
    x = torch.randn(2, 3, 256, 256)
    lat = torch.tensor([33.4484, 25.7617])
    lon = torch.tensor([-112.0740, -80.1918])
    
    outputs = model(x, lat, lon)
    
    print("Training outputs:")
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
    print(f"\nInference output: {outputs_eval['main'].shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compare with base MAMNet
    base_model = MAMNet(num_classes=2, pretrained=False, use_aux=True)
    base_params = sum(p.numel() for p in base_model.parameters())
    additional_params = total_params - base_params
    print(f"Additional parameters from SegDesic: {additional_params:,} ({100*additional_params/base_params:.2f}% increase)")