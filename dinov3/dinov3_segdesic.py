"""
DINOv3 Model with SegDesic Module for Geographic Domain Adaptation
Integrates geographic coordinate embeddings into DINOv3 architecture.

Key Features:
- DINOv3 ViT-S/16 backbone (perfect 384÷16=24 patches)
- SegDesic module attached to deepest features (block 11)
- Geographic domain loss for cross-city transfer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dinov3_backbone import DINOv3Backbone
from dinov3_decoder import DINOv3Decoder

# Import SegDesic module from MAMNet (reusable)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mamnet'))
from models.segdesic_module import SegDesicModule


class DINOv3SegDesic(nn.Module):
    """
    DINOv3 with integrated SegDesic module for geographic domain adaptation.
    
    Architecture:
    - DINOv3 ViT-S/16 backbone (pretrained)
    - Lightweight decoder for segmentation
    - SegDesic module attached to deepest features (block 11)
    - Geographic domain loss during training
    
    Input: [B, 3, 384, 384] images + optional (lat, lon) coordinates
    Output: Segmentation logits + geographic encodings (if lat/lon provided)
    """
    
    def __init__(self, num_classes=2, model_name='dinov3_vits16', weights_path=None,
                 pretrained=True, frozen_stages=-1,
                 segdesic_hidden_dim=256, segdesic_num_scales=10):
        """
        Args:
            num_classes: Number of segmentation classes
            model_name: DINOv3 variant ('dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16')
            weights_path: Path to pretrained DINOv3 weights
            pretrained: Use pretrained DINOv3 weights
            frozen_stages: Number of backbone stages to freeze (-1 = train all)
            segdesic_hidden_dim: Hidden dimension for SegDesic MLP
            segdesic_num_scales: Number of scales for GRID encoding
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize DINOv3 backbone
        print('Initializing DINOv3 backbone...')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages
        )
        
        # Get embedding dimension (384 for ViT-S, 768 for ViT-B, 1024 for ViT-L)
        self.embed_dim = self.backbone.embed_dim
        
        # Initialize decoder
        print('Initializing decoder...')
        self.decoder = DINOv3Decoder(
            num_classes=num_classes,
            embed_dim=self.embed_dim
        )
        
        # Initialize SegDesic module
        # Attach to deepest features (block 11) with embed_dim channels
        print('Initializing SegDesic module...')
        self.segdesic = SegDesicModule(
            in_channels=self.embed_dim,  # 384 for ViT-S, 768 for ViT-B, etc.
            hidden_dim=segdesic_hidden_dim,
            num_scales=segdesic_num_scales
        )
        
        print(f'\nDINOv3 SegDesic initialized:')
        print(f'  Model: {model_name}')
        print(f'  Embedding dim: {self.embed_dim}')
        print(f'  SegDesic attached to feat_block11')
        print(f'  SegDesic hidden dim: {segdesic_hidden_dim}')
        print(f'  SegDesic num scales: {segdesic_num_scales}')
        print(f'  GRID encoding dim: {4 * segdesic_num_scales}')
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        segdesic_params = sum(p.numel() for p in self.segdesic.parameters())
        
        print(f'  Total parameters: {total_params:,}')
        print(f'  Trainable parameters: {trainable_params:,}')
        print(f'  SegDesic parameters: {segdesic_params:,} ({100*segdesic_params/total_params:.2f}% of total)')
    
    def forward(self, x, lat=None, lon=None):
        """
        Forward pass with optional geographic coordinates.
        
        Args:
            x: Input images [B, 3, H, W] (typically 384×384)
            lat: Latitude values [B] (optional, for training with domain adaptation)
            lon: Longitude values [B] (optional, for training with domain adaptation)
        
        Returns:
            Dictionary containing:
                - 'main': Segmentation logits [B, num_classes, H, W]
                - 'geo': Geographic outputs (if lat/lon provided)
                    - 'pred_encoding': Predicted coordinate encoding [B, encoding_dim]
                    - 'gt_encoding': Ground truth coordinate encoding [B, encoding_dim]
        """
        B, C, H, W = x.shape
        
        # Verify input size is compatible with patch size 16
        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                f"Input size ({H}×{W}) must be divisible by patch size (16). "
                f"Expected 384×384 or other multiples of 16."
            )
        
        # Extract features from backbone
        features = self.backbone(x)  # All features at 1/16 resolution
        # features = {
        #     'feat_block3': [B, 384, H/16, W/16],
        #     'feat_block6': [B, 384, H/16, W/16],
        #     'feat_block9': [B, 384, H/16, W/16],
        #     'feat_block11': [B, 384, H/16, W/16]
        # }
        
        # Geographic prediction from deepest features (block 11)
        geo_outputs = None
        if lat is not None and lon is not None:
            geo_outputs = self.segdesic(features['feat_block11'], lat, lon)
        
        # Decode to segmentation mask
        seg_output = self.decoder(features)  # [B, num_classes, H, W]
        
        # Ensure output matches input size
        if seg_output.shape[2] != H or seg_output.shape[3] != W:
            seg_output = F.interpolate(
                seg_output,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        # Prepare output dictionary
        output = {'main': seg_output}
        
        # Add geographic outputs if computed
        if geo_outputs is not None:
            output['geo'] = geo_outputs
        
        # Return tensor directly during inference, dict during training
        if self.training:
            return output
        else:
            return seg_output  # Return tensor directly for visualization
    
    def get_predictions(self, x):
        """
        Get binary predictions for inference.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Binary predictions [B, H, W] with values {0, 1}
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            logits = outputs['main']
            preds = torch.argmax(logits, dim=1)
        return preds
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for fine-tuning"""
        print('Unfreezing backbone...')
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        print('Freezing backbone...')
        for param in self.backbone.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    # Test DINOv3 SegDesic
    print("=" * 70)
    print("Testing DINOv3 SegDesic")
    print("=" * 70)
    
    try:
        # Initialize model
        model = DINOv3SegDesic(
            num_classes=2,
            model_name='dinov3_vits16',
            weights_path=None,  # Will need actual weights path
            pretrained=False,  # Set to False for testing without weights
            frozen_stages=-1,
            segdesic_hidden_dim=256,
            segdesic_num_scales=10
        )
        
        print("\n" + "=" * 70)
        print("Testing forward pass WITH geographic coordinates (training mode)...")
        print("=" * 70)
        
        # Training mode with geographic coordinates
        model.train()
        x = torch.randn(2, 3, 384, 384)
        lat = torch.tensor([33.4484, 25.7617])  # Phoenix, Miami
        lon = torch.tensor([-112.0740, -80.1918])
        
        print(f"\nInput shape: {x.shape}")
        print(f"Latitude: {lat}")
        print(f"Longitude: {lon}")
        
        outputs = model(x, lat, lon)
        
        print("\nOutputs:")
        print(f"  Segmentation (main): {outputs['main'].shape}")
        if 'geo' in outputs:
            print(f"  Geographic outputs:")
            for key, val in outputs['geo'].items():
                print(f"    {key}: {val.shape}")
        
        print("\n" + "=" * 70)
        print("Testing forward pass WITHOUT geographic coordinates (inference mode)...")
        print("=" * 70)
        
        # Inference mode without geographic coordinates
        model.eval()
        outputs_eval = model(x)
        print(f"\nSegmentation output: {outputs_eval['main'].shape}")
        print(f"Has geographic outputs: {'geo' in outputs_eval}")
        
        # Test predictions
        preds = model.get_predictions(x)
        print(f"\nBinary predictions: {preds.shape}, unique values: {torch.unique(preds)}")
        
        print("\n" + "=" * 70)
        print("Model summary:")
        print("=" * 70)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        segdesic_params = sum(p.numel() for p in model.segdesic.parameters())
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Backbone parameters: {backbone_params:,} ({backbone_params/total_params*100:.1f}%)")
        print(f"Decoder parameters: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
        print(f"SegDesic parameters: {segdesic_params:,} ({segdesic_params/total_params*100:.1f}%)")
        
        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()