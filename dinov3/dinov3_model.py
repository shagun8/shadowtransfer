"""
DINOv3 Model for Shadow Detection
Complete model integrating DINOv3 backbone (ViT-S/16) and lightweight decoder.

Key advantage: Patch size 16 means 384÷16 = 24 patches exactly!
No padding/cropping needed - much cleaner than DINOv2's patch size 14.
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


class DINOv3ShadowDetector(nn.Module):
    """
    DINOv3-based Shadow Detection Network
    
    Architecture:
    1. DINOv3-S (ViT-S/16) pretrained backbone
    2. Lightweight progressive upsampling decoder
    3. Single output (no auxiliary branches)
    
    Key features:
    - Input: 384×384 images (24×24 patches, perfect fit!)
    - Output: 384×384 segmentation masks
    - ~22M parameters (comparable to ResNet-34)
    """
    
    def __init__(self, num_classes=2, model_name='dinov3_vits16', weights_path=None, 
                 pretrained=True, frozen_stages=-1):
        """
        Args:
            num_classes: Number of output classes (default: 2 for binary shadow detection)
            model_name: DINOv3 variant
                - 'dinov3_vits16': ViT-S/16 (~22M params, recommended)
                - 'dinov3_vitb16': ViT-B/16 (~86M params)
                - 'dinov3_vitl16': ViT-L/16 (~304M params)
            weights_path: Path to pretrained weights .pth file
            pretrained: Load pretrained DINOv3 weights
            frozen_stages: Number of backbone stages to freeze (-1 = train all)
        """
        super(DINOv3ShadowDetector, self).__init__()
        
        self.num_classes = num_classes
        self.model_name = model_name
        
        # Initialize backbone
        print('Initializing DINOv3 backbone...')
        self.backbone = DINOv3Backbone(
            model_name=model_name,
            weights_path=weights_path,
            pretrained=pretrained,
            frozen_stages=frozen_stages
        )
        
        # Get embedding dimension
        embed_dim = self.backbone.embed_dim
        
        # Initialize decoder
        print('Initializing decoder...')
        self.decoder = DINOv3Decoder(
            num_classes=num_classes,
            embed_dim=embed_dim
        )
        
        print(f'\nDINOv3 Shadow Detector initialized:')
        print(f'  Model: {model_name}')
        print(f'  Input size: 384×384 (24×24 patches)')
        print(f'  Output classes: {num_classes}')
        
        # Print parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Total parameters: {total_params:,}')
        print(f'  Trainable parameters: {trainable_params:,}')
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input RGB images [B, 3, 384, 384]
        
        Returns:
            Segmentation logits [B, num_classes, 384, 384]
        """
        B, C, H, W = x.shape
        
        # Verify input size is compatible with patch size 16
        if H % 16 != 0 or W % 16 != 0:
            raise ValueError(
                f"Input size ({H}×{W}) must be divisible by patch size (16). "
                f"Expected 384×384 or other multiples of 16."
            )
        
        # Extract features from backbone
        features = self.backbone(x)  # Features at 1/16 resolution
        
        # Decode to segmentation mask
        output = self.decoder(features)  # [B, num_classes, H, W]
        
        # Ensure output matches input size
        if output.shape[2] != H or output.shape[3] != W:
            output = F.interpolate(
                output,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        
        return output
    
    def get_predictions(self, x):
        """
        Get binary predictions (for inference/evaluation).
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Binary predictions [B, H, W] with values {0, 1}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # [B, 2, H, W]
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
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
    # Test complete model
    print("=" * 60)
    print("Testing DINOv3 Shadow Detector")
    print("=" * 60)
    
    try:
        # Initialize model
        model = DINOv3ShadowDetector(
            num_classes=2,
            model_name='dinov3_vits16',
            weights_path=None,  # Will try torch.hub first
            pretrained=True,
            frozen_stages=-1
        )
        
        print("\n" + "=" * 60)
        print("Testing forward pass...")
        print("=" * 60)
        
        # Test with 384x384 input (perfect 24×24 patches)
        x = torch.randn(2, 3, 384, 384)
        print(f"\nInput shape: {x.shape}")
        
        # Forward pass
        model.train()
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"✓ Output matches input size!")
        
        # Test inference mode
        model.eval()
        preds = model.get_predictions(x)
        print(f"\nBinary predictions shape: {preds.shape}")
        print(f"Unique prediction values: {torch.unique(preds)}")
        
        print("\n" + "=" * 60)
        print("Testing with different input sizes...")
        print("=" * 60)
        
        # Test with other multiples of 16
        for size in [256, 384, 512]:
            x_test = torch.randn(1, 3, size, size)
            output_test = model(x_test)
            print(f"  {size}×{size} input → {output_test.shape} output ✓")
        
        print("\n" + "=" * 60)
        print("Model summary:")
        print("=" * 60)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        decoder_params = sum(p.numel() for p in model.decoder.parameters())
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Backbone parameters: {backbone_params:,}")
        print(f"Decoder parameters: {decoder_params:,}")
        print(f"Backbone ratio: {backbone_params/total_params*100:.1f}%")
        print(f"Decoder ratio: {decoder_params/total_params*100:.1f}%")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        print("\nNote: If model loading failed, provide weights_path:")
        print("  model = DINOv3ShadowDetector(")
        print("      model_name='dinov3_vits16',")
        print("      weights_path='path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'")
        print("  )")