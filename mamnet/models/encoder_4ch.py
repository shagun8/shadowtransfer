"""
ResNet-34 Encoder with 4-channel input (RGB + Contrast)
Modified for Task 2: Contrast as 4th input channel
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Encoder4Ch(nn.Module):
    """
    ResNet-34 encoder with 4-channel input (RGBC).
    
    Modifications:
    - First conv layer accepts 4 channels instead of 3
    - RGB weights loaded from pretrained ResNet
    - Contrast channel weights initialized randomly
    
    Returns features at 4 scales, same as original encoder.
    """
    
    def __init__(self, pretrained=True):
        super(ResNet34Encoder4Ch, self).__init__()
        
        # Load pretrained ResNet-34
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Create new first conv layer with 4 input channels
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        # Initialize with pretrained weights for RGB channels
        if pretrained:
            # Copy RGB weights from pretrained model
            pretrained_weights = resnet.conv1.weight.data  # [64, 3, 7, 7]
            
            # Initialize new conv1
            with torch.no_grad():
                # Copy RGB weights
                self.conv1.weight[:, :3, :, :] = pretrained_weights
                
                # Initialize contrast channel weights
                # Use average of RGB weights as initialization
                self.conv1.weight[:, 3:4, :, :] = pretrained_weights.mean(dim=1, keepdim=True)
            
            print("4-channel encoder initialized:")
            print("  RGB channels: Loaded from pretrained ResNet-34")
            print("  Contrast channel: Initialized with mean of RGB weights")
        
        # Extract other layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        
        self.layer1 = resnet.layer1    # Output: 64 channels
        self.layer2 = resnet.layer2    # Output: 128 channels
        self.layer3 = resnet.layer3    # Output: 256 channels
        self.layer4 = resnet.layer4    # Output: 512 channels
        
        # Downsample layer
        self.downsample = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 4, H, W] (RGBC)
            
        Returns:
            Dictionary of features at different scales
        """
        # Initial convolution
        x = self.conv1(x)       # [B, 64, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        
        # Multi-scale features
        feat1 = self.layer1(x)              # [B, 64, H, W]
        feat2 = self.layer2(feat1)          # [B, 128, H/2, W/2]
        feat3 = self.layer3(feat2)          # [B, 256, H/4, W/4]
        feat4 = self.layer4(feat3)          # [B, 512, H/8, W/8]
        feat5 = self.downsample(feat4)      # [B, 512, H/16, W/16]
        
        return {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4,
            'feat5': feat5,
        }


if __name__ == "__main__":
    # Test 4-channel encoder
    print("Testing 4-channel encoder...")
    
    encoder = ResNet34Encoder4Ch(pretrained=True)
    
    # Test input (RGBC)
    x = torch.randn(2, 4, 256, 256)
    features = encoder(x)
    
    print("\nEncoder output shapes:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Compare with 3-channel encoder
    print("\nComparing with 3-channel encoder:")
    from encoder import ResNet34Encoder
    encoder_3ch = ResNet34Encoder(pretrained=True)
    params_3ch = sum(p.numel() for p in encoder_3ch.parameters())
    
    print(f"3-channel encoder: {params_3ch:,} parameters")
    print(f"4-channel encoder: {total_params:,} parameters")
    print(f"Difference: {total_params - params_3ch:,} parameters")