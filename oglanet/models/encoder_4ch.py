"""
ResNet-34 Encoder with 4-channel input (RGB + Contrast)
Modified for contrast channel support in OGLANet
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34Encoder4Ch(nn.Module):
    """
    ResNet-34 encoder with 4-channel input (RGBC).
    
    Modifications:
    - First conv layer accepts 4 channels instead of 3
    - RGB weights loaded from pretrained ResNet
    - Contrast channel weights initialized as mean of RGB weights
    
    Returns features at 4 scales, same as original encoder.
    """
    
    def __init__(self, pretrained=True):
        super(ResNet34Encoder4Ch, self).__init__()
        
        # Load pretrained ResNet-34
        resnet = models.resnet34(pretrained=pretrained)
        
        # Create new first conv layer with 4 input channels
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
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
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1    # 64 channels
        self.layer2 = resnet.layer2    # 128 channels
        self.layer3 = resnet.layer3    # 256 channels
        self.layer4 = resnet.layer4    # 512 channels
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 4, H, W] (RGBC)
            
        Returns:
            Dictionary of features at different scales
        """
        # Initial layers
        x = self.conv1(x)      # H/2, W/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # H/4, W/4
        
        # ResNet stages
        feat1 = self.layer1(x)     # 64 channels, H/4, W/4
        feat2 = self.layer2(feat1)  # 128 channels, H/8, W/8
        feat3 = self.layer3(feat2)  # 256 channels, H/16, W/16
        feat4 = self.layer4(feat3)  # 512 channels, H/32, W/32
        
        return {
            'feat1': feat1,
            'feat2': feat2,
            'feat3': feat3,
            'feat4': feat4
        }


if __name__ == "__main__":
    # Test 4-channel encoder
    print("Testing 4-channel encoder...")
    
    encoder = ResNet34Encoder4Ch(pretrained=True)
    
    # Test input (RGBC)
    x = torch.randn(2, 4, 384, 384)
    features = encoder(x)
    
    print("\nEncoder output shapes:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")