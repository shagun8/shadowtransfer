"""
ResNet-34 Encoder for MAMNet
Extracts multi-scale features from input images
"""

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 encoder with pretrained ImageNet weights.
    Returns features at 4 scales: 1/4, 1/8, 1/16, 1/32 of input resolution.
    
    For 256x256 input:
        - feat1: 64 x 64 x 64
        - feat2: 128 x 32 x 32
        - feat3: 256 x 16 x 16
        - feat4: 512 x 8 x 8
        - feat5: 512 x 16 x 16 (after upsampling)
    """
    
    def __init__(self, pretrained=True):
        super(ResNet34Encoder, self).__init__()
        
        # Load pretrained ResNet-34
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # Modify conv1 to stride=1 instead of stride=2 to maintain 256x256
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        
        # Extract layers
        self.conv1 = resnet.conv1      # 7x7, stride=2, channels: 3->64
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = nn.Identity()
        
        self.layer1 = resnet.layer1    # Output: 64 channels
        self.layer2 = resnet.layer2    # Output: 128 channels
        self.layer3 = resnet.layer3    # Output: 256 channels
        self.layer4 = resnet.layer4    # Output: 512 channels

        self.downsample = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of features at different scales
        """
        # Initial convolution
        x = self.conv1(x)       # [B, 64, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        
        # Multi-scale features
        feat1 = self.layer1(x)  # [B, 64, H, W]
        feat2 = self.layer2(feat1)  # [B, 128, H/2, W/2]
        feat3 = self.layer3(feat2)  # [B, 256, H/4, W/4]
        feat4 = self.layer4(feat3)  # [B, 512, H/8, W/8]
        feat5 = self.downsample(feat4)  # [B, 512, H/16, W/16]
        
        return {
            'feat1': feat1,  # 64 x 256 x 256
            'feat2': feat2,  # 128 x 128 x 128
            'feat3': feat3,  # 256 x 64 x 64
            'feat4': feat4,  # 512 x 32 x 32
            'feat5': feat5,  # 512 x 16 x 16 (input to MSCAF)
        }


if __name__ == "__main__":
    # Test encoder
    encoder = ResNet34Encoder(pretrained=True)
    x = torch.randn(2, 3, 256, 256)
    features = encoder(x)
    
    print("Encoder output shapes:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")