"""
ResNet-34 Encoder for feature extraction.
Used as the Local Feature Extraction Module (LFEM) in GLAM.

ASSUMPTION: Using ResNet-34 instead of ResNet-101 (paper specification)
to ensure fair comparison with MAMNet implementation and avoid
performance gains solely from increased parameter count.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet34Encoder(nn.Module):
    """
    ResNet-34 encoder for extracting multi-scale features.
    
    Returns features at 5 different scales corresponding to ResNet stages.
    """
    
    def __init__(self, pretrained=True):
        super(ResNet34Encoder, self).__init__()
        
        # Load pretrained ResNet-34
        resnet = models.resnet34(pretrained=pretrained)
        
        # Initial conv + bn + relu + maxpool
        self.conv1 = resnet.conv1      # 7×7, stride=2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 3×3, stride=2
        
        # ResNet stages
        self.layer1 = resnet.layer1    # 64 channels
        self.layer2 = resnet.layer2    # 128 channels
        self.layer3 = resnet.layer3    # 256 channels
        self.layer4 = resnet.layer4    # 512 channels
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary of features at different scales:
            - feat1: [B, 64, H/4, W/4]
            - feat2: [B, 128, H/8, W/8]
            - feat3: [B, 256, H/16, W/16]
            - feat4: [B, 512, H/32, W/32]
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
    # Test encoder
    encoder = ResNet34Encoder(pretrained=True)
    x = torch.randn(2, 3, 384, 384)
    
    features = encoder(x)
    
    print("ResNet-34 Encoder Output:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")