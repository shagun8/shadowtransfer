"""
Global-Local Aware Module (GLAM)
Combines Local Feature Extraction Module (LFEM) and Global Feature Extraction Module (GFEM).
Based on Figure 3 in the paper.
"""

import torch
import torch.nn as nn
from .gfem import GFEM
import torch.nn.functional as F


class GLAM(nn.Module):
    """
    Global-Local Aware Module combining local and global features.
    
    Architecture (from Figure 3a):
    - LFEM: ResNet block for local features
    - GFEM: Self-attention for global features (applied twice)
    - Pattern: Input → [LFEM, GFEM] → Concat → GFEM → ... → Output
    
    ASSUMPTION: After LFEM+GFEM concatenation, we use a 1×1 conv
    to reduce channels before passing to next GFEM. This is not
    explicitly shown in the paper but is necessary for dimension matching.
    """
    
    def __init__(self, lfem_block, in_channels, out_channels, lfem_downsample=False):
        """
        Args:
            lfem_block: ResNet block for local feature extraction
            in_channels: Number of input channels
            out_channels: Number of output channels
            lfem_downsample: Whether LFEM downsamples (True for layer2/3/4, False for layer1)
        """
        super(GLAM, self).__init__()
        
        # Local Feature Extraction Module (ResNet block)
        self.lfem = lfem_block
        
        # Global Feature Extraction Modules
        # GFEM1: downsample only if LFEM downsamples (to match spatial dims)
        self.gfem1 = GFEM(in_channels, downsample=lfem_downsample)
        
        # After concatenating LFEM (out_channels) and GFEM1 (in_channels) outputs
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # GFEM2: do not downsample, just refine features at current resolution
        self.gfem2 = GFEM(out_channels, downsample=False)
        
    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C_in, H, W]
            
        Returns:
            Output feature map after GLAM processing
        """
        # Local features from LFEM (ResNet block)
        local_feat = self.lfem(x)  # [B, C_out, H, W] or [B, C_out, H/2, W/2]
        
        # Global features from first GFEM
        global_feat1 = self.gfem1(x)  # [B, C_in, H/2, W/2]
        
        # Adjust spatial dimensions if they don't match
        # GFEM always downsamples by 2, so we need to match local_feat to it
        if local_feat.size(2) != global_feat1.size(2):
            # Downsample local_feat to match global_feat1
            local_feat = F.adaptive_avg_pool2d(local_feat, global_feat1.shape[2:])
        
        # Concatenate local and global features (Figure 3a, red box (c))
        concat_feat = torch.cat([local_feat, global_feat1], dim=1)
        
        # Reduce channels before next GFEM
        reduced_feat = self.channel_reduce(concat_feat)
        
        # Second GFEM
        output = self.gfem2(reduced_feat)
        
        return output


class GLAMEncoder(nn.Module):
    """
    Complete encoder with 5 GLAM stages.
    Mimics ResNet-34 structure but replaces each stage with GLAM.
    
    Based on Figure 2 encoder section:
    - GLAM1: 256×256×64   (at 384×384 input: 192×192×64)
    - GLAM2: 128×128×128  (at 384×384 input: 96×96×128)
    - GLAM3: 64×64×256    (at 384×384 input: 48×48×256)
    - GLAM4: 32×32×512    (at 384×384 input: 24×24×512)
    - GLAM5: 16×16×1024   (at 384×384 input: 12×12×1024)
    
    ASSUMPTION: We build GLAM stages using ResNet-34 blocks as LFEM base.
    Each GLAM includes the ResNet block plus 2 GFEM modules.
    """
    
    def __init__(self, pretrained=True, use_contrast=False):
        super(GLAMEncoder, self).__init__()
        
        # Import here to avoid circular dependency
        from .encoder import ResNet34Encoder
        from .encoder_4ch import ResNet34Encoder4Ch
        

        # LFEM: Local Feature Extraction Module (ResNet-34 or ResNet-34-4Ch)
        if use_contrast:
            self.resnet_encoder = ResNet34Encoder4Ch(pretrained=pretrained)
        else:
            self.resnet_encoder = ResNet34Encoder(pretrained=pretrained)
        
        # Wrap ResNet layers with GLAM
        # NOTE: For simplicity, we use the ResNet blocks directly as LFEM
        # and add GFEM modules around them
        
        # Initial conv before GLAM stages
        self.initial = nn.Sequential(
            self.resnet_encoder.conv1,
            self.resnet_encoder.bn1,
            self.resnet_encoder.relu,
            self.resnet_encoder.maxpool
        )
        
        # GLAM stages built on top of ResNet layers
        self.glam1 = self._make_glam_stage(self.resnet_encoder.layer1, 64, 64, lfem_downsample=False)
        self.glam2 = self._make_glam_stage(self.resnet_encoder.layer2, 64, 128, lfem_downsample=True)
        self.glam3 = self._make_glam_stage(self.resnet_encoder.layer3, 128, 256, lfem_downsample=True)
        self.glam4 = self._make_glam_stage(self.resnet_encoder.layer4, 256, 512, lfem_downsample=True)
        
        # GLAM5: Additional stage to get 1024 channels
        # ASSUMPTION: Add extra conv block to double channels to 1024
        self.glam5_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        self.glam5_gfem = GFEM(1024, downsample=False)
        
    def _make_glam_stage(self, resnet_layer, in_channels, out_channels, lfem_downsample=False):
        """Create a GLAM stage from a ResNet layer"""
        return GLAM(resnet_layer, in_channels, out_channels, lfem_downsample)
    
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W] (H=W=384)
            
        Returns:
            Dictionary with 5 feature maps at different scales
        """
        # Initial convolution
        x = self.initial(x)  # [B, 64, H/4, W/4] = [B, 64, 96, 96]
        
        # GLAM stages
        feat1 = self.glam1(x)      # [B, 64, H/4, W/4] = [B, 64, 96, 96]
        feat2 = self.glam2(feat1)   # [B, 128, H/8, W/8] = [B, 128, 48, 48]
        feat3 = self.glam3(feat2)   # [B, 256, H/16, W/16] = [B, 256, 24, 24]
        feat4 = self.glam4(feat3)   # [B, 512, H/32, W/32] = [B, 512, 12, 12]
        
        # GLAM5: Extra downsampling to get 1024 channels
        feat5 = self.glam5_conv(feat4)  # [B, 1024, H/64, W/64] = [B, 1024, 6, 6]
        feat5 = self.glam5_gfem(feat5)   # Further processing
        
        return {
            'feat1': feat1,  # S1_E in paper
            'feat2': feat2,  # S2_E
            'feat3': feat3,  # S3_E
            'feat4': feat4,  # S4_E
            'feat5': feat5   # S5_E
        }


if __name__ == "__main__":
    # Test GLAMEncoder
    encoder = GLAMEncoder(pretrained=True)
    x = torch.randn(2, 3, 384, 384)
    
    features = encoder(x)
    
    print("GLAMEncoder Output:")
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")