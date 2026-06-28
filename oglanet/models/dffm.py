"""
Dense Feature Fusion Module (DFFM)
Implements dense connections between encoder and decoder with multi-scale downsampling.
Based on Figure 5 in the paper.
"""

import torch
import torch.nn as nn


class DFFM(nn.Module):
    """
    Dense Feature Fusion Module for fusing encoder features at multiple scales.
    
    From Figure 5 and Formula (2):
    - S4_D receives: 4×D(S1_E), 3×D(S2_E), 2×D(S3_E), 1×D(S4_E), S5_E
    - S3_D receives: 3×D(S1_E), 2×D(S2_E), 1×D(S3_E), S4_D
    - S2_D receives: 2×D(S1_E), 1×D(S2_E), S2_E, S3_D
    - S1_D receives: 1×D(S1_E), S1_E, S2_D
    
    where D = Conv3×3 with stride=2 (downsampling operation)
    and n×D means applying D operation n times sequentially
    
    ASSUMPTION: After concatenating all features, we use a 1×1 conv
    to reduce channels to target dimension. This is standard practice
    and necessary for manageable channel counts.
    """
    
    def __init__(self):
        super(DFFM, self).__init__()
        
        # Downsampling operations (D in Figure 5)
        # Each downsampling is a Conv3×3 stride=2
        self.downsample_conv = lambda in_c: nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
        
        # Channel reduction after concatenation
        # Channels: 64+128+256+512+1024 = 1984 -> 512
        self.reduce_s4 = nn.Sequential(
            nn.Conv2d(1984, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Channels: 64+128+256+512 = 960 -> 256
        self.reduce_s3 = nn.Sequential(
            nn.Conv2d(960, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Channels: 64+128+256 = 448 -> 128
        self.reduce_s2 = nn.Sequential(
            nn.Conv2d(448, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Channels: 64+128 = 192 -> 64
        self.reduce_s1 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
    def _apply_downsample_n_times(self, x, n):
        """
        Apply downsampling operation n times.
        
        Args:
            x: Input feature [B, C, H, W]
            n: Number of times to downsample
            
        Returns:
            Downsampled feature [B, C, H/(2^n), W/(2^n)]
        """
        for _ in range(n):
            downsample = self.downsample_conv(x.size(1))
            if torch.cuda.is_available():
                downsample = downsample.cuda()
            x = downsample(x)
        return x
    
    def forward(self, encoder_features, decoder_features=None):
        """
        Args:
            encoder_features: Dict with keys ['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
                              from GLAMEncoder (S1_E through S5_E)
            decoder_features: Dict with keys ['s4_d', 's3_d', 's2_d'] (optional, built progressively)
            
        Returns:
            Dict with decoder features ['s4_d', 's3_d', 's2_d', 's1_d']
        """
        s1_e = encoder_features['feat1']  # [B, 64, 96, 96]
        s2_e = encoder_features['feat2']  # [B, 128, 48, 48]
        s3_e = encoder_features['feat3']  # [B, 256, 24, 24]
        s4_e = encoder_features['feat4']  # [B, 512, 12, 12]
        s5_e = encoder_features['feat5']  # [B, 1024, 6, 6]
        
        # === Build S4_D ===
        # Concatenate: 4×D(S1_E), 3×D(S2_E), 2×D(S3_E), 1×D(S4_E), S5_E
        s1_to_s4 = self._apply_downsample_n_times(s1_e, 4)   # [B, 64, 6, 6]
        s2_to_s4 = self._apply_downsample_n_times(s2_e, 3)   # [B, 128, 6, 6]
        s3_to_s4 = self._apply_downsample_n_times(s3_e, 2)   # [B, 256, 6, 6]
        s4_to_s4 = self._apply_downsample_n_times(s4_e, 1)   # [B, 512, 6, 6]
        
        s4_d_concat = torch.cat([s1_to_s4, s2_to_s4, s3_to_s4, s4_to_s4, s5_e], dim=1)
        # Total channels: 64+128+256+512+1024 = 1984
        
        # Reduce to 512 channels
        if self.reduce_s4 is None:
            self.reduce_s4 = nn.Sequential(
                nn.Conv2d(s4_d_concat.size(1), 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
            if torch.cuda.is_available():
                self.reduce_s4 = self.reduce_s4.cuda()
        
        s4_d = self.reduce_s4(s4_d_concat)  # [B, 512, 6, 6]
        
        # Upsample S4_D for next stage
        s4_d_up = nn.functional.interpolate(s4_d, size=s4_e.shape[2:], mode='bilinear', align_corners=False)
        
        # === Build S3_D ===
        # Concatenate: 3×D(S1_E), 2×D(S2_E), 1×D(S3_E), S4_D_upsampled
        s1_to_s3 = self._apply_downsample_n_times(s1_e, 3)   # [B, 64, 12, 12]
        s2_to_s3 = self._apply_downsample_n_times(s2_e, 2)   # [B, 128, 12, 12]
        s3_to_s3 = self._apply_downsample_n_times(s3_e, 1)   # [B, 256, 12, 12]
        
        s3_d_concat = torch.cat([s1_to_s3, s2_to_s3, s3_to_s3, s4_d_up], dim=1)
        # Total channels: 64+128+256+512 = 960
        
        if self.reduce_s3 is None:
            self.reduce_s3 = nn.Sequential(
                nn.Conv2d(s3_d_concat.size(1), 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            if torch.cuda.is_available():
                self.reduce_s3 = self.reduce_s3.cuda()
        
        s3_d = self.reduce_s3(s3_d_concat)  # [B, 256, 12, 12]
        
        # Upsample S3_D for next stage
        s3_d_up = nn.functional.interpolate(s3_d, size=s3_e.shape[2:], mode='bilinear', align_corners=False)
        
        # === Build S2_D ===
        # Concatenate: 2×D(S1_E), 1×D(S2_E), S3_D (NOT S2_E)
        s1_to_s2 = self._apply_downsample_n_times(s1_e, 2)   # [B, 64, 24, 24]
        s2_to_s2 = self._apply_downsample_n_times(s2_e, 1)   # [B, 128, 24, 24]

        # Upsample S3_D to target resolution (24x24)
        s3_d_to_s2 = nn.functional.interpolate(s3_d, size=(s1_to_s2.shape[2], s1_to_s2.shape[3]), 
                                            mode='bilinear', align_corners=False)

        s2_d_concat = torch.cat([s1_to_s2, s2_to_s2, s3_d_to_s2], dim=1)
        # Total channels: 64+128+256 = 448

        if self.reduce_s2 is None:
            self.reduce_s2 = nn.Sequential(
                nn.Conv2d(s2_d_concat.size(1), 128, kernel_size=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            if torch.cuda.is_available():
                self.reduce_s2 = self.reduce_s2.cuda()

        s2_d = self.reduce_s2(s2_d_concat)  # [B, 128, 24, 24]

        # === Build S1_D ===
        # Concatenate: 1×D(S1_E), S2_D (NOT S1_E)
        s1_to_s1 = self._apply_downsample_n_times(s1_e, 1)   # [B, 64, 48, 48]

        # Upsample S2_D to target resolution (48x48)
        s2_d_to_s1 = nn.functional.interpolate(s2_d, size=(s1_to_s1.shape[2], s1_to_s1.shape[3]),
                                            mode='bilinear', align_corners=False)

        s1_d_concat = torch.cat([s1_to_s1, s2_d_to_s1], dim=1)
        # Total channels: 64+128 = 192

        if self.reduce_s1 is None:
            self.reduce_s1 = nn.Sequential(
                nn.Conv2d(s1_d_concat.size(1), 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            if torch.cuda.is_available():
                self.reduce_s1 = self.reduce_s1.cuda()

        s1_d = self.reduce_s1(s1_d_concat)  # [B, 64, 48, 48]
        
        return {
            's4_d': s4_d,  # [B, 512, 6, 6]
            's3_d': s3_d,  # [B, 256, 12, 12]
            's2_d': s2_d,  # [B, 128, 48, 48]
            's1_d': s1_d   # [B, 64, 96, 96]
        }


if __name__ == "__main__":
    # Test DFFM
    dffm = DFFM()
    
    # Simulate encoder features
    encoder_feats = {
        'feat1': torch.randn(2, 64, 96, 96),
        'feat2': torch.randn(2, 128, 48, 48),
        'feat3': torch.randn(2, 256, 24, 24),
        'feat4': torch.randn(2, 512, 12, 12),
        'feat5': torch.randn(2, 1024, 6, 6)
    }
    
    decoder_feats = dffm(encoder_feats)
    
    print("DFFM Outputs:")
    for name, feat in decoder_feats.items():
        print(f"{name}: {feat.shape}")