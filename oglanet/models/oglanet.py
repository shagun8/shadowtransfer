"""
OGLANet: Omni-scale Global-Local Aware Network for Shadow Extraction
Complete model integrating GLAM encoder, DFFM, Decoder, and OAM.

Paper: Xie et al. (2022) - ISPRS Journal of Photogrammetry and Remote Sensing

CRITICAL ASSUMPTION:
Using ResNet-34 instead of ResNet-101 (paper specification) to ensure
fair comparison with MAMNet implementation and avoid performance gains
solely from increased parameter count.
"""

import torch
import torch.nn as nn
from .glam import GLAMEncoder
from .dffm import DFFM
from .decoder import Decoder
from .oam import OAM


class OGLANet(nn.Module):
    """
    Complete OGLANet architecture for shadow detection.
    
    Architecture:
    1. GLAM Encoder: 5 stages with global-local aware modules
    2. DFFM: Dense feature fusion between encoder and decoder
    3. Decoder: Progressive upsampling with convolution blocks
    4. OAM: Omni-scale aggregation with 6 prediction outputs
    
    Training outputs: 6 predictions (P1-P6) for deep supervision
    Inference output: P6 (final aggregated prediction)
    """
    
    def __init__(self, num_classes=2, pretrained=True, img_size=384, use_contrast=False):
        super(OGLANet, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.use_contrast = use_contrast
        
        # 1. GLAM Encoder
        self.encoder = GLAMEncoder(pretrained=pretrained, use_contrast = use_contrast)
        
        # 2. Dense Feature Fusion Module
        self.dffm = DFFM()
        
        # 3. Decoder
        self.decoder = Decoder(target_size=(img_size, img_size))
        
        # 4. Omni-scale Aggregation Module
        self.oam = OAM(num_classes=num_classes, target_size=(img_size, img_size))
        
    def forward(self, x):
        """
        Args:
            x: Input RGB images [B, 3, H, W] (H=W=384)
            
        Returns:
            If training:
                Dict with 6 predictions {'p1', 'p2', 'p3', 'p4', 'p5', 'p6'}
                Each prediction is [B, num_classes, H, W]
            If inference:
                Final prediction P6 [B, num_classes, H, W]
        """
        B, _, H, W = x.size()
        
        # 1. Encoder: Extract multi-scale features with GLAM
        encoder_features = self.encoder(x)
        # Returns: {'feat1', 'feat2', 'feat3', 'feat4', 'feat5'}
        
        # 2. DFFM: Dense feature fusion
        dffm_features = self.dffm(encoder_features)
        # Returns: {'s4_d', 's3_d', 's2_d', 's1_d'}
        
        # 3. Decoder: Upsample to original resolution
        decoder_features = self.decoder(dffm_features)
        # Returns: {'s4_d_up', 's3_d_up', 's2_d_up', 's1_d_up'}
        
        # 4. OAM: Multi-scale prediction
        predictions = self.oam(decoder_features)
        # Returns: {'p1', 'p2', 'p3', 'p4', 'p5', 'p6'}
        
        if self.training:
            # Return all 6 predictions for deep supervision loss
            return predictions
        else:
            # Return only final prediction P6 for inference
            return predictions['p6']
    
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


if __name__ == "__main__":
    # Test complete OGLANet
    print("="*50)
    print("Testing OGLANet")
    print("="*50)
    
    model = OGLANet(num_classes=2, pretrained=False, img_size=384)
    
    # Training mode
    model.train()
    x = torch.randn(2, 3, 384, 384)
    outputs_train = model(x)
    
    print("\nTraining outputs:")
    for key, val in outputs_train.items():
        print(f"{key}: {val.shape}")
    
    # Inference mode
    model.eval()
    outputs_eval = model(x)
    print(f"\nInference output: {outputs_eval.shape}")
    
    # Get binary predictions
    preds = model.get_predictions(x)
    print(f"Binary predictions: {preds.shape}, unique values: {torch.unique(preds)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")