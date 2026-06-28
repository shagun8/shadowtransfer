"""
MAMNet: Multi-Scale Spatial Channel Attention Network for Shadow Detection
Complete model integrating encoder, MSCAF, CCA, decoder, and auxiliary branches.
"""

import torch
import torch.nn as nn
from .encoder import ResNet34Encoder
from .encoder_4ch import ResNet34Encoder4Ch
from .mscaf import MSCAF
from .decoder import Decoder
from .auxiliary import AuxiliaryModule


class MAMNet(nn.Module):
    """
    MAMNet: Full-Scale Shadow Detection Network Based on Multiple Attention Mechanisms
    
    Architecture:
    1. ResNet-34 Encoder (pretrained ImageNet)
    2. MSCAF module for multi-scale feature fusion
    3. Decoder with CCA modules at each stage
    4. Auxiliary branches for deep supervision
    
    Paper: Zhang et al. (2024)
    """
    
    def __init__(self, num_classes=2, pretrained=True, use_aux=True, use_contrast=False):
        super(MAMNet, self).__init__()
        
        self.num_classes = num_classes
        self.use_aux = use_aux
        self.use_contrast = use_contrast
        
        # Encoder: ResNet-34
        # self.encoder = ResNet34Encoder(pretrained=pretrained)
        if use_contrast:
            self.encoder = ResNet34Encoder4Ch(pretrained=pretrained)
            print("Using 4-channel encoder (RGB + Contrast)")
        else:
            self.encoder = ResNet34Encoder(pretrained=pretrained)
        
        # MSCAF: Multi-Scale Spatial Channel Attention Fusion
        self.mscaf = MSCAF(in_channels=512)
        
        # Decoder with CCA modules
        self.decoder = Decoder(num_classes=num_classes)
        
        # Auxiliary branches
        if use_aux:
            self.aux_module = AuxiliaryModule(num_classes=num_classes, dropout_rate=0.3)
        
    def forward(self, x):
        """
        Args:
            x: Input RGB images [B, 3, H, W] or RGBC images [B, 4, H, W] if use_contrast=True
            
        Returns:
            If training (use_aux=True):
                Dictionary with 'main' and auxiliary outputs ('aux1', 'aux2', 'aux3')
            If inference (use_aux=False):
                Main prediction only [B, num_classes, H, W]
        """
        B, _, H, W = x.size()
        
        
        # Encoder: Extract multi-scale features
        enc_features = self.encoder(x)
        # enc_features contains: feat1 (64), feat2 (128), feat3 (256), feat4 (512)
        
        # Apply MSCAF to deepest features
        mscaf_out = self.mscaf(enc_features['feat5'])  # [B, 512, H/16, W/16]
        
        # Decoder with CCA
        decoder_outputs = self.decoder(mscaf_out, enc_features)
        main_out = decoder_outputs['main']  # [B, num_classes, H, W]
        
        # Auxiliary branches (only during training)
        if self.use_aux and self.training:
            aux_outputs = self.aux_module(
                decoder_outputs['dec_feat1'],  # [B, 256, 32, 32]
                decoder_outputs['dec_feat2'],  # [B, 128, 64, 64]
                decoder_outputs['dec_feat3'],  # [B, 64, 128, 128]
                target_size=(H, W)
            )
            
            # Return all outputs for loss calculation
            return {
                'main': main_out,
                'aux1': aux_outputs['aux1'],
                'aux2': aux_outputs['aux2'],
                'aux3': aux_outputs['aux3']
            }
        else:
            # Inference: return only main output
            return main_out
    
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
    # Test complete MAMNet
    model = MAMNet(num_classes=2, pretrained=True, use_aux=True)
    
    # Training mode
    model.train()
    x = torch.randn(2, 3, 256, 256)
    outputs_train = model(x)
    
    print("Training outputs:")
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