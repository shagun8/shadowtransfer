"""
DINOv3 Backbone for Shadow Detection
Loads the actual DINOv3 (ViT-S/16) model and extracts multi-scale features.

NOTE: DINOv3 uses patch size 16 (not 14 like DINOv2)
This means 384÷16 = 24 patches exactly - no padding needed!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DINOv3Backbone(nn.Module):
    """
    DINOv3 Vision Transformer Backbone (ViT-S/16)
    
    Architecture:
    - Patch size: 16×16 (perfect for 384×384 input!)
    - Embedding dim: 384
    - 12 transformer blocks
    - Extracts features from blocks [3, 6, 9, 11] for skip connections
    
    Input: [B, 3, H, W] where H, W are multiples of 16
    Output: Dictionary of multi-scale features at 1/16 resolution
    """
    
    def __init__(self, model_name='dinov3_vits16', weights_path=None, pretrained=True, frozen_stages=-1):
        """
        Args:
            model_name: DINOv3 model variant
                - 'dinov3_vits16': ViT-S/16 (384 dim, 12 blocks, ~22M params)
                - 'dinov3_vitb16': ViT-B/16 (768 dim, 12 blocks, ~86M params)
                - 'dinov3_vitl16': ViT-L/16 (1024 dim, 24 blocks, ~304M params)
            weights_path: Path to pretrained weights .pth file
            pretrained: Load pretrained weights
            frozen_stages: Number of stages to freeze (-1 = train all)
        """
        super(DINOv3Backbone, self).__init__()
        
        self.model_name = model_name
        self.frozen_stages = frozen_stages
        
        # Load pretrained DINOv3
        if pretrained:
            print(f'Loading pretrained {model_name}...')
            
            # Use the official DINOv3 package
            try:
                # from dinov3.dinov3.models import vision_transformer as vits
                import sys
                # Add cloned repo root so internal 'from dinov3.xxx' imports resolve correctly
                dinov3_repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dinov3')
                if dinov3_repo_root not in sys.path:
                    sys.path.insert(0, dinov3_repo_root)
                from dinov3.models import vision_transformer as vits
                
                # Build model based on variant
                if 'vits16' in model_name:
                    self.dinov3 = vits.vit_small(patch_size=16, img_size=384, init_values=1.0, block_chunks=0)
                elif 'vitb16' in model_name:
                    self.dinov3 = vits.vit_base(patch_size=16, img_size=384, init_values=1.0, block_chunks=0)
                elif 'vitl16' in model_name:
                    self.dinov3 = vits.vit_large(patch_size=16, img_size=384, init_values=1.0, block_chunks=0)
                else:
                    raise ValueError(f"Unknown model variant: {model_name}")
                
                # Load weights if path provided
                if weights_path and os.path.exists(weights_path):
                    print(f'Loading weights from: {weights_path}')
                    state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
                    
                    # Handle different state dict formats
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'teacher' in state_dict:
                        state_dict = state_dict['teacher']
                    
                    # Remove 'module.' prefix if present
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    
                    # Load state dict
                    msg = self.dinov3.load_state_dict(state_dict, strict=False)
                    if len(msg.missing_keys) > 0:
                        print(f"  Missing keys: {msg.missing_keys[:5]}...")  # Show first 5
                    if len(msg.unexpected_keys) > 0:
                        print(f"  Unexpected keys: {msg.unexpected_keys[:5]}...")  # Show first 5
                    print(f'✓ Loaded from local weights')
                else:
                    print('Warning: No weights_path provided, using random initialization')
                    
            except ImportError as e:
                raise ImportError(
                    f"\nCould not import the dinov3 package. Clone the upstream repo into\n"
                    f"the path this backbone expects (a 'dinov3/' dir next to dinov3_backbone.py):\n"
                    f"  git clone https://github.com/facebookresearch/dinov3.git {dinov3_repo_root}\n"
                    f"  pip install -e {dinov3_repo_root}\n\n"
                    f"Original error: {e}"
                )
        else:
            raise NotImplementedError("Non-pretrained DINOv3 not implemented")
        
        # Get model dimensions
        self.embed_dim = self.dinov3.embed_dim  # 384 for ViT-S
        self.patch_size = self.dinov3.patch_size  # 16 for DINOv3
        self.num_blocks = len(self.dinov3.blocks)  # 12 for ViT-S
        
        # Feature extraction points (for skip connections)
        self.feature_blocks = [3, 6, 9, 11]
        
        print(f'DINOv3 Backbone initialized:')
        print(f'  Model: {model_name}')
        print(f'  Embed dim: {self.embed_dim}')
        print(f'  Patch size: {self.patch_size}')
        print(f'  Num blocks: {self.num_blocks}')
        print(f'  Feature blocks: {self.feature_blocks}')
        print(f'  384÷16 = 24 patches (perfect fit!)')
        
        # Freeze stages if specified
        if frozen_stages >= 0:
            self._freeze_stages(frozen_stages)

    def _load_from_local_weights(self, model_name, weights_path):
        """
        Load DINOv3 model from local weights file.
        
        This requires the dinov3 package to be installed:
            git clone https://github.com/facebookresearch/dinov3
            cd dinov3
            pip install -e .
        """
        try:
            # Try importing from installed dinov3 package
            import dinov3
            from dinov3.dinov3.models import vision_transformer as vit
            
            # Build model based on variant
            if 'vits16' in model_name:
                model = vit.vit_small(patch_size=16)
            elif 'vitb16' in model_name:
                model = vit.vit_base(patch_size=16)
            elif 'vitl16' in model_name:
                model = vit.vit_large(patch_size=16)
            else:
                raise ValueError(f"Unknown model variant: {model_name}")
            
            # Load weights
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
            
            # Handle different state dict formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Load state dict
            msg = model.load_state_dict(state_dict, strict=False)
            if len(msg.missing_keys) > 0:
                print(f"  Missing keys: {msg.missing_keys}")
            if len(msg.unexpected_keys) > 0:
                print(f"  Unexpected keys: {msg.unexpected_keys}")
            
            return model
            
        except ImportError as e:
            raise ImportError(
                f"\nCould not import dinov3 package. Please install it:\n"
                f"  git clone https://github.com/facebookresearch/dinov3\n"
                f"  cd dinov3\n"
                f"  pip install -e .\n\n"
                f"Original error: {e}"
            )
    
    def _freeze_stages(self, num_stages):
        """Freeze the first num_stages blocks"""
        print(f'Freezing first {num_stages} blocks...')
        
        # Freeze patch embedding
        if hasattr(self.dinov3, 'patch_embed'):
            self.dinov3.patch_embed.eval()
            for param in self.dinov3.patch_embed.parameters():
                param.requires_grad = False
        
        # Freeze positional embedding
        if hasattr(self.dinov3, 'pos_embed'):
            self.dinov3.pos_embed.requires_grad = False
        
        # Freeze specified blocks
        for i in range(min(num_stages, self.num_blocks)):
            block = self.dinov3.blocks[i]
            block.eval()
            for param in block.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Extract multi-scale features from DINOv3
        
        Args:
            x: Input images [B, 3, H, W] (H, W should be multiples of 16)
        
        Returns:
            Dictionary with features from blocks [3, 6, 9, 11]
        """
        B, C, H, W = x.shape
        
        # Ensure input is multiple of patch_size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size ({H}, {W}) must be divisible by patch_size ({self.patch_size})"
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        # Use DINOv3's built-in method to get intermediate layer features
        # This is the correct way to extract features at different depths
        features_list = self.dinov3.get_intermediate_layers(
            x, 
            n=self.feature_blocks,  # [3, 6, 9, 11]
            return_class_token=False,
            reshape=True  # This will reshape to spatial format
        )
        
        # Convert list to dictionary with proper keys
        features = {}
        for i, feat_idx in enumerate(self.feature_blocks):
            # features_list[i] should be [B, embed_dim, H/16, W/16]
            features[f'feat_block{feat_idx}'] = features_list[i]
        
        return features
    
    def get_feature_dims(self):
        """Return the dimensions of extracted features"""
        return {
            'feat_block3': self.embed_dim,
            'feat_block6': self.embed_dim,
            'feat_block9': self.embed_dim,
            'feat_block11': self.embed_dim,
        }


if __name__ == "__main__":
    # Test backbone
    print("="*60)
    print("Testing DINOv3 Backbone")
    print("="*60)
    
    # Try loading with torch.hub (may fail if repo not accessible)
    try:
        backbone = DINOv3Backbone(model_name='dinov3_vits16', pretrained=True)
        
        # Test with 384x384 input (24×24 patches - perfect!)
        x = torch.randn(2, 3, 384, 384)
        print(f"\nInput shape: {x.shape}")
        
        features = backbone(x)
        
        print("\nExtracted features:")
        for name, feat in features.items():
            print(f"  {name}: {feat.shape}")
        
        print(f"\nFeature dimensions: {backbone.get_feature_dims()}")
        
        # Count parameters
        total_params = sum(p.numel() for p in backbone.parameters())
        trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("\n" + "="*60)
        print("✓ Test passed!")
        print("="*60)
        
    except Exception as e:
        print(f"\nNote: {e}")
        print("\nTo test with local weights, use:")
        print("  backbone = DINOv3Backbone(")
        print("      model_name='dinov3_vits16',")
        print("      weights_path='path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'")
        print("  )")