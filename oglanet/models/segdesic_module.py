"""
SegDesicNet: Geographic Coordinate Embeddings for Domain Adaptation
WACV 2025 - Verma et al.

Encodes geographic coordinates (lat/lon) using spherical GRID positional encoding
to enable domain adaptation through geographic metadata.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GRIDPositionalEncoding(nn.Module):
    """
    GRID (Geographic Rotational Invariant Distribution) Positional Encoding.
    
    Encodes lat/lon coordinates on a unit sphere using sinusoidal functions
    across multiple scales (like original Transformer positional encoding but for geography).
    """
    
    def __init__(self, num_scales=10, lambda_min=1e-4, lambda_max=1e4):
        """
        Args:
            num_scales: Number of scales for multi-scale encoding
            lambda_min: Minimum wavelength
            lambda_max: Maximum wavelength
        """
        super().__init__()
        self.num_scales = num_scales
        
        # Compute wavelengths (logarithmically spaced)
        lambdas = torch.logspace(
            math.log10(lambda_min),
            math.log10(lambda_max),
            num_scales
        )
        self.register_buffer('lambdas', lambdas)
        
        # Output dimension: 4 values per scale (sin/cos for both lat and lon)
        self.encoding_dim = 4 * num_scales
    
    def forward(self, lat, lon):
        """
        Encode geographic coordinates.
        
        Args:
            lat: Latitude in degrees, tensor [B]
            lon: Longitude in degrees, tensor [B]
        
        Returns:
            Encoded coordinates [B, encoding_dim]
        """
        # Convert to radians
        lat_rad = lat * (math.pi / 180.0)
        lon_rad = lon * (math.pi / 180.0)
        
        # Project onto unit sphere (x, y, z)
        x = torch.cos(lat_rad) * torch.cos(lon_rad)
        y = torch.cos(lat_rad) * torch.sin(lon_rad)
        z = torch.sin(lat_rad)
        
        # Stack coordinates
        coords = torch.stack([x, y], dim=1)  # [B, 2] (we use x, y for encoding)
        
        # Compute multi-scale sinusoidal encoding
        encodings = []
        for lambda_val in self.lambdas:
            # Frequency = 2*pi / lambda
            freq = (2 * math.pi) / lambda_val
            
            # Sin and cos for each coordinate
            sin_x = torch.sin(freq * coords[:, 0])
            cos_x = torch.cos(freq * coords[:, 0])
            sin_y = torch.sin(freq * coords[:, 1])
            cos_y = torch.cos(freq * coords[:, 1])
            
            encodings.extend([sin_x, cos_x, sin_y, cos_y])
        
        # Concatenate all encodings
        encoded = torch.stack(encodings, dim=1)  # [B, 4*num_scales]
        
        return encoded


class SegDesicModule(nn.Module):
    """
    SegDesic Module: Geographic coordinate embedding and prediction.
    
    Architecture:
    - Takes encoder features as input
    - Predicts geographic coordinates
    - Uses for domain loss calculation during training
    """
    
    def __init__(self, in_channels=512, hidden_dim=256, num_scales=10):
        """
        Args:
            in_channels: Number of input channels from encoder (e.g., 512 from ResNet-34)
            hidden_dim: Hidden dimension for MLP
            num_scales: Number of scales for GRID encoding
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.encoding_dim = 4 * num_scales
        
        # GRID positional encoding
        self.grid_encoding = GRIDPositionalEncoding(num_scales=num_scales)
        
        # MLP to predict geographic coordinates from features
        # Input: encoder features [B, C, H, W]
        # Output: predicted coordinate encoding [B, encoding_dim]
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # [B, C, H, W] -> [B, C, 1, 1]
        
        self.predictor = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, self.encoding_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, lat=None, lon=None):
        """
        Forward pass.
        
        Args:
            features: Encoder features [B, C, H, W]
            lat: Ground truth latitude [B] (optional, for training)
            lon: Ground truth longitude [B] (optional, for training)
        
        Returns:
            Dictionary containing:
                - 'pred_encoding': Predicted coordinate encoding [B, encoding_dim]
                - 'gt_encoding': Ground truth coordinate encoding [B, encoding_dim] (if lat/lon provided)
        """
        B = features.size(0)
        
        # Global average pooling
        pooled = self.global_pool(features)  # [B, C, 1, 1]
        pooled = pooled.view(B, -1)  # [B, C]
        
        # Predict coordinate encoding
        pred_encoding = self.predictor(pooled)  # [B, encoding_dim]
        
        # Normalize to unit hypersphere (for cosine similarity)
        pred_encoding = F.normalize(pred_encoding, p=2, dim=1)
        
        result = {'pred_encoding': pred_encoding}
        
        # Encode ground truth coordinates if provided
        if lat is not None and lon is not None:
            gt_encoding = self.grid_encoding(lat, lon)  # [B, encoding_dim]
            gt_encoding = F.normalize(gt_encoding, p=2, dim=1)
            result['gt_encoding'] = gt_encoding
        
        return result


if __name__ == "__main__":
    # Test GRID encoding
    print("Testing GRID Positional Encoding...")
    grid_enc = GRIDPositionalEncoding(num_scales=10)
    
    # Test with some cities
    cities_coords = {
        'Phoenix': (33.4484, -112.0740),
        'Miami': (25.7617, -80.1918),
        'Chicago': (41.8781, -87.6298)
    }
    
    for city, (lat, lon) in cities_coords.items():
        lat_t = torch.tensor([lat])
        lon_t = torch.tensor([lon])
        encoding = grid_enc(lat_t, lon_t)
        print(f"{city}: lat={lat:.4f}, lon={lon:.4f} -> encoding shape={encoding.shape}")
    
    # Test SegDesic module
    print("\nTesting SegDesic Module...")
    module = SegDesicModule(in_channels=512, hidden_dim=256, num_scales=10)
    
    # Dummy features
    features = torch.randn(4, 512, 16, 16)
    lat = torch.tensor([33.4484, 25.7617, 41.8781, 33.5])
    lon = torch.tensor([-112.0740, -80.1918, -87.6298, -112.1])
    
    output = module(features, lat, lon)
    print(f"Predicted encoding shape: {output['pred_encoding'].shape}")
    print(f"GT encoding shape: {output['gt_encoding'].shape}")
    
    # Test cosine similarity
    cos_sim = F.cosine_similarity(output['pred_encoding'], output['gt_encoding'], dim=1)
    print(f"Cosine similarity: {cos_sim}")