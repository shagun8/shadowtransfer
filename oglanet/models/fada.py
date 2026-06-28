"""
FADA: Frequency-Adapted Learning Module for Domain Generalized Shadow Detection

Adapted from: "Learning Frequency-Adapted Vision Foundation Model for
Domain Generalized Semantic Segmentation" (Bi et al., NeurIPS 2024)

Original paper targets ViT backbones; this adaptation targets ResNet-34 (CNN).

===============================================================================
ASSUMPTIONS for CNN (ResNet-34) adaptation:
===============================================================================

1. LAYER SELECTION:
   FADA is applied after encoder stages 3, 4, 5 (feat3, feat4, feat5) where
   spatial dimensions are manageable for token-based attention:
     feat3: [B, 256,  96,  96] → Haar LL: 48×48 = 2304 tokens
     feat4: [B, 512,  48,  48] → Haar LL: 24×24 =  576 tokens
     feat5: [B, 512,  24,  24] → Haar LL: 12×12 =  144 tokens
   feat1 (384²=147K tokens) and feat2 (192²=37K tokens) are left unchanged
   because the token attention similarity map would be prohibitively large.
   Paper's Table 6 shows applying to all layers is best, but shallow layers
   are slightly more important; our 3-stage coverage is a practical compromise.

2. SPATIAL vs. SEQUENCE:
   The paper operates on ViT token sequences (f_i ∈ R^{c×n}). For CNN feature
   maps (f ∈ R^{B×C×H×W}), we reshape to [B, n, C] (n=H*W) for token-based
   attention, then reshape back. The 2D Haar DWT is applied on the spatial
   dimensions (H, W) of the feature map.

3. BRANCH ROLES (following paper Sec. 4.2 & 4.3):
   - Low-frequency branch (LL): Learnable tokens exploit scene content from
     the low-frequency component. Content is stable across cities; tokens
     learn to identify and stabilize it (Eq. 5-7).
   - High-frequency branch (LH, HL, HH): Instance Normalization on the
     feature-token similarity map removes city-specific style information.
     This does NOT destroy boundary structure — it removes domain-specific
     *patterns* in how the model attends to edges (Eq. 8-11).

4. LORA DECOMPOSITION (following paper Sec. 3.1):
   Tokens are parameterized as low-rank matrices: T = A @ B where
   A ∈ R^{m×r}, B ∈ R^{r×c}, r << min(m, c). Default r=16 (Table 3).

5. HYPERPARAMETERS (paper defaults):
   - Token length m = 100 (stable in 75-125 range, Fig. 8)
   - LoRA rank r = 16 (best at 16-32, Table 3)
   - These are configurable via constructor arguments.
===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# Haar Wavelet Transform (2D, single-level)
# Paper Sec. 3.2 & Eq. 3-4
# ---------------------------------------------------------------------------

class HaarWaveletTransform2D:
    """
    Single-level 2D Haar Wavelet Transform for CNN feature maps.

    Forward: [B, C, H, W] → (LL, LH, HL, HH), each [B, C, H/2, W/2]
    Inverse: (LL, LH, HL, HH) → [B, C, H, W]

    Kernels (Eq. 3):
        L^T = (1/√2) [1,  1]
        H^T = (1/√2) [-1, 1]

    2D subbands (Eq. 4):
        LL = L^T ⊗ L^T  → low-freq approx  (content)
        LH = L^T ⊗ H^T  → horizontal detail (style / boundary)
        HL = H^T ⊗ L^T  → vertical detail   (style / boundary)
        HH = H^T ⊗ H^T  → diagonal detail   (style / boundary)
    """

    @staticmethod
    def forward(x):
        """
        Forward 2D Haar DWT.

        Args:
            x: Feature map [B, C, H, W]

        Returns:
            LL, LH, HL, HH: Each [B, C, H/2, W/2]
            original_size: Tuple (H, W) for inverse (handles odd dims)
        """
        B, C, H, W = x.shape

        # Pad odd dimensions with reflection to ensure even H, W
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Polyphase decomposition (efficient strided indexing)
        x00 = x[:, :, 0::2, 0::2]  # even row, even col
        x01 = x[:, :, 0::2, 1::2]  # even row, odd  col
        x10 = x[:, :, 1::2, 0::2]  # odd  row, even col
        x11 = x[:, :, 1::2, 1::2]  # odd  row, odd  col

        # 2D Haar forward transform (separable application of L and H)
        LL = (x00 + x01 + x10 + x11) / 2.0   # average → content
        LH = (-x00 + x01 - x10 + x11) / 2.0  # horizontal edges
        HL = (-x00 - x01 + x10 + x11) / 2.0  # vertical edges
        HH = (x00 - x01 - x10 + x11) / 2.0   # diagonal detail

        return LL, LH, HL, HH, (H, W)

    @staticmethod
    def inverse(LL, LH, HL, HH, original_size=None):
        """
        Inverse 2D Haar DWT.

        Reconstructs the original feature map from four subbands.
        Uses the orthogonal property: inverse = transpose of forward.

        Args:
            LL, LH, HL, HH: Subbands, each [B, C, h, w]
            original_size: Optional (H, W) to crop if forward used padding

        Returns:
            Reconstructed feature map [B, C, H, W]
        """
        B, C, h, w = LL.shape
        H_out, W_out = h * 2, w * 2

        out = LL.new_zeros(B, C, H_out, W_out)

        out[:, :, 0::2, 0::2] = (LL - LH - HL + HH) / 2.0
        out[:, :, 0::2, 1::2] = (LL + LH - HL - HH) / 2.0
        out[:, :, 1::2, 0::2] = (LL - LH + HL - HH) / 2.0
        out[:, :, 1::2, 1::2] = (LL + LH + HL + HH) / 2.0

        # Crop to original spatial dimensions if forward had padding
        if original_size is not None:
            orig_H, orig_W = original_size
            if H_out != orig_H or W_out != orig_W:
                out = out[:, :, :orig_H, :orig_W]

        return out


# ---------------------------------------------------------------------------
# LoRA Token  (Sec. 3.1)
# ---------------------------------------------------------------------------

class LoRAToken(nn.Module):
    """
    Low-rank parameterized token: T = A @ B

    A ∈ R^{m × r}, B ∈ R^{r × c}, where r << min(m, c).
    This follows the LoRA decomposition from Sec. 3.1:
        T_i = A_i B_i

    Initialization:
        A ~ N(0, 1/r)  — scales with rank to keep product magnitude stable
        B ~ N(0, 1/c)  — scales with channel dim
    """

    def __init__(self, token_length, channels, rank):
        super().__init__()
        self.A = nn.Parameter(torch.randn(token_length, rank) / math.sqrt(rank))
        self.B = nn.Parameter(torch.randn(rank, channels) / math.sqrt(channels))

    def forward(self):
        """Returns token matrix T = A @ B of shape [m, c]."""
        return self.A @ self.B


# ---------------------------------------------------------------------------
# Low-Frequency Branch  (Sec. 4.2, Eq. 5–7)
# ---------------------------------------------------------------------------

class LowFrequencyBranch(nn.Module):
    """
    Low-frequency adaptation branch.

    Stabilises scene content from the LL subband using learnable LoRA tokens.
    The key insight is that the LL component captures average pixel responses
    which are more robust to cross-domain style variation and thus carry
    scene content (Sec. 4.1).

    Pipeline:
        1. Compute similarity map S_L between LL features and token T_L  (Eq. 5)
        2. Project token → feature space, weight by S_L                  (Eq. 6)
        3. Fuse projected tokens with LL features via skip connection     (Eq. 7)
    """

    def __init__(self, channels, token_length=100, rank=16):
        super().__init__()
        self.channels = channels
        self.token_length = token_length

        # LoRA token T_L = A_L @ B_L   (Sec. 3.1)
        self.token = LoRAToken(token_length, channels, rank)

        # MLP_1: project token to feature space  (Eq. 6: W1, b1)
        self.mlp1 = nn.Linear(channels, channels)

        # MLP_2: fuse projected tokens + features (Eq. 7: W2, b2)
        self.mlp2 = nn.Linear(channels, channels)

        self.scale = 1.0 / math.sqrt(channels)

    def forward(self, f_LL):
        """
        Args:
            f_LL: Low-frequency features [B, C, h, w]

        Returns:
            Adapted low-frequency features [B, C, h, w]
        """
        B, C, h, w = f_LL.shape
        n = h * w  # number of spatial positions (analogous to ViT patch count)

        # Reshape to sequence: [B, n, C]
        f_flat = f_LL.permute(0, 2, 3, 1).reshape(B, n, C)

        # Get token: [m, C]
        T_L = self.token()

        # Eq. 5: similarity map S_L = Softmax(f_LL × T_L^T / √c)
        # Shape: [B, n, m]
        S_L = torch.softmax(f_flat @ T_L.t() * self.scale, dim=-1)

        # Eq. 6: f̄_LL = S_L × MLP_1(T_L)
        T_proj = self.mlp1(T_L)   # [m, C]
        f_bar = S_L @ T_proj      # [B, n, C]

        # Eq. 7: f̃_LL = f_LL + MLP_2(f̄_LL + f_LL)
        f_tilde = f_flat + self.mlp2(f_bar + f_flat)

        # Reshape back to spatial: [B, C, h, w]
        return f_tilde.reshape(B, h, w, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# High-Frequency Branch  (Sec. 4.3, Eq. 8–11)
# ---------------------------------------------------------------------------

class HighFrequencyBranch(nn.Module):
    """
    High-frequency adaptation branch.

    Mitigates city-specific style information from the LH, HL, HH subbands
    while preserving structural/boundary information relevant to shadows.

    Key mechanism: Instance Normalization on the feature-token similarity map
    (Eq. 9) removes domain-specific attention patterns. This does NOT destroy
    edge information — it normalises *how* the model attends to edges, making
    the attention distribution style-invariant (see Fig. 5 in paper).

    Pipeline:
        1. Concatenate three HF components along spatial dim             (Sec. 4.3)
        2. Compute similarity map S_H between HF features and token T_H  (Eq. 8)
        3. Apply Instance Normalization to S_H                            (Eq. 9)
        4. Project token → feature space, weight by normalised S_H        (Eq. 10)
        5. Fuse with HF features via skip connection                      (Eq. 11)
    """

    def __init__(self, channels, token_length=100, rank=16):
        super().__init__()
        self.channels = channels
        # Token size is 3× because three HF components are concatenated
        self.token_length_3 = token_length * 3

        # LoRA token T_H = A_H @ B_H   (3m × c)
        self.token = LoRAToken(self.token_length_3, channels, rank)

        # MLP_3: project token  (Eq. 10: W3, b3)
        self.mlp3 = nn.Linear(channels, channels)

        # MLP_4: fuse           (Eq. 11: W4, b4)
        self.mlp4 = nn.Linear(channels, channels)

        self.scale = 1.0 / math.sqrt(channels)

    def forward(self, f_LH, f_HL, f_HH):
        """
        Args:
            f_LH, f_HL, f_HH: High-frequency features, each [B, C, h, w]

        Returns:
            Tuple of adapted (f_LH, f_HL, f_HH), each [B, C, h, w]
        """
        B, C, h, w = f_LH.shape
        n = h * w

        # Reshape and concatenate along spatial dimension: [B, 3n, C]
        f_LH_flat = f_LH.permute(0, 2, 3, 1).reshape(B, n, C)
        f_HL_flat = f_HL.permute(0, 2, 3, 1).reshape(B, n, C)
        f_HH_flat = f_HH.permute(0, 2, 3, 1).reshape(B, n, C)
        f_cat = torch.cat([f_LH_flat, f_HL_flat, f_HH_flat], dim=1)  # [B, 3n, C]

        # Get token: [3m, C]
        T_H = self.token()

        # Eq. 8: similarity map S_H = Softmax([f_LH, f_HL, f_HH] × T_H^T / √c)
        # Shape: [B, 3n, 3m]
        S_H = torch.softmax(f_cat @ T_H.t() * self.scale, dim=-1)

        # Eq. 9: Instance Normalization on S_H
        # Normalise over the token dimension (3m) for each (batch, spatial_position).
        # This removes domain-specific bias in attention patterns —
        # high responses reflecting city-specific styles are suppressed.
        mu = S_H.mean(dim=-1, keepdim=True)         # [B, 3n, 1]
        sigma = S_H.std(dim=-1, keepdim=True) + 1e-5  # [B, 3n, 1]
        S_H_normed = (S_H - mu) / sigma

        # Eq. 10: f̄_H = S̃_H × MLP_3(T_H)
        T_proj = self.mlp3(T_H)          # [3m, C]
        f_bar = S_H_normed @ T_proj      # [B, 3n, C]

        # Eq. 11: [f̃_LH, f̃_HL, f̃_HH] = concat + MLP_4(f̄_H + concat)
        f_tilde = f_cat + self.mlp4(f_bar + f_cat)  # [B, 3n, C]

        # Split back into three components and reshape to spatial
        f_LH_out = f_tilde[:, :n, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
        f_HL_out = f_tilde[:, n:2*n, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
        f_HH_out = f_tilde[:, 2*n:, :].reshape(B, h, w, C).permute(0, 3, 1, 2)

        return f_LH_out, f_HL_out, f_HH_out


# ---------------------------------------------------------------------------
# Complete FADA Block  (Fig. 2)
# ---------------------------------------------------------------------------

class FADABlock(nn.Module):
    """
    Complete FADA block for one encoder stage.

    Pipeline (Fig. 2):
        Feature → Haar DWT →  [LL → LowFreqBranch]
                              +[LH, HL, HH → HighFreqBranch]
                           → Inverse DWT → Adapted Feature
    """

    def __init__(self, channels, token_length=100, rank=16):
        """
        Args:
            channels: Number of feature channels at this encoder stage
            token_length: Base token length m (paper default: 100)
            rank: LoRA rank r (paper default: 16)
        """
        super().__init__()
        self.haar = HaarWaveletTransform2D()
        self.low_freq_branch = LowFrequencyBranch(channels, token_length, rank)
        self.high_freq_branch = HighFrequencyBranch(channels, token_length, rank)

    def forward(self, x):
        """
        Args:
            x: Feature map [B, C, H, W] from a frozen encoder stage

        Returns:
            Frequency-adapted feature map [B, C, H, W]
        """
        # Step 1: Haar wavelet decomposition (Sec. 4.1)
        LL, LH, HL, HH, orig_size = self.haar.forward(x)

        # Step 2: Low-frequency adaptation — content stabilisation (Sec. 4.2)
        LL_adapted = self.low_freq_branch(LL)

        # Step 3: High-frequency adaptation — style removal (Sec. 4.3)
        LH_adapted, HL_adapted, HH_adapted = self.high_freq_branch(LH, HL, HH)

        # Step 4: Inverse Haar wavelet reconstruction
        out = self.haar.inverse(LL_adapted, LH_adapted, HL_adapted, HH_adapted,
                                orig_size)

        return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing FADA modules...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test Haar wavelet transform
    print("\n--- Haar Wavelet Transform ---")
    x = torch.randn(2, 256, 96, 96, device=device)
    haar = HaarWaveletTransform2D()
    LL, LH, HL, HH, orig_size = haar.forward(x)
    print(f"Input:  {x.shape}")
    print(f"LL:     {LL.shape}  LH: {LH.shape}  HL: {HL.shape}  HH: {HH.shape}")
    x_recon = haar.inverse(LL, LH, HL, HH, orig_size)
    print(f"Recon:  {x_recon.shape}")
    recon_err = (x - x_recon).abs().max().item()
    print(f"Max reconstruction error: {recon_err:.2e} (should be ~0)")

    # Test with odd dimensions
    x_odd = torch.randn(2, 128, 97, 95, device=device)
    LL_o, LH_o, HL_o, HH_o, orig_o = haar.forward(x_odd)
    x_odd_recon = haar.inverse(LL_o, LH_o, HL_o, HH_o, orig_o)
    print(f"\nOdd input: {x_odd.shape} → recon: {x_odd_recon.shape}")
    recon_err_odd = (x_odd - x_odd_recon).abs().max().item()
    print(f"Max reconstruction error (odd): {recon_err_odd:.2e}")

    # Test complete FADA block
    print("\n--- FADA Block ---")
    for ch, spatial in [(256, 96), (512, 48), (512, 24)]:
        block = FADABlock(channels=ch, token_length=100, rank=16).to(device)
        x_test = torch.randn(2, ch, spatial, spatial, device=device)
        out = block(x_test)
        n_params = sum(p.numel() for p in block.parameters())
        print(f"  ch={ch:3d}, spatial={spatial:2d}×{spatial:2d}: "
              f"in={x_test.shape} → out={out.shape}, "
              f"params={n_params:,}")

    # Total FADA parameter count
    total_fada = 0
    for ch in [256, 512, 512]:
        b = FADABlock(channels=ch, token_length=100, rank=16)
        total_fada += sum(p.numel() for p in b.parameters())
    print(f"\nTotal FADA parameters (3 blocks): {total_fada:,}")