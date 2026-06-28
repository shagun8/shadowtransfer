"""
Contrast computation utilities for shadow detection.
Implements RMS (Root Mean Square) contrast calculation.
"""

import numpy as np
import cv2
import torch
from scipy.ndimage import uniform_filter


def compute_rms_contrast(image, window_size=7):
    """
    Compute RMS (Root Mean Square) contrast for an image.
    
    RMS contrast = sqrt(mean((I - mean(I))^2))
    
    Computed locally using a sliding window.
    
    Args:
        image: Input image, can be:
            - PIL Image
            - numpy array [H, W, 3] (RGB) or [H, W] (grayscale)
            - torch tensor [3, H, W] or [H, W]
        window_size: Size of local window for contrast computation (default: 7)
        
    Returns:
        Contrast map as numpy array [H, W] normalized to [0, 1]
    """
    # Convert to numpy grayscale
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            # [3, H, W] -> [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
    
    if hasattr(image, 'convert'):  # PIL Image
        image = np.array(image.convert('L'))
    elif image.ndim == 3:  # RGB
        # Convert to grayscale: 0.299*R + 0.587*G + 0.114*B
        image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # Normalize to [0, 1]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    
    # Compute local mean using uniform filter
    local_mean = uniform_filter(image, size=window_size, mode='reflect')
    
    # Compute local variance
    local_var = uniform_filter(image ** 2, size=window_size, mode='reflect') - local_mean ** 2
    
    # RMS contrast = sqrt(variance)
    rms_contrast = np.sqrt(np.maximum(local_var, 0))  # Ensure non-negative
    
    # Normalize to [0, 1]
    if rms_contrast.max() > 0:
        rms_contrast = rms_contrast / rms_contrast.max()
    
    return rms_contrast


def compute_shadow_contrast(mask, image):
    """
    Compute contrast between shadow region and its surrounding area.
    
    This is the brightness_contrast metric used in the analysis:
    contrast = |mean(shadow) - mean(surround)| / (mean(shadow) + mean(surround))
    
    Args:
        mask: Binary mask [H, W] with values {0, 1}
        image: RGB image [H, W, 3] or grayscale [H, W]
        
    Returns:
        Contrast value (scalar)
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    
    # Normalize
    if gray.max() > 1.0:
        gray = gray / 255.0
    
    # Get shadow region
    shadow_mask = mask > 0.5
    
    # Get surrounding region (dilate by 10 pixels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    surround_mask = (dilated > 0.5) & (~shadow_mask)
    
    if not shadow_mask.any() or not surround_mask.any():
        return 0.0
    
    # Compute means
    shadow_mean = gray[shadow_mask].mean()
    surround_mean = gray[surround_mask].mean()
    
    # Compute contrast
    if shadow_mean + surround_mean == 0:
        return 0.0
    
    contrast = abs(shadow_mean - surround_mean) / (shadow_mean + surround_mean)
    
    return float(contrast)


def add_contrast_channel(image):
    """
    Add RMS contrast as a 4th channel to RGB image.
    
    Args:
        image: PIL Image or numpy array [H, W, 3]
        
    Returns:
        numpy array [H, W, 4] with RGBC channels
    """
    # Convert to numpy if PIL
    if hasattr(image, 'convert'):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Compute contrast
    contrast = compute_rms_contrast(image_np, window_size=7)
    
    # Stack as 4th channel
    contrast_expanded = contrast[:, :, np.newaxis]
    rgbc = np.concatenate([image_np, contrast_expanded * 255], axis=2)
    
    return rgbc.astype(np.uint8)


if __name__ == "__main__":
    # Test contrast computation
    import matplotlib.pyplot as plt
    
    # Create test image with gradient
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    test_img[:128, :, :] = 50   # Dark region
    test_img[128:, :, :] = 200  # Bright region
    
    # Add some texture
    noise = np.random.randint(-20, 20, (256, 256, 3))
    test_img = np.clip(test_img + noise, 0, 255).astype(np.uint8)
    
    # Compute contrast
    contrast = compute_rms_contrast(test_img, window_size=7)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(contrast, cmap='hot')
    axes[1].set_title('RMS Contrast Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('./contrast_test.png', dpi=150, bbox_inches='tight')
    print("Test visualization saved to ./contrast_test.png")
    print(f"Contrast range: [{contrast.min():.3f}, {contrast.max():.3f}]")