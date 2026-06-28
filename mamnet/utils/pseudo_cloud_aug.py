"""
Pseudo-Cloud Augmentation for Aerial Imagery
WACV 2023 - Tang et al.

Simulates cloud coverage and shadows to improve robustness of aerial shadow detection.
"""

import torch
import numpy as np
import cv2
from typing import Tuple


class PseudoCloudAugmentation:
    """
    Generate pseudo-cloud effects on aerial images.
    
    Creates realistic cloud-like patterns using:
    - Perlin noise for cloud texture
    - Gaussian blur for soft edges
    - Random opacity and size
    """
    
    def __init__(self, p=0.3, cloud_opacity_range=(0.3, 0.7), 
                 cloud_size_range=(0.2, 0.5), num_clouds_range=(1, 3)):
        """
        Args:
            p: Probability of applying augmentation
            cloud_opacity_range: Range for cloud opacity (0=transparent, 1=opaque)
            cloud_size_range: Range for cloud size as fraction of image size
            num_clouds_range: Range for number of clouds to generate
        """
        self.p = p
        self.cloud_opacity_range = cloud_opacity_range
        self.cloud_size_range = cloud_size_range
        self.num_clouds_range = num_clouds_range
    
    def generate_cloud_mask(self, height, width):
        """
        Generate a single cloud mask using Perlin-like noise.
        
        Args:
            height: Image height
            width: Image width
        
        Returns:
            Cloud mask [H, W] with values in [0, 1]
        """
        # Create random blob using Gaussian blobs
        cloud = np.zeros((height, width), dtype=np.float32)
        
        # Random cloud center
        cy = np.random.randint(0, height)
        cx = np.random.randint(0, width)
        
        # Random cloud size
        cloud_size = np.random.uniform(*self.cloud_size_range)
        radius_y = int(height * cloud_size)
        radius_x = int(width * cloud_size)
        
        # Create elliptical cloud
        y, x = np.ogrid[:height, :width]
        dist = ((x - cx) / radius_x) ** 2 + ((y - cy) / radius_y) ** 2
        cloud[dist <= 1] = 1.0
        
        # Apply Gaussian blur for soft edges
        kernel_size = max(3, int(min(radius_y, radius_x) * 0.5))
        if kernel_size % 2 == 0:
            kernel_size += 1
        cloud = cv2.GaussianBlur(cloud, (kernel_size, kernel_size), 0)
        
        # Normalize to [0, 1]
        if cloud.max() > 0:
            cloud = cloud / cloud.max()
        
        return cloud
    
    def apply_cloud_effect(self, image, cloud_mask, opacity):
        """
        Apply cloud effect to image.
        
        Args:
            image: RGB image [H, W, 3] in range [0, 255]
            cloud_mask: Cloud mask [H, W] in range [0, 1]
            opacity: Cloud opacity in range [0, 1]
        
        Returns:
            Image with cloud effect [H, W, 3]
        """
        # Convert to float
        image = image.astype(np.float32)
        
        # Create white cloud color with slight variation
        cloud_color = np.random.uniform(200, 255)
        
        # Blend image with cloud color based on mask and opacity
        cloud_mask_3d = cloud_mask[:, :, np.newaxis]
        cloud_mask_3d = cloud_mask_3d * opacity
        
        clouded_image = image * (1 - cloud_mask_3d) + cloud_color * cloud_mask_3d
        clouded_image = np.clip(clouded_image, 0, 255).astype(np.uint8)
        
        return clouded_image
    
    def __call__(self, image):
        """
        Apply pseudo-cloud augmentation to image.
        
        Args:
            image: RGB image as numpy array [H, W, 3] in range [0, 255]
        
        Returns:
            Augmented image [H, W, 3]
        """
        if np.random.random() > self.p:
            return image
        
        height, width = image.shape[:2]
        
        # Number of clouds to generate
        num_clouds = np.random.randint(*self.num_clouds_range)
        
        # Generate and apply multiple clouds
        result = image.copy()
        for _ in range(num_clouds):
            # Generate cloud mask
            cloud_mask = self.generate_cloud_mask(height, width)
            
            # Random opacity
            opacity = np.random.uniform(*self.cloud_opacity_range)
            
            # Apply cloud effect
            result = self.apply_cloud_effect(result, cloud_mask, opacity)
        
        return result


class CloudShadowAugmentation:
    """
    Generate cloud shadow effects (darker regions).
    """
    
    def __init__(self, p=0.2, shadow_intensity_range=(0.4, 0.7)):
        """
        Args:
            p: Probability of applying augmentation
            shadow_intensity_range: Range for shadow intensity (0=black, 1=no change)
        """
        self.p = p
        self.shadow_intensity_range = shadow_intensity_range
    
    def __call__(self, image):
        """
        Apply cloud shadow augmentation.
        
        Args:
            image: RGB image [H, W, 3] in range [0, 255]
        
        Returns:
            Image with cloud shadows [H, W, 3]
        """
        if np.random.random() > self.p:
            return image
        
        height, width = image.shape[:2]
        
        # Generate shadow mask (similar to cloud but darker)
        shadow = np.zeros((height, width), dtype=np.float32)
        
        # Random shadow center and size
        cy = np.random.randint(0, height)
        cx = np.random.randint(0, width)
        radius = int(min(height, width) * np.random.uniform(0.3, 0.6))
        
        y, x = np.ogrid[:height, :width]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        shadow[dist <= radius] = 1.0
        
        # Blur
        kernel_size = max(3, int(radius * 0.3))
        if kernel_size % 2 == 0:
            kernel_size += 1
        shadow = cv2.GaussianBlur(shadow, (kernel_size, kernel_size), 0)
        
        # Normalize
        if shadow.max() > 0:
            shadow = shadow / shadow.max()
        
        # Apply shadow (darken)
        intensity = np.random.uniform(*self.shadow_intensity_range)
        shadow_mask = 1 - (shadow * (1 - intensity))
        shadow_mask_3d = shadow_mask[:, :, np.newaxis]
        
        result = (image * shadow_mask_3d).astype(np.uint8)
        
        return result


class CombinedAerialAugmentation:
    """
    Combined augmentation pipeline for aerial imagery.
    Applies pseudo-clouds and cloud shadows.
    """
    
    def __init__(self, cloud_p=0.3, shadow_p=0.2):
        """
        Args:
            cloud_p: Probability of applying cloud augmentation
            shadow_p: Probability of applying cloud shadow augmentation
        """
        self.cloud_aug = PseudoCloudAugmentation(p=cloud_p)
        self.shadow_aug = CloudShadowAugmentation(p=shadow_p)
    
    def __call__(self, image):
        """
        Apply combined augmentation.
        
        Args:
            image: RGB image [H, W, 3] in range [0, 255]
        
        Returns:
            Augmented image [H, W, 3]
        """
        # Apply cloud augmentation
        image = self.cloud_aug(image)
        
        # Apply shadow augmentation
        image = self.shadow_aug(image)
        
        return image


if __name__ == "__main__":
    # Test pseudo-cloud augmentation
    import matplotlib.pyplot as plt
    
    # Create test image
    test_img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    
    # Apply augmentations
    cloud_aug = PseudoCloudAugmentation(p=1.0)
    shadow_aug = CloudShadowAugmentation(p=1.0)
    combined_aug = CombinedAerialAugmentation(cloud_p=1.0, shadow_p=1.0)
    
    img_cloud = cloud_aug(test_img.copy())
    img_shadow = shadow_aug(test_img.copy())
    img_combined = combined_aug(test_img.copy())
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_cloud)
    axes[0, 1].set_title('Pseudo-Cloud')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(img_shadow)
    axes[1, 0].set_title('Cloud Shadow')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(img_combined)
    axes[1, 1].set_title('Combined')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('pseudo_cloud_test.png', dpi=150, bbox_inches='tight')
    print("Pseudo-cloud augmentation test saved to pseudo_cloud_test.png")