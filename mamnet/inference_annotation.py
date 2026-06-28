"""
Inference Script for Annotation Sessions
Runs trained shadow detection models on organized annotation data
"""

import os
import argparse
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet import MAMNet
from utils.postprocessing import filter_small_predictions


def get_args():
    parser = argparse.ArgumentParser(description='Run inference on annotation session')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True,
                      help='Directory containing images to process')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for masks')
    parser.add_argument('--city', type=str, required=True,
                      help='City name (chicago/miami)')
    parser.add_argument('--resolution', type=str, required=True,
                      help='Resolution (highres/midres)')
    parser.add_argument('--session_num', type=int, required=True,
                      help='Session number')
    parser.add_argument('--img_size', type=int, default=384,
                      help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device (cuda/cpu)')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f'Loading model from {checkpoint_path}')
    
    model = MAMNet(num_classes=2, pretrained=False, use_aux=True).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def get_transform(img_size):
    """Get image transformation pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def run_inference(model, image_dir, output_dir, transform, device):
    """
    Run inference on all images in directory and save masks
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PNG images
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    if len(image_paths) == 0:
        print(f'No images found in {image_dir}')
        return
    
    print(f'Found {len(image_paths)} images to process')
    
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc='Processing images'):
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Get original filename (e.g., img_001.png)
            original_filename = os.path.basename(image_path)
            
            # Create mask filename (e.g., mask_img_001.png)
            mask_filename = f'mask_{original_filename}'
            mask_path = os.path.join(output_dir, mask_filename)
            
            # Transform image
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Forward pass
            output = model(image_tensor)

            # Filter small predictions
            output = filter_small_predictions(output, min_pixels=10)  # [1, 2, H, W]
            
            # Get prediction (threshold at 0.5)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            
            # Save mask
            mask_image = Image.fromarray(pred_binary, mode='L')
            mask_image.save(mask_path)
    
    print(f'Saved {len(image_paths)} masks to {output_dir}')


def main():
    args = get_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('\n' + '='*80)
    print(f'Running Inference on Annotation Session {args.session_num:02d}')
    print(f'City: {args.city} | Resolution: {args.resolution}')
    print('='*80)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Get transform
    transform = get_transform(args.img_size)
    
    # Check if image directory exists
    if not os.path.exists(args.image_dir):
        print(f'ERROR: Image directory not found: {args.image_dir}')
        return
    
    # Run inference
    run_inference(model, args.image_dir, args.output_dir, transform, device)
    
    print('\n' + '='*80)
    print('Inference completed!')
    print(f'Masks saved to: {args.output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()