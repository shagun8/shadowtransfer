"""
Run inference on test images and save predictions
Supports: within-city, LOCO, and cross-resolution transfer

Usage:
    python run_inference.py --eval_type within --city chicago --target_resolution highres
    python run_inference.py --eval_type loco --city chicago --target_resolution highres
    python run_inference.py --eval_type cross_res --city chicago --source_resolution midres --target_resolution highres
    python run_inference.py --eval_type all --city chicago --target_resolution highres --source_resolution midres
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mamnet import MAMNet
from utils.utils import load_config, load_metadata, find_test_images, create_output_directories
import glob
from data.dataset import get_dataloaders
from utils.postprocessing import filter_small_predictions


def find_latest_checkpoint(checkpoints_dir: str, pattern: str) -> str:
    """Find the latest checkpoint matching the pattern"""
    folders = glob.glob(os.path.join(checkpoints_dir, pattern))
    if not folders:
        return None
    
    # Sort by timestamp
    folders.sort(key=lambda x: x.split('_')[-2] + '_' + x.split('_')[-1])
    latest = folders[-1]
    
    checkpoint_path = os.path.join(latest, 'checkpoint_best.pth')
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load MAMNet model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    model = MAMNet(num_classes=2, pretrained=False, use_aux=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def run_inference(model: nn.Module, dataloader: DataLoader, 
                 output_dir: Path, device: torch.device):
    """
    Run inference and save predictions
    """
    model.eval()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            images = batch['image'].to(device)
            # Note: Your dataloader might not have 'filename' in batch
            # We'll need to get filenames from the dataset
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            if isinstance(outputs, tuple):
                preds = outputs[0]
            else:
                preds = outputs
            
            # Filter small predictions
            preds = filter_small_predictions(preds, min_pixels=10)

            # Convert to binary
            preds_binary = torch.argmax(preds, dim=1).cpu().numpy()  # [B, H, W]
            
            # Save each prediction
            batch_size = images.size(0)
            for i in range(batch_size):
                # Get filename from dataset
                # Calculate global index
                global_idx = batch_idx * dataloader.batch_size + i
                if global_idx < len(dataloader.dataset.img_files):
                    filename = os.path.basename(dataloader.dataset.img_files[global_idx])
                    
                    pred = preds_binary[i]
                    
                    # Convert to uint8 (0 or 255)
                    pred_img = (pred * 255).astype(np.uint8)
                    
                    # Save as PNG
                    output_path = output_dir / filename
                    cv2.imwrite(str(output_path), pred_img)
                    saved_count += 1
    
    print(f"Saved {saved_count} predictions to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Run inference on test images')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--city', type=str, required=True,
                       help='City to process')
    parser.add_argument('--eval_type', type=str, required=True,
                       choices=['within', 'loco', 'cross_res', 'all'],
                       help='Type of evaluation: within, loco, cross_res, or all')
    parser.add_argument('--target_resolution', type=str, required=True,
                       choices=['highres', 'midres'],
                       help='Resolution to test on (where test images are loaded from)')
    parser.add_argument('--source_resolution', type=str, default=None,
                       choices=['highres', 'midres'],
                       help='For cross_res: resolution the model was trained on')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu), overrides config')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.eval_type == 'cross_res':
        if args.source_resolution is None:
            parser.error("--source_resolution is required when --eval_type is cross_res")
        if args.source_resolution == args.target_resolution:
            parser.error("--source_resolution and --target_resolution must be different for cross_res")
    
    if args.eval_type == 'all':
        if args.source_resolution is None:
            parser.error("--source_resolution is required when --eval_type is all (needed for cross_res)")
        # Run all three evaluations
        eval_types_to_run = ['within', 'loco', 'cross_res']
    else:
        eval_types_to_run = [args.eval_type]
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified
    if args.device is not None:
        config['inference']['device'] = args.device
    
    device = torch.device(config['inference']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = load_metadata(config['paths']['metadata_json'])
    
    # Find test images
    print(f"Finding test images for {args.city}/{args.target_resolution}...")
    test_images = find_test_images(
        config['paths']['base_data_root'],
        args.city,
        args.target_resolution,
        metadata_df,
        config['data'].get('max_images_per_city', None)
    )
    
    if len(test_images) == 0:
        print(f"No test images found for {args.city}/{args.target_resolution}")
        return
    
    print(f"Found {len(test_images)} test images at {args.target_resolution}")
    
    # For cross_res, filter to only paired images
    if 'cross_res' in eval_types_to_run:
        paired_images = [img for img in test_images if img['pair_id'] is not None]
        print(f"  Paired images (for cross-resolution): {len(paired_images)}")
        if len(paired_images) == 0:
            print(f"ERROR: No paired images found for cross-resolution evaluation!")
            return
        # Use paired images for cross_res, but keep all for within/loco
        if args.eval_type == 'cross_res':
            test_images = paired_images
    
    # Find checkpoints based on eval_type
    checkpoints_to_run = {}
    
    for eval_type in eval_types_to_run:
        if eval_type == 'within':
            # Within-city: trained and tested on same city/resolution
            pattern = config['inference']['models'][0]['within_checkpoint_pattern'].format(
                city=args.city,
                resolution=args.target_resolution
            )
            checkpoint = find_latest_checkpoint(config['paths']['checkpoints_dir'], pattern)
            
            if checkpoint:
                checkpoints_to_run['within'] = {
                    'path': checkpoint,
                    'test_resolution': args.target_resolution,
                    'test_images': test_images,
                    'save_dir': f'within'
                }
                print(f"Found within-city checkpoint: {checkpoint}")
            else:
                print(f"WARNING: No within-city checkpoint found for pattern: {pattern}")
        
        elif eval_type == 'loco':
            # LOCO: trained on other cities, tested on target city
            pattern = config['inference']['models'][0]['loco_checkpoint_pattern'].format(
                city=args.city,
                resolution=args.target_resolution
            )
            checkpoint = find_latest_checkpoint(config['paths']['checkpoints_dir'], pattern)
            
            if checkpoint:
                checkpoints_to_run['loco'] = {
                    'path': checkpoint,
                    'test_resolution': args.target_resolution,
                    'test_images': test_images,
                    'save_dir': f'loco'
                }
                print(f"Found LOCO checkpoint: {checkpoint}")
            else:
                print(f"WARNING: No LOCO checkpoint found for pattern: {pattern}")
        
        elif eval_type == 'cross_res':
            # Cross-resolution: trained on source_res, tested on target_res
            pattern = config['inference']['models'][0]['within_checkpoint_pattern'].format(
                city=args.city,
                resolution=args.source_resolution
            )
            checkpoint = find_latest_checkpoint(config['paths']['checkpoints_dir'], pattern)
            
            if checkpoint:
                # Use only paired images for cross_res
                cross_res_images = [img for img in test_images if img['pair_id'] is not None]
                
                checkpoints_to_run[f'cross_res_{args.source_resolution}_to_{args.target_resolution}'] = {
                    'path': checkpoint,
                    'test_resolution': args.target_resolution,
                    'test_images': cross_res_images,
                    'save_dir': f'cross_res_{args.source_resolution}_to_{args.target_resolution}'
                }
                print(f"Found cross-resolution checkpoint: {checkpoint}")
                print(f"  Trained on: {args.source_resolution}, Testing on: {args.target_resolution}")
                print(f"  Using {len(cross_res_images)} paired images")
            else:
                print(f"WARNING: No checkpoint found for pattern: {pattern}")
    
    if len(checkpoints_to_run) == 0:
        print(f"ERROR: No checkpoints found for {args.city}")
        return
    
    print(f"\nWill run inference for: {list(checkpoints_to_run.keys())}")
    
    # Create output directories
    output_dirs = create_output_directories(config)
    
    # Run inference for each checkpoint
    for eval_name, checkpoint_info in checkpoints_to_run.items():
        checkpoint_path = checkpoint_info['path']
        test_resolution = checkpoint_info['test_resolution']
        test_images_subset = checkpoint_info['test_images']
        save_dir = checkpoint_info['save_dir']
        
        print(f"\n{'='*60}")
        print(f"Running {eval_name.upper()} inference")
        if 'cross_res' in eval_name:
            print(f"Model trained on: {args.source_resolution}")
            print(f"Testing on: {test_resolution}")
            print(f"Images: {len(test_images_subset)} (paired only)")
        else:
            print(f"Images: {len(test_images_subset)}")
        print(f"{'='*60}")
        
        # Load model
        model = load_model(checkpoint_path, device)
        
        # Create dataloader using existing get_dataloaders function
        # This ensures proper normalization/transforms are applied
        test_data_root = os.path.join(
            config['paths']['base_data_root'],
            args.city,
            test_resolution
        )
        
        dataloaders = get_dataloaders(
            data_root=test_data_root,
            base_data_root=None,
            mode='single',
            cities=None,
            resolution=None,
            fold_id=None,
            batch_size=config['inference']['batch_size'],
            num_workers=config['inference']['num_workers'],
            img_size=config['inference']['img_size']
        )
        
        dataloader = dataloaders['test']
        
        # Verify images
        print(f"Dataloader has {len(dataloader.dataset)} images")

        # Create output directory
        pred_output_dir = output_dirs['predictions'] / save_dir / args.city / test_resolution
        
        # Run inference
        run_inference(model, dataloader, pred_output_dir, device)
        
        print(f"Saved {eval_name} predictions to: {pred_output_dir}")
    
    print(f"\n{'='*60}")
    print("All inference complete!")
    print(f"Predictions saved to: {output_dirs['predictions']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()