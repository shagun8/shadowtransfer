"""
Generate a single sample split and save to JSON
Separates sampling from training for reproducibility
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('/mnt/user-data/outputs')

from utils.spatial_sampling import select_patches_by_strategy


def get_args():
    parser = argparse.ArgumentParser(description='Generate sample split for fine-tuning')
    
    parser.add_argument('--city', type=str, required=True,
                       choices=['chicago', 'miami', 'phoenix'])
    parser.add_argument('--resolution', type=str, required=True,
                       choices=['highres', 'midres'])
    parser.add_argument('--n_samples', type=int, required=True)
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['random', 'clustered', 'dispersed', 'original'])
    parser.add_argument('--random_seed', type=int, required=True)
    
    parser.add_argument('--base_data_root', type=str,
                       default=os.path.join(os.environ["PROJECT_ROOT"], 'data', 'Final_data_test') + os.sep)
    parser.add_argument('--metadata_dir', type=str,
                       default=os.path.join(os.environ["PROJECT_ROOT"], 'data', 'Final_data_test', 'metadata') + os.sep)
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.environ["PROJECT_ROOT"], 'data', 'mamnet', 'splits') + os.sep)
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualization directory
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating Split:")
    print(f"  City: {args.city}")
    print(f"  Resolution: {args.resolution}")
    print(f"  Strategy: {args.strategy}")
    print(f"  N: {args.n_samples}")
    print(f"  Seed: {args.random_seed}")
    print(f"{'='*60}\n")
    
    # Generate split filename
    # N=600 uses no seed suffix (shared across all strategies/seeds)
    if args.n_samples == 600:
        split_filename = f"{args.city}_{args.resolution}_{args.strategy}_N{args.n_samples:03d}.json"
    else:
        split_filename = f"{args.city}_{args.resolution}_{args.strategy}_N{args.n_samples:03d}_seed{args.random_seed}.json"
    split_path = os.path.join(args.output_dir, split_filename)
    
    # Check if already exists
    if os.path.exists(split_path):
        print(f"Split already exists: {split_path}")
        print("Skipping generation.")
        return
    
    # Visualization path
    if args.n_samples == 600:
        viz_filename = f"{args.city}_{args.resolution}_{args.strategy}_N{args.n_samples:03d}.png"
    else:
        viz_filename = f"{args.city}_{args.resolution}_{args.strategy}_N{args.n_samples:03d}_seed{args.random_seed}.png"
    viz_path = os.path.join(viz_dir, viz_filename)
    
    # Generate split
    try:
        if args.n_samples == 0:
            # N=0: No fine-tuning, just store metadata
            result = {
                'city': args.city,
                'resolution': args.resolution,
                'strategy': args.strategy,
                'n_samples': 0,
                'random_seed': args.random_seed,
                'train_filenames': [],
                'val_filenames': [],
                'spatial_metrics': {
                    'note': 'No fine-tuning (N=0)' if args.n_samples == 0 else 'Original split (N=600)'
                },
                'visualization_path': None,
                'generated_at': datetime.now().isoformat()
            }
        else:
            # Generate actual split
            selection_result = select_patches_by_strategy(
                city=args.city,
                resolution=args.resolution,
                n_samples=args.n_samples,
                strategy=args.strategy,
                random_seed=args.random_seed,
                metadata_dir=args.metadata_dir,
                base_data_root=args.base_data_root,
                split_ratio=(0.75, 0.25),
                save_visualization=viz_path
            )
            
            # Package result
            result = {
                'city': args.city,
                'resolution': args.resolution,
                'strategy': args.strategy,
                'n_samples': args.n_samples,
                'random_seed': args.random_seed,
                'train_filenames': selection_result['train_filenames'],
                'val_filenames': selection_result['val_filenames'],
                'spatial_metrics': selection_result['spatial_metrics'],
                'visualization_path': viz_path if os.path.exists(viz_path) else None,
                'generated_at': datetime.now().isoformat()
            }
        
        # Save to JSON
        with open(split_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Split saved: {split_path}")
        if result.get('visualization_path'):
            print(f"Visualization: {result['visualization_path']}")
        print(f"{'='*60}\n")
        
        # Print spatial metrics
        if args.n_samples > 0:
            print("Spatial Metrics:")
            for key, value in result['spatial_metrics'].items():
                print(f"  {key}: {value}")
        
        print("\nGeneration complete!")
        
    except Exception as e:
        print(f"\nERROR generating split: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())