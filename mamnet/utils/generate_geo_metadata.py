"""
Helper script to generate geocoordinate metadata JSON file.
Creates mapping.json with default coordinates per city.
"""

import json
import os
import argparse


# Default city coordinates (city centers)
CITY_COORDS = {
    'phoenix': {'lat': 33.4484, 'lon': -112.0740},
    'miami': {'lat': 25.7617, 'lon': -80.1918},
    'chicago': {'lat': 41.8781, 'lon': -87.6298}
}


def generate_metadata(data_root, output_path, city_name=None):
    """
    Generate geocoordinate metadata JSON for all images in a dataset.
    
    Args:
        data_root: Root directory of dataset (containing train/val/test folders)
        output_path: Path to save mapping.json
        city_name: Name of city (to use default coordinates)
    """
    metadata = {}
    
    # Try to infer city from path if not provided
    if city_name is None:
        for city in CITY_COORDS.keys():
            if city.lower() in data_root.lower():
                city_name = city
                break
    
    if city_name is None:
        print("Warning: Could not infer city name, using Phoenix coordinates as default")
        city_name = 'phoenix'
    
    coords = CITY_COORDS[city_name.lower()]
    print(f"Using coordinates for {city_name}: lat={coords['lat']}, lon={coords['lon']}")
    
    # Collect all image files
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(data_root, split, 'images')
        if not os.path.exists(img_dir):
            continue
        
        for filename in os.listdir(img_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                # For now, assign same coordinates to all images in a city
                # In practice, you might have different coords per image
                metadata[filename] = {
                    'lat': coords['lat'],
                    'lon': coords['lon']
                }
    
    # Save metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated metadata for {len(metadata)} images")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate geocoordinate metadata')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of dataset')
    parser.add_argument('--output_path', type=str, default='mapping.json',
                       help='Output path for metadata JSON')
    parser.add_argument('--city', type=str, default=None,
                       choices=['phoenix', 'miami', 'chicago'],
                       help='City name (will auto-detect from path if not provided)')
    
    args = parser.parse_args()
    
    generate_metadata(args.data_root, args.output_path, args.city)