"""
Convert detailed mapping.json to simplified mapping_segdesic.json format.
Extracts lat/lon coordinates for each image based on original_filename.
"""

import json
import argparse
import sys


def convert_mapping(input_path, output_path):
    """
    Convert detailed mapping.json to simplified coordinate mapping.
    
    Args:
        input_path: Path to input mapping.json (array of objects)
        output_path: Path to save mapping_segdesic.json
    """
    # Load input mapping
    with open(input_path, 'r') as f:
        detailed_mapping = json.load(f)
    
    if not isinstance(detailed_mapping, list):
        raise ValueError("Expected mapping.json to be an array of objects")
    
    # Convert to simplified format
    segdesic_mapping = {}
    
    for idx, entry in enumerate(detailed_mapping):
        # Check for required fields
        if 'original_filename' not in entry:
            raise KeyError(f"Entry at index {idx} missing 'original_filename'")
        
        if 'center_lat' not in entry or 'center_lon' not in entry:
            raise KeyError(f"Entry '{entry.get('original_filename', 'unknown')}' "
                         f"missing 'center_lat' or 'center_lon'")
        
        filename = entry['original_filename']
        
        # Create simplified entry
        segdesic_mapping[filename] = {
            'lat': entry['center_lat'],
            'lon': entry['center_lon']
        }
    
    # Save output
    with open(output_path, 'w') as f:
        json.dump(segdesic_mapping, f, indent=2)
    
    print(f"✓ Successfully converted {len(segdesic_mapping)} image entries")
    print(f"✓ Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert detailed mapping.json to mapping_segdesic.json format'
    )
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input mapping.json')
    parser.add_argument('--output_path', type=str, default='mapping_segdesic.json',
                       help='Output path for simplified mapping')
    
    args = parser.parse_args()
    
    try:
        convert_mapping(args.input_path, args.output_path)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)