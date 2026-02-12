#!/usr/bin/env python3
"""Convert bubblified YAML collision spheres to JSON format.

This script converts a YAML file containing collision spheres (from bubblify)
to a JSON format compatible with pyroki, with the following transformations:
- Remove top-level 'collision_spheres' key
- Convert 'center' to 'centers' (list of centers)
- Convert 'radius' to 'radii' (list of radii)

Usage:
    python bubblify_to_json.py input.yml output.json
    python bubblify_to_json.py ur5e.urdf_spherized.yml ur5e.urdf_spherized.json
"""

import sys
import yaml
import json
from pathlib import Path


def convert_bubblify_to_json(yaml_path: str, json_path: str) -> None:
    """Convert bubblified YAML to JSON format.

    Args:
        yaml_path: Path to input YAML file
        json_path: Path to output JSON file
    """
    # Load YAML file
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Transform data
    output = {}
    collision_spheres = data.get('collision_spheres', {})

    for link_name, spheres in collision_spheres.items():
        centers = []
        radii = []

        for sphere in spheres:
            centers.append(sphere['center'])
            radii.append(sphere['radius'])

        output[link_name] = {
            'centers': centers,
            'radii': radii
        }

    # Save as JSON
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Converted {yaml_path} to {json_path}")
    print(f"Total links: {len(output)}")
    total_spheres = 0
    for link_name, link_data in output.items():
        num_spheres = len(link_data['centers'])
        total_spheres += num_spheres
        print(f"  {link_name}: {num_spheres} spheres")
    print(f"Total spheres: {total_spheres}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: No input file specified")
        sys.exit(1)

    yaml_path = sys.argv[1]

    # Default output path: same as input but with .json extension
    if len(sys.argv) >= 3:
        json_path = sys.argv[2]
    else:
        json_path = str(Path(yaml_path).with_suffix('.json'))

    # Check if input file exists
    if not Path(yaml_path).exists():
        print(f"Error: Input file '{yaml_path}' not found")
        sys.exit(1)

    # Convert
    convert_bubblify_to_json(yaml_path, json_path)


if __name__ == '__main__':
    main()
