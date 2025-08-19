#!/usr/bin/env python3
"""
Test script for dog detection
"""

import argparse
import cv2
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Parse arguments
parser = argparse.ArgumentParser(description='Run dog detection with Hailo')
parser.add_argument('--image', type=str, help='Path to image file')
parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold (default: 0.5)')
parser.add_argument('--device', type=str, default='CPU', help='Device to run inference on (default: CPU)')
args = parser.parse_args()

# Get image path
image_path = args.image
if not image_path:
    data_dir = Path(__file__).parent.parent / "data" / "sample"
    image_path = str(data_dir / "sample_dog.jpg")
    print(f"No image provided, using default: {image_path}")

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    sys.exit(1)

# Read image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image {image_path}")
    sys.exit(1)

print(f"Successfully loaded image: {image_path}")
print(f"Image dimensions: {img.shape}")

# Create output directory
output_dir = Path(__file__).parent.parent / "results" / "visualizations"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "dog_detection_test.jpg"

# Save a copy of the image to verify
cv2.imwrite(str(output_path), img)
print(f"Saved test image to {output_path}")
print("Test completed successfully!")
