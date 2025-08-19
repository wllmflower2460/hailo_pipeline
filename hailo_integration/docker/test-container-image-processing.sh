#!/bin/bash

echo "Running dog detection test in container..."
podman exec -it hailo-dog-training python3 -c "
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Set up paths
sample_path = '/app/data/sample/sample_dog.jpg'
output_dir = '/app/results/visualizations'
output_path = os.path.join(output_dir, 'test_output.jpg')

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Check if image exists
if not os.path.exists(sample_path):
    print(f'Error: Sample image not found at {sample_path}')
    sys.exit(1)

# Read image
img = cv2.imread(sample_path)
if img is None:
    print(f'Error: Could not read image {sample_path}')
    sys.exit(1)

# Print image info
print(f'Successfully loaded image: {sample_path}')
print(f'Image dimensions: {img.shape}')

# Save a test output
try:
    cv2.imwrite(output_path, img)
    print(f'Test image saved to {output_path}')
    print('Test completed successfully!')
except Exception as e:
    print(f'Error saving output: {e}')
    # Try with different permissions
    alt_path = '/tmp/test_output.jpg'
    try:
        cv2.imwrite(alt_path, img)
        print(f'Test image saved to alternative path: {alt_path}')
    except Exception as e2:
        print(f'Error saving to alternative path: {e2}')
"
