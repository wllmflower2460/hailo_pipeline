#!/usr/bin/env python3
"""
Example Hailo integration for dog detection
This script demonstrates how to use Hailo for inference
"""

import os
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Check if hailo_platform is available
try:
    import hailo
    HAILO_AVAILABLE = True
except ImportError:
    print("Hailo package not found. Running in simulation mode.")
    HAILO_AVAILABLE = False

def detect_dog_with_hailo(image_path, hailo_device=None):
    """Detect dogs in an image using Hailo."""
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Resize image for model input (example size, adjust based on your model)
    input_size = (640, 640)
    img_resized = cv2.resize(img, input_size)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize image (example normalization, adjust based on your model)
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # Prepare input tensor
    input_tensor = np.expand_dims(img_norm, axis=0)
    
    # Simulate Hailo inference if not available
    if not HAILO_AVAILABLE or hailo_device is None:
        print("Simulating Hailo inference...")
        time.sleep(1)  # Simulate processing time
        
        # Dummy detection results
        boxes = [[100, 150, 300, 400], [200, 250, 350, 450]]  # [x1, y1, x2, y2]
        scores = [0.92, 0.85]
        classes = [16, 16]  # Class 16 is dog in COCO
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'image': img
        }
    
    else:
        print("Running inference on Hailo device...")
        try:
            # The actual Hailo inference code would be here
            # This is a placeholder for the actual implementation
            
            # Example with hailo_platform API (placeholder code)
            # Note: The actual API might differ - refer to Hailo docs
            hailo_network = hailo.load_network("path/to/hailo/model")
            outputs = hailo_network.infer(input_tensor)
            
            # Parse outputs (example parsing, adjust based on your model)
            boxes = []
            scores = []
            classes = []
            
            # Process outputs to get detections
            # ...
            
            return {
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
                'image': img
            }
            
        except Exception as e:
            print(f"Hailo inference error: {e}")
            return None

def visualize_results(results):
    """Visualize detection results."""
    if results is None:
        return None
    
    # Get image and detection results
    img = results['image'].copy()
    boxes = results['boxes']
    scores = results['scores']
    classes = results['classes']
    
    # COCO class names (simplified list with focus on animals)
    class_names = {
        1: 'person', 16: 'dog', 17: 'cat', 18: 'horse', 
        19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear'
    }
    
    # Colors for different classes
    colors = {
        1: (0, 255, 0),    # person: green
        16: (0, 0, 255),   # dog: red
        17: (255, 0, 0),   # cat: blue
        18: (255, 255, 0), # horse: cyan
        19: (0, 255, 255), # sheep: yellow
        20: (255, 0, 255), # cow: magenta
        21: (128, 128, 0), # elephant: olive
        22: (0, 128, 128)  # bear: teal
    }
    
    # Draw bounding boxes
    for box, score, cls in zip(boxes, scores, classes):
        # Skip low confidence detections
        if score < 0.5:
            continue
        
        # Get coordinates
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Get class name and color
        class_name = class_names.get(cls, f"class_{cls}")
        color = colors.get(cls, (255, 255, 255))  # Default to white
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{class_name}: {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

def main():
    """Main function."""
    # Get sample image path
    sample_dir = Path("/app/data/sample")
    sample_image = sample_dir / "sample_dog.jpg"
    
    # Check if sample image exists
    if not sample_image.exists():
        print(f"Sample image not found at {sample_image}")
        print("Creating sample directory...")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Download a sample image
        import urllib.request
        print("Downloading a sample dog image...")
        url = "https://images.dog.ceo/breeds/retriever-golden/n02099601_7130.jpg"
        urllib.request.urlretrieve(url, sample_image)
        print(f"Image downloaded to {sample_image}")
    
    # Initialize Hailo device if available
    hailo_device = None
    if HAILO_AVAILABLE:
        try:
            # This is a placeholder for actual Hailo device initialization
            # hailo_device = hailo.DeviceOptions().device()
            pass
        except Exception as e:
            print(f"Error initializing Hailo device: {e}")
    
    # Run detection
    print(f"Running dog detection on image: {sample_image}")
    results = detect_dog_with_hailo(str(sample_image), hailo_device)
    
    # Visualize results
    if results:
        output_image = visualize_results(results)
        
        # Save output image
        output_path = "/app/results/hailo_detection_result.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, output_image)
        print(f"Detection results saved to {output_path}")
    else:
        print("Detection failed")

if __name__ == "__main__":
    main()
