#!/usr/bin/env python3
"""
CPU Performance Test Script for Dog Training Pipeline.
This script benchmarks various ML operations using CPU-only PyTorch.
"""

import os
import time
import torch
import numpy as np
import platform
import psutil
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def get_system_info():
    """Get system information."""
    print_header("System Information")
    
    print(f"OS: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU: {platform.processor()}")
    
    # Get more detailed CPU info
    try:
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            print(f"CPU frequency: current={cpu_freq.current:.1f}MHz, "
                  f"min={cpu_freq.min:.1f}MHz, max={cpu_freq.max:.1f}MHz")
    except Exception:
        pass
    
    print(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
    print(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Thread settings
    print("\nThread settings:")
    for var in ['MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

def test_matrix_operations(sizes=[(1000, 1000), (2000, 2000), (4000, 4000)]):
    """Test basic matrix operations."""
    print_header("Matrix Operations Benchmark")
    
    results = []
    
    for size in sizes:
        # Create matrices
        a = torch.randn(*size)
        b = torch.randn(*size)
        
        # Multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        elapsed = time.time() - start_time
        
        # Print results
        n, m = size
        print(f"Matrix multiplication {n}x{m}: {elapsed:.4f} seconds")
        results.append((f"{n}x{m}", elapsed))
    
    return results

def test_neural_network_forward(batch_sizes=[1, 8, 16, 32]):
    """Test neural network forward pass."""
    print_header("Neural Network Forward Pass Benchmark")
    
    # Load a pretrained model
    model = resnet50(pretrained=False)
    model.eval()
    
    results = []
    
    for batch_size in batch_sizes:
        # Create random input
        input_tensor = torch.randn(batch_size, 3, 224, 224)
        
        # Time forward pass
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        elapsed = time.time() - start_time
        
        # Print results
        print(f"ResNet-50 forward pass (batch size {batch_size}): {elapsed:.4f} seconds")
        print(f"Images per second: {batch_size / elapsed:.1f}")
        results.append((f"batch-{batch_size}", elapsed))
    
    return results

def test_image_processing(num_iterations=100):
    """Test image processing operations."""
    print_header("Image Processing Benchmark")
    
    # Create a random image
    img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    pil_img = transforms.ToPILImage()(img)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Time the transformations
    start_time = time.time()
    for _ in range(num_iterations):
        transformed = transform(pil_img)
    elapsed = time.time() - start_time
    
    print(f"Image transformations ({num_iterations} iterations): {elapsed:.4f} seconds")
    print(f"Average time per image: {elapsed / num_iterations * 1000:.2f} ms")
    
    return [("image_transform", elapsed / num_iterations)]

def plot_results(results_dict):
    """Plot benchmark results."""
    print_header("Benchmark Visualization")
    
    plt.figure(figsize=(12, 8))
    
    for idx, (test_name, results) in enumerate(results_dict.items()):
        labels, times = zip(*results)
        x = range(len(labels))
        plt.subplot(len(results_dict), 1, idx+1)
        plt.bar(x, times)
        plt.xticks(x, labels)
        plt.ylabel('Time (s)')
        plt.title(test_name)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = 'cpu_benchmark_results.png'
    plt.savefig(plot_path)
    print(f"Benchmark results saved to {plot_path}")
    
    # Try to display the plot
    try:
        plt.show()
    except Exception:
        pass

def main():
    """Main function."""
    print("CPU Performance Test for Dog Training Pipeline")
    print("="*50)
    
    get_system_info()
    
    # Run benchmarks
    matrix_results = test_matrix_operations()
    nn_results = test_neural_network_forward()
    img_results = test_image_processing()
    
    # Plot results
    results_dict = {
        "Matrix Multiplication": matrix_results,
        "ResNet-50 Forward Pass": nn_results,
        "Image Processing": img_results
    }
    
    try:
        plot_results(results_dict)
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    # Print summary
    print_header("Performance Summary")
    print("Your CPU-based setup is ready for ML tasks.")
    print("Based on the benchmarks:")
    
    if nn_results[1][1] < 1.0:  # Check if batch-8 is faster than 1 second
        print("✓ Your system has good neural network inference performance")
    else:
        print("⚠️ Neural network inference is a bit slow, consider optimization")
    
    if matrix_results[1][1] < 1.0:  # 2000x2000 matrix multiplication
        print("✓ Your system has good matrix operation performance")
    else:
        print("⚠️ Matrix operations are a bit slow, check BLAS configuration")
    
    avg_img_time = img_results[0][1] * 1000  # Convert to ms
    if avg_img_time < 10:
        print("✓ Your system has excellent image processing performance")
    else:
        print("⚠️ Image processing could be faster, check PIL/OpenCV configuration")
        
    print("\nOverall, your CPU-based setup should be sufficient for most development tasks.")
    print("For production or large dataset processing, you may want to consider:")
    print("1. Optimizing batch sizes and thread settings")
    print("2. Using model quantization techniques")
    print("3. Distributed computing for large workloads")

if __name__ == "__main__":
    main()
