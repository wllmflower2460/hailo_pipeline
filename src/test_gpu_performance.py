#!/usr/bin/env python3
"""
GPU Performance Testing for Dog Training Pipeline
This script tests the AMD GPU performance and configuration.
"""

import os
import time
import subprocess
import sys
import platform

def print_section(title):
    """Print a section title with dividers."""
    divider = "=" * 60
    print(f"\n{divider}")
    print(f"{title}")
    print(f"{divider}")

def run_command(cmd):
    """Run a shell command and print the output."""
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return output.strip()
    except subprocess.CalledProcessError as e:
        return f"Error running command: {e.output}"

def check_system_info():
    """Check system information."""
    print_section("System Information")
    
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    
    cpu_info = run_command("lscpu | grep 'Model name'")
    print(f"CPU: {cpu_info}")
    
    ram_info = run_command("free -h | grep Mem")
    print(f"RAM: {ram_info}")
    
    gpu_info = run_command("lspci | grep -i 'vga\\|3d\\|display'")
    print(f"GPU (PCI): {gpu_info}")

def check_rocm_info():
    """Check ROCm information."""
    print_section("ROCm Information")
    
    rocm_version = run_command("ls -la /opt | grep rocm")
    print(f"ROCm installation: {rocm_version}")
    
    print("\nROCm environment variables:")
    for var in ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_ENABLE_SDMA"]:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    print("\nROCm GPU detection:")
    rocm_smi = run_command("which rocm-smi && rocm-smi || echo 'rocm-smi not found'")
    print(rocm_smi)

def check_pytorch_gpu():
    """Check PyTorch GPU support."""
    print_section("PyTorch GPU Support")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"GPU available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("No GPU detected by PyTorch")
    except ImportError:
        print("PyTorch not installed")

def run_gpu_benchmark():
    """Run a simple GPU benchmark."""
    print_section("GPU Performance Benchmark")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("GPU not available for benchmarking")
            return
        
        # Warm up
        x = torch.randn(5000, 5000, device="cuda")
        y = torch.randn(5000, 5000, device="cuda")
        torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # Benchmark
        print("Running matrix multiplication benchmark (5000x5000)...")
        iterations = 10
        start_time = time.time()
        
        for _ in range(iterations):
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            
        end_time = time.time()
        avg_time = (end_time - start_time) / iterations
        
        print(f"Average time per multiplication: {avg_time:.4f} seconds")
        print(f"Performance: {(5000 * 5000 * 5000 * 2) / (avg_time * 1e9):.2f} TFLOPS")
        
        # Memory usage
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    except ImportError:
        print("PyTorch not installed")
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")

def main():
    """Main function."""
    check_system_info()
    check_rocm_info()
    check_pytorch_gpu()
    run_gpu_benchmark()

if __name__ == "__main__":
    main()
