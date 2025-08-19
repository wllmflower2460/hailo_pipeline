#!/usr/bin/env python3
"""
PyTorch ROCm Diagnostic Script
This script performs detailed diagnostics to help troubleshoot 
AMD GPU support in PyTorch with ROCm.
"""

import os
import sys
import platform
import subprocess
import glob

def print_header(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def run_command(cmd):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def check_system():
    """Check system information."""
    print_header("System Information")
    print(f"OS: {platform.platform()}")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    cpu_info = run_command("lscpu | grep 'Model name'")
    print(f"CPU: {cpu_info}")
    
    mem_info = run_command("free -h | head -2")
    print(f"Memory:\n{mem_info}")

def check_rocm_installation():
    """Check ROCm installation status."""
    print_header("ROCm Installation")
    
    # Check if ROCm is installed
    rocm_path = run_command("ls -ld /opt/rocm 2>/dev/null || echo 'Not found'")
    print(f"ROCm path: {rocm_path}")
    
    # Check ROCm version
    rocm_version = run_command("cat /opt/rocm/.info/version 2>/dev/null || echo 'Version file not found'")
    print(f"ROCm version: {rocm_version}")
    
    # Check ROCm environment variables
    print("\nROCm environment variables:")
    rocm_vars = ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_ENABLE_SDMA", 
                 "ROC_ENABLE_PRE_VEGA", "HSA_OVERRIDE_GFX_VERSION"]
    for var in rocm_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")
    
    # Check ROCm in PATH
    path = os.environ.get("PATH", "")
    if "/opt/rocm" in path:
        print("\nROCm found in PATH")
    else:
        print("\nROCm not found in PATH")
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if "/opt/rocm" in ld_path:
        print("ROCm found in LD_LIBRARY_PATH")
    else:
        print("ROCm not found in LD_LIBRARY_PATH")

def check_gpu_detection():
    """Check if the system detects the AMD GPU."""
    print_header("GPU Detection")
    
    # Check PCI devices
    pci_info = run_command("lspci | grep -i 'vga\\|display\\|3d\\|amd'")
    print("PCI GPU devices:")
    print(pci_info)
    
    # Check kernel modules
    kernel_modules = run_command("lsmod | grep -i 'amdgpu\\|radeon'")
    print("\nAMD GPU kernel modules:")
    print(kernel_modules if kernel_modules else "No AMD GPU kernel modules found")
    
    # Check if ROCm detects the GPU
    if os.path.exists("/opt/rocm/bin/rocminfo"):
        rocminfo = run_command("/opt/rocm/bin/rocminfo | grep -i 'agent\\|gpu' | head -10")
        print("\nROCm GPU detection (rocminfo):")
        print(rocminfo if rocminfo else "No GPU detected by rocminfo")
    else:
        print("\nrocminfo not found")
    
    if os.path.exists("/opt/rocm/bin/rocm-smi"):
        rocmsmi = run_command("/opt/rocm/bin/rocm-smi | grep -i 'gpu\\|temp\\|mhz' | head -5")
        print("\nROCm SMI output:")
        print(rocmsmi if rocmsmi else "No output from rocm-smi")
    else:
        print("\nrocm-smi not found")

def check_pytorch():
    """Check PyTorch installation and GPU support."""
    print_header("PyTorch Information")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        
        # Check if PyTorch was built with ROCm
        if '+rocm' in torch.__version__:
            print("PyTorch was built with ROCm support")
        else:
            print("PyTorch was NOT built with ROCm support")
        
        # Check for CUDA availability (used for both CUDA and ROCm in PyTorch)
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            
            # Try a simple GPU operation
            print("\nAttempting GPU operation...")
            try:
                x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
                y = x + 1
                print(f"Operation successful: {y}")
            except Exception as e:
                print(f"GPU operation failed: {str(e)}")
        else:
            print("No GPU available for PyTorch")
            print("Checking CUDA/ROCm libraries...")
            
            torch_lib_dir = os.path.dirname(torch.__file__) + '/lib'
            print(f"Torch library directory: {torch_lib_dir}")
            
            hip_libs = glob.glob(f"{torch_lib_dir}/*hip*")
            print(f"HIP libraries: {hip_libs}")
    except ImportError:
        print("PyTorch is not installed")

def check_library_dependencies():
    """Check library dependencies for PyTorch with ROCm."""
    print_header("Library Dependencies")
    
    try:
        import torch
        torch_so = os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch_hip.so')
        
        if os.path.exists(torch_so):
            ldd_output = run_command(f"ldd {torch_so} | grep -i 'not found'")
            
            if ldd_output:
                print(f"Missing dependencies for {torch_so}:")
                print(ldd_output)
            else:
                print(f"No missing dependencies for {torch_so}")
                
            # Check for specific ROCm libraries
            rocm_libs = run_command(f"ldd {torch_so} | grep -i 'rocm\\|hip\\|hsa'")
            print("\nROCm libraries linked:")
            print(rocm_libs if rocm_libs else "No ROCm libraries found in dependencies")
        else:
            print(f"Cannot find libtorch_hip.so at {torch_so}")
    except ImportError:
        print("PyTorch is not installed")

def check_alternative_gpu_libs():
    """Check if other GPU libraries can detect the GPU."""
    print_header("Alternative GPU Libraries")
    
    # Try TensorFlow if available
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")
    except ImportError:
        print("TensorFlow not installed")
    
    # Try JAX if available
    try:
        import jax
        print(f"\nJAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("\nJAX not installed")

def suggest_fixes():
    """Suggest possible fixes for common problems."""
    print_header("Suggested Fixes")
    
    print("1. Check if your AMD GPU is supported by ROCm:")
    print("   - Visit: https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html")
    
    print("\n2. Ensure ROCm environment variables are set:")
    print("   export HSA_OVERRIDE_GFX_VERSION=11.0.0")
    print("   export ROC_ENABLE_PRE_VEGA=1  # For older GPUs")
    
    print("\n3. Update LD_LIBRARY_PATH:")
    print("   export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/hip/lib:$LD_LIBRARY_PATH")
    
    print("\n4. Try reinstalling PyTorch with ROCm support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6")
    
    print("\n5. For integrated AMD GPUs (like Radeon 780M):")
    print("   - Try adding HSA_ENABLE_SDMA=0 to environment variables")
    print("   - Update to latest AMD drivers")
    print("   - Some newer integrated GPUs may have limited ROCm support")
    
    print("\n6. If all else fails, consider CPU-only mode:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")

def main():
    """Main function."""
    print("PyTorch ROCm Diagnostic Tool")
    print("============================")
    
    check_system()
    check_rocm_installation()
    check_gpu_detection()
    check_pytorch()
    check_library_dependencies()
    check_alternative_gpu_libs()
    suggest_fixes()

if __name__ == "__main__":
    main()
