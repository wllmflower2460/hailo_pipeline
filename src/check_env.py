#!/usr/bin/env python3
"""
Environment checker for dog training pipeline.
This script verifies that all required libraries are correctly installed
and configured, including GPU acceleration.
"""

import os
import sys
import importlib
from pathlib import Path


def check_import(module_name, package_name=None):
    """Try to import a module and report success/failure."""
    if package_name is None:
        package_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "Unknown")
        print(f"✓ {package_name} {version}")
        return module
    except ImportError:
        print(f"✗ {package_name} not found. Try: pip install {package_name}")
        return None


def check_gpu():
    """Check if GPU is available for PyTorch."""
    torch = check_import("torch")
    if torch:
        print("\nGPU Information:")
        print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"Current device: {torch.cuda.current_device()}")
        else:
            print("No CUDA/ROCm devices detected. Using CPU only.")


def check_env_vars():
    """Check if environment variables are set."""
    print("\nEnvironment Variables:")
    
    required_vars = [
        "PROJECT_ROOT",
        "DATA_DIR",
        "MODELS_DIR",
    ]
    
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var} = {value}")
        else:
            print(f"✗ {var} not set. Did you run source setup_env.sh?")


def main():
    """Main function to run environment checks."""
    print("Checking Python environment...")
    print(f"Python version: {sys.version}")
    print(f"Executable path: {sys.executable}")
    
    print("\nChecking required packages:")
    check_import("numpy")
    check_import("pandas")
    check_import("matplotlib")
    check_import("torch")
    check_import("torchvision")
    check_import("cv2", "opencv-python")
    check_import("PIL", "pillow")
    check_import("sklearn", "scikit-learn")
    check_import("ultralytics")
    
    check_gpu()
    check_env_vars()
    
    # Check for project directories
    print("\nProject Directory Structure:")
    project_root = Path(__file__).parent.absolute()
    for directory in ["data", "results", "notebooks", "src"]:
        path = project_root / directory
        if path.exists():
            print(f"✓ {path} exists")
        else:
            print(f"✗ {path} not found. Creating...")
            path.mkdir(exist_ok=True)
            print(f"  Created {path}")


if __name__ == "__main__":
    main()
