#!/bin/bash

# Script to fix the environment for AMD Radeon 780M GPU support in PyTorch

echo "=== AMD Radeon 780M GPU Support Setup Script ==="
echo ""

# Activate the virtual environment
source .venv/bin/activate

# Update the setup_env.sh file
echo "Updating setup_env.sh with correct ROCm paths and environment variables..."
# Already done in previous step

# Install required PyTorch packages
echo "Installing PyTorch with ROCm support..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.6
pip install numpy==1.24.3  # Ensure compatible numpy version

# Source the environment script
echo "Activating the updated environment variables..."
source setup_env.sh

# Verify installation
echo "Verifying PyTorch installation and GPU detection..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Is GPU available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Create a test script for more detailed testing
echo "Creating a basic GPU test script..."
cat > test_gpu.py << 'EOL'
import torch
import time

print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Run a simple benchmark
    print("\nRunning simple benchmark...")
    
    # CPU benchmark
    start_time = time.time()
    a = torch.randn(5000, 5000)
    b = torch.randn(5000, 5000)
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # GPU benchmark
    start_time = time.time()
    a_gpu = a.cuda()
    b_gpu = b.cuda()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"GPU time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("GPU not available. Using CPU only.")
    print("\nPossible reasons:")
    print("1. Your AMD Radeon 780M may not be fully supported by ROCm")
    print("2. Environment variables may not be correctly set")
    print("3. ROCm installation may be incomplete")
    print("\nSuggested workarounds:")
    print("1. Use CPU-only mode for PyTorch (it's still fast)")
    print("2. Try updating your system drivers")
    print("3. For production use, consider a more supported GPU")
EOL

echo ""
echo "Setup complete. To test if GPU acceleration works, run:"
echo "source setup_env.sh && python test_gpu.py"
echo ""
echo "If GPU is not detected, consider using CPU mode which is still fast for most tasks."
echo "In that case, reinstall PyTorch with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
