#!/bin/bash

# Script to fix ROCm environment for PyTorch

echo "Fixing ROCm environment for PyTorch..."

# Activate virtual environment
source .venv/bin/activate

# Check ROCm installation
echo "Checking ROCm installation..."
if ! command -v rocminfo &> /dev/null; then
    echo "rocminfo not found. ROCm may not be installed properly."
    echo "Try installing ROCm using the official AMD instructions:"
    echo "https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    exit 1
fi

# Check for AMD GPU
echo "Checking for AMD GPU..."
if ! rocminfo 2>&1 | grep -i "gpu agent" &> /dev/null; then
    echo "No AMD GPU detected by ROCm."
    echo "Make sure your GPU is supported by ROCm:"
    echo "https://rocm.docs.amd.com/en/latest/about/compatibility.html"
    exit 1
fi

# Set up system paths
echo "Setting up system paths..."
export PATH=$PATH:/opt/rocm/bin
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:$LD_LIBRARY_PATH

# Set AMD GPU environment variables
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0

# Reinstall PyTorch with ROCm support
echo "Reinstalling PyTorch with ROCm support..."
pip uninstall -y torch torchvision torchaudio
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.6

# Ensure NumPy is at the correct version (1.24.3)
pip install numpy==1.24.3

# Verify installation
echo -e "\nVerifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Is CUDA/ROCm available: {torch.cuda.is_available()}'); print(f'GPU Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else \"N/A\"}');" || echo "PyTorch import failed"

# Update setup_env.sh with correct paths
echo -e "\nUpdating setup_env.sh with correct paths..."
sed -i 's|export PATH=$PATH:/opt/rocm-5.6.0|export PATH=$PATH:/opt/rocm|' setup_env.sh
sed -i 's|export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-5.6.0/lib|export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:$LD_LIBRARY_PATH|' setup_env.sh

echo -e "\nDone! Try running: source setup_env.sh"
