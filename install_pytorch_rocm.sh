#!/bin/bash

# Script to install PyTorch with ROCm support for AMD GPUs

# Activate virtual environment
source .venv/bin/activate

# Uninstall current PyTorch installations
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Set up ROCm environment variables
source setup_env.sh

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'ROCm devices: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
