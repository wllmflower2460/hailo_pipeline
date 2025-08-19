#!/bin/bash

# Script to install PyTorch with CPU support for the dog training pipeline
# Use this if ROCm/GPU acceleration isn't working with your AMD GPU

# Activate virtual environment
source .venv/bin/activate

# Uninstall current PyTorch installations
pip uninstall -y torch torchvision torchaudio

# Install PyTorch with CPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Make sure numpy is at the correct version
pip install numpy==1.24.3

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CPU-only mode: {not torch.cuda.is_available()}')"

echo ""
echo "PyTorch has been installed in CPU-only mode."
echo "While GPU acceleration isn't available, modern CPUs like the AMD Ryzen 9 are still very powerful for most ML tasks."
echo ""
echo "To continue setup, run: source setup_env.sh"
