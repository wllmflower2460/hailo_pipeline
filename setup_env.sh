#!/bin/bash

# Script to set up environment variables for the dog training AI pipeline

# Activate virtual environment if not already activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    source .venv/bin/activate
fi

# Set project root directory
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Data directories
export DATA_DIR="${PROJECT_ROOT}/data"
export RAW_DATA_DIR="${DATA_DIR}/raw"
export PROCESSED_DATA_DIR="${DATA_DIR}/processed"
export CACHE_DIR="${DATA_DIR}/cache"

# Model directories
export MODELS_DIR="${PROJECT_ROOT}/results/models"
export ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts"

# Hailo directories (if using Hailo)
export HAILO_MODELS_DIR="${PROJECT_ROOT}/hailo_integration/models"

# CPU optimization settings
export MKL_NUM_THREADS=$(nproc)
export OMP_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)

# Optional: reduce threads if you want to use the system for other tasks simultaneously
# export MKL_NUM_THREADS=$(($(nproc) / 2))
# export OMP_NUM_THREADS=$(($(nproc) / 2))

# Note: ROCm environment variables are disabled as we're using CPU-only mode
# To re-enable GPU support in the future, uncomment the following:
# export HIP_VISIBLE_DEVICES=0
# export ROCR_VISIBLE_DEVICES=0
# export HSA_ENABLE_SDMA=0
# export HSA_OVERRIDE_GFX_VERSION=11.0.0
# export ROC_ENABLE_PRE_VEGA=1
# export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/opt/rocm/lib:/opt/rocm/hip/lib:$LD_LIBRARY_PATH

# Print environment information
echo "Environment set up successfully"
echo "Python: $(which python)"
echo "Project root: ${PROJECT_ROOT}"
echo "Data directory: ${DATA_DIR}"
echo "Models directory: ${MODELS_DIR}"

# Check for GPU
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'ROCm devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not installed or not configured correctly"

# Standard library path for system libraries
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Print CPU information for performance reference
cores=$(nproc)
echo "CPU cores available: ${cores}"
echo "Thread settings:"
echo " - MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo " - OMP_NUM_THREADS: ${OMP_NUM_THREADS}" 
echo "Use 'deactivate' to exit the virtual environment"
