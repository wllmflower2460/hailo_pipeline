#!/bin/bash

# Script to launch Jupyter notebook with CPU optimization

# Activate the environment and set up variables
source ./setup_env.sh

# Set CPU optimization parameters
echo "Setting up optimized CPU environment for Jupyter..."
export MKL_NUM_THREADS=$(nproc)
export OMP_NUM_THREADS=$(nproc)
export NUMEXPR_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)

# Print optimization information
echo "CPU optimization settings:"
echo "- Cores available: $(nproc)"
echo "- MKL_NUM_THREADS: ${MKL_NUM_THREADS}"
echo "- OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
echo "- NUMEXPR_NUM_THREADS: ${NUMEXPR_NUM_THREADS}"
echo "- OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"

# Launch Jupyter notebook server with optimized settings
echo "Starting Jupyter notebook server..."
jupyter notebook --notebook-dir=notebooks
