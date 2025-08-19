#!/bin/bash
# This script performs a comprehensive check of the Hailo environment setup
# It verifies the Docker/Podman containers, libraries, and functionality

echo "==== Hailo Environment Check ===="

# Check if Docker or Podman is installed
if command -v docker &> /dev/null; then
    CONTAINER_TOOL="docker"
    echo "✅ Docker is installed"
elif command -v podman &> /dev/null; then
    CONTAINER_TOOL="podman"
    echo "✅ Podman is installed"
else
    echo "❌ Neither Docker nor Podman is installed"
    echo "Please install Docker or Podman first"
    exit 1
fi

# Check if containers are running
echo -e "\nChecking containers..."

# Check dev container
if $CONTAINER_TOOL ps | grep -q hailo-dog-training; then
    echo "✅ Hailo development container is running"
    
    # Check if OpenCV is working in the container
    echo -e "\nTesting OpenCV in development container..."
    $CONTAINER_TOOL exec -it hailo-dog-training python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" && echo "✅ OpenCV is working" || echo "❌ OpenCV test failed"
    
    # Check if other libraries are available
    echo -e "\nTesting Python libraries in development container..."
    $CONTAINER_TOOL exec -it hailo-dog-training python3 -c "import numpy as np; import torch; print('NumPy version:', np.__version__); print('PyTorch version:', torch.__version__)" && echo "✅ Python libraries are working" || echo "❌ Python libraries test failed"
else
    echo "❌ Hailo development container is not running"
    echo "Run ./start-hailo-docker.sh to start the container"
fi

# Check inference container
if $CONTAINER_TOOL ps | grep -q hailo-inference; then
    echo "✅ Hailo inference container is running"
else
    echo "❌ Hailo inference container is not running"
    echo "Run ./start-hailo-docker.sh to start the container"
fi

# Check Jupyter notebook server
echo -e "\nChecking Jupyter notebook server..."
if curl -s -I http://localhost:8889 &> /dev/null; then
    echo "✅ Jupyter notebook server is accessible at http://localhost:8889"
else
    echo "❌ Jupyter notebook server is not accessible"
    echo "The Jupyter notebook server should be running in the container"
fi

# Check sample data
echo -e "\nChecking sample data..."
if [ -f "../data/sample/sample_dog.jpg" ]; then
    echo "✅ Sample data is available"
else
    echo "❌ Sample data is not available"
    echo "Creating sample directory and downloading a sample image..."
    mkdir -p ../data/sample
    curl -o ../data/sample/sample_dog.jpg https://images.dog.ceo/breeds/retriever-golden/n02099601_7130.jpg
    echo "✅ Sample image downloaded"
fi

# Final summary
echo -e "\n==== Environment Check Summary ===="
echo "Container tool: $CONTAINER_TOOL"
echo "Development container: $($CONTAINER_TOOL ps | grep -q hailo-dog-training && echo 'Running' || echo 'Not running')"
echo "Inference container: $($CONTAINER_TOOL ps | grep -q hailo-inference && echo 'Running' || echo 'Not running')"
echo "Jupyter notebook: $(curl -s -I http://localhost:8889 &> /dev/null && echo 'Accessible' || echo 'Not accessible')"
echo "Sample data: $([ -f "../data/sample/sample_dog.jpg" ] && echo 'Available' || echo 'Not available')"
echo -e "\nCheck completed. If any issues were found, please address them before proceeding."
