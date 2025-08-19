#!/bin/bash

# Simple script to start Hailo containers with Podman

echo "Using Podman to start Hailo containers"

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Clean up any existing containers
podman rm -f hailo-dog-training hailo-inference 2>/dev/null || true

# Build the image
podman build -t hailo-dog-training:latest -f "${PROJECT_ROOT}/hailo_integration/docker/Dockerfile" "$PROJECT_ROOT"

# Create directories if they don't exist
mkdir -p "${PROJECT_ROOT}/data" "${PROJECT_ROOT}/results"

# Start the development container
echo "Starting development container..."
podman run -d --name hailo-dog-training \
    -v "${PROJECT_ROOT}:/app:Z" \
    -e PYTHONPATH=/app \
    -e OMP_NUM_THREADS=8 \
    -e MKL_NUM_THREADS=8 \
    -e NUMEXPR_NUM_THREADS=8 \
    -e OPENBLAS_NUM_THREADS=8 \
    -p 8889:8889 \
    hailo-dog-training:latest \
    bash -c "cd /app && python3 -m jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root --NotebookApp.token='hailo'"

# Start the inference container
echo "Starting inference container..."
if [ -e "/dev/hailo0" ]; then
    podman run -d --name hailo-inference \
        -v "${PROJECT_ROOT}:/app:Z" \
        -e PYTHONPATH=/app \
        --device /dev/hailo0:/dev/hailo0 \
        hailo-dog-training:latest \
        bash -c "cd /app && echo 'Hailo inference service is running. Connect with: podman exec -it hailo-inference bash' && tail -f /dev/null"
else
    echo "Warning: Hailo device (/dev/hailo0) not found. Inference container will be started without Hailo device access."
    podman run -d --name hailo-inference \
        -v "${PROJECT_ROOT}:/app:Z" \
        -e PYTHONPATH=/app \
        hailo-dog-training:latest \
        bash -c "cd /app && echo 'Hailo inference service is running. Connect with: podman exec -it hailo-inference bash' && tail -f /dev/null"
fi

echo ""
echo "Hailo environment started!"
echo "-----------------------------"
echo "Jupyter notebook: http://localhost:8889 (Token: hailo)"
echo "To access the development container: podman exec -it hailo-dog-training bash"
echo "To access the inference container: podman exec -it hailo-inference bash"
echo ""
echo "To stop the environment: podman stop hailo-dog-training hailo-inference"
