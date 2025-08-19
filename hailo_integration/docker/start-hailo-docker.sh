#!/bin/bash

# Script to start Hailo Docker cont# Start the development container with bind mounts instead of volumes
    echo "Starting development container..."
    echo "Project root: $PROJECT_ROOT"
    
    # Create mount directories if they don't exist
    mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/results"
    
    podman run -d --name hailo-dog-training 
        --mount type=bind,source="$PROJECT_ROOT",target=/app 
        --mount type=bind,source="$PROJECT_ROOT/data",target=/app/data 
        --mount type=bind,source="$PROJECT_ROOT/results",target=/app/results 
        -e PYTHONPATH=/app 
        -e OMP_NUM_THREADS=8 
        -e MKL_NUM_THREADS=8 
        -e NUMEXPR_NUM_THREADS=8 
        -e OPENBLAS_NUM_THREADS=8 
        -p 8888:8888 
        hailo-dog-training:latest 
        bash -c "cd /app && python3 -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='hailo'"
    
    # Start the inference container if Hailo device exists
    echo "Starting inference container..."
    if [ -e "/dev/hailo0" ]; then
        podman run -d --name hailo-inference 
            --mount type=bind,source="$PROJECT_ROOT",target=/app 
            --mount type=bind,source="$PROJECT_ROOT/data",target=/app/data 
            --mount type=bind,source="$PROJECT_ROOT/results",target=/app/results 
            -e PYTHONPATH=/app 
            --device /dev/hailo0:/dev/hailo0 
            -t 
            hailo-dog-training:latest 
            bash -c "cd /app && echo 'Hailo inference service is running. Connect with: podman exec -it hailo-inference bash'"
    else
        echo "Warning: Hailo device (/dev/hailo0) not found. Inference container will be started without Hailo device access."
        podman run -d --name hailo-inference 
            --mount type=bind,source="$PROJECT_ROOT",target=/app 
            --mount type=bind,source="$PROJECT_ROOT/data",target=/app/data 
            --mount type=bind,source="$PROJECT_ROOT/results",target=/app/results 
            -e PYTHONPATH=/app 
            -t 
            hailo-dog-training:latest 
            bash -c "cd /app && echo 'Hailo inference service is running. Connect with: podman exec -it hailo-inference bash'"
    fitermine whether to use Podman or Docker (prioritizing Podman)
if command -v podman &> /dev/null; then
    CONTAINER_ENGINE="podman"
    if command -v podman-compose &> /dev/null; then
        COMPOSE_CMD="podman-compose"
    else
        echo "Warning: podman-compose not found. Will use direct podman commands."
    fi
    echo "Using Podman as container engine"
    
    # Create simpler rootless container configuration for Podman
    echo "Setting up rootless Podman configuration..."
    # Ensure Podman is configured for rootless mode
    export PODMAN_USERNS="keep-id"
elif command -v docker &> /dev/null; then
    CONTAINER_ENGINE="docker"
    COMPOSE_CMD="docker-compose"
    echo "Using Docker as container engine"
else
    echo "Error: Neither Podman nor Docker found. Please install Podman or Docker."
    exit 1
fi

# Navigate to directory containing the docker-compose.yml file
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Build and start containers
if [[ "$CONTAINER_ENGINE" == "podman" ]]; then
    # For Podman, run individual containers directly to avoid networking issues
    echo "Starting containers with Podman..."
    
    # Build the image first
    podman build -t hailo-dog-training:latest -f Dockerfile ../../
    
    # Make sure there are no previous instances running
    podman rm -f hailo-dog-training 2>/dev/null || true
    podman rm -f hailo-inference 2>/dev/null || true
    
    # Get absolute paths for volumes
    SCRIPT_DIR_ABS="$( cd "$( dirname "{BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PROJECT_ROOT="$( cd "{SCRIPT_DIR_ABS}/../.." &> /dev/null && pwd )"
    
    # Start the development container
    podman run -d --name hailo-dog-training \
        -v "{PROJECT_ROOT}":/app:Z \
        -v "{PROJECT_ROOT}/data":/app/data:Z \
        -v "{PROJECT_ROOT}/results":/app/results:Z \
        -e PYTHONPATH=/app \
        -e OMP_NUM_THREADS=8 \
        -e MKL_NUM_THREADS=8 \
        -e NUMEXPR_NUM_THREADS=8 \
        -e OPENBLAS_NUM_THREADS=8 \
        -p 8888:8888 \
        hailo-dog-training:latest \
        bash -c "cd /app && python3 -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='hailo'"
    
    # Start the inference container if Hailo device exists
    if [ -e "/dev/hailo0" ]; then
        podman run -d --name hailo-inference \
            -v "{PROJECT_ROOT}":/app:Z \
            -v "{PROJECT_ROOT}/data":/app/data:Z \
            -v "{PROJECT_ROOT}/results":/app/results:Z \
            -e PYTHONPATH=/app \
            --device /dev/hailo0:/dev/hailo0 \
            --tty \
            hailo-dog-training:latest \
            bash -c "cd /app && echo 'Hailo inference service is running. Connect with: podman exec -it hailo-inference bash'"
    else
        echo "Warning: Hailo device (/dev/hailo0) not found. Inference container will be started without Hailo device access."
        podman run -d --name hailo-inference \
            -v "{PROJECT_ROOT}":/app:Z \
            -v "{PROJECT_ROOT}/data":/app/data:Z \
            -v "{PROJECT_ROOT}/results":/app/results:Z \
            -e PYTHONPATH=/app \
            --tty \
            hailo-dog-training:latest \
            bash -c "cd /app && echo 'Hailo inference service is running. Connect with: podman exec -it hailo-inference bash'"
    fi
else
    # Use docker-compose for Docker
    COMPOSE_CMD up -d
fi

echo ""
echo "Hailo environment started!"
echo "-----------------------------"
echo "Jupyter notebook: http://localhost:8888 (Token: hailo)"
echo "To access the development container: $CONTAINER_ENGINE exec -it hailo-dog-training bash"
echo "To access the inference container: $CONTAINER_ENGINE exec -it hailo-inference bash"
echo ""
echo "To stop the environment: $CONTAINER_ENGINE stop hailo-dog-training hailo-inference"
