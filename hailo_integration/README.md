# Hailo Dog Detection Integration

This directory contains integration code for running dog detection using Hailo AI accelerators. The integration provides a complete environment for developing and running dog detection models using Hailo hardware accelerators.

## Directory Structure
- `docker/` - Docker/Podman configuration files and utility scripts
- `models/` - Hailo model files and model conversion tools
- `benchmarks/` - Performance benchmark results and comparison data
- `configs/` - Configuration files for Hailo models and runtime options
- `test_dog_detection.py` - Simple test script for dog detection
- `hailo_dog_detection.py` - Main dog detection implementation

## Quick Start

### Setup the Environment

```bash
# Start the Hailo Docker/Podman containers
cd docker
./start-hailo-docker-simple.sh
```

This will start two containers:
- `hailo-dog-training`: Development container with Python, PyTorch, and other ML tools
- `hailo-inference`: Container for running Hailo inference with hardware acceleration

### Run Dog Detection

```bash
# Run dog detection on a sample image
cd docker
./run-container-dog-detection.sh
```

The results will be saved to `results/visualizations/dog_detection_result.jpg`.

### Testing the Environment

```bash
# Comprehensive environment check
cd docker
./check-hailo-environment.sh
```

This will verify:
- Container status
- Library installations
- Sample data availability
- Jupyter notebook accessibility

## Available Scripts

### Docker/Container Management
- `start-hailo-docker.sh` - Starts the Hailo Docker/Podman containers
- `start-hailo-docker-simple.sh` - Simplified version for more reliable startup

### Testing and Verification
- `check-hailo-environment.sh` - Comprehensive environment check
- `test-hailo-environment.sh` - Tests basic environment functionality
- `test-container-image-processing.sh` - Tests image processing in the container

### Dog Detection
- `run-container-dog-detection.sh` - Runs dog detection on a sample image
- `container_dog_detection.py` - Python script for dog detection inside container

## Working with Jupyter Notebooks

A Jupyter notebook server is available at http://localhost:8889 (Token: hailo).

We've provided a sample notebook at `notebooks/hailo_dog_detection_demo.ipynb` that demonstrates:
- Loading and preprocessing images
- Running inference with Hailo (or simulation)
- Visualizing detection results
- Measuring performance

## Troubleshooting

### Permission Issues

If you encounter permission issues when saving files from the container, use the `/tmp` directory
inside the container and then copy the files to the host using `podman cp` or `docker cp`:

```bash
podman cp hailo-dog-training:/tmp/file.jpg /path/on/host/
```

### Docker vs Podman

This setup works with both Docker and Podman. If you're using Podman, you may see warnings about 
CNI config validation, but these can be safely ignored.

### OpenCV Issues

If you encounter issues with OpenCV in the container, try reinstalling it:

```bash
podman exec -it hailo-dog-training pip install --no-cache-dir --upgrade opencv-python
```

### Jupyter Access Problems

If you can't access the Jupyter server, check if the port is correctly mapped:

```bash
podman ps
```

You should see port 8889 mapped to the container.
