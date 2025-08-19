# Dog Training AI Pipeline

High-performance machine learning environment for dog training AI pipeline development.

## System Configuration

- **CPU**: AMD Ryzen R9-7940HS
- **RAM**: 32GB DDR5
- **Storage**: 2TB NVMe SSD
- **GPU**: AMD Radeon 780M
- **OS**: Pop!_OS (Linux)

## Development Environment

This project is set up for machine learning with a focus on dog training AI pipeline development.

### GPU Support Status

The AMD Radeon 780M integrated GPU is not fully supported by ROCm at this time, despite our efforts to configure it. The environment is currently working in CPU-only mode, which is still very performant for most tasks.

If you need GPU acceleration and are experiencing issues with the AMD Radeon 780M:

1. Run our diagnostic script: `python src/rocm_diagnostic.py`
2. Try the GPU setup script: `./setup_gpu.sh`
3. For production use, consider a dedicated AMD GPU with better ROCm support or an NVIDIA GPU with CUDA support

### Prerequisites

The following system packages are required:

```bash
sudo apt install -y build-essential cmake git curl wget python3-dev python3-pip \
    python3-venv libopenblas-dev liblapack-dev libblas-dev gfortran \
    pkg-config htop btop iotop
```

For AMD GPU support, ROCm is required (but may not fully support newer integrated GPUs):

```bash
# Add ROCm repository and install packages
sudo apt install rocm-hip-sdk rocm-dkms
```

### Setup Instructions

1. Clone this repository:
```bash
git clone <repository-url>
cd dog_training_pipeline
```

2. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
source setup_env.sh
```

5. Verify your setup:
```bash
python src/check_env.py
```

### Running Jupyter Notebooks

Start the Jupyter notebook server:
```bash
./start_jupyter.sh
```

Then open `environment_test.ipynb` to verify that your environment is working correctly.

## Project Structure

```
dog_training_pipeline/
├── artifacts/              # Model artifacts and outputs
├── data/                   # Data directory
│   ├── cache/              # Cached data
│   ├── processed/          # Processed data
│   └── raw/                # Raw data
├── hailo_integration/      # Hailo AI accelerator integration
├── logs/                   # Log files
├── notebooks/              # Jupyter notebooks
│   └── environment_test.ipynb  # Environment test notebook
├── results/                # Results directory
│   ├── models/             # Trained models
│   ├── reports/            # Reports and analysis
│   └── visualizations/     # Visualizations
├── src/                    # Source code
│   ├── check_env.py        # Environment checker
│   ├── data/               # Data processing modules
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── .venv/                  # Virtual environment
├── requirements.txt        # Python dependencies
├── setup_env.sh            # Environment setup script
└── start_jupyter.sh        # Jupyter notebook launcher
```

## Hailo Integration

This project includes comprehensive integration with Hailo AI accelerators for optimized inference on edge devices. The Hailo integration provides:

- Docker/Podman containerized environment
- Jupyter notebook integration
- Sample dog detection models
- Performance benchmarking tools

To get started with Hailo:

1. Navigate to the Hailo integration directory:
```bash
cd hailo_integration
```

2. Start the Hailo containers:
```bash
cd docker
./start-hailo-docker-simple.sh
```

3. Run the environment check:
```bash
./check-hailo-environment.sh
```

4. Try the dog detection demo:
```bash
./run-container-dog-detection.sh
```

For detailed instructions, see the [Hailo Integration README](hailo_integration/README.md).

### Hailo Setup

1. Navigate to the hailo_integration directory:
```bash
cd hailo_integration/docker
```

2. Start the Hailo Docker/Podman containers:
```bash
./start-hailo-docker.sh
```

3. Test the environment:
```bash
./test-hailo-environment.sh
```

4. Run dog detection on a sample image:
```bash
./run-container-dog-detection.sh
```

For more details, check out the [Hailo integration documentation](hailo_integration/README.md).

### Jupyter Notebook Demo

A demonstration notebook for Hailo-based dog detection is available at `notebooks/hailo_dog_detection_demo.ipynb`. You can run this notebook from the Jupyter server started with:

```bash
./start_jupyter.sh
```

Or access the container's Jupyter server at http://localhost:8889 (Token: hailo)

## License

[Add license information here]

## Contact

[Add contact information here]
