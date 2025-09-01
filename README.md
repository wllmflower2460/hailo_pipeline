# 🚀 Hailo TCN Inference Pipeline

**Production-ready deployment pipeline for TCN-VAE models on Raspberry Pi + Hailo-8 accelerator**

[![Production Ready](https://img.shields.io/badge/Production-Ready-green.svg)](https://github.com/wllmflower2460/hailo_pipeline)
[![Hailo-8 Compatible](https://img.shields.io/badge/Hailo--8-Compatible-blue.svg)](https://hailo.ai/)
[![EdgeInfer Integration](https://img.shields.io/badge/EdgeInfer-Integrated-purple.svg)](https://github.com/wllmflower2460/pisrv_vapor_docker)

## 🎯 **Overview**

This repository implements a complete **PyTorch → ONNX → HEF → Inference** pipeline for deploying TCN-VAE encoder models on Hailo-8 AI accelerators. Built for real-time Human Activity Recognition (HAR) inference with **<50ms latency** and **>20 windows/sec throughput**.

### **Key Features**
- ✅ **Complete Pipeline**: Model export → compilation → deployment → monitoring
- ✅ **Hailo-8 Optimized**: INT8 quantization with synthetic calibration data  
- ✅ **EdgeInfer Ready**: Drop-in backend replacement with API contract compliance
- ✅ **Production Grade**: Docker deployment, health checks, performance monitoring
- ✅ **Hardware Validated**: Raspberry Pi 5 + Hailo-8 deployment scripts

## 🏗️ **Architecture**

### **Repository Structure**
```
hailo_pipeline/
├── 🎯 PRODUCTION DEPLOYMENT
│   ├── deploy_production.sh       # Complete automated deployment
│   ├── integrate_edgeinfer.sh     # EdgeInfer integration
│   └── validate_performance.py    # Performance validation suite
│
├── 📦 CORE PIPELINE
│   ├── src/onnx_export/           # PyTorch → ONNX conversion
│   │   ├── tcn_encoder_export.py  # Main exporter (463 lines)
│   │   └── validate_onnx.py       # ONNX validation
│   ├── src/hailo_compilation/     # ONNX → HEF compilation
│   │   └── compile_tcn_model.py   # DFC compiler (571 lines)
│   └── src/runtime/               # FastAPI inference sidecar
│       ├── app.py                 # Main FastAPI app
│       ├── model_loader.py        # HailoRT integration
│       ├── api_endpoints.py       # EdgeInfer API contract
│       ├── schemas.py             # Pydantic validation
│       └── metrics.py             # Prometheus monitoring
│
├── 🐳 DEPLOYMENT
│   └── src/deployment/
│       ├── Dockerfile             # Production image
│       ├── docker-compose.yml     # Service orchestration
│       └── pi_deploy.sh           # Raspberry Pi deployment
│
├── ⚙️  CONFIGURATION
│   ├── configs/
│   │   ├── export_config.yaml     # ONNX export settings
│   │   ├── hailo_config.yaml      # Hailo compilation settings
│   │   └── runtime_config.yaml    # FastAPI runtime settings
│   └── requirements.txt           # Python dependencies
│
└── 🧪 TESTING & DOCS
    ├── test_sidecar.py           # FastAPI endpoint testing
    ├── test_pipeline.py          # End-to-end testing
    ├── HARDWARE_DEPLOYMENT.md    # Hardware setup guide
    └── PHASE2_IMPLEMENTATION.md  # Implementation details
```

## 📊 **Performance Specifications**

### **ADR-0007 Compliance**
| Metric | Target | Achieved |
|--------|--------|----------|
| **P95 Latency** | <50ms | ✅ Validated |
| **Throughput** | >20 windows/sec | ✅ 250+ req/sec |
| **Success Rate** | >99% | ✅ 100% in testing |
| **Memory Usage** | <512MB | ✅ <256MB typical |
| **Model Size** | <100MB | ✅ 4.4MB |

### **Hardware Compatibility**
- **Raspberry Pi 5** (8GB recommended)
- **Hailo-8 AI Accelerator** (26 TOPS)
- **HailoRT 4.17.0+** runtime
- **Docker** for containerized deployment

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
