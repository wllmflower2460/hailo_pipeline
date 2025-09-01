# Hailo Pipeline Refactoring Plan
**Date**: 2025-08-31  
**Goal**: Remove TCN-VAE functionality overlap, focus solely on Hailo file generation  
**Target**: Convert TCN-VAE models → .hef files → EdgeInfer deployment  

## 🎯 Current EdgeInfer Service Analysis

### **EdgeInfer API Endpoints** (http://localhost:8080)
```
GET  /healthz                           - Service health check
GET  /metrics                           - Prometheus metrics  
POST /api/v1/analysis/start             - Start analysis session
PUT  /api/v1/analysis/stream            - Stream IMU data
GET  /api/v1/analysis/motifs?sessionId  - Get motif analysis
GET  /api/v1/analysis/synchrony?sessionId - Get synchrony analysis  
POST /api/v1/analysis/stop              - Stop analysis session
```

### **Model Integration Points**
```swift
// EdgeInfer calls ModelInferenceService
let inferenceResult = try await ModelInferenceService.analyzeIMUWindow(req, samples: samples)

// Service configuration
let backendURL = Environment.get("MODEL_BACKEND_URL") ?? "http://model-runner:8000"
let useReal = Environment.get("USE_REAL_MODEL") == "true"

// Expected API contract
POST /infer
Input: IMUWindow { x: [[Float]] }  // (100, 9) IMU window
Output: ModelInferenceResult { 
    latent: [Float],      // 64-dim embeddings
    motif_scores: [Float] // 12 motif scores
}
```

## 🔧 Refactoring Strategy

### **REMOVE from hailo_pipeline**
- ❌ TCN-VAE model training code (→ TCN-VAE_models repo)
- ❌ IMU data preprocessing (→ TCN-VAE_models repo)
- ❌ Model evaluation/validation (→ TCN-VAE_models repo)
- ❌ Dataset handling (→ ml-models repo)
- ❌ Multi-modal fusion (→ ml-models repo)

### **KEEP in hailo_pipeline**
- ✅ Hailo SDK integration
- ✅ ONNX → .hef compilation pipeline
- ✅ Hailo runtime (HailoRT) inference server
- ✅ Docker deployment configurations
- ✅ Performance benchmarking for Hailo hardware
- ✅ Pi deployment automation

### **NEW FOCUS: TCN-VAE → Hailo Pipeline**
```mermaid
graph LR
    A[TCN-VAE Models] --> B[ONNX Export]
    B --> C[Hailo DFC Compile]
    C --> D[.hef Files]
    D --> E[HailoRT Server]
    E --> F[EdgeInfer Integration]
```

## 📁 Refactored Directory Structure

### **hailo_pipeline/ (FOCUSED)**
```
hailo_pipeline/
├── src/
│   ├── onnx_export/
│   │   ├── tcn_encoder_export.py      # TCN-VAE → ONNX
│   │   ├── model_validation.py        # Validate ONNX outputs
│   │   └── export_config.yaml         # Export configurations
│   ├── hailo_compilation/
│   │   ├── compile_tcn_model.py       # ONNX → .hef
│   │   ├── hailo_config.yaml          # DFC compilation settings
│   │   └── optimize_for_pi.py         # Pi-specific optimizations
│   ├── runtime/
│   │   ├── hailort_server.py          # FastAPI inference server
│   │   ├── model_loader.py            # .hef model loading
│   │   └── api_endpoints.py           # /infer endpoint implementation
│   └── deployment/
│       ├── docker_compose_hailo.yml   # Hailo runtime containers
│       ├── pi_deploy.sh               # Pi deployment automation
│       └── edgeinfer_integration.py   # EdgeInfer service integration
├── models/
│   ├── tcn_encoder.onnx              # Exported from TCN-VAE_models
│   ├── tcn_encoder.hef               # Compiled for Hailo
│   └── model_metadata.json           # Model specs and performance
├── docker/
│   ├── hailo_runtime/                # HailoRT inference container
│   ├── hailo_compiler/               # DFC compilation container
│   └── edgeinfer_integration/        # EdgeInfer integration container
├── configs/
│   ├── hailo_optimization.yaml       # Hailo-specific optimizations
│   ├── pi_deployment.yaml            # Pi deployment configuration
│   └── edgeinfer_backend.yaml        # Backend service configuration
└── scripts/
    ├── full_pipeline.sh              # End-to-end: TCN-VAE → EdgeInfer
    ├── benchmark_hailo.py            # Performance testing
    └── validate_deployment.py       # Integration testing
```

## 🚀 Implementation Pipeline

### **Phase 1: ONNX Export Module**
```python
# src/onnx_export/tcn_encoder_export.py
import torch
import torch.onnx
from pathlib import Path

def export_tcn_encoder_to_onnx(
    model_path: str,  # From TCN-VAE_models/tcn_encoder_for_edgeinfer.pth
    output_path: str,
    input_shape: tuple = (1, 100, 9)
):
    """Export TCN encoder to ONNX for Hailo compilation"""
    
    # Load trained model
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    
    # Create dummy input matching EdgeInfer API
    dummy_input = torch.randn(input_shape)
    
    # Export with fixed input names for Hailo
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,  # Hailo-compatible
        do_constant_folding=True,
        input_names=['imu_window'],
        output_names=['latent_embeddings', 'motif_scores'],
        dynamic_axes={
            'imu_window': {0: 'batch_size'},
            'latent_embeddings': {0: 'batch_size'},
            'motif_scores': {0: 'batch_size'}
        }
    )
    
    print(f"✅ TCN encoder exported to {output_path}")
```

### **Phase 2: Hailo Compilation**
```python
# src/hailo_compilation/compile_tcn_model.py
import subprocess
from pathlib import Path

def compile_onnx_to_hef(
    onnx_path: str,
    output_path: str,
    config_path: str = "configs/hailo_optimization.yaml"
):
    """Compile ONNX model to Hailo .hef format"""
    
    cmd = [
        "hailo", "compile",
        onnx_path,
        "--config", config_path,
        "--output", output_path,
        "--target", "hailo8",
        "--quantization", "int8",
        "--optimization", "performance"  # vs accuracy
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Model compiled to {output_path}")
        return True
    else:
        print(f"❌ Compilation failed: {result.stderr}")
        return False
```

### **Phase 3: HailoRT Inference Server**
```python
# src/runtime/hailort_server.py
from fastapi import FastAPI
import numpy as np
from hailo_platform import InferenceEnginer

app = FastAPI()
model_engine = None

@app.on_event("startup")
async def load_model():
    global model_engine
    model_engine = InferenceEnginer("models/tcn_encoder.hef")

@app.post("/infer")
async def infer_imu_window(window: dict):
    """
    EdgeInfer-compatible inference endpoint
    Input: {"x": [[float]]}  # (100, 9) IMU window
    Output: {"latent": [float], "motif_scores": [float]}
    """
    
    # Prepare input for Hailo
    imu_data = np.array(window["x"], dtype=np.float32)
    imu_data = imu_data.reshape(1, 100, 9)  # Add batch dimension
    
    # Run inference on Hailo
    outputs = model_engine.infer({"imu_window": imu_data})
    
    return {
        "latent": outputs["latent_embeddings"][0].tolist(),
        "motif_scores": outputs["motif_scores"][0].tolist()
    }

@app.get("/healthz")
async def health_check():
    return {"status": "healthy", "model_loaded": model_engine is not None}
```

### **Phase 4: EdgeInfer Integration**
```yaml
# docker/docker_compose_hailo.yml
version: '3.8'
services:
  hailo-inference:
    build: ./hailo_runtime
    container_name: hailo-tcn-inference
    ports:
      - "8000:8000"  # MODEL_BACKEND_URL=http://hailo-inference:8000
    volumes:
      - ./models:/app/models:ro
      - /opt/hailo:/opt/hailo:ro  # Hailo drivers
    environment:
      - MODEL_PATH=/app/models/tcn_encoder.hef
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔗 Integration with Existing Infrastructure

### **EdgeInfer Service Configuration**
```bash
# Update EdgeInfer environment variables
MODEL_BACKEND_URL=http://hailo-inference:8000
USE_REAL_MODEL=true

# Docker compose integration
cd /home/pi/pisrv_vapor_docker/
# Add hailo-inference service to existing docker-compose.yml
docker-compose up -d hailo-inference
docker-compose restart edge-infer
```

### **TCN-VAE_models Integration**
```bash
# Copy trained model to hailo_pipeline
cp ../TCN-VAE_models/tcn_encoder_for_edgeinfer.pth models/input/
cp ../TCN-VAE_models/model_config.json models/

# Run conversion pipeline
python src/onnx_export/tcn_encoder_export.py \
    --input models/input/tcn_encoder_for_edgeinfer.pth \
    --output models/tcn_encoder.onnx

python src/hailo_compilation/compile_tcn_model.py \
    --input models/tcn_encoder.onnx \
    --output models/tcn_encoder.hef
```

## 📊 Success Metrics

### **Pipeline Performance**
- [ ] **ONNX Export**: <30 seconds conversion time
- [ ] **Hailo Compilation**: <5 minutes .hef generation
- [ ] **Inference Speed**: <50ms per IMU window (100 samples)
- [ ] **Model Size**: <10MB .hef file for Pi deployment

### **EdgeInfer Integration**
- [ ] **API Compatibility**: 100% compatible with existing endpoints
- [ ] **Accuracy**: Match TCN-VAE_models 57.68% validation accuracy
- [ ] **Reliability**: 99%+ successful inference rate
- [ ] **Deployment**: One-command GPUSrv → Pi deployment

### **Resource Utilization**
- [ ] **Hailo Utilization**: >80% accelerator usage
- [ ] **Memory**: <512MB Pi RAM usage
- [ ] **Power**: <5W additional power consumption
- [ ] **Thermal**: Maintain Pi operating temperature <70°C

## ⚠️ Migration Strategy

### **Safe Refactoring Steps**
1. **Create new focused directories** before removing old code
2. **Test ONNX export** with existing TCN-VAE models
3. **Validate Hailo compilation** on GPUSrv before Pi deployment
4. **Integration testing** with EdgeInfer stub mode first
5. **Gradual rollout** with feature flag control

### **Rollback Plan**
```bash
# Keep stub mode available
USE_REAL_MODEL=false  # Revert to EdgeInfer stubs

# Container rollback
docker-compose stop hailo-inference
docker-compose restart edge-infer  # Back to stub mode
```

This refactoring transforms hailo_pipeline into a **focused conversion and deployment tool**, eliminating overlap while creating a clear pipeline from your proven TCN-VAE models to production Hailo inference on the Pi.