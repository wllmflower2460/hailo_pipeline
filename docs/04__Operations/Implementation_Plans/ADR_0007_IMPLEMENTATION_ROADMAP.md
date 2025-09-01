# ADR-0007 Implementation Roadmap
**Date**: 2025-08-31  
**Session ID**: ADR-0007-2025-08-31  
**AI Handoff**: ChatGPT → Claude Code → Copilot  
**Component**: HailoRT Sidecar for EdgeInfer  

## 🎯 Mission Statement
Implement a **single-purpose hailo_pipeline** that converts TCN-VAE encoder → ONNX → .hef → FastAPI HailoRT sidecar, serving as a drop-in backend for EdgeInfer with contract-first API stability.

## 📋 ADR-0007 Key Decisions Summary

### **Scope Reduction** ✅
- **REMOVE**: Training, datasets, multi-modal fusion → separate repos
- **KEEP**: Export → compile → serve pipeline only
- **FOCUS**: TCN-VAE encoder → .hef → HailoRT sidecar

### **API Contract** (Non-negotiable)
```python
# POST /infer
Request:  {"x": [[Float; 9]] * 100}     # shape (100,9)
Response: {"latent":[Float;64], "motif_scores":[Float;M]}

# GET /healthz
Response: {"ok": bool, "model": str}

# GET /metrics  
Response: Prometheus format
```

### **EdgeInfer Integration**
- **Service**: `hailo-inference:9000`
- **Environment**: `MODEL_BACKEND_URL=http://hailo-inference:9000`
- **Feature Flag**: `USE_REAL_MODEL=true/false`
- **Fallback**: Stub responses when disabled

### **Performance SLOs**
- **Latency**: p95 < 50ms per window
- **Throughput**: ≥20 windows/sec sustained
- **Success Rate**: ≥99.5% under load

## 🏗️ Implementation Architecture

### **Directory Structure** (ADR-Compliant)
```
hailo_pipeline/
├── src/
│   ├── onnx_export/          # PyTorch → ONNX
│   │   ├── tcn_encoder_export.py
│   │   ├── export_config.yaml
│   │   └── validate_onnx.py
│   ├── hailo_compilation/    # ONNX → .hef (DFC)
│   │   ├── compile_tcn_model.py
│   │   ├── hailo_config.yaml
│   │   └── calibration_data.py
│   ├── runtime/              # FastAPI HailoRT sidecar
│   │   ├── app.py            # Main FastAPI app
│   │   ├── api_endpoints.py  # /healthz, /infer, /metrics
│   │   ├── model_loader.py   # HailoRT integration
│   │   ├── schemas.py        # Pydantic models
│   │   └── metrics.py        # Prometheus metrics
│   └── deployment/           # Docker + Pi deploy
│       ├── Dockerfile
│       ├── docker-compose.yml
│       └── pi_deploy.sh
├── exports/                  # *.onnx files
├── artifacts/                # *.hef files
├── telemetry/                # *.ndjson metrics
└── configs/
    ├── export_config.yaml
    ├── hailo_optimization.yaml
    └── runtime_config.yaml
```

### **Service Architecture**
```mermaid
graph LR
    A[TCN-VAE Model] --> B[ONNX Export]
    B --> C[Hailo DFC Compile]
    C --> D[.hef Artifacts]
    D --> E[HailoRT Sidecar]
    E --> F[EdgeInfer /infer]
    F --> G[iOS App]
```

## 🚀 Phase 1: FastAPI Sidecar Skeleton

### **1.1 Pydantic Models & Contract**
```python
# src/runtime/schemas.py
from pydantic import BaseModel, conlist, validator
from typing import List
import math

class IMUWindow(BaseModel):
    """IMU window with exactly 100 timesteps × 9 channels"""
    x: conlist(
        conlist(float, min_items=9, max_items=9), 
        min_items=100, max_items=100
    )
    
    @validator('x')
    def validate_finite_values(cls, v):
        for timestep in v:
            for value in timestep:
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite value detected: {value}")
        return v

class InferResponse(BaseModel):
    """Model inference output"""
    latent: List[float]        # 64-dim embedding
    motif_scores: List[float]  # M motif scores (configurable)

class HealthResponse(BaseModel):
    """Health check response"""
    ok: bool
    model: str
    version: str = "1.0.0"
```

### **1.2 FastAPI Application**
```python
# src/runtime/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging

from .api_endpoints import router
from .model_loader import ModelLoader
from .metrics import setup_prometheus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HailoRT TCN Inference Sidecar",
    version="1.0.0",
    description="EdgeInfer-compatible TCN-VAE inference on Hailo-8"
)

# CORS for EdgeInfer communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Global model loader
model_loader: ModelLoader = None

@app.on_event("startup")
async def startup_event():
    global model_loader
    
    hef_path = os.getenv("HEF_PATH", "artifacts/tcn_encoder.hef")
    num_motifs = int(os.getenv("NUM_MOTIFS", "12"))
    
    logger.info(f"Loading HEF model: {hef_path}")
    
    try:
        model_loader = ModelLoader(hef_path, num_motifs)
        await model_loader.initialize()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        model_loader = None

@app.on_event("shutdown") 
async def shutdown_event():
    global model_loader
    if model_loader:
        await model_loader.cleanup()

# Include API routes
app.include_router(router)

# Health check for model state
@app.get("/")
async def root():
    return {"status": "HailoRT Sidecar", "ready": model_loader is not None}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=9000,
        log_level="info"
    )
```

### **1.3 API Endpoints**
```python
# src/runtime/api_endpoints.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
import time
import logging

from .schemas import IMUWindow, InferResponse, HealthResponse
from .model_loader import ModelLoader
from .metrics import request_duration, inference_counter, model_loaded_gauge

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """EdgeInfer health check endpoint"""
    from .app import model_loader
    
    if model_loader and model_loader.is_ready():
        return HealthResponse(
            ok=True,
            model=model_loader.model_name
        )
    else:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded or unavailable"
        )

@router.post("/infer", response_model=InferResponse)
async def infer_imu_window(window: IMUWindow, request: Request):
    """EdgeInfer-compatible inference endpoint"""
    from .app import model_loader
    
    start_time = time.time()
    
    if not model_loader or not model_loader.is_ready():
        inference_counter.labels(status="model_unavailable").inc()
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    try:
        # Run inference
        latent, motif_scores = await model_loader.infer(window.x)
        
        response = InferResponse(
            latent=latent.tolist(),
            motif_scores=motif_scores.tolist()
        )
        
        # Record metrics
        duration = time.time() - start_time
        request_duration.observe(duration)
        inference_counter.labels(status="success").inc()
        
        logger.debug(f"Inference completed in {duration:.3f}s")
        return response
        
    except Exception as e:
        inference_counter.labels(status="error").inc()
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )

@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    from .metrics import generate_metrics
    return generate_metrics()
```

## 🔧 Phase 2: HailoRT Integration

### **2.1 Model Loader with HailoRT**
```python
# src/runtime/model_loader.py
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List
import asyncio

# HailoRT imports (placeholder - actual SDK needed)
try:
    from hailo_platform import (
        VDevice, HefModel, InferInterface,
        InputVStreamParams, OutputVStreamParams
    )
    HAILORT_AVAILABLE = True
except ImportError:
    HAILORT_AVAILABLE = False
    logging.warning("HailoRT SDK not available - using stub mode")

logger = logging.getLogger(__name__)

class ModelLoader:
    """HailoRT model loader and inference engine"""
    
    def __init__(self, hef_path: str, num_motifs: int = 12):
        self.hef_path = Path(hef_path)
        self.num_motifs = num_motifs
        self.model_name = self.hef_path.stem
        self.device = None
        self.model = None
        self.infer_interface = None
        self._ready = False
        
    async def initialize(self):
        """Initialize Hailo device and load model"""
        if not HAILORT_AVAILABLE:
            logger.warning("Using stub inference - HailoRT not available")
            self._ready = True
            return
            
        if not self.hef_path.exists():
            raise FileNotFoundError(f"HEF file not found: {self.hef_path}")
            
        try:
            # Initialize Hailo device
            self.device = VDevice()
            
            # Load HEF model
            self.model = HefModel(str(self.hef_path))
            
            # Create inference interface
            self.infer_interface = InferInterface(
                self.model, 
                device=self.device
            )
            
            # Configure input/output streams
            input_params = InputVStreamParams.from_model(self.model)
            output_params = OutputVStreamParams.from_model(self.model) 
            
            await self.infer_interface.configure_streams(
                input_params, output_params
            )
            
            self._ready = True
            logger.info(f"✅ HEF model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hailo model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self._ready
    
    async def infer(self, imu_window: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on IMU window
        
        Args:
            imu_window: List of 100 timesteps × 9 channels
            
        Returns:
            (latent_embeddings, motif_scores) as numpy arrays
        """
        if not self._ready:
            raise RuntimeError("Model not initialized")
            
        # Convert to numpy array with correct shape
        input_data = np.array(imu_window, dtype=np.float32)
        
        if not HAILORT_AVAILABLE:
            # Stub inference for testing
            latent = np.random.randn(64).astype(np.float32)
            motif_scores = np.random.rand(self.num_motifs).astype(np.float32)
            return latent, motif_scores
        
        try:
            # Prepare input batch (add batch dimension)
            batch_input = input_data.reshape(1, 100, 9)
            
            # Run inference on Hailo
            outputs = await self.infer_interface.infer({
                "imu_window": batch_input
            })
            
            # Extract outputs (remove batch dimension)
            latent = outputs["latent_embeddings"][0]  # (64,)
            motif_scores = outputs["motif_scores"][0]  # (M,)
            
            return latent, motif_scores
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    async def cleanup(self):
        """Clean up resources"""
        if self.infer_interface:
            await self.infer_interface.cleanup()
        if self.device:
            self.device.release()
        self._ready = False
```

## 🐳 Phase 3: Docker Deployment

### **3.1 Dockerfile**
```dockerfile
# src/deployment/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Hailo runtime (when available)
# COPY hailo_runtime.deb /tmp/
# RUN dpkg -i /tmp/hailo_runtime.deb || true

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/runtime/ ./src/runtime/
COPY configs/ ./configs/

# Create directories for models and telemetry
RUN mkdir -p artifacts telemetry

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9000/healthz || exit 1

# Expose port
EXPOSE 9000

# Run application
CMD ["python", "-m", "src.runtime.app"]
```

### **3.2 Docker Compose Integration**
```yaml
# src/deployment/docker-compose.yml
version: '3.8'

services:
  hailo-inference:
    build:
      context: ../..
      dockerfile: src/deployment/Dockerfile
    container_name: hailo-tcn-inference
    ports:
      - "9000:9000"
    volumes:
      - ../../artifacts:/app/artifacts:ro
      - ../../telemetry:/app/telemetry:rw
    devices:
      - /dev/hailo0:/dev/hailo0  # Hailo device mapping
    environment:
      - HEF_PATH=/app/artifacts/tcn_encoder.hef
      - NUM_MOTIFS=12
      - LOG_LEVEL=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s
```

## ⚡ Phase 4: ONNX Export Pipeline

### **4.1 TCN-VAE Export Script**
```python
# src/onnx_export/tcn_encoder_export.py
import torch
import torch.onnx
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_export_config(config_path: str = "configs/export_config.yaml") -> Dict[str, Any]:
    """Load export configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def export_tcn_encoder_to_onnx(
    model_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> bool:
    """
    Export TCN encoder to ONNX format
    
    Args:
        model_path: Path to trained TCN-VAE model (.pth)
        output_path: Output path for ONNX model
        config: Export configuration
        
    Returns:
        True if export successful
    """
    try:
        # Load trained model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract encoder if full TCN-VAE model
        if hasattr(checkpoint, 'encoder'):
            model = checkpoint.encoder
        else:
            model = checkpoint
            
        model.eval()
        
        # Create dummy input with fixed shape
        input_shape = config['input_shape']  # [1, 100, 9]
        dummy_input = torch.randn(*input_shape)
        
        logger.info(f"Exporting to ONNX with input shape: {input_shape}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=config['opset_version'],  # 11
            do_constant_folding=True,
            input_names=['imu_window'],
            output_names=['latent_embeddings', 'motif_scores'],
            dynamic_axes=config.get('dynamic_axes', {})
        )
        
        logger.info(f"✅ ONNX export completed: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX export failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Input TCN-VAE model path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--config", default="configs/export_config.yaml")
    
    args = parser.parse_args()
    
    config = load_export_config(args.config)
    success = export_tcn_encoder_to_onnx(args.model, args.output, config)
    
    exit(0 if success else 1)
```

## 📊 Success Validation

### **Two-Day Checklist** (ADR-0007)
**Day 1**:
- [ ] ✅ Branch + skeleton created
- [ ] ✅ API contract documented  
- [ ] ✅ ONNX export working
- [ ] ✅ First .hef compiled
- [ ] ✅ Docker compose up successful
- [ ] ✅ Curl smoke tests pass

**Day 2**:
- [ ] EdgeInfer integration via env
- [ ] Telemetry + AL stub
- [ ] CI contract + image build
- [ ] Self-hosted HEF job planned

### **Integration Testing**
```bash
# Health check
curl -s http://localhost:9000/healthz | jq .

# Inference smoke test (100x9 zeros)
curl -s -X POST http://localhost:9000/infer \
  -H 'Content-Type: application/json' \
  -d '{"x": ['"$(yes '[0,0,0,0,0,0,0,0,0],' | head -n 99)"'[0,0,0,0,0,0,0,0,0]]}' \
  | jq .

# EdgeInfer integration test
MODEL_BACKEND_URL=http://localhost:9000 USE_REAL_MODEL=true \
docker-compose -f ../pisrv_vapor_docker/docker-compose.yml restart edge-infer
```

This implementation roadmap follows ADR-0007 exactly, creating a focused pipeline from your proven TCN-VAE models to production Hailo inference serving EdgeInfer's API contract.