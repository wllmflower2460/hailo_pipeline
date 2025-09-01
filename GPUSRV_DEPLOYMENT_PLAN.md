# GPUSrv Deployment Plan for Hailo Pipeline
**Date**: 2025-08-31  
**Current Status**: Mac setup complete, ready for GPUSrv migration  
**Goal**: Deploy hailo_pipeline to GPUSrv with RTX 2060 for accelerated development  

## 🔍 Current Project Analysis

### **Local Mac Setup (Completed)**
- ✅ Repository cloned to `/Users/willflower/Documents/Projects/hailo_pipeline/`
- ✅ Python virtual environment created
- ✅ PyTorch + NumPy installed (CPU-only for Mac)
- ✅ Project structure analyzed

### **Current Implementation Status**
**Core Components Identified**:
- **Docker/Podman Integration**: Complete containerized Hailo development environment
- **Dog Detection Pipeline**: hailo_dog_detection.py with inference capabilities
- **Jupyter Environment**: Available at localhost:8889 with hailo token
- **Performance Testing**: CPU/GPU benchmark scripts in `src/`
- **ROCm Diagnostics**: Comprehensive AMD GPU testing utilities

**Key Findings**:
- Project designed for **AMD Radeon 780M** but GPU support incomplete
- **CPU-only fallback** currently functional  
- **Containerized workflow** ready for deployment
- **TCN-VAE integration missing** - needs connection to EdgeInfer work

## 🚀 GPUSrv Migration Strategy

### **Phase 1: Repository Migration (Day 1)**
```bash
# On GPUSrv
cd /opt/
git clone https://github.com/wllmflower2460/hailo_pipeline.git
cd hailo_pipeline

# Setup virtual environment with CUDA support
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support for RTX 2060
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install jupyter matplotlib scipy scikit-learn
```

### **Phase 2: GPU Environment Validation (Day 1)**
```bash
# Test NVIDIA GPU instead of AMD
python src/test_gpu_performance.py  # Modify for CUDA
nvidia-smi  # Verify RTX 2060 visibility

# Update environment scripts for CUDA
sed -i 's/ROCm/CUDA/g' setup_gpu.sh
sed -i 's/HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES/g' setup_env.sh
```

### **Phase 3: TCN-VAE Integration (Day 2-3)**
```bash
# Connect to your existing TCN-VAE work
mkdir -p models/tcn_vae/
# Copy best_tcn_vae_57pct.pth from TCN-VAE_models repo

# Create TCN-VAE training pipeline
cat > src/tcn_vae_training.py << 'EOF'
import torch
import torch.nn as nn
from pathlib import Path

class TCNVAETrainer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load existing 57.68% accuracy model or train new one
        
    def train_enhanced_model(self):
        # Enhanced training with RTX 2060 capabilities
        # Target: >60% accuracy with larger batch sizes
        pass
EOF
```

### **Phase 4: Hailo Integration Pipeline (Day 3-5)**
```bash
# Create ONNX export pipeline
cat > src/export_to_hailo.py << 'EOF'
import torch
import torch.onnx
from models.tcn_vae import TCNVAEEncoder

def export_tcn_encoder_to_onnx():
    # Load trained model
    model = torch.load('models/tcn_vae/best_tcn_vae_57pct.pth')
    encoder = model.encoder  # Extract encoder only
    
    # Export to ONNX with fixed input shape
    dummy_input = torch.randn(1, 100, 9)  # [batch, time, features]
    torch.onnx.export(
        encoder, 
        dummy_input,
        'models/hailo/tcn_encoder.onnx',
        input_names=['imu_data'],
        output_names=['latent_embedding']
    )
    
if __name__ == '__main__':
    export_tcn_encoder_to_onnx()
EOF
```

## 📊 GPUSrv Advantages for This Project

### **Hardware Benefits**
- **RTX 2060 (6GB VRAM)**: 4x more GPU memory than integrated solutions
- **CUDA Ecosystem**: Mature PyTorch + CUDA integration vs experimental ROCm
- **2TB SSD**: Massive storage for model experiments and datasets
- **Clean Environment**: Dedicated ML system without development conflicts

### **Development Workflow Improvements**
```bash
# Enhanced batch sizes possible with RTX 2060
BATCH_SIZE=64  # vs 16 on CPU/integrated GPU
SEQUENCE_LENGTH=100
LATENT_DIM=64

# Parallel training experiments
tmux new-session -d -s tcn_training_64dim
tmux new-session -d -s tcn_training_128dim
tmux new-session -d -s tcn_training_256dim
```

### **Performance Expectations**
- **Training Speed**: 5-10x faster than CPU-only
- **Model Experimentation**: Multiple variants in parallel
- **Larger Models**: Enable bigger TCN architectures
- **Hyperparameter Search**: Automated grid search feasible

## 🔗 Integration with EdgeInfer Pipeline

### **Connection Points**
```mermaid
graph LR
    A[GPUSrv TCN Training] --> B[ONNX Export]
    B --> C[Hailo DFC Compile]
    C --> D[Pi EdgeInfer Deploy]
    D --> E[iOS Integration]
```

### **Automated Pipeline**
```bash
# GPUSrv → Pi deployment workflow
cat > deploy_to_pi.sh << 'EOF'
#!/bin/bash
# 1. Train enhanced model on GPUSrv
python src/tcn_vae_training.py --target_accuracy 0.65

# 2. Export to ONNX
python src/export_to_hailo.py

# 3. Compile for Hailo (if Hailo SDK available)
# hailo compile tcn_encoder.onnx --config hailo_config.yaml

# 4. Deploy to Pi EdgeInfer
rsync -av models/hailo/ pi@pisrv.local:/home/pi/pisrv_vapor_docker/EdgeInfer/models/

# 5. Update EdgeInfer feature flag
ssh pi@pisrv.local "cd pisrv_vapor_docker && docker-compose restart edge-infer"
EOF
```

## ⚠️ Migration Risks & Mitigations

### **Technical Risks**
1. **CUDA vs ROCm Code**: Project designed for AMD, needs NVIDIA adaptation
2. **Container Dependencies**: Docker/Podman configuration may need updates
3. **Model Compatibility**: Ensure trained models work across environments

### **Mitigation Strategies**
```bash
# 1. Gradual migration approach
# Keep Mac environment as fallback during GPUSrv setup

# 2. Containerized consistency
# Use Docker on GPUSrv to match development environment

# 3. Model validation pipeline
python test_model_compatibility.py --source mac --target gpusrv
```

## 📋 Action Items for GPUSrv Deployment

### **Immediate (This Week)**
- [ ] SSH access to GPUSrv configured
- [ ] Clone repository to `/opt/hailo_pipeline/`
- [ ] Install CUDA PyTorch environment
- [ ] Test RTX 2060 with existing benchmark scripts

### **Short-term (Next Week)**
- [ ] Adapt ROCm scripts for CUDA
- [ ] Import TCN-VAE models from existing work
- [ ] Enhanced training pipeline with larger batches
- [ ] ONNX export workflow for Hailo integration

### **Medium-term (Following Weeks)**
- [ ] Hailo DFC compilation pipeline (if SDK available)
- [ ] Automated GPUSrv → Pi deployment
- [ ] Integration with EdgeInfer docker-compose
- [ ] Performance benchmarking vs current 57.68% baseline

## 🎯 Success Metrics

### **Technical Goals**
- [ ] **GPU Utilization**: >80% during training sessions
- [ ] **Model Accuracy**: >60% validation (vs current 57.68%)
- [ ] **Training Speed**: <30 minutes for full training run
- [ ] **Deployment Pipeline**: One-command GPUSrv → Pi deployment

### **Operational Goals**
- [ ] **Reliability**: 95% successful training completion rate
- [ ] **Automation**: Fully scripted training + deployment
- [ ] **Monitoring**: Real-time training progress tracking
- [ ] **Documentation**: Complete setup and troubleshooting guides

This migration plan transforms the hailo_pipeline from a development concept into a production-ready training and deployment pipeline, leveraging your GPUSrv's superior hardware capabilities while integrating with your existing EdgeInfer infrastructure.