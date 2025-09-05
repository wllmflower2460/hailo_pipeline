#!/bin/bash
set -e

# Edge Platform Setup for TCN-VAE 72.13% Model Deployment
# Sets up EdgeInfer infrastructure on Raspberry Pi 5 + Hailo-8

EDGE_HOST="192.168.50.88"
EDGE_USER="pi"
REPO_URL="https://github.com/your-username/hailo_pipeline.git"  # Update with actual repo URL

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

print_header() {
    echo "================================================================================"
    echo -e "${BLUE}🔧 Edge Platform Setup for TCN-VAE Deployment${NC}"
    echo "   Target: ${EDGE_HOST}"
    echo "   User: ${EDGE_USER}"
    echo "   Purpose: EdgeInfer infrastructure setup"
    echo "================================================================================"
}

setup_directories() {
    log "Setting up directory structure on Edge platform..."
    
    ssh "${EDGE_USER}@${EDGE_HOST}" "
        # Create deployment directories
        sudo mkdir -p /opt/edgeinfer/artifacts
        sudo mkdir -p /opt/edgeinfer/logs
        sudo mkdir -p /opt/edgeinfer/config
        
        # Set ownership
        sudo chown -R ${EDGE_USER}:${EDGE_USER} /opt/edgeinfer
        
        # Create home directory structure
        mkdir -p ~/Development
        mkdir -p ~/logs
        
        echo 'Directory structure created'
    "
    
    success "Directory structure created"
}

install_dependencies() {
    log "Installing required dependencies on Edge platform..."
    
    ssh "${EDGE_USER}@${EDGE_HOST}" "
        # Update system
        sudo apt update
        
        # Install essential packages
        sudo apt install -y git curl wget docker.io docker-compose
        
        # Add user to docker group
        sudo usermod -aG docker ${EDGE_USER}
        
        # Install Python dependencies
        sudo apt install -y python3-pip python3-venv
        pip3 install --user requests numpy
        
        echo 'Dependencies installed'
    "
    
    success "Dependencies installed"
}

setup_docker_environment() {
    log "Setting up Docker environment for EdgeInfer..."
    
    ssh "${EDGE_USER}@${EDGE_HOST}" "
        cd /opt/edgeinfer
        
        # Create docker-compose.yml for EdgeInfer
        cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  edgeinfer-sidecar:
    image: edgeinfer:latest
    container_name: edgeinfer-sidecar
    restart: unless-stopped
    ports:
      - \"9000:9000\"
    volumes:
      - ./artifacts:/app/artifacts:ro
      - ./logs:/app/logs
      - ./config:/app/config:ro
    environment:
      - MODEL_VERSION=v72pct
      - MODEL_ACCURACY=72.13%
      - HEF_PATH=/app/artifacts/tcn_encoder_v72pct.hef
      - HAILO_DEVICE_ID=0
      - BATCH_SIZE=1
      - LOG_LEVEL=INFO
    devices:
      - /dev/hailo0:/dev/hailo0
    command: [\"python\", \"app.py\", \"--model-path\", \"/app/artifacts/tcn_encoder_v72pct.hef\"]
    
  # Health check service
  healthcheck:
    image: curlimages/curl:latest
    container_name: edgeinfer-healthcheck
    depends_on:
      - edgeinfer-sidecar
    command: >
      sh -c \"
      while true; do
        sleep 30;
        curl -f http://edgeinfer-sidecar:9000/healthz || echo 'Health check failed';
      done\"
    restart: unless-stopped

networks:
  default:
    name: edgeinfer-network
EOF
        
        echo 'Docker Compose configuration created'
    "
    
    success "Docker environment configured"
}

create_edgeinfer_app() {
    log "Creating basic EdgeInfer application..."
    
    ssh "${EDGE_USER}@${EDGE_HOST}" "
        cd /opt/edgeinfer
        
        # Create basic FastAPI app for EdgeInfer
        cat > app.py << 'EOF'
#!/usr/bin/env python3
\"\"\"
Basic EdgeInfer FastAPI Application for TCN-VAE Model
Serves as a placeholder until full EdgeInfer implementation
\"\"\"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import numpy as np
import logging
from typing import List
import os

app = FastAPI(title=\"EdgeInfer TCN-VAE\", version=\"1.0.0\")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IMURequest(BaseModel):
    imu_window: List[List[float]]

class EmbeddingResponse(BaseModel):
    latent_embedding: List[float]
    model_version: str
    accuracy: str

@app.get(\"/healthz\")
async def health_check():
    \"\"\"Health check endpoint\"\"\"
    return {
        \"status\": \"healthy\",
        \"model_version\": os.getenv(\"MODEL_VERSION\", \"v72pct\"),
        \"model_accuracy\": os.getenv(\"MODEL_ACCURACY\", \"72.13%\"),
        \"hef_path\": os.getenv(\"HEF_PATH\", \"/app/artifacts/tcn_encoder_v72pct.hef\")
    }

@app.post(\"/encode\", response_model=EmbeddingResponse)
async def encode_imu_data(request: IMURequest):
    \"\"\"Encode IMU data to latent embeddings\"\"\"
    try:
        # Validate input shape
        imu_data = np.array(request.imu_window)
        
        if imu_data.shape != (100, 9):
            raise HTTPException(
                status_code=400, 
                detail=f\"Invalid input shape {imu_data.shape}. Expected (100, 9)\"
            )
        
        # TODO: Replace with actual Hailo inference
        # For now, return mock 64-dimensional embedding
        mock_embedding = np.random.normal(0, 1, 64).tolist()
        
        logger.info(f\"Processed IMU window shape: {imu_data.shape}\")
        
        return EmbeddingResponse(
            latent_embedding=mock_embedding,
            model_version=os.getenv(\"MODEL_VERSION\", \"v72pct\"),
            accuracy=os.getenv(\"MODEL_ACCURACY\", \"72.13%\")
        )
        
    except Exception as e:
        logger.error(f\"Encoding failed: {e}\")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == \"__main__\":
    import uvicorn
    uvicorn.run(app, host=\"0.0.0.0\", port=9000)
EOF

        # Create requirements.txt
        cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
requests==2.31.0
EOF

        echo 'EdgeInfer application created'
    "
    
    success "EdgeInfer application created"
}

setup_systemd_service() {
    log "Setting up systemd service for EdgeInfer..."
    
    ssh "${EDGE_USER}@${EDGE_HOST}" "
        # Create systemd service file
        sudo tee /etc/systemd/system/edgeinfer.service > /dev/null << 'EOF'
[Unit]
Description=EdgeInfer TCN-VAE Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/edgeinfer
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0
User=pi

[Install]
WantedBy=multi-user.target
EOF

        # Enable service
        sudo systemctl daemon-reload
        sudo systemctl enable edgeinfer
        
        echo 'SystemD service configured'
    "
    
    success "SystemD service configured"
}

clone_hailo_pipeline() {
    log "Setting up hailo_pipeline repository..."
    
    ssh "${EDGE_USER}@${EDGE_HOST}" "
        cd ~/Development
        
        # For now, create a basic structure since we don't have the repo URL
        # This will be replaced when you have the actual repo
        mkdir -p hailo_pipeline/{models,artifacts,configs,src}
        
        cd hailo_pipeline
        
        # Create basic README
        cat > README.md << 'EOF'
# Hailo Pipeline - Edge Platform

This is the Edge platform setup for TCN-VAE 72.13% model deployment.

## Structure
- models/     - Model artifacts (ONNX, HEF)
- artifacts/  - Deployment artifacts 
- configs/    - Configuration files
- src/        - Source code
EOF
        
        echo 'Hailo pipeline structure created'
    "
    
    success "Hailo pipeline structure created"
}

test_setup() {
    log "Testing Edge platform setup..."
    
    ssh "${EDGE_USER}@${EDGE_HOST}" "
        # Test directory structure
        echo 'Directory structure:'
        ls -la /opt/edgeinfer/
        
        # Test Docker
        echo 'Docker version:'
        docker --version
        
        # Test Python
        echo 'Python version:'
        python3 --version
        
        # Test if Hailo device is present (may fail if not installed)
        echo 'Checking for Hailo device:'
        ls -la /dev/hailo* 2>/dev/null || echo 'No Hailo device found (install HailoRT)'
        
        echo 'Setup test completed'
    "
    
    success "Edge platform setup test completed"
}

main() {
    print_header
    
    log "Setting up Edge platform for TCN-VAE deployment..."
    
    setup_directories
    install_dependencies
    setup_docker_environment
    create_edgeinfer_app
    setup_systemd_service
    clone_hailo_pipeline
    test_setup
    
    echo ""
    echo "================================================================================"
    success "🎉 Edge Platform Setup Complete!"
    echo "   EdgeInfer infrastructure ready at: ${EDGE_HOST}"
    echo "   Directory: /opt/edgeinfer/"
    echo "   Service: systemctl status edgeinfer"
    echo "================================================================================"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Install HailoRT drivers on Edge platform"
    echo "   2. Build/pull EdgeInfer Docker image"  
    echo "   3. Deploy TCN-VAE model: ./deploy_to_edge.sh deploy"
    echo "   4. Start services: ssh ${EDGE_USER}@${EDGE_HOST} 'sudo systemctl start edgeinfer'"
    echo ""
}

# Handle command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "test")
        log "Testing Edge platform connectivity..."
        ssh "${EDGE_USER}@${EDGE_HOST}" "echo 'SSH connection successful'"
        success "Edge platform connectivity test passed"
        ;;
    "clean")
        log "Cleaning up Edge platform setup..."
        ssh "${EDGE_USER}@${EDGE_HOST}" "
            sudo systemctl stop edgeinfer || true
            sudo systemctl disable edgeinfer || true  
            sudo rm -f /etc/systemd/system/edgeinfer.service
            sudo rm -rf /opt/edgeinfer
            rm -rf ~/Development/hailo_pipeline
        "
        success "Edge platform cleaned up"
        ;;
    *)
        echo "Usage: $0 [setup|test|clean]"
        echo "  setup - Set up Edge platform infrastructure (default)"
        echo "  test  - Test SSH connectivity to Edge platform"
        echo "  clean - Clean up Edge platform setup"
        exit 1
        ;;
esac