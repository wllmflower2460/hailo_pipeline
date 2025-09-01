# Hardware Deployment Guide: Raspberry Pi + Hailo-8

🎯 **Production deployment of TCN-VAE inference on Hailo-8 accelerator**

## 🔧 **Hardware Requirements**

### **Target Hardware Stack**
- **Raspberry Pi 5** (8GB RAM recommended)
- **Hailo-8 AI Accelerator** (M.2 or HAT+ form factor)
- **High-speed microSD** (64GB Class 10 or better)
- **Power Supply** (27W official Pi 5 PSU or 65W USB-C)
- **Cooling** (Active cooling fan for sustained performance)

### **Performance Targets**
- **Inference Latency**: p95 < 50ms per IMU window
- **Throughput**: ≥20 windows/sec sustained  
- **Memory Usage**: <512MB Pi RAM for inference
- **Power Consumption**: <25W total system (Pi + Hailo)

## 🐧 **Operating System Setup**

### **Raspberry Pi OS Configuration**
```bash
# Use Raspberry Pi OS Bookworm (64-bit)
# Download: https://www.raspberrypi.com/software/

# Enable necessary features
sudo raspi-config
# Advanced Options > Expand Filesystem
# Interface Options > Enable SSH
# Advanced Options > Memory Split > 128MB GPU

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    docker.io docker-compose \
    build-essential cmake \
    python3-pip python3-venv \
    git curl wget \
    htop iotop \
    ufw
```

### **Hailo-8 Driver Installation**
```bash
# Download Hailo runtime from https://hailo.ai/developer-zone/
# Note: Requires Hailo developer account

# Install HailoRT runtime (example version)
wget https://hailo.ai/downloads/hailort-4.17.0-linux-aarch64.tar.gz
tar -xzf hailort-4.17.0-linux-aarch64.tar.gz
cd hailort-4.17.0-linux-aarch64

# Install runtime
sudo dpkg -i hailort_4.17.0_arm64.deb
sudo apt-get install -f  # Fix dependencies if needed

# Install Python bindings
pip3 install hailo_platform-4.17.0-cp39-cp39-linux_aarch64.whl

# Verify installation
hailo fw-control identify
# Should show Hailo-8 device information
```

### **Device Permissions**
```bash
# Add user to hailo group
sudo usermod -a -G hailo $USER

# Set device permissions
sudo udevadm control --reload-rules
sudo udevadm trigger

# Verify device access
ls -la /dev/hailo0
# Should show: crw-rw---- 1 root hailo 511, 0 ...

# Test device access
hailo scan
# Should list connected Hailo devices
```

## 🚀 **Pipeline Deployment**

### **1. Clone and Setup Repository**
```bash
# Clone the hailo_pipeline repository
git clone <repository-url> hailo_pipeline
cd hailo_pipeline

# Switch to implementation branch
git checkout feature/hailo-sidecar-implementation

# Create directories for artifacts
mkdir -p artifacts exports telemetry
```

### **2. Model Artifacts Preparation**
```bash
# Download TCN-VAE models from releases
# Replace with actual release URL from TCN-VAE_models repo
wget https://github.com/wllmflower2460/TCN-VAE_models/releases/download/v0.1.0/tcn_encoder_for_edgeinfer.pth
mv tcn_encoder_for_edgeinfer.pth models/

# Export to ONNX format
python src/onnx_export/tcn_encoder_export.py \
    --model models/tcn_encoder_for_edgeinfer.pth \
    --version v0.1.0

# Validate ONNX export
python src/onnx_export/validate_onnx.py \
    --model exports/tcn_encoder_v0.1.0.onnx \
    --benchmark

# Compile to Hailo HEF format
python src/hailo_compilation/compile_tcn_model.py \
    --onnx exports/tcn_encoder_v0.1.0.onnx \
    --version v0.1.0 \
    --output-report telemetry/compilation_report.json
```

### **3. Docker Deployment**
```bash
# Build production Docker image
docker build -f src/deployment/Dockerfile -t hailo-tcn-inference .

# Deploy with docker-compose
docker-compose -f src/deployment/docker-compose.yml up -d

# Verify deployment
docker ps
docker logs hailo-tcn-inference

# Check health status
curl -f http://localhost:9000/healthz
```

### **4. Performance Validation**
```bash
# Run comprehensive performance tests
python test_sidecar.py --test all --url http://localhost:9000

# Benchmark inference performance
python test_sidecar.py --test load --url http://localhost:9000

# Monitor system resources during load
htop  # CPU and memory usage
iotop # Disk I/O usage

# Check Hailo device utilization
hailo monitor  # If available
```

## 🔗 **EdgeInfer Integration**

### **Environment Configuration**
```bash
# In pisrv_vapor_docker/.env
MODEL_BACKEND_URL=http://hailo-inference:9000/infer
USE_REAL_MODEL=true
BACKEND_TIMEOUT_MS=100
```

### **Network Configuration**
```yaml
# Add to pisrv_vapor_docker/docker-compose.yml
networks:
  - edgeinfer-network

# Ensure hailo-inference service is on same network
```

### **Integration Testing**
```bash
# Test EdgeInfer → Hailo communication
# From pisrv_vapor_docker directory:

# Start EdgeInfer with Hailo backend
USE_REAL_MODEL=true MODEL_BACKEND_URL=http://hailo-inference:9000 \
docker-compose up -d

# Test motif analysis endpoint
curl -s -X POST http://localhost:8080/api/v1/analysis/motifs \
  -H 'Content-Type: application/json' \
  -d '{"sessionId": "hardware-test"}' | jq .

# Expected response with real model predictions
{
  "motifs": [...],
  "confidence": 0.xx,
  "model_backend": "hailo-inference",
  "inference_time_ms": xx.x
}
```

## 📊 **Production Monitoring**

### **System Metrics Collection**
```bash
# Install monitoring stack (optional)
docker-compose -f src/deployment/docker-compose.yml \
  --profile monitoring up -d

# Access Grafana dashboard
# http://raspberry-pi:3000 (admin/hailo123)

# View Prometheus metrics
curl http://localhost:9000/metrics
```

### **Performance Monitoring**
```python
# Custom monitoring script
python3 << 'EOF'
import requests
import time
import json
from datetime import datetime

def monitor_hailo_performance():
    """Monitor Hailo inference performance"""
    
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "samples": []
    }
    
    # Generate test IMU data
    test_data = {
        "x": [[0.0] * 9 for _ in range(100)]  # Zero pattern
    }
    
    print("🔍 Starting Hailo performance monitoring...")
    
    for i in range(50):  # 50 test inferences
        start_time = time.time()
        
        try:
            response = requests.post(
                "http://localhost:9000/infer",
                json=test_data,
                timeout=1.0
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                
                sample = {
                    "sample": i + 1,
                    "latency_ms": latency_ms,
                    "status": "success",
                    "latent_range": [min(result["latent"]), max(result["latent"])],
                    "motif_range": [min(result["motif_scores"]), max(result["motif_scores"])]
                }
                
                print(f"Sample {i+1:2d}: {latency_ms:5.1f}ms - ✅")
            else:
                sample = {
                    "sample": i + 1,
                    "latency_ms": latency_ms,
                    "status": "error",
                    "error": response.status_code
                }
                print(f"Sample {i+1:2d}: {latency_ms:5.1f}ms - ❌ {response.status_code}")
            
            metrics["samples"].append(sample)
            
        except Exception as e:
            print(f"Sample {i+1:2d}: ERROR - {e}")
            
        time.sleep(0.1)  # 100ms between requests
    
    # Calculate statistics
    successful_samples = [s for s in metrics["samples"] if s["status"] == "success"]
    
    if successful_samples:
        latencies = [s["latency_ms"] for s in successful_samples]
        
        print(f"\n📊 Performance Results:")
        print(f"   Successful samples: {len(successful_samples)}/50")
        print(f"   Mean latency: {sum(latencies)/len(latencies):.2f}ms")
        print(f"   P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
        print(f"   Max latency: {max(latencies):.2f}ms")
        print(f"   Min latency: {min(latencies):.2f}ms")
        
        # Check if targets are met
        p95_latency = sorted(latencies)[int(len(latencies)*0.95)]
        mean_latency = sum(latencies)/len(latencies)
        
        if p95_latency < 50:
            print(f"   ✅ P95 latency target met: {p95_latency:.1f}ms < 50ms")
        else:
            print(f"   ❌ P95 latency target missed: {p95_latency:.1f}ms > 50ms")
            
        if mean_latency < 50:
            throughput = 1000 / mean_latency
            print(f"   ✅ Throughput estimate: {throughput:.1f} windows/sec")
        else:
            print(f"   ⚠️  High latency may impact throughput")
    
    # Save detailed metrics
    with open("telemetry/performance_test.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Detailed results saved: telemetry/performance_test.json")

if __name__ == "__main__":
    monitor_hailo_performance()
EOF
```

## 🛠 **Troubleshooting**

### **Common Issues**

#### **Hailo Device Not Found**
```bash
# Check device detection
lsusb | grep -i hailo
lspci | grep -i hailo

# Check driver loading
dmesg | grep hailo
sudo modprobe hailo_pci  # If needed

# Verify device permissions
ls -la /dev/hailo*
sudo chmod 666 /dev/hailo0  # Temporary fix
```

#### **Docker Permission Issues**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Restart Docker service
sudo systemctl restart docker

# Check Docker daemon status
sudo systemctl status docker
```

#### **Memory Issues**
```bash
# Check memory usage
free -h
cat /proc/meminfo

# Increase swap if needed
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Monitor Docker memory usage
docker stats hailo-tcn-inference
```

#### **Performance Issues**
```bash
# Check CPU frequency scaling
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
# Should be "performance" for maximum speed

# Set performance governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Monitor thermal throttling
vcgencmd measure_temp
# Keep below 70°C for optimal performance

# Check system load
uptime
iostat 1 5
```

## ✅ **Deployment Checklist**

### **Pre-Deployment**
- [ ] Raspberry Pi 5 with 8GB RAM
- [ ] Hailo-8 device properly seated and detected
- [ ] 64GB+ Class 10 microSD card
- [ ] Active cooling solution installed
- [ ] Network connectivity established
- [ ] SSH access configured

### **Software Installation**
- [ ] Raspberry Pi OS Bookworm (64-bit) installed
- [ ] HailoRT runtime installed and verified
- [ ] Docker and docker-compose installed
- [ ] Python 3.9+ with pip available
- [ ] Repository cloned and configured

### **Model Pipeline**
- [ ] TCN-VAE model downloaded from releases
- [ ] ONNX export completed successfully
- [ ] ONNX validation passed (>0.99 similarity)
- [ ] HEF compilation completed
- [ ] HEF validation passed

### **Deployment Validation**
- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] Health check endpoint responds
- [ ] Inference endpoint functional
- [ ] Performance targets met (<50ms p95)
- [ ] EdgeInfer integration working

### **Production Readiness**
- [ ] Monitoring metrics available
- [ ] Log aggregation configured
- [ ] Automated restart on failure
- [ ] Resource limits configured
- [ ] Security hardening applied
- [ ] Backup and recovery tested

## 🎯 **Success Criteria**

### **Functional Requirements**
- ✅ **Device Detection**: Hailo-8 recognized and accessible
- ✅ **Model Loading**: HEF model loads without errors  
- ✅ **Inference Working**: Produces valid 64-dim latent + 12 motifs
- ✅ **API Compliance**: EdgeInfer-compatible responses
- ✅ **Health Monitoring**: Reliable health checks

### **Performance Requirements**
- ✅ **Latency**: P95 < 50ms per inference
- ✅ **Throughput**: >20 windows/sec sustained
- ✅ **Memory**: <512MB Pi RAM usage
- ✅ **Stability**: 24/7 operation without crashes
- ✅ **Accuracy**: >95% similarity to ONNX reference

### **Integration Requirements**
- ✅ **EdgeInfer**: Seamless backend replacement
- ✅ **Docker**: Production container deployment
- ✅ **Monitoring**: Prometheus metrics collection
- ✅ **Logging**: Structured log output
- ✅ **Recovery**: Graceful error handling

## 🚀 **Next Steps: Production Scaling**

Once hardware deployment is validated:

1. **Fleet Deployment**: Scale to multiple Pi + Hailo units
2. **Load Balancing**: Distribute inference across devices  
3. **Edge Orchestration**: Kubernetes or Docker Swarm
4. **Model Updates**: Hot-swappable model deployment
5. **Telemetry Pipeline**: Centralized metrics and logging

**Ready to deploy on real hardware!** 🎯🤘