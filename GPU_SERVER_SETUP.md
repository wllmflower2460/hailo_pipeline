# GPU Server Setup Instructions

**Complete deployment guide for Hailo TCN Inference Pipeline (running directly on gpusrv)**

---

## 🖥️ **Prerequisites**

### **System Requirements**
- Ubuntu/Debian-based Linux distribution
- Python 3.9+ with pip
- Docker and docker-compose
- Git access to repository
- Network connectivity for package installation

### **Check System Compatibility**
```bash
# Verify system info
uname -a
cat /etc/os-release
python3 --version
docker --version
```

---

## 🚀 **Quick Start Deployment**

### **1. Clone Repository**
```bash
# Navigate to your projects directory
cd ~/projects  # or wherever you keep code

# Clone the hailo_pipeline repository
git clone <repository-url> hailo_pipeline
cd hailo_pipeline

# Switch to the feature branch with all enhancements
git checkout feature/hailo-sidecar-implementation

# Verify you have the latest commits
git log --oneline -3
# Should show: "feat: implement enhanced configuration management..."
```

### **2. Environment Setup**
```bash
# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install additional development dependencies
pip install pytest requests pyyaml numpy

# Verify installation
python -c "import fastapi, pydantic, uvicorn; print('✅ FastAPI stack ready')"
```

### **3. Configuration**
```bash
# Create necessary directories
mkdir -p {models,exports,artifacts,telemetry,logs}

# Set environment variables (add to ~/.bashrc for persistence)
export HEF_PATH="artifacts/tcn_encoder_stub.hef"
export NUM_MOTIFS="12"
export FASTAPI_HOST="0.0.0.0"
export FASTAPI_PORT="9000"
export LOG_LEVEL="info"

# Create a stub HEF file for testing (since we don't have real Hailo hardware)
echo "# Stub HEF file for development" > artifacts/tcn_encoder_stub.hef
```

---

## 🧪 **Testing the Implementation**

### **4. Start the FastAPI Sidecar**
```bash
# Activate virtual environment
source .venv/bin/activate

# Start the inference sidecar
cd src/runtime
python -m uvicorn app:app --host 0.0.0.0 --port 9000 --reload

# Should see:
# INFO: Uvicorn running on http://0.0.0.0:9000
# INFO: HailoRT TCN Inference Sidecar
# INFO: HailoRT not available - enabling stub inference mode
```

### **5. Test Enhanced Health Checks** (New Terminal)
```bash
# Test the enhanced health endpoint
curl -s http://localhost:9000/healthz | jq .

# Expected output with new fields:
# {
#   "ok": true,
#   "model": "tcn_encoder_stub",
#   "uptime_s": 123,
#   "config_version": "hailo_pipeline_production_config-2025-09-01", 
#   "hef_sha256": "file_not_found",
#   "version": "1.0.0",
#   "device": "hailo8",
#   "latency_ms": 0.0
# }
```

### **6. Test Enhanced Prometheus Metrics**
```bash
# Check the new build_info and config_ok metrics
curl -s http://localhost:9000/metrics | grep -E "hailo_build_info|hailo_config_ok"

# Expected output:
# hailo_build_info{version="v1.0.0",hef_sha="file_not_found",config="hailo_pipeline_production_config-2025-09-01"} 1.0
# hailo_config_ok{expected="hailo_pipeline_production_config-2025-09-01",actual="hailo_pipeline_production_config-2025-09-01"} 1.0
```

### **7. Test Inference Contract**
```bash
# Test the corrected JSON payload format (100x9)
payload=$(python3 - <<'PY'
import json
print(json.dumps({"x":[[0.0]*9 for _ in range(100)]}))
PY
)

# Send inference request
curl -s -X POST http://localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d "$payload" | jq .

# Expected output:
# {
#   "latent": [64 float values],
#   "motif_scores": [12 float values]
# }

# Validate contract compliance
resp=$(curl -s -X POST http://localhost:9000/infer -H "Content-Type: application/json" -d "$payload")
echo "$resp" | jq -e '.latent|length==64 and .motif_scores|length==12' >/dev/null && echo "✅ Contract validated" || echo "❌ Contract failed"
```

---

## 🐳 **Docker Deployment Testing**

### **8. Docker Build and Run**
```bash
# Build the production Docker image
docker build -f src/deployment/Dockerfile -t hailo-tcn-inference .

# Run the container
docker run -d \
  --name hailo-inference-test \
  -p 9000:9000 \
  -e HEF_PATH="/app/artifacts/tcn_encoder_stub.hef" \
  -e LOG_LEVEL="info" \
  hailo-tcn-inference

# Check container status
docker ps | grep hailo-inference
docker logs hailo-inference-test

# Test containerized service
curl -s http://localhost:9000/healthz | jq .
```

---

## ⚡ **Performance Validation**

### **9. Run Performance Tests**
```bash
# Make validation script executable
chmod +x validate_performance.py

# Run comprehensive performance validation
python3 validate_performance.py --target-url http://localhost:9000

# Expected output:
# 🔍 Starting Hailo performance validation...
# Testing latency performance...
# Testing throughput capacity...
# Testing concurrent load handling...
# 📊 Performance Results:
#   Mean latency: XX.Xms
#   P95 latency: XX.Xms  
#   Max throughput: XXX req/sec
```

---

## 🔧 **Development and Debugging**

### **10. Development Mode**
```bash
# For active development with auto-reload
cd src/runtime
python -m uvicorn app:app --host 0.0.0.0 --port 9000 --reload --log-level debug

# View detailed logs
tail -f logs/hailo_sidecar.log  # if logging to file

# Test API documentation
open http://localhost:9000/docs  # FastAPI auto-generated docs
```

### **11. Troubleshooting**
```bash
# Check Python environment
python3 -c "
import sys
print('Python version:', sys.version)
import fastapi, pydantic, uvicorn
print('✅ FastAPI stack installed')
"

# Check port availability
ss -tulpn | grep :9000  # Linux equivalent of netstat

# Check container logs if using Docker
docker logs hailo-inference-test --tail 50

# Restart services
pkill -f uvicorn  # Kill any running FastAPI instances
# or
docker stop hailo-inference-test && docker rm hailo-inference-test
```

---

## 🚀 **Production Deployment**

### **12. Docker Compose Deployment**
```bash
# Use docker-compose for production-like deployment
cd src/deployment

# Start full stack
docker-compose up -d

# Monitor services
docker-compose ps
docker-compose logs -f hailo-inference

# Scale if needed
docker-compose up -d --scale hailo-inference=2
```

### **13. Monitoring Setup**
```bash
# Check metrics endpoint
curl -s http://localhost:9000/metrics > current_metrics.txt
wc -l current_metrics.txt  # Should show many metric lines

# Monitor key metrics
watch -n 5 'curl -s http://localhost:9000/metrics | grep -E "hailo_(requests_total|inference_duration|build_info)"'
```

---

## ✅ **Validation Checklist**

After setup, verify these items work:

### **Basic Functionality**
- [ ] FastAPI server starts without errors
- [ ] Health endpoint returns 200 with enhanced fields
- [ ] Metrics endpoint returns Prometheus format
- [ ] Inference endpoint accepts 100x9 JSON payload
- [ ] Inference returns 64-dim latent + 12 motif scores

### **Enhanced Features** 
- [ ] `uptime_s` increments in health checks
- [ ] `config_version` shows in health response
- [ ] `hef_sha256` computed (even if "file_not_found" in stub mode)
- [ ] `hailo_build_info` metric present
- [ ] `hailo_config_ok` metric shows status

### **Performance**
- [ ] Inference latency reasonable (<100ms in stub mode)
- [ ] Concurrent requests handled properly
- [ ] Memory usage stable under load
- [ ] Docker container runs without issues

### **Documentation**
- [ ] API docs accessible at `/docs` endpoint
- [ ] Configuration matches revised documentation
- [ ] All test commands work as documented

---

## 🎯 **Next Steps**

Once everything is working on gpusrv:

1. **EdgeInfer Integration**: Test with pisrv_vapor_docker integration
2. **Model Pipeline**: Generate real TCN-VAE models for inference
3. **Hardware Migration**: Deploy to actual Raspberry Pi + Hailo-8 setup
4. **Production Monitoring**: Set up Prometheus/Grafana dashboard

---

## 📞 **Support**

If you encounter issues:

1. **Check logs**: Look for error messages in console output
2. **Verify environment**: Ensure all dependencies installed correctly  
3. **Test incrementally**: Start with health checks before inference testing
4. **Docker fallback**: Use Docker if Python environment has issues

The implementation includes comprehensive error handling and fallback modes, so it should work even without real Hailo hardware.

**Ready to deploy! 🚀**