# 🚀 EdgeInfer 72.13% Model Deployment Guide

Complete deployment pipeline for the breakthrough 72.13% accuracy TCN-VAE model from GPUSRV training to EdgeInfer production on Raspberry Pi 5 + Hailo-8.

## 📋 Overview

**Model Performance**: 72.13% validation accuracy (+25.1% improvement over baseline)  
**Deployment Path**: GPUSRV → Tailscale mesh → Edge Platform → EdgeInfer sidecar  
**Target Hardware**: Raspberry Pi 5 (16GB) + Hailo-8 AI accelerator  
**Performance Requirements**: <50ms P95 latency, 250+ req/sec throughput  

## 🎯 Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ GPUSRV Training │───▶│ Tailscale Mesh   │───▶│ Edge Platform       │
│ - Model Export  │    │ - Secure Transfer │    │ - EdgeInfer Service │
│ - ONNX/HEF      │    │ - SSH/SCP        │    │ - Hailo Runtime     │
│ - Validation    │    │ - Automated      │    │ - Performance       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## 🔧 Prerequisites

### Network Configuration
- **GPUSRV**: `gpusrv.tailfdc654.ts.net` (100.97.130.8)
- **Edge Platform**: `edge.tailfdc654.ts.net` (100.127.242.78)
- **Tailscale mesh networking**: Active and accessible
- **SSH key authentication**: Configured between GPUSRV and Edge

### Software Requirements
- **GPUSRV**: Python 3.10+, PyTorch, ONNX, ONNX Runtime
- **Edge Platform**: Docker, Docker Compose, EdgeInfer stack, HailoRT 4.17.0+
- **Hailo-8 accelerator**: Properly installed and detected

## 📦 Deployment Pipeline

### Phase 1: Model Export (GPUSRV)
```bash
# Navigate to training directory
cd /home/wllmflower/Development/tcn-vae-training

# Export 72.13% model to ONNX
python export_best_model.py

# Verify export success
ls -la export/
# Expected: tcn_encoder_for_edgeinfer.onnx (1.7MB)
#           model_config.json
#           best_tcn_vae_72pct.pth
```

**Export Results**:
- ✅ ONNX model: `tcn_encoder_for_edgeinfer.onnx` (1.7MB)
- ✅ Model config: Normalization parameters, accuracy metadata
- ✅ Validation: 1.000000 cosine similarity with PyTorch reference

### Phase 2: Hailo Compilation (GPUSRV)
```bash
# Navigate to hailo pipeline
cd /home/wllmflower/Development/hailo_pipeline

# Compile ONNX to HEF for Hailo-8
python compile_72pct_model.py

# Verify compilation artifacts
ls -la artifacts/
# Expected: tcn_encoder_v72pct.hef
#           deployment_metadata.json
#           calibration_data/
```

**Compilation Results**:
- ✅ HEF model: `tcn_encoder_v72pct.hef` (Hailo-8 optimized)
- ✅ Calibration: 2000 synthetic IMU samples (6.9MB dataset)
- ✅ Metadata: Deployment configuration and performance targets

### Phase 3: Automated Deployment (GPUSRV → Edge)
```bash
# Test deployment prerequisites
./deploy_to_edge.sh test

# Deploy model to Edge platform
./deploy_to_edge.sh deploy

# Monitor deployment progress
# - Backup current model
# - Copy artifacts via Tailscale
# - Update EdgeInfer configuration
# - Restart services
# - Validate health
```

**Deployment Features**:
- ✅ **Automated backup**: Previous model backed up before deployment
- ✅ **Zero-downtime**: Rolling restart with health checks
- ✅ **Rollback capability**: Automatic rollback on deployment failure
- ✅ **Validation**: Health check and endpoint testing post-deployment

### Phase 4: Performance Validation (Remote/Edge)
```bash
# Run comprehensive performance validation
python validate_edge_performance.py

# Quick validation (for testing)
python validate_edge_performance.py --quick

# Validate specific host
python validate_edge_performance.py --host edge.tailfdc654.ts.net --port 9000
```

**Validation Metrics**:
- 🎯 **Latency**: P95 < 50ms (target)
- 🎯 **Throughput**: >250 req/sec (target)  
- 🎯 **Quality**: 64-dim embeddings, sanity checks
- 🎯 **Integration**: `/healthz` and `/encode` endpoints

## 📊 Performance Expectations

### Model Specifications
| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 72.13% | +25.1% improvement |
| **Input Shape** | [1, 100, 9] | IMU window (100 timesteps, 9 channels) |
| **Output Shape** | [1, 64] | Latent embeddings |
| **Model Size** | 1.7MB ONNX | 4.4MB PyTorch checkpoint |

### Performance Targets
| Metric | Target | Validation |
|--------|--------|------------|
| **P95 Latency** | <50ms | Automated benchmark |
| **Throughput** | >250 req/sec | Load testing |
| **Memory Usage** | <512MB | Container monitoring |
| **Success Rate** | >99% | Error rate monitoring |

### Hardware Utilization
| Component | Expected Usage | Monitoring |
|-----------|---------------|------------|
| **Hailo-8** | <50% TOPS | HailoRT metrics |
| **CPU** | <30% (4 cores) | System monitoring |
| **Memory** | <2GB total | Docker stats |
| **Storage** | <100MB artifacts | Disk usage |

## 🔧 Configuration Management

### Normalization Parameters (Critical)
```json
{
  "method": "z_score_per_channel",
  "mean": [0.12, -0.08, 9.78, 0.002, -0.001, 0.003, 22.4, -8.7, 43.2],
  "std": [3.92, 3.87, 2.45, 1.24, 1.31, 0.98, 28.5, 31.2, 24.8],
  "channel_order": ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
}
```

### EdgeInfer Integration
```yaml
# Docker Compose environment variables
environment:
  - MODEL_VERSION=v72pct
  - MODEL_ACCURACY=72.13%
  - HEF_PATH=/app/artifacts/tcn_encoder_v72pct.hef
  - HAILO_DEVICE_ID=0
  - BATCH_SIZE=1
```

### API Contract
```bash
# Health check
curl http://edge.tailfdc654.ts.net:9000/healthz

# Inference endpoint
curl -X POST http://edge.tailfdc654.ts.net:9000/encode \
  -H "Content-Type: application/json" \
  -d '{"imu_window": [[0.12, -0.08, 9.78, ...], ...]}'
```

## 🔄 Operational Procedures

### Monitoring Commands
```bash
# Check EdgeInfer service status
docker-compose ps

# Monitor logs
docker-compose logs -f edgeinfer-sidecar

# Performance metrics
curl http://localhost:9000/metrics

# System resources
htop
nvidia-smi  # For Hailo monitoring
```

### Troubleshooting
```bash
# Service not responding
sudo systemctl restart docker
docker-compose down && docker-compose up -d

# Check Hailo device
hailortcli fw-control identify

# Validate model files
ls -la /opt/edgeinfer/artifacts/
file /opt/edgeinfer/artifacts/tcn_encoder_v72pct.hef
```

### Rollback Procedure
```bash
# Automatic rollback (if deployment fails)
./deploy_to_edge.sh rollback

# Manual rollback
ssh edge@edge.tailfdc654.ts.net
cd /opt/edgeinfer/artifacts
ln -sf backup_20250904_181500.hef current_model.hef
docker-compose restart
```

## 📈 Success Metrics

### Deployment Success Criteria
- ✅ Model artifacts copied to Edge platform
- ✅ EdgeInfer services restart without errors  
- ✅ Health check returns HTTP 200
- ✅ `/encode` endpoint accepts test requests
- ✅ Performance validation passes all targets

### Production Readiness Checklist
- [ ] **Latency**: P95 < 50ms measured on real hardware
- [ ] **Throughput**: >250 req/sec sustained load
- [ ] **Memory**: <512MB container memory usage
- [ ] **Accuracy**: Output quality validation passes
- [ ] **Monitoring**: Prometheus metrics collection active
- [ ] **Alerts**: Performance degradation alerts configured
- [ ] **Documentation**: Operational runbooks updated

## 🚨 Emergency Procedures

### Deployment Failure
1. **Automatic rollback** will be triggered by deployment script
2. **Check logs**: `docker-compose logs edgeinfer-sidecar`
3. **Verify connectivity**: `ping edge.tailfdc654.ts.net`
4. **Manual intervention**: SSH to Edge platform and diagnose

### Performance Degradation
1. **Check system resources**: CPU, memory, storage
2. **Validate Hailo device**: `hailortcli device scan`
3. **Restart services**: `docker-compose restart`
4. **Escalate**: If issues persist, roll back to previous model

### Network Connectivity Issues
1. **Verify Tailscale**: `tailscale status`
2. **Test SSH**: `ssh edge@edge.tailfdc654.ts.net`
3. **Check firewall**: EdgeInfer port 9000 accessibility
4. **Network diagnostics**: `traceroute`, `telnet`

## 📞 Next Steps

### Immediate Actions (Post-Deployment)
1. **Run performance validation**: `python validate_edge_performance.py`
2. **Monitor service stability**: Watch logs and metrics for 24 hours
3. **Test integration**: Verify active learning pipeline connectivity
4. **Update documentation**: Record actual performance measurements

### Medium-term (1-2 weeks)
1. **Production load testing**: Real-world usage patterns
2. **A/B testing**: Compare against previous 57.68% model
3. **Performance optimization**: Fine-tune if needed
4. **Monitoring dashboards**: Create Grafana visualizations

### Long-term (1 month+)
1. **Model retraining pipeline**: Automated model updates
2. **Edge platform scaling**: Multi-device deployment
3. **Active learning integration**: Close-the-loop training
4. **Production monitoring**: SLA tracking and alerting

---

## 📝 Deployment Log Template

```
Deployment Date: ___________
Model Version: v72pct (72.13% accuracy)
Deployed By: ___________
Source: GPUSRV training completion
Target: EdgeInfer on edge.tailfdc654.ts.net

Pre-deployment Checklist:
□ ONNX export completed successfully
□ HEF compilation completed
□ Tailscale connectivity verified
□ SSH access confirmed
□ EdgeInfer services running

Deployment Results:
□ Artifacts copied successfully
□ Services restarted without errors
□ Health check passed
□ Performance validation: ___/___
□ Production ready: Yes/No

Notes:
___________________________________
___________________________________
```

---

*This deployment guide ensures reliable, automated deployment of the breakthrough 72.13% TCN-VAE model to production EdgeInfer infrastructure with comprehensive validation and rollback capabilities.*