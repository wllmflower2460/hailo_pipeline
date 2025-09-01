# Hailo Pipeline Documentation

This repository contains comprehensive documentation for the Hailo Pipeline project following the ADR-0007 architecture decision.

## 📁 Documentation Structure

### 04__Operations/
Following the Obsidian vault organization pattern for operational documentation.

#### Architecture_Decisions/
- **ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer.md**
  - Core architectural decision record
  - Scope reduction to export→compile→serve pipeline
  - API contract and performance SLOs

#### AI_Collaboration/
- **AI_Handoff_ChatGPT_to_Claude_Code-ADR-0007_2025-08-31.md**
  - Strategic to system AI handoff documentation
  - Context transfer and implementation guidance
  - Session continuity preservation

#### Development_Sessions/
- **EdgeInfer_HailoRT-Input-2025-08-31.md**
  - Development session planning and context
  - Implementation requirements and success criteria
  - Copilot collaboration framework

#### Implementation_Plans/
- **ADR_0007_IMPLEMENTATION_ROADMAP.md**
  - Complete phase-by-phase implementation guide
  - Code examples and architectural patterns
  - Two-day delivery checklist
- **REFACTORING_PLAN.md**
  - Detailed refactoring strategy
  - Removal of TCN-VAE training overlap
  - EdgeInfer integration specifications
- **GPUSRV_DEPLOYMENT_PLAN.md**
  - GPU server deployment strategy
  - Infrastructure optimization for RTX 2060
  - Integration with existing ML ecosystem

## 🎯 Project Mission

Transform hailo_pipeline into a **single-purpose conversion and serving tool** that:

1. **Converts** trained TCN-VAE models to ONNX format
2. **Compiles** ONNX models to Hailo .hef format 
3. **Serves** inference via FastAPI HailoRT sidecar
4. **Integrates** seamlessly with EdgeInfer service

## 🔗 Key Integration Points

- **Input**: TCN-VAE models from `../TCN-VAE_models/`
- **Output**: HailoRT sidecar at `hailo-inference:9000/infer`
- **EdgeInfer**: `MODEL_BACKEND_URL` environment integration
- **Deployment**: Pi + Hailo-8 edge inference

## 📊 Success Metrics

- **Performance**: p95 < 50ms per IMU window
- **Throughput**: ≥20 windows/sec sustained
- **API Contract**: 100% EdgeInfer compatible
- **Integration**: Feature flag rollback capability

## 🤖 AI Collaboration Context

**Session ID**: ADR-0007-2025-08-31  
**AI Stack**: ChatGPT (Strategic) → Claude Code (System) → Copilot (Implementation)  
**Repository**: https://github.com/wllmflower2460/hailo_pipeline

This documentation maintains session continuity across AI collaborations and provides comprehensive implementation guidance for the focused Hailo Pipeline architecture.