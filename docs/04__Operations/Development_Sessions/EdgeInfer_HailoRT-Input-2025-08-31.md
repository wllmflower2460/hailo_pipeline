[[Master_MOC]] • [[04__Operations/README]] • [[Development Sessions]]

# Copilot Session: EdgeInfer_HailoRT-Input-2025-08-31

**Date**: 2025-08-31  
**Status**: Planning  
**Session Type**: Refactor  
**Estimated Time**: 3 hours  
**Tags**: #copilot-input #hailo_pipeline #edgeinfer #development  
**Priority**: High  
**Sprint**: 2025-09 Sprint 1 (Canary HEF)  
**Linked Output**: [[EdgeInfer_HailoRT-Output-2025-09-01]]  
**Pair ID**: ADR-0007  
**Time Spent**: 0 minutes  
**Session Start**: 2025-08-31 15:00  
**Session End**: 2025-08-31 18:00  

---
**Navigation**: [[Master_MOC]] • [[Operations & Project Management]] • [[Development Sessions]]

**Related**: [[ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]] • [[AI Handoff: ChatGPT → Claude Code]] • [[Master_Task_Board]]

---

## 🤖 AI Collaboration Context

### Strategic Input (ChatGPT → Claude → Copilot)
**High-Level Direction**: Narrow `hailo_pipeline` scope to encoder export → `.hef` compile → HailoRT sidecar; keep API contract with EdgeInfer stable; ship canary quickly with a feature-flag rollback.  
**Business Context**: Faster, safer edge releases unlock trainer-facing latency targets and enable telemetry-driven active learning.  
**System Design Context**: Sidecar microservice (`hailo-inference`) exposes `/infer` and `/healthz`; EdgeInfer calls via internal URL; artifacts versioned in `artifacts/`. Compile on GPUSrv (self-hosted runner).  
**Cross-Stream Coordination**: Consumes encoder weights from `TCN-VAE_models`; integrates with EdgeInfer backend URL/env; optional references to `DataDogsServer/h8-examples` for device plumbing.

### Implementation Handoff (Claude → Copilot)
**Architecture Context**: Contract-first HTTP service; static input shape (100×9); outputs `latent:[64]`, `motif_scores:[M]`.  
**Code Patterns to Follow**: Python 3.10+, FastAPI + Pydantic, strict type hints, structured logging (`uvicorn`), small modules (`api_endpoints.py`, `model_loader.py`).  
**Integration Points**: Docker Compose service name `hailo-inference`; env: `HEF_PATH`, `NUM_MOTIFS`; EdgeInfer env: `MODEL_BACKEND_URL`, `USE_REAL_MODEL`.  
**Hardware Context**: Runtime on Raspberry Pi + Hailo-8 (device `/dev/hailo0`); compile on GPUSrv with Hailo SDK/toolchain.

### Expected Feedback (Copilot → Claude/ChatGPT)
**Implementation Reality Check**: What APIs from HailoRT SDK are available in the current image; any device binding differences.  
**Architecture Impact**: Whether batch dimension should be static; stability of motif head size.  
**Cross-AI Learning**: Prompts/code patterns that produced robust FastAPI + inference glue.

---

## 🎯 Session Objectives

### Primary Goal
Stand up the HailoRT sidecar with stubbed outputs, compile the first working `.hef` from the TCN encoder, and integrate with EdgeInfer via env configuration.

### Success Criteria
- [ ] `/healthz` returns `{ ok: true, model: <hef-name> }` on Pi/Hailo.  
- [ ] `/infer` accepts a (100×9) window and returns `{ latent:[64], motif_scores:[M] }`.  
- [ ] First `.hef` built from ONNX via DFC on GPUSrv and mounted into sidecar.  

### Context & Background
ADR-0007 accepted: repo is single-purpose (export→compile→serve). Telemetry + AL loop planned to push beyond ~80% accuracy; contract stability prevents regressions during rollout.

**Related Epic/Feature**: [[EdgeInfer]]  
**Technical Debt Context**: Legacy `model-runner` not Hailo-aware; mixed training/deployment concerns now being separated.  
**Business Value**: Production-grade edge latency/power; faster iteration cycles; safer canary rollouts.

---

## 📋 Pre-Session Planning

### Current State Assessment
**Files/Components Involved**:
- `hailort_sidecar/app.py` — to be created (FastAPI service with `/healthz`, `/infer`).  
- `docker-compose.yml` — to define `hailo-inference` service and device bind.  
- `scripts/compile_hef.sh` — DFC pipeline: parse → optimize → compile (emits `.hef`).  
- `hailo_models/tcn_encoder.yaml` — DFC model/config (input/output shapes, calib).  
- `exports/tcn_encoder.onnx` — ONNX from TCN encoder (to be generated or copied).  

**Known Issues/Technical Debt**:
- Shape/order drift between PyTorch and ONNX/Hailo — **High** impact.  
- Normalization parity and calibration windows — **Medium** impact.  
- Hailo SDK/API version differences across hosts — **Medium** impact.

**Dependencies**:
- Best TCN encoder weights — **Ready**.  
- Calibration IMU windows — **Available (small set)**.  
- Hailo toolchain on GPUSrv — **Installed**.  
- Pi device mapping `/dev/hailo0` — **Available**.

### Architecture Considerations
**Design Pattern**: Sidecar microservice, contract-first, SRP.  
**Performance Requirements**: p95 < 50 ms/window; ≥ 20 windows/sec sustained.  
**Security Considerations**: Local network only; validate input shape/values; no secrets in image.  
**Testing Strategy**: Schema/unit tests; golden-parity cosine vs PyTorch; load test with 2k windows.

---

## 🤖 Copilot Instructions

### Context for AI Assistant
```
PROJECT: hailo_pipeline
COMPONENT: hailort_sidecar
LANGUAGE: Python 3.10+
FRAMEWORK: FastAPI + Uvicorn
AI STACK ROLE: Implementation (Copilot) - Focus on tactical code generation
UPSTREAM AI CONTEXT: ADR-0007 + AI Handoff notes
```

**Current Architecture**:
EdgeInfer → HTTP → `hailo-inference:/infer` → HailoRT execution of `.hef`. Versioned artifacts in `artifacts/`; env selects model file.

**Code Style Preferences**:
- Pydantic models for request/response schemas; explicit shape checks.  
- Type hints and docstrings; small functions; early returns on validation errors.  
- Structured logging (request id, latency) suitable for Prometheus labels.

### Specific Implementation Requirements

#### Core Functionality
```markdown
REQUIREMENT 1: Implement POST /infer
- Input: JSON {"x": [[float]*9]*100} (shape [100,9], finite values only)
- Output: {"latent":[float]*64, "motif_scores":[float]*M} (M via env NUM_MOTIFS)
- Constraints: Static shapes; preprocessing consistent with training normalization
- Edge Cases: Reject wrong shapes/NaNs with HTTP 400
```

```markdown
REQUIREMENT 2: Implement GET /healthz and /metrics
- Input: None
- Output: Health JSON with model name; Prometheus metrics (requests, latency hist, model_loaded gauge)
- Constraints: Ready before /infer; non-blocking
- Edge Cases: If HEF not found, health shows ok:false and logs error
```

#### Error Handling
```markdown
- Missing/invalid JSON → 400 with message and schema hint
- Wrong shape or non-finite values → 400 with shape/validation error
- Device init or model load failure → 503 with "model_unavailable" code
```

#### Performance Targets
- p95 latency: Target < 50 ms/window on Pi/Hailo  
- Throughput: Target ≥ 20 windows/sec sustained  
- CPU usage: Keep under 40% on canary Pi during sustained load

### Integration Points
**APIs to Call**:
- Local HailoRT inference API/bindings inside the sidecar (Python SDK/module).

**Data Models**:
```python
from pydantic import BaseModel, conlist
class IMUWindow(BaseModel):
    x: conlist(conlist(float, min_items=9, max_items=9), min_items=100, max_items=100)

class InferResponse(BaseModel):
    latent: list[float]   # 64
    motif_scores: list[float]  # M
```

**Existing Functions to Leverage**:
- None yet; create `model_loader.py` with `load_model(HEF_PATH)` and `run_inference(window: np.ndarray) -> tuple[np.ndarray, np.ndarray]`.

---

## 🔧 Technical Approach

### Implementation Strategy
**Phase 1**: Sidecar scaffold
- [ ] Create FastAPI app (`/healthz`, `/infer`) and Pydantic models  
- [ ] Add stub inference (zeros) and Prometheus metrics

**Phase 2**: Hailo integration  
- [ ] Implement `model_loader.py` using HailoRT SDK  
- [ ] Wire env `HEF_PATH`, `NUM_MOTIFS`; add startup/shutdown hooks

**Phase 3**: Packaging & deploy
- [ ] Dockerfile + `docker-compose.yml` with `/dev/hailo0` device mapping  
- [ ] Mount `artifacts/` read-only; update EdgeInfer env to point to sidecar

### Testing Plan
**Unit Tests**:
- [ ] Schema/shape validation for `/infer`  
- [ ] Health behavior when HEF missing

**Integration Tests**:
- [ ] End-to-end infer with stub → returns correct lengths  
- [ ] Golden-parity cosine vs PyTorch on fixed windows (threshold TBD)

**Manual Testing Checklist**:
- [ ] `curl /healthz` and `curl /infer` locally  
- [ ] `docker compose up` on Pi; smoke passes  
- [ ] Prometheus scrapes `/metrics`

---

## ⚠️ Risk Assessment

### Technical Risks
- **Quantization drift vs PyTorch**: Medium / High — Mitigation: calibration set and parity tests  
- **SDK/API incompatibilities**: Medium / Medium — Mitigation: pin versions; adapt bindings

### Decision Points
- **Batch dimension static vs dynamic**: Options {static, dynamic-1} — Criteria: latency, SDK support  
- **Motif head size (M)**: Options {12, configurable} — Criteria: downstream consumers

### Fallback Options
- **Plan B**: Stub inference with zeros while collecting telemetry  
- **Rollback Strategy**: Flip `USE_REAL_MODEL=false` in EdgeInfer

---

## 📖 Reference Materials

### Documentation
- [[ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]] — architecture & contract  
- [[AI Handoff: ChatGPT → Claude Code]] — implementation guidance  
- Hailo DFC/Model Zoo README (local docs) — DFC compile steps

### Similar Implementations
- `DataDogsServer/h8-examples` — device examples — Lessons: device mapping and init order  
- `edgeinfer/model-runner` (legacy) — HTTP contract — Lessons: request validation patterns

### Research/Design Decisions
- [[Architecture_Decisions]] — sidecar rationale and SLOs

---

## 🎮 Session Execution Plan

### Environment Setup
- [ ] Ensure Hailo SDK on GPUSrv; Pi exposes `/dev/hailo0`  
- [ ] Python 3.10 venv on dev machine; `pip install fastapi uvicorn pydantic prometheus-client numpy`  
- [ ] Docker/Compose installed on Pi

### Development Workflow
1. **Scaffold** — Create FastAPI app + models + stubs — Est. 45m  
2. **Integrate Hailo** — Implement loader + inference — Est. 75m  
3. **Package & Smoke** — Dockerize, compose up on Pi, curl tests — Est. 60m

### Validation Steps
- [ ] Code compiles and lints cleanly  
- [ ] All unit/integration tests pass  
- [ ] p95 latency target observed on canary  
- [ ] Health and metrics endpoints verified  
- [ ] Ready for PR & code review

---

## 📝 Implementation Notes

### Key Decisions Made
*[To be filled during development]*

### Unexpected Challenges
*[To be filled during development]*

### Performance Observations
*[To be filled during development]*

### Code Quality Metrics
*[To be filled during development]*

---

## 🤖 AI Collaboration Feedback Capture

### Copilot Implementation Reality
**What Actually Worked**: *[Implementation discoveries vs planned approach]*  
**Code Pattern Effectiveness**: *[Which patterns Copilot handled well/poorly]*  
**Architecture Feedback**: *[Implementation insights affecting system design]*

### Cross-AI Learning Insights
**Effective Prompts**: *[What worked well for Copilot in this session]*  
**Integration Points**: *[How well the Claude → Copilot handoff worked]*  
**System Impact**: *[Implementation discoveries that affect Claude/ChatGPT planning]*

### Knowledge Spillover Capture
**Paper Note Replacement**: *[Critical details that would otherwise be lost]*  
**Future Session Inputs**: *[Context that should inform future AI sessions]*  
**Technical Debt Created**: *[Shortcuts or compromises that need future attention]*

---

## 🔗 Session Links

**Output Documentation**: [[EdgeInfer_HailoRT-Output-2025-09-01]]  
**Related Sessions**: [[ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]] | [[AI Handoff: ChatGPT → Claude Code]]  
**Feature Epic**: [[EdgeInfer]]  
**Sprint Board**: [[Master_Task_Board]]  
**AI Architecture Session**: [[ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]]  
**AI Strategic Session**: [[AI Handoff: ChatGPT → Claude Code]]

---

## 📋 Session Checklist

### Pre-Development
- [ ] Requirements clearly defined
- [ ] Architecture approach documented
- [ ] Test cases identified
- [ ] Dependencies verified
- [ ] Environment ready
- [ ] AI context handoff complete

### During Development  
- [ ] Code follows style guidelines
- [ ] Error handling implemented
- [ ] Performance targets considered
- [ ] Security requirements met
- [ ] Tests written and passing
- [ ] AI collaboration documented

### Post-Development
- [ ] Code reviewed
- [ ] Documentation updated
- [ ] Output session created
- [ ] Git commits well-structured
- [ ] Next steps identified
- [ ] AI feedback captured for upstream planning

---

*Session: EdgeInfer_HailoRT-Input-2025-08-31*  
*Developer: Will*  
*AI Stack: ChatGPT (Strategic) → Claude Code (System) → Copilot (Implementation)*  
*Copilot Version: latest*  
*IDE: VS Code (latest)*
