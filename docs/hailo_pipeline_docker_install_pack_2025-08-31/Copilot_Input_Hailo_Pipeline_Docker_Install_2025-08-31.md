# Copilot Session: Hailo_Pipeline_Docker_Install-Input

**Date**: 2025-08-31  
**Status**: Planning  
**Session Type**: Refactor/Build & Deploy  
**Estimated Time**: 3 hours  
**Tags**: #copilot-input #hailo_pipeline #docker #buildx #development  
**Priority**: High  
**Sprint**: 2025-09 S1 (Canary HEF)  
**Linked Output**: [[Hailo_Pipeline_Docker_Install-Output-2025-09-01]]  
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
**High-Level Direction**: Separate compile/runtime images; build arm64 runtime via Buildx; compile `.hef` in x86_64 container on GPUSrv; mount artifacts at runtime.  
**Business Context**: Faster, safer edge releases; secure handling of licensed toolchain and artifacts.  
**System Design Context**: Sidecar (arm64) exposes `/infer` and `/healthz`; compiler container (x86_64) emits `.hef`; two Compose profiles.  
**Cross-Stream Coordination**: Consumes ONNX from models export; publishes `.hef` for PiSrv/EdgeInfer integration.

### Implementation Handoff (Claude → Copilot)
**Architecture Context**: Contract-first (`/infer` expects (100×9), outputs latent[64], motif_scores[12]).  
**Code Patterns to Follow**: FastAPI + Pydantic; env-driven config; non-root container.  
**Integration Points**: `HEF_PATH`, `NUM_MOTIFS`, `INFER_MODE`; volumes `/models:ro`; device `/dev/hailo0` on Pi.  
**Hardware Context**: GPUSrv (x86_64) & Pi 5 + Hailo-8 (arm64).

### Expected Feedback (Copilot → Claude/ChatGPT)
- What base images and SDK commands actually worked on arm64?  
- Any changes needed to env var names or compose files?

---

## 🎯 Session Objectives

### Primary Goal
Stand up a reproducible Docker install + buildx pipeline on GPUSrv; compile `.hef`; build/push arm64 runtime image; validate with both compose profiles.

### Success Criteria
- [ ] Runtime image builds for **linux/arm64** and pushes to registry.  
- [ ] `.hef` compiled and saved under `artifacts/` with `hef_sha256sum.txt`.  
- [ ] GPUSrv stub passes contract smoke (`latent=64`, `motif_scores=12`).  

### Context & Background
ADR-0007 accepted; repo separation finalized. We need the install/build plan to enable canary on Pi this sprint.

**Related Epic/Feature**: [[EdgeInfer]]  
**Technical Debt Context**: Prior builds baked artifacts/secrets into images.  
**Business Value**: Repeatable builds; safer deployments.

---

## 📋 Pre-Session Planning

### Current State Assessment
**Files/Components Involved**:
- `docker/hailo_runtime/Dockerfile` — runtime image (arm64)  
- `docker/hailo_compiler/Dockerfile` — compiler image (x86_64)  
- `artifacts/`, `exports/`, `hailo_models/` — builds & configs

**Known Issues/Technical Debt**:
- No buildx/QEMU on GPUSrv — **High**  
- Images not multi-arch — **High**

**Dependencies**:
- Hailo SDK/DFC on compiler image — **Ready** (licensed)  
- ONNX export in `exports/` — **Available**

### Architecture Considerations
**Design Pattern**: Two-image split (compile/runtime), artifact mount.  
**Performance Requirements**: p95 < 50 ms/window on Pi.  
**Security Considerations**: No secrets or artifacts baked into public images.  
**Testing Strategy**: Smoke + contract; parity later.

---

## 🤖 Copilot Instructions

### Context for AI Assistant
```
PROJECT: hailo_pipeline
COMPONENT: docker_install_build
LANGUAGE: Dockerfile, Bash, Python
FRAMEWORK: FastAPI runtime (Python 3.11)
AI STACK ROLE: Implementation (Copilot)
UPSTREAM AI CONTEXT: ADR-0007, AI Handoff (Docker Install)
```

**Current Architecture**:
Compiler container (x86_64) → emits `.hef` → mounted into runtime container (arm64) → `/infer`.

**Code Style Preferences**:
- Minimal layers, non-root user, pinned versions where feasible.

### Specific Implementation Requirements

#### Core Functionality
```markdown
REQUIREMENT 1: Install Docker/Buildx/QEMU on GPUSrv
- Input: Ubuntu host
- Output: buildx with linux/arm64 support
- Constraints: use official Docker apt repo; verify with `buildx ls`
```

```markdown
REQUIREMENT 2: Build/push arm64 runtime image
- Input: docker/hailo_runtime/Dockerfile
- Output: ghcr.io/<org>/hailo-sidecar:<tag> (linux/arm64)
- Constraints: non-root runtime, slim base
```

```markdown
REQUIREMENT 3: Compile .hef in x86_64 compiler image
- Input: exports/tcn_encoder.onnx + hailo_models/tcn_encoder.yaml
- Output: artifacts/tcn_encoder-*.hef + hef_sha256sum.txt
- Constraints: no license in image, output to bind mount
```

#### Error Handling
```markdown
- Missing ONNX → fail with actionable message
- Hailo tool not found → fail and print PATH/Hailo docs hint
- Wrong platform → guardrail: refuse to run runtime on amd64
```

#### Performance Targets
- Image build time < 5 min on GPUSrv (after cache)
- Sidecar cold start < 300 ms; p95 < 50 ms/window on Pi

### Integration Points
**APIs**: `/healthz`, `/infer`, `/metrics` on sidecar

**Data Models**:
```python
# request: {"x": [[float]*9]*100}
# response: {"latent":[float]*64, "motif_scores":[float]*12}
```

**Existing Functions**: N/A

---

## 🔧 Technical Approach

### Implementation Strategy
**Phase 1**: Host install (Docker + Buildx + QEMU)  
- [ ] Add script `scripts/install_docker_buildx.sh`

**Phase 2**: Build & compile  
- [ ] Build/push runtime (arm64)  
- [ ] Build compiler (x86_64), run compile to produce `.hef`

**Phase 3**: Compose + smoke  
- [ ] `docker-compose.gpusrv.yaml` (stub)  
- [ ] `docker-compose.pi.yaml` (real)  
- [ ] Smoke scripts for `/healthz` and contract

### Testing Plan
**Unit**: N/A  
**Integration**: compose up + curl health/infer on both profiles  
**Manual**: verify metrics exist; check non-root user

---

## ⚠️ Risk Assessment

### Technical Risks
- **Multi-arch build errors**: Medium / Medium — Mitigate with Buildx/QEMU and pinned bases.  
- **Compile failures**: Medium / Medium — Validate ONNX ops; provide actionable logs.

### Decision Points
- Base images; Python version in runtime; DFC CLI versions.

### Fallback Options
- Stub mode for runtime on GPUSrv; rollback image tag on Pi.

---

## 📖 Reference Materials

- ADR-0007 (architecture & contract)  
- DOCKER_INSTALLATION_PLAN (this doc)  
- Hailo DFC docs (local)

---

## 🎮 Session Execution Plan

### Environment Setup
- [ ] Install Docker + Buildx + QEMU on GPUSrv
- [ ] Login to GHCR (or registry): `echo $TOKEN | docker login ghcr.io -u USER --password-stdin`

### Development Workflow
1. Build/push runtime (arm64) — 30m  
2. Build compiler + run compile — 45m  
3. Compose smoke (stub + real) — 45m

### Validation Steps
- [ ] `/healthz` OK  
- [ ] Contract check: latent=64, motifs=12  
- [ ] Metrics exposed

---

## 📝 Implementation Notes

_(to be filled during the session)_
