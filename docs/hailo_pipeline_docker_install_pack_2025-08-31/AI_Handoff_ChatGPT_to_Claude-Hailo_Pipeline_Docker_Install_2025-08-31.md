# AI Handoff: ChatGPT → Claude Code

**Date**: 2025-08-31  
**Handoff Type**: Strategic→System  
**Project**: DataDogs / Synchrony — hailo-pipeline  
**Feature/Component**: HailoRT Sidecar Docker Install & Deployment (GPUSrv + Pi)  
**Session Continuity ID**: ADR-0007-hailo-docker-2025-08-31

---
**Navigation**: [[Master_MOC]] • [[Operations & Project Management]] • [[AI Collaboration]]

---

## 🔄 Context Transfer

### Previous Work Summary
**Upstream AI Session**: [[ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]]  
**Key Decisions Made**: Split repos (models/artifacts vs pipeline/export vs edge API); compile `.hef` on GPUSrv; runtime sidecar on Pi; contract-first `/infer` (100×9 → latent[64], motif_scores[12]); Buildx multi-arch for arm64.  
**Current State**: ONNX export path defined; compiler & runtime Dockerfiles drafted; sidecar stub ready; Compose profiles outlined.  
**Completion Status**: Architecture accepted; ready to implement Docker install + build/publish flow.

### Handoff Objective
**Next AI Goal**: Produce a repeatable Docker install + buildx setup on GPUSrv, compile `.hef`, build/push arm64 runtime image, and deliver two working Compose profiles (GPUSrv stub + Pi real).  
**Success Criteria**: Contract smoke passes in both environments; metrics/health live; images published to registry; `.hef` versioned with checksum.  
**Time Constraint**: Start now; first canary within 2 days.  
**Complexity Level**: medium

---

## 🎯 Technical Context for Claude Code

### Architecture Context
**System Design**: Compiler container (x86_64) produces `.hef`; runtime container (arm64) serves `/infer`. Sidecar mounts `/models:ro` at runtime; device `/dev/hailo0` only on Pi.  
**Design Patterns**: Single-responsibility images; versioned artifacts; contract-first API; feature-flag fallback.  
**Performance Requirements**: p95 < 50 ms/window; ≥ 20 windows/sec on Pi.  
**Integration Constraints**: EdgeInfer calls `http://hailo-inference:9000/infer`; flags `USE_REAL_MODEL` toggle stub vs real.

### Implementation Context
**Code Patterns**: FastAPI + Uvicorn; Pydantic schema checks; structured logging; Prometheus client.  
**Style Guidelines**: Small modules; env-driven config; non-root container user.  
**Testing Requirements**: `/infer` key/length contract; health check; parity tests later.  
**Documentation Standards**: README per dir; Obsidian notes mirrored under docs/.

### Hardware/Environment Context
**Deployment Targets**: GPUSrv (Ubuntu x86_64) for build/compile; Raspberry Pi 5 + Hailo-8 for runtime.  
**Container Dependencies**: Buildx/QEMU for arm64; volumes: `./artifacts:/models:ro`; device mapping on Pi.  
**Cross-Repo Impact**: Consumes ONNX from models export; outputs `.hef` for PiSrv + EdgeInfer.  
**Infrastructure Changes**: Self-hosted runner `hailo-builder` for CI jobs.

---

## 🧠 Decision Context (Prevent Knowledge Loss)

### Why These Decisions Were Made
**Business Rationale**: Faster, safer edge releases; reproducible builds; clear ownership.  
**Technical Rationale**: Separate compile/runtime; multi-arch images; avoid baking secrets.  
**Constraint Rationale**: Licensed Hailo toolchain; static input shapes; device access on Pi.  
**Timeline Rationale**: Canary needed to close telemetry loop ASAP.

### Alternative Approaches Considered
**Option A**: Single image with compile+runtime — Rejected: secrets in image, heavy footprint.  
**Option B**: Build on Pi — Rejected: slow, brittle; no x86 toolchain.  
**Option C**: Run without stub on GPUSrv — Rejected: blocks contract tests.

### Critical Dependencies
**Upstream**: ONNX export available; calibration windows for DFC.  
**Downstream**: EdgeInfer environment points to sidecar; PiSrv health proxies backend.  
**Cross-AI**: Copilot for Dockerfile fixes; CI bot for runner jobs.

---

## 📋 Implementation Guidance for Claude Code

### Specific Instructions
**Approach**: Install Docker + Buildx; enable QEMU; build/push `linux/arm64` runtime; compile `.hef` in x86_64 compiler container; write two compose files and smoke scripts.  
**Focus Areas**: Multi-arch correctness; artifact mounting; non-root runtime; health/metrics.  
**Avoid**: Baking `.hef` or licenses into images; amd64 runtime images; device mapping on GPUSrv.  
**Prioritize**: GPUSrv stub contract + Pi real device smoke.

### Quality Expectations
**Code Quality**: professional  
**Documentation Level**: standard  
**Testing Coverage**: integration & smoke  
**Performance Level**: production-ready runtime

### Success Metrics
**Completion Criteria**: Images build & push; `.hef` artifact + checksum; both compose profiles pass contract smoke.  
**Quality Gates**: Lint passes; container runs as non-root; no secrets inside images.  
**Performance Targets**: p95 latency < 50 ms/window on Pi.  
**Integration Validation**: Edge-to-sidecar E2E smoke in staging.

---

## 🔄 Expected Feedback Collection

### Implementation Reality Check
**Expected Discoveries**: Base image quirks (arm64), DFC CLI paths, device mapping issues on Pi.  
**Architecture Feedback Needed**: Confirm final env var names (HEF_PATH/NUM_MOTIFS/INFER_MODE).  
**Resource Requirements**: If compile takes long, cache parse/optimize steps.

### Upstream Planning Impact
**ChatGPT Strategy Updates**: Update ADR with finalized env & compose profiles.  
**Claude System Updates**: Document any SDK CLI deviations.  
**Future Session Planning**: Automate parity tests in CI.

---

## 📝 Knowledge Spillover Prevention

**Paper Note Replacement**: Record `hef_sha256` and `config_version` in `/healthz` + `/metrics`.  
**Cross-Session Context**: Keep `Session Continuity ID` in PR titles.  
**Decision Rationale**: Separate compile/runtime images for security & speed.  
**Implementation Constraints**: Self-hosted runner for DFC; Pi requires `/dev/hailo0`.

---

## 🔗 Handoff Links

**Previous AI Session**: [[ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]]  
**Target AI Session**: [[Hailo_Pipeline_Docker_Install-Output-2025-09-01]] *(to be created)*  
**Related Architecture**: [[EdgeInfer]], [[Hailo_Pipeline]]  
**Sprint Context**: [[2025-09 S1]]  
**Hardware Context**: [[GPUSrv Runner]], [[Pi 5 + Hailo-8]]
 
---

## 📋 Handoff Checklist

- [x] Context documented
- [x] Technical requirements specified
- [x] Success criteria measurable
- [x] Dependencies identified
- [x] Quality expectations communicated

---

*AI Handoff: ChatGPT → Claude Code*  
*Session: Hailo Sidecar Docker Install*  
*AI Stack: ChatGPT (Strategic) → Claude Code (System) → Copilot (Implementation)*  
*Handoff Quality: Professional knowledge transfer with spillover prevention*
