[[Master_MOC]] • [[04__Operations/README]] • [[AI Collaboration]]

# AI Handoff: ChatGPT → Claude Code

**Date**: 2025-08-31  
**Handoff Type**: Strategic→System  
**Project**: DataDogs / Synchrony — EdgeInfer  
**Feature/Component**: Hailo Pipeline (TCN Encoder → .hef → HailoRT sidecar)  
**Session Continuity ID**: ADR-0007-2025-08-31

---

## 🔄 Context Transfer

### Previous Work Summary
**Upstream AI Session**: [[04__Operations/Architecture_Decisions/ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]]  
**Key Decisions Made**: Narrow repo scope to export→compile→serve; enforce EdgeInfer API contract; use FastAPI HailoRT sidecar; compile on self-hosted GPUSrv; telemetry + active learning loop.  
**Current State**: ADR accepted; repo skeleton/design agreed; awaiting first ONNX export and `.hef` compile; sidecar stub ready to implement.  
**Completion Status**: Decision recorded; implementation kickoff pending.

### Handoff Objective
**Next AI Goal**: Stand up the HailoRT sidecar and produce the first working `.hef` from the TCN encoder, wired into EdgeInfer via env config.  
**Success Criteria**: `/healthz` and `/infer` pass smoke on Pi/Hailo; p95 < 50 ms/window on canary; EdgeInfer reads `{latent, motif_scores}` with correct shapes.  
**Time Constraint**: Begin today; first working canary within 2 days.  
**Complexity Level**: medium

---

## 🎯 Technical Context for Claude Code

### Architecture Context
**System Design**: Sidecar microservice exposes `/infer` and `/healthz`; EdgeInfer calls via internal URL `hailo-inference:9000`; artifacts versioned under `artifacts/`.  
**Design Patterns**: Contract-first API, SRP for repo scope, versioned artifacts, feature-flag fallback (`USE_REAL_MODEL`).  
**Performance Requirements**: ≥ 20 windows/sec; p95 < 50 ms/window on Pi/Hailo.  
**Integration Constraints**: Maintain `POST /infer` with input shape (100×9) and outputs `latent:[64]`, `motif_scores:[M]` (M≈12, configurable).

### Implementation Context
**Code Patterns**: Python 3.10+, FastAPI + Pydantic; strict type hints; small modules (`api_endpoints.py`, `model_loader.py`); env-configured HEF path.  
**Style Guidelines**: Docstrings per function, README in each directory, Obsidian-linked docs for contracts.  
**Testing Requirements**: Contract schema tests; golden-parity (cosine similarity) vs PyTorch encoder on a fixed window set; load test 2k windows.  
**Documentation Standards**: Include navigation header, Dataview keys (`status, project, component, updated`), change log entries.

### Hardware/Environment Context
**Deployment Targets**: Raspberry Pi + Hailo-8 for runtime; GPUSrv (RTX 2060) for export/compile with Hailo toolchain.  
**Container Dependencies**: `hailo-inference` service in `docker-compose.yml`; device mapping `/dev/hailo0`; mount `artifacts/` read-only.  
**Cross-Repo Impact**: Consumes encoder weights from `TCN-VAE_models`; EdgeInfer expects stable backend URL and schema.  
**Infrastructure Changes**: Self-hosted runner labeled `hailo-builder` for HEF compile job.

---

## 🧠 Decision Context (Prevent Knowledge Loss)

### Why These Decisions Were Made
**Business Rationale**: Faster, safer releases; clear ownership boundaries.  
**Technical Rationale**: Deterministic shapes, quantized Hailo model for edge latency/power.  
**Constraint Rationale**: Licensed toolchain; static shapes; Pi/Hailo device access.  
**Timeline Rationale**: Need canary quickly to close the telemetry/AL loop and push accuracy >80%.

### Alternative Approaches Considered
**Option A**: Keep monorepo — Rejected because: tight coupling, slow CI/CD.  
**Option B**: GPU inference on Pi — Rejected because: latency/power inferior to Hailo.  
**Option C**: Keep legacy model-runner — Deferred because: doesn’t advance edge path; retained only as stub fallback.

### Critical Dependencies
**Upstream Dependencies**: Best TCN encoder weights; calibration IMU windows; export parity on normalization/channel order.  
**Downstream Impact**: EdgeInfer release toggles `USE_REAL_MODEL` and updates backend URL; trainer-facing features rely on low latency.  
**Cross-AI Dependencies**: Copilot for boilerplate implementation; NotebookLM/Docs bots for contract docs; CI bot for self-hosted runner jobs.

---

## 📋 Implementation Guidance for Claude Code

### Specific Instructions
**Approach**: Implement `/healthz` and `/infer` first with stub outputs; then wire HailoRT execution with HEF; add `/metrics` for Prometheus.  
**Focus Areas**: Input validation (shape [100,9], finite floats); deterministic preprocessing; device error handling; versioned HEF selection.  
**Avoid**: Dynamic shapes; silent fallback without telemetry; mixing training code into this repo.  
**Prioritize**: Contract tests, first `.hef`, canary deploy, latency SLO.

### Quality Expectations
**Code Quality**: professional  
**Documentation Level**: standard (contract README + module READMEs)  
**Testing Coverage**: integration + golden-parity; contract unit tests  
**Performance Level**: production (edge latency)

### Success Metrics
**Completion Criteria**: Can run `curl /healthz` and `curl /infer` on Pi/Hailo successfully; EdgeInfer reads proper shapes.  
**Quality Gates**: CI job passes contract/golden tests; self-hosted compile job succeeds; image builds reproducibly.  
**Performance Targets**: p95 < 50 ms/window, ≥ 20 windows/sec sustained on canary Pi.  
**Integration Validation**: EdgeInfer end-to-end smoke in staging with `USE_REAL_MODEL=true`.

---

## 🔄 Expected Feedback Collection

### Implementation Reality Check
**Expected Discoveries**: Quantization drift; device driver quirks; ONNX op compatibility issues.  
**Architecture Feedback Needed**: Confirm if batch dimension should be static; verify motif head output size stability.  
**Resource Requirements**: If compilation times exceed expectations, consider caching ONNX parse/optimize steps.

### Upstream Planning Impact
**ChatGPT Strategy Updates**: If parity fails, adjust export normalization or calibration set curation.  
**Claude System Updates**: If device mapping or SDK APIs differ, document and version-pin.  
**Future Session Planning**: Schedule telemetry/AL integration and automated golden-parity in CI next sprint.

---

## 📝 Knowledge Spillover Prevention

### Critical Context (Don't Lose This)
**Paper Note Replacement**: Per-channel z-score parameters must match training; maintain in `export_config.yaml`.  
**Cross-Session Context**: Keep `Session Continuity ID` in future dev notes and PR titles.  
**Decision Rationale**: Sidecar gives safer rollbacks and isolates vendor-specific code.  
**Implementation Constraints**: Self-hosted runner required for compile; Pi must expose `/dev/hailo0` to container.

### Learning Capture
**AI Collaboration Patterns**: Contract-first + small deliverables worked best.  
**Effective Prompts**: “Generate stub + curl smokes + Makefile targets” produced rapid progress.  
**Process Improvements**: Add golden-parity early; standardize telemetry fields in a schema file.

---

## 🔗 Handoff Links

**Previous AI Session**: [[04__Operations/Architecture_Decisions/ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer]]  
**Target AI Session**: [[04__Operations/Development_Sessions/EdgeInfer_HailoRT-Output-2025-09-01]] *(to be created)*  
**Related Architecture**: [[EdgeInfer]], [[Hailo_Pipeline]]  
**Sprint Context**: [[04__Operations/Master_Task_Board]]  
**Hardware Context**: [[03__Hardware/Hailo-8_Pi_Setup]]

---

## 📋 Handoff Checklist

### Pre-Handoff Validation
- [x] All critical context documented above
- [x] Technical requirements clearly specified
- [x] Success criteria defined and measurable
- [x] Dependencies identified and status confirmed
- [x] Quality expectations communicated

### Post-Handoff Follow-up
- [ ] Target AI session created successfully
- [ ] Implementation progressing as expected
- [ ] Feedback loop established for discoveries
- [ ] Knowledge spillover captured in target session
- [ ] Course corrections documented if needed

---

*AI Handoff: ChatGPT → Claude Code*  
*Session: ADR-0007 Hailo Pipeline Refactor*  
*AI Stack: ChatGPT (Strategic) → Claude Code (System) → Copilot (Implementation)*  
*Handoff Quality: Professional knowledge transfer with spillover prevention*
