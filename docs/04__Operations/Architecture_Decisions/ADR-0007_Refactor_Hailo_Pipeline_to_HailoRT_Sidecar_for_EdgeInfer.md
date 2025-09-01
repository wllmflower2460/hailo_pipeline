[[Master_MOC]] • [[04__Operations/README]] • [[Architecture_Decisions]]

# ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer

> **Purpose:** Record the decision to scope `hailo_pipeline` to encoder export → `.hef` compile → HailoRT sidecar that’s a drop-in backend for EdgeInfer.

---
Version: 1.0  
Last Updated: 2025-08-31  
Review Due: 2025-09-14  
Maintainer: Will (wllmflower)

Dataview::
status:: accepted  
priority:: high  
project:: DataDogs / Synchrony / EdgeInfer  
component:: hailo_pipeline  
created:: 2025-08-31  
updated:: 2025-08-31  
review-due:: 2025-09-14  
tags:: #adr #architecture #hailo #EdgeInfer #Synchrony  
suggested_path:: 04__Operations/Architecture_Decisions/ADR-0007_Refactor_Hailo_Pipeline_to_HailoRT_Sidecar_for_EdgeInfer.md
---

## Summary
Refactor `hailo_pipeline` into a single-purpose repo that converts a trained **TCN-VAE encoder** to **ONNX → .hef** and serves it via a **FastAPI HailoRT sidecar**, compatible with **EdgeInfer** (`POST /infer` → `{ latent, motif_scores }`). Training, datasets, and multi-modal fusion move to dedicated repos (optional submodules only).

**Why now?** Clear ownership, faster CI/CD, safer rollouts (feature-flag fallback), and a stable API contract for EdgeInfer.

---

## Context
- Existing repo intermixes training + deployment, slowing iteration and raising rollback risk.  
- Hailo compilation requires licensed toolchain/hardware → self-hosted runner on GPUSrv.  
- EdgeInfer expects a fixed 100×9 IMU window and specific JSON keys.

**Constraints**
- Static shapes preferred for Hailo (time=100, channels=9).  
- Maintain parity with training-time normalization/channel order.  
- Preserve EdgeInfer contract and feature flags.

---

## Decision
**Adopt a sidecar architecture** and **shrink repo scope** to:  
TCN-VAE **encoder** → ONNX export → Hailo DFC compile → **`.hef`** → **HailoRT HTTP sidecar**.

**API (contract-first)**
```
POST /infer
Req:  { "x": [[Float; 9]] * 100 }     # shape (100,9)
Resp: { "latent":[Float;64], "motif_scores":[Float;M] }

GET /healthz → { ok, model }
GET /metrics → Prometheus counters/histograms
```

**EdgeInfer integration**
- Service: `hailo-inference:9000` (configurable)  
- Env: `MODEL_BACKEND_URL=http://hailo-inference:9000`, `USE_REAL_MODEL=true`  
- Fallback: stub responses when `USE_REAL_MODEL=false`

---

## Alternatives Considered
- **Monorepo** (keep training + deploy together) → rejected (tight coupling, slow CI).  
- **GPU inference on Pi** (no Hailo) → rejected (latency/power).  
- **Keep legacy model-runner** → rejected (doesn’t advance edge path).

---

## Consequences
**Positive**
- Faster builds/releases; safer canaries; clearer ownership boundaries.  
- Contract-driven integration reduces regressions.

**Negative / Risks**
- Quantization drift vs PyTorch encoder.  
- Hailo SDK/version churn.  
- Device availability for compile/inference.

**Mitigations**
- Lock normalization & channel order in `export_config.yaml`.  
- Golden-parity (cosine similarity) tests vs PyTorch.  
- Pin toolchain versions; run compile on self-hosted runner.

---

## Technical Details

### Minimal Repo Skeleton
```
hailo_pipeline/
  src/onnx_export/          # PyTorch → ONNX
  src/hailo_compilation/    # ONNX → .hef (DFC)
  src/runtime/              # FastAPI HailoRT sidecar
  src/deployment/           # docker-compose, Pi deploy scripts
  exports/                  # *.onnx
  artifacts/                # *.hef
  telemetry/                # *.ndjson (metrics/events)
  al/                       # active learning selection
```

### Targets & SLOs
- p95 latency < **50 ms/window** on Pi/Hailo; ≥ **20 windows/sec** sustained.  
- ≥ **99.5%** success rate under load; golden parity cosine ≥ threshold vs PyTorch.

### Example Smoke
```bash
# health
curl -s localhost:9000/healthz | jq .
# infer (100x9 zeros)
curl -s -X POST localhost:9000/infer -H 'content-type: application/json'   -d '{"x": ['"$(yes ' [0,0,0,0,0,0,0,0,0],' | head -n 99)"' [0,0,0,0,0,0,0,0,0] ] }' | jq .
```

---

## Tasks & Next Actions
- [!] Lock **api_contract.md** with exact shapes & keys.
- [>] Export ONNX (`exports/tcn_encoder.onnx`) from best encoder weights.
- [>] Compile first `.hef` on **GPUSrv** (self-hosted runner) with calibration windows.
- [>] Implement FastAPI sidecar (`/healthz`, `/infer`, `/metrics`) and compose service `hailo-inference`.
- [?] Add telemetry (NDJSON) and Prometheus metrics; wire daily **active learning** selector.
- [<] Golden-parity CI job (PyTorch vs Hailo latent cosine).

**Two-Day Checklist**
- Day 1: branch + skeleton; contract doc; ONNX export; first `.hef`; compose up; curl smoke.  
- Day 2: hook EdgeInfer via env; telemetry + AL stub; CI (contract + image build); plan self-hosted HEF job.

---

## Backrefs & MOC Integration
Mentions: [[EdgeInfer]] • [[Hailo_Pipeline]] • [[Development_Sessions]]  
Context links: [[04__Operations/Architecture_Decisions/README]]  
Section links: See **Tasks & Next Actions** above.  
Block refs: > ^adr-0007-core-contract

---

## Links & References
- Repo: https://github.com/wllmflower2460/hailo_pipeline  
- Related: [[Development_Sessions|Dev Sessions]], [[Architecture_Decisions]]  
- Notes: REFACTORING_PLAN, GPUSrv plan, HEF integration outline, Close-the-Loop checklist.

---

## Change Log
- **2025-08-31** — v1.0 Initial ADR created and accepted.
