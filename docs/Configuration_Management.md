[[Master_MOC]] • [[04__Operations/README]] • [[Runbooks]]

# Configuration Management (Revised)

_Last updated: 2025-09-01_

This document defines how we configure, verify, and monitor the the Edge stack across **PiSrv (Vapor)** and the **Hailo sidecar**. It includes corrected request examples, stronger health/metrics contracts, and testable acceptance checks.

---

## 1) Network & Service Configuration

### 1.1 Environment variables
- `MODEL_BACKEND_URL` → default `http://hailo-inference:9000/infer`
- `USE_REAL_MODEL` → `true|false` (fallback to stub if `false`)
- `BACKEND_TIMEOUT_MS` → default `250`
- `BACKEND_RETRIES` → default `0`
- `METRICS_PORT` → default `8080`
- `HEALTHZ_GRACE_S` → default `5`

### 1.2 Correct `/infer` request (100×9 JSON)
> Previous examples used Python-style list math and/or the wrong shape. Use this generator to ensure valid **JSON** with **100 rows × 9 columns**:

```bash
payload=$(python - <<'PY'
import json
print(json.dumps({"x":[[0.0]*9 for _ in range(100)]}))
PY
)
curl -sSf -X POST "$MODEL_BACKEND_URL" -H 'Content-Type: application/json' -d "$payload" | jq .
```

### 1.3 Service ports
- **PiSrv (Vapor)**: `:8080` → `/healthz`, `/metrics`, app endpoints  
- **Hailo sidecar**: `:9000` → `/infer`, `/healthz`, `/metrics`

---

## 2) Health & Status Contracts

### 2.1 Hailo sidecar `/healthz`
```json
{
  "ok": true,
  "model": "tcn_encoder-v0.1.0.hef",
  "uptime_s": 12345,
  "config_version": "hailo_pipeline_production_config-2025-09-01",
  "hef_sha256": "cafe...beef"
}
```

### 2.2 PiSrv (Vapor) `/healthz` (aggregate)
```json
{
  "ok": true,
  "backend_ok": true,
  "version": "pisrv-1.3.0",
  "uptime_s": 2345,
  "backend_url": "http://hailo-inference:9000/infer"
}
```

> **Notes**
> - `backend_ok` is `false` if backend call exceeds `BACKEND_TIMEOUT_MS` or non-2xx.  
> - Both services should return `HTTP 200` when healthy.  
> - Cold-start: model load may add **100–300 ms** on first request.

---

## 3) Metrics (Prometheus)

### 3.1 Required
- **PiSrv**
  - `pisrv_requests_total{route,code}`
  - `pisrv_request_latency_ms_bucket{route}` (histogram with p95/p99 in PromQL)
  - `pisrv_backend_errors_total{reason}`
  - `pisrv_backend_ok` (0/1 gauge)

- **Hailo sidecar**
  - `infer_requests_total`
  - `infer_latency_ms_bucket`
  - `model_loaded` (0/1 gauge)

### 3.2 Release & config tracking (additions)
Add these _extra_ gauges to simplify fleet auditing and alerting:

```
build_info{version="v1.0.0", hef_sha="cafe...beef", config="hailo_pipeline_production_config-2025-09-01"} 1
config_ok{expected="hailo_pipeline_production_config-2025-09-01", actual="hailo_pipeline_production_config-2025-09-01"} 1
```

> Alert if `config_ok` ever becomes `0` on any device.

---

## 4) Automated Testing (Smoke & Contract)

### 4.1 Sidecar smoke (`/infer` + shape check)
```bash
set -euo pipefail

MODEL_BACKEND_URL="${MODEL_BACKEND_URL:-http://hailo-inference:9000/infer}"

payload=$(python - <<'PY'
import json
print(json.dumps({"x":[[0.0]*9 for _ in range(100)]}))
PY
)

resp=$(curl -sSf -X POST "$MODEL_BACKEND_URL" -H 'Content-Type: application/json' -d "$payload")

# Validate keys and lengths: latent=64, motifs=12
echo "$resp" | jq -e '.latent|length==64 and .motif_scores|length==12' >/dev/null
echo "OK: contract validated (latent=64, motif_scores=12)"
```

### 4.2 PiSrv aggregate health
```bash
curl -sSf http://localhost:8080/healthz | jq .
```

---

## 5) Backups & Secrets

### 5.1 Encrypted backups of models/metadata
```bash
# Using age (recommended). Substitute your public recipient key.
tar -czf - models/ | age -r <YOUR_AGE_RECIPIENT> > backup_models_$(date +%Y%m%d).tar.gz.age

# Alternatively, using gpg (passphrase-based):
tar -czf - models/ | gpg --symmetric --cipher-algo AES256 -o backup_models_$(date +%Y%m%d).tar.gz.gpg
```

### 5.2 Config integrity
- Keep `config_version` in deployment manifests and mirror it in `/healthz`.  
- Compare `hef_sha256` against the artifact's published checksum (from the Models release).

---

## 6) Performance Notes

- **Bench context**: _Pi 5 (8 GB) + Hailo‑8, ambient 22 °C, single‑client load._  
- Targets: **p95 < 50 ms/window**, throughput **≥ 20 windows/sec** on canary.  
- Warm-start overhead: **100–300 ms** on first request after model load.

---

## 7) EdgeInfer Contract Self‑Report (optional)

Expose a convenience endpoint (either service) that echoes the runtime contract for quick validation:

```json
{
  "window_len": 100,
  "channels": 9,
  "latent_dims": 64,
  "motif_classes": 12
}
```

---

## 8) Change Log
- 2025-09-01: Updated with Hailo Pipeline implementation; enhanced health checks with config_version/hef_sha256; added build_info/config_ok metrics; validated against production Raspberry Pi + Hailo-8 deployment; integrated with EdgeInfer contract compliance.
- 2025-08-31: Revised examples to valid 100×9 JSON; added `config_version` and `hef_sha256` to `/healthz`; strengthened metrics with `build_info` and `config_ok`; clarified performance context; added encrypted backup commands; corrected automated contract test.