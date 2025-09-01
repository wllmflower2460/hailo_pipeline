# DOCKER_INSTALLATION_PLAN — hailo-pipeline (GPUSrv build + Pi runtime)

**Date**: 2025-08-31  
**Owner**: Will  
**Targets**: GPUSrv (Ubuntu, x86_64) for build + Hailo compile; Raspberry Pi 5 + Hailo-8 for runtime

---

## 0) Scope & Outcomes

This plan installs Docker (Engine, Compose v2, Buildx) on **GPUSrv**, sets up **multi-arch** builds for **linux/arm64**, compiles the **Hailo .hef** artifact from ONNX, and publishes a **slim FastAPI runtime image** for the Pi. It also includes a **stub profile** to run the sidecar on GPUSrv (no device) for contract tests.

**Outcomes**
- Reproducible Docker install on GPUSrv
- Buildx/QEMU enabled; image built for **linux/arm64**
- `.hef` compiled in a separate **compiler** container; artifact stored under `artifacts/`
- Runtime image pushed to GHCR (or your registry)
- Two Compose profiles: **GPUSrv (stub)** and **Pi (real /dev/hailo0)**
- Smoke tests for `/healthz` and `/infer` (latent=64, motif_scores=12)

---

## 1) Install Docker Engine + Compose + Buildx on GPUSrv

```bash
# Docker apt repo (Ubuntu 22.04/24.04)
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release; echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Enable Buildx and install QEMU emulation for arm64
docker run --privileged --rm tonistiigi/binfmt --install arm64
docker buildx create --name multi --use
docker buildx inspect --bootstrap

# Sanity
docker version && docker compose version
docker buildx ls
```

**Expect**: a builder named `multi` with `linux/amd64, linux/arm64` support.

---

## 2) Image boundaries (keep secrets safe)

- **Compiler image (x86_64)**: contains Hailo SDK/DFC/Model Zoo; **inputs** = ONNX + calibration set; **output** = `.hef` → write to a bind mount. **Never** commit or bake license files into public images.
- **Runtime image (linux/arm64)**: FastAPI sidecar only; reads `.hef` from a **mounted** `/models:ro` volume at runtime; accesses `/dev/hailo0` **on Pi only**.

---

## 3) Build & publish runtime image (arm64)

```bash
# From repo root (hailo_pipeline)
export IMAGE=ghcr.io/wllmflower2460/hailo-sidecar:v0.1.0

docker buildx build   -f docker/hailo_runtime/Dockerfile   --platform linux/arm64   -t $IMAGE   --push .
```

**Notes**
- Use an **arm64 base** for the runtime image (e.g., `python:3.11-slim-bookworm` on arm64 or multi-arch).
- Keep the image slim; run as non-root (`USER app`).

---

## 4) Compile .hef in a container (x86_64 on GPUSrv)

```bash
# Build the compiler image (contains Hailo tools). Do NOT push publicly.
docker build -f docker/hailo_compiler/Dockerfile -t local/hailo-compiler:x86_64 .

# Prepare mounts
mkdir -p artifacts exports

# Assumes exports/tcn_encoder.onnx already present; run DFC steps
docker run --rm   -v "$PWD/exports:/in:ro"   -v "$PWD/artifacts:/out"   local/hailo-compiler:x86_64   bash -lc 'set -euo pipefail;     cp /in/tcn_encoder.onnx .;     hailomz parse --model hailo_models/tcn_encoder.yaml;     hailomz optimize --model hailo_models/tcn_encoder.yaml;     hailomz compile --model hailo_models/tcn_encoder.yaml;     mv *.hef /out/ && cd /out && sha256sum *.hef > hef_sha256sum.txt && ls -lh'
```

**Expect**: `artifacts/tcn_encoder-v0.1.0.hef` and `artifacts/hef_sha256sum.txt`.

---

## 5) Compose profiles

### 5.1 GPUSrv (stub mode) — contract & smoke tests
```yaml
# docker-compose.gpusrv.yaml
services:
  hailo-inference:
    image: ghcr.io/wllmflower2460/hailo-sidecar:v0.1.0
    ports: ["9000:9000"]
    volumes:
      - ./artifacts:/models:ro
    environment:
      HEF_PATH: "/models/tcn_encoder-v0.1.0.hef"
      NUM_MOTIFS: "12"
      INFER_MODE: "stub"    # no /dev/hailo0 on GPUSrv
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:9000/healthz"]
      interval: 10s
      timeout: 2s
      retries: 10
```

### 5.2 Pi (real device) — production
```yaml
# docker-compose.pi.yaml
services:
  hailo-inference:
    image: ghcr.io/wllmflower2460/hailo-sidecar:v0.1.0
    ports: ["9000:9000"]
    devices: ["/dev/hailo0:/dev/hailo0"]
    volumes:
      - ./artifacts:/models:ro
    environment:
      HEF_PATH: "/models/tcn_encoder-v0.1.0.hef"
      NUM_MOTIFS: "12"
      INFER_MODE: "hailo"
```

---

## 6) Smoke tests

### 6.1 Start
```bash
# GPUSrv stub
docker compose -f docker-compose.gpusrv.yaml up -d

# Pi
docker compose -f docker-compose.pi.yaml up -d
```

### 6.2 Health
```bash
curl -sSf http://localhost:9000/healthz | jq .
```

### 6.3 Contract (latent=64, motifs=12)
```bash
payload=$(python - <<'PY'
import json; print(json.dumps({"x":[[0.0]*9 for _ in range(100)]}))
PY
)
curl -sSf -X POST http://localhost:9000/infer -H 'Content-Type: application/json' -d "$payload" | jq -e '.latent|length==64 and .motif_scores|length==12' >/dev/null && echo "OK"
```

---

## 7) CI on self-hosted runner (label: hailo-builder)

**Stages**
1) **Contract tests** (schema/unit) — GitHub-hosted runner  
2) **Export ONNX** (if scripted) — GitHub-hosted  
3) **Compile .hef** — **self-hosted** (`hailo-builder`) on GPUSrv; save to `artifacts/`  
4) **Buildx runtime** `--platform linux/arm64` and **push** to registry  
5) **Smoke (stub)** on GPUSrv: run compose + curl `/healthz` and `/infer`

**Secrets**
- Inject license keys and protected tools **at job runtime only**; do **not** copy into images or commit to repo.

---

## 8) Security & hardening

- Run runtime image as **non-root**; expose only port **9000**.  
- Keep `.hef` as **mounted** `/models:ro` (don’t bake into image).  
- Restrict sidecar to LAN; if public, front with reverse proxy + auth.  
- Resource limits on Pi (`cpus`, `mem_limit`) to avoid saturation.

---

## 9) Go/No-Go checklist

- [ ] Buildx with arm64 available (`docker buildx ls`)  
- [ ] Runtime image pushed for `linux/arm64`  
- [ ] `.hef` present in `artifacts/` + `hef_sha256sum.txt`  
- [ ] GPUSrv stub passes contract (64 latent, 12 motifs)  
- [ ] Pi run uses `/dev/hailo0` and exports Prometheus metrics

---

## 10) Rollback plan

- Revert to prior image tag on Pi (`docker compose pull && up -d`)  
- Flip EdgeInfer flag `USE_REAL_MODEL=false` to fall back to stub  
- Keep previous `.hef` in `artifacts/` (versioned) for quick restore
