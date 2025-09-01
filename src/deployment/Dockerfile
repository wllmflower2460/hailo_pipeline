# HailoRT TCN Inference Sidecar
# Production Docker image for Raspberry Pi + Hailo-8 deployment

FROM python:3.10-slim

# Install system dependencies for Hailo and ML workloads
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install HailoRT runtime (placeholder for actual package)
# In production, copy the actual .deb from Hailo:
# COPY hailo_runtime_4.17.0_arm64.deb /tmp/
# RUN dpkg -i /tmp/hailo_runtime_4.17.0_arm64.deb || apt-get install -f -y

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for model artifacts and telemetry
RUN mkdir -p artifacts telemetry exports

# Copy any default model files (if available)
# COPY artifacts/*.hef ./artifacts/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash hailo
RUN chown -R hailo:hailo /app
USER hailo

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9000/healthz || exit 1

# Expose the sidecar port
EXPOSE 9000

# Environment defaults
ENV HEF_PATH=/app/artifacts/tcn_encoder.hef
ENV NUM_MOTIFS=12
ENV LOG_LEVEL=info
ENV SIDECAR_HOST=0.0.0.0
ENV SIDECAR_PORT=9000

# Run the FastAPI sidecar
CMD ["python", "-m", "src.runtime.app"]