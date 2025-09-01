"""
HailoRT TCN Inference Sidecar - Main FastAPI Application

Production-ready sidecar serving TCN-VAE encoder inference via Hailo-8 acceleration.
Implements EdgeInfer API contract from ADR-0007 with comprehensive monitoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
import os
import logging
import asyncio
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from .api_endpoints import router, set_model_loader
from .model_loader import ModelLoader
from .metrics import record_startup_metrics, record_shutdown_metrics, get_metrics_collector

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Global model loader for lifecycle management
model_loader: ModelLoader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting HailoRT TCN Inference Sidecar")
    
    global model_loader
    
    # Load configuration from environment
    hef_path = os.getenv("HEF_PATH", "artifacts/tcn_encoder.hef")
    num_motifs = int(os.getenv("NUM_MOTIFS", "12"))
    
    logger.info(f"Configuration - HEF: {hef_path}, Motifs: {num_motifs}")
    
    try:
        # Initialize model loader
        model_loader = ModelLoader(hef_path, num_motifs)
        await model_loader.initialize()
        
        # Inject model loader into API endpoints
        set_model_loader(model_loader)
        
        # Record startup metrics
        record_startup_metrics(model_loader)
        
        if model_loader.is_ready():
            logger.info("✅ HailoRT sidecar startup complete - ready for inference")
        else:
            logger.warning("⚠️  Sidecar started but model not ready")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Continue startup even if model loading fails (allows health checks to report status)
        model_loader = None
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down HailoRT sidecar...")
    
    try:
        if model_loader:
            await model_loader.cleanup()
        
        record_shutdown_metrics()
        logger.info("✅ Graceful shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# Create FastAPI application
app = FastAPI(
    title="HailoRT TCN Inference Sidecar",
    version="1.0.0",
    description="EdgeInfer-compatible TCN-VAE inference on Hailo-8 accelerator",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # EdgeInfer communication
    allow_methods=["GET", "POST", "HEAD"],
    allow_headers=["*"],
    allow_credentials=True
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API routes
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with sidecar status"""
    global model_loader
    
    return {
        "service": "HailoRT TCN Inference Sidecar",
        "version": "1.0.0",
        "model_ready": model_loader.is_ready() if model_loader else False,
        "endpoints": {
            "health": "/healthz",
            "inference": "/infer", 
            "metrics": "/metrics",
            "status": "/status",
            "docs": "/docs"
        },
        "config": {
            "hef_path": os.getenv("HEF_PATH", "artifacts/tcn_encoder.hef"),
            "num_motifs": int(os.getenv("NUM_MOTIFS", "12")),
            "log_level": os.getenv("LOG_LEVEL", "info")
        }
    }


# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def create_app():
    """Factory function for creating app instance"""
    return app


def main():
    """Main entry point for running the sidecar"""
    
    # Environment configuration
    host = os.getenv("SIDECAR_HOST", "0.0.0.0")
    port = int(os.getenv("SIDECAR_PORT", "9000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    workers = int(os.getenv("WORKERS", "1"))
    
    # Validate configuration
    hef_path = os.getenv("HEF_PATH", "artifacts/tcn_encoder.hef")
    if not Path(hef_path).exists() and hef_path != "artifacts/tcn_encoder.hef":
        logger.warning(f"HEF file not found: {hef_path} - will use stub mode")
    
    logger.info(
        f"Starting sidecar on {host}:{port} "
        f"(log_level={log_level}, workers={workers})"
    )
    
    # Run with uvicorn
    uvicorn.run(
        "src.runtime.app:app",
        host=host,
        port=port,
        log_level=log_level,
        workers=workers,
        access_log=log_level == "debug",
        reload=False  # Production mode
    )


if __name__ == "__main__":
    main()