"""
FastAPI endpoints for EdgeInfer-compatible HailoRT inference sidecar.

Implements the exact API contract from ADR-0007:
- POST /infer: IMU window inference
- GET /healthz: Health check for EdgeInfer
- GET /metrics: Prometheus telemetry
"""

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
import time
import logging
from typing import Optional

from .schemas import IMUWindow, InferResponse, HealthResponse, ErrorResponse
from .model_loader import ModelLoader
from .metrics import (
    get_metrics_collector, generate_metrics, get_metrics_content_type,
    InferenceTimer, record_health_check_success, record_health_check_failure
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Global model loader reference (injected by app.py)
_model_loader: Optional[ModelLoader] = None


def set_model_loader(model_loader: ModelLoader):
    """Inject model loader for endpoints"""
    global _model_loader
    _model_loader = model_loader


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    EdgeInfer health check endpoint
    
    Returns 200 OK if model is loaded and ready for inference.
    Returns 503 Service Unavailable if model is not ready.
    
    EdgeInfer uses this to determine if the backend is healthy.
    """
    try:
        if _model_loader and _model_loader.is_ready():
            status = _model_loader.get_status()
            
            record_health_check_success()
            
            return HealthResponse(
                ok=True,
                model=_model_loader.model_name,
                device="hailo8",
                latency_ms=_model_loader.last_inference_time
            )
        else:
            record_health_check_failure()
            
            # Return 503 to indicate service unavailable
            raise HTTPException(
                status_code=503,
                detail="Model not loaded or unavailable"
            )
            
    except Exception as e:
        record_health_check_failure()
        logger.error(f"Health check failed: {e}")
        
        raise HTTPException(
            status_code=503,
            detail=f"Health check error: {str(e)}"
        )


@router.post("/infer", response_model=InferResponse)
async def infer_imu_window(window: IMUWindow, request: Request):
    """
    EdgeInfer-compatible inference endpoint
    
    Processes 100x9 IMU windows and returns 64-dim latent + motif scores.
    This is the core endpoint that EdgeInfer calls for real-time analysis.
    
    Request body:
        {"x": [[float; 9]] * 100}  # 100 timesteps × 9 channels
        
    Response:
        {"latent": [float; 64], "motif_scores": [float; M]}
    """
    metrics_collector = get_metrics_collector()
    
    # Input validation already handled by Pydantic
    client_ip = request.client.host if request.client else "unknown"
    logger.debug(f"Inference request from {client_ip}")
    
    if not _model_loader or not _model_loader.is_ready():
        metrics_collector.record_error("model_unavailable")
        
        raise HTTPException(
            status_code=503,
            detail="Model not available for inference"
        )
    
    # Time the inference operation
    with InferenceTimer(metrics_collector) as timer:
        try:
            # Run inference through HailoRT
            latent, motif_scores = await _model_loader.infer(window.x)
            
            # Validate output shapes
            if len(latent) != 64:
                raise ValueError(f"Invalid latent dimension: {len(latent)}, expected 64")
            
            if len(motif_scores) != _model_loader.num_motifs:
                raise ValueError(
                    f"Invalid motif count: {len(motif_scores)}, "
                    f"expected {_model_loader.num_motifs}"
                )
            
            # Mark timing as successful
            timer.mark_success()
            
            response = InferResponse(
                latent=latent.tolist(),
                motif_scores=motif_scores.tolist()
            )
            
            logger.debug(
                f"Inference completed - Latent range: [{min(latent):.3f}, {max(latent):.3f}], "
                f"Motif range: [{min(motif_scores):.3f}, {max(motif_scores):.3f}]"
            )
            
            return response
            
        except ValueError as e:
            # Input validation or output shape errors
            metrics_collector.record_error("validation_error")
            logger.error(f"Validation error: {e}")
            
            raise HTTPException(
                status_code=400,
                detail=f"Input validation error: {str(e)}"
            )
            
        except RuntimeError as e:
            # Model execution errors
            metrics_collector.record_error("inference_error")
            logger.error(f"Inference runtime error: {e}")
            
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {str(e)}"
            )
            
        except Exception as e:
            # Unexpected errors
            metrics_collector.record_error("unexpected_error")
            logger.error(f"Unexpected inference error: {e}")
            
            raise HTTPException(
                status_code=500,
                detail="Internal inference error"
            )


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus text format for monitoring:
    - Inference request counts and latency
    - Model loading status
    - Hardware utilization (Hailo device)
    - Error counts by type
    """
    try:
        metrics_text = generate_metrics()
        
        # Set appropriate content type for Prometheus
        return Response(
            content=metrics_text,
            media_type=get_metrics_content_type()
        )
        
    except Exception as e:
        logger.error(f"Metrics generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate metrics"
        )


@router.get("/status", response_model=dict)
async def detailed_status():
    """
    Detailed status endpoint for debugging
    
    Returns comprehensive information about:
    - Model loading status
    - HailoRT device state
    - Recent inference performance
    - Configuration parameters
    """
    try:
        if _model_loader:
            status = _model_loader.get_status()
            
            # Add runtime information
            status.update({
                "endpoint_status": "healthy",
                "api_version": "1.0.0",
                "hailort_available": hasattr(_model_loader, 'device') and _model_loader.device is not None
            })
            
            return status
        else:
            return {
                "endpoint_status": "model_not_loaded",
                "ready": False,
                "error": "No model loader initialized"
            }
            
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "endpoint_status": "error",
            "ready": False,
            "error": str(e)
        }


# Optional: Add a simple test endpoint for development
@router.post("/test", response_model=dict)
async def test_inference():
    """
    Simple test endpoint with dummy IMU data
    
    Useful for quick smoke testing without constructing full payloads.
    Generates a 100x9 zero matrix for inference testing.
    """
    if not _model_loader or not _model_loader.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not available"
        )
    
    # Generate test IMU window (100x9 zeros)
    test_window = [[0.0] * 9 for _ in range(100)]
    
    try:
        latent, motif_scores = await _model_loader.infer(test_window)
        
        return {
            "test_status": "success",
            "latent_shape": len(latent),
            "motif_shape": len(motif_scores),
            "latent_range": [float(min(latent)), float(max(latent))],
            "motif_range": [float(min(motif_scores)), float(max(motif_scores))],
            "inference_time_ms": _model_loader.last_inference_time
        }
        
    except Exception as e:
        logger.error(f"Test inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Test failed: {str(e)}"
        )