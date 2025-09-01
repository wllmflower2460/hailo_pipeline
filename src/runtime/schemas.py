"""
Pydantic models for EdgeInfer-compatible HailoRT inference sidecar.

Implements the exact API contract specified in ADR-0007:
- POST /infer: {"x": [[Float; 9]] * 100} -> {"latent":[Float;64], "motif_scores":[Float;M]}
- GET /healthz: {"ok": bool, "model": str}
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Annotated
import math


class IMUWindow(BaseModel):
    """
    IMU window with exactly 100 timesteps × 9 channels
    
    Channel order: [ax, ay, az, gx, gy, gz, mx, my, mz]
    Shape: (100, 9) - 100 timesteps at 100Hz sampling rate
    """
    x: Annotated[
        List[List[float]], 
        Field(description="IMU data as 100x9 matrix")
    ]
    
    @field_validator('x')
    @classmethod
    def validate_shape(cls, v):
        """Validate exact 100x9 shape"""
        if len(v) != 100:
            raise ValueError(f"Expected 100 timesteps, got {len(v)}")
        
        for i, timestep in enumerate(v):
            if len(timestep) != 9:
                raise ValueError(f"Timestep {i} has {len(timestep)} channels, expected 9")
        
        return v
    
    @field_validator('x')
    @classmethod
    def validate_finite_values(cls, v):
        """Ensure all IMU values are finite (no NaN/Inf)"""
        for timestep_idx, timestep in enumerate(v):
            for channel_idx, value in enumerate(timestep):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value at timestep {timestep_idx}, "
                        f"channel {channel_idx}: {value}"
                    )
        return v
    
    @field_validator('x')
    @classmethod
    def validate_reasonable_ranges(cls, v):
        """Basic sanity check for IMU sensor ranges"""
        for timestep_idx, timestep in enumerate(v):
            # Check accelerometer range (±50g is very generous)
            for i in range(3):
                if abs(timestep[i]) > 500:  # m/s²
                    raise ValueError(f"Accelerometer value out of range: {timestep[i]}")
            
            # Check gyroscope range (±2000°/s = ~35 rad/s)
            for i in range(3, 6):
                if abs(timestep[i]) > 35:  # rad/s
                    raise ValueError(f"Gyroscope value out of range: {timestep[i]}")
            
            # Check magnetometer range (±2000µT is very generous)
            for i in range(6, 9):
                if abs(timestep[i]) > 2000:  # µT
                    raise ValueError(f"Magnetometer value out of range: {timestep[i]}")
                    
        return v


class InferResponse(BaseModel):
    """Model inference output matching EdgeInfer expectations"""
    latent: List[float] = Field(..., description="64-dimensional latent embedding")
    motif_scores: List[float] = Field(..., description="Motif probability scores")
    
    @field_validator('latent')
    @classmethod
    def validate_latent_dim(cls, v):
        """Ensure latent vector has exactly 64 dimensions"""
        if len(v) != 64:
            raise ValueError(f"Latent vector must have 64 dimensions, got {len(v)}")
        return v
    
    @field_validator('motif_scores')
    @classmethod
    def validate_motif_scores_range(cls, v):
        """Ensure motif scores are valid probabilities [0,1]"""
        for i, score in enumerate(v):
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Motif score {i} out of range [0,1]: {score}")
        return v


class HealthResponse(BaseModel):
    """Health check response for EdgeInfer monitoring"""
    ok: bool = Field(..., description="Model is ready for inference")
    model: str = Field(..., description="Loaded model identifier")
    uptime_s: int = Field(..., description="Service uptime in seconds")
    config_version: str = Field(..., description="Configuration version identifier")
    hef_sha256: str = Field(..., description="HEF model file SHA256 hash")
    version: str = Field(default="1.0.0", description="Sidecar version")
    device: str = Field(default="hailo8", description="Hardware accelerator")
    latency_ms: float = Field(default=0.0, description="Last inference latency")


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(..., description="Error message")
    detail: str = Field(default="", description="Additional error details")
    model_available: bool = Field(default=False, description="Model loading status")