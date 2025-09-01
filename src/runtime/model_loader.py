"""
HailoRT model loader and inference engine for TCN-VAE encoder.

Handles loading .hef files and running inference with proper normalization
using the exact parameters from TCN-VAE_models/v0.1.0/normalization.json.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import asyncio
import time
import json
import hashlib

# HailoRT imports - graceful fallback for development
try:
    from hailo_platform import (
        VDevice, HefModel, InferInterface,
        InputVStreamParams, OutputVStreamParams
    )
    HAILORT_AVAILABLE = True
except ImportError:
    HAILORT_AVAILABLE = False
    logging.warning("HailoRT SDK not available - using stub inference mode")

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    HailoRT model loader and inference engine for TCN-VAE encoder
    
    Features:
    - Loads .hef compiled models via HailoRT
    - Applies exact normalization from training metadata
    - Provides async inference with proper error handling
    - Graceful fallback to stub mode for development
    """
    
    # Normalization parameters from TCN-VAE_models v0.1.0
    # These MUST match exactly for inference parity
    NORMALIZATION_PARAMS = {
        "zscore_mean": [0.12, -0.08, 9.78, 0.002, -0.001, 0.003, 22.4, -8.7, 43.2],
        "zscore_std": [3.92, 3.87, 2.45, 1.24, 1.31, 0.98, 28.5, 31.2, 24.8],
        "channel_order": ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz"]
    }
    
    def __init__(self, hef_path: str, num_motifs: int = 12):
        self.hef_path = Path(hef_path)
        self.num_motifs = num_motifs
        self.model_name = self.hef_path.stem if self.hef_path.exists() else "tcn_encoder_stub"
        
        # HailoRT components
        self.device: Optional[Any] = None
        self.model: Optional[Any] = None
        self.infer_interface: Optional[Any] = None
        
        # State tracking
        self._ready = False
        self._initialization_error: Optional[str] = None
        self.last_inference_time: float = 0.0
        
        # Enhanced health tracking
        self.start_time = time.time()
        self.config_version = "hailo_pipeline_production_config-2025-09-01"
        self.hef_sha256 = self._compute_hef_hash()
        
        # Normalization arrays (precomputed for performance)
        self.norm_mean = np.array(self.NORMALIZATION_PARAMS["zscore_mean"], dtype=np.float32)
        self.norm_std = np.array(self.NORMALIZATION_PARAMS["zscore_std"], dtype=np.float32)
        
        logger.info(f"ModelLoader initialized - HEF: {self.hef_path}, Motifs: {num_motifs}")
        logger.info(f"HEF SHA256: {self.hef_sha256}")
        logger.info(f"Config version: {self.config_version}")

    def _compute_hef_hash(self) -> str:
        """Compute SHA256 hash of HEF file for integrity verification"""
        if not self.hef_path.exists():
            return "file_not_found"
        
        try:
            hash_sha256 = hashlib.sha256()
            with open(self.hef_path, "rb") as f:
                # Read file in chunks to handle large HEF files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()[:16]  # First 16 chars for readability
        except Exception as e:
            logger.error(f"Failed to compute HEF hash: {e}")
            return "hash_error"

    def get_uptime(self) -> int:
        """Get service uptime in seconds"""
        return int(time.time() - self.start_time)
        
    async def initialize(self):
        """Initialize Hailo device and load model"""
        try:
            if not HAILORT_AVAILABLE:
                logger.warning("HailoRT not available - enabling stub inference mode")
                self._ready = True
                return
                
            if not self.hef_path.exists():
                error_msg = f"HEF file not found: {self.hef_path}"
                logger.error(error_msg)
                self._initialization_error = error_msg
                raise FileNotFoundError(error_msg)
                
            logger.info(f"Initializing HailoRT with model: {self.hef_path}")
            
            # Initialize Hailo device
            self.device = VDevice()
            logger.debug("✅ Hailo device initialized")
            
            # Load HEF model
            self.model = HefModel(str(self.hef_path))
            logger.debug("✅ HEF model loaded")
            
            # Create inference interface
            self.infer_interface = InferInterface(
                self.model, 
                device=self.device
            )
            logger.debug("✅ Inference interface created")
            
            # Configure input/output streams
            input_params = InputVStreamParams.from_model(self.model)
            output_params = OutputVStreamParams.from_model(self.model)
            
            await self.infer_interface.configure_streams(input_params, output_params)
            logger.debug("✅ Streams configured")
            
            self._ready = True
            logger.info(f"🚀 HailoRT model ready: {self.model_name}")
            
        except Exception as e:
            error_msg = f"Failed to initialize HailoRT model: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self._initialization_error = error_msg
            self._ready = False
            raise RuntimeError(error_msg) from e
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self._ready
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed model status"""
        return {
            "ready": self._ready,
            "model_name": self.model_name,
            "hef_path": str(self.hef_path),
            "hef_exists": self.hef_path.exists(),
            "hailort_available": HAILORT_AVAILABLE,
            "initialization_error": self._initialization_error,
            "last_inference_time": self.last_inference_time,
            "num_motifs": self.num_motifs
        }
    
    def _normalize_imu_data(self, imu_window: np.ndarray) -> np.ndarray:
        """
        Apply per-channel z-score normalization
        
        Critical: Must use exact same μ/σ values as training for inference parity
        """
        # Input shape: (100, 9) or (1, 100, 9)
        if imu_window.ndim == 3:
            imu_window = imu_window.squeeze(0)  # Remove batch dim if present
            
        # Apply per-channel normalization: (x - μ) / σ
        normalized = (imu_window - self.norm_mean) / self.norm_std
        
        logger.debug(f"Applied normalization - Input range: [{imu_window.min():.3f}, {imu_window.max():.3f}], "
                    f"Output range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        return normalized.astype(np.float32)
    
    async def infer(self, imu_window: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on IMU window
        
        Args:
            imu_window: List of 100 timesteps × 9 channels (raw sensor data)
            
        Returns:
            (latent_embeddings, motif_scores) as numpy arrays
            - latent_embeddings: shape (64,) float32
            - motif_scores: shape (num_motifs,) float32 in [0,1]
        """
        start_time = time.time()
        
        if not self._ready:
            raise RuntimeError(f"Model not ready for inference: {self._initialization_error}")
            
        try:
            # Convert to numpy array and validate shape
            input_data = np.array(imu_window, dtype=np.float32)
            if input_data.shape != (100, 9):
                raise ValueError(f"Invalid input shape: {input_data.shape}, expected (100, 9)")
            
            # Apply normalization (critical for accuracy)
            normalized_data = self._normalize_imu_data(input_data)
            
            if not HAILORT_AVAILABLE:
                # Stub inference for development/testing
                await asyncio.sleep(0.005)  # Simulate ~5ms inference latency
                
                # Generate realistic stub outputs
                latent = np.random.randn(64).astype(np.float32)
                latent = np.clip(latent, -4.0, 4.0)  # Typical latent space range
                
                motif_scores = np.random.beta(2, 5, self.num_motifs).astype(np.float32)
                motif_scores = np.clip(motif_scores, 0.0, 1.0)
                
                self.last_inference_time = (time.time() - start_time) * 1000  # ms
                logger.debug(f"Stub inference completed in {self.last_inference_time:.2f}ms")
                
                return latent, motif_scores
            
            # Real HailoRT inference
            batch_input = normalized_data.reshape(1, 100, 9)  # Add batch dimension
            
            # Run inference on Hailo accelerator
            outputs = await self.infer_interface.infer({
                "imu_window": batch_input
            })
            
            # Extract outputs (remove batch dimension)
            latent = outputs["latent_embeddings"][0]  # (64,)
            motif_scores = outputs["motif_scores"][0]  # (num_motifs,)
            
            # Ensure outputs are in expected ranges
            latent = np.clip(latent, -10.0, 10.0)  # Reasonable latent bounds
            motif_scores = np.clip(motif_scores, 0.0, 1.0)  # Probability bounds
            
            self.last_inference_time = (time.time() - start_time) * 1000  # ms
            logger.debug(f"HailoRT inference completed in {self.last_inference_time:.2f}ms")
            
            return latent, motif_scores
            
        except Exception as e:
            error_msg = f"Inference failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def cleanup(self):
        """Clean up HailoRT resources"""
        logger.info("Cleaning up HailoRT resources...")
        
        try:
            if self.infer_interface:
                await self.infer_interface.cleanup()
                self.infer_interface = None
                
            if self.device:
                self.device.release()
                self.device = None
                
            self.model = None
            self._ready = False
            
            logger.info("✅ HailoRT cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")
    
    def __repr__(self) -> str:
        status = "ready" if self._ready else "not ready"
        return f"ModelLoader({self.model_name}, {status}, motifs={self.num_motifs})"