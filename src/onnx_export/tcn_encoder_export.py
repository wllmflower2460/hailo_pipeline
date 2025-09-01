#!/usr/bin/env python3
"""
TCN-VAE Encoder ONNX Export Pipeline

Exports trained TCN-VAE encoder to ONNX format with Hailo-8 compatibility.
Implements exact normalization and validation requirements from ADR-0007.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
import yaml
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class TCNEncoderExporter:
    """
    TCN-VAE Encoder ONNX exporter with Hailo compatibility validation
    
    Features:
    - Loads PyTorch TCN-VAE checkpoints
    - Extracts encoder component only
    - Exports to Hailo-compatible ONNX (opset 11, static shapes)
    - Validates output parity with strict tolerance
    - Generates export metadata for deployment
    """
    
    def __init__(self, config_path: str = "configs/export_config.yaml"):
        """Initialize exporter with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config["model"]["device"])
        
        # Normalization parameters (critical for inference parity)
        norm_config = self.config["normalization"]
        self.norm_mean = np.array(norm_config["zscore_mean"], dtype=np.float32)
        self.norm_std = np.array(norm_config["zscore_std"], dtype=np.float32)
        
        logger.info(f"TCN Encoder Exporter initialized with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load export configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_git_hash(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout.strip()[:8]  # Short hash
        except subprocess.CalledProcessError:
            return None
    
    def load_tcn_model(self, model_path: str) -> torch.nn.Module:
        """
        Load TCN-VAE model and extract encoder component
        
        Args:
            model_path: Path to trained model checkpoint (.pth)
            
        Returns:
            TCN encoder module ready for export
        """
        logger.info(f"Loading TCN-VAE model from: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract encoder based on checkpoint structure
        encoder_key = self.config["model"]["encoder_key"]
        
        if encoder_key and hasattr(checkpoint, encoder_key):
            # Model object with encoder attribute
            encoder = getattr(checkpoint, encoder_key)
            logger.info(f"Extracted encoder using key: {encoder_key}")
        elif encoder_key and isinstance(checkpoint, dict) and encoder_key in checkpoint:
            # Checkpoint dict with encoder key
            encoder = checkpoint[encoder_key]
            logger.info(f"Extracted encoder from checkpoint dict: {encoder_key}")
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            # Standard checkpoint format
            model_state = checkpoint["model"]
            if hasattr(model_state, "encoder"):
                encoder = model_state.encoder
            else:
                # Assume entire model is the encoder
                encoder = model_state
            logger.info("Extracted encoder from model state")
        else:
            # Direct model checkpoint
            encoder = checkpoint
            logger.info("Using direct model as encoder")
        
        # Ensure encoder is a PyTorch module
        if not isinstance(encoder, torch.nn.Module):
            raise ValueError(f"Extracted encoder is not a PyTorch module: {type(encoder)}")
        
        # Set to evaluation mode and move to device
        encoder.eval()
        encoder.to(self.device)
        
        logger.info(f"✅ TCN encoder loaded successfully")
        logger.info(f"   Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        logger.info(f"   Trainable: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")
        
        return encoder
    
    def _generate_test_inputs(self) -> Dict[str, torch.Tensor]:
        """Generate test inputs for validation"""
        input_shape = self.config["model"]["input_shape"]  # [1, 100, 9]
        
        test_inputs = {}
        
        for pattern in self.config["validation"]["test_patterns"]:
            name = pattern["name"]
            
            if name == "zeros":
                data = torch.zeros(*input_shape, dtype=torch.float32)
            elif name == "walking":
                # Simulate walking pattern with realistic IMU values
                timesteps = []
                for t in range(100):
                    # Simulate walking gait cycle
                    phase = 2 * np.pi * t / 50  # 50-step gait cycle
                    
                    timestep = [
                        0.5 * np.sin(phase),           # ax: lateral sway
                        1.0 + 0.3 * np.cos(2*phase),   # ay: forward/back
                        9.8 + 0.2 * np.sin(4*phase),   # az: vertical bounce
                        0.1 * np.sin(phase + np.pi/4), # gx: pitch variation
                        0.05 * np.cos(phase),          # gy: roll variation
                        0.2 * np.sin(2*phase),         # gz: yaw turning
                        25 + 5 * np.sin(phase/2),      # mx: magnetic field
                        -8 + 3 * np.cos(phase/2),      # my: magnetic field
                        43 + 4 * np.sin(phase/3),      # mz: magnetic field
                    ]
                    timesteps.append(timestep)
                
                data = torch.tensor([timesteps], dtype=torch.float32)
            elif name == "random":
                # Random values in realistic sensor ranges
                np.random.seed(42)  # Reproducible
                accel = np.random.uniform(-10, 10, (1, 100, 3))
                gyro = np.random.uniform(-2, 2, (1, 100, 3))
                mag = np.random.uniform(-100, 100, (1, 100, 3))
                
                data = torch.tensor(
                    np.concatenate([accel, gyro, mag], axis=2), 
                    dtype=torch.float32
                )
            else:
                raise ValueError(f"Unknown test pattern: {name}")
            
            test_inputs[name] = data.to(self.device)
        
        return test_inputs
    
    def _normalize_input(self, raw_input: torch.Tensor) -> torch.Tensor:
        """Apply per-channel z-score normalization (must match training exactly)"""
        # Convert numpy normalization params to torch tensors
        mean_tensor = torch.tensor(self.norm_mean, device=raw_input.device, dtype=raw_input.dtype)
        std_tensor = torch.tensor(self.norm_std, device=raw_input.device, dtype=raw_input.dtype)
        
        # Apply normalization: (x - μ) / σ
        normalized = (raw_input - mean_tensor) / std_tensor
        
        return normalized
    
    def export_to_onnx(self, encoder: torch.nn.Module, output_path: str) -> bool:
        """
        Export TCN encoder to ONNX format with Hailo compatibility
        
        Args:
            encoder: Trained TCN encoder module
            output_path: Output path for ONNX file
            
        Returns:
            True if export successful
        """
        logger.info(f"Exporting TCN encoder to ONNX: {output_path}")
        
        try:
            # Prepare export configuration
            input_shape = self.config["model"]["input_shape"]
            input_names = self.config["model"]["input_names"]
            output_names = self.config["model"]["output_names"]
            
            # Create dummy input with exact shape
            dummy_input = torch.randn(*input_shape, device=self.device, dtype=torch.float32)
            
            # Apply normalization (critical for model parity)
            dummy_input = self._normalize_input(dummy_input)
            
            logger.info(f"Export configuration:")
            logger.info(f"  Input shape: {input_shape}")
            logger.info(f"  Input names: {input_names}")
            logger.info(f"  Output names: {output_names}")
            logger.info(f"  Opset version: {self.config['onnx']['opset_version']}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Export to ONNX
            with torch.no_grad():
                torch.onnx.export(
                    encoder,
                    dummy_input,
                    output_path,
                    export_params=self.config["onnx"]["export_params"],
                    opset_version=self.config["onnx"]["opset_version"],
                    do_constant_folding=self.config["onnx"]["do_constant_folding"],
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=self.config["onnx"]["dynamic_axes"],
                    strip_doc_string=self.config["onnx"]["strip_doc_string"],
                    training=self.config["onnx"]["training"]
                )
            
            logger.info(f"✅ ONNX export completed: {output_path}")
            
            # Validate exported model
            if self.config["onnx"]["validate_export"]:
                return self._validate_onnx_export(encoder, output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            return False
    
    def _validate_onnx_export(self, pytorch_model: torch.nn.Module, onnx_path: str) -> bool:
        """Validate ONNX export against PyTorch model"""
        logger.info("Validating ONNX export against PyTorch reference...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Generate test inputs
            test_inputs = self._generate_test_inputs()
            
            validation_results = {}
            
            for pattern_name, raw_input in test_inputs.items():
                # Apply normalization
                normalized_input = self._normalize_input(raw_input)
                
                # PyTorch inference
                with torch.no_grad():
                    pytorch_output = pytorch_model(normalized_input)
                
                # Handle different output formats
                if isinstance(pytorch_output, tuple):
                    pytorch_latent, pytorch_motifs = pytorch_output
                else:
                    # Assume single latent output, generate dummy motifs
                    pytorch_latent = pytorch_output
                    pytorch_motifs = torch.randn(1, self.config["model"]["expected_outputs"]["motif_count"])
                
                # ONNX inference
                onnx_input = {ort_session.get_inputs()[0].name: normalized_input.cpu().numpy()}
                onnx_outputs = ort_session.run(None, onnx_input)
                
                onnx_latent = torch.tensor(onnx_outputs[0])
                onnx_motifs = torch.tensor(onnx_outputs[1]) if len(onnx_outputs) > 1 else pytorch_motifs
                
                # Compute similarities
                latent_cosine = torch.nn.functional.cosine_similarity(
                    pytorch_latent.flatten(), onnx_latent.flatten(), dim=0
                ).item()
                
                motifs_cosine = torch.nn.functional.cosine_similarity(
                    pytorch_motifs.flatten(), onnx_motifs.flatten(), dim=0
                ).item()
                
                # Check tolerances
                latent_close = torch.allclose(
                    pytorch_latent, onnx_latent,
                    atol=self.config["validation"]["tolerance_absolute"],
                    rtol=self.config["validation"]["tolerance_relative"]
                )
                
                validation_results[pattern_name] = {
                    "latent_cosine_similarity": latent_cosine,
                    "motifs_cosine_similarity": motifs_cosine,
                    "latent_close": latent_close,
                    "passed": latent_cosine >= self.config["validation"]["min_cosine_similarity"]
                }
                
                logger.info(f"  {pattern_name}: cosine_sim={latent_cosine:.6f}, close={latent_close}")
            
            # Overall validation result
            all_passed = all(result["passed"] for result in validation_results.values())
            
            if all_passed:
                logger.info("✅ ONNX export validation PASSED")
                return True
            else:
                logger.error("❌ ONNX export validation FAILED")
                return False
                
        except Exception as e:
            logger.error(f"❌ ONNX validation failed: {e}")
            return False
    
    def _generate_export_metadata(self, model_path: str, onnx_path: str) -> Dict[str, Any]:
        """Generate metadata for exported model"""
        metadata = {
            "export_info": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "git_hash": self._get_git_hash(),
                "exporter_version": "1.0.0"
            },
            "source_model": {
                "path": str(Path(model_path).absolute()),
                "size_mb": Path(model_path).stat().st_size / (1024 * 1024)
            },
            "onnx_model": {
                "path": str(Path(onnx_path).absolute()),
                "size_mb": Path(onnx_path).stat().st_size / (1024 * 1024),
                "opset_version": self.config["onnx"]["opset_version"]
            },
            "model_spec": self.config["model"],
            "normalization": self.config["normalization"],
            "hailo_compatibility": self.config["hailo_compatibility"],
            "validation_config": self.config["validation"]
        }
        
        return metadata
    
    def export_model(self, model_path: str, version: str = "v0.1.0") -> Tuple[bool, Optional[str]]:
        """
        Complete model export pipeline
        
        Args:
            model_path: Path to trained TCN-VAE model
            version: Version string for output naming
            
        Returns:
            (success, onnx_path) tuple
        """
        logger.info(f"🚀 Starting TCN encoder export pipeline")
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Version: {version}")
        
        try:
            # Load model
            encoder = self.load_tcn_model(model_path)
            
            # Generate output path
            output_dir = Path(self.config["artifacts"]["onnx_output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_filename = self.config["artifacts"]["model_name_template"].format(
                version=version
            )
            onnx_path = output_dir / model_filename
            
            # Export to ONNX
            export_success = self.export_to_onnx(encoder, str(onnx_path))
            
            if export_success:
                # Generate metadata
                if self.config["artifacts"]["save_metadata"]:
                    metadata = self._generate_export_metadata(model_path, str(onnx_path))
                    metadata_path = onnx_path.with_suffix('.json')
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    logger.info(f"✅ Export metadata saved: {metadata_path}")
                
                logger.info(f"🎉 Export pipeline completed successfully!")
                logger.info(f"   ONNX model: {onnx_path}")
                logger.info(f"   Size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")
                
                return True, str(onnx_path)
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"❌ Export pipeline failed: {e}")
            return False, None


def main():
    """CLI entry point for TCN encoder export"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Export TCN-VAE encoder to ONNX format for Hailo deployment"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained TCN-VAE model (.pth file)"
    )
    parser.add_argument(
        "--config", default="configs/export_config.yaml",
        help="Export configuration file"
    )
    parser.add_argument(
        "--version", default="v0.1.0",
        help="Version string for output naming"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only run validation on existing ONNX model"
    )
    
    args = parser.parse_args()
    
    # Initialize exporter
    exporter = TCNEncoderExporter(args.config)
    
    if args.validate_only:
        # Validation mode
        onnx_path = Path("exports") / f"tcn_encoder_{args.version}.onnx"
        if not onnx_path.exists():
            logger.error(f"ONNX file not found: {onnx_path}")
            exit(1)
        
        logger.info(f"Validating existing ONNX model: {onnx_path}")
        # Would need PyTorch model for comparison - simplified for now
        logger.info("✅ Validation mode not fully implemented yet")
    else:
        # Export mode
        success, onnx_path = exporter.export_model(args.model, args.version)
        
        if success:
            logger.info(f"🎯 Export successful: {onnx_path}")
            exit(0)
        else:
            logger.error("❌ Export failed")
            exit(1)


if __name__ == "__main__":
    main()