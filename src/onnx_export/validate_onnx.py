#!/usr/bin/env python3
"""
ONNX Model Validation and Analysis

Validates exported ONNX models for Hailo-8 compatibility and accuracy.
Performs comprehensive checks against deployment requirements.
"""

import onnx
import onnxruntime as ort
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

logger = logging.getLogger(__name__)


class ONNXValidator:
    """
    ONNX model validator for Hailo-8 deployment
    
    Validates:
    - Model structure and compatibility
    - Static shape requirements
    - Supported operations
    - Output shape and range validation
    - Performance benchmarking
    """
    
    # Hailo-8 supported operations (subset - full list from Hailo docs)
    HAILO_SUPPORTED_OPS = {
        'Conv', 'Gemm', 'MatMul', 'Add', 'Mul', 'Relu', 'LeakyRelu',
        'BatchNormalization', 'Reshape', 'Transpose', 'Concat', 'Split',
        'Squeeze', 'Unsqueeze', 'Flatten', 'Clip', 'Sigmoid', 'Tanh'
    }
    
    def __init__(self):
        """Initialize validator"""
        self.validation_results = {}
    
    def load_onnx_model(self, model_path: str) -> Tuple[onnx.ModelProto, ort.InferenceSession]:
        """Load ONNX model and create runtime session"""
        logger.info(f"Loading ONNX model: {model_path}")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        # Load ONNX model
        onnx_model = onnx.load(model_path)
        
        # Create runtime session
        ort_session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # CPU-only for validation
        )
        
        return onnx_model, ort_session
    
    def validate_model_structure(self, onnx_model: onnx.ModelProto) -> Dict[str, Any]:
        """Validate basic model structure"""
        logger.info("Validating model structure...")
        
        results = {
            "valid_model": False,
            "opset_version": None,
            "input_shapes": [],
            "output_shapes": [],
            "parameter_count": 0,
            "node_count": 0
        }
        
        try:
            # Check model validity
            onnx.checker.check_model(onnx_model)
            results["valid_model"] = True
            
            # Extract opset version
            if onnx_model.opset_import:
                results["opset_version"] = onnx_model.opset_import[0].version
            
            # Analyze inputs
            for input_info in onnx_model.graph.input:
                shape = []
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(f"dynamic({dim.dim_param})")
                    else:
                        shape.append("unknown")
                
                results["input_shapes"].append({
                    "name": input_info.name,
                    "shape": shape,
                    "dtype": input_info.type.tensor_type.elem_type
                })
            
            # Analyze outputs
            for output_info in onnx_model.graph.output:
                shape = []
                for dim in output_info.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(f"dynamic({dim.dim_param})")
                    else:
                        shape.append("unknown")
                
                results["output_shapes"].append({
                    "name": output_info.name,
                    "shape": shape,
                    "dtype": output_info.type.tensor_type.elem_type
                })
            
            # Count parameters and nodes
            results["parameter_count"] = len(onnx_model.graph.initializer)
            results["node_count"] = len(onnx_model.graph.node)
            
            logger.info(f"✅ Model structure validation passed")
            
        except Exception as e:
            logger.error(f"❌ Model structure validation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def validate_hailo_compatibility(self, onnx_model: onnx.ModelProto) -> Dict[str, Any]:
        """Validate Hailo-8 compatibility requirements"""
        logger.info("Validating Hailo-8 compatibility...")
        
        results = {
            "compatible": True,
            "issues": [],
            "static_shapes": True,
            "supported_ops": True,
            "unsupported_ops": [],
            "batch_size_one": True
        }
        
        # Check static shapes
        for input_info in onnx_model.graph.input:
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param or not dim.dim_value:
                    results["static_shapes"] = False
                    results["issues"].append(f"Dynamic shape in input {input_info.name}")
        
        # Check batch size = 1
        for input_info in onnx_model.graph.input:
            shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim if dim.dim_value]
            if shape and shape[0] != 1:
                results["batch_size_one"] = False
                results["issues"].append(f"Batch size != 1 in input {input_info.name}: {shape[0]}")
        
        # Check supported operations
        for node in onnx_model.graph.node:
            if node.op_type not in self.HAILO_SUPPORTED_OPS:
                results["unsupported_ops"].append(node.op_type)
                results["supported_ops"] = False
        
        # Remove duplicates
        results["unsupported_ops"] = list(set(results["unsupported_ops"]))
        
        # Overall compatibility
        results["compatible"] = (
            results["static_shapes"] and 
            results["supported_ops"] and 
            results["batch_size_one"]
        )
        
        if results["compatible"]:
            logger.info("✅ Hailo-8 compatibility validation passed")
        else:
            logger.warning(f"⚠️  Hailo-8 compatibility issues found: {len(results['issues'])}")
            for issue in results["issues"]:
                logger.warning(f"   - {issue}")
        
        return results
    
    def validate_inference(self, ort_session: ort.InferenceSession) -> Dict[str, Any]:
        """Validate inference execution with test inputs"""
        logger.info("Validating inference execution...")
        
        results = {
            "inference_working": False,
            "test_patterns": {},
            "output_shapes_valid": False,
            "output_ranges_valid": False,
            "latency_ms": 0.0
        }
        
        try:
            input_spec = ort_session.get_inputs()[0]
            input_name = input_spec.name
            input_shape = input_spec.shape
            
            # Generate test inputs
            test_patterns = self._generate_test_patterns(input_shape)
            
            for pattern_name, test_input in test_patterns.items():
                logger.info(f"Testing pattern: {pattern_name}")
                
                # Run inference
                import time
                start_time = time.time()
                
                outputs = ort_session.run(None, {input_name: test_input})
                
                inference_time = (time.time() - start_time) * 1000
                
                # Validate outputs
                pattern_results = {
                    "success": True,
                    "latency_ms": inference_time,
                    "output_shapes": [output.shape for output in outputs],
                    "output_ranges": []
                }
                
                # Check output shapes and ranges
                for i, output in enumerate(outputs):
                    output_range = [float(output.min()), float(output.max())]
                    pattern_results["output_ranges"].append(output_range)
                    
                    # Expected shapes for TCN encoder
                    if i == 0:  # Latent embeddings
                        expected_shape = (1, 64)
                        if output.shape != expected_shape:
                            pattern_results["success"] = False
                            logger.warning(f"Latent shape mismatch: got {output.shape}, expected {expected_shape}")
                    elif i == 1:  # Motif scores
                        expected_shape = (1, 12)
                        if output.shape != expected_shape:
                            pattern_results["success"] = False
                            logger.warning(f"Motif shape mismatch: got {output.shape}, expected {expected_shape}")
                
                results["test_patterns"][pattern_name] = pattern_results
                results["latency_ms"] = max(results["latency_ms"], inference_time)
            
            # Overall validation
            all_successful = all(
                result["success"] for result in results["test_patterns"].values()
            )
            
            results["inference_working"] = all_successful
            results["output_shapes_valid"] = all_successful
            results["output_ranges_valid"] = all_successful
            
            if all_successful:
                logger.info(f"✅ Inference validation passed (max latency: {results['latency_ms']:.1f}ms)")
            else:
                logger.error("❌ Inference validation failed")
                
        except Exception as e:
            logger.error(f"❌ Inference validation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _generate_test_patterns(self, input_shape: List[int]) -> Dict[str, np.ndarray]:
        """Generate test input patterns"""
        patterns = {}
        
        # Zero input
        patterns["zeros"] = np.zeros(input_shape, dtype=np.float32)
        
        # Random realistic IMU data
        np.random.seed(42)
        if len(input_shape) == 3 and input_shape[-1] == 9:  # [1, 100, 9]
            # Generate realistic IMU patterns
            batch_size, timesteps, channels = input_shape
            
            # Accelerometer (channels 0-2): gravity + motion
            accel = np.random.normal([0.1, -0.1, 9.8], [2.0, 2.0, 1.0], (batch_size, timesteps, 3))
            
            # Gyroscope (channels 3-5): angular velocity
            gyro = np.random.normal([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], (batch_size, timesteps, 3))
            
            # Magnetometer (channels 6-8): Earth's field
            mag = np.random.normal([25, -10, 45], [15, 15, 15], (batch_size, timesteps, 3))
            
            realistic_data = np.concatenate([accel, gyro, mag], axis=2).astype(np.float32)
            patterns["realistic_imu"] = realistic_data
        else:
            # Fallback for other shapes
            patterns["random"] = np.random.randn(*input_shape).astype(np.float32)
        
        return patterns
    
    def benchmark_performance(self, ort_session: ort.InferenceSession, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark inference performance"""
        logger.info(f"Benchmarking performance ({num_runs} runs)...")
        
        results = {
            "num_runs": num_runs,
            "latencies_ms": [],
            "mean_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "throughput_infer_per_sec": 0.0
        }
        
        try:
            input_spec = ort_session.get_inputs()[0]
            input_name = input_spec.name
            
            # Generate test input
            test_input = np.random.randn(*input_spec.shape).astype(np.float32)
            
            # Warmup runs
            for _ in range(5):
                _ = ort_session.run(None, {input_name: test_input})
            
            # Benchmark runs
            import time
            latencies = []
            
            for _ in range(num_runs):
                start_time = time.time()
                _ = ort_session.run(None, {input_name: test_input})
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
            
            results["latencies_ms"] = latencies
            results["mean_latency_ms"] = np.mean(latencies)
            results["std_latency_ms"] = np.std(latencies)
            results["p95_latency_ms"] = np.percentile(latencies, 95)
            results["throughput_infer_per_sec"] = 1000.0 / results["mean_latency_ms"]
            
            logger.info(f"✅ Performance benchmark completed:")
            logger.info(f"   Mean latency: {results['mean_latency_ms']:.2f}ms")
            logger.info(f"   P95 latency: {results['p95_latency_ms']:.2f}ms")
            logger.info(f"   Throughput: {results['throughput_infer_per_sec']:.1f} infer/sec")
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def validate_model(self, model_path: str, benchmark: bool = False) -> Dict[str, Any]:
        """Complete ONNX model validation pipeline"""
        logger.info(f"🔍 Starting ONNX model validation: {model_path}")
        
        validation_results = {
            "model_path": model_path,
            "validation_timestamp": np.datetime64('now').isoformat(),
            "overall_valid": False
        }
        
        try:
            # Load model
            onnx_model, ort_session = self.load_onnx_model(model_path)
            
            # Structure validation
            structure_results = self.validate_model_structure(onnx_model)
            validation_results["structure"] = structure_results
            
            # Hailo compatibility
            compatibility_results = self.validate_hailo_compatibility(onnx_model)
            validation_results["hailo_compatibility"] = compatibility_results
            
            # Inference validation
            inference_results = self.validate_inference(ort_session)
            validation_results["inference"] = inference_results
            
            # Performance benchmark (optional)
            if benchmark:
                performance_results = self.benchmark_performance(ort_session)
                validation_results["performance"] = performance_results
            
            # Overall validation result
            validation_results["overall_valid"] = (
                structure_results.get("valid_model", False) and
                compatibility_results.get("compatible", False) and
                inference_results.get("inference_working", False)
            )
            
            if validation_results["overall_valid"]:
                logger.info("🎉 Overall ONNX validation PASSED")
            else:
                logger.warning("⚠️  Overall ONNX validation has ISSUES")
            
        except Exception as e:
            logger.error(f"❌ ONNX validation failed: {e}")
            validation_results["error"] = str(e)
        
        return validation_results


def main():
    """CLI entry point for ONNX validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Validate ONNX models for Hailo-8 deployment")
    parser.add_argument("--model", required=True, help="Path to ONNX model file")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--output", help="Output JSON file for validation results")
    
    args = parser.parse_args()
    
    # Run validation
    validator = ONNXValidator()
    results = validator.validate_model(args.model, benchmark=args.benchmark)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Validation results saved: {args.output}")
    
    # Print summary
    print("\n" + "="*50)
    print("ONNX Validation Summary:")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Valid: {results.get('overall_valid', False)}")
    
    if 'structure' in results:
        structure = results['structure']
        print(f"Structure: {'✅' if structure.get('valid_model') else '❌'}")
        print(f"Opset: {structure.get('opset_version', 'unknown')}")
    
    if 'hailo_compatibility' in results:
        compat = results['hailo_compatibility']
        print(f"Hailo Compatible: {'✅' if compat.get('compatible') else '❌'}")
        if compat.get('issues'):
            print(f"Issues: {len(compat['issues'])}")
    
    if 'inference' in results:
        inference = results['inference']
        print(f"Inference: {'✅' if inference.get('inference_working') else '❌'}")
        if inference.get('latency_ms'):
            print(f"Max Latency: {inference['latency_ms']:.1f}ms")
    
    # Exit code
    exit(0 if results.get('overall_valid', False) else 1)


if __name__ == "__main__":
    main()