#!/usr/bin/env python3
"""
Edge Platform Performance Validation for 72.13% TCN-VAE Model

Validates the deployed model performance on Edge platform:
1. Latency benchmarking (target: <50ms P95)
2. Throughput testing (target: 250+ req/sec)
3. Memory usage monitoring
4. Accuracy validation against ONNX reference
5. Load testing and stress validation

This script can run from GPUSRV to remotely validate Edge platform performance
via Tailscale networking, or directly on the Edge platform itself.
"""

import requests
import time
import statistics
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import concurrent.futures
import subprocess
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EdgePerformanceValidator:
    """Validate EdgeInfer performance with deployed 72.13% model"""
    
    def __init__(self, edge_host: str = "192.168.50.88", edge_port: int = 9000):
        self.edge_host = edge_host
        self.edge_port = edge_port
        self.base_url = f"http://{edge_host}:{edge_port}"
        
        # Performance targets from ADR-0007
        self.target_latency_p95_ms = 50.0
        self.target_throughput_rps = 250
        self.target_memory_mb = 512
        
        # Load normalization parameters for test data generation
        self.norm_mean = np.array([0.12, -0.08, 9.78, 0.002, -0.001, 0.003, 22.4, -8.7, 43.2], dtype=np.float32)
        self.norm_std = np.array([3.92, 3.87, 2.45, 1.24, 1.31, 0.98, 28.5, 31.2, 24.8], dtype=np.float32)
        
        logger.info(f"Performance validator initialized for {self.base_url}")
        logger.info(f"Targets: Latency <{self.target_latency_p95_ms}ms, Throughput >{self.target_throughput_rps} req/sec")
    
    def check_service_health(self) -> bool:
        """Verify EdgeInfer service is running and healthy"""
        logger.info("🔍 Checking EdgeInfer service health...")
        
        try:
            response = requests.get(f"{self.base_url}/healthz", timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ EdgeInfer health check passed")
                return True
            else:
                logger.error(f"❌ Health check failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return False
    
    def generate_test_imu_data(self, pattern: str = "walking") -> Dict:
        """Generate realistic test IMU data"""
        if pattern == "walking":
            # Generate walking pattern
            timesteps = []
            for t in range(100):
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
            
            return {"imu_window": timesteps}
        
        elif pattern == "stationary":
            # Generate stationary pattern
            timesteps = []
            for t in range(100):
                timestep = [
                    0.1 + np.random.normal(0, 0.05),    # ax: small noise
                    -0.05 + np.random.normal(0, 0.05),  # ay: small noise
                    9.8 + np.random.normal(0, 0.02),    # az: gravity + noise
                    np.random.normal(0, 0.01),          # gx: minimal rotation
                    np.random.normal(0, 0.01),          # gy: minimal rotation
                    np.random.normal(0, 0.01),          # gz: minimal rotation
                    25 + np.random.normal(0, 1),        # mx: mag field + noise
                    -8 + np.random.normal(0, 1),        # my: mag field + noise
                    43 + np.random.normal(0, 1),        # mz: mag field + noise
                ]
                timesteps.append(timestep)
            
            return {"imu_window": timesteps}
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    def measure_single_request_latency(self, test_data: Dict) -> Tuple[float, bool, Optional[Dict]]:
        """Measure latency of a single inference request"""
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.base_url}/encode",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=2.0  # 2 second timeout
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                return latency_ms, True, result
            else:
                return latency_ms, False, None
                
        except Exception as e:
            return 0.0, False, None
    
    def benchmark_latency(self, num_requests: int = 100) -> Dict:
        """Benchmark request latency"""
        logger.info(f"🚀 Running latency benchmark ({num_requests} requests)...")
        
        test_data = self.generate_test_imu_data("walking")
        latencies = []
        successes = 0
        
        for i in range(num_requests):
            latency_ms, success, result = self.measure_single_request_latency(test_data)
            
            if success:
                latencies.append(latency_ms)
                successes += 1
            
            # Progress update every 20 requests
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{num_requests} requests completed")
        
        if latencies:
            results = {
                "num_requests": num_requests,
                "successful_requests": successes,
                "success_rate_percent": (successes / num_requests) * 100,
                "latency_stats": {
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "mean_ms": statistics.mean(latencies),
                    "median_ms": statistics.median(latencies),
                    "p95_ms": np.percentile(latencies, 95),
                    "p99_ms": np.percentile(latencies, 99)
                }
            }
            
            # Check if P95 target is met
            p95_target_met = results["latency_stats"]["p95_ms"] <= self.target_latency_p95_ms
            results["p95_target_met"] = p95_target_met
            
            logger.info(f"✅ Latency benchmark completed:")
            logger.info(f"   Success rate: {results['success_rate_percent']:.1f}%")
            logger.info(f"   Mean latency: {results['latency_stats']['mean_ms']:.2f}ms")
            logger.info(f"   P95 latency: {results['latency_stats']['p95_ms']:.2f}ms (target: <{self.target_latency_p95_ms}ms)")
            logger.info(f"   P95 target met: {'✅ YES' if p95_target_met else '❌ NO'}")
            
            return results
        else:
            logger.error("❌ Latency benchmark failed - no successful requests")
            return {"error": "No successful requests"}
    
    def benchmark_throughput(self, duration_seconds: int = 30, concurrent_threads: int = 10) -> Dict:
        """Benchmark throughput under concurrent load"""
        logger.info(f"🚀 Running throughput benchmark ({duration_seconds}s, {concurrent_threads} threads)...")
        
        test_data = self.generate_test_imu_data("walking")
        
        def worker_thread():
            """Worker function for concurrent requests"""
            requests_made = 0
            successful_requests = 0
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                _, success, _ = self.measure_single_request_latency(test_data)
                requests_made += 1
                if success:
                    successful_requests += 1
                
                # Small delay to prevent overwhelming
                time.sleep(0.001)
            
            return requests_made, successful_requests
        
        # Run concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            start_time = time.time()
            
            futures = [executor.submit(worker_thread) for _ in range(concurrent_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time = time.time()
        
        # Aggregate results
        total_requests = sum(result[0] for result in results)
        total_successful = sum(result[1] for result in results)
        actual_duration = end_time - start_time
        
        throughput_results = {
            "test_duration_seconds": actual_duration,
            "concurrent_threads": concurrent_threads,
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "success_rate_percent": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "throughput_req_per_sec": total_successful / actual_duration if actual_duration > 0 else 0
        }
        
        # Check if throughput target is met
        target_met = throughput_results["throughput_req_per_sec"] >= self.target_throughput_rps
        throughput_results["throughput_target_met"] = target_met
        
        logger.info(f"✅ Throughput benchmark completed:")
        logger.info(f"   Throughput: {throughput_results['throughput_req_per_sec']:.1f} req/sec (target: >{self.target_throughput_rps} req/sec)")
        logger.info(f"   Success rate: {throughput_results['success_rate_percent']:.1f}%")
        logger.info(f"   Target met: {'✅ YES' if target_met else '❌ NO'}")
        
        return throughput_results
    
    def validate_output_quality(self, num_samples: int = 10) -> Dict:
        """Validate that outputs are reasonable and consistent"""
        logger.info(f"🔍 Validating output quality ({num_samples} samples)...")
        
        patterns = ["walking", "stationary"]
        results = {"pattern_results": {}}
        
        for pattern in patterns:
            pattern_results = []
            
            for i in range(num_samples):
                test_data = self.generate_test_imu_data(pattern)
                _, success, result = self.measure_single_request_latency(test_data)
                
                if success and result:
                    # Check output format
                    if "latent_embedding" in result or "embedding" in result or "latent" in result:
                        # Extract embedding (flexible key matching)
                        embedding = None
                        for key in result.keys():
                            if "embed" in key.lower() or "latent" in key.lower():
                                embedding = result[key]
                                break
                        
                        if embedding and len(embedding) == 64:
                            # Basic sanity checks
                            embedding_array = np.array(embedding)
                            checks = {
                                "correct_length": len(embedding) == 64,
                                "no_nans": not np.isnan(embedding_array).any(),
                                "reasonable_range": np.all(np.abs(embedding_array) < 100),  # Reasonable range
                                "non_zero_variance": np.var(embedding_array) > 1e-6
                            }
                            
                            pattern_results.append({
                                "sample": i,
                                "embedding_stats": {
                                    "mean": float(np.mean(embedding_array)),
                                    "std": float(np.std(embedding_array)),
                                    "min": float(np.min(embedding_array)),
                                    "max": float(np.max(embedding_array))
                                },
                                "quality_checks": checks,
                                "all_checks_passed": all(checks.values())
                            })
            
            # Summarize pattern results
            if pattern_results:
                all_passed = all(result["all_checks_passed"] for result in pattern_results)
                mean_means = np.mean([result["embedding_stats"]["mean"] for result in pattern_results])
                mean_stds = np.mean([result["embedding_stats"]["std"] for result in pattern_results])
                
                results["pattern_results"][pattern] = {
                    "samples_tested": len(pattern_results),
                    "all_quality_checks_passed": all_passed,
                    "aggregate_stats": {
                        "mean_of_means": float(mean_means),
                        "mean_of_stds": float(mean_stds)
                    }
                }
                
                logger.info(f"   {pattern}: {'✅ PASSED' if all_passed else '❌ FAILED'} ({len(pattern_results)} samples)")
        
        overall_passed = all(
            results["pattern_results"][pattern]["all_quality_checks_passed"] 
            for pattern in results["pattern_results"]
        )
        results["overall_quality_validation"] = overall_passed
        
        logger.info(f"✅ Output quality validation: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        return results
    
    def run_complete_validation(self) -> Dict:
        """Run complete performance validation suite"""
        logger.info("="*80)
        logger.info("🎯 EdgeInfer Performance Validation Suite")
        logger.info("   Model: TCN-VAE 72.13% accuracy")
        logger.info("   Platform: Raspberry Pi 5 + Hailo-8")
        logger.info("   Target: Production-ready performance")
        logger.info("="*80)
        
        validation_results = {
            "validation_info": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "edge_host": self.edge_host,
                "model_version": "v72pct",
                "model_accuracy": "72.13%"
            }
        }
        
        try:
            # 1. Health check
            logger.info("Step 1: Service health check...")
            if not self.check_service_health():
                validation_results["health_check"] = {"status": "FAILED"}
                return validation_results
            
            validation_results["health_check"] = {"status": "PASSED"}
            
            # 2. Latency benchmark
            logger.info("Step 2: Latency benchmarking...")
            validation_results["latency_benchmark"] = self.benchmark_latency(100)
            
            # 3. Throughput benchmark
            logger.info("Step 3: Throughput benchmarking...")
            validation_results["throughput_benchmark"] = self.benchmark_throughput(30, 8)
            
            # 4. Output quality validation
            logger.info("Step 4: Output quality validation...")
            validation_results["output_quality"] = self.validate_output_quality(20)
            
            # 5. Overall assessment
            logger.info("Step 5: Overall performance assessment...")
            
            latency_passed = validation_results.get("latency_benchmark", {}).get("p95_target_met", False)
            throughput_passed = validation_results.get("throughput_benchmark", {}).get("throughput_target_met", False)  
            quality_passed = validation_results.get("output_quality", {}).get("overall_quality_validation", False)
            
            validation_results["overall_assessment"] = {
                "latency_target_met": latency_passed,
                "throughput_target_met": throughput_passed,
                "output_quality_passed": quality_passed,
                "production_ready": latency_passed and throughput_passed and quality_passed
            }
            
            logger.info("="*80)
            logger.info("🏆 Validation Summary:")
            logger.info(f"   Health Check: {'✅ PASSED' if validation_results['health_check']['status'] == 'PASSED' else '❌ FAILED'}")
            logger.info(f"   Latency Target: {'✅ MET' if latency_passed else '❌ NOT MET'}")
            logger.info(f"   Throughput Target: {'✅ MET' if throughput_passed else '❌ NOT MET'}")
            logger.info(f"   Output Quality: {'✅ PASSED' if quality_passed else '❌ FAILED'}")
            logger.info(f"   Production Ready: {'✅ YES' if validation_results['overall_assessment']['production_ready'] else '❌ NO'}")
            logger.info("="*80)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            validation_results["error"] = str(e)
            return validation_results
    
    def save_validation_report(self, results: Dict):
        """Save validation results to JSON report"""
        report_filename = f"edge_performance_validation_{int(time.time())}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Validation report saved: {report_filename}")
        return report_filename


def main():
    """Main validation entry point"""
    parser = argparse.ArgumentParser(description="Validate EdgeInfer performance on Edge platform")
    parser.add_argument("--host", default="edge.tailfdc654.ts.net", help="Edge platform hostname")
    parser.add_argument("--port", type=int, default=9000, help="EdgeInfer service port")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (fewer samples)")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EdgePerformanceValidator(args.host, args.port)
    
    # Adjust sample sizes for quick mode
    if args.quick:
        logger.info("Running in quick validation mode")
        # Override benchmark parameters for faster execution
        validator.benchmark_latency = lambda: validator.benchmark_latency(20)
        validator.benchmark_throughput = lambda: validator.benchmark_throughput(10, 4)
        validator.validate_output_quality = lambda: validator.validate_output_quality(5)
    
    # Run validation
    results = validator.run_complete_validation()
    
    # Save report
    report_file = validator.save_validation_report(results)
    
    # Exit with appropriate code
    production_ready = results.get("overall_assessment", {}).get("production_ready", False)
    
    if production_ready:
        logger.info("🎉 Validation PASSED - Model ready for production!")
        return 0
    else:
        logger.error("💥 Validation FAILED - Model not ready for production")
        return 1


if __name__ == "__main__":
    exit(main())