#!/usr/bin/env python3
"""
Production Performance Validation Suite

Comprehensive testing of Hailo TCN inference performance on Raspberry Pi + Hailo-8.
Validates latency, throughput, accuracy, and stability requirements from ADR-0007.
"""

import asyncio
import aiohttp
import time
import json
import numpy as np
import statistics
import psutil
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import subprocess
import socket


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    timestamp: str
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    throughput_rps: float
    duration_seconds: float
    
    # System metrics during test
    cpu_usage_percent: float
    memory_usage_mb: float
    system_load: float
    
    # Validation against targets
    latency_target_met: bool
    throughput_target_met: bool
    stability_target_met: bool


class HailoPerformanceValidator:
    """
    Comprehensive performance validator for Hailo TCN inference
    
    Tests:
    - Inference latency under various loads
    - Sustained throughput performance
    - Memory and CPU usage patterns
    - Error rates and stability
    - Real IMU data processing
    - EdgeInfer integration performance
    """
    
    def __init__(self, hailo_url: str = "http://localhost:9000", 
                 edgeinfer_url: str = "http://localhost:8080"):
        self.hailo_url = hailo_url
        self.edgeinfer_url = edgeinfer_url
        self.logger = logging.getLogger(__name__)
        
        # Performance targets from ADR-0007
        self.targets = {
            "p95_latency_ms": 50,
            "throughput_rps": 20,
            "success_rate": 99.0,
            "max_memory_mb": 512,
            "max_cpu_percent": 80
        }
        
        # Test results storage
        self.test_results: List[PerformanceMetrics] = []
        
    async def generate_realistic_imu_data(self, pattern: str = "walking") -> List[List[float]]:
        """Generate realistic IMU data patterns for testing"""
        if pattern == "walking":
            # Simulate walking gait pattern
            timesteps = []
            for t in range(100):
                phase = 2 * np.pi * t / 50  # 50-step gait cycle
                
                # Accelerometer (gravity + motion)
                ax = 0.1 + 0.8 * np.sin(phase)
                ay = -0.05 + 0.6 * np.sin(2 * phase + np.pi/4)  
                az = 9.8 + 0.3 * np.sin(4 * phase)
                
                # Gyroscope (angular velocity)
                gx = 0.05 * np.sin(phase + np.pi/3)
                gy = 0.03 * np.cos(phase)
                gz = 0.1 * np.sin(2 * phase)
                
                # Magnetometer (Earth's field with orientation)
                mx = 25 + 3 * np.sin(phase/2)
                my = -8 + 2 * np.cos(phase/3)
                mz = 43 + 2 * np.sin(phase/4)
                
                # Add realistic noise
                noise = np.random.normal(0, 0.01, 9)
                timestep = [ax, ay, az, gx, gy, gz, mx, my, mz] + noise
                timesteps.append(timestep.tolist())
                
            return timesteps
            
        elif pattern == "running":
            # Higher frequency and amplitude for running
            timesteps = []
            for t in range(100):
                phase = 2 * np.pi * t / 30  # Faster gait
                
                ax = 0.2 + 2.0 * np.sin(phase)
                ay = -0.1 + 1.5 * np.sin(2 * phase + np.pi/4)
                az = 9.8 + 1.0 * np.sin(4 * phase)
                
                gx = 0.2 * np.sin(phase + np.pi/3)
                gy = 0.15 * np.cos(phase)
                gz = 0.3 * np.sin(2 * phase)
                
                mx = 25 + 8 * np.sin(phase/2)
                my = -8 + 5 * np.cos(phase/3)
                mz = 43 + 6 * np.sin(phase/4)
                
                noise = np.random.normal(0, 0.02, 9)
                timestep = [ax, ay, az, gx, gy, gz, mx, my, mz] + noise
                timesteps.append(timestep.tolist())
                
            return timesteps
            
        elif pattern == "stationary":
            # Device at rest with only noise
            timesteps = []
            for _ in range(100):
                # Gravity + small noise
                ax = 0.12 + np.random.normal(0, 0.05)
                ay = -0.08 + np.random.normal(0, 0.05)
                az = 9.78 + np.random.normal(0, 0.02)
                
                # Near-zero rotation
                gx = np.random.normal(0, 0.01)
                gy = np.random.normal(0, 0.01)
                gz = np.random.normal(0, 0.01)
                
                # Earth magnetic field
                mx = 22.4 + np.random.normal(0, 1.0)
                my = -8.7 + np.random.normal(0, 1.0)
                mz = 43.2 + np.random.normal(0, 0.8)
                
                timestep = [ax, ay, az, gx, gy, gz, mx, my, mz]
                timesteps.append(timestep)
                
            return timesteps
        else:
            # Random data within sensor ranges
            return [[np.random.uniform(-10, 10) for _ in range(9)] for _ in range(100)]
    
    async def test_single_inference(self, session: aiohttp.ClientSession, 
                                  pattern: str = "walking") -> Tuple[float, bool, Dict]:
        """Test single inference request"""
        imu_data = await self.generate_realistic_imu_data(pattern)
        payload = {"x": imu_data}
        
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.hailo_url}/infer",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    latency = (time.time() - start_time) * 1000
                    
                    # Validate response structure
                    if ("latent" in result and len(result["latent"]) == 64 and
                        "motif_scores" in result and len(result["motif_scores"]) == 12):
                        return latency, True, result
                    else:
                        return latency, False, {"error": "invalid response structure"}
                else:
                    return (time.time() - start_time) * 1000, False, {"error": f"HTTP {response.status}"}
                    
        except asyncio.TimeoutError:
            return 5000, False, {"error": "timeout"}
        except Exception as e:
            return (time.time() - start_time) * 1000, False, {"error": str(e)}
    
    async def test_latency_performance(self, num_requests: int = 100) -> PerformanceMetrics:
        """Test inference latency performance"""
        self.logger.info(f"Testing latency performance ({num_requests} requests)...")
        
        start_time = time.time()
        latencies = []
        successes = 0
        failures = 0
        
        # Record system metrics before test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        async with aiohttp.ClientSession() as session:
            for i in range(num_requests):
                pattern = ["walking", "running", "stationary"][i % 3]
                latency, success, result = await self.test_single_inference(session, pattern)
                
                latencies.append(latency)
                if success:
                    successes += 1
                else:
                    failures += 1
                
                # Log progress every 10 requests
                if (i + 1) % 10 == 0:
                    avg_latency = statistics.mean(latencies)
                    self.logger.info(f"Progress: {i+1}/{num_requests}, avg latency: {avg_latency:.1f}ms")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
        
        # Record system metrics after test
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        cpu_percent = psutil.cpu_percent(interval=1)
        system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        
        duration = time.time() - start_time
        success_rate = (successes / num_requests) * 100
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = max(latencies)
        min_latency = min(latencies)
        throughput = num_requests / duration
        
        # Validate against targets
        latency_target_met = p95_latency <= self.targets["p95_latency_ms"]
        throughput_target_met = throughput >= self.targets["throughput_rps"]
        stability_target_met = success_rate >= self.targets["success_rate"]
        
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat() + "Z",
            test_name="latency_performance",
            total_requests=num_requests,
            successful_requests=successes,
            failed_requests=failures,
            success_rate=success_rate,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_rps=throughput,
            duration_seconds=duration,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=final_memory - initial_memory,
            system_load=system_load,
            latency_target_met=latency_target_met,
            throughput_target_met=throughput_target_met,
            stability_target_met=stability_target_met
        )
        
        self.test_results.append(metrics)
        return metrics
    
    async def test_sustained_throughput(self, duration_seconds: int = 60) -> PerformanceMetrics:
        """Test sustained throughput performance"""
        self.logger.info(f"Testing sustained throughput ({duration_seconds}s)...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        requests_sent = 0
        successes = 0
        failures = 0
        latencies = []
        
        async with aiohttp.ClientSession() as session:
            while time.time() < end_time:
                pattern = ["walking", "running", "stationary"][requests_sent % 3]
                latency, success, result = await self.test_single_inference(session, pattern)
                
                requests_sent += 1
                latencies.append(latency)
                
                if success:
                    successes += 1
                else:
                    failures += 1
                
                # Log progress every 5 seconds
                elapsed = time.time() - start_time
                if requests_sent % 50 == 0:
                    current_rps = requests_sent / elapsed
                    avg_latency = statistics.mean(latencies[-50:]) if len(latencies) >= 50 else statistics.mean(latencies)
                    self.logger.info(f"Sustained test: {elapsed:.1f}s, {current_rps:.1f} RPS, {avg_latency:.1f}ms avg")
        
        actual_duration = time.time() - start_time
        success_rate = (successes / requests_sent) * 100 if requests_sent > 0 else 0
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        
        # Calculate statistics
        mean_latency = statistics.mean(latencies) if latencies else 0
        median_latency = statistics.median(latencies) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        p99_latency = np.percentile(latencies, 99) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        throughput = requests_sent / actual_duration
        
        # Validate against targets
        latency_target_met = p95_latency <= self.targets["p95_latency_ms"]
        throughput_target_met = throughput >= self.targets["throughput_rps"]
        stability_target_met = success_rate >= self.targets["success_rate"]
        
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat() + "Z",
            test_name="sustained_throughput",
            total_requests=requests_sent,
            successful_requests=successes,
            failed_requests=failures,
            success_rate=success_rate,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_rps=throughput,
            duration_seconds=actual_duration,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            system_load=system_load,
            latency_target_met=latency_target_met,
            throughput_target_met=throughput_target_met,
            stability_target_met=stability_target_met
        )
        
        self.test_results.append(metrics)
        return metrics
    
    async def test_concurrent_load(self, concurrent_requests: int = 10, total_requests: int = 100) -> PerformanceMetrics:
        """Test concurrent request handling"""
        self.logger.info(f"Testing concurrent load ({concurrent_requests} concurrent, {total_requests} total)...")
        
        start_time = time.time()
        
        async def worker(session: aiohttp.ClientSession, worker_id: int, num_requests: int):
            """Worker coroutine for concurrent requests"""
            results = []
            for i in range(num_requests):
                pattern = ["walking", "running", "stationary"][(worker_id + i) % 3]
                latency, success, result = await self.test_single_inference(session, pattern)
                results.append((latency, success, result))
                await asyncio.sleep(0.01)  # Small delay between requests
            return results
        
        # Calculate requests per worker
        requests_per_worker = total_requests // concurrent_requests
        
        # Launch concurrent workers
        async with aiohttp.ClientSession() as session:
            tasks = []
            for worker_id in range(concurrent_requests):
                task = asyncio.create_task(
                    worker(session, worker_id, requests_per_worker)
                )
                tasks.append(task)
            
            # Wait for all workers to complete
            worker_results = await asyncio.gather(*tasks)
        
        # Aggregate results
        all_latencies = []
        total_successes = 0
        total_failures = 0
        
        for worker_result in worker_results:
            for latency, success, result in worker_result:
                all_latencies.append(latency)
                if success:
                    total_successes += 1
                else:
                    total_failures += 1
        
        actual_total = total_successes + total_failures
        duration = time.time() - start_time
        success_rate = (total_successes / actual_total) * 100 if actual_total > 0 else 0
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_mb = psutil.virtual_memory().used / 1024 / 1024
        system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        
        # Calculate statistics
        mean_latency = statistics.mean(all_latencies) if all_latencies else 0
        median_latency = statistics.median(all_latencies) if all_latencies else 0
        p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0
        p99_latency = np.percentile(all_latencies, 99) if all_latencies else 0
        max_latency = max(all_latencies) if all_latencies else 0
        min_latency = min(all_latencies) if all_latencies else 0
        throughput = actual_total / duration
        
        # Validate against targets
        latency_target_met = p95_latency <= self.targets["p95_latency_ms"]
        throughput_target_met = throughput >= self.targets["throughput_rps"]
        stability_target_met = success_rate >= self.targets["success_rate"]
        
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow().isoformat() + "Z",
            test_name="concurrent_load",
            total_requests=actual_total,
            successful_requests=total_successes,
            failed_requests=total_failures,
            success_rate=success_rate,
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max_latency,
            min_latency_ms=min_latency,
            throughput_rps=throughput,
            duration_seconds=duration,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            system_load=system_load,
            latency_target_met=latency_target_met,
            throughput_target_met=throughput_target_met,
            stability_target_met=stability_target_met
        )
        
        self.test_results.append(metrics)
        return metrics
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "platform": {
                "system": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown",
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage('/').total / (1024**3),
            },
            "hailo_device": {
                "device_exists": subprocess.run(['ls', '/dev/hailo0'], capture_output=True).returncode == 0,
                "driver_loaded": subprocess.run(['lsmod'], capture_output=True, text=True).stdout.find('hailo') != -1,
            },
            "services": {
                "hailo_url": self.hailo_url,
                "edgeinfer_url": self.edgeinfer_url,
            },
            "performance_targets": self.targets
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.test_results:
            return {"error": "No test results available"}
        
        # Aggregate statistics across all tests
        all_latencies = []
        all_success_rates = []
        all_throughputs = []
        
        for result in self.test_results:
            # Weight by number of requests
            weight = result.total_requests
            all_latencies.extend([result.mean_latency_ms] * weight)
            all_success_rates.append(result.success_rate)
            all_throughputs.append(result.throughput_rps)
        
        # Overall performance summary
        overall_summary = {
            "total_tests": len(self.test_results),
            "total_requests": sum(r.total_requests for r in self.test_results),
            "overall_success_rate": statistics.mean(all_success_rates),
            "overall_mean_latency_ms": statistics.mean(all_latencies),
            "overall_p95_latency_ms": np.percentile(all_latencies, 95) if all_latencies else 0,
            "overall_max_throughput_rps": max(all_throughputs) if all_throughputs else 0,
            "targets_met": {
                "latency": all(r.latency_target_met for r in self.test_results),
                "throughput": all(r.throughput_target_met for r in self.test_results),
                "stability": all(r.stability_target_met for r in self.test_results),
            }
        }
        
        return {
            "report_metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "report_type": "hailo_performance_validation",
                "version": "1.0.0"
            },
            "system_info": self.get_system_info(),
            "performance_targets": self.targets,
            "overall_summary": overall_summary,
            "detailed_results": [asdict(result) for result in self.test_results]
        }
    
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete performance validation suite"""
        self.logger.info("🚀 Starting comprehensive performance validation...")
        
        start_time = time.time()
        
        try:
            # Test 1: Basic latency performance
            self.logger.info("📊 Test 1: Latency Performance")
            await self.test_latency_performance(num_requests=50)
            
            # Test 2: Sustained throughput
            self.logger.info("📊 Test 2: Sustained Throughput")  
            await self.test_sustained_throughput(duration_seconds=30)
            
            # Test 3: Concurrent load
            self.logger.info("📊 Test 3: Concurrent Load")
            await self.test_concurrent_load(concurrent_requests=5, total_requests=50)
            
            # Generate comprehensive report
            report = self.generate_performance_report()
            
            total_duration = time.time() - start_time
            self.logger.info(f"✅ Validation completed in {total_duration:.1f}s")
            
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {e}")
            return {"error": str(e), "partial_results": [asdict(r) for r in self.test_results]}


async def main():
    """Main validation execution"""
    parser = argparse.ArgumentParser(description="Hailo TCN Performance Validation")
    parser.add_argument("--hailo-url", default="http://localhost:9000",
                       help="Hailo service URL")
    parser.add_argument("--edgeinfer-url", default="http://localhost:8080", 
                       help="EdgeInfer service URL")
    parser.add_argument("--test", choices=["latency", "throughput", "concurrent", "all"],
                       default="all", help="Specific test to run")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize validator
    validator = HailoPerformanceValidator(args.hailo_url, args.edgeinfer_url)
    
    # Run validation
    if args.test == "all":
        results = await validator.run_full_validation()
    elif args.test == "latency":
        result = await validator.test_latency_performance()
        results = {"single_test": asdict(result)}
    elif args.test == "throughput":
        result = await validator.test_sustained_throughput()
        results = {"single_test": asdict(result)}
    elif args.test == "concurrent":
        result = await validator.test_concurrent_load()
        results = {"single_test": asdict(result)}
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {args.output}")
    
    # Print summary
    if "overall_summary" in results:
        summary = results["overall_summary"]
        print(f"\n🎯 PERFORMANCE VALIDATION SUMMARY")
        print(f"================================")
        print(f"Total requests: {summary['total_requests']}")
        print(f"Success rate: {summary['overall_success_rate']:.1f}%")
        print(f"Mean latency: {summary['overall_mean_latency_ms']:.1f}ms")
        print(f"P95 latency: {summary['overall_p95_latency_ms']:.1f}ms")
        print(f"Max throughput: {summary['overall_max_throughput_rps']:.1f} RPS")
        
        targets_met = summary["targets_met"]
        print(f"\n🎯 Targets:")
        print(f"  Latency: {'✅' if targets_met['latency'] else '❌'}")
        print(f"  Throughput: {'✅' if targets_met['throughput'] else '❌'}")  
        print(f"  Stability: {'✅' if targets_met['stability'] else '❌'}")
        
        all_met = all(targets_met.values())
        print(f"\n{'🎉 ALL TARGETS MET!' if all_met else '⚠️  Some targets not met'}")
    
    # Exit code based on success
    success = "error" not in results
    exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())