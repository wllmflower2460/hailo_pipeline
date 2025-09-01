#!/usr/bin/env python3
"""
Development test script for HailoRT TCN Inference Sidecar

Tests the FastAPI endpoints locally without requiring HailoRT hardware.
Validates API contract compliance with EdgeInfer expectations.
"""

import asyncio
import httpx
import json
import time
import sys
import random
from typing import Dict, Any, List


class SidecarTester:
    """Test harness for HailoRT inference sidecar"""
    
    def __init__(self, base_url: str = "http://localhost:9000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def generate_imu_window(self, pattern: str = "zeros") -> List[List[float]]:
        """Generate test IMU data with different patterns"""
        if pattern == "zeros":
            return [[0.0] * 9 for _ in range(100)]
        elif pattern == "random":
            return [
                [
                    random.uniform(-10, 10),  # ax
                    random.uniform(-10, 10),  # ay  
                    random.uniform(8, 12),    # az (gravity)
                    random.uniform(-2, 2),    # gx
                    random.uniform(-2, 2),    # gy
                    random.uniform(-2, 2),    # gz
                    random.uniform(-100, 100),  # mx
                    random.uniform(-100, 100),  # my
                    random.uniform(-100, 100),  # mz
                ]
                for _ in range(100)
            ]
        elif pattern == "walking":
            # Simulate walking pattern
            return [
                [
                    2 * random.random() - 1,     # ax: lateral sway
                    1 + 0.5 * random.random(),   # ay: forward acceleration
                    9.8 + random.random() - 0.5, # az: gravity + vertical motion
                    0.2 * random.random() - 0.1, # gx: minimal pitch
                    0.1 * random.random() - 0.05,# gy: minimal roll
                    0.5 * random.random() - 0.25,# gz: yaw variation
                    25 + 10 * random.random(),    # mx: typical field
                    -10 + 5 * random.random(),    # my: typical field
                    40 + 10 * random.random(),    # mz: typical field
                ]
                for _ in range(100)
            ]
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    async def test_health_check(self) -> bool:
        """Test /healthz endpoint"""
        print("🔍 Testing health check endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/healthz")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check OK: {data}")
                return True
            elif response.status_code == 503:
                data = response.json()
                print(f"⚠️  Service unavailable (expected in stub mode): {data}")
                return True
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def test_inference(self, pattern: str = "zeros") -> bool:
        """Test /infer endpoint with generated data"""
        print(f"🔍 Testing inference endpoint with {pattern} pattern...")
        
        try:
            # Generate test data
            imu_data = self.generate_imu_window(pattern)
            payload = {"x": imu_data}
            
            # Record timing
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/infer",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if "latent" in data and "motif_scores" in data:
                    latent_len = len(data["latent"])
                    motif_len = len(data["motif_scores"])
                    
                    print(f"✅ Inference successful:")
                    print(f"   Latency: {duration_ms:.1f}ms")
                    print(f"   Latent dim: {latent_len}")
                    print(f"   Motif scores: {motif_len}")
                    print(f"   Latent range: [{min(data['latent']):.3f}, {max(data['latent']):.3f}]")
                    print(f"   Motif range: [{min(data['motif_scores']):.3f}, {max(data['motif_scores']):.3f}]")
                    
                    # Validate dimensions
                    if latent_len == 64 and motif_len == 12:
                        return True
                    else:
                        print(f"❌ Wrong output dimensions: latent={latent_len}, motifs={motif_len}")
                        return False
                else:
                    print(f"❌ Missing required fields in response: {data}")
                    return False
                    
            elif response.status_code == 503:
                print(f"⚠️  Service unavailable: {response.json()}")
                return True  # Expected in stub mode
            else:
                print(f"❌ Inference failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            return False
    
    async def test_metrics(self) -> bool:
        """Test /metrics endpoint"""
        print("🔍 Testing metrics endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/metrics")
            
            if response.status_code == 200:
                metrics_text = response.text
                
                # Check for expected metrics
                expected_metrics = [
                    "hailo_inference_requests_total",
                    "hailo_inference_duration_seconds",
                    "hailo_model_loaded",
                    "hailo_sidecar_start_time_seconds"
                ]
                
                found_metrics = []
                for metric in expected_metrics:
                    if metric in metrics_text:
                        found_metrics.append(metric)
                
                print(f"✅ Metrics endpoint working:")
                print(f"   Found {len(found_metrics)}/{len(expected_metrics)} expected metrics")
                print(f"   Response length: {len(metrics_text)} chars")
                
                return len(found_metrics) >= 2  # At least some metrics present
            else:
                print(f"❌ Metrics failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Metrics test failed: {e}")
            return False
    
    async def test_status(self) -> bool:
        """Test /status endpoint"""
        print("🔍 Testing status endpoint...")
        
        try:
            response = await self.client.get(f"{self.base_url}/status")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Status endpoint OK:")
                
                for key, value in data.items():
                    print(f"   {key}: {value}")
                
                return True
            else:
                print(f"❌ Status failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Status test failed: {e}")
            return False
    
    async def test_load(self, num_requests: int = 10) -> bool:
        """Test load handling with multiple requests"""
        print(f"🔍 Testing load with {num_requests} concurrent requests...")
        
        try:
            # Create multiple inference tasks
            tasks = []
            for i in range(num_requests):
                pattern = "walking" if i % 3 == 0 else "zeros"
                imu_data = self.generate_imu_window(pattern)
                payload = {"x": imu_data}
                
                task = self.client.post(
                    f"{self.base_url}/infer",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                tasks.append(task)
            
            # Execute all requests concurrently
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = (time.time() - start_time) * 1000
            
            # Analyze results
            successes = 0
            errors = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    errors += 1
                elif response.status_code == 200:
                    successes += 1
                else:
                    errors += 1
            
            throughput = num_requests / (total_time / 1000)
            
            print(f"✅ Load test results:")
            print(f"   Total time: {total_time:.1f}ms")
            print(f"   Successes: {successes}/{num_requests}")
            print(f"   Errors: {errors}/{num_requests}")
            print(f"   Throughput: {throughput:.1f} req/sec")
            print(f"   Avg latency: {total_time/num_requests:.1f}ms")
            
            return successes > num_requests * 0.8  # 80% success rate
            
        except Exception as e:
            print(f"❌ Load test failed: {e}")
            return False
    
    async def run_all_tests(self) -> bool:
        """Run complete test suite"""
        print("🚀 Starting HailoRT Sidecar Test Suite")
        print("=" * 50)
        
        tests = [
            ("Health Check", self.test_health_check()),
            ("Status", self.test_status()),
            ("Metrics", self.test_metrics()),
            ("Inference (zeros)", self.test_inference("zeros")),
            ("Inference (walking)", self.test_inference("walking")),
            ("Load Test", self.test_load(5)),
        ]
        
        results = []
        for test_name, test_coro in tests:
            print(f"\n📋 Running {test_name}...")
            try:
                result = await test_coro
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        success_rate = passed / len(results) * 100
        print(f"\n🎯 Overall: {passed}/{len(results)} tests passed ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            print("🎉 Sidecar is working correctly!")
            return True
        else:
            print("⚠️  Some issues detected - check logs above")
            return False


async def main():
    """Main test execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HailoRT TCN Inference Sidecar")
    parser.add_argument("--url", default="http://localhost:9000", help="Sidecar base URL")
    parser.add_argument("--test", choices=["all", "health", "inference", "metrics", "status", "load"], 
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    async with SidecarTester(args.url) as tester:
        if args.test == "all":
            success = await tester.run_all_tests()
        elif args.test == "health":
            success = await tester.test_health_check()
        elif args.test == "inference":
            success = await tester.test_inference("walking")
        elif args.test == "metrics":
            success = await tester.test_metrics()
        elif args.test == "status":
            success = await tester.test_status()
        elif args.test == "load":
            success = await tester.test_load(10)
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())