# HailoRT TCN Inference Test Samples

This directory contains sample IMU data for testing the HailoRT TCN Inference Sidecar service.

## Sample Files

### 1. realistic_imu_sample.json
- **Format:** 100x9 IMU data with realistic sensor ranges
- **Channels:** [ax, ay, az, gx, gy, gz, mx, my, mz]
- **Ranges:**
  - Accelerometer (ax, ay, az): ±4 m/s² with gravity bias on Z-axis
  - Gyroscope (gx, gy, gz): ±250 deg/s
  - Magnetometer (mx, my, mz): ±50 μT
- **Use case:** Most realistic for production testing

### 2. static_imu_sample.json
- **Format:** 100x9 IMU data with predictable pattern
- **Pattern:** Repeating 10-sample cycle with small variations
- **Base values:** Typical stationary device with slight movement
- **Use case:** Deterministic testing and debugging

### 3. random_imu_sample.json
- **Format:** 100x9 IMU data with random values
- **Ranges:** ±2.0 across all channels
- **Use case:** Stress testing and validation

## API Testing

### Service Information
- **Base URL:** `http://localhost:9000`
- **Health Check:** `GET /healthz`
- **Inference:** `POST /infer`
- **Documentation:** `GET /docs`

### Quick Tests

**Health Check:**
```bash
curl http://localhost:9000/healthz
```

**Test with realistic data:**
```bash
curl -X POST http://localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d @data/test_samples/realistic_imu_sample.json
```

**Test with static pattern:**
```bash
curl -X POST http://localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d @data/test_samples/static_imu_sample.json
```

**Test with random data:**
```bash
curl -X POST http://localhost:9000/infer \
  -H "Content-Type: application/json" \
  -d @data/test_samples/random_imu_sample.json
```

## Expected Response Format

```json
{
  "latent": [64 float values],
  "motif_scores": [12 float values]
}
```

## Performance Benchmarks
- **Inference latency:** ~5-10ms
- **Throughput:** ~100 req/s
- **Model:** tcn_encoder_stub (development mode)

## Integration Notes

This service is EdgeInfer-compatible and ready for PiSrv integration. The `/healthz` endpoint provides EdgeInfer health monitoring, and the `/infer` endpoint accepts the standard 100x9 IMU window format.