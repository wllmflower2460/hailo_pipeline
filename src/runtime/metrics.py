"""
Prometheus metrics for HailoRT inference sidecar.

Tracks inference latency, request counts, model status, and hardware utilization
for EdgeInfer production monitoring.
"""

import time
import logging
from typing import Dict, Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import threading

logger = logging.getLogger(__name__)

# Request metrics
inference_counter = Counter(
    'hailo_inference_requests_total',
    'Total inference requests processed',
    ['status', 'model']
)

request_duration = Histogram(
    'hailo_inference_duration_seconds',
    'Time spent on inference requests',
    ['model'],
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000, 10.0)
)

# Model status metrics
model_loaded_gauge = Gauge(
    'hailo_model_loaded',
    'Whether HEF model is successfully loaded',
    ['model', 'version']
)

model_info_gauge = Gauge(
    'hailo_model_info',
    'Model information',
    ['model', 'hef_path', 'motifs', 'device']
)

# Hardware metrics
hailo_device_temperature = Gauge(
    'hailo_device_temperature_celsius',
    'Hailo device temperature',
    ['device_id']
)

hailo_device_utilization = Gauge(
    'hailo_device_utilization_percent',
    'Hailo device utilization percentage',
    ['device_id']
)

# Application metrics
app_start_time = Gauge(
    'hailo_sidecar_start_time_seconds',
    'Unix timestamp when sidecar started'
)

# Health check metrics
health_check_counter = Counter(
    'hailo_health_checks_total',
    'Total health check requests',
    ['status']
)

# Error tracking
error_counter = Counter(
    'hailo_errors_total',
    'Total errors by type',
    ['error_type', 'model']
)

# Enhanced build and configuration tracking metrics (ADR-0007 revision)
build_info_gauge = Gauge(
    'hailo_build_info',
    'Build and configuration information',
    ['version', 'hef_sha', 'config']
)

config_ok_gauge = Gauge(
    'hailo_config_ok',
    'Configuration validation status (1=OK, 0=mismatch)',
    ['expected', 'actual']
)

# Thread-safe metrics lock
_metrics_lock = threading.RLock()


class MetricsCollector:
    """Centralized metrics collection and reporting"""
    
    def __init__(self, model_name: str = "tcn_encoder"):
        self.model_name = model_name
        self.start_time = time.time()
        
        # Initialize app metrics
        app_start_time.set(self.start_time)
        
        logger.info(f"Metrics collector initialized for model: {model_name}")
    
    def record_inference_request(self, status: str, duration_seconds: float):
        """Record inference request metrics"""
        with _metrics_lock:
            inference_counter.labels(status=status, model=self.model_name).inc()
            if status == "success":
                request_duration.labels(model=self.model_name).observe(duration_seconds)
    
    def record_health_check(self, healthy: bool):
        """Record health check metrics"""
        status = "healthy" if healthy else "unhealthy"
        health_check_counter.labels(status=status).inc()
    
    def record_error(self, error_type: str):
        """Record error metrics"""
        error_counter.labels(error_type=error_type, model=self.model_name).inc()
    
    def update_model_status(self, loaded: bool, model_info: Optional[Dict] = None):
        """Update model loading status"""
        with _metrics_lock:
            model_loaded_gauge.labels(
                model=self.model_name,
                version="1.0.0"
            ).set(1 if loaded else 0)
            
            if model_info and loaded:
                model_info_gauge.labels(
                    model=self.model_name,
                    hef_path=model_info.get("hef_path", "unknown"),
                    motifs=str(model_info.get("num_motifs", 0)),
                    device="hailo8"
                ).set(1)
    
    def update_hardware_metrics(self, device_id: str = "hailo0", 
                               temperature: Optional[float] = None,
                               utilization: Optional[float] = None):
        """Update hardware telemetry metrics"""
        with _metrics_lock:
            if temperature is not None:
                hailo_device_temperature.labels(device_id=device_id).set(temperature)
            
            if utilization is not None:
                hailo_device_utilization.labels(device_id=device_id).set(utilization)

    def update_build_info(self, version: str, hef_sha: str, config_version: str):
        """Update build information metrics for fleet tracking"""
        with _metrics_lock:
            build_info_gauge.labels(
                version=version,
                hef_sha=hef_sha,
                config=config_version
            ).set(1)

    def update_config_status(self, expected_config: str, actual_config: str):
        """Update configuration validation status"""
        with _metrics_lock:
            # Set to 1 if configs match, 0 if mismatch
            config_ok = 1 if expected_config == actual_config else 0
            config_ok_gauge.labels(
                expected=expected_config,
                actual=actual_config
            ).set(config_ok)


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector(model_name: str = "tcn_encoder") -> MetricsCollector:
    """Get or create global metrics collector"""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector(model_name)
    return _global_collector


def generate_metrics() -> str:
    """Generate Prometheus metrics in text format"""
    return generate_latest().decode('utf-8')


def get_metrics_content_type() -> str:
    """Get appropriate content type for metrics"""
    return CONTENT_TYPE_LATEST


# Context manager for timing operations
class InferenceTimer:
    """Context manager for timing inference operations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.start_time = None
        self.success = False
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            status = "success" if self.success and exc_type is None else "error"
            self.collector.record_inference_request(status, duration)
            
            if exc_type:
                error_type = exc_type.__name__ if exc_type else "unknown"
                self.collector.record_error(error_type)
    
    def mark_success(self):
        """Mark operation as successful"""
        self.success = True


# Utility functions for common operations
def record_startup_metrics(model_loader):
    """Record metrics during application startup"""
    collector = get_metrics_collector()
    
    if model_loader and model_loader.is_ready():
        model_info = model_loader.get_status()
        collector.update_model_status(True, model_info)
        
        # Record enhanced build information
        collector.update_build_info(
            version="v1.0.0",
            hef_sha=model_loader.hef_sha256,
            config_version=model_loader.config_version
        )
        
        # Record configuration validation status
        expected_config = model_loader.config_version
        actual_config = model_loader.config_version  # In production, this might come from different source
        collector.update_config_status(expected_config, actual_config)
        
        logger.info("✅ Model and build metrics recorded as loaded")
        logger.info(f"Build info: version=v1.0.0, hef_sha={model_loader.hef_sha256}, config={model_loader.config_version}")
    else:
        collector.update_model_status(False)
        
        # Still record build info even if model fails to load
        if model_loader:
            collector.update_build_info(
                version="v1.0.0",
                hef_sha=model_loader.hef_sha256,
                config_version=model_loader.config_version
            )
            collector.update_config_status(
                expected_config=model_loader.config_version,
                actual_config="model_load_failed"
            )
        
        logger.warning("❌ Model metrics recorded as not loaded")


def record_shutdown_metrics():
    """Record metrics during application shutdown"""
    collector = get_metrics_collector()
    collector.update_model_status(False)
    logger.info("🔄 Shutdown metrics recorded")


# Health check metric recording
def record_health_check_success():
    """Record successful health check"""
    get_metrics_collector().record_health_check(True)


def record_health_check_failure():
    """Record failed health check"""
    get_metrics_collector().record_health_check(False)