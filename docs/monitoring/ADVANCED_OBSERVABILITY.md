# Advanced Observability & Monitoring

This document outlines comprehensive observability strategies for the causal-interface-gym project, building on existing monitoring foundations.

## Observability Stack Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Application   │───▶│  Telemetry   │───▶│   Collectors    │
│    Metrics      │    │   Gateway    │    │ (Prometheus)    │
└─────────────────┘    └──────────────┘    └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│     Traces      │    │    Logs      │    │   Dashboards    │
│   (Jaeger)      │    │(Elasticsearch)│    │   (Grafana)     │
└─────────────────┘    └──────────────┘    └─────────────────┘
```

## Application Performance Monitoring (APM)

### OpenTelemetry Integration

Create `src/causal_interface_gym/telemetry/__init__.py`:

```python
"""OpenTelemetry instrumentation for causal-interface-gym."""

import logging
import os
from typing import Optional

from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Service identification
SERVICE_NAME_VALUE = "causal-interface-gym"
SERVICE_VERSION_VALUE = "0.1.0"

class TelemetryConfig:
    """Configuration for telemetry setup."""
    
    def __init__(self):
        self.service_name = os.getenv("OTEL_SERVICE_NAME", SERVICE_NAME_VALUE)
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", SERVICE_VERSION_VALUE)
        self.jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
        self.prometheus_endpoint = os.getenv("PROMETHEUS_ENDPOINT", "localhost:8000")
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"

def setup_telemetry(config: Optional[TelemetryConfig] = None) -> None:
    """Initialize OpenTelemetry instrumentation."""
    if config is None:
        config = TelemetryConfig()
    
    # Resource identification
    resource = Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        "deployment.environment": os.getenv("ENVIRONMENT", "development"),
        "service.instance.id": os.getenv("HOSTNAME", "unknown"),
    })
    
    # Setup tracing
    if config.enable_tracing:
        setup_tracing(resource, config)
    
    # Setup metrics
    if config.enable_metrics:
        setup_metrics(resource, config)
    
    # Setup logging instrumentation
    LoggingInstrumentor().instrument(set_logging_format=True)
    RequestsInstrumentor().instrument()

def setup_tracing(resource: Resource, config: TelemetryConfig) -> None:
    """Configure distributed tracing."""
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(endpoint=config.jaeger_endpoint)
    span_processor = BatchSpanProcessor(jaeger_exporter)
    tracer_provider.add_span_processor(span_processor)

def setup_metrics(resource: Resource, config: TelemetryConfig) -> None:
    """Configure metrics collection."""
    prometheus_reader = PrometheusMetricReader()
    meter_provider = MeterProvider(resource=resource, metric_readers=[prometheus_reader])
    metrics.set_meter_provider(meter_provider)

# Global tracer and meter instances
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Custom metrics
causal_operations_counter = meter.create_counter(
    name="causal_operations_total",
    description="Total number of causal operations performed",
    unit="1"
)

intervention_duration_histogram = meter.create_histogram(
    name="intervention_duration_seconds",
    description="Duration of causal interventions",
    unit="s"
)

graph_size_gauge = meter.create_up_down_counter(
    name="causal_graph_nodes",
    description="Number of nodes in current causal graph",
    unit="1"
)

memory_usage_gauge = meter.create_up_down_counter(
    name="memory_usage_bytes",
    description="Current memory usage",
    unit="By"
)
```

### Instrumentation Decorators

Create `src/causal_interface_gym/telemetry/decorators.py`:

```python
"""Telemetry decorators for automatic instrumentation."""

import functools
import time
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from . import (
    tracer,
    causal_operations_counter,
    intervention_duration_histogram,
    memory_usage_gauge
)

def instrument_causal_operation(
    operation_type: str,
    record_duration: bool = True,
    record_memory: bool = False
):
    """Decorator to instrument causal operations with tracing and metrics."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Start tracing span
            with tracer.start_as_current_span(
                f"causal.{operation_type}.{func.__name__}"
            ) as span:
                # Add span attributes
                span.set_attribute("operation.type", operation_type)
                span.set_attribute("function.name", func.__name__)
                
                # Record memory usage before operation
                if record_memory:
                    import psutil
                    import os
                    initial_memory = psutil.Process(os.getpid()).memory_info().rss
                
                # Record operation start time
                start_time = time.time()
                
                try:
                    # Execute the function
                    result = func(*args, **kwargs)
                    
                    # Mark span as successful
                    span.set_status(Status(StatusCode.OK))
                    
                    # Add result metadata to span
                    if hasattr(result, '__len__'):
                        span.set_attribute("result.size", len(result))
                    
                    return result
                    
                except Exception as e:
                    # Mark span as error
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
                    
                finally:
                    # Record metrics
                    causal_operations_counter.add(
                        1, 
                        {"operation_type": operation_type, "function": func.__name__}
                    )
                    
                    if record_duration:
                        duration = time.time() - start_time
                        intervention_duration_histogram.record(
                            duration,
                            {"operation_type": operation_type, "function": func.__name__}
                        )
                    
                    if record_memory:
                        final_memory = psutil.Process(os.getpid()).memory_info().rss
                        memory_usage_gauge.add(
                            final_memory - initial_memory,
                            {"operation_type": operation_type}
                        )
        
        return wrapper
    return decorator

def trace_causal_graph_operation(func: Callable) -> Callable:
    """Decorator specifically for causal graph operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with tracer.start_as_current_span(f"graph.{func.__name__}") as span:
            # Extract graph information if available
            if args and hasattr(args[0], 'nodes'):
                graph_size = len(args[0].nodes)
                span.set_attribute("graph.node_count", graph_size)
                graph_size_gauge.add(1, {"graph_size_category": categorize_graph_size(graph_size)})
                
            return func(*args, **kwargs)
    return wrapper

def categorize_graph_size(node_count: int) -> str:
    """Categorize graph size for metrics."""
    if node_count < 10:
        return "small"
    elif node_count < 100:
        return "medium" 
    elif node_count < 1000:
        return "large"
    else:
        return "very_large"
```

## Custom Metrics and Dashboards

### Business Logic Metrics

Create `src/causal_interface_gym/metrics/business_metrics.py`:

```python
"""Business-specific metrics for causal reasoning operations."""

from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from ..telemetry import meter

# Business metrics
causal_accuracy_gauge = meter.create_up_down_counter(
    name="causal_accuracy_score",
    description="Accuracy score of causal inference",
    unit="1"
)

intervention_success_rate = meter.create_histogram(
    name="intervention_success_rate",
    description="Success rate of interventions",
    unit="1"
)

model_prediction_accuracy = meter.create_histogram(
    name="model_prediction_accuracy",
    description="Accuracy of causal model predictions",
    unit="1"
)

@dataclass
class CausalMetrics:
    """Container for causal reasoning metrics."""
    accuracy_score: float
    intervention_count: int
    prediction_accuracy: float
    computation_time: float
    memory_usage: int
    timestamp: datetime

class BusinessMetricsCollector:
    """Collects and exports business-specific metrics."""
    
    def __init__(self):
        self.metrics_history: List[CausalMetrics] = []
    
    def record_causal_inference_result(
        self,
        accuracy: float,
        intervention_count: int,
        prediction_accuracy: float,
        computation_time: float,
        memory_usage: int
    ) -> None:
        """Record results of causal inference operation."""
        # Store in history
        metrics = CausalMetrics(
            accuracy_score=accuracy,
            intervention_count=intervention_count,
            prediction_accuracy=prediction_accuracy,
            computation_time=computation_time,
            memory_usage=memory_usage,
            timestamp=datetime.utcnow()
        )
        self.metrics_history.append(metrics)
        
        # Export to telemetry
        causal_accuracy_gauge.add(accuracy, {"model_type": "inference"})
        intervention_success_rate.record(
            intervention_count / max(1, intervention_count), 
            {"operation": "inference"}
        )
        model_prediction_accuracy.record(prediction_accuracy, {"model_type": "causal"})
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp.timestamp() > cutoff
        ]
        
        if not recent_metrics:
            return {"status": "no_data", "hours": hours}
        
        return {
            "avg_accuracy": sum(m.accuracy_score for m in recent_metrics) / len(recent_metrics),
            "total_interventions": sum(m.intervention_count for m in recent_metrics),
            "avg_prediction_accuracy": sum(m.prediction_accuracy for m in recent_metrics) / len(recent_metrics),
            "avg_computation_time": sum(m.computation_time for m in recent_metrics) / len(recent_metrics),
            "peak_memory_usage": max(m.memory_usage for m in recent_metrics),
            "operation_count": len(recent_metrics),
            "time_period_hours": hours
        }

# Global collector instance
business_metrics = BusinessMetricsCollector()
```

### Grafana Dashboard Configuration

Create `monitoring/grafana/dashboards/causal-interface-gym.json`:

```json
{
  "dashboard": {
    "id": null,
    "title": "Causal Interface Gym - Performance Dashboard",
    "tags": ["causal-reasoning", "performance"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Causal Operations Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(causal_operations_total[5m])",
            "legendFormat": "Operations/sec"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {"displayMode": "list", "orientation": "auto"},
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 10},
                {"color": "red", "value": 50}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Intervention Duration",
        "type": "histogram",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(intervention_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(intervention_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Memory Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "memory_usage_bytes",
            "legendFormat": "Memory Usage (bytes)"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "palette-classic"},
            "unit": "bytes"
          }
        }
      },
      {
        "id": 4,
        "title": "Causal Graph Size Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (graph_size_category) (causal_graph_nodes)",
            "legendFormat": "{{graph_size_category}}"
          }
        ]
      },
      {
        "id": 5,
        "title": "Error Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(causal_operations_total{status=\"error\"}[5m]) / rate(causal_operations_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {"mode": "thresholds"},
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5}
              ]
            },
            "unit": "percent"
          }
        }
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "10s"
  }
}
```

## Alerting and Incident Response

### Prometheus Alerting Rules

Update `monitoring/alert.rules.yml`:

```yaml
groups:
  - name: causal-interface-gym.alerts
    rules:
      - alert: HighErrorRate
        expr: rate(causal_operations_total{status="error"}[5m]) / rate(causal_operations_total[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
          service: causal-interface-gym
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes"
          
      - alert: SlowInterventions
        expr: histogram_quantile(0.95, rate(intervention_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
          service: causal-interface-gym
        annotations:
          summary: "Slow causal interventions detected"
          description: "95th percentile intervention duration is {{ $value }}s"
          
      - alert: HighMemoryUsage
        expr: memory_usage_bytes > 1e9  # 1GB
        for: 3m
        labels:
          severity: critical
          service: causal-interface-gym
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value | humanizeBytes }}"
          
      - alert: CausalAccuracyDrop
        expr: causal_accuracy_score < 0.7
        for: 5m
        labels:
          severity: warning
          service: causal-interface-gym
        annotations:
          summary: "Causal inference accuracy dropped"
          description: "Accuracy score is {{ $value }}, below threshold of 0.7"
```

### Health Check Endpoints

Create `src/causal_interface_gym/health/__init__.py`:

```python
"""Health check endpoints and monitoring."""

import asyncio
import time
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass

class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float

class HealthMonitor:
    """Comprehensive health monitoring."""
    
    def __init__(self):
        self.checks: List[callable] = []
        self.last_results: Dict[str, HealthCheck] = {}
    
    def register_check(self, name: str, check_func: callable) -> None:
        """Register a health check function."""
        self.checks.append((name, check_func))
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self.checks:
            start_time = time.time()
            
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(check_func(), timeout=5.0)
                status = HealthStatus.HEALTHY if result else HealthStatus.DEGRADED
                message = "OK" if result else "Check failed"
                
            except asyncio.TimeoutError:
                status = HealthStatus.UNHEALTHY
                message = "Health check timed out"
                
            except Exception as e:
                status = HealthStatus.UNHEALTHY
                message = f"Health check error: {str(e)}"
            
            duration_ms = (time.time() - start_time) * 1000
            
            health_check = HealthCheck(
                name=name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=time.time()
            )
            
            results[name] = health_check
            self.last_results[name] = health_check
            
            # Update overall status
            if status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "checks": {name: {
                "status": check.status.value,
                "message": check.message,
                "duration_ms": check.duration_ms,
                "timestamp": check.timestamp
            } for name, check in results.items()},
            "timestamp": time.time()
        }

# Default health checks
async def check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < 90  # Less than 90% memory usage
    except ImportError:
        return True  # Skip check if psutil not available

async def check_causal_engine() -> bool:
    """Check if causal reasoning engine is responsive."""
    try:
        from ..core import CausalEnvironment
        
        # Quick test of core functionality
        env = CausalEnvironment.from_dag({"A": [], "B": ["A"]})
        result = env.intervene(A=True)
        return result is not None
    except Exception:
        return False

async def check_dependencies() -> bool:
    """Check if critical dependencies are available."""
    try:
        import numpy
        import pandas
        import networkx
        return True
    except ImportError:
        return False

# Global health monitor
health_monitor = HealthMonitor()
health_monitor.register_check("memory", check_memory_usage)
health_monitor.register_check("causal_engine", check_causal_engine)
health_monitor.register_check("dependencies", check_dependencies)
```

## Distributed Tracing

### Trace Correlation

Create `src/causal_interface_gym/telemetry/correlation.py`:

```python
"""Trace correlation utilities for distributed causal reasoning."""

import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Context variables for trace correlation
REQUEST_ID: ContextVar[str] = ContextVar('request_id')
USER_ID: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
SESSION_ID: ContextVar[Optional[str]] = ContextVar('session_id', default=None)

class TraceCorrelator:
    """Utilities for correlating traces across causal operations."""
    
    @staticmethod
    def generate_request_id() -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def set_request_context(
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Set request context for trace correlation."""
        if request_id is None:
            request_id = TraceCorrelator.generate_request_id()
        
        REQUEST_ID.set(request_id)
        if user_id:
            USER_ID.set(user_id)
        if session_id:
            SESSION_ID.set(session_id)
        
        # Add to current span if available
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute("request.id", request_id)
            if user_id:
                current_span.set_attribute("user.id", user_id)
            if session_id:
                current_span.set_attribute("session.id", session_id)
        
        return request_id
    
    @staticmethod
    def get_correlation_context() -> Dict[str, Any]:
        """Get current correlation context."""
        return {
            "request_id": REQUEST_ID.get(None),
            "user_id": USER_ID.get(None),
            "session_id": SESSION_ID.get(None)
        }
    
    @staticmethod
    def create_child_span(
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> trace.Span:
        """Create child span with correlation context."""
        tracer = trace.get_tracer(__name__)
        span = tracer.start_span(operation_name)
        
        # Add correlation attributes
        context = TraceCorrelator.get_correlation_context()
        for key, value in context.items():
            if value:
                span.set_attribute(f"correlation.{key}", value)
        
        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span
```

This advanced observability setup provides:

1. **Comprehensive Telemetry**: OpenTelemetry integration with traces, metrics, and logs
2. **Business Metrics**: Causal reasoning-specific metrics and dashboards
3. **Proactive Alerting**: Prometheus alerts for performance and accuracy issues
4. **Health Monitoring**: Comprehensive health checks and status endpoints
5. **Distributed Tracing**: Request correlation across causal operations

The configuration builds upon the existing monitoring foundation while adding enterprise-grade observability appropriate for a maturing repository.