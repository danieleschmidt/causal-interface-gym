# Performance Monitoring Guide

This guide outlines the comprehensive performance monitoring setup for the Causal Interface Gym project, ensuring optimal performance and early detection of regressions.

## 🎯 Monitoring Overview

### Performance Metrics Categories

1. **Computational Performance**
   - Causal inference algorithm execution time
   - Memory usage during graph operations
   - Intervention computation efficiency
   - UI rendering performance

2. **Research Accuracy Metrics**
   - Causal reasoning correctness validation
   - Benchmark consistency over time
   - Result reproducibility tracking
   - Statistical significance monitoring

3. **System Resource Metrics**
   - CPU usage patterns
   - Memory allocation efficiency
   - I/O operations performance
   - Network latency (for distributed setups)

## 📊 Performance Benchmarking

### Automated Benchmark Suite

Our benchmarking framework tracks performance across multiple dimensions:

```python
# Example benchmark configuration
benchmark_configs = {
    "causal_inference": {
        "test_cases": [
            {"nodes": 10, "edges": 15, "samples": 1000},
            {"nodes": 50, "edges": 100, "samples": 5000},
            {"nodes": 100, "edges": 200, "samples": 10000},
        ],
        "metrics": ["execution_time", "memory_peak", "accuracy"],
        "baseline_thresholds": {
            "execution_time": {"regression_limit": 1.1},  # 10% slowdown limit
            "memory_peak": {"regression_limit": 1.2},     # 20% memory increase limit
            "accuracy": {"minimum_threshold": 0.95}       # 95% accuracy minimum
        }
    },
    "ui_rendering": {
        "test_cases": [
            {"graph_size": "small", "interactions": 10},
            {"graph_size": "medium", "interactions": 50},
            {"graph_size": "large", "interactions": 100},
        ],
        "metrics": ["render_time", "interaction_latency"],
        "baseline_thresholds": {
            "render_time": {"regression_limit": 1.15},
            "interaction_latency": {"regression_limit": 1.1}
        }
    }
}
```

### Performance Regression Detection

```bash
# Automated performance regression detection
python scripts/performance_monitor.py \
    --baseline-branch main \
    --current-branch feature/new-algorithm \
    --regression-threshold 0.1 \
    --output-format json
```

## 🔍 Continuous Monitoring

### GitHub Actions Integration

Performance monitoring is integrated into our CI/CD pipeline:

```yaml
# .github/workflows/performance.yml
name: Performance Monitoring

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  performance-benchmarks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for comparison
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev,ui]"
          pip install pytest-benchmark memory_profiler
      
      - name: Run benchmarks
        run: |
          pytest tests/benchmarks/ \
            --benchmark-json=benchmark-results.json \
            --benchmark-compare-fail=mean:10%
      
      - name: Performance regression check
        run: |
          python scripts/performance_monitor.py \
            --benchmark-file benchmark-results.json \
            --check-regression
      
      - name: Upload benchmark results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark-results.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
```

### Real-time Performance Dashboard

Performance metrics are collected and visualized using Grafana and Prometheus:

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert.rules.yml:/etc/prometheus/alert.rules.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

volumes:
  grafana-storage:
```

## 📈 Performance Metrics Collection

### Code Instrumentation

Performance-critical functions are instrumented for monitoring:

```python
# src/causal_interface_gym/monitoring.py
import time
import psutil
import functools
from typing import Dict, Any, Callable
from prometheus_client import Counter, Histogram, Gauge

# Metrics collectors
COMPUTATION_TIME = Histogram(
    'causal_computation_seconds',
    'Time spent on causal computations',
    ['operation_type', 'graph_size']
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Current memory usage',
    ['component']
)

ACCURACY_SCORE = Gauge(
    'causal_accuracy_score',
    'Current causal reasoning accuracy score',
    ['test_type']
)

def monitor_performance(operation_type: str = "default"):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                
                # Record successful execution
                execution_time = time.time() - start_time
                COMPUTATION_TIME.labels(
                    operation_type=operation_type,
                    graph_size=getattr(args[0], 'graph_size', 'unknown')
                ).observe(execution_time)
                
                # Record memory usage
                end_memory = psutil.Process().memory_info().rss
                MEMORY_USAGE.labels(component=func.__name__).set(end_memory)
                
                return result
                
            except Exception as e:
                # Record failed execution
                ERROR_COUNTER.labels(
                    operation_type=operation_type,
                    error_type=type(e).__name__
                ).inc()
                raise
                
        return wrapper
    return decorator
```

### Benchmark Test Examples

```python
# tests/benchmarks/test_performance.py
import pytest
import numpy as np
from causal_interface_gym import CausalEnvironment, CausalMetrics

class TestCausalPerformance:
    """Performance benchmarks for causal reasoning operations."""
    
    @pytest.mark.benchmark(group="causal_inference")
    def test_dag_creation_performance(self, benchmark):
        """Benchmark DAG creation with various sizes."""
        def create_large_dag():
            # Create a complex DAG with 100 nodes
            dag = {}
            for i in range(100):
                parents = [f"node_{j}" for j in range(max(0, i-3), i)]
                dag[f"node_{i}"] = parents
            return CausalEnvironment.from_dag(dag)
        
        result = benchmark(create_large_dag)
        assert result.graph.number_of_nodes() == 100
    
    @pytest.mark.benchmark(group="intervention")
    def test_intervention_performance(self, benchmark):
        """Benchmark intervention computation."""
        # Create test environment
        dag = {f"var_{i}": [f"var_{j}" for j in range(max(0, i-2), i)] 
              for i in range(20)}
        env = CausalEnvironment.from_dag(dag)
        
        def run_intervention():
            return env.intervene(var_10=1, var_15=0)
        
        result = benchmark(run_intervention)
        assert "intervention_applied" in result
    
    @pytest.mark.benchmark(group="metrics")
    def test_metric_computation_performance(self, benchmark):
        """Benchmark metric computation performance."""
        metrics = CausalMetrics()
        
        # Generate large test data
        agent_responses = [
            {"belief": np.random.random(), "type": "interventional"}
            for _ in range(1000)
        ]
        ground_truth = [
            {"belief": np.random.random(), "type": "interventional"}
            for _ in range(1000)
        ]
        
        def compute_metrics():
            return metrics.intervention_test(agent_responses, ground_truth)
        
        score = benchmark(compute_metrics)
        assert 0 <= score <= 1

    @pytest.mark.benchmark(group="memory")
    def test_memory_efficiency(self, benchmark):
        """Test memory usage efficiency."""
        import tracemalloc
        
        def memory_intensive_operation():
            tracemalloc.start()
            
            # Create multiple environments
            environments = []
            for i in range(10):
                dag = {f"node_{j}": [] for j in range(50)}
                env = CausalEnvironment.from_dag(dag)
                environments.append(env)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return {"current": current, "peak": peak, "envs": len(environments)}
        
        result = benchmark(memory_intensive_operation)
        
        # Assert reasonable memory usage (adjust thresholds as needed)
        assert result["peak"] < 100 * 1024 * 1024  # 100MB threshold
        assert result["envs"] == 10
```

## 🚨 Performance Alerting

### Alert Rules Configuration

```yaml
# monitoring/alert.rules.yml
groups:
  - name: causal_interface_gym_performance
    rules:
      - alert: HighComputationTime
        expr: causal_computation_seconds > 5.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High computation time detected"
          description: "Causal computation taking longer than 5 seconds"

      - alert: MemoryLeakDetected  
        expr: increase(memory_usage_bytes[5m]) > 100000000  # 100MB increase
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Potential memory leak detected"
          description: "Memory usage increased by more than 100MB in 5 minutes"

      - alert: AccuracyDegradation
        expr: causal_accuracy_score < 0.90
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Causal accuracy below threshold"
          description: "Causal reasoning accuracy dropped below 90%"

      - alert: PerformanceRegression
        expr: rate(causal_computation_seconds[5m]) > 1.2 * rate(causal_computation_seconds[1h] offset 1d)
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Performance regression detected"
          description: "Current performance is 20% slower than yesterday"
```

### Notification Channels

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@causal-gym.org'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#causal-gym-alerts'
        title: 'Causal Interface Gym Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        
  - name: 'email'
    email_configs:
      - to: 'team@causal-gym.org'
        subject: 'Causal Interface Gym Performance Alert'
        body: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## 📋 Performance Testing Checklist

### Pre-Release Performance Validation

- [ ] **Benchmark Comparison**: New version performance vs. baseline
- [ ] **Memory Leak Testing**: Extended runtime memory monitoring
- [ ] **Stress Testing**: Performance under high load conditions
- [ ] **Accuracy Validation**: Ensure performance optimizations don't hurt accuracy
- [ ] **Cross-Platform Testing**: Performance validation on different OS/hardware
- [ ] **Scalability Testing**: Performance with varying input sizes
- [ ] **Regression Testing**: Compare against multiple previous versions

### Continuous Monitoring Setup

- [ ] **Metrics Collection**: All key metrics instrumented
- [ ] **Dashboard Configuration**: Grafana dashboards configured
- [ ] **Alert Rules**: Performance alerts properly configured
- [ ] **Notification Setup**: Team notifications working
- [ ] **Data Retention**: Proper retention policies for metrics
- [ ] **Backup Strategy**: Performance data backup procedures

## 🔧 Performance Optimization Guidelines

### Code-Level Optimizations

1. **Algorithm Efficiency**
   - Use efficient graph algorithms from NetworkX
   - Implement caching for repeated computations
   - Vectorize operations using NumPy where possible

2. **Memory Management**
   - Use generators for large datasets
   - Implement object pooling for frequently created objects
   - Clean up resources properly in finally blocks

3. **I/O Optimization**
   - Batch file operations
   - Use appropriate data formats (Parquet for large datasets)
   - Implement lazy loading for large resources

### Infrastructure Optimizations

1. **Container Configuration**
   - Optimize Docker image layers
   - Use appropriate resource limits
   - Implement health checks

2. **Caching Strategy**
   - Redis for distributed caching
   - Local caching for frequently accessed data
   - CDN for static assets

3. **Database Optimization**
   - Proper indexing strategies
   - Query optimization
   - Connection pooling

This comprehensive performance monitoring setup ensures that the Causal Interface Gym maintains high performance while supporting rigorous research requirements.