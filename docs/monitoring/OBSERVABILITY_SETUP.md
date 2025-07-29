# Observability Setup Guide

This document outlines the monitoring, logging, and observability setup for the Causal Interface Gym project.

## Overview

The observability stack includes:
- **Metrics Collection**: Application and system metrics
- **Structured Logging**: Comprehensive log management
- **Health Checks**: System health monitoring
- **Performance Monitoring**: Response time and resource tracking
- **Error Tracking**: Exception and error monitoring

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Monitoring    │───▶│   Exporters     │
│                 │    │   Library       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Structured    │    │   Health        │    │   Prometheus    │
│   Logging       │    │   Checks        │    │   Grafana       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Metrics Collection

### Application Metrics

The monitoring system tracks:

- **Experiments**: Number of experiments run
- **Interventions**: Interventions applied
- **Environments**: Environments created
- **UI Components**: Components created
- **Errors**: Error count and rates
- **Performance**: Response times and latencies
- **Resources**: Memory and CPU usage

### Usage Example

```python
from causal_interface_gym.monitoring import metrics, track_experiment, monitor_performance

# Track an experiment
with track_experiment("smoking_intervention", participants=10):
    # Run experiment
    env = CausalEnvironment.from_dag(dag)
    result = env.intervene(smoking=False)

# Use performance monitoring
@monitor_performance
def run_complex_analysis():
    # Complex analysis code
    pass

# Manual metrics
metrics.increment('custom_metric')
metrics.record_time('operation_duration', 1.23)
metrics.set_gauge('active_users', 42)
```

### Metrics Export

Export metrics for monitoring systems:

```python
from causal_interface_gym.monitoring import export_metrics_endpoint

# Get metrics in different formats
endpoints = export_metrics_endpoint()
print(endpoints['prometheus'])  # Prometheus format
print(endpoints['json'])        # JSON format
print(endpoints['health'])      # Health check results
```

## Structured Logging

### Log Configuration

```python
from causal_interface_gym.monitoring import setup_logging

# Basic setup
setup_logging(log_level="INFO")

# Advanced setup with file output
setup_logging(
    log_level="DEBUG",
    log_file="/var/log/causal-gym/app.log"
)
```

### Log Format

Logs are structured with contextual information:

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "causal_interface_gym.core",
  "message": "Experiment completed",
  "experiment_id": "smoking_study_1642248645",
  "duration": 12.34,
  "success": true,
  "metadata": {
    "participants": 10,
    "interventions": 5
  }
}
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical errors requiring immediate attention

## Health Checks

### Built-in Health Checks

The system includes several built-in health checks:

1. **Basic Functionality**: Tests core system functions
2. **Memory Usage**: Monitors system memory
3. **Disk Space**: Monitors available disk space
4. **Database Connectivity** (if applicable)
5. **External Service Connectivity**

### Custom Health Checks

Add custom health checks:

```python
from causal_interface_gym.monitoring import health_checker

def check_llm_connectivity():
    """Check if LLM service is accessible."""
    try:
        # Test LLM connectivity
        response = llm_client.health_check()
        return {'status': 'connected', 'latency': response.latency}
    except Exception as e:
        raise Exception(f"LLM service unavailable: {e}")

# Register the check
health_checker.register_check('llm_connectivity', check_llm_connectivity)

# Run all checks
results = health_checker.run_checks()
```

### Health Check Endpoint

```python
from flask import Flask, jsonify
from causal_interface_gym.monitoring import health_checker

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify(health_checker.run_checks())

@app.route('/health/live')
def liveness():
    # Simple liveness check
    return jsonify({'status': 'alive', 'timestamp': time.time()})

@app.route('/health/ready')
def readiness():
    # Readiness check with dependencies
    checks = health_checker.run_checks()
    status_code = 200 if checks['status'] == 'healthy' else 503
    return jsonify(checks), status_code
```

## Performance Monitoring

### Response Time Tracking

Monitor function execution times:

```python
from causal_interface_gym.monitoring import monitor_performance

@monitor_performance
def complex_intervention_analysis(environment, interventions):
    # Analysis code
    return results
```

### Resource Monitoring

Track system resource usage:

```python
import psutil
from causal_interface_gym.monitoring import metrics

def update_system_metrics():
    """Update system resource metrics."""
    # CPU usage
    cpu_percent = psutil.cpu_percent()
    metrics.set_gauge('cpu_usage_percent', cpu_percent)
    
    # Memory usage
    memory = psutil.virtual_memory()
    metrics.set_gauge('memory_usage_percent', memory.percent)
    metrics.set_gauge('memory_used_mb', memory.used / (1024 * 1024))
    
    # Disk usage
    disk = psutil.disk_usage('/')
    metrics.set_gauge('disk_usage_percent', (disk.used / disk.total) * 100)
```

## Integration with Monitoring Systems

### Prometheus Integration

Export metrics for Prometheus:

```python
from flask import Flask, Response
from causal_interface_gym.monitoring import metrics

app = Flask(__name__)

@app.route('/metrics')
def prometheus_metrics():
    return Response(
        metrics.export_prometheus(),
        mimetype='text/plain'
    )
```

Prometheus configuration (`prometheus.yml`):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'causal-interface-gym'
    static_configs:
      - targets: ['localhost:5000']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Grafana Dashboard

Example Grafana dashboard configuration:

```json
{
  "dashboard": {
    "title": "Causal Interface Gym Monitoring",
    "panels": [
      {
        "title": "Experiments per Hour",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(causal_gym_experiments_run[1h])"
          }
        ]
      },
      {
        "title": "Average Response Time",
        "type": "singlestat",
        "targets": [
          {
            "expr": "avg(causal_gym_response_times)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(causal_gym_errors[5m])"
          }
        ]
      }
    ]
  }
}
```

### ELK Stack Integration

For centralized logging with Elasticsearch, Logstash, and Kibana:

**Logstash Configuration** (`logstash.conf`):

```ruby
input {
  file {
    path => "/var/log/causal-gym/app.log"
    start_position => "beginning"
    codec => "json"
  }
}

filter {
  if [logger] == "causal_interface_gym" {
    mutate {
      add_field => { "service" => "causal-interface-gym" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "causal-gym-logs-%{+YYYY.MM.dd}"
  }
}
```

**Filebeat Configuration** (`filebeat.yml`):

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/causal-gym/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "causal-gym-logs-%{+yyyy.MM.dd}"

setup.kibana:
  host: "localhost:5601"
```

## Alerting

### Prometheus Alerting Rules

Define alerting rules (`alert.rules.yml`):

```yaml
groups:
- name: causal-interface-gym
  rules:
  - alert: HighErrorRate
    expr: rate(causal_gym_errors[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighMemoryUsage
    expr: causal_gym_memory_usage_percent > 90
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}%"

  - alert: ServiceDown
    expr: up{job="causal-interface-gym"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "Causal Interface Gym service is not responding"
```

### Alertmanager Configuration

Configure alert routing (`alertmanager.yml`):

```yaml
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
  email_configs:
  - to: 'admin@causal-gym.org'
    subject: 'Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

## Docker Monitoring Setup

### Docker Compose for Monitoring Stack

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./logs:/var/log/causal-gym

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert.rules.yml:/etc/prometheus/alert.rules.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana-storage:
```

## Production Deployment

### Environment Variables

Configure monitoring through environment variables:

```bash
# Logging
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/causal-gym/app.log

# Metrics
export METRICS_ENABLED=true
export METRICS_PORT=8080

# Health checks
export HEALTH_CHECK_INTERVAL=30

# External services
export PROMETHEUS_GATEWAY=http://prometheus:9091
export GRAFANA_URL=http://grafana:3000
```

### Kubernetes Deployment

Example Kubernetes configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: causal-interface-gym
spec:
  replicas: 3
  selector:
    matchLabels:
      app: causal-interface-gym
  template:
    metadata:
      labels:
        app: causal-interface-gym
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: app
        image: causal-interface-gym:latest
        ports:
        - containerPort: 5000
        - containerPort: 8080
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: METRICS_ENABLED
          value: "true"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks in experiment code
   - Monitor large dataset processing
   - Implement data streaming for large experiments

2. **High Error Rates**
   - Check LLM service connectivity
   - Validate input data formats
   - Review exception handling

3. **Slow Response Times**
   - Profile slow functions
   - Optimize database queries
   - Consider caching frequently accessed data

### Debug Mode

Enable debug mode for detailed logging:

```python
from causal_interface_gym.monitoring import setup_logging

setup_logging(log_level="DEBUG", log_file="debug.log")
```

### Performance Profiling

Add performance profiling for detailed analysis:

```python
import cProfile
import pstats

def profile_experiment():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run experiment
    run_experiment()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)
```

This observability setup provides comprehensive monitoring capabilities for production deployment of the Causal Interface Gym framework.