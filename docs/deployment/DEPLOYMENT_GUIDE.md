# Deployment Guide for Causal Interface Gym

*Last Updated: 2025-08-02*

## Overview

This guide covers various deployment options for the Causal Interface Gym, from local development to production-ready deployments. The project supports multiple deployment methods to accommodate different research and operational needs.

## Quick Start

### Local Development with Docker Compose

```bash
# Clone and setup
git clone https://github.com/yourusername/causal-interface-gym.git
cd causal-interface-gym

# Start development environment
docker-compose up dev

# Access services
# - Streamlit App: http://localhost:8501
# - Jupyter Lab: http://localhost:8888
```

### Local Python Installation

```bash
# Install in development mode
pip install -e ".[dev,ui,docs]"

# Run example
python examples/basic_usage.py

# Run tests
make test
```

## Deployment Options

### 1. Docker Compose (Recommended for Research)

**Best for**: Research environments, multi-service setups, local development

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d prod

# With database and caching
docker-compose up -d prod postgres redis

# Monitoring setup
docker-compose up -d prod postgres redis prometheus grafana
```

**Services Available:**
- `dev`: Development environment with hot reload
- `prod`: Production-optimized container
- `test`: Automated testing environment
- `docs`: Documentation server
- `jupyter`: Jupyter Lab server
- `benchmark`: Performance benchmarking
- `quality`: Code quality checks
- `postgres`: Database for experiment storage
- `redis`: Caching layer

### 2. Cloud Platforms

#### Google Cloud Platform

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/[PROJECT-ID]/causal-interface-gym:latest

# Deploy to Cloud Run
gcloud run deploy causal-interface-gym \
  --image gcr.io/[PROJECT-ID]/causal-interface-gym:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 10
```

#### AWS (Amazon ECS/Fargate)

```yaml
# task-definition.json
{
  "family": "causal-interface-gym",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "causal-interface-gym",
      "image": "your-account.dkr.ecr.region.amazonaws.com/causal-interface-gym:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/causal-interface-gym",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Microsoft Azure (Container Instances)

```bash
# Create resource group
az group create --name causal-gym-rg --location eastus

# Deploy container
az container create \
  --resource-group causal-gym-rg \
  --name causal-interface-gym \
  --image youracr.azurecr.io/causal-interface-gym:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8501 \
  --dns-name-label causal-gym-app \
  --environment-variables ENVIRONMENT=production
```

### 3. Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: causal-interface-gym
  labels:
    app: causal-interface-gym
spec:
  replicas: 3
  selector:
    matchLabels:
      app: causal-interface-gym
  template:
    metadata:
      labels:
        app: causal-interface-gym
    spec:
      containers:
      - name: causal-interface-gym
        image: causal-interface-gym:latest
        ports:
        - containerPort: 8501
        env:
        - name: ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: causal-interface-gym-service
spec:
  selector:
    app: causal-interface-gym
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8501
  type: LoadBalancer
```

### 4. Serverless Functions

#### AWS Lambda (for API endpoints)

```python
# lambda_handler.py
import json
from causal_interface_gym.core import CausalEnvironment

def lambda_handler(event, context):
    """Handle causal reasoning API requests."""
    try:
        # Parse request
        dag = json.loads(event['body'])['dag']
        intervention = json.loads(event['body'])['intervention']
        
        # Create environment and run intervention
        env = CausalEnvironment.from_dag(dag)
        result = env.intervene(**intervention)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'causal_effect': result.causal_effect,
                'confidence': result.confidence
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

#### Vercel (for frontend components)

```json
// vercel.json
{
  "builds": [
    {
      "src": "frontend/package.json",
      "use": "@vercel/node"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "/api/$1"
    },
    {
      "src": "/(.*)",
      "dest": "/frontend/$1"
    }
  ],
  "env": {
    "NODE_ENV": "production"
  }
}
```

## Environment Configuration

### Required Environment Variables

```bash
# Core Configuration
ENVIRONMENT=production
PYTHONPATH=/app/src

# LLM Provider APIs (choose relevant ones)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Database (optional)
DATABASE_URL=postgresql://user:pass@host:5432/causal_experiments
REDIS_URL=redis://host:6379/0

# Monitoring (optional)
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
PROMETHEUS_METRICS_PORT=8090

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret

# Performance
MAX_GRAPH_NODES=1000
WORKER_TIMEOUT=300
```

### Configuration Files

Create configuration files for different environments:

```yaml
# config/production.yaml
database:
  url: ${DATABASE_URL}
  pool_size: 10
  max_overflow: 20

redis:
  url: ${REDIS_URL}
  max_connections: 50

logging:
  level: INFO
  format: json
  handlers:
    - type: stream
      stream: stdout
    - type: file
      filename: /var/log/causal-gym.log

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  sampling_rate: 0.1

causal_reasoning:
  max_graph_nodes: 1000
  intervention_timeout: 30
  belief_extraction_method: "llm_parser"
```

## Security Considerations

### Container Security

```dockerfile
# Security best practices in Dockerfile
FROM python:3.11-slim as base

# Run as non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

# Update system packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies with security checks
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip check  # Verify no conflicting dependencies

# Switch to non-root user
USER appuser

# Set secure defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
```

### Network Security

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    networks:
      - frontend
      - backend
    environment:
      - CORS_ALLOWED_ORIGINS=https://yourdomain.com
    
  postgres:
    networks:
      - backend
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
    secrets:
      - postgres_password

  nginx:
    networks:
      - frontend
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs:ro

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
```

### API Security

```python
# Add rate limiting and authentication
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

security = HTTPBearer()

@app.post("/api/causal-inference")
@limiter.limit("10/minute")
async def causal_inference(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Validate API key
    if not validate_api_key(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Process request...
```

## Monitoring and Observability

### Health Checks

```python
# health.py
from fastapi import FastAPI, status
from causal_interface_gym.core import CausalEnvironment

app = FastAPI()

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    try:
        # Test core functionality
        env = CausalEnvironment.from_dag({"A": [], "B": ["A"]})
        env.intervene(A=1)
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": __version__
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }, status.HTTP_503_SERVICE_UNAVAILABLE

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    # Check database connectivity, external APIs, etc.
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "llm_apis": await check_llm_apis()
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return {"status": "not_ready", "checks": checks}, status.HTTP_503_SERVICE_UNAVAILABLE
```

### Logging Configuration

```yaml
# logging.yaml
version: 1
disable_existing_loggers: False

formatters:
  standard:
    format: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(levelname)s %(name)s %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: json
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /var/log/causal-gym.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  syslog:
    class: logging.handlers.SysLogHandler
    level: WARNING
    formatter: standard
    address: ['localhost', 514]

loggers:
  causal_interface_gym:
    level: INFO
    handlers: [console, file]
    propagate: no
    
  requests:
    level: WARNING
    handlers: [console]
    
  urllib3:
    level: WARNING
    handlers: [console]

root:
  level: INFO
  handlers: [console, file]
```

### Metrics Collection

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
causal_inference_requests = Counter(
    'causal_inference_requests_total',
    'Total causal inference requests',
    ['method', 'status']
)

causal_inference_duration = Histogram(
    'causal_inference_duration_seconds',
    'Time spent on causal inference',
    ['method']
)

active_experiments = Gauge(
    'active_experiments',
    'Number of active causal experiments'
)

graph_size = Histogram(
    'causal_graph_size',
    'Size of causal graphs processed',
    buckets=[10, 50, 100, 500, 1000, 5000]
)

def track_causal_inference(method: str):
    """Decorator to track causal inference metrics."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                causal_inference_requests.labels(
                    method=method, 
                    status=status
                ).inc()
                causal_inference_duration.labels(method=method).observe(duration)
        
        return wrapper
    return decorator

# Start metrics server
start_http_server(8090)
```

## Performance Optimization

### Caching Strategy

```python
# caching.py
import redis
import pickle
from functools import wraps
from typing import Any, Callable

redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))

def cache_causal_result(expiry: int = 3600):
    """Cache causal inference results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key
            cache_key = f"causal:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)
            
            # Compute result
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key, 
                expiry, 
                pickle.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

@cache_causal_result(expiry=1800)  # 30 minutes
def compute_causal_effect(dag, intervention, query):
    """Cached causal effect computation."""
    # Expensive causal inference computation
    pass
```

### Load Balancing

```nginx
# nginx.conf
upstream causal_gym_backend {
    least_conn;
    server app1:8501 max_fails=3 fail_timeout=30s;
    server app2:8501 max_fails=3 fail_timeout=30s;
    server app3:8501 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name causal-gym.yourdomain.com;
    
    location / {
        proxy_pass http://causal_gym_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 8k;
        proxy_buffers 16 8k;
    }
    
    location /health {
        access_log off;
        proxy_pass http://causal_gym_backend/health;
    }
}
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Graphs**
   ```bash
   # Increase container memory limits
   docker run -m 4g causal-interface-gym:latest
   
   # Monitor memory usage
   docker stats causal-interface-gym
   ```

2. **Slow LLM API Responses**
   ```python
   # Implement timeout and retry logic
   import asyncio
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(
       stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=4, max=10)
   )
   async def query_llm_with_retry(prompt: str):
       async with asyncio.timeout(30):  # 30 second timeout
           return await llm_client.query(prompt)
   ```

3. **Database Connection Issues**
   ```python
   # Connection pooling and health checks
   from sqlalchemy import create_engine
   from sqlalchemy.pool import QueuePool
   
   engine = create_engine(
       DATABASE_URL,
       poolclass=QueuePool,
       pool_size=10,
       max_overflow=20,
       pool_pre_ping=True,  # Validate connections
       pool_recycle=3600    # Recycle connections hourly
   )
   ```

### Debugging Commands

```bash
# Container debugging
docker exec -it causal-interface-gym bash
docker logs causal-interface-gym --follow

# Performance profiling
docker exec causal-interface-gym python -m cProfile -o profile.prof -m causal_interface_gym.examples.benchmark
docker cp causal-interface-gym:/app/profile.prof .

# Memory profiling
docker exec causal-interface-gym python -m memory_profiler examples/large_graph_test.py

# Network debugging
docker exec causal-interface-gym netstat -tulpn
docker exec causal-interface-gym curl -I http://localhost:8501/health
```

## Scaling Considerations

### Horizontal Scaling

- **Stateless Design**: All components should be stateless to enable horizontal scaling
- **Load Balancing**: Use nginx or cloud load balancers to distribute traffic
- **Caching**: Implement Redis for shared caching across instances
- **Database**: Use connection pooling and read replicas for database scaling

### Vertical Scaling

- **Memory**: Increase container memory for larger causal graphs
- **CPU**: Add more CPU cores for parallel processing
- **Storage**: Use SSD storage for faster I/O operations

### Auto-scaling

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: causal-gym-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: causal-interface-gym
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

*This deployment guide is updated regularly. For the latest information, check the [repository documentation](https://github.com/yourusername/causal-interface-gym/docs).*