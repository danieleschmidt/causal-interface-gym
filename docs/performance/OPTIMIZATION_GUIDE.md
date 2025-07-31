# Advanced Performance Optimization Guide

## Overview
This guide outlines advanced performance optimization strategies for the Causal Interface Gym project, focusing on computational efficiency, memory optimization, and scalability.

## Performance Profiling

### 1. CPU Profiling
```python
# Using py-spy for production profiling
py-spy record -o profile.svg -- python -m causal_interface_gym

# Using cProfile for detailed analysis
python -m cProfile -o profile.prof main.py
snakeviz profile.prof

# Using line_profiler for line-by-line analysis
kernprof -l -v causal_reasoning_benchmark.py
```

### 2. Memory Profiling
```python
# Memory usage tracking
from memory_profiler import profile
import tracemalloc

@profile
def causal_inference_heavy():
    # Your causal inference code here
    pass

# Tracemalloc for memory leak detection
tracemalloc.start()
# ... your code ...
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
```

### 3. GPU Profiling (if applicable)
```python
# NVIDIA profiling
nvprof python causal_gpu_computation.py

# PyTorch profiler integration
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/causal_reasoning')
) as prof:
    # Your causal reasoning code
    pass
```

## Computational Optimization

### 1. Algorithmic Improvements

#### Causal Graph Operations
```python
# Optimized DAG traversal using topological sorting
def optimized_causal_paths(graph):
    """Compute causal paths with O(V+E) complexity."""
    # Use NetworkX's optimized algorithms
    return nx.all_simple_paths(graph, source, target, cutoff=max_depth)

# Vectorized causal effect computation
def vectorized_causal_effects(interventions, observations):
    """Compute causal effects using NumPy vectorization."""
    return np.einsum('ij,jk->ik', interventions, observations)
```

#### Do-Calculus Optimization
```python
# Cached computation for repeated do-calculus operations
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_do_calculus(graph_hash, intervention, outcome):
    """Cache do-calculus results for repeated queries."""
    return compute_do_calculus(graph_hash, intervention, outcome)
```

### 2. Data Structure Optimization

#### Sparse Matrix Operations
```python
from scipy.sparse import csr_matrix, linalg

# Use sparse matrices for large causal graphs
def sparse_causal_matrix(adjacency_list):
    """Convert adjacency list to sparse matrix representation."""
    row, col, data = [], [], []
    for node, neighbors in adjacency_list.items():
        for neighbor, weight in neighbors.items():
            row.append(node)
            col.append(neighbor)
            data.append(weight)
    return csr_matrix((data, (row, col)))
```

#### Memory-Efficient Data Loading
```python
# Lazy loading for large datasets
class LazyDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_parquet(self.data_path, engine='pyarrow')
        return self._data
```

## Parallel Processing

### 1. Multiprocessing for CPU-bound Tasks
```python
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

def parallel_causal_discovery(datasets):
    """Parallel causal discovery across multiple datasets."""
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [
            executor.submit(discover_causal_structure, dataset)
            for dataset in datasets
        ]
        results = [future.result() for future in futures]
    return results
```

### 2. Async Processing for I/O-bound Tasks
```python
import asyncio
import aiohttp

async def async_llm_evaluation(models, prompts):
    """Async evaluation of multiple LLM models."""
    async with aiohttp.ClientSession() as session:
        tasks = [
            evaluate_model_async(session, model, prompt)
            for model in models
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
    return results
```

### 3. GPU Acceleration
```python
import cupy as cp  # GPU-accelerated NumPy

def gpu_causal_computation(data):
    """GPU-accelerated causal computation using CuPy."""
    gpu_data = cp.asarray(data)
    # Perform causal computations on GPU
    result = cp.linalg.inv(gpu_data @ gpu_data.T)
    return cp.asnumpy(result)  # Convert back to CPU if needed
```

## Memory Optimization

### 1. Memory Pool Management
```python
import numpy as np
from pympler import tracker

class MemoryEfficientCausalEngine:
    def __init__(self, max_memory_gb=4):
        self.max_memory = max_memory_gb * 1024**3
        self.memory_tracker = tracker.SummaryTracker()
        
    def __enter__(self):
        self.memory_tracker.print_diff()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.memory_tracker.print_diff()
```

### 2. Streaming Data Processing
```python
def stream_process_causal_data(file_path, chunk_size=10000):
    """Process large datasets in chunks to manage memory."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield process_causal_relationships(chunk)
```

## Caching Strategies

### 1. Multi-level Caching
```python
from functools import lru_cache
import redis
import pickle

class CausalCache:
    def __init__(self, redis_client=None, max_local_size=1000):
        self.redis = redis_client or redis.Redis()
        self.local_cache = {}
        self.max_local_size = max_local_size
    
    def get(self, key):
        # L1: Local cache
        if key in self.local_cache:
            return self.local_cache[key]
        
        # L2: Redis cache
        value = self.redis.get(key)
        if value:
            result = pickle.loads(value)
            self.local_cache[key] = result
            return result
        
        return None
    
    def set(self, key, value, ttl=3600):
        # Store in both caches
        self.local_cache[key] = value
        self.redis.setex(key, ttl, pickle.dumps(value))
        
        # Manage local cache size
        if len(self.local_cache) > self.max_local_size:
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
```

### 2. Persistent Result Caching
```python
import diskcache as dc

# Persistent cache for expensive computations
cache = dc.Cache('./cache/causal_results')

@cache.memoize(expire=86400)  # 24 hours
def expensive_causal_computation(graph, intervention):
    """Cache expensive causal computations to disk."""
    return compute_causal_effect(graph, intervention)
```

## Network Optimization

### 1. Connection Pooling
```python
import httpx
from urllib3.util.retry import Retry

class OptimizedLLMClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            timeout=httpx.Timeout(30.0),
            transport=httpx.AsyncHTTPTransport(retries=3)
        )
    
    async def batch_evaluate(self, prompts):
        """Batch LLM evaluation with connection reuse."""
        tasks = [self.evaluate_single(prompt) for prompt in prompts]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### 2. Request Batching
```python
class BatchedLLMEvaluator:
    def __init__(self, batch_size=10, flush_interval=5.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_requests = []
        
    async def evaluate(self, prompt):
        """Add to batch and process when full."""
        self.pending_requests.append(prompt)
        
        if len(self.pending_requests) >= self.batch_size:
            return await self._flush_batch()
        
        # Set timer for partial batch
        asyncio.create_task(self._flush_after_delay())
    
    async def _flush_batch(self):
        """Process accumulated batch."""
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        return await self._process_batch(batch)
```

## Database Optimization

### 1. Query Optimization
```python
# Optimized database queries for causal data
def get_causal_relationships_optimized(session, limit=1000):
    """Use efficient queries with proper indexing."""
    return session.query(CausalRelationship)\
        .options(joinedload(CausalRelationship.variables))\
        .filter(CausalRelationship.confidence > 0.8)\
        .order_by(CausalRelationship.created_at.desc())\
        .limit(limit)\
        .all()
```

### 2. Connection Pool Tuning
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Optimized database connection pool
engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Disable SQL logging in production
)
```

## Performance Monitoring

### 1. Custom Metrics
```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Performance metrics
causal_computation_duration = Histogram(
    'causal_computation_duration_seconds',
    'Time spent on causal computations',
    ['operation_type']
)

llm_evaluation_counter = Counter(
    'llm_evaluations_total',
    'Total LLM evaluations performed',
    ['model_name', 'result']
)

memory_usage_gauge = Gauge(
    'memory_usage_bytes',
    'Current memory usage'
)

@causal_computation_duration.labels(operation_type='do_calculus').time()
def timed_do_calculus(graph, intervention):
    """Timed causal computation with metrics."""
    return compute_do_calculus(graph, intervention)
```

### 2. Performance Alerts
```yaml
# Prometheus alerting rules
groups:
  - name: causal_interface_gym_performance
    rules:
      - alert: HighLatency
        expr: causal_computation_duration_seconds{quantile="0.95"} > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency in causal computations"
          
      - alert: HighMemoryUsage
        expr: memory_usage_bytes / (1024^3) > 8  # 8GB
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage detected"
```

## Benchmarking Framework

### 1. Performance Benchmarks
```python
import pytest
from pytest_benchmark import benchmark

class TestCausalPerformance:
    def test_do_calculus_performance(self, benchmark):
        """Benchmark do-calculus computation."""
        graph = create_test_graph(nodes=1000, edges=5000)
        intervention = {"X": 1}
        
        result = benchmark(compute_do_calculus, graph, intervention)
        assert result is not None
    
    def test_llm_evaluation_performance(self, benchmark):
        """Benchmark LLM evaluation throughput."""
        prompts = generate_test_prompts(100)
        
        result = benchmark(batch_evaluate_llm, prompts)
        assert len(result) == 100
```

### 2. Load Testing
```python
import locust
from locust import HttpUser, task, between

class CausalInterfaceUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def evaluate_causal_reasoning(self):
        """Simulate causal reasoning evaluation."""
        self.client.post("/api/v1/evaluate", json={
            "graph": self.generate_test_graph(),
            "intervention": {"X": 1},
            "model": "gpt-4"
        })
    
    @task(1)
    def get_metrics(self):
        """Check metrics endpoint."""
        self.client.get("/metrics")
```

## Optimization Checklist

### Development Phase
- [ ] Profile code with py-spy and cProfile
- [ ] Implement caching for expensive operations
- [ ] Use vectorized operations where possible
- [ ] Optimize data structures (sparse matrices, efficient containers)
- [ ] Implement lazy loading for large datasets

### Testing Phase
- [ ] Run performance benchmarks
- [ ] Conduct load testing
- [ ] Monitor memory usage patterns
- [ ] Profile database queries
- [ ] Test with realistic data volumes

### Production Phase
- [ ] Enable production-grade caching (Redis)
- [ ] Configure connection pooling
- [ ] Set up performance monitoring
- [ ] Implement alerting for performance degradation
- [ ] Regular performance audits

## Performance Targets

### Response Time Targets
- **Causal Graph Creation**: < 100ms (p95)
- **Do-Calculus Computation**: < 500ms (p95)
- **LLM Evaluation**: < 2s (p95)
- **UI Interactions**: < 50ms (p95)

### Throughput Targets
- **Concurrent Users**: 1000+
- **API Requests/sec**: 500+
- **Causal Computations/sec**: 100+
- **LLM Evaluations/min**: 1000+

### Resource Targets
- **Memory Usage**: < 4GB per instance
- **CPU Utilization**: < 70% average
- **Database Connections**: < 50 per instance
- **Cache Hit Rate**: > 80%

## Troubleshooting

### Common Performance Issues
1. **Memory Leaks**: Use tracemalloc and memory_profiler
2. **Slow Database Queries**: Enable query logging and analyze execution plans
3. **High CPU Usage**: Profile with py-spy and optimize hot paths
4. **Network Bottlenecks**: Implement connection pooling and request batching
5. **Cache Misses**: Analyze cache patterns and adjust TTL settings

### Debug Tools
```bash
# Memory debugging
python -m memory_profiler your_script.py

# CPU profiling
py-spy record -o profile.svg -- python your_script.py

# Database query analysis
EXPLAIN ANALYZE SELECT * FROM causal_relationships WHERE ...

# Network debugging
curl -w "@curl-format.txt" -o /dev/null -s "http://api/endpoint"
```

This optimization guide provides a comprehensive framework for achieving high performance in the Causal Interface Gym project across all dimensions of computational efficiency.