# Performance Testing Guide

This guide covers performance testing strategies and implementation for the causal-interface-gym project.

## Overview

Performance testing ensures the causal reasoning algorithms scale efficiently and maintain responsiveness under various workloads.

## Testing Framework

### Pytest-Benchmark Integration

Install performance testing dependencies:

```bash
pip install pytest-benchmark pytest-profiling memory-profiler psutil
```

### Configuration

Create `pytest-benchmark.ini`:

```ini
[tool:pytest-benchmark]
min_rounds = 5
max_time = 2.0
min_time = 0.000005
timer = time.perf_counter
calibration_precision = 10
warmup = true
warmup_iterations = 10000
disable_gc = true
sort = min
columns = min, max, mean, stddev, median, iqr, outliers, rounds, iterations
histogram = true
```

## Performance Test Categories

### 1. Algorithm Performance Tests

#### Core Operations
- Environment creation with varying graph sizes
- Intervention operations (single and batch)
- Observation processing
- Causal effect estimation

#### Graph Algorithms
- Topological sorting
- Backdoor path identification
- D-separation testing
- Causal discovery algorithms

### 2. Scalability Tests

#### Linear Scalability
```python
@pytest.mark.parametrize("size", [10, 50, 100, 500, 1000])
def test_algorithm_scalability(size):
    """Test algorithm performance scales linearly with graph size."""
    dag = create_random_dag(size)
    env = CausalEnvironment.from_dag(dag)
    
    start_time = time.perf_counter()
    result = env.run_algorithm()
    duration = time.perf_counter() - start_time
    
    # Assert reasonable time bounds
    assert duration < size * 0.001  # 1ms per node max
```

#### Memory Usage
```python
def test_memory_efficiency(memory_profiler):
    """Test memory usage remains bounded."""
    initial_memory = memory_profiler()
    
    for size in [100, 500, 1000]:
        env = create_large_environment(size)
        current_memory = memory_profiler()
        memory_per_node = (current_memory - initial_memory) / size
        
        assert memory_per_node < 0.1  # 100KB per node max
```

### 3. Load Testing

#### Concurrent Operations
```python
import concurrent.futures
import threading

def test_concurrent_interventions():
    """Test performance under concurrent load."""
    env = create_test_environment()
    
    def run_intervention(intervention_id):
        return env.intervene(variable=intervention_id % 10)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_intervention, i) for i in range(100)]
        results = [f.result() for f in futures]
    
    assert len(results) == 100
    assert all(r is not None for r in results)
```

### 4. Stress Testing

#### Large Graph Handling
```python
@pytest.mark.slow
def test_very_large_graphs():
    """Test handling of very large causal graphs."""
    # 10,000 node graph
    dag = create_sparse_dag(10000, max_parents=3)
    
    # Should complete within reasonable time
    with timeout(60):  # 1 minute max
        env = CausalEnvironment.from_dag(dag)
        assert len(env.nodes) == 10000
```

## Performance Benchmarking

### Running Performance Tests

```bash
# Run all performance tests
pytest tests/performance/ -v

# Run with benchmark output
pytest tests/performance/ --benchmark-only --benchmark-verbose

# Run specific benchmark group
pytest tests/performance/ -k "environment_creation" --benchmark-only

# Generate benchmark report
pytest tests/performance/ --benchmark-json=benchmark_results.json

# Compare benchmarks across runs
py.test-benchmark compare benchmark_results.json --group-by=name
```

### Continuous Performance Monitoring

Add to CI pipeline:

```yaml
- name: Performance Tests
  run: |
    pytest tests/performance/ --benchmark-json=benchmark_results.json
    
- name: Performance Regression Check
  uses: benchmark-action/github-action-benchmark@v1
  with:
    tool: 'pytest'
    output-file-path: benchmark_results.json
    github-token: ${{ secrets.GITHUB_TOKEN }}
    alert-threshold: '150%'  # Alert if 50% slower
    comment-on-alert: true
    fail-on-alert: true
```

## Performance Profiling

### CPU Profiling

```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    """Decorator for profiling individual functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
        # Save profile data
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats(f'{func.__name__}_profile.prof')
        
        return result
    return wrapper

# Usage
@profile_function
def expensive_causal_computation():
    # Your algorithm here
    pass
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    """Function decorated for line-by-line memory profiling."""
    large_data = create_large_dataset()
    env = CausalEnvironment.from_data(large_data)
    return env.estimate_effects()

# Run with: python -m memory_profiler script.py
```

### Line Profiler

```python
# Install: pip install line_profiler

@profile  # Add this decorator
def bottleneck_function():
    """Function to profile line-by-line execution time."""
    for i in range(1000):
        expensive_operation(i)

# Run with: kernprof -l -v script.py
```

## Performance Optimization Guidelines

### Algorithm Optimization

1. **Use appropriate data structures**
   - NetworkX for graph operations
   - NumPy arrays for numerical computations
   - Pandas for data manipulation

2. **Implement caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def expensive_computation(graph_hash, parameters):
       # Cached computation
       return result
   ```

3. **Vectorized operations**
   ```python
   # Use NumPy vectorization
   import numpy as np
   
   # Slow: Python loops
   result = [func(x) for x in data]
   
   # Fast: Vectorized operations
   result = np.vectorize(func)(data)
   ```

### Memory Optimization

1. **Lazy evaluation**
   ```python
   def lazy_computation():
       """Generate results on-demand."""
       for result in compute_incrementally():
           yield result
   ```

2. **Memory-efficient data structures**
   ```python
   # Use generators instead of lists when possible
   def process_large_dataset():
       for chunk in read_data_chunks():
           yield process_chunk(chunk)
   ```

### Parallel Processing

```python
from multiprocessing import Pool
import concurrent.futures

def parallel_causal_analysis(datasets):
    """Process multiple datasets in parallel."""
    with Pool() as pool:
        results = pool.map(analyze_causal_structure, datasets)
    return results

async def async_causal_inference(graph_list):
    """Async processing for I/O bound operations."""
    async with aiohttp.ClientSession() as session:
        tasks = [infer_causality(graph, session) for graph in graph_list]
        results = await asyncio.gather(*tasks)
    return results
```

## Performance Metrics and Targets

### Response Time Targets

| Operation | Small Graph (<20 nodes) | Medium Graph (20-100 nodes) | Large Graph (100+ nodes) |
|-----------|-------------------------|------------------------------|---------------------------|
| Environment Creation | <1ms | <10ms | <100ms |
| Single Intervention | <0.1ms | <1ms | <10ms |
| Effect Estimation | <10ms | <100ms | <1s |
| Backdoor Identification | <1ms | <10ms | <100ms |

### Memory Usage Targets

| Graph Size | Peak Memory Usage | Memory per Node |
|------------|-------------------|-----------------|
| 100 nodes | <10MB | <100KB |
| 1000 nodes | <50MB | <50KB |
| 10000 nodes | <200MB | <20KB |

### Throughput Targets

- **Interventions per second**: >1000 for small graphs, >100 for large graphs
- **Concurrent users**: Support 10+ concurrent operations
- **Batch processing**: Process 100+ graphs in parallel

## Regression Detection

### Automated Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor and alert on performance regressions."""
    
    def __init__(self, baseline_file="performance_baseline.json"):
        self.baseline = self.load_baseline(baseline_file)
    
    def check_regression(self, current_results, threshold=1.5):
        """Check if current results show performance regression."""
        regressions = []
        
        for test_name, current_time in current_results.items():
            baseline_time = self.baseline.get(test_name)
            if baseline_time and current_time > baseline_time * threshold:
                regressions.append({
                    'test': test_name,
                    'baseline': baseline_time,
                    'current': current_time,
                    'regression': current_time / baseline_time
                })
        
        return regressions
    
    def update_baseline(self, new_results):
        """Update performance baseline with new results."""
        self.baseline.update(new_results)
        self.save_baseline()
```

### Integration with CI/CD

```yaml
# GitHub Actions workflow snippet
- name: Run Performance Tests
  run: pytest tests/performance/ --benchmark-json=results.json

- name: Check Performance Regression
  run: |
    python scripts/check_performance_regression.py \
      --results results.json \
      --baseline performance_baseline.json \
      --threshold 1.2

- name: Update Performance Baseline
  if: github.ref == 'refs/heads/main'
  run: |
    python scripts/update_performance_baseline.py \
      --results results.json
```

This comprehensive performance testing framework ensures that the causal reasoning algorithms maintain optimal performance as the codebase evolves and scales.