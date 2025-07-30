"""Performance testing configuration and fixtures."""

import pytest
import time
from typing import Generator, Dict, Any
from pathlib import Path

@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "warmup_rounds": 3,
        "measurement_rounds": 10,
        "timeout_seconds": 30,
        "memory_threshold_mb": 100,
        "cpu_threshold_percent": 80
    }

@pytest.fixture
def performance_data_dir() -> Path:
    """Directory for performance test data."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir

@pytest.fixture
def timer() -> Generator[callable, None, None]:
    """High-precision timer for performance measurements."""
    def _timer():
        return time.perf_counter()
    yield _timer

@pytest.fixture
def memory_profiler():
    """Memory usage profiler fixture."""
    try:
        import psutil
        import os
        
        def get_memory_usage():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        
        return get_memory_usage
    except ImportError:
        pytest.skip("psutil not available for memory profiling")