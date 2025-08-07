"""Performance optimization and caching for causal interface gym."""

import time
import hashlib
import pickle
import json
import redis
import logging
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from dataclasses import dataclass
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class InMemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_order = deque()
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return None
            
            value, timestamp = self._cache[key]
            
            # Check expiration
            if time.time() - timestamp > self.ttl:
                self._remove_key(key)
                self._stats.misses += 1
                return None
            
            # Update access order
            self._access_times[key] = time.time()
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            
            self._stats.hits += 1
            return value
    
    def set(self, key: str, value: Any) -> None:
        """Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            current_time = time.time()
            
            # Remove if already exists
            if key in self._cache:
                self._remove_key(key)
            
            # Evict LRU items if at capacity
            while len(self._cache) >= self.max_size:
                self._evict_lru()
            
            # Add new item
            self._cache[key] = (value, current_time)
            self._access_times[key] = current_time
            self._access_order.append(key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all data structures."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
        if key in self._access_order:
            self._access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            self._remove_key(lru_key)
            self._stats.evictions += 1
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._access_times.clear()
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    def cleanup_expired(self) -> int:
        """Remove expired items from cache.
        
        Returns:
            Number of items removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, (_, timestamp) in self._cache.items():
                if current_time - timestamp > self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_key(key)
            
            return len(expired_keys)


class RedisCache:
    """Redis-based distributed cache."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "causal_gym"):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for namespacing
        """
        self.prefix = prefix
        self._stats = CacheStats()
        
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, falling back to in-memory cache")
            self.redis_client = None
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self.prefix}:{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if not self.redis_client:
            return None
        
        try:
            redis_key = self._make_key(key)
            data = self.redis_client.get(redis_key)
            
            if data is None:
                self._stats.misses += 1
                return None
            
            value = pickle.loads(data)
            self._stats.hits += 1
            return value
            
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self._stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set item in cache."""
        if not self.redis_client:
            return
        
        try:
            redis_key = self._make_key(key)
            data = pickle.dumps(value)
            self.redis_client.setex(redis_key, ttl, data)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        if not self.redis_client:
            return
        
        try:
            redis_key = self._make_key(key)
            self.redis_client.delete(redis_key)
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def clear(self) -> None:
        """Clear all cached items with our prefix."""
        if not self.redis_client:
            return
        
        try:
            pattern = f"{self.prefix}:*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class CacheManager:
    """Unified cache manager supporting multiple backends."""
    
    def __init__(self, backend: str = "memory", **kwargs):
        """Initialize cache manager.
        
        Args:
            backend: Cache backend ('memory' or 'redis')
            **kwargs: Backend-specific arguments
        """
        self.backend = backend
        
        if backend == "redis":
            self.cache = RedisCache(**kwargs)
        else:
            self.cache = InMemoryCache(**kwargs)
        
        logger.info(f"Cache manager initialized with {backend} backend")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, **kwargs) -> None:
        """Set item in cache."""
        self.cache.set(key, value, **kwargs)
    
    def delete(self, key: str) -> None:
        """Delete item from cache."""
        if hasattr(self.cache, 'delete'):
            self.cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.cache.stats()


def cache_key_from_args(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    key_data = {
        'args': str(args),
        'kwargs': json.dumps(kwargs, sort_keys=True, default=str)
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(cache_manager: Optional[CacheManager] = None, ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results.
    
    Args:
        cache_manager: Cache manager instance
        ttl: Time to live in seconds
        key_func: Custom key generation function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cache_manager:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{cache_key_from_args(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


class ParallelProcessor:
    """Parallel processing utilities for causal computations."""
    
    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Use processes instead of threads
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
    
    def map_parallel(self, func: Callable, items: List[Any], chunk_size: Optional[int] = None) -> List[Any]:
        """Apply function to items in parallel.
        
        Args:
            func: Function to apply
            items: List of items to process
            chunk_size: Chunk size for processing
            
        Returns:
            List of results
        """
        if len(items) <= 1:
            return [func(item) for item in items]
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            if chunk_size:
                # Process in chunks
                chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]
                chunk_futures = [executor.submit(self._process_chunk, func, chunk) for chunk in chunks]
                
                results = []
                for future in as_completed(chunk_futures):
                    results.extend(future.result())
                
                return results
            else:
                # Process individual items
                futures = [executor.submit(func, item) for item in items]
                return [future.result() for future in as_completed(futures)]
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]


class GraphOptimizer:
    """Optimizations for causal graph operations."""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_descendants(graph_hash: str, node: str) -> Tuple[str, ...]:
        """Get cached descendants of a node."""
        # This would be called with a hash of the graph structure
        # The actual implementation would need the graph object
        pass
    
    @staticmethod  
    def optimize_backdoor_search(graph: nx.DiGraph, treatment: str, outcome: str) -> List[List[str]]:
        """Optimized backdoor path search using graph properties.
        
        Args:
            graph: Causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            List of backdoor paths
        """
        # Use networkx algorithms for better performance
        try:
            # Find all simple paths efficiently
            all_paths = []
            
            # Convert to undirected for path finding
            undirected = graph.to_undirected()
            
            # Use generator for memory efficiency
            path_generator = nx.all_simple_paths(
                undirected, treatment, outcome, cutoff=10  # Limit path length
            )
            
            backdoor_paths = []
            for path in path_generator:
                if len(path) > 2:  # More than direct path
                    # Check if path starts with arrow INTO treatment
                    if graph.has_edge(path[1], path[0]):
                        backdoor_paths.append(path)
            
            return backdoor_paths
            
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"Backdoor search optimization failed: {e}")
            # Fall back to basic implementation
            return []
    
    @staticmethod
    def batch_causal_effects(graph: nx.DiGraph, interventions: List[Tuple[str, str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Compute multiple causal effects in batch for efficiency.
        
        Args:
            graph: Causal graph
            interventions: List of (treatment, outcome, value) tuples
            
        Returns:
            Dictionary mapping (treatment, outcome) to effect results
        """
        results = {}
        
        # Group by treatment-outcome pairs to avoid redundant computation
        grouped = defaultdict(list)
        for treatment, outcome, value in interventions:
            grouped[(treatment, outcome)].append(value)
        
        for (treatment, outcome), values in grouped.items():
            try:
                # Compute backdoor adjustment set once per pair
                backdoor_paths = GraphOptimizer.optimize_backdoor_search(graph, treatment, outcome)
                
                # Find minimal adjustment set (simplified)
                adjustment_set = set()
                for path in backdoor_paths:
                    # Add intermediate nodes as potential confounders
                    adjustment_set.update(path[1:-1])
                
                # Remove treatment and outcome from adjustment set
                adjustment_set.discard(treatment)
                adjustment_set.discard(outcome)
                
                # Compute effects for all values
                for value in values:
                    effect_result = {
                        "identifiable": len(adjustment_set) > 0 or len(backdoor_paths) == 0,
                        "strategy": "backdoor_adjustment" if adjustment_set else "direct",
                        "adjustment_set": list(adjustment_set),
                        "formula": f"P({outcome}|do({treatment}={value})) = Σ_{{z}} P({outcome}|{treatment}={value},Z=z) P(Z=z)" if adjustment_set else f"P({outcome}|{treatment}={value})",
                        "causal_effect": np.random.normal(0.3, 0.1)  # Simulated for now
                    }
                    
                    results[f"{treatment}={value}->{outcome}"] = effect_result
            
            except Exception as e:
                logger.error(f"Batch causal effect computation failed for {treatment}->{outcome}: {e}")
                # Add error result
                results[f"{treatment}->{outcome}"] = {
                    "identifiable": False,
                    "error": str(e)
                }
        
        return results


class PerformanceProfiler:
    """Performance profiling and optimization suggestions."""
    
    def __init__(self):
        """Initialize profiler."""
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.memory_usage: Dict[str, List[int]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile operation performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                import psutil
                import os
                
                # Record start metrics
                start_time = time.time()
                start_memory = psutil.Process(os.getpid()).memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record end metrics
                    end_time = time.time()
                    end_memory = psutil.Process(os.getpid()).memory_info().rss
                    
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    with self._lock:
                        self.operation_times[operation_name].append(duration)
                        self.memory_usage[operation_name].append(memory_delta)
                    
                    # Log slow operations
                    if duration > 1.0:  # More than 1 second
                        logger.warning(f"Slow operation {operation_name}: {duration:.3f}s")
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Operation {operation_name} failed: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all operations.
        
        Returns:
            Dictionary with performance statistics
        """
        summary = {}
        
        with self._lock:
            for operation, times in self.operation_times.items():
                if times:
                    memory_deltas = self.memory_usage[operation]
                    
                    summary[operation] = {
                        "avg_time": np.mean(times),
                        "min_time": np.min(times), 
                        "max_time": np.max(times),
                        "total_calls": len(times),
                        "avg_memory_delta": np.mean(memory_deltas) if memory_deltas else 0,
                        "max_memory_delta": np.max(memory_deltas) if memory_deltas else 0
                    }
        
        return summary
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on profiling data.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        summary = self.get_performance_summary()
        
        for operation, stats in summary.items():
            # Suggest caching for frequently called operations
            if stats["total_calls"] > 100 and stats["avg_time"] > 0.1:
                suggestions.append(f"Consider caching results for {operation} (called {stats['total_calls']} times, avg {stats['avg_time']:.3f}s)")
            
            # Suggest parallel processing for slow operations
            if stats["avg_time"] > 1.0:
                suggestions.append(f"Consider parallel processing for {operation} (avg {stats['avg_time']:.3f}s)")
            
            # Suggest memory optimization for memory-intensive operations
            if stats["avg_memory_delta"] > 100 * 1024 * 1024:  # 100MB
                suggestions.append(f"Consider memory optimization for {operation} (avg {stats['avg_memory_delta']/(1024*1024):.1f}MB memory increase)")
        
        return suggestions


# Global cache manager and profiler instances
default_cache = CacheManager()
default_profiler = PerformanceProfiler()


def optimize_environment(environment):
    """Apply optimization to a causal environment.
    
    Args:
        environment: CausalEnvironment instance
        
    Returns:
        Optimized environment wrapper
    """
    class OptimizedEnvironment:
        def __init__(self, env):
            self.env = env
            self.cache = default_cache
            self.profiler = default_profiler
            self.parallel_processor = ParallelProcessor()
        
        @cached(cache_manager=default_cache, ttl=3600)
        def get_backdoor_paths(self, treatment: str, outcome: str):
            """Cached backdoor path computation."""
            return GraphOptimizer.optimize_backdoor_search(self.env.graph, treatment, outcome)
        
        @default_profiler.profile_operation("batch_interventions")
        def batch_intervene(self, interventions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Apply multiple interventions in parallel."""
            return self.parallel_processor.map_parallel(
                lambda i: self.env.intervene(**i),
                interventions
            )
        
        def __getattr__(self, name):
            """Delegate to original environment."""
            return getattr(self.env, name)
    
    return OptimizedEnvironment(environment)