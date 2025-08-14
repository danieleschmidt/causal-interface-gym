"""Advanced performance optimization with adaptive caching and memory management."""

import asyncio
import time
import threading
import weakref
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import logging
import hashlib
import pickle
import psutil
import numpy as np
from functools import wraps, lru_cache
import gc
import resource

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cache_hit_rates: Dict[str, float] = field(default_factory=dict)
    throughput_metrics: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    resource_utilization: Dict[str, float] = field(default_factory=dict)


class PerformanceOptimizer:
    """Intelligent performance optimization system with adaptive learning."""
    
    def __init__(self, 
                 target_response_time: float = 0.1,
                 memory_threshold: float = 0.8,
                 optimization_interval: int = 300):
        """Initialize performance optimizer.
        
        Args:
            target_response_time: Target response time in seconds
            memory_threshold: Memory usage threshold (0-1)
            optimization_interval: Optimization check interval in seconds
        """
        self.target_response_time = target_response_time
        self.memory_threshold = memory_threshold
        self.optimization_interval = optimization_interval
        
        self.metrics = PerformanceMetrics()
        self.cache_manager = CacheManager()
        self.memory_manager = MemoryManager()
        
        self._optimization_thread = None
        self._stop_optimization = False
        self._performance_history: Dict[str, List[float]] = defaultdict(list)
        self._adaptive_thresholds: Dict[str, float] = {}
        
        # Start continuous optimization
        self.start_optimization_loop()
        
    def start_optimization_loop(self):
        """Start the continuous optimization loop."""
        if self._optimization_thread is None or not self._optimization_thread.is_alive():
            self._stop_optimization = False
            self._optimization_thread = threading.Thread(
                target=self._optimization_loop, daemon=True)
            self._optimization_thread.start()
            logger.info("Started performance optimization loop")
            
    def stop_optimization_loop(self):
        """Stop the optimization loop."""
        self._stop_optimization = True
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5.0)
        logger.info("Stopped performance optimization loop")
        
    def _optimization_loop(self):
        """Continuous optimization monitoring loop."""
        while not self._stop_optimization:
            try:
                self._analyze_and_optimize()
                time.sleep(self.optimization_interval)
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(60)  # Wait longer after error
                
    def _analyze_and_optimize(self):
        """Analyze performance and apply optimizations."""
        # Collect current metrics
        current_metrics = self._collect_current_metrics()
        
        # Update performance history
        for metric, value in current_metrics.items():
            self._performance_history[metric].append(value)
            # Keep only recent history
            if len(self._performance_history[metric]) > 1000:
                self._performance_history[metric] = self._performance_history[metric][-1000:]
                
        # Analyze trends and apply optimizations
        self._optimize_based_on_trends()
        
        # Memory management
        if current_metrics.get('memory_usage', 0) > self.memory_threshold:
            self.memory_manager.emergency_cleanup()
            
        # Cache optimization
        self.cache_manager.optimize_cache_sizes()
        
        logger.debug(f"Performance optimization completed: {current_metrics}")
        
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system performance metrics."""
        process = psutil.Process()
        
        metrics = {
            'memory_usage': process.memory_percent() / 100.0,
            'cpu_usage': process.cpu_percent() / 100.0,
            'memory_bytes': process.memory_info().rss / (1024 * 1024),  # MB
            'open_files': len(process.open_files()),
            'threads': process.num_threads()
        }
        
        # Add cache metrics
        cache_stats = self.cache_manager.get_cache_statistics()
        metrics.update(cache_stats)
        
        return metrics
        
    def _optimize_based_on_trends(self):
        """Apply optimizations based on performance trends."""
        # Analyze response time trends
        if 'response_time' in self._performance_history:
            recent_times = self._performance_history['response_time'][-50:]
            if recent_times and np.mean(recent_times) > self.target_response_time * 1.5:
                # Response times are too high
                self._apply_response_time_optimizations()
                
        # Analyze memory trends
        if 'memory_usage' in self._performance_history:
            recent_memory = self._performance_history['memory_usage'][-20:]
            if recent_memory and np.mean(recent_memory) > self.memory_threshold:
                # Memory usage is too high
                self._apply_memory_optimizations()
                
    def _apply_response_time_optimizations(self):
        """Apply optimizations for response time."""
        logger.info("Applying response time optimizations")
        
        # Increase cache sizes for frequently accessed data
        self.cache_manager.increase_cache_sizes(factor=1.2)
        
        # Enable more aggressive caching
        self.cache_manager.set_aggressive_caching(True)
        
    def _apply_memory_optimizations(self):
        """Apply optimizations for memory usage."""
        logger.info("Applying memory optimizations")
        
        # Reduce cache sizes
        self.cache_manager.reduce_cache_sizes(factor=0.8)
        
        # Force garbage collection
        self.memory_manager.force_garbage_collection()
        
        # Clean up weak references
        self.memory_manager.cleanup_weak_references()
        
    def track_function_performance(self, func_name: str = None):
        """Decorator to track function performance."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = e
                    result = None
                    
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                # Record metrics
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                self.metrics.execution_times.append(execution_time)
                self.metrics.memory_usage.append(end_memory / (1024 * 1024))  # MB
                
                name = func_name or func.__name__
                self._performance_history[f'{name}_time'].append(execution_time)
                self._performance_history[f'{name}_memory'].append(memory_delta)
                
                if not success:
                    self._performance_history[f'{name}_errors'].append(1)
                    raise error
                    
                return result
                
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    success = False
                    error = e
                    result = None
                    
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss
                
                # Record metrics
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                self.metrics.execution_times.append(execution_time)
                self.metrics.memory_usage.append(end_memory / (1024 * 1024))  # MB
                
                name = func_name or func.__name__
                self._performance_history[f'{name}_time'].append(execution_time)
                self._performance_history[f'{name}_memory'].append(memory_delta)
                
                if not success:
                    self._performance_history[f'{name}_errors'].append(1)
                    raise error
                    
                return result
                
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'current_metrics': self._collect_current_metrics(),
            'cache_statistics': self.cache_manager.get_cache_statistics(),
            'memory_statistics': self.memory_manager.get_memory_statistics(),
            'performance_trends': {}
        }
        
        # Analyze trends
        for metric, values in self._performance_history.items():
            if values:
                report['performance_trends'][metric] = {
                    'current': values[-1],
                    'average_recent': np.mean(values[-50:]) if len(values) >= 50 else np.mean(values),
                    'trend': 'improving' if len(values) > 10 and np.mean(values[-10:]) < np.mean(values[-20:-10]) else 'stable',
                    'count': len(values)
                }
                
        return report


class CacheManager:
    """Advanced caching system with adaptive cache sizing and intelligent eviction."""
    
    def __init__(self, 
                 initial_size: int = 1000,
                 max_memory_mb: float = 512.0):
        """Initialize cache manager.
        
        Args:
            initial_size: Initial cache size
            max_memory_mb: Maximum memory for caches in MB
        """
        self.initial_size = initial_size
        self.max_memory_mb = max_memory_mb
        
        # Multi-level caches
        self._l1_cache = OrderedDict()  # Hot data
        self._l2_cache = OrderedDict()  # Warm data  
        self._l3_cache = OrderedDict()  # Cold data
        
        self._cache_sizes = {
            'l1': initial_size // 4,
            'l2': initial_size // 2,
            'l3': initial_size
        }
        
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._access_times: Dict[str, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._aggressive_caching = False
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with promotion logic."""
        with self._lock:
            current_time = time.time()
            
            # Check L1 cache first
            if key in self._l1_cache:
                value = self._l1_cache[key]
                # Move to end (most recent)
                self._l1_cache.move_to_end(key)
                self._access_counts[key] += 1
                self._access_times[key] = current_time
                self._cache_hits += 1
                return value
                
            # Check L2 cache
            if key in self._l2_cache:
                value = self._l2_cache.pop(key)
                # Promote to L1 if frequently accessed
                self._access_counts[key] += 1
                self._access_times[key] = current_time
                
                if self._access_counts[key] > 5:  # Promote to L1
                    self._put_l1(key, value)
                else:
                    self._l2_cache[key] = value
                    self._l2_cache.move_to_end(key)
                    
                self._cache_hits += 1
                return value
                
            # Check L3 cache
            if key in self._l3_cache:
                value = self._l3_cache.pop(key)
                self._access_counts[key] += 1
                self._access_times[key] = current_time
                
                # Promote to L2
                self._put_l2(key, value)
                self._cache_hits += 1
                return value
                
            self._cache_misses += 1
            return None
            
    def put(self, key: str, value: Any, ttl: Optional[float] = None):
        """Put value in cache with intelligent placement."""
        with self._lock:
            current_time = time.time()
            
            # Store in L3 initially (will be promoted with access)
            self._put_l3(key, value)
            self._access_times[key] = current_time
            
            # Set TTL if specified
            if ttl:
                self._access_times[f"{key}_ttl"] = current_time + ttl
                
    def _put_l1(self, key: str, value: Any):
        """Put in L1 cache with eviction."""
        self._l1_cache[key] = value
        if len(self._l1_cache) > self._cache_sizes['l1']:
            # Evict to L2
            oldest_key, oldest_value = self._l1_cache.popitem(last=False)
            self._put_l2(oldest_key, oldest_value)
            
    def _put_l2(self, key: str, value: Any):
        """Put in L2 cache with eviction."""
        self._l2_cache[key] = value
        if len(self._l2_cache) > self._cache_sizes['l2']:
            # Evict to L3
            oldest_key, oldest_value = self._l2_cache.popitem(last=False)
            self._put_l3(oldest_key, oldest_value)
            
    def _put_l3(self, key: str, value: Any):
        """Put in L3 cache with eviction."""
        self._l3_cache[key] = value
        if len(self._l3_cache) > self._cache_sizes['l3']:
            # True eviction
            self._l3_cache.popitem(last=False)
            
    def invalidate(self, key: str):
        """Invalidate cache entry."""
        with self._lock:
            self._l1_cache.pop(key, None)
            self._l2_cache.pop(key, None)
            self._l3_cache.pop(key, None)
            self._access_counts.pop(key, None)
            self._access_times.pop(key, None)
            
    def clear_expired(self):
        """Clear expired cache entries."""
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            for key, expiry_time in list(self._access_times.items()):
                if key.endswith('_ttl') and expiry_time <= current_time:
                    original_key = key[:-4]  # Remove '_ttl'
                    expired_keys.append(original_key)
                    
            for key in expired_keys:
                self.invalidate(key)
                
    def optimize_cache_sizes(self):
        """Optimize cache sizes based on usage patterns."""
        with self._lock:
            total_accesses = sum(self._access_counts.values())
            if total_accesses == 0:
                return
                
            # Calculate hit rates for each level
            l1_hits = sum(count for key, count in self._access_counts.items() 
                         if key in self._l1_cache)
            l2_hits = sum(count for key, count in self._access_counts.items() 
                         if key in self._l2_cache)
                         
            l1_hit_rate = l1_hits / total_accesses if total_accesses > 0 else 0
            l2_hit_rate = l2_hits / total_accesses if total_accesses > 0 else 0
            
            # Adjust sizes based on hit rates
            if l1_hit_rate > 0.8:  # L1 very effective
                self._cache_sizes['l1'] = min(self._cache_sizes['l1'] * 1.1, self.initial_size)
            elif l1_hit_rate < 0.3:  # L1 not effective
                self._cache_sizes['l1'] = max(self._cache_sizes['l1'] * 0.9, self.initial_size // 8)
                
            if l2_hit_rate > 0.6:  # L2 effective
                self._cache_sizes['l2'] = min(self._cache_sizes['l2'] * 1.05, self.initial_size * 2)
                
    def increase_cache_sizes(self, factor: float = 1.2):
        """Increase cache sizes by factor."""
        with self._lock:
            for level in self._cache_sizes:
                self._cache_sizes[level] = int(self._cache_sizes[level] * factor)
                
    def reduce_cache_sizes(self, factor: float = 0.8):
        """Reduce cache sizes by factor."""
        with self._lock:
            for level in self._cache_sizes:
                self._cache_sizes[level] = max(1, int(self._cache_sizes[level] * factor))
                
            # Evict excess entries
            self._enforce_size_limits()
            
    def _enforce_size_limits(self):
        """Enforce current size limits by evicting excess entries."""
        while len(self._l1_cache) > self._cache_sizes['l1']:
            self._l1_cache.popitem(last=False)
        while len(self._l2_cache) > self._cache_sizes['l2']:
            self._l2_cache.popitem(last=False)  
        while len(self._l3_cache) > self._cache_sizes['l3']:
            self._l3_cache.popitem(last=False)
            
    def set_aggressive_caching(self, aggressive: bool):
        """Enable/disable aggressive caching mode."""
        self._aggressive_caching = aggressive
        if aggressive:
            # Increase all cache sizes
            self.increase_cache_sizes(1.5)
        else:
            # Return to normal sizes
            self._cache_sizes = {
                'l1': self.initial_size // 4,
                'l2': self.initial_size // 2,
                'l3': self.initial_size
            }
            
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'cache_hits': self._cache_hits,
                'cache_misses': self._cache_misses,
                'l1_size': len(self._l1_cache),
                'l2_size': len(self._l2_cache),
                'l3_size': len(self._l3_cache),
                'l1_capacity': self._cache_sizes['l1'],
                'l2_capacity': self._cache_sizes['l2'],
                'l3_capacity': self._cache_sizes['l3'],
                'aggressive_mode': self._aggressive_caching
            }


class MemoryManager:
    """Advanced memory management with automatic cleanup and optimization."""
    
    def __init__(self, 
                 cleanup_threshold: float = 0.8,
                 cleanup_interval: int = 300):
        """Initialize memory manager.
        
        Args:
            cleanup_threshold: Memory threshold for automatic cleanup (0-1)
            cleanup_interval: Cleanup check interval in seconds
        """
        self.cleanup_threshold = cleanup_threshold
        self.cleanup_interval = cleanup_interval
        
        self._weak_references: Set[weakref.ref] = set()
        self._cleanup_callbacks: List[Callable] = []
        self._memory_pools: Dict[str, List[Any]] = defaultdict(list)
        
        self._cleanup_thread = None
        self._stop_cleanup = False
        
        # Start memory monitoring
        self.start_memory_monitoring()
        
    def start_memory_monitoring(self):
        """Start memory monitoring thread."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup = False
            self._cleanup_thread = threading.Thread(
                target=self._memory_monitoring_loop, daemon=True)
            self._cleanup_thread.start()
            logger.info("Started memory monitoring")
            
    def stop_memory_monitoring(self):
        """Stop memory monitoring thread."""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        logger.info("Stopped memory monitoring")
        
    def _memory_monitoring_loop(self):
        """Memory monitoring loop."""
        while not self._stop_cleanup:
            try:
                current_usage = psutil.Process().memory_percent() / 100.0
                
                if current_usage > self.cleanup_threshold:
                    logger.warning(f"High memory usage: {current_usage:.2%}")
                    self.emergency_cleanup()
                    
                # Regular maintenance cleanup
                self.cleanup_weak_references()
                
                time.sleep(self.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)
                
    def register_weak_reference(self, obj: Any) -> weakref.ref:
        """Register object for weak reference tracking."""
        weak_ref = weakref.ref(obj, self._weak_ref_callback)
        self._weak_references.add(weak_ref)
        return weak_ref
        
    def _weak_ref_callback(self, weak_ref):
        """Callback when weak reference is garbage collected."""
        self._weak_references.discard(weak_ref)
        
    def register_cleanup_callback(self, callback: Callable):
        """Register callback for emergency cleanup."""
        self._cleanup_callbacks.append(callback)
        
    def get_object_from_pool(self, pool_name: str, factory: Callable) -> Any:
        """Get object from memory pool or create new one."""
        pool = self._memory_pools[pool_name]
        
        if pool:
            return pool.pop()
        else:
            return factory()
            
    def return_to_pool(self, pool_name: str, obj: Any, reset_func: Optional[Callable] = None):
        """Return object to memory pool for reuse."""
        if reset_func:
            reset_func(obj)
            
        pool = self._memory_pools[pool_name]
        if len(pool) < 100:  # Limit pool size
            pool.append(obj)
            
    def force_garbage_collection(self):
        """Force garbage collection with full collection."""
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        
        # Force collection of all generations
        for generation in range(3):
            collected += gc.collect(generation)
            
        return collected
        
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        logger.warning("Performing emergency memory cleanup")
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Run registered cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
                
        # Clear memory pools
        total_cleared = 0
        for pool_name, pool in self._memory_pools.items():
            cleared = len(pool)
            pool.clear()
            total_cleared += cleared
            
        logger.info(f"Cleared {total_cleared} objects from memory pools")
        
        # Force garbage collection
        collected = self.force_garbage_collection()
        
        # Clean up weak references
        self.cleanup_weak_references()
        
        final_memory = psutil.Process().memory_info().rss
        memory_freed = initial_memory - final_memory
        
        logger.info(f"Emergency cleanup completed. Freed {memory_freed / (1024*1024):.1f} MB")
        
    def cleanup_weak_references(self):
        """Clean up dead weak references."""
        dead_refs = [ref for ref in self._weak_references if ref() is None]
        for ref in dead_refs:
            self._weak_references.discard(ref)
            
        if dead_refs:
            logger.debug(f"Cleaned up {len(dead_refs)} dead weak references")
            
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'memory_percent': process.memory_percent(),
            'memory_rss': memory_info.rss / (1024 * 1024),  # MB
            'memory_vms': memory_info.vms / (1024 * 1024),  # MB
            'memory_pools': {name: len(pool) for name, pool in self._memory_pools.items()},
            'weak_references': len(self._weak_references),
            'gc_counts': gc.get_count(),
            'gc_threshold': gc.get_threshold()
        }