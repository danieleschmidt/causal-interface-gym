"""Caching and performance optimization for causal interface gym."""

import time
import hashlib
import json
import threading
from typing import Any, Dict, Optional, Callable, Union, List
from functools import wraps
from collections import OrderedDict
import pickle
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.stats['hits'] += 1
                return value
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing item
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
                self.stats['evictions'] += 1
            
            self.cache[key] = value
            self.stats['size'] = len(self.cache)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats['size'] = len(self.cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class TTLCache:
    """Time-To-Live cache with automatic expiration."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize TTL cache.
        
        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.expiry_times = {}
        self.lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'expirations': 0,
            'size': 0
        }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        return key in self.expiry_times and time.time() > self.expiry_times[key]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.expiry_times.items()
            if current_time > expiry_time
        ]
        
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
                del self.expiry_times[key]
                self.stats['expirations'] += 1
        
        self.stats['size'] = len(self.cache)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self.lock:
            self._cleanup_expired()
            
            if key in self.cache and not self._is_expired(key):
                self.stats['hits'] += 1
                return self.cache[key]
            else:
                self.stats['misses'] += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self.lock:
            # Clean up expired entries
            self._cleanup_expired()
            
            # Evict oldest entries if at capacity
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.expiry_times:
                    del self.expiry_times[oldest_key]
            
            # Add new entry
            self.cache[key] = value
            expire_time = time.time() + (ttl or self.default_ttl)
            self.expiry_times[key] = expire_time
            self.stats['size'] = len(self.cache)
    
    def delete(self, key: str) -> bool:
        """Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.expiry_times:
                    del self.expiry_times[key]
                self.stats['size'] = len(self.cache)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self.lock:
            self.cache.clear()
            self.expiry_times.clear()
            self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class CacheManager:
    """Centralized cache management with multiple cache types."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.caches = {
            'llm_responses': TTLCache(max_size=10000, default_ttl=3600),  # 1 hour
            'graph_computations': LRUCache(max_size=5000),
            'belief_queries': TTLCache(max_size=50000, default_ttl=1800),  # 30 minutes
            'experiment_results': TTLCache(max_size=1000, default_ttl=7200),  # 2 hours
            'user_sessions': TTLCache(max_size=10000, default_ttl=86400),  # 24 hours
        }
        self.lock = threading.RLock()
    
    def get_cache(self, cache_name: str) -> Union[LRUCache, TTLCache]:
        """Get cache by name.
        
        Args:
            cache_name: Name of the cache
            
        Returns:
            Cache instance
        """
        return self.caches.get(cache_name)
    
    def cache_llm_response(self, prompt: str, model: str, 
                          response_generator: Callable) -> Any:
        """Cache LLM response with automatic key generation.
        
        Args:
            prompt: LLM prompt
            model: Model name
            response_generator: Function that generates the response
            
        Returns:
            Cached or generated response
        """
        cache_key = self._generate_cache_key('llm', prompt, model)
        cache = self.caches['llm_responses']
        
        # Try to get from cache first
        cached_response = cache.get(cache_key)
        if cached_response is not None:
            logger.debug(f"Cache hit for LLM response: {cache_key[:16]}...")
            return cached_response
        
        # Generate new response
        logger.debug(f"Cache miss for LLM response: {cache_key[:16]}...")
        response = response_generator()
        
        # Cache the response
        cache.put(cache_key, response, ttl=3600)  # 1 hour TTL
        
        return response
    
    def cache_graph_computation(self, graph_id: str, operation: str,
                              computation_func: Callable) -> Any:
        """Cache graph computation results.
        
        Args:
            graph_id: Unique identifier for the graph
            operation: Type of operation (e.g., 'backdoor_paths')
            computation_func: Function that performs the computation
            
        Returns:
            Cached or computed result
        """
        cache_key = self._generate_cache_key('graph', graph_id, operation)
        cache = self.caches['graph_computations']
        
        # Try cache first
        cached_result = cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for graph computation: {operation}")
            return cached_result
        
        # Compute new result
        logger.debug(f"Cache miss for graph computation: {operation}")
        result = computation_func()
        
        # Cache the result
        cache.put(cache_key, result)
        
        return result
    
    def cache_belief_query(self, belief_statement: str, condition: str,
                          query_func: Callable) -> float:
        """Cache belief query results.
        
        Args:
            belief_statement: The belief statement
            condition: Query condition
            query_func: Function that executes the query
            
        Returns:
            Cached or computed belief probability
        """
        cache_key = self._generate_cache_key('belief', belief_statement, condition)
        cache = self.caches['belief_queries']
        
        # Try cache first
        cached_belief = cache.get(cache_key)
        if cached_belief is not None:
            logger.debug(f"Cache hit for belief query: {belief_statement[:30]}...")
            return cached_belief
        
        # Execute new query
        logger.debug(f"Cache miss for belief query: {belief_statement[:30]}...")
        belief_prob = query_func()
        
        # Cache the result
        cache.put(cache_key, belief_prob, ttl=1800)  # 30 minutes
        
        return belief_prob
    
    def _generate_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments.
        
        Args:
            prefix: Cache key prefix
            *args: Arguments to include in key
            
        Returns:
            SHA-256 hash of serialized arguments
        """
        try:
            # Create deterministic string representation
            key_data = f"{prefix}:" + ":".join(str(arg) for arg in args)
            
            # Generate hash
            return hashlib.sha256(key_data.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            # Fallback to simple concatenation
            return f"{prefix}:{'_'.join(str(arg)[:50] for arg in args)}"
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches.
        
        Returns:
            Dictionary mapping cache names to their statistics
        """
        with self.lock:
            return {
                name: cache.get_stats()
                for name, cache in self.caches.items()
            }
    
    def clear_all_caches(self) -> None:
        """Clear all caches."""
        with self.lock:
            for cache in self.caches.values():
                cache.clear()
            logger.info("All caches cleared")
    
    def optimize_caches(self) -> Dict[str, Any]:
        """Optimize cache performance and return optimization report.
        
        Returns:
            Optimization report
        """
        report = {
            'optimizations_applied': [],
            'cache_stats': self.get_all_stats(),
            'recommendations': []
        }
        
        for name, cache in self.caches.items():
            stats = cache.get_stats()
            
            # Check hit rate and recommend size adjustments
            if stats.get('hit_rate', 0) < 0.5 and stats.get('total_requests', 0) > 100:
                report['recommendations'].append(
                    f"Consider increasing size of {name} cache (low hit rate: {stats['hit_rate']:.2%})"
                )
            
            # Clean up TTL caches
            if isinstance(cache, TTLCache):
                old_size = stats['size']
                cache._cleanup_expired()
                new_size = len(cache.cache)
                
                if old_size > new_size:
                    expired_count = old_size - new_size
                    report['optimizations_applied'].append(
                        f"Cleaned {expired_count} expired entries from {name} cache"
                    )
        
        return report


# Global cache manager instance
cache_manager = CacheManager()


def cached_method(cache_name: str, ttl: Optional[int] = None):
    """Decorator to cache method results.
    
    Args:
        cache_name: Name of the cache to use
        ttl: Time-to-live for TTL caches
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()
            
            # Get appropriate cache
            cache = cache_manager.get_cache(cache_name)
            if not cache:
                logger.warning(f"Cache {cache_name} not found, executing without cache")
                return func(*args, **kwargs)
            
            # Try cache first
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if isinstance(cache, TTLCache):
                cache.put(cache_key, result, ttl)
            else:
                cache.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


def warm_cache(cache_name: str, warm_func: Callable, *args, **kwargs) -> None:
    """Warm up cache with pre-computed values.
    
    Args:
        cache_name: Name of the cache to warm
        warm_func: Function to generate cache values
        *args: Arguments for warm function
        **kwargs: Keyword arguments for warm function
    """
    try:
        logger.info(f"Warming up {cache_name} cache...")
        warm_func(*args, **kwargs)
        logger.info(f"Cache {cache_name} warmed successfully")
    except Exception as e:
        logger.error(f"Failed to warm cache {cache_name}: {e}")


class BatchProcessor:
    """Batch processing for improved performance."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 5.0):
        """Initialize batch processor.
        
        Args:
            batch_size: Maximum batch size before auto-flush
            flush_interval: Maximum time between flushes in seconds
        """
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.batch = []
        self.last_flush = time.time()
        self.lock = threading.Lock()
        self.processor_func = None
    
    def set_processor(self, func: Callable[[List], None]) -> None:
        """Set the batch processing function.
        
        Args:
            func: Function that processes a batch of items
        """
        self.processor_func = func
    
    def add_item(self, item: Any) -> None:
        """Add item to batch.
        
        Args:
            item: Item to add to batch
        """
        with self.lock:
            self.batch.append(item)
            
            # Check if we should flush
            should_flush = (
                len(self.batch) >= self.batch_size or
                time.time() - self.last_flush > self.flush_interval
            )
            
            if should_flush:
                self._flush()
    
    def _flush(self) -> None:
        """Flush current batch."""
        if not self.batch or not self.processor_func:
            return
        
        try:
            logger.debug(f"Flushing batch of {len(self.batch)} items")
            self.processor_func(self.batch.copy())
            self.batch.clear()
            self.last_flush = time.time()
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Don't clear batch on failure - retry later
    
    def force_flush(self) -> None:
        """Force flush current batch."""
        with self.lock:
            self._flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            return {
                'current_batch_size': len(self.batch),
                'time_since_last_flush': time.time() - self.last_flush,
                'configured_batch_size': self.batch_size,
                'configured_flush_interval': self.flush_interval
            }