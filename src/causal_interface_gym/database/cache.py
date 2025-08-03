"""Caching layer for causal interface gym."""

import os
import json
import hashlib
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for expensive computations and API calls."""
    
    def __init__(self, redis_url: Optional[str] = None, ttl: int = 3600):
        """Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL
            ttl: Default time to live in seconds
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.ttl = ttl
        self.redis_client = None
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        if self.redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache")
                self.redis_client = None
        else:
            logger.info("Using in-memory cache (Redis not available)")
    
    def _get_cache_key(self, key: str, prefix: str = "causal_gym") -> str:
        """Generate cache key with prefix.
        
        Args:
            key: Base key
            prefix: Key prefix
            
        Returns:
            Full cache key
        """
        return f"{prefix}:{key}"
    
    def _hash_complex_key(self, key_data: Any) -> str:
        """Create hash for complex key data.
        
        Args:
            key_data: Complex data to hash
            
        Returns:
            Hash string
        """
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        cache_key = self._get_cache_key(key)
        
        if self.redis_client:
            try:
                value = self.redis_client.get(cache_key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Fallback to memory cache
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if datetime.now() <= entry['expires_at']:
                return entry['value']
            else:
                # Expired
                del self.memory_cache[cache_key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        cache_key = self._get_cache_key(key)
        ttl = ttl or self.ttl
        
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(value, default=str)
                )
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Fallback to memory cache
        self.memory_cache[cache_key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
    
    def delete(self, key: str) -> None:
        """Delete value from cache.
        
        Args:
            key: Cache key
        """
        cache_key = self._get_cache_key(key)
        
        if self.redis_client:
            try:
                self.redis_client.delete(cache_key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Also remove from memory cache
        self.memory_cache.pop(cache_key, None)
    
    def cache_computation(self, key_data: Any, computation_func, ttl: Optional[int] = None) -> Any:
        """Cache the result of an expensive computation.
        
        Args:
            key_data: Data to use for cache key generation
            computation_func: Function to compute result if not cached
            ttl: Time to live for cache entry
            
        Returns:
            Computation result (cached or fresh)
        """
        cache_key = self._hash_complex_key(key_data)
        
        # Try to get from cache first
        cached_result = self.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for key: {cache_key}")
            return cached_result
        
        # Compute and cache result
        logger.debug(f"Cache miss for key: {cache_key}, computing...")
        result = computation_func()
        self.set(cache_key, result, ttl)
        
        return result
    
    def cache_causal_computation(self, graph: Dict[str, Any], treatment: str, 
                                outcome: str, computation_func) -> Any:
        """Cache causal computation results.
        
        Args:
            graph: Causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            computation_func: Function to compute causal effect
            
        Returns:
            Causal computation result
        """
        key_data = {
            'type': 'causal_computation',
            'graph': graph,
            'treatment': treatment,
            'outcome': outcome
        }
        
        return self.cache_computation(
            key_data, 
            computation_func, 
            ttl=7200  # 2 hours for causal computations
        )
    
    def cache_llm_response(self, prompt: str, model: str, response_func) -> Any:
        """Cache LLM API responses.
        
        Args:
            prompt: Input prompt
            model: Model name
            response_func: Function to get LLM response
            
        Returns:
            LLM response
        """
        key_data = {
            'type': 'llm_response',
            'prompt': prompt,
            'model': model
        }
        
        return self.cache_computation(
            key_data,
            response_func,
            ttl=86400  # 24 hours for LLM responses
        )
    
    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cache entries.
        
        Args:
            pattern: Pattern to match keys (None for all)
        """
        if self.redis_client:
            try:
                if pattern:
                    keys = self.redis_client.keys(f"causal_gym:{pattern}*")
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    # Clear all causal_gym keys
                    keys = self.redis_client.keys("causal_gym:*")
                    if keys:
                        self.redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        # Clear memory cache
        if pattern:
            to_remove = [
                key for key in self.memory_cache.keys()
                if pattern in key
            ]
            for key in to_remove:
                del self.memory_cache[key]
        else:
            self.memory_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        stats = {
            'backend': 'redis' if self.redis_client else 'memory',
            'memory_cache_size': len(self.memory_cache)
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_connected': True,
                    'redis_memory_used': info.get('used_memory_human', 'unknown'),
                    'redis_keys': info.get('db0', {}).get('keys', 0) if 'db0' in info else 0
                })
            except Exception:
                stats['redis_connected'] = False
        
        return stats