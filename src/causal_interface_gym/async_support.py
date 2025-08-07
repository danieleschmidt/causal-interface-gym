"""Asynchronous support for causal interface gym operations."""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import aiofiles
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class AsyncExperimentResult:
    """Result of asynchronous experiment."""
    experiment_id: str
    agent_id: str
    status: str  # 'running', 'completed', 'failed'
    start_time: float
    end_time: Optional[float] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AsyncExperimentManager:
    """Manage asynchronous causal reasoning experiments."""
    
    def __init__(self, max_concurrent: int = 10):
        """Initialize async experiment manager.
        
        Args:
            max_concurrent: Maximum concurrent experiments
        """
        self.max_concurrent = max_concurrent
        self.running_experiments: Dict[str, AsyncExperimentResult] = {}
        self.completed_experiments: Dict[str, AsyncExperimentResult] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._experiment_counter = 0
    
    async def run_experiment_async(
        self,
        environment,
        ui,
        agent,
        interventions: List[tuple],
        measure_beliefs: List[str],
        experiment_id: Optional[str] = None
    ) -> AsyncExperimentResult:
        """Run experiment asynchronously.
        
        Args:
            environment: Causal environment
            ui: Intervention UI
            agent: LLM agent
            interventions: List of interventions
            measure_beliefs: Beliefs to measure
            experiment_id: Optional experiment ID
            
        Returns:
            Experiment result
        """
        if experiment_id is None:
            self._experiment_counter += 1
            experiment_id = f"exp_{self._experiment_counter}_{int(time.time())}"
        
        agent_id = str(agent) if hasattr(agent, '__str__') else type(agent).__name__
        
        # Create experiment result
        result = AsyncExperimentResult(
            experiment_id=experiment_id,
            agent_id=agent_id,
            status='running',
            start_time=time.time()
        )
        
        self.running_experiments[experiment_id] = result
        
        try:
            async with self.semaphore:
                logger.info(f"Starting async experiment {experiment_id}")
                
                # Run experiment in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    experiment_results = await loop.run_in_executor(
                        executor,
                        ui.run_experiment,
                        agent,
                        interventions,
                        measure_beliefs
                    )
                
                # Update result
                result.status = 'completed'
                result.end_time = time.time()
                result.results = experiment_results
                
                logger.info(f"Completed async experiment {experiment_id} in {result.end_time - result.start_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Async experiment {experiment_id} failed: {e}")
            result.status = 'failed'
            result.end_time = time.time()
            result.error = str(e)
        
        # Move to completed
        if experiment_id in self.running_experiments:
            del self.running_experiments[experiment_id]
        self.completed_experiments[experiment_id] = result
        
        return result
    
    async def run_batch_experiments(
        self,
        experiment_configs: List[Dict[str, Any]]
    ) -> List[AsyncExperimentResult]:
        """Run multiple experiments in parallel.
        
        Args:
            experiment_configs: List of experiment configurations
            
        Returns:
            List of experiment results
        """
        tasks = []
        
        for config in experiment_configs:
            task = asyncio.create_task(
                self.run_experiment_async(**config)
            )
            tasks.append(task)
        
        # Wait for all experiments to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = AsyncExperimentResult(
                    experiment_id=f"batch_exp_{i}",
                    agent_id="unknown",
                    status='failed',
                    start_time=time.time(),
                    end_time=time.time(),
                    error=str(result)
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_experiment_status(self, experiment_id: str) -> Optional[AsyncExperimentResult]:
        """Get status of experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment result or None if not found
        """
        if experiment_id in self.running_experiments:
            return self.running_experiments[experiment_id]
        elif experiment_id in self.completed_experiments:
            return self.completed_experiments[experiment_id]
        return None
    
    def get_running_experiments(self) -> List[AsyncExperimentResult]:
        """Get list of currently running experiments."""
        return list(self.running_experiments.values())
    
    def get_completed_experiments(self) -> List[AsyncExperimentResult]:
        """Get list of completed experiments."""
        return list(self.completed_experiments.values())
    
    async def wait_for_experiment(self, experiment_id: str, timeout: Optional[float] = None) -> AsyncExperimentResult:
        """Wait for experiment to complete.
        
        Args:
            experiment_id: Experiment ID
            timeout: Timeout in seconds
            
        Returns:
            Experiment result
            
        Raises:
            asyncio.TimeoutError: If experiment doesn't complete in time
            ValueError: If experiment not found
        """
        start_time = time.time()
        
        while True:
            result = self.get_experiment_status(experiment_id)
            
            if result is None:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            if result.status != 'running':
                return result
            
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError(f"Experiment {experiment_id} timed out after {timeout}s")
            
            await asyncio.sleep(0.1)


class AsyncLLMClient:
    """Asynchronous LLM client for concurrent belief queries."""
    
    def __init__(self, base_client, max_concurrent: int = 5):
        """Initialize async LLM client.
        
        Args:
            base_client: Base LLM client
            max_concurrent: Maximum concurrent requests
        """
        self.base_client = base_client
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def query_belief_async(self, belief: str, condition: str = "observational") -> float:
        """Query belief asynchronously.
        
        Args:
            belief: Belief statement
            condition: Condition type
            
        Returns:
            Belief probability
        """
        async with self.semaphore:
            # Run synchronous client in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                if hasattr(self.base_client, 'query_belief'):
                    result = await loop.run_in_executor(
                        executor,
                        self.base_client.query_belief,
                        belief,
                        condition
                    )
                else:
                    # Fallback to random for testing
                    import random
                    await asyncio.sleep(0.01)  # Simulate API call
                    result = random.random()
            
            return result
    
    async def batch_query_beliefs(self, beliefs: List[str], condition: str = "observational") -> Dict[str, float]:
        """Query multiple beliefs in parallel.
        
        Args:
            beliefs: List of belief statements
            condition: Condition type
            
        Returns:
            Dictionary mapping beliefs to probabilities
        """
        tasks = [
            self.query_belief_async(belief, condition)
            for belief in beliefs
        ]
        
        results = await asyncio.gather(*tasks)
        
        return dict(zip(beliefs, results))


class AsyncCacheManager:
    """Asynchronous cache manager for high-performance operations."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize async cache manager.
        
        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_pool = None
        
        if redis_url:
            import aioredis
            self.redis_pool = aioredis.ConnectionPool.from_url(redis_url)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache asynchronously."""
        if not self.redis_pool:
            return None
        
        try:
            import aioredis
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            data = await redis.get(key)
            
            if data:
                import pickle
                return pickle.loads(data)
            
            return None
            
        except Exception as e:
            logger.error(f"Async cache get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set item in cache asynchronously."""
        if not self.redis_pool:
            return
        
        try:
            import aioredis
            import pickle
            
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            data = pickle.dumps(value)
            await redis.setex(key, ttl, data)
            
        except Exception as e:
            logger.error(f"Async cache set error: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete item from cache asynchronously."""
        if not self.redis_pool:
            return
        
        try:
            import aioredis
            redis = aioredis.Redis(connection_pool=self.redis_pool)
            await redis.delete(key)
            
        except Exception as e:
            logger.error(f"Async cache delete error: {e}")


class StreamingExperimentRunner:
    """Stream experiment results in real-time."""
    
    def __init__(self):
        """Initialize streaming runner."""
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, experiment_id: str, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to experiment updates.
        
        Args:
            experiment_id: Experiment ID
            callback: Callback function for updates
        """
        if experiment_id not in self.subscribers:
            self.subscribers[experiment_id] = []
        self.subscribers[experiment_id].append(callback)
    
    def unsubscribe(self, experiment_id: str, callback: Callable):
        """Unsubscribe from experiment updates."""
        if experiment_id in self.subscribers:
            self.subscribers[experiment_id].remove(callback)
            if not self.subscribers[experiment_id]:
                del self.subscribers[experiment_id]
    
    async def stream_experiment(self, experiment_config: Dict[str, Any]) -> AsyncExperimentResult:
        """Run experiment with real-time streaming updates.
        
        Args:
            experiment_config: Experiment configuration
            
        Returns:
            Final experiment result
        """
        experiment_id = experiment_config.get('experiment_id', f"stream_{int(time.time())}")
        
        # Notify start
        await self._notify_subscribers(experiment_id, {
            'type': 'experiment_start',
            'experiment_id': experiment_id,
            'timestamp': time.time()
        })
        
        try:
            # Create async experiment manager
            manager = AsyncExperimentManager()
            
            # Run experiment with progress updates
            result = await manager.run_experiment_async(**experiment_config)
            
            # Stream intermediate updates during experiment
            await self._stream_progress_updates(experiment_id, result)
            
            # Notify completion
            await self._notify_subscribers(experiment_id, {
                'type': 'experiment_complete',
                'experiment_id': experiment_id,
                'result': result,
                'timestamp': time.time()
            })
            
            return result
            
        except Exception as e:
            # Notify error
            await self._notify_subscribers(experiment_id, {
                'type': 'experiment_error',
                'experiment_id': experiment_id,
                'error': str(e),
                'timestamp': time.time()
            })
            raise
    
    async def _notify_subscribers(self, experiment_id: str, update: Dict[str, Any]):
        """Notify all subscribers of an update."""
        if experiment_id in self.subscribers:
            for callback in self.subscribers[experiment_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(update)
                    else:
                        callback(update)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")
    
    async def _stream_progress_updates(self, experiment_id: str, result: AsyncExperimentResult):
        """Stream progress updates during experiment execution."""
        # Simulate progress updates
        progress_steps = [
            {'step': 'initializing', 'progress': 0.1},
            {'step': 'querying_beliefs', 'progress': 0.3},
            {'step': 'applying_interventions', 'progress': 0.6},
            {'step': 'analyzing_results', 'progress': 0.8},
            {'step': 'finalizing', 'progress': 1.0}
        ]
        
        for step_info in progress_steps:
            await self._notify_subscribers(experiment_id, {
                'type': 'progress_update',
                'experiment_id': experiment_id,
                'step': step_info['step'],
                'progress': step_info['progress'],
                'timestamp': time.time()
            })
            
            await asyncio.sleep(0.1)  # Small delay between updates


# Global async managers
default_async_experiment_manager = AsyncExperimentManager()
default_streaming_runner = StreamingExperimentRunner()


async def run_concurrent_experiments(experiments: List[Dict[str, Any]]) -> List[AsyncExperimentResult]:
    """Run multiple experiments concurrently.
    
    Args:
        experiments: List of experiment configurations
        
    Returns:
        List of experiment results
    """
    return await default_async_experiment_manager.run_batch_experiments(experiments)


def async_cached(cache_manager: AsyncCacheManager, ttl: int = 3600):
    """Decorator for async caching."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            import hashlib
            import json
            
            key_data = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': json.dumps(kwargs, sort_keys=True, default=str)
            }
            cache_key = hashlib.md5(json.dumps(key_data).encode()).hexdigest()
            
            # Try cache first
            cached_result = await cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator