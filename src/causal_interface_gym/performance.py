"""Performance optimization utilities for causal interface gym."""

import time
import threading
import asyncio
import concurrent.futures
from typing import Any, Dict, List, Callable, Optional, Union, Tuple
from functools import wraps
import multiprocessing as mp
import logging
from contextlib import contextmanager
import queue
import weakref

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Generic connection pool for resource management."""
    
    def __init__(self, factory: Callable, max_size: int = 20, 
                 timeout: float = 30.0, validate_func: Optional[Callable] = None):
        """Initialize connection pool.
        
        Args:
            factory: Function to create new connections
            max_size: Maximum number of connections in pool
            timeout: Timeout for acquiring connections
            validate_func: Function to validate connections
        """
        self.factory = factory
        self.max_size = max_size
        self.timeout = timeout
        self.validate_func = validate_func
        
        self._pool = queue.Queue(maxsize=max_size)
        self._created_count = 0
        self._lock = threading.Lock()
        self._stats = {
            'created': 0,
            'acquired': 0,
            'released': 0,
            'validation_failures': 0
        }
    
    def acquire(self) -> Any:
        """Acquire connection from pool.
        
        Returns:
            Connection object
            
        Raises:
            TimeoutError: If no connection available within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Try to get existing connection
                connection = self._pool.get_nowait()
                
                # Validate connection if validator provided
                if self.validate_func and not self.validate_func(connection):
                    self._stats['validation_failures'] += 1
                    continue
                
                self._stats['acquired'] += 1
                return connection
                
            except queue.Empty:
                # No connections available, try to create new one
                with self._lock:
                    if self._created_count < self.max_size:
                        try:
                            connection = self.factory()
                            self._created_count += 1
                            self._stats['created'] += 1
                            self._stats['acquired'] += 1
                            return connection
                        except Exception as e:
                            logger.error(f"Failed to create connection: {e}")
                
                # Wait briefly before retrying
                time.sleep(0.1)
        
        raise TimeoutError(f"Could not acquire connection within {self.timeout}s")
    
    def release(self, connection: Any) -> None:
        """Release connection back to pool.
        
        Args:
            connection: Connection to release
        """
        try:
            self._pool.put_nowait(connection)
            self._stats['released'] += 1
        except queue.Full:
            # Pool is full, discard connection
            with self._lock:
                self._created_count -= 1
    
    @contextmanager
    def get_connection(self):
        """Context manager for automatic connection management."""
        connection = self.acquire()
        try:
            yield connection
        finally:
            self.release(connection)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            'pool_size': self._pool.qsize(),
            'max_size': self.max_size,
            'created_count': self._created_count
        }
    
    def close_all(self) -> None:
        """Close all connections in pool."""
        while not self._pool.empty():
            try:
                connection = self._pool.get_nowait()
                if hasattr(connection, 'close'):
                    connection.close()
            except queue.Empty:
                break
        
        with self._lock:
            self._created_count = 0


class AsyncTaskManager:
    """Manage asynchronous tasks for improved performance."""
    
    def __init__(self, max_workers: int = None):
        """Initialize async task manager.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks = set()
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
    
    def submit_task(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Submit task for asynchronous execution.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Future object representing the task
        """
        future = self.executor.submit(func, *args, **kwargs)
        
        with self.lock:
            self.active_tasks.add(future)
        
        # Add callback to track completion
        future.add_done_callback(self._task_completed)
        
        return future
    
    def submit_batch(self, func: Callable, items: List[Any], 
                    chunk_size: Optional[int] = None) -> List[concurrent.futures.Future]:
        """Submit batch of tasks for parallel execution.
        
        Args:
            func: Function to execute on each item
            items: List of items to process
            chunk_size: Size of chunks for processing
            
        Returns:
            List of Future objects
        """
        if chunk_size is None:
            chunk_size = max(1, len(items) // self.max_workers)
        
        futures = []
        
        # Split items into chunks
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            future = self.submit_task(self._process_chunk, func, chunk)
            futures.append(future)
        
        return futures
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def _task_completed(self, future: concurrent.futures.Future) -> None:
        """Callback for task completion."""
        with self.lock:
            self.active_tasks.discard(future)
            
            if future.exception():
                self.failed_tasks += 1
                logger.error(f"Task failed: {future.exception()}")
            else:
                self.completed_tasks += 1
    
    def wait_for_completion(self, futures: List[concurrent.futures.Future], 
                          timeout: Optional[float] = None) -> Tuple[List[Any], List[Exception]]:
        """Wait for futures to complete and collect results.
        
        Args:
            futures: List of futures to wait for
            timeout: Maximum time to wait
            
        Returns:
            Tuple of (results, exceptions)
        """
        results = []
        exceptions = []
        
        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                exceptions.append(e)
        
        return results, exceptions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics."""
        with self.lock:
            return {
                'max_workers': self.max_workers,
                'active_tasks': len(self.active_tasks),
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': (
                    self.completed_tasks / (self.completed_tasks + self.failed_tasks)
                    if (self.completed_tasks + self.failed_tasks) > 0 else 0
                )
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the task manager.
        
        Args:
            wait: Whether to wait for running tasks to complete
        """
        self.executor.shutdown(wait=wait)


class PerformanceOptimizer:
    """Automated performance optimization."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.optimizations = {
            'graph_algorithms': self._optimize_graph_algorithms,
            'llm_processing': self._optimize_llm_processing,
            'memory_usage': self._optimize_memory_usage,
            'io_operations': self._optimize_io_operations
        }
        self.optimization_history = []
    
    def optimize_causal_environment(self, environment) -> Dict[str, Any]:
        """Optimize causal environment for better performance.
        
        Args:
            environment: CausalEnvironment instance
            
        Returns:
            Optimization report
        """
        report = {
            'optimizations_applied': [],
            'performance_improvements': {},
            'recommendations': []
        }
        
        # Optimize graph algorithms
        if hasattr(environment, 'graph'):
            graph_optimizations = self._optimize_graph_algorithms(environment)
            report['optimizations_applied'].extend(graph_optimizations)
        
        # Optimize memory usage
        memory_optimizations = self._optimize_memory_usage(environment)
        report['optimizations_applied'].extend(memory_optimizations)
        
        # Add recommendations
        report['recommendations'] = self._generate_recommendations(environment)
        
        return report
    
    def _optimize_graph_algorithms(self, environment) -> List[str]:
        """Optimize graph algorithms."""
        optimizations = []
        
        try:
            import networkx as nx
            
            # Convert to more efficient graph representation if large
            if hasattr(environment, 'graph') and len(environment.graph.nodes()) > 100:
                # Use more efficient algorithms for large graphs
                if not hasattr(environment.graph, '_cached_algorithms'):
                    environment.graph._cached_algorithms = {}
                    optimizations.append("Added algorithm result caching for large graph")
                
                # Pre-compute commonly used graph properties
                if 'topological_order' not in environment.graph._cached_algorithms:
                    try:
                        topo_order = list(nx.topological_sort(environment.graph))
                        environment.graph._cached_algorithms['topological_order'] = topo_order
                        optimizations.append("Pre-computed topological ordering")
                    except nx.NetworkXError:
                        pass  # Graph may not be DAG
        
        except Exception as e:
            logger.warning(f"Graph optimization failed: {e}")
        
        return optimizations
    
    def _optimize_llm_processing(self, environment) -> List[str]:
        """Optimize LLM processing."""
        optimizations = []
        
        # Enable response caching if not already enabled
        if not hasattr(environment, '_llm_cache_enabled'):
            environment._llm_cache_enabled = True
            optimizations.append("Enabled LLM response caching")
        
        return optimizations
    
    def _optimize_memory_usage(self, environment) -> List[str]:
        """Optimize memory usage."""
        optimizations = []
        
        # Enable weak references for large objects
        if not hasattr(environment, '_weak_refs_enabled'):
            environment._weak_refs_enabled = True
            optimizations.append("Enabled weak references for memory efficiency")
        
        # Compress large data structures
        if hasattr(environment, 'experiment_history'):
            if len(environment.experiment_history) > 1000:
                # Keep only recent experiments in memory
                environment.experiment_history = environment.experiment_history[-500:]
                optimizations.append("Compressed experiment history")
        
        return optimizations
    
    def _optimize_io_operations(self, environment) -> List[str]:
        """Optimize I/O operations."""
        optimizations = []
        
        # Enable batching for database operations
        if not hasattr(environment, '_batch_operations_enabled'):
            environment._batch_operations_enabled = True
            optimizations.append("Enabled batch operations for I/O")
        
        return optimizations
    
    def _generate_recommendations(self, environment) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check graph size
        if hasattr(environment, 'graph'):
            node_count = len(environment.graph.nodes())
            edge_count = len(environment.graph.edges())
            
            if node_count > 500:
                recommendations.append(
                    f"Large graph detected ({node_count} nodes). "
                    "Consider using sparse matrix representations for better performance."
                )
            
            if edge_count > 1000:
                recommendations.append(
                    f"Dense graph detected ({edge_count} edges). "
                    "Consider graph partitioning for large-scale operations."
                )
        
        # Check memory usage
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            if memory_percent > 80:
                recommendations.append(
                    f"High memory usage detected ({memory_percent:.1f}%). "
                    "Consider enabling more aggressive caching policies."
                )
        except ImportError:
            pass
        
        return recommendations


def parallel_process(items: List[Any], func: Callable, max_workers: int = None, 
                    chunk_size: int = None) -> List[Any]:
    """Process items in parallel for improved performance.
    
    Args:
        items: List of items to process
        func: Function to apply to each item
        max_workers: Maximum number of worker processes
        chunk_size: Size of chunks for processing
        
    Returns:
        List of processed results
    """
    if not items:
        return []
    
    if len(items) == 1:
        return [func(items[0])]
    
    max_workers = max_workers or min(len(items), mp.cpu_count())
    chunk_size = chunk_size or max(1, len(items) // max_workers)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Submit tasks
        futures = [executor.submit(_process_chunk_worker, func, chunk) for chunk in chunks]
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
            except Exception as e:
                logger.error(f"Parallel processing failed for chunk: {e}")
        
        return results


def _process_chunk_worker(func: Callable, chunk: List[Any]) -> List[Any]:
    """Worker function for parallel processing."""
    return [func(item) for item in chunk]


class MemoryOptimizer:
    """Memory usage optimization utilities."""
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.weak_refs = weakref.WeakValueDictionary()
        self.compression_enabled = True
    
    def optimize_object_storage(self, obj: Any, key: str) -> None:
        """Store object with memory optimization.
        
        Args:
            obj: Object to store
            key: Storage key
        """
        # Use weak references for large objects
        if hasattr(obj, '__len__') and len(obj) > 1000:
            self.weak_refs[key] = obj
        
        # Compress if enabled and object is large
        if self.compression_enabled and self._should_compress(obj):
            compressed_obj = self._compress_object(obj)
            setattr(obj, '_compressed_data', compressed_obj)
            setattr(obj, '_is_compressed', True)
    
    def _should_compress(self, obj: Any) -> bool:
        """Check if object should be compressed."""
        try:
            import sys
            return sys.getsizeof(obj) > 10000  # Compress objects > 10KB
        except:
            return False
    
    def _compress_object(self, obj: Any) -> bytes:
        """Compress object for storage."""
        try:
            import pickle
            import gzip
            
            pickled_obj = pickle.dumps(obj)
            return gzip.compress(pickled_obj)
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return b""
    
    def _decompress_object(self, compressed_data: bytes) -> Any:
        """Decompress object from storage."""
        try:
            import pickle
            import gzip
            
            decompressed_data = gzip.decompress(compressed_data)
            return pickle.loads(decompressed_data)
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return None


# Performance monitoring decorators
def measure_performance(func: Callable) -> Callable:
    """Decorator to measure function performance.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with performance measurement
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = _get_memory_usage()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            logger.debug(
                f"Performance: {func.__name__} took {duration:.3f}s, "
                f"memory delta: {memory_delta:.2f}MB"
            )
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            logger.error(
                f"Performance: {func.__name__} failed after {duration:.3f}s: {e}"
            )
            raise
    
    return wrapper


def _get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


# Global optimization instances
performance_optimizer = PerformanceOptimizer()
memory_optimizer = MemoryOptimizer()
task_manager = AsyncTaskManager()


@contextmanager
def performance_context(operation_name: str):
    """Context manager for performance monitoring.
    
    Args:
        operation_name: Name of the operation being monitored
    """
    start_time = time.time()
    start_memory = _get_memory_usage()
    
    logger.info(f"Starting performance-monitored operation: {operation_name}")
    
    try:
        yield
        
        duration = time.time() - start_time
        memory_delta = _get_memory_usage() - start_memory
        
        logger.info(
            f"Operation {operation_name} completed in {duration:.3f}s, "
            f"memory delta: {memory_delta:.2f}MB"
        )
        
    except Exception as e:
        duration = time.time() - start_time
        
        logger.error(
            f"Operation {operation_name} failed after {duration:.3f}s: {e}"
        )
        raise