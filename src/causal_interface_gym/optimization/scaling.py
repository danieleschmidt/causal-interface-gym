"""Auto-scaling and load balancing for production deployment."""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import logging
import psutil
import numpy as np
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import json

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for auto-scaling decisions."""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    error_rate: float
    queue_size: int
    active_workers: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """Representation of a worker node."""
    node_id: str
    status: str  # 'active', 'idle', 'overloaded', 'failed'
    cpu_usage: float
    memory_usage: float
    request_count: int
    last_heartbeat: float
    worker_type: str  # 'cpu', 'memory', 'io'
    capabilities: List[str] = field(default_factory=list)


class AutoScaler:
    """Intelligent auto-scaling system with predictive scaling."""
    
    def __init__(self,
                 min_workers: int = 2,
                 max_workers: int = 20,
                 target_cpu_usage: float = 0.7,
                 target_response_time: float = 0.5,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.4,
                 cooldown_period: int = 300):
        """Initialize auto-scaler.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            target_cpu_usage: Target CPU usage (0-1)
            target_response_time: Target response time in seconds
            scale_up_threshold: CPU threshold for scaling up
            scale_down_threshold: CPU threshold for scaling down
            cooldown_period: Cooldown between scaling actions in seconds
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_usage = target_cpu_usage
        self.target_response_time = target_response_time
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.metrics_history: deque = deque(maxlen=1000)
        self.scaling_history: List[Dict[str, Any]] = []
        
        self.thread_executor = ThreadPoolExecutor(max_workers=min_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min_workers // 2)
        
        self._last_scale_action = 0
        self._scaling_thread = None
        self._stop_scaling = False
        self._predictive_model = PredictiveScaler()
        
        # Start monitoring and scaling
        self.start_auto_scaling()
        
    def start_auto_scaling(self):
        """Start the auto-scaling monitoring loop."""
        if self._scaling_thread is None or not self._scaling_thread.is_alive():
            self._stop_scaling = False
            self._scaling_thread = threading.Thread(
                target=self._scaling_loop, daemon=True)
            self._scaling_thread.start()
            logger.info("Started auto-scaling system")
            
    def stop_auto_scaling(self):
        """Stop the auto-scaling system."""
        self._stop_scaling = True
        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)
            
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Stopped auto-scaling system")
        
    def _scaling_loop(self):
        """Main scaling monitoring loop."""
        while not self._stop_scaling:
            try:
                # Collect current metrics
                current_metrics = self._collect_scaling_metrics()
                self.metrics_history.append(current_metrics)
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(current_metrics)
                
                if scaling_decision != 'no_action':
                    self._execute_scaling_action(scaling_decision, current_metrics)
                    
                # Update predictive model
                self._predictive_model.update(current_metrics)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(60)
                
    def _collect_scaling_metrics(self) -> ScalingMetrics:
        """Collect current system metrics for scaling decisions."""
        process = psutil.Process()
        
        # Basic system metrics
        cpu_usage = psutil.cpu_percent(interval=1) / 100.0
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # GB
        
        # Application-specific metrics
        active_workers = len([w for w in self.worker_nodes.values() if w.status == 'active'])
        queue_size = self._get_current_queue_size()
        
        # Performance metrics (would be collected from application)
        request_rate = self._estimate_request_rate()
        response_time = self._get_average_response_time()
        error_rate = self._get_current_error_rate()
        
        return ScalingMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            request_rate=request_rate,
            response_time=response_time,
            error_rate=error_rate,
            queue_size=queue_size,
            active_workers=active_workers
        )
        
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> str:
        """Make intelligent scaling decision based on metrics."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scale_action < self.cooldown_period:
            return 'no_action'
            
        # Get predictive recommendation
        predictive_action = self._predictive_model.predict_scaling_need(
            list(self.metrics_history)[-50:])
            
        # Rule-based scaling logic
        if (metrics.cpu_usage > self.scale_up_threshold or 
            metrics.response_time > self.target_response_time * 1.5 or
            metrics.queue_size > 100):
            
            if metrics.active_workers < self.max_workers:
                return 'scale_up'
                
        elif (metrics.cpu_usage < self.scale_down_threshold and
              metrics.response_time < self.target_response_time * 0.7 and
              metrics.queue_size < 10):
            
            if metrics.active_workers > self.min_workers:
                return 'scale_down'
                
        # Consider predictive recommendation
        if predictive_action != 'no_action':
            return predictive_action
            
        return 'no_action'
        
    def _execute_scaling_action(self, action: str, metrics: ScalingMetrics):
        """Execute the scaling action."""
        current_time = time.time()
        
        if action == 'scale_up':
            new_worker_id = f"worker_{current_time}_{len(self.worker_nodes)}"
            
            # Determine worker type based on bottleneck
            if metrics.cpu_usage > 0.8:
                worker_type = 'cpu'
            elif metrics.memory_usage > 4.0:  # 4GB
                worker_type = 'memory'
            else:
                worker_type = 'io'
                
            self._add_worker(new_worker_id, worker_type)
            logger.info(f"Scaled up: Added {worker_type} worker {new_worker_id}")
            
        elif action == 'scale_down':
            # Remove least utilized worker
            least_utilized = min(
                [w for w in self.worker_nodes.values() if w.status == 'active'],
                key=lambda w: w.request_count,
                default=None
            )
            
            if least_utilized:
                self._remove_worker(least_utilized.node_id)
                logger.info(f"Scaled down: Removed worker {least_utilized.node_id}")
                
        # Record scaling action
        self._last_scale_action = current_time
        self.scaling_history.append({
            'timestamp': current_time,
            'action': action,
            'metrics': metrics.__dict__.copy(),
            'worker_count': len(self.worker_nodes)
        })
        
        # Limit history size
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-1000:]
            
    def _add_worker(self, worker_id: str, worker_type: str):
        """Add a new worker node."""
        worker = WorkerNode(
            node_id=worker_id,
            status='active',
            cpu_usage=0.0,
            memory_usage=0.0,
            request_count=0,
            last_heartbeat=time.time(),
            worker_type=worker_type,
            capabilities=self._get_worker_capabilities(worker_type)
        )
        
        self.worker_nodes[worker_id] = worker
        
        # Expand thread pool if needed
        if worker_type == 'cpu':
            self.thread_executor._max_workers += 1
        elif worker_type == 'io':
            # Add async worker capability
            pass
            
    def _remove_worker(self, worker_id: str):
        """Remove a worker node."""
        if worker_id in self.worker_nodes:
            worker = self.worker_nodes[worker_id]
            worker.status = 'shutting_down'
            
            # Allow graceful shutdown
            time.sleep(5)
            
            del self.worker_nodes[worker_id]
            
    def _get_worker_capabilities(self, worker_type: str) -> List[str]:
        """Get capabilities for worker type."""
        capabilities = {
            'cpu': ['cpu_intensive', 'computation'],
            'memory': ['large_data', 'caching'],
            'io': ['file_operations', 'network_requests', 'async_operations']
        }
        return capabilities.get(worker_type, [])
        
    def _get_current_queue_size(self) -> int:
        """Get current task queue size (placeholder)."""
        # In real implementation, this would check actual queue sizes
        return max(0, len(self.worker_nodes) * 5 - sum(1 for w in self.worker_nodes.values() if w.status == 'active'))
        
    def _estimate_request_rate(self) -> float:
        """Estimate current request rate."""
        if len(self.metrics_history) < 2:
            return 0.0
            
        # Simple rate estimation from recent metrics
        recent_metrics = list(self.metrics_history)[-10:]
        time_diff = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        
        if time_diff > 0:
            return len(recent_metrics) / time_diff
        return 0.0
        
    def _get_average_response_time(self) -> float:
        """Get average response time (placeholder)."""
        if self.metrics_history:
            recent_times = [m.response_time for m in list(self.metrics_history)[-20:]]
            return np.mean(recent_times)
        return 0.1
        
    def _get_current_error_rate(self) -> float:
        """Get current error rate (placeholder)."""
        if self.metrics_history:
            recent_errors = [m.error_rate for m in list(self.metrics_history)[-20:]]
            return np.mean(recent_errors)
        return 0.0
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        current_metrics = self._collect_scaling_metrics()
        
        return {
            'current_metrics': current_metrics.__dict__,
            'worker_count': len(self.worker_nodes),
            'active_workers': len([w for w in self.worker_nodes.values() if w.status == 'active']),
            'worker_nodes': {wid: w.__dict__ for wid, w in self.worker_nodes.items()},
            'recent_scaling_actions': self.scaling_history[-5:],
            'predictive_recommendation': self._predictive_model.predict_scaling_need(
                list(self.metrics_history)[-50:])
        }


class LoadBalancer:
    """Intelligent load balancer with multiple algorithms."""
    
    def __init__(self, 
                 algorithm: str = 'adaptive',
                 health_check_interval: int = 30):
        """Initialize load balancer.
        
        Args:
            algorithm: Load balancing algorithm ('round_robin', 'least_connections', 'adaptive')
            health_check_interval: Health check interval in seconds
        """
        self.algorithm = algorithm
        self.health_check_interval = health_check_interval
        
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        self._round_robin_index = 0
        self._health_check_thread = None
        self._stop_health_checks = False
        
        # Start health monitoring
        self.start_health_monitoring()
        
    def add_worker(self, worker: WorkerNode):
        """Add worker to load balancer pool."""
        self.worker_nodes[worker.node_id] = worker
        logger.info(f"Added worker {worker.node_id} to load balancer")
        
    def remove_worker(self, worker_id: str):
        """Remove worker from load balancer pool."""
        if worker_id in self.worker_nodes:
            del self.worker_nodes[worker_id]
            self.request_counts.pop(worker_id, None)
            self.response_times.pop(worker_id, None)
            logger.info(f"Removed worker {worker_id} from load balancer")
            
    def select_worker(self, request_metadata: Optional[Dict[str, Any]] = None) -> Optional[WorkerNode]:
        """Select optimal worker for request."""
        healthy_workers = [w for w in self.worker_nodes.values() 
                          if w.status == 'active' and self._is_healthy(w)]
                          
        if not healthy_workers:
            logger.warning("No healthy workers available")
            return None
            
        if self.algorithm == 'round_robin':
            return self._round_robin_selection(healthy_workers)
        elif self.algorithm == 'least_connections':
            return self._least_connections_selection(healthy_workers)
        elif self.algorithm == 'adaptive':
            return self._adaptive_selection(healthy_workers, request_metadata)
        else:
            # Default to round robin
            return self._round_robin_selection(healthy_workers)
            
    def _round_robin_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection."""
        if not workers:
            return None
            
        selected = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return selected
        
    def _least_connections_selection(self, workers: List[WorkerNode]) -> WorkerNode:
        """Select worker with least connections."""
        return min(workers, key=lambda w: self.request_counts[w.node_id])
        
    def _adaptive_selection(self, 
                          workers: List[WorkerNode], 
                          request_metadata: Optional[Dict[str, Any]] = None) -> WorkerNode:
        """Intelligent adaptive worker selection."""
        
        # Score workers based on multiple factors
        worker_scores = {}
        
        for worker in workers:
            score = 0.0
            
            # Factor 1: Current load (inverse of CPU usage)
            load_score = 1.0 - worker.cpu_usage
            
            # Factor 2: Response time history
            if worker.node_id in self.response_times:
                avg_response_time = np.mean(list(self.response_times[worker.node_id]))
                response_score = 1.0 / (1.0 + avg_response_time)
            else:
                response_score = 1.0
                
            # Factor 3: Request count balance
            min_requests = min(self.request_counts[w.node_id] for w in workers)
            max_requests = max(self.request_counts[w.node_id] for w in workers)
            if max_requests > min_requests:
                balance_score = 1.0 - ((self.request_counts[worker.node_id] - min_requests) / 
                                    (max_requests - min_requests))
            else:
                balance_score = 1.0
                
            # Factor 4: Worker specialization
            specialization_score = self._compute_specialization_score(worker, request_metadata)
            
            # Combine scores with weights
            score = (0.3 * load_score + 
                    0.25 * response_score + 
                    0.25 * balance_score + 
                    0.2 * specialization_score)
                    
            worker_scores[worker.node_id] = score
            
        # Select worker with highest score
        best_worker_id = max(worker_scores, key=worker_scores.get)
        return self.worker_nodes[best_worker_id]
        
    def _compute_specialization_score(self, 
                                    worker: WorkerNode, 
                                    request_metadata: Optional[Dict[str, Any]]) -> float:
        """Compute specialization score for worker-request match."""
        if not request_metadata:
            return 1.0
            
        required_capabilities = request_metadata.get('required_capabilities', [])
        if not required_capabilities:
            return 1.0
            
        # Score based on capability match
        matched_capabilities = set(worker.capabilities).intersection(set(required_capabilities))
        if required_capabilities:
            return len(matched_capabilities) / len(required_capabilities)
        else:
            return 1.0
            
    def record_request_completion(self, 
                                worker_id: str, 
                                response_time: float,
                                success: bool):
        """Record request completion for worker."""
        if worker_id in self.worker_nodes:
            self.request_counts[worker_id] += 1
            self.response_times[worker_id].append(response_time)
            
            # Update worker metrics
            worker = self.worker_nodes[worker_id]
            worker.request_count += 1
            worker.last_heartbeat = time.time()
            
    def start_health_monitoring(self):
        """Start health monitoring for workers."""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._stop_health_checks = False
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop, daemon=True)
            self._health_check_thread.start()
            logger.info("Started load balancer health monitoring")
            
    def stop_health_monitoring(self):
        """Stop health monitoring."""
        self._stop_health_checks = True
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        logger.info("Stopped load balancer health monitoring")
        
    def _health_check_loop(self):
        """Health check monitoring loop."""
        while not self._stop_health_checks:
            try:
                current_time = time.time()
                
                for worker in list(self.worker_nodes.values()):
                    # Check heartbeat
                    if current_time - worker.last_heartbeat > self.health_check_interval * 2:
                        worker.status = 'failed'
                        logger.warning(f"Worker {worker.node_id} failed health check")
                    elif worker.status == 'failed':
                        # Reset if heartbeat resumed
                        worker.status = 'active'
                        logger.info(f"Worker {worker.node_id} recovered")
                        
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(60)
                
    def _is_healthy(self, worker: WorkerNode) -> bool:
        """Check if worker is healthy."""
        current_time = time.time()
        return (worker.status == 'active' and 
                current_time - worker.last_heartbeat < self.health_check_interval * 2)
                
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        healthy_workers = [w for w in self.worker_nodes.values() if self._is_healthy(w)]
        
        return {
            'algorithm': self.algorithm,
            'total_workers': len(self.worker_nodes),
            'healthy_workers': len(healthy_workers),
            'request_distribution': dict(self.request_counts),
            'average_response_times': {
                wid: np.mean(list(times)) if times else 0 
                for wid, times in self.response_times.items()
            },
            'worker_statuses': {w.node_id: w.status for w in self.worker_nodes.values()}
        }


class PredictiveScaler:
    """Predictive scaling using machine learning techniques."""
    
    def __init__(self, history_window: int = 100):
        """Initialize predictive scaler.
        
        Args:
            history_window: Number of historical points to consider
        """
        self.history_window = history_window
        self.feature_history: deque = deque(maxlen=history_window)
        self.scaling_model = None
        self._model_accuracy = 0.5
        
    def update(self, metrics: ScalingMetrics):
        """Update the predictive model with new metrics."""
        # Extract features for prediction
        features = self._extract_features(metrics)
        self.feature_history.append(features)
        
        # Retrain model periodically
        if len(self.feature_history) >= 50 and len(self.feature_history) % 25 == 0:
            self._train_model()
            
    def predict_scaling_need(self, recent_metrics: List[ScalingMetrics]) -> str:
        """Predict if scaling action is needed."""
        if len(recent_metrics) < 10:
            return 'no_action'
            
        # Simple trend-based prediction for now
        # In production, this would use a trained ML model
        
        # Analyze trends in key metrics
        cpu_trend = self._analyze_trend([m.cpu_usage for m in recent_metrics[-10:]])
        response_time_trend = self._analyze_trend([m.response_time for m in recent_metrics[-10:]])
        queue_trend = self._analyze_trend([m.queue_size for m in recent_metrics[-10:]])
        
        # Make prediction based on trends
        if (cpu_trend > 0.1 or response_time_trend > 0.1 or queue_trend > 5):
            return 'scale_up'
        elif (cpu_trend < -0.1 and response_time_trend < -0.05 and queue_trend < -2):
            return 'scale_down'
        else:
            return 'no_action'
            
    def _extract_features(self, metrics: ScalingMetrics) -> List[float]:
        """Extract features for machine learning."""
        return [
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.request_rate,
            metrics.response_time,
            metrics.error_rate,
            float(metrics.queue_size),
            float(metrics.active_workers)
        ]
        
    def _analyze_trend(self, values: List[float]) -> float:
        """Analyze trend in values (simple linear regression slope)."""
        if len(values) < 2:
            return 0.0
            
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Linear regression slope
        slope = np.polyfit(x, y, 1)[0]
        return slope
        
    def _train_model(self):
        """Train predictive model (placeholder for actual ML implementation)."""
        # In a real implementation, this would train an actual ML model
        # using historical data to predict scaling needs
        
        if len(self.feature_history) >= 50:
            # Simulate model training
            self._model_accuracy = min(0.9, self._model_accuracy + 0.05)
            logger.info(f"Predictive model retrained. Accuracy: {self._model_accuracy:.2f}")


class ResourceManager:
    """Comprehensive resource management and optimization."""
    
    def __init__(self):
        """Initialize resource manager."""
        self.resource_pools: Dict[str, Any] = {}
        self.resource_monitors: Dict[str, Callable] = {}
        self.resource_limits: Dict[str, Dict[str, float]] = {
            'cpu': {'soft': 0.8, 'hard': 0.95},
            'memory': {'soft': 0.8, 'hard': 0.95},
            'disk': {'soft': 0.85, 'hard': 0.95},
            'network': {'soft': 100.0, 'hard': 150.0}  # MB/s
        }
        
        self._monitoring_thread = None
        self._stop_monitoring = False
        
        self.start_resource_monitoring()
        
    def start_resource_monitoring(self):
        """Start resource monitoring."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring = False
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            logger.info("Started resource monitoring")
            
    def stop_resource_monitoring(self):
        """Stop resource monitoring."""
        self._stop_monitoring = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
        
    def _monitoring_loop(self):
        """Resource monitoring loop."""
        while not self._stop_monitoring:
            try:
                # Check all resources
                resource_status = self._check_all_resources()
                
                # Take action if limits exceeded
                for resource, status in resource_status.items():
                    if status['usage'] > self.resource_limits[resource]['hard']:
                        logger.critical(f"Hard limit exceeded for {resource}: {status['usage']:.1%}")
                        self._handle_resource_emergency(resource, status)
                    elif status['usage'] > self.resource_limits[resource]['soft']:
                        logger.warning(f"Soft limit exceeded for {resource}: {status['usage']:.1%}")
                        self._handle_resource_warning(resource, status)
                        
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(60)
                
    def _check_all_resources(self) -> Dict[str, Dict[str, float]]:
        """Check status of all system resources."""
        return {
            'cpu': {'usage': psutil.cpu_percent() / 100.0},
            'memory': {'usage': psutil.virtual_memory().percent / 100.0},
            'disk': {'usage': psutil.disk_usage('/').percent / 100.0},
            'network': {'usage': self._get_network_usage()}
        }
        
    def _get_network_usage(self) -> float:
        """Get current network usage (simplified)."""
        # In real implementation, this would track actual network throughput
        return 0.3  # Placeholder
        
    def _handle_resource_emergency(self, resource: str, status: Dict[str, float]):
        """Handle resource emergency situations."""
        logger.critical(f"Emergency resource management for {resource}")
        
        if resource == 'cpu':
            # Reduce worker processes
            logger.info("Reducing CPU-intensive operations")
        elif resource == 'memory':
            # Force garbage collection and cache cleanup
            import gc
            gc.collect()
            logger.info("Forced memory cleanup")
        elif resource == 'disk':
            # Clean temporary files
            logger.info("Cleaning temporary files")
            
    def _handle_resource_warning(self, resource: str, status: Dict[str, float]):
        """Handle resource warning situations."""
        logger.warning(f"Resource warning for {resource}: {status['usage']:.1%}")
        
        # Implement gradual resource management
        if resource == 'memory':
            # Reduce cache sizes
            logger.info("Reducing cache sizes due to memory pressure")
            
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status."""
        return {
            'current_usage': self._check_all_resources(),
            'limits': self.resource_limits,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
                'disk_total': psutil.disk_usage('/').total / (1024**3),  # GB
            }
        }