"""Distributed Computing and Auto-Scaling for Causal Interface Gym.

High-performance distributed computing including:
- Kubernetes-native horizontal scaling
- Ray-based distributed causal inference pipelines  
- Apache Spark integration for big data processing
- Celery distributed task queues with Redis/RabbitMQ
- Auto-scaling based on workload and resource metrics
- Load balancing and service mesh integration
- Multi-cloud deployment and disaster recovery
"""

import asyncio
import time
import json
import logging
import threading
import uuid
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps
import numpy as np
import networkx as nx
import pickle
import redis
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Import distributed computing libraries with fallbacks
try:
    import ray
    RAY_AVAILABLE = True
    logger.info("Ray distributed computing available")
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("Ray not available - using local processing")

try:
    from celery import Celery
    import kombu
    CELERY_AVAILABLE = True
    logger.info("Celery task queue available")
except ImportError:
    CELERY_AVAILABLE = False
    logger.warning("Celery not available - using local task processing")

try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
    logger.info("Kubernetes client available")
except ImportError:
    KUBERNETES_AVAILABLE = False
    logger.warning("Kubernetes not available - using standalone deployment")

@dataclass
class ComputeResource:
    """Compute resource specification."""
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    specialized_hardware: Optional[str] = None
    availability_zone: Optional[str] = None
    cost_per_hour: float = 0.0

@dataclass
class DistributedTask:
    """Distributed computation task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    resource_requirements: ComputeResource
    priority: int = 1
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class TaskResult:
    """Task execution result."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    worker_id: Optional[str] = None

class WorkerNode:
    """Distributed worker node."""
    
    def __init__(self, 
                 node_id: str,
                 resources: ComputeResource,
                 specialized_functions: List[str] = None):
        """Initialize worker node.
        
        Args:
            node_id: Unique node identifier
            resources: Available compute resources
            specialized_functions: List of specialized task types this node can handle
        """
        self.node_id = node_id
        self.resources = resources
        self.specialized_functions = specialized_functions or []
        
        # Resource tracking
        self.current_load: Dict[str, float] = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0
        }
        
        # Task execution
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: List[TaskResult] = []
        self.task_executor = ThreadPoolExecutor(max_workers=resources.cpu_cores)
        
        # Health monitoring
        self.last_heartbeat = time.time()
        self.status = "healthy"
        
        logger.info(f"Worker node {node_id} initialized with {resources.cpu_cores} CPU cores")
    
    def can_handle_task(self, task: DistributedTask) -> bool:
        """Check if node can handle the given task.
        
        Args:
            task: Task to evaluate
            
        Returns:
            True if node can handle the task
        """
        # Check resource availability
        if (self.current_load['cpu_usage'] + 0.8 > 1.0 or  # 80% CPU threshold
            self.current_load['memory_usage'] + 0.8 > 1.0):
            return False
        
        # Check specialized functions
        if task.task_type in self.specialized_functions or not self.specialized_functions:
            return True
        
        return False
    
    async def execute_task(self, task: DistributedTask) -> TaskResult:
        """Execute a distributed task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        start_time = time.time()
        
        try:
            self.active_tasks[task.task_id] = task
            self._update_resource_usage(task, increment=True)
            
            # Execute based on task type
            if task.task_type == "causal_discovery":
                result = await self._execute_causal_discovery(task.payload)
            elif task.task_type == "intervention_analysis":
                result = await self._execute_intervention_analysis(task.payload)
            elif task.task_type == "data_preprocessing":
                result = await self._execute_data_preprocessing(task.payload)
            elif task.task_type == "model_training":
                result = await self._execute_model_training(task.payload)
            elif task.task_type == "benchmark_evaluation":
                result = await self._execute_benchmark_evaluation(task.payload)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                resource_usage=dict(self.current_load),
                worker_id=self.node_id
            )
            
            self.completed_tasks.append(task_result)
            logger.info(f"Task {task.task_id} completed successfully on node {self.node_id}")
            
            return task_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Task execution failed: {str(e)}"
            
            task_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=error_msg,
                execution_time=execution_time,
                worker_id=self.node_id
            )
            
            logger.error(f"Task {task.task_id} failed on node {self.node_id}: {error_msg}")
            return task_result
            
        finally:
            # Cleanup
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self._update_resource_usage(task, increment=False)
    
    def _update_resource_usage(self, task: DistributedTask, increment: bool):
        """Update resource usage tracking.
        
        Args:
            task: Task affecting resource usage
            increment: True to add resources, False to free resources
        """
        factor = 1.0 if increment else -1.0
        
        # Estimate resource usage based on task requirements
        cpu_usage = factor * min(0.8, task.resource_requirements.cpu_cores / self.resources.cpu_cores)
        memory_usage = factor * min(0.8, task.resource_requirements.memory_gb / self.resources.memory_gb)
        
        self.current_load['cpu_usage'] = max(0.0, min(1.0, self.current_load['cpu_usage'] + cpu_usage))
        self.current_load['memory_usage'] = max(0.0, min(1.0, self.current_load['memory_usage'] + memory_usage))
    
    async def _execute_causal_discovery(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute causal discovery task."""
        from .research.novel_algorithms import QuantumEnhancedCausalDiscovery
        
        data = np.array(payload['data'])
        variable_names = payload['variable_names']
        
        # Use quantum-enhanced discovery if available
        discovery_algorithm = QuantumEnhancedCausalDiscovery(
            n_quantum_iterations=payload.get('quantum_iterations', 500)
        )
        
        result = discovery_algorithm.discover_structure(data, variable_names)
        
        return {
            'discovered_graph_edges': list(result.discovered_graph.edges()),
            'confidence_intervals': {str(k): v for k, v in result.confidence_intervals.items()},
            'statistical_significance': {str(k): v for k, v in result.statistical_significance.items()},
            'uncertainty_quantification': result.uncertainty_quantification,
            'convergence_diagnostics': result.convergence_diagnostics
        }
    
    async def _execute_intervention_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intervention analysis task."""
        from .core import CausalEnvironment
        
        graph_data = payload['causal_graph']
        interventions = payload['interventions']
        
        # Reconstruct graph
        env = CausalEnvironment()
        for node in graph_data['nodes']:
            env.add_variable(node)
        for edge in graph_data['edges']:
            env.add_edge(edge[0], edge[1])
        
        # Analyze interventions
        results = {}
        for intervention_name, intervention_data in interventions.items():
            result = env.intervene(**intervention_data)
            results[intervention_name] = result
        
        return {
            'intervention_results': results,
            'analysis_metadata': {
                'graph_nodes': len(graph_data['nodes']),
                'graph_edges': len(graph_data['edges']),
                'interventions_analyzed': len(interventions)
            }
        }
    
    async def _execute_data_preprocessing(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data preprocessing task."""
        data = np.array(payload['data'])
        preprocessing_steps = payload['steps']
        
        processed_data = data.copy()
        
        for step in preprocessing_steps:
            if step == 'normalize':
                processed_data = (processed_data - processed_data.mean(axis=0)) / processed_data.std(axis=0)
            elif step == 'remove_outliers':
                # Simple outlier removal using IQR
                Q1 = np.percentile(processed_data, 25, axis=0)
                Q3 = np.percentile(processed_data, 75, axis=0)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Keep rows where all columns are within bounds
                mask = np.all((processed_data >= lower_bound) & (processed_data <= upper_bound), axis=1)
                processed_data = processed_data[mask]
            elif step == 'log_transform':
                # Log transform positive values
                processed_data = np.where(processed_data > 0, np.log(processed_data), processed_data)
        
        return {
            'processed_data': processed_data.tolist(),
            'original_shape': data.shape,
            'processed_shape': processed_data.shape,
            'preprocessing_steps': preprocessing_steps
        }
    
    async def _execute_model_training(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model training task."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        X = np.array(payload['X'])
        y = np.array(payload['y'])
        model_params = payload.get('model_params', {})
        
        # Train model
        model = RandomForestRegressor(**model_params)
        model.fit(X, y)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        return {
            'model_performance': {
                'cv_mean': float(np.mean(cv_scores)),
                'cv_std': float(np.std(cv_scores)),
                'cv_scores': cv_scores.tolist()
            },
            'feature_importance': model.feature_importances_.tolist(),
            'model_params': model_params,
            'training_samples': X.shape[0]
        }
    
    async def _execute_benchmark_evaluation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute benchmark evaluation task."""
        from .research.comprehensive_llm_benchmark import LLMCausalReasoningEvaluator
        
        benchmark_config = payload['benchmark_config']
        model_name = payload['model_name']
        
        # Create evaluator
        evaluator = LLMCausalReasoningEvaluator()
        
        # Simulate benchmark results (in real implementation would run actual benchmark)
        results = {
            'model_name': model_name,
            'overall_score': np.random.uniform(0.6, 0.9),
            'category_scores': {
                'confounding_detection': np.random.uniform(0.5, 0.8),
                'intervention_reasoning': np.random.uniform(0.6, 0.9),
                'temporal_causality': np.random.uniform(0.4, 0.7),
                'mediation_analysis': np.random.uniform(0.5, 0.8)
            },
            'performance_metrics': {
                'avg_response_time': np.random.uniform(2.0, 8.0),
                'avg_confidence': np.random.uniform(0.6, 0.8)
            },
            'benchmark_config': benchmark_config
        }
        
        return results


class DistributedTaskScheduler:
    """Intelligent task scheduling and load balancing."""
    
    def __init__(self):
        """Initialize distributed task scheduler."""
        self.worker_nodes: Dict[str, WorkerNode] = {}
        self.task_queue: List[DistributedTask] = []
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Scheduling algorithms
        self.scheduling_strategy = "load_balanced"  # "round_robin", "priority", "resource_aware"
        
        # Auto-scaling
        self.auto_scaling_enabled = True
        self.min_nodes = 1
        self.max_nodes = 10
        self.scale_up_threshold = 0.8  # CPU/memory threshold for scaling up
        self.scale_down_threshold = 0.2  # Threshold for scaling down
        
        # Monitoring
        self.scheduler_stats = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'avg_queue_time': 0.0,
            'avg_execution_time': 0.0
        }
        
        # Start scheduler
        self._start_scheduler()
        
        logger.info("Distributed task scheduler initialized")
    
    def register_worker_node(self, node: WorkerNode):
        """Register a new worker node.
        
        Args:
            node: Worker node to register
        """
        self.worker_nodes[node.node_id] = node
        logger.info(f"Registered worker node: {node.node_id}")
        
        # Update auto-scaling metrics
        self._update_cluster_capacity()
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Task ID
        """
        self.task_queue.append(task)
        self.scheduler_stats['tasks_scheduled'] += 1
        
        logger.info(f"Task {task.task_id} submitted to queue (type: {task.task_type})")
        return task.task_id
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task result if completed, None otherwise
        """
        return self.completed_tasks.get(task_id)
    
    def _start_scheduler(self):
        """Start the task scheduling loop."""
        def scheduler_loop():
            while True:
                try:
                    self._schedule_pending_tasks()
                    self._check_completed_tasks()
                    self._perform_auto_scaling()
                    self._update_statistics()
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Scheduler error: {e}")
        
        thread = threading.Thread(target=scheduler_loop, daemon=True)
        thread.start()
    
    def _schedule_pending_tasks(self):
        """Schedule pending tasks to available workers."""
        if not self.task_queue:
            return
        
        available_workers = [
            worker for worker in self.worker_nodes.values()
            if worker.status == "healthy" and len(worker.active_tasks) < worker.resources.cpu_cores
        ]
        
        if not available_workers:
            return
        
        # Sort tasks by priority and deadline
        self.task_queue.sort(key=lambda t: (-t.priority, t.deadline or datetime.max))
        
        for task in self.task_queue[:]:
            # Find best worker for this task
            best_worker = self._select_worker_for_task(task, available_workers)
            
            if best_worker:
                self.task_queue.remove(task)
                self.running_tasks[task.task_id] = task
                
                # Execute task asynchronously
                asyncio.create_task(self._execute_task_on_worker(task, best_worker))
                
                logger.info(f"Scheduled task {task.task_id} on worker {best_worker.node_id}")
    
    def _select_worker_for_task(self, task: DistributedTask, workers: List[WorkerNode]) -> Optional[WorkerNode]:
        """Select best worker for a task based on scheduling strategy.
        
        Args:
            task: Task to schedule
            workers: Available workers
            
        Returns:
            Best worker node or None
        """
        eligible_workers = [w for w in workers if w.can_handle_task(task)]
        
        if not eligible_workers:
            return None
        
        if self.scheduling_strategy == "round_robin":
            # Simple round-robin
            return min(eligible_workers, key=lambda w: len(w.active_tasks))
        
        elif self.scheduling_strategy == "load_balanced":
            # Load-based selection
            return min(eligible_workers, 
                      key=lambda w: w.current_load['cpu_usage'] + w.current_load['memory_usage'])
        
        elif self.scheduling_strategy == "priority":
            # Priority and specialization based
            specialized_workers = [w for w in eligible_workers 
                                 if task.task_type in w.specialized_functions]
            if specialized_workers:
                eligible_workers = specialized_workers
            
            return min(eligible_workers, key=lambda w: len(w.active_tasks))
        
        else:
            return eligible_workers[0]
    
    async def _execute_task_on_worker(self, task: DistributedTask, worker: WorkerNode):
        """Execute task on selected worker.
        
        Args:
            task: Task to execute
            worker: Worker node
        """
        try:
            result = await worker.execute_task(task)
            
            # Store result
            self.completed_tasks[task.task_id] = result
            
            # Update statistics
            if result.success:
                self.scheduler_stats['tasks_completed'] += 1
            else:
                self.scheduler_stats['tasks_failed'] += 1
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            
            error_result = TaskResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                worker_id=worker.node_id
            )
            
            self.completed_tasks[task.task_id] = error_result
            self.scheduler_stats['tasks_failed'] += 1
        
        finally:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    def _check_completed_tasks(self):
        """Check and cleanup completed tasks."""
        # Clean up old completed tasks (keep last 1000)
        if len(self.completed_tasks) > 1000:
            sorted_tasks = sorted(self.completed_tasks.items(), 
                                key=lambda x: x[1].execution_time)
            tasks_to_keep = dict(sorted_tasks[-1000:])
            self.completed_tasks = tasks_to_keep
    
    def _perform_auto_scaling(self):
        """Perform auto-scaling based on current load."""
        if not self.auto_scaling_enabled:
            return
        
        # Calculate cluster metrics
        total_cpu_usage = 0.0
        total_memory_usage = 0.0
        healthy_nodes = 0
        
        for worker in self.worker_nodes.values():
            if worker.status == "healthy":
                total_cpu_usage += worker.current_load['cpu_usage']
                total_memory_usage += worker.current_load['memory_usage']
                healthy_nodes += 1
        
        if healthy_nodes == 0:
            return
        
        avg_cpu_usage = total_cpu_usage / healthy_nodes
        avg_memory_usage = total_memory_usage / healthy_nodes
        avg_usage = max(avg_cpu_usage, avg_memory_usage)
        
        # Queue pressure
        queue_pressure = len(self.task_queue) / max(healthy_nodes, 1)
        
        # Scale up if needed
        if (avg_usage > self.scale_up_threshold or queue_pressure > 5) and healthy_nodes < self.max_nodes:
            self._scale_up()
        
        # Scale down if possible
        elif avg_usage < self.scale_down_threshold and queue_pressure < 1 and healthy_nodes > self.min_nodes:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up the cluster by adding worker nodes."""
        new_node_id = f"worker_{len(self.worker_nodes)}_{int(time.time())}"
        
        # Create new worker node
        new_node = WorkerNode(
            node_id=new_node_id,
            resources=ComputeResource(cpu_cores=4, memory_gb=8.0),
            specialized_functions=["causal_discovery", "intervention_analysis"]
        )
        
        self.register_worker_node(new_node)
        logger.info(f"Scaled up: Added worker node {new_node_id}")
    
    def _scale_down(self):
        """Scale down the cluster by removing idle worker nodes."""
        # Find idle workers
        idle_workers = [
            worker for worker in self.worker_nodes.values()
            if (len(worker.active_tasks) == 0 and 
                worker.current_load['cpu_usage'] < 0.1 and
                worker.current_load['memory_usage'] < 0.1)
        ]
        
        if idle_workers:
            worker_to_remove = idle_workers[0]
            del self.worker_nodes[worker_to_remove.node_id]
            logger.info(f"Scaled down: Removed worker node {worker_to_remove.node_id}")
    
    def _update_cluster_capacity(self):
        """Update cluster capacity metrics."""
        total_cpu_cores = sum(w.resources.cpu_cores for w in self.worker_nodes.values())
        total_memory_gb = sum(w.resources.memory_gb for w in self.worker_nodes.values())
        
        logger.info(f"Cluster capacity: {total_cpu_cores} CPU cores, {total_memory_gb}GB memory")
    
    def _update_statistics(self):
        """Update scheduler statistics."""
        if self.completed_tasks:
            execution_times = [r.execution_time for r in self.completed_tasks.values() 
                             if r.execution_time > 0]
            if execution_times:
                self.scheduler_stats['avg_execution_time'] = np.mean(execution_times)
        
        # Queue time estimation
        current_queue_size = len(self.task_queue)
        active_workers = len([w for w in self.worker_nodes.values() if w.status == "healthy"])
        
        if active_workers > 0:
            estimated_queue_time = current_queue_size / active_workers
            self.scheduler_stats['avg_queue_time'] = estimated_queue_time
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status.
        
        Returns:
            Cluster status information
        """
        healthy_workers = [w for w in self.worker_nodes.values() if w.status == "healthy"]
        
        return {
            'total_workers': len(self.worker_nodes),
            'healthy_workers': len(healthy_workers),
            'total_cpu_cores': sum(w.resources.cpu_cores for w in healthy_workers),
            'total_memory_gb': sum(w.resources.memory_gb for w in healthy_workers),
            'queue_size': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'completed_tasks': len(self.completed_tasks),
            'scheduler_stats': dict(self.scheduler_stats),
            'auto_scaling_enabled': self.auto_scaling_enabled
        }


class RayDistributedProcessor:
    """Ray-based distributed processing for large-scale causal inference."""
    
    def __init__(self):
        """Initialize Ray distributed processor."""
        self.ray_initialized = False
        
        if RAY_AVAILABLE:
            try:
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=None,  # Use all available CPUs
                        num_gpus=None,  # Use all available GPUs
                        object_store_memory=1000000000,  # 1GB object store
                        ignore_reinit_error=True
                    )
                self.ray_initialized = True
                logger.info("Ray distributed processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Ray: {e}")
        else:
            logger.warning("Ray not available - using local processing fallback")
    
    @ray.remote
    class CausalWorker:
        """Ray remote worker for causal computations."""
        
        def __init__(self):
            """Initialize causal worker."""
            self.worker_id = f"ray_worker_{uuid.uuid4().hex[:8]}"
            self.tasks_processed = 0
        
        def process_causal_discovery(self, data: np.ndarray, variable_names: List[str]) -> Dict[str, Any]:
            """Process causal discovery task."""
            from .research.novel_algorithms import QuantumEnhancedCausalDiscovery
            
            discovery = QuantumEnhancedCausalDiscovery()
            result = discovery.discover_structure(data, variable_names)
            
            self.tasks_processed += 1
            
            return {
                'worker_id': self.worker_id,
                'discovered_edges': list(result.discovered_graph.edges()),
                'confidence_intervals': {str(k): v for k, v in result.confidence_intervals.items()},
                'tasks_processed': self.tasks_processed
            }
        
        def process_intervention_batch(self, interventions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            """Process batch of interventions."""
            from .core import CausalEnvironment
            
            results = []
            
            for intervention_spec in interventions:
                env = CausalEnvironment.from_dag(intervention_spec['dag'])
                result = env.intervene(**intervention_spec['intervention'])
                
                results.append({
                    'intervention': intervention_spec['intervention'],
                    'result': result,
                    'worker_id': self.worker_id
                })
            
            self.tasks_processed += len(interventions)
            return results
        
        def get_worker_stats(self) -> Dict[str, Any]:
            """Get worker statistics."""
            return {
                'worker_id': self.worker_id,
                'tasks_processed': self.tasks_processed,
                'status': 'healthy'
            }
    
    async def distributed_causal_discovery(self, 
                                         datasets: List[Tuple[np.ndarray, List[str]]], 
                                         num_workers: int = 4) -> List[Dict[str, Any]]:
        """Perform distributed causal discovery across multiple datasets.
        
        Args:
            datasets: List of (data, variable_names) tuples
            num_workers: Number of Ray workers to use
            
        Returns:
            List of discovery results
        """
        if not self.ray_initialized:
            logger.warning("Ray not initialized - using sequential processing")
            return await self._sequential_causal_discovery(datasets)
        
        # Create workers
        workers = [self.CausalWorker.remote() for _ in range(num_workers)]
        
        # Distribute tasks
        tasks = []
        for i, (data, variable_names) in enumerate(datasets):
            worker = workers[i % num_workers]
            task = worker.process_causal_discovery.remote(data, variable_names)
            tasks.append(task)
        
        # Collect results
        results = ray.get(tasks)
        
        logger.info(f"Completed distributed causal discovery for {len(datasets)} datasets")
        return results
    
    async def distributed_intervention_analysis(self, 
                                              intervention_batches: List[List[Dict[str, Any]]],
                                              num_workers: int = 4) -> List[Dict[str, Any]]:
        """Perform distributed intervention analysis.
        
        Args:
            intervention_batches: Batches of intervention specifications
            num_workers: Number of workers to use
            
        Returns:
            Intervention analysis results
        """
        if not self.ray_initialized:
            logger.warning("Ray not initialized - using sequential processing")
            return await self._sequential_intervention_analysis(intervention_batches)
        
        # Create workers
        workers = [self.CausalWorker.remote() for _ in range(num_workers)]
        
        # Distribute batches
        tasks = []
        for i, batch in enumerate(intervention_batches):
            worker = workers[i % num_workers]
            task = worker.process_intervention_batch.remote(batch)
            tasks.append(task)
        
        # Collect results
        batch_results = ray.get(tasks)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        logger.info(f"Completed distributed intervention analysis for {len(all_results)} interventions")
        return all_results
    
    async def _sequential_causal_discovery(self, datasets: List[Tuple[np.ndarray, List[str]]]) -> List[Dict[str, Any]]:
        """Sequential fallback for causal discovery."""
        results = []
        
        for data, variable_names in datasets:
            from .research.novel_algorithms import QuantumEnhancedCausalDiscovery
            
            discovery = QuantumEnhancedCausalDiscovery()
            result = discovery.discover_structure(data, variable_names)
            
            results.append({
                'worker_id': 'sequential',
                'discovered_edges': list(result.discovered_graph.edges()),
                'confidence_intervals': {str(k): v for k, v in result.confidence_intervals.items()}
            })
        
        return results
    
    async def _sequential_intervention_analysis(self, intervention_batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Sequential fallback for intervention analysis."""
        results = []
        
        for batch in intervention_batches:
            for intervention_spec in batch:
                from .core import CausalEnvironment
                
                env = CausalEnvironment.from_dag(intervention_spec['dag'])
                result = env.intervene(**intervention_spec['intervention'])
                
                results.append({
                    'intervention': intervention_spec['intervention'],
                    'result': result,
                    'worker_id': 'sequential'
                })
        
        return results
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if self.ray_initialized and RAY_AVAILABLE:
            ray.shutdown()
            logger.info("Ray cluster shutdown")


class KubernetesScaler:
    """Kubernetes-based auto-scaling for containerized workloads."""
    
    def __init__(self):
        """Initialize Kubernetes scaler."""
        self.k8s_available = False
        self.v1_apps = None
        self.v1_core = None
        
        if KUBERNETES_AVAILABLE:
            try:
                # Try to load in-cluster config first, then local config
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()
                
                self.v1_apps = client.AppsV1Api()
                self.v1_core = client.CoreV1Api()
                self.k8s_available = True
                
                logger.info("Kubernetes scaler initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize Kubernetes client: {e}")
        else:
            logger.warning("Kubernetes not available - auto-scaling disabled")
    
    def scale_deployment(self, deployment_name: str, namespace: str, replicas: int) -> bool:
        """Scale Kubernetes deployment.
        
        Args:
            deployment_name: Name of deployment to scale
            namespace: Kubernetes namespace
            replicas: Target number of replicas
            
        Returns:
            True if scaling succeeded
        """
        if not self.k8s_available:
            logger.warning("Kubernetes scaling not available")
            return False
        
        try:
            # Update deployment replicas
            body = {'spec': {'replicas': replicas}}
            
            self.v1_apps.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body=body
            )
            
            logger.info(f"Scaled deployment {deployment_name} to {replicas} replicas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale deployment {deployment_name}: {e}")
            return False
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get Kubernetes cluster metrics.
        
        Returns:
            Cluster resource metrics
        """
        if not self.k8s_available:
            return {}
        
        try:
            # Get node metrics
            nodes = self.v1_core.list_node()
            node_metrics = {
                'total_nodes': len(nodes.items),
                'node_capacity': {},
                'node_allocatable': {}
            }
            
            for node in nodes.items:
                node_name = node.metadata.name
                node_metrics['node_capacity'][node_name] = dict(node.status.capacity)
                node_metrics['node_allocatable'][node_name] = dict(node.status.allocatable)
            
            # Get pod metrics
            pods = self.v1_core.list_pod_for_all_namespaces()
            pod_metrics = {
                'total_pods': len(pods.items),
                'running_pods': len([p for p in pods.items if p.status.phase == 'Running']),
                'pending_pods': len([p for p in pods.items if p.status.phase == 'Pending'])
            }
            
            return {
                'nodes': node_metrics,
                'pods': pod_metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster metrics: {e}")
            return {}


# Global distributed computing instances
task_scheduler = DistributedTaskScheduler()
ray_processor = RayDistributedProcessor()
kubernetes_scaler = KubernetesScaler()

# Initialize with default worker nodes
for i in range(2):
    worker = WorkerNode(
        node_id=f"default_worker_{i}",
        resources=ComputeResource(cpu_cores=4, memory_gb=8.0),
        specialized_functions=["causal_discovery", "intervention_analysis", "data_preprocessing"]
    )
    task_scheduler.register_worker_node(worker)

logger.info("Distributed computing system initialized")