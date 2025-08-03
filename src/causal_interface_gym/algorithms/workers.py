"""Async job processing and queue management for causal reasoning experiments."""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from abc import ABC, abstractmethod
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, PriorityQueue
import weakref

from ..core import CausalEnvironment
from ..metrics import BeliefState
from .scoring import CausalScore, CausalScorer
from .pipeline import AnalysisPipeline, ExperimentData

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class JobType(Enum):
    """Types of jobs that can be processed."""
    EXPERIMENT_ANALYSIS = "experiment_analysis"
    SCORING = "scoring"
    DATA_PROCESSING = "data_processing"
    AGGREGATION = "aggregation"
    CLEANUP = "cleanup"
    NOTIFICATION = "notification"
    EXPORT = "export"


@dataclass
class JobResult:
    """Result of job execution."""
    job_id: str
    status: JobStatus
    result_data: Any = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    """Job to be processed by workers."""
    job_id: str
    job_type: JobType
    priority: JobPriority
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0
    timeout_seconds: int = 300
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """Compare jobs for priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


class JobWorker(ABC):
    """Abstract base class for job workers."""
    
    def __init__(self, worker_id: str, job_types: List[JobType]):
        """Initialize job worker.
        
        Args:
            worker_id: Unique worker identifier
            job_types: Types of jobs this worker can handle
        """
        self.worker_id = worker_id
        self.job_types = job_types
        self.is_running = False
        self.current_job: Optional[Job] = None
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.started_at: Optional[datetime] = None
    
    @abstractmethod
    async def process_job(self, job: Job) -> JobResult:
        """Process a job.
        
        Args:
            job: Job to process
            
        Returns:
            Job result
        """
        pass
    
    def can_handle(self, job: Job) -> bool:
        """Check if worker can handle job type.
        
        Args:
            job: Job to check
            
        Returns:
            True if worker can handle job
        """
        return job.job_type in self.job_types
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics.
        
        Returns:
            Worker statistics
        """
        uptime = (datetime.now() - self.started_at).total_seconds() if self.started_at else 0
        
        return {
            "worker_id": self.worker_id,
            "job_types": [jt.value for jt in self.job_types],
            "is_running": self.is_running,
            "current_job": self.current_job.job_id if self.current_job else None,
            "jobs_processed": self.jobs_processed,
            "jobs_failed": self.jobs_failed,
            "success_rate": self.jobs_processed / max(self.jobs_processed + self.jobs_failed, 1),
            "uptime_seconds": uptime
        }


class ExperimentAnalysisWorker(JobWorker):
    """Worker for processing experiment analysis jobs."""
    
    def __init__(self, worker_id: str):
        """Initialize experiment analysis worker."""
        super().__init__(worker_id, [JobType.EXPERIMENT_ANALYSIS])
        self.pipeline = AnalysisPipeline()
        self.scorer = CausalScorer()
    
    async def process_job(self, job: Job) -> JobResult:
        """Process experiment analysis job.
        
        Args:
            job: Analysis job
            
        Returns:
            Job result
        """
        start_time = datetime.now()
        
        try:
            # Extract experiment data from payload
            payload = job.payload
            experiment_data = ExperimentData(
                experiment_id=payload["experiment_id"],
                agent_id=payload["agent_id"],
                environment=payload["environment"],  # Should be CausalEnvironment object
                belief_history=payload["belief_history"],  # Should be List[BeliefState]
                interventions=payload["interventions"],
                ground_truth=payload["ground_truth"],
                raw_data=payload.get("raw_data", {}),
                metadata=payload.get("metadata", {})
            )
            
            # Process through analysis pipeline
            pipeline_result = await self.pipeline.process_experiment(experiment_data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                result_data=pipeline_result,
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now(),
                metadata={
                    "pipeline_id": pipeline_result.get("pipeline_id"),
                    "final_score": pipeline_result.get("final_score"),
                    "success": pipeline_result.get("success", False)
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Experiment analysis job {job.job_id} failed: {e}")
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now()
            )


class ScoringWorker(JobWorker):
    """Worker for scoring causal reasoning performance."""
    
    def __init__(self, worker_id: str):
        """Initialize scoring worker."""
        super().__init__(worker_id, [JobType.SCORING])
        self.scorer = CausalScorer()
    
    async def process_job(self, job: Job) -> JobResult:
        """Process scoring job.
        
        Args:
            job: Scoring job
            
        Returns:
            Job result
        """
        start_time = datetime.now()
        
        try:
            payload = job.payload
            
            # Extract required components
            environment = payload["environment"]
            belief_history = payload["belief_history"]
            interventions = payload["interventions"]
            ground_truth = payload["ground_truth"]
            
            # Calculate score
            score = self.scorer.score_causal_reasoning(
                environment, belief_history, interventions, ground_truth
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                result_data=asdict(score),
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now(),
                metadata={
                    "total_score": score.total_score,
                    "scoring_method": score.metadata.get("scoring_method")
                }
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Scoring job {job.job_id} failed: {e}")
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now()
            )


class DataProcessingWorker(JobWorker):
    """Worker for data processing and transformation tasks."""
    
    def __init__(self, worker_id: str):
        """Initialize data processing worker."""
        super().__init__(worker_id, [JobType.DATA_PROCESSING])
    
    async def process_job(self, job: Job) -> JobResult:
        """Process data processing job.
        
        Args:
            job: Data processing job
            
        Returns:
            Job result
        """
        start_time = datetime.now()
        
        try:
            payload = job.payload
            processing_type = payload.get("processing_type", "unknown")
            
            if processing_type == "belief_aggregation":
                result = await self._process_belief_aggregation(payload)
            elif processing_type == "intervention_analysis":
                result = await self._process_intervention_analysis(payload)
            elif processing_type == "data_export":
                result = await self._process_data_export(payload)
            else:
                raise ValueError(f"Unknown processing type: {processing_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                result_data=result,
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now(),
                metadata={"processing_type": processing_type}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Data processing job {job.job_id} failed: {e}")
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now()
            )
    
    async def _process_belief_aggregation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process belief aggregation task."""
        belief_histories = payload["belief_histories"]
        
        # Aggregate beliefs across multiple experiments
        aggregated_beliefs = {}
        
        for history in belief_histories:
            for belief_state in history:
                for relationship, strength in belief_state.causal_beliefs.items():
                    if relationship not in aggregated_beliefs:
                        aggregated_beliefs[relationship] = []
                    aggregated_beliefs[relationship].append(strength)
        
        # Calculate statistics
        belief_stats = {}
        for relationship, values in aggregated_beliefs.items():
            belief_stats[relationship] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5
            }
        
        return {
            "aggregated_beliefs": belief_stats,
            "total_experiments": len(belief_histories),
            "unique_relationships": len(belief_stats)
        }
    
    async def _process_intervention_analysis(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process intervention effectiveness analysis."""
        interventions_list = payload["interventions_list"]
        
        # Analyze intervention patterns
        intervention_stats = {
            "total_interventions": sum(len(interventions) for interventions in interventions_list),
            "avg_interventions_per_experiment": sum(len(interventions) for interventions in interventions_list) / len(interventions_list),
            "variable_frequency": {},
            "value_distribution": []
        }
        
        for interventions in interventions_list:
            for intervention in interventions:
                variable = intervention.get("variable", "unknown")
                value = intervention.get("value", 0)
                
                intervention_stats["variable_frequency"][variable] = intervention_stats["variable_frequency"].get(variable, 0) + 1
                intervention_stats["value_distribution"].append(value)
        
        return intervention_stats
    
    async def _process_data_export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process data export task."""
        export_format = payload.get("format", "json")
        data = payload["data"]
        
        if export_format == "json":
            exported_data = json.dumps(data, indent=2, default=str)
        elif export_format == "csv":
            # Simple CSV export (would need more sophisticated implementation)
            exported_data = "CSV export not fully implemented"
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return {
            "exported_data": exported_data,
            "format": export_format,
            "size_bytes": len(exported_data) if isinstance(exported_data, str) else 0
        }


class NotificationWorker(JobWorker):
    """Worker for sending notifications."""
    
    def __init__(self, worker_id: str):
        """Initialize notification worker."""
        super().__init__(worker_id, [JobType.NOTIFICATION])
    
    async def process_job(self, job: Job) -> JobResult:
        """Process notification job.
        
        Args:
            job: Notification job
            
        Returns:
            Job result
        """
        start_time = datetime.now()
        
        try:
            payload = job.payload
            notification_type = payload.get("type", "unknown")
            
            if notification_type == "experiment_complete":
                result = await self._send_experiment_notification(payload)
            elif notification_type == "system_alert":
                result = await self._send_system_alert(payload)
            else:
                raise ValueError(f"Unknown notification type: {notification_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                result_data=result,
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now(),
                metadata={"notification_type": notification_type}
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Notification job {job.job_id} failed: {e}")
            
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now()
            )
    
    async def _send_experiment_notification(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send experiment completion notification."""
        # Simulate notification sending
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            "notification_sent": True,
            "recipients": payload.get("recipients", []),
            "experiment_id": payload.get("experiment_id"),
            "message": "Experiment completed successfully"
        }
    
    async def _send_system_alert(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send system alert notification."""
        # Simulate alert sending
        await asyncio.sleep(0.1)
        
        return {
            "alert_sent": True,
            "severity": payload.get("severity", "info"),
            "message": payload.get("message", "System alert")
        }


class QueueManager:
    """Manages job queues and worker assignment."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize queue manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.job_queue = PriorityQueue()
        self.workers: Dict[str, JobWorker] = {}
        self.active_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, JobResult] = {}
        self.failed_jobs: Dict[str, JobResult] = {}
        
        self.is_running = False
        self.stats = {
            "jobs_queued": 0,
            "jobs_processed": 0,
            "jobs_failed": 0,
            "total_processing_time": 0.0
        }
        
        # Event for graceful shutdown
        self.shutdown_event = asyncio.Event()
        
        # Register default workers
        self._register_default_workers()
    
    def _register_default_workers(self):
        """Register default workers."""
        workers = [
            ExperimentAnalysisWorker("analysis_worker_1"),
            ExperimentAnalysisWorker("analysis_worker_2"),
            ScoringWorker("scoring_worker_1"),
            DataProcessingWorker("data_worker_1"),
            NotificationWorker("notification_worker_1")
        ]
        
        for worker in workers:
            self.register_worker(worker)
    
    def register_worker(self, worker: JobWorker):
        """Register a worker.
        
        Args:
            worker: Worker to register
        """
        self.workers[worker.worker_id] = worker
        logger.info(f"Registered worker: {worker.worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker.
        
        Args:
            worker_id: ID of worker to unregister
        """
        if worker_id in self.workers:
            del self.workers[worker_id]
            logger.info(f"Unregistered worker: {worker_id}")
    
    async def submit_job(self, job: Job) -> str:
        """Submit a job to the queue.
        
        Args:
            job: Job to submit
            
        Returns:
            Job ID
        """
        # Set scheduled time if not set
        if job.scheduled_at is None:
            job.scheduled_at = datetime.now()
        
        # Add to queue
        await self.job_queue.put(job)
        self.stats["jobs_queued"] += 1
        
        logger.info(f"Submitted job {job.job_id} of type {job.job_type.value}")
        return job.job_id
    
    async def create_and_submit_job(self, job_type: JobType, payload: Dict[str, Any],
                                   priority: JobPriority = JobPriority.NORMAL,
                                   **kwargs) -> str:
        """Create and submit a job.
        
        Args:
            job_type: Type of job
            payload: Job payload
            priority: Job priority
            **kwargs: Additional job parameters
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            priority=priority,
            payload=payload,
            **kwargs
        )
        
        return await self.submit_job(job)
    
    async def start(self):
        """Start the queue manager."""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        logger.info("Starting queue manager")
        
        # Start worker tasks
        worker_tasks = []
        for worker in list(self.workers.values())[:self.max_workers]:
            task = asyncio.create_task(self._worker_loop(worker))
            worker_tasks.append(task)
        
        # Wait for shutdown or worker completion
        try:
            await asyncio.gather(*worker_tasks)
        except asyncio.CancelledError:
            logger.info("Queue manager tasks cancelled")
    
    async def stop(self):
        """Stop the queue manager."""
        if not self.is_running:
            return
        
        logger.info("Stopping queue manager")
        self.is_running = False
        self.shutdown_event.set()
        
        # Mark all workers as stopped
        for worker in self.workers.values():
            worker.is_running = False
    
    async def _worker_loop(self, worker: JobWorker):
        """Main worker loop.
        
        Args:
            worker: Worker to run
        """
        worker.is_running = True
        worker.started_at = datetime.now()
        
        logger.info(f"Started worker {worker.worker_id}")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Get next job from queue (with timeout)
                    job = await asyncio.wait_for(
                        self._get_next_job_for_worker(worker),
                        timeout=1.0
                    )
                    
                    if job is None:
                        continue
                    
                    # Process the job
                    await self._process_job_with_worker(worker, job)
                    
                except asyncio.TimeoutError:
                    # No job available, continue
                    continue
                except Exception as e:
                    logger.error(f"Worker {worker.worker_id} error: {e}")
                    await asyncio.sleep(1)  # Brief pause before retrying
        
        finally:
            worker.is_running = False
            logger.info(f"Stopped worker {worker.worker_id}")
    
    async def _get_next_job_for_worker(self, worker: JobWorker) -> Optional[Job]:
        """Get next suitable job for worker.
        
        Args:
            worker: Worker requesting job
            
        Returns:
            Next job or None if no suitable job available
        """
        # Check if queue is empty
        if self.job_queue.empty():
            return None
        
        # Get job from queue
        try:
            job = self.job_queue.get_nowait()
            
            # Check if worker can handle this job
            if worker.can_handle(job):
                # Check if job is ready to run
                if job.scheduled_at and job.scheduled_at > datetime.now():
                    # Put job back and return None
                    await self.job_queue.put(job)
                    return None
                
                return job
            else:
                # Put job back for other workers
                await self.job_queue.put(job)
                return None
                
        except:
            return None
    
    async def _process_job_with_worker(self, worker: JobWorker, job: Job):
        """Process job with specific worker.
        
        Args:
            worker: Worker to use
            job: Job to process
        """
        worker.current_job = job
        self.active_jobs[job.job_id] = job
        
        start_time = datetime.now()
        
        try:
            # Process job with timeout
            result = await asyncio.wait_for(
                worker.process_job(job),
                timeout=job.timeout_seconds
            )
            
            # Record successful completion
            self.completed_jobs[job.job_id] = result
            worker.jobs_processed += 1
            self.stats["jobs_processed"] += 1
            self.stats["total_processing_time"] += result.execution_time
            
            # Execute callback if provided
            if job.callback:
                try:
                    await job.callback(result)
                except Exception as e:
                    logger.error(f"Job callback failed for {job.job_id}: {e}")
            
            logger.info(f"Completed job {job.job_id} in {result.execution_time:.2f}s")
            
        except asyncio.TimeoutError:
            # Job timed out
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=f"Job timed out after {job.timeout_seconds}s",
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now()
            )
            
            await self._handle_job_failure(worker, job, result)
            
        except Exception as e:
            # Job failed
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                started_at=start_time,
                completed_at=datetime.now()
            )
            
            await self._handle_job_failure(worker, job, result)
        
        finally:
            # Clean up
            worker.current_job = None
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _handle_job_failure(self, worker: JobWorker, job: Job, result: JobResult):
        """Handle job failure and retry logic.
        
        Args:
            worker: Worker that processed the job
            job: Failed job
            result: Failure result
        """
        worker.jobs_failed += 1
        self.stats["jobs_failed"] += 1
        
        # Check if retry is possible
        if job.retry_count < job.max_retries:
            job.retry_count += 1
            result.status = JobStatus.RETRYING
            
            # Reschedule with delay
            job.scheduled_at = datetime.now() + timedelta(seconds=2 ** job.retry_count)
            await self.job_queue.put(job)
            
            logger.warning(f"Retrying job {job.job_id} (attempt {job.retry_count}/{job.max_retries})")
        else:
            # Max retries exceeded
            self.failed_jobs[job.job_id] = result
            logger.error(f"Job {job.job_id} failed permanently: {result.error_message}")
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status or None if not found
        """
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": JobStatus.RUNNING.value,
                "job_type": job.job_type.value,
                "priority": job.priority.name,
                "created_at": job.created_at.isoformat(),
                "retry_count": job.retry_count
            }
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            result = self.completed_jobs[job_id]
            return {
                "job_id": job_id,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "completed_at": result.completed_at.isoformat() if result.completed_at else None
            }
        
        # Check failed jobs
        if job_id in self.failed_jobs:
            result = self.failed_jobs[job_id]
            return {
                "job_id": job_id,
                "status": result.status.value,
                "error_message": result.error_message,
                "execution_time": result.execution_time
            }
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Queue statistics
        """
        return {
            "queue_size": self.job_queue.qsize(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "registered_workers": len(self.workers),
            "running_workers": sum(1 for w in self.workers.values() if w.is_running),
            "is_running": self.is_running,
            "stats": self.stats,
            "worker_stats": [worker.get_stats() for worker in self.workers.values()]
        }
    
    async def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> Optional[JobResult]:
        """Wait for a job to complete.
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum time to wait
            
        Returns:
            Job result or None if timeout
        """
        start_time = time.time()
        
        while True:
            # Check if job is completed
            if job_id in self.completed_jobs:
                return self.completed_jobs[job_id]
            
            if job_id in self.failed_jobs:
                return self.failed_jobs[job_id]
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            # Brief pause
            await asyncio.sleep(0.1)
    
    async def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed and failed jobs.
        
        Args:
            max_age_hours: Maximum age of jobs to keep
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean completed jobs
        to_remove = []
        for job_id, result in self.completed_jobs.items():
            if result.completed_at and result.completed_at < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.completed_jobs[job_id]
        
        # Clean failed jobs
        to_remove = []
        for job_id, result in self.failed_jobs.items():
            if result.completed_at and result.completed_at < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.failed_jobs[job_id]
        
        logger.info(f"Cleaned up {len(to_remove)} old job records")