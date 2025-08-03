"""Event-driven workflow system for causal reasoning experiments."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import json
import uuid
import weakref
from collections import defaultdict

from ..core import CausalEnvironment
from ..metrics import BeliefState
from .scoring import CausalScore
from .workers import QueueManager, JobType, JobPriority

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events in the system."""
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED = "experiment_failed"
    BELIEF_UPDATED = "belief_updated"
    INTERVENTION_PERFORMED = "intervention_performed"
    SCORE_CALCULATED = "score_calculated"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    WORKER_FAILED = "worker_failed"
    SYSTEM_ERROR = "system_error"
    DATA_PROCESSED = "data_processed"
    NOTIFICATION_SENT = "notification_sent"
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Event:
    """System event."""
    event_id: str
    event_type: EventType
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)


@dataclass
class EventHandler:
    """Event handler registration."""
    handler_id: str
    event_types: Set[EventType]
    handler_func: Callable
    priority: int = 0
    async_handler: bool = True
    enabled: bool = True
    filter_func: Optional[Callable] = None


class EventBus:
    """Central event bus for managing event publishing and subscription."""
    
    def __init__(self):
        """Initialize event bus."""
        self.handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_history: List[Event] = []
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "handlers_executed": 0,
            "errors": 0
        }
        
        # Event filtering
        self.global_filters: List[Callable] = []
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict] = {}
        
    def subscribe(self, handler: EventHandler):
        """Subscribe an event handler.
        
        Args:
            handler: Event handler to register
        """
        for event_type in handler.event_types:
            self.handlers[event_type].append(handler)
            # Sort by priority (higher priority first)
            self.handlers[event_type].sort(key=lambda h: h.priority, reverse=True)
        
        logger.info(f"Registered event handler {handler.handler_id} for {len(handler.event_types)} event types")
    
    def unsubscribe(self, handler_id: str):
        """Unsubscribe an event handler.
        
        Args:
            handler_id: ID of handler to remove
        """
        removed_count = 0
        for event_type, handler_list in self.handlers.items():
            original_length = len(handler_list)
            self.handlers[event_type] = [h for h in handler_list if h.handler_id != handler_id]
            removed_count += original_length - len(self.handlers[event_type])
        
        logger.info(f"Unregistered event handler {handler_id} ({removed_count} registrations removed)")
    
    async def publish(self, event: Event) -> List[Any]:
        """Publish an event to all subscribers.
        
        Args:
            event: Event to publish
            
        Returns:
            List of handler results
        """
        self.stats["events_published"] += 1
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > 10000:  # Keep last 10k events
            self.event_history = self.event_history[-10000:]
        
        # Apply global filters
        if not self._passes_filters(event):
            return []
        
        # Check rate limiting
        if not self._check_rate_limit(event):
            logger.warning(f"Event {event.event_id} rate limited")
            return []
        
        # Get handlers for this event type
        handlers = self.handlers.get(event.event_type, [])
        
        # Execute handlers
        results = []
        for handler in handlers:
            if not handler.enabled:
                continue
            
            # Apply handler-specific filter
            if handler.filter_func and not handler.filter_func(event):
                continue
            
            try:
                if handler.async_handler:
                    result = await handler.handler_func(event)
                else:
                    result = handler.handler_func(event)
                
                results.append(result)
                self.stats["handlers_executed"] += 1
                
            except Exception as e:
                logger.error(f"Handler {handler.handler_id} failed for event {event.event_id}: {e}")
                self.stats["errors"] += 1
        
        self.stats["events_processed"] += 1
        
        logger.debug(f"Published event {event.event_id} to {len(results)} handlers")
        return results
    
    def add_global_filter(self, filter_func: Callable[[Event], bool]):
        """Add a global event filter.
        
        Args:
            filter_func: Function that returns True if event should be processed
        """
        self.global_filters.append(filter_func)
    
    def set_rate_limit(self, source: str, max_events: int, window_seconds: int):
        """Set rate limit for events from a source.
        
        Args:
            source: Event source
            max_events: Maximum events allowed
            window_seconds: Time window in seconds
        """
        self.rate_limits[source] = {
            "max_events": max_events,
            "window_seconds": window_seconds,
            "events": []
        }
    
    def _passes_filters(self, event: Event) -> bool:
        """Check if event passes global filters.
        
        Args:
            event: Event to check
            
        Returns:
            True if event passes all filters
        """
        for filter_func in self.global_filters:
            try:
                if not filter_func(event):
                    return False
            except Exception as e:
                logger.error(f"Global filter failed: {e}")
        
        return True
    
    def _check_rate_limit(self, event: Event) -> bool:
        """Check if event passes rate limiting.
        
        Args:
            event: Event to check
            
        Returns:
            True if event is within rate limits
        """
        if event.source not in self.rate_limits:
            return True
        
        limit_config = self.rate_limits[event.source]
        now = datetime.now()
        cutoff = now - timedelta(seconds=limit_config["window_seconds"])
        
        # Remove old events
        limit_config["events"] = [
            timestamp for timestamp in limit_config["events"]
            if timestamp > cutoff
        ]
        
        # Check if under limit
        if len(limit_config["events"]) >= limit_config["max_events"]:
            return False
        
        # Add current event
        limit_config["events"].append(now)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics.
        
        Returns:
            Event bus statistics
        """
        handler_count = sum(len(handlers) for handlers in self.handlers.values())
        
        return {
            "total_handlers": handler_count,
            "event_types_handled": len(self.handlers),
            "events_in_history": len(self.event_history),
            "stats": self.stats,
            "rate_limits_configured": len(self.rate_limits)
        }
    
    def get_recent_events(self, limit: int = 100, event_type: Optional[EventType] = None) -> List[Event]:
        """Get recent events.
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type (optional)
            
        Returns:
            List of recent events
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return events[-limit:]


class WorkflowEngine:
    """Event-driven workflow engine."""
    
    def __init__(self, event_bus: EventBus, queue_manager: Optional[QueueManager] = None):
        """Initialize workflow engine.
        
        Args:
            event_bus: Event bus for communication
            queue_manager: Optional queue manager for job processing
        """
        self.event_bus = event_bus
        self.queue_manager = queue_manager
        self.workflows: Dict[str, 'Workflow'] = {}
        self.active_workflows: Dict[str, 'WorkflowInstance'] = {}
        
        # Register built-in event handlers
        self._register_builtin_handlers()
    
    def _register_builtin_handlers(self):
        """Register built-in event handlers."""
        # Experiment lifecycle handlers
        self.event_bus.subscribe(EventHandler(
            handler_id="experiment_lifecycle",
            event_types={EventType.EXPERIMENT_STARTED, EventType.EXPERIMENT_COMPLETED, EventType.EXPERIMENT_FAILED},
            handler_func=self._handle_experiment_lifecycle,
            priority=100
        ))
        
        # Score calculation handler
        self.event_bus.subscribe(EventHandler(
            handler_id="score_processor",
            event_types={EventType.SCORE_CALCULATED},
            handler_func=self._handle_score_calculated,
            priority=50
        ))
        
        # Error handling
        self.event_bus.subscribe(EventHandler(
            handler_id="error_handler",
            event_types={EventType.SYSTEM_ERROR, EventType.WORKER_FAILED},
            handler_func=self._handle_system_error,
            priority=200
        ))
    
    async def _handle_experiment_lifecycle(self, event: Event):
        """Handle experiment lifecycle events."""
        experiment_id = event.data.get("experiment_id")
        
        if event.event_type == EventType.EXPERIMENT_STARTED:
            logger.info(f"Experiment {experiment_id} started")
            
            # Trigger analysis workflow if queue manager available
            if self.queue_manager:
                await self.queue_manager.create_and_submit_job(
                    job_type=JobType.EXPERIMENT_ANALYSIS,
                    payload=event.data,
                    priority=JobPriority.NORMAL
                )
        
        elif event.event_type == EventType.EXPERIMENT_COMPLETED:
            logger.info(f"Experiment {experiment_id} completed")
            
            # Send notification
            if self.queue_manager:
                await self.queue_manager.create_and_submit_job(
                    job_type=JobType.NOTIFICATION,
                    payload={
                        "type": "experiment_complete",
                        "experiment_id": experiment_id,
                        "recipients": event.data.get("notification_recipients", [])
                    },
                    priority=JobPriority.LOW
                )
        
        elif event.event_type == EventType.EXPERIMENT_FAILED:
            logger.error(f"Experiment {experiment_id} failed: {event.data.get('error')}")
            
            # Send failure notification
            if self.queue_manager:
                await self.queue_manager.create_and_submit_job(
                    job_type=JobType.NOTIFICATION,
                    payload={
                        "type": "experiment_failed",
                        "experiment_id": experiment_id,
                        "error": event.data.get("error"),
                        "recipients": event.data.get("notification_recipients", [])
                    },
                    priority=JobPriority.HIGH
                )
    
    async def _handle_score_calculated(self, event: Event):
        """Handle score calculation events."""
        score = event.data.get("score")
        experiment_id = event.data.get("experiment_id")
        
        if not score:
            return
        
        # Check for significant achievements
        if score.get("total_score", 0) > 0.9:
            # High score achievement
            await self.event_bus.publish(Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.THRESHOLD_EXCEEDED,
                source="workflow_engine",
                timestamp=datetime.now(),
                data={
                    "threshold_type": "high_score",
                    "experiment_id": experiment_id,
                    "score": score,
                    "message": "Exceptional causal reasoning performance achieved"
                },
                priority=EventPriority.HIGH,
                tags={"achievement", "high_performance"}
            ))
    
    async def _handle_system_error(self, event: Event):
        """Handle system errors."""
        error_type = event.data.get("error_type", "unknown")
        error_message = event.data.get("message", "No message")
        
        logger.error(f"System error ({error_type}): {error_message}")
        
        # Critical errors might need immediate attention
        if event.priority == EventPriority.CRITICAL:
            # Send urgent notification
            if self.queue_manager:
                await self.queue_manager.create_and_submit_job(
                    job_type=JobType.NOTIFICATION,
                    payload={
                        "type": "system_alert",
                        "severity": "critical",
                        "message": f"Critical system error: {error_message}",
                        "recipients": ["admin@system.com"]  # Configure as needed
                    },
                    priority=JobPriority.CRITICAL
                )
    
    def register_workflow(self, workflow: 'Workflow'):
        """Register a workflow.
        
        Args:
            workflow: Workflow to register
        """
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Registered workflow: {workflow.workflow_id}")
    
    async def start_workflow(self, workflow_id: str, initial_data: Dict[str, Any]) -> str:
        """Start a workflow instance.
        
        Args:
            workflow_id: ID of workflow to start
            initial_data: Initial workflow data
            
        Returns:
            Workflow instance ID
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        instance_id = str(uuid.uuid4())
        
        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow=workflow,
            data=initial_data.copy(),
            engine=self
        )
        
        self.active_workflows[instance_id] = instance
        
        # Start the workflow
        await instance.start()
        
        return instance_id
    
    def get_workflow_status(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow instance status.
        
        Args:
            instance_id: Workflow instance ID
            
        Returns:
            Workflow status or None if not found
        """
        if instance_id not in self.active_workflows:
            return None
        
        instance = self.active_workflows[instance_id]
        return {
            "instance_id": instance_id,
            "workflow_id": instance.workflow.workflow_id,
            "status": instance.status.value,
            "current_step": instance.current_step,
            "started_at": instance.started_at.isoformat() if instance.started_at else None,
            "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
            "data": instance.data
        }


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStep(ABC):
    """Abstract workflow step."""
    
    def __init__(self, step_id: str, description: str):
        """Initialize workflow step.
        
        Args:
            step_id: Unique step identifier
            description: Step description
        """
        self.step_id = step_id
        self.description = description
    
    @abstractmethod
    async def execute(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step.
        
        Args:
            data: Workflow data
            context: Execution context
            
        Returns:
            Updated workflow data
        """
        pass


class EventTriggerStep(WorkflowStep):
    """Workflow step that triggers an event."""
    
    def __init__(self, step_id: str, event_type: EventType, event_data_template: Dict[str, Any]):
        """Initialize event trigger step.
        
        Args:
            step_id: Step identifier
            event_type: Type of event to trigger
            event_data_template: Template for event data
        """
        super().__init__(step_id, f"Trigger {event_type.value} event")
        self.event_type = event_type
        self.event_data_template = event_data_template
    
    async def execute(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute event trigger step."""
        # Build event data from template and workflow data
        event_data = {}
        for key, value in self.event_data_template.items():
            if isinstance(value, str) and value.startswith("$"):
                # Template variable
                var_name = value[1:]
                event_data[key] = data.get(var_name, value)
            else:
                event_data[key] = value
        
        # Create and publish event
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=self.event_type,
            source=f"workflow_{context.get('workflow_id', 'unknown')}",
            timestamp=datetime.now(),
            data=event_data
        )
        
        engine = context.get("engine")
        if engine:
            await engine.event_bus.publish(event)
        
        return data


class JobSubmissionStep(WorkflowStep):
    """Workflow step that submits a job."""
    
    def __init__(self, step_id: str, job_type: JobType, job_priority: JobPriority = JobPriority.NORMAL):
        """Initialize job submission step.
        
        Args:
            step_id: Step identifier
            job_type: Type of job to submit
            job_priority: Job priority
        """
        super().__init__(step_id, f"Submit {job_type.value} job")
        self.job_type = job_type
        self.job_priority = job_priority
    
    async def execute(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute job submission step."""
        engine = context.get("engine")
        
        if not engine or not engine.queue_manager:
            raise ValueError("No queue manager available for job submission")
        
        # Submit job with workflow data as payload
        job_id = await engine.queue_manager.create_and_submit_job(
            job_type=self.job_type,
            payload=data.copy(),
            priority=self.job_priority
        )
        
        # Add job ID to workflow data
        data["submitted_job_id"] = job_id
        
        return data


class ConditionalStep(WorkflowStep):
    """Workflow step with conditional execution."""
    
    def __init__(self, step_id: str, condition_func: Callable, true_step: WorkflowStep, false_step: Optional[WorkflowStep] = None):
        """Initialize conditional step.
        
        Args:
            step_id: Step identifier
            condition_func: Function that evaluates condition
            true_step: Step to execute if condition is true
            false_step: Step to execute if condition is false (optional)
        """
        super().__init__(step_id, "Conditional execution")
        self.condition_func = condition_func
        self.true_step = true_step
        self.false_step = false_step
    
    async def execute(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute conditional step."""
        condition_result = self.condition_func(data, context)
        
        if condition_result and self.true_step:
            return await self.true_step.execute(data, context)
        elif not condition_result and self.false_step:
            return await self.false_step.execute(data, context)
        
        return data


class Workflow:
    """Workflow definition."""
    
    def __init__(self, workflow_id: str, name: str, description: str, steps: List[WorkflowStep]):
        """Initialize workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            name: Workflow name
            description: Workflow description
            steps: List of workflow steps
        """
        self.workflow_id = workflow_id
        self.name = name
        self.description = description
        self.steps = steps


class WorkflowInstance:
    """Running instance of a workflow."""
    
    def __init__(self, instance_id: str, workflow: Workflow, data: Dict[str, Any], engine: WorkflowEngine):
        """Initialize workflow instance.
        
        Args:
            instance_id: Instance identifier
            workflow: Workflow definition
            data: Initial workflow data
            engine: Workflow engine
        """
        self.instance_id = instance_id
        self.workflow = workflow
        self.data = data
        self.engine = engine
        
        self.status = WorkflowStatus.PENDING
        self.current_step = 0
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error_message: Optional[str] = None
    
    async def start(self):
        """Start workflow execution."""
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.now()
        
        try:
            context = {
                "workflow_id": self.workflow.workflow_id,
                "instance_id": self.instance_id,
                "engine": self.engine
            }
            
            # Execute steps sequentially
            for i, step in enumerate(self.workflow.steps):
                self.current_step = i
                
                logger.debug(f"Executing workflow step {i}: {step.step_id}")
                
                self.data = await step.execute(self.data, context)
            
            self.status = WorkflowStatus.COMPLETED
            self.completed_at = datetime.now()
            
            logger.info(f"Workflow {self.workflow.workflow_id} completed successfully")
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.completed_at = datetime.now()
            self.error_message = str(e)
            
            logger.error(f"Workflow {self.workflow.workflow_id} failed: {e}")
            
            # Publish failure event
            await self.engine.event_bus.publish(Event(
                event_id=str(uuid.uuid4()),
                event_type=EventType.SYSTEM_ERROR,
                source=f"workflow_{self.workflow.workflow_id}",
                timestamp=datetime.now(),
                data={
                    "error_type": "workflow_failure",
                    "workflow_id": self.workflow.workflow_id,
                    "instance_id": self.instance_id,
                    "error": str(e)
                },
                priority=EventPriority.HIGH
            ))


# Example workflow factory functions

def create_experiment_analysis_workflow() -> Workflow:
    """Create a workflow for experiment analysis."""
    steps = [
        EventTriggerStep(
            step_id="trigger_analysis_start",
            event_type=EventType.DATA_PROCESSED,
            event_data_template={
                "experiment_id": "$experiment_id",
                "stage": "analysis_started"
            }
        ),
        JobSubmissionStep(
            step_id="submit_analysis_job",
            job_type=JobType.EXPERIMENT_ANALYSIS,
            job_priority=JobPriority.NORMAL
        ),
        ConditionalStep(
            step_id="check_score_threshold",
            condition_func=lambda data, ctx: data.get("final_score", {}).get("total_score", 0) > 0.8,
            true_step=EventTriggerStep(
                step_id="trigger_high_score",
                event_type=EventType.THRESHOLD_EXCEEDED,
                event_data_template={
                    "threshold_type": "high_score",
                    "experiment_id": "$experiment_id",
                    "score": "$final_score"
                }
            )
        ),
        JobSubmissionStep(
            step_id="submit_notification",
            job_type=JobType.NOTIFICATION,
            job_priority=JobPriority.LOW
        )
    ]
    
    return Workflow(
        workflow_id="experiment_analysis",
        name="Experiment Analysis Workflow",
        description="Complete workflow for analyzing causal reasoning experiments",
        steps=steps
    )


def create_monitoring_workflow() -> Workflow:
    """Create a workflow for system monitoring."""
    steps = [
        JobSubmissionStep(
            step_id="check_system_health",
            job_type=JobType.DATA_PROCESSING,
            job_priority=JobPriority.LOW
        ),
        ConditionalStep(
            step_id="check_for_issues",
            condition_func=lambda data, ctx: data.get("health_status") != "healthy",
            true_step=EventTriggerStep(
                step_id="trigger_alert",
                event_type=EventType.SYSTEM_ERROR,
                event_data_template={
                    "error_type": "health_check_failed",
                    "status": "$health_status",
                    "details": "$health_details"
                }
            )
        )
    ]
    
    return Workflow(
        workflow_id="system_monitoring",
        name="System Monitoring Workflow",
        description="Monitors system health and triggers alerts",
        steps=steps
    )