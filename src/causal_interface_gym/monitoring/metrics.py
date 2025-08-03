"""Metrics collection and monitoring."""

import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary,
        CollectorRegistry, generate_latest,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Metric value with timestamp."""
    value: Union[int, float]
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Basic metrics collector for custom metrics."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metric values to keep in memory
        """
        self.max_history = max_history
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Metric name
            value: Value to increment by
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value.
        
        Args:
            name: Metric name
            value: Value to record
            labels: Optional labels
        """
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(MetricValue(value, labels=labels or {}))
    
    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.
        
        Args:
            name: Timer metric name
            labels: Optional labels
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            with self.lock:
                key = self._make_key(name, labels)
                self.timers[key].append(MetricValue(duration, labels=labels or {}))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics.
        
        Returns:
            Dictionary of all metrics
        """
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: [{
                        "value": mv.value,
                        "timestamp": mv.timestamp.isoformat(),
                        "labels": mv.labels
                    } for mv in values]
                    for name, values in self.histograms.items()
                },
                "timers": {
                    name: [{
                        "duration_seconds": mv.value,
                        "timestamp": mv.timestamp.isoformat(),
                        "labels": mv.labels
                    } for mv in values]
                    for name, values in self.timers.items()
                },
                "timestamp": datetime.now().isoformat()
            }
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Summary statistics
        """
        summary = {"name": name, "type": "unknown", "data": {}}
        
        with self.lock:
            # Check different metric types
            if name in self.counters:
                summary["type"] = "counter"
                summary["data"] = {"value": self.counters[name]}
            
            elif name in self.gauges:
                summary["type"] = "gauge"
                summary["data"] = {"value": self.gauges[name]}
            
            elif name in self.histograms:
                summary["type"] = "histogram"
                values = [mv.value for mv in self.histograms[name]]
                if values:
                    summary["data"] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "latest": values[-1] if values else None
                    }
            
            elif name in self.timers:
                summary["type"] = "timer"
                values = [mv.value for mv in self.timers[name]]
                if values:
                    summary["data"] = {
                        "count": len(values),
                        "min_seconds": min(values),
                        "max_seconds": max(values),
                        "mean_seconds": sum(values) / len(values),
                        "total_seconds": sum(values),
                        "latest_seconds": values[-1] if values else None
                    }
        
        return summary
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a key for metric storage.
        
        Args:
            name: Metric name
            labels: Optional labels
            
        Returns:
            Storage key
        """
        if not labels:
            return name
        
        label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{name}{{{label_str}}}"


class PrometheusMetrics:
    """Prometheus metrics integration."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize Prometheus metrics.
        
        Args:
            registry: Optional custom registry
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client is required for Prometheus metrics")
        
        self.registry = registry or CollectorRegistry()
        
        # Define application metrics
        self.experiment_counter = Counter(
            'causal_gym_experiments_total',
            'Total number of causal reasoning experiments',
            ['agent_type', 'status'],
            registry=self.registry
        )
        
        self.experiment_duration = Histogram(
            'causal_gym_experiment_duration_seconds',
            'Duration of causal reasoning experiments',
            ['agent_type'],
            registry=self.registry
        )
        
        self.causal_score = Histogram(
            'causal_gym_causal_score',
            'Causal reasoning scores',
            ['agent_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
        
        self.llm_requests = Counter(
            'causal_gym_llm_requests_total',
            'Total LLM API requests',
            ['provider', 'model', 'status'],
            registry=self.registry
        )
        
        self.llm_response_time = Histogram(
            'causal_gym_llm_response_time_seconds',
            'LLM API response times',
            ['provider', 'model'],
            registry=self.registry
        )
        
        self.belief_extractions = Counter(
            'causal_gym_belief_extractions_total',
            'Total belief extractions from LLM responses',
            ['extraction_method', 'success'],
            registry=self.registry
        )
        
        self.database_operations = Counter(
            'causal_gym_database_operations_total',
            'Total database operations',
            ['operation_type', 'table', 'status'],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'causal_gym_cache_operations_total',
            'Total cache operations',
            ['operation_type', 'status'],
            registry=self.registry
        )
        
        self.active_experiments = Gauge(
            'causal_gym_active_experiments',
            'Number of currently active experiments',
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_experiment(self, agent_type: str, duration: float, 
                         causal_score: float, success: bool) -> None:
        """Record experiment metrics.
        
        Args:
            agent_type: Type of agent used
            duration: Experiment duration in seconds
            causal_score: Causal reasoning score (0-1)
            success: Whether experiment succeeded
        """
        status = "success" if success else "failure"
        
        self.experiment_counter.labels(
            agent_type=agent_type,
            status=status
        ).inc()
        
        if success:
            self.experiment_duration.labels(agent_type=agent_type).observe(duration)
            self.causal_score.labels(agent_type=agent_type).observe(causal_score)
    
    def record_llm_request(self, provider: str, model: str, 
                          response_time: float, success: bool) -> None:
        """Record LLM request metrics.
        
        Args:
            provider: LLM provider name
            model: Model name
            response_time: Response time in seconds
            success: Whether request succeeded
        """
        status = "success" if success else "failure"
        
        self.llm_requests.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        if success:
            self.llm_response_time.labels(
                provider=provider,
                model=model
            ).observe(response_time)
    
    def record_belief_extraction(self, method: str, success: bool) -> None:
        """Record belief extraction metrics.
        
        Args:
            method: Extraction method used
            success: Whether extraction succeeded
        """
        self.belief_extractions.labels(
            extraction_method=method,
            success="success" if success else "failure"
        ).inc()
    
    def record_database_operation(self, operation: str, table: str, success: bool) -> None:
        """Record database operation metrics.
        
        Args:
            operation: Operation type (select, insert, update, delete)
            table: Table name
            success: Whether operation succeeded
        """
        self.database_operations.labels(
            operation_type=operation,
            table=table,
            status="success" if success else "failure"
        ).inc()
    
    def record_cache_operation(self, operation: str, success: bool) -> None:
        """Record cache operation metrics.
        
        Args:
            operation: Operation type (get, set, delete)
            success: Whether operation succeeded
        """
        self.cache_operations.labels(
            operation_type=operation,
            status="success" if success else "failure"
        ).inc()
    
    def set_active_experiments(self, count: int) -> None:
        """Set number of active experiments.
        
        Args:
            count: Number of active experiments
        """
        self.active_experiments.set(count)
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format.
        
        Returns:
            Prometheus metrics text
        """
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get content type for metrics endpoint.
        
        Returns:
            Content type string
        """
        return CONTENT_TYPE_LATEST