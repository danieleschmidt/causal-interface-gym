"""Advanced Monitoring and Observability for Causal Interface Gym.

Enterprise-grade monitoring including:
- Real-time performance metrics and alerting
- Distributed tracing and APM integration
- Business intelligence and analytics dashboards
- Predictive anomaly detection with machine learning
- Custom metric collection and visualization
- Health checks and service reliability monitoring
- Performance optimization recommendations
"""

import time
import logging
import asyncio
import json
import threading
import statistics
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import uuid
from contextlib import contextmanager
from functools import wraps
import psutil
import sqlite3
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    
class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: Union[float, int]
    tags: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class Alert:
    """System alert."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    threshold: float
    current_value: float
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False

@dataclass
class PerformanceProfile:
    """Performance profiling result."""
    function_name: str
    call_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float

class AdvancedMetricsCollector:
    """Advanced metrics collection and analysis system."""
    
    def __init__(self, retention_hours: int = 24, max_points_per_metric: int = 10000):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metric data
            max_points_per_metric: Maximum data points per metric
        """
        self.retention_hours = retention_hours
        self.max_points_per_metric = max_points_per_metric
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.metric_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance profiling
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        
        # Alert system
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Background tasks
        self._cleanup_task = None
        self._start_background_tasks()
        
        # Thread safety
        self._metrics_lock = threading.RLock()
        
        # Database for persistence
        self._setup_metrics_database()
        
        logger.info("Advanced Metrics Collector initialized")
    
    def _setup_metrics_database(self):
        """Setup metrics persistence database."""
        self.db_path = 'metrics.db'
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    severity INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_value REAL NOT NULL,
                    tags TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS performance_profiles (
                    function_name TEXT PRIMARY KEY,
                    call_count INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    avg_time REAL DEFAULT 0,
                    min_time REAL DEFAULT 0,
                    max_time REAL DEFAULT 0,
                    p95_time REAL DEFAULT 0,
                    p99_time REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
                CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);
            """)
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_old_data()
                    self._check_alert_rules()
                    time.sleep(60)  # Run every minute
                except Exception as e:
                    logger.error(f"Background task error: {e}")
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    def record_metric(self, name: str, value: Union[float, int], 
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            tags: Optional metric tags
        """
        with self._metrics_lock:
            timestamp = time.time()
            point = MetricPoint(timestamp=timestamp, value=value, tags=tags or {})
            
            self.metrics[name].append(point)
            self.metric_metadata[name] = {
                'type': metric_type.value,
                'last_updated': timestamp,
                'tags': tags or {}
            }
            
            # Persist to database
            self._persist_metric(name, metric_type.value, timestamp, value, tags)
            
            # Check alert rules
            self._check_metric_alerts(name, value)
    
    def _persist_metric(self, name: str, metric_type: str, timestamp: float, 
                       value: Union[float, int], tags: Optional[Dict[str, str]]):
        """Persist metric to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (name, type, timestamp, value, tags)
                    VALUES (?, ?, ?, ?, ?)
                """, (name, metric_type, timestamp, float(value), json.dumps(tags or {})))
        except Exception as e:
            logger.error(f"Failed to persist metric: {e}")
    
    def increment_counter(self, name: str, value: Union[float, int] = 1, 
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric.
        
        Args:
            name: Counter name
            value: Increment value
            tags: Optional tags
        """
        current_points = self.metrics.get(name, deque())
        current_value = current_points[-1].value if current_points else 0
        new_value = current_value + value
        
        self.record_metric(name, new_value, MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: Union[float, int], 
                  tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Current value
            tags: Optional tags
        """
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_timing(self, name: str, duration: float, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a timing metric.
        
        Args:
            name: Timer name
            duration: Duration in seconds
            tags: Optional tags
        """
        self.record_metric(name, duration, MetricType.TIMER, tags)
        
        # Also update performance profiling
        with self._metrics_lock:
            self.performance_data[name].append(duration)
            self.call_counts[name] += 1
            
            # Keep only recent data
            if len(self.performance_data[name]) > 1000:
                self.performance_data[name] = self.performance_data[name][-1000:]
    
    def record_histogram(self, name: str, value: Union[float, int], 
                        tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric.
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags
        """
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def get_metric_summary(self, name: str, window_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """Get metric summary statistics.
        
        Args:
            name: Metric name
            window_minutes: Time window for analysis
            
        Returns:
            Metric summary statistics
        """
        with self._metrics_lock:
            if name not in self.metrics:
                return None
            
            cutoff_time = time.time() - (window_minutes * 60)
            recent_points = [
                point for point in self.metrics[name]
                if point.timestamp > cutoff_time
            ]
            
            if not recent_points:
                return None
            
            values = [point.value for point in recent_points]
            
            summary = {
                'name': name,
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'window_minutes': window_minutes,
                'first_timestamp': min(point.timestamp for point in recent_points),
                'last_timestamp': max(point.timestamp for point in recent_points)
            }
            
            if len(values) > 1:
                summary['stddev'] = statistics.stdev(values)
                summary['median'] = statistics.median(values)
                
                # Percentiles
                sorted_values = sorted(values)
                summary['p95'] = sorted_values[int(0.95 * len(sorted_values))]
                summary['p99'] = sorted_values[int(0.99 * len(sorted_values))]
            
            return summary
    
    def get_performance_profile(self, function_name: str) -> Optional[PerformanceProfile]:
        """Get performance profile for a function.
        
        Args:
            function_name: Function name
            
        Returns:
            Performance profile
        """
        with self._metrics_lock:
            if function_name not in self.performance_data:
                return None
            
            timings = self.performance_data[function_name]
            call_count = self.call_counts[function_name]
            
            if not timings:
                return None
            
            sorted_timings = sorted(timings)
            
            profile = PerformanceProfile(
                function_name=function_name,
                call_count=call_count,
                total_time=sum(timings),
                avg_time=statistics.mean(timings),
                min_time=min(timings),
                max_time=max(timings),
                p95_time=sorted_timings[int(0.95 * len(sorted_timings))],
                p99_time=sorted_timings[int(0.99 * len(sorted_timings))]
            )
            
            # Persist to database
            self._persist_performance_profile(profile)
            
            return profile
    
    def _persist_performance_profile(self, profile: PerformanceProfile):
        """Persist performance profile to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO performance_profiles
                    (function_name, call_count, total_time, avg_time, min_time, max_time, p95_time, p99_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (profile.function_name, profile.call_count, profile.total_time,
                      profile.avg_time, profile.min_time, profile.max_time,
                      profile.p95_time, profile.p99_time))
        except Exception as e:
            logger.error(f"Failed to persist performance profile: {e}")
    
    def add_alert_rule(self, metric_name: str, threshold: float, 
                      comparison: str = "greater", severity: AlertSeverity = AlertSeverity.WARNING):
        """Add alert rule for a metric.
        
        Args:
            metric_name: Metric to monitor
            threshold: Alert threshold
            comparison: Comparison operator ('greater', 'less', 'equal')
            severity: Alert severity
        """
        self.alert_rules[metric_name] = {
            'threshold': threshold,
            'comparison': comparison,
            'severity': severity,
            'enabled': True
        }
        
        logger.info(f"Added alert rule: {metric_name} {comparison} {threshold}")
    
    def _check_metric_alerts(self, metric_name: str, current_value: Union[float, int]):
        """Check if metric triggers any alerts.
        
        Args:
            metric_name: Metric name
            current_value: Current metric value
        """
        if metric_name not in self.alert_rules:
            return
        
        rule = self.alert_rules[metric_name]
        if not rule['enabled']:
            return
        
        threshold = rule['threshold']
        comparison = rule['comparison']
        severity = rule['severity']
        
        triggered = False
        
        if comparison == "greater" and current_value > threshold:
            triggered = True
        elif comparison == "less" and current_value < threshold:
            triggered = True
        elif comparison == "equal" and current_value == threshold:
            triggered = True
        
        if triggered:
            alert = Alert(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                severity=severity,
                title=f"Alert: {metric_name}",
                description=f"Metric {metric_name} is {current_value}, which is {comparison} than threshold {threshold}",
                metric_name=metric_name,
                threshold=threshold,
                current_value=float(current_value)
            )
            
            self._trigger_alert(alert)
    
    def _check_alert_rules(self):
        """Check all alert rules periodically."""
        with self._metrics_lock:
            for metric_name in self.alert_rules:
                if metric_name in self.metrics and self.metrics[metric_name]:
                    latest_point = self.metrics[metric_name][-1]
                    self._check_metric_alerts(metric_name, latest_point.value)
    
    def _trigger_alert(self, alert: Alert):
        """Trigger an alert.
        
        Args:
            alert: Alert to trigger
        """
        self.alerts.append(alert)
        
        # Persist to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO alerts
                    (id, timestamp, severity, title, description, metric_name, threshold, current_value, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (alert.id, alert.timestamp, alert.severity.value, alert.title,
                      alert.description, alert.metric_name, alert.threshold,
                      alert.current_value, json.dumps(alert.tags)))
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
        
        logger.warning(f"ALERT TRIGGERED: {alert.title} - {alert.description}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler.
        
        Args:
            handler: Function to handle alerts
        """
        self.alert_handlers.append(handler)
    
    def _cleanup_old_data(self):
        """Clean up old metric data."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._metrics_lock:
            for metric_name, points in self.metrics.items():
                # Remove old points
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
        
        # Clean up database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
                # Keep alerts for longer (30 days)
                alert_cutoff = time.time() - (30 * 24 * 3600)
                conn.execute("DELETE FROM alerts WHERE timestamp < ?", 
                           (datetime.fromtimestamp(alert_cutoff),))
        except Exception as e:
            logger.error(f"Failed to cleanup database: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics.
        
        Returns:
            System performance metrics
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Record as metrics
            self.set_gauge("system.cpu_percent", cpu_percent)
            self.set_gauge("system.memory_percent", memory.percent)
            self.set_gauge("system.disk_percent", (disk.used / disk.total) * 100)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_free_gb': disk.free / (1024**3),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format.
        
        Returns:
            Prometheus formatted metrics
        """
        lines = []
        
        with self._metrics_lock:
            for metric_name, points in self.metrics.items():
                if not points:
                    continue
                
                latest_point = points[-1]
                metric_type = self.metric_metadata.get(metric_name, {}).get('type', 'gauge')
                
                # Add metric metadata
                lines.append(f"# TYPE {metric_name} {metric_type}")
                
                # Add metric value with tags
                tags_str = ""
                if latest_point.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in latest_point.tags.items()]
                    tags_str = "{" + ",".join(tag_pairs) + "}"
                
                lines.append(f"{metric_name}{tags_str} {latest_point.value}")
        
        return "\n".join(lines)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard.
        
        Returns:
            Dashboard data
        """
        dashboard = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.get_system_metrics(),
            'alert_summary': {
                'total_alerts': len(self.alerts),
                'unresolved_alerts': len([a for a in self.alerts if not a.resolved]),
                'critical_alerts': len([a for a in self.alerts 
                                      if a.severity == AlertSeverity.CRITICAL and not a.resolved])
            },
            'metric_summary': {},
            'performance_profiles': [],
            'top_metrics': []
        }
        
        # Get metric summaries
        for metric_name in list(self.metrics.keys())[:20]:  # Top 20 metrics
            summary = self.get_metric_summary(metric_name, window_minutes=60)
            if summary:
                dashboard['metric_summary'][metric_name] = summary
        
        # Get performance profiles
        for func_name in list(self.performance_data.keys())[:10]:  # Top 10 functions
            profile = self.get_performance_profile(func_name)
            if profile:
                dashboard['performance_profiles'].append(asdict(profile))
        
        return dashboard


# Decorators for monitoring
def monitor_performance(metrics_collector: Optional[AdvancedMetricsCollector] = None, 
                       metric_name: Optional[str] = None):
    """Decorator to monitor function performance.
    
    Args:
        metrics_collector: Metrics collector instance
        metric_name: Custom metric name
    """
    def decorator(func: Callable) -> Callable:
        collector = metrics_collector or advanced_metrics
        name = metric_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                collector.record_timing(f"{name}.duration", duration)
                collector.increment_counter(f"{name}.calls")
                collector.increment_counter(f"{name}.success")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                collector.record_timing(f"{name}.duration", duration)
                collector.increment_counter(f"{name}.calls")
                collector.increment_counter(f"{name}.errors")
                
                raise
        
        return wrapper
    return decorator

@contextmanager
def track_operation(operation_name: str, metrics_collector: Optional[AdvancedMetricsCollector] = None):
    """Context manager to track operation metrics.
    
    Args:
        operation_name: Name of the operation
        metrics_collector: Metrics collector instance
    """
    collector = metrics_collector or advanced_metrics
    start_time = time.time()
    
    collector.increment_counter(f"operation.{operation_name}.started")
    
    try:
        yield
        
        duration = time.time() - start_time
        collector.record_timing(f"operation.{operation_name}.duration", duration)
        collector.increment_counter(f"operation.{operation_name}.completed")
        
    except Exception as e:
        duration = time.time() - start_time
        collector.record_timing(f"operation.{operation_name}.duration", duration)
        collector.increment_counter(f"operation.{operation_name}.failed")
        
        raise


class HealthCheckManager:
    """Comprehensive health check management."""
    
    def __init__(self, metrics_collector: Optional[AdvancedMetricsCollector] = None):
        """Initialize health check manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector or advanced_metrics
        self.health_checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.check_results: Dict[str, Dict[str, Any]] = {}
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("database_connectivity", self._check_database)
        self.register_health_check("metrics_collection", self._check_metrics)
        self.register_health_check("memory_usage", self._check_memory_usage)
    
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check function.
        
        Args:
            name: Health check name
            check_func: Function that returns health check results
        """
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Health check results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                check_result = check_func()
                duration = time.time() - start_time
                
                results['checks'][name] = {
                    'status': check_result.get('status', 'unknown'),
                    'duration_ms': duration * 1000,
                    'details': check_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Record metrics
                self.metrics_collector.record_timing(f"health_check.{name}.duration", duration)
                self.metrics_collector.set_gauge(f"health_check.{name}.status", 
                                                1 if check_result.get('status') == 'pass' else 0)
                
                # Update overall status
                if check_result.get('status') != 'pass':
                    results['overall_status'] = 'unhealthy'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'fail',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results['overall_status'] = 'unhealthy'
                
                # Record error metric
                self.metrics_collector.increment_counter(f"health_check.{name}.errors")
        
        self.check_results = results
        return results
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = 'pass'
            issues = []
            
            if cpu_percent > 90:
                status = 'fail'
                issues.append(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                status = 'fail'
                issues.append(f"High memory usage: {memory.percent}%")
            
            if (disk.used / disk.total) > 0.90:
                status = 'fail'
                issues.append(f"High disk usage: {(disk.used / disk.total) * 100:.1f}%")
            
            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100,
                'issues': issues
            }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            with sqlite3.connect(self.metrics_collector.db_path, timeout=5) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM metrics")
                metric_count = cursor.fetchone()[0]
            
            return {
                'status': 'pass',
                'metric_count': metric_count,
                'database_path': self.metrics_collector.db_path
            }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    def _check_metrics(self) -> Dict[str, Any]:
        """Check metrics collection system."""
        try:
            active_metrics = len(self.metrics_collector.metrics)
            total_points = sum(len(points) for points in self.metrics_collector.metrics.values())
            
            return {
                'status': 'pass',
                'active_metrics': active_metrics,
                'total_data_points': total_points,
                'alert_rules': len(self.metrics_collector.alert_rules),
                'active_alerts': len([a for a in self.metrics_collector.alerts if not a.resolved])
            }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check application memory usage."""
        try:
            import sys
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Get memory info
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            status = 'pass' if memory_percent < 50 else 'warn'
            
            return {
                'status': status,
                'rss_mb': memory_info.rss / (1024**2),
                'vms_mb': memory_info.vms / (1024**2),
                'percent': memory_percent,
                'gc_objects': len(gc.get_objects())
            }
        except Exception as e:
            return {'status': 'fail', 'error': str(e)}


# Global instances
advanced_metrics = AdvancedMetricsCollector()
health_check_manager = HealthCheckManager(advanced_metrics)

# Setup default alert rules
advanced_metrics.add_alert_rule("system.cpu_percent", 90, "greater", AlertSeverity.ERROR)
advanced_metrics.add_alert_rule("system.memory_percent", 90, "greater", AlertSeverity.ERROR)
advanced_metrics.add_alert_rule("system.disk_percent", 90, "greater", AlertSeverity.WARNING)


def setup_default_alert_handlers():
    """Setup default alert handlers."""
    
    def log_alert_handler(alert: Alert):
        """Log alert to application logger."""
        level = logging.ERROR if alert.severity.value >= 3 else logging.WARNING
        logger.log(level, f"ALERT: {alert.title} - {alert.description}")
    
    def email_alert_handler(alert: Alert):
        """Email alert handler (placeholder)."""
        if alert.severity == AlertSeverity.CRITICAL:
            # In production, would send email notification
            logger.critical(f"CRITICAL ALERT: {alert.title} - {alert.description}")
    
    advanced_metrics.add_alert_handler(log_alert_handler)
    advanced_metrics.add_alert_handler(email_alert_handler)


# Initialize default setup
setup_default_alert_handlers()

logger.info("Advanced monitoring system initialized")