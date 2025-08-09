"""Monitoring and observability utilities for causal interface gym."""

import time
import logging
import functools
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('causal_interface_gym.log')
    ]
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect application metrics for monitoring."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            'experiments_run': 0,
            'interventions_applied': 0,
            'environments_created': 0,
            'ui_components_created': 0,
            'errors': 0,
            'response_times': [],
            'active_sessions': 0
        }
        self.start_time = time.time()
    
    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric."""
        if metric in self.metrics:
            self.metrics[metric] += value
        else:
            self.metrics[metric] = value
    
    def record_time(self, metric: str, duration: float) -> None:
        """Record a timing metric."""
        timing_key = f"{metric}_times"
        if timing_key not in self.metrics:
            self.metrics[timing_key] = []
        self.metrics[timing_key].append(duration)
    
    def set_gauge(self, metric: str, value: Any) -> None:
        """Set a gauge metric."""
        self.metrics[metric] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        uptime = time.time() - self.start_time
        return {
            **self.metrics,
            'uptime_seconds': uptime,
            'timestamp': time.time()
        }
    
    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        metrics = self.get_metrics()
        lines = []
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lines.append(f"causal_gym_{key} {value}")
        
        return '\n'.join(lines)


# Global metrics collector instance
metrics = MetricsCollector()


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper function."""
        start_time = time.time()
        function_name = f"{func.__module__}.{func.__name__}"
        
        try:
            logger.info(f"Starting {function_name}", extra={
                'function': function_name,
                'args_count': len(args),
                'kwargs_count': len(kwargs)
            })
            
            result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            metrics.record_time(function_name, duration)
            
            logger.info(f"Completed {function_name}", extra={
                'function': function_name,
                'duration': duration,
                'success': True
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.increment('errors')
            
            logger.error(f"Error in {function_name}: {str(e)}", extra={
                'function': function_name,
                'duration': duration,
                'error': str(e),
                'success': False
            })
            raise
    
    return wrapper


@contextmanager
def track_experiment(experiment_name: str, **metadata):
    """Context manager to track experiment execution."""
    start_time = time.time()
    experiment_id = f"{experiment_name}_{int(start_time)}"
    
    logger.info(f"Starting experiment: {experiment_name}", extra={
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        **metadata
    })
    
    metrics.increment('experiments_run')
    metrics.increment('active_sessions')
    
    try:
        yield experiment_id
        
        duration = time.time() - start_time
        logger.info(f"Experiment completed: {experiment_name}", extra={
            'experiment_id': experiment_id,
            'duration': duration,
            'success': True
        })
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Experiment failed: {experiment_name}", extra={
            'experiment_id': experiment_id,
            'duration': duration,
            'error': str(e),
            'success': False
        })
        raise
    
    finally:
        metrics.increment('active_sessions', -1)


class HealthChecker:
    """Health check utilities for the application."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            'timestamp': time.time(),
            'status': 'healthy',
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                check_result = check_func()
                duration = time.time() - start_time
                
                results['checks'][name] = {
                    'status': 'pass',
                    'duration': duration,
                    'details': check_result
                }
                
            except Exception as e:
                results['checks'][name] = {
                    'status': 'fail',
                    'error': str(e)
                }
                results['status'] = 'unhealthy'
        
        return results


# Global health checker instance
health_checker = HealthChecker()


# Register default health checks
def check_basic_functionality():
    """Basic functionality health check."""
    from .core import CausalEnvironment
    
    # Test basic environment creation
    env = CausalEnvironment.from_dag({"test": []})
    result = env.intervene(test="health_check")
    
    return {
        'environment_creation': True,
        'intervention_working': 'intervention_applied' in result
    }


def check_memory_usage():
    """Memory usage health check."""
    import psutil
    
    memory = psutil.virtual_memory()
    return {
        'memory_percent': memory.percent,
        'memory_available_mb': memory.available / (1024 * 1024),
        'status': 'ok' if memory.percent < 90 else 'warning'
    }


def check_disk_space():
    """Disk space health check."""
    import psutil
    
    disk = psutil.disk_usage('/')
    percent_used = (disk.used / disk.total) * 100
    
    return {
        'disk_percent_used': percent_used,
        'disk_free_gb': disk.free / (1024 ** 3),
        'status': 'ok' if percent_used < 90 else 'warning'
    }


# Register default health checks
health_checker.register_check('basic_functionality', check_basic_functionality)

try:
    import psutil
    health_checker.register_check('memory_usage', check_memory_usage)
    health_checker.register_check('disk_space', check_disk_space)
except ImportError:
    logger.warning("psutil not available, system health checks disabled")


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup structured logging configuration."""
    
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"Logging configured with level {log_level}")


def export_metrics_endpoint():
    """Export metrics in various formats for monitoring systems."""
    
    return {
        'prometheus': metrics.export_prometheus(),
        'json': json.dumps(metrics.get_metrics(), indent=2),
        'health': health_checker.run_checks()
    }


class AlertManager:
    """Manage alerts and notifications for critical events."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alert_handlers = []
        self.alert_history = []
        self.alert_thresholds = {
            'error_rate_per_minute': 10,
            'response_time_p95_ms': 5000,
            'memory_usage_percent': 90,
            'disk_usage_percent': 90,
            'failed_experiments_per_hour': 5
        }
    
    def register_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register an alert handler function.
        
        Args:
            handler: Function that takes alert data and sends notification
        """
        self.alert_handlers.append(handler)
    
    def check_thresholds(self) -> None:
        """Check all metrics against alert thresholds."""
        current_metrics = metrics.get_metrics()
        alerts = []
        
        # Check error rate
        if current_metrics.get('errors', 0) > self.alert_thresholds['error_rate_per_minute']:
            alerts.append({
                'type': 'error_rate',
                'severity': 'high',
                'message': f"High error rate: {current_metrics['errors']} errors",
                'metric_value': current_metrics['errors'],
                'threshold': self.alert_thresholds['error_rate_per_minute']
            })
        
        # Check response times
        response_times = current_metrics.get('response_times', [])
        if response_times:
            p95 = sorted(response_times)[int(0.95 * len(response_times))]
            if p95 * 1000 > self.alert_thresholds['response_time_p95_ms']:
                alerts.append({
                    'type': 'slow_response',
                    'severity': 'medium',
                    'message': f"Slow response time: {p95*1000:.2f}ms (p95)",
                    'metric_value': p95 * 1000,
                    'threshold': self.alert_thresholds['response_time_p95_ms']
                })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send alert to all registered handlers.
        
        Args:
            alert_data: Alert information
        """
        alert_data['timestamp'] = time.time()
        self.alert_history.append(alert_data)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        logger.warning(f"ALERT: {alert_data['type']} - {alert_data['message']}")
        
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")


# Global alert manager
alert_manager = AlertManager()


def log_alert_handler(alert_data: Dict[str, Any]) -> None:
    """Default log-based alert handler.
    
    Args:
        alert_data: Alert information
    """
    logger.critical(f"ALERT: {alert_data['type']} - {alert_data['message']} "
                   f"(value: {alert_data.get('metric_value')}, "
                   f"threshold: {alert_data.get('threshold')})")


# Register default alert handler
alert_manager.register_handler(log_alert_handler)