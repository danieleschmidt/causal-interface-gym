"""Health monitoring and system diagnostics."""

import time
import logging
import psutil
import networkx as nx
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class HealthMetric:
    """Health metric data structure."""
    name: str
    value: float
    unit: str
    status: str  # OK, WARNING, CRITICAL
    timestamp: float
    description: str = ""


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: str  # HEALTHY, DEGRADED, UNHEALTHY
    metrics: List[HealthMetric]
    timestamp: float
    uptime: float


class PerformanceMonitor:
    """Monitor system performance and health."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize performance monitor.
        
        Args:
            retention_hours: How long to retain metrics
        """
        self.retention_hours = retention_hours
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.start_time = time.time()
        self.last_health_check = 0
        self.health_check_interval = 60  # seconds
        self._lock = threading.Lock()
        
    def record_metric(self, name: str, value: float, unit: str = "", description: str = ""):
        """Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            description: Metric description
        """
        timestamp = time.time()
        metric = HealthMetric(
            name=name,
            value=value,
            unit=unit,
            status="OK",
            timestamp=timestamp,
            description=description
        )
        
        with self._lock:
            self.metrics_history[name].append(metric)
            
        # Cleanup old metrics
        self._cleanup_old_metrics()
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        for name, metrics in self.metrics_history.items():
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()
    
    def get_metric_history(self, name: str, hours: int = 1) -> List[HealthMetric]:
        """Get metric history for specified time period.
        
        Args:
            name: Metric name
            hours: Number of hours of history
            
        Returns:
            List of metrics
        """
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            return [m for m in self.metrics_history[name] if m.timestamp >= cutoff_time]
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health status.
        
        Returns:
            System health information
        """
        now = time.time()
        uptime = now - self.start_time
        
        metrics = []
        overall_status = "HEALTHY"
        
        # CPU usage
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = "OK"
            if cpu_percent > 90:
                cpu_status = "CRITICAL"
                overall_status = "UNHEALTHY"
            elif cpu_percent > 70:
                cpu_status = "WARNING"
                if overall_status == "HEALTHY":
                    overall_status = "DEGRADED"
                    
            metrics.append(HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                status=cpu_status,
                timestamp=now,
                description="CPU utilization percentage"
            ))
        except Exception as e:
            logger.warning(f"Failed to get CPU metrics: {e}")
        
        # Memory usage
        try:
            memory = psutil.virtual_memory()
            mem_status = "OK"
            if memory.percent > 90:
                mem_status = "CRITICAL"
                overall_status = "UNHEALTHY"
            elif memory.percent > 80:
                mem_status = "WARNING"
                if overall_status == "HEALTHY":
                    overall_status = "DEGRADED"
                    
            metrics.append(HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                status=mem_status,
                timestamp=now,
                description="Memory utilization percentage"
            ))
        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
        
        # Disk usage
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_status = "OK"
            if disk_percent > 95:
                disk_status = "CRITICAL"
                overall_status = "UNHEALTHY"
            elif disk_percent > 85:
                disk_status = "WARNING"
                if overall_status == "HEALTHY":
                    overall_status = "DEGRADED"
                    
            metrics.append(HealthMetric(
                name="disk_usage",
                value=disk_percent,
                unit="%",
                status=disk_status,
                timestamp=now,
                description="Disk space utilization percentage"
            ))
        except Exception as e:
            logger.warning(f"Failed to get disk metrics: {e}")
        
        # Application-specific metrics
        app_metrics = self._get_application_metrics()
        metrics.extend(app_metrics)
        
        return SystemHealth(
            overall_status=overall_status,
            metrics=metrics,
            timestamp=now,
            uptime=uptime
        )
    
    def _get_application_metrics(self) -> List[HealthMetric]:
        """Get application-specific health metrics.
        
        Returns:
            List of application metrics
        """
        metrics = []
        now = time.time()
        
        # Average response time
        if "response_time" in self.metrics_history:
            recent_times = [m.value for m in self.get_metric_history("response_time", 0.25)]  # Last 15 minutes
            if recent_times:
                avg_response = sum(recent_times) / len(recent_times)
                status = "OK"
                if avg_response > 5.0:  # 5 seconds
                    status = "CRITICAL"
                elif avg_response > 2.0:  # 2 seconds
                    status = "WARNING"
                
                metrics.append(HealthMetric(
                    name="avg_response_time",
                    value=avg_response,
                    unit="seconds",
                    status=status,
                    timestamp=now,
                    description="Average response time over last 15 minutes"
                ))
        
        # Error rate
        if "errors" in self.metrics_history and "requests" in self.metrics_history:
            recent_errors = len(self.get_metric_history("errors", 0.25))
            recent_requests = len(self.get_metric_history("requests", 0.25))
            
            if recent_requests > 0:
                error_rate = (recent_errors / recent_requests) * 100
                status = "OK"
                if error_rate > 10:  # 10% error rate
                    status = "CRITICAL"
                elif error_rate > 5:  # 5% error rate
                    status = "WARNING"
                
                metrics.append(HealthMetric(
                    name="error_rate",
                    value=error_rate,
                    unit="%",
                    status=status,
                    timestamp=now,
                    description="Error rate over last 15 minutes"
                ))
        
        return metrics


class HealthChecker:
    """Automated health checking system."""
    
    def __init__(self):
        """Initialize health checker."""
        self.monitors = []
        self.checks = []
        self.alerts = []
        self.performance_monitor = PerformanceMonitor()
        
    def add_monitor(self, monitor_func: Callable[[], Dict[str, Any]]):
        """Add a monitoring function.
        
        Args:
            monitor_func: Function that returns health metrics
        """
        self.monitors.append(monitor_func)
    
    def add_health_check(self, check_func: Callable[[], bool], name: str):
        """Add a health check function.
        
        Args:
            check_func: Function that returns True if healthy
            name: Name of the health check
        """
        self.checks.append((check_func, name))
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks.
        
        Returns:
            Health check results
        """
        results = {
            "timestamp": time.time(),
            "overall_healthy": True,
            "checks": {},
            "metrics": {}
        }
        
        # Run individual health checks
        for check_func, name in self.checks:
            try:
                healthy = check_func()
                results["checks"][name] = {"healthy": healthy, "error": None}
                if not healthy:
                    results["overall_healthy"] = False
            except Exception as e:
                results["checks"][name] = {"healthy": False, "error": str(e)}
                results["overall_healthy"] = False
                logger.error(f"Health check '{name}' failed: {e}")
        
        # Run monitoring functions
        for monitor_func in self.monitors:
            try:
                metrics = monitor_func()
                results["metrics"].update(metrics)
            except Exception as e:
                logger.error(f"Monitor function failed: {e}")
        
        # Get system health
        system_health = self.performance_monitor.get_system_health()
        results["system_health"] = {
            "status": system_health.overall_status,
            "uptime": system_health.uptime,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "status": m.status,
                    "description": m.description
                }
                for m in system_health.metrics
            ]
        }
        
        if system_health.overall_status != "HEALTHY":
            results["overall_healthy"] = False
        
        return results
    
    def get_health_report(self) -> str:
        """Generate human-readable health report.
        
        Returns:
            Health report string
        """
        health_data = self.run_health_checks()
        
        report = []
        report.append(f"# System Health Report")
        report.append(f"Generated at: {datetime.fromtimestamp(health_data['timestamp'])}")
        report.append(f"Overall Status: {'✅ HEALTHY' if health_data['overall_healthy'] else '❌ UNHEALTHY'}")
        report.append(f"System Uptime: {health_data['system_health']['uptime']:.1f} seconds")
        report.append("")
        
        # System metrics
        report.append("## System Metrics")
        for metric in health_data['system_health']['metrics']:
            status_icon = {"OK": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(metric['status'], "❓")
            report.append(f"- {metric['name']}: {metric['value']:.1f}{metric['unit']} {status_icon}")
        report.append("")
        
        # Health checks
        if health_data['checks']:
            report.append("## Health Checks")
            for name, result in health_data['checks'].items():
                status_icon = "✅" if result['healthy'] else "❌"
                report.append(f"- {name}: {status_icon}")
                if result['error']:
                    report.append(f"  Error: {result['error']}")
            report.append("")
        
        # Custom metrics
        if health_data['metrics']:
            report.append("## Application Metrics")
            for name, value in health_data['metrics'].items():
                report.append(f"- {name}: {value}")
        
        return "\n".join(report)


# Causal-specific health checks
def check_graph_validity(environment) -> bool:
    """Check if causal graph is valid."""
    try:
        if not isinstance(environment.graph, nx.DiGraph):
            return False
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(environment.graph):
            logger.warning("Causal graph contains cycles")
            return False
        
        # Check for isolated nodes
        isolated = list(nx.isolates(environment.graph))
        if len(isolated) > len(environment.graph.nodes()) * 0.5:
            logger.warning(f"Too many isolated nodes: {len(isolated)}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Graph validity check failed: {e}")
        return False


def check_intervention_capability(environment) -> bool:
    """Check if interventions can be performed."""
    try:
        # Try a dummy intervention if graph has nodes
        if environment.graph.nodes():
            test_var = list(environment.graph.nodes())[0]
            result = environment.intervene(**{test_var: True})
            return "interventions_applied" in result
        return True
    except Exception as e:
        logger.error(f"Intervention capability check failed: {e}")
        return False


def monitor_graph_metrics(environment) -> Dict[str, Any]:
    """Monitor graph-related metrics."""
    try:
        return {
            "graph_nodes": environment.graph.number_of_nodes(),
            "graph_edges": environment.graph.number_of_edges(),
            "graph_density": nx.density(environment.graph),
            "strongly_connected_components": nx.number_strongly_connected_components(environment.graph)
        }
    except Exception as e:
        logger.error(f"Graph metrics monitoring failed: {e}")
        return {}


# Global health checker instance
health_checker = HealthChecker()

# Default health checks for causal environments
def setup_default_health_checks(environment):
    """Set up default health checks for a causal environment.
    
    Args:
        environment: CausalEnvironment instance
    """
    health_checker.add_health_check(
        lambda: check_graph_validity(environment),
        "graph_validity"
    )
    
    health_checker.add_health_check(
        lambda: check_intervention_capability(environment),
        "intervention_capability"  
    )
    
    health_checker.add_monitor(
        lambda: monitor_graph_metrics(environment)
    )


def get_health_endpoint():
    """Get health status for HTTP endpoint.
    
    Returns:
        Dictionary suitable for JSON response
    """
    health_data = health_checker.run_health_checks()
    
    return {
        "status": "healthy" if health_data["overall_healthy"] else "unhealthy",
        "timestamp": health_data["timestamp"],
        "checks": health_data["checks"],
        "metrics": health_data.get("metrics", {}),
        "system": health_data.get("system_health", {})
    }