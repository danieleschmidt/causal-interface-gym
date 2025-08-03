"""Health check system for monitoring application status."""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthCheck:
    """Individual health check implementation."""
    
    def __init__(self, name: str, check_function: Callable, 
                 timeout: float = 5.0, critical: bool = True):
        """Initialize health check.
        
        Args:
            name: Name of the health check
            check_function: Function to execute for health check
            timeout: Timeout in seconds
            critical: Whether this check is critical for overall health
        """
        self.name = name
        self.check_function = check_function
        self.timeout = timeout
        self.critical = critical
    
    async def execute(self) -> HealthCheckResult:
        """Execute the health check.
        
        Returns:
            Health check result
        """
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_check(),
                timeout=self.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if result.get("healthy", True):
                status = HealthStatus.HEALTHY
                message = result.get("message", "OK")
            else:
                status = HealthStatus.UNHEALTHY
                message = result.get("message", "Health check failed")
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                metadata=result.get("metadata", {})
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration_ms=duration_ms,
                metadata={"error": str(e)}
            )
    
    async def _run_check(self) -> Dict[str, Any]:
        """Run the actual check function.
        
        Returns:
            Check result dictionary
        """
        if asyncio.iscoroutinefunction(self.check_function):
            return await self.check_function()
        else:
            return self.check_function()


class HealthChecker:
    """Centralized health check manager."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.start_time = datetime.now()
    
    def register_check(self, health_check: HealthCheck) -> None:
        """Register a health check.
        
        Args:
            health_check: Health check to register
        """
        self.checks[health_check.name] = health_check
        logger.info(f"Registered health check: {health_check.name}")
    
    def register_database_check(self, db_manager) -> None:
        """Register database health check.
        
        Args:
            db_manager: Database manager instance
        """
        def db_check():
            try:
                # Simple query to test database connectivity
                result = db_manager.execute_query("SELECT 1")
                return {
                    "healthy": True,
                    "message": "Database connection OK",
                    "metadata": {"db_type": db_manager.db_type}
                }
            except Exception as e:
                return {
                    "healthy": False,
                    "message": f"Database connection failed: {str(e)}",
                    "metadata": {"error": str(e)}
                }
        
        check = HealthCheck("database", db_check, timeout=3.0, critical=True)
        self.register_check(check)
    
    def register_cache_check(self, cache_manager) -> None:
        """Register cache health check.
        
        Args:
            cache_manager: Cache manager instance
        """
        def cache_check():
            try:
                # Test cache connectivity
                test_key = "health_check_test"
                cache_manager.set(test_key, "test_value", ttl=10)
                value = cache_manager.get(test_key)
                cache_manager.delete(test_key)
                
                if value == "test_value":
                    return {
                        "healthy": True,
                        "message": "Cache connection OK",
                        "metadata": cache_manager.get_cache_stats()
                    }
                else:
                    return {
                        "healthy": False,
                        "message": "Cache test failed"
                    }
            except Exception as e:
                return {
                    "healthy": False,
                    "message": f"Cache connection failed: {str(e)}",
                    "metadata": {"error": str(e)}
                }
        
        check = HealthCheck("cache", cache_check, timeout=2.0, critical=False)
        self.register_check(check)
    
    def register_llm_provider_check(self, llm_client) -> None:
        """Register LLM provider health check.
        
        Args:
            llm_client: LLM client instance
        """
        def llm_check():
            try:
                if not llm_client.provider.is_available():
                    return {
                        "healthy": False,
                        "message": "LLM provider not configured"
                    }
                
                # Simple test query
                response = llm_client.provider.generate(
                    "Say 'OK' if you can respond.",
                    max_tokens=10,
                    temperature=0
                )
                
                return {
                    "healthy": True,
                    "message": "LLM provider responsive",
                    "metadata": {
                        "provider": llm_client.provider.get_provider_name(),
                        "model": llm_client.provider.model
                    }
                }
            except Exception as e:
                return {
                    "healthy": False,
                    "message": f"LLM provider check failed: {str(e)}",
                    "metadata": {"error": str(e)}
                }
        
        check = HealthCheck("llm_provider", llm_check, timeout=10.0, critical=False)
        self.register_check(check)
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks.
        
        Returns:
            Overall health status and individual check results
        """
        if not self.checks:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks registered",
                "checks": {},
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
        
        # Run all checks concurrently
        results = await asyncio.gather(
            *[check.execute() for check in self.checks.values()],
            return_exceptions=True
        )
        
        # Process results
        check_results = {}
        critical_failures = 0
        total_failures = 0
        
        for i, result in enumerate(results):
            check_name = list(self.checks.keys())[i]
            check = self.checks[check_name]
            
            if isinstance(result, Exception):
                # Exception occurred during check execution
                result = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(result)}",
                    duration_ms=0
                )
            
            check_results[check_name] = {
                "status": result.status.value,
                "message": result.message,
                "duration_ms": result.duration_ms,
                "timestamp": result.timestamp.isoformat(),
                "critical": check.critical,
                "metadata": result.metadata
            }
            
            # Track failures
            if result.status != HealthStatus.HEALTHY:
                total_failures += 1
                if check.critical:
                    critical_failures += 1
            
            # Store last result
            self.last_results[check_name] = result
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{critical_failures} critical health check(s) failing"
        elif total_failures > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"{total_failures} non-critical health check(s) failing"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All health checks passing"
        
        return {
            "status": overall_status.value,
            "message": message,
            "checks": check_results,
            "summary": {
                "total_checks": len(self.checks),
                "passing": len(self.checks) - total_failures,
                "failing": total_failures,
                "critical_failing": critical_failures
            },
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def get_last_results(self) -> Dict[str, Any]:
        """Get the last health check results without running new checks.
        
        Returns:
            Last health check results
        """
        if not self.last_results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks have been run yet",
                "checks": {},
                "timestamp": datetime.now().isoformat()
            }
        
        # Build results from last check
        check_results = {}
        for name, result in self.last_results.items():
            check_results[name] = {
                "status": result.status.value,
                "message": result.message,
                "duration_ms": result.duration_ms,
                "timestamp": result.timestamp.isoformat(),
                "metadata": result.metadata
            }
        
        return {
            "status": "cached",
            "message": "Returning cached health check results",
            "checks": check_results,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }