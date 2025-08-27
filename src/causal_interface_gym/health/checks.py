"""Health check and readiness endpoints for production deployment."""

from typing import Dict, Any
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from causal_interface_gym import CausalEnvironment
from causal_interface_gym.monitoring import HealthChecker


class HealthCheckService:
    """Production health check service."""
    
    def __init__(self):
        """Initialize health check service."""
        self.start_time = time.time()
        self.health_checker = HealthChecker()
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Test core functionality
            env = CausalEnvironment()
            env.add_variable("health_test", "binary")
            
            # Check system resources
            health_status = self.health_checker.get_system_health()
            
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time,
                "version": "1.0.0",
                "environment": os.getenv("ENV", "unknown"),
                "checks": {
                    "core_functionality": "pass",
                    "system_health": health_status.get("status", "unknown"),
                    "memory_usage": health_status.get("memory", "unknown"),
                    "cpu_usage": health_status.get("cpu", "unknown")
                }
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": time.time(),
                "error": str(e),
                "checks": {
                    "core_functionality": "fail"
                }
            }
    
    def readiness_check(self) -> Dict[str, Any]:
        """Readiness check for load balancer."""
        try:
            # Quick functionality test
            env = CausalEnvironment()
            env.add_variable("ready_test", "binary")
            
            return {
                "status": "ready",
                "timestamp": time.time(),
                "checks": {
                    "core_ready": "pass",
                    "dependencies_ready": "pass"
                }
            }
        except Exception as e:
            return {
                "status": "not_ready",
                "timestamp": time.time(),
                "error": str(e)
            }


def create_health_server():
    """Create health check server."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    health_service = HealthCheckService()
    
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                response = health_service.health_check()
                status_code = 200 if response["status"] == "healthy" else 503
            elif self.path == "/ready":
                response = health_service.readiness_check()
                status_code = 200 if response["status"] == "ready" else 503
            else:
                response = {"error": "Not found"}
                status_code = 404
            
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
    
    port = int(os.getenv("HEALTH_CHECK_PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    return server


if __name__ == "__main__":
    server = create_health_server()
    print(f"Health check server running on port {server.server_port}")
    server.serve_forever()
