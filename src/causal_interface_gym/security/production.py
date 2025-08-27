"""Security middleware for production deployment."""

import os
import hashlib
import time
from typing import Dict, Any, List


class SecurityMiddleware:
    """Production security middleware."""
    
    def __init__(self):
        """Initialize security middleware."""
        self.rate_limit_store = {}
        self.blocked_ips = set()
    
    def add_security_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add security headers to response."""
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
        
        headers.update(security_headers)
        return headers
    
    def rate_limit_check(self, client_ip: str, endpoint: str, limit: int = 100, window: int = 3600) -> bool:
        """Check rate limiting for client."""
        now = time.time()
        key = f"{client_ip}:{endpoint}"
        
        # Clean old entries
        if key in self.rate_limit_store:
            self.rate_limit_store[key] = [
                timestamp for timestamp in self.rate_limit_store[key]
                if now - timestamp < window
            ]
        else:
            self.rate_limit_store[key] = []
        
        # Check limit
        if len(self.rate_limit_store[key]) >= limit:
            return False
        
        # Add current request
        self.rate_limit_store[key].append(now)
        return True
    
    def validate_input(self, input_data: str) -> bool:
        """Validate input for security threats."""
        if not isinstance(input_data, str):
            return False
        
        # Check for common injection patterns
        dangerous_patterns = [
            "<script", "javascript:", "onload=", "onerror=",
            "eval(", "exec(", "__import__", "system(",
            "DROP TABLE", "DELETE FROM", "INSERT INTO",
            "../", "..\\", "/etc/passwd", "cmd.exe"
        ]
        
        input_lower = input_data.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in input_lower:
                return False
        
        return True
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring."""
        import logging
        
        security_logger = logging.getLogger("security")
        security_logger.warning(f"Security event: {event_type} - {details}")


# Environment-specific security settings
class ProductionSecurity:
    """Production security configuration."""
    
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
    CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    
    # Rate limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS = 1000
    RATE_LIMIT_WINDOW = 3600
    
    # Input validation
    MAX_INPUT_LENGTH = 1000
    VALIDATE_ALL_INPUTS = True
    
    # Security headers
    SECURITY_HEADERS_ENABLED = True
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with salt."""
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return salt + pwdhash
    
    @staticmethod
    def verify_password(stored_password: bytes, provided_password: str) -> bool:
        """Verify password against stored hash."""
        salt = stored_password[:32]
        stored_key = stored_password[32:]
        new_key = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, 100000)
        return stored_key == new_key
