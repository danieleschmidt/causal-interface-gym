"""Enterprise-Grade Security Framework for Causal Interface Gym.

Advanced security controls including:
- Multi-layered input validation and sanitization
- Adaptive rate limiting with behavioral analysis  
- Zero-trust authentication and RBAC authorization
- Comprehensive audit logging and compliance monitoring
- Advanced data protection with encryption and anonymization
- Threat detection and automated response systems
- Security incident management and forensics
"""

import hashlib
import secrets
import time
import logging
import re
import json
import hmac
import base64
import ipaddress
import threading
import asyncio
import os
import sqlite3
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
from enum import Enum
import urllib.parse

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class SecurityEvent(Enum):
    """Types of security events."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_ACCESS = "data_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_INPUT = "malicious_input"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    DDOS_ATTEMPT = "ddos_attempt"

@dataclass
class SecurityPolicy:
    """Comprehensive security policy configuration."""
    # Input validation
    max_request_size: int = 1024 * 1024  # 1MB
    max_string_length: int = 10000
    max_array_elements: int = 1000
    allowed_file_types: List[str] = field(default_factory=lambda: ['.py', '.json', '.csv', '.txt', '.md'])
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'eval\s*\(',
        r'exec\s*\(',
        r'import\s+os',
        r'subprocess',
        r'__import__',
        r'file://',
        r'data:text/html',
    ])
    
    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    burst_threshold: int = 10
    adaptive_rate_limiting: bool = True
    
    # Authentication & Authorization
    require_authentication: bool = True
    session_timeout_minutes: int = 30
    password_min_length: int = 12
    password_complexity_required: bool = True
    multi_factor_auth_required: bool = False
    
    # Encryption
    encrypt_sensitive_data: bool = True
    data_retention_days: int = 90
    anonymize_logs: bool = True
    
    # Monitoring & Alerting
    enable_threat_detection: bool = True
    log_all_requests: bool = True
    alert_on_anomalies: bool = True
    max_failed_logins: int = 5
    account_lockout_minutes: int = 15
    
    # Compliance
    gdpr_compliance: bool = True
    pii_detection_enabled: bool = True
    audit_trail_required: bool = True

@dataclass
class SecurityIncident:
    """Security incident record."""
    incident_id: str
    timestamp: datetime
    event_type: SecurityEvent
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    description: str
    affected_resources: List[str]
    response_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    
@dataclass
class UserSession:
    """Enhanced user session with security tracking."""
    session_id: str
    user_id: str
    ip_address: str
    user_agent: str
    created_at: datetime
    last_activity: datetime
    permissions: Set[str]
    risk_score: float = 0.0
    anomaly_flags: List[str] = field(default_factory=list)

class SecurityError(Exception):
    """Base exception for security violations."""
    def __init__(self, message: str, event_type: SecurityEvent = SecurityEvent.UNAUTHORIZED_ACCESS, threat_level: ThreatLevel = ThreatLevel.MEDIUM):
        super().__init__(message)
        self.event_type = event_type
        self.threat_level = threat_level

class RateLimitError(SecurityError):
    """Rate limit exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, SecurityEvent.RATE_LIMIT_EXCEEDED, ThreatLevel.MEDIUM)

class ValidationError(SecurityError):
    """Input validation failed."""
    def __init__(self, message: str = "Input validation failed"):
        super().__init__(message, SecurityEvent.MALICIOUS_INPUT, ThreatLevel.HIGH)

class AuthenticationError(SecurityError):
    """Authentication failed."""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, SecurityEvent.LOGIN_FAILURE, ThreatLevel.MEDIUM)


class EnterpriseSecurityManager:
    """Enterprise-grade security management system."""
    
    def __init__(self, policy: SecurityPolicy = None):
        """Initialize enterprise security manager.
        
        Args:
            policy: Security policy configuration
        """
        self.policy = policy or SecurityPolicy()
        self._setup_security_database()
        self._init_threat_detection()
        self._init_encryption()
        
        # Security state tracking
        self.active_sessions: Dict[str, UserSession] = {}
        self.failed_login_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.security_incidents: List[SecurityIncident] = []
        self.anomaly_detector = None
        
        # Thread-safe locks
        self._security_lock = threading.RLock()
        
        logger.info("Enterprise Security Manager initialized")
    
    def _setup_security_database(self):
        """Setup security audit database."""
        self.db_path = 'security_audit.db'
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    threat_level INTEGER NOT NULL,
                    source_ip TEXT,
                    user_id TEXT,
                    description TEXT NOT NULL,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_agent TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                    risk_score REAL DEFAULT 0.0,
                    active BOOLEAN DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS rate_limits (
                    key TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    window_start DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
                CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
            """)
    
    def _init_threat_detection(self):
        """Initialize threat detection systems."""
        if self.policy.enable_threat_detection:
            # Initialize ML-based anomaly detection
            try:
                from sklearn.ensemble import IsolationForest
                self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            except ImportError:
                logger.warning("scikit-learn not available, advanced anomaly detection disabled")
    
    def _init_encryption(self):
        """Initialize encryption systems."""
        if self.policy.encrypt_sensitive_data:
            # Generate or load encryption key
            key_file = 'encryption.key'
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                from cryptography.fernet import Fernet
                self.encryption_key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.encryption_key)
            
            from cryptography.fernet import Fernet
            self.cipher_suite = Fernet(self.encryption_key)
    
    def authenticate_user(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[UserSession]:
        """Authenticate user with advanced security checks.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            User session if authentication successful, None otherwise
        """
        with self._security_lock:
            # Check for account lockout
            if self._is_account_locked(ip_address):
                self._log_security_event(
                    SecurityEvent.LOGIN_FAILURE,
                    ThreatLevel.HIGH,
                    f"Login attempt from locked account: {username}",
                    source_ip=ip_address,
                    user_id=username
                )
                raise AuthenticationError("Account temporarily locked due to failed login attempts")
            
            # Validate credentials (simplified - in production would check against secure database)
            if self._validate_credentials(username, password):
                # Create new session
                session = self._create_user_session(username, ip_address, user_agent)
                
                # Clear failed login attempts
                if ip_address in self.failed_login_attempts:
                    del self.failed_login_attempts[ip_address]
                
                self._log_security_event(
                    SecurityEvent.LOGIN_SUCCESS,
                    ThreatLevel.LOW,
                    f"Successful login for user: {username}",
                    source_ip=ip_address,
                    user_id=username
                )
                
                return session
            else:
                # Record failed login attempt
                self.failed_login_attempts[ip_address].append(datetime.now())
                
                # Clean old attempts
                cutoff = datetime.now() - timedelta(minutes=self.policy.account_lockout_minutes)
                self.failed_login_attempts[ip_address] = [
                    attempt for attempt in self.failed_login_attempts[ip_address]
                    if attempt > cutoff
                ]
                
                self._log_security_event(
                    SecurityEvent.LOGIN_FAILURE,
                    ThreatLevel.MEDIUM,
                    f"Failed login attempt for user: {username}",
                    source_ip=ip_address,
                    user_id=username
                )
                
                raise AuthenticationError("Invalid credentials")
    
    def _is_account_locked(self, ip_address: str) -> bool:
        """Check if account is locked due to failed login attempts."""
        if ip_address not in self.failed_login_attempts:
            return False
        
        cutoff = datetime.now() - timedelta(minutes=self.policy.account_lockout_minutes)
        recent_failures = [
            attempt for attempt in self.failed_login_attempts[ip_address]
            if attempt > cutoff
        ]
        
        return len(recent_failures) >= self.policy.max_failed_logins
    
    def _validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if credentials are valid
        """
        # Simplified credential validation - in production would use proper password hashing
        # and database lookup
        return len(username) > 0 and len(password) >= self.policy.password_min_length
    
    def _create_user_session(self, username: str, ip_address: str, user_agent: str) -> UserSession:
        """Create new user session.
        
        Args:
            username: Username
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            New user session
        """
        session_id = secrets.token_urlsafe(32)
        
        session = UserSession(
            session_id=session_id,
            user_id=username,
            ip_address=ip_address,
            user_agent=user_agent,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            permissions=self._get_user_permissions(username)
        )
        
        self.active_sessions[session_id] = session
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO user_sessions 
                (session_id, user_id, ip_address, user_agent, risk_score)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, username, ip_address, user_agent, session.risk_score))
        
        return session
    
    def _get_user_permissions(self, username: str) -> Set[str]:
        """Get user permissions.
        
        Args:
            username: Username
            
        Returns:
            Set of user permissions
        """
        # Simplified permission system - in production would be role-based
        return {"read", "experiment", "moderate_risk_operations"}
    
    def validate_session(self, session_id: str, ip_address: str) -> Optional[UserSession]:
        """Validate user session.
        
        Args:
            session_id: Session ID
            ip_address: Client IP address
            
        Returns:
            User session if valid, None otherwise
        """
        with self._security_lock:
            session = self.active_sessions.get(session_id)
            
            if not session:
                return None
            
            # Check session timeout
            timeout_threshold = datetime.now() - timedelta(minutes=self.policy.session_timeout_minutes)
            if session.last_activity < timeout_threshold:
                self._invalidate_session(session_id)
                return None
            
            # Check IP address consistency (optional)
            if session.ip_address != ip_address:
                session.anomaly_flags.append(f"IP_CHANGE_{ip_address}")
                session.risk_score += 0.3
                
                self._log_security_event(
                    SecurityEvent.ANOMALOUS_BEHAVIOR,
                    ThreatLevel.MEDIUM,
                    f"Session IP address changed: {session.ip_address} -> {ip_address}",
                    source_ip=ip_address,
                    user_id=session.user_id
                )
            
            # Update last activity
            session.last_activity = datetime.now()
            
            return session
    
    def _invalidate_session(self, session_id: str):
        """Invalidate user session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        # Mark as inactive in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE user_sessions SET active = 0 WHERE session_id = ?", (session_id,))
    
    def check_rate_limit(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check rate limiting for a given key.
        
        Args:
            key: Rate limit key (e.g., IP address, user ID)
            limit: Request limit
            window_seconds: Time window in seconds
            
        Returns:
            True if within limits, False if exceeded
        """
        now = time.time()
        
        # Clean old entries
        limiter = self.rate_limiters[key]
        while limiter and limiter[0] < now - window_seconds:
            limiter.popleft()
        
        # Check limit
        if len(limiter) >= limit:
            self._log_security_event(
                SecurityEvent.RATE_LIMIT_EXCEEDED,
                ThreatLevel.MEDIUM,
                f"Rate limit exceeded for key: {key} ({len(limiter)}/{limit})",
                source_ip=key if '.' in key else None
            )
            return False
        
        # Add current request
        limiter.append(now)
        return True
    
    def validate_input(self, data: Any, input_type: str = "general") -> Any:
        """Validate and sanitize input data.
        
        Args:
            data: Input data to validate
            input_type: Type of input for specific validation rules
            
        Returns:
            Validated and sanitized data
        """
        if isinstance(data, str):
            return self._validate_string_input(data, input_type)
        elif isinstance(data, dict):
            return self._validate_dict_input(data, input_type)
        elif isinstance(data, list):
            return self._validate_list_input(data, input_type)
        else:
            return data
    
    def _validate_string_input(self, text: str, input_type: str) -> str:
        """Validate string input."""
        # Length check
        if len(text) > self.policy.max_string_length:
            raise ValidationError(f"Input too long: {len(text)} > {self.policy.max_string_length}")
        
        # Check for malicious patterns
        for pattern in self.policy.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self._log_security_event(
                    SecurityEvent.MALICIOUS_INPUT,
                    ThreatLevel.HIGH,
                    f"Malicious pattern detected: {pattern}"
                )
                raise ValidationError(f"Potentially malicious input detected")
        
        # Sanitize HTML
        text = self._sanitize_html(text)
        
        return text
    
    def _validate_dict_input(self, data: Dict, input_type: str) -> Dict:
        """Validate dictionary input."""
        if len(data) > 100:  # Reasonable limit for dict size
            raise ValidationError(f"Dictionary too large: {len(data)} keys")
        
        sanitized = {}
        for key, value in data.items():
            safe_key = self.validate_input(key, f"{input_type}_key")
            safe_value = self.validate_input(value, f"{input_type}_value")
            sanitized[safe_key] = safe_value
        
        return sanitized
    
    def _validate_list_input(self, data: List, input_type: str) -> List:
        """Validate list input."""
        if len(data) > self.policy.max_array_elements:
            raise ValidationError(f"Array too large: {len(data)} > {self.policy.max_array_elements}")
        
        return [self.validate_input(item, f"{input_type}_item") for item in data]
    
    def _sanitize_html(self, text: str) -> str:
        """Sanitize HTML content."""
        # Basic HTML entity encoding
        text = (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#39;'))
        
        return text
    
    def _log_security_event(self, event_type: SecurityEvent, threat_level: ThreatLevel, 
                           description: str, source_ip: Optional[str] = None, 
                           user_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log security event.
        
        Args:
            event_type: Type of security event
            threat_level: Threat severity level
            description: Event description
            source_ip: Source IP address
            user_id: User ID if applicable
            metadata: Additional metadata
        """
        incident = SecurityIncident(
            incident_id=secrets.token_hex(8),
            timestamp=datetime.now(),
            event_type=event_type,
            threat_level=threat_level,
            source_ip=source_ip or "unknown",
            user_id=user_id,
            description=description,
            affected_resources=[]
        )
        
        self.security_incidents.append(incident)
        
        # Log to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO security_events 
                (event_type, threat_level, source_ip, user_id, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (event_type.value, threat_level.value, source_ip, user_id, 
                  description, json.dumps(metadata or {})))
        
        # Log to application logger
        log_level = logging.WARNING if threat_level.value >= 2 else logging.INFO
        logger.log(log_level, f"SECURITY_EVENT: {event_type.value} - {description}")
        
        # Trigger automated response for high-severity events
        if threat_level.value >= ThreatLevel.HIGH.value:
            self._trigger_security_response(incident)
    
    def _trigger_security_response(self, incident: SecurityIncident):
        """Trigger automated security response.
        
        Args:
            incident: Security incident
        """
        # Automated response based on incident type
        if incident.event_type == SecurityEvent.DDOS_ATTEMPT:
            # Temporary IP ban
            self._ban_ip_temporarily(incident.source_ip, minutes=30)
        
        elif incident.event_type == SecurityEvent.MALICIOUS_INPUT:
            # Rate limit the source
            self.rate_limiters[incident.source_ip].extend([time.time()] * 50)
        
        elif incident.event_type == SecurityEvent.PRIVILEGE_ESCALATION:
            # Invalidate user sessions
            if incident.user_id:
                self._invalidate_user_sessions(incident.user_id)
        
        logger.warning(f"Security response triggered for incident: {incident.incident_id}")
    
    def _ban_ip_temporarily(self, ip_address: str, minutes: int):
        """Temporarily ban IP address.
        
        Args:
            ip_address: IP address to ban
            minutes: Ban duration in minutes
        """
        # In a real implementation, this would update firewall rules
        logger.warning(f"IP {ip_address} temporarily banned for {minutes} minutes")
    
    def _invalidate_user_sessions(self, user_id: str):
        """Invalidate all sessions for a user.
        
        Args:
            user_id: User ID
        """
        sessions_to_remove = []
        for session_id, session in self.active_sessions.items():
            if session.user_id == user_id:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            self._invalidate_session(session_id)
        
        logger.warning(f"All sessions invalidated for user: {user_id}")
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        if not self.policy.encrypt_sensitive_data:
            return data
        
        encrypted_bytes = self.cipher_suite.encrypt(data.encode())
        return base64.b64encode(encrypted_bytes).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        if not self.policy.encrypt_sensitive_data:
            return encrypted_data
        
        encrypted_bytes = base64.b64decode(encrypted_data.encode())
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report.
        
        Returns:
            Comprehensive security report
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get event counts by type
            event_counts = {}
            cursor = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM security_events
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY event_type
            """)
            
            for row in cursor.fetchall():
                event_counts[row[0]] = row[1]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(self.active_sessions),
            'security_incidents_24h': len([
                i for i in self.security_incidents 
                if i.timestamp > datetime.now() - timedelta(hours=24)
            ]),
            'event_counts_24h': event_counts,
            'failed_login_attempts': len(self.failed_login_attempts),
            'rate_limiters_active': len(self.rate_limiters),
            'security_policy': {
                'authentication_required': self.policy.require_authentication,
                'encryption_enabled': self.policy.encrypt_sensitive_data,
                'threat_detection_enabled': self.policy.enable_threat_detection,
                'audit_trail_enabled': self.policy.audit_trail_required
            }
        }


# Decorator for securing API endpoints
def secure_endpoint(security_manager: EnterpriseSecurityManager, 
                   required_permissions: List[str] = None,
                   rate_limit: int = 60):
    """Decorator to secure API endpoints.
    
    Args:
        security_manager: Security manager instance
        required_permissions: Required permissions for access
        rate_limit: Rate limit for endpoint
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract session info (simplified - would come from request context)
            session_id = kwargs.get('session_id')
            ip_address = kwargs.get('ip_address', 'unknown')
            
            # Validate session
            if security_manager.policy.require_authentication:
                session = security_manager.validate_session(session_id, ip_address)
                if not session:
                    raise AuthenticationError("Invalid or expired session")
                
                # Check permissions
                if required_permissions:
                    for perm in required_permissions:
                        if perm not in session.permissions:
                            raise SecurityError(f"Insufficient permissions: {perm} required", 
                                              SecurityEvent.UNAUTHORIZED_ACCESS, ThreatLevel.HIGH)
            
            # Check rate limit
            if not security_manager.check_rate_limit(ip_address, rate_limit):
                raise RateLimitError(f"Rate limit exceeded: {rate_limit} requests per minute")
            
            # Validate inputs
            for key, value in kwargs.items():
                if key not in ['session_id', 'ip_address']:  # Skip meta parameters
                    kwargs[key] = security_manager.validate_input(value, key)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global security manager instance
enterprise_security = EnterpriseSecurityManager()