"""Security and validation utilities for causal interface gym."""

import re
import logging
from typing import Any, Dict, List, Set, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)

# Security configuration
MAX_GRAPH_NODES = 1000
MAX_GRAPH_EDGES = 5000
MAX_VARIABLE_NAME_LENGTH = 100
MAX_EXPERIMENT_DURATION = 300  # seconds
MAX_BELIEF_QUERIES = 10000

# Dangerous patterns to block
DANGEROUS_PATTERNS = [
    r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>',  # Script injection
    r'javascript:',  # JavaScript protocol
    r'on\w+\s*=',   # Event handlers
    r'eval\s*\(',   # Eval calls
    r'exec\s*\(',   # Exec calls
    r'__import__',  # Import statements
    r'globals\s*\(',  # Globals access
    r'locals\s*\(',   # Locals access
]

SAFE_VARIABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


class SecurityError(Exception):
    """Security-related error."""
    pass


class ValidationError(Exception):
    """Input validation error."""
    pass


def validate_variable_name(name: str) -> str:
    """Validate and sanitize variable name.
    
    Args:
        name: Variable name to validate
        
    Returns:
        Sanitized variable name
        
    Raises:
        ValidationError: If name is invalid
    """
    if not isinstance(name, str):
        raise ValidationError(f"Variable name must be string, got {type(name)}")
    
    name = name.strip()
    
    if not name:
        raise ValidationError("Variable name cannot be empty")
    
    if len(name) > MAX_VARIABLE_NAME_LENGTH:
        raise ValidationError(f"Variable name too long: {len(name)} > {MAX_VARIABLE_NAME_LENGTH}")
    
    if not SAFE_VARIABLE_NAME_PATTERN.match(name):
        raise ValidationError(f"Invalid variable name: {name}. Must be alphanumeric with underscores")
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            raise SecurityError(f"Potentially dangerous pattern detected in variable name: {name}")
    
    return name


def validate_graph_size(num_nodes: int, num_edges: int) -> None:
    """Validate graph size limits.
    
    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        
    Raises:
        SecurityError: If graph exceeds size limits
    """
    if num_nodes > MAX_GRAPH_NODES:
        raise SecurityError(f"Graph too large: {num_nodes} nodes > {MAX_GRAPH_NODES} limit")
    
    if num_edges > MAX_GRAPH_EDGES:
        raise SecurityError(f"Graph too large: {num_edges} edges > {MAX_GRAPH_EDGES} limit")


def sanitize_html_input(text: str) -> str:
    """Sanitize HTML input to prevent XSS.
    
    Args:
        text: Input text
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return str(text)
    
    # HTML escape
    text = (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#39;'))
    
    # Remove dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text


def validate_probability(value: Any, name: str = "probability") -> float:
    """Validate probability value.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Returns:
        Valid probability value
        
    Raises:
        ValidationError: If value is not a valid probability
    """
    try:
        prob = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if not 0.0 <= prob <= 1.0:
        raise ValidationError(f"{name} must be between 0 and 1, got {prob}")
    
    return prob


def validate_intervention_value(value: Any, var_type: str, name: str) -> Any:
    """Validate intervention value based on variable type.
    
    Args:
        value: Intervention value
        var_type: Variable type (binary, continuous, categorical)
        name: Variable name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If value is invalid for variable type
    """
    if var_type == "binary":
        if value not in [True, False, 0, 1]:
            raise ValidationError(f"Binary variable '{name}' requires boolean/binary value, got {value}")
        return bool(value)
    
    elif var_type == "continuous":
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"Continuous variable '{name}' requires numeric value, got {value}")
    
    elif var_type == "categorical":
        return str(value)
    
    else:
        raise ValidationError(f"Unknown variable type: {var_type}")


def rate_limit(max_calls: int = 100, window: int = 60):
    """Rate limiting decorator.
    
    Args:
        max_calls: Maximum calls per window
        window: Time window in seconds
    """
    calls = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            now = time.time()
            
            # Clean old entries
            cutoff = now - window
            calls.clear()  # Simple implementation - clear all old entries
            
            # Count current calls
            if func.__name__ not in calls:
                calls[func.__name__] = []
            
            calls[func.__name__].append(now)
            
            # Check limit
            recent_calls = [t for t in calls[func.__name__] if t > cutoff]
            if len(recent_calls) > max_calls:
                raise SecurityError(f"Rate limit exceeded: {len(recent_calls)} > {max_calls} calls in {window}s")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class SecureEnvironment:
    """Security wrapper for CausalEnvironment."""
    
    def __init__(self, environment):
        """Initialize secure wrapper.
        
        Args:
            environment: CausalEnvironment instance to wrap
        """
        self.environment = environment
        self._experiment_count = 0
        self._belief_query_count = 0
    
    @rate_limit(max_calls=10, window=60)
    def add_variable(self, name: str, **kwargs):
        """Securely add variable."""
        name = validate_variable_name(name)
        validate_graph_size(
            self.environment.graph.number_of_nodes() + 1,
            self.environment.graph.number_of_edges()
        )
        return self.environment.add_variable(name, **kwargs)
    
    @rate_limit(max_calls=20, window=60)
    def add_edge(self, parent: str, child: str, **kwargs):
        """Securely add edge."""
        parent = validate_variable_name(parent)
        child = validate_variable_name(child)
        validate_graph_size(
            self.environment.graph.number_of_nodes(),
            self.environment.graph.number_of_edges() + 1
        )
        return self.environment.add_edge(parent, child, **kwargs)
    
    @rate_limit(max_calls=5, window=60)
    def intervene(self, **interventions):
        """Securely apply interventions."""
        validated_interventions = {}
        
        for var, value in interventions.items():
            var = validate_variable_name(var)
            var_type = self.environment.variable_types.get(var, "binary")
            validated_value = validate_intervention_value(value, var_type, var)
            validated_interventions[var] = validated_value
        
        return self.environment.intervene(**validated_interventions)
    
    def run_experiment(self, *args, **kwargs):
        """Securely run experiment."""
        self._experiment_count += 1
        if self._experiment_count > 100:  # Daily limit
            raise SecurityError("Daily experiment limit exceeded")
        
        return self.environment.run_experiment(*args, **kwargs)


def create_secure_environment(dag: Optional[Dict] = None) -> SecureEnvironment:
    """Create a secure causal environment.
    
    Args:
        dag: Optional DAG specification
        
    Returns:
        Secure environment instance
    """
    from .core import CausalEnvironment
    
    if dag:
        # Validate DAG structure
        if not isinstance(dag, dict):
            raise ValidationError("DAG must be a dictionary")
        
        # Validate all variable names
        for var, parents in dag.items():
            validate_variable_name(var)
            if not isinstance(parents, list):
                raise ValidationError(f"Parents of '{var}' must be a list")
            for parent in parents:
                validate_variable_name(parent)
        
        # Check graph size
        num_nodes = len(dag)
        num_edges = sum(len(parents) for parents in dag.values())
        validate_graph_size(num_nodes, num_edges)
        
        env = CausalEnvironment.from_dag(dag)
    else:
        env = CausalEnvironment()
    
    return SecureEnvironment(env)


# Input sanitization utilities
def sanitize_belief_query(query: str) -> str:
    """Sanitize belief query string.
    
    Args:
        query: Raw belief query
        
    Returns:
        Sanitized query string
    """
    if not isinstance(query, str):
        raise ValidationError("Belief query must be string")
    
    query = query.strip()
    if not query:
        raise ValidationError("Belief query cannot be empty")
    
    if len(query) > 500:  # Reasonable limit
        raise ValidationError(f"Belief query too long: {len(query)} characters")
    
    # Basic pattern validation for probability expressions
    if not re.match(r'^P\([^)]+\)(?:\|.*)?$', query):
        logger.warning(f"Unusual belief query format: {query}")
    
    return sanitize_html_input(query)


# Audit logging
class SecurityLogger:
    """Security event logger."""
    
    def __init__(self):
        self.logger = logging.getLogger('causal_gym.security')
    
    def log_experiment_start(self, agent_id: str, experiment_type: str):
        """Log experiment start."""
        self.logger.info(f"EXPERIMENT_START: agent={agent_id}, type={experiment_type}")
    
    def log_intervention(self, agent_id: str, intervention: Dict):
        """Log intervention attempt."""
        self.logger.info(f"INTERVENTION: agent={agent_id}, intervention={intervention}")
    
    def log_security_violation(self, event_type: str, details: str):
        """Log security violation."""
        self.logger.warning(f"SECURITY_VIOLATION: type={event_type}, details={details}")
    
    def log_rate_limit_hit(self, function: str, limit: int):
        """Log rate limit violation."""
        self.logger.warning(f"RATE_LIMIT_HIT: function={function}, limit={limit}")


# Create global security logger instance
security_logger = SecurityLogger()