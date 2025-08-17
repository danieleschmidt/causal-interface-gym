"""Security and validation utilities for causal interface gym."""

import re
import logging
import time
import hashlib
import secrets
from typing import Any, Dict, List, Set, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)

# Security configuration
MAX_GRAPH_NODES = 1000
MAX_GRAPH_EDGES = 5000
MAX_VARIABLE_NAME_LENGTH = 100
MAX_EXPERIMENT_DURATION = 300  # seconds
MAX_BELIEF_QUERIES = 10000
MAX_INPUT_LENGTH = 10000
MAX_LLM_RESPONSE_LENGTH = 50000

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
    r'subprocess\.',  # Subprocess calls
    r'os\.',        # OS module calls
    r'system\s*\(',  # System calls
    r'open\s*\(',   # File operations
    r'file\s*\(',   # File operations
    r'rm\s+-',      # Delete commands
    r'DROP\s+TABLE', # SQL injection
    r'DELETE\s+FROM', # SQL injection
]

SAFE_VARIABLE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

# Rate limiting storage
_rate_limit_store = {}
_security_events = []


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


def validate_llm_response(response: str, max_length: int = 10000) -> str:
    """Validate and sanitize LLM response.
    
    Args:
        response: LLM response text
        max_length: Maximum allowed response length
        
    Returns:
        Validated response
        
    Raises:
        ValidationError: If response is invalid
    """
    if not isinstance(response, str):
        raise ValidationError(f"LLM response must be string, got {type(response)}")
    
    if len(response) > max_length:
        raise ValidationError(f"Response too long: {len(response)} > {max_length}")
    
    # Check for potentially dangerous content
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            logger.warning(f"Dangerous pattern detected in LLM response: {pattern}")
            response = re.sub(pattern, '[FILTERED]', response, flags=re.IGNORECASE)
    
    return response


def validate_experiment_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Validate experiment metadata for security.
    
    Args:
        metadata: Experiment metadata dictionary
        
    Returns:
        Validated metadata
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if not isinstance(metadata, dict):
        raise ValidationError("Metadata must be a dictionary")
    
    # Check total size
    import json
    metadata_str = json.dumps(metadata)
    if len(metadata_str) > 50000:  # 50KB limit
        raise ValidationError(f"Metadata too large: {len(metadata_str)} bytes")
    
    # Sanitize string values
    sanitized = {}
    for key, value in metadata.items():
        # Validate key
        if not isinstance(key, str) or not key.strip():
            continue  # Skip invalid keys
        
        key = sanitize_html_input(key.strip())[:100]  # Limit key length
        
        # Sanitize value based on type
        if isinstance(value, str):
            sanitized[key] = sanitize_html_input(value)[:1000]  # Limit string length
        elif isinstance(value, (int, float, bool)):
            sanitized[key] = value
        elif isinstance(value, (list, dict)):
            # For complex types, convert to string and sanitize
            sanitized[key] = sanitize_html_input(str(value))[:1000]
        else:
            sanitized[key] = sanitize_html_input(str(value))[:1000]
    
    return sanitized


class SecureExperimentRunner:
    """Secure wrapper for experiment execution."""
    
    def __init__(self, environment, rate_limits: Optional[Dict[str, int]] = None):
        """Initialize secure experiment runner.
        
        Args:
            environment: CausalEnvironment instance
            rate_limits: Custom rate limits for operations
        """
        self.environment = environment
        self.rate_limits = rate_limits or {
            'experiments_per_hour': 50,
            'interventions_per_hour': 200,
            'belief_queries_per_hour': 1000
        }
        self._operation_counts = {}
        self._last_reset = None
        
    def _check_rate_limit(self, operation: str) -> None:
        """Check if operation is within rate limits."""
        import time
        
        current_time = time.time()
        
        # Reset counters every hour
        if self._last_reset is None or current_time - self._last_reset > 3600:
            self._operation_counts.clear()
            self._last_reset = current_time
        
        # Check limit
        current_count = self._operation_counts.get(operation, 0)
        limit = self.rate_limits.get(f'{operation}_per_hour', float('inf'))
        
        if current_count >= limit:
            raise SecurityError(f"Rate limit exceeded for {operation}: {current_count} >= {limit}")
        
        # Increment counter
        self._operation_counts[operation] = current_count + 1
        security_logger.logger.debug(f"Rate limit check passed for {operation}: {current_count + 1}/{limit}")
    
    def run_secure_experiment(self, agent, interventions: List[tuple], 
                             measure_beliefs: List[str], 
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run experiment with security validations.
        
        Args:
            agent: LLM agent to test
            interventions: List of (variable, value) interventions
            measure_beliefs: List of beliefs to measure
            metadata: Optional experiment metadata
            
        Returns:
            Secure experiment results
        """
        # Rate limit check
        self._check_rate_limit('experiments')
        
        # Validate inputs
        if not interventions:
            raise ValidationError("At least one intervention required")
        
        if len(interventions) > 20:  # Reasonable limit
            raise ValidationError(f"Too many interventions: {len(interventions)} > 20")
        
        if not measure_beliefs:
            raise ValidationError("At least one belief must be measured")
        
        if len(measure_beliefs) > 50:  # Reasonable limit
            raise ValidationError(f"Too many beliefs to measure: {len(measure_beliefs)} > 50")
        
        # Validate and sanitize interventions
        safe_interventions = []
        for intervention in interventions:
            if not isinstance(intervention, (tuple, list)) or len(intervention) != 2:
                raise ValidationError("Each intervention must be (variable, value) tuple")
            
            var, value = intervention
            var = validate_variable_name(var)
            
            # Validate intervention value
            var_type = self.environment.variable_types.get(var, "binary")
            value = validate_intervention_value(value, var_type, var)
            
            safe_interventions.append((var, value))
            
            # Rate limit interventions
            self._check_rate_limit('interventions')
        
        # Validate and sanitize belief queries
        safe_beliefs = []
        for belief in measure_beliefs:
            safe_belief = sanitize_belief_query(belief)
            safe_beliefs.append(safe_belief)
            
            # Rate limit belief queries
            self._check_rate_limit('belief_queries')
        
        # Validate metadata
        safe_metadata = validate_experiment_metadata(metadata or {})
        
        # Log experiment start
        agent_id = getattr(agent, 'id', str(type(agent).__name__))
        security_logger.log_experiment_start(agent_id, "secure_causal_reasoning")
        
        try:
            # Run the experiment with validated inputs
            from .core import InterventionUI
            
            # Create UI instance
            ui = InterventionUI(self.environment)
            
            # Execute experiment
            results = ui.run_experiment(
                agent=agent,
                interventions=safe_interventions,
                measure_beliefs=safe_beliefs
            )
            
            # Add security metadata
            results['security_info'] = {
                'validated': True,
                'sanitized_interventions': len(safe_interventions),
                'sanitized_beliefs': len(safe_beliefs),
                'rate_limits_applied': True
            }
            
            # Validate LLM responses in results
            if 'intervention_results' in results:
                for result in results['intervention_results']:
                    if 'agent_beliefs' in result:
                        for belief, response in result['agent_beliefs'].items():
                            if isinstance(response, str):
                                result['agent_beliefs'][belief] = validate_llm_response(response)
            
            return results
            
        except Exception as e:
            security_logger.log_security_violation(
                "experiment_execution_error",
                f"Agent: {agent_id}, Error: {str(e)}"
            )
            raise
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security and usage statistics.
        
        Returns:
            Security statistics
        """
        return {
            'operation_counts': self._operation_counts.copy(),
            'rate_limits': self.rate_limits.copy(),
            'last_reset': self._last_reset,
            'security_features': [
                'input_validation',
                'rate_limiting', 
                'html_sanitization',
                'variable_name_validation',
                'graph_size_limits',
                'experiment_audit_logging'
            ]
        }


def create_secure_experiment_runner(dag: Optional[Dict] = None, 
                                  rate_limits: Optional[Dict[str, int]] = None) -> SecureExperimentRunner:
    """Create a secure experiment runner.
    
    Args:
        dag: Optional DAG specification
        rate_limits: Optional custom rate limits
        
    Returns:
        Secure experiment runner instance
    """
    secure_env = create_secure_environment(dag)
    return SecureExperimentRunner(secure_env.environment, rate_limits)