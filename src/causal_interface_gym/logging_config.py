"""Comprehensive logging configuration for causal interface gym."""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'message', 'exc_info', 'exc_text',
                          'stack_info']:
                log_obj[key] = value
        
        return json.dumps(log_obj)


class CausalGymFilter(logging.Filter):
    """Filter for causal gym specific logs."""
    
    def filter(self, record):
        """Filter records to only include causal gym related logs."""
        return record.name.startswith('causal_interface_gym') or record.name.startswith('causal_gym')


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_structured: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """Set up comprehensive logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_structured: Use structured JSON format
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    if enable_structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        console_handler.addFilter(CausalGymFilter())
        root_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # General log file
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "causal_gym.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        file_handler.addFilter(CausalGymFilter())
        root_logger.addHandler(file_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "causal_gym_errors.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        error_handler.addFilter(CausalGymFilter())
        root_logger.addHandler(error_handler)
        
        # Security log file
        security_handler = logging.handlers.RotatingFileHandler(
            log_dir / "causal_gym_security.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(detailed_formatter)
        
        # Filter for security-related logs
        def security_filter(record):
            return (record.name.startswith('causal_gym.security') or 
                   'security' in record.getMessage().lower() or
                   'violation' in record.getMessage().lower())
        
        security_handler.addFilter(security_filter)
        root_logger.addHandler(security_handler)
        
        # Performance log file
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / "causal_gym_performance.log",
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter if enable_structured else detailed_formatter)
        
        # Filter for performance-related logs
        def performance_filter(record):
            perf_keywords = ['performance', 'metrics', 'timing', 'latency', 'throughput']
            return any(keyword in record.getMessage().lower() for keyword in perf_keywords)
        
        perf_handler.addFilter(performance_filter)
        root_logger.addHandler(perf_handler)
    
    # Set specific logger levels
    logging.getLogger('causal_interface_gym').setLevel(getattr(logging, level.upper()))
    logging.getLogger('causal_gym.security').setLevel(logging.WARNING)
    logging.getLogger('causal_gym.performance').setLevel(logging.INFO)
    
    # Suppress noisy third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('networkx').setLevel(logging.WARNING)


class TimingLogger:
    """Context manager for timing operations."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        """Initialize timing logger.
        
        Args:
            operation: Name of operation being timed
            logger: Logger instance to use
        """
        self.operation = operation
        self.logger = logger or logging.getLogger('causal_gym.performance')
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        self.logger.debug(f"Started {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            if exc_type:
                self.logger.error(f"{self.operation} failed after {duration:.3f}s: {exc_val}")
            else:
                self.logger.info(f"{self.operation} completed in {duration:.3f}s")


def log_experiment_start(agent_id: str, experiment_type: str, interventions: list):
    """Log experiment start with details."""
    logger = logging.getLogger('causal_interface_gym.experiments')
    logger.info(f"Experiment started", extra={
        'event_type': 'experiment_start',
        'agent_id': agent_id,
        'experiment_type': experiment_type,
        'intervention_count': len(interventions),
        'interventions': interventions
    })


def log_intervention(agent_id: str, intervention: dict, result: dict):
    """Log intervention details."""
    logger = logging.getLogger('causal_interface_gym.interventions')
    logger.info(f"Intervention applied", extra={
        'event_type': 'intervention',
        'agent_id': agent_id,
        'intervention': intervention,
        'success': 'error' not in str(result).lower()
    })


def log_causal_analysis(analysis_type: str, variables: list, result: dict):
    """Log causal analysis results."""
    logger = logging.getLogger('causal_interface_gym.analysis')
    logger.info(f"Causal analysis completed", extra={
        'event_type': 'causal_analysis',
        'analysis_type': analysis_type,
        'variables': variables,
        'identifiable': result.get('identifiable', False),
        'causal_effect': result.get('causal_effect', None)
    })


def log_security_event(event_type: str, details: dict, severity: str = 'WARNING'):
    """Log security-related events."""
    logger = logging.getLogger('causal_gym.security')
    log_func = getattr(logger, severity.lower())
    log_func(f"Security event: {event_type}", extra={
        'event_type': 'security_event',
        'security_event_type': event_type,
        'severity': severity,
        **details
    })


def log_performance_metric(metric_name: str, value: float, unit: str = "", context: dict = None):
    """Log performance metrics."""
    logger = logging.getLogger('causal_gym.performance')
    logger.info(f"Performance metric: {metric_name} = {value}{unit}", extra={
        'event_type': 'performance_metric',
        'metric_name': metric_name,
        'value': value,
        'unit': unit,
        'context': context or {}
    })


# Initialize default logging
def init_default_logging():
    """Initialize default logging configuration."""
    setup_logging(
        level="INFO",
        enable_console=True,
        enable_file=True,
        enable_structured=False
    )


# Context managers for logging
class ExperimentLogger:
    """Context manager for experiment logging."""
    
    def __init__(self, agent_id: str, experiment_type: str):
        self.agent_id = agent_id
        self.experiment_type = experiment_type
        self.logger = logging.getLogger('causal_interface_gym.experiments')
    
    def __enter__(self):
        self.logger.info(f"Starting experiment: {self.experiment_type} with agent {self.agent_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"Experiment {self.experiment_type} failed: {exc_val}")
        else:
            self.logger.info(f"Experiment {self.experiment_type} completed successfully")


# Auto-initialize logging if not already configured
if not logging.getLogger().handlers:
    init_default_logging()