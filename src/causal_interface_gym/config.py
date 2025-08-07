"""Configuration management for causal interface gym."""

import os
import json
import yaml
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass 
class SecurityConfig:
    """Security configuration settings."""
    max_graph_nodes: int = 1000
    max_graph_edges: int = 5000
    max_variable_name_length: int = 100
    max_experiment_duration: int = 300
    max_belief_queries: int = 10000
    rate_limit_calls: int = 100
    rate_limit_window: int = 60
    enable_input_sanitization: bool = True
    enable_audit_logging: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration settings."""
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_cache_size: int = 1000
    enable_performance_monitoring: bool = True
    metric_retention_hours: int = 24
    health_check_interval: int = 60
    max_concurrent_experiments: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    enable_console: bool = True
    enable_file: bool = True
    enable_structured: bool = False
    log_dir: str = "logs"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///causal_gym.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    enable_migrations: bool = True


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    default_provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-sonnet"
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.1


@dataclass
class UIConfig:
    """UI configuration settings."""
    default_theme: str = "light"
    enable_live_updates: bool = True
    auto_save_experiments: bool = True
    max_graph_nodes_display: int = 100
    animation_duration: int = 300


@dataclass
class CausalGymConfig:
    """Main configuration class."""
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    debug: bool = False
    environment: str = "development"


class ConfigManager:
    """Configuration manager with file loading and environment overrides."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config = CausalGymConfig()
        self._loaded = False
        
    def load_config(self) -> CausalGymConfig:
        """Load configuration from file and environment variables.
        
        Returns:
            Loaded configuration
        """
        if self._loaded and self._config:
            return self._config
            
        # Load from file if specified
        if self.config_path and self.config_path.exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self._validate_config()
        
        self._loaded = True
        logger.info(f"Configuration loaded from {self.config_path or 'environment'}")
        
        return self._config
    
    def _load_from_file(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._update_config_from_dict(data)
            logger.info(f"Configuration loaded from file: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            raise
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Security
            'CAUSAL_GYM_MAX_GRAPH_NODES': ('security.max_graph_nodes', int),
            'CAUSAL_GYM_MAX_GRAPH_EDGES': ('security.max_graph_edges', int),
            'CAUSAL_GYM_RATE_LIMIT_CALLS': ('security.rate_limit_calls', int),
            
            # Performance
            'CAUSAL_GYM_CACHE_ENABLED': ('performance.cache_enabled', self._parse_bool),
            'CAUSAL_GYM_CACHE_TTL': ('performance.cache_ttl', int),
            'CAUSAL_GYM_MAX_CONCURRENT': ('performance.max_concurrent_experiments', int),
            
            # Logging
            'CAUSAL_GYM_LOG_LEVEL': ('logging.level', str),
            'CAUSAL_GYM_LOG_DIR': ('logging.log_dir', str),
            'CAUSAL_GYM_STRUCTURED_LOGGING': ('logging.enable_structured', self._parse_bool),
            
            # Database
            'CAUSAL_GYM_DATABASE_URL': ('database.url', str),
            'CAUSAL_GYM_DB_POOL_SIZE': ('database.pool_size', int),
            
            # LLM
            'OPENAI_API_KEY': ('llm.openai_api_key', str),
            'ANTHROPIC_API_KEY': ('llm.anthropic_api_key', str),
            'CAUSAL_GYM_LLM_PROVIDER': ('llm.default_provider', str),
            'CAUSAL_GYM_LLM_TIMEOUT': ('llm.timeout', int),
            
            # General
            'CAUSAL_GYM_DEBUG': ('debug', self._parse_bool),
            'CAUSAL_GYM_ENVIRONMENT': ('environment', str),
        }
        
        for env_var, (config_path, parser) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    parsed_value = parser(value)
                    self._set_nested_attr(self._config, config_path, parsed_value)
                    logger.debug(f"Set {config_path} = {parsed_value} from {env_var}")
                except Exception as e:
                    logger.warning(f"Failed to parse environment variable {env_var}={value}: {e}")
    
    def _parse_bool(self, value: str) -> bool:
        """Parse boolean from string."""
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    def _update_config_from_dict(self, data: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section, values in data.items():
            if hasattr(self._config, section) and isinstance(values, dict):
                section_obj = getattr(self._config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def _set_nested_attr(self, obj: Any, path: str, value: Any):
        """Set nested attribute using dot notation."""
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    def _validate_config(self):
        """Validate configuration values."""
        # Security validation
        if self._config.security.max_graph_nodes <= 0:
            raise ValueError("max_graph_nodes must be positive")
        if self._config.security.max_graph_edges <= 0:
            raise ValueError("max_graph_edges must be positive")
        
        # Performance validation
        if self._config.performance.cache_ttl < 0:
            raise ValueError("cache_ttl must be non-negative")
        if self._config.performance.max_concurrent_experiments <= 0:
            raise ValueError("max_concurrent_experiments must be positive")
        
        # LLM validation
        if self._config.llm.temperature < 0 or self._config.llm.temperature > 2:
            logger.warning(f"LLM temperature {self._config.llm.temperature} is outside typical range [0, 2]")
        
        # Logging validation
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._config.logging.level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self._config.logging.level}")
    
    def save_config(self, path: Optional[Union[str, Path]] = None):
        """Save current configuration to file.
        
        Args:
            path: Path to save configuration (default: original path)
        """
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No save path specified")
        
        # Convert config to dictionary
        config_dict = self._config_to_dict()
        
        # Save based on file extension
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        
        return dataclass_to_dict(self._config)
    
    @property
    def config(self) -> CausalGymConfig:
        """Get current configuration."""
        if not self._loaded:
            return self.load_config()
        return self._config


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> CausalGymConfig:
    """Get global configuration instance.
    
    Returns:
        Configuration object
    """
    return _config_manager.load_config()


def load_config(config_path: Union[str, Path]) -> CausalGymConfig:
    """Load configuration from specific file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    manager = ConfigManager(config_path)
    return manager.load_config()


def create_default_config_file(path: Union[str, Path], format: str = 'yaml'):
    """Create a default configuration file.
    
    Args:
        path: Path to create configuration file
        format: File format ('yaml' or 'json')
    """
    config = CausalGymConfig()
    manager = ConfigManager()
    manager._config = config
    
    if format.lower() == 'yaml':
        save_path = Path(path).with_suffix('.yaml')
    else:
        save_path = Path(path).with_suffix('.json')
    
    manager.save_config(save_path)
    logger.info(f"Default configuration created at {save_path}")


# Environment-specific configurations
def get_development_config() -> CausalGymConfig:
    """Get development environment configuration."""
    config = CausalGymConfig()
    config.debug = True
    config.environment = "development"
    config.logging.level = "DEBUG"
    config.database.url = "sqlite:///causal_gym_dev.db"
    config.database.echo = True
    return config


def get_production_config() -> CausalGymConfig:
    """Get production environment configuration."""
    config = CausalGymConfig()
    config.debug = False
    config.environment = "production"
    config.logging.level = "INFO"
    config.logging.enable_structured = True
    config.security.enable_audit_logging = True
    config.performance.enable_performance_monitoring = True
    return config


def get_testing_config() -> CausalGymConfig:
    """Get testing environment configuration."""
    config = CausalGymConfig()
    config.debug = True
    config.environment = "testing"
    config.logging.level = "WARNING"
    config.database.url = "sqlite:///:memory:"
    config.security.max_graph_nodes = 100
    config.performance.cache_enabled = False
    return config