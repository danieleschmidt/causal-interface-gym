"""Production configuration for Terragon Causal Interface Gym."""

import os
from typing import List, Optional

class ProductionConfig:
    """Production environment configuration."""
    
    # Basic settings
    ENV = "production"
    DEBUG = False
    TESTING = False
    
    # Server settings
    HOST = "0.0.0.0"
    PORT = int(os.getenv("PORT", "8000"))
    WORKERS = int(os.getenv("WORKER_PROCESSES", "4"))
    
    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
    CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
    
    # Database settings
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Caching
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Monitoring
    PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
    HEALTH_CHECK_PORT = int(os.getenv("HEALTH_CHECK_PORT", "8080"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    MAX_REQUESTS = int(os.getenv("MAX_REQUESTS", "1000"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    # Features
    ENABLE_METRICS = True
    ENABLE_TRACING = True
    ENABLE_PROFILING = False
