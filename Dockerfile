# Multi-stage build for causal-interface-gym
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e ".[dev,ui]"

# Development stage
FROM base as development

# Copy source code
COPY --chown=appuser:appuser . .

# Install development dependencies
RUN pip install -e ".[dev,ui,docs]"

# Switch to non-root user
USER appuser

# Expose port for development server
EXPOSE 8501

# Default command for development
CMD ["python", "-m", "streamlit", "run", "examples/basic_usage.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser pyproject.toml /app/
COPY --chown=appuser:appuser README.md /app/

# Install production dependencies only
RUN pip install -e ".[ui]"

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import causal_interface_gym; print('OK')" || exit 1

# Default command for production
CMD ["python", "-c", "import causal_interface_gym; print('Causal Interface Gym is ready')"]

# Testing stage
FROM development as testing

# Run tests
RUN make test

# Benchmarking stage
FROM development as benchmarking

# Install additional benchmark dependencies
RUN pip install pytest-benchmark memory_profiler

# Run benchmarks
CMD ["pytest", "tests/benchmarks/", "--benchmark-json=benchmark-results.json"]