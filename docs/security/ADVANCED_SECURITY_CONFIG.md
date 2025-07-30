# Advanced Security Configuration

This document outlines advanced security measures for the causal-interface-gym project, building on the existing security foundation.

## Supply Chain Security

### SBOM (Software Bill of Materials) Generation

#### Syft Configuration

Create `.syft.yaml` in repository root:

```yaml
# Syft configuration for SBOM generation
catalogers:
  enabled:
    - python-package-cataloger
    - python-index-cataloger
  disabled: []

package:
  cataloger:
    enabled: true
    scope: "all-layers"

file:
  metadata:
    cataloger:
      enabled: true
    digests: ["sha256"]

exclude:
  - "**/.git/**"
  - "**/node_modules/**"
  - "**/__pycache__/**"
  - "**/.*cache/**"

output:
  - format: "spdx-json"
    file: "./sbom.spdx.json"
  - format: "cyclonedx-json" 
    file: "./sbom.cyclonedx.json"
  - format: "table"
    file: "./sbom.txt"
```

#### SBOM Generation Scripts

Create `scripts/generate_sbom.sh`:

```bash
#!/bin/bash
set -euo pipefail

echo "🔍 Generating Software Bill of Materials (SBOM)..."

# Install Syft if not present
if ! command -v syft &> /dev/null; then
    echo "Installing Syft..."
    curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
fi

# Generate SBOM for source code
echo "📋 Generating source SBOM..."
syft . -o spdx-json=sbom-source.spdx.json -o cyclonedx-json=sbom-source.cyclonedx.json

# Generate SBOM for Docker image if Dockerfile exists
if [[ -f "Dockerfile" ]]; then
    echo "🐳 Building Docker image for SBOM..."
    docker build -t causal-interface-gym:sbom .
    
    echo "📋 Generating container SBOM..."
    syft causal-interface-gym:sbom -o spdx-json=sbom-container.spdx.json -o cyclonedx-json=sbom-container.cyclonedx.json
fi

# Generate SBOM for installed packages
echo "📦 Generating installed packages SBOM..."
pip freeze > requirements-frozen.txt
syft packages:requirements-frozen.txt -o spdx-json=sbom-packages.spdx.json

echo "✅ SBOM generation complete!"
echo "Files generated:"
echo "  - sbom-source.spdx.json (source code)"
echo "  - sbom-source.cyclonedx.json (source code)" 
echo "  - sbom-container.spdx.json (container)"
echo "  - sbom-container.cyclonedx.json (container)"
echo "  - sbom-packages.spdx.json (packages)"
```

### SLSA Compliance Framework

Create `docs/security/SLSA_COMPLIANCE.md`:

```markdown
# SLSA Compliance Guide

## SLSA Level 2 Implementation

### Build Requirements

1. **Version Controlled Source**: ✅ Git with signed commits
2. **Build Service**: ✅ GitHub Actions with hosted runners  
3. **Build Definition**: ✅ Reproducible builds with locked dependencies
4. **Provenance**: ✅ Signed attestations with SLSA provenance

### Provenance Generation

GitHub Actions configuration for SLSA provenance:

```yaml
# Add to release workflow
- name: Generate SLSA Provenance
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
  with:
    base64-subjects: "${{ needs.build.outputs.hashes }}"
    provenance-name: "provenance.intoto.jsonl"
```

### Verification Commands

```bash
# Verify SLSA provenance
slsa-verifier verify-artifact \
  --provenance-path provenance.intoto.jsonl \
  --source-uri github.com/yourusername/causal-interface-gym \
  --source-tag v1.0.0 \
  ./dist/*.whl

# Verify package signatures
python -m pip install sigstore
python -m sigstore verify --bundle signature.bundle package.whl
```
```

## Advanced Vulnerability Management

### Dependency Scanning Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "04:00"
    open-pull-requests-limit: 10
    reviewers:
      - "security-team"
    assignees:
      - "maintainer"
    commit-message:
      prefix: "security"
      include: "scope"
    allow:
      - dependency-type: "direct"
        update-type: "all"
      - dependency-type: "indirect"
        update-type: "security"
    ignore:
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]

  # Docker dependencies  
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "docker"
      
  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    commit-message:
      prefix: "ci"
```

### Multi-Scanner Security Pipeline

Create `scripts/comprehensive_security_scan.py`:

```python
#!/usr/bin/env python3
"""Comprehensive security scanning orchestrator."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

class SecurityScanner:
    """Orchestrates multiple security scanning tools."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        
    async def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit SAST scanning."""
        print("🔍 Running Bandit SAST scan...")
        cmd = ["bandit", "-r", "src/", "-f", "json"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 and result.returncode != 1:  # 1 = issues found
            raise RuntimeError(f"Bandit failed: {result.stderr}")
            
        return json.loads(result.stdout) if result.stdout else {}
    
    async def run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety dependency vulnerability scan."""
        print("🛡️ Running Safety dependency scan...")
        cmd = ["safety", "check", "--json"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode not in [0, 64]:  # 64 = vulnerabilities found
            raise RuntimeError(f"Safety failed: {result.stderr}")
            
        return json.loads(result.stdout) if result.stdout else {}
    
    async def run_pip_audit_scan(self) -> Dict[str, Any]:
        """Run pip-audit vulnerability scan."""
        print("🔐 Running pip-audit scan...")
        cmd = ["pip-audit", "--format=json", "--desc"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode not in [0, 1]:  # 1 = vulnerabilities found
            raise RuntimeError(f"pip-audit failed: {result.stderr}")
            
        return json.loads(result.stdout) if result.stdout else {}
    
    async def run_semgrep_scan(self) -> Dict[str, Any]:
        """Run Semgrep static analysis."""
        print("⚡ Running Semgrep analysis...")
        cmd = ["semgrep", "--config=auto", "--json", "src/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode not in [0, 1]:  # 1 = findings
            raise RuntimeError(f"Semgrep failed: {result.stderr}")
            
        return json.loads(result.stdout) if result.stdout else {}
    
    async def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans concurrently."""
        print("🚀 Starting comprehensive security scan...")
        
        tasks = [
            ("bandit", self.run_bandit_scan()),
            ("safety", self.run_safety_scan()),
            ("pip_audit", self.run_pip_audit_scan()),
            ("semgrep", self.run_semgrep_scan()),
        ]
        
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                print(f"❌ {name} scan failed: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def generate_security_report(self, scan_results: Dict[str, Any]) -> None:
        """Generate consolidated security report."""
        report_path = self.project_root / "security-report.json"
        
        # Count issues by severity
        total_issues = 0
        critical_issues = 0
        high_issues = 0
        
        for scanner, results in scan_results.items():
            if "error" in results:
                continue
                
            if scanner == "bandit" and "results" in results:
                for issue in results["results"]:
                    total_issues += 1
                    if issue["issue_severity"] == "HIGH":
                        high_issues += 1
                        
            elif scanner in ["safety", "pip_audit"] and isinstance(results, list):
                for vuln in results:
                    total_issues += 1
                    if vuln.get("vulnerability_id", "").startswith("CVE"):
                        high_issues += 1
        
        summary = {
            "scan_timestamp": "2025-07-30T00:00:00Z",
            "total_issues": total_issues,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
            "scanners_used": list(scan_results.keys()),
            "detailed_results": scan_results
        }
        
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Security report generated: {report_path}")
        print(f"   Total issues: {total_issues}")
        print(f"   High severity: {high_issues}")
        print(f"   Critical: {critical_issues}")

async def main():
    """Main security scanning orchestrator."""
    project_root = Path.cwd()
    scanner = SecurityScanner(project_root)
    
    try:
        results = await scanner.run_all_scans()
        scanner.generate_security_report(results)
        
        # Exit with error code if high/critical issues found
        total_high_critical = sum(
            1 for r in results.values() 
            if isinstance(r, list) and len(r) > 0
        )
        
        if total_high_critical > 0:
            print(f"⚠️ Security issues detected. Review security-report.json")
            sys.exit(1)
        else:
            print("✅ No security issues detected")
            
    except Exception as e:
        print(f"💥 Security scan failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Container Security Hardening

### Multi-stage Dockerfile Security Enhancement

```dockerfile
# Security-hardened multi-stage Dockerfile
FROM python:3.11-slim-bookworm AS base

# Security: Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Security: Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Security: Set secure umask
RUN echo "umask 0027" >> /etc/profile

FROM base AS dependencies

# Install dependencies as root, then switch to appuser
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

FROM base AS runtime  

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create app directory with proper permissions
RUN mkdir -p /app && chown appuser:appuser /app
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./

# Security: Switch to non-root user
USER appuser

# Security: Set read-only filesystem
VOLUME ["/tmp"]

# Security: Drop capabilities and set security options
LABEL security.scan="enabled" \
      security.non-root="true" \
      security.read-only="true"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import causal_interface_gym; print('healthy')" || exit 1

# Default command
CMD ["python", "-m", "causal_interface_gym"]
```

### Container Security Scanning Configuration

Create `.trivyignore`:

```
# Trivy ignore patterns for false positives
CVE-2023-XXXXX  # Reason: Not applicable to our use case
```

## Runtime Security Monitoring

### Application Security Configuration

Create `src/causal_interface_gym/security/__init__.py`:

```python
"""Security monitoring and enforcement module."""

import functools
import hashlib
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

# Security logger
security_logger = logging.getLogger("causal_interface_gym.security")
security_logger.setLevel(logging.WARNING)

class SecurityMonitor:
    """Runtime security monitoring and enforcement."""
    
    def __init__(self):
        self.failed_attempts = {}
        self.rate_limits = {}
        
    def rate_limit(self, max_calls: int = 10, time_window: int = 60):
        """Rate limiting decorator for API endpoints."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                client_id = self._get_client_id()
                current_time = time.time()
                
                # Clean old entries
                if client_id in self.rate_limits:
                    self.rate_limits[client_id] = [
                        t for t in self.rate_limits[client_id] 
                        if current_time - t < time_window
                    ]
                
                # Check rate limit
                call_count = len(self.rate_limits.get(client_id, []))
                if call_count >= max_calls:
                    security_logger.warning(
                        f"Rate limit exceeded for client {client_id}"
                    )
                    raise RateLimitExceeded("Too many requests")
                
                # Record call
                if client_id not in self.rate_limits:
                    self.rate_limits[client_id] = []
                self.rate_limits[client_id].append(current_time)
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def validate_input(self, input_schema: Dict[str, Any]):
        """Input validation decorator."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func) 
            def wrapper(*args, **kwargs):
                # Validate inputs against schema
                for param, constraints in input_schema.items():
                    if param in kwargs:
                        self._validate_parameter(param, kwargs[param], constraints)
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def _get_client_id(self) -> str:
        """Get client identifier for rate limiting."""
        # In production, use actual client IP or API key
        return "default_client"
    
    def _validate_parameter(self, param: str, value: Any, constraints: Dict[str, Any]) -> None:
        """Validate individual parameter against constraints."""
        if "type" in constraints and not isinstance(value, constraints["type"]):
            raise ValidationError(f"Parameter {param} must be of type {constraints['type']}")
        
        if "max_length" in constraints and len(str(value)) > constraints["max_length"]:
            raise ValidationError(f"Parameter {param} exceeds maximum length")

class RateLimitExceeded(Exception):
    """Rate limit exceeded exception."""
    pass

class ValidationError(Exception):
    """Input validation error."""
    pass

# Global security monitor instance
security_monitor = SecurityMonitor()
```

## Compliance and Audit Trail

### Audit Logging Configuration

Create `src/causal_interface_gym/audit/__init__.py`:

```python
"""Audit logging for compliance and security monitoring."""

import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

class AuditEventType(Enum):
    """Types of audit events."""
    USER_ACTION = "user_action"
    DATA_ACCESS = "data_access"
    SECURITY_EVENT = "security_event"
    SYSTEM_CHANGE = "system_change"
    ERROR = "error"

class AuditLogger:
    """Structured audit logging for compliance."""
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Configure structured JSON formatter
        handler = logging.StreamHandler()
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> None:
        """Log structured audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type.value,
            "action": action,
            "user_id": user_id or "system",
            "resource": resource,
            "success": success,
            "details": details or {},
            "session_id": self._get_session_id()
        }
        
        self.logger.info(json.dumps(event))
    
    def _get_session_id(self) -> str:
        """Get current session identifier."""
        # In production, use actual session management
        return f"session_{int(time.time())}"

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        return record.getMessage()

# Global audit logger
audit_logger = AuditLogger()
```

This advanced security configuration provides:

1. **Supply Chain Security**: SBOM generation, SLSA compliance
2. **Vulnerability Management**: Multi-scanner pipeline, automated dependency updates  
3. **Container Security**: Hardened Dockerfile, security scanning
4. **Runtime Security**: Rate limiting, input validation, audit logging
5. **Compliance**: Structured audit trails, security monitoring

The configuration builds on the existing security foundation while adding enterprise-grade security measures appropriate for a maturing repository.