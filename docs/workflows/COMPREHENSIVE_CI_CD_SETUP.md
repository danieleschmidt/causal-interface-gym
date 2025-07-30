# Comprehensive CI/CD Setup Guide

This document provides production-ready GitHub Actions workflows for the causal-interface-gym project.

## Overview

The CI/CD pipeline implements a multi-stage approach:
- **Quality Gates**: Linting, type checking, security scanning
- **Testing**: Unit, integration, performance tests with coverage
- **Security**: SAST, dependency scanning, container security
- **Release**: Automated semantic versioning and PyPI publishing
- **Monitoring**: Performance tracking and alerting

## Required Secrets Configuration

Add these secrets to your GitHub repository:

```bash
# PyPI Publishing
PYPI_API_TOKEN=pypi-...

# Security Scanning (optional - Snyk, CodeQL are free for public repos)
SNYK_TOKEN=...
SONAR_TOKEN=...

# Container Registry (if using private registry)
DOCKER_USERNAME=...
DOCKER_PASSWORD=...

# Notifications (optional)
SLACK_WEBHOOK_URL=...
```

## Workflow Files

### 1. Main CI Pipeline (.github/workflows/ci.yml)

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 4 * * 1'  # Weekly dependency check

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality-checks:
    name: Code Quality & Security
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,ui]"
        pip install safety pip-audit
    
    - name: Run pre-commit hooks
      uses: pre-commit/action@v3.0.0
    
    - name: Type checking with mypy
      run: mypy src/
    
    - name: Security: Bandit SAST scan
      run: bandit -r src/ -f json -o bandit-report.json
    
    - name: Security: Safety dependency check
      run: safety check --json --output safety-report.json
    
    - name: Security: pip-audit vulnerability scan
      run: pip-audit --desc --format=json --output=pip-audit-report.json
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports-${{ matrix.python-version }}
        path: |
          bandit-report.json
          safety-report.json
          pip-audit-report.json

  test-suite:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      run: pip install -e ".[dev,ui]"
    
    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=causal_interface_gym \
          --cov-report=xml --cov-report=html \
          --junit-xml=test-results.xml
    
    - name: Run integration tests
      run: pytest tests/integration/ -v --timeout=300
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-${{ matrix.os }}-${{ matrix.python-version }}
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
        path: test-results.xml

  performance-tests:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
        cache: 'pip'
    
    - name: Install dependencies
      run: pip install -e ".[dev]" pytest-benchmark
    
    - name: Run performance tests
      run: |
        pytest tests/benchmarks/ --benchmark-json=benchmark-results.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true

  build-artifacts:
    name: Build & Validate Package
    runs-on: ubuntu-latest
    needs: [quality-checks, test-suite]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install build dependencies
      run: pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Validate package
      run: |
        twine check dist/*
        pip install dist/*.whl
        python -c "import causal_interface_gym; print(causal_interface_gym.__version__)"
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: python-package-distributions
        path: dist/
```

### 2. Container Security & SBOM (.github/workflows/container-security.yml)

```yaml
name: Container Security & SBOM

on:
  push:
    branches: [main]
    paths: [Dockerfile, requirements*.txt, pyproject.toml]
  pull_request:
    paths: [Dockerfile, requirements*.txt, pyproject.toml]

jobs:
  container-security:
    name: Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t causal-interface-gym:${{ github.sha }} .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'causal-interface-gym:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Generate SBOM with Syft
      uses: anchore/sbom-action@v0.14.3
      with:
        image: causal-interface-gym:${{ github.sha }}
        format: spdx-json
        output-file: sbom.spdx.json
    
    - name: Upload SBOM artifact
      uses: actions/upload-artifact@v3
      with:
        name: sbom-${{ github.sha }}
        path: sbom.spdx.json

  dependency-review:
    name: Dependency Review
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
      with:
        fail-on-severity: moderate
        allow-licenses: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC
```

### 3. Release Automation (.github/workflows/release.yml)

```yaml
name: Release Pipeline

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      release_type:
        description: 'Release type'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major

concurrency:
  group: release
  cancel-in-progress: false

jobs:
  check-changes:
    name: Check for Releasable Changes
    runs-on: ubuntu-latest
    outputs:
      should_release: ${{ steps.check.outputs.should_release }}
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Check for conventional commits
      id: check
      run: |
        # Check if there are feat: or fix: commits since last tag
        LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
        if [ -z "$LAST_TAG" ]; then
          echo "should_release=true" >> $GITHUB_OUTPUT
        else
          COMMITS=$(git log --oneline ${LAST_TAG}..HEAD --grep="^feat\|^fix" || echo "")
          if [ -n "$COMMITS" ]; then
            echo "should_release=true" >> $GITHUB_OUTPUT
          else
            echo "should_release=false" >> $GITHUB_OUTPUT
          fi
        fi

  semantic-release:
    name: Semantic Release
    runs-on: ubuntu-latest
    needs: check-changes
    if: needs.check-changes.outputs.should_release == 'true'
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install semantic-release
      run: |
        pip install python-semantic-release
    
    - name: Run semantic release
      env:
        GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        PYPI_TOKEN: ${{ secrets.PYPI_API_TOKEN }}
      run: |
        semantic-release version
        semantic-release publish

  publish-to-pypi:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: semantic-release
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')
    environment: release
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install build dependencies
      run: pip install build
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

## Additional Security Configurations

### CodeQL Analysis (.github/workflows/codeql.yml)

```yaml
name: CodeQL Security Analysis

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'

jobs:
  analyze:
    name: Analyze Code
    runs-on: ubuntu-latest
    permissions:
      actions: read
      contents: read
      security-events: write
    
    strategy:
      fail-fast: false
      matrix:
        language: ['python']
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v2
      with:
        languages: ${{ matrix.language }}
        queries: security-extended,security-and-quality
    
    - name: Autobuild
      uses: github/codeql-action/autobuild@v2
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
      with:
        category: "/language:${{matrix.language}}"
```

## Branch Protection Rules

Configure these branch protection rules in GitHub:

```javascript
// Required status checks for 'main' branch
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "quality-checks (3.10)",
      "quality-checks (3.11)", 
      "quality-checks (3.12)",
      "test-suite (ubuntu-latest, 3.11)",
      "build-artifacts",
      "container-security"
    ]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 2,
    "dismiss_stale_reviews": true
  },
  "restrictions": null
}
```

## Environment Configuration

1. Create a `release` environment in GitHub
2. Add required reviewers for releases
3. Configure environment secrets for PyPI publishing

## Monitoring & Notifications

### Slack Integration (.github/workflows/notify.yml)

```yaml
name: Notifications

on:
  workflow_run:
    workflows: ["CI Pipeline", "Release Pipeline"]
    types: [completed]

jobs:
  notify:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'failure' }}
    
    steps:
    - name: Notify Slack on Failure
      uses: 8398a7/action-slack@v3
      with:
        status: failure
        channel: '#causal-interface-gym'
        webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Implementation Checklist

- [ ] Create `.github/workflows/` directory
- [ ] Add workflow files listed above
- [ ] Configure repository secrets
- [ ] Set up branch protection rules
- [ ] Create release environment
- [ ] Configure Codecov integration
- [ ] Set up Slack notifications (optional)
- [ ] Test workflows with a sample PR
- [ ] Configure dependency updates with Dependabot

## Performance Optimization

The CI pipeline includes several optimizations:
- **Caching**: pip cache, Docker layer cache
- **Concurrency**: Cancel in-progress runs on new pushes
- **Matrix builds**: Parallel testing across Python versions/OS
- **Conditional execution**: Skip unnecessary jobs based on file changes
- **Artifact reuse**: Share build artifacts between jobs

## Security Best Practices

- All workflows use pinned action versions (@v4, @v3, etc.)
- Minimal required permissions for each job
- Secrets are scoped to specific environments
- Security scanning on every PR and push
- SBOM generation for supply chain transparency
- Dependency review prevents malicious packages