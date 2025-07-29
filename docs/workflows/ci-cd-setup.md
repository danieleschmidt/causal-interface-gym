# CI/CD Workflow Setup Guide

This document provides templates and setup instructions for GitHub Actions workflows for the Causal Interface Gym project.

## Overview

The CI/CD setup for this Python research framework includes:
- **Continuous Integration**: Testing, linting, security scanning
- **Quality Gates**: Code coverage, performance benchmarks
- **Security**: Dependency scanning, SBOM generation
- **Release Management**: Automated versioning and PyPI publishing

## Required Workflows

### 1. Main CI/CD Pipeline

Create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        make install-dev
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests with coverage
      run: make test
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./htmlcov/index.html
        fail_ci_if_error: true

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Bandit Security Scan
      uses: securecodewarrior/github-action-bandit@v1
      with:
        config_file: '.bandit'
    
    - name: Run Safety Check
      run: |
        pip install safety
        safety check --json --output safety-report.json
    
    - name: Generate SBOM
      uses: anchore/sbom-action@v0
      with:
        path: .
        format: spdx-json
```

### 2. Security Scanning

Create `.github/workflows/security.yml`:

```yaml
name: Security Scanning

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM
  push:
    branches: [ main ]

jobs:
  codeql:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Initialize CodeQL
      uses: github/codeql-action/init@v3
      with:
        languages: python
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v3

  dependency-review:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
      with:
        fail-on-severity: high
```

### 3. Performance Benchmarks

Create `.github/workflows/benchmarks.yml`:

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        make install-dev
        pip install pytest-benchmark memory_profiler
    
    - name: Run benchmarks
      run: |
        pytest tests/benchmarks/ --benchmark-json=benchmark.json
    
    - name: Upload benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        name: Python Benchmark
        tool: 'pytest'
        output-file-path: benchmark.json
```

### 4. Release Automation

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false
```

## Configuration Files

### Security Configuration

Create `.bandit`:

```yaml
skips: ['B101', 'B601']
exclude_dirs: ['tests', 'docs']
```

Create `.safety-policy.yml`:

```yaml
security:
  ignore-cvss-severity-below: 7.0
  ignore-cvss-unknown-severity: false
```

### Performance Configuration

Create `pytest.ini` addition:

```ini
[tool:pytest]
addopts = --benchmark-skip
markers =
    benchmark: marks tests as benchmarks
    integration: marks tests as integration tests
    slow: marks tests as slow
```

## Required Secrets

Configure these secrets in GitHub repository settings:

- `PYPI_API_TOKEN`: For PyPI publishing
- `CODECOV_TOKEN`: For coverage reporting
- `SECURITY_SCAN_TOKEN`: For security scanning services

## Branch Protection Rules

Configure the following branch protection rules for `main`:

- Require pull request reviews before merging
- Require status checks to pass before merging:
  - `test (3.10)`
  - `test (3.11)` 
  - `test (3.12)`
  - `security`
  - `codeql`
- Require branches to be up to date before merging
- Restrict pushes that create files

## Monitoring and Alerts

### GitHub Insights
- Enable dependency insights
- Configure security advisories
- Set up automated security updates

### External Services
- **Codecov**: Coverage reporting and trends
- **Snyk**: Continuous security monitoring  
- **PyUp**: Automated dependency updates

## Rollback Procedures

### Failed Deployment
1. Identify failed workflow run
2. Review logs and error messages
3. Create hotfix branch if needed
4. Revert problematic changes
5. Re-run deployment pipeline

### Security Alert Response
1. Review security advisory details
2. Assess impact on codebase
3. Create security fix branch
4. Implement fixes and test
5. Deploy emergency release if critical

## Integration with Development Workflow

### Pre-commit Integration
- All commits automatically formatted and linted
- Security checks run locally before push
- Documentation consistency verified

### Pull Request Workflow
1. Create feature branch from `develop`
2. Make changes and commit with conventional commit messages
3. Push branch and create pull request
4. CI pipeline runs automatically
5. Code review and approval required
6. Merge to `develop` or `main`

### Release Process
1. Create release branch from `main`
2. Update version in `pyproject.toml`
3. Update `CHANGELOG.md`
4. Create and push version tag
5. GitHub Actions automatically publishes to PyPI
6. Create GitHub release with generated notes

This CI/CD setup provides comprehensive automation while maintaining security and quality standards appropriate for a research framework.