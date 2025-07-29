# GitHub Actions Workflows

This directory contains templates and documentation for GitHub Actions workflows. Note that actual workflow files should be created in `.github/workflows/` directory by repository maintainers.

## Recommended Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and quality checks on every push and pull request.

**Triggers**:
- Push to main branch
- Pull requests
- Manual dispatch

**Jobs**:
- **Test Matrix**: Python 3.10, 3.11, 3.12 on Ubuntu, macOS, Windows
- **Linting**: Black, Ruff, MyPy
- **Security**: Bandit, Safety checks
- **Coverage**: Upload to Codecov

**Template**:
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
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
        pip install -e ".[dev]"
    
    - name: Lint with ruff
      run: ruff check .
    
    - name: Type check with mypy
      run: mypy .
    
    - name: Test with pytest
      run: pytest --cov=causal_interface_gym --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### 2. Security Scanning (`security.yml`)

**Purpose**: Scan for vulnerabilities and security issues.

**Triggers**:
- Schedule (weekly)
- Push to main
- Manual dispatch

**Jobs**:
- **Dependency Scan**: Check for known vulnerabilities
- **Code Scan**: Static analysis for security issues
- **Secrets Scan**: Detect exposed credentials

### 3. Release Automation (`release.yml`)

**Purpose**: Automate package building and publishing.

**Triggers**:
- Tag creation (v*)
- Manual dispatch

**Jobs**:
- **Build**: Create distribution packages
- **Test**: Verify packages work correctly
- **Publish**: Upload to PyPI
- **GitHub Release**: Create release with notes

### 4. Documentation (`docs.yml`)

**Purpose**: Build and deploy documentation.

**Triggers**:
- Push to main (docs changes)
- Manual dispatch

**Jobs**:
- **Build Docs**: Generate Sphinx documentation
- **Deploy**: Publish to GitHub Pages
- **Link Check**: Verify external links

## Setup Instructions

### 1. Create Workflow Files

Copy the templates above to `.github/workflows/` directory:

```bash
mkdir -p .github/workflows
# Create ci.yml, security.yml, release.yml, docs.yml
```

### 2. Configure Secrets

Add the following secrets in GitHub repository settings:

- `PYPI_API_TOKEN`: For publishing to PyPI
- `CODECOV_TOKEN`: For coverage reporting
- `SECURITY_SCAN_TOKEN`: For security scanning (if needed)

### 3. Configure Branch Protection

Enable branch protection for `main`:
- Require status checks to pass
- Require up-to-date branches
- Include administrators
- Require linear history

### 4. Set Up Environments

Create deployment environments:
- **Development**: Auto-deploy on main branch
- **Staging**: Manual approval required
- **Production**: Manual approval + reviews

## Workflow Best Practices

### 1. Performance Optimization

- Cache dependencies between runs
- Use matrix builds efficiently
- Parallelize independent jobs
- Fail fast on critical errors

### 2. Security

- Use official actions when possible
- Pin action versions to specific commits
- Limit permissions (GITHUB_TOKEN)
- Never log sensitive information

### 3. Maintainability

- Use descriptive job and step names
- Comment complex workflow logic
- Keep workflows focused and single-purpose
- Regular dependency updates

### 4. Monitoring

- Set up notifications for failures
- Monitor workflow run times
- Track success/failure rates
- Review security alerts promptly

## Integration with External Services

### 1. Code Coverage (Codecov)

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    flags: unittests
    name: codecov-umbrella
```

### 2. Security Scanning (CodeQL)

```yaml
- name: Initialize CodeQL
  uses: github/codeql-action/init@v2
  with:
    languages: python

- name: Perform CodeQL Analysis
  uses: github/codeql-action/analyze@v2
```

### 3. Dependency Updates (Dependabot)

Configure in `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

## Troubleshooting

### Common Issues

1. **Test failures in CI but not locally**
   - Check Python version differences
   - Verify dependency versions
   - Review environment variables

2. **Workflow permissions errors**
   - Update GITHUB_TOKEN permissions
   - Check repository settings
   - Verify action versions

3. **Cache issues**
   - Clear workflow caches
   - Update cache keys
   - Check cache size limits

### Debug Strategies

- Enable debug logging: `ACTIONS_STEP_DEBUG: true`
- Use `tmate` action for SSH debugging
- Add diagnostic steps to workflows
- Review workflow run logs carefully