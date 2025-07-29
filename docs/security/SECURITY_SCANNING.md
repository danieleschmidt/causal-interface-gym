# Security Scanning Guide

This document outlines the security scanning tools and procedures for the Causal Interface Gym project.

## Overview

Security scanning is integrated into the development workflow through:
- **Static Analysis**: Code scanning for security vulnerabilities
- **Dependency Scanning**: Vulnerability checking in dependencies
- **Container Security**: Docker image vulnerability scanning
- **SBOM Generation**: Software Bill of Materials creation

## Tools Configuration

### 1. Bandit (Static Analysis)

Bandit scans Python code for common security issues.

**Configuration**: `.bandit`

**Usage**:
```bash
# Run security scan
bandit -r src/ -f json -o security-report.json

# Run with configuration file
bandit -c .bandit -r src/

# Generate HTML report
bandit -r src/ -f html -o reports/security.html
```

**Common Issues Detected**:
- Hardcoded passwords
- SQL injection vulnerabilities
- Shell injection risks
- Insecure random number generation
- Use of insecure cryptography

### 2. Safety (Dependency Scanning)

Safety checks Python dependencies for known security vulnerabilities.

**Configuration**: `.safety-policy.yml`

**Usage**:
```bash
# Check requirements file
safety check -r requirements-dev.txt

# Check with policy file
safety check --policy-file .safety-policy.yml

# Generate JSON report
safety check --json --output safety-report.json

# Check only production dependencies
safety check -r requirements.txt --ignore-unpinned
```

**Integration with CI/CD**:
```bash
# Fail build on high-severity vulnerabilities
safety check --policy-file .safety-policy.yml --exit-code
```

### 3. Semgrep (Advanced Static Analysis)

Semgrep provides more sophisticated security pattern matching.

**Setup**:
```bash
pip install semgrep
```

**Usage**:
```bash
# Run with security rules
semgrep scan --config=auto src/

# Run specific rulesets
semgrep scan --config=p/security-audit src/

# Generate SARIF output for GitHub
semgrep scan --sarif --output=semgrep-results.sarif src/
```

### 4. Trivy (Container Security)

Trivy scans Docker images for vulnerabilities.

**Usage**:
```bash
# Scan Docker image
trivy image causal-interface-gym:latest

# Generate JSON report
trivy image --format json --output image-scan.json causal-interface-gym:latest

# Scan filesystem
trivy fs --security-checks vuln,config .
```

### 5. SBOM Generation

Generate Software Bill of Materials for compliance.

**Using CycloneDX**:
```bash
pip install cyclonedx-bom

# Generate SBOM from requirements
cyclonedx-py -r requirements-dev.txt -o sbom.json

# Generate SBOM from environment
cyclonedx-py -e -o sbom-env.json
```

**Using SPDX**:
```bash
pip install spdx-tools

# Generate SPDX SBOM
spdx-tools convert --from cyclonedx --to spdx sbom.json sbom.spdx
```

## Security Workflows

### Pre-commit Security Checks

Add to `.pre-commit-config.yaml`:

```yaml
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.5'
    hooks:
      - id: bandit
        args: ['-c', '.bandit']
        
  - repo: https://github.com/gitguardian/ggshield
    rev: v1.25.0
    hooks:
      - id: ggshield
        language: python
        stages: [commit]
```

### CI/CD Integration

**GitHub Actions Security Job**:
```yaml
security:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit
      run: |
        pip install bandit[toml]
        bandit -c .bandit -r src/ -f json -o bandit-report.json
    
    - name: Run Safety
      run: |
        pip install safety
        safety check --policy-file .safety-policy.yml --json --output safety-report.json
    
    - name: Run Semgrep
      uses: semgrep/semgrep-action@v1
      with:
        config: >-
          p/security-audit
          p/python
    
    - name: Upload SARIF
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: semgrep.sarif
```

### Vulnerability Response Workflow

1. **Detection**: Automated scanning identifies vulnerability
2. **Assessment**: Evaluate impact and exploitability
3. **Prioritization**: Assign severity based on CVSS score
4. **Remediation**: Apply patches or implement workarounds
5. **Verification**: Confirm fix resolves vulnerability
6. **Documentation**: Update security documentation

## Security Monitoring

### Dependency Monitoring

**Automated Updates**:
```bash
# Use Dependabot or Renovate for automated PRs
# Configure in .github/dependabot.yml

pip install pip-tools
pip-compile --upgrade requirements.in
```

**Manual Review**:
```bash
# Check for outdated packages
pip list --outdated

# Security-focused update check
safety check --policy-file .safety-policy.yml --full-report
```

### Runtime Security

**Container Runtime Security**:
```bash
# Run with minimal privileges
docker run --user 1000:1000 --read-only causal-interface-gym

# Security scanning in runtime
trivy image --exit-code 1 causal-interface-gym:latest
```

## Compliance and Reporting

### Security Reports

Generate comprehensive security reports:

```bash
#!/bin/bash
# security-report.sh

echo "Generating Security Report..."

# Create reports directory
mkdir -p reports/security

# Run Bandit
bandit -c .bandit -r src/ -f html -o reports/security/bandit.html
bandit -c .bandit -r src/ -f json -o reports/security/bandit.json

# Run Safety
safety check --policy-file .safety-policy.yml --output reports/security/safety.txt
safety check --policy-file .safety-policy.yml --json --output reports/security/safety.json

# Run Semgrep
semgrep scan --config=p/security-audit --json --output=reports/security/semgrep.json src/

# Generate SBOM
cyclonedx-py -r requirements-dev.txt -o reports/security/sbom.json

echo "Security reports generated in reports/security/"
```

### Compliance Artifacts

**For Audit Requirements**:
- Security scan results
- Vulnerability remediation records
- SBOM documents
- Security policy documentation
- Incident response logs

## Security Best Practices

### Code Security

1. **Input Validation**: Validate all inputs
2. **Output Encoding**: Encode outputs to prevent injection
3. **Authentication**: Use strong authentication mechanisms
4. **Authorization**: Implement proper access controls
5. **Cryptography**: Use established cryptographic libraries

### Infrastructure Security

1. **Container Security**: Use minimal base images
2. **Network Security**: Limit network exposure
3. **Secrets Management**: Never hardcode secrets
4. **Logging**: Log security-relevant events
5. **Monitoring**: Monitor for security anomalies

### Development Security

1. **Secure Defaults**: Use secure configurations by default
2. **Principle of Least Privilege**: Grant minimal necessary permissions
3. **Defense in Depth**: Implement multiple security layers
4. **Regular Updates**: Keep dependencies updated
5. **Security Testing**: Include security tests in test suites

## Incident Response

### Security Incident Procedure

1. **Immediate Response**:
   - Contain the incident
   - Assess the scope
   - Notify stakeholders

2. **Investigation**:
   - Collect evidence
   - Analyze attack vectors
   - Identify root cause

3. **Remediation**:
   - Apply fixes
   - Verify resolution
   - Update security measures

4. **Recovery**:
   - Restore services
   - Monitor for recurrence
   - Update documentation

5. **Lessons Learned**:
   - Conduct post-incident review
   - Update procedures
   - Share knowledge

## Contact Information

For security issues:
- **Email**: security@causal-gym.org
- **PGP Key**: Available at keybase.io/causal-gym
- **Response Time**: 24 hours for critical, 72 hours for others

## Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.org/dev/security/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CIS Controls](https://www.cisecurity.org/controls)