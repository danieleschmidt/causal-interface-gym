#!/bin/bash

# Security scanning script for Causal Interface Gym
# Runs comprehensive security checks and generates reports

set -euo pipefail

# Configuration
REPORTS_DIR="reports/security"
SRC_DIR="src"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_tool() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed. Please install it first."
        return 1
    fi
    return 0
}

# Create reports directory
mkdir -p "$REPORTS_DIR"

log_info "Starting security scan at $(date)"

# Check if required tools are installed
log_info "Checking required tools..."
TOOLS_MISSING=0

if ! check_tool "bandit"; then
    log_warn "Installing bandit..."
    pip install bandit[toml] || { log_error "Failed to install bandit"; TOOLS_MISSING=1; }
fi

if ! check_tool "safety"; then
    log_warn "Installing safety..."
    pip install safety || { log_error "Failed to install safety"; TOOLS_MISSING=1; }
fi

if ! check_tool "semgrep"; then
    log_warn "Installing semgrep..."
    pip install semgrep || { log_error "Failed to install semgrep"; TOOLS_MISSING=1; }
fi

if ! check_tool "cyclonedx-py"; then
    log_warn "Installing cyclonedx-bom..."
    pip install cyclonedx-bom || { log_error "Failed to install cyclonedx-bom"; TOOLS_MISSING=1; }
fi

if [ $TOOLS_MISSING -eq 1 ]; then
    log_error "Some tools failed to install. Please install them manually."
    exit 1
fi

# 1. Bandit - Static security analysis
log_info "Running Bandit static security analysis..."
if [ -f ".bandit" ]; then
    bandit -c .bandit -r "$SRC_DIR" -f json -o "$REPORTS_DIR/bandit_${TIMESTAMP}.json" || log_warn "Bandit found security issues"
    bandit -c .bandit -r "$SRC_DIR" -f html -o "$REPORTS_DIR/bandit_${TIMESTAMP}.html" || log_warn "Bandit found security issues"
    bandit -c .bandit -r "$SRC_DIR" || log_warn "Bandit found security issues (console output)"
else
    log_warn "No .bandit config found, using defaults"
    bandit -r "$SRC_DIR" -f json -o "$REPORTS_DIR/bandit_${TIMESTAMP}.json" || log_warn "Bandit found security issues"
    bandit -r "$SRC_DIR" -f html -o "$REPORTS_DIR/bandit_${TIMESTAMP}.html" || log_warn "Bandit found security issues"
fi

# 2. Safety - Dependency vulnerability scanning
log_info "Running Safety dependency vulnerability scan..."
if [ -f ".safety-policy.yml" ]; then
    safety check --policy-file .safety-policy.yml --json --output "$REPORTS_DIR/safety_${TIMESTAMP}.json" || log_warn "Safety found vulnerabilities"
    safety check --policy-file .safety-policy.yml --output "$REPORTS_DIR/safety_${TIMESTAMP}.txt" || log_warn "Safety found vulnerabilities"
    safety check --policy-file .safety-policy.yml || log_warn "Safety found vulnerabilities (console output)"
else
    log_warn "No .safety-policy.yml found, using defaults"
    safety check --json --output "$REPORTS_DIR/safety_${TIMESTAMP}.json" || log_warn "Safety found vulnerabilities"
    safety check --output "$REPORTS_DIR/safety_${TIMESTAMP}.txt" || log_warn "Safety found vulnerabilities"
fi

# 3. Semgrep - Advanced static analysis
log_info "Running Semgrep advanced static analysis..."
semgrep scan --config=p/security-audit --json --output="$REPORTS_DIR/semgrep_${TIMESTAMP}.json" "$SRC_DIR" || log_warn "Semgrep found issues"
semgrep scan --config=p/python --json --output="$REPORTS_DIR/semgrep_python_${TIMESTAMP}.json" "$SRC_DIR" || log_warn "Semgrep found Python issues"

# Generate SARIF for GitHub integration
semgrep scan --config=p/security-audit --sarif --output="$REPORTS_DIR/semgrep_${TIMESTAMP}.sarif" "$SRC_DIR" || log_warn "Semgrep found issues (SARIF)"

# 4. Generate SBOM (Software Bill of Materials)
log_info "Generating Software Bill of Materials (SBOM)..."
if [ -f "requirements-dev.txt" ]; then
    cyclonedx-py -r requirements-dev.txt -o "$REPORTS_DIR/sbom_${TIMESTAMP}.json" || log_warn "SBOM generation failed"
fi

if [ -f "pyproject.toml" ]; then
    cyclonedx-py -p . -o "$REPORTS_DIR/sbom_project_${TIMESTAMP}.json" || log_warn "Project SBOM generation failed"
fi

# 5. Check for secrets (if available)
if command -v "gitleaks" &> /dev/null; then
    log_info "Running Gitleaks secret detection..."
    gitleaks detect --report-path "$REPORTS_DIR/gitleaks_${TIMESTAMP}.json" --report-format json || log_warn "Gitleaks found potential secrets"
else
    log_warn "Gitleaks not available. Consider installing for secret detection."
fi

# 6. License compliance check
if command -v "license-checker" &> /dev/null; then
    log_info "Running license compliance check..."
    license-checker --json --out "$REPORTS_DIR/licenses_${TIMESTAMP}.json" || log_warn "License check failed"
else
    log_warn "license-checker not available. Consider installing for license compliance."
fi

# 7. Container security (if Docker is available)
if command -v "docker" &> /dev/null && [ -f "Dockerfile" ]; then
    log_info "Building Docker image for security scanning..."
    docker build -t causal-interface-gym:security-scan . || log_warn "Docker build failed"
    
    if command -v "trivy" &> /dev/null; then
        log_info "Running Trivy container security scan..."
        trivy image --format json --output "$REPORTS_DIR/trivy_${TIMESTAMP}.json" causal-interface-gym:security-scan || log_warn "Trivy found vulnerabilities"
    else
        log_warn "Trivy not available. Consider installing for container security scanning."
    fi
else
    log_warn "Docker not available or no Dockerfile found. Skipping container security scan."
fi

# Generate summary report
log_info "Generating security summary report..."
cat > "$REPORTS_DIR/security_summary_${TIMESTAMP}.md" << EOF
# Security Scan Summary

**Scan Date**: $(date)
**Timestamp**: ${TIMESTAMP}

## Tools Used

- **Bandit**: Static security analysis for Python
- **Safety**: Dependency vulnerability scanning
- **Semgrep**: Advanced static analysis
- **CycloneDX**: SBOM generation

## Reports Generated

- \`bandit_${TIMESTAMP}.json\` - Bandit security issues (JSON)
- \`bandit_${TIMESTAMP}.html\` - Bandit security issues (HTML)
- \`safety_${TIMESTAMP}.json\` - Safety vulnerability report (JSON)
- \`safety_${TIMESTAMP}.txt\` - Safety vulnerability report (Text)
- \`semgrep_${TIMESTAMP}.json\` - Semgrep security findings
- \`semgrep_python_${TIMESTAMP}.json\` - Semgrep Python-specific findings
- \`semgrep_${TIMESTAMP}.sarif\` - Semgrep findings (SARIF format)
- \`sbom_${TIMESTAMP}.json\` - Software Bill of Materials

## Next Steps

1. Review each report for security issues
2. Prioritize fixes based on severity
3. Update dependencies with known vulnerabilities
4. Address static analysis findings
5. Integrate security scanning into CI/CD pipeline

## Resources

- [Security Documentation](../security/SECURITY_SCANNING.md)
- [Vulnerability Response Guide](../security/VULNERABILITY_RESPONSE.md)
- [Security Best Practices](../security/BEST_PRACTICES.md)
EOF

# Create latest symlinks
ln -sf "bandit_${TIMESTAMP}.json" "$REPORTS_DIR/bandit_latest.json"
ln -sf "safety_${TIMESTAMP}.json" "$REPORTS_DIR/safety_latest.json"
ln -sf "semgrep_${TIMESTAMP}.json" "$REPORTS_DIR/semgrep_latest.json"
ln -sf "sbom_${TIMESTAMP}.json" "$REPORTS_DIR/sbom_latest.json"

log_info "Security scan completed successfully!"
log_info "Reports saved to: $REPORTS_DIR"
log_info "Summary report: $REPORTS_DIR/security_summary_${TIMESTAMP}.md"

# Exit with appropriate code
if grep -q '"issue"' "$REPORTS_DIR/bandit_${TIMESTAMP}.json" 2>/dev/null || \
   grep -q '"vulnerabilities"' "$REPORTS_DIR/safety_${TIMESTAMP}.json" 2>/dev/null; then
    log_warn "Security issues found. Please review reports."
    exit 1
else
    log_info "No critical security issues detected."
    exit 0
fi