# GitHub Actions Setup Instructions

*Last Updated: 2025-08-02*

## Overview

This document provides step-by-step instructions for setting up GitHub Actions workflows for the Causal Interface Gym project. Due to GitHub App permission limitations, these workflows must be manually created by repository maintainers.

## Required Repository Secrets

Before setting up workflows, configure the following secrets in your GitHub repository (`Settings > Secrets and variables > Actions`):

### Required Secrets

```bash
# Database and Infrastructure
DATABASE_URL=postgresql://user:pass@host:5432/causal_experiments
REDIS_URL=redis://host:6379/0

# LLM Provider API Keys (configure relevant ones)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}

# Security and Authentication
JWT_SECRET_KEY=your_jwt_secret_key_here
SECRET_KEY=your_application_secret_key

# Deployment Credentials
AZURE_WEBAPP_PUBLISH_PROFILE_STAGING=<publish_profile_xml>
AZURE_WEBAPP_PUBLISH_PROFILE_PRODUCTION=<publish_profile_xml>

# Communication
SLACK_WEBHOOK=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
DISCORD_WEBHOOK=https://discord.com/api/webhooks/000000000000000000/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Email Notifications
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password

# External Services
HONEYCOMB_API_KEY=your_honeycomb_api_key
GITLEAKS_LICENSE=your_gitleaks_license_key

# PyPI Publishing (for automated releases)
PYPI_API_TOKEN=pypi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Workflow Setup Steps

### Step 1: Create GitHub Actions Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Templates

Copy the workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Main CI pipeline
cp docs/workflows/examples/ci.yml .github/workflows/ci.yml

# Deployment pipeline
cp docs/workflows/examples/cd.yml .github/workflows/cd.yml

# Security scanning
cp docs/workflows/examples/security-scan.yml .github/workflows/security-scan.yml

# Dependency updates
cp docs/workflows/examples/dependency-update.yml .github/workflows/dependency-update.yml
```

### Step 3: Configure Branch Protection Rules

Set up branch protection rules for `main` branch:

1. Go to `Settings > Branches`
2. Click "Add rule" for `main` branch
3. Configure the following:

```yaml
Branch protection rules:
  - Require a pull request before merging: ✅
    - Require approvals: 1
    - Dismiss stale PR approvals when new commits are pushed: ✅
    - Require review from code owners: ✅
  - Require status checks to pass before merging: ✅
    - Require branches to be up to date before merging: ✅
    - Status checks:
      - test (3.10, unit)
      - test (3.11, unit) 
      - test (3.12, unit)
      - test (3.10, integration)
      - test (3.11, integration)
      - test (3.12, integration)
      - frontend-test
      - security-scan
      - docker-build
  - Require linear history: ✅
  - Include administrators: ✅
```

### Step 4: Environment Configuration

Create deployment environments:

1. Go to `Settings > Environments`
2. Create the following environments:

#### Staging Environment
```yaml
Environment name: staging
Deployment branches: main
Environment protection rules:
  - Required reviewers: (optional)
  - Wait timer: 0 minutes
Environment secrets:
  - DATABASE_URL: (staging database URL)
  - REDIS_URL: (staging redis URL)
```

#### Production Environment
```yaml
Environment name: production
Deployment branches: main, tags matching v*
Environment protection rules:
  - Required reviewers: [List of reviewers]
  - Wait timer: 5 minutes
Environment secrets:
  - DATABASE_URL: (production database URL)
  - REDIS_URL: (production redis URL)
```

### Step 5: Configure Code Scanning

Enable GitHub Advanced Security features:

1. Go to `Settings > Security & analysis`
2. Enable:
   - Dependency graph: ✅
   - Dependabot alerts: ✅
   - Dependabot security updates: ✅
   - Code scanning: ✅
   - Secret scanning: ✅

### Step 6: Set up Notifications

#### Slack Integration
1. Create a Slack app for your workspace
2. Add incoming webhook capability
3. Configure webhook URL in repository secrets
4. Test notification:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"GitHub Actions test from Causal Interface Gym"}' \
     $SLACK_WEBHOOK
   ```

#### Discord Integration
1. Create a Discord webhook in your server
2. Configure webhook URL in repository secrets
3. Test notification:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"content":"GitHub Actions test from Causal Interface Gym"}' \
     $DISCORD_WEBHOOK
   ```

## Workflow Customization

### Adjusting Test Matrix

Modify the test matrix in `.github/workflows/ci.yml`:

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']  # Add/remove versions
    test-type: ['unit', 'integration']        # Add 'performance' if needed
    os: ['ubuntu-latest']                     # Add 'windows-latest', 'macos-latest'
```

### Customizing Deployment Targets

Update deployment configuration in `.github/workflows/cd.yml`:

```yaml
# For AWS deployment
- name: Deploy to AWS
  uses: aws-actions/configure-aws-credentials@v2
  with:
    aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
    aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    aws-region: us-west-2

# For Google Cloud deployment
- name: Deploy to GCP
  uses: google-github-actions/setup-gcloud@v1
  with:
    service_account_key: ${{ secrets.GCP_SA_KEY }}
    project_id: ${{ secrets.GCP_PROJECT_ID }}
```

### Adding Performance Benchmarks

Enable performance testing in CI:

```yaml
# Add to ci.yml
performance-benchmark:
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request'
  steps:
    - uses: actions/checkout@v4
    - name: Run benchmarks
      run: |
        pytest tests/benchmarks/ --benchmark-json=output.json
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: output.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '200%'
        max-items-in-chart: 20
```

## Monitoring and Alerting

### Workflow Status Monitoring

Set up monitoring for workflow failures:

```yaml
# Add to any workflow
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#ci-alerts'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
    fields: repo,commit,author,action,eventName,ref,workflow
```

### Performance Monitoring

Track workflow performance:

```yaml
# Add to workflows for monitoring
- name: Workflow timing
  run: |
    echo "Workflow started at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Job duration: ${{ github.event.inputs.duration || 'N/A' }}"
```

## Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Check branch protection rules
   - Verify workflow file syntax with [yaml-lint](https://www.yamllint.com/)
   - Ensure proper indentation (use spaces, not tabs)

2. **Secret not available**
   - Verify secret name matches exactly (case-sensitive)
   - Check if secret is available in the environment
   - For organization secrets, ensure repository has access

3. **Test failures in CI but not locally**
   - Check environment differences (Python version, dependencies)
   - Verify database/service availability
   - Review CI logs for specific error messages

4. **Deployment failures**
   - Verify deployment credentials are current
   - Check target environment health
   - Review deployment logs

### Debugging Commands

```bash
# Check workflow syntax
npx @github/workflow-validator .github/workflows/ci.yml

# Test workflow locally (using act)
act -s GITHUB_TOKEN=$GITHUB_TOKEN

# Check secret availability
echo "::add-mask::$SECRET_VALUE"
echo "Secret is available: $([ -n '$SECRET_VALUE' ] && echo 'Yes' || echo 'No')"
```

## Security Best Practices

### Secret Management
- Use environment-specific secrets for different deployment stages
- Rotate secrets regularly (quarterly recommended)
- Use GitHub's built-in secret scanner
- Avoid logging secret values

### Workflow Security
- Pin action versions to specific commits or tags
- Use least-privilege permissions for GITHUB_TOKEN
- Review third-party actions before use
- Enable security scanning for all workflows

### Access Control
- Limit who can approve deployments to production
- Use environment protection rules
- Enable audit logging
- Regular review of repository permissions

## Maintenance

### Regular Tasks

1. **Monthly**: Review and update action versions
2. **Quarterly**: Rotate secrets and review access permissions
3. **Bi-annually**: Audit workflow efficiency and update as needed

### Updating Workflows

When updating workflows:
1. Test changes in a feature branch first
2. Review workflow run history for patterns
3. Update documentation when making significant changes
4. Notify team of any changes to deployment processes

---

*For additional help with GitHub Actions setup, consult the [GitHub Actions documentation](https://docs.github.com/en/actions) or contact the development team.*