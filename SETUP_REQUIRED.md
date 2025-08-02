# Manual Setup Required

*This file documents required manual setup steps due to GitHub App permission limitations.*

## GitHub Workflows Setup

The Causal Interface Gym project includes comprehensive CI/CD workflows that must be manually created due to GitHub App permission restrictions.

### Required Actions

1. **Copy Workflow Files**
   ```bash
   mkdir -p .github/workflows
   cp docs/workflows/examples/*.yml .github/workflows/
   ```

2. **Configure Repository Secrets**
   - See `docs/workflows/SETUP_INSTRUCTIONS.md` for complete list
   - Add required API keys, deployment credentials, and service tokens

3. **Set Up Branch Protection**
   - Enable branch protection for `main` branch
   - Require status checks from CI workflows
   - Require pull request reviews

4. **Enable Security Features**
   - Enable Dependabot alerts and security updates
   - Enable code scanning and secret scanning
   - Configure SARIF upload for security tools

### Available Workflow Templates

- **`ci.yml`**: Comprehensive CI pipeline with testing, linting, and security scanning
- **`cd.yml`**: Deployment pipeline for staging and production environments  
- **`security-scan.yml`**: Automated security scanning with multiple tools
- **`dependency-update.yml`**: Automated dependency updates and security patches

### Documentation

Complete setup instructions are available in:
- `docs/workflows/SETUP_INSTRUCTIONS.md` - Detailed setup guide
- `docs/workflows/README.md` - Workflow overview and architecture
- `docs/workflows/examples/` - Template workflow files

### Benefits After Setup

Once properly configured, the workflows will provide:

- ✅ Automated testing across Python 3.10-3.12
- ✅ Comprehensive security scanning (SAST, dependency, container)
- ✅ Automated deployments with proper approval gates
- ✅ Dependency updates with automated testing
- ✅ Performance benchmarking and regression detection
- ✅ Comprehensive monitoring and alerting

### Next Steps

1. Review `docs/workflows/SETUP_INSTRUCTIONS.md`
2. Configure repository secrets and environments
3. Copy workflow templates to `.github/workflows/`
4. Test workflows with a small PR
5. Enable branch protection rules
6. Configure notification channels (Slack, Discord, email)

This setup will provide enterprise-grade CI/CD capabilities for the project.