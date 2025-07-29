# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in Causal Interface Gym, please follow these steps:

### Private Disclosure

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email security reports to: **causal-gym-security@yourdomain.com**
3. Include a detailed description of the vulnerability
4. Provide steps to reproduce the issue
5. Include any potential impact assessment

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions
- Potential impact
- Any suggested fixes (optional)

### Response Timeline

- **Acknowledgment**: Within 24 hours
- **Initial Assessment**: Within 72 hours  
- **Status Updates**: Weekly until resolved
- **Fix Timeline**: Target 30 days for critical issues

### Disclosure Policy

- We will coordinate disclosure timing with you
- We prefer coordinated disclosure after a fix is available
- We will credit you in security advisories (unless you prefer anonymity)

### Security Best Practices

When using Causal Interface Gym:

- Keep dependencies updated
- Use secure communication channels for LLM API keys
- Validate all user inputs in custom environments
- Follow principle of least privilege for file system access
- Use environment variables for sensitive configuration

### Known Security Considerations

- LLM API keys should be stored securely (environment variables)
- Custom causal environments should validate inputs
- UI components should sanitize user-generated content
- File uploads in examples should be restricted by type and size

### Security Features

- Input validation for causal graph specifications
- Secure handling of experiment data
- No storage of LLM responses by default
- Configurable privacy settings for data collection

For general security questions or concerns, please reach out to the maintainers.