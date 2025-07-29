# Development Guide

This guide covers the development setup and workflow for Causal Interface Gym.

## Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/causal-interface-gym.git
cd causal-interface-gym
```

2. **Set up development environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
make install-dev
```

3. **Run tests**:
```bash
make test
```

4. **Format and lint code**:
```bash
make format
make lint
```

## Project Structure

```
causal-interface-gym/
├── src/causal_interface_gym/  # Main package
│   ├── core.py               # Core environment and UI classes
│   ├── metrics.py            # Causal reasoning metrics
│   └── __init__.py           # Package initialization
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Example usage
├── pyproject.toml           # Project configuration
├── Makefile                 # Development commands
└── README.md                # Main documentation
```

## Development Workflow

### 1. Feature Development

- Create feature branch: `git checkout -b feature/your-feature`
- Make changes following code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 2. Code Quality

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pytest**: Testing framework
- **Pre-commit**: Git hooks for quality checks

### 3. Testing

Run the full test suite:
```bash
make test
```

Run specific test files:
```bash
pytest tests/test_core.py
```

Run with coverage:
```bash
pytest --cov=causal_interface_gym --cov-report=html
```

### 4. Documentation

Build documentation:
```bash
make docs
```

The documentation uses Sphinx with MyST parser for Markdown support.

## Architecture Overview

### Core Components

1. **CausalEnvironment**: Manages causal graphs and interventions
2. **InterventionUI**: Builds user interfaces for causal reasoning
3. **CausalMetrics**: Evaluates causal reasoning performance
4. **BeliefTracker**: Tracks belief evolution over time

### Extension Points

- **Custom Environments**: Extend `CausalEnvironment` for specific domains
- **Custom Metrics**: Implement new evaluation metrics
- **UI Components**: Add new interface elements
- **LLM Integrations**: Support additional language models

## Contributing Guidelines

### Code Style

- Use Black for formatting (88 character line length)
- Follow PEP 8 conventions
- Add type hints for all public functions
- Write docstrings using Google style

### Testing Requirements

- Write unit tests for all new functionality
- Aim for >90% code coverage
- Include integration tests for major features
- Mock external dependencies (LLM APIs)

### Documentation

- Update README.md for significant changes
- Add docstrings for all public APIs
- Include examples in docstrings
- Update CHANGELOG.md

## Debugging

### Common Issues

1. **Import errors**: Ensure you've installed the package with `pip install -e .`
2. **Test failures**: Check that all dependencies are installed with `make install-dev`
3. **Type errors**: Run `mypy .` to check type annotations

### Debugging Tips

- Use the debugger: `python -m pdb your_script.py`
- Add print statements for quick debugging
- Use pytest's `-v` flag for verbose output
- Check the logs in `pytest.log` for detailed error information

## Release Process

1. Update version in `src/causal_interface_gym/__init__.py`
2. Update `CHANGELOG.md` with new features and fixes
3. Create release branch: `git checkout -b release/v0.x.x`
4. Run full test suite and quality checks
5. Create pull request for review
6. Tag release after merge: `git tag v0.x.x`
7. Push tags: `git push --tags`

## Performance Considerations

- Use numpy for numerical computations
- Cache expensive calculations
- Profile code with `cProfile` for bottlenecks
- Consider async patterns for UI responsiveness

## Security

- Never commit API keys or secrets
- Use environment variables for configuration
- Validate all user inputs
- Follow secure coding practices for web interfaces