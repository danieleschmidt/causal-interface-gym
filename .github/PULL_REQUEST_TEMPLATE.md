# Pull Request

## 📝 Description

<!-- Provide a clear and concise description of your changes -->

## 🔗 Related Issues

<!-- Link to any related issues using "Fixes #123" or "Closes #123" -->

## 🧪 Type of Change

<!-- Check all that apply -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 Documentation update
- [ ] 🔧 Refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] 🧹 Code cleanup
- [ ] 🔒 Security improvement

## 🧮 Changes Made

<!-- Describe the changes in detail -->

### Core Changes
- [ ] Modified core causal environment functionality
- [ ] Updated intervention interface components  
- [ ] Enhanced metrics and evaluation methods
- [ ] Improved LLM integration capabilities

### Infrastructure Changes
- [ ] Updated build/deployment configuration
- [ ] Modified CI/CD workflows
- [ ] Enhanced testing infrastructure
- [ ] Updated documentation

## 🧪 Testing

<!-- Describe the tests you've added or run -->

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] End-to-end tests added/updated
- [ ] Performance benchmarks run
- [ ] Manual testing completed

### Test Results
```bash
# Paste test output here
make test
```

## 📊 Performance Impact

<!-- If applicable, describe performance implications -->

- [ ] No performance impact
- [ ] Performance improvement (describe below)
- [ ] Performance regression (describe mitigation)

**Performance Details:**
<!-- Benchmark results, memory usage, etc. -->

## 🔒 Security Considerations

<!-- Address any security implications -->

- [ ] No security implications
- [ ] Security improvement (describe below)
- [ ] Potential security concerns (describe mitigation)

## 📖 Documentation

<!-- Check all that apply -->

- [ ] Code is self-documenting with appropriate docstrings
- [ ] README updated if needed
- [ ] API documentation updated
- [ ] Examples updated/added
- [ ] Migration guide provided (for breaking changes)

## ✅ Checklist

<!-- Verify these items before submitting -->

### Code Quality
- [ ] Code follows project style guidelines (black, ruff)
- [ ] Type hints are included where appropriate
- [ ] No unused imports or variables
- [ ] Complex logic is commented/documented

### Testing & Validation
- [ ] All tests pass locally
- [ ] New tests cover added functionality
- [ ] Existing tests still pass
- [ ] Code coverage maintained or improved

### Dependencies
- [ ] No new dependencies added, or they are justified
- [ ] Dependency versions are pinned appropriately
- [ ] Security vulnerabilities checked

### Research Validity
- [ ] Causal reasoning logic is theoretically sound
- [ ] Experimental methodology is rigorous
- [ ] Results are reproducible
- [ ] Evaluation metrics are appropriate

## 🤔 Questions for Reviewers

<!-- Any specific areas you'd like reviewers to focus on -->

## 📸 Screenshots/Examples

<!-- If applicable, add screenshots or example outputs -->

```python
# Example usage of new features
from causal_interface_gym import YourNewFeature

example = YourNewFeature()
result = example.demonstrate()
print(result)
```

---

**Reviewer Guidelines:**
- Focus on causal reasoning correctness
- Verify research methodology soundness  
- Check for potential bias in evaluation
- Ensure reproducibility of results
- Review code quality and maintainability