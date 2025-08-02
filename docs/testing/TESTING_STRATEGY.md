# Testing Strategy for Causal Interface Gym

*Last Updated: 2025-08-02*

## Overview

This document outlines the comprehensive testing strategy for the Causal Interface Gym project. Our testing approach ensures robust causal reasoning functionality, reliable LLM integrations, and high-quality user interfaces.

## Testing Philosophy

### Core Principles
1. **Correctness First**: Causal inference correctness is non-negotiable
2. **Fast Feedback**: Unit tests provide immediate feedback during development
3. **Real-World Validation**: Integration tests use realistic causal scenarios
4. **Performance Awareness**: Performance tests ensure scalability for research use
5. **Security by Design**: Security tests prevent malicious causal model injection

### Testing Pyramid

```
    /\
   /  \  E2E Tests (10%)
  /____\  
 /      \ Integration Tests (30%)
/__Unit__\  Unit Tests (60%)
```

## Test Categories

### 1. Unit Tests (`tests/test_*.py`)

**Purpose**: Test individual components in isolation
**Coverage Target**: >95%
**Runtime**: <5 seconds total

**Key Areas:**
- Causal graph construction and validation
- Do-calculus operations
- Intervention mechanisms
- Belief tracking algorithms
- UI component logic

```python
# Example unit test
def test_causal_environment_creation(sample_dag):
    env = CausalEnvironment.from_dag(sample_dag)
    assert env.graph.number_of_nodes() == 4
    assert env.is_valid_dag()
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and realistic workflows
**Coverage Target**: All major user workflows
**Runtime**: <30 seconds total

**Key Scenarios:**
- End-to-end causal reasoning experiments
- LLM agent interactions with causal environments
- UI component integration with backend
- Multi-agent experiment coordination

```python
# Example integration test
@pytest.mark.integration
def test_complete_causal_experiment(causal_environment, mock_agent):
    ui = InterventionUI(causal_environment)
    results = ui.run_experiment(
        agent=mock_agent,
        interventions=[("sprinkler", True)],
        measure_beliefs=["P(rain|wet_grass)"]
    )
    assert "causal_score" in results
    assert results["causal_score"] >= 0
```

### 3. Performance Tests (`tests/performance/`)

**Purpose**: Ensure scalability for research applications
**Benchmarks**: Response time, memory usage, throughput
**Runtime**: <120 seconds total

**Performance Targets:**
- Causal graph operations: <100ms for graphs with <1000 nodes
- Intervention computation: <10ms for typical scenarios
- UI updates: <50ms for real-time feedback
- LLM response processing: <500ms per query

### 4. Property-Based Tests

**Purpose**: Test causal reasoning properties across many inputs
**Tool**: Hypothesis library
**Focus**: Mathematical properties of causal operations

```python
from hypothesis import given, strategies as st

@given(st.dictionaries(st.text(), st.lists(st.text())))
def test_causal_graph_properties(dag_dict):
    if is_valid_dag_structure(dag_dict):
        env = CausalEnvironment.from_dag(dag_dict)
        # Test that interventions preserve graph properties
        assert env.graph.is_directed()
        assert not env.has_cycles()
```

### 5. Security Tests (`tests/security/`)

**Purpose**: Prevent malicious causal model injection and data leaks
**Tools**: Bandit, custom security checks
**Focus**: Input validation, data sanitization

## Test Data Management

### Test Fixtures (`tests/conftest.py`)

**Standard Fixtures:**
- `sample_dag`: Simple rain-sprinkler-grass DAG
- `complex_dag`: Multi-variable smoking-cancer scenario
- `mock_agent`: Simulated LLM agent with predictable responses
- `temp_directory`: Isolated filesystem for test artifacts

### Real-World Scenarios (`tests/fixtures/`)

**Curated Test Cases:**
- Classic causal inference scenarios (Simpson's Paradox, etc.)
- LLM reasoning patterns and failure modes
- UI interaction sequences
- Performance stress test scenarios

### Data Generation (`tests/generators.py`)

**Synthetic Data:**
- Random DAG generation with causal constraints
- Synthetic observational/interventional datasets
- Mock LLM response patterns
- UI interaction sequences

## LLM Testing Strategy

### Mock Testing (Development)
```python
class MockLLMAgent:
    def __init__(self, response_pattern="correct_causal"):
        self.responses = load_response_pattern(response_pattern)
    
    def query(self, prompt):
        return self.responses.get_next()
```

### Live Testing (CI/CD)
```python
@pytest.mark.llm
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No API key")
def test_openai_causal_reasoning():
    agent = OpenAIAgent(model="gpt-3.5-turbo")  # Use cheaper model for CI
    result = run_causal_benchmark(agent, scenario="simple_confounding")
    assert result.accuracy > 0.6  # Lower bar for automated testing
```

## UI Testing Strategy

### Component Testing
- **React Testing Library**: Test component behavior
- **Jest**: Unit test UI logic
- **Storybook**: Visual regression testing

### End-to-End Testing
- **Playwright**: Full user workflows
- **Visual Testing**: Screenshot comparisons
- **Accessibility**: WCAG compliance

```javascript
// Example UI test
test('intervention button updates causal graph', async ({ page }) => {
  await page.goto('/causal-experiment/rain-sprinkler');
  await page.click('[data-testid="sprinkler-intervention"]');
  await expect(page.locator('[data-testid="causal-graph"]')).toHaveAttribute('data-intervention', 'active');
});
```

## Continuous Integration Strategy

### Pre-commit Hooks
```yaml
- pytest tests/unit/ --maxfail=1  # Fast unit tests
- pytest tests/integration/ -x    # Stop on first integration failure
- pytest tests/ --cov=80         # Coverage threshold
```

### CI Pipeline (GitHub Actions)

**Pull Request Validation:**
1. **Fast Tests** (2 minutes): Unit tests + linting
2. **Integration Tests** (5 minutes): Component integration
3. **Security Scan** (3 minutes): Dependency vulnerabilities
4. **Performance Baseline** (10 minutes): Ensure no regressions

**Nightly Builds:**
1. **Full Test Suite**: All tests including slow ones
2. **LLM Integration Tests**: Live API testing with all providers
3. **Performance Benchmarking**: Detailed performance analysis
4. **Documentation Testing**: Ensure all examples work

### Test Environment Matrix

```yaml
strategy:
  matrix:
    python-version: [3.10, 3.11, 3.12]
    os: [ubuntu-latest, macos-latest, windows-latest]
    dependency-version: [minimal, latest]
```

## Quality Gates

### Code Coverage
- **Minimum**: 90% line coverage
- **Target**: 95% line coverage
- **Branch Coverage**: 85% minimum
- **Critical Paths**: 100% coverage for causal reasoning logic

### Performance Benchmarks
- **Regression Threshold**: 10% performance degradation fails CI
- **Memory Usage**: No memory leaks in long-running tests
- **Startup Time**: <2 seconds for typical causal environments

### Security Requirements
- **Dependency Scanning**: No high/critical vulnerabilities
- **Code Scanning**: Pass all Bandit security checks
- **Input Validation**: 100% coverage of external inputs

## Testing Best Practices

### Test Organization
```
tests/
├── unit/                    # Fast, isolated tests
│   ├── test_causal_core.py
│   ├── test_interventions.py
│   └── test_belief_tracking.py
├── integration/             # Component interaction tests
│   ├── test_end_to_end.py
│   └── test_llm_integration.py
├── performance/            # Scalability and speed tests
│   ├── test_large_graphs.py
│   └── test_concurrent_agents.py
├── fixtures/               # Test data and scenarios
│   ├── causal_scenarios/
│   └── llm_responses/
└── conftest.py            # Shared fixtures and configuration
```

### Test Naming Conventions
```python
# Good test names
def test_intervention_updates_belief_correctly()
def test_complex_dag_validates_acyclicity()
def test_llm_agent_handles_confounding_scenario()

# Bad test names
def test_function()
def test_dag()
def test_agent()
```

### Assertion Patterns
```python
# Causal reasoning assertions
def assert_causal_effect(result, expected_effect, tolerance=0.1):
    assert abs(result.causal_effect - expected_effect) < tolerance
    assert result.confidence > 0.8
    assert result.method in ["backdoor", "frontdoor", "instrumental"]

# LLM response assertions
def assert_belief_extraction(response, expected_beliefs):
    extracted = extract_beliefs(response)
    for var, belief in expected_beliefs.items():
        assert var in extracted
        assert 0 <= extracted[var] <= 1
```

## Test Data and Scenarios

### Standard Causal Scenarios
1. **Rain-Sprinkler-Grass**: Basic confounding
2. **Smoking-Cancer**: Classic causation with genetics confounder
3. **Simpson's Paradox**: Gender bias in admissions
4. **Instrumental Variables**: Treatment assignment randomization
5. **Mediation Analysis**: Direct vs indirect effects

### LLM Failure Modes
1. **Correlation-Causation Confusion**: Misinterpreting observational data
2. **Backdoor Path Ignorance**: Missing confounding variables
3. **Intervention Misunderstanding**: Treating do(X) same as observing X
4. **Counterfactual Errors**: Incorrect "what if" reasoning

### Performance Test Scenarios
1. **Large Graphs**: 1000+ node causal networks
2. **Many Agents**: 100+ concurrent LLM queries
3. **Real-time Updates**: Sub-second UI responsiveness
4. **Memory Stress**: Long-running experiment sessions

## Monitoring and Reporting

### Test Results Dashboard
- Real-time test status across all environments
- Performance trend analysis
- Coverage reports with drill-down capability
- Flaky test identification and management

### Quality Metrics
- **Test Stability**: Flaky test rate <5%
- **Execution Speed**: Test suite runtime trending
- **Coverage Evolution**: Coverage changes over time
- **Bug Escape Rate**: Issues found in production vs tests

## Future Enhancements

### Planned Improvements
1. **Mutation Testing**: Verify test quality with PIT testing
2. **Chaos Engineering**: Test robustness under failure conditions
3. **A/B Test Framework**: Test different causal reasoning approaches
4. **Automated Test Generation**: Generate tests from causal models

### Research Applications
1. **Benchmark Suite**: Standardized LLM causal reasoning evaluation
2. **Human Studies**: A/B test UI designs with human participants
3. **Educational Assessment**: Test causal learning effectiveness
4. **Real-world Validation**: Test on actual causal inference problems

---

*This testing strategy evolves with the project. Updates are tracked in the [CHANGELOG](../../CHANGELOG.md) and discussed in team retrospectives.*