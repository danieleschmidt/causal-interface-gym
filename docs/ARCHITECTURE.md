# Architecture Overview

This document describes the high-level architecture of Causal Interface Gym.

## System Overview

Causal Interface Gym is designed as a modular framework for creating interactive causal reasoning experiments. The system enables researchers to build interfaces where LLM agents can perform causal interventions and have their reasoning evaluated.

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   UI Frontend   │────▶│ Intervention │────▶│ Causal Backend  │
│  (React/Svelte) │     │   Engine     │     │  (Do-Calculus)  │
└─────────────────┘     └──────────────┘     └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  LLM Agents     │     │ Belief       │     │ Analysis Suite  │
│                 │     │ Tracker      │     │                 │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Core Components

### 1. Causal Environment (`core.py`)

The `CausalEnvironment` class manages:
- Causal graph structure (DAG representation)
- Variable definitions and relationships
- Intervention operations (do-calculus)
- Observational and interventional queries

**Key responsibilities:**
- Maintain causal graph integrity
- Execute interventions correctly
- Provide ground truth for evaluation

### 2. Intervention UI (`core.py`)

The `InterventionUI` class provides:
- UI component builders
- Experiment orchestration
- Agent-interface coordination
- Data collection and logging

**Supported components:**
- Intervention buttons/controls
- Observation panels
- Causal graph visualizations
- Belief meters and displays

### 3. Metrics System (`metrics.py`)

The metrics system evaluates:
- Intervention vs observation understanding
- Backdoor path identification
- Counterfactual reasoning
- Belief update accuracy

**Core metrics:**
- `intervention_test()`: P(Y|do(X)) vs P(Y|X) distinction
- `backdoor_test()`: Confounding variable identification
- `counterfactual_test()`: What-if scenario reasoning

### 4. Belief Tracking (`metrics.py`)

The `BeliefTracker` monitors:
- Belief evolution over time
- Intervention-triggered updates
- Confidence measures
- Response patterns

## Data Flow

### 1. Experiment Setup
```python
# 1. Define causal environment
env = CausalEnvironment.from_dag(dag_structure)

# 2. Build intervention interface
ui = InterventionUI(env)
ui.add_intervention_button("variable", "label")

# 3. Configure metrics
metrics = CausalMetrics()
tracker = BeliefTracker(agent)
```

### 2. Experiment Execution
```python
# 1. Present scenario to agent
agent.observe(env.initial_state)

# 2. Apply interventions
result = env.intervene(variable=value)

# 3. Track belief updates
tracker.record("P(outcome)", "post_intervention")

# 4. Evaluate reasoning
score = metrics.intervention_test(agent_beliefs, ground_truth)
```

### 3. Analysis and Evaluation
```python
# 1. Compute causal reasoning scores
scores = metrics.comprehensive_evaluation(experiment_data)

# 2. Generate visualizations
tracker.plot_belief_evolution(variable, conditions)

# 3. Export results
results.export_to_paper_format()
```

## Design Principles

### 1. Modularity
- Components can be used independently
- Clear interfaces between modules
- Easy to extend and customize

### 2. Flexibility
- Support multiple LLM providers
- Configurable UI frameworks
- Pluggable metrics and environments

### 3. Reproducibility
- Deterministic experiment execution
- Comprehensive logging
- Version control for environments

### 4. Performance
- Efficient graph operations
- Lazy evaluation where possible
- Minimal memory footprint

## Extension Points

### 1. Custom Environments
```python
class EpidemicEnvironment(CausalEnvironment):
    def __init__(self):
        # Define domain-specific causal structure
        super().__init__(epidemic_dag)
    
    def intervene(self, **interventions):
        # Custom intervention logic
        return super().intervene(**interventions)
```

### 2. Custom Metrics
```python
class DomainSpecificMetric(BaseMetric):
    def compute(self, agent_trace, environment):
        # Domain-specific evaluation logic
        return score
```

### 3. UI Components
```python
class CustomVisualization(UIComponent):
    def render(self, data):
        # Custom visualization logic
        return html_output
```

## Security Considerations

### 1. Input Validation
- Validate all DAG specifications
- Sanitize user-provided content
- Limit computational complexity

### 2. API Security
- Secure LLM API key storage
- Rate limiting for experiments
- Data privacy controls

### 3. Code Safety
- No arbitrary code execution
- Sandboxed environments
- Input size limitations

## Performance Characteristics

### 1. Scalability
- Linear complexity in graph size
- Efficient belief tracking
- Batch processing support

### 2. Memory Usage
- Graph structures: O(V + E)
- Belief histories: O(T × V)
- UI components: O(C)

### 3. Computational Complexity
- Intervention queries: O(V²)
- Backdoor identification: O(V³)
- Metric computation: O(T × V)

## Future Architecture Considerations

### 1. Distributed Computing
- Multi-agent experiments
- Parallel evaluation
- Cloud deployment

### 2. Real-time Capabilities
- Live belief updates
- Streaming metrics
- Interactive visualizations

### 3. Integration Ecosystem
- Plugin architecture
- Third-party extensions
- API standardization