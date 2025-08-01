# Causal Interface Gym

*Last Updated: 2025-08-01*


[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Paper](https://img.shields.io/badge/Paper-CausaLM-red.svg)](https://arxiv.org/abs/2404.causallm)
[![Demo](https://img.shields.io/badge/Demo-Live-green.svg)](https://causal-interface-gym.demo)

Toolkit to embed do-calculus interventions directly into UI prototypes and measure how LLM agents update causal world-models. Building on Stanford's CausaLM findings (April 2025) that exposed critical gaps in LLM causal reasoning.

## 🎯 Overview

LLMs often fail at causal reasoning, confusing correlation with causation. This toolkit lets you:
- Build interactive UIs where users perform causal interventions
- Measure how LLM agents update their beliefs based on interventions vs observations
- Quantify causal reasoning capabilities across different models
- Design interfaces that teach causal thinking to both humans and AI

## ✨ Key Features

- **Do-Calculus Engine**: Implements Pearl's causal interventions in interactive UIs
- **LLM Agent Integration**: Test any LLM's causal reasoning in real-time
- **Visual Causal Graphs**: Interactive DAG editor with live intervention effects  
- **Belief Tracking**: Monitor how agents update P(Y|do(X)) vs P(Y|X)
- **A/B Testing Framework**: Compare causal vs correlational interface designs
- **Educational Mode**: Teach causal reasoning through interactive examples

## 🚀 Quick Start

```python
from causal_interface_gym import CausalEnvironment, InterventionUI
import openai

# Create a causal environment
env = CausalEnvironment.from_dag({
    "rain": [],
    "sprinkler": ["rain"],  
    "wet_grass": ["rain", "sprinkler"],
    "slippery": ["wet_grass"]
})

# Build an intervention interface
ui = InterventionUI(env)
ui.add_intervention_button("sprinkler", "Turn Sprinkler On/Off")
ui.add_observation_panel("wet_grass", "Grass Wetness")

# Test an LLM agent
agent = openai.ChatCompletion()
belief_trajectory = ui.run_experiment(
    agent=agent,
    interventions=[("sprinkler", True)],
    measure_beliefs=["P(slippery)", "P(rain|wet_grass)"]
)

# Analyze causal reasoning
analysis = env.analyze_causal_reasoning(belief_trajectory)
print(f"Causal score: {analysis.causal_score}")
print(f"Confusion matrix: {analysis.intervention_vs_observation}")
```

## 📊 Example: Simpson's Paradox Interface

<p align="center">
  <img src="docs/images/simpsons_paradox_demo.gif" width="600" alt="Simpson's Paradox Demo">
</p>

```python
# Create an interface that teaches Simpson's Paradox
from causal_interface_gym.examples import SimpsonsParadoxDemo

demo = SimpsonsParadoxDemo()
demo.create_interactive_plot(
    title="College Admissions: Correlation vs Causation",
    variables=["gender", "department", "admission_rate"],
    allow_interventions=["department_policy"]
)

# Measure if users/LLMs learn the correct causal structure
results = demo.run_study(participants=["gpt-4", "claude-3", "human_users"])
```

## 🏗️ Architecture

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

## 🧪 Pre-Built Environments

### 1. Classic Causal Scenarios

```python
from causal_interface_gym.environments import ClassicScenarios

# Smoking → Tar → Cancer (with genetic confounder)
smoking_env = ClassicScenarios.smoking_cancer()

# Ice Cream Sales ← Temperature → Crime
spurious_env = ClassicScenarios.ice_cream_crime()

# Drug → Recovery ← Severity (confounded treatment)
medical_env = ClassicScenarios.confounded_treatment()
```

### 2. Economic Decision Making

```python
from causal_interface_gym.environments import EconomicGym

# Build a supply/demand intervention interface
market_env = EconomicGym.supply_demand_system(
    goods=["wheat", "bread", "flour"],
    allow_price_interventions=True,
    include_external_shocks=True
)

# Test if LLMs understand price interventions vs observations
market_env.test_intervention_understanding(
    agent=your_llm,
    scenarios=["price_ceiling", "supply_shock", "demand_intervention"]
)
```

### 3. Debugging LLM Reasoning

```python
from causal_interface_gym.debugging import CausalDebugger

debugger = CausalDebugger()

# Analyze where LLMs fail
failure_modes = debugger.diagnose_model(
    model="gpt-4",
    test_suite="comprehensive",
    return_failure_types=True
)

# Common failures: 
# - Confusing P(Y|X) with P(Y|do(X))
# - Ignoring backdoor paths
# - Reversing causal direction
# - Treating all associations as causal
```

## 📈 Measurement & Metrics

### Causal Reasoning Score

```python
from causal_interface_gym.metrics import CausalMetrics

metrics = CausalMetrics()

# Measure intervention understanding
intervention_score = metrics.intervention_test(
    agent_responses=agent_beliefs,
    ground_truth=env.do_calculus_results()
)

# Measure backdoor adjustment understanding  
backdoor_score = metrics.backdoor_test(
    agent_graph=agent.inferred_dag,
    true_graph=env.causal_graph
)

# Measure counterfactual reasoning
counterfactual_score = metrics.counterfactual_test(
    agent_predictions=agent.what_if_predictions,
    true_counterfactuals=env.compute_counterfactuals()
)
```

### Belief Evolution Tracking

```python
# Track how beliefs change with interventions
tracker = BeliefTracker(agent)

# Initial observational belief
tracker.record("P(rain|wet_grass)", condition="observational")

# Post-intervention belief
env.intervene(sprinkler=True)
tracker.record("P(rain|wet_grass)", condition="do(sprinkler=on)")

# Visualize belief trajectory
tracker.plot_belief_evolution(
    variable="rain",
    conditions=["obs", "do(sprinkler)"],
    expected_change="decrease"  # Rain less likely if grass wet due to sprinkler
)
```

## 🎨 UI Components

### React Components

```jsx
import { CausalGraph, InterventionPanel, BeliefMeter } from 'causal-interface-gym'

function CausalReasoningInterface() {
  return (
    <div>
      <CausalGraph 
        nodes={nodes}
        edges={edges}
        onIntervene={(node, value) => handleIntervention(node, value)}
        highlightBackdoors={true}
      />
      
      <InterventionPanel
        variables={variables}
        interventionTypes={['force', 'prevent', 'randomize']}
        onApply={applyIntervention}
      />
      
      <BeliefMeter
        beliefs={agentBeliefs}
        groundTruth={doCalculusResults}
        showDivergence={true}
      />
    </div>
  )
}
```

### Python UI Builder

```python
from causal_interface_gym.ui import UIBuilder

builder = UIBuilder()

# Add causal graph visualization
builder.add_graph(
    env.causal_graph,
    layout="hierarchical",
    intervention_mode="interactive"
)

# Add intervention controls
builder.add_intervention_panel(
    variables=env.variables,
    intervention_types=["set_value", "randomize", "disconnect"]
)

# Add belief displays
builder.add_belief_display(
    tracked_beliefs=["P(Y|X)" for X, Y in env.variable_pairs],
    comparison_mode="intervention_vs_observation"
)

# Generate standalone HTML or embed in notebook
html = builder.render()
```

## 🔬 Research Applications

### 1. LLM Benchmarking

```python
from causal_interface_gym.benchmarks import CausalBenchmark

benchmark = CausalBenchmark()

# Run comprehensive causal reasoning tests
results = benchmark.evaluate_models(
    models=["gpt-4", "claude-3", "llama-3", "gemini-1.5"],
    test_categories=[
        "intervention_vs_observation",
        "backdoor_identification", 
        "frontdoor_adjustment",
        "instrumental_variables",
        "counterfactual_reasoning"
    ]
)

# Generate paper-ready figures
benchmark.plot_results(output_dir="figures/")
```

### 2. Interface Design Studies

```python
# A/B test causal vs traditional interfaces
from causal_interface_gym.studies import InterfaceStudy

study = InterfaceStudy()

# Version A: Traditional correlation-based UI
ui_traditional = study.create_baseline_ui(
    show_correlations=True,
    show_causal_graph=False
)

# Version B: Causal intervention UI  
ui_causal = study.create_causal_ui(
    show_interventions=True,
    show_do_calculus=True
)

# Run study with LLMs and humans
study_results = study.run_ab_test(
    participants=["gpt-4", "human_mturk_workers"],
    metrics=["decision_quality", "causal_understanding", "confidence"]
)
```

### 3. Curriculum Learning

```python
from causal_interface_gym.curriculum import CausalCurriculum

# Create a curriculum that teaches causal reasoning
curriculum = CausalCurriculum()

# Start with simple chains
curriculum.add_lesson(
    name="direct_causation",
    env=SimpleCausalChain("A → B → C"),
    concepts=["direct_effect", "indirect_effect"]
)

# Progress to confounding
curriculum.add_lesson(
    name="confounding", 
    env=ConfoundedRelationship("X ← Z → Y, X → Y"),
    concepts=["backdoor_path", "spurious_correlation"]
)

# Advance to complex scenarios
curriculum.add_lesson(
    name="mediation_moderation",
    env=ComplexDAG.from_real_scenario("job_market"),
    concepts=["mediation", "effect_modification"]
)

# Train an agent through curriculum
trained_agent = curriculum.train(
    base_agent=your_llm,
    progression_criterion="mastery_based"
)
```

## 📦 Installation

```bash
# Basic installation
pip install causal-interface-gym

# With UI components
pip install causal-interface-gym[ui]

# With all examples and benchmarks
pip install causal-interface-gym[full]

# Development installation
git clone https://github.com/yourusername/causal-interface-gym.git
cd causal-interface-gym
pip install -e ".[dev]"
```

## 🧩 Extending the Framework

### Custom Environments

```python
from causal_interface_gym import CausalEnvironment

class EpidemicEnvironment(CausalEnvironment):
    def __init__(self):
        super().__init__()
        
        # Define causal structure
        self.add_variable("vaccination_rate", type="continuous")
        self.add_variable("mask_mandate", type="binary")
        self.add_variable("infections", type="count")
        self.add_variable("hospitalizations", type="count")
        
        # Define causal relationships
        self.add_edge("vaccination_rate", "infections", 
                     mechanism=lambda v: np.exp(-2 * v))
        self.add_edge("mask_mandate", "infections",
                     mechanism=lambda m: 0.7 if m else 1.0)
        self.add_edge("infections", "hospitalizations",
                     mechanism=lambda i: 0.1 * i)
    
    def intervene(self, **interventions):
        # Custom intervention logic
        return super().intervene(**interventions)
```

### Custom Metrics

```python
from causal_interface_gym.metrics import BaseMetric

class CausalAttentionMetric(BaseMetric):
    """Measures if LLMs attend to causal vs spurious features"""
    
    def compute(self, agent_trace, environment):
        # Analyze attention patterns during causal reasoning
        attention_weights = agent_trace.get_attention_weights()
        causal_features = environment.get_causal_features()
        spurious_features = environment.get_spurious_features()
        
        causal_attention = attention_weights[causal_features].mean()
        spurious_attention = attention_weights[spurious_features].mean()
        
        return {
            "causal_attention_ratio": causal_attention / (causal_attention + spurious_attention),
            "feature_importance_alignment": self.compute_alignment(attention_weights, environment)
        }
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- New causal environments and scenarios
- UI components for causal reasoning
- Integration with more LLM providers
- Metrics for causal understanding
- Real-world application examples

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

```bibtex
@software{causal_interface_gym,
  title={Causal Interface Gym: Interactive Environments for LLM Causal Reasoning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/causal-interface-gym}
}

@article{causallm2025,
  title={CausaLM: Exposing Causal Reasoning Gaps in Large Language Models},
  author={Stanford AI Lab},
  journal={arXiv preprint arXiv:2404.xxxxx},
  year={2025}
}
```

## 📚 Resources

- [Documentation](https://causal-interface-gym.readthedocs.io)
- [Paper](https://arxiv.org/abs/2407.causal-gym)
- [Blog Post: Why LLMs Fail at Causation](https://blog.causal-gym.org/llm-failures)
- [Video Tutorials](https://youtube.com/causal-interface-gym)
- [Community Discord](https://discord.gg/causal-reasoning)

## 📝 License

BSD 3-Clause License

## 🙏 Acknowledgments

- Stanford AI Lab for the CausaLM paper and inspiration
- Judea Pearl for causal inference foundations
- The PyWhy community for causal discovery tools

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/causal-interface-gym/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/causal-interface-gym/discussions)
- **Email**: causal-gym@yourdomain.com
