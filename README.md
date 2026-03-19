# causal-interface-gym

A gym for testing LLM agents on **causal reasoning** tasks — do-calculus interventions, backdoor/frontdoor adjustment, counterfactuals, and intervention prediction.

## What this is

Language models are good at pattern matching. Causal reasoning is something different: it requires understanding *what would happen* under an intervention (`do(X=x)`), not just correlations in training data.

This gym provides:
- A **Structural Causal Model (SCM)** engine with a proper `do()` operator
- Four task types covering Pearl's Ladder of Causation (association, intervention, counterfactual)
- A **TaskEvaluator** that scores agents on causal accuracy
- Built-in agents for comparison: random baseline, naive correlation, oracle

## Quickstart

```bash
git clone https://github.com/danieleschmidt/causal-interface-gym
cd causal-interface-gym
pip install -e .
python demo.py
```

**No external dependencies** — pure Python 3.9+.

## Core concepts

### CausalGraph

A DAG with structural equations and the `do()` operator:

```python
from causal_gym import CausalGraph

g = CausalGraph(seed=42)
g.add_node("Z", noise_std=1.0)                        # confounder
g.add_node("X", equation=lambda p, n: 0.8*p["Z"] + n) # treatment
g.add_node("Y", equation=lambda p, n: 1.5*p["X"] + 0.5*p["Z"] + n)  # outcome
g.add_edge("Z", "X")
g.add_edge("Z", "Y")
g.add_edge("X", "Y")

# Observational sample (includes confounding)
obs = g.sample()

# Interventional sample: do(X=1) — cuts Z->X, fixes X=1
int_sample = g.sample(interventions={"X": 1.0})

# Average Causal Effect via Monte Carlo
ig = g.do({"X": 1.0})
e1 = ig.expected_value("Y", n_samples=10000)
e0 = g.do({"X": 0.0}).expected_value("Y", n_samples=10000)
print(f"ACE = {e1 - e0:.3f}")  # → 1.5 (true causal effect)
```

### CausalWorld

Gym-style environment with intervention budget:

```python
from causal_gym import CausalWorld

world = CausalWorld(graph, intervention_budget=5)
obs = world.reset()
world.intervene({"X": 1.0})       # do(X=1), costs 1 from budget
y_val = world.query("Y")
ground_truth = world.expected_outcome("Y", {"X": 1.0})
score = world.score(y_val, ground_truth, tolerance=0.1)
```

### Tasks

Four tasks covering key causal reasoning challenges:

| Task | Concept | Ground truth |
|------|---------|-------------|
| `BackdoorAdjustmentTask` | Identify and remove confounding | ACE via do() |
| `FrontdoorAdjustmentTask` | Frontdoor criterion with hidden confounders | ACE via mediator |
| `CounterfactualTask` | "What would have happened if..." | Twin-network abduction |
| `InterventionPredictionTask` | Learn causal effect from experimental queries | E[Y\|do(X=2)] |

```python
from causal_gym import BackdoorAdjustmentTask, TaskEvaluator

task = BackdoorAdjustmentTask(seed=42)

# What the agent sees
print(task.question())
# "Given the causal graph Z -> X -> Y and Z -> Y, what is the ACE of X on Y?"

ctx = task.context()
# ctx["variables"], ctx["edges"], ctx["observational_samples"], ...

# Evaluate an answer
result = task.evaluate(answer=1.5)
print(result.score)  # 1.0 if within tolerance of ground truth
```

### TaskEvaluator

Run any agent on a task suite:

```python
from causal_gym import TaskEvaluator, BackdoorAdjustmentTask, CounterfactualTask

tasks = [BackdoorAdjustmentTask(), CounterfactualTask()]
evaluator = TaskEvaluator(tasks)

def my_llm_agent(ctx):
    # ctx["question"] is the natural language question
    # ctx["observational_samples"] is 10 observational draws
    # ctx["edges"] is the graph structure
    response = call_your_llm(ctx["question"], ctx)
    return float(response)

report = evaluator.run(my_llm_agent)
print(report)
```

## Benchmarks

Example scores on the 5-task demo suite:

| Agent | Mean Score | Notes |
|-------|-----------|-------|
| Random | ~0.20 | Baseline |
| Naive correlation | ~0.45 | Uses observational data, ignores confounding |
| **Your LLM** | ? | Should exceed naive to show causal reasoning |
| Oracle | 1.00 | Upper bound |

A score above naive correlation (~0.45) indicates the model is doing more than memorizing statistical patterns.

## Architecture

```
causal_gym/
├── graph.py       # CausalGraph + InterventionalGraph (do() operator)
├── world.py       # CausalWorld (gym environment)
├── tasks.py       # CausalTask base + 4 concrete tasks
├── evaluator.py   # TaskEvaluator + built-in agents
└── __init__.py    # Public API
tests/
├── test_graph.py  # Graph + do() operator tests
├── test_world.py  # World interaction tests
└── test_tasks.py  # Task + evaluator tests
demo.py            # 5-task benchmark with 3 agents
```

## Running tests

```bash
python -m pytest tests/ -v
```

46 tests, no external dependencies.

## Background

This gym implements concepts from Judea Pearl's *The Book of Why* and *Causality*:

- **Backdoor criterion**: A set Z blocks all backdoor paths from X to Y if Z contains no descendants of X and blocks every path from X to Y that starts with an arrow into X.
- **Frontdoor criterion**: When an unobserved confounder exists but a mediator M is observable and fully mediates the effect of X on Y, the frontdoor formula recovers the causal effect.
- **Counterfactuals**: Using the abduction-action-prediction algorithm (twin networks), we can answer "what would Y have been under a different X, given what we observed?"
- **do-calculus**: The formal language for expressing and computing interventional distributions.

## License

MIT
