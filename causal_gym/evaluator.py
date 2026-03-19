"""
TaskEvaluator: runs agents (or mock agents) on causal tasks and scores
their causal reasoning accuracy.

Usage::

    evaluator = TaskEvaluator(tasks=[...])
    results = evaluator.run(agent)
    evaluator.report(results)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .tasks import CausalTask, TaskResult


# An agent is any callable: context -> answer
AgentFn = Callable[[Dict[str, Any]], Any]


@dataclass
class EvaluationReport:
    """Aggregate results across all tasks."""
    results: List[TaskResult]
    total_score: float
    mean_score: float
    task_scores: Dict[str, float]
    elapsed_seconds: float

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "  Causal Gym Evaluation Report",
            "=" * 60,
            f"  Tasks evaluated : {len(self.results)}",
            f"  Mean score      : {self.mean_score:.3f}",
            f"  Total score     : {self.total_score:.3f} / {len(self.results):.1f}",
            f"  Elapsed         : {self.elapsed_seconds:.2f}s",
            "",
            "  Per-task breakdown:",
        ]
        for r in self.results:
            bar = "█" * int(r.score * 10) + "░" * (10 - int(r.score * 10))
            lines.append(
                f"  [{bar}] {r.score:.3f}  {r.task_name}"
            )
            lines.append(f"           Q: {r.question[:70]}...")
            lines.append(f"           Agent: {r.agent_answer!r}  |  GT: {r.ground_truth!r}")
            if r.explanation:
                lines.append(f"           Note: {r.explanation}")
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class TaskEvaluator:
    """
    Runs an agent on a list of causal tasks and reports scores.

    Args:
        tasks: List of CausalTask instances to evaluate.
        verbose: Print progress during evaluation.
    """

    def __init__(self, tasks: List[CausalTask], verbose: bool = True):
        self.tasks = tasks
        self.verbose = verbose

    def run(self, agent: AgentFn) -> EvaluationReport:
        """
        Evaluate an agent on all tasks.

        Args:
            agent: Callable that takes task context (dict) and returns an answer.
                   The context contains: variables, edges, observational_samples,
                   intervention_budget, question.

        Returns:
            EvaluationReport with per-task and aggregate scores.
        """
        results: List[TaskResult] = []
        start = time.time()

        for task in self.tasks:
            if self.verbose:
                print(f"  → Running: {task.name}")

            # Build context for agent
            ctx = task.context()
            ctx["question"] = task.question()
            ctx["task_name"] = task.name
            ctx["description"] = task.description

            # Get agent's answer
            try:
                answer = agent(ctx)
            except Exception as e:
                answer = None
                if self.verbose:
                    print(f"    ⚠ Agent raised exception: {e}")

            # Evaluate
            result = task.evaluate(answer)

            if self.verbose:
                print(
                    f"    Score: {result.score:.3f}  "
                    f"(agent={result.agent_answer!r}, gt={result.ground_truth!r})"
                )

            results.append(result)

        elapsed = time.time() - start
        total = sum(r.score for r in results)
        mean = total / len(results) if results else 0.0
        task_scores = {r.task_name: r.score for r in results}

        return EvaluationReport(
            results=results,
            total_score=total,
            mean_score=mean,
            task_scores=task_scores,
            elapsed_seconds=elapsed,
        )

    def run_with_interventions(
        self,
        agent: "InteractiveAgentFn",
        task: CausalTask,
    ) -> TaskResult:
        """
        Run an interactive agent that can perform interventions before answering.

        The interactive agent receives a world handle and can call:
          - world.observe()
          - world.intervene({...})
          - world.query(variable)

        Args:
            agent: Callable(task_context, world) -> answer
            task: Single task to evaluate.

        Returns:
            TaskResult for this task.
        """
        ctx = task.context()
        ctx["question"] = task.question()
        task.world.reset()

        try:
            answer = agent(ctx, task.world)
        except Exception as e:
            answer = None
            if self.verbose:
                print(f"  ⚠ Interactive agent error: {e}")

        return task.evaluate(answer)


# ---------------------------------------------------------------------------
# Built-in mock agents for testing and demonstration
# ---------------------------------------------------------------------------


def random_agent(ctx: Dict[str, Any]) -> float:
    """Agent that answers randomly — baseline."""
    import random
    return random.uniform(-2, 5)


def oracle_agent(task: CausalTask) -> AgentFn:
    """
    Agent that always returns the ground truth — upper bound.
    Created per-task since it needs access to the task's ground_truth().
    """
    def _agent(ctx: Dict[str, Any]) -> Any:
        return task.ground_truth()
    return _agent


def naive_correlation_agent(ctx: Dict[str, Any]) -> float:
    """
    Agent that estimates causal effect using naive correlation
    from observational samples — ignores confounding.

    Computes (mean Y | high X) - (mean Y | low X) from samples.
    """
    samples = ctx.get("observational_samples", [])
    if not samples:
        return 0.0

    # Find X and Y variables (first two after any confounders)
    vars_ = ctx.get("variables", [])
    edges = ctx.get("edges", [])

    # Identify treatment: node with outgoing edges but no incoming (or first node)
    out_deg = {v: 0 for v in vars_}
    in_deg = {v: 0 for v in vars_}
    for src, dst in edges:
        out_deg[src] += 1
        in_deg[dst] += 1

    treatments = [v for v in vars_ if in_deg[v] == 0 and out_deg[v] > 0]
    outcomes = [v for v in vars_ if out_deg[v] == 0 and in_deg[v] > 0]

    if not treatments or not outcomes:
        return 0.0

    treatment, outcome = treatments[0], outcomes[0]
    median_x = sorted(s[treatment] for s in samples)[len(samples) // 2]

    high = [s[outcome] for s in samples if s[treatment] > median_x]
    low = [s[outcome] for s in samples if s[treatment] <= median_x]

    if not high or not low:
        return 0.0

    diff = sum(high) / len(high) - sum(low) / len(low)
    # Normalize to unit treatment difference
    x_vals = [s[treatment] for s in samples]
    x_range = max(x_vals) - min(x_vals) or 1.0
    return diff / x_range * 2  # rough scaling to unit effect
