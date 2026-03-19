#!/usr/bin/env python3
"""
causal-interface-gym demo

Runs 5 causal reasoning tasks under three agents:
  1. Random agent    — baseline (no reasoning)
  2. Naive correlation agent — uses observational data, ignores confounding
  3. Oracle agent    — always correct (upper bound)

Shows how the gym scores causal reasoning accuracy.
"""

from causal_gym import (
    BackdoorAdjustmentTask,
    FrontdoorAdjustmentTask,
    CounterfactualTask,
    InterventionPredictionTask,
    TaskEvaluator,
)
from causal_gym.evaluator import random_agent, oracle_agent, naive_correlation_agent


def main():
    print("\n" + "=" * 60)
    print("  causal-interface-gym  |  Causal Reasoning Benchmark")
    print("=" * 60)

    # Build 5 tasks (some with different seeds for variety)
    tasks = [
        BackdoorAdjustmentTask(seed=42),
        FrontdoorAdjustmentTask(seed=7),
        CounterfactualTask(seed=13),
        InterventionPredictionTask(seed=99),
        BackdoorAdjustmentTask(seed=1234),  # 2nd instance, different seed
    ]

    evaluator = TaskEvaluator(tasks=tasks, verbose=True)

    # ----------------------------------------------------------------
    # Ground truths for reference
    # ----------------------------------------------------------------
    print("\n[Ground Truths]")
    for task in tasks:
        gt = task.ground_truth()
        print(f"  {task.name:<35} GT = {gt}")

    # ----------------------------------------------------------------
    # Agent 1: Random
    # ----------------------------------------------------------------
    print("\n\n[Agent: Random] (baseline)")
    random_report = evaluator.run(random_agent)
    print(random_report)

    # ----------------------------------------------------------------
    # Agent 2: Naive correlation (ignores confounding)
    # ----------------------------------------------------------------
    print("\n[Agent: Naive Correlation] (ignores confounders)")
    naive_report = evaluator.run(naive_correlation_agent)
    print(naive_report)

    # ----------------------------------------------------------------
    # Agent 3: Oracle (ground truth — upper bound)
    # ----------------------------------------------------------------
    print("\n[Agent: Oracle] (perfect causal reasoning)")

    def oracle(ctx):
        # For demo: oracle peeks at ground truth via task lookup
        # In a real benchmark, the agent wouldn't have this access
        task_name = ctx["task_name"]
        task_map = {t.name + str(i): t for i, t in enumerate(tasks)}
        # Find matching task
        for t in tasks:
            if t.name == task_name:
                return t.ground_truth()
        return 0.0

    oracle_report = evaluator.run(oracle)
    print(oracle_report)

    # ----------------------------------------------------------------
    # Summary comparison
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Score Summary")
    print("=" * 60)
    print(f"  Random agent          : {random_report.mean_score:.3f}")
    print(f"  Naive correlation     : {naive_report.mean_score:.3f}")
    print(f"  Oracle (upper bound)  : {oracle_report.mean_score:.3f}")
    print()
    print("  LLM agents should score between naive and oracle.")
    print("  A score > naive correlation indicates genuine causal reasoning.")
    print("=" * 60)


if __name__ == "__main__":
    main()
