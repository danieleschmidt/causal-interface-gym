"""Tests for causal tasks and evaluator."""

import pytest
from causal_gym import (
    BackdoorAdjustmentTask,
    FrontdoorAdjustmentTask,
    CounterfactualTask,
    InterventionPredictionTask,
    TaskEvaluator,
)
from causal_gym.evaluator import oracle_agent, random_agent


class TestBackdoorTask:
    def test_ground_truth_near_1_5(self):
        task = BackdoorAdjustmentTask(seed=42)
        gt = task.ground_truth()
        assert abs(gt - 1.5) < 0.1, f"GT={gt}, expected ≈ 1.5"

    def test_oracle_scores_1(self):
        task = BackdoorAdjustmentTask(seed=42)
        result = task.evaluate(task.ground_truth())
        assert result.score == 1.0

    def test_wrong_answer_scores_low(self):
        task = BackdoorAdjustmentTask(seed=42)
        result = task.evaluate(10.0)  # wildly wrong
        assert result.score < 0.1

    def test_context_has_expected_keys(self):
        task = BackdoorAdjustmentTask(seed=0)
        ctx = task.context()
        assert "variables" in ctx
        assert "edges" in ctx
        assert "observational_samples" in ctx
        assert len(ctx["observational_samples"]) == 10

    def test_confounding_graph_structure(self):
        task = BackdoorAdjustmentTask(seed=0)
        edges = task.world.graph.edges()
        # Should have Z->X, Z->Y, X->Y
        assert ("Z", "X") in edges
        assert ("Z", "Y") in edges
        assert ("X", "Y") in edges


class TestFrontdoorTask:
    def test_ground_truth_near_1_6(self):
        task = FrontdoorAdjustmentTask(seed=7)
        gt = task.ground_truth()
        assert abs(gt - 1.6) < 0.1, f"GT={gt}, expected ≈ 1.6"

    def test_oracle_scores_1(self):
        task = FrontdoorAdjustmentTask(seed=7)
        result = task.evaluate(task.ground_truth())
        assert result.score == 1.0

    def test_mediator_in_graph(self):
        task = FrontdoorAdjustmentTask(seed=7)
        assert "M" in task.world.graph.nodes()
        assert ("X", "M") in task.world.graph.edges()
        assert ("M", "Y") in task.world.graph.edges()


class TestCounterfactualTask:
    def test_ground_truth_formula(self):
        """Y_cf = Y_obs - 3 * X_obs (when do(X=0))."""
        task = CounterfactualTask(seed=42)
        obs = task._get_factual()
        gt = task.ground_truth()
        expected = obs["Y"] - 3.0 * obs["X"]
        assert abs(gt - expected) < 0.01

    def test_oracle_scores_1(self):
        task = CounterfactualTask(seed=42)
        result = task.evaluate(task.ground_truth())
        assert result.score == 1.0

    def test_question_contains_observations(self):
        task = CounterfactualTask(seed=13)
        q = task.question()
        assert "X=" in q and "Y=" in q


class TestInterventionPredictionTask:
    def test_ground_truth_near_2_1(self):
        task = InterventionPredictionTask(seed=99)
        gt = task.ground_truth()
        assert abs(gt - 2.1) < 0.15, f"GT={gt}, expected ≈ 2.1"

    def test_oracle_scores_1(self):
        task = InterventionPredictionTask(seed=99)
        result = task.evaluate(task.ground_truth())
        assert result.score == 1.0

    def test_chain_structure(self):
        task = InterventionPredictionTask(seed=0)
        edges = task.world.graph.edges()
        assert ("X", "Y") in edges
        assert ("Y", "Z") in edges


class TestTaskEvaluator:
    def setup_method(self):
        self.tasks = [
            BackdoorAdjustmentTask(seed=42),
            CounterfactualTask(seed=13),
        ]
        self.evaluator = TaskEvaluator(self.tasks, verbose=False)

    def test_run_random_agent(self):
        report = self.evaluator.run(random_agent)
        assert len(report.results) == 2
        assert 0.0 <= report.mean_score <= 1.0

    def test_run_oracle_gets_perfect(self):
        def perfect_oracle(ctx):
            task_name = ctx["task_name"]
            for t in self.tasks:
                if t.name == task_name:
                    return t.ground_truth()
        report = self.evaluator.run(perfect_oracle)
        assert report.mean_score == pytest.approx(1.0, abs=0.01)

    def test_report_has_all_fields(self):
        report = self.evaluator.run(random_agent)
        assert hasattr(report, "results")
        assert hasattr(report, "mean_score")
        assert hasattr(report, "total_score")
        assert hasattr(report, "task_scores")
        assert hasattr(report, "elapsed_seconds")
        assert report.elapsed_seconds > 0

    def test_agent_exception_gives_zero(self):
        def bad_agent(ctx):
            raise ValueError("I crashed")

        report = self.evaluator.run(bad_agent)
        # Should still produce results, just with None answers and low scores
        assert len(report.results) == 2

    def test_task_scores_dict(self):
        report = self.evaluator.run(random_agent)
        assert "BackdoorAdjustmentTask" in report.task_scores
        assert "CounterfactualTask" in report.task_scores
