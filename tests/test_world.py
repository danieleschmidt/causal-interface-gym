"""Tests for CausalWorld."""

import pytest
from causal_gym import CausalGraph, CausalWorld


def make_world(seed=0):
    g = CausalGraph(seed=seed)
    g.add_node("X", noise_std=1.0)
    g.add_node("Y", equation=lambda p, n: 2.0 * p["X"] + n, noise_std=0.1)
    g.add_edge("X", "Y")
    return CausalWorld(g, intervention_budget=3, seed=seed)


class TestCausalWorld:
    def test_reset_returns_state(self):
        w = make_world()
        state = w.reset()
        assert "X" in state and "Y" in state

    def test_observe_returns_current_state(self):
        w = make_world()
        state = w.reset()
        obs = w.observe()
        assert obs == state

    def test_intervene_fixes_variable(self):
        w = make_world()
        w.reset()
        new_state = w.intervene({"X": 10.0})
        assert new_state["X"] == 10.0
        assert new_state["Y"] == pytest.approx(20.0, abs=0.5)

    def test_query_returns_value(self):
        w = make_world()
        w.reset()
        w.intervene({"X": 3.0})
        assert w.query("X") == 3.0
        assert abs(w.query("Y") - 6.0) < 0.5

    def test_intervention_budget_enforced(self):
        w = make_world()
        w.reset()
        w.intervene({"X": 1.0})
        w.intervene({"X": 2.0})
        w.intervene({"X": 3.0})
        with pytest.raises(RuntimeError, match="budget"):
            w.intervene({"X": 4.0})

    def test_budget_remaining(self):
        w = make_world()
        w.reset()
        assert w.interventions_remaining == 3
        w.intervene({"X": 1.0})
        assert w.interventions_remaining == 2

    def test_expected_outcome_causal(self):
        """E[Y | do(X=2)] should ≈ 4."""
        w = make_world(seed=1)
        ev = w.expected_outcome("Y", {"X": 2.0}, n_samples=5000)
        assert abs(ev - 4.0) < 0.2

    def test_score_perfect(self):
        w = make_world()
        assert w.score(1.5, 1.5) == 1.0

    def test_score_within_tolerance(self):
        w = make_world()
        # 5% error within 10% tolerance
        assert w.score(1.55, 1.5, tolerance=0.1) == 1.0

    def test_score_zero_for_large_error(self):
        w = make_world()
        s = w.score(100.0, 1.0)
        assert s < 0.01

    def test_counterfactual(self):
        """
        Counterfactual under the twin-network approximation:
        fixes non-intervened context variables and applies the intervention.

        For X -> Y (Y = 3*X + noise):
          factual obs: X=2, Y=6 (noise_Y ≈ 0)
          do(X=0): since X is in the intervention, the context does NOT fix X.
          Non-intervened context = {} (only X observed, and X is intervened).
          So Y is sampled freely under do(X=0): E[Y|do(X=0)] ≈ 0.
        """
        g = CausalGraph(seed=5)
        g.add_node("X", equation=lambda p, n: 2.0 + n, noise_std=0.5)
        g.add_node("Y", equation=lambda p, n: 3.0 * p["X"] + n, noise_std=0.1)
        g.add_edge("X", "Y")
        world = CausalWorld(g, intervention_budget=10)

        # Only pass X in obs; counterfactual sets X=0
        obs = {"X": 2.0}
        cf = world.counterfactual(obs, {"X": 0.0}, "Y", n_samples=2000)
        # Under do(X=0), context has no free variables, so E[Y] = 3*0 = 0
        assert abs(cf - 0.0) < 0.3, f"Expected ≈ 0.0, got {cf:.3f}"
