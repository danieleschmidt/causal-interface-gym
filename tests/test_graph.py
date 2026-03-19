"""Tests for CausalGraph."""

import math
import pytest
from causal_gym import CausalGraph


def make_simple_graph(seed=42):
    """X -> Y simple linear graph."""
    g = CausalGraph(seed=seed)
    g.add_node("X", noise_std=1.0)
    g.add_node("Y", equation=lambda p, n: 2.0 * p["X"] + n, noise_std=0.1)
    g.add_edge("X", "Y")
    return g


def make_confounded_graph(seed=42):
    """Z -> X -> Y, Z -> Y (backdoor)."""
    g = CausalGraph(seed=seed)
    g.add_node("Z", noise_std=1.0)
    g.add_node("X", equation=lambda p, n: 0.8 * p["Z"] + n, noise_std=0.3)
    g.add_node("Y", equation=lambda p, n: 1.5 * p["X"] + 0.5 * p["Z"] + n, noise_std=0.3)
    g.add_edge("Z", "X")
    g.add_edge("Z", "Y")
    g.add_edge("X", "Y")
    return g


class TestCausalGraphBasics:
    def test_add_nodes_and_edges(self):
        g = make_simple_graph()
        assert "X" in g.nodes()
        assert "Y" in g.nodes()
        assert ("X", "Y") in g.edges()

    def test_parents_children(self):
        g = make_simple_graph()
        assert g.parents("Y") == ["X"]
        assert g.children("X") == ["Y"]
        assert g.parents("X") == []
        assert g.children("Y") == []

    def test_cycle_detection(self):
        g = CausalGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")
        with pytest.raises(ValueError, match="cycle"):
            g.add_edge("B", "A")

    def test_missing_node_raises(self):
        g = CausalGraph()
        g.add_node("A")
        with pytest.raises(ValueError):
            g.add_edge("A", "MISSING")

    def test_sample_returns_all_nodes(self):
        g = make_simple_graph()
        sample = g.sample()
        assert "X" in sample
        assert "Y" in sample

    def test_sample_n_returns_list(self):
        g = make_simple_graph()
        samples = g.sample(n=10)
        assert isinstance(samples, list)
        assert len(samples) == 10

    def test_topological_order_respected(self):
        """Y should depend on X in all samples."""
        g = CausalGraph(seed=0)
        g.add_node("X", noise_std=1.0)
        g.add_node("Y", equation=lambda p, n: p["X"] * 10 + n, noise_std=0.01)
        g.add_edge("X", "Y")

        for _ in range(20):
            s = g.sample()
            # Y ≈ 10 * X
            assert abs(s["Y"] - 10 * s["X"]) < 0.5


class TestDoInterventions:
    def test_intervention_fixes_variable(self):
        """do(X=5) should set X=5 in every sample."""
        g = make_simple_graph(seed=1)
        for _ in range(20):
            s = g.sample(interventions={"X": 5.0})
            assert s["X"] == 5.0

    def test_intervention_cuts_incoming_edges(self):
        """Under do(X=x), X should not vary with Z."""
        g = make_confounded_graph(seed=2)
        # Without intervention, X varies with Z
        obs = g.sample(n=100)
        x_vals_obs = [s["X"] for s in obs]

        # Under do(X=3), all X values should be 3
        int_samples = g.sample(interventions={"X": 3.0}, n=20)
        for s in int_samples:
            assert s["X"] == 3.0

    def test_ace_backdoor(self):
        """E[Y|do(X=1)] - E[Y|do(X=0)] should ≈ 1.5 in backdoor graph."""
        g = make_confounded_graph(seed=42)
        e1 = sum(s["Y"] for s in g.sample(interventions={"X": 1.0}, n=10000)) / 10000
        e0 = sum(s["Y"] for s in g.sample(interventions={"X": 0.0}, n=10000)) / 10000
        ace = e1 - e0
        assert abs(ace - 1.5) < 0.1, f"ACE was {ace:.3f}, expected ≈ 1.5"

    def test_interventional_graph_do(self):
        g = make_simple_graph()
        ig = g.do({"X": 3.0})
        ev = ig.expected_value("Y", n_samples=5000)
        # E[Y | do(X=3)] = 2 * 3 = 6
        assert abs(ev - 6.0) < 0.2, f"Expected ≈ 6.0, got {ev:.3f}"

    def test_observational_vs_interventional_differ(self):
        """Naive correlation should differ from causal effect under confounding."""
        g = make_confounded_graph(seed=10)
        obs = g.sample(n=5000)
        # Naive: corr(X, Y) includes confounder Z
        x_mean = sum(s["X"] for s in obs) / len(obs)
        y_mean = sum(s["Y"] for s in obs) / len(obs)
        cov_xy = sum((s["X"] - x_mean) * (s["Y"] - y_mean) for s in obs) / len(obs)
        var_x = sum((s["X"] - x_mean) ** 2 for s in obs) / len(obs)
        naive_effect = cov_xy / var_x if var_x > 0 else 0

        # True effect
        e1 = sum(s["Y"] for s in g.sample(interventions={"X": 1.0}, n=5000)) / 5000
        e0 = sum(s["Y"] for s in g.sample(interventions={"X": 0.0}, n=5000)) / 5000
        true_effect = e1 - e0

        # Naive should overestimate due to Z -> Y confounding
        assert naive_effect > true_effect + 0.1, (
            f"Naive={naive_effect:.3f} should exceed true={true_effect:.3f}"
        )


class TestGraphStructure:
    def test_ancestors(self):
        g = CausalGraph()
        for n in ["A", "B", "C", "D"]:
            g.add_node(n)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        assert g.ancestors("D") == {"A", "B", "C"}
        assert g.ancestors("A") == set()

    def test_descendants(self):
        g = CausalGraph()
        for n in ["A", "B", "C"]:
            g.add_node(n)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        assert g.descendants("A") == {"B", "C"}

    def test_has_path(self):
        g = make_simple_graph()
        assert g.has_path("X", "Y")
        assert not g.has_path("Y", "X")

    def test_backdoor_paths(self):
        g = make_confounded_graph()
        # X has a backdoor path through Z: X <- Z -> Y
        paths = g.backdoor_paths("X", "Y")
        assert len(paths) >= 1
        # Path should go through Z
        assert any("Z" in path for path in paths)
