"""
CausalGraph: DAG representation with structural equations and do() operator.

A Structural Causal Model (SCM) is defined by:
  - A set of variables V
  - A directed acyclic graph G over V
  - Structural equations: V_i = f_i(pa(V_i), U_i)
    where pa(V_i) are parents of V_i and U_i is an exogenous noise term

The do() operator simulates intervention: do(X=x) removes all incoming
edges to X and fixes X to value x, breaking the natural causal mechanism.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class CausalGraph:
    """
    Directed Acyclic Graph with structural equations supporting the do() operator.

    Each node has:
      - A structural equation: f(parent_values, noise) -> value
      - An optional noise distribution (default: Gaussian)

    Example usage::

        g = CausalGraph()
        g.add_node("X", equation=lambda parents, noise: noise)
        g.add_node("Y", equation=lambda parents, noise: 2 * parents["X"] + noise)
        g.add_edge("X", "Y")

        # Observational sample
        obs = g.sample()

        # Interventional sample: do(X=1)
        int_sample = g.sample(interventions={"X": 1.0})
    """

    def __init__(self, seed: Optional[int] = None):
        self._nodes: Dict[str, Dict] = {}
        self._edges: List[Tuple[str, str]] = []
        self._parents: Dict[str, List[str]] = defaultdict(list)
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_node(
        self,
        name: str,
        equation: Optional[Callable] = None,
        noise_std: float = 0.1,
        noise_fn: Optional[Callable] = None,
    ) -> "CausalGraph":
        """
        Add a variable node.

        Args:
            name: Variable name.
            equation: Callable(parent_values: dict, noise: float) -> float.
                      Defaults to identity of noise (exogenous variable).
            noise_std: Std dev of Gaussian noise (used when noise_fn is None).
            noise_fn: Custom noise generator (callable returning a float).
        """
        if equation is None:
            equation = lambda parents, noise: noise  # noqa: E731

        self._nodes[name] = {
            "equation": equation,
            "noise_std": noise_std,
            "noise_fn": noise_fn,
        }
        return self

    def add_edge(self, parent: str, child: str) -> "CausalGraph":
        """Add a directed edge parent -> child."""
        if parent not in self._nodes:
            raise ValueError(f"Node '{parent}' not in graph. Add it first.")
        if child not in self._nodes:
            raise ValueError(f"Node '{child}' not in graph. Add it first.")
        if self._creates_cycle(parent, child):
            raise ValueError(
                f"Adding edge {parent} -> {child} would create a cycle."
            )
        self._edges.append((parent, child))
        self._parents[child].append(parent)
        self._children[parent].append(child)
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        interventions: Optional[Dict[str, float]] = None,
        n: int = 1,
    ) -> Dict[str, float] | List[Dict[str, float]]:
        """
        Draw samples from the SCM.

        Args:
            interventions: Dict of {variable: value} for do() operations.
                           Under do(X=x), X is fixed to x regardless of
                           its structural equation (incoming edges cut).
            n: Number of samples. Returns dict if n=1, list of dicts if n>1.

        Returns:
            Assignment of all variables to values.
        """
        interventions = interventions or {}
        if n == 1:
            return self._sample_once(interventions)
        return [self._sample_once(interventions) for _ in range(n)]

    def _sample_once(self, interventions: Dict[str, float]) -> Dict[str, float]:
        order = self._topological_sort()
        values: Dict[str, float] = {}

        for node in order:
            if node in interventions:
                # do(node=value): fix value, ignore structural equation
                values[node] = float(interventions[node])
            else:
                node_info = self._nodes[node]
                noise = self._draw_noise(node_info)
                parent_vals = {p: values[p] for p in self._parents[node]}
                values[node] = float(node_info["equation"](parent_vals, noise))

        return values

    def do(self, interventions: Dict[str, float]) -> "InterventionalGraph":
        """
        Return a mutilated graph under do(X=x, ...).

        The mutilated graph has all incoming edges to intervened variables
        removed and those variables fixed to their intervention values.
        """
        return InterventionalGraph(self, interventions)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def parents(self, node: str) -> List[str]:
        return list(self._parents[node])

    def children(self, node: str) -> List[str]:
        return list(self._children[node])

    def ancestors(self, node: str) -> Set[str]:
        result = set()
        queue = list(self._parents[node])
        while queue:
            n = queue.pop()
            if n not in result:
                result.add(n)
                queue.extend(self._parents[n])
        return result

    def descendants(self, node: str) -> Set[str]:
        result = set()
        queue = list(self._children[node])
        while queue:
            n = queue.pop()
            if n not in result:
                result.add(n)
                queue.extend(self._children[n])
        return result

    def nodes(self) -> List[str]:
        return list(self._nodes.keys())

    def edges(self) -> List[Tuple[str, str]]:
        return list(self._edges)

    def has_path(self, source: str, target: str, directed: bool = True) -> bool:
        """Check if there's a (directed) path from source to target."""
        visited = set()
        queue = [source]
        while queue:
            node = queue.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            neighbors = self._children[node] if directed else (
                self._children[node] + self._parents[node]
            )
            queue.extend(neighbors)
        return False

    def backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        Find all backdoor paths: paths from treatment to outcome that
        start with an arrow INTO treatment (i.e., via a common cause).
        """
        # A backdoor path goes: treatment <- ... -> outcome
        # We enumerate paths in the skeleton that go through parents of treatment
        paths = []
        for parent in self._parents[treatment]:
            found = self._find_paths_undirected(parent, outcome, visited={treatment})
            for path in found:
                paths.append([treatment, parent] + path[1:])
        return paths

    def _find_paths_undirected(
        self,
        start: str,
        end: str,
        visited: Set[str],
        current: Optional[List[str]] = None,
    ) -> List[List[str]]:
        if current is None:
            current = [start]
        if start == end:
            return [list(current)]
        paths = []
        neighbors = set(self._children[start]) | set(self._parents[start])
        neighbors -= visited
        for n in neighbors:
            paths.extend(
                self._find_paths_undirected(n, end, visited | {start}, current + [n])
            )
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_noise(self, node_info: Dict) -> float:
        if node_info["noise_fn"] is not None:
            return node_info["noise_fn"]()
        std = node_info["noise_std"]
        # Box-Muller for Gaussian without numpy dependency
        import math

        u1 = self._rng.random() or 1e-10
        u2 = self._rng.random()
        return std * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    def _topological_sort(self) -> List[str]:
        in_degree = {n: len(self._parents[n]) for n in self._nodes}
        queue = [n for n, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self._children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._nodes):
            raise ValueError("Graph contains a cycle — not a valid DAG.")
        return order

    def _creates_cycle(self, parent: str, child: str) -> bool:
        # Would adding parent->child create a cycle?
        # Check if child can reach parent already
        return self.has_path(child, parent)

    def __repr__(self) -> str:
        return (
            f"CausalGraph(nodes={list(self._nodes.keys())}, "
            f"edges={self._edges})"
        )


class InterventionalGraph:
    """
    A CausalGraph mutilated by do-interventions.

    Under do(X=x), the structural equation for X is replaced by a constant,
    and all incoming edges to X are removed.
    """

    def __init__(self, base: CausalGraph, interventions: Dict[str, float]):
        self._base = base
        self._interventions = interventions

    def sample(self, n: int = 1) -> Dict[str, float] | List[Dict[str, float]]:
        return self._base.sample(interventions=self._interventions, n=n)

    def expected_value(self, target: str, n_samples: int = 10000) -> float:
        """Monte-Carlo estimate of E[target | do(interventions)]."""
        samples = self.sample(n=n_samples)
        return sum(s[target] for s in samples) / n_samples

    def __repr__(self) -> str:
        return f"InterventionalGraph(do={self._interventions})"
