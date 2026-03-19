"""
CausalWorld: environment where an agent observes variables, performs
interventions, and queries outcomes.

The world wraps a CausalGraph and provides a step-based interface:
  - observe()  -> current variable values
  - intervene(X=x) -> set do(X=x), resample the world
  - query(Y) -> value of Y after current interventions
  - reset() -> new observational draw, clear interventions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .graph import CausalGraph


class CausalWorld:
    """
    Gym-style environment backed by a Structural Causal Model.

    An agent interacts with the world by:
      1. Observing variables (passive, no intervention)
      2. Performing interventions: do(X=x)
      3. Querying outcome variables
      4. Collecting a reward based on how close their answer is to ground truth

    Example::

        world = CausalWorld(graph)
        obs = world.reset()               # fresh observational sample
        world.intervene({"X": 1.0})       # do(X=1)
        result = world.query("Y")         # E[Y | do(X=1)]
        reward = world.score(result, ground_truth=2.1)
    """

    def __init__(
        self,
        graph: CausalGraph,
        intervention_budget: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Args:
            graph: The underlying CausalGraph (SCM).
            intervention_budget: Max number of interventions per episode.
            seed: Random seed for reproducibility.
        """
        self.graph = graph
        self.intervention_budget = intervention_budget
        self._current_state: Dict[str, float] = {}
        self._active_interventions: Dict[str, float] = {}
        self._intervention_count: int = 0
        self._history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, float]:
        """
        Reset to a fresh observational sample (no interventions).
        Returns the observed variable values.
        """
        self._active_interventions = {}
        self._intervention_count = 0
        self._history = []
        self._current_state = self.graph.sample()
        return dict(self._current_state)

    def observe(self) -> Dict[str, float]:
        """Return current variable values (post-intervention if any)."""
        return dict(self._current_state)

    def intervene(self, interventions: Dict[str, float]) -> Dict[str, float]:
        """
        Perform do-interventions and resample the world.

        Args:
            interventions: {variable: value} pairs to intervene on.

        Returns:
            New state after intervention.

        Raises:
            RuntimeError: If intervention budget is exhausted.
        """
        remaining = self.intervention_budget - self._intervention_count
        if len(interventions) > remaining:
            raise RuntimeError(
                f"Intervention budget exceeded: {remaining} interventions left, "
                f"tried to use {len(interventions)}."
            )

        self._active_interventions.update(interventions)
        self._intervention_count += len(interventions)
        self._current_state = self.graph.sample(
            interventions=self._active_interventions
        )

        self._history.append({
            "action": "intervene",
            "interventions": dict(interventions),
            "resulting_state": dict(self._current_state),
        })

        return dict(self._current_state)

    def query(self, variable: str) -> float:
        """
        Query the current value of a variable.

        Args:
            variable: Name of the variable to query.

        Returns:
            Current value (may reflect active interventions).
        """
        if variable not in self._current_state:
            raise ValueError(f"Variable '{variable}' not found in world state.")
        return self._current_state[variable]

    def expected_outcome(
        self, target: str, interventions: Dict[str, float], n_samples: int = 5000
    ) -> float:
        """
        Compute E[target | do(interventions)] via Monte Carlo.

        This is the ground-truth causal effect — used by TaskEvaluator.
        """
        samples = self.graph.sample(interventions=interventions, n=n_samples)
        return sum(s[target] for s in samples) / n_samples

    def counterfactual(
        self,
        factual_obs: Dict[str, float],
        intervention: Dict[str, float],
        target: str,
        n_samples: int = 5000,
    ) -> float:
        """
        Estimate counterfactual: "What would Y have been if do(X=x),
        given that we observed factual_obs?"

        Uses the twin-network / abduction-action-prediction approach:
          1. Abduction: infer noise from factual obs (approximate via mean)
          2. Action: apply intervention
          3. Prediction: propagate through mutilated graph

        For linear SCMs this is exact; for non-linear it's approximate.
        """
        # Simplified: fix observed non-intervened variables as "context" and
        # apply intervention — this is the Twin Network approximation
        context = {k: v for k, v in factual_obs.items() if k not in intervention}
        combined = {**context, **intervention}
        samples = self.graph.sample(interventions=combined, n=n_samples)
        return sum(s[target] for s in samples) / n_samples

    def score(self, prediction: float, ground_truth: float, tolerance: float = 0.1) -> float:
        """
        Score a numeric prediction against ground truth.

        Returns 1.0 if within tolerance, else continuous score based on
        relative error. Returns 0.0 if prediction is None.
        """
        if prediction is None:
            return 0.0
        try:
            prediction = float(prediction)
        except (TypeError, ValueError):
            return 0.0
        if abs(ground_truth) < 1e-8:
            error = abs(prediction - ground_truth)
        else:
            error = abs(prediction - ground_truth) / abs(ground_truth)

        if error <= tolerance:
            return 1.0
        # Exponential decay for larger errors
        import math
        return max(0.0, math.exp(-5 * error))

    @property
    def interventions_remaining(self) -> int:
        return self.intervention_budget - self._intervention_count

    @property
    def active_interventions(self) -> Dict[str, float]:
        return dict(self._active_interventions)

    def __repr__(self) -> str:
        return (
            f"CausalWorld(nodes={self.graph.nodes()}, "
            f"interventions_remaining={self.interventions_remaining})"
        )
