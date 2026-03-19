"""
Causal reasoning tasks for LLM agent evaluation.

Each task presents a causal scenario and asks the agent to reason about
interventions, adjustments, or counterfactuals. Tasks provide:
  - A natural language description of the causal scenario
  - A question to answer
  - Ground truth computation via the underlying SCM
  - Scoring logic

Task hierarchy:
  CausalTask (base)
  ├── BackdoorAdjustmentTask
  ├── FrontdoorAdjustmentTask
  ├── CounterfactualTask
  └── InterventionPredictionTask
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .graph import CausalGraph
from .world import CausalWorld


@dataclass
class TaskResult:
    """Result of a single task attempt."""
    task_name: str
    question: str
    agent_answer: Any
    ground_truth: Any
    score: float
    explanation: str = ""
    metadata: Dict = field(default_factory=dict)


class CausalTask(abc.ABC):
    """
    Base class for causal reasoning tasks.

    A task defines a causal scenario, a question, and ground-truth
    computation. Agents interact with the task's CausalWorld to
    gather evidence before committing to an answer.
    """

    name: str = "CausalTask"
    description: str = ""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._world: Optional[CausalWorld] = None

    @property
    def world(self) -> CausalWorld:
        if self._world is None:
            self._world = self._build_world()
        return self._world

    @abc.abstractmethod
    def _build_world(self) -> CausalWorld:
        """Construct the CausalWorld for this task."""
        ...

    @abc.abstractmethod
    def question(self) -> str:
        """Return the natural language question to answer."""
        ...

    @abc.abstractmethod
    def ground_truth(self) -> Any:
        """Compute the correct answer."""
        ...

    @abc.abstractmethod
    def score(self, agent_answer: Any) -> float:
        """Score the agent's answer (0.0 to 1.0)."""
        ...

    def context(self) -> Dict[str, Any]:
        """
        Return observable context for the agent:
        variable names, edges, and a sample of observational data.
        """
        obs = [self.world.graph.sample() for _ in range(10)]
        return {
            "variables": self.world.graph.nodes(),
            "edges": self.world.graph.edges(),
            "observational_samples": obs,
            "intervention_budget": self.world.intervention_budget,
        }

    def evaluate(self, agent_answer: Any) -> TaskResult:
        gt = self.ground_truth()
        s = self.score(agent_answer)
        return TaskResult(
            task_name=self.name,
            question=self.question(),
            agent_answer=agent_answer,
            ground_truth=gt,
            score=s,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Concrete Tasks
# ---------------------------------------------------------------------------


class BackdoorAdjustmentTask(CausalTask):
    """
    Task: Estimate the causal effect of X on Y using backdoor adjustment.

    Structure: Z -> X -> Y, Z -> Y
      - Z is a confounder (common cause of X and Y)
      - Naive association P(Y|X) is biased
      - Backdoor-adjusted effect: E[Y|do(X=x)] via controlling for Z

    The agent must identify that Z is a confounder and report the
    correct interventional expectation E[Y | do(X=1)] - E[Y | do(X=0)].
    """

    name = "BackdoorAdjustmentTask"
    description = (
        "Estimate the causal effect of X on Y when Z is a confounder. "
        "Distinguish spurious correlation from true causal effect."
    )

    def _build_world(self) -> CausalWorld:
        g = CausalGraph(seed=self.seed)

        # Z ~ N(0, 1) — confounder
        g.add_node("Z", noise_std=1.0)

        # X = 0.8*Z + noise — treatment (Z causes X)
        g.add_node(
            "X",
            equation=lambda p, n: 0.8 * p["Z"] + n,
            noise_std=0.3,
        )

        # Y = 1.5*X + 0.5*Z + noise — outcome (both X and Z cause Y)
        g.add_node(
            "Y",
            equation=lambda p, n: 1.5 * p["X"] + 0.5 * p["Z"] + n,
            noise_std=0.3,
        )

        g.add_edge("Z", "X")
        g.add_edge("Z", "Y")
        g.add_edge("X", "Y")

        return CausalWorld(g, intervention_budget=10, seed=self.seed)

    def question(self) -> str:
        return (
            "Given the causal graph Z -> X -> Y and Z -> Y, "
            "what is the Average Causal Effect (ACE) of X on Y? "
            "That is, compute E[Y | do(X=1)] - E[Y | do(X=0)]."
        )

    def ground_truth(self) -> float:
        # Under do(X=x), Z no longer affects X but still affects Y
        # E[Y | do(X=x)] = 1.5*x + 0.5*E[Z] = 1.5*x (since E[Z]=0)
        # ACE = E[Y|do(X=1)] - E[Y|do(X=0)] = 1.5
        w = self.world
        e1 = w.expected_outcome("Y", {"X": 1.0}, n_samples=10000)
        e0 = w.expected_outcome("Y", {"X": 0.0}, n_samples=10000)
        return round(e1 - e0, 3)

    def score(self, agent_answer: float) -> float:
        gt = self.ground_truth()
        return self.world.score(agent_answer, gt, tolerance=0.15)


class FrontdoorAdjustmentTask(CausalTask):
    """
    Task: Estimate causal effect via the frontdoor criterion.

    Structure: X -> M -> Y, X -> Y (unobserved U confounds X and Y)
      - There is an unobserved confounder U affecting both X and Y
      - Backdoor adjustment is impossible (U is hidden)
      - Frontdoor adjustment via mediator M allows identification:
        E[Y|do(X=x)] = sum_m P(M=m|X=x) * sum_x' P(Y|X=x', M=m) * P(X=x')

    The agent must recognize the frontdoor structure and apply the
    frontdoor formula (or use interventions to estimate the effect).
    """

    name = "FrontdoorAdjustmentTask"
    description = (
        "Estimate the causal effect of X on Y via mediator M, "
        "where X and Y share an unobserved confounder U."
    )

    def _build_world(self) -> CausalWorld:
        g = CausalGraph(seed=self.seed)

        # U ~ N(0,1) — hidden confounder (we include it in the graph
        # but agents can't directly observe it in the context)
        g.add_node("U", noise_std=1.0)

        # X = 0.5*U + noise
        g.add_node("X", equation=lambda p, n: 0.5 * p["U"] + n, noise_std=0.5)

        # M = 2.0*X + noise — mediator (X causes M fully)
        g.add_node("M", equation=lambda p, n: 2.0 * p["X"] + n, noise_std=0.3)

        # Y = 0.8*M + 0.6*U + noise — outcome (M and U cause Y)
        g.add_node(
            "Y",
            equation=lambda p, n: 0.8 * p["M"] + 0.6 * p["U"] + n,
            noise_std=0.3,
        )

        g.add_edge("U", "X")
        g.add_edge("U", "Y")
        g.add_edge("X", "M")
        g.add_edge("M", "Y")

        return CausalWorld(g, intervention_budget=10, seed=self.seed)

    def question(self) -> str:
        return (
            "In the graph X -> M -> Y with hidden confounder U affecting both X and Y, "
            "what is E[Y | do(X=1)] - E[Y | do(X=0)]? "
            "Note: U is unobserved; you cannot condition on it directly."
        )

    def ground_truth(self) -> float:
        # Under do(X=x): M = 2x + noise, Y = 0.8*M + 0.6*U + noise
        # E[Y|do(X=x)] = 0.8 * E[M|do(X=x)] + 0.6 * E[U]
        #               = 0.8 * 2x + 0 = 1.6x
        # ACE = 1.6
        w = self.world
        e1 = w.expected_outcome("Y", {"X": 1.0}, n_samples=10000)
        e0 = w.expected_outcome("Y", {"X": 0.0}, n_samples=10000)
        return round(e1 - e0, 3)

    def score(self, agent_answer: float) -> float:
        gt = self.ground_truth()
        return self.world.score(agent_answer, gt, tolerance=0.15)


class CounterfactualTask(CausalTask):
    """
    Task: Answer a counterfactual query.

    "Given that we observed X=2 and Y=4, what would Y have been if X=0?"

    Uses the Rubin potential outcomes / Pearl counterfactual framework:
    abduction (infer noise from observation), action (mutilate graph),
    prediction (forward propagate).

    Structure: X -> Y (direct linear effect)
    """

    name = "CounterfactualTask"
    description = (
        "Answer a counterfactual: given a factual observation, "
        "what would have happened under a hypothetical intervention?"
    )

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._factual_obs: Optional[Dict[str, float]] = None
        self._cf_intervention: Dict[str, float] = {"X": 0.0}
        self._cf_target: str = "Y"

    def _build_world(self) -> CausalWorld:
        g = CausalGraph(seed=self.seed)

        # X ~ N(2, 0.5) — treatment
        g.add_node("X", equation=lambda p, n: 2.0 + n, noise_std=0.5)

        # Y = 3*X + noise
        g.add_node("Y", equation=lambda p, n: 3.0 * p["X"] + n, noise_std=0.3)

        g.add_edge("X", "Y")

        return CausalWorld(g, intervention_budget=10, seed=self.seed)

    def question(self) -> str:
        obs = self._get_factual()
        return (
            f"We observed X={obs['X']:.2f} and Y={obs['Y']:.2f}. "
            f"What would Y have been if X had been set to 0 instead? "
            f"(Structure: Y = 3*X + noise)"
        )

    def _get_factual(self) -> Dict[str, float]:
        if self._factual_obs is None:
            # Fix a specific factual observation
            import random
            rng = random.Random(self.seed or 42)
            x_val = 2.0 + rng.gauss(0, 0.5)
            y_val = 3.0 * x_val + rng.gauss(0, 0.3)
            self._factual_obs = {"X": x_val, "Y": y_val}
        return self._factual_obs

    def ground_truth(self) -> float:
        """
        Counterfactual Y_{X=0} given factual observation.

        For Y = 3X + U_Y:
          Abduction: U_Y = Y_factual - 3*X_factual
          Action: X = 0
          Prediction: Y_cf = 3*0 + U_Y = U_Y = Y_factual - 3*X_factual
        """
        obs = self._get_factual()
        u_y = obs["Y"] - 3.0 * obs["X"]  # abduced noise
        y_cf = 3.0 * 0.0 + u_y  # Y under X=0
        return round(y_cf, 3)

    def score(self, agent_answer: float) -> float:
        gt = self.ground_truth()
        return self.world.score(agent_answer, gt, tolerance=0.2)


class InterventionPredictionTask(CausalTask):
    """
    Task: Predict the outcome of an intervention without prior knowledge
    of the structural equations — must learn from experimental queries.

    The agent is given a budget of interventional queries and must estimate
    E[Y | do(X=x)] for a target x using those queries.

    Structure: X -> Y -> Z (chain), with noise
    The agent must estimate E[Z | do(X=2)].
    """

    name = "InterventionPredictionTask"
    description = (
        "Learn the causal effect of X on Z through a mediating variable Y "
        "by performing interventional queries within a budget."
    )

    def _build_world(self) -> CausalWorld:
        g = CausalGraph(seed=self.seed)

        # X ~ N(0, 1)
        g.add_node("X", noise_std=1.0)

        # Y = 0.5*X^2 + noise (nonlinear)
        g.add_node("Y", equation=lambda p, n: 0.5 * p["X"] ** 2 + n, noise_std=0.2)

        # Z = 1.2*Y - 0.3 + noise
        g.add_node(
            "Z",
            equation=lambda p, n: 1.2 * p["Y"] - 0.3 + n,
            noise_std=0.2,
        )

        g.add_edge("X", "Y")
        g.add_edge("Y", "Z")

        return CausalWorld(g, intervention_budget=20, seed=self.seed)

    def question(self) -> str:
        return (
            "In the causal chain X -> Y -> Z, estimate E[Z | do(X=2)]. "
            "You have a budget of 20 interventional queries. "
            "Use them to build an estimate."
        )

    def ground_truth(self) -> float:
        # E[Y | do(X=2)] = 0.5 * 4 = 2.0
        # E[Z | do(X=2)] = 1.2 * 2.0 - 0.3 = 2.1
        w = self.world
        return round(w.expected_outcome("Z", {"X": 2.0}, n_samples=10000), 3)

    def score(self, agent_answer: float) -> float:
        gt = self.ground_truth()
        return self.world.score(agent_answer, gt, tolerance=0.2)
