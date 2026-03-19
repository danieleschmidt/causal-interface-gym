"""
causal_gym — A gym for testing LLM agents on causal reasoning tasks.

Implements do-calculus interventions embedded in structured environments,
enabling rigorous evaluation of causal understanding in language models.
"""

from .graph import CausalGraph
from .world import CausalWorld
from .tasks import (
    CausalTask,
    BackdoorAdjustmentTask,
    FrontdoorAdjustmentTask,
    CounterfactualTask,
    InterventionPredictionTask,
)
from .evaluator import TaskEvaluator

__all__ = [
    "CausalGraph",
    "CausalWorld",
    "CausalTask",
    "BackdoorAdjustmentTask",
    "FrontdoorAdjustmentTask",
    "CounterfactualTask",
    "InterventionPredictionTask",
    "TaskEvaluator",
]

__version__ = "0.1.0"
