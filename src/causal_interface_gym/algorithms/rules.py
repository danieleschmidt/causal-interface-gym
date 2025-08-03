"""Business rules engine for causal reasoning experiments."""

import logging
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import json
import re

from ..core import CausalEnvironment
from ..metrics import BeliefState
from .scoring import CausalScore

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of business rules."""
    VALIDATION = "validation"
    CONSTRAINT = "constraint"
    TRANSFORMATION = "transformation"
    SCORING_ADJUSTMENT = "scoring_adjustment"
    EXPERIMENT_CONTROL = "experiment_control"
    SAFETY = "safety"


class RulePriority(Enum):
    """Rule execution priority levels."""
    CRITICAL = 1  # Must execute first (safety, validation)
    HIGH = 2     # Important business logic
    MEDIUM = 3   # Standard rules
    LOW = 4      # Optional enhancements


@dataclass
class RuleResult:
    """Result of rule execution."""
    rule_id: str
    success: bool
    message: str
    action_taken: str
    modifications: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleContext:
    """Context for rule execution."""
    experiment_id: str
    agent_id: str
    environment: CausalEnvironment
    belief_history: List[BeliefState]
    interventions: List[Dict[str, Any]]
    ground_truth: Dict[str, Any]
    current_score: Optional[CausalScore] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BusinessRule(ABC):
    """Abstract base class for business rules."""
    
    def __init__(self, rule_id: str, rule_type: RuleType, priority: RulePriority,
                 description: str, enabled: bool = True):
        """Initialize business rule.
        
        Args:
            rule_id: Unique rule identifier
            rule_type: Type of rule
            priority: Execution priority
            description: Human-readable description
            enabled: Whether rule is active
        """
        self.rule_id = rule_id
        self.rule_type = rule_type
        self.priority = priority
        self.description = description
        self.enabled = enabled
        self.execution_count = 0
        self.last_executed = None
    
    @abstractmethod
    def evaluate(self, context: RuleContext) -> bool:
        """Evaluate if rule should be applied.
        
        Args:
            context: Rule execution context
            
        Returns:
            True if rule should be applied
        """
        pass
    
    @abstractmethod
    def execute(self, context: RuleContext) -> RuleResult:
        """Execute the rule action.
        
        Args:
            context: Rule execution context
            
        Returns:
            Rule execution result
        """
        pass
    
    def can_execute(self, context: RuleContext) -> bool:
        """Check if rule can execute.
        
        Args:
            context: Rule execution context
            
        Returns:
            True if rule can execute
        """
        return self.enabled and self.evaluate(context)


class ExperimentValidationRule(BusinessRule):
    """Validates experiment data quality and completeness."""
    
    def __init__(self):
        super().__init__(
            rule_id="experiment_validation",
            rule_type=RuleType.VALIDATION,
            priority=RulePriority.CRITICAL,
            description="Validates experiment data quality and completeness"
        )
        
        self.min_beliefs = 2
        self.max_belief_change = 2.0
        self.required_fields = ["experiment_id", "agent_id"]
    
    def evaluate(self, context: RuleContext) -> bool:
        """Always evaluate validation rules."""
        return True
    
    def execute(self, context: RuleContext) -> RuleResult:
        """Execute validation checks."""
        self.execution_count += 1
        self.last_executed = datetime.now()
        
        issues = []
        
        # Check required fields
        for field in self.required_fields:
            if not getattr(context, field, None):
                issues.append(f"Missing required field: {field}")
        
        # Check belief history
        if len(context.belief_history) < self.min_beliefs:
            issues.append(f"Insufficient belief history: {len(context.belief_history)} < {self.min_beliefs}")
        
        # Check for reasonable belief changes
        if len(context.belief_history) >= 2:
            for i in range(1, len(context.belief_history)):
                prev_beliefs = context.belief_history[i-1].causal_beliefs
                curr_beliefs = context.belief_history[i].causal_beliefs
                
                total_change = sum(abs(curr_beliefs.get(k, 0) - prev_beliefs.get(k, 0))
                                 for k in set(prev_beliefs.keys()) | set(curr_beliefs.keys()))
                
                if total_change > self.max_belief_change:
                    issues.append(f"Excessive belief change at step {i}: {total_change:.3f}")
        
        # Check environment consistency
        if not context.environment.variables:
            issues.append("Environment has no variables defined")
        
        success = len(issues) == 0
        
        return RuleResult(
            rule_id=self.rule_id,
            success=success,
            message="Validation passed" if success else f"Validation failed: {'; '.join(issues)}",
            action_taken="validation_check",
            metadata={
                "issues_found": len(issues),
                "issues": issues,
                "belief_count": len(context.belief_history),
                "intervention_count": len(context.interventions)
            }
        )


class SafetyConstraintRule(BusinessRule):
    """Enforces safety constraints on experiments."""
    
    def __init__(self):
        super().__init__(
            rule_id="safety_constraint",
            rule_type=RuleType.SAFETY,
            priority=RulePriority.CRITICAL,
            description="Enforces safety constraints on interventions"
        )
        
        self.max_intervention_value = 5.0
        self.max_interventions_per_experiment = 100
        self.forbidden_variables = set()  # Variables that cannot be intervened on
    
    def evaluate(self, context: RuleContext) -> bool:
        """Evaluate if safety check is needed."""
        return len(context.interventions) > 0
    
    def execute(self, context: RuleContext) -> RuleResult:
        """Execute safety checks."""
        self.execution_count += 1
        self.last_executed = datetime.now()
        
        violations = []
        
        # Check intervention count
        if len(context.interventions) > self.max_interventions_per_experiment:
            violations.append(f"Too many interventions: {len(context.interventions)}")
        
        # Check intervention values
        for i, intervention in enumerate(context.interventions):
            value = intervention.get("value", 0)
            if abs(value) > self.max_intervention_value:
                violations.append(f"Intervention {i} value too extreme: {value}")
            
            variable = intervention.get("variable")
            if variable in self.forbidden_variables:
                violations.append(f"Intervention on forbidden variable: {variable}")
        
        success = len(violations) == 0
        
        return RuleResult(
            rule_id=self.rule_id,
            success=success,
            message="Safety check passed" if success else f"Safety violations: {'; '.join(violations)}",
            action_taken="safety_check",
            metadata={
                "violations_found": len(violations),
                "violations": violations,
                "max_intervention_value": max(abs(i.get("value", 0)) for i in context.interventions) if context.interventions else 0
            }
        )


class BeliefConsistencyRule(BusinessRule):
    """Ensures belief evolution is consistent and reasonable."""
    
    def __init__(self):
        super().__init__(
            rule_id="belief_consistency",
            rule_type=RuleType.CONSTRAINT,
            priority=RulePriority.HIGH,
            description="Ensures belief evolution is consistent"
        )
        
        self.max_sudden_change = 0.8
        self.min_confidence_threshold = 0.1
    
    def evaluate(self, context: RuleContext) -> bool:
        """Evaluate if belief consistency check is needed."""
        return len(context.belief_history) >= 2
    
    def execute(self, context: RuleContext) -> RuleResult:
        """Execute belief consistency checks."""
        self.execution_count += 1
        self.last_executed = datetime.now()
        
        issues = []
        
        # Check for sudden, unexplained belief changes
        for i in range(1, len(context.belief_history)):
            prev_beliefs = context.belief_history[i-1].causal_beliefs
            curr_beliefs = context.belief_history[i].causal_beliefs
            
            for relationship in set(prev_beliefs.keys()) | set(curr_beliefs.keys()):
                prev_val = prev_beliefs.get(relationship, 0)
                curr_val = curr_beliefs.get(relationship, 0)
                change = abs(curr_val - prev_val)
                
                if change > self.max_sudden_change:
                    # Check if there was an intervention that might explain this
                    intervention_nearby = any(
                        intervention.get("variable") in relationship
                        for j, intervention in enumerate(context.interventions)
                        if abs(j - i) <= 1  # Within 1 step
                    )
                    
                    if not intervention_nearby:
                        issues.append(f"Unexplained belief change for {relationship}: {change:.3f}")
        
        # Check belief confidence levels
        for belief_state in context.belief_history:
            avg_confidence = sum(abs(v) for v in belief_state.causal_beliefs.values()) / max(len(belief_state.causal_beliefs), 1)
            if avg_confidence < self.min_confidence_threshold:
                issues.append(f"Very low confidence beliefs at {belief_state.timestamp}")
        
        success = len(issues) == 0
        
        return RuleResult(
            rule_id=self.rule_id,
            success=success,
            message="Belief consistency check passed" if success else f"Consistency issues: {'; '.join(issues[:3])}",
            action_taken="consistency_check",
            metadata={
                "issues_found": len(issues),
                "issues": issues[:10],  # Limit to first 10
                "avg_belief_change": self._calculate_avg_belief_change(context.belief_history)
            }
        )
    
    def _calculate_avg_belief_change(self, belief_history: List[BeliefState]) -> float:
        """Calculate average belief change across history."""
        if len(belief_history) < 2:
            return 0.0
        
        total_changes = []
        for i in range(1, len(belief_history)):
            prev = belief_history[i-1].causal_beliefs
            curr = belief_history[i].causal_beliefs
            
            total_change = sum(abs(curr.get(k, 0) - prev.get(k, 0))
                             for k in set(prev.keys()) | set(curr.keys()))
            total_changes.append(total_change)
        
        return sum(total_changes) / len(total_changes)


class ScoringAdjustmentRule(BusinessRule):
    """Adjusts scores based on specific criteria."""
    
    def __init__(self):
        super().__init__(
            rule_id="scoring_adjustment",
            rule_type=RuleType.SCORING_ADJUSTMENT,
            priority=RulePriority.MEDIUM,
            description="Adjusts scores based on experiment characteristics"
        )
        
        self.bonus_for_efficiency = 0.1
        self.penalty_for_excessive_interventions = 0.05
        self.bonus_for_early_discovery = 0.05
    
    def evaluate(self, context: RuleContext) -> bool:
        """Evaluate if scoring adjustment is needed."""
        return context.current_score is not None
    
    def execute(self, context: RuleContext) -> RuleResult:
        """Execute scoring adjustments."""
        self.execution_count += 1
        self.last_executed = datetime.now()
        
        if not context.current_score:
            return RuleResult(
                rule_id=self.rule_id,
                success=False,
                message="No score available for adjustment",
                action_taken="no_action"
            )
        
        adjustments = {}
        total_adjustment = 0.0
        
        # Efficiency bonus
        if context.interventions:
            avg_info_gain = context.current_score.intervention_efficiency
            if avg_info_gain > 0.8:
                efficiency_bonus = self.bonus_for_efficiency
                adjustments["efficiency_bonus"] = efficiency_bonus
                total_adjustment += efficiency_bonus
        
        # Penalty for excessive interventions
        intervention_ratio = len(context.interventions) / max(len(context.belief_history), 1)
        if intervention_ratio > 0.5:  # More than 0.5 interventions per belief update
            excess_penalty = self.penalty_for_excessive_interventions * (intervention_ratio - 0.5)
            adjustments["excess_intervention_penalty"] = -excess_penalty
            total_adjustment -= excess_penalty
        
        # Early discovery bonus
        if len(context.belief_history) >= 3:
            early_discoveries = 0
            for i, belief_state in enumerate(context.belief_history[:3]):  # First 3 states
                strong_beliefs = sum(1 for v in belief_state.causal_beliefs.values() if v > 0.7)
                early_discoveries += strong_beliefs
            
            if early_discoveries > 0:
                discovery_bonus = self.bonus_for_early_discovery * min(early_discoveries / 5, 1)
                adjustments["early_discovery_bonus"] = discovery_bonus
                total_adjustment += discovery_bonus
        
        # Apply adjustments
        if total_adjustment != 0:
            original_score = context.current_score.total_score
            new_score = min(1.0, max(0.0, original_score + total_adjustment))
            
            modifications = {
                "original_score": original_score,
                "adjustment": total_adjustment,
                "adjusted_score": new_score
            }
        else:
            modifications = {}
        
        return RuleResult(
            rule_id=self.rule_id,
            success=True,
            message=f"Applied {len(adjustments)} scoring adjustments",
            action_taken="score_adjustment",
            modifications=modifications,
            metadata={
                "adjustments": adjustments,
                "total_adjustment": total_adjustment
            }
        )


class ExperimentTerminationRule(BusinessRule):
    """Determines when experiments should be terminated."""
    
    def __init__(self):
        super().__init__(
            rule_id="experiment_termination",
            rule_type=RuleType.EXPERIMENT_CONTROL,
            priority=RulePriority.HIGH,
            description="Determines when experiments should be terminated"
        )
        
        self.max_experiment_steps = 100
        self.convergence_threshold = 0.01
        self.convergence_window = 5
        self.min_steps_before_termination = 10
    
    def evaluate(self, context: RuleContext) -> bool:
        """Evaluate if termination check is needed."""
        return len(context.belief_history) >= self.min_steps_before_termination
    
    def execute(self, context: RuleContext) -> RuleResult:
        """Execute termination logic."""
        self.execution_count += 1
        self.last_executed = datetime.now()
        
        should_terminate = False
        termination_reason = None
        
        # Check maximum steps
        if len(context.belief_history) >= self.max_experiment_steps:
            should_terminate = True
            termination_reason = "max_steps_reached"
        
        # Check for convergence
        elif self._check_convergence(context.belief_history):
            should_terminate = True
            termination_reason = "belief_convergence"
        
        # Check for no progress
        elif self._check_no_progress(context.belief_history):
            should_terminate = True
            termination_reason = "no_progress"
        
        action_taken = "terminate" if should_terminate else "continue"
        
        return RuleResult(
            rule_id=self.rule_id,
            success=True,
            message=f"Experiment should {action_taken}" + (f" ({termination_reason})" if termination_reason else ""),
            action_taken=action_taken,
            metadata={
                "should_terminate": should_terminate,
                "reason": termination_reason,
                "steps_completed": len(context.belief_history),
                "max_steps": self.max_experiment_steps
            }
        )
    
    def _check_convergence(self, belief_history: List[BeliefState]) -> bool:
        """Check if beliefs have converged."""
        if len(belief_history) < self.convergence_window:
            return False
        
        # Check recent belief changes
        recent_beliefs = belief_history[-self.convergence_window:]
        
        total_changes = []
        for i in range(1, len(recent_beliefs)):
            prev = recent_beliefs[i-1].causal_beliefs
            curr = recent_beliefs[i].causal_beliefs
            
            total_change = sum(abs(curr.get(k, 0) - prev.get(k, 0))
                             for k in set(prev.keys()) | set(curr.keys()))
            total_changes.append(total_change)
        
        avg_change = sum(total_changes) / len(total_changes) if total_changes else 0
        return avg_change < self.convergence_threshold
    
    def _check_no_progress(self, belief_history: List[BeliefState]) -> bool:
        """Check if experiment is making no progress."""
        if len(belief_history) < 10:
            return False
        
        # Check if beliefs are essentially unchanged for last 10 steps
        recent_beliefs = belief_history[-10:]
        first_recent = recent_beliefs[0].causal_beliefs
        last_recent = recent_beliefs[-1].causal_beliefs
        
        total_change = sum(abs(last_recent.get(k, 0) - first_recent.get(k, 0))
                          for k in set(first_recent.keys()) | set(last_recent.keys()))
        
        return total_change < 0.01  # Very small change over 10 steps


class DataTransformationRule(BusinessRule):
    """Transforms or normalizes experiment data."""
    
    def __init__(self):
        super().__init__(
            rule_id="data_transformation",
            rule_type=RuleType.TRANSFORMATION,
            priority=RulePriority.LOW,
            description="Transforms and normalizes experiment data"
        )
    
    def evaluate(self, context: RuleContext) -> bool:
        """Evaluate if transformation is needed."""
        # Check if any beliefs are outside normal range
        for belief_state in context.belief_history:
            for value in belief_state.causal_beliefs.values():
                if value < -1 or value > 1:
                    return True
        return False
    
    def execute(self, context: RuleContext) -> RuleResult:
        """Execute data transformation."""
        self.execution_count += 1
        self.last_executed = datetime.now()
        
        transformations_applied = []
        
        # Normalize belief values to [-1, 1] range
        for belief_state in context.belief_history:
            for relationship, value in belief_state.causal_beliefs.items():
                if value < -1 or value > 1:
                    # Clamp to valid range
                    original_value = value
                    normalized_value = max(-1, min(1, value))
                    belief_state.causal_beliefs[relationship] = normalized_value
                    
                    transformations_applied.append({
                        "relationship": relationship,
                        "original_value": original_value,
                        "normalized_value": normalized_value
                    })
        
        return RuleResult(
            rule_id=self.rule_id,
            success=True,
            message=f"Applied {len(transformations_applied)} data transformations",
            action_taken="data_normalization",
            modifications={
                "transformations": transformations_applied
            },
            metadata={
                "transformation_count": len(transformations_applied)
            }
        )


class BusinessRulesEngine:
    """Central business rules engine for managing and executing rules."""
    
    def __init__(self):
        """Initialize business rules engine."""
        self.rules: Dict[str, BusinessRule] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default business rules."""
        default_rules = [
            ExperimentValidationRule(),
            SafetyConstraintRule(),
            BeliefConsistencyRule(),
            ScoringAdjustmentRule(),
            ExperimentTerminationRule(),
            DataTransformationRule()
        ]
        
        for rule in default_rules:
            self.register_rule(rule)
    
    def register_rule(self, rule: BusinessRule):
        """Register a business rule.
        
        Args:
            rule: Business rule to register
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Registered business rule: {rule.rule_id}")
    
    def unregister_rule(self, rule_id: str):
        """Unregister a business rule.
        
        Args:
            rule_id: ID of rule to unregister
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Unregistered business rule: {rule_id}")
    
    def enable_rule(self, rule_id: str):
        """Enable a business rule.
        
        Args:
            rule_id: ID of rule to enable
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """Disable a business rule.
        
        Args:
            rule_id: ID of rule to disable
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
    
    def execute_rules(self, context: RuleContext) -> Dict[str, Any]:
        """Execute all applicable business rules.
        
        Args:
            context: Rule execution context
            
        Returns:
            Execution results
        """
        start_time = datetime.now()
        
        # Sort rules by priority
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.priority.value
        )
        
        results = []
        critical_failures = []
        total_modifications = {}
        
        for rule in sorted_rules:
            if rule.can_execute(context):
                try:
                    result = rule.execute(context)
                    results.append(result)
                    
                    # Track modifications
                    if result.modifications:
                        total_modifications[rule.rule_id] = result.modifications
                    
                    # Check for critical failures
                    if not result.success and rule.priority == RulePriority.CRITICAL:
                        critical_failures.append(result)
                    
                    logger.debug(f"Executed rule {rule.rule_id}: {result.message}")
                    
                except Exception as e:
                    logger.error(f"Error executing rule {rule.rule_id}: {e}")
                    results.append(RuleResult(
                        rule_id=rule.rule_id,
                        success=False,
                        message=f"Execution error: {str(e)}",
                        action_taken="error"
                    ))
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record execution
        execution_record = {
            "timestamp": start_time.isoformat(),
            "experiment_id": context.experiment_id,
            "rules_executed": len(results),
            "critical_failures": len(critical_failures),
            "execution_time": execution_time,
            "results": [
                {
                    "rule_id": r.rule_id,
                    "success": r.success,
                    "message": r.message,
                    "action_taken": r.action_taken
                }
                for r in results
            ]
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        return {
            "timestamp": start_time.isoformat(),
            "execution_time": execution_time,
            "rules_executed": len(results),
            "successful_rules": sum(1 for r in results if r.success),
            "failed_rules": sum(1 for r in results if not r.success),
            "critical_failures": critical_failures,
            "results": results,
            "modifications": total_modifications,
            "overall_success": len(critical_failures) == 0
        }
    
    def get_rule_status(self) -> Dict[str, Any]:
        """Get status of all registered rules.
        
        Returns:
            Rule status information
        """
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "disabled_rules": sum(1 for r in self.rules.values() if not r.enabled),
            "rules_by_type": {
                rule_type.value: sum(1 for r in self.rules.values() if r.rule_type == rule_type)
                for rule_type in RuleType
            },
            "rules_by_priority": {
                priority.name: sum(1 for r in self.rules.values() if r.priority == priority)
                for priority in RulePriority
            },
            "rule_details": [
                {
                    "rule_id": rule.rule_id,
                    "type": rule.rule_type.value,
                    "priority": rule.priority.name,
                    "enabled": rule.enabled,
                    "execution_count": rule.execution_count,
                    "last_executed": rule.last_executed.isoformat() if rule.last_executed else None,
                    "description": rule.description
                }
                for rule in self.rules.values()
            ]
        }
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent rule execution history.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            Execution history
        """
        return self.execution_history[-limit:]
    
    def create_custom_rule(self, rule_config: Dict[str, Any]) -> str:
        """Create a custom rule from configuration.
        
        Args:
            rule_config: Rule configuration
            
        Returns:
            Rule ID of created rule
        """
        # This would implement custom rule creation from config
        # For now, return placeholder
        rule_id = f"custom_{len(self.rules)}"
        logger.info(f"Custom rule creation requested: {rule_id}")
        return rule_id