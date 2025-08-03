"""Causal reasoning scoring and ranking algorithms."""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math

from ..core import CausalEnvironment
from ..metrics import BeliefState

logger = logging.getLogger(__name__)


class ScoringMethod(Enum):
    """Causal scoring methods."""
    BELIEF_ACCURACY = "belief_accuracy"
    INTERVENTION_EFFICIENCY = "intervention_efficiency"
    CAUSAL_DISCOVERY = "causal_discovery"
    COMPOSITE = "composite"


@dataclass
class CausalScore:
    """Causal reasoning score with components."""
    total_score: float
    belief_accuracy: float
    intervention_efficiency: float
    causal_discovery: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformance:
    """Agent performance metrics."""
    agent_id: str
    scores: List[CausalScore]
    ranking: int
    avg_score: float
    consistency: float
    improvement_rate: float
    last_updated: datetime = field(default_factory=datetime.now)


class CausalScorer:
    """Advanced causal reasoning scoring system."""
    
    def __init__(self, scoring_method: ScoringMethod = ScoringMethod.COMPOSITE):
        """Initialize causal scorer.
        
        Args:
            scoring_method: Primary scoring method to use
        """
        self.scoring_method = scoring_method
        self.scoring_weights = {
            "belief_accuracy": 0.4,
            "intervention_efficiency": 0.3,
            "causal_discovery": 0.3
        }
        
    def score_causal_reasoning(self, 
                             environment: CausalEnvironment,
                             agent_beliefs: List[BeliefState],
                             interventions: List[Dict[str, Any]],
                             ground_truth: Dict[str, Any]) -> CausalScore:
        """Score agent's causal reasoning performance.
        
        Args:
            environment: Causal environment
            agent_beliefs: Agent's belief states over time
            interventions: Interventions performed by agent
            ground_truth: True causal structure and relationships
            
        Returns:
            Comprehensive causal score
        """
        # Calculate component scores
        belief_score = self._score_belief_accuracy(agent_beliefs, ground_truth)
        intervention_score = self._score_intervention_efficiency(
            interventions, environment, ground_truth
        )
        discovery_score = self._score_causal_discovery(agent_beliefs, ground_truth)
        
        # Calculate composite score
        if self.scoring_method == ScoringMethod.COMPOSITE:
            total_score = (
                belief_score * self.scoring_weights["belief_accuracy"] +
                intervention_score * self.scoring_weights["intervention_efficiency"] +
                discovery_score * self.scoring_weights["causal_discovery"]
            )
        elif self.scoring_method == ScoringMethod.BELIEF_ACCURACY:
            total_score = belief_score
        elif self.scoring_method == ScoringMethod.INTERVENTION_EFFICIENCY:
            total_score = intervention_score
        elif self.scoring_method == ScoringMethod.CAUSAL_DISCOVERY:
            total_score = discovery_score
        else:
            total_score = (belief_score + intervention_score + discovery_score) / 3
        
        # Calculate confidence based on consistency of beliefs
        confidence = self._calculate_confidence(agent_beliefs)
        
        return CausalScore(
            total_score=total_score,
            belief_accuracy=belief_score,
            intervention_efficiency=intervention_score,
            causal_discovery=discovery_score,
            confidence=confidence,
            metadata={
                "scoring_method": self.scoring_method.value,
                "num_beliefs": len(agent_beliefs),
                "num_interventions": len(interventions),
                "scoring_weights": self.scoring_weights
            }
        )
    
    def _score_belief_accuracy(self, beliefs: List[BeliefState], 
                              ground_truth: Dict[str, Any]) -> float:
        """Score accuracy of agent's beliefs about causal relationships.
        
        Args:
            beliefs: Agent's belief states
            ground_truth: True causal relationships
            
        Returns:
            Belief accuracy score (0-1)
        """
        if not beliefs:
            return 0.0
        
        latest_beliefs = beliefs[-1]
        true_edges = ground_truth.get("causal_edges", {})
        
        total_score = 0.0
        num_relationships = 0
        
        # Score causal relationship beliefs
        for relationship, true_strength in true_edges.items():
            if relationship in latest_beliefs.causal_beliefs:
                believed_strength = latest_beliefs.causal_beliefs[relationship]
                
                # Use negative squared error for scoring
                error = abs(believed_strength - true_strength)
                score = max(0, 1 - error)
                total_score += score
                num_relationships += 1
        
        # Penalize for missing relationships
        for relationship in true_edges:
            if relationship not in latest_beliefs.causal_beliefs:
                num_relationships += 1
        
        return total_score / max(num_relationships, 1)
    
    def _score_intervention_efficiency(self, interventions: List[Dict[str, Any]],
                                     environment: CausalEnvironment,
                                     ground_truth: Dict[str, Any]) -> float:
        """Score efficiency of interventions performed.
        
        Args:
            interventions: List of interventions
            environment: Causal environment
            ground_truth: True causal structure
            
        Returns:
            Intervention efficiency score (0-1)
        """
        if not interventions:
            return 0.0
        
        total_info_gain = 0.0
        total_cost = 0.0
        
        for intervention in interventions:
            # Calculate information gain from intervention
            info_gain = self._calculate_information_gain(intervention, ground_truth)
            total_info_gain += info_gain
            
            # Calculate intervention cost (complexity, resources)
            cost = self._calculate_intervention_cost(intervention)
            total_cost += cost
        
        # Efficiency = information gained per unit cost
        efficiency = total_info_gain / max(total_cost, 0.1)
        
        # Normalize to 0-1 range
        return min(1.0, efficiency / len(interventions))
    
    def _score_causal_discovery(self, beliefs: List[BeliefState],
                               ground_truth: Dict[str, Any]) -> float:
        """Score agent's causal discovery capabilities.
        
        Args:
            beliefs: Agent's belief evolution
            ground_truth: True causal structure
            
        Returns:
            Causal discovery score (0-1)
        """
        if len(beliefs) < 2:
            return 0.0
        
        true_structure = ground_truth.get("causal_graph", {})
        discovered_edges = set()
        
        # Track discovery of causal relationships over time
        for i, belief_state in enumerate(beliefs):
            for relationship, strength in belief_state.causal_beliefs.items():
                if strength > 0.7:  # High confidence threshold
                    discovered_edges.add(relationship)
        
        # Calculate precision and recall
        true_edges = set(true_structure.keys())
        
        if not discovered_edges:
            return 0.0
        
        true_positives = len(discovered_edges.intersection(true_edges))
        precision = true_positives / len(discovered_edges)
        recall = true_positives / max(len(true_edges), 1)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Bonus for discovery speed (earlier discovery = higher score)
        speed_bonus = self._calculate_discovery_speed_bonus(beliefs, true_edges)
        
        return min(1.0, f1_score + speed_bonus)
    
    def _calculate_confidence(self, beliefs: List[BeliefState]) -> float:
        """Calculate confidence based on belief consistency.
        
        Args:
            beliefs: Agent's belief states over time
            
        Returns:
            Confidence score (0-1)
        """
        if len(beliefs) < 2:
            return 0.5
        
        # Calculate variance in beliefs over time
        belief_variances = []
        
        # Track variance for each causal relationship
        all_relationships = set()
        for belief in beliefs:
            all_relationships.update(belief.causal_beliefs.keys())
        
        for relationship in all_relationships:
            values = []
            for belief in beliefs:
                if relationship in belief.causal_beliefs:
                    values.append(belief.causal_beliefs[relationship])
            
            if len(values) > 1:
                variance = np.var(values)
                belief_variances.append(variance)
        
        if not belief_variances:
            return 0.5
        
        # Lower variance = higher confidence
        avg_variance = np.mean(belief_variances)
        confidence = max(0, 1 - avg_variance)
        
        return confidence
    
    def _calculate_information_gain(self, intervention: Dict[str, Any],
                                   ground_truth: Dict[str, Any]) -> float:
        """Calculate information gain from an intervention.
        
        Args:
            intervention: Intervention details
            ground_truth: True causal structure
            
        Returns:
            Information gain value
        """
        # Simple heuristic: interventions on more central nodes provide more info
        target_variable = intervention.get("variable")
        causal_graph = ground_truth.get("causal_graph", {})
        
        # Count connections (centrality measure)
        connections = 0
        for edge, strength in causal_graph.items():
            if target_variable in edge:
                connections += strength
        
        # Information gain proportional to centrality
        return min(1.0, connections / 5.0)  # Normalize
    
    def _calculate_intervention_cost(self, intervention: Dict[str, Any]) -> float:
        """Calculate cost of performing an intervention.
        
        Args:
            intervention: Intervention details
            
        Returns:
            Cost value
        """
        # Simple cost model based on intervention complexity
        base_cost = 1.0
        
        # Higher cost for extreme interventions
        intervention_value = intervention.get("value", 0)
        if abs(intervention_value) > 2:  # Standard deviations from mean
            base_cost *= 1.5
        
        # Cost increases with intervention scope
        scope = intervention.get("scope", "single")
        if scope == "multiple":
            base_cost *= 2.0
        
        return base_cost
    
    def _calculate_discovery_speed_bonus(self, beliefs: List[BeliefState],
                                        true_edges: set) -> float:
        """Calculate bonus for discovering relationships quickly.
        
        Args:
            beliefs: Belief evolution
            true_edges: True causal relationships
            
        Returns:
            Speed bonus (0-0.2)
        """
        discovery_times = []
        
        for true_edge in true_edges:
            for i, belief in enumerate(beliefs):
                if (true_edge in belief.causal_beliefs and 
                    belief.causal_beliefs[true_edge] > 0.7):
                    discovery_times.append(i)
                    break
        
        if not discovery_times:
            return 0.0
        
        # Earlier discovery = higher bonus
        avg_discovery_time = np.mean(discovery_times)
        max_time = len(beliefs) - 1
        
        if max_time == 0:
            return 0.0
        
        speed_score = 1 - (avg_discovery_time / max_time)
        return speed_score * 0.2  # Max 0.2 bonus


class AgentRanking:
    """Agent performance ranking system."""
    
    def __init__(self, decay_factor: float = 0.95):
        """Initialize ranking system.
        
        Args:
            decay_factor: Factor for weighting recent performance
        """
        self.decay_factor = decay_factor
        self.agent_performances: Dict[str, AgentPerformance] = {}
    
    def update_agent_score(self, agent_id: str, score: CausalScore) -> None:
        """Update agent's performance with new score.
        
        Args:
            agent_id: Agent identifier
            score: New causal score
        """
        if agent_id not in self.agent_performances:
            self.agent_performances[agent_id] = AgentPerformance(
                agent_id=agent_id,
                scores=[],
                ranking=0,
                avg_score=0.0,
                consistency=0.0,
                improvement_rate=0.0
            )
        
        performance = self.agent_performances[agent_id]
        performance.scores.append(score)
        
        # Keep only last 100 scores
        if len(performance.scores) > 100:
            performance.scores = performance.scores[-100:]
        
        # Update metrics
        self._update_performance_metrics(performance)
        
        # Recalculate rankings
        self._update_rankings()
    
    def _update_performance_metrics(self, performance: AgentPerformance) -> None:
        """Update performance metrics for an agent.
        
        Args:
            performance: Agent performance object
        """
        scores = [s.total_score for s in performance.scores]
        
        if not scores:
            return
        
        # Calculate weighted average (recent scores weighted more)
        weights = [self.decay_factor ** (len(scores) - i - 1) 
                  for i in range(len(scores))]
        weighted_avg = np.average(scores, weights=weights)
        performance.avg_score = weighted_avg
        
        # Calculate consistency (1 - variance)
        if len(scores) > 1:
            performance.consistency = max(0, 1 - np.var(scores))
        else:
            performance.consistency = 1.0
        
        # Calculate improvement rate (slope of recent trend)
        if len(scores) >= 5:
            recent_scores = scores[-10:]  # Last 10 scores
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            performance.improvement_rate = slope
        else:
            performance.improvement_rate = 0.0
        
        performance.last_updated = datetime.now()
    
    def _update_rankings(self) -> None:
        """Update agent rankings based on current performance."""
        # Sort agents by average score (descending)
        sorted_agents = sorted(
            self.agent_performances.values(),
            key=lambda p: p.avg_score,
            reverse=True
        )
        
        # Assign rankings
        for i, performance in enumerate(sorted_agents):
            performance.ranking = i + 1
    
    def get_top_agents(self, n: int = 10) -> List[AgentPerformance]:
        """Get top N performing agents.
        
        Args:
            n: Number of top agents to return
            
        Returns:
            List of top agent performances
        """
        sorted_agents = sorted(
            self.agent_performances.values(),
            key=lambda p: p.avg_score,
            reverse=True
        )
        
        return sorted_agents[:n]
    
    def get_agent_performance(self, agent_id: str) -> Optional[AgentPerformance]:
        """Get performance data for specific agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent performance or None if not found
        """
        return self.agent_performances.get(agent_id)
    
    def get_leaderboard(self) -> Dict[str, Any]:
        """Get current leaderboard.
        
        Returns:
            Leaderboard data
        """
        top_agents = self.get_top_agents(20)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(self.agent_performances),
            "top_agents": [
                {
                    "agent_id": agent.agent_id,
                    "ranking": agent.ranking,
                    "avg_score": round(agent.avg_score, 3),
                    "consistency": round(agent.consistency, 3),
                    "improvement_rate": round(agent.improvement_rate, 4),
                    "num_experiments": len(agent.scores),
                    "last_updated": agent.last_updated.isoformat()
                }
                for agent in top_agents
            ]
        }


class DynamicScoring:
    """Dynamic scoring that adapts to experiment difficulty."""
    
    def __init__(self):
        """Initialize dynamic scoring system."""
        self.base_scorer = CausalScorer()
        self.difficulty_history: List[float] = []
        self.performance_history: List[float] = []
    
    def score_with_difficulty_adjustment(self,
                                       environment: CausalEnvironment,
                                       agent_beliefs: List[BeliefState],
                                       interventions: List[Dict[str, Any]],
                                       ground_truth: Dict[str, Any]) -> CausalScore:
        """Score with difficulty adjustment.
        
        Args:
            environment: Causal environment
            agent_beliefs: Agent beliefs
            interventions: Interventions performed
            ground_truth: True causal structure
            
        Returns:
            Difficulty-adjusted causal score
        """
        # Calculate base score
        base_score = self.base_scorer.score_causal_reasoning(
            environment, agent_beliefs, interventions, ground_truth
        )
        
        # Calculate experiment difficulty
        difficulty = self._calculate_experiment_difficulty(environment, ground_truth)
        
        # Adjust score based on difficulty
        difficulty_multiplier = self._get_difficulty_multiplier(difficulty)
        adjusted_score = base_score.total_score * difficulty_multiplier
        
        # Update histories
        self.difficulty_history.append(difficulty)
        self.performance_history.append(base_score.total_score)
        
        # Keep only recent history
        if len(self.difficulty_history) > 1000:
            self.difficulty_history = self.difficulty_history[-1000:]
            self.performance_history = self.performance_history[-1000:]
        
        # Create adjusted score object
        adjusted_causal_score = CausalScore(
            total_score=adjusted_score,
            belief_accuracy=base_score.belief_accuracy,
            intervention_efficiency=base_score.intervention_efficiency,
            causal_discovery=base_score.causal_discovery,
            confidence=base_score.confidence,
            metadata={
                **base_score.metadata,
                "difficulty": difficulty,
                "difficulty_multiplier": difficulty_multiplier,
                "base_score": base_score.total_score
            }
        )
        
        return adjusted_causal_score
    
    def _calculate_experiment_difficulty(self, environment: CausalEnvironment,
                                       ground_truth: Dict[str, Any]) -> float:
        """Calculate experiment difficulty.
        
        Args:
            environment: Causal environment
            ground_truth: True causal structure
            
        Returns:
            Difficulty score (0-1, higher = more difficult)
        """
        difficulty_factors = []
        
        # Factor 1: Number of variables
        num_variables = len(environment.variables)
        var_difficulty = min(1.0, num_variables / 20)  # Normalize to max 20 vars
        difficulty_factors.append(var_difficulty)
        
        # Factor 2: Graph complexity (number of edges)
        causal_graph = ground_truth.get("causal_graph", {})
        num_edges = len(causal_graph)
        edge_difficulty = min(1.0, num_edges / 50)  # Normalize to max 50 edges
        difficulty_factors.append(edge_difficulty)
        
        # Factor 3: Confounding complexity
        confounders = ground_truth.get("confounders", [])
        confound_difficulty = min(1.0, len(confounders) / 10)
        difficulty_factors.append(confound_difficulty)
        
        # Factor 4: Non-linear relationships
        nonlinear_count = sum(1 for edge, info in causal_graph.items() 
                            if isinstance(info, dict) and 
                            info.get("type") == "nonlinear")
        nonlinear_difficulty = min(1.0, nonlinear_count / 10)
        difficulty_factors.append(nonlinear_difficulty)
        
        # Average difficulty
        return np.mean(difficulty_factors)
    
    def _get_difficulty_multiplier(self, difficulty: float) -> float:
        """Get score multiplier based on difficulty.
        
        Args:
            difficulty: Difficulty score (0-1)
            
        Returns:
            Multiplier for score adjustment
        """
        # Higher difficulty = higher multiplier (reward for harder tasks)
        # Range: 1.0 (easy) to 2.0 (very hard)
        return 1.0 + difficulty