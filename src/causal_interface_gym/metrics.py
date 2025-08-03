"""Metrics for evaluating causal reasoning capabilities."""

from typing import Dict, List, Any, Optional, Set
import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict


class CausalMetrics:
    """Metrics for measuring causal reasoning performance."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def intervention_test(self, agent_responses: List[Dict[str, Any]], 
                         ground_truth: List[Dict[str, Any]]) -> float:
        """Test understanding of interventions vs observations.
        
        Args:
            agent_responses: Agent's belief updates
            ground_truth: Correct belief updates
            
        Returns:
            Score between 0 and 1
        """
        if not agent_responses or not ground_truth:
            return 0.0
        
        scores = []
        
        for agent_resp, true_resp in zip(agent_responses, ground_truth):
            # Compare P(Y|do(X)) vs P(Y|X) understanding
            if "intervention_belief" in agent_resp and "observational_belief" in agent_resp:
                agent_intervention = agent_resp["intervention_belief"]
                agent_observational = agent_resp["observational_belief"]
                
                true_intervention = true_resp.get("intervention_belief", 0.5)
                true_observational = true_resp.get("observational_belief", 0.5)
                
                # Score based on whether agent correctly distinguishes the two
                agent_diff = abs(agent_intervention - agent_observational)
                true_diff = abs(true_intervention - true_observational)
                
                if true_diff < 0.05:  # No real difference expected
                    score = 1.0 if agent_diff < 0.1 else max(0.0, 1.0 - agent_diff / 0.5)
                else:  # Difference expected
                    score = min(1.0, agent_diff / true_diff) if true_diff > 0 else 0.0
                
                scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def backdoor_test(self, agent_graph: Any, true_graph: Any) -> float:
        """Test backdoor path identification.
        
        Args:
            agent_graph: Agent's inferred causal graph
            true_graph: True causal graph
            
        Returns:
            Score between 0 and 1
        """
        if not hasattr(agent_graph, 'nodes') or not hasattr(true_graph, 'nodes'):
            return 0.0
        
        scores = []
        
        # Test backdoor identification for all treatment-outcome pairs
        nodes = list(true_graph.nodes())
        
        for treatment in nodes:
            for outcome in nodes:
                if treatment != outcome:
                    # Get true backdoor paths
                    true_backdoors = self._get_backdoor_paths(true_graph, treatment, outcome)
                    
                    # Get agent's identified backdoor paths (if available)
                    if hasattr(agent_graph, 'get_backdoor_paths'):
                        agent_backdoors = agent_graph.get_backdoor_paths(treatment, outcome)
                    else:
                        agent_backdoors = self._get_backdoor_paths(agent_graph, treatment, outcome)
                    
                    # Score based on path overlap
                    if not true_backdoors and not agent_backdoors:
                        scores.append(1.0)  # Both found no backdoors
                    elif not true_backdoors:
                        scores.append(0.0 if agent_backdoors else 1.0)
                    elif not agent_backdoors:
                        scores.append(0.0)
                    else:
                        # Calculate Jaccard similarity of backdoor paths
                        true_paths = set(tuple(path) for path in true_backdoors)
                        agent_paths = set(tuple(path) for path in agent_backdoors)
                        
                        intersection = len(true_paths.intersection(agent_paths))
                        union = len(true_paths.union(agent_paths))
                        
                        score = intersection / union if union > 0 else 0.0
                        scores.append(score)
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _get_backdoor_paths(self, graph: nx.DiGraph, treatment: str, outcome: str) -> List[List[str]]:
        """Get backdoor paths between treatment and outcome.
        
        Args:
            graph: Causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            List of backdoor paths
        """
        backdoor_paths = []
        
        try:
            undirected = graph.to_undirected()
            all_paths = list(nx.all_simple_paths(undirected, treatment, outcome))
            
            for path in all_paths:
                if len(path) > 2:  # More than direct path
                    # Check if path starts with arrow INTO treatment
                    if graph.has_edge(path[1], path[0]):
                        backdoor_paths.append(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
            
        return backdoor_paths
    
    def counterfactual_test(self, agent_predictions: List[float],
                           true_counterfactuals: List[float]) -> float:
        """Test counterfactual reasoning.
        
        Args:
            agent_predictions: Agent's counterfactual predictions
            true_counterfactuals: True counterfactual values
            
        Returns:
            Score between 0 and 1
        """
        if not agent_predictions or not true_counterfactuals:
            return 0.0
        
        if len(agent_predictions) != len(true_counterfactuals):
            return 0.0
        
        try:
            # Correlation-based score
            correlation = np.corrcoef(agent_predictions, true_counterfactuals)[0, 1]
            if np.isnan(correlation):
                # Use MSE-based score if correlation fails
                mse = np.mean((np.array(agent_predictions) - np.array(true_counterfactuals)) ** 2)
                score = max(0.0, 1.0 - mse)  # Convert MSE to score
            else:
                score = max(0.0, correlation)  # Ensure non-negative
            
            return float(score)
        except Exception:
            return 0.0
    
    def comprehensive_evaluation(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive evaluation of causal reasoning capabilities.
        
        Args:
            experiment_data: Complete experiment results
            
        Returns:
            Comprehensive evaluation metrics
        """
        evaluation = {
            "overall_score": 0.0,
            "intervention_understanding": 0.0,
            "backdoor_identification": 0.0,
            "counterfactual_reasoning": 0.0,
            "belief_consistency": 0.0,
            "confounding_awareness": 0.0,
            "detailed_scores": {}
        }
        
        if "causal_analysis" in experiment_data:
            causal_analysis = experiment_data["causal_analysis"]
            
            # Intervention understanding score
            if "intervention_vs_observation" in causal_analysis:
                intervention_scores = []
                for var_data in causal_analysis["intervention_vs_observation"].values():
                    intervention_scores.append(var_data.get("score", 0.0))
                
                evaluation["intervention_understanding"] = np.mean(intervention_scores) if intervention_scores else 0.0
            
            # Overall causal score
            evaluation["overall_score"] = causal_analysis.get("causal_score", 0.0)
        
        # Belief consistency analysis
        if "intervention_results" in experiment_data:
            consistency_scores = []
            
            for result in experiment_data["intervention_results"]:
                if "agent_beliefs" in result:
                    beliefs = result["agent_beliefs"]
                    # Check if beliefs are in valid probability range
                    valid_probs = all(0 <= p <= 1 for p in beliefs.values())
                    consistency_scores.append(1.0 if valid_probs else 0.0)
            
            evaluation["belief_consistency"] = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Compute weighted overall score
        weights = {
            "intervention_understanding": 0.4,
            "backdoor_identification": 0.2,
            "counterfactual_reasoning": 0.2,
            "belief_consistency": 0.2
        }
        
        weighted_score = sum(
            evaluation[metric] * weight
            for metric, weight in weights.items()
            if metric in evaluation
        )
        
        evaluation["overall_score"] = max(evaluation["overall_score"], weighted_score)
        
        return evaluation


class BeliefTracker:
    """Track belief evolution during causal reasoning."""
    
    def __init__(self, agent: Any):
        """Initialize belief tracker.
        
        Args:
            agent: Agent to track beliefs for
        """
        self.agent = agent
        self.beliefs: Dict[str, List[Dict[str, Any]]] = {}
    
    def record(self, belief: str, condition: str, value: Optional[float] = None) -> None:
        """Record a belief measurement.
        
        Args:
            belief: Belief statement (e.g., "P(rain|wet_grass)")
            condition: Condition type ("observational" or "do(...)")
            value: Belief strength/probability
        """
        if belief not in self.beliefs:
            self.beliefs[belief] = []
        
        # Get actual belief value if not provided
        if value is None:
            if hasattr(self.agent, 'query_belief'):
                value = self.agent.query_belief(belief, condition)
            else:
                value = np.random.random()  # Fallback for testing
        
        # Validate probability value
        value = max(0.0, min(1.0, float(value)))
        
        self.beliefs[belief].append({
            "condition": condition,
            "value": value,
            "timestamp": len(self.beliefs[belief]),
            "belief_statement": belief
        })
    
    def plot_belief_evolution(self, variable: str, conditions: List[str], 
                            expected_change: str) -> Dict[str, Any]:
        """Plot belief evolution over time.
        
        Args:
            variable: Variable to plot beliefs for
            conditions: List of conditions to compare
            expected_change: Expected direction of change
            
        Returns:
            Plot data and analysis
        """
        plot_data = {
            "variable": variable,
            "conditions": conditions,
            "expected_change": expected_change,
            "belief_trajectories": {},
            "analysis": {}
        }
        
        # Extract belief trajectories for each condition
        for belief_name, belief_history in self.beliefs.items():
            if variable in belief_name:
                for condition in conditions:
                    condition_beliefs = [
                        entry for entry in belief_history 
                        if condition in entry["condition"]
                    ]
                    
                    if condition_beliefs:
                        plot_data["belief_trajectories"][condition] = {
                            "timestamps": [entry["timestamp"] for entry in condition_beliefs],
                            "values": [entry["value"] for entry in condition_beliefs],
                            "belief_name": belief_name
                        }
        
        # Analyze belief change patterns
        if len(plot_data["belief_trajectories"]) >= 2:
            trajectory_pairs = list(plot_data["belief_trajectories"].items())
            
            for i, (cond1, traj1) in enumerate(trajectory_pairs):
                for cond2, traj2 in trajectory_pairs[i+1:]:
                    if traj1["values"] and traj2["values"]:
                        # Compare final belief values
                        final_diff = traj1["values"][-1] - traj2["values"][-1]
                        
                        plot_data["analysis"][f"{cond1}_vs_{cond2}"] = {
                            "belief_difference": final_diff,
                            "expected_direction": expected_change,
                            "correct_direction": self._check_direction(final_diff, expected_change)
                        }
        
        return plot_data
    
    def _check_direction(self, observed_diff: float, expected_change: str) -> bool:
        """Check if observed difference matches expected direction.
        
        Args:
            observed_diff: Observed difference in beliefs
            expected_change: Expected direction (increase, decrease, none)
            
        Returns:
            True if direction matches expectation
        """
        if expected_change.lower() == "increase":
            return observed_diff > 0.05
        elif expected_change.lower() == "decrease":
            return observed_diff < -0.05
        elif expected_change.lower() == "none":
            return abs(observed_diff) < 0.1
        else:
            return True  # Unknown expectation
    
    def get_belief_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked beliefs.
        
        Returns:
            Summary of belief tracking data
        """
        summary = {
            "total_beliefs_tracked": len(self.beliefs),
            "total_measurements": sum(len(history) for history in self.beliefs.values()),
            "belief_summaries": {},
            "condition_types": set()
        }
        
        for belief_name, history in self.beliefs.items():
            if history:
                values = [entry["value"] for entry in history]
                conditions = [entry["condition"] for entry in history]
                
                summary["belief_summaries"][belief_name] = {
                    "measurements": len(history),
                    "mean_value": np.mean(values),
                    "std_value": np.std(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "conditions": list(set(conditions))
                }
                
                summary["condition_types"].update(conditions)
        
        summary["condition_types"] = list(summary["condition_types"])
        
        return summary