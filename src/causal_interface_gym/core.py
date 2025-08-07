"""Core classes for causal environments and intervention interfaces."""

from typing import Dict, List, Any, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np
import itertools
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class CausalEnvironment:
    """Environment for causal reasoning experiments."""
    
    def __init__(self, dag: Optional[Dict[str, List[str]]] = None):
        """Initialize causal environment.
        
        Args:
            dag: Dictionary representing DAG structure {node: [parents]}
        """
        self.graph = nx.DiGraph()
        self.variables: Dict[str, Any] = {}
        self.variable_types: Dict[str, str] = {}
        self.mechanisms: Dict[str, Any] = {}
        self.observational_data: Dict[str, List[float]] = {}
        
        if dag:
            self._build_from_dag(dag)
    
    def _build_from_dag(self, dag: Dict[str, List[str]]) -> None:
        """Build graph from DAG specification."""
        for node, parents in dag.items():
            self.graph.add_node(node)
            for parent in parents:
                self.graph.add_edge(parent, node)
    
    @classmethod
    def from_dag(cls, dag: Dict[str, List[str]]) -> "CausalEnvironment":
        """Create environment from DAG specification."""
        return cls(dag)
    
    def add_variable(self, name: str, var_type: str = "binary", 
                    mechanism: Optional[Any] = None) -> None:
        """Add a variable to the causal environment.
        
        Args:
            name: Variable name
            var_type: Type of variable (binary, continuous, categorical)
            mechanism: Causal mechanism function
            
        Raises:
            ValueError: If name is empty or var_type is invalid
            TypeError: If name is not a string
        """
        if not isinstance(name, str):
            raise TypeError(f"Variable name must be string, got {type(name)}")
        if not name or not name.strip():
            raise ValueError("Variable name cannot be empty")
        if var_type not in ["binary", "continuous", "categorical"]:
            raise ValueError(f"Invalid variable type: {var_type}. Must be one of: binary, continuous, categorical")
        
        name = name.strip()
        if name in self.graph.nodes():
            logger.warning(f"Variable '{name}' already exists in graph")
        
        self.graph.add_node(name)
        self.variable_types[name] = var_type
        if mechanism:
            self.mechanisms[name] = mechanism
        self.observational_data[name] = []
    
    def add_edge(self, parent: str, child: str, mechanism: Optional[Any] = None) -> None:
        """Add causal edge between variables.
        
        Args:
            parent: Parent variable name
            child: Child variable name  
            mechanism: Causal mechanism function
            
        Raises:
            ValueError: If parent or child variables don't exist or are the same
            TypeError: If parent or child are not strings
        """
        if not isinstance(parent, str) or not isinstance(child, str):
            raise TypeError("Parent and child must be strings")
        if not parent.strip() or not child.strip():
            raise ValueError("Parent and child names cannot be empty")
        if parent.strip() == child.strip():
            raise ValueError("Cannot add self-loop: parent and child are the same")
        
        parent, child = parent.strip(), child.strip()
        
        if parent not in self.graph.nodes():
            raise ValueError(f"Parent variable '{parent}' not found in graph. Add it first with add_variable()")
        if child not in self.graph.nodes():
            raise ValueError(f"Child variable '{child}' not found in graph. Add it first with add_variable()")
        
        if self.graph.has_edge(parent, child):
            logger.warning(f"Edge '{parent}' -> '{child}' already exists")
        
        self.graph.add_edge(parent, child)
        if mechanism:
            self.mechanisms[f"{parent}->{child}"] = mechanism
    
    def get_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find all backdoor paths between treatment and outcome.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            List of backdoor paths
            
        Raises:
            ValueError: If treatment or outcome variables don't exist
            TypeError: If treatment or outcome are not strings
        """
        if not isinstance(treatment, str) or not isinstance(outcome, str):
            raise TypeError("Treatment and outcome must be strings")
        if not treatment.strip() or not outcome.strip():
            raise ValueError("Treatment and outcome names cannot be empty")
            
        treatment, outcome = treatment.strip(), outcome.strip()
        
        if treatment not in self.graph.nodes():
            raise ValueError(f"Treatment variable '{treatment}' not found in graph")
        if outcome not in self.graph.nodes():
            raise ValueError(f"Outcome variable '{outcome}' not found in graph")
        if treatment == outcome:
            raise ValueError("Treatment and outcome cannot be the same variable")
        
        backdoor_paths = []
        
        # Find all undirected paths between treatment and outcome
        try:
            undirected = self.graph.to_undirected()
            all_paths = list(nx.all_simple_paths(undirected, treatment, outcome))
            
            for path in all_paths:
                if len(path) > 2:  # More than direct path
                    # Check if path starts with arrow INTO treatment
                    if len(path) > 2 and self.graph.has_edge(path[1], path[0]):
                        backdoor_paths.append(path)
        except nx.NetworkXNoPath:
            logger.debug(f"No paths found between {treatment} and {outcome}")
        except Exception as e:
            logger.error(f"Error finding backdoor paths: {e}")
            raise ValueError(f"Failed to find backdoor paths: {e}")
            
        return backdoor_paths
    
    def identify_backdoor_set(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """Identify minimal backdoor adjustment set.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Minimal backdoor adjustment set or None if none exists
        """
        backdoor_paths = self.get_backdoor_paths(treatment, outcome)
        if not backdoor_paths:
            return set()  # No backdoor paths, empty set suffices
            
        # Find all possible confounders
        all_confounders = set()
        for path in backdoor_paths:
            # Add all intermediate nodes (potential confounders)
            all_confounders.update(path[1:-1])
        
        # Remove treatment and outcome from candidates
        all_confounders.discard(treatment)
        all_confounders.discard(outcome)
        
        # Find minimal set that blocks all backdoor paths
        for r in range(len(all_confounders) + 1):
            for candidate_set in itertools.combinations(all_confounders, r):
                if self._blocks_all_backdoor_paths(set(candidate_set), treatment, outcome):
                    return set(candidate_set)
        
        return None  # No valid backdoor set found
    
    def _blocks_all_backdoor_paths(self, adjustment_set: Set[str], 
                                  treatment: str, outcome: str) -> bool:
        """Check if adjustment set blocks all backdoor paths.
        
        Args:
            adjustment_set: Variables to adjust for
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            True if all backdoor paths are blocked
        """
        backdoor_paths = self.get_backdoor_paths(treatment, outcome)
        
        for path in backdoor_paths:
            path_blocked = False
            for node in path[1:-1]:  # Intermediate nodes
                if node in adjustment_set:
                    path_blocked = True
                    break
            if not path_blocked:
                return False
        
        return True
    
    def do_calculus(self, treatment: str, outcome: str, 
                   treatment_value: Any) -> Dict[str, Any]:
        """Compute causal effect using do-calculus.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Value to set treatment to
            
        Returns:
            Causal effect and identification strategy
        """
        # Step 1: Try backdoor adjustment
        backdoor_set = self.identify_backdoor_set(treatment, outcome)
        
        if backdoor_set is not None:
            return {
                "identifiable": True,
                "strategy": "backdoor_adjustment",
                "adjustment_set": list(backdoor_set),
                "formula": f"P({outcome}|do({treatment}={treatment_value})) = Σ_{{z}} P({outcome}|{treatment}={treatment_value},Z=z) P(Z=z)",
                "causal_effect": self._compute_backdoor_effect(treatment, outcome, treatment_value, backdoor_set)
            }
        
        # Step 2: Try frontdoor adjustment (simplified)
        frontdoor_set = self._find_frontdoor_set(treatment, outcome)
        if frontdoor_set:
            return {
                "identifiable": True, 
                "strategy": "frontdoor_adjustment",
                "mediator_set": list(frontdoor_set),
                "formula": f"P({outcome}|do({treatment}={treatment_value})) = Σ_{{m}} P(M=m|{treatment}={treatment_value}) Σ_{{t}} P({outcome}|M=m,{treatment}=t) P({treatment}=t)",
                "causal_effect": self._compute_frontdoor_effect(treatment, outcome, treatment_value, frontdoor_set)
            }
        
        return {
            "identifiable": False,
            "strategy": "not_identifiable",
            "reason": "No valid backdoor or frontdoor adjustment set found"
        }
    
    def _compute_backdoor_effect(self, treatment: str, outcome: str, 
                               treatment_value: Any, adjustment_set: Set[str]) -> float:
        """Compute causal effect using backdoor adjustment.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Treatment value
            adjustment_set: Variables to adjust for
            
        Returns:
            Estimated causal effect
        """
        # Simplified simulation-based computation
        # In real implementation, this would use actual data
        
        if not adjustment_set:
            # No confounders, direct effect
            return np.random.normal(0.5, 0.1)  # Simulated effect
        
        # Simulate adjustment for confounders
        effect = 0.0
        num_strata = 10  # Discretize continuous confounders
        
        for _ in range(num_strata):
            # Simulate P(Y|X=x,Z=z) * P(Z=z)
            conditional_effect = np.random.normal(0.3, 0.2)
            stratum_weight = 1.0 / num_strata
            effect += conditional_effect * stratum_weight
        
        return float(effect)
    
    def _find_frontdoor_set(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """Find frontdoor adjustment set (mediators).
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Frontdoor adjustment set or None
        """
        # Simplified frontdoor identification
        # Look for mediators on all paths from treatment to outcome
        
        try:
            all_paths = list(nx.all_simple_paths(self.graph, treatment, outcome))
            if not all_paths:
                return None
                
            # Find nodes that are on ALL paths from treatment to outcome
            path_intersections = None
            for path in all_paths:
                path_nodes = set(path[1:-1])  # Exclude treatment and outcome
                if path_intersections is None:
                    path_intersections = path_nodes
                else:
                    path_intersections = path_intersections.intersection(path_nodes)
            
            if path_intersections:
                return path_intersections
                
        except nx.NetworkXNoPath:
            pass
            
        return None
    
    def _compute_frontdoor_effect(self, treatment: str, outcome: str,
                                treatment_value: Any, mediator_set: Set[str]) -> float:
        """Compute causal effect using frontdoor adjustment.
        
        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Treatment value
            mediator_set: Mediator variables
            
        Returns:
            Estimated causal effect
        """
        # Simplified frontdoor computation
        # Real implementation would compute:
        # Σ_m P(M=m|X=x) Σ_x' P(Y|M=m,X=x') P(X=x')
        
        return np.random.normal(0.4, 0.15)  # Simulated effect
    
    def intervene(self, **interventions: Any) -> Dict[str, Any]:
        """Apply causal interventions.
        
        Args:
            **interventions: Variable assignments for intervention
            
        Returns:
            Results after intervention with causal effects
            
        Raises:
            ValueError: If no interventions provided or invalid values
            TypeError: If intervention values are invalid types
        """
        if not interventions:
            raise ValueError("At least one intervention must be provided")
        
        results = {"interventions_applied": interventions}
        
        for treatment, value in interventions.items():
            if not isinstance(treatment, str):
                raise TypeError(f"Treatment variable name must be string, got {type(treatment)}")
            
            treatment = treatment.strip()
            if not treatment:
                raise ValueError("Treatment variable name cannot be empty")
                
            if treatment not in self.graph.nodes:
                error_msg = f"Variable {treatment} not in causal graph"
                results[f"error_{treatment}"] = error_msg
                logger.error(error_msg)
                continue
            
            # Validate intervention value based on variable type
            var_type = self.variable_types.get(treatment, "binary")
            if var_type == "binary" and value not in [True, False, 0, 1]:
                logger.warning(f"Binary variable '{treatment}' got non-binary value: {value}")
            elif var_type == "continuous" and not isinstance(value, (int, float)):
                logger.warning(f"Continuous variable '{treatment}' got non-numeric value: {value}")
                
            try:
                # Compute effects on all downstream variables
                downstream = list(nx.descendants(self.graph, treatment))
                
                for outcome in downstream:
                    try:
                        causal_analysis = self.do_calculus(treatment, outcome, value)
                        results[f"effect_{treatment}_on_{outcome}"] = causal_analysis
                    except Exception as e:
                        error_msg = f"Failed to compute causal effect {treatment} -> {outcome}: {e}"
                        results[f"error_{treatment}_on_{outcome}"] = error_msg
                        logger.error(error_msg)
                        
            except Exception as e:
                error_msg = f"Failed to compute downstream effects for {treatment}: {e}"
                results[f"error_{treatment}"] = error_msg
                logger.error(error_msg)
        
        return results
    
    def analyze_causal_reasoning(self, belief_trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality of causal reasoning from belief trajectory.
        
        Args:
            belief_trajectory: Agent's belief updates over time
            
        Returns:
            Causal reasoning analysis
        """
        analysis = {
            "causal_score": 0.0,
            "intervention_vs_observation": {},
            "confounding_detection": {},
            "belief_updates": []
        }
        
        # Analyze intervention vs observation understanding
        if "intervention_beliefs" in belief_trajectory and "observational_beliefs" in belief_trajectory:
            intervention_beliefs = belief_trajectory["intervention_beliefs"]
            observational_beliefs = belief_trajectory["observational_beliefs"]
            
            # Check if agent correctly distinguished P(Y|do(X)) from P(Y|X)
            for variable in intervention_beliefs:
                if variable in observational_beliefs:
                    intervention_prob = intervention_beliefs[variable]
                    observational_prob = observational_beliefs[variable]
                    
                    # Score based on appropriate difference
                    expected_difference = self._get_expected_difference(variable)
                    actual_difference = abs(intervention_prob - observational_prob)
                    
                    score = min(1.0, actual_difference / max(expected_difference, 0.1))
                    analysis["intervention_vs_observation"][variable] = {
                        "score": score,
                        "intervention_belief": intervention_prob,
                        "observational_belief": observational_prob,
                        "difference": actual_difference
                    }
        
        # Compute overall causal reasoning score
        if analysis["intervention_vs_observation"]:
            scores = [item["score"] for item in analysis["intervention_vs_observation"].values()]
            analysis["causal_score"] = np.mean(scores)
        
        return analysis
    
    def _get_expected_difference(self, belief_expression: str) -> float:
        """Get expected difference between P(Y|do(X)) and P(Y|X) for variable.
        
        Args:
            belief_expression: Belief expression like "P(wet_grass|do(sprinkler=on))"
            
        Returns:
            Expected difference (0 if no confounding)
        """
        # Extract variable name from belief expression
        variable = self._extract_variable_from_belief(belief_expression)
        
        # Simplified: assume difference exists if there are potential confounders
        if variable in self.graph.nodes():
            parents = list(self.graph.predecessors(variable))
            if len(parents) > 1:  # Multiple parents suggest potential confounding
                return 0.2  # Expected moderate difference
        return 0.05  # Small expected difference
    
    def _extract_variable_from_belief(self, belief_expression: str) -> str:
        """Extract main variable from belief expression.
        
        Args:
            belief_expression: Expression like "P(wet_grass|do(sprinkler=on))"
            
        Returns:
            Variable name like "wet_grass"
        """
        import re
        # Match P(variable|...) or P(variable)
        match = re.match(r'P\(([^|,)]+)', belief_expression)
        if match:
            return match.group(1).strip()
        # Fallback: return the expression as-is
        return belief_expression


class InterventionUI:
    """UI builder for causal intervention interfaces."""
    
    def __init__(self, environment: CausalEnvironment):
        """Initialize intervention UI.
        
        Args:
            environment: Causal environment to interface with
        """
        self.environment = environment
        self.components: List[Dict[str, Any]] = []
        self.experiment_log: List[Dict[str, Any]] = []
        self.current_beliefs: Dict[str, float] = {}
        self.intervention_history: List[Dict[str, Any]] = []
    
    def add_intervention_button(self, variable: str, label: str) -> None:
        """Add intervention button for a variable.
        
        Args:
            variable: Variable name to intervene on
            label: Button label text
        """
        self.components.append({
            "type": "button",
            "variable": variable,
            "label": label
        })
    
    def add_observation_panel(self, variable: str, label: str) -> None:
        """Add observation panel for a variable.
        
        Args:
            variable: Variable name to observe
            label: Panel label text
        """
        self.components.append({
            "type": "panel", 
            "variable": variable,
            "label": label
        })
    
    def add_belief_display(self, beliefs: List[str], comparison_mode: str = "intervention_vs_observation") -> None:
        """Add belief display component.
        
        Args:
            beliefs: List of belief statements to display
            comparison_mode: How to compare beliefs
        """
        self.components.append({
            "type": "belief_display",
            "beliefs": beliefs,
            "comparison_mode": comparison_mode
        })
    
    def add_graph_visualization(self, layout: str = "hierarchical", 
                              show_backdoors: bool = True) -> None:
        """Add causal graph visualization.
        
        Args:
            layout: Graph layout algorithm
            show_backdoors: Whether to highlight backdoor paths
        """
        self.components.append({
            "type": "graph_viz",
            "layout": layout,
            "show_backdoors": show_backdoors,
            "graph": self.environment.graph
        })
    
    def generate_html(self) -> str:
        """Generate standalone HTML interface.
        
        Returns:
            HTML string for the interface
        """
        html_components = []
        
        for component in self.components:
            if component["type"] == "button":
                html_components.append(
                    f'<button onclick="intervene(\'{component["variable"]}\')">{component["label"]}</button>'
                )
            elif component["type"] == "panel":
                html_components.append(
                    f'<div class="observation-panel"><h3>{component["label"]}</h3><div id="{component["variable"]}-value"></div></div>'
                )
            elif component["type"] == "graph_viz":
                html_components.append(
                    '<div id="causal-graph" class="graph-container"></div>'
                )
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Causal Interface Experiment</title>
            <style>
                .intervention-ui {{ padding: 20px; font-family: Arial, sans-serif; }}
                .observation-panel {{ border: 1px solid #ccc; padding: 10px; margin: 10px; }}
                .graph-container {{ width: 100%; height: 400px; border: 1px solid #ddd; }}
                button {{ padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; cursor: pointer; }}
                button:hover {{ background: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="intervention-ui">
                <h1>Causal Reasoning Experiment</h1>
                {''.join(html_components)}
            </div>
            <script>
                function intervene(variable) {{
                    console.log('Intervening on:', variable);
                    // In real implementation, this would trigger intervention
                }}
            </script>
        </body>
        </html>
        """
    
    def run_experiment(self, agent: Any, interventions: List[Tuple[str, Any]], 
                      measure_beliefs: List[str]) -> Dict[str, Any]:
        """Run causal reasoning experiment.
        
        Args:
            agent: LLM agent to test
            interventions: List of (variable, value) interventions
            measure_beliefs: List of beliefs to measure
            
        Returns:
            Experiment results with causal analysis
            
        Raises:
            ValueError: If inputs are invalid
            TypeError: If inputs have wrong types
        """
        if agent is None:
            raise ValueError("Agent cannot be None")
        if not isinstance(interventions, list):
            raise TypeError("Interventions must be a list")
        if not isinstance(measure_beliefs, list):
            raise TypeError("Measure beliefs must be a list")
        if not interventions:
            raise ValueError("At least one intervention must be provided")
        if not measure_beliefs:
            raise ValueError("At least one belief must be measured")
        
        # Validate interventions format
        for i, intervention in enumerate(interventions):
            if not isinstance(intervention, (tuple, list)) or len(intervention) != 2:
                raise ValueError(f"Intervention {i} must be a tuple/list of (variable, value)")
            variable, value = intervention
            if not isinstance(variable, str) or not variable.strip():
                raise ValueError(f"Intervention {i} variable must be a non-empty string")
        
        experiment_id = len(self.experiment_log)
        logger.info(f"Starting experiment {experiment_id} with agent {type(agent).__name__}")
        
        try:
            # Record initial observational beliefs
            initial_beliefs = self._query_agent_beliefs(agent, measure_beliefs, "observational")
            
            # Apply interventions and record belief updates
            intervention_results = []
            
            for variable, value in interventions:
                try:
                    # Apply intervention in environment
                    env_result = self.environment.intervene(**{variable: value})
                    
                    # Query agent's post-intervention beliefs
                    post_beliefs = self._query_agent_beliefs(agent, measure_beliefs, f"do({variable}={value})")
                    
                    intervention_results.append({
                        "intervention": (variable, value),
                        "environment_result": env_result,
                        "agent_beliefs": post_beliefs
                    })
                    
                    self.intervention_history.append({
                        "variable": variable,
                        "value": value,
                        "timestamp": len(self.intervention_history)
                    })
                    
                except Exception as e:
                    error_msg = f"Failed to apply intervention ({variable}, {value}): {e}"
                    logger.error(error_msg)
                    intervention_results.append({
                        "intervention": (variable, value),
                        "error": error_msg
                    })
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            raise ValueError(f"Experiment failed: {e}")
        
        # Analyze causal reasoning quality
        belief_trajectory = {
            "observational_beliefs": initial_beliefs,
            "intervention_beliefs": {}
        }
        
        for result in intervention_results:
            belief_trajectory["intervention_beliefs"].update(result["agent_beliefs"])
        
        causal_analysis = self.environment.analyze_causal_reasoning(belief_trajectory)
        
        experiment_result = {
            "experiment_id": experiment_id,
            "agent": type(agent).__name__ if hasattr(agent, '__class__') else str(agent),
            "interventions": interventions,
            "initial_beliefs": initial_beliefs,
            "intervention_results": intervention_results,
            "causal_analysis": causal_analysis,
            "measured_beliefs": measure_beliefs
        }
        
        self.experiment_log.append(experiment_result)
        
        return experiment_result
    
    def _query_agent_beliefs(self, agent: Any, beliefs: List[str], condition: str) -> Dict[str, float]:
        """Query agent for belief probabilities.
        
        Args:
            agent: Agent to query
            beliefs: List of belief statements
            condition: Condition type (observational or intervention)
            
        Returns:
            Dictionary of belief probabilities
        """
        belief_probs = {}
        
        for belief in beliefs:
            if hasattr(agent, 'query_belief'):
                # If agent has belief query method
                prob = agent.query_belief(belief, condition)
            elif hasattr(agent, 'predict_proba'):
                # If agent is sklearn-like
                prob = agent.predict_proba([belief])[0]
            else:
                # Simulate belief query for testing
                # In real implementation, this would query the LLM
                prob = np.random.random()  # Placeholder
                
                # Add some realistic variation based on condition
                if "intervention" in condition.lower() or "do(" in condition:
                    # Interventional beliefs might be different
                    prob = np.clip(prob + np.random.normal(0, 0.1), 0, 1)
            
            belief_probs[belief] = float(prob)
        
        return belief_probs
    
    def export_results(self, format: str = "json") -> str:
        """Export experiment results.
        
        Args:
            format: Export format (json, csv, paper)
            
        Returns:
            Formatted results string
        """
        if format == "json":
            import json
            return json.dumps(self.experiment_log, indent=2)
        elif format == "paper":
            # Generate paper-ready summary
            if not self.experiment_log:
                return "No experiments conducted yet."
            
            latest = self.experiment_log[-1]
            causal_score = latest["causal_analysis"].get("causal_score", 0)
            
            return f"""
            Causal Reasoning Experiment Results
            ===================================
            
            Agent: {latest['agent']}
            Interventions: {len(latest['interventions'])}
            Causal Score: {causal_score:.3f}
            
            Intervention vs Observation Understanding:
            {self._format_intervention_analysis(latest['causal_analysis'])}
            """
        
        return str(self.experiment_log)
    
    def _format_intervention_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format intervention vs observation analysis for display.
        
        Args:
            analysis: Causal analysis results
            
        Returns:
            Formatted analysis string
        """
        if "intervention_vs_observation" not in analysis:
            return "No intervention analysis available."
        
        lines = []
        for variable, data in analysis["intervention_vs_observation"].items():
            lines.append(f"  {variable}: Score {data['score']:.3f} (Δ={data['difference']:.3f})")
        
        return "\n".join(lines) if lines else "No variables analyzed."