"""Adaptive intervention recommendation and optimization engine."""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from dataclasses import dataclass
import logging
from itertools import combinations, product
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class InterventionRecommendation:
    """Recommended intervention with supporting analysis."""
    target_variables: List[str]
    intervention_type: str  # 'set', 'randomize', 'prevent'
    values: Dict[str, Any]
    expected_effect_size: float
    confidence: float
    causal_reasoning: str
    alternative_interventions: List[Dict[str, Any]]
    cost_benefit_analysis: Dict[str, float]


@dataclass
class InterventionResult:
    """Result of an applied intervention."""
    intervention_id: str
    pre_intervention_beliefs: Dict[str, float]
    post_intervention_beliefs: Dict[str, float]
    belief_changes: Dict[str, float]
    causal_understanding_score: float
    intervention_effectiveness: float


class AdaptiveInterventionEngine:
    """Intelligent intervention recommendation system using causal analysis."""
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 exploration_weight: float = 0.2,
                 max_interventions_per_recommendation: int = 3):
        """Initialize adaptive intervention engine.
        
        Args:
            learning_rate: Rate for updating intervention preferences
            exploration_weight: Weight for exploration vs exploitation
            max_interventions_per_recommendation: Max interventions to suggest
        """
        self.learning_rate = learning_rate
        self.exploration_weight = exploration_weight
        self.max_interventions = max_interventions_per_recommendation
        self.intervention_history: List[InterventionResult] = []
        self.effectiveness_models: Dict[str, Any] = {}
        self.variable_importance_scores: Dict[str, float] = {}
        
    def recommend_interventions(self,
                              causal_graph: nx.DiGraph,
                              target_outcome: str,
                              current_beliefs: Dict[str, float],
                              agent_profile: Dict[str, Any],
                              constraints: Optional[Dict[str, Any]] = None) -> List[InterventionRecommendation]:
        """Recommend optimal interventions for causal learning.
        
        Args:
            causal_graph: Current causal graph
            target_outcome: Variable we want to understand causally
            current_beliefs: Current belief state of the agent
            agent_profile: Agent characteristics and history
            constraints: Optional constraints on interventions
            
        Returns:
            List of ranked intervention recommendations
        """
        logger.info(f"Generating intervention recommendations for target: {target_outcome}")
        
        # Analyze causal structure for intervention opportunities
        intervention_candidates = self._identify_intervention_candidates(
            causal_graph, target_outcome, constraints)
            
        # Score each candidate intervention
        scored_interventions = []
        
        for candidate in intervention_candidates:
            # Compute expected information gain
            info_gain = self._compute_information_gain(
                candidate, causal_graph, current_beliefs, agent_profile)
                
            # Estimate effect size
            effect_size = self._estimate_effect_size(
                candidate, causal_graph, target_outcome)
                
            # Compute intervention difficulty/cost
            cost = self._compute_intervention_cost(candidate, causal_graph)
            
            # Generate causal reasoning explanation
            reasoning = self._generate_causal_reasoning(
                candidate, causal_graph, target_outcome)
                
            # Create recommendation
            recommendation = InterventionRecommendation(
                target_variables=candidate['variables'],
                intervention_type=candidate['type'],
                values=candidate['values'],
                expected_effect_size=effect_size,
                confidence=info_gain / (1 + cost),  # Utility-adjusted confidence
                causal_reasoning=reasoning,
                alternative_interventions=self._generate_alternatives(candidate),
                cost_benefit_analysis={
                    'information_gain': info_gain,
                    'effect_size': effect_size,
                    'cost': cost,
                    'utility': info_gain - 0.1 * cost
                }
            )
            
            scored_interventions.append((recommendation.cost_benefit_analysis['utility'], recommendation))
            
        # Sort by utility and return top recommendations
        scored_interventions.sort(key=lambda x: x[0], reverse=True)
        
        recommendations = [rec for _, rec in scored_interventions[:self.max_interventions]]
        
        logger.info(f"Generated {len(recommendations)} intervention recommendations")
        return recommendations
        
    def _identify_intervention_candidates(self,
                                       causal_graph: nx.DiGraph,
                                       target_outcome: str,
                                       constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Identify potential intervention points in the causal graph."""
        candidates = []
        constraints = constraints or {}
        
        # Type 1: Direct causes of target (backdoor blocking)
        direct_causes = list(causal_graph.predecessors(target_outcome))
        for cause in direct_causes:
            if self._is_interventible(cause, constraints):
                candidates.append({
                    'variables': [cause],
                    'type': 'set',
                    'values': {cause: self._suggest_intervention_value(cause, causal_graph)},
                    'mechanism': 'direct_cause',
                    'rationale': f'Direct intervention on {cause} to measure direct causal effect on {target_outcome}'
                })
                
        # Type 2: Confounders (backdoor path blocking)
        confounders = self._identify_confounders(causal_graph, target_outcome)
        for confounder in confounders:
            if self._is_interventible(confounder, constraints):
                candidates.append({
                    'variables': [confounder],
                    'type': 'randomize',
                    'values': {confounder: 'randomize'},
                    'mechanism': 'confounder_control',
                    'rationale': f'Control confounding by randomizing {confounder}'
                })
                
        # Type 3: Mediators (pathway analysis)
        mediators = self._identify_mediators(causal_graph, target_outcome)
        for mediator in mediators:
            if self._is_interventible(mediator, constraints):
                candidates.append({
                    'variables': [mediator],
                    'type': 'prevent',
                    'values': {mediator: 'block'},
                    'mechanism': 'mediation_analysis',
                    'rationale': f'Block mediation pathway through {mediator} to isolate direct effects'
                })
                
        # Type 4: Multi-variable interventions (for complex confounding)
        if len(direct_causes) > 1:
            for cause_pair in combinations(direct_causes[:3], 2):  # Limit complexity
                if all(self._is_interventible(c, constraints) for c in cause_pair):
                    candidates.append({
                        'variables': list(cause_pair),
                        'type': 'set',
                        'values': {c: self._suggest_intervention_value(c, causal_graph) for c in cause_pair},
                        'mechanism': 'joint_intervention',
                        'rationale': f'Joint intervention on {cause_pair} to control multiple confounding paths'
                    })
                    
        # Type 5: Instrumental variable interventions
        instruments = self._identify_instrumental_variables(causal_graph, target_outcome)
        for instrument in instruments:
            if self._is_interventible(instrument, constraints):
                candidates.append({
                    'variables': [instrument],
                    'type': 'set', 
                    'values': {instrument: self._suggest_intervention_value(instrument, causal_graph)},
                    'mechanism': 'instrumental_variable',
                    'rationale': f'Use {instrument} as instrumental variable for causal identification'
                })
                
        return candidates
        
    def _identify_confounders(self, graph: nx.DiGraph, target: str) -> List[str]:
        """Identify potential confounding variables."""
        confounders = []
        
        # Find variables that have paths to target through multiple routes
        for node in graph.nodes():
            if node == target:
                continue
                
            # Check if node has both direct and indirect paths to target
            if graph.has_edge(node, target):
                # Look for indirect paths
                graph_copy = graph.copy()
                graph_copy.remove_edge(node, target)
                
                try:
                    if nx.has_path(graph_copy, node, target):
                        confounders.append(node)
                except nx.NetworkXNoPath:
                    pass
                    
        return confounders
        
    def _identify_mediators(self, graph: nx.DiGraph, target: str) -> List[str]:
        """Identify mediating variables."""
        mediators = []
        
        # Find nodes on causal paths to target
        for node in graph.nodes():
            if node == target:
                continue
                
            # Check if node mediates effects to target
            if graph.has_edge(node, target):
                # Check if there are causes of this node that don't directly cause target
                causes = list(graph.predecessors(node))
                for cause in causes:
                    if not graph.has_edge(cause, target):
                        mediators.append(node)
                        break
                        
        return list(set(mediators))
        
    def _identify_instrumental_variables(self, graph: nx.DiGraph, target: str) -> List[str]:
        """Identify potential instrumental variables."""
        instruments = []
        
        # Find variables that affect causes of target but not target directly
        causes = list(graph.predecessors(target))
        
        for node in graph.nodes():
            if node == target or node in causes:
                continue
                
            # Check if node affects causes of target but not target directly
            affects_causes = any(nx.has_path(graph, node, cause) for cause in causes)
            affects_target_directly = graph.has_edge(node, target)
            
            if affects_causes and not affects_target_directly:
                instruments.append(node)
                
        return instruments
        
    def _is_interventible(self, variable: str, constraints: Dict[str, Any]) -> bool:
        """Check if a variable can be intervened upon."""
        if not constraints:
            return True
            
        forbidden = constraints.get('forbidden_interventions', [])
        return variable not in forbidden
        
    def _suggest_intervention_value(self, variable: str, graph: nx.DiGraph) -> Any:
        """Suggest an appropriate intervention value for a variable."""
        # For now, return simple binary intervention values
        # In practice, this would be more sophisticated based on variable type
        return True
        
    def _compute_information_gain(self,
                                candidate: Dict[str, Any],
                                graph: nx.DiGraph,
                                current_beliefs: Dict[str, float],
                                agent_profile: Dict[str, Any]) -> float:
        """Compute expected information gain from intervention."""
        
        # Base information gain from intervention type
        mechanism_gains = {
            'direct_cause': 0.8,
            'confounder_control': 0.6,
            'mediation_analysis': 0.5,
            'joint_intervention': 0.7,
            'instrumental_variable': 0.4
        }
        
        base_gain = mechanism_gains.get(candidate['mechanism'], 0.3)
        
        # Adjust based on current uncertainty
        target_variables = candidate['variables']
        uncertainty = 0.0
        
        for var in target_variables:
            if var in current_beliefs:
                # Higher uncertainty = higher potential gain
                belief = current_beliefs[var]
                var_uncertainty = 1.0 - abs(2 * belief - 1)  # Uncertainty peaks at 0.5 belief
                uncertainty += var_uncertainty
                
        uncertainty = uncertainty / len(target_variables) if target_variables else 0.5
        
        # Agent-specific adjustments
        experience_factor = 1.0 - agent_profile.get('causal_reasoning_experience', 0.5)
        
        # Exploration bonus for novel interventions
        intervention_novelty = self._compute_intervention_novelty(candidate)
        exploration_bonus = self.exploration_weight * intervention_novelty
        
        total_gain = base_gain * (0.5 + 0.5 * uncertainty) * (0.7 + 0.3 * experience_factor) + exploration_bonus
        
        return min(1.0, total_gain)
        
    def _estimate_effect_size(self,
                            candidate: Dict[str, Any],
                            graph: nx.DiGraph,
                            target_outcome: str) -> float:
        """Estimate the causal effect size of the intervention."""
        
        intervention_vars = candidate['variables']
        
        # Simple heuristic based on graph structure
        total_effect = 0.0
        
        for var in intervention_vars:
            # Direct effect
            if graph.has_edge(var, target_outcome):
                total_effect += 0.5
                
            # Indirect effects through mediators
            try:
                paths = list(nx.all_simple_paths(graph, var, target_outcome, cutoff=3))
                for path in paths:
                    if len(path) > 2:  # Indirect path
                        # Diminishing effect with path length
                        indirect_effect = 0.3 / (len(path) - 1)
                        total_effect += indirect_effect
            except nx.NetworkXNoPath:
                pass
                
        # Normalize and add mechanism-specific adjustments
        mechanism_multipliers = {
            'direct_cause': 1.0,
            'confounder_control': 0.6,  # Effect through unblocking
            'mediation_analysis': 0.4,  # Effect through blocking
            'joint_intervention': 0.8,
            'instrumental_variable': 0.3
        }
        
        multiplier = mechanism_multipliers.get(candidate['mechanism'], 0.5)
        return min(1.0, total_effect * multiplier)
        
    def _compute_intervention_cost(self, candidate: Dict[str, Any], graph: nx.DiGraph) -> float:
        """Compute the cost/difficulty of performing the intervention."""
        
        # Base costs by intervention type
        type_costs = {
            'set': 0.3,
            'randomize': 0.5,
            'prevent': 0.7
        }
        
        base_cost = type_costs.get(candidate['type'], 0.5)
        
        # Multi-variable interventions are more complex
        complexity_cost = 0.2 * (len(candidate['variables']) - 1)
        
        # Some mechanisms are inherently more difficult
        mechanism_costs = {
            'direct_cause': 0.0,
            'confounder_control': 0.2,
            'mediation_analysis': 0.3,
            'joint_intervention': 0.4,
            'instrumental_variable': 0.5
        }
        
        mechanism_cost = mechanism_costs.get(candidate['mechanism'], 0.3)
        
        total_cost = base_cost + complexity_cost + mechanism_cost
        return min(1.0, total_cost)
        
    def _generate_causal_reasoning(self,
                                 candidate: Dict[str, Any],
                                 graph: nx.DiGraph,
                                 target_outcome: str) -> str:
        """Generate human-readable causal reasoning for the intervention."""
        
        variables = candidate['variables']
        mechanism = candidate['mechanism']
        intervention_type = candidate['type']
        
        reasoning_templates = {
            'direct_cause': f"By {intervention_type}ting {', '.join(variables)}, we can directly observe the causal effect on {target_outcome}. This intervention eliminates confounding from other variables and allows us to measure the true causal relationship.",
            
            'confounder_control': f"The variable(s) {', '.join(variables)} confound the relationship with {target_outcome}. By {intervention_type}ting these confounders, we block backdoor paths and can identify the true causal effect.",
            
            'mediation_analysis': f"By {intervention_type}ting the mediator(s) {', '.join(variables)}, we can determine how much of the causal effect on {target_outcome} flows through this pathway versus direct effects.",
            
            'joint_intervention': f"A joint intervention on {', '.join(variables)} allows us to control multiple confounding pathways simultaneously, providing cleaner causal identification for {target_outcome}.",
            
            'instrumental_variable': f"Using {', '.join(variables)} as an instrument provides exogenous variation that helps identify causal effects on {target_outcome} even in the presence of unobserved confounding."
        }
        
        base_reasoning = reasoning_templates.get(mechanism, 
            f"Intervening on {', '.join(variables)} will help clarify causal relationships with {target_outcome}.")
            
        # Add specific graph structure insights
        graph_insights = []
        
        for var in variables:
            # Count incoming and outgoing edges
            in_degree = graph.in_degree(var)
            out_degree = graph.out_degree(var)
            
            if in_degree == 0:
                graph_insights.append(f"{var} is a root cause variable")
            elif out_degree == 0:
                graph_insights.append(f"{var} is an outcome variable")
            else:
                graph_insights.append(f"{var} has {in_degree} causes and affects {out_degree} other variables")
                
        if graph_insights:
            base_reasoning += f" Graph structure: {'. '.join(graph_insights)}."
            
        return base_reasoning
        
    def _generate_alternatives(self, primary_candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative intervention strategies."""
        alternatives = []
        
        # Alternative 1: Different intervention type on same variables
        if primary_candidate['type'] == 'set':
            alternatives.append({
                'variables': primary_candidate['variables'],
                'type': 'randomize',
                'rationale': 'Randomize instead of setting specific values'
            })
        elif primary_candidate['type'] == 'randomize':
            alternatives.append({
                'variables': primary_candidate['variables'],
                'type': 'set',
                'rationale': 'Set specific values instead of randomizing'
            })
            
        # Alternative 2: Subset of variables (if multi-variable intervention)
        if len(primary_candidate['variables']) > 1:
            alternatives.append({
                'variables': primary_candidate['variables'][:1],
                'type': primary_candidate['type'],
                'rationale': 'Simpler single-variable intervention'
            })
            
        # Alternative 3: Different values (if applicable)
        if primary_candidate['type'] == 'set':
            alt_values = {}
            for var, val in primary_candidate['values'].items():
                alt_values[var] = not val if isinstance(val, bool) else 'alternative_value'
            alternatives.append({
                'variables': primary_candidate['variables'],
                'type': 'set',
                'values': alt_values,
                'rationale': 'Alternative intervention values'
            })
            
        return alternatives[:2]  # Limit to 2 alternatives
        
    def _compute_intervention_novelty(self, candidate: Dict[str, Any]) -> float:
        """Compute how novel this intervention is based on history."""
        if not self.intervention_history:
            return 1.0  # Maximum novelty if no history
            
        # Simple novelty measure: how different from past interventions
        novelty_score = 1.0
        
        for past_result in self.intervention_history[-10:]:  # Look at recent history
            # This would compare intervention characteristics
            # For now, return a moderate novelty score
            pass
            
        return max(0.1, novelty_score)
        
    def update_from_intervention_result(self, result: InterventionResult) -> None:
        """Update the intervention engine based on intervention results."""
        self.intervention_history.append(result)
        
        # Update effectiveness models
        intervention_type = result.intervention_id.split('_')[0] if '_' in result.intervention_id else 'unknown'
        
        if intervention_type not in self.effectiveness_models:
            self.effectiveness_models[intervention_type] = {
                'success_rate': 0.5,
                'avg_effectiveness': 0.5,
                'count': 0
            }
            
        model = self.effectiveness_models[intervention_type]
        model['count'] += 1
        
        # Update running averages
        alpha = self.learning_rate
        model['avg_effectiveness'] = ((1 - alpha) * model['avg_effectiveness'] + 
                                     alpha * result.intervention_effectiveness)
        
        # Update variable importance based on belief changes
        for var, change in result.belief_changes.items():
            if var not in self.variable_importance_scores:
                self.variable_importance_scores[var] = 0.5
                
            # Variables with larger belief changes are more important
            importance_update = abs(change) * alpha
            self.variable_importance_scores[var] = ((1 - alpha) * self.variable_importance_scores[var] + 
                                                   alpha * importance_update)
            
        logger.info(f"Updated intervention models from result: {result.intervention_id}")


class OptimalInterventionSelector:
    """Advanced intervention selection using optimization techniques."""
    
    def __init__(self, optimization_method: str = 'bayesian'):
        """Initialize optimal intervention selector.
        
        Args:
            optimization_method: Method for optimization ('bayesian', 'genetic', 'greedy')
        """
        self.optimization_method = optimization_method
        self.intervention_space_cache: Dict[str, Any] = {}
        
    def select_optimal_sequence(self,
                              available_interventions: List[InterventionRecommendation],
                              sequence_length: int,
                              optimization_objective: str = 'information_gain') -> List[InterventionRecommendation]:
        """Select optimal sequence of interventions using mathematical optimization.
        
        Args:
            available_interventions: Pool of intervention candidates
            sequence_length: Length of intervention sequence
            optimization_objective: Objective to optimize ('information_gain', 'effect_size', 'utility')
            
        Returns:
            Optimally ordered sequence of interventions
        """
        
        if len(available_interventions) <= sequence_length:
            return available_interventions
            
        if self.optimization_method == 'bayesian':
            return self._bayesian_optimization_sequence(
                available_interventions, sequence_length, optimization_objective)
        elif self.optimization_method == 'genetic':
            return self._genetic_algorithm_sequence(
                available_interventions, sequence_length, optimization_objective)
        else:
            return self._greedy_sequence_selection(
                available_interventions, sequence_length, optimization_objective)
            
    def _bayesian_optimization_sequence(self,
                                      interventions: List[InterventionRecommendation],
                                      length: int,
                                      objective: str) -> List[InterventionRecommendation]:
        """Use Bayesian optimization for intervention sequence selection."""
        
        # Simplified Bayesian approach using acquisition function
        selected = []
        remaining = interventions.copy()
        
        for step in range(min(length, len(interventions))):
            # Acquisition function: Upper Confidence Bound
            best_intervention = None
            best_score = -float('inf')
            
            for intervention in remaining:
                # Expected utility
                if objective == 'information_gain':
                    expected_value = intervention.cost_benefit_analysis['information_gain']
                elif objective == 'effect_size':
                    expected_value = intervention.expected_effect_size
                else:  # utility
                    expected_value = intervention.cost_benefit_analysis['utility']
                    
                # Confidence bound (higher for less confident interventions)
                confidence_bonus = (1.0 - intervention.confidence) * 0.3
                
                # Diversity bonus (prefer different types of interventions)
                diversity_bonus = self._compute_diversity_bonus(intervention, selected)
                
                acquisition_score = expected_value + confidence_bonus + diversity_bonus
                
                if acquisition_score > best_score:
                    best_score = acquisition_score
                    best_intervention = intervention
                    
            if best_intervention:
                selected.append(best_intervention)
                remaining.remove(best_intervention)
                
        return selected
        
    def _genetic_algorithm_sequence(self,
                                  interventions: List[InterventionRecommendation],
                                  length: int,
                                  objective: str) -> List[InterventionRecommendation]:
        """Use genetic algorithm for sequence optimization."""
        
        # Simplified genetic algorithm
        population_size = min(50, len(interventions) * 2)
        generations = 20
        
        # Initialize population with random sequences
        population = []
        for _ in range(population_size):
            sequence_indices = np.random.choice(len(interventions), 
                                              size=min(length, len(interventions)), 
                                              replace=False)
            population.append(sequence_indices)
            
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for sequence_indices in population:
                sequence = [interventions[i] for i in sequence_indices]
                fitness = self._evaluate_sequence_fitness(sequence, objective)
                fitness_scores.append(fitness)
                
            # Selection, crossover, mutation (simplified)
            new_population = []
            
            # Keep best individuals (elitism)
            sorted_population = sorted(zip(population, fitness_scores), 
                                     key=lambda x: x[1], reverse=True)
            elite_size = population_size // 4
            new_population.extend([ind for ind, _ in sorted_population[:elite_size]])
            
            # Generate offspring through crossover and mutation
            while len(new_population) < population_size:
                # Select parents (tournament selection)
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self._crossover(parent1, parent2, len(interventions))
                
                # Mutation
                if np.random.random() < 0.1:
                    child = self._mutate(child, len(interventions))
                    
                new_population.append(child)
                
            population = new_population
            
        # Return best sequence
        final_fitness = [self._evaluate_sequence_fitness([interventions[i] for i in seq], objective) 
                        for seq in population]
        best_sequence_indices = population[np.argmax(final_fitness)]
        
        return [interventions[i] for i in best_sequence_indices]
        
    def _greedy_sequence_selection(self,
                                 interventions: List[InterventionRecommendation], 
                                 length: int,
                                 objective: str) -> List[InterventionRecommendation]:
        """Simple greedy selection of intervention sequence."""
        
        # Sort by objective and select top interventions
        if objective == 'information_gain':
            key_func = lambda x: x.cost_benefit_analysis['information_gain']
        elif objective == 'effect_size':
            key_func = lambda x: x.expected_effect_size
        else:  # utility
            key_func = lambda x: x.cost_benefit_analysis['utility']
            
        sorted_interventions = sorted(interventions, key=key_func, reverse=True)
        return sorted_interventions[:length]
        
    def _compute_diversity_bonus(self, 
                               intervention: InterventionRecommendation,
                               selected: List[InterventionRecommendation]) -> float:
        """Compute bonus for intervention diversity."""
        if not selected:
            return 0.0
            
        # Penalize similar intervention types
        similar_types = sum(1 for sel in selected 
                          if sel.intervention_type == intervention.intervention_type)
        
        # Penalize overlapping target variables
        selected_vars = set()
        for sel in selected:
            selected_vars.update(sel.target_variables)
            
        overlap = len(set(intervention.target_variables).intersection(selected_vars))
        
        diversity_bonus = 0.2 * (1 / (1 + similar_types)) * (1 / (1 + overlap))
        return diversity_bonus
        
    def _evaluate_sequence_fitness(self,
                                 sequence: List[InterventionRecommendation],
                                 objective: str) -> float:
        """Evaluate fitness of an intervention sequence."""
        
        total_fitness = 0.0
        diminishing_factor = 1.0
        
        for i, intervention in enumerate(sequence):
            if objective == 'information_gain':
                value = intervention.cost_benefit_analysis['information_gain']
            elif objective == 'effect_size':
                value = intervention.expected_effect_size
            else:  # utility
                value = intervention.cost_benefit_analysis['utility']
                
            # Apply diminishing returns
            total_fitness += value * diminishing_factor
            diminishing_factor *= 0.9  # Each subsequent intervention has less impact
            
        # Bonus for sequence diversity
        types_used = set(interv.intervention_type for interv in sequence)
        diversity_bonus = 0.1 * len(types_used)
        
        return total_fitness + diversity_bonus
        
    def _tournament_selection(self, population: List, fitness_scores: List[float]) -> List:
        """Tournament selection for genetic algorithm."""
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), size=tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx]
        
    def _crossover(self, parent1: List, parent2: List, max_val: int) -> List:
        """Crossover operation for genetic algorithm."""
        # Order crossover
        length = len(parent1)
        start, end = sorted(np.random.choice(length, size=2, replace=False))
        
        child = [-1] * length
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2
        parent2_remaining = [x for x in parent2 if x not in child]
        child_remaining_positions = [i for i, val in enumerate(child) if val == -1]
        
        for i, pos in enumerate(child_remaining_positions):
            if i < len(parent2_remaining):
                child[pos] = parent2_remaining[i]
            else:
                # Fill with random valid values
                available = [x for x in range(max_val) if x not in child]
                if available:
                    child[pos] = np.random.choice(available)
                    
        return [x for x in child if x != -1]
        
    def _mutate(self, individual: List, max_val: int) -> List:
        """Mutation operation for genetic algorithm."""
        mutated = individual.copy()
        
        if len(mutated) > 1:
            # Swap mutation
            i, j = np.random.choice(len(mutated), size=2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
        return mutated