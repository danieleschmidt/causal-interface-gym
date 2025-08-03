"""Prompt templates for causal reasoning tasks."""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class PromptTemplate(ABC):
    """Base class for prompt templates."""
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the prompt template with given parameters."""
        pass


class CausalPromptBuilder:
    """Builder for causal reasoning prompts."""
    
    def __init__(self):
        """Initialize prompt builder."""
        self.base_instructions = """
You are an expert in causal reasoning and Pearl's causal hierarchy. 
You understand the difference between:
1. Association: P(Y|X) - observational relationships
2. Intervention: P(Y|do(X)) - causal effects from interventions
3. Counterfactuals: P(Y_x|X',Y') - what would have happened

Always distinguish between correlation and causation.
"""
    
    def build_belief_query(self, belief_statement: str, condition: str, 
                          context: Dict[str, Any]) -> str:
        """Build prompt for belief probability query.
        
        Args:
            belief_statement: Belief to query
            condition: Condition type
            context: Additional context
            
        Returns:
            Formatted prompt
        """
        context_str = self._format_context(context)
        
        if "do(" in condition:
            # Interventional query
            prompt = f"""{self.base_instructions}

{context_str}

Given the above causal structure and an INTERVENTION where we {condition}, 
what is the probability of {belief_statement}?

This is asking for P({belief_statement.replace('P(', '').replace(')', '')}|{condition}), 
which is different from just observing the variables.

Provide your answer as a single probability value between 0 and 1.
Example format: 0.75

Probability:"""
        else:
            # Observational query
            prompt = f"""{self.base_instructions}

{context_str}

Given the above causal structure, through OBSERVATION only (no interventions), 
what is the probability of {belief_statement}?

This is asking for the observational probability based on natural associations.

Provide your answer as a single probability value between 0 and 1.
Example format: 0.75

Probability:"""
        
        return prompt
    
    def build_observational_query(self, graph: Dict[str, Any], 
                                treatment: str, outcome: str) -> str:
        """Build observational probability query.
        
        Args:
            graph: Causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Formatted prompt
        """
        graph_str = self._format_graph(graph)
        
        return f"""{self.base_instructions}

Causal Graph:
{graph_str}

Through OBSERVATION only, if we see that {treatment} has occurred, 
what is the probability that {outcome} will occur?

This is P({outcome}|{treatment}) - an observational, associational relationship.
Remember: correlation does not imply causation.

Provide your answer as a single probability value between 0 and 1.
Probability:"""
    
    def build_interventional_query(self, graph: Dict[str, Any], 
                                 treatment: str, outcome: str) -> str:
        """Build interventional probability query.
        
        Args:
            graph: Causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Formatted prompt
        """
        graph_str = self._format_graph(graph)
        
        return f"""{self.base_instructions}

Causal Graph:
{graph_str}

If we INTERVENE to set {treatment} (written as do({treatment})), 
what is the probability that {outcome} will occur?

This is P({outcome}|do({treatment})) - a causal effect from intervention.
This is different from just observing {treatment} because:
- Intervention breaks incoming causal arrows to {treatment}
- We're asking about the direct causal effect
- Confounders are "blocked" by the intervention

Provide your answer as a single probability value between 0 and 1.
Probability:"""
    
    def build_explanation_prompt(self, scenario: Dict[str, Any], 
                               intervention: Dict[str, Any]) -> str:
        """Build prompt for causal explanation.
        
        Args:
            scenario: Scenario description
            intervention: Intervention details
            
        Returns:
            Formatted prompt
        """
        scenario_str = self._format_scenario(scenario)
        intervention_str = self._format_intervention(intervention)
        
        return f"""{self.base_instructions}

Scenario:
{scenario_str}

Intervention:
{intervention_str}

Please explain the causal reasoning for this scenario:
1. What is the causal structure?
2. How does the intervention work?
3. What confounders, if any, are present?
4. How would the intervention differ from just observing?

Provide a clear, step-by-step causal analysis."""
    
    def build_confounder_identification_prompt(self, graph: Dict[str, Any], 
                                             treatment: str, outcome: str) -> str:
        """Build prompt for confounder identification.
        
        Args:
            graph: Causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Formatted prompt
        """
        graph_str = self._format_graph(graph)
        
        return f"""{self.base_instructions}

Causal Graph:
{graph_str}

For the relationship between {treatment} (treatment) and {outcome} (outcome), 
identify all CONFOUNDERS that would need to be controlled for to estimate 
the causal effect.

A confounder is a variable that:
1. Influences both the treatment and the outcome
2. Creates a "backdoor path" between treatment and outcome
3. Could create spurious associations

List the confounders, one per line, using this format:
- variable_name

If there are no confounders, respond with: "No confounders"

Confounders:"""
    
    def build_counterfactual_prompt(self, scenario: Dict[str, Any], 
                                  counterfactual_condition: str) -> str:
        """Build prompt for counterfactual reasoning.
        
        Args:
            scenario: Original scenario
            counterfactual_condition: Counterfactual condition
            
        Returns:
            Formatted prompt
        """
        scenario_str = self._format_scenario(scenario)
        
        return f"""{self.base_instructions}

Original Scenario:
{scenario_str}

Counterfactual Question:
{counterfactual_condition}

This is asking for counterfactual reasoning: given what we observed, 
what would have happened if things had been different?

Counterfactual reasoning requires:
1. Understanding the actual causal mechanisms
2. Considering what would change if we altered the past
3. Keeping individual-specific factors constant

Provide your counterfactual analysis:"""
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted context string
        """
        if not context:
            return "No additional context provided."
        
        parts = []
        
        if "graph" in context:
            parts.append(f"Causal Graph:\n{self._format_graph(context['graph'])}")
        
        if "scenario" in context:
            parts.append(f"Scenario:\n{self._format_scenario(context['scenario'])}")
        
        if "variables" in context:
            parts.append(f"Variables: {', '.join(context['variables'])}")
        
        if "description" in context:
            parts.append(f"Description: {context['description']}")
        
        return "\n\n".join(parts) if parts else "Context provided but not formatted."
    
    def _format_graph(self, graph: Dict[str, Any]) -> str:
        """Format causal graph for display.
        
        Args:
            graph: Graph structure
            
        Returns:
            Formatted graph string
        """
        if "nodes" in graph and "edges" in graph:
            # NetworkX-style format
            nodes = graph["nodes"]
            edges = graph["edges"]
            
            graph_str = f"Nodes: {', '.join(nodes)}\n"
            graph_str += "Edges (causal relationships):\n"
            
            for edge in edges:
                if isinstance(edge, dict):
                    graph_str += f"  {edge['from']} → {edge['to']}\n"
                elif isinstance(edge, (list, tuple)) and len(edge) == 2:
                    graph_str += f"  {edge[0]} → {edge[1]}\n"
            
            return graph_str
        else:
            # DAG format {child: [parents]}
            graph_str = "Causal relationships:\n"
            for child, parents in graph.items():
                if parents:
                    for parent in parents:
                        graph_str += f"  {parent} → {child}\n"
                else:
                    graph_str += f"  {child} (no parents)\n"
            
            return graph_str
    
    def _format_scenario(self, scenario: Dict[str, Any]) -> str:
        """Format scenario description.
        
        Args:
            scenario: Scenario dictionary
            
        Returns:
            Formatted scenario string
        """
        if isinstance(scenario, str):
            return scenario
        
        parts = []
        
        if "description" in scenario:
            parts.append(scenario["description"])
        
        if "variables" in scenario:
            parts.append(f"Variables involved: {', '.join(scenario['variables'])}")
        
        if "setting" in scenario:
            parts.append(f"Setting: {scenario['setting']}")
        
        return "\n".join(parts) if parts else str(scenario)
    
    def _format_intervention(self, intervention: Dict[str, Any]) -> str:
        """Format intervention description.
        
        Args:
            intervention: Intervention dictionary
            
        Returns:
            Formatted intervention string
        """
        if isinstance(intervention, str):
            return intervention
        
        parts = []
        
        if "variable" in intervention and "value" in intervention:
            parts.append(f"Intervene on {intervention['variable']} = {intervention['value']}")
        
        if "description" in intervention:
            parts.append(intervention["description"])
        
        if "type" in intervention:
            parts.append(f"Intervention type: {intervention['type']}")
        
        return "\n".join(parts) if parts else str(intervention)


class BeliefExtractionPrompts:
    """Prompts specifically for belief extraction."""
    
    @staticmethod
    def probability_extraction_prompt(text: str, target_belief: str) -> str:
        """Prompt for extracting probability from text.
        
        Args:
            text: Text containing probability
            target_belief: Target belief statement
            
        Returns:
            Extraction prompt
        """
        return f"""
Extract the probability value for "{target_belief}" from the following text.
Look for numerical values between 0 and 1, percentages, or probability phrases.

Text:
{text}

Return only the probability as a decimal between 0 and 1.
If no clear probability is found, return 0.5 as default.

Probability:"""
    
    @staticmethod
    def confidence_extraction_prompt(text: str) -> str:
        """Prompt for extracting confidence level.
        
        Args:
            text: Text containing confidence
            
        Returns:
            Extraction prompt
        """
        return f"""
Extract the confidence level or certainty from the following text.
Look for words like "certain", "confident", "sure", "likely", etc.

Text:
{text}

Return a confidence score between 0 and 1, where:
- 0.0 = completely uncertain
- 0.5 = neutral/unsure
- 1.0 = completely certain

Confidence:"""


class InterventionPrompts:
    """Prompts for intervention understanding tasks."""
    
    @staticmethod
    def intervention_vs_observation_prompt(scenario: str, variable: str) -> str:
        """Prompt to test intervention vs observation understanding.
        
        Args:
            scenario: Scenario description
            variable: Variable of interest
            
        Returns:
            Test prompt
        """
        return f"""
Scenario: {scenario}

Question 1: What is the probability of {variable} if we simply OBSERVE the current situation?
This is asking for P({variable}|current observations).

Question 2: What is the probability of {variable} if we INTERVENE to change the situation?
This is asking for P({variable}|do(intervention)).

Explain the difference between these two probabilities and why they might differ.
Provide specific probability values for both questions.

Answer:"""
    
    @staticmethod
    def backdoor_path_prompt(graph: Dict[str, Any], treatment: str, outcome: str) -> str:
        """Prompt for backdoor path identification.
        
        Args:
            graph: Causal graph
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Backdoor identification prompt
        """
        graph_str = CausalPromptBuilder()._format_graph(graph)
        
        return f"""
Causal Graph:
{graph_str}

Identify all BACKDOOR PATHS from {treatment} to {outcome}.

A backdoor path is a path that:
1. Starts with an arrow INTO the treatment variable
2. Connects treatment to outcome through confounders
3. Could create spurious associations

For each backdoor path found, list the variables in order.
If no backdoor paths exist, state "No backdoor paths".

Backdoor paths:"""


class TestPrompts:
    """Prompts for testing causal reasoning capabilities."""
    
    @staticmethod
    def generate_test_scenarios() -> List[Dict[str, Any]]:
        """Generate standard test scenarios.
        
        Returns:
            List of test scenario dictionaries
        """
        return [
            {
                "id": "smoking_cancer",
                "type": "medical",
                "description": "Smoking and lung cancer with genetic confounding",
                "graph": {
                    "smoking": ["genetics"],
                    "cancer": ["smoking", "genetics"],
                    "genetics": []
                },
                "intervention_test": {
                    "graph": {"smoking": ["genetics"], "cancer": ["smoking", "genetics"], "genetics": []},
                    "treatment": "smoking",
                    "outcome": "cancer"
                },
                "confounder_test": {
                    "graph": {"smoking": ["genetics"], "cancer": ["smoking", "genetics"], "genetics": []},
                    "treatment": "smoking",
                    "outcome": "cancer",
                    "true_confounders": ["genetics"]
                }
            },
            {
                "id": "education_income",
                "type": "economics",
                "description": "Education and income with ability confounding",
                "graph": {
                    "ability": [],
                    "education": ["ability"],
                    "income": ["education", "ability"]
                },
                "intervention_test": {
                    "graph": {"ability": [], "education": ["ability"], "income": ["education", "ability"]},
                    "treatment": "education",
                    "outcome": "income"
                },
                "confounder_test": {
                    "graph": {"ability": [], "education": ["ability"], "income": ["education", "ability"]},
                    "treatment": "education",
                    "outcome": "income",
                    "true_confounders": ["ability"]
                }
            },
            {
                "id": "simple_chain",
                "type": "basic",
                "description": "Simple causal chain with no confounding",
                "graph": {
                    "A": [],
                    "B": ["A"],
                    "C": ["B"]
                },
                "intervention_test": {
                    "graph": {"A": [], "B": ["A"], "C": ["B"]},
                    "treatment": "A",
                    "outcome": "C"
                },
                "confounder_test": {
                    "graph": {"A": [], "B": ["A"], "C": ["B"]},
                    "treatment": "A",
                    "outcome": "C",
                    "true_confounders": []
                }
            }
        ]