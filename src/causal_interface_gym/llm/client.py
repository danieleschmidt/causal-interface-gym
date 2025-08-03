"""High-level LLM client for causal reasoning tasks."""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .providers import LLMProvider, LLMResponse, create_provider
from .prompts import CausalPromptBuilder
from .belief_extraction import BeliefExtractor
from .response_parser import ResponseParser
from ..database import CacheManager

logger = logging.getLogger(__name__)


class LLMClient:
    """High-level client for LLM interactions in causal reasoning tasks."""
    
    def __init__(self, provider: LLMProvider, cache_manager: Optional[CacheManager] = None):
        """Initialize LLM client.
        
        Args:
            provider: LLM provider instance
            cache_manager: Optional cache manager for response caching
        """
        self.provider = provider
        self.cache = cache_manager
        self.prompt_builder = CausalPromptBuilder()
        self.belief_extractor = BeliefExtractor()
        self.response_parser = ResponseParser()
        
        logger.info(f"Initialized LLM client with {provider.get_provider_name()} provider")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], cache_manager: Optional[CacheManager] = None) -> 'LLMClient':
        """Create client from configuration.
        
        Args:
            config: Configuration dictionary
            cache_manager: Optional cache manager
            
        Returns:
            LLM client instance
        """
        provider_type = config.get("provider", "openai")
        model = config.get("model", "gpt-4")
        provider_config = config.get("provider_config", {})
        
        provider = create_provider(provider_type, model, **provider_config)
        return cls(provider, cache_manager)
    
    def query_belief(self, belief_statement: str, condition: str, 
                    context: Optional[Dict[str, Any]] = None) -> float:
        """Query LLM for belief probability.
        
        Args:
            belief_statement: Belief to query (e.g., "P(rain|wet_grass)")
            condition: Condition type (observational, interventional)
            context: Additional context (graph, scenario, etc.)
            
        Returns:
            Belief probability between 0 and 1
        """
        # Build prompt for belief query
        prompt = self.prompt_builder.build_belief_query(
            belief_statement=belief_statement,
            condition=condition,
            context=context or {}
        )
        
        # Check cache first
        if self.cache:
            cached_response = self.cache.cache_llm_response(
                prompt, 
                self.provider.model,
                lambda: self._generate_response(prompt)
            )
        else:
            cached_response = self._generate_response(prompt)
        
        # Extract belief probability from response
        belief_prob = self.belief_extractor.extract_probability(
            cached_response.content,
            belief_statement
        )
        
        logger.debug(f"Extracted belief {belief_statement}: {belief_prob}")
        return belief_prob
    
    def explain_causal_reasoning(self, scenario: Dict[str, Any], 
                               intervention: Dict[str, Any]) -> str:
        """Get explanation of causal reasoning for scenario.
        
        Args:
            scenario: Causal scenario description
            intervention: Intervention details
            
        Returns:
            Explanation text
        """
        prompt = self.prompt_builder.build_explanation_prompt(
            scenario=scenario,
            intervention=intervention
        )
        
        response = self._generate_response(prompt)
        return response.content
    
    def evaluate_intervention_understanding(self, graph: Dict[str, Any], 
                                          treatment: str, outcome: str) -> Dict[str, Any]:
        """Evaluate LLM's understanding of interventions vs observations.
        
        Args:
            graph: Causal graph structure
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            Evaluation results
        """
        # Query observational belief
        obs_prompt = self.prompt_builder.build_observational_query(
            graph=graph,
            treatment=treatment,
            outcome=outcome
        )
        obs_response = self._generate_response(obs_prompt)
        obs_belief = self.belief_extractor.extract_probability(
            obs_response.content,
            f"P({outcome}|{treatment})"
        )
        
        # Query interventional belief
        int_prompt = self.prompt_builder.build_interventional_query(
            graph=graph,
            treatment=treatment,
            outcome=outcome
        )
        int_response = self._generate_response(int_prompt)
        int_belief = self.belief_extractor.extract_probability(
            int_response.content,
            f"P({outcome}|do({treatment}))"
        )
        
        # Analyze understanding
        difference = abs(int_belief - obs_belief)
        understanding_score = min(1.0, difference / 0.3)  # Normalize to 0-1
        
        return {
            "observational_belief": obs_belief,
            "interventional_belief": int_belief,
            "difference": difference,
            "understanding_score": understanding_score,
            "responses": {
                "observational": obs_response.content,
                "interventional": int_response.content
            }
        }
    
    def identify_confounders(self, graph: Dict[str, Any], 
                           treatment: str, outcome: str) -> List[str]:
        """Ask LLM to identify confounders in causal graph.
        
        Args:
            graph: Causal graph structure
            treatment: Treatment variable
            outcome: Outcome variable
            
        Returns:
            List of identified confounders
        """
        prompt = self.prompt_builder.build_confounder_identification_prompt(
            graph=graph,
            treatment=treatment,
            outcome=outcome
        )
        
        response = self._generate_response(prompt)
        confounders = self.response_parser.parse_variable_list(
            response.content,
            graph_variables=list(graph.get("nodes", graph.keys()))
        )
        
        return confounders
    
    def generate_counterfactual(self, scenario: Dict[str, Any], 
                              counterfactual_condition: str) -> str:
        """Generate counterfactual reasoning response.
        
        Args:
            scenario: Original scenario
            counterfactual_condition: Counterfactual condition
            
        Returns:
            Counterfactual reasoning response
        """
        prompt = self.prompt_builder.build_counterfactual_prompt(
            scenario=scenario,
            counterfactual_condition=counterfactual_condition
        )
        
        response = self._generate_response(prompt)
        return response.content
    
    def batch_query_beliefs(self, belief_queries: List[Dict[str, Any]]) -> List[float]:
        """Query multiple beliefs in batch for efficiency.
        
        Args:
            belief_queries: List of belief query dictionaries
            
        Returns:
            List of belief probabilities
        """
        results = []
        
        for query in belief_queries:
            belief_prob = self.query_belief(
                belief_statement=query["belief_statement"],
                condition=query["condition"],
                context=query.get("context")
            )
            results.append(belief_prob)
        
        return results
    
    def test_causal_reasoning_capabilities(self, test_scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test LLM's causal reasoning across multiple scenarios.
        
        Args:
            test_scenarios: List of test scenarios
            
        Returns:
            Comprehensive test results
        """
        results = {
            "scenario_results": [],
            "overall_scores": {},
            "capabilities": {
                "intervention_understanding": [],
                "confounder_identification": [],
                "counterfactual_reasoning": []
            }
        }
        
        for scenario in test_scenarios:
            scenario_result = {
                "scenario_id": scenario.get("id", "unknown"),
                "type": scenario.get("type", "general")
            }
            
            # Test intervention understanding
            if "intervention_test" in scenario:
                int_test = scenario["intervention_test"]
                evaluation = self.evaluate_intervention_understanding(
                    graph=int_test["graph"],
                    treatment=int_test["treatment"],
                    outcome=int_test["outcome"]
                )
                scenario_result["intervention_evaluation"] = evaluation
                results["capabilities"]["intervention_understanding"].append(
                    evaluation["understanding_score"]
                )
            
            # Test confounder identification
            if "confounder_test" in scenario:
                conf_test = scenario["confounder_test"]
                identified = self.identify_confounders(
                    graph=conf_test["graph"],
                    treatment=conf_test["treatment"],
                    outcome=conf_test["outcome"]
                )
                
                # Score against known confounders
                true_confounders = set(conf_test.get("true_confounders", []))
                identified_set = set(identified)
                
                precision = len(true_confounders & identified_set) / len(identified_set) if identified_set else 0
                recall = len(true_confounders & identified_set) / len(true_confounders) if true_confounders else 1
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                scenario_result["confounder_evaluation"] = {
                    "identified": identified,
                    "true_confounders": list(true_confounders),
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
                results["capabilities"]["confounder_identification"].append(f1_score)
            
            results["scenario_results"].append(scenario_result)
        
        # Calculate overall scores
        for capability, scores in results["capabilities"].items():
            if scores:
                results["overall_scores"][capability] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        return results
    
    def _generate_response(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from provider.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LLM response
        """
        try:
            response = self.provider.generate(prompt, **kwargs)
            logger.debug(f"Generated response: {len(response.content)} characters")
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return fallback response
            return LLMResponse(
                content="Error: Unable to generate response",
                model=self.provider.model,
                provider=self.provider.get_provider_name(),
                metadata={"error": str(e)}
            )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Usage statistics
        """
        stats = {
            "provider": self.provider.get_provider_name(),
            "model": self.provider.model,
            "cache_enabled": self.cache is not None
        }
        
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            stats["cache_stats"] = cache_stats
        
        return stats