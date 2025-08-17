"""High-level LLM client for causal reasoning tasks."""

import logging
from typing import Dict, Any, Optional, List

from .providers import LLMProvider, get_provider

logger = logging.getLogger(__name__)


class LLMClient:
    """High-level client for LLM interactions in causal reasoning tasks."""
    
    def __init__(self, provider: LLMProvider):
        """Initialize LLM client.
        
        Args:
            provider: LLM provider instance
        """
        self.provider = provider
        logger.info(f"Initialized LLM client with {type(provider).__name__} provider")
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LLMClient':
        """Create client from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            LLM client instance
        """
        provider_type = config.get("provider", "local")
        model = config.get("model", "test-model")
        provider_config = config.get("provider_config", {})
        
        provider = get_provider(provider_type, model_name=model, **provider_config)
        return cls(provider)
    
    def query_belief(self, belief_statement: str, condition: str) -> float:
        """Query LLM for belief probability.
        
        Args:
            belief_statement: Belief to query (e.g., "P(rain|wet_grass)")
            condition: Condition type (observational, interventional)
            
        Returns:
            Belief probability between 0 and 1
        """
        return self.provider.query_belief(belief_statement, condition)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        response = self.provider.generate_response(prompt, **kwargs)
        return response.get("text", "")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Usage statistics
        """
        return {
            "provider": type(self.provider).__name__,
            "model": getattr(self.provider, 'model_name', 'unknown')
        }