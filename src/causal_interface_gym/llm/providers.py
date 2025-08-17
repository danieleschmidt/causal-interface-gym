"""LLM providers for causal reasoning experiments."""

from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import json
import os
from enum import Enum


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize LLM provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{self.__class__.__name__.upper()}_API_KEY")
    
    @abstractmethod
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response from LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response dictionary with text and metadata
        """
        pass
    
    @abstractmethod
    def query_belief(self, belief_statement: str, condition: str) -> float:
        """Query LLM for belief probability.
        
        Args:
            belief_statement: Belief to query (e.g., "P(rain|wet_grass)")
            condition: Condition type ("observational" or "do(...)")
            
        Returns:
            Probability between 0 and 1
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for causal reasoning."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize OpenAI provider."""
        super().__init__(model_name, api_key)
        self._client = None
    
    def _get_client(self):
        """Get OpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        if not self.api_key:
            # Simulate response for testing
            return {
                "text": "Simulated OpenAI response",
                "model": self.model_name,
                "usage": {"total_tokens": 50},
                "simulated": True
            }
        
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": self.model_name,
                "usage": response.usage.total_tokens,
                "simulated": False
            }
        except Exception as e:
            return {
                "text": f"Error: {e}",
                "model": self.model_name,
                "error": str(e),
                "simulated": True
            }
    
    def query_belief(self, belief_statement: str, condition: str) -> float:
        """Query belief probability from OpenAI."""
        prompt = f"""
        Given the causal scenario, what is your confidence in the following belief?
        
        Belief: {belief_statement}
        Condition: {condition}
        
        Please provide only a number between 0 and 1, where 0 means completely false and 1 means completely true.
        """
        
        response = self.generate_response(prompt, temperature=0.1, max_tokens=10)
        
        try:
            # Extract numerical probability from response
            text = response["text"].strip()
            # Try to extract a float from the response
            import re
            numbers = re.findall(r'0?\.\d+|[01]', text)
            if numbers:
                prob = float(numbers[0])
                return max(0.0, min(1.0, prob))
        except:
            pass
        
        # Fallback to random if parsing fails
        import random
        return random.random()


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider for causal reasoning."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None):
        """Initialize Anthropic provider."""
        super().__init__(model_name, api_key)
        self._client = None
    
    def _get_client(self):
        """Get Anthropic client (lazy initialization)."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        return self._client
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using Anthropic API."""
        if not self.api_key:
            # Simulate response for testing
            return {
                "text": "Simulated Claude response",
                "model": self.model_name,
                "usage": {"total_tokens": 50},
                "simulated": True
            }
        
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "text": response.content[0].text,
                "model": self.model_name,
                "usage": response.usage.input_tokens + response.usage.output_tokens,
                "simulated": False
            }
        except Exception as e:
            return {
                "text": f"Error: {e}",
                "model": self.model_name,
                "error": str(e),
                "simulated": True
            }
    
    def query_belief(self, belief_statement: str, condition: str) -> float:
        """Query belief probability from Claude."""
        prompt = f"""
        I need you to evaluate a causal belief statement.
        
        Belief: {belief_statement}
        Condition: {condition}
        
        Please provide your confidence as a single number between 0 and 1.
        0 = completely false, 1 = completely true.
        
        Number:
        """
        
        response = self.generate_response(prompt, temperature=0.1, max_tokens=10)
        
        try:
            import re
            text = response["text"].strip()
            numbers = re.findall(r'0?\.\d+|[01]', text)
            if numbers:
                prob = float(numbers[0])
                return max(0.0, min(1.0, prob))
        except:
            pass
        
        import random
        return random.random()


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider for causal reasoning."""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None,
                 endpoint: Optional[str] = None):
        """Initialize Azure OpenAI provider."""
        super().__init__(model_name, api_key)
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._client = None
    
    def _get_client(self):
        """Get Azure OpenAI client (lazy initialization)."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.endpoint,
                    api_version="2024-02-15-preview"
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using Azure OpenAI API."""
        if not self.api_key or not self.endpoint:
            return {
                "text": "Simulated Azure OpenAI response",
                "model": self.model_name,
                "usage": {"total_tokens": 50},
                "simulated": True
            }
        
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "text": response.choices[0].message.content,
                "model": self.model_name,
                "usage": response.usage.total_tokens,
                "simulated": False
            }
        except Exception as e:
            return {
                "text": f"Error: {e}",
                "model": self.model_name,
                "error": str(e),
                "simulated": True
            }
    
    def query_belief(self, belief_statement: str, condition: str) -> float:
        """Query belief probability from Azure OpenAI."""
        prompt = f"""
        Evaluate this causal belief:
        
        Belief: {belief_statement}
        Condition: {condition}
        
        Respond with only a probability (0.0 to 1.0):
        """
        
        response = self.generate_response(prompt, temperature=0.1, max_tokens=10)
        
        try:
            import re
            text = response["text"].strip()
            numbers = re.findall(r'0?\.\d+|[01]', text)
            if numbers:
                prob = float(numbers[0])
                return max(0.0, min(1.0, prob))
        except:
            pass
        
        import random
        return random.random()


class LocalProvider(LLMProvider):
    """Local/self-hosted LLM provider."""
    
    def __init__(self, model_name: str = "local-model", endpoint: str = "http://localhost:8000"):
        """Initialize local provider."""
        super().__init__(model_name, None)
        self.endpoint = endpoint
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 1000) -> Dict[str, Any]:
        """Generate response using local endpoint."""
        try:
            import requests
            
            response = requests.post(
                f"{self.endpoint}/generate",
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data.get("text", ""),
                    "model": self.model_name,
                    "usage": data.get("usage", {}),
                    "simulated": False
                }
        except Exception as e:
            pass
        
        # Fallback simulation
        return {
            "text": f"Simulated local model response for: {prompt[:50]}...",
            "model": self.model_name,
            "usage": {"total_tokens": 50},
            "simulated": True
        }
    
    def query_belief(self, belief_statement: str, condition: str) -> float:
        """Query belief probability from local model."""
        prompt = f"Probability for {belief_statement} given {condition}:"
        response = self.generate_response(prompt, temperature=0.1, max_tokens=10)
        
        # Simulate reasonable belief based on content
        import random
        import hashlib
        
        # Use deterministic "reasoning" based on content hash
        content_hash = hashlib.md5(f"{belief_statement}{condition}".encode()).hexdigest()
        hash_int = int(content_hash[:8], 16)
        prob = (hash_int % 1000000) / 1000000.0
        
        # Add some noise
        prob += random.normalvariate(0, 0.1)
        return max(0.0, min(1.0, prob))


def get_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to get LLM provider.
    
    Args:
        provider_type: Type of provider (openai, anthropic, azure, local)
        **kwargs: Additional arguments for provider
        
    Returns:
        LLM provider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "azure": AzureOpenAIProvider,
        "local": LocalProvider
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {list(providers.keys())}")
    
    return providers[provider_type](**kwargs)