"""LLM provider implementations."""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, model: str, **kwargs):
        """Initialize provider.
        
        Args:
            model: Model name/identifier
            **kwargs: Provider-specific configuration
        """
        self.model = model
        self.config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LLM response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available.
        
        Returns:
            True if provider can be used
        """
        pass
    
    def get_provider_name(self) -> str:
        """Get provider name.
        
        Returns:
            Provider name
        """
        return self.__class__.__name__.replace("Provider", "").lower()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, **kwargs):
        """Initialize OpenAI provider.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            **kwargs: Additional OpenAI parameters
        """
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            if "organization" in kwargs:
                openai.organization = kwargs["organization"]
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        return OPENAI_AVAILABLE and self.api_key is not None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LLM response
        """
        if not self.is_available():
            raise RuntimeError("OpenAI provider not available")
        
        # Set default parameters
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }
        
        # Add any additional parameters
        params.update({k: v for k, v in kwargs.items() if k not in params})
        
        try:
            response = openai.ChatCompletion.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content.strip(),
                model=self.model,
                provider="openai",
                usage=dict(response.usage) if hasattr(response, 'usage') else None,
                metadata={"response_id": response.id}
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: Optional[str] = None, **kwargs):
        """Initialize Anthropic provider.
        
        Args:
            model: Claude model name
            api_key: Anthropic API key
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        return ANTHROPIC_AVAILABLE and self.api_key is not None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LLM response
        """
        if not self.is_available():
            raise RuntimeError("Anthropic provider not available")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return LLMResponse(
                content=response.content[0].text.strip(),
                model=self.model,
                provider="anthropic",
                usage=dict(response.usage) if hasattr(response, 'usage') else None,
                metadata={"response_id": response.id}
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI provider."""
    
    def __init__(self, model: str = "gpt-4", **kwargs):
        """Initialize Azure OpenAI provider.
        
        Args:
            model: Model deployment name
            **kwargs: Azure configuration
        """
        super().__init__(model, **kwargs)
        self.api_key = kwargs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = kwargs.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = kwargs.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
        if OPENAI_AVAILABLE and self.api_key and self.endpoint:
            openai.api_type = "azure"
            openai.api_key = self.api_key
            openai.api_base = self.endpoint
            openai.api_version = self.api_version
    
    def is_available(self) -> bool:
        """Check if Azure OpenAI is available."""
        return OPENAI_AVAILABLE and self.api_key is not None and self.endpoint is not None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Azure OpenAI.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LLM response
        """
        if not self.is_available():
            raise RuntimeError("Azure OpenAI provider not available")
        
        params = {
            "engine": self.model,  # Azure uses 'engine' instead of 'model'
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }
        
        try:
            response = openai.ChatCompletion.create(**params)
            
            return LLMResponse(
                content=response.choices[0].message.content.strip(),
                model=self.model,
                provider="azure_openai",
                usage=dict(response.usage) if hasattr(response, 'usage') else None,
                metadata={"response_id": response.id}
            )
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise


class LocalProvider(LLMProvider):
    """Local/self-hosted model provider."""
    
    def __init__(self, model: str, endpoint: str, **kwargs):
        """Initialize local provider.
        
        Args:
            model: Model name
            endpoint: Local API endpoint
            **kwargs: Additional configuration
        """
        super().__init__(model, **kwargs)
        self.endpoint = endpoint
        self.headers = kwargs.get("headers", {})
    
    def is_available(self) -> bool:
        """Check if local provider is available."""
        # Could add endpoint health check here
        return self.endpoint is not None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using local API.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LLM response
        """
        if not self.is_available():
            raise RuntimeError("Local provider not available")
        
        import requests
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }
        
        try:
            response = requests.post(
                f"{self.endpoint}/v1/completions",
                json=payload,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            return LLMResponse(
                content=data["choices"][0]["text"].strip(),
                model=self.model,
                provider="local",
                usage=data.get("usage"),
                metadata={"endpoint": self.endpoint}
            )
        except Exception as e:
            logger.error(f"Local API error: {e}")
            raise


def create_provider(provider_type: str, model: str, **kwargs) -> LLMProvider:
    """Factory function to create LLM providers.
    
    Args:
        provider_type: Type of provider (openai, anthropic, azure, local)
        model: Model name
        **kwargs: Provider-specific configuration
        
    Returns:
        LLM provider instance
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "azure": AzureOpenAIProvider,
        "azure_openai": AzureOpenAIProvider,
        "local": LocalProvider,
    }
    
    if provider_type not in providers:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    provider_class = providers[provider_type]
    return provider_class(model, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available providers.
    
    Returns:
        List of available provider names
    """
    available = []
    
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        available.append("openai")
    
    if OPENAI_AVAILABLE and os.getenv("AZURE_OPENAI_API_KEY"):
        available.append("azure_openai")
    
    if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    
    # Local provider could be available if endpoint is configured
    available.append("local")
    
    return available