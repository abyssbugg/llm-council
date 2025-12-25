"""Provider abstraction layer for LLM providers.

This module defines the base interface that all LLM providers must implement,
enabling interchangeable usage and fallback logic.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class ProviderStatus(Enum):
    """Status of a provider."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"


@dataclass
class ProviderResponse:
    """Unified response format from all providers."""

    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    reasoning_details: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    cached: bool = False  # True if response came from cache

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "reasoning_details": self.reasoning_details,
            "cached": self.cached,
        }


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    
    api_key: str
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    enabled: bool = True


class BaseLLMProvider(ABC):
    """Base abstract class for all LLM providers.
    
    All providers must inherit from this class and implement the required methods.
    This enables the registry pattern and makes providers interchangeable.
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider configuration including API key and settings.
        """
        self.config = config
        self._status = ProviderStatus.AVAILABLE
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique name of this provider."""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Return list of available model identifiers for this provider."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ProviderResponse:
        """Generate a response from the LLM.
        
        Args:
            prompt: The user's prompt/question.
            model: The model identifier to use.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system prompt to guide behavior.
            conversation_history: Optional list of previous messages for context.
        
        Returns:
            ProviderResponse with content, token usage, and cost.
        
        Raises:
            ProviderError: If the request fails after retries.
        """
        pass
    
    @abstractmethod
    async def estimate_cost(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
    ) -> float:
        """Estimate the cost of a request before making it.
        
        Args:
            prompt: The user's prompt.
            model: The model identifier.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            Estimated cost in USD.
        """
        pass
    
    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: The model identifier to check.
        
        Returns:
            True if the model is supported, False otherwise.
        """
        pass
    
    @property
    def status(self) -> ProviderStatus:
        """Get the current status of this provider."""
        return self._status
    
    def set_status(self, status: ProviderStatus) -> None:
        """Set the status of this provider.
        
        This is used by the registry to mark providers as unavailable
        when they consistently fail.
        """
        self._status = status
    
    def is_available(self) -> bool:
        """Check if this provider is currently available."""
        return self._status == ProviderStatus.AVAILABLE


class ProviderError(Exception):
    """Base exception for provider errors."""
    
    def __init__(self, message: str, provider: str, recoverable: bool = True):
        """Initialize the error.
        
        Args:
            message: Error message.
            provider: Name of the provider that raised the error.
            recoverable: Whether this error is recoverable (e.g., rate limit)
                or permanent (e.g., invalid API key).
        """
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.recoverable = recoverable


class RateLimitError(ProviderError):
    """Raised when a provider rate limits the request."""
    
    def __init__(self, message: str, provider: str, retry_after: Optional[int] = None):
        """Initialize the rate limit error.
        
        Args:
            message: Error message.
            provider: Name of the provider.
            retry_after: Seconds to wait before retrying, if provided.
        """
        super().__init__(message, provider, recoverable=True)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when provider authentication fails."""
    
    def __init__(self, message: str, provider: str):
        """Initialize the authentication error.
        
        Args:
            message: Error message.
            provider: Name of the provider.
        """
        super().__init__(message, provider, recoverable=False)
