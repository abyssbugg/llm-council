"""Provider abstraction layer for LLM providers.

This package defines the base interface that all LLM providers must implement,
enabling interchangeable usage and fallback logic.
"""

from .base import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderConfig,
    ProviderStatus,
    ProviderError,
    RateLimitError,
    AuthenticationError,
)
from .registry import ProviderRegistry, get_registry, reset_registry
from .openrouter_provider import OpenRouterProvider
from .huggingface_provider import HuggingFaceProvider
from .chutes_provider import ChutesProvider

__all__ = [
    "BaseLLMProvider",
    "ProviderResponse",
    "ProviderConfig",
    "ProviderStatus",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ProviderRegistry",
    "get_registry",
    "reset_registry",
    "OpenRouterProvider",
    "HuggingFaceProvider",
    "ChutesProvider",
]
