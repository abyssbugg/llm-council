"""Provider Registry for managing LLM provider instances.

This module implements the Registry pattern for dynamically managing
multiple LLM providers with fallback support.
"""

from typing import Dict, List, Optional, Type, Callable
import logging
from .base import (
    BaseLLMProvider,
    ProviderConfig,
    ProviderResponse,
    ProviderStatus,
    ProviderError,
)

# Import cache service
from ..cache import generate_cache_key, get_cache_service

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for managing LLM provider instances.
    
    The registry allows dynamic registration and retrieval of providers,
    supports fallback chains, and tracks provider health status.
    """
    
    def __init__(self) -> None:
        """Initialize an empty provider registry."""
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._provider_classes: Dict[str, Type[BaseLLMProvider]] = {}
        self._fallback_chains: Dict[str, List[str]] = {}
        self._failure_counts: Dict[str, int] = {}
        self._max_failures_before_disable: int = 5
    
    def register_provider_class(
        self,
        name: str,
        provider_class: Type[BaseLLMProvider],
    ) -> None:
        """Register a provider class for later instantiation.
        
        Args:
            name: Unique name for this provider type.
            provider_class: The provider class to register.
        """
        self._provider_classes[name] = provider_class
        logger.info(f"Registered provider class: {name}")
    
    def register_provider(
        self,
        provider: BaseLLMProvider,
        priority: int = 0,
    ) -> None:
        """Register a provider instance.
        
        Args:
            provider: The provider instance to register.
            priority: Priority for this provider (higher = preferred).
        """
        name = provider.provider_name
        self._providers[name] = provider
        self._failure_counts[name] = 0
        logger.info(f"Registered provider instance: {name} (priority: {priority})")
    
    def unregister_provider(self, name: str) -> None:
        """Unregister a provider by name.
        
        Args:
            name: Name of the provider to unregister.
        """
        if name in self._providers:
            del self._providers[name]
            del self._failure_counts[name]
            logger.info(f"Unregistered provider: {name}")
    
    def get_provider(self, name: str) -> Optional[BaseLLMProvider]:
        """Get a registered provider by name.
        
        Args:
            name: Name of the provider to retrieve.
        
        Returns:
            The provider instance, or None if not found.
        """
        return self._providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List names of all registered providers.
        
        Returns:
            List of provider names.
        """
        return list(self._providers.keys())
    
    def list_available_providers(self) -> List[str]:
        """List names of available (not disabled) providers.
        
        Returns:
            List of available provider names.
        """
        return [
            name for name, provider in self._providers.items()
            if provider.is_available()
        ]
    
    def set_fallback_chain(
        self,
        primary: str,
        fallbacks: List[str],
    ) -> None:
        """Configure a fallback chain for a provider.
        
        Args:
            primary: Primary provider name.
            fallbacks: List of fallback provider names in order.
        """
        self._fallback_chains[primary] = fallbacks
        logger.info(f"Set fallback chain for {primary}: {fallbacks}")
    
    def get_fallback_chain(self, primary: str) -> List[str]:
        """Get the fallback chain for a provider.
        
        Args:
            primary: Primary provider name.
        
        Returns:
            List of fallback provider names, or empty list if none configured.
        """
        return self._fallback_chains.get(primary, [])
    
    def report_failure(self, provider_name: str, error: Exception) -> None:
        """Report a provider failure and potentially disable it.
        
        Args:
            provider_name: Name of the provider that failed.
            error: The error that occurred.
        """
        self._failure_counts[provider_name] = (
            self._failure_counts.get(provider_name, 0) + 1
        )
        
        count = self._failure_counts[provider_name]
        logger.warning(
            f"Provider {provider_name} failed (count: {count}): {error}"
        )
        
        # Disable provider after too many failures
        if count >= self._max_failures_before_disable:
            provider = self.get_provider(provider_name)
            if provider:
                provider.set_status(ProviderStatus.UNAVAILABLE)
                logger.error(
                    f"Provider {provider_name} disabled after {count} failures"
                )
    
    def report_success(self, provider_name: str) -> None:
        """Report a successful request from a provider.
        
        This resets the failure count and marks the provider as available.
        
        Args:
            provider_name: Name of the provider that succeeded.
        """
        if provider_name in self._failure_counts:
            self._failure_counts[provider_name] = 0
        
        provider = self.get_provider(provider_name)
        if provider and not provider.is_available():
            provider.set_status(ProviderStatus.AVAILABLE)
            logger.info(f"Provider {provider_name} restored to available status")
    
    def create_provider(
        self,
        provider_type: str,
        config: ProviderConfig,
    ) -> Optional[BaseLLMProvider]:
        """Create a new provider instance from a registered class.
        
        Args:
            provider_type: Type name of the provider to create.
            config: Configuration for the provider.
        
        Returns:
            New provider instance, or None if type not registered.
        """
        provider_class = self._provider_classes.get(provider_type)
        if not provider_class:
            logger.error(f"Unknown provider type: {provider_type}")
            return None
        
        try:
            provider = provider_class(config)
            logger.info(f"Created provider instance: {provider_type}")
            return provider
        except Exception as e:
            logger.error(f"Failed to create provider {provider_type}: {e}")
            return None
    
    async def generate_with_fallback(
        self,
        prompt: str,
        model: str,
        primary_provider: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True,
    ) -> ProviderResponse:
        """Generate a response with automatic fallback and optional caching.

        Attempts to generate a response using the primary provider.
        If it fails, tries each fallback in order until one succeeds.

        If caching is enabled, checks the cache before making API calls.

        Args:
            prompt: The user's prompt.
            model: Model identifier.
            primary_provider: Primary provider to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system prompt.
            conversation_history: Optional conversation history.
            use_cache: Whether to check cache before API calls (default: True).

        Returns:
            ProviderResponse from the first successful provider.

        Raises:
            ProviderError: If all providers fail.
        """
        # Check cache first if enabled
        cache_key = None
        if use_cache:
            cache = get_cache_service()
            cache_key = generate_cache_key(
                model,
                prompt,
                parameters={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "system_prompt": system_prompt,
                },
            )

            cached_response = await cache.get(cache_key)
            if cached_response is not None:
                logger.info(f"Cache hit for {model}, returning cached response")
                return ProviderResponse(
                    content=cached_response["content"],
                    model=cached_response["model"],
                    provider=cached_response.get("provider", primary_provider),
                    prompt_tokens=cached_response.get("prompt_tokens", 0),
                    completion_tokens=cached_response.get("completion_tokens", 0),
                    total_tokens=cached_response.get("total_tokens", 0),
                    cost_usd=cached_response.get("cost_usd", 0.0),
                    reasoning_details=cached_response.get("reasoning_details"),
                    raw_response=cached_response.get("raw_response"),
                    cached=True,  # Mark as from cache
                )

        providers_to_try = [primary_provider] + self.get_fallback_chain(primary_provider)
        last_error: Optional[Exception] = None

        for provider_name in providers_to_try:
            provider = self.get_provider(provider_name)
            if not provider or not provider.is_available():
                logger.warning(f"Provider {provider_name} not available, skipping")
                continue

            if not provider.supports_model(model):
                logger.warning(
                    f"Provider {provider_name} does not support model {model}"
                )
                continue

            try:
                logger.info(f"Attempting generation with {provider_name}")
                response = await provider.generate_response(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                    conversation_history=conversation_history,
                )
                self.report_success(provider_name)
                logger.info(f"Generation succeeded with {provider_name}")

                # Cache the response if caching is enabled
                if use_cache and cache_key:
                    cache = get_cache_service()
                    await cache.set(cache_key, {
                        "content": response.content,
                        "model": response.model,
                        "provider": response.provider,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.total_tokens,
                        "cost_usd": response.cost_usd,
                        "reasoning_details": response.reasoning_details,
                        "raw_response": response.raw_response,
                    })
                    logger.debug(f"Cached response for {model}")

                return response

            except ProviderError as e:
                last_error = e
                self.report_failure(provider_name, e)

                if not e.recoverable:
                    logger.error(
                        f"Provider {provider_name} has unrecoverable error, "
                        f"removing from fallback chain"
                    )
                    break

            except Exception as e:
                last_error = e
                self.report_failure(provider_name, e)

        # All providers failed
        error_msg = f"All providers failed for model {model}"
        if last_error:
            error_msg += f". Last error: {last_error}"
        raise ProviderError(error_msg, primary_provider, recoverable=True)


# Global registry instance
_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry instance.
    
    Creates the registry if it doesn't exist.
    
    Returns:
        The global ProviderRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
        logger.info("Initialized global provider registry")
    return _registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    _registry = None
    logger.info("Reset global provider registry")
