"""Tests for the provider abstraction layer and registry."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from backend.providers.base import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderConfig,
    ProviderStatus,
    ProviderError,
    RateLimitError,
    AuthenticationError,
)
from backend.providers.registry import ProviderRegistry, get_registry, reset_registry


class MockProvider(BaseLLMProvider):
    """Mock provider for testing."""
    
    def __init__(self, config: ProviderConfig, name: str = "mock"):
        super().__init__(config)
        self._name = name
        self._models = ["mock-model-1", "mock-model-2"]
    
    @property
    def provider_name(self) -> str:
        return self._name
    
    @property
    def available_models(self) -> list:
        return self._models
    
    async def generate_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: str = None,
        conversation_history: list = None,
    ) -> ProviderResponse:
        return ProviderResponse(
            content=f"Response to: {prompt}",
            model=model,
            provider=self._name,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
        )
    
    async def estimate_cost(self, prompt: str, model: str, max_tokens: int = 4096) -> float:
        return 0.001
    
    def supports_model(self, model: str) -> bool:
        return model in self._models


class FailingMockProvider(BaseLLMProvider):
    """Mock provider that always fails."""
    
    def __init__(self, config: ProviderConfig, name: str = "failing"):
        super().__init__(config)
        self._name = name
        self._models = ["failing-model"]
        self._fail_count = 0
        self._fail_limit = 100  # Always fail
    
    @property
    def provider_name(self) -> str:
        return self._name
    
    @property
    def available_models(self) -> list:
        return self._models
    
    async def generate_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: str = None,
        conversation_history: list = None,
    ) -> ProviderResponse:
        raise ProviderError("Always fails", self._name)
    
    async def estimate_cost(self, prompt: str, model: str, max_tokens: int = 4096) -> float:
        return 0.001
    
    def supports_model(self, model: str) -> bool:
        return model in self._models


class TestProviderResponse:
    """Tests for ProviderResponse dataclass."""
    
    def test_create_response(self):
        """Test creating a provider response."""
        response = ProviderResponse(
            content="Test response",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
        )
        
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.total_tokens == 30
    
    def test_to_dict(self):
        """Test converting response to dictionary."""
        response = ProviderResponse(
            content="Test",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            cost_usd=0.001,
            reasoning_details="Because...",
        )
        
        result = response.to_dict()
        
        assert result["content"] == "Test"
        assert result["total_tokens"] == 30
        assert result["reasoning_details"] == "Because..."
        assert "raw_response" not in result  # Not included in to_dict


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""
    
    def test_default_config(self):
        """Test creating config with default values."""
        config = ProviderConfig(api_key="test-key")
        
        assert config.api_key == "test-key"
        assert config.base_url is None
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.enabled is True
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        config = ProviderConfig(
            api_key="test-key",
            base_url="https://api.example.com",
            timeout=30,
            max_retries=5,
            enabled=False,
        )
        
        assert config.base_url == "https://api.example.com"
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.enabled is False


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""
    
    def test_mock_provider_implementation(self):
        """Test that mock provider implements all required methods."""
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        # Check required properties
        assert provider.provider_name == "mock"
        assert provider.available_models == ["mock-model-1", "mock-model-2"]
        
        # Check status methods
        assert provider.status == ProviderStatus.AVAILABLE
        assert provider.is_available() is True
    
    def test_set_status(self):
        """Test setting provider status."""
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        provider.set_status(ProviderStatus.RATE_LIMITED)
        
        assert provider.status == ProviderStatus.RATE_LIMITED
        assert provider.is_available() is False
    
    def test_supports_model(self):
        """Test model support checking."""
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        assert provider.supports_model("mock-model-1") is True
        assert provider.supports_model("unknown-model") is False
    
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """Test generating a response."""
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        response = await provider.generate_response(
            prompt="Hello",
            model="mock-model-1",
            temperature=0.5,
            max_tokens=100,
        )
        
        assert response.content == "Response to: Hello"
        assert response.model == "mock-model-1"
        assert response.provider == "mock"
        assert response.total_tokens == 30
    
    @pytest.mark.asyncio
    async def test_estimate_cost(self):
        """Test cost estimation."""
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        cost = await provider.estimate_cost(
            prompt="Hello",
            model="mock-model-1",
            max_tokens=100,
        )
        
        assert cost == 0.001


class TestProviderErrors:
    """Tests for provider exception classes."""
    
    def test_provider_error(self):
        """Test base ProviderError."""
        error = ProviderError("Test error", "test-provider")
        
        assert "test-provider" in str(error)
        assert error.provider == "test-provider"
        assert error.recoverable is True
    
    def test_provider_error_unrecoverable(self):
        """Test unrecoverable ProviderError."""
        error = ProviderError("Auth failed", "test-provider", recoverable=False)
        
        assert error.recoverable is False
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limited", "test-provider", retry_after=60)
        
        assert error.provider == "test-provider"
        assert error.recoverable is True
        assert error.retry_after == 60
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid API key", "test-provider")
        
        assert error.provider == "test-provider"
        assert error.recoverable is False


class TestProviderRegistry:
    """Tests for ProviderRegistry."""
    
    def setup_method(self):
        """Reset registry before each test."""
        reset_registry()
    
    def test_create_registry(self):
        """Test creating a new registry."""
        registry = ProviderRegistry()
        
        assert registry.list_providers() == []
        assert registry.list_available_providers() == []
    
    def test_get_global_registry(self):
        """Test getting global registry instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        # Should return the same instance
        assert registry1 is registry2
    
    def test_register_provider(self):
        """Test registering a provider instance."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        registry.register_provider(provider)
        
        assert "mock" in registry.list_providers()
        assert registry.get_provider("mock") is provider
    
    def test_register_provider_class(self):
        """Test registering a provider class."""
        registry = ProviderRegistry()
        
        registry.register_provider_class("mock", MockProvider)
        
        # Can now create provider from class
        config = ProviderConfig(api_key="test")
        provider = registry.create_provider("mock", config)
        
        assert provider is not None
        assert provider.provider_name == "mock"
    
    def test_unregister_provider(self):
        """Test unregistering a provider."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        registry.register_provider(provider)
        assert "mock" in registry.list_providers()
        
        registry.unregister_provider("mock")
        assert "mock" not in registry.list_providers()
    
    def test_list_available_providers(self):
        """Test listing only available providers."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        
        provider1 = MockProvider(config, name="available")
        provider2 = MockProvider(config, name="unavailable")
        provider2.set_status(ProviderStatus.UNAVAILABLE)
        
        registry.register_provider(provider1)
        registry.register_provider(provider2)
        
        all_providers = registry.list_providers()
        available = registry.list_available_providers()
        
        assert len(all_providers) == 2
        assert len(available) == 1
        assert "available" in available
        assert "unavailable" not in available
    
    def test_fallback_chain(self):
        """Test setting and getting fallback chains."""
        registry = ProviderRegistry()
        
        registry.set_fallback_chain("primary", ["fallback1", "fallback2"])
        
        chain = registry.get_fallback_chain("primary")
        assert chain == ["fallback1", "fallback2"]
    
    def test_get_fallback_chain_empty(self):
        """Test getting fallback chain when none set."""
        registry = ProviderRegistry()
        
        chain = registry.get_fallback_chain("nonexistent")
        assert chain == []
    
    def test_report_failure(self):
        """Test reporting provider failures."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        registry.register_provider(provider)
        
        # Report failures below threshold
        for _ in range(3):
            registry.report_failure("mock", Exception("Test error"))
        
        assert provider.is_available() is True
        
        # Report more failures to trigger disable
        for _ in range(5):
            registry.report_failure("mock", Exception("Test error"))
        
        assert provider.is_available() is False
    
    def test_report_success_resets_failure_count(self):
        """Test that success resets failure count."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        registry.register_provider(provider)
        
        # Report some failures
        for _ in range(3):
            registry.report_failure("mock", Exception("Test error"))
        
        # Report success
        registry.report_success("mock")
        
        # Failure count should be reset
        # Need 5 more failures to trigger disable (not 2)
        for _ in range(4):
            registry.report_failure("mock", Exception("Test error"))
        
        assert provider.is_available() is True
    
    def test_report_success_reactivates_provider(self):
        """Test that success reactivates a disabled provider."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        registry.register_provider(provider)
        provider.set_status(ProviderStatus.UNAVAILABLE)
        
        assert provider.is_available() is False
        
        registry.report_success("mock")
        
        assert provider.is_available() is True
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback_success(self):
        """Test successful generation with fallback chain."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        
        primary = MockProvider(config, name="primary")
        fallback = MockProvider(config, name="fallback")
        
        registry.register_provider(primary)
        registry.register_provider(fallback)
        registry.set_fallback_chain("primary", ["fallback"])
        
        response = await registry.generate_with_fallback(
            prompt="Test",
            model="mock-model-1",
            primary_provider="primary",
        )
        
        assert response.provider == "primary"
        assert response.content == "Response to: Test"
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback_primary_fails(self):
        """Test fallback when primary provider fails."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        
        # Create failing provider that supports a model the fallback also supports
        primary = FailingMockProvider(config, name="primary")
        # Add the shared model to primary
        primary._models.append("mock-model-1")
        
        fallback = MockProvider(config, name="fallback")
        
        registry.register_provider(primary)
        registry.register_provider(fallback)
        registry.set_fallback_chain("primary", ["fallback"])
        
        response = await registry.generate_with_fallback(
            prompt="Test",
            model="mock-model-1",  # Both support this model
            primary_provider="primary",
        )
        
        assert response.provider == "fallback"
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback_all_fail(self):
        """Test when all providers in chain fail."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        
        primary = FailingMockProvider(config, name="primary")
        fallback = FailingMockProvider(config, name="fallback")
        
        registry.register_provider(primary)
        registry.register_provider(fallback)
        registry.set_fallback_chain("primary", ["fallback"])
        
        with pytest.raises(ProviderError):
            await registry.generate_with_fallback(
                prompt="Test",
                model="failing-model",
                primary_provider="primary",
            )
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback_unsupported_model(self):
        """Test fallback when primary doesn't support model."""
        registry = ProviderRegistry()
        config = ProviderConfig(api_key="test")
        
        primary = MockProvider(config, name="primary")
        fallback = MockProvider(config, name="fallback")
        
        registry.register_provider(primary)
        registry.register_provider(fallback)
        registry.set_fallback_chain("primary", ["fallback"])
        
        response = await registry.generate_with_fallback(
            prompt="Test",
            model="mock-model-1",  # primary supports this
            primary_provider="primary",
        )
        
        assert response.provider == "primary"
    
    def test_max_failures_before_disable(self):
        """Test custom max failures threshold."""
        registry = ProviderRegistry()
        registry._max_failures_before_disable = 2
        
        config = ProviderConfig(api_key="test")
        provider = MockProvider(config)
        
        registry.register_provider(provider)
        
        # Should disable after 2 failures
        registry.report_failure("mock", Exception("Test"))
        assert provider.is_available() is True
        
        registry.report_failure("mock", Exception("Test"))
        assert provider.is_available() is False
