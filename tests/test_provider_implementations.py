"""Tests for specific provider implementations."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from backend.providers import (
    ProviderConfig,
    ProviderResponse,
    OpenRouterProvider,
    HuggingFaceProvider,
    ChutesProvider,
)
from backend.providers.base import ProviderError, RateLimitError, AuthenticationError


class TestOpenRouterProvider:
    """Tests for OpenRouter provider."""
    
    @pytest.fixture
    def config(self):
        return ProviderConfig(api_key="test-or-key")
    
    @pytest.fixture
    def provider(self, config):
        return OpenRouterProvider(config)
    
    def test_provider_name(self, provider):
        assert provider.provider_name == "openrouter"
    
    def test_available_models(self, provider):
        assert "anthropic/claude-3-sonnet" in provider.available_models
        assert "openai/gpt-4o" in provider.available_models
    
    def test_supports_model(self, provider):
        assert provider.supports_model("anthropic/claude-3-sonnet") is True
        assert provider.supports_model("gpt-4o") is True  # Without provider prefix
        assert provider.supports_model("unknown-model") is False
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, provider):
        """Test successful response generation."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Test response",
                        "role": "assistant",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "headers": {
                "x-model-cost": "0.0001",
            },
        }
        
        with patch("backend.providers.openrouter_provider._make_request", return_value=mock_response):
            response = await provider.generate_response(
                prompt="Hello",
                model="anthropic/claude-3-sonnet",
            )
            
            assert response.content == "Test response"
            assert response.provider == "openrouter"
            assert response.total_tokens == 30
            assert response.cost_usd == 0.0001
    
    @pytest.mark.asyncio
    async def test_generate_response_with_system_prompt(self, provider):
        """Test response generation with system prompt."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Response with context",
                    },
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }
        
        with patch("backend.providers.openrouter_provider._make_request", return_value=mock_response):
            response = await provider.generate_response(
                prompt="Hello",
                model="openai/gpt-4o",
                system_prompt="You are a helpful assistant.",
            )
            
            assert response.content == "Response with context"
    
    @pytest.mark.asyncio
    async def test_generate_response_with_history(self, provider):
        """Test response generation with conversation history."""
        mock_response = {
            "choices": [{"message": {"content": "Contextual response"}}],
            "usage": {"prompt_tokens": 25, "completion_tokens": 15, "total_tokens": 40},
        }
        
        history = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First response"},
        ]
        
        with patch("backend.providers.openrouter_provider._make_request", return_value=mock_response):
            response = await provider.generate_response(
                prompt="Second message",
                model="openai/gpt-4o",
                conversation_history=history,
            )
            
            assert response.content == "Contextual response"
    
    @pytest.mark.asyncio
    async def test_generate_response_authentication_error(self, provider):
        """Test handling of authentication errors."""
        import httpx
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        
        error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=mock_response)
        
        with patch("backend.providers.openrouter_provider._make_request", side_effect=error):
            with pytest.raises(AuthenticationError):
                await provider.generate_response("Hello", "openai/gpt-4o")
    
    @pytest.mark.asyncio
    async def test_generate_response_rate_limit(self, provider):
        """Test handling of rate limit errors."""
        import httpx
        
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {"Retry-After": "60"}
        
        error = httpx.HTTPStatusError("Rate limited", request=Mock(), response=mock_response)
        
        with patch("backend.providers.openrouter_provider._make_request", side_effect=error):
            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate_response("Hello", "openai/gpt-4o")
            
            assert exc_info.value.retry_after == 60
    
    @pytest.mark.asyncio
    async def test_estimate_cost(self, provider):
        """Test cost estimation."""
        cost = await provider.estimate_cost(
            prompt="Hello world, this is a test prompt.",
            model="openai/gpt-4o",
            max_tokens=1000,
        )
        
        # Cost should be positive
        assert cost >= 0
    
    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Test closing the provider."""
        # Create a mock client
        mock_client = Mock()
        mock_client.aclose = AsyncMock()
        provider._client = mock_client
        
        await provider.close()
        
        # Verify close was called and client was set to None
        mock_client.aclose.assert_called_once()
        assert provider._client is None


class TestHuggingFaceProvider:
    """Tests for HuggingFace provider."""
    
    @pytest.fixture
    def config(self):
        return ProviderConfig(api_key="test-hf-key")
    
    @pytest.fixture
    def provider(self, config):
        # Skip if huggingface-hub not installed
        try:
            return HuggingFaceProvider(config)
        except ImportError:
            pytest.skip("huggingface-hub not installed")
    
    def test_provider_name(self, provider):
        assert provider.provider_name == "huggingface"
    
    def test_available_models(self, provider):
        assert "meta-llama/Llama-3-70b-chat-hf" in provider.available_models
        assert "mistralai/Mistral-7B-Instruct-v0.2" in provider.available_models
    
    def test_supports_model(self, provider):
        assert provider.supports_model("meta-llama/Llama-3-70b-chat-hf") is True
        assert provider.supports_model("Llama-3-70b-chat-hf") is True
        assert provider.supports_model("unknown-model") is False
    
    def test_build_prompt(self, provider):
        """Test prompt building with various components."""
        # Simple prompt
        result = provider._build_prompt("Hello", None, None)
        assert "Hello" in result
        assert "User:" in result
        
        # With system prompt
        result = provider._build_prompt(
            "Hello",
            "You are helpful.",
            None,
        )
        assert "System: You are helpful." in result
        
        # With conversation history
        history = [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response"},
        ]
        result = provider._build_prompt("Hello", None, history)
        assert "User: First" in result
        assert "Assistant: Response" in result
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, provider):
        """Test successful response generation."""
        with patch.object(provider, "_get_client") as mock_client_get:
            mock_client = AsyncMock()
            mock_client.text_generation = AsyncMock(return_value="Generated text")
            mock_client_get.return_value = mock_client
            
            response = await provider.generate_response(
                prompt="Hello",
                model="meta-llama/Llama-3-8b-chat-hf",
            )
            
            assert response.content == "Generated text"
            assert response.provider == "huggingface"
            assert response.total_tokens > 0
    
    @pytest.mark.asyncio
    async def test_estimate_cost(self, provider):
        """Test cost estimation (should be free for serverless)."""
        cost = await provider.estimate_cost("Hello", "meta-llama/Llama-3-8b-chat-hf")
        
        # HuggingFace serverless is free
        assert cost == 0.0


class TestChutesProvider:
    """Tests for Chutes AI provider."""
    
    @pytest.fixture
    def config(self):
        return ProviderConfig(api_key="test-chutes-key")
    
    @pytest.fixture
    def provider(self, config):
        return ChutesProvider(config)
    
    def test_provider_name(self, provider):
        assert provider.provider_name == "chutes"
    
    def test_available_models(self, provider):
        assert "gpt-4o" in provider.available_models
        assert "claude-3-sonnet" in provider.available_models
        assert "llama-3-70b" in provider.available_models
    
    def test_supports_model(self, provider):
        # Direct Chutes format
        assert provider.supports_model("gpt-4o") is True
        assert provider.supports_model("claude-3-sonnet") is True
        
        # OpenRouter format
        assert provider.supports_model("openai/gpt-4o") is True
        assert provider.supports_model("anthropic/claude-3-sonnet") is True
        
        # Unknown model
        assert provider.supports_model("unknown-model") is False
    
    def test_normalize_model_id(self, provider):
        """Test model ID normalization."""
        # Already normalized
        assert provider._normalize_model_id("gpt-4o") == "gpt-4o"
        
        # OpenRouter format
        assert provider._normalize_model_id("openai/gpt-4o") == "gpt-4o"
        assert provider._normalize_model_id("anthropic/claude-3-sonnet") == "claude-3-sonnet"
        
        # Full path
        assert provider._normalize_model_id("providers/openai/gpt-4o") == "gpt-4o"
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, provider):
        """Test successful response generation."""
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Chutes response",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "cost": 0.0005,
        }
        
        with patch("backend.providers.chutes_provider._make_request", return_value=mock_response):
            response = await provider.generate_response(
                prompt="Hello",
                model="gpt-4o",
            )
            
            assert response.content == "Chutes response"
            assert response.provider == "chutes"
            assert response.total_tokens == 30
            assert response.cost_usd == 0.0005
    
    @pytest.mark.asyncio
    async def test_generate_response_with_openrouter_model(self, provider):
        """Test using OpenRouter model ID format."""
        mock_response = {
            "choices": [{"message": {"content": "Response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "cost": 0.0003,
        }
        
        with patch("backend.providers.chutes_provider._make_request", return_value=mock_response):
            response = await provider.generate_response(
                prompt="Hello",
                model="anthropic/claude-3-sonnet",  # OpenRouter format
            )
            
            assert response.content == "Response"
            # Model in response should be the original requested model
            assert response.model == "anthropic/claude-3-sonnet"
    
    @pytest.mark.asyncio
    async def test_generate_response_authentication_error(self, provider):
        """Test handling of authentication errors."""
        import httpx
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        
        error = httpx.HTTPStatusError("Unauthorized", request=Mock(), response=mock_response)
        
        with patch("backend.providers.chutes_provider._make_request", side_effect=error):
            with pytest.raises(AuthenticationError):
                await provider.generate_response("Hello", "gpt-4o")
    
    @pytest.mark.asyncio
    async def test_estimate_cost(self, provider):
        """Test cost estimation."""
        cost = await provider.estimate_cost(
            prompt="Hello world, this is a test.",
            model="gpt-4o",
            max_tokens=1000,
        )
        
        assert cost >= 0
    
    @pytest.mark.asyncio
    async def test_close(self, provider):
        """Test closing the provider."""
        mock_client = Mock()
        mock_client.aclose = AsyncMock()
        provider._client = mock_client
        
        await provider.close()
        
        # Verify close was called and client was set to None
        mock_client.aclose.assert_called_once()
        assert provider._client is None


class TestProviderIntegration:
    """Integration tests for providers with registry."""
    
    def test_register_providers(self):
        """Test registering all provider types."""
        from backend.providers import get_registry
        
        registry = get_registry()
        
        # Register provider classes
        registry.register_provider_class("openrouter", OpenRouterProvider)
        registry.register_provider_class("chutes", ChutesProvider)
        
        # Create instances
        config = ProviderConfig(api_key="test-key")
        
        or_provider = registry.create_provider("openrouter", config)
        assert or_provider is not None
        assert or_provider.provider_name == "openrouter"
        
        chutes_provider = registry.create_provider("chutes", config)
        assert chutes_provider is not None
        assert chutes_provider.provider_name == "chutes"
    
    def test_fallback_chain_across_providers(self):
        """Test fallback chain with different provider types."""
        from backend.providers import get_registry, reset_registry
        
        reset_registry()
        registry = get_registry()
        
        config = ProviderConfig(api_key="test-key")
        
        primary = ChutesProvider(config)
        fallback1 = OpenRouterProvider(config)
        
        registry.register_provider(primary)
        registry.register_provider(fallback1)
        
        # Try to add HF if available
        try:
            fallback2 = HuggingFaceProvider(config)
            registry.register_provider(fallback2)
            has_hf = True
        except ImportError:
            has_hf = False
        
        # Set fallback chain from Chutes -> OpenRouter -> HuggingFace
        chain = ["openrouter"]
        if has_hf:
            chain.append("huggingface")
        registry.set_fallback_chain("chutes", chain)
        
        retrieved_chain = registry.get_fallback_chain("chutes")
        assert "openrouter" in retrieved_chain
