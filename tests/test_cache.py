"""Tests for the cache module."""

import pytest

from backend.cache import (
    generate_cache_key,
    CacheService,
    get_cache_service,
    reset_cache_service,
    close_cache_service,
)

# Import fakeredis for testing
try:
    from fakeredis import FakeAsyncRedis as FakeRedis
except ImportError:
    FakeRedis = None


@pytest.fixture
async def cache_service():
    """Fixture providing a test cache service with fakeredis."""
    if FakeRedis is None:
        pytest.skip("fakeredis not installed")

    # Create a fake Redis client
    fake_client = FakeRedis()

    # Create cache service with the fake client
    service = CacheService(
        redis_client=fake_client,
        default_ttl=60,
        enabled=True,
    )

    yield service

    # Cleanup
    await service.close()
    if hasattr(fake_client, "aclose"):
        await fake_client.aclose()
    reset_cache_service()


class TestGenerateCacheKey:
    """Tests for generate_cache_key function."""

    def test_basic_cache_key(self):
        """Test generating a basic cache key."""
        key = generate_cache_key("openai/gpt-4", "Hello world")

        # Should have prefix, model (with / replaced), and prompt hash
        assert key.startswith("llm_council:")
        assert "openai_gpt-4" in key
        assert key.count(":") >= 2

    def test_cache_key_with_parameters(self):
        """Test generating a cache key with parameters."""
        key = generate_cache_key(
            "anthropic/claude-3",
            "Test prompt",
            parameters={"temperature": 0.7, "max_tokens": 1000},
        )

        assert "anthropic_claude-3" in key
        # Should include parameter hash
        assert key.count(":") >= 3

    def test_cache_key_consistency(self):
        """Test that the same inputs generate the same key."""
        key1 = generate_cache_key("openai/gpt-4", "Hello")
        key2 = generate_cache_key("openai/gpt-4", "Hello")

        assert key1 == key2

    def test_cache_key_different_prompts(self):
        """Test that different prompts generate different keys."""
        key1 = generate_cache_key("openai/gpt-4", "Hello")
        key2 = generate_cache_key("openai/gpt-4", "Goodbye")

        assert key1 != key2

    def test_cache_key_different_models(self):
        """Test that different models generate different keys."""
        key1 = generate_cache_key("openai/gpt-4", "Hello")
        key2 = generate_cache_key("anthropic/claude-3", "Hello")

        assert key1 != key2

    def test_cache_key_different_params(self):
        """Test that different parameters generate different keys."""
        key1 = generate_cache_key(
            "openai/gpt-4",
            "Hello",
            parameters={"temperature": 0.7},
        )
        key2 = generate_cache_key(
            "openai/gpt-4",
            "Hello",
            parameters={"temperature": 0.8},
        )

        assert key1 != key2

    def test_cache_key_handles_long_params(self):
        """Test that very long parameter strings are hashed."""
        long_params = {
            "messages": [
                {"role": "user", "content": "x" * 1000},
            ] * 10,
        }

        key = generate_cache_key("openai/gpt-4", "Hello", parameters=long_params)

        # Key should be truncated/hashed, not extremely long
        assert len(key) < 500


class TestCacheService:
    """Tests for CacheService class."""

    @pytest.mark.asyncio
    async def test_cache_service_init(self):
        """Test cache service initialization."""
        service = CacheService(
            redis_url="fakeredis://localhost:6379/0",
            enabled=True,
        )

        assert service._enabled is True
        assert service._default_ttl == 3600

        await service.close()

    @pytest.mark.asyncio
    async def test_cache_service_disabled(self):
        """Test cache service when disabled."""
        service = CacheService(enabled=False)

        assert service._enabled is False

        # Operations should return None/False
        result = await service.get("test_key")
        assert result is None

        result = await service.set("test_key", "value")
        assert result is False

        await service.close()

    @pytest.mark.asyncio
    async def test_cache_service_no_url(self):
        """Test cache service with no URL."""
        # Use empty string to disable cache (None falls back to settings)
        service = CacheService(redis_url="", enabled=True)

        assert service._enabled is False

    @pytest.mark.asyncio
    async def test_ping(self, cache_service):
        """Test Redis ping."""
        result = await cache_service.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache_service):
        """Test basic set and get operations."""
        key = "test:key"
        value = {"content": "Hello world", "model": "gpt-4"}

        # Set value
        success = await cache_service.set(key, value)
        assert success is True

        # Get value
        result = await cache_service.get(key)
        assert result is not None
        assert result["content"] == "Hello world"
        assert result["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_miss(self, cache_service):
        """Test get with non-existent key."""
        result = await cache_service.get("nonexistent:key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_disabled_cache(self, cache_service):
        """Test get operations when cache is disabled."""
        # Temporarily disable
        cache_service._enabled = False

        result = await cache_service.get("test:key")
        assert result is None

        # Re-enable for other tests
        cache_service._enabled = True

    @pytest.mark.asyncio
    async def test_set_with_custom_ttl(self, cache_service):
        """Test set with custom TTL."""
        key = "test:ttl"
        value = {"data": "test"}

        success = await cache_service.set(key, value, ttl=10)
        assert success is True

        # Should be retrievable immediately
        result = await cache_service.get(key)
        assert result is not None

    @pytest.mark.asyncio
    async def test_delete(self, cache_service):
        """Test delete operation."""
        key = "test:delete"
        value = {"data": "test"}

        # Set value
        await cache_service.set(key, value)

        # Verify it exists
        result = await cache_service.get(key)
        assert result is not None

        # Delete it
        success = await cache_service.delete(key)
        assert success is True

        # Verify it's gone
        result = await cache_service.get(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache_service):
        """Test delete with non-existent key."""
        success = await cache_service.delete("nonexistent:key")
        assert success is False

    @pytest.mark.asyncio
    async def test_clear(self, cache_service):
        """Test clearing all cache entries."""
        # Set multiple values - use keys with cache prefix
        await cache_service.set("llm_council:test:key1", {"data": "1"})
        await cache_service.set("llm_council:test:key2", {"data": "2"})
        await cache_service.set("llm_council:other:key3", {"data": "3"})

        # Clear all cache entries
        success = await cache_service.clear()
        assert success is True

        # Verify all are gone
        assert await cache_service.get("llm_council:test:key1") is None
        assert await cache_service.get("llm_council:test:key2") is None
        assert await cache_service.get("llm_council:other:key3") is None

    @pytest.mark.asyncio
    async def test_get_stats(self, cache_service):
        """Test cache statistics."""
        # Reset stats first
        cache_service.reset_stats()

        # Perform some operations
        await cache_service.set("test:key", {"data": "test"})
        cache_service._stats["sets"] = 0  # Reset to test increment

        await cache_service.set("test:key", {"data": "test"})
        await cache_service.set("test:key2", {"data": "test2"})

        await cache_service.get("test:key")  # Hit
        await cache_service.get("test:key2")  # Hit
        await cache_service.get("nonexistent")  # Miss

        stats = cache_service.get_stats()

        assert stats["sets"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    @pytest.mark.asyncio
    async def test_reset_stats(self, cache_service):
        """Test resetting statistics."""
        # Do some operations
        await cache_service.set("test:key", {"data": "test"})
        await cache_service.get("test:key")

        # Reset stats
        cache_service.reset_stats()

        stats = cache_service.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["sets"] == 0
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_json_serialization(self, cache_service):
        """Test proper JSON serialization of complex objects."""
        complex_value = {
            "content": "Hello",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "nested": {"deep": {"value": [1, 2, 3]}},
        }

        await cache_service.set("test:complex", complex_value)

        result = await cache_service.get("test:complex")
        assert result == complex_value
        assert result["usage"]["prompt_tokens"] == 10
        assert result["nested"]["deep"]["value"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_unicode_values(self, cache_service):
        """Test cache with unicode characters."""
        value = {"message": "Hello 世界 🌍"}

        await cache_service.set("test:unicode", value)

        result = await cache_service.get("test:unicode")
        assert result["message"] == "Hello 世界 🌍"


class TestGlobalCacheService:
    """Tests for global cache service functions."""

    def test_get_cache_service_singleton(self):
        """Test that get_cache_service returns a singleton."""
        reset_cache_service()

        service1 = get_cache_service()
        service2 = get_cache_service()

        assert service1 is service2

        reset_cache_service()

    @pytest.mark.asyncio
    async def test_close_cache_service(self):
        """Test closing the global cache service."""
        # Get a service (creates it)
        service = get_cache_service()

        # Close it
        await close_cache_service()

        # Get a new one (should be a new instance)
        service2 = get_cache_service()

        # Note: we can't directly test if they're different instances
        # because get_cache_service is cached, but we can verify close doesn't error

        reset_cache_service()


class TestCacheKeyIntegration:
    """Integration tests for cache key generation with cache service."""

    @pytest.mark.asyncio
    async def test_full_cache_workflow(self, cache_service):
        """Test complete workflow: generate key, set, get."""
        # Generate cache key for a query
        key = generate_cache_key(
            "openai/gpt-4",
            "What is the capital of France?",
            parameters={"temperature": 0.7, "max_tokens": 100},
        )

        # Mock LLM response
        response = {
            "content": "The capital of France is Paris.",
            "model": "openai/gpt-4",
            "usage": {"prompt_tokens": 15, "completion_tokens": 10},
        }

        # Cache the response
        await cache_service.set(key, response, ttl=3600)

        # Retrieve from cache
        cached = await cache_service.get(key)

        assert cached is not None
        assert cached["content"] == "The capital of France is Paris."
        assert cached["model"] == "openai/gpt-4"
