"""Response caching with Redis.

This module provides caching functionality for LLM responses to reduce
API calls and improve response times for repeated queries.
"""

import json
import logging
import hashlib
from typing import Optional, Any, Dict
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .config import settings

logger = logging.getLogger(__name__)


# Cache key prefix (no trailing separator to avoid double colons)
CACHE_PREFIX = "llm_council"
CACHE_KEY_SEPARATOR = ":"


def generate_cache_key(
    model: str,
    prompt: str,
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a consistent cache key for an LLM query.

    Args:
        model: Model identifier.
        prompt: The user prompt.
        parameters: Optional model parameters (temperature, max_tokens, etc).

    Returns:
        A consistent cache key string.

    Examples:
        >>> generate_cache_key("openai/gpt-4", "Hello world")
        'llm_council:openai_gpt-4:a1b2c3d4'

        >>> generate_cache_key("anthropic/claude-3", "Hello", {"temperature": 0.7})
        'llm_council:anthropic_claude-3:e5f6g7h8:temp_0.7'
    """
    # Sanitize model name for cache key (replace / with _)
    safe_model = model.replace("/", "_").replace(":", "_")

    # Hash the prompt (for length and special characters)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    # Build key components
    key_parts = [CACHE_PREFIX, safe_model, prompt_hash]

    # Add parameters to key if provided
    if parameters:
        # Sort parameters for consistency
        sorted_params = sorted(parameters.items())
        # Create a param string
        param_parts = [f"{k}_{v}" for k, v in sorted_params]
        param_str = "_".join(param_parts)
        # Hash params if too long
        if len(param_str) > 50:
            param_str = hashlib.sha256(param_str.encode()).hexdigest()[:16]
        key_parts.append(param_str)

    return CACHE_KEY_SEPARATOR.join(key_parts)


class CacheService:
    """Service for caching LLM responses using Redis.

    Features:
    - Async Redis connection pooling
    - Configurable TTL per cache entry
    - JSON serialization for complex responses
    - Graceful degradation on Redis errors
    - Cache statistics (hits, misses, errors)
    """

    def __init__(
        self,
        redis_url: str = None,
        default_ttl: int = 3600,
        enabled: bool = True,
        redis_client: Redis = None,
    ):
        """Initialize the cache service.

        Args:
            redis_url: Redis connection URL (defaults to settings.REDIS_URL).
            default_ttl: Default TTL in seconds (defaults to 1 hour).
            enabled: Whether caching is enabled (defaults to True).
            redis_client: Optional pre-configured Redis client (for testing).
        """
        self._redis_url = redis_url if redis_url is not None else settings.redis_url
        self._default_ttl = default_ttl
        self._enabled = enabled and bool(self._redis_url or redis_client)
        self._client: Optional[Redis] = redis_client

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0,
        }

        logger.info(
            f"CacheService initialized: enabled={self._enabled}, "
            f"redis_url={'***' if self._redis_url else 'None'}, "
            f"default_ttl={self._default_ttl}s"
        )

    async def _get_client(self) -> Redis:
        """Get or create the Redis client.

        Returns:
            Redis client instance.
        """
        if self._client is None:
            if not self._redis_url:
                raise RedisError("Redis URL not configured")

            # Parse URL and create connection pool
            self._client = Redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
            )
            logger.info("Redis client created")

        return self._client

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client:
            if hasattr(self._client, "aclose"):
                await self._client.aclose()
            else:
                await self._client.close()
            self._client = None
            logger.info("Redis connection closed")

    async def ping(self) -> bool:
        """Check if Redis is available.

        Returns:
            True if Redis is reachable, False otherwise.
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            return await client.ping()
        except RedisError as e:
            logger.warning(f"Redis ping failed: {e}")
            return False

    async def get(
        self,
        key: str,
    ) -> Optional[Any]:
        """Get a value from the cache.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found or error occurs.
        """
        if not self._enabled:
            return None

        try:
            client = await self._get_client()
            value = await client.get(key)

            if value is None:
                self._stats["misses"] += 1
                logger.debug(f"Cache miss: {key}")
                return None

            self._stats["hits"] += 1
            logger.debug(f"Cache hit: {key}")

            # Deserialize JSON
            return json.loads(value)

        except RedisError as e:
            self._stats["errors"] += 1
            logger.warning(f"Cache get error for {key}: {e}")
            return None
        except json.JSONDecodeError as e:
            self._stats["errors"] += 1
            logger.warning(f"Cache JSON decode error for {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache (will be JSON serialized).
            ttl: Time to live in seconds (defaults to default_ttl).

        Returns:
            True if successful, False otherwise.
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()

            # Serialize value to JSON
            serialized = json.dumps(value)

            # Set with TTL
            ttl_seconds = ttl if ttl is not None else self._default_ttl
            await client.setex(key, ttl_seconds, serialized)

            self._stats["sets"] += 1
            logger.debug(f"Cache set: {key} (TTL: {ttl_seconds}s)")
            return True

        except RedisError as e:
            self._stats["errors"] += 1
            logger.warning(f"Cache set error for {key}: {e}")
            return False

    async def delete(
        self,
        key: str,
    ) -> bool:
        """Delete a value from the cache.

        Args:
            key: Cache key.

        Returns:
            True if key was deleted, False otherwise.
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()
            result = await client.delete(key)
            logger.debug(f"Cache delete: {key} (deleted: {result > 0})")
            return result > 0

        except RedisError as e:
            self._stats["errors"] += 1
            logger.warning(f"Cache delete error for {key}: {e}")
            return False

    async def clear(self) -> bool:
        """Clear all cached values with the cache prefix.

        Returns:
            True if successful, False otherwise.
        """
        if not self._enabled:
            return False

        try:
            client = await self._get_client()

            # Find all keys with the cache prefix
            pattern = f"{CACHE_PREFIX}*"
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")

            return True

        except RedisError as e:
            self._stats["errors"] += 1
            logger.warning(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, errors, sets).
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0,
        }


# Singleton cache service instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get the global cache service instance.

    Returns:
        The global CacheService instance.
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        logger.info("Initialized global cache service")
    return _cache_service


async def reset_cache_service() -> None:
    """Reset the global cache service (useful for testing)."""
    global _cache_service
    if _cache_service:
        await _cache_service.close()
    _cache_service = None
    logger.info("Reset global cache service")


async def close_cache_service() -> None:
    """Close the global cache service connection."""
    global _cache_service
    if _cache_service:
        await _cache_service.close()
        _cache_service = None
