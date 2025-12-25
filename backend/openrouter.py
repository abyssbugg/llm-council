"""OpenRouter API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL

logger = logging.getLogger(__name__)


def _should_retry_http_error(exception: Exception) -> bool:
    """
    Check if an HTTP error should trigger a retry.

    Retries on:
    - 429 (Rate Limit)
    - 500, 502, 503, 504 (Server errors)

    Does not retry on:
    - 4xx client errors (except 429)
    - Network errors that won't recover
    """
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        # Retry on rate limiting and server errors
        return status_code in {429, 500, 502, 503, 504}
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.HTTPStatusError) & retry_if_exception_type(_should_retry_http_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API with automatic retry.

    Retries on:
    - 429 (Rate Limit)
    - 500, 502, 503, 504 (Server errors)

    Uses exponential backoff with jitter (1s -> 2s -> 4s... max 10s)

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            return {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }

    except httpx.HTTPStatusError as e:
        # Log and let tenacity decide whether to retry
        logger.warning(
            f"HTTP error querying model {model}: {e.response.status_code} - {e.response.text}"
        )
        raise  # Re-raise for tenacity to handle retry logic

    except Exception as e:
        # Non-retryable errors - return None immediately
        logger.error(f"Non-retryable error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
