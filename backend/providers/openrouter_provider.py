"""OpenRouter provider implementation using the abstraction layer."""

import httpx
import logging
from typing import List, Optional, Dict, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from ..config import OPENROUTER_API_KEY, OPENROUTER_API_URL
from .base import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderConfig,
    ProviderError,
    RateLimitError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)


def _should_retry_http_error(exception: Exception) -> bool:
    """Check if an HTTP error should trigger a retry."""
    if isinstance(exception, httpx.HTTPStatusError):
        status_code = exception.response.status_code
        return status_code in {429, 500, 502, 503, 504}
    return False


from tenacity import retry_if_exception

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(lambda e: isinstance(e, httpx.HTTPStatusError) and _should_retry_http_error(e)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def _make_request(client: httpx.AsyncClient, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make HTTP request with retry logic.
    
    Args:
        client: HTTP client to use.
        url: Request URL.
        headers: Request headers.
        payload: Request body.
    
    Returns:
        Parsed JSON response.
    
    Raises:
        httpx.HTTPStatusError: If request fails after retries.
    """
    response = await client.post(url, headers=headers, json=payload, timeout=60.0)
    response.raise_for_status()
    return response.json()


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter API provider implementation.
    
    OpenRouter aggregates multiple LLM providers and provides a unified API.
    This provider implements the BaseLLMProvider interface for OpenRouter.
    """
    
    # Common models available on OpenRouter
    AVAILABLE_MODELS = [
        # OpenAI
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "openai/gpt-4-turbo",
        "openai/gpt-3.5-turbo",
        # Anthropic
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "anthropic/claude-2.1",
        # Google
        "google/gemini-pro",
        "google/gemini-pro-1.5",
        # Meta
        "meta-llama/llama-3-70b-instruct",
        "meta-llama/llama-3-8b-instruct",
        # Mistral
        "mistralai/mistral-large",
        "mistralai/mistral-medium",
        "mistralai/mistral-small",
        # X.AI
        "x-ai/grok-2",
    ]
    
    def __init__(self, config: ProviderConfig):
        """Initialize OpenRouter provider.
        
        Args:
            config: Provider configuration. API key should be from OPENROUTER_API_KEY.
        """
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def provider_name(self) -> str:
        return "openrouter"
    
    @property
    def available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            base_url = self.config.base_url or OPENROUTER_API_URL
            self._client = httpx.AsyncClient(base_url=base_url)
        return self._client
    
    async def generate_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ProviderResponse:
        """Generate a response using OpenRouter API.
        
        Args:
            prompt: The user's prompt.
            model: Model identifier (e.g., "anthropic/claude-3-sonnet").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system prompt.
            conversation_history: Optional conversation history.
        
        Returns:
            ProviderResponse with content and usage info.
        
        Raises:
            ProviderError: If the request fails.
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://llm-council.app",
            "X-Title": "LLM Council",
            "Content-Type": "application/json",
        }
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            client = self._get_client()
            data = await _make_request(
                client,
                "/chat/completions",
                headers,
                payload,
            )
            
            # Extract response
            choice = data["choices"][0]
            content = choice["message"]["content"]
            usage = data.get("usage", {})
            
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
            
            # Calculate cost (OpenRouter provides this in response)
            cost_usd = 0.0
            if "x-model-cost" in data.get("headers", {}):
                # Cost provided by OpenRouter
                cost_usd = float(data["headers"]["x-model-cost"])
            else:
                # Fallback: estimate from typical pricing
                cost_usd = self._estimate_cost_from_tokens(model, prompt_tokens, completion_tokens)
            
            # Extract reasoning details if present (for models that support it)
            reasoning_details = None
            if "reasoning" in choice.get("message", {}):
                reasoning_details = choice["message"]["reasoning"]
            
            return ProviderResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                reasoning_details=reasoning_details,
                raw_response=data,
            )
        
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            
            if status_code == 401:
                raise AuthenticationError(
                    "Invalid OpenRouter API key",
                    self.provider_name,
                )
            elif status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    raise RateLimitError(
                        f"Rate limited. Retry after {retry_after}s",
                        self.provider_name,
                        retry_after=int(retry_after),
                    )
                raise RateLimitError("Rate limited", self.provider_name)
            
            raise ProviderError(
                f"HTTP {status_code}: {e.response.text}",
                self.provider_name,
                recoverable=status_code >= 500,
            )
        
        except httpx.RequestError as e:
            raise ProviderError(
                f"Network error: {e}",
                self.provider_name,
                recoverable=True,
            )
    
    async def estimate_cost(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
    ) -> float:
        """Estimate the cost of a request.
        
        Args:
            prompt: User's prompt.
            model: Model identifier.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            Estimated cost in USD.
        """
        # Rough estimation: count characters as proxy for tokens
        # Average: 1 token ≈ 4 characters
        estimated_prompt_tokens = len(prompt) // 4
        estimated_completion_tokens = min(max_tokens, estimated_prompt_tokens)
        
        return self._estimate_cost_from_tokens(
            model,
            estimated_prompt_tokens,
            estimated_completion_tokens,
        )
    
    def _estimate_cost_from_tokens(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate cost from token counts.
        
        Uses typical pricing for major models. In production,
        this should query OpenRouter's pricing API.
        
        Args:
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
        
        Returns:
            Estimated cost in USD.
        """
        # Typical pricing per 1M tokens (input/output)
        # These are conservative estimates
        pricing = {
            # GPT-4
            "gpt-4": (30.0, 60.0),
            "gpt-4-turbo": (10.0, 30.0),
            # Claude
            "claude-3-opus": (15.0, 75.0),
            "claude-3-sonnet": (3.0, 15.0),
            "claude-3-haiku": (0.25, 1.25),
            # Gemini
            "gemini-pro": (0.5, 1.5),
            # Llama
            "llama-3": (0.1, 0.1),
        }
        
        # Find matching pricing
        input_price = 1.0  # Default $1/M tokens
        output_price = 2.0  # Default $2/M tokens
        
        for key, (inp, out) in pricing.items():
            if key in model.lower():
                input_price = inp
                output_price = out
                break
        
        # Calculate cost
        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = (completion_tokens / 1_000_000) * output_price
        
        return input_cost + output_cost
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: Model identifier to check.
        
        Returns:
            True if the model appears to be an OpenRouter model.
        """
        # OpenRouter uses "provider/model" format
        if "/" in model:
            return True
        
        # Also check against known models
        return model in [m.split("/")[-1] for m in self.AVAILABLE_MODELS]
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
