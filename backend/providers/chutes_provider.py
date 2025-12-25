"""Chutes AI provider implementation using the abstraction layer."""

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

from ..config import CHUTES_AI_API_KEY
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


class ChutesProvider(BaseLLMProvider):
    """Chutes AI provider implementation.
    
    Chutes AI provides LLM services with competitive pricing.
    This provider implements the BaseLLMProvider interface for Chutes AI.
    """
    
    # Chutes AI model mappings (Chutes model ID -> OpenRouter compatible ID)
    MODEL_MAPPING = {
        # GPT models
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
        # Claude models
        "claude-3-opus": "anthropic/claude-3-opus",
        "claude-3-sonnet": "anthropic/claude-3-sonnet",
        "claude-3-haiku": "anthropic/claude-3-haiku",
        # Llama models
        "llama-3-70b": "meta-llama/llama-3-70b-instruct",
        "llama-3-8b": "meta-llama/llama-3-8b-instruct",
    }
    
    AVAILABLE_MODELS = list(MODEL_MAPPING.keys())
    
    # Chutes AI API endpoint
    DEFAULT_BASE_URL = "https://api.chutes.ai/v1"
    
    def __init__(self, config: ProviderConfig):
        """Initialize Chutes AI provider.
        
        Args:
            config: Provider configuration. API key from CHUTES_AI_API_KEY.
        """
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = config.base_url or self.DEFAULT_BASE_URL
    
    @property
    def provider_name(self) -> str:
        return "chutes"
    
    @property
    def available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url)
        return self._client
    
    def _normalize_model_id(self, model: str) -> str:
        """Convert model ID to Chutes AI format.
        
        Args:
            model: Model identifier (could be Chutes or OpenRouter format).
        
        Returns:
            Chutes AI model ID.
        """
        # If already a Chutes model ID, return as-is
        if model in self.AVAILABLE_MODELS:
            return model
        
        # Try to map from OpenRouter format
        for chutes_id, openrouter_id in self.MODEL_MAPPING.items():
            if model == openrouter_id or model.endswith(openrouter_id):
                return chutes_id
        
        # Default: try removing provider prefix
        if "/" in model:
            return model.split("/")[-1]
        
        return model
    
    async def generate_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ProviderResponse:
        """Generate a response using Chutes AI API.
        
        Args:
            prompt: The user's prompt.
            model: Model identifier.
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
            "Content-Type": "application/json",
        }
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        # Normalize model ID for Chutes API
        chutes_model = self._normalize_model_id(model)
        
        payload = {
            "model": chutes_model,
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
            
            # Chutes AI typically provides cost in response
            cost_usd = data.get("cost", 0.0)
            if not cost_usd:
                # Fallback estimation
                cost_usd = self._estimate_cost_from_tokens(
                    chutes_model,
                    prompt_tokens,
                    completion_tokens,
                )
            
            return ProviderResponse(
                content=content,
                model=model,  # Return original model ID
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
                raw_response=data,
            )
        
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            
            if status_code == 401:
                raise AuthenticationError(
                    "Invalid Chutes AI API key",
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
        estimated_prompt_tokens = len(prompt) // 4
        estimated_completion_tokens = min(max_tokens, estimated_prompt_tokens)
        
        chutes_model = self._normalize_model_id(model)
        return self._estimate_cost_from_tokens(
            chutes_model,
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
        
        Chutes AI typically offers competitive pricing.
        These are conservative estimates.
        
        Args:
            model: Model identifier.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
        
        Returns:
            Estimated cost in USD.
        """
        # Chutes AI pricing (estimated, per 1M tokens)
        pricing = {
            # GPT models
            "gpt-4": (5.0, 15.0),
            "gpt-4-turbo": (1.0, 3.0),
            "gpt-3.5": (0.1, 0.3),
            # Claude models
            "claude-opus": (3.0, 15.0),
            "claude-sonnet": (1.0, 5.0),
            "claude-haiku": (0.1, 0.5),
            # Llama models
            "llama": (0.05, 0.05),
        }
        
        input_price = 0.5  # Default $0.5/M
        output_price = 1.5  # Default $1.5/M
        
        model_lower = model.lower()
        for key, (inp, out) in pricing.items():
            if key in model_lower:
                input_price = inp
                output_price = out
                break
        
        input_cost = (prompt_tokens / 1_000_000) * input_price
        output_cost = (completion_tokens / 1_000_000) * output_price
        
        return input_cost + output_cost
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: Model identifier to check.
        
        Returns:
            True if the model is supported.
        """
        # Check direct match
        if model in self.AVAILABLE_MODELS:
            return True
        
        # Check if model is in mapping (OpenRouter format)
        if model in self.MODEL_MAPPING.values():
            return True
        
        # Check partial matches
        model_lower = model.lower()
        for supported in self.AVAILABLE_MODELS:
            if supported in model_lower or model_lower in supported:
                return True
        
        return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
