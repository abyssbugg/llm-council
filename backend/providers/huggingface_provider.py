"""Hugging Face provider implementation using the abstraction layer."""

import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from ..config import HUGGINGFACE_API_KEY
from .base import (
    BaseLLMProvider,
    ProviderResponse,
    ProviderConfig,
    ProviderError,
    AuthenticationError,
)

logger = logging.getLogger(__name__)

# Type hint for AsyncInferenceClient (only used during type checking)
if TYPE_CHECKING:
    from huggingface_hub import AsyncInferenceClient

# Try to import huggingface_hub
try:
    from huggingface_hub import AsyncInferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("huggingface-hub not installed. HuggingFace provider will be unavailable.")


class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face Inference API provider implementation.
    
    This provider uses the Hugging Face Inference API to generate responses
    from hosted models. It serves as a secondary/fallback provider.
    """
    
    # Common free inference models
    AVAILABLE_MODELS = [
        # Meta
        "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Llama-3-8b-chat-hf",
        # Mistral
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        # Google
        "google/gemma-7b-it",
        "google/gemma-2-9b-it",
        # Qwen
        "Qwen/Qwen2-72B-Instruct",
        # Others
        "tiiuae/falcon-180B-chat",
        "cognitivecomputations/dolphin-2.6-mixtral-8x7b",
    ]
    
    def __init__(self, config: ProviderConfig):
        """Initialize HuggingFace provider.
        
        Args:
            config: Provider configuration. API key from HUGGINGFACE_API_KEY.
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "huggingface-hub is required for HuggingFaceProvider. "
                "Install with: pip install huggingface-hub"
            )
        
        super().__init__(config)
        self._clients: Dict[str, Any] = {}  # Per-model client cache
    
    @property
    def provider_name(self) -> str:
        return "huggingface"
    
    @property
    def available_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
    
    def _get_client(self, model: str) -> Any:
        """Get or create inference client for a model.
        
        Args:
            model: Model identifier.
        
        Returns:
            AsyncInferenceClient instance for the specified model.
        """
        if model not in self._clients:
            token = self.config.api_key or HUGGINGFACE_API_KEY
            if not token:
                raise AuthenticationError(
                    "HuggingFace API key required. Set HUGGINGFACE_API_KEY.",
                    self.provider_name,
                )
            self._clients[model] = AsyncInferenceClient(model=model, token=token)
        return self._clients[model]
    
    async def generate_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> ProviderResponse:
        """Generate a response using HuggingFace Inference API.
        
        Args:
            prompt: The user's prompt.
            model: Model identifier (e.g., "meta-llama/Llama-3-8b-chat-hf").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system prompt (added to prompt).
            conversation_history: Optional conversation history.
        
        Returns:
            ProviderResponse with content and usage info.
        
        Raises:
            ProviderError: If the request fails.
        """
        try:
            # Build the full prompt with conversation context
            full_prompt = self._build_prompt(prompt, system_prompt, conversation_history)
            
            # Get client
            client = self._get_client(model)
            
            # Generate response
            parameters = {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
            }
            
            # HF text generation API
            output = await client.text_generation(
                full_prompt,
                **parameters,
            )
            
            content = output if isinstance(output, str) else str(output)
            
            # Estimate token usage (HF doesn't provide this)
            # Rough estimation: 1 token ≈ 4 characters
            prompt_tokens = len(full_prompt) // 4
            completion_tokens = len(content) // 4
            total_tokens = prompt_tokens + completion_tokens
            
            cost_usd = await self.estimate_cost(prompt, model, max_tokens)
            
            return ProviderResponse(
                content=content,
                model=model,
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            )
        
        except Exception as e:
            error_msg = str(e)
            
            # Check for authentication errors
            if "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise AuthenticationError(
                    "Invalid HuggingFace API key",
                    self.provider_name,
                )
            
            # Check for model not found
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
                raise ProviderError(
                    f"Model {model} not found on HuggingFace",
                    self.provider_name,
                    recoverable=False,
                )
            
            raise ProviderError(
                f"HuggingFace API error: {error_msg}",
                self.provider_name,
                recoverable=True,
            )
    
    def _build_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str],
        conversation_history: Optional[List[Dict[str, str]]],
    ) -> str:
        """Build the full prompt from components.
        
        Args:
            prompt: User's current prompt.
            system_prompt: Optional system prompt.
            conversation_history: Optional conversation history.
        
        Returns:
            Full formatted prompt string.
        """
        parts = []
        
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role.capitalize()}: {content}")
        
        parts.append(f"User: {prompt}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    async def estimate_cost(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
    ) -> float:
        """Estimate the cost of a request.
        
        HuggingFace serverless inference is free for most models,
        but there are rate limits. Prodedicated inference has costs.
        
        Args:
            prompt: User's prompt.
            model: Model identifier.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            Estimated cost in USD (typically 0 for free tier).
        """
        # HuggingFace serverless inference is typically free
        # with rate limits. Dedicated endpoints have costs.
        return 0.0
    
    def supports_model(self, model: str) -> bool:
        """Check if this provider supports the given model.
        
        Args:
            model: Model identifier to check.
        
        Returns:
            True if the model appears to be a HuggingFace model.
        """
        # HuggingFace models use "org/model" format
        if "/" in model:
            return True
        
        # Check against known models
        return model in [m.split("/")[-1] for m in self.AVAILABLE_MODELS]
    
    async def close(self) -> None:
        """Close the client if needed."""
        if self._client:
            self._client = None
