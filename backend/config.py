"""Configuration for the LLM Council."""

import os
from functools import lru_cache
from typing import List
from dotenv import load_dotenv

load_dotenv()


@lru_cache
def get_settings() -> "Settings":
    """Get cached settings instance."""
    return Settings()


class Settings:
    """Application settings."""

    def __init__(self):
        # Database connection
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_council"
        )

        # JWT settings
        self.secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm: str = "HS256"
        self.access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

        # Redis cache
        self.redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
        self.cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"

        # API keys
        self.openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
        self.chutes_ai_api_key: str | None = os.getenv("CHUTES_AI_API_KEY")
        self.huggingface_api_key: str | None = os.getenv("HUGGINGFACE_API_KEY")

        # Council configuration - HuggingFace and Chutes AI models
        self.council_models: List[str] = [
            # HuggingFace models
            "meta-llama/Llama-3-70b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2-72B-Instruct",
            # Chutes AI models (if API key is set)
            "gpt-4o",
            "claude-3-sonnet",
        ]
        self.chairman_model: str = "meta-llama/Llama-3-70b-chat-hf"

        # API endpoints
        self.openrouter_api_url: str = "https://openrouter.ai/api/v1/chat/completions"

        # Data directory for conversation storage (legacy, will be replaced by DB)
        self.data_dir: str = "data/conversations"


# Create global settings instance
settings = get_settings()

# Legacy exports for backward compatibility
DATABASE_URL = settings.database_url
SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
REDIS_URL = settings.redis_url
CACHE_TTL = settings.cache_ttl
CACHE_ENABLED = settings.cache_enabled
OPENROUTER_API_KEY = settings.openrouter_api_key
CHUTES_AI_API_KEY = settings.chutes_ai_api_key
HUGGINGFACE_API_KEY = settings.huggingface_api_key
COUNCIL_MODELS = settings.council_models
CHAIRMAN_MODEL = settings.chairman_model
OPENROUTER_API_URL = settings.openrouter_api_url
DATA_DIR = settings.data_dir
