"""Configuration API endpoints for user settings and preferences."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from backend.database import get_db
from backend.models import User, UserConfig
from backend.schemas import (
    UserConfigResponse,
    UserConfigUpdate,
    FullUserConfig,
    CouncilConfig,
    GenerationParams,
    ContextConfig,
    ErrorResponse,
)
from backend.auth.routes import get_current_user

router = APIRouter(prefix="/config", tags=["configuration"])

# Default configuration values
DEFAULT_COUNCIL_MODELS = [
    # HuggingFace models
    "meta-llama/Llama-3-70b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-2-9b-it",
]
DEFAULT_CHAIRMAN_MODEL = "meta-llama/Llama-3-70b-chat-hf"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_HISTORY_MESSAGES = 10


async def get_or_create_user_config(
    user_id: int,
    db: AsyncSession
) -> UserConfig:
    """Get existing user config or create default."""
    result = await db.execute(
        select(UserConfig).where(UserConfig.user_id == user_id)
    )
    config = result.scalar_one_or_none()

    if config is None:
        # Create default configuration
        config = UserConfig(
            user_id=user_id,
            council_models=DEFAULT_COUNCIL_MODELS.copy(),
            chairman_model=DEFAULT_CHAIRMAN_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
            system_prompt=None,
            max_history_messages=DEFAULT_MAX_HISTORY_MESSAGES,
        )
        db.add(config)
        await db.commit()
        await db.refresh(config)

    return config


@router.get("/", response_model=FullUserConfig)
async def get_user_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the current user's configuration."""
    config = await get_or_create_user_config(current_user.id, db)

    return FullUserConfig(
        council=CouncilConfig(
            models=config.council_models or [],
            chairman_model=config.chairman_model,
        ),
        generation=GenerationParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
        ),
        context=ContextConfig(
            max_history_messages=config.max_history_messages,
        ),
    )


@router.put("/", response_model=FullUserConfig)
async def update_user_config(
    config_update: UserConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the user's configuration."""
    config = await get_or_create_user_config(current_user.id, db)

    # Update fields that are provided
    if config_update.council_models is not None:
        if not config_update.council_models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Council models cannot be empty",
            )
        config.council_models = config_update.council_models

    if config_update.chairman_model is not None:
        if not config_update.chairman_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chairman model cannot be empty",
            )
        config.chairman_model = config_update.chairman_model

    if config_update.temperature is not None:
        config.temperature = config_update.temperature

    if config_update.max_tokens is not None:
        config.max_tokens = config_update.max_tokens

    if config_update.system_prompt is not None:
        config.system_prompt = config_update.system_prompt

    if config_update.max_history_messages is not None:
        config.max_history_messages = config_update.max_history_messages

    await db.commit()
    await db.refresh(config)

    return FullUserConfig(
        council=CouncilConfig(
            models=config.council_models or [],
            chairman_model=config.chairman_model,
        ),
        generation=GenerationParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
        ),
        context=ContextConfig(
            max_history_messages=config.max_history_messages,
        ),
    )


@router.post("/reset", response_model=FullUserConfig)
async def reset_user_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Reset the user's configuration to defaults."""
    result = await db.execute(
        select(UserConfig).where(UserConfig.user_id == current_user.id)
    )
    config = result.scalar_one_or_none()

    if config:
        # Reset to defaults
        config.council_models = DEFAULT_COUNCIL_MODELS.copy()
        config.chairman_model = DEFAULT_CHAIRMAN_MODEL
        config.temperature = DEFAULT_TEMPERATURE
        config.max_tokens = DEFAULT_MAX_TOKENS
        config.system_prompt = None
        config.max_history_messages = DEFAULT_MAX_HISTORY_MESSAGES

        await db.commit()
        await db.refresh(config)
    else:
        # Create new default config
        config = await get_or_create_user_config(current_user.id, db)

    return FullUserConfig(
        council=CouncilConfig(
            models=config.council_models or [],
            chairman_model=config.chairman_model,
        ),
        generation=GenerationParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            system_prompt=config.system_prompt,
        ),
        context=ContextConfig(
            max_history_messages=config.max_history_messages,
        ),
    )


@router.get("/council", response_model=CouncilConfig)
async def get_council_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the council model configuration."""
    config = await get_or_create_user_config(current_user.id, db)

    return CouncilConfig(
        models=config.council_models or [],
        chairman_model=config.chairman_model,
    )


@router.put("/council", response_model=CouncilConfig)
async def update_council_config(
    council_config: CouncilConfig,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the council model configuration."""
    if not council_config.models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Council models cannot be empty",
        )

    config = await get_or_create_user_config(current_user.id, db)
    config.council_models = council_config.models
    config.chairman_model = council_config.chairman_model

    await db.commit()
    await db.refresh(config)

    return CouncilConfig(
        models=config.council_models or [],
        chairman_model=config.chairman_model,
    )


@router.get("/generation", response_model=GenerationParams)
async def get_generation_params(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the generation parameters."""
    config = await get_or_create_user_config(current_user.id, db)

    return GenerationParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        system_prompt=config.system_prompt,
    )


@router.put("/generation", response_model=GenerationParams)
async def update_generation_params(
    params: GenerationParams,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the generation parameters."""
    config = await get_or_create_user_config(current_user.id, db)
    config.temperature = params.temperature
    config.max_tokens = params.max_tokens
    config.system_prompt = params.system_prompt

    await db.commit()
    await db.refresh(config)

    return GenerationParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        system_prompt=config.system_prompt,
    )


@router.get("/context", response_model=ContextConfig)
async def get_context_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the context/memory configuration."""
    config = await get_or_create_user_config(current_user.id, db)

    return ContextConfig(
        max_history_messages=config.max_history_messages,
    )


@router.put("/context", response_model=ContextConfig)
async def update_context_config(
    context_config: ContextConfig,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the context/memory configuration."""
    config = await get_or_create_user_config(current_user.id, db)
    config.max_history_messages = context_config.max_history_messages

    await db.commit()
    await db.refresh(config)

    return ContextConfig(
        max_history_messages=config.max_history_messages,
    )
