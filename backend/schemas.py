"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime


# ============================================================================
# Authentication Schemas
# ============================================================================

class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    user_id: Optional[int] = None
    email: Optional[str] = None


class UserCreate(BaseModel):
    """Schema for user registration."""
    email: str
    password: str
    display_name: Optional[str] = None

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password meets requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if len(v) > 128:
            raise ValueError('Password must be less than 128 characters')
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate and sanitize email."""
        # Basic trimming
        v = v.strip()
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v


class UserLogin(BaseModel):
    """Schema for user login."""
    email: str
    password: str


class UserResponse(BaseModel):
    """Schema for user response."""
    id: int
    email: str
    display_name: Optional[str] = None
    daily_budget_limit: float
    created_at: datetime
    is_active: bool


class UserChangePassword(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str


class UserUpdate(BaseModel):
    """Schema for updating user profile."""
    display_name: Optional[str] = None
    daily_budget_limit: Optional[float] = None

    @field_validator('daily_budget_limit')
    @classmethod
    def validate_budget(cls, v: Optional[float]) -> Optional[float]:
        """Validate budget limit."""
        if v is not None and (v < 0 or v > 10000):
            raise ValueError('Budget limit must be between 0 and 10000')
        return v


# ============================================================================
# Configuration Schemas
# ============================================================================

class CouncilConfig(BaseModel):
    """Council model configuration."""
    models: List[str] = Field(
        default_factory=list,
        description="List of model IDs for the council"
    )
    chairman_model: str = Field(
        default="google/gemini-pro",
        description="Model ID for the chairman"
    )


class GenerationParams(BaseModel):
    """LLM generation parameters."""
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 to 2.0)"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens to generate"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to guide model behavior"
    )


class ContextConfig(BaseModel):
    """Context/memory management configuration."""
    max_history_messages: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Maximum number of previous messages to include as context"
    )


class UserConfigCreate(BaseModel):
    """Schema for creating user configuration."""
    council_models: Optional[List[str]] = None
    chairman_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    max_history_messages: Optional[int] = None


class UserConfigUpdate(BaseModel):
    """Schema for updating user configuration."""
    council_models: Optional[List[str]] = None
    chairman_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    max_history_messages: Optional[int] = None


class UserConfigResponse(BaseModel):
    """Schema for user configuration response."""
    council_models: List[str]
    chairman_model: str
    temperature: float
    max_tokens: int
    system_prompt: Optional[str]
    max_history_messages: int
    updated_at: datetime


class FullUserConfig(BaseModel):
    """Complete user configuration including all sections."""
    council: CouncilConfig
    generation: GenerationParams
    context: ContextConfig


# ============================================================================
# Conversation Schemas
# ============================================================================

class ConversationCreate(BaseModel):
    """Schema for creating a conversation."""
    title: Optional[str] = Field(
        default="New Conversation",
        max_length=500
    )


class ConversationResponse(BaseModel):
    """Schema for conversation response."""
    id: str
    user_id: int
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class MessageCreate(BaseModel):
    """Schema for creating a user message."""
    content: str = Field(..., min_length=1, max_length=50000)


class MessageResponse(BaseModel):
    """Schema for message response."""
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    stage1: Optional[Dict[str, Any]] = None
    stage2: Optional[Dict[str, Any]] = None
    stage3: Optional[Dict[str, Any]] = None
    extra_metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Cost Tracking Schemas
# ============================================================================

class CostResponse(BaseModel):
    """Schema for cost entry response."""
    id: int
    user_id: int
    conversation_id: Optional[str]
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    created_at: datetime


class CostStatistics(BaseModel):
    """Schema for cost statistics."""
    period_days: int
    total_cost: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    request_count: int
    daily_breakdown: List[Dict[str, Any]]
    by_provider: List[Dict[str, Any]]
    by_model: List[Dict[str, Any]]


class BudgetCheck(BaseModel):
    """Schema for budget check response."""
    within_budget: bool
    current_spending: float
    budget_limit: float
    remaining_budget: float


# ============================================================================
# Error Schemas
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail for validation errors."""
    field: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    details: Optional[List[ErrorDetail]] = None
