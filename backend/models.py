"""SQLAlchemy ORM models for LLM Council."""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, DateTime, ForeignKey, Text, JSON, Integer, Float, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class User(Base):
    """User model for authentication and configuration."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement="auto")
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # User preferences
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    daily_budget_limit: Mapped[float] = mapped_column(Float, default=10.0)  # USD

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )
    costs: Mapped[List["Cost"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan"
    )


class Conversation(Base):
    """Conversation model representing a chat session."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # UUID as string
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    title: Mapped[str] = mapped_column(String(500), default="New Conversation")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="conversations")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )


class Message(Base):
    """Message model for individual user/assistant messages."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement="auto")
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user' or 'assistant'
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # For assistant messages: store the 3-stage council results
    stage1: Mapped[Optional[dict]] = mapped_column(JSON)  # Individual model responses
    stage2: Mapped[Optional[dict]] = mapped_column(JSON)  # Peer rankings
    stage3: Mapped[Optional[dict]] = mapped_column(JSON)  # Final synthesis

    # Metadata (label_to_model, aggregate_rankings, etc.)
    extra_metadata: Mapped[Optional[dict]] = mapped_column(JSON)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    conversation: Mapped["Conversation"] = relationship(back_populates="messages")


class Cost(Base):
    """Cost tracking model for API usage."""

    __tablename__ = "costs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement="auto")
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL")
    )

    # Provider and model used
    provider: Mapped[str] = mapped_column(String(100), nullable=False)  # 'openrouter', 'chutes', 'huggingface'
    model: Mapped[str] = mapped_column(String(200), nullable=False)

    # Token usage and cost
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True
    )

    # Relationship
    user: Mapped["User"] = relationship(back_populates="costs")


class UserConfig(Base):
    """User configuration for council models and preferences."""

    __tablename__ = "user_config"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement="auto")
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    # Council model selection
    council_models: Mapped[list] = mapped_column(JSON, default=list)  # List of model IDs
    chairman_model: Mapped[str] = mapped_column(String(200), default="google/gemini-pro")

    # Generation parameters
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int] = mapped_column(Integer, default=4096)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text)

    # Context management
    max_history_messages: Mapped[int] = mapped_column(Integer, default=10)

    # Timestamp
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
